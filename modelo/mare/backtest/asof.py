"""Visão de mercado limitada a uma data — look-ahead impossível por construção.

A maioria dos backtests evita look-ahead por disciplina: quem escreve o sinal
lembra de usar `.shift(1)`. Disciplina falha, e falha em silêncio, produzindo
uma curva de patrimônio bonita e mentirosa.

Aqui a defesa é estrutural. O engine nunca entrega os painéis completos à função
de sinal; entrega um `AsOfView` cujos DataFrames já foram FISICAMENTE fatiados em
`<= as_of`. Não existe dado futuro para acessar por engano. E qualquer consulta
explícita a uma data posterior levanta `LookAheadError` em vez de devolver NaN,
porque um NaN silencioso vira zero em algum lugar e some.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mare.io.universe import Membership


class LookAheadError(RuntimeError):
    """Tentativa de acessar dado posterior à data do sinal."""


@dataclass(frozen=True)
class AsOfView:
    """Tudo que a estratégia pode saber no fechamento de `as_of`."""

    as_of: pd.Timestamp
    returns: pd.DataFrame
    close_raw: pd.DataFrame
    volume: pd.DataFrame
    members: frozenset[str]

    def tail(self, frame: pd.DataFrame, window: int) -> pd.DataFrame:
        if window > len(frame):
            raise ValueError(f"janela {window} maior que o histórico {len(frame)}")
        return frame.iloc[-window:]

    def at(self, date) -> pd.Series:
        """Retornos de uma data específica, com guarda explícita."""
        stamp = pd.Timestamp(date)
        if stamp > self.as_of:
            raise LookAheadError(
                f"sinal em {self.as_of:%Y-%m-%d} tentou ler {stamp:%Y-%m-%d}")
        return self.returns.loc[stamp]

    def dollar_volume(self, window: int) -> pd.Series:
        """Mediana do dólar-volume na janela — o filtro de liquidez.

        Usa Close BRUTO de propósito: o ajustado reescreve o passado a cada
        provento e distorceria a elegibilidade histórica.
        """
        prices = self.tail(self.close_raw, window)
        volumes = self.tail(self.volume, window)
        return (prices * volumes).median()

    def eligible(self, *, min_dollar_volume: float, min_price: float,
                 min_valid_returns: int, adv_window: int,
                 history: int) -> list[str]:
        """Universo negociável: membro do índice E líquido E com histórico.

        Os três filtros usam apenas dado até `as_of`, e a ordem importa pouco —
        o que importa é que `members` já é point-in-time, então nunca entra aqui
        uma empresa que só viraria membro no futuro.
        """
        candidates = [c for c in self.returns.columns if c in self.members]
        if not candidates:
            return []

        window = self.tail(self.returns, history)[candidates]
        enough = window.notna().sum() >= min_valid_returns

        adv = self.dollar_volume(adv_window).reindex(candidates)
        liquid = adv >= min_dollar_volume

        last_price = self.tail(self.close_raw, 1)[candidates].iloc[-1]
        priced = last_price >= min_price

        keep = enough & liquid.fillna(False) & priced.fillna(False)
        return sorted(keep.index[keep.to_numpy()])


@dataclass(frozen=True)
class MarketData:
    """Painéis completos. Só o engine toca nisto; a estratégia recebe views."""

    returns: pd.DataFrame
    close_raw: pd.DataFrame
    volume: pd.DataFrame
    membership: Membership

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.returns.index)

    def view(self, as_of) -> AsOfView:
        """Fatia os painéis em `<= as_of`. É esta linha que garante o invariante."""
        stamp = pd.Timestamp(as_of)
        cut = self.returns.index <= stamp
        if not cut.any():
            raise ValueError(f"sem histórico até {stamp:%Y-%m-%d}")
        return AsOfView(
            as_of=stamp,
            returns=self.returns.loc[cut],
            close_raw=self.close_raw.loc[self.close_raw.index <= stamp],
            volume=self.volume.loc[self.volume.index <= stamp],
            members=frozenset(self.membership.on(stamp)),
        )

    def forward_returns(self, start, end) -> pd.DataFrame:
        """Retornos realizados entre dois rebalanceamentos.

        Só o engine chama isto, para apurar o resultado de posições JÁ definidas.
        Nunca é exposto à função de sinal.
        """
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        mask = (self.returns.index > lo) & (self.returns.index <= hi)
        return self.returns.loc[mask]
