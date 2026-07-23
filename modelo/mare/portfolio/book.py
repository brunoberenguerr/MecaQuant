"""Livro de posições: entrar, segurar e sair conforme o modelo manda.

Por que este módulo existe. A primeira versão recalculava o top-N a cada
rebalanceamento e substituía a carteira inteira. Isso mediu giro de 1,11 por
semana — 58x ao ano — e contradizia o próprio modelo: a máquina de primeira
passagem estima que a reversão leva 13 a 20 dias, mas a carteira era desmontada
a cada 7. Uma posição que estava revertendo (s indo de −1,8 para −0,9) era
vendida para comprar outra mais barata, o que significa pagar spread para
desfazer justamente o trade que o modelo pediu.

A regra correta vem do modelo, não de otimização:

    ENTRA   quando s ≤ entry_s  e os portões de validade passam
    SEGURA  enquanto entry_s < s < exit_s
    SAI     quando s ≥ exit_s (reverteu, realiza)
            ou    s ≤ stop_s (rompeu, corta)
            ou    a meia-vida sai da banda (o modelo deixou de valer)
            ou    o nome perde elegibilidade

Cada saída registra o motivo — é a justificativa de troca que vai ao dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from mare.config import Config


@dataclass
class Position:
    symbol: str
    entry_date: pd.Timestamp
    entry_s: float


@dataclass
class BookUpdate:
    """Resultado de um rebalanceamento, pronto para virar linha de relatório."""

    held: list[str]
    opened: list[str]
    closed: pd.DataFrame           # symbol, motivo, s_score, dias_no_livro


@dataclass
class PositionBook:
    """Estado das posições entre rebalanceamentos."""

    positions: dict[str, Position] = field(default_factory=dict)

    def update(self, table: pd.DataFrame, ranked: pd.DataFrame,
               cfg: Config, as_of: pd.Timestamp) -> BookUpdate:
        closed = self._close_expired(table, cfg, as_of)
        opened = self._open_new(ranked, cfg)
        return BookUpdate(held=list(self.positions), opened=opened, closed=closed)

    def _close_expired(self, table: pd.DataFrame, cfg: Config,
                       as_of: pd.Timestamp) -> pd.DataFrame:
        rows = []
        for symbol, position in list(self.positions.items()):
            reason, score = self._exit_reason(symbol, table, cfg)
            if reason is None:
                continue
            rows.append({
                "symbol": symbol, "motivo": reason, "s_score": score,
                "dias_no_livro": (as_of - position.entry_date).days,
                "entry_s": position.entry_s, "as_of": as_of,
            })
            del self.positions[symbol]
        return pd.DataFrame(rows)

    def _exit_reason(self, symbol: str, table: pd.DataFrame,
                     cfg: Config) -> tuple[str | None, float]:
        """Motivo de saída, ou None para manter.

        A banda de meia-vida NÃO aparece aqui de propósito. Ela é filtro de
        ENTRADA: no momento de abrir, queremos confiança de que o nome reverte
        em velocidade negociável. Como gatilho de saída ela é destrutiva — com
        ~3 ciclos independentes na janela de 60 dias, a meia-vida estimada
        oscila em torno da fronteira por ruído puro, e usá-la para liquidar
        gerou 52,6% das saídas com mediana de 7 dias, ou seja, um único
        rebalanceamento. Isso é girar a carteira num parâmetro incidental.

        As saídas legítimas são econômicas: o resíduo reverteu (realiza), rompeu
        o stop (corta), ou o nome deixou de ser negociável.
        """
        if symbol not in table.index:
            return "perdeu_elegibilidade", float("nan")
        row = table.loc[symbol]
        score = float(row["s_score"])
        if score >= cfg.ou.exit_s:
            return "reverteu", score
        if score <= cfg.ou.stop_s:
            return "stop", score
        return None, score

    def _open_new(self, ranked: pd.DataFrame, cfg: Config) -> list[str]:
        slots = cfg.portfolio.n_names - len(self.positions)
        if slots <= 0 or ranked.empty:
            return []
        fresh = [s for s in ranked.index if s not in self.positions][:slots]
        for symbol in fresh:
            self.positions[symbol] = Position(
                symbol=symbol,
                entry_date=pd.Timestamp(ranked.loc[symbol, "as_of"]),
                entry_s=float(ranked.loc[symbol, "s_score"]))
        return fresh

    def symbols(self) -> list[str]:
        return sorted(self.positions)

    def reset(self) -> None:
        self.positions.clear()
