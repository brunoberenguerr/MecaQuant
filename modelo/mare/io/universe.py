"""Reconstrução do membership point-in-time do S&P 500.

Usar a composição de hoje para representar o passado é viés de sobrevivência
clássico. Aqui a composição histórica é reconstruída caminhando o change-log
PARA TRÁS a partir da lista vigente: para saber quem era membro antes de uma
mudança, remove-se quem entrou e devolve-se quem saiu.

A reconstrução é determinística e não depende de rede — `build_membership`
recebe DataFrames, então é testável com dados sintéticos.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

# Classificação do motivo da saída. Importa porque define o que fazer com o
# retorno do nome deslistado: aquisição liquida perto do preço de tela; falência
# destrói capital e ignorá-la infla o backtest.
_BANKRUPT = re.compile(r"bankrupt|chapter 11|liquidat|delist", re.I)
_ACQUIRED = re.compile(r"acquir|merge|bought|purchas|taken private|spun off|spin-off", re.I)


def normalize_ticker(ticker: str) -> str:
    """Converte o símbolo da Wikipedia para o formato do Yahoo (BRK.B -> BRK-B)."""
    return str(ticker).strip().upper().replace(".", "-")


def classify_reason(reason: str) -> str:
    """Disposição do nome que saiu: bankrupt | acquired | index_change."""
    text = str(reason)
    if _BANKRUPT.search(text):
        return "bankrupt"
    if _ACQUIRED.search(text):
        return "acquired"
    return "index_change"


@dataclass(frozen=True)
class Membership:
    """Composição por intervalos: `boundaries` é (valid_from, símbolos) ordenado."""

    boundaries: tuple[tuple[pd.Timestamp, frozenset[str]], ...]

    def on(self, date) -> frozenset[str]:
        """Membros vigentes em `date`."""
        stamp = pd.Timestamp(date)
        chosen: frozenset[str] = frozenset()
        for valid_from, members in self.boundaries:
            if valid_from <= stamp:
                chosen = members
            else:
                break
        return chosen

    def all_symbols(self) -> list[str]:
        """Todo símbolo que já foi membro no período — inclusive os que sumiram."""
        out: set[str] = set()
        for _, members in self.boundaries:
            out |= members
        return sorted(out)

    def counts(self, dates) -> pd.Series:
        return pd.Series({pd.Timestamp(d): len(self.on(d)) for d in dates}, name="n_members")


def build_membership(constituents: pd.DataFrame, changes: pd.DataFrame,
                     asof: pd.Timestamp | None = None) -> Membership:
    """Caminha o change-log para trás e devolve a composição por intervalos.

    `constituents` precisa da coluna `symbol`; `changes` de `date`, `added`,
    `removed`. Mudanças com data futura em relação a `asof` são ignoradas.
    """
    asof = pd.Timestamp(asof or pd.Timestamp.today().normalize())
    current = {normalize_ticker(s) for s in constituents["symbol"] if str(s).strip()}

    events = changes.loc[changes["date"] <= asof].sort_values("date", ascending=False)
    boundaries: list[tuple[pd.Timestamp, frozenset[str]]] = []
    members = set(current)

    for date, group in events.groupby("date", sort=False):
        # `members` vale de `date` (inclusive) até a próxima fronteira à direita.
        boundaries.append((pd.Timestamp(date), frozenset(members)))
        for _, row in group.iterrows():
            added, removed = normalize_ticker(row["added"]), normalize_ticker(row["removed"])
            if added:
                members.discard(added)
            if removed:
                members.add(removed)

    # Estado anterior à mudança mais antiga conhecida.
    boundaries.append((pd.Timestamp.min + pd.Timedelta(days=1), frozenset(members)))
    boundaries.sort(key=lambda item: item[0])
    return Membership(tuple(boundaries))


def dispositions(changes: pd.DataFrame) -> pd.DataFrame:
    """Para cada saída do índice: quando saiu e por quê."""
    out = changes.loc[changes["removed"].astype(str).str.strip() != ""].copy()
    out["symbol"] = out["removed"].map(normalize_ticker)
    out["disposition"] = out["reason"].map(classify_reason)
    return out[["date", "symbol", "disposition", "reason"]].reset_index(drop=True)


def audit_membership(membership: Membership, dates) -> pd.DataFrame:
    """Contagem reconstruída por data, para medir a lacuna do change-log.

    O índice tem ~500 membros por construção. Se a contagem reconstruída
    despencar ao voltar no tempo, o change-log da Wikipedia ("Selected changes")
    está incompleto naquele período — e isso precisa ir para o relatório, não
    ser escondido.
    """
    counts = membership.counts(dates)
    frame = counts.to_frame()
    frame["gap_vs_500"] = 500 - frame["n_members"]
    return frame
