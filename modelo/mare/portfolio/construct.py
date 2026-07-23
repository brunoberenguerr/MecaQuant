"""Função de sinal completa: da view de mercado aos pesos alvo.

É o ponto de entrada que o engine chama. Mantido fino de propósito — cada etapa
mora no seu módulo, e aqui só se costura a sequência, para que a estratégia
inteira caiba em uma tela e possa ser explicada à banca sem rolar a página.

    view -> tabela de candidatos -> livro de posições -> pesos por risco
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from mare.backtest.asof import AsOfView
from mare.config import Config
from mare.portfolio.book import PositionBook
from mare.portfolio.size import build_weights
from mare.signal.score import candidate_table, rank_candidates


@dataclass
class SignalDiagnostics:
    """Histórico das decisões, para o dashboard e a análise crítica."""

    candidates: dict = field(default_factory=dict)
    eligible_pool: dict = field(default_factory=dict)   # nomes baratos (o pool escolhido)
    universe_pool: dict = field(default_factory=dict)    # todos os líquidos (o universo)
    trades: list = field(default_factory=list)
    k_factors: dict = field(default_factory=dict)
    n_approved: dict = field(default_factory=dict)

    def trades_frame(self) -> pd.DataFrame:
        return pd.concat(self.trades, ignore_index=True) if self.trades else pd.DataFrame()


def make_signal_fn(sectors: pd.Series | None = None,
                   diagnostics: SignalDiagnostics | None = None,
                   book: PositionBook | None = None,
                   table_fn: Callable[[AsOfView, Config], pd.DataFrame] | None = None):
    """Constrói a função de sinal, com livro de posições persistente.

    `table_fn` substitui a construção padrão da tabela de candidatos (PCA +
    resíduo + OU) por outra implementação com a mesma interface — por exemplo
    `naive.candidate_table_naive`, usada para isolar se a residualização por
    fatores agrega algo sobre reversão simples. O resto do pipeline (livro de
    posições, sizing, caps) é idêntico, para uma comparação justa.
    """
    book = book if book is not None else PositionBook()
    k_history: list[int] = []

    def signal_fn(view: AsOfView, cfg: Config) -> pd.Series:
        table = (table_fn(view, cfg) if table_fn is not None
                else candidate_table(view, cfg, k_history=k_history))
        if table.empty:
            book.reset()
            return pd.Series(dtype=float)

        ranked = rank_candidates(table, cfg.portfolio.n_names,
                                 rank_by=cfg.passage.rank_by)
        update = book.update(table, ranked, cfg, view.as_of)
        _record(diagnostics, view, table, ranked, update)

        symbols = book.symbols()
        if not symbols:
            return pd.Series(dtype=float)

        weights = build_weights(
            view.returns, symbols,
            scheme=cfg.portfolio.weighting, lam=cfg.portfolio.ewma_lambda,
            cap=cfg.portfolio.max_weight_mult / cfg.portfolio.n_names,
            sectors=sectors, sector_cap=cfg.portfolio.max_sector_weight)

        # Menos posições que o alvo significa menos capital investido, não
        # concentração forçada nas que sobraram. O caixa rende a taxa livre de
        # risco no engine.
        invested = min(1.0, len(symbols) / cfg.portfolio.n_names)
        return weights * invested * cfg.portfolio.max_leverage

    return signal_fn


def _record(diag: SignalDiagnostics | None, view: AsOfView, table: pd.DataFrame,
            ranked: pd.DataFrame, update) -> None:
    if diag is None:
        return
    diag.candidates[view.as_of] = ranked
    # Dois pools para dois placebos distintos e complementares:
    #  - universe_pool: TODOS os líquidos -> testa se a reversão residual (o filtro
    #    "estar barato") tem valor contra uma carteira líquida aleatória.
    #  - eligible_pool: só os baratos -> testa se selecionar os MAIS baratos
    #    adiciona algo além de estar no bucket barato.
    diag.universe_pool[view.as_of] = list(table.index)
    diag.eligible_pool[view.as_of] = list(table.loc[table["eligible"]].index)
    diag.n_approved[view.as_of] = int(table["eligible"].sum())
    if not table.empty and "k_factors" in table.columns:
        diag.k_factors[view.as_of] = int(table["k_factors"].iloc[0])
    if not update.closed.empty:
        diag.trades.append(update.closed)
