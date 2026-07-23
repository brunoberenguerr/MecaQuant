"""Montagem do `MarketData` a partir dos artefatos gerados por 01_build_universe.

Separa "buscar dado" de "usar dado": tudo aqui lê de `data/interim`, então o
backtest roda offline e de forma reprodutível, sem depender do que o Yahoo
resolve devolver hoje.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mare.backtest.asof import MarketData
from mare.config import Config
from mare.features.returns import simple_returns
from mare.io import macro, prices, universe, wikipedia


@dataclass(frozen=True)
class Bundle:
    """Tudo que o backtest e o relatório precisam."""

    market: MarketData
    sectors: pd.Series
    benchmark: pd.Series      # retornos do SPY
    risk_free: pd.Series      # retorno diário do T-bill
    vix: pd.Series


def _membership_from_parquet(path: Path) -> universe.Membership:
    frame = pd.read_parquet(path)
    boundaries = tuple(
        (pd.Timestamp(row.valid_from), frozenset(row.symbols.split("|") if row.symbols else []))
        for row in frame.itertuples())
    return universe.Membership(boundaries)


def load_bundle(cfg: Config, interim: Path | None = None) -> Bundle:
    """Carrega painéis, membership, benchmark, taxa livre de risco e VIX."""
    interim = Path(interim or cfg.root() / "data" / "interim")
    panel = pd.read_parquet(interim / "prices_panel.parquet")

    close_adj = prices.to_wide(panel, "close_adj")
    close_raw = prices.to_wide(panel, "close_raw")
    volume = prices.to_wide(panel, "volume")
    returns = simple_returns(close_adj)

    calendar = returns.index
    bench_px = macro.fetch_series(cfg.data.benchmark, cfg.data.start,
                                  cfg.data.end, cfg.data.cache_dir)
    vix_px = macro.fetch_series(cfg.data.vix, cfg.data.start, cfg.data.end,
                                cfg.data.cache_dir)
    rf_frame = macro.fetch_fred(cfg.data.riskfree_series, cfg.data.cache_dir)

    market = MarketData(
        returns=returns, close_raw=close_raw, volume=volume,
        membership=_membership_from_parquet(interim / "membership.parquet"))

    return Bundle(
        market=market,
        sectors=_sectors(cfg),
        benchmark=bench_px.pct_change().reindex(calendar),
        risk_free=macro.daily_risk_free(rf_frame, calendar),
        vix=vix_px.reindex(calendar).ffill(),
    )


def _sectors(cfg: Config) -> pd.Series:
    """Setor GICS por símbolo.

    Limitação declarada: a Wikipedia só publica o setor VIGENTE, então empresas
    reclassificadas ao longo do tempo carregam o setor de hoje para trás. O
    efeito no teto setorial é de segunda ordem — reclassificações são raras e o
    teto é grosso — mas precisa constar no relatório.
    """
    table = wikipedia.fetch_constituents(cfg.data.cache_dir)
    table = table.assign(symbol=table["symbol"].map(universe.normalize_ticker))
    return table.set_index("symbol")["sector"]
