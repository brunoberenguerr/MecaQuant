"""Séries macro: taxa livre de risco (FRED) e VIX.

A taxa livre de risco não é detalhe. A amostra vai de 2007 (Fed funds a 5%) a
2021 (zero) e volta a 4%+; tratar caixa como rendendo zero subestima o retorno
da estratégia justamente nos períodos em que ela fica desalocada, e distorce
qualquer comparação de Sharpe entre subperíodos.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from mare.io.cache import cached_frame

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
TRADING_DAYS = 252


def fetch_fred(series: str, cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """Baixa uma série do FRED como DataFrame [date, value]."""
    def build() -> pd.DataFrame:
        text = requests.get(FRED_CSV.format(series=series), timeout=60).text
        frame = pd.read_csv(io.StringIO(text))
        frame.columns = ["date", "value"]
        frame["date"] = pd.to_datetime(frame["date"])
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        return frame.dropna().reset_index(drop=True)

    return cached_frame(Path(cache_dir) / f"fred_{series}.parquet", build,
                        refresh=refresh, source=f"FRED:{series}")


def daily_risk_free(series_frame: pd.DataFrame,
                    index: pd.DatetimeIndex) -> pd.Series:
    """Converte a taxa anualizada em percentual para retorno diário simples."""
    annual = (series_frame.set_index("date")["value"] / 100.0).sort_index()
    aligned = annual.reindex(index.union(annual.index)).ffill().reindex(index)
    return (aligned / TRADING_DAYS).fillna(0.0)


def fetch_series(symbol: str, start: str, end: str, cache_dir: Path,
                 refresh: bool = False) -> pd.Series:
    """Série de fechamento ajustado de um único ticker (benchmark, VIX)."""
    from mare.io.prices import download_prices

    panel = download_prices([symbol], start, end, Path(cache_dir) / "series",
                            refresh=refresh)
    if panel.empty:
        return pd.Series(dtype=float, name=symbol)
    out = panel.set_index("date")["close_adj"]
    out.name = symbol
    return out.sort_index()
