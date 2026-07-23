"""Download de preços do Yahoo, com cache e auditoria de cobertura.

Duas decisões deliberadas:

1. `auto_adjust=False`. Precisamos das DUAS séries: o Close bruto é o que
   define o filtro de preço mínimo e o dólar-volume (o ajustado reescreve o
   passado a cada split/dividendo e faria uma ação de US$ 3 em 2008 parecer
   elegível hoje); o ajustado só entra no cálculo de retorno.

2. Download em lotes com cache por lote. Pedir 858 símbolos de uma vez falha de
   forma silenciosa e parcial — e uma falha parcial não detectada é exatamente
   como o viés de sobrevivência volta pela porta dos fundos.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from mare.io.cache import cached_frame

_FIELDS = {"Close": "close_raw", "Adj Close": "close_adj",
           "Volume": "volume", "High": "high", "Low": "low"}


def _download_batch(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(symbols, start=start, end=end, auto_adjust=False,
                          progress=False, group_by="column", threads=True)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "symbol", *_FIELDS.values()])
    return _to_long(raw, symbols)


def _to_long(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Achata o DataFrame multi-nível do yfinance para formato longo."""
    frames = []
    for field, name in _FIELDS.items():
        if field not in raw.columns.get_level_values(0):
            continue
        block = raw[field]
        if isinstance(block, pd.Series):  # um único símbolo
            block = block.to_frame(symbols[0])
        frames.append(block.stack().rename(name))
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", *_FIELDS.values()])
    out = pd.concat(frames, axis=1).reset_index()
    out.columns = ["date", "symbol", *[c for c in out.columns[2:]]]
    return out.dropna(subset=["close_adj"], how="all")


def download_prices(symbols: list[str], start: str, end: str, cache_dir: Path,
                    batch_size: int = 60, refresh: bool = False) -> pd.DataFrame:
    """Painel longo [date, symbol, close_raw, close_adj, volume, high, low]."""
    cache_dir = Path(cache_dir) / "prices"
    symbols = sorted(set(symbols))
    frames = []
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        tag = f"{i:04d}_{len(batch)}"
        frames.append(cached_frame(
            cache_dir / f"batch_{tag}.parquet",
            lambda b=batch: _download_batch(b, start, end),
            refresh=refresh, source="yfinance"))
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", *_FIELDS.values()])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def to_wide(panel: pd.DataFrame, field: str) -> pd.DataFrame:
    """Pivota o painel longo para datas × símbolos."""
    return panel.pivot_table(index="date", columns="symbol", values=field,
                             aggfunc="last").sort_index()


def coverage_report(panel: pd.DataFrame, requested: list[str],
                    dispositions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Quem foi pedido, quem veio, e o que aconteceu com quem não veio.

    Esta tabela é o exhibit central sobre viés de sobrevivência. Dropar
    silenciosamente os símbolos sem dados reintroduz o viés — e pior, com a
    falsa sensação de que ele foi resolvido.
    """
    got = panel.groupby("symbol")["close_adj"].agg(["count", "first", "last"])
    frame = pd.DataFrame(index=pd.Index(sorted(set(requested)), name="symbol"))
    frame["n_obs"] = got["count"].reindex(frame.index).fillna(0).astype(int)
    frame["has_data"] = frame["n_obs"] > 0
    spans = panel.groupby("symbol")["date"].agg(["min", "max"])
    frame["first_date"] = spans["min"].reindex(frame.index)
    frame["last_date"] = spans["max"].reindex(frame.index)
    if dispositions is not None and not dispositions.empty:
        last = dispositions.sort_values("date").groupby("symbol").last()
        frame["disposition"] = last["disposition"].reindex(frame.index).fillna("still_member")
    return frame.reset_index()
