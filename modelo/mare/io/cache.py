"""Cache em disco para os downloads.

Existe por dois motivos além de velocidade: (1) o backtest precisa ser
reprodutível mesmo que o yfinance mude os dados retroativamente — o que ele faz,
via reajuste de preços; (2) sem cache, iterar no sinal significa martelar a API.

Cada artefato guarda um `.meta.json` com data de captura e origem, para o
relatório poder dizer exatamente de quando são os dados.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Callable

import pandas as pd


def cached_frame(path: Path, builder: Callable[[], pd.DataFrame], *,
                 refresh: bool = False, source: str = "") -> pd.DataFrame:
    """Devolve o parquet em `path`, construindo-o via `builder` se preciso."""
    path = Path(path)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    frame = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    _write_meta(path, source=source, rows=len(frame))
    return frame


def _write_meta(path: Path, *, source: str, rows: int) -> None:
    meta = {
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "rows": rows,
    }
    path.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")


def read_meta(path: Path) -> dict:
    meta_path = Path(path).with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))
