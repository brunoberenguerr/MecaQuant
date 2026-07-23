"""Captura bruta das tabelas do S&P 500 na Wikipedia.

Isolado do módulo de reconstrução de propósito: aqui só se busca e normaliza o
formato das tabelas. A lógica de membership vive em `universe.py` e é testável
sem rede.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from mare.io.cache import cached_frame

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MARE-research/0.1)"}


def _get_tables() -> list[pd.DataFrame]:
    html = requests.get(WIKI_URL, headers=_HEADERS, timeout=60).text
    return pd.read_html(io.StringIO(html))


def fetch_constituents(cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """Constituintes vigentes hoje: symbol, security, sector, sub_industry, date_added."""
    def build() -> pd.DataFrame:
        table = _get_tables()[0]
        out = table.rename(columns={
            "Symbol": "symbol", "Security": "security",
            "GICS Sector": "sector", "GICS Sub-Industry": "sub_industry",
            "Date added": "date_added", "CIK": "cik",
        })[["symbol", "security", "sector", "sub_industry", "date_added", "cik"]]
        out["symbol"] = out["symbol"].astype(str).str.strip()
        out["date_added"] = pd.to_datetime(out["date_added"], errors="coerce")
        return out.reset_index(drop=True)

    return cached_frame(Path(cache_dir) / "sp500_constituents.parquet", build,
                        refresh=refresh, source=WIKI_URL)


def fetch_changes(cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """Change-log: date, added, removed, reason.

    Atenção: a tabela se chama "Selected changes" — não há garantia de que seja
    exaustiva para os anos mais antigos. `universe.audit_membership` mede essa
    lacuna comparando a contagem reconstruída contra 500.
    """
    def build() -> pd.DataFrame:
        table = _get_tables()[1]
        table.columns = ["date", "added", "added_security",
                         "removed", "removed_security", "reason"]
        out = table[table["date"].astype(str).str.lower() != "effective date"].copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce", format="mixed")
        for col in ("added", "removed"):
            out[col] = out[col].astype(str).str.strip().replace({"nan": "", "None": ""})
        out["reason"] = out["reason"].astype(str).str.replace(r"\[\d+\]", "", regex=True)
        return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    return cached_frame(Path(cache_dir) / "sp500_changes.parquet", build,
                        refresh=refresh, source=WIKI_URL)
