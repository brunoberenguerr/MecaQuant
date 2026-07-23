"""Reconstrói o universo point-in-time e audita a cobertura de dados.

Rodar ANTES de qualquer backtest. A tabela de cobertura que este script gera
decide se o período de teste é defensável: se muitos membros históricos não têm
preço, o viés de sobrevivência sobrevive à reconstrução do membership e isso
precisa ser dimensionado, não ignorado.

    python scripts/01_build_universe.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

from mare.config import load_config
from mare.io import prices, universe, wikipedia

warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="ignora o cache")
    args = parser.parse_args()

    cfg = load_config()
    root = cfg.root()
    interim = root / "data" / "interim"
    interim.mkdir(parents=True, exist_ok=True)

    print("[1/4] Wikipedia: constituintes e change-log")
    cons = wikipedia.fetch_constituents(cfg.data.cache_dir, refresh=args.refresh)
    changes = wikipedia.fetch_changes(cfg.data.cache_dir, refresh=args.refresh)
    print(f"      {len(cons)} constituintes hoje | {len(changes)} mudanças "
          f"({changes.date.min():%Y-%m-%d} a {changes.date.max():%Y-%m-%d})")

    print("[2/4] Reconstruindo membership point-in-time")
    members = universe.build_membership(cons, changes, asof=cfg.data.end)
    symbols = members.all_symbols()
    years = pd.date_range(cfg.data.start, cfg.data.end, freq="YS")
    audit = universe.audit_membership(members, years)
    print(f"      união histórica: {len(symbols)} símbolos")
    print(f"      contagem reconstruída: min={audit.n_members.min()} "
          f"max={audit.n_members.max()} (índice real ~503 tickers)")

    print(f"[3/4] Baixando preços de {len(symbols)} símbolos (lotes em cache)")
    panel = prices.download_prices(symbols, cfg.data.start, cfg.data.end,
                                   cfg.data.cache_dir, refresh=args.refresh)
    print(f"      {len(panel):,} linhas | {panel.symbol.nunique()} símbolos com dado")

    print("[4/4] Auditoria de cobertura")
    disp = universe.dispositions(changes)
    cover = prices.coverage_report(panel, symbols, disp)
    _report_coverage(cover, members, years)

    _save(interim, members, panel, cover, audit, disp)
    print(f"\nArtefatos gravados em {interim}")


def _report_coverage(cover: pd.DataFrame, members: universe.Membership,
                     years: pd.DatetimeIndex) -> None:
    missing = cover.loc[~cover.has_data]
    print(f"      sem dado no Yahoo: {len(missing)} de {len(cover)} "
          f"({len(missing) / len(cover):.1%})")
    if "disposition" in missing.columns and not missing.empty:
        print("      disposição dos ausentes:")
        for name, count in missing.disposition.value_counts().items():
            print(f"        {name:>14}: {count}")

    have = set(cover.loc[cover.has_data, "symbol"])
    rows = []
    for date in years:
        active = members.on(date)
        if not active:
            continue
        covered = len(active & have)
        rows.append({"ano": date.year, "membros_pit": len(active),
                     "com_dado": covered,
                     "cobertura": covered / len(active)})
    table = pd.DataFrame(rows)
    print("\n      Cobertura por ano (membros PIT que têm preço no Yahoo):")
    for _, r in table.iterrows():
        bar = "#" * int(r.cobertura * 40)
        print(f"        {int(r.ano)}  {int(r.com_dado):>3}/{int(r.membros_pit):>3}  "
              f"{r.cobertura:6.1%}  {bar}")


def _save(interim: Path, members: universe.Membership, panel: pd.DataFrame,
          cover: pd.DataFrame, audit: pd.DataFrame, disp: pd.DataFrame) -> None:
    rows = [{"valid_from": d, "symbols": "|".join(sorted(s))}
            for d, s in members.boundaries]
    pd.DataFrame(rows).to_parquet(interim / "membership.parquet")
    panel.to_parquet(interim / "prices_panel.parquet")
    cover.to_parquet(interim / "coverage.parquet")
    audit.to_csv(interim / "membership_audit.csv")
    disp.to_parquet(interim / "dispositions.parquet")


if __name__ == "__main__":
    main()
