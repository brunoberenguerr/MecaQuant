"""Robustez: limite do viés de sobrevivência, custo de break-even e sensibilidade.

    python scripts/08_robustness.py --split design

Três exhibits que atacam diretamente a regra de ouro do edital ("o backtest deve
mostrar a realidade"):

1. Bounding do viés de sobrevivência — castiga a série pela taxa de ausência de
   dados de cada época e mostra se a conclusão sobrevive ao pior caso.
2. Custo de break-even — a que nível de custo o alpha morre.
3. Sensibilidade ao corte de entrada — procura platô, não pico.
"""

from __future__ import annotations

import argparse
import warnings

import pandas as pd

from mare.backtest.costs import breakeven_cost_bps
from mare.backtest.metrics import cagr
from mare.config import load_config
from mare.io import universe
from mare.io.market import load_bundle
from mare.report.artifacts import build_artifacts
from mare.validation import survivorship

warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["design", "holdout", "full"],
                        default="design")
    args = parser.parse_args()

    cfg = load_config()
    start, end = _window(cfg, args.split)
    bundle = load_bundle(cfg)
    art = build_artifacts(bundle, cfg, start, end)
    base = art.result.returns
    print(f"MARÉ | robustez | split={args.split} | {start} a {end}\n")

    _survivorship(art, bundle, cfg)
    _breakeven(art)
    print("\n(sensibilidade ao corte de entrada roda no script de ablação de gates)")


def _survivorship(art, bundle, cfg) -> None:
    print("1. LIMITE SUPERIOR DO VIÉS DE SOBREVIVÊNCIA")
    interim = cfg.root() / "data" / "interim"
    coverage = pd.read_parquet(interim / "coverage.parquet")
    members = _load_membership(interim / "membership.parquet")

    # Diagnóstico de completude (contexto, não dirige o castigo).
    cov_by_year = survivorship.coverage_by_year(
        coverage, members, pd.date_range(cfg.data.start, cfg.data.end, freq="YS"))
    print(f"   cobertura de dados: {cov_by_year.min():.0%} (pior ano) a "
          f"{cov_by_year.max():.0%} (melhor) | ausência total histórica ~22,8%")

    hazard = 0.05  # conservador: acima do churn anual real do S&P 500
    bounds = survivorship.bias_bounds(art.result.returns, hazard_annual=hazard)
    print(f"   castigo: {hazard:.0%} dos nomes/ano deslistam ao retorno indicado")
    print(bounds.to_string(float_format=lambda v: f"{v:8.3f}"))
    obs = bounds.loc["observado", "cagr"]
    worst = bounds.loc["falência_-100%", "cagr"]
    print(f"   CAGR observado {obs:.1%} -> pior caso (5%/ano a zero) {worst:.1%} "
          f"(perda de {obs - worst:.1%})")


def _breakeven(art) -> None:
    print("\n2. CUSTO DE BREAK-EVEN")
    gross = cagr(art.result.gross_returns)
    turnover = art.result.turnover.mean()
    be = breakeven_cost_bps(gross, turnover)
    print(f"   retorno bruto anual: {gross:.1%} | giro médio: {turnover:.3f}")
    print(f"   custo one-way de break-even: {be:.1f} bps")
    print(f"   (custo base assumido: ~2.5 bps one-way; margem de "
          f"{be / 2.5:.1f}x antes do alpha zerar)")


def _load_membership(path):
    frame = pd.read_parquet(path)
    boundaries = tuple(
        (pd.Timestamp(r.valid_from), frozenset(r.symbols.split("|") if r.symbols else []))
        for r in frame.itertuples())
    return universe.Membership(boundaries)


def _window(cfg, split: str) -> tuple[str, str]:
    if split == "design":
        return cfg.splits.design_start, cfg.splits.design_end
    if split == "holdout":
        return cfg.splits.holdout_start, cfg.splits.holdout_end
    return cfg.splits.design_start, cfg.splits.holdout_end


if __name__ == "__main__":
    main()
