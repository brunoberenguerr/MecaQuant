"""Gera o dashboard HTML autocontido a partir de um backtest.

    python scripts/09_dashboard.py --split design

Roda o backtest base + overlays, calcula atribuição e placebo, e escreve
`reports/<split>/dashboard.html` — abrível offline, sem servidor.
"""

from __future__ import annotations

import argparse
import warnings

from mare.backtest.costs import breakeven_cost_bps
from mare.backtest.metrics import cagr, sharpe
from mare.config import load_config
from mare.io.market import load_bundle
from mare.report.artifacts import build_artifacts
from mare.report.dashboard import build_dashboard
from mare.validation import attribution, placebo

warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["design", "holdout", "full"],
                        default="design")
    args = parser.parse_args()

    cfg = load_config()
    start, end = _window(cfg, args.split)
    bundle = load_bundle(cfg)
    print(f"MARÉ | dashboard | split={args.split} | {start} a {end}")

    art = build_artifacts(bundle, cfg, start, end)

    reg = _attribution(art, cfg)
    null = _placebo(art, bundle, cfg)
    be = _breakeven(art)

    out = cfg.root() / "reports" / args.split / "dashboard.html"
    path = build_dashboard(art, cfg, split=args.split, attribution=reg,
                           placebo_null=null, breakeven=be, out_path=out)
    size_mb = path.stat().st_size / 1e6
    print(f"dashboard: {path} ({size_mb:.1f} MB, autocontido)")


def _attribution(art, cfg):
    from mare.backtest.metrics import three_series
    try:
        factors = attribution.load_factors(cfg.data.cache_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"  (atribuição indisponível: {exc})")
        return None
    active = three_series(art.ablation["regime+vol_target"], art.benchmark)["ativo"]
    return attribution.regress(active, factors)


def _placebo(art, bundle, cfg):
    eligible = {d: pool for d, pool in art.diagnostics.eligible_pool.items()
                if len(pool) > cfg.portfolio.n_names}
    if not eligible:
        return None
    return placebo.random_selection_sharpes(
        bundle.market.returns, eligible, cfg.portfolio.n_names, n_draws=1000)


def _breakeven(art):
    gross = cagr(art.result.gross_returns)
    turnover = art.result.turnover.mean()
    return gross, turnover, breakeven_cost_bps(gross, turnover)


def _window(cfg, split: str) -> tuple[str, str]:
    if split == "design":
        return cfg.splits.design_start, cfg.splits.design_end
    if split == "holdout":
        return cfg.splits.holdout_start, cfg.splits.holdout_end
    return cfg.splits.design_start, cfg.splits.holdout_end


if __name__ == "__main__":
    main()
