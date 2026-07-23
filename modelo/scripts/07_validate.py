"""Validação completa: ablação de overlays, atribuição, DSR e placebo.

    python scripts/07_validate.py --split design

Roda o backtest base uma vez e deriva tudo dele. Salva os artefatos em
`reports/<split>/` para o dashboard consumir os mesmos números.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from mare import registry
from mare.backtest.metrics import summary, three_series
from mare.config import load_config
from mare.io.market import load_bundle
from mare.report.artifacts import build_artifacts, save_artifacts
from mare.validation import attribution, dsr, placebo

warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["design", "holdout", "full"],
                        default="design")
    parser.add_argument("--placebo-draws", type=int, default=500)
    parser.add_argument("--dashboard", action="store_true",
                        help="também gera reports/<split>/dashboard.html")
    args = parser.parse_args()

    cfg = load_config()
    start, end = _window(cfg, args.split)
    bundle = load_bundle(cfg)
    print(f"MARÉ | validação | split={args.split} | {start} a {end} | cfg={cfg.hash}\n")

    art = build_artifacts(bundle, cfg, start, end)
    rf = art.risk_free

    _ablation_table(art, rf)
    reg = _attribution(art, cfg, bundle)
    _deflated_sharpe(art)
    null = _placebo(art, bundle, cfg, args.placebo_draws)

    save_artifacts(art, cfg.root() / "reports" / args.split)
    registry.log_run(cfg.hash, summary(art.ablation["regime+vol_target"], rf),
                     split=args.split, label="validate")
    if args.dashboard:
        _dashboard(art, cfg, args.split, reg, null)
    print(f"\nartefatos em reports/{args.split}/ | N distinto no registry = "
          f"{registry.n_trials()}")


def _dashboard(art, cfg, split, reg, null) -> None:
    from mare.backtest.costs import breakeven_cost_bps
    from mare.backtest.metrics import cagr
    from mare.report.dashboard import build_dashboard

    gross = cagr(art.result.gross_returns)
    turnover = art.result.turnover.mean()
    be = (gross, turnover, breakeven_cost_bps(gross, turnover))
    out = cfg.root() / "reports" / split / "dashboard.html"
    path = build_dashboard(art, cfg, split=split, attribution=reg,
                           placebo_null=null, breakeven=be, out_path=out)
    print(f"dashboard: {path} ({path.stat().st_size / 1e6:.1f} MB)")


def _ablation_table(art, rf) -> None:
    print("ABLAÇÃO DE OVERLAYS (Pilar 2) — retorno total")
    rows = [summary(art.ablation[col], rf, label=col) for col in art.ablation.columns]
    rows.append(summary(art.benchmark, rf, label="SPY"))
    frame = pd.DataFrame(rows).set_index("label")
    print(frame[["cagr", "vol", "sharpe", "sortino", "max_drawdown",
                 "calmar", "t_stat_nw"]].to_string(
        float_format=lambda v: f"{v:8.3f}"))

    print("\nSÉRIE ATIVA (estratégia − SPY) por variante")
    active = [summary(three_series(art.ablation[col], art.benchmark)["ativo"],
                      label=col) for col in art.ablation.columns]
    fa = pd.DataFrame(active).set_index("label")
    print(fa[["cagr", "vol", "sharpe", "t_stat_nw"]].to_string(
        float_format=lambda v: f"{v:8.3f}"))


def _attribution(art, cfg, bundle):
    print("\nATRIBUIÇÃO FAMA-FRENCH 5 + MOMENTUM (sobre a série ATIVA)")
    try:
        factors = attribution.load_factors(cfg.data.cache_dir)
    except Exception as exc:  # noqa: BLE001 — rede pode falhar; não travar a validação
        print(f"  (fatores indisponíveis: {exc})")
        return None
    active = three_series(art.ablation["regime+vol_target"], art.benchmark)["ativo"]
    reg = attribution.regress(active, factors)
    if reg.empty:
        print("  (amostra insuficiente)")
        return None
    print(reg[["coef_anual", "t_stat", "p_valor"]].to_string(
        float_format=lambda v: f"{v:8.4f}"))
    r2 = attribution.r_squared(active, factors)
    print(f"  R² dos fatores: {r2:.3f} | alpha anualizado: "
          f"{reg.loc['alpha', 'coef_anual']:.3%} (t={reg.loc['alpha', 't_stat']:.2f})")
    return reg


def _deflated_sharpe(art) -> None:
    print("\nDEFLATED SHARPE RATIO")
    n = max(registry.n_trials(), 1)
    for col in ("base", "regime+vol_target"):
        d = dsr.deflated_sharpe_ratio(art.ablation[col], n_trials=n)
        mtrl = dsr.min_track_record_length(art.ablation[col])
        print(f"  {col:>18}: SR={d.get('sharpe_anual', float('nan')):.3f} "
              f"SR0={d.get('sr0_anual', float('nan')):.3f} "
              f"DSR={d.get('dsr', float('nan')):.3f} "
              f"PSR>0={d.get('psr_vs_0', float('nan')):.3f} "
              f"| N={n} minTRL={mtrl:.0f}d")


def _placebo(art, bundle, cfg, draws) -> None:
    print(f"\nPLACEBO ({draws} amostras)")
    from mare.backtest.metrics import sharpe

    observed = sharpe(art.ablation["base"])
    print(f"  Sharpe observado (base): {observed:.3f}")
    null = None

    # (a) Contra o UNIVERSO líquido inteiro: testa se a reversão residual (o
    # filtro "estar barato") tem valor. É o placebo que valida o sinal.
    universe = {d: p for d, p in art.diagnostics.universe_pool.items()
                if len(p) > cfg.portfolio.n_names}
    if universe:
        null = placebo.random_selection_sharpes(
            bundle.market.returns, universe, cfg.portfolio.n_names,
            n_draws=min(draws, 500))
        res = placebo.summarize(observed, null, label="vs universo líquido")
        print(f"  [vs universo líquido] nulo: média={res.get('nulo_media', float('nan')):.3f}"
              f" p95={res.get('nulo_p95', float('nan')):.3f}"
              f" | percentil do observado: {res.get('percentil', float('nan')):.1%}"
              f" | p-valor: {res.get('p_valor', float('nan')):.3f}")

    # (b) Contra o pool BARATO: testa se selecionar os MAIS baratos adiciona algo
    # além de estar no bucket barato. (Resultado esperado: pouco — o edge é o
    # filtro, não a ordenação fina.)
    eligible = {d: p for d, p in art.diagnostics.eligible_pool.items()
                if len(p) > cfg.portfolio.n_names}
    if eligible:
        rnd = placebo.random_selection_sharpes(
            bundle.market.returns, eligible, cfg.portfolio.n_names,
            n_draws=min(draws, 500))
        res = placebo.summarize(observed, rnd, label="vs pool barato")
        print(f"  [vs pool barato]      nulo: média={res.get('nulo_media', float('nan')):.3f}"
              f" p95={res.get('nulo_p95', float('nan')):.3f}"
              f" | percentil do observado: {res.get('percentil', float('nan')):.1%}"
              f" | p-valor: {res.get('p_valor', float('nan')):.3f}")

    # (b) Block bootstrap: estabilidade do Sharpe (a média é preservada, então
    # centra no observado — mede dispersão, não significância contra zero).
    boot = placebo.block_bootstrap_sharpes(
        art.ablation["base"], n_draws=draws,
        block=cfg.validation.bootstrap_block_days)
    res = placebo.summarize(observed, boot, label="block bootstrap")
    print(f"  [block bootstrap]   nulo: média={res.get('nulo_media', float('nan')):.3f}"
          f" p05={boot.dropna().quantile(0.05):.3f}"
          f" p95={res.get('nulo_p95', float('nan')):.3f}"
          f" (dispersão do Sharpe)")
    return null


def _window(cfg, split: str) -> tuple[str, str]:
    if split == "design":
        return cfg.splits.design_start, cfg.splits.design_end
    if split == "holdout":
        return cfg.splits.holdout_start, cfg.splits.holdout_end
    return cfg.splits.design_start, cfg.splits.holdout_end


if __name__ == "__main__":
    main()
