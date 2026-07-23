"""Número efetivo de apostas: MARÉ segura N tickers, mas quantas apostas
INDEPENDENTES isso vale?

    python scripts/12_effective_bets.py --split design

Responde a crítica correta da Lei Fundamental: amplitude conta apostas
independentes, não tickers. Mede, nas datas REAIS de rebalanceamento do
backtest, o número efetivo de apostas (Meucci 2010) de duas formas:

  (a) sobre o RETORNO BRUTO dos nomes segurados — o que a crítica assume;
  (b) sobre o RESÍDUO pós-PCA/RMT desses mesmos nomes — o que MARÉ realmente
      segura, depois de remover os k fatores comuns.

Se a residualização funciona, (b) deve ter ENB muito mais perto de N do que (a).
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from mare.backtest.engine import run_backtest
from mare.config import load_config
from mare.io.market import load_bundle
from mare.portfolio.construct import SignalDiagnostics, make_signal_fn
from mare.signal import pca
from mare.signal.residual import fit_residuals
from mare.validation.effective_bets import effective_bets

warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["design", "holdout", "full"],
                        default="design")
    parser.add_argument("--sample-every", type=int, default=10,
                        help="usa 1 a cada N rebalanceamentos (custo: PCA por data)")
    args = parser.parse_args()

    cfg = load_config()
    start, end = _window(cfg, args.split)
    bundle = load_bundle(cfg)
    print(f"MARÉ | apostas efetivas | split={args.split} | {start} a {end}\n")

    diag = SignalDiagnostics()
    signal_fn = make_signal_fn(sectors=bundle.sectors, diagnostics=diag)
    run_backtest(bundle.market, cfg, signal_fn, start=start, end=end,
                risk_free=bundle.risk_free)

    dates = sorted(diag.candidates)[::args.sample_every]
    print(f"amostrando {len(dates)} de {len(diag.candidates)} rebalanceamentos\n")

    rows = []
    for as_of in dates:
        held = _held_symbols(diag, as_of, cfg.portfolio.n_names)
        if len(held) < cfg.portfolio.n_names // 2:
            continue
        raw_enb, resid_enb, k = _compute_both(bundle, cfg, as_of, held)
        if raw_enb is None:
            continue
        rows.append({"as_of": as_of, "n": len(held), "k_factors": k,
                     "enb_bruto": raw_enb["enb_meucci"],
                     "corr_bruto": raw_enb["avg_corr"],
                     "enb_residuo": resid_enb["enb_meucci"],
                     "corr_residuo": resid_enb["avg_corr"]})

    table = pd.DataFrame(rows).set_index("as_of")
    print(table.to_string(float_format=lambda v: f"{v:8.2f}"))
    print("\n" + "=" * 70)
    print(f"medianas | n={table['n'].median():.0f} tickers"
          f" | k fatores removidos={table['k_factors'].median():.0f}")
    print(f"  ENB sobre retorno BRUTO:   {table['enb_bruto'].median():.1f}"
          f"  (corr média={table['corr_bruto'].median():.3f})")
    print(f"  ENB sobre RESÍDUO (MARÉ):  {table['enb_residuo'].median():.1f}"
          f"  (corr média={table['corr_residuo'].median():.3f})")
    ratio = table['enb_residuo'].median() / table['enb_bruto'].median()
    print(f"\nresidualizar multiplica as apostas efetivas por ~{ratio:.1f}x")


def _held_symbols(diag: SignalDiagnostics, as_of, n_names: int) -> list[str]:
    ranked = diag.candidates.get(as_of)
    if ranked is None or ranked.empty:
        return []
    return list(ranked.index[:n_names])


def _compute_both(bundle, cfg, as_of, held):
    view = bundle.market.view(as_of)
    eligible = view.eligible(
        min_dollar_volume=cfg.universe.min_dollar_volume,
        min_price=cfg.universe.min_price, min_valid_returns=cfg.universe.min_valid_returns,
        adv_window=cfg.universe.adv_window,
        history=min(cfg.factor.window, len(view.returns)))
    from mare.features.returns import winsorize
    history = view.tail(view.returns, cfg.factor.window)[eligible]
    clean = winsorize(history, cfg.sanity.winsorize_abs)
    clean = clean.loc[:, clean.notna().sum() >= cfg.universe.min_valid_returns].fillna(0.0)
    if not set(held).issubset(clean.columns):
        held = [s for s in held if s in clean.columns]
    if len(held) < 5:
        return None, None, None

    factors, _, k = pca.build_factors(
        clean, halflife=cfg.factor.halflife, k_method=cfg.factor.k_method,
        k_fixed=cfg.factor.k_fixed, k_min=cfg.factor.k_min, k_max=cfg.factor.k_max)
    fit = fit_residuals(clean, factors, beta_window=cfg.residual.beta_window,
                        beta_gap=cfg.residual.beta_gap,
                        resid_window=cfg.residual.resid_window)

    raw_window = clean[held].tail(cfg.residual.resid_window)
    resid_window = fit.residuals[held]

    weights = pd.Series(1.0 / len(held), index=held)
    return (effective_bets(raw_window, weights),
           effective_bets(resid_window, weights), k)


def _window(cfg, split: str) -> tuple[str, str]:
    if split == "design":
        return cfg.splits.design_start, cfg.splits.design_end
    if split == "holdout":
        return cfg.splits.holdout_start, cfg.splits.holdout_end
    return cfg.splits.design_start, cfg.splits.holdout_end


if __name__ == "__main__":
    main()
