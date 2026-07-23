"""Mede o poder preditivo do s-score antes de qualquer construção de carteira.

    python scripts/03_signal_quality.py --split design --every 10

Roda em uma amostra de datas (não em todas) porque o objetivo é diagnóstico, não
apuração de P&L — e amostrar deixa o feedback em minutos em vez de dezenas.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from mare.calendar import schedule
from mare.config import load_config
from mare.io.market import load_bundle
from mare.signal.score import candidate_table
from mare.validation import signal_quality as sq

warnings.filterwarnings("ignore")
HORIZONS = (5, 10, 21)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["design", "holdout", "full"],
                        default="design")
    parser.add_argument("--every", type=int, default=10,
                        help="usa 1 a cada N datas de rebalanceamento")
    args = parser.parse_args()

    cfg = load_config()
    start, end = _window(cfg, args.split)
    bundle = load_bundle(cfg)
    print(f"MARÉ | qualidade do sinal | split={args.split} | {start} a {end}")

    plan = schedule(bundle.market.dates, weekday=cfg.backtest.rebalance_weekday,
                    lag_days=cfg.backtest.signal_lag_days,
                    frequency=cfg.backtest.rebalance)
    plan = plan[(plan.signal_date >= pd.Timestamp(start))
                & (plan.trade_date <= pd.Timestamp(end))].iloc[::args.every]
    print(f"  {len(plan)} datas amostradas\n")

    ics, deciles, coverage = _collect(bundle, cfg, plan)
    _report(ics, deciles, coverage, bundle, cfg, plan)


def _collect(bundle, cfg, plan):
    ics = {h: {} for h in HORIZONS}
    deciles = {h: [] for h in HORIZONS}
    coverage = {}
    k_history: list[int] = []

    for row in plan.itertuples():
        view = bundle.market.view(row.signal_date)
        table = candidate_table(view, cfg, k_history=k_history)
        if table.empty:
            continue
        coverage[row.signal_date] = {
            "n_universo": len(table),
            "n_baratos": int(table["cheap"].sum()),
            "n_aprovados": int(table["eligible"].sum()),
            "k": int(table["k_factors"].iloc[0]),
        }
        signal = table["s_score"]
        for horizon in HORIZONS:
            fwd = sq.forward_excess_returns(bundle.market.returns, row.trade_date,
                                            list(signal.index), horizon)
            if fwd.empty:
                continue
            ics[horizon][row.signal_date] = sq.information_coefficient(signal, fwd)
            dec = sq.decile_returns(signal, fwd)
            if not dec.empty:
                deciles[horizon].append(dec)
            coverage.setdefault(row.signal_date, {}).update(
                _basket_returns(table, fwd, cfg, horizon))
    return ics, deciles, coverage


def _basket_returns(table: pd.DataFrame, fwd: pd.Series, cfg,
                    horizon: int) -> dict[str, float]:
    """Compara o que a estratégia REALMENTE compra contra alternativas simples.

    Os gates (P(alvo) mínima, banda de meia-vida) e o ranking por mu_s não são
    neutros: eles escolhem um subconjunto dos nomes baratos. Se esse subconjunto
    render MENOS que simplesmente pegar os mais baratos, os gates estão
    selecionando contra o próprio sinal — e é melhor descobrir isso aqui do que
    depois de otimizar pesos em cima de uma seleção ruim.
    """
    n = cfg.portfolio.n_names
    out: dict[str, float] = {}

    cheapest = table.nsmallest(n, "s_score").index
    out[f"r{horizon}_mais_baratos"] = float(fwd.reindex(cheapest).mean())

    approved = table.loc[table["eligible"]]
    if not approved.empty:
        by_mu = approved.nlargest(n, "mu_s").index
        out[f"r{horizon}_aprovados_mu_s"] = float(fwd.reindex(by_mu).mean())
        by_s = approved.nsmallest(n, "s_score").index
        out[f"r{horizon}_aprovados_s"] = float(fwd.reindex(by_s).mean())
    return out


def _report(ics, deciles, coverage, bundle, cfg, plan) -> None:
    print("INFORMATION COEFFICIENT (ic_reversao > 0 confirma a hipótese)")
    rows = [sq.summarize_ic(pd.Series(ics[h]), h) for h in HORIZONS]
    print(pd.DataFrame(rows).set_index("horizonte").to_string(
        float_format=lambda v: f"{v:9.4f}"))

    print("\nRETORNO EXCEDENTE POR DECIL DE s-SCORE (decil 1 = mais barato)")
    for horizon in HORIZONS:
        if not deciles[horizon]:
            continue
        avg = pd.concat(deciles[horizon], axis=1).mean(axis=1) * 100
        spread = avg.iloc[0] - avg.iloc[-1]
        bars = "  ".join(f"{v:+6.3f}" for v in avg)
        print(f"  h={horizon:>2}d  {bars}   | spread D1-D10: {spread:+.3f}%")

    cov = pd.DataFrame(coverage).T
    print(f"\nCOBERTURA  universo={cov.n_universo.median():.0f}"
          f" | baratos={cov.n_baratos.median():.0f}"
          f" | aprovados={cov.n_aprovados.median():.0f}"
          f" | k={cov.k.median():.0f}")

    print("\nOS GATES AJUDAM OU ATRAPALHAM? (retorno excedente médio da cesta, %)")
    for horizon in HORIZONS:
        cols = [c for c in cov.columns if c.startswith(f"r{horizon}_")]
        if not cols:
            continue
        means = cov[cols].astype(float).mean() * 100
        parts = "  ".join(f"{c.split('_', 1)[1]}={means[c]:+.3f}" for c in cols)
        print(f"  h={horizon:>2}d  {parts}")

    print("\nIC POR ANO (horizonte 10d)")
    by_year = pd.Series(ics[10]).dropna()
    if not by_year.empty:
        annual = (-by_year).groupby(by_year.index.year).agg(["mean", "count"])
        for year, r in annual.iterrows():
            mark = "+" if r["mean"] > 0 else "-"
            print(f"  {year}  ic_reversao={r['mean']:+.4f}  ({int(r['count'])} datas) {mark}")

    _test_liquidity_thesis(ics, bundle)


def _test_liquidity_thesis(ics, bundle) -> None:
    """Testa a tese econômica diretamente: o edge é maior quando a liquidez seca?

    Nagel (2012) e Grossman & Miller (1988): reversão de curto prazo é prêmio por
    prover liquidez a fluxo impaciente, e o prêmio sobe quando o capital de
    intermediação está restrito. O VIX é a proxy observável dessa restrição.

    Se a tese vale, o IC tem de ser materialmente maior no tercil superior de VIX.
    Isso transforma o overlay de regime de filtro decorativo em implicação
    econômica TESTADA — e é o que separa "achamos um padrão" de "entendemos por
    que o padrão existe".
    """
    print("\nTESE DE PROVISÃO DE LIQUIDEZ: IC condicionado ao VIX")
    for horizon in HORIZONS:
        series = pd.Series(ics[horizon]).dropna()
        if len(series) < 15:
            continue
        vix = bundle.vix.reindex(series.index).ffill()
        joined = pd.DataFrame({"ic": -series, "vix": vix}).dropna()
        if len(joined) < 15:
            continue
        tercile = pd.qcut(joined["vix"], 3, labels=["VIX baixo", "VIX médio", "VIX alto"])
        table = joined.groupby(tercile, observed=True)["ic"].agg(["mean", "std", "count"])
        parts = [f"{name}={r['mean']:+.4f}" for name, r in table.iterrows()]
        low = table.loc["VIX baixo", "mean"]
        high = table.loc["VIX alto", "mean"]
        print(f"  h={horizon:>2}d  " + "  ".join(parts) + f"   | alto−baixo: {high - low:+.4f}")


def _window(cfg, split: str) -> tuple[str, str]:
    if split == "design":
        return cfg.splits.design_start, cfg.splits.design_end
    if split == "holdout":
        return cfg.splits.holdout_start, cfg.splits.holdout_end
    return cfg.splits.design_start, cfg.splits.holdout_end


if __name__ == "__main__":
    main()
