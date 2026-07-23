"""Busca a tese econômica que sobrevive aos dados.

A hipótese original (Nagel 2012: reversão é prêmio por liquidez, maior sob
estresse) foi REFUTADA — o IC é maior em VIX baixo, não alto. Este script testa
mecanismos alternativos que seriam consistentes com "funciona em mercado calmo".

    python scripts/05_econ_thesis.py --every 2

Candidatos testados, cada um com uma previsão condicional distinta:

1. Pressão de fluxo de fim de mês / rebalanceamento institucional. Fundos
   rebalanceiam perto da virada do mês; a demanda inelástica empurra preços e
   depois reverte. Previsão: IC maior em janelas que cruzam a virada do mês.

2. Dispersão cross-sectional. Reversão precisa de dispersão para haver o que
   reverter. Previsão: IC maior quando a dispersão de retornos idiossincráticos
   está alta — o que NÃO é a mesma coisa que VIX alto.

3. Concentração setorial do sinal. Se os "baratos" vêm todos de um setor, o que
   parece reversão idiossincrática é aposta setorial. Previsão: IC condicionado
   a quão espalhados por setor estão os nomes selecionados.
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
from mare.validation.signal_quality import (forward_excess_returns,
                                            information_coefficient)

warnings.filterwarnings("ignore")
HORIZON = 21


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--every", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config()
    bundle = load_bundle(cfg)
    plan = schedule(bundle.market.dates, weekday=cfg.backtest.rebalance_weekday,
                    lag_days=cfg.backtest.signal_lag_days,
                    frequency=cfg.backtest.rebalance)
    plan = plan[(plan.signal_date >= pd.Timestamp(cfg.splits.design_start))
                & (plan.trade_date <= pd.Timestamp(cfg.splits.design_end))
                ].iloc[::args.every]
    print(f"MARÉ | busca de tese econômica | {len(plan)} datas | horizonte {HORIZON}d\n")

    rows = _collect(bundle, cfg, plan)
    frame = pd.DataFrame(rows).dropna(subset=["ic"])
    _analyze(frame, bundle)

    print("\n5. CRUZAMENTO HORIZONTE × VIX (teste da tese liquidez-vs-informação)")
    print("   esperado: relação alto-baixo POSITIVA em 1-3d, NEGATIVA em 10-21d")
    crossover = horizon_vix_crossover(bundle, cfg, plan)
    print(crossover.to_string(float_format=lambda v: f"{v:+.4f}"))


def _collect(bundle, cfg, plan) -> list[dict]:
    rows, k_history = [], []
    for row in plan.itertuples():
        table = candidate_table(bundle.market.view(row.signal_date), cfg,
                                k_history=k_history)
        if table.empty:
            continue
        signal = table["s_score"]
        fwd = forward_excess_returns(bundle.market.returns, row.trade_date,
                                     list(signal.index), HORIZON)
        if fwd.empty:
            continue
        rows.append({
            "date": row.trade_date,
            "ic": -information_coefficient(signal, fwd),   # >0 confirma reversão
            "crosses_month_end": _crosses_month_end(row.trade_date, HORIZON),
            "dispersion": float(signal.std()),
            "vix": float(bundle.vix.get(row.signal_date, np.nan)),
        })
    return rows


def _crosses_month_end(date: pd.Timestamp, horizon: int) -> bool:
    """A janela de holding contém uma virada de mês?"""
    window = pd.bdate_range(date, periods=horizon + 1)
    return window.month.nunique() > 1


def _analyze(frame: pd.DataFrame, bundle) -> None:
    overall = frame["ic"].mean()
    t = frame["ic"].mean() / frame["ic"].std(ddof=1) * np.sqrt(len(frame))
    print(f"IC médio geral: {overall:+.4f}  (t≈{t:.2f}, n={len(frame)})\n")

    print("1. FLUXO DE FIM DE MÊS")
    _binary_split(frame, "crosses_month_end", "janela cruza virada de mês")

    print("\n2. DISPERSÃO CROSS-SECTIONAL (tercis)")
    _tercile_split(frame, "dispersion", "dispersão de s-score")

    print("\n3. VIX (replica a refutação da tese original, para contraste)")
    _tercile_split(frame, "vix", "VIX")

    print("\n4. INTERAÇÃO dispersão × VIX")
    frame = frame.copy()
    frame["disp_hi"] = frame["dispersion"] > frame["dispersion"].median()
    frame["vix_hi"] = frame["vix"] > frame["vix"].median()
    grid = frame.groupby(["disp_hi", "vix_hi"])["ic"].agg(["mean", "count"])
    print(grid.to_string(float_format=lambda v: f"{v:+.4f}"))


def horizon_vix_crossover(bundle, cfg, plan) -> pd.DataFrame:
    """O teste decisivo da tese liquidez-vs-informação.

    Previsão: a relação IC × VIX INVERTE de sinal com o horizonte.
    - Curto prazo (1-3d): liquidez intraday/overnight domina; comprar quedas paga
      MAIS em estresse (é o resultado de Nagel 2012 para reversão diária).
    - Médio prazo (10-21d): a queda em estresse teve tempo de ser confirmada por
      informação e NÃO reverte; o edge migra para o regime calmo.

    Se o cruzamento aparecer, temos a reconciliação com Nagel via horizonte, e o
    overlay de regime ganha uma implicação testada: reduzir exposição em VIX alto.
    """
    horizons = (1, 2, 3, 5, 10, 21)
    rows, k_history = [], []
    for row in plan.itertuples():
        table = candidate_table(bundle.market.view(row.signal_date), cfg,
                                k_history=k_history)
        if table.empty:
            continue
        signal = table["s_score"]
        vix = float(bundle.vix.get(row.signal_date, np.nan))
        rec = {"vix": vix}
        for h in horizons:
            fwd = forward_excess_returns(bundle.market.returns, row.trade_date,
                                         list(signal.index), h)
            rec[h] = -information_coefficient(signal, fwd) if not fwd.empty else np.nan
        rows.append(rec)

    frame = pd.DataFrame(rows).dropna(subset=["vix"])
    hi = frame["vix"] > frame["vix"].median()
    out = {}
    for h in horizons:
        out[h] = {"VIX_baixo": frame.loc[~hi, h].mean(),
                  "VIX_alto": frame.loc[hi, h].mean()}
    result = pd.DataFrame(out).T
    result["alto_menos_baixo"] = result["VIX_alto"] - result["VIX_baixo"]
    return result


def _binary_split(frame: pd.DataFrame, column: str, label: str) -> None:
    grouped = frame.groupby(column)["ic"].agg(["mean", "std", "count"])
    for value, r in grouped.iterrows():
        tag = f"{label}=SIM" if value else f"{label}=não"
        t = r["mean"] / r["std"] * np.sqrt(r["count"]) if r["std"] > 0 else np.nan
        print(f"  {tag:>32}: IC={r['mean']:+.4f}  t={t:+.2f}  (n={int(r['count'])})")


def _tercile_split(frame: pd.DataFrame, column: str, label: str) -> None:
    sub = frame.dropna(subset=[column])
    if len(sub) < 9:
        return
    terciles = pd.qcut(sub[column], 3, labels=["baixo", "médio", "alto"])
    grouped = sub.groupby(terciles, observed=True)["ic"].agg(["mean", "std", "count"])
    for value, r in grouped.iterrows():
        t = r["mean"] / r["std"] * np.sqrt(r["count"]) if r["std"] > 0 else np.nan
        print(f"  {label} {value:>6}: IC={r['mean']:+.4f}  t={t:+.2f}  (n={int(r['count'])})")


if __name__ == "__main__":
    main()
