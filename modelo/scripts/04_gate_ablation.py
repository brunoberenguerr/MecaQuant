"""Ablação dos gates: qual filtro ajuda e qual destrói o sinal.

    python scripts/04_gate_ablation.py --every 4

A tabela de candidatos é calculada UMA vez por data e depois avaliada sob várias
configurações de seleção. Isso torna a ablação barata e — mais importante —
garante que todas as variantes vejam exatamente o mesmo sinal, então a diferença
medida é do filtro, não de ruído de estimação.

Motivação: o diagnóstico de qualidade de sinal mostrou que a cesta efetivamente
comprada rende MENOS que simplesmente pegar os mais baratos. Isso significa que
algum filtro está selecionando contra o próprio sinal, e é preciso saber qual.
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
from mare.validation.signal_quality import forward_excess_returns

warnings.filterwarnings("ignore")
HORIZONS = (10, 21)


def variants(cfg) -> dict:
    """Cada variante é (filtro, chave de ordenação, crescente)."""
    hl_lo, hl_hi = cfg.ou.halflife_min, cfg.ou.halflife_max
    entry, min_prob = cfg.ou.entry_s, cfg.passage.min_prob

    def cheap(t):
        return t["s_score"] <= entry

    return {
        "baseline_todos_baratos": (cheap, "s_score", True),
        "+gate_meia_vida": (lambda t: cheap(t) & t["half_life"].between(hl_lo, hl_hi),
                            "s_score", True),
        "+gate_r2": (lambda t: cheap(t) & t["passes_r2"], "s_score", True),
        "+gate_prob": (lambda t: cheap(t) & (t["prob"] >= min_prob).fillna(False),
                       "s_score", True),
        "todos_gates_rank_s": (lambda t: t["eligible"], "s_score", True),
        "todos_gates_rank_mu_s": (lambda t: t["eligible"], "mu_s", False),
        "sem_gates_rank_mu_s": (cheap, "mu_s", False),
        # Tentativas de consertar a especificação dos gates, não de removê-los.
        # Hipótese: o piso de meia-vida em 10 dias exclui os revertedores mais
        # rápidos, que são justamente os que entregam o retorno.
        "meia_vida_sem_piso": (lambda t: cheap(t) & (t["half_life"] <= hl_hi),
                               "s_score", True),
        "meia_vida_so_piso": (lambda t: cheap(t) & (t["half_life"] >= hl_lo),
                              "s_score", True),
        "prob_relaxada_030": (lambda t: cheap(t) & (t["prob"] >= 0.30).fillna(False),
                              "s_score", True),
        # E se o problema for o corte de entrada em vez dos gates?
        "mais_extremos_s<=-2": (lambda t: t["s_score"] <= -2.0, "s_score", True),
        "rank_por_retorno_esp": (cheap, "expected_return", False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--every", type=int, default=4)
    parser.add_argument("--split", choices=["design", "holdout"], default="design")
    args = parser.parse_args()

    cfg = load_config()
    bundle = load_bundle(cfg)
    start = cfg.splits.design_start if args.split == "design" else cfg.splits.holdout_start
    end = cfg.splits.design_end if args.split == "design" else cfg.splits.holdout_end

    plan = schedule(bundle.market.dates, weekday=cfg.backtest.rebalance_weekday,
                    lag_days=cfg.backtest.signal_lag_days,
                    frequency=cfg.backtest.rebalance)
    plan = plan[(plan.signal_date >= pd.Timestamp(start))
                & (plan.trade_date <= pd.Timestamp(end))].iloc[::args.every]
    print(f"MARÉ | ablação de gates | {args.split} | {len(plan)} datas\n")

    specs = variants(cfg)
    records = {name: {h: [] for h in HORIZONS} for name in specs}
    sizes = {name: [] for name in specs}
    k_history: list[int] = []

    for row in plan.itertuples():
        table = candidate_table(bundle.market.view(row.signal_date), cfg,
                                k_history=k_history)
        if table.empty:
            continue
        forwards = {h: forward_excess_returns(bundle.market.returns, row.trade_date,
                                              list(table.index), h)
                    for h in HORIZONS}
        for name, (mask_fn, key, ascending) in specs.items():
            picks = _select(table, mask_fn, key, ascending, cfg.portfolio.n_names)
            sizes[name].append(len(picks))
            for h in HORIZONS:
                if not forwards[h].empty and len(picks):
                    records[name][h].append(float(forwards[h].reindex(picks).mean()))

    _report(records, sizes, specs)


def _select(table, mask_fn, key, ascending, n):
    subset = table.loc[mask_fn(table).fillna(False)]
    subset = subset.dropna(subset=[key])
    if subset.empty:
        return []
    return subset.sort_values(key, ascending=ascending).head(n).index


def _report(records, sizes, specs) -> None:
    rows = []
    for name in specs:
        row = {"variante": name, "n_medio": float(np.mean(sizes[name] or [0]))}
        for h in HORIZONS:
            values = np.array(records[name][h], dtype=float)
            row[f"ret_{h}d_%"] = values.mean() * 100 if values.size else np.nan
            row[f"t_{h}d"] = (values.mean() / values.std(ddof=1) * np.sqrt(values.size)
                              if values.size > 5 and values.std(ddof=1) > 0 else np.nan)
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("variante")
    print("Retorno excedente médio da cesta e t-stat, por horizonte")
    print(frame.to_string(float_format=lambda v: f"{v:8.3f}"))

    base = frame.loc["baseline_todos_baratos", "ret_21d_%"]
    print(f"\nCusto de cada filtro em 21d (vs baseline {base:+.3f}%):")
    for name in specs:
        if name == "baseline_todos_baratos":
            continue
        delta = frame.loc[name, "ret_21d_%"] - base
        verdict = "AJUDA" if delta > 0.02 else ("NEUTRO" if delta > -0.02 else "ATRAPALHA")
        print(f"  {name:>24}: {delta:+.3f} pp  {verdict}")


if __name__ == "__main__":
    main()
