"""MARÉ (resíduo residualizado) vs reversão simples sobre o retorno bruto.

    python scripts/10_naive_baseline.py --split design

Pergunta: vale a pena decompor o retorno em fator sistemático + resíduo
(PCA/RMT, betas congelados out-of-sample), ou uma reversão à média direta sobre
o preço já entrega o mesmo edge? Roda os dois sinais através do MESMO engine,
MESMO livro de posições, MESMO sizing — a única diferença é a tabela de
candidatos (`score.candidate_table` vs `naive.candidate_table_naive`).
"""

from __future__ import annotations

import argparse
import warnings

import pandas as pd

from mare.backtest.engine import run_backtest
from mare.backtest.metrics import summary, three_series
from mare.config import load_config
from mare.io.market import load_bundle
from mare.portfolio.construct import SignalDiagnostics, make_signal_fn
from mare.signal.naive import candidate_table_naive
from mare.validation.dsr import deflated_sharpe_ratio

warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["design", "holdout", "full"],
                        default="design")
    parser.add_argument("--set", nargs="*", default=[], metavar="CHAVE=VALOR",
                        help="override de config, ex.: portfolio.n_names=10")
    args = parser.parse_args()

    cfg = load_config(**_parse_overrides(args.set))
    start, end = _window(cfg, args.split)
    bundle = load_bundle(cfg)
    print(f"MARÉ vs reversão simples | split={args.split} | {start} a {end}\n")

    variants = {
        "MARÉ (resíduo PCA/RMT)": None,
        "reversão simples (retorno bruto)": candidate_table_naive,
    }

    rows = []
    for label, table_fn in variants.items():
        diag = SignalDiagnostics()
        signal_fn = make_signal_fn(sectors=bundle.sectors, diagnostics=diag,
                                   table_fn=table_fn)
        result = run_backtest(bundle.market, cfg, signal_fn, start=start, end=end,
                              risk_free=bundle.risk_free)
        series = three_series(result.returns, bundle.benchmark)
        row = summary(series["ativo"], label=label)
        row["sharpe_total"] = summary(result.returns, bundle.risk_free)["sharpe"]
        row["dsr"] = deflated_sharpe_ratio(result.returns, n_trials=1).get("dsr", float("nan"))
        row["turnover"] = result.turnover.mean()
        rows.append(row)

    table = pd.DataFrame(rows).set_index("label")
    print("Serie ATIVA (estrategia menos SPY), mesmo engine/livro/sizing:")
    print(table[["cagr", "vol", "sharpe", "sharpe_total", "dsr", "turnover"]]
         .to_string(float_format=lambda v: f"{v:8.3f}"))
    print("\nsharpe = série ativa | sharpe_total = retorno total da carteira")


def _parse_overrides(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        out[key] = _coerce(value)
    return out


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


def _window(cfg, split: str) -> tuple[str, str]:
    if split == "design":
        return cfg.splits.design_start, cfg.splits.design_end
    if split == "holdout":
        return cfg.splits.holdout_start, cfg.splits.holdout_end
    return cfg.splits.design_start, cfg.splits.holdout_end


if __name__ == "__main__":
    main()
