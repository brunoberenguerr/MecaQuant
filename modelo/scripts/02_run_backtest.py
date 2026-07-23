"""Roda o backtest e registra o resultado no run registry.

    python scripts/02_run_backtest.py --split design
    python scripts/02_run_backtest.py --split holdout   # só depois do congelamento

O split `holdout` existe separado de propósito: toda calibração acontece em
`design`, e cada vez que o holdout é olhado fica registrado.
"""

from __future__ import annotations

import argparse
import warnings

import pandas as pd

from mare import registry
from mare.backtest.engine import run_backtest
from mare.backtest.metrics import summary, three_series
from mare.config import load_config
from mare.io.market import load_bundle
from mare.portfolio.construct import SignalDiagnostics, make_signal_fn

warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["design", "holdout", "full"],
                        default="design")
    parser.add_argument("--label", default="", help="rótulo do run no registry")
    parser.add_argument("--set", nargs="*", default=[],
                        metavar="CHAVE=VALOR",
                        help="override de config, ex.: portfolio.n_names=25")
    args = parser.parse_args()

    cfg = load_config(**_parse_overrides(args.set))
    start, end = _window(cfg, args.split)
    print(f"MARÉ | split={args.split} | {start} a {end} | config={cfg.hash}")

    bundle = load_bundle(cfg)
    print(f"  painel: {bundle.market.returns.shape[0]} pregões × "
          f"{bundle.market.returns.shape[1]} símbolos")

    diagnostics = SignalDiagnostics()
    signal_fn = make_signal_fn(sectors=bundle.sectors, diagnostics=diagnostics)
    result = run_backtest(bundle.market, cfg, signal_fn, start=start, end=end,
                          risk_free=bundle.risk_free)

    _report(result, bundle, cfg, args, diagnostics, start, end)


def _report(result, bundle, cfg, args, diagnostics, start, end) -> None:
    rf = bundle.risk_free
    series = three_series(result.returns, bundle.benchmark)
    rows = [summary(series[col], rf, label=col) for col in series.columns]
    rows.append(summary(bundle.benchmark.reindex(series.index), rf, label="SPY"))
    table = pd.DataFrame(rows).set_index("label")

    print("\n" + "=" * 78)
    print(table[["n_dias", "cagr", "vol", "sharpe", "sortino",
                 "max_drawdown", "t_stat_nw"]].to_string(
        float_format=lambda v: f"{v:8.3f}"))
    print("=" * 78)

    n_rebal = len(result.turnover)
    print(f"\nrebalanceamentos: {n_rebal} | giro médio: {result.turnover.mean():.3f}"
          f" | custo total: {result.costs.sum():.4f}")
    print(f"exposição média: {result.exposure.mean():.3f}"
          f" | posições médias: {(result.weights > 0).sum(axis=1).mean():.1f}")
    approved = pd.Series(diagnostics.n_approved)
    if not approved.empty:
        print(f"candidatos aprovados por data: mediana={approved.median():.0f}"
              f" min={approved.min()} max={approved.max()}"
              f" | datas sem candidato: {(approved == 0).sum()}")
    trades = diagnostics.trades_frame()
    if not trades.empty:
        print(f"\nsaídas por motivo ({len(trades)} no total):")
        for motivo, count in trades["motivo"].value_counts().items():
            share = count / len(trades)
            dias = trades.loc[trades["motivo"] == motivo, "dias_no_livro"].median()
            print(f"  {motivo:>22}: {count:5d} ({share:5.1%})  mediana {dias:.0f} dias")
    print(f"\nfingerprint da curva: {result.fingerprint()}")

    metrics = table.loc["total"].to_dict()
    metrics.update({"turnover": float(result.turnover.mean()),
                    "n_rebalances": n_rebal,
                    "fingerprint": result.fingerprint()})
    registry.log_run(cfg.hash, metrics, split=args.split, label=args.label,
                     extra={"start": start, "end": end,
                            "overrides": " ".join(args.set)})
    print(f"registrado no registry (N distinto = {registry.n_trials()})")


def _window(cfg, split: str) -> tuple[str, str]:
    if split == "design":
        return cfg.splits.design_start, cfg.splits.design_end
    if split == "holdout":
        return cfg.splits.holdout_start, cfg.splits.holdout_end
    return cfg.splits.design_start, cfg.splits.holdout_end


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


if __name__ == "__main__":
    main()
