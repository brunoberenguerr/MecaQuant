"""Curva de sensibilidade de `n_names`: procura platô, não pico.

    python scripts/11_n_names_sensitivity.py --split design

Testa a Lei Fundamental (Grinold) na prática: IR ≈ IC × amplitude. Varia o
número de posições e mede a série ATIVA (o que importa: alpha de seleção, não
beta emprestado do mercado), turnover (custo) e DSR ajustado pelo N real de
configurações testadas — cada ponto desta curva soma ao N do registry.

Roda o backtest inteiro uma vez por N, carregando o painel de dados uma única
vez para não pagar o custo de I/O a cada ponto.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from mare import registry
from mare.backtest.engine import run_backtest
from mare.backtest.metrics import sharpe, summary, three_series
from mare.config import load_config
from mare.io.market import load_bundle
from mare.portfolio.construct import SignalDiagnostics, make_signal_fn
from mare.validation.dsr import deflated_sharpe_ratio

warnings.filterwarnings("ignore")

GRID = [10, 15, 20, 25, 30, 35, 40, 50, 60]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["design", "holdout", "full"],
                        default="design")
    parser.add_argument("--grid", type=int, nargs="*", default=GRID)
    args = parser.parse_args()

    cfg = load_config()
    start, end = _window(cfg, args.split)
    bundle = load_bundle(cfg)
    print(f"MARÉ | sensibilidade de n_names | split={args.split} | {start} a {end}\n")

    rows = []
    for n in args.grid:
        cfg_n = load_config(**{"portfolio.n_names": n})
        diag = SignalDiagnostics()
        signal_fn = make_signal_fn(sectors=bundle.sectors, diagnostics=diag)
        result = run_backtest(bundle.market, cfg_n, signal_fn, start=start, end=end,
                              risk_free=bundle.risk_free)

        series = three_series(result.returns, bundle.benchmark)
        active = summary(series["ativo"], label=f"n={n}")
        active["sharpe_total"] = sharpe(result.returns, bundle.risk_free)
        active["turnover"] = result.turnover.mean()
        active["dsr"] = deflated_sharpe_ratio(
            result.returns, n_trials=len(args.grid)).get("dsr", float("nan"))
        active["n_names"] = n
        rows.append(active)

        registry.log_run(cfg_n.hash, summary(result.returns, bundle.risk_free),
                         split=args.split, label=f"sensitivity n_names={n}")
        print(f"  n={n:>3} concluido | sharpe_ativo={active['sharpe']:+.3f}"
              f" | sharpe_total={active['sharpe_total']:.3f}"
              f" | turnover={active['turnover']:.3f}")

    table = pd.DataFrame(rows).set_index("n_names")
    print("\n" + "=" * 78)
    print("SERIE ATIVA (estrategia menos SPY) por n_names:")
    print(table[["cagr", "vol", "sharpe", "t_stat_nw", "sharpe_total",
                "turnover", "dsr"]].to_string(float_format=lambda v: f"{v:8.3f}"))
    print("=" * 78)

    best_active = table["sharpe"].idxmax()
    best_total = table["sharpe_total"].idxmax()
    print(f"\npico de sharpe ativo:   n={best_active}"
          f" (sharpe={table.loc[best_active, 'sharpe']:.3f})")
    print(f"pico de sharpe total:   n={best_total}"
          f" (sharpe={table.loc[best_total, 'sharpe_total']:.3f})")
    _plateau_report(table)


def _plateau_report(table: pd.DataFrame, sharpe_col: str = "sharpe_total") -> None:
    """Reporta o platô (faixa estatisticamente indistinguível do pico), não só o pico.

    Um pico isolado numa curva ruidosa é sobreajuste; um platô é o que se
    escolhe na prática. Usa o erro-padrão assintótico de Lo (2002) para o
    Sharpe ratio ANUALIZADO — não a heurística ingênua 1/√T, que subestima o
    erro em quase 20× e faria qualquer ruído parecer sinal:

        SE(SR_a) = √(252/T) · √(1 + ½·SR_d²),  SR_d = SR_a/√252

    Compara cada ponto contra o pico via z-score no erro conjunto (√(SE_pico²+SE_n²)),
    que é CONSERVADOR a favor de achar diferença — a correlação positiva entre
    variantes vizinhas (mesmo sinal, quase os mesmos nomes) tornaria a diferença
    real ainda menor. Mesmo assim, se nada passa do 1.96, não há pico defensável.
    """
    n_days = table["n_dias"].median() if "n_dias" in table else 2500

    def se_sharpe(sr: float) -> float:
        sr_daily = sr / np.sqrt(252)
        return float(np.sqrt(252.0 / n_days) * np.sqrt(1 + 0.5 * sr_daily ** 2))

    peak_n = table[sharpe_col].idxmax()
    peak_sr = table.loc[peak_n, sharpe_col]
    se_peak = se_sharpe(peak_sr)

    print(f"\nErro-padrão de Lo (2002) para T={n_days:.0f} dias: ~{se_peak:.3f} "
          f"(por comparação, a heurística 1/√T daria {1/np.sqrt(n_days):.3f} — "
          f"~{se_peak / (1/np.sqrt(n_days)):.0f}x menor, e enganosa)")
    print(f"\nz-score de cada n contra o pico (n={peak_n}, {sharpe_col}={peak_sr:.3f}):")
    tied = []
    for n, row in table.iterrows():
        sr = row[sharpe_col]
        se = se_sharpe(sr)
        z = abs(peak_sr - sr) / np.sqrt(se_peak ** 2 + se ** 2)
        flag = "" if z > 1.96 else "  <- empatado com o pico (p>0.05)"
        print(f"  n={n:>3}  {sharpe_col}={sr:.3f}  z={z:.2f}{flag}")
        if z <= 1.96:
            tied.append(n)
    print(f"\nPLATÔ (estatisticamente indistinguível do pico): n em {tied}")
    if len(tied) == len(table):
        print("-> TODA a grade é um único platô: a curva não sustenta um 'N ótimo' "
              "pontual. A escolha deve se apoiar em critérios que variam de forma "
              "monotônica e não-ruidosa (custo, vol), não no pico do Sharpe.")


def _window(cfg, split: str) -> tuple[str, str]:
    if split == "design":
        return cfg.splits.design_start, cfg.splits.design_end
    if split == "holdout":
        return cfg.splits.holdout_start, cfg.splits.holdout_end
    return cfg.splits.design_start, cfg.splits.holdout_end


if __name__ == "__main__":
    main()
