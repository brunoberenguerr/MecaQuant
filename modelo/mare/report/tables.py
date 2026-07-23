"""Tabelas e stat tiles em HTML puro para o dashboard.

Números vivem em text tokens (nunca na cor da série); tabulares alinhados à
direita com `tabular-nums`. Mantido separado das figuras para o dashboard só
compor blocos prontos.
"""

from __future__ import annotations

import pandas as pd

from mare.backtest.metrics import summary


def stat_tiles(returns: pd.Series, benchmark: pd.Series,
               risk_free: pd.Series) -> str:
    """Linha de KPIs da estratégia principal vs SPY."""
    strat = summary(returns, risk_free, label="MARÉ")
    spy = summary(benchmark, risk_free, label="SPY")
    tiles = [
        ("CAGR", f"{strat['cagr']:.1%}", f"SPY {spy['cagr']:.1%}"),
        ("Sharpe", f"{strat['sharpe']:.2f}", f"SPY {spy['sharpe']:.2f}"),
        ("Sortino", f"{strat['sortino']:.2f}", f"SPY {spy['sortino']:.2f}"),
        ("Max Drawdown", f"{strat['max_drawdown']:.1%}", f"SPY {spy['max_drawdown']:.1%}"),
        ("Calmar", f"{strat['calmar']:.2f}", f"SPY {spy['calmar']:.2f}"),
        ("t-stat (NW)", f"{strat['t_stat_nw']:.2f}", "ativo"),
    ]
    cells = "".join(
        f'<div class="tile"><div class="tile-label">{label}</div>'
        f'<div class="tile-value">{value}</div>'
        f'<div class="tile-sub">{sub}</div></div>'
        for label, value, sub in tiles)
    return f'<div class="tiles">{cells}</div>'


def metrics_table(ablation: pd.DataFrame, benchmark: pd.Series,
                  risk_free: pd.Series) -> str:
    """Tabela de ablação de overlays: uma linha por variante + SPY."""
    rows = [summary(ablation[c], risk_free, label=c) for c in ablation.columns]
    rows.append(summary(benchmark, risk_free, label="SPY"))
    frame = pd.DataFrame(rows).set_index("label")
    cols = {"cagr": "CAGR", "vol": "Vol", "sharpe": "Sharpe",
            "sortino": "Sortino", "max_drawdown": "Max DD", "calmar": "Calmar",
            "t_stat_nw": "t (NW)"}
    fmt = {"cagr": "{:.1%}", "vol": "{:.1%}", "max_drawdown": "{:.1%}"}
    return _html_table(frame[list(cols)].rename(columns=cols), fmt, cols_pct=cols)


def trades_table(trades: pd.DataFrame, limit: int = 12) -> str:
    """Amostra das trocas mais recentes com motivo e dias no livro."""
    if trades.empty:
        return "<p>Sem trocas registradas.</p>"
    recent = trades.sort_values("as_of").tail(limit)[
        ["as_of", "symbol", "motivo", "s_score", "dias_no_livro"]].copy()
    recent["as_of"] = pd.to_datetime(recent["as_of"]).dt.strftime("%Y-%m-%d")
    recent.columns = ["Data", "Ativo", "Motivo", "s-score", "Dias"]
    return _html_table(recent.set_index("Data"), {"s-score": "{:.2f}"})


def _html_table(frame: pd.DataFrame, fmt: dict | None = None,
                cols_pct: dict | None = None) -> str:
    fmt = fmt or {}
    header = "".join(f"<th>{c}</th>" for c in ["", *frame.columns])
    body = []
    for idx, row in frame.iterrows():
        cells = [f"<th>{idx}</th>"]
        for col, value in row.items():
            cells.append(f"<td>{_fmt_cell(value)}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (f'<table class="metrics"><thead><tr>{header}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def _fmt_cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
