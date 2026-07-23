"""Figuras Plotly do dashboard.

Cada função devolve uma figura (dict Plotly) e é curta e independente, para poder
ser testada e recomposta. O estilo vem de `theme.py`; nada de cor hardcoded aqui.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mare.backtest.metrics import drawdown_series
from mare.report import theme


def _equity(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0.0)).cumprod()


def equity_curve(ablation: pd.DataFrame, benchmark: pd.Series,
                 regime: pd.Series | None = None) -> go.Figure:
    """Curvas de patrimônio das variantes + SPY, com sombreado de regime."""
    fig = go.Figure()
    if regime is not None:
        _add_regime_shading(fig, regime, ablation.index)

    for col in ablation.columns:
        fig.add_trace(go.Scatter(
            x=ablation.index, y=_equity(ablation[col]), name=col,
            line={"width": 2, "color": theme.ROLE.get(col, theme.MUTED)},
            hovertemplate="%{y:.2f}<extra>" + col + "</extra>"))

    fig.add_trace(go.Scatter(
        x=benchmark.index, y=_equity(benchmark), name="SPY",
        line={"width": 2, "color": theme.ROLE["SPY"], "dash": "dot"},
        hovertemplate="%{y:.2f}<extra>SPY</extra>"))

    layout = theme.base_layout("Patrimônio acumulado (base 1,0)", height=420)
    layout["yaxis"]["type"] = "log"
    fig.update_layout(**layout)
    return fig


def _add_regime_shading(fig: go.Figure, regime: pd.Series,
                        index: pd.DatetimeIndex) -> None:
    """Faixas verticais nos períodos de regime estressado."""
    states = regime.reindex(index).fillna(0)
    stressed = states >= states.max()
    starts = index[stressed & ~stressed.shift(1, fill_value=False)]
    ends = index[stressed & ~stressed.shift(-1, fill_value=False)]
    for start, end in zip(starts, ends):
        fig.add_vrect(x0=start, x1=end, fillcolor=theme.ROLE["stress"],
                      opacity=0.07, line_width=0, layer="below")


def drawdown_chart(ablation: pd.DataFrame, benchmark: pd.Series) -> go.Figure:
    """Underwater plot: drawdown da variante principal vs SPY."""
    fig = go.Figure()
    main = "regime+vol_target" if "regime+vol_target" in ablation else ablation.columns[0]
    for series, name, color in [
        (ablation[main], main, theme.ROLE.get(main, theme.SERIES["blue"])),
        (benchmark, "SPY", theme.ROLE["SPY"]),
    ]:
        dd = drawdown_series(series) * 100
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd, name=name, fill="tozeroy",
            line={"width": 1.5, "color": color},
            fillcolor=_rgba(color, 0.10),
            hovertemplate="%{y:.1f}%<extra>" + name + "</extra>"))
    layout = theme.base_layout("Drawdown (%)", height=300)
    fig.update_layout(**layout)
    return fig


def rolling_sharpe(ablation: pd.DataFrame, window: int = 252) -> go.Figure:
    """Sharpe móvel de 1 ano das variantes — mostra estabilidade do edge no tempo."""
    fig = go.Figure()
    for col in ablation.columns:
        r = ablation[col]
        roll = (r.rolling(window).mean() / r.rolling(window).std()) * np.sqrt(252)
        fig.add_trace(go.Scatter(
            x=roll.index, y=roll, name=col,
            line={"width": 2, "color": theme.ROLE.get(col, theme.MUTED)},
            hovertemplate="%{y:.2f}<extra>" + col + "</extra>"))
    fig.add_hline(y=0, line_width=1, line_color=theme.AXIS)
    fig.update_layout(**theme.base_layout(
        f"Sharpe móvel ({window // 21} meses)", height=300))
    return fig


def attribution_bars(regression: pd.DataFrame) -> go.Figure:
    """Betas de fator com barra de alpha destacada."""
    factors = [i for i in regression.index if i != "alpha"]
    colors = [theme.SERIES["blue"] if abs(regression.loc[f, "t_stat"]) > 2
              else theme.MUTED for f in factors]
    fig = go.Figure(go.Bar(
        x=factors, y=[regression.loc[f, "coef"] for f in factors],
        marker_color=colors, width=0.6,
        hovertemplate="beta=%{y:.3f}<extra></extra>"))
    layout = theme.base_layout("Exposição a fatores (FF5 + MOM, série ativa)", height=320)
    layout["hovermode"] = "closest"
    fig.update_layout(**layout)
    fig.add_hline(y=0, line_width=1, line_color=theme.AXIS)
    return fig


def placebo_hist(null_sharpes: pd.Series, observed: float,
                 title: str) -> go.Figure:
    """Histograma da distribuição nula com o Sharpe observado marcado."""
    clean = null_sharpes.dropna()
    fig = go.Figure(go.Histogram(
        x=clean, nbinsx=40, marker_color=theme.MUTED, opacity=0.7,
        name="nulo", hovertemplate="Sharpe %{x:.2f}<extra></extra>"))
    fig.add_vline(x=observed, line_width=2.5, line_color=theme.SERIES["blue"],
                  annotation_text=f"observado {observed:.2f}",
                  annotation_font_color=theme.INK)
    layout = theme.base_layout(title, height=300)
    layout["hovermode"] = "closest"
    fig.update_layout(**layout)
    return fig


def cost_scenarios(gross_cagr: float, turnover: float, breakeven_bps: float) -> go.Figure:
    """Retorno líquido em função do custo one-way, com o break-even marcado."""
    costs = np.linspace(0, breakeven_bps * 1.4, 60)
    net = gross_cagr - costs * 1e-4 * turnover * 52
    fig = go.Figure(go.Scatter(
        x=costs, y=net * 100, line={"width": 2, "color": theme.SERIES["blue"]},
        hovertemplate="custo %{x:.1f}bps → %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=0, line_width=1, line_color=theme.AXIS)
    fig.add_vline(x=breakeven_bps, line_width=2, line_dash="dash",
                  line_color=theme.SERIES["red"],
                  annotation_text=f"break-even {breakeven_bps:.0f} bps")
    layout = theme.base_layout("Retorno líquido anual vs custo de transação", height=300)
    layout["hovermode"] = "closest"
    fig.update_layout(**layout)
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
