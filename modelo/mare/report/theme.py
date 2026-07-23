"""Paleta e layout base das figuras do dashboard.

Valores da paleta de referência validada da skill de dataviz (checagem de CVD,
contraste e banda de luminosidade já feita). Centralizados aqui para que todas as
figuras leiam por papel e o tema claro/escuro troque em um só lugar.
"""

from __future__ import annotations

# Superfícies e tinta (modo claro; o dashboard fixa modo claro para impressão).
SURFACE = "#fcfcfb"
PLANE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Slots categóricos na ordem CVD-segura da paleta de referência.
SERIES = {
    "blue": "#2a78d6",
    "green": "#008300",
    "magenta": "#e87ba4",
    "yellow": "#eda100",
    "aqua": "#1baf7a",
    "orange": "#eb6834",
    "violet": "#4a3aa7",
    "red": "#e34948",
}

# Papéis semânticos das séries do dashboard.
ROLE = {
    "base": SERIES["blue"],
    "regime": SERIES["aqua"],
    "vol_target": SERIES["violet"],
    "regime+vol_target": SERIES["orange"],
    "SPY": MUTED,
    "drawdown": SERIES["red"],
    "stress": SERIES["red"],
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def base_layout(title: str = "", height: int = 360) -> dict:
    """Layout Plotly consistente: fundo da superfície, grade recessiva, tipografia UI."""
    return {
        "title": {"text": title, "font": {"size": 15, "color": INK, "family": FONT},
                  "x": 0, "xanchor": "left"},
        "height": height,
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "font": {"family": FONT, "color": INK_2, "size": 12},
        "margin": {"l": 56, "r": 20, "t": 44, "b": 40},
        "xaxis": {"gridcolor": GRID, "linecolor": AXIS, "zeroline": False,
                  "tickfont": {"color": MUTED}},
        "yaxis": {"gridcolor": GRID, "linecolor": AXIS, "zeroline": False,
                  "tickfont": {"color": MUTED}},
        "legend": {"orientation": "h", "y": -0.18, "x": 0, "font": {"color": INK_2}},
        "hovermode": "x unified",
    }
