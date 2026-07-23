"""Monta o dashboard HTML autocontido a partir dos artefatos de um backtest.

Autocontido de verdade: a biblioteca Plotly é embutida uma única vez, então o
arquivo abre offline, sem servidor e sem CDN — requisito para um entregável que a
banca vai abrir na própria máquina.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from mare.report import figures, tables, theme

_PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def _fig_div(fig: go.Figure, include_js: bool) -> str:
    return pio.to_html(fig, include_plotlyjs=("inline" if include_js else False),
                       full_html=False, config=_PLOTLY_CONFIG,
                       default_height="100%")


def build_dashboard(artifacts, cfg, *, split: str, attribution=None,
                    placebo_null=None, breakeven=None, out_path: Path) -> Path:
    """Escreve o dashboard e devolve o caminho do arquivo."""
    ab = artifacts.ablation
    bench = artifacts.benchmark
    rf = artifacts.risk_free

    blocks: list[str] = []
    blocks.append(tables.stat_tiles(ab.iloc[:, 0], bench, rf))

    first = True

    def section(title: str, body: str, note: str = "") -> None:
        nonlocal first
        note_html = f'<p class="note">{note}</p>' if note else ""
        blocks.append(f'<section><h2>{title}</h2>{note_html}'
                      f'<div class="card">{body}</div></section>')

    section("Patrimônio acumulado",
            _fig_div(figures.equity_curve(ab, bench, artifacts.regime_states), first),
            "Faixas sombreadas: regime estressado detectado pelo jump model. "
            "Escala logarítmica.")
    first = False

    section("Ablação de overlays de risco (Pilar 2)",
            tables.metrics_table(ab, bench, rf),
            "Efeito isolado de cada camada de risco. O vol-target corta o "
            "drawdown mantendo — ou elevando — o Sharpe.")

    section("Drawdown", _fig_div(figures.drawdown_chart(ab, bench), False))
    section("Sharpe móvel (12 meses)", _fig_div(figures.rolling_sharpe(ab), False),
            "Estabilidade do edge ao longo do tempo, não só na média.")

    if attribution is not None and not attribution.empty:
        section("Atribuição de fatores (série ativa)",
                _fig_div(figures.attribution_bars(attribution), False),
                "Regressão da série ativa (estratégia − SPY) contra Fama-French "
                "5 + momentum. Barras em azul: |t| > 2.")

    if placebo_null is not None and len(placebo_null.dropna()):
        from mare.backtest.metrics import sharpe
        section("Teste de placebo (seleção aleatória)",
                _fig_div(figures.placebo_hist(placebo_null, sharpe(ab.iloc[:, 0]),
                                              "Sharpe: estratégia vs carteiras aleatórias"),
                         False),
                "Distribuição do Sharpe de carteiras sorteadas do mesmo universo "
                "elegível, mesmo N e cadência. Isola a skill de seleção.")

    if breakeven is not None:
        section("Custo de break-even",
                _fig_div(figures.cost_scenarios(*breakeven), False),
                "A que nível de custo de transação o alpha desaparece.")

    trades = artifacts.diagnostics.trades_frame()
    section("Trocas recentes", tables.trades_table(trades),
            "Cada saída registra o motivo derivado do modelo.")

    html = _page(cfg.meta.name, split, "".join(blocks))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _page(name: str, split: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{name} — Dashboard de Backtest ({split})</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
<header><h1>{name}</h1>
<p class="tag">Mean-reversion Adaptive Residual Engine · backtest {split} · reversão residual no S&P 500</p>
</header>
{body}
<footer>Gerado pelo pipeline MARÉ. Dados: yfinance (preços), Ken French (fatores),
FRED (taxa livre de risco). Backtest com lag de execução de 1 dia e custos de
transação. Todos os números são reproduzíveis via <code>scripts/</code>.</footer>
</div></body></html>"""


_CSS = f"""
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: {theme.PLANE}; color: {theme.INK};
  font-family: {theme.FONT}; line-height: 1.5; }}
.wrap {{ max-width: 1080px; margin: 0 auto; padding: 32px 24px 64px; }}
header h1 {{ margin: 0; font-size: 30px; letter-spacing: -0.02em; }}
.tag {{ color: {theme.INK_2}; margin: 4px 0 24px; }}
h2 {{ font-size: 17px; margin: 32px 0 8px; }}
.note {{ color: {theme.INK_2}; font-size: 13px; margin: 0 0 10px; max-width: 68ch; }}
.card {{ background: {theme.SURFACE}; border: 1px solid rgba(11,11,11,0.10);
  border-radius: 10px; padding: 12px; overflow-x: auto; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px; margin-bottom: 8px; }}
.tile {{ background: {theme.SURFACE}; border: 1px solid rgba(11,11,11,0.10);
  border-radius: 10px; padding: 14px 16px; }}
.tile-label {{ color: {theme.MUTED}; font-size: 12px; }}
.tile-value {{ font-size: 26px; font-weight: 600; letter-spacing: -0.01em;
  margin: 2px 0; }}
.tile-sub {{ color: {theme.INK_2}; font-size: 12px; }}
table.metrics {{ border-collapse: collapse; width: 100%; font-size: 13px;
  font-variant-numeric: tabular-nums; }}
table.metrics th, table.metrics td {{ padding: 7px 12px; text-align: right;
  border-bottom: 1px solid {theme.GRID}; }}
table.metrics thead th {{ color: {theme.MUTED}; font-weight: 500;
  border-bottom: 1px solid {theme.AXIS}; }}
table.metrics tbody th {{ text-align: left; color: {theme.INK}; font-weight: 600; }}
footer {{ color: {theme.MUTED}; font-size: 12px; margin-top: 40px;
  border-top: 1px solid {theme.GRID}; padding-top: 16px; max-width: 72ch; }}
code {{ background: {theme.GRID}; padding: 1px 5px; border-radius: 4px; }}
"""
