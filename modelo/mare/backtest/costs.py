"""Modelo de custo de transação.

Custo mata mais estratégia do que sinal ruim, e reversão de curto prazo é
especialmente exposta: o mecanismo econômico É prover liquidez, então o spread
que se paga é o mesmo prêmio que se tenta capturar.

Três componentes: comissão (fixa), meio-spread (o que se paga por cruzar) e
impacto (função da participação no volume diário). O impacto usa a raiz quadrada
da participação, forma funcional padrão da literatura de execução — dobrar o
tamanho não dobra o impacto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mare.config import CostScenario

BPS = 1e-4


def one_way_rate(scenario: CostScenario) -> float:
    """Custo fixo de uma ponta, em fração do notional (sem impacto)."""
    return (scenario.commission_bps + scenario.half_spread_bps) * BPS


def impact_rate(participation: pd.Series | float,
                scenario: CostScenario) -> pd.Series | float:
    """Impacto de mercado: coef · √(participação no ADV)."""
    return scenario.impact_coef_bps * BPS * np.sqrt(np.clip(participation, 0.0, None))


def trade_costs(traded: pd.Series, scenario: CostScenario,
                adv: pd.Series | None = None,
                capital: float = 0.0) -> float:
    """Custo total de um rebalanceamento, em fração do patrimônio.

    `traded` é |peso novo − peso anterior| por nome; a soma é o notional
    negociado como fração do patrimônio.
    """
    if traded.empty:
        return 0.0
    fixed = float(traded.sum()) * one_way_rate(scenario)
    if adv is None or capital <= 0:
        return fixed
    notional = traded * capital
    participation = (notional / adv.reindex(traded.index)).replace(
        [np.inf, -np.inf], np.nan).fillna(0.0)
    variable = float((traded * impact_rate(participation, scenario)).sum())
    return fixed + variable


def breakeven_cost_bps(gross_return: float, turnover: float,
                       periods_per_year: int = 52) -> float:
    """Custo one-way (bps) que zera o retorno bruto anualizado.

    "A que nível de custo o alpha morre?" é uma linha de código e o exhibit mais
    persuasivo do relatório: transforma a discussão de custos, que costuma ser
    hand-waving, num número que a banca pode conferir contra a própria mesa.
    """
    annual_turnover = turnover * periods_per_year
    if annual_turnover <= 0:
        return float("inf")
    return gross_return / annual_turnover / BPS
