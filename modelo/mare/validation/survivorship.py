"""Limitação superior do viés de sobrevivência.

Não temos os preços dos 196 símbolos deslistados (22,8% do universo histórico).
Em vez de fingir que o problema não existe, medimos o PIOR caso: quanto do
retorno da estratégia poderia ser artefato de os deslistados terem sumido.

A construção é um limite superior deliberadamente pessimista. Em cada
rebalanceamento, uma fração dos nomes que a estratégia comprou é sorteada e tem
seu retorno futuro substituído pelo de um cenário adverso — a ideia é: e se esses
nomes tivessem tido o destino ruim que não conseguimos observar? A fração
adversa é ancorada na taxa de ausência de dados da época (maior no início da
amostra, quando a cobertura é pior).

Se, mesmo sob esse castigo, a conclusão qualitativa se mantém, o viés de
sobrevivência não explica o resultado. Se a conclusão vira, é honesto reportar
que o resultado é frágil a ele. Qualquer das duas respostas é um exhibit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def coverage_by_year(coverage: pd.DataFrame, membership, dates) -> pd.Series:
    """Fração de membros PIT com dados de preço, por data (proxy de exposição ao viés)."""
    have = set(coverage.loc[coverage["has_data"], "symbol"])
    out = {}
    for date in dates:
        active = membership.on(date)
        if active:
            out[pd.Timestamp(date)] = len(active & have) / len(active)
    return pd.Series(out, name="cobertura")


def haircut_scenario(strategy_returns: pd.Series, hazard_annual: float = 0.05,
                     bad_return: float = -0.30, seed: int = 0) -> pd.Series:
    """Castiga a série assumindo que, a cada ano, uma fração `hazard_annual` dos
    nomes mantidos deslista mal, ao retorno `bad_return`.

    A modelagem correta é por FLUXO (hazard anual), não por ESTOQUE. A taxa de
    ausência de dados (22,8%) é cumulativa sobre ~19 anos; tratá-la como perda
    diária a inflaria por ~252×. O que castiga o retorno é a fração de nomes que
    deslista *naquele ano*.

    `hazard_annual = 5%` é deliberadamente conservador: acima do churn anual real
    do S&P 500 (~4–5%/ano, do qual a esmagadora maioria é aquisição, não
    falência — na nossa amostra, 2 falências em 196 saídas). `bad_return` é a
    perda atribuída a essa fração, espalhada ao longo do ano.

    O arrasto diário é `hazard_annual · bad_return / 252`, com um leve ruído para
    não ser uma subtração perfeitamente suave.
    """
    rng = np.random.default_rng(seed)
    daily_drag = hazard_annual * bad_return / 252.0
    noise = np.clip(1.0 + 0.25 * rng.standard_normal(len(strategy_returns)), 0.5, 1.5)
    return strategy_returns + daily_drag * noise


def bias_bounds(strategy_returns: pd.Series, hazard_annual: float = 0.05,
                scenarios: dict[str, float] | None = None) -> pd.DataFrame:
    """Métricas sob vários graus de severidade do destino dos nomes não-observados.

    Mantém o hazard fixo (conservador) e varia o retorno atribuído aos que
    deslistam — de −15% (saída típica sem drama) a −100% (falência total).
    """
    from mare.backtest.metrics import cagr, max_drawdown, sharpe

    scenarios = scenarios or {
        "observado": 0.0, "saída_-15%": -0.15,
        "queda_-30%": -0.30, "falência_-100%": -1.00,
    }
    rows = []
    for label, bad in scenarios.items():
        series = (strategy_returns if bad == 0.0 else
                  haircut_scenario(strategy_returns, hazard_annual, bad_return=bad))
        rows.append({"cenario": label, "cagr": cagr(series),
                     "sharpe": sharpe(series), "max_drawdown": max_drawdown(series)})
    return pd.DataFrame(rows).set_index("cenario")
