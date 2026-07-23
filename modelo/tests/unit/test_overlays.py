"""Overlays de risco: escala correta, defasagem e caixa remunerado."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mare.backtest import overlays
from mare.portfolio import voltarget


@pytest.fixture
def strategy() -> pd.Series:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=500)
    return pd.Series(rng.standard_normal(500) * 0.01 + 0.0003, index=idx)


def test_vol_target_reduces_realized_vol(strategy):
    """Mirar 10% de vol tem de aproximar a vol realizada de 10%."""
    scaled = overlays.vol_target_scaled(strategy, target_vol=0.10, max_leverage=1.0)
    realized = scaled.std(ddof=1) * np.sqrt(252)
    base = strategy.std(ddof=1) * np.sqrt(252)
    assert realized <= base + 1e-9
    assert realized < 0.16


def test_vol_target_never_leverages_above_cap(strategy):
    """Com max_leverage=1.0, a escala nunca passa de 1 (long-only honesto)."""
    scale = voltarget.apply_vol_target(strategy, target_vol=0.50, max_leverage=1.0)
    assert (scale <= 1.0 + 1e-12).all()


def test_regime_overlay_cuts_exposure_in_stress():
    """No estado estressado a exposição cai; o caixa liberado rende rf."""
    idx = pd.bdate_range("2020-01-01", periods=5)
    strat = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02], index=idx)
    mult = pd.Series([1.0, 1.0, 0.5, 0.5, 1.0], index=idx)
    rf = pd.Series(0.0004, index=idx)

    scaled = overlays.regime_scaled(strat, mult, risk_free=rf)
    # dia 0: escala defasada = 1 (não há dia anterior) -> retorno cheio
    assert scaled.iloc[0] == pytest.approx(strat.iloc[0])
    # dia 3: escala de ontem (dia 2) = 0.5 -> metade investida, metade em caixa
    expected = 0.5 * strat.iloc[3] + 0.5 * rf.iloc[3]
    assert scaled.iloc[3] == pytest.approx(expected)


def test_overlay_scale_is_lagged(strategy):
    """A escala de hoje usa a informação de ontem, nunca a de hoje."""
    mult = pd.Series(1.0, index=strategy.index)
    mult.iloc[100] = 0.0     # exposição zero num dia
    scaled = overlays.regime_scaled(strategy, mult, risk_free=None)
    # o efeito aparece em t=101 (defasagem), não em t=100
    assert scaled.iloc[100] == pytest.approx(strategy.iloc[100])
    assert scaled.iloc[101] == pytest.approx(0.0)


def test_ablation_has_four_series(strategy):
    mult = pd.Series(1.0, index=strategy.index)
    table = overlays.ablation(strategy, mult, target_vol=0.10)
    assert list(table.columns) == ["base", "vol_target", "regime", "regime+vol_target"]
    # regime neutro (multiplicador 1) deve devolver a série base
    pd.testing.assert_series_equal(table["base"], table["regime"],
                                   check_names=False)


def test_breakeven_cost_is_positive_for_profitable_strategy():
    from mare.backtest.costs import breakeven_cost_bps
    be = breakeven_cost_bps(gross_return=0.05, turnover=0.4, periods_per_year=52)
    assert be > 0
    assert np.isinf(breakeven_cost_bps(0.05, turnover=0.0))
