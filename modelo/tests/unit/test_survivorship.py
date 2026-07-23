"""Bounding do viés de sobrevivência: o castigo tem a direção e a magnitude certas."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mare.validation import survivorship


def _strategy(seed: int = 0, drift: float = 0.0004) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2007-01-01", periods=2500)
    return pd.Series(rng.standard_normal(2500) * 0.01 + drift, index=idx)


def test_haircut_reduces_return():
    """O castigo só pode reduzir o retorno, nunca aumentá-lo."""
    strat = _strategy()
    punished = survivorship.haircut_scenario(strat, hazard_annual=0.05, bad_return=-0.30)
    assert punished.sum() < strat.sum()


def test_worse_scenario_hurts_more():
    """Retorno de deslistagem mais severo tem de doer mais (monotonicidade)."""
    strat = _strategy(1)
    mild = survivorship.haircut_scenario(strat, bad_return=-0.15, seed=0).sum()
    harsh = survivorship.haircut_scenario(strat, bad_return=-1.00, seed=0).sum()
    assert harsh < mild


def test_higher_hazard_hurts_more():
    """Hazard maior (mais nomes deslistando) tem de doer mais."""
    strat = _strategy(2)
    low = survivorship.haircut_scenario(strat, hazard_annual=0.02, seed=0).sum()
    high = survivorship.haircut_scenario(strat, hazard_annual=0.10, seed=0).sum()
    assert high < low


def test_bound_magnitude_is_sane():
    """Com hazard 5% e queda −30%, o arrasto anual deve ficar em torno de 1,5%.

    É o teste que pega o bug antigo (arrasto diário ~252× grande demais). Mede a
    DIFERENÇA punished − base, para isolar o castigo do ruído amostral da série.
    """
    strat = _strategy(3)
    punished = survivorship.haircut_scenario(strat, hazard_annual=0.05, bad_return=-0.30)
    annual_drag = (punished - strat).mean() * 252
    assert -0.03 < annual_drag < -0.005, f"arrasto anual implausível: {annual_drag:.3f}"


def test_bounds_table_is_monotonic_in_cagr():
    """A tabela de cenários deve ter CAGR decrescente com a severidade."""
    strat = _strategy(4)
    bounds = survivorship.bias_bounds(strat)
    cagrs = bounds["cagr"].to_numpy()
    assert np.all(np.diff(cagrs) <= 1e-9), "CAGR deveria cair com o castigo"
