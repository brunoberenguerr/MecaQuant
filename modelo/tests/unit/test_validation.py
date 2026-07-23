"""Validação estatística: DSR, placebo e atribuição.

Estes números vão para o relatório; se estiverem errados, a honestidade que eles
deveriam demonstrar vira o contrário. Daí os testes com casos de referência.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mare.validation import dsr, placebo


def _returns(sharpe_annual: float, n: int = 2000, seed: int = 0) -> pd.Series:
    """Série gaussiana com Sharpe anualizado alvo."""
    rng = np.random.default_rng(seed)
    daily_sharpe = sharpe_annual / np.sqrt(252)
    x = rng.standard_normal(n) * 0.01
    x = x - x.mean() + daily_sharpe * x.std()
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(x, index=idx)


def test_expected_max_sharpe_grows_with_trials():
    """Testar mais configurações eleva o Sharpe máximo esperado sob o nulo."""
    few = dsr.expected_max_sharpe(n_trials=5, sr_variance=1 / 252)
    many = dsr.expected_max_sharpe(n_trials=500, sr_variance=1 / 252)
    assert many > few > 0


def test_deflated_sharpe_penalizes_many_trials():
    """O MESMO backtest tem DSR menor quando muitas configurações foram testadas."""
    returns = _returns(1.0, seed=1)
    lenient = dsr.deflated_sharpe_ratio(returns, n_trials=1)["dsr"]
    strict = dsr.deflated_sharpe_ratio(returns, n_trials=300)["dsr"]
    assert strict < lenient
    assert 0 <= strict <= 1


def test_probabilistic_sharpe_high_for_strong_track_record():
    """Sharpe alto e amostra longa -> PSR próximo de 1."""
    strong = _returns(1.5, n=3000, seed=2)
    assert dsr.probabilistic_sharpe_ratio(strong, benchmark_sr=0.0) > 0.95


def test_probabilistic_sharpe_low_for_zero_edge():
    """Sem edge, PSR contra zero fica em torno de 0.5, não perto de 1."""
    flat = _returns(0.0, n=2000, seed=3)
    assert 0.2 < dsr.probabilistic_sharpe_ratio(flat, benchmark_sr=0.0) < 0.8


def test_min_track_record_length_is_finite_for_edge():
    strong = _returns(1.0, n=2000, seed=4)
    mtrl = dsr.min_track_record_length(strong, target_confidence=0.95)
    assert 0 < mtrl < 2000


def test_block_bootstrap_centers_near_observed():
    """A média do bootstrap deve ficar perto do Sharpe da série reamostrada."""
    returns = _returns(0.8, n=1500, seed=5)
    from mare.backtest.metrics import sharpe

    boot = placebo.block_bootstrap_sharpes(returns, n_draws=300, block=21, seed=1)
    assert abs(boot.mean() - sharpe(returns)) < 0.5


def test_placebo_percentile_detects_strong_signal():
    """Um Sharpe muito acima do nulo cai num percentil alto."""
    null = pd.Series(np.random.default_rng(6).normal(0.0, 1.0, 1000))
    assert placebo.percentile_of(2.5, null) > 0.98
    assert placebo.percentile_of(0.0, null) == pytest.approx(0.5, abs=0.05)


def test_summarize_reports_pvalue():
    null = pd.Series(np.random.default_rng(7).normal(0.0, 1.0, 500))
    out = placebo.summarize(2.0, null, label="teste")
    assert out["p_valor"] < 0.05
    assert out["percentil"] > 0.95
