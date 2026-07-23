"""Recuperação de parâmetros do OU e quantificação dos vieses de estimação.

Estes testes não checam só "o código roda": eles medem o viés de Kendall e
mostram que a correção funciona. O número que sai daqui vai para o relatório.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mare.signal.ou import (OUFit, fit_ou, kendall_correct, shrink_ar_coefficient,
                            simulate_ou)


def _panel(kappa: float, sigma: float, n_obs: int, n_series: int,
           seed: int) -> pd.DataFrame:
    paths = simulate_ou(kappa=kappa, theta=0.0, sigma=sigma, x0=0.0,
                        n_steps=n_obs - 1, n_paths=n_series, seed=seed)
    return pd.DataFrame(paths.T, columns=[f"S{i}" for i in range(n_series)])


def test_recovers_kappa_on_long_sample():
    """Com amostra longa o estimador tem de convergir para a verdade."""
    true_kappa = 0.05
    panel = _panel(true_kappa, sigma=0.02, n_obs=4000, n_series=20, seed=1)
    fit = fit_ou(panel, kendall=False, shrinkage=False)
    assert fit.kappa.median() == pytest.approx(true_kappa, rel=0.10)


def test_recovers_sigma_eq():
    """σ_eq = σ/√(2κ) é o que normaliza o s-score; errar aqui desloca tudo."""
    true_kappa, true_sigma = 0.04, 0.02
    expected = true_sigma / np.sqrt(2 * true_kappa)
    panel = _panel(true_kappa, true_sigma, n_obs=4000, n_series=20, seed=2)
    fit = fit_ou(panel, kendall=False, shrinkage=False)
    assert fit.sigma_eq.median() == pytest.approx(expected, rel=0.10)


def test_kendall_bias_inflates_kappa_on_short_windows():
    """O viés existe, tem a direção perigosa, e a correção o reduz.

    Direção perigosa: κ superestimado -> σ_eq subestimado -> |s-score| inflado
    -> a estratégia opera demais, e justamente onde o estimador é mais ruidoso.
    """
    true_kappa = 0.035          # meia-vida ~20 dias
    panel = _panel(true_kappa, sigma=0.02, n_obs=60, n_series=400, seed=3)

    naive = fit_ou(panel, kendall=False, shrinkage=False).kappa.median()
    fixed = fit_ou(panel, kendall=True, shrinkage=False).kappa.median()

    assert naive > true_kappa, "esperado viés para cima em janela curta"
    assert abs(fixed - true_kappa) < abs(naive - true_kappa)


def test_kendall_correction_is_monotonic():
    corrected = [kendall_correct(b, 60) for b in (0.80, 0.90, 0.95)]
    assert all(np.diff(corrected) > 0)
    assert all(c > b for c, b in zip(corrected, (0.80, 0.90, 0.95)))


def test_shrinkage_pulls_toward_pooled_mean():
    """Estimativas ruidosas devem encolher para o consenso cross-sectional."""
    noisy = pd.Series(np.random.default_rng(4).normal(0.9, 0.25, 300))
    shrunk = shrink_ar_coefficient(noisy, n_obs=60)
    assert shrunk.std() < noisy.std()
    assert shrunk.mean() == pytest.approx(noisy.mean(), rel=0.05)


def test_shrinkage_reduces_dispersion_of_half_life():
    """Efeito prático: menos meias-vidas absurdas entrando na carteira."""
    panel = _panel(0.035, sigma=0.02, n_obs=60, n_series=400, seed=5)
    loose = fit_ou(panel, kendall=True, shrinkage=False).half_life.dropna()
    tight = fit_ou(panel, kendall=True, shrinkage=True).half_life.dropna()
    assert tight.std() < loose.std()


def test_s_score_is_standardized():
    """s = (X−θ)/σ_eq deve ter escala ~1 quando o processo está no equilíbrio."""
    panel = _panel(0.04, sigma=0.02, n_obs=2000, n_series=200, seed=6)
    fit = fit_ou(panel, kendall=True, shrinkage=True)
    scores = fit.s_score(panel.iloc[-1]).dropna()
    assert 0.7 < scores.std() < 1.4
    assert abs(scores.mean()) < 0.3


def test_random_walk_yields_no_usable_reversion():
    """Passeio aleatório não pode produzir meia-vida curta e confiável.

    É o cenário em que o gate de meia-vida precisa proteger a carteira.
    """
    rng = np.random.default_rng(7)
    walk = pd.DataFrame(rng.standard_normal((60, 200)).cumsum(axis=0) * 0.02,
                        columns=[f"S{i}" for i in range(200)])
    fit = fit_ou(walk, kendall=True, shrinkage=True)
    usable = fit.half_life.between(10, 45).mean()
    assert usable < 0.5, "gate de meia-vida deixaria passar ruído demais"


def test_fit_returns_expected_shape():
    panel = _panel(0.04, 0.02, n_obs=60, n_series=5, seed=8)
    fit = fit_ou(panel)
    assert isinstance(fit, OUFit)
    for series in (fit.kappa, fit.theta, fit.sigma_eq, fit.half_life):
        assert list(series.index) == list(panel.columns)
