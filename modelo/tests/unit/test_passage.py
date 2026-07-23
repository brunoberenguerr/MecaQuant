"""Valida a quadratura de primeira passagem contra simulação e casos analíticos.

Fórmula fechada sem verificação numérica é a forma clássica de contaminar um
backtest inteiro sem perceber.
"""

from __future__ import annotations

import numpy as np
import pytest

from mare.signal.passage import (expected_exit_time, hit_probability,
                                 monte_carlo_check, trade_economics)


def test_probability_is_monotonic_in_start():
    """Começar mais perto do alvo só pode aumentar a chance de atingi-lo."""
    probs = [hit_probability(z0, -3.0, -0.5) for z0 in (-2.5, -2.0, -1.5, -1.0)]
    assert all(np.diff(probs) > 0)
    assert all(0 < p < 1 for p in probs)


def test_probability_at_barriers():
    """Nos limites, a probabilidade tende a 0 e 1."""
    assert hit_probability(-2.999, -3.0, -0.5) < 0.02
    assert hit_probability(-0.501, -3.0, -0.5) > 0.98


def test_symmetric_case_matches_brownian_intuition():
    """Barreiras simétricas em torno de zero dão probabilidade 1/2 por simetria.

    O OU normalizado tem drift ímpar em torno de zero, então partir do centro
    entre barreiras simétricas é um coin flip exato.
    """
    assert hit_probability(0.0, -2.0, 2.0) == pytest.approx(0.5, abs=1e-6)


def test_exit_time_scales_inversely_with_kappa():
    """E[T] em dias é o tempo normalizado dividido por κ."""
    fast = expected_exit_time(-1.25, -3.0, -0.5, kappa=0.10)
    slow = expected_exit_time(-1.25, -3.0, -0.5, kappa=0.05)
    assert slow == pytest.approx(2 * fast, rel=1e-9)


@pytest.mark.parametrize("z0", [-2.0, -1.25, -0.8])
def test_quadrature_matches_monte_carlo(z0):
    """O teste que importa: a teoria bate com a simulação do próprio processo."""
    z_stop, z_target, kappa = -3.0, -0.5, 0.05
    analytic_p = hit_probability(z0, z_stop, z_target)
    analytic_t = expected_exit_time(z0, z_stop, z_target, kappa)

    mc = monte_carlo_check(z0, z_stop, z_target, kappa, n_paths=60_000,
                           dt=0.02, seed=7)

    assert mc["unresolved"] < 0.01, "caminhos demais sem resolver: aumente max_steps"
    # Com ponte browniana e atribuição no meio do passo, o desvio remanescente é
    # erro de amostragem (~0.2pp com 60k caminhos), não viés de discretização.
    assert analytic_p == pytest.approx(mc["prob"], abs=0.004)
    assert analytic_t == pytest.approx(mc["expected_days"], rel=0.01)


def test_quadrature_matches_closed_form_erfi():
    """Checagem independente da função de escala: S(x) = √(π/2)·erfi(x/√2).

    Confirma a quadratura sem passar por simulação — se os dois caminhos
    (quadratura numérica e forma fechada) concordam, o erro teria de estar na
    própria dedução, não na implementação.
    """
    from scipy.special import erfi

    def closed(z0, a, b):
        f = lambda x: erfi(x / np.sqrt(2))
        return (f(z0) - f(a)) / (f(b) - f(a))

    for z0 in (-2.0, -1.25, -0.8):
        assert hit_probability(z0, -3.0, -0.5) == pytest.approx(
            closed(z0, -3.0, -0.5), abs=1e-6)


def test_discrete_monitoring_bias_is_real():
    """Sem a correção de ponte, o MC erra na direção conhecida.

    Monitorar só nos instantes amostrados perde cruzamentos, o que superestima
    o tempo de saída. Documentar o viés protege contra alguém "simplificar" a
    correção mais tarde.
    """
    args = dict(z0=-1.25, z_stop=-3.0, z_target=-0.5, kappa=0.05,
                n_paths=40_000, dt=0.02, seed=7)
    naive = monte_carlo_check(bridge=False, **args)
    corrected = monte_carlo_check(bridge=True, **args)
    exact = expected_exit_time(-1.25, -3.0, -0.5, 0.05)
    assert naive["expected_days"] > corrected["expected_days"] > exact * 0.98
    assert abs(naive["expected_days"] - exact) > 5 * abs(corrected["expected_days"] - exact)


def test_mu_s_prefers_faster_reversion():
    """Mesmo s-score, reversão mais rápida: retorno por unidade de tempo maior.

    É a razão de ranquear por mu_s e não por s-score puro.
    """
    common = dict(z0=-1.5, z_stop=-3.0, z_target=-0.5, sigma_eq=0.04)
    fast = trade_economics(kappa=0.14, **common)   # meia-vida ~5 dias
    slow = trade_economics(kappa=0.02, **common)   # meia-vida ~35 dias
    assert fast["expected_return"] == pytest.approx(slow["expected_return"], rel=1e-9)
    assert fast["mu_s"] > slow["mu_s"]


def test_invalid_ordering_returns_nan():
    """Barreiras fora de ordem não podem devolver um número plausível."""
    assert np.isnan(hit_probability(-4.0, -3.0, -0.5))
    assert np.isnan(expected_exit_time(-1.25, -0.5, -3.0, kappa=0.05))
