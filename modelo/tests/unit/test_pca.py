"""Bloco fatorial: recuperação de k e robustez do critério de Marchenko–Pastur."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mare.features.vol import weighted_corr
from mare.signal import pca


def _synthetic(n_assets: int, n_obs: int, k_true: int, market_share: float,
               seed: int) -> pd.DataFrame:
    """Painel com um modo de mercado dominante, k_true−1 fatores e ruído."""
    rng = np.random.default_rng(seed)
    market = rng.standard_normal((n_obs, 1))
    factors = rng.standard_normal((n_obs, k_true - 1))
    loads_m = rng.uniform(0.7, 1.3, (1, n_assets))
    loads_f = rng.standard_normal((k_true - 1, n_assets)) * 0.35
    noise = rng.standard_normal((n_obs, n_assets))
    signal = market @ loads_m * market_share + factors @ loads_f
    data = signal + noise * (1 - market_share) * 1.4
    return pd.DataFrame(data * 0.01,
                        columns=[f"A{i}" for i in range(n_assets)])


def test_mp_edge_matches_formula():
    assert pca.mp_upper_edge(100, 400, 1.0) == pytest.approx((1 + 0.5) ** 2)
    assert pca.mp_upper_edge(100, 400, 4.0) == pytest.approx(4 * (1.5 ** 2))


def test_eigenvalues_are_sorted_descending():
    panel = _synthetic(80, 400, k_true=5, market_share=0.5, seed=1)
    corr, _ = weighted_corr(panel, halflife=None)
    vals, _ = pca.eigen_decomposition(corr)
    assert np.all(np.diff(vals) <= 1e-9)
    assert vals.sum() == pytest.approx(corr.shape[0], rel=1e-6)


def test_recovers_planted_factor_count():
    """Com estrutura conhecida, o critério tem de achar aproximadamente k."""
    panel = _synthetic(200, 500, k_true=6, market_share=0.6, seed=2)
    corr, _ = weighted_corr(panel, halflife=None)
    vals, _ = pca.eigen_decomposition(corr)
    k = pca.select_k(vals, 200, 500, k_min=1, k_max=50)
    assert 4 <= k <= 10, f"k={k} fora do esperado para 6 fatores plantados"


def test_pure_noise_gives_only_market_mode():
    """Sem estrutura, o critério não pode inventar fatores."""
    rng = np.random.default_rng(3)
    panel = pd.DataFrame(rng.standard_normal((600, 150)) * 0.01,
                         columns=[f"A{i}" for i in range(150)])
    corr, _ = weighted_corr(panel, halflife=None)
    vals, _ = pca.eigen_decomposition(corr)
    assert pca.select_k(vals, 150, 600, k_min=1, k_max=40) <= 3


def test_trace_preserving_variant_is_degenerate():
    """Documenta por que a formulação alternativa foi rejeitada.

    Com um modo de mercado dominante, iterar σ² = média dos autovalores fora do
    sinal entra em espiral: cada passo derruba a borda e admite mais fatores.
    Este teste trava a decisão para que ninguém a "simplifique" de volta.
    """
    panel = _synthetic(300, 500, k_true=6, market_share=0.7, seed=4)
    corr, _ = weighted_corr(panel, halflife=None)
    vals, _ = pca.eigen_decomposition(corr)
    n, t = 300, 500

    k, sigma2 = 0, 1.0
    for _ in range(50):
        sigma2 = vals[k:].sum() / (n - k)
        k_new = int((vals > pca.mp_upper_edge(n, t, sigma2)).sum())
        if k_new == k:
            break
        k = k_new

    conditioned = pca.select_k(vals, n, t, k_min=1, k_max=100)
    # Em dado sintético a degeneração é mais branda (ruído quase iid). No painel
    # real do S&P 500 a razão medida foi de 5x a 8x: k = 97–189 contra 19–23.
    assert k > 2 * conditioned, "variante degenerada deveria superestimar muito"


def test_residual_is_invariant_to_eigenvector_sign():
    """O resíduo depende só do SPAN dos regressores, não do sinal dos autovetores.

    Importa para diagnóstico: trocar o sinal de um autovetor muda os gráficos e
    a interpretação, mas não pode mudar uma única posição da carteira.
    """
    panel = _synthetic(60, 300, k_true=4, market_share=0.5, seed=5)
    corr, vols = weighted_corr(panel, halflife=None)
    _, vecs = pca.eigen_decomposition(corr)

    w1 = pca.eigenportfolio_weights(vecs, vols, k=4)
    w2 = pca.eigenportfolio_weights(vecs * np.array([1, -1, 1, -1] + [1] * 56), vols, k=4)

    f1 = pca.factor_returns(panel, w1).to_numpy()
    f2 = pca.factor_returns(panel, w2).to_numpy()
    y = panel.iloc[:, 0].to_numpy()

    def residual(f):
        design = np.column_stack([np.ones(len(f)), f])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        return y - design @ coef

    assert np.allclose(residual(f1), residual(f2), atol=1e-10)


def test_hysteresis_smooths_jumps():
    """Um salto isolado não pode arrastar k: a mediana da janela o absorve."""
    # mediana de [15, 15, 15, 16, 30] = 15 — o salto para 30 é descartado
    assert pca.apply_hysteresis([15, 15, 15, 16], 30, window=5) == 15
    # mas um movimento sustentado passa
    assert pca.apply_hysteresis([28, 29, 30, 30], 30, window=5) == 30
    assert pca.apply_hysteresis([], 20, window=1) == 20
