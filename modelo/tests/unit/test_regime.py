"""Jump model: recuperação de regimes, persistência e ausência de look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mare.portfolio import regime


def _two_regime_series(seed: int = 0) -> pd.Series:
    """Série sintética com blocos alternados de baixa e alta volatilidade."""
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(8):
        blocks.append(rng.standard_normal(120) * 0.006)   # calmo
        blocks.append(rng.standard_normal(60) * 0.028)    # estressado
    data = np.concatenate(blocks)
    idx = pd.bdate_range("2005-01-01", periods=len(data))
    return pd.Series(data, index=idx)


def test_recovers_high_vol_regime():
    """O estado rotulado 1 deve concentrar a volatilidade alta."""
    series = _two_regime_series()
    feats = regime.regime_features(series, halflives=(10, 20, 40), return_halflife=40)
    std = regime.expanding_standardize(feats, min_obs=120)
    states, _ = regime.fit_jump_model(std, jump_penalty=1.0)
    states = pd.Series(states, index=std.index)

    realized = series.reindex(states.index)
    vol_high = realized[states == 1].std()
    vol_low = realized[states == 0].std()
    assert vol_high > 1.5 * vol_low


def test_jump_penalty_increases_persistence():
    """λ maior tem de reduzir o número de trocas de estado (é o ponto do modelo)."""
    series = _two_regime_series(seed=1)
    feats = regime.regime_features(series, halflives=(10, 20, 40), return_halflife=40)
    std = regime.expanding_standardize(feats, min_obs=120)

    def switches(penalty):
        states, _ = regime.fit_jump_model(std, jump_penalty=penalty)
        return int((np.diff(states) != 0).sum())

    # Loss normalizada por feature: λ na casa de unidades, não dezenas.
    assert switches(8.0) < switches(0.2)


def test_expanding_standardize_has_no_lookahead():
    """A padronização em t não pode usar dados de t+1.

    Verifica truncando a série: a estatística num ponto deve ser idêntica quer o
    futuro exista ou não.
    """
    series = _two_regime_series(seed=2)
    feats = regime.regime_features(series, halflives=(10, 20, 40), return_halflife=40)

    full = regime.expanding_standardize(feats, min_obs=120)
    cut = full.index[400]
    truncated = regime.expanding_standardize(feats.loc[feats.index <= cut], min_obs=120)

    common = truncated.index
    pd.testing.assert_frame_equal(full.loc[common], truncated.loc[common])


def test_online_regime_is_causal():
    """O estado online num dia não pode mudar quando o futuro é removido.

    É o análogo do teste de look-ahead do backtest, para o detector de regime.
    """
    series = _two_regime_series(seed=3)
    feats = regime.regime_features(series, halflives=(10, 20, 40), return_halflife=40)
    std = regime.expanding_standardize(feats, min_obs=120)

    full = regime.online_regime(std, jump_penalty=1.0, refit_every=21,
                                lookback=500, min_train=200)
    cut = full.index[len(full) // 2]
    truncated = regime.online_regime(std.loc[std.index <= cut], jump_penalty=1.0,
                                     refit_every=21, lookback=500, min_train=200)

    common = truncated.index[truncated.index <= cut]
    # Refit periódico ancorado no início: os estados até o corte coincidem.
    np.testing.assert_array_equal(full.reindex(common).to_numpy(),
                                  truncated.reindex(common).to_numpy())


def test_exposure_multiplier_reduces_in_stress():
    states = pd.Series([0, 0, 1, 1, 0], index=pd.bdate_range("2020-01-01", periods=5))
    mult = regime.exposure_multiplier(states, bear_multiplier=0.5)
    assert mult.tolist() == [1.0, 1.0, 0.5, 0.5, 1.0]


def test_labels_are_stable_by_risk():
    """O rótulo do estado de risco não pode depender da inicialização aleatória."""
    series = _two_regime_series(seed=4)
    feats = regime.regime_features(series, halflives=(10, 20, 40), return_halflife=40)
    std = regime.expanding_standardize(feats, min_obs=120)

    _, thetas_a = regime.fit_jump_model(std, jump_penalty=1.0, seed=1)
    _, thetas_b = regime.fit_jump_model(std, jump_penalty=1.0, seed=2)
    # Estado 1 (alto risco) tem downside deviation maior que o estado 0, sempre.
    assert thetas_a[1, :-1].sum() >= thetas_a[0, :-1].sum()
    assert thetas_b[1, :-1].sum() >= thetas_b[0, :-1].sum()
