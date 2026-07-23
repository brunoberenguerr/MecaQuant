"""Dimensionamento de posições.

Vol inversa em vez de peso igual porque o sinal é cross-sectional: um s-score de
−1.5 numa ação de vol 60% e outro numa de vol 15% não são a mesma aposta. Sem a
normalização por risco, a carteira fica implicitamente concentrada nos nomes mais
voláteis, e a maior parte da variância do resultado passa a vir de quem por acaso
era volátil — não do sinal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mare.features.vol import ewma_vol


def equal_weights(symbols: list[str]) -> pd.Series:
    if not symbols:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(symbols), index=symbols)


def inverse_vol_weights(returns: pd.DataFrame, symbols: list[str],
                        lam: float = 0.94) -> pd.Series:
    """Pesos proporcionais a 1/σ, com σ estimado por EWMA (RiskMetrics)."""
    if not symbols:
        return pd.Series(dtype=float)
    vol = ewma_vol(returns[symbols], lam=lam).iloc[-1]
    inv = 1.0 / vol.replace(0.0, np.nan)
    inv = inv.replace([np.inf, -np.inf], np.nan)
    if inv.isna().all():
        return equal_weights(symbols)
    inv = inv.fillna(inv.median())
    return inv / inv.sum()


def apply_cap(weights: pd.Series, cap: float) -> pd.Series:
    """Limita o peso por nome e redistribui o excesso entre os não limitados.

    Iterativo porque redistribuir pode estourar o teto de outro nome.
    """
    if weights.empty or cap <= 0:
        return weights
    w = weights.copy()
    for _ in range(100):
        excess = (w - cap).clip(lower=0.0)
        if excess.sum() <= 1e-12:
            break
        w = w.clip(upper=cap)
        free = w < cap - 1e-12
        if not free.any():
            return pd.Series(1.0 / len(w), index=w.index)
        w[free] += excess.sum() * w[free] / w[free].sum()
    return w / w.sum()


def apply_sector_cap(weights: pd.Series, sectors: pd.Series,
                     cap: float) -> pd.Series:
    """Limita a exposição agregada por setor GICS.

    Reversão residual tende a concentrar setor: quando um setor inteiro cai, ele
    domina a lista de "baratos". Sem teto, a carteira vira uma aposta setorial
    disfarçada de arbitragem estatística.
    """
    if weights.empty or sectors is None or cap >= 1.0:
        return weights
    w = weights.copy()
    groups = sectors.reindex(w.index).fillna("UNKNOWN")
    for _ in range(50):
        totals = w.groupby(groups).sum()
        over = totals[totals > cap + 1e-12]
        if over.empty:
            break
        for sector, total in over.items():
            mask = groups == sector
            w[mask] *= cap / total
        w = w / w.sum()
    return w


def build_weights(returns: pd.DataFrame, symbols: list[str], *,
                  scheme: str = "inverse_vol", lam: float = 0.94,
                  cap: float | None = None,
                  sectors: pd.Series | None = None,
                  sector_cap: float = 1.0) -> pd.Series:
    """Pipeline de sizing: esquema base -> teto por nome -> teto por setor."""
    if scheme == "equal":
        weights = equal_weights(symbols)
    elif scheme == "inverse_vol":
        weights = inverse_vol_weights(returns, symbols, lam=lam)
    else:
        raise ValueError(f"esquema de sizing desconhecido: {scheme}")

    if cap is not None:
        weights = apply_cap(weights, cap)
    if sectors is not None and sector_cap < 1.0:
        weights = apply_sector_cap(weights, sectors, sector_cap)
        if cap is not None:
            weights = apply_cap(weights, cap)
    return weights
