"""Estimadores de volatilidade.

EWMA é o cavalo de batalha (sizing, padronização). GARCH entra só no relógio de
negócio do Pilar 1, onde precisamos da variância *prevista* para converter o
tempo esperado de trade de volta para tempo de calendário.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def ewma_vol(returns: pd.DataFrame | pd.Series, lam: float = 0.94,
             annualize: bool = False) -> pd.DataFrame | pd.Series:
    """Volatilidade EWMA no padrão RiskMetrics (λ=0.94 para dado diário)."""
    var = returns.pow(2).ewm(alpha=1 - lam, adjust=False, min_periods=20).mean()
    vol = np.sqrt(var)
    return vol * np.sqrt(TRADING_DAYS) if annualize else vol


def exponential_weights(n: int, halflife: float) -> np.ndarray:
    """Pesos normalizados com o dado mais recente por último."""
    age = np.arange(n - 1, -1, -1, dtype=float)
    w = 0.5 ** (age / halflife)
    return w / w.sum()


def weighted_corr(returns: pd.DataFrame, halflife: float | None = None
                  ) -> tuple[pd.DataFrame, pd.Series]:
    """Matriz de correlação com ponderação exponencial e vetor de vols.

    Devolve os dois porque o eigenportfolio de Avellaneda & Lee precisa da
    correlação (para os autovetores) e das vols (para converter autovetor em peso).
    """
    data = returns.dropna(axis=1, how="all")
    n = len(data)
    w = exponential_weights(n, halflife) if halflife else np.full(n, 1 / n)
    values = data.to_numpy(dtype=float)
    filled = np.where(np.isnan(values), 0.0, values)

    mean = w @ filled
    centered = filled - mean
    cov = (centered * w[:, None]).T @ centered
    cov /= (1 - (w ** 2).sum())
    vols = pd.Series(np.sqrt(np.diag(cov)), index=data.columns)

    safe = vols.to_numpy()
    safe = np.where(safe <= 0, np.nan, safe)
    corr = cov / np.outer(safe, safe)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=data.columns, columns=data.columns), vols


def realized_vol(returns: pd.DataFrame | pd.Series, window: int = 21,
                 annualize: bool = True) -> pd.DataFrame | pd.Series:
    vol = returns.rolling(window, min_periods=max(5, window // 2)).std()
    return vol * np.sqrt(TRADING_DAYS) if annualize else vol


def downside_deviation(returns: pd.Series, halflife: int) -> pd.Series:
    """Semi-desvio EWM — feature do modelo de regime.

    Só o lado negativo: o que define regime ruim é a cauda esquerda, e incluir a
    alta trata rali volátil como se fosse crise.
    """
    negative = returns.clip(upper=0.0)
    var = negative.pow(2).ewm(halflife=halflife, min_periods=halflife).mean()
    return np.sqrt(var)
