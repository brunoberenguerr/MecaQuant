"""Métricas de performance e risco.

Duas decisões que evitam inflar os números:

1. O t-stat do Sharpe usa Newey–West. As janelas de sinal se sobrepõem
   (59 de 60 dias em comum entre rebalanceamentos consecutivos), então os
   retornos são autocorrelacionados e o erro-padrão ingênuo subestima a
   incerteza — o t-stat sai inflado.

2. Sharpe é sempre reportado sobre o EXCESSO em relação à taxa livre de risco.
   Numa amostra que inclui 2007 (juros a 5%) e 2021 (juros a 0%), ignorar isso
   distorce a comparação entre subperíodos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    years = len(clean) / TRADING_DAYS
    total = float((1 + clean).prod())
    return total ** (1 / years) - 1 if years > 0 and total > 0 else np.nan


def volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, risk_free: pd.Series | float = 0.0) -> float:
    excess = _excess(returns, risk_free)
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan


def sortino(returns: pd.Series, risk_free: pd.Series | float = 0.0) -> float:
    excess = _excess(returns, risk_free)
    downside = excess.clip(upper=0.0)
    dd = np.sqrt((downside ** 2).mean())
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS)) if dd > 0 else np.nan


def drawdown_series(returns: pd.Series) -> pd.Series:
    curve = (1 + returns.fillna(0.0)).cumprod()
    return curve / curve.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def calmar(returns: pd.Series) -> float:
    mdd = abs(max_drawdown(returns))
    return float(cagr(returns) / mdd) if mdd > 0 else np.nan


def newey_west_tstat(returns: pd.Series, lags: int | None = None) -> float:
    """t-stat da média com erro-padrão robusto a autocorrelação e heterocedasticidade."""
    x = returns.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 20:
        return np.nan
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))  # regra de Newey–West
    dev = x - x.mean()
    var = (dev @ dev) / n
    for lag in range(1, lags + 1):
        cov = (dev[lag:] @ dev[:-lag]) / n
        var += 2 * (1 - lag / (lags + 1)) * cov
    if var <= 0:
        return np.nan
    return float(x.mean() / np.sqrt(var / n))


def hit_rate(returns: pd.Series) -> float:
    clean = returns.dropna()
    return float((clean > 0).mean()) if len(clean) else np.nan


def summary(returns: pd.Series, risk_free: pd.Series | float = 0.0,
            label: str = "") -> dict[str, float]:
    """Bloco padrão de métricas. Usado em todo lugar para não divergir."""
    clean = returns.dropna()
    return {
        "label": label,
        "n_dias": len(clean),
        "cagr": cagr(clean),
        "vol": volatility(clean),
        "sharpe": sharpe(clean, risk_free),
        "sortino": sortino(clean, risk_free),
        "max_drawdown": max_drawdown(clean),
        "calmar": calmar(clean),
        "hit_rate": hit_rate(clean),
        "t_stat_nw": newey_west_tstat(_excess(clean, risk_free)),
        "skew": float(clean.skew()),
        "kurtosis": float(clean.kurtosis()),
    }


def beta_alpha(returns: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
    """Beta e alpha anualizado contra um benchmark, por OLS simples."""
    joined = pd.concat([returns, benchmark], axis=1).dropna()
    if len(joined) < 20:
        return np.nan, np.nan
    y, x = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
    var = x.var(ddof=1)
    if var <= 0:
        return np.nan, np.nan
    beta = float(np.cov(y, x, ddof=1)[0, 1] / var)
    alpha = float((y.mean() - beta * x.mean()) * TRADING_DAYS)
    return beta, alpha


def three_series(strategy: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """Retorno total, ativo e beta-hedged.

    Reportar as três desarma o ataque mais previsível a uma carteira long-only
    selecionada por resíduo — "vocês só compraram beta" — e custa uma subtração
    de séries. A série ativa é a que deve ser usada na atribuição de fatores.
    """
    joined = pd.concat([strategy.rename("total"), benchmark.rename("bench")],
                       axis=1).dropna()
    beta, _ = beta_alpha(joined["total"], joined["bench"])
    return pd.DataFrame({
        "total": joined["total"],
        "ativo": joined["total"] - joined["bench"],
        "beta_hedged": joined["total"] - beta * joined["bench"],
    })


def _excess(returns: pd.Series, risk_free: pd.Series | float) -> pd.Series:
    if isinstance(risk_free, pd.Series):
        return (returns - risk_free.reindex(returns.index).fillna(0.0)).dropna()
    return returns - risk_free
