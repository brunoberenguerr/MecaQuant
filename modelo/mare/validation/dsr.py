"""Sharpe Probabilístico e Deflacionado (Bailey & López de Prado).

Um Sharpe reportado é sempre o MÁXIMO de várias tentativas, e o máximo de N
variáveis aleatórias tem valor esperado positivo mesmo quando todas têm média
zero. Quem testa 100 configurações e reporta a melhor exibe um número inflado
por construção — sem má-fé nenhuma.

O Deflated Sharpe corrige por três coisas ao mesmo tempo: número de tentativas,
tamanho da amostra e não-normalidade dos retornos (assimetria e curtose, que em
estratégia de reversão são justamente ruins — ganhos pequenos e frequentes,
perdas raras e grandes).

    SR₀ = E[max SR | N tentativas]
        = E[SR] + √V[SR]·((1−γ)·Z⁻¹[1−1/N] + γ·Z⁻¹[1−1/(N·e)])

    DSR = Z[ (SR − SR₀)·√(T−1) / √(1 − γ₃·SR + ((γ₄−1)/4)·SR²) ]

com γ ≈ 0.5772 (Euler–Mascheroni). O N vem do run registry, não de estimativa
de memória — ver `mare/registry.py`.

Atenção à convenção: todas as funções aqui usam Sharpe POR PERÍODO (diário), não
anualizado. Misturar as duas escalas é o erro clássico neste cálculo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329
TRADING_DAYS = 252


def sharpe_per_period(returns: pd.Series) -> float:
    clean = returns.dropna()
    sd = clean.std(ddof=1)
    return float(clean.mean() / sd) if sd > 0 else np.nan


def probabilistic_sharpe_ratio(returns: pd.Series,
                               benchmark_sr: float = 0.0) -> float:
    """P(SR verdadeiro > `benchmark_sr`), corrigindo por assimetria e curtose."""
    clean = returns.dropna()
    n = len(clean)
    if n < 30:
        return np.nan
    sr = sharpe_per_period(clean)
    if not np.isfinite(sr):
        return np.nan
    skew = float(clean.skew())
    kurt = float(clean.kurtosis()) + 3.0          # pandas devolve curtose EXCESSO
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom <= 0:
        return np.nan
    return float(norm.cdf((sr - benchmark_sr) * np.sqrt(n - 1) / np.sqrt(denom)))


def expected_max_sharpe(n_trials: int, sr_variance: float,
                        sr_mean: float = 0.0) -> float:
    """E[max SR] sob a hipótese nula de que nenhuma tentativa tem edge."""
    if n_trials <= 1:
        return sr_mean
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    factor = (1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2
    return float(sr_mean + np.sqrt(max(sr_variance, 0.0)) * factor)


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int,
                          trial_sharpes: pd.Series | None = None) -> dict[str, float]:
    """DSR e os ingredientes que entraram nele.

    `trial_sharpes` são os Sharpes (por período) de TODAS as configurações
    testadas — a variância cross-sectional deles é o que mede quanta dispersão a
    busca gerou. Sem eles, usa-se a aproximação 1/(T−1) para V[SR], que é
    conservadora no sentido errado (subestima a deflação).
    """
    clean = returns.dropna()
    n = len(clean)
    sr = sharpe_per_period(clean)
    if not np.isfinite(sr) or n < 30:
        return {"sharpe_anual": np.nan, "sr0_anual": np.nan, "dsr": np.nan,
                "n_trials": n_trials, "psr_vs_0": np.nan}

    if trial_sharpes is not None and len(trial_sharpes.dropna()) > 2:
        sr_var = float(trial_sharpes.dropna().var(ddof=1))
    else:
        sr_var = 1.0 / (n - 1)

    sr0 = expected_max_sharpe(n_trials, sr_var)
    return {
        "sharpe_anual": sr * np.sqrt(TRADING_DAYS),
        "sr0_anual": sr0 * np.sqrt(TRADING_DAYS),
        "dsr": probabilistic_sharpe_ratio(clean, benchmark_sr=sr0),
        "psr_vs_0": probabilistic_sharpe_ratio(clean, benchmark_sr=0.0),
        "n_trials": n_trials,
        "n_obs": n,
    }


def min_track_record_length(returns: pd.Series, target_confidence: float = 0.95,
                            benchmark_sr: float = 0.0) -> float:
    """Quantas observações seriam necessárias para o SR ser significante.

    Se o resultado for maior que a amostra disponível, o track record ainda não
    prova nada — e dizer isso no relatório vale mais do que omitir.
    """
    clean = returns.dropna()
    sr = sharpe_per_period(clean)
    if not np.isfinite(sr) or abs(sr - benchmark_sr) < 1e-12:
        return np.inf
    skew = float(clean.skew())
    kurt = float(clean.kurtosis()) + 3.0
    z = norm.ppf(target_confidence)
    numer = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    return float(1 + numer * (z / (sr - benchmark_sr)) ** 2)
