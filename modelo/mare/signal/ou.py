"""Ajuste do processo de Ornstein–Uhlenbeck sobre o resíduo acumulado.

Modelo: dX_t = κ(θ − X_t)dt + σ dW_t.

A discretização EXATA do OU é um AR(1) — não uma aproximação de Euler:

    X_{t+1} = a + b·X_t + ε,   b = e^{−κΔt},  a = θ(1−b),  Var(ε) = σ²(1−b²)/(2κ)

de onde κ = −ln(b)/Δt, θ = a/(1−b) e, o que mais importa para o sinal,
σ_eq = σ/√(2κ) = s_ε/√(1−b²).

Dois vieses de estimação são corrigidos aqui porque ambos empurram o backtest
na direção bonita:

1. Viés de Kendall. O estimador de b num AR(1) com intercepto tem viés
   descendente de ordem −(1+3b)/T. b subestimado significa κ SUPERestimado, o
   que reduz σ_eq, o que infla |s-score|, o que faz operar demais — e justamente
   nos estados em que o estimador é mais ruidoso.

2. Não-identificação. Com meia-vida de ~20 dias numa janela de 60, há cerca de
   3 ciclos independentes: κ por nome é quase não-identificado. O shrinkage
   empirical-Bayes para o pooled cross-sectional usa ~N vezes mais informação e
   é a correção de estimação mais valiosa do projeto.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OUFit:
    """Parâmetros do OU por símbolo, já corrigidos e encolhidos."""

    kappa: pd.Series      # velocidade de reversão (por dia)
    theta: pd.Series      # nível de equilíbrio
    sigma_eq: pd.Series   # desvio-padrão estacionário
    b: pd.Series          # coeficiente AR(1) usado
    half_life: pd.Series  # ln2 / kappa, em dias

    def s_score(self, level: pd.Series) -> pd.Series:
        """s = (X − θ)/σ_eq. Negativo = barato em relação ao equilíbrio."""
        return (level - self.theta) / self.sigma_eq.replace(0.0, np.nan)


def _ar1_ols(series: np.ndarray) -> tuple[float, float, float]:
    """OLS de X_{t+1} sobre X_t com intercepto. Devolve (a, b, sigma_eps)."""
    x, y = series[:-1], series[1:]
    n = len(x)
    if n < 10:
        return np.nan, np.nan, np.nan
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx <= 0:
        return np.nan, np.nan, np.nan
    b = ((x - xm) * (y - ym)).sum() / sxx
    a = ym - b * xm
    resid = y - (a + b * x)
    return a, b, float(np.sqrt((resid ** 2).sum() / max(n - 2, 1)))


def kendall_correct(b: float, n_obs: int) -> float:
    """Remove o viés de amostra pequena do coeficiente AR(1)."""
    if not np.isfinite(b) or n_obs <= 3:
        return b
    return float(np.clip(b + (1.0 + 3.0 * b) / n_obs, -0.999, 0.9999))


def shrink_ar_coefficient(b: pd.Series, n_obs: int) -> pd.Series:
    """Shrinkage empirical-Bayes de b para a média cross-sectional.

    Peso w = τ²/(τ² + s²), com s² = (1−b²)/T a variância amostral de b̂ e τ² a
    dispersão cross-sectional verdadeira, estimada removendo o ruído amostral da
    variância observada. τ² pequeno (nomes parecidos, estimativas ruidosas) leva
    tudo para o pooled, que é o comportamento desejado.
    """
    valid = b.dropna()
    if len(valid) < 5:
        return b
    pooled = float(valid.mean())
    sampling_var = float(((1 - valid.clip(-0.999, 0.999) ** 2) / n_obs).mean())
    tau2 = max(float(valid.var(ddof=1)) - sampling_var, 0.0)
    weight = tau2 / (tau2 + sampling_var) if (tau2 + sampling_var) > 0 else 0.0
    return b.where(b.isna(), weight * b + (1 - weight) * pooled)


def fit_ou(cumulative: pd.DataFrame, *, kendall: bool = True,
           shrinkage: bool = True, shrink_theta_to_zero: bool = True,
           dt: float = 1.0) -> OUFit:
    """Ajusta o OU coluna a coluna sobre o resíduo acumulado."""
    records: dict[str, tuple[float, float, float]] = {}
    for symbol in cumulative.columns:
        series = cumulative[symbol].to_numpy(dtype=float)
        series = series[np.isfinite(series)]
        records[symbol] = _ar1_ols(series)

    frame = pd.DataFrame(records, index=["a", "b", "sigma_eps"]).T
    n_obs = max(len(cumulative) - 1, 2)

    if kendall:
        frame["b"] = frame["b"].map(lambda v: kendall_correct(v, n_obs))
    if shrinkage:
        frame["b"] = shrink_ar_coefficient(frame["b"], n_obs)

    b = frame["b"].clip(upper=0.9999)
    valid = b.notna() & (b > 0) & (b < 1)

    kappa = pd.Series(np.nan, index=frame.index)
    kappa[valid] = -np.log(b[valid]) / dt

    theta = pd.Series(0.0, index=frame.index)
    if not shrink_theta_to_zero:
        theta[valid] = frame["a"][valid] / (1 - b[valid])

    sigma_eq = pd.Series(np.nan, index=frame.index)
    sigma_eq[valid] = frame["sigma_eps"][valid] / np.sqrt(1 - b[valid] ** 2)

    half_life = pd.Series(np.nan, index=frame.index)
    half_life[valid] = np.log(2.0) / kappa[valid]

    return OUFit(kappa=kappa, theta=theta, sigma_eq=sigma_eq, b=b,
                 half_life=half_life)


def simulate_ou(kappa: float, theta: float, sigma: float, x0: float,
                n_steps: int, n_paths: int = 1, dt: float = 1.0,
                seed: int | None = None) -> np.ndarray:
    """Simula o OU pela discretização exata. Usado nos testes e no Monte Carlo.

    Exata, não Euler: o passo usa b = e^{−κΔt} e ruído com a variância
    estacionária correta, então o resultado é válido para qualquer Δt.
    """
    rng = np.random.default_rng(seed)
    b = np.exp(-kappa * dt)
    noise_sd = sigma * np.sqrt((1 - b ** 2) / (2 * kappa)) if kappa > 0 else sigma
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x0
    shocks = rng.standard_normal((n_paths, n_steps)) * noise_sd
    for step in range(n_steps):
        paths[:, step + 1] = theta + b * (paths[:, step] - theta) + shocks[:, step]
    return paths
