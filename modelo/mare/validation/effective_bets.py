"""Número efetivo de apostas independentes (Meucci 2010).

A Lei Fundamental (IR ≈ IC×√amplitude) conta apostas INDEPENDENTES, não tickers.
Se os N nomes segurados forem altamente correlacionados, N tickers valem muito
menos que N apostas — o "ganho de amplitude" de aumentar `n_names` seria em boa
parte cosmético.

Isso é exatamente o que a residualização por PCA/RMT deveria consertar: o que
seguramos não é o retorno bruto (correlacionado ao mercado e ao setor), é o
RESÍDUO depois de remover os k fatores comuns. Este módulo mede, com os dados de
MARÉ, se isso realmente funciona — em vez de assumir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def effective_bets(returns: pd.DataFrame, weights: pd.Series | None = None) -> dict:
    """Número efetivo de apostas via entropia dos autovalores (Meucci).

    Decompõe a covariância em componentes principais e mede quanto da variância
    do portfólio está concentrada em poucos componentes (correlacionados) vs
    espalhada (independentes). ENB = exp(entropia de Shannon da distribuição de
    variância pelos componentes). ENB=1 significa que só há 1 aposta real
    independente por trás de N nomes; ENB=N significa apostas perfeitamente
    independentes.

    Também devolve a versão simplificada por correlação média pairwise,
    ENB_naive = N / (1 + (N-1)·ρ̄), que é a heurística mais comum e mais fácil de
    citar — as duas devem concordar em ordem de grandeza.
    """
    symbols = returns.columns
    n = len(symbols)
    if n < 2:
        return {"n_tickers": n, "enb_meucci": float(n), "enb_naive": float(n),
               "avg_corr": float("nan")}

    if weights is None:
        weights = pd.Series(1.0 / n, index=symbols)
    w = weights.reindex(symbols).fillna(0.0).to_numpy()

    cov = returns.cov().to_numpy()
    corr = returns.corr().to_numpy()
    avg_corr = float((corr.sum() - n) / (n * (n - 1)))

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)  # numérico: covariância é PSD
    port_var = float(w @ cov @ w)
    if port_var <= 0:
        return {"n_tickers": n, "enb_meucci": float("nan"),
               "enb_naive": n / (1 + (n - 1) * avg_corr), "avg_corr": avg_corr}

    # Contribuição de cada componente principal à variância do portfólio.
    proj = eigvecs.T @ w
    contrib = (proj ** 2) * eigvals
    p = contrib / contrib.sum()
    p = p[p > 1e-12]
    entropy = -(p * np.log(p)).sum()
    enb_meucci = float(np.exp(entropy))

    enb_naive = n / (1 + (n - 1) * avg_corr) if avg_corr > -1 / (n - 1) else float(n)

    return {"n_tickers": n, "enb_meucci": enb_meucci, "enb_naive": enb_naive,
           "avg_corr": avg_corr}
