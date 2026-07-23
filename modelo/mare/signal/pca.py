"""Bloco fatorial: decomposição espectral do operador de covariância.

É aqui que a análise funcional é load-bearing e não decorativa. Os retornos
vivem em L²; o operador de covariância é compacto, auto-adjunto e positivo, e o
teorema espectral garante uma base ortonormal de autovetores. A expansão de
Karhunen–Loève truncada em k termos é a componente sistemática do retorno, e o
resíduo que a estratégia negocia é literalmente a projeção no complemento
ortogonal do span dos k primeiros autovetores.

A escolha de k não é um parâmetro livre: vem da teoria de matrizes aleatórias.
Autovalores dentro do suporte de Marchenko–Pastur são indistinguíveis de ruído
de amostragem, então só os que ultrapassam a borda superior λ₊ carregam
estrutura (Laloux et al. 1999; Bouchaud & Potters).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mare.features.vol import weighted_corr


def eigen_decomposition(corr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Autovalores (decrescentes) e autovetores da matriz de correlação."""
    matrix = np.nan_to_num(corr.to_numpy(dtype=float), nan=0.0)
    matrix = (matrix + matrix.T) / 2.0  # simetriza contra erro numérico
    vals, vecs = np.linalg.eigh(matrix)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def mp_upper_edge(n_assets: int, n_obs: int, sigma2: float = 1.0) -> float:
    """Borda superior do suporte de Marchenko–Pastur: λ₊ = σ²(1+√q)², q = N/T."""
    q = n_assets / n_obs
    return sigma2 * (1.0 + np.sqrt(q)) ** 2


def estimate_bulk_sigma2(eigenvalues: np.ndarray, n_market_modes: int = 1) -> float:
    """Nível de ruído σ², estimado APÓS condicionar no modo de mercado.

    O modo de mercado não pertence ao bulk de ruído: ele é um fator conhecido e
    universal que, sozinho, concentra 26–56% do traço da correlação (medido no
    nosso universo). Como o traço é fixo em N, deixá-lo dentro do cálculo faz a
    média dos autovalores restantes despencar.

    Duas alternativas foram testadas e rejeitadas empiricamente:

    - Ponto fixo preservando o traço (σ² = média dos autovalores fora do sinal,
      iterando em k): DEGENERA. Cada iteração remove autovalores do topo, o que
      derruba σ², o que derruba a borda, o que admite mais autovalores. Mediu-se
      k = 97 a 189 conforme a data — sem sentido para N ≈ 350.
    - σ² = 1 fixo (sem ajuste): subestima, dando k = 5 a 12 e variando muito com
      q. Cinco fatores deixam estrutura setorial inteira dentro do "resíduo".

    Condicionar no modo de mercado dá k = 19 a 23, estável entre 2009 e 2021 e
    consistente com os 15 de Avellaneda & Lee (2010).
    """
    rest = eigenvalues[n_market_modes:]
    return float(rest.mean()) if rest.size else 1.0


def select_k(eigenvalues: np.ndarray, n_assets: int, n_obs: int,
             k_min: int = 5, k_max: int = 30, method: str = "marchenko_pastur",
             k_fixed: int = 15) -> int:
    """Número de fatores.

    `marchenko_pastur` (padrão) condiciona no modo de mercado e não tem
    parâmetro livre. `mp_raw` usa σ²=1 e existe só para o estudo de
    sensibilidade do relatório. `fixed` reproduz o valor de A&L.
    """
    if method == "fixed":
        return int(np.clip(k_fixed, k_min, k_max))
    if method == "mp_raw":
        k = int((eigenvalues > mp_upper_edge(n_assets, n_obs, 1.0)).sum())
        return int(np.clip(k, k_min, k_max))
    if method != "marchenko_pastur":
        raise ValueError(f"método de seleção de k desconhecido: {method}")

    sigma2 = estimate_bulk_sigma2(eigenvalues, n_market_modes=1)
    edge = mp_upper_edge(n_assets, n_obs, sigma2)
    k = 1 + int((eigenvalues[1:] > edge).sum())  # o modo de mercado sempre entra
    return int(np.clip(k, k_min, k_max))


def apply_hysteresis(history: list[int], current: int, window: int = 21) -> int:
    """Suaviza k pela mediana da janela recente.

    k saltando entre rebalanceamentos muda o span dos regressores e gera
    turnover que é puro ruído de estimação, não sinal.
    """
    if window <= 1:
        return current
    recent = ([*history, current])[-window:]
    return int(np.median(recent))


def eigenportfolio_weights(eigenvectors: np.ndarray, vols: pd.Series,
                           k: int) -> pd.DataFrame:
    """Pesos dos eigenportfolios: Q_ij = v_i^(j) / σ_i (Avellaneda & Lee 2010).

    Dividir pela vol converte o autovetor da correlação em uma carteira
    investível de risco equilibrado. As colunas são normalizadas para somar 1 em
    módulo, o que só afeta a escala dos fatores — o resíduo é invariante.
    """
    inv_vol = 1.0 / vols.replace(0.0, np.nan)
    weights = eigenvectors[:, :k] * inv_vol.to_numpy()[:, None]
    norm = np.abs(weights).sum(axis=0)
    norm[norm == 0] = 1.0
    return pd.DataFrame(weights / norm, index=vols.index,
                        columns=[f"F{i + 1}" for i in range(k)])


def factor_returns(returns: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Séries de retorno dos eigenportfolios."""
    aligned = returns.reindex(columns=weights.index).fillna(0.0)
    return pd.DataFrame(aligned.to_numpy() @ weights.to_numpy(),
                        index=returns.index, columns=weights.columns)


def build_factors(returns: pd.DataFrame, *, halflife: float | None,
                  k_method: str = "marchenko_pastur", k_fixed: int = 15,
                  k_min: int = 5, k_max: int = 30
                  ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Pipeline completo: correlação ponderada → espectro → k → eigenportfolios.

    Devolve (retornos dos fatores, pesos, k).
    """
    corr, vols = weighted_corr(returns, halflife=halflife)
    vals, vecs = eigen_decomposition(corr)
    k = select_k(vals, n_assets=corr.shape[0], n_obs=len(returns),
                 k_min=k_min, k_max=k_max, method=k_method, k_fixed=k_fixed)
    weights = eigenportfolio_weights(vecs, vols, k)
    return factor_returns(returns, weights), weights, k


def explained_variance(eigenvalues: np.ndarray, k: int) -> float:
    total = float(eigenvalues.sum())
    return float(eigenvalues[:k].sum() / total) if total else 0.0
