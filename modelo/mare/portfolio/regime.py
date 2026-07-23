"""Detecção de regime de mercado por Statistical Jump Model.

Motivação econômica, e não decorativa. A análise de qualidade de sinal mostrou
que a reversão residual só tem edge em regime de baixa volatilidade: em estresse,
quedas idiossincráticas carregam informação (revisão de fundamentos, contágio) e
não revertem. Logo o overlay de regime tem uma tarefa concreta — reduzir
exposição quando o mercado entra em estado de alta volatilidade.

Modelo (Nystrup, Kolm & Lindström; pacote de referência `jumpmodels`):

    min_{Θ,S}  Σ_t ½‖y_t − θ_{s_t}‖²  +  λ Σ_t 1{s_t ≠ s_{t−1}}

O termo de salto λ penaliza trocas de estado, o que dá regimes PERSISTENTES —
a diferença central para um HMM gaussiano, que troca de estado a cada respiro do
mercado (whipsaw). Ajuste por coordinate descent, alternando:

  (1) dado S, θ_k = média dos y_t no estado k        (passo tipo k-means)
  (2) dado Θ, S por programação dinâmica com custo λ  (passo tipo Viterbi)

DUAS armadilhas fatais, tratadas explicitamente:

- Look-ahead pela padronização. Padronizar as features com média/desvio de toda
  a amostra vaza o futuro. Aqui a padronização é EXPANDING: cada dia usa só o
  passado.
- Look-ahead pelo estado suavizado. A sequência que o DP devolve é suavizada —
  o estado no meio da janela conhece o que veio depois. Para decidir a exposição
  no dia t só se pode usar o ÚLTIMO ponto de um ajuste feito com dados até t.
  `online_regime` respeita isso reajustando ao longo do tempo e lendo sempre a
  ponta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mare.features.vol import downside_deviation


def regime_features(market_returns: pd.Series,
                    halflives: tuple[int, ...] = (20, 60, 120),
                    return_halflife: int = 120) -> pd.DataFrame:
    """Features do detector: semi-desvios EWM em várias escalas + retorno EWM.

    Semi-desvio (só o lado negativo) porque o que define regime ruim é a cauda
    esquerda; incluir a alta trataria rali volátil como crise.
    """
    feats = {f"dd_{hl}": downside_deviation(market_returns, hl) for hl in halflives}
    feats[f"ret_{return_halflife}"] = market_returns.ewm(
        halflife=return_halflife, min_periods=return_halflife).mean()
    return pd.DataFrame(feats).dropna()


def expanding_standardize(features: pd.DataFrame, min_obs: int = 252) -> pd.DataFrame:
    """Padroniza cada coluna com média/desvio ACUMULADOS até cada linha.

    Sem look-ahead por construção: a estatística de padronização no dia t usa
    apenas as linhas até t.
    """
    mean = features.expanding(min_periods=min_obs).mean()
    std = features.expanding(min_periods=min_obs).std(ddof=0)
    return ((features - mean) / std.replace(0.0, np.nan)).dropna()


def _fit_thetas(y: np.ndarray, states: np.ndarray, n_states: int) -> np.ndarray:
    thetas = np.zeros((n_states, y.shape[1]))
    for k in range(n_states):
        mask = states == k
        thetas[k] = y[mask].mean(axis=0) if mask.any() else y.mean(axis=0)
    return thetas


def _viterbi(y: np.ndarray, thetas: np.ndarray, jump_penalty: float) -> np.ndarray:
    """Programação dinâmica: sequência de estados de menor custo com penalidade λ.

    A loss é a MÉDIA sobre as features (não a soma), o que torna λ invariante ao
    número de features e interpretável: com features padronizadas, a loss por
    ponto é O(1), então λ compara diretamente com "quantos pontos de loss média
    vale evitar uma troca de estado". Sem essa normalização, λ teria de ser
    reescalado toda vez que se adiciona ou remove uma feature.
    """
    n, n_states = len(y), len(thetas)
    loss = 0.5 * ((y[:, None, :] - thetas[None, :, :]) ** 2).mean(axis=2)  # (n, k)

    cost = np.empty((n, n_states))
    back = np.zeros((n, n_states), dtype=int)
    cost[0] = loss[0]
    for t in range(1, n):
        prev = cost[t - 1][:, None] + jump_penalty * (1 - np.eye(n_states))
        back[t] = prev.argmin(axis=0)
        cost[t] = loss[t] + prev.min(axis=0)

    states = np.empty(n, dtype=int)
    states[-1] = cost[-1].argmin()
    for t in range(n - 1, 0, -1):
        states[t - 1] = back[t, states[t]]
    return states


def fit_jump_model(features: pd.DataFrame, n_states: int = 2,
                   jump_penalty: float = 50.0, n_init: int = 5,
                   max_iter: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Ajusta o jump model. Devolve (estados suavizados, thetas).

    ATENÇÃO: a sequência devolvida é SUAVIZADA. Só usar diretamente para análise
    retrospectiva; para decisão em tempo real, passar por `online_regime`.
    """
    y = features.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    best_states, best_thetas, best_cost = None, None, np.inf

    for _ in range(n_init):
        states = rng.integers(0, n_states, len(y))
        for _ in range(max_iter):
            thetas = _fit_thetas(y, states, n_states)
            new_states = _viterbi(y, thetas, jump_penalty)
            if np.array_equal(new_states, states):
                break
            states = new_states
        cost = _total_cost(y, states, thetas, jump_penalty)
        if cost < best_cost:
            best_states, best_thetas, best_cost = states, thetas, cost

    return _label_by_risk(best_states, best_thetas)


def _total_cost(y, states, thetas, jump_penalty) -> float:
    fit = 0.5 * ((y - thetas[states]) ** 2).mean(axis=1).sum()
    jumps = int((np.diff(states) != 0).sum())
    return float(fit + jump_penalty * jumps)


def _label_by_risk(states: np.ndarray, thetas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotula o estado de maior downside deviation como 1 (alto risco).

    Sem isso, o rótulo 0/1 é arbitrário e o overlay poderia inverter o sinal.
    A soma das features de downside (todas menos a última, que é retorno) mede
    risco: quanto maior, pior o regime.
    """
    risk = thetas[:, :-1].sum(axis=1)
    order = np.argsort(risk)          # estado de menor risco vira 0
    relabel = np.empty_like(order)
    relabel[order] = np.arange(len(order))
    return relabel[states], thetas[order]


def online_regime(features: pd.DataFrame, n_states: int = 2,
                  jump_penalty: float = 1.0, refit_every: int = 21,
                  lookback: int = 2000, min_train: int = 504,
                  seed: int = 0) -> pd.Series:
    """Estado de regime FILTRADO: cada ponto usa apenas informação até ele.

    Implementado como um filtro forward, não como um smoother re-executado. A
    distinção é a armadilha central deste modelo: a sequência que o Viterbi
    completo devolve é SUAVIZADA — o estado no meio da janela enxerga o futuro
    dela. Aqui mantém-se um vetor de custo acumulado e, a cada dia, o estado
    filtrado é o argmin desse vetor:

        c_t[k] = loss_t[k] + min_j (c_{t-1}[j] + λ·1{j≠k})
        estado_t = argmin_k c_t[k]

    que usa apenas dados até t. É O(estados) por dia — além de correto, é ~ordem
    de grandeza mais rápido que re-rodar o Viterbi na janela toda a cada dia.

    Os parâmetros θ (médias dos estados) são reajustados a cada `refit_every`
    dias na janela recente; entre refits ficam congelados. O reajuste usa só o
    passado, então nada vaza.
    """
    y = features.to_numpy(dtype=float)
    idx = features.index
    n = len(y)
    if n <= min_train:
        return pd.Series(dtype="float64")

    thetas = None
    cost = np.zeros(n_states)
    out = np.full(n, np.nan)
    transition = jump_penalty * (1.0 - np.eye(n_states))

    for pos in range(min_train, n):
        if thetas is None or (pos - min_train) % refit_every == 0:
            start = max(0, pos - lookback)
            # `fit_jump_model` já devolve θ ordenado por risco (estado 0 = calmo,
            # estado n-1 = estressado), então o argmin do custo já é o rótulo
            # correto e ele fica estável entre reajustes.
            _, thetas = fit_jump_model(features.iloc[start:pos + 1],
                                       n_states=n_states,
                                       jump_penalty=jump_penalty, seed=seed)

        loss = 0.5 * ((y[pos] - thetas) ** 2).mean(axis=1)
        cost = loss + (cost[:, None] + transition).min(axis=0)
        cost -= cost.min()               # estabiliza a escala; não muda o argmin
        out[pos] = int(cost.argmin())

    return pd.Series(out, index=idx).dropna()


def exposure_multiplier(states: pd.Series, bear_multiplier: float = 0.5,
                        n_states: int = 2) -> pd.Series:
    """Traduz estado de regime em multiplicador de exposição.

    Estado de maior risco (o de índice máximo após o relabel) recebe o
    multiplicador reduzido. Com 2 estados: calmo=1.0, estressado=bear_multiplier.
    """
    top = n_states - 1
    return states.map(lambda s: bear_multiplier if s >= top else 1.0)
