"""Testes de placebo: qual é a distribuição do Sharpe sob a hipótese nula?

O argumento mais forte que um trabalho quantitativo pode apresentar, e o mais
barato. Métricas como o Deflated Sharpe exigem que a banca aceite uma fórmula;
o placebo mostra o resultado direto: rodamos o MESMO pipeline com sinal
destruído mil vezes e vemos onde a estratégia real cai nessa distribuição.

Dois nulos complementares, porque destroem coisas diferentes:

1. `random_selection`: mantém o universo, o número de nomes, a frequência de
   rebalanceamento e o esquema de pesos — só troca a SELEÇÃO por sorteio. Isola
   o valor do sinal contra "qualquer cesta de 30 ações teria feito isso".

2. `block_bootstrap`: reamostra a própria série de retornos da estratégia em
   blocos, preservando autocorrelação de curto prazo mas destruindo a ordem
   temporal. Testa se o Sharpe observado é compatível com sorte.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mare.backtest.metrics import sharpe

TRADING_DAYS = 252


def random_selection_sharpes(returns: pd.DataFrame, eligible_by_date: dict,
                             n_names: int, n_draws: int = 1000,
                             seed: int = 0) -> pd.Series:
    """Sharpes de carteiras aleatórias com mesmo N e mesma cadência.

    `eligible_by_date` mapeia data de rebalanceamento -> lista de símbolos
    elegíveis NAQUELA data (o mesmo universo que a estratégia real enxergou),
    para que a comparação isole a seleção e não a elegibilidade.

    Implementado em numpy: pré-computa, por período entre rebalanceamentos, a
    matriz de retornos dos elegíveis, e cada sorteio é uma média de colunas. Sem
    isso, 400 sorteios × 500 datas × slice de pandas leva minutos.
    """
    rng = np.random.default_rng(seed)
    dates = sorted(eligible_by_date)
    if len(dates) < 2:
        return pd.Series(dtype=float)

    col_pos = {c: i for i, c in enumerate(returns.columns)}
    values = returns.to_numpy(dtype=float)
    index = returns.index

    # Para cada período, guarda o bloco de retornos (dias × elegíveis) uma vez.
    blocks = []
    for i, date in enumerate(dates):
        pool = [col_pos[c] for c in eligible_by_date[date] if c in col_pos]
        if not pool:
            continue
        end = dates[i + 1] if i + 1 < len(dates) else index[-1]
        rows = np.where((index > date) & (index <= end))[0]
        if rows.size:
            blocks.append((values[np.ix_(rows, pool)], len(pool)))

    out = np.full(n_draws, np.nan)
    for draw in range(n_draws):
        daily = [block[:, rng.choice(width, size=min(n_names, width), replace=False)]
                 .mean(axis=1) for block, width in blocks]
        if daily:
            series = np.concatenate(daily)
            sd = series.std(ddof=1)
            out[draw] = series.mean() / sd * np.sqrt(252) if sd > 0 else np.nan
    return pd.Series(out, name="sharpe_placebo")


def block_bootstrap_sharpes(returns: pd.Series, n_draws: int = 1000,
                            block: int = 21, seed: int = 0) -> pd.Series:
    """Sharpes de reamostragens em blocos da série da estratégia."""
    clean = returns.dropna().to_numpy(dtype=float)
    n = len(clean)
    if n < 3 * block:
        return pd.Series(dtype=float)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block, size=(n_draws, n_blocks))
    out = np.empty(n_draws)
    for draw in range(n_draws):
        sample = np.concatenate([clean[s:s + block] for s in starts[draw]])[:n]
        sd = sample.std(ddof=1)
        out[draw] = sample.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan
    return pd.Series(out, name="sharpe_bootstrap")


def percentile_of(observed: float, distribution: pd.Series) -> float:
    """Em que percentil da distribuição nula o resultado observado cai."""
    clean = distribution.dropna()
    if clean.empty or not np.isfinite(observed):
        return np.nan
    return float((clean < observed).mean())


def summarize(observed: float, distribution: pd.Series,
              label: str = "") -> dict[str, float]:
    """Bloco para o relatório: onde caímos e qual o p-valor empírico."""
    clean = distribution.dropna()
    if clean.empty:
        return {"label": label, "observado": observed}
    return {
        "label": label,
        "observado": observed,
        "nulo_media": float(clean.mean()),
        "nulo_p50": float(clean.median()),
        "nulo_p95": float(clean.quantile(0.95)),
        "nulo_p99": float(clean.quantile(0.99)),
        "percentil": percentile_of(observed, clean),
        "p_valor": float((clean >= observed).mean()),
        "n_amostras": len(clean),
    }
