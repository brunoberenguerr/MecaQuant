"""Qualidade do sinal, antes de qualquer construção de carteira.

Diagnóstico que precede o backtest. Se o s-score não prevê retorno futuro, nenhum
esquema de pesos, alvo de vol ou overlay de regime vai salvar a estratégia — vão
só redistribuir o mesmo nada. Rodar isto primeiro evita atribuir a construção de
carteira um fracasso que é do sinal (ou o contrário).

Duas medidas:

- **Information Coefficient**: correlação de Spearman entre o sinal na data t e
  o retorno realizado em t+h, calculada no cross-section e depois agregada. O
  IC médio dividido pelo desvio das ICs é o "IR do sinal" — quanto do resultado
  é edge e quanto é sorte de período.

- **Decis**: ordena por s-score e mede o retorno futuro de cada decil. Se a
  hipótese de reversão vale, o decil mais barato (s mais negativo) deve render
  mais que o mais caro, e a relação deve ser aproximadamente monótona. Um
  padrão não-monótono é sinal de que o efeito vem de poucos outliers.

Os retornos futuros são medidos em EXCESSO sobre a média do universo elegível na
mesma data. O sinal é residual (já removeu os fatores), então avaliá-lo contra
retorno bruto misturaria o beta de mercado na medida.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def forward_excess_returns(returns: pd.DataFrame, date: pd.Timestamp,
                           symbols: list[str], horizon: int) -> pd.Series:
    """Retorno composto dos próximos `horizon` pregões, líquido da média do universo."""
    future = returns.loc[returns.index > date]
    if len(future) < horizon:
        return pd.Series(dtype=float)
    window = future.iloc[:horizon]
    available = [s for s in symbols if s in window.columns]
    total = (1 + window[available].fillna(0.0)).prod() - 1
    return total - total.mean()


def information_coefficient(signal: pd.Series, forward: pd.Series) -> float:
    """Spearman entre sinal e retorno futuro no cross-section de uma data."""
    joined = pd.concat([signal.rename("s"), forward.rename("r")], axis=1).dropna()
    if len(joined) < 20:
        return np.nan
    return float(stats.spearmanr(joined["s"], joined["r"]).statistic)


def decile_returns(signal: pd.Series, forward: pd.Series,
                   n_bins: int = 10) -> pd.Series:
    """Retorno futuro médio por decil de sinal (decil 1 = s mais negativo)."""
    joined = pd.concat([signal.rename("s"), forward.rename("r")], axis=1).dropna()
    if len(joined) < n_bins * 3:
        return pd.Series(dtype=float)
    bins = pd.qcut(joined["s"].rank(method="first"), n_bins, labels=False) + 1
    return joined.groupby(bins)["r"].mean()


def summarize_ic(ic_series: pd.Series, horizon: int) -> dict[str, float]:
    """IC médio, desvio, IR do sinal e t-stat.

    O sinal de referência importa: como esperamos REVERSÃO, um s-score baixo
    deve prever retorno alto, logo o IC esperado é NEGATIVO. Reportamos também
    o IC com sinal trocado para leitura direta ("quanto maior, melhor").
    """
    clean = ic_series.dropna()
    if len(clean) < 10:
        return {"horizonte": horizon, "n_datas": len(clean)}
    mean, sd = float(clean.mean()), float(clean.std(ddof=1))
    return {
        "horizonte": horizon,
        "n_datas": len(clean),
        "ic_medio": mean,
        "ic_reversao": -mean,          # positivo = hipótese de reversão confirmada
        "ic_desvio": sd,
        "ic_ir": -mean / sd if sd > 0 else np.nan,
        "t_stat": -mean / sd * np.sqrt(len(clean)) if sd > 0 else np.nan,
        "pct_datas_favoraveis": float((clean < 0).mean()),
    }
