"""Retornos e sanidade de dados.

Este módulo é o P0 mais inegociável do projeto. A estratégia compra quedas por
construção: um tick errado no Yahoo cria uma queda que nunca existiu e uma
"recuperação" no dia seguinte, e cada tick ruim vira dinheiro de graça no
backtest. Sem esta camada, o alpha medido é majoritariamente erro de dado.

A limpeza é conservadora de propósito: winsoriza (limita) em vez de descartar,
porque descartar retornos extremos verdadeiros — 2008, COVID — apagaria
justamente os episódios que a estratégia precisa enfrentar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Retornos aritméticos a partir do preço ajustado.

    Aritméticos e não logarítmicos para acompanhar Avellaneda & Lee (2010), cuja
    regressão fatorial e definição de resíduo são feitas em retorno simples.
    """
    return prices.sort_index().pct_change()


def winsorize(returns: pd.DataFrame, limit: float = 0.25) -> pd.DataFrame:
    """Limita retornos a ±`limit`. Aplicar ANTES da PCA.

    Sem isso, um outlier de dado contamina a matriz de covariância inteira e
    portanto todos os eigenportfolios, não só o nome afetado.
    """
    return returns.clip(lower=-limit, upper=limit)


def suspect_flags(returns: pd.DataFrame, limit: float = 0.50) -> pd.DataFrame:
    """Retornos grandes demais para passarem sem auditoria. Não corrige nada."""
    mask = returns.abs() > limit
    idx = np.where(mask.to_numpy())
    return pd.DataFrame({
        "date": returns.index[idx[0]],
        "symbol": returns.columns[idx[1]],
        "ret": returns.to_numpy()[idx],
    }).sort_values("ret", key=np.abs, ascending=False).reset_index(drop=True)


def reversal_suspects(returns: pd.DataFrame, move: float = 0.20,
                      tolerance: float = 0.03) -> pd.DataFrame:
    """Detecta a assinatura específica de tick ruim: salto grande imediatamente desfeito.

    Um preço errado num dia produz `r_t` grande e `r_{t+1}` de módulo parecido e
    sinal oposto, de modo que a composição dos dois volta a ~zero. Movimento
    econômico real raramente reverte com essa precisão em um dia.

    É o padrão mais perigoso para esta estratégia: é exatamente o que ela lucraria
    comprando, e o lucro seria inteiramente fictício.
    """
    nxt = returns.shift(-1)
    compounded = (1 + returns) * (1 + nxt) - 1
    mask = (returns.abs() > move) & (np.sign(returns) != np.sign(nxt)) & \
           (compounded.abs() < tolerance)
    idx = np.where(mask.to_numpy())
    return pd.DataFrame({
        "date": returns.index[idx[0]],
        "symbol": returns.columns[idx[1]],
        "ret": returns.to_numpy()[idx],
        "ret_next": nxt.to_numpy()[idx],
        "compounded": compounded.to_numpy()[idx],
    }).sort_values("ret", key=np.abs, ascending=False).reset_index(drop=True)


def sanity_summary(returns: pd.DataFrame, winsor: float = 0.25,
                   flag: float = 0.50) -> dict[str, float]:
    """Números para o relatório: quanto do dado foi tocado e por quê."""
    total = int(returns.notna().to_numpy().sum())
    clipped = int((returns.abs() > winsor).to_numpy().sum())
    flagged = int((returns.abs() > flag).to_numpy().sum())
    return {
        "observacoes": total,
        "winsorizadas": clipped,
        "pct_winsorizadas": clipped / total if total else 0.0,
        "flagradas": flagged,
        "reversoes_suspeitas": len(reversal_suspects(returns)),
    }
