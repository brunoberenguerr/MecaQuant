"""Volatility targeting no nível de portfólio (Moreira & Muir 2017).

Escala a exposição para mirar uma volatilidade constante: reduz quando a vol
realizada recente está alta, mantém quando está baixa. O racional de Moreira &
Muir é que mudanças na volatilidade não são compensadas por mudanças
proporcionais no retorno esperado, então mirar vol constante melhora o Sharpe.

Restrição dura: alavancagem máxima 1.0. A estratégia é long-only e só DESALOCA —
nunca toma emprestado para amplificar em vol baixa. Isso torna o vol target uma
ferramenta de redução de risco, não de amplificação, e mantém o produto honesto
com o mandato long-only.

A vol usada é a realizada até a data do sinal (sem look-ahead). A previsão para o
próximo período assume persistência — razoável para vol, que é muito autocorrelada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def realized_portfolio_vol(returns: pd.Series, window: int = 21) -> float:
    """Vol anualizada dos últimos `window` retornos do portfólio."""
    recent = returns.dropna().iloc[-window:]
    if len(recent) < max(5, window // 2):
        return np.nan
    return float(recent.std(ddof=1) * np.sqrt(TRADING_DAYS))


def target_scale(current_vol: float, target_vol: float,
                 max_leverage: float = 1.0) -> float:
    """Fator de escala = alvo / realizada, limitado a [0, max_leverage]."""
    if not np.isfinite(current_vol) or current_vol <= 0:
        return 1.0
    return float(np.clip(target_vol / current_vol, 0.0, max_leverage))


def apply_vol_target(portfolio_returns: pd.Series, target_vol: float,
                     window: int = 21, max_leverage: float = 1.0) -> pd.Series:
    """Série de fatores de escala, um por dia, aplicável com defasagem de 1 dia.

    Trabalha sobre a série de retornos JÁ REALIZADA da estratégia (ex-post), para
    o estudo de ablação. No backtest ao vivo, o mesmo cálculo é feito dentro do
    engine com a vol até a data corrente, e por isso o resultado deve ser
    aplicado com `.shift(1)` — a escala de hoje usa a vol de ontem.
    """
    rolling = portfolio_returns.rolling(window, min_periods=max(5, window // 2))
    vol = rolling.std() * np.sqrt(TRADING_DAYS)
    scale = (target_vol / vol).clip(0.0, max_leverage)
    return scale.fillna(1.0)
