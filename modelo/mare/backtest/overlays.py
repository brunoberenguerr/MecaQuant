"""Overlays de risco aplicados sobre a série da estratégia base.

Regime e vol-target são multiplicadores de EXPOSIÇÃO — não mudam quais nomes a
estratégia compra, só quanto capital aloca. Isso permite calculá-los sobre a
série de retornos já produzida pelo backtest base, o que tem três vantagens:
a ablação sai de um único backtest; a atribuição fica limpa (dá para ver o efeito
isolado de cada overlay); e o feedback do vol-target sobre a própria vol da
estratégia é tratado de forma consistente.

Toda escala é aplicada com defasagem de um dia — a exposição de hoje usa a
informação de ontem. Quando a exposição cai abaixo de 1, o capital liberado rende
a taxa livre de risco, exatamente como o engine trata o caixa. Ignorar isso
subestimaria o retorno das versões com overlay.
"""

from __future__ import annotations

import pandas as pd

from mare.portfolio.voltarget import apply_vol_target


def _scaled(strategy: pd.Series, scale: pd.Series,
            risk_free: pd.Series | None) -> pd.Series:
    """Aplica a escala com defasagem; o caixa liberado rende rf."""
    lagged = scale.reindex(strategy.index).shift(1).fillna(1.0).clip(0.0, 1.0)
    invested = strategy * lagged
    if risk_free is None:
        return invested
    cash = (1.0 - lagged) * risk_free.reindex(strategy.index).fillna(0.0)
    return invested + cash


def regime_scaled(strategy: pd.Series, regime_multiplier: pd.Series,
                  risk_free: pd.Series | None = None) -> pd.Series:
    """Estratégia com overlay de regime (reduz exposição no estado estressado)."""
    return _scaled(strategy, regime_multiplier, risk_free)


def vol_target_scaled(strategy: pd.Series, target_vol: float, window: int = 21,
                      max_leverage: float = 1.0,
                      risk_free: pd.Series | None = None) -> pd.Series:
    """Estratégia com vol-target (mira volatilidade constante, só desaloca)."""
    scale = apply_vol_target(strategy, target_vol, window, max_leverage)
    return _scaled(strategy, scale, risk_free)


def combined_scaled(strategy: pd.Series, regime_multiplier: pd.Series,
                    target_vol: float, window: int = 21, max_leverage: float = 1.0,
                    risk_free: pd.Series | None = None) -> pd.Series:
    """Os dois overlays juntos: multiplicadores compostos, um único caixa residual."""
    vol_scale = apply_vol_target(strategy, target_vol, window, max_leverage)
    regime = regime_multiplier.reindex(strategy.index).fillna(1.0)
    return _scaled(strategy, vol_scale * regime, risk_free)


def ablation(strategy: pd.Series, regime_multiplier: pd.Series, target_vol: float,
             window: int = 21, max_leverage: float = 1.0,
             risk_free: pd.Series | None = None) -> pd.DataFrame:
    """As quatro séries da tabela de ablação do Pilar 2.

    É o entregável central da análise de risco: mostra o efeito isolado de cada
    camada e das duas juntas, para a banca ver o que cada uma fez — em vez de um
    número final impossível de atribuir.
    """
    return pd.DataFrame({
        "base": strategy,
        "vol_target": vol_target_scaled(strategy, target_vol, window,
                                        max_leverage, risk_free),
        "regime": regime_scaled(strategy, regime_multiplier, risk_free),
        "regime+vol_target": combined_scaled(strategy, regime_multiplier, target_vol,
                                             window, max_leverage, risk_free),
    }).dropna(how="all")
