"""Valida o detector de regime antes de integrá-lo ao backtest.

    python scripts/06_regime_check.py

Duas perguntas:
1. O estado de alto risco coincide com as crises conhecidas (2008, 2011, 2015-16)?
2. Condicionar a estratégia ao regime melhora a SÉRIE ATIVA? Se a tese está certa
   (edge só em calmaria), reduzir exposição no estado estressado deve elevar o
   Sharpe ativo — a validação econômica direta.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mare.backtest.metrics import sharpe, summary
from mare.config import load_config
from mare.io.market import load_bundle
from mare.portfolio import regime

warnings.filterwarnings("ignore")


def main() -> None:
    cfg = load_config()
    bundle = load_bundle(cfg)
    market = bundle.benchmark.dropna()

    feats = regime.regime_features(
        market, halflives=tuple(cfg.regime.feature_halflives),
        return_halflife=cfg.regime.return_halflife)
    std = regime.expanding_standardize(feats)
    print(f"MARÉ | regime | {len(std)} dias com features padronizadas")

    states = regime.online_regime(
        std, n_states=cfg.regime.n_states, jump_penalty=cfg.regime.jump_penalty,
        refit_every=cfg.regime.refit_every_days, lookback=cfg.regime.fit_lookback)
    _describe_states(states, market)
    _condition_strategy(states, bundle, cfg)


def _describe_states(states: pd.Series, market: pd.Series) -> None:
    print(f"\nfração de dias em estado estressado: {(states == 1).mean():.1%}")
    stressed = states[states == 1].index
    print("anos com mais dias estressados:")
    by_year = pd.Series(1, index=stressed).groupby(stressed.year).sum()
    for year, count in by_year.sort_values(ascending=False).head(6).items():
        print(f"  {year}: {count} dias")

    fwd = market.reindex(states.index)
    calm_vol = fwd[states == 0].std() * np.sqrt(252)
    stress_vol = fwd[states == 1].std() * np.sqrt(252)
    print(f"\nvol realizada do SPY | calmo: {calm_vol:.1%} | estressado: {stress_vol:.1%}")
    print("(sanidade: o estado estressado tem de ter vol MUITO maior)")


def _condition_strategy(states: pd.Series, bundle, cfg) -> None:
    """Reconstrói a série da estratégia a partir do registry seria ideal;

    aqui aproximamos aplicando o multiplicador de regime aos retornos do SPY
    como prova de conceito do overlay. A validação definitiva roda no backtest
    com o overlay integrado (ablação), mas este atalho já indica se o sinal de
    regime tem valor antes de gastar 15 minutos de backtest.
    """
    mult = regime.exposure_multiplier(states, bear_multiplier=cfg.regime.bear_multiplier)
    spy = bundle.benchmark.reindex(states.index).fillna(0.0)
    print("\nPROVA DE CONCEITO (aplicada ao SPY, não à estratégia):")
    base = summary(spy, label="SPY buy-and-hold")
    timed = summary(spy * mult.shift(1).fillna(1.0), label="SPY com overlay de regime")
    frame = pd.DataFrame([base, timed]).set_index("label")
    print(frame[["cagr", "vol", "sharpe", "max_drawdown"]].to_string(
        float_format=lambda v: f"{v:7.3f}"))
    print("\n(se o overlay reduz drawdown mantendo Sharpe, o detector tem sinal)")


if __name__ == "__main__":
    main()
