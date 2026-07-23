"""Baseline de reversão à média SEM residualização — sobre o retorno bruto.

Pergunta que este módulo responde: o trabalho de decompor o retorno em fator
sistemático + resíduo (PCA/RMT, betas congelados) agrega algo, ou uma reversão
simples sobre o preço já captura o mesmo edge?

Mesma máquina de OU/s-score do sinal principal (`score.py`), aplicada
diretamente à soma acumulada do retorno BRUTO na janela de 60 dias — sem
remover fatores. Se este baseline performar igual ou melhor, todo o aparato de
decomposição espectral (Pilar 1) é injustificado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mare.backtest.asof import AsOfView
from mare.config import Config
from mare.features.returns import winsorize
from mare.signal.ou import fit_ou


def candidate_table_naive(view: AsOfView, cfg: Config) -> pd.DataFrame:
    """Uma linha por nome elegível, s-score sobre retorno acumulado BRUTO.

    Espelha `score.candidate_table`, mas pula PCA e residualização: usa a soma
    acumulada do retorno bruto na mesma janela (`resid_window`) como a série a
    que o OU é ajustado.
    """
    eligible = view.eligible(
        min_dollar_volume=cfg.universe.min_dollar_volume,
        min_price=cfg.universe.min_price,
        min_valid_returns=cfg.universe.min_valid_returns,
        adv_window=cfg.universe.adv_window,
        history=min(cfg.factor.window, len(view.returns)),
    )
    if len(eligible) < 30:
        return _empty_table()

    window = view.tail(view.returns, cfg.residual.resid_window)[eligible]
    clean = winsorize(window, cfg.sanity.winsorize_abs)
    clean = clean.loc[:, clean.notna().sum() >= cfg.residual.resid_window // 2]
    clean = clean.fillna(0.0)

    cumulative = clean.cumsum()
    ou = fit_ou(cumulative, kendall=cfg.ou.kendall_correction,
               shrinkage=cfg.ou.shrinkage,
               shrink_theta_to_zero=cfg.ou.shrink_theta_to_zero)

    level = cumulative.iloc[-1]
    table = pd.DataFrame({
        "s_score": ou.s_score(level),
        "half_life": ou.half_life,
        "kappa": ou.kappa,
        "sigma_eq": ou.sigma_eq,
        "level": level,
    }).dropna(subset=["s_score"])

    table["as_of"] = view.as_of
    table["cheap"] = table["s_score"] <= cfg.ou.entry_s
    table["eligible"] = table["cheap"]
    return table.sort_values("s_score", ascending=True)


def _empty_table() -> pd.DataFrame:
    columns = ["s_score", "half_life", "kappa", "sigma_eq", "level", "as_of",
              "cheap", "eligible"]
    return pd.DataFrame(columns=columns)
