"""Orquestração do sinal: da view de mercado à tabela ranqueada de candidatos.

Encadeia bloco fatorial -> resíduo com betas congelados -> OU -> primeira
passagem, e devolve um DataFrame com uma linha por candidato e todas as métricas
que motivaram (ou barraram) a decisão. Essa tabela alimenta tanto a construção
de carteira quanto a justificativa de cada troca no dashboard — a banca vai
perguntar "por que esse nome entrou?", e a resposta precisa ser um número.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mare.backtest.asof import AsOfView
from mare.config import Config
from mare.features.returns import winsorize
from mare.signal import pca, passage
from mare.signal.ou import fit_ou
from mare.signal.residual import fit_residuals


def candidate_table(view: AsOfView, cfg: Config,
                    k_history: list[int] | None = None) -> pd.DataFrame:
    """Uma linha por nome elegível, com s-score, meia-vida, P(alvo) e mu_s."""
    eligible = view.eligible(
        min_dollar_volume=cfg.universe.min_dollar_volume,
        min_price=cfg.universe.min_price,
        min_valid_returns=cfg.universe.min_valid_returns,
        adv_window=cfg.universe.adv_window,
        history=min(cfg.factor.window, len(view.returns)),
    )
    if len(eligible) < 30:
        return _empty_table()

    history = view.tail(view.returns, cfg.factor.window)[eligible]
    clean = winsorize(history, cfg.sanity.winsorize_abs)
    clean = clean.loc[:, clean.notna().sum() >= cfg.universe.min_valid_returns]
    clean = clean.fillna(0.0)

    factors, _, k = pca.build_factors(
        clean, halflife=cfg.factor.halflife, k_method=cfg.factor.k_method,
        k_fixed=cfg.factor.k_fixed, k_min=cfg.factor.k_min, k_max=cfg.factor.k_max)
    if k_history is not None:
        k = pca.apply_hysteresis(k_history, k, cfg.factor.k_hysteresis_days)
        k_history.append(k)
        factors = factors.iloc[:, :k]

    fit = fit_residuals(clean, factors, beta_window=cfg.residual.beta_window,
                        beta_gap=cfg.residual.beta_gap,
                        resid_window=cfg.residual.resid_window)
    ou = fit_ou(fit.cumulative, kendall=cfg.ou.kendall_correction,
                shrinkage=cfg.ou.shrinkage,
                shrink_theta_to_zero=cfg.ou.shrink_theta_to_zero)

    level = fit.cumulative.iloc[-1]
    table = pd.DataFrame({
        "s_score": ou.s_score(level),
        "half_life": ou.half_life,
        "kappa": ou.kappa,
        "sigma_eq": ou.sigma_eq,
        "r2": fit.r2,
        "level": level,
    }).dropna(subset=["s_score"])

    table["k_factors"] = k
    table["as_of"] = view.as_of
    return _apply_gates(table, cfg)


def _empty_table() -> pd.DataFrame:
    columns = ["s_score", "half_life", "kappa", "sigma_eq", "r2", "level",
               "k_factors", "as_of", "passes_half_life", "passes_r2",
               "prob", "expected_days", "expected_return", "mu_s", "eligible"]
    return pd.DataFrame(columns=columns)


def _apply_gates(table: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Portões de validade do modelo e economia esperada do trade.

    Não há gate de ADF/variance-ratio de propósito: com 60 observações o poder
    contra meia-vida de 20 dias é ~10-20%, então o teste rejeita quase ao acaso
    e — pior — selecionar quem passa in-sample é selecionar ruído, o que infla
    o backtest em vez de protegê-lo.
    """
    table = table.copy()
    table["passes_half_life"] = table["half_life"].between(
        cfg.ou.halflife_min, cfg.ou.halflife_max)
    table["passes_r2"] = table["r2"] >= cfg.residual.min_r2
    table["cheap"] = table["s_score"] <= cfg.ou.entry_s

    table = _attach_passage(table, cfg)

    # Produção (passage desligada) seleciona pelo s-score, um nível normalizado
    # robusto. A ablação (script 04) mostrou que os gates derivados de kappa —
    # meia-vida, P(alvo), ranking por mu_s — pioram o retorno, então cada um só
    # entra em `eligible` sob a flag correspondente.
    gates = [table["cheap"], table["passes_r2"]]
    if cfg.ou.use_half_life_gate:
        gates.append(table["passes_half_life"])
    if cfg.passage.enabled:
        table["passes_prob"] = (table["prob"] >= cfg.passage.min_prob).fillna(False)
        gates.append(table["passes_prob"])
    else:
        table["passes_prob"] = True

    table["eligible"] = np.logical_and.reduce([g.to_numpy() for g in gates])
    rank_col = cfg.passage.rank_by if cfg.passage.enabled else "s_score"
    return table.sort_values(rank_col, ascending=(rank_col == "s_score"))


def _attach_passage(table: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Anexa P(alvo), E[T], retorno esperado e mu_s.

    Quando a máquina de primeira passagem está desligada (produção), as colunas
    existem como NaN para manter o schema estável — os scripts de diagnóstico
    ligam `passage.enabled` e recalculam. Isso também evita pagar ~100 quadraturas
    por rebalanceamento no backtest de produção, que é o que o tornava lento.
    """
    empty = {"prob": np.nan, "expected_days": np.nan,
             "expected_return": np.nan, "mu_s": np.nan}
    if not cfg.passage.enabled:
        return table.assign(**empty)

    cost = _round_trip_cost(cfg)
    econ = [
        passage.trade_economics(
            z0=row.s_score, z_stop=cfg.ou.stop_s, z_target=cfg.ou.exit_s,
            kappa=row.kappa, sigma_eq=row.sigma_eq, cost=cost,
            n_quad=cfg.passage.n_quad)
        if (row.cheap and np.isfinite(row.kappa) and row.kappa > 0
            and cfg.ou.stop_s < row.s_score < cfg.ou.exit_s)
        else dict(empty)
        for row in table.itertuples()
    ]
    return pd.concat([table, pd.DataFrame(econ, index=table.index)], axis=1)


def _round_trip_cost(cfg: Config) -> float:
    """Custo de ida e volta em fração, para entrar na economia esperada do trade."""
    scenario = cfg.costs.active
    one_way = (scenario.commission_bps + scenario.half_spread_bps) / 10_000
    return 2 * one_way


def rank_candidates(table: pd.DataFrame, n: int, rank_by: str = "s_score") -> pd.DataFrame:
    """Top-n entre os aprovados, ordenado pelo critério pedido.

    Default `s_score` (crescente = mais barato primeiro), que foi o vencedor da
    ablação. `mu_s` fica disponível para o exhibit comparativo.
    """
    if table.empty:
        return table
    approved = table.loc[table["eligible"]]
    if approved.empty:
        return approved
    column = rank_by if rank_by in approved.columns and \
        approved[rank_by].notna().any() else "s_score"
    return approved.sort_values(column, ascending=(column == "s_score")).head(n)
