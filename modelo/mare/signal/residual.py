"""Resíduo idiossincrático com betas congelados.

O detalhe que decide se este projeto mede alpha ou artefato:

Rodar OLS com intercepto numa janela e acumular os resíduos DAQUELA MESMA janela
força Σε = 0 por construção do estimador. O processo acumulado X_t nasce em zero
e morre em zero — é um objeto do tipo ponte browniana. Ele exibe reversão à média
mecânica que não tem nada a ver com reversão negociável, e um κ estimado nele
mede a projeção, não o mercado. Pior: o s-score é avaliado no último ponto da
janela, exatamente onde o processo está preso em zero.

A correção é estimar os betas em [t-504, t-60] e calcular os resíduos
OUT-OF-SAMPLE em [t-60, t] com esses betas congelados. Custa 60 dias de defasagem
no beta — irrelevante para large caps — e devolve ao nível de X_t significado
econômico.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResidualFit:
    """Resíduos acumulados e diagnósticos da regressão fatorial."""

    cumulative: pd.DataFrame   # datas × símbolos, resíduo acumulado OOS
    residuals: pd.DataFrame    # datas × símbolos, resíduo diário OOS
    betas: pd.DataFrame        # fatores × símbolos
    r2: pd.Series              # R² da regressão na janela de estimação
    alpha: pd.Series           # intercepto (drift) estimado na janela


def _design(factors: pd.DataFrame) -> np.ndarray:
    """Matriz de desenho com intercepto."""
    return np.column_stack([np.ones(len(factors)), factors.to_numpy(dtype=float)])


def fit_residuals(returns: pd.DataFrame, factors: pd.DataFrame, *,
                  beta_window: int = 504, beta_gap: int = 60,
                  resid_window: int = 60) -> ResidualFit:
    """Estima betas na janela antiga e devolve resíduos acumulados na recente.

    `returns` e `factors` devem terminar na data do sinal. As janelas são
    contadas para trás a partir do fim: estimação em [-beta_window, -beta_gap),
    resíduos em [-resid_window, fim].
    """
    if len(returns) < beta_window:
        raise ValueError(f"histórico insuficiente: {len(returns)} < {beta_window}")

    est = slice(len(returns) - beta_window, len(returns) - beta_gap)
    out = slice(len(returns) - resid_window, len(returns))

    r_est = returns.iloc[est]
    f_est = factors.iloc[est]
    valid = r_est.notna().sum() >= 0.8 * len(r_est)
    r_est = r_est.loc[:, valid].fillna(0.0)

    design = _design(f_est)
    coef, *_ = np.linalg.lstsq(design, r_est.to_numpy(dtype=float), rcond=None)

    fitted = design @ coef
    resid_est = r_est.to_numpy(dtype=float) - fitted
    sse = (resid_est ** 2).sum(axis=0)
    sst = ((r_est.to_numpy(dtype=float) - r_est.to_numpy(dtype=float).mean(axis=0)) ** 2).sum(axis=0)
    r2 = pd.Series(np.where(sst > 0, 1 - sse / np.where(sst == 0, np.nan, sst), np.nan),
                   index=r_est.columns)

    r_out = returns.iloc[out].loc[:, r_est.columns].fillna(0.0)
    resid_out = r_out.to_numpy(dtype=float) - _design(factors.iloc[out]) @ coef
    residuals = pd.DataFrame(resid_out, index=r_out.index, columns=r_est.columns)

    return ResidualFit(
        cumulative=residuals.cumsum(),
        residuals=residuals,
        betas=pd.DataFrame(coef[1:], index=factors.columns, columns=r_est.columns),
        r2=r2,
        alpha=pd.Series(coef[0], index=r_est.columns),
    )


def fit_residuals_in_sample(returns: pd.DataFrame, factors: pd.DataFrame,
                            window: int = 60) -> pd.DataFrame:
    """Versão ingênua (in-sample) — existe SÓ para o exhibit comparativo.

    Não usar em produção. Serve para mostrar no relatório que o resíduo
    acumulado in-sample é uma ponte: começa e termina em zero, e o s-score
    resultante tem distribuição visivelmente distinta da versão correta.
    """
    r = returns.iloc[-window:].fillna(0.0)
    f = factors.iloc[-window:]
    design = _design(f)
    coef, *_ = np.linalg.lstsq(design, r.to_numpy(dtype=float), rcond=None)
    resid = r.to_numpy(dtype=float) - design @ coef
    return pd.DataFrame(resid, index=r.index, columns=r.columns).cumsum()
