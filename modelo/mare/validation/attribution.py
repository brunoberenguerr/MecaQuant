"""Atribuição de fatores contra Fama–French 5 + momentum.

A pergunta que a banca vai fazer sobre uma carteira long-only de ações baratas:
"isso é alpha ou é só exposição a valor, tamanho e beta?". A resposta tem de ser
uma regressão, não uma opinião.

Decisão importante: a regressão roda sobre a série ATIVA (estratégia − SPY), não
sobre a total. Regredir a total contra fatores devolveria um beta de mercado
próximo de 1 com t-stat enorme e um alpha que ninguém consegue ler. A série
ativa é onde o alpha, se existir, aparece limpo.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

FF5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
MOM_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Momentum_Factor_daily_CSV.zip")
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MARE-research/0.1)"}


def _read_french_zip(url: str) -> pd.DataFrame:
    blob = requests.get(url, headers=_HEADERS, timeout=120).content
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        name = archive.namelist()[0]
        text = archive.read(name).decode("latin-1")

    lines = [ln for ln in text.splitlines() if _is_data_row(ln)]
    frame = pd.read_csv(io.StringIO("\n".join(lines)), header=None)
    # Os arquivos da Ken French têm vírgula final (o momentum vira 3 colunas em
    # vez de 2). Descarta colunas totalmente vazias antes de qualquer coisa.
    frame = frame.dropna(axis=1, how="all")
    frame = frame.rename(columns={frame.columns[0]: "date"})
    frame["date"] = pd.to_datetime(frame["date"].astype(int).astype(str),
                                   format="%Y%m%d")
    return frame.set_index("date").astype(float) / 100.0


def _is_data_row(line: str) -> bool:
    head = line.split(",")[0].strip()
    return len(head) == 8 and head.isdigit()


def load_factors(cache_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """FF5 + MOM diários, em fração. Colunas: mkt_rf, smb, hml, rmw, cma, mom, rf."""
    path = Path(cache_dir) / "ff5_mom_daily.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    ff5 = _read_french_zip(FF5_URL)
    ff5.columns = ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"][:ff5.shape[1]]
    mom = _read_french_zip(MOM_URL)
    mom = mom.iloc[:, [0]]                     # só a coluna do fator momentum
    mom.columns = ["mom"]

    out = ff5.join(mom, how="inner")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return out


def regress(returns: pd.Series, factors: pd.DataFrame,
            columns: list[str] | None = None,
            subtract_rf: bool = False) -> pd.DataFrame:
    """OLS com t-stats de Newey–West e alpha anualizado.

    `subtract_rf=False` é o padrão porque a série ativa já é uma diferença de
    retornos — subtrair a taxa livre de risco dela seria contá-la duas vezes.
    """
    columns = columns or ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
    joined = pd.concat([returns.rename("y"), factors[columns + ["rf"]]],
                       axis=1).dropna()
    if len(joined) < 60:
        return pd.DataFrame()

    y = (joined["y"] - joined["rf"]).to_numpy() if subtract_rf else joined["y"].to_numpy()
    x = np.column_stack([np.ones(len(joined)), joined[columns].to_numpy()])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef

    se = _newey_west_se(x, resid)
    tstat = coef / np.where(se > 0, se, np.nan)
    names = ["alpha", *columns]
    return pd.DataFrame({
        "coef": coef,
        "coef_anual": [coef[0] * 252, *coef[1:]],
        "std_err": se,
        "t_stat": tstat,
        "p_valor": 2 * (1 - stats.norm.cdf(np.abs(tstat))),
    }, index=names)


def _newey_west_se(x: np.ndarray, resid: np.ndarray,
                   lags: int | None = None) -> np.ndarray:
    """Erros-padrão robustos a heterocedasticidade e autocorrelação."""
    n, k = x.shape
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    xtx_inv = np.linalg.pinv(x.T @ x)

    weighted = x * resid[:, None]
    meat = weighted.T @ weighted
    for lag in range(1, lags + 1):
        gamma = weighted[lag:].T @ weighted[:-lag]
        meat += (1 - lag / (lags + 1)) * (gamma + gamma.T)

    cov = xtx_inv @ meat @ xtx_inv
    return np.sqrt(np.clip(np.diag(cov), 0, None))


def r_squared(returns: pd.Series, factors: pd.DataFrame,
              columns: list[str] | None = None) -> float:
    """Quanto da variação da série é explicada pelos fatores conhecidos."""
    columns = columns or ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
    joined = pd.concat([returns.rename("y"), factors[columns]], axis=1).dropna()
    if len(joined) < 60:
        return np.nan
    y = joined["y"].to_numpy()
    x = np.column_stack([np.ones(len(joined)), joined[columns].to_numpy()])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    sst = ((y - y.mean()) ** 2).sum()
    return float(1 - (resid ** 2).sum() / sst) if sst > 0 else np.nan
