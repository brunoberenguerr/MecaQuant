# -*- coding: utf-8 -*-
"""
Baixa os dados do backtest MecaQuant e salva em data/.

Uso:  python download_data.py

Requer internet: Yahoo Finance (preços diários ajustados) e
API SGS do Banco Central (CDI diário, série 12).
"""
import datetime as dt
import os

import pandas as pd
import requests
import yfinance as yf

TICKERS = ['^BVSP', 'SPY', 'IEF', 'BRL=X', 'EURUSD=X', 'JPY=X', 'GLD', 'DBC']
START = '2003-01-01'  # ~2 anos antes de 2005 para aquecer o sinal de 12m e a vol de 60d

os.makedirs('data', exist_ok=True)

# ---------- Preços diários ajustados (Yahoo Finance) ----------
px = yf.download(TICKERS, start=START, auto_adjust=True, progress=False)['Close']
px = px[TICKERS].dropna(how='all')
px.to_csv('data/prices.csv')
print(f"data/prices.csv — {len(px)} linhas ({px.index[0]:%Y-%m-%d} a {px.index[-1]:%Y-%m-%d})")
print("Cobertura por ativo (fração de dias com preço):")
print(px.notna().mean().round(3).to_string())


# ---------- CDI diário (série 12 do SGS/BCB, % ao dia) ----------
# A API limita séries diárias a 10 anos por chamada, então baixamos em blocos.
def sgs_cdi(inicio, fim):
    url = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
           f"?formato=json&dataInicial={inicio:%d/%m/%Y}&dataFinal={fim:%d/%m/%Y}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
    df['valor'] = df['valor'].astype(float)
    return df.set_index('data')['valor']


try:
    blocos, ini = [], dt.date(2003, 1, 1)
    hoje = dt.date.today()
    while ini <= hoje:
        fim = min(dt.date(ini.year + 8, 12, 31), hoje)
        blocos.append(sgs_cdi(ini, fim))
        ini = dt.date(fim.year + 1, 1, 1)
    cdi = pd.concat(blocos)
    cdi = cdi[~cdi.index.duplicated()].sort_index()
    cdi.rename('cdi_pct_dia').to_csv('data/cdi.csv')
    print(f"data/cdi.csv — {len(cdi)} linhas")
except Exception as e:
    print(f"AVISO: falha ao baixar o CDI ({e}).")
    print("O backtest usará uma taxa constante como aproximação (ver backtest.py).")
