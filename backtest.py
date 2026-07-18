# -*- coding: utf-8 -*-
"""
Backtest v1.1 — Time-Series Momentum multi-ativo com volatility targeting.

Especificação completa em CLAUDE.md. Parâmetros canônicos da literatura
(Moskowitz, Ooi & Pedersen 2012) — NÃO otimizar por grid search.

Uso:  python backtest.py        (requer data/prices.csv; ver download_data.py)

Saídas em results/:
  metricas.csv, retornos_anuais.csv, serie_diaria.csv,
  equity_curve.png, drawdown.png, exposicao.png, retornos_anuais.png

Anti-look-ahead: o sinal do mês usa preços só até o fim do mês anterior;
a vol usa dados até o último pregão do mês; os pesos valem a partir do
pregão SEGUINTE ao cálculo (retorno começa a contar no fechamento seguinte).

v1.1: caixa remunerado a CDI — modelo padrão de implementação via futuros
(margem aplicada em caixa): retorno total = CDI + PnL do overlay - custos.
No v1 o caixa rendia zero, o que subestimava a estratégia contra um
benchmark de caixa de ~11%% a.a. no Brasil.

Simplificações documentadas (v1):
  - Retornos de preço em moeda local por série (sem conversão cambial do PnL);
    padrão dos papers de TSMOM, que usam retornos em excesso de futuros.
  - Sem custo de financiamento da alavancagem (trava em ALAV_MAX compensa).
  - Turnover medido como soma |peso novo - peso anterior| no rebalanceamento.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------- Parâmetros (canônicos — não otimizar) ----------------
TICKERS    = ['^BVSP', 'SPY', 'IEF', 'BRL=X', 'EURUSD=X', 'JPY=X', 'GLD', 'DBC']
INICIO     = '2005-01-01'  # início da janela de avaliação
JANELA_VOL = 60            # dias úteis para vol e covariância
VOL_ALVO   = 0.10          # vol alvo do portfólio (a.a.)
ALAV_MAX   = 3.0           # trava de alavancagem bruta (soma dos |pesos|)
CUSTO      = 0.001         # 0,1% sobre o notional negociado
CDI_CONST  = 0.11          # CDI ~11% a.a. (média 2005-2025), só se faltar data/cdi.csv
ANN        = 252

# Paleta dos gráficos (validada p/ daltonismo; fundo claro p/ o relatório)
COR = {'Estratégia': '#2a78d6', 'Ibovespa': '#008300', 'CDI': '#e87ba4'}
COR_ATIVOS = ['#2a78d6', '#008300', '#e87ba4', '#eda100',
              '#1baf7a', '#eb6834', '#4a3aa7', '#e34948']
SURFACE, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'text.color': INK, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.edgecolor': '#d0cfc9', 'axes.grid': True,
    'grid.color': '#e6e5df', 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 10, 'axes.titlesize': 11,
})


def carregar_precos():
    px = pd.read_csv('data/prices.csv', index_col=0, parse_dates=True)[TICKERS]
    # Séries de FX do Yahoo têm buracos: forward-fill limitado a 5 pregões
    return px.ffill(limit=5)


def carregar_cdi(index):
    """Retorno diário do CDI alinhado ao índice de datas do backtest."""
    if os.path.exists('data/cdi.csv'):
        cdi = pd.read_csv('data/cdi.csv', index_col=0, parse_dates=True).iloc[:, 0]
        return (cdi / 100.0).reindex(index).fillna(0.0), 'CDI (SGS 12/BCB)'
    r_dia = (1 + CDI_CONST) ** (1 / ANN) - 1
    return pd.Series(r_dia, index=index), f'CDI aproximado constante ({CDI_CONST:.0%} a.a.)'


def construir_pesos(px):
    """Pesos diários da estratégia + custos de transação nos rebalanceamentos."""
    rets = px.pct_change()
    vol = rets.rolling(JANELA_VOL, min_periods=40).std() * np.sqrt(ANN)

    meses = px.index.to_period('M')
    pxm = px.groupby(meses).last()                       # fechamento mensal
    mom = pxm.shift(1) / pxm.shift(12) - 1               # 12-1: fim de t-12 a fim de t-1
    sinal = np.sign(mom).fillna(0.0)                     # +1 comprado / -1 vendido / 0 sem histórico

    ult_pregao = pd.Series(px.index, index=meses).groupby(level=0).last()

    W = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    custos = pd.Series(0.0, index=px.index)
    hist = []  # (mês, pesos, alavancagem) para o gráfico de exposição
    w_ant = pd.Series(0.0, index=px.columns)

    lista_meses = list(pxm.index)
    for i, m in enumerate(lista_meses[:-1]):
        dt_rebal = ult_pregao[m]
        s, v = sinal.loc[m], vol.loc[dt_rebal]
        bruto = (s / v).where(v.notna() & (v > 0), 0.0)

        ativos = bruto.index[bruto != 0]                 # só quem tem sinal e vol válida
        if len(ativos) == 0:
            w = bruto * 0.0
        else:
            w_norm = bruto / bruto.abs().sum()           # peso ∝ sinal / vol
            cov = (rets[ativos].loc[:dt_rebal].tail(JANELA_VOL).cov() * ANN).fillna(0.0)
            wa = w_norm[ativos]
            vol_port = float(np.sqrt(max(wa @ cov @ wa, 1e-12)))
            alav = min(VOL_ALVO / vol_port, ALAV_MAX)    # escala p/ vol alvo 10% a.a.
            w = w_norm * alav

        prox = lista_meses[i + 1]
        dias = px.index[meses == prox]
        if len(dias) == 0:
            continue
        W.loc[dias] = w.values
        custos.loc[dias[0]] = CUSTO * (w - w_ant).abs().sum()
        hist.append((prox.to_timestamp('M'), w.copy(), w.abs().sum()))
        w_ant = w

    return W, custos, hist


def metricas(r, cdi_d):
    r = r.dropna()
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (ANN / len(r)) - 1
    vol = r.std() * np.sqrt(ANN)
    exc = r - cdi_d.reindex(r.index).fillna(0.0)
    sharpe = exc.mean() / exc.std() * np.sqrt(ANN) if exc.std() > 0 else np.nan
    dd = eq / eq.cummax() - 1
    por_ano = (1 + r).groupby(r.index.year).prod() - 1
    return {
        'CAGR': cagr,
        'Vol a.a.': vol,
        'Sharpe (exc. CDI)': sharpe,
        'Max drawdown': dd.min(),
        'Pior ano': por_ano.min(),
        'Ano do pior': int(por_ano.idxmin()),
        'Retorno 2008': por_ano.get(2008, np.nan),
        'Retorno 2020': por_ano.get(2020, np.nan),
    }, eq, dd, por_ano


def rotulo_direto(ax, x, y, texto, cor):
    ax.annotate(texto, (x, y), xytext=(6, 0), textcoords='offset points',
                color=cor, fontsize=9, fontweight='bold', va='center')


def graficos(series, dds, anuais, hist, nome_cdi):
    os.makedirs('results', exist_ok=True)

    # --- Curva de patrimônio (escala log, base 100) ---
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for nome, eq in series.items():
        base100 = 100 * eq / eq.iloc[0]
        ax.plot(base100.index, base100.values, color=COR[nome], lw=1.8, label=nome)
        rotulo_direto(ax, base100.index[-1], base100.iloc[-1], nome, COR[nome])
    ax.set_yscale('log')
    ax.set_title('Curva de patrimônio — base 100, escala log')
    ax.set_ylabel('Patrimônio (base 100)')
    ax.legend(frameon=False, loc='upper left')
    ax.grid(axis='x', visible=False)
    ax.margins(x=0.08)
    fig.tight_layout()
    fig.savefig('results/equity_curve.png', dpi=160)
    plt.close(fig)

    # --- Drawdown ---
    fig, ax = plt.subplots(figsize=(10, 4))
    dd_e, dd_i = dds['Estratégia'], dds['Ibovespa']
    ax.fill_between(dd_e.index, dd_e.values, 0, color=COR['Estratégia'], alpha=0.35, lw=0)
    ax.plot(dd_e.index, dd_e.values, color=COR['Estratégia'], lw=1.5, label='Estratégia')
    ax.plot(dd_i.index, dd_i.values, color=COR['Ibovespa'], lw=1.2, label='Ibovespa')
    ax.set_title('Drawdown')
    ax.yaxis.set_major_formatter(lambda v, _: f'{v:.0%}')
    ax.legend(frameon=False, loc='lower left')
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig('results/drawdown.png', dpi=160)
    plt.close(fig)

    # --- Retornos anuais (barras agrupadas) ---
    fig, ax = plt.subplots(figsize=(11, 4.5))
    nomes = list(anuais.columns)
    x = np.arange(len(anuais.index))
    larg = 0.27
    for k, nome in enumerate(nomes):
        ax.bar(x + (k - 1) * larg, anuais[nome].values, width=larg * 0.92,
               color=COR[nome], label=nome)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(anuais.index, rotation=45, fontsize=8)
    ax.yaxis.set_major_formatter(lambda v, _: f'{v:.0%}')
    ax.set_title('Retornos por ano')
    ax.legend(frameon=False, loc='upper right', ncols=3)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig('results/retornos_anuais.png', dpi=160)
    plt.close(fig)

    # --- Exposição por ativo + alavancagem bruta ---
    datas = [h[0] for h in hist]
    pesos = pd.DataFrame([h[1] for h in hist], index=datas)
    alav = pd.Series([h[2] for h in hist], index=datas)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   height_ratios=[2.2, 1])
    for k, t in enumerate(pesos.columns):
        ax1.plot(pesos.index, pesos[t].values, color=COR_ATIVOS[k], lw=1.4, label=t)
    ax1.axhline(0, color=INK2, lw=0.8)
    ax1.set_title('Pesos por ativo no rebalanceamento mensal')
    ax1.legend(frameon=False, ncols=4, fontsize=8, loc='upper left')
    ax1.grid(axis='x', visible=False)
    ax2.plot(alav.index, alav.values, color=COR['Estratégia'], lw=1.8)
    ax2.set_title('Alavancagem bruta (soma dos |pesos|)')
    ax2.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig('results/exposicao.png', dpi=160)
    plt.close(fig)


def main():
    px = carregar_precos()
    rets = px.pct_change().fillna(0.0)
    W, custos, hist = construir_pesos(px)

    r_estrategia = ((W * rets).sum(axis=1) - custos).loc[INICIO:]
    cdi_d, nome_cdi = carregar_cdi(r_estrategia.index)
    r_estrategia = r_estrategia + cdi_d  # v1.1: caixa a CDI (futuros/margem)
    r_ibov = rets['^BVSP'].loc[r_estrategia.index]
    r_cdi = cdi_d

    os.makedirs('results', exist_ok=True)
    tabela, series, dds, anuais = {}, {}, {}, {}
    for nome, r in [('Estratégia', r_estrategia), ('Ibovespa', r_ibov), ('CDI', r_cdi)]:
        m, eq, dd, por_ano = metricas(r, cdi_d)
        tabela[nome] = m
        series[nome], dds[nome], anuais[nome] = eq, dd, por_ano

    df_metricas = pd.DataFrame(tabela).T
    df_anuais = pd.DataFrame(anuais)
    df_metricas.to_csv('results/metricas.csv')
    df_anuais.to_csv('results/retornos_anuais.csv')
    pd.DataFrame({'retorno': r_estrategia,
                  'equity': series['Estratégia']}).to_csv('results/serie_diaria.csv')

    graficos(series, dds, df_anuais, hist, nome_cdi)

    fmt = df_metricas.copy()
    fmt['Ano do pior'] = fmt['Ano do pior'].astype(int)
    for c in fmt.columns:
        if c == 'Ano do pior':
            continue
        fmt[c] = fmt[c].map(lambda v: f'{v:.2%}' if abs(v) < 3 else f'{v:.2f}')
    fmt['Sharpe (exc. CDI)'] = df_metricas['Sharpe (exc. CDI)'].map(lambda v: f'{v:.2f}')
    print(f"Período: {r_estrategia.index[0]:%Y-%m-%d} a {r_estrategia.index[-1]:%Y-%m-%d}"
          f" | Benchmark de caixa: {nome_cdi}")
    print(fmt.to_string())

    sharpe = df_metricas.loc['Estratégia', 'Sharpe (exc. CDI)']
    if sharpe > 1.5:
        print("\n*** ALERTA DE SANIDADE: Sharpe > 1.5 — provável bug ou viés. "
              "Investigar antes de aceitar (ver checklist no CLAUDE.md). ***")
    elif sharpe < 0.3:
        print("\nAviso: Sharpe abaixo do esperado (0.6-1.0). Conferir dados e sinais.")
    print("\nGráficos e tabelas salvos em results/")


if __name__ == '__main__':
    main()
