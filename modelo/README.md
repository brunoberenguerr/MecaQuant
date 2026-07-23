# MARÉ — Mean-reversion Adaptive Residual Engine

Estratégia quantitativa de reversão à média sobre retornos **residuais** de ações
do S&P 500, para o Itaú Quant AI Challenge 2026.

> A parte sistemática do retorno (fatores de mercado) é removida por decomposição
> espectral; a parte idiossincrática, acumulada, é modelada como um processo de
> Ornstein–Uhlenbeck e negociada quando se afasta do equilíbrio. Uma camada de
> gestão de risco (vol targeting + regime) protege a jornada.

## Como está organizado

```
mare/
  config.py, calendar.py, registry.py   # configuração, cronograma, log de runs
  io/          # universo PIT, preços, fatores, macro, montagem do MarketData
  features/    # retornos, sanidade, liquidez, volatilidade
  signal/      # PCA/RMT, resíduo, OU, primeira passagem, score
  portfolio/   # sizing, livro de posições, vol target, regime, construção
  backtest/    # engine com AsOfView, custos, métricas, overlays
  validation/  # DSR, placebo, atribuição FF5, survivorship, qualidade de sinal
  report/      # figuras, tabelas, dashboard
scripts/       # 00..09, pipeline ponta a ponta
tests/         # unit + integration (look-ahead, determinismo)
docs/          # relatório, decisões, log de IA, identidade
config/base.yaml   # ÚNICA fonte de parâmetros
```

## Reproduzir

```bash
python -m venv .venv && .venv/Scripts/activate    # Windows
pip install -e .

python scripts/01_build_universe.py    # universo PIT + preços + auditoria
python scripts/02_run_backtest.py --split design
python scripts/03_signal_quality.py    # IC e qualidade do sinal
python scripts/04_gate_ablation.py     # por que o sinal simples vence
python scripts/07_validate.py --split design --dashboard
python scripts/08_robustness.py --split design
python scripts/10_naive_baseline.py --split design   # residuo vs reversao crua
python scripts/11_n_names_sensitivity.py --split design   # curva de n_names
```

O dashboard sai em `reports/design/dashboard.html` (autocontido, abre offline).

## Os achados que definem o trabalho

1. **O sinal tem edge in-sample mas NÃO sobrevive ao out-of-sample.** No design
   (2007–2016) bate o SPY e o placebo (percentil 88); no holdout pré-registrado
   (2017–2026) perde para o SPY, com alpha ativo negativo. Causa econômica limpa:
   o edge vive em alta dispersão, e a dispersão colapsou pós-2016. Reportamos isso
   de frente — é o que o edital pede.
2. **A gestão de risco generaliza; o alpha de seleção não.** O vol target corta o
   drawdown pela metade nos dois períodos. É o componente entregável.
3. **O aparato matemático sofisticado não paga.** A ablação (`scripts/04`) mostra
   que gates derivados de κ pioram o retorno. Operamos o sinal simples; a máquina
   OU/primeira-passagem vira exhibit de análise crítica, validada contra Monte
   Carlo. Entender o limite do modelo é o resultado.
4. **Look-ahead é impossível por construção**, não por disciplina. O engine só
   entrega ao sinal uma view fatiada no passado; o teste apaga o futuro e exige
   pesos bit-idênticos.

## Rigor de backtest

- Universo point-in-time (sem viés de sobrevivência no nome); auditoria de
  cobertura de dados reportada.
- Custos de transação com custo de break-even; Deflated Sharpe com N do registry;
  placebo de seleção aleatória; walk-forward purgado; holdout 2017+ pré-registrado.

Ver `docs/relatorio.md` para o relatório completo e `docs/decisions.md` para a
justificativa de cada parâmetro.
