# Registro de decisões — parâmetros, fontes e razões

Cada parâmetro do modelo está listado com seu valor, se é **fixado por citação**
(não tunado, não entra na contagem N do Deflated Sharpe) ou **livre** (tunado,
entra no N), e a razão econômica ou estatística. A fonte única e viva desses
valores é `config/base.yaml`; esta tabela é a justificativa.

O princípio: cada parâmetro fixado por citação é um parâmetro que não precisamos
defender e que reduz o N do teste de deflação. Por isso preferimos ancorar em
literatura sempre que possível.

## Universo

| Parâmetro | Valor | Tipo | Razão |
|---|---|---|---|
| Índice | S&P 500 | — | Profundidade do cross-section para os eigenportfolios; dados PIT reconstruíveis; fatores Ken French disponíveis. |
| Membership | PIT via change-log da Wikipedia | — | Reconstruído para trás a partir da lista atual. Evita viés de sobrevivência no *nome*. Cobertura de dados auditada (`data/interim/coverage.parquet`). |
| `min_dollar_volume` | US$ 20M (mediana 20d) | livre | Corte de liquidez; participação viável a % baixo do ADV. |
| `min_price` | US$ 5 (Close bruto) | citação | Convenção de mercado; abaixo disso o ruído de tick domina. Close **bruto**, não ajustado, para não reescrever a elegibilidade histórica. |
| `min_valid_returns` | 200 de 252 | livre | História suficiente para a regressão fatorial. |

## Sanidade de dados

| Parâmetro | Valor | Tipo | Razão |
|---|---|---|---|
| `winsorize_abs` | ±25% (antes da PCA) | livre | A estratégia compra quedas; tick ruim vira dinheiro de graça. Winsorizar limita sem descartar 2008/COVID reais. |
| `flag_abs` | ±50% | livre | Acima disso, auditoria manual — não é auto-corrigido. |

## Bloco fatorial

| Parâmetro | Valor | Tipo | Razão |
|---|---|---|---|
| `window` | 504 dias | citação/estatística | q = N/T. Com N≈450 e T=252, q≈1,8 > 1 e a covariância é singular. Com T=504, q≈0,9 e Marchenko–Pastur é bem-posto. |
| `halflife` | 126 dias | livre | Ponderação exponencial: mantém peso no dado recente sem perder o condicionamento. |
| Seleção de k | Marchenko–Pastur condicionado no modo de mercado | citação | Laloux et al. (1999); Bouchaud & Potters. **Sem parâmetro livre.** Condicionar no modo de mercado é obrigatório: ele detém 26–56% do traço e sua inclusão degenera a estimativa de σ² (medimos k=97–189 sem o condicionamento vs k=19–23 com). |
| `k_hysteresis_days` | 21 (mediana) | livre | Saltos em k mudam o span dos regressores e geram turnover puro-ruído. |

## Resíduo e OU

| Parâmetro | Valor | Tipo | Razão |
|---|---|---|---|
| Betas | estimados em [t−504, t−60], congelados | citação | Conserta o "bug da ponte": resíduo acumulado in-sample tem soma zero por construção do OLS e exibe reversão mecânica falsa. Resíduo OOS quebra a ponte. |
| `resid_window` | 60 dias | citação | Avellaneda & Lee (2010). |
| Correção de Kendall | ligada | citação | AR(1) tem viés −(1+3b)/T que superestima κ na direção que infla o backtest. |
| Shrinkage de κ | empirical-Bayes para o pooled | citação | ~3 ciclos independentes na janela → κ por nome quase não-identificado. |
| `entry_s` | −1,25 | **citação** | Avellaneda & Lee (2010). **Não tunado.** |
| `exit_s` | −0,50 | **citação** | Avellaneda & Lee (2010). **Não tunado.** |
| `stop_s` | −3,00 | livre | ~3σ do equilíbrio. |
| Gate de meia-vida | **desligado na seleção** | empírico | A ablação (script 04) mediu −0,30pp em 21d: a banda é função de κ, o parâmetro mais mal-identificado; filtrar por ela injeta ruído. Mantido só como gatilho de saída. |
| Gate ADF/variance-ratio | **cortado** | citação/empírico | Poder ~10–20% com 60 obs. Selecionar quem passa in-sample é selecionar ruído. |

## Primeira passagem (usada como exhibit, não em produção)

| Parâmetro | Valor | Tipo | Razão |
|---|---|---|---|
| Método | função de escala / medida de velocidade | citação | Feller. Exato para difusão 1-D. Substitui Bertram (Erfi/digamma), que resolve passagem unilateral sem stop — não é o nosso problema. |
| `passage.enabled` | **false em produção** | empírico | Gate de P(alvo) −0,19pp, ranking por μ_s −0,43pp (script 04). A máquina fica como exhibit de análise crítica, validada contra Monte Carlo (ver `test_passage.py`). |

## Carteira

| Parâmetro | Valor | Tipo | Razão |
|---|---|---|---|
| `n_names` | 40 (era 30) | livre | Curva de sensibilidade completa (`scripts/11_n_names_sensitivity.py`, grade n∈{10,15,20,25,30,35,40,50,60}, split design). Ver análise abaixo — o Sharpe é estatisticamente plano em toda a grade; a escolha de 40 vem de custo e vol, não de um pico. |
| `weighting` | vol inversa | citação | Sinal cross-sectional: sem normalizar por risco, a variância vem de quem é volátil, não do sinal. |
| `ewma_lambda` | 0,94 | **citação** | RiskMetrics. |
| `max_weight_mult` | 2/N por nome | livre | Evita concentração. |
| `max_sector_weight` | 25% | livre | Reversão residual concentra setor; sem teto vira aposta setorial. |
| `vol_target` | 10% a.a. | livre | Moreira & Muir (2017). |
| `max_leverage` | 1,0 | citação | Long-only honesto: só desaloca, nunca alavanca. |

### Análise completa da curva de sensibilidade de `n_names`

Motivação: um teste pontual anterior (n=10 vs n=30, ver §histórico) tinha sugerido
Sharpe ativo maior em n=10 (0,35 vs 0,19), mas sem curva completa nem teste de
significância — exatamente o tipo de "pico" que pode ser sobreajuste. Rodamos a
grade inteira (9 pontos, 9 backtests completos no design, cada um logado no
registry com hash próprio) e testamos se o pico é estatisticamente real.

**Resultado bruto (retorno total da carteira, split design):**

| n | Sharpe | Vol | CAGR | Turnover | Max DD |
|---|---|---|---|---|---|
| 10 | 0,623 | 0,235 | 14,4% | 0,586 | −49,9% |
| 15 | 0,580 | 0,222 | 12,7% | 0,522 | −48,9% |
| 20 | 0,613 | 0,208 | 12,9% | 0,486 | −49,4% |
| 25 | 0,595 | 0,205 | 12,3% | 0,450 | −48,2% |
| 30 | 0,588 | 0,207 | 12,2% | 0,429 | −51,6% |
| 35 | 0,585 | 0,207 | 12,2% | 0,414 | −51,9% |
| **40** | **0,614** | **0,204** | 12,7% | **0,396** | −49,7% |
| 50 | 0,619 | 0,197 | 12,5% | 0,377 | −50,6% |
| 60 | 0,606 | 0,195 | 12,1% | 0,361 | −51,0% |

**O teste de significância que muda a conclusão.** A curva NÃO é suave — oscila
sem padrão (0,623 → 0,580 → 0,613 → ...) — sinal clássico de ruído amostral, não
de um efeito real. Testamos com o erro-padrão assintótico de **Lo (2002)** para o
Sharpe ratio anualizado:

$$SE(SR_a) = \sqrt{252/T} \cdot \sqrt{1 + \tfrac{1}{2}SR_d^2}, \quad SR_d = SR_a/\sqrt{252}$$

Com T≈4.900 dias, isso dá **SE ≈ 0,227** — **16× maior** que a heurística ingênua
1/√T (≈0,014) que teria sido usada por engano. Comparando cada ponto contra o
pico (n=10) via z-score no erro conjunto: **todos os z ficam abaixo de 0,13**,
muito longe do limiar de 1,96 para 95% de confiança. **A grade inteira é
estatisticamente um único platô — não há pico defensável.**

**Critério de decisão, dado que Sharpe não decide.** Como o Sharpe é
estatisticamente plano em toda a grade, a escolha deve se apoiar em dimensões que
se movem de forma monotônica e não-ruidosa — porque são efeitos mecânicos de
portfólio, não estimativas de alpha:
- **Vol cai monotonicamente** de 0,235 (n=10) a 0,195 (n=60): diversificação
  genuína de risco idiossincrático, esperada por construção.
- **Turnover cai monotonicamente** de 0,586 a 0,361: menos giro relativo por
  rebalanceamento conforme o livro cresce — menos custo de transação real.

Essas duas dimensões favorecem **N maior**, não menor — o oposto do que o pico
ruidoso em n=10 sugeria à primeira vista. Isso também é coerente com o placebo
de seleção fina (§Backtest e validação): já mostramos que não há skill para
escolher "os melhores" dentro do bucket barato, então concentrar em poucos nomes
não compra nada — só assume mais risco idiossincrático e mais custo de giro.

**Por que 40 e não 50 ou 60.** n=50 e n=60 têm vol/turnover levemente melhores
ainda, mas o pool elegível diário tem mediana 87 e **mínimo 56** candidatos (nos
0,19% piores dias) — n=50 deixa margem de só 6 nomes nesse extremo, n=60 quase
zero. n=40 empata estatisticamente com o pico (z=0,03) e mantém folga
confortável contra o pior caso do universo elegível.

| Parâmetro | Valor | Tipo | Razão |
|---|---|---|---|
| Modelo | Statistical Jump Model, 2 estados | citação | Nystrup, Kolm & Lindström. Penalidade de salto dá persistência (menos whipsaw que HMM). |
| `jump_penalty` | 1,0 | livre | Loss normalizada por feature → λ invariante à dimensão. Calibrado em [0,6, 2,0], estável em real e sintético. |
| Inferência | filtro forward, online | citação | O Viterbi completo é smoother (look-ahead). O filtro forward usa só o passado — correto e ~10× mais rápido. |
| Padronização | expanding | citação | Padronizar com estatística full-sample vaza o futuro. |
| `bear_multiplier` | 0,5 | livre | Não zera: 0 viraria market timing puro. |

## Custos

| Parâmetro | Valor | Tipo | Razão |
|---|---|---|---|
| Base | 0,5 comissão + 2,0 meio-spread + impacto | livre | Impacto = 10 bps × √(participação), forma funcional padrão de execução. |
| Cenários | 2 / 5 / 15 bps one-way | livre | Reportar o **custo de break-even** é o exhibit central sobre custos. |

## Backtest e validação

| Parâmetro | Valor | Tipo | Razão |
|---|---|---|---|
| Rebalance | quarta, sinal até terça | citação | Lag explícito de 1 dia. |
| Design / holdout | 2007–2016 / 2017–2026 | — | Holdout intocado até o congelamento; pré-registrado pelo hash do config. |
| Walk-forward | 3a treino / 1a teste, purge+embargo 10d | citação | López de Prado. |
| DSR / PSR | Bailey & López de Prado | citação | N vem do run registry, não de estimativa. |
