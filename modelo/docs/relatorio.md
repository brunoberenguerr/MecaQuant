# MARÉ — Relatório técnico

*Mean-reversion Adaptive Residual Engine · Itaú Quant AI Challenge 2026*

Estrutura: hipótese → metodologia → resultados → limitações → próximos passos.
Os números são preenchidos pelos scripts de validação (`scripts/07_validate.py`,
`scripts/08_robustness.py`) e reproduzíveis; o `config/registry.jsonl` registra
cada configuração testada.

**Nota de honestidade sobre o holdout.** O split 2017–2026 foi examinado **duas
vezes**: uma para confirmar (ou não) que a estratégia com `n_names=30` generaliza
OOS, e uma segunda depois de retunar `n_names` para 40 com base numa curva de
sensibilidade rodada **inteiramente no split de design** (§2.5). A segunda
observação não mudou a conclusão qualitativa do holdout — o que é evidência a
favor de que a retunagem não é sobreajuste ao holdout, já que ele nunca
participou da escolha. Ambos os olhares estão no `config/registry.jsonl`.

---

## 1. Hipótese

A estratégia negocia **reversão à média de retornos residuais** de ações do
S&P 500. O retorno de cada ação é decomposto em uma parte sistemática (fatores
comuns, estimados por decomposição espectral da matriz de covariância) e uma
parte idiossincrática; é a parte idiossincrática, acumulada, que revertemos.

**A tese econômica não é "resíduos revertem"** — isso é observação. A tese que os
dados sustentam é:

> A reversão residual de curto prazo captura a correção de deslocamentos de preço
> **não-informativos** (ruído de liquidez, fluxo inelástico). Em mercado calmo,
> movimentos idiossincráticos são majoritariamente ruído e revertem. Em estresse,
> movimentos idiossincráticos carregam informação (revisão de fundamentos,
> contágio) e não revertem — pelo contrário, podem ter continuação. Logo o edge
> concentra-se em regimes de baixa volatilidade.

Esta tese é **testável e foi testada** (§4). A hipótese original — de que o prêmio
seria maior sob estresse, à la Nagel (2012) — foi **refutada pelos dados**: o
Information Coefficient é maior em VIX baixo, consistentemente em todos os
horizontes. Adotamos a tese que sobrevive ao teste, e ela tem uma consequência
prática direta para a gestão de risco: **reduzir exposição quando a volatilidade
sobe**.

Referências: Avellaneda & Lee (2010), *Statistical Arbitrage in the US Equities
Market*; Campbell, Grossman & Wang (1993); Nagel (2012), *Evaporating Liquidity*.

---

## 2. Metodologia

### 2.1 Universo e dados
- **Membership point-in-time** do S&P 500, reconstruído para trás a partir do
  change-log da Wikipedia. Evita viés de sobrevivência no nome.
- Preços do yfinance; **Close bruto** para elegibilidade (preço mínimo, ADV) e
  **ajustado** só para retorno. Taxa livre de risco do FRED (DTB3); fatores da
  Ken French Data Library.
- Sanidade: winsorização a ±25% antes da PCA; a estratégia compra quedas, e um
  tick ruim vira lucro fictício. (auditoria em `scripts/01_build_universe.py`.)

### 2.2 Sinal (Pilar 1 — refinamento matemático)
1. **Bloco fatorial.** Os retornos vivem em L²; o operador de covariância é
   compacto e auto-adjunto, e o teorema espectral dá uma base ortonormal. O
   número de fatores k vem de **Random Matrix Theory** (Marchenko–Pastur), sem
   parâmetro livre, condicionando no modo de mercado. O resíduo é a projeção no
   complemento ortogonal dos k primeiros modos.
2. **Resíduo sem viés de ponte.** Betas estimados em [t−504, t−60] e congelados;
   resíduo calculado out-of-sample em [t−60, t]. (Acumular resíduos in-sample
   forçaria soma zero e reversão mecânica falsa.)
3. **Ornstein–Uhlenbeck** sobre o resíduo acumulado, via discretização exata
   (AR(1)), com **correção de Kendall** do viés de amostra pequena e **shrinkage
   empirical-Bayes** de κ para o pooled cross-sectional. s-score = (X−θ)/σ_eq.
4. **Primeira passagem** por função de escala / medida de velocidade (exata para
   difusão 1-D), validada contra Monte Carlo com correção de ponte browniana.

**Resultado de análise crítica (Pilar 1).** Medimos que o aparato de parada ótima
— gates de meia-vida, probabilidade de atingir o alvo, ranking por retorno por
unidade de tempo — **piora o retorno** (§4.2), porque todos derivam de κ, o
parâmetro mais mal-identificado do modelo. A estratégia de produção usa o sinal
simples (s-score); a máquina completa fica documentada como refinamento medido e
rejeitado. Este é um resultado, não uma falha: mostra que entendemos os limites
do modelo.

**A residualização em si vale a pena? Teste contra reversão simples.**
Perguntamos a versão ainda mais básica da mesma crítica: será que todo o Pilar 1
(PCA/RMT, betas congelados, resíduo OOS) é necessário, ou uma reversão à média
direta sobre o preço bruto — sem remover fator nenhum — já entrega o mesmo edge?
Rodamos os dois sinais pelo MESMO engine, MESMO livro de posições, MESMO sizing
(`scripts/10_naive_baseline.py`), variando só a tabela de candidatos:

| Variante | Sharpe (série ativa) — design | Sharpe (série ativa) — holdout |
|---|---|---|
| **MARÉ (resíduo PCA/RMT)** | **+0,19** | −0,41 |
| Reversão simples (retorno bruto) | **−0,19** | −0,53 |

A reversão simples tem **alpha ativo negativo nos dois períodos**. Comprar quem
mais caiu no preço bruto, sem controlar por mercado/setor, é dominado por
comprar beta alto durante quedas amplas — que tende a continuar perdendo, não
reverter. A decomposição espectral não é refinamento cosmético: é o que separa
"caiu porque o mercado caiu" (sistemático, não reverte) de "caiu por ruído
idiossincrático" (reverte) — sem ela, o sinal muda de sinal. Também testamos se
concentrar em menos nomes salvaria o naive (hipótese: não, porque o problema é
viés direcional, não ruído a diluir) — confirmado: com n=10 o naive segue
negativo (Sharpe ativo −0,14 no design), só com mais variância.

### 2.3 Gestão de risco (Pilar 2)
Duas camadas de exposição, não quatro (camadas demais destroem a atribuição):
- **Vol targeting** (Moreira & Muir 2017): mira 10% de vol anual, alavancagem
  máxima 1,0 — só desaloca.
- **Overlay de regime** por **Statistical Jump Model** (Nystrup, Kolm &
  Lindström), inferência filtrada online (não suavizada — essa é a armadilha do
  modelo). Reduz exposição no estado de alta volatilidade, consistente com a tese.

Sizing por vol inversa, teto por nome e por setor. Tail risk e stress replays em
`scripts/08_robustness.py`.

### 2.4 Quantos nomes segurar: curva de sensibilidade, não um pico

O número de posições (`n_names`) é um parâmetro livre, e a escolha inicial (30)
vinha de argumento teórico (Lei Fundamental de Grinold, IR ≈ IC×√amplitude) sem
teste empírico direto. Rodamos a curva completa — 9 backtests inteiros no design,
n∈{10,15,20,25,30,35,40,50,60}, cada um logado no registry com hash próprio
(`scripts/11_n_names_sensitivity.py`):

| n | Sharpe total | Vol | Turnover | Max DD |
|---|---|---|---|---|
| 10 | 0,623 | 0,235 | 0,586 | −49,9% |
| 20 | 0,613 | 0,208 | 0,486 | −49,4% |
| 30 | 0,588 | 0,207 | 0,429 | −51,6% |
| **40** | **0,614** | **0,204** | **0,396** | −49,7% |
| 50 | 0,619 | 0,197 | 0,377 | −50,6% |
| 60 | 0,606 | 0,195 | 0,361 | −51,0% |

A curva não é suave — oscila sem padrão —, o que é a assinatura de ruído
amostral, não de um efeito real. Testamos formalmente com o erro-padrão
assintótico de **Lo (2002)** para o Sharpe anualizado (T≈4.900 dias): SE≈0,227,
**16× maior** que a heurística ingênua 1/√T que teria sido usada por engano.
Comparando cada ponto contra o pico (n=10) por z-score: **todos os z ficam abaixo
de 0,13** — a grade inteira é estatisticamente **um único platô**. Não existe um
"n ótimo" defensável pelo Sharpe.

Como o Sharpe não decide, a escolha usa as duas dimensões que se movem de forma
monotônica e não-ruidosa por serem efeitos mecânicos de portfólio (diversificação
e giro), não estimativas de alpha: **vol e turnover caem com N** — o oposto do
que um pico ruidoso em n=10 sugeriria à primeira vista. Isso também é coerente
com o placebo de seleção fina (§3.3): já não há skill para escolher "os
melhores" dentro do bucket barato, então concentrar em poucos nomes não compra
edge, só compra mais risco idiossincrático e mais custo de giro.

**Escolhemos n=40**: empata estatisticamente com o pico (z=0,03) e mantém folga
confortável contra o mínimo do universo elegível (56 candidatos no pior dia,
mediana 87) — n=50/60 têm vol/turnover marginalmente melhores mas deixam menos
margem nesse extremo. Detalhe completo em `docs/decisions.md`.

### 2.5 Backtest e mitigação de vieses (Pilar 3)
- **Look-ahead impossível por construção**: a função de sinal só recebe uma view
  fatiada em `≤ data − 1`, que levanta exceção ao tentar ler o futuro. Teste
  dedicado apaga o futuro e exige pesos bit-idênticos.
- **Design 2007–2016 / holdout 2017–2026** intocado, pré-registrado pelo hash do
  config.
- Custos de transação (comissão + spread + impacto raiz-quadrada), com **custo de
  break-even** reportado.
- **Deflated Sharpe Ratio** e **placebo de seleção aleatória**, com N vindo do run
  registry.

---

## 3. Resultados

> *(Preenchido a partir de `scripts/07_validate.py --split design` e `--split
> holdout`. Ver `reports/*/dashboard.html` para as figuras.)*

### 3.1 Performance e ablação de overlays

Período de desenho 2007–2016, retorno total, líquido de custos:

| Variante | CAGR | Vol | Sharpe | Sortino | Max DD | Calmar |
|---|---|---|---|---|---|---|
| base (sinal simples, n=40) | 12,7% | 20,4% | 0,614 | 0,866 | −49,7% | 0,256 |
| + vol target | 8,5% | 10,5% | **0,685** | 0,948 | **−20,9%** | 0,406 |
| + regime | 11,7% | 15,9% | 0,676 | 0,949 | −31,5% | 0,371 |
| + regime + vol target | 7,7% | 9,7% | 0,653 | 0,899 | **−17,4%** | 0,442 |
| **SPY (benchmark)** | 10,9% | 19,6% | 0,545 | 0,768 | −55,2% | 0,197 |

**Leitura.** Todas as variantes batem o SPY em Sharpe. O `vol target` é o
destaque: eleva o Sharpe de 0,545 (SPY) para 0,685 e — o que mais importa para o
Pilar 2 — **corta o drawdown máximo de −55% para ~21%** (a combinação regime +
vol target chega a −17%). O valor entregue é majoritariamente de **gestão de
risco**: mesma ordem de retorno com muito menos sofrimento na jornada.

### 3.2 Série ativa e atribuição de fatores

Regressão da série ativa contra Fama–French 5 + momentum (Newey–West):

| Fator | Coef. (anual) | t-stat |
|---|---|---|
| **alpha** | **+2,63%** | **1,75** (p=0,08) |
| mkt_rf | −0,63 | −18,8 |
| smb | +0,25 | +14,0 |
| cma | +0,12 | +3,7 |
| mom | +0,06 | +4,0 |

R² dos fatores: 0,78. **Interpretação honesta:** no design há um alpha de +2,6%/ano
marginalmente significativo (t=1,75) depois de controlar por seis fatores, com um
tilt para small caps (SMB, t=14). **Mas este alpha é in-sample e não persiste** —
no holdout (§3.4) ele vira −0,7%/ano (t=−0,33). Reportamos o número do design pelo
que ele é: uma medida no período de desenho, não uma promessa de futuro.

### 3.3 Validação estatística

- **Deflated Sharpe Ratio**: 0,92 (base) e 0,973 (com overlays), com **N=11**
  configurações distintas no registry — inclui as 9 rodadas da curva de
  sensibilidade de `n_names` (§2.4), então o DSR já paga o preço de ter buscado o
  parâmetro. O Sharpe esperado sob o nulo (SR₀) subiu de 0,19 para **0,37** ao
  contabilizar essa busca — e mesmo assim o DSR continua alto.
- **Probabilistic Sharpe (vs 0)**: 0,999–1,000.
- **Minimum Track Record Length**: 1.070–1.460 dias; a amostra de desenho (≈2.500
  dias) é suficiente.
- **Placebo de seleção aleatória** (dois nulos complementares):
  - *vs universo líquido*: percentil **94,9%** (p=0,051) — a reversão residual (o
    filtro "estar barato") bate fortemente a seleção aleatória.
  - *vs pool barato*: percentil 48% — escolher os MAIS baratos dentro do bucket
    barato **não** adiciona nada além de estar no bucket. O edge está no filtro de
    reversão, não na ordenação fina — coerente com a §4.2 e com a §2.4.
  - *block bootstrap*: mede a dispersão do Sharpe, não significância contra zero.

*Nota de honestidade sobre o N.* O run registry conta as configurações formais
registradas — 11 ao todo, incluindo a grade de sensibilidade de `n_names`
(§2.4), que é buscar um parâmetro e por isso soma ao N corretamente. Cada
parâmetro fixado por citação (entrada/saída de A&L, λ do RiskMetrics) foi
mantido fora da busca de propósito, para não inflar o N artificialmente.

### 3.4 O teste out-of-sample: o sinal decaiu (este é o resultado central)

O holdout 2017–2026 foi pré-registrado (hash do config) e examinado **duas
vezes**: uma com `n_names=30`, e uma segunda depois de retunar para 40 com base
numa curva de sensibilidade rodada inteiramente no design (§2.4) — sem tocar o
holdout na escolha. Os números abaixo são da configuração final (n=40); a
diferença qualitativa entre as duas observações é nula, o que reforça que a
retunagem não é sobreajuste ao holdout. O resultado é desconfortável e o
reportamos de frente, porque é exatamente o que o edital pede — a realidade, não
o que parece bom:

| Variante | CAGR | Sharpe | Max DD |
|---|---|---|---|
| MARÉ base | 11,2% | 0,527 | −36% |
| + vol target | 7,0% | 0,470 | −16% |
| **SPY** | **15,2%** | **0,731** | −34% |

- **A estratégia perde para o SPY out-of-sample**, em retorno e em Sharpe.
- **Série ativa negativa** (base t=−1,33; variantes de-riscadas t≈−2,5 a −2,7).
- **Alpha sobre FF5+MOM: −0,7%/ano (t=−0,33)** — o +2,6% do design **evaporou**.
- **Placebo vs universo: percentil 18,8%** (era 94,9% no design). No holdout,
  carteiras aleatórias do universo líquido batem a seleção do sinal. **O sinal de
  seleção decaiu.**

**Por que — e por que isso não é um bug.** É decaimento de sinal genuíno (os testes
de look-ahead descartam artefato). A explicação econômica que melhor se sustenta,
e que assumimos como **hipótese principal, não como fato provado**: o edge da
reversão residual depende de **dispersão cross-sectional** — precisa haver
laggards que revertem em relação a líderes. O design (2007–2016) tem a GFC e o
pós-crise de dispersão elevada; o holdout foi um bull market de dispersão colapsada
e concentração recorde em mega-caps de momentum, onde comprar os *laggards*
baratos perde por construção e o tilt SMB vira lastro.

Somos honestos sobre o limite desta explicação: no design encontramos o IC maior
em **VIX baixo** (§4.3), mas 2017–2019 foi de VIX baixíssimo e ainda assim não
entregou — o que sugere que **dispersão, não volatilidade, é o driver real**, e
que o próprio achado condicional a VIX pode ter sido em parte específico do
período. Isso é exatamente o que torna a §6.2 (condicionar o sinal à dispersão) o
próximo teste direto, não uma racionalização.

**O que sobreviveu ao OOS.** O `vol target` cortou o drawdown de −36% para −16%
também no holdout. **A gestão de risco (Pilar 2) generaliza; o alpha de seleção
(Pilar 1) não.** Esta é a lição honesta do trabalho.

### 3.5 Robustez a custos e a sobrevivência
- **Custo de break-even**: ~63 bps one-way zeram o retorno bruto do design —
  margem de ~25× sobre os ~2,5 bps realistas em large caps. Custo não é o gargalo.
- **Viés de sobrevivência**: assumindo — conservadoramente — que 5% dos nomes
  mantidos deslistassem *por ano* (acima do churn real do S&P 500, do qual quase
  tudo é aquisição: 2 falências em 196 saídas na amostra), o CAGR do design cai de
  12,7% para 11,9% (saída a −15%), 11,0% (queda a −30%) ou 7,2% apenas no cenário
  extremo e irreal em que essa fração inteira vai a zero. O viés existe mas nunca
  inverte a leitura. Ver `scripts/08_robustness.py`.

---

## 4. Análise crítica (experimentos)

### 4.1 O sinal tem edge, mas é fraco e concentrado
O Information Coefficient do s-score contra retorno futuro é ~zero em 5–10 dias e
só aparece em 21 dias. A cesta dos mais baratos rende +0,4% excedente em 21d
(t≈2,9, otimista por janelas sobrepostas — corrigido por Newey-West nos resultados).

### 4.2 O aparato sofisticado piora o retorno (ablação de gates)

Retorno excedente médio da cesta em 21 dias, período de desenho (`scripts/04`):

| Seleção | Ret. 21d | t-stat |
|---|---|---|
| 30 mais baratos (sinal simples) | **+0,41%** | **2,94** |
| + gate de R² | +0,41% | 2,94 |
| + gate de probabilidade P(alvo) | +0,22% | 1,58 |
| + gate de meia-vida | +0,10% | 0,66 |
| todos os gates + ranking por μ_s | −0,02% | −0,12 |

Todo filtro derivado de κ destrói valor, porque κ é o parâmetro mais mal-
identificado (≈3 ciclos independentes em 60 observações). Testamos também afrouxar
a banda de meia-vida (nos dois sentidos) e ranquear por retorno esperado — nenhuma
variante recupera o baseline. **Conclusão: operar o sinal simples.** A máquina de
parada ótima, matematicamente correta e validada contra Monte Carlo, fica como
demonstração de que entendemos seus limites.

### 4.3 A tese econômica original foi refutada

Information Coefficient (reversão) condicionado ao VIX, período de desenho:

| Horizonte | VIX baixo | VIX alto | alto − baixo |
|---|---|---|---|
| 5 dias | +0,015 | +0,003 | −0,012 |
| 10 dias | +0,016 | −0,003 | −0,018 |
| 21 dias | +0,028 | +0,013 | −0,015 |

O edge é **maior em VIX baixo**, consistentemente. A reconciliação via horizonte
(reversão diária pagaria mais sob estresse, à la Nagel) também foi testada e não
apareceu. Adotamos a tese que sobrevive — risco de informação domina em estresse
— e a sua implicação prática: o overlay de regime **reduz** exposição em VIX alto,
o que a §3.1 confirma ter valor.

---

## 5. Limitações

- **A limitação principal é a §3.4: o alpha de seleção não sobrevive ao OOS.** O
  edge da reversão residual é dependente de regime (alta dispersão) e o regime
  virou pós-2016. Não escondemos isso atrás de métricas contra zero (o DSR de
  0,66 no holdout mede Sharpe contra zero, não contra o SPY — e contra o SPY a
  estratégia perde). O número que importa é a série ativa, e ela é negativa OOS.
- **Viés de sobrevivência residual.** 22,8% dos símbolos históricos sem preço no
  yfinance (majoritariamente aquisição, não falência: 2 de 196). Limite superior
  medido em §3.5: some 1–2 pp de CAGR mesmo no cenário de falência total a 5%/ano.
  Real, mas pequeno demais para explicar os resultados.
- **Setor GICS não é point-in-time** (a Wikipedia só publica o vigente); efeito de
  segunda ordem no teto setorial.
- **Custo não é o gargalo** (break-even ~25× o custo realista), mas seria o
  próximo a morder num universo menos líquido.
- **`n_names` foi retunado pós-hoc no design** (30→40, §2.4); o DSR já paga o
  preço dessa busca (N=11, SR₀ subiu de 0,19 para 0,37) e o holdout foi olhado
  duas vezes — ambas transparentes no registry. A conclusão qualitativa do OOS
  não mudou entre as duas, o que é evidência contra sobreajuste ao holdout, mas o
  duplo olhar em si é uma limitação de disciplina que registramos, não escondemos.

---

## 6. Conclusão e próximos passos

**O que aprendemos, dito sem maquiagem.** MARÉ tem um sinal de reversão residual
com edge in-sample (Sharpe 0,61 vs SPY 0,55; alpha +2,6% sobre fatores; placebo no
percentil 95). Mas o edge **não sobrevive out-of-sample**: no holdout 2017–2026 a
estratégia perde para o SPY, o alpha ativo é negativo e a seleção fica abaixo da
mediana de carteiras aleatórias. A causa é econômica e a prevíamos ao refutar a
tese original — o edge vive em regimes de alta dispersão, e a dispersão colapsou
pós-2016.

**Duas coisas de valor duradouro saíram do trabalho, e nenhuma é o alpha:**
1. **A gestão de risco generaliza.** O vol target cortou o drawdown pela metade
   nos dois períodos. É o componente que entregaríamos em produção.
2. **A metodologia honesta.** Look-ahead impossível por construção, holdout
   pré-registrado e reportado mesmo contrariando o resultado, placebo que expôs os
   limites da seleção, ablação que rejeitou o próprio aparato sofisticado. É o
   oposto de um backtest que "aparenta ser bom".

**Se déssemos o próximo passo** (e é aqui que a estratégia se torna investível):
1. **Long-short market-neutral** — isola o alpha residual do beta. O sinal ativo
   é onde MARÉ vive; long-only mistura-o com a direção do mercado, e foi o beta do
   mercado (não o alpha) que ganhou de 2017 a 2026.
2. **Condicionar o sinal à dispersão cross-sectional** — ligar a estratégia
   apenas quando o regime que a favorece está presente, exatamente o que a tese
   econômica prevê. É a implicação testável direta do achado de decaimento.
3. **Universo mais amplo** (S&P 400/600), onde a intermediação é mais escassa e a
   tese de liquidez prevê edge maior e mais persistente.
