# Log de uso de IA generativa

O edital exige uso de IA generativa em pelo menos uma etapa, com descrição
objetiva de onde entrou. Este documento registra, por fase, **o que a IA
produziu, o que a equipe manteve e o que a equipe rejeitou ou corrigiu**. As
rejeições são registradas de propósito: elas mostram supervisão crítica, que é o
que separa usar IA de terceirizar o julgamento para ela.

Ferramenta: Claude (Anthropic), em fluxo de par-programação e revisão adversarial.

---

## Fase 1 — Desenho da estratégia e crítica adversarial

**O que a IA fez.** A partir do rascunho de três pilares da equipe (reversão à
média com refinamento matemático, gestão de risco, backtest com dashboard), a IA
pesquisou literatura (Avellaneda & Lee 2010, Bertram 2010, Nystrup et al.,
Bailey & López de Prado, Nagel 2012) e submeteu o desenho a uma crítica técnica
no papel de banca.

**O que a equipe MANTEVE.** A tese de reversão residual estilo Avellaneda–Lee; a
decomposição espectral com denoising por Marchenko–Pastur; o OU com correção de
Kendall e shrinkage; a primeira passagem por função de escala; o overlay de
regime por jump model; a bateria de validação (DSR, placebo, walk-forward).

**O que a equipe REJEITOU por recomendação da própria crítica da IA.**
- A justificativa via teorema de Dambis–Dubins–Schwarz para o tempo de negócio.
  A IA apontou que DDS vale para martingale, e o OU tem drift — trocar o tempo
  dá κ estocástico, não vol constante. Substituído por premissa de modelagem
  explícita (subordinação, Clark 1973; Ané & Geman 2000).
- As fórmulas de Bertram com Erfi/digamma, que resolvem passagem unilateral sem
  stop — não o nosso problema de duas barreiras. Substituídas por função de
  escala/medida de velocidade.
- A FPCA/Karhunen–Loève como sinal ativo, rebaixada a resultado negativo (as
  autofunções empíricas reproduzem a base teórica do browniano).

**Decisão HUMANA sobre a IA.** Onde a IA ofereceu opções, a equipe escolheu:
25–40 nomes (não 10–15), long-only com reporte de 3 séries, holdout 2017+.

---

## Fase 2 — Reconstrução do universo point-in-time

**O que a IA fez.** Implementou o parser do change-log da Wikipedia e a
reconstrução para trás do membership, além da auditoria de cobertura de dados.

**Achado que mudou o rumo.** A auditoria (sugerida na crítica) revelou que 22,8%
dos símbolos históricos não têm preço no Yahoo. A IA aprofundou: 122 dos 196
ausentes saíram por aquisição, com mediana de 8,5 anos até a saída — viés real
mas limitado, não catastrófico. A equipe manteve a decisão de reportar a tabela
de cobertura como exhibit em vez de escondê-la.

---

## Fase 3 — Implementação e descoberta empírica

**O que a IA fez.** Implementou o pipeline (sinal, engine com AsOfView anti-
look-ahead, custos, métricas) e a bateria de testes.

**Bugs que os testes da IA pegaram, não a intuição.**
- Monte Carlo de primeira passagem divergindo da quadratura: a IA diagnosticou
  que a quadratura estava certa (bate com forma fechada em erfi a 3e-7) e o erro
  era viés de monitoramento discreto no MC, corrigido por ponte browniana.
- Seleção de k por Marchenko–Pastur degenerando (k=97–189): a IA identificou que
  o modo de mercado precisa ser condicionado antes de estimar σ².

**A descoberta empírica central, e a decisão humana sobre ela.** O diagnóstico de
qualidade de sinal (IC e ablação de gates) mostrou que **o aparato sofisticado de
parada ótima PIORA o retorno** — todo filtro derivado de κ (meia-vida, P(alvo),
ranking por μ_s) destrói valor, porque κ é o parâmetro mais mal-identificado. A
IA apresentou o resultado sem enfeitá-lo. A equipe decidiu: operar o sinal
simples e transformar a máquina inteira em exhibit de análise crítica
("construímos, medimos, rejeitamos"). Este é o resultado do qual a equipe mais se
orgulha, e ele só existe porque a IA reportou um número desconfortável em vez de
esconder.

**Tese econômica refutada pelos dados.** A hipótese inicial (Nagel 2012: reversão
paga mais sob estresse) foi testada e REJEITADA — o edge concentra-se em baixa
volatilidade. A IA testou a reconciliação via horizonte e ela também não passou.
A equipe adotou a tese que sobrevive (risco de informação domina em estresse;
reversão captura ruído de liquidez, que domina em calmaria) e a implicação
testável para o Pilar 2: reduzir exposição em VIX alto.

---

## Como verificar

Todo experimento citado é reproduzível: os scripts em `scripts/` geram os
números, e o `config/registry.jsonl` registra cada configuração testada (é de
onde sai o N do Deflated Sharpe). Nenhum número neste log foi gerado por IA sem
execução de código verificável.
