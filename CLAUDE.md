# Projeto MecaQuant — Desafio Quant AI 2026 (Itaú Asset)

## Contexto
Equipe de 3 estudantes de eng. mecatrônica (3º ano), base ok em Python/matemática/estatística.
Estratégia definida: **Time-Series Momentum (trend following) multi-ativo com volatility targeting**.
Referências: Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum"; Hurst, Ooi & Pedersen (2017) "A Century of Evidence on Trend-Following Investing".

## Prazos
- Pré-relatório: **31/07/2026**
- Relatório final: **16/08/2026** — PDF, máx. 5 páginas, 16:9, anônimo, ~750 palavras, visual (não acadêmico)

## Critérios de avaliação (pesos)
Conceito 20% · Modelagem 20% · Backtest 15% · Análise de resultados 15% · Uso de IA generativa 15% · Conclusão 10% · Robô/identidade 5%.
Nota: complexidade NÃO pontua. Simples e bem executado > complexo mal justificado.

## Universo (já selecionado — ver correlation_matrix_candidatos.csv e universe_selection.py)
Funil: 26 candidatos → clustering hierárquico (corte ρ≈0,65) + histórico ≥15 anos + corte ρ>0,6 → 8 ativos:

| Ticker Yahoo | Classe | Observação |
|---|---|---|
| ^BVSP | Ações BR | Negociável via BOVA11; índice já é retorno total |
| SPY | Ações globais | |
| IEF | Juros EUA 7-10a | |
| BRL=X | USD/BRL | ρ −36 c/ Ibov (diversificador-chave) |
| EURUSD=X | Moeda G10 | |
| JPY=X | USD/JPY | |
| GLD | Ouro | |
| DBC | Commodities | |

Corr média pareada 0,04 → 6,1 apostas efetivas de 8. BTC-USD fora do core (histórico 11,7 anos, sem 2008); usar só como teste de robustez satélite.

## Especificação do backtest v1
- Dados: yfinance, preços diários ajustados, 2005-01 → hoje (inclui 2008 e 2020).
- Sinal: retorno acumulado 12 meses excluindo o último mês (12-1). Positivo → comprado; negativo → vendido.
- Sizing: peso ∝ 1/vol (vol EWMA ou janela 60 dias), vol alvo do portfólio 10% a.a.
- Rebalanceamento: mensal (último dia útil).
- Custos: 0,1% por trade sobre o notional negociado.
- Benchmarks: Ibovespa e CDI (CDI: usar série ~taxa Selic aproximada ou constante justificada).
- Métricas: CAGR, vol, Sharpe, max drawdown, pior ano, desempenho em 2008 e 2020.
- Sanidade: Sharpe esperado 0,6–1,0. Sharpe > 1,5 = provável bug/viés — investigar antes de aceitar.

## Vieses a evitar (checklist)
- Look-ahead: sinal em t usa apenas dados até t-1; executar no fechamento seguinte.
- Survivorship: universo fixo definido ex-ante (já é o caso).
- Overfitting: NÃO otimizar parâmetros por grid search; usar valores canônicos da literatura (12-1, vol 10%).
- Meses faltantes nas séries de FX do Yahoo (~20% em BRL=X): forward-fill limitado ou alinhamento por datas comuns.

## Extensões (v2, se der tempo)
- Combinar horizontes 1m/3m/12m (média dos sinais).
- Robustez: variar janela de vol, custos 2x, sub-períodos.
- Satélite BTC desde 2015 (mostrar impacto marginal).

## Uso de IA generativa (documentar SEMPRE — vale 15% da nota)
Manter log em `ia_log.md`: cada uso concreto (seleção de universo 26→8 já feita com IA, geração de código, crítica de vieses, síntese de literatura), com exemplo e limitação encontrada (ex.: sandbox sem acesso a dados de mercado → contornado via browser).

## Robô (pendente decisão do grupo)
Candidatos: VETOR (segue direção e magnitude) ou GIRO (momentum angular). Identidade visual via IA de imagem.

## Regras do relatório final
Proibido: nomes, equipe, universidade, logos. Obrigatório: nome do robô + identidade visual + explicação do nome, integrados ao conteúdo.
