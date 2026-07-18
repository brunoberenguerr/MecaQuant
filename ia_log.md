# Log de uso de IA generativa — MecaQuant

> Critério do desafio: uso de IA generativa vale **15% da nota**. Registrar aqui
> cada uso concreto: o que foi pedido, o que a IA entregou, e limitações
> encontradas. Atualizar SEMPRE que usar IA no projeto.

## 2026-07 — Seleção de universo (Claude / Cowork)

- **O quê:** funil de 26 ativos candidatos → 8 finais, via clustering
  hierárquico sobre a matriz de correlações mensais (corte ρ≈0,65), filtro de
  histórico ≥15 anos e corte de correlação pareada >0,6.
- **Entrega:** `universe_selection.py`, `correlation_matrix_candidatos.csv`,
  heatmaps. Corr média pareada final 0,04 (≈6,1 apostas efetivas de 8).
- **Limitação encontrada:** o sandbox do Cowork não tinha acesso direto a dados
  de mercado; correlações levantadas via navegação e conferidas manualmente.

## 2026-07 — Setup do repositório e backtest v1 (Claude Code na web)

- **O quê:** publicação do projeto no GitHub para trabalho em equipe; geração
  do código do backtest v1 completo conforme especificação do CLAUDE.md
  (momentum 12-1, sizing 1/vol com alvo de 10% a.a., rebalanceamento mensal,
  custos de 0,1%, benchmarks Ibovespa e CDI, métricas e gráficos).
- **Entrega:** `download_data.py`, `backtest.py`, `README.md`,
  `requirements.txt`. Código validado de ponta a ponta com dados sintéticos.
- **Limitação encontrada:** o ambiente remoto do Claude Code tem política de
  rede que bloqueia Yahoo Finance/BCB; o download dos dados precisa ser feito
  localmente (ou liberando os domínios na configuração do ambiente).

<!-- Modelo de entrada nova:
## AAAA-MM — Título (ferramenta)
- **O quê:** ...
- **Entrega:** ...
- **Limitação encontrada:** ...
-->

## 2026-07-18 — Diagnóstico do Sharpe negativo e correção v1.1 (Claude)
- Backtest v1 imprimiu Sharpe exc. CDI de -0,24. A IA diagnosticou que não era
  bug de sinal: o PnL do overlay era comparado ao CDI (~9% a.a.) sem remunerar
  o caixa. Na implementação real via futuros, a margem rende CDI.
- Correção v1.1 (1 linha): retorno total = CDI + PnL - custos.
  Resultado: CAGR 15,5% | Sharpe exc. CDI 0,57 | MaxDD -14,5% | 2008: +9,9% | 2020: +8,0%.
- Limitação de ambiente documentada: sandbox da IA sem acesso a Yahoo/BCB;
  dados baixados pelo navegador do usuário (CSVs gerados via JS) e validados
  (cobertura por ativo, inícios de série coerentes com inception dos ETFs).

## 2026-07-18 — Pré-relatório e identidade do robô (Claude)
- **O quê:** IA montou o pré-relatório completo (5 págs, 16:9, ~830 palavras)
  e criou a identidade visual do MIYAGI por código vetorial (matplotlib).
- **Entrega:** `prerelatorio/` (PDF final, PPTX editável, emblemas, scripts).
- **Limitação encontrada:** contagem de palavras acima da referência na 1ª
  versão (905) → revisão de concisão via IA até 832.
