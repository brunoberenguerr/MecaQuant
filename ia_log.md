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
