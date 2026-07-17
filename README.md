# MecaQuant — Desafio Quant AI 2026 (Itaú Asset)

Estratégia **Time-Series Momentum (trend following) multi-ativo com volatility
targeting**, em um universo de 8 ativos pouco correlacionados selecionado
ex-ante. O plano completo do projeto (especificação, prazos, critérios de
avaliação e checklist de vieses) está em [`CLAUDE.md`](CLAUDE.md).

## Como rodar

```bash
pip install -r requirements.txt
python download_data.py   # baixa preços (Yahoo Finance) e CDI (BCB) para data/
python backtest.py        # roda o backtest e salva tudo em results/
```

Saídas do backtest em `results/`: `metricas.csv` (CAGR, vol, Sharpe, max
drawdown, pior ano, 2008, 2020), `retornos_anuais.csv` e os gráficos
`equity_curve.png`, `drawdown.png`, `retornos_anuais.png`, `exposicao.png`.

## Arquivos

| Arquivo | O que é |
|---|---|
| `CLAUDE.md` | Plano do projeto e especificação do backtest v1 |
| `universe_selection.py` | Seleção do universo (26 candidatos → 8 ativos) |
| `correlation_matrix_candidatos.csv` | Matriz de correlação dos candidatos |
| `heatmap_*.png` | Heatmaps da seleção de universo |
| `download_data.py` | Download dos dados (preços + CDI) para `data/` |
| `backtest.py` | Backtest v1 (momentum 12-1, vol targeting 10% a.a.) |
| `ia_log.md` | Log do uso de IA generativa (vale 15% da nota!) |

## Fluxo de trabalho da equipe

1. `git clone https://github.com/brunoberenguerr/mecaquant.git`
2. Criar uma branch para sua alteração: `git checkout -b minha-mudanca`
3. Commitar, dar push e abrir um Pull Request para revisão dos colegas.

**Importante:** toda vez que usar IA (Claude, ChatGPT etc.) em algo do
projeto, registrar no `ia_log.md` — uso de IA generativa vale 15% da nota.
