const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE"; // 13.33 x 7.5

const INK = "16181B", RED = "C63D2F", STEEL = "5B6770",
      LIGHT = "EEF0F2", WHITE = "FFFFFF", MUTED = "6E6C66",
      DARKBG = "14161A", CARD_D = "1E2228";
const F = "Arial";

// ---------------- SLIDE 1 — Identidade + Conceito (dark) ----------------
let s = p.addSlide();
s.background = { color: DARKBG };
s.addImage({ path: "miyagi_emblem.png", x: 0.55, y: 1.35, w: 4.1, h: 4.1 });
s.addText("MIYAGI", { x: 0.55, y: 5.45, w: 4.1, h: 0.85, fontFace: F, fontSize: 48,
  bold: true, color: WHITE, align: "center", margin: 0 });
s.addText("Robô de tendência multi-ativo com alvo de risco", { x: 0.55, y: 6.25, w: 4.1,
  h: 0.4, fontFace: F, fontSize: 13, color: LIGHT, align: "center", margin: 0 });
s.addText("“Não prevê o golpe. Observa o movimento — e responde com técnica.”",
  { x: 5.1, y: 0.55, w: 7.6, h: 0.75, fontFace: F, fontSize: 19, italic: true,
    color: WHITE, margin: 0 });

s.addShape(p.ShapeType.roundRect, { x: 5.1, y: 1.5, w: 7.65, h: 1.95,
  fill: { color: CARD_D }, rectRadius: 0.09 });
s.addText([
  { text: "HIPÓTESE CENTRAL", options: { fontSize: 12.5, bold: true, color: RED, breakLine: true, paraSpaceAfter: 6 } },
  { text: "Mercados sub-reagem à informação e tendências persistem por meses. Seguir o movimento — comprado no que sobe, vendido no que cai — captura um prêmio documentado em 58 mercados e um século de dados (Moskowitz et al. 2012; Hurst et al. 2017).",
    options: { fontSize: 13.5, color: WHITE } },
], { x: 5.35, y: 1.68, w: 7.15, h: 1.6, fontFace: F, margin: 0, valign: "top" });

s.addShape(p.ShapeType.roundRect, { x: 5.1, y: 3.7, w: 7.65, h: 1.75,
  fill: { color: CARD_D }, rectRadius: 0.09 });
s.addText([
  { text: "POR QUE A INEFICIÊNCIA EXISTE — E PERSISTE", options: { fontSize: 12.5, bold: true, color: RED, breakLine: true, paraSpaceAfter: 6 } },
  { text: "Sub-reação: preços incorporam notícias aos poucos (ancoragem).  ", options: { fontSize: 13, color: WHITE, breakLine: true, paraSpaceAfter: 3 } },
  { text: "Manada: fluxos seguem retornos e amplificam o movimento.  ", options: { fontSize: 13, color: WHITE, breakLine: true, paraSpaceAfter: 3 } },
  { text: "Limites à arbitragem: quem aposta contra a tendência sangra antes de ter razão.", options: { fontSize: 13, color: WHITE } },
], { x: 5.35, y: 3.88, w: 7.15, h: 1.45, fontFace: F, margin: 0, valign: "top" });

s.addShape(p.ShapeType.roundRect, { x: 5.1, y: 5.72, w: 7.65, h: 1.28,
  fill: { color: CARD_D }, rectRadius: 0.09 });
s.addText([
  { text: "POR QUE “MIYAGI”?", options: { fontSize: 12.5, bold: true, color: RED, breakLine: true, paraSpaceAfter: 5 } },
  { text: "Como o mestre: movimentos simples repetidos com disciplina — e defesa antes do ataque. Sinal mensal simples; controle de risco primeiro.",
    options: { fontSize: 13, color: WHITE } },
], { x: 5.35, y: 5.88, w: 7.15, h: 1.05, fontFace: F, margin: 0, valign: "top" });

// ---------------- SLIDE 2 — Universo (light) ----------------
s = p.addSlide();
s.background = { color: WHITE };
s.addText("Diversificação é o motor: 26 candidatos, 8 ativos, 6 classes",
  { x: 0.55, y: 0.35, w: 12.2, h: 0.65, fontFace: F, fontSize: 28, bold: true, color: INK, margin: 0 });

const funil = [
  ["1", "26 candidatos em 6 classes", "ações BR e globais, juros, moedas, commodities, cripto — todos líquidos e com dados públicos"],
  ["2", "Filtro estatístico e de histórico", "clustering hierárquico de correlações (corte ρ≈0,65) + mínimo de 15 anos de dados (inclui 2008)"],
  ["3", "8 ativos ≈ 6,1 apostas independentes", "1 ativo por cluster; correlação média entre eles: 0,04"],
];
funil.forEach((f, i) => {
  const y = 1.25 + i * 1.28;
  s.addShape(p.ShapeType.roundRect, { x: 0.55, y, w: 6.1, h: 1.1,
    fill: { color: LIGHT }, rectRadius: 0.08 });
  s.addShape(p.ShapeType.ellipse, { x: 0.75, y: y + 0.28, w: 0.54, h: 0.54, fill: { color: RED } });
  s.addText(f[0], { x: 0.75, y: y + 0.28, w: 0.54, h: 0.54, fontFace: F, fontSize: 18,
    bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  s.addText([
    { text: f[1], options: { fontSize: 14.5, bold: true, color: INK, breakLine: true, paraSpaceAfter: 3 } },
    { text: f[2], options: { fontSize: 11.5, color: MUTED } },
  ], { x: 1.5, y: y + 0.1, w: 5.0, h: 0.95, fontFace: F, margin: 0, valign: "middle" });
  if (i < 2) s.addShape(p.ShapeType.downArrow, { x: 3.35, y: y + 1.08, w: 0.28, h: 0.24, fill: { color: STEEL } });
});

s.addShape(p.ShapeType.roundRect, { x: 0.55, y: 5.25, w: 6.1, h: 1.85,
  fill: { color: "FDF1EF" }, rectRadius: 0.08 });
s.addText([
  { text: "Sharpe ∝ taxa de informação × √(nº de apostas independentes)", options: { fontSize: 13.5, bold: true, color: INK, breakLine: true, paraSpaceAfter: 6 } },
  { text: "Com o mesmo sinal, diversificar multiplica o Sharpe: 8 ativos só de ações (ρ médio 0,69) valem 1,4 apostas; os nossos 8, 6,1.", options: { fontSize: 12.5, color: INK } },
], { x: 0.8, y: 5.42, w: 5.6, h: 1.55, fontFace: F, margin: 0, valign: "top" });

s.addImage({ path: "heatmap_universo_final.png", x: 7.0, y: 1.15, w: 5.85, h: 5.05 });
s.addText("Ibovespa (via BOVA11), S&P 500, Treasuries 7–10a, USD/BRL, EUR/USD, USD/JPY, ouro e commodities. USD/BRL: ρ −36 com o Ibovespa — proteção natural do book local.",
  { x: 7.0, y: 6.3, w: 5.85, h: 0.85, fontFace: F, fontSize: 10.5, color: MUTED, margin: 0 });

// ---------------- SLIDE 3 — Modelagem (light) ----------------
s = p.addSlide();
s.background = { color: WHITE };
s.addText("Como Miyagi decide: sinal simples, defesa primeiro",
  { x: 0.55, y: 0.35, w: 12.2, h: 0.65, fontFace: F, fontSize: 28, bold: true, color: INK, margin: 0 });

const passos = [
  ["OBSERVA", "Preços diários ajustados dos 8 ativos (Yahoo Finance) e CDI oficial (BCB/SGS)"],
  ["SINAL", "Retorno acumulado de 12 meses, excluindo o último mês: positivo → comprado; negativo → vendido"],
  ["TAMANHO", "Peso proporcional a 1/volatilidade (janela de 60 dias): cada ativo contribui com risco parecido"],
  ["DEFESA", "Carteira escalada para volatilidade alvo de 10% a.a. (alavancagem limitada a 3×)"],
  ["EXECUTA", "Rebalanceamento mensal no fechamento; custo de 0,1% sobre o valor negociado"],
];
passos.forEach((f, i) => {
  const x = 0.5 + i * 2.51;
  s.addShape(p.ShapeType.roundRect, { x, y: 1.35, w: 2.24, h: 2.5,
    fill: { color: i === 3 ? "FDF1EF" : LIGHT }, rectRadius: 0.09 });
  s.addText([
    { text: f[0], options: { fontSize: 13, bold: true, color: i === 3 ? RED : STEEL, breakLine: true, paraSpaceAfter: 6 } },
    { text: f[1], options: { fontSize: 11.3, color: INK } },
  ], { x: x + 0.16, y: 1.55, w: 1.94, h: 2.15, fontFace: F, margin: 0, valign: "top" });
  if (i < 4) s.addShape(p.ShapeType.rightArrow, { x: x + 2.25, y: 2.44, w: 0.25, h: 0.28, fill: { color: STEEL } });
});

s.addShape(p.ShapeType.roundRect, { x: 0.55, y: 4.2, w: 5.9, h: 1.45,
  fill: { color: CARD_D }, rectRadius: 0.08 });
s.addText([
  { text: "wᵢ ∝ sinal(r 12m→1m) ÷ σᵢ(60d)", options: { fontSize: 16, bold: true, color: WHITE, breakLine: true, paraSpaceAfter: 5 } },
  { text: "carteira × (10% ÷ σ_carteira)", options: { fontSize: 16, bold: true, color: WHITE } },
], { x: 0.85, y: 4.4, w: 5.35, h: 1.1, fontFace: F, margin: 0, valign: "middle" });

s.addShape(p.ShapeType.roundRect, { x: 6.85, y: 4.2, w: 6.0, h: 2.75,
  fill: { color: LIGHT }, rectRadius: 0.08 });
s.addText([
  { text: "TRÊS VIESES, TRÊS DEFESAS", options: { fontSize: 12.5, bold: true, color: RED, breakLine: true, paraSpaceAfter: 6 } },
  { text: "Look-ahead: o sinal do mês usa apenas dados até o mês anterior; os pesos valem a partir do pregão seguinte.", options: { fontSize: 12, color: INK, bullet: true, breakLine: true, paraSpaceAfter: 4 } },
  { text: "Survivorship: universo fixo, definido ex-ante — nenhum ativo entrou por ter “dado certo”.", options: { fontSize: 12, color: INK, bullet: true, breakLine: true, paraSpaceAfter: 4 } },
  { text: "Overfitting: zero grid search. Sinal 12-1, vol 60d e alvo de 10% vêm dos papers, não dos nossos resultados.", options: { fontSize: 12, color: INK, bullet: true } },
], { x: 7.15, y: 4.4, w: 5.4, h: 2.4, fontFace: F, margin: 0, valign: "top" });

s.addText("Simplificação documentada: retornos por série em moeda local, sem conversão do PnL — padrão dos papers de momentum com futuros.",
  { x: 0.55, y: 5.85, w: 5.9, h: 0.8, fontFace: F, fontSize: 10.5, color: MUTED, margin: 0 });

// ---------------- SLIDE 4 — Backtest (light) ----------------
s = p.addSlide();
s.background = { color: WHITE };
s.addText("21 anos, duas crises — sem pânico",
  { x: 0.55, y: 0.35, w: 12.2, h: 0.65, fontFace: F, fontSize: 28, bold: true, color: INK, margin: 0 });

const rows = [
  [{ text: "", options: {} }, { text: "CAGR", options: { bold: true } }, { text: "Vol a.a.", options: { bold: true } },
   { text: "Sharpe*", options: { bold: true } }, { text: "Max DD", options: { bold: true } },
   { text: "2008", options: { bold: true } }, { text: "2020", options: { bold: true } }],
  [{ text: "MIYAGI", options: { bold: true, color: RED } }, { text: "15,5%", options: { bold: true } }, { text: "10,9%", options: {} },
   { text: "0,57", options: { bold: true } }, { text: "−14,5%", options: { bold: true } }, { text: "+9,9%", options: { bold: true } }, { text: "+8,0%", options: { bold: true } }],
  [{ text: "Ibovespa", options: {} }, { text: "7,9%", options: {} }, { text: "23,9%", options: {} },
   { text: "0,07", options: {} }, { text: "−60,0%", options: {} }, { text: "−41,2%", options: {} }, { text: "+2,9%", options: {} }],
  [{ text: "CDI", options: {} }, { text: "9,2%", options: {} }, { text: "0,3%", options: {} },
   { text: "—", options: {} }, { text: "0,0%", options: {} }, { text: "+12,4%", options: {} }, { text: "+2,8%", options: {} }],
];
s.addTable(rows, { x: 0.55, y: 1.15, w: 6.6, colW: [1.5, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85],
  fontFace: F, fontSize: 12, color: INK, align: "center", valign: "middle",
  border: { type: "solid", color: "DDDBD4", pt: 0.75 }, fill: { color: "FFFFFF" },
  rowH: 0.38 });
s.addText("*Sharpe do excesso sobre o CDI; caixa remunerado a CDI (futuros). 2005–2026, custos de 0,1%/trade.",
  { x: 0.55, y: 2.9, w: 6.6, h: 0.6, fontFace: F, fontSize: 10, color: MUTED, margin: 0 });

s.addImage({ path: "equity_curve.png", x: 0.55, y: 3.6, w: 6.55, h: 3.5 });
s.addImage({ path: "drawdown.png", x: 7.35, y: 3.85, w: 5.45, h: 2.2 });

s.addShape(p.ShapeType.roundRect, { x: 7.35, y: 1.15, w: 5.45, h: 2.5,
  fill: { color: LIGHT }, rectRadius: 0.08 });
s.addText([
  { text: "ANÁLISE CRÍTICA — O QUE OS NÚMEROS ESCONDEM", options: { fontSize: 12, bold: true, color: RED, breakLine: true, paraSpaceAfter: 6 } },
  { text: "Pior ano: 2016 (−8,4%). Reversões bruscas de tendência são o calcanhar do momentum — a defesa é a diversificação entre 6 classes.", options: { fontSize: 11.5, color: INK, bullet: true, breakLine: true, paraSpaceAfter: 4 } },
  { text: "Transparência: a v1 comparava PnL com caixa a zero contra CDI de 9% a.a. — e “perdia”. Diagnóstico com IA levou à v1.1: margem remunerada a CDI, como na prática.", options: { fontSize: 11.5, color: INK, bullet: true, breakLine: true, paraSpaceAfter: 4 } },
  { text: "Sharpe 0,57 na faixa da literatura (0,5–1,0) — muito acima disso seria sinal de erro.", options: { fontSize: 11.5, color: INK, bullet: true } },
], { x: 7.6, y: 1.32, w: 4.95, h: 2.2, fontFace: F, margin: 0, valign: "top" });

s.addText("Nas duas maiores crises do período, Miyagi ficou no positivo: a perna vendida e o USD/BRL defenderam o book.",
  { x: 7.35, y: 6.25, w: 5.45, h: 0.8, fontFace: F, fontSize: 11.5, italic: true, color: STEEL, margin: 0 });

// ---------------- SLIDE 5 — IA + Conclusão (dark) ----------------
s = p.addSlide();
s.background = { color: DARKBG };
s.addText("IA generativa como coautora — e os próximos golpes",
  { x: 0.55, y: 0.35, w: 12.2, h: 0.65, fontFace: F, fontSize: 28, bold: true, color: WHITE, margin: 0 });

const usos = [
  ["Seleção do universo", "IA executou o funil 26→8: coleta, clustering de correlações, heatmaps. Limitação: sandbox sem acesso a dados de mercado — contornado via navegador."],
  ["Código do backtest", "Gerado por IA a partir de especificação conjunta; validado com dados sintéticos antes dos reais."],
  ["Crítica de resultados", "IA diagnosticou o Sharpe negativo da v1 como artefato contábil (caixa sem remuneração) e propôs a v1.1 — o uso mais valioso: IA como revisora adversarial."],
  ["Identidade do robô", "Emblema gerado por código de IA. Log completo de usos e limitações versionado no repositório."],
];
usos.forEach((u, i) => {
  const y = 1.25 + i * 1.13;
  s.addShape(p.ShapeType.roundRect, { x: 0.55, y, w: 6.6, h: 0.98, fill: { color: CARD_D }, rectRadius: 0.08 });
  s.addText([
    { text: u[0], options: { fontSize: 12, bold: true, color: RED, breakLine: true, paraSpaceAfter: 3 } },
    { text: u[1], options: { fontSize: 10.5, color: LIGHT } },
  ], { x: 0.8, y: y + 0.08, w: 6.1, h: 0.85, fontFace: F, margin: 0, valign: "top" });
});

s.addShape(p.ShapeType.roundRect, { x: 7.45, y: 1.25, w: 5.35, h: 2.6, fill: { color: CARD_D }, rectRadius: 0.08 });
s.addText([
  { text: "PRÓXIMOS PASSOS (ATÉ O RELATÓRIO FINAL)", options: { fontSize: 12, bold: true, color: RED, breakLine: true, paraSpaceAfter: 6 } },
  { text: "Robustez: custos 2×, outras janelas de vol, sub-períodos.", options: { fontSize: 12, color: WHITE, bullet: true, breakLine: true, paraSpaceAfter: 4 } },
  { text: "Combinar horizontes de 1, 3 e 12 meses (média de sinais).", options: { fontSize: 12, color: WHITE, bullet: true, breakLine: true, paraSpaceAfter: 4 } },
  { text: "Satélite BTC (fora do core: só 11,7 anos de histórico, sem 2008).", options: { fontSize: 12, color: WHITE, bullet: true } },
], { x: 7.7, y: 1.42, w: 4.85, h: 2.3, fontFace: F, margin: 0, valign: "top" });

s.addShape(p.ShapeType.roundRect, { x: 7.45, y: 4.05, w: 5.35, h: 1.7, fill: { color: CARD_D }, rectRadius: 0.08 });
s.addText([
  { text: "VIABILIDADE PRÁTICA", options: { fontSize: 12, bold: true, color: RED, breakLine: true, paraSpaceAfter: 6 } },
  { text: "Implementável hoje com BOVA11, futuros da B3 (dólar, índice) e ETFs líquidos. Giro baixo (mensal), capacidade alta, sem dados proprietários.", options: { fontSize: 12, color: WHITE } },
], { x: 7.7, y: 4.22, w: 4.85, h: 1.4, fontFace: F, margin: 0, valign: "top" });

s.addShape(p.ShapeType.roundRect, { x: 7.45, y: 5.95, w: 5.35, h: 1.15, fill: { color: RED }, rectRadius: 0.08 });
s.addText("Simples, disciplinado e defensivo. Como o mestre: primeiro aprende a não perder — depois, vence.",
  { x: 7.7, y: 6.08, w: 4.85, h: 0.9, fontFace: F, fontSize: 13, bold: true, color: WHITE, margin: 0, valign: "middle" });

s.addText("Riscos conhecidos: correlações sobem em crises agudas; reversões rápidas machucam antes do rebalanceamento mensal.",
  { x: 0.55, y: 6.05, w: 6.6, h: 0.8, fontFace: F, fontSize: 10.5, color: LIGHT, margin: 0 });

p.writeFile({ fileName: "prerelatorio_miyagi.pptx" }).then(() => console.log("pptx ok"));
