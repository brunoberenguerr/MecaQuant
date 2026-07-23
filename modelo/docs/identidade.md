# MARÉ — identidade do robô

## Nome

**MARÉ** — *Mean-reversion Adaptive Residual Engine*.

O acrônimo descreve a máquina em inglês; a palavra descreve a ideia em português.
Uma maré é o exemplo mais familiar de reversão à média que existe: sobe e desce
em torno de um nível, de forma previsível, movida por forças que a gente entende.
E — o que casa exatamente com a nossa tese — a maré tem **regimes**: há a maré
mansa de um dia calmo e a ressaca de uma tempestade. A estratégia foi construída
justamente para saber a diferença entre os dois, e recolher as redes quando o mar
vira.

## O que MARÉ faz, em uma frase

MARÉ compra ações do S&P 500 que caíram além do que os fatores de mercado
explicam — a parte "puxada pela corrente", não pela notícia — apostando que o
nível volta; e reduz a exposição quando o mar fica revolto, porque em tempestade
a queda costuma ser informação, não ruído.

## Princípios de projeto (a personalidade da máquina)

1. **Cética por construção.** MARÉ não confia no próprio backtest. Todo número
   passa por teste de look-ahead, deflação de Sharpe e placebo. O código levanta
   exceção se alguém tentar espiar o futuro.
2. **Honesta sobre o que não funciona.** MARÉ mediu que o próprio refinamento
   matemático mais sofisticado não paga, e disse isso em vez de escondê-lo. O
   resultado negativo é parte da identidade, não um constrangimento.
3. **Cuidadosa com a jornada, não só com o destino.** O valor que MARÉ entrega
   está tanto na redução de drawdown quanto no retorno — a maré protege o barco,
   não só enche a rede.

## Elementos visuais sugeridos

- **Paleta**: azul-maré (o azul da paleta de dados, `#2a78d6`) para a série
  principal; cinza para o benchmark; vermelho reservado para drawdown e regime de
  tempestade. Já é a paleta do dashboard.
- **Símbolo**: uma onda estilizada que também lê como uma curva de reversão à
  média — sobe, cruza a linha do equilíbrio, e volta. A linha do equilíbrio é o θ
  do Ornstein–Uhlenbeck.
- **Mascote (opcional)**: um farol — não navega, sinaliza o regime. Aceso em mar
  calmo (posição cheia), piscando na tempestade (exposição reduzida).

## Uma linha para a apresentação

> "MARÉ não tenta adivinhar o oceano. Ela conhece o nível ao qual a água volta, e
> sabe quando é tempestade demais para estar na água."
