"""Primeira passagem com duas barreiras: função de escala e medida de velocidade.

Por que não as fórmulas de Bertram (2010). Aquelas expressões com Erfi e séries
de digamma resolvem a passagem UNILATERAL, sem stop-loss. O problema real da
estratégia é outro: atingir o alvo ANTES do stop. Isso é um problema de duas
barreiras, que não tem forma fechada elementar — exigiria funções cilíndrico-
parabólicas, e com horizonte finito não tem forma fechada nenhuma.

A saída é mais simples e mais geral. Para qualquer difusão unidimensional, a
teoria de Feller dá tudo por duas quadraturas, via densidade de escala
s(x) = exp(−∫2μ/σ²) e densidade de velocidade m(x) = 1/(σ²s):

    P(atinge b antes de a | X₀=x) = (S(x) − S(a)) / (S(b) − S(a))
    E[T] = 2/(S(b)−S(a)) · [(S(b)−S(x))·∫ₐˣ(S(y)−S(a))m dy
                            + (S(x)−S(a))·∫ₓᵇ(S(b)−S(y))m dy]

A boa ideia de Bertram — ranquear por retorno por unidade de tempo em vez de
s-score puro — sobrevive inteira. Só a álgebra frágil morre.

Trabalhamos em coordenadas normalizadas z = (X−θ)/σ_eq e τ = κt, onde o OU vira
dZ = −Z dτ + √2 dB, com s(z) = e^{z²/2} e m(z) = e^{−z²/2}/2. Isso não é só
conveniência numérica (evita overflow): o z normalizado É o s-score, o que torna
as barreiras diretamente interpretáveis como desvios-padrão do equilíbrio.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid


def _grid(z_low: float, z_high: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Grade em z com a função de escala acumulada S, normalizada com S(a)=0."""
    z = np.linspace(z_low, z_high, n)
    density = np.exp(np.clip(z ** 2 / 2.0, None, 300.0))
    scale = cumulative_trapezoid(density, z, initial=0.0)
    return z, scale


def hit_probability(z0: float, z_stop: float, z_target: float,
                    n_quad: int = 400) -> float:
    """P(atingir `z_target` antes de `z_stop`), partindo de `z0`.

    Para uma compra: z0 e z_stop são negativos (barato / muito barato) e
    z_target é o nível de saída, acima de z0.
    """
    if not (z_stop < z0 < z_target):
        return float("nan")
    z, scale = _grid(z_stop, z_target, n_quad)
    total = scale[-1]
    if total <= 0:
        return float("nan")
    return float(np.interp(z0, z, scale) / total)


def expected_exit_time(z0: float, z_stop: float, z_target: float, kappa: float,
                       n_quad: int = 400) -> float:
    """E[T] até tocar qualquer uma das barreiras, em DIAS.

    O tempo sai em unidades de τ = κt e é convertido dividindo por κ.
    """
    if not (z_stop < z0 < z_target) or not np.isfinite(kappa) or kappa <= 0:
        return float("nan")
    z, scale = _grid(z_stop, z_target, n_quad)
    total = scale[-1]
    if total <= 0:
        return float("nan")

    speed = np.exp(np.clip(-z ** 2 / 2.0, -300.0, None)) / 2.0
    lower = cumulative_trapezoid(scale * speed, z, initial=0.0)
    upper_cum = cumulative_trapezoid((total - scale) * speed, z, initial=0.0)
    upper = upper_cum[-1] - upper_cum

    s0 = float(np.interp(z0, z, scale))
    tau = 2.0 / total * ((total - s0) * np.interp(z0, z, lower)
                         + s0 * np.interp(z0, z, upper))
    return float(tau / kappa)


def trade_economics(z0: float, z_stop: float, z_target: float, kappa: float,
                    sigma_eq: float, cost: float = 0.0,
                    n_quad: int = 400) -> dict[str, float]:
    """Economia esperada de um trade e seu retorno por unidade de tempo.

    `sigma_eq` converte desvios-padrão em retorno: uma reversão de z0 até
    z_target vale (z_target − z0)·σ_eq no resíduo acumulado.

    `mu_s` é o critério de ranqueamento — dois candidatos com o mesmo s-score
    não são equivalentes se um reverte em 8 dias e o outro em 40.
    """
    prob = hit_probability(z0, z_stop, z_target, n_quad)
    horizon = expected_exit_time(z0, z_stop, z_target, kappa, n_quad)
    if not np.isfinite(prob) or not np.isfinite(horizon) or horizon <= 0:
        return {"prob": np.nan, "expected_days": np.nan,
                "expected_return": np.nan, "mu_s": np.nan}

    gain = (z_target - z0) * sigma_eq
    loss = (z0 - z_stop) * sigma_eq
    expected = prob * gain + (1 - prob) * loss - cost
    return {"prob": prob, "expected_days": horizon,
            "expected_return": expected, "mu_s": expected / horizon}


def monte_carlo_check(z0: float, z_stop: float, z_target: float, kappa: float,
                      n_paths: int = 100_000, dt: float = 0.02,
                      max_steps: int = 50_000, bridge: bool = True,
                      seed: int | None = 0) -> dict[str, float]:
    """Valida a quadratura simulando o OU normalizado dZ = −Z dτ + √2 dB.

    Existe para ser rodado nos testes: fórmula fechada sem verificação numérica
    é exatamente o tipo de erro que passa despercebido até o backtest inteiro
    estar contaminado.

    Monitorar barreira só nos instantes amostrados enviesa o resultado — o
    caminho pode cruvar e voltar dentro de um passo. O viés é de ordem √Δt e
    aparece nas duas direções: subestima toques e superestima o tempo de saída.
    A correção de ponte browniana (Broadie–Glasserman–Kou) reintroduz a
    probabilidade de cruzamento entre observações,

        P(cruzou b | z_t, z_{t+1}) = exp(−2(b−z_t)(b−z_{t+1}) / σ²Δt)

    com σ² = 2 para o OU normalizado, e faz a convergência deixar de depender de
    um Δt minúsculo. Passe `bridge=False` para demonstrar o viés no relatório.
    """
    rng = np.random.default_rng(seed)
    decay = np.exp(-dt)
    noise = np.sqrt(1 - decay ** 2)  # variância estacionária do OU normalizado é 1
    var_step = 2.0 * dt              # coeficiente de difusão da ponte local

    z = np.full(n_paths, float(z0))
    alive = np.ones(n_paths, dtype=bool)
    hit = np.zeros(n_paths, dtype=bool)
    steps = np.zeros(n_paths)

    for step in range(max_steps):
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        prev = z[idx]
        nxt = decay * prev + noise * rng.standard_normal(idx.size)
        z[idx] = nxt
        # O cruzamento acontece em algum ponto DENTRO do passo; atribuí-lo ao fim
        # superestima E[T] em ~Δt/2 por caminho. O meio do passo é a escolha não
        # enviesada em primeira ordem.
        steps[idx] = (step + 0.5) * dt

        won = nxt >= z_target
        lost = nxt <= z_stop
        if bridge:
            p_up = np.exp(-2.0 * (z_target - prev) * (z_target - nxt) / var_step)
            p_dn = np.exp(-2.0 * (prev - z_stop) * (nxt - z_stop) / var_step)
            draw = rng.random((2, idx.size))
            won |= ~won & ~lost & (draw[0] < p_up)
            lost |= ~won & ~lost & (draw[1] < p_dn)

        hit[idx[won]] = True
        alive[idx[won | lost]] = False

    return {"prob": float(hit.mean()),
            "expected_days": float(steps.mean() / kappa) if kappa > 0 else np.nan,
            "unresolved": float(alive.mean())}
