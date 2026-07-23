"""Motor de backtest orientado a eventos.

Contrato central: a função de sinal recebe SOMENTE um `AsOfView`, nunca os
painéis completos. Isso torna look-ahead estruturalmente impossível em vez de
depender de disciplina (ver `asof.py`).

Fluxo de cada rebalanceamento:
    sinal calculado no fechamento de `signal_date`
    -> executado no fechamento de `trade_date` (posterior, por construção)
    -> posições rendem de `trade_date`+1 até o próximo `trade_date`, derivando
       com os preços no meio do caminho.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from mare.backtest import costs as cost_model
from mare.backtest.asof import AsOfView, MarketData
from mare.calendar import schedule
from mare.config import Config

SignalFn = Callable[[AsOfView, Config], pd.Series]


@dataclass
class BacktestResult:
    """Tudo que o relatório precisa, sem precisar rodar de novo."""

    returns: pd.Series                      # líquido de custos
    gross_returns: pd.Series
    weights: pd.DataFrame                   # datas de trade × símbolos
    turnover: pd.Series
    costs: pd.Series
    exposure: pd.Series
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def equity_curve(self) -> pd.Series:
        return (1 + self.returns.fillna(0.0)).cumprod()

    def fingerprint(self) -> str:
        """Hash da curva de patrimônio — usado no teste de determinismo."""
        import hashlib
        blob = np.round(self.returns.fillna(0.0).to_numpy(), 12).tobytes()
        return hashlib.sha256(blob).hexdigest()[:16]


def run_backtest(market: MarketData, cfg: Config, signal_fn: SignalFn, *,
                 start: str | None = None, end: str | None = None,
                 risk_free: pd.Series | None = None) -> BacktestResult:
    """Roda o backtest no intervalo pedido."""
    dates = market.dates
    plan = schedule(dates, weekday=cfg.backtest.rebalance_weekday,
                    lag_days=cfg.backtest.signal_lag_days,
                    frequency=cfg.backtest.rebalance)
    plan = _restrict(plan, dates, cfg, start, end)
    if plan.empty:
        raise ValueError("nenhuma data de rebalanceamento no intervalo pedido")

    state = _State(cfg, risk_free)
    for i, row in enumerate(plan.itertuples()):
        target = signal_fn(market.view(row.signal_date), cfg)
        state.rebalance(row.trade_date, target)
        next_trade = (plan.trade_date.iloc[i + 1] if i + 1 < len(plan)
                      else market.dates[-1])
        state.accrue(market.forward_returns(row.trade_date, next_trade))

    return state.result()


@dataclass
class _State:
    """Contabilidade do portfólio. Isolada para manter `run_backtest` legível."""

    cfg: Config
    risk_free: pd.Series | None = None
    held: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    daily: dict = field(default_factory=dict)
    gross_daily: dict = field(default_factory=dict)
    weight_log: dict = field(default_factory=dict)
    turnover_log: dict = field(default_factory=dict)
    cost_log: dict = field(default_factory=dict)
    exposure_log: dict = field(default_factory=dict)

    def rebalance(self, trade_date: pd.Timestamp, target: pd.Series) -> None:
        """Aplica os pesos alvo e cobra o custo do giro no dia da execução."""
        target = target[target.abs() > 1e-10] if not target.empty else target
        union = self.held.index.union(target.index)
        traded = (target.reindex(union).fillna(0.0)
                  - self.held.reindex(union).fillna(0.0)).abs()

        cost = cost_model.trade_costs(traded, self.cfg.costs.active)
        self.turnover_log[trade_date] = float(traded.sum())
        self.cost_log[trade_date] = cost
        self.weight_log[trade_date] = target
        self.exposure_log[trade_date] = float(target.sum())

        self.daily[trade_date] = self.daily.get(trade_date, 0.0) - cost
        self.gross_daily.setdefault(trade_date, 0.0)
        self.held = target

    def accrue(self, forward: pd.DataFrame) -> None:
        """Deixa as posições renderem e derivarem até o próximo rebalanceamento."""
        for date, row in forward.iterrows():
            if self.held.empty:
                gross = self._cash_return(date, 1.0)
            else:
                rets = row.reindex(self.held.index).fillna(0.0)
                invested = float(self.held.sum())
                gross = float((self.held * rets).sum())
                gross += self._cash_return(date, max(0.0, 1.0 - invested))
                grown = self.held * (1 + rets)
                total = grown.sum()
                if total > 0:
                    self.held = grown * (invested / total)
            self.gross_daily[date] = self.gross_daily.get(date, 0.0) + gross
            self.daily[date] = self.daily.get(date, 0.0) + gross

    def _cash_return(self, date: pd.Timestamp, weight: float) -> float:
        """Caixa rende a taxa livre de risco. Assumir zero subestima o retorno."""
        if self.risk_free is None or weight <= 0:
            return 0.0
        rate = self.risk_free.get(date, np.nan)
        return 0.0 if not np.isfinite(rate) else float(weight * rate)

    def result(self) -> BacktestResult:
        index = sorted(self.daily)
        weights = pd.DataFrame(self.weight_log).T.sort_index().fillna(0.0)
        return BacktestResult(
            returns=pd.Series({d: self.daily[d] for d in index}).sort_index(),
            gross_returns=pd.Series(
                {d: self.gross_daily.get(d, 0.0) for d in index}).sort_index(),
            weights=weights,
            turnover=pd.Series(self.turnover_log).sort_index(),
            costs=pd.Series(self.cost_log).sort_index(),
            exposure=pd.Series(self.exposure_log).sort_index(),
        )


def _restrict(plan: pd.DataFrame, dates: pd.DatetimeIndex, cfg: Config,
              start: str | None, end: str | None) -> pd.DataFrame:
    """Corta o cronograma e garante histórico suficiente para o primeiro sinal."""
    warmup = cfg.factor.window + 5
    if len(dates) <= warmup:
        return plan.iloc[0:0]
    earliest = dates[warmup]
    lo = max(pd.Timestamp(start), earliest) if start else earliest
    out = plan[plan.signal_date >= lo]
    if end:
        out = out[out.trade_date <= pd.Timestamp(end)]
    return out.reset_index(drop=True)
