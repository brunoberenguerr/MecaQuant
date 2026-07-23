"""Calendário de rebalanceamento e o lag explícito entre sinal e execução.

Separar `signal_date` de `trade_date` num módulo próprio é deliberado: é o ponto
onde look-ahead entra na maioria dos backtests. Aqui a regra fica visível e testável.

Convenção: sinal calculado com dados até o fechamento de `signal_date`, executado
no fechamento de `trade_date`, com `trade_date` estritamente posterior.
"""

from __future__ import annotations

import pandas as pd


def trading_days(index: pd.DatetimeIndex, start: str | None = None,
                 end: str | None = None) -> pd.DatetimeIndex:
    """Recorta o calendário de pregões para o intervalo pedido."""
    days = pd.DatetimeIndex(index).sort_values().unique()
    if start is not None:
        days = days[days >= pd.Timestamp(start)]
    if end is not None:
        days = days[days <= pd.Timestamp(end)]
    return pd.DatetimeIndex(days)


def rebalance_dates(index: pd.DatetimeIndex, weekday: int = 2,
                    frequency: str = "weekly") -> pd.DatetimeIndex:
    """Datas de execução.

    `weekday` segue a convenção do pandas (0=segunda ... 4=sexta); 2 = quarta-feira.
    Se o pregão-alvo não existir na semana (feriado), usa o último pregão anterior
    daquela semana — nunca o seguinte, que seria look-ahead.
    """
    days = pd.DatetimeIndex(index).sort_values().unique()
    if frequency == "daily":
        return pd.DatetimeIndex(days)
    if frequency == "monthly":
        return _last_of_period(days, "M")
    if frequency != "weekly":
        raise ValueError(f"frequência não suportada: {frequency}")

    frame = pd.DataFrame({"date": days}, index=days)
    frame["week"] = days.isocalendar().year.astype(str) + "-" + \
        days.isocalendar().week.astype(str).str.zfill(2)
    frame["dow"] = days.dayofweek
    eligible = frame[frame["dow"] <= weekday]
    picked = eligible.groupby("week")["date"].max()
    return pd.DatetimeIndex(sorted(picked.values))


def _last_of_period(days: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    ser = pd.Series(days, index=days)
    return pd.DatetimeIndex(sorted(ser.resample(freq).last().dropna().values))


def signal_date(trade_date: pd.Timestamp, index: pd.DatetimeIndex,
                lag_days: int = 1) -> pd.Timestamp:
    """Último pregão cujo fechamento pode alimentar o sinal executado em `trade_date`.

    Com `lag_days=1`, um trade na quarta usa dados até o fechamento de terça.
    `lag_days=0` é permitido apenas para testes; em produção significa executar ao
    mesmo preço que gerou o sinal, o que é irrealista.
    """
    if lag_days < 0:
        raise ValueError("lag_days não pode ser negativo")
    days = pd.DatetimeIndex(index).sort_values()
    pos = days.searchsorted(pd.Timestamp(trade_date), side="right") - 1
    target = pos - lag_days
    if target < 0:
        raise IndexError(f"histórico insuficiente para {trade_date} com lag {lag_days}")
    return days[target]


def schedule(index: pd.DatetimeIndex, weekday: int = 2, lag_days: int = 1,
             frequency: str = "weekly") -> pd.DataFrame:
    """Tabela (signal_date, trade_date) do backtest inteiro.

    Materializar o cronograma é o que permite auditá-lo: o teste de look-ahead
    verifica que nenhuma `signal_date` é maior ou igual à sua `trade_date`.
    """
    trades = rebalance_dates(index, weekday=weekday, frequency=frequency)
    rows = []
    for trade in trades:
        try:
            rows.append({"signal_date": signal_date(trade, index, lag_days),
                         "trade_date": trade})
        except IndexError:
            continue
    out = pd.DataFrame(rows)
    if not out.empty and (out["signal_date"] >= out["trade_date"]).any():
        raise AssertionError("cronograma inválido: signal_date >= trade_date")
    return out
