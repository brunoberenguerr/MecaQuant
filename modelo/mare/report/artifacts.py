"""Produção e persistência dos artefatos de um backtest completo.

Centraliza o que o script de validação e o dashboard consomem, para que os dois
partam exatamente dos mesmos números — divergência entre o que o relatório
mostra e o que o dashboard plota é um jeito clássico de perder credibilidade.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mare.backtest.engine import BacktestResult, run_backtest
from mare.backtest.overlays import ablation
from mare.config import Config
from mare.io.market import Bundle
from mare.portfolio import regime as regime_mod
from mare.portfolio.construct import SignalDiagnostics, make_signal_fn


@dataclass
class RunArtifacts:
    result: BacktestResult
    diagnostics: SignalDiagnostics
    regime_states: pd.Series
    ablation: pd.DataFrame
    benchmark: pd.Series
    risk_free: pd.Series


def compute_regime(bundle: Bundle, cfg: Config) -> pd.Series:
    """Estado de regime filtrado (online, sem look-ahead) sobre o benchmark."""
    feats = regime_mod.regime_features(
        bundle.benchmark.dropna(),
        halflives=tuple(cfg.regime.feature_halflives),
        return_halflife=cfg.regime.return_halflife)
    std = regime_mod.expanding_standardize(feats)
    return regime_mod.online_regime(
        std, n_states=cfg.regime.n_states, jump_penalty=cfg.regime.jump_penalty,
        refit_every=cfg.regime.refit_every_days, lookback=cfg.regime.fit_lookback)


def build_artifacts(bundle: Bundle, cfg: Config, start: str, end: str) -> RunArtifacts:
    """Roda o backtest base, o regime e a ablação de overlays."""
    diagnostics = SignalDiagnostics()
    signal_fn = make_signal_fn(sectors=bundle.sectors, diagnostics=diagnostics)
    result = run_backtest(bundle.market, cfg, signal_fn, start=start, end=end,
                          risk_free=bundle.risk_free)

    states = compute_regime(bundle, cfg)
    multiplier = regime_mod.exposure_multiplier(
        states, bear_multiplier=cfg.regime.bear_multiplier,
        n_states=cfg.regime.n_states)

    table = ablation(result.returns, multiplier, cfg.portfolio.vol_target,
                     max_leverage=cfg.portfolio.max_leverage,
                     risk_free=bundle.risk_free)

    return RunArtifacts(
        result=result, diagnostics=diagnostics,
        regime_states=states.reindex(result.returns.index).ffill(),
        ablation=table,
        benchmark=bundle.benchmark.reindex(result.returns.index),
        risk_free=bundle.risk_free.reindex(result.returns.index))


def save_artifacts(artifacts: RunArtifacts, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts.ablation.to_parquet(out_dir / "ablation.parquet")
    artifacts.result.weights.to_parquet(out_dir / "weights.parquet")
    pd.DataFrame({
        "returns": artifacts.result.returns,
        "benchmark": artifacts.benchmark,
        "risk_free": artifacts.risk_free,
        "regime": artifacts.regime_states,
    }).to_parquet(out_dir / "series.parquet")
    trades = artifacts.diagnostics.trades_frame()
    if not trades.empty:
        trades.to_parquet(out_dir / "trades.parquet")
