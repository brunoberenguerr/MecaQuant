"""Carregamento e validação da configuração.

`config/base.yaml` é a única fonte de parâmetros. Este módulo transforma o YAML
em dataclasses tipadas (um typo vira erro no load, não um backtest silenciosamente
errado) e produz o hash canônico usado pelo run registry.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "base.yaml"


@dataclass(frozen=True)
class MetaCfg:
    name: str = "MARE"
    descricao: str = ""


@dataclass(frozen=True)
class DataCfg:
    start: str = "2005-01-01"
    end: str = "2026-07-18"
    cache_dir: str = "data/raw"
    benchmark: str = "SPY"
    vix: str = "^VIX"
    riskfree_series: str = "DTB3"


@dataclass(frozen=True)
class SplitsCfg:
    design_start: str = "2007-01-01"
    design_end: str = "2016-12-31"
    holdout_start: str = "2017-01-01"
    holdout_end: str = "2026-07-18"


@dataclass(frozen=True)
class UniverseCfg:
    index: str = "SP500"
    adv_window: int = 20
    min_dollar_volume: float = 20_000_000
    min_price: float = 5.0
    min_valid_returns: int = 200


@dataclass(frozen=True)
class SanityCfg:
    winsorize_abs: float = 0.25
    flag_abs: float = 0.50


@dataclass(frozen=True)
class FactorCfg:
    window: int = 504
    halflife: int = 126
    k_method: str = "marchenko_pastur"
    k_fixed: int = 15
    k_min: int = 5
    k_max: int = 30
    k_hysteresis_days: int = 21


@dataclass(frozen=True)
class ResidualCfg:
    beta_window: int = 504
    beta_gap: int = 60
    resid_window: int = 60
    min_r2: float = 0.10


@dataclass(frozen=True)
class OUCfg:
    kendall_correction: bool = True
    shrinkage: bool = True
    shrink_theta_to_zero: bool = True
    halflife_min: float = 10.0
    halflife_max: float = 45.0
    entry_s: float = -1.25
    exit_s: float = -0.50
    stop_s: float = -3.00
    use_half_life_gate: bool = False


@dataclass(frozen=True)
class PassageCfg:
    enabled: bool = True
    n_quad: int = 400
    min_prob: float = 0.55
    rank_by: str = "mu_s"


@dataclass(frozen=True)
class PortfolioCfg:
    n_names: int = 40
    weighting: str = "inverse_vol"
    ewma_lambda: float = 0.94
    max_weight_mult: float = 2.0
    max_sector_weight: float = 0.25
    vol_target: float = 0.10
    max_leverage: float = 1.0


@dataclass(frozen=True)
class RegimeCfg:
    enabled: bool = True
    model: str = "jump"
    n_states: int = 2
    online_only: bool = True
    standardize: str = "expanding"
    feature_halflives: tuple[int, ...] = (20, 60, 120)
    return_halflife: int = 120
    refit_every_days: int = 126
    fit_lookback: int = 2000
    jump_penalty: float = 1.0
    bear_multiplier: float = 0.5


@dataclass(frozen=True)
class BacktestCfg:
    rebalance: str = "weekly"
    rebalance_weekday: int = 2
    signal_lag_days: int = 1
    initial_capital: float = 1_000_000


@dataclass(frozen=True)
class CostScenario:
    commission_bps: float
    half_spread_bps: float
    impact_coef_bps: float


@dataclass(frozen=True)
class CostsCfg:
    scenario: str = "base"
    scenarios: dict[str, CostScenario] = field(default_factory=dict)
    max_participation: float = 0.05

    @property
    def active(self) -> CostScenario:
        return self.scenarios[self.scenario]


@dataclass(frozen=True)
class ValidationCfg:
    purge_days: int = 10
    embargo_days: int = 10
    wf_train_years: int = 3
    wf_test_years: int = 1
    placebo_draws: int = 1000
    bootstrap_block_days: int = 21
    cscv_splits: int = 16


@dataclass(frozen=True)
class Config:
    meta: MetaCfg
    data: DataCfg
    splits: SplitsCfg
    universe: UniverseCfg
    sanity: SanityCfg
    factor: FactorCfg
    residual: ResidualCfg
    ou: OUCfg
    passage: PassageCfg
    portfolio: PortfolioCfg
    regime: RegimeCfg
    backtest: BacktestCfg
    costs: CostsCfg
    validation: ValidationCfg
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def hash(self) -> str:
        """Hash canônico e estável do config. Identifica o run no registry."""
        return config_hash(self.raw)

    def root(self) -> Path:
        return REPO_ROOT


def _build(cls: type, data: dict[str, Any] | None) -> Any:
    """Instancia a dataclass `cls` a partir de `data`, rejeitando chaves desconhecidas."""
    data = dict(data or {})
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: chaves desconhecidas no YAML: {sorted(unknown)}")
    for f in fields(cls):
        if f.name in data and isinstance(f.type, type) and is_dataclass(f.type):
            data[f.name] = _build(f.type, data[f.name])
    if cls is RegimeCfg and "feature_halflives" in data:
        data["feature_halflives"] = tuple(data["feature_halflives"])
    return cls(**data)


def _build_costs(data: dict[str, Any] | None) -> CostsCfg:
    data = dict(data or {})
    scen = {k: CostScenario(**v) for k, v in (data.pop("scenarios", {}) or {}).items()}
    cfg = _build(CostsCfg, {**data, "scenarios": {}})
    object.__setattr__(cfg, "scenarios", scen)
    if cfg.scenario not in scen:
        raise ValueError(f"cenário de custo '{cfg.scenario}' não está em {sorted(scen)}")
    return cfg


def config_hash(raw: dict[str, Any]) -> str:
    """SHA-256 (12 chars) do config serializado canonicamente."""
    blob = json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Lê o YAML e devolve um `Config` validado.

    `overrides` aceita caminhos pontilhados, ex.: ``load_config(**{"portfolio.n_names": 20})``.
    Overrides entram no hash, então cada variação testada vira um run distinto no
    registry — que é exatamente de onde sai o N honesto do Deflated Sharpe.
    """
    path = Path(path or DEFAULT_CONFIG)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for dotted, value in overrides.items():
        _set_dotted(raw, dotted, value)
    return Config(
        meta=_build(MetaCfg, raw.get("meta")),
        data=_build(DataCfg, raw.get("data")),
        splits=_build(SplitsCfg, raw.get("splits")),
        universe=_build(UniverseCfg, raw.get("universe")),
        sanity=_build(SanityCfg, raw.get("sanity")),
        factor=_build(FactorCfg, raw.get("factor")),
        residual=_build(ResidualCfg, raw.get("residual")),
        ou=_build(OUCfg, raw.get("ou")),
        passage=_build(PassageCfg, raw.get("passage")),
        portfolio=_build(PortfolioCfg, raw.get("portfolio")),
        regime=_build(RegimeCfg, raw.get("regime")),
        backtest=_build(BacktestCfg, raw.get("backtest")),
        costs=_build_costs(raw.get("costs")),
        validation=_build(ValidationCfg, raw.get("validation")),
        raw=raw,
    )


def _set_dotted(d: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value
