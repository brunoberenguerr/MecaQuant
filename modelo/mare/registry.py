"""Run registry: log append-only de todo backtest executado.

Motivo de existir: o Deflated Sharpe Ratio precisa de N — o número de
configurações efetivamente testadas. Sem um registro automático, N vira chute,
e com um assistente de IA no loop o N real chega facilmente às centenas.
Reportar "testamos N=340 configurações e o Sharpe deflacionado é X" é o sinal de
honestidade mais forte que o relatório pode emitir, e ataca diretamente a regra
de ouro do edital.

O arquivo é versionado de propósito. Apagá-lo para "melhorar" o DSR seria fraude.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

DEFAULT_REGISTRY = Path(__file__).resolve().parent.parent / "config" / "registry.jsonl"


def log_run(config_hash: str, metrics: dict[str, Any], *, split: str,
            label: str = "", path: Path | None = None,
            extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Registra um run. Chamado automaticamente pelo engine, nunca à mão."""
    entry = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config_hash": config_hash,
        "split": split,
        "label": label,
        "metrics": {k: _jsonable(v) for k, v in metrics.items()},
    }
    if extra:
        entry["extra"] = {k: _jsonable(v) for k, v in extra.items()}
    path = Path(path or DEFAULT_REGISTRY)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def read_runs(path: Path | None = None) -> list[dict[str, Any]]:
    path = Path(path or DEFAULT_REGISTRY)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def n_trials(path: Path | None = None, split: str | None = None) -> int:
    """N para o Deflated Sharpe: configurações DISTINTAS testadas.

    Rodar o mesmo config duas vezes não é um novo teste; variar um parâmetro é.
    Por padrão conta sobre todos os splits, porque uma escolha feita olhando o
    período de desenho contamina o holdout do mesmo jeito.
    """
    runs = read_runs(path)
    if split is not None:
        runs = [r for r in runs if r.get("split") == split]
    return len({r["config_hash"] for r in runs})


def summary(path: Path | None = None) -> dict[str, Any]:
    runs = read_runs(path)
    return {
        "total_runs": len(runs),
        "distinct_configs": len({r["config_hash"] for r in runs}),
        "by_split": {s: len({r["config_hash"] for r in runs if r.get("split") == s})
                     for s in sorted({r.get("split", "?") for r in runs})},
        "first": runs[0]["ts"] if runs else None,
        "last": runs[-1]["ts"] if runs else None,
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
