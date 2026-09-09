"""Metrics logging: one JSON line per training iteration, plus a console summary."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

CONSOLE_KEYS: tuple[str, ...] = (
    "episode/return",
    "episode/return_dense",
    "episode/return_sparse",
    "episode/return_final",
    "loss/entropy",
    "grad/cos/dense_final",
    "grad/norm_ratio/dense_sparse",
)


def to_scalars(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Device arrays -> Python floats (NaN stays NaN)."""
    return {k: float(np.asarray(v)) for k, v in metrics.items()}


class MetricsLogger:
    """Appends `{"iteration", "timesteps", ...metrics}` lines to `path` as JSONL."""

    def __init__(self, path: Path, console_every: int = 1) -> None:
        self.path = path
        self.console_every = console_every
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a")

    def log(self, iteration: int, timesteps: int, metrics: Mapping[str, Any]) -> dict[str, float]:
        scalars = to_scalars(metrics)
        record = {"iteration": iteration, "timesteps": timesteps, **scalars}
        self._file.write(json.dumps(record, allow_nan=True) + "\n")
        self._file.flush()
        if iteration % self.console_every == 0:
            shown = ", ".join(
                f"{k.split('/', 1)[1]}={scalars[k]:.3g}"
                for k in CONSOLE_KEYS
                if k in scalars and not math.isnan(scalars[k])
            )
            log.info("it %d, %d steps: %s", iteration, timesteps, shown)
        return scalars

    def close(self) -> None:
        self._file.close()
