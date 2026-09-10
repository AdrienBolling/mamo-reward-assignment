"""Helpers to turn Hydra config nodes into plain Python objects."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


def to_dict(node: DictConfig | None) -> dict[str, Any]:
    """A config node as a plain `dict[str, Any]` (empty for `None`)."""
    if node is None:
        return {}
    container = OmegaConf.to_container(node, resolve=True)
    if not isinstance(container, dict):
        msg = f"expected a mapping node, got {type(container).__name__}"
        raise TypeError(msg)
    return {str(k): v for k, v in container.items()}


def env_params(cfg: DictConfig) -> dict[str, Any]:
    """The `env.params` mapping of a config (empty if absent)."""
    return to_dict(cfg.env.get("params"))
