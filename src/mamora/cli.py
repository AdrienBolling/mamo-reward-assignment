"""`mamora-run`: compose a Hydra config and run a random-policy rollout."""

from __future__ import annotations

import logging
from typing import Any

import hydra
import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf

from mamora.envs.factory import make_channel_env
from mamora.paths import CONF_DIR
from mamora.rollout import random_rollout

log = logging.getLogger(__name__)


def env_params(cfg: DictConfig) -> dict[str, Any]:
    """The `env.params` mapping of the config as a plain dict (empty if absent)."""
    params = OmegaConf.to_container(cfg.env.get("params", {}), resolve=True)
    if not isinstance(params, dict):
        msg = f"env.params must be a mapping, got {type(params).__name__}"
        raise TypeError(msg)
    return {str(k): v for k, v in params.items()}


@hydra.main(config_path=str(CONF_DIR), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point: `uv run mamora-run [overrides...]`."""
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))
    if cfg.dry_run:
        log.info("dry_run=true: config composed, exiting before any computation")
        return
    env = make_channel_env(cfg.env.name, env_params(cfg))
    stats = random_rollout(
        env,
        jax.random.PRNGKey(cfg.seed),
        num_envs=cfg.rollout.num_envs,
        steps=cfg.rollout.steps,
    )
    returns = np.asarray(stats.returns)
    channels = np.asarray(stats.channel_returns.stack())  # (channel, agent)
    peak_stats = {k: np.asarray(v) for k, v in stats.peak_stats.items()}
    for i, agent in enumerate(env.agents):
        extra = ", ".join(f"peak {k} {v[i]:.2f}" for k, v in peak_stats.items())
        log.info(
            "%s: mean return %.3f (dense %.3f, sparse %.3f, final %.3f), %s",
            agent,
            returns[i],
            channels[0, i],
            channels[1, i],
            channels[2, i],
            extra,
        )


if __name__ == "__main__":
    main()
