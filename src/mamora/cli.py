"""`mamora-run`: compose a Hydra config and run a random-policy rollout."""

from __future__ import annotations

import logging

import hydra
import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf

from mamora.envs.factory import make_env
from mamora.paths import CONF_DIR
from mamora.rollout import random_rollout

log = logging.getLogger(__name__)


@hydra.main(config_path=str(CONF_DIR), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point: `uv run mamora-run [overrides...]`."""
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))
    if cfg.dry_run:
        log.info("dry_run=true: config composed, exiting before any computation")
        return
    env = make_env(cfg.env.name)
    stats = random_rollout(
        env,
        jax.random.PRNGKey(cfg.seed),
        num_envs=cfg.rollout.num_envs,
        steps=cfg.rollout.steps,
    )
    returns = np.asarray(stats.returns)
    achievements = np.asarray(stats.achievements)
    for i, agent in enumerate(env.agents):
        log.info(
            "%s: mean return %.3f, mean achievements unlocked %.2f",
            agent,
            returns[i],
            achievements[i],
        )


if __name__ == "__main__":
    main()
