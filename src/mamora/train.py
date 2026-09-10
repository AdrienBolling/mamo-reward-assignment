"""`mamora-train`: PPO with credit-channel forensics, configured by `conf/train.yaml`."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import jax
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from mamora.algos.ppo import PPO, PPOConfig, RunConfig
from mamora.config import env_params, to_dict
from mamora.envs.factory import make_channel_env
from mamora.metrics import MetricsLogger
from mamora.paths import CONF_DIR

log = logging.getLogger(__name__)


def train(cfg: DictConfig, output_dir: Path) -> dict[str, float]:
    """Run the training loop; return the metrics of the last iteration."""
    env = make_channel_env(cfg.env.name, env_params(cfg))
    run = RunConfig(
        num_envs=cfg.num_envs,
        num_steps=cfg.num_steps,
        num_iterations=int(cfg.total_timesteps) // (cfg.num_envs * cfg.num_steps),
        forensics_every=cfg.forensics_every,
    )
    ppo = PPO(env, PPOConfig(**to_dict(cfg.algo)), run)
    runner = ppo.init(jax.random.PRNGKey(cfg.seed))
    logger = MetricsLogger(output_dir / "metrics.jsonl", console_every=cfg.log_every)
    log.info(
        "%s: %d agents, %d iterations of %d envs x %d steps",
        cfg.env.name,
        ppo.num_agents,
        run.num_iterations,
        run.num_envs,
        run.num_steps,
    )
    scalars: dict[str, float] = {}
    for iteration in range(run.num_iterations):
        runner, metrics = ppo.iteration(runner, iteration)
        timesteps = (iteration + 1) * run.num_envs * run.num_steps
        scalars = logger.log(iteration, timesteps, metrics)
    logger.close()
    return scalars


@hydra.main(config_path=str(CONF_DIR), config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point: `uv run mamora-train [overrides...]`."""
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))
    if cfg.dry_run:
        log.info("dry_run=true: config composed, exiting before any computation")
        return
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    train(cfg, output_dir)


if __name__ == "__main__":
    main()
