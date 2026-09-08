"""Jitted random-policy rollout over vmapped environments (smoke/benchmark utility)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from mamora.envs.base import ChannelEnv, EnvState
from mamora.rewards.channels import RewardChannels


class RolloutStats(NamedTuple):
    """Per-agent statistics averaged over the vmapped environments."""

    returns: jax.Array  # (num_agents,) sum of scalar rewards over the rollout
    channel_returns: RewardChannels  # each (num_agents,) sum of that channel
    episode_stats: dict[str, jax.Array]  # each (num_agents,) env indicator at the end


def random_rollout(env: ChannelEnv, key: jax.Array, *, num_envs: int, steps: int) -> RolloutStats:
    """Run `steps` uniform-random steps in `num_envs` parallel copies of `env`."""
    agents = tuple(env.agents)
    num_actions = int(env.action_space(agents[0]).n)

    def sample_actions(key: jax.Array) -> dict[str, jax.Array]:
        keys = jax.random.split(key, len(agents))
        return {
            agent: jax.random.randint(k, (num_envs,), 0, num_actions)
            for agent, k in zip(agents, keys, strict=True)
        }

    def step(
        carry: tuple[EnvState, jax.Array], _: None
    ) -> tuple[tuple[EnvState, jax.Array], tuple[jax.Array, RewardChannels]]:
        state, key = carry
        key, k_act, k_step = jax.random.split(key, 3)
        step_keys = jax.random.split(k_step, num_envs)
        _obs, state, rewards, _dones, info = jax.vmap(env.step)(
            step_keys, state, sample_actions(k_act)
        )
        # (num_agents, num_envs); channels are (num_envs, num_agents) each
        scalar = jnp.stack([rewards[agent] for agent in agents])
        return (state, key), (scalar, info["reward_channels"])

    @jax.jit
    def run(key: jax.Array) -> RolloutStats:
        k_reset, k_run = jax.random.split(key)
        _obs, state = jax.vmap(env.reset)(jax.random.split(k_reset, num_envs))
        (state, _), (rewards, channels) = jax.lax.scan(step, (state, k_run), None, length=steps)
        # rewards: (steps, num_agents, num_envs); channels: (steps, num_envs, num_agents)
        return RolloutStats(
            returns=rewards.sum(axis=0).mean(axis=-1),
            channel_returns=jax.tree.map(lambda c: c.sum(axis=0).mean(axis=0), channels),
            episode_stats=jax.tree.map(
                lambda s: s.mean(axis=0), jax.vmap(env.episode_stats)(state)
            ),
        )

    return run(key)
