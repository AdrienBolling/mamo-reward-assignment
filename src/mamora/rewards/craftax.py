"""Reward channels for MA-Craftax (Craftax-MA and Craftax-Coop).

The upstream per-agent reward is the sum of two terms (see ``craftax_step`` in
``craftax_ma/game_logic.py`` and ``craftax_coop/game_logic.py``):

- achievement reward: first-time achievement unlocks times a coefficient
  (1, 3, 5 or 8) from ``ACHIEVEMENT_REWARD_MAP``;
- health reward: 0.1 times the health change during the step. Craftax-MA
  masks this term with the alive flag from the start of the step; Craftax-Coop
  does not.

When ``EnvParams.shared_reward`` is on (Craftax-Coop default) the per-agent
rewards are summed and every agent receives the sum.

Channel mapping used here:

- ``dense``  = health reward;
- ``sparse`` = achievement reward;
- ``final``  = 0. MA-Craftax has no terminal reward; the episode ends on
  death of all players, on the step limit, or when the boss is beaten.

The channels sum to the upstream scalar reward. Sharing is applied per channel,
so the identity also holds under ``shared_reward``.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Literal

import jax
import jax.numpy as jnp

from mamora.envs.factory import CraftaxEnv
from mamora.rewards.channels import RewardChannels

CraftaxFamily = Literal["ma", "coop"]

# MA-Craftax `EnvState` is a flax struct dataclass from an untyped package.
type EnvState = Any

HEALTH_REWARD_SCALE = 0.1


def craftax_family(env_name: str) -> CraftaxFamily:
    """Map an MA-Craftax environment name to its game family."""
    return "coop" if "Coop" in env_name else "ma"


def achievement_coefficients(family: CraftaxFamily) -> jax.Array:
    """Per-achievement reward coefficients of a family, as float32."""
    # Imported here so that only the family in use loads its assets.
    if family == "coop":
        from craftax_coop.constants import ACHIEVEMENT_REWARD_MAP
    else:
        from craftax_ma.constants import ACHIEVEMENT_REWARD_MAP
    return jnp.asarray(ACHIEVEMENT_REWARD_MAP, dtype=jnp.float32)


class CraftaxChannelEnv:
    """MA-Craftax environment whose `step` also returns the reward channels.

    `step` mirrors the JaxMARL `MultiAgentEnv.step` auto-reset, but computes the
    channels from the pre-reset transition. The scalar rewards are the upstream
    ones, unchanged. `info["reward_channels"]` holds a :class:`RewardChannels`
    with one value per agent, in `agents` order.
    """

    def __init__(self, env: CraftaxEnv, family: CraftaxFamily) -> None:
        self.env = env
        self.family = family
        self.agents: list[str] = list(env.agents)
        self.num_agents: int = len(self.agents)
        self.coefficients = achievement_coefficients(family)
        self.mask_health_by_alive = family == "ma"
        self.shared_reward = bool(env.default_params.shared_reward)

    def action_space(self, agent: str) -> Any:
        return self.env.action_space(agent)

    def observation_space(self, agent: str) -> Any:
        return self.env.observation_space(agent)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.Array) -> tuple[dict[str, jax.Array], EnvState]:
        return self.env.reset(key)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, key: jax.Array, state: EnvState, actions: dict[str, jax.Array]
    ) -> tuple[dict[str, jax.Array], EnvState, dict[str, jax.Array], dict[str, jax.Array], dict]:
        key, key_reset = jax.random.split(key)
        obs_step, state_step, rewards, dones, info = self.env.step_env(key, state, actions)
        channels = self.channels(state, state_step)

        obs_reset, state_reset = self.env.reset(key_reset)
        done = dones["__all__"]
        state_next = jax.tree.map(lambda a, b: jax.lax.select(done, a, b), state_reset, state_step)
        obs = jax.tree.map(lambda a, b: jax.lax.select(done, a, b), obs_reset, obs_step)
        return obs, state_next, rewards, dones, {**info, "reward_channels": channels}

    def channels(self, prev_state: EnvState, next_state: EnvState) -> RewardChannels:
        """Reward channels of the transition `prev_state -> next_state` (no reset)."""
        unlocked = next_state.achievements.astype(jnp.int32) - prev_state.achievements.astype(
            jnp.int32
        )
        sparse = (unlocked * self.coefficients).sum(axis=1)
        dense = (next_state.player_health - prev_state.player_health) * HEALTH_REWARD_SCALE
        if self.mask_health_by_alive:
            dense = jnp.where(prev_state.player_alive, dense, 0.0)
        final = jnp.zeros_like(dense)
        if self.shared_reward:
            dense, sparse, final = (_share(r) for r in (dense, sparse, final))
        return RewardChannels(dense=dense, sparse=sparse, final=final)


def _share(reward: jax.Array) -> jax.Array:
    """Every agent receives the sum over agents (upstream `shared_reward`)."""
    return jnp.broadcast_to(reward.sum(), reward.shape)
