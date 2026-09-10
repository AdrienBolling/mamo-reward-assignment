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
- ``final``  = episode outcome, paid once on the terminal step. MA-Craftax has
  no terminal reward of its own, so this channel is a research addition: one of
  three values for the three ways an episode ends, ordered
  ``death < timeout < boss`` (:class:`OutcomeReward`). Every agent receives the
  same value; it is not summed under ``shared_reward``. The upstream step limit
  is 100 000, so the wrapper enforces its own ``max_episode_steps`` to make
  timeouts happen.

The scalar reward returned by `step` is ``upstream reward + final``, and the
channels sum to it. Sharing of ``dense`` and ``sparse`` follows the upstream
``shared_reward`` rule.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import jax
import jax.numpy as jnp

from mamora.envs.base import EnvState
from mamora.envs.factory import CraftaxEnv
from mamora.rewards.channels import RewardChannels

CraftaxFamily = Literal["ma", "coop"]

HEALTH_REWARD_SCALE = 0.1
DEFAULT_MAX_EPISODE_STEPS = 10_000
OUTCOMES: tuple[str, ...] = ("death", "timeout", "boss")


@dataclass(frozen=True)
class OutcomeReward:
    """Final-channel value of each episode outcome; must satisfy death < timeout < boss."""

    death: float = -1.0
    timeout: float = 0.0
    boss: float = 10.0

    def __post_init__(self) -> None:
        if not self.death < self.timeout < self.boss:
            msg = f"outcome rewards must satisfy death < timeout < boss, got {self}"
            raise ValueError(msg)

    @classmethod
    def from_config(cls, value: OutcomeReward | Mapping[str, float] | None) -> OutcomeReward:
        if value is None:
            return cls()
        if isinstance(value, OutcomeReward):
            return value
        return cls(**{str(k): float(v) for k, v in value.items()})


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
    channels from the pre-reset transition and adds the wrapper's episode limit.
    The scalar rewards are the upstream ones plus the final channel.
    `info["reward_channels"]` holds a :class:`RewardChannels` with one value per
    agent, in `agents` order; `info["episode_stats"]` holds the achievement
    count and the outcome indicators of the terminal step.
    """

    def __init__(
        self,
        env: CraftaxEnv,
        family: CraftaxFamily,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        final_reward: OutcomeReward | Mapping[str, float] | None = None,
    ) -> None:
        self.env = env
        self.family = family
        self.agents: list[str] = list(env.agents)
        self.num_agents: int = len(self.agents)
        self.coefficients = achievement_coefficients(family)
        self.mask_health_by_alive = family == "ma"
        self.shared_reward = bool(env.default_params.shared_reward)
        self.max_episode_steps = int(max_episode_steps)
        self.final_reward = OutcomeReward.from_config(final_reward)

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
        done = jnp.logical_or(dones["__all__"], state_step.timestep >= self.max_episode_steps)
        outcome = self.outcome(state_step, done)
        channels = self.channels(state, state_step, outcome)
        rewards = {a: rewards[a] + channels.final[i] for i, a in enumerate(self.agents)}
        dones = dict.fromkeys(self.agents, done) | {"__all__": done}

        obs_reset, state_reset = self.env.reset(key_reset)
        state_next = jax.tree.map(lambda a, b: jax.lax.select(done, a, b), state_reset, state_step)
        obs = jax.tree.map(lambda a, b: jax.lax.select(done, a, b), obs_reset, obs_step)
        info = {
            **info,
            "reward_channels": channels,
            "episode_stats": self.episode_stats(state_step)
            | _per_agent_indicators(outcome, self.num_agents),
        }
        return obs, state_next, rewards, dones, info

    def episode_stats(self, state: EnvState) -> dict[str, jax.Array]:
        """Per-agent (num_agents,) count of achievements unlocked so far."""
        return {"achievements": state.achievements.sum(axis=-1).astype(jnp.float32)}

    def outcome(self, state: EnvState, done: jax.Array) -> dict[str, jax.Array]:
        """Mutually exclusive outcome flags of a terminal step (all false when not done).

        `boss` when the boss is beaten, else `death` when no player is alive,
        else `timeout`.
        """
        boss = state.boss_progress >= self.env.static_env_params.num_levels - 1
        dead = jnp.logical_not(state.player_alive.any())
        return {
            "boss": done & boss,
            "death": done & ~boss & dead,
            "timeout": done & ~boss & ~dead,
        }

    def channels(
        self, prev_state: EnvState, next_state: EnvState, outcome: dict[str, jax.Array]
    ) -> RewardChannels:
        """Reward channels of the transition `prev_state -> next_state` (no reset)."""
        unlocked = next_state.achievements.astype(jnp.int32) - prev_state.achievements.astype(
            jnp.int32
        )
        sparse = (unlocked * self.coefficients).sum(axis=1)
        dense = (next_state.player_health - prev_state.player_health) * HEALTH_REWARD_SCALE
        if self.mask_health_by_alive:
            dense = jnp.where(prev_state.player_alive, dense, 0.0)
        if self.shared_reward:
            dense, sparse = _share(dense), _share(sparse)
        value = self.final_reward
        final_scalar = (
            outcome["boss"] * value.boss
            + outcome["death"] * value.death
            + outcome["timeout"] * value.timeout
        )
        final = jnp.broadcast_to(jnp.asarray(final_scalar, dense.dtype), dense.shape)
        return RewardChannels(dense=dense, sparse=sparse, final=final)


def _per_agent_indicators(outcome: dict[str, jax.Array], num_agents: int) -> dict[str, jax.Array]:
    """Outcome flags as per-agent (num_agents,) float indicators."""
    return {k: jnp.broadcast_to(v.astype(jnp.float32), (num_agents,)) for k, v in outcome.items()}


def _share(reward: jax.Array) -> jax.Array:
    """Every agent receives the sum over agents (upstream `shared_reward`)."""
    return jnp.broadcast_to(reward.sum(), reward.shape)
