"""Timescale corridor: a tiny MDP with conflicting dense, sparse and final rewards.

Roadmap section 17 (`design_docs/03`). One agent, three actions:

- ``harvest`` (0): earns the dense reward and breaks any streak;
- ``invest`` (1): earns the (zero or negative) invest reward and advances a
  streak; ``invest_steps`` consecutive invests reach the milestone, which pays
  the sparse reward once;
- ``commit`` (2): after the milestone, ``commit_steps`` consecutive commits
  end the episode with the final reward. Before the milestone it only breaks
  the invest streak.

The episode also ends after ``horizon`` steps. Every step returns
``info["reward_channels"]`` and the scalar reward is their sum, so the dense
channel rewards the tempting behaviour that the final channel must overrule.

The interface mirrors the JaxMARL `MultiAgentEnv` used by MA-Craftax (dicts
keyed by agent name, ``dones["__all__"]``, auto-reset) so the same rollout and
training code applies to both.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct
from jaxmarl.environments import spaces

from mamora.rewards.channels import RewardChannels

HARVEST, INVEST, COMMIT = 0, 1, 2
NUM_ACTIONS = 3
AGENT = "agent_0"
OBS_SIZE = 4


@dataclass(frozen=True)
class CorridorParams:
    """Static parameters; every field is a config key (`conf/env/corridor.yaml`)."""

    invest_steps: int = 5
    commit_steps: int = 3
    horizon: int = 32
    harvest_reward: float = 0.1
    invest_reward: float = 0.0
    milestone_reward: float = 1.0
    final_reward: float = 10.0


@struct.dataclass
class CorridorState:
    invests: jax.Array  # consecutive invest actions, capped at invest_steps
    milestone: jax.Array  # bool, sticky once reached
    commits: jax.Array  # consecutive commit actions after the milestone
    success: jax.Array  # bool, set on the step that pays the final reward
    timestep: jax.Array


class TimescaleCorridor:
    """Single-agent corridor MDP with reward channels (see module docstring)."""

    def __init__(self, params: CorridorParams | None = None) -> None:
        self.params = params or CorridorParams()
        self.agents: list[str] = [AGENT]
        self.num_agents: int = 1
        self.action_spaces = {AGENT: spaces.Discrete(NUM_ACTIONS)}
        self.observation_spaces = {AGENT: spaces.Box(0.0, 1.0, (OBS_SIZE,), dtype=jnp.float32)}

    def action_space(self, agent: str) -> spaces.Discrete:
        return self.action_spaces[agent]

    def observation_space(self, agent: str) -> spaces.Box:
        return self.observation_spaces[agent]

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.Array) -> tuple[dict[str, jax.Array], CorridorState]:
        del key  # deterministic start
        state = CorridorState(
            invests=jnp.int32(0),
            milestone=jnp.bool_(False),
            commits=jnp.int32(0),
            success=jnp.bool_(False),
            timestep=jnp.int32(0),
        )
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, key: jax.Array, state: CorridorState, actions: dict[str, jax.Array]
    ) -> tuple[
        dict[str, jax.Array], CorridorState, dict[str, jax.Array], dict[str, jax.Array], dict
    ]:
        state_step, channels = self.transition(state, actions[AGENT])
        done = self.is_terminal(state_step)
        _obs_reset, state_reset = self.reset(key)
        state_next = jax.tree.map(lambda a, b: jax.lax.select(done, a, b), state_reset, state_step)
        return (
            self.get_obs(state_next),
            state_next,
            {AGENT: channels.total()[0]},
            {AGENT: done, "__all__": done},
            {"reward_channels": channels, "episode_stats": self.episode_stats(state_step)},
        )

    def transition(
        self, state: CorridorState, action: jax.Array
    ) -> tuple[CorridorState, RewardChannels]:
        """One transition without reset; channels have shape (num_agents,)."""
        p = self.params
        harvest, invest, commit = action == HARVEST, action == INVEST, action == COMMIT

        invests = jnp.where(invest & ~state.milestone, state.invests + 1, 0)
        reached = invests >= p.invest_steps
        milestone = state.milestone | reached
        commits = jnp.where(commit & state.milestone, state.commits + 1, 0)
        success = commits >= p.commit_steps

        dense = jnp.where(harvest, p.harvest_reward, jnp.where(invest, p.invest_reward, 0.0))
        sparse = jnp.where(reached & ~state.milestone, p.milestone_reward, 0.0)
        final = jnp.where(success, p.final_reward, 0.0)
        channels = RewardChannels(
            dense=jnp.asarray(dense, jnp.float32)[None],
            sparse=jnp.asarray(sparse, jnp.float32)[None],
            final=jnp.asarray(final, jnp.float32)[None],
        )
        next_state = CorridorState(
            invests=jnp.minimum(invests, p.invest_steps),
            milestone=milestone,
            commits=commits,
            success=success,
            timestep=state.timestep + 1,
        )
        return next_state, channels

    def is_terminal(self, state: CorridorState) -> jax.Array:
        return state.success | (state.timestep >= self.params.horizon)

    def get_obs(self, state: CorridorState) -> dict[str, jax.Array]:
        p = self.params
        obs = jnp.stack(
            [
                state.invests / p.invest_steps,
                state.milestone.astype(jnp.float32),
                state.commits / p.commit_steps,
                state.timestep / p.horizon,
            ]
        ).astype(jnp.float32)
        return {AGENT: obs}

    def episode_stats(self, state: CorridorState) -> dict[str, jax.Array]:
        """Per-agent (num_agents,) indicators for rollout summaries."""
        return {
            "milestone": state.milestone.astype(jnp.float32)[None],
            "success": state.success.astype(jnp.float32)[None],
        }
