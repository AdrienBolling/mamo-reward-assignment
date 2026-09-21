"""The timescale corridor: a toy task where the three channels conflict by design.

One agent, three actions, one corridor. Document 03, section 17, defines it:

- ``harvest`` pays a dense reward at once and makes no progress.
- ``invest`` pays little or nothing. After `invest_steps` consecutive invests
  the agent reaches the milestone, which pays the sparse reward once.
- ``commit`` counts only after the milestone. After `commit_steps` consecutive
  commits the episode ends with success and pays the final reward.

Any other action resets the streak in progress. The episode truncates at
`horizon` steps. The optimal return is the milestone plus the final reward when
the harvest reward is small, and `horizon` times the harvest reward otherwise.
The dense channel therefore pulls the policy away from the final channel by
construction, which gives the diagnostics a known conflict to detect.

The environment is one instance. The runner vectorizes it with ``jax.vmap``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from mamora.contract import ChannelSpec, StepOutput, stack, timescale_spec

HARVEST = 0
INVEST = 1
COMMIT = 2
NUM_ACTIONS = 3


@dataclass(frozen=True, slots=True)
class CorridorConfig:
    """The corridor geometry and the reward of each event.

    `harvest_reward` sets the strength of the temptation. `invest_reward` and
    `commit_reward` exist for the control where every action pays the same
    dense reward, so the conflict between the channels disappears.
    """

    invest_steps: int = 5
    commit_steps: int = 3
    horizon: int = 32
    harvest_reward: float = 0.1
    invest_reward: float = 0.0
    commit_reward: float = 0.0
    milestone_reward: float = 1.0
    final_reward: float = 10.0

    def __post_init__(self) -> None:
        if self.invest_steps < 1 or self.commit_steps < 1:
            raise ValueError("invest_steps and commit_steps must be at least 1")
        shortest = self.invest_steps + self.commit_steps
        if self.horizon < shortest:
            raise ValueError(
                f"horizon {self.horizon} is shorter than the {shortest} steps success needs"
            )


class CorridorState(NamedTuple):
    """Where the agent stands in the corridor. Every field is a scalar array."""

    time: jax.Array
    """Steps taken in this episode."""

    invest: jax.Array
    """Consecutive invests so far, before the milestone."""

    milestone: jax.Array
    """True once the milestone is reached."""

    commit: jax.Array
    """Consecutive commits so far, after the milestone."""


class TimescaleCorridor:
    """The corridor as a :class:`~mamora.contract.ChannelEnv`.

    The observation has four entries, each in ``[0, 1]``: invest progress,
    milestone reached, commit progress, and time. It is the whole state, so the
    task is fully observed and a memoryless policy can solve it.
    """

    num_agents = 1
    num_actions = NUM_ACTIONS
    obs_size = 4

    def __init__(self, config: CorridorConfig | None = None) -> None:
        self.config = config if config is not None else CorridorConfig()

    @property
    def channel_spec(self) -> ChannelSpec:
        return timescale_spec()

    def reset(self, key: jax.Array) -> tuple[jax.Array, CorridorState]:
        """Start at the corridor entrance. The start is deterministic; `key` is unused."""
        del key
        zero = jnp.int32(0)
        state = CorridorState(time=zero, invest=zero, milestone=jnp.bool_(False), commit=zero)
        return self.observe(state), state

    def observe(self, state: CorridorState) -> jax.Array:
        """The observation of `state`, with shape ``(num_agents, obs_size)``."""
        cfg = self.config
        row = jnp.stack(
            [
                state.invest / cfg.invest_steps,
                state.milestone.astype(jnp.float32),
                state.commit / cfg.commit_steps,
                state.time / cfg.horizon,
            ]
        ).astype(jnp.float32)
        return row[None, :]

    def step(self, key: jax.Array, state: CorridorState, actions: jax.Array) -> StepOutput:
        """Apply the action of the single agent, ``actions[0]``."""
        cfg = self.config
        action = actions[0]
        harvest = action == HARVEST
        invest = action == INVEST
        commit = action == COMMIT

        # Progress before the milestone: a streak of invests, broken by anything else.
        invest_streak = jnp.where(~state.milestone & invest, state.invest + 1, 0)
        milestone_now = ~state.milestone & (invest_streak >= cfg.invest_steps)
        milestone = state.milestone | milestone_now

        # Progress after the milestone: a streak of commits, which starts the step after.
        commit_streak = jnp.where(state.milestone & commit, state.commit + 1, 0)
        success = state.milestone & (commit_streak >= cfg.commit_steps)

        time = state.time + 1
        truncated = ~success & (time >= cfg.horizon)
        episode_done = success | truncated

        dense = (
            harvest * cfg.harvest_reward + invest * cfg.invest_reward + commit * cfg.commit_reward
        )
        sparse = milestone_now * cfg.milestone_reward
        final = success * cfg.final_reward
        reward = stack([jnp.reshape(r, (1,)).astype(jnp.float32) for r in (dense, sparse, final)])

        reached = CorridorState(
            time=time, invest=invest_streak, milestone=milestone, commit=commit_streak
        )
        final_obs = self.observe(reached)
        reset_obs, reset_state = self.reset(key)
        obs = jnp.where(episode_done, reset_obs, final_obs)
        next_state = jax.tree.map(
            lambda fresh, kept: jnp.where(episode_done, fresh, kept), reset_state, reached
        )
        return StepOutput(
            obs=obs,
            state=next_state,
            reward=reward,
            done=jnp.reshape(episode_done, (1,)),
            episode_done=episode_done,
            final_obs=final_obs,
            info={"success": success, "milestone": milestone_now, "truncated": truncated},
        )
