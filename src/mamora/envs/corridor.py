"""The timescale corridor: a toy task where the three channels conflict by design.

One agent, three actions, a fixed horizon. Document 03, section 17, defines
the corridor. This version fixes the horizon, so the length of an episode does
not depend on the policy and the dense return of a policy is the sum of what
its actions pay.

- ``harvest`` pays `harvest_reward` at once and makes no progress.
- ``invest`` pays `invest_reward`. After `invest_steps` consecutive invests
  the agent reaches the milestone, which pays the sparse reward once.
- ``commit`` pays `commit_reward`. After `commit_steps` consecutive commits,
  after the milestone, the episode is a success.
- The episode ends at `horizon`. That last step pays the final reward if the
  episode is a success, and nothing otherwise.

Any other action breaks the streak in progress. Success costs the
``invest_steps + commit_steps`` steps that do not harvest, and nothing else:
when the sequence runs makes no difference to the return. The optimal return
is ``(horizon - cost) * harvest_reward + milestone_reward + final_reward``
when the milestone and the final reward are worth more than the harvests
they replace, and ``horizon * harvest_reward`` otherwise. In the control
where every action pays the same dense reward, the dense channel is
indifferent to success, and the conflict disappears.

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
        if self.horizon < self.success_cost:
            raise ValueError(
                f"horizon {self.horizon} is shorter than the {self.success_cost} steps success needs"
            )

    @property
    def success_cost(self) -> int:
        """The steps a success takes away from harvesting."""
        return self.invest_steps + self.commit_steps

    @property
    def optimal_return(self) -> float:
        """The best scalar return: succeed once, and harvest every other step, or never succeed."""
        harvest_all = self.horizon * self.harvest_reward
        succeed = (
            (self.horizon - self.success_cost) * self.harvest_reward
            + self.invest_steps * self.invest_reward
            + self.commit_steps * self.commit_reward
            + self.milestone_reward
            + self.final_reward
        )
        return max(harvest_all, succeed)


class CorridorState(NamedTuple):
    """Where the agent stands in the corridor. Every field is a scalar array."""

    time: jax.Array
    """Steps taken in this episode."""

    invest: jax.Array
    """Consecutive invests so far, before the milestone."""

    milestone: jax.Array
    """True once the milestone is reached."""

    commit: jax.Array
    """Consecutive commits so far, after the milestone and before success."""

    success: jax.Array
    """True once the episode is a success."""


class TimescaleCorridor:
    """The corridor as a :class:`~mamora.contract.ChannelEnv`.

    The observation has five entries, each in ``[0, 1]``: invest progress,
    milestone reached, commit progress, success, and time. It is the whole
    state, so the task is fully observed and a memoryless policy can solve it.

    ``info`` carries the ``milestone`` and ``success`` flags after the step.
    On the last step of an episode they are its outcome.
    """

    num_agents = 1
    num_actions = NUM_ACTIONS
    obs_size = 5

    def __init__(self, config: CorridorConfig | None = None) -> None:
        self.config = config if config is not None else CorridorConfig()

    @property
    def channel_spec(self) -> ChannelSpec:
        return timescale_spec()

    def reset(self, key: jax.Array) -> tuple[jax.Array, CorridorState]:
        """Start at the corridor entrance. The start is deterministic; `key` is unused."""
        del key
        zero = jnp.int32(0)
        no = jnp.bool_(False)
        state = CorridorState(time=zero, invest=zero, milestone=no, commit=zero, success=no)
        return self.observe(state), state

    def observe(self, state: CorridorState) -> jax.Array:
        """The observation of `state`, with shape ``(num_agents, obs_size)``."""
        cfg = self.config
        row = jnp.stack(
            [
                state.invest / cfg.invest_steps,
                state.milestone.astype(jnp.float32),
                state.commit / cfg.commit_steps,
                state.success.astype(jnp.float32),
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
        commit_streak = jnp.where(state.milestone & ~state.success & commit, state.commit + 1, 0)
        success_now = state.milestone & ~state.success & (commit_streak >= cfg.commit_steps)
        success = state.success | success_now

        time = state.time + 1
        episode_done = time >= cfg.horizon

        # The dense reward depends on the action alone, never on progress.
        dense = (
            harvest * cfg.harvest_reward + invest * cfg.invest_reward + commit * cfg.commit_reward
        )
        sparse = milestone_now * cfg.milestone_reward
        final = episode_done * success * cfg.final_reward
        reward = stack([jnp.reshape(r, (1,)).astype(jnp.float32) for r in (dense, sparse, final)])

        reached = CorridorState(
            time=time,
            invest=invest_streak,
            milestone=milestone,
            commit=commit_streak,
            success=success,
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
            info={"milestone": milestone, "success": success},
        )
