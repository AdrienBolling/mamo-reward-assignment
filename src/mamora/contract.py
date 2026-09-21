"""The reward-channel contract.

A credit channel is one reward stream that keeps its identity through the value
and the gradient computation. An environment declares its channels once, in a
:class:`ChannelSpec`, and then reports one reward for each channel at each step.

Conventions:

- The channel axis is axis 0 of every reward array. See :data:`CHANNEL_AXIS`.
- A reward array has shape ``(num_channels, num_agents)``.
- A :class:`ChannelSpec` is static. It is frozen and hashable, so a jitted
  function can take it as a static argument.
- The channels sum to the scalar reward of the environment, unless the
  connector or the environment documents a difference.

The spec carries two labels for each channel. ``granularity`` says how often the
channel pays. ``objective`` says which objective the channel belongs to. A
single-objective experiment uses one objective and three granularities. A
multi-objective experiment crosses the two; see :func:`crossed_spec`.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

CHANNEL_AXIS = 0

# Environment state. The concrete type belongs to the environment.
type EnvState = Any


class Granularity(enum.Enum):
    """How often a channel pays, relative to the episode."""

    DENSE = "dense"
    """Frequent, low delay. Survival, resources, local competence."""

    SPARSE = "sparse"
    """Infrequent events. Achievements, milestones, subgoals."""

    FINAL = "final"
    """End of the episode. Task success, long-horizon outcome."""


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    """The channels of one environment, in the order of the channel axis.

    Args:
        names: The name of each channel. The names are unique.
        granularity: The granularity of each channel.
        objective: The objective index of each channel, into `objectives`.
        objectives: The name of each objective. One name for a single objective.
    """

    names: tuple[str, ...]
    granularity: tuple[Granularity, ...]
    objective: tuple[int, ...]
    objectives: tuple[str, ...] = ("return",)

    def __post_init__(self) -> None:
        n = len(self.names)
        if n == 0:
            raise ValueError("a spec needs at least one channel")
        if len(self.granularity) != n or len(self.objective) != n:
            raise ValueError(
                "names, granularity and objective must have the same length; "
                f"got {n}, {len(self.granularity)}, {len(self.objective)}"
            )
        if len(set(self.names)) != n:
            raise ValueError(f"channel names must be unique; got {self.names}")
        if len(set(self.objectives)) != len(self.objectives):
            raise ValueError(f"objective names must be unique; got {self.objectives}")
        for index in self.objective:
            if not 0 <= index < len(self.objectives):
                raise ValueError(f"objective index {index} is outside {self.objectives}")

    def __len__(self) -> int:
        return len(self.names)

    @property
    def num_channels(self) -> int:
        """The size of the channel axis."""
        return len(self.names)

    @property
    def num_objectives(self) -> int:
        """The number of objectives the channels belong to."""
        return len(self.objectives)

    def index(self, name: str) -> int:
        """The position of `name` on the channel axis."""
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"no channel named {name!r} in {self.names}") from None

    def select(
        self,
        *,
        granularity: Granularity | None = None,
        objective: str | None = None,
    ) -> tuple[int, ...]:
        """The positions of the channels that match every given label."""
        wanted = None if objective is None else self.objectives.index(objective)
        return tuple(
            i
            for i in range(len(self))
            if (granularity is None or self.granularity[i] is granularity)
            and (wanted is None or self.objective[i] == wanted)
        )

    def label(self, position: int) -> str:
        """A log label for one channel: ``objective/name``."""
        return f"{self.objectives[self.objective[position]]}/{self.names[position]}"


def timescale_spec(objective: str = "return") -> ChannelSpec:
    """The dense, sparse and final split of one objective."""
    return ChannelSpec(
        names=("dense", "sparse", "final"),
        granularity=(Granularity.DENSE, Granularity.SPARSE, Granularity.FINAL),
        objective=(0, 0, 0),
        objectives=(objective,),
    )


def crossed_spec(
    objectives: Sequence[str],
    granularities: Sequence[Granularity] = tuple(Granularity),
) -> ChannelSpec:
    """One channel for each pair of an objective and a granularity.

    The channel order is objective first, then granularity. A channel is named
    ``objective_granularity``, which keeps the names unique.
    """
    if not objectives:
        raise ValueError("a spec needs at least one objective")
    if not granularities:
        raise ValueError("a spec needs at least one granularity")
    names = tuple(
        f"{objective}_{granularity.value}"
        for objective in objectives
        for granularity in granularities
    )
    return ChannelSpec(
        names=names,
        granularity=tuple(granularities) * len(objectives),
        objective=tuple(index for index, _ in enumerate(objectives) for _ in granularities),
        objectives=tuple(objectives),
    )


def stack(channels: Sequence[jax.Array]) -> jax.Array:
    """Stack one reward array for each channel on the channel axis."""
    return jnp.stack(list(channels), axis=CHANNEL_AXIS)


def total(rewards: jax.Array) -> jax.Array:
    """The scalar reward: the sum over the channel axis."""
    return jnp.sum(rewards, axis=CHANNEL_AXIS)


def check_rewards(spec: ChannelSpec, rewards: jax.Array, num_agents: int) -> None:
    """Raise if `rewards` does not have the shape the contract requires.

    The check reads shapes only, so it also runs inside a jitted function.
    """
    expected = (spec.num_channels, num_agents)
    if rewards.shape != expected:
        raise ValueError(f"reward shape {rewards.shape} does not match the contract {expected}")


class StepOutput(NamedTuple):
    """What every environment returns from one step.

    Attributes:
        obs: Observations, with shape ``(num_agents, ...)``.
        state: The next environment state.
        reward: Channel rewards, with shape ``(num_channels, num_agents)``.
        done: Per-agent termination, with shape ``(num_agents,)``.
        episode_done: True when the episode ended and the environment reset.
        info: Extra per-step quantities, for the record and the analysis.
    """

    obs: jax.Array
    state: EnvState
    reward: jax.Array
    done: jax.Array
    episode_done: jax.Array
    info: dict[str, Any]


@runtime_checkable
class ChannelEnv(Protocol):
    """An environment that reports its reward as channels.

    The environment resets itself when the episode ends. The reward and the
    observation of that step belong to the episode that ended, not to the new
    one. ``episode_done`` marks the step where this happens.
    """

    @property
    def channel_spec(self) -> ChannelSpec:
        """The channels this environment pays, in channel-axis order."""
        ...

    @property
    def num_agents(self) -> int:
        """The size of the agent axis."""
        ...

    def reset(self, key: jax.Array) -> tuple[jax.Array, EnvState]:
        """Start an episode; return the observations and the state."""
        ...

    def step(self, key: jax.Array, state: EnvState, actions: jax.Array) -> StepOutput:
        """Apply `actions`, with shape ``(num_agents,)``."""
        ...
