"""Interface shared by every environment that reports reward channels."""

from __future__ import annotations

from typing import Any, Protocol

import jax

# Environment state: a flax struct dataclass whose type depends on the env.
type EnvState = Any


class ChannelEnv(Protocol):
    """JaxMARL-style environment whose `step` info carries `reward_channels`.

    `step` auto-resets on ``dones["__all__"]``; ``info["reward_channels"]`` is a
    :class:`~mamora.rewards.channels.RewardChannels` computed on the pre-reset
    transition, and ``info["episode_stats"]`` holds per-agent (num_agents,)
    indicators that summarize the episode so far.
    """

    agents: list[str]
    num_agents: int

    def action_space(self, agent: str) -> Any: ...

    def observation_space(self, agent: str) -> Any: ...

    def reset(self, key: jax.Array) -> tuple[dict[str, jax.Array], EnvState]: ...

    def step(
        self, key: jax.Array, state: EnvState, actions: dict[str, jax.Array]
    ) -> tuple[
        dict[str, jax.Array], EnvState, dict[str, jax.Array], dict[str, jax.Array], dict
    ]: ...

    def episode_stats(self, state: EnvState) -> dict[str, jax.Array]: ...
