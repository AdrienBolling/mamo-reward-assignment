"""Reward channel interface.

A reward channel is one stream of reward with its own temporal statistics:

- ``dense``: frequent, low-delay feedback (survival, resources).
- ``sparse``: infrequent event feedback (achievements, milestones).
- ``final``: end-of-episode feedback (task success).

Every channel environment returns ``info["reward_channels"]`` as a
:class:`RewardChannels` whose arrays have the per-agent shape of the scalar
reward. The channels sum to the scalar environment reward unless the module
that builds them documents a difference.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

CHANNEL_NAMES: tuple[str, ...] = ("dense", "sparse", "final")
NUM_CHANNELS = len(CHANNEL_NAMES)


class RewardChannels(NamedTuple):
    """Per-channel rewards; each field has the shape of the scalar reward."""

    dense: jax.Array
    sparse: jax.Array
    final: jax.Array

    def total(self) -> jax.Array:
        """Sum of the channels, the scalar reward the channels decompose."""
        return self.dense + self.sparse + self.final

    def stack(self) -> jax.Array:
        """Channels stacked on a leading axis, in :data:`CHANNEL_NAMES` order."""
        return jnp.stack([self.dense, self.sparse, self.final])

    @classmethod
    def zeros_like(cls, reward: jax.Array) -> RewardChannels:
        """All-zero channels with the shape and dtype of `reward`."""
        zeros = jnp.zeros_like(reward)
        return cls(dense=zeros, sparse=zeros, final=zeros)
