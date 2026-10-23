"""Networks: a shared encoder, an actor head, and a critic with one value for each channel."""

from __future__ import annotations

import numpy as np
from flax import linen as nn
from flax.linen.initializers import constant, orthogonal
from jax import Array


class MLP(nn.Module):
    """Dense layers with tanh, the encoder every head shares."""

    hidden: tuple[int, ...]

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for width in self.hidden:
            x = nn.Dense(width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.tanh(x)
        return x


class ActorCritic(nn.Module):
    """Logits over actions, and one value for each reward channel.

    The parameter tree has three modules, ``encoder``, ``actor`` and
    ``critic``, so the gradient diagnostics can tell them apart.
    """

    num_actions: int
    num_channels: int
    hidden: tuple[int, ...] = (64, 64)

    @nn.compact
    def __call__(self, obs: Array) -> tuple[Array, Array]:
        """Return ``logits`` with shape ``(..., num_actions)`` and ``values`` with ``(..., num_channels)``."""
        h = MLP(self.hidden, name="encoder")(obs)
        logits = nn.Dense(
            self.num_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor"
        )(h)
        values = nn.Dense(
            self.num_channels, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic"
        )(h)
        return logits, values
