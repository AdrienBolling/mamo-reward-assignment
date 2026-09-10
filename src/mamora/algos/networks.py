"""Actor-critic network of the PPO baseline.

Modules (the top-level parameter keys, used by the per-module gradient geometry):

- ``encoder``: shared MLP trunk;
- ``actor``: policy logits;
- ``critic``: scalar value head, the one PPO uses;
- ``channel_critic``: auxiliary per-channel value head on stop-gradient
  features. It exists only to estimate per-channel advantages for the gradient
  forensics; it does not change the policy update or the shared representation.
"""

from __future__ import annotations

from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

ACTIVATIONS: dict[str, Callable[[jax.Array], jax.Array]] = {"tanh": nn.tanh, "relu": nn.relu}


class Encoder(nn.Module):
    hidden_size: int
    num_layers: int
    activation: str

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        act = ACTIVATIONS[self.activation]
        # One feature vector per observation, whatever its shape (pixels included).
        x = obs.reshape((obs.shape[0], -1))
        for _ in range(self.num_layers):
            x = act(nn.Dense(self.hidden_size, kernel_init=nn.initializers.orthogonal(2**0.5))(x))
        return x


class ActorCritic(nn.Module):
    """Returns ``(logits, value, channel_values)`` for a batch of observations."""

    num_actions: int
    num_channels: int
    hidden_size: int = 128
    num_layers: int = 2
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        features = Encoder(self.hidden_size, self.num_layers, self.activation, name="encoder")(obs)
        logits = nn.Dense(
            self.num_actions, kernel_init=nn.initializers.orthogonal(0.01), name="actor"
        )(features)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0), name="critic")(features)
        # Zero-initialised: a channel with no reward yet yields exactly zero
        # advantages instead of bootstrap noise from a random head.
        channel_values = nn.Dense(
            self.num_channels, kernel_init=nn.initializers.zeros, name="channel_critic"
        )(jax.lax.stop_gradient(features))
        return logits, jnp.squeeze(value, axis=-1), channel_values
