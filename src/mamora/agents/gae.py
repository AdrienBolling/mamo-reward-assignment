"""Generalized advantage estimation, one trace for each channel.

Time is axis 0. A transition ``t`` goes from ``obs[t]`` to ``final_obs[t]`` and
pays ``rewards[t]``. Two flags describe its end:

- ``terminals[t]``: the episode ended and no value follows, so the target does
  not bootstrap.
- ``episode_dones[t]``: the environment reset after this transition, so the
  trace stops here. A truncation sets this and not ``terminals``, and then the
  target bootstraps from ``next_values[t]``, the value of ``final_obs[t]``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def gae(
    rewards: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    terminals: jax.Array,
    episode_dones: jax.Array,
    gamma: jax.Array,
    lam: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Advantages and value targets for one channel; every array is ``(T, *batch)``."""
    continues = 1.0 - terminals.astype(values.dtype)
    traces = 1.0 - episode_dones.astype(values.dtype)

    def backward(carry: jax.Array, step: tuple[jax.Array, ...]) -> tuple[jax.Array, jax.Array]:
        reward, value, next_value, cont, trace = step
        delta = reward + gamma * next_value * cont - value
        advantage = delta + gamma * lam * trace * carry
        return advantage, advantage

    _, advantages = jax.lax.scan(
        backward,
        jnp.zeros_like(values[0]),
        (rewards, values, next_values, continues, traces),
        reverse=True,
    )
    return advantages, advantages + values


def channel_gae(
    rewards: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    terminals: jax.Array,
    episode_dones: jax.Array,
    gamma: jax.Array,
    lam: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """:func:`gae` over the channel axis; the reward and value arrays are ``(N, T, *batch)``.

    `gamma` and `lam` have shape ``(N,)``, one horizon for each channel. The
    flags are shared, with shape ``(T, *batch)``.
    """
    return jax.vmap(gae, in_axes=(0, 0, 0, None, None, 0, 0))(
        rewards, values, next_values, terminals, episode_dones, gamma, lam
    )
