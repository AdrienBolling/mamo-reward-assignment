"""Generalized advantage estimation, for scalar and per-channel rewards."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    last_value: jax.Array,
    *,
    gamma: float,
    lam: float,
) -> tuple[jax.Array, jax.Array]:
    """GAE advantages and value targets for a batch of trajectories.

    Shapes: `rewards`, `values`, `dones` are ``(T, B)``; `last_value` is ``(B,)``,
    the value of the state after the last step. ``dones[t]`` ends the episode at
    step ``t``, so nothing after ``t`` is credited to it. Returns
    ``(advantages, targets)`` with ``targets = advantages + values``.
    """

    def backward(
        carry: tuple[jax.Array, jax.Array], step: tuple[jax.Array, jax.Array, jax.Array]
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        next_advantage, next_value = carry
        reward, value, done = step
        not_done = 1.0 - done.astype(reward.dtype)
        delta = reward + gamma * next_value * not_done - value
        advantage = delta + gamma * lam * not_done * next_advantage
        return (advantage, value), advantage

    zero = jnp.zeros_like(last_value)
    _, advantages = jax.lax.scan(
        backward, (zero, last_value), (rewards, values, dones), reverse=True
    )
    return advantages, advantages + values


def channel_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    last_value: jax.Array,
    *,
    gamma: float,
    lam: float,
) -> tuple[jax.Array, jax.Array]:
    """:func:`gae` applied to every channel of a trailing channel axis.

    Shapes: `rewards`, `values` are ``(T, B, C)``, `dones` is ``(T, B)``,
    `last_value` is ``(B, C)``. Outputs are ``(T, B, C)``.
    """
    per_channel = jax.vmap(
        lambda r, v, lv: gae(r, v, dones, lv, gamma=gamma, lam=lam),
        in_axes=(-1, -1, -1),
        out_axes=-1,
    )
    return per_channel(rewards, values, last_value)
