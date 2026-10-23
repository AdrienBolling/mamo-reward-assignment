"""Aggregation operators: from a stack of channel gradients to one update direction.

An aggregation operator takes a channel-gradient stack, a pytree whose every
leaf has the channel axis first, and the channel weights, and returns one
gradient pytree with the shape of the parameters. The weighted sum reproduces
the scalarized baseline. Every other operator, such as PCGrad or a learned
interaction matrix, is a drop-in replacement with the same signature.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

# A pytree whose every leaf has the channel axis first.
type ChannelGrads = Any
# (stack, weights) -> gradient with the shape of the parameters.
type Aggregator = Callable[[ChannelGrads, jax.Array], Any]


def weighted_sum(stack: ChannelGrads, weights: jax.Array) -> Any:
    """``sum_k weights[k] * g_k``: the scalarized update."""
    return jax.tree.map(lambda leaf: jnp.tensordot(weights, leaf, axes=1), stack)
