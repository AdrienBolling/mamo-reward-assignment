"""Metric operators: measurements of the credit geometry.

The input is a channel-gradient stack: a pytree whose every leaf carries the
channel axis first, with shape ``(num_channels, *param_shape)``. That is what
``jax.vmap`` over per-channel losses, or ``jax.jacrev`` of a vector loss,
returns. The operators never flatten the stack into one long vector. They
reduce leaf by leaf, so the working memory is one ``(N, N)`` matrix.

Every operator is a pure function, and every one runs under ``jax.jit``.

Conventions for degenerate input:

- A zero channel has norm 0, and cosine 0 with every channel, itself included.
  Not NaN. A final channel pays nothing early in training, so this is the
  common case, not the exception.
- The effective rank and the stable rank of an all-zero stack are 0.

The inner product is Euclidean. A learned metric, as in document 01 section 10,
replaces :func:`gram` and keeps everything downstream.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from mamora.contract import ChannelSpec

# A pytree whose every leaf has the channel axis first.
type ChannelGrads = Any


def num_channels(grads: ChannelGrads) -> int:
    """The size of the channel axis, checked on every leaf."""
    leaves = jax.tree.leaves(grads)
    if not leaves:
        raise ValueError("a channel-gradient stack needs at least one leaf")
    sizes = {leaf.shape[0] if leaf.ndim else None for leaf in leaves}
    if len(sizes) != 1 or None in sizes:
        raise ValueError(f"every leaf must carry the channel axis first; got leading sizes {sizes}")
    return leaves[0].shape[0]


def gram(grads: ChannelGrads) -> jax.Array:
    """The Gram matrix ``K[i, j] = <g_i, g_j>``, with shape ``(N, N)``.

    The sum runs leaf by leaf, in at least float32 and at full matmul
    precision. On a GPU, the default float32 matmul is TF32, which keeps about
    three decimal digits; that is too coarse for a cosine near zero.
    """
    n = num_channels(grads)
    total = jnp.zeros((n, n), dtype=jnp.float32)
    for leaf in jax.tree.leaves(grads):
        flat = leaf.reshape(n, -1)
        flat = flat.astype(jnp.promote_types(flat.dtype, jnp.float32))
        total = total + jnp.matmul(flat, flat.T, precision=jax.lax.Precision.HIGHEST)
    return total


def norms(gram_matrix: jax.Array) -> jax.Array:
    """The norm of each channel gradient, from the diagonal of the Gram matrix."""
    return jnp.sqrt(jnp.clip(jnp.diagonal(gram_matrix), min=0.0))


def cosines(gram_matrix: jax.Array) -> jax.Array:
    """Pairwise cosine similarities, in ``[-1, 1]``; 0 where a channel is zero."""
    scale = norms(gram_matrix)
    denominator = jnp.outer(scale, scale)
    defined = denominator > 0.0
    safe = jnp.where(defined, denominator, 1.0)
    return jnp.where(defined, jnp.clip(gram_matrix / safe, -1.0, 1.0), 0.0)


def eigenvalues(gram_matrix: jax.Array) -> jax.Array:
    """The eigenvalues of the Gram matrix, largest first.

    An eigenvalue below ``N * eps * lambda_max`` is set to 0, the numerical-rank
    rule of ``numpy.linalg.matrix_rank``. Without it, the square root in
    :func:`effective_rank` turns the float noise of a zero eigenvalue into a
    visible error.
    """
    spectrum = jnp.linalg.eigvalsh(gram_matrix)[::-1]
    tolerance = spectrum.shape[0] * jnp.finfo(spectrum.dtype).eps * spectrum[0]
    return jnp.where(spectrum > tolerance, spectrum, 0.0)


def effective_rank(gram_matrix: jax.Array) -> jax.Array:
    """The effective rank of the gradient matrix ``G`` (Roy and Vetterli, 2007).

    ``exp`` of the entropy of the normalized singular values of ``G``, which are
    the square roots of the eigenvalues of ``K = G^T G``. It is 1 when every
    channel points the same way, N when the channels are orthogonal with equal
    norm, and 0 when every channel is zero.
    """
    singular = jnp.sqrt(eigenvalues(gram_matrix))
    total = jnp.sum(singular)
    p = singular / jnp.where(total > 0.0, total, 1.0)
    entropy = -jnp.sum(jnp.where(p > 0.0, p * jnp.log(jnp.where(p > 0.0, p, 1.0)), 0.0))
    return jnp.where(total > 0.0, jnp.exp(entropy), 0.0)


def stable_rank(gram_matrix: jax.Array) -> jax.Array:
    """``trace(K) / lambda_max(K)``: the Frobenius norm over the spectral norm, squared.

    Same limits as :func:`effective_rank`, and cheaper to read: it says how many
    of the largest direction it takes to carry the total gradient energy.
    """
    spectrum = eigenvalues(gram_matrix)
    largest = spectrum[0]
    return jnp.where(largest > 0.0, jnp.sum(spectrum) / jnp.where(largest > 0.0, largest, 1.0), 0.0)


def _key_name(key: Any) -> str:
    """One path element of a pytree, as a short name."""
    if isinstance(key, jax.tree_util.DictKey):
        return str(key.key)
    if isinstance(key, jax.tree_util.SequenceKey):
        return str(key.idx)
    if isinstance(key, jax.tree_util.GetAttrKey):
        return key.name
    if isinstance(key, jax.tree_util.FlattenedIndexKey):
        return str(key.key)
    return str(key)


def gram_by_module(grads: ChannelGrads, depth: int = 1) -> dict[str, jax.Array]:
    """One Gram matrix for each module, where a module is a path prefix of `depth` keys.

    With flax parameters ``{"params": {"encoder": ..., "head": ...}}``, depth 2
    gives ``params/encoder`` and ``params/head``. The module Gram matrices sum
    to :func:`gram` of the whole stack.
    """
    if depth < 1:
        raise ValueError(f"depth must be at least 1; got {depth}")
    groups: dict[str, list[jax.Array]] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(grads)[0]:
        name = "/".join(_key_name(key) for key in path[:depth]) or "root"
        groups.setdefault(name, []).append(leaf)
    return {name: gram(leaves) for name, leaves in groups.items()}


def _summarize(prefix: str, spec: ChannelSpec, gram_matrix: jax.Array) -> dict[str, jax.Array]:
    """The scalars of one Gram matrix, keyed under `prefix`."""
    if gram_matrix.shape != (spec.num_channels, spec.num_channels):
        raise ValueError(
            f"Gram matrix {gram_matrix.shape} does not match the {spec.num_channels} channels of the spec"
        )
    out: dict[str, jax.Array] = {}
    scale = norms(gram_matrix)
    cosine = cosines(gram_matrix)
    spectrum = eigenvalues(gram_matrix)
    for i, name in enumerate(spec.names):
        out[f"{prefix}/norm/{name}"] = scale[i]
    for i, first in enumerate(spec.names):
        for j in range(i + 1, spec.num_channels):
            out[f"{prefix}/cosine/{first}|{spec.names[j]}"] = cosine[i, j]
    for k in range(spec.num_channels):
        out[f"{prefix}/eigenvalue/{k}"] = spectrum[k]
    out[f"{prefix}/effective_rank"] = effective_rank(gram_matrix)
    out[f"{prefix}/stable_rank"] = stable_rank(gram_matrix)
    return out


def summary(
    spec: ChannelSpec, grads: ChannelGrads, *, depth: int | None = None
) -> dict[str, jax.Array]:
    """Flat scalars for the run record, keyed by channel name.

    Keys: ``grad/norm/<name>``, ``grad/cosine/<a>|<b>`` for each pair,
    ``grad/eigenvalue/<k>``, ``grad/effective_rank`` and ``grad/stable_rank``.
    With `depth`, the same keys repeat under ``grad/<module>/`` for each module
    of :func:`gram_by_module`.
    """
    out = _summarize("grad", spec, gram(grads))
    if depth is not None:
        for module, module_gram in gram_by_module(grads, depth).items():
            out.update(_summarize(f"grad/{module}", spec, module_gram))
    return out
