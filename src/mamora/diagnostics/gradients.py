"""Gradient geometry across credit channels (roadmap Phase 1, sections 3 and 21).

Given one gradient pytree per channel, compute norms, pairwise dot products,
cosine similarities, and the Gram matrix ``K = G^T G`` without materializing a
flattened gradient: every quantity is a sum of per-leaf inner products. The same
functions apply to a sub-tree, so per-module geometry is the same computation on
the sub-trees selected by :func:`split_by_module`.

All functions are pure and jit-compatible; :func:`gradient_geometry` is the
per-minibatch entry point, :func:`geometry_to_metrics` flattens it for logging.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

# A gradient pytree (nested dicts of arrays, as produced by flax/optax).
type PyTree = Any

EPS = 1e-8


def tree_dot(a: PyTree, b: PyTree) -> jax.Array:
    """Inner product of two pytrees with the same structure."""
    leaves_a = jax.tree.leaves(a)
    leaves_b = jax.tree.leaves(b)
    if len(leaves_a) != len(leaves_b):
        msg = f"pytrees have {len(leaves_a)} and {len(leaves_b)} leaves"
        raise ValueError(msg)
    partial_dots = [jnp.vdot(x, y) for x, y in zip(leaves_a, leaves_b, strict=True)]
    return jnp.sum(jnp.stack(partial_dots)) if partial_dots else jnp.asarray(0.0)


def tree_norm(a: PyTree) -> jax.Array:
    """Euclidean norm of a pytree."""
    return jnp.sqrt(tree_dot(a, a))


def gram_matrix(grads: Sequence[PyTree]) -> jax.Array:
    """Gram matrix ``K[i, j] = <g_i, g_j>`` of `k` gradient pytrees, shape (k, k).

    Only the ``k (k + 1) / 2`` unique inner products are computed.
    """
    k = len(grads)
    rows = []
    cache: dict[tuple[int, int], jax.Array] = {}
    for i in range(k):
        row = []
        for j in range(k):
            key = (min(i, j), max(i, j))
            if key not in cache:
                cache[key] = tree_dot(grads[key[0]], grads[key[1]])
            row.append(cache[key])
        rows.append(jnp.stack(row))
    return jnp.stack(rows)


def cosine_matrix(gram: jax.Array, eps: float = EPS) -> jax.Array:
    """Pairwise cosine similarities from a Gram matrix."""
    norms = jnp.sqrt(jnp.clip(jnp.diag(gram), 0.0))
    return gram / (jnp.outer(norms, norms) + eps)


class GradientGeometry(NamedTuple):
    """Geometry of `k` channel gradients on one parameter (sub-)tree."""

    norms: jax.Array  # (k,)
    gram: jax.Array  # (k, k)
    cosines: jax.Array  # (k, k)
    eigenvalues: jax.Array  # (k,), ascending


def gradient_geometry(grads: Sequence[PyTree], eps: float = EPS) -> GradientGeometry:
    """Norms, Gram matrix, cosines and Gram eigenvalues of the channel gradients."""
    gram = gram_matrix(grads)
    return GradientGeometry(
        norms=jnp.sqrt(jnp.clip(jnp.diag(gram), 0.0)),
        gram=gram,
        cosines=cosine_matrix(gram, eps),
        eigenvalues=jnp.linalg.eigvalsh(gram),
    )


def split_by_module(tree: PyTree, depth: int = 1) -> dict[str, PyTree]:
    """Split a nested-dict pytree into sub-trees keyed by their first `depth` keys.

    Keys are joined with ``/``. Leaves closer to the root than `depth` are keyed
    by their own path. Use it to compute per-module gradient geometry.
    """
    modules: dict[str, PyTree] = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        parts = [_key_name(entry) for entry in path]
        module = "/".join(parts[:depth])
        rest = parts[depth:]
        node = modules.setdefault(module, {})
        if not rest:
            modules[module] = leaf
            continue
        for part in rest[:-1]:
            node = node.setdefault(part, {})
        node[rest[-1]] = leaf
    return modules


def per_module_geometry(
    grads: Sequence[PyTree], depth: int = 1, eps: float = EPS
) -> dict[str, GradientGeometry]:
    """:func:`gradient_geometry` on every module sub-tree (see :func:`split_by_module`)."""
    per_grad = [split_by_module(g, depth) for g in grads]
    modules = per_grad[0].keys()
    return {m: gradient_geometry([g[m] for g in per_grad], eps) for m in modules}


def geometry_to_metrics(
    geometry: GradientGeometry, names: Sequence[str], prefix: str = "grad"
) -> dict[str, jax.Array]:
    """Flatten a geometry into scalar metrics named after the channels.

    Keys: ``{prefix}/norm/{a}``, ``{prefix}/cos/{a}_{b}`` for ``a < b``,
    ``{prefix}/norm_ratio/{a}_{b}`` (norm of `a` over norm of `b`),
    ``{prefix}/gram_eig/{i}`` (ascending) and ``{prefix}/gram_cond``.
    """
    k = len(names)
    if geometry.norms.shape != (k,):
        msg = f"geometry has {geometry.norms.shape[0]} channels, {k} names given"
        raise ValueError(msg)
    metrics: dict[str, jax.Array] = {}
    for i, a in enumerate(names):
        metrics[f"{prefix}/norm/{a}"] = geometry.norms[i]
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                metrics[f"{prefix}/cos/{a}_{b}"] = geometry.cosines[i, j]
                metrics[f"{prefix}/norm_ratio/{a}_{b}"] = geometry.norms[i] / (
                    geometry.norms[j] + EPS
                )
    for i in range(k):
        metrics[f"{prefix}/gram_eig/{i}"] = geometry.eigenvalues[i]
    metrics[f"{prefix}/gram_cond"] = geometry.eigenvalues[-1] / (
        jnp.clip(geometry.eigenvalues[0], 0.0) + EPS
    )
    return metrics


def _key_name(entry: Any) -> str:
    """Readable name of a pytree path entry (dict key, sequence index, attribute)."""
    for attr in ("key", "idx", "name"):
        if hasattr(entry, attr):
            return str(getattr(entry, attr))
    return str(entry)
