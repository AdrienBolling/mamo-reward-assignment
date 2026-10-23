"""Metric operators: the Gram matrix, its derived quantities, and the record keys."""

import jax
import jax.numpy as jnp
import pytest

from mamora.contract import timescale_spec
from mamora.operators.metrics import (
    cosines,
    effective_rank,
    eigenvalues,
    gram,
    gram_by_module,
    norms,
    num_channels,
    stable_rank,
    summary,
)


def _flatten(grads):
    """The explicit (N, P) gradient matrix the operators avoid building."""
    n = num_channels(grads)
    return jnp.concatenate([leaf.reshape(n, -1) for leaf in jax.tree.leaves(grads)], axis=1)


def _reference_gram(grads):
    """``G @ G.T`` at full precision, so the comparison does not inherit TF32 noise."""
    flat = _flatten(grads)
    return jnp.matmul(flat, flat.T, precision=jax.lax.Precision.HIGHEST)


@pytest.fixture
def grads():
    """Three channels over a two-module tree with leaves of different rank."""
    keys = jax.random.split(jax.random.key(0), 3)
    return {
        "params": {
            "encoder": {
                "kernel": jax.random.normal(keys[0], (3, 4, 5)),
                "bias": jax.random.normal(keys[1], (3, 5)),
            },
            "head": {"kernel": jax.random.normal(keys[2], (3, 5, 2))},
        }
    }


def _stack(*channels):
    """A one-leaf stack from explicit channel vectors."""
    return {"w": jnp.stack([jnp.asarray(c, dtype=jnp.float32) for c in channels])}


def test_gram_matches_the_explicit_product_and_runs_under_jit(grads):
    reference = _reference_gram(grads)
    assert jnp.allclose(gram(grads), reference, atol=1e-4)
    assert jnp.allclose(jax.jit(gram)(grads), reference, atol=1e-4)


def test_gram_accepts_any_pytree_and_promotes_to_float32():
    leaves = [jnp.ones((2, 3), dtype=jnp.bfloat16), jnp.ones((2, 1), dtype=jnp.bfloat16)]
    result = gram(leaves)
    assert result.dtype == jnp.float32
    assert jnp.allclose(result, jnp.full((2, 2), 4.0))


@pytest.mark.parametrize(
    ("grads", "message"),
    [
        ({}, "at least one leaf"),
        ({"a": jnp.ones((3, 2)), "b": jnp.ones((2, 2))}, "channel axis first"),
        ({"a": jnp.ones((3, 2)), "b": jnp.float32(1.0)}, "channel axis first"),
    ],
)
def test_gram_refuses_a_stack_without_a_shared_channel_axis(grads, message):
    with pytest.raises(ValueError, match=message):
        gram(grads)


def test_norms_are_the_row_norms_of_the_gradient_matrix(grads):
    reference = jnp.linalg.norm(_flatten(grads), axis=1)
    assert jnp.allclose(norms(gram(grads)), reference, atol=1e-4)


def test_cosines_report_alignment_and_give_zero_to_a_zero_channel():
    stack = _stack([1.0, 0.0], [2.0, 0.0], [0.0, 3.0], [-1.0, 0.0], [0.0, 0.0])
    cosine = cosines(gram(stack))
    assert jnp.allclose(cosine[0, 1], 1.0)  # parallel
    assert jnp.allclose(cosine[0, 2], 0.0)  # orthogonal
    assert jnp.allclose(cosine[0, 3], -1.0)  # opposed
    assert jnp.all(cosine[4] == 0.0)  # a zero channel, its diagonal included
    assert jnp.all(cosine[:, 4] == 0.0)
    assert not jnp.any(jnp.isnan(cosine))
    assert jnp.all(jnp.abs(cosine) <= 1.0)


def test_eigenvalues_are_sorted_non_negative_and_sum_to_the_trace(grads):
    spectrum = eigenvalues(gram(grads))
    assert jnp.all(spectrum[:-1] >= spectrum[1:])
    assert jnp.all(spectrum >= 0.0)
    assert jnp.allclose(jnp.sum(spectrum), jnp.trace(gram(grads)), rtol=1e-4)


@pytest.mark.parametrize(
    ("stack", "expected"),
    [
        (_stack([1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [-2.0, 0.0, 0.0]), 1.0),  # one direction
        (_stack([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), 3.0),  # orthogonal, equal
        (_stack([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), 0.0),  # nothing
    ],
)
def test_effective_rank_and_stable_rank_count_independent_directions(stack, expected):
    k = gram(stack)
    assert jnp.allclose(effective_rank(k), expected, atol=1e-5)
    assert jnp.allclose(stable_rank(k), expected, atol=1e-5)
    assert jnp.allclose(jax.jit(effective_rank)(k), expected, atol=1e-5)


def test_eigenvalues_below_the_numerical_rank_cutoff_are_zero():
    k = gram(_stack([1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [-2.0, 0.0, 0.0]))
    spectrum = eigenvalues(k)
    assert jnp.allclose(spectrum[0], 14.0, rtol=1e-5)
    assert jnp.all(spectrum[1:] == 0.0)


def test_module_gram_matrices_sum_to_the_whole(grads):
    by_module = gram_by_module(grads, depth=2)
    assert list(by_module) == ["params/encoder", "params/head"]
    assert jnp.allclose(sum(by_module.values()), gram(grads), atol=1e-4)
    assert list(gram_by_module(grads, depth=1)) == ["params"]
    assert list(gram_by_module(jax.tree.leaves(grads))) == ["0", "1", "2"]
    with pytest.raises(ValueError, match="at least 1"):
        gram_by_module(grads, depth=0)


def test_summary_keys_use_the_channel_names_of_the_spec(grads):
    spec = timescale_spec()
    record = summary(spec, grads)
    assert set(record) == {
        "grad/norm/dense",
        "grad/norm/sparse",
        "grad/norm/final",
        "grad/cosine/dense|sparse",
        "grad/cosine/dense|final",
        "grad/cosine/sparse|final",
        "grad/eigenvalue/0",
        "grad/eigenvalue/1",
        "grad/eigenvalue/2",
        "grad/effective_rank",
        "grad/stable_rank",
    }
    assert all(value.shape == () for value in record.values())


def test_summary_repeats_the_keys_for_each_module_and_runs_under_jit(grads):
    spec = timescale_spec()
    record = jax.jit(lambda g: summary(spec, g, depth=2))(grads)
    assert "grad/params/encoder/cosine/dense|final" in record
    assert "grad/params/head/effective_rank" in record
    assert jnp.allclose(record["grad/norm/dense"], norms(gram(grads))[0], atol=1e-4)


def test_summary_refuses_a_stack_that_does_not_match_the_spec():
    with pytest.raises(ValueError, match="does not match the 3 channels"):
        summary(timescale_spec(), _stack([1.0], [2.0]))
