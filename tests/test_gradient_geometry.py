import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mamora.diagnostics.gradients import (
    geometry_to_metrics,
    gradient_geometry,
    gram_matrix,
    per_module_geometry,
    split_by_module,
    tree_dot,
    tree_norm,
)

NAMES = ("dense", "sparse", "final")


def _random_tree(key, scale=1.0):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "encoder": {"kernel": scale * jax.random.normal(k1, (4, 3)), "bias": jnp.zeros((3,))},
        "head": {"kernel": scale * jax.random.normal(k2, (3, 2))},
        "gate": scale * jax.random.normal(k3, (2,)),
    }


def _flat(tree):
    return np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(tree)])


@pytest.fixture
def grads():
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    return [_random_tree(k, scale=s) for k, s in zip(keys, (1.0, 0.1, 0.01), strict=True)]


def test_tree_dot_and_norm_match_flat(grads):
    a, b = grads[0], grads[1]
    assert np.isclose(float(tree_dot(a, b)), _flat(a) @ _flat(b), rtol=1e-5)
    assert np.isclose(float(tree_norm(a)), np.linalg.norm(_flat(a)), rtol=1e-5)


def test_tree_dot_rejects_mismatched_trees(grads):
    with pytest.raises(ValueError, match="leaves"):
        tree_dot(grads[0], {"only": jnp.ones(3)})


def test_gram_matches_flat_and_is_symmetric(grads):
    gram = np.asarray(gram_matrix(grads))
    g = np.stack([_flat(t) for t in grads])
    assert np.allclose(gram, g @ g.T, rtol=1e-5)
    assert np.allclose(gram, gram.T)


def test_geometry_cosines_and_eigenvalues(grads):
    geo = gradient_geometry(grads)
    assert np.allclose(np.diag(np.asarray(geo.cosines)), 1.0, atol=1e-5)
    assert np.all(np.abs(np.asarray(geo.cosines)) <= 1.0 + 1e-5)
    assert np.isclose(float(geo.eigenvalues.sum()), float(jnp.trace(geo.gram)), rtol=1e-5)
    assert float(geo.norms[0]) > float(geo.norms[1]) > float(geo.norms[2])


def test_geometry_is_jittable(grads):
    geo = jax.jit(gradient_geometry)(grads)
    assert geo.gram.shape == (3, 3)


def test_split_by_module_groups_top_level_keys(grads):
    modules = split_by_module(grads[0])
    assert set(modules) == {"encoder", "head", "gate"}
    assert set(modules["encoder"]) == {"kernel", "bias"}
    assert modules["gate"].shape == (2,)
    deeper = split_by_module(grads[0], depth=2)
    assert "encoder/kernel" in deeper


def test_per_module_geometry_sums_to_global(grads):
    per_module = per_module_geometry(grads)
    total = sum(geo.gram for geo in per_module.values())
    assert np.allclose(np.asarray(total), np.asarray(gradient_geometry(grads).gram), rtol=1e-5)


def test_metrics_names(grads):
    metrics = geometry_to_metrics(gradient_geometry(grads), NAMES)
    assert "grad/norm/dense" in metrics
    assert "grad/cos/dense_final" in metrics
    assert "grad/norm_ratio/dense_sparse" in metrics
    assert "grad/gram_eig/2" in metrics
    assert "grad/gram_cond" in metrics
    assert float(metrics["grad/norm_ratio/dense_sparse"]) > 1.0
    with pytest.raises(ValueError, match="channels"):
        geometry_to_metrics(gradient_geometry(grads), NAMES[:2])
