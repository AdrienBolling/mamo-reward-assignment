import jax.numpy as jnp
import numpy as np

from mamora.algos.gae import channel_gae, gae

GAMMA, LAM = 0.9, 0.8


def _naive_gae(rewards, values, dones, last_value):
    T = len(rewards)
    adv = np.zeros(T)
    next_adv, next_value = 0.0, last_value
    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * next_value * not_done - values[t]
        adv[t] = delta + GAMMA * LAM * not_done * next_adv
        next_adv, next_value = adv[t], values[t]
    return adv


def test_gae_matches_naive_recursion():
    rewards = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
    values = np.array([[0.5, 0.1], [0.2, 0.3], [0.4, 0.4], [0.1, 0.2]])
    dones = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    last_value = np.array([0.3, 0.7])
    adv, target = gae(
        jnp.asarray(rewards),
        jnp.asarray(values),
        jnp.asarray(dones),
        jnp.asarray(last_value),
        gamma=GAMMA,
        lam=LAM,
    )
    for b in range(2):
        expected = _naive_gae(rewards[:, b], values[:, b], dones[:, b], last_value[b])
        assert np.allclose(np.asarray(adv[:, b]), expected, atol=1e-6)
    assert np.allclose(np.asarray(target), np.asarray(adv) + values, atol=1e-6)


def test_channel_gae_is_linear_in_the_channels():
    rng = np.random.default_rng(0)
    rewards = rng.normal(size=(6, 3, 3)).astype(np.float32)
    values = rng.normal(size=(6, 3, 3)).astype(np.float32)
    dones = (rng.random((6, 3)) < 0.3).astype(np.float32)
    last_value = rng.normal(size=(3, 3)).astype(np.float32)
    adv_c, target_c = channel_gae(
        jnp.asarray(rewards),
        jnp.asarray(values),
        jnp.asarray(dones),
        jnp.asarray(last_value),
        gamma=GAMMA,
        lam=LAM,
    )
    adv_sum, target_sum = gae(
        jnp.asarray(rewards.sum(-1)),
        jnp.asarray(values.sum(-1)),
        jnp.asarray(dones),
        jnp.asarray(last_value.sum(-1)),
        gamma=GAMMA,
        lam=LAM,
    )
    assert adv_c.shape == (6, 3, 3)
    assert np.allclose(np.asarray(adv_c.sum(-1)), np.asarray(adv_sum), atol=1e-5)
    assert np.allclose(np.asarray(target_c.sum(-1)), np.asarray(target_sum), atol=1e-5)
