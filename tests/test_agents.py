"""The agent: network layout, channel-wise GAE, and the exact split of the PPO gradient."""

from typing import cast

import jax
import jax.numpy as jnp
import pytest

from mamora.agents.gae import channel_gae, gae
from mamora.agents.networks import ActorCritic
from mamora.agents.ppo import (
    Batch,
    PPOConfig,
    channel_losses,
    create_train_state,
    entropy_loss,
    flatten,
    gradient_stacks,
    normalize_advantages,
    update,
    update_stack,
)
from mamora.operators.aggregate import weighted_sum

OBS_SIZE = 4
NUM_ACTIONS = 3
NUM_CHANNELS = 3


def test_actor_critic_returns_logits_and_one_value_per_channel():
    model = ActorCritic(num_actions=NUM_ACTIONS, num_channels=NUM_CHANNELS)
    params = model.init(jax.random.key(0), jnp.zeros((5, OBS_SIZE)))
    logits, values = cast(
        tuple[jax.Array, jax.Array], model.apply(params, jnp.zeros((5, OBS_SIZE)))
    )
    assert logits.shape == (5, NUM_ACTIONS)
    assert values.shape == (5, NUM_CHANNELS)
    assert set(params["params"]) == {"encoder", "actor", "critic"}


def _reference_gae(rewards, values, next_values, terminals, dones, gamma, lam):
    """The recursion, written as a loop."""
    advantages = [0.0] * len(rewards)
    carry = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - terminals[t]) - values[t]
        carry = delta + gamma * lam * (1 - dones[t]) * carry
        advantages[t] = carry
    return advantages


REWARDS = [1.0, 0.0, 2.0, 0.5]
VALUES = [0.5, 0.4, 0.3, 0.2]
NEXT_VALUES = [0.4, 0.3, 0.2, 0.1]


@pytest.mark.parametrize(
    ("terminals", "dones"),
    [
        ([0, 0, 0, 0], [0, 0, 0, 0]),  # one long episode
        ([0, 0, 0, 0], [0, 1, 0, 0]),  # truncation at t=1: bootstrap, then cut the trace
        ([0, 1, 0, 0], [0, 1, 0, 0]),  # termination at t=1: no bootstrap, cut the trace
    ],
)
def test_gae_matches_the_recursion_and_treats_truncation_apart(terminals, dones):
    gamma, lam = 0.9, 0.8
    args = (
        jnp.array(REWARDS),
        jnp.array(VALUES),
        jnp.array(NEXT_VALUES),
        jnp.array(terminals, dtype=bool),
        jnp.array(dones, dtype=bool),
    )
    advantages, targets = gae(*args, jnp.float32(gamma), jnp.float32(lam))
    expected = _reference_gae(REWARDS, VALUES, NEXT_VALUES, terminals, dones, gamma, lam)
    assert jnp.allclose(advantages, jnp.array(expected), atol=1e-6)
    assert jnp.allclose(targets, advantages + jnp.array(VALUES))
    if dones[1]:
        bootstrap = 0.0 if terminals[1] else gamma * NEXT_VALUES[1]
        assert jnp.isclose(advantages[1], REWARDS[1] + bootstrap - VALUES[1])


def test_channel_gae_gives_each_channel_its_own_horizon():
    rewards = jnp.stack([jnp.array(REWARDS), jnp.array(REWARDS)])[:, :, None]
    values = jnp.stack([jnp.array(VALUES), jnp.array(VALUES)])[:, :, None]
    next_values = jnp.stack([jnp.array(NEXT_VALUES), jnp.array(NEXT_VALUES)])[:, :, None]
    flags = jnp.zeros((4, 1), dtype=bool)
    gamma = jnp.array([0.5, 0.99])
    lam = jnp.array([0.9, 0.95])
    advantages, _ = channel_gae(rewards, values, next_values, flags, flags, gamma, lam)
    assert advantages.shape == (2, 4, 1)
    for k in range(2):
        single, _ = gae(rewards[k], values[k], next_values[k], flags, flags, gamma[k], lam[k])
        assert jnp.allclose(advantages[k], single)
    assert not jnp.allclose(advantages[0], advantages[1])


@pytest.fixture
def model():
    return ActorCritic(num_actions=NUM_ACTIONS, num_channels=NUM_CHANNELS, hidden=(8,))


@pytest.fixture
def params(model):
    return model.init(jax.random.key(0), jnp.zeros((1, OBS_SIZE)))


@pytest.fixture
def batch(model, params):
    """A rollout of 4 steps over 2 envs, with ratios that leave the clip range."""
    keys = jax.random.split(jax.random.key(1), 5)
    obs = jax.random.normal(keys[0], (4, 2, OBS_SIZE))
    action = jax.random.randint(keys[1], (4, 2), 0, NUM_ACTIONS)
    logits, _ = model.apply(params, obs)
    log_prob = jnp.take_along_axis(jax.nn.log_softmax(logits), action[..., None], axis=-1)[..., 0]
    shift = jax.random.uniform(keys[2], (4, 2), minval=-0.5, maxval=0.5)
    return Batch(
        obs=obs,
        action=action,
        log_prob=log_prob - shift,
        advantage=jax.random.normal(keys[3], (NUM_CHANNELS, 4, 2)),
        target=jax.random.normal(keys[4], (NUM_CHANNELS, 4, 2)),
    )


def test_flatten_merges_time_and_batch_and_keeps_the_channel_axis_first(batch):
    flat = flatten(batch)
    assert flat.obs.shape == (8, OBS_SIZE)
    assert flat.action.shape == (8,)
    assert flat.advantage.shape == (NUM_CHANNELS, 8)
    assert jnp.allclose(flat.advantage[1, 3], batch.advantage[1, 1, 1])


def test_normalization_centers_each_channel_and_scales_the_aggregate(batch):
    weights = jnp.array([1.0, 2.0, 0.5])
    normalized = normalize_advantages(flatten(batch).advantage, weights)
    assert jnp.allclose(normalized.mean(axis=1), 0.0, atol=1e-6)
    assert jnp.isclose(jnp.tensordot(weights, normalized, axes=1).std(), 1.0, atol=1e-4)


def _standard_ppo_loss(params, apply_fn, flat, cfg, weights):
    """PPO as usually written, with the aggregate advantage and a value loss per head."""
    logits, values = apply_fn(params, flat.obs)
    log_p = jax.nn.log_softmax(logits)
    log_prob = jnp.take_along_axis(log_p, flat.action[:, None], axis=1)[:, 0]
    ratio = jnp.exp(log_prob - flat.log_prob)
    aggregate = jnp.tensordot(weights, flat.advantage, axes=1)
    clipped = jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
    policy = -jnp.mean(jnp.minimum(ratio * aggregate, clipped * aggregate))
    value = 0.5 * jnp.mean(jnp.square(values.T - flat.target), axis=1)
    entropy = jnp.mean(-jnp.sum(jnp.exp(log_p) * log_p, axis=1))
    return policy + cfg.vf_coef * jnp.sum(weights * value) - cfg.ent_coef * entropy


def test_the_weighted_update_stack_plus_entropy_is_the_standard_ppo_gradient(model, params, batch):
    cfg = PPOConfig(channel_weights=(1.0, 0.5, 2.0))
    weights = cfg.weights(NUM_CHANNELS)
    flat = flatten(batch)
    flat = flat._replace(advantage=normalize_advantages(flat.advantage, weights))

    # An algebraic identity: keep TF32 out of it on a GPU.
    with jax.default_matmul_precision("highest"):
        policy, value, aux = gradient_stacks(params, model.apply, flat, cfg, weights)
        stack = update_stack(policy, value, cfg.vf_coef)
        entropy_grad = jax.grad(entropy_loss)(params, model.apply, flat.obs, cfg.ent_coef)
        ours = jax.tree.map(jnp.add, weighted_sum(stack, weights), entropy_grad)
        reference = jax.grad(_standard_ppo_loss)(params, model.apply, flat, cfg, weights)

    assert 0.0 < float(aux["clip_fraction"]) < 1.0, "the batch must exercise the clip"
    for mine, theirs in zip(jax.tree.leaves(ours), jax.tree.leaves(reference), strict=True):
        assert jnp.allclose(mine, theirs, atol=1e-6)


def test_each_stack_has_one_gradient_per_channel(model, params, batch):
    cfg = PPOConfig()
    policy, value, _ = gradient_stacks(
        params, model.apply, flatten(batch), cfg, cfg.weights(NUM_CHANNELS)
    )
    for stack in (policy, value):
        for leaf, param in zip(jax.tree.leaves(stack), jax.tree.leaves(params), strict=True):
            assert leaf.shape == (NUM_CHANNELS, *param.shape)


def test_a_clipped_sample_gives_no_policy_gradient_to_any_channel(model, params, batch):
    cfg = PPOConfig(normalize_advantage=False)
    flat = flatten(batch)
    logits, _ = model.apply(params, flat.obs)
    log_prob = jnp.take_along_axis(jax.nn.log_softmax(logits), flat.action[:, None], axis=1)[:, 0]
    # ratio = e for every sample, above 1 + clip_eps, with a positive advantage everywhere
    flat = flat._replace(log_prob=log_prob - 1.0, advantage=jnp.abs(flat.advantage) + 0.1)
    policy, _, aux = gradient_stacks(params, model.apply, flat, cfg, cfg.weights(NUM_CHANNELS))
    assert float(aux["clip_fraction"]) == 1.0
    assert all(jnp.allclose(leaf, 0.0) for leaf in jax.tree.leaves(policy))


def test_the_policy_stack_never_touches_the_critic_and_the_value_stack_never_the_actor(
    model, params, batch
):
    cfg = PPOConfig()
    policy, value, _ = gradient_stacks(
        params, model.apply, flatten(batch), cfg, cfg.weights(NUM_CHANNELS)
    )
    assert all(jnp.allclose(leaf, 0.0) for leaf in jax.tree.leaves(policy["params"]["critic"]))
    assert not all(jnp.allclose(leaf, 0.0) for leaf in jax.tree.leaves(policy["params"]["actor"]))
    assert all(jnp.allclose(leaf, 0.0) for leaf in jax.tree.leaves(value["params"]["actor"]))
    assert not all(jnp.allclose(leaf, 0.0) for leaf in jax.tree.leaves(value["params"]["critic"]))


def test_channel_losses_report_the_policy_losses_then_the_value_losses(model, params, batch):
    cfg = PPOConfig()
    losses, aux = channel_losses(
        params, model.apply, flatten(batch), cfg, cfg.weights(NUM_CHANNELS)
    )
    assert losses.shape == (2 * NUM_CHANNELS,)
    assert jnp.allclose(losses[:NUM_CHANNELS], aux["policy_loss"])
    assert jnp.allclose(losses[NUM_CHANNELS:], aux["value_loss"])


def test_update_runs_under_jit_and_stacks_the_measurements(model, batch):
    cfg = PPOConfig(epochs=2, minibatches=2)
    state = create_train_state(model, OBS_SIZE, cfg, jax.random.key(0))

    def measure(stack):
        return jnp.sum(jnp.stack([jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(stack)]))

    step = jax.jit(lambda s, b, k: update(s, b, k, cfg, measure=measure, measured="update"))
    new_state, info = step(state, batch, jax.random.key(1))

    assert int(new_state.step) == 4
    assert info["policy_loss"].shape == (4, NUM_CHANNELS)
    assert info["measure"].shape == (4,)
    assert jnp.all(info["measure"] > 0.0)
    changed = [
        not jnp.allclose(a, b)
        for a, b in zip(
            jax.tree.leaves(state.params), jax.tree.leaves(new_state.params), strict=True
        )
    ]
    assert any(changed)


def test_update_refuses_a_rollout_that_does_not_split_into_minibatches(model, batch):
    cfg = PPOConfig(minibatches=3)
    state = create_train_state(model, OBS_SIZE, cfg, jax.random.key(0))
    with pytest.raises(ValueError, match="do not split into 3"):
        update(state, batch, jax.random.key(1), cfg)


def test_per_channel_settings_need_one_value_per_channel():
    cfg = PPOConfig(gamma=(0.9, 0.99))
    assert jnp.allclose(cfg.per_channel(cfg.gamma, 2), jnp.array([0.9, 0.99]))
    assert jnp.allclose(cfg.per_channel([0.9, 0.99], 2), jnp.array([0.9, 0.99]))  # a config list
    assert jnp.allclose(cfg.per_channel(0.5, 3), jnp.full((3,), 0.5))
    with pytest.raises(ValueError, match="expected 3 values"):
        cfg.per_channel(cfg.gamma, 3)
