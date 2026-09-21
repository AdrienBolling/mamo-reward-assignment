"""PPO that keeps the channel identity of its gradient.

The rollout carries one advantage and one value target for each channel. The
loss is a vector: the policy surrogate of each channel, then the value loss of
each critic head. Its Jacobian gives two channel-gradient stacks, one for the
policy losses and one for the value losses. The update stack is
``policy + vf_coef * value``, the exact split of the update gradient by the
channel that produced it. An aggregation operator turns it into the update.
The metrics measure the policy stack by default, as document 03 asks.

The surrogate of every channel uses the clipping mask of the aggregate
objective, ``sum_k w_k A_k``. This is what makes the split exact: the weighted
sum of the update stack, plus the entropy gradient, is the gradient of standard
PPO with the aggregate advantage. A per-channel clip would instead let a
channel push where the aggregate objective is clipped.

Advantage normalization keeps the split linear too. Each channel is centered
on its own mean, and every channel is divided by one scale, the standard
deviation of the aggregate advantage. Per-channel scaling would erase the
size difference between channels that hypothesis H1 measures.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from mamora.agents.networks import ActorCritic
from mamora.operators.aggregate import Aggregator, ChannelGrads, weighted_sum

# A hook that measures a channel-gradient stack, for the run record.
type Measure = Callable[[ChannelGrads], Any]


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Hyperparameters. A sequence gives one value for each channel; a float, the same for all."""

    learning_rate: float = 3e-4
    gamma: float | Sequence[float] = 0.99
    lam: float | Sequence[float] = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatches: int = 4
    channel_weights: Sequence[float] | None = None
    normalize_advantage: bool = True

    def per_channel(self, value: float | Sequence[float], num_channels: int) -> jax.Array:
        """A ``(num_channels,)`` array from a float or from a sequence of that length."""
        if isinstance(value, Sequence):
            if len(value) != num_channels:
                raise ValueError(f"expected {num_channels} values, got {value}")
            return jnp.asarray(tuple(value), dtype=jnp.float32)
        return jnp.full((num_channels,), value, dtype=jnp.float32)

    def weights(self, num_channels: int) -> jax.Array:
        """The channel weights of the aggregate objective; ones when unset."""
        return self.per_channel(
            1.0 if self.channel_weights is None else self.channel_weights, num_channels
        )


class Batch(NamedTuple):
    """A rollout, or a minibatch of one. ``*batch`` is any number of leading dims after time."""

    obs: jax.Array
    """``(T, *batch, obs_size)``"""

    action: jax.Array
    """``(T, *batch)``"""

    log_prob: jax.Array
    """``(T, *batch)``: the log-probability of `action` under the behaviour policy."""

    advantage: jax.Array
    """``(N, T, *batch)``"""

    target: jax.Array
    """``(N, T, *batch)``: the value target of each channel."""


def flatten(batch: Batch) -> Batch:
    """Merge time and batch into one leading axis ``M``; the channel axis stays first."""
    m = batch.action.size
    return Batch(
        obs=batch.obs.reshape(m, -1),
        action=batch.action.reshape(m),
        log_prob=batch.log_prob.reshape(m),
        advantage=batch.advantage.reshape(batch.advantage.shape[0], m),
        target=batch.target.reshape(batch.target.shape[0], m),
    )


def normalize_advantages(advantage: jax.Array, weights: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Center each channel, and divide every channel by the scale of the aggregate."""
    centered = advantage - advantage.mean(axis=1, keepdims=True)
    aggregate = jnp.tensordot(weights, centered, axes=1)
    return centered / (aggregate.std() + eps)


def _log_probs(logits: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The log-probability of `action`, and the entropy, for each row of `logits`."""
    log_p = jax.nn.log_softmax(logits)
    chosen = jnp.take_along_axis(log_p, action[:, None], axis=1)[:, 0]
    entropy = -jnp.sum(jnp.exp(log_p) * log_p, axis=1)
    return chosen, entropy


def channel_losses(
    params: Any, apply_fn: Callable[..., Any], batch: Batch, cfg: PPOConfig, weights: jax.Array
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """The policy loss of each channel, then the value loss of each: shape ``(2N,)``."""
    logits, values = apply_fn(params, batch.obs)
    log_prob, _ = _log_probs(logits, batch.action)
    ratio = jnp.exp(log_prob - batch.log_prob)

    aggregate = jnp.tensordot(weights, batch.advantage, axes=1)
    unclipped = ratio * aggregate
    clipped = jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * aggregate
    mask = jax.lax.stop_gradient((unclipped <= clipped).astype(ratio.dtype))

    policy = -jnp.mean(mask * ratio * batch.advantage, axis=1)
    value = 0.5 * jnp.mean(jnp.square(values.T - batch.target), axis=1)
    aux = {
        "policy_loss": policy,
        "value_loss": value,
        "clip_fraction": 1.0 - jnp.mean(mask),
        "approx_kl": jnp.mean(batch.log_prob - log_prob),
    }
    return jnp.concatenate([policy, value]), aux


def entropy_loss(
    params: Any, apply_fn: Callable[..., Any], obs: jax.Array, ent_coef: float
) -> jax.Array:
    """``-ent_coef`` times the mean entropy: the one term without a channel."""
    logits, _ = apply_fn(params, obs)
    _, entropy = _log_probs(logits, jnp.zeros(obs.shape[0], dtype=jnp.int32))
    return -ent_coef * jnp.mean(entropy)


def gradient_stacks(
    params: Any, apply_fn: Callable[..., Any], batch: Batch, cfg: PPOConfig, weights: jax.Array
) -> tuple[ChannelGrads, ChannelGrads, dict[str, jax.Array]]:
    """The policy stack and the value stack: two pytrees whose leaves are ``(N, *param_shape)``.

    One Jacobian of :func:`channel_losses`, split in two. The update stack is
    ``policy + vf_coef * value``; see :func:`update_stack`.
    """
    jacobian, aux = jax.jacrev(channel_losses, has_aux=True)(params, apply_fn, batch, cfg, weights)
    num_channels = batch.advantage.shape[0]
    policy = jax.tree.map(lambda leaf: leaf[:num_channels], jacobian)
    value = jax.tree.map(lambda leaf: leaf[num_channels:], jacobian)
    return policy, value, aux


def update_stack(policy: ChannelGrads, value: ChannelGrads, vf_coef: float) -> ChannelGrads:
    """``policy + vf_coef * value``: the split of the update gradient by channel."""
    return jax.tree.map(lambda p, v: p + vf_coef * v, policy, value)


def create_train_state(
    model: ActorCritic, obs_size: int, cfg: PPOConfig, key: jax.Array
) -> TrainState:
    """Initialize the parameters and the optimizer: global-norm clipping, then Adam."""
    params = model.init(key, jnp.zeros((1, obs_size)))
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.learning_rate))
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def _take(batch: Batch, index: jax.Array) -> Batch:
    """The rows `index` of a flat batch; the channel axis stays first."""
    return Batch(
        obs=batch.obs[index],
        action=batch.action[index],
        log_prob=batch.log_prob[index],
        advantage=batch.advantage[:, index],
        target=batch.target[:, index],
    )


def update(
    train_state: TrainState,
    batch: Batch,
    key: jax.Array,
    cfg: PPOConfig,
    *,
    aggregate: Aggregator = weighted_sum,
    measure: Measure | None = None,
    measured: Literal["policy", "update"] = "policy",
) -> tuple[TrainState, dict[str, Any]]:
    """Run `cfg.epochs` passes of `cfg.minibatches` minibatches over the rollout.

    Every minibatch builds the update stack, aggregates it, adds the entropy
    gradient, and applies the result. `measure` runs on the policy stack, or
    on the update stack when `measured` says so; its outputs come back under
    ``info["measure"]``, stacked with the losses over the
    ``epochs * minibatches`` updates.
    """
    num_channels = batch.advantage.shape[0]
    weights = cfg.weights(num_channels)
    flat = flatten(batch)
    if cfg.normalize_advantage:
        flat = flat._replace(advantage=normalize_advantages(flat.advantage, weights))
    size = flat.action.shape[0]
    if size % cfg.minibatches:
        raise ValueError(f"{size} samples do not split into {cfg.minibatches} minibatches")

    def minibatch_step(state: TrainState, minibatch: Batch) -> tuple[TrainState, dict[str, Any]]:
        policy, value, aux = gradient_stacks(state.params, state.apply_fn, minibatch, cfg, weights)
        stack = update_stack(policy, value, cfg.vf_coef)
        entropy_grad = jax.grad(entropy_loss)(
            state.params, state.apply_fn, minibatch.obs, cfg.ent_coef
        )
        grads = jax.tree.map(jnp.add, aggregate(stack, weights), entropy_grad)
        info = dict(aux)
        if measure is not None:
            info["measure"] = measure(policy if measured == "policy" else stack)
        return state.apply_gradients(grads=grads), info

    def epoch_step(
        carry: tuple[TrainState, jax.Array], _: None
    ) -> tuple[tuple[TrainState, jax.Array], dict[str, Any]]:
        state, key = carry
        key, subkey = jax.random.split(key)
        order = jax.random.permutation(subkey, size).reshape(cfg.minibatches, -1)
        minibatches = jax.vmap(_take, in_axes=(None, 0))(flat, order)
        state, info = jax.lax.scan(minibatch_step, state, minibatches)
        return (state, key), info

    (train_state, _), info = jax.lax.scan(epoch_step, (train_state, key), None, length=cfg.epochs)
    return train_state, jax.tree.map(lambda x: x.reshape((-1, *x.shape[2:])), info)
