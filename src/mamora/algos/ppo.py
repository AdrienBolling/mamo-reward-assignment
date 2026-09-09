"""PPO baseline with credit-channel instrumentation (roadmap Phase 1).

The policy update is standard clipped PPO on the scalar reward with one shared
network for all agents (IPPO with parameter sharing). Instrumentation, which
does not change that update:

- an auxiliary per-channel critic (on stop-gradient features) estimates
  per-channel advantages with :func:`~mamora.algos.gae.channel_gae`;
- on instrumented iterations, the policy-loss gradient is recomputed once per
  channel (and once with the scalar advantage) on the first minibatch, and the
  gradient geometry of ``[dense, sparse, final, total]`` is logged, globally and
  per module. These gradients are diagnostics; the optimizer never sees them.

Advantages used for the forensics are the raw (unnormalized) GAE advantages so
that the channel gradients are comparable; the actual update normalizes the
scalar advantage per minibatch, as usual. A channel that never pays reward
(``final`` on MA-Craftax) keeps a zero critic and zero advantages, so its
gradient norm stays at zero and its cosines are reported as zero; read
``reward/rate/<channel>`` before interpreting a channel's geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import optax

from mamora.algos.gae import channel_gae, gae
from mamora.algos.networks import ActorCritic
from mamora.diagnostics.gradients import (
    geometry_to_metrics,
    gradient_geometry,
    per_module_geometry,
)
from mamora.envs.base import ChannelEnv, EnvState
from mamora.rewards.channels import CHANNEL_NAMES, NUM_CHANNELS

# Flax parameter / optax state pytrees.
type PyTree = Any

FORENSICS_NAMES: tuple[str, ...] = (*CHANNEL_NAMES, "total")
EPS = 1e-8


@dataclass(frozen=True)
class PPOConfig:
    """Algorithm hyperparameters; every field is a key of `conf/algo/ppo.yaml`."""

    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    channel_vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4
    hidden_size: int = 128
    num_layers: int = 2
    activation: str = "tanh"


@dataclass(frozen=True)
class RunConfig:
    """Rollout and schedule settings; keys of `conf/train.yaml`."""

    num_envs: int = 64
    num_steps: int = 16
    num_iterations: int = 100
    forensics_every: int = 1


class Transition(NamedTuple):
    """One rollout step for the `B = num_envs * num_agents` agent-trajectories."""

    obs: jax.Array  # (B, obs_dim)
    action: jax.Array  # (B,)
    log_prob: jax.Array  # (B,)
    value: jax.Array  # (B,)
    channel_values: jax.Array  # (B, C)
    reward: jax.Array  # (B,)
    channel_rewards: jax.Array  # (B, C)
    done: jax.Array  # (B,)
    episode_stats: dict[str, jax.Array]  # each (B,), pre-reset
    episode_return: jax.Array  # (B,) return of the episode ending here (valid if done)
    episode_channel_return: jax.Array  # (B, C)
    episode_length: jax.Array  # (B,)


class RunnerState(NamedTuple):
    params: PyTree
    opt_state: PyTree
    env_state: EnvState
    obs: jax.Array  # (B, obs_dim)
    key: jax.Array
    episode_return: jax.Array  # (B,) running return of the current episode
    episode_channel_return: jax.Array  # (B, C)
    episode_length: jax.Array  # (B,)


class Batch(NamedTuple):
    """Flattened minibatch of the PPO update."""

    obs: jax.Array
    action: jax.Array
    log_prob: jax.Array
    advantage: jax.Array
    target: jax.Array
    channel_advantage: jax.Array  # (N, C)
    channel_target: jax.Array  # (N, C)


class PPO:
    """PPO trainer bound to one environment and one configuration."""

    def __init__(self, env: ChannelEnv, cfg: PPOConfig, run: RunConfig) -> None:
        self.env = env
        self.cfg = cfg
        self.run = run
        self.agents = tuple(env.agents)
        self.num_agents = len(self.agents)
        self.batch_size = run.num_envs * self.num_agents
        self.num_actions = int(env.action_space(self.agents[0]).n)
        self.obs_shape = tuple(env.observation_space(self.agents[0]).shape)
        self.network = ActorCritic(
            num_actions=self.num_actions,
            num_channels=NUM_CHANNELS,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            activation=cfg.activation,
        )
        total_updates = run.num_iterations * cfg.update_epochs * cfg.num_minibatches
        schedule = optax.linear_schedule(cfg.lr, 0.0, total_updates) if cfg.anneal_lr else cfg.lr
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(schedule, eps=1e-5)
        )
        self.iteration = jax.jit(self._iteration)

    # ------------------------------------------------------------------ setup
    def init(self, key: jax.Array) -> RunnerState:
        key, k_params, k_reset = jax.random.split(key, 3)
        params = self.network.init(k_params, jnp.zeros((1, *self.obs_shape), jnp.float32))
        obs, env_state = jax.vmap(self.env.reset)(jax.random.split(k_reset, self.run.num_envs))
        zeros = jnp.zeros((self.batch_size,), jnp.float32)
        return RunnerState(
            params=params,
            opt_state=self.optimizer.init(params),
            env_state=env_state,
            obs=self._stack_agents(obs),
            key=key,
            episode_return=zeros,
            episode_channel_return=jnp.zeros((self.batch_size, NUM_CHANNELS), jnp.float32),
            episode_length=zeros,
        )

    def _forward(self, params: PyTree, obs: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """(logits, value, channel_values) of the network on a batch of observations."""
        return cast(tuple[jax.Array, jax.Array, jax.Array], self.network.apply(params, obs))

    def _stack_agents(self, per_agent: dict[str, jax.Array]) -> jax.Array:
        """(num_envs, ...) per agent -> (B, ...) with agents as the fast axis."""
        stacked = jnp.stack([per_agent[a] for a in self.agents], axis=1)
        return stacked.reshape(self.batch_size, *stacked.shape[2:]).astype(jnp.float32)

    def _split_agents(self, flat: jax.Array) -> dict[str, jax.Array]:
        """(B,) -> {agent: (num_envs,)}."""
        per_env = flat.reshape(self.run.num_envs, self.num_agents)
        return {a: per_env[:, i] for i, a in enumerate(self.agents)}

    # ---------------------------------------------------------------- rollout
    def _env_step(self, runner: RunnerState, _: None) -> tuple[RunnerState, Transition]:
        key, k_action, k_step = jax.random.split(runner.key, 3)
        logits, value, channel_values = self._forward(runner.params, runner.obs)
        action = jax.random.categorical(k_action, logits)
        log_prob = jnp.take_along_axis(jax.nn.log_softmax(logits), action[:, None], axis=1)[:, 0]

        step_keys = jax.random.split(k_step, self.run.num_envs)
        obs, env_state, rewards, dones, info = jax.vmap(self.env.step)(
            step_keys, runner.env_state, self._split_agents(action)
        )
        reward = self._stack_agents(rewards)
        channel_rewards = jnp.stack(
            [self._stack_agents(c) for c in _per_agent_channels(info, self.agents)], axis=-1
        )
        done = jnp.repeat(dones["__all__"], self.num_agents).astype(jnp.float32)
        episode_stats = {
            k: self._stack_agents(v)
            for k, v in _per_agent(info["episode_stats"], self.agents).items()
        }

        episode_return = runner.episode_return + reward
        episode_channel_return = runner.episode_channel_return + channel_rewards
        episode_length = runner.episode_length + 1.0
        transition = Transition(
            obs=runner.obs,
            action=action,
            log_prob=log_prob,
            value=value,
            channel_values=channel_values,
            reward=reward,
            channel_rewards=channel_rewards,
            done=done,
            episode_stats=episode_stats,
            episode_return=episode_return,
            episode_channel_return=episode_channel_return,
            episode_length=episode_length,
        )
        keep = 1.0 - done
        next_runner = RunnerState(
            params=runner.params,
            opt_state=runner.opt_state,
            env_state=env_state,
            obs=self._stack_agents(obs),
            key=key,
            episode_return=episode_return * keep,
            episode_channel_return=episode_channel_return * keep[:, None],
            episode_length=episode_length * keep,
        )
        return next_runner, transition

    # ----------------------------------------------------------------- losses
    def _policy_loss(self, params: PyTree, batch: Batch, advantage: jax.Array) -> jax.Array:
        """Clipped PPO surrogate with the given (already scaled) advantages."""
        logits, _value, _channel_values = self._forward(params, batch.obs)
        log_prob = jnp.take_along_axis(jax.nn.log_softmax(logits), batch.action[:, None], axis=1)[
            :, 0
        ]
        ratio = jnp.exp(log_prob - batch.log_prob)
        clipped = jnp.clip(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps)
        return -jnp.minimum(ratio * advantage, clipped * advantage).mean()

    def _loss(self, params: PyTree, batch: Batch) -> tuple[jax.Array, dict[str, jax.Array]]:
        cfg = self.cfg
        logits, value, channel_values = self._forward(params, batch.obs)
        log_softmax = jax.nn.log_softmax(logits)
        log_prob = jnp.take_along_axis(log_softmax, batch.action[:, None], axis=1)[:, 0]
        ratio = jnp.exp(log_prob - batch.log_prob)
        advantage = (batch.advantage - batch.advantage.mean()) / (batch.advantage.std() + EPS)
        clipped = jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
        policy_loss = -jnp.minimum(ratio * advantage, clipped * advantage).mean()
        value_loss = 0.5 * jnp.square(value - batch.target).mean()
        channel_value_loss = 0.5 * jnp.square(channel_values - batch.channel_target).mean()
        entropy = -(jnp.exp(log_softmax) * log_softmax).sum(axis=-1).mean()
        loss = (
            policy_loss
            + cfg.vf_coef * value_loss
            + cfg.channel_vf_coef * channel_value_loss
            - cfg.ent_coef * entropy
        )
        aux = {
            "loss/policy": policy_loss,
            "loss/value": value_loss,
            "loss/channel_value": channel_value_loss,
            "loss/entropy": entropy,
            "loss/approx_kl": ((ratio - 1.0) - jnp.log(ratio)).mean(),
            "loss/clip_fraction": (jnp.abs(ratio - 1.0) > cfg.clip_eps).mean(),
        }
        return loss, aux

    def _forensics(self, params: PyTree, batch: Batch) -> dict[str, jax.Array]:
        """Gradient geometry of the per-channel and total policy gradients."""
        grad_fn = jax.grad(self._policy_loss)
        grads = [grad_fn(params, batch, batch.channel_advantage[:, k]) for k in range(NUM_CHANNELS)]
        grads.append(grad_fn(params, batch, batch.advantage))
        metrics = geometry_to_metrics(gradient_geometry(grads), FORENSICS_NAMES, prefix="grad")
        # Flax wraps the variables in a "params" collection; split below it.
        modules = per_module_geometry([g["params"] for g in grads])
        for module, geometry in modules.items():
            metrics.update(geometry_to_metrics(geometry, FORENSICS_NAMES, prefix=f"grad/{module}"))
        for k, name in enumerate(CHANNEL_NAMES):
            metrics[f"adv/mean/{name}"] = batch.channel_advantage[:, k].mean()
            metrics[f"adv/std/{name}"] = batch.channel_advantage[:, k].std()
        metrics["adv/mean/total"] = batch.advantage.mean()
        metrics["adv/std/total"] = batch.advantage.std()
        return metrics

    # ---------------------------------------------------------------- update
    def _iteration(
        self, runner: RunnerState, iteration: jax.Array
    ) -> tuple[RunnerState, dict[str, jax.Array]]:
        cfg, run = self.cfg, self.run
        runner, traj = jax.lax.scan(self._env_step, runner, None, length=run.num_steps)

        _logits, last_value, last_channel_values = self._forward(runner.params, runner.obs)
        advantage, target = gae(
            traj.reward, traj.value, traj.done, last_value, gamma=cfg.gamma, lam=cfg.gae_lambda
        )
        channel_advantage, channel_target = channel_gae(
            traj.channel_rewards,
            traj.channel_values,
            traj.done,
            last_channel_values,
            gamma=cfg.gamma,
            lam=cfg.gae_lambda,
        )
        flat = Batch(
            obs=traj.obs,
            action=traj.action,
            log_prob=traj.log_prob,
            advantage=advantage,
            target=target,
            channel_advantage=channel_advantage,
            channel_target=channel_target,
        )
        n = run.num_steps * self.batch_size
        flat = jax.tree.map(lambda x: x.reshape(n, *x.shape[2:]), flat)

        def epoch(
            carry: tuple[PyTree, PyTree, jax.Array], _: None
        ) -> tuple[tuple[PyTree, PyTree, jax.Array], dict[str, jax.Array]]:
            params, opt_state, key = carry
            key, k_perm = jax.random.split(key)
            perm = jax.random.permutation(k_perm, n)
            minibatches = jax.tree.map(
                lambda x: x[perm].reshape(cfg.num_minibatches, -1, *x.shape[1:]), flat
            )

            def minibatch(
                carry: tuple[PyTree, PyTree], batch: Batch
            ) -> tuple[tuple[PyTree, PyTree], dict[str, jax.Array]]:
                params, opt_state = carry
                (_loss, aux), grads = jax.value_and_grad(self._loss, has_aux=True)(params, batch)
                updates, opt_state = self.optimizer.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), aux

            (params, opt_state), aux = jax.lax.scan(minibatch, (params, opt_state), minibatches)
            return (params, opt_state, key), jax.tree.map(lambda x: x.mean(), aux)

        first_minibatch = jax.tree.map(lambda x: x[: n // cfg.num_minibatches], flat)
        nan_forensics = jax.tree.map(
            lambda s: jnp.full(s.shape, jnp.nan, s.dtype),
            jax.eval_shape(self._forensics, runner.params, first_minibatch),
        )
        forensics = jax.lax.cond(
            iteration % run.forensics_every == 0,
            lambda: self._forensics(runner.params, first_minibatch),
            lambda: nan_forensics,
        )

        (params, opt_state, key), aux = jax.lax.scan(
            epoch, (runner.params, runner.opt_state, runner.key), None, length=cfg.update_epochs
        )
        metrics = {
            **jax.tree.map(lambda x: x.mean(), aux),
            **_episode_metrics(traj),
            **_critic_metrics(traj, target, channel_target),
            **forensics,
        }
        runner = runner._replace(params=params, opt_state=opt_state, key=key)
        return runner, metrics


def _per_agent(
    tree: dict[str, jax.Array], agents: tuple[str, ...]
) -> dict[str, dict[str, jax.Array]]:
    """{stat: (num_envs, num_agents)} -> {stat: {agent: (num_envs,)}}."""
    return {k: {a: v[:, i] for i, a in enumerate(agents)} for k, v in tree.items()}


def _per_agent_channels(info: dict, agents: tuple[str, ...]) -> list[dict[str, jax.Array]]:
    """info["reward_channels"] (each (num_envs, num_agents)) -> one {agent: (num_envs,)} per channel."""
    channels = info["reward_channels"]
    return [{a: c[:, i] for i, a in enumerate(agents)} for c in channels]


def _masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    count = mask.sum()
    return jnp.where(count > 0, (values * mask).sum() / jnp.maximum(count, 1.0), jnp.nan)


def _episode_metrics(traj: Transition) -> dict[str, jax.Array]:
    """Averages over the episodes that ended during the rollout (NaN if none)."""
    done = traj.done
    metrics = {
        "episode/count": done.sum(),
        "episode/return": _masked_mean(traj.episode_return, done),
        "episode/length": _masked_mean(traj.episode_length, done),
    }
    for k, name in enumerate(CHANNEL_NAMES):
        metrics[f"episode/return_{name}"] = _masked_mean(traj.episode_channel_return[..., k], done)
    for name, values in traj.episode_stats.items():
        metrics[f"episode/{name}"] = _masked_mean(values, done)
    return metrics


def _explained_variance(prediction: jax.Array, target: jax.Array) -> jax.Array:
    return 1.0 - jnp.var(target - prediction) / (jnp.var(target) + EPS)


def _critic_metrics(
    traj: Transition, target: jax.Array, channel_target: jax.Array
) -> dict[str, jax.Array]:
    metrics = {"critic/explained_variance/total": _explained_variance(traj.value, target)}
    for k, name in enumerate(CHANNEL_NAMES):
        metrics[f"critic/explained_variance/{name}"] = _explained_variance(
            traj.channel_values[..., k], channel_target[..., k]
        )
        metrics[f"reward/rate/{name}"] = (traj.channel_rewards[..., k] != 0).mean()
    return metrics
