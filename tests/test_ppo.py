import math
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from mamora.algos.networks import ActorCritic
from mamora.algos.ppo import PPO, Batch, PPOConfig, RunConfig
from mamora.diagnostics.gradients import tree_norm
from mamora.envs.corridor import CorridorParams, TimescaleCorridor
from mamora.envs.factory import make_channel_env
from mamora.paths import CONF_DIR
from mamora.rewards.channels import NUM_CHANNELS
from mamora.train import train


def _params_and_batch(ppo, key, n=8):
    params = ppo.network.init(key, jnp.zeros((1, *ppo.obs_shape), jnp.float32))
    k_obs, k_act, k_adv = jax.random.split(key, 3)
    obs = jax.random.uniform(k_obs, (n, *ppo.obs_shape))
    action = jax.random.randint(k_act, (n,), 0, ppo.num_actions)
    adv = jax.random.normal(k_adv, (n, NUM_CHANNELS))
    batch = Batch(
        obs=obs,
        action=action,
        log_prob=jnp.full((n,), -math.log(ppo.num_actions)),
        advantage=adv.sum(-1),
        target=adv.sum(-1),
        channel_advantage=adv,
        channel_target=adv,
    )
    return params, batch


@pytest.fixture
def corridor_ppo():
    env = TimescaleCorridor(CorridorParams(invest_steps=3, commit_steps=2, horizon=16))
    return PPO(env, PPOConfig(hidden_size=16), RunConfig(num_envs=4, num_steps=8, num_iterations=2))


@pytest.mark.parametrize("obs_shape", [(4,), (3, 3, 2)])
def test_network_outputs_and_modules(obs_shape):
    """Vector and pixel-like observations both give one action distribution per row."""
    net = ActorCritic(num_actions=5, num_channels=NUM_CHANNELS, hidden_size=8)
    obs = jnp.zeros((2, *obs_shape))
    params = net.init(jax.random.PRNGKey(0), obs)
    logits, value, channel_values = cast(
        tuple[jax.Array, jax.Array, jax.Array], net.apply(params, obs)
    )
    assert logits.shape == (2, 5)
    assert value.shape == (2,)
    assert channel_values.shape == (2, NUM_CHANNELS)
    assert set(params["params"]) == {"encoder", "actor", "critic", "channel_critic"}


def test_gradient_routing(corridor_ppo):
    """Roadmap section 22: the channel critic never touches the encoder or the actor."""
    params, batch = _params_and_batch(corridor_ppo, jax.random.PRNGKey(1))

    def channel_critic_loss(p):
        _logits, _value, channel_values = corridor_ppo.network.apply(p, batch.obs)
        return jnp.square(channel_values - batch.channel_target).mean()

    grads = jax.grad(channel_critic_loss)(params)["params"]
    assert float(tree_norm(grads["channel_critic"])) > 0.0
    assert float(tree_norm(grads["encoder"])) == 0.0
    assert float(tree_norm(grads["actor"])) == 0.0

    policy_grads = jax.grad(corridor_ppo._policy_loss)(params, batch, batch.advantage)["params"]
    assert float(tree_norm(policy_grads["actor"])) > 0.0
    assert float(tree_norm(policy_grads["encoder"])) > 0.0
    assert float(tree_norm(policy_grads["critic"])) == 0.0
    assert float(tree_norm(policy_grads["channel_critic"])) == 0.0


def test_channel_critic_never_changes_the_ppo_update():
    """Codex P1: a huge auxiliary loss must leave encoder, actor and critic updates identical."""
    env = TimescaleCorridor(CorridorParams(invest_steps=3, commit_steps=2, horizon=16))
    run = RunConfig(num_envs=4, num_steps=8, num_iterations=2)
    updated = {}
    for coef in (0.0, 1e6):
        ppo = PPO(env, PPOConfig(hidden_size=16, channel_vf_coef=coef), run)
        params, batch = _params_and_batch(ppo, jax.random.PRNGKey(3))
        batch = batch._replace(channel_target=batch.channel_target + 1e3)  # large aux error
        opt_state = ppo.optimizer.init(params)
        _loss, grads = jax.value_and_grad(ppo._loss, has_aux=True)(params, batch)
        updates, _ = ppo.optimizer.update(grads, opt_state, params)
        updated[coef] = jax.tree.map(np.asarray, cast(dict[str, Any], updates)["params"])
    for module in ("encoder", "actor", "critic"):
        for leaf_a, leaf_b in zip(
            jax.tree.leaves(updated[0.0][module]),
            jax.tree.leaves(updated[1e6][module]),
            strict=True,
        ):
            assert np.array_equal(leaf_a, leaf_b)
    assert float(tree_norm(updated[1e6]["channel_critic"])) > 0.0


def test_forensics_minibatch_is_sampled_over_time(corridor_ppo):
    """Codex P1: the forensics batch must not be the first timesteps of every env."""
    ppo = corridor_ppo
    seen: dict[str, jax.Array] = {}
    original = ppo._forensics

    def spy(params, batch):
        seen["obs"] = batch.obs
        return original(params, batch)

    ppo._forensics = spy
    runner = ppo.init(jax.random.PRNGKey(0))
    ppo._iteration(runner, jnp.asarray(0))
    # obs[..., 3] is timestep / horizon: the 8 rollout steps give 8 distinct values.
    steps = np.unique(np.round(np.asarray(seen["obs"][:, 3]) * 16))
    assert len(steps) > 2


def test_forensics_metrics_are_finite(corridor_ppo):
    params, batch = _params_and_batch(corridor_ppo, jax.random.PRNGKey(2))
    metrics = corridor_ppo._forensics(params, batch)
    assert "grad/cos/dense_final" in metrics
    assert "grad/encoder/cos/dense_final" in metrics
    assert "grad/actor/norm/total" in metrics
    assert "adv/std/final" in metrics
    assert all(np.isfinite(float(v)) for v in metrics.values())


def test_iteration_runs_and_skips_forensics(corridor_ppo):
    ppo = corridor_ppo
    ppo.run = RunConfig(num_envs=4, num_steps=8, num_iterations=2, forensics_every=2)
    ppo.iteration = jax.jit(ppo._iteration)
    runner = ppo.init(jax.random.PRNGKey(0))
    runner, metrics_0 = ppo.iteration(runner, 0)
    runner, metrics_1 = ppo.iteration(runner, 1)
    assert np.isfinite(float(metrics_0["grad/norm/dense"]))
    assert np.isnan(float(metrics_1["grad/norm/dense"]))
    assert np.isfinite(float(metrics_1["loss/policy"]))
    assert runner.obs.shape == (4, 4)


def test_ppo_solves_the_corridor(tmp_path):
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "env.params.invest_steps=3",
                "env.params.commit_steps=2",
                "env.params.horizon=16",
                "total_timesteps=6e4",
                "num_envs=32",
                "num_steps=16",
                "algo.hidden_size=32",
                "log_every=1000",
            ],
        )
    final = train(cfg, tmp_path)
    assert final["episode/success"] > 0.9
    assert (tmp_path / "metrics.jsonl").exists()


def test_iteration_on_craftax_ma():
    env = make_channel_env("Craftax-MA-Symbolic")
    ppo = PPO(env, PPOConfig(hidden_size=16), RunConfig(num_envs=2, num_steps=2, num_iterations=1))
    runner = ppo.init(jax.random.PRNGKey(0))
    runner, metrics = ppo.iteration(runner, 0)
    assert runner.obs.shape == (2 * env.num_agents, *ppo.obs_shape)
    assert np.isfinite(float(metrics["loss/policy"]))
    assert np.isfinite(float(metrics["grad/cos/dense_sparse"]))
