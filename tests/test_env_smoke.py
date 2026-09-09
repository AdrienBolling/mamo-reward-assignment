import jax
import jax.numpy as jnp
import pytest
from craftax_ma.craftax_state import EnvState  # top-level namespace on purpose, see envs/factory.py

from mamora.envs.factory import ENV_NAMES, make_channel_env, make_env
from mamora.rollout import random_rollout


@pytest.fixture(scope="module")
def ma_env():
    # One instance per module: the env's jit caches are keyed on the env object.
    return make_env("Craftax-MA-Symbolic")


def test_unknown_env_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown MA-Craftax environment"):
        make_env("Craftax-Nope")
    with pytest.raises(ValueError, match="Unknown environment"):
        make_channel_env("Craftax-Nope")


def test_env_names_are_exposed():
    assert "Craftax-MA-Symbolic" in ENV_NAMES


def test_reset_and_step_shapes(ma_env):
    key = jax.random.PRNGKey(0)
    obs, state = ma_env.reset(key)
    assert isinstance(state, EnvState)  # guards the craftax.craftax_ma vs craftax_ma hazard
    assert set(obs) == set(ma_env.agents)
    actions = {agent: jnp.zeros((), jnp.int32) for agent in ma_env.agents}
    obs, state, rewards, dones, info = ma_env.step(key, state, actions)
    assert set(rewards) == set(ma_env.agents)
    assert set(dones) == set(ma_env.agents) | {"__all__"}
    for agent in ma_env.agents:
        assert obs[agent].shape == ma_env.observation_space(agent).shape
        assert rewards[agent].shape == ()
    assert "user_info" in info


def test_random_rollout_runs():
    env = make_channel_env("Craftax-MA-Symbolic")
    stats = random_rollout(env, jax.random.PRNGKey(0), num_envs=2, steps=3)
    assert stats.returns.shape == (env.num_agents,)
    assert stats.peak_stats["achievements"].shape == (env.num_agents,)
    assert stats.channel_returns.stack().shape == (3, env.num_agents)
    assert jnp.allclose(stats.channel_returns.total(), stats.returns, atol=1e-5)
