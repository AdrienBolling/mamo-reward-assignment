import jax
import jax.numpy as jnp
import pytest

from mamora.envs.factory import make_channel_env
from mamora.rewards.channels import CHANNEL_NAMES, RewardChannels
from mamora.rewards.craftax import craftax_family

NUM_STEPS = 12


def _random_steps(env, key, steps):
    """Yield (rewards, dones, channels) for `steps` random-action transitions."""
    _obs, state = env.reset(key)
    num_actions = env.action_space(env.agents[0]).n
    for _ in range(steps):
        key, k_act, k_step = jax.random.split(key, 3)
        keys = jax.random.split(k_act, env.num_agents)
        actions = {
            agent: jax.random.randint(k, (), 0, num_actions)
            for agent, k in zip(env.agents, keys, strict=True)
        }
        _obs, state, rewards, dones, info = env.step(k_step, state, actions)
        yield rewards, dones, info["reward_channels"]


@pytest.fixture(scope="module")
def ma_env():
    return make_channel_env("Craftax-MA-Symbolic")


@pytest.fixture(scope="module")
def coop_env():
    return make_channel_env("Craftax-Coop-Symbolic")


def test_family_from_name():
    assert craftax_family("Craftax-MA-Symbolic") == "ma"
    assert craftax_family("Craftax-Coop-Pixels") == "coop"


def test_reward_channels_helpers():
    channels = RewardChannels(
        dense=jnp.array([1.0, 2.0]), sparse=jnp.array([3.0, 0.0]), final=jnp.array([0.0, 5.0])
    )
    assert channels.total().tolist() == [4.0, 7.0]
    assert channels.stack().shape == (len(CHANNEL_NAMES), 2)
    assert RewardChannels.zeros_like(channels.dense).total().tolist() == [0.0, 0.0]


@pytest.mark.parametrize("env_fixture", ["ma_env", "coop_env"])
def test_channels_sum_to_reward(env_fixture, request):
    env = request.getfixturevalue(env_fixture)
    for rewards, _dones, channels in _random_steps(env, jax.random.PRNGKey(1), NUM_STEPS):
        scalar = jnp.stack([rewards[agent] for agent in env.agents])
        assert channels.total().shape == (env.num_agents,)
        assert jnp.allclose(channels.total(), scalar, atol=1e-6)
        assert bool((channels.final == 0).all())  # MA-Craftax has no terminal reward


def test_shared_reward_gives_every_agent_the_same_channels(coop_env):
    assert coop_env.shared_reward
    for _rewards, _dones, channels in _random_steps(coop_env, jax.random.PRNGKey(2), NUM_STEPS):
        for channel in channels:
            assert bool((channel == channel[0]).all())


def test_channels_use_pre_reset_transition(ma_env):
    key = jax.random.PRNGKey(3)
    _obs, state = ma_env.reset(key)
    # Kill every player: the step terminates the episode and auto-resets.
    dead = state.replace(
        player_health=jnp.zeros_like(state.player_health),
        player_alive=jnp.zeros_like(state.player_alive),
        timestep=jnp.asarray(500, dtype=state.timestep.dtype),
    )
    actions = {agent: jnp.zeros((), jnp.int32) for agent in ma_env.agents}
    _obs, next_state, _rewards, dones, info = ma_env.step(key, dead, actions)
    assert bool(dones["__all__"])
    assert int(next_state.timestep) == 0  # reset state
    channels = info["reward_channels"]
    assert bool((channels.dense == 0).all())  # dead players earn no health reward
    assert channels.total().shape == (ma_env.num_agents,)


def test_channels_vmap(ma_env):
    num_envs = 2
    keys = jax.random.split(jax.random.PRNGKey(4), num_envs)
    _obs, state = jax.vmap(ma_env.reset)(keys)
    actions = {agent: jnp.zeros((num_envs,), jnp.int32) for agent in ma_env.agents}
    _obs, _state, _rewards, _dones, info = jax.vmap(ma_env.step)(keys, state, actions)
    for channel in info["reward_channels"]:
        assert channel.shape == (num_envs, ma_env.num_agents)
