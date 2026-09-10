import jax
import jax.numpy as jnp
import pytest

from mamora.envs.factory import make_channel_env
from mamora.rewards.channels import CHANNEL_NAMES, RewardChannels
from mamora.rewards.craftax import OUTCOMES, CraftaxChannelEnv, OutcomeReward, craftax_family

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
    for rewards, dones, channels in _random_steps(env, jax.random.PRNGKey(1), NUM_STEPS):
        scalar = jnp.stack([rewards[agent] for agent in env.agents])
        assert channels.total().shape == (env.num_agents,)
        assert jnp.allclose(channels.total(), scalar, atol=1e-6)
        if not bool(dones["__all__"]):
            assert bool((channels.final == 0).all())  # the final channel pays only at the end


def test_reward_is_upstream_plus_final(ma_env):
    key = jax.random.PRNGKey(5)
    _obs, state = ma_env.reset(key)
    actions = {agent: jnp.zeros((), jnp.int32) for agent in ma_env.agents}
    _obs, _s, upstream, _d, _i = ma_env.env.step_env(key, state, actions)
    _obs, _s, rewards, _d, info = ma_env.step(key, state, actions)
    final = info["reward_channels"].final
    for i, agent in enumerate(ma_env.agents):
        assert jnp.isclose(rewards[agent], upstream[agent] + final[i])


def test_outcome_rewards_are_ordered():
    with pytest.raises(ValueError, match="death < timeout < boss"):
        OutcomeReward(death=1.0, timeout=0.0, boss=10.0)
    assert OutcomeReward.from_config({"death": -2, "timeout": -1, "boss": 3}).boss == 3.0


@pytest.fixture(scope="module")
def short_env():
    return make_channel_env(
        "Craftax-MA-Symbolic",
        {"max_episode_steps": 50, "final_reward": {"death": -1.0, "timeout": 0.0, "boss": 10.0}},
    )


def _terminal_step(env, state, actions=None):
    key = jax.random.PRNGKey(0)
    actions = actions or {agent: jnp.zeros((), jnp.int32) for agent in env.agents}
    _obs, next_state, _rewards, dones, info = env.step(key, state, actions)
    stats = info["episode_stats"]
    outcome = {k: float(stats[k][0]) for k in OUTCOMES}
    return next_state, dones, info["reward_channels"], outcome


def test_death_outcome(short_env):
    assert isinstance(short_env, CraftaxChannelEnv)
    _obs, state = short_env.reset(jax.random.PRNGKey(0))
    dead = state.replace(
        player_health=jnp.zeros_like(state.player_health),
        player_alive=jnp.zeros_like(state.player_alive),
    )
    next_state, dones, channels, outcome = _terminal_step(short_env, dead)
    assert bool(dones["__all__"])
    assert int(next_state.timestep) == 0
    assert channels.final.tolist() == [short_env.final_reward.death] * short_env.num_agents
    assert outcome == {"death": 1.0, "timeout": 0.0, "boss": 0.0}


def test_timeout_outcome_at_wrapper_limit(short_env):
    _obs, state = short_env.reset(jax.random.PRNGKey(0))
    late = state.replace(
        timestep=jnp.asarray(short_env.max_episode_steps - 1, state.timestep.dtype)
    )
    next_state, dones, channels, outcome = _terminal_step(short_env, late)
    assert bool(dones["__all__"])
    assert int(next_state.timestep) == 0
    assert channels.final.tolist() == [short_env.final_reward.timeout] * short_env.num_agents
    assert outcome == {"death": 0.0, "timeout": 1.0, "boss": 0.0}


def test_boss_outcome(short_env):
    _obs, state = short_env.reset(jax.random.PRNGKey(0))
    levels = short_env.env.static_env_params.num_levels
    win = state.replace(boss_progress=jnp.asarray(levels - 1, state.boss_progress.dtype))
    _next_state, dones, channels, outcome = _terminal_step(short_env, win)
    assert bool(dones["__all__"])
    assert channels.final.tolist() == [short_env.final_reward.boss] * short_env.num_agents
    assert outcome == {"death": 0.0, "timeout": 0.0, "boss": 1.0}


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
    assert bool((channels.final == ma_env.final_reward.death).all())
    assert channels.total().shape == (ma_env.num_agents,)


def test_channels_vmap(ma_env):
    num_envs = 2
    keys = jax.random.split(jax.random.PRNGKey(4), num_envs)
    _obs, state = jax.vmap(ma_env.reset)(keys)
    actions = {agent: jnp.zeros((num_envs,), jnp.int32) for agent in ma_env.agents}
    _obs, _state, _rewards, _dones, info = jax.vmap(ma_env.step)(keys, state, actions)
    for channel in info["reward_channels"]:
        assert channel.shape == (num_envs, ma_env.num_agents)
