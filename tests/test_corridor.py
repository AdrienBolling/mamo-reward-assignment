import jax
import jax.numpy as jnp
import pytest

from mamora.envs.corridor import AGENT, COMMIT, HARVEST, INVEST, CorridorParams, TimescaleCorridor
from mamora.envs.factory import make_channel_env
from mamora.rollout import random_rollout

PARAMS = CorridorParams(invest_steps=3, commit_steps=2, horizon=10)


@pytest.fixture
def env():
    return TimescaleCorridor(PARAMS)


def _play(env, actions):
    """Step through `actions`; return per-step (reward, done, channels) and the final state."""
    key = jax.random.PRNGKey(0)
    _obs, state = env.reset(key)
    trace = []
    for action in actions:
        _obs, state, rewards, dones, info = env.step(key, state, {AGENT: jnp.int32(action)})
        trace.append((float(rewards[AGENT]), bool(dones["__all__"]), info["reward_channels"]))
    return trace, state


def test_optimal_play_pays_milestone_then_final(env):
    actions = [INVEST] * PARAMS.invest_steps + [COMMIT] * PARAMS.commit_steps
    trace, state = _play(env, actions)
    rewards = [r for r, _, _ in trace]
    dones = [d for _, d, _ in trace]
    assert rewards[PARAMS.invest_steps - 1] == PARAMS.milestone_reward
    assert rewards[-1] == PARAMS.final_reward
    assert dones == [False] * (len(actions) - 1) + [True]
    assert int(state.timestep) == 0  # auto-reset after success
    channels = trace[-1][2]
    assert float(channels.final[0]) == PARAMS.final_reward
    assert float(channels.dense[0]) == 0.0


def test_harvest_only_pays_dense_until_horizon(env):
    trace, _state = _play(env, [HARVEST] * PARAMS.horizon)
    assert all(r == pytest.approx(PARAMS.harvest_reward) for r, _, _ in trace)
    assert [d for _, d, _ in trace][-1] is True
    assert all(float(ch.sparse[0]) == 0.0 and float(ch.final[0]) == 0.0 for _, _, ch in trace)


def test_harvest_breaks_the_invest_streak(env):
    actions = [INVEST] * (PARAMS.invest_steps - 1) + [HARVEST] + [INVEST] * PARAMS.invest_steps
    trace, _state = _play(env, actions)
    sparse = [float(ch.sparse[0]) for _, _, ch in trace]
    assert sparse[PARAMS.invest_steps - 1] == 0.0  # streak broken before the milestone
    assert sparse[-1] == PARAMS.milestone_reward  # paid once the streak is complete


def test_commit_before_milestone_does_nothing(env):
    trace, state = _play(env, [COMMIT] * (PARAMS.horizon - 1))
    assert all(r == 0.0 for r, _, _ in trace)
    assert not bool(state.milestone)


def test_channels_sum_to_reward_and_obs_in_bounds(env):
    key = jax.random.PRNGKey(1)
    obs, state = env.reset(key)
    for _ in range(PARAMS.horizon + 2):
        key, k_act = jax.random.split(key)
        action = jax.random.randint(k_act, (), 0, 3)
        obs, state, rewards, _dones, info = env.step(key, state, {AGENT: action})
        channels = info["reward_channels"]
        assert channels.total().shape == (1,)
        assert jnp.allclose(channels.total()[0], rewards[AGENT])
        assert obs[AGENT].shape == env.observation_space(AGENT).shape
        assert bool((obs[AGENT] >= 0).all())
        assert bool((obs[AGENT] <= 1).all())


def test_factory_builds_corridor_from_params():
    env = make_channel_env("TimescaleCorridor", {"invest_steps": 2, "horizon": 5})
    assert isinstance(env, TimescaleCorridor)
    assert env.params.invest_steps == 2
    assert env.params.commit_steps == CorridorParams().commit_steps
    with pytest.raises(TypeError, match="horizon"):
        make_channel_env("Craftax-MA-Symbolic", {"horizon": 5})


def test_random_rollout_on_corridor(env):
    stats = random_rollout(env, jax.random.PRNGKey(0), num_envs=4, steps=PARAMS.horizon)
    assert stats.returns.shape == (1,)
    assert set(stats.peak_stats) == {"milestone", "success"}
    assert jnp.allclose(stats.channel_returns.total(), stats.returns, atol=1e-5)


def test_rollout_keeps_pre_reset_stats():
    # One invest then one commit succeeds, so random play over 64 envs reaches
    # success; the peak must survive the auto-reset that follows it.
    env = TimescaleCorridor(CorridorParams(invest_steps=1, commit_steps=1, horizon=4))
    stats = random_rollout(env, jax.random.PRNGKey(0), num_envs=64, steps=8)
    success = float(stats.peak_stats["success"][0])
    milestone = float(stats.peak_stats["milestone"][0])
    assert 0.0 < success <= 1.0
    assert milestone >= success
    assert float(stats.channel_returns.final[0]) > 0.0
