"""The timescale corridor: its dynamics, its rewards, and its contract."""

import jax
import jax.numpy as jnp
import pytest

from mamora.contract import ChannelEnv, check_rewards, total
from mamora.envs.corridor import (
    COMMIT,
    HARVEST,
    INVEST,
    CorridorConfig,
    TimescaleCorridor,
)


def _act(action):
    return jnp.array([action], dtype=jnp.int32)


def _play(env, actions):
    """Run a fixed action sequence; return every step output in order."""
    key = jax.random.key(0)
    _, state = env.reset(key)
    outputs = []
    for step, action in enumerate(actions):
        out = env.step(jax.random.fold_in(key, step), state, _act(action))
        outputs.append(out)
        state = out.state
    return outputs


def _returns(outputs):
    """The per-channel return of the sequence, as a (num_channels,) vector."""
    return sum(out.reward[:, 0] for out in outputs)


OPTIMAL = [INVEST] * 5 + [COMMIT] * 3


def test_the_corridor_obeys_the_contract():
    env = TimescaleCorridor()
    assert isinstance(env, ChannelEnv)
    obs, state = env.reset(jax.random.key(0))
    assert obs.shape == (1, env.obs_size)
    assert jnp.all(obs == 0.0)
    out = env.step(jax.random.key(1), state, _act(HARVEST))
    check_rewards(env.channel_spec, out.reward, env.num_agents)
    assert out.done.shape == (env.num_agents,)
    assert out.episode_done.shape == ()


def test_the_optimal_sequence_pays_the_milestone_then_the_final_reward():
    env = TimescaleCorridor()
    spec = env.channel_spec
    outputs = _play(env, OPTIMAL)

    milestone = outputs[4]
    assert bool(milestone.info["milestone"])
    assert milestone.reward[spec.index("sparse"), 0] == 1.0
    assert not bool(milestone.episode_done)

    success = outputs[7]
    assert bool(success.info["success"])
    assert bool(success.episode_done)
    assert not bool(success.info["truncated"])
    assert success.reward[spec.index("final"), 0] == 10.0
    assert jnp.allclose(_returns(outputs), jnp.array([0.0, 1.0, 10.0]))
    assert total(_returns(outputs)) == 11.0


def test_success_resets_the_episode_and_keeps_the_terminal_observation():
    env = TimescaleCorridor()
    success = _play(env, OPTIMAL)[-1]
    # The caller acts on a fresh corridor.
    assert jnp.all(success.obs == 0.0)
    assert int(success.state.time) == 0
    assert not bool(success.state.milestone)
    # The terminal observation shows the milestone and the full commit streak.
    assert jnp.allclose(success.final_obs[0], jnp.array([0.0, 1.0, 1.0, 8 / 32]))


def test_harvesting_forever_pays_the_dense_channel_until_truncation():
    cfg = CorridorConfig(harvest_reward=0.3)
    env = TimescaleCorridor(cfg)
    outputs = _play(env, [HARVEST] * cfg.horizon)
    assert jnp.allclose(_returns(outputs), jnp.array([0.3 * cfg.horizon, 0.0, 0.0]))
    assert all(not bool(out.episode_done) for out in outputs[:-1])
    last = outputs[-1]
    assert bool(last.episode_done)
    assert bool(last.info["truncated"])
    assert not bool(last.info["success"])
    assert jnp.all(last.obs == 0.0)  # the next episode starts


def test_a_harvest_breaks_the_invest_streak():
    env = TimescaleCorridor()
    outputs = _play(env, [INVEST, INVEST, HARVEST, INVEST, INVEST, INVEST])
    assert not any(bool(out.info["milestone"]) for out in outputs)
    assert outputs[2].final_obs[0, 0] == 0.0  # progress went back to zero
    assert jnp.allclose(outputs[-1].final_obs[0, 0], 3 / 5)


def test_a_commit_before_the_milestone_makes_no_progress():
    env = TimescaleCorridor()
    outputs = _play(env, [COMMIT] * 3 + OPTIMAL)
    assert jnp.allclose(_returns(outputs), jnp.array([0.0, 1.0, 10.0]))
    assert bool(outputs[-1].info["success"])
    assert not any(bool(out.info["success"]) for out in outputs[:-1])


def test_an_invest_after_the_milestone_breaks_the_commit_streak():
    env = TimescaleCorridor()
    outputs = _play(env, [INVEST] * 5 + [COMMIT, COMMIT, INVEST, COMMIT, COMMIT, COMMIT])
    assert bool(outputs[-1].info["success"])
    assert not bool(outputs[-2].info["success"])
    assert int(outputs[7].state.commit) == 0


def test_the_control_pays_every_action_the_same_dense_reward():
    cfg = CorridorConfig(harvest_reward=0.3, invest_reward=0.3, commit_reward=0.3)
    outputs = _play(TimescaleCorridor(cfg), OPTIMAL)
    assert jnp.allclose(_returns(outputs), jnp.array([0.3 * 8, 1.0, 10.0]))


def test_the_corridor_vectorizes_and_jits():
    env = TimescaleCorridor()
    keys = jax.random.split(jax.random.key(0), 16)
    obs, state = jax.vmap(env.reset)(keys)
    assert obs.shape == (16, 1, env.obs_size)
    actions = jnp.full((16, 1), INVEST, dtype=jnp.int32)
    out = jax.jit(jax.vmap(env.step))(keys, state, actions)
    assert out.reward.shape == (16, 3, 1)
    assert out.obs.shape == (16, 1, env.obs_size)
    assert out.episode_done.shape == (16,)
    assert jnp.allclose(out.obs[:, 0, 0], 1 / 5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"invest_steps": 0}, "at least 1"),
        ({"horizon": 7}, "shorter than the 8 steps"),
    ],
)
def test_the_config_refuses_a_corridor_that_cannot_be_solved(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CorridorConfig(**kwargs)
