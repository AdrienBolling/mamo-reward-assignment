"""The reward-channel contract: the spec, the labels and the reward shape."""

import jax
import jax.numpy as jnp
import pytest

from mamora.contract import (
    CHANNEL_AXIS,
    ChannelEnv,
    ChannelSpec,
    Granularity,
    StepOutput,
    check_rewards,
    crossed_spec,
    stack,
    timescale_spec,
    total,
)


def test_timescale_spec_has_one_channel_for_each_granularity():
    spec = timescale_spec()
    assert spec.num_channels == 3
    assert spec.names == ("dense", "sparse", "final")
    assert spec.num_objectives == 1
    assert spec.select(granularity=Granularity.FINAL) == (2,)
    assert spec.label(0) == "return/dense"


def test_index_finds_a_channel_and_reports_an_unknown_one():
    spec = timescale_spec()
    assert spec.index("sparse") == 1
    with pytest.raises(KeyError, match="no channel named 'reward'"):
        spec.index("reward")


def test_crossed_spec_orders_objective_first_then_granularity():
    spec = crossed_spec(("food", "wood"), (Granularity.DENSE, Granularity.FINAL))
    assert spec.names == ("food_dense", "food_final", "wood_dense", "wood_final")
    assert spec.objective == (0, 0, 1, 1)
    assert spec.select(objective="wood") == (2, 3)
    assert spec.select(granularity=Granularity.DENSE) == (0, 2)
    assert spec.select(objective="food", granularity=Granularity.FINAL) == (1,)
    assert spec.label(3) == "wood/wood_final"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"names": ()}, "at least one channel"),
        ({"granularity": (Granularity.DENSE,)}, "same length"),
        ({"names": ("dense", "dense", "final")}, "must be unique"),
        ({"objective": (0, 0, 1)}, "outside"),
    ],
)
def test_spec_refuses_an_inconsistent_definition(kwargs, message):
    fields = {
        "names": ("dense", "sparse", "final"),
        "granularity": tuple(Granularity),
        "objective": (0, 0, 0),
    }
    with pytest.raises(ValueError, match=message):
        ChannelSpec(**(fields | kwargs))


def test_channels_stack_on_axis_zero_and_sum_to_the_scalar_reward():
    dense = jnp.array([1.0, 2.0])
    sparse = jnp.array([0.0, 3.0])
    final = jnp.array([10.0, 0.0])
    rewards = stack([dense, sparse, final])
    assert CHANNEL_AXIS == 0
    assert rewards.shape == (3, 2)
    assert jnp.allclose(total(rewards), jnp.array([11.0, 5.0]))


def test_check_rewards_reads_the_shape_only_and_runs_under_jit():
    spec = timescale_spec()
    rewards = jnp.zeros((3, 2))
    jax.jit(lambda r: (check_rewards(spec, r, num_agents=2), r)[1])(rewards)
    with pytest.raises(ValueError, match="does not match the contract"):
        check_rewards(spec, jnp.zeros((2, 2)), num_agents=2)


class _OneStepEnv:
    """The smallest environment that obeys the contract."""

    channel_spec = timescale_spec()
    num_agents = 2

    def reset(self, key):
        return jnp.zeros((self.num_agents,)), jnp.int32(0)

    def step(self, key, state, actions):
        reward = stack([jnp.ones(self.num_agents), jnp.zeros(self.num_agents)] * 2)[:3]
        return StepOutput(
            obs=jnp.zeros((self.num_agents,)),
            state=state + 1,
            reward=reward,
            done=jnp.zeros((self.num_agents,), dtype=bool),
            episode_done=jnp.bool_(False),
            info={},
        )


def test_an_environment_that_follows_the_contract_satisfies_the_protocol():
    env = _OneStepEnv()
    assert isinstance(env, ChannelEnv)
    out = env.step(jax.random.key(0), jnp.int32(0), jnp.zeros((2,), dtype=int))
    check_rewards(env.channel_spec, out.reward, env.num_agents)
