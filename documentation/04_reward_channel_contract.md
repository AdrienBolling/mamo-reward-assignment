# The Reward-Channel Contract

**Code contract — v0.1**
**Module:** `mamora.contract`

---

## 1. What a channel is

A credit channel is one reward stream that keeps its identity through the value
and the gradient computation. The project does not sum the channels into one
reward before the update. It sums them, or combines them in another way, only
where an operator says so.

Two labels describe a channel:

- **granularity** — how often the channel pays: `DENSE`, `SPARSE` or `FINAL`.
- **objective** — which objective the channel belongs to.

A single-objective experiment uses one objective and three granularities. A
multi-objective experiment crosses the two, so the number of channels is the
number of objectives times the number of granularities.

## 2. The spec

An environment declares its channels once, in a `ChannelSpec`:

```python
spec = timescale_spec()  # dense, sparse, final of one objective
spec = crossed_spec(("food", "wood"))  # one channel for each pair
```

The spec is frozen and hashable. A jitted function can take it as a static
argument. The spec holds no array, so it never enters a pytree.

Use `spec.select(...)` to get the positions of the channels with a given label,
and `spec.label(i)` to get the log name of one channel.

## 3. The arrays

1. The channel axis is axis 0 of every reward array.
2. A reward array has shape `(num_channels, num_agents)`.
3. `stack([...])` builds one from a list, one array for each channel.
4. `total(rewards)` gives the scalar reward, the sum over the channel axis.
5. `check_rewards(spec, rewards, num_agents)` reads shapes only, so it also runs
   inside a jitted function.

## 4. The environment interface

An environment implements `ChannelEnv`:

- `channel_spec` and `num_agents` describe it.
- `reset(key)` returns the observations and the state.
- `step(key, state, actions)` returns a `StepOutput`.

The environment resets itself when the episode ends. The reward of that step
belongs to the episode that ended. `episode_done` marks that step.

## 5. Rules for a split

1. The channels sum to the scalar reward of the environment. Document every
   difference, in the module that builds the channels.
2. The split defines the experiment. Do not change a split silently; a changed
   split makes two runs incomparable.
3. Do not force an environment into a taxonomy it does not have. Where a clean
   final reward does not exist, say so and describe what replaces it.
4. Keep the split in the connector or in the environment. An agent never
   decides what a channel means.
