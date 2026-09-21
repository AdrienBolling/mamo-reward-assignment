"""Factorized credit assignment for multi-objective, multi-agent RL.

A credit channel is one reward stream kept separate from the others. The
project splits reward by granularity (dense, sparse, final), keeps the channel
identity through the value and gradient computation, and then applies explicit
aggregation and metric operators before the policy update.

The package has these layers:

- ``contract``   - the reward-channel types that every other layer obeys.
- ``operators``  - aggregation and metric operators on channel gradients.
- ``envs``       - environments written for this project.
- ``connectors`` - outside environments adapted to the contract.
- ``agents``     - learners that consume channels and call operators.
- ``runner``     - binds a config to an environment, an agent and a record.
- ``analysis``   - statistics and figures computed from run records.

Each layer states its own import rule in its docstring.
"""
