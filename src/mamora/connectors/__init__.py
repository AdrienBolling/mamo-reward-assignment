"""Adapters that make an outside environment obey the reward-channel contract.

A connector wraps one external environment. It splits the environment reward
into channels, it reports the episode statistics that the analysis needs, and it
keeps the external dependency out of every other layer.

Rules for this layer:

- One module for each external environment.
- The split is explicit and documented, because the split defines the experiment.
- This layer imports ``mamora.contract`` and the external library only.
"""
