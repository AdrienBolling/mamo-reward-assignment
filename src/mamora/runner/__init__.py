"""Execution of one experiment.

The runner binds a configuration to an environment, an agent and a run record.
It owns the training loop, the seed control and the record format.

This is the only layer that may import every other layer.
"""
