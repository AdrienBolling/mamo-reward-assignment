"""Learners that keep the channel identity of the reward.

An agent owns its networks, its losses and its update step. It reads the
channels through ``mamora.contract``, and it calls ``mamora.operators`` to
aggregate or to measure the channel gradients.

This layer imports ``mamora.contract`` and ``mamora.operators``. It never
imports a specific environment, so the same agent runs on a toy environment and
on a connected one.
"""
