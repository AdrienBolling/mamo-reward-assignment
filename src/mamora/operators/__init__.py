"""Aggregation and metric operators on credit channels.

An aggregation operator maps the channel gradients to one update direction.
Examples: weighted sum, normalized sum, PCGrad, CAGrad, GradNorm, a learned
interaction matrix.

A metric operator maps the channel gradients to a measurement. Examples:
per-channel norm, pairwise cosine, the Gram matrix, effective rank.

Rules for this layer:

- An operator is a pure function of pytrees. It holds no environment state.
- This layer imports ``mamora.contract`` and nothing else from ``mamora``.
- Every operator is testable without an environment and without an agent.
"""
