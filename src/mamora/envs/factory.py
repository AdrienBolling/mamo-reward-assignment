"""Thin factory over the MA-Craftax environments.

Import rule for MA-Craftax internals: state, constants and helpers come from the
top-level `craftax_ma` / `craftax_coop` / `environment_base` namespaces, never from
`craftax.craftax_ma...`. The env modules themselves import those top-level names,
so only they yield the classes that runtime objects are instances of. The only
thing imported from the `craftax.` namespace is this factory.
"""

from __future__ import annotations

from typing import Any

from craftax.craftax_env import make_craftax_env_from_name

# MA-Craftax envs are untyped; they implement the jaxmarl `MultiAgentEnv` interface.
type CraftaxEnv = Any

ENV_NAMES: tuple[str, ...] = (
    "Craftax-MA-Symbolic",
    "Craftax-MA-Pixels",
    "Craftax-Coop-Symbolic",
    "Craftax-Coop-Pixels",
)


def make_env(name: str) -> CraftaxEnv:
    """Build an MA-Craftax environment by name (agent count is the env default)."""
    if name not in ENV_NAMES:
        msg = f"Unknown environment {name!r}; expected one of {ENV_NAMES}"
        raise ValueError(msg)
    return make_craftax_env_from_name(name)
