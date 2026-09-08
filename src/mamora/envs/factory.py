"""Factory over the channel environments: MA-Craftax and the timescale corridor.

Import rule for MA-Craftax internals: state, constants and helpers come from the
top-level `craftax_ma` / `craftax_coop` / `environment_base` namespaces, never from
`craftax.craftax_ma...`. The env modules themselves import those top-level names,
so only they yield the classes that runtime objects are instances of. The only
thing imported from the `craftax.` namespace is this factory.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from craftax.craftax_env import make_craftax_env_from_name

from mamora.envs.corridor import CorridorParams, TimescaleCorridor

if TYPE_CHECKING:
    from mamora.envs.base import ChannelEnv

# MA-Craftax envs are untyped; they implement the jaxmarl `MultiAgentEnv` interface.
type CraftaxEnv = Any

CRAFTAX_ENV_NAMES: tuple[str, ...] = (
    "Craftax-MA-Symbolic",
    "Craftax-MA-Pixels",
    "Craftax-Coop-Symbolic",
    "Craftax-Coop-Pixels",
)
CORRIDOR_ENV_NAME = "TimescaleCorridor"
ENV_NAMES: tuple[str, ...] = (*CRAFTAX_ENV_NAMES, CORRIDOR_ENV_NAME)


def make_env(name: str) -> CraftaxEnv:
    """Build a raw MA-Craftax environment by name (agent count is the env default)."""
    if name not in CRAFTAX_ENV_NAMES:
        msg = f"Unknown MA-Craftax environment {name!r}; expected one of {CRAFTAX_ENV_NAMES}"
        raise ValueError(msg)
    return make_craftax_env_from_name(name)


def make_channel_env(name: str, params: Mapping[str, Any] | None = None) -> ChannelEnv:
    """Build an environment that reports reward channels.

    `params` are the keyword arguments of the environment's parameter class
    (only the corridor has one); they come from the `env` config group.
    """
    if name == CORRIDOR_ENV_NAME:
        return TimescaleCorridor(CorridorParams(**dict(params or {})))
    if name not in CRAFTAX_ENV_NAMES:
        msg = f"Unknown environment {name!r}; expected one of {ENV_NAMES}"
        raise ValueError(msg)
    if params:
        msg = f"{name} takes no parameters, got {sorted(params)}"
        raise ValueError(msg)
    # Imported here: `mamora.rewards.craftax` imports this module.
    from mamora.rewards.craftax import CraftaxChannelEnv, craftax_family

    return CraftaxChannelEnv(make_env(name), craftax_family(name))
