"""Force CPU for the whole test session (must run before jax is imported anywhere)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
