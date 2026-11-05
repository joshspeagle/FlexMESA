"""Smoke tests for the FlexMESA package initialization and precision."""

from __future__ import annotations

import jax.numpy as jnp

import flexmesa


def test_flexmesa_enables_jax_float64_by_default() -> None:
    """Verify that importing flexmesa enables 64-bit JAX arrays by default."""
    sample = jnp.asarray([1.0, 2.0, 3.0])
    assert sample.dtype == jnp.float64
    assert flexmesa.__version__ == "0.1.0"
