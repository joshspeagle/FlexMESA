# SPDX-License-Identifier: LGPL-3.0-or-later
"""Smoke tests for the FlexMESA package initialization and precision."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import flexmesa


def test_flexmesa_enables_jax_float64_by_default() -> None:
    """Verify that flexmesa enables and restores 64-bit JAX arrays."""
    sample = jnp.asarray([1.0, 2.0, 3.0])
    assert sample.dtype == jnp.float64
    assert flexmesa.__version__ == "0.1.0"

    jax.config.update("jax_enable_x64", False)
    assert jnp.asarray([1.0]).dtype == jnp.float32
    assert flexmesa.enable_float64() is True
    assert jnp.asarray([1.0]).dtype == jnp.float64
