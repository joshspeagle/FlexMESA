# SPDX-License-Identifier: LGPL-3.0-or-later
"""FlexMESA: Modular stellar astrophysics in Safe Rust and Python/JAX.

This package provides the Python and JAX (`float64`) implementation of MESA's
foundational numerics, microphysics, and stellar evolution modules. Importing
`flexmesa` enables 64-bit floating-point precision in JAX (`jax_enable_x64`) so
that thermodynamic, spline, and linear-algebra calculations maintain IEEE-754
double precision by default.
"""

from __future__ import annotations

import jax


def enable_float64() -> bool:
    """Enable 64-bit IEEE-754 floating-point precision in JAX.

    Returns:
        True once `jax_enable_x64` has been configured.

    Examples:
        >>> import flexmesa
        >>> flexmesa.enable_float64()
        True
    """
    jax.config.update("jax_enable_x64", True)
    return True


_X64_ENABLED: bool = enable_float64()

__version__: str = "0.1.0"
__all__: list[str] = ["__version__", "enable_float64"]
