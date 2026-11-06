# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend dispatch helper for the Three-Tier Hybrid Architecture.

Provides the :data:`Backend` type alias (``"auto" | "rust" | "jax"``) and
:func:`resolve_backend` so Tier B and Tier C routines can route:

- Python ``float`` / ``int`` scalars and NumPy ``ndarray`` inputs to the
  compiled Safe Rust ``flexmesa._rust`` (`PyO3`) extension when
  ``backend == "auto"`` or ``backend == "rust"``.
- JAX ``Array`` and traced differentiation values (``jax.grad``, ``jax.jvp``,
  ``jax.vmap``, ``jax.jit``) to branchless ``@jax.jit`` kernels when
  ``backend == "auto"`` or ``backend == "jax"``.
"""

from __future__ import annotations

from typing import Any, Literal

import jax
import numpy as np
import numpy.typing as npt

from flexmesa import _rust

Backend = Literal["auto", "rust", "jax"]
_JAX_TYPES: tuple[type[Any], ...] = (jax.Array, jax.core.Tracer)


def is_jax_input(*values: Any) -> bool:
    """Return ``True`` if any argument in ``values`` is a JAX array or tracer.

    Args:
        *values: Positional scalar or array arguments to inspect.

    Returns:
        ``True`` when at least one argument is a ``jax.Array`` or a
        ``jax.core.Tracer`` (such as inside ``jax.jit``, ``jax.grad``, or
        ``jax.vmap``), and ``False`` otherwise.
    """
    for v in values:
        if isinstance(v, _JAX_TYPES):
            return True
    return False


def resolve_backend(backend: Backend, *values: Any) -> Literal["rust", "jax"]:
    """Resolve ``backend`` (``"auto" | "rust" | "jax"``) for given ``values``.

    When ``backend == "auto"``, inspects ``values`` via :func:`is_jax_input`
    so JAX arrays and tracers route to ``"jax"`` while Python scalars and
    NumPy ``ndarray`` inputs route to ``"rust"``.

    Args:
        backend: Requested backend selector (``"auto"``, ``"rust"``, or
            ``"jax"``).
        *values: Input values whose runtime types determine automatic routing.

    Returns:
        Resolved backend string (``"rust"`` or ``"jax"``).

    Raises:
        ValueError: If ``backend`` is not one of ``"auto"``, ``"rust"``, or
            ``"jax"``.
    """
    if backend == "auto":
        return "jax" if is_jax_input(*values) else "rust"
    if backend == "rust":
        return "rust"
    if backend == "jax":
        return "jax"
    msg = f"Invalid backend {backend!r}; expected 'auto', 'rust', or 'jax'."
    raise ValueError(msg)


def as_c_f64(arr: Any) -> npt.NDArray[np.float64]:
    """Convert ``arr`` to a 1D C-contiguous ``float64`` NumPy array.

    Args:
        arr: Array-like input.

    Returns:
        Flat 1D C-contiguous ``np.float64`` view or copy.
    """
    return np.ascontiguousarray(arr, dtype=np.float64).ravel()


__all__ = [
    "Backend",
    "_rust",
    "as_c_f64",
    "is_jax_input",
    "resolve_backend",
]
