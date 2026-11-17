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

import contextlib
from collections.abc import Iterator
from typing import Any, Literal, cast

import jax
import numpy as np
import numpy.typing as npt

from flexmesa import _rust

Backend = Literal["auto", "rust", "jax"]
NumericsMode = Literal["strict", "fast"]
VALID_NUMERICS_MODES: tuple[NumericsMode, ...] = ("strict", "fast")
_JAX_TYPES: tuple[type[Any], ...] = (jax.Array, jax.core.Tracer)


def validate_numerics_mode(
    mode: str, *, label: str = "numerics_mode"
) -> NumericsMode:
    """Validate and normalize a ``"strict"`` / ``"fast"`` mode string.

    Args:
        mode: Raw mode string to normalize.
        label: Parameter or environment-variable name used in error messages.

    Returns:
        Canonical lowercase :data:`NumericsMode`.

    Raises:
        ValueError: If ``mode`` is not ``"strict"`` or ``"fast"``.
    """
    norm = mode.strip().lower()
    if norm == "strict":
        return "strict"
    if norm == "fast":
        return "fast"
    raise ValueError(f"{label} must be 'strict' or 'fast', got {mode!r}")


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


def get_numerics_mode() -> NumericsMode:
    """Return the active Rust arithmetic mode (``"strict"`` or ``"fast"``).

    Returns:
        Active :data:`NumericsMode` (defaults to ``"strict"``).
    """
    return cast(NumericsMode, _rust.get_numerics_mode())


def set_numerics_mode(mode: NumericsMode | str) -> None:
    """Set the process-wide Rust arithmetic mode (``"strict"`` or ``"fast"``).

    ``"strict"`` (default) evaluates vendored ``crlibm`` (``CRMATH``) routines
    with no fused multiply-add contraction and is the sole mode compared
    against MESA. ``"fast"`` evaluates platform ``f64`` methods with ``fma``
    enabled and is validated against ``"strict"``.

    Args:
        mode: Target numerics mode (``"strict"`` or ``"fast"``).

    Raises:
        ValueError: If ``mode`` is not ``"strict"`` or ``"fast"``.
    """
    _rust.set_numerics_mode(mode)


@contextlib.contextmanager
def numerics_mode_context(
    mode: NumericsMode | str,
) -> Iterator[NumericsMode]:
    """Temporarily switch the Rust arithmetic mode and restore on exit.

    Args:
        mode: Target numerics mode (``"strict"`` or ``"fast"``).

    Yields:
        Normalized active :data:`NumericsMode`.
    """
    prev = get_numerics_mode()
    set_numerics_mode(mode)
    try:
        yield get_numerics_mode()
    finally:
        set_numerics_mode(prev)


__all__ = [
    "VALID_NUMERICS_MODES",
    "Backend",
    "NumericsMode",
    "_rust",
    "as_c_f64",
    "get_numerics_mode",
    "is_jax_input",
    "numerics_mode_context",
    "resolve_backend",
    "set_numerics_mode",
    "validate_numerics_mode",
]
