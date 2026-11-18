# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend dispatch helper for canonical Safe Rust and derived JAX.

Provides the :data:`Backend` type alias (``"auto" | "rust" | "jax"``) and
:func:`resolve_backend` so routines with both Safe Rust and JAX backends route:

- Python ``float`` / ``int`` scalars and NumPy ``ndarray`` inputs to the
  canonical Safe Rust ``flexmesa._rust`` (`PyO3`) extension when
  ``backend == "auto"`` or ``backend == "rust"``.
- JAX ``Array`` and traced differentiation values (``jax.grad``, ``jax.jvp``,
  ``jax.vmap``, ``jax.jit``) to derived ``@jax.jit`` kernels (validated
  against Rust ``strict`` mode) when ``backend == "auto"`` or
  ``backend == "jax"``.
"""

from __future__ import annotations

import collections.abc
import contextlib
import dataclasses
from collections.abc import Iterator
from typing import Any, Final, Literal, cast

import jax
import numpy as np
import numpy.typing as npt

from flexmesa import _rust

Backend = Literal["auto", "rust", "jax"]
NumericsMode = Literal["strict", "fast"]
VALID_NUMERICS_MODES: tuple[NumericsMode, ...] = ("strict", "fast")
FROZEN_JAX_ENGINE_STATUS: Final[str] = "frozen-unverified-until-#143"
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


def _contains_jax_value(val: Any, _seen: set[int] | None = None) -> bool:
    """Recursively return ``True`` when ``val`` contains a JAX value."""
    if isinstance(val, _JAX_TYPES) or callable(
        getattr(val, "__jax_array__", None)
    ):
        return True
    if val is None or isinstance(
        val, (bool, int, float, complex, str, bytes, bytearray, memoryview)
    ):
        return False
    if isinstance(val, np.ndarray):
        if not val.dtype.hasobject:
            return False
        seen = set() if _seen is None else _seen
        obj_id = id(val)
        if obj_id in seen:
            return False
        seen.add(obj_id)
        return any(
            _contains_jax_value(
                item.tolist() if isinstance(item, np.void) else item, seen
            )
            for item in val.flat
        )

    is_mapping = isinstance(val, collections.abc.Mapping)
    is_dc = dataclasses.is_dataclass(val) and not isinstance(val, type)
    is_seq_or_set = isinstance(
        val, (collections.abc.Sequence, collections.abc.Set)
    )
    is_pytree = (
        not is_mapping
        and not is_dc
        and not is_seq_or_set
        and bool(jax.tree_util.is_tree_node(type(val)))
    )
    if not (is_mapping or is_dc or is_seq_or_set or is_pytree):
        return False

    seen = set() if _seen is None else _seen
    obj_id = id(val)
    if obj_id in seen:
        return False
    seen.add(obj_id)

    if is_mapping:
        return any(
            _contains_jax_value(k, seen) or _contains_jax_value(v, seen)
            for k, v in val.items()
        )
    if is_dc:
        return any(
            _contains_jax_value(getattr(val, f.name), seen)
            for f in dataclasses.fields(cast(Any, val))
        )
    if is_seq_or_set:
        return any(_contains_jax_value(item, seen) for item in val)
    children, _ = jax.tree_util.flatten_one_level(val)
    return any(_contains_jax_value(child, seen) for child in children)


def is_jax_input(*values: Any) -> bool:
    """Return ``True`` if any argument in ``values`` is a JAX array or tracer.

    Inspects top-level values as well as nested containers, dataclasses,
    object-dtype NumPy arrays, ``__jax_array__`` objects, and registered JAX
    PyTrees so ``backend="auto"`` detects traced arguments inside composite
    inputs.

    Args:
        *values: Positional scalar, array, or nested container arguments.

    Returns:
        ``True`` when at least one leaf argument is a ``jax.Array`` or a
        ``jax.core.Tracer`` (such as inside ``jax.jit``, ``jax.grad``, or
        ``jax.vmap``), and ``False`` otherwise.
    """
    return any(_contains_jax_value(v) for v in values)


def resolve_backend_with_jax(
    backend: Backend, *values: Any
) -> tuple[Literal["rust", "jax"], bool]:
    """Resolve ``backend`` and report whether ``values`` contain JAX inputs.

    Avoids rescanning ``values`` when a caller needs both the resolved backend
    and whether to preserve JAX arrays/tracers rather than unboxing scalars.

    Args:
        backend: Requested backend selector (``"auto"``, ``"rust"``, or
            ``"jax"``).
        *values: Input values whose runtime types determine automatic routing.

    Returns:
        Tuple ``(resolved_backend, has_jax_input)`` where
        ``resolved_backend`` is ``"rust"`` or ``"jax"``.

    Raises:
        ValueError: If ``backend`` is not one of ``"auto"``, ``"rust"``, or
            ``"jax"``.
    """
    if backend == "rust":
        return "rust", False
    if backend == "auto":
        has_jax = is_jax_input(*values)
        return ("jax" if has_jax else "rust"), has_jax
    if backend == "jax":
        return "jax", is_jax_input(*values)
    msg = f"Invalid backend {backend!r}; expected 'auto', 'rust', or 'jax'."
    raise ValueError(msg)


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
    resolved, _ = resolve_backend_with_jax(backend, *values)
    return resolved


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
    "FROZEN_JAX_ENGINE_STATUS",
    "VALID_NUMERICS_MODES",
    "Backend",
    "NumericsMode",
    "_rust",
    "as_c_f64",
    "get_numerics_mode",
    "is_jax_input",
    "numerics_mode_context",
    "resolve_backend",
    "resolve_backend_with_jax",
    "set_numerics_mode",
    "validate_numerics_mode",
]
