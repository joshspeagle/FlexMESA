# SPDX-License-Identifier: LGPL-3.0-or-later
"""Sparse matrix format conversions (CSR and CSC, 1-based and 0-based).

Ports upstream `mtx_formats.dek` and `mtx_support.f90` while fixing the
upstream 0-based index bounds bugs in `row_sparse_0_based_to_dense` and
`find_loc_in_column_0_based`.
"""

from __future__ import annotations

import enum

import numpy as np
import numpy.typing as npt


class SparseFormat(enum.IntEnum):
    """Sparse matrix storage convention matching upstream `mtx_def.f90`."""

    COMPRESSED_COLUMN_SPARSE = 0
    COMPRESSED_ROW_SPARSE = 1
    COMPRESSED_COL_SPARSE_0_BASED = 2
    COMPRESSED_ROW_SPARSE_0_BASED = 3


def _dense_to_compressed(
    a: npt.ArrayLike,
    *,
    is_row_major: bool,
    include_diag: bool,
    base: int,
) -> tuple[
    npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]
]:
    """Converts dense `(N, N)` matrix to CSR or CSC with given index base."""
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
        raise ValueError("Matrix a must have non-empty 2D square shape (N, N).")
    n = int(arr.shape[0])
    ptr = np.zeros(n + 1, dtype=np.int64)
    indices: list[int] = []
    vals: list[float] = []
    for outer in range(n):
        ptr[outer] = len(vals) + base
        for inner in range(n):
            i, j = (outer, inner) if is_row_major else (inner, outer)
            val = float(arr[i, j])
            if val == 0.0 and (i != j or not include_diag):
                continue
            indices.append(inner + base)
            vals.append(val)
    ptr[n] = len(vals) + base
    return (
        ptr,
        np.asarray(indices, dtype=np.int64),
        np.asarray(vals, dtype=np.float64),
    )


def _compressed_to_dense(
    n: int,
    ptr: npt.ArrayLike,
    ind: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    is_row_major: bool,
    base: int,
) -> npt.NDArray[np.float64]:
    """Converts CSR or CSC arrays with given index base to dense `(N, N)`."""
    if n <= 0:
        raise ValueError("Matrix dimension n must be >= 1.")
    ptr_arr = np.asarray(ptr, dtype=np.int64)
    ind_arr = np.asarray(ind, dtype=np.int64)
    val_arr = np.asarray(values, dtype=np.float64)
    nz = int(val_arr.shape[0])
    if ptr_arr.shape[0] < n + 1 or ind_arr.shape[0] < nz:
        raise ValueError("Sparse pointer or index array is too short.")
    a = np.zeros((n, n), dtype=np.float64)
    for outer in range(n):
        start = int(ptr_arr[outer]) - base
        end = int(ptr_arr[outer + 1]) - base
        if start < 0 or end > nz or start > end:
            raise ValueError("Sparse pointer range out of bounds.")
        for k in range(start, end):
            inner = int(ind_arr[k]) - base
            if inner < 0 or inner >= n:
                raise ValueError("Sparse inner index out of bounds.")
            i, j = (outer, inner) if is_row_major else (inner, outer)
            a[i, j] = val_arr[k]
    return a


def dense_to_row_sparse(
    a: npt.ArrayLike,
    *,
    include_diag: bool = False,
) -> tuple[
    npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]
]:
    """Converts a dense `(N, N)` matrix into 1-based CSR `(rowptr, colind, v)`.

    Args:
        a: Dense square matrix of shape `(N, N)` in `float64` (dimensionless).
        include_diag: Retain zero diagonal entries when `True`.

    Returns:
        Tuple `(rowptr, colind, values)` with 1-based indices in `int64` and
        non-zero values in `float64`.
    """
    return _dense_to_compressed(
        a, is_row_major=True, include_diag=include_diag, base=1
    )


def dense_to_row_0_based(
    a: npt.ArrayLike,
    *,
    include_diag: bool = False,
) -> tuple[
    npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]
]:
    """Converts a dense `(N, N)` matrix into 0-based CSR `(rowptr, colind, v)`.

    Args:
        a: Dense square matrix of shape `(N, N)` in `float64` (dimensionless).
        include_diag: Retain zero diagonal entries when `True`.

    Returns:
        Tuple `(rowptr, colind, values)` with 0-based indices in `int64` and
        values in `float64`.
    """
    return _dense_to_compressed(
        a, is_row_major=True, include_diag=include_diag, base=0
    )


def row_sparse_to_dense(
    n: int,
    rowptr: npt.ArrayLike,
    colind: npt.ArrayLike,
    values: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Converts a 1-based CSR matrix into a dense `(N, N)` `float64` array.

    Args:
        n: Matrix dimension $N \\ge 1$.
        rowptr: 1-based row pointer array of shape `(N + 1,)` in `int64`.
        colind: 1-based column index array of shape `(nz,)` in `int64`.
        values: Non-zero values of shape `(nz,)` in `float64` (dimensionless).

    Returns:
        Dense matrix of shape `(N, N)` in `float64`.
    """
    return _compressed_to_dense(
        n, rowptr, colind, values, is_row_major=True, base=1
    )


def row_sparse_0_based_to_dense(
    n: int,
    rowptr: npt.ArrayLike,
    colind: npt.ArrayLike,
    values: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Converts a 0-based CSR matrix into a dense `(N, N)` `float64` array.

    Fixes the upstream `mtx_support.f90` 0-based CSR indexing bug.

    Args:
        n: Matrix dimension $N \\ge 1$.
        rowptr: 0-based row pointer array of shape `(N + 1,)` in `int64`.
        colind: 0-based column index array of shape `(nz,)` in `int64`.
        values: Non-zero values of shape `(nz,)` in `float64` (dimensionless).

    Returns:
        Dense matrix of shape `(N, N)` in `float64`.
    """
    return _compressed_to_dense(
        n, rowptr, colind, values, is_row_major=True, base=0
    )


def dense_to_col_sparse(
    a: npt.ArrayLike,
    *,
    include_diag: bool = False,
) -> tuple[
    npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]
]:
    """Converts a dense `(N, N)` matrix into 1-based CSC `(colptr, rowind, v)`.

    Args:
        a: Dense square matrix of shape `(N, N)` in `float64` (dimensionless).
        include_diag: Retain zero diagonal entries when `True`.

    Returns:
        Tuple `(colptr, rowind, values)` with 1-based indices in `int64`.
    """
    return _dense_to_compressed(
        a, is_row_major=False, include_diag=include_diag, base=1
    )


def dense_to_col_0_based(
    a: npt.ArrayLike,
    *,
    include_diag: bool = False,
) -> tuple[
    npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]
]:
    """Converts a dense `(N, N)` matrix into 0-based CSC `(colptr, rowind, v)`.

    Args:
        a: Dense square matrix of shape `(N, N)` in `float64` (dimensionless).
        include_diag: Retain zero diagonal entries when `True`.

    Returns:
        Tuple `(colptr, rowind, values)` with 0-based indices in `int64`.
    """
    return _dense_to_compressed(
        a, is_row_major=False, include_diag=include_diag, base=0
    )


def column_sparse_to_dense(
    n: int,
    colptr: npt.ArrayLike,
    rowind: npt.ArrayLike,
    values: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Converts a 1-based CSC matrix into a dense `(N, N)` `float64` array.

    Args:
        n: Matrix dimension $N \\ge 1$.
        colptr: 1-based column pointer array of shape `(N + 1,)` in `int64`.
        rowind: 1-based row index array of shape `(nz,)` in `int64`.
        values: Non-zero values of shape `(nz,)` in `float64` (dimensionless).

    Returns:
        Dense matrix of shape `(N, N)` in `float64`.
    """
    return _compressed_to_dense(
        n, colptr, rowind, values, is_row_major=False, base=1
    )


def col_sparse_0_based_to_dense(
    n: int,
    colptr: npt.ArrayLike,
    rowind: npt.ArrayLike,
    values: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Converts a 0-based CSC matrix into a dense `(N, N)` `float64` array.

    Args:
        n: Matrix dimension $N \\ge 1$.
        colptr: 0-based column pointer array of shape `(N + 1,)` in `int64`.
        rowind: 0-based row index array of shape `(nz,)` in `int64`.
        values: Non-zero values of shape `(nz,)` in `float64` (dimensionless).

    Returns:
        Dense matrix of shape `(N, N)` in `float64`.
    """
    return _compressed_to_dense(
        n, colptr, rowind, values, is_row_major=False, base=0
    )


def find_loc_in_sparse(
    sparse_format: SparseFormat,
    n: int,
    ia: npt.ArrayLike,
    ja: npt.ArrayLike,
    i: int,
    j: int,
    hint: int | None = None,
) -> int | None:
    """Finds the position of entry `(i, j)` in a sparse matrix `(ia, ja)`.

    Uses 1-based `(i, j)` and return index for 1-based formats, and 0-based
    `(i, j)` and return index for 0-based formats (fixing upstream
    `find_loc_in_column_0_based`).

    Args:
        sparse_format: Storage convention from [`SparseFormat`].
        n: Matrix dimension $N \\ge 1$.
        ia: Outer pointer array of shape `(N + 1,)` in `int64`.
        ja: Inner index array of shape `(nz,)` in `int64`.
        i: Row index (1-based or 0-based matching `sparse_format`).
        j: Column index (1-based or 0-based matching `sparse_format`).
        hint: Optional previous location hint in the same index base.

    Returns:
        Index into `values` (in the format's index base) if present, else
        `None`.

    Raises:
        ValueError: If `i` or `j` or pointer bounds are invalid.
    """
    ia_arr = np.asarray(ia, dtype=np.int64)
    ja_arr = np.asarray(ja, dtype=np.int64)
    if sparse_format in (
        SparseFormat.COMPRESSED_COLUMN_SPARSE,
        SparseFormat.COMPRESSED_ROW_SPARSE,
    ):
        base = 1
    else:
        base = 0
    is_col = sparse_format in (
        SparseFormat.COMPRESSED_COLUMN_SPARSE,
        SparseFormat.COMPRESSED_COL_SPARSE_0_BASED,
    )
    outer, inner = (j, i) if is_col else (i, j)
    outer_0 = outer - base
    inner_0 = inner - base
    if outer_0 < 0 or outer_0 >= n or inner_0 < 0 or inner_0 >= n:
        raise ValueError("Row or column index out of bounds.")
    start = int(ia_arr[outer_0]) - base
    end = int(ia_arr[outer_0 + 1]) - base
    if start < 0 or end > int(ja_arr.shape[0]) or start > end:
        raise ValueError("Pointer array range out of bounds.")
    if hint is not None:
        next_0 = (hint + 1) - base
        if start <= next_0 < end and int(ja_arr[next_0]) == inner:
            return next_0 + base
    for offset in range(start, end):
        if int(ja_arr[offset]) == inner:
            return offset + base
    return None
