# SPDX-License-Identifier: LGPL-3.0-or-later
"""Dense LU factorization, linear system solvers, and JAX differentiable solve.

Stores dense $N \\times N$ matrices in row-major order (`a[i, j]`, `float64`).
Provides `dgetrf` (LU factorization with partial pivoting), `dgetrs`
(forward/backward substitution for 1D vectors and 2D multi-RHS matrices),
`solve_dense`, `multiply_xa`, `lu_factored_square_mv`, `lu_factored_square_mm`,
`rcond_dense`, and the JAX-differentiable `solve_dense_jax`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt


def _validate_square(
    a: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], int]:
    """Validates a 2D finite square `float64` matrix `(N, N)`."""
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
        raise ValueError("Matrix a must have non-empty 2D square shape (N, N).")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Matrix a contains NaN or infinite entries.")
    return arr, int(arr.shape[0])


def dgetrf(
    a: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Computes the LU factorization $P A = L U$ with partial pivoting.

    Args:
        a: Dense square matrix of shape `(N, N)` in `float64` (dimensionless).

    Returns:
        Tuple `(lu, ipiv)` where `lu` has shape `(N, N)` in `float64` storing
        unit lower-triangular $L$ strictly below the diagonal and upper
        triangular $U$ on/above the diagonal, and `ipiv` has shape `(N,)` in
        `int64` storing 0-based pivot row indices.

    Raises:
        ValueError: If `a` is not a finite 2D square matrix or is singular.
    """
    arr, n = _validate_square(a)
    lu = arr.copy()
    ipiv = np.zeros(n, dtype=np.int64)
    for k in range(n):
        sub_col = np.abs(lu[k:, k])
        pivot_rel = int(np.argmax(sub_col))
        pivot_row = k + pivot_rel
        ipiv[k] = pivot_row
        if sub_col[pivot_rel] == 0.0:
            raise ValueError(f"Singular matrix at pivot index {k}.")
        if pivot_row != k:
            lu[[k, pivot_row], :] = lu[[pivot_row, k], :]
        diag = lu[k, k]
        if k + 1 < n:
            lu[k + 1 :, k] /= diag
            lu[k + 1 :, k + 1 :] -= np.outer(
                lu[k + 1 :, k], lu[k, k + 1 :]
            )
    return lu, ipiv


def dgetrs(
    lu: npt.ArrayLike,
    ipiv: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    trans: bool = False,
) -> npt.NDArray[np.float64]:
    """Solves $A x = b$ (or $A^{\\top} x = b$) from [`dgetrf`] LU factors.

    Args:
        lu: Factored LU matrix of shape `(N, N)` in `float64` (dimensionless).
        ipiv: 0-based pivot array of shape `(N,)` in `int64` (dimensionless).
        b: Right-hand side vector of shape `(N,)` or matrix of shape
            `(N, nrhs)` in `float64` (dimensionless).
        trans: When `True`, solves the transposed system $A^{\\top} x = b$.

    Returns:
        Solution array `x` of the same shape as `b` in `float64`.

    Raises:
        ValueError: If shapes mismatch, inputs are non-finite, or $U$ is
            singular.
    """
    lu_arr, n = _validate_square(lu)
    piv = np.asarray(ipiv, dtype=np.int64)
    rhs = np.asarray(b, dtype=np.float64)
    if piv.shape != (n,):
        raise ValueError("ipiv must have shape (N,).")
    if not np.all(np.isfinite(rhs)):
        raise ValueError("Right-hand side b contains NaN or infinite entries.")
    is_1d = rhs.ndim == 1
    if is_1d:
        if rhs.shape != (n,):
            raise ValueError("1D right-hand side b must have shape (N,).")
        work = rhs[:, np.newaxis].copy()
    elif rhs.ndim == 2 and rhs.shape[0] == n and rhs.shape[1] >= 1:
        work = rhs.copy()
    else:
        raise ValueError("b must have shape (N,) or (N, nrhs).")

    if np.any((piv < 0) | (piv >= n)):
        raise ValueError("Pivot index out of bounds in ipiv.")

    if not trans:
        for k in range(n):
            p = int(piv[k])
            if p != k:
                work[[k, p], :] = work[[p, k], :]
        for i in range(1, n):
            work[i, :] -= lu_arr[i, :i] @ work[:i, :]
        for i in range(n - 1, -1, -1):
            diag = lu_arr[i, i]
            if diag == 0.0:
                raise ValueError(f"Singular U factor at diagonal {i}.")
            if i + 1 < n:
                work[i, :] -= lu_arr[i, i + 1 :] @ work[i + 1 :, :]
            work[i, :] /= diag
    else:
        for i in range(n):
            diag = lu_arr[i, i]
            if diag == 0.0:
                raise ValueError(f"Singular U factor at diagonal {i}.")
            if i > 0:
                work[i, :] -= lu_arr[:i, i] @ work[:i, :]
            work[i, :] /= diag
        for i in range(n - 2, -1, -1):
            work[i, :] -= lu_arr[i + 1 :, i] @ work[i + 1 :, :]
        for k in range(n - 1, -1, -1):
            p = int(piv[k])
            if p != k:
                work[[k, p], :] = work[[p, k], :]

    return work[:, 0] if is_1d else work


def solve_dense(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Solves the dense linear system $A x = b$ in `float64`.

    Args:
        a: Dense coefficient matrix of shape `(N, N)` in `float64`
            (dimensionless).
        b: Right-hand side vector of shape `(N,)` or matrix `(N, nrhs)` in
            `float64` (dimensionless).

    Returns:
        Solution array `x` matching the shape of `b` in `float64`.

    Raises:
        ValueError: If `a` is singular or input dimensions are incompatible.
    """
    lu, ipiv = dgetrf(a)
    return dgetrs(lu, ipiv, b)


def multiply_xa(
    a: npt.ArrayLike,
    x: npt.ArrayLike,
    c: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Evaluates $b = x^{\\top} A$ (or $x^{\\top} A + c$) in `float64`.

    Args:
        a: Dense matrix of shape `(N, N)` in `float64` (dimensionless).
        x: Left vector of shape `(N,)` in `float64` (dimensionless).
        c: Optional additive vector of shape `(N,)` in `float64`
            (dimensionless).

    Returns:
        Row-vector product of shape `(N,)` in `float64`.

    Raises:
        ValueError: If input dimensions do not match `(N, N)` and `(N,)`.
    """
    a_arr, n = _validate_square(a)
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.shape != (n,):
        raise ValueError("Vector x must have shape (N,).")
    out: npt.NDArray[np.float64] = x_arr @ a_arr
    if c is not None:
        c_arr = np.asarray(c, dtype=np.float64)
        if c_arr.shape != (n,):
            raise ValueError("Vector c must have shape (N,).")
        out = out + c_arr
    return out


def lu_factored_square_mv(
    lu: npt.ArrayLike,
    ipiv: npt.ArrayLike,
    x: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Evaluates $y = A x = P^{-1} L U x$ directly from [`dgetrf`] factors.

    Args:
        lu: Factored LU matrix of shape `(N, N)` in `float64` (dimensionless).
        ipiv: 0-based pivot indices of shape `(N,)` in `int64` (dimensionless).
        x: Input vector of shape `(N,)` in `float64` (dimensionless).

    Returns:
        Product vector $y = A x$ of shape `(N,)` in `float64`.

    Raises:
        ValueError: If shapes do not match `(N, N)` and `(N,)`.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1:
        raise ValueError("x must be a 1D vector of shape (N,).")
    res = lu_factored_square_mm(lu, ipiv, x_arr[:, np.newaxis])
    return res[:, 0]


def lu_factored_square_mm(
    lu: npt.ArrayLike,
    ipiv: npt.ArrayLike,
    b: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Evaluates $C = A B = P^{-1} L U B$ directly from [`dgetrf`] factors.

    Args:
        lu: Factored LU matrix of shape `(N, N)` in `float64` (dimensionless).
        ipiv: 0-based pivot indices of shape `(N,)` in `int64` (dimensionless).
        b: Right-hand matrix of shape `(N, M)` in `float64` (dimensionless).

    Returns:
        Product matrix $C = A B$ of shape `(N, M)` in `float64`.

    Raises:
        ValueError: If dimensions do not match `(N, N)` and `(N, M)`.
    """
    lu_arr, n = _validate_square(lu)
    piv = np.asarray(ipiv, dtype=np.int64)
    b_arr = np.asarray(b, dtype=np.float64)
    if piv.shape != (n,) or b_arr.ndim != 2 or b_arr.shape[0] != n:
        raise ValueError("lu, ipiv, and b dimensions are incompatible.")
    u_b = np.triu(lu_arr) @ b_arr
    prod: npt.NDArray[np.float64] = u_b + np.tril(lu_arr, k=-1) @ u_b
    for k in range(n - 1, -1, -1):
        p = int(piv[k])
        if p < 0 or p >= n:
            raise ValueError("Pivot index out of bounds.")
        if p != k:
            prod[[k, p], :] = prod[[p, k], :]
    return prod


def rcond_dense(
    a: npt.ArrayLike,
    lu: npt.ArrayLike,
    ipiv: npt.ArrayLike,
) -> float:
    """Estimates the 1-norm reciprocal condition number $\\mathrm{rcond}(A)$.

    Args:
        a: Unfactored matrix of shape `(N, N)` in `float64` (dimensionless).
        lu: Factored LU matrix of shape `(N, N)` in `float64` (dimensionless).
        ipiv: 0-based pivot indices of shape `(N,)` in `int64` (dimensionless).

    Returns:
        Reciprocal 1-norm condition number in $[0, 1]$ (dimensionless).
    """
    a_arr, n = _validate_square(a)
    anorm = float(np.max(np.sum(np.abs(a_arr), axis=0)))
    if anorm == 0.0:
        return 0.0
    trial = np.full(n, 1.0 / n, dtype=np.float64)
    ainv_norm = 0.0
    for _ in range(5):
        y = dgetrs(lu, ipiv, trial, trans=False)
        est = float(np.sum(np.abs(y)))
        if est <= ainv_norm:
            break
        ainv_norm = est
        signs = np.where(y >= 0.0, 1.0, -1.0)
        z = dgetrs(lu, ipiv, signs, trans=True)
        max_idx = int(np.argmax(np.abs(z)))
        if abs(float(z[max_idx])) <= float(np.dot(z, trial)):
            break
        trial.fill(0.0)
        trial[max_idx] = 1.0
    if ainv_norm <= 0.0 or not np.isfinite(ainv_norm):
        return 0.0
    return float(1.0 / (anorm * ainv_norm))


def solve_dense_jax(
    a: jax.Array,
    b: jax.Array,
) -> jax.Array:
    """Differentiable dense linear solver $A(p) x(p) = b(p)$ in `float64`.

    Args:
        a: Dense JAX array of shape `(N, N)` in `float64` (dimensionless).
        b: Right-hand side JAX array of shape `(N,)` or `(N, nrhs)` in
            `float64` (dimensionless).

    Returns:
        Solution JAX array `x` in `float64` supporting `jax.grad`, `jax.jvp`,
        and `jax.vmap` via implicit linear-solve differentiation.
    """
    a64 = jnp.asarray(a, dtype=jnp.float64)
    b64 = jnp.asarray(b, dtype=jnp.float64)
    sol: jax.Array = jnp.linalg.solve(a64, b64)
    return sol
