# SPDX-License-Identifier: LGPL-3.0-or-later
"""Banded matrix storage conversions, LU factorization, and linear solvers.

For an $N \\times N$ banded matrix with $k_l$ sub-diagonals and $k_u$
super-diagonals stored in a buffer `ab` of shape `(ldab, n)` in `float64`:
- In general band conversions (`dense_to_band`, `band_to_dense`) with
  `ldab >= kl + ku + 1`, entry $A_{i,j}$ resides at
  `row = (ldab - 1 - kl) + i - j` and column `j`.
- In banded LU factorization (`dgbtrf`, `dgbtrs`, `solve_banded`) with
  `ldab >= 2 * kl + ku + 1`, the top `kl` rows (`0..kl`) provide workspace for
  pivot fill-in and the original diagonal resides at `row = ldab - 1 - kl`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def dense_to_band(
    a: npt.ArrayLike,
    ml: int,
    mu: int,
    ldab: int | None = None,
) -> npt.NDArray[np.float64]:
    """Converts a dense `(N, N)` matrix `a` into band storage `(ldab, N)`.

    Args:
        a: Dense square matrix of shape `(N, N)` in `float64` (dimensionless).
        ml: Number of sub-diagonals $m_l \\ge 0$.
        mu: Number of super-diagonals $m_u \\ge 0$.
        ldab: Leading row dimension of output buffer (`>= ml + mu + 1`).
            Defaults to `ml + mu + 1`.

    Returns:
        Banded matrix `ab` of shape `(ldab, N)` in `float64`.

    Raises:
        ValueError: If bandwidths or `ldab` are invalid for `a`.
    """
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
        raise ValueError("Matrix a must have non-empty 2D square shape (N, N).")
    n = int(arr.shape[0])
    min_ld = ml + mu + 1
    ld = min_ld if ldab is None else int(ldab)
    if ml < 0 or mu < 0 or min_ld > n or ld < min_ld:
        raise ValueError("Invalid band dimensions (ml, mu, ldab) for N.")
    ab = np.zeros((ld, n), dtype=np.float64)
    diag_row = ld - 1 - ml
    for j in range(n):
        i_min = max(0, j - mu)
        i_max = min(n - 1, j + ml)
        for i in range(i_min, i_max + 1):
            ab[(diag_row + i) - j, j] = arr[i, j]
    return ab


def band_to_dense(
    ab: npt.ArrayLike,
    ml: int,
    mu: int,
) -> npt.NDArray[np.float64]:
    """Converts a banded matrix `ab` of shape `(ldab, N)` to dense `(N, N)`.

    Args:
        ab: Banded matrix of shape `(ldab, N)` in `float64` (dimensionless).
        ml: Number of sub-diagonals $m_l \\ge 0$.
        mu: Number of super-diagonals $m_u \\ge 0$.

    Returns:
        Dense matrix `a` of shape `(N, N)` in `float64`.

    Raises:
        ValueError: If `ab` shape or `(ml, mu)` bandwidths are invalid.
    """
    ab_arr = np.asarray(ab, dtype=np.float64)
    if ab_arr.ndim != 2 or ab_arr.shape[1] == 0:
        raise ValueError("ab must be a non-empty 2D array of shape (ldab, N).")
    ldab, n = int(ab_arr.shape[0]), int(ab_arr.shape[1])
    if ml < 0 or mu < 0 or ml + mu + 1 > n or ldab < ml + mu + 1:
        raise ValueError("Invalid band dimensions (ml, mu, ldab) for N.")
    a = np.zeros((n, n), dtype=np.float64)
    diag_row = ldab - 1 - ml
    for j in range(n):
        i_min = max(0, j - mu)
        i_max = min(n - 1, j + ml)
        for i in range(i_min, i_max + 1):
            a[i, j] = ab_arr[(diag_row + i) - j, j]
    return a


def band_multiply_xa(
    ab: npt.ArrayLike,
    kl: int,
    ku: int,
    x: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Evaluates $b = x^{\\top} A$ for a compact banded matrix `ab`.

    Matches upstream `do_band_multiply_xa` where $A_{i,j}$ resides at
    `ab[ku + i - j, j]`.

    Args:
        ab: Compact band matrix of shape `(ldab, N)` (`ldab >= kl + ku + 1`)
            in `float64` (dimensionless).
        kl: Lower bandwidth $k_l \\ge 0$.
        ku: Upper bandwidth $k_u \\ge 0$.
        x: Left vector of shape `(N,)` in `float64` (dimensionless).

    Returns:
        Row-vector product $b = x^{\\top} A$ of shape `(N,)` in `float64`.

    Raises:
        ValueError: If dimensions do not match `(ldab, N)` and `(N,)`.
    """
    ab_arr = np.asarray(ab, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    if ab_arr.ndim != 2 or x_arr.ndim != 1 or ab_arr.shape[1] != x_arr.shape[0]:
        raise ValueError("ab and x shapes are incompatible.")
    ldab, n = int(ab_arr.shape[0]), int(ab_arr.shape[1])
    if kl < 0 or ku < 0 or ldab < kl + ku + 1:
        raise ValueError("ldab must be >= kl + ku + 1.")
    out = np.zeros(n, dtype=np.float64)
    for j in range(n):
        i_min = max(0, j - ku)
        i_max = min(n - 1, j + kl)
        total = 0.0
        for i in range(i_min, i_max + 1):
            total += float(x_arr[i]) * float(ab_arr[(ku + i) - j, j])
        out[j] = total
    return out


def dgbtrf(
    ab: npt.ArrayLike,
    kl: int,
    ku: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Computes the banded LU factorization $P A = L U$ (`dgbtrf`).

    Args:
        ab: Banded matrix of shape `(ldab, N)` with `ldab >= 2 * kl + ku + 1`
            and diagonal at `row = ldab - 1 - kl` in `float64` (dimensionless).
        kl: Lower bandwidth $k_l \\ge 0$.
        ku: Upper bandwidth $k_u \\ge 0$.

    Returns:
        Tuple `(lu_ab, ipiv)` where `lu_ab` has shape `(ldab, N)` in `float64`
        and `ipiv` has shape `(N,)` in `int64` storing 0-based pivot indices.

    Raises:
        ValueError: If dimensions are invalid, inputs are non-finite, or a
            pivot is zero.
    """
    ab_arr = np.asarray(ab, dtype=np.float64)
    if ab_arr.ndim != 2 or ab_arr.shape[1] == 0:
        raise ValueError("ab must be a non-empty 2D array of shape (ldab, N).")
    ldab, n = int(ab_arr.shape[0]), int(ab_arr.shape[1])
    if kl < 0 or ku < 0 or ldab < 2 * kl + ku + 1:
        raise ValueError("ldab must be >= 2 * kl + ku + 1 for dgbtrf.")
    if not np.all(np.isfinite(ab_arr)):
        raise ValueError("Banded matrix ab contains NaN or infinite entries.")

    lu = ab_arr.copy()
    ipiv = np.zeros(n, dtype=np.int64)
    diag = ldab - 1 - kl
    fill_end = max(0, diag - ku)
    if fill_end > 0:
        lu[:fill_end, :] = 0.0

    ju = 0
    for j in range(n):
        km = min(kl, n - 1 - j)
        jp = 0
        max_val = abs(float(lu[diag, j]))
        for l_off in range(1, km + 1):
            val = abs(float(lu[diag + l_off, j]))
            if val > max_val:
                max_val = val
                jp = l_off
        ipiv[j] = j + jp
        if max_val == 0.0:
            raise ValueError(f"Singular banded matrix at column {j}.")
        ju = max(ju, min(n - 1, j + ku + jp))
        if jp != 0:
            for col in range(j, ju + 1):
                r1 = (diag + j) - col
                r2 = (diag + j + jp) - col
                lu[r1, col], lu[r2, col] = lu[r2, col], lu[r1, col]
        if km > 0:
            pivot = float(lu[diag, j])
            lu[diag + 1 : diag + 1 + km, j] /= pivot
            for col in range(j + 1, ju + 1):
                r_top = (diag + j) - col
                temp = float(lu[r_top, col])
                if temp != 0.0:
                    for l_off in range(1, km + 1):
                        r_sub = r_top + l_off
                        if r_sub <= diag + kl:
                            lu[r_sub, col] -= float(lu[diag + l_off, j]) * temp
    return lu, ipiv


def dgbtrs(
    lu_ab: npt.ArrayLike,
    kl: int,
    ku: int,
    ipiv: npt.ArrayLike,
    b: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Solves $A x = b$ using the banded LU factorization from [`dgbtrf`].

    Args:
        lu_ab: Factored band buffer of shape `(ldab, N)` in `float64`
            (dimensionless).
        kl: Lower bandwidth $k_l \\ge 0$.
        ku: Upper bandwidth $k_u \\ge 0$.
        ipiv: 0-based pivot indices of shape `(N,)` in `int64` (dimensionless).
        b: Right-hand side vector of shape `(N,)` in `float64` (dimensionless).

    Returns:
        Solution vector `x` of shape `(N,)` in `float64`.

    Raises:
        ValueError: If shapes mismatch, inputs are non-finite, or $U$ is
            singular.
    """
    lu = np.asarray(lu_ab, dtype=np.float64)
    piv = np.asarray(ipiv, dtype=np.int64)
    rhs = np.asarray(b, dtype=np.float64)
    if lu.ndim != 2 or rhs.ndim != 1 or lu.shape[1] != rhs.shape[0]:
        raise ValueError("Incompatible shapes in dgbtrs.")
    ldab, n = int(lu.shape[0]), int(lu.shape[1])
    if piv.shape != (n,) or ldab < 2 * kl + ku + 1:
        raise ValueError("Invalid ipiv or ldab in dgbtrs.")
    if not np.all(np.isfinite(rhs)):
        raise ValueError("Right-hand side b contains NaN or infinite entries.")

    x = rhs.copy()
    diag = ldab - 1 - kl
    for j in range(n):
        lm = min(kl, n - 1 - j)
        p = int(piv[j])
        if p < 0 or p >= n:
            raise ValueError("Pivot index out of bounds.")
        if p != j:
            x[p], x[j] = x[j], x[p]
        bj = float(x[j])
        for i_off in range(1, lm + 1):
            x[j + i_off] -= float(lu[diag + i_off, j]) * bj

    for j in range(n - 1, -1, -1):
        u_jj = float(lu[diag, j])
        if u_jj == 0.0:
            raise ValueError(f"Singular banded U factor at column {j}.")
        x[j] /= u_jj
        bj = float(x[j])
        lm = min(j, kl + ku)
        for k_off in range(1, lm + 1):
            x[j - k_off] -= float(lu[diag - k_off, j]) * bj
    return x


def solve_banded(
    ab: npt.ArrayLike,
    kl: int,
    ku: int,
    b: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Factors and solves a banded linear system $A x = b$ in `float64`.

    Args:
        ab: Banded matrix of shape `(ldab, N)` (`ldab >= 2 * kl + ku + 1`)
            in `float64` (dimensionless).
        kl: Lower bandwidth $k_l \\ge 0$.
        ku: Upper bandwidth $k_u \\ge 0$.
        b: Right-hand side vector of shape `(N,)` in `float64` (dimensionless).

    Returns:
        Solution vector `x` of shape `(N,)` in `float64`.
    """
    lu, ipiv = dgbtrf(ab, kl, ku)
    return dgbtrs(lu, kl, ku, ipiv, b)


def rcond_banded(
    ab: npt.ArrayLike,
    lu_ab: npt.ArrayLike,
    kl: int,
    ku: int,
    ipiv: npt.ArrayLike,
) -> float:
    """Estimates the 1-norm reciprocal condition number of a banded matrix.

    Args:
        ab: Unfactored band buffer `(ldab, N)` in `float64` (dimensionless).
        lu_ab: Factored band buffer `(ldab, N)` in `float64` (dimensionless).
        kl: Lower bandwidth $k_l \\ge 0$.
        ku: Upper bandwidth $k_u \\ge 0$.
        ipiv: 0-based pivot indices `(N,)` in `int64` (dimensionless).

    Returns:
        Reciprocal 1-norm condition number in $[0, 1]$ (dimensionless).
    """
    ab_arr = np.asarray(ab, dtype=np.float64)
    ldab, n = int(ab_arr.shape[0]), int(ab_arr.shape[1])
    diag = ldab - 1 - kl
    anorm = 0.0
    for j in range(n):
        i_min = max(0, j - ku)
        i_max = min(n - 1, j + kl)
        col_sum = sum(
            abs(float(ab_arr[(diag + i) - j, j]))
            for i in range(i_min, i_max + 1)
        )
        anorm = max(anorm, col_sum)
    if anorm == 0.0:
        return 0.0
    trial = np.full(n, 1.0 / n, dtype=np.float64)
    y = dgbtrs(lu_ab, kl, ku, ipiv, trial)
    ainv_est = float(np.sum(np.abs(y)))
    if ainv_est <= 0.0 or not np.isfinite(ainv_est):
        return 0.0
    return float(1.0 / (anorm * ainv_est))
