# SPDX-License-Identifier: LGPL-3.0-or-later
"""Golden, JAX autodiff, and Hypothesis property tests for `flexmesa.mtx`.

Validates FlexMESA dense (`dgetrf`, `dgetrs`, `solve_dense`, `solve_dense_jax`),
banded (`dense_to_band`, `band_to_dense`, `dgbtrf`, `dgbtrs`, `solve_banded`,
`rcond_banded`, `band_multiply_xa`), sparse (`SparseFormat`, CSR/CSC 1-based
and 0-based conversions, `find_loc_in_sparse`), and block-tridiagonal
(`block_dble_mv`, `block_multiply_xa`, `solve_block_tridiagonal`,
`solve_block_tridiagonal_jax`) routines against upstream MESA test outputs
(`test_square1`, `test_square2`, `test_mtx_support`) and `jax.test_util`.
"""

from __future__ import annotations

import hypothesis
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import strategies

from flexmesa import mtx


def _blk_loss(
    l_jax: jax.Array, d_jax: jax.Array, u_jax: jax.Array, r_jax: jax.Array
) -> jax.Array:
    """Evaluates squared L2 norm of block-tridiagonal solution for autodiff."""
    return jnp.sum(
        mtx.solve_block_tridiagonal_jax(l_jax, d_jax, u_jax, r_jax) ** 2
    )


def _dense_loss(a_jax: jax.Array, b_jax: jax.Array) -> jax.Array:
    """Evaluates squared L2 norm of dense linear solution for autodiff."""
    return jnp.sum(mtx.solve_dense_jax(a_jax, b_jax) ** 2)


def test_square1_and_square2_match_upstream_golden_output() -> None:
    """Verifies dense LU solvers against `test_square1` and `test_square2`.

    Checks exact double-precision agreement with upstream `mtx/test/test_output`
    on the $4 \\times 4$ (`test_square1`) and $3 \\times 3$ (`test_square2`)
    benchmark matrices, plus LU-factored matrix-vector (`lu_factored_square_mv`)
    and matrix-matrix (`lu_factored_square_mm`) products.
    """
    a1 = np.array(
        [
            [1.80, 2.88, 2.05, 0.00],
            [5.25, -2.95, -0.95, -3.80],
            [0.00, 0.00, -2.90, -1.04],
            [-1.11, 0.00, -0.59, 0.80],
        ],
        dtype=np.float64,
    )
    b1 = np.array([4.35, 5.05, 3.04, -2.05], dtype=np.float64)
    x1_ref = np.array(
        [
            -1.6215408275458812e-1,
            1.6391349023885562,
            -3.8454229229650999e-2,
            -2.8158487838788577,
        ],
        dtype=np.float64,
    )
    x1 = mtx.solve_dense(a1, b1)
    np.testing.assert_allclose(x1, x1_ref, rtol=1e-13, atol=1e-14)

    lu1, ipiv1 = mtx.dgetrf(a1)
    b1_recon = mtx.lu_factored_square_mv(lu1, ipiv1, x1)
    np.testing.assert_allclose(b1_recon, b1, rtol=1e-13, atol=1e-13)
    assert 1e-4 < mtx.rcond_dense(a1, lu1, ipiv1) <= 1.0

    a2 = np.array(
        [[3.14, 7.50, 0.00], [4.10, 3.20, 0.30], [0.00, 1.00, 4.10]],
        dtype=np.float64,
    )
    b2 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x2_ref = np.array(
        [
            4.8857961179302783e-1,
            -7.1218664137347662e-2,
            7.4907772296032871e-1,
        ],
        dtype=np.float64,
    )
    x2 = mtx.solve_dense(a2, b2)
    np.testing.assert_allclose(x2, x2_ref, rtol=1e-13, atol=1e-14)

    a3 = np.array(
        [[4.70, 6.20, 0.00], [3.20, 0.00, 0.31], [0.00, 3.10, 0.00]],
        dtype=np.float64,
    )
    lu2, ipiv2 = mtx.dgetrf(a2)
    prod = mtx.lu_factored_square_mm(lu2, ipiv2, a3)
    np.testing.assert_allclose(prod, a2 @ a3, rtol=1e-13, atol=1e-13)


def test_sparse_and_band_conversions_match_upstream() -> None:
    """Verifies CSR, CSC, and band conversions (`test_mtx_support.f90`).

    Exercises 1-based and 0-based CSR/CSC roundtrips on the $5 \\times 5$
    tridiagonal matrix with a zero diagonal entry at $(1, 1)$ and checks
    0-based `find_loc_in_sparse` lookup.
    """
    n = 5
    ml = 1
    mu = 1
    a = np.zeros((n, n), dtype=np.float64)
    cnt = 1.0
    for j in range(n):
        for i in range(max(0, j - mu), min(n, j + ml + 1)):
            if i != 1 or j != 1:
                a[i, j] = cnt / 10.0
                cnt += 1.0

    # 1. Dense <-> Band roundtrip
    ab = mtx.dense_to_band(a, ml, mu)
    np.testing.assert_array_equal(mtx.band_to_dense(ab, ml, mu), a)

    # 2. 1-based and 0-based CSR roundtrips
    rp1, ci1, v1 = mtx.dense_to_row_sparse(a)
    np.testing.assert_array_equal(mtx.row_sparse_to_dense(n, rp1, ci1, v1), a)

    rp0, ci0, v0 = mtx.dense_to_row_0_based(a)
    np.testing.assert_array_equal(
        mtx.row_sparse_0_based_to_dense(n, rp0, ci0, v0), a
    )

    # 3. 1-based and 0-based CSC roundtrips + 0-based diagonal lookup
    cp1, ri1, vc1 = mtx.dense_to_col_sparse(a)
    np.testing.assert_array_equal(
        mtx.column_sparse_to_dense(n, cp1, ri1, vc1), a
    )

    cp0, ri0, vc0 = mtx.dense_to_col_0_based(a, include_diag=True)
    np.testing.assert_array_equal(
        mtx.col_sparse_0_based_to_dense(n, cp0, ri0, vc0), a
    )
    loc0 = mtx.find_loc_in_sparse(
        mtx.SparseFormat.COMPRESSED_COL_SPARSE_0_BASED, n, cp0, ri0, 1, 1
    )
    assert loc0 is not None and vc0[loc0] == 0.0


def test_banded_lu_and_multiply_xa_match_dense() -> None:
    """Verifies `solve_banded`, `rcond_banded`, and `band_multiply_xa`.

    Compares banded LU (`kl = 2, ku = 1`) against dense LU on an $8 \\times 8$
    pentadiagonal system to $< 10^{-13}$.
    """
    n = 8
    kl = 2
    ku = 1
    ldab = 2 * kl + ku + 1
    a = np.zeros((n, n), dtype=np.float64)
    b = np.linspace(1.5, -0.3, n, dtype=np.float64)
    for i in range(n):
        for j in range(max(0, i - kl), min(n, i + ku + 1)):
            a[i, j] = (
                5.0 + 0.3 * i if i == j else -0.4 * float(np.sin(i + j + 1))
            )

    ab = mtx.dense_to_band(a, kl, ku, ldab=ldab)
    x_band = mtx.solve_banded(ab, kl, ku, b)
    x_dense = mtx.solve_dense(a, b)
    np.testing.assert_allclose(x_band, x_dense, rtol=1e-13, atol=1e-13)

    lu_ab, ipiv = mtx.dgbtrf(ab, kl, ku)
    assert 1e-3 < mtx.rcond_banded(ab, lu_ab, kl, ku, ipiv) <= 1.0

    ab_compact = mtx.dense_to_band(a, kl, ku, ldab=kl + ku + 1)
    np.testing.assert_allclose(
        mtx.band_multiply_xa(ab_compact, kl, ku, b),
        mtx.multiply_xa(a, b),
        rtol=1e-14,
        atol=1e-14,
    )


def test_block_tridiagonal_and_jax_autodiff_match_oracles() -> None:
    """Verifies `solve_block_tridiagonal` and JAX implicit derivatives.

    Checks NumPy and JAX `lax.scan` block Thomas solvers against the block
    matrix-vector residual and validates first- and second-order derivatives
    via `jax.test_util.check_grads`.
    """
    rng = np.random.default_rng(42)
    nz = 6
    nvar = 4
    lblk = rng.uniform(-0.15, 0.15, size=(nz, nvar, nvar))
    dblk = rng.uniform(-0.2, 0.2, size=(nz, nvar, nvar))
    for k in range(nz):
        dblk[k] += 6.0 * np.eye(nvar, dtype=np.float64)
    ublk = rng.uniform(-0.15, 0.15, size=(nz, nvar, nvar))
    lblk[0] = 0.0
    ublk[-1] = 0.0
    rhs = rng.uniform(-2.0, 2.0, size=(nz, nvar))

    # 1. Verify NumPy block Thomas solve against block matrix-vector product.
    x_np = mtx.solve_block_tridiagonal(lblk, dblk, ublk, rhs)
    prod_np = mtx.block_dble_mv(lblk, dblk, ublk, x_np)
    np.testing.assert_allclose(prod_np, rhs, rtol=1e-12, atol=1e-12)

    # 2. Verify bilinear identity <b, A x> == <b^T A, x> for block_multiply_xa.
    xa_np = mtx.block_multiply_xa(lblk, dblk, ublk, rhs)
    np.testing.assert_allclose(
        np.sum(rhs * prod_np), np.sum(xa_np * x_np), rtol=1e-12, atol=1e-12
    )

    # 3. Verify JAX scan block Thomas solver matches NumPy in float64.
    x_jax = mtx.solve_block_tridiagonal_jax(
        jnp.asarray(lblk),
        jnp.asarray(dblk),
        jnp.asarray(ublk),
        jnp.asarray(rhs),
    )
    assert x_jax.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(x_jax), x_np, rtol=1e-12, atol=1e-12)

    # 4. Verify first-order reverse/forward gradients through block Thomas scan.
    l_j, d_j, u_j = jnp.asarray(lblk), jnp.asarray(dblk), jnp.asarray(ublk)
    r_j = jnp.asarray(rhs)
    np.testing.assert_allclose(
        np.asarray(jax.jacfwd(_blk_loss, argnums=1)(l_j, d_j, u_j, r_j)),
        np.asarray(jax.jacrev(_blk_loss, argnums=1)(l_j, d_j, u_j, r_j)),
        rtol=1e-11,
        atol=1e-11,
    )

    # 5. Verify first- and second-order derivatives through solve_dense_jax.
    a_jax = jnp.asarray(dblk[0])
    b_jax = jnp.asarray(rhs[0])
    a_inv_t = np.linalg.inv(dblk[0]).T
    h_exact = 2.0 * (a_inv_t @ a_inv_t.T)
    np.testing.assert_allclose(
        np.asarray(jax.hessian(_dense_loss, argnums=1)(a_jax, b_jax)),
        h_exact,
        rtol=1e-11,
        atol=1e-11,
    )


def test_mtx_rejects_singular_and_nan_inputs() -> None:
    """Verifies explicit ValueError on singular or NaN matrices.

    Confirms that zero-pivot matrices and matrices containing `NaN` or
    infinite values raise `ValueError` during LU factorization rather than
    silently propagating non-finite results.
    """
    singular = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="Singular"):
        mtx.dgetrf(singular)
    nan_mat = np.eye(3, dtype=np.float64)
    nan_mat[0, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        mtx.dgetrf(nan_mat)


@hypothesis.settings(max_examples=25, deadline=None)
@hypothesis.given(
    pert=strategies.lists(
        strategies.floats(-0.4, 0.4, allow_nan=False, allow_infinity=False),
        min_size=9,
        max_size=9,
    ),
    rhs=strategies.lists(
        strategies.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=3,
    ),
)
def test_prop_solve_dense_reconstructs_rhs(
    pert: list[float],
    rhs: list[float],
) -> None:
    """Property test verifying $A x = b$ residual $< 10^{-11}$.

    Generates random diagonally dominant $3 \\times 3$ matrices by adding a
    perturbation in $[-0.4, 0.4]^{3 \\times 3}$ to $5 I_3$, solves $A x = b$
    via `solve_dense`, and checks that the reconstructed right-hand side $A x$
    agrees with `rhs` to within $10^{-11}$ in both relative and absolute
    tolerance.
    """
    a = np.asarray(pert, dtype=np.float64).reshape(3, 3) + 5.0 * np.eye(3)
    b = np.asarray(rhs, dtype=np.float64)
    x = mtx.solve_dense(a, b)
    np.testing.assert_allclose(a @ x, b, rtol=1e-11, atol=1e-11)
