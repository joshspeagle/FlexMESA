# SPDX-License-Identifier: LGPL-3.0-or-later
"""Dense, banded, sparse, and block-tridiagonal linear algebra (`mtx`).

Re-exports dense (`dgetrf`, `dgetrs`, `solve_dense`, `solve_dense_jax`),
banded (`dense_to_band`, `band_to_dense`, `dgbtrf`, `dgbtrs`, `solve_banded`),
sparse (`SparseFormat`, CSR/CSC conversions, `find_loc_in_sparse`), and
block-tridiagonal (`block_dble_mv`, `block_thomas_factor`,
`block_thomas_solve`, `solve_block_tridiagonal`, `solve_block_tridiag_pcr`,
`solve_block_tridiag_rust_ffi`, `solve_block_tridiagonal_jax`) routines in
`float64`.

Storage Conventions:
    - Dense matrices use row-major `(N, N)` `float64` arrays.
    - Banded matrices use `(ldab, N)` `float64` arrays where $A_{i,j}$ resides
      at `row = (ldab - 1 - kl) + i - j` and column `j`, with
      `ldab >= kl + ku + 1` for general conversions and
      `ldab >= 2 * kl + ku + 1` for LU factorization.
    - Sparse matrices support both 1-based (`COMPRESSED_ROW_SPARSE`,
      `COMPRESSED_COLUMN_SPARSE`) and 0-based (`COMPRESSED_ROW_SPARSE_0_BASED`,
      `COMPRESSED_COL_SPARSE_0_BASED`) pointer and index conventions.
    - Block-tridiagonal systems use `(nz, nvar, nvar)` `float64` arrays
      (`lblk`, `dblk`, `ublk`) across `nz` shells and `nvar` variables.
"""

from __future__ import annotations

from flexmesa.mtx import band, block_tridiag, dense, pcr, sparse

dgetrf = dense.dgetrf
dgetrs = dense.dgetrs
solve_dense = dense.solve_dense
multiply_xa = dense.multiply_xa
lu_factored_square_mv = dense.lu_factored_square_mv
lu_factored_square_mm = dense.lu_factored_square_mm
rcond_dense = dense.rcond_dense
solve_dense_jax = dense.solve_dense_jax

dense_to_band = band.dense_to_band
band_to_dense = band.band_to_dense
band_multiply_xa = band.band_multiply_xa
dgbtrf = band.dgbtrf
dgbtrs = band.dgbtrs
solve_banded = band.solve_banded
rcond_banded = band.rcond_banded

SparseFormat = sparse.SparseFormat
dense_to_row_sparse = sparse.dense_to_row_sparse
dense_to_row_0_based = sparse.dense_to_row_0_based
row_sparse_to_dense = sparse.row_sparse_to_dense
row_sparse_0_based_to_dense = sparse.row_sparse_0_based_to_dense
dense_to_col_sparse = sparse.dense_to_col_sparse
dense_to_col_0_based = sparse.dense_to_col_0_based
column_sparse_to_dense = sparse.column_sparse_to_dense
col_sparse_0_based_to_dense = sparse.col_sparse_0_based_to_dense
find_loc_in_sparse = sparse.find_loc_in_sparse

SolverMode = pcr.SolverMode
block_dble_mv = block_tridiag.block_dble_mv
block_multiply_xa = block_tridiag.block_multiply_xa
block_thomas_factor = block_tridiag.block_thomas_factor
block_thomas_solve = block_tridiag.block_thomas_solve
solve_block_tridiag_pcr = pcr.solve_block_tridiag_pcr
solve_block_tridiag_rust_ffi = pcr.solve_block_tridiag_rust_ffi
solve_block_tridiagonal = block_tridiag.solve_block_tridiagonal
solve_block_tridiagonal_jax = block_tridiag.solve_block_tridiagonal_jax

__all__ = [
    "SolverMode",
    "SparseFormat",
    "band",
    "band_multiply_xa",
    "band_to_dense",
    "block_dble_mv",
    "block_multiply_xa",
    "block_thomas_factor",
    "block_thomas_solve",
    "block_tridiag",
    "col_sparse_0_based_to_dense",
    "column_sparse_to_dense",
    "dense",
    "dense_to_band",
    "dense_to_col_0_based",
    "dense_to_col_sparse",
    "dense_to_row_0_based",
    "dense_to_row_sparse",
    "dgbtrf",
    "dgbtrs",
    "dgetrf",
    "dgetrs",
    "find_loc_in_sparse",
    "lu_factored_square_mm",
    "lu_factored_square_mv",
    "multiply_xa",
    "pcr",
    "rcond_banded",
    "rcond_dense",
    "row_sparse_0_based_to_dense",
    "row_sparse_to_dense",
    "solve_banded",
    "solve_block_tridiag_pcr",
    "solve_block_tridiag_rust_ffi",
    "solve_block_tridiagonal",
    "solve_block_tridiagonal_jax",
    "solve_dense",
    "solve_dense_jax",
    "sparse",
]
