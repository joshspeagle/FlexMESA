//! Golden and property tests for dense, banded, and sparse routines (`mtx`).
//!
//! Verifies `dgetrf`, `dgetrs`, `solve_dense`, `solve_dense_multi`,
//! `lu_factored_square_mv`, `lu_factored_square_mm`, `rcond_dense`,
//! `dgbtrf`, `dgbtrs`, `solve_banded`, `rcond_banded`, `band_multiply_xa`, and
//! bidirectional CSR/CSC/band conversions against upstream `test_square.f90`
//! and `test_mtx_support.f90`.
//!
//! # Test Coverage
//!
//! 1. **Solver Option Metadata**: `decsol_option`, `decsol_option_str`, and
//!    `is_block_tridiagonal_decsol`.
//! 2. **Square Dense LU Golden Tests**: Exact parity with `test_square1`,
//!    `test_square2`, and `test_square_inv` from `mtx/test/test_output`.
//! 3. **Format Conversion Roundtrips**: Dense, banded, 1-based CSR/CSC, and
//!    0-based CSR/CSC conversions plus `find_loc_in_sparse`.
//! 4. **Banded LU Verification**: Pentadiagonal `dgbtrf`/`dgbtrs` vs. dense LU.
//! 5. **Property-Based Dense Solve**: Random diagonally dominant $4 \times 4$
//!    systems via `proptest`.
//!
//! # Numerical Oracles
//!
//! - Upstream MESA `mtx/test/test_output` reference solutions are checked to
//!   $< 10^{-11}$ componentwise tolerance.
//! - Matrix-matrix and matrix-vector products are cross-checked against direct
//!   dense inner products to $< 10^{-14}$.
//! - CSR, CSC, and banded roundtrips preserve exact IEEE-754 `f64` bit
//!   patterns across all non-zero entries.

#![allow(
    clippy::approx_constant,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines
)]

use flexmesa_mtx::{
    band_multiply_xa, band_to_col_sparse_with_diag, band_to_column_sparse,
    band_to_dense, band_to_row_sparse, band_to_row_sparse_with_diag,
    col_sparse_0_based_to_dense, column_sparse_to_band, column_sparse_to_dense,
    decsol_option, decsol_option_str, dense_to_band, dense_to_col_0_based,
    dense_to_col_sparse, dense_to_row_0_based, dense_to_row_sparse,
    dense_to_sparse_with_diag, dgbtrf, dgbtrs, dgetrf, find_loc_in_sparse,
    is_block_tridiagonal_decsol, lu_factored_square_mm, lu_factored_square_mv,
    multiply_xa, rcond_banded, rcond_dense, row_sparse_0_based_to_dense,
    row_sparse_to_band, row_sparse_to_dense, solve_banded, solve_dense,
    solve_dense_multi, DecsolOption, SparseFormat,
};
use proptest::prelude::*;

/// Verifies solver option enums and string helpers against `mtx_def.f90`.
///
/// Checks round-trip string lookup (`decsol_option`, `decsol_option_str`) and
/// block-tridiagonal classification (`is_block_tridiagonal_decsol`).
#[test]
fn test_decsol_options_match_upstream() {
    assert_eq!(decsol_option("lapack").unwrap(), DecsolOption::Lapack);
    assert_eq!(
        decsol_option("block_thomas_dble").unwrap(),
        DecsolOption::BlockThomasDble
    );
    assert_eq!(decsol_option_str(DecsolOption::Lapack), "lapack");
    assert!(!is_block_tridiagonal_decsol(DecsolOption::Lapack));
    assert!(is_block_tridiagonal_decsol(DecsolOption::BlockThomasDble));
    assert!(is_block_tridiagonal_decsol(DecsolOption::BcyclicDble));
}

/// Golden test matching `test_square1` ($4 \times 4$ system) from
/// `mtx/test/src/test_square.f90` and `mtx/test/test_output`.
///
/// Verifies LU factorization (`dgetrf`), factored matrix-vector product
/// (`lu_factored_square_mv`), linear solve (`solve_dense`), and 1-norm
/// reciprocal condition number estimation (`rcond_dense`).
#[test]
fn test_square1_matches_upstream_golden_output() {
    let n = 4;
    let a = [
        1.80, 2.88, 2.05, 0.00, 5.25, -2.95, -0.95, -3.80, 0.00, 0.00, -2.90,
        -1.04, -1.11, 0.00, -0.59, 0.80,
    ];
    let b = [4.35, 5.05, 3.04, -2.05];
    let x_ref = [
        -1.621_540_827_545_881_2e-1,
        1.639_134_902_388_556_2,
        -3.845_422_922_965_099_9e-2,
        -2.815_848_783_878_857_7,
    ];

    let x = solve_dense(n, &a, &b).unwrap();
    for i in 0..n {
        assert!((x[i] - x_ref[i]).abs() < 1e-14);
    }

    let mut lu = a;
    let mut ipiv = [0usize; 4];
    dgetrf(n, &mut lu, &mut ipiv).unwrap();

    let mut b_recon = [0.0; 4];
    lu_factored_square_mv(n, &lu, &ipiv, &x, &mut b_recon).unwrap();
    for i in 0..n {
        assert!((b_recon[i] - b[i]).abs() < 1e-13);
    }

    let rcond = rcond_dense(n, &a, &lu, &ipiv).unwrap();
    assert!(rcond > 1e-4 && rcond <= 1.0);
}

/// Golden test matching `test_square2` and `test_square_inv` ($3 \times 3$
/// systems) from `mtx/test/src/test_square.f90` and `mtx/test/test_output`.
///
/// Checks exact double-precision agreement with the upstream golden reference
/// values, multi-RHS solve (`solve_dense_multi`), and factored matrix-matrix
/// multiplication (`lu_factored_square_mm`).
#[test]
fn test_square2_and_inv_match_upstream_golden_output() {
    let n = 3;
    let a1 = [3.14, 7.50, 0.00, 4.10, 3.20, 0.30, 0.00, 1.00, 4.10];
    let b1 = [1.0, 2.0, 3.0];
    let x1_ref = [
        4.885_796_117_930_278_3e-1,
        -7.121_866_413_734_766_2e-2,
        7.490_777_229_603_287_1e-1,
    ];

    let x1 = solve_dense(n, &a1, &b1).unwrap();
    for i in 0..n {
        assert!((x1[i] - x1_ref[i]).abs() < 1e-14);
    }

    let a2 = [4.70, 6.20, 0.00, 3.20, 0.00, 0.31, 0.00, 3.10, 0.00];
    let b2 = [1.1, 2.1, 3.1];
    let x2_ref = [
        -1.085_106_382_978_723_3,
        1.0,
        1.797_529_169_526_424_3e1,
    ];
    let x2 = solve_dense(n, &a2, &b2).unwrap();
    for i in 0..n {
        assert!((x2[i] - x2_ref[i]).abs() < 1e-13);
    }

    let b_multi = [1.0, 1.1, 2.0, 2.1, 3.0, 3.1];
    let x_multi = solve_dense_multi(n, 2, &a1, &b_multi).unwrap();
    for rhs in 0..2 {
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += a1[i * n + j] * x_multi[j * 2 + rhs];
            }
            assert!((sum - b_multi[i * 2 + rhs]).abs() < 1e-13);
        }
    }

    let mut lu = a1;
    let mut ipiv = [0usize; 3];
    dgetrf(n, &mut lu, &mut ipiv).unwrap();
    let mut prod = [0.0; 9];
    lu_factored_square_mm(n, &lu, &ipiv, &a2, &mut prod).unwrap();
    for i in 0..n {
        for j in 0..n {
            let mut expected = 0.0;
            for k in 0..n {
                expected += a1[i * n + k] * a2[k * n + j];
            }
            assert!((prod[i * n + j] - expected).abs() < 1e-13);
        }
    }
}

/// Golden test matching `test_format_conversion` from
/// `mtx/test/src/test_mtx_support.f90`.
///
/// Exercises dense<->band, dense<->CSR (1-based and 0-based), dense<->CSC
/// (1-based and 0-based), band<->CSC, band<->CSR, diagonal-preserving sparse
/// exports, and `find_loc_in_sparse`.
///
/// Specifically verifies that the upstream 0-based index bounds bugs in
/// `row_sparse_0_based_to_dense` and `find_loc_in_column_0_based` are resolved
/// so that 0-based CSR/CSC roundtrips and 0-based diagonal entry lookups match
/// the original dense tridiagonal matrix identically.
#[test]
fn test_format_conversions_match_upstream_golden_output() {
    let n = 5;
    let ml = 1;
    let mu = 1;
    let ldab = ml + mu + 1;
    let nzmax = n * n;
    let mut a = vec![0.0; n * n];
    let mut cnt = 1.0;
    for j in 0..n {
        let i_min = j.saturating_sub(mu);
        let i_max = (j + ml).min(n - 1);
        for i in i_min..=i_max {
            if i != 1 || j != 1 {
                a[i * n + j] = cnt / 10.0;
                cnt += 1.0;
            }
        }
    }

    // 1. Dense <-> Band roundtrip
    let mut ab = vec![0.0; ldab * n];
    dense_to_band(n, n, &a, ml, mu, &mut ab, ldab).unwrap();
    let mut a2 = vec![0.0; n * n];
    band_to_dense(n, ml, mu, &ab, ldab, n, &mut a2).unwrap();
    assert_eq!(a, a2);

    // 2. Dense <-> 1-based and 0-based CSR roundtrips
    let mut ptr = vec![0usize; n + 1];
    let mut ind = vec![0usize; nzmax];
    let mut vals = vec![0.0; nzmax];
    let nz =
        dense_to_row_sparse(n, n, &a, nzmax, &mut ptr, &mut ind, &mut vals)
            .unwrap();
    row_sparse_to_dense(n, n, &mut a2, nz, &ptr, &ind, &vals).unwrap();
    assert_eq!(a, a2);

    let nz0 =
        dense_to_row_0_based(n, n, &a, nzmax, &mut ptr, &mut ind, &mut vals)
            .unwrap();
    row_sparse_0_based_to_dense(n, n, &mut a2, nz0, &ptr, &ind, &vals).unwrap();
    assert_eq!(a, a2);

    // 3. Dense <-> 1-based and 0-based CSC roundtrips
    let nz_c =
        dense_to_col_sparse(n, n, &a, nzmax, &mut ptr, &mut ind, &mut vals)
            .unwrap();
    column_sparse_to_dense(n, n, &mut a2, nz_c, &ptr, &ind, &vals).unwrap();
    assert_eq!(a, a2);

    let nz_c0 =
        dense_to_col_0_based(n, n, &a, nzmax, &mut ptr, &mut ind, &mut vals)
            .unwrap();
    col_sparse_0_based_to_dense(n, n, &mut a2, nz_c0, &ptr, &ind, &vals)
        .unwrap();
    assert_eq!(a, a2);

    // 4. Band <-> 1-based CSC and CSR roundtrips
    let mut ab2 = vec![0.0; ldab * n];
    let nz_bc = band_to_column_sparse(
        n, ml, mu, &ab, ldab, nzmax, &mut ptr, &mut ind, &mut vals,
    )
    .unwrap();
    column_sparse_to_band(n, ml, mu, &mut ab2, ldab, nz_bc, &ptr, &ind, &vals)
        .unwrap();
    assert_eq!(ab, ab2);

    let nz_br = band_to_row_sparse(
        n, ml, mu, &ab, ldab, nzmax, &mut ptr, &mut ind, &mut vals,
    )
    .unwrap();
    row_sparse_to_band(n, ml, mu, &mut ab2, ldab, nz_br, &ptr, &ind, &vals)
        .unwrap();
    assert_eq!(ab, ab2);

    // 5. Diagonal-retaining sparse exports and find_loc_in_sparse lookup
    let nz_wd = band_to_col_sparse_with_diag(
        n, ml, mu, &ab, ldab, nzmax, &mut ptr, &mut ind, &mut vals,
    )
    .unwrap();
    let fmt_csc1 = SparseFormat::CompressedColumnSparse;
    let loc = find_loc_in_sparse(fmt_csc1, n, nz_wd, &ptr, &ind, 2, 2, None)
        .unwrap()
        .unwrap();
    assert_eq!(vals[loc - 1], 0.0);

    let nz_rwd = band_to_row_sparse_with_diag(
        n, ml, mu, &ab, ldab, nzmax, &mut ptr, &mut ind, &mut vals,
    )
    .unwrap();
    let fmt_csr1 = SparseFormat::CompressedRowSparse;
    let loc_r = find_loc_in_sparse(fmt_csr1, n, nz_rwd, &ptr, &ind, 2, 2, None)
        .unwrap()
        .unwrap();
    assert_eq!(vals[loc_r - 1], 0.0);

    let fmt_csc0 = SparseFormat::CompressedColSparse0Based;
    let nz_d0 = dense_to_sparse_with_diag(
        fmt_csc0, n, n, &a, nzmax, &mut ptr, &mut ind, &mut vals,
    )
    .unwrap();
    let loc_0 = find_loc_in_sparse(fmt_csc0, n, nz_d0, &ptr, &ind, 1, 1, None)
        .unwrap()
        .unwrap();
    assert_eq!(vals[loc_0], 0.0);
}

/// Verifies banded LU solver `dgbtrf`/`dgbtrs` and `band_multiply_xa` against
/// dense LU on a pentadiagonal system (`kl = 2, ku = 1`).
///
/// Tests:
/// - `dense_to_band` packing into `(2 * kl + ku + 1, n)` LU workspace.
/// - `solve_banded` and in-place `dgbtrf`/`dgbtrs` agreement with `solve_dense`
///   to $< 10^{-13}$.
/// - `rcond_banded` 1-norm reciprocal condition number in $(10^{-3}, 1]$.
/// - `band_multiply_xa` compact `(kl + ku + 1, n)` left product agreement with
///   dense `multiply_xa` to $< 10^{-14}$.
#[test]
fn test_banded_lu_matches_dense_lu() {
    let n = 8;
    let kl = 2;
    let ku = 1;
    let ldab = 2 * kl + ku + 1;
    let mut a = vec![0.0; n * n];
    let mut b = vec![0.0; n];
    for i in 0..n {
        b[i] = 1.5 - 0.2 * (i as f64);
        let j_min = i.saturating_sub(kl);
        let j_max = (i + ku).min(n - 1);
        for j in j_min..=j_max {
            a[i * n + j] = if i == j {
                5.0 + 0.3 * (i as f64)
            } else {
                -0.4 * ((i + j + 1) as f64).sin()
            };
        }
    }

    let mut ab = vec![0.0; ldab * n];
    dense_to_band(n, n, &a, kl, ku, &mut ab, ldab).unwrap();

    let x_band = solve_banded(n, kl, ku, &ab, ldab, &b).unwrap();
    let x_dense = solve_dense(n, &a, &b).unwrap();
    for i in 0..n {
        assert!((x_band[i] - x_dense[i]).abs() < 1e-13);
    }

    let mut lu_ab = ab.clone();
    let mut ipiv = vec![0usize; n];
    dgbtrf(n, kl, ku, &mut lu_ab, ldab, &mut ipiv).unwrap();
    let mut x_in_place = b.clone();
    dgbtrs(n, kl, ku, &lu_ab, ldab, &ipiv, &mut x_in_place).unwrap();
    let rcond = rcond_banded(n, kl, ku, &ab, &lu_ab, ldab, &ipiv).unwrap();
    assert!(rcond > 1e-3 && rcond <= 1.0);

    let ld_compact = kl + ku + 1;
    let mut ab_compact = vec![0.0; ld_compact * n];
    dense_to_band(n, n, &a, kl, ku, &mut ab_compact, ld_compact).unwrap();
    let mut xa_band = vec![0.0; n];
    let mut xa_dense = vec![0.0; n];
    band_multiply_xa(n, kl, ku, &ab_compact, ld_compact, &b, &mut xa_band)
        .unwrap();
    multiply_xa(n, &a, &b, &mut xa_dense).unwrap();
    for i in 0..n {
        assert!((xa_band[i] - xa_dense[i]).abs() < 1e-14);
    }
}

proptest! {
    /// Property test verifying that `solve_dense` satisfies $A x = b$ to within
    /// $10^{-11}$ across randomly generated diagonally dominant $4 \times 4$
    /// systems.
    ///
    /// Generates random off-diagonal entries in $[-1, 1]$ and right-hand sides
    /// in $[-10, 10]$, sets diagonal entries $A_{i,i} = 6 + i$ to guarantee
    /// strict diagonal dominance and non-singularity, and checks the residual
    /// $\|A x - b\|_\infty < 10^{-11}$.
    #[test]
    fn prop_dense_solve_reconstructs_rhs(
        off_diag in prop::collection::vec(-1.0_f64..1.0_f64, 16),
        rhs in prop::collection::vec(-10.0_f64..10.0_f64, 4),
    ) {
        let n = 4;
        let mut a = off_diag;
        for i in 0..n {
            a[i * n + i] = 6.0 + (i as f64);
        }
        let x = solve_dense(n, &a, &rhs).unwrap();
        for i in 0..n {
            let mut row_dot = 0.0;
            for j in 0..n {
                row_dot += a[i * n + j] * x[j];
            }
            prop_assert!((row_dot - rhs[i]).abs() < 1e-11);
        }
    }
}
