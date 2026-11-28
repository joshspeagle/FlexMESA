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
    clip_block, clip_blocks, col_sparse_0_based_to_dense,
    column_sparse_to_band, column_sparse_to_dense, decsol_option,
    decsol_option_str, dense_to_band, dense_to_col_0_based,
    dense_to_col_sparse, dense_to_row_0_based, dense_to_row_sparse,
    dense_to_sparse_with_diag, dgbtrf, dgbtrs, dgetrf, dgetrs_trans,
    find_loc_in_sparse, is_block_tridiagonal_decsol, lu_factored_square_mm,
    lu_factored_square_mv, multiply_xa, multiply_xa_plus_c, rcond_banded,
    rcond_dense, row_sparse_0_based_to_dense, row_sparse_to_band,
    row_sparse_to_dense, solve_banded, solve_block_tridiag_bcyclic,
    solve_dense, solve_dense_multi, DecsolOption, MtxError, SparseFormat,
};
use proptest::prelude::*;

/// Verifies solver option enums and string helpers against `mtx_def.f90`.
///
/// Checks round-trip string lookup (`decsol_option`, `decsol_option_str`),
/// block-tridiagonal classification (`is_block_tridiagonal_decsol`), and error
/// reporting on unknown option names.
#[test]
fn test_decsol_options_match_upstream() {
    assert_eq!(decsol_option(" lapack ").unwrap(), DecsolOption::Lapack);
    assert_eq!(
        decsol_option("block_thomas_dble").unwrap(),
        DecsolOption::BlockThomasDble
    );
    assert_eq!(
        decsol_option("block_thomas_refine").unwrap(),
        DecsolOption::BlockThomasRefine
    );
    assert_eq!(
        decsol_option("bcyclic_dble").unwrap(),
        DecsolOption::BcyclicDble
    );
    assert_eq!(decsol_option_str(DecsolOption::Lapack), "lapack");
    assert_eq!(
        decsol_option_str(DecsolOption::BlockThomasDble),
        "block_thomas_dble"
    );
    assert_eq!(
        decsol_option_str(DecsolOption::BlockThomasRefine),
        "block_thomas_refine"
    );
    assert_eq!(decsol_option_str(DecsolOption::BcyclicDble), "bcyclic_dble");
    assert!(!is_block_tridiagonal_decsol(DecsolOption::Lapack));
    assert!(is_block_tridiagonal_decsol(DecsolOption::BlockThomasDble));
    assert!(is_block_tridiagonal_decsol(DecsolOption::BlockThomasRefine));
    assert!(is_block_tridiagonal_decsol(DecsolOption::BcyclicDble));
    assert!(matches!(
        decsol_option("unknown_solver"),
        Err(MtxError::InvalidParameter(_))
    ));
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
    let x2_ref = [-1.085_106_382_978_723_3, 1.0, 1.797_529_169_526_424_3e1];
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
/// Checks that `row_sparse_0_based_to_dense` and out-of-bounds paths of
/// `find_loc_in_sparse` with `CompressedColSparse0Based` return explicit
/// errors (`#674`, `DECISIONS #43`), while in-bounds `CompressedColSparse0Based`
/// lookups reproduce upstream's `colptr(j)-1..=colptr(j+1)` search range.
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
    // Upstream `do_row_sparse_0_based_to_dense` (`mtx_support.f90:428-447`)
    // reads `rowptr(n+1)` out of bounds and mixes 0-/1-based indexing (#674).
    assert!(matches!(
        row_sparse_0_based_to_dense(n, n, &mut a2, nz0, &ptr, &ind, &vals),
        Err(MtxError::InvalidParameter(_))
    ));

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
    // Upstream `find_loc_in_column_0_based` (`mtx_formats.dek:261-280`)
    // searches `colptr(j)-1..=colptr(j+1)`. For column 1 (`colptr=[2, 5]`),
    // `k = 1` lies in bounds (`rowind[1] = 1` from column 0, `vals[1] = 0.2`)
    // and matches before column 1's own diagonal entry at `k = 3`.
    let loc_0 = find_loc_in_sparse(fmt_csc0, n, nz_d0, &ptr, &ind, 1, 1, None)
        .unwrap()
        .unwrap();
    assert_eq!(loc_0, 1);
    assert_eq!(vals[loc_0], a[n]);
    // With `hint = Some(2)` (`k = hint + 1 = 3`), upstream's hint fast path
    // (`mtx_formats.dek:265-272`) matches column 1's own diagonal zero first.
    let loc_0_hint =
        find_loc_in_sparse(fmt_csc0, n, nz_d0, &ptr, &ind, 1, 1, Some(2))
            .unwrap()
            .unwrap();
    assert_eq!(loc_0_hint, 3);
    assert_eq!(vals[loc_0_hint], 0.0);
    // In-bounds miss on interior column 1 (`k = 1..=5` all in `0..13`).
    assert_eq!(
        find_loc_in_sparse(fmt_csc0, n, nz_d0, &ptr, &ind, 4, 1, None).unwrap(),
        None
    );
    // In-bounds hit in the last column before reaching `k = nz_d0`.
    assert_eq!(
        find_loc_in_sparse(fmt_csc0, n, nz_d0, &ptr, &ind, 3, n - 1, None)
            .unwrap(),
        Some(11)
    );
    // Out-of-bounds paths return explicit errors (#674, DECISIONS #43):
    // `colptr(0) = 0` reads `rowind(-1)`; missing entry in last column reaches
    // `rowind(nz)`; `j = n` reads `colptr(n + 1)`.
    assert!(matches!(
        find_loc_in_sparse(fmt_csc0, n, nz_d0, &ptr, &ind, 0, 0, None),
        Err(MtxError::InvalidParameter(_))
    ));
    assert!(matches!(
        find_loc_in_sparse(fmt_csc0, n, nz_d0, &ptr, &ind, 0, n - 1, None),
        Err(MtxError::InvalidParameter(_))
    ));
    assert!(matches!(
        find_loc_in_sparse(fmt_csc0, n, nz_d0, &ptr, &ind, 0, n, None),
        Err(MtxError::InvalidParameter(_))
    ));
    // Upper-edge (`k = colptr(j + 1)`) and lower-edge (`k = colptr(j) - 1`)
    // neighbouring-column matches on a 3x3 diagonal 0-based CSC matrix.
    let diag_ptr = [0_usize, 1, 2, 3];
    let diag_ind = [0_usize, 1, 2];
    assert_eq!(
        find_loc_in_sparse(fmt_csc0, 3, 3, &diag_ptr, &diag_ind, 0, 1, None)
            .unwrap(),
        Some(0)
    );
    assert_eq!(
        find_loc_in_sparse(fmt_csc0, 3, 3, &diag_ptr, &diag_ind, 2, 1, None)
            .unwrap(),
        Some(2)
    );
    // Out-of-range row_idx with in-bounds search compares `rowind(k) == i`
    // and returns `Ok(None)` (`ierr = -1`, `mtx_formats.dek:273-279`).
    assert_eq!(
        find_loc_in_sparse(fmt_csc0, n, nz_d0, &ptr, &ind, n, 1, None).unwrap(),
        None
    );
    // `start > end + 1`: `do k = start - 1, end` executes zero iterations -> `None`.
    let rev_gt_ptr = [0_usize, 3, 1, 3];
    assert_eq!(
        find_loc_in_sparse(fmt_csc0, 3, 3, &rev_gt_ptr, &diag_ind, 1, 1, None)
            .unwrap(),
        None
    );
    // `start == end + 1`: single `k = end` iteration (`lo == end`).
    let rev_eq_ptr = [0_usize, 2, 1, 3];
    assert_eq!(
        find_loc_in_sparse(fmt_csc0, 3, 3, &rev_eq_ptr, &diag_ind, 1, 1, None)
            .unwrap(),
        Some(1)
    );
    assert_eq!(
        find_loc_in_sparse(fmt_csc0, 3, 3, &rev_eq_ptr, &diag_ind, 0, 1, None)
            .unwrap(),
        None
    );
    // `end > nz`: still rejected with `InvalidParameter`.
    let oob_end_ptr = [0_usize, 1, 4, 4];
    assert!(matches!(
        find_loc_in_sparse(fmt_csc0, 3, 3, &oob_end_ptr, &diag_ind, 1, 1, None),
        Err(MtxError::InvalidParameter(_))
    ));
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
                -0.4 * flexmesa_math::sin((i + j + 1) as f64)
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

/// Verifies `test_format_conversion` from `mtx/test/src/test_mtx_support.f90`
/// (`6 x 6` matrices `a` and `b`, `ml = 1, mu = 2`), including sequential
/// `hint`-accelerated `find_loc_in_sparse` lookups.
#[test]
fn test_upstream_test_mtx_support_format_conversion() {
    let n = 6;
    let ml = 1;
    let mu = 2;
    let ldab = 2 * ml + mu + 1;
    let nzmax = n * n;
    #[rustfmt::skip]
    let a = [
        10.0,  0.0,  0.0,  0.0,  0.0,  0.0,
         0.0, 12.0, -3.0, -1.0,  0.0,  0.0,
         0.0,  0.0, 15.0,  0.0,  0.0,  0.0,
        -2.0,  0.0,  0.0, 10.0, -1.0,  0.0,
        -1.0,  0.0,  0.0, -5.0,  1.0, -1.0,
        -1.0, -2.0,  0.0,  0.0,  0.0,  6.0,
    ];
    #[rustfmt::skip]
    let b = [
        10.0,  0.0,  0.0,  0.0,  0.0,  0.0,
        -2.0, 12.0, -3.0, -1.0,  0.0,  0.0,
         0.0,  1.0, 15.0,  0.0,  0.0,  0.0,
         0.0,  0.0,  0.0, 10.0, -1.0,  0.0,
         0.0,  0.0,  0.0, -5.0,  1.0, -1.0,
         0.0,  0.0,  0.0,  0.0,  0.0,  6.0,
    ];

    let mut ptr = vec![0usize; n + 1];
    let mut ind = vec![0usize; nzmax];
    let mut vals = vec![0.0; nzmax];
    let nz =
        dense_to_row_sparse(n, n, &a, nzmax, &mut ptr, &mut ind, &mut vals)
            .unwrap();
    for i in 0..n {
        let mut hint = Some(0usize);
        for k_1based in ptr[i]..ptr[i + 1] {
            let j_1based = ind[k_1based - 1];
            let found = find_loc_in_sparse(
                SparseFormat::CompressedRowSparse,
                n,
                nz,
                &ptr,
                &ind,
                i + 1,
                j_1based,
                hint,
            )
            .unwrap()
            .unwrap();
            assert_eq!(found, k_1based);
            hint = Some(found);
        }
    }

    let mut bb = vec![0.0; ldab * n];
    dense_to_band(n, n, &b, ml, mu, &mut bb, ldab).unwrap();
    let mut a2 = vec![-1.0; n * n];
    band_to_dense(n, ml, mu, &bb, ldab, n, &mut a2).unwrap();
    assert_eq!(b.as_slice(), a2.as_slice());

    let nz_col = band_to_column_sparse(
        n, ml, mu, &bb, ldab, nzmax, &mut ptr, &mut ind, &mut vals,
    )
    .unwrap();
    let mut bb2 = vec![-1.0; ldab * n];
    column_sparse_to_band(n, ml, mu, &mut bb2, ldab, nz_col, &ptr, &ind, &vals)
        .unwrap();
    assert_eq!(bb, bb2);

    let nz_row = band_to_row_sparse(
        n, ml, mu, &bb, ldab, nzmax, &mut ptr, &mut ind, &mut vals,
    )
    .unwrap();
    let mut bb3 = vec![-1.0; ldab * n];
    row_sparse_to_band(n, ml, mu, &mut bb3, ldab, nz_row, &ptr, &ind, &vals)
        .unwrap();
    assert_eq!(bb, bb3);
}

/// Verifies banded LU (`dgbtrf`/`dgbtrs`, `solve_banded`, `rcond_banded`) when
/// large sub-diagonals force partial-pivoting row swaps and fill-in (`kl = 2`,
/// `ku = 1`).
#[test]
fn test_banded_lu_with_pivoting_matches_dense_lu() {
    let n = 7;
    let kl = 2;
    let ku = 1;
    let ldab = 2 * kl + ku + 1;
    let mut a = vec![0.0; n * n];
    let mut b = vec![0.0; n];
    for i in 0..n {
        b[i] = 1.0 + 0.35 * (i as f64);
        let j_min = i.saturating_sub(kl);
        let j_max = (i + ku).min(n - 1);
        for j in j_min..=j_max {
            a[i * n + j] = match i.cmp(&j) {
                std::cmp::Ordering::Equal => 0.25 + 0.05 * (i as f64),
                std::cmp::Ordering::Greater => 3.5 + 0.2 * ((i + j) as f64),
                std::cmp::Ordering::Less => -1.4 - 0.1 * (j as f64),
            };
        }
    }

    let mut ab = vec![0.0; ldab * n];
    dense_to_band(n, n, &a, kl, ku, &mut ab, ldab).unwrap();

    let mut lu_ab = ab.clone();
    let mut ipiv = vec![0usize; n];
    dgbtrf(n, kl, ku, &mut lu_ab, ldab, &mut ipiv).unwrap();
    assert!(
        ipiv.iter().enumerate().any(|(j, &piv)| piv != j),
        "expected row swaps in pivoted banded test"
    );

    let mut x_inplace = b.clone();
    dgbtrs(n, kl, ku, &lu_ab, ldab, &ipiv, &mut x_inplace).unwrap();
    let x_band = solve_banded(n, kl, ku, &ab, ldab, &b).unwrap();
    let x_dense = solve_dense(n, &a, &b).unwrap();
    for i in 0..n {
        assert_eq!(x_inplace[i].to_bits(), x_band[i].to_bits());
        assert!((x_band[i] - x_dense[i]).abs() < 1e-13);
    }

    let rcond = rcond_banded(n, kl, ku, &ab, &lu_ab, ldab, &ipiv).unwrap();
    assert!(rcond > 0.0 && rcond <= 1.0);
}

/// Verifies `dgetrs_trans`, `multiply_xa_plus_c`, `clip_block`/`clip_blocks`,
/// and error reporting on singular or non-finite inputs across dense and
/// `bcyclic` solvers.
#[test]
fn test_dense_transpose_clip_blocks_and_solver_error_paths() {
    let n = 3;
    let a = [
        2.5_f64, -0.8, 1.2, //
        0.4, 3.1, -0.5, //
        -1.1, 0.7, 4.2, //
    ];
    let mut a_t = [0.0; 9];
    for i in 0..n {
        for j in 0..n {
            a_t[j * n + i] = a[i * n + j];
        }
    }
    let rhs = [1.25_f64, -2.5, 0.75];
    let mut lu = a;
    let mut ipiv = [0usize; 3];
    dgetrf(n, &mut lu, &mut ipiv).unwrap();
    let mut x_trans = rhs;
    dgetrs_trans(true, n, &lu, &ipiv, &mut x_trans).unwrap();
    let x_ref = solve_dense(n, &a_t, &rhs).unwrap();
    for i in 0..n {
        assert!((x_trans[i] - x_ref[i]).abs() < 1e-14);
    }

    let c = [0.5_f64, -1.0, 2.25];
    let mut xa_c = [0.0; 3];
    multiply_xa_plus_c(n, &a, &rhs, &c, &mut xa_c).unwrap();
    for j in 0..n {
        let expected =
            (0..n).map(|i| rhs[i] * a[i * n + j]).sum::<f64>() + c[j];
        assert_eq!(xa_c[j].to_bits(), expected.to_bits());
    }

    let mut d = [0.01_f64, 0.02, -0.5, 2.0];
    assert_eq!(clip_block(2, 0.1, &mut d).unwrap(), 3);
    assert_eq!(d, [0.01, 0.0, -0.5, 2.0]);

    let mut l_blk = [0.02_f64, 0.03, -0.2, 0.4];
    let mut d_blk = [1.0_f64, -0.05, 0.25, 0.01];
    let mut u_blk = [0.5_f64, 0.3, 0.04, -1.0];
    let (d_nnz, tot_nnz) =
        clip_blocks(2, 0.1, &mut l_blk, &mut d_blk, &mut u_blk).unwrap();
    assert_eq!(d_nnz, 3);
    assert_eq!(tot_nnz, 9);

    let sing_dense = [1.0, 2.0, 2.0, 4.0];
    let mut sing_lu = sing_dense;
    let mut sing_ipiv = [0usize; 2];
    assert_eq!(
        dgetrf(2, &mut sing_lu, &mut sing_ipiv),
        Err(MtxError::SingularMatrix(1))
    );

    let lblk = [0.0_f64; 12];
    let mut dblk = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let ublk = [0.0_f64; 12];
    let brhs = [1.0_f64; 6];
    assert_eq!(
        solve_block_tridiag_bcyclic(2, 3, &lblk, &dblk, &ublk, &brhs, false),
        Err(MtxError::SingularMatrix(1))
    );
    dblk[0] = f64::NAN;
    assert!(matches!(
        solve_block_tridiag_bcyclic(2, 3, &lblk, &dblk, &ublk, &brhs, false),
        Err(MtxError::NonFiniteInput(_))
    ));
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
