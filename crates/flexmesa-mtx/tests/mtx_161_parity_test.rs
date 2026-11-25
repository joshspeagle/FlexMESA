// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone zero-`$MESA_DIR` MESA `26.4.1` parity tests for issue #161.
//!
//! Covers:
//! - Hardcoded 0-ULP verification of all 22 `mtx` module-hex outputs
//! - Banded transpose solver (`dgbtrs_trans`) and Hager 1-norm `rcond_banded`
//! - Matrix preconditioners (`pre_conditioners.f90`) and `dgbsvx_*` wrappers
//!   (`DGBSVX_wrapper.f90`, including bounded lower-band loop)
//! - LAPACK dense/banded equilibration (`dgeequ`, `dlaqge`, `dgbequ`, `dlaqgb`)
//! - Reusable `bcyclic_factor` / `bcyclic_solve` (`bcyclic.f90`) 0-ULP parity
//! - Harwell-Boeing malformed input bounds, sparse `usize::MAX` hint,
//!   `check_blocks` overflow guard, and extreme-scale `dgeev_right` parity

#![forbid(unsafe_code)]
#![allow(
    clippy::approx_constant,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use flexmesa_math::{with_numerics_mode, NumericsMode};
use flexmesa_mtx::{
    bcyclic_factor, bcyclic_solve, block_dble_mv, compute_band_preconditioner,
    compute_block_left_preconditioner, compute_block_preconditioner,
    dense_to_band, dgbequ, dgbsvx_banded, dgbsvx_banded_from_bands,
    dgbsvx_block_tridiagonal, dgbtrf, dgbtrs, dgbtrs_trans, dgeequ,
    dgeev_right, dgetrf, dgetrs, dgetrs_multi, dgetrs_trans,
    find_loc_in_sparse, multiply_xa, parse_block_tridiagonal_text,
    rcond_banded, rcond_dense, solve_banded_equilibrated,
    solve_block_tridiag_bcyclic, solve_block_tridiagonal,
    solve_dense_equilibrated, write_block_tridiagonal_text, EigenResult,
    Equilibration, MtxError, SparseFormat,
};

fn transpose_square(n: usize, a: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[j * n + i] = a[i * n + j];
        }
    }
    out
}

#[test]
fn standalone_zero_mesa_dir_22_module_hex_zero_ulp() {
    with_numerics_mode(NumericsMode::Strict, || {
        // 1. `test_square1` (`mtx/test/src/test_square.f90:140-171`, 4 values).
        {
            const N: usize = 4;
            let mut a = [
                1.80_f64, 2.88, 2.05, 0.00, //
                5.25, -2.95, -0.95, -3.80, //
                0.00, 0.00, -2.90, -1.04, //
                -1.11, 0.00, -0.59, 0.80, //
            ];
            let b_init = [4.35_f64, 5.05, 3.04, -2.05];
            let mut b_multi = b_init;
            let mut b_single = b_init;
            let mut ipiv = [0_usize; N];
            dgetrf(N, &mut a, &mut ipiv).expect("dgetrf");
            dgetrs_multi(N, 1, &a, &ipiv, &mut b_multi).expect("dgetrs_multi");
            dgetrs(N, &a, &ipiv, &mut b_single).expect("dgetrs");
            let want: [u64; 4] = [
                0xbfc4c177092c02bd,
                0x3ffa39e584f7db24,
                0xbfa3b045d1de547c,
                0xc00686dbba29edb4,
            ];
            for i in 0..N {
                assert_eq!(b_single[i].to_bits(), want[i]);
                assert_eq!(b_multi[i].to_bits(), want[i]);
            }
        }

        // 2. `test_square2` (`mtx/test/src/test_square.f90:95-138`, 6 values).
        {
            const N: usize = 3;
            let mut a1 = [
                3.14_f64, 7.50, 0.00, //
                4.10, 3.20, 0.30, //
                0.00, 1.00, 4.10, //
            ];
            let mut b1 = [1.0_f64, 2.0, 3.0];
            let mut ipiv1 = [0_usize; N];
            dgetrf(N, &mut a1, &mut ipiv1).expect("dgetrf a1");
            dgetrs(N, &a1, &ipiv1, &mut b1).expect("dgetrs b1");
            let want1: [u64; 3] =
                [0x3fdf44e36b892df7, 0xbfb23b62e955aff7, 0x3fe7f871d848db38];
            for i in 0..N {
                assert_eq!(b1[i].to_bits(), want1[i]);
            }

            let mut a2 = [
                4.70_f64, 6.20, 0.00, //
                3.20, 0.00, 0.31, //
                0.00, 3.10, 0.00, //
            ];
            let mut b2 = [1.1_f64, 2.1, 3.1];
            let mut ipiv2 = [0_usize; N];
            dgetrf(N, &mut a2, &mut ipiv2).expect("dgetrf a2");
            dgetrs(N, &a2, &ipiv2, &mut b2).expect("dgetrs b2");
            let want2: [u64; 3] =
                [0xbff15c9882b93105, 0x3ff0000000000000, 0x4031f9acb76f3867];
            for i in 0..N {
                assert_eq!(b2[i].to_bits(), want2[i]);
            }
        }

        // 3. `test_square_inv` (`mtx/test/src/test_square.f90:36-93`, 12 values).
        {
            const N: usize = 3;
            let a1_init = [
                3.14_f64, 7.50, 0.00, //
                4.10, 3.20, 0.30, //
                0.00, 1.00, 4.10, //
            ];
            let mut a1 = a1_init;
            let b1 = [1.0_f64, 2.0, 3.0];
            let mut x1 = b1;
            let mut ipiv1 = [0_usize; N];
            dgetrf(N, &mut a1, &mut ipiv1).expect("dgetrf inv1");
            dgetrs(N, &a1, &ipiv1, &mut x1).expect("dgetrs inv1");
            let a1_t = transpose_square(N, &a1_init);
            let mut prod1 = [0.0_f64; N];
            multiply_xa(N, &a1_t, &x1, &mut prod1).expect("multiply_xa inv1");
            let want_b1: [u64; 3] =
                [0x3ff0000000000000, 0x4000000000000000, 0x4008000000000000];
            let want_prod1: [u64; 3] =
                [0x3ff0000000000000, 0x4000000000000000, 0x4007ffffffffffff];
            for i in 0..N {
                assert_eq!(b1[i].to_bits(), want_b1[i]);
                assert_eq!(prod1[i].to_bits(), want_prod1[i]);
            }

            let a2_init = [
                0.00_f64, 3.10, 0.00, //
                4.70, 6.20, 0.00, //
                3.20, 0.00, 0.31, //
            ];
            let mut a2 = a2_init;
            let b2 = [1.1_f64, 2.1, 3.1];
            let mut x2 = b2;
            let mut ipiv2 = [0_usize; N];
            dgetrf(N, &mut a2, &mut ipiv2).expect("dgetrf inv2");
            dgetrs(N, &a2, &ipiv2, &mut x2).expect("dgetrs inv2");
            let a2_t = transpose_square(N, &a2_init);
            let mut prod2 = [0.0_f64; N];
            multiply_xa(N, &a2_t, &x2, &mut prod2).expect("multiply_xa inv2");
            let want_b2: [u64; 3] =
                [0x3ff199999999999a, 0x4000cccccccccccd, 0x4008cccccccccccd];
            for i in 0..N {
                assert_eq!(b2[i].to_bits(), want_b2[i]);
                assert_eq!(prod2[i].to_bits(), want_b2[i]);
            }
        }
    });
}

#[test]
fn dgbtrs_trans_solves_banded_transpose_with_pivoting() {
    const N: usize = 5;
    const KL: usize = 1;
    const KU: usize = 2;
    const LDAB: usize = 2 * KL + KU + 1;
    // Large subdiagonals at (1,0) and (2,1) force actual partial pivoting.
    let a_dense = [
        2.0, -1.0, 0.5, 0.0, 0.0, //
        10.0, 3.0, -2.0, 1.2, 0.0, //
        0.0, 12.0, 2.5, -0.5, 2.5, //
        0.0, 0.0, -4.0, 7.0, 1.0, //
        0.0, 0.0, 0.0, 2.0, 8.0, //
    ];
    let mut ab = vec![0.0_f64; LDAB * N];
    dense_to_band(N, N, &a_dense, KL, KU, &mut ab, LDAB)
        .expect("dense_to_band");
    let mut lu_band = ab.clone();
    let mut ipiv_band = vec![0_usize; N];
    dgbtrf(N, KL, KU, &mut lu_band, LDAB, &mut ipiv_band).expect("dgbtrf");
    assert_ne!(ipiv_band[0], 0, "test expects pivoting on column 0");
    assert_ne!(ipiv_band[1], 1, "test expects pivoting on column 1");

    let mut lu_dense = a_dense.to_vec();
    let mut ipiv_dense = vec![0_usize; N];
    dgetrf(N, &mut lu_dense, &mut ipiv_dense).expect("dgetrf");

    let rhs = [1.0, -2.5, 3.25, 0.5, -4.0];

    let mut x_fwd_a = rhs;
    let mut x_fwd_b = rhs;
    dgbtrs(N, KL, KU, &lu_band, LDAB, &ipiv_band, &mut x_fwd_a)
        .expect("dgbtrs");
    dgbtrs_trans(false, N, KL, KU, &lu_band, LDAB, &ipiv_band, &mut x_fwd_b)
        .expect("dgbtrs_trans(false)");
    for i in 0..N {
        assert_eq!(x_fwd_a[i].to_bits(), x_fwd_b[i].to_bits());
    }

    let mut x_trans_band = rhs;
    let mut x_trans_dense = rhs;
    dgbtrs_trans(
        true,
        N,
        KL,
        KU,
        &lu_band,
        LDAB,
        &ipiv_band,
        &mut x_trans_band,
    )
    .expect("dgbtrs_trans(true)");
    dgetrs_trans(true, N, &lu_dense, &ipiv_dense, &mut x_trans_dense)
        .expect("dgetrs_trans(true)");

    for i in 0..N {
        assert!(
            (x_trans_band[i] - x_trans_dense[i]).abs() < 1e-13,
            "transpose solve mismatch at {i}: {} vs {}",
            x_trans_band[i],
            x_trans_dense[i]
        );
        let mut at_x_i = 0.0;
        for row in 0..N {
            at_x_i += a_dense[row * N + i] * x_trans_band[row];
        }
        assert!(
            (at_x_i - rhs[i]).abs() < 1e-12,
            "A^T x residual mismatch at {i}: got {at_x_i}, want {}",
            rhs[i]
        );
    }

    // Trailing slack beyond `ldab * n` must not trigger `NonFiniteInput`.
    let mut lu_with_slack = lu_band.clone();
    lu_with_slack.push(f64::NAN);
    let mut x_slack = rhs;
    dgbtrs(N, KL, KU, &lu_with_slack, LDAB, &ipiv_band, &mut x_slack)
        .expect("slack past ldab * n should be ignored");

    // Out-of-bounds pivot must error cleanly without mutating.
    let mut bad_ipiv = ipiv_band;
    bad_ipiv[N - 1] = N;
    let mut x_bad = rhs;
    let err =
        dgbtrs_trans(true, N, KL, KU, &lu_band, LDAB, &bad_ipiv, &mut x_bad)
            .unwrap_err();
    assert!(matches!(err, MtxError::InvalidParameter(_)));
    assert_eq!(x_bad, rhs);
}

#[test]
fn rcond_banded_and_dense_hager_parity_and_validation() {
    // Upper bidiagonal 4x4 matrix where ||A||_1 = 11, ||A^-1||_1 = 1111, so
    // exact rcond_1 = 1 / 12221; single-shot [1/n; n] misses the worst column.
    const N: usize = 4;
    const KL: usize = 0;
    const KU: usize = 1;
    const LDAB: usize = 2 * KL + KU + 1;
    #[rustfmt::skip]
    let a_dense = [
        1.0, 10.0,  0.0,  0.0,
        0.0,  1.0, 10.0,  0.0,
        0.0,  0.0,  1.0, 10.0,
        0.0,  0.0,  0.0,  1.0,
    ];
    let mut ab = vec![0.0_f64; LDAB * N];
    dense_to_band(N, N, &a_dense, KL, KU, &mut ab, LDAB).unwrap();
    let mut lu_band = ab.clone();
    let mut ipiv_band = vec![0_usize; N];
    dgbtrf(N, KL, KU, &mut lu_band, LDAB, &mut ipiv_band).unwrap();

    let mut lu_dense = a_dense.to_vec();
    let mut ipiv_dense = vec![0_usize; N];
    dgetrf(N, &mut lu_dense, &mut ipiv_dense).unwrap();

    let rc_band =
        rcond_banded(N, KL, KU, &ab, &lu_band, LDAB, &ipiv_band).unwrap();
    let rc_dense = rcond_dense(N, &a_dense, &lu_dense, &ipiv_dense).unwrap();
    let exact_rcond = 1.0 / (11.0 * 1111.0);
    assert!(
        (rc_band - exact_rcond).abs() < 1e-14,
        "expected exact Hager rcond {exact_rcond}, got {rc_band}"
    );
    assert!((rc_band - rc_dense).abs() < 1e-15);

    // Non-finite input rejection rather than silent Ok(0.0).
    let mut ab_nan = ab.clone();
    ab_nan[LDAB * N - 1] = f64::NAN;
    assert!(matches!(
        rcond_banded(N, KL, KU, &ab_nan, &lu_band, LDAB, &ipiv_band),
        Err(MtxError::NonFiniteInput(_))
    ));
    let mut dense_nan = a_dense;
    dense_nan[0] = f64::NAN;
    assert!(matches!(
        rcond_dense(N, &dense_nan, &lu_dense, &ipiv_dense),
        Err(MtxError::NonFiniteInput(_))
    ));
}

#[test]
fn preconditioners_and_dgbsvx_wrappers_match_upstream() {
    // 4x4 left-aligned banded system with kl = 2 (> 1 exercises bounded DGBSVX_wrapper:86).
    const N: usize = 4;
    const KL: usize = 2;
    const KU: usize = 1;
    let s0 = 1e-18;
    let s1 = 1.0;
    let s2 = 1e18;
    let s3 = 1e-12;
    let row_scales = [s0, s1, s2, s3];
    // Original matrix rows before scaling:
    // row 0: [ 4.0, -1.0,  0.0,  0.0]
    // row 1: [-1.0,  5.0, -1.0,  0.0]
    // row 2: [ 0.5, -1.0,  6.0, -1.5]
    // row 3: [ 0.0,  0.4, -0.8,  4.5]
    #[rustfmt::skip]
    let bands = [
        // u1 (superdiag 1, cols 0..3): (0,1), (1,2), (2,3), pad
        -s0, -s1, -1.5 * s2, 0.0,
        // diag (cols 0..4)
         4.0 * s0,  5.0 * s1,  6.0 * s2, 4.5 * s3,
        // l1 (subdiag 1, cols 0..3): (1,0), (2,1), (3,2), pad
        -s1, -s2, -0.8 * s3, 0.0,
        // l2 (subdiag 2, cols 0..2): (2,0), (3,1), pad, pad
         0.5 * s2,  0.4 * s3, 0.0, 0.0,
    ];
    let mut pc = vec![0.0_f64; N];
    compute_band_preconditioner(N, KL, KU, &bands, &mut pc).unwrap();
    assert!((pc[0] - 4.0 * s0).abs() <= 1e-32);
    assert!((pc[1] - 5.0 * s1).abs() <= 1e-15);
    assert!((pc[2] - 6.0 * s2).abs() / (6.0 * s2) <= 1e-15);
    assert!((pc[3] - 4.5 * s3).abs() <= 1e-27);

    let x_true = [1.25, -0.75, 2.5, -1.5];
    #[rustfmt::skip]
    let a_full = [
         4.0 * s0, -s0,       0.0,       0.0,
        -s1,        5.0 * s1, -s1,       0.0,
         0.5 * s2, -s2,       6.0 * s2, -1.5 * s2,
         0.0,       0.4 * s3, -0.8 * s3,  4.5 * s3,
    ];
    let mut rhs = [0.0_f64; N];
    for r in 0..N {
        rhs[r] = (0..N).map(|c| a_full[r * N + c] * x_true[c]).sum::<f64>();
    }

    let sol_bands =
        dgbsvx_banded_from_bands(N, KL, KU, &bands, &rhs).expect("from_bands");
    for i in 0..N {
        assert!((sol_bands.x[i] - x_true[i]).abs() < 1e-12);
    }
    assert!(sol_bands.rcond > 0.05);

    let ldab: usize = 2 * KL + KU + 1;
    let mut ab = vec![0.0_f64; ldab * N];
    dense_to_band(N, N, &a_full, KL, KU, &mut ab, ldab).unwrap();
    let sol_ab = dgbsvx_banded(N, KL, KU, &ab, ldab, &rhs).expect("dgbsvx_ab");
    for i in 0..N {
        assert!((sol_ab.x[i] - x_true[i]).abs() < 1e-12);
        assert_eq!(
            sol_ab.row_scale[i],
            row_scales[i] * [4.0, 5.0, 6.0, 4.5][i]
        );
    }

    // Block-tridiagonal preconditioner & two-sided `dgbsvx_block_tridiagonal`.
    let nvar = 2_usize;
    let nz = 3_usize;
    let blk_sz = nvar * nvar;
    let lblk = vec![
        0.0, 0.0, 0.0, 0.0, // row 0 unused
        0.5e-10, -0.2, 0.1e10, 0.4, // row 1
        -0.3, 0.2e-10, 0.4e10, -0.5, // row 2
    ];
    let dblk = vec![
        4.0e-10, 1.0, -0.5e10, 5.0, // row 0
        3.5e-10, -0.4, 0.3e10, 6.0, // row 1
        5.0e-10, 0.7, -0.2e10, 4.5, // row 2
    ];
    let ublk = vec![
        0.2e-10, 0.3, 0.1e10, -0.2, // row 0
        -0.1e-10, 0.6, -0.2e10, 0.5, // row 1
        0.0, 0.0, 0.0, 0.0, // row 2 unused
    ];
    let mut right_pc = vec![0.0_f64; nz * nvar];
    compute_block_preconditioner(nvar, nz, &lblk, &dblk, &ublk, &mut right_pc)
        .unwrap();
    let mut left_pc = vec![0.0_f64; nz * nvar];
    compute_block_left_preconditioner(
        nvar,
        nz,
        &lblk,
        &dblk,
        &ublk,
        &right_pc,
        &mut left_pc,
    )
    .unwrap();
    assert!(right_pc.iter().all(|&v| v > 0.0));
    assert!(left_pc.iter().all(|&v| v > 0.0));
    assert_eq!(lblk.len(), nz * blk_sz);

    let x_blk_true = [1.0, -2.0, 0.5, 1.5, -0.75, 2.25];
    let mut b_blk = vec![0.0_f64; nz * nvar];
    block_dble_mv(nvar, nz, &lblk, &dblk, &ublk, &x_blk_true, &mut b_blk)
        .unwrap();

    let sol_blk =
        dgbsvx_block_tridiagonal(nvar, nz, &lblk, &dblk, &ublk, &b_blk)
            .expect("dgbsvx_block_tridiagonal");
    for i in 0..(nz * nvar) {
        assert!(
            (sol_blk.x[i] - x_blk_true[i]).abs() < 1e-10,
            "block pc solve mismatch at {i}: {} vs {}",
            sol_blk.x[i],
            x_blk_true[i]
        );
    }
    assert!(sol_blk.rcond > 0.01);
}

#[test]
fn equilibration_dgeequ_dgbequ_and_extreme_scale_solves() {
    // Exact powers of two (`2^80` and `2^-80`) create extreme ~10^48 dynamic
    // range across rows and columns while keeping exact IEEE-754 representation.
    let big = f64::from_bits(0x44f0000000000000); // 2^80
    let small = f64::from_bits(0x3af0000000000000); // 2^-80
    let inv_small = big;
    #[rustfmt::skip]
    let a = [
        4.0 * big, -big,        2.0 * big * inv_small,
        small,      5.0 * small, -1.0,
        0.5,       -2.0,        6.0 * inv_small,
    ];
    let x_true = [1.5_f64, -2.25, 0.75 * small];
    let b: Vec<f64> = (0..3)
        .map(|i| (0..3).map(|j| a[i * 3 + j] * x_true[j]).sum())
        .collect();

    let eq = dgeequ(3, &a).expect("dgeequ");
    assert_eq!(eq.equed, Equilibration::Both);
    let sol =
        solve_dense_equilibrated(3, &a, &b).expect("solve_dense_equilibrated");
    for i in 0..3 {
        let tol = 1e-12 * x_true[i].abs().max(1e-30);
        assert!(
            (sol.x[i] - x_true[i]).abs() <= tol,
            "equilibrated dense mismatch at {i}: got {}, want {}",
            sol.x[i],
            x_true[i]
        );
    }
    assert!(sol.rcond > 0.05);

    // Banded tridiagonal equilibration (kl = 1, ku = 1), balanced RHS scales.
    #[rustfmt::skip]
    let a_band = [
        4.0 * big, -big,        0.0,
        small,      5.0 * small, -small * small,
        0.0,       -2.0,        6.0 * small,
    ];
    let x_band_true = [1.5_f64, -2.25, 0.75 * big];
    let b_band: Vec<f64> = (0..3)
        .map(|i| (0..3).map(|j| a_band[i * 3 + j] * x_band_true[j]).sum())
        .collect();
    let ldab = 4_usize;
    let mut ab = vec![0.0_f64; ldab * 3];
    dense_to_band(3, 3, &a_band, 1, 1, &mut ab, ldab).unwrap();
    let eq_b = dgbequ(3, 1, 1, &ab, ldab).expect("dgbequ");
    assert_eq!(eq_b.equed, Equilibration::Both);
    let sol_b = solve_banded_equilibrated(3, 1, 1, &ab, ldab, &b_band)
        .expect("solve_banded_equilibrated");
    for i in 0..3 {
        let tol = 1e-12 * x_band_true[i].abs().max(1e-30);
        assert!(
            (sol_b.x[i] - x_band_true[i]).abs() <= tol,
            "equilibrated band mismatch at {i}: got {}, want {}",
            sol_b.x[i],
            x_band_true[i]
        );
    }

    // Zero row returns Equilibration::None and SingularMatrix during solve.
    let a_zero_row = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0];
    let eq_zero = dgeequ(3, &a_zero_row).unwrap();
    assert_eq!(eq_zero.equed, Equilibration::None);
    assert!(matches!(
        solve_dense_equilibrated(3, &a_zero_row, &[1.0, 0.0, 2.0]),
        Err(MtxError::SingularMatrix(_))
    ));
}

#[test]
fn bcyclic_factor_and_solve_zero_ulp_vs_oneshot_and_multi_rhs() {
    let nvar = 3_usize;
    let blk_sz = nvar * nvar;
    for &nz in &[1_usize, 2, 5, 7, 16, 35] {
        let mut lblk = vec![0.0_f64; nz * blk_sz];
        let mut dblk = vec![0.0_f64; nz * blk_sz];
        let mut ublk = vec![0.0_f64; nz * blk_sz];
        let mut rhs1 = vec![0.0_f64; nz * nvar];
        let mut rhs2 = vec![0.0_f64; nz * nvar];

        for k in 0..nz {
            let kf = (k + 1) as f64;
            for r in 0..nvar {
                rhs1[k * nvar + r] =
                    ((k * 3 + r + 1) as f64) * 0.37 - 0.11 * kf;
                rhs2[k * nvar + r] =
                    ((nz - k + r) as f64) * -0.23 + 0.5 * ((r + 1) as f64);
                for c in 0..nvar {
                    let off = k * blk_sz + r * nvar + c;
                    let diag_shift = if r == c { 8.0 + 0.1 * kf } else { 0.0 };
                    dblk[off] = diag_shift + 0.25 * ((r + 1) as f64)
                        - 0.15 * ((c + 1) as f64);
                    if k > 0 {
                        lblk[off] = if r == c {
                            -0.8
                        } else {
                            0.05 * ((r + c) as f64)
                        };
                    }
                    if k + 1 < nz {
                        ublk[off] = if r == c {
                            -0.6
                        } else {
                            -0.04 * ((r + 1) as f64)
                        };
                    }
                }
            }
        }

        for &parallel in &[false, true] {
            let factor =
                bcyclic_factor(nvar, nz, &lblk, &dblk, &ublk, parallel)
                    .expect("bcyclic_factor");
            assert_eq!(factor.nvar(), nvar);
            assert_eq!(factor.nz(), nz);

            let mut x1 = rhs1.clone();
            bcyclic_solve(&factor, &mut x1, parallel).expect("bcyclic_solve 1");
            let ref1 = solve_block_tridiag_bcyclic(
                nvar, nz, &lblk, &dblk, &ublk, &rhs1, parallel,
            )
            .expect("oneshot 1");
            let x2 = factor.solve(&rhs2, parallel).expect("factor.solve 2");
            let ref2 = solve_block_tridiag_bcyclic(
                nvar, nz, &lblk, &dblk, &ublk, &rhs2, parallel,
            )
            .expect("oneshot 2");

            for i in 0..(nz * nvar) {
                assert_eq!(
                    x1[i].to_bits(),
                    ref1[i].to_bits(),
                    "nz={nz} parallel={parallel} rhs1 0-ULP mismatch at {i}"
                );
                assert_eq!(
                    x2[i].to_bits(),
                    ref2[i].to_bits(),
                    "nz={nz} parallel={parallel} rhs2 0-ULP mismatch at {i}"
                );
            }

            let mut mv1 = vec![0.0_f64; nz * nvar];
            block_dble_mv(nvar, nz, &lblk, &dblk, &ublk, &x1, &mut mv1)
                .unwrap();
            for i in 0..(nz * nvar) {
                assert!((mv1[i] - rhs1[i]).abs() < 1e-12);
            }
        }
    }
}

fn verify_eigen_residuals(n: usize, a: &[f64], res: &EigenResult, norm_a: f64) {
    let mut j = 0usize;
    while j < n {
        if res.wi[j] == 0.0 {
            let wr = res.wr[j];
            let mut norm2 = 0.0_f64;
            for i in 0..n {
                let v = res.vr[i * n + j];
                assert!(v.is_finite(), "non-finite real eigvec component");
                norm2 += v * v;
            }
            assert!((norm2.sqrt() - 1.0).abs() < 1e-12);
            for i in 0..n {
                let mut av = 0.0_f64;
                for c in 0..n {
                    av += a[i * n + c] * res.vr[c * n + j];
                }
                let err = (av - wr * res.vr[i * n + j]).abs();
                assert!(
                    err <= 1e-10 * norm_a,
                    "real residual {err} exceeds scale {norm_a}"
                );
            }
            j += 1;
        } else {
            let wr = res.wr[j];
            let wi = res.wi[j];
            let mut norm2 = 0.0_f64;
            for i in 0..n {
                let vr_i = res.vr[i * n + j];
                let vi_i = res.vr[i * n + j + 1];
                assert!(vr_i.is_finite() && vi_i.is_finite());
                norm2 += vr_i * vr_i + vi_i * vi_i;
            }
            assert!((norm2.sqrt() - 1.0).abs() < 1e-12);
            for i in 0..n {
                let (mut avr, mut avi) = (0.0_f64, 0.0_f64);
                for c in 0..n {
                    avr += a[i * n + c] * res.vr[c * n + j];
                    avi += a[i * n + c] * res.vr[c * n + j + 1];
                }
                let target_r =
                    wr * res.vr[i * n + j] - wi * res.vr[i * n + j + 1];
                let target_i =
                    wr * res.vr[i * n + j + 1] + wi * res.vr[i * n + j];
                assert!((avr - target_r).abs() <= 1e-10 * norm_a);
                assert!((avi - target_i).abs() <= 1e-10 * norm_a);
            }
            j += 2;
        }
    }
}

#[test]
fn edge_cases_io_sparse_and_extreme_eigen() {
    // 1. Malformed Harwell-Boeing block with extra token on colptr line.
    let hb_ok = write_block_tridiagonal_text(
        2,
        1,
        &[0.0; 4],
        &[1.0, 0.0, 0.0, 2.0],
        &[0.0; 4],
    )
    .expect("write hb");
    let mut lines: Vec<String> = hb_ok.lines().map(ToOwned::to_owned).collect();
    // Line 0 is "<nvar> <nz>", lines 1..=4 are HB block header, line 5 is colptr.
    lines[5].push_str(" 99");
    let hb_bad = lines.join("\n");
    assert!(matches!(
        parse_block_tridiagonal_text(&hb_bad),
        Err(MtxError::ParseError(_))
    ));

    // 2. `find_loc_in_sparse` with `hint = Some(usize::MAX)` avoids overflow.
    let ia = [1_usize, 3, 4];
    let ja = [1_usize, 2, 2];
    let loc = find_loc_in_sparse(
        SparseFormat::CompressedRowSparse,
        2,
        3,
        &ia,
        &ja,
        1,
        2,
        Some(usize::MAX),
    )
    .expect("usize::MAX hint");
    assert_eq!(loc, Some(2));

    // 3. `check_blocks` multiplication overflow returns clean error.
    let empty: [f64; 0] = [];
    assert!(matches!(
        solve_block_tridiagonal(usize::MAX, 2, &empty, &empty, &empty, &empty),
        Err(MtxError::InvalidParameter(_))
    ));

    // 4. Extreme-scale complex & real eigenvalues/eigenvectors (`> 1e150`, `< 1e-150`).
    let big = 1.0e200_f64;
    #[rustfmt::skip]
    let a_big = [
         1.0 * big, -2.0 * big,  0.3 * big,  0.0,
         2.0 * big,  1.0 * big,  0.4 * big, -0.2 * big,
         0.0,        0.1 * big, -3.0 * big,  0.5 * big,
         0.1 * big,  0.0,       -0.5 * big, -3.0 * big,
    ];
    let res_big = dgeev_right(4, &a_big).expect("dgeev_right big");
    verify_eigen_residuals(4, &a_big, &res_big, 4.0 * big);

    let tiny = 1.0e-200_f64;
    let a_tiny: Vec<f64> = a_big.iter().map(|&v| (v / big) * tiny).collect();
    let res_tiny = dgeev_right(4, &a_tiny).expect("dgeev_right tiny");
    verify_eigen_residuals(4, &a_tiny, &res_tiny, 4.0 * tiny);

    // 5. Large off-diagonal block coupling (`ca` balance_matrix guard).
    #[rustfmt::skip]
    let a_isolated = [
        2.0, 1.0e290, 0.0, 0.0,
        0.0, 1.0,     1.0e20, 0.0,
        0.0, 1.0e-20, 3.0,    1.0,
        0.0, 0.0,     1.0,    4.0,
    ];
    let res_iso = dgeev_right(4, &a_isolated).expect("dgeev_right isolated");
    assert!(
        res_iso.wr.iter().all(|v| v.is_finite())
            && res_iso.wi.iter().all(|v| v.is_finite())
    );
    assert!(res_iso.vr.iter().all(|v| v.is_finite()));
    for j in 0..4 {
        if res_iso.wi[j] == 0.0 {
            let norm2: f64 = (0..4)
                .map(|i| res_iso.vr[i * 4 + j] * res_iso.vr[i * 4 + j])
                .sum();
            assert!((norm2.sqrt() - 1.0).abs() < 1e-12);
        }
    }

    // 6. Highly non-normal 23x23 upper-triangular chain where unscaled
    // eigenvector back-substitution would exceed f64::MAX (`1e308`).
    let nt: usize = 23;
    let mut a_tri = vec![0.0_f64; nt * nt];
    for i in 0..nt {
        a_tri[i * nt + i] = 1.0 + 0.5 * (i as f64);
        if i + 1 < nt {
            a_tri[i * nt + (i + 1)] = 1.0e15;
        }
    }
    let res_tri = dgeev_right(nt, &a_tri).expect("dgeev_right non-normal");
    verify_eigen_residuals(nt, &a_tri, &res_tri, 1.0e15);
}
