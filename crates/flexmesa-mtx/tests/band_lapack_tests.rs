//! Hand-checkable tests for the Reference LAPACK 3.12.0 band condition
//! routines in `flexmesa_mtx::band_lapack` (`#676`, part 1): `dlangb`,
//! `dgbcon`, `dlatbs`, `drscl` and `dtbsv`.
//!
//! # Oracles
//!
//! - Small band matrices whose norms, triangular solves and condition
//!   estimates follow from exact binary arithmetic by hand (comments give
//!   the derivation). These compare at 0 ULP.
//! - A dense Gauss-Jordan inverse for the 1-norm and infinity-norm of
//!   `inv(A)`: `DLACN2` returns a lower bound on `norm(inv(A))`, so the
//!   `DGBCON` estimate is bracketed by the exact value and `n` times it.
//!
//! No MESA-generated oracle is used here; those arrive with part 2.

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_lines
)]

use flexmesa_mtx::band_lapack::dgbtrf::{
    dgbtrf_block_size, dgbtrf_uses_blocked_code,
};
use flexmesa_mtx::band_lapack::{
    dgbcon, dgbmv, dgbrfs, dgbsvx_fact_n, dlangb, dlantb, dlatbs, drscl, dtbsv,
    Refinement,
};
use flexmesa_mtx::lapack::{dlamch, ix, LapackError};
use flexmesa_mtx::{
    dense_to_band, dgbsvx_banded_from_bands, dgbsvx_block_tridiagonal, dgbtrf,
    dgbtrs_trans, MtxError,
};

/// Column-major `AB(ldab, n)` from `(i, j, value)` triples; unlisted
/// entries hold `fill`.
fn band(
    ldab: usize,
    n: usize,
    fill: f64,
    e: &[(usize, usize, f64)],
) -> Vec<f64> {
    let mut ab = vec![fill; ldab * n];
    for &(i, j, v) in e {
        ab[ix(i, j, ldab)] = v;
    }
    ab
}

// ---------------------------------------------------------------- dlangb

/// `A = [[1,-2,0],[3,4,-5],[0,6,7]]`, `kl = ku = 1`, stored as
/// `AB(ku+1+i-j, j)` with `1e300` in the two unused corners.
fn dlangb_matrix() -> Vec<f64> {
    band(
        3,
        3,
        1.0e300,
        &[
            (2, 1, 1.0),
            (3, 1, 3.0),
            (1, 2, -2.0),
            (2, 2, 4.0),
            (3, 2, 6.0),
            (1, 3, -5.0),
            (2, 3, 7.0),
        ],
    )
}

#[test]
fn dlangb_tridiagonal_norms_match_hand_values() {
    let ab = dlangb_matrix();
    let mut work = vec![0.0; 3];
    // max |a_ij| = 7
    assert_eq!(dlangb(b'M', 3, 1, 1, &ab, 3, &mut work).unwrap(), 7.0);
    // column sums 1+3, 2+4+6, 5+7 -> 12
    assert_eq!(dlangb(b'1', 3, 1, 1, &ab, 3, &mut work).unwrap(), 12.0);
    assert_eq!(dlangb(b'O', 3, 1, 1, &ab, 3, &mut work).unwrap(), 12.0);
    // row sums 1+2, 3+4+5, 6+7 -> 13
    assert_eq!(dlangb(b'I', 3, 1, 1, &ab, 3, &mut work).unwrap(), 13.0);
}

#[test]
fn dlangb_asymmetric_bandwidth_reads_only_the_band() {
    // n = 4, kl = 2, ku = 0 (lower triangular): ldab = 3,
    // A(i,j) at AB(1+i-j, j). Rows of A: [1], [2,3], [4,5,6], [0,7,8,9].
    let ab = band(
        3,
        4,
        1.0e300,
        &[
            (1, 1, 1.0),
            (2, 1, 2.0),
            (3, 1, 4.0),
            (1, 2, 3.0),
            (2, 2, 5.0),
            (3, 2, 7.0),
            (1, 3, 6.0),
            (2, 3, 8.0),
            (1, 4, 9.0),
        ],
    );
    let mut work = vec![0.0; 4];
    assert_eq!(dlangb(b'M', 4, 2, 0, &ab, 3, &mut work).unwrap(), 9.0);
    // column sums 1+2+4, 3+5+7, 6+8, 9 -> 15
    assert_eq!(dlangb(b'1', 4, 2, 0, &ab, 3, &mut work).unwrap(), 15.0);
    // row sums 1, 5, 15, 24 -> 24
    assert_eq!(dlangb(b'I', 4, 2, 0, &ab, 3, &mut work).unwrap(), 24.0);
}

#[test]
fn dlangb_empty_and_nan_and_unsupported_norms() {
    let mut work = vec![0.0; 3];
    assert_eq!(dlangb(b'1', 0, 1, 1, &[], 3, &mut work).unwrap(), 0.0);
    let mut ab = dlangb_matrix();
    ab[ix(2, 2, 3)] = f64::NAN;
    assert!(dlangb(b'M', 3, 1, 1, &ab, 3, &mut work).unwrap().is_nan());
    assert!(dlangb(b'1', 3, 1, 1, &ab, 3, &mut work).unwrap().is_nan());
    assert!(dlangb(b'I', 3, 1, 1, &ab, 3, &mut work).unwrap().is_nan());
    let ab = dlangb_matrix();
    assert_eq!(
        dlangb(b'F', 3, 1, 1, &ab, 3, &mut work),
        Err(LapackError::UnportedPath {
            routine: "DLANGB",
            lines: "203-214"
        })
    );
    assert_eq!(
        dlangb(b'Z', 3, 1, 1, &ab, 3, &mut work),
        Err(LapackError::IllegalArgument {
            routine: "DLANGB",
            info: -1
        })
    );
}

// ----------------------------------------------------------------- drscl

#[test]
fn drscl_divides_by_a_representable_scalar_exactly() {
    // MUL = CNUM/CDEN = 1/2 on the first pass.
    let mut x = [2.0, 4.0, -6.0];
    drscl(3, 2.0, &mut x, 1);
    assert_eq!(x, [1.0, 2.0, -3.0]);
    // Stride 2 touches only every second element.
    let mut y = [8.0, 99.0, 16.0];
    drscl(2, 4.0, &mut y, 2);
    assert_eq!(y, [2.0, 99.0, 4.0]);
    // n = 0 returns without reading.
    drscl(0, 0.0, &mut [], 1);
}

#[test]
fn drscl_huge_and_tiny_scalars_take_the_two_stage_path() {
    // sa = 2^1000: CDEN1 = sa*smlnum = 2^-22 < 1, so the first pass is the
    // plain MUL = 1/sa = 2^-1000 (exact).
    let sa = f64::from_bits(((1000 + 1023) as u64) << 52);
    let mut x = [f64::from_bits(((900 + 1023) as u64) << 52)];
    drscl(1, sa, &mut x, 1);
    assert_eq!(x[0], f64::from_bits(((-100_i64 + 1023) as u64) << 52));
    // sa = 2^-1060 (subnormal): 1/sa is not finite, so MUL = BIGNUM = 2^1022
    // first (cnum1 > cden), then MUL = CNUM/CDEN = 2^-1022/2^-1060 = 2^38.
    // Net x * 2^1060: x = 2^-1000 gives 2^60.
    let sa = f64::from_bits(1_u64 << (1074 - 1060));
    let mut x = [f64::from_bits(((-1000_i64 + 1023) as u64) << 52)];
    drscl(1, sa, &mut x, 1);
    assert_eq!(x[0], f64::from_bits(((60 + 1023) as u64) << 52));
    // sa = 2^1023: CDEN1 = 2^1023 * 2^-1022 = 2 > 1, so MUL = SMLNUM first,
    // then MUL = 1/2: x = 4 becomes 4 * 2^-1023 = 2^-1021 exactly.
    let sa = f64::from_bits(2046_u64 << 52);
    let mut x = [4.0];
    drscl(1, sa, &mut x, 1);
    assert_eq!(x[0], f64::from_bits(2_u64 << 52));
}

// ----------------------------------------------------------------- dtbsv

/// Upper `A = [[2,1,0],[0,4,1],[0,0,5]]`, `k = 1`, `AB(k+1+i-j,j)`.
fn upper3() -> Vec<f64> {
    band(
        2,
        3,
        1.0e300,
        &[
            (2, 1, 2.0),
            (1, 2, 1.0),
            (2, 2, 4.0),
            (1, 3, 1.0),
            (2, 3, 5.0),
        ],
    )
}

/// Lower `L = upper3()^T`, `AB(1+i-j,j)`.
fn lower3() -> Vec<f64> {
    band(
        2,
        3,
        1.0e300,
        &[
            (1, 1, 2.0),
            (2, 1, 1.0),
            (1, 2, 4.0),
            (2, 2, 1.0),
            (1, 3, 5.0),
        ],
    )
}

fn strided(v: &[f64], inc: usize) -> Vec<f64> {
    let mut s = vec![-7.0; (v.len() - 1) * inc + 1];
    for (i, &e) in v.iter().enumerate() {
        s[i * inc] = e;
    }
    s
}

#[test]
fn dtbsv_solves_all_four_triangular_cases_exactly() {
    // x = [1,2,3]. Upper A*x = [4,11,15]; A^T*x = [2,9,17].
    let cases: [(u8, u8, Vec<f64>, [f64; 3]); 4] = [
        (b'U', b'N', upper3(), [4.0, 11.0, 15.0]),
        (b'U', b'T', upper3(), [2.0, 9.0, 17.0]),
        (b'L', b'N', lower3(), [2.0, 9.0, 17.0]),
        (b'L', b'T', lower3(), [4.0, 11.0, 15.0]),
    ];
    for (uplo, trans, a, rhs) in cases {
        for inc in [1_usize, 2, 3] {
            let mut x = strided(&rhs, inc);
            dtbsv(uplo, trans, b'N', 3, 1, &a, 2, &mut x, inc).unwrap();
            for i in 0..3 {
                assert_eq!(
                    x[i * inc],
                    (i + 1) as f64,
                    "uplo={} trans={} inc={inc}",
                    uplo as char,
                    trans as char
                );
            }
            if inc > 1 {
                assert_eq!(x[1], -7.0, "gap element untouched");
            }
        }
    }
}

#[test]
fn dtbsv_unit_diagonal_ignores_stored_diagonal_and_skips_zero_x() {
    // Unit upper, stored diagonal ignored: x3 = 15, x2 = 11 - 15 = -4,
    // x1 = 4 - (-4) = 8.
    let mut x = [4.0, 11.0, 15.0];
    dtbsv(b'U', b'N', b'U', 3, 1, &upper3(), 2, &mut x, 1).unwrap();
    assert_eq!(x, [8.0, -4.0, 15.0]);
    // x(j) == 0 skips the update (dtbsv.f:276): A has a NaN super-diagonal
    // that must not be touched when x3 == 0 and x2 == 0 after division.
    let mut a = upper3();
    a[ix(1, 3, 2)] = f64::NAN;
    a[ix(1, 2, 2)] = f64::NAN;
    let mut x = [0.0, 0.0, 0.0];
    dtbsv(b'U', b'N', b'N', 3, 1, &a, 2, &mut x, 1).unwrap();
    assert_eq!(x, [0.0, 0.0, 0.0]);
}

#[test]
fn dtbsv_rejects_bad_arguments_with_upstream_info() {
    let a = upper3();
    let mut x = [1.0, 2.0, 3.0];
    let e = |info| {
        Err(LapackError::IllegalArgument {
            routine: "DTBSV",
            info,
        })
    };
    assert_eq!(dtbsv(b'X', b'N', b'N', 3, 1, &a, 2, &mut x, 1), e(1));
    assert_eq!(dtbsv(b'U', b'X', b'N', 3, 1, &a, 2, &mut x, 1), e(2));
    assert_eq!(dtbsv(b'U', b'N', b'X', 3, 1, &a, 2, &mut x, 1), e(3));
    assert_eq!(dtbsv(b'U', b'N', b'N', 3, 1, &a, 1, &mut x, 1), e(7));
    assert_eq!(dtbsv(b'U', b'N', b'N', 3, 1, &a, 2, &mut x, 0), e(9));
    assert_eq!(dtbsv(b'U', b'N', b'N', 0, 1, &a, 2, &mut x, 1), Ok(()));
}

// ---------------------------------------------------------------- dlatbs

#[test]
fn dlatbs_uses_dtbsv_and_reports_unit_scale_when_growth_is_small() {
    let cases: [(u8, u8, Vec<f64>, [f64; 3]); 4] = [
        (b'U', b'N', upper3(), [4.0, 11.0, 15.0]),
        (b'U', b'T', upper3(), [2.0, 9.0, 17.0]),
        (b'L', b'N', lower3(), [2.0, 9.0, 17.0]),
        (b'L', b'T', lower3(), [4.0, 11.0, 15.0]),
    ];
    for (uplo, trans, a, rhs) in cases {
        let mut x = rhs.to_vec();
        let mut cnorm = [0.0; 3];
        let scale =
            dlatbs(uplo, trans, b'N', b'N', 3, 1, &a, 2, &mut x, &mut cnorm)
                .unwrap();
        assert_eq!(scale, 1.0);
        assert_eq!(
            x,
            [1.0, 2.0, 3.0],
            "uplo={} trans={}",
            uplo as char,
            trans as char
        );
        // Off-diagonal column 1-norms: upper [0,1,1], lower [1,1,0].
        let want = if uplo == b'U' {
            [0.0, 1.0, 1.0]
        } else {
            [1.0, 1.0, 0.0]
        };
        assert_eq!(cnorm, want);
    }
}

#[test]
fn dlatbs_normin_y_reuses_cnorm_without_reading_the_matrix() {
    // With NORMIN = 'Y' the supplied CNORM is used as is: poison the
    // off-diagonals' effect by passing a CNORM that differs from the true
    // one; the solve itself (via DTBSV) is unchanged, CNORM is returned
    // untouched.
    let mut x = [4.0, 11.0, 15.0];
    let mut cnorm = [0.5, 0.25, 0.125];
    let scale = dlatbs(
        b'U',
        b'N',
        b'N',
        b'Y',
        3,
        1,
        &upper3(),
        2,
        &mut x,
        &mut cnorm,
    )
    .unwrap();
    assert_eq!(scale, 1.0);
    assert_eq!(x, [1.0, 2.0, 3.0]);
    assert_eq!(cnorm, [0.5, 0.25, 0.125]);
}

#[test]
fn dlatbs_tiny_diagonal_takes_the_scaled_path_and_keeps_a_x_equal_scale_b() {
    // A = diag(d, d) with d = 1e-300 < SMLNUM (about 1e-292), kd = 0.
    // GROW is driven to 0 (dlatbs.f:408-417), so the Level 1 solve with
    // scaling runs (dlatbs.f:520-795). The invariant A*x = scale*b holds
    // to rounding of one division and one multiplication per element.
    let d = 1.0e-300_f64;
    let smlnum = dlamch::SFMIN / dlamch::PREC;
    assert!(d < smlnum);
    for (uplo, trans) in
        [(b'U', b'N'), (b'U', b'T'), (b'L', b'N'), (b'L', b'T')]
    {
        let ab = band(1, 2, 0.0, &[(1, 1, d), (1, 2, d)]);
        let b = [1.0, 1.0];
        let mut x = b;
        let mut cnorm = [0.0; 2];
        let scale =
            dlatbs(uplo, trans, b'N', b'N', 2, 0, &ab, 1, &mut x, &mut cnorm)
                .unwrap();
        assert!(scale > 0.0 && scale < 1.0, "scale = {scale}");
        for i in 0..2 {
            let ax = d * x[i];
            let sb = scale * b[i];
            assert!(
                (ax - sb).abs() <= 4.0 * f64::EPSILON * sb.abs(),
                "uplo={} trans={}: A x = {ax}, scale b = {sb}",
                uplo as char,
                trans as char
            );
        }
    }
}

#[test]
fn dlatbs_exactly_singular_diagonal_returns_scale_zero_and_null_vector() {
    // A = diag(1, 0): GROW becomes 0 at the zero pivot, so the scaled path
    // runs and finds A(2,2) = 0: x = e2, scale = 0 (dlatbs.f:591-603).
    let ab = band(1, 2, 0.0, &[(1, 1, 1.0), (1, 2, 0.0)]);
    let mut x = [1.0, 1.0];
    let mut cnorm = [0.0; 2];
    let scale =
        dlatbs(b'U', b'N', b'N', b'N', 2, 0, &ab, 1, &mut x, &mut cnorm)
            .unwrap();
    assert_eq!(scale, 0.0);
    assert_eq!(x, [0.0, 1.0]);
}

#[test]
fn dlatbs_rejects_bad_arguments_with_upstream_info() {
    let a = upper3();
    let mut x = [1.0, 2.0, 3.0];
    let mut c = [0.0; 3];
    let e = |info| {
        Err(LapackError::IllegalArgument {
            routine: "DLATBS",
            info,
        })
    };
    assert_eq!(
        dlatbs(b'X', b'N', b'N', b'N', 3, 1, &a, 2, &mut x, &mut c),
        e(-1)
    );
    assert_eq!(
        dlatbs(b'U', b'X', b'N', b'N', 3, 1, &a, 2, &mut x, &mut c),
        e(-2)
    );
    assert_eq!(
        dlatbs(b'U', b'N', b'X', b'N', 3, 1, &a, 2, &mut x, &mut c),
        e(-3)
    );
    assert_eq!(
        dlatbs(b'U', b'N', b'N', b'X', 3, 1, &a, 2, &mut x, &mut c),
        e(-4)
    );
    assert_eq!(
        dlatbs(b'U', b'N', b'N', b'N', 3, 1, &a, 1, &mut x, &mut c),
        e(-8)
    );
    assert_eq!(
        dlatbs(b'U', b'N', b'N', b'N', 0, 1, &a, 2, &mut x, &mut c),
        Ok(1.0)
    );
}

// ---------------------------------------------------------------- dgbcon

#[test]
fn dgbcon_quick_returns_and_argument_errors() {
    let ab = [1.0];
    let e = |info| {
        Err(LapackError::IllegalArgument {
            routine: "DGBCON",
            info,
        })
    };
    assert_eq!(dgbcon(b'1', 0, 0, 0, &ab, 1, &[], 1.0), Ok(1.0));
    assert_eq!(dgbcon(b'1', 1, 0, 0, &ab, 1, &[1], 0.0), Ok(0.0));
    assert_eq!(dgbcon(b'F', 1, 0, 0, &ab, 1, &[1], 1.0), e(-1));
    assert_eq!(dgbcon(b'1', 1, 1, 0, &ab, 2, &[1], 1.0), e(-6));
    assert_eq!(dgbcon(b'1', 1, 0, 0, &ab, 1, &[1], -1.0), e(-8));
}

#[test]
fn dgbcon_one_by_one_is_exactly_one() {
    // n = 1: DLACN2 takes the N == 1 exit with est = |1/a| = 1/4;
    // rcond = (1/est)/anorm = 4/4 = 1.
    for &norm in b"1I" {
        assert_eq!(dgbcon(norm, 1, 0, 0, &[4.0], 1, &[1], 4.0), Ok(1.0));
    }
}

#[test]
fn dgbcon_diagonal_matrix_gives_exact_quarter() {
    // A = diag(2,4,8), kl = ku = 0, ldab = 1, ipiv = identity.
    // inv(A) has 1-norm 1/2. DLACN2: x = 1/3 each; est = (1/3)(1/2+1/4+1/8);
    // the sign vector is all +1, x = e1 after the transpose solve
    // (idamax = 1), est = 1/2, the sign vector repeats, so the final-stage
    // alternating vector gives temp = 2*(9/8)/9 = 1/4 < 1/2 (dlacn2.f:196-200).
    // rcond = (1/(1/2)) / 8 = 1/4 exactly.
    let ab = [2.0, 4.0, 8.0];
    for &norm in b"1I" {
        assert_eq!(dgbcon(norm, 3, 0, 0, &ab, 1, &[1, 2, 3], 8.0), Ok(0.25));
    }
}

#[test]
fn dgbcon_pivoted_2x2_matches_hand_estimate_in_both_norms() {
    // A = [[1,2],[4,6]], kl = ku = 1. Partial pivoting swaps the rows:
    // P A = [[4,6],[1,2]] = L U with L21 = 1/4, U = [[4,6],[0,1/2]],
    // ipiv = [2,2]. Factors in AB(LDAB=4, 2) with the diagonal in row
    // kl+ku+1 = 3 and the multiplier in row kd+1 = 4.
    let ab = band(
        4,
        2,
        0.0,
        &[(3, 1, 4.0), (4, 1, 0.25), (2, 2, 6.0), (3, 2, 0.5)],
    );
    let ipiv = [2, 2];
    // inv(A) = [[-3,1],[2,-1/2]]: ||inv||_1 = 5, ||inv||_inf = 4.
    // DLACN2 reaches the exact value in both norms (hand trace: it
    // converges on the second pass). ||A||_1 = 8, ||A||_inf = 10.
    assert_eq!(
        dgbcon(b'1', 2, 1, 1, &ab, 4, &ipiv, 8.0),
        Ok((1.0 / 5.0) / 8.0)
    );
    assert_eq!(
        dgbcon(b'I', 2, 1, 1, &ab, 4, &ipiv, 10.0),
        Ok((1.0 / 4.0) / 10.0)
    );
}

#[test]
fn dgbcon_rescales_with_drscl_when_the_scaled_solve_fits() {
    // A = diag(d, d), d = 1e-300 < SMLNUM = DLAMCH('S')/DLAMCH('P'), so
    // DLATBS scales (scale about 2e-8, |x| about BIGNUM about 1e292). In
    // DGBCON SMLNUM = DLAMCH('S') = 2^-1022 (dgbcon.f:222), so
    // |x|*SMLNUM about 2e-16 < scale: the scale test at dgbcon.f:291 is
    // false and DRSCL runs (dgbcon.f:293). ||inv(A)||_1 = 1e300 and
    // ||A||_1 = d, so the exact rcond is 1.
    let d = 1.0e-300_f64;
    let ab = [d, d];
    let rcond = dgbcon(b'1', 2, 0, 0, &ab, 1, &[1, 2], d).unwrap();
    assert!((rcond - 1.0).abs() <= 8.0 * f64::EPSILON, "rcond = {rcond}");
}

#[test]
fn dgbcon_returns_zero_when_unscaling_would_overflow() {
    // A = diag(d, d), d = 1e-310 (subnormal): DLATBS gives scale about
    // 2e-18 with |x| about 1e292, so |x|*DLAMCH('S') about 2.2e-16 > scale
    // and dgbcon.f:291-292 jumps to label 40 with RCOND = 0.
    let d = 1.0e-310_f64;
    let ab = [d, d];
    assert_eq!(dgbcon(b'1', 2, 0, 0, &ab, 1, &[1, 2], d), Ok(0.0));
}

/// Dense `A` (row-major `n*n`) of the band `kl`, `ku` test matrix.
fn dense_band(n: usize, kl: usize, ku: usize) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            if i <= j + kl && j <= i + ku {
                a[i * n + j] = if i == j {
                    (i % 3 + 1) as f64
                } else {
                    ((5 * i + 3 * j) % 7) as f64 - 3.0
                };
            }
        }
    }
    a
}

/// Gauss-Jordan inverse with partial pivoting (test oracle only).
fn dense_inverse(n: usize, a: &[f64]) -> Vec<f64> {
    let mut m = vec![0.0; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            m[i * 2 * n + j] = a[i * n + j];
        }
        m[i * 2 * n + n + i] = 1.0;
    }
    for c in 0..n {
        let p = (c..n)
            .max_by(|&r, &s| {
                m[r * 2 * n + c].abs().total_cmp(&m[s * 2 * n + c].abs())
            })
            .unwrap();
        for j in 0..2 * n {
            m.swap(c * 2 * n + j, p * 2 * n + j);
        }
        let piv = m[c * 2 * n + c];
        assert!(piv != 0.0, "singular test matrix");
        for j in 0..2 * n {
            m[c * 2 * n + j] /= piv;
        }
        for r in 0..n {
            if r != c {
                let f = m[r * 2 * n + c];
                for j in 0..2 * n {
                    m[r * 2 * n + j] -= f * m[c * 2 * n + j];
                }
            }
        }
    }
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = m[i * 2 * n + n + j];
        }
    }
    inv
}

fn norm1(n: usize, a: &[f64]) -> f64 {
    (0..n)
        .map(|j| (0..n).map(|i| a[i * n + j].abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

fn norm_inf(n: usize, a: &[f64]) -> f64 {
    (0..n)
        .map(|i| (0..n).map(|j| a[i * n + j].abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

#[test]
fn dgbcon_brackets_the_exact_condition_number_for_banded_lu_factors() {
    for &(n, kl, ku) in
        &[(6_usize, 2_usize, 1_usize), (7, 1, 2), (5, 0, 2), (6, 2, 0)]
    {
        let a = dense_band(n, kl, ku);
        let inv = dense_inverse(n, &a);
        // Original A in plain band storage for dlangb: AB(ku+1+i-j, j).
        let ldab0 = kl + ku + 1;
        let mut ab0 = vec![0.0; ldab0 * n];
        for j in 0..n {
            for i in j.saturating_sub(ku)..=(j + kl).min(n - 1) {
                ab0[ix(ku + 1 + i - j, j + 1, ldab0)] = a[i * n + j];
            }
        }
        // Factors from the row-major crate dgbtrf, repacked column-major
        // (same row index, `ab[row*n + j]` -> `ab[row + j*ldab]`).
        let ldab = 2 * kl + ku + 1;
        let mut rm = vec![0.0; ldab * n];
        for j in 0..n {
            for i in j.saturating_sub(ku)..=(j + kl).min(n - 1) {
                rm[(kl + ku + i - j) * n + j] = a[i * n + j];
            }
        }
        let mut ipiv0 = vec![0usize; n];
        dgbtrf(n, kl, ku, &mut rm, ldab, &mut ipiv0).unwrap();
        let mut cm = vec![0.0; ldab * n];
        for r in 0..ldab {
            for j in 0..n {
                cm[r + j * ldab] = rm[r * n + j];
            }
        }
        let ipiv: Vec<usize> = ipiv0.iter().map(|&p| p + 1).collect();
        let mut work = vec![0.0; n];
        for (norm, exact_a, exact_inv) in [
            (b'1', norm1(n, &a), norm1(n, &inv)),
            (b'I', norm_inf(n, &a), norm_inf(n, &inv)),
        ] {
            let anorm =
                dlangb(norm, n, kl, ku, &ab0, ldab0, &mut work).unwrap();
            assert_eq!(
                anorm, exact_a,
                "n={n} kl={kl} ku={ku} norm={}",
                norm as char
            );
            let rcond =
                dgbcon(norm, n, kl, ku, &cm, ldab, &ipiv, anorm).unwrap();
            let exact = 1.0 / (exact_a * exact_inv);
            assert!(
                rcond >= exact * (1.0 - 1.0e-12) && rcond <= exact * n as f64,
                "n={n} kl={kl} ku={ku} norm={}: rcond={rcond}, exact={exact}",
                norm as char
            );
        }
    }
}

// ----------------------------------------------------------------- dgbmv

#[test]
fn dgbmv_matches_hand_products_for_both_transposes_and_strides() {
    // A = [[1,-2,0],[3,4,-5],[0,6,7]], x = [1,2,3]:
    // A x = [1-4, 3+8-15, 12+21] = [-3,-4,33].
    let a = dlangb_matrix();
    let x = [1.0, 2.0, 3.0];
    // y = 3*y + 2*A x with y = [1,1,1]: [3-6, 3-8, 3+66].
    let mut y = [1.0, 1.0, 1.0];
    dgbmv(b'N', 3, 3, 1, 1, 2.0, &a, 3, &x, 1, 3.0, &mut y, 1).unwrap();
    assert_eq!(y, [-3.0, -5.0, 69.0]);
    // A^T x = [1+6, -2+8+18, -10+21] = [7,24,11] (beta = 0 overwrites y).
    let mut yt = [99.0, 99.0, 99.0];
    dgbmv(b'T', 3, 3, 1, 1, 1.0, &a, 3, &x, 1, 0.0, &mut yt, 1).unwrap();
    assert_eq!(yt, [7.0, 24.0, 11.0]);
    // Strided x (2) and y (3), same product A x.
    let xs = [1.0, -7.0, 2.0, -7.0, 3.0];
    let mut ys = [0.0, 8.0, 8.0, 0.0, 8.0, 8.0, 0.0];
    dgbmv(b'N', 3, 3, 1, 1, 1.0, &a, 3, &xs, 2, 0.0, &mut ys, 3).unwrap();
    assert_eq!(ys, [-3.0, 8.0, 8.0, -4.0, 8.0, 8.0, 33.0]);
    let mut yts = [0.0, 8.0, 0.0, 8.0, 0.0];
    dgbmv(b'T', 3, 3, 1, 1, 1.0, &a, 3, &xs, 2, 0.0, &mut yts, 2).unwrap();
    assert_eq!(yts, [7.0, 8.0, 24.0, 8.0, 11.0]);
}

#[test]
fn dgbmv_rectangular_quick_returns_and_argument_errors() {
    // A = [[1,2,0],[0,3,4]] (m = 2, n = 3, kl = 0, ku = 1).
    let a = band(
        2,
        3,
        1.0e300,
        &[(2, 1, 1.0), (1, 2, 2.0), (2, 2, 3.0), (1, 3, 4.0)],
    );
    let mut y = [0.0; 2];
    dgbmv(b'N', 2, 3, 0, 1, 1.0, &a, 2, &[1.0; 3], 1, 0.0, &mut y, 1).unwrap();
    assert_eq!(y, [3.0, 7.0]);
    let mut yt = [0.0; 3];
    dgbmv(b'T', 2, 3, 0, 1, 1.0, &a, 2, &[1.0; 2], 1, 0.0, &mut yt, 1).unwrap();
    assert_eq!(yt, [1.0, 5.0, 4.0]);
    // alpha = 0, beta = 1 returns without touching y; alpha = 0 scales only.
    let mut keep = [5.0, 6.0];
    dgbmv(
        b'N', 2, 3, 0, 1, 0.0, &a, 2, &[1.0; 3], 1, 1.0, &mut keep, 1,
    )
    .unwrap();
    assert_eq!(keep, [5.0, 6.0]);
    dgbmv(
        b'N', 2, 3, 0, 1, 0.0, &a, 2, &[1.0; 3], 1, 2.0, &mut keep, 1,
    )
    .unwrap();
    assert_eq!(keep, [10.0, 12.0]);
    let e = |info| {
        Err(LapackError::IllegalArgument {
            routine: "DGBMV",
            info,
        })
    };
    let x = [1.0; 3];
    assert_eq!(
        dgbmv(b'X', 2, 3, 0, 1, 1.0, &a, 2, &x, 1, 0.0, &mut y, 1),
        e(1)
    );
    assert_eq!(
        dgbmv(b'N', 2, 3, 0, 1, 1.0, &a, 1, &x, 1, 0.0, &mut y, 1),
        e(8)
    );
    assert_eq!(
        dgbmv(b'N', 2, 3, 0, 1, 1.0, &a, 2, &x, 0, 0.0, &mut y, 1),
        e(10)
    );
    assert_eq!(
        dgbmv(b'N', 2, 3, 0, 1, 1.0, &a, 2, &x, 1, 0.0, &mut y, 0),
        e(13)
    );
}

// ----------------------------------------------------------------- dlantb

#[test]
fn dlantb_norms_match_hand_values_for_upper_and_lower_triangles() {
    let mut work = vec![0.0; 3];
    // Upper U = [[2,1,0],[0,4,1],[0,0,5]], k = 1 (poisoned unused corner).
    let u = upper3();
    let mut n = |norm, uplo, diag, a: &[f64]| {
        dlantb(norm, uplo, diag, 3, 1, a, 2, &mut work).unwrap()
    };
    assert_eq!(n(b'M', b'U', b'N', &u), 5.0);
    assert_eq!(n(b'M', b'U', b'U', &u), 1.0); // unit diagonal: max(1, |off|)
    assert_eq!(n(b'1', b'U', b'N', &u), 6.0); // columns 2, 5, 6
    assert_eq!(n(b'1', b'U', b'U', &u), 2.0); // 1, 1+1, 1+1
    assert_eq!(n(b'I', b'U', b'N', &u), 5.0); // rows 3, 5, 5
    assert_eq!(n(b'I', b'U', b'U', &u), 2.0); // rows 1+1, 1+1, 1
                                              // Lower L = [[2,0,0],[1,4,0],[0,1,5]].
    let l = lower3();
    assert_eq!(n(b'M', b'L', b'N', &l), 5.0);
    assert_eq!(n(b'1', b'L', b'N', &l), 5.0); // columns 3, 5, 5
    assert_eq!(n(b'I', b'L', b'N', &l), 6.0); // rows 2, 5, 6
    assert_eq!(n(b'1', b'L', b'U', &l), 2.0); // 1+1, 1+1, 1
    assert_eq!(n(b'I', b'L', b'U', &l), 2.0); // rows 1, 1+1, 1+1
    assert_eq!(n(b'M', b'L', b'U', &l), 1.0);
}

#[test]
fn dlantb_empty_nan_and_unsupported_norms() {
    let mut work = vec![0.0; 3];
    assert_eq!(dlantb(b'M', b'U', b'N', 0, 1, &[], 2, &mut work), Ok(0.0));
    let mut u = upper3();
    u[ix(2, 2, 2)] = f64::NAN;
    assert!(dlantb(b'M', b'U', b'N', 3, 1, &u, 2, &mut work)
        .unwrap()
        .is_nan());
    assert_eq!(
        dlantb(b'F', b'U', b'N', 3, 1, &upper3(), 2, &mut work),
        Err(LapackError::UnportedPath {
            routine: "DLANTB",
            lines: "308-351"
        })
    );
    assert_eq!(
        dlantb(b'Z', b'U', b'N', 3, 1, &upper3(), 2, &mut work),
        Err(LapackError::IllegalArgument {
            routine: "DLANTB",
            info: -1
        })
    );
}

// ----------------------------------------------------------------- dgbrfs

/// `inv(A)` and `inv(A)^T` of `A = [[1,2],[4,6]]` as `DGBTRS` stand-ins.
#[allow(clippy::unnecessary_wraps)]
fn solve_2x2(tr: u8, v: &mut [f64]) -> Result<(), LapackError> {
    let (a, b) = (v[0], v[1]);
    if tr == b'N' {
        v[0] = -3.0 * a + b;
        v[1] = 2.0 * a - 0.5 * b;
    } else {
        v[0] = -3.0 * a + 2.0 * b;
        v[1] = a - 0.5 * b;
    }
    Ok(())
}

#[test]
fn dgbrfs_exact_solution_has_zero_backward_error_and_eps_sized_ferr() {
    // A = diag(2,4,8), exact x = [1,1,1]: residual 0, berr 0. With NZ =
    // min(kl+ku+2, n+1) = 2 the weights are NZ*eps*(|b|+|A||x|), so
    // ||inv(A) diag(W)||_1 = 2*NZ*eps = 4*eps and ferr = 4*eps / 1.
    let ab = [2.0, 4.0, 8.0];
    let b = [2.0, 4.0, 8.0];
    let mut x = [1.0, 1.0, 1.0];
    let mut solve = |_tr: u8, v: &mut [f64]| -> Result<(), LapackError> {
        v[0] /= 2.0;
        v[1] /= 4.0;
        v[2] /= 8.0;
        Ok(())
    };
    let r = dgbrfs(b'N', 3, 0, 0, &ab, 1, &b, &mut x, &mut solve).unwrap();
    assert_eq!(r.berr, 0.0);
    assert_eq!(x, [1.0, 1.0, 1.0]);
    let four_eps = 4.0 * dlamch::EPS;
    assert!((r.ferr - four_eps).abs() <= 1e-14 * four_eps, "{}", r.ferr);
}

#[test]
fn dgbrfs_one_refinement_step_recovers_the_exact_solution() {
    // x0 = [1.5,1,1] for diag(2,4,8) x = [2,4,8]: r = [-1,0,0],
    // berr = max(1/(2+3), 0, 0) = 0.2 > eps; the step adds inv(A) r =
    // [-0.5,0,0], so x = [1,1,1] and the second residual is 0.
    let ab = [2.0, 4.0, 8.0];
    let b = [2.0, 4.0, 8.0];
    let mut x = [1.5, 1.0, 1.0];
    let mut calls = Vec::new();
    let mut solve = |tr: u8, v: &mut [f64]| -> Result<(), LapackError> {
        calls.push(tr);
        v[0] /= 2.0;
        v[1] /= 4.0;
        v[2] /= 8.0;
        Ok(())
    };
    let r = dgbrfs(b'N', 3, 0, 0, &ab, 1, &b, &mut x, &mut solve).unwrap();
    assert_eq!(x, [1.0, 1.0, 1.0]);
    assert_eq!(r.berr, 0.0);
    assert_eq!(calls[0], b'N', "the refinement solve uses TRANS");
}

#[test]
fn dgbrfs_transposed_refinement_uses_the_transposed_matrix() {
    // A = [[1,2],[4,6]] band (kl = ku = 1), solve A^T x = [9,14], exact
    // x = [1,2]. x0 = [1.5,2]: r = [-0.5,-1], inv(A^T) r = [-0.5,0].
    let ab = band(
        3,
        2,
        0.0,
        &[(2, 1, 1.0), (3, 1, 4.0), (1, 2, 2.0), (2, 2, 6.0)],
    );
    let b = [9.0, 14.0];
    let mut x = [1.5, 2.0];
    let mut solve = solve_2x2;
    let r = dgbrfs(b'T', 2, 1, 1, &ab, 3, &b, &mut x, &mut solve).unwrap();
    assert_eq!(x, [1.0, 2.0]);
    assert_eq!(r.berr, 0.0);
}

#[test]
fn dgbrfs_argument_errors_and_empty_system() {
    let mut solve = solve_2x2;
    let mut x: [f64; 0] = [];
    let r = dgbrfs(b'N', 0, 0, 0, &[], 1, &[], &mut x, &mut solve).unwrap();
    assert_eq!((r.ferr, r.berr), (0.0, 0.0));
    let mut x2 = [0.0; 2];
    let e: Result<Refinement, LapackError> =
        dgbrfs(b'X', 2, 1, 1, &[0.0; 6], 3, &[0.0; 2], &mut x2, &mut solve);
    assert_eq!(
        e,
        Err(LapackError::IllegalArgument {
            routine: "DGBRFS",
            info: -1
        })
    );
    let e: Result<Refinement, LapackError> =
        dgbrfs(b'N', 2, 1, 1, &[0.0; 6], 2, &[0.0; 2], &mut x2, &mut solve);
    assert_eq!(
        e,
        Err(LapackError::IllegalArgument {
            routine: "DGBRFS",
            info: -7
        })
    );
}

// ------------------------------------------------------------ dgbsvx FACT=N

/// `A = [[1,2],[4,6]]`, `kl = ku = 1`, plain band storage `LDAB = 3`.
fn pivoted_2x2_ab() -> Vec<f64> {
    band(
        3,
        2,
        0.0,
        &[(2, 1, 1.0), (3, 1, 4.0), (1, 2, 2.0), (2, 2, 6.0)],
    )
}

#[test]
fn dgbsvx_fact_n_solves_a_pivoted_system_exactly_and_estimates_rcond() {
    // b = A*[3,-1] = [1,6]; P A = L U with L21 = 1/4, U = [[4,6],[0,1/2]].
    // ||A||_1 = 8 and ||inv(A)||_1 = 5 (DLACN2 reaches it, see the DGBCON
    // test), so rcond = (1/5)/8. rpvgrw = max|A| / max|U| = 6/6 = 1.
    let ab = pivoted_2x2_ab();
    let s = dgbsvx_fact_n(b'N', 2, 1, 1, &ab, 3, &[1.0, 6.0]).unwrap();
    assert_eq!(s.x, [3.0, -1.0]);
    assert_eq!(s.rcond, (1.0 / 5.0) / 8.0);
    assert_eq!(s.rpvgrw, 1.0);
    assert_eq!(s.berr, 0.0);
    assert_eq!(s.info, 0);
    assert!(s.ferr > 0.0 && s.ferr < 1e-14, "{}", s.ferr);
}

#[test]
fn dgbsvx_fact_n_transposed_uses_the_infinity_norm() {
    // A^T x = [9,14] has x = [1,2]; ||A||_inf = 10, ||inv(A)||_inf = 4.
    let ab = pivoted_2x2_ab();
    let s = dgbsvx_fact_n(b'T', 2, 1, 1, &ab, 3, &[9.0, 14.0]).unwrap();
    assert_eq!(s.x, [1.0, 2.0]);
    assert_eq!(s.rcond, (1.0 / 4.0) / 10.0);
    assert_eq!(s.info, 0);
}

#[test]
fn dgbsvx_fact_n_reports_info_n_plus_1_when_rcond_is_below_eps() {
    // A = diag(1, 1e-17): rcond is about 1e-17 < DLAMCH('E') = 2^-53.
    let ab = [1.0, 1.0e-17];
    let s = dgbsvx_fact_n(b'N', 2, 0, 0, &ab, 1, &[1.0, 1.0e-17]).unwrap();
    assert!(s.rcond < dlamch::EPS, "{}", s.rcond);
    assert_eq!(s.info, 3);
    assert_eq!(s.x, [1.0, 1.0], "X is computed even when INFO = N + 1");
}

#[test]
fn dgbsvx_fact_n_exactly_singular_matrix_is_an_error() {
    // A = [[1,2],[2,4]]: the second pivot is exactly zero (INFO = 2).
    let ab = band(
        3,
        2,
        0.0,
        &[(2, 1, 1.0), (3, 1, 2.0), (1, 2, 2.0), (2, 2, 4.0)],
    );
    assert_eq!(
        dgbsvx_fact_n(b'N', 2, 1, 1, &ab, 3, &[1.0, 1.0]),
        Err(MtxError::SingularMatrix(1))
    );
}

#[test]
fn dgbsvx_fact_n_rejects_bad_shapes() {
    let ab = pivoted_2x2_ab();
    assert!(matches!(
        dgbsvx_fact_n(b'N', 0, 1, 1, &ab, 3, &[]),
        Err(MtxError::InvalidParameter(_))
    ));
    assert!(matches!(
        dgbsvx_fact_n(b'N', 2, 1, 1, &ab, 2, &[1.0, 1.0]),
        Err(MtxError::InvalidParameter(_))
    ));
    assert!(matches!(
        dgbsvx_fact_n(b'N', 2, 1, 1, &ab, 3, &[1.0]),
        Err(MtxError::DimensionMismatch(_))
    ));
    assert!(matches!(
        dgbsvx_fact_n(b'X', 2, 1, 1, &ab, 3, &[1.0, 1.0]),
        Err(MtxError::InvalidParameter(_))
    ));
}

// ------------------------------------------------------- block wrapper (#676)

/// `nvar = 1`, `nz = 3` tridiagonal `[[4,1,0],[2,5,1],[0,3,6]]` in block
/// layout (row `i` of `lblk` couples block `i` to `i-1`; `ublk` row `i`
/// couples `i` to `i+1`).
fn tridiagonal_blocks() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    (
        vec![0.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![1.0, 1.0, 0.0],
    )
}

#[test]
fn block_wrapper_returns_ierr_zero_and_the_solution_for_a_good_system() {
    let (l, d, u) = tridiagonal_blocks();
    // x = [1,2,3]: b = [4+2, 2+10+3, 6+18] = [6,15,24].
    let s =
        dgbsvx_block_tridiagonal(1, 3, &l, &d, &u, &[6.0, 15.0, 24.0]).unwrap();
    assert_eq!(s.ierr, 0);
    assert!(s.rcond > 1e-3 && s.rcond <= 1.0, "{}", s.rcond);
    for (xi, want) in s.x.iter().zip([1.0, 2.0, 3.0]) {
        assert!((xi - want).abs() <= 1e-14, "{xi} vs {want}");
    }
}

#[test]
fn block_wrapper_sets_ierr_n_plus_1_when_rcond_is_below_1e_minus_10() {
    // A = [[1,1],[1,1+1e-12]] (nvar = 1, nz = 2): nonsingular but
    // rcond is about 1e-13 < 1d-10 (DGBSVX_wrapper.f90:252-276): ierr = N + 1 = 3
    // and x is still returned.
    let l = vec![0.0, 1.0];
    let d = vec![1.0, 1.0 + 1.0e-12];
    let u = vec![1.0, 0.0];
    let s = dgbsvx_block_tridiagonal(1, 2, &l, &d, &u, &[2.0, 2.0 + 1.0e-12])
        .unwrap();
    assert_eq!(s.ierr, 3);
    assert!(s.rcond < 1.0e-10, "{}", s.rcond);
    assert!(s.x.iter().all(|v| v.is_finite()));
}

#[test]
fn block_wrapper_exactly_singular_system_is_an_error_not_an_unset_x() {
    // A = [[1,1],[1,1]]: DGBTRF finds a zero pivot; upstream leaves x_flat
    // unset (DGBSVX_wrapper.f90:287), here an explicit error.
    let l = vec![0.0, 1.0];
    let d = vec![1.0, 1.0];
    let u = vec![1.0, 0.0];
    assert!(matches!(
        dgbsvx_block_tridiagonal(1, 2, &l, &d, &u, &[1.0, 1.0]),
        Err(MtxError::SingularMatrix(_))
    ));
}

// ------------------------------------------------- DGBSVX banded wrapper #695

#[test]
fn banded_wrapper_with_kl_at_least_two_returns_the_695_error() {
    // bands for n = 4, ku = 1, kl = 2 (shape (ku+kl+1, n) = 4 x 4).
    let bands = [1.0_f64; 16];
    for kl in [2_usize, 3] {
        let len = (kl + 2) * 4;
        match dgbsvx_banded_from_bands(
            4,
            kl,
            1,
            &bands[..len.min(16)],
            &[1.0; 4],
        ) {
            Err(MtxError::InvalidParameter(msg)) => {
                assert!(msg.contains("#695"), "{msg}");
            }
            other => {
                panic!("kl = {kl}: expected the #695 error, got {other:?}")
            }
        }
    }
}

#[test]
fn banded_wrapper_with_kl_one_matches_a_direct_solve() {
    // Tridiagonal [[4,-1,0],[-1,5,-1],[0,-1,6]] (ku = kl = 1), n = 3, bands
    // are upper / diagonal / lower rows aligned on the column index.
    let bands = [
        -1.0, -1.0, 0.0, //
        4.0, 5.0, 6.0, //
        -1.0, -1.0, 0.0, //
    ];
    // x = [1,2,3]: b = [4-2, -1+10-3, -2+18] = [2,6,16].
    let s =
        dgbsvx_banded_from_bands(3, 1, 1, &bands, &[2.0, 6.0, 16.0]).unwrap();
    for (xi, want) in s.x.iter().zip([1.0, 2.0, 3.0]) {
        assert!((xi - want).abs() <= 1e-14, "{xi} vs {want}");
    }
    assert_eq!(s.ierr, 0);
}

// ------------------------------------------------- DGBTRF blocked code

#[test]
fn dgbtrf_block_size_is_one_up_to_ku_64_and_blocked_code_needs_kl_32() {
    // ilaenv.f:433-446: NB = 1 for KU <= 64, else 32; dgbtrf.f:225 takes
    // the unblocked DGBTF2 when NB <= 1 or NB > KL.
    assert_eq!(dgbtrf_block_size(64), 1);
    assert_eq!(dgbtrf_block_size(65), 32);
    assert!(!dgbtrf_uses_blocked_code(200, 64));
    assert!(!dgbtrf_uses_blocked_code(31, 65));
    assert!(dgbtrf_uses_blocked_code(32, 65));
    // The DGBSVX_tridiagonal_wrapper bands KL = KU = 2*nvar: nvar >= 33.
    assert!(!dgbtrf_uses_blocked_code(64, 64));
    assert!(dgbtrf_uses_blocked_code(66, 66));
}

/// Row-major dense `A = P_1 L_1 P_2 L_2 ... P_n L_n U` (0-based `k`):
/// `P_k` swaps rows `k` and `k + d` for each `(k, d)` in `swaps`, `L_k` is
/// the identity plus multipliers `l_ik` in `{-1/2, 1/2, 1/4}` (`|l| < 1`),
/// and `U` has integer entries in `-3..=3` and diagonal in `{2, -1, 1, -2}`.
///
/// Partial pivoting on `A` reproduces `P_k` (the pivot candidates of step
/// `k` are `u_kk` at row `k + d` and `l_ik u_kk` elsewhere) and `L_k`
/// exactly. Every partially reduced matrix is an `M_k = P_k L_k ... U`
/// whose rows are integer rows of `U` plus `l * (integer row)`, so all
/// entries are small multiples of `1/4`; every product and sum in the
/// factorisation and in both solves below is therefore exact, whatever the
/// operation order. Returns `(A, ipiv)` with 0-based pivots `k + d`.
fn exact_pivoted_band_matrix(
    n: usize,
    kl: usize,
    ku: usize,
    swaps: &[(usize, usize)],
) -> (Vec<f64>, Vec<usize>) {
    let mut ipiv: Vec<usize> = (0..n).collect();
    for &(k, d) in swaps {
        ipiv[k] = k + d;
    }
    let target = |i: usize| swaps.iter().any(|&(k, d)| k + d == i);
    // Upper bandwidth of row k of U: rows moved down by d may reach KU+d
    // (fill-in), rows moved up by d must stop at KU-d.
    let bw = |k: usize| -> usize {
        for &(m, d) in swaps {
            if m == k {
                return (ku + d).min(kl + ku);
            }
            if m + d == k {
                return ku - d;
            }
        }
        ku
    };
    let mut a = vec![0.0_f64; n * n];
    for k in 0..n {
        a[k * n + k] = [2.0, -1.0, 1.0, -2.0][k % 4];
        for j in (k + 1)..=(k + bw(k)).min(n - 1) {
            a[k * n + j] = ((3 * k + 5 * j) % 7) as f64 - 3.0;
        }
    }
    for k in (0..n).rev() {
        // Rows of L_k: none into a pivot target row, and below k + d only
        // for a swapped column (its row of U is the long one).
        let d = swaps.iter().find(|&&(m, _)| m == k).map_or(0, |&(_, d)| d);
        let first = k + 1 + d;
        let second = first + (5 * k) % kl;
        let mut rows = vec![(first, if k % 2 == 0 { 0.5 } else { -0.5 })];
        if second != first {
            rows.push((second, 0.25));
        }
        for (i, l) in rows {
            if i < n && i <= k + kl && !target(i) {
                for j in 0..n {
                    a[i * n + j] += l * a[k * n + j];
                }
            }
        }
        if d > 0 {
            for j in 0..n {
                a.swap(k * n + j, (k + d) * n + j);
            }
        }
    }
    for i in 0..n {
        for j in 0..n {
            let v = a[i * n + j];
            assert_eq!((4.0 * v).fract(), 0.0, "A({i},{j}) = {v}");
            if v != 0.0 {
                assert!(
                    i <= j + kl && j <= i + ku,
                    "A({i},{j}) = {v} outside the band"
                );
            }
        }
    }
    (a, ipiv)
}

fn blocked_case(kl: usize, ku: usize, swaps: &[(usize, usize)]) {
    let n = 200;
    assert!(dgbtrf_uses_blocked_code(kl, ku));
    let (a, ipiv_want) = exact_pivoted_band_matrix(n, kl, ku, swaps);
    let ldab = 2 * kl + ku + 1;
    let mut ab = vec![0.0_f64; ldab * n];
    dense_to_band(n, n, &a, kl, ku, &mut ab, ldab).unwrap();
    let mut lu = ab.clone();
    let mut ipiv = vec![0usize; n];
    dgbtrf(n, kl, ku, &mut lu, ldab, &mut ipiv).unwrap();
    assert_eq!(ipiv, ipiv_want, "kl={kl} ku={ku}");

    // x = integers; b = A x and b^T = A^T x are exact (multiples of 1/4).
    let x: Vec<f64> = (0..n).map(|i| ((7 * i) % 9) as f64 - 4.0).collect();
    let mut bn = vec![0.0_f64; n];
    let mut bt = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            bn[i] += a[i * n + j] * x[j];
            bt[j] += a[i * n + j] * x[i];
        }
    }
    let mut xn = bn.clone();
    dgbtrs_trans(false, n, kl, ku, &lu, ldab, &ipiv, &mut xn).unwrap();
    assert_eq!(xn, x, "A x = b, kl={kl} ku={ku}");
    let mut xt = bt.clone();
    dgbtrs_trans(true, n, kl, ku, &lu, ldab, &ipiv, &mut xt).unwrap();
    assert_eq!(xt, x, "A^T x = b, kl={kl} ku={ku}");

    // DGBSVX(FACT='N') on compact storage AB(ku+1+i-j, j): exact x, zero
    // residual so BERR = 0. U is far from well conditioned, so only the
    // INFO/RCOND consistency (dgbsvx.f:631-632) is checked for RCOND.
    let ldc = kl + ku + 1;
    let mut abc = vec![0.0_f64; ldc * n];
    for j in 1..=n {
        for i in j.saturating_sub(ku).max(1)..=(j + kl).min(n) {
            abc[ix(ku + 1 + i - j, j, ldc)] = a[(i - 1) * n + (j - 1)];
        }
    }
    for (tr, b) in [(b'N', &bn), (b'T', &bt)] {
        let s = dgbsvx_fact_n(tr, n, kl, ku, &abc, ldc, b).unwrap();
        assert_eq!(s.x, x, "dgbsvx trans={}", tr as char);
        assert_eq!(s.berr, 0.0);
        let info = if s.rcond < dlamch::EPS { n + 1 } else { 0 };
        assert_eq!(s.info, info);
        assert!(s.rcond > 0.0 && s.rcond <= 1.0, "{}", s.rcond);
        assert!(s.ferr > 0.0 && s.ferr.is_finite(), "{}", s.ferr);
    }

    // A zero column j gives a zero pivot there (INFO = j+1), reported
    // after the blocked code completes.
    let mut az = a.clone();
    for i in 0..n {
        az[i * n + 150] = 0.0;
    }
    let mut abz = vec![0.0_f64; ldab * n];
    dense_to_band(n, n, &az, kl, ku, &mut abz, ldab).unwrap();
    let mut ipz = vec![0usize; n];
    assert_eq!(
        dgbtrf(n, kl, ku, &mut abz, ldab, &mut ipz),
        Err(MtxError::SingularMatrix(150))
    );
}

#[test]
fn blocked_dgbtrf_kl_ku_66_recovers_pivots_and_exact_solutions() {
    // KL = KU = 66 (DGBSVX_tridiagonal_wrapper with nvar = 33), NB = 32.
    // (31, 66): JJ = 32 pivots on row 98 >= J+KL = 67, the WORK31 branch
    // (dgbtrf.f:318-321, 494-495); (126, 40): JJ = 127 in the block J = 97
    // pivots on row 167 >= 163, the same branch; (40, 3) and (150, 1) are
    // in-band swaps (dgbtrf.f:311-312, 488-489). Rows 31..97 reach KV.
    blocked_case(66, 66, &[(31, 66), (40, 3), (126, 40), (150, 1)]);
}

#[test]
fn blocked_dgbtrf_kl_34_ku_70_recovers_pivots_and_exact_solutions() {
    // KL < KU: I2 = KL-JB = 2 rows of A22, I3 = JB rows of A31.
    blocked_case(34, 70, &[(31, 34), (40, 3), (126, 20), (150, 1)]);
}

// ------------------------------------------- DGBTRS TRANS='T' order

/// `2^-59`, `2^-60` and `2^-61`, exact (powers of two).
const P2M59: f64 = 1.0 / 576_460_752_303_423_488.0;
const P2M60: f64 = P2M59 / 2.0;
const P2M61: f64 = P2M60 / 2.0;

/// `A = U = [[1, 2^-60, 1], [0, 1, 1], [0, 0, 1]]` (`kl = 0`, `ku = 2`) in
/// `dgbtrf` row-major storage `ab[row*n + j]`, `ldab = 3`, diagonal row 2.
fn order_u() -> (Vec<f64>, usize, usize) {
    let e = P2M60;
    let a = [1.0, e, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    let mut ab = vec![0.0_f64; 9];
    dense_to_band(3, 3, &a, 0, 2, &mut ab, 3).unwrap();
    (ab, 0, 2)
}

#[test]
fn dgbtrs_transposed_u_step_sums_rows_in_ascending_order() {
    // U^T x = [1, 0, 1]: x1 = 1, x2 = 0 - 2^-60*1 = -2^-60, and DTBSV
    // (dtbsv.f:343-345) forms x3 = (1 - 1*x1) - 1*x2 = 2^-60 (exact).
    // Descending order would give (1 + 2^-60) - 1 = 0.
    let (ab, kl, ku) = order_u();
    let e = P2M60;
    let mut lu = ab.clone();
    let mut ipiv = [0usize; 3];
    dgbtrf(3, kl, ku, &mut lu, 3, &mut ipiv).unwrap();
    assert_eq!(ipiv, [0, 1, 2]);
    let mut x = [1.0, 0.0, 1.0];
    dgbtrs_trans(true, 3, kl, ku, &lu, 3, &ipiv, &mut x).unwrap();
    assert_eq!(x, [1.0, -e, e]);
}

#[test]
fn dgbsvx_transposed_u_step_gives_exact_x_and_hand_ferr() {
    // Same system through DGBSVX: the residual is 0 (DGBMV rounds
    // 1 - 2^-60 + 2^-60 to 1), so BERR = 0. W = NZ*eps*(|b| + |A^T||x|)
    // with NZ = 4 is (8, 2^-57, 8)*eps, and DLACN2 estimates
    // ||C||_1 for C = diag(W) inv(A) (KASE = 1 solves with TRANST = 'N').
    // x = 1/3: inv(A) x = (0, 0, 1/3) (DTBSV skips the zero x2), est =
    // 8*eps/3; x = sign = 1: C^T x = inv(A^T) (W1, W2, W3) peaks at j = 1;
    // C e1 = (W1, 0, 0), est = 8*eps, the same signs, so the alternating
    // test runs: C (1, -1.5, 2) = (-8, ~0, 16)*eps, 2*24*eps/9 < 8*eps.
    // FERR = 8*eps / ||x||_inf = 8*eps (the exact ||C||_1 is 16*eps).
    let e = P2M60;
    let abc = band(
        3,
        3,
        0.0,
        &[
            (3, 1, 1.0),
            (2, 2, e),
            (3, 2, 1.0),
            (1, 3, 1.0),
            (2, 3, 1.0),
            (3, 3, 1.0),
        ],
    );
    let s = dgbsvx_fact_n(b'T', 3, 0, 2, &abc, 3, &[1.0, 0.0, 1.0]).unwrap();
    assert_eq!(s.x, [1.0, -e, e]);
    assert_eq!(s.berr, 0.0);
    assert_eq!(s.ferr, 8.0 * dlamch::EPS);
}

#[test]
fn dgbtrs_transposed_l_step_accumulates_the_dgemv_sum_before_subtracting() {
    // A = L = [[1,0,0],[1/2,1,0],[1/2,0,1]] (kl = 2, ku = 0, no pivoting
    // since |l| < 1). L^T x = [1, 2, 2^-59]: x3 = 2^-59, x2 = 2, and DGEMV
    // (dgemv.f:304-308) forms TEMP = (0 + 2*(1/2)) + 2^-59*(1/2) = 1 (the
    // 2^-60 is below half an ulp), x1 = 1 + (-1)*1 = 0. Subtracting term
    // by term would give (1 - 1) - 2^-60 = -2^-60.
    let a = [1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.5, 0.0, 1.0];
    let ldab = 5;
    let mut lu = vec![0.0_f64; ldab * 3];
    dense_to_band(3, 3, &a, 2, 0, &mut lu, ldab).unwrap();
    let mut ipiv = [0usize; 3];
    dgbtrf(3, 2, 0, &mut lu, ldab, &mut ipiv).unwrap();
    assert_eq!(ipiv, [0, 1, 2]);
    let mut x = [1.0, 2.0, P2M59];
    dgbtrs_trans(true, 3, 2, 0, &lu, ldab, &ipiv, &mut x).unwrap();
    assert_eq!(x, [0.0, 2.0, P2M59]);

    // DGBSVX: the computed residual b - A^T x is also 0 (DGBMV rounds the
    // same sum to 1), so DGBRFS keeps x1 = 0 with BERR = 0, and FERR still
    // bounds the true relative error 2^-60 / ||x||_inf = 2^-61.
    let abc = band(
        3,
        3,
        0.0,
        &[
            (1, 1, 1.0),
            (2, 1, 0.5),
            (3, 1, 0.5),
            (1, 2, 1.0),
            (1, 3, 1.0),
        ],
    );
    let s = dgbsvx_fact_n(b'T', 3, 2, 0, &abc, 3, &[1.0, 2.0, P2M59]).unwrap();
    assert_eq!(s.x, [0.0, 2.0, P2M59]);
    assert_eq!(s.berr, 0.0);
    assert!(s.ferr >= P2M61 && s.ferr < 64.0 * dlamch::EPS, "{}", s.ferr);
}
