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

use flexmesa_mtx::band_lapack::{dgbcon, dlangb, dlatbs, drscl, dtbsv};
use flexmesa_mtx::dgbtrf;
use flexmesa_mtx::lapack::{dlamch, ix, LapackError};

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
