//! Banded matrix storage, LU factorization (`dgbtrf`), and solvers (`dgbtrs`).
//!
//! For an $N \times N$ banded matrix with $k_l$ sub-diagonals and $k_u$
//! super-diagonals stored in a buffer `ab` of shape `(ldab, n)`
//! (`ab[row * n + j]`):
//! - In general band conversions (`dense_to_band`, `band_to_dense`) with
//!   `ldab >= kl + ku + 1`, entry $A_{i,j}$ resides at
//!   `row = (ldab - 1 - kl) + i - j` and column `j`.
//! - In banded LU factorization (`dgbtrf`, `dgbtrs`) with
//!   `ldab >= 2 * kl + ku + 1`, the top `kl` rows (`0..kl`) provide workspace
//!   for pivot fill-in and the original diagonal resides at
//!   `row = ldab - 1 - kl`.
//! - In left band-vector multiplication (`band_multiply_xa`), entry $A_{i,j}$
//!   is stored in compact upper-aligned layout at `row = ku + i - j`.
//! - All routines reject non-finite matrix and vector entries (`NaN`, `Inf`)
//!   with [`MtxError::NonFiniteInput`].

use crate::MtxError;
use flexmesa_utils::is_bad_slice;

/// Converts a row-major dense $N \times N$ matrix `a` (row stride `ndim >= n`)
/// into band storage `ab` of shape `(ldab, n)` with `ldab >= ml + mu + 1`.
///
/// Arguments:
/// - `n`: Matrix dimension $N \ge 1$.
/// - `ndim`: Leading row stride of `a` (`ndim >= n`).
/// - `a`: Input row-major dense slice of length $\ge N \cdot \mathrm{ndim}$.
/// - `ml`, `mu`: Lower and upper bandwidths ($m_l + m_u + 1 \le N$).
/// - `ab`: Output band buffer of length $\ge \mathrm{ldab} \cdot N$.
/// - `ldab`: Leading row count of `ab` (`ldab >= ml + mu + 1`).
///
/// # Errors
///
/// Returns [`MtxError::InvalidParameter`] if `ml + mu + 1 > n` or
/// `ldab < ml + mu + 1`, or [`MtxError::DimensionMismatch`] if slice lengths
/// are too small.
pub fn dense_to_band(
    n: usize,
    ndim: usize,
    a: &[f64],
    ml: usize,
    mu: usize,
    ab: &mut [f64],
    ldab: usize,
) -> Result<(), MtxError> {
    if n == 0 || ndim < n || ml + mu + 1 > n || ldab < ml + mu + 1 {
        return Err(MtxError::InvalidParameter(
            "invalid band dimensions for dense_to_band",
        ));
    }
    if a.len() < n * ndim || ab.len() < ldab * n {
        return Err(MtxError::DimensionMismatch(
            "a or ab buffer too small for dense_to_band",
        ));
    }
    ab[..(ldab * n)].fill(0.0);
    let diag_row = ldab - 1 - ml;
    for j in 0..n {
        let i_min = j.saturating_sub(mu);
        let i_max = (j + ml).min(n - 1);
        for i in i_min..=i_max {
            let row = (diag_row + i) - j;
            ab[row * n + j] = a[i * ndim + j];
        }
    }
    Ok(())
}

/// Converts a banded matrix `ab` of shape `(ldab, n)` (`ldab >= ml + mu + 1`)
/// into a row-major dense matrix `a` of shape `(n, ndim)`.
///
/// # Errors
///
/// Returns [`MtxError::InvalidParameter`] if `ml + mu + 1 > n` or
/// `ldab < ml + mu + 1`, or [`MtxError::DimensionMismatch`] if slice lengths
/// are too small.
pub fn band_to_dense(
    n: usize,
    ml: usize,
    mu: usize,
    ab: &[f64],
    ldab: usize,
    ndim: usize,
    a: &mut [f64],
) -> Result<(), MtxError> {
    if n == 0 || ndim < n || ml + mu + 1 > n || ldab < ml + mu + 1 {
        return Err(MtxError::InvalidParameter(
            "invalid band dimensions for band_to_dense",
        ));
    }
    if ab.len() < ldab * n || a.len() < n * ndim {
        return Err(MtxError::DimensionMismatch(
            "ab or a buffer too small for band_to_dense",
        ));
    }
    a[..(n * ndim)].fill(0.0);
    let diag_row = ldab - 1 - ml;
    for j in 0..n {
        let i_min = j.saturating_sub(mu);
        let i_max = (j + ml).min(n - 1);
        for i in i_min..=i_max {
            let row = (diag_row + i) - j;
            a[i * ndim + j] = ab[row * n + j];
        }
    }
    Ok(())
}

/// Evaluates $b = x^{\top} A = A^{\top} x$ for a compact banded matrix `ab` of
/// shape `(ldab, n)` (`ldab >= kl + ku + 1`) where $A_{i,j}$ is stored at
/// `row = ku + i - j` (matching upstream `do_band_multiply_xa`).
///
/// # Errors
///
/// Returns [`MtxError::InvalidParameter`] if `ldab < kl + ku + 1` or
/// [`MtxError::DimensionMismatch`] if slice lengths do not match.
pub fn band_multiply_xa(
    n: usize,
    kl: usize,
    ku: usize,
    ab: &[f64],
    ldab: usize,
    x: &[f64],
    b: &mut [f64],
) -> Result<(), MtxError> {
    if ldab < kl + ku + 1 {
        return Err(MtxError::InvalidParameter(
            "ldab must be >= kl + ku + 1 for band_multiply_xa",
        ));
    }
    if ab.len() < ldab * n || x.len() != n || b.len() != n {
        return Err(MtxError::DimensionMismatch(
            "slice lengths do not match (ldab, n) for band_multiply_xa",
        ));
    }
    for j in 0..n {
        let i_min = j.saturating_sub(ku);
        let i_max = (j + kl).min(n.saturating_sub(1));
        let mut sum = 0.0;
        for i in i_min..=i_max {
            let row = (ku + i) - j;
            sum += x[i] * ab[row * n + j];
        }
        b[j] = sum;
    }
    Ok(())
}

/// Computes the in-place banded LU factorization $P A = L U$ with partial
/// pivoting (`dgbtrf`) for `ab` of shape `(ldab, n)` with
/// `ldab >= 2 * kl + ku + 1` and diagonal at `row = ldab - 1 - kl`.
///
/// As LAPACK `DGBTRF` (`SRC/dgbtrf.f:218-229`), the block size is
/// `NB = ILAENV(1, 'DGBTRF', ...)` (`1` for `ku <= 64`, else `32`,
/// `SRC/ilaenv.f:433-446`): for `NB <= 1` or `NB > kl` the unblocked
/// `DGBTF2` loop below runs; otherwise (`ku > 64` and `kl >= 32`) the
/// blocked code [`crate::band_lapack::dgbtrf::dgbtrf_blocked`]
/// (`dgbtrf.f:230-506`) runs on a column-major copy of the band, and the
/// factorisation is completed before a zero pivot is reported.
///
/// # Errors
///
/// Returns [`MtxError::InvalidParameter`] if `ldab < 2 * kl + ku + 1`,
/// [`MtxError::DimensionMismatch`] if slice lengths do not match,
/// [`MtxError::NonFiniteInput`] if `ab` contains `NaN` or $\pm\infty$, or
/// [`MtxError::SingularMatrix`] if a pivot is zero.
pub fn dgbtrf(
    n: usize,
    kl: usize,
    ku: usize,
    ab: &mut [f64],
    ldab: usize,
    ipiv: &mut [usize],
) -> Result<(), MtxError> {
    if n == 0 || ldab < 2 * kl + ku + 1 {
        return Err(MtxError::InvalidParameter(
            "n must be >= 1 and ldab must be >= 2 * kl + ku + 1",
        ));
    }
    if ab.len() < ldab * n || ipiv.len() != n {
        return Err(MtxError::DimensionMismatch(
            "ab must have length >= ldab * n and ipiv must have length n",
        ));
    }
    if is_bad_slice(&ab[..(ldab * n)]) {
        return Err(MtxError::NonFiniteInput(
            "banded matrix ab contains NaN/Inf",
        ));
    }

    let diag = ldab - 1 - kl;
    let ku_eff = diag;
    for j in 0..n {
        let fill_end = diag.saturating_sub(ku);
        for row in 0..fill_end {
            ab[row * n + j] = 0.0;
        }
    }

    // dgbtrf.f:218-225: blocked code unless NB <= 1 or NB > KL.
    if crate::band_lapack::dgbtrf::dgbtrf_uses_blocked_code(kl, ku) {
        return dgbtrf_blocked_row_major(n, kl, ku, ab, ldab, ipiv);
    }

    // dgbtrf.f:229: DGBTF2 (SRC/dgbtf2.f).
    let mut ju = 0usize;
    for j in 0..n {
        let km = kl.min(n - 1 - j);
        let mut jp = 0usize;
        let mut max_val = ab[diag * n + j].abs();
        for l in 1..=km {
            let val = ab[(diag + l) * n + j].abs();
            if val > max_val {
                max_val = val;
                jp = l;
            }
        }
        ipiv[j] = j + jp;
        if max_val == 0.0 {
            return Err(MtxError::SingularMatrix(j));
        }
        ju = ju.max((j + ku + jp).min(n - 1));
        if jp != 0 {
            for col in j..=ju {
                let r1 = (diag + j) - col;
                let r2 = (diag + j + jp) - col;
                ab.swap(r1 * n + col, r2 * n + col);
            }
        }
        if km > 0 {
            // Match LAPACK `dgbtf2`: `CALL DSCAL(KM, ONE / AB(KV+1, J), ...)`
            // and `CALL DGER(KM, JU-J, -ONE, ...)` (#165 Item 11, N1-8).
            let inv_pivot = 1.0 / ab[diag * n + j];
            for l in 1..=km {
                ab[(diag + l) * n + j] *= inv_pivot;
            }
            for col in (j + 1)..=ju {
                let r_top = (diag + j) - col;
                let temp = ab[r_top * n + col];
                if temp != 0.0 {
                    for l in 1..=km {
                        let r_sub = r_top + l;
                        if r_sub <= diag + kl && r_sub <= ku_eff + kl {
                            let mult = ab[(diag + l) * n + j];
                            let idx = r_sub * n + col;
                            ab[idx] = flexmesa_math::fma(-mult, temp, ab[idx]);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Runs the blocked `DGBTRF` code (`SRC/dgbtrf.f:230-506`) for
/// [`dgbtrf`] on a column-major copy `AB(2*kl+ku+1, n)` of the bottom
/// `2 * kl + ku + 1` rows of the row-major band, then copies the factors and
/// the 0-based pivots back.
fn dgbtrf_blocked_row_major(
    n: usize,
    kl: usize,
    ku: usize,
    ab: &mut [f64],
    ldab: usize,
    ipiv: &mut [usize],
) -> Result<(), MtxError> {
    let ldf = 2 * kl + ku + 1;
    let off = ldab - ldf;
    let mut cm = vec![0.0_f64; ldf * n];
    for r in 0..ldf {
        for j in 0..n {
            cm[r + j * ldf] = ab[(off + r) * n + j];
        }
    }
    let mut ipiv1 = vec![0usize; n];
    let info = crate::band_lapack::dgbtrf::dgbtrf_blocked(
        n, n, kl, ku, &mut cm, ldf, &mut ipiv1,
    )?;
    for r in 0..ldf {
        for j in 0..n {
            ab[(off + r) * n + j] = cm[r + j * ldf];
        }
    }
    for (p0, &p1) in ipiv.iter_mut().zip(ipiv1.iter()) {
        *p0 = p1 - 1;
    }
    if info > 0 {
        // DGBTRF INFO = j > 0 (dgbtrf.f:344-345): U(j,j) is exactly zero.
        return Err(MtxError::SingularMatrix(info - 1));
    }
    Ok(())
}

/// Solves $A x = b$ (when `trans == false`) or $A^{\top} x = b$ (when
/// `trans == true`) in-place using the banded LU factorization from
/// [`dgbtrf`], matching LAPACK `DGBTRS`.
///
/// # Errors
///
/// Returns [`MtxError::InvalidParameter`] if dimensions or pivot indices are
/// invalid, [`MtxError::DimensionMismatch`] on buffer size mismatch,
/// [`MtxError::NonFiniteInput`] if inputs contain `NaN`/`Inf`, or
/// [`MtxError::SingularMatrix`] if a diagonal pivot entry is zero.
#[allow(clippy::too_many_arguments)]
pub fn dgbtrs_trans(
    trans: bool,
    n: usize,
    kl: usize,
    ku: usize,
    ab: &[f64],
    ldab: usize,
    ipiv: &[usize],
    b: &mut [f64],
) -> Result<(), MtxError> {
    if n == 0 || ldab < 2 * kl + ku + 1 {
        return Err(MtxError::InvalidParameter(
            "n must be >= 1 and ldab must be >= 2 * kl + ku + 1",
        ));
    }
    if ab.len() < ldab * n || ipiv.len() != n || b.len() != n {
        return Err(MtxError::DimensionMismatch(
            "ab, ipiv, or b length mismatch in dgbtrs",
        ));
    }
    if is_bad_slice(&ab[..(ldab * n)]) || is_bad_slice(b) {
        return Err(MtxError::NonFiniteInput("ab or rhs b contains NaN/Inf"));
    }
    if ipiv.iter().any(|&l| l >= n) {
        return Err(MtxError::InvalidParameter("pivot index out of bounds"));
    }

    let diag = ldab - 1 - kl;
    if !trans {
        for j in 0..n {
            let lm = kl.min(n - 1 - j);
            let l = ipiv[j];
            if l != j {
                b.swap(l, j);
            }
            let bj = b[j];
            for i in 1..=lm {
                let idx = j + i;
                b[idx] =
                    flexmesa_math::fma(-ab[(diag + i) * n + j], bj, b[idx]);
            }
        }

        for j in (0..n).rev() {
            let u_jj = ab[diag * n + j];
            if u_jj == 0.0 {
                return Err(MtxError::SingularMatrix(j));
            }
            b[j] /= u_jj;
            let bj = b[j];
            let lm = j.min(kl + ku);
            for k in 1..=lm {
                let row = diag - k;
                let idx = j - k;
                b[idx] = flexmesa_math::fma(-ab[row * n + j], bj, b[idx]);
            }
        }
    } else {
        // dgbtrs.f:241-247: DTBSV('Upper', 'Transpose', 'Non-unit', N,
        // KL+KU, AB, LDAB, B, 1), i.e. dtbsv.f:340-348 with K = KL+KU (the
        // zero-diagonal error is this crate's, DTBSV divides by it).
        let k = kl + ku;
        for j in 0..n {
            let mut temp = b[j];
            // DO 90 I = MAX(1,J-K), J-1 (ascending)
            for i in j.saturating_sub(k)..j {
                temp -= ab[(diag + i - j) * n + j] * b[i];
            }
            let u_jj = ab[diag * n + j];
            if u_jj == 0.0 {
                return Err(MtxError::SingularMatrix(j));
            }
            temp /= u_jj;
            b[j] = temp;
        }

        // dgbtrs.f:251-260: solve L**T*X = B (LNOTI = KL > 0).
        if kl > 0 {
            // DO 40 J = N-1, 1, -1
            for j in (0..n - 1).rev() {
                let lm = kl.min(n - 1 - j);
                // DGEMV('Transpose', LM, 1, -ONE, B(J+1), LDB, AB(KD+1,J),
                // 1, ONE, B(J), LDB): dgemv.f:303-308 with ALPHA = -ONE.
                let alpha = -1.0_f64;
                let mut temp = 0.0_f64;
                for i in 1..=lm {
                    temp += b[j + i] * ab[(diag + i) * n + j];
                }
                b[j] += alpha * temp;
                let l = ipiv[j];
                if l != j {
                    b.swap(l, j);
                }
            }
        }
    }
    Ok(())
}

/// Solves $A x = b$ in-place using the banded LU factorization from [`dgbtrf`].
///
/// # Errors
///
/// Returns [`MtxError::InvalidParameter`] or [`MtxError::DimensionMismatch`] on
/// invalid shapes, [`MtxError::NonFiniteInput`] on non-finite inputs, or
/// [`MtxError::SingularMatrix`] if a diagonal entry is zero.
pub fn dgbtrs(
    n: usize,
    kl: usize,
    ku: usize,
    ab: &[f64],
    ldab: usize,
    ipiv: &[usize],
    b: &mut [f64],
) -> Result<(), MtxError> {
    dgbtrs_trans(false, n, kl, ku, ab, ldab, ipiv, b)
}

/// Solves a banded system $A x = b$ given `ab` of shape `(ldab, n)` with
/// `ldab >= 2 * kl + ku + 1`.
///
/// # Errors
///
/// Propagates any error from [`dgbtrf`] or [`dgbtrs`].
pub fn solve_banded(
    n: usize,
    kl: usize,
    ku: usize,
    ab: &[f64],
    ldab: usize,
    b: &[f64],
) -> Result<Vec<f64>, MtxError> {
    if ab.len() < ldab * n || b.len() != n {
        return Err(MtxError::DimensionMismatch(
            "ab or b length mismatch in solve_banded",
        ));
    }
    let mut lu = ab[..(ldab * n)].to_vec();
    let mut ipiv = vec![0usize; n];
    let mut x = b.to_vec();
    dgbtrf(n, kl, ku, &mut lu, ldab, &mut ipiv)?;
    dgbtrs(n, kl, ku, &lu, ldab, &ipiv, &mut x)?;
    Ok(x)
}

/// Estimates the 1-norm reciprocal condition number of a banded matrix from its
/// unfactored band buffer `ab` and factored LU buffer `lu_ab` using Hager's
/// 1-norm estimator (matching LAPACK `DGBCON`, `mtx/public/mtx_rcond.dek:32-62`).
///
/// # Errors
///
/// Returns [`MtxError::InvalidParameter`] on invalid band/pivot bounds,
/// [`MtxError::DimensionMismatch`] on buffer size mismatch, or
/// [`MtxError::NonFiniteInput`] if `ab` or `lu_ab` contains `NaN`/`Inf`.
pub fn rcond_banded(
    n: usize,
    kl: usize,
    ku: usize,
    ab: &[f64],
    lu_ab: &[f64],
    ldab: usize,
    ipiv: &[usize],
) -> Result<f64, MtxError> {
    if n == 0 || ldab < 2 * kl + ku + 1 {
        return Err(MtxError::InvalidParameter(
            "invalid dimensions for rcond_banded",
        ));
    }
    if ab.len() < ldab * n || lu_ab.len() < ldab * n || ipiv.len() != n {
        return Err(MtxError::DimensionMismatch(
            "buffer length mismatch in rcond_banded",
        ));
    }
    if is_bad_slice(&ab[..(ldab * n)]) || is_bad_slice(&lu_ab[..(ldab * n)]) {
        return Err(MtxError::NonFiniteInput(
            "ab or lu_ab contains NaN/Inf in rcond_banded",
        ));
    }
    if ipiv.iter().any(|&p| p >= n) {
        return Err(MtxError::InvalidParameter(
            "pivot index out of bounds in rcond_banded",
        ));
    }
    let diag = ldab - 1 - kl;
    let mut anorm = 0.0_f64;
    for j in 0..n {
        let i_min = j.saturating_sub(ku);
        let i_max = (j + kl).min(n - 1);
        let mut col_sum = 0.0;
        for i in i_min..=i_max {
            let row = (diag + i) - j;
            col_sum += ab[row * n + j].abs();
        }
        anorm = anorm.max(col_sum);
    }
    crate::dense::hager_rcond_1norm(n, anorm, |trans, v| {
        dgbtrs_trans(trans, n, kl, ku, lu_ab, ldab, ipiv, v)
    })
}
