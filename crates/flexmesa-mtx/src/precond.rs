// SPDX-License-Identifier: GPL-3.0-or-later
//! Matrix preconditioners, `DGBSVX`-style scaled banded/block-tridiagonal
//! solvers, and LAPACK row/column equilibration (`dgeequ`, `dlaqge`, `dgbequ`,
//! `dlaqgb`).
//!
//! Parity targets in upstream MESA `26.4.1`:
//! - `mtx/private/pre_conditioners.f90` (`compute_block_preconditioner`,
//!   `compute_block_left_preconditioner`, `compute_band_preconditioner`)
//! - `mtx/private/DGBSVX_wrapper.f90` (`DGBSVX_banded_wrapper`,
//!   `DGBSVX_tridiagonal_wrapper`)
//! - `star/private/star_bcyclic.f90:623-681,849-968` (`DGESVX` row/column
//!   equilibration via `DGEEQU` / `DLAQGE`)

use crate::band::{dgbtrf, dgbtrs, rcond_banded};
use crate::band_lapack::dgbsvx_fact_n;
use crate::block_tridiag::check_blocks;
use crate::dense::{dgetrf, dgetrs, rcond_dense};
use crate::MtxError;
use flexmesa_utils::is_bad_slice;

/// Safe lower positive bound matching LAPACK `DLAMCH('S')`.
const SMLNUM: f64 = f64::MIN_POSITIVE;
/// Safe upper reciprocal bound `1 / DLAMCH('S')`.
const BIGNUM: f64 = 1.0 / SMLNUM;
/// Equilibration ratio threshold (`THRESH = 0.1` in `DLAQGE` / `DLAQGB`).
const EQUED_THRESH: f64 = 0.1;
/// Lower magnitude threshold (`DLAMCH('S') / DLAMCH('P')` in `DLAQGE`/`DLAQGB`).
const EQUED_SMALL: f64 = SMLNUM / f64::EPSILON;
/// Upper magnitude threshold (`1 / EQUED_SMALL` in `DLAQGE`/`DLAQGB`).
const EQUED_LARGE: f64 = 1.0 / EQUED_SMALL;

/// Form of row/column equilibration chosen by [`dgeequ`] / [`dgbequ`],
/// matching LAPACK's `EQUED` (`'N'`, `'R'`, `'C'`, `'B'`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Equilibration {
    /// No equilibration applied (`EQUED = 'N'`).
    None,
    /// Row equilibration applied: $D_r A$ (`EQUED = 'R'`).
    Row,
    /// Column equilibration applied: $A D_c$ (`EQUED = 'C'`).
    Col,
    /// Both row and column equilibration applied: $D_r A D_c$ (`EQUED = 'B'`).
    Both,
}

/// Equilibration scale factors and diagnostics from [`dgeequ`] or [`dgbequ`].
#[derive(Debug, Clone, PartialEq)]
pub struct EquilibrationFactors {
    /// Row scale factors $r_i$ of length `n`.
    pub row_scale: Vec<f64>,
    /// Column scale factors $c_j$ of length `n`.
    pub col_scale: Vec<f64>,
    /// Ratio `min(r_i) / max(r_i)` (clamped to `[SMLNUM, BIGNUM]`), or `0.0`
    /// when a zero row was encountered.
    pub rowcnd: f64,
    /// Ratio `min(c_j) / max(c_j)` after row scaling, or `0.0` when a zero
    /// column was encountered.
    pub colcnd: f64,
    /// Maximum entry magnitude $\max_{i,j} |A_{i,j}|$.
    pub amax: f64,
    /// Recommended equilibration mode per LAPACK `DLAQGE` / `DLAQGB` thresholds.
    pub equed: Equilibration,
}

/// Solution vector, 1-norm reciprocal condition estimate, and applied scale
/// factors returned by `dgbsvx_*` and equilibrated solvers.
#[derive(Debug, Clone, PartialEq)]
pub struct PreconditionedSolve {
    /// Solution vector $x$ solving $A x = b$.
    pub x: Vec<f64>,
    /// Estimated 1-norm reciprocal condition number of the scaled system.
    pub rcond: f64,
    /// Row scale multipliers (`pc` or `r`) applied to the linear system.
    pub row_scale: Vec<f64>,
    /// Column scale multipliers (`c` or `1.0` when unscaled) applied to $x$.
    pub col_scale: Vec<f64>,
    /// `ierr` of `DGBSVX_tridiagonal_wrapper`: `0`, or `N + 1` when `DGBSVX`
    /// reports `INFO /= 0` or `rcond < 1d-10`
    /// (`DGBSVX_wrapper.f90:252-276`). Always `0` for the other solvers.
    pub ierr: usize,
}

/// Decides which equilibration mode (`None`, `Row`, `Col`, `Both`) applies
/// using LAPACK `DLAQGE` / `DLAQGB` rules.
#[must_use]
fn classify_equilibration(
    rowcnd: f64,
    colcnd: f64,
    amax: f64,
) -> Equilibration {
    if amax == 0.0 || rowcnd == 0.0 || colcnd == 0.0 {
        return Equilibration::None;
    }
    let skip_row =
        rowcnd >= EQUED_THRESH && (EQUED_SMALL..=EQUED_LARGE).contains(&amax);
    let skip_col = colcnd >= EQUED_THRESH;
    match (!skip_row, !skip_col) {
        (false, false) => Equilibration::None,
        (true, false) => Equilibration::Row,
        (false, true) => Equilibration::Col,
        (true, true) => Equilibration::Both,
    }
}

/// Finalizes LAPACK row scale factors from raw row maxima in `r`, returning
/// `(rcmax, rowcnd, has_zero)`.
fn finalize_scale_factors(scale: &mut [f64]) -> (f64, f64, bool) {
    let mut rcmin = BIGNUM;
    let mut rcmax = 0.0_f64;
    for &v in scale.iter() {
        rcmax = rcmax.max(v);
        rcmin = rcmin.min(v);
    }
    if rcmin == 0.0 {
        return (rcmax, 0.0, true);
    }
    for v in scale.iter_mut() {
        *v = 1.0 / v.clamp(SMLNUM, BIGNUM);
    }
    let cond = rcmin.max(SMLNUM) / rcmax.min(BIGNUM);
    (rcmax, cond, false)
}

/// Computes row preconditioner factors for a banded matrix stored in left-aligned
/// band-major format `(ku + kl + 1, n)`, matching upstream
/// `mtx/private/pre_conditioners.f90:98-134` (`compute_band_preconditioner`).
///
/// Band row ordering in `bands` is `u_{ku}, ..., u_1, diag, l_1, ..., l_{kl}`,
/// where superdiagonal `d in 1..=ku` (`row = ku - d`) has valid column indices
/// `0..(n - d)` at `A[col, col + d]` and subdiagonal `d in 1..=kl`
/// (`row = ku + d`) has valid column indices `0..(n - d)` at `A[col + d, col]`.
///
/// # Errors
///
/// Returns [`MtxError::InvalidParameter`] if `n == 0`,
/// [`MtxError::DimensionMismatch`] if `bands` or `pre_conditioner` lengths do
/// not match `(ku + kl + 1) * n` and `n`, or [`MtxError::NonFiniteInput`] if
/// `bands` contains `NaN`/`Inf`.
pub fn compute_band_preconditioner(
    n: usize,
    kl: usize,
    ku: usize,
    bands: &[f64],
    pre_conditioner: &mut [f64],
) -> Result<(), MtxError> {
    if n == 0 {
        return Err(MtxError::InvalidParameter(
            "n must be >= 1 in compute_band_preconditioner",
        ));
    }
    let n_bands = ku
        .checked_add(kl)
        .and_then(|s| s.checked_add(1))
        .ok_or(MtxError::InvalidParameter("band count overflow"))?;
    let expected = n_bands
        .checked_mul(n)
        .ok_or(MtxError::InvalidParameter("band buffer size overflow"))?;
    if bands.len() != expected || pre_conditioner.len() != n {
        return Err(MtxError::DimensionMismatch(
            "bands or pre_conditioner length mismatch",
        ));
    }
    if is_bad_slice(bands) {
        return Err(MtxError::NonFiniteInput(
            "bands contains NaN/Inf in compute_band_preconditioner",
        ));
    }

    pre_conditioner.fill(0.0);

    // Upper bands (superdiagonal d = ku - b, entry A[i, i + d]).
    for b in 0..ku {
        let d = ku - b;
        let row_off = b * n;
        for (i, pc_i) in pre_conditioner
            .iter_mut()
            .enumerate()
            .take(n.saturating_sub(d))
        {
            *pc_i = pc_i.max(bands[row_off + i].abs());
        }
    }

    // Diagonal band (entry A[i, i]).
    let diag_off = ku * n;
    for (i, pc_i) in pre_conditioner.iter_mut().enumerate() {
        *pc_i = pc_i.max(bands[diag_off + i].abs());
    }

    // Lower bands (subdiagonal d = l + 1, entry A[col + d, col]).
    for l in 0..kl {
        let d = l + 1;
        let row_off = (ku + d) * n;
        for col in 0..n.saturating_sub(d) {
            let row = col + d;
            pre_conditioner[row] =
                pre_conditioner[row].max(bands[row_off + col].abs());
        }
    }

    Ok(())
}

/// Computes row preconditioner factors (`row_scale[j * nvar + k]`) for a
/// `(nz, nvar, nvar)` block-tridiagonal system, matching upstream
/// `mtx/private/pre_conditioners.f90:23-46` (`compute_block_preconditioner`).
///
/// Uses `flexmesa-mtx` block-tridiagonal layout where `lblk[j]` is the
/// subdiagonal block in block row `j > 0` and `ublk[j]` is the superdiagonal
/// block in block row `j < nz - 1`.
///
/// # Errors
///
/// Returns [`MtxError::DimensionMismatch`] on mismatched block or output slice
/// lengths, or [`MtxError::NonFiniteInput`] if any block contains `NaN`/`Inf`.
pub fn compute_block_preconditioner(
    nvar: usize,
    nz: usize,
    lblk: &[f64],
    dblk: &[f64],
    ublk: &[f64],
    pre_conditioner: &mut [f64],
) -> Result<(), MtxError> {
    let blk_sz = check_blocks(nvar, nz, lblk, dblk, ublk)?;
    if pre_conditioner.len() != nz * nvar {
        return Err(MtxError::DimensionMismatch(
            "pre_conditioner length must equal nz * nvar",
        ));
    }
    if is_bad_slice(lblk) || is_bad_slice(dblk) || is_bad_slice(ublk) {
        return Err(MtxError::NonFiniteInput(
            "block arrays contain NaN/Inf in compute_block_preconditioner",
        ));
    }

    pre_conditioner.fill(0.0);
    for j in 0..nz {
        let blk_off = j * blk_sz;
        for k in 0..nvar {
            let row_off = blk_off + k * nvar;
            // `pre_conditioner(k,j) = pre_conditioner(k,j) + sum(abs(...))`
            // for `dblk`, then `ublk`, then `lblk`
            // (`mtx/private/pre_conditioners.f90:31-44`): each `sum` is its
            // own left-to-right reduction, added to the running value (#662).
            let row_sum = |blk: &[f64]| -> f64 {
                let mut s = 0.0_f64;
                for m in 0..nvar {
                    s += blk[row_off + m].abs();
                }
                s
            };
            let mut p = 0.0_f64;
            p += row_sum(dblk);
            if j + 1 < nz {
                p += row_sum(ublk);
            }
            if j > 0 {
                p += row_sum(lblk);
            }
            pre_conditioner[j * nvar + k] = p;
        }
    }
    Ok(())
}

/// Computes column (left-scaling of $x$) preconditioner factors for a
/// `(nz, nvar, nvar)` block-tridiagonal system given `right_pre_conditioner`
/// from [`compute_block_preconditioner`], matching upstream
/// `mtx/private/pre_conditioners.f90:60-87`.
///
/// # Errors
///
/// Returns [`MtxError::DimensionMismatch`] on mismatched lengths,
/// [`MtxError::NonFiniteInput`] on non-finite block/preconditioner entries, or
/// [`MtxError::SingularMatrix`] if any entry of `right_pre_conditioner` is `0`.
pub fn compute_block_left_preconditioner(
    nvar: usize,
    nz: usize,
    lblk: &[f64],
    dblk: &[f64],
    ublk: &[f64],
    right_pre_conditioner: &[f64],
    pre_conditioner: &mut [f64],
) -> Result<(), MtxError> {
    let blk_sz = check_blocks(nvar, nz, lblk, dblk, ublk)?;
    let n_eq = nz * nvar;
    if right_pre_conditioner.len() != n_eq || pre_conditioner.len() != n_eq {
        return Err(MtxError::DimensionMismatch(
            "preconditioner lengths must equal nz * nvar",
        ));
    }
    if is_bad_slice(lblk)
        || is_bad_slice(dblk)
        || is_bad_slice(ublk)
        || is_bad_slice(right_pre_conditioner)
    {
        return Err(MtxError::NonFiniteInput(
            "inputs contain NaN/Inf in compute_block_left_preconditioner",
        ));
    }
    if let Some((idx, _)) = right_pre_conditioner
        .iter()
        .enumerate()
        .find(|&(_, &v)| v == 0.0)
    {
        return Err(MtxError::SingularMatrix(idx));
    }

    pre_conditioner.fill(0.0);
    for j in 0..nz {
        let j_blk = j * blk_sz;
        for k in 0..nvar {
            let mut sum = 0.0_f64;
            for k2 in 0..nvar {
                sum += (dblk[j_blk + k2 * nvar + k]
                    / right_pre_conditioner[j * nvar + k2])
                    .abs();
                if j + 1 < nz {
                    sum += (lblk[(j + 1) * blk_sz + k2 * nvar + k]
                        / right_pre_conditioner[(j + 1) * nvar + k2])
                        .abs();
                }
                if j > 0 {
                    sum += (ublk[(j - 1) * blk_sz + k2 * nvar + k]
                        / right_pre_conditioner[(j - 1) * nvar + k2])
                        .abs();
                }
            }
            pre_conditioner[j * nvar + k] = sum;
        }
    }
    Ok(())
}

/// Solves a banded system $A x = b$ stored in left-aligned band-major layout
/// `bands` of shape `(ku + kl + 1, n)` using row preconditioning and `dgbtrf` /
/// `dgbtrs` / `rcond_banded`, matching upstream
/// `mtx/private/DGBSVX_wrapper.f90:24-126` (`DGBSVX_banded_wrapper`).
///
/// Upstream's lower-band loop `do j=1,matrix_size + n_lower_bands - i - 1`
/// (`DGBSVX_wrapper.f90:84-89`) reads `pre_conditioner(i+j)` past
/// `matrix_size` for every `kl >= 2` (and writes `AB` past it from
/// `kl >= 3`). Upstream bug, Deferred bug #695: for `kl >= 2` this returns
/// [`MtxError::InvalidParameter`] instead of the out-of-bounds access
/// (DECISIONS #43). For `kl <= 1` the loops match upstream exactly.
///
/// # Errors
///
/// Returns [`MtxError::InvalidParameter`] or [`MtxError::DimensionMismatch`] on
/// invalid shapes or for `kl >= 2` (#695), [`MtxError::NonFiniteInput`] on
/// non-finite inputs, or [`MtxError::SingularMatrix`] if any preconditioner
/// row is zero or LU encounters a zero pivot.
pub fn dgbsvx_banded_from_bands(
    n: usize,
    kl: usize,
    ku: usize,
    bands: &[f64],
    b: &[f64],
) -> Result<PreconditionedSolve, MtxError> {
    if n == 0 {
        return Err(MtxError::InvalidParameter(
            "n must be >= 1 in dgbsvx_banded_from_bands",
        ));
    }
    if kl >= 2 {
        // DGBSVX_wrapper.f90:84-89 reads pre_conditioner(i+j) out of bounds
        // (Deferred bug #695, DECISIONS #43).
        return Err(MtxError::InvalidParameter(
            "dgbsvx_banded_from_bands: upstream DGBSVX_wrapper.f90:86-87 reads \
             past matrix_size for kl >= 2 (Deferred bug #695)",
        ));
    }
    if b.len() != n {
        return Err(MtxError::DimensionMismatch(
            "rhs b length mismatch in dgbsvx_banded_from_bands",
        ));
    }
    if is_bad_slice(b) {
        return Err(MtxError::NonFiniteInput(
            "rhs b contains NaN/Inf in dgbsvx_banded_from_bands",
        ));
    }

    let mut pc = vec![0.0_f64; n];
    compute_band_preconditioner(n, kl, ku, bands, &mut pc)?;
    if let Some((idx, _)) = pc.iter().enumerate().find(|&(_, &v)| v == 0.0) {
        return Err(MtxError::SingularMatrix(idx));
    }

    let mut x = vec![0.0_f64; n];
    for i in 0..n {
        x[i] = b[i] / pc[i];
    }

    let ldab = 2 * kl + ku + 1;
    let diag = ldab - 1 - kl;
    let mut ab_orig = vec![0.0_f64; ldab * n];

    // Upper bands: superdiagonal d = ku - band_row, entry (row = j, col = j + d).
    for b_row in 0..ku {
        let d = ku - b_row;
        let in_off = b_row * n;
        for row in 0..n.saturating_sub(d) {
            let col = row + d;
            ab_orig[(diag - d) * n + col] = bands[in_off + row] / pc[row];
        }
    }

    // Diagonal: (row = i, col = i).
    let diag_in_off = ku * n;
    for i in 0..n {
        ab_orig[diag * n + i] = bands[diag_in_off + i] / pc[i];
    }

    // Lower bands: subdiagonal d = l + 1, entry (row = col + d, col = col).
    // For kl <= 1 the column bound n - d is upstream's (DGBSVX_wrapper.f90:86);
    // kl >= 2 returned above (#695).
    for l in 0..kl {
        let d = l + 1;
        let in_off = (ku + d) * n;
        for col in 0..n.saturating_sub(d) {
            let row = col + d;
            ab_orig[(diag + d) * n + col] = bands[in_off + col] / pc[row];
        }
    }

    let mut lu = ab_orig.clone();
    let mut ipiv = vec![0usize; n];
    dgbtrf(n, kl, ku, &mut lu, ldab, &mut ipiv)?;
    dgbtrs(n, kl, ku, &lu, ldab, &ipiv, &mut x)?;
    let rcond = rcond_banded(n, kl, ku, &ab_orig, &lu, ldab, &ipiv)?;

    Ok(PreconditionedSolve {
        x,
        rcond,
        row_scale: pc,
        col_scale: vec![1.0; n],
        ierr: 0,
    })
}

/// Solves a banded system $A x = b$ stored in `band.rs` `(ldab, n)` layout
/// (`ldab >= 2 * kl + ku + 1`, diagonal at `row = ldab - 1 - kl`) using
/// row-max preconditioning, LU factorization, and `rcond_banded`.
///
/// # Errors
///
/// Returns [`MtxError::InvalidParameter`] or [`MtxError::DimensionMismatch`] on
/// invalid shapes, [`MtxError::NonFiniteInput`] on non-finite inputs, or
/// [`MtxError::SingularMatrix`] on zero preconditioner row or zero LU pivot.
pub fn dgbsvx_banded(
    n: usize,
    kl: usize,
    ku: usize,
    ab: &[f64],
    ldab: usize,
    b: &[f64],
) -> Result<PreconditionedSolve, MtxError> {
    if n == 0 || ldab < 2 * kl + ku + 1 {
        return Err(MtxError::InvalidParameter(
            "n must be >= 1 and ldab must be >= 2 * kl + ku + 1",
        ));
    }
    if ab.len() < ldab * n || b.len() != n {
        return Err(MtxError::DimensionMismatch(
            "ab or b length mismatch in dgbsvx_banded",
        ));
    }
    if is_bad_slice(&ab[..(ldab * n)]) || is_bad_slice(b) {
        return Err(MtxError::NonFiniteInput(
            "ab or b contains NaN/Inf in dgbsvx_banded",
        ));
    }

    let diag = ldab - 1 - kl;
    let mut pc = vec![0.0_f64; n];
    for j in 0..n {
        let i_min = j.saturating_sub(ku);
        let i_max = (j + kl).min(n - 1);
        for i in i_min..=i_max {
            let row = (diag + i) - j;
            pc[i] = pc[i].max(ab[row * n + j].abs());
        }
    }
    if let Some((idx, _)) = pc.iter().enumerate().find(|&(_, &v)| v == 0.0) {
        return Err(MtxError::SingularMatrix(idx));
    }

    let mut ab_scaled = vec![0.0_f64; ldab * n];
    for j in 0..n {
        let i_min = j.saturating_sub(ku);
        let i_max = (j + kl).min(n - 1);
        for i in i_min..=i_max {
            let row = (diag + i) - j;
            ab_scaled[row * n + j] = ab[row * n + j] / pc[i];
        }
    }
    let mut x: Vec<f64> =
        b.iter().zip(pc.iter()).map(|(&bi, &pi)| bi / pi).collect();

    let mut lu = ab_scaled.clone();
    let mut ipiv = vec![0usize; n];
    dgbtrf(n, kl, ku, &mut lu, ldab, &mut ipiv)?;
    dgbtrs(n, kl, ku, &lu, ldab, &ipiv, &mut x)?;
    let rcond = rcond_banded(n, kl, ku, &ab_scaled, &lu, ldab, &ipiv)?;

    Ok(PreconditionedSolve {
        x,
        rcond,
        row_scale: pc,
        col_scale: vec![1.0; n],
        ierr: 0,
    })
}

/// Solves a `(nz, nvar, nvar)` block-tridiagonal system via two-sided
/// block row/column preconditioning and `DGBSVX(FACT = 'N')` with
/// `KL = KU = 2 * nvar`, matching upstream
/// `mtx/private/DGBSVX_wrapper.f90:140-292` (`DGBSVX_tridiagonal_wrapper`).
///
/// The banded matrix is built as `AB(LDAB,*)` with `LDAB = KL+KU+1`
/// (`DGBSVX_wrapper.f90:181, 205-244`) and solved by
/// [`crate::band_lapack::dgbsvx_fact_n`], which follows
/// `SRC/dgbsvx.f` (`DGBTRF`, `DGBCON`, `DGBTRS`, `DGBRFS`).
/// [`PreconditionedSolve::ierr`] is the wrapper's `ierr`: `N + 1` when
/// `DGBSVX` returns `INFO /= 0` (its `INFO = N + 1` for `rcond < eps`) or when
/// `rcond < 1d-10` (`DGBSVX_wrapper.f90:252-276`), else `0`; `x` is computed in
/// every such case, as upstream does. Upstream's side effects on that path
/// (messages and the `lower.data`, `upper.data`, `diagonal.data` dumps,
/// `DGBSVX_wrapper.f90:253-280`) are not reproduced.
///
/// Zero or non-finite preconditioner sums are not rejected here: upstream
/// divides by them and the resulting `Inf`/`NaN` band entries reach
/// `DGBTRF`. In this crate [`crate::band::dgbtrf`] still returns
/// [`MtxError::NonFiniteInput`] for them, and
/// [`compute_block_preconditioner`] / [`compute_block_left_preconditioner`]
/// keep their own input checks; neither is upstream's behaviour.
///
/// # Errors
///
/// Returns [`MtxError::DimensionMismatch`] on shape mismatches,
/// [`MtxError::NonFiniteInput`] on non-finite entries reaching a checked
/// routine, or [`MtxError::SingularMatrix`] when `DGBTRF` finds an exactly
/// zero pivot (upstream then reads `x_flat` unset, `DGBSVX_wrapper.f90:287`;
/// Deferred bug #752, DECISIONS #43).
pub fn dgbsvx_block_tridiagonal(
    nvar: usize,
    nz: usize,
    lblk: &[f64],
    dblk: &[f64],
    ublk: &[f64],
    b: &[f64],
) -> Result<PreconditionedSolve, MtxError> {
    let blk_sz = check_blocks(nvar, nz, lblk, dblk, ublk)?;
    let n = nz * nvar;
    if b.len() != n {
        return Err(MtxError::DimensionMismatch(
            "rhs b length must equal nz * nvar",
        ));
    }

    // DGBSVX_wrapper.f90:184-187
    let mut row_pc = vec![0.0_f64; n];
    compute_block_preconditioner(nvar, nz, lblk, dblk, ublk, &mut row_pc)?;
    let mut col_pc = vec![0.0_f64; n];
    compute_block_left_preconditioner(
        nvar,
        nz,
        lblk,
        dblk,
        ublk,
        &row_pc,
        &mut col_pc,
    )?;

    // KL = KU = 2*nvar, LDAB = KL+KU+1 (DGBSVX_wrapper.f90:170-181).
    let kl = 2 * nvar;
    let ku = 2 * nvar;
    let ldab = kl + ku + 1;
    // 0-based (row, col) of AB(KU+1+d, col+1) is ku + d.
    let mut ab = vec![0.0_f64; ldab * n];
    // DGBSVX_wrapper.f90:205-220 (dblk), :222-244 (lblk and ublk).
    for i in 0..nz {
        let blk_off = i * blk_sz;
        let eq_off = i * nvar;
        for r in 0..nvar {
            for c in 0..nvar {
                let col_m = eq_off + c;
                ab[(ku + r - c) + col_m * ldab] = dblk[blk_off + r * nvar + c]
                    / row_pc[eq_off + r]
                    / col_pc[col_m];
            }
        }
        if i + 1 < nz {
            let l_off = (i + 1) * blk_sz;
            let u_off = i * blk_sz;
            for r in 0..nvar {
                for c in 0..nvar {
                    let col_l = i * nvar + c;
                    let col_u = (i + 1) * nvar + c;
                    ab[(ku + nvar + r - c) + col_l * ldab] = lblk
                        [l_off + r * nvar + c]
                        / row_pc[(i + 1) * nvar + r]
                        / col_pc[col_l];
                    ab[(ku - nvar + r - c) + col_u * ldab] = ublk
                        [u_off + r * nvar + c]
                        / row_pc[i * nvar + r]
                        / col_pc[col_u];
                }
            }
        }
    }

    // DGBSVX_wrapper.f90:196-203: b_flat(i) = b(i) / pre_conditioner(i).
    let b_flat: Vec<f64> = b
        .iter()
        .zip(row_pc.iter())
        .map(|(&bi, &ri)| bi / ri)
        .collect();

    // DGBSVX_wrapper.f90:246-249
    let sol = dgbsvx_fact_n(b'N', n, kl, ku, &ab, ldab, &b_flat)?;
    // DGBSVX_wrapper.f90:252 and :276
    let ierr = if sol.info != 0 || sol.rcond < 1.0e-10 {
        n + 1
    } else {
        0
    };
    // DGBSVX_wrapper.f90:283-290: x = x_flat / left_pre_conditioner.
    let x: Vec<f64> = sol
        .x
        .iter()
        .zip(col_pc.iter())
        .map(|(&xi, &ci)| xi / ci)
        .collect();

    Ok(PreconditionedSolve {
        x,
        rcond: sol.rcond,
        row_scale: row_pc,
        col_scale: col_pc,
        ierr,
    })
}

/// Computes row and column equilibration scale factors for a row-major dense
/// $n \times n$ matrix `a`, matching LAPACK `DGEEQU` (used by `DGESVX(FACT='E')`
/// in `star/private/star_bcyclic.f90:639-652`).
///
/// # Errors
///
/// Returns [`MtxError::DimensionMismatch`] if `n == 0` or `a.len() != n * n`,
/// or [`MtxError::NonFiniteInput`] if `a` contains `NaN`/`Inf`.
pub fn dgeequ(n: usize, a: &[f64]) -> Result<EquilibrationFactors, MtxError> {
    if n == 0 || a.len() != n * n {
        return Err(MtxError::DimensionMismatch(
            "a must have length n * n with n >= 1 in dgeequ",
        ));
    }
    if is_bad_slice(a) {
        return Err(MtxError::NonFiniteInput("a contains NaN/Inf in dgeequ"));
    }

    let mut row_scale = vec![0.0_f64; n];
    for (i, ri) in row_scale.iter_mut().enumerate() {
        let mut rmax = 0.0_f64;
        for j in 0..n {
            rmax = rmax.max(a[i * n + j].abs());
        }
        *ri = rmax;
    }
    let (amax, rowcnd, row_zero) = finalize_scale_factors(&mut row_scale);
    if row_zero {
        return Ok(EquilibrationFactors {
            row_scale,
            col_scale: vec![1.0; n],
            rowcnd: 0.0,
            colcnd: 1.0,
            amax,
            equed: Equilibration::None,
        });
    }

    let mut col_scale = vec![0.0_f64; n];
    for j in 0..n {
        let mut cmax = 0.0_f64;
        for (i, &ri) in row_scale.iter().enumerate() {
            cmax = cmax.max((a[i * n + j] * ri).abs());
        }
        col_scale[j] = cmax;
    }
    let (_, colcnd, col_zero) = finalize_scale_factors(&mut col_scale);
    let equed = if col_zero {
        Equilibration::None
    } else {
        classify_equilibration(rowcnd, colcnd, amax)
    };

    Ok(EquilibrationFactors {
        row_scale,
        col_scale,
        rowcnd,
        colcnd,
        amax,
        equed,
    })
}

/// Applies the recommended row/column equilibration from `factors` in-place to
/// row-major dense $n \times n$ matrix `a`, matching LAPACK `DLAQGE`.
///
/// # Errors
///
/// Returns [`MtxError::DimensionMismatch`] on buffer length mismatch, or
/// [`MtxError::NonFiniteInput`] if `a` or scale factors contain `NaN`/`Inf`.
pub fn dlaqge(
    n: usize,
    a: &mut [f64],
    factors: &EquilibrationFactors,
) -> Result<Equilibration, MtxError> {
    if n == 0
        || a.len() != n * n
        || factors.row_scale.len() != n
        || factors.col_scale.len() != n
    {
        return Err(MtxError::DimensionMismatch(
            "matrix or scale length mismatch in dlaqge",
        ));
    }
    if is_bad_slice(a)
        || is_bad_slice(&factors.row_scale)
        || is_bad_slice(&factors.col_scale)
    {
        return Err(MtxError::NonFiniteInput("non-finite input in dlaqge"));
    }

    let equed = factors.equed;
    match equed {
        Equilibration::None => {}
        Equilibration::Row => {
            for i in 0..n {
                let ri = factors.row_scale[i];
                for j in 0..n {
                    a[i * n + j] *= ri;
                }
            }
        }
        Equilibration::Col => {
            for i in 0..n {
                for j in 0..n {
                    a[i * n + j] *= factors.col_scale[j];
                }
            }
        }
        Equilibration::Both => {
            for i in 0..n {
                let ri = factors.row_scale[i];
                for j in 0..n {
                    a[i * n + j] *= ri * factors.col_scale[j];
                }
            }
        }
    }
    Ok(equed)
}

/// Computes row and column equilibration scale factors for a banded matrix `ab`
/// in `band.rs` `(ldab, n)` layout (`ldab >= 2 * kl + ku + 1`), matching LAPACK
/// `DGBEQU`.
///
/// # Errors
///
/// Returns [`MtxError::InvalidParameter`] if `n == 0` or `ldab < 2 * kl + ku + 1`,
/// [`MtxError::DimensionMismatch`] if `ab.len() < ldab * n`, or
/// [`MtxError::NonFiniteInput`] if band entries contain `NaN`/`Inf`.
pub fn dgbequ(
    n: usize,
    kl: usize,
    ku: usize,
    ab: &[f64],
    ldab: usize,
) -> Result<EquilibrationFactors, MtxError> {
    if n == 0 || ldab < 2 * kl + ku + 1 {
        return Err(MtxError::InvalidParameter(
            "n must be >= 1 and ldab must be >= 2 * kl + ku + 1 in dgbequ",
        ));
    }
    if ab.len() < ldab * n {
        return Err(MtxError::DimensionMismatch(
            "ab length must be >= ldab * n in dgbequ",
        ));
    }
    if is_bad_slice(&ab[..(ldab * n)]) {
        return Err(MtxError::NonFiniteInput("ab contains NaN/Inf in dgbequ"));
    }

    let diag = ldab - 1 - kl;
    let mut row_scale = vec![0.0_f64; n];
    for j in 0..n {
        let i_min = j.saturating_sub(ku);
        let i_max = (j + kl).min(n - 1);
        for i in i_min..=i_max {
            let row = (diag + i) - j;
            row_scale[i] = row_scale[i].max(ab[row * n + j].abs());
        }
    }
    let (amax, rowcnd, row_zero) = finalize_scale_factors(&mut row_scale);
    if row_zero {
        return Ok(EquilibrationFactors {
            row_scale,
            col_scale: vec![1.0; n],
            rowcnd: 0.0,
            colcnd: 1.0,
            amax,
            equed: Equilibration::None,
        });
    }

    let mut col_scale = vec![0.0_f64; n];
    for j in 0..n {
        let i_min = j.saturating_sub(ku);
        let i_max = (j + kl).min(n - 1);
        let mut cmax = 0.0_f64;
        for i in i_min..=i_max {
            let row = (diag + i) - j;
            cmax = cmax.max((ab[row * n + j] * row_scale[i]).abs());
        }
        col_scale[j] = cmax;
    }
    let (_, colcnd, col_zero) = finalize_scale_factors(&mut col_scale);
    let equed = if col_zero {
        Equilibration::None
    } else {
        classify_equilibration(rowcnd, colcnd, amax)
    };

    Ok(EquilibrationFactors {
        row_scale,
        col_scale,
        rowcnd,
        colcnd,
        amax,
        equed,
    })
}

/// Applies the recommended row/column equilibration from `factors` in-place to
/// banded matrix `ab` (`band.rs` `(ldab, n)` layout), matching LAPACK `DLAQGB`.
///
/// # Errors
///
/// Returns [`MtxError::InvalidParameter`] or [`MtxError::DimensionMismatch`] on
/// invalid sizes, or [`MtxError::NonFiniteInput`] on non-finite inputs.
pub fn dlaqgb(
    n: usize,
    kl: usize,
    ku: usize,
    ab: &mut [f64],
    ldab: usize,
    factors: &EquilibrationFactors,
) -> Result<Equilibration, MtxError> {
    if n == 0 || ldab < 2 * kl + ku + 1 {
        return Err(MtxError::InvalidParameter(
            "n must be >= 1 and ldab must be >= 2 * kl + ku + 1 in dlaqgb",
        ));
    }
    if ab.len() < ldab * n
        || factors.row_scale.len() != n
        || factors.col_scale.len() != n
    {
        return Err(MtxError::DimensionMismatch(
            "ab or scale length mismatch in dlaqgb",
        ));
    }
    if is_bad_slice(&ab[..(ldab * n)])
        || is_bad_slice(&factors.row_scale)
        || is_bad_slice(&factors.col_scale)
    {
        return Err(MtxError::NonFiniteInput("non-finite input in dlaqgb"));
    }

    let diag = ldab - 1 - kl;
    let equed = factors.equed;
    match equed {
        Equilibration::None => {}
        Equilibration::Row => {
            for j in 0..n {
                for i in j.saturating_sub(ku)..=(j + kl).min(n - 1) {
                    let row = (diag + i) - j;
                    ab[row * n + j] *= factors.row_scale[i];
                }
            }
        }
        Equilibration::Col => {
            for j in 0..n {
                let cj = factors.col_scale[j];
                for i in j.saturating_sub(ku)..=(j + kl).min(n - 1) {
                    let row = (diag + i) - j;
                    ab[row * n + j] *= cj;
                }
            }
        }
        Equilibration::Both => {
            for j in 0..n {
                let cj = factors.col_scale[j];
                for i in j.saturating_sub(ku)..=(j + kl).min(n - 1) {
                    let row = (diag + i) - j;
                    ab[row * n + j] *= factors.row_scale[i] * cj;
                }
            }
        }
    }
    Ok(equed)
}

/// Applies row scaling `b_eq = D_r b` when equilibration scaled rows.
fn apply_row_equilibration(
    x: &mut [f64],
    factors: &EquilibrationFactors,
    equed: Equilibration,
) {
    if matches!(equed, Equilibration::Row | Equilibration::Both) {
        for (xi, &ri) in x.iter_mut().zip(factors.row_scale.iter()) {
            *xi *= ri;
        }
    }
}

/// Applies column post-scaling `x = D_c y` and packages `PreconditionedSolve`.
fn finish_equilibrated_solve(
    n: usize,
    mut x: Vec<f64>,
    rcond: f64,
    factors: EquilibrationFactors,
    equed: Equilibration,
) -> PreconditionedSolve {
    let row_scale = if matches!(equed, Equilibration::Row | Equilibration::Both)
    {
        factors.row_scale
    } else {
        vec![1.0; n]
    };
    let col_scale = if matches!(equed, Equilibration::Col | Equilibration::Both)
    {
        for (xi, &ci) in x.iter_mut().zip(factors.col_scale.iter()) {
            *xi *= ci;
        }
        factors.col_scale
    } else {
        vec![1.0; n]
    };
    PreconditionedSolve {
        x,
        rcond,
        row_scale,
        col_scale,
        ierr: 0,
    }
}

/// Solves dense system $A x = b$ with automatic LAPACK `DGEEQU`/`DLAQGE`
/// row/column equilibration and `rcond_dense` estimation (matching
/// `DGESVX(FACT='E')` in `star/private/star_bcyclic.f90:639-652,849-968`).
///
/// # Errors
///
/// Propagates dimension, non-finite, or singular errors from [`dgeequ`],
/// [`dlaqge`], [`dgetrf`], [`dgetrs`], or [`rcond_dense`].
pub fn solve_dense_equilibrated(
    n: usize,
    a: &[f64],
    b: &[f64],
) -> Result<PreconditionedSolve, MtxError> {
    if n == 0 || a.len() != n * n || b.len() != n {
        return Err(MtxError::DimensionMismatch(
            "a or b length mismatch in solve_dense_equilibrated",
        ));
    }
    if is_bad_slice(b) {
        return Err(MtxError::NonFiniteInput(
            "rhs b contains NaN/Inf in solve_dense_equilibrated",
        ));
    }

    let factors = dgeequ(n, a)?;
    let mut a_eq = a.to_vec();
    let equed = dlaqge(n, &mut a_eq, &factors)?;
    let mut x = b.to_vec();
    apply_row_equilibration(&mut x, &factors, equed);

    let mut lu = a_eq.clone();
    let mut ipiv = vec![0usize; n];
    dgetrf(n, &mut lu, &mut ipiv)?;
    dgetrs(n, &lu, &ipiv, &mut x)?;
    let rcond = rcond_dense(n, &a_eq, &lu, &ipiv)?;

    Ok(finish_equilibrated_solve(n, x, rcond, factors, equed))
}

/// Solves banded system $A x = b$ (`band.rs` `(ldab, n)` layout) with automatic
/// LAPACK `DGBEQU`/`DLAQGB` equilibration and `rcond_banded` estimation
/// (matching `DGBSVX(FACT='E')`).
///
/// # Errors
///
/// Propagates dimension, parameter, non-finite, or singular errors from
/// [`dgbequ`], [`dlaqgb`], [`dgbtrf`], [`dgbtrs`], or [`rcond_banded`].
pub fn solve_banded_equilibrated(
    n: usize,
    kl: usize,
    ku: usize,
    ab: &[f64],
    ldab: usize,
    b: &[f64],
) -> Result<PreconditionedSolve, MtxError> {
    if n == 0 || ldab < 2 * kl + ku + 1 {
        return Err(MtxError::InvalidParameter(
            "n must be >= 1 and ldab must be >= 2 * kl + ku + 1",
        ));
    }
    if ab.len() < ldab * n || b.len() != n {
        return Err(MtxError::DimensionMismatch(
            "ab or b length mismatch in solve_banded_equilibrated",
        ));
    }
    if is_bad_slice(b) {
        return Err(MtxError::NonFiniteInput(
            "rhs b contains NaN/Inf in solve_banded_equilibrated",
        ));
    }

    let factors = dgbequ(n, kl, ku, ab, ldab)?;
    let mut ab_eq = ab[..(ldab * n)].to_vec();
    let equed = dlaqgb(n, kl, ku, &mut ab_eq, ldab, &factors)?;
    let mut x = b.to_vec();
    apply_row_equilibration(&mut x, &factors, equed);

    let mut lu = ab_eq.clone();
    let mut ipiv = vec![0usize; n];
    dgbtrf(n, kl, ku, &mut lu, ldab, &mut ipiv)?;
    dgbtrs(n, kl, ku, &lu, ldab, &ipiv, &mut x)?;
    let rcond = rcond_banded(n, kl, ku, &ab_eq, &lu, ldab, &ipiv)?;

    Ok(finish_equilibrated_solve(n, x, rcond, factors, equed))
}
