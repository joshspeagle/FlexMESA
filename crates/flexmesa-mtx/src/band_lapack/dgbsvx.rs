// SPDX-License-Identifier: GPL-3.0-or-later
//! `DGBSVX` with `FACT = 'N'`: banded solve with condition estimate and
//! error bounds.

use super::dgbcon::dgbcon;
use super::dgbrfs::dgbrfs;
use super::dlangb::dlangb;
use super::dlantb::dlantb;
use crate::band::{dgbtrf, dgbtrs, dgbtrs_trans};
use crate::lapack::{dlamch, ix, lsame, LapackError};
use crate::MtxError;

impl From<LapackError> for MtxError {
    fn from(e: LapackError) -> Self {
        match e {
            LapackError::IllegalArgument { .. } => MtxError::InvalidParameter(
                "illegal argument to a LAPACK routine",
            ),
            LapackError::UnportedPath { .. } => MtxError::InvalidParameter(
                "LAPACK branch not ported in flexmesa_mtx::band_lapack",
            ),
        }
    }
}

/// Result of [`dgbsvx_fact_n`].
#[derive(Debug, Clone, PartialEq)]
pub struct DgbsvxSolution {
    /// Solution `X` (one right-hand side).
    pub x: Vec<f64>,
    /// `RCOND`: reciprocal condition number estimate from [`dgbcon`].
    pub rcond: f64,
    /// `WORK(1)`: reciprocal pivot growth factor.
    pub rpvgrw: f64,
    /// `FERR(1)`: forward error bound.
    pub ferr: f64,
    /// `BERR(1)`: backward error.
    pub berr: f64,
    /// `INFO`: `0`, or `n + 1` when `rcond < DLAMCH('E')`
    /// (`dgbsvx.f:631-632`).
    pub info: usize,
}

/// `DGBSVX(FACT = 'N', TRANS, N, KL, KU, NRHS = 1, ...)`
/// (`SRC/dgbsvx.f:366-635`) on column-major band storage.
///
/// `ab` is `AB(LDAB,*)` with `A(i,j)` at `AB(ku+1+i-j,j)`, `ldab = kl+ku+1`
/// (`LDAB` may be larger). `FACT = 'N'` forces `EQUED = 'N'`
/// (`dgbsvx.f:405-408`), so no scaling is applied. The factorisation,
/// `DGBTRS` and the solves inside [`dgbrfs`] are the crate's row-major
/// [`dgbtrf`]/[`dgbtrs`]/[`dgbtrs_trans`], reached through a transposed copy
/// of `AFB` (same row of the band, `ab[row*n + j]` against
/// `AFB(row+1, j+1)`), with `IPIV` 0-based there and 1-based for [`dgbcon`].
///
/// # Errors
///
/// - [`MtxError::SingularMatrix`] when `DGBTRF` reports `INFO = j > 0`
///   (an exactly zero pivot). Upstream then returns `RCOND = 0` with `X`
///   never computed (`dgbsvx.f:546-567`), which the wrapper reads unset
///   (`DGBSVX_wrapper.f90:287`); this explicit error stands in for that
///   read (Deferred bug #752, DECISIONS #43).
/// - [`MtxError::NonFiniteInput`] and the other errors of
///   [`dgbtrf`] / [`dgbtrs`] (their NaN/Inf and pivot checks are not
///   upstream's).
/// - [`MtxError::InvalidParameter`] / [`MtxError::DimensionMismatch`] for
///   bad shapes, and converted [`LapackError`]s.
pub fn dgbsvx_fact_n(
    trans: u8,
    n: usize,
    kl: usize,
    ku: usize,
    ab: &[f64],
    ldab: usize,
    b: &[f64],
) -> Result<DgbsvxSolution, MtxError> {
    if n == 0 || ldab < kl + ku + 1 {
        return Err(MtxError::InvalidParameter(
            "n must be >= 1 and ldab >= kl + ku + 1 in dgbsvx_fact_n",
        ));
    }
    if !lsame(trans, b'N') && !lsame(trans, b'T') && !lsame(trans, b'C') {
        return Err(LapackError::IllegalArgument {
            routine: "DGBSVX",
            info: -2,
        }
        .into());
    }
    if ab.len() < ldab * n || b.len() != n {
        return Err(MtxError::DimensionMismatch(
            "ab or b length mismatch in dgbsvx_fact_n",
        ));
    }
    let notran = lsame(trans, b'N');
    let ldafb = 2 * kl + ku + 1;

    // dgbsvx.f:531-541: AFB(KL+KU+1-J+J1, J) <- AB(KU+1-J+J1, J).
    let mut afb = vec![0.0_f64; ldafb * n];
    for j in 1..=n {
        let j1 = j.saturating_sub(ku).max(1);
        let j2 = (j + kl).min(n);
        for t in 0..=(j2 - j1) {
            afb[ix(kl + ku + 1 + j1 - j + t, j, ldafb)] =
                ab[ix(ku + 1 + j1 - j + t, j, ldab)];
        }
    }
    // dgbsvx.f:542: DGBTRF through the row-major factoriser.
    let mut rm = vec![0.0_f64; ldafb * n];
    for r in 0..ldafb {
        for j in 0..n {
            rm[r * n + j] = afb[r + j * ldafb];
        }
    }
    let mut ipiv0 = vec![0usize; n];
    dgbtrf(n, kl, ku, &mut rm, ldafb, &mut ipiv0)?;
    for r in 0..ldafb {
        for j in 0..n {
            afb[r + j * ldafb] = rm[r * n + j];
        }
    }
    let ipiv1: Vec<usize> = ipiv0.iter().map(|&p| p + 1).collect();

    // dgbsvx.f:574-593
    let norm = if notran { b'1' } else { b'I' };
    let mut work = vec![0.0_f64; n];
    let anorm = dlangb(norm, n, kl, ku, ab, ldab, &mut work)?;
    let mut rpvgrw =
        dlantb(b'M', b'U', b'N', n, kl + ku, &afb, ldafb, &mut work)?;
    if rpvgrw == 0.0 {
        rpvgrw = 1.0;
    } else {
        rpvgrw = dlangb(b'M', n, kl, ku, ab, ldab, &mut work)? / rpvgrw;
    }
    let rcond = dgbcon(norm, n, kl, ku, &afb, ldafb, &ipiv1, anorm)?;
    // dgbsvx.f:594-596: DLACPY then DGBTRS.
    let mut x = b.to_vec();
    if notran {
        dgbtrs(n, kl, ku, &rm, ldafb, &ipiv0, &mut x)?;
    } else {
        dgbtrs_trans(true, n, kl, ku, &rm, ldafb, &ipiv0, &mut x)?;
    }
    // dgbsvx.f:601-603: DGBRFS.
    let mut solve = |tr: u8, v: &mut [f64]| -> Result<(), MtxError> {
        if lsame(tr, b'N') {
            dgbtrs(n, kl, ku, &rm, ldafb, &ipiv0, v)
        } else {
            dgbtrs_trans(true, n, kl, ku, &rm, ldafb, &ipiv0, v)
        }
    };
    let ref_out = dgbrfs(trans, n, kl, ku, ab, ldab, b, &mut x, &mut solve)?;
    // dgbsvx.f:631-632 (EQUED = 'N': no scaling of X or FERR).
    let info = if rcond < dlamch::EPS { n + 1 } else { 0 };
    Ok(DgbsvxSolution {
        x,
        rcond,
        rpvgrw,
        ferr: ref_out.ferr,
        berr: ref_out.berr,
        info,
    })
}
