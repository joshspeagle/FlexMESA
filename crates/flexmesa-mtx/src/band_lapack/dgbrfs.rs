// SPDX-License-Identifier: GPL-3.0-or-later
//! `DGBRFS`: iterative refinement and error bounds for a banded solve.

use super::dgbmv::dgbmv;
use crate::lapack::aux::{dlacn2, fmax, Lacn2State};
use crate::lapack::blas::{daxpy, dcopy};
use crate::lapack::{dlamch, ix, lsame, LapackError};

/// Maximum number of refinement steps (`dgbrfs.f:223-224`).
const ITMAX: usize = 5;

/// Forward and backward error bounds returned by [`dgbrfs`] for one
/// right-hand side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Refinement {
    /// `FERR(1)`: estimated forward error bound.
    pub ferr: f64,
    /// `BERR(1)`: componentwise relative backward error.
    pub berr: f64,
}

/// A `DGBTRS(trans, N, KL, KU, 1, AFB, LDAFB, IPIV, v, N, INFO)` stand-in.
pub type SolveFn<'a, E> = dyn FnMut(u8, &mut [f64]) -> Result<(), E> + 'a;

fn bad(info: i32) -> LapackError {
    LapackError::IllegalArgument {
        routine: "DGBRFS",
        info,
    }
}

/// `DGBRFS` (`SRC/dgbrfs.f:202-461`) for one right-hand side (`NRHS = 1`):
/// improve the solution `x` of `op(A)*x = b` and bound its error.
///
/// `ab` is `AB(LDAB,*)` of the original matrix (`ldab >= kl+ku+1`). The
/// factored matrix `AFB` and `IPIV` of the Fortran enter only through
/// `solve`, which must be `CALL DGBTRS(trans, N, KL, KU, 1, AFB, LDAFB,
/// IPIV, v, N, INFO)` on the vector `v` (`dgbrfs.f:381-382,428-429,440-441`).
///
/// # Errors
///
/// [`LapackError::IllegalArgument`] with upstream's `INFO`
/// (`dgbrfs.f:258-283`), and any error from `solve` (converted with `From`).
pub fn dgbrfs<E: From<LapackError>>(
    trans: u8,
    n: usize,
    kl: usize,
    ku: usize,
    ab: &[f64],
    ldab: usize,
    b: &[f64],
    x: &mut [f64],
    solve: &mut SolveFn<'_, E>,
) -> Result<Refinement, E> {
    // dgbrfs.f:258-283
    let notran = lsame(trans, b'N');
    if !notran && !lsame(trans, b'T') && !lsame(trans, b'C') {
        return Err(bad(-1).into());
    } else if ldab < kl + ku + 1 {
        return Err(bad(-7).into());
    }
    // dgbrfs.f:287-293 (NRHS = 1)
    if n == 0 {
        return Ok(Refinement {
            ferr: 0.0,
            berr: 0.0,
        });
    }
    // dgbrfs.f:295-299
    let transt = if notran { b'T' } else { b'N' };
    // dgbrfs.f:303-307
    let nz = (kl + ku + 2).min(n + 1);
    let eps = dlamch::EPS;
    let safmin = dlamch::SFMIN;
    let safe1 = nz as f64 * safmin;
    let safe2 = safe1 / eps;

    // WORK(1:N) = work1, WORK(N+1:2N) = work2, WORK(2N+1:3N) = work3
    let mut work1 = vec![0.0_f64; n];
    let mut work2 = vec![0.0_f64; n];
    let mut work3 = vec![0.0_f64; n];

    // dgbrfs.f:313-315
    let mut count = 1_usize;
    let mut lstres = 3.0_f64;
    let berr;
    loop {
        // Loop 20: residual R = B - op(A)*X into WORK(N+1) (dgbrfs.f:322-324).
        dcopy(n, b, 1, &mut work2, 1);
        dgbmv(
            trans, n, n, kl, ku, -1.0, ab, ldab, x, 1, 1.0, &mut work2, 1,
        )?;
        // |op(A)|*|X| + |B| into WORK(1) (dgbrfs.f:335-358).
        for i in 0..n {
            work1[i] = b[i].abs();
        }
        if notran {
            for k in 1..=n {
                let xk = x[k - 1].abs();
                for i in 1.max(k.saturating_sub(ku))..=n.min(k + kl) {
                    // KK = KU + 1 - K, AB( KK+I, K )
                    work1[i - 1] += ab[ix(ku + 1 + i - k, k, ldab)].abs() * xk;
                }
            }
        } else {
            for k in 1..=n {
                let mut s = 0.0_f64;
                for i in 1.max(k.saturating_sub(ku))..=n.min(k + kl) {
                    s += ab[ix(ku + 1 + i - k, k, ldab)].abs() * x[i - 1].abs();
                }
                work1[k - 1] += s;
            }
        }
        // dgbrfs.f:359-368
        let mut s = 0.0_f64;
        for i in 0..n {
            if work1[i] > safe2 {
                s = fmax(s, work2[i].abs() / work1[i]);
            } else {
                s = fmax(s, (work2[i].abs() + safe1) / (work1[i] + safe1));
            }
        }
        let bj = s;
        // dgbrfs.f:376-387
        if bj > eps && 2.0 * bj <= lstres && count <= ITMAX {
            solve(trans, &mut work2)?;
            daxpy(n, 1.0, &work2, 1, x, 1);
            lstres = bj;
            count += 1;
            continue;
        }
        berr = bj;
        break;
    }

    // dgbrfs.f:411-417
    for i in 0..n {
        if work1[i] > safe2 {
            work1[i] = work2[i].abs() + nz as f64 * eps * work1[i];
        } else {
            work1[i] = work2[i].abs() + nz as f64 * eps * work1[i] + safe1;
        }
    }
    // dgbrfs.f:419-444
    let mut ferr = 0.0_f64;
    let mut st = Lacn2State::default();
    loop {
        dlacn2(n, &mut work3, &mut work2, &mut ferr, &mut st);
        if st.kase == 0 {
            break;
        }
        if st.kase == 1 {
            // Multiply by diag(W)*inv(op(A)**T) (dgbrfs.f:428-432).
            solve(transt, &mut work2)?;
            for i in 0..n {
                work2[i] *= work1[i];
            }
        } else {
            // Multiply by inv(op(A))*diag(W) (dgbrfs.f:437-441).
            for i in 0..n {
                work2[i] *= work1[i];
            }
            solve(trans, &mut work2)?;
        }
    }
    // dgbrfs.f:448-453
    let mut lstres = 0.0_f64;
    for xi in x.iter().take(n) {
        lstres = fmax(lstres, xi.abs());
    }
    if lstres != 0.0 {
        ferr /= lstres;
    }
    Ok(Refinement { ferr, berr })
}
