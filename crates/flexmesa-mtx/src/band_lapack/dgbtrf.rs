// SPDX-License-Identifier: GPL-3.0-or-later
//! `DGBTRF` block-size choice and blocked code (`SRC/dgbtrf.f:217-506`) on
//! column-major band storage.
//!
//! The unblocked code `DGBTF2` that `DGBTRF` calls for small bands
//! (`dgbtrf.f:225-229`) is [`crate::band::dgbtrf`]'s own loop; that function
//! makes the choice below and calls [`dgbtrf_blocked`] otherwise.

use crate::lapack::aux::dlaswp;
use crate::lapack::blas::{dcopy, dgemm, dger, dscal, dswap_in, dtrsm, idamax};
use crate::lapack::{ix, LapackError};

/// `NBMAX` (`dgbtrf.f:160-161`): column count of `WORK13` / `WORK31`.
const NBMAX: usize = 64;
/// `LDWORK = NBMAX+1` (`dgbtrf.f:160-161`).
const LDWORK: usize = NBMAX + 1;

/// `NB` of `DGBTRF(M, N, KL, KU, ...)`: `ILAENV(1, 'DGBTRF', ' ', M, N, KL,
/// KU)` (`dgbtrf.f:218`), which is `1` for `N4 = KU <= 64` and `32`
/// otherwise (`SRC/ilaenv.f:433-446`; the single- and double-precision
/// branches agree), capped at `NBMAX = 64` (`dgbtrf.f:223`).
#[must_use]
pub const fn dgbtrf_block_size(ku: usize) -> usize {
    let nb = if ku <= 64 { 1 } else { 32 };
    if nb < NBMAX {
        nb
    } else {
        NBMAX
    }
}

/// Whether `DGBTRF` takes its blocked code: not `NB <= 1 .OR. NB > KL`
/// (`dgbtrf.f:225`), i.e. `KU > 64` and `KL >= 32`.
#[must_use]
pub const fn dgbtrf_uses_blocked_code(kl: usize, ku: usize) -> bool {
    let nb = dgbtrf_block_size(ku);
    !(nb <= 1 || nb > kl)
}

/// Copies the `m`-by-`n` sub-matrix starting at `a[base]` with leading
/// dimension `ld` into a dense column-major buffer with leading dimension
/// `m`. Used for the read-only BLAS operands that live in `AB` next to the
/// output: Fortran passes those regions as separate, non-overlapping
/// arguments, so a copy reads the same values.
fn submatrix(
    a: &[f64],
    base: usize,
    ld: usize,
    m: usize,
    n: usize,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; m * n];
    for c in 0..n {
        for r in 0..m {
            out[r + c * m] = a[base + r + c * ld];
        }
    }
    out
}

/// Blocked `DGBTRF` (`SRC/dgbtrf.f:230-506`, the `ELSE` branch of
/// `dgbtrf.f:225`): LU factorisation with partial pivoting of the `m`-by-`n`
/// band matrix in `AB(LDAB,*)` (`ab[ix(i, j, ldab)]`, `A(i,j)` at
/// `AB(KL+KU+1+i-j, j)`), rows `1..KL` being fill-in workspace.
///
/// `ipiv` receives the 1-based `IPIV`. Returns `INFO`: `0`, or the first
/// `j` with `U(j,j) = 0` (`dgbtrf.f:339-346`); as upstream, the
/// factorisation is completed in that case.
///
/// The caller must have checked the arguments (`dgbtrf.f:195-214`) and that
/// [`dgbtrf_uses_blocked_code`] holds; `nb` is [`dgbtrf_block_size`].
///
/// # Errors
///
/// Propagates [`LapackError`] from the BLAS calls (not reachable for valid
/// arguments).
#[allow(clippy::many_single_char_names)]
pub fn dgbtrf_blocked(
    m: usize,
    n: usize,
    kl: usize,
    ku: usize,
    ab: &mut [f64],
    ldab: usize,
    ipiv: &mut [usize],
) -> Result<usize, LapackError> {
    // dgbtrf.f:191
    let kv = ku + kl;
    let nb = dgbtrf_block_size(ku);
    let mut info = 0usize;
    let p = |r: usize, c: usize| ix(r, c, ldab);
    let ldm1 = ldab - 1;
    let mut work13 = vec![0.0_f64; LDWORK * NBMAX];
    let mut work31 = vec![0.0_f64; LDWORK * NBMAX];

    // dgbtrf.f:236-240: zero the superdiagonal elements of WORK13.
    for j in 1..=nb {
        for i in 1..j {
            work13[ix(i, j, LDWORK)] = 0.0;
        }
    }
    // dgbtrf.f:244-248: zero the subdiagonal elements of WORK31.
    for j in 1..=nb {
        for i in (j + 1)..=nb {
            work31[ix(i, j, LDWORK)] = 0.0;
        }
    }
    // dgbtrf.f:254-259: fill-in elements in columns KU+2 to KV.
    for j in (ku + 2)..=kv.min(n) {
        for i in (kv + 2 - j)..=kl {
            ab[p(i, j)] = 0.0;
        }
    }
    // dgbtrf.f:263
    let mut ju = 1usize;

    let mn = m.min(n);
    // dgbtrf.f:265 (DO 180 J = 1, MIN(M,N), NB)
    for j in (1..=mn).step_by(nb) {
        // dgbtrf.f:266-281
        let jb = nb.min(mn - j + 1);
        let i2 = (kl - jb).min(m + 1 - j - jb);
        let i3 = (jb as isize).min(m as isize - j as isize - kl as isize + 1);

        // dgbtrf.f:287 (DO 80 JJ = J, J+JB-1)
        for jj in j..j + jb {
            // dgbtrf.f:291-295
            if jj + kv <= n {
                for i in 1..=kl {
                    ab[p(i, jj + kv)] = 0.0;
                }
            }
            // dgbtrf.f:300-302
            let km = kl.min(m - jj);
            let jp = idamax(km + 1, &ab[p(kv + 1, jj)..], 1);
            ipiv[jj - 1] = jp + jj - j;
            // dgbtrf.f:303
            if ab[p(kv + jp, jj)] != 0.0 {
                // dgbtrf.f:304
                ju = ju.max((jj + ku + jp - 1).min(n));
                // dgbtrf.f:305-323
                if jp != 1 {
                    if jp + jj - 1 < j + kl {
                        // dgbtrf.f:311-312
                        dswap_in(
                            jb,
                            ab,
                            p(kv + 1 + jj - j, j),
                            ldm1,
                            p(kv + jp + jj - j, j),
                            ldm1,
                        );
                    } else {
                        // dgbtrf.f:318-321: columns J..JJ-1 of A31 are in
                        // WORK31.
                        let ox = p(kv + 1 + jj - j, j);
                        let oy = ix(jp + jj - j - kl, 1, LDWORK);
                        for t in 0..(jj - j) {
                            std::mem::swap(
                                &mut ab[ox + t * ldm1],
                                &mut work31[oy + t * LDWORK],
                            );
                        }
                        dswap_in(
                            j + jb - jj,
                            ab,
                            p(kv + 1, jj),
                            ldm1,
                            p(kv + jp, jj),
                            ldm1,
                        );
                    }
                }
                // dgbtrf.f:327-328
                let da = 1.0 / ab[p(kv + 1, jj)];
                dscal(km, da, &mut ab[p(kv + 2, jj)..], 1);
                // dgbtrf.f:334-338
                let jm = ju.min(j + jb - 1);
                if jm > jj {
                    let x = submatrix(ab, p(kv + 2, jj), 1, km, 1);
                    let y = submatrix(ab, p(kv, jj + 1), ldm1, 1, jm - jj);
                    dger(
                        km,
                        jm - jj,
                        -1.0,
                        &x,
                        1,
                        &y,
                        1,
                        &mut ab[p(kv + 1, jj + 1)..],
                        ldm1,
                    )?;
                }
            } else if info == 0 {
                // dgbtrf.f:344-345
                info = jj;
            }
            // dgbtrf.f:350-353
            let nw = ((jj - j + 1) as isize).min(i3);
            if nw > 0 {
                let src = p(kv + kl + 1 + j - jj, jj);
                dcopy(
                    nw as usize,
                    &ab[src..],
                    1,
                    &mut work31[ix(1, jj - j + 1, LDWORK)..],
                    1,
                );
            }
        }
        // dgbtrf.f:355
        if j + jb <= n {
            // dgbtrf.f:359-360
            // Signed as in Fortran: JU-J+1 is <= 0 when every pivot so far
            // was zero (JU stays at its initial value).
            let j2 =
                (ju as isize - j as isize + 1).min(kv as isize) - jb as isize;
            let j3 = (ju as isize - j as isize - kv as isize + 1).max(0);
            // dgbtrf.f:365-366
            if j2 > 0 {
                dlaswp(
                    j2 as usize,
                    &mut ab[p(kv + 1 - jb, j + jb)..],
                    ldm1,
                    1,
                    jb,
                    &ipiv[j - 1..],
                    1,
                );
            }
            // dgbtrf.f:370-372
            for i in j..j + jb {
                ipiv[i - 1] += j - 1;
            }
            // dgbtrf.f:377-391: interchanges in A13, A23 and A33.
            let k2 = j as isize - 1 + jb as isize + j2;
            for i in 1..=j3 {
                let jj = (k2 + i) as usize;
                // DO 100 II = J+I-1, J+JB-1
                for ii in (j + i as usize - 1)..(j + jb) {
                    let ip = ipiv[ii - 1];
                    if ip != ii {
                        let ra = p(kv + 1 + ii - jj, jj);
                        let rb = p(kv + 1 + ip - jj, jj);
                        ab.swap(ra, rb);
                    }
                }
            }
            // dgbtrf.f:392-419
            if j2 > 0 {
                let j2 = j2 as usize;
                // dgbtrf.f:396-398: update A12.
                let l11 = submatrix(ab, p(kv + 1, j), ldm1, jb, jb);
                dtrsm(
                    b'L',
                    b'L',
                    b'N',
                    b'U',
                    jb,
                    j2,
                    1.0,
                    &l11,
                    jb,
                    &mut ab[p(kv + 1 - jb, j + jb)..],
                    ldm1,
                )?;
                let a12 = submatrix(ab, p(kv + 1 - jb, j + jb), ldm1, jb, j2);
                if i2 > 0 {
                    // dgbtrf.f:404-407: update A22.
                    let a21 = submatrix(ab, p(kv + 1 + jb, j), ldm1, i2, jb);
                    dgemm(
                        b'N',
                        b'N',
                        i2,
                        j2,
                        jb,
                        -1.0,
                        &a21,
                        i2,
                        &a12,
                        jb,
                        1.0,
                        &mut ab[p(kv + 1, j + jb)..],
                        ldm1,
                    )?;
                }
                if i3 > 0 {
                    // dgbtrf.f:414-417: update A32.
                    dgemm(
                        b'N',
                        b'N',
                        i3 as usize,
                        j2,
                        jb,
                        -1.0,
                        &work31,
                        LDWORK,
                        &a12,
                        jb,
                        1.0,
                        &mut ab[p(kv + kl + 1 - jb, j + jb)..],
                        ldm1,
                    )?;
                }
            }
            // dgbtrf.f:421-464
            if j3 > 0 {
                let j3 = j3 as usize;
                // dgbtrf.f:426-430: lower triangle of A13 into WORK13.
                for jj in 1..=j3 {
                    for ii in jj..=jb {
                        work13[ix(ii, jj, LDWORK)] =
                            ab[p(ii + 1 - jj, jj + j + kv - 1)];
                    }
                }
                // dgbtrf.f:434-436: update A13 in WORK13.
                let l11 = submatrix(ab, p(kv + 1, j), ldm1, jb, jb);
                dtrsm(
                    b'L',
                    b'L',
                    b'N',
                    b'U',
                    jb,
                    j3,
                    1.0,
                    &l11,
                    jb,
                    &mut work13,
                    LDWORK,
                )?;
                if i2 > 0 {
                    // dgbtrf.f:442-445: update A23.
                    let a21 = submatrix(ab, p(kv + 1 + jb, j), ldm1, i2, jb);
                    dgemm(
                        b'N',
                        b'N',
                        i2,
                        j3,
                        jb,
                        -1.0,
                        &a21,
                        i2,
                        &work13,
                        LDWORK,
                        1.0,
                        &mut ab[p(1 + jb, j + kv)..],
                        ldm1,
                    )?;
                }
                if i3 > 0 {
                    // dgbtrf.f:452-454: update A33.
                    dgemm(
                        b'N',
                        b'N',
                        i3 as usize,
                        j3,
                        jb,
                        -1.0,
                        &work31,
                        LDWORK,
                        &work13,
                        LDWORK,
                        1.0,
                        &mut ab[p(1 + kl, j + kv)..],
                        ldm1,
                    )?;
                }
                // dgbtrf.f:459-463: lower triangle of A13 back into place.
                for jj in 1..=j3 {
                    for ii in jj..=jb {
                        ab[p(ii + 1 - jj, jj + j + kv - 1)] =
                            work13[ix(ii, jj, LDWORK)];
                    }
                }
            }
        } else {
            // dgbtrf.f:469-471
            for i in j..j + jb {
                ipiv[i - 1] += j - 1;
            }
        }
        // dgbtrf.f:478-505: partially undo the interchanges in the current
        // block and copy the upper triangle of A31 back into place.
        for jj in (j..j + jb).rev() {
            let jp = ipiv[jj - 1] + 1 - jj;
            if jp != 1 {
                if jp + jj - 1 < j + kl {
                    // dgbtrf.f:488-489
                    dswap_in(
                        jj - j,
                        ab,
                        p(kv + 1 + jj - j, j),
                        ldm1,
                        p(kv + jp + jj - j, j),
                        ldm1,
                    );
                } else {
                    // dgbtrf.f:494-495
                    let ox = p(kv + 1 + jj - j, j);
                    let oy = ix(jp + jj - j - kl, 1, LDWORK);
                    for t in 0..(jj - j) {
                        std::mem::swap(
                            &mut ab[ox + t * ldm1],
                            &mut work31[oy + t * LDWORK],
                        );
                    }
                }
            }
            // dgbtrf.f:501-504
            let nw = i3.min((jj - j + 1) as isize);
            if nw > 0 {
                let dst = p(kv + kl + 1 + j - jj, jj);
                dcopy(
                    nw as usize,
                    &work31[ix(1, jj - j + 1, LDWORK)..],
                    1,
                    &mut ab[dst..],
                    1,
                );
            }
        }
    }
    Ok(info)
}
