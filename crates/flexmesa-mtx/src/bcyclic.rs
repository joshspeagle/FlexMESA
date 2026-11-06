//! Odd-even parallel block cyclic reduction (`bcyclic`) for block-tridiagonal
//! linear systems (Hirshman et al. 2010, *J. Comput. Phys.* 229, 8392).
//!
//! Solves a coupled $N_z$-shell block-tridiagonal system with $N_{\mathrm{var}}
//! \times N_{\mathrm{var}}$ blocks:
//! $$L_k x_{k-1} + D_k x_k + U_k x_{k+1} = b_k, \qquad k \in 0..N_z,$$
//! in $\lceil \log_2 N_z \rceil$ odd-even reduction levels rather than $N_z$
//! sequential Thomas elimination steps.
//!
//! # Algorithm
//!
//! At reduction level $s \ge 0$ with stride $p = 2^s$:
//! 1. **Odd-Row Normalization**: For every active odd multiple $j = (2m + 1)p <
//!    N_z$, factor $D_j^{(s)} = P_j^{-1} L_j^{(D)} U_j^{(D)}$ and normalize the
//!    off-diagonal blocks and right-hand side in place via $\tilde{L}_j =
//!    (D_j^{(s)})^{-1} L_j^{(s)}$, $\tilde{U}_j = (D_j^{(s)})^{-1} U_j^{(s)}$,
//!    and $\tilde{b}_j = (D_j^{(s)})^{-1} b_j^{(s)}$.
//! 2. **Even-Row Schur Update**: Substitute $x_j = \tilde{b}_j - \tilde{L}_j
//!    x_{j-p} - \tilde{U}_j x_{j+p}$ into every active even multiple
//!    $k = 2m p < N_z$ to obtain $D_k^{(s+1)} = D_k^{(s)} - L_k^{(s)}
//!    \tilde{U}_{k-p} - U_k^{(s)} \tilde{L}_{k+p}$, $L_k^{(s+1)} = -L_k^{(s)}
//!    \tilde{L}_{k-p}$, $U_k^{(s+1)} = -U_k^{(s)} \tilde{U}_{k+p}$, and
//!    $b_k^{(s+1)} = b_k^{(s)} - L_k^{(s)} \tilde{b}_{k-p} - U_k^{(s)}
//!    \tilde{b}_{k+p}$.
//! 3. **Root Solve & Backward Substitution**: Once $2^s \ge N_z$, solve row
//!    $k = 0$ via $x_0 = \left(D_0^{(s)}\right)^{-1} b_0^{(s)}$, then traverse
//!    levels $s = S-1 \dots 0$ in reverse to recover all odd-multiple shells:
//!    $$x_j = \tilde{b}_j - \tilde{L}_j x_{j-p} - \tilde{U}_j x_{j+p}.$$

use rayon::prelude::*;

use crate::block_tridiag::check_blocks;
use crate::dense::{dgetrf, dgetrs, dgetrs_multi};
use crate::MtxError;
use flexmesa_utils::is_bad_slice;

/// Subtract a block product $C \leftarrow C - A B$ for $n \times n$ row-major
/// blocks.
#[inline]
fn sub_block_mm(n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    for i in 0..n {
        for m in 0..n {
            let a_im = a[i * n + m];
            if a_im != 0.0 {
                for j in 0..n {
                    c[i * n + j] -= a_im * b[m * n + j];
                }
            }
        }
    }
}

/// Negated block product $C \leftarrow -A B$ for $n \times n$ row-major blocks.
#[inline]
fn neg_block_mm(n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    c.fill(0.0);
    sub_block_mm(n, a, b, c);
}

/// Subtract a block-vector product $y \leftarrow y - A x$ for an $n \times n$
/// block `a` and length-$n$ vectors `x`, `y`.
#[inline]
fn sub_block_mv(n: usize, a: &[f64], x: &[f64], y: &mut [f64]) {
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += a[i * n + j] * x[j];
        }
        y[i] -= s;
    }
}

/// Normalizes a single odd block row $(L_j, D_j, U_j, b_j)$ in place by
/// factoring $D_j$ and solving $D_j \tilde{L}_j = L_j$, $D_j \tilde{U}_j =
/// U_j$, and $D_j \tilde{b}_j = b_j$.
fn normalize_odd_row(
    nvar: usize,
    has_left: bool,
    has_right: bool,
    l_j: &mut [f64],
    d_j: &mut [f64],
    u_j: &mut [f64],
    b_j: &mut [f64],
    ipiv: &mut [usize],
) -> Result<(), MtxError> {
    dgetrf(nvar, d_j, ipiv)?;
    if has_left {
        dgetrs_multi(nvar, nvar, d_j, ipiv, l_j)?;
    } else {
        l_j.fill(0.0);
    }
    if has_right {
        dgetrs_multi(nvar, nvar, d_j, ipiv, u_j)?;
    } else {
        u_j.fill(0.0);
    }
    dgetrs(nvar, d_j, ipiv, b_j)
}

/// Computes the Schur complement update for even row $k$ from normalized
/// neighbors $k - p$ and $k + p$.
#[allow(clippy::too_many_arguments)]
fn update_even_row(
    nvar: usize,
    k: usize,
    stride: usize,
    nz: usize,
    l_cur: &[f64],
    d_cur: &[f64],
    u_cur: &[f64],
    b_cur: &[f64],
    l_next: &mut [f64],
    d_next: &mut [f64],
    u_next: &mut [f64],
    b_next: &mut [f64],
) {
    let blk_sz = nvar * nvar;
    let l_k = &l_cur[k * blk_sz..(k + 1) * blk_sz];
    let d_k = &d_cur[k * blk_sz..(k + 1) * blk_sz];
    let u_k = &u_cur[k * blk_sz..(k + 1) * blk_sz];
    let b_k = &b_cur[k * nvar..(k + 1) * nvar];

    d_next.copy_from_slice(d_k);
    b_next.copy_from_slice(b_k);

    if k >= stride {
        let jm = k - stride;
        let l_jm = &l_cur[jm * blk_sz..(jm + 1) * blk_sz];
        let u_jm = &u_cur[jm * blk_sz..(jm + 1) * blk_sz];
        let b_jm = &b_cur[jm * nvar..(jm + 1) * nvar];
        sub_block_mm(nvar, l_k, u_jm, d_next);
        sub_block_mv(nvar, l_k, b_jm, b_next);
        if jm >= stride {
            neg_block_mm(nvar, l_k, l_jm, l_next);
        } else {
            l_next.fill(0.0);
        }
    } else {
        l_next.fill(0.0);
    }

    if k + stride < nz {
        let jp = k + stride;
        let l_jp = &l_cur[jp * blk_sz..(jp + 1) * blk_sz];
        let u_jp = &u_cur[jp * blk_sz..(jp + 1) * blk_sz];
        let b_jp = &b_cur[jp * nvar..(jp + 1) * nvar];
        sub_block_mm(nvar, u_k, l_jp, d_next);
        sub_block_mv(nvar, u_k, b_jp, b_next);
        if jp + stride < nz {
            neg_block_mm(nvar, u_k, u_jp, u_next);
        } else {
            u_next.fill(0.0);
        }
    } else {
        u_next.fill(0.0);
    }
}

/// Solves a block-tridiagonal system $A x = b$ using Hirshman et al. (2010)
/// odd-even block cyclic reduction (`bcyclic`).
///
/// When `parallel` is `true`, odd-row block factorizations and even-row Schur
/// updates at each reduction level execute in parallel across the active
/// `rayon` thread pool.
///
/// # Arguments
/// * `nvar` - Number of variables per shell ($N_{\mathrm{var}} \ge 1$).
/// * `nz` - Number of radial shells ($N_z \ge 1$).
/// * `lblk` - Sub-diagonal blocks of shape `(nz, nvar, nvar)`.
/// * `dblk` - Diagonal blocks of shape `(nz, nvar, nvar)`.
/// * `ublk` - Super-diagonal blocks of shape `(nz, nvar, nvar)`.
/// * `rhs` - Right-hand side vector of length `nz * nvar`.
/// * `parallel` - Whether to parallelize each cyclic reduction level with
///   `rayon`.
///
/// # Errors
///
/// Returns [`MtxError::DimensionMismatch`], [`MtxError::NonFiniteInput`], or
/// [`MtxError::SingularMatrix`] if any diagonal block is singular.
pub fn solve_block_tridiag_bcyclic(
    nvar: usize,
    nz: usize,
    lblk: &[f64],
    dblk: &[f64],
    ublk: &[f64],
    rhs: &[f64],
    parallel: bool,
) -> Result<Vec<f64>, MtxError> {
    let blk_sz = check_blocks(nvar, nz, lblk, dblk, ublk)?;
    if rhs.len() != nz * nvar {
        return Err(MtxError::DimensionMismatch(
            "rhs length mismatch in solve_block_tridiag_bcyclic",
        ));
    }
    if is_bad_slice(lblk)
        || is_bad_slice(dblk)
        || is_bad_slice(ublk)
        || is_bad_slice(rhs)
    {
        return Err(MtxError::NonFiniteInput(
            "block-tridiagonal input contains NaN or Inf",
        ));
    }

    let mut l_work = lblk.to_vec();
    let mut d_work = dblk.to_vec();
    let mut u_work = ublk.to_vec();
    let mut b_work = rhs.to_vec();
    let mut l_tmp = vec![0.0_f64; nz * blk_sz];
    let mut d_tmp = vec![0.0_f64; nz * blk_sz];
    let mut u_tmp = vec![0.0_f64; nz * blk_sz];
    let mut b_tmp = vec![0.0_f64; nz * nvar];
    let mut ipiv_work = vec![0usize; nz * nvar];

    let mut stride = 1usize;
    let mut strides = Vec::new();

    while stride < nz {
        strides.push(stride);
        let step = 2 * stride;
        let n_odd = (nz.saturating_sub(stride)).div_ceil(step);

        // Step 1: Normalize odd multiples j = (2m + 1) * stride in-place.
        if parallel && n_odd >= 16 {
            l_work
                .par_chunks_mut(blk_sz)
                .zip(d_work.par_chunks_mut(blk_sz))
                .zip(u_work.par_chunks_mut(blk_sz))
                .zip(b_work.par_chunks_mut(nvar))
                .zip(ipiv_work.par_chunks_mut(nvar))
                .enumerate()
                .try_for_each(|(j, ((((lj, dj), uj), bj), ipiv))| {
                    if j % step != stride {
                        return Ok(());
                    }
                    normalize_odd_row(
                        nvar,
                        j >= stride,
                        j + stride < nz,
                        lj,
                        dj,
                        uj,
                        bj,
                        ipiv,
                    )
                    .map_err(|e| match e {
                        MtxError::SingularMatrix(_) => {
                            MtxError::SingularMatrix(j)
                        }
                        other => other,
                    })
                })?;
        } else {
            for m in 0..n_odd {
                let j = (2 * m + 1) * stride;
                normalize_odd_row(
                    nvar,
                    j >= stride,
                    j + stride < nz,
                    &mut l_work[j * blk_sz..(j + 1) * blk_sz],
                    &mut d_work[j * blk_sz..(j + 1) * blk_sz],
                    &mut u_work[j * blk_sz..(j + 1) * blk_sz],
                    &mut b_work[j * nvar..(j + 1) * nvar],
                    &mut ipiv_work[j * nvar..(j + 1) * nvar],
                )
                .map_err(|e| match e {
                    MtxError::SingularMatrix(_) => MtxError::SingularMatrix(j),
                    other => other,
                })?;
            }
        }

        // Step 2: Schur complement update on even multiples k = 2m * stride.
        let n_even = nz.div_ceil(step);
        if parallel && n_even >= 16 {
            l_tmp
                .par_chunks_mut(blk_sz)
                .zip(d_tmp.par_chunks_mut(blk_sz))
                .zip(u_tmp.par_chunks_mut(blk_sz))
                .zip(b_tmp.par_chunks_mut(nvar))
                .enumerate()
                .for_each(|(k, (((ln, dn), un), bn))| {
                    if k % step == 0 {
                        update_even_row(
                            nvar, k, stride, nz, &l_work, &d_work, &u_work,
                            &b_work, ln, dn, un, bn,
                        );
                    }
                });
            for m in 0..n_even {
                let k = 2 * m * stride;
                let bs = k * blk_sz..(k + 1) * blk_sz;
                let vs = k * nvar..(k + 1) * nvar;
                l_work[bs.clone()].copy_from_slice(&l_tmp[bs.clone()]);
                d_work[bs.clone()].copy_from_slice(&d_tmp[bs.clone()]);
                u_work[bs.clone()].copy_from_slice(&u_tmp[bs]);
                b_work[vs.clone()].copy_from_slice(&b_tmp[vs]);
            }
        } else {
            for m in 0..n_even {
                let k = 2 * m * stride;
                let bs = k * blk_sz..(k + 1) * blk_sz;
                let vs = k * nvar..(k + 1) * nvar;
                update_even_row(
                    nvar,
                    k,
                    stride,
                    nz,
                    &l_work,
                    &d_work,
                    &u_work,
                    &b_work,
                    &mut l_tmp[bs.clone()],
                    &mut d_tmp[bs.clone()],
                    &mut u_tmp[bs.clone()],
                    &mut b_tmp[vs.clone()],
                );
                l_work[bs.clone()].copy_from_slice(&l_tmp[bs.clone()]);
                d_work[bs.clone()].copy_from_slice(&d_tmp[bs.clone()]);
                u_work[bs.clone()].copy_from_slice(&u_tmp[bs]);
                b_work[vs.clone()].copy_from_slice(&b_tmp[vs]);
            }
        }

        stride = step;
    }

    // Step 3: Solve root shell k = 0.
    dgetrf(nvar, &mut d_work[0..blk_sz], &mut ipiv_work[0..nvar])?;
    dgetrs(
        nvar,
        &d_work[0..blk_sz],
        &ipiv_work[0..nvar],
        &mut b_work[0..nvar],
    )?;

    // Step 4: Backward substitution across strides in reverse order.
    let mut x = b_work;
    for &p in strides.iter().rev() {
        let step = 2 * p;
        let n_odd = (nz.saturating_sub(p)).div_ceil(step);
        for m in 0..n_odd {
            let j = (2 * m + 1) * p;
            let vs = j * nvar..(j + 1) * nvar;
            b_tmp[vs.clone()].copy_from_slice(&x[vs.clone()]);
            let x_j = &mut b_tmp[vs.clone()];
            if j >= p {
                let jm = j - p;
                let l_j = &l_work[j * blk_sz..(j + 1) * blk_sz];
                sub_block_mv(nvar, l_j, &x[jm * nvar..(jm + 1) * nvar], x_j);
            }
            if j + p < nz {
                let jp = j + p;
                let u_j = &u_work[j * blk_sz..(j + 1) * blk_sz];
                sub_block_mv(nvar, u_j, &x[jp * nvar..(jp + 1) * nvar], x_j);
            }
            x[vs].copy_from_slice(x_j);
        }
    }

    Ok(x)
}
