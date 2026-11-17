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
//! Mirrors upstream MESA `mtx/private/bcyclic.f90` and
//! `star/private/star_bcyclic.f90` (`cycle_onestep`, `cycle_rhs`, `cycle_solve`)
//! in upstream's exact floating-point operation order (`#165`, `N1-8`):
//!
//! At reduction level $s \ge 0$ with stride $p = 2^s$:
//! 1. **Odd-Row Normalization**: For every eliminated row $j = (2m + 1)p < N_z$
//!    (upstream 1-based even block index `ns = 2, 4, ...`), factor
//!    $D_j^{(s)} = P_j^{-1} L_j^{(D)} U_j^{(D)}$, negate the normalized
//!    off-diagonal blocks $\hat{L}_j = -(D_j^{(s)})^{-1} L_j^{(s)}$ and
//!    $\hat{U}_j = -(D_j^{(s)})^{-1} U_j^{(s)}$, and solve
//!    $\hat{b}_j = (D_j^{(s)})^{-1} b_j^{(s)}$.
//! 2. **Surviving-Row Schur Update**: For every surviving row $k = 2m p < N_z$
//!    (upstream 1-based odd block index `ns = 1, 3, ...`), update
//!    $D_k^{(s+1)} = D_k^{(s)} + U_k^{(s)} \hat{L}_{k+p} + L_k^{(s)} \hat{U}_{k-p}$
//!    via `my_gemm_plus_mm` (`umat0 * lnext` interleaved with `lmat0 * uprev`),
//!    $L_k^{(s+1)} = L_k^{(s)} \hat{L}_{k-p}$, $U_k^{(s+1)} = U_k^{(s)} \hat{U}_{k+p}$,
//!    and $b_k^{(s+1)} = b_k^{(s)} - U_k^{(s)} \hat{b}_{k+p} - L_k^{(s)} \hat{b}_{k-p}$
//!    via `my_gemv_mv`.
//! 3. **Root Solve & Backward Substitution**: Once $2^s \ge N_z$, solve row
//!    $k = 0$ via $x_0 = \left(D_0^{(s)}\right)^{-1} b_0^{(s)}$, then traverse
//!    levels $s = S-1 \dots 0$ in reverse via `my_gemv_p_mv`:
//!    $$x_j = \hat{b}_j + \hat{U}_j x_{j+p} + \hat{L}_j x_{j-p}.$$

use std::ops::Range;

use rayon::prelude::*;

use crate::block_tridiag::{
    add_block_mm, add_block_mv, add_two_block_mm, add_two_block_mv,
    check_blocks, sub_block_mv, sub_two_block_mv,
};
use crate::dense::{dgetrf, dgetrs, dgetrs_multi};
use crate::MtxError;
use flexmesa_utils::is_bad_slice;

/// Minimum number of active odd or even rows required to dispatch a cyclic
/// reduction level across the `rayon` thread pool.
const MIN_PARALLEL_ROWS: usize = 16;

/// Returns the flat slice range for shell `r` in a `(nz, nvar, nvar)` block
/// buffer (`blk_sz = nvar * nvar`).
#[inline]
const fn blk_range(r: usize, blk_sz: usize) -> Range<usize> {
    (r * blk_sz)..((r + 1) * blk_sz)
}

/// Returns the flat slice range for shell `r` in a `(nz, nvar)` vector buffer.
#[inline]
const fn vec_range(r: usize, nvar: usize) -> Range<usize> {
    (r * nvar)..((r + 1) * nvar)
}

/// Block product $C \leftarrow A B$ in upstream `my_gemm0_p1` order
/// (`mtx/public/mtx_solve_routines.inc:313-325`).
#[inline]
fn zero_then_add_block_mm(n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    c.fill(0.0);
    add_block_mm(n, a, b, c);
}

/// Normalizes an eliminated block row $j = (2m + 1)p$ in place matching
/// `cycle_onestep` (`star/private/star_bcyclic.f90:312-355`) and `cycle_rhs`
/// (`star/private/star_bcyclic.f90:467-485`): factors $D_j$, solves and negates
/// $\hat{L}_j = -D_j^{-1} L_j$ and $\hat{U}_j = -D_j^{-1} U_j$, and solves
/// $\hat{b}_j = D_j^{-1} b_j$.
fn normalize_odd_row(
    nvar: usize,
    row_idx: usize,
    has_right: bool,
    l_j: &mut [f64],
    d_j: &mut [f64],
    u_j: &mut [f64],
    b_j: &mut [f64],
    ipiv: &mut [usize],
) -> Result<(), MtxError> {
    dgetrf(nvar, d_j, ipiv).map_err(|e| match e {
        MtxError::SingularMatrix(_) => MtxError::SingularMatrix(row_idx),
        other => other,
    })?;
    dgetrs_multi(nvar, nvar, d_j, ipiv, l_j)?;
    for val in l_j.iter_mut() {
        *val = -*val;
    }
    if has_right {
        dgetrs_multi(nvar, nvar, d_j, ipiv, u_j)?;
        for val in u_j.iter_mut() {
            *val = -*val;
        }
    } else {
        u_j.fill(0.0);
    }
    dgetrs(nvar, d_j, ipiv, b_j)
}

/// Computes the Schur complement update for surviving row $k = 2m p$ from
/// normalized neighbors $k - p$ and $k + p$ in upstream `cycle_onestep` /
/// `cycle_rhs` order (`star/private/star_bcyclic.f90:402-426, 506-524`).
#[allow(clippy::too_many_arguments)]
fn update_even_row(
    nvar: usize,
    k: usize,
    stride: usize,
    nz: usize,
    l_cur: &[f64],
    u_cur: &[f64],
    b_cur: &[f64],
    d_k: &mut [f64],
    l_next: &mut [f64],
    u_next: &mut [f64],
    b_next: &mut [f64],
) {
    let blk_sz = nvar * nvar;
    let l_k = &l_cur[blk_range(k, blk_sz)];
    let u_k = &u_cur[blk_range(k, blk_sz)];
    b_next.copy_from_slice(&b_cur[vec_range(k, nvar)]);

    let has_left = k >= stride;
    let has_right = k + stride < nz;

    if k >= 2 * stride {
        let l_jm = &l_cur[blk_range(k - stride, blk_sz)];
        zero_then_add_block_mm(nvar, l_k, l_jm, l_next);
    } else {
        l_next.fill(0.0);
    }

    if k + 2 * stride < nz {
        let u_jp = &u_cur[blk_range(k + stride, blk_sz)];
        zero_then_add_block_mm(nvar, u_k, u_jp, u_next);
    } else {
        u_next.fill(0.0);
    }

    match (has_right, has_left) {
        (true, true) => {
            let jp = k + stride;
            let jm = k - stride;
            let l_jp = &l_cur[blk_range(jp, blk_sz)];
            let b_jp = &b_cur[vec_range(jp, nvar)];
            let u_jm = &u_cur[blk_range(jm, blk_sz)];
            let b_jm = &b_cur[vec_range(jm, nvar)];
            add_two_block_mm(nvar, u_k, l_jp, l_k, u_jm, d_k);
            sub_two_block_mv(u_k, b_jp, l_k, b_jm, b_next);
        }
        (true, false) => {
            let jp = k + stride;
            add_block_mm(nvar, u_k, &l_cur[blk_range(jp, blk_sz)], d_k);
            sub_block_mv(u_k, &b_cur[vec_range(jp, nvar)], b_next);
        }
        (false, true) => {
            let jm = k - stride;
            add_block_mm(nvar, l_k, &u_cur[blk_range(jm, blk_sz)], d_k);
            sub_block_mv(l_k, &b_cur[vec_range(jm, nvar)], b_next);
        }
        (false, false) => {}
    }
}

/// Solves a block-tridiagonal system $A x = b$ using Hirshman et al. (2010)
/// odd-even block cyclic reduction (`bcyclic`).
///
/// When `parallel` is `true`, odd-row block factorizations and even-row Schur
/// updates at each reduction level with at least `16` active rows execute in
/// parallel across the enclosing `rayon` thread pool.
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
/// [`MtxError::SingularMatrix`] (carrying the 0-based singular shell index) if
/// any diagonal block is singular.
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
        if parallel && n_odd >= MIN_PARALLEL_ROWS {
            let blk_step = step * blk_sz;
            let vec_step = step * nvar;
            let b_off = stride * blk_sz;
            let v_off = stride * nvar;
            l_work
                .par_chunks_mut(blk_step)
                .zip(d_work.par_chunks_mut(blk_step))
                .zip(u_work.par_chunks_mut(blk_step))
                .zip(b_work.par_chunks_mut(vec_step))
                .zip(ipiv_work.par_chunks_mut(vec_step))
                .enumerate()
                .try_for_each(|(m, ((((l_c, d_c), u_c), b_c), ipiv_c))| {
                    let j = (2 * m + 1) * stride;
                    if j >= nz {
                        return Ok(());
                    }
                    normalize_odd_row(
                        nvar,
                        j,
                        j + stride < nz,
                        &mut l_c[b_off..b_off + blk_sz],
                        &mut d_c[b_off..b_off + blk_sz],
                        &mut u_c[b_off..b_off + blk_sz],
                        &mut b_c[v_off..v_off + nvar],
                        &mut ipiv_c[v_off..v_off + nvar],
                    )
                })?;
        } else {
            for m in 0..n_odd {
                let j = (2 * m + 1) * stride;
                normalize_odd_row(
                    nvar,
                    j,
                    j + stride < nz,
                    &mut l_work[blk_range(j, blk_sz)],
                    &mut d_work[blk_range(j, blk_sz)],
                    &mut u_work[blk_range(j, blk_sz)],
                    &mut b_work[vec_range(j, nvar)],
                    &mut ipiv_work[vec_range(j, nvar)],
                )?;
            }
        }

        // Step 2: Schur complement update on even multiples k = 2m * stride.
        let n_even = nz.div_ceil(step);
        if parallel && n_even >= MIN_PARALLEL_ROWS {
            let blk_step = step * blk_sz;
            let vec_step = step * nvar;
            d_work
                .par_chunks_mut(blk_step)
                .zip(l_tmp.par_chunks_mut(blk_step))
                .zip(u_tmp.par_chunks_mut(blk_step))
                .zip(b_tmp.par_chunks_mut(vec_step))
                .enumerate()
                .for_each(|(m, (((dk_c, ln_c), un_c), bn_c))| {
                    let k = 2 * m * stride;
                    update_even_row(
                        nvar,
                        k,
                        stride,
                        nz,
                        &l_work,
                        &u_work,
                        &b_work,
                        &mut dk_c[..blk_sz],
                        &mut ln_c[..blk_sz],
                        &mut un_c[..blk_sz],
                        &mut bn_c[..nvar],
                    );
                });
        } else {
            for m in 0..n_even {
                let k = 2 * m * stride;
                let bs = blk_range(k, blk_sz);
                let vs = vec_range(k, nvar);
                update_even_row(
                    nvar,
                    k,
                    stride,
                    nz,
                    &l_work,
                    &u_work,
                    &b_work,
                    &mut d_work[bs.clone()],
                    &mut l_tmp[bs.clone()],
                    &mut u_tmp[bs],
                    &mut b_tmp[vs],
                );
            }
        }
        for m in 0..n_even {
            let k = 2 * m * stride;
            let bs = blk_range(k, blk_sz);
            let vs = vec_range(k, nvar);
            l_work[bs.clone()].copy_from_slice(&l_tmp[bs.clone()]);
            u_work[bs.clone()].copy_from_slice(&u_tmp[bs]);
            b_work[vs.clone()].copy_from_slice(&b_tmp[vs]);
        }

        stride = step;
    }

    // Step 3: Solve root shell k = 0.
    dgetrf(nvar, &mut d_work[0..blk_sz], &mut ipiv_work[0..nvar]).map_err(
        |e| match e {
            MtxError::SingularMatrix(_) => MtxError::SingularMatrix(0),
            other => other,
        },
    )?;
    dgetrs(
        nvar,
        &d_work[0..blk_sz],
        &ipiv_work[0..nvar],
        &mut b_work[0..nvar],
    )?;

    // Step 4: In-place backward substitution across strides in reverse order
    // matching `cycle_solve` (`star/private/star_bcyclic.f90:561-585`).
    let mut x = b_work;
    for &p in strides.iter().rev() {
        let step = 2 * p;
        let n_odd = (nz.saturating_sub(p)).div_ceil(step);
        for m in 0..n_odd {
            let j = (2 * m + 1) * p;
            let jm = j - p;
            let (left, rest) = x.split_at_mut(j * nvar);
            let (x_j, right) = rest.split_at_mut(nvar);
            let l_hat = &l_work[blk_range(j, blk_sz)];
            let x_prev = &left[vec_range(jm, nvar)];
            if j + p < nz {
                let right_off = (p - 1) * nvar;
                let u_hat = &u_work[blk_range(j, blk_sz)];
                let x_next = &right[right_off..right_off + nvar];
                add_two_block_mv(u_hat, x_next, l_hat, x_prev, x_j);
            } else {
                add_block_mv(l_hat, x_prev, x_j);
            }
        }
    }

    Ok(x)
}
