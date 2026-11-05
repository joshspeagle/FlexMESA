//! Block-tridiagonal matrix-vector products and Thomas LU factorization.
//!
//! Represents a block-tridiagonal matrix across `nz` shells and `nvar`
//! variables per shell via three contiguous row-major block arrays `lblk`,
//! `dblk`, and `ublk` of shape `(nz, nvar, nvar)` (`len == nz * nvar * nvar`),
//! where shell `k` has sub-diagonal block `lblk[k]` ($k > 0$), diagonal block
//! `dblk[k]`, and super-diagonal block `ublk[k]` ($k + 1 < n_z$).
//!
//! # Block Thomas Elimination
//!
//! 1. Forward sweep ($k = 0..n_z - 1$): factor $\hat{D}_k = P_k^{-1} L_k U_k$,
//!    replace $\hat{U}_k \leftarrow \hat{D}_k^{-1} U_k$, and update
//!    $\hat{D}_{k+1} \leftarrow D_{k+1} - L_{k+1} \hat{U}_k$.
//! 2. Forward RHS sweep ($k = 0..n_z - 1$): update
//!    $b_k \leftarrow \hat{D}_k^{-1} (b_k - L_k b_{k-1})$.
//! 3. Backward RHS sweep ($k = n_z - 2..=0$): update
//!    $x_k \leftarrow b_k - \hat{U}_k x_{k+1}$.

use crate::dense::{dgetrf, dgetrs, dgetrs_multi};
use crate::dense_ops::{
    lu_factored_square_mv, multiply_xa, multiply_xa_plus_c,
};
use crate::MtxError;
use flexmesa_utils::is_bad_slice;

/// Validates common block-tridiagonal dimensions `(nvar, nz)` and slice lengths
/// for `lblk`, `dblk`, and `ublk`.
fn check_blocks(
    nvar: usize,
    nz: usize,
    lblk: &[f64],
    dblk: &[f64],
    ublk: &[f64],
) -> Result<usize, MtxError> {
    let blk_sz = nvar * nvar;
    if nvar == 0
        || nz == 0
        || lblk.len() != nz * blk_sz
        || dblk.len() != nz * blk_sz
        || ublk.len() != nz * blk_sz
    {
        return Err(MtxError::DimensionMismatch(
            "slice lengths do not match (nz, nvar) for block-tridiagonal",
        ));
    }
    Ok(blk_sz)
}

/// Evaluates the block-tridiagonal matrix-vector product
/// $y_k = L_k b_{k-1} + D_k b_k + U_k b_{k+1}$ for $k \in 0..n_z$.
///
/// # Errors
///
/// Returns [`MtxError::DimensionMismatch`] if any slice length does not match
/// `(nz, nvar)`.
pub fn block_dble_mv(
    nvar: usize,
    nz: usize,
    lblk: &[f64],
    dblk: &[f64],
    ublk: &[f64],
    vec_b: &[f64],
    prod: &mut [f64],
) -> Result<(), MtxError> {
    let blk_sz = check_blocks(nvar, nz, lblk, dblk, ublk)?;
    if vec_b.len() != nz * nvar || prod.len() != nz * nvar {
        return Err(MtxError::DimensionMismatch(
            "vec_b or prod length mismatch in block_dble_mv",
        ));
    }
    for k in 0..nz {
        let d_k = &dblk[k * blk_sz..(k + 1) * blk_sz];
        let b_k = &vec_b[k * nvar..(k + 1) * nvar];
        for i in 0..nvar {
            let mut sum = 0.0;
            for j in 0..nvar {
                sum += d_k[i * nvar + j] * b_k[j];
            }
            if k > 0 {
                let l_k = &lblk[k * blk_sz..(k + 1) * blk_sz];
                let b_km1 = &vec_b[(k - 1) * nvar..k * nvar];
                for j in 0..nvar {
                    sum += l_k[i * nvar + j] * b_km1[j];
                }
            }
            if k + 1 < nz {
                let u_k = &ublk[k * blk_sz..(k + 1) * blk_sz];
                let b_kp1 = &vec_b[(k + 1) * nvar..(k + 2) * nvar];
                for j in 0..nvar {
                    sum += u_k[i * nvar + j] * b_kp1[j];
                }
            }
            prod[k * nvar + i] = sum;
        }
    }
    Ok(())
}

/// Evaluates the left block-tridiagonal product $b = x^{\top} A$ matching
/// upstream `do_block_multiply_xa`.
///
/// # Errors
///
/// Returns [`MtxError::DimensionMismatch`] if any slice length does not match.
pub fn block_multiply_xa(
    nvar: usize,
    nz: usize,
    lblk: &[f64],
    dblk: &[f64],
    ublk: &[f64],
    vec_x: &[f64],
    out_b: &mut [f64],
) -> Result<(), MtxError> {
    let blk_sz = check_blocks(nvar, nz, lblk, dblk, ublk)?;
    if vec_x.len() != nz * nvar || out_b.len() != nz * nvar {
        return Err(MtxError::DimensionMismatch(
            "vec_x or out_b length mismatch in block_multiply_xa",
        ));
    }
    let mut tmp = vec![0.0; nvar];
    for k in 0..nz {
        let d_k = &dblk[k * blk_sz..(k + 1) * blk_sz];
        let x_k = &vec_x[k * nvar..(k + 1) * nvar];
        let b_k = &mut out_b[k * nvar..(k + 1) * nvar];
        multiply_xa(nvar, d_k, x_k, b_k)?;
        if k > 0 {
            let u_km1 = &ublk[(k - 1) * blk_sz..k * blk_sz];
            let x_km1 = &vec_x[(k - 1) * nvar..k * nvar];
            tmp.copy_from_slice(b_k);
            multiply_xa_plus_c(nvar, u_km1, x_km1, &tmp, b_k)?;
        }
        if k + 1 < nz {
            let l_kp1 = &lblk[(k + 1) * blk_sz..(k + 2) * blk_sz];
            let x_kp1 = &vec_x[(k + 1) * nvar..(k + 2) * nvar];
            tmp.copy_from_slice(b_k);
            multiply_xa_plus_c(nvar, l_kp1, x_kp1, &tmp, b_k)?;
        }
    }
    Ok(())
}

/// Evaluates $y = A b$ when each diagonal block `dblk[k]` has been LU-factored
/// with pivot indices `ipiv[k * nvar..(k + 1) * nvar]` (matching upstream
/// `do_LU_factored_block_dble_mv`).
///
/// # Errors
///
/// Returns [`MtxError::DimensionMismatch`] on invalid slice sizes.
#[allow(clippy::too_many_arguments)]
pub fn lu_factored_block_dble_mv(
    nvar: usize,
    nz: usize,
    lblk: &[f64],
    dblk: &[f64],
    ublk: &[f64],
    ipiv: &[usize],
    vec_b: &[f64],
    prod: &mut [f64],
) -> Result<(), MtxError> {
    let blk_sz = check_blocks(nvar, nz, lblk, dblk, ublk)?;
    if ipiv.len() != nz * nvar
        || vec_b.len() != nz * nvar
        || prod.len() != nz * nvar
    {
        return Err(MtxError::DimensionMismatch(
            "slice lengths do not match in lu_factored_block_dble_mv",
        ));
    }
    for k in 0..nz {
        let d_k = &dblk[k * blk_sz..(k + 1) * blk_sz];
        let ipiv_k = &ipiv[k * nvar..(k + 1) * nvar];
        let b_k = &vec_b[k * nvar..(k + 1) * nvar];
        let p_k = &mut prod[k * nvar..(k + 1) * nvar];
        lu_factored_square_mv(nvar, d_k, ipiv_k, b_k, p_k)?;
        if k > 0 {
            let l_k = &lblk[k * blk_sz..(k + 1) * blk_sz];
            let b_km1 = &vec_b[(k - 1) * nvar..k * nvar];
            for i in 0..nvar {
                for j in 0..nvar {
                    p_k[i] += l_k[i * nvar + j] * b_km1[j];
                }
            }
        }
        if k + 1 < nz {
            let u_k = &ublk[k * blk_sz..(k + 1) * blk_sz];
            let b_kp1 = &vec_b[(k + 1) * nvar..(k + 2) * nvar];
            for i in 0..nvar {
                for j in 0..nvar {
                    p_k[i] += u_k[i * nvar + j] * b_kp1[j];
                }
            }
        }
    }
    Ok(())
}

/// Factors a block-tridiagonal system in-place using the block Thomas algorithm
/// with partial pivoting on each $N_{\mathrm{var}} \times N_{\mathrm{var}}$
/// diagonal block.
///
/// For $k = 0..n_z - 1$:
/// 1. Factors $\hat{D}_k = P_k^{-1} L_k^{(D)} U_k^{(D)}$.
/// 2. If $k + 1 < n_z$, updates $\hat{U}_k = \hat{D}_k^{-1} U_k$ in-place and
///    eliminates $\hat{D}_{k+1} \leftarrow D_{k+1} - L_{k+1} \hat{U}_k$.
///
/// # Errors
///
/// Returns [`MtxError::DimensionMismatch`], [`MtxError::NonFiniteInput`], or
/// [`MtxError::SingularMatrix`] if any diagonal block is singular.
pub fn block_thomas_factor(
    nvar: usize,
    nz: usize,
    lblk: &[f64],
    dblk: &mut [f64],
    ublk: &mut [f64],
    ipiv: &mut [usize],
) -> Result<(), MtxError> {
    let blk_sz = check_blocks(nvar, nz, lblk, dblk, ublk)?;
    if ipiv.len() != nz * nvar {
        return Err(MtxError::DimensionMismatch(
            "ipiv length mismatch in block_thomas_factor",
        ));
    }
    if is_bad_slice(lblk) || is_bad_slice(dblk) || is_bad_slice(ublk) {
        return Err(MtxError::NonFiniteInput(
            "block-tridiagonal matrix contains NaN/Inf",
        ));
    }

    for k in 0..nz {
        let d_slice = &mut dblk[k * blk_sz..(k + 1) * blk_sz];
        let ipiv_slice = &mut ipiv[k * nvar..(k + 1) * nvar];
        dgetrf(nvar, d_slice, ipiv_slice)?;
        if k + 1 < nz {
            let d_factored = &dblk[k * blk_sz..(k + 1) * blk_sz];
            let u_k = &mut ublk[k * blk_sz..(k + 1) * blk_sz];
            dgetrs_multi(nvar, nvar, d_factored, ipiv_slice, u_k)?;
            let l_kp1 = &lblk[(k + 1) * blk_sz..(k + 2) * blk_sz];
            let u_k_ref = &ublk[k * blk_sz..(k + 1) * blk_sz];
            let d_kp1 = &mut dblk[(k + 1) * blk_sz..(k + 2) * blk_sz];
            for i in 0..nvar {
                for m in 0..nvar {
                    let l_im = l_kp1[i * nvar + m];
                    if l_im != 0.0 {
                        for j in 0..nvar {
                            d_kp1[i * nvar + j] -= l_im * u_k_ref[m * nvar + j];
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Solves a block-tridiagonal system in-place using factors from
/// [`block_thomas_factor`].
///
/// # Errors
///
/// Returns [`MtxError::DimensionMismatch`], [`MtxError::NonFiniteInput`], or
/// propagates errors from [`dgetrs`].
pub fn block_thomas_solve(
    nvar: usize,
    nz: usize,
    lblk: &[f64],
    dblk: &[f64],
    ublk: &[f64],
    ipiv: &[usize],
    rhs: &mut [f64],
) -> Result<(), MtxError> {
    let blk_sz = check_blocks(nvar, nz, lblk, dblk, ublk)?;
    if ipiv.len() != nz * nvar || rhs.len() != nz * nvar {
        return Err(MtxError::DimensionMismatch(
            "ipiv or rhs length mismatch in block_thomas_solve",
        ));
    }
    if is_bad_slice(rhs) {
        return Err(MtxError::NonFiniteInput("rhs contains NaN/Inf"));
    }

    for k in 0..nz {
        if k > 0 {
            let l_k = &lblk[k * blk_sz..(k + 1) * blk_sz];
            for i in 0..nvar {
                let mut sum = 0.0;
                for j in 0..nvar {
                    sum += l_k[i * nvar + j] * rhs[(k - 1) * nvar + j];
                }
                rhs[k * nvar + i] -= sum;
            }
        }
        let d_k = &dblk[k * blk_sz..(k + 1) * blk_sz];
        let ipiv_k = &ipiv[k * nvar..(k + 1) * nvar];
        dgetrs(nvar, d_k, ipiv_k, &mut rhs[k * nvar..(k + 1) * nvar])?;
    }

    for k in (0..nz.saturating_sub(1)).rev() {
        let u_k = &ublk[k * blk_sz..(k + 1) * blk_sz];
        for i in 0..nvar {
            let mut sum = 0.0;
            for j in 0..nvar {
                sum += u_k[i * nvar + j] * rhs[(k + 1) * nvar + j];
            }
            rhs[k * nvar + i] -= sum;
        }
    }
    Ok(())
}

/// Factors and solves a block-tridiagonal system $A x = b$, returning $x$.
///
/// # Errors
///
/// Propagates any error from [`block_thomas_factor`] or [`block_thomas_solve`].
pub fn solve_block_tridiagonal(
    nvar: usize,
    nz: usize,
    lblk: &[f64],
    dblk: &[f64],
    ublk: &[f64],
    rhs: &[f64],
) -> Result<Vec<f64>, MtxError> {
    let mut d_work = dblk.to_vec();
    let mut u_work = ublk.to_vec();
    let mut ipiv = vec![0usize; nz * nvar];
    let mut sol = rhs.to_vec();
    block_thomas_factor(nvar, nz, lblk, &mut d_work, &mut u_work, &mut ipiv)?;
    block_thomas_solve(nvar, nz, lblk, &d_work, &u_work, &ipiv, &mut sol)?;
    Ok(sol)
}

/// Clips off-diagonal entries of a single $m \times m$ block with magnitude
/// below `clip_limit` to zero and returns the count of non-zero entries.
///
/// # Errors
///
/// Returns [`MtxError::DimensionMismatch`] if `dmat.len() != mblk * mblk`.
pub fn clip_block(
    mblk: usize,
    clip_limit: f64,
    dmat: &mut [f64],
) -> Result<usize, MtxError> {
    if dmat.len() != mblk * mblk {
        return Err(MtxError::DimensionMismatch(
            "dmat must have length mblk * mblk",
        ));
    }
    let mut nnz = 0usize;
    for i in 0..mblk {
        for j in 0..mblk {
            let idx = i * mblk + j;
            if i != j && dmat[idx].abs() < clip_limit {
                dmat[idx] = 0.0;
            }
            if dmat[idx] != 0.0 {
                nnz += 1;
            }
        }
    }
    Ok(nnz)
}

/// Clips off-diagonal entries of `(lmat, dmat, umat)` below `clip_limit` and
/// returns `(dmat_nnz, total_nnz)`.
///
/// # Errors
///
/// Returns [`MtxError::DimensionMismatch`] if any slice length does not equal
/// `mblk * mblk`.
pub fn clip_blocks(
    mblk: usize,
    clip_limit: f64,
    lmat: &mut [f64],
    dmat: &mut [f64],
    umat: &mut [f64],
) -> Result<(usize, usize), MtxError> {
    let sz = mblk * mblk;
    if lmat.len() != sz || dmat.len() != sz || umat.len() != sz {
        return Err(MtxError::DimensionMismatch(
            "lmat, dmat, umat must have length mblk * mblk",
        ));
    }
    let l_nnz = clip_block(mblk, clip_limit, lmat)?;
    let d_nnz = clip_block(mblk, clip_limit, dmat)?;
    let u_nnz = clip_block(mblk, clip_limit, umat)?;
    Ok((d_nnz, l_nnz + d_nnz + u_nnz))
}
