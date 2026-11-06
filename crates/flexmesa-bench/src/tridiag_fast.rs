//! Pre-validated block-tridiagonal fixture and Thomas solver (`tridiag_fast`).
//!
//! Provides `make_block_tridiag_fixture` and `fast_block_thomas_solve` for
//! Stage 2 and Stage 3 benchmarks without repeating $O(n_z n_{\mathrm{var}}^2)$
//! NaN validation scans on pre-validated workspace buffers.

use flexmesa_mtx::{dgetrf, dgetrs, dgetrs_multi};

/// Constructs a diagonally dominant, well-conditioned $5 \times 5$
/// block-tridiagonal system across `nz` shells matching a 5-variable Henyey
/// stellar structure matrix.
///
/// # Arguments
/// * `nz` - Number of radial zones ($N_{\mathrm{zones}} \ge 1$).
///
/// # Returns
/// Four-tuple `(lblk, dblk, ublk, rhs)` where `lblk`, `dblk`, and `ublk` have
/// flattened row-major shape `(nz * 25,)` and `rhs` has length `nz * 5`.
pub fn make_block_tridiag_fixture(
    nz: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let nvar = 5usize;
    let blk = nvar * nvar;
    let mut lblk = vec![0.0_f64; nz * blk];
    let mut dblk = vec![0.0_f64; nz * blk];
    let mut ublk = vec![0.0_f64; nz * blk];
    let mut rhs = vec![0.0_f64; nz * nvar];

    for k in 0..nz {
        let s = (k + 1) as f64 / (nz as f64);
        for i in 0..nvar {
            rhs[k * nvar + i] = 1.0 + 0.2 * (i as f64) - 0.1 * s;
            for j in 0..nvar {
                let idx = k * blk + i * nvar + j;
                if i == j {
                    dblk[idx] = 4.0 + 0.5 * (i as f64) + 0.2 * s;
                } else {
                    dblk[idx] = 0.05 / ((i.abs_diff(j) + 1) as f64);
                }
                if k > 0 {
                    lblk[idx] = -0.12 / ((i + j + 2) as f64);
                }
                if k + 1 < nz {
                    ublk[idx] = -0.10 / ((i + j + 2) as f64);
                }
            }
        }
    }
    (lblk, dblk, ublk, rhs)
}

/// Factors and solves a pre-validated block-tridiagonal system in-place using
/// pre-allocated workspace buffers (`dblk`, `ublk`, `ipiv`, `rhs`) without
/// repeating $O(n_z n_{\mathrm{var}}^2)$ NaN validation scans.
///
/// # Arguments
/// * `nvar` - Block dimension ($N_{\mathrm{var}} = 5$).
/// * `nz` - Number of radial zones ($N_{\mathrm{zones}}$).
/// * `lblk` - Sub-diagonal blocks of length `nz * nvar * nvar` (read-only).
/// * `dblk` - Diagonal blocks of length `nz * nvar * nvar` (overwritten with
///   LU factors).
/// * `ublk` - Super-diagonal blocks of length `nz * nvar * nvar` (overwritten
///   with $D_k^{-1} U_k$).
/// * `ipiv` - Pre-allocated pivot index buffer of length `nz * nvar`.
/// * `rhs` - Right-hand side buffer of length `nz * nvar` (overwritten in place
///   with the solution vector $x$).
pub fn fast_block_thomas_solve(
    nvar: usize,
    nz: usize,
    lblk: &[f64],
    dblk: &mut [f64],
    ublk: &mut [f64],
    ipiv: &mut [usize],
    rhs: &mut [f64],
) {
    let blk_sz = nvar * nvar;
    for k in 0..nz {
        let d_slice = &mut dblk[k * blk_sz..(k + 1) * blk_sz];
        let ipiv_slice = &mut ipiv[k * nvar..(k + 1) * nvar];
        dgetrf(nvar, d_slice, ipiv_slice).unwrap();
        if k + 1 < nz {
            let d_factored = &dblk[k * blk_sz..(k + 1) * blk_sz];
            let u_k = &mut ublk[k * blk_sz..(k + 1) * blk_sz];
            dgetrs_multi(nvar, nvar, d_factored, ipiv_slice, u_k).unwrap();
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
    for k in 0..nz {
        if k > 0 {
            let (prev, curr) = rhs.split_at_mut(k * nvar);
            let l_k = &lblk[k * blk_sz..(k + 1) * blk_sz];
            let p_km1 = &prev[(k - 1) * nvar..k * nvar];
            let p_k = &mut curr[..nvar];
            for i in 0..nvar {
                let mut s = 0.0;
                for j in 0..nvar {
                    s += l_k[i * nvar + j] * p_km1[j];
                }
                p_k[i] -= s;
            }
        }
        dgetrs(
            nvar,
            &dblk[k * blk_sz..(k + 1) * blk_sz],
            &ipiv[k * nvar..(k + 1) * nvar],
            &mut rhs[k * nvar..(k + 1) * nvar],
        )
        .unwrap();
    }
    for k in (0..nz.saturating_sub(1)).rev() {
        let (curr, next) = rhs.split_at_mut((k + 1) * nvar);
        let u_k = &ublk[k * blk_sz..(k + 1) * blk_sz];
        let p_kp1 = &next[..nvar];
        let p_k = &mut curr[k * nvar..(k + 1) * nvar];
        for i in 0..nvar {
            let mut s = 0.0;
            for j in 0..nvar {
                s += u_k[i * nvar + j] * p_kp1[j];
            }
            p_k[i] -= s;
        }
    }
}
