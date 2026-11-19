// SPDX-License-Identifier: LGPL-3.0-or-later
//! Bicubic spline construction, cell lookup, and 2D evaluation for `(logQ, logT)` EOS tables.
//!
//! Ports `Load1_eosDT_Table`, `Make_XEoS_Interpolation_Data`, and
//! `Check_XEoS_Interpolation_Data` from `eos/private/eosdt_load_tables.f90`,
//! `Do_EoS_Interpolations` from `eos/private/eosdt_support.f90`, and
//! `Locate_logQ`, `Locate_logT`, and `Get1_eosdt_XTable_Results` from
//! `eos/private/eosdt_eval.f90`.

use crate::types::{
    EosError, EosEvalOutput, EOS_BASIC_RESULT_NAMES, I_LN_S,
    NUM_EOS_BASIC_RESULTS,
};
use flexmesa_const::{LN10, ONE_SIXTH};
use flexmesa_interp2d::{interp_mkbicub_db, BicubicBc};
use flexmesa_utils::is_bad;
use rayon::prelude::*;

pub use crate::tables_4d::table_io::DT_RAW_BIN_MAGIC;

/// Number of basic thermodynamic quantities interpolated in `(logQ, logT)` tables (`nv = 16`).
pub const NUM_EOS_TABLE_VALS: usize = 16;

/// Identifies which 4D `(X, Z, logQ, logT)` EOS table family is being queried.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
#[repr(u8)]
pub enum EosDtTableKind {
    /// Combined OPAL and Saumon-Chabrier-Van Horn (`eosDT_data/mesa-eosDT_*.data`).
    OpalScvh = 0,
    /// Alan Irwin's `FreeEOS` (`eosFreeEOS_data/mesa-FreeEOS_*.data`).
    FreeEos = 1,
}

/// Bicubic spline coefficients `[f, f_xx, f_yy, f_xxyy]` for all 16 quantities at one `(iQ, jT)` vertex.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EosDtVertex {
    /// Packed `16 * 4 = 64` spline coefficients (`coeffs[k * 4 + m]` for variable `k`, moment `m`).
    pub coeffs: [f64; NUM_EOS_TABLE_VALS * 4],
}

impl Default for EosDtVertex {
    fn default() -> Self {
        Self {
            coeffs: [0.0; NUM_EOS_TABLE_VALS * 4],
        }
    }
}

/// Single `(X, Z)` table over `(logQ, logT)` (`EosDT_XZ_Info` in `eos_def.f90`).
#[derive(Debug, Clone)]
pub struct EosDtXzTable {
    /// Table format version number (typically `51`).
    pub version: i32,
    /// Hydrogen mass fraction $X$ of this table.
    pub x: f64,
    /// Heavy-element mass fraction $Z$ of this table.
    pub z: f64,
    /// Number of $\log_{10} T$ grid points (`306` in standard MESA tables).
    pub num_log_ts: usize,
    /// Minimum $\log_{10}(T/\mathrm{K})$ (`2.10`).
    pub log_t_min: f64,
    /// Maximum $\log_{10}(T/\mathrm{K})$ (`8.20`).
    pub log_t_max: f64,
    /// Grid spacing $\Delta \log_{10} T$ (`0.02`).
    pub del_log_t: f64,
    /// Number of $\log_{10} Q$ grid points (`527` in standard MESA tables).
    pub num_log_qs: usize,
    /// Minimum $\log_{10} Q = \log_{10} \rho - 2 \log_{10} T + 12$ (`-10.09`).
    pub log_q_min: f64,
    /// Maximum $\log_{10} Q$ (`5.69`).
    pub log_q_max: f64,
    /// Grid spacing $\Delta \log_{10} Q$ (`0.03`).
    pub del_log_q: f64,
    /// Full dense grid of `num_log_qs * num_log_ts` vertices.
    pub dense: Option<Vec<EosDtVertex>>,
}

impl EosDtXzTable {
    /// Looks up a 1-based `(i_q, j_t)` vertex from the dense array.
    #[inline]
    #[must_use]
    pub fn get_vertex(&self, i_q: usize, j_t: usize) -> Option<&EosDtVertex> {
        let dense = self.dense.as_ref()?;
        let idx = (i_q - 1) * self.num_log_ts + (j_t - 1);
        dense.get(idx)
    }
}

#[inline]
fn locate_uniform_axis(
    val_in: f64,
    v_min: f64,
    v_max: f64,
    del_v: f64,
    n_pts: usize,
    return_err_beyond_bounds: bool,
    ctx: &'static str,
    name: &'static str,
) -> Result<(usize, f64, f64, f64), EosError> {
    let i_raw = ((val_in - v_min) / del_v + 1.0e-4) as isize + 1;
    if i_raw < 1 || (i_raw as usize) >= n_pts {
        if return_err_beyond_bounds {
            return Err(EosError::OffTable {
                context: ctx,
                detail: format!("{name}={val_in} outside [{v_min}, {v_max}]"),
            });
        }
        let idx = if i_raw < 1 { 1 } else { n_pts - 1 };
        let v0 = v_min + ((idx - 1) as f64) * del_v;
        let v1 = v0 + del_v;
        let v_clamped = if i_raw < 1 { v0 } else { v1 };
        Ok((idx, v0, v_clamped, v1))
    } else {
        let idx = i_raw as usize;
        let v0 = v_min + ((idx - 1) as f64) * del_v;
        Ok((idx, v0, val_in, v0 + del_v))
    }
}

/// Locates the 1-based cell index `i_q` and cell bounds `(log_q0, log_q_clamped, log_q1)`.
#[inline]
pub fn locate_log_q(
    ep: &EosDtXzTable,
    log_q_in: f64,
    return_err_beyond_bounds: bool,
) -> Result<(usize, f64, f64, f64), EosError> {
    locate_uniform_axis(
        log_q_in,
        ep.log_q_min,
        ep.log_q_max,
        ep.del_log_q,
        ep.num_log_qs,
        return_err_beyond_bounds,
        "locate_log_q",
        "logQ",
    )
}

/// Locates the 1-based cell index `j_t` and cell bounds `(log_t0, log_t_clamped, log_t1)`.
#[inline]
pub fn locate_log_t(
    ep: &EosDtXzTable,
    log_t_in: f64,
    return_err_beyond_bounds: bool,
) -> Result<(usize, f64, f64, f64), EosError> {
    locate_uniform_axis(
        log_t_in,
        ep.log_t_min,
        ep.log_t_max,
        ep.del_log_t,
        ep.num_log_ts,
        return_err_beyond_bounds,
        "locate_log_t",
        "logT",
    )
}

/// Evaluates all 16 bicubic splines and their $(x, y)$ partials on a single cell.
///
/// Mirrors `Do_EoS_Interpolations` in `eos/private/eosdt_support.f90` (lines 32..172)
/// with the exact floating-point operation order of the Fortran SIMD loop.
#[inline]
#[must_use]
pub fn do_eos_interpolations(
    v_ij: &EosDtVertex,
    v_ip1_j: &EosDtVertex,
    v_i_jp1: &EosDtVertex,
    v_ip1_jp1: &EosDtVertex,
    x0: f64,
    xget: f64,
    x1: f64,
    y0: f64,
    yget: f64,
    y1: f64,
) -> (
    [f64; NUM_EOS_TABLE_VALS],
    [f64; NUM_EOS_TABLE_VALS],
    [f64; NUM_EOS_TABLE_VALS],
) {
    let (hx, hy) = (x1 - x0, y1 - y0);
    let (hxi, hyi) = (1.0 / hx, 1.0 / hy);
    let (hx2, hy2) = (hx * hx, hy * hy);

    let (xp, yp) = ((xget - x0) * hxi, (yget - y0) * hyi);
    let (xpi, ypi) = (1.0 - xp, 1.0 - yp);
    let (xp2, xpi2) = (xp * xp, xpi * xpi);
    let (yp2, ypi2) = (yp * yp, ypi * ypi);

    let (cx, cxi) = (xp * (xp2 - 1.0), xpi * (xpi2 - 1.0));
    let (cxd, cxdi) = (3.0 * xp2 - 1.0, -3.0 * xpi2 + 1.0);
    let (cy, cyi) = (yp * (yp2 - 1.0), ypi * (ypi2 - 1.0));
    let (cyd, cydi) = (3.0 * yp2 - 1.0, -3.0 * ypi2 + 1.0);

    let (sixth_hx2, sixth_hy2) = (ONE_SIXTH * hx2, ONE_SIXTH * hy2);
    let z36th_hx2_hy2 = sixth_hx2 * sixth_hy2;
    let (sixth_hx, sixth_hxi_hy2) = (ONE_SIXTH * hx, sixth_hy2 * hxi);
    let z36th_hx_hy2 = sixth_hx * sixth_hy2;
    let (sixth_hx2_hyi, sixth_hy) = (sixth_hx2 * hyi, ONE_SIXTH * hy);
    let z36th_hx2_hy = sixth_hx2 * sixth_hy;

    let mut fval = [0.0_f64; NUM_EOS_TABLE_VALS];
    let mut df_dx = [0.0_f64; NUM_EOS_TABLE_VALS];
    let mut df_dy = [0.0_f64; NUM_EOS_TABLE_VALS];

    for k in 0..NUM_EOS_TABLE_VALS {
        let base = k * 4;
        let c_00 = &v_ij.coeffs[base..];
        let c_10 = &v_ip1_j.coeffs[base..];
        let c_01 = &v_i_jp1.coeffs[base..];
        let c_11 = &v_ip1_jp1.coeffs[base..];
        let (f1_ij, f2_ij, f3_ij, f4_ij) = (c_00[0], c_00[1], c_00[2], c_00[3]);
        let (f1_ip1_j, f2_ip1_j, f3_ip1_j, f4_ip1_j) =
            (c_10[0], c_10[1], c_10[2], c_10[3]);
        let (f1_i_jp1, f2_i_jp1, f3_i_jp1, f4_i_jp1) =
            (c_01[0], c_01[1], c_01[2], c_01[3]);
        let (f1_ip1_jp1, f2_ip1_jp1, f3_ip1_jp1, f4_ip1_jp1) =
            (c_11[0], c_11[1], c_11[2], c_11[3]);

        fval[k] = xpi * (ypi * f1_ij + yp * f1_i_jp1)
            + xp * (ypi * f1_ip1_j + yp * f1_ip1_jp1)
            + sixth_hx2
                * (cxi * (ypi * f2_ij + yp * f2_i_jp1)
                    + cx * (ypi * f2_ip1_j + yp * f2_ip1_jp1))
            + sixth_hy2
                * (xpi * (cyi * f3_ij + cy * f3_i_jp1)
                    + xp * (cyi * f3_ip1_j + cy * f3_ip1_jp1))
            + z36th_hx2_hy2
                * (cxi * (cyi * f4_ij + cy * f4_i_jp1)
                    + cx * (cyi * f4_ip1_j + cy * f4_ip1_jp1));

        df_dx[k] = hxi
            * (-(ypi * f1_ij + yp * f1_i_jp1)
                + (ypi * f1_ip1_j + yp * f1_ip1_jp1))
            + sixth_hx
                * (cxdi * (ypi * f2_ij + yp * f2_i_jp1)
                    + cxd * (ypi * f2_ip1_j + yp * f2_ip1_jp1))
            + sixth_hxi_hy2
                * (-(cyi * f3_ij + cy * f3_i_jp1)
                    + (cyi * f3_ip1_j + cy * f3_ip1_jp1))
            + z36th_hx_hy2
                * (cxdi * (cyi * f4_ij + cy * f4_i_jp1)
                    + cxd * (cyi * f4_ip1_j + cy * f4_ip1_jp1));

        df_dy[k] = hyi
            * (xpi * (-f1_ij + f1_i_jp1) + xp * (-f1_ip1_j + f1_ip1_jp1))
            + sixth_hx2_hyi
                * (cxi * (-f2_ij + f2_i_jp1) + cx * (-f2_ip1_j + f2_ip1_jp1))
            + sixth_hy
                * (xpi * (cydi * f3_ij + cyd * f3_i_jp1)
                    + xp * (cydi * f3_ip1_j + cyd * f3_ip1_jp1))
            + z36th_hx2_hy
                * (cxi * (cydi * f4_ij + cyd * f4_i_jp1)
                    + cx * (cydi * f4_ip1_j + cyd * f4_ip1_jp1));
    }

    (fval, df_dx, df_dy)
}

/// Evaluates the 16 tabulated EOS channels and $(\ln\rho, \ln T)$ partials.
#[inline]
pub fn eval_xtable_16(
    ep: &EosDtXzTable,
    log_rho_in: f64,
    log_t_in: f64,
) -> Result<
    (
        [f64; NUM_EOS_TABLE_VALS],
        [f64; NUM_EOS_TABLE_VALS],
        [f64; NUM_EOS_TABLE_VALS],
    ),
    EosError,
> {
    let log_q_in = log_rho_in - 2.0 * log_t_in + 12.0;
    let (iq, log_q0, log_q, log_q1) = locate_log_q(ep, log_q_in, true)?;
    let (jt, log_t0, log_t, log_t1) = locate_log_t(ep, log_t_in, true)?;

    let vtx = |i: usize, j: usize| {
        ep.get_vertex(i, j).ok_or_else(|| EosError::InvalidState {
            context: "eval_xtable",
            detail: format!(
                "missing table vertex (iq={i}, jt={j}) for (X={}, Z={})",
                ep.x, ep.z
            ),
        })
    };
    let v_ij = vtx(iq, jt)?;
    let v_ip1_j = vtx(iq + 1, jt)?;
    let v_i_jp1 = vtx(iq, jt + 1)?;
    let v_ip1_jp1 = vtx(iq + 1, jt + 1)?;

    let (fval, mut df_dx, mut df_dy) = do_eos_interpolations(
        v_ij, v_ip1_j, v_i_jp1, v_ip1_jp1, log_q0, log_q, log_q1, log_t0,
        log_t, log_t1,
    );

    if is_bad(fval[I_LN_S]) || fval[I_LN_S] > LN10 * 100.0 {
        return Err(EosError::InvalidState {
            context: "eval_xtable",
            detail: format!("bad interpolated lnS={}", fval[I_LN_S]),
        });
    }

    for k in 0..NUM_EOS_TABLE_VALS {
        let dlnd_k = df_dx[k] / LN10;
        let dlnt_k = df_dy[k] / LN10 - 2.0 * dlnd_k;
        df_dx[k] = dlnd_k;
        df_dy[k] = dlnt_k;
    }

    if is_bad(df_dx[I_LN_S]) || is_bad(df_dy[I_LN_S]) {
        return Err(EosError::InvalidState {
            context: "eval_xtable",
            detail: "non-finite d_dlnd[I_LN_S] or d_dlnT[I_LN_S]".to_string(),
        });
    }

    Ok((fval, df_dx, df_dy))
}

/// Evaluates a single `(X, Z)` table at `(log_rho_in, log_t_in)`.
///
/// Mirrors `Get1_eosdt_XTable_Results` in `eos/private/eosdt_eval.f90` (lines 2083..2241).
#[inline]
pub fn eval_xtable(
    ep: &EosDtXzTable,
    log_rho_in: f64,
    log_t_in: f64,
) -> Result<EosEvalOutput, EosError> {
    let (fval, dlnd, dlnt) = eval_xtable_16(ep, log_rho_in, log_t_in)?;
    let mut res = [0.0_f64; NUM_EOS_BASIC_RESULTS];
    let mut d_dlnd = [0.0_f64; NUM_EOS_BASIC_RESULTS];
    let mut d_dln_t = [0.0_f64; NUM_EOS_BASIC_RESULTS];
    res[..NUM_EOS_TABLE_VALS].copy_from_slice(&fval);
    d_dlnd[..NUM_EOS_TABLE_VALS].copy_from_slice(&dlnd);
    d_dln_t[..NUM_EOS_TABLE_VALS].copy_from_slice(&dlnt);
    Ok(EosEvalOutput {
        res,
        d_dlnd,
        d_dln_t,
        d_dxa: Vec::new(),
    })
}

/// Validates and rejects non-finite coordinates, grid sizes, or raw/fitted
/// 4D EOS spline arrays (`N1-12`).
///
/// Upstream MESA's `Check_XEoS_Interpolation_Data` (`eos/private/eosdt_load_tables.f90:409-424`)
/// replaces `is_bad` spline coefficients with `0.0`. All 116 pinned MESA `26.4.1`
/// raw slices (`eos_opal_scvh`, `eos_free_eos`, `eos_cms`) have zero non-finite
/// values or spline coefficients, so rejecting any non-finite entry with an error
/// naming table, variable, and cell detects corruption without changing MESA
/// `26.4.1` physics.
pub(crate) fn build_4d_slice_vertices(
    context: &'static str,
    table_label: &str,
    axis1: &[f64],
    axis2: &[f64],
    raw_vars: &[f64],
) -> Result<Vec<EosDtVertex>, EosError> {
    let n1 = axis1.len();
    let n2 = axis2.len();
    if n1 < 2 || n2 < 2 {
        return Err(EosError::InvalidState {
            context,
            detail: format!(
                "{table_label}: grid too small ({n1} x {n2}); need at least 2 x 2"
            ),
        });
    }
    let ngrid = n1.checked_mul(n2).ok_or_else(|| EosError::InvalidState {
        context,
        detail: format!("{table_label}: grid size overflow ({n1} * {n2})"),
    })?;
    let expected_raw =
        NUM_EOS_TABLE_VALS.checked_mul(ngrid).ok_or_else(|| {
            EosError::InvalidState {
                context,
                detail: format!("{table_label}: raw buffer size overflow"),
            }
        })?;
    if raw_vars.len() != expected_raw {
        return Err(EosError::InvalidState {
            context,
            detail: format!(
                "{table_label}: expected {expected_raw} raw values, got {}",
                raw_vars.len()
            ),
        });
    }

    for (v, &val) in raw_vars.iter().enumerate() {
        if is_bad(val) {
            let var_idx = v / ngrid;
            let cell = v % ngrid;
            let i1 = cell / n2;
            let i2 = cell % n2;
            let var_name = EOS_BASIC_RESULT_NAMES
                .get(var_idx)
                .copied()
                .unwrap_or("var");
            return Err(EosError::InvalidState {
                context,
                detail: format!(
                    "non-finite input value ({val}) in {table_label} var={var_idx} ({var_name}) at cell=({i1}, {i2})"
                ),
            });
        }
    }

    let fitted: Vec<Vec<[f64; 4]>> = (0..NUM_EOS_TABLE_VALS)
        .into_par_iter()
        .map(|v| {
            let z_slice = &raw_vars[v * ngrid..(v + 1) * ngrid];
            interp_mkbicub_db(
                axis1,
                axis2,
                z_slice,
                BicubicBc::NotAKnot,
                &[],
                BicubicBc::NotAKnot,
                &[],
                BicubicBc::NotAKnot,
                &[],
                BicubicBc::NotAKnot,
                &[],
            )
            .map(|(coeffs, _, _)| coeffs)
            .map_err(|e| {
                let var_name =
                    EOS_BASIC_RESULT_NAMES.get(v).copied().unwrap_or("var");
                EosError::InvalidState {
                    context,
                    detail: format!(
                        "{table_label}: interp_mkbicub_db failed for var={v} ({var_name}): {e}"
                    ),
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut dense = vec![EosDtVertex::default(); ngrid];
    for (v, coeffs) in fitted.iter().enumerate() {
        let var_name = EOS_BASIC_RESULT_NAMES.get(v).copied().unwrap_or("var");
        for (cell, c4) in coeffs.iter().enumerate() {
            let i1 = cell / n2;
            let i2 = cell % n2;
            for (m, &val) in c4.iter().enumerate() {
                if is_bad(val) {
                    return Err(EosError::InvalidState {
                        context,
                        detail: format!(
                            "non-finite bicubic spline coefficient ({val}) in {table_label} var={v} ({var_name}) cell=({i1}, {i2}) coeff={m}"
                        ),
                    });
                }
                dense[cell].coeffs[v * 4 + m] = val;
            }
        }
    }
    Ok(dense)
}

/// Validates that every fitted spline coefficient in `dense` is finite (`N1-12`).
pub(crate) fn validate_dense_vertices(
    context: &'static str,
    table_label: &str,
    n1: usize,
    n2: usize,
    dense: &[EosDtVertex],
) -> Result<(), EosError> {
    let expected = n1.checked_mul(n2).ok_or_else(|| EosError::TableIo {
        context,
        detail: format!("{table_label}: grid size overflow ({n1} * {n2})"),
    })?;
    if dense.len() != expected {
        return Err(EosError::TableIo {
            context,
            detail: format!(
                "{table_label}: expected {expected} vertices, got {}",
                dense.len()
            ),
        });
    }
    for (cell, vert) in dense.iter().enumerate() {
        let i1 = cell / n2;
        let i2 = cell % n2;
        for (idx, &val) in vert.coeffs.iter().enumerate() {
            if is_bad(val) {
                let var_idx = idx / 4;
                let coeff_idx = idx % 4;
                let var_name = EOS_BASIC_RESULT_NAMES
                    .get(var_idx)
                    .copied()
                    .unwrap_or("var");
                return Err(EosError::TableIo {
                    context,
                    detail: format!(
                        "non-finite spline coefficient ({val}) in {table_label} var={var_idx} ({var_name}) cell=({i1}, {i2}) coeff={coeff_idx}"
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Validates uniform 4D table header parameters before memory allocation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_4d_grid_header(
    context: &'static str,
    table_label: &str,
    x: f64,
    z: f64,
    n1: usize,
    a1_min: f64,
    a1_max: f64,
    d_a1: f64,
    n2: usize,
    a2_min: f64,
    a2_max: f64,
    d_a2: f64,
) -> Result<usize, EosError> {
    if is_bad(x)
        || is_bad(z)
        || is_bad(a1_min)
        || is_bad(a1_max)
        || is_bad(d_a1)
        || is_bad(a2_min)
        || is_bad(a2_max)
        || is_bad(d_a2)
        || n1 < 2
        || n2 < 2
        || d_a1 <= 0.0
        || d_a2 <= 0.0
    {
        return Err(EosError::TableIo {
            context,
            detail: format!(
                "{table_label}: invalid header grid ({n1} x {n2}, X={x}, Z={z}, d1={d_a1}, d2={d_a2})"
            ),
        });
    }
    let ngrid = n1.checked_mul(n2).ok_or_else(|| EosError::TableIo {
        context,
        detail: format!("{table_label}: grid size overflow ({n1} * {n2})"),
    })?;
    let _ = NUM_EOS_TABLE_VALS.checked_mul(ngrid * 8).ok_or_else(|| {
        EosError::TableIo {
            context,
            detail: format!("{table_label}: table allocation size overflow"),
        }
    })?;
    Ok(ngrid)
}
