// SPDX-License-Identifier: LGPL-3.0-or-later
//! Bicubic spline table builder and evaluator for `PC` electron
//! exchange-correlation (`EXCOR7`).
//!
//! Ports upstream MESA `26.4.1` `eos/private/create_EXCOR7_table.f90` and
//! `eos/private/pc_support.f90` (`get_EXCOR7`, `Locate_lnRS`, `Locate_lnGAME`,
//! `Do_Interpolations`, `load_eosPC_support_Info`, `Make_Interpolation_Data`,
//! `Check_Interpolation_Data`).

use std::sync::OnceLock;

use flexmesa_const::{LN10, ONE_SIXTH};
use flexmesa_interp2d::{interp_mkbicub_db, BicubicBc};
use flexmesa_math::{
    atan, cosh, exp, exp10, log, numerics_mode, pow2, pow3, sinh, NumericsMode,
};
use flexmesa_utils::is_bad;

/// Number of `EXCOR7` quantities stored in the bicubic spline table
/// (`pc_support.f90:32-39`).
pub const NVALS_EXCOR7: usize = 7;

/// Variable index for `FXC` (`pc_support.f90:32`).
pub const J_FXC: usize = 0;
/// Variable index for `UXC` (`pc_support.f90:33`).
pub const J_UXC: usize = 1;
/// Variable index for `PXC` (`pc_support.f90:34`).
pub const J_PXC: usize = 2;
/// Variable index for `CVXC` (`pc_support.f90:35`).
pub const J_CVXC: usize = 3;
/// Variable index for `SXC` (`pc_support.f90:36`).
pub const J_SXC: usize = 4;
/// Variable index for `PDTXC` (`pc_support.f90:37`).
pub const J_PDTXC: usize = 5;
/// Variable index for `PDRXC` (`pc_support.f90:38`).
pub const J_PDRXC: usize = 6;

/// Minimum $\log_{10} r_s$ (`create_EXCOR7_table.f90:43`).
pub const EXCOR7_LOG_RS_MIN: f64 = -3.5;
/// Maximum $\log_{10} r_s$ (`create_EXCOR7_table.f90:44`).
pub const EXCOR7_LOG_RS_MAX: f64 = 0.0;
/// Grid spacing $\Delta \log_{10} r_s$ (`create_EXCOR7_table.f90:45`).
pub const EXCOR7_DLOG_RS: f64 = 1.0e-2;
/// Number of $\log_{10} r_s$ grid points (`create_EXCOR7_table.f90:46`).
pub const EXCOR7_N_LN_RS: usize = 351;

/// Minimum $\log_{10} \Gamma_e$ (`create_EXCOR7_table.f90:48`).
pub const EXCOR7_LOG_GAME_MIN: f64 = -2.0;
/// Maximum $\log_{10} \Gamma_e$ (`create_EXCOR7_table.f90:49`).
pub const EXCOR7_LOG_GAME_MAX: f64 = 4.0;
/// Grid spacing $\Delta \log_{10} \Gamma_e$ (`create_EXCOR7_table.f90:50`).
pub const EXCOR7_DLOG_GAME: f64 = 1.0e-2;
/// Number of $\log_{10} \Gamma_e$ grid points (`create_EXCOR7_table.f90:51`).
pub const EXCOR7_N_LN_GAME: usize = 601;

/// Interpolated values and $(\ln r_s, \ln \Gamma_e)$ first derivatives from
/// `get_EXCOR7` (`pc_support.f90:310-383`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Excor7Interp {
    /// Values `[FXC, UXC, PXC, CVXC, SXC, PDTXC, PDRXC]`.
    pub fval: [f64; NVALS_EXCOR7],
    /// Partial derivatives $\partial / \partial \ln r_s$.
    pub df_dln_rs: [f64; NVALS_EXCOR7],
    /// Partial derivatives $\partial / \partial \ln \Gamma_e$.
    pub df_dln_game: [f64; NVALS_EXCOR7],
}

/// Packed `7 x 4` bicubic spline moments `[f, f_xx, f_yy, f_xxyy]` at one
/// `(i_rs, j_game)` grid vertex.
#[derive(Debug, Clone, Copy)]
struct Excor7Vertex {
    coeffs: [[f64; 4]; NVALS_EXCOR7],
}

impl Default for Excor7Vertex {
    fn default() -> Self {
        Self {
            coeffs: [[0.0; 4]; NVALS_EXCOR7],
        }
    }
}

/// Pre-fitted `351 x 601` bicubic spline table for `EXCOR7`
/// (`pc_support.f90:583-901`).
#[derive(Debug, Clone)]
pub struct Excor7Table {
    n_ln_rs: usize,
    ln_rs_min: f64,
    ln_rs_max: f64,
    dln_rs: f64,
    n_ln_game: usize,
    ln_game_min: f64,
    ln_game_max: f64,
    dln_game: f64,
    vertices: Vec<Excor7Vertex>,
}

static EXCOR7_TABLE_STRICT: OnceLock<Excor7Table> = OnceLock::new();
static EXCOR7_TABLE_FAST: OnceLock<Excor7Table> = OnceLock::new();

/// Evaluates the scalar `EXCOR7(RS, GAME)` kernel from
/// `eos/private/create_EXCOR7_table.f90:92-299` (note the `THETA > 0.005`
/// threshold at line 136, distinct from `0.04` in `pc_eos.f90:1920`).
///
/// Returns `[FXC, UXC, PXC, CVXC, SXC, PDTXC, PDRXC]`.
#[must_use]
pub fn excor7_table_point(rs: f64, game: f64) -> [f64; NVALS_EXCOR7] {
    let theta = 0.543 * rs / game;
    let sqth = theta.sqrt();
    let theta2 = theta * theta;
    let theta3 = theta2 * theta;
    let theta4 = theta3 * theta;

    let (t1, t2, t1dh, t1dhh, t2dh, t2dhh) = if theta > 0.005 {
        let cht1 = cosh(1.0 / theta);
        let sht1 = sinh(1.0 / theta);
        let cht2 = cosh(1.0 / sqth);
        let sht2 = sinh(1.0 / sqth);
        let t1 = sht1 / cht1;
        let t2 = sht2 / cht2;
        let t1dh = -1.0 / ((theta * cht1) * (theta * cht1));
        let t1dhh = 2.0 / pow3(theta * cht1) * (cht1 - sht1 / theta);
        let t2dh = -0.5 * sqth / ((theta * cht2) * (theta * cht2));
        let t2dhh = (0.75 * sqth * cht2 - 0.5 * sht2) / pow3(theta * cht2);
        (t1, t2, t1dh, t1dhh, t2dh, t2dhh)
    } else {
        (1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    };

    let a0 = 0.75 + 3.04363 * theta2 - 0.09227 * theta3 + 1.7035 * theta4;
    let a0dh = 6.08726 * theta - 0.27681 * theta2 + 6.814 * theta3;
    let a0dhh = 6.08726 - 0.55362 * theta + 20.442 * theta2;
    let a1 = 1.0 + 8.31051 * theta2 + 5.1105 * theta4;
    let a1dh = 16.62102 * theta + 20.442 * theta3;
    let a1dhh = 16.62102 + 61.326 * theta2;
    let a = 0.610887 * a0 / a1 * t1;
    let ah = a0dh / a0 - a1dh / a1 + t1dh / t1;
    let adh = a * ah;
    let adhh = adh * ah
        + a * (a0dhh / a0 - pow2(a0dh / a0) - a1dhh / a1
            + pow2(a1dh / a1)
            + t1dhh / t1
            - pow2(t1dh / t1));

    let b0 = 0.341308 + 12.070873 * theta2 + 1.148889 * theta4;
    let b0dh = 24.141746 * theta + 4.595556 * theta3;
    let b0dhh = 24.141746 + 13.786668 * theta2;
    let b1 = 1.0 + 10.495346 * theta2 + 1.326623 * theta4;
    let b1dh = 20.990692 * theta + 5.306492 * theta3;
    let b1dhh = 20.990692 + 15.919476 * theta2;
    let b = sqth * t2 * b0 / b1;
    let bh = 0.5 / theta + t2dh / t2 + b0dh / b0 - b1dh / b1;
    let bdh = b * bh;
    let bdhh = bdh * bh
        + b * (-0.5 / theta2 + t2dhh / t2 - pow2(t2dh / t2) + b0dhh / b0
            - pow2(b0dh / b0)
            - b1dhh / b1
            + pow2(b1dh / b1));

    let d0 = 0.614925 + 16.996055 * theta2 + 1.489056 * theta4;
    let d0dh = 33.99211 * theta + 5.956224 * theta3;
    let d0dhh = 33.99211 + 17.868672 * theta2;
    let d1 = 1.0 + 10.10935 * theta2 + 1.22184 * theta4;
    let d1dh = 20.2187 * theta + 4.88736 * theta3;
    let d1dhh = 20.2187 + 14.66208 * theta2;
    let d = sqth * t2 * d0 / d1;
    let dh = 0.5 / theta + t2dh / t2 + d0dh / d0 - d1dh / d1;
    let ddh = d * dh;
    let ddhh = ddh * dh
        + d * (-0.5 / theta2 + t2dhh / t2 - pow2(t2dh / t2) + d0dhh / d0
            - pow2(d0dh / d0)
            - d1dhh / d1
            + pow2(d1dh / d1));

    let e0 = 0.539409 + 2.522206 * theta2 + 0.178484 * theta4;
    let e0dh = 5.044412 * theta + 0.713936 * theta3;
    let e0dhh = 5.044412 + 2.141808 * theta2;
    let e1 = 1.0 + 2.555501 * theta2 + 0.146319 * theta4;
    let e1dh = 5.111002 * theta + 0.585276 * theta3;
    let e1dhh = 5.111002 + 1.755828 * theta2;
    let e = theta * t1 * e0 / e1;
    let eh = 1.0 / theta + t1dh / t1 + e0dh / e0 - e1dh / e1;
    let edh = e * eh;
    let edhh = edh * eh
        + e * (t1dhh / t1 - pow2(t1dh / t1) + e0dhh / e0 - pow2(e0dh / e0)
            - e1dhh / e1
            + pow2(e1dh / e1)
            - 1.0 / theta2);

    let exp1th = exp(-1.0 / theta);
    let c = (0.872496 + 0.025248 * exp1th) * e;
    let cdh = 0.025248 * exp1th / theta2 * e + c * edh / e;
    let cdhh = 0.025248 * exp1th / theta2
        * (edh + (1.0 - 2.0 * theta) / theta2 * e)
        + cdh * edh / e
        + c * edhh / e
        - c * pow2(edh / e);

    let discr = (4.0 * e - d * d).sqrt();
    let didh = 0.5 / discr * (4.0 * edh - 2.0 * d * ddh);
    let didhh = (-pow2((2.0 * edh - d * ddh) / discr) + 2.0 * edhh
        - ddh * ddh
        - d * ddhh)
        / discr;

    let s1 = -c / e * game;
    let s1h = cdh / c - edh / e;
    let s1dh = s1 * s1h;
    let s1dhh = s1dh * s1h
        + s1 * (cdhh / c - pow2(cdh / c) - edhh / e + pow2(edh / e));
    let s1dg = -c / e;
    let s1dhg = s1dg * (cdh / c - edh / e);

    let b2 = b - c * d / e;
    let b2dh = bdh - (cdh * d + c * ddh) / e + c * d * edh / (e * e);
    let b2dhh = bdhh - (cdhh * d + 2.0 * cdh * ddh + c * ddhh) / e
        + (2.0 * (cdh * d + c * ddh - c * d * edh / e) * edh + c * d * edhh)
            / (e * e);

    let sqge = game.sqrt();
    let s2 = -2.0 / e * b2 * sqge;
    let s2h = b2dh / b2 - edh / e;
    let s2dh = s2 * s2h;
    let s2dhh = s2dh * s2h
        + s2 * (b2dhh / b2 - pow2(b2dh / b2) - edhh / e + pow2(edh / e));
    let s2dg = 0.5 * s2 / game;
    let s2dgg = -0.5 * s2dg / game;
    let s2dhg = 0.5 * s2dh / game;

    let r3 = e * game + d * sqge + 1.0;
    let r3dh = edh * game + ddh * sqge;
    let r3dhh = edhh * game + ddhh * sqge;
    let r3dg = e + 0.5 * d / sqge;
    let r3dgg = -0.25 * d / (game * sqge);
    let r3dhg = edh + 0.5 * ddh / sqge;

    let b3 = a - c / e;
    let b3dh = adh - cdh / e + c * edh / (e * e);
    let b3dhh = adhh - cdhh / e + (2.0 * cdh * edh + c * edhh) / (e * e)
        - 2.0 * c * edh * edh / pow3(e);

    let c3 = (d / e * b2 - b3) / e;
    let c3dh = (ddh * b2 + d * b2dh + b3 * edh) / (e * e)
        - 2.0 * d * b2 * edh / pow3(e)
        - b3dh / e;
    let c3dhh = (-b3dhh
        + (ddhh * b2
            + 2.0 * ddh * b2dh
            + d * b2dhh
            + b3dh * edh
            + b3 * edhh
            + b3dh * edh)
            / e
        - 2.0
            * ((ddh * b2 + d * b2dh + b3 * edh + ddh * b2 + d * b2dh) * edh
                + d * b2 * edhh)
            / (e * e)
        + 6.0 * d * b2 * edh * edh / pow3(e))
        / e;

    let s3 = c3 * log(r3);
    let s3dh = s3 * c3dh / c3 + c3 * r3dh / r3;
    let s3dhh = (s3dh * c3dh + s3 * c3dhh) / c3 - s3 * pow2(c3dh / c3)
        + (c3dh * r3dh + c3 * r3dhh) / r3
        - c3 * pow2(r3dh / r3);
    let s3dg = c3 * r3dg / r3;
    let s3dgg = c3 * (r3dgg / r3 - pow2(r3dg / r3));
    let s3dhg = (c3dh * r3dg + c3 * r3dhg) / r3 - c3 * r3dg * r3dh / (r3 * r3);

    let b4 = 2.0 - d * d / e;
    let b4dh = edh * (d / e) * (d / e) - 2.0 * d * ddh / e;
    let b4dhh = edhh * (d / e) * (d / e)
        + 2.0 * edh * (d / e) * (d / e) * (ddh / d - edh / e)
        - 2.0 * (ddh * ddh + d * ddhh) / e
        + 2.0 * d * ddh * edh / (e * e);

    let c4 = 2.0 * e * sqge + d;
    let c4dh = 2.0 * edh * sqge + ddh;
    let c4dhh = 2.0 * edhh * sqge + ddhh;
    let c4dg = e / sqge;
    let c4dgg = -0.5 * e / (game * sqge);
    let c4dhg = edh / sqge;

    let s4a = 2.0 / e / discr;
    let s4ah = edh / e + didh / discr;
    let s4adh = -s4a * s4ah;
    let s4adhh = -s4adh * s4ah
        - s4a
            * (edhh / e - (edh / e) * (edh / e) + didhh / discr
                - pow2(didh / discr));

    let s4b = d * b3 + b4 * b2;
    let s4bdh = ddh * b3 + d * b3dh + b4dh * b2 + b4 * b2dh;
    let s4bdhh = ddhh * b3
        + 2.0 * ddh * b3dh
        + d * b3dhh
        + b4dhh * b2
        + 2.0 * b4dh * b2dh
        + b4 * b2dhh;

    let s4c = atan(c4 / discr) - atan(d / discr);
    let up1 = c4dh * discr - c4 * didh;
    let dn1 = discr * discr + c4 * c4;
    let up2 = ddh * discr - d * didh;
    let dn2 = discr * discr + d * d;
    let s4cdh = up1 / dn1 - up2 / dn2;
    let s4cdhh = (c4dhh * discr - c4 * didhh) / dn1
        - up1 * 2.0 * (discr * didh + c4 * c4dh) / (dn1 * dn1)
        - (ddhh * discr - d * didhh) / dn2
        + up2 * 2.0 * (discr * didh + d * ddh) / (dn2 * dn2);
    let s4cdg = c4dg * discr / dn1;
    let s4cdgg = c4dgg * discr / dn1 - 2.0 * c4 * discr * pow2(c4dg / dn1);
    let s4cdhg = (c4dhg * discr + c4dg * didh
        - c4dg * discr / dn1 * 2.0 * (discr * didh + c4 * c4dh))
        / dn1;

    let s4 = s4a * s4b * s4c;
    let s4dh = s4adh * s4b * s4c + s4a * s4bdh * s4c + s4a * s4b * s4cdh;
    let s4dhh = s4adhh * s4b * s4c
        + s4a * s4bdhh * s4c
        + s4a * s4b * s4cdhh
        + 2.0 * (s4adh * s4bdh * s4c + s4adh * s4b * s4cdh + s4a * s4bdh * s4cdh);
    let s4dg = s4a * s4b * s4cdg;
    let s4dgg = s4a * s4b * s4cdgg;
    let s4dhg = s4a * s4b * s4cdhg + s4cdg * (s4adh * s4b + s4a * s4bdh);

    let fxc = s1 + s2 + s3 + s4;
    let fxcdh = s1dh + s2dh + s3dh + s4dh;
    let fxcdg = s1dg + s2dg + s3dg + s4dg;
    let fxcdhh = s1dhh + s2dhh + s3dhh + s4dhh;
    let fxcdgg = s2dgg + s3dgg + s4dgg;
    let fxcdhg = s1dhg + s2dhg + s3dhg + s4dhg;

    let pxc = (game * fxcdg - 2.0 * theta * fxcdh) / 3.0;
    let uxc = game * fxcdg - theta * fxcdh;
    let mut sxc = (game * s2dg - s2 + game * s3dg - s3
        + s4a * s4b * (game * s4cdg - s4c))
        - theta * fxcdh;
    if sxc.abs() < 1.0e-9 * (theta * fxcdh).abs() {
        sxc = 0.0;
    }
    let mut cvxc = 2.0 * theta * (game * fxcdhg - fxcdh)
        - theta * theta * fxcdhh
        - game * game * fxcdgg;
    if cvxc.abs() < 1.0e-9 * (game * game * fxcdgg).abs() {
        cvxc = 0.0;
    }
    let pdlh = theta * (game * fxcdhg - 2.0 * fxcdh - 2.0 * theta * fxcdhh) / 3.0;
    let pdlg = game * (fxcdg + game * fxcdgg - 2.0 * theta * fxcdhg) / 3.0;
    let pdrxc = pxc + (pdlg - 2.0 * pdlh) / 3.0;
    let pdtxc = game * (theta * fxcdhg - game * fxcdgg / 3.0)
        - theta * (fxcdh / 0.75 + theta * fxcdhh / 1.5);

    [fxc, uxc, pxc, cvxc, sxc, pdtxc, pdrxc]
}

/// Locates the 0-based cell index and cell bounds `(v0, v_clamped, v1)`
/// matching upstream `Locate_lnRS` / `Locate_lnGAME` (`pc_support.f90:386-437`).
#[inline]
fn locate_axis(
    mut val: f64,
    n: usize,
    v_min: f64,
    dv: f64,
) -> (usize, f64, f64, f64) {
    let iq = ((val - v_min) / dv + 1.0e-4) as isize + 1;
    let (iq_1b, v0, v1) = if iq < 1 || (iq as usize) >= n {
        if iq < 1 {
            let v0 = v_min;
            let v1 = v0 + dv;
            val = v0;
            (1_usize, v0, v1)
        } else {
            let iq_clamped = n - 1;
            let v0 = v_min + ((iq_clamped - 1) as f64) * dv;
            let v1 = v0 + dv;
            val = v1;
            (iq_clamped, v0, v1)
        }
    } else {
        let iq_u = iq as usize;
        let v0 = v_min + ((iq_u - 1) as f64) * dv;
        let v1 = v0 + dv;
        (iq_u, v0, v1)
    };
    (iq_1b - 1, v0, val, v1)
}

impl Excor7Table {
    /// Builds the `351 x 601` `EXCOR7` bicubic spline table matching
    /// `create_EXCOR7_table.f90` + `pc_support.f90:658-901`.
    ///
    /// # Panics
    ///
    /// Panics if the internal `interp_mkbicub_db` bicubic spline fit fails on
    /// the static `351 x 601` strictly increasing uniform grid (which cannot
    /// occur for valid grid dimensions).
    #[must_use]
    pub fn build() -> Self {
        let n_ln_rs = EXCOR7_N_LN_RS;
        let n_ln_game = EXCOR7_N_LN_GAME;
        let ln_rs_min = EXCOR7_LOG_RS_MIN * LN10;
        let ln_rs_max = EXCOR7_LOG_RS_MAX * LN10;
        let dln_rs = EXCOR7_DLOG_RS * LN10;
        let ln_game_min = EXCOR7_LOG_GAME_MIN * LN10;
        let ln_game_max = EXCOR7_LOG_GAME_MAX * LN10;
        let dln_game = EXCOR7_DLOG_GAME * LN10;

        // Axis grids built by repeated addition as in `pc_support.f90:685-695`.
        let mut ln_rs_grid = vec![0.0_f64; n_ln_rs];
        ln_rs_grid[0] = ln_rs_min;
        for i in 1..(n_ln_rs - 1) {
            ln_rs_grid[i] = ln_rs_grid[i - 1] + dln_rs;
        }
        ln_rs_grid[n_ln_rs - 1] = ln_rs_max;

        let mut ln_game_grid = vec![0.0_f64; n_ln_game];
        ln_game_grid[0] = ln_game_min;
        for j in 1..(n_ln_game - 1) {
            ln_game_grid[j] = ln_game_grid[j - 1] + dln_game;
        }
        ln_game_grid[n_ln_game - 1] = ln_game_max;

        // Precompute `GAME = exp10(logGAME)` along the inner axis once.
        let mut game_vals = vec![0.0_f64; n_ln_game];
        for (j, g_slot) in game_vals.iter_mut().enumerate() {
            let log_game = EXCOR7_LOG_GAME_MIN + (j as f64) * EXCOR7_DLOG_GAME;
            *g_slot = exp10(log_game);
        }

        let ngrid = n_ln_rs * n_ln_game;
        let mut raw_vars = vec![vec![0.0_f64; ngrid]; NVALS_EXCOR7];
        for i in 0..n_ln_rs {
            let log_rs = EXCOR7_LOG_RS_MIN + (i as f64) * EXCOR7_DLOG_RS;
            let rs = exp10(log_rs);
            let row_offset = i * n_ln_game;
            for (j, &game) in game_vals.iter().enumerate() {
                let vals = excor7_table_point(rs, game);
                let cell = row_offset + j;
                for v in 0..NVALS_EXCOR7 {
                    raw_vars[v][cell] = vals[v];
                }
            }
        }

        // Fit 7 bicubic splines sequentially with not-a-knot BCs (`pc_support.f90:853-899`).
        // Avoid nested rayon parallelism inside `OnceLock` called from zone loops.
        let mut vertices = vec![Excor7Vertex::default(); ngrid];
        for (v, z_slice) in raw_vars.iter().enumerate() {
            let (coeffs, _, _) = interp_mkbicub_db(
                &ln_rs_grid,
                &ln_game_grid,
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
            .expect("EXCOR7 351x601 bicubic spline construction must succeed");

            // Mirror `Check_Interpolation_Data` (`pc_support.f90:800-813`).
            for (cell, c4) in coeffs.iter().enumerate() {
                let mut clean = *c4;
                for val in &mut clean {
                    if is_bad(*val) {
                        *val = 0.0;
                    }
                }
                vertices[cell].coeffs[v] = clean;
            }
        }

        Self {
            n_ln_rs,
            ln_rs_min,
            ln_rs_max,
            dln_rs,
            n_ln_game,
            ln_game_min,
            ln_game_max,
            dln_game,
            vertices,
        }
    }

    /// Returns the process-wide cached `EXCOR7` table for the active
    /// [`NumericsMode`], preventing `Fast`-built coefficients from leaking into
    /// `Strict` mode.
    #[must_use]
    pub fn global() -> &'static Self {
        match numerics_mode() {
            NumericsMode::Strict => EXCOR7_TABLE_STRICT.get_or_init(Self::build),
            NumericsMode::Fast => EXCOR7_TABLE_FAST.get_or_init(Self::build),
        }
    }

    /// Evaluates `get_EXCOR7(RS, GAME)` (`pc_support.f90:310-383`).
    ///
    /// Returns `None` (`skip = .true.`) when `(ln RS, ln GAME)` lies outside
    /// `[lnRS_min, lnRS_max] x [lnGAME_min, lnGAME_max]` or is non-finite.
    #[must_use]
    pub fn eval(&self, rs: f64, game: f64) -> Option<Excor7Interp> {
        let ln_rs = log(rs);
        let ln_game = log(game);
        if !(ln_rs >= self.ln_rs_min
            && ln_rs <= self.ln_rs_max
            && ln_game >= self.ln_game_min
            && ln_game <= self.ln_game_max)
        {
            return None;
        }

        let (i_rs, x0, xget, x1) =
            locate_axis(ln_rs, self.n_ln_rs, self.ln_rs_min, self.dln_rs);
        let (j_game, y0, yget, y1) = locate_axis(
            ln_game,
            self.n_ln_game,
            self.ln_game_min,
            self.dln_game,
        );

        let ny = self.n_ln_game;
        let v_ij = &self.vertices[i_rs * ny + j_game];
        let v_i_jp1 = &self.vertices[i_rs * ny + j_game + 1];
        let v_ip1_j = &self.vertices[(i_rs + 1) * ny + j_game];
        let v_ip1_jp1 = &self.vertices[(i_rs + 1) * ny + j_game + 1];

        // Exact operation order of `Do_Interpolations` (`pc_support.f90:469-578`).
        let hx = x1 - x0;
        let hxi = 1.0 / hx;
        let hx2 = hx * hx;
        let xp = (xget - x0) * hxi;
        let xpi = 1.0 - xp;
        let xp2 = xp * xp;
        let xpi2 = xpi * xpi;
        let cx = xp * (xp2 - 1.0);
        let cxi = xpi * (xpi2 - 1.0);
        let cxd = 3.0 * xp2 - 1.0;
        let cxdi = -3.0 * xpi2 + 1.0;

        let hy = y1 - y0;
        let hyi = 1.0 / hy;
        let hy2 = hy * hy;
        let yp = (yget - y0) * hyi;
        let ypi = 1.0 - yp;
        let yp2 = yp * yp;
        let ypi2 = ypi * ypi;
        let cy = yp * (yp2 - 1.0);
        let cyi = ypi * (ypi2 - 1.0);
        let cyd = 3.0 * yp2 - 1.0;
        let cydi = -3.0 * ypi2 + 1.0;

        let sixth_hx2 = ONE_SIXTH * hx2;
        let sixth_hy2 = ONE_SIXTH * hy2;
        let z36th_hx2_hy2 = sixth_hx2 * sixth_hy2;

        let sixth_hx = ONE_SIXTH * hx;
        let sixth_hxi_hy2 = ONE_SIXTH * hxi * hy2;
        let z36th_hx_hy2 = sixth_hx * sixth_hy2;

        let sixth_hx2_hyi = sixth_hx2 * hyi;
        let sixth_hy = ONE_SIXTH * hy;
        let z36th_hx2_hy = sixth_hx2 * sixth_hy;

        let mut fval = [0.0_f64; NVALS_EXCOR7];
        let mut df_dln_rs = [0.0_f64; NVALS_EXCOR7];
        let mut df_dln_game = [0.0_f64; NVALS_EXCOR7];

        for k in 0..NVALS_EXCOR7 {
            let [f1_ij, f2_ij, f3_ij, f4_ij] = v_ij.coeffs[k];
            let [f1_ip1_j, f2_ip1_j, f3_ip1_j, f4_ip1_j] = v_ip1_j.coeffs[k];
            let [f1_i_jp1, f2_i_jp1, f3_i_jp1, f4_i_jp1] = v_i_jp1.coeffs[k];
            let [f1_ip1_jp1, f2_ip1_jp1, f3_ip1_jp1, f4_ip1_jp1] =
                v_ip1_jp1.coeffs[k];

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

            df_dln_rs[k] = hxi
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

            df_dln_game[k] = hyi
                * (xpi * (-f1_ij + f1_i_jp1) + xp * (-f1_ip1_j + f1_ip1_jp1))
                + sixth_hx2_hyi
                    * (cxi * (-f2_ij + f2_i_jp1)
                        + cx * (-f2_ip1_j + f2_ip1_jp1))
                + sixth_hy
                    * (xpi * (cydi * f3_ij + cyd * f3_i_jp1)
                        + xp * (cydi * f3_ip1_j + cyd * f3_ip1_jp1))
                + z36th_hx2_hy
                    * (cxi * (cydi * f4_ij + cyd * f4_i_jp1)
                        + cx * (cydi * f4_ip1_j + cyd * f4_ip1_jp1));
        }

        Some(Excor7Interp {
            fval,
            df_dln_rs,
            df_dln_game,
        })
    }

    /// Returns the raw table value at 0-based grid node `(i_rs, j_game)` for
    /// variable `v` (`0..7`), used for diagnostics and parity checks.
    #[must_use]
    pub fn node_value(&self, i_rs: usize, j_game: usize, var: usize) -> f64 {
        self.vertices[i_rs * self.n_ln_game + j_game].coeffs[var][0]
    }
}

/// Evaluates `get_EXCOR7(RS, GAME)` against the process-wide cached bicubic
/// spline table (`pc_support.f90:310-383`).
#[must_use]
pub fn get_excor7(rs: f64, game: f64) -> Option<Excor7Interp> {
    Excor7Table::global().eval(rs, game)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pc::excor::{excor7, excor7_analytic};
    use flexmesa_autodiff::AutoDiffReal2VarOrder1;

    #[test]
    fn test_excor7_table_coefficients_finite_and_reproduce_nodes() {
        let tbl = Excor7Table::global();
        assert_eq!(tbl.vertices.len(), EXCOR7_N_LN_RS * EXCOR7_N_LN_GAME);
        for vert in &tbl.vertices {
            for c4 in &vert.coeffs {
                for &val in c4 {
                    assert!(
                        !is_bad(val),
                        "EXCOR7 spline coefficient must be finite, got {val}"
                    );
                }
            }
        }

        // Verify grid nodes reproduce `excor7_table_point` and spline eval matches nodes.
        for (i, j) in [(0, 0), (100, 150), (200, 350), (349, 599), (350, 600)] {
            let log_rs = EXCOR7_LOG_RS_MIN + (i as f64) * EXCOR7_DLOG_RS;
            let log_game = EXCOR7_LOG_GAME_MIN + (j as f64) * EXCOR7_DLOG_GAME;
            let rs = exp10(log_rs);
            let game = exp10(log_game);
            let direct = excor7_table_point(rs, game);
            let ev = tbl.eval(rs, game).expect("in-bounds grid node");
            for v in 0..NVALS_EXCOR7 {
                assert_eq!(tbl.node_value(i, j, v), direct[v]);
                let tol = 1.0e-12 * direct[v].abs().max(1.0);
                assert!(
                    (ev.fval[v] - direct[v]).abs() <= tol,
                    "node ({i}, {j}) var {v}: ev={} direct={}",
                    ev.fval[v],
                    direct[v]
                );
            }
        }
    }

    #[test]
    fn test_excor7_table_derivatives_match_central_fd_and_bounds_skip() {
        // Out-of-bounds and non-finite inputs must return `None` (`skip = .true.`).
        assert!(get_excor7(1.0e-5, 1.0).is_none());
        assert!(get_excor7(2.0, 1.0).is_none());
        assert!(get_excor7(0.01, 1.0e-3).is_none());
        assert!(get_excor7(0.01, 1.0e5).is_none());
        assert!(get_excor7(0.0, 1.0).is_none());
        assert!(get_excor7(f64::NAN, 1.0).is_none());

        // Fallback to analytical `EXCOR7` outside table bounds.
        let rs_out = AutoDiffReal2VarOrder1 {
            val: 2.5,
            d1: [0.0, -0.8],
        };
        let game_out = AutoDiffReal2VarOrder1 {
            val: 1.5,
            d1: [-0.2, 0.5],
        };
        let got_out = excor7(rs_out, game_out);
        let ana_out = excor7_analytic(rs_out, game_out);
        assert_eq!(got_out.f.val, ana_out.f.val);
        assert_eq!(got_out.f.d1, ana_out.f.d1);
        assert_eq!(got_out.pdr.val, ana_out.pdr.val);
        assert_eq!(got_out.pdr.d1, ana_out.pdr.d1);

        // Interior bicubic spline derivatives must match central finite differences.
        let rs0 = exp10(-1.734);
        let game0 = exp10(0.847);
        let ev0 = get_excor7(rs0, game0).expect("interior point");
        let h = 1.0e-6;
        let ev_rp = get_excor7(rs0 * exp(h), game0).expect("rs+h");
        let ev_rm = get_excor7(rs0 * exp(-h), game0).expect("rs-h");
        let ev_gp = get_excor7(rs0, game0 * exp(h)).expect("game+h");
        let ev_gm = get_excor7(rs0, game0 * exp(-h)).expect("game-h");
        for k in 0..NVALS_EXCOR7 {
            let fd_rs = (ev_rp.fval[k] - ev_rm.fval[k]) / (2.0 * h);
            let fd_gm = (ev_gp.fval[k] - ev_gm.fval[k]) / (2.0 * h);
            assert!(
                (ev0.df_dln_rs[k] - fd_rs).abs()
                    <= 1.0e-7 * fd_rs.abs().max(1.0e-12),
                "var {k} df/dlnRS: spline={} fd={}",
                ev0.df_dln_rs[k],
                fd_rs
            );
            assert!(
                (ev0.df_dln_game[k] - fd_gm).abs()
                    <= 1.0e-7 * fd_gm.abs().max(1.0e-12),
                "var {k} df/dlnGAME: spline={} fd={}",
                ev0.df_dln_game[k],
                fd_gm
            );
        }
    }
}
