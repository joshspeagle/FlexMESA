// SPDX-License-Identifier: LGPL-3.0-or-later
//! Analytical blend fraction and partial-derivative evaluators for `FreeEOS` and `OPAL/SCVH`.
//!
//! Ports `Get_FreeEOS_alfa` and `get_opal_scvh_alfa_and_partials` from
//! `eos/private/eosdt_eval.f90` (lines 808..1461) along with the default control
//! parameters from `eos/defaults/eos.defaults` (lines 84..152).
//!
//! Each evaluator returns `(alfa, d_alfa_dlog_t, d_alfa_dlog_rho)`, where
//! $\alpha \in [0, 1]$ is the raw fallback fraction (`0.0` = 100% of this EOS
//! component, `1.0` = 100% of the next level in the cascade) prior to quintic
//! smoothing in `combine_two_eos_components`.

use crate::types::EosError;
use flexmesa_autodiff::AutoDiffReal2VarOrder1;

/// Configuration parameters for `FreeEOS` domain and cut boundaries (`eos.defaults`).
///
/// Matches the default `FreeEOS` namelist controls in `eos/defaults/eos.defaults`
/// (lines 84..125).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FreeEosConfig {
    /// Upper $\log_{10} Q$ outer boundary where `FreeEOS` is 0% (default `5.6`).
    pub log_q_max_hi: f64,
    /// Upper $\log_{10} Q$ inner boundary where `FreeEOS` is 100% (default `5.5`).
    pub log_q_max_lo: f64,
    /// Lower $\log_{10} Q$ inner boundary where `FreeEOS` is 100% (default `-9.9`).
    pub log_q_min_hi: f64,
    /// Lower $\log_{10} Q$ outer boundary where `FreeEOS` is 0% (default `-10.0`).
    pub log_q_min_lo: f64,
    /// Lower $\log_{10} \rho$ inner boundary where `FreeEOS` is 100% (default `-18.0`).
    pub log_rho_min_hi: f64,
    /// Lower $\log_{10} \rho$ outer boundary where `FreeEOS` is 0% (default `-18.1`).
    pub log_rho_min_lo: f64,
    /// Upper $\log_{10} \rho$ outer boundary where `FreeEOS` is 0% (default `10.1`).
    pub log_rho_max_hi: f64,
    /// Upper $\log_{10} \rho$ inner boundary where `FreeEOS` is 100% (default `10.0`).
    pub log_rho_max_lo: f64,
    /// Lower $\log_{10} T$ inner boundary where `FreeEOS` is 100% (default `3.1`).
    pub log_t_min_hi: f64,
    /// Lower $\log_{10} T$ outer boundary where `FreeEOS` is 0% (default `3.0`).
    pub log_t_min_lo: f64,
    /// Upper $\log_{10} T$ outer boundary where `FreeEOS` is 0% (default `8.2`).
    pub log_t_max_hi: f64,
    /// Upper $\log_{10} T$ inner boundary where `FreeEOS` is 100% (default `8.1`).
    pub log_t_max_lo: f64,
    /// Metallicity threshold $Z$ separating low-$Z$ and high-$Z$ $\log_{10} Q$ cuts (default `0.3`).
    pub log_q_cut_lo_z_max: f64,
    /// Low-$Z$ $\log_{10} Q$ cut upper boundary where cut is 0% `FreeEOS` (default `0.5`).
    pub log_q_cut_lo_z_hi: f64,
    /// Low-$Z$ $\log_{10} Q$ cut lower boundary where cut is 100% `FreeEOS` (default `-0.5`).
    pub log_q_cut_lo_z_lo: f64,
    /// High-$Z$ $\log_{10} Q$ cut upper boundary where cut is 0% `FreeEOS` (default `-0.5`).
    pub log_q_cut_hi_z_hi: f64,
    /// High-$Z$ $\log_{10} Q$ cut lower boundary where cut is 100% `FreeEOS` (default `-1.5`).
    pub log_q_cut_hi_z_lo: f64,
    /// Density cut upper boundary $\log_{10} \rho$ where cut is 0% `FreeEOS` (default `-5.5`).
    pub log_rho_cut_hi: f64,
    /// Density cut lower boundary $\log_{10} \rho$ where cut is 100% `FreeEOS` (default `-6.5`).
    pub log_rho_cut_lo: f64,
    /// Temperature cut upper boundary $\log_{10} T$ where cut is 100% `FreeEOS` (default `6.5`).
    pub log_t_cut_hi: f64,
    /// Temperature cut lower boundary $\log_{10} T$ where cut is 0% `FreeEOS` (default `6.4`).
    pub log_t_cut_lo: f64,
    /// Use linear instead of quadratic/cubic interpolation in $X$ (default `false`).
    pub use_linear_interp_for_x: bool,
    /// Numerical tolerance for table coordinate comparisons (default `1e-6`).
    pub tiny_fuzz: f64,
}

impl Default for FreeEosConfig {
    fn default() -> Self {
        Self {
            log_q_max_hi: 5.6,
            log_q_max_lo: 5.5,
            log_q_min_hi: -9.9,
            log_q_min_lo: -10.0,
            log_rho_min_hi: -18.0,
            log_rho_min_lo: -18.1,
            log_rho_max_hi: 10.1,
            log_rho_max_lo: 10.0,
            log_t_min_hi: 3.1,
            log_t_min_lo: 3.0,
            log_t_max_hi: 8.2,
            log_t_max_lo: 8.1,
            log_q_cut_lo_z_max: 0.3,
            log_q_cut_lo_z_hi: 0.5,
            log_q_cut_lo_z_lo: -0.5,
            log_q_cut_hi_z_hi: -0.5,
            log_q_cut_hi_z_lo: -1.5,
            log_rho_cut_hi: -5.5,
            log_rho_cut_lo: -6.5,
            log_t_cut_hi: 6.5,
            log_t_cut_lo: 6.4,
            use_linear_interp_for_x: false,
            tiny_fuzz: 1.0e-6,
        }
    }
}

/// Configuration parameters for `OPAL/SCVH` domain and blend boundaries (`eos.defaults`).
///
/// Matches the default `OPAL/SCVH` namelist controls in `eos/defaults/eos.defaults`
/// (lines 127..152).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpalScvhConfig {
    /// Lower $\log_{10} T$ boundary where fallback (`HELM`/`ideal`) is 100% (`logT8`, default `2.2`).
    pub log_t_low_all_helm: f64,
    /// Lower $\log_{10} T$ boundary where `SCVH` is 100% (`logT7`, default `2.3`).
    pub log_t_low_all_scvh: f64,
    /// Upper $\log_{10} T$ boundary where `OPAL` is 100% (`logT2`, default `7.5`).
    pub log_t_all_opal: f64,
    /// Upper $\log_{10} T$ boundary where `HELM` is 100% (`logT1`, default `7.6`).
    pub log_t_all_helm: f64,
    /// Upper $\log_{10} \rho$ limit where `OPAL/SCVH` is 0% (`logRho1`, default `3.50`).
    pub log_rho1_opal_scvh_limit: f64,
    /// Upper $\log_{10} \rho$ limit where `OPAL/SCVH` is 100% (`logRho2`, default `3.48`).
    pub log_rho2_opal_scvh_limit: f64,
    /// Minimum $\log_{10} \rho$ limit for `OPAL/SCVH` (`logRho5`, default `-14.299`).
    pub log_rho_min_opal_scvh_limit: f64,
    /// Maximum $\log_{10} Q$ limit for `OPAL/SCVH` (`logQmax`, default `5.3`).
    pub log_q_max_opal_scvh: f64,
    /// Minimum $\log_{10} Q$ limit for `OPAL/SCVH` (`logQ3 = logQ4`, default `-8.0`).
    pub log_q_min_opal_scvh: f64,
    /// Metallicity $Z$ below which `OPAL/SCVH` is unblended in $Z$ (`Z_no_HELM`, default `0.035`).
    pub z_all_opal: f64,
    /// Metallicity $Z$ above which `OPAL/SCVH` is 0% (`Z_all_HELM`, default `0.040`).
    pub z_all_helm: f64,
    /// Use linear instead of quadratic/cubic interpolation in $X$ (default `false`).
    pub use_linear_interp_for_x: bool,
    /// Numerical tolerance for table coordinate comparisons (default `1e-6`).
    pub tiny_fuzz: f64,
}

impl Default for OpalScvhConfig {
    fn default() -> Self {
        Self {
            log_t_low_all_helm: 2.2,
            log_t_low_all_scvh: 2.3,
            log_t_all_opal: 7.5,
            log_t_all_helm: 7.6,
            log_rho1_opal_scvh_limit: 3.50,
            log_rho2_opal_scvh_limit: 3.48,
            log_rho_min_opal_scvh_limit: -14.299,
            log_q_max_opal_scvh: 5.3,
            log_q_min_opal_scvh: -8.0,
            z_all_opal: 0.035,
            z_all_helm: 0.040,
            use_linear_interp_for_x: false,
            tiny_fuzz: 1.0e-6,
        }
    }
}

/// Evaluates a trapezoidal 1D ramp in `AutoDiffReal2VarOrder1` rising on `[lo_0, lo_1]`
/// and falling on `[hi_1, hi_0]`.
#[inline]
fn trapezoidal_ramp(
    x: AutoDiffReal2VarOrder1,
    lo_0: f64,
    lo_1: f64,
    hi_1: f64,
    hi_0: f64,
) -> AutoDiffReal2VarOrder1 {
    if x.val < lo_0 {
        AutoDiffReal2VarOrder1::zero()
    } else if x.val < lo_1 {
        (x - lo_0) / (lo_1 - lo_0)
    } else if x.val < hi_1 {
        AutoDiffReal2VarOrder1::constant(1.0)
    } else if x.val < hi_0 {
        (x - hi_0) / (hi_1 - hi_0)
    } else {
        AutoDiffReal2VarOrder1::zero()
    }
}

/// Computes the raw `FreeEOS` fallback fraction `(alfa, d_alfa_dlogT, d_alfa_dlogRho)`.
///
/// Mirrors `Get_FreeEOS_alfa` in `eos/private/eosdt_eval.f90` (lines 808..973).
/// Here `alfa = 0` means 100% `FreeEOS` and `alfa = 1` means 100% fallback (`OPAL/SCVH` or `HELM`).
#[must_use]
pub fn get_free_eos_alfa(
    cfg: &FreeEosConfig,
    log_rho: f64,
    log_t: f64,
    z: f64,
) -> (f64, f64, f64) {
    let log_rho_auto = AutoDiffReal2VarOrder1::var1(log_rho);
    let log_t_auto = AutoDiffReal2VarOrder1::var2(log_t);
    let log_q_auto = log_rho_auto - 2.0 * log_t_auto + 12.0;

    let blend_log_t = trapezoidal_ramp(
        log_t_auto,
        cfg.log_t_min_lo,
        cfg.log_t_min_hi,
        cfg.log_t_max_lo,
        cfg.log_t_max_hi,
    );
    let blend_log_rho = trapezoidal_ramp(
        log_rho_auto,
        cfg.log_rho_min_lo,
        cfg.log_rho_min_hi,
        cfg.log_rho_max_lo,
        cfg.log_rho_max_hi,
    );
    let blend_log_q = trapezoidal_ramp(
        log_q_auto,
        cfg.log_q_min_lo,
        cfg.log_q_min_hi,
        cfg.log_q_max_lo,
        cfg.log_q_max_hi,
    );

    let blend_log_rho_cut = if log_rho_auto.val < cfg.log_rho_cut_lo {
        AutoDiffReal2VarOrder1::constant(1.0)
    } else if log_rho_auto.val < cfg.log_rho_cut_hi {
        (log_rho_auto - cfg.log_rho_cut_hi)
            / (cfg.log_rho_cut_lo - cfg.log_rho_cut_hi)
    } else {
        AutoDiffReal2VarOrder1::zero()
    };

    let blend_log_t_cut = if log_t_auto.val < cfg.log_t_cut_lo {
        AutoDiffReal2VarOrder1::zero()
    } else if log_t_auto.val < cfg.log_t_cut_hi {
        (log_t_auto - cfg.log_t_cut_lo) / (cfg.log_t_cut_hi - cfg.log_t_cut_lo)
    } else {
        AutoDiffReal2VarOrder1::constant(1.0)
    };

    let (log_q_cut_hi, log_q_cut_lo) = if z <= cfg.log_q_cut_lo_z_max {
        (cfg.log_q_cut_lo_z_hi, cfg.log_q_cut_lo_z_lo)
    } else {
        (cfg.log_q_cut_hi_z_hi, cfg.log_q_cut_hi_z_lo)
    };

    let blend_log_q_cut = if log_q_auto.val < log_q_cut_lo {
        AutoDiffReal2VarOrder1::constant(1.0)
    } else if log_q_auto.val < log_q_cut_hi {
        (log_q_auto - log_q_cut_hi) / (log_q_cut_lo - log_q_cut_hi)
    } else {
        AutoDiffReal2VarOrder1::zero()
    };

    let blend_cut = if blend_log_t_cut.val == 0.0 {
        if blend_log_q_cut.val == 0.0 {
            blend_log_rho_cut
        } else if blend_log_q_cut.val == 1.0 {
            AutoDiffReal2VarOrder1::constant(1.0)
        } else {
            blend_log_rho_cut.max(blend_log_q_cut)
        }
    } else if blend_log_t_cut.val == 1.0 {
        AutoDiffReal2VarOrder1::constant(1.0)
    } else if blend_log_q_cut.val == 0.0 {
        blend_log_t_cut
    } else if blend_log_q_cut.val == 1.0 {
        AutoDiffReal2VarOrder1::constant(1.0)
    } else {
        blend_log_t_cut.max(blend_log_q_cut)
    };

    let blend = blend_cut * blend_log_t * blend_log_rho * blend_log_q;
    (1.0 - blend.val, -blend.d1[1], -blend.d1[0])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpalScvhRegion {
    UseNone,
    UseAll,
    BlendInX,
    BlendInY,
    BlendCornerOut,
    BlendCornerIn,
    BlendDiagonal,
}

/// Computes the raw `OPAL/SCVH` fallback fraction `(alfa, d_alfa_dlogRho, d_alfa_dlogT)`.
///
/// Mirrors `get_opal_scvh_alfa_and_partials` in `eos/private/eosdt_eval.f90`
/// (lines 1061..1461). Note that following the upstream subroutine signature,
/// this returns `(alfa, d_alfa_dlog_rho, d_alfa_dlog_t)`.
pub fn get_opal_scvh_alfa_and_partials(
    cfg: &OpalScvhConfig,
    log_t: f64,
    log_rho: f64,
    z: f64,
) -> Result<(f64, f64, f64), EosError> {
    let log_rho1_max = 3.71;
    let log_q1 = 5.69;
    let log_q2 = 5.68;
    let log_q_max = cfg.log_q_max_opal_scvh;
    let log_q3 = cfg.log_q_min_opal_scvh;
    let log_q4 = cfg.log_q_min_opal_scvh;

    let log_rho5 = cfg.log_rho_min_opal_scvh_limit;
    let log_rho6 = log_rho5 - 1.0e-3;
    let log_rho7 = -14.90;
    let log_rho8 = -14.99;

    let log_rho1 = cfg.log_rho1_opal_scvh_limit;
    if log_rho1 > log_rho1_max {
        return Err(EosError::InvalidState {
            context: "get_opal_scvh_alfa_and_partials",
            detail: format!(
                "logRho1_OPAL_SCVH_limit ({log_rho1}) > {log_rho1_max}"
            ),
        });
    }
    let log_rho2 = cfg.log_rho2_opal_scvh_limit;

    let log_t8 = cfg.log_t_low_all_helm;
    let log_t7 = cfg.log_t_low_all_scvh;
    let log_t6 = 4.890;
    let log_t5 = 4.899;
    let log_t2 = cfg.log_t_all_opal;
    let log_t1 = cfg.log_t_all_helm;

    let z_all_helm = cfg.z_all_helm;
    let z_no_helm = cfg.z_all_opal;

    if log_t >= log_t1 {
        return Ok((1.0, 0.0, 0.0));
    }

    let log_t3 = 0.5 * (log_rho1 - log_q1 + 12.0);
    let log_t4 = 0.5 * (log_rho2 - log_q2 + 12.0);
    let log_rho3 = log_q1 + 2.0 * log_t7 - 12.0;
    let log_rho4 = log_q2 + 2.0 * log_t7 - 12.0;

    if log_rho1 <= log_rho2
        || log_rho2 <= log_rho3
        || log_rho3 <= log_rho4
        || log_rho4 <= log_rho5
        || log_rho5 <= log_rho6
        || log_rho6 <= log_rho7
        || log_rho7 <= log_rho8
        || log_t1 <= log_t2
        || log_t2 <= log_t3
        || log_t3 <= log_t4
        || log_t7 <= log_t8
    {
        return Err(EosError::InvalidState {
            context: "get_opal_scvh_alfa_and_partials",
            detail: "non-decreasing logT/logRho region boundaries".to_string(),
        });
    }

    let log_q = log_rho - 2.0 * log_t + 12.0;
    let mut c_dx = 0.0;
    let mut c_dy = 0.0;
    let mut d_dx_dlog_rho = 0.0;
    let mut d_dy_dlog_t = 0.0;

    let iregion = if z >= z_all_helm
        || log_t >= log_t1
        || log_t <= log_t8
        || log_rho >= log_rho1
        || log_q <= log_q4
        || log_q >= log_q_max
        || (log_rho <= log_rho6 && log_t <= log_t6)
    {
        OpalScvhRegion::UseNone
    } else if log_q <= log_q3 && log_t >= log_t5 {
        let d_alfa_dlog_q = 1.0 / (log_q4 - log_q3);
        c_dx = (log_q - log_q3) * d_alfa_dlog_q;
        d_dx_dlog_rho = d_alfa_dlog_q;
        d_dy_dlog_t = -2.0 * d_alfa_dlog_q;
        OpalScvhRegion::BlendDiagonal
    } else if log_t >= log_t2 {
        d_dy_dlog_t = if log_t1 - log_t2 < 0.01 {
            0.0
        } else {
            1.0 / (log_t1 - log_t2)
        };
        c_dy = (log_t - log_t2) * d_dy_dlog_t;
        if log_rho > log_rho2 {
            d_dx_dlog_rho = 1.0 / (log_rho1 - log_rho2);
            c_dx = (log_rho - log_rho2) * d_dx_dlog_rho;
            OpalScvhRegion::BlendCornerOut
        } else {
            OpalScvhRegion::BlendInY
        }
    } else if log_t >= log_t3 {
        if log_rho > log_rho2 {
            d_dx_dlog_rho = 1.0 / (log_rho1 - log_rho2);
            c_dx = (log_rho - log_rho2) * d_dx_dlog_rho;
            OpalScvhRegion::BlendInX
        } else {
            OpalScvhRegion::UseAll
        }
    } else if log_t >= log_t4 {
        let log_rho_hi = log_q1 + 2.0 * log_t - 12.0;
        if log_rho >= log_rho_hi {
            OpalScvhRegion::UseNone
        } else if log_rho > log_rho2 {
            d_dx_dlog_rho = 1.0 / (log_rho_hi - log_rho2);
            c_dx = (log_rho - log_rho2) * d_dx_dlog_rho;
            OpalScvhRegion::BlendInX
        } else {
            OpalScvhRegion::UseAll
        }
    } else if log_rho > log_rho4 {
        if log_t > log_t7 {
            let a =
                ((log_q1 + 2.0 * log_t4 - 12.0) - log_rho3) / (log_t4 - log_t7);
            let log_rho_hi = log_rho3 + (log_t - log_t7) * a;
            let b = (log_rho2 - log_rho4) / (log_t4 - log_t7);
            let log_rho_lo = log_rho4 + (log_t - log_t7) * b;
            if log_rho >= log_rho_hi {
                OpalScvhRegion::UseNone
            } else if log_rho >= log_rho_lo {
                c_dx = (log_rho - log_rho_lo) / (log_rho_hi - log_rho_lo);
                d_dx_dlog_rho =
                    1.0 / (log_rho3 - log_rho4 + (a - b) * (log_t - log_t7));
                OpalScvhRegion::BlendInX
            } else {
                OpalScvhRegion::UseAll
            }
        } else if log_rho > log_rho3 {
            OpalScvhRegion::UseNone
        } else {
            d_dx_dlog_rho = 1.0 / (log_rho3 - log_rho4);
            c_dx = (log_rho - log_rho4) * d_dx_dlog_rho;
            d_dy_dlog_t = 1.0 / (log_t8 - log_t7);
            c_dy = (log_t - log_t7) * d_dy_dlog_t;
            OpalScvhRegion::BlendCornerOut
        }
    } else if log_rho >= log_rho5 || log_t > log_t5 {
        OpalScvhRegion::UseAll
    } else if log_t >= log_t6 {
        if log_rho <= log_rho6 {
            d_dy_dlog_t = 1.0 / (log_t6 - log_t5);
            c_dy = (log_t - log_t5) * d_dy_dlog_t;
            OpalScvhRegion::BlendInY
        } else {
            d_dx_dlog_rho = 1.0 / (log_rho5 - log_rho6);
            c_dx = (log_rho - log_rho6) * d_dx_dlog_rho;
            d_dy_dlog_t = 1.0 / (log_t5 - log_t6);
            c_dy = (log_t - log_t6) * d_dy_dlog_t;
            OpalScvhRegion::BlendCornerIn
        }
    } else {
        d_dx_dlog_rho = 1.0 / (log_rho6 - log_rho5);
        c_dx = (log_rho - log_rho5) * d_dx_dlog_rho;
        OpalScvhRegion::BlendInX
    };

    let mut d_alfa_dlog_t = 0.0;
    let mut d_alfa_dlog_rho = 0.0;
    let mut alfa = match iregion {
        OpalScvhRegion::UseNone => 1.0,
        OpalScvhRegion::UseAll => 0.0,
        OpalScvhRegion::BlendInY => {
            d_alfa_dlog_t = d_dy_dlog_t;
            c_dy
        }
        OpalScvhRegion::BlendInX => {
            d_alfa_dlog_rho = d_dx_dlog_rho;
            c_dx
        }
        OpalScvhRegion::BlendDiagonal => {
            d_alfa_dlog_rho = d_dx_dlog_rho;
            d_alfa_dlog_t = d_dy_dlog_t;
            c_dx
        }
        OpalScvhRegion::BlendCornerOut => {
            let a = (c_dx * c_dx + c_dy * c_dy).sqrt();
            if a >= 1.0 {
                1.0
            } else if a < 1.0e-10 {
                0.0
            } else {
                d_alfa_dlog_t = c_dy * d_dy_dlog_t / a;
                d_alfa_dlog_rho = c_dx * d_dx_dlog_rho / a;
                a
            }
        }
        OpalScvhRegion::BlendCornerIn => {
            let beta = (c_dx * c_dx + c_dy * c_dy).sqrt();
            let a = 1.0 - beta;
            if a >= 1.0 {
                1.0
            } else if a < 1.0e-10 {
                0.0
            } else {
                d_alfa_dlog_t = -c_dy * d_dy_dlog_t / beta;
                d_alfa_dlog_rho = -c_dx * d_dx_dlog_rho / beta;
                a
            }
        }
    };

    if z > z_no_helm && z < z_all_helm && alfa < 1.0 {
        let zfactor = (z - z_no_helm) / (z_all_helm - z_no_helm);
        alfa = zfactor + (1.0 - zfactor) * alfa;
        d_alfa_dlog_rho *= 1.0 - zfactor;
        d_alfa_dlog_t *= 1.0 - zfactor;
    }

    Ok((alfa, d_alfa_dlog_rho, d_alfa_dlog_t))
}
