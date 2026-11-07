// SPDX-License-Identifier: LGPL-3.0-or-later
//! Radiative opacity regime blending (`Get_kap_Results_blend_T`,
//! `Get_kap_highT_Results`, `Get_kap_lowT_Results` in `kap/private/kap_eval.f90`).

use crate::co::get1_kap_co_results;
use crate::fixed::get1_kap_fixed_metal_results;
use crate::low_t::{aesopus_get, kapcn_get};
use crate::types::{KapConfig, KapError, KapLowTOption};
use flexmesa_autodiff::AutoDiffReal2VarOrder1;
use flexmesa_math::{exp10, safe_log10};

const KAP_MIN_LOG_RHO: f64 = -40.0;
const KAP_CN_ZC: f64 = 0.1644;
const KAP_CN_ZN: f64 = 0.0532;

/// Quintic smootherstep $S_2(u) = 6u^5 - 15u^4 + 10u^3$ over `AutoDiffReal2VarOrder1`.
pub(crate) fn quintic_smootherstep(
    u: AutoDiffReal2VarOrder1,
) -> AutoDiffReal2VarOrder1 {
    let u2 = u * u;
    let u3 = u2 * u;
    -(u3 * ((u * (u * (-6.0) + 15.0)) - 10.0))
}

/// Evaluate radiative opacity across the low-$T$ to high-$T$ blend
/// (`Get_kap_Results_blend_T` in `kap/private/kap_eval.f90`).
///
/// Returns `(frac_lowT, frac_highT, frac_Type2, kap, dlnkap_dln_rho, dlnkap_dln_t)`.
///
/// # Errors
/// Returns [`KapError`] if the underlying high-$T$ or low-$T$ evaluation fails.
pub fn get_kap_results_blend_t(
    cfg: &KapConfig,
    x: f64,
    z: f64,
    xc: f64,
    xn: f64,
    xo: f64,
    xne: f64,
    log_rho_in: f64,
    log_t: f64,
) -> Result<(f64, f64, f64, f64, f64, f64), KapError> {
    let (log_rho, clipped_rho) = if log_rho_in < KAP_MIN_LOG_RHO {
        (KAP_MIN_LOG_RHO, true)
    } else {
        (log_rho_in, false)
    };

    let high_t_min = 3.75_f64;
    let low_t_max = match cfg.kap_low_t_option {
        KapLowTOption::KapCn => 4.05,
        _ => 4.50,
    };
    let lower_bdy = cfg.kap_blend_log_t_lower_bdy.max(high_t_min);
    let upper_bdy = cfg.kap_blend_log_t_upper_bdy.min(low_t_max);

    let (
        frac_low_t,
        frac_high_t,
        frac_type2,
        kap,
        mut dlnkap_dln_rho,
        dlnkap_dln_t,
    ) = if log_t <= lower_bdy {
        let (k_lt, dk_lt_dr, dk_lt_dt) =
            get_kap_low_t_results(cfg, x, z, xc, xn, xo, log_rho, log_t)?;
        (1.0, 0.0, 0.0, k_lt, dk_lt_dr, dk_lt_dt)
    } else if log_t >= upper_bdy {
        let (f_t2, k_ht, dk_ht_dr, dk_ht_dt) =
            get_kap_high_t_results(cfg, x, z, xc, xn, xo, xne, log_rho, log_t)?;
        (0.0, 1.0, f_t2, k_ht, dk_ht_dr, dk_ht_dt)
    } else {
        let (f_t2, kap1, dlnkap1_dr, dlnkap1_dt) =
            get_kap_high_t_results(cfg, x, z, xc, xn, xo, xne, log_rho, log_t)?;
        let (kap2, dlnkap2_dr, dlnkap2_dt) =
            get_kap_low_t_results(cfg, x, z, xc, xn, xo, log_rho, log_t)?;

        let log_t_ad = AutoDiffReal2VarOrder1 {
            val: log_t,
            d1: [0.0, 1.0],
        };
        let u = (log_t_ad - lower_bdy) / (upper_bdy - lower_bdy);
        let alfa_ad = quintic_smootherstep(u);
        let alfa = alfa_ad.val;
        let d_alfa_dlog_t = alfa_ad.d1[1];
        let beta = 1.0 - alfa;
        let d_beta_dlog_t = -d_alfa_dlog_t;

        let log_kap1 = safe_log10(kap1);
        let log_kap2 = safe_log10(kap2);
        let log_kap = alfa * log_kap1 + beta * log_kap2;
        let kap_blend = exp10(log_kap);
        let dlnkap_dr = alfa * dlnkap1_dr + beta * dlnkap2_dr;
        let dlnkap_dt = alfa * dlnkap1_dt
            + beta * dlnkap2_dt
            + d_alfa_dlog_t * log_kap1
            + d_beta_dlog_t * log_kap2;
        (beta, alfa, f_t2, kap_blend, dlnkap_dr, dlnkap_dt)
    };

    if clipped_rho {
        dlnkap_dln_rho = 0.0;
    }
    Ok((
        frac_low_t,
        frac_high_t,
        frac_type2,
        kap,
        dlnkap_dln_rho,
        dlnkap_dln_t,
    ))
}

/// Evaluate high-temperature radiative opacity across Type 1 and Type 2 tables
/// (`Get_kap_highT_Results` in `kap/private/kap_eval.f90`).
///
/// Returns `(frac_Type2, kap, dlnkap_dln_rho, dlnkap_dln_t)`.
///
/// # Errors
/// Returns [`KapError`] if table evaluation fails or `zbase < 0` when required.
pub fn get_kap_high_t_results(
    cfg: &KapConfig,
    x: f64,
    z: f64,
    xc: f64,
    _xn: f64,
    xo: f64,
    _xne: f64,
    log_rho: f64,
    log_t: f64,
) -> Result<(f64, f64, f64, f64), KapError> {
    if !cfg.use_type2_opacities {
        let (lk, dlk_dr, dlk_dt) = get1_kap_fixed_metal_results(
            cfg,
            cfg.kap_option.prefix(),
            z,
            x,
            log_rho,
            log_t,
        )?;
        return Ok((0.0, exp10(lk), dlk_dr, dlk_dt));
    }

    let zbase = cfg.zbase;
    if zbase < 0.0 {
        return Err(KapError::InvalidComposition(
            "must supply zbase for Type 2 opacities",
        ));
    }

    let z_type1 = if cfg.use_zbase_for_type1 {
        z.min(zbase)
    } else {
        z
    };

    let (fc, _fn, fo, _fne) = cfg.kap_co_option.ref_zfracs();
    let dxc = (xc - fc * zbase).max(0.0);
    let dxo = (xo - fo * zbase).max(0.0);

    let frac_x = if x >= cfg.kap_type2_full_off_x {
        0.0
    } else if x <= cfg.kap_type2_full_on_x {
        1.0
    } else {
        (cfg.kap_type2_full_off_x - x)
            / (cfg.kap_type2_full_off_x - cfg.kap_type2_full_on_x)
    };

    let dz = z - zbase;
    let alfa = if dz <= cfg.kap_type2_full_off_dz {
        0.0
    } else if dz >= cfg.kap_type2_full_on_dz {
        frac_x
    } else {
        frac_x * (dz - cfg.kap_type2_full_off_dz)
            / (cfg.kap_type2_full_on_dz - cfg.kap_type2_full_off_dz)
    };
    let beta = 1.0 - alfa;
    let frac_type2 = alfa;

    let (kap_alfa, dlnkap_alfa_dr, dlnkap_alfa_dt) = if alfa > 0.0 {
        let (lk, dr, dt) =
            get1_kap_co_results(cfg, zbase, x, dxc, dxo, log_rho, log_t)?;
        if beta == 0.0 {
            return Ok((frac_type2, exp10(lk), dr, dt));
        }
        (exp10(lk), dr, dt)
    } else {
        (0.0, 0.0, 0.0)
    };

    let (kap_beta, dlnkap_beta_dr, dlnkap_beta_dt) = if beta > 0.0 {
        let (lk, dr, dt) = get1_kap_fixed_metal_results(
            cfg,
            cfg.kap_option.prefix(),
            z_type1,
            x,
            log_rho,
            log_t,
        )?;
        if alfa == 0.0 {
            return Ok((frac_type2, exp10(lk), dr, dt));
        }
        (exp10(lk), dr, dt)
    } else {
        (0.0, 0.0, 0.0)
    };

    let kap = alfa * kap_alfa + beta * kap_beta;
    let dlnkap_dln_rho = (alfa * kap_alfa * dlnkap_alfa_dr
        + beta * kap_beta * dlnkap_beta_dr)
        / kap;
    let dlnkap_dln_t = (alfa * kap_alfa * dlnkap_alfa_dt
        + beta * kap_beta * dlnkap_beta_dt)
        / kap;
    Ok((frac_type2, kap, dlnkap_dln_rho, dlnkap_dln_t))
}

/// Evaluate low-temperature radiative opacity (`Get_kap_lowT_Results` in `kap_eval.f90`).
///
/// Returns `(kap, dlnkap_dln_rho, dlnkap_dln_t)`.
///
/// # Errors
/// Returns [`KapError`] if the underlying low-$T$ table evaluation fails or `zbase < 0`
/// for `kapCN` / `AESOPUS`.
pub fn get_kap_low_t_results(
    cfg: &KapConfig,
    x: f64,
    z: f64,
    xc: f64,
    xn: f64,
    xo: f64,
    log_rho: f64,
    log_t: f64,
) -> Result<(f64, f64, f64), KapError> {
    match cfg.kap_low_t_option {
        KapLowTOption::KapCn => {
            if cfg.zbase < 0.0 {
                return Err(KapError::InvalidComposition(
                    "must supply zbase >= 0 for kapCN lowT opacities",
                ));
            }
            let zbase = cfg.zbase;
            let (fc, fn_val) = if zbase <= 0.0 {
                (1.0, 1.0)
            } else {
                (xc / (KAP_CN_ZC * zbase), xn / (KAP_CN_ZN * zbase))
            };
            kapcn_get(zbase, x, fc, fn_val, log_rho, log_t)
        }
        KapLowTOption::Aesopus => {
            if cfg.zbase < 0.0 {
                return Err(KapError::InvalidComposition(
                    "must supply zbase >= 0 for AESOPUS lowT opacities",
                ));
            }
            aesopus_get(cfg.zbase, x, xc, xn, xo, log_rho, log_t)
        }
        _ => {
            let (lk, dlk_dr, dlk_dt) = get1_kap_fixed_metal_results(
                cfg,
                cfg.kap_low_t_option.prefix(),
                z,
                x,
                log_rho,
                log_t,
            )?;
            Ok((exp10(lk), dlk_dr, dlk_dt))
        }
    }
}
