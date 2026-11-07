// SPDX-License-Identifier: LGPL-3.0-or-later
//! Top-level opacity evaluation and regime blending (`kap_eval.f90` and `kap_lib.f90`).
//!
//! Mirrors `Get_kap_Results`, `combine_rad_with_conduction`, `kap_get`, and
//! `kap_get_zones`.

use crate::blend::quintic_smootherstep;
pub use crate::blend::{
    get_kap_high_t_results, get_kap_low_t_results, get_kap_results_blend_t,
};
use crate::compton::compton_opacity;
use crate::conduction::do_electron_conduction;
use crate::types::{
    KapConfig, KapError, KapEvalOutput, I_FRAC_COMPTON, I_FRAC_HIGH_T,
    I_FRAC_LOW_T, I_FRAC_TYPE2, NUM_KAP_FRACS,
};
use flexmesa_autodiff::AutoDiffReal2VarOrder1;
use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_math::{exp10, safe_log10};
use rayon::prelude::*;

/// Evaluate full radiative, Compton, and electron-conduction opacity at explicit
/// composition moments (`Get_kap_Results` in `kap/private/kap_eval.f90`).
///
/// # Errors
/// Returns [`KapError`] if any underlying table evaluation fails.
pub fn get_kap_results(
    cfg: &KapConfig,
    zbar: f64,
    x: f64,
    z: f64,
    xc: f64,
    xn: f64,
    xo: f64,
    xne: f64,
    log_rho: f64,
    log_t: f64,
    lnfree_e: f64,
    d_lnfree_e_dln_rho: f64,
    d_lnfree_e_dln_t: f64,
    eta: f64,
    d_eta_dln_rho: f64,
    d_eta_dln_t: f64,
) -> Result<KapEvalOutput, KapError> {
    if !(log_rho.is_finite()
        && log_t.is_finite()
        && x.is_finite()
        && z.is_finite()
        && zbar.is_finite())
    {
        return Err(KapError::NumericalFailure(
            "non-finite input to get_kap_results",
        ));
    }

    let rho = exp10(log_rho);
    let t = exp10(log_t);

    let log_rho_ad = AutoDiffReal2VarOrder1 {
        val: log_rho,
        d1: [1.0, 0.0],
    };
    let log_t_ad = AutoDiffReal2VarOrder1 {
        val: log_t,
        d1: [0.0, 1.0],
    };
    let log_r_ad = log_rho_ad - log_t_ad * 3.0 + 18.0;

    let log_t_compton_blend_hi = cfg.log_t_compton_blend_hi;
    let log_t_compton_blend_lo = log_t_compton_blend_hi - 0.5;
    let log_r_compton_blend_lo = cfg.log_r_compton_blend_lo;
    let log_r_compton_blend_hi = log_r_compton_blend_lo + 0.5;

    let blend_log_t = if log_t_ad.val <= log_t_compton_blend_lo {
        AutoDiffReal2VarOrder1::from(0.0)
    } else if log_t_ad.val <= log_t_compton_blend_hi {
        let u = (log_t_ad - log_t_compton_blend_lo)
            / (log_t_compton_blend_hi - log_t_compton_blend_lo);
        quintic_smootherstep(u)
    } else {
        AutoDiffReal2VarOrder1::from(1.0)
    };

    let blend_log_r = if log_r_ad.val <= log_r_compton_blend_lo {
        AutoDiffReal2VarOrder1::from(1.0)
    } else if log_r_ad.val <= log_r_compton_blend_hi {
        let u = (log_r_compton_blend_hi - log_r_ad)
            / (log_r_compton_blend_hi - log_r_compton_blend_lo);
        quintic_smootherstep(u)
    } else {
        AutoDiffReal2VarOrder1::from(0.0)
    };

    let one = AutoDiffReal2VarOrder1::from(1.0);
    let blend = one - (one - blend_log_t) * (one - blend_log_r);

    let mut kap_fracs = [0.0_f64; NUM_KAP_FRACS];
    kap_fracs[I_FRAC_COMPTON] = blend.val;

    let logkap_compton = if blend.val > 0.0 {
        let (kc, dlnkc_dr, dlnkc_dt) = compton_opacity(
            rho,
            t,
            lnfree_e,
            d_lnfree_e_dln_rho,
            d_lnfree_e_dln_t,
            eta,
            d_eta_dln_rho,
            d_eta_dln_t,
        );
        AutoDiffReal2VarOrder1 {
            val: safe_log10(kc),
            d1: [dlnkc_dr, dlnkc_dt],
        }
    } else {
        AutoDiffReal2VarOrder1::from(0.0)
    };

    let logkap_rad_tbl = if blend.val < 1.0 {
        let (f_low, f_high, f_type2, kr, dlnkr_dr, dlnkr_dt) =
            get_kap_results_blend_t(
                cfg, x, z, xc, xn, xo, xne, log_rho, log_t,
            )?;
        kap_fracs[I_FRAC_LOW_T] = (1.0 - blend.val) * f_low;
        kap_fracs[I_FRAC_HIGH_T] = (1.0 - blend.val) * f_high;
        kap_fracs[I_FRAC_TYPE2] = f_type2;
        AutoDiffReal2VarOrder1 {
            val: safe_log10(kr),
            d1: [dlnkr_dr, dlnkr_dt],
        }
    } else {
        kap_fracs[I_FRAC_LOW_T] = 0.0;
        kap_fracs[I_FRAC_HIGH_T] = 0.0;
        kap_fracs[I_FRAC_TYPE2] = 0.0;
        AutoDiffReal2VarOrder1::from(0.0)
    };

    let logkap_rad = blend * logkap_compton + (one - blend) * logkap_rad_tbl;
    let kap_rad = exp10(logkap_rad.val);
    let dlnkap_rad_dln_rho = logkap_rad.d1[0];
    let dlnkap_rad_dln_t = logkap_rad.d1[1];

    let (kap, dlnkap_dln_rho, dlnkap_dln_t) = combine_rad_with_conduction(
        cfg,
        log_rho,
        log_t,
        zbar,
        kap_rad,
        dlnkap_rad_dln_rho,
        dlnkap_rad_dln_t,
    )?;

    Ok(KapEvalOutput {
        kap,
        log10_kap: safe_log10(kap),
        dlnkap_dln_rho,
        dlnkap_dln_t,
        kap_fracs,
    })
}

/// Combine radiative opacity $\kappa_{\mathrm{rad}}$ with conductive opacity
/// $\kappa_{\mathrm{ec}}$ in harmonic parallel (`combine_rad_with_conduction`).
///
/// # Errors
/// Returns [`KapError`] if electron conduction evaluation fails.
pub fn combine_rad_with_conduction(
    cfg: &KapConfig,
    log_rho: f64,
    log_t: f64,
    zbar: f64,
    kap_rad: f64,
    dlnkap_rad_dln_rho: f64,
    dlnkap_rad_dln_t: f64,
) -> Result<(f64, f64, f64), KapError> {
    if !cfg.include_electron_conduction {
        return Ok((kap_rad, dlnkap_rad_dln_rho, dlnkap_rad_dln_t));
    }
    let (kap_ec, dlnkap_ec_dln_rho, dlnkap_ec_dln_t) =
        do_electron_conduction(cfg, zbar, log_rho, log_t)?;
    let kap = 1.0 / (1.0 / kap_rad + 1.0 / kap_ec);
    let dlnkap_dln_rho = (kap / kap_rad) * dlnkap_rad_dln_rho
        + (kap / kap_ec) * dlnkap_ec_dln_rho;
    let dlnkap_dln_t =
        (kap / kap_rad) * dlnkap_rad_dln_t + (kap / kap_ec) * dlnkap_ec_dln_t;
    Ok((kap, dlnkap_dln_rho, dlnkap_dln_t))
}

/// Evaluate total opacity from a nuclear species list `(chem_id, xa)` (`kap_get` in `kap_lib.f90`).
///
/// `chem_id` contains 1-based MESA nuclide indices (`ih1 = 2`, `ihe4 = 7`, etc.).
///
/// # Errors
/// Returns [`KapError`] if `chem_id.len() != xa.len()` or table evaluation fails.
pub fn kap_get(
    cfg: &KapConfig,
    chem_id: &[usize],
    xa: &[f64],
    log_rho: f64,
    log_t: f64,
    lnfree_e: f64,
    d_lnfree_e_dln_rho: f64,
    d_lnfree_e_dln_t: f64,
    eta: f64,
    d_eta_dln_rho: f64,
    d_eta_dln_t: f64,
) -> Result<KapEvalOutput, KapError> {
    if chem_id.len() != xa.len() {
        return Err(KapError::InvalidComposition("chem_id.len() != xa.len()"));
    }
    let catalog = NuclideCatalog::builtin_ref();
    let comp =
        get_composition_info(catalog, chem_id, xa, true).map_err(|_| {
            KapError::InvalidComposition("invalid chem_id in kap_get")
        })?;
    let mut xc = 0.0_f64;
    let mut xn = 0.0_f64;
    let mut xo = 0.0_f64;
    let mut xne = 0.0_f64;
    for (&cid, &x_i) in chem_id.iter().zip(xa.iter()) {
        if let Some(iso) = catalog.get_by_id(cid) {
            match iso.z {
                6 => xc += x_i,
                7 => xn += x_i,
                8 => xo += x_i,
                10 => xne += x_i,
                _ => {}
            }
        }
    }
    get_kap_results(
        cfg,
        comp.zbar,
        comp.xh,
        comp.xz,
        xc,
        xn,
        xo,
        xne,
        log_rho,
        log_t,
        lnfree_e,
        d_lnfree_e_dln_rho,
        d_lnfree_e_dln_t,
        eta,
        d_eta_dln_rho,
        d_eta_dln_t,
    )
}

/// Evaluate `get_kap_results` across `N` stellar zones in parallel using `rayon`.
///
/// # Errors
/// Returns the first [`KapError`] encountered across the zone batch.
pub fn kap_get_zones(
    cfg: &KapConfig,
    zbar: &[f64],
    x: &[f64],
    z: &[f64],
    xc: &[f64],
    xn: &[f64],
    xo: &[f64],
    xne: &[f64],
    log_rho: &[f64],
    log_t: &[f64],
    lnfree_e: &[f64],
    d_lnfree_e_dln_rho: &[f64],
    d_lnfree_e_dln_t: &[f64],
    eta: &[f64],
    d_eta_dln_rho: &[f64],
    d_eta_dln_t: &[f64],
) -> Result<Vec<KapEvalOutput>, KapError> {
    let n = log_rho.len();
    if zbar.len() != n
        || x.len() != n
        || z.len() != n
        || xc.len() != n
        || xn.len() != n
        || xo.len() != n
        || xne.len() != n
        || log_t.len() != n
        || lnfree_e.len() != n
        || d_lnfree_e_dln_rho.len() != n
        || d_lnfree_e_dln_t.len() != n
        || eta.len() != n
        || d_eta_dln_rho.len() != n
        || d_eta_dln_t.len() != n
    {
        return Err(KapError::InvalidComposition(
            "mismatched zone slice lengths in kap_get_zones",
        ));
    }
    (0..n)
        .into_par_iter()
        .map(|k| {
            get_kap_results(
                cfg,
                zbar[k],
                x[k],
                z[k],
                xc[k],
                xn[k],
                xo[k],
                xne[k],
                log_rho[k],
                log_t[k],
                lnfree_e[k],
                d_lnfree_e_dln_rho[k],
                d_lnfree_e_dln_t[k],
                eta[k],
                d_eta_dln_rho[k],
                d_eta_dln_t[k],
            )
        })
        .collect()
}
