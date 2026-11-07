//! Potekhin-Chabrier (`PC`) fully ionized electron-ion plasma equation of state
//! (`eos/private/pc_eos.f90` and `eos/private/eospc_eval.f90`).

pub mod blin;
pub mod blin_c;
pub mod coulomb_ion;
pub mod coulomb_screen;
pub mod elect;
pub mod eosfi;
pub mod excor;
pub mod ferinv;
pub mod melange;

pub use melange::{melange9, Melange9Output};

use flexmesa_autodiff::AutoDiffReal2VarOrder1;
use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{AMU, AVO, FOUR_THIRDS_PI, KERG, ONE_THIRD, QE};
use rayon::prelude::*;

use crate::types::{
    EosError, EosEvalOutput, I_CHI_RHO, I_CHI_T, I_CP, I_CV, I_DE_DRHO,
    I_DS_DRHO, I_DS_DT, I_ETA, I_FRAC_PC, I_GAMMA1, I_GAMMA3, I_GRAD_AD,
    I_LATENT_DDLNRHO, I_LATENT_DDLNT, I_LN_E, I_LN_FREE_E, I_LN_PGAS, I_LN_S,
    I_MU, I_PHASE,
};

/// Configuration controls for the `PC` equation of state (`eos.defaults`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcConfig {
    /// Minimum species mass fraction included in `PC` (`1e-4`).
    pub mass_fraction_limit_for_pc: f64,
    /// Upper density threshold (`log10(rho)`) for pure `PC` (`2.999`).
    pub log_rho1_pc_limit: f64,
    /// Lower density threshold (`log10(rho)`) below which `PC` is excluded (`2.8`).
    pub log_rho2_pc_limit: f64,
    /// Lower temperature threshold (`log10(T)`) for pure `PC` when `T` limits are used (`7.6`).
    pub log_t1_pc_limit: f64,
    /// Upper temperature threshold (`log10(T)`) above which `PC` is excluded (`7.7`).
    pub log_t2_pc_limit: f64,
    /// Use electron coupling `\log_{10} \Gamma_e` limits instead of `\log_{10} T` (`true`).
    pub pc_use_gamma_limit_instead_of_t: bool,
    /// `\log_{10} \Gamma_e` threshold below which pure `HELM` is used (`-1.8`).
    pub log_gamma_e_all_helm: f64,
    /// `\log_{10} \Gamma_e` threshold above which pure `PC` is used (`-1.7`).
    pub log_gamma_e_all_pc: f64,
    /// Mean ion `\Gamma_i` where liquid-to-solid crystallization blend begins (`170.0`).
    pub pc_gamma_start_crystal: f64,
    /// Mean ion `\Gamma_i` where plasma is fully crystallized (`180.0`).
    pub pc_gamma_full_crystal: f64,
    /// Include Planckian photon radiation pressure and energy (`true`).
    pub include_radiation: bool,
}

impl Default for PcConfig {
    fn default() -> Self {
        Self {
            mass_fraction_limit_for_pc: 1.0e-4,
            log_rho1_pc_limit: 2.999,
            log_rho2_pc_limit: 2.8,
            log_t1_pc_limit: 7.6,
            log_t2_pc_limit: 7.7,
            pc_use_gamma_limit_instead_of_t: true,
            log_gamma_e_all_helm: -1.8,
            log_gamma_e_all_pc: -1.7,
            pc_gamma_start_crystal: 170.0,
            pc_gamma_full_crystal: 180.0,
            include_radiation: true,
        }
    }
}

/// Evaluate `Get_PC_alfa` (`eospc_eval.f90` lines 31-96), returning
/// `(alfa, d_alfa_dlogT, d_alfa_dlogRho)` where `alfa = 0` is pure `PC` and
/// `alfa = 1` is zero `PC` (pure fallback).
#[must_use]
pub fn get_pc_alfa(
    config: &PcConfig,
    log_rho: f64,
    log_t: f64,
    abar: f64,
    zbar: f64,
) -> (f64, f64, f64) {
    let mut log_rho_lo = config.log_rho2_pc_limit;
    let log_rho_hi = config.log_rho1_pc_limit;

    if config.pc_use_gamma_limit_instead_of_t {
        let log_ge0 =
            (QE * QE * (FOUR_THIRDS_PI * AVO * zbar / abar).powf(ONE_THIRD)
                / KERG)
                .log10();
        let log_ge = log_ge0 + log_rho / 3.0 - log_t;
        let log_ge_lo = config.log_gamma_e_all_helm;
        let log_ge_hi = config.log_gamma_e_all_pc;
        let log_t_lo = log_ge0 + log_rho_lo / 3.0 - log_ge_lo;
        let log_t_hi = log_ge0 + log_rho_hi / 3.0 - log_ge_hi;

        if log_rho <= log_rho_lo || log_ge <= log_ge_lo {
            (1.0, 0.0, 0.0)
        } else if log_rho >= log_rho_hi && log_ge >= log_ge_hi {
            (0.0, 0.0, 0.0)
        } else if log_t >= log_t_hi {
            let span = log_ge_hi - log_ge_lo;
            let alfa = (log_ge_hi - log_ge) / span;
            let d_alfa_dlog_t = 1.0 / span;
            let d_alfa_dlog_rho = -(1.0 / 3.0) / span;
            (alfa, d_alfa_dlog_t, d_alfa_dlog_rho)
        } else {
            if log_t >= log_t_lo {
                log_rho_lo = (log_t_lo + log_ge_lo - log_ge0) * 3.0;
            }
            let span = log_rho_hi - log_rho_lo;
            let alfa = (log_rho_hi - log_rho) / span;
            let d_alfa_dlog_rho = -1.0 / span;
            (alfa, 0.0, d_alfa_dlog_rho)
        }
    } else {
        let log_t_lo = config.log_t1_pc_limit;
        let log_t_hi = config.log_t2_pc_limit;
        if log_rho <= log_rho_lo || log_t >= log_t_hi {
            (1.0, 0.0, 0.0)
        } else if log_rho >= log_rho_hi && log_t <= log_t_lo {
            (0.0, 0.0, 0.0)
        } else if log_t >= log_t_lo {
            let span = log_t_hi - log_t_lo;
            let alfa = (log_t - log_t_lo) / span;
            (alfa, 1.0 / span, 0.0)
        } else {
            let span = log_rho_hi - log_rho_lo;
            let alfa = (log_rho_hi - log_rho) / span;
            (alfa, 0.0, -1.0 / span)
        }
    }
}

#[inline(always)]
fn safe_log_order1(x: AutoDiffReal2VarOrder1) -> AutoDiffReal2VarOrder1 {
    if x.val < 1.0e-99 {
        AutoDiffReal2VarOrder1::constant(1.0e-99_f64.ln())
    } else {
        x.ln()
    }
}

/// Evaluate `Get_PC_Results` / `get_PC_for_eosdt` (`eospc_eval.f90` lines
/// 99-363) directly from species mass fractions `xa`, atomic charges `azion`,
/// and atomic weights `acmi` (`\mathrm{amu}`).
pub fn eval_pc_eos_from_weights(
    config: &PcConfig,
    rho_real: f64,
    t_real: f64,
    abar: f64,
    zbar: f64,
    xa: &[f64],
    azion: &[f64],
    acmi: &[f64],
) -> Result<EosEvalOutput, EosError> {
    const UN_T6: f64 = 0.3157746;
    if rho_real <= 0.0 || t_real <= 0.0 || abar <= 0.0 {
        return Err(EosError::InvalidState {
            context: "eval_pc_eos_from_weights",
            detail: format!(
                "invalid state (rho={rho_real:e}, T={t_real:e}, abar={abar:e})"
            ),
        });
    }
    let species = xa.len().min(azion.len()).min(acmi.len());
    let mut ay = vec![0.0_f64; species];
    for j in 0..species {
        if xa[j] >= config.mass_fraction_limit_for_pc {
            ay[j] = xa[j] / acmi[j];
        }
    }

    let t = AutoDiffReal2VarOrder1::var1(t_real);
    let rho = AutoDiffReal2VarOrder1::var2(rho_real);
    let temp = t * (1.0e-6 / UN_T6);

    let start_crystal = config.pc_gamma_start_crystal;
    let full_crystal = config.pc_gamma_full_crystal;

    let m = melange9(&ay, azion, acmi, rho, temp, start_crystal, full_crystal)?;

    let phase = if m.gami_mean.val < start_crystal {
        AutoDiffReal2VarOrder1::zero()
    } else if m.gami_mean.val > full_crystal {
        AutoDiffReal2VarOrder1::constant(1.0)
    } else {
        (m.gami_mean - start_crystal) / (full_crystal - start_crystal)
    };

    let tnk = rho * t * (8.31447e7 / m.cmi_mean);
    let pgas = m.p_nkt * tnk;
    let mut p_nkt = m.p_nkt;
    let mut u_nkt = m.u_nkt;
    let mut s_nk = m.s_nk;
    let mut cv_nkt = m.cv;
    let mut chir = m.chir;
    let mut chit = m.chit;

    if config.include_radiation {
        let p_tot_nkt = p_nkt + m.prad_nkt;
        chit = chit * (p_nkt / p_tot_nkt) + m.prad_nkt * 4.0 / p_tot_nkt;
        chir = chir * (p_nkt / p_tot_nkt);
        p_nkt = p_tot_nkt;
        u_nkt = u_nkt + m.prad_nkt * 3.0;
        s_nk = s_nk + m.prad_nkt * 4.0;
        cv_nkt = cv_nkt + m.prad_nkt * 12.0;
    }

    let p = p_nkt * tnk;
    let n = 1.0 / (abar * AMU);
    let cv = cv_nkt * (n * KERG);
    let gamma3 = p / rho * chit / (t * cv) + 1.0;
    let gamma1 = chit * (gamma3 - 1.0) + chir;
    let grad_ad = (gamma3 - 1.0) / gamma1;
    let cp = cv * gamma1 / chir;
    let de_drho = (-chit + 1.0) * p / (rho * rho);
    let ds_dt = cv / t;
    let ds_drho = -p * chit / (rho * rho * t);
    let mu = AutoDiffReal2VarOrder1::constant(abar / (1.0 + zbar));
    let lnfree_e = AutoDiffReal2VarOrder1::constant((zbar / abar).ln());
    let ln_pgas = pgas.ln();
    let ln_e = safe_log_order1(u_nkt * t * (n * KERG));
    let ln_s = safe_log_order1(s_nk * (n * KERG));

    let mut out = EosEvalOutput::zeros(xa.len());
    let channels: [(usize, AutoDiffReal2VarOrder1); 17] = [
        (I_LN_PGAS, ln_pgas),
        (I_LN_E, ln_e),
        (I_LN_S, ln_s),
        (I_GRAD_AD, grad_ad),
        (I_CHI_RHO, chir),
        (I_CHI_T, chit),
        (I_CP, cp),
        (I_CV, cv),
        (I_DE_DRHO, de_drho),
        (I_DS_DT, ds_dt),
        (I_DS_DRHO, ds_drho),
        (I_MU, mu),
        (I_LN_FREE_E, lnfree_e),
        (I_GAMMA1, gamma1),
        (I_GAMMA3, gamma3),
        (I_ETA, m.chi),
        (I_PHASE, phase),
    ];

    for (idx, ad) in channels {
        out.res[idx] = ad.val;
        out.d_dln_t[idx] = ad.d1[0] * t.val;
        out.d_dlnd[idx] = ad.d1[1] * rho.val;
    }

    for idx in [I_LATENT_DDLNT, I_LATENT_DDLNRHO] {
        out.res[idx] = 0.0;
        out.d_dln_t[idx] = 0.0;
        out.d_dlnd[idx] = 0.0;
    }
    out.set_single_component_fraction(I_FRAC_PC);
    Ok(out)
}

/// Evaluate `get_PC_for_eosdt` from 1-based MESA `chem_id` identifiers and
/// species mass fractions `xa`.
pub fn eval_pc_eos(
    config: &PcConfig,
    rho: f64,
    t: f64,
    abar: f64,
    zbar: f64,
    chem_id: &[usize],
    xa: &[f64],
) -> Result<EosEvalOutput, EosError> {
    if chem_id.len() != xa.len() {
        return Err(EosError::InvalidState {
            context: "eval_pc_eos",
            detail: "chem_id and xa length mismatch".to_string(),
        });
    }
    let catalog = NuclideCatalog::builtin();
    let (_, azion, acmi, _) =
        crate::blend_dt::extract_species_weights(&catalog, chem_id, "eval_pc_eos")?;
    eval_pc_eos_from_weights(config, rho, t, abar, zbar, xa, &azion, &acmi)
}

/// Parallel multi-zone `PC` evaluation over `n_zones` stellar zones using `rayon`.
pub fn eval_pc_eos_zones(
    config: &PcConfig,
    rho: &[f64],
    t: &[f64],
    abar: &[f64],
    zbar: &[f64],
    xa_flat: &[f64],
    azion: &[f64],
    acmi: &[f64],
) -> Result<Vec<EosEvalOutput>, EosError> {
    let n_zones = rho.len();
    let species = azion.len();
    if t.len() != n_zones
        || abar.len() != n_zones
        || zbar.len() != n_zones
        || acmi.len() != species
        || xa_flat.len() != n_zones * species
    {
        return Err(EosError::InvalidState {
            context: "eval_pc_eos_zones",
            detail: "zone slice length mismatch".to_string(),
        });
    }
    (0..n_zones)
        .into_par_iter()
        .map(|i| {
            let xa_i = &xa_flat[i * species..(i + 1) * species];
            eval_pc_eos_from_weights(
                config, rho[i], t[i], abar[i], zbar[i], xa_i, azion, acmi,
            )
        })
        .collect()
}
