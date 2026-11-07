//! `Skye` fully ionized degenerate/non-degenerate equation of state
//! (`eos/private/skye.f90`, `skye_ideal.f90`, `skye_coulomb*.f90`, and
//! `skye_thermodynamics.f90`).
//!
//! Reference: Jermyn, Schwab, Bauer, Timmes & Potekhin (2021), ApJ, 913, 72.

pub mod coulomb;
pub mod coulomb_liquid;
pub mod coulomb_solid;

pub use coulomb::{
    decide_phase, extrapolate_free_energy, nonideal_corrections,
    ocp_free_energy, SkyeNonidealOutput,
};
pub use coulomb_solid::SkyeSolidMixingRule;

use flexmesa_autodiff::{AutoDiffReal2VarOrder1, AutoDiffReal2VarOrder3};
use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{CRAD, KERG, MP};
use rayon::prelude::*;

use crate::blend::{is_contained, min_distance_to_polygon};
use crate::free_energy::{
    compute_f_ideal_ion, compute_f_rad, compute_xne, pack_for_export,
};
use crate::helm::{
    HelmTable, HELM_LOGDHI, HELM_LOGDLO, HELM_LOGTHI, HELM_LOGTLO,
};
use crate::ion_offset::compute_ion_offset_from_za;
use crate::types::{EosError, EosEvalOutput, I_FRAC_SKYE};

/// Configuration controls for the `Skye` equation of state (`eos.defaults`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkyeConfig {
    /// Minimum species mass fraction included in `Skye` (`1e-4`).
    pub mass_fraction_limit_for_skye: f64,
    /// Offset ideal ion free energy so neutral ground-state energy is zero (`true`).
    pub skye_use_ion_offsets: bool,
    /// Minimum `\Gamma_i` for solid fit before boundary extrapolation (`170.0`).
    pub skye_min_gamma_for_solid: f64,
    /// Maximum `\Gamma_i` for liquid fit before boundary extrapolation (`200.0`).
    pub skye_max_gamma_for_liquid: f64,
    /// Solid linear-mixing-rule prescription (`Pc` or `Ogata`).
    pub skye_solid_mixing_rule: SkyeSolidMixingRule,
    /// Use rectangular `logRho`/`logT` blend instead of 8-vertex polygon (`false`).
    pub use_simple_skye_blends: bool,
    /// Rectangular blend lower density bound for any `Skye` (`2.3`).
    pub log_rho_min_for_any_skye: f64,
    /// Rectangular blend density bound for pure `Skye` (`2.4`).
    pub log_rho_min_for_all_skye: f64,
    /// Rectangular blend lower temperature bound for any `Skye` (`7.2`).
    pub log_t_min_for_any_skye: f64,
    /// Rectangular blend temperature bound for pure `Skye` (`7.4`).
    pub log_t_min_for_all_skye: f64,
}

impl Default for SkyeConfig {
    fn default() -> Self {
        Self {
            mass_fraction_limit_for_skye: 1.0e-4,
            skye_use_ion_offsets: true,
            skye_min_gamma_for_solid: 170.0,
            skye_max_gamma_for_liquid: 200.0,
            skye_solid_mixing_rule: SkyeSolidMixingRule::Pc,
            use_simple_skye_blends: false,
            log_rho_min_for_any_skye: 2.3,
            log_rho_min_for_all_skye: 2.4,
            log_t_min_for_any_skye: 7.2,
            log_t_min_for_all_skye: 7.4,
        }
    }
}

/// Evaluate `Get_Skye_alfa` (`skye.f90` lines 36-117) using the 8-vertex
/// `(logRho, logT)` polygon and `quintic_blend` distance. Returns
/// `(alfa, d_alfa_dlogT, d_alfa_dlogRho)` where `alfa = 0` inside the `Skye`
/// polygon (pure `Skye`) and `alfa = 1` outside (no `Skye`).
#[must_use]
pub fn get_skye_alfa(
    _ht: &HelmTable,
    log_rho: f64,
    log_t: f64,
    abar: f64,
    zbar: f64,
) -> (f64, f64, f64) {
    let skye_blend_width = 0.1_f64;
    let t_ion = 7.3_f64.max((1.0e5 * zbar * zbar).log10()) + skye_blend_width;
    let rho_ion =
        2.3_f64.max((abar * zbar * zbar * zbar).log10()) + skye_blend_width;
    let rad_offset =
        (abar * MP * CRAD / (3.0 * KERG * (zbar + 1.0))).log10() - 6.0;

    let bounds: [[f64; 2]; 8] = [
        [HELM_LOGDLO, 8.3],
        [HELM_LOGDLO, t_ion],
        [rho_ion, t_ion],
        [rho_ion, HELM_LOGTLO],
        [HELM_LOGDHI, HELM_LOGTLO],
        [HELM_LOGDHI, HELM_LOGTHI],
        [3.0 * HELM_LOGTHI + rad_offset, HELM_LOGTHI],
        [3.0 * 8.3 + rad_offset, 8.3],
    ];

    let p = [
        AutoDiffReal2VarOrder1::var1(log_rho),
        AutoDiffReal2VarOrder1::var2(log_t),
    ];
    let contained = is_contained(&bounds, p);
    let mut dist = min_distance_to_polygon(&bounds, p);
    if contained {
        dist = -dist;
    }
    dist = dist / skye_blend_width;
    let blend = dist
        .max(AutoDiffReal2VarOrder1::zero())
        .min(AutoDiffReal2VarOrder1::constant(1.0));
    (blend.val, blend.d1[1], blend.d1[0])
}

/// Evaluate `Get_Skye_alfa_simple` (`skye.f90` lines 120-173) using
/// rectangular `logRho` and `logT` ramps. Returns
/// `(alfa, d_alfa_dlogT, d_alfa_dlogRho)`.
#[must_use]
pub fn get_skye_alfa_simple(
    config: &SkyeConfig,
    log_rho: f64,
    log_t: f64,
) -> (f64, f64, f64) {
    let log_rho_auto = AutoDiffReal2VarOrder1::var1(log_rho);
    let log_t_auto = AutoDiffReal2VarOrder1::var2(log_t);

    let blend_log_t = if log_t_auto.val < config.log_t_min_for_any_skye {
        AutoDiffReal2VarOrder1::zero()
    } else if log_t_auto.val <= config.log_t_min_for_all_skye {
        (log_t_auto - config.log_t_min_for_any_skye)
            / (config.log_t_min_for_all_skye - config.log_t_min_for_any_skye)
    } else {
        AutoDiffReal2VarOrder1::constant(1.0)
    };

    let blend_log_rho = if log_rho_auto.val < config.log_rho_min_for_any_skye {
        AutoDiffReal2VarOrder1::zero()
    } else if log_rho_auto.val <= config.log_rho_min_for_all_skye {
        (log_rho_auto - config.log_rho_min_for_any_skye)
            / (config.log_rho_min_for_all_skye
                - config.log_rho_min_for_any_skye)
    } else {
        AutoDiffReal2VarOrder1::constant(1.0)
    };

    let blend = (-blend_log_rho + 1.0) * (-blend_log_t + 1.0);
    (blend.val, blend.d1[1], blend.d1[0])
}

/// Evaluate `skye_eos` / `get_Skye_for_eosdt` (`skye.f90` lines 176-399)
/// directly from species mass fractions `xa`, integer charges `z`, atomic
/// weights `acmi` (`\mathrm{amu}`), and baryon numbers `a`.
#[allow(clippy::too_many_arguments)]
pub fn eval_skye_eos_from_weights(
    config: &SkyeConfig,
    ht: &HelmTable,
    temp_in: f64,
    den_in: f64,
    abar: f64,
    zbar: f64,
    xa: &[f64],
    z: &[usize],
    acmi_in: &[f64],
    a_in: &[f64],
) -> Result<EosEvalOutput, EosError> {
    if temp_in <= 0.0 || den_in <= 0.0 || abar <= 0.0 {
        return Err(EosError::InvalidState {
            context: "eval_skye_eos_from_weights",
            detail: format!(
                "invalid state (T={temp_in:e}, rho={den_in:e}, abar={abar:e})"
            ),
        });
    }
    let species = xa.len().min(z.len()).min(acmi_in.len()).min(a_in.len());
    let temp = AutoDiffReal2VarOrder3::var1(temp_in);
    let logtemp = temp.log10();
    let den = AutoDiffReal2VarOrder3::var2(den_in);

    let ytot1 = 1.0 / abar;
    let f_rad = compute_f_rad(temp, den);

    let mut azion = Vec::with_capacity(species);
    let mut acmi = Vec::with_capacity(species);
    let mut a_sel = Vec::with_capacity(species);
    let mut select_xa = Vec::with_capacity(species);
    let mut norm = 0.0_f64;

    for j in 0..species {
        if xa[j] > config.mass_fraction_limit_for_skye {
            azion.push(z[j] as f64);
            acmi.push(acmi_in[j]);
            a_sel.push(a_in[j]);
            select_xa.push(xa[j]);
            norm += xa[j];
        }
    }
    if select_xa.is_empty() || norm <= 0.0 {
        return Err(EosError::InvalidState {
            context: "eval_skye_eos_from_weights",
            detail: "no species above mass_fraction_limit_for_skye".to_string(),
        });
    }
    for x in &mut select_xa {
        *x /= norm;
    }

    let mut ya = Vec::with_capacity(select_xa.len());
    let mut ya_norm = 0.0_f64;
    for (&x, &aj) in select_xa.iter().zip(a_sel.iter()) {
        let yj = x / aj;
        ya.push(yj);
        ya_norm += yj;
    }
    for y in &mut ya {
        *y /= ya_norm;
    }

    let mut f_ideal_ion = compute_f_ideal_ion(temp, den, abar, &acmi, &ya);
    if config.skye_use_ion_offsets {
        let offset = compute_ion_offset_from_za(
            &xa[..species],
            &z[..species],
            &a_in[..species],
        );
        f_ideal_ion = f_ideal_ion + offset;
    }

    let (f_ele, etaele, _xnefer_table) = ht.eval_electron_positron_order3(
        temp.val,
        logtemp.val,
        den.val,
        abar,
        zbar,
    );
    let xnefer = compute_xne(den, ytot1, zbar);

    let coul = nonideal_corrections(
        &ya,
        &azion,
        &acmi,
        config.skye_min_gamma_for_solid,
        config.skye_max_gamma_for_liquid,
        config.skye_solid_mixing_rule,
        den,
        temp,
        xnefer,
        abar,
    );

    let mut out = pack_for_export(
        f_ideal_ion,
        coul.df,
        f_rad,
        f_ele,
        temp,
        den,
        xnefer,
        etaele,
        abar,
        zbar,
        coul.phase,
        coul.latent_ddln_t,
        coul.latent_ddln_rho,
        xa.len(),
    )?;
    out.set_single_component_fraction(I_FRAC_SKYE);
    Ok(out)
}

/// Evaluate `get_Skye_for_eosdt` from 1-based MESA `chem_id` identifiers and
/// species mass fractions `xa`.
#[allow(clippy::too_many_arguments)]
pub fn eval_skye_eos(
    config: &SkyeConfig,
    ht: &HelmTable,
    rho: f64,
    t: f64,
    abar: f64,
    zbar: f64,
    chem_id: &[usize],
    xa: &[f64],
) -> Result<EosEvalOutput, EosError> {
    if chem_id.len() != xa.len() {
        return Err(EosError::InvalidState {
            context: "eval_skye_eos",
            detail: "chem_id and xa length mismatch".to_string(),
        });
    }
    let catalog = NuclideCatalog::builtin();
    let (z, _, acmi, a) =
        crate::blend_dt::extract_species_weights(&catalog, chem_id, "eval_skye_eos")?;
    eval_skye_eos_from_weights(
        config, ht, t, rho, abar, zbar, xa, &z, &acmi, &a,
    )
}

/// Parallel multi-zone `Skye` evaluation over `n_zones` stellar zones using `rayon`.
#[allow(clippy::too_many_arguments)]
pub fn eval_skye_eos_zones(
    config: &SkyeConfig,
    ht: &HelmTable,
    rho: &[f64],
    t: &[f64],
    abar: &[f64],
    zbar: &[f64],
    xa_flat: &[f64],
    z: &[usize],
    acmi: &[f64],
    a: &[f64],
) -> Result<Vec<EosEvalOutput>, EosError> {
    let n_zones = rho.len();
    let species = z.len();
    if t.len() != n_zones
        || abar.len() != n_zones
        || zbar.len() != n_zones
        || acmi.len() != species
        || a.len() != species
        || xa_flat.len() != n_zones * species
    {
        return Err(EosError::InvalidState {
            context: "eval_skye_eos_zones",
            detail: "zone slice length mismatch".to_string(),
        });
    }
    (0..n_zones)
        .into_par_iter()
        .map(|i| {
            let xa_i = &xa_flat[i * species..(i + 1) * species];
            eval_skye_eos_from_weights(
                config, ht, t[i], rho[i], abar[i], zbar[i], xa_i, z, acmi, a,
            )
        })
        .collect()
}
