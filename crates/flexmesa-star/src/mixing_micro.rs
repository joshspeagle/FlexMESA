// SPDX-License-Identifier: LGPL-3.0-or-later
//! Single-cell and slice-scoped microphysics helpers (`do_eos_for_cell`,
//! `solve_eos_given_PgasT`, `do_kap_for_cell`) used by verbatim MESA `26.4.1`
//! predictive mixing (`predictive_mix.f90`) and convective premixing
//! (`conv_premix.f90`).

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_const::{AVO, CRAD, FOUR_PI, KERG, LN10, QE};
use flexmesa_eos::{
    eos_dt_get, eos_dt_get_rho, EosEvalOutput, I_CHI_RHO, I_CHI_T, I_CP, I_CV,
    I_ETA, I_FRAC_PC, I_GAMMA1, I_GAMMA3, I_GRAD_AD, I_LATENT_DDLNRHO,
    I_LATENT_DDLNT, I_LN_E, I_LN_FREE_E, I_LN_PGAS, I_LN_S, I_MU,
    NUM_EOS_BASIC_RESULTS,
};
use flexmesa_kap::get_kap_results;
use rayon::prelude::*;

use crate::controls::StarError;
use crate::micro::builtin_helm_table;
use crate::model::StarModel;
use crate::star_ad::{StarAdExt, StarAdOrder1};

/// Maximum root-finder iterations for inverse EOS solves
/// (`max_iter_for_solve` in `star/private/eos_support.f90:38`).
const MAX_ITER_FOR_SOLVE: usize = 100;

/// Tolerance on $\log_{10}\rho$ for `solve_eos_given_PgasT`
/// (`star/private/micro.f90:330`).
const LOG_RHO_TOL_PGAS_T: f64 = 1.0e-8;

#[inline]
fn clamp_unit_roundoff(val: f64) -> f64 {
    if (-1.0e-3..=1.0 + 1.0e-3).contains(&val) {
        val.clamp(0.0, 1.0)
    } else {
        val
    }
}

/// Writes an [`EosEvalOutput`] into all cell-centered EOS fields of `model` at
/// cell `k` identically to `crate::micro::do_eos`.
fn write_cell_eos_output(
    model: &mut StarModel,
    k: usize,
    out: &EosEvalOutput,
) {
    let nz = model.nz;
    let species = model.species;
    let n_sp = nz * species;
    if model.aux.eos_frac_pc.len() != nz {
        model.aux.eos_frac_pc.resize(nz, 0.0);
    }
    if model.aux.latent_ddlnt.len() != nz {
        model.aux.latent_ddlnt.resize(nz, 0.0);
    }
    if model.aux.latent_ddlnrho.len() != nz {
        model.aux.latent_ddlnrho.resize(nz, 0.0);
    }
    if model.aux.dln_e_dxa_for_partials.len() != n_sp {
        model.aux.dln_e_dxa_for_partials.resize(n_sp, 0.0);
    }
    if model.aux.dln_peos_dxa_for_partials.len() != n_sp {
        model.aux.dln_peos_dxa_for_partials.resize(n_sp, 0.0);
    }

    let t_k = model.t[k];
    let rho_k = model.rho[k];
    let ln_pgas = out.res[I_LN_PGAS];
    let pgas = flexmesa_math::exp(ln_pgas);
    let prad = CRAD * flexmesa_math::pow4(t_k) / 3.0;
    let peos = pgas + prad;
    let ln_peos = flexmesa_math::log(peos);

    let d_lnpgas_dlnd = out.d_dlnd[I_LN_PGAS];
    let d_lnpgas_dlnt = out.d_dln_t[I_LN_PGAS];
    let alfa_rad = prad / peos;
    let beta_gas = pgas / peos;
    let d_lnpeos_dlnd = beta_gas * d_lnpgas_dlnd;
    let d_lnpeos_dlnt = beta_gas * d_lnpgas_dlnt + 4.0 * alfa_rad;

    let ln_e = out.res[I_LN_E];
    let energy = flexmesa_math::exp(ln_e);
    let ln_s = out.res[I_LN_S];
    let entropy = flexmesa_math::exp(ln_s);
    let gamma1 = out.res[I_GAMMA1];
    let csound = (gamma1 * peos / rho_k).max(0.0).sqrt();

    let a_e = flexmesa_math::pow(
        3.0 / (FOUR_PI * AVO * rho_k * model.zbar[k].max(1.0e-10)
            / model.abar[k]),
        1.0 / 3.0,
    );
    let gam = model.z53bar[k] * QE * QE / (a_e * KERG * t_k);

    model.pgas[k] = pgas;
    model.ln_pgas[k] = ln_pgas;
    model.prad[k] = prad;
    model.peos[k] = peos;
    model.ln_peos[k] = ln_peos;
    model.energy[k] = energy;
    model.ln_e[k] = ln_e;
    model.entropy[k] = entropy;
    model.ln_s[k] = ln_s;
    model.grada[k] = out.res[I_GRAD_AD];
    model.cv[k] = out.res[I_CV];
    model.cp[k] = out.res[I_CP];
    model.chi_rho[k] = out.res[I_CHI_RHO];
    model.chi_t[k] = out.res[I_CHI_T];
    model.gamma1[k] = gamma1;
    model.gamma3[k] = out.res[I_GAMMA3];
    model.eta[k] = out.res[I_ETA];
    model.gam[k] = gam;
    model.mu[k] = out.res[I_MU];
    model.lnfree_e[k] = out.res[I_LN_FREE_E];
    model.csound[k] = csound;

    model.aux.eos_frac_pc[k] = out.res[I_FRAC_PC];
    model.aux.latent_ddlnt[k] = out.res[I_LATENT_DDLNT];
    model.aux.latent_ddlnrho[k] = out.res[I_LATENT_DDLNRHO];
    let sp_base = k * species;
    for j in 0..species {
        let [d_lnpgas_dxa, d_lne_dxa] =
            out.d_dxa.get(j).copied().unwrap_or([0.0, 0.0]);
        model.aux.dln_e_dxa_for_partials[sp_base + j] = d_lne_dxa;
        model.aux.dln_peos_dxa_for_partials[sp_base + j] =
            beta_gas * d_lnpgas_dxa;
    }

    model.ln_peos_ad[k] =
        StarAdOrder1::from_dlnd_dlnt_00(ln_peos, d_lnpeos_dlnd, d_lnpeos_dlnt);
    model.ln_e_ad[k] = StarAdOrder1::from_dlnd_dlnt_00(
        ln_e,
        out.d_dlnd[I_LN_E],
        out.d_dln_t[I_LN_E],
    );
    model.ln_s_ad[k] = StarAdOrder1::from_dlnd_dlnt_00(
        ln_s,
        out.d_dlnd[I_LN_S],
        out.d_dln_t[I_LN_S],
    );

    model.eos_res[k] = out.res;
    model.eos_d_dlnd[k] = out.d_dlnd;
    model.eos_d_dlnt[k] = out.d_dln_t;
}

/// Computes `(xc, xn, xo, xne)` C/N/O/Ne mass fractions from `xa_k`.
fn eval_cno_ne_mass_fractions(
    catalog: &NuclideCatalog,
    chem_id: &[usize],
    xa_k: &[f64],
) -> (f64, f64, f64, f64) {
    let mut xc = 0.0;
    let mut xn = 0.0;
    let mut xo = 0.0;
    let mut xne = 0.0;
    for (&cid, &xj) in chem_id.iter().zip(xa_k.iter()) {
        let z = catalog.get_by_id(cid).map_or(0, |n| n.z);
        match z {
            6 => xc += xj,
            7 => xn += xj,
            8 => xo += xj,
            10 => xne += xj,
            _ => {}
        }
    }
    (xc, xn, xo, xne)
}

/// Composition moments for a single cell.
#[derive(Debug, Clone, Copy)]
struct CellCompMoments {
    abar: f64,
    zbar: f64,
    z2bar: f64,
    z53bar: f64,
    ye: f64,
    xh_mass: f64,
    yhe_mass: f64,
    z_mass: f64,
}

/// Per-cell EOS + opacity outputs returned from parallel workers.
struct CellEosKapUpdate {
    comp: CellCompMoments,
    lnd: f64,
    rho: f64,
    t: f64,
    eos: EosEvalOutput,
    op: f64,
    dop_d: f64,
    dop_t: f64,
}

#[allow(clippy::too_many_arguments)]
fn eval_one_cell_eos_kap(
    ht: &flexmesa_eos::HelmTable,
    chem_id: &[usize],
    eos_cfg: &flexmesa_eos::EosConfig,
    kap_cfg: &flexmesa_kap::KapConfig,
    kap_factor: f64,
    k: usize,
    xa_k: &[f64],
    comp: CellCompMoments,
    cno_ne: (f64, f64, f64, f64),
    ln_t_k: f64,
    lnd_k: f64,
    ln_pgas_k: f64,
    fix_pgas: bool,
) -> Result<CellEosKapUpdate, StarError> {
    let (lnd_new, rho_new, t_new, eos_out) = if fix_pgas {
        let inv = eos_dt_get_rho(
            ht,
            chem_id,
            xa_k,
            ln_t_k / LN10,
            I_LN_PGAS as isize,
            ln_pgas_k,
            lnd_k / LN10,
            LOG_RHO_TOL_PGAS_T,
            LOG_RHO_TOL_PGAS_T * LN10,
            MAX_ITER_FOR_SOLVE,
            eos_cfg,
        )
        .map_err(|e| {
            StarError::MicrophysicsError(format!(
                "solve_eos_given_PgasT failed at zone {k}: {e}"
            ))
        })?;
        let lnd_out = inv.log10_result * LN10;
        (
            lnd_out,
            flexmesa_math::exp(lnd_out),
            flexmesa_math::exp(ln_t_k),
            inv.eos,
        )
    } else {
        let r_new = flexmesa_math::exp(lnd_k);
        let t_out = flexmesa_math::exp(ln_t_k);
        let out = eos_dt_get(ht, r_new, t_out, chem_id, xa_k, eos_cfg)
            .map_err(|e| {
                StarError::MicrophysicsError(format!(
                    "do_eos_for_cell failed at zone {k}: {e}"
                ))
            })?;
        (lnd_k, r_new, t_out, out)
    };

    let (xc, xn, xo, xne) = cno_ne;
    let kap_out = get_kap_results(
        kap_cfg,
        comp.zbar,
        comp.xh_mass,
        comp.z_mass,
        xc,
        xn,
        xo,
        xne,
        flexmesa_math::log10(rho_new),
        flexmesa_math::log10(t_new),
        eos_out.res[I_LN_FREE_E],
        eos_out.d_dlnd[I_LN_FREE_E],
        eos_out.d_dln_t[I_LN_FREE_E],
        eos_out.res[I_ETA],
        eos_out.d_dlnd[I_ETA],
        eos_out.d_dln_t[I_ETA],
    )
    .map_err(|e| {
        StarError::MicrophysicsError(format!(
            "do_kap_for_cell failed at zone {k}: {e}"
        ))
    })?;

    Ok(CellEosKapUpdate {
        comp,
        lnd: lnd_new,
        rho: rho_new,
        t: t_new,
        eos: eos_out,
        op: kap_out.kap * kap_factor,
        dop_d: kap_out.kap * kap_out.dlnkap_dln_rho * kap_factor,
        dop_t: kap_out.kap * kap_out.dlnkap_dln_t * kap_factor,
    })
}

fn write_cell_eos_kap_update(
    model: &mut StarModel,
    k: usize,
    upd: &CellEosKapUpdate,
) {
    model.abar[k] = upd.comp.abar;
    model.zbar[k] = upd.comp.zbar;
    model.z2bar[k] = upd.comp.z2bar;
    model.z53bar[k] = upd.comp.z53bar;
    model.ye[k] = upd.comp.ye;
    model.xh_mass[k] = upd.comp.xh_mass;
    model.yhe_mass[k] = upd.comp.yhe_mass;
    model.z_mass[k] = upd.comp.z_mass;
    model.lnd[k] = upd.lnd;
    model.rho[k] = upd.rho;
    model.t[k] = upd.t;
    write_cell_eos_output(model, k, &upd.eos);
    model.opacity[k] = upd.op;
    model.d_opacity_dlnd[k] = upd.dop_d;
    model.d_opacity_dlnt[k] = upd.dop_t;
}

/// Evaluates uniform-composition EOS (`solve_eos_given_PgasT` when `fix_pgas`
/// is true, else `do_eos_for_cell`) and opacity (`do_kap_for_cell`) in
/// parallel across `k_lo..=k_hi` using `rayon`, then writes all cell arrays in
/// deterministic index order (`eval_mixing_coeffs` in
/// `star/private/predictive_mix.f90:720-774`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if composition, EOS, or opacity
/// evaluation fails for any cell in `k_lo..=k_hi`.
pub(crate) fn eval_uniform_xa_eos_kap_range(
    model: &mut StarModel,
    k_lo: usize,
    k_hi: usize,
    xa_mx: &[f64],
    fix_pgas: bool,
) -> Result<(), StarError> {
    if k_lo > k_hi {
        return Ok(());
    }
    let catalog = NuclideCatalog::builtin_ref();
    let comp_raw = get_composition_info(catalog, &model.chem_id, xa_mx, false)
        .map_err(|e| {
            StarError::MicrophysicsError(format!(
                "composition error in predictive_mix: {e}"
            ))
        })?;
    let comp = CellCompMoments {
        abar: comp_raw.abar,
        zbar: comp_raw.zbar,
        z2bar: comp_raw.z2bar,
        z53bar: comp_raw.z53bar,
        ye: comp_raw.ye,
        xh_mass: clamp_unit_roundoff(comp_raw.xh),
        yhe_mass: clamp_unit_roundoff(comp_raw.xhe),
        z_mass: clamp_unit_roundoff(comp_raw.xz),
    };
    let cno_ne = eval_cno_ne_mass_fractions(catalog, &model.chem_id, xa_mx);
    let ht = builtin_helm_table()?;
    let species = model.species;
    let kap_factor = model.controls.opacity_factor;

    let chem_id = &model.chem_id;
    let eos_cfg = &model.eos_config;
    let kap_cfg = &model.kap_config;
    let ln_t = &model.ln_t;
    let lnd = &model.lnd;
    let ln_pgas = &model.ln_pgas;

    let results: Vec<CellEosKapUpdate> = (k_lo..=k_hi)
        .into_par_iter()
        .map(|k| {
            eval_one_cell_eos_kap(
                ht,
                chem_id,
                eos_cfg,
                kap_cfg,
                kap_factor,
                k,
                xa_mx,
                comp,
                cno_ne,
                ln_t[k],
                lnd[k],
                ln_pgas[k],
                fix_pgas,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    for (offset, upd) in results.iter().enumerate() {
        let k = k_lo + offset;
        model.xa[k * species..(k + 1) * species].copy_from_slice(xa_mx);
        write_cell_eos_kap_update(model, k, upd);
    }
    Ok(())
}

/// Evaluates per-cell EOS (`fix_pgas_by_cell[k]` selects fixed `(Pgas, T)` vs
/// fixed `(rho, T)`) and opacity across `k_lo..=k_hi` in parallel via `rayon`,
/// writing outputs back in deterministic index order (`update_model` in
/// `star/private/conv_premix.f90:1311-1335`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if composition, EOS, or opacity
/// evaluation fails.
pub(crate) fn eval_eos_kap_range_by_mode(
    model: &mut StarModel,
    k_lo: usize,
    k_hi: usize,
    fix_pgas_by_cell: &[bool],
) -> Result<(), StarError> {
    if k_lo > k_hi {
        return Ok(());
    }
    let catalog = NuclideCatalog::builtin_ref();
    let ht = builtin_helm_table()?;
    let species = model.species;
    let kap_factor = model.controls.opacity_factor;

    let chem_id = &model.chem_id;
    let eos_cfg = &model.eos_config;
    let kap_cfg = &model.kap_config;
    let xa = &model.xa;
    let ln_t = &model.ln_t;
    let lnd = &model.lnd;
    let ln_pgas = &model.ln_pgas;

    let results: Vec<CellEosKapUpdate> = (k_lo..=k_hi)
        .into_par_iter()
        .map(|k| -> Result<CellEosKapUpdate, StarError> {
            let xa_k = &xa[k * species..(k + 1) * species];
            let comp_raw = get_composition_info(catalog, chem_id, xa_k, false)
                .map_err(|e| {
                    StarError::MicrophysicsError(format!(
                        "composition error at zone {k}: {e}"
                    ))
                })?;
            let comp = CellCompMoments {
                abar: comp_raw.abar,
                zbar: comp_raw.zbar,
                z2bar: comp_raw.z2bar,
                z53bar: comp_raw.z53bar,
                ye: comp_raw.ye,
                xh_mass: clamp_unit_roundoff(comp_raw.xh),
                yhe_mass: clamp_unit_roundoff(comp_raw.xhe),
                z_mass: clamp_unit_roundoff(comp_raw.xz),
            };
            let cno_ne = eval_cno_ne_mass_fractions(catalog, chem_id, xa_k);
            eval_one_cell_eos_kap(
                ht,
                chem_id,
                eos_cfg,
                kap_cfg,
                kap_factor,
                k,
                xa_k,
                comp,
                cno_ne,
                ln_t[k],
                lnd[k],
                ln_pgas[k],
                fix_pgas_by_cell[k],
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    for (offset, upd) in results.iter().enumerate() {
        write_cell_eos_kap_update(model, k_lo + offset, upd);
    }
    Ok(())
}

/// Snapshot of cell microphysics + composition + density over `k_lo..=k_hi`
/// for fast, allocation-light save/restore in `predictive_mix` and
/// `conv_premix`.
#[derive(Debug, Clone)]
pub(crate) struct CellRangeSnapshot {
    pub(crate) k_lo: usize,
    pub(crate) k_hi: usize,
    pub(crate) xa: Vec<f64>,
    pub(crate) lnd: Vec<f64>,
    pub(crate) rho: Vec<f64>,
    pub(crate) abar: Vec<f64>,
    pub(crate) zbar: Vec<f64>,
    pub(crate) z2bar: Vec<f64>,
    pub(crate) z53bar: Vec<f64>,
    pub(crate) ye: Vec<f64>,
    pub(crate) xh_mass: Vec<f64>,
    pub(crate) yhe_mass: Vec<f64>,
    pub(crate) z_mass: Vec<f64>,
    pub(crate) pgas: Vec<f64>,
    pub(crate) ln_pgas: Vec<f64>,
    pub(crate) prad: Vec<f64>,
    pub(crate) peos: Vec<f64>,
    pub(crate) ln_peos: Vec<f64>,
    pub(crate) energy: Vec<f64>,
    pub(crate) ln_e: Vec<f64>,
    pub(crate) entropy: Vec<f64>,
    pub(crate) ln_s: Vec<f64>,
    pub(crate) grada: Vec<f64>,
    pub(crate) cv: Vec<f64>,
    pub(crate) cp: Vec<f64>,
    pub(crate) chi_rho: Vec<f64>,
    pub(crate) chi_t: Vec<f64>,
    pub(crate) gamma1: Vec<f64>,
    pub(crate) gamma3: Vec<f64>,
    pub(crate) eta: Vec<f64>,
    pub(crate) gam: Vec<f64>,
    pub(crate) mu: Vec<f64>,
    pub(crate) lnfree_e: Vec<f64>,
    pub(crate) csound: Vec<f64>,
    pub(crate) eos_frac_pc: Vec<f64>,
    pub(crate) latent_ddlnt: Vec<f64>,
    pub(crate) latent_ddlnrho: Vec<f64>,
    pub(crate) dln_e_dxa: Vec<f64>,
    pub(crate) dln_peos_dxa: Vec<f64>,
    pub(crate) ln_peos_ad: Vec<StarAdOrder1>,
    pub(crate) ln_e_ad: Vec<StarAdOrder1>,
    pub(crate) ln_s_ad: Vec<StarAdOrder1>,
    pub(crate) eos_res: Vec<[f64; NUM_EOS_BASIC_RESULTS]>,
    pub(crate) eos_d_dlnd: Vec<[f64; NUM_EOS_BASIC_RESULTS]>,
    pub(crate) eos_d_dlnt: Vec<[f64; NUM_EOS_BASIC_RESULTS]>,
    pub(crate) opacity: Vec<f64>,
    pub(crate) d_opacity_dlnd: Vec<f64>,
    pub(crate) d_opacity_dlnt: Vec<f64>,
}

impl CellRangeSnapshot {
    /// Captures all composition, density, EOS, and opacity state across
    /// `k_lo..=k_hi`.
    #[must_use]
    pub(crate) fn capture(
        model: &StarModel,
        k_lo: usize,
        k_hi: usize,
    ) -> Self {
        let sp = model.species;
        let sp_lo = k_lo * sp;
        let sp_hi = (k_hi + 1) * sp;
        let r = k_lo..=k_hi;
        Self {
            k_lo,
            k_hi,
            xa: model.xa[sp_lo..sp_hi].to_vec(),
            lnd: model.lnd[r.clone()].to_vec(),
            rho: model.rho[r.clone()].to_vec(),
            abar: model.abar[r.clone()].to_vec(),
            zbar: model.zbar[r.clone()].to_vec(),
            z2bar: model.z2bar[r.clone()].to_vec(),
            z53bar: model.z53bar[r.clone()].to_vec(),
            ye: model.ye[r.clone()].to_vec(),
            xh_mass: model.xh_mass[r.clone()].to_vec(),
            yhe_mass: model.yhe_mass[r.clone()].to_vec(),
            z_mass: model.z_mass[r.clone()].to_vec(),
            pgas: model.pgas[r.clone()].to_vec(),
            ln_pgas: model.ln_pgas[r.clone()].to_vec(),
            prad: model.prad[r.clone()].to_vec(),
            peos: model.peos[r.clone()].to_vec(),
            ln_peos: model.ln_peos[r.clone()].to_vec(),
            energy: model.energy[r.clone()].to_vec(),
            ln_e: model.ln_e[r.clone()].to_vec(),
            entropy: model.entropy[r.clone()].to_vec(),
            ln_s: model.ln_s[r.clone()].to_vec(),
            grada: model.grada[r.clone()].to_vec(),
            cv: model.cv[r.clone()].to_vec(),
            cp: model.cp[r.clone()].to_vec(),
            chi_rho: model.chi_rho[r.clone()].to_vec(),
            chi_t: model.chi_t[r.clone()].to_vec(),
            gamma1: model.gamma1[r.clone()].to_vec(),
            gamma3: model.gamma3[r.clone()].to_vec(),
            eta: model.eta[r.clone()].to_vec(),
            gam: model.gam[r.clone()].to_vec(),
            mu: model.mu[r.clone()].to_vec(),
            lnfree_e: model.lnfree_e[r.clone()].to_vec(),
            csound: model.csound[r.clone()].to_vec(),
            eos_frac_pc: model.aux.eos_frac_pc[r.clone()].to_vec(),
            latent_ddlnt: model.aux.latent_ddlnt[r.clone()].to_vec(),
            latent_ddlnrho: model.aux.latent_ddlnrho[r.clone()].to_vec(),
            dln_e_dxa: model.aux.dln_e_dxa_for_partials[sp_lo..sp_hi].to_vec(),
            dln_peos_dxa: model.aux.dln_peos_dxa_for_partials[sp_lo..sp_hi]
                .to_vec(),
            ln_peos_ad: model.ln_peos_ad[r.clone()].to_vec(),
            ln_e_ad: model.ln_e_ad[r.clone()].to_vec(),
            ln_s_ad: model.ln_s_ad[r.clone()].to_vec(),
            eos_res: model.eos_res[r.clone()].to_vec(),
            eos_d_dlnd: model.eos_d_dlnd[r.clone()].to_vec(),
            eos_d_dlnt: model.eos_d_dlnt[r.clone()].to_vec(),
            opacity: model.opacity[r.clone()].to_vec(),
            d_opacity_dlnd: model.d_opacity_dlnd[r.clone()].to_vec(),
            d_opacity_dlnt: model.d_opacity_dlnt[r.clone()].to_vec(),
        }
    }

    /// Restores the captured range back into `model`.
    pub(crate) fn restore(&self, model: &mut StarModel) {
        let sp = model.species;
        let sp_lo = self.k_lo * sp;
        let sp_hi = (self.k_hi + 1) * sp;
        let r = self.k_lo..=self.k_hi;
        model.xa[sp_lo..sp_hi].copy_from_slice(&self.xa);
        model.lnd[r.clone()].copy_from_slice(&self.lnd);
        model.rho[r.clone()].copy_from_slice(&self.rho);
        model.abar[r.clone()].copy_from_slice(&self.abar);
        model.zbar[r.clone()].copy_from_slice(&self.zbar);
        model.z2bar[r.clone()].copy_from_slice(&self.z2bar);
        model.z53bar[r.clone()].copy_from_slice(&self.z53bar);
        model.ye[r.clone()].copy_from_slice(&self.ye);
        model.xh_mass[r.clone()].copy_from_slice(&self.xh_mass);
        model.yhe_mass[r.clone()].copy_from_slice(&self.yhe_mass);
        model.z_mass[r.clone()].copy_from_slice(&self.z_mass);
        model.pgas[r.clone()].copy_from_slice(&self.pgas);
        model.ln_pgas[r.clone()].copy_from_slice(&self.ln_pgas);
        model.prad[r.clone()].copy_from_slice(&self.prad);
        model.peos[r.clone()].copy_from_slice(&self.peos);
        model.ln_peos[r.clone()].copy_from_slice(&self.ln_peos);
        model.energy[r.clone()].copy_from_slice(&self.energy);
        model.ln_e[r.clone()].copy_from_slice(&self.ln_e);
        model.entropy[r.clone()].copy_from_slice(&self.entropy);
        model.ln_s[r.clone()].copy_from_slice(&self.ln_s);
        model.grada[r.clone()].copy_from_slice(&self.grada);
        model.cv[r.clone()].copy_from_slice(&self.cv);
        model.cp[r.clone()].copy_from_slice(&self.cp);
        model.chi_rho[r.clone()].copy_from_slice(&self.chi_rho);
        model.chi_t[r.clone()].copy_from_slice(&self.chi_t);
        model.gamma1[r.clone()].copy_from_slice(&self.gamma1);
        model.gamma3[r.clone()].copy_from_slice(&self.gamma3);
        model.eta[r.clone()].copy_from_slice(&self.eta);
        model.gam[r.clone()].copy_from_slice(&self.gam);
        model.mu[r.clone()].copy_from_slice(&self.mu);
        model.lnfree_e[r.clone()].copy_from_slice(&self.lnfree_e);
        model.csound[r.clone()].copy_from_slice(&self.csound);
        model.aux.eos_frac_pc[r.clone()].copy_from_slice(&self.eos_frac_pc);
        model.aux.latent_ddlnt[r.clone()].copy_from_slice(&self.latent_ddlnt);
        model.aux.latent_ddlnrho[r.clone()]
            .copy_from_slice(&self.latent_ddlnrho);
        model.aux.dln_e_dxa_for_partials[sp_lo..sp_hi]
            .copy_from_slice(&self.dln_e_dxa);
        model.aux.dln_peos_dxa_for_partials[sp_lo..sp_hi]
            .copy_from_slice(&self.dln_peos_dxa);
        model.ln_peos_ad[r.clone()].copy_from_slice(&self.ln_peos_ad);
        model.ln_e_ad[r.clone()].copy_from_slice(&self.ln_e_ad);
        model.ln_s_ad[r.clone()].copy_from_slice(&self.ln_s_ad);
        model.eos_res[r.clone()].copy_from_slice(&self.eos_res);
        model.eos_d_dlnd[r.clone()].copy_from_slice(&self.eos_d_dlnd);
        model.eos_d_dlnt[r.clone()].copy_from_slice(&self.eos_d_dlnt);
        model.opacity[r.clone()].copy_from_slice(&self.opacity);
        model.d_opacity_dlnd[r.clone()].copy_from_slice(&self.d_opacity_dlnd);
        model.d_opacity_dlnt[r].copy_from_slice(&self.d_opacity_dlnt);
    }
}

/// Snapshot of face-centered MLT variables over `k_lo..=k_hi` so trial
/// `do1_mlt_2` calls in `predictive_mix` and `conv_premix` can be rolled back
/// to 0 ULP without repeating opacity/MLT evaluations.
#[derive(Debug, Clone)]
pub(crate) struct FaceMltSnapshot {
    pub(crate) k_lo: usize,
    pub(crate) k_hi: usize,
    pub(crate) gradt: Vec<f64>,
    pub(crate) gradt_ad: Vec<StarAdOrder1>,
    pub(crate) gradr: Vec<f64>,
    pub(crate) gradr_ad: Vec<StarAdOrder1>,
    pub(crate) gradl: Vec<f64>,
    pub(crate) gradl_ad: Vec<StarAdOrder1>,
    pub(crate) scale_height: Vec<f64>,
    pub(crate) scale_height_ad: Vec<StarAdOrder1>,
    pub(crate) mlt_vc: Vec<f64>,
    pub(crate) mlt_vc_ad: Vec<StarAdOrder1>,
    pub(crate) mlt_d: Vec<f64>,
    pub(crate) mlt_d_ad: Vec<StarAdOrder1>,
    pub(crate) mlt_cdc: Vec<f64>,
    pub(crate) mlt_cdc_ad: Vec<StarAdOrder1>,
    pub(crate) mlt_gamma: Vec<f64>,
    pub(crate) mlt_mixing_type: Vec<flexmesa_const::MixingType>,
    pub(crate) mlt_mixing_length: Vec<f64>,
    pub(crate) grada_face: Vec<f64>,
    pub(crate) grada_face_ad: Vec<StarAdOrder1>,
    pub(crate) gradt_sub_grada: Vec<f64>,
    pub(crate) mlt_gradt: Vec<f64>,
    pub(crate) y_face: Vec<f64>,
    pub(crate) l_conv: Vec<f64>,
    pub(crate) superad_reduction_factor: Vec<f64>,
    pub(crate) gradt_excess_effect: Vec<f64>,
}

impl FaceMltSnapshot {
    /// Captures all face MLT outputs over `k_lo..=k_hi`.
    #[must_use]
    pub(crate) fn capture(
        model: &StarModel,
        k_lo: usize,
        k_hi: usize,
    ) -> Self {
        let r = k_lo..=k_hi;
        let m = &model.aux.mixing;
        Self {
            k_lo,
            k_hi,
            gradt: model.gradt[r.clone()].to_vec(),
            gradt_ad: model.gradt_ad[r.clone()].to_vec(),
            gradr: model.gradr[r.clone()].to_vec(),
            gradr_ad: model.gradr_ad[r.clone()].to_vec(),
            gradl: model.gradl[r.clone()].to_vec(),
            gradl_ad: model.gradl_ad[r.clone()].to_vec(),
            scale_height: model.scale_height[r.clone()].to_vec(),
            scale_height_ad: model.scale_height_ad[r.clone()].to_vec(),
            mlt_vc: model.mlt_vc[r.clone()].to_vec(),
            mlt_vc_ad: model.mlt_vc_ad[r.clone()].to_vec(),
            mlt_d: model.mlt_d[r.clone()].to_vec(),
            mlt_d_ad: model.mlt_d_ad[r.clone()].to_vec(),
            mlt_cdc: model.mlt_cdc[r.clone()].to_vec(),
            mlt_cdc_ad: model.mlt_cdc_ad[r.clone()].to_vec(),
            mlt_gamma: model.mlt_gamma[r.clone()].to_vec(),
            mlt_mixing_type: m.mlt_mixing_type[r.clone()].to_vec(),
            mlt_mixing_length: m.mlt_mixing_length[r.clone()].to_vec(),
            grada_face: m.grada_face[r.clone()].to_vec(),
            grada_face_ad: m.grada_face_ad[r.clone()].to_vec(),
            gradt_sub_grada: m.gradt_sub_grada[r.clone()].to_vec(),
            mlt_gradt: m.mlt_gradt[r.clone()].to_vec(),
            y_face: m.y_face[r.clone()].to_vec(),
            l_conv: m.l_conv[r.clone()].to_vec(),
            superad_reduction_factor: m.superad_reduction_factor[r.clone()]
                .to_vec(),
            gradt_excess_effect: m.gradt_excess_effect[r].to_vec(),
        }
    }

    /// Restores the captured face MLT outputs over `k_lo..=k_hi`.
    pub(crate) fn restore(&self, model: &mut StarModel) {
        let r = self.k_lo..=self.k_hi;
        model.gradt[r.clone()].copy_from_slice(&self.gradt);
        model.gradt_ad[r.clone()].copy_from_slice(&self.gradt_ad);
        model.gradr[r.clone()].copy_from_slice(&self.gradr);
        model.gradr_ad[r.clone()].copy_from_slice(&self.gradr_ad);
        model.gradl[r.clone()].copy_from_slice(&self.gradl);
        model.gradl_ad[r.clone()].copy_from_slice(&self.gradl_ad);
        model.scale_height[r.clone()].copy_from_slice(&self.scale_height);
        model.scale_height_ad[r.clone()].copy_from_slice(&self.scale_height_ad);
        model.mlt_vc[r.clone()].copy_from_slice(&self.mlt_vc);
        model.mlt_vc_ad[r.clone()].copy_from_slice(&self.mlt_vc_ad);
        model.mlt_d[r.clone()].copy_from_slice(&self.mlt_d);
        model.mlt_d_ad[r.clone()].copy_from_slice(&self.mlt_d_ad);
        model.mlt_cdc[r.clone()].copy_from_slice(&self.mlt_cdc);
        model.mlt_cdc_ad[r.clone()].copy_from_slice(&self.mlt_cdc_ad);
        model.mlt_gamma[r.clone()].copy_from_slice(&self.mlt_gamma);
        let m = &mut model.aux.mixing;
        m.mlt_mixing_type[r.clone()].copy_from_slice(&self.mlt_mixing_type);
        m.mlt_mixing_length[r.clone()].copy_from_slice(&self.mlt_mixing_length);
        m.grada_face[r.clone()].copy_from_slice(&self.grada_face);
        m.grada_face_ad[r.clone()].copy_from_slice(&self.grada_face_ad);
        m.gradt_sub_grada[r.clone()].copy_from_slice(&self.gradt_sub_grada);
        m.mlt_gradt[r.clone()].copy_from_slice(&self.mlt_gradt);
        m.y_face[r.clone()].copy_from_slice(&self.y_face);
        m.l_conv[r.clone()].copy_from_slice(&self.l_conv);
        m.superad_reduction_factor[r.clone()]
            .copy_from_slice(&self.superad_reduction_factor);
        m.gradt_excess_effect[r].copy_from_slice(&self.gradt_excess_effect);
    }
}
