// SPDX-License-Identifier: LGPL-3.0-or-later
//! Zone-level equation of state (`do_eos`), thermal neutrino loss (`do_neu`),
//! Rosseland mean opacity and optical depth (`do_kap`, `get_tau`), and nuclear
//! reaction network (`do_net`) evaluation (`star/private/micro.f90` and
//! `star/private/net.f90`).

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_const::{AVO, CRAD, FOUR_PI, KERG, QE};
use flexmesa_eos::{
    eos_dt_get_zones, HelmTable, I_CHI_RHO, I_CHI_T, I_CP, I_CV, I_ETA,
    I_GAMMA1, I_GAMMA3, I_GRAD_AD, I_LN_E, I_LN_FREE_E, I_LN_PGAS, I_LN_S,
    I_MU,
};
use flexmesa_kap::get_kap_results;
use flexmesa_net::{net_get, NetEvalResult, NetGeneralInfo};
use flexmesa_neu::{neu_get, NeuFlags, NeuResult};
use rayon::prelude::*;

use crate::controls::{StarError, VarLayout};
use crate::model::StarModel;
use crate::star_ad::{StarAdExt, StarAdOrder1};

static BASIC_NET_INFO: OnceLock<NetGeneralInfo> = OnceLock::new();
static NET_CACHE: OnceLock<Mutex<HashMap<String, Arc<NetGeneralInfo>>>> =
    OnceLock::new();

/// Returns the cached [`NetGeneralInfo`] for `"basic.net"`.
#[must_use]
pub fn global_basic_net() -> &'static NetGeneralInfo {
    BASIC_NET_INFO.get_or_init(|| {
        NetGeneralInfo::from_name("basic.net")
            .expect("valid built-in basic.net")
    })
}

/// Returns a cached [`Arc<NetGeneralInfo>`] for `net_name` (loading from the
/// embedded 83-network bundle or optional `search_dir` on first request).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if the `.net` file cannot be parsed.
pub fn get_or_load_net(
    net_name: &str,
    search_dir: Option<&Path>,
) -> Result<Arc<NetGeneralInfo>, StarError> {
    let key = net_name.trim();
    let cache = NET_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(guard) = cache.lock() {
        if let Some(info) = guard.get(key) {
            return Ok(Arc::clone(info));
        }
    }
    let info = Arc::new(
        NetGeneralInfo::from_name_or_dir(key, search_dir).map_err(|e| {
            StarError::MicrophysicsError(format!(
                "failed to load network '{key}': {e}"
            ))
        })?,
    );
    if let Ok(mut guard) = cache.lock() {
        guard.insert(key.to_string(), Arc::clone(&info));
    }
    Ok(info)
}

/// Returns a cached [`Arc<NetGeneralInfo>`] matching `model.job.net_name` and
/// `model.species`, or `None` if the network cannot be resolved or its species
/// count does not match `model.species`.
#[must_use]
pub fn resolve_model_net(model: &StarModel) -> Option<Arc<NetGeneralInfo>> {
    let net_info = get_or_load_net(&model.job.net_name, None).ok()?;
    (net_info.num_isos == model.species).then_some(net_info)
}

/// Projects `model` from its current isotope set onto `new_net_name`
/// (`change_net` in `star/private/adjust_xyz.f90:584-712`), preserving exact
/// mass fractions for shared isotopes and dumping any removed trace mass into
/// the heaviest retained metal so $\sum_j X_{j,k} = 1$ in every zone.
///
/// # Errors
/// Returns [`StarError`] if `new_net_name` cannot be loaded.
pub fn change_model_network(
    model: &mut StarModel,
    new_net_name: &str,
    search_dir: Option<&Path>,
) -> Result<(), StarError> {
    let net_info = get_or_load_net(new_net_name, search_dir)?;
    let new_chem_id = net_info.chem_id.clone();
    let new_species = new_chem_id.len();
    if new_species == 0 {
        return Err(StarError::InvalidInput(format!(
            "empty species list in network '{new_net_name}'"
        )));
    }
    let old_species = model.species;
    let nz = model.nz;
    let mut new_xa = vec![0.0_f64; nz * new_species];
    let mut new_net_iso = vec![None; 7857];
    let catalog = NuclideCatalog::builtin_ref();
    let mut new_iso_names = Vec::with_capacity(new_species);
    for (j_new, &cid) in new_chem_id.iter().enumerate() {
        let name = catalog
            .get_by_id(cid)
            .map_or_else(|| format!("iso_{cid}"), |n| n.name.clone());
        new_iso_names.push(name);
        if cid < new_net_iso.len() {
            new_net_iso[cid] = Some(j_new);
        }
    }

    for k in 0..nz {
        let mut transferred = 0.0_f64;
        for (j_old, &cid_old) in model.chem_id.iter().enumerate() {
            let x_val = model.xa[k * old_species + j_old].max(0.0);
            if let Some(Some(j_new)) = new_net_iso.get(cid_old) {
                new_xa[k * new_species + *j_new] += x_val;
                transferred += x_val;
            }
        }
        let missing = (1.0 - transferred).max(0.0);
        if missing > 0.0 {
            new_xa[k * new_species + (new_species - 1)] += missing;
        }
        let sum: f64 =
            new_xa[k * new_species..(k + 1) * new_species].iter().sum();
        if sum > 0.0 {
            for j in 0..new_species {
                new_xa[k * new_species + j] /= sum;
            }
        }
    }

    model.job.net_name = new_net_name.trim().to_string();
    model.species = new_species;
    model.chem_id = new_chem_id;
    model.iso_names = new_iso_names;
    model.net_iso = new_net_iso;
    model.layout = VarLayout::new(
        new_species,
        model.controls.v_flag,
        model.controls.u_flag,
    );
    model.xa.clone_from(&new_xa);
    model.xa_start.clone_from(&new_xa);
    model.xa_old = new_xa;
    model.d_epsnuc_dx.resize(nz * new_species, 0.0);
    model.dxdt_nuc.resize(nz * new_species, 0.0);
    model.d_dxdt_nuc_dlnd.resize(nz * new_species, 0.0);
    model.d_dxdt_nuc_dlnt.resize(nz * new_species, 0.0);
    model
        .d_dxdt_nuc_dx
        .resize(nz * new_species * new_species, 0.0);
    model.aux.resize(nz, new_species);
    Ok(())
}

/// Evaluates `rho`, `t`, `r`, `rmid`, `grav`, and composition moments (`abar`,
/// `zbar`, `z2bar`, `z53bar`, `ye`, `xh_mass`, `yhe_mass`, `z_mass`) across all
/// `nz` zones (`set_basic_vars` in `star/private/hydro_vars.f90:511-754`).
///
/// # Errors
/// Returns [`StarError`] if any zone contains non-finite state or invalid
/// composition.
pub fn set_basic_vars(model: &mut StarModel) -> Result<(), StarError> {
    let nz = model.nz;
    let species = model.species;
    let catalog = NuclideCatalog::builtin_ref();

    model.set_m_and_dm();
    model.set_m_grav_and_grav();

    for k in 0..nz {
        let lnd_k = model.lnd[k];
        let lnt_k = model.ln_t[k];
        if !lnd_k.is_finite()
            || !lnt_k.is_finite()
            || !model.ln_r[k].is_finite()
        {
            return Err(StarError::MicrophysicsError(format!(
                "non-finite primary state at zone {k}: lnd={lnd_k}, lnT={lnt_k}"
            )));
        }
        model.rho[k] = lnd_k.exp();
        model.t[k] = lnt_k.exp();

        let xa_k = &model.xa[k * species..(k + 1) * species];
        let comp = get_composition_info(catalog, &model.chem_id, xa_k, false)
            .map_err(|e| {
            StarError::MicrophysicsError(format!(
                "composition error at zone {k}: {e}"
            ))
        })?;
        model.abar[k] = comp.abar;
        model.zbar[k] = comp.zbar;
        model.z2bar[k] = comp.z2bar;
        model.z53bar[k] = comp.z53bar;
        model.ye[k] = comp.ye;
        model.xh_mass[k] = comp.xh;
        model.yhe_mass[k] = comp.xhe;
        model.z_mass[k] = comp.xz;
    }
    Ok(())
}

/// Evaluates `eos_dt_get_zones` across all `nz` zones in parallel and populates
/// all cell-centered thermodynamic quantities and `StarAdOrder1` variables
/// (`do_eos` in `star/private/micro.f90:151-656`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if EOS evaluation fails in any zone.
pub fn do_eos(model: &mut StarModel) -> Result<(), StarError> {
    let ht = HelmTable::builtin();
    let eos_outputs = match eos_dt_get_zones(
        ht,
        &model.rho,
        &model.t,
        &model.chem_id,
        &model.xa,
        &model.eos_config,
    ) {
        Ok(outputs) => outputs,
        Err(_)
            if model.eos_config.use_free_eos
                || model.eos_config.use_opal_scvh =>
        {
            model.eos_config.use_free_eos = false;
            model.eos_config.use_opal_scvh = false;
            eos_dt_get_zones(
                ht,
                &model.rho,
                &model.t,
                &model.chem_id,
                &model.xa,
                &model.eos_config,
            )
            .map_err(|e| {
                StarError::MicrophysicsError(format!("do_eos failed: {e}"))
            })?
        }
        Err(e) => {
            return Err(StarError::MicrophysicsError(format!(
                "do_eos failed: {e}"
            )));
        }
    };

    for (k, out) in eos_outputs.into_iter().enumerate() {
        let t_k = model.t[k];
        let rho_k = model.rho[k];
        let ln_pgas = out.res[I_LN_PGAS];
        let pgas = ln_pgas.exp();
        let prad = CRAD * t_k.powi(4) / 3.0;
        let peos = pgas + prad;
        let ln_peos = peos.ln();

        // Gas and total pressure logarithmic derivatives
        // d(ln Pgas)/d(lnd) and d(ln Pgas)/d(lnT) from EOS res/d_dlnd/d_dln_t
        let d_lnpgas_dlnd = out.d_dlnd[I_LN_PGAS];
        let d_lnpgas_dlnt = out.d_dln_t[I_LN_PGAS];
        let alfa_rad = prad / peos;
        let beta_gas = pgas / peos;
        let d_lnpeos_dlnd = beta_gas * d_lnpgas_dlnd;
        let d_lnpeos_dlnt = beta_gas * d_lnpgas_dlnt + 4.0 * alfa_rad;

        let ln_e = out.res[I_LN_E];
        let energy = ln_e.exp();
        let ln_s = out.res[I_LN_S];
        let entropy = ln_s.exp();
        let gamma1 = out.res[I_GAMMA1];
        let csound = (gamma1 * peos / rho_k).max(0.0).sqrt();

        // Coulomb coupling parameter Gamma_Coulomb (`micro.f90:619`)
        let a_e = (3.0
            / (FOUR_PI * AVO * rho_k * model.zbar[k].max(1.0e-10)
                / model.abar[k]))
            .cbrt();
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

        model.ln_peos_ad[k] = StarAdOrder1::from_dlnd_dlnt_00(
            ln_peos,
            d_lnpeos_dlnd,
            d_lnpeos_dlnt,
        );
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
    Ok(())
}

/// Evaluates thermal neutrino loss rates `non_nuc_neu` and partial derivatives
/// across all `nz` zones (`do_neu` in `star/private/micro.f90:1278-1370`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if `neu_get` fails.
pub fn do_neu(model: &mut StarModel) -> Result<(), StarError> {
    let nz = model.nz;
    let min_t = model.controls.min_t_for_neu;
    let factor = model.controls.non_nuc_neu_factor;
    if factor <= 0.0 {
        model.non_nuc_neu.fill(0.0);
        model.d_non_nuc_neu_dlnd.fill(0.0);
        model.d_non_nuc_neu_dlnt.fill(0.0);
        return Ok(());
    }

    let flags = NeuFlags::default_stellar();
    let results: Result<Vec<(f64, f64, f64)>, StarError> = (0..nz)
        .into_par_iter()
        .map(|k| {
            let t_k = model.t[k];
            if t_k < min_t {
                return Ok((0.0, 0.0, 0.0));
            }
            let rho_k = model.rho[k];
            let log_t = t_k.log10();
            let log_rho = rho_k.log10();
            let NeuResult { total, .. } = neu_get(
                Some(t_k),
                Some(log_t),
                Some(rho_k),
                Some(log_rho),
                model.abar[k],
                model.zbar[k],
                7.5,
                flags,
            )
            .map_err(|e| {
                StarError::MicrophysicsError(format!(
                    "neu_get failed at zone {k}: {e}"
                ))
            })?;
            Ok((
                total.val * factor,
                total.d_rho * rho_k * factor,
                total.d_t * t_k * factor,
            ))
        })
        .collect();

    for (k, (val, dlnd, dlnt)) in results?.into_iter().enumerate() {
        model.non_nuc_neu[k] = val;
        model.d_non_nuc_neu_dlnd[k] = dlnd;
        model.d_non_nuc_neu_dlnt[k] = dlnt;
    }
    Ok(())
}

/// Evaluates Rosseland mean opacity `opacity` (radiative + Compton + electron
/// conduction) in parallel across all `nz` zones and integrates optical depth
/// `tau` inward from the surface (`do_kap` and `get_tau` in
/// `star/private/micro.f90:739-1052`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if `get_kap_results` fails.
pub fn do_kap(model: &mut StarModel) -> Result<(), StarError> {
    let nz = model.nz;
    let species = model.species;
    let catalog = NuclideCatalog::builtin_ref();
    let idx_c12 = catalog
        .get_nuclide_index("c12")
        .and_then(|cid| model.net_iso.get(cid).copied().flatten());
    let idx_n14 = catalog
        .get_nuclide_index("n14")
        .and_then(|cid| model.net_iso.get(cid).copied().flatten());
    let idx_o16 = catalog
        .get_nuclide_index("o16")
        .and_then(|cid| model.net_iso.get(cid).copied().flatten());
    let idx_ne20 = catalog
        .get_nuclide_index("ne20")
        .and_then(|cid| model.net_iso.get(cid).copied().flatten());
    let kap_factor = model.controls.opacity_factor;
    let kap_cfg = &model.kap_config;

    let kap_results: Result<Vec<(f64, f64, f64)>, StarError> = (0..nz)
        .into_par_iter()
        .map(|k| {
            let xa_k = &model.xa[k * species..(k + 1) * species];
            let xc = idx_c12.map_or(0.0, |j| xa_k[j]);
            let xn = idx_n14.map_or(0.0, |j| xa_k[j]);
            let xo = idx_o16.map_or(0.0, |j| xa_k[j]);
            let xne = idx_ne20.map_or(0.0, |j| xa_k[j]);
            let rho_k = model.rho[k];
            let t_k = model.t[k];
            let lnfree_e = model.lnfree_e[k];
            let out = get_kap_results(
                kap_cfg,
                model.zbar[k],
                model.xh_mass[k],
                model.z_mass[k],
                xc,
                xn,
                xo,
                xne,
                rho_k.log10(),
                t_k.log10(),
                lnfree_e,
                model.eos_d_dlnd[k][I_LN_FREE_E],
                model.eos_d_dlnt[k][I_LN_FREE_E],
                model.eta[k],
                model.eos_d_dlnd[k][I_ETA],
                model.eos_d_dlnt[k][I_ETA],
            )
            .map_err(|e| {
                StarError::MicrophysicsError(format!(
                    "get_kap_results failed at zone {k}: {e}"
                ))
            })?;
            Ok((
                out.kap * kap_factor,
                out.kap * out.dlnkap_dln_rho * kap_factor,
                out.kap * out.dlnkap_dln_t * kap_factor,
            ))
        })
        .collect();

    for (k, (kap, dlnd, dlnt)) in kap_results?.into_iter().enumerate() {
        model.opacity[k] = kap;
        model.d_opacity_dlnd[k] = dlnd;
        model.d_opacity_dlnt[k] = dlnt;
    }

    get_tau(model);
    Ok(())
}

/// Integrates optical depth $\tau_k$ inward from $\tau_0 = \tau_{\mathrm{factor}} \tau_{\mathrm{base}}$
/// (`get_tau` in `star/private/micro.f90:994-1052`).
pub fn get_tau(model: &mut StarModel) {
    let nz = model.nz;
    if nz == 0 {
        return;
    }
    model.tau[0] = model.controls.tau_factor * model.tau_base;
    for k in 1..nz {
        let dtau = model.dm[k - 1] * model.opacity[k - 1]
            / (FOUR_PI * model.rmid[k - 1] * model.rmid[k - 1]);
        model.tau[k] = model.tau[k - 1] + dtau;
    }
}

/// Evaluates the nuclear reaction network (`net_get`) across all `nz` zones in
/// parallel (`do_net` in `star/private/net.f90:62-436`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if `net_get` fails.
pub fn do_net(
    model: &mut StarModel,
    net_info: &NetGeneralInfo,
    skip_jacobian: bool,
) -> Result<(), StarError> {
    let nz = model.nz;
    let species = model.species;
    let min_logt = model.controls.min_logt_for_net;
    let eps_fac = model.controls.eps_nuc_factor;
    let dxdt_fac = model.controls.dxdt_nuc_factor;
    let screening_mode = model.controls.screening_mode;
    let rate_factors = vec![1.0; net_info.num_reactions];

    let net_results: Result<Vec<Option<NetEvalResult>>, StarError> = (0..nz)
        .into_par_iter()
        .map(|k| {
            let t_k = model.t[k];
            let log_t = t_k.log10();
            if log_t < min_logt {
                return Ok(None);
            }
            let rho_k = model.rho[k];
            let log_rho = rho_k.log10();
            let xa_k = &model.xa[k * species..(k + 1) * species];
            let res = net_get(
                net_info,
                skip_jacobian,
                xa_k,
                t_k,
                log_t,
                rho_k,
                log_rho,
                model.abar[k],
                model.zbar[k],
                model.z2bar[k],
                model.ye[k],
                model.eta[k],
                model.eos_d_dlnt[k][I_ETA],
                model.eos_d_dlnd[k][I_ETA],
                &rate_factors,
                1.0,
                screening_mode,
            )
            .map_err(|e| {
                StarError::MicrophysicsError(format!(
                    "net_get failed at zone {k}: {e}"
                ))
            })?;
            Ok(Some(res))
        })
        .collect();

    for (k, opt_res) in net_results?.into_iter().enumerate() {
        let base_s = k * species;
        let base_ss = k * species * species;
        if let Some(res) = opt_res {
            let rho_k = model.rho[k];
            let t_k = model.t[k];
            model.eps_nuc[k] = res.eps_nuc * eps_fac;
            model.d_epsnuc_dlnd[k] = res.d_eps_nuc_drho * rho_k * eps_fac;
            model.d_epsnuc_dlnt[k] = res.d_eps_nuc_dt * t_k * eps_fac;
            model.eps_nuc_neu_total[k] = res.eps_neu_total * eps_fac;
            for j in 0..species {
                model.d_epsnuc_dx[base_s + j] = res.d_eps_nuc_dx[j] * eps_fac;
                model.dxdt_nuc[base_s + j] = res.dxdt[j] * dxdt_fac;
                model.d_dxdt_nuc_dlnd[base_s + j] =
                    res.d_dxdt_drho[j] * rho_k * dxdt_fac;
                model.d_dxdt_nuc_dlnt[base_s + j] =
                    res.d_dxdt_dt[j] * t_k * dxdt_fac;
            }
            if !skip_jacobian && res.d_dxdt_dx.len() == species * species {
                for idx in 0..species * species {
                    model.d_dxdt_nuc_dx[base_ss + idx] =
                        res.d_dxdt_dx[idx] * dxdt_fac;
                }
            }
        } else {
            model.eps_nuc[k] = 0.0;
            model.d_epsnuc_dlnd[k] = 0.0;
            model.d_epsnuc_dlnt[k] = 0.0;
            model.eps_nuc_neu_total[k] = 0.0;
            model.d_epsnuc_dx[base_s..base_s + species].fill(0.0);
            model.dxdt_nuc[base_s..base_s + species].fill(0.0);
            model.d_dxdt_nuc_dlnd[base_s..base_s + species].fill(0.0);
            model.d_dxdt_nuc_dlnt[base_s..base_s + species].fill(0.0);
            model.d_dxdt_nuc_dx[base_ss..base_ss + species * species].fill(0.0);
        }
    }
    Ok(())
}
