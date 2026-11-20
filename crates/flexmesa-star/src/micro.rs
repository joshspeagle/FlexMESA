// SPDX-License-Identifier: LGPL-3.0-or-later
//! Zone-level equation of state (`do_eos`), thermal neutrino loss (`do_neu`),
//! Rosseland mean opacity and optical depth (`do_kap`, `get_tau`), and nuclear
//! reaction network (`do_net`) evaluation (`star/private/micro.f90` and
//! `star/private/net.f90`).

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_const::{
    AVO, CRAD, FOUR_PI, FOUR_THIRDS_PI, KERG, LN10, ONE_THIRD, QE,
};
use flexmesa_eos::{
    eos_dt_get_zones_with_logs, EosEvalOutput, HelmTable, I_CHI_RHO, I_CHI_T,
    I_CP, I_CV, I_ETA, I_FRAC_PC, I_GAMMA1, I_GAMMA3, I_GRAD_AD,
    I_LATENT_DDLNRHO, I_LATENT_DDLNT, I_LN_E, I_LN_FREE_E, I_LN_PGAS, I_LN_S,
    I_MU, NUM_EOS_BASIC_RESULTS,
};
use flexmesa_kap::get_kap_results;
use flexmesa_net::NetGeneralInfo;
use flexmesa_neu::{neu_get, NeuFlags, NeuResult};
use flexmesa_utils::is_bad;
use rayon::prelude::*;

use crate::controls::{StarError, VarLayout};
pub use crate::micro_net::{build_rate_factors, do_net};
use crate::model::StarModel;
use crate::star_ad::{StarAdExt, StarAdOrder1};

static BASIC_NET_INFO: OnceLock<NetGeneralInfo> = OnceLock::new();
type NetCacheKey = (String, u64, u64, u64, u64, u64, bool, bool);
static NET_CACHE: OnceLock<Mutex<HashMap<NetCacheKey, Arc<NetGeneralInfo>>>> =
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
    get_or_load_net_with_fe56ec(net_name, search_dir, 1.0e-7, 3.0e9)
}

/// Returns a cached [`Arc<NetGeneralInfo>`] for `net_name` with explicit
/// `fe56ec_fake_factor` and `min_t_for_fe56ec_fake_factor` applied.
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if the `.net` file cannot be parsed.
pub fn get_or_load_net_with_fe56ec(
    net_name: &str,
    search_dir: Option<&Path>,
    fe56ec_fake_factor: f64,
    min_t_for_fe56ec_fake_factor: f64,
) -> Result<Arc<NetGeneralInfo>, StarError> {
    get_or_load_net_with_star_params(
        net_name,
        search_dir,
        fe56ec_fake_factor,
        min_t_for_fe56ec_fake_factor,
        -1.0,
        -1.0,
        1.0,
        false,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn get_or_load_net_with_star_params(
    net_name: &str,
    search_dir: Option<&Path>,
    fe56ec_fake_factor: f64,
    min_t_for_fe56ec_fake_factor: f64,
    net_logtcut_lo: f64,
    net_logtcut_lim: f64,
    reaction_neuqs_factor: f64,
    use_3a_fl87: bool,
    use_suzuki_weak_rates: bool,
) -> Result<Arc<NetGeneralInfo>, StarError> {
    let key = net_name.trim();
    let cache_key = (
        key.to_string(),
        fe56ec_fake_factor.to_bits(),
        min_t_for_fe56ec_fake_factor.to_bits(),
        net_logtcut_lo.to_bits(),
        net_logtcut_lim.to_bits(),
        reaction_neuqs_factor.to_bits(),
        use_3a_fl87,
        use_suzuki_weak_rates,
    );
    let cache = NET_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(guard) = cache.lock() {
        if let Some(info) = guard.get(&cache_key) {
            return Ok(Arc::clone(info));
        }
    }
    let mut loaded = NetGeneralInfo::from_name_or_dir(key, search_dir)
        .map_err(|e| {
            StarError::MicrophysicsError(format!(
                "failed to load network '{key}': {e}"
            ))
        })?;
    // `star/private/net.f90:685,741-748` and `net/public/net_def.f90:588-606`:
    // unconditionally assign `s% net_logTcut_{lo,lim}` and `s% fe56ec_*`
    // (overriding standalone `net_def` defaults despite the stale `<= 0`
    // comment in `controls.defaults:6786`), plus `s% reaction_neuQs_factor`
    // (`star/private/net.f90:202-210`) and `use_3a_fl87` / `use_suzuki_weak_rates`.
    loaded.fe56ec_fake_factor = fe56ec_fake_factor;
    loaded.min_t_for_fe56ec_fake_factor = min_t_for_fe56ec_fake_factor;
    loaded.log_tcut_lo = net_logtcut_lo;
    loaded.log_tcut_lim = net_logtcut_lim;
    loaded.reaction_neuqs_factor = reaction_neuqs_factor;
    loaded.use_3a_fl87 = use_3a_fl87;
    loaded.use_suzuki_weak_rates = use_suzuki_weak_rates;
    let info = Arc::new(loaded);
    if let Ok(mut guard) = cache.lock() {
        guard.insert(cache_key, Arc::clone(&info));
    }
    Ok(info)
}

/// Returns a cached [`Arc<NetGeneralInfo>`] matching `model.job.net_name`,
/// `model.species`, and `model.controls` / `model.job` rate settings.
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if `model.job.net_name` cannot be
/// loaded or if `net_info.num_isos != model.species`.
pub fn resolve_model_net(
    model: &StarModel,
) -> Result<Arc<NetGeneralInfo>, StarError> {
    let c = &model.controls;
    let j = &model.job;
    let net_info = get_or_load_net_with_star_params(
        &j.net_name,
        None,
        c.fe56ec_fake_factor,
        c.min_t_for_fe56ec_fake_factor,
        c.net_logtcut_lo,
        c.net_logtcut_lim,
        c.reaction_neuqs_factor,
        j.use_3a_fl87,
        j.use_suzuki_weak_rates,
    )?;
    if net_info.num_isos != model.species {
        return Err(StarError::MicrophysicsError(format!(
            "network '{}' has {} species, expected model.species={}",
            model.job.net_name, net_info.num_isos, model.species
        )));
    }
    Ok(net_info)
}

/// Projects `model` from its current isotope set onto `new_net_name`
/// (`change_net` in `star/private/adjust_xyz.f90:51-59`), preserving exact
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
    model.layout = VarLayout::with_flags(
        new_species,
        model.controls.v_flag,
        model.controls.u_flag,
        model.controls.rsp2_flag,
        model.controls.rti_flag,
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
/// `nz` zones (`set_basic_vars` in `star/private/hydro_vars.f90:696-760`).
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
        model.rho[k] = flexmesa_math::exp(lnd_k);
        model.t[k] = flexmesa_math::exp(lnt_k);

        let xa_k = &model.xa[k * species..(k + 1) * species];
        let comp = get_composition_info(catalog, &model.chem_id, xa_k, true)
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
        // `star/private/star_utils.f90:2029-2043` (`get_XYZ`).
        let xh = comp.xh.clamp(0.0, 1.0);
        let xhe = comp.xhe.clamp(0.0, 1.0);
        model.xh_mass[k] = xh;
        model.yhe_mass[k] = xhe;
        model.z_mass[k] = (1.0 - (xh + xhe)).clamp(0.0, 1.0);
    }
    Ok(())
}

/// Returns the shared [`HelmTable`] or maps table I/O errors into [`StarError`].
#[inline]
pub(crate) fn builtin_helm_table() -> Result<&'static HelmTable, StarError> {
    HelmTable::builtin()
        .map_err(|e| StarError::MicrophysicsError(e.to_string()))
}

/// Evaluates `eosDT_get` across all `nz` zones in parallel and populates
/// cell-centered thermodynamic quantities, composition partial derivatives
/// (`dln_e_dxa_for_partials`, `dln_peos_dxa_for_partials`), and `StarAdOrder1`
/// variables (`do_eos_for_cell` and `store_eos_for_cell` in
/// `star/private/micro.f90:292-545` and `star/private/eos_support.f90:44-101`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if EOS evaluation fails or yields
/// non-physical thermodynamic derivatives in any zone.
pub fn do_eos(model: &mut StarModel) -> Result<(), StarError> {
    let nz = model.nz;
    let mut log_rho_zones = Vec::with_capacity(nz);
    let mut log_t_zones = Vec::with_capacity(nz);
    for k in 0..nz {
        let log_rho = model.lnd[k] / LN10;
        let log_t = model.ln_t[k] / LN10;
        if log_rho < -25.0 {
            return Err(StarError::MicrophysicsError(format!(
                "do_eos failed at zone {k}: logRho ({log_rho}) < -25"
            )));
        }
        log_rho_zones.push(log_rho);
        log_t_zones.push(log_t);
    }

    let ht = builtin_helm_table()?;
    let eos_outputs = eos_dt_get_zones_with_logs(
        ht,
        &model.rho,
        &log_rho_zones,
        &model.t,
        &log_t_zones,
        &model.chem_id,
        &model.xa,
        &model.eos_config,
    )
    .map_err(|e| StarError::MicrophysicsError(format!("do_eos failed: {e}")))?;

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

    for (k, out) in eos_outputs.into_iter().enumerate() {
        store_eos_for_cell(model, k, &out)?;
    }
    Ok(())
}

/// Validates and writes an [`EosEvalOutput`] into all cell-centered EOS fields
/// of `model` at cell `k` (`store_eos_for_cell` in
/// `star/private/micro.f90:428-545`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if any EOS output or derivative is
/// non-finite, $T$ or $\rho$ exceeds $10^{15}$, or any thermodynamic quantity
/// (`gamma1`, `grada`, `csound`, `chiT`, `chiRho`, `Cv`, `Cp`) is non-positive.
pub(crate) fn store_eos_for_cell(
    model: &mut StarModel,
    k: usize,
    out: &EosEvalOutput,
) -> Result<(), StarError> {
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

    for i in 0..NUM_EOS_BASIC_RESULTS {
        if is_bad(out.res[i] + out.d_dlnd[i] + out.d_dln_t[i]) {
            return Err(StarError::MicrophysicsError(format!(
                "store_eos_for_cell: non-finite EOS result[{i}] at zone {k}"
            )));
        }
    }

    let t_k = model.t[k];
    let rho_k = model.rho[k];
    if t_k > 1.0e15 || rho_k > 1.0e15 {
        return Err(StarError::MicrophysicsError(format!(
            "store_eos_for_cell: T ({t_k}) or rho ({rho_k}) > 1e15 at zone {k}"
        )));
    }

    let ln_pgas = out.res[I_LN_PGAS];
    let pgas = flexmesa_math::exp(ln_pgas);
    // `star/private/micro.f90:464`: `s% Prad(k) = crad * T*T*T*T / 3d0`.
    let prad = ((((CRAD * t_k) * t_k) * t_k) * t_k) / 3.0;
    let peos = prad + pgas;
    let ln_peos = flexmesa_math::log(peos);

    let d_lnpgas_dlnd = out.d_dlnd[I_LN_PGAS];
    let d_lnpgas_dlnt = out.d_dln_t[I_LN_PGAS];
    let d_lnpeos_dlnd = pgas * d_lnpgas_dlnd / peos;
    let d_lnpeos_dlnt = (pgas * d_lnpgas_dlnt + 4.0 * prad) / peos;

    let ln_e = out.res[I_LN_E];
    let energy = flexmesa_math::exp(ln_e);
    let ln_s = out.res[I_LN_S];
    let entropy = flexmesa_math::exp(ln_s);
    let grada = out.res[I_GRAD_AD];
    let cv = out.res[I_CV];
    let cp = out.res[I_CP];
    let chi_rho = out.res[I_CHI_RHO];
    let chi_t = out.res[I_CHI_T];
    let gamma1 = out.res[I_GAMMA1];

    let cs2 = gamma1 * peos / rho_k;
    if cs2 < 0.0 {
        return Err(StarError::MicrophysicsError(format!(
            "eval_csound: negative cs^2 ({cs2}) at zone {k}"
        )));
    }
    let csound = cs2.sqrt();
    if gamma1 <= 0.0
        || grada <= 0.0
        || csound <= 0.0
        || is_bad(csound)
        || chi_t <= 0.0
        || chi_rho <= 0.0
        || cv <= 0.0
        || cp <= 0.0
    {
        return Err(StarError::MicrophysicsError(format!(
            concat!(
                "store_eos_for_cell: non-positive thermodynamic state at zone {} ",
                "(gamma1={}, grada={}, csound={}, chiT={}, chiRho={}, Cv={}, Cp={})"
            ),
            k, gamma1, grada, csound, chi_t, chi_rho, cv, cp
        )));
    }

    // `star/private/micro.f90:480-481`.
    let gam = model.z53bar[k]
        * QE
        * QE
        * flexmesa_math::pow(
            FOUR_THIRDS_PI * AVO * rho_k * model.zbar[k] / model.abar[k],
            ONE_THIRD,
        )
        / (KERG * t_k);

    model.pgas[k] = pgas;
    model.ln_pgas[k] = ln_pgas;
    model.prad[k] = prad;
    model.peos[k] = peos;
    model.ln_peos[k] = ln_peos;
    model.energy[k] = energy;
    model.ln_e[k] = ln_e;
    model.entropy[k] = entropy;
    model.ln_s[k] = ln_s;
    model.grada[k] = grada;
    model.cv[k] = cv;
    model.cp[k] = cp;
    model.chi_rho[k] = chi_rho;
    model.chi_t[k] = chi_t;
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
            pgas * d_lnpgas_dxa / peos;
    }

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
    Ok(())
}

/// Evaluates thermal neutrino loss rates `non_nuc_neu`, per-channel losses
/// (`nonnucneu_{plas,brem,phot,pair,reco}`), and partial derivatives across all
/// `nz` zones (`do_neu_for_cell` in `star/private/neu.f90:49-127`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if `use_other_neu` is active without a
/// hook, or [`StarError::MicrophysicsError`] if `neu_get` fails.
pub fn do_neu(model: &mut StarModel) -> Result<(), StarError> {
    if model.controls.use_other_neu {
        return Err(StarError::InvalidInput(String::from(
            "use_other_neu = .true. requires an other_neu hook",
        )));
    }
    let nz = model.nz;
    if model.aux.nonnucneu_channels.len() != nz {
        model.aux.nonnucneu_channels.resize(nz, [0.0; 5]);
    }
    let min_t = model.controls.min_t_for_neu;
    let factor = model.controls.non_nuc_neu_factor;
    if factor <= 0.0 {
        model.non_nuc_neu.fill(0.0);
        model.d_non_nuc_neu_dlnd.fill(0.0);
        model.d_non_nuc_neu_dlnt.fill(0.0);
        model.aux.nonnucneu_channels.fill([0.0; 5]);
        return Ok(());
    }

    let flags = NeuFlags::all();
    let results: Result<Vec<(f64, f64, f64, [f64; 5])>, StarError> = (0..nz)
        .into_par_iter()
        .map(|k| {
            let t_k = model.t[k];
            if t_k < min_t {
                return Ok((0.0, 0.0, 0.0, [0.0; 5]));
            }
            let rho_k = model.rho[k];
            let log_t = model.ln_t[k] / LN10;
            let log_rho = model.lnd[k] / LN10;
            let NeuResult {
                total,
                pair,
                plas,
                phot,
                brem,
                reco,
                ..
            } = neu_get(
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
            let val = total.val * factor;
            if is_bad(val) {
                return Err(StarError::MicrophysicsError(format!(
                    "do_neu_for_cell: non-finite non_nuc_neu ({val}) at zone {k}"
                )));
            }
            // `star/private/neu.f90:106-115`: `loss(:)` is scaled by
            // `non_nuc_neu_factor` while `sources(:, ineu)` channels remain unscaled.
            Ok((
                val,
                total.d_rho * rho_k * factor,
                total.d_t * t_k * factor,
                [plas.val, brem.val, phot.val, pair.val, reco.val],
            ))
        })
        .collect();

    for (k, (val, dlnd, dlnt, channels)) in results?.into_iter().enumerate() {
        model.non_nuc_neu[k] = val;
        model.d_non_nuc_neu_dlnd[k] = dlnd;
        model.d_non_nuc_neu_dlnt[k] = dlnt;
        model.aux.nonnucneu_channels[k] = channels;
    }
    Ok(())
}

/// Evaluates Rosseland mean opacity `opacity` (radiative + Compton + electron
/// conduction, or simple electron scattering) in parallel across all `nz` zones
/// and integrates optical depth `tau` inward from the surface (`do_kap_for_cell`
/// in `star/private/micro.f90:548-664` and `star/private/kap_support.f90:245-307`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if `use_other_kap` is active without a
/// hook, or [`StarError::MicrophysicsError`] if opacity evaluation fails.
pub fn do_kap(model: &mut StarModel) -> Result<(), StarError> {
    if model.nz > 0 {
        eval_kap_for_range(model, 0, model.nz - 1)?;
    } else if model.controls.use_other_kap {
        return Err(StarError::InvalidInput(String::from(
            "use_other_kap = .true. requires an other_kap hook",
        )));
    }
    get_tau(model);
    Ok(())
}

/// Evaluates `do_kap_for_cell` (`star/private/micro.f90:548-664` and
/// `star/private/kap_support.f90:245-307`) in parallel across cells
/// `k_lo..=k_hi`, updating `opacity`, `d_opacity_dlnd`, `d_opacity_dlnt`, and
/// `aux.kap_frac` in deterministic index order without touching `tau`.
///
/// Shared by [`do_kap`], `predictive_mix::eval_mixing_coeffs`, and
/// `conv_premix::update_model`.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if `use_other_kap` is active without a
/// hook, or [`StarError::MicrophysicsError`] if composition or opacity
/// evaluation fails.
pub(crate) fn eval_kap_for_range(
    model: &mut StarModel,
    k_lo: usize,
    k_hi: usize,
) -> Result<(), StarError> {
    if model.controls.use_other_kap {
        return Err(StarError::InvalidInput(String::from(
            "use_other_kap = .true. requires an other_kap hook",
        )));
    }
    let nz = model.nz;
    if nz == 0 || k_lo > k_hi || k_hi >= nz {
        return Ok(());
    }
    let species = model.species;
    if model.aux.kap_frac.len() != nz {
        model.aux.kap_frac.resize(nz, [0.0; 4]);
    }
    let catalog = NuclideCatalog::builtin_ref();
    let iso_z: Vec<usize> = model
        .chem_id
        .iter()
        .map(|&cid| catalog.get_by_id(cid).map_or(0, |n| n.z))
        .collect();
    let iso_h12: Vec<bool> = model
        .chem_id
        .iter()
        .map(|&cid| {
            catalog.get_by_id(cid).is_some_and(|n| {
                n.z == 1 && (n.z_plus_n == 1 || n.z_plus_n == 2)
            })
        })
        .collect();
    let iso_he34: Vec<bool> = model
        .chem_id
        .iter()
        .map(|&cid| {
            catalog.get_by_id(cid).is_some_and(|n| {
                n.z == 2 && (n.z_plus_n == 3 || n.z_plus_n == 4)
            })
        })
        .collect();
    let base_kap_factor = model.controls.opacity_factor;
    let use_simple_es = model.controls.use_simple_es_for_kap;
    // `micro.f90:589-595` reads `xa_start` and `zbar_start` whenever the control
    // is set, with no fallback to the current composition.
    let use_start_comp = model.controls.use_starting_composition_for_kap;
    if use_start_comp && model.xa_start.len() != nz * species {
        return Err(StarError::InvalidInput(format!(
            "use_starting_composition_for_kap = .true. but xa_start holds {} \
             values where nz * species = {}",
            model.xa_start.len(),
            nz * species
        )));
    }
    let opacity_max = model.controls.opacity_max;
    let opacity_min = model.controls.opacity_min;
    let min_off = model.controls.rsp.min_logt_for_opacity_factor_off;
    let min_on = model.controls.rsp.min_logt_for_opacity_factor_on;
    let max_on = model.controls.rsp.max_logt_for_opacity_factor_on;
    let max_off = model.controls.rsp.max_logt_for_opacity_factor_off;
    let kap_cfg = &model.kap_config;
    let chem_id = &model.chem_id;

    let kap_results: Result<Vec<(f64, f64, f64, [f64; 4])>, StarError> =
        (k_lo..=k_hi)
            .into_par_iter()
            .map(|k| {
                let (xa_k, zbar_k, xh_k, xhe_k, z_k) = if use_start_comp {
                    let xs = &model.xa_start[k * species..(k + 1) * species];
                    let comp = get_composition_info(catalog, chem_id, xs, true)
                        .map_err(|e| {
                            StarError::MicrophysicsError(format!(
                                "get_kap starting composition error at zone {k}: {e}"
                            ))
                        })?;
                    let xh_s = comp.xh.clamp(0.0, 1.0);
                    let xhe_s = comp.xhe.clamp(0.0, 1.0);
                    let z_s = (1.0 - (xh_s + xhe_s)).clamp(0.0, 1.0);
                    (xs, comp.zbar, xh_s, xhe_s, z_s)
                } else {
                    // `star/private/kap_support.f90:250` + `star_utils.f90:2029-2043`
                    // (`get_XYZ` directly from `s% xa(:, k)` so partial EOS
                    // updates in `predictive_mix` see trial `xa_mx` without
                    // requiring `xh_mass`/`yhe_mass`/`z_mass` to be overwritten).
                    let xa_slice = &model.xa[k * species..(k + 1) * species];
                    let mut xh_s = 0.0;
                    let mut xhe_s = 0.0;
                    for (j, &xj) in xa_slice.iter().enumerate() {
                        if iso_h12[j] {
                            xh_s += xj;
                        } else if iso_he34[j] {
                            xhe_s += xj;
                        }
                    }
                    let xh_c = xh_s.clamp(0.0, 1.0);
                    let xhe_c = xhe_s.clamp(0.0, 1.0);
                    let z_c = (1.0 - (xh_c + xhe_c)).clamp(0.0, 1.0);
                    (xa_slice, model.zbar[k], xh_c, xhe_c, z_c)
                };

                let (mut xc, mut xn, mut xo, mut xne) = (0.0, 0.0, 0.0, 0.0);
                for (&z, &xj) in iso_z.iter().zip(xa_k.iter()) {
                    match z {
                        6 => xc += xj,
                        7 => xn += xj,
                        8 => xo += xj,
                        10 => xne += xj,
                        _ => {}
                    }
                }
                // `star/private/kap_support.f90:251,276-288`: `eps = 1d-6`.
                const EPS: f64 = 1.0e-6;
                if !(-EPS..=1.0 + EPS).contains(&xc)
                    || !(-EPS..=1.0 + EPS).contains(&xn)
                    || !(-EPS..=1.0 + EPS).contains(&xo)
                    || !(-EPS..=1.0 + EPS).contains(&xne)
                    || xh_k < 0.0
                    || xhe_k < 0.0
                    || z_k < 0.0
                {
                    return Err(StarError::MicrophysicsError(format!(
                        concat!(
                            "get_kap: invalid mass fractions at zone {} ",
                            "(xh={}, xhe={}, z={}, xc={}, xn={}, xo={}, xne={})"
                        ),
                        k, xh_k, xhe_k, z_k, xc, xn, xo, xne
                    )));
                }

                let log_rho = model.lnd[k] / LN10;
                let log_t = model.ln_t[k] / LN10;
                let (kap_raw, dlnkap_dlnd, dlnkap_dlnt, kap_fracs) =
                    if use_simple_es {
                        (0.2 * (1.0 + xh_k), 0.0, 0.0, [0.0; 4])
                    } else {
                        let out = get_kap_results(
                            kap_cfg,
                            zbar_k,
                            xh_k,
                            z_k,
                            xc,
                            xn,
                            xo,
                            xne,
                            log_rho,
                            log_t,
                            model.lnfree_e[k],
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
                        (
                            out.kap,
                            out.dlnkap_dln_rho,
                            out.dlnkap_dln_t,
                            out.kap_fracs,
                        )
                    };

                if is_bad(kap_raw) {
                    return Err(StarError::MicrophysicsError(format!(
                        "do_kap_for_cell: non-finite opacity ({kap_raw}) at zone {k}"
                    )));
                }

                // `star/private/micro.f90:622-652`: piecewise-linear `log10_T` ramp
                // on `opacity_factor` (held constant in `d_opacity_dlnT`) followed
                // by `opacity_max` and `opacity_min` clamps.
                let mut opacity_factor = base_kap_factor;
                if min_off > 0.0 {
                    if log_t >= max_off || log_t <= min_off {
                        opacity_factor = 1.0;
                    } else if log_t > max_on {
                        opacity_factor = 1.0
                            + (opacity_factor - 1.0) * (log_t - max_off)
                                / (max_on - max_off);
                    } else if log_t < min_on {
                        opacity_factor = 1.0
                            + (opacity_factor - 1.0) * (log_t - min_off)
                                / (min_on - min_off);
                    }
                }

                let mut kap = kap_raw * opacity_factor;
                let mut dlnd = kap * dlnkap_dlnd;
                let mut dlnt = kap * dlnkap_dlnt;
                if opacity_max > 0.0 && kap > opacity_max {
                    kap = opacity_max;
                    dlnd = 0.0;
                    dlnt = 0.0;
                }
                if opacity_min > 0.0 && kap < opacity_min {
                    kap = opacity_min;
                    dlnd = 0.0;
                    dlnt = 0.0;
                }
                if is_bad(kap) {
                    return Err(StarError::MicrophysicsError(format!(
                        "do_kap_for_cell: non-finite scaled opacity ({kap}) at zone {k}"
                    )));
                }
                Ok((kap, dlnd, dlnt, kap_fracs))
            })
            .collect();

    for (offset, (kap, dlnd, dlnt, fracs)) in
        kap_results?.into_iter().enumerate()
    {
        let k = k_lo + offset;
        model.opacity[k] = kap;
        model.d_opacity_dlnd[k] = dlnd;
        model.d_opacity_dlnt[k] = dlnt;
        model.aux.kap_frac[k] = fracs;
    }

    Ok(())
}

/// Integrates optical depth $\tau_k$ inward from $\tau_0 = \tau_{\mathrm{factor}} \tau_{\mathrm{base}}$
/// (`get_tau` in `star/private/star_utils.f90:871-902`).
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
