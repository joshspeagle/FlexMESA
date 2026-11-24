// SPDX-License-Identifier: GPL-3.0-or-later
//! The subset of the public star interface (`star/public/star_lib.f90`) that
//! test-side `run_star_extras` ports use to build a starting model cell by
//! cell, as `make_env`'s `create_env` does
//! (`star/test_suite/make_env/src/run_star_extras.f90:158-568`).
//!
//! Each function wraps the same private routine its `star_lib` counterpart
//! calls. Cell indices `k` are 0-based (upstream's `k - 1`).

use std::path::Path;

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_const::LN10;
use flexmesa_eos::{eos_dt_get_rho, EosEvalOutput, I_LN_PGAS};
use flexmesa_math::{exp10, log10};

use crate::controls::{StarControls, StarError, StarJobControls};
use crate::model::StarModel;

/// `arg_not_provided` (`const/public/const_def.f90`).
const ARG_NOT_PROVIDED: f64 = -9.0e99;
/// `MAX_ITER_FOR_SOLVE` (`star/private/eos_support.f90:38`).
const MAX_ITER_FOR_SOLVE: usize = 100;

/// `star_set_net`, `star_set_var_info`, `star_set_chem_names` and
/// `star_allocate_arrays` (`star/public/star_lib.f90:301-345`) on a new star:
/// a model of `nz` cells on network `net_name`, with the starting values
/// `set_starting_star_data` gives the fields a builder reads before setting
/// them (`star/private/init.f90:428-775`).
///
/// `search_dir` is searched for `net_name` before the built-in networks
/// (`net/private/net_initialize.f90:286-302`).
///
/// # Errors
/// Returns [`StarError`] when the network cannot be loaded.
pub fn star_new_model_for_build(
    net_name: &str,
    nz: usize,
    controls: StarControls,
    mut job: StarJobControls,
    search_dir: Option<&Path>,
) -> Result<StarModel, StarError> {
    // `star_set_net` sets `s% net_name` (`star/public/star_lib.f90:309`).
    let net = crate::micro::get_or_load_net(net_name, search_dir)?;
    job.net_name = net_name.to_string();
    let mut model = StarModel::new(nz, net.chem_id.clone(), job, controls);
    // `set_starting_star_data` (`star/private/init.f90:486, 541-542, 554-560,
    // 676, 686, 770-771`).
    model.model_number = 0;
    model.time_sec = 0.0;
    model.star_age = 0.0;
    model.dt = 0.0;
    model.dt_next = -1.0;
    model.v_center = 0.0;
    model.r_center = 0.0;
    model.l_center = 0.0;
    model.m_center = 0.0;
    model.aux.okay_to_set_mlt_vc = false;
    model.aux.have_mlt_vc = false;
    model.aux.crystal_core_boundary_mass = -1.0;
    model.aux.phase_sep_mixing_mass = -1.0;
    model.aux.mixing.resize(nz);
    Ok(model)
}

/// `star_normalize_dqs` (`star/public/star_lib.f90:2355-2367`).
///
/// # Errors
/// Returns [`StarError`] from `normalize_dqs`.
pub fn star_normalize_dqs(model: &mut StarModel) -> Result<(), StarError> {
    model.normalize_dqs()
}

/// `star_set_qs` (`star/public/star_lib.f90:2370-2381`).
pub fn star_set_qs(model: &mut StarModel) {
    model.set_qs();
}

/// `star_set_m_and_dm` (`star/public/star_lib.f90:2384-2393`).
pub fn star_set_m_and_dm(model: &mut StarModel) {
    model.set_m_and_dm();
}

/// `star_set_dm_bar` (`star/public/star_lib.f90:2396-2405`).
pub fn star_set_dm_bar(model: &mut StarModel) {
    model.set_dm_bar();
}

/// `change_to_xa_for_accretion` (`star/public/star_lib.f90:1038-1044`,
/// `do_change_to_xa_for_accretion` in `star/private/adjust_xyz.f90:950-987`)
/// for the 0-based cells `k_lo..=k_hi`.
///
/// # Errors
/// Returns [`StarError`] from `get_xa_for_accretion`.
pub fn change_to_xa_for_accretion(
    model: &mut StarModel,
    k_lo: usize,
    k_hi: usize,
) -> Result<(), StarError> {
    let xa = crate::accretion_xa::get_xa_for_accretion(model)?;
    let hi = k_hi.min(model.nz.saturating_sub(1));
    for k in k_lo..=hi {
        model.cell_xa_mut(k).copy_from_slice(&xa);
    }
    model.aux.need_to_setvars = true;
    Ok(())
}

/// `basic_composition_info` (`chem/public/chem_lib.f90`) for cell `k`,
/// storing `X`, `Y`, `Z`, `abar`, `zbar`, `z2bar`, `z53bar` and `ye`.
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] for an unknown isotope.
pub fn star_basic_composition_info(
    model: &mut StarModel,
    k: usize,
) -> Result<(), StarError> {
    let catalog = NuclideCatalog::builtin_ref();
    let comp =
        get_composition_info(catalog, &model.chem_id, model.cell_xa(k), true)
            .map_err(|e| {
            StarError::MicrophysicsError(format!(
                "basic_composition_info failed at cell {k}: {e}"
            ))
        })?;
    model.xh_mass[k] = comp.xh;
    model.yhe_mass[k] = comp.xhe;
    model.z_mass[k] = comp.xz;
    model.abar[k] = comp.abar;
    model.zbar[k] = comp.zbar;
    model.z2bar[k] = comp.z2bar;
    model.z53bar[k] = comp.z53bar;
    model.ye[k] = comp.ye;
    Ok(())
}

/// `star_get_atm_PT` (`star/public/star_lib.f90:2223-2247`, `get_atm_PT` in
/// `star/private/atm_support.f90:37-147`) for `atm_option = 'T_tau'`:
/// returns $(\ln T_{\rm surf}, \ln P_{\rm surf})$.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] for another `atm_option`, and the
/// atmosphere error otherwise.
#[allow(clippy::too_many_arguments)]
pub fn star_get_atm_pt(
    model: &StarModel,
    tau_surf: f64,
    l: f64,
    r: f64,
    m: f64,
    cgrav: f64,
    teff: f64,
) -> Result<(f64, f64), StarError> {
    if model.controls.atm_option != crate::controls::AtmOption::TTau {
        return Err(StarError::InvalidInput(String::from(
            "star_get_atm_PT is available for atm_option = 'T_tau' only \
             (star/private/atm_support.f90:37-147)",
        )));
    }
    let pt = crate::hydro_brunt_surf::eval_grey_t_tau_surf(
        model, tau_surf, l, r, m, cgrav, teff,
    )?;
    Ok((pt.ln_t, pt.ln_p))
}

/// `star_get_surf_PT` with `need_atm_Psurf = need_atm_Tsurf = .true.`
/// (`star/public/star_lib.f90:2250-2270`, `get_surf_PT` in
/// `star/private/hydro_vars.f90:779-1030`) at `s% Teff`: returns
/// $(\ln T_{\rm surf}, \ln P_{\rm surf})$.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] when the controls do not ask for the
/// atmosphere's `P_surf` and `T_surf`, and the atmosphere error otherwise.
pub fn star_get_surf_pt(
    model: &mut StarModel,
) -> Result<(f64, f64), StarError> {
    let c = &model.controls;
    if c.use_compression_outer_bc
        || c.use_zero_pgas_outer_bc
        || c.use_fixed_psurf_outer_bc
        || c.use_fixed_vsurf_outer_bc
        || c.rsp2_flag
        || c.use_rsp_l_eqn_outer_bc
    {
        return Err(StarError::InvalidInput(String::from(
            "star_get_surf_PT with need_atm_Psurf and need_atm_Tsurf needs \
             the atmosphere outer boundary condition",
        )));
    }
    let teff = model.teff;
    crate::hydro_brunt_surf::get_surf_pt(model, teff)?;
    Ok((model.surf_pt.ln_t, model.surf_pt.ln_p))
}

/// `star_solve_eos_given_PgasT` (`star/public/star_lib.f90:2114-2139`,
/// `solve_eos_given_PgasT` in `star/private/eos_support.f90:315-358`):
/// returns $\log_{10}\rho$ and the EOS results at it.
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] when the root solve fails.
#[allow(clippy::too_many_arguments)]
pub fn star_solve_eos_given_pgas_t(
    model: &StarModel,
    xa: &[f64],
    log_t: f64,
    log_pgas: f64,
    log_rho_guess: f64,
    log_rho_tol: f64,
    log_pgas_tol: f64,
) -> Result<(f64, EosEvalOutput), StarError> {
    let ht = crate::micro::builtin_helm_table()?;
    let inv = eos_dt_get_rho(
        ht,
        &model.chem_id,
        xa,
        log_t,
        I_LN_PGAS as isize,
        log_pgas * LN10,
        log_rho_guess,
        log_rho_tol,
        log_pgas_tol * LN10,
        MAX_ITER_FOR_SOLVE,
        &model.eos_config,
    )
    .map_err(|e| {
        StarError::MicrophysicsError(format!(
            "eosDT_get_Rho failed in solve_eos_given_PgasT (logT = {log_t}, \
             logPgas = {log_pgas}, logRho_guess = {log_rho_guess}): {e}"
        ))
    })?;
    Ok((inv.log10_result, inv.eos))
}

/// `star_solve_eos_given_PgasT_auto` (`star/public/star_lib.f90:2142-2165`,
/// `solve_eos_given_PgasT_auto` in `star/private/eos_support.f90:365-411`):
/// the $\gamma = 5/3$ ideal-gas density guess of `eos_gamma_PT_get`
/// (`eos/public/eos_lib.f90:380-389`), then [`star_solve_eos_given_pgas_t`].
///
/// # Errors
/// Returns [`StarError`] for an unknown isotope or a failed root solve.
pub fn star_solve_eos_given_pgas_t_auto(
    model: &StarModel,
    xa: &[f64],
    log_t: f64,
    log_pgas: f64,
    log_rho_tol: f64,
    log_pgas_tol: f64,
) -> Result<(f64, EosEvalOutput), StarError> {
    let catalog = NuclideCatalog::builtin_ref();
    let comp = get_composition_info(catalog, &model.chem_id, xa, true)
        .map_err(|e| {
            StarError::MicrophysicsError(format!(
                "basic_composition_info failed in solve_eos_given_PgasT_auto: \
                 {e}"
            ))
        })?;
    let p = exp10(log_pgas);
    let t = exp10(log_t);
    // `eos_gamma_PT_get` fails only for `P <= 0` or `T <= 0`
    // (`star/private/eos_support.f90:399-402`).
    let log_rho_guess = if p <= 0.0 || t <= 0.0 {
        ARG_NOT_PROVIDED
    } else {
        let rho =
            (p / t) * comp.abar / (flexmesa_const::AVO * flexmesa_const::KERG);
        log10(rho)
    };
    star_solve_eos_given_pgas_t(
        model,
        xa,
        log_t,
        log_pgas,
        log_rho_guess,
        log_rho_tol,
        log_pgas_tol,
    )
}

/// `star_do_eos_for_cell` (`star/public/star_lib.f90:2197-2207`,
/// `do_eos_for_cell` in `star/private/micro.f90:292-413`) from `lnd(k)`,
/// `lnT(k)`, `rho(k)` and `T(k)`.
///
/// # Errors
/// Returns [`StarError`] from the composition, EOS or store step.
pub fn star_do_eos_for_cell(
    model: &mut StarModel,
    k: usize,
) -> Result<(), StarError> {
    star_basic_composition_info(model, k)?;
    let log_t = model.ln_t[k] / LN10;
    let log_rho = model.lnd[k] / LN10;
    let ctx = crate::micro::EosZoneContext::new(&model.chem_id)?;
    let out = ctx.eval_cell(
        k,
        model.cell_xa(k),
        model.rho[k],
        log_rho,
        model.t[k],
        log_t,
        &model.eos_config,
    )?;
    crate::micro::store_eos_for_cell(model, k, &out)
}

/// `star_do_kap_for_cell` (`star/public/star_lib.f90:2210-2220`,
/// `do_kap_for_cell` in `star/private/micro.f90:548-754`). The opacity
/// factor is `opacity_factor`, which `set_hydro_vars` copies into every
/// `extra_opacity_factor(k)`.
///
/// # Errors
/// Returns [`StarError`] from the opacity evaluation.
pub fn star_do_kap_for_cell(
    model: &mut StarModel,
    k: usize,
) -> Result<(), StarError> {
    crate::micro::eval_kap_for_range(model, k, k)
}

/// `star_set_mlt_vars` (`star/public/star_lib.f90:3064-3074`,
/// `set_mlt_vars` in `star/private/turb_info.f90:40-67`) for the 0-based
/// faces `k_lo..=k_hi`.
///
/// # Errors
/// Returns [`StarError`] from the MLT evaluation.
pub fn star_set_mlt_vars(
    model: &mut StarModel,
    k_lo: usize,
    k_hi: usize,
) -> Result<(), StarError> {
    crate::hydro_vars::set_mlt_vars_range(model, k_lo, k_hi)
}

/// `star_write_model` (`star/public/star_lib.f90:483-489`, `do_write_model`
/// in `star/private/write_model.f90`).
///
/// # Errors
/// Returns [`StarError::IoError`] when the file cannot be written.
pub fn star_write_model(
    model: &StarModel,
    path: &Path,
) -> Result<(), StarError> {
    let text = crate::model_io::write_mod_string(model)?;
    std::fs::write(path, text).map_err(|e| {
        StarError::IoError(format!(
            "star_write_model failed for {}: {e}",
            path.display()
        ))
    })
}
