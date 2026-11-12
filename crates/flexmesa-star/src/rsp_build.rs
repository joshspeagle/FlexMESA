// SPDX-License-Identifier: LGPL-3.0-or-later
//! Static nonlinear Radial Stellar Pulsation (`RSP`) envelope builder
//! (`build_rsp_model` in `star/private/rsp.f90:70-217` and `do_rsp_build` in
//! `star/private/rsp_build.f90:46-298`) and linear non-adiabatic (`LINA`)
//! radial pulsation mode driver (`star/private/rsp_lina.f90`, Issue `#5`).

use std::path::Path;

use flexmesa_const::{M_SUN, SECYER};
use flexmesa_eos::EosConfig;
use flexmesa_kap::{KapCoOption, KapConfig, KapLowTOption, KapOption};

use crate::controls::{StarControls, StarError, StarJobControls};
use crate::micro::get_or_load_net;
use crate::model::StarModel;
use crate::rsp_def::{RspDerivedConsts, MAX_RSP_NZN};
use crate::rsp_eos_kap::RspChemCache;
use crate::rsp_grid::{cleanup_for_lina, finish_build_rsp_model, MesaRanmar};
use crate::rsp_lina::{do_lina, RspLinaOutput};
use crate::rsp_relax_env::relax_env;
use crate::rsp_stah::stah;
use crate::zams::load_zams_model;

pub use crate::rsp_crossing::{
    eval_rsp_2nd_crossing_edges, eval_rsp_2nd_crossing_with_controls,
    Rsp2ndCrossingOffset, Rsp2ndCrossingPoint, Rsp2ndCrossingReport,
};

/// Linear non-adiabatic (`LINA`) radial pulsation eigenmode returned by
/// [`eval_rsp_lina_modes`] (`star/private/rsp_lina.f90`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RspLinaMode {
    /// Radial overtone index (`0` = fundamental F, `1` = first overtone 1O,
    /// `2` = second overtone 2O).
    pub mode_index: usize,
    /// Linear non-adiabatic period $P_i$ in days.
    pub period_days: f64,
    /// Fractional amplitude growth rate per cycle
    /// $\eta_i = 4\pi\,\omega_{\mathrm{imag}} / \omega_{\mathrm{real}}$.
    pub growth_rate: f64,
}

/// Builds a complete static `RSP` envelope model via `STAH`, optional
/// `RELAX_ENV`, `cleanup_for_LINA`, optional `do_LINA`, velocity kick, and
/// `finish_build_rsp_model` (`rsp.f90:70-217`, `rsp_build.f90:46-298`).
///
/// # Errors
/// Returns [`StarError`] if network loading, `STAH`, `RELAX_ENV`, or `do_LINA` fails.
pub fn do_rsp_build_model(
    mut controls: StarControls,
    job: StarJobControls,
    eos_config: EosConfig,
    kap_config: KapConfig,
    log_dir: Option<&Path>,
) -> Result<(StarModel, Option<RspLinaOutput>), StarError> {
    controls.rsp_flag = true;
    controls.v_flag = true;
    let net_name = if job.change_net && !job.new_net_name.trim().is_empty() {
        job.new_net_name.trim()
    } else {
        "o18_and_ne22.net"
    };
    let net = get_or_load_net(net_name, None)?;
    let nzn_init = controls.rsp.nz.clamp(4, MAX_RSP_NZN - 1);
    let mut model =
        StarModel::new(nzn_init, net.chem_id.clone(), job, controls);
    model.eos_config = eos_config;
    model.kap_config = kap_config;
    model.controls.tau_factor =
        if model.controls.rsp.use_atm_grey_with_kap_for_psurf {
            model.controls.rsp.tau_surf_for_atm_grey_with_kap / model.tau_base
        } else {
            model.controls.rsp.surface_tau / model.tau_base
        };

    let dc_init = RspDerivedConsts::from_config(&model.controls.rsp, nzn_init);
    let chem = RspChemCache::from_controls(
        &model.chem_id,
        &model.job,
        &model.controls,
    )?;
    let mut rng = MesaRanmar::new();

    let mx = model.controls.rsp_mass * dc_init.sunm;
    let l_surf = model.controls.rsp_l * dc_init.sunl;
    let te = model.controls.rsp_teff;
    let th0 = model.controls.rsp.t_anchor;
    let tin = model.controls.rsp.t_inner;
    let nzt = model.controls.rsp.nz_outer;

    let mut grid = stah(
        &mut model, &chem, &dc_init, mx, l_surf, te, th0, tin, nzt, nzn_init,
    )?;
    let nzn = grid.nzn;
    let dc = RspDerivedConsts::from_config(&model.controls.rsp, nzn);

    if model.controls.rsp.relax_initial_model {
        relax_env(
            &mut model, &chem, &dc, &mut rng, &mut grid, l_surf, th0, te, nzt,
        )?;
    }

    cleanup_for_lina(&mut model, &dc, &mut grid);

    let nmodes = model.controls.rsp.nmodes;
    let lina_opt = if nmodes > 0
        && !model.controls.rsp.use_rsp_new_start_scheme
        && !model.controls.rsp.use_other_rsp_linear_analysis
    {
        let lina =
            do_lina(&mut model, &chem, &dc, &grid, l_surf, nmodes, log_dir)?;
        for m in 0..nmodes.min(3) {
            model.aux.rsp.lina_periods[m] = lina.periods_sec[m];
            model.aux.rsp.lina_growth_rates[m] = lina.growth_rates[m];
        }
        model.aux.rsp.have_set_velocities = true;
        Some(lina)
    } else {
        None
    };

    model.aux.rsp.iwork = 0;
    model.teff = te;
    model.v.fill(0.0);
    model.v_center = 0.0;
    model.aux.rsp.period = model.controls.rsp.default_periodlin;

    if !model.controls.rsp.use_rsp_new_start_scheme {
        let mode_idx = model.controls.rsp.mode_for_setting_periodlin;
        let p_lin = lina_opt
            .as_ref()
            .and_then(|lo| lo.periods_sec.get(mode_idx).copied())
            .unwrap_or(0.0);
        model.aux.rsp.period_lin = p_lin;
        model.aux.rsp.period = p_lin;
        let amix1 = model.controls.rsp.fraction_1st_overtone;
        let amix2 = model.controls.rsp.fraction_2nd_overtone;
        let kick = 1.0e5 * model.controls.rsp.kick_vsurf_km_per_sec;
        if let Some(ref lo) = lina_opt {
            for i in 1..=nzn {
                let v0 = lo.vel_modes.first().map_or(0.0, |v| v[i]);
                let v1 = lo.vel_modes.get(1).map_or(0.0, |v| v[i]);
                let v2 = lo.vel_modes.get(2).map_or(0.0, |v| v[i]);
                model.v[nzn - i] = kick
                    * ((1.0 - amix1 - amix2) * v0 + amix1 * v1 + amix2 * v2);
            }
        }
        if model.controls.rsp.avel != 0.0 || model.controls.rsp.arnd != 0.0 {
            for i in 1..=nzn {
                let k = nzn - i;
                let rv = rng.next_f64();
                model.v[k] += 1.0e5 * (i as f64) / (nzn as f64)
                    * (model.controls.rsp.avel
                        + model.controls.rsp.arnd * (2.0 * rv - 1.0));
            }
            model.aux.rsp.have_set_velocities = true;
        }
    }

    model.aux.rsp.period_lin = model.aux.rsp.period;
    model.aux.rsp.rsp_dt = if model.aux.rsp.period > 0.0
        && model.controls.rsp_target_steps_per_cycle > 0
    {
        model.aux.rsp.period
            / (model.controls.rsp_target_steps_per_cycle as f64)
    } else {
        1.0
    };
    model.dt = model.aux.rsp.rsp_dt;
    model.dt_next = model.aux.rsp.rsp_dt;

    finish_build_rsp_model(&mut model, &chem, &dc, &grid, l_surf);
    model.mstar_start = model.mstar;
    model.mstar_old = model.mstar;
    model.save_start_values();
    model.lum_start.fill(0.0);

    Ok((model, lina_opt))
}

/// Evaluates the linear non-adiabatic (`LINA`) radial pulsation periods $P_i$
/// (days) and $\kappa$-mechanism growth rates $\eta_i$ for a stellar envelope
/// with $(M/M_\odot, L/L_\odot, T_{\mathrm{eff}}, X, Z)$ by running `STAH` +
/// `RELAX_ENV` + `do_LINA` (`star/private/rsp_lina.f90`).
///
/// # Errors
/// Returns [`StarError`] if static envelope construction or `do_LINA` fails.
pub fn eval_rsp_lina_modes(
    m_msun: f64,
    l_lsun: f64,
    teff_k: f64,
    x_h: f64,
    z_metal: f64,
    n_modes: usize,
) -> Result<Vec<RspLinaMode>, StarError> {
    let count = n_modes.clamp(1, 15);
    let mut controls = StarControls {
        rsp_mass: m_msun,
        rsp_l: l_lsun,
        rsp_teff: teff_k,
        rsp_x: x_h,
        rsp_z: z_metal,
        zbase: z_metal,
        ..Default::default()
    };
    controls.rsp.nmodes = count;
    let job = StarJobControls {
        create_rsp_model: true,
        initial_zfracs: 6,
        ..Default::default()
    };
    let eos_config = EosConfig::default();
    let kap_config = KapConfig {
        kap_option: KapOption::A09,
        kap_co_option: KapCoOption::A09Co,
        kap_low_t_option: KapLowTOption::LowTFa05A09p,
        zbase: z_metal,
        ..Default::default()
    };

    let (_model, lina_opt) =
        do_rsp_build_model(controls, job, eos_config, kap_config, None)?;
    let lina = lina_opt.ok_or_else(|| {
        StarError::InvalidInput(String::from("do_LINA produced no output"))
    })?;
    let mut modes = Vec::with_capacity(count);
    for idx in 0..count {
        modes.push(RspLinaMode {
            mode_index: idx,
            period_days: lina.periods_sec[idx] / 86_400.0,
            growth_rate: lina.growth_rates[idx],
        });
    }
    Ok(modes)
}

/// Evaluates `LINA` radial pulsation modes and optional nonlinear target period
/// (`x_ctrl(1)`) from `model` without passing any target period into `LINA`.
#[must_use]
pub fn eval_model_rsp_lina_modes(
    model: &StarModel,
    n_modes: usize,
) -> (Vec<RspLinaMode>, Option<f64>) {
    let target_p = model.controls.x_ctrl.get(1).copied().filter(|&p| p > 0.0);
    let count = n_modes.clamp(1, 3);
    if model.aux.rsp.lina_periods[0] > 0.0 {
        let mut cached = Vec::with_capacity(count);
        for idx in 0..count {
            if model.aux.rsp.lina_periods[idx] > 0.0 {
                cached.push(RspLinaMode {
                    mode_index: idx,
                    period_days: model.aux.rsp.lina_periods[idx] / 86_400.0,
                    growth_rate: model.aux.rsp.lina_growth_rates[idx],
                });
            }
        }
        if !cached.is_empty() {
            return (cached, target_p);
        }
    }

    let mut ctrl = model.controls.clone();
    ctrl.rsp.nmodes = count;
    ctrl.rsp.use_other_rsp_build_model = false;
    ctrl.rsp.use_other_rsp_linear_analysis = false;
    ctrl.rsp.use_rsp_new_start_scheme = false;
    let mut job = model.job.clone();
    job.create_rsp_model = true;

    if let Ok((_built, Some(lina))) = do_rsp_build_model(
        ctrl,
        job,
        model.eos_config,
        model.kap_config.clone(),
        None,
    ) {
        let mut out = Vec::with_capacity(count);
        for idx in 0..count {
            out.push(RspLinaMode {
                mode_index: idx,
                period_days: lina.periods_sec[idx] / 86_400.0,
                growth_rate: lina.growth_rates[idx],
            });
        }
        return (out, target_p);
    }

    let fallback = eval_rsp_lina_modes(
        model.controls.rsp_mass.max(model.controls.initial_mass),
        model.controls.rsp_l.max(10.0),
        model.controls.rsp_teff.max(model.teff).max(3000.0),
        model.controls.rsp_x.max(0.70),
        model.controls.rsp_z.max(0.0001),
        count,
    )
    .unwrap_or_default();
    (fallback, target_p)
}

/// Builds a temporary Henyey-compatible envelope model for `create_RSP_model`
/// until Commit 4 wires `rsp_one_step` (`HYD`) into `star_evolve_step`.
///
/// # Errors
/// Returns [`StarError`] if base model loading fails.
pub fn build_rsp_envelope_model(
    mut controls: StarControls,
    job: StarJobControls,
) -> Result<StarModel, StarError> {
    let rsp_m = if controls.rsp_mass > 0.0 {
        controls.rsp_mass
    } else {
        controls.initial_mass.max(0.25)
    };
    let rsp_teff = if controls.rsp_teff > 0.0 {
        controls.rsp_teff
    } else {
        6000.0
    };
    let mut base_ctrl = controls.clone();
    base_ctrl.initial_mass = 1.0;
    let mut model = load_zams_model(base_ctrl, job.clone())?;
    controls.initial_mass = rsp_m;
    controls.rsp_flag = true;
    model.controls = controls;
    model.job = job;

    let old_mstar = model.mstar.max(1.0);
    let new_mstar = rsp_m.max(1.0e-5) * M_SUN;
    let dln_r = (new_mstar / old_mstar).ln() / 3.0;
    let l_scale = new_mstar / old_mstar;
    for k in 0..model.nz {
        model.ln_r[k] += dln_r;
        model.r[k] = model.ln_r[k].exp();
        model.lum[k] *= l_scale;
    }
    model.mstar = new_mstar;
    model.xmstar = (new_mstar - model.m_center).max(0.0);
    model.mstar_start = model.mstar;
    model.mstar_old = model.mstar;
    model.controls.op_split_burn = true;
    model.controls.min_logt_for_net = 8.5;
    crate::star_job_init::equilibrate_tier2_scaled_model(&mut model);
    model.sync_mesh_and_pack()?;
    crate::hydro_vars::set_hydro_vars(&mut model, false, false)?;
    model.save_start_values();
    model.teff = rsp_teff;
    let p_sec = model.controls.rsp.default_periodlin.max(60.0);
    model.dt = (p_sec / 40.0).min(0.1 * SECYER);
    model.dt_next = model.dt;
    Ok(model)
}
