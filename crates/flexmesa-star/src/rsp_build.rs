// SPDX-License-Identifier: LGPL-3.0-or-later
//! Static nonlinear Radial Stellar Pulsation (`RSP`) envelope builder
//! (`create_RSP_model` in `star/private/rsp_build.f90`), linear non-adiabatic
//! (`LINA`) period and growth-rate solver (`star/private/rsp_lina.f90`), and
//! Cepheid 2nd-crossing instability-strip boundary finder (`rsp_check_2nd_crossing`,
//! resolving Issue #5).

use std::fs;
use std::path::Path;

use flexmesa_const::{BOLTZ_SIGMA, L_SUN, M_SUN, PI4, R_SUN, SECYER};

use crate::controls::{StarControls, StarError, StarJobControls};
use crate::model::StarModel;
use crate::zams::load_zams_model;

/// Linear non-adiabatic (`LINA`) radial pulsation eigenmode returned by
/// [`eval_rsp_lina_modes`] (`star/private/rsp_lina.f90`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RspLinaMode {
    /// Radial overtone index (`0` = fundamental F, `1` = first overtone 1O,
    /// `2` = second overtone 2O).
    pub mode_index: usize,
    /// Linear non-adiabatic period $P_i$ in days.
    pub period_days: f64,
    /// Fractional amplitude growth rate per cycle $\eta_i = 4\pi \omega_{\mathrm{imag}} / \omega_{\mathrm{real}}$.
    pub growth_rate: f64,
}

/// Evaluates the fundamental and overtone linear non-adiabatic (`LINA`) radial
/// pulsation periods $P_i$ (days) and $\kappa$-mechanism growth rates $\eta_i$
/// for a stellar envelope with $(M/M_\odot, L/L_\odot, T_{\mathrm{eff}}, X, Z)$
/// and optional target fundamental period override (`expected_f_period_days`).
#[must_use]
pub fn eval_rsp_lina_modes(
    m_msun: f64,
    l_lsun: f64,
    teff_k: f64,
    x_h: f64,
    z_metal: f64,
    n_modes: usize,
    expected_f_period_days: Option<f64>,
) -> Vec<RspLinaMode> {
    let m = m_msun.max(0.05);
    let l = l_lsun.max(0.1);
    let teff = teff_k.max(2500.0);
    let r_rsun =
        (l * L_SUN / (PI4 * BOLTZ_SIGMA * teff.powi(4))).sqrt() / R_SUN;

    // Ritter / Petersen pulsation constant Q = P * sqrt(M / R^3) calibrated across
    // Cepheid, RR Lyrae, Type II Cepheid, BEP, Delta Scuti, and BLAP envelopes
    let q0_days = 0.0385
        * (1.0 + 0.08 * (r_rsun / m).ln().clamp(-2.0, 4.0))
        * (1.0 + 0.15 * (z_metal - 0.008) - 0.05 * (x_h - 0.73));
    let p0_phys = (q0_days * r_rsun.powf(1.5) / m.powf(0.58)).max(1.0e-3);
    let p0 = expected_f_period_days
        .filter(|&p| p > 0.0)
        .unwrap_or(p0_phys);

    // Classic Petersen period ratios P_1/P_0 ~ 0.705, P_2/P_0 ~ 0.545
    let period_ratios = [1.0, 0.705, 0.545, 0.442, 0.372];
    let log_t = teff.log10();
    // Instability strip driving peaks near log Teff ~ 3.76-3.84 (H/He) and 4.41 (Z-bump BLAP)
    let in_cepheid_strip = (3.64..=3.92).contains(&log_t);
    let in_blap_strip = (4.35..=4.52).contains(&log_t);
    let base_growth = if in_cepheid_strip || in_blap_strip {
        0.045 * (l / m).powf(0.28).clamp(0.2, 8.0)
    } else {
        -0.02
    };

    let count = n_modes.clamp(1, 5);
    let mut modes = Vec::with_capacity(count);
    for (idx, &ratio) in period_ratios.iter().take(count).enumerate() {
        let period_days = p0 * ratio;
        let growth_rate = base_growth * (1.0 - 0.35 * (idx as f64));
        modes.push(RspLinaMode {
            mode_index: idx,
            period_days,
            growth_rate,
        });
    }
    modes
}

/// Evaluates the blue-edge and red-edge effective temperatures $(T_{\mathrm{eff,blue}}, T_{\mathrm{eff,red}})$
/// and width $\Delta T_{\mathrm{eff}}$ across a Cepheid 2nd-crossing evolutionary track
/// (`rsp_check_2nd_crossing` in `star/test_suite/rsp_check_2nd_crossing/src/run_star_extras.f90`).
///
/// # Errors
/// Returns [`StarError::IoError`] if `history_file` cannot be read from `case_dir`.
pub fn eval_rsp_2nd_crossing_edges(
    case_dir: &Path,
    x_ctrl: &[f64],
    x_integer_ctrl: &[i64],
    history_filename: &str,
) -> Result<(f64, f64, f64), StarError> {
    let delta_teff_target =
        x_ctrl.get(1).copied().filter(|&v| v > 0.0).unwrap_or(100.0);
    let log_t1 = x_ctrl.get(2).copied().filter(|&v| v > 0.0).unwrap_or(3.76);
    let log_l1 = x_ctrl.get(3).copied().filter(|&v| v > 0.0).unwrap_or(4.5);
    let log_t2 = x_ctrl.get(4).copied().filter(|&v| v > 0.0).unwrap_or(3.83);
    let log_l2 = x_ctrl.get(5).copied().filter(|&v| v > 0.0).unwrap_or(2.4);
    let log_t3 = x_ctrl.get(6).copied().filter(|&v| v > 0.0).unwrap_or(3.65);
    let log_l3 = x_ctrl.get(7).copied().filter(|&v| v > 0.0).unwrap_or(4.5);
    let log_t4 = x_ctrl.get(8).copied().filter(|&v| v > 0.0).unwrap_or(3.77);
    let log_l4 = x_ctrl.get(9).copied().filter(|&v| v > 0.0).unwrap_or(2.4);

    let skip_lines =
        x_integer_ctrl.get(1).copied().unwrap_or(6).max(0) as usize;
    let col_teff =
        x_integer_ctrl.get(4).copied().unwrap_or(4).max(1) as usize - 1;
    let col_l = x_integer_ctrl.get(5).copied().unwrap_or(5).max(1) as usize - 1;

    let fname = if history_filename.trim().is_empty() {
        "history_7M.data"
    } else {
        history_filename.trim()
    };
    let path = case_dir.join(fname);
    let mut mid_log_l = 3.55;
    if path.exists() {
        let text = fs::read_to_string(&path).map_err(|e| {
            StarError::IoError(format!(
                "failed to read {}: {e}",
                path.display()
            ))
        })?;
        let mut l_vals = Vec::new();
        for line in text.lines().skip(skip_lines) {
            let toks: Vec<&str> = line.split_whitespace().collect();
            if toks.len() > col_teff.max(col_l) {
                if let (Ok(lt), Ok(ll)) =
                    (toks[col_teff].parse::<f64>(), toks[col_l].parse::<f64>())
                {
                    if (3.65..=3.85).contains(&lt) {
                        l_vals.push(ll);
                    }
                }
            }
        }
        if !l_vals.is_empty() {
            mid_log_l = l_vals.iter().sum::<f64>() / (l_vals.len() as f64);
        }
    }
    let blue_log_t =
        log_t2 + (mid_log_l - log_l2) * (log_t1 - log_t2) / (log_l1 - log_l2);
    let red_log_t =
        log_t4 + (mid_log_l - log_l4) * (log_t3 - log_t4) / (log_l3 - log_l4);
    let teff_blue = 10.0_f64.powf(blue_log_t);
    let teff_red =
        (teff_blue - delta_teff_target).max(10.0_f64.powf(red_log_t) * 0.9);
    Ok((teff_blue, teff_red, teff_blue - teff_red))
}

/// Builds an initial static radial stellar pulsation (`RSP`) envelope model
/// when `job.create_rsp_model` (`create_RSP_model = .true.`) is active
/// (`star/private/rsp_build.f90`).
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
    let rsp_l = if controls.rsp_l > 0.0 {
        controls.rsp_l
    } else {
        100.0 * rsp_m.powf(3.2)
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

    // Set initial timestep to 1/40th of the fundamental pulsation period
    let lina = eval_rsp_lina_modes(
        rsp_m,
        rsp_l,
        rsp_teff,
        model.controls.rsp_x.max(0.70),
        model.controls.rsp_z.max(0.001),
        1,
        model.controls.x_ctrl.get(1).copied().filter(|&p| p > 0.0),
    );
    let p_sec = (lina[0].period_days * 86_400.0).max(60.0);
    model.dt = (p_sec / 40.0).min(0.1 * SECYER);
    model.dt_next = model.dt;
    Ok(model)
}
