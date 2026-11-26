// SPDX-License-Identifier: GPL-3.0-or-later
//! Pulsation-data text export for `GYRE`, `FGONG` and `OSC`
//! (`export_pulse_data`, `star/private/pulse.f90`), from the faithful
//! producers [`crate::pulse_gyre_data::get_gyre_data`],
//! [`crate::pulse_fgong::get_fgong_data`] and
//! [`crate::pulse_osc::get_osc_data`] and the writers in
//! [`crate::pulse_write`] (#491), plus the $T(\tau)$ atmosphere of the
//! in-house pulsation grid ([`build_pulse_atmosphere_points`]).

use flexmesa_const::PI4;

use crate::model::StarModel;
use crate::pulse_fgong::{get_fgong_data, PulseDataInput};
use crate::pulse_gyre_data::{star_get_pulse_data_gyre, GyreDataError};
use crate::pulse_osc::get_osc_data;
use crate::pulse_utils::make_pulse_point_on_segment;
pub use crate::pulse_utils::{
    build_pulse_points, build_pulse_points_with_options, eval_center_d2,
    eval_center_parabolic, eval_center_rho, eval_face, eval_face_a_ast,
    set_segment_indices, PulseExportOptions, PulsePoint, PULSE_ISO_NAMES,
};
use crate::pulse_write::{
    write_fgong_data, write_gyre_data, write_osc_data, PulseWriteError,
};

/// Builds an optically thin $T(\tau)$ atmosphere above `model`'s surface (`build_atm`).
#[must_use]
pub fn build_pulse_atmosphere_points(
    model: &StarModel,
    r_tot: f64,
    m_tot: f64,
    n_atm: usize,
) -> Vec<PulsePoint> {
    if model.nz == 0 || n_atm < 2 {
        return Vec::new();
    }
    let surf = make_pulse_point_on_segment(
        model,
        0,
        0,
        model.nz - 1,
        r_tot,
        m_tot,
        true,
    );
    let tau_base = model.tau_base.max(2.0 / 3.0);
    let tau_top = (1.0e-4 * tau_base).min(1.0e-4);
    let g_surf = (model.cgrav[0] * m_tot / (r_tot * r_tot)).max(1.0e-30);
    let teff = if model.teff > 100.0 {
        model.teff
    } else {
        surf.t
    };
    let hp_surf = (surf.p / (g_surf * surf.rho)).max(1.0);
    let mut atm = Vec::with_capacity(n_atm - 1);
    for i in (1..n_atm).rev() {
        let frac = (i as f64) / (n_atm as f64);
        let tau = flexmesa_math::exp(
            flexmesa_math::log(tau_top)
                + frac
                    * (flexmesa_math::log(tau_base)
                        - flexmesa_math::log(tau_top)),
        );
        let t_tau = teff * flexmesa_math::pow(0.75 * (tau + 2.0 / 3.0), 0.25);
        let p_ratio = (tau / tau_base).max(1.0e-8);
        let (p_tau, rho_tau) = (
            surf.p * p_ratio,
            surf.rho * p_ratio * (surf.t / t_tau.max(1.0)),
        );
        let r = (r_tot - hp_surf * flexmesa_math::log(p_ratio)).max(r_tot);
        let g = (model.cgrav[0] * m_tot / (r * r)).max(1.0e-30);
        let grad_t = 0.1875 * tau / (tau + 2.0 / 3.0);
        let n2 =
            g * g * (rho_tau / p_tau) * surf.delta * (surf.grad_ad - grad_t);
        let mut pt = surf;
        pt.r = r;
        pt.p = p_tau;
        pt.pgas = p_tau * (surf.pgas / surf.p.max(1.0e-30));
        pt.t = t_tau;
        pt.rho = rho_tau;
        pt.grad_t = grad_t;
        pt.grad_r = grad_t;
        pt.tau = tau;
        pt.eps_nuc = 0.0;
        pt.eps = 0.0;
        pt.eps_grav = 0.0;
        pt.entropy = 0.0;
        pt.r_x_h1 = 0.0;
        pt.n2 = n2;
        pt.v_dziem = g * r * rho_tau / p_tau;
        pt.v_g = pt.v_dziem / pt.gamma1;
        pt.a_star = r * n2 / g;
        pt.u_dziem = PI4 * flexmesa_math::pow3(r) * rho_tau / m_tot;
        pt.c_1 = flexmesa_math::pow3(r / r_tot);
        atm.push(pt);
    }
    atm
}

/// `star_export_pulse_data(id, data_format, filename, add_center_point,
/// keep_surface_point, add_atmosphere, ierr)` for the text formats `GYRE`,
/// `FGONG` and `OSC` (`export_pulse_data`, `star/private/pulse.f90:43-86`):
/// `get_pulse_data` then `write_pulse_data`, returning the file text.
/// `use_other_export_pulse_data` is not supported.
///
/// # Errors
///
/// [`PulseWriteError`] for an unknown format, a producer error
/// ([`GyreDataError`]) or a writer error.
pub fn export_pulse_data_text(
    model: &StarModel,
    data_format: &str,
    add_center_point: bool,
    keep_surface_point: bool,
    add_atmosphere: bool,
) -> Result<String, PulseWriteError> {
    let get_err = |e: GyreDataError| PulseWriteError::Upstream(e.to_string());
    let c = &model.controls;
    match data_format.trim().to_ascii_uppercase().as_str() {
        "GYRE" => {
            let (g, pd) = star_get_pulse_data_gyre(
                model,
                add_center_point,
                keep_surface_point,
                add_atmosphere,
            )
            .map_err(get_err)?;
            write_gyre_data(&g, &pd, c.gyre_data_schema)
        }
        "FGONG" => {
            let input = PulseDataInput::from_model(model).map_err(get_err)?;
            let (g, pd) = get_fgong_data(
                &input,
                add_center_point,
                keep_surface_point,
                add_atmosphere,
            )
            .map_err(get_err)?;
            write_fgong_data(&g, &pd, &default_fgong_header(), c.fgong_ivers)
        }
        "OSC" => {
            let input = PulseDataInput::from_model(model).map_err(get_err)?;
            let (g, pd) = get_osc_data(
                &input,
                add_center_point,
                keep_surface_point,
                add_atmosphere,
            )
            .map_err(get_err)?;
            write_osc_data(&g, &pd, MESA_VERSION_NUMBER, &c.format_for_osc_data)
        }
        other => Err(PulseWriteError::Unsupported(format!(
            "data_format '{other}' (text formats GYRE, FGONG, OSC)"
        ))),
    }
}

/// `version_number` of MESA 26.4.1 (`$MESA_DIR/data/version_number`).
pub const MESA_VERSION_NUMBER: &str = "r26.4.1";

/// `fgong_header` defaults (`star/defaults/controls.defaults:655-658`).
#[must_use]
pub fn default_fgong_header() -> [String; 4] {
    [
        String::from("FGONG file"),
        String::from("Created by MESAstar"),
        String::new(),
        String::new(),
    ]
}

/// [`export_pulse_data_text`] with `data_format = 'GYRE'`.
///
/// # Errors
///
/// As [`export_pulse_data_text`].
pub fn export_gyre(
    model: &StarModel,
    add_center_point: bool,
    keep_surface_point: bool,
    add_atmosphere: bool,
) -> Result<String, PulseWriteError> {
    export_pulse_data_text(
        model,
        "GYRE",
        add_center_point,
        keep_surface_point,
        add_atmosphere,
    )
}

/// [`export_pulse_data_text`] with `data_format = 'FGONG'`.
///
/// # Errors
///
/// As [`export_pulse_data_text`].
pub fn export_fgong(
    model: &StarModel,
    add_center_point: bool,
    keep_surface_point: bool,
    add_atmosphere: bool,
) -> Result<String, PulseWriteError> {
    export_pulse_data_text(
        model,
        "FGONG",
        add_center_point,
        keep_surface_point,
        add_atmosphere,
    )
}

/// [`export_pulse_data_text`] with `data_format = 'OSC'`.
///
/// # Errors
///
/// As [`export_pulse_data_text`].
pub fn export_osc(
    model: &StarModel,
    add_center_point: bool,
    keep_surface_point: bool,
    add_atmosphere: bool,
) -> Result<String, PulseWriteError> {
    export_pulse_data_text(
        model,
        "OSC",
        add_center_point,
        keep_surface_point,
        add_atmosphere,
    )
}
