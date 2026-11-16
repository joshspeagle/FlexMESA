// SPDX-License-Identifier: LGPL-3.0-or-later
//! `PyO3` bindings for Stage `E2E-5` (`#96` and `#114`) Tier 5/6 `star` custom
//! physics hooks, 91-case active `star/test_suite` runner, analytical orbital
//! oracles, and `binary/test_suite` runners (`flexmesa-binary`).

use std::path::Path;

use flexmesa_binary::{
    eval_brott_wind_msun_yr, eval_gr_expected_separation_rsun,
    eval_tauris_vdh_expected_separation_rsun, run_binary_suite,
    run_binary_test_suite_case, BinaryCaseRunReport, TIER5_BINARY_CASES,
};
use flexmesa_star::{
    all_active_star_case_names, eval_grafener_kudritzki_wind_msun_yr,
    eval_overshoot_dbl_exp, run_all_active_star_suite, run_tier5_star_suite,
    verify_step_redo_bit_parity, TIER5_STAR_CASES,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::star_e2e_tier4_py::{run_star_suite_detached, str_slice_to_vec};
use crate::star_helpers_py::to_py_err;
use crate::star_py::PyStarModel;

pub(crate) fn binary_report_to_pydict<'py>(
    py: Python<'py>,
    rep: &BinaryCaseRunReport,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("case_name", &rep.case_name)?;
    d.set_item("passed", rep.passed)?;
    d.set_item("termination_code", &rep.termination_code)?;
    d.set_item("total_steps", rep.total_steps)?;
    d.set_item("final_m1_msun", rep.final_m1_msun)?;
    d.set_item("final_m2_msun", rep.final_m2_msun)?;
    d.set_item("final_separation_rsun", rep.final_separation_rsun)?;
    d.set_item("final_period_days", rep.final_period_days)?;
    d.set_item("final_abs_mdot_msun_yr", rep.final_abs_mdot_msun_yr)?;
    d.set_item("max_relative_oracle_error", rep.max_relative_oracle_error)?;
    d.set_item("wall_time_ms", rep.wall_time_ms)?;
    Ok(d)
}

/// Returns the 4 Tier 5/6 `star/test_suite` case names (`check_redo`,
/// `other_physics_hooks`, `relax_composition_j_entropy`, `timing`).
#[pyfunction]
#[must_use]
pub fn star_tier5_star_case_names() -> Vec<String> {
    str_slice_to_vec(&TIER5_STAR_CASES)
}

/// Returns all 91 active `star/test_suite` case names across Tiers 1-5.
#[pyfunction]
#[must_use]
pub fn star_all_active_star_case_names() -> Vec<String> {
    str_slice_to_vec(&all_active_star_case_names())
}

/// Evaluates double-exponential convective overshooting $D_{\mathrm{ov}}(r)$
/// (`other_overshooting_scheme` in `other_physics_hooks`).
#[pyfunction]
#[must_use]
pub fn star_eval_overshoot_dbl_exp(
    d0: f64,
    delta_r: f64,
    hp_bdy: f64,
    f_ov1: f64,
    f_ov2: f64,
    d_ov_trans: f64,
) -> f64 {
    let d_min = (d_ov_trans * 1.0e-10).clamp(1.0, 1.0e2);
    eval_overshoot_dbl_exp(hp_bdy, d0, 1.0e5, 0.0, f_ov1, f_ov2, d_min, delta_r)
        .0
}

/// Evaluates the Gräfener & Kudritzki (2008) / Vink et al. (2011) high-$\Gamma_e$
/// massive-star wind mass-loss rate $\dot{M}$ ($M_\odot\,\mathrm{yr}^{-1}$)
/// (`other_wind`).
#[pyfunction]
#[must_use]
pub fn star_eval_grafener_kudritzki_wind_msun_yr(
    m_surf_g: f64,
    l_surf_erg_s: f64,
    r_surf_cm: f64,
    t_surf_k: f64,
    x_surf: f64,
    z_surf: f64,
    w_scale: f64,
) -> f64 {
    let y_surf = (1.0 - x_surf - z_surf).max(0.0);
    w_scale
        * eval_grafener_kudritzki_wind_msun_yr(
            l_surf_erg_s,
            m_surf_g,
            r_surf_cm,
            t_surf_k,
            x_surf,
            y_surf,
            z_surf,
            0.1,
        )
        .abs()
}

/// Verifies bit-identical state restoration and step redo parity (`check_redo`).
#[pyfunction]
pub fn star_verify_step_redo_bit_parity(model: &PyStarModel) -> PyResult<bool> {
    let (c1, c2) =
        verify_step_redo_bit_parity(&model.inner).map_err(to_py_err)?;
    Ok(c1 == c2)
}

/// Runs all 4 Tier 5/6 `star/test_suite` cases under `mesa_dir`.
#[pyfunction]
#[pyo3(signature = (mesa_dir, skip_optional=true, max_steps_per_stage=None, verify_restart=true))]
pub fn star_run_tier5_star_suite<'py>(
    py: Python<'py>,
    mesa_dir: &str,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> PyResult<Bound<'py, PyList>> {
    let _ = skip_optional;
    run_star_suite_detached(py, mesa_dir, |root| {
        run_tier5_star_suite(root, max_steps_per_stage, verify_restart)
    })
}

/// Runs all 91 active `star/test_suite` cases across Tiers 1-5 under `mesa_dir`.
#[pyfunction]
#[pyo3(signature = (mesa_dir, max_steps_per_stage=None, verify_restart=true))]
pub fn star_run_all_active_star_suite<'py>(
    py: Python<'py>,
    mesa_dir: &str,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> PyResult<Bound<'py, PyList>> {
    run_star_suite_detached(py, mesa_dir, |root| {
        run_all_active_star_suite(root, max_steps_per_stage, verify_restart)
    })
}

/// Returns all 8 active `binary/test_suite` case names in Stage `E2E-5`.
#[pyfunction]
#[must_use]
pub fn binary_tier5_case_names() -> Vec<String> {
    str_slice_to_vec(&TIER5_BINARY_CASES)
}

/// Evaluates Peters (1964) analytical gravitational-wave inspiral separation
/// $a(t)$ ($R_\odot$) (`jdot_gr_check`).
#[pyfunction]
#[must_use]
pub fn binary_eval_gr_expected_separation_rsun(
    a0_rsun: f64,
    m1_msun: f64,
    m2_msun: f64,
    time_sec: f64,
) -> f64 {
    eval_gr_expected_separation_rsun(a0_rsun, m1_msun, m2_msun, time_sec)
}

/// Evaluates Tauris & van den Heuvel (2003, Eq. 16.20) analytical separation
/// $a(q)$ ($R_\odot$) under non-conservative mass transfer (`jdot_ml_check`).
#[pyfunction]
#[must_use]
pub fn binary_eval_tauris_vdh_expected_separation_rsun(
    a0_rsun: f64,
    q0: f64,
    q: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
    gamma: f64,
) -> f64 {
    eval_tauris_vdh_expected_separation_rsun(
        a0_rsun, q0, q, alpha, beta, delta, gamma,
    )
}

/// Evaluates the Brott et al. (2011) massive-star wind mass-loss rate $\dot{M}$
/// ($M_\odot\,\mathrm{yr}^{-1}$) (`double_bh`).
#[pyfunction]
#[must_use]
pub fn binary_eval_brott_wind_msun_yr(
    l_surf_lsun: f64,
    m_surf_msun: f64,
    r_surf_rsun: f64,
    t_surf_k: f64,
    x_surf: f64,
    y_surf: f64,
    z_div_z_solar: f64,
) -> f64 {
    eval_brott_wind_msun_yr(
        l_surf_lsun,
        m_surf_msun,
        r_surf_rsun,
        t_surf_k,
        x_surf,
        y_surf,
        z_div_z_solar,
    )
}

/// Runs a single `binary/test_suite/<case>` directory.
#[pyfunction]
#[pyo3(signature = (case_dir, max_steps_cap=None))]
pub fn binary_run_test_suite_case<'py>(
    py: Python<'py>,
    case_dir: &str,
    max_steps_cap: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let dir = Path::new(case_dir).to_path_buf();
    let rep = py
        .detach(|| run_binary_test_suite_case(&dir, max_steps_cap))
        .map_err(to_py_err)?;
    binary_report_to_pydict(py, &rep)
}

/// Runs all 8 `binary/test_suite` cases under `mesa_dir`.
#[pyfunction]
#[pyo3(signature = (mesa_dir, max_steps_cap=None))]
pub fn binary_run_suite<'py>(
    py: Python<'py>,
    mesa_dir: &str,
    max_steps_cap: Option<usize>,
) -> PyResult<Bound<'py, PyList>> {
    let root = Path::new(mesa_dir).to_path_buf();
    let reports = py
        .detach(|| run_binary_suite(&root, max_steps_cap))
        .map_err(to_py_err)?;
    let out = PyList::empty(py);
    for rep in &reports {
        out.append(binary_report_to_pydict(py, rep)?)?;
    }
    Ok(out)
}

/// Registers Stage `E2E-5` `PyO3` functions on `_rust`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(star_tier5_star_case_names, m)?)?;
    m.add_function(wrap_pyfunction!(star_all_active_star_case_names, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_overshoot_dbl_exp, m)?)?;
    m.add_function(wrap_pyfunction!(
        star_eval_grafener_kudritzki_wind_msun_yr,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(star_verify_step_redo_bit_parity, m)?)?;
    m.add_function(wrap_pyfunction!(star_run_tier5_star_suite, m)?)?;
    m.add_function(wrap_pyfunction!(star_run_all_active_star_suite, m)?)?;
    m.add_function(wrap_pyfunction!(binary_tier5_case_names, m)?)?;
    m.add_function(wrap_pyfunction!(
        binary_eval_gr_expected_separation_rsun,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        binary_eval_tauris_vdh_expected_separation_rsun,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(binary_eval_brott_wind_msun_yr, m)?)?;
    m.add_function(wrap_pyfunction!(binary_run_test_suite_case, m)?)?;
    m.add_function(wrap_pyfunction!(binary_run_suite, m)?)?;
    Ok(())
}
