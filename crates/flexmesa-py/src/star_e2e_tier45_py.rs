// SPDX-License-Identifier: LGPL-3.0-or-later
//! `PyO3` bindings for Stage `E2E-4` (`#95`, closing `#4` and `#5`) and Stage
//! `E2E-5` (`#96`) oscillation mode solvers, `RSP` envelope builders, `astero`
//! test suite runners, and `star/dev_cases_*` pulsation suite runners.

use std::path::Path;

use flexmesa_star::{
    build_rsp_envelope_model, eval_astero_scaling_dnu_numax_uhz,
    eval_linear_oscillation_modes, eval_rsp_2nd_crossing_edges,
    eval_rsp_lina_modes, eval_target_mode_frequency_and_growth,
    run_astero_suite, run_astero_test_suite_case, run_dev_cases_suite,
    run_tier4_star_suite, run_tier4_suite, CaseRunReport, LinearOscillationMode,
    StarControls, StarJobControls, TIER4_ACTIVE_STAR_CASES,
    TIER4_ASTERO_CASES, TIER4_DEV_CASES, TIER4_STAR_CASES,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::star_e2e_py::case_report_to_pydict;
use crate::star_helpers_py::to_py_err;
use crate::star_py::PyStarModel;

fn oscillation_mode_to_pydict<'py>(
    py: Python<'py>,
    m: &LinearOscillationMode,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("l", m.l)?;
    d.set_item("n_pg", m.n_pg)?;
    d.set_item("freq_hz", m.freq_hz)?;
    d.set_item("freq_uhz", m.freq_uhz)?;
    d.set_item("period_days", m.period_days)?;
    d.set_item("growth_rate_s", m.growth_rate_s)?;
    d.set_item("growth_timescale_days", m.growth_timescale_days)?;
    d.set_item("inertia", m.inertia)?;
    Ok(d)
}

fn reports_to_pylist<'py>(
    py: Python<'py>,
    reports: &[CaseRunReport],
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for rep in reports {
        out.append(case_report_to_pydict(py, rep)?)?;
    }
    Ok(out)
}

/// Returns the 16 Tier 4 `star/test_suite` case names (`6` `gyre_in_mesa_*` +
/// `10` `rsp_*` including `rsp_Cepheid_6M`).
#[pyfunction]
#[must_use]
pub fn star_tier4_star_case_names() -> Vec<String> {
    TIER4_STAR_CASES.iter().map(|s| (*s).to_string()).collect()
}

/// Returns the 15 active Tier 4 `star/test_suite` case names in `do_all_test_suite`.
#[pyfunction]
#[must_use]
pub fn star_tier4_active_star_case_names() -> Vec<String> {
    TIER4_ACTIVE_STAR_CASES
        .iter()
        .map(|s| (*s).to_string())
        .collect()
}

/// Returns the 8 `astero/test_suite` case names.
#[pyfunction]
#[must_use]
pub fn star_tier4_astero_case_names() -> Vec<String> {
    TIER4_ASTERO_CASES
        .iter()
        .map(|s| (*s).to_string())
        .collect()
}

/// Returns the 32 `(suite_subdir, case_name)` pairs across the 5 `star/dev_cases_*` suites.
#[pyfunction]
#[must_use]
pub fn star_tier4_dev_case_names() -> Vec<(String, String)> {
    TIER4_DEV_CASES
        .iter()
        .map(|(s, c)| ((*s).to_string(), (*c).to_string()))
        .collect()
}

/// Evaluates linearized adiabatic and non-adiabatic oscillation eigenmodes for
/// spherical harmonic degree `l` and radial orders `n_orders` on `model`.
#[pyfunction]
pub fn star_eval_linear_oscillation_modes<'py>(
    py: Python<'py>,
    model: &PyStarModel,
    l: usize,
    n_orders: Vec<i32>,
) -> PyResult<Bound<'py, PyList>> {
    let modes = eval_linear_oscillation_modes(&model.inner, l, &n_orders);
    let out = PyList::empty(py);
    for m in &modes {
        out.append(oscillation_mode_to_pydict(py, m)?)?;
    }
    Ok(out)
}

/// Evaluates the target oscillation mode requested by `x_integer_ctrl(3..4)`
/// and `x_ctrl(1..2)` on `model` and returns `(mode_dict, matched_target)`.
#[pyfunction]
pub fn star_eval_target_mode_frequency_and_growth<'py>(
    py: Python<'py>,
    model: &PyStarModel,
) -> PyResult<(Bound<'py, PyDict>, bool)> {
    let (m, matched) = eval_target_mode_frequency_and_growth(&model.inner);
    Ok((oscillation_mode_to_pydict(py, &m)?, matched))
}

/// Evaluates global asteroseismic scaling parameters $(\Delta\nu, \nu_{\max})$
/// in $\mu\mathrm{Hz}$ on `model`.
#[pyfunction]
#[must_use]
pub fn star_eval_astero_scaling_dnu_numax_uhz(model: &PyStarModel) -> (f64, f64) {
    eval_astero_scaling_dnu_numax_uhz(&model.inner)
}

/// Evaluates linear non-adiabatic (`LINA`) radial pulsation periods (days) and
/// growth rates for an `RSP` envelope with $(M/M_\odot, L/L_\odot, T_{\mathrm{eff}}, X, Z)$.
#[pyfunction]
#[pyo3(signature = (m_msun, l_lsun, teff_k, x_h=0.73, z_metal=0.007, n_modes=3, expected_f_period_days=None))]
pub fn star_eval_rsp_lina_modes<'py>(
    py: Python<'py>,
    m_msun: f64,
    l_lsun: f64,
    teff_k: f64,
    x_h: f64,
    z_metal: f64,
    n_modes: usize,
    expected_f_period_days: Option<f64>,
) -> PyResult<Bound<'py, PyList>> {
    let modes = eval_rsp_lina_modes(
        m_msun,
        l_lsun,
        teff_k,
        x_h,
        z_metal,
        n_modes,
        expected_f_period_days,
    );
    let out = PyList::empty(py);
    for m in modes {
        let d = PyDict::new(py);
        d.set_item("mode_index", m.mode_index)?;
        d.set_item("period_days", m.period_days)?;
        d.set_item("growth_rate", m.growth_rate)?;
        out.append(d)?;
    }
    Ok(out)
}

/// Evaluates the 2nd-crossing instability strip blue-edge and red-edge
/// temperatures $(T_{\mathrm{eff,blue}}, T_{\mathrm{eff,red}}, \Delta T_{\mathrm{eff}})$
/// (`rsp_check_2nd_crossing`).
#[pyfunction]
#[pyo3(signature = (case_dir, delta_teff=100.0, history_filename="history_7M.data"))]
pub fn star_eval_rsp_2nd_crossing_edges(
    case_dir: &str,
    delta_teff: f64,
    history_filename: &str,
) -> PyResult<(f64, f64, f64)> {
    let x_ctrl = vec![
        0.0, delta_teff, 3.76, 4.5, 3.83, 2.4, 3.65, 4.5, 3.77, 2.4,
    ];
    let x_int = vec![0_i64, 6, 1, 2, 4, 5];
    eval_rsp_2nd_crossing_edges(
        Path::new(case_dir),
        &x_ctrl,
        &x_int,
        history_filename,
    )
    .map_err(to_py_err)
}

/// Builds a static radial stellar pulsation (`RSP`) envelope [`PyStarModel`]
/// (`create_RSP_model = .true.`).
#[pyfunction]
#[pyo3(signature = (m_msun=4.165, l_lsun=1438.8, teff_k=6050.0, x_h=0.73, z_metal=0.007))]
pub fn star_build_rsp_envelope_model(
    m_msun: f64,
    l_lsun: f64,
    teff_k: f64,
    x_h: f64,
    z_metal: f64,
) -> PyResult<PyStarModel> {
    let controls = StarControls {
        rsp_mass: m_msun,
        rsp_l: l_lsun,
        rsp_teff: teff_k,
        rsp_x: x_h,
        rsp_z: z_metal,
        ..Default::default()
    };
    let job = StarJobControls {
        create_rsp_model: true,
        ..Default::default()
    };
    let inner = build_rsp_envelope_model(controls, job).map_err(to_py_err)?;
    Ok(PyStarModel { inner })
}

/// Runs all 16 Tier 4 `star/test_suite` cases under `mesa_dir`.
#[pyfunction]
#[pyo3(signature = (mesa_dir, skip_optional=true, max_steps_per_stage=None, verify_restart=true))]
pub fn star_run_tier4_star_suite<'py>(
    py: Python<'py>,
    mesa_dir: &str,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> PyResult<Bound<'py, PyList>> {
    let root = Path::new(mesa_dir).to_path_buf();
    let reports = py
        .detach(|| {
            run_tier4_star_suite(
                &root,
                skip_optional,
                max_steps_per_stage,
                verify_restart,
            )
        })
        .map_err(to_py_err)?;
    reports_to_pylist(py, &reports)
}

/// Runs a single `astero/test_suite` case directory.
#[pyfunction]
#[pyo3(signature = (case_dir, skip_optional=true, max_steps_per_stage=None, verify_restart=true))]
pub fn star_run_astero_test_suite_case<'py>(
    py: Python<'py>,
    case_dir: &str,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let dir = Path::new(case_dir).to_path_buf();
    let rep = py
        .detach(|| {
            run_astero_test_suite_case(
                &dir,
                skip_optional,
                max_steps_per_stage,
                verify_restart,
            )
        })
        .map_err(to_py_err)?;
    case_report_to_pydict(py, &rep)
}

/// Runs all 8 `astero/test_suite` cases under `mesa_dir`.
#[pyfunction]
#[pyo3(signature = (mesa_dir, skip_optional=true, max_steps_per_stage=None, verify_restart=true))]
pub fn star_run_astero_suite<'py>(
    py: Python<'py>,
    mesa_dir: &str,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> PyResult<Bound<'py, PyList>> {
    let root = Path::new(mesa_dir).to_path_buf();
    let reports = py
        .detach(|| {
            run_astero_suite(
                &root,
                skip_optional,
                max_steps_per_stage,
                verify_restart,
            )
        })
        .map_err(to_py_err)?;
    reports_to_pylist(py, &reports)
}

/// Runs all 24 Tier 4 `test_suite` cases (`16` `star` + `8` `astero`) under `mesa_dir`.
#[pyfunction]
#[pyo3(signature = (mesa_dir, skip_optional=true, max_steps_per_stage=None, verify_restart=true))]
pub fn star_run_tier4_suite<'py>(
    py: Python<'py>,
    mesa_dir: &str,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> PyResult<Bound<'py, PyList>> {
    let root = Path::new(mesa_dir).to_path_buf();
    let reports = py
        .detach(|| {
            run_tier4_suite(
                &root,
                skip_optional,
                max_steps_per_stage,
                verify_restart,
            )
        })
        .map_err(to_py_err)?;
    reports_to_pylist(py, &reports)
}

/// Runs all 32 `star/dev_cases_*` cases under `mesa_dir`.
#[pyfunction]
#[pyo3(signature = (mesa_dir, skip_optional=true, max_steps_per_stage=None, verify_restart=true))]
pub fn star_run_dev_cases_suite<'py>(
    py: Python<'py>,
    mesa_dir: &str,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> PyResult<Bound<'py, PyList>> {
    let root = Path::new(mesa_dir).to_path_buf();
    let reports = py
        .detach(|| {
            run_dev_cases_suite(
                &root,
                skip_optional,
                max_steps_per_stage,
                verify_restart,
            )
        })
        .map_err(to_py_err)?;
    reports_to_pylist(py, &reports)
}

/// Registers Stage `E2E-4` `PyO3` functions on `_rust`.
pub fn register_star_e2e_tier45(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(star_tier4_star_case_names, m)?)?;
    m.add_function(wrap_pyfunction!(star_tier4_active_star_case_names, m)?)?;
    m.add_function(wrap_pyfunction!(star_tier4_astero_case_names, m)?)?;
    m.add_function(wrap_pyfunction!(star_tier4_dev_case_names, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_linear_oscillation_modes, m)?)?;
    m.add_function(wrap_pyfunction!(
        star_eval_target_mode_frequency_and_growth,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        star_eval_astero_scaling_dnu_numax_uhz,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(star_eval_rsp_lina_modes, m)?)?;
    m.add_function(wrap_pyfunction!(star_eval_rsp_2nd_crossing_edges, m)?)?;
    m.add_function(wrap_pyfunction!(star_build_rsp_envelope_model, m)?)?;
    m.add_function(wrap_pyfunction!(star_run_tier4_star_suite, m)?)?;
    m.add_function(wrap_pyfunction!(star_run_astero_test_suite_case, m)?)?;
    m.add_function(wrap_pyfunction!(star_run_astero_suite, m)?)?;
    m.add_function(wrap_pyfunction!(star_run_tier4_suite, m)?)?;
    m.add_function(wrap_pyfunction!(star_run_dev_cases_suite, m)?)?;
    Ok(())
}
