// SPDX-License-Identifier: LGPL-3.0-or-later
//! `PyO3` bindings for Stage `E2E-4` (`#95`, closing `#4` and `#5`) and Stage
//! `E2E-5` (`#96`) oscillation mode solvers, `RSP` envelope builders, `astero`
//! test suite runners, and `star/dev_cases_*` pulsation suite runners.

use std::path::Path;

use flexmesa_binary::{
    eval_brott_wind_msun_yr, eval_gr_expected_separation_rsun,
    eval_tauris_vdh_expected_separation_rsun, run_binary_suite,
    run_binary_test_suite_case, BinaryCaseRunReport, TIER5_BINARY_CASES,
};
use flexmesa_star::{
    all_active_star_case_names, build_rsp_envelope_model,
    eval_astero_scaling_dnu_numax_uhz, eval_grafener_kudritzki_wind_msun_yr,
    eval_linear_oscillation_modes, eval_overshoot_dbl_exp,
    eval_rsp_2nd_crossing_edges, eval_rsp_lina_modes,
    eval_target_mode_frequency_and_growth, run_all_active_star_suite,
    run_astero_suite, run_astero_test_suite_case, run_dev_cases_suite,
    run_tier4_star_suite, run_tier4_suite, run_tier5_star_suite,
    verify_step_redo_bit_parity, CaseRunReport, LinearOscillationMode,
    StarControls, StarJobControls, TIER4_ACTIVE_STAR_CASES, TIER4_ASTERO_CASES,
    TIER4_DEV_CASES, TIER4_STAR_CASES, TIER5_STAR_CASES,
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

fn str_slice_to_vec(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_string()).collect()
}

/// Returns the 16 Tier 4 `star/test_suite` case names.
#[pyfunction]
#[must_use]
pub fn star_tier4_star_case_names() -> Vec<String> {
    str_slice_to_vec(&TIER4_STAR_CASES)
}

/// Returns the 15 active Tier 4 `star/test_suite` case names in `do_all_test_suite`.
#[pyfunction]
#[must_use]
pub fn star_tier4_active_star_case_names() -> Vec<String> {
    str_slice_to_vec(&TIER4_ACTIVE_STAR_CASES)
}

/// Returns the 8 `astero/test_suite` case names.
#[pyfunction]
#[must_use]
pub fn star_tier4_astero_case_names() -> Vec<String> {
    str_slice_to_vec(&TIER4_ASTERO_CASES)
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
pub fn star_eval_astero_scaling_dnu_numax_uhz(
    model: &PyStarModel,
) -> (f64, f64) {
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
    let x_ctrl =
        vec![0.0, delta_teff, 3.76, 4.5, 3.83, 2.4, 3.65, 4.5, 3.77, 2.4];
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

fn run_star_suite_detached<'py, F>(
    py: Python<'py>,
    mesa_dir: &str,
    runner: F,
) -> PyResult<Bound<'py, PyList>>
where
    F: FnOnce(&Path) -> Result<Vec<CaseRunReport>, flexmesa_star::StarError>
        + Send,
{
    let root = Path::new(mesa_dir).to_path_buf();
    let reports = py.detach(|| runner(&root)).map_err(to_py_err)?;
    reports_to_pylist(py, &reports)
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
    run_star_suite_detached(py, mesa_dir, |root| {
        run_tier4_star_suite(
            root,
            skip_optional,
            max_steps_per_stage,
            verify_restart,
        )
    })
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
    run_star_suite_detached(py, mesa_dir, |root| {
        run_astero_suite(
            root,
            skip_optional,
            max_steps_per_stage,
            verify_restart,
        )
    })
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
    run_star_suite_detached(py, mesa_dir, |root| {
        run_tier4_suite(
            root,
            skip_optional,
            max_steps_per_stage,
            verify_restart,
        )
    })
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
    run_star_suite_detached(py, mesa_dir, |root| {
        run_dev_cases_suite(
            root,
            skip_optional,
            max_steps_per_stage,
            verify_restart,
        )
    })
}

fn binary_report_to_pydict<'py>(
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

/// Registers Stage `E2E-4` and `E2E-5` `PyO3` functions on `_rust`.
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
