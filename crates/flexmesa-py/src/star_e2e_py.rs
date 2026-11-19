// SPDX-License-Identifier: LGPL-3.0-or-later
//! `PyO3` bindings for `flexmesa-star` E2E-0 `inlist` bundle parsing, `./rn`
//! stage resolution, photo restart parity verification, and multi-stage
//! `test_suite` case execution (`inlist` and `inlist_runner`).

use std::path::Path;

use flexmesa_astero::run_astero_test_suite_case_with_options;
use flexmesa_binary::{
    run_binary_test_suite_case_with_extras, BinaryCaseRunReport,
    BinaryLogsOptions,
};
use flexmesa_star::inlist_runner::{
    run_test_suite_case_with_extras, StarLogsOptions,
};
use flexmesa_star::{
    build_rsp_envelope_model, eval_astero_scaling_dnu_numax_uhz,
    eval_linear_oscillation_modes, eval_rsp_lina_modes,
    eval_target_mode_frequency_and_growth, parse_rn_stages,
    verify_photo_restart_parity, CaseRunReport, InlistBundle,
    InlistLoadOptions, LinearOscillationMode, RnStage, StarControls,
    StarJobControls,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::extras_py::{run_with_binary_extras, run_with_star_extras};
use crate::star_helpers_py::to_py_err;
use crate::star_py::PyStarModel;

/// Copies a `star`, `binary`, or `astero` case directory from `src_dir` to
/// `dst_dir` while stripping unsupported or unused namelist keys.
#[pyfunction]
#[pyo3(signature = (src_dir, dst_dir, suite="star"))]
pub fn star_copy_cleaned_case(
    src_dir: &str,
    dst_dir: &str,
    suite: &str,
) -> PyResult<()> {
    let (src, dst) = (Path::new(src_dir), Path::new(dst_dir));
    match suite.trim().to_ascii_lowercase().as_str() {
        "star" => flexmesa_star::copy_cleaned_star_case(src, dst)
            .map_err(to_py_err),
        "binary" => flexmesa_binary::copy_cleaned_binary_case(src, dst)
            .map_err(to_py_err),
        "astero" => flexmesa_astero::copy_cleaned_astero_case(src, dst)
            .map_err(to_py_err),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported suite '{other}' (expected 'star', 'binary', or 'astero')"
        ))),
    }
}

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
    d.set_item("wall_time_ms", rep.wall_time_ms)?;
    d.set_item("binary_retries", rep.binary_retries)?;
    d.set_item("implicit_trials", rep.implicit_trials)?;
    d.set_item("dt_cuts", rep.dt_cuts)?;
    d.set_item("hard_limit_retries", rep.hard_limit_retries)?;
    d.set_item("donor_retries", rep.donor_retries)?;
    d.set_item("accretor_retries", rep.accretor_retries)?;
    Ok(d)
}

/// Parses a MESA `inlist` file (recursively resolving all `read_extra_*_inlist`
/// includes) and returns a summary dictionary of parsed controls.
#[pyfunction]
#[pyo3(signature = (inlist_path, allow_unsupported_controls=false))]
pub fn star_parse_inlist_bundle<'py>(
    py: Python<'py>,
    inlist_path: &str,
    allow_unsupported_controls: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let bundle = InlistBundle::from_file_with_options(
        Path::new(inlist_path),
        InlistLoadOptions {
            allow_unsupported_controls,
        },
    )
    .map_err(to_py_err)?;
    let dict = PyDict::new(py);
    dict.set_item("entry_inlist", &bundle.entry_inlist)?;
    dict.set_item("create_pre_ms_model", bundle.job.create_pre_ms_model)?;
    dict.set_item("create_initial_model", bundle.job.create_initial_model)?;
    dict.set_item("create_rsp_model", bundle.job.create_rsp_model)?;
    dict.set_item("load_saved_model", bundle.job.load_saved_model)?;
    dict.set_item("load_model_filename", &bundle.job.saved_model_name)?;
    dict.set_item(
        "save_model_when_terminate",
        bundle.job.save_model_when_terminate,
    )?;
    dict.set_item("save_model_filename", &bundle.job.save_model_filename)?;
    dict.set_item("change_net", bundle.job.change_net)?;
    dict.set_item("new_net_name", &bundle.job.new_net_name)?;
    dict.set_item(
        "required_termination_code_string",
        &bundle.job.required_termination_code_string,
    )?;
    dict.set_item("initial_mass", bundle.controls.initial_mass)?;
    dict.set_item("initial_z", bundle.controls.initial_z)?;
    dict.set_item("initial_y", bundle.controls.initial_y)?;
    dict.set_item("max_model_number", bundle.controls.max_model_number)?;
    dict.set_item("max_age", bundle.controls.max_age)?;
    dict.set_item("stop_near_zams", bundle.controls.stop_near_zams)?;
    dict.set_item("v_flag", bundle.controls.v_flag)?;
    dict.set_item("u_flag", bundle.controls.u_flag)?;
    dict.set_item("rti_flag", bundle.controls.rti_flag)?;
    dict.set_item("rotation_flag", bundle.controls.rotation_flag)?;
    dict.set_item("rsp_flag", bundle.controls.rsp_flag)?;
    dict.set_item("rsp2_flag", bundle.controls.rsp2_flag)?;
    dict.set_item("change_rsp2_flag", bundle.controls.rsp2.change_rsp2_flag)?;
    dict.set_item("new_rsp2_flag", bundle.controls.rsp2.new_rsp2_flag)?;
    dict.set_item("okay_to_remesh", bundle.controls.okay_to_remesh)?;
    dict.set_item("use_gold_tolerances", bundle.controls.use_gold_tolerances)?;
    dict.set_item(
        "use_gold2_tolerances",
        bundle.controls.use_gold2_tolerances,
    )?;
    dict.set_item("op_split_burn", bundle.controls.op_split_burn)?;
    dict.set_item("op_split_burn_min_t", bundle.controls.op_split_burn_min_t)?;
    dict.set_item("max_newton_iters", bundle.controls.max_newton_iters)?;
    dict.set_item("use_other_cgrav", bundle.controls.use_other_cgrav)?;
    let sections: Vec<String> = bundle.sections.keys().cloned().collect();
    dict.set_item("sections", sections)?;
    Ok(dict)
}

/// Parses an upstream MESA `./rn` script in `case_dir` into a list of
/// `(stage_kind, arg1, arg2)` tuples.
#[pyfunction]
#[pyo3(signature = (case_dir, skip_optional=true))]
pub fn star_parse_rn_stages(
    case_dir: &str,
    skip_optional: bool,
) -> PyResult<Vec<(String, String, String)>> {
    let stages = parse_rn_stages(Path::new(case_dir), skip_optional)
        .map_err(to_py_err)?;
    let mut out = Vec::with_capacity(stages.len());
    for st in stages {
        match st {
            RnStage::DoOne {
                inlist, output_mod, ..
            } => {
                out.push((
                    String::from("do_one"),
                    inlist,
                    output_mod.unwrap_or_default(),
                ));
            }
            RnStage::CopyCheckpoint { src_mod, dst_mod } => {
                out.push((String::from("cp"), src_mod, dst_mod));
            }
        }
    }
    Ok(out)
}

/// Verifies bit-identical `.photo` restart parity on `model` over `n_steps`.
#[pyfunction]
#[pyo3(signature = (model, n_steps=1))]
pub fn star_verify_photo_restart(
    model: &PyStarModel,
    n_steps: usize,
) -> PyResult<(u64, u64)> {
    verify_photo_restart_parity(&model.inner, n_steps).map_err(to_py_err)
}

pub(crate) fn case_report_to_pydict<'py>(
    py: Python<'py>,
    rep: &CaseRunReport,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("case_name", &rep.case_name)?;
    out.set_item("tier", rep.tier)?;
    out.set_item("passed", rep.passed)?;
    out.set_item("termination_code", &rep.termination_code)?;
    out.set_item("total_steps", rep.total_steps)?;
    out.set_item("total_newton_iters", rep.total_newton_iterations)?;
    out.set_item("total_newton_iterations", rep.total_newton_iterations)?;
    out.set_item("total_retries", rep.total_retries)?;
    out.set_item("total_runtime_sec", rep.total_runtime_sec)?;
    out.set_item("wall_time_ms", rep.wall_time_ms)?;
    out.set_item("final_teff", rep.final_teff)?;
    out.set_item("final_log_l", rep.final_log_l)?;
    out.set_item("final_log_r", rep.final_log_r)?;
    out.set_item("final_log_tc", rep.final_log_tc)?;
    out.set_item("final_log_rhoc", rep.final_log_rhoc)?;
    out.set_item("final_center_h1", rep.final_center_h1)?;
    out.set_item("final_center_he4", rep.final_center_he4)?;
    out.set_item("photo_restart_passed", rep.photo_restart_verified)?;
    out.set_item("photo_restart_verified", rep.photo_restart_verified)?;
    out.set_item("assertions_passed", rep.assertions_passed)?;

    let stages_list = PyList::empty(py);
    for st in &rep.stages {
        let sdict = PyDict::new(py);
        sdict.set_item("inlist_name", &st.inlist_name)?;
        sdict.set_item("steps_taken", st.steps_taken)?;
        sdict
            .set_item("total_newton_iterations", st.total_newton_iterations)?;
        sdict.set_item("total_retries", st.total_retries)?;
        sdict.set_item("termination_code", &st.termination_code)?;
        sdict.set_item("runtime_sec", st.runtime_sec)?;
        sdict.set_item("star_age", st.final_history.star_age)?;
        sdict.set_item("star_mass", st.final_history.star_mass)?;
        sdict.set_item("log_l", st.final_history.log_l)?;
        sdict.set_item("log_teff", st.final_history.log_teff)?;
        sdict.set_item("log_r", st.final_history.log_r)?;
        sdict.set_item("log_center_t", st.final_history.log_center_t)?;
        sdict.set_item("log_center_rho", st.final_history.log_center_rho)?;
        sdict.set_item("center_h1", st.final_history.center_h1)?;
        sdict.set_item("center_he4", st.final_history.center_he4)?;
        sdict.set_item("num_zones", st.final_history.num_zones)?;
        stages_list.append(sdict)?;
    }
    out.set_item("stages", stages_list)?;
    Ok(out)
}

/// Runs a multi-stage upstream MESA `test_suite` case directory and returns a
/// structured telemetry dictionary.
#[pyfunction]
#[pyo3(signature = (
    case_dir,
    skip_optional=true,
    max_steps_per_stage=None,
    verify_restart=false,
    allow_unsupported_controls=false,
    extras=None
))]
pub fn star_run_test_suite_case<'py>(
    py: Python<'py>,
    case_dir: &str,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
    allow_unsupported_controls: bool,
    extras: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let load_opts = InlistLoadOptions {
        allow_unsupported_controls,
    };
    let rep = run_with_star_extras(py, extras, |ex| {
        run_test_suite_case_with_extras(
            Path::new(case_dir),
            skip_optional,
            max_steps_per_stage,
            verify_restart,
            StarLogsOptions::default(),
            load_opts,
            ex,
        )
    })?;
    case_report_to_pydict(py, &rep)
}

/// Runs a single `astero/test_suite/<case>` directory and returns a report dict.
#[pyfunction]
#[pyo3(signature = (
    case_dir,
    skip_optional=true,
    max_steps_per_stage=None,
    verify_restart=true,
    allow_unsupported_controls=false
))]
pub fn star_run_astero_test_suite_case<'py>(
    py: Python<'py>,
    case_dir: &str,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
    allow_unsupported_controls: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let dir = Path::new(case_dir).to_path_buf();
    let load_opts = InlistLoadOptions {
        allow_unsupported_controls,
    };
    let rep = py
        .detach(|| {
            run_astero_test_suite_case_with_options(
                &dir,
                skip_optional,
                max_steps_per_stage,
                verify_restart,
                StarLogsOptions::default(),
                load_opts,
            )
        })
        .map_err(to_py_err)?;
    case_report_to_pydict(py, &rep)
}

/// Runs a single coupled `binary/test_suite/<case>` directory and returns a report dict.
#[pyfunction]
#[pyo3(signature = (
    case_dir,
    max_steps_cap=None,
    allow_unsupported_controls=false,
    extras=None
))]
pub fn binary_run_test_suite_case<'py>(
    py: Python<'py>,
    case_dir: &str,
    max_steps_cap: Option<usize>,
    allow_unsupported_controls: bool,
    extras: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let dir = Path::new(case_dir).to_path_buf();
    let load_opts = InlistLoadOptions {
        allow_unsupported_controls,
    };
    let logs_opts = BinaryLogsOptions {
        output_root: None,
        explicit_binary_history_file: None,
        history_interval_override: None,
        write_star_logs: false,
    };
    let rep = run_with_binary_extras(py, extras, |ex| {
        run_binary_test_suite_case_with_extras(
            &dir,
            max_steps_cap,
            logs_opts,
            load_opts,
            ex,
        )
    })?;
    binary_report_to_pydict(py, &rep)
}

/// Evaluates adiabatic/non-adiabatic linear oscillation modes at degree `l`
/// across radial orders `n_orders` on `model`.
#[pyfunction]
#[pyo3(signature = (model, l=1, n_orders=vec![1, 2, 3, 4, 5]))]
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
#[pyo3(signature = (m_msun, l_lsun, teff_k, x_h=0.73, z_metal=0.007, n_modes=3))]
pub fn star_eval_rsp_lina_modes<'py>(
    py: Python<'py>,
    m_msun: f64,
    l_lsun: f64,
    teff_k: f64,
    x_h: f64,
    z_metal: f64,
    n_modes: usize,
) -> PyResult<Bound<'py, PyList>> {
    let modes =
        eval_rsp_lina_modes(m_msun, l_lsun, teff_k, x_h, z_metal, n_modes)
            .map_err(to_py_err)?;
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

/// Builds a static `RSP` envelope [`PyStarModel`] with $(M/M_\odot, L/L_\odot,
/// T_{\mathrm{eff}}, X, Z)$.
#[pyfunction]
#[pyo3(signature = (m_msun, l_lsun, teff_k, x_h=0.73, z_metal=0.007))]
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
    let inner =
        build_rsp_envelope_model(controls, job).map_err(to_py_err)?;
    Ok(PyStarModel { inner })
}

/// Registers end-to-end runner functions on `flexmesa._rust`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(star_parse_inlist_bundle, m)?)?;
    m.add_function(wrap_pyfunction!(star_copy_cleaned_case, m)?)?;
    m.add_function(wrap_pyfunction!(star_parse_rn_stages, m)?)?;
    m.add_function(wrap_pyfunction!(star_verify_photo_restart, m)?)?;
    m.add_function(wrap_pyfunction!(star_run_test_suite_case, m)?)?;
    m.add_function(wrap_pyfunction!(star_run_astero_test_suite_case, m)?)?;
    m.add_function(wrap_pyfunction!(binary_run_test_suite_case, m)?)?;
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
    m.add_function(wrap_pyfunction!(star_build_rsp_envelope_model, m)?)?;
    Ok(())
}
