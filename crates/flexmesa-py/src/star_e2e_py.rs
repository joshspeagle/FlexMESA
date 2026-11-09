// SPDX-License-Identifier: LGPL-3.0-or-later
//! `PyO3` bindings for `flexmesa-star` E2E-0 `inlist` bundle parsing, `./rn`
//! stage resolution, photo restart parity verification, and multi-stage
//! `test_suite` case execution (`inlist` and `inlist_runner`).

use std::path::Path;

use flexmesa_star::{
    parse_rn_stages, run_test_suite_case, verify_photo_restart_parity,
    InlistBundle, RnStage,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::star_helpers_py::to_py_err;
use crate::star_py::PyStarModel;

/// Parses a MESA `inlist` file (recursively resolving all `read_extra_*_inlist`
/// includes) and returns a summary dictionary of parsed controls.
#[pyfunction]
pub fn star_parse_inlist_bundle<'py>(
    py: Python<'py>,
    inlist_path: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let bundle =
        InlistBundle::from_file(Path::new(inlist_path)).map_err(to_py_err)?;
    let dict = PyDict::new(py);
    dict.set_item("entry_inlist", &bundle.entry_inlist)?;
    dict.set_item("create_pre_ms_model", bundle.job.create_pre_ms_model)?;
    dict.set_item("create_initial_model", bundle.job.create_initial_model)?;
    dict.set_item("create_rsp_model", bundle.job.create_rsp_model)?;
    dict.set_item("load_saved_model", bundle.job.load_saved_model)?;
    dict.set_item("saved_model_name", &bundle.job.saved_model_name)?;
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
    dict.set_item("rotation_flag", bundle.controls.rotation_flag)?;
    dict.set_item("rsp_flag", bundle.controls.rsp_flag)?;
    dict.set_item("rsp2_flag", bundle.controls.rsp2_flag)?;
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

/// Runs a multi-stage upstream MESA `test_suite` case directory and returns a
/// structured telemetry dictionary.
#[pyfunction]
#[pyo3(signature = (
    case_dir,
    skip_optional=true,
    max_steps_per_stage=None,
    verify_restart=false
))]
pub fn star_run_test_suite_case<'py>(
    py: Python<'py>,
    case_dir: &str,
    skip_optional: bool,
    max_steps_per_stage: Option<usize>,
    verify_restart: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let rep = py
        .detach(|| {
            run_test_suite_case(
                Path::new(case_dir),
                skip_optional,
                max_steps_per_stage,
                verify_restart,
            )
        })
        .map_err(to_py_err)?;

    let out = PyDict::new(py);
    out.set_item("case_name", &rep.case_name)?;
    out.set_item("total_steps", rep.total_steps)?;
    out.set_item("total_newton_iterations", rep.total_newton_iterations)?;
    out.set_item("total_retries", rep.total_retries)?;
    out.set_item("total_runtime_sec", rep.total_runtime_sec)?;
    out.set_item("photo_restart_verified", rep.photo_restart_verified)?;

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

/// Registers `E2E-0` runner functions on `flexmesa._rust`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(star_parse_inlist_bundle, m)?)?;
    m.add_function(wrap_pyfunction!(star_parse_rn_stages, m)?)?;
    m.add_function(wrap_pyfunction!(star_verify_photo_restart, m)?)?;
    m.add_function(wrap_pyfunction!(star_run_test_suite_case, m)?)?;
    Ok(())
}
