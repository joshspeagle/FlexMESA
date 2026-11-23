// SPDX-License-Identifier: LGPL-3.0-or-later
//! `PyO3` bindings for `flexmesa trace` (`star`, `binary`, `astero` incremental
//! MESA log tracers) and the 6-line MESA ASCII log reader (`#143`, `#163`).

use std::collections::HashSet;
use std::path::Path;

use flexmesa_astero::{
    run_astero_test_suite_case_with_case_hooks, AsteroCaseHooks,
};
use flexmesa_binary::{
    run_binary_test_suite_case_with_extras, BinaryLogsOptions,
};
use flexmesa_star::inlist_runner::{
    run_test_suite_case_with_extras, StarLogsOptions,
};
use flexmesa_star::mesa_log_reader::{read_mesa_log_file, MesaHeaderValue};
use flexmesa_star::InlistLoadOptions;
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::extras_py::{run_with_binary_extras, run_with_star_extras};
use crate::star_e2e_py::{binary_report_to_pydict, case_report_to_pydict};
use crate::star_helpers_py::to_py_err;

fn parse_opt_mode(
    mode: Option<&str>,
) -> PyResult<Option<flexmesa_math::NumericsMode>> {
    mode.map(|raw| {
        raw.parse::<flexmesa_math::NumericsMode>()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
    .transpose()
}

fn run_in_opt_mode<T, F>(mode: Option<flexmesa_math::NumericsMode>, f: F) -> T
where
    F: FnOnce() -> T,
{
    if let Some(m) = mode {
        flexmesa_math::with_numerics_mode(m, f)
    } else {
        f()
    }
}

/// Runs an upstream `star/test_suite/<case>` directory while streaming
/// incremental MESA-format `LOGS/history.data` (and optional profiles).
#[pyfunction]
#[pyo3(signature = (
    case_dir,
    out_target,
    max_steps_per_stage=None,
    skip_optional=true,
    respect_intervals=false,
    write_profiles=false,
    is_logs_dir=true,
    numerics_mode=None,
    allow_unsupported_controls=false,
    extras=None
))]
#[allow(clippy::too_many_arguments)]
pub fn trace_star_case<'py>(
    py: Python<'py>,
    case_dir: &str,
    out_target: &str,
    max_steps_per_stage: Option<usize>,
    skip_optional: bool,
    respect_intervals: bool,
    write_profiles: bool,
    is_logs_dir: bool,
    numerics_mode: Option<&str>,
    allow_unsupported_controls: bool,
    extras: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let mode = parse_opt_mode(numerics_mode)?;
    let load_opts = InlistLoadOptions {
        allow_unsupported_controls,
    };
    let opts = StarLogsOptions::from_target(
        Path::new(out_target),
        is_logs_dir,
        respect_intervals,
        write_profiles,
    );
    let rep = run_with_star_extras(py, extras, |ex| {
        run_in_opt_mode(mode, || {
            run_test_suite_case_with_extras(
                Path::new(case_dir),
                skip_optional,
                max_steps_per_stage,
                false,
                opts,
                load_opts,
                ex,
            )
        })
    })?;
    case_report_to_pydict(py, &rep)
}

/// Runs an upstream `binary/test_suite/<case>` directory while streaming
/// incremental MESA-format `binary_history.data` and component `LOGS1/`/`LOGS2/`.
#[pyfunction]
#[pyo3(signature = (
    case_dir,
    out_target,
    max_steps=None,
    respect_intervals=false,
    is_logs_dir=true,
    numerics_mode=None,
    allow_unsupported_controls=false,
    extras=None
))]
#[allow(clippy::too_many_arguments)]
pub fn trace_binary_case<'py>(
    py: Python<'py>,
    case_dir: &str,
    out_target: &str,
    max_steps: Option<usize>,
    respect_intervals: bool,
    is_logs_dir: bool,
    numerics_mode: Option<&str>,
    allow_unsupported_controls: bool,
    extras: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let mode = parse_opt_mode(numerics_mode)?;
    let load_opts = InlistLoadOptions {
        allow_unsupported_controls,
    };
    let out_p = Path::new(out_target);
    let hist_override = (!respect_intervals).then_some(1);
    let opts = if is_logs_dir {
        BinaryLogsOptions {
            output_root: Some(out_p),
            explicit_binary_history_file: None,
            history_interval_override: hist_override,
            write_star_logs: true,
        }
    } else {
        BinaryLogsOptions {
            output_root: None,
            explicit_binary_history_file: Some(out_p),
            history_interval_override: hist_override,
            write_star_logs: false,
        }
    };
    let rep = run_with_binary_extras(py, extras, |ex| {
        run_in_opt_mode(mode, || {
            run_binary_test_suite_case_with_extras(
                Path::new(case_dir),
                max_steps,
                opts,
                load_opts,
                ex,
            )
        })
    })?;
    binary_report_to_pydict(py, &rep)
}

/// Runs an upstream `astero/test_suite/<case>` directory while streaming
/// incremental MESA-format `LOGS/history.data` and asteroseismic tables.
#[pyfunction]
#[pyo3(signature = (
    case_dir,
    out_target,
    max_steps_per_stage=None,
    skip_optional=true,
    respect_intervals=false,
    write_profiles=false,
    is_logs_dir=true,
    numerics_mode=None,
    allow_unsupported_controls=false,
    extras=None
))]
#[allow(clippy::too_many_arguments)]
pub fn trace_astero_case<'py>(
    py: Python<'py>,
    case_dir: &str,
    out_target: &str,
    max_steps_per_stage: Option<usize>,
    skip_optional: bool,
    respect_intervals: bool,
    write_profiles: bool,
    is_logs_dir: bool,
    numerics_mode: Option<&str>,
    allow_unsupported_controls: bool,
    extras: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let mode = parse_opt_mode(numerics_mode)?;
    let mut set_param =
        crate::astero_case_hooks_py::adapter_from_extras(py, extras)?;
    let load_opts = InlistLoadOptions {
        allow_unsupported_controls,
    };
    let rep = py
        .detach(|| {
            run_in_opt_mode(mode, || {
                let opts = StarLogsOptions::from_target(
                    Path::new(out_target),
                    is_logs_dir,
                    respect_intervals,
                    write_profiles,
                );
                run_astero_test_suite_case_with_case_hooks(
                    Path::new(case_dir),
                    skip_optional,
                    max_steps_per_stage,
                    false,
                    opts,
                    load_opts,
                    set_param.as_mut().map(|a| a as &mut dyn AsteroCaseHooks),
                )
            })
        })
        .map_err(to_py_err)?;
    case_report_to_pydict(py, &rep)
}

/// Reads a 6-line-header MESA ASCII table (`history.data`, `profileN.data`, or
/// `binary_history.data`) into a dictionary of headers and 1D `float64` arrays.
#[pyfunction]
pub fn mesa_read_log_file<'py>(
    py: Python<'py>,
    path: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let tbl = py
        .detach(|| read_mesa_log_file(Path::new(path)))
        .map_err(to_py_err)?;
    let out = PyDict::new(py);
    let hdr = PyDict::new(py);
    for (k, v) in &tbl.header {
        match v {
            MesaHeaderValue::Text(s) => hdr.set_item(k, s)?,
            MesaHeaderValue::Int(i) => hdr.set_item(k, *i)?,
            MesaHeaderValue::Float(f) => hdr.set_item(k, *f)?,
        }
    }
    out.set_item("header", hdr)?;
    out.set_item("column_names", &tbl.column_names)?;
    out.set_item("num_rows", tbl.num_rows)?;

    let cols = PyDict::new(py);
    let pos_list = PyList::empty(py);
    let mut seen_names = HashSet::with_capacity(tbl.column_names.len());
    for (name, vec) in tbl.column_names.iter().zip(tbl.column_values) {
        let arr = PyArray1::from_vec(py, vec);
        if seen_names.insert(name.as_str()) {
            cols.set_item(name, &arr)?;
        }
        pos_list.append(arr)?;
    }
    out.set_item("columns", cols)?;
    out.set_item("column_values", pos_list)?;
    Ok(out)
}

/// Registers incremental tracer and MESA log-reader functions on `flexmesa._rust`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trace_star_case, m)?)?;
    m.add_function(wrap_pyfunction!(trace_binary_case, m)?)?;
    m.add_function(wrap_pyfunction!(trace_astero_case, m)?)?;
    m.add_function(wrap_pyfunction!(mesa_read_log_file, m)?)?;
    Ok(())
}
