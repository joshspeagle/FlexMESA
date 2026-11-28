// SPDX-License-Identifier: GPL-3.0-or-later
//! PyO3 adapter for an `astero` case's `set_param`,
//! `set_constraint_value`, and optional `extras_after_evolve`
//! (`run_star_extras.f90`), supplied by test-side extras.
//!
//! The Python object implements
//! `set_param(snapshot: dict, name: str, val: float) -> dict | None`. The
//! snapshot holds the `astero_def` values a `set_param` reads and the current
//! `&star_job` composition fields; the returned dictionary lists the
//! `&star_job` / `&controls` assignments to make, applied in order.
//!
//! It also implements
//! `set_constraint_value(snapshot: dict, name: str) -> float | str`. The
//! snapshot holds the model values the routine reads (`surface_h1`,
//! `surface_he3`, `surface_he4`, `mixing_type`, `r`, `Rsun`,
//! `Z_div_X_solar`). Returning the string `"star_get_history_output"` asks for
//! `star_get_history_output(s, name, ierr)`, the routine's `case default`.
//!
//! It may also implement `extras_after_evolve(snapshot: dict)`, receiving
//! `best_chi2` and surface/scalar model fields and returning `None`, a
//! custom termination code string (`"xtra1"`..`"xtra9"` or `"t_xtra1"`..
//! `"t_xtra9"`), or a dictionary with `"termination_code"` /
//! `"termination_code_slot"` and optional `"termination_code_str"`.

use std::path::Path;

use flexmesa_astero::{AsteroCaseHooks, AsteroSearchControls};
use flexmesa_star::controls::StarError;
use flexmesa_star::inlist::InlistBundle;
use flexmesa_star::model::StarModel;
use flexmesa_star::report_utils::{surface_avg_x_by_name, surface_avg_x_slot};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Wraps a Python object exposing `set_param(snapshot, name, val)` and
/// `set_constraint_value(snapshot, name)`, plus optional `extras_after_evolve`.
pub struct PyAsteroCaseHooksAdapter {
    py_obj: Py<PyAny>,
    has_after_evolve: bool,
}

impl PyAsteroCaseHooksAdapter {
    /// Builds the adapter.
    ///
    /// # Errors
    /// Returns `ValueError` when `obj` has no `set_param` method.
    pub fn new(py: Python<'_>, obj: Py<PyAny>) -> PyResult<Self> {
        let o = obj.bind(py);
        if !o.hasattr("set_param")? || !o.hasattr("set_constraint_value")? {
            return Err(PyValueError::new_err(
                "astero extras must implement set_param(snapshot, name, val) \
                 and set_constraint_value(snapshot, name)",
            ));
        }
        let has_after_evolve = o.hasattr("extras_after_evolve")?;
        Ok(Self {
            py_obj: obj,
            has_after_evolve,
        })
    }
}

fn format_py_err(py: Python<'_>, err: &PyErr) -> String {
    err.value(py)
        .str()
        .map_or_else(|_| err.to_string(), |s| s.to_string_lossy().into_owned())
}

fn build_snapshot<'py>(
    py: Python<'py>,
    bundle: &InlistBundle,
    ctrl: &AsteroSearchControls,
) -> PyResult<Bound<'py, PyDict>> {
    let snap = PyDict::new(py);
    // `astero_def` module variables (`astero/public/astero_def.f90`).
    snap.set_item("Y_frac_he3", ctrl.y_frac_he3)?;
    snap.set_item("Z_div_X_solar", ctrl.z_div_x_solar)?;
    snap.set_item("Y_depends_on_Z", ctrl.y_depends_on_z)?;
    snap.set_item("dYdZ", ctrl.dydz)?;
    snap.set_item("Y0", ctrl.y0)?;
    snap.set_item("f0_ov_div_f_ov", ctrl.f0_ov_div_f_ov)?;
    // `s% job% initial_*`, read back by `set_param('initial_FeH')`.
    snap.set_item("initial_h1", bundle.job.initial_h1)?;
    snap.set_item("initial_h2", bundle.job.initial_h2)?;
    snap.set_item("initial_he3", bundle.job.initial_he3)?;
    snap.set_item("initial_he4", bundle.job.initial_he4)?;
    Ok(snap)
}

fn apply_updates(
    bundle: &mut InlistBundle,
    updates: &Bound<'_, PyDict>,
) -> PyResult<()> {
    for (key_obj, val) in updates.iter() {
        let key: String = key_obj.extract()?;
        let c = &mut bundle.controls;
        let j = &mut bundle.job;
        match key.as_str() {
            "new_mass" => j.new_mass = val.extract()?,
            "relax_initial_mass" => j.relax_initial_mass = val.extract()?,
            "initial_mass" => c.initial_mass = val.extract()?,
            "initial_h1" => j.initial_h1 = val.extract()?,
            "initial_h2" => j.initial_h2 = val.extract()?,
            "initial_he3" => j.initial_he3 = val.extract()?,
            "initial_he4" => j.initial_he4 = val.extract()?,
            "set_uniform_initial_composition" => {
                j.set_uniform_initial_composition = val.extract()?;
            }
            "mixing_length_alpha" => c.mixing_length_alpha = val.extract()?,
            "overshoot_scheme(1)" => {
                c.mix_wind.overshoot_sets[0].scheme = val.extract()?;
            }
            "overshoot_zone_type(1)" => {
                c.mix_wind.overshoot_sets[0].zone_type = val.extract()?;
            }
            "overshoot_zone_loc(1)" => {
                c.mix_wind.overshoot_sets[0].zone_loc = val.extract()?;
            }
            "overshoot_bdy_loc(1)" => {
                c.mix_wind.overshoot_sets[0].bdy_loc = val.extract()?;
            }
            "overshoot_f(1)" => {
                c.mix_wind.overshoot_sets[0].f = val.extract()?;
            }
            "overshoot_f0(1)" => {
                c.mix_wind.overshoot_sets[0].f0 = val.extract()?;
            }
            "alpha_semiconvection" => {
                c.alpha_semiconvection = val.extract()?;
            }
            "thermohaline_coeff" => c.thermohaline_coeff = val.extract()?,
            "diffusion_class_factor(:)" => {
                let v: f64 = val.extract()?;
                c.diffusion.diffusion_class_factor.fill(v);
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported astero set_param update key '{other}'"
                )));
            }
        }
    }
    Ok(())
}

/// Model values `set_constraint_value` reads.
fn build_constraint_snapshot<'py>(
    py: Python<'py>,
    model: &StarModel,
    ctrl: &AsteroSearchControls,
) -> PyResult<Bound<'py, PyDict>> {
    let slot = |iso: &str| {
        model
            .iso_names
            .iter()
            .position(|n| n.eq_ignore_ascii_case(iso))
    };
    let snap = PyDict::new(py);
    // `s% surface_h1 = surface_avg_x(s, h1)` and the same for `he3`, `he4`
    // (`star/private/report.f90:260-264`).
    snap.set_item("surface_h1", surface_avg_x_slot(model, slot("h1")))?;
    snap.set_item("surface_he3", surface_avg_x_slot(model, slot("he3")))?;
    snap.set_item("surface_he4", surface_avg_x_slot(model, slot("he4")))?;
    let mixing_type: Vec<i32> =
        model.mixing_type.iter().map(|&m| m as i32).collect();
    snap.set_item("mixing_type", mixing_type)?;
    snap.set_item("r", model.r[..model.nz].to_vec())?;
    snap.set_item("Rsun", flexmesa_const::RSUN)?;
    snap.set_item("Z_div_X_solar", ctrl.z_div_x_solar)?;
    Ok(snap)
}

fn normalize_xtra_termination_code(raw: &str) -> Result<String, StarError> {
    let trimmed = raw.trim();
    let code = trimmed.strip_prefix("t_").unwrap_or(trimmed);
    let valid = code
        .strip_prefix("xtra")
        .and_then(|s| s.parse::<u8>().ok())
        .is_some_and(|n| (1..=9).contains(&n));
    if !valid {
        return Err(StarError::InvalidInput(format!(
            "extras_after_evolve: unsupported termination code '{raw}'; \
             expected 'xtra1'..'xtra9' or 't_xtra1'..'t_xtra9'"
        )));
    }
    Ok(code.to_string())
}

fn build_after_evolve_snapshot<'py>(
    py: Python<'py>,
    model: &StarModel,
    best_chi2: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let snap = PyDict::new(py);
    let photosphere_r = if model.nz > 0 {
        model.r[0] / flexmesa_const::R_SUN
    } else {
        0.0
    };
    let photosphere_l = if model.nz > 0 {
        model.lum[0] / flexmesa_const::L_SUN
    } else {
        0.0
    };
    snap.set_item("best_chi2", best_chi2)?;
    snap.set_item("model_number", model.model_number)?;
    snap.set_item("star_age", model.star_age)?;
    snap.set_item("star_mass", model.mstar / flexmesa_const::M_SUN)?;
    snap.set_item("teff", model.teff)?;
    snap.set_item("photosphere_r", photosphere_r)?;
    snap.set_item("photosphere_l", photosphere_l)?;
    snap.set_item("surface_h1", surface_avg_x_by_name(model, "h1"))?;
    snap.set_item("surface_he3", surface_avg_x_by_name(model, "he3"))?;
    snap.set_item("surface_he4", surface_avg_x_by_name(model, "he4"))?;
    Ok(snap)
}

/// Adds the `astero_def` variables that a case's `extras_after_evolve` reads
/// to move the track's `best_model_*` files
/// (`astero/test_suite/fast_scan_grid/src/run_star_extras.f90:205-244`):
/// `astero_results_directory` (resolved against the run's output root),
/// `next_sample_number` (`sample_number + 1`), `num_digits`,
/// `sample_results_prefix` and the four `best_model_*_filename` values.
fn add_sample_files_snapshot(
    snap: &Bound<'_, PyDict>,
    ctrl: &AsteroSearchControls,
    results_dir: &Path,
    next_sample_number: usize,
) -> PyResult<()> {
    snap.set_item(
        "astero_results_directory",
        results_dir.to_string_lossy().into_owned(),
    )?;
    snap.set_item("next_sample_number", next_sample_number)?;
    snap.set_item("num_digits", ctrl.num_digits)?;
    snap.set_item("sample_results_prefix", &ctrl.sample_results_prefix)?;
    snap.set_item(
        "best_model_fgong_filename",
        &ctrl.best_model_fgong_filename,
    )?;
    snap.set_item("best_model_gyre_filename", &ctrl.best_model_gyre_filename)?;
    snap.set_item(
        "best_model_profile_filename",
        &ctrl.best_model_profile_filename,
    )?;
    snap.set_item(
        "best_model_save_model_filename",
        &ctrl.best_model_save_model_filename,
    )?;
    Ok(())
}

fn parse_after_evolve_result(
    res: &Bound<'_, PyAny>,
) -> Result<Option<String>, StarError> {
    if res.is_none() {
        return Ok(None);
    }
    if let Ok(raw) = res.extract::<String>() {
        return normalize_xtra_termination_code(&raw).map(Some);
    }
    let Ok(dict) = res.cast::<PyDict>() else {
        let ty = res
            .get_type()
            .name()
            .map_or_else(|_| String::from("unknown"), |n| n.to_string());
        return Err(StarError::InvalidInput(format!(
            "extras_after_evolve: expected None, str, or dict, got {ty}"
        )));
    };
    for (key_obj, _) in dict.iter() {
        let key: String = key_obj.extract().map_err(|_| {
            StarError::InvalidInput(String::from(
                "extras_after_evolve: dict keys must be strings",
            ))
        })?;
        match key.as_str() {
            "termination_code"
            | "termination_code_slot"
            | "termination_code_str" => {}
            other => {
                return Err(StarError::InvalidInput(format!(
                    "extras_after_evolve: unknown return dict key '{other}'"
                )));
            }
        }
    }
    let label = if let Some(v) =
        dict.get_item("termination_code_str").map_err(|e| {
            StarError::InvalidInput(format!("extras_after_evolve: {e}"))
        })? {
        if v.is_none() {
            None
        } else {
            let s: String = v.extract().map_err(|_| {
                StarError::InvalidInput(String::from(
                    "extras_after_evolve: 'termination_code_str' must be a str",
                ))
            })?;
            if s.trim().is_empty() {
                return Err(StarError::InvalidInput(String::from(
                    "extras_after_evolve: 'termination_code_str' \
                     must not be empty",
                )));
            }
            Some(s)
        }
    } else {
        None
    };

    let mut chosen: Option<String> = None;
    for key in ["termination_code_slot", "termination_code"] {
        if let Some(v) = dict.get_item(key).map_err(|e| {
            StarError::InvalidInput(format!("extras_after_evolve: {e}"))
        })? {
            if v.is_none() {
                continue;
            }
            let raw: String = v.extract().map_err(|_| {
                StarError::InvalidInput(format!(
                    "extras_after_evolve: '{key}' must be a str"
                ))
            })?;
            let norm = normalize_xtra_termination_code(&raw)?;
            if let Some(ref prev) = chosen {
                if *prev != norm {
                    return Err(StarError::InvalidInput(format!(
                        "extras_after_evolve: conflicting termination codes \
                         '{prev}' vs '{norm}'"
                    )));
                }
            } else {
                chosen = Some(norm);
            }
        }
    }
    if chosen.is_none() && label.is_some() {
        return Err(StarError::InvalidInput(String::from(
            "extras_after_evolve: 'termination_code_str' requires \
             'termination_code' or 'termination_code_slot'",
        )));
    }
    Ok(chosen)
}

impl AsteroCaseHooks for PyAsteroCaseHooksAdapter {
    fn set_constraint_value(
        &mut self,
        model: &StarModel,
        ctrl: &AsteroSearchControls,
        name: &str,
    ) -> Result<f64, StarError> {
        let history = Python::attach(|py| -> Result<Option<f64>, StarError> {
            let to_err = |e: &PyErr| {
                StarError::InvalidInput(format!(
                    "set_constraint_value: {}",
                    format_py_err(py, e)
                ))
            };
            let snap = build_constraint_snapshot(py, model, ctrl)
                .map_err(|e| to_err(&e))?;
            let res = self
                .py_obj
                .bind(py)
                .call_method1("set_constraint_value", (snap, name))
                .map_err(|e| to_err(&e))?;
            if let Ok(tag) = res.extract::<String>() {
                if tag == "star_get_history_output" {
                    return Ok(None);
                }
                return Err(StarError::InvalidInput(format!(
                    "set_constraint_value: unexpected string '{tag}'"
                )));
            }
            res.extract::<f64>().map(Some).map_err(|e| to_err(&e))
        })?;
        match history {
            Some(v) => Ok(v),
            // `case default`: `star_get_history_output`, with
            // `mesa_error` when it fails.
            None => flexmesa_star::star_get_history_output(model, name),
        }
    }

    fn set_param(
        &mut self,
        bundle: &mut InlistBundle,
        ctrl: &AsteroSearchControls,
        name: &str,
        val: f64,
    ) -> Result<(), StarError> {
        Python::attach(|py| {
            let to_err = |e: &PyErr| {
                StarError::InvalidInput(format!(
                    "set_param: {}",
                    format_py_err(py, e)
                ))
            };
            let snap =
                build_snapshot(py, bundle, ctrl).map_err(|e| to_err(&e))?;
            let res = self
                .py_obj
                .bind(py)
                .call_method1("set_param", (snap, name, val))
                .map_err(|e| to_err(&e))?;
            if let Ok(d) = res.cast::<PyDict>() {
                apply_updates(bundle, d).map_err(|e| to_err(&e))?;
            } else if !res.is_none() {
                return Err(StarError::InvalidInput(String::from(
                    "set_param: hook must return None or a dict of updates",
                )));
            }
            Ok(())
        })
    }

    fn extras_after_evolve(
        &mut self,
        model: &StarModel,
        best_chi2: f64,
    ) -> Result<Option<String>, StarError> {
        if !self.has_after_evolve {
            return Ok(None);
        }
        Python::attach(|py| {
            let to_err = |e: &PyErr| {
                StarError::InvalidInput(format!(
                    "extras_after_evolve: {}",
                    format_py_err(py, e)
                ))
            };
            let snap = build_after_evolve_snapshot(py, model, best_chi2)
                .map_err(|e| to_err(&e))?;
            let res = self
                .py_obj
                .bind(py)
                .call_method1("extras_after_evolve", (snap,))
                .map_err(|e| to_err(&e))?;
            parse_after_evolve_result(&res)
        })
    }

    fn extras_after_evolve_for_sample(
        &mut self,
        model: &StarModel,
        best_chi2: f64,
        ctrl: &AsteroSearchControls,
        results_dir: &Path,
        next_sample_number: usize,
    ) -> Result<Option<String>, StarError> {
        if !self.has_after_evolve {
            return Ok(None);
        }
        Python::attach(|py| {
            let to_err = |e: &PyErr| {
                StarError::InvalidInput(format!(
                    "extras_after_evolve: {}",
                    format_py_err(py, e)
                ))
            };
            let snap = build_after_evolve_snapshot(py, model, best_chi2)
                .map_err(|e| to_err(&e))?;
            add_sample_files_snapshot(
                &snap,
                ctrl,
                results_dir,
                next_sample_number,
            )
            .map_err(|e| to_err(&e))?;
            let res = self
                .py_obj
                .bind(py)
                .call_method1("extras_after_evolve", (snap,))
                .map_err(|e| to_err(&e))?;
            parse_after_evolve_result(&res)
        })
    }
}

/// Wraps optional test-side astero extras for
/// `run_astero_test_suite_case_with_case_hooks`.
///
/// # Errors
/// Returns `ValueError` when `extras` has no `set_param` method.
pub fn adapter_from_extras(
    py: Python<'_>,
    extras: Option<Py<PyAny>>,
) -> PyResult<Option<PyAsteroCaseHooksAdapter>> {
    extras
        .map(|obj| PyAsteroCaseHooksAdapter::new(py, obj))
        .transpose()
}
