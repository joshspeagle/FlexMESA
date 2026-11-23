// SPDX-License-Identifier: LGPL-3.0-or-later
//! PyO3 adapter for an `astero` case's own `set_param` (its
//! `run_star_extras.f90`), supplied by test-side extras.
//!
//! The Python object implements
//! `set_param(snapshot: dict, name: str, val: float) -> dict | None`. The
//! snapshot holds the `astero_def` values a `set_param` reads and the current
//! `&star_job` composition fields; the returned dictionary lists the
//! `&star_job` / `&controls` assignments to make, applied in order.

use flexmesa_astero::{AsteroSearchControls, AsteroSetParam};
use flexmesa_star::controls::StarError;
use flexmesa_star::inlist::InlistBundle;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Wraps a Python object exposing `set_param(snapshot, name, val)`.
pub struct PyAsteroSetParamAdapter {
    py_obj: Py<PyAny>,
}

impl PyAsteroSetParamAdapter {
    /// Builds the adapter.
    ///
    /// # Errors
    /// Returns `ValueError` when `obj` has no `set_param` method.
    pub fn new(py: Python<'_>, obj: Py<PyAny>) -> PyResult<Self> {
        if !obj.bind(py).hasattr("set_param")? {
            return Err(PyValueError::new_err(
                "astero extras must implement set_param(snapshot, name, val)",
            ));
        }
        Ok(Self { py_obj: obj })
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
            "relax_initial_mass" => j.relax_mass = val.extract()?,
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

impl AsteroSetParam for PyAsteroSetParamAdapter {
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
}

/// Wraps optional test-side astero extras for
/// `run_astero_test_suite_case_with_set_param`.
///
/// # Errors
/// Returns `ValueError` when `extras` has no `set_param` method.
pub fn adapter_from_extras(
    py: Python<'_>,
    extras: Option<Py<PyAny>>,
) -> PyResult<Option<PyAsteroSetParamAdapter>> {
    extras
        .map(|obj| PyAsteroSetParamAdapter::new(py, obj))
        .transpose()
}
