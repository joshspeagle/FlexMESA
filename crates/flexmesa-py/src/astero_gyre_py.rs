// SPDX-License-Identifier: GPL-3.0-or-later
//! Python access to astero's GYRE interface (`astero/private/gyre_support.f90`,
//! ported in `flexmesa_astero::gyre_support`; #170, G3c) for test-side
//! extras: `AsteroGyre` owns the GYRE state of a run, and
//! `StarModelHandle` carries the live star model into it.

use std::path::Path;

use flexmesa_astero::gyre_support::AsteroGyre;
use flexmesa_astero::parse_astero_search_controls;
use flexmesa_star::model::StarModel;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// A copy of the star model at the hook call, handed to extras whose object
/// sets `wants_star_model = True`.
#[pyclass(name = "StarModelHandle", module = "flexmesa._rust")]
pub struct PyStarModelHandle {
    pub(crate) model: StarModel,
}

#[pymethods]
impl PyStarModelHandle {
    /// `star_get_pulse_data(id, 'GYRE', ...)` on the model: `(global_data,
    /// point_data)` with one row of columns per point, centre first.
    fn gyre_pulse_data(
        &self,
        add_center_point: bool,
        keep_surface_point: bool,
        add_atmosphere: bool,
    ) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
        flexmesa_star::pulse_gyre_data::star_get_pulse_data_gyre(
            &self.model,
            add_center_point,
            keep_surface_point,
            add_atmosphere,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// `star_export_pulse_data(id, 'FGONG', filename, ...)` on the model:
    /// the text upstream writes to `filename`
    /// (`star/private/pulse.f90:43-86`).
    fn fgong_pulse_text(
        &self,
        add_center_point: bool,
        keep_surface_point: bool,
        add_atmosphere: bool,
    ) -> PyResult<String> {
        flexmesa_star::export_fgong(
            &self.model,
            add_center_point,
            keep_surface_point,
            add_atmosphere,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// The model number.
    #[getter]
    fn model_number(&self) -> usize {
        self.model.model_number
    }
}

impl PyStarModelHandle {
    /// Wraps a copy of `model`.
    #[must_use]
    pub fn new(model: &StarModel) -> Self {
        Self {
            model: model.clone(),
        }
    }
}

/// `init_gyre` and `astero_gyre_get_modes` for one astero run.
#[pyclass(name = "AsteroGyre", module = "flexmesa._rust", unsendable)]
pub struct PyAsteroGyre {
    inner: AsteroGyre,
}

#[pymethods]
impl PyAsteroGyre {
    /// Reads the case's `&astero_search_controls` and runs `init_gyre` on
    /// its `gyre_input_file` (`astero_run_support.f90:109-115`).
    #[new]
    fn new(case_dir: &str, mesa_dir: &str) -> PyResult<Self> {
        let dir = Path::new(case_dir);
        let ctrl = parse_astero_search_controls(dir)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let inner = AsteroGyre::init(dir, &ctrl, mesa_dir)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// `astero_gyre_get_modes(id, el, store_model, ierr)`
    /// (`astero/public/astero_lib.f90:90-101`): the stored
    /// `(el, order, em, inertia, cyclic_freq, growth_rate)` of each mode.
    fn get_modes(
        &mut self,
        model: &PyStarModelHandle,
        el: i32,
        store_model: bool,
    ) -> PyResult<Vec<(i32, i32, i32, f64, f64, f64)>> {
        let res = self
            .inner
            .get_modes(&model.model, el, store_model)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(res
            .into_iter()
            .map(|r| {
                (r.el, r.order, r.em, r.inertia, r.cyclic_freq, r.growth_rate)
            })
            .collect())
    }
}

/// Registers the classes.
///
/// # Errors
/// Returns [`PyErr`] if registration fails.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStarModelHandle>()?;
    m.add_class::<PyAsteroGyre>()?;
    Ok(())
}
