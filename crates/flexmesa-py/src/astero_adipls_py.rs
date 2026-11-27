// SPDX-License-Identifier: GPL-3.0-or-later
//! Python access to astero's ADIPLS interface (`astero/private/adipls_support.f90`,
//! ported in `flexmesa_astero::adipls_support`; `#171`, A3b) for test-side
//! extras: `AsteroAdipls` owns the ADIPLS state of a run, reusing
//! `StarModelHandle` from `astero_gyre_py`.

use std::path::Path;

use flexmesa_astero::adipls_support::AsteroAdipls;
use flexmesa_astero::gyre_support::OscResult;
use flexmesa_astero::parse_astero_search_controls;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::astero_gyre_py::PyStarModelHandle;

fn osc_results_to_tuples(
    results: Vec<OscResult>,
) -> Vec<(i32, i32, i32, f64, f64, f64)> {
    results
        .into_iter()
        .map(|r| {
            (r.el, r.order, r.em, r.inertia, r.cyclic_freq, r.growth_rate)
        })
        .collect()
}

/// `run_adipls` startup and `do_adipls_get_one_el_info` for one astero run.
#[pyclass(name = "AsteroAdipls", module = "flexmesa._rust", unsendable)]
pub struct PyAsteroAdipls {
    inner: AsteroAdipls,
}

#[pymethods]
impl PyAsteroAdipls {
    /// Reads the case's `&astero_search_controls` and initializes ADIPLS from
    /// `adipls.c.pruned.in`, also parsing `redistrb.c.pruned.in` when
    /// `do_redistribute_mesh` is `.true.` (`astero/private/astero_run_support.f90:117-124`).
    #[new]
    fn new(case_dir: &str) -> PyResult<Self> {
        let dir = Path::new(case_dir);
        let ctrl = parse_astero_search_controls(dir)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let inner = AsteroAdipls::init_with_redistribute(
            dir,
            &ctrl,
            ctrl.do_redistribute_mesh,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Evaluates `do_adipls_get_one_el_info`
    /// (`astero/private/adipls_support.f90:49-164`) for degree `el` over
    /// `[nu1, nu2]` microHz, returning
    /// `(el, order, em, inertia, cyclic_freq, growth_rate)` per mode sorted by
    /// ascending frequency.
    ///
    /// `add_center_point` and `keep_surface_point` preserve the
    /// `adipls_get_one_el_info` call signature
    /// (`astero/public/astero_lib.f90:67-87`); upstream
    /// `store_model_for_adipls` (`astero/private/adipls_support.f90:217-244`)
    /// shadows them with `&astero_search_controls` defaults (`#480`).
    fn get_one_el_info(
        &mut self,
        model: &PyStarModelHandle,
        el: usize,
        nu1: f64,
        nu2: f64,
        iscan: i32,
        r_cm: f64,
        g_cgs: f64,
        m_g: f64,
        add_center_point: bool,
        keep_surface_point: bool,
        add_atmosphere: bool,
        do_redistribute_mesh: bool,
        store_for_adipls: bool,
    ) -> PyResult<Vec<(i32, i32, i32, f64, f64, f64)>> {
        let _ = (add_center_point, keep_surface_point);
        let l = i32::try_from(el).map_err(|_| {
            PyValueError::new_err(format!("degree el={el} overflows i32"))
        })?;
        let res = self
            .inner
            .do_adipls_get_one_el_info(
                &model.model,
                l,
                nu1,
                nu2,
                iscan,
                r_cm,
                g_cgs,
                m_g,
                0.0,
                add_atmosphere,
                do_redistribute_mesh,
                store_for_adipls,
                0,
                1.0,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(osc_results_to_tuples(res))
    }

    /// Evaluates `get_one_el_info` (`astero/private/astero_support.f90:55-315`)
    /// using photospheric geometry and `sig_fac_astero_support`, returning
    /// `(el, order, em, inertia, cyclic_freq, growth_rate)` sorted by ascending
    /// frequency.
    #[allow(clippy::too_many_arguments)]
    fn get_astero_one_el_info(
        &mut self,
        model: &PyStarModelHandle,
        el: usize,
        nu1: f64,
        nu2: f64,
        iscan: i32,
        delta_nu_model: f64,
        store_for_adipls: bool,
    ) -> PyResult<Vec<(i32, i32, i32, f64, f64, f64)>> {
        let res = self
            .inner
            .get_one_el_info(
                &model.model,
                el,
                nu1,
                nu2,
                iscan,
                delta_nu_model,
                store_for_adipls,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(osc_results_to_tuples(res))
    }
}

/// Registers `AsteroAdipls` on `flexmesa._rust`.
///
/// # Errors
/// Returns [`PyErr`] if registration fails.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAsteroAdipls>()?;
    Ok(())
}
