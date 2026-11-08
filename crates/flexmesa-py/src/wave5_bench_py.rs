// SPDX-License-Identifier: LGPL-3.0-or-later
//! Batch PyO3 bindings (`flexmesa._rust`) for the Wave 5 four-stage benchmark
//! workloads (`w5_diff_rot`, `w5_riemann_rsp2`, `w5_binary_rlof`, and
//! `w5_astero_seism`).

use std::sync::Arc;

use flexmesa_bench::{
    eval_astero_seism_workload, eval_binary_rlof_zone, eval_diff_rot_zone,
    RiemannRsp2Grid,
};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

/// Evaluates the `w5_diff_rot` benchmark profile across `nz` zones (`(nz,)`).
///
/// Dispatches across the persistent 4-thread worker team when `parallel` is
/// `true`.
#[pyfunction]
#[pyo3(signature = (nz, parallel=false))]
pub fn w5_eval_diff_rot_batch<'py>(
    py: Python<'py>,
    nz: usize,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let arr2 = crate::eval_batch_2d::<1, _>(py, nz, parallel, 16, move |k| {
        [eval_diff_rot_zone(k, nz)]
    })?;
    arr2.reshape([nz])
}

/// Evaluates the `w5_riemann_rsp2` benchmark profile across `nz` zones
/// (`(nz,)`).
///
/// Dispatches across the persistent 4-thread worker team when `parallel` is
/// `true`.
#[pyfunction]
#[pyo3(signature = (nz, parallel=false))]
pub fn w5_eval_riemann_rsp2_batch<'py>(
    py: Python<'py>,
    nz: usize,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let grid = Arc::new(RiemannRsp2Grid::new(nz));
    let arr2 = crate::eval_batch_2d::<1, _>(py, nz, parallel, 16, move |k| {
        [grid.eval_zone(k)]
    })?;
    arr2.reshape([nz])
}

/// Evaluates the `w5_binary_rlof` benchmark profile across `nz` binary
/// configurations (`(nz,)`).
///
/// Dispatches across the persistent 4-thread worker team when `parallel` is
/// `true`.
#[pyfunction]
#[pyo3(signature = (nz, parallel=false))]
pub fn w5_eval_binary_rlof_batch<'py>(
    py: Python<'py>,
    nz: usize,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let arr2 = crate::eval_batch_2d::<1, _>(py, nz, parallel, 16, move |k| {
        [eval_binary_rlof_zone(k, nz)]
    })?;
    arr2.reshape([nz])
}

/// Evaluates the `w5_astero_seism` benchmark scalar digest for `nz` zones.
#[pyfunction]
pub fn w5_eval_astero_seism(nz: usize) -> f64 {
    eval_astero_seism_workload(nz)
}

/// Registers the Wave 5 batch benchmark functions on `flexmesa._rust`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(w5_eval_diff_rot_batch, m)?)?;
    m.add_function(wrap_pyfunction!(w5_eval_riemann_rsp2_batch, m)?)?;
    m.add_function(wrap_pyfunction!(w5_eval_binary_rlof_batch, m)?)?;
    m.add_function(wrap_pyfunction!(w5_eval_astero_seism, m)?)?;
    Ok(())
}
