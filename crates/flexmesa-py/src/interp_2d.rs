// SPDX-License-Identifier: LGPL-3.0-or-later
//! Tier B (`interp_2d`) `PyO3` bindings for 2D bicubic and piecewise-monotonic
//! surface interpolation.
//!
//! Exposes contiguous `float64` NumPy array bindings for:
//! - `interp_mkbicub_db`: PSPLINE / MESA 2D bicubic spline coefficient builder.
//! - `interp_evbicub_db`: scalar 6-component bicubic spline evaluator
//!   `[f, df_dx, df_dy, d2f_dx2, d2f_dy2, d2f_dxdy]`.
//! - `interp_evbicub_batch`: multi-zone batch bicubic spline evaluator with
//!   persistent 4-thread `rayon` parallelism.
//! - `interp_mkbipm_db` and `interp_evbipm_db`: 2D Steffen (1990)
//!   piecewise-monotonic cubic setup and evaluation.
//!
//! # Grid Layout and Boundary Conditions
//! Grid values `z_flat` have length `nx * ny` in row-major `(nx, ny)` order
//! (`z[ix * ny + iy] = f(x[ix], y[iy])`). Boundary condition codes `ibc*`
//! follow upstream PSPLINE / MESA (`-1..=7`, with `0` = not-a-knot default).

use flexmesa_bench::bicub_fast::eval_bicub_prevalidated;
use flexmesa_interp2d::BicubicBc;
use numpy::{
    PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::eval_batch_2d;
use crate::interp_1d::slice_to_coeffs4;

/// Constructs 2D bicubic spline coefficients `(nx, ny, 4)` and spacing flags
/// `(ilinx, iliny)`.
///
/// Matches upstream MESA `interp_2d_lib_db::interp_mkbicub_db`.
///
/// # Arguments
/// - `x`, `y`: strictly ascending 1D knot arrays of length `nx` and `ny`.
/// - `z_flat`: flattened `(nx, ny)` function values of length `nx * ny`.
/// - `ibcxmin`, `bcxmin`: lower `x` BC code (`-1..=7`) and values `(ny,)`.
/// - `ibcxmax`, `bcxmax`: upper `x` BC code (`-1..=7`) and values `(ny,)`.
/// - `ibcymin`, `bcymin`: lower `y` BC code (`-1..=7`) and values `(nx,)`.
/// - `ibcymax`, `bcymax`: upper `y` BC code (`-1..=7`) and values `(nx,)`.
#[pyfunction]
fn interp_mkbicub_db<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    z_flat: PyReadonlyArray1<'py, f64>,
    ibcxmin: i32,
    bcxmin: PyReadonlyArray1<'py, f64>,
    ibcxmax: i32,
    bcxmax: PyReadonlyArray1<'py, f64>,
    ibcymin: i32,
    bcymin: PyReadonlyArray1<'py, f64>,
    ibcymax: i32,
    bcymax: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray3<f64>>, bool, bool)> {
    let xs = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
    let ys = y
        .as_slice()
        .map_err(|_| PyValueError::new_err("y must be C-contiguous"))?;
    let zs = z_flat
        .as_slice()
        .map_err(|_| PyValueError::new_err("z must be C-contiguous"))?;
    let bxmin = bcxmin
        .as_slice()
        .map_err(|_| PyValueError::new_err("bcxmin must be C-contiguous"))?;
    let bxmax = bcxmax
        .as_slice()
        .map_err(|_| PyValueError::new_err("bcxmax must be C-contiguous"))?;
    let bymin = bcymin
        .as_slice()
        .map_err(|_| PyValueError::new_err("bcymin must be C-contiguous"))?;
    let bymax = bcymax
        .as_slice()
        .map_err(|_| PyValueError::new_err("bcymax must be C-contiguous"))?;
    let c_xmin = BicubicBc::from_code(ibcxmin)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let c_xmax = BicubicBc::from_code(ibcxmax)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let c_ymin = BicubicBc::from_code(ibcymin)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let c_ymax = BicubicBc::from_code(ibcymax)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (coeffs, ilinx, iliny) = flexmesa_interp2d::interp_mkbicub_db(
        xs, ys, zs, c_xmin, bxmin, c_xmax, bxmax, c_ymin, bymin, c_ymax, bymax,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (nx, ny) = (xs.len(), ys.len());
    let arr = PyArray1::from_slice(py, coeffs.as_flattened())
        .reshape([nx, ny, 4])?;
    Ok((arr, ilinx, iliny))
}

/// Evaluates a 2D bicubic spline at scalar `(xget, yget)`.
///
/// Returns the 6-element array `[f, df_dx, df_dy, d2f_dx2, d2f_dy2, d2f_dxdy]`
/// matching upstream MESA `interp_2d_lib_db::interp_evbicub_db` with
/// `ict = [1, 1, 1, 1, 1, 1]`.
#[pyfunction]
fn interp_evbicub_db(
    xget: f64,
    yget: f64,
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    ilinx: bool,
    iliny: bool,
    coeffs_flat: PyReadonlyArray1<'_, f64>,
) -> PyResult<[f64; 6]> {
    let xs = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
    let ys = y
        .as_slice()
        .map_err(|_| PyValueError::new_err("y must be C-contiguous"))?;
    let cf = coeffs_flat
        .as_slice()
        .map_err(|_| PyValueError::new_err("coeffs must be C-contiguous"))?;
    let coeffs = slice_to_coeffs4(cf)?;
    let ev = flexmesa_interp2d::interp_evbicub_db(
        xget, yget, xs, ys, ilinx, iliny, coeffs,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok([ev.f, ev.df_dx, ev.df_dy, ev.d2f_dx2, ev.d2f_dy2, ev.d2f_dxdy])
}

/// Evaluates a 2D bicubic spline over 1D query arrays `x_q, y_q` of shape
/// `(nz,)`.
///
/// Uses `eval_bicub_prevalidated` to amortize grid monotonicity checks and
/// dispatches across the persistent 4-thread `rayon` worker pool when
/// `parallel=True` and `nz >= 64`. Returns a 2D `(nz, 6)` array.
#[pyfunction]
#[pyo3(signature = (x_q, y_q, x, y, coeffs_flat, parallel=false))]
fn interp_evbicub_batch<'py>(
    py: Python<'py>,
    x_q: PyReadonlyArray1<'py, f64>,
    y_q: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    coeffs_flat: PyReadonlyArray1<'py, f64>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let xq = x_q
        .as_slice()
        .map_err(|_| PyValueError::new_err("x_q must be C-contiguous"))?;
    let yq = y_q
        .as_slice()
        .map_err(|_| PyValueError::new_err("y_q must be C-contiguous"))?;
    let xs = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
    let ys = y
        .as_slice()
        .map_err(|_| PyValueError::new_err("y must be C-contiguous"))?;
    let cf = coeffs_flat
        .as_slice()
        .map_err(|_| PyValueError::new_err("coeffs must be C-contiguous"))?;
    // Validate query length agreement, minimum 4x4 grid dimensions, strict
    // ascending grid spacing, and exact `(nx, ny, 4)` coefficient table size
    // before dispatching to `eval_bicub_prevalidated`.
    if xq.len() != yq.len() {
        return Err(PyValueError::new_err("x_q and y_q length mismatch"));
    }
    if xs.len() < 4 || ys.len() < 4 {
        return Err(PyValueError::new_err(
            "interp_evbicub_batch requires nx >= 4 and ny >= 4",
        ));
    }
    flexmesa_interp2d::splinck_db(xs, 1.0e-3, "x")
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    flexmesa_interp2d::splinck_db(ys, 1.0e-3, "y")
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let coeffs = slice_to_coeffs4(cf)?;
    if coeffs.len() != xs.len().saturating_mul(ys.len()) {
        return Err(PyValueError::new_err(
            "coeffs_flat length must equal nx * ny * 4",
        ));
    }
    let (xqv, yqv, xsv, ysv, cfv) = (
        xq.to_vec(),
        yq.to_vec(),
        xs.to_vec(),
        ys.to_vec(),
        coeffs.to_vec(),
    );
    eval_batch_2d(py, xqv.len(), parallel, 64, move |k| {
        eval_bicub_prevalidated(xqv[k], yqv[k], &xsv, &ysv, &cfv)
    })
}

/// Constructs 2D piecewise-monotonic cubic spline coefficients `(nx, ny, 4)`.
///
/// Matches upstream MESA `interp_2d_lib_db::interp_mkbipm_db` by fitting 1D
/// Steffen (1990) monotonic splines along `x` at every `y` slice.
#[pyfunction]
fn interp_mkbipm_db<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    z_flat: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let xs = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
    let ys = y
        .as_slice()
        .map_err(|_| PyValueError::new_err("y must be C-contiguous"))?;
    let zs = z_flat
        .as_slice()
        .map_err(|_| PyValueError::new_err("z must be C-contiguous"))?;
    let coeffs = flexmesa_interp2d::interp_mkbipm_db(xs, ys, zs)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (nx, ny) = (xs.len(), ys.len());
    PyArray1::from_slice(py, coeffs.as_flattened()).reshape([nx, ny, 4])
}

/// Evaluates a 2D piecewise-monotonic cubic spline at `(xget, yget)`.
///
/// Matches upstream MESA `interp_2d_lib_db::interp_evbipm_db` by evaluating
/// four bracketing `x`-splines and performing a 4-point Steffen (1990)
/// interpolation along `y`.
#[pyfunction]
fn interp_evbipm_db(
    xget: f64,
    yget: f64,
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    coeffs_flat: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let xs = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
    let ys = y
        .as_slice()
        .map_err(|_| PyValueError::new_err("y must be C-contiguous"))?;
    let cf = coeffs_flat
        .as_slice()
        .map_err(|_| PyValueError::new_err("coeffs must be C-contiguous"))?;
    let coeffs = slice_to_coeffs4(cf)?;
    flexmesa_interp2d::interp_evbipm_db(xget, yget, xs, ys, coeffs)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Registers Tier B `interp_2d` functions on `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(interp_mkbicub_db, m)?)?;
    m.add_function(wrap_pyfunction!(interp_evbicub_db, m)?)?;
    m.add_function(wrap_pyfunction!(interp_evbicub_batch, m)?)?;
    m.add_function(wrap_pyfunction!(interp_mkbipm_db, m)?)?;
    m.add_function(wrap_pyfunction!(interp_evbipm_db, m)?)?;
    Ok(())
}
