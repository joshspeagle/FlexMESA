// SPDX-License-Identifier: LGPL-3.0-or-later
//! Tier B (`interp_1d`) `PyO3` bindings for 1D piecewise-monotonic and
//! monotonicity-preserving cubic spline interpolation.
//!
//! Exposes contiguous `float64` NumPy array bindings for:
//! - `interp_pm`: Steffen (1990) piecewise-monotonic cubic spline builder.
//! - `interp_m3a`, `interp_m3b`, `interp_m3q`: Huynh (1993) average, superbee,
//!   and quartic monotonicity-preserving cubic spline builders.
//! - `interp_values` and `interp_values_and_slopes`: vectorized 1D cubic
//!   spline evaluators with boundary clamping.
//! - `integrate_values`: analytical piecewise-cubic sub-interval quadrature.
//!
//! # Coefficient Convention
//! Each spline builder returns an `(n, 4)` array where row `i` holds
//! `[c0, c1, c2, c3]` for knot $x_i$, representing the local Taylor expansion:
//! $$p_i(x) = c_{0,i} + \Delta x\,(c_{1,i} + \Delta x\,(
//!   c_{2,i} + \Delta x\,c_{3,i}))$$
//! for $\Delta x = x - x_i$.

use flexmesa_interp1d::{CubicCoeffs, Interp1dError};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Borrows a flat `&[f64]` slice of length `n * 4` zero-copy as `&[[f64; 4]]`.
///
/// Raises `ValueError` if `flat.len()` is not a multiple of 4.
pub(crate) fn slice_to_coeffs4(flat: &[f64]) -> PyResult<&[[f64; 4]]> {
    let (chunks, rem) = flat.as_chunks::<4>();
    if !rem.is_empty() {
        return Err(PyValueError::new_err(
            "coefficient buffer length must be a multiple of 4",
        ));
    }
    Ok(chunks)
}

/// Converts `Vec<[f64; 4]>` into a 2D `(n, 4)` `f64` NumPy array without
/// intermediate Rust heap copies via `as_flattened()`.
fn coeffs4_to_py2<'py>(
    py: Python<'py>,
    coeffs: Vec<CubicCoeffs>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n = coeffs.len();
    PyArray1::from_slice(py, coeffs.as_flattened()).reshape([n, 4])
}

/// Shared wrapper for 1D cubic spline coefficient builders `(x, y) -> (n, 4)`.
fn build_1d_coeffs<'py, F>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    builder: F,
) -> PyResult<Bound<'py, PyArray2<f64>>>
where
    F: FnOnce(&[f64], &[f64]) -> Result<Vec<CubicCoeffs>, Interp1dError>,
{
    let xs = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
    let ys = y
        .as_slice()
        .map_err(|_| PyValueError::new_err("y must be C-contiguous"))?;
    let coeffs =
        builder(xs, ys).map_err(|e| PyValueError::new_err(e.to_string()))?;
    coeffs4_to_py2(py, coeffs)
}

/// Unpacks `(init_x, coeffs_flat, x_query)` zero-copy into Rust slices.
fn unpack_eval_1d<'a>(
    init_x: &'a PyReadonlyArray1<'a, f64>,
    coeffs_flat: &'a PyReadonlyArray1<'a, f64>,
    x_query: &'a PyReadonlyArray1<'a, f64>,
) -> PyResult<(&'a [f64], &'a [[f64; 4]], &'a [f64])> {
    let xs = init_x
        .as_slice()
        .map_err(|_| PyValueError::new_err("init_x must be C-contiguous"))?;
    let cf = coeffs_flat
        .as_slice()
        .map_err(|_| PyValueError::new_err("coeffs must be C-contiguous"))?;
    let xq = x_query
        .as_slice()
        .map_err(|_| PyValueError::new_err("x_query must be C-contiguous"))?;
    let coeffs = slice_to_coeffs4(cf)?;
    Ok((xs, coeffs, xq))
}

/// Builds Steffen (1990) piecewise-monotonic cubic spline coefficients
/// `(n, 4)`.
///
/// Matches upstream MESA `interp_1d_pm::interp_pm`. Requires at least 2 knots
/// with strictly monotonic abscissas `x`.
#[pyfunction]
fn interp_pm<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    build_1d_coeffs(py, x, y, flexmesa_interp1d::interp_pm)
}

/// Builds Huynh (1993) `interp_m3a` (average limiter) cubic spline
/// coefficients `(n, 4)`.
///
/// Matches upstream MESA `interp_1d_mp::interp_m3a`. Requires at least 4 knots
/// with strictly monotonic abscissas `x`.
#[pyfunction]
fn interp_m3a<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    build_1d_coeffs(py, x, y, flexmesa_interp1d::interp_m3a)
}

/// Builds Huynh (1993) `interp_m3b` (superbee limiter) cubic spline
/// coefficients `(n, 4)`.
///
/// Matches upstream MESA `interp_1d_mp::interp_m3b`. Requires at least 4 knots
/// with strictly monotonic abscissas `x`.
#[pyfunction]
fn interp_m3b<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    build_1d_coeffs(py, x, y, flexmesa_interp1d::interp_m3b)
}

/// Builds Huynh (1993) `interp_m3q` (quartic limiter) cubic spline
/// coefficients `(n, 4)`.
///
/// Matches upstream MESA `interp_1d_mp::interp_m3q`. Requires at least 4 knots
/// with strictly monotonic abscissas `x`.
#[pyfunction]
fn interp_m3q<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    build_1d_coeffs(py, x, y, flexmesa_interp1d::interp_m3q)
}

/// Evaluates a 1D cubic spline at query points `x_query`, clamping queries
/// outside `[init_x[0], init_x[-1]]` to the boundary knots.
///
/// Matches upstream MESA `interp_1d_lib::interp_values`.
#[pyfunction]
fn interp_values<'py>(
    py: Python<'py>,
    init_x: PyReadonlyArray1<'py, f64>,
    coeffs_flat: PyReadonlyArray1<'py, f64>,
    x_query: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (xs, coeffs, xq) = unpack_eval_1d(&init_x, &coeffs_flat, &x_query)?;
    let vals = flexmesa_interp1d::interp_values(xs, coeffs, xq)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, vals))
}

/// Evaluates both interpolated values and first derivatives of a 1D cubic
/// spline at `x_query`.
///
/// Matches upstream MESA `interp_1d_lib::interp_values_and_slopes`.
#[pyfunction]
fn interp_values_and_slopes<'py>(
    py: Python<'py>,
    init_x: PyReadonlyArray1<'py, f64>,
    coeffs_flat: PyReadonlyArray1<'py, f64>,
    x_query: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (xs, coeffs, xq) = unpack_eval_1d(&init_x, &coeffs_flat, &x_query)?;
    let (vals, slopes) =
        flexmesa_interp1d::interp_values_and_slopes(xs, coeffs, xq)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        PyArray1::from_vec(py, vals),
        PyArray1::from_vec(py, slopes),
    ))
}

/// Evaluates consecutive sub-interval definite integrals of a 1D cubic spline
/// over `x_query`.
///
/// Matches upstream MESA `interp_1d_lib::integrate_values`.
#[pyfunction]
fn integrate_values<'py>(
    py: Python<'py>,
    init_x: PyReadonlyArray1<'py, f64>,
    coeffs_flat: PyReadonlyArray1<'py, f64>,
    x_query: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (xs, coeffs, xq) = unpack_eval_1d(&init_x, &coeffs_flat, &x_query)?;
    let vals = flexmesa_interp1d::integrate_values(xs, coeffs, xq)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, vals))
}

/// Registers Tier B `interp_1d` functions on `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(interp_pm, m)?)?;
    m.add_function(wrap_pyfunction!(interp_m3a, m)?)?;
    m.add_function(wrap_pyfunction!(interp_m3b, m)?)?;
    m.add_function(wrap_pyfunction!(interp_m3q, m)?)?;
    m.add_function(wrap_pyfunction!(interp_values, m)?)?;
    m.add_function(wrap_pyfunction!(interp_values_and_slopes, m)?)?;
    m.add_function(wrap_pyfunction!(integrate_values, m)?)?;
    Ok(())
}
