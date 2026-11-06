// SPDX-License-Identifier: LGPL-3.0-or-later
//! Tier A/B/C (`auto_diff`, `math`, `mtx`) `PyO3` bindings.
//!
//! Exposes scalar and contiguous `float64` NumPy array bindings for:
//! - `flexmesa_autodiff`: `eval_autodiff_order2_zone` and
//!   `eval_autodiff_order2_batch` (2nd-order bivariate forward-mode dual
//!   numbers with persistent `rayon` multi-core execution).
//! - `flexmesa_math`: `safe_sqrt`, `safe_log`, `safe_log10`, `exp10`, `pow`,
//!   `sinpi`, `cospi`, `tanpi`, `sgn`, `heaviside`, `str_to_vector`, plus
//!   vectorized 1D array variants (`safe_log_array`, `safe_log10_array`,
//!   `safe_sqrt_array`, `exp10_array`).
//! - `flexmesa_mtx`: `solve_block_tridiagonal`, `solve_block_tridiag_bcyclic`
//!   (Hirshman et al. 2010 odd-even block cyclic reduction), and
//!   `block_multiply_xa`.

use flexmesa_autodiff::AutoDiffReal2VarOrder2;
use flexmesa_bench::stage3_pool;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::eval_batch_2d;

/// Evaluates `f(x, y) = exp(0.1*x*y) + x^1.5 / (y + 2)` in
/// `AutoDiffReal2VarOrder2`.
///
/// Returns `[val, df_dx, df_dy, d2f_dx2, d2f_dxdy, d2f_dy2]`.
#[pyfunction]
fn eval_autodiff_order2_zone(x: f64, y: f64) -> [f64; 6] {
    let xv = AutoDiffReal2VarOrder2::var1(x);
    let yv = AutoDiffReal2VarOrder2::var2(y);
    let z = (xv * yv * 0.1).exp() + xv.powf(1.5) / (yv + 2.0);
    [
        z.val,
        z.d1val1,
        z.d1val2,
        z.d2val1,
        z.d1val1_d1val2,
        z.d2val2,
    ]
}

/// Evaluates `eval_autodiff_order2_zone` across 1D arrays `x, y` of shape
/// `(nz,)`.
///
/// Dispatches across the persistent 4-thread `rayon` pool when `parallel=True`
/// and `nz >= 64`.
#[pyfunction]
#[pyo3(signature = (x, y, parallel=false))]
fn eval_autodiff_order2_batch<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let xs = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
    let ys = y
        .as_slice()
        .map_err(|_| PyValueError::new_err("y must be C-contiguous"))?;
    if xs.len() != ys.len() {
        return Err(PyValueError::new_err("x and y length mismatch"));
    }
    eval_batch_2d(py, xs.len(), parallel, 64, |k| {
        eval_autodiff_order2_zone(xs[k], ys[k])
    })
}

/// Evaluates domain-clamped `safe_sqrt(x) = sqrt(max(x, 0))` on a scalar `f64`.
///
/// Matches upstream MESA `math_lib::safe_sqrt`.
#[pyfunction]
fn safe_sqrt(x: f64) -> f64 {
    flexmesa_math::safe_sqrt(x)
}

/// Evaluates domain-clamped `safe_log(x) = ln(max(x, 1e-99))` on a scalar
/// `f64`.
///
/// Matches upstream MESA `math_lib::safe_log`.
#[pyfunction]
fn safe_log(x: f64) -> f64 {
    flexmesa_math::safe_log(x)
}

/// Evaluates domain-clamped `safe_log10(x) = log10(max(x, 1e-99))` on a scalar
/// `f64`.
///
/// Matches upstream MESA `math_lib::safe_log10`.
#[pyfunction]
fn safe_log10(x: f64) -> f64 {
    flexmesa_math::safe_log10(x)
}

/// Evaluates base-10 exponential $10^x$ on a scalar `f64`.
///
/// Matches upstream MESA `math_lib::exp10`.
#[pyfunction]
fn exp10(x: f64) -> f64 {
    flexmesa_math::exp10(x)
}

/// Evaluates real power $x^y$ on scalar `f64` inputs.
///
/// Matches upstream MESA `math_lib::pow`.
#[pyfunction]
fn pow(x: f64, y: f64) -> f64 {
    flexmesa_math::pow(x, y)
}

/// Evaluates range-reduced $\sin(\pi x)$ on a scalar `f64`.
///
/// Matches upstream MESA `math_lib::sinpi` with exact zeros at integers.
#[pyfunction]
fn sinpi(x: f64) -> f64 {
    flexmesa_math::sinpi(x)
}

/// Evaluates range-reduced $\cos(\pi x)$ on a scalar `f64`.
///
/// Matches upstream MESA `math_lib::cospi` with exact zeros at half-integers.
#[pyfunction]
fn cospi(x: f64) -> f64 {
    flexmesa_math::cospi(x)
}

/// Evaluates range-reduced $\tan(\pi x)$ on a scalar `f64`.
///
/// Matches upstream MESA `math_lib::tanpi`.
#[pyfunction]
fn tanpi(x: f64) -> f64 {
    flexmesa_math::tanpi(x)
}

/// Evaluates the three-valued signum function $\operatorname{sgn}(x)$ on `f64`.
///
/// Returns `-1.0` for $x < 0$, `0.0` for $x = 0$, and `+1.0` for $x > 0$.
#[pyfunction]
fn sgn(x: f64) -> f64 {
    flexmesa_math::sgn(x)
}

/// Evaluates the Heaviside step function $H(x)$ (`0.5` at `x == 0`) on `f64`.
///
/// Returns `0.0` for $x < 0$, `0.5` for $x = 0$, and `1.0` for $x > 0$.
#[pyfunction]
fn heaviside(x: f64) -> f64 {
    flexmesa_math::heaviside(x)
}

/// Parses whitespace/comma-separated numbers from `text` into a 1D `f64` array.
///
/// Supports Fortran `D`/`d` and exponent-less floating-point literals.
#[pyfunction]
fn str_to_vector<'py>(
    py: Python<'py>,
    text: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let vals = flexmesa_math::str_to_vector(text)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, vals))
}

/// Applies scalar function `f` elementwise across a 1D C-contiguous `f64`
/// array.
fn map_unary_array<'py>(
    py: Python<'py>,
    xs: PyReadonlyArray1<'py, f64>,
    f: fn(f64) -> f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let s = xs
        .as_slice()
        .map_err(|_| PyValueError::new_err("xs must be C-contiguous"))?;
    Ok(PyArray1::from_vec(py, s.iter().copied().map(f).collect()))
}

/// Evaluates `safe_log` elementwise over a 1D C-contiguous `f64` array.
#[pyfunction]
fn safe_log_array<'py>(
    py: Python<'py>,
    xs: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    map_unary_array(py, xs, flexmesa_math::safe_log)
}

/// Evaluates `safe_log10` elementwise over a 1D C-contiguous `f64` array.
#[pyfunction]
fn safe_log10_array<'py>(
    py: Python<'py>,
    xs: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    map_unary_array(py, xs, flexmesa_math::safe_log10)
}

/// Evaluates `safe_sqrt` elementwise over a 1D C-contiguous `f64` array.
#[pyfunction]
fn safe_sqrt_array<'py>(
    py: Python<'py>,
    xs: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    map_unary_array(py, xs, flexmesa_math::safe_sqrt)
}

/// Evaluates `exp10` elementwise over a 1D C-contiguous `f64` array.
#[pyfunction]
fn exp10_array<'py>(
    py: Python<'py>,
    xs: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    map_unary_array(py, xs, flexmesa_math::exp10)
}

/// Unpacks `(lblk, dblk, ublk, rhs)` C-contiguous slices for block-tridiagonal
/// linear algebra routines.
fn unpack_block_tridiag<'a>(
    lblk: &'a PyReadonlyArray1<'a, f64>,
    dblk: &'a PyReadonlyArray1<'a, f64>,
    ublk: &'a PyReadonlyArray1<'a, f64>,
    rhs: &'a PyReadonlyArray1<'a, f64>,
) -> PyResult<(&'a [f64], &'a [f64], &'a [f64], &'a [f64])> {
    let ls = lblk
        .as_slice()
        .map_err(|_| PyValueError::new_err("lblk must be C-contiguous"))?;
    let ds = dblk
        .as_slice()
        .map_err(|_| PyValueError::new_err("dblk must be C-contiguous"))?;
    let us = ublk
        .as_slice()
        .map_err(|_| PyValueError::new_err("ublk must be C-contiguous"))?;
    let rs = rhs
        .as_slice()
        .map_err(|_| PyValueError::new_err("rhs must be C-contiguous"))?;
    Ok((ls, ds, us, rs))
}

/// Solves a block-tridiagonal system $A x = b$ via sequential block-Thomas.
///
/// Expects flat C-contiguous `lblk, dblk, ublk` of length `nz * nvar * nvar`
/// and `rhs` of length `nz * nvar`. Returns solution of shape `(nz, nvar)`.
#[pyfunction]
fn solve_block_tridiagonal<'py>(
    py: Python<'py>,
    nvar: usize,
    nz: usize,
    lblk: PyReadonlyArray1<'py, f64>,
    dblk: PyReadonlyArray1<'py, f64>,
    ublk: PyReadonlyArray1<'py, f64>,
    rhs: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (ls, ds, us, rs) = unpack_block_tridiag(&lblk, &dblk, &ublk, &rhs)?;
    let sol = flexmesa_mtx::solve_block_tridiagonal(nvar, nz, ls, ds, us, rs)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    PyArray1::from_vec(py, sol).reshape([nz, nvar])
}

/// Solves a block-tridiagonal system $A x = b$ via Hirshman et al. (2010)
/// odd-even block cyclic reduction (`bcyclic`).
///
/// When `parallel=True`, executes level reductions inside the persistent
/// 4-thread `rayon` worker pool.
#[pyfunction]
#[pyo3(signature = (nvar, nz, lblk, dblk, ublk, rhs, parallel=false))]
fn solve_block_tridiag_bcyclic<'py>(
    py: Python<'py>,
    nvar: usize,
    nz: usize,
    lblk: PyReadonlyArray1<'py, f64>,
    dblk: PyReadonlyArray1<'py, f64>,
    ublk: PyReadonlyArray1<'py, f64>,
    rhs: PyReadonlyArray1<'py, f64>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (ls, ds, us, rs) = unpack_block_tridiag(&lblk, &dblk, &ublk, &rhs)?;
    let sol = if parallel {
        stage3_pool().install(|| {
            flexmesa_mtx::solve_block_tridiag_bcyclic(
                nvar, nz, ls, ds, us, rs, true,
            )
        })
    } else {
        flexmesa_mtx::solve_block_tridiag_bcyclic(
            nvar, nz, ls, ds, us, rs, false,
        )
    }
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    PyArray1::from_vec(py, sol).reshape([nz, nvar])
}

/// Evaluates the left block-tridiagonal product $b = x^{\top} A$.
///
/// Equivalent to the transpose product $A^{\top} x$ used in adjoint / reverse
/// differentiation of block-tridiagonal linear solves.
#[pyfunction]
fn block_multiply_xa<'py>(
    py: Python<'py>,
    nvar: usize,
    nz: usize,
    lblk: PyReadonlyArray1<'py, f64>,
    dblk: PyReadonlyArray1<'py, f64>,
    ublk: PyReadonlyArray1<'py, f64>,
    vec_x: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (ls, ds, us, xs) = unpack_block_tridiag(&lblk, &dblk, &ublk, &vec_x)?;
    let mut out = vec![0.0_f64; nz * nvar];
    flexmesa_mtx::block_multiply_xa(nvar, nz, ls, ds, us, xs, &mut out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    PyArray1::from_vec(py, out).reshape([nz, nvar])
}

/// Registers Tier A/B/C `auto_diff`, `math`, and `mtx` functions on `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eval_autodiff_order2_zone, m)?)?;
    m.add_function(wrap_pyfunction!(eval_autodiff_order2_batch, m)?)?;
    m.add_function(wrap_pyfunction!(safe_sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(safe_log, m)?)?;
    m.add_function(wrap_pyfunction!(safe_log10, m)?)?;
    m.add_function(wrap_pyfunction!(exp10, m)?)?;
    m.add_function(wrap_pyfunction!(pow, m)?)?;
    m.add_function(wrap_pyfunction!(sinpi, m)?)?;
    m.add_function(wrap_pyfunction!(cospi, m)?)?;
    m.add_function(wrap_pyfunction!(tanpi, m)?)?;
    m.add_function(wrap_pyfunction!(sgn, m)?)?;
    m.add_function(wrap_pyfunction!(heaviside, m)?)?;
    m.add_function(wrap_pyfunction!(str_to_vector, m)?)?;
    m.add_function(wrap_pyfunction!(safe_log_array, m)?)?;
    m.add_function(wrap_pyfunction!(safe_log10_array, m)?)?;
    m.add_function(wrap_pyfunction!(safe_sqrt_array, m)?)?;
    m.add_function(wrap_pyfunction!(exp10_array, m)?)?;
    m.add_function(wrap_pyfunction!(solve_block_tridiagonal, m)?)?;
    m.add_function(wrap_pyfunction!(solve_block_tridiag_bcyclic, m)?)?;
    m.add_function(wrap_pyfunction!(block_multiply_xa, m)?)?;
    Ok(())
}
