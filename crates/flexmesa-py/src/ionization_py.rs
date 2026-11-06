// SPDX-License-Identifier: LGPL-3.0-or-later
//! `PyO3` bindings for `flexmesa-ionization` (`Wave 2b`).
//!
//! Exposes:
//! - **Fe56 in He4 (`eval_charge_of_Fe56_in_He4`)**: Scalar, analytical
//!   partial-derivative, and `rayon` multi-zone evaluation on the $105 \times
//!   30$ Thomas-Fermi-Dirac-Hubbard grid, plus precomputed spline coefficients
//!   for `@jax.jit`.
//! - **4D Ionization Tables (`eval_ionization`)**: 3-stage $(Z, X, \log Q,
//!   \log T)$ interpolator over 28 output quantities (`logPgas`, `logpp_H..Fe`,
//!   `Z_H..Fe`, `fneut_H..Fe`) across single and `rayon` multi-zone inputs.
//! - **Diffusion Typical Charge (`chi_info`, `chi_effective`,
//!   `eval_typical_charge`)**: Paquette et al. (1986) + Dupuis et al. (1992)
//!   continuum depression and Allen (1973) ionization potentials ($Z \le 30$).
//!
//! # Threading and Memory Layout
//! - Multi-zone evaluators (`py_ion_charge_of_fe56_in_he4_zones` and
//!   `py_ion_eval_ionization_zones`) accept 1D C-contiguous `float64`
//!   NumPy arrays, release the Python GIL during parallel zone evaluation,
//!   and return 2D row-major `(n_zones, K)` arrays.

use flexmesa_autodiff::DualOrder1;
use flexmesa_ionization::{
    Fe56He4Table, ION_RESULT_NAMES, IONIZATION_POTENTIALS, IonTableGrid,
    NUM_ION_VALS, NUM_ION_XS, NUM_ION_ZS, PARALLEL_ZONE_THRESHOLD,
    chi_effective, chi_info, eval_ionization, eval_typical_charge,
};
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::chem_py::builtin_catalog;
use crate::eval_batch_2d;

/// Return the ordered list of 28 `eval_ionization` output channel names.
///
/// Matches the 1-based ordering in upstream MESA `ionization_def.f90`:
/// `logPgas` (`0`), `logpp_H..Fe` (`1..10`), `Z_H..Fe` (`10..19`), and
/// `fneut_H..Fe` (`19..28`).
#[pyfunction]
fn py_ion_result_names() -> Vec<&'static str> {
    ION_RESULT_NAMES.to_vec()
}

/// Return `(log_ne, log_t, coeffs)` for the built-in $105 \times 30$
/// `Fe56He4Table`, where `coeffs` has shape `(105 * 30, 4)`.
///
/// Used by `flexmesa.ionization.fe56_he4` to initialize the `@jax.jit`
/// bicubic spline evaluator.
#[pyfunction]
fn py_ion_fe56_he4_grid<'py>(
    py: Python<'py>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let tbl = Fe56He4Table::builtin();
    let nx = tbl.log_ne.len();
    let ny = tbl.log_t.len();
    let flat: Vec<f64> = tbl.coeffs.iter().flatten().copied().collect();
    let log_ne_arr = PyArray1::from_slice(py, &tbl.log_ne);
    let log_t_arr = PyArray1::from_slice(py, &tbl.log_t);
    let coeffs_arr = PyArray1::from_vec(py, flat).reshape([nx * ny, 4])?;
    Ok((log_ne_arr, log_t_arr, coeffs_arr))
}

/// Return `(log_q_min, del_log_q, num_log_qs, log_t_min, del_log_t, num_log_ts,
/// coeffs)` for [`IonTableGrid::builtin`], where `coeffs` has shape
/// `(5 * 6 * 28, num_log_qs * num_log_ts * 4)`.
#[pyfunction]
fn py_ion_builtin_cell_grid<'py>(
    py: Python<'py>,
) -> PyResult<(f64, f64, usize, f64, f64, usize, Bound<'py, PyArray2<f64>>)> {
    let grid = IonTableGrid::builtin();
    let n_rows = NUM_ION_ZS * NUM_ION_XS * NUM_ION_VALS;
    let row_len = grid.num_log_qs * grid.num_log_ts * 4;
    let mut flat = Vec::with_capacity(n_rows * row_len);
    for iz in 0..NUM_ION_ZS {
        for ix in 0..NUM_ION_XS {
            for v in 0..NUM_ION_VALS {
                flat.extend_from_slice(&grid.coeffs[iz][ix][v]);
            }
        }
    }
    let arr = PyArray1::from_vec(py, flat).reshape([n_rows, row_len])?;
    Ok((
        grid.log_q_min,
        grid.del_log_q,
        grid.num_log_qs,
        grid.log_t_min,
        grid.del_log_t,
        grid.num_log_ts,
        arr,
    ))
}

/// Return the `(30, 30)` Allen (1973) ionization potential table in
/// $\mathrm{eV}$.
///
/// Row `z - 1` ($Z \in [1, 30]$) and column `stage - 1` ($k \in [1, Z]$) store
/// the ionization threshold $I_{Z, k}$ in $\mathrm{eV}$.
#[pyfunction]
fn py_ion_potentials_table<'py>(
    py: Python<'py>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let flat: Vec<f64> =
        IONIZATION_POTENTIALS.iter().flatten().copied().collect();
    PyArray1::from_vec(py, flat).reshape([30, 30])
}

/// Evaluate `(z, dz_dlog10_ne, dz_dlog10_t)` for ${}^{56}\mathrm{Fe}$ in
/// ${}^4\mathrm{He}$ at $(\log_{10} n_e, \log_{10} T)$.
///
/// Evaluates the 2D not-a-knot bicubic spline on the $105 \times 30$
/// Thomas-Fermi-Dirac-Hubbard grid (`eval_charge_of_Fe56_in_He4`).
#[pyfunction]
fn py_ion_charge_of_fe56_in_he4(
    log10_ne: f64,
    log10_t: f64,
) -> PyResult<(f64, f64, f64)> {
    let ev = Fe56He4Table::builtin()
        .eval_full(log10_ne, log10_t)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((ev.f, ev.df_dx, ev.df_dy))
}

/// Evaluate `[z, dz_dlog10_ne, dz_dlog10_t]` across `n_zones` stellar zones.
///
/// Dispatches to the persistent 4-thread `rayon` worker pool when
/// `parallel=true` and `n_zones >= 64`.
#[pyfunction]
#[pyo3(signature = (log10_ne, log10_t, parallel=false))]
fn py_ion_charge_of_fe56_in_he4_zones<'py>(
    py: Python<'py>,
    log10_ne: PyReadonlyArray1<'py, f64>,
    log10_t: PyReadonlyArray1<'py, f64>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let lne = log10_ne
        .as_slice()
        .map_err(|_| PyValueError::new_err("log10_ne must be C-contiguous"))?;
    let lt = log10_t
        .as_slice()
        .map_err(|_| PyValueError::new_err("log10_t must be C-contiguous"))?;
    if lne.len() != lt.len() {
        return Err(PyValueError::new_err(
            "log10_ne and log10_t must have equal length",
        ));
    }
    let tbl = Fe56He4Table::builtin();
    let x_min = tbl.log_ne[0];
    let x_max = tbl.log_ne[tbl.log_ne.len() - 1];
    let y_min = tbl.log_t[0];
    let y_max = tbl.log_t[tbl.log_t.len() - 1];
    for (&x, &y) in lne.iter().zip(lt) {
        if !x.is_finite()
            || !y.is_finite()
            || !(x_min..=x_max).contains(&x)
            || !(y_min..=y_max).contains(&y)
        {
            return Err(PyValueError::new_err(format!(
                "query ({x}, {y}) out of range [{x_min}, {x_max}] \
                 x [{y_min}, {y_max}]"
            )));
        }
    }
    eval_batch_2d::<3, _>(
        py,
        lne.len(),
        parallel,
        PARALLEL_ZONE_THRESHOLD,
        |k| {
            let ev = tbl.eval_full(lne[k], lt[k]).expect("prevalidated zone");
            [ev.f, ev.df_dx, ev.df_dy]
        },
    )
}

/// Evaluate all 28 `eval_ionization` channels for a single zone.
///
/// Accepts either linear or `log10` density and temperature arguments.
#[pyfunction]
#[pyo3(signature = (z, x, rho=None, log10_rho=None, t=None, log10_t=None))]
fn py_ion_eval_ionization(
    z: f64,
    x: f64,
    rho: Option<f64>,
    log10_rho: Option<f64>,
    t: Option<f64>,
    log10_t: Option<f64>,
) -> PyResult<Vec<f64>> {
    let res = eval_ionization(z, x, rho, log10_rho, t, log10_t)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(res.to_vec())
}

/// Evaluate all 28 `eval_ionization` channels and their $28 \times 4$ Jacobian
/// with respect to `(z, x, log10_rho, log10_t)` via `DualOrder1<4>`.
#[pyfunction]
fn py_ion_eval_ionization_dual<'py>(
    py: Python<'py>,
    z: f64,
    x: f64,
    log10_rho: f64,
    log10_t: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let zd = DualOrder1::<4>::independent(z, 0);
    let xd = DualOrder1::<4>::independent(x, 1);
    let lrhod = DualOrder1::<4>::independent(log10_rho, 2);
    let ltd = DualOrder1::<4>::independent(log10_t, 3);
    let out = IonTableGrid::builtin().eval_generic(zd, xd, lrhod, ltd);
    let vals: Vec<f64> = out.iter().map(|d| d.val).collect();
    let mut jac_flat = Vec::with_capacity(NUM_ION_VALS * 4);
    for d in &out {
        jac_flat.extend_from_slice(&d.d1);
    }
    let vals_arr = PyArray1::from_vec(py, vals);
    let jac_arr = PyArray1::from_vec(py, jac_flat)
        .reshape([NUM_ION_VALS, 4])
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((vals_arr, jac_arr))
}

/// Evaluate all 28 `eval_ionization` channels across `n_zones` stellar zones.
///
/// Dispatches to the persistent 4-thread `rayon` worker pool when
/// `parallel=true` and `n_zones >= 64`.
#[pyfunction]
#[pyo3(signature = (z, x, log10_rho, log10_t, parallel=false))]
fn py_ion_eval_ionization_zones<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray1<'py, f64>,
    log10_rho: PyReadonlyArray1<'py, f64>,
    log10_t: PyReadonlyArray1<'py, f64>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let z_s = z
        .as_slice()
        .map_err(|_| PyValueError::new_err("z must be C-contiguous"))?;
    let x_s = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
    let lrho_s = log10_rho
        .as_slice()
        .map_err(|_| PyValueError::new_err("log10_rho must be C-contiguous"))?;
    let lt_s = log10_t
        .as_slice()
        .map_err(|_| PyValueError::new_err("log10_t must be C-contiguous"))?;
    let n = z.shape()[0];
    if x_s.len() != n || lrho_s.len() != n || lt_s.len() != n {
        return Err(PyValueError::new_err(
            "z, x, log10_rho, and log10_t must have equal length",
        ));
    }
    let grid = IonTableGrid::builtin();
    eval_batch_2d::<28, _>(py, n, parallel, PARALLEL_ZONE_THRESHOLD, |k| {
        grid.eval_generic(z_s[k], x_s[k], lrho_s[k], lt_s[k])
    })
}

/// Compute $(\chi, c_0, c_1, c_2)$ (`chi_info` in `mod_ionization.f90`).
///
/// Evaluates the Paquette et al. (1986, Eq. 20) and Dupuis et al. (1992,
/// Eq. 3) continuum-depression parameters in $\mathrm{eV}$.
#[pyfunction]
fn py_ion_chi_info(
    a1: f64,
    z1: f64,
    t: f64,
    log10_t: f64,
    rho: f64,
    log10_rho: f64,
) -> (f64, f64, f64, f64) {
    chi_info(a1, z1, t, log10_t, rho, log10_rho)
}

/// Compute $\chi_{\mathrm{eff}}$ (`chi_effective` in `mod_ionization.f90`).
///
/// Evaluates $\chi_{\mathrm{eff}}(z_2) = \chi + c_0 / z_2^3 + \min[c_1 z_2,
/// c_2 ((z_2 / z_1)^{2/3} + 0.6)]$ in $\mathrm{eV}$.
#[pyfunction]
fn py_ion_chi_effective(
    chi: f64,
    c0: f64,
    c1: f64,
    c2: f64,
    z1: f64,
    z2: f64,
) -> f64 {
    chi_effective(chi, c0, c1, c2, z1, z2)
}

/// Compute whole-number typical charge (`eval_typical_charge`).
///
/// Compares $\chi_{\mathrm{eff}}(z_2)$ against the Allen (1973) ionization
/// potentials (`IONIZATION_POTENTIALS`) for nuclide `cid` ($1 \le Z \le 30$).
#[pyfunction]
fn py_ion_eval_typical_charge(
    cid: usize,
    abar: f64,
    free_e: f64,
    t: f64,
    log10_t: f64,
    rho: f64,
    log10_rho: f64,
) -> PyResult<f64> {
    eval_typical_charge(
        builtin_catalog(),
        cid,
        abar,
        free_e,
        t,
        log10_t,
        rho,
        log10_rho,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Register all `ionization` functions on the `flexmesa._rust` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_ion_result_names, m)?)?;
    m.add_function(wrap_pyfunction!(py_ion_fe56_he4_grid, m)?)?;
    m.add_function(wrap_pyfunction!(py_ion_builtin_cell_grid, m)?)?;
    m.add_function(wrap_pyfunction!(py_ion_potentials_table, m)?)?;
    m.add_function(wrap_pyfunction!(py_ion_charge_of_fe56_in_he4, m)?)?;
    m.add_function(wrap_pyfunction!(py_ion_charge_of_fe56_in_he4_zones, m)?)?;
    m.add_function(wrap_pyfunction!(py_ion_eval_ionization, m)?)?;
    m.add_function(wrap_pyfunction!(py_ion_eval_ionization_dual, m)?)?;
    m.add_function(wrap_pyfunction!(py_ion_eval_ionization_zones, m)?)?;
    m.add_function(wrap_pyfunction!(py_ion_chi_info, m)?)?;
    m.add_function(wrap_pyfunction!(py_ion_chi_effective, m)?)?;
    m.add_function(wrap_pyfunction!(py_ion_eval_typical_charge, m)?)?;
    Ok(())
}
