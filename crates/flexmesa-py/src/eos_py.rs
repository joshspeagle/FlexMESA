// SPDX-License-Identifier: LGPL-3.0-or-later
//! `PyO3` bindings for `flexmesa-eos` (Wave 3a analytical and single-table EOS
//! core).

use flexmesa_autodiff::AutoDiffReal2VarOrder1;
use flexmesa_eos::{
    blend_eos_results, compute_ion_offset, compute_ion_offset_from_za, dfermi,
    eval_ideal_eos, eval_ideal_eos_from_weights, is_contained_scalar,
    min_distance_to_polygon, quintic_blend, EosEvalOutput,
    EOS_BASIC_RESULT_NAMES, IDEAL_MASS_FRACTION_LIMIT, NUM_EOS_BASIC_RESULTS,
};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::eval_batch_2d;

pub(crate) type PyEosTuple<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

fn parse_polygon_coords(
    coords: &PyReadonlyArray2<'_, f64>,
) -> PyResult<Vec<[f64; 2]>> {
    let arr = coords.as_array();
    if arr.ndim() != 2 || arr.shape()[1] != 2 {
        return Err(PyValueError::new_err("coords must have shape (N, 2)"));
    }
    Ok(arr.outer_iter().map(|row| [row[0], row[1]]).collect())
}

pub(crate) fn into_py_eos_tuple<'py>(
    py: Python<'py>,
    out: &EosEvalOutput,
) -> PyEosTuple<'py> {
    (
        PyArray1::from_slice(py, &out.res),
        PyArray1::from_slice(py, &out.d_dlnd),
        PyArray1::from_slice(py, &out.d_dln_t),
    )
}

fn parse_26_slice<'a>(
    arr: &'a PyReadonlyArray1<'a, f64>,
    name: &str,
) -> PyResult<[f64; NUM_EOS_BASIC_RESULTS]> {
    let s = arr.as_slice().map_err(|_| {
        PyValueError::new_err(format!("{name} must be contiguous"))
    })?;
    s.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must have length {NUM_EOS_BASIC_RESULTS}"
        ))
    })
}

/// Returns the 26 canonical `eos_def` result channel names (`0..26`).
#[pyfunction]
pub fn eos_basic_result_names() -> Vec<&'static str> {
    EOS_BASIC_RESULT_NAMES.to_vec()
}

/// Evaluates `is_contained` on an `(N, 2)` polygon coordinate array and a 2D
/// point `(x, y)`.
#[pyfunction]
pub fn eos_is_contained(
    coords: PyReadonlyArray2<'_, f64>,
    x: f64,
    y: f64,
) -> PyResult<bool> {
    let pts = parse_polygon_coords(&coords)?;
    Ok(is_contained_scalar(&pts, [x, y]))
}

/// Evaluates `min_distance_to_polygon` on an `(N, 2)` polygon coordinate array
/// and a 2D dual point `(x, y)` with first-order tangent seeds
/// `(dx_d1, dx_d2)` and `(dy_d1, dy_d2)`.
#[pyfunction]
#[pyo3(signature = (coords, x, y, dx_d1=1.0, dx_d2=0.0, dy_d1=0.0, dy_d2=1.0))]
pub fn eos_min_distance_to_polygon(
    coords: PyReadonlyArray2<'_, f64>,
    x: f64,
    y: f64,
    dx_d1: f64,
    dx_d2: f64,
    dy_d1: f64,
    dy_d2: f64,
) -> PyResult<(f64, f64, f64)> {
    let pts = parse_polygon_coords(&coords)?;
    let p = [
        AutoDiffReal2VarOrder1::new(x, [dx_d1, dx_d2]),
        AutoDiffReal2VarOrder1::new(y, [dy_d1, dy_d2]),
    ];
    let d = min_distance_to_polygon(&pts, p);
    Ok((d.val, d.d1[0], d.d1[1]))
}

/// Evaluates the $C^2$ quintic blend polynomial and its derivative
/// `(s, ds_dalpha0)` at `alpha0`.
#[pyfunction]
pub fn eos_quintic_blend(alpha0: f64) -> (f64, f64) {
    quintic_blend(alpha0)
}

/// Blends two 26-channel EOS outputs (`res_1`, `res_2`) and their logarithmic
/// derivatives using `blend_eos_results`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn eos_blend_results<'py>(
    py: Python<'py>,
    alpha_in: f64,
    d_alpha_dlog_t_in: f64,
    d_alpha_dlog_rho_in: f64,
    linear_blend: bool,
    res_1: PyReadonlyArray1<'py, f64>,
    d_dlnd_1: PyReadonlyArray1<'py, f64>,
    d_dln_t_1: PyReadonlyArray1<'py, f64>,
    res_2: PyReadonlyArray1<'py, f64>,
    d_dlnd_2: PyReadonlyArray1<'py, f64>,
    d_dln_t_2: PyReadonlyArray1<'py, f64>,
) -> PyResult<PyEosTuple<'py>> {
    let out1 = EosEvalOutput {
        res: parse_26_slice(&res_1, "res_1")?,
        d_dlnd: parse_26_slice(&d_dlnd_1, "d_dlnd_1")?,
        d_dln_t: parse_26_slice(&d_dln_t_1, "d_dln_t_1")?,
        d_dxa: Vec::new(),
    };
    let out2 = EosEvalOutput {
        res: parse_26_slice(&res_2, "res_2")?,
        d_dlnd: parse_26_slice(&d_dlnd_2, "d_dlnd_2")?,
        d_dln_t: parse_26_slice(&d_dln_t_2, "d_dln_t_2")?,
        d_dxa: Vec::new(),
    };
    let blended = blend_eos_results(
        alpha_in,
        d_alpha_dlog_t_in,
        d_alpha_dlog_rho_in,
        linear_blend,
        &out1,
        &out2,
    );
    Ok(into_py_eos_tuple(py, &blended))
}

/// Evaluates the relativistic Fermi-Dirac integral `dfermi(dk, eta, theta)`
/// returning `(fd, fd_deta, fd_dtheta)`.
#[pyfunction]
pub fn eos_dfermi(dk: f64, eta: f64, theta: f64) -> (f64, f64, f64) {
    dfermi(dk, eta, theta)
}

/// Evaluates `dfermi` across `nz` zones returning an `(nz, 3)` array of
/// `[fd, fd_deta, fd_dtheta]`.
#[pyfunction]
#[pyo3(signature = (dk, eta, theta, parallel=true))]
pub fn eos_dfermi_zones<'py>(
    py: Python<'py>,
    dk: f64,
    eta: PyReadonlyArray1<'py, f64>,
    theta: PyReadonlyArray1<'py, f64>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let eta_s = eta
        .as_slice()
        .map_err(|_| PyValueError::new_err("eta must be contiguous"))?
        .to_vec();
    let theta_s = theta
        .as_slice()
        .map_err(|_| PyValueError::new_err("theta must be contiguous"))?
        .to_vec();
    if eta_s.len() != theta_s.len() {
        return Err(PyValueError::new_err(
            "eta and theta must have equal length",
        ));
    }
    let nz = eta_s.len();
    eval_batch_2d::<3, _>(py, nz, parallel, 64, move |k| {
        let (fd, fdeta, fdtheta) = dfermi(dk, eta_s[k], theta_s[k]);
        [fd, fdeta, fdtheta]
    })
}

/// Computes the ground-state ionization energy offset (in $\mathrm{erg/g}$)
/// from species mass fractions `xa` and 1-based MESA `chem_id` identifiers.
#[pyfunction]
pub fn eos_compute_ion_offset(
    xa: PyReadonlyArray1<'_, f64>,
    chem_id: Vec<usize>,
) -> PyResult<f64> {
    let xa_s = xa
        .as_slice()
        .map_err(|_| PyValueError::new_err("xa must be contiguous"))?;
    compute_ion_offset(xa_s, &chem_id)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Computes the ground-state ionization energy offset (in $\mathrm{erg/g}$)
/// directly from `(xa, z, a)`.
#[pyfunction]
pub fn eos_compute_ion_offset_from_za(
    xa: PyReadonlyArray1<'_, f64>,
    z: Vec<usize>,
    a: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let xa_s = xa
        .as_slice()
        .map_err(|_| PyValueError::new_err("xa must be contiguous"))?;
    let a_s = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("a must be contiguous"))?;
    Ok(compute_ion_offset_from_za(xa_s, &z, a_s))
}

/// Evaluates the neutral ideal-gas + radiation EOS at a single zone from
/// `(temp, rho, abar, zbar, xa, chem_id)`.
#[pyfunction]
pub fn eos_eval_ideal<'py>(
    py: Python<'py>,
    temp: f64,
    rho: f64,
    abar: f64,
    zbar: f64,
    xa: PyReadonlyArray1<'py, f64>,
    chem_id: Vec<usize>,
) -> PyResult<PyEosTuple<'py>> {
    let xa_s = xa
        .as_slice()
        .map_err(|_| PyValueError::new_err("xa must be contiguous"))?;
    let out = eval_ideal_eos(temp, rho, abar, zbar, xa_s, &chem_id)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(into_py_eos_tuple(py, &out))
}

/// Evaluates the neutral ideal-gas + radiation EOS at a single zone directly
/// from `(temp, rho, abar, zbar, xa, weights, a)`.
#[pyfunction]
pub fn eos_eval_ideal_from_weights<'py>(
    py: Python<'py>,
    temp: f64,
    rho: f64,
    abar: f64,
    zbar: f64,
    xa: PyReadonlyArray1<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    a: PyReadonlyArray1<'py, f64>,
) -> PyResult<PyEosTuple<'py>> {
    let xa_s = xa
        .as_slice()
        .map_err(|_| PyValueError::new_err("xa must be contiguous"))?;
    let w_s = weights
        .as_slice()
        .map_err(|_| PyValueError::new_err("weights must be contiguous"))?;
    let a_s = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("a must be contiguous"))?;
    let out =
        eval_ideal_eos_from_weights(temp, rho, abar, zbar, xa_s, w_s, a_s)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(into_py_eos_tuple(py, &out))
}

/// Evaluates the neutral ideal-gas + radiation EOS across `nz` zones (at fixed
/// composition `(abar, zbar, xa, weights, a)`), returning an `(nz, 78)` array
/// packing `[res[0..26], d_dlnd[0..26], d_dln_t[0..26]]`.
#[pyfunction]
#[pyo3(signature = (temp, rho, abar, zbar, xa, weights, a, parallel=true))]
#[allow(clippy::too_many_arguments)]
pub fn eos_eval_ideal_zones<'py>(
    py: Python<'py>,
    temp: PyReadonlyArray1<'py, f64>,
    rho: PyReadonlyArray1<'py, f64>,
    abar: f64,
    zbar: f64,
    xa: PyReadonlyArray1<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    a: PyReadonlyArray1<'py, f64>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let t_s = temp
        .as_slice()
        .map_err(|_| PyValueError::new_err("temp must be contiguous"))?
        .to_vec();
    let r_s = rho
        .as_slice()
        .map_err(|_| PyValueError::new_err("rho must be contiguous"))?
        .to_vec();
    if t_s.len() != r_s.len() {
        return Err(PyValueError::new_err(
            "temp and rho must have equal length",
        ));
    }
    let xa_v = xa
        .as_slice()
        .map_err(|_| PyValueError::new_err("xa must be contiguous"))?
        .to_vec();
    let w_v = weights
        .as_slice()
        .map_err(|_| PyValueError::new_err("weights must be contiguous"))?
        .to_vec();
    let a_v = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("a must be contiguous"))?
        .to_vec();
    let sp = xa_v.len().min(w_v.len()).min(a_v.len());
    if abar <= 0.0 || !xa_v[..sp].iter().any(|&x| x > IDEAL_MASS_FRACTION_LIMIT)
    {
        return Err(PyValueError::new_err(
            "invalid composition for eos_eval_ideal_zones",
        ));
    }
    if t_s.iter().any(|&t| t <= 0.0 || t.is_nan())
        || r_s.iter().any(|&r| r <= 0.0 || r.is_nan())
    {
        return Err(PyValueError::new_err(
            "all temp and rho values must be strictly positive",
        ));
    }
    let nz = t_s.len();
    eval_batch_2d::<78, _>(py, nz, parallel, 64, move |k| {
        let mut row = [f64::NAN; 78];
        if let Ok(out) = eval_ideal_eos_from_weights(
            t_s[k], r_s[k], abar, zbar, &xa_v, &w_v, &a_v,
        ) {
            row[..NUM_EOS_BASIC_RESULTS].copy_from_slice(&out.res);
            row[NUM_EOS_BASIC_RESULTS..2 * NUM_EOS_BASIC_RESULTS]
                .copy_from_slice(&out.d_dlnd);
            row[2 * NUM_EOS_BASIC_RESULTS..3 * NUM_EOS_BASIC_RESULTS]
                .copy_from_slice(&out.d_dln_t);
        }
        row
    })
}

/// Registers `flexmesa-eos` bindings on the `flexmesa._rust` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eos_basic_result_names, m)?)?;
    m.add_function(wrap_pyfunction!(eos_is_contained, m)?)?;
    m.add_function(wrap_pyfunction!(eos_min_distance_to_polygon, m)?)?;
    m.add_function(wrap_pyfunction!(eos_quintic_blend, m)?)?;
    m.add_function(wrap_pyfunction!(eos_blend_results, m)?)?;
    m.add_function(wrap_pyfunction!(eos_dfermi, m)?)?;
    m.add_function(wrap_pyfunction!(eos_dfermi_zones, m)?)?;
    m.add_function(wrap_pyfunction!(eos_compute_ion_offset, m)?)?;
    m.add_function(wrap_pyfunction!(eos_compute_ion_offset_from_za, m)?)?;
    m.add_function(wrap_pyfunction!(eos_eval_ideal, m)?)?;
    m.add_function(wrap_pyfunction!(eos_eval_ideal_from_weights, m)?)?;
    m.add_function(wrap_pyfunction!(eos_eval_ideal_zones, m)?)?;
    Ok(())
}
