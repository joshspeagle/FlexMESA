// SPDX-License-Identifier: LGPL-3.0-or-later
//! `PyO3` bindings for `flexmesa-colors` (`Wave 2c`).
//!
//! Exposes the canonical Safe Rust implementations of `colors` to Python and
//! JAX (`flexmesa._rust`):
//! - **Built-in Catalogs (`py_colors_builtin_grid`, `py_colors_column_names`,
//!   `py_colors_filter_names`, `py_colors_filter_curve`)**:
//!   Zero-copy access to the 4D `Kurucz2003all` atmosphere SED cube
//!   $(T_{\mathrm{eff}}, \log g, [\mathrm{M/H}], \lambda)$ and 7 Johnson
//!   passbands (`B, I, J, M, R, U, V`) with Vega, AB, and ST zero points.
//! - **3D SED Interpolation (`py_colors_eval_sed`)**:
//!   Evaluates diluted stellar spectra across `"Hermite"`, `"Linear"`, and
//!   `"KNN"` interpolation methods.
//! - **10-Column History Photometry (`py_colors_eval_colors`,
//!   `py_colors_eval_colors_dual`, `py_colors_eval_colors_zones`)**:
//!   Evaluates `[Mag_bol, Flux_bol, Interp_rad, B, I, J, M, R, U, V]` in
//!   single-point `f64`, forward-mode `DualOrder1<4>` Jacobian `(10, 4)`, and
//!   persistent `rayon` multi-zone batch `(n_zones, 10)` modes.
//! - **Photometric & Quadrature Utilities**:
//!   Composite non-uniform Simpson (`py_colors_simpson_integration`) and
//!   trapezoidal (`py_colors_trapezoidal_integration`) quadrature, geometric
//!   dilution (`py_colors_dilute_flux`), piecewise-linear filter resampling
//!   (`py_colors_interpolate_array`), bolometric photometry, synthetic filter
//!   magnitudes, and Vega/AB/ST zero-point integrals.

use flexmesa_colors::{
    builtin_sed_cube, calculate_bolometric_phot, calculate_synthetic,
    compute_ab_zero_point, compute_st_zero_point, compute_vega_zero_point,
    dilute_flux, interpolate_array, simpson_integration,
    trapezoidal_integration, MagSystem, SedInterpMethod, COLOR_COLUMN_NAMES,
    D_10PC_CM, JOHNSON_FILTERS, JOHNSON_FILTER_NAMES, NUM_COLORS_OUT,
    NUM_JOHNSON_FILTERS, PARALLEL_ZONE_THRESHOLD,
};
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::eval_batch_2d;

type PyArr1<'py> = Bound<'py, PyArray1<f64>>;
type PyArr2<'py> = Bound<'py, PyArray2<f64>>;
type PyRoArr1<'py> = PyReadonlyArray1<'py, f64>;

/// Return tuple type for [`py_colors_filter_curve`].
type PyFilterCurve<'py> = (PyArr1<'py>, PyArr1<'py>, f64, f64, f64);

/// Return tuple type for [`py_colors_builtin_grid`].
type PyBuiltinGrid<'py> = (
    PyArr1<'py>, PyArr1<'py>, PyArr1<'py>, PyArr1<'py>,
    Bound<'py, PyArray1<i64>>, PyArr2<'py>, PyArr2<'py>,
    PyArr1<'py>, PyArr1<'py>, PyArr1<'py>, PyArr1<'py>,
);

/// Converts a `flexmesa-colors` error into a Python `ValueError`.
#[inline]
fn py_err(e: impl core::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Returns the ordered 10 `colors` history output column names:
/// `["Mag_bol", "Flux_bol", "Interp_rad", "B", "I", "J", "M", "R", "U", "V"]`.
#[pyfunction]
fn py_colors_column_names() -> Vec<&'static str> {
    COLOR_COLUMN_NAMES.to_vec()
}

/// Returns the ordered 7 Johnson filter names (`["B", "I", "J", "M", "R", "U",
/// "V"]`).
#[pyfunction]
fn py_colors_filter_names() -> Vec<&'static str> {
    JOHNSON_FILTER_NAMES.to_vec()
}

/// Returns `(wavelengths, transmission, vega_zp, ab_zp, st_zp)` for Johnson
/// filter `idx` (`0..7`).
#[pyfunction]
fn py_colors_filter_curve(
    py: Python<'_>,
    idx: usize,
) -> PyResult<PyFilterCurve<'_>> {
    if idx >= NUM_JOHNSON_FILTERS {
        return Err(PyValueError::new_err(format!(
            "Johnson filter index {idx} out of range 0..{NUM_JOHNSON_FILTERS}"
        )));
    }
    let f = &JOHNSON_FILTERS[idx];
    Ok((
        PyArray1::from_slice(py, f.wavelengths),
        PyArray1::from_slice(py, f.transmission),
        f.vega_zero_point,
        f.ab_zero_point,
        f.st_zero_point,
    ))
}

/// Returns the built-in `Kurucz2003all` + `Johnson` grid arrays for `@jax.jit`:
/// `(teff_grid, logg_grid, meta_grid, wavelengths, slot_map, sed_table,
/// filters_on_sed_grid, filter_norm_integrals, vega_zp, ab_zp, st_zp)`.
#[pyfunction]
fn py_colors_builtin_grid(py: Python<'_>) -> PyResult<PyBuiltinGrid<'_>> {
    let cube = builtin_sed_cube();
    let nlam = cube.wavelengths.len();
    let n_slots = cube.sed_table.len() / nlam;
    let slot_i64: Vec<i64> = cube.slot_map.iter().map(|&s| s as i64).collect();
    let sed_arr = PyArray1::from_slice(py, &cube.sed_table)
        .reshape([n_slots, nlam])?;
    let filt_flat: Vec<f64> = cube
        .filters_on_sed_grid
        .iter()
        .flatten()
        .copied()
        .collect();
    let filt_arr = PyArray1::from_vec(py, filt_flat)
        .reshape([NUM_JOHNSON_FILTERS, nlam])?;
    let vega_zp: Vec<f64> =
        JOHNSON_FILTERS.iter().map(|f| f.vega_zero_point).collect();
    let ab_zp: Vec<f64> =
        JOHNSON_FILTERS.iter().map(|f| f.ab_zero_point).collect();
    let st_zp: Vec<f64> =
        JOHNSON_FILTERS.iter().map(|f| f.st_zero_point).collect();

    Ok((
        PyArray1::from_slice(py, &cube.teff_grid),
        PyArray1::from_slice(py, &cube.logg_grid),
        PyArray1::from_slice(py, &cube.meta_grid),
        PyArray1::from_slice(py, &cube.wavelengths),
        PyArray1::from_vec(py, slot_i64),
        sed_arr,
        filt_arr,
        PyArray1::from_slice(py, &cube.filter_norm_integrals),
        PyArray1::from_vec(py, vega_zp),
        PyArray1::from_vec(py, ab_zp),
        PyArray1::from_vec(py, st_zp),
    ))
}

/// Evaluates the diluted stellar SED `(wavelengths, fluxes)` at
/// `(teff, log_g, metallicity, r, d)` using `method` (`"Hermite"`, `"Linear"`,
/// or `"KNN"`).
///
/// # Arguments
/// * `teff` - Effective temperature $T_{\mathrm{eff}}$ in $\mathrm{K}$.
/// * `log_g` - Surface gravity $\log_{10}(g / \mathrm{cm\,s^{-2}})$.
/// * `metallicity` - Metallicity $[\mathrm{M/H}]$ in dex.
/// * `r` - Stellar photospheric radius $R$ in $\mathrm{cm}$.
/// * `d` - Observer distance $d$ in $\mathrm{cm}$ (default $10\,\mathrm{pc}$).
/// * `method` - SED interpolator (`"Hermite"`, `"Linear"`, or `"KNN"`).
#[pyfunction]
#[pyo3(signature = (
    teff, log_g, metallicity, r, d = D_10PC_CM, method = "Hermite"
))]
fn py_colors_eval_sed<'py>(
    py: Python<'py>,
    teff: f64,
    log_g: f64,
    metallicity: f64,
    r: f64,
    d: f64,
    method: &str,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let m = SedInterpMethod::parse(method).map_err(py_err)?;
    let cube = builtin_sed_cube();
    let fluxes = cube.eval_sed(teff, log_g, metallicity, r, d, m);
    Ok((
        PyArray1::from_slice(py, &cube.wavelengths),
        PyArray1::from_vec(py, fluxes),
    ))
}

/// Evaluates the 10-column `colors` output vector
/// `[Mag_bol, Flux_bol, Interp_rad, B, I, J, M, R, U, V]` at a single point.
///
/// # Arguments
/// * `teff` - Effective temperature $T_{\mathrm{eff}}$ in $\mathrm{K}$.
/// * `log_g` - Surface gravity $\log_{10}(g / \mathrm{cm\,s^{-2}})$.
/// * `metallicity` - Metallicity $[\mathrm{M/H}]$ in dex.
/// * `r` - Stellar photospheric radius $R$ in $\mathrm{cm}$.
/// * `d` - Observer distance $d$ in $\mathrm{cm}$ (default $10\,\mathrm{pc}$).
/// * `mag_system` - Zero-point system (`"Vega"`, `"AB"`, or `"ST"`).
/// * `method` - SED interpolator (`"Hermite"`, `"Linear"`, or `"KNN"`).
#[pyfunction]
#[pyo3(signature = (
    teff, log_g, metallicity, r, d = D_10PC_CM,
    mag_system = "Vega", method = "Hermite"
))]
fn py_colors_eval_colors<'py>(
    py: Python<'py>,
    teff: f64,
    log_g: f64,
    metallicity: f64,
    r: f64,
    d: f64,
    mag_system: &str,
    method: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let sys = MagSystem::parse(mag_system).map_err(py_err)?;
    let m = SedInterpMethod::parse(method).map_err(py_err)?;
    let out = builtin_sed_cube()
        .eval_colors(teff, log_g, metallicity, r, d, sys, m)
        .map_err(py_err)?;
    Ok(PyArray1::from_slice(py, &out))
}

/// Evaluates `(values, jacobian)` for `eval_colors` using `DualOrder1<4>`
/// seeded on `(teff, log_g, metallicity, r)` at fixed `d`, where `values` has
/// shape `(10,)` and `jacobian` has shape `(10, 4)`.
#[pyfunction]
#[pyo3(signature = (
    teff, log_g, metallicity, r, d = D_10PC_CM, mag_system = "Vega"
))]
fn py_colors_eval_colors_dual<'py>(
    py: Python<'py>,
    teff: f64,
    log_g: f64,
    metallicity: f64,
    r: f64,
    d: f64,
    mag_system: &str,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let sys = MagSystem::parse(mag_system).map_err(py_err)?;
    let dual_out = builtin_sed_cube()
        .eval_colors_dual(teff, log_g, metallicity, r, d, sys)
        .map_err(py_err)?;
    let vals = dual_out.map(|d_val| d_val.val);
    let mut jac = [0.0_f64; NUM_COLORS_OUT * 4];
    for (i, d_val) in dual_out.iter().enumerate() {
        jac[4 * i..4 * (i + 1)].copy_from_slice(&d_val.d1);
    }
    let jac_arr =
        PyArray1::from_slice(py, &jac).reshape([NUM_COLORS_OUT, 4])?;
    Ok((PyArray1::from_slice(py, &vals), jac_arr))
}

/// Evaluates the 10-column `colors` output across `n_zones` stellar models in
/// parallel on the persistent `rayon` worker pool, returning an `(n_zones, 10)`
/// array.
#[pyfunction]
#[pyo3(signature = (
    teff, log_g, metallicity, r, d = D_10PC_CM,
    mag_system = "Vega", method = "Hermite", parallel = true
))]
fn py_colors_eval_colors_zones<'py>(
    py: Python<'py>,
    teff: PyRoArr1<'py>,
    log_g: PyRoArr1<'py>,
    metallicity: PyRoArr1<'py>,
    r: PyRoArr1<'py>,
    d: f64,
    mag_system: &str,
    method: &str,
    parallel: bool,
) -> PyResult<PyArr2<'py>> {
    let nz = teff.len();
    if log_g.len() != nz || metallicity.len() != nz || r.len() != nz {
        return Err(PyValueError::new_err(
            "teff, log_g, metallicity, and r must have identical 1D lengths",
        ));
    }
    let sys = MagSystem::parse(mag_system).map_err(py_err)?;
    let m = SedInterpMethod::parse(method).map_err(py_err)?;
    let (t_s, g_s, m_s, r_s) = (
        teff.as_slice()?,
        log_g.as_slice()?,
        metallicity.as_slice()?,
        r.as_slice()?,
    );
    let cube = builtin_sed_cube();
    eval_batch_2d::<NUM_COLORS_OUT, _>(
        py,
        nz,
        parallel,
        PARALLEL_ZONE_THRESHOLD,
        |k| {
            cube.eval_colors(t_s[k], g_s[k], m_s[k], r_s[k], d, sys, m)
                .unwrap_or([f64::NAN; NUM_COLORS_OUT])
        },
    )
}

/// Evaluates composite non-uniform Simpson quadrature
/// `simpson_integration(x, y)`.
#[pyfunction]
fn py_colors_simpson_integration(
    x: PyRoArr1<'_>, y: PyRoArr1<'_>,
) -> PyResult<f64> {
    simpson_integration(x.as_slice()?, y.as_slice()?).map_err(py_err)
}

/// Evaluates composite trapezoidal quadrature `trapezoidal_integration(x, y)`.
#[pyfunction]
fn py_colors_trapezoidal_integration(
    x: PyRoArr1<'_>, y: PyRoArr1<'_>,
) -> PyResult<f64> {
    trapezoidal_integration(x.as_slice()?, y.as_slice()?).map_err(py_err)
}

/// Applies geometric flux dilution `surface_flux * (r / d)^2`.
#[pyfunction]
fn py_colors_dilute_flux<'py>(
    py: Python<'py>, surface_flux: PyRoArr1<'py>, r: f64, d: f64,
) -> PyResult<PyArr1<'py>> {
    let out = dilute_flux(surface_flux.as_slice()?, r, d);
    Ok(PyArray1::from_vec(py, out))
}

/// Resamples 1D curve `(x_in, y_in)` onto `x_out` via piecewise-linear
/// interpolation with flat clamping outside `[x_in[0], x_in[-1]]`.
#[pyfunction]
fn py_colors_interpolate_array<'py>(
    py: Python<'py>, x_in: PyRoArr1<'py>, y_in: PyRoArr1<'py>,
    x_out: PyRoArr1<'py>,
) -> PyResult<PyArr1<'py>> {
    let out =
        interpolate_array(x_in.as_slice()?, y_in.as_slice()?, x_out.as_slice()?)
            .map_err(py_err)?;
    Ok(PyArray1::from_vec(py, out))
}

/// Computes `(bol_mag, bol_flux)` from `(wavelengths, fluxes)`.
#[pyfunction]
fn py_colors_calculate_bolometric_phot(
    wavelengths: PyRoArr1<'_>, fluxes: PyRoArr1<'_>,
) -> PyResult<(f64, f64)> {
    calculate_bolometric_phot(wavelengths.as_slice()?, fluxes.as_slice()?)
        .map_err(py_err)
}

/// Computes synthetic filter magnitude `calculate_synthetic` from an SED and
/// filter transmission curve.
#[pyfunction]
fn py_colors_calculate_synthetic(
    wavelengths: PyRoArr1<'_>, fluxes: PyRoArr1<'_>,
    filter_wavelengths: PyRoArr1<'_>, filter_trans: PyRoArr1<'_>,
    zero_point_flux: f64,
) -> PyResult<f64> {
    let res = calculate_synthetic(
        wavelengths.as_slice()?, fluxes.as_slice()?,
        filter_wavelengths.as_slice()?, filter_trans.as_slice()?,
        zero_point_flux,
    );
    res.map_err(py_err)
}

/// Computes the Vega photon-counting zero-point flux for a filter curve.
#[pyfunction]
fn py_colors_compute_vega_zero_point(
    vega_wave: PyRoArr1<'_>, vega_flux: PyRoArr1<'_>,
    filt_wave: PyRoArr1<'_>, filt_trans: PyRoArr1<'_>,
) -> PyResult<f64> {
    let res = compute_vega_zero_point(
        vega_wave.as_slice()?, vega_flux.as_slice()?,
        filt_wave.as_slice()?, filt_trans.as_slice()?,
    );
    res.map_err(py_err)
}

/// Computes the AB photon-counting zero-point flux ($f_\nu = 3631\,$Jy).
#[pyfunction]
fn py_colors_compute_ab_zero_point(
    filt_wave: PyRoArr1<'_>, filt_trans: PyRoArr1<'_>,
) -> PyResult<f64> {
    compute_ab_zero_point(filt_wave.as_slice()?, filt_trans.as_slice()?)
        .map_err(py_err)
}

/// Computes the ST photon-counting zero-point flux ($f_\lambda = 3.63\times
/// 10^{-9}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$).
#[pyfunction]
fn py_colors_compute_st_zero_point(
    filt_wave: PyRoArr1<'_>, filt_trans: PyRoArr1<'_>,
) -> PyResult<f64> {
    compute_st_zero_point(filt_wave.as_slice()?, filt_trans.as_slice()?)
        .map_err(py_err)
}

/// Registers all `colors` functions on `flexmesa._rust`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_colors_column_names, m)?)?;
    m.add_function(wrap_pyfunction!(py_colors_filter_names, m)?)?;
    m.add_function(wrap_pyfunction!(py_colors_filter_curve, m)?)?;
    m.add_function(wrap_pyfunction!(py_colors_builtin_grid, m)?)?;
    m.add_function(wrap_pyfunction!(py_colors_eval_sed, m)?)?;
    m.add_function(wrap_pyfunction!(py_colors_eval_colors, m)?)?;
    m.add_function(wrap_pyfunction!(py_colors_eval_colors_dual, m)?)?;
    m.add_function(wrap_pyfunction!(py_colors_eval_colors_zones, m)?)?;
    m.add_function(wrap_pyfunction!(py_colors_simpson_integration, m)?)?;
    m.add_function(wrap_pyfunction!(py_colors_trapezoidal_integration, m)?)?;
    m.add_function(wrap_pyfunction!(py_colors_dilute_flux, m)?)?;
    m.add_function(wrap_pyfunction!(py_colors_interpolate_array, m)?)?;
    m.add_function(wrap_pyfunction!(py_colors_calculate_bolometric_phot, m)?)?;
    m.add_function(wrap_pyfunction!(py_colors_calculate_synthetic, m)?)?;
    m.add_function(wrap_pyfunction!(py_colors_compute_vega_zero_point, m)?)?;
    m.add_function(wrap_pyfunction!(py_colors_compute_ab_zero_point, m)?)?;
    m.add_function(wrap_pyfunction!(py_colors_compute_st_zero_point, m)?)?;
    Ok(())
}
