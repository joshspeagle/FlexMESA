// SPDX-License-Identifier: LGPL-3.0-or-later
//! Tier A and Tier B `PyO3` bindings for `flexmesa-chem`.
//!
//! Part of the Wave 2a (`chem`) microphysics layer in `FlexMESA`.
//! Verified against 100% of upstream MESA `chem/test/test_output`
//! (`o16`, `AG89`, `Lodders 2003`, `composition_info`, `reaction_Qtotal`).
//! Enforces `#![forbid(unsafe_code)]` and IEEE-754 `f64` precision.
//!
//! Exposes the static periodic table, Anders & Grevesse (1989) + Lodders (2003)
//! 286-isotope catalog, 10 solar elemental abundance scales (`AG89`..`AAG21`),
//! built-in [`NuclideCatalog`] (cached in a process-wide [`OnceLock`]),
//! `reaction_Qtotal`, and single/multi-zone `get_composition_info`.
//!
//! # Architecture & Performance Notes
//! - The 347-entry built-in [`NuclideCatalog`] is initialized once per process
//!   in [`builtin_catalog`] and shared lock-free across Python and `rayon`
//!   worker threads.
//! - [`py_chem_get_composition_info_zones`] borrows a 2D C-contiguous `float64`
//!   NumPy array zero-copy via `PyReadonlyArray2::as_slice`, releases the GIL,
//!   and evaluates all 10 composition moment columns across the persistent
//!   4-thread `rayon` worker pool when `parallel=True` and `nz >= 64`.

use std::sync::OnceLock;

use flexmesa_chem::{
    AG89_TABLE, NuclideCatalog, PARALLEL_ZONE_THRESHOLD, SolarScale,
    SpeciesProps, ag89_yesol, ag89_zsol, category_id, category_name,
    chem_get_element_id, chem_get_solar, chem_m_div_h, chem_xsol,
    element_long_name, element_main_iso_name, element_name,
    eval_composition_moments_generic, generate_long_nuclide_names,
    generate_nuclide_names, get_composition_info, get_partition_fcn_indx,
    get_stable_mass_frac, lodders03_element_atom_percent, solar_cno_fraction,
    solar_element_zfrac,
};
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::eval_batch_2d;

/// Global cached built-in [`NuclideCatalog`].
///
/// Initialized lazily on first access via [`OnceLock::get_or_init`] so
/// repeated Python calls incur zero catalog construction overhead.
fn builtin_catalog() -> &'static NuclideCatalog {
    static CATALOG: OnceLock<NuclideCatalog> = OnceLock::new();
    CATALOG.get_or_init(NuclideCatalog::builtin)
}

/// Look up the 1-based atomic number $Z \in [1, 112]$ for an element symbol.
#[pyfunction]
fn py_chem_get_element_id(name: &str) -> Option<usize> {
    chem_get_element_id(name)
}

/// Return the short element symbol for $Z \in [0, 112]$.
#[pyfunction]
fn py_chem_element_name(z: usize) -> Option<String> {
    element_name(z).map(str::to_string)
}

/// Return the full element name for $Z \in [0, 112]$.
#[pyfunction]
fn py_chem_element_long_name(z: usize) -> Option<String> {
    element_long_name(z).map(str::to_string)
}

/// Return the dominant isotope name for $Z \in [1, 112]$.
#[pyfunction]
fn py_chem_element_main_iso_name(z: usize) -> Option<String> {
    element_main_iso_name(z).map(str::to_string)
}

/// Return the 1-based reaction category ID ($1..=24$) or `0` if unknown.
#[pyfunction]
fn py_chem_category_id(name: &str) -> usize {
    category_id(name)
}

/// Return the reaction category name for 1-based `id` ($1..=24$).
#[pyfunction]
fn py_chem_category_name(id: usize) -> Option<String> {
    category_name(id).map(str::to_string)
}

/// Return the 1-based partition function temperature index for $T_9$
/// ($\mathrm{GK}$).
#[pyfunction]
fn py_chem_get_partition_fcn_indx(t9: f64) -> usize {
    get_partition_fcn_indx(t9)
}

/// Generate short nuclide names from $(Z, A)$ arrays.
#[pyfunction]
fn py_chem_generate_nuclide_names(z: Vec<usize>, a: Vec<usize>) -> Vec<String> {
    generate_nuclide_names(&z, &a)
}

/// Generate long nuclide names from $(Z, A)$ arrays.
#[pyfunction]
fn py_chem_generate_long_nuclide_names(
    z: Vec<usize>,
    a: Vec<usize>,
) -> Vec<String> {
    generate_long_nuclide_names(&z, &a)
}

/// Return the 1-based `chem_id` for a nuclide name (`"o16"`, `"fe56"`, etc.).
#[pyfunction]
fn py_chem_get_nuclide_index(name: &str) -> Option<usize> {
    builtin_catalog().get_nuclide_index(name)
}

/// Return `(chem_id, name, z, n, spin, mass_excess, binding_energy, w, z53,
/// isomeric_state)` for a 1-based `chem_id`.
///
/// Raises `PyValueError` if `chem_id` is not in the catalog.
#[pyfunction]
fn py_chem_nuclide_info(
    chem_id: usize,
) -> PyResult<(usize, String, usize, usize, f64, f64, f64, f64, f64, u8)> {
    let iso = builtin_catalog().get_by_id(chem_id).ok_or_else(|| {
        PyValueError::new_err(format!("unknown chem_id {chem_id}"))
    })?;
    Ok((
        iso.chem_id,
        iso.name.clone(),
        iso.z,
        iso.n,
        iso.spin,
        iso.mass_excess,
        iso.binding_energy,
        iso.w,
        iso.z53,
        iso.isomeric_state,
    ))
}

/// Return `(z_int, z, a, w, z53)` vectors for a list of 1-based `chem_ids`.
///
/// Used by `flexmesa.chem.composition` to populate cached JAX constant arrays.
#[pyfunction]
fn py_chem_species_props(
    chem_ids: Vec<usize>,
) -> PyResult<(Vec<usize>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let props = SpeciesProps::from_catalog(builtin_catalog(), &chem_ids)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((props.z_int, props.z, props.a, props.w, props.z53))
}

/// Return the Anders & Grevesse (1989) bulk solar metallicity `zsol`.
#[pyfunction]
fn py_chem_ag89_zsol() -> f64 {
    ag89_zsol()
}

/// Return the Anders & Grevesse (1989) solar electron fraction `yesol`.
#[pyfunction]
fn py_chem_ag89_yesol() -> f64 {
    ag89_yesol()
}

/// Return the Anders & Grevesse (1989) solar mass fraction for `name`.
#[pyfunction]
fn py_chem_xsol(name: &str) -> f64 {
    chem_xsol(name)
}

/// Return `(Z, A, X_sol)` for `name` in the Anders & Grevesse (1989) table.
#[pyfunction]
fn py_chem_get_solar(name: &str) -> Option<(f64, f64, f64)> {
    chem_get_solar(name)
}

/// Return the Lodders (2003, Table 6) isotopic atom percentage for `name`.
#[pyfunction]
fn py_chem_lodders03_atom_percent(name: &str) -> f64 {
    lodders03_element_atom_percent(name)
}

/// Return the ordered 286 isotope names (`namsol(1:286)`) in the AG89 table.
#[pyfunction]
fn py_chem_ag89_namsol() -> Vec<String> {
    AG89_TABLE
        .iter()
        .map(|&(nm, _, _, _, _, _)| nm.to_string())
        .collect()
}

/// Map unstable nuclear network abundances onto the 286 stable solar isotopes.
#[pyfunction]
fn py_chem_get_stable_mass_frac(
    z: Vec<usize>,
    a: Vec<usize>,
    abun_in: Vec<f64>,
) -> Vec<f64> {
    get_stable_mass_frac(&z, &a, &abun_in)
}

/// Return the 112-element solar metal mass-fraction array for `scale`.
///
/// Raises `PyValueError` if `scale` is not one of the 10 supported scales.
#[pyfunction]
fn py_chem_solar_element_zfrac(scale: &str) -> PyResult<Vec<f64>> {
    let sc = SolarScale::from_name(scale)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(solar_element_zfrac(sc, builtin_catalog()).to_vec())
}

/// Return the CNO mass fraction of $Z$ for `scale`.
#[pyfunction]
fn py_chem_solar_cno_fraction(scale: &str) -> PyResult<f64> {
    let sc = SolarScale::from_name(scale)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(solar_cno_fraction(sc, builtin_catalog()))
}

/// Compute `[M/H]` (`chem_M_div_h`) for hydrogen fraction `x`, metallicity `z`,
/// and solar scale `scale`.
#[pyfunction]
fn py_chem_m_div_h(x: f64, z: f64, scale: &str) -> PyResult<f64> {
    let sc = SolarScale::from_name(scale)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    chem_m_div_h(x, z, sc).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Compute the total reaction $Q$-value (in $\mathrm{MeV}$) for `num_in`
/// incoming and `num_out` outgoing species.
#[pyfunction]
fn py_chem_reaction_qtotal(
    num_in: usize,
    num_out: usize,
    reactants: Vec<usize>,
) -> PyResult<f64> {
    builtin_catalog()
        .reaction_qtotal(num_in, num_out, &reactants)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Compute single-zone composition moments and optional partial derivatives.
///
/// Returns `((xh, xhe, xz, abar, zbar, z2bar, z53bar, ye, mass_correction,
/// sumx), dabar_dx, dzbar_dx, dmc_dx)`.
#[pyfunction]
fn py_chem_get_composition_info(
    chem_ids: Vec<usize>,
    x: Vec<f64>,
    skip_partials: bool,
) -> PyResult<(
    (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64),
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
)> {
    let res =
        get_composition_info(builtin_catalog(), &chem_ids, &x, skip_partials)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        (
            res.xh,
            res.xhe,
            res.xz,
            res.abar,
            res.zbar,
            res.z2bar,
            res.z53bar,
            res.ye,
            res.mass_correction,
            res.sumx,
        ),
        res.dabar_dx,
        res.dzbar_dx,
        res.dmc_dx,
    ))
}

/// Evaluate 10 composition moment columns `(xh, xhe, xz, abar, zbar, z2bar,
/// z53bar, ye, mass_correction, sumx)` across `nz` zones (`x_zones` of shape
/// `(nz, n_species)`).
#[pyfunction]
#[pyo3(signature = (chem_ids, x_zones, parallel=false))]
fn py_chem_get_composition_info_zones<'py>(
    py: Python<'py>,
    chem_ids: Vec<usize>,
    x_zones: PyReadonlyArray2<'py, f64>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = x_zones.shape();
    let nz = shape[0];
    let n_species = shape[1];
    if n_species != chem_ids.len() {
        return Err(PyValueError::new_err(format!(
            "x_zones second dimension ({n_species}) != chem_ids length ({})",
            chem_ids.len()
        )));
    }
    let props = SpeciesProps::from_catalog(builtin_catalog(), &chem_ids)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let x_slice = x_zones
        .as_slice()
        .map_err(|_| PyValueError::new_err("x_zones must be C-contiguous"))?;
    eval_batch_2d::<10, _>(py, nz, parallel, PARALLEL_ZONE_THRESHOLD, |k| {
        let row = &x_slice[k * n_species..(k + 1) * n_species];
        eval_composition_moments_generic::<f64>(&props, row).to_array()
    })
}

/// Register all Wave 2a `chem` functions on the `flexmesa._rust` module.
///
/// Attaches both Tier A static catalog queries (`py_chem_get_element_id`,
/// `py_chem_nuclide_info`, `py_chem_ag89_zsol`, `py_chem_solar_element_zfrac`,
/// `py_chem_reaction_qtotal`) and Tier B composition moment kernels
/// (`py_chem_get_composition_info`, `py_chem_get_composition_info_zones`).
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_chem_get_element_id, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_element_name, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_element_long_name, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_element_main_iso_name, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_category_id, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_category_name, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_get_partition_fcn_indx, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_generate_nuclide_names, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_generate_long_nuclide_names, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_get_nuclide_index, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_nuclide_info, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_species_props, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_ag89_zsol, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_ag89_yesol, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_xsol, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_get_solar, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_lodders03_atom_percent, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_ag89_namsol, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_get_stable_mass_frac, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_solar_element_zfrac, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_solar_cno_fraction, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_m_div_h, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_reaction_qtotal, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_get_composition_info, m)?)?;
    m.add_function(wrap_pyfunction!(py_chem_get_composition_info_zones, m)?)?;
    Ok(())
}
