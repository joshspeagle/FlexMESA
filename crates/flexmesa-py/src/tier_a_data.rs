// SPDX-License-Identifier: LGPL-3.0-or-later
//! Tier A (`data`) `PyO3` bindings for `SHA-256` verification and ASCII table
//! conversion.
//!
//! Exposes the canonical Safe Rust implementations of:
//! - `flexmesa_data::sha256`: `is_valid_sha256_hex`, `compute_sha256_bytes`,
//!   and `compute_sha256_file` (FIPS 180-4 `SHA-256`).
//! - `flexmesa_data::converters`: `parse_lodders03_text`,
//!   `parse_isotopes_text`, and `parse_helm_table_text` returning NumPy
//!   tensors and metadata dictionaries ready for `.safetensors` serialization.

use std::path::Path;

use flexmesa_data::converters::{
    parse_helm_table_text as rust_parse_helm,
    parse_isotopes_text as rust_parse_isotopes,
    parse_lodders03_text as rust_parse_lodders03,
};
use flexmesa_data::safetensors::{SafeTensorBundle, TensorData};
use flexmesa_data::sha256::{
    compute_sha256_bytes as rust_sha256_bytes,
    compute_sha256_file as rust_sha256_file,
    is_valid_sha256_hex as rust_valid_sha256,
};
use numpy::{PyArray1, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::{PyFileNotFoundError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Converts a [`SafeTensorBundle`] into `(tensors_dict, metadata_dict)`.
///
/// Populates `tensors_dict` with shaped `float64` (`F64`) and `int64` (`I64`)
/// NumPy arrays and `metadata_dict` with string key-value pairs.
fn bundle_to_py<'py>(
    py: Python<'py>,
    bundle: SafeTensorBundle,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let tensors = PyDict::new(py);
    for (name, data) in bundle.tensors {
        match data {
            TensorData::Float64 { shape, values } => {
                let arr: Bound<'py, PyArrayDyn<f64>> =
                    PyArray1::from_vec(py, values).reshape(shape.as_slice())?;
                tensors.set_item(name, arr)?;
            }
            TensorData::Int64 { shape, values } => {
                let arr: Bound<'py, PyArrayDyn<i64>> =
                    PyArray1::from_vec(py, values).reshape(shape.as_slice())?;
                tensors.set_item(name, arr)?;
            }
        }
    }
    let meta = PyDict::new(py);
    for (k, v) in bundle.metadata {
        meta.set_item(k, v)?;
    }
    Ok((tensors, meta))
}

/// Returns `True` if `digest` is a valid 64-character hexadecimal string.
///
/// Accepts both lowercase and uppercase ASCII hexadecimal digits.
#[pyfunction]
fn is_valid_sha256_hex(digest: &str) -> bool {
    rust_valid_sha256(digest)
}

/// Computes the lowercase 64-character `SHA-256` hex digest of `data`.
///
/// Uses the pure-Rust FIPS 180-4 implementation in `flexmesa_data::sha256`.
#[pyfunction]
fn compute_sha256_bytes(data: &[u8]) -> String {
    rust_sha256_bytes(data)
}

/// Computes the lowercase 64-character `SHA-256` hex digest of a file at
/// `path`.
///
/// Raises `FileNotFoundError` if `path` does not exist or is not a regular
/// file.
#[pyfunction]
fn compute_sha256_file(path: &str) -> PyResult<String> {
    let p = Path::new(path);
    if !p.is_file() {
        return Err(PyFileNotFoundError::new_err(format!(
            "file not found: {path}"
        )));
    }
    rust_sha256_file(p).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Parses `lodders03.data` solar abundance text into `(tensors, metadata)`.
///
/// Extracts `atomic_number`, `mass_number`, and `abundance` arrays along with
/// the space-delimited `isotopes` metadata list.
#[pyfunction]
fn parse_lodders03_text<'py>(
    py: Python<'py>,
    text: &str,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let bundle = rust_parse_lodders03(text)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    bundle_to_py(py, bundle)
}

/// Parses `isotopes.data` nuclear partition table text into
/// `(tensors, metadata)`.
///
/// Extracts `atomic_number`, `neutron_number`, `atomic_mass`, `spin`,
/// `binding_energy`, and the 24-temperature `partition_function` matrix.
#[pyfunction]
fn parse_isotopes_text<'py>(
    py: Python<'py>,
    text: &str,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let bundle = rust_parse_isotopes(text)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    bundle_to_py(py, bundle)
}

/// Parses `helm_table.dat` electron-positron EOS text into
/// `(tensors, metadata)`.
///
/// Extracts the 9 Helmholtz free-energy tables (`f`..`fddt`), the 4 pressure
/// derivative tables (`dpdf`..`dpdfdt`), and the 3 electron number density
/// tables (`ef`..`efdt`) on the `(imax, jmax)` `(logd, logt)` grid.
#[pyfunction]
fn parse_helm_table_text<'py>(
    py: Python<'py>,
    text: &str,
    imax: usize,
    jmax: usize,
    logdlo: f64,
    logdhi: f64,
    logtlo: f64,
    logthi: f64,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let bundle =
        rust_parse_helm(text, imax, jmax, logdlo, logdhi, logtlo, logthi)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    bundle_to_py(py, bundle)
}

/// Registers Tier A `data` functions on `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_valid_sha256_hex, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sha256_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sha256_file, m)?)?;
    m.add_function(wrap_pyfunction!(parse_lodders03_text, m)?)?;
    m.add_function(wrap_pyfunction!(parse_isotopes_text, m)?)?;
    m.add_function(wrap_pyfunction!(parse_helm_table_text, m)?)?;
    Ok(())
}
