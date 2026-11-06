// SPDX-License-Identifier: LGPL-3.0-or-later
//! Tier A (`utils`, `testing`, `schema`) `PyO3` bindings.
//!
//! Exposes the canonical Safe Rust implementations of:
//! - `flexmesa_utils`: IEEE-754 finiteness checks (`is_bad`, `is_bad_array`,
//!   `replace_bad_nums`), string normalizers (`remove_underbars`,
//!   `split_line`), and the `.net` stream lexer (`tokenize`).
//! - `flexmesa_testing`: Fortran `D`/`d`/exponent-less float tokenizers
//!   (`is_fortran_float_token`, `parse_fortran_float`) and `test_output`
//!   golden file parser (`parse_golden_text`).
//! - `flexmesa_schema`: Fortran literal parser (`parse_control_value`),
//!   `.defaults` schema extractor (`parse_defaults_text`), `*_columns.list`
//!   extractor (`parse_columns_text`), and `&namelist ... /` parser
//!   (`parse_namelist_text`).

use flexmesa_schema::{ControlType, ControlValue};
use flexmesa_utils::Token;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Converts a [`ControlValue`] into a `(type_str, Py<PyAny>)` pair for Python.
///
/// Maps `Logical`, `Integer`, `Float`, and `String` variants to their native
/// Python counterparts (`bool`, `int`, `float`, `str`).
fn control_val_to_py(
    py: Python<'_>,
    val: &ControlValue,
) -> PyResult<(&'static str, Py<PyAny>)> {
    match val {
        ControlValue::Logical(b) => {
            let obj = b.into_pyobject(py)?.to_owned().into_any().unbind();
            Ok(("logical", obj))
        }
        ControlValue::Integer(i) => {
            Ok(("integer", i.into_pyobject(py)?.into_any().unbind()))
        }
        ControlValue::Float(f) => {
            Ok(("float", f.into_pyobject(py)?.into_any().unbind()))
        }
        ControlValue::String(s) => {
            Ok(("string", s.into_pyobject(py)?.into_any().unbind()))
        }
    }
}

/// Converts a [`ControlType`] enum into its lowercase string tag.
const fn control_type_str(ct: ControlType) -> &'static str {
    match ct {
        ControlType::Logical => "logical",
        ControlType::Integer => "integer",
        ControlType::Float => "float",
        ControlType::String => "string",
    }
}

/// Returns `True` when scalar `x` is `NaN` or $\pm\infty$.
///
/// Matches upstream MESA `utils_lib::is_bad`.
#[pyfunction]
fn is_bad(x: f64) -> bool {
    flexmesa_utils::is_bad(x)
}

/// Returns `True` if any element in the 1D `f64` array `xs` is non-finite.
///
/// Inspects the contiguous buffer zero-copy via `flexmesa_utils::is_bad_slice`.
#[pyfunction]
fn is_bad_array(xs: PyReadonlyArray1<'_, f64>) -> PyResult<bool> {
    let slice = xs
        .as_slice()
        .map_err(|_| PyValueError::new_err("xs must be C-contiguous"))?;
    Ok(flexmesa_utils::is_bad_slice(slice))
}

/// Replaces non-finite elements in 1D `f64` array `xs` with `replacement`.
///
/// Uses branch-free conditional selection rather than multiplicative masking
/// so `0.0 * NaN` never propagates.
#[pyfunction]
fn replace_bad_nums<'py>(
    py: Python<'py>,
    xs: PyReadonlyArray1<'py, f64>,
    replacement: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice = xs
        .as_slice()
        .map_err(|_| PyValueError::new_err("xs must be C-contiguous"))?;
    Ok(PyArray1::from_vec(
        py,
        flexmesa_utils::replace_bad_nums(slice, replacement),
    ))
}

/// Replaces underscores with spaces after trimming trailing whitespace.
///
/// Matches upstream MESA `utils_lib::remove_underbars`.
#[pyfunction]
fn remove_underbars(text: &str) -> String {
    flexmesa_utils::remove_underbars(text)
}

/// Splits `line` on whitespace into at most `num` tokens.
///
/// Matches upstream MESA `utils_lib::split_line`.
#[pyfunction]
fn split_line(line: &str, num: usize) -> Vec<String> {
    flexmesa_utils::split_line(line, num)
}

/// Tokenizes MESA `.net` or list text into `(kind_str, value_str)` pairs.
///
/// Returns tokens terminated by `("eof", "")`.
#[pyfunction]
fn tokenize(text: &str) -> Vec<(&'static str, String)> {
    flexmesa_utils::tokenize(text)
        .into_iter()
        .map(|tok| match tok {
            Token::Name(s) => ("name", s),
            Token::String(s) => ("string", s),
            Token::LeftParen => ("left_paren", "(".to_string()),
            Token::RightParen => ("right_paren", ")".to_string()),
            Token::Comma => ("comma", ",".to_string()),
            Token::Eof => ("eof", String::new()),
        })
        .collect()
}

/// Returns `True` if `token` is a Fortran floating-point literal.
///
/// Recognizes standard IEEE literals, Fortran `D`/`d` exponents, and
/// exponent-less `1.0-10` notation.
#[pyfunction]
fn is_fortran_float_token(token: &str) -> bool {
    flexmesa_testing::is_fortran_float_token(token)
}

/// Parses a Fortran or IEEE-754 floating-point literal into `f64`.
///
/// Raises `ValueError` if `token` is not a valid floating-point literal.
#[pyfunction]
fn parse_fortran_float(token: &str) -> PyResult<f64> {
    flexmesa_testing::parse_fortran_float(token)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Parses MESA golden `test_output` text into a list of tuples
/// `(line_number, section, label, indices, values_array)`.
#[pyfunction]
fn parse_golden_text<'py>(
    py: Python<'py>,
    text: &str,
) -> PyResult<
    Vec<(usize, String, String, Vec<i64>, Bound<'py, PyArray1<f64>>)>,
> {
    let records = flexmesa_testing::parse_golden_text(text)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(records
        .into_iter()
        .map(|r| {
            (
                r.line_number,
                r.section,
                r.label,
                r.indices,
                PyArray1::from_vec(py, r.values),
            )
        })
        .collect())
}

/// Parses a single Fortran control literal into `(type_str, value)`.
///
/// Supports `.true.`/`.false.`, integers, Fortran `d`/`e` floats, and single-
/// or double-quoted strings.
#[pyfunction]
fn parse_control_value(
    py: Python<'_>,
    raw: &str,
) -> PyResult<(&'static str, Py<PyAny>)> {
    let val = flexmesa_schema::parse_control_value(raw, 1)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    control_val_to_py(py, &val)
}

/// Parses a MESA `.defaults` text into a list of tuples
/// `(name, type_str, default_val, is_array, overrides, section, doc, line_no)`.
#[pyfunction]
fn parse_defaults_text(
    py: Python<'_>,
    text: &str,
) -> PyResult<
    Vec<(
        String,
        &'static str,
        Py<PyAny>,
        bool,
        Vec<(String, Py<PyAny>)>,
        String,
        String,
        usize,
    )>,
> {
    let controls = flexmesa_schema::parse_defaults_text(text)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut out = Vec::with_capacity(controls.len());
    for c in controls {
        let (_, def_py) = control_val_to_py(py, &c.default_value)?;
        let mut ov_py = Vec::with_capacity(c.element_overrides.len());
        for (sub, v) in &c.element_overrides {
            let (_, vp) = control_val_to_py(py, v)?;
            ov_py.push((sub.clone(), vp));
        }
        out.push((
            c.name,
            control_type_str(c.control_type),
            def_py,
            c.is_array,
            ov_py,
            c.section,
            c.doc,
            c.line_number,
        ));
    }
    Ok(out)
}

/// Parses a MESA `*_columns.list` text into `(name, section, comment,
/// line_no)`.
#[pyfunction]
fn parse_columns_text(text: &str) -> Vec<(String, String, String, usize)> {
    flexmesa_schema::parse_columns_text(text)
        .into_iter()
        .map(|e| (e.name, e.section, e.comment, e.line_number))
        .collect()
}

/// Parses Fortran `&namelist ... /` text into a list of
/// `(block_name, Vec<(name, subscript, type_str, val, line_no)>)`.
#[pyfunction]
fn parse_namelist_text(
    py: Python<'_>,
    text: &str,
) -> PyResult<
    Vec<(
        String,
        Vec<(String, Option<String>, &'static str, Py<PyAny>, usize)>,
    )>,
> {
    let blocks = flexmesa_schema::parse_namelist_text(text)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut out = Vec::with_capacity(blocks.len());
    for b in blocks {
        let mut assigns = Vec::with_capacity(b.assignments.len());
        for a in b.assignments {
            let (t_str, v_py) = control_val_to_py(py, &a.value)?;
            assigns.push((a.name, a.subscript, t_str, v_py, a.line_number));
        }
        out.push((b.name, assigns));
    }
    Ok(out)
}

/// Registers Tier A `utils`, `testing`, and `schema` functions on `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_bad, m)?)?;
    m.add_function(wrap_pyfunction!(is_bad_array, m)?)?;
    m.add_function(wrap_pyfunction!(replace_bad_nums, m)?)?;
    m.add_function(wrap_pyfunction!(remove_underbars, m)?)?;
    m.add_function(wrap_pyfunction!(split_line, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(is_fortran_float_token, m)?)?;
    m.add_function(wrap_pyfunction!(parse_fortran_float, m)?)?;
    m.add_function(wrap_pyfunction!(parse_golden_text, m)?)?;
    m.add_function(wrap_pyfunction!(parse_control_value, m)?)?;
    m.add_function(wrap_pyfunction!(parse_defaults_text, m)?)?;
    m.add_function(wrap_pyfunction!(parse_columns_text, m)?)?;
    m.add_function(wrap_pyfunction!(parse_namelist_text, m)?)?;
    Ok(())
}
