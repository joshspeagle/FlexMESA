// SPDX-License-Identifier: LGPL-3.0-or-later
//! Safe Rust ASCII-to-`.safetensors` converters for MESA nuclear and EOS
//! tables.
//!
//! Converts MESA's `chem/data/lodders03.data`, `chem/data/isotopes.data`, and
//! `eos/helm_table.dat` into [`SafeTensorBundle`] values obeying the Mandatory
//! Index & Memory-Order Contract (`docs/design/wave-0-spec.md` §2.2):
//! - `lodders03.data`: 1-D arrays of length $N_{\text{iso}}$
//!   (`atomic_number_z`, `mass_number_a`, `isotopic_percent`,
//!   `number_abundance_si`).
//! - `isotopes.data`: 1-D arrays of length $N_{\text{nuc}}$ plus row-major 2-D
//!   `partition_function` of shape $(N_{\text{nuc}}, 24)$.
//! - `helm_table.dat`: row-major 2-D arrays of shape $(i_{\max}, j_{\max})$ and
//!   3-D jet arrays of shape $(i_{\max}, j_{\max}, N_{\text{col}})$ indexed as
//!   `[i_rho, j_temp]`.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use flexmesa_testing::parse_fortran_float;

use crate::safetensors::{save_safetensors_file, SafeTensorBundle, TensorData};
use crate::DataError;

/// Number of temperature grid points per nuclide in `isotopes.data` (`3 * 8`).
const PARTITION_FCN_LEN: usize = 24;
/// Helmholtz free-energy jet column names (`f` through `fddtt`).
const HELM_F_COLS: [&str; 9] = [
    "f", "fd", "ft", "fdd", "ftt", "fdt", "fddt", "fdtt", "fddtt",
];
/// Helmholtz pressure-derivative jet column names (`dpdf` through `dpdfdt`).
const HELM_DPDF_COLS: [&str; 4] = ["dpdf", "dpdfd", "dpdft", "dpdfdt"];
/// Helmholtz electron chemical-potential jet column names (`ef` through `efdt`).
const HELM_EF_COLS: [&str; 4] = ["ef", "efd", "eft", "efdt"];
/// Helmholtz number-density jet column names (`xf` through `xfdt`).
const HELM_XF_COLS: [&str; 4] = ["xf", "xfd", "xft", "xfdt"];

/// Parse a Fortran floating-point token, mapping errors to
/// [`DataError::ParseError`].
fn parse_float_tok(token: &str) -> Result<f64, DataError> {
    parse_fortran_float(token).map_err(|e| DataError::ParseError(e.to_string()))
}

/// Parse a 64-bit signed integer token, mapping errors to
/// [`DataError::ParseError`].
fn parse_int_tok(token: &str, label: &str) -> Result<i64, DataError> {
    token.parse::<i64>().map_err(|_| {
        DataError::ParseError(format!("invalid {label} token {token:?}"))
    })
}

/// Collect stripped non-empty lines from `text`.
fn non_empty_lines(text: &str) -> Vec<&str> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect()
}

/// Parse the integer isomeric state suffix (`al26-1` -> `1`, else `0`).
fn parse_isomeric_state(name: &str) -> Result<i64, DataError> {
    let Some((_, suffix)) = name.split_once('-') else {
        return Ok(0);
    };
    let state = suffix.parse::<i64>().map_err(|_| {
        DataError::ParseError(format!(
            "invalid isomeric state in nuclide name {name:?}"
        ))
    })?;
    if !(0..=99).contains(&state) {
        return Err(DataError::ParseError(format!(
            "isomeric state {state} out of range [0, 99] in {name:?}"
        )));
    }
    Ok(state)
}

/// Build a [`BTreeMap`] of named tensors from static string keys.
fn tensor_map<const N: usize>(
    entries: [(&str, TensorData); N],
) -> BTreeMap<String, TensorData> {
    entries
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect()
}

/// Convert an ASCII table file on disk to `.safetensors` and return its
/// lowercase 64-character `SHA-256` hex digest.
///
/// # Errors
/// Returns [`DataError::IoError`] if `input_path` cannot be read or
/// `output_path` cannot be written, or [`DataError::ParseError`] if `parser`
/// rejects the table contents.
pub fn convert_ascii_table_file<F>(
    input_path: &Path,
    output_path: &Path,
    parser: F,
) -> Result<String, DataError>
where
    F: FnOnce(&str) -> Result<SafeTensorBundle, DataError>,
{
    let text =
        fs::read_to_string(input_path).map_err(|err| DataError::IoError {
            path: input_path.display().to_string(),
            message: err.to_string(),
        })?;
    save_safetensors_file(output_path, &parser(&text)?, true)
}

/// Parse MESA `chem/data/lodders03.data` text into a [`SafeTensorBundle`].
///
/// Extracts 1-D `I64` arrays `atomic_number_z` and `mass_number_a`, plus 1-D
/// `F64` arrays `isotopic_percent` and `number_abundance_si` (stripping comma
/// thousands separators such as `768,900` on the $^{56}\text{Fe}$ entry).
///
/// # Errors
/// Returns [`DataError::ParseError`] if no abundance rows are found or if any
/// row fails to parse.
///
/// # Examples
/// ```
/// use flexmesa_data::converters::parse_lodders03_text;
///
/// let bundle = parse_lodders03_text("1 H 1 99.99806 2.431e10\n").unwrap();
/// let z = bundle.tensors["atomic_number_z"].as_i64_slice();
/// assert_eq!(z, Some(&[1][..]));
/// ```
pub fn parse_lodders03_text(text: &str) -> Result<SafeTensorBundle, DataError> {
    let mut z_vals = Vec::new();
    let mut a_vals = Vec::new();
    let mut pct_vals = Vec::new();
    let mut nsi_vals = Vec::new();
    let mut iso_names = Vec::new();

    for (idx, raw_line) in text.lines().enumerate() {
        let code = raw_line.split('!').next().unwrap_or("").trim();
        if code.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = code.split_whitespace().collect();
        let [z_s, el_s, a_s, pct_s, nsi_s] = tokens.as_slice() else {
            return Err(DataError::ParseError(format!(
                "malformed lodders03 row at line {}: {raw_line:?}",
                idx + 1
            )));
        };
        let z_val = parse_int_tok(z_s, "Z")?;
        let a_val = parse_int_tok(a_s, "A")?;
        let nsi_raw: Cow<'_, str> = if nsi_s.contains(',') {
            Cow::Owned(nsi_s.replace(',', ""))
        } else {
            Cow::Borrowed(nsi_s)
        };
        z_vals.push(z_val);
        a_vals.push(a_val);
        pct_vals.push(parse_float_tok(pct_s)?);
        nsi_vals.push(parse_float_tok(&nsi_raw)?);
        iso_names.push(format!("{}{a_val}", el_s.to_ascii_lowercase()));
    }
    if z_vals.is_empty() {
        return Err(DataError::ParseError(
            "no isotopic abundance rows found in lodders03 input".to_owned(),
        ));
    }
    let n = z_vals.len();
    let tensors = tensor_map([
        ("atomic_number_z", TensorData::i64(vec![n], z_vals)),
        ("mass_number_a", TensorData::i64(vec![n], a_vals)),
        ("isotopic_percent", TensorData::f64(vec![n], pct_vals)),
        ("number_abundance_si", TensorData::f64(vec![n], nsi_vals)),
    ]);
    let metadata = BTreeMap::from([
        ("table".to_owned(), "lodders03".to_owned()),
        ("isotopes".to_owned(), iso_names.join(",")),
    ]);
    Ok(SafeTensorBundle { tensors, metadata })
}

/// Parse MESA `chem/data/isotopes.data` text into a [`SafeTensorBundle`].
///
/// Each nuclide record spans 4 non-empty lines: a 6-token header
/// (`name W Z N spin mass_excess_mev`) followed by 3 lines of 8
/// partition-function values (`24` temperature grid points per nuclide).
///
/// # Errors
/// Returns [`DataError::ParseError`] if the non-empty line count is zero or not
/// a multiple of 4, or if any record is malformed.
pub fn parse_isotopes_text(text: &str) -> Result<SafeTensorBundle, DataError> {
    let lines = non_empty_lines(text);
    if lines.is_empty() || !lines.len().is_multiple_of(4) {
        return Err(DataError::ParseError(format!(
            "isotopes.data must have a positive multiple of 4 lines, got {}",
            lines.len()
        )));
    }
    let n = lines.len() / 4;
    let mut z_vals = Vec::with_capacity(n);
    let mut n_vals = Vec::with_capacity(n);
    let mut iso_vals = Vec::with_capacity(n);
    let mut w_vals = Vec::with_capacity(n);
    let mut spin_vals = Vec::with_capacity(n);
    let mut mex_vals = Vec::with_capacity(n);
    let mut pf_vals = Vec::with_capacity(n * PARTITION_FCN_LEN);
    let mut names = Vec::with_capacity(n);

    for (rec_idx, rec) in lines.as_chunks::<4>().0.iter().enumerate() {
        let hdr: Vec<&str> = rec[0].split_whitespace().collect();
        let [name, w_s, z_s, n_s, spin_s, mex_s] = hdr.as_slice() else {
            return Err(DataError::ParseError(format!(
                "malformed isotopes.data header at record {}: {:?}",
                rec_idx + 1,
                rec[0]
            )));
        };
        names.push((*name).to_owned());
        w_vals.push(parse_float_tok(w_s)?);
        z_vals.push(parse_int_tok(z_s, "Z")?);
        n_vals.push(parse_int_tok(n_s, "N")?);
        iso_vals.push(parse_isomeric_state(name)?);
        spin_vals.push(parse_float_tok(spin_s)?);
        mex_vals.push(parse_float_tok(mex_s)?);
        for pf_line in &rec[1..4] {
            let mut count = 0usize;
            for tok in pf_line.split_whitespace() {
                pf_vals.push(parse_float_tok(tok)?);
                count += 1;
            }
            if count != 8 {
                return Err(DataError::ParseError(format!(
                    "expected 8 partition values for {name:?}, got {count}"
                )));
            }
        }
    }

    let tensors = tensor_map([
        ("atomic_number_z", TensorData::i64(vec![n], z_vals)),
        ("neutron_number_n", TensorData::i64(vec![n], n_vals)),
        ("isomeric_state", TensorData::i64(vec![n], iso_vals)),
        ("atomic_mass_w", TensorData::f64(vec![n], w_vals)),
        ("spin", TensorData::f64(vec![n], spin_vals)),
        ("mass_excess_mev", TensorData::f64(vec![n], mex_vals)),
        (
            "partition_function",
            TensorData::f64(vec![n, PARTITION_FCN_LEN], pf_vals),
        ),
    ]);
    let metadata = BTreeMap::from([
        ("table".to_owned(), "isotopes".to_owned()),
        ("nuclide_names".to_owned(), names.join(",")),
    ]);
    Ok(SafeTensorBundle { tensors, metadata })
}

/// Parse one `jmax * imax` section of `helm_table.dat` into
/// `(imax, jmax, n_cols)` row-major order and insert both the packed 3-D jet
/// tensor (`jet_name`) and its 2-D `(imax, jmax)` column slices into `tensors`.
fn parse_and_insert_helm_section(
    lines: &[&str],
    start_idx: usize,
    imax: usize,
    jmax: usize,
    jet_name: &str,
    col_names: &[&str],
    tensors: &mut BTreeMap<String, TensorData>,
) -> Result<(), DataError> {
    let n_cols = col_names.len();
    let mut jet_vals = vec![0.0_f64; imax * jmax * n_cols];
    for j in 0..jmax {
        for i in 0..imax {
            let line = lines[start_idx + j * imax + i];
            let dst_base = (i * jmax + j) * n_cols;
            let mut k = 0usize;
            for tok in line.split_whitespace() {
                if k >= n_cols {
                    return Err(DataError::ParseError(format!(
                        "too many columns in helm_table row {}",
                        start_idx + j * imax + i + 1
                    )));
                }
                jet_vals[dst_base + k] = parse_float_tok(tok)?;
                k += 1;
            }
            if k != n_cols {
                return Err(DataError::ParseError(format!(
                    "expected {n_cols} columns in helm_table row {}, got {k}",
                    start_idx + j * imax + i + 1
                )));
            }
        }
    }
    for (col_idx, &col_name) in col_names.iter().enumerate() {
        let col_vec: Vec<f64> = (0..(imax * jmax))
            .map(|cell| jet_vals[cell * n_cols + col_idx])
            .collect();
        tensors.insert(
            col_name.to_owned(),
            TensorData::f64(vec![imax, jmax], col_vec),
        );
    }
    tensors.insert(
        jet_name.to_owned(),
        TensorData::f64(vec![imax, jmax, n_cols], jet_vals),
    );
    Ok(())
}

/// Parse MESA `eos/helm_table.dat` text into a [`SafeTensorBundle`].
///
/// Follows MESA's `read_helm_table` loop order (`j = 0..jmax` outer temperature
/// loop, `i = 0..imax` inner density loop) and stores row-major `(imax, jmax)`
/// tensors where `[i_rho, j_temp]` matches Fortran
/// `h% f(i_rho + 1, j_temp + 1)`.
///
/// Constructs the log10 and linear density/temperature coordinate grids
/// (`log_d_grid`, `log_t_grid`, `d_grid`, `t_grid`) together with all 21
/// scalar tables and 4 packed 3-D jet tensors (`f_jet`, `dpdf_jet`, `ef_jet`,
/// `xf_jet`).
///
/// # Errors
/// Returns [`DataError::ParseError`] if `imax < 2`, `jmax < 2`, or the
/// non-empty line count does not equal `4 * imax * jmax`.
#[allow(clippy::cast_precision_loss)]
pub fn parse_helm_table_text(
    text: &str,
    imax: usize,
    jmax: usize,
    log_rho_min: f64,
    log_rho_max: f64,
    log_temp_min: f64,
    log_temp_max: f64,
) -> Result<SafeTensorBundle, DataError> {
    if imax < 2 || jmax < 2 {
        return Err(DataError::ParseError(format!(
            "imax and jmax must be >= 2, got ({imax}, {jmax})"
        )));
    }
    let lines = non_empty_lines(text);
    let n_block = imax * jmax;
    if lines.len() != 4 * n_block {
        return Err(DataError::ParseError(format!(
            "expected {} rows for ({imax}, {jmax}) helm_table, got {}",
            4 * n_block,
            lines.len()
        )));
    }
    let d_step = (log_rho_max - log_rho_min) / ((imax - 1) as f64);
    let t_step = (log_temp_max - log_temp_min) / ((jmax - 1) as f64);
    let log_d: Vec<f64> = (0..imax)
        .map(|i| log_rho_min + (i as f64) * d_step)
        .collect();
    let log_t: Vec<f64> = (0..jmax)
        .map(|j| log_temp_min + (j as f64) * t_step)
        .collect();
    let d_grid: Vec<f64> = log_d.iter().map(|&x| 10.0_f64.powf(x)).collect();
    let t_grid: Vec<f64> = log_t.iter().map(|&x| 10.0_f64.powf(x)).collect();

    let mut tensors = tensor_map([
        ("log_d_grid", TensorData::f64(vec![imax], log_d)),
        ("log_t_grid", TensorData::f64(vec![jmax], log_t)),
        ("d_grid", TensorData::f64(vec![imax], d_grid)),
        ("t_grid", TensorData::f64(vec![jmax], t_grid)),
    ]);
    for (sec, jet_name, cols) in [
        (0usize, "f_jet", &HELM_F_COLS[..]),
        (1, "dpdf_jet", &HELM_DPDF_COLS[..]),
        (2, "ef_jet", &HELM_EF_COLS[..]),
        (3, "xf_jet", &HELM_XF_COLS[..]),
    ] {
        parse_and_insert_helm_section(
            &lines,
            sec * n_block,
            imax,
            jmax,
            jet_name,
            cols,
            &mut tensors,
        )?;
    }

    let metadata = BTreeMap::from([
        ("table".to_owned(), "helm_table".to_owned()),
        ("imax".to_owned(), imax.to_string()),
        ("jmax".to_owned(), jmax.to_string()),
    ]);
    Ok(SafeTensorBundle { tensors, metadata })
}
