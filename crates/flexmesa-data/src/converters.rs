// SPDX-License-Identifier: LGPL-3.0-or-later
//! Safe Rust ASCII-to-`.safetensors` converters for MESA nuclear and EOS tables.
//!
//! Converts MESA's `chem/data/lodders03.data`, `chem/data/isotopes.data`, and
//! `eos/helm_table.dat` into [`SafeTensorBundle`] values obeying the Mandatory
//! Index & Memory-Order Contract (`docs/design/wave-0-spec.md` §2.2).

use std::collections::BTreeMap;

use flexmesa_testing::parse_fortran_float;

use crate::safetensors::{SafeTensorBundle, TensorData};
use crate::DataError;

const PARTITION_FCN_LEN: usize = 24;
const HELM_F_COLS: [&str; 9] = [
    "f", "fd", "ft", "fdd", "ftt", "fdt", "fddt", "fdtt", "fddtt",
];
const HELM_DPDF_COLS: [&str; 4] = ["dpdf", "dpdfd", "dpdft", "dpdfdt"];
const HELM_EF_COLS: [&str; 4] = ["ef", "efd", "eft", "efdt"];
const HELM_XF_COLS: [&str; 4] = ["xf", "xfd", "xft", "xfdt"];

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

/// Parse MESA `chem/data/lodders03.data` text into a [`SafeTensorBundle`].
///
/// # Errors
/// Returns [`DataError::ParseError`] if no abundance rows are found or if any
/// row fails to parse.
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
        if tokens.len() != 5 {
            return Err(DataError::ParseError(format!(
                "malformed lodders03 row at line {}: {raw_line:?}",
                idx + 1
            )));
        }
        let z_val = tokens[0].parse::<i64>().map_err(|_| {
            DataError::ParseError(format!("invalid Z on line {}", idx + 1))
        })?;
        let el_sym = tokens[1].to_ascii_lowercase();
        let a_val = tokens[2].parse::<i64>().map_err(|_| {
            DataError::ParseError(format!("invalid A on line {}", idx + 1))
        })?;
        let pct_val = parse_fortran_float(tokens[3]).map_err(|e| {
            DataError::ParseError(format!("line {}: {e}", idx + 1))
        })?;
        let nsi_clean = tokens[4].replace(',', "");
        let nsi_val = parse_fortran_float(&nsi_clean).map_err(|e| {
            DataError::ParseError(format!("line {}: {e}", idx + 1))
        })?;

        z_vals.push(z_val);
        a_vals.push(a_val);
        pct_vals.push(pct_val);
        nsi_vals.push(nsi_val);
        iso_names.push(format!("{el_sym}{a_val}"));
    }

    if z_vals.is_empty() {
        return Err(DataError::ParseError(
            "no isotopic abundance rows found in lodders03 input".to_owned(),
        ));
    }

    let n = z_vals.len();
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "atomic_number_z".to_owned(),
        TensorData::Int64 {
            shape: vec![n],
            values: z_vals,
        },
    );
    tensors.insert(
        "mass_number_a".to_owned(),
        TensorData::Int64 {
            shape: vec![n],
            values: a_vals,
        },
    );
    tensors.insert(
        "isotopic_percent".to_owned(),
        TensorData::Float64 {
            shape: vec![n],
            values: pct_vals,
        },
    );
    tensors.insert(
        "number_abundance_si".to_owned(),
        TensorData::Float64 {
            shape: vec![n],
            values: nsi_vals,
        },
    );
    let mut metadata = BTreeMap::new();
    metadata.insert("table".to_owned(), "lodders03".to_owned());
    metadata.insert("isotopes".to_owned(), iso_names.join(","));
    Ok(SafeTensorBundle { tensors, metadata })
}

/// Parsed 4-line nuclide record (`name, W, Z, N, isomeric_state, spin, mass_excess`).
type NuclideHeader = (String, f64, i64, i64, i64, f64, f64);

/// Parse one 4-line nuclide record from `isotopes.data` and append its 24
/// partition-function values to `pf_vals`.
fn parse_nuclide_record(
    lines: &[&str],
    rec_idx: usize,
    pf_vals: &mut Vec<f64>,
) -> Result<NuclideHeader, DataError> {
    let base = rec_idx * 4;
    let hdr: Vec<&str> = lines[base].split_whitespace().collect();
    if hdr.len() != 6 {
        return Err(DataError::ParseError(format!(
            "malformed isotopes.data header at record {}: {:?}",
            rec_idx + 1,
            lines[base]
        )));
    }
    let name = hdr[0];
    let w_v = parse_fortran_float(hdr[1])
        .map_err(|e| DataError::ParseError(e.to_string()))?;
    let z_v = hdr[2].parse::<i64>().map_err(|_| {
        DataError::ParseError(format!("invalid Z for {name:?}"))
    })?;
    let n_v = hdr[3].parse::<i64>().map_err(|_| {
        DataError::ParseError(format!("invalid N for {name:?}"))
    })?;
    let iso_v = parse_isomeric_state(name)?;
    let spin_v = parse_fortran_float(hdr[4])
        .map_err(|e| DataError::ParseError(e.to_string()))?;
    let mex_v = parse_fortran_float(hdr[5])
        .map_err(|e| DataError::ParseError(e.to_string()))?;

    for offset in 1..=3 {
        let row: Vec<&str> = lines[base + offset].split_whitespace().collect();
        if row.len() != 8 {
            return Err(DataError::ParseError(format!(
                "expected 8 partition-function values for {name:?}, got {}",
                row.len()
            )));
        }
        for tok in row {
            pf_vals.push(
                parse_fortran_float(tok)
                    .map_err(|e| DataError::ParseError(e.to_string()))?,
            );
        }
    }
    Ok((name.to_owned(), w_v, z_v, n_v, iso_v, spin_v, mex_v))
}

/// Parse MESA `chem/data/isotopes.data` text into a [`SafeTensorBundle`].
///
/// # Errors
/// Returns [`DataError::ParseError`] if the non-empty line count is zero or not
/// a multiple of 4, or if any record is malformed.
pub fn parse_isotopes_text(text: &str) -> Result<SafeTensorBundle, DataError> {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if lines.is_empty() || !lines.len().is_multiple_of(4) {
        return Err(DataError::ParseError(format!(
            "isotopes.data must have a positive multiple of 4 non-empty lines, got {}",
            lines.len()
        )));
    }
    let n_nuclides = lines.len() / 4;
    let mut z_vals = Vec::with_capacity(n_nuclides);
    let mut n_vals = Vec::with_capacity(n_nuclides);
    let mut iso_vals = Vec::with_capacity(n_nuclides);
    let mut w_vals = Vec::with_capacity(n_nuclides);
    let mut spin_vals = Vec::with_capacity(n_nuclides);
    let mut mex_vals = Vec::with_capacity(n_nuclides);
    let mut pf_vals = Vec::with_capacity(n_nuclides * PARTITION_FCN_LEN);
    let mut names = Vec::with_capacity(n_nuclides);

    for rec_idx in 0..n_nuclides {
        let (name, w_v, z_v, n_v, iso_v, spin_v, mex_v) =
            parse_nuclide_record(&lines, rec_idx, &mut pf_vals)?;
        names.push(name);
        z_vals.push(z_v);
        n_vals.push(n_v);
        iso_vals.push(iso_v);
        w_vals.push(w_v);
        spin_vals.push(spin_v);
        mex_vals.push(mex_v);
    }

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "atomic_number_z".to_owned(),
        TensorData::Int64 {
            shape: vec![n_nuclides],
            values: z_vals,
        },
    );
    tensors.insert(
        "neutron_number_n".to_owned(),
        TensorData::Int64 {
            shape: vec![n_nuclides],
            values: n_vals,
        },
    );
    tensors.insert(
        "isomeric_state".to_owned(),
        TensorData::Int64 {
            shape: vec![n_nuclides],
            values: iso_vals,
        },
    );
    tensors.insert(
        "atomic_mass_w".to_owned(),
        TensorData::Float64 {
            shape: vec![n_nuclides],
            values: w_vals,
        },
    );
    tensors.insert(
        "spin".to_owned(),
        TensorData::Float64 {
            shape: vec![n_nuclides],
            values: spin_vals,
        },
    );
    tensors.insert(
        "mass_excess_mev".to_owned(),
        TensorData::Float64 {
            shape: vec![n_nuclides],
            values: mex_vals,
        },
    );
    tensors.insert(
        "partition_function".to_owned(),
        TensorData::Float64 {
            shape: vec![n_nuclides, PARTITION_FCN_LEN],
            values: pf_vals,
        },
    );
    let mut metadata = BTreeMap::new();
    metadata.insert("table".to_owned(), "isotopes".to_owned());
    metadata.insert("nuclide_names".to_owned(), names.join(","));
    Ok(SafeTensorBundle { tensors, metadata })
}

/// Parse one `jmax * imax` section of `helm_table.dat` into row-major `(imax, jmax, n_cols)`.
fn parse_helm_section(
    lines: &[&str],
    start_idx: usize,
    imax: usize,
    jmax: usize,
    n_cols: usize,
) -> Result<Vec<f64>, DataError> {
    let mut out = vec![0.0_f64; imax * jmax * n_cols];
    for j in 0..jmax {
        for i in 0..imax {
            let line = lines[start_idx + j * imax + i];
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.len() != n_cols {
                return Err(DataError::ParseError(format!(
                    "expected {n_cols} columns in helm_table row {}, got {}",
                    start_idx + j * imax + i + 1,
                    tokens.len()
                )));
            }
            let dst_base = (i * jmax + j) * n_cols;
            for (k, tok) in tokens.iter().enumerate() {
                out[dst_base + k] = parse_fortran_float(tok)
                    .map_err(|e| DataError::ParseError(e.to_string()))?;
            }
        }
    }
    Ok(out)
}

/// Insert both the packed `(imax, jmax, n_cols)` jet tensor and individual
/// `(imax, jmax)` column slices into `tensors`.
fn insert_helm_jet_and_columns(
    tensors: &mut BTreeMap<String, TensorData>,
    jet_name: &str,
    col_names: &[&str],
    jet_vals: Vec<f64>,
    imax: usize,
    jmax: usize,
) {
    let n_cols = col_names.len();
    for (col_idx, &col_name) in col_names.iter().enumerate() {
        let mut col_vec = Vec::with_capacity(imax * jmax);
        for cell in 0..(imax * jmax) {
            col_vec.push(jet_vals[cell * n_cols + col_idx]);
        }
        tensors.insert(
            col_name.to_owned(),
            TensorData::Float64 {
                shape: vec![imax, jmax],
                values: col_vec,
            },
        );
    }
    tensors.insert(
        jet_name.to_owned(),
        TensorData::Float64 {
            shape: vec![imax, jmax, n_cols],
            values: jet_vals,
        },
    );
}

/// Parse MESA `eos/helm_table.dat` text into a [`SafeTensorBundle`].
///
/// Follows MESA's `read_helm_table` loop order (`j = 0..jmax` outer temperature
/// loop, `i = 0..imax` inner density loop) and stores row-major `(imax, jmax)`
/// tensors where index `[i_rho, j_temp]` matches Fortran `h% f(i_rho + 1, j_temp + 1)`.
///
/// # Errors
/// Returns [`DataError::ParseError`] if `imax < 2`, `jmax < 2`, or the non-empty
/// line count does not equal `4 * imax * jmax`.
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
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let n_block = imax * jmax;
    if lines.len() != 4 * n_block {
        return Err(DataError::ParseError(format!(
            "expected {} non-empty rows for ({imax}, {jmax}) helm_table, got {}",
            4 * n_block,
            lines.len()
        )));
    }
    let f_jet = parse_helm_section(&lines, 0, imax, jmax, 9)?;
    let dpdf_jet = parse_helm_section(&lines, n_block, imax, jmax, 4)?;
    let ef_jet = parse_helm_section(&lines, 2 * n_block, imax, jmax, 4)?;
    let xf_jet = parse_helm_section(&lines, 3 * n_block, imax, jmax, 4)?;

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

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "log_d_grid".to_owned(),
        TensorData::Float64 {
            shape: vec![imax],
            values: log_d,
        },
    );
    tensors.insert(
        "log_t_grid".to_owned(),
        TensorData::Float64 {
            shape: vec![jmax],
            values: log_t,
        },
    );
    tensors.insert(
        "d_grid".to_owned(),
        TensorData::Float64 {
            shape: vec![imax],
            values: d_grid,
        },
    );
    tensors.insert(
        "t_grid".to_owned(),
        TensorData::Float64 {
            shape: vec![jmax],
            values: t_grid,
        },
    );
    insert_helm_jet_and_columns(
        &mut tensors,
        "f_jet",
        &HELM_F_COLS,
        f_jet,
        imax,
        jmax,
    );
    insert_helm_jet_and_columns(
        &mut tensors,
        "dpdf_jet",
        &HELM_DPDF_COLS,
        dpdf_jet,
        imax,
        jmax,
    );
    insert_helm_jet_and_columns(
        &mut tensors,
        "ef_jet",
        &HELM_EF_COLS,
        ef_jet,
        imax,
        jmax,
    );
    insert_helm_jet_and_columns(
        &mut tensors,
        "xf_jet",
        &HELM_XF_COLS,
        xf_jet,
        imax,
        jmax,
    );

    let mut metadata = BTreeMap::new();
    metadata.insert("table".to_owned(), "helm_table".to_owned());
    metadata.insert("imax".to_owned(), imax.to_string());
    metadata.insert("jmax".to_owned(), jmax.to_string());
    Ok(SafeTensorBundle { tensors, metadata })
}
