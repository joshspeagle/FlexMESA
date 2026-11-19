// SPDX-License-Identifier: LGPL-3.0-or-later
//! Binary (`FMDTRW01`) and ASCII (`.data`) loaders and 2D bicubic spline builders
//! for `(X, Z)` `(logQ, logT)` EOS slices (`eos/private/eosdt_load_tables.f90`).

use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use flexmesa_const::LN10;
use flexmesa_utils::is_bad;

use crate::tables_4d::spline::{
    build_4d_slice_vertices, validate_4d_grid_header, EosDtXzTable,
    NUM_EOS_TABLE_VALS,
};
use crate::types::{
    EosError, I_CHI_RHO, I_CHI_T, I_CP, I_CV, I_DE_DRHO, I_DS_DRHO, I_DS_DT,
    I_ETA, I_GAMMA1, I_GAMMA3, I_GRAD_AD, I_LN_E, I_LN_FREE_E, I_LN_PGAS,
    I_LN_S, I_MU,
};

/// Magic header for compact raw `(logQ, logT)` `OPAL/SCVH` and `FreeEOS` binary slices.
pub const DT_RAW_BIN_MAGIC: &[u8; 8] = b"FMDTRW01";

impl EosDtXzTable {
    #[inline]
    fn store_mapped_cell(
        raw_vars: &mut [f64],
        ngrid: usize,
        cell: usize,
        file_vals: &[f64; NUM_EOS_TABLE_VALS],
    ) {
        raw_vars[I_LN_PGAS * ngrid + cell] = file_vals[0] * LN10;
        raw_vars[I_LN_E * ngrid + cell] = file_vals[1] * LN10;
        raw_vars[I_LN_S * ngrid + cell] = file_vals[2] * LN10;
        raw_vars[I_GRAD_AD * ngrid + cell] = file_vals[14];
        raw_vars[I_CHI_RHO * ngrid + cell] = file_vals[3];
        raw_vars[I_CHI_T * ngrid + cell] = file_vals[4];
        raw_vars[I_CP * ngrid + cell] = file_vals[5];
        raw_vars[I_CV * ngrid + cell] = file_vals[6];
        raw_vars[I_DE_DRHO * ngrid + cell] = file_vals[7];
        raw_vars[I_DS_DT * ngrid + cell] = file_vals[8];
        raw_vars[I_DS_DRHO * ngrid + cell] = file_vals[9];
        raw_vars[I_MU * ngrid + cell] = file_vals[10];
        raw_vars[I_LN_FREE_E * ngrid + cell] = file_vals[11].max(-4.0) * LN10;
        raw_vars[I_GAMMA1 * ngrid + cell] = file_vals[12];
        raw_vars[I_GAMMA3 * ngrid + cell] = file_vals[13];
        raw_vars[I_ETA * ngrid + cell] = file_vals[15];
    }

    #[allow(clippy::too_many_arguments)]
    fn build_from_raw_vars(
        table_label: &str,
        version: i32,
        x: f64,
        z: f64,
        num_log_ts: usize,
        log_t_min: f64,
        log_t_max: f64,
        del_log_t: f64,
        num_log_qs: usize,
        log_q_min: f64,
        log_q_max: f64,
        del_log_q: f64,
        raw_vars: &[f64],
    ) -> Result<Self, EosError> {
        let log_qs: Vec<f64> = (0..num_log_qs)
            .map(|iq| log_q_min + (iq as f64) * del_log_q)
            .collect();
        let log_ts: Vec<f64> = (0..num_log_ts)
            .map(|jt| log_t_min + (jt as f64) * del_log_t)
            .collect();
        let dense = build_4d_slice_vertices(
            "EosDtXzTable::build_from_raw_vars",
            table_label,
            &log_qs,
            &log_ts,
            raw_vars,
        )?;

        Ok(Self {
            version,
            x,
            z,
            num_log_ts,
            log_t_min,
            log_t_max,
            del_log_t,
            num_log_qs,
            log_q_min,
            log_q_max,
            del_log_q,
            dense: Some(dense),
        })
    }

    /// Reads a compact `FMDTRW01` raw binary slice (`eos_4d_raw/*.raw`) and fits its 16 bicubic splines.
    pub fn from_raw_bin(path: &Path) -> Result<Self, EosError> {
        let table_label = path.display().to_string();
        let err = |msg: String| EosError::TableIo {
            context: "EosDtXzTable::from_raw_bin",
            detail: format!("{table_label}: {msg}"),
        };
        let file = std::fs::File::open(path).map_err(|e| err(e.to_string()))?;
        let mut reader = BufReader::with_capacity(512 * 1024, file);
        let mut hdr = [0u8; 84];
        reader
            .read_exact(&mut hdr)
            .map_err(|e| err(e.to_string()))?;
        if &hdr[0..8] != DT_RAW_BIN_MAGIC {
            return Err(err("invalid FMDTRW01 magic".to_string()));
        }
        let f64_at = |off: usize| -> Result<f64, EosError> {
            let bytes: [u8; 8] = hdr[off..off + 8]
                .try_into()
                .map_err(|_| err("short FMDTRW01 header slice".to_string()))?;
            Ok(f64::from_le_bytes(bytes))
        };
        let i32_at = |off: usize| -> Result<i32, EosError> {
            let bytes: [u8; 4] = hdr[off..off + 4]
                .try_into()
                .map_err(|_| err("short FMDTRW01 header slice".to_string()))?;
            Ok(i32::from_le_bytes(bytes))
        };
        let u32_at = |off: usize| -> Result<usize, EosError> {
            let bytes: [u8; 4] = hdr[off..off + 4]
                .try_into()
                .map_err(|_| err("short FMDTRW01 header slice".to_string()))?;
            Ok(u32::from_le_bytes(bytes) as usize)
        };
        let version = i32_at(8)?;
        let x = f64_at(12)?;
        let z = f64_at(20)?;
        let num_log_ts = u32_at(28)?;
        let log_t_min = f64_at(32)?;
        let log_t_max = f64_at(40)?;
        let del_log_t = f64_at(48)?;
        let num_log_qs = u32_at(56)?;
        let log_q_min = f64_at(60)?;
        let log_q_max = f64_at(68)?;
        let del_log_q = f64_at(76)?;

        let ngrid = validate_4d_grid_header(
            "EosDtXzTable::from_raw_bin",
            &table_label,
            x,
            z,
            num_log_qs,
            log_q_min,
            log_q_max,
            del_log_q,
            num_log_ts,
            log_t_min,
            log_t_max,
            del_log_t,
        )?;
        let mut raw_vars = vec![0.0_f64; NUM_EOS_TABLE_VALS * ngrid];
        let mut row_bytes = [0u8; NUM_EOS_TABLE_VALS * 8];
        for cell in 0..ngrid {
            reader
                .read_exact(&mut row_bytes)
                .map_err(|e| err(e.to_string()))?;
            let iq = cell / num_log_ts;
            let jt = cell % num_log_ts;
            let mut file_vals = [0.0_f64; NUM_EOS_TABLE_VALS];
            for (k, slot) in file_vals.iter_mut().enumerate() {
                let s = k * 8;
                let bytes: [u8; 8] =
                    row_bytes[s..s + 8].try_into().map_err(|_| {
                        err("short FMDTRW01 cell slice".to_string())
                    })?;
                let val = f64::from_le_bytes(bytes);
                if is_bad(val) {
                    return Err(err(format!(
                        "non-finite raw column {k} ({val}) at cell=({iq}, {jt})"
                    )));
                }
                *slot = val;
            }
            Self::store_mapped_cell(&mut raw_vars, ngrid, cell, &file_vals);
        }
        let mut trailing = [0_u8; 1];
        if reader.read(&mut trailing).map_err(|e| err(e.to_string()))? != 0 {
            return Err(err(
                "unexpected trailing bytes after FMDTRW01 payload".to_string(),
            ));
        }

        Self::build_from_raw_vars(
            &table_label,
            version,
            x,
            z,
            num_log_ts,
            log_t_min,
            log_t_max,
            del_log_t,
            num_log_qs,
            log_q_min,
            log_q_max,
            del_log_q,
            &raw_vars,
        )
    }

    /// Parses an ASCII `.data` table and builds all 16 bicubic splines (`Load1_eosDT_Table`).
    pub fn from_ascii_reader<R: BufRead>(reader: R) -> Result<Self, EosError> {
        let mut lines = reader
            .lines()
            .map(|l| {
                l.map_err(|e| EosError::InvalidState {
                    context: "EosDtXzTable::from_ascii_reader",
                    detail: e.to_string(),
                })
            })
            .filter(|res| match res {
                Ok(s) => !s.trim().is_empty(),
                Err(_) => true,
            });

        let _header_names =
            lines.next().ok_or_else(|| EosError::InvalidState {
                context: "EosDtXzTable::from_ascii_reader",
                detail: "missing header names line".to_string(),
            })??;
        let header_vals_line =
            lines.next().ok_or_else(|| EosError::InvalidState {
                context: "EosDtXzTable::from_ascii_reader",
                detail: "missing header values line".to_string(),
            })??;

        let hdr: Vec<f64> = header_vals_line
            .split_whitespace()
            .map(|t| {
                t.parse::<f64>().map_err(|e| EosError::InvalidState {
                    context: "EosDtXzTable::from_ascii_reader",
                    detail: format!("invalid header float '{t}': {e}"),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if hdr.len() < 11 {
            return Err(EosError::InvalidState {
                context: "EosDtXzTable::from_ascii_reader",
                detail: format!("expected 11 header values, got {}", hdr.len()),
            });
        }

        let version = hdr[0] as i32;
        let x = hdr[1];
        let z = hdr[2];
        let num_log_ts = hdr[3] as usize;
        let log_t_min = hdr[4];
        let log_t_max = hdr[5];
        let del_log_t = hdr[6];
        let num_log_qs = hdr[7] as usize;
        let log_q_min = hdr[8];
        let log_q_max = hdr[9];
        let del_log_q = hdr[10];

        let ngrid = validate_4d_grid_header(
            "EosDtXzTable::from_ascii_reader",
            "ascii",
            x,
            z,
            num_log_qs,
            log_q_min,
            log_q_max,
            del_log_q,
            num_log_ts,
            log_t_min,
            log_t_max,
            del_log_t,
        )?;
        let mut raw_vars = vec![0.0_f64; NUM_EOS_TABLE_VALS * ngrid];

        for iq in 0..num_log_qs {
            for label in ["logQ label", "logQ value", "column names"] {
                let _ =
                    lines.next().ok_or_else(|| EosError::InvalidState {
                        context: "EosDtXzTable::from_ascii_reader",
                        detail: format!("missing {label} at iq={iq}"),
                    })??;
            }

            for jt in 0..num_log_ts {
                let row_line =
                    lines.next().ok_or_else(|| EosError::InvalidState {
                        context: "EosDtXzTable::from_ascii_reader",
                        detail: format!("missing data row at iq={iq}, jt={jt}"),
                    })??;
                let mut toks = row_line.split_whitespace();
                let _log_t_tok =
                    toks.next().ok_or_else(|| EosError::InvalidState {
                        context: "EosDtXzTable::from_ascii_reader",
                        detail: "empty data row".to_string(),
                    })?;
                let mut file_vals = [0.0_f64; NUM_EOS_TABLE_VALS];
                for (k, fv) in file_vals.iter_mut().enumerate() {
                    let s =
                        toks.next().ok_or_else(|| EosError::InvalidState {
                            context: "EosDtXzTable::from_ascii_reader",
                            detail: "fewer than 16 EOS columns in row"
                                .to_string(),
                        })?;
                    let val = s.parse::<f64>().map_err(|e| {
                        EosError::InvalidState {
                            context: "EosDtXzTable::from_ascii_reader",
                            detail: format!("bad float '{s}': {e}"),
                        }
                    })?;
                    if is_bad(val) {
                        return Err(EosError::InvalidState {
                            context: "EosDtXzTable::from_ascii_reader",
                            detail: format!(
                                "non-finite ASCII column {k} ({val}) in eosDT(X={x}, Z={z}) cell=({iq}, {jt})"
                            ),
                        });
                    }
                    *fv = val;
                }

                let cell = iq * num_log_ts + jt;
                Self::store_mapped_cell(&mut raw_vars, ngrid, cell, &file_vals);
            }
        }

        let label = format!("eosDT(X={x}, Z={z})");
        Self::build_from_raw_vars(
            &label, version, x, z, num_log_ts, log_t_min, log_t_max, del_log_t,
            num_log_qs, log_q_min, log_q_max, del_log_q, &raw_vars,
        )
    }

    /// Loads an `(X, Z)` `.data` file from disk and builds its bicubic spline table.
    pub fn from_data_file(path: &Path) -> Result<Self, EosError> {
        let file =
            std::fs::File::open(path).map_err(|e| EosError::TableIo {
                context: "EosDtXzTable::from_data_file",
                detail: format!("{}: {e}", path.display()),
            })?;
        let reader = BufReader::new(file);
        Self::from_ascii_reader(reader)
    }
}
