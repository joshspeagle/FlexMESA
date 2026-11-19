// SPDX-License-Identifier: LGPL-3.0-or-later
//! Shared loader helpers for `benchmarks/fortran_mesa/eos_oracles/mesa_26_4_1_eos_oracles.tsv`
//! (`#161` Slice 1, `T2-15`).

#![allow(dead_code)]

use flexmesa_eos::{
    EosEvalOutput, EOS_BASIC_RESULT_NAMES, NUM_EOS_BASIC_RESULTS,
};

pub const EOS_ORACLES_TSV: &str = include_str!(
    "../../../../benchmarks/fortran_mesa/eos_oracles/mesa_26_4_1_eos_oracles.tsv"
);

/// Loads a `R x C` indexed numeric table (`HELM` `16x5` or `EOS` `26x3`) by `case_id`.
pub fn load_indexed_rows<const R: usize, const C: usize>(
    prefix: &str,
    case_id: &str,
) -> [[f64; C]; R] {
    let mut out = [[0.0_f64; C]; R];
    let mut seen = [false; R];
    for line in EOS_ORACLES_TSV.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut cols = line.split('\t');
        if cols.next() != Some(prefix) || cols.next() != Some(case_id) {
            continue;
        }
        let idx: usize = cols.next().unwrap().trim().parse().unwrap();
        for item in &mut out[idx] {
            *item = cols.next().unwrap().trim().parse().unwrap();
        }
        seen[idx] = true;
    }
    assert!(
        seen.iter().all(|&b| b),
        "missing {prefix} oracle rows for {case_id}"
    );
    out
}

/// Loads a fixed-width scalar row (`DFERMI`, `ALFA`, `INV`) by `prefix` and `case_id`.
pub fn load_scalar_row<const C: usize>(prefix: &str, case_id: &str) -> [f64; C] {
    for line in EOS_ORACLES_TSV.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut cols = line.split('\t');
        if cols.next() != Some(prefix) || cols.next() != Some(case_id) {
            continue;
        }
        let mut out = [0.0_f64; C];
        for item in &mut out {
            *item = cols.next().unwrap().trim().parse().unwrap();
        }
        return out;
    }
    panic!("missing {prefix} scalar oracle row for {case_id}");
}

/// Loads an integer error-code check row (`IERR`) by `case_id`.
pub fn load_ierr_row(case_id: &str) -> (i32, i32) {
    for line in EOS_ORACLES_TSV.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut cols = line.split('\t');
        if cols.next() != Some("IERR") || cols.next() != Some(case_id) {
            continue;
        }
        let ierr_code: i32 = cols.next().unwrap().trim().parse().unwrap();
        let off_table: i32 = cols.next().unwrap().trim().parse().unwrap();
        return (ierr_code, off_table);
    }
    panic!("missing IERR oracle row for {case_id}");
}

/// Asserts an `EosEvalOutput` against `EOS\t<case_id>` in `mesa_26_4_1_eos_oracles.tsv`,
/// applying `val_tol` (`col 0`) and `deriv_tol` (`cols 1..=2`), with optional
/// per-entry residual bounds `(&str, usize, usize, f64)` when `allow_declared` is true.
pub fn assert_eos_against_mesa(
    case_id: &str,
    out: &EosEvalOutput,
    val_tol: f64,
    deriv_tol: f64,
    declared_residuals: &[(&str, usize, usize, f64)],
    allow_declared: bool,
) {
    let expected =
        load_indexed_rows::<NUM_EOS_BASIC_RESULTS, 3>("EOS", case_id);
    for (idx, exp) in expected.iter().enumerate() {
        let name = EOS_BASIC_RESULT_NAMES[idx];
        let got = [out.res[idx], out.d_dlnd[idx], out.d_dln_t[idx]];
        for (col, (&g, &e)) in got.iter().zip(exp.iter()).enumerate() {
            let base_tol = if col == 0 { val_tol } else { deriv_tol };
            let tol = if allow_declared {
                declared_residuals
                    .iter()
                    .find(|&&(c, i, k, _)| c == case_id && i == idx && k == col)
                    .map_or(base_tol, |&(_, _, _, bound)| bound)
            } else {
                base_tol
            };
            let rel_err = (g - e).abs() / e.abs().max(1.0);
            assert!(
                rel_err < tol,
                "{case_id} [{idx}: {name}] col {col}: got {g:.16e}, \
                 mesa_26_4_1 {e:.16e} (rel_err = {rel_err:.3e} >= {tol:.1e})"
            );
        }
    }
}
