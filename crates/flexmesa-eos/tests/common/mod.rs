// SPDX-License-Identifier: LGPL-3.0-or-later
//! Shared loader helpers for `benchmarks/fortran_mesa/eos_oracles/mesa_26_4_1_eos_oracles.tsv`
//! (`#161` Slice 1, `T2-15`).

#![allow(dead_code)]

use flexmesa_eos::{
    eval_helm_eos, EosEvalOutput, EOS_BASIC_RESULT_NAMES, NUM_EOS_BASIC_RESULTS,
};

pub const EOS_ORACLES_TSV: &str = include_str!(
    "../../../../benchmarks/fortran_mesa/eos_oracles/mesa_26_4_1_eos_oracles.tsv"
);

/// Tight absolute tolerance (`1.0e-14`) applied only when the upstream MESA
/// `26.4.1` reference entry is `0.0` or a cancelled near-zero Hermite stencil
/// slot (`|e| <= 1.0e-13`, e.g. fully ionized `d(lnfree_e)/dlnRho`).
pub const ZERO_ATOL: f64 = 1.0e-14;
const NEAR_ZERO_CUTOFF: f64 = 1.0e-13;

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

fn assert_entry_unfloored(
    case_id: &str,
    idx: usize,
    name: &str,
    col: usize,
    got: f64,
    exp: f64,
    rtol: f64,
) {
    let abs_err = (got - exp).abs();
    if exp.abs() <= NEAR_ZERO_CUTOFF {
        assert!(
            abs_err <= ZERO_ATOL,
            "{case_id} [{idx}: {name}] col {col} (near-zero): got {got:.16e}, \
             mesa_26_4_1 {exp:.16e} (abs_err = {abs_err:.3e} > {ZERO_ATOL:.1e})"
        );
    } else {
        let rel_err = abs_err / exp.abs();
        assert!(
            rel_err < rtol,
            "{case_id} [{idx}: {name}] col {col}: got {got:.16e}, \
             mesa_26_4_1 {exp:.16e} (rel_err = {rel_err:.3e} >= {rtol:.1e})"
        );
    }
}

/// Asserts a `16 x 5` `HELM` evaluation against `HELM\t<case_id>` using
/// unfloored relative error `|g - e| / |e| < rtol` (`ZERO_ATOL` for near-zero).
pub fn assert_helm_against_mesa(
    case_id: &str,
    temp: f64,
    den: f64,
    abar: f64,
    zbar: f64,
    rtol: f64,
) {
    let expected = load_indexed_rows::<16, 5>("HELM", case_id);
    let (out, deriv_abar, deriv_charge) =
        eval_helm_eos(temp, den, abar, zbar).expect(case_id);
    for (idx, exp) in expected.iter().enumerate() {
        let name = EOS_BASIC_RESULT_NAMES[idx];
        let got = [
            out.res[idx],
            out.d_dlnd[idx],
            out.d_dln_t[idx],
            deriv_abar[idx],
            deriv_charge[idx],
        ];
        for (col, (&g, &e)) in got.iter().zip(exp.iter()).enumerate() {
            assert_entry_unfloored(case_id, idx, name, col, g, e, rtol);
        }
    }
}

/// Asserts an `EosEvalOutput` against `EOS\t<case_id>` in `mesa_26_4_1_eos_oracles.tsv`,
/// applying unfloored `val_tol` (`col 0`) and `deriv_tol` (`cols 1..=2`), with
/// optional per-entry residual bounds `(&str, usize, usize, f64)` when
/// `allow_declared` is true.
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
            assert_entry_unfloored(case_id, idx, name, col, g, e, tol);
        }
    }
}
