// SPDX-License-Identifier: GPL-3.0-or-later
//! Donor structure of the original binary RLOF fixture compared bit for bit
//! with an instrumented MESA `26.4.1` run
//! (`benchmarks/fortran_mesa/binary_rlof_fixture/instrumented/`).
//!
//! The tables record `lnd lnT lnR L` per zone as IEEE-754 hex, right after
//! ZAMS interpolation (`get_zams_model`, `star/private/init_model.f90:65-135`)
//! and at the first solver call of model 1, after `prepare_for_new_step` has
//! reset `rho = dm/dV` (`star/private/evolve.f90:1854-1857`).

#![forbid(unsafe_code)]

mod common;

use std::path::PathBuf;

use flexmesa_star::StarModel;
use flexmesa_star::star_utils::set_rho_to_dm_div_dv_for_new_step;

/// Reads a reference table: rows of `k` followed by hex bit patterns.
fn read_hex_table(name: &str) -> Vec<Vec<u64>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/fortran_mesa/binary_rlof_fixture/instrumented")
        .join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("missing MESA reference {}: {e}", path.display())
    });
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let header = lines.next().expect("column header");
    assert!(
        header.starts_with("k\tlnd\tlnT\tlnR\tL"),
        "header: {header}"
    );
    lines
        .map(|l| {
            l.split('\t')
                .skip(1)
                .map(|h| u64::from_str_radix(h, 16).expect("hex bit pattern"))
                .collect()
        })
        .collect()
}

/// Asserts `lnd lnT lnR L` of every zone equal the table's bit patterns.
fn assert_structure_bits(model: &StarModel, table: &[Vec<u64>], what: &str) {
    assert_eq!(model.nz, table.len(), "{what}: zone count");
    let cols: [(&str, &[f64]); 4] = [
        ("lnd", &model.lnd),
        ("lnT", &model.ln_t),
        ("lnR", &model.ln_r),
        ("L", &model.lum),
    ];
    for (j, (name, values)) in cols.iter().enumerate() {
        let n_diff = (0..model.nz)
            .filter(|&k| values[k].to_bits() != table[k][j])
            .count();
        let first = (0..model.nz).find(|&k| values[k].to_bits() != table[k][j]);
        assert_eq!(
            n_diff,
            0,
            "{what}: {name} differs from MESA 26.4.1 in {n_diff} zones, first at k = {:?}",
            first.map(|k| k + 1)
        );
    }
}

#[test]
fn zams_model0_structure_matches_mesa_26_4_1_bit_for_bit() {
    let sys = common::build_rlof_binary_original();
    let table = read_hex_table("zams_model0_26_4_1.tsv");
    assert_structure_bits(
        &sys.donor,
        &table,
        "model 0 after ZAMS interpolation",
    );
}

#[test]
fn rho_reset_gives_mesa_26_4_1_first_solver_call_structure_bit_for_bit() {
    let mut sys = common::build_rlof_binary_original();
    set_rho_to_dm_div_dv_for_new_step(&mut sys.donor).expect("rho > 0");
    let table = read_hex_table("first_call_state_26_4_1.tsv");
    assert_structure_bits(&sys.donor, &table, "first solver call of model 1");
}
