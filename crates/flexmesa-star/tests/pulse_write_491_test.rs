// SPDX-License-Identifier: GPL-3.0-or-later
//! Pulse-data writers against gfortran output (#491).
//!
//! `tests/data/pulse_fortran_format/fmt_oracle.f90` runs the write
//! statements of MESA 26.4.1's `write_gyre_data`, `write_fgong_data` and
//! `write_osc_data` on synthetic data with the MESA SDK gfortran 15.2.0;
//! `fmt_oracle_expected.txt` is its output.

// The literals are copied from the Fortran oracle as written there.
#![allow(clippy::excessive_precision, clippy::unreadable_literal)]

use flexmesa_star::pulse_write::{
    fortran_1pe, write_fgong_data, write_gyre_data, write_osc_data,
    DEFAULT_FORMAT_FOR_OSC_DATA,
};

const EXPECTED: &str =
    include_str!("data/pulse_fortran_format/fmt_oracle_expected.txt");

/// The oracle's values, global data and point data.
fn oracle_data() -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let v: Vec<f64> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.988409870698050380e33,
        6.674300000000000413e-8,
        -5.522647688173986325e1,
        1e99,
        1e100,
        -1e100,
        1.5e-100,
        1e-99,
        9.99999999999999999e99,
        0.5,
        2.5,
        1e-300,
        4.9406564584124654e-324,
        2.2250738585072014e-308,
        1.7976931348623157e308,
        flexmesa_math::log(f64::MIN_POSITIVE),
        123456789.123456789,
        0.1,
        9.9999999999999999e-1,
        3.14159265358979e250,
    ];
    let n = v.len();
    let g: Vec<f64> = (1..=15).map(|i| v[(i - 1) % n]).collect();
    let p: Vec<Vec<f64>> = (1..=2)
        .map(|j| (1..=40).map(|i| v[(i + 7 * j - 1) % n]).collect())
        .collect();
    (v, g, p)
}

fn section(name: &str) -> String {
    let start = EXPECTED
        .find(&format!("=== {name}\n"))
        .expect("section present")
        + name.len()
        + 5;
    let rest = &EXPECTED[start..];
    let end = rest.find("=== ").unwrap_or(rest.len());
    rest[..end].to_owned()
}

#[test]
fn gyre_writer_matches_gfortran_bytes() {
    let (_, g, p) = oracle_data();
    let rows: Vec<Vec<f64>> = p.iter().map(|r| r[..18].to_vec()).collect();
    let got = write_gyre_data(&g[..3], &rows, 101).expect("schema 101");
    assert_eq!(got, section("gyre"));
}

#[test]
fn fgong_writers_match_gfortran_bytes_for_both_versions() {
    let (_, g, p) = oracle_data();
    let hdr = [
        String::from("FGONG file"),
        String::from("Created by MESAstar"),
        String::new(),
        String::new(),
    ];
    for (ivers, name) in [(1300, "fgong1300"), (300, "fgong300")] {
        let got = write_fgong_data(&g, &p, &hdr, ivers).expect("ivers");
        assert_eq!(got, section(name), "fgong_ivers {ivers}");
    }
    assert!(write_fgong_data(&g, &p, &hdr, 200).is_err());
}

#[test]
fn osc_writer_matches_gfortran_bytes() {
    let (_, g, p) = oracle_data();
    let rows: Vec<Vec<f64>> = p.iter().map(|r| r[..36].to_vec()).collect();
    let got = write_osc_data(&g, &rows, "r26.4.1", DEFAULT_FORMAT_FOR_OSC_DATA)
        .expect("default format");
    assert_eq!(got, section("osc"));
    assert!(write_osc_data(&g, &rows, "r26.4.1", "(1P4E20.12)").is_err());
}

#[test]
fn edit_descriptors_match_gfortran_for_edge_values() {
    let (v, _, _) = oracle_data();
    let mut got = String::new();
    for x in v {
        got.push_str("||\n");
        for (w, d, e) in [
            (26, 16, None),
            (26, 18, Some(3)),
            (16, 9, None),
            (19, 12, None),
        ] {
            got.push_str(&fortran_1pe(x, w, d, e));
            got.push('\n');
        }
    }
    assert_eq!(got, section("values"));
}
