// SPDX-License-Identifier: LGPL-3.0-or-later
//! Level-2 IEEE-754 module-hex parity tests for `flexmesa-chem` against MESA
//! `26.4.1` `chem/test/test_output` (`exact_tested_values.tsv`, 1,102 outputs).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_lossless,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use std::path::PathBuf;

use flexmesa_chem::{
    ag89_yesol, ag89_zsol, chem_xsol, get_composition_info,
    lodders03_element_atom_percent, solar_cno_fraction, NuclideCatalog,
    SolarScale,
};
use flexmesa_math::{with_numerics_mode, NumericsMode};
use flexmesa_utils::{require_zero_skips_enabled, resolve_flexmesa_data_dir};

#[derive(Debug, Clone)]
struct HexRow {
    record_idx: usize,
    src_loc: String,
    label: String,
    raw_bits: u64,
}

fn signed_magnitude(bits: u64) -> i128 {
    if (bits >> 63) != 0 {
        -((bits ^ (1_u64 << 63)) as i128)
    } else {
        bits as i128
    }
}

fn ulp_distance(actual: f64, expected_bits: u64) -> u64 {
    let expected = f64::from_bits(expected_bits);
    if actual.is_nan() || expected.is_nan() {
        return if actual.is_nan() && expected.is_nan() {
            0
        } else {
            u64::MAX
        };
    }
    let a = signed_magnitude(actual.to_bits());
    let b = signed_magnitude(expected_bits);
    u64::try_from((a - b).unsigned_abs()).unwrap_or(u64::MAX)
}

fn parse_hex_bits(raw: &str) -> u64 {
    let stripped = raw
        .trim()
        .strip_prefix("0x")
        .or_else(|| raw.trim().strip_prefix("0X"))
        .unwrap_or_else(|| {
            panic!("expected 0x-prefixed hex bits, got {raw:?}")
        });
    u64::from_str_radix(stripped, 16)
        .unwrap_or_else(|e| panic!("invalid hex bits {raw:?}: {e}"))
}

fn module_hex_chem_tsv_path() -> Option<PathBuf> {
    let root = std::env::var_os("FLEXMESA_MODULE_HEX_DIR")
        .filter(|v| !v.is_empty())
        .map_or_else(
            || {
                resolve_flexmesa_data_dir().parent().map_or_else(
                    || PathBuf::from(".mesa_module_hex"),
                    |p| p.join("mesa_module_hex"),
                )
            },
            PathBuf::from,
        );
    let path = root.join("chem").join("exact_tested_values.tsv");
    if path.exists() {
        return Some(path);
    }
    assert!(
        !require_zero_skips_enabled(),
        "missing required MESA 26.4.1 module-hex file `{}` (set \
         FLEXMESA_MODULE_HEX_DIR, run `flexmesa data fetch \
         --include-module-hex`, or set FLEXMESA_REQUIRE_ZERO_SKIPS=0 locally)",
        path.display()
    );
    eprintln!(
        "NOT RUN (FLEXMESA_REQUIRE_ZERO_SKIPS opted out): missing {}; the \
         calling test checks nothing",
        path.display()
    );
    None
}

fn load_module_hex_rows(path: &std::path::Path) -> Vec<HexRow> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let mut rows = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.starts_with("record_idx\t")
        {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 8, "expected 8 columns in TSV line: {line:?}");
        let record_idx = cols[0].trim().parse::<usize>().unwrap_or_else(|e| {
            panic!("invalid record_idx {:?}: {e}", cols[0])
        });
        rows.push(HexRow {
            record_idx,
            src_loc: cols[2].trim().to_owned(),
            label: cols[3].trim().to_owned(),
            raw_bits: parse_hex_bits(cols[7]),
        });
    }
    rows
}

#[test]
fn mesa_26_4_1_chem_module_hex_zero_ulp() {
    let Some(path) = module_hex_chem_tsv_path() else {
        return;
    };
    let rows = load_module_hex_rows(&path);
    assert_eq!(
        rows.len(),
        1102,
        "expected 1102 chem module-hex rows in {}",
        path.display()
    );

    with_numerics_mode(NumericsMode::Strict, || {
        let catalog = NuclideCatalog::builtin_ref();
        let oxygen_16_id = catalog
            .get_nuclide_index("o16")
            .expect("o16 missing from NuclideCatalog");
        let o16_entry = catalog
            .get_by_id(oxygen_16_id)
            .expect("o16 entry missing from NuclideCatalog");
        let carbon_12_id = catalog
            .get_nuclide_index("c12")
            .expect("c12 missing from NuclideCatalog");
        let iron_56_id = catalog
            .get_nuclide_index("fe56")
            .expect("fe56 missing from NuclideCatalog");
        let helium_3_id = catalog
            .get_nuclide_index("he3")
            .expect("he3 missing from NuclideCatalog");
        let helium_4_id = catalog
            .get_nuclide_index("he4")
            .expect("he4 missing from NuclideCatalog");
        let hydrogen_1_id = catalog
            .get_nuclide_index("h1")
            .expect("h1 missing from NuclideCatalog");

        let comp = get_composition_info(
            catalog,
            &[carbon_12_id, iron_56_id],
            &[0.5_f64, 0.5_f64],
            false,
        )
        .expect("get_composition_info(c12, fe56) failed");

        let qtotal = catalog
            .reaction_qtotal(
                2,
                3,
                &[
                    helium_3_id,
                    helium_3_id,
                    helium_4_id,
                    hydrogen_1_id,
                    hydrogen_1_id,
                ],
            )
            .expect("reaction_qtotal(2 he3 => he4 + 2 h1) failed");

        let ag89_cno_from_xsol =
            (chem_xsol("c12") + chem_xsol("n14") + chem_xsol("o16"))
                / ag89_zsol();

        let mut failures = Vec::new();
        let mut lodders_count = 0_usize;
        let mut do_tests_count = 0_usize;

        for (idx, row) in rows.iter().enumerate() {
            let actual = match row.src_loc.as_str() {
                "test_chem.f90:189" => {
                    lodders_count += 1;
                    lodders03_element_atom_percent(&row.label)
                }
                "test_chem.f90:67" => {
                    do_tests_count += 1;
                    assert_eq!(row.label, "chem_W(io16)");
                    o16_entry.w
                }
                "test_chem.f90:69" => {
                    do_tests_count += 1;
                    assert_eq!(row.label, "chem_Z(io16)");
                    f64::from(
                        u32::try_from(o16_entry.z).expect("o16 Z fits in u32"),
                    )
                }
                "test_chem.f90:71" => {
                    do_tests_count += 1;
                    assert_eq!(row.label, "chem_binding_energy(io16)");
                    o16_entry.binding_energy
                }
                "test_chem.f90:85" => {
                    do_tests_count += 1;
                    assert_eq!(row.label, "Anders & Grevesse 1989 zsol");
                    ag89_zsol()
                }
                "test_chem.f90:86" => {
                    do_tests_count += 1;
                    assert_eq!(row.label, "Anders & Grevesse 1989 yesol");
                    ag89_yesol()
                }
                "test_chem.f90:100" => {
                    do_tests_count += 1;
                    assert_eq!(row.label, "Anders & Grevesse 1989");
                    ag89_cno_from_xsol
                }
                "test_chem.f90:103" => {
                    do_tests_count += 1;
                    assert_eq!(row.label, "Grevesse and Noels 1993");
                    solar_cno_fraction(SolarScale::Gn93, catalog)
                }
                "test_chem.f90:106" => {
                    do_tests_count += 1;
                    assert_eq!(row.label, "Grevesse and Sauval 1998");
                    solar_cno_fraction(SolarScale::Gs98, catalog)
                }
                "test_chem.f90:109" => {
                    do_tests_count += 1;
                    assert_eq!(row.label, "Lodders 2003");
                    solar_cno_fraction(SolarScale::L03, catalog)
                }
                "test_chem.f90:112" => {
                    do_tests_count += 1;
                    assert_eq!(row.label, "Asplund, Grevesse and Sauval 2005");
                    solar_cno_fraction(SolarScale::Ags05, catalog)
                }
                "test_chem.f90:151" => {
                    do_tests_count += 1;
                    assert_eq!(row.label, "Abar");
                    comp.abar
                }
                "test_chem.f90:152" => {
                    do_tests_count += 1;
                    assert_eq!(row.label, "Zbar");
                    comp.zbar
                }
                "test_chem.f90:153" => {
                    do_tests_count += 1;
                    assert_eq!(row.label, "Z2bar");
                    comp.z2bar
                }
                "test_chem.f90:154" => {
                    do_tests_count += 1;
                    assert_eq!(row.label, "Ye");
                    comp.ye
                }
                "test_chem.f90:155" => {
                    do_tests_count += 1;
                    assert_eq!(row.label, "sum(X*W/A)");
                    comp.mass_correction
                }
                "test_chem.f90:176" => {
                    do_tests_count += 1;
                    assert_eq!(row.label, "Qtotal");
                    qtotal
                }
                other => {
                    panic!(
                        "unexpected src_loc {other:?} at row {idx} (record {})",
                        row.record_idx
                    );
                }
            };

            let ulp = ulp_distance(actual, row.raw_bits);
            if ulp != 0 {
                failures.push(format!(
                    "row {idx} (record {}, {} `{}`): got {:#018x} ({actual:.17e}), \
                     expected {:#018x} ({:.17e}), ulp={ulp}",
                    row.record_idx,
                    row.src_loc,
                    row.label,
                    actual.to_bits(),
                    row.raw_bits,
                    f64::from_bits(row.raw_bits),
                ));
            }
        }

        assert_eq!(lodders_count, 286);
        assert_eq!(do_tests_count, 816);
        eprintln!(
            "mesa_26_4_1_chem_module_hex_zero_ulp: {}/{} at 0 ULP (max_ulp=0)",
            rows.len() - failures.len(),
            rows.len(),
        );
        assert!(
            failures.is_empty(),
            "chem module-hex zero-ULP mismatches ({} / {}):\n{}",
            failures.len(),
            rows.len(),
            failures.join("\n")
        );
    });
}
