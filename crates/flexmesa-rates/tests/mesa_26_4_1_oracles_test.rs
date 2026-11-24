// SPDX-License-Identifier: GPL-3.0-or-later
//! Bit-level MESA `26.4.1` Fortran oracle verification for `flexmesa-rates` (`#161`).
//!
//! Parses `benchmarks/fortran_mesa/rates_oracles/mesa_26_4_1_rates_oracles.tsv`
//! (captured directly against compiled MESA `r26.4.1` `librates.a` + `CRMATH`)
//! and verifies all `185` oracle outputs in `NumericsMode::Strict`:
//! - `0 ULP` across all `screen` (`33`), `reaclib` (`80`, including `T9 = 11.5 > 10`
//!   partition extrapolation and `d/dlnT`), `raw_rate` (`18`), `pycno` (`2`),
//!   and `weak` (`36`, including `Suzuki2016` interior and out-of-bounds fallback)
//!   outputs (`169 / 169` at `0 ULP`).
//! - `0 ULP` across all `coulomb` (`16`) outputs across `None`, `DGC1973`,
//!   `I1993`, `PCR2009`, `ThomasFermi`, and `Itoh2002`.

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines
)]

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::KEV;
use flexmesa_math::{with_numerics_mode, NumericsMode};
use flexmesa_rates::{
    compute_rev_ratio_coeff1, eval_fl_epsnuc_3alf, eval_mui_coulomb,
    eval_reaclib_handle, eval_vs_coulomb, eval_weak_reaction_pair, screen_pair,
    set_raw_rate, CoulombMuiOption, CoulombVsOption, RatesCatalog,
    ScreeningMode, TFactors,
};

fn oracles_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/fortran_mesa/rates_oracles")
}

fn decode_hex_f64(hex_tok: &str) -> f64 {
    let trimmed = hex_tok
        .trim()
        .strip_prefix("0x")
        .or_else(|| hex_tok.trim().strip_prefix("0X"))
        .unwrap_or_else(|| panic!("expected 0x hex token, got {hex_tok:?}"));
    let bits = u64::from_str_radix(trimmed, 16)
        .unwrap_or_else(|e| panic!("invalid hex {hex_tok:?}: {e}"));
    f64::from_bits(bits)
}

fn ulp_distance(a: f64, b: f64) -> u64 {
    if a == 0.0 && b == 0.0 {
        return 0;
    }
    let ua = a.to_bits();
    let ub = b.to_bits();
    if (ua >> 63) != (ub >> 63) {
        return u64::MAX;
    }
    ua.abs_diff(ub)
}

#[derive(Debug, Clone)]
struct OracleRow {
    row_id: usize,
    section: String,
    label: String,
    inputs: [f64; 5],
    out_name: String,
    expected: f64,
}

#[test]
fn test_rates_oracles_bundle_records_mesa_26_4_1_provenance_and_hygiene() {
    let dir = oracles_dir();
    let forbidden = [
        concat!("/", "home/"),
        concat!("/", "tmp/"),
        concat!("/", "mnt/"),
        concat!(".", "venvs"),
        concat!("file", "://"),
        "\u{2014}",
    ];
    for rel in [
        "Makefile",
        "README.md",
        "gen_rates_oracles.f90",
        "mesa_26_4_1_rates_oracles.tsv",
    ] {
        let path = dir.join(rel);
        let text = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("missing {}: {e}", path.display()));
        for tok in forbidden {
            assert!(
                !text.contains(tok),
                "forbidden token {tok:?} in {}",
                path.display()
            );
        }
    }

    let tsv = fs::read_to_string(dir.join("mesa_26_4_1_rates_oracles.tsv"))
        .expect("read rates oracle TSV");
    for header in [
        "# FlexMESA MESA 26.4.1 nuclear reaction rates (rates) oracle vectors (#161)",
        "# mesa_version: r26.4.1",
        "# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd",
        "# mesasdk_version: x86_64-linux-26.6.1",
        "# math_backend: CRMATH",
    ] {
        assert!(tsv.contains(header), "missing provenance header {header:?}");
    }
}

#[test]
fn test_all_185_mesa_26_4_1_rates_oracles_match_in_strict_mode() {
    let path = oracles_dir().join("mesa_26_4_1_rates_oracles.tsv");
    let text = fs::read_to_string(&path).expect("read rates oracle TSV");
    let mut rows = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') || t.starts_with("row_id\t") {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 11, "malformed line {}: {line:?}", idx + 1);
        rows.push(OracleRow {
            row_id: cols[0].trim().parse().expect("row_id"),
            section: cols[1].trim().to_string(),
            label: cols[2].trim().to_string(),
            inputs: [
                decode_hex_f64(cols[3]),
                decode_hex_f64(cols[4]),
                decode_hex_f64(cols[5]),
                decode_hex_f64(cols[6]),
                decode_hex_f64(cols[7]),
            ],
            out_name: cols[8].trim().to_string(),
            expected: decode_hex_f64(cols[9]),
        });
    }
    assert_eq!(rows.len(), 185, "expected 185 rates oracle rows");

    with_numerics_mode(NumericsMode::Strict, || {
        let chem = NuclideCatalog::builtin_ref();
        let cat = RatesCatalog::builtin();
        let mut coulomb_groups: BTreeMap<String, Vec<OracleRow>> =
            BTreeMap::new();

        for row in &rows {
            if row.section == "coulomb" {
                coulomb_groups
                    .entry(row.label.clone())
                    .or_default()
                    .push(row.clone());
                continue;
            }

            let got = match row.section.as_str() {
                "screen" => {
                    let parts: Vec<&str> = row.label.split(':').collect();
                    let mode = ScreeningMode::from_name(parts[0])
                        .expect("screening mode");
                    let pair_str = *parts.last().expect("pair");
                    let (n1, n2) = match pair_str {
                        "h1_h1" => ("h1", "h1"),
                        "he4_c12" => ("he4", "c12"),
                        "h1_n14" => ("h1", "n14"),
                        "c12_c12" => ("c12", "c12"),
                        "o16_o16" => ("o16", "o16"),
                        "neut_fe" => ("neut", "si28"),
                        "mg24_mg24" => ("mg24", "mg24"),
                        "c12_o16" => ("c12", "o16"),
                        other => panic!("unexpected screen pair {other}"),
                    };
                    let iso1 = chem.get_by_name(n1).expect("iso1");
                    let iso2 = chem.get_by_name(n2).expect("iso2");
                    let s = screen_pair(
                        mode,
                        row.inputs[0],
                        row.inputs[1],
                        row.inputs[2],
                        row.inputs[3],
                        row.inputs[4],
                        iso1.w,
                        iso1.z as f64,
                        iso2.w,
                        iso2.z as f64,
                    );
                    match row.out_name.as_str() {
                        "scor" => s.scor,
                        "dscor_dt" => s.dscor_dt,
                        "dscor_dd" => s.dscor_dd,
                        other => panic!("unexpected screen output {other}"),
                    }
                }
                "reaclib" => {
                    let (lam, dlam, rlam, drlam) =
                        eval_reaclib_handle(&row.label, row.inputs[0], false)
                            .expect("eval_reaclib_handle");
                    match row.out_name.as_str() {
                        "lambda" => lam,
                        "dlambda_dlnT" => dlam,
                        "rlambda" => rlam,
                        "drlambda_dlnT" => drlam,
                        other => panic!("unexpected reaclib output {other}"),
                    }
                }
                "raw_rate" => {
                    let ir = cat
                        .reaction_id_by_name(&row.label)
                        .expect("reaction_id_by_name");
                    let tf = TFactors::from_log_t(row.inputs[0], row.inputs[1]);
                    set_raw_rate(ir, row.inputs[1], &tf).expect("set_raw_rate")
                }
                "pycno" => eval_fl_epsnuc_3alf(
                    row.inputs[0],
                    row.inputs[1],
                    row.inputs[2],
                    row.inputs[3],
                ),
                "weak" => {
                    let pair_str = row.label.split(':').nth(1).expect("pair");
                    let (lhs, rhs) = pair_str.split_once('_').expect("lhs_rhs");
                    let use_suzuki = row.inputs[2] != 0.0;
                    let w = eval_weak_reaction_pair(
                        lhs,
                        rhs,
                        row.inputs[0],
                        row.inputs[1],
                        use_suzuki,
                    )
                    .expect("eval_weak_reaction_pair");
                    match row.out_name.as_str() {
                        "lambda" => w.lambda,
                        "dlambda_dlnT" => w.dlambda_dln_t,
                        "dlambda_dlnRho" => w.dlambda_dln_rho,
                        "Q" => w.q,
                        "dQ_dlnT" => w.dq_dln_t,
                        "dQ_dlnRho" => w.dq_dln_rho,
                        "Qneu" => w.qneu,
                        "dQneu_dlnT" => w.dqneu_dln_t,
                        "dQneu_dlnRho" => w.dqneu_dln_rho,
                        other => panic!("unexpected weak output {other}"),
                    }
                }
                other => panic!("unknown section {other}"),
            };

            let ulp = ulp_distance(got, row.expected);
            assert_eq!(
                ulp,
                0,
                "row {} ({}:{} {}) nonzero ULP={ulp}: got={got:.17e} expected={:.17e}",
                row.row_id,
                row.section,
                row.label,
                row.out_name,
                row.expected
            );
        }

        assert_eq!(coulomb_groups.len(), 4);
        for (lbl, crows) in &coulomb_groups {
            let (mui_name, vs_name) = lbl.split_once(':').expect("mui:vs");
            let mui_opt =
                CoulombMuiOption::from_name(mui_name).expect("mui option");
            let vs_opt =
                CoulombVsOption::from_name(vs_name).expect("vs option");
            let r0 = &crows[0];
            let (t, rho, zb, ab, z1) = (
                r0.inputs[0],
                r0.inputs[1],
                r0.inputs[2],
                r0.inputs[3],
                r0.inputs[4],
            );
            let z2 = crows[1].inputs[4];
            let mu1 = eval_mui_coulomb(mui_opt, t, rho, zb, ab, z1);
            let mu2 = eval_mui_coulomb(mui_opt, t, rho, zb, ab, z2);
            let dmu_ev = (mu1 - mu2).abs() * KEV * t;
            let vs = eval_vs_coulomb(vs_opt, t, rho, zb, ab, z1);
            for crow in crows {
                let got = match crow.out_name.as_str() {
                    "mu1" => mu1,
                    "mu2" => mu2,
                    "dmu_ev" => dmu_ev,
                    "vs" => vs,
                    other => panic!("unexpected coulomb output {other}"),
                };
                let ulp = ulp_distance(got, crow.expected);
                assert!(
                    ulp == 0,
                    "coulomb row {} ({lbl} {}) ULP={ulp} != 0: got={got:.17e} expected={:.17e}",
                    crow.row_id,
                    crow.out_name,
                    crow.expected
                );
            }
        }
    });
}

#[test]
fn test_coulomb_option_aliases_and_detailed_balance_coeff1_match_upstream() {
    assert_eq!(
        CoulombMuiOption::from_name("dgc1973"),
        Ok(CoulombMuiOption::Dcw1973)
    );
    assert_eq!(
        CoulombMuiOption::from_name("i1993"),
        Ok(CoulombMuiOption::Slr1982)
    );
    assert_eq!(
        CoulombMuiOption::from_name("pcr2009"),
        Ok(CoulombMuiOption::Pcr2009)
    );
    assert_eq!(
        CoulombVsOption::from_name("thomasfermi"),
        Ok(CoulombVsOption::Throwe1983)
    );
    assert_eq!(
        CoulombVsOption::from_name("itoh2002"),
        Ok(CoulombVsOption::Itoh2002)
    );
    assert!(CoulombMuiOption::from_name("unknown_mui").is_err());
    assert!(CoulombVsOption::from_name("unknown_vs").is_err());

    with_numerics_mode(NumericsMode::Strict, || {
        let chem = NuclideCatalog::builtin_ref();
        let cat = RatesCatalog::builtin();
        let c12 = chem.get_by_name("c12").expect("c12");
        let he4 = chem.get_by_name("he4").expect("he4");
        let o16 = chem.get_by_name("o16").expect("o16");
        let c1 = compute_rev_ratio_coeff1(
            &[c12.w, he4.w],
            &[c12.spin, he4.spin],
            &[o16.w],
            &[o16.spin],
        );
        let (_, terms) =
            cat.reaclib_terms("r_c12_ag_o16").expect("r_c12_ag_o16");
        assert_eq!(ulp_distance(c1, terms[0].inv_c1), 0);
    });
}
