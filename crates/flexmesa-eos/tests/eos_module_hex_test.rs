// SPDX-License-Identifier: LGPL-3.0-or-later
//! Level-2 IEEE-754 module-hex parity tests for `flexmesa-eos` against MESA
//! `26.4.1` `eos/test/test_output` (`exact_tested_values.tsv`, 391 outputs).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use std::path::PathBuf;

use flexmesa_autodiff::AutoDiffReal2VarOrder1;
use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_const::{CRAD, LN10};
use flexmesa_eos::{
    eos_dt_get, eos_dt_get_rho, eos_dt_get_t, eos_pt_get_with_linear,
    is_contained, min_distance_to_polygon, EosConfig, HelmTable,
    EOS_BASIC_RESULT_NAMES, I_CHI_RHO, I_CHI_T, I_ETA, I_GAMMA1, I_GAMMA3,
    I_GRAD_AD, I_LN_E, I_LN_FREE_E, I_LN_PGAS, I_LN_S, I_MU,
    NUM_EOS_BASIC_RESULTS, WHICH_OTHER_LOG_PTOT,
};
use flexmesa_math::{exp, exp10, log, log10, with_numerics_mode, NumericsMode};
use flexmesa_utils::{require_zero_skips_enabled, resolve_flexmesa_data_dir};

#[derive(Debug, Clone)]
struct HexRow {
    src_loc: String,
    label: String,
    raw_bits: u64,
    value: f64,
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
    let trimmed = raw
        .trim()
        .strip_prefix("0x")
        .or_else(|| raw.trim().strip_prefix("0X"))
        .unwrap_or_else(|| {
            panic!("expected 0x-prefixed hex bits, got {raw:?}")
        });
    u64::from_str_radix(trimmed, 16)
        .unwrap_or_else(|e| panic!("invalid hex bits {raw:?}: {e}"))
}

fn resolve_module_hex_tsv() -> Option<PathBuf> {
    if let Some(explicit) =
        std::env::var_os("FLEXMESA_MODULE_HEX_DIR").filter(|v| !v.is_empty())
    {
        let path = PathBuf::from(explicit)
            .join("eos")
            .join("exact_tested_values.tsv");
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
            "NOT RUN (FLEXMESA_REQUIRE_ZERO_SKIPS opted out): missing {}; \
             the calling test checks nothing",
            path.display()
        );
        return None;
    }

    let mut candidates = Vec::new();
    if let Some(parent) = resolve_flexmesa_data_dir().parent() {
        candidates.push(parent.join("mesa_module_hex"));
    }
    if let Some(home) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
        candidates.push(
            PathBuf::from(home)
                .join(".cache")
                .join("flexmesa")
                .join("mesa_module_hex"),
        );
    }
    candidates.push(PathBuf::from(".mesa_module_hex"));

    for root in &candidates {
        let path = root.join("eos").join("exact_tested_values.tsv");
        if path.exists() {
            return Some(path);
        }
    }

    let primary = candidates[0].join("eos").join("exact_tested_values.tsv");
    assert!(
        !require_zero_skips_enabled(),
        "missing required MESA 26.4.1 module-hex file `{}` (set \
         FLEXMESA_MODULE_HEX_DIR, run `flexmesa data fetch \
         --include-module-hex`, or set FLEXMESA_REQUIRE_ZERO_SKIPS=0 locally)",
        primary.display()
    );
    eprintln!(
        "NOT RUN (FLEXMESA_REQUIRE_ZERO_SKIPS opted out): missing {}; the \
         calling test checks nothing",
        primary.display()
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
        let raw_bits = parse_hex_bits(cols[7]);
        rows.push(HexRow {
            src_loc: cols[2].trim().to_string(),
            label: cols[3].trim().to_string(),
            raw_bits,
            value: f64::from_bits(raw_bits),
        });
    }
    rows
}

/// Builds the 7-species composition `(chem_id, xa)` matching `Init_Composition`
/// in `eos/test/src/eos_support.f90:34-91`.
fn init_composition(
    x_in: f64,
    zinit_in: f64,
    xc_in: f64,
    xo_in: f64,
) -> (Vec<usize>, Vec<f64>) {
    const ZFRAC_C: f64 = 0.173312;
    const ZFRAC_N: f64 = 0.053177;
    const ZFRAC_O: f64 = 0.482398;
    const ZFRAC_NE: f64 = 0.098675;

    let cat = NuclideCatalog::builtin_ref();
    let names = ["h1", "he4", "c12", "n14", "o16", "ne20", "mg24"];
    let chem_id: Vec<usize> = names
        .iter()
        .map(|n| cat.get_by_name(n).expect("valid isotope").chem_id)
        .collect();

    let mut xc = xc_in;
    let mut xo = xo_in;
    let mut y = 1.0 - (x_in + zinit_in + xc + xo);
    if y < 0.0 {
        let frac = (1.0 - x_in - zinit_in) / (xc + xo);
        xc *= frac;
        xo *= frac;
        y = (1.0 - (x_in + zinit_in + xc + xo)).max(0.0);
    }

    let mut xa = vec![0.0_f64; 7];
    xa[0] = x_in;
    xa[1] = y;
    xa[2] = zinit_in * ZFRAC_C + xc;
    xa[3] = zinit_in * ZFRAC_N;
    xa[4] = zinit_in * ZFRAC_O + xo;
    xa[5] = zinit_in * ZFRAC_NE;
    let sum6: f64 = xa[..6].iter().sum();
    xa[6] = 1.0 - sum6;

    (chem_id, xa)
}

#[test]
fn mesa_26_4_1_eos_module_hex_zero_ulp() {
    let Some(tsv_path) = resolve_module_hex_tsv() else {
        return;
    };
    let rows = load_module_hex_rows(&tsv_path);
    assert_eq!(rows.len(), 391, "expected 391 data rows in eos hex TSV");

    with_numerics_mode(NumericsMode::Strict, || {
        let ht = HelmTable::builtin().expect("builtin HELM table");
        let cfg = EosConfig::default();
        let cat = NuclideCatalog::builtin_ref();

        let mut checked = 0_usize;
        let mut zeros = 0_usize;
        let mut max_ulp = 0_u64;
        let mut failures: Vec<String> = Vec::new();

        let mut record =
            |row_idx: usize, ctx: &str, expected_label: &str, got: f64| {
                let want = &rows[row_idx];
                assert_eq!(
                    want.label, expected_label,
                    "TSV label mismatch at row[{row_idx:03}] ({ctx})"
                );
                let ulp = ulp_distance(got, want.raw_bits);
                checked += 1;
                if ulp == 0 {
                    zeros += 1;
                } else {
                    max_ulp = max_ulp.max(ulp);
                    failures.push(format!(
                        "  row[{row_idx:03}] [{ctx}] {} ({}): got={got:.17e} \
                         ({:#018x}) want={:.17e} ({:#018x}) ulp={ulp}",
                        want.label,
                        want.src_loc,
                        got.to_bits(),
                        want.value,
                        want.raw_bits
                    ));
                }
            };

        // 1. `Do_One` (`test_eos_support.f90:158-230, 583-658`, 320 rows).
        let compositions = [
            ("pure Helium", 0.0, 0.0),
            ("pure Hydrogen", 1.0, 0.0),
            ("mixed Z with H&He 3:1 ratio", 0.75 * (1.0 - 0.03), 0.03),
            ("solar", 0.70, 0.02),
        ];
        let trho_points = [
            (1.0e6_f64, 1.0e-2_f64),
            (1.0e4_f64, 1.0e-1_f64),
            (1.0e5_f64, 1.0e-1_f64),
            (2.0e8_f64, 1.0e2_f64),
        ];

        let mut row_cursor = 0_usize;
        for (cname, x_in, z_init) in compositions {
            let (chem_id, xa) = init_composition(x_in, z_init, 0.0, 0.0);
            let comp = get_composition_info(cat, &chem_id, &xa, false)
                .expect("composition_info");
            for (pt_idx, &(t, rho)) in trho_points.iter().enumerate() {
                let tag = format!("Do_One {cname} pt#{pt_idx}");
                let out = eos_dt_get(ht, rho, t, &chem_id, &xa, &cfg)
                    .expect("eos_dt_get");
                let prad = CRAD * t * t * t * t / 3.0;
                let pgas = exp(out.res[I_LN_PGAS]);
                let p = pgas + prad;
                let e = exp(out.res[I_LN_E]);
                let s = exp(out.res[I_LN_S]);

                // Note: `Do_One_TRho` sets module-global `Z = Zinit + dXC + dXO`
                // (`test_eos_support.f90:599`), and `Init_Composition` declares
                // a local `Z` (`eos_support.f90:43`), so line 609 prints `z_init`
                // while `X` and `Y` come from `composition_info`.
                let expected_vals = [
                    ("X", comp.xh),
                    ("Y", comp.xhe),
                    ("Z", z_init),
                    ("abar", comp.abar),
                    ("zbar", comp.zbar),
                    ("logRho", log10(rho)),
                    ("logT", log10(t)),
                    ("T6", t * 1.0e-6),
                    ("P", p),
                    ("E", e),
                    ("S", s),
                    ("mu", out.res[I_MU]),
                    ("lnfree_e", out.res[I_LN_FREE_E]),
                    ("eta", out.res[I_ETA]),
                    ("grad_ad", out.res[I_GRAD_AD]),
                    ("chiRho", out.res[I_CHI_RHO]),
                    ("chiT", out.res[I_CHI_T]),
                    ("gamma1", out.res[I_GAMMA1]),
                    ("gamma3", out.res[I_GAMMA3]),
                    ("eta", out.res[I_ETA]),
                ];
                for (label, val) in expected_vals {
                    record(row_cursor, &tag, label, val);
                    row_cursor += 1;
                }
            }
        }
        assert_eq!(row_cursor, 320);

        // 2. `test_get_Rho_T` (`test_eos_support.f90:191-204, 232-357`, 16 rows).
        {
            let (chem_id, xa) = init_composition(0.70, 0.02, 0.0, 0.0);
            let t_init = exp10(4.8);
            let rho_init = 1.0e-7;
            let log10_rho = log10(rho_init);
            let log10_t = log10(t_init);
            let ln_s = log(2.968_064_512_e9);
            let tol = 1.0e-5;
            let othertol = 1.0e-12;
            let max_iter = 100;

            record(row_cursor, "get_Rho_T", "tolerance", tol);
            row_cursor += 1;

            let log_rho_guess = log10(rho_init * 0.5);
            let inv_rho = eos_dt_get_rho(
                ht,
                &chem_id,
                &xa,
                log10_t,
                I_LN_S as isize,
                ln_s,
                log_rho_guess,
                tol,
                othertol,
                max_iter,
                &cfg,
            )
            .expect("eos_dt_get_rho");
            record(row_cursor, "get_Rho", "actual logRho", log10_rho);
            record(row_cursor + 1, "get_Rho", "guess logRho", log_rho_guess);
            record(
                row_cursor + 2,
                "get_Rho",
                "found logRho",
                inv_rho.log10_result,
            );
            record(row_cursor + 3, "get_Rho", "wanted logS", ln_s / LN10);
            record(
                row_cursor + 4,
                "get_Rho",
                "got logS",
                inv_rho.eos.res[I_LN_S] / LN10,
            );
            row_cursor += 5;

            let log_t_guess = log10(t_init * 2.0);
            let inv_t_s = eos_dt_get_t(
                ht,
                &chem_id,
                &xa,
                log10_rho,
                I_LN_S as isize,
                ln_s,
                log_t_guess,
                tol,
                othertol,
                max_iter,
                &cfg,
            )
            .expect("eos_dt_get_t lnS");
            record(row_cursor, "get_T_S", "actual logT", log10_t);
            record(row_cursor + 1, "get_T_S", "guess logT", log_t_guess);
            record(
                row_cursor + 2,
                "get_T_S",
                "found logT",
                inv_t_s.log10_result,
            );
            record(row_cursor + 3, "get_T_S", "wanted logS", ln_s / LN10);
            record(
                row_cursor + 4,
                "get_T_S",
                "got logS",
                inv_t_s.eos.res[I_LN_S] / LN10,
            );
            row_cursor += 5;

            let t_for_p = exp10(log10_t);
            let prad = CRAD * t_for_p * t_for_p * t_for_p * t_for_p / 3.0;
            let pgas = exp(inv_t_s.eos.res[I_LN_PGAS]);
            let log_p = log10(prad + pgas);
            let log_t_guess_p = log10(t_for_p * 2.0);
            let inv_t_p = eos_dt_get_t(
                ht,
                &chem_id,
                &xa,
                log10_rho,
                WHICH_OTHER_LOG_PTOT,
                log_p,
                log_t_guess_p,
                tol,
                othertol,
                max_iter,
                &cfg,
            )
            .expect("eos_dt_get_t logPtot");
            let t_res = exp10(inv_t_p.log10_result);
            let prad_res = CRAD * t_res * t_res * t_res * t_res / 3.0;
            let pgas_res = exp(inv_t_p.eos.res[I_LN_PGAS]);
            let got_log_p = log10(prad_res + pgas_res);
            record(row_cursor, "get_T_P", "actual logT", log10_t);
            record(row_cursor + 1, "get_T_P", "guess logT", log_t_guess_p);
            record(
                row_cursor + 2,
                "get_T_P",
                "found logT",
                inv_t_p.log10_result,
            );
            record(row_cursor + 3, "get_T_P", "wanted logP", log_p);
            record(row_cursor + 4, "get_T_P", "got logP", got_log_p);
            row_cursor += 5;
        }
        assert_eq!(row_cursor, 336);

        // 3. `test1_eosPT_for_ck` (`test_eos_support.f90:15-24, 36-122`, 40 rows).
        {
            let z_in = 0.02_f64;
            let x_in = 0.7_f64;
            let (chem_id, xa) = init_composition(x_in, z_in, 0.0, 0.0);
            let comp = get_composition_info(cat, &chem_id, &xa, false)
                .expect("composition_info");
            let log_t = 3.0_f64;
            let log_pgas = 4.2_f64;
            let t = exp10(log_t);
            let pgas = exp10(log_pgas);

            let pt = eos_pt_get_with_linear(
                ht,
                &chem_id,
                &xa,
                log_pgas,
                log_t,
                Some(pgas),
                Some(t),
                &cfg,
            )
            .expect("eos_pt_get_with_linear");

            let prad = CRAD * t * t * t * t / 3.0;
            let p = pgas + prad;

            // `test1_eosPT` prints its own dummy arguments `Z` and `X` at
            // `test_eos_support.f90:76-77` together with module `abar`/`zbar`.
            let header_vals = [
                ("logT", log_t),
                ("logPgas", log_pgas),
                ("T", t),
                ("Pgas", pgas),
                ("Z", z_in),
                ("X", x_in),
                ("abar", comp.abar),
                ("zbar", comp.zbar),
                ("rho", pt.rho),
                ("logRho", pt.log10_rho),
                ("logT", log_t),
                ("logPgas", log_pgas),
                ("dlnRho_dlnPgas_c_T", pt.dln_rho_dln_pgas_const_t),
                ("dlnRho_dlnT_c_Pgas", pt.dln_rho_dln_t_const_pgas),
                ("P", p),
                ("E", exp(pt.eos.res[I_LN_E])),
                ("S", exp(pt.eos.res[I_LN_S])),
            ];
            for (label, val) in header_vals {
                record(row_cursor, "test1_eosPT", label, val);
                row_cursor += 1;
            }
            for slot in 3..NUM_EOS_BASIC_RESULTS {
                record(
                    row_cursor,
                    "test1_eosPT res",
                    EOS_BASIC_RESULT_NAMES[slot],
                    pt.eos.res[slot],
                );
                row_cursor += 1;
            }
        }
        assert_eq!(row_cursor, 376);

        // 4. `do_test_eos_blend` (`test_eos_blend.f90:14-53`, 15 rows).
        {
            let coords = [[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0], [1.0, -1.0]];
            let pts = [
                ([0.0, 1.0], true),
                ([-1.0, 1.0], false),
                ([0.0, 0.0], true),
                ([-2.0, 0.0], false),
                ([-1.0, 3.0], false),
            ];
            for (idx, ([x, y], exp_contained)) in pts.into_iter().enumerate() {
                let p = [
                    AutoDiffReal2VarOrder1::constant(x),
                    AutoDiffReal2VarOrder1::constant(y),
                ];
                assert_eq!(is_contained(&coords, p), exp_contained);
                let d = min_distance_to_polygon(&coords, p);
                let tag = format!("blend pt#{idx}");
                record(row_cursor, &tag, "-", d.val);
                record(row_cursor + 1, &tag, "-", d.d1[0]);
                record(row_cursor + 2, &tag, "-", d.d1[1]);
                row_cursor += 3;
            }
        }
        assert_eq!(row_cursor, 391);
        assert_eq!(checked, 391);

        eprintln!(
            "mesa_26_4_1_eos_module_hex_zero_ulp: {zeros}/{checked} at 0 ULP \
             (max_ulp={max_ulp})"
        );
        for line in &failures {
            eprintln!("{line}");
        }
        assert_eq!(
            max_ulp, 0,
            "{zeros}/{checked} at 0 ULP (max_ulp={max_ulp})"
        );
    });
}
