// SPDX-License-Identifier: GPL-3.0-or-later
//! Level-2 IEEE-754 module-hex parity tests for `flexmesa-net` against MESA
//! `26.4.1` `net/test/test_output` (`exact_tested_values.tsv`, 116 `net_get`
//! outputs across 7 networks + 8 one-zone burn outputs).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use std::collections::HashMap;
use std::path::PathBuf;

use flexmesa_chem::{
    category_name, get_composition_info, NuclideCatalog, NUM_CATEGORIES,
};
use flexmesa_eos::{EosConfig, HelmTable};
use flexmesa_math::{exp10, log10, with_numerics_mode, NumericsMode};
use flexmesa_net::{
    burn_1_zone, burn_1_zone_const_p, net_get, Burn1ZoneInput, BurnConstPInput,
    NetGeneralInfo,
};
use flexmesa_num::RosenbrockVariant;
use flexmesa_rates::ScreeningMode;
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
    let path = root.join("net").join("exact_tested_values.tsv");
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

fn build_test_composition(
    net_name: &str,
    g: &NetGeneralInfo,
    cat: &NuclideCatalog,
) -> Vec<f64> {
    let mut idx = HashMap::with_capacity(g.num_isos);
    for (slot, &cid) in g.chem_id.iter().enumerate() {
        let name = cat
            .get_by_id(cid)
            .unwrap_or_else(|| panic!("unknown chem_id {cid}"))
            .name
            .as_str();
        idx.insert(name, slot);
    }
    let mut xin = vec![0.0_f64; g.num_isos];
    match net_name {
        "basic.net" | "pp_extras.net" => {
            xin[idx["h1"]] = f64::from(0.655_186_f32);
            xin[idx["he4"]] = 0.310_021_64;
            xin[idx["c12"]] = 0.002_725e-1;
            xin[idx["n14"]] =
                0.203_101e-1 + 0.612_124e-6 + 0.109_305e-2 + 0.356_004e-4;
            xin[idx["o16"]] = 0.094_000e-1;
            xin[idx["ne20"]] = 0.162_163e-2;
            xin[idx["mg24"]] = 0.658_226e-25;
            xin[idx["he3"]] = 0.201_852e-2;
            if net_name == "pp_extras.net" {
                xin[idx["h2"]] = 0.632_956e-17;
                xin[idx["li7"]] = 0.664_160e-15;
                xin[idx["be7"]] = 0.103_866e-15;
            }
        }
        "o18_and_ne22.net" => {
            xin[idx["h1"]] = f64::from(0.655_186_f32);
            xin[idx["he4"]] = 0.310_021_64;
            xin[idx["c12"]] = 0.002_725e-1;
            xin[idx["n14"]] = 0.203_101e-1 + 0.612_124e-6 + 0.109_305e-2;
            xin[idx["o16"]] = 0.094_000e-1;
            xin[idx["o18"]] = 1.0e-20;
            xin[idx["ne20"]] = 0.162_163e-2;
            xin[idx["mg24"]] = 0.658_226e-25;
            xin[idx["ne22"]] = 0.201_852e-2;
        }
        "cno_extras.net" => {
            xin[idx["h1"]] = 0.173_891_680_788e-1;
            xin[idx["he4"]] = 0.963_245_225_401;
            xin[idx["c12"]] = 0.238_935_745_993e-3;
            xin[idx["n14"]] = 0.134_050_688_300e-1;
            xin[idx["o16"]] = 0.268_791_618_452e-3;
            xin[idx["ne20"]] = 0.180_001_692_845e-2;
            xin[idx["mg24"]] = 0.353_667_702_698e-2;
            xin[idx["c13"]] = 0.717_642_727_071e-4;
            xin[idx["n13"]] = 0.370_732_258_156e-9;
            xin[idx["n15"]] = 0.450_484_708_137e-6;
            xin[idx["o14"]] = 0.100_000_000_000e-49;
            xin[idx["o15"]] = 0.874_815_374_966e-10;
            xin[idx["f17"]] = 0.100_000_000_000e-49;
            xin[idx["f18"]] = 0.100_000_000_000e-49;
            xin[idx["ne18"]] = 0.100_000_000_000e-49;
            xin[idx["ne19"]] = 0.100_000_000_000e-49;
            xin[idx["mg22"]] = 0.439_011_547_696e-4;
        }
        "approx21.net" | "approx21_plus_co56.net" => {
            xin[idx["fe56"]] = 8.038_702_148_431_816_6e-1;
            xin[idx["fe54"]] = 1.609_664_873_676_083_2e-1;
            xin[idx["cr56"]] = 2.948_094_553_592_052_5e-2;
            xin[idx["he4"]] = 4.862_463_716_132_056_5e-3;
            xin[idx["ni56"]] = 2.837_627_073_189_036_0e-4;
            xin[idx["si28"]] = 1.501_862_890_613_573_9e-4;
            xin[idx["s32"]] = 1.161_327_163_557_345_7e-4;
            xin[idx["prot"]] = 1.113_943_163_367_365_3e-4;
            xin[idx["ca40"]] = 5.368_837_747_349_418_5e-5;
            xin[idx["ar36"]] = 5.270_283_182_256_706_2e-5;
            xin[idx["fe52"]] = 3.786_650_413_193_518_5e-5;
            xin[idx["cr48"]] = 5.840_112_303_766_797_4e-6;
            xin[idx["neut"]] = 4.914_122_770_311_839_7e-6;
            xin[idx["ti44"]] = 1.508_503_856_115_474_6e-6;
            xin[idx["o16"]] = 9.538_401_960_904_925_5e-7;
            xin[idx["mg24"]] = 6.380_820_771_772_558_0e-7;
            xin[idx["c12"]] = 2.904_865_667_399_186_8e-7;
            xin[idx["ne20"]] = 9.646_886_502_360_942_7e-9;
            xin[idx["he3"]] = 4.620_360_386_226_309_6e-80;
            xin[idx["n14"]] = 7.586_747_222_584_123_5e-99;
        }
        "approx21_cr60_plus_co56.net" => {
            xin[idx["he4"]] = 3.455_539_253_481_393_9e-1;
            xin[idx["o16"]] = 1.936_777_842_043_093_7e-1;
            xin[idx["h1"]] = 1.905_293_136_717_250_1e-1;
            xin[idx["si28"]] = 9.102_393_603_206_460_1e-2;
            xin[idx["ni56"]] = 5.849_245_965_329_500_5e-2;
            xin[idx["s32"]] = 4.141_664_247_699_990_8e-2;
            xin[idx["ne20"]] = 2.092_778_233_247_755_8e-2;
            xin[idx["c12"]] = 2.058_961_710_444_831_2e-2;
            xin[idx["mg24"]] = 1.897_591_410_840_610_4e-2;
            xin[idx["ar36"]] = 7.060_033_878_593_995_6e-3;
            xin[idx["ca40"]] = 4.918_217_198_135_372_6e-3;
            xin[idx["fe52"]] = 3.161_882_080_600_528_0e-3;
            xin[idx["n14"]] = 1.987_846_008_276_095_2e-3;
            xin[idx["fe56"]] = 7.166_877_662_113_019_0e-4;
            xin[idx["co56"]] = 4.497_497_109_992_118_1e-4;
            xin[idx["fe54"]] = 3.343_242_389_098_842_2e-4;
            xin[idx["cr48"]] = 1.760_262_690_776_976_2e-4;
            xin[idx["he3"]] = 5.165_009_719_163_154_5e-6;
            xin[idx["ti44"]] = 2.645_242_220_338_990_6e-6;
            xin[idx["cr60"]] = 4.765_335_309_593_798_2e-8;
            xin[idx["prot"]] = 1.273_902_240_724_625_0e-11;
            xin[idx["neut"]] = 0.0;
        }
        other => panic!("unsupported test network {other:?}"),
    }
    for v in &mut xin {
        if *v < 1.0e-99 {
            *v = 1.0e-99;
        }
    }
    let sum: f64 = xin.iter().copied().sum();
    xin[idx["mg24"]] += 1.0 - sum;
    xin
}

#[test]
fn mesa_26_4_1_net_module_hex_zero_ulp() {
    let Some(tsv_path) = resolve_module_hex_tsv() else {
        return;
    };
    let rows = load_module_hex_rows(&tsv_path);
    assert_eq!(rows.len(), 140, "expected 140 data rows in net hex TSV");

    let cat = NuclideCatalog::builtin_ref();
    let mut cat_map = HashMap::with_capacity(NUM_CATEGORIES);
    for c in 1..=NUM_CATEGORIES {
        let name = category_name(c).expect("valid category id");
        cat_map.insert(name, c - 1);
    }

    let blocks: [(&str, usize, usize); 7] = [
        ("basic.net", 0, 10),
        ("o18_and_ne22.net", 10, 20),
        ("pp_extras.net", 20, 30),
        ("cno_extras.net", 30, 40),
        ("approx21.net", 40, 84),
        ("approx21_plus_co56.net", 84, 126),
        ("approx21_cr60_plus_co56.net", 126, 130),
    ];

    with_numerics_mode(NumericsMode::Strict, || {
        let mut checked = 0_usize;
        let mut zeros = 0_usize;
        let mut max_ulp = 0_u64;
        let mut failures: Vec<String> = Vec::new();

        for (net_name, start, end) in blocks {
            let g = NetGeneralInfo::from_name(net_name)
                .unwrap_or_else(|e| panic!("failed to load {net_name}: {e}"));
            let xin = build_test_composition(net_name, &g, cat);
            let comp = get_composition_info(cat, &g.chem_id, &xin, false)
                .unwrap_or_else(|e| panic!("composition_info failed: {e}"));

            let lt = rows[start].value;
            let lr = rows[start + 1].value;
            let (temp, rho, eta) = if net_name == "approx21_cr60_plus_co56.net"
            {
                (exp10(9.6), exp10(6.0), -22.590_260_158_215_202)
            } else {
                (exp10(lt), exp10(lr), 0.0)
            };
            let rate_factors = vec![1.0_f64; g.num_reactions];

            let res = net_get(
                &g,
                false,
                &xin,
                temp,
                lt,
                rho,
                lr,
                comp.abar,
                comp.zbar,
                comp.z2bar,
                comp.ye,
                eta,
                0.0,
                0.0,
                &rate_factors,
                1.0,
                ScreeningMode::Extended,
            )
            .unwrap_or_else(|e| panic!("net_get failed for {net_name}: {e}"));

            let mut record = |label: &str, got: f64, want: &HexRow| {
                let ulp = ulp_distance(got, want.raw_bits);
                checked += 1;
                if ulp == 0 {
                    zeros += 1;
                } else {
                    max_ulp = max_ulp.max(ulp);
                    failures.push(format!(
                        "  [{net_name}] {label} ({}): got={got:.17e} \
                         ({:#018x}) want={:.17e} ({:#018x}) ulp={ulp}",
                        want.src_loc,
                        got.to_bits(),
                        want.value,
                        want.raw_bits
                    ));
                }
            };

            if net_name == "approx21_cr60_plus_co56.net" {
                for (row, rxn_name) in [
                    (&rows[start + 2], "rco56ec_to_fe56"),
                    (&rows[start + 3], "rni56ec_to_co56"),
                ] {
                    let rxn_idx =
                        g.reaction_index_by_name(rxn_name).unwrap_or_else(
                            || panic!("missing reaction {rxn_name}"),
                        ) - 1;
                    record(rxn_name, res.rate_raw[rxn_idx], row);
                }
                continue;
            }

            let vals = [
                ("log(eps_nuc)", log10(res.eps_nuc)),
                ("eps_nuc", res.eps_nuc),
                ("d_lneps_dlnT", res.d_eps_nuc_dt * temp / res.eps_nuc),
                ("d_lneps_dlnRho", res.d_eps_nuc_drho * rho / res.eps_nuc),
            ];
            let ref_slice = &rows[start + 2..end];
            for (j, (label, got)) in vals.into_iter().enumerate() {
                assert_eq!(
                    ref_slice[j].label, label,
                    "label mismatch at {net_name}"
                );
                record(label, got, &ref_slice[j]);
            }
            for k in (4..ref_slice.len()).step_by(2) {
                let cname = ref_slice[k].label.as_str();
                assert_eq!(ref_slice[k + 1].label, cname);
                let cval = res.eps_nuc_categories[cat_map[cname]];
                record(&format!("log10({cname})"), log10(cval), &ref_slice[k]);
                record(cname, cval, &ref_slice[k + 1]);
            }
        }

        eprintln!(
            "mesa_26_4_1_net_module_hex_zero_ulp: {zeros}/{checked} at 0 ULP \
             (max_ulp={max_ulp})"
        );
        for line in &failures {
            eprintln!("{line}");
        }
        assert_eq!(checked, 116, "expected 116 checked net_get hex values");
        assert_eq!(
            max_ulp, 0,
            "{zeros}/{checked} at 0 ULP (max_ulp={max_ulp})"
        );
    });
}

#[test]
fn mesa_26_4_1_net_module_hex_burn_diagnostics() {
    let Some(tsv_path) = resolve_module_hex_tsv() else {
        return;
    };
    let rows = load_module_hex_rows(&tsv_path);
    assert_eq!(rows.len(), 140);

    with_numerics_mode(NumericsMode::Strict, || {
        let g = NetGeneralInfo::from_name("approx21.net").unwrap();
        let ht = HelmTable::builtin().expect("builtin HELM table");
        let eos_cfg = EosConfig::default();
        let cat = NuclideCatalog::builtin_ref();
        let idx =
            |name: &str| g.net_iso[cat.get_nuclide_index(name).unwrap()] - 1;
        let rate_factors = vec![1.0_f64; g.num_reactions];

        // Case 8a: inlist_test_one_zone_burn_small_net (TSV rows 131..133).
        let mut x1 = vec![0.0_f64; g.num_isos];
        x1[idx("he4")] = 1.0;
        let times = [0.0, 100.0, 150.0];
        let log10_ts = [8.3, 8.4, 8.3];
        let log10_rhos = [7.0, 7.5, 8.1];
        let etas = [0.0, 0.0, 0.0];
        let out1 = burn_1_zone(
            &g,
            ht,
            &eos_cfg,
            &Burn1ZoneInput {
                t_start: 0.0,
                t_end: 150.0,
                starting_x: &x1,
                times: &times,
                log10_ts: &log10_ts,
                log10_rhos: &log10_rhos,
                etas: &etas,
                dxdt_source_term: None,
                rate_factors: &rate_factors,
                weak_rate_factor: 1.0,
                reaction_qs: &[],
                reaction_neu_qs: &[],
                screening_mode: ScreeningMode::Extended,
                stptry_in: 0.0,
                max_steps: 50_000,
                eps: 1.0e-8,
                odescal: 1.0e-12,
            },
        )
        .unwrap();
        let xsum1: f64 = out1.ending_x.iter().sum();

        // Case 8b: inlist_test_one_zone_burn_const_P (TSV rows 135..139).
        let mut x2 = vec![0.0_f64; g.num_isos];
        x2[idx("c12")] = 0.37;
        x2[idx("o16")] = 0.63;
        let out2 = burn_1_zone_const_p(
            &g,
            ht,
            &eos_cfg,
            &BurnConstPInput {
                solver: RosenbrockVariant::Rodas4,
                starting_temp: 3.0e9,
                starting_x: &x2,
                clip: false,
                t_end: 1.0e-4,
                pressure: exp10(log10(1.0e24)),
                rate_factors: &rate_factors,
                weak_rate_factor: 1.0,
                reaction_qs: &[],
                reaction_neu_qs: &[],
                screening_mode: ScreeningMode::Extended,
                h0: 1.0e-2 * 1.0e-4,
                max_step_size: 1.0,
                max_steps: 1000,
                rtol: 1.0e-7,
                atol: 1.0e-8,
                x_min: -1.0e-10,
            },
        )
        .unwrap();
        let xsum2: f64 = out2.ending_x.iter().sum();

        // Tightened parity gates after fixing `netint` state reuse (Case 8a:
        // matches MESA `nstep=27` and steps 1..19 to 17 digits; step 20 `he4`
        // shifts 2 ULP before 2.7e8x `net_pzextr` cancellation at step 21 and
        // a 3.1e-10 dt shift on the retried step 22) and `burn_const_p`
        // `eos_pt_get_with_linear` + direct `dydt` + accumulation order (Case
        // 8b: step 1 `dt/logT/c12/o16` match at 0 ULP, `logRho` differs by
        // 5.87e-9 from `eos_pt_get` 1e-8 root tolerance).
        let burn_checks: [(&str, f64, &HexRow, f64); 8] = [
            (
                "burn_1_zone c12",
                out1.ending_x[idx("c12")],
                &rows[131],
                5.0e-11,
            ),
            (
                "burn_1_zone he4",
                out1.ending_x[idx("he4")],
                &rows[132],
                2.0e-9,
            ),
            ("burn_1_zone xsum", xsum1, &rows[133], 1.0e-14),
            (
                "burn_const_P o16",
                out2.ending_x[idx("o16")],
                &rows[135],
                1.0e-13,
            ),
            (
                "burn_const_P si28",
                out2.ending_x[idx("si28")],
                &rows[136],
                1.0e-13,
            ),
            (
                "burn_const_P s32",
                out2.ending_x[idx("s32")],
                &rows[137],
                1.0e-12,
            ),
            (
                "burn_const_P mg24",
                out2.ending_x[idx("mg24")],
                &rows[138],
                1.0e-12,
            ),
            ("burn_const_P xsum", xsum2, &rows[139], 1.0e-15),
        ];

        for (label, got, row, tol) in burn_checks {
            let ulp = ulp_distance(got, row.raw_bits);
            let rel = (got - row.value).abs() / row.value.abs();
            eprintln!(
                "  [burn] {label} (tsv={}, {}): got={got:.17e} want={:.17e} \
                 rel={rel:.3e} ulp={ulp}",
                row.label, row.src_loc, row.value
            );
            assert!(
                rel <= tol,
                "{label}: rel {rel:.3e} exceeds {tol:.3e} (got={got:.17e}, want={:.17e})",
                row.value
            );
        }
    });
}
