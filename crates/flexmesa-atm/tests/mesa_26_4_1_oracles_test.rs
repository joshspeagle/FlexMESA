// SPDX-License-Identifier: LGPL-3.0-or-later
//! Bit-level MESA `26.4.1` Fortran oracle and `F2-23` surface boundary condition
//! verification for `flexmesa-atm` (`#161`).
//!
//! Parses `benchmarks/fortran_mesa/atm_oracles/mesa_26_4_1_atm_oracles.tsv`
//! (captured directly against compiled MESA `r26.4.1` `libatm.a` + `libeos.a` +
//! `libkap.a` + `CRMATH` with `skip_partials = .false.`) and verifies all `234`
//! rows under `NumericsMode::Strict`, plus centered finite-difference partial
//! verification and loud error handling for `F2-23`.

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use flexmesa_atm::{
    atm_teff, eval_e2, eval_irradiated, eval_paczynski_gradr, eval_t_tau,
    eval_t_tau_atm, eval_t_tau_dq_dtau, eval_table, eval_table_atm,
    get_t_tau_base, get_t_tau_id, get_table_alfa_beta, get_table_base,
    get_table_id, radiation_pressure, AtmEosCallbackResult, AtmError,
    AtmKapCallbackResult, AtmTTauOpacity, ATM_TABLE_PHOTOSPHERE,
    ATM_TABLE_TAU_100,
};
use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{LN10, LSUN, MSUN, RSUN, STANDARD_CGRAV};
use flexmesa_eos::{eos_pt_get, EosConfig, HelmTable, I_ETA, I_LN_FREE_E};
use flexmesa_kap::{kap_get, KapConfig, KapLowTOption};
use flexmesa_math::{exp, log10, with_numerics_mode, NumericsMode};

fn oracles_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/fortran_mesa/atm_oracles")
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

/// Unfloored relative difference `|actual - expected| / |expected|` (`CLAUDE.md`
/// Strict Numerics: never floor denominators with `max(..., 1.0)`).
fn assert_unfloored_rel_close(
    actual: f64,
    expected: f64,
    rtol: f64,
    ctx: &str,
) {
    if expected == 0.0 {
        assert_eq!(actual, 0.0, "{ctx}: expected exact 0.0, got {actual:.17e}");
        return;
    }
    let rel = ((actual - expected) / expected).abs();
    assert!(
        rel <= rtol,
        "{ctx}: rel={rel:.3e} > rtol={rtol:.3e} (actual={actual:.17e}, expected={expected:.17e})"
    );
}

fn solar_species_and_xa(x: f64, z: f64) -> (Vec<usize>, Vec<f64>) {
    let cat = NuclideCatalog::builtin_ref();
    let names = ["h1", "he4", "c12", "n14", "o16", "ne20", "mg24"];
    let chem_id: Vec<usize> = names
        .iter()
        .map(|n| cat.get_by_name(n).unwrap().chem_id)
        .collect();
    let y = 1.0 - (x + z);
    let xc = 3.272_459_210_526_323_5e-3;
    let xn = 9.502_384_210_526_329_2e-4;
    let xo = 8.821_800_000_000_060_1e-3;
    let sum6 = x + y + xc + xn + xo;
    let xa = vec![x, y, xc, xn, xo, 0.0, 1.0 - sum6];
    (chem_id, xa)
}

fn make_eos_proc(
    chem_id: Vec<usize>,
    xa: Vec<f64>,
) -> impl FnMut(f64, f64) -> Result<AtmEosCallbackResult, AtmError> {
    let ht = HelmTable::builtin().expect("builtin HELM table");
    let cfg = EosConfig::default();
    move |ln_p: f64, ln_t: f64| {
        let t = exp(ln_t);
        let p = exp(ln_p);
        let prad = radiation_pressure(t);
        let pgas = (p - prad).max(1.0e-99);
        let log_pgas = log10(pgas);
        let log_t = ln_t / LN10;
        let pt = eos_pt_get(ht, &chem_id, &xa, log_pgas, log_t, &cfg).map_err(
            |e| AtmError::CallbackError {
                context: "eos_proc",
                detail: e.to_string(),
            },
        )?;
        Ok(AtmEosCallbackResult {
            ln_rho: pt.log10_rho * LN10,
            res: pt.eos.res,
            dres_dln_rho: pt.eos.d_dlnd,
            dres_dln_t: pt.eos.d_dln_t,
        })
    }
}

fn make_kap_proc(
    chem_id: Vec<usize>,
    xa: Vec<f64>,
    zbase: f64,
    low_t_option: KapLowTOption,
) -> impl FnMut(
    f64,
    f64,
    &AtmEosCallbackResult,
) -> Result<AtmKapCallbackResult, AtmError> {
    let cfg = KapConfig {
        zbase,
        kap_low_t_option: low_t_option,
        ..KapConfig::default()
    };
    move |ln_rho: f64, ln_t: f64, eos: &AtmEosCallbackResult| {
        let out = kap_get(
            &cfg,
            &chem_id,
            &xa,
            ln_rho / LN10,
            ln_t / LN10,
            eos.res[I_LN_FREE_E],
            eos.dres_dln_rho[I_LN_FREE_E],
            eos.dres_dln_t[I_LN_FREE_E],
            eos.res[I_ETA],
            eos.dres_dln_rho[I_ETA],
            eos.dres_dln_t[I_ETA],
        )
        .map_err(|e| AtmError::CallbackError {
            context: "kap_proc",
            detail: e.to_string(),
        })?;
        Ok(AtmKapCallbackResult {
            kap: out.kap,
            dlnkap_dln_rho: out.dlnkap_dln_rho,
            dlnkap_dln_t: out.dlnkap_dln_t,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct GroupKey {
    section: String,
    label: String,
    in_bits: [u64; 5],
}

#[test]
fn test_atm_oracles_bundle_records_mesa_26_4_1_provenance_and_hygiene() {
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
        "gen_atm_oracles.f90",
        "mesa_26_4_1_atm_oracles.tsv",
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

    let tsv = fs::read_to_string(dir.join("mesa_26_4_1_atm_oracles.tsv"))
        .expect("read atm oracle TSV");
    for header in [
        "# FlexMESA MESA 26.4.1 surface atmosphere (atm + F2-23 atm_support) oracle vectors (#161)",
        "# mesa_version: r26.4.1",
        "# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd",
        "# mesasdk_version: x86_64-linux-26.6.1",
        "# math_backend: CRMATH",
    ] {
        assert!(tsv.contains(header), "missing provenance header {header:?}");
    }
}

#[test]
fn test_all_234_mesa_26_4_1_atm_oracles_match_in_strict_mode() {
    let path = oracles_dir().join("mesa_26_4_1_atm_oracles.tsv");
    let text = fs::read_to_string(&path).expect("read atm oracle TSV");

    let mut groups: BTreeMap<GroupKey, BTreeMap<String, f64>> = BTreeMap::new();
    let mut row_count = 0_usize;

    for (idx, line) in text.lines().enumerate() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') || t.starts_with("row_id\t") {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 11, "malformed line {}: {line:?}", idx + 1);
        let key = GroupKey {
            section: cols[1].trim().to_string(),
            label: cols[2].trim().to_string(),
            in_bits: [
                decode_hex_f64(cols[3]).to_bits(),
                decode_hex_f64(cols[4]).to_bits(),
                decode_hex_f64(cols[5]).to_bits(),
                decode_hex_f64(cols[6]).to_bits(),
                decode_hex_f64(cols[7]).to_bits(),
            ],
        };
        groups
            .entry(key)
            .or_default()
            .insert(cols[8].trim().to_string(), decode_hex_f64(cols[9]));
        row_count += 1;
    }
    assert_eq!(row_count, 234, "expected 234 atm oracle rows");

    with_numerics_mode(NumericsMode::Strict, || {
        for (key, outs) in &groups {
            let v1 = f64::from_bits(key.in_bits[0]);
            let v2 = f64::from_bits(key.in_bits[1]);
            let v3 = f64::from_bits(key.in_bits[2]);
            let v4 = f64::from_bits(key.in_bits[3]);
            let v5 = f64::from_bits(key.in_bits[4]);

            match key.section.as_str() {
                "t_tau_rel" => {
                    let rel_id = v1.round() as i32;
                    if let Some(&exp_base) = outs.get("tau_base") {
                        let got =
                            get_t_tau_base(rel_id).expect("get_t_tau_base");
                        assert_eq!(ulp_distance(got, exp_base), 0);
                    } else {
                        let got_lnt =
                            eval_t_tau(rel_id, v2, v3).expect("eval_t_tau");
                        let got_dq = eval_t_tau_dq_dtau(rel_id, v2)
                            .expect("eval_t_tau_dq_dtau");
                        assert_eq!(ulp_distance(got_lnt, outs["lnT"]), 0);
                        assert_eq!(ulp_distance(got_dq, outs["dq_dtau"]), 0);
                    }
                }
                "e2" => {
                    let (e2_val, de2_val) = eval_e2(v1).expect("eval_e2");
                    assert_eq!(ulp_distance(e2_val, outs["e2"]), 0);
                    assert_eq!(ulp_distance(de2_val, outs["de2_dx"]), 0);
                }
                "paczynski" => {
                    let gradr = eval_paczynski_gradr(
                        v1,
                        v2,
                        v3,
                        v4,
                        v5,
                        LSUN,
                        MSUN,
                        RSUN,
                        STANDARD_CGRAV,
                    );
                    assert_eq!(ulp_distance(gradr, outs["gradr"]), 0);
                }
                "t_tau_atm" => {
                    let parts: Vec<&str> = key.label.split(':').collect();
                    let rel_name = parts[1];
                    let op_mode = parts[2];
                    let kap_g = match op_mode {
                        "fixed" => {
                            if parts[0].contains("solar") {
                                0.35
                            } else {
                                0.34
                            }
                        }
                        "iterated" => 0.50,
                        "varying" => 0.40,
                        other => panic!("unexpected op_mode {other}"),
                    };
                    let errtol =
                        if op_mode == "varying" { 1.0e-9 } else { 1.0e-6 };
                    let (cid, xa) = solar_species_and_xa(0.70, 0.02);
                    let teff = atm_teff(v3, v4);
                    let res = eval_t_tau_atm(
                        v1,
                        v3,
                        v4,
                        v5,
                        STANDARD_CGRAV,
                        kap_g,
                        v2,
                        rel_name,
                        op_mode,
                        make_eos_proc(cid.clone(), xa.clone()),
                        make_kap_proc(
                            cid,
                            xa,
                            0.02,
                            KapLowTOption::LowTFa05Gs98,
                        ),
                        errtol,
                        30,
                        500,
                        false,
                        teff,
                    )
                    .expect("eval_t_tau_atm");

                    let pairs = [
                        ("lnT", res.ln_t),
                        ("dlnT_dL", res.dln_t_dl),
                        ("dlnT_dlnR", res.dln_t_dln_r),
                        ("dlnT_dlnM", res.dln_t_dln_m),
                        ("dlnT_dlnkap", res.dln_t_dln_kap),
                        ("lnP", res.ln_p),
                        ("dlnP_dL", res.dln_p_dl),
                        ("dlnP_dlnR", res.dln_p_dln_r),
                        ("dlnP_dlnM", res.dln_p_dln_m),
                        ("dlnP_dlnkap", res.dln_p_dln_kap),
                        ("kap", res.kap),
                    ];

                    for (name, got) in pairs {
                        let expected = outs[name];
                        if op_mode == "fixed" || op_mode == "iterated" {
                            assert_eq!(
                                ulp_distance(got, expected),
                                0,
                                "t_tau_atm {} {name} nonzero ULP: got={got:.17e} expected={expected:.17e}",
                                key.label
                            );
                        } else {
                            // `varying` integrates `dP/dtau` via adaptive `dopri5`
                            // and finite-differences `lnP` in `ln Teff`.
                            match name {
                                "lnT" | "dlnT_dL" | "dlnT_dlnR"
                                | "dlnT_dlnM" | "dlnT_dlnkap"
                                | "dlnP_dlnkap" | "kap" => {
                                    assert_eq!(
                                        ulp_distance(got, expected),
                                        0,
                                        "varying {} {name} nonzero ULP",
                                        key.label
                                    );
                                }
                                "lnP" => assert_unfloored_rel_close(
                                    got, expected, 5.0e-12, &key.label,
                                ),
                                _ => assert_unfloored_rel_close(
                                    got, expected, 1.0e-7, &key.label,
                                ),
                            }
                        }
                    }
                }
                "table" => {
                    let tid = get_table_id(&key.label).expect("get_table_id");
                    let res = eval_table(
                        v1,
                        v2,
                        v3,
                        STANDARD_CGRAV,
                        tid,
                        v4,
                        false,
                        v5,
                    )
                    .expect("eval_table");
                    let pairs = [
                        ("lnT", res.ln_t),
                        ("dlnT_dL", res.dln_t_dl),
                        ("dlnT_dlnR", res.dln_t_dln_r),
                        ("dlnT_dlnM", res.dln_t_dln_m),
                        ("dlnT_dlnkap", res.dln_t_dln_kap),
                        ("lnP", res.ln_p),
                        ("dlnP_dL", res.dln_p_dl),
                        ("dlnP_dlnR", res.dln_p_dln_r),
                        ("dlnP_dlnM", res.dln_p_dln_m),
                        ("dlnP_dlnkap", res.dln_p_dln_kap),
                        ("kap", res.kap),
                    ];
                    for (name, got) in pairs {
                        assert_unfloored_rel_close(
                            got, outs[name], 5.0e-14, &key.label,
                        );
                    }
                }
                "table_ab" => {
                    let tid = if key.label.contains("photosphere") {
                        ATM_TABLE_PHOTOSPHERE
                    } else {
                        ATM_TABLE_TAU_100
                    };
                    let teff = if v4 > 0.0 { v4 } else { atm_teff(v1, v2) };
                    let (a, b) = get_table_alfa_beta(
                        v1,
                        teff,
                        v2,
                        v3,
                        STANDARD_CGRAV,
                        tid,
                    )
                    .expect("get_table_alfa_beta");
                    assert_eq!(ulp_distance(a, outs["alfa"]), 0);
                    assert_eq!(ulp_distance(b, outs["beta"]), 0);
                    if let Some(&exp_base) = outs.get("tau_base") {
                        let tbase =
                            get_table_base(tid).expect("get_table_base");
                        assert_eq!(ulp_distance(tbase, exp_base), 0);
                    }
                }
                "table_blend" => {
                    let off_rel = if key.label.contains("edge") {
                        "Eddington"
                    } else {
                        "solar_Hopf"
                    };
                    let (cid, xa) = solar_species_and_xa(0.70, v4);
                    let teff = atm_teff(v1, v2);
                    let (res, tau_base) = eval_table_atm(
                        v1,
                        v2,
                        v3,
                        STANDARD_CGRAV,
                        "WD_tau_25",
                        v4,
                        1.0,
                        "T_tau",
                        off_rel,
                        "fixed",
                        v5,
                        1.0,
                        make_eos_proc(cid.clone(), xa.clone()),
                        make_kap_proc(cid, xa, v4, KapLowTOption::LowTFa05Gs98),
                        1.0e-6,
                        0,
                        500,
                        false,
                        teff,
                    )
                    .expect("eval_table_atm");

                    // `kap` column in `table_blend` stores `tau_base` (`25.0` on
                    // partial blend vs `0.4116433502` on pure off-table).
                    assert_eq!(
                        ulp_distance(tau_base, outs["kap"]),
                        0,
                        "tau_base mismatch on {}",
                        key.label
                    );
                    let pairs = [
                        ("lnT", res.ln_t),
                        ("dlnT_dL", res.dln_t_dl),
                        ("dlnT_dlnR", res.dln_t_dln_r),
                        ("dlnT_dlnM", res.dln_t_dln_m),
                        ("dlnT_dlnkap", res.dln_t_dln_kap),
                        ("lnP", res.ln_p),
                        ("dlnP_dL", res.dln_p_dl),
                        ("dlnP_dlnR", res.dln_p_dln_r),
                        ("dlnP_dlnM", res.dln_p_dln_m),
                        ("dlnP_dlnkap", res.dln_p_dln_kap),
                    ];
                    for (name, got) in pairs {
                        if key.label.contains("pure_off_table") {
                            assert_eq!(
                                ulp_distance(got, outs[name]),
                                0,
                                "pure_off_table {name} nonzero ULP"
                            );
                        } else {
                            assert_unfloored_rel_close(
                                got, outs[name], 5.0e-14, &key.label,
                            );
                        }
                    }
                }
                "irradiated" => {
                    let (cid, xa) = solar_species_and_xa(0.70, 0.01);
                    let res = eval_irradiated(
                        v1,
                        v2,
                        v3,
                        STANDARD_CGRAV,
                        v4,
                        1.0e6,
                        1.5e-2,
                        4.0e-3,
                        v5,
                        make_eos_proc(cid.clone(), xa.clone()),
                        make_kap_proc(
                            cid,
                            xa,
                            0.01,
                            KapLowTOption::LowTFreedman11,
                        ),
                        1.0e-6,
                        30,
                        false,
                    )
                    .expect("eval_irradiated");
                    let pairs = [
                        ("kap", res.kap),
                        ("tau", res.tau),
                        ("lnT", res.ln_t),
                        ("dlnT_dL", res.dln_t_dl),
                        ("dlnT_dlnR", res.dln_t_dln_r),
                        ("dlnT_dlnM", res.dln_t_dln_m),
                        ("dlnT_dlnkap", res.dln_t_dln_kap),
                    ];
                    for (name, got) in pairs {
                        assert_unfloored_rel_close(
                            got, outs[name], 1.0e-14, &key.label,
                        );
                    }
                }
                other => panic!("unexpected atm section {other}"),
            }
        }
    });
}

#[test]
fn test_f2_23_iterated_partials_match_centered_finite_differences() {
    with_numerics_mode(NumericsMode::Strict, || {
        let (cid, xa) = solar_species_and_xa(0.70, 0.02);
        let eval_at = |l_in: f64, r_in: f64, m_in: f64| {
            let teff = atm_teff(l_in, r_in);
            eval_t_tau_atm(
                2.0 / 3.0,
                l_in,
                r_in,
                m_in,
                STANDARD_CGRAV,
                0.50,
                1.5,
                "solar_Hopf",
                "iterated",
                make_eos_proc(cid.clone(), xa.clone()),
                make_kap_proc(
                    cid.clone(),
                    xa.clone(),
                    0.02,
                    KapLowTOption::LowTFa05Gs98,
                ),
                1.0e-11,
                50,
                500,
                false,
                teff,
            )
            .expect("eval_t_tau_atm iterated")
        };

        let base = eval_at(LSUN, RSUN, MSUN);
        assert_eq!(base.dln_p_dln_kap, 0.0);

        let h = 1.0e-6;
        let dl = LSUN * h;
        let p_lp = eval_at(LSUN + dl, RSUN, MSUN).ln_p;
        let p_lm = eval_at(LSUN - dl, RSUN, MSUN).ln_p;
        assert_unfloored_rel_close(
            base.dln_p_dl,
            (p_lp - p_lm) / (2.0 * dl),
            5.0e-6,
            "iterated dlnP_dL vs FD",
        );

        let p_rp = eval_at(LSUN, RSUN * exp(h), MSUN).ln_p;
        let p_rm = eval_at(LSUN, RSUN * exp(-h), MSUN).ln_p;
        assert_unfloored_rel_close(
            base.dln_p_dln_r,
            (p_rp - p_rm) / (2.0 * h),
            5.0e-6,
            "iterated dlnP_dlnR vs FD",
        );

        let p_mp = eval_at(LSUN, RSUN, MSUN * exp(h)).ln_p;
        let p_mm = eval_at(LSUN, RSUN, MSUN * exp(-h)).ln_p;
        assert_unfloored_rel_close(
            base.dln_p_dln_m,
            (p_mp - p_mm) / (2.0 * h),
            5.0e-6,
            "iterated dlnP_dlnM vs FD",
        );
    });
}

#[test]
fn test_f2_23_invalid_options_and_callback_errors_fail_loudly() {
    assert!(matches!(
        get_t_tau_id("simple_photosphere"),
        Err(AtmError::InvalidOption { .. })
    ));
    assert!(matches!(
        get_t_tau_id("nonexistent_relation"),
        Err(AtmError::InvalidOption { .. })
    ));
    assert!(matches!(
        AtmTTauOpacity::from_name("bogus_opacity"),
        Err(AtmError::InvalidOption { .. })
    ));
    assert!(matches!(
        get_table_id("nonexistent_table"),
        Err(AtmError::InvalidOption { .. })
    ));

    let bad_eos =
        |_lnp: f64, _lnt: f64| -> Result<AtmEosCallbackResult, AtmError> {
            Err(AtmError::CallbackError {
                context: "eos_proc",
                detail: "synthetic EOS failure".to_string(),
            })
        };
    let bad_kap = |_lnr: f64,
                   _lnt: f64,
                   _eos: &AtmEosCallbackResult|
     -> Result<AtmKapCallbackResult, AtmError> {
        Err(AtmError::CallbackError {
            context: "kap_proc",
            detail: "synthetic kap failure".to_string(),
        })
    };

    // `'iterated'` with `max_iters == 0` fails loudly.
    let err_zero_iters = eval_t_tau_atm(
        2.0 / 3.0,
        LSUN,
        RSUN,
        MSUN,
        STANDARD_CGRAV,
        0.4,
        1.0,
        "Eddington",
        "iterated",
        bad_eos,
        bad_kap,
        1.0e-6,
        0,
        500,
        false,
        5777.0,
    );
    assert!(matches!(
        err_zero_iters,
        Err(AtmError::InvalidArgument { .. })
    ));

    // `eval_table_atm` rejects `tau_factor != 1.0`.
    let err_tau_factor = eval_table_atm(
        LSUN,
        RSUN,
        MSUN,
        STANDARD_CGRAV,
        "photosphere",
        0.02,
        2.0,
        "T_tau",
        "Eddington",
        "fixed",
        0.4,
        1.0,
        bad_eos,
        bad_kap,
        1.0e-6,
        0,
        500,
        false,
        5777.0,
    );
    assert!(matches!(
        err_tau_factor,
        Err(AtmError::InvalidArgument { .. })
    ));

    // Off-table query (`Teff = 80,000 K`) with empty `atm_off_table_option` fails loudly.
    let l_hot = LSUN * 1.0e5;
    let teff_hot = atm_teff(l_hot, RSUN);
    let err_empty_off = eval_table_atm(
        l_hot,
        RSUN,
        MSUN,
        STANDARD_CGRAV,
        "tau_100",
        0.02,
        1.0,
        "",
        "Eddington",
        "fixed",
        0.4,
        1.0,
        bad_eos,
        bad_kap,
        1.0e-6,
        0,
        500,
        false,
        teff_hot,
    );
    assert!(matches!(err_empty_off, Err(AtmError::InvalidOption { .. })));

    // Off-table query with `atm_off_table_option = "T_tau"` and `'iterated'` propagates callback errors.
    let err_cb = eval_table_atm(
        l_hot,
        RSUN,
        MSUN,
        STANDARD_CGRAV,
        "tau_100",
        0.02,
        1.0,
        "T_tau",
        "Eddington",
        "iterated",
        0.4,
        1.0,
        bad_eos,
        bad_kap,
        1.0e-6,
        30,
        500,
        false,
        teff_hot,
    );
    assert!(matches!(err_cb, Err(AtmError::CallbackError { .. })));
}
