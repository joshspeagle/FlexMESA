// SPDX-License-Identifier: GPL-3.0-or-later
//! Bit-level MESA `26.4.1` Fortran oracle verification for `flexmesa-neu` (`#161`).
//!
//! Parses `benchmarks/fortran_mesa/neu_oracles/mesa_26_4_1_neu_oracles.tsv`
//! (captured directly against compiled MESA `r26.4.1` `libneu.a` + `CRMATH`)
//! and verifies `0 ULP` agreement across all `1,080` outputs (`36` cases across
//! low-temperature cutoffs, pair-production `T9 < 10` / `T9 >= 10`, plasmon
//! decay, all three photoneutrino Fourier bands, bremsstrahlung blending,
//! recombination `zfermim12` / `ifermi12` `#193` rational approximations, and
//! per-channel `NeuFlags` masks).

#![forbid(unsafe_code)]
#![allow(
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines
)]

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use flexmesa_math::{exp10, with_numerics_mode, NumericsMode};
use flexmesa_neu::{ifermi12, neu_get, zfermim12, NeuFlags};

fn oracles_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/fortran_mesa/neu_oracles")
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

fn decode_flags(mask: &str) -> NeuFlags {
    let bytes = mask.trim().as_bytes();
    assert_eq!(bytes.len(), 5, "expected 5-bit flags mask, got {mask:?}");
    NeuFlags {
        pair: bytes[0] == b'1',
        plas: bytes[1] == b'1',
        phot: bytes[2] == b'1',
        brem: bytes[3] == b'1',
        reco: bytes[4] == b'1',
    }
}

#[derive(Debug)]
struct NeuOracleCase {
    label: String,
    flags: NeuFlags,
    log10_t: f64,
    log10_rho: f64,
    abar: f64,
    zbar: f64,
    log10_tlim: f64,
    outputs: [Option<f64>; 30],
}

#[test]
fn test_neu_oracles_bundle_records_mesa_26_4_1_provenance_and_hygiene() {
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
        "gen_neu_oracles.f90",
        "mesa_26_4_1_neu_oracles.tsv",
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

    let tsv = fs::read_to_string(dir.join("mesa_26_4_1_neu_oracles.tsv"))
        .expect("read neu oracle TSV");
    for header in [
        "# FlexMESA MESA 26.4.1 thermal neutrino (neu) oracle vectors (#161)",
        "# mesa_version: r26.4.1",
        "# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd",
        "# mesasdk_version: x86_64-linux-26.6.1",
        "# math_backend: CRMATH",
    ] {
        assert!(tsv.contains(header), "missing provenance header {header:?}");
    }
}

#[test]
fn test_all_1080_mesa_26_4_1_neu_oracles_match_to_zero_ulp_in_strict_mode() {
    let path = oracles_dir().join("mesa_26_4_1_neu_oracles.tsv");
    let text = fs::read_to_string(&path).expect("read neu oracle TSV");

    let mut cases: BTreeMap<usize, NeuOracleCase> = BTreeMap::new();
    let mut row_count = 0_usize;

    for (line_idx, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.starts_with("case_id\t")
        {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(
            cols.len(),
            12,
            "malformed neu oracle line {}: {line:?}",
            line_idx + 1
        );
        let case_id: usize = cols[0].trim().parse().expect("case_id");
        let out_idx: usize = cols[8].trim().parse().expect("out_idx");
        assert!(out_idx < 30, "out_idx out of bounds: {out_idx}");
        let val = decode_hex_f64(cols[10]);

        let entry = cases.entry(case_id).or_insert_with(|| NeuOracleCase {
            label: cols[1].trim().to_string(),
            flags: decode_flags(cols[2]),
            log10_t: decode_hex_f64(cols[3]),
            log10_rho: decode_hex_f64(cols[4]),
            abar: decode_hex_f64(cols[5]),
            zbar: decode_hex_f64(cols[6]),
            log10_tlim: decode_hex_f64(cols[7]),
            outputs: [None; 30],
        });
        entry.outputs[out_idx] = Some(val);
        row_count += 1;
    }

    assert_eq!(cases.len(), 36, "expected 36 neu oracle cases");
    assert_eq!(row_count, 1080, "expected 1080 neu oracle rows");

    with_numerics_mode(NumericsMode::Strict, || {
        for (case_id, c) in &cases {
            let t = exp10(c.log10_t);
            let rho = exp10(c.log10_rho);
            let res = neu_get(
                Some(t),
                Some(c.log10_t),
                Some(rho),
                Some(c.log10_rho),
                c.abar,
                c.zbar,
                c.log10_tlim,
                c.flags,
            )
            .unwrap_or_else(|e| {
                panic!("neu_get failed on case {case_id} ({}): {e}", c.label)
            });

            // Also verify `None` linear T/Rho reconstruction matches bit-for-bit.
            let res_from_logs = neu_get(
                None,
                Some(c.log10_t),
                None,
                Some(c.log10_rho),
                c.abar,
                c.zbar,
                c.log10_tlim,
                c.flags,
            )
            .expect("neu_get from log10_T/log10_Rho");
            assert_eq!(res.loss, res_from_logs.loss);
            assert_eq!(res.sources, res_from_logs.sources);

            let mut actual = [0.0_f64; 30];
            actual[..5].copy_from_slice(&res.loss);
            for ch in 0..5 {
                for d in 0..5 {
                    actual[5 + ch * 5 + d] = res.sources[ch][d];
                }
            }

            for idx in 0..30 {
                let expected = c.outputs[idx].unwrap_or_else(|| {
                    panic!("missing out_idx {idx} in case {case_id}")
                });
                let got = actual[idx];
                let ulp = ulp_distance(got, expected);
                assert_eq!(
                    ulp,
                    0,
                    "case {case_id} ({}) out[{idx}] nonzero ULP={ulp}: got={got:.17e} (0x{:016X}) expected={expected:.17e} (0x{:016X})",
                    c.label,
                    got.to_bits(),
                    expected.to_bits()
                );
            }
        }
    });
}

#[test]
fn test_reco_fermi_dirac_approximations_preserve_mesa_26_4_1_branches() {
    // Locks down the non-degenerate, transitional, and degenerate rational fits
    // in `neu/private/mod_neu.f90:574-649` (including `#193` deferred audit notes).
    with_numerics_mode(NumericsMode::Strict, || {
        let mut prev_nu = f64::NEG_INFINITY;
        for f in [1.0e-5, 0.25, 4.0, 25.0] {
            let nu = ifermi12(f);
            assert!(nu.is_finite() && nu > prev_nu);
            prev_nu = nu;
        }
        let mut prev_fm12 = 0.0;
        for nu in [-8.0, -0.5, 0.5, 3.0, 12.0] {
            let fm12 = zfermim12(nu);
            assert!(fm12 > prev_fm12);
            prev_fm12 = fm12;
        }
    });
}
