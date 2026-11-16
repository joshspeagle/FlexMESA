// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration and 100% golden `test_output` parity tests for `flexmesa-neu`.
//!
//! Verifies:
//! 1. All 10,000 rows (300,000 output values and partial derivatives) of
//!    upstream `neu/test/test_output` (plus the 6 inlined golden rows).
//! 2. All 194 solar-profile zones of `neu/test/test_1.0.data`.
//! 3. Exact zero cutoff below $\log_{10} T = 7.0$ ($T \le 10^7\,\mathrm{K}$).
//! 4. Channel flag isolation and additive consistency across all 5 channels.
//! 5. Centered finite-difference partial derivative verification across all
//!    5 Itoh et al. (1996) channels (`pair`, `plas`, `phot`, `brem`, `reco`).
//! 6. Input validation for missing, non-positive, or non-finite inputs.

#![allow(
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use std::fs;
use std::path::PathBuf;

use flexmesa_math::{exp10, log10};
use flexmesa_neu::{neu_get, NeuChannel, NeuError, NeuFlags, NeuRate};

/// Representative golden rows from upstream `neu/test/test_output`.
#[rustfmt::skip]
const GOLDEN_ROWS: [[f64; 34]; 6] = [
    [
        7.35, 6.0, 54.1, 1.0,
        2.506304370798582e-05, 6.142254689714302e-12, 3.49748866088052e-11, -1.109752819494082e-06, 6.04286899609018e-05,
        5.087480723393688e-237, 2.4667269458859518e-05, 4.802510786004552e-09, 3.909403005044627e-07, 3.143783583316795e-11,
        1.2984720806319637e-241, 6.0234751304502245e-12, 2.452270390955503e-15, 1.1631126166544558e-13, 1.6027207676106186e-17,
        -6.815259917322703e-242, 3.479220696747712e-11, -7.5336026348655e-15, 1.9032778479375864e-13, -1.1454083080843881e-16,
        1.1657138345625388e-237, -1.0990661076956863e-06, 5.04822892580582e-11, -1.0738730187375997e-08, 1.5360997222785757e-12,
        -6.306511844983335e-236, 5.945947642633662e-05, -2.731091848860948e-09, 9.719056036415043e-07, 3.902277253835819e-11,
    ],
    [
        7.35, 8.25, 30.5, 13.5,
        5.649612198514403e-05, 1.8739985551069504e-11, -1.1361557539517236e-14, -1.830377418252603e-06, 8.07219108145557e-06,
        0.0, 5.548195060010974e-23, 5.713559028118851e-50, 5.649612198514403e-05, 0.0,
        0.0, 1.6279513207337813e-28, 1.5984380634973777e-55, 1.8739985551069504e-11, 0.0,
        0.0, -6.2099018510533345e-30, -1.1235843420965503e-56, -1.136155753951723e-14, 0.0,
        0.0, 3.4387282275703705e-23, 6.36364364167618e-50, -1.830377418252603e-06, 0.0,
        0.0, -7.768978588214542e-23, -1.4377120820083222e-49, 8.07219108145557e-06, 0.0,
    ],
    [
        7.8, 6.45, 12.8, 6.0,
        4.6515505270632715, 4.872993884392351e-07, -2.9642711357854675e-08, -0.3568710960490215, 0.765577562287946,
        2.833017027507156e-95, 4.620313918675809, 2.368298713781117e-07, 0.031236371557591926, 0.0,
        1.1078563749074264e-100, 4.839411054277924e-07, 6.175741580703057e-14, 3.358221254026858e-09, 0.0,
        -1.9698256790595408e-100, 8.524751696331071e-10, -4.0420947005798934e-13, -3.0494782318017725e-08, 0.0,
        4.115954194278984e-95, -0.36114972813618984, 7.049899997568551e-08, 0.004278561588168344, 0.0,
        -8.780702281128497e-95, 0.7704527533572053, -1.503978666147958e-07, -0.004875040671392619, 0.0,
    ],
    [
        8.25, 8.25, 30.5, 13.5,
        1441.83755938058, 7.645193481143817e-05, -9.27074652380869e-06, 6.767620596181869, -14.050798630043268,
        8.217551134215202e-53, 1421.244571564982, 5.093004080655104e-09, 20.59298781050505, 0.0,
        6.824997901170556e-59, 7.57440723069843e-05, 7.779054765996164e-16, 7.078625036759573e-07, 0.0,
        -1.2759944388711824e-59, -9.264442415501296e-06, -2.243626346204645e-16, -6.3041080830312875e-09, 0.0,
        7.17016107090701e-53, 7.417447466647171, 1.1411456154790384e-09, -0.6498268716064474, 0.0,
        -1.6199252789826947e-52, -16.757936869091765, -2.578143797934123e-09, 2.7071382416266405, 0.0,
    ],
    [
        9.15, 9.15, 42.3, 18.5,
        529369873.7562263, 1.636285579919671, 0.2081990513167074, -19465680.947214354, 44807347.60447707,
        0.1434029313340968, 521529056.631546, 53275.396529930156, 7787541.584747452, 0.0,
        3.192618313020971e-09, 1.6018018086022443, 0.00044913402042604444, 0.0340346341043822, 0.0,
        -8.115165278361871e-10, 0.20856339351638842, -8.864141363809853e-05, -0.00027569997452639525, 0.0,
        0.023709092962562916, -19293916.794500574, 1700.565679487452, -173464.74210235855, 0.0,
        -0.054210520665751966, 44115280.02202021, -3888.320445530766, 695955.9571129074, 0.0,
    ],
    [
        10.05, 10.05, 54.1, 23.5,
        115279402027351.03, 125365.80360269622, -22445.697782695333, 2523824338862.973, -5697503166136.036,
        99262057103942.56, 5847583753280.683, 6511023400657.995, 3658737769469.789, 0.0,
        115898.06215107988, 1730.2464479451487, 5629.019149436875, 2108.475854234317, 0.0,
        -22213.00836521715, 467.3165065068549, -645.4384546311078, -54.56746935392931, 0.0,
        2772125619398.017, -205008524885.154, 13510446792.318666, -56803202442.209015, 0.0,
        -6381787064231.179, 471955795586.6735, -31102773253.80594, 243430875762.274, 0.0,
    ],
];

fn mesa_neu_test_dir() -> (PathBuf, bool) {
    if let Some(dir) = std::env::var_os("MESA_DIR").filter(|v| !v.is_empty()) {
        return (PathBuf::from(dir).join("neu/test"), true);
    }
    if let Some(xdg) =
        std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty())
    {
        return (
            PathBuf::from(xdg).join("flexmesa/mesa_test_inputs/neu/test"),
            false,
        );
    }
    if let Some(home) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
        return (
            PathBuf::from(home)
                .join(".cache/flexmesa/mesa_test_inputs/neu/test"),
            false,
        );
    }
    (PathBuf::from(".flexmesa_test_inputs/neu/test"), false)
}

/// Parses a Fortran floating-point token, handling 3-digit exponents formatted
/// without `'E'` (e.g., `3.2760346416369289-283`).
fn parse_fortran_float(tok: &str) -> f64 {
    if let Ok(v) = tok.parse::<f64>() {
        return v;
    }
    let bytes = tok.as_bytes();
    for i in (1..bytes.len()).rev() {
        if (bytes[i] == b'-' || bytes[i] == b'+')
            && bytes[i - 1] != b'e'
            && bytes[i - 1] != b'E'
            && bytes[i - 1] != b'd'
            && bytes[i - 1] != b'D'
        {
            let mut fixed = String::with_capacity(tok.len() + 1);
            fixed.push_str(&tok[..i]);
            fixed.push('e');
            fixed.push_str(&tok[i..]);
            return fixed.parse::<f64>().expect("valid Fortran float");
        }
    }
    panic!("failed to parse Fortran float: {tok}");
}

fn assert_rel_close(actual: f64, expected: f64, rtol: f64, label: &str) {
    if expected.abs() <= 1e-200 && actual.abs() <= 1e-200 {
        return;
    }
    let scale = expected.abs().max(actual.abs());
    let rel = (actual - expected).abs() / scale;
    assert!(
        rel <= rtol,
        "{label}: actual={actual:.16e}, expected={expected:.16e}, rel={rel:.3e}"
    );
}

fn verify_row(row_idx: usize, row: &[f64], rtol: f64) {
    let (log10_t, log10_rho, abar, zbar) = (row[0], row[1], row[2], row[3]);
    let (t, rho) = (exp10(log10_t), exp10(log10_rho));
    let res = neu_get(
        Some(t),
        Some(log10_t),
        Some(rho),
        Some(log10_rho),
        abar,
        zbar,
        7.5,
        NeuFlags::all(),
    )
    .expect("valid golden row");

    for (d_idx, &actual) in res.loss.iter().enumerate() {
        assert_rel_close(
            actual,
            row[4 + d_idx],
            rtol,
            &format!("row {row_idx} loss[{d_idx}]"),
        );
    }
    for (c_idx, ch) in NeuChannel::ALL.iter().enumerate() {
        let rate = res.channel(*ch).to_array();
        for (d_idx, &actual) in rate.iter().enumerate() {
            assert_rel_close(
                actual,
                row[9 + d_idx * 5 + c_idx],
                rtol,
                &format!("row {row_idx} ch {c_idx} deriv {d_idx}"),
            );
        }
    }
}

/// Verifies all 10,000 rows of upstream `neu/test/test_output`
/// and the 6 representative inlined golden rows.
#[test]
fn test_all_10000_golden_rows_parity_against_upstream_mesa() {
    for (idx, row) in GOLDEN_ROWS.iter().enumerate() {
        verify_row(idx, row, 1e-12);
    }

    let (dir, strict) = mesa_neu_test_dir();
    let path = dir.join("test_output");
    if strict {
        assert!(path.is_file(), "missing {}", path.display());
    }
    if path.is_file() {
        let content = fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!("failed to read {}: {e}", path.display())
        });
        let mut count = 0usize;
        for (line_idx, line) in content.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let cols: Vec<f64> = trimmed
                .split_whitespace()
                .map(parse_fortran_float)
                .collect();
            assert_eq!(cols.len(), 34, "row {line_idx} must have 34 columns");
            verify_row(line_idx, &cols, 1e-10);
            count += 1;
        }
        assert_eq!(
            count, 10_000,
            "expected 10,000 rows in neu/test/test_output"
        );
    }
}

/// Verifies all 194 stellar-profile rows of `neu/test/test_1.0.data`.
#[test]
fn test_solar_profile_test_1_0_data_all_194_zones() {
    // Unconditional check on representative active core and passive envelope zones
    for (rho, t, expect_active) in
        [(154.0, 1.56e7, true), (1.0e-2, 4.0e6, false)]
    {
        let res = neu_get(
            Some(t),
            Some(log10(t)),
            Some(rho),
            Some(log10(rho)),
            1.3,
            1.1,
            7.5,
            NeuFlags::all(),
        )
        .expect("valid inlined solar zone");
        if expect_active {
            assert!(res.total.val > 0.0 && res.total.d_t > 0.0);
        } else {
            assert_eq!(res.total, NeuRate::ZERO);
        }
    }

    let (dir, strict) = mesa_neu_test_dir();
    let path = dir.join("test_1.0.data");
    if strict {
        assert!(path.is_file(), "missing {}", path.display());
    }
    if path.is_file() {
        let content = fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!("failed to read {}: {e}", path.display())
        });
        let mut n_rows = 0usize;
        let mut n_active = 0usize;
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let cols: Vec<f64> = trimmed
                .split_whitespace()
                .map(parse_fortran_float)
                .collect();
            assert_eq!(cols.len(), 5);
            let (rho, t) = (cols[2], cols[3]);
            let res = neu_get(
                Some(t),
                Some(log10(t)),
                Some(rho),
                Some(log10(rho)),
                1.3,
                1.1,
                7.5,
                NeuFlags::all(),
            )
            .expect("valid solar zone");
            if t <= 1.0e7 {
                assert_eq!(res.total, NeuRate::ZERO);
            } else {
                assert!(res.total.val > 0.0 && res.total.d_t > 0.0);
                n_active += 1;
            }
            n_rows += 1;
        }
        assert_eq!(n_rows, 194);
        assert!(n_active > 50);
    }
}

#[test]
fn test_low_temperature_cutoff_returns_exact_zero() {
    let flags = NeuFlags::all();
    let res = neu_get(
        Some(5.0e6),
        Some(6.698970004336019),
        Some(1.0e5),
        Some(5.0),
        12.0,
        6.0,
        7.5,
        flags,
    )
    .expect("valid inputs below log10_T = 7.0");
    assert_eq!(res.total, NeuRate::ZERO);
    for ch in NeuChannel::ALL {
        assert_eq!(res.channel(ch), NeuRate::ZERO);
    }
}

#[test]
fn test_pair_plas_phot_brem_reco_additive_and_flag_isolation() {
    let (log10_t, log10_rho, abar, zbar) = (8.25, 8.25, 30.5, 13.5);
    let (t, rho) = (exp10(log10_t), exp10(log10_rho));
    let all = neu_get(
        Some(t),
        Some(log10_t),
        Some(rho),
        Some(log10_rho),
        abar,
        zbar,
        7.5,
        NeuFlags::all(),
    )
    .expect("all channels");

    let mut sum = [0.0; 5];
    for ch in NeuChannel::ALL {
        let mut single_flag = NeuFlags::none();
        match ch {
            NeuChannel::Pair => single_flag.pair = true,
            NeuChannel::Plas => single_flag.plas = true,
            NeuChannel::Phot => single_flag.phot = true,
            NeuChannel::Brem => single_flag.brem = true,
            NeuChannel::Reco => single_flag.reco = true,
        }
        let isolated = neu_get(
            Some(t),
            Some(log10_t),
            Some(rho),
            Some(log10_rho),
            abar,
            zbar,
            7.5,
            single_flag,
        )
        .expect("single channel");
        for (i, sum_val) in sum.iter_mut().enumerate() {
            assert_rel_close(
                isolated.loss[i],
                all.channel(ch).to_array()[i],
                1e-15,
                "isolated channel",
            );
            *sum_val += isolated.loss[i];
        }
    }
    for (i, &sum_val) in sum.iter().enumerate() {
        assert_rel_close(all.loss[i], sum_val, 1e-15, "additive sum");
    }
}

#[test]
fn test_centered_finite_difference_derivatives_all_channels() {
    let states = [
        (7.35, 6.0, 54.1, 1.0),
        (8.25, 7.5, 24.0, 11.0),
        (10.05, 9.0, 54.1, 23.5),
    ];
    let step = 1.0e-6;
    let flags = NeuFlags::all();

    for (log10_t, log10_rho, abar, zbar) in states {
        let (t, rho) = (exp10(log10_t), exp10(log10_rho));
        let eval = |t_v: f64, r_v: f64, a_v: f64, z_v: f64| {
            neu_get(
                Some(t_v),
                Some(log10(t_v)),
                Some(r_v),
                Some(log10(r_v)),
                a_v,
                z_v,
                7.5,
                flags,
            )
            .unwrap()
        };
        let base = eval(t, rho, abar, zbar);

        let diffs = [
            (
                eval(t * (1.0 + step), rho, abar, zbar),
                eval(t * (1.0 - step), rho, abar, zbar),
                t * step,
            ),
            (
                eval(t, rho * (1.0 + step), abar, zbar),
                eval(t, rho * (1.0 - step), abar, zbar),
                rho * step,
            ),
            (
                eval(t, rho, abar * (1.0 + step), zbar),
                eval(t, rho, abar * (1.0 - step), zbar),
                abar * step,
            ),
            (
                eval(t, rho, abar, zbar * (1.0 + step)),
                eval(t, rho, abar, zbar * (1.0 - step)),
                zbar * step,
            ),
        ];

        for (k, (rp, rm, h)) in diffs.iter().enumerate() {
            let fd_tot = (rp.loss[0] - rm.loss[0]) / (2.0 * h);
            assert_rel_close(base.loss[k + 1], fd_tot, 1e-6, "total deriv");
            for ch in NeuChannel::ALL {
                let idx = ch as usize;
                if base.sources[idx][0] > 1.0e-30 {
                    let fd_ch =
                        (rp.sources[idx][0] - rm.sources[idx][0]) / (2.0 * h);
                    assert_rel_close(
                        base.sources[idx][k + 1],
                        fd_ch,
                        1e-6,
                        "channel deriv",
                    );
                }
            }
        }
    }
}

#[test]
fn test_invalid_inputs_return_domain_errors() {
    let flags = NeuFlags::all();
    assert_eq!(
        neu_get(None, None, Some(1.0e5), Some(5.0), 12.0, 6.0, 7.5, flags),
        Err(NeuError::MissingTemperature)
    );
    assert_eq!(
        neu_get(Some(1.0e8), Some(8.0), None, None, 12.0, 6.0, 7.5, flags),
        Err(NeuError::MissingDensity)
    );
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        for (t, rho, abar) in
            [(bad, 1.0e5, 12.0), (1.0e8, bad, 12.0), (1.0e8, 1.0e5, bad)]
        {
            let res = neu_get(
                Some(t),
                Some(8.0),
                Some(rho),
                Some(5.0),
                abar,
                6.0,
                7.5,
                flags,
            );
            assert!(matches!(res, Err(NeuError::InvalidInput(_))));
        }
    }
}
