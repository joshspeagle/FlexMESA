// SPDX-License-Identifier: GPL-3.0-or-later
//! Hermetic MESA `26.4.1` IEEE-754 binary64 `0-ULP` parity oracles extracted
//! from `mesa-stepdump-windows-v1` (`star/make_planets`, `star/high_z`,
//! `star/hot_cool_wind`, `star/conserve_angular_momentum`, and
//! `star/wd_stable_h_burn`).
//!
//! Verifies both full-model `sizequ` residual norms (`CALL` `residual_norm`,
//! `max_residual`, and `argmax` over MESA's 448-zone `EQU` matrix) and
//! multi-zone `STATE`/`AUX` microphysics (`set_basic_vars`, `do_eos`,
//! `do_kap`, `do_net`: `Peos`, `opacity`, `grada`, `eps_nuc`, `chiRho`,
//! `chiT`, `Cp`) in `NumericsMode::Strict`.

#![forbid(unsafe_code)]
#![allow(clippy::too_many_lines)]

use std::path::PathBuf;

use flexmesa_chem::NuclideCatalog;
use flexmesa_eos::EosConfig;
use flexmesa_kap::{KapConfig, KapLowTOption, KapOption};
use flexmesa_math::{with_numerics_mode, NumericsMode};
use flexmesa_star::micro::{
    do_eos, do_kap, do_net, resolve_model_net, set_basic_vars,
};
use flexmesa_star::solver_norms::compute_residual_norms_with_argmax;
use flexmesa_star::{StarControls, StarJobControls, StarModel};

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/stepdump_windows")
}

fn hex_bits(s: &str) -> u64 {
    u64::from_str_radix(s.trim(), 16)
        .unwrap_or_else(|e| panic!("invalid IEEE-754 hex '{s}': {e}"))
}

fn hex_f64(s: &str) -> f64 {
    f64::from_bits(hex_bits(s))
}

fn ulp_distance(a: f64, b: f64) -> u64 {
    let ba = a.to_bits();
    let bb = b.to_bits();
    if (ba >> 63) == (bb >> 63) {
        ba.abs_diff(bb)
    } else {
        (ba & 0x7fff_ffff_ffff_ffff).saturating_add(bb & 0x7fff_ffff_ffff_ffff)
    }
}

fn assert_bits(label: &str, actual: f64, want_hex: &str) {
    let want_bits = hex_bits(want_hex);
    let got_bits = actual.to_bits();
    if got_bits != want_bits {
        let want = f64::from_bits(want_bits);
        let ulps = ulp_distance(actual, want);
        panic!(
            "{label}: MESA {want:.17e} (0x{want_bits:016X}) vs \
             FlexMESA {actual:.17e} (0x{got_bits:016X}) [{ulps} ULP]"
        );
    }
}

#[test]
fn test_make_planets_model23_sizequ_matches_mesa_stepdump_bit_for_bit() {
    let tsv_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/stepdump_windows/make_planets_s3_k1_m23_equ.tsv");
    let raw = std::fs::read_to_string(&tsv_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", tsv_path.display()));
    let mut lines =
        raw.lines().filter(|l| !l.starts_with('#') && !l.is_empty());
    let header = lines.next().expect("TSV header");
    assert!(header.starts_with("zone\teq1\t"));

    let mut equ = Vec::with_capacity(448 * 12);
    let mut nz = 0_usize;
    for row in lines {
        let mut cols = row.split('\t');
        let zone: usize = cols.next().expect("zone").parse().expect("zone u");
        assert_eq!(zone, nz + 1);
        for _ in 0..12 {
            equ.push(hex_f64(cols.next().expect("eq hex")));
        }
        assert!(cols.next().is_none());
        nz += 1;
    }
    assert_eq!(nz, 448);

    let catalog = NuclideCatalog::builtin_ref();
    let chem_id: Vec<usize> =
        ["h1", "he3", "he4", "c12", "n14", "o16", "ne20", "mg24"]
            .iter()
            .map(|name| catalog.get_nuclide_index(name).expect("nuclide"))
            .collect();
    let controls = StarControls {
        do_burn: true,
        do_mix: true,
        ..StarControls::default()
    };
    let model =
        StarModel::new(nz, chem_id, StarJobControls::default(), controls);
    assert_eq!(model.layout.nvar_hydro, 4);

    let (rnorm, rmax, argmax) =
        compute_residual_norms_with_argmax(&model, 12, &equ);
    assert_bits("make_planets m=23 residual_norm", rnorm, "4002521375EBD08E");
    assert_bits("make_planets m=23 max_residual", rmax, "409CF307C4BE3FB6");
    // MESA 1-based CALL tail is (zone=33, eq=4).
    assert_eq!(argmax, Some((32, 3)));
}

fn configure_case_physics(
    case_name: &str,
) -> (String, StarControls, EosConfig, KapConfig) {
    let eos_config = EosConfig::default();
    match case_name {
        "star/high_z" => (
            String::from("o18_and_ne22.net"),
            StarControls {
                zbase: 0.07,
                ..StarControls::default()
            },
            eos_config,
            KapConfig {
                zbase: 0.07,
                use_type2_opacities: true,
                ..KapConfig::default()
            },
        ),
        "star/hot_cool_wind" => (
            String::from("o18_and_ne22.net"),
            StarControls {
                zbase: 0.02,
                ..StarControls::default()
            },
            eos_config,
            KapConfig {
                zbase: 0.02,
                use_type2_opacities: true,
                ..KapConfig::default()
            },
        ),
        "star/conserve_angular_momentum" => (
            String::from("basic.net"),
            StarControls {
                zbase: 0.02,
                ..StarControls::default()
            },
            eos_config,
            KapConfig {
                zbase: 0.02,
                kap_option: KapOption::Gs98,
                ..KapConfig::default()
            },
        ),
        "star/make_planets" => (
            String::from("basic.net"),
            StarControls {
                zbase: 0.02,
                ..StarControls::default()
            },
            eos_config,
            KapConfig {
                zbase: 0.02,
                kap_option: KapOption::Gs98,
                kap_low_t_option: KapLowTOption::LowTFreedman11,
                ..KapConfig::default()
            },
        ),
        "star/wd_stable_h_burn" => (
            String::from("cno_extras.net"),
            StarControls {
                zbase: 0.02,
                net_logtcut_lo: 0.0,
                net_logtcut_lim: 0.0,
                ..StarControls::default()
            },
            eos_config,
            KapConfig {
                zbase: 0.02,
                use_type2_opacities: true,
                use_zbase_for_type1: false,
                ..KapConfig::default()
            },
        ),
        other => panic!("unexpected stepdump case {other}"),
    }
}

#[test]
fn test_star_stepdump_window_zones_match_mesa_aux_bit_for_bit() {
    let tsv_path = data_dir().join("star_windows_aux.tsv");
    let raw = std::fs::read_to_string(&tsv_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", tsv_path.display()));
    let mut lines =
        raw.lines().filter(|l| !l.starts_with('#') && !l.is_empty());
    let header = lines.next().expect("TSV header");
    assert!(header.starts_with("case\twindow_sha256\t"));

    let catalog = NuclideCatalog::builtin_ref();
    let mut checked = 0_usize;
    let mut failures = Vec::new();

    for row in lines {
        let cols: Vec<&str> = row.split('\t').collect();
        assert_eq!(cols.len(), 20, "expected 20 columns: {row}");
        let case_name = cols[0];
        let model_num = cols[2];
        let zone = cols[3];
        let sp: usize = cols[5].parse().expect("species");
        let species_list: Vec<&str> = cols[6].split(',').collect();
        assert_eq!(species_list.len(), sp);
        let xa_hex: Vec<&str> = cols[19].split(',').collect();
        assert_eq!(xa_hex.len(), sp);

        let (net_name, controls, eos_config, kap_config) =
            configure_case_physics(case_name);
        let chem_id: Vec<usize> = species_list
            .iter()
            .map(|name| catalog.get_nuclide_index(name).expect("nuclide"))
            .collect();
        let job = StarJobControls {
            net_name,
            ..StarJobControls::default()
        };
        let mut model = StarModel::new(1, chem_id, job, controls);
        model.eos_config = eos_config;
        model.kap_config = kap_config;
        model.lnd[0] = hex_f64(cols[7]);
        model.ln_t[0] = hex_f64(cols[8]);
        model.ln_r[0] = hex_f64(cols[9]);
        model.lum[0] = hex_f64(cols[10]);
        model.dm[0] = hex_f64(cols[11]);
        for (j, h) in xa_hex.iter().enumerate() {
            model.xa[j] = hex_f64(h);
        }

        with_numerics_mode(NumericsMode::Strict, || {
            set_basic_vars(&mut model).expect("set_basic_vars");
            do_eos(&mut model).expect("do_eos");
            do_kap(&mut model).expect("do_kap");
            let net_info =
                resolve_model_net(&model).expect("resolve_model_net");
            do_net(&mut model, &net_info, true).expect("do_net");
        });

        let tag = format!("{case_name} m={model_num} z={zone}");
        let channels = [
            ("Peos", model.peos[0], cols[12]),
            ("opacity", model.opacity[0], cols[13]),
            ("eps_nuc", model.eps_nuc[0], cols[14]),
            ("grada", model.grada[0], cols[15]),
            ("chiRho", model.chi_rho[0], cols[16]),
            ("chiT", model.chi_t[0], cols[17]),
            ("Cp", model.cp[0], cols[18]),
        ];
        for (ch, actual, want_hex) in channels {
            let want_bits = hex_bits(want_hex);
            let got_bits = actual.to_bits();
            if got_bits != want_bits {
                let want = f64::from_bits(want_bits);
                let ulps = ulp_distance(actual, want);
                failures.push(format!(
                    "{tag} {ch}: MESA {want:.17e} (0x{want_bits:016X}) \
                     vs Flex {actual:.17e} (0x{got_bits:016X}) [{ulps} ULP]"
                ));
            }
        }
        checked += 1;
    }

    assert_eq!(checked, 21);
    assert!(
        failures.is_empty(),
        "stepdump AUX 0-ULP mismatches ({} failures):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}
