// SPDX-License-Identifier: GPL-3.0-or-later
//! Hermetic MESA `26.4.1` IEEE-754 binary64 `0-ULP` parity oracles extracted
//! from `mesa-stepdump-windows-v1` (`binary/wind_fed_bhhmxb` and
//! `binary/star_plus_point_mass_explicit_mdot`).
//!
//! Verifies donor star 1's 909-zone `sizequ` residual norms (`CALL`
//! `residual_norm`, `max_residual`, and `argmax` over MESA's `EQU` matrix) and
//! multi-zone `STATE`/`AUX` microphysics (`set_basic_vars`, `do_eos`,
//! `do_kap`, `do_net`) in `NumericsMode::Strict`.

#![forbid(unsafe_code)]

use std::path::PathBuf;

use flexmesa_math::{with_numerics_mode, NumericsMode};
use flexmesa_star::micro::{
    do_eos, do_kap, do_net, resolve_model_net, set_basic_vars,
};
use flexmesa_star::solver_norms::compute_residual_norms_with_argmax;
use flexmesa_star::{
    load_zams_model, StarControls, StarJobControls, StarModel,
};

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

fn basic_net_template(
    nz: usize,
    use_free_eos: bool,
    controls: StarControls,
) -> StarModel {
    let zams =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("load basic.net ZAMS template");
    assert_eq!(
        zams.iso_names,
        ["h1", "he3", "he4", "c12", "n14", "o16", "ne20", "mg24"]
    );
    let mut model = StarModel::new(
        nz,
        zams.chem_id.clone(),
        StarJobControls::default(),
        controls,
    );
    model.eos_config = zams.eos_config;
    model.eos_config.use_free_eos = use_free_eos;
    model.kap_config = zams.kap_config;
    model.kap_config.zbase = 0.02;
    model
}

#[test]
fn test_wind_fed_bhhmxb_donor_m101_sizequ_matches_mesa_stepdump_bit_for_bit() {
    let tsv_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/stepdump_windows/wind_fed_bhhmxb_s1_k1_m101_equ.tsv");
    let raw = std::fs::read_to_string(&tsv_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", tsv_path.display()));
    let mut lines =
        raw.lines().filter(|l| !l.starts_with('#') && !l.is_empty());
    let header = lines.next().expect("TSV header");
    assert!(header.starts_with("zone\teq1\t"));

    let mut equ = Vec::with_capacity(909 * 12);
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
    assert_eq!(nz, 909);

    let controls = StarControls {
        do_burn: true,
        do_mix: true,
        zbase: 0.02,
        ..StarControls::default()
    };
    let model = basic_net_template(nz, false, controls);
    assert_eq!(model.layout.nvar_hydro, 4);

    let (rnorm, rmax, argmax) =
        compute_residual_norms_with_argmax(&model, 12, &equ);
    assert_bits(
        "wind_fed_bhhmxb m=101 residual_norm",
        rnorm,
        "3FA2A90C4F579808",
    );
    assert_bits(
        "wind_fed_bhhmxb m=101 max_residual",
        rmax,
        "40373E14E51D8CA0",
    );
    // MESA 1-based CALL tail is (zone=201, eq=4).
    assert_eq!(argmax, Some((200, 3)));
}

#[test]
fn test_binary_stepdump_donor_zones_match_mesa_aux_bit_for_bit() {
    let tsv_path = data_dir().join("binary_windows_aux.tsv");
    let raw = std::fs::read_to_string(&tsv_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", tsv_path.display()));
    let mut lines =
        raw.lines().filter(|l| !l.starts_with('#') && !l.is_empty());
    let header = lines.next().expect("TSV header");
    assert!(header.starts_with("case\twindow_sha256\t"));

    let mut checked = 0_usize;
    let mut failures = Vec::new();

    for row in lines {
        let cols: Vec<&str> = row.split('\t').collect();
        assert_eq!(cols.len(), 20, "expected 20 columns: {row}");
        let case_name = cols[0];
        let model_num = cols[2];
        let zone = cols[3];
        let xa_hex: Vec<&str> = cols[19].split(',').collect();
        assert_eq!(xa_hex.len(), 8);

        let use_free_eos = case_name != "binary/wind_fed_bhhmxb";
        let controls = StarControls {
            zbase: 0.02,
            ..StarControls::default()
        };
        let mut model = basic_net_template(1, use_free_eos, controls);
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

    assert_eq!(checked, 8);
    assert!(
        failures.is_empty(),
        "binary stepdump AUX 0-ULP mismatches ({} failures):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}
