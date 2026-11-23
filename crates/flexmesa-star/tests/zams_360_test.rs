// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration tests for GitHub issue `#360`: ZAMS exact-mass interpolation
//! onto `m1`'s grid (`alfa = 0`), edge mass clamping, and MESA `26.4.1`
//! first-model `nz` and initial profile comparison (`star/private/init_model.f90`).

#![forbid(unsafe_code)]
#![allow(clippy::float_cmp)]

use std::path::{Path, PathBuf};

use flexmesa_const::{M_SUN, SECYER};
use flexmesa_star::{
    extract_history_row, interp_q, load_zams_model,
    load_zams_model_from_catalog, read_mesa_log_file, set_hydro_vars,
    star_evolve_step, yrs_for_init_timestep, StarControls, StarError,
    StarJobControls, ZamsCatalog, ZamsEntry,
};
use flexmesa_utils::{
    require_mesa_test_input_path, require_zero_skips_enabled,
    resolve_flexmesa_data_dir,
};

fn resolve_mesa_histories_dir() -> PathBuf {
    if let Some(val) = std::env::var_os("FLEXMESA_HISTORIES_DIR") {
        if !val.is_empty() {
            return PathBuf::from(val);
        }
    }
    let data_dir = resolve_flexmesa_data_dir();
    if let Some(parent) = data_dir.parent() {
        let cand = parent.join("mesa_histories");
        if cand.is_dir() {
            return cand;
        }
    }
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        if !xdg.is_empty() {
            return PathBuf::from(xdg).join("flexmesa").join("mesa_histories");
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        if !home.is_empty() {
            return PathBuf::from(home)
                .join(".cache")
                .join("flexmesa")
                .join("mesa_histories");
        }
    }
    PathBuf::from(".flexmesa_histories")
}

fn require_mesa_history_path(rel_path: &str) -> Option<PathBuf> {
    let path = resolve_mesa_histories_dir().join(rel_path);
    if path.exists() {
        return Some(path);
    }
    assert!(
        !require_zero_skips_enabled(),
        "missing required MESA 26.4.1 reference history `{}` (run \
         `flexmesa data fetch --include-test-inputs` or set \
         FLEXMESA_REQUIRE_ZERO_SKIPS=0)",
        path.display()
    );
    None
}

fn verify_exact_upper_bracket_interp(
    cat: &ZamsCatalog,
    idx1: usize,
    idx2: usize,
    mass_msun: f64,
    label: &str,
) {
    let e1: &ZamsEntry = &cat.models[idx1];
    let e2: &ZamsEntry = &cat.models[idx2];
    assert_eq!(e2.index_m_msun, mass_msun, "{label} index mass mismatch");
    assert_ne!(
        e1.nz, e2.nz,
        "{label} test requires distinct nz between m1 and m2"
    );

    let controls = StarControls {
        initial_mass: mass_msun,
        initial_z: 0.02,
        zbase: 0.02,
        ..StarControls::default()
    };
    let model =
        load_zams_model_from_catalog(cat, controls, StarJobControls::default())
            .expect("load exact catalog mass ZAMS model");

    assert_eq!(model.model_number, 0, "{label}: model_number must be 0");
    assert_eq!(model.star_age, 0.0, "{label}: star_age must be 0");
    assert_eq!(
        model.nz, e1.nz,
        "{label}: expected m1's nz={}, got {} (must not copy m2's nz={})",
        e1.nz, model.nz, e2.nz
    );
    assert!(
        (model.mstar - mass_msun * M_SUN).abs() <= 1.0e-12 * model.mstar,
        "{label}: mstar mismatch"
    );
    assert_eq!(model.controls.initial_mass, mass_msun);

    // Verify `alfa == 0` in `init_model.f90:444-446` and bit-exact profile match
    // against `interp_q` of `m2` onto `m1`'s `q` grid.
    let ln_mass_1 = flexmesa_math::log(e1.index_m_msun);
    let ln_mass_2 = flexmesa_math::log(e2.index_m_msun);
    let alfa =
        (flexmesa_math::log(mass_msun) - ln_mass_2) / (ln_mass_1 - ln_mass_2);
    let beta = 1.0 - alfa;
    assert_eq!(alfa.abs(), 0.0, "{label}: alfa must be zero");
    assert_eq!(beta, 1.0, "{label}: beta must be one");

    let q_density = interp_q(&e2.dq, &e2.lnd, &e1.dq).expect("interp_q lnd");
    let q_temp = interp_q(&e2.dq, &e2.ln_t, &e1.dq).expect("interp_q lnT");
    let q_radius = interp_q(&e2.dq, &e2.ln_r, &e1.dq).expect("interp_q lnR");
    let q_lum = interp_q(&e2.dq, &e2.lum, &e1.dq).expect("interp_q lum");

    let mut max_dq_rel_diff = 0.0_f64;
    for k in 0..model.nz {
        let dq_rel = (model.dq[k] - e1.dq[k]).abs() / e1.dq[k];
        if dq_rel > max_dq_rel_diff {
            max_dq_rel_diff = dq_rel;
        }
        assert_eq!(
            model.lnd[k].to_bits(),
            (alfa * e1.lnd[k] + beta * q_density[k]).to_bits(),
            "{label}: lnd bit mismatch at zone {k}"
        );
        assert_eq!(
            model.ln_t[k].to_bits(),
            (alfa * e1.ln_t[k] + beta * q_temp[k]).to_bits(),
            "{label}: ln_t bit mismatch at zone {k}"
        );
        assert_eq!(
            model.ln_r[k].to_bits(),
            (alfa * e1.ln_r[k] + beta * q_radius[k]).to_bits(),
            "{label}: ln_r bit mismatch at zone {k}"
        );
        assert_eq!(
            model.lum[k].to_bits(),
            (alfa * e1.lum[k] + beta * q_lum[k]).to_bits(),
            "{label}: lum bit mismatch at zone {k}"
        );
    }
    assert!(
        max_dq_rel_diff <= 1.0e-15,
        "{label}: dq must match m1's dq grid (max_dq_rel_diff={max_dq_rel_diff:.3e})"
    );

    // Ensure profile values come from `m2` interpolated onto `m1`'s grid rather
    // than copying `m1`'s profile arrays.
    assert!(
        (model.lum[0] - e1.lum[0]).abs() > 0.1 * e1.lum[0].abs(),
        "{label}: surface luminosity must come from m2, not m1"
    );
    assert!(
        (model.ln_r[0] - e1.ln_r[0]).abs() > 0.05,
        "{label}: surface radius must come from m2, not m1"
    );

    // Verify species mass fractions match `interp_q` of `m2` onto `m1`'s grid bit-for-bit.
    let sp = model.species;
    for j in 0..sp {
        let x2_col: Vec<f64> = (0..e2.nz).map(|k| e2.xa[k * sp + j]).collect();
        let x2_interp = interp_q(&e2.dq, &x2_col, &e1.dq).expect("interp_q xa");
        for k in 0..model.nz {
            let expected = alfa * e1.xa[k * sp + j] + beta * x2_interp[k];
            let actual = model.xa[k * sp + j];
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{label}: xa bit mismatch at zone {k}, species {j}: actual={actual}, expected={expected}"
            );
        }
    }
}

#[test]
fn test_zams_360_exact_interior_catalog_masses_interpolate_with_alfa_zero() {
    let builtin = ZamsCatalog::builtin();
    // 1.0 Msun: m1 = 0.79432823 (index 8, nz = 3813), m2 = 1.00000000 (index 9, nz = 3817)
    verify_exact_upper_bracket_interp(builtin, 8, 9, 1.0, "builtin 1.0 Msun");
    // 10.0 Msun: m1 = 7.94328235 (index 18, nz = 2477), m2 = 10.00000000 (index 19, nz = 2460)
    verify_exact_upper_bracket_interp(
        builtin,
        18,
        19,
        10.0,
        "builtin 10.0 Msun",
    );
    // 100.0 Msun: m1 = 79.43282347 (index 28, nz = 2371), m2 = 100.00000000 (index 29, nz = 2018)
    verify_exact_upper_bracket_interp(
        builtin,
        28,
        29,
        100.0,
        "builtin 100.0 Msun",
    );
    // Exact max index entry 316.22776602: m1 = 251.18864315 (index 33, nz = 2009),
    // m2 = 316.22776602 (index 34, nz = 2022)
    verify_exact_upper_bracket_interp(
        builtin,
        33,
        34,
        316.227_766_02,
        "builtin max 316.22776602 Msun",
    );

    if let Some(ascii_path) = require_mesa_test_input_path(
        "data/star_data/zams_models/zams_z2m2_y28.data",
    ) {
        let text = std::fs::read_to_string(&ascii_path)
            .expect("read upstream zams_z2m2_y28.data");
        let ascii_cat = ZamsCatalog::parse_ascii(&text)
            .expect("parse upstream zams_z2m2_y28.data");
        verify_exact_upper_bracket_interp(
            &ascii_cat,
            8,
            9,
            1.0,
            "ASCII 1.0 Msun",
        );
        verify_exact_upper_bracket_interp(
            &ascii_cat,
            18,
            19,
            10.0,
            "ASCII 10.0 Msun",
        );
    }
}

#[test]
fn test_zams_360_exact_min_and_out_of_range_masses_clamp_per_upstream() {
    let cat = ZamsCatalog::builtin();
    let min_entry = &cat.models[0];
    let max_entry = cat.models.last().expect("non-empty catalog");
    assert_eq!(min_entry.index_m_msun, 0.125_892_54);
    assert_eq!(min_entry.nz, 3777);
    assert_eq!(max_entry.index_m_msun, 316.227_766_02);
    assert_eq!(max_entry.nz, 2022);

    // Exact minimum index mass (`i = 1` in Fortran / `0` in Rust) and masses
    // below minimum clamp to entry 0 without interpolation (`init_model.f90:221-228`).
    for &mass_msun in &[0.01, 0.08, 0.10, 0.125_892_54] {
        let model = load_zams_model(
            StarControls {
                initial_mass: mass_msun,
                initial_z: 0.02,
                zbase: 0.02,
                ..StarControls::default()
            },
            StarJobControls::default(),
        )
        .expect("below-or-at-min mass should clamp cleanly");

        assert_eq!(model.nz, min_entry.nz);
        assert_eq!(model.controls.initial_mass, min_entry.index_m_msun);
        assert_eq!(model.mstar, min_entry.index_m_msun * M_SUN);
        assert_eq!(model.xmstar, model.mstar);
        assert_eq!(
            model.dt,
            yrs_for_init_timestep(min_entry.index_m_msun) * SECYER
        );
        assert_eq!(model.lnd, min_entry.lnd);
        assert_eq!(model.ln_t, min_entry.ln_t);
        assert_eq!(model.ln_r, min_entry.ln_r);
        assert_eq!(model.lum, min_entry.lum);
    }

    // Masses above catalog maximum clamp to the last entry (`init_model.f90:212-220`).
    for &mass_msun in &[316.3, 350.0, 400.0, 1000.0] {
        let model = load_zams_model(
            StarControls {
                initial_mass: mass_msun,
                initial_z: 0.02,
                zbase: 0.02,
                ..StarControls::default()
            },
            StarJobControls::default(),
        )
        .expect("above-max mass should clamp cleanly");

        assert_eq!(model.nz, max_entry.nz);
        assert_eq!(model.controls.initial_mass, max_entry.index_m_msun);
        assert_eq!(model.mstar, max_entry.index_m_msun * M_SUN);
        assert_eq!(model.xmstar, model.mstar);
        assert_eq!(
            model.dt,
            yrs_for_init_timestep(max_entry.index_m_msun) * SECYER
        );
        assert_eq!(model.lnd, max_entry.lnd);
        assert_eq!(model.ln_t, max_entry.ln_t);
        assert_eq!(model.ln_r, max_entry.ln_r);
        assert_eq!(model.lum, max_entry.lum);
    }

    // Non-positive and non-finite masses fail loudly (`init_model.f90:83-86`, `relax.f90:157`).
    for &bad_mass in &[0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let err = load_zams_model(
            StarControls {
                initial_mass: bad_mass,
                initial_z: 0.02,
                zbase: 0.02,
                ..StarControls::default()
            },
            StarJobControls::default(),
        )
        .expect_err("non-positive or non-finite mass must error");
        assert!(matches!(err, StarError::InvalidInput(_)));
    }
}

#[test]
fn test_zams_360_1msun_matches_mesa_26_4_1_reference_first_model_nz_and_state()
{
    let mut model = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            initial_z: 0.02,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 1.0 Msun ZAMS");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    let row0 = extract_history_row(&model);
    assert_eq!(row0.num_zones, 3813);

    // Repo-checked MESA 26.4.1 reference histories for 1.0 Msun ZAMS cases
    // where model 1 precedes any remesh: both record `num_zones = 3813` (not 3817).
    let nz_expected_f64 =
        f64::from(u32::try_from(row0.num_zones).expect("nz fits u32"));
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for rel in [
        "benchmarks/history_harness/reference_histories/star/T_tau_gradr.data",
        "benchmarks/history_harness/reference_histories/star/check_pulse_atm.data",
    ] {
        let tbl = read_mesa_log_file(&repo_root.join(rel))
            .expect("read repo MESA 26.4.1 reference history");
        let model_nums = tbl.column("model_number").expect("model_number");
        let num_zones = tbl.column("num_zones").expect("num_zones");
        let star_mass = tbl.column("star_mass").expect("star_mass");
        assert_eq!(model_nums[0], 1.0, "{rel}: expected model 1 first");
        assert!(
            (star_mass[0] - 1.0).abs() <= 1.0e-12,
            "{rel}: expected 1.0 Msun"
        );
        assert_eq!(
            num_zones[0], 3813.0,
            "{rel}: MESA 26.4.1 first-model num_zones must be 3813"
        );
        assert_eq!(nz_expected_f64, num_zones[0]);
    }

    // Data-cache MESA 26.4.1 binary star-1 histories (`evolve_both_stars/LOGS1`
    // and `jdot_ml_check/LOGS1`): verify model-1 `num_zones = 3813` and close
    // initial-state agreement with our 1.0 Msun ZAMS Model 0.
    for rel in [
        "runs/binary/evolve_both_stars/LOGS1/history.data",
        "runs/binary/jdot_ml_check/LOGS1/history.data",
    ] {
        if let Some(hist_path) = require_mesa_history_path(rel) {
            let tbl = read_mesa_log_file(&hist_path)
                .expect("read cached MESA 26.4.1 binary LOGS1/history.data");
            let model_nums = tbl.column("model_number").expect("model_number");
            let num_zones = tbl.column("num_zones").expect("num_zones");
            let star_mass = tbl.column("star_mass").expect("star_mass");
            let log_teff = tbl.column("log_Teff").expect("log_Teff");
            let log_l = tbl.column("log_L").expect("log_L");
            let log_r = tbl.column("log_R").expect("log_R");
            let center_h1 = tbl.column("center_h1").expect("center_h1");
            let mass_conv_core =
                tbl.column("mass_conv_core").expect("mass_conv_core");

            assert_eq!(model_nums[0], 1.0);
            assert!((star_mass[0] - 1.0).abs() <= 1.0e-12);
            assert_eq!(num_zones[0], 3813.0);
            assert_eq!(nz_expected_f64, num_zones[0]);

            // Model 0 vs Model 1 (at dt = 1e5 yr) agrees closely in surface and
            // central structure.
            assert!((row0.log_teff - log_teff[0]).abs() < 2.0e-3);
            assert!((row0.log_l - log_l[0]).abs() < 1.0e-2);
            assert!((row0.log_r - log_r[0]).abs() < 3.0e-3);
            assert!((row0.center_h1 - center_h1[0]).abs() < 1.0e-4);
            assert!((row0.mass_conv_core - mass_conv_core[0]).abs() < 5.0e-3);
        }
    }

    // Evolve 1 step (`dt = 1e5 yr`, `varcontrol_target = 1e-3`) matching
    // `binary/evolve_both_stars` Star 1 startup and compare Model 1 `num_zones`
    // and summary scalars directly against MESA 26.4.1 Model 1.
    let mut step1_model = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            initial_z: 0.02,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 1.0 Msun ZAMS for step 1");
    step1_model.controls.varcontrol_target = 1.0e-3;
    set_hydro_vars(&mut step1_model, false, false).expect("set_hydro_vars");
    step1_model.save_start_values().expect("save_start_values");
    step1_model.save_old_values();
    let step_rep = star_evolve_step(&mut step1_model, true, true, true)
        .expect("step 1 on 1.0 Msun ZAMS");
    assert!(step_rep.converged);
    let row1 = extract_history_row(&step1_model);
    assert_eq!(row1.model_number, 1);
    assert_eq!(row1.num_zones, 3813);

    if let Some(hist_path) = require_mesa_history_path(
        "runs/binary/evolve_both_stars/LOGS1/history.data",
    ) {
        let tbl = read_mesa_log_file(&hist_path)
            .expect("read evolve_both_stars LOGS1/history.data");
        let log_teff = tbl.column("log_Teff").expect("log_Teff");
        let log_l = tbl.column("log_L").expect("log_L");
        let log_r = tbl.column("log_R").expect("log_R");
        let center_h1 = tbl.column("center_h1").expect("center_h1");
        let mass_conv_core =
            tbl.column("mass_conv_core").expect("mass_conv_core");
        assert!((row1.log_teff - log_teff[0]).abs() < 1.0e-6);
        assert!((row1.log_l - log_l[0]).abs() < 1.0e-6);
        assert!((row1.log_r - log_r[0]).abs() < 1.0e-6);
        assert!((row1.center_h1 - center_h1[0]).abs() < 1.0e-9);
        assert!((row1.mass_conv_core - mass_conv_core[0]).abs() < 1.0e-6);
    }
}

#[test]
fn test_zams_360_1msun_model0_matches_mesa_26_4_1_hex_dump_at_0_ulp() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let tsv_path = repo_root.join(
        "benchmarks/fortran_mesa/zams_1p0_oracle/zams_model0_1p0Msun_26_4_1.tsv",
    );
    let text =
        std::fs::read_to_string(&tsv_path).expect("read 1.0 Msun hex oracle");

    let mut rows: Vec<(usize, u64, u64, u64, u64)> = Vec::new();
    let mut saw_header = false;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if !saw_header {
            assert_eq!(line, "k\tlnd\tlnT\tlnR\tL");
            saw_header = true;
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 5);
        rows.push((
            cols[0].parse::<usize>().expect("parse k"),
            u64::from_str_radix(cols[1], 16).expect("parse lnd hex"),
            u64::from_str_radix(cols[2], 16).expect("parse lnT hex"),
            u64::from_str_radix(cols[3], 16).expect("parse lnR hex"),
            u64::from_str_radix(cols[4], 16).expect("parse L hex"),
        ));
    }
    assert_eq!(rows.len(), 3813, "MESA 26.4.1 1.0 Msun ZAMS0 nz must be 3813");

    let model = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            initial_z: 0.02,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 1.0 Msun ZAMS");
    assert_eq!(model.nz, 3813);

    for (idx, &(k_1b, exp_lnd, exp_lnt, exp_lnr, exp_lum)) in
        rows.iter().enumerate()
    {
        assert_eq!(k_1b, idx + 1);
        assert_eq!(
            model.lnd[idx].to_bits(),
            exp_lnd,
            "zone {k_1b} lnd 0-ULP mismatch"
        );
        assert_eq!(
            model.ln_t[idx].to_bits(),
            exp_lnt,
            "zone {k_1b} lnT 0-ULP mismatch"
        );
        assert_eq!(
            model.ln_r[idx].to_bits(),
            exp_lnr,
            "zone {k_1b} lnR 0-ULP mismatch"
        );
        assert_eq!(
            model.lum[idx].to_bits(),
            exp_lum,
            "zone {k_1b} L 0-ULP mismatch"
        );
    }
}
