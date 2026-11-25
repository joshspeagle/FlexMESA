// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration tests for `#171` ADIPLS Package A3 (`AsteroAdipls` in
//! `flexmesa-astero::adipls_support`, `"adipls"` wiring in `sample_eval`, `#479`
//! uninitialized AMDL row preservation, `#480` `astero_def` flag shadowing,
//! loud failure checks, and optional `FLEXMESA_ADIPLS_CAPTURES` parity).

#![forbid(unsafe_code)]
#![allow(clippy::float_cmp, clippy::field_reassign_with_default)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use flexmesa_adipls::AdiplsError;
use flexmesa_const::{MSUN, RSUN, STANDARD_CGRAV};
use flexmesa_star::job_controls::{AtmOption, AtmTTauOpacity, AtmTTauRelation};
use flexmesa_star::model_io::read_mod_str_with_dir;
use flexmesa_star::{
    eval_photosphere_info, load_zams_model, set_hydro_vars, InlistBundle,
    InlistLoadOptions, StarControls, StarError, StarJobControls, StarModel,
};

use flexmesa_astero::adipls_support::AsteroAdipls;
use flexmesa_astero::inlist_astero::{AsteroParamSet, AsteroSearchControls};
use flexmesa_astero::inlist_parser::parse_astero_search_controls_with_options;
use flexmesa_astero::sample_eval::eval_model_astero_sample;

const ADIPLS_PRUNED_IN: &str = "\
2  'amdl.mesa'
9  'ttt.adipls.prt'
11 'agsm.mesa'
15 'ttt.adipls.ssm'
16 'ttt.adipls.fsm'
-1 ''
mod.osc.cst.int.out
,,,,,,,
,,1,,,,,,,,,,,,,,,,,,,,
   0,3,0,1,,,,,,,
    1,  220,   ,    1,10,,,,,,,,
,2,100,1000,,,,,,,,,,,,
,,,,,,,,,,,,,,,,
6.67430e-8
,,0,,
1,,,,,,,,,,,,,,,
5,0,0.99,,,,,,,,,,,,,,
,,,,,-1,,,,,,,,,,,,,,
9,10,,20,1,,,,,,,,
1,,,,,,,,,,,,,,,
0,,,,6,,,,,,
,,,,,-5000,100,,,,,,,,,,,,,,,,,,,,,,,,
";

const REDISTRB_PRUNED_IN: &str = "\
2001,,,
24,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
30,,,,,,,,,
,,,,,,,,,,,
";

static TEST_DIR_SEQ: AtomicUsize = AtomicUsize::new(1);

struct TempCaseDir {
    path: PathBuf,
}

impl TempCaseDir {
    fn new(tag: &str) -> Self {
        let id = TEST_DIR_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "flexmesa_adipls_a3_{tag}_{}_{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create temp case dir");
        Self { path }
    }

    fn write_file(&self, rel: &str, body: &str) {
        fs::write(self.path.join(rel), body).expect("write fixture file");
    }
}

impl Drop for TempCaseDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn hydrated_solar_zams() -> StarModel {
    let controls = StarControls {
        zbase: 0.02,
        calculate_brunt_n2: true,
        atm_option: AtmOption::TTau,
        atm_t_tau_relation: AtmTTauRelation::Eddington,
        atm_t_tau_opacity: AtmTTauOpacity::Fixed,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(controls, StarJobControls::default()).expect("zams");
    set_hydro_vars(&mut model, false, true).expect("set_hydro_vars");
    model
}

#[test]
fn test_adipls_init_validation_and_redistrb_startup_parsing() {
    let dir = TempCaseDir::new("init");

    // Missing adipls.c.pruned.in fails loudly.
    let mut ctrl = AsteroSearchControls::default();
    ctrl.oscillation_code = String::from("adipls");
    let err_missing = AsteroAdipls::init(&dir.path, &ctrl)
        .expect_err("missing adipls.c.pruned.in must fail");
    assert!(matches!(err_missing, StarError::IoError(_)));

    // Wrong oscillation_code fails loudly even when file exists.
    dir.write_file("adipls.c.pruned.in", ADIPLS_PRUNED_IN);
    let mut bad_code = ctrl.clone();
    bad_code.oscillation_code = String::from("gyre");
    assert!(matches!(
        AsteroAdipls::init(&dir.path, &bad_code),
        Err(StarError::InvalidInput(_))
    ));

    // Standard init succeeds and parses adipls.c.pruned.in without redistrb.
    let driver = AsteroAdipls::init(&dir.path, &ctrl).expect("init adipls");
    assert!((driver.input().cst.cgrav - 6.67430e-8).abs() <= 1.0e-20);
    assert!(driver.redistrb_input().is_none());

    // Startup with do_redistribute_mesh = true requires redistrb.c.pruned.in.
    assert!(matches!(
        AsteroAdipls::init_with_redistribute(&dir.path, &ctrl, true),
        Err(StarError::IoError(_))
    ));
    dir.write_file("redistrb.c.pruned.in", REDISTRB_PRUNED_IN);
    let driver_redist =
        AsteroAdipls::init_with_redistribute(&dir.path, &ctrl, true)
            .expect("init with redistrb.c.pruned.in");
    assert!(driver_redist.redistrb_input().is_some());

    // AsteroSearchControls defaults `do_redistribute_mesh = true`.
    ctrl = AsteroSearchControls::default();
    assert!(ctrl.do_redistribute_mesh);
}

#[test]
fn test_adipls_store_model_479_unset_rows_480_shadowing_and_redist_fail() {
    let dir = TempCaseDir::new("store");
    dir.write_file("adipls.c.pruned.in", ADIPLS_PRUNED_IN);

    let mut ctrl = AsteroSearchControls::default();
    ctrl.oscillation_code = String::from("adipls");
    ctrl.add_center_point = true;
    ctrl.keep_surface_point = false;

    let mut driver = AsteroAdipls::init(&dir.path, &ctrl).expect("init");
    let model = hydrated_solar_zams();

    // Runtime redist_mesh = true fails explicitly.
    let redist_err = driver
        .store_model(&model, false, true)
        .expect_err("runtime redist_mesh = true must fail");
    assert!(matches!(redist_err, StarError::InvalidInput(_)));

    // Store model honoring #479 (rows 6..=9 unset).
    let st_default = driver
        .store_model(&model, false, false)
        .expect("store_model")
        .clone();
    assert_eq!(st_default.nn, model.nz);
    for row in [1usize, 2, 3, 4, 5, 10] {
        assert!(
            st_default.aa.row(row).is_ok(),
            "expected AMDL row {row} to be populated"
        );
    }
    for row in 6usize..=9 {
        assert_eq!(
            st_default.aa.row(row),
            Err(AdiplsError::UnsetAmdlRow { row }),
            "expected #479 unset error on row {row}"
        );
    }

    // Store with contradictory caller flags -> #480 shadows with ctrl defaults.
    let st_shadow = driver
        .store_model_with_caller_flags(&model, false, true, false, false)
        .expect("store_model_with_caller_flags")
        .clone();
    assert_eq!(st_shadow.nn, st_default.nn);
    assert_eq!(st_shadow.data, st_default.data);
    assert_eq!(st_shadow.x, st_default.x);
    for row in [1usize, 2, 3, 4, 5, 10] {
        assert_eq!(
            st_shadow.aa.row(row).unwrap(),
            st_default.aa.row(row).unwrap()
        );
    }
}

#[test]
fn test_adipls_get_one_el_info_reuses_store_and_sorts_modes() {
    let dir = TempCaseDir::new("one_el");
    dir.write_file("adipls.c.pruned.in", ADIPLS_PRUNED_IN);

    let mut ctrl = AsteroSearchControls::default();
    ctrl.oscillation_code = String::from("adipls");
    ctrl.add_atmosphere = false;
    ctrl.add_center_point = true;
    ctrl.keep_surface_point = false;

    let mut driver = AsteroAdipls::init(&dir.path, &ctrl).expect("init");
    let model = hydrated_solar_zams();

    // store_for_adipls = false before first store must fail loudly.
    let err_no_store = driver
        .get_one_el_info(&model, 0, 1000.0, 3500.0, 120, 135.0, false)
        .expect_err("must require stored model");
    assert!(matches!(err_no_store, StarError::InvalidInput(_)));

    // Scan l=0 with store_for_adipls = true.
    let modes_l0 = driver
        .get_one_el_info(&model, 0, 1000.0, 3500.0, 120, 135.0, true)
        .expect("get_one_el_info l=0");
    assert!(
        modes_l0.len() >= 10,
        "expected >= 10 radial modes, got {}",
        modes_l0.len()
    );
    for (idx, m) in modes_l0.iter().enumerate() {
        assert_eq!(m.el, 0);
        assert_eq!(m.em, 0);
        assert_eq!(m.growth_rate, 0.0);
        assert!(m.inertia > 0.0 && m.inertia.is_finite());
        assert!(m.cyclic_freq >= 990.0 && m.cyclic_freq <= 3510.0);
        if idx > 0 {
            assert!(
                m.cyclic_freq >= modes_l0[idx - 1].cyclic_freq,
                "modes not sorted ascending"
            );
        }
    }

    // Scan l=1 reusing cached store (store_for_adipls = false).
    let modes_l1 = driver
        .get_one_el_info(&model, 1, 1000.0, 3500.0, 120, 135.0, false)
        .expect("get_one_el_info l=1");
    assert_ne!(modes_l1.len(), 0);
    assert!(modes_l1.iter().all(|m| m.el == 1));

    // Direct adipls_support path (`sig_fac_adipls_support`) with atmosphere.
    let phot = eval_photosphere_info(&model);
    let r_cm = RSUN * phot.r_rsun;
    let g_cgs = STANDARD_CGRAV;
    let m_g = model.m_grav[0];
    let modes_l0_atm = driver
        .do_adipls_get_one_el_info(
            &model, 0, 1000.0, 3500.0, 120, r_cm, g_cgs, m_g, 135.0, true,
            false, true, 0, 1.0,
        )
        .expect("do_adipls_get_one_el_info with atm");
    assert!(modes_l0_atm.len() >= 10);

    // Direct call with do_redistribute_mesh = true must fail even on reuse.
    assert!(matches!(
        driver.do_adipls_get_one_el_info(
            &model, 0, 1000.0, 3500.0, 120, r_cm, g_cgs, m_g, 135.0, false,
            true, false, 0, 1.0,
        ),
        Err(StarError::InvalidInput(_))
    ));
}

#[test]
fn test_sample_eval_adipls_branch_without_fake_fallback() {
    let dir = TempCaseDir::new("sample_eval");
    let model = hydrated_solar_zams();

    dir.write_file("adipls.c.pruned.in", ADIPLS_PRUNED_IN);
    let mut ctrl = AsteroSearchControls::default();
    ctrl.oscillation_code = String::from("adipls");
    ctrl.add_atmosphere = false;
    ctrl.add_center_point = true;
    ctrl.keep_surface_point = false;
    ctrl.do_redistribute_mesh = false;
    ctrl.nu_lower_factor = 0.8;
    ctrl.nu_upper_factor = 1.2;
    ctrl.weights.chi2_seismo_fraction = 1.0;

    let mut driver = AsteroAdipls::init(&dir.path, &ctrl).expect("init");
    let raw_l0 = driver
        .get_one_el_info(&model, 0, 1500.0, 3500.0, 120, 135.0, true)
        .expect("discover l=0");
    let raw_l1 = driver
        .get_one_el_info(&model, 1, 1500.0, 3500.0, 120, 135.0, false)
        .expect("discover l=1");
    assert!(raw_l0.len() >= 3 && raw_l1.len() >= 2);

    ctrl.nl = [3, 2, 0, 0];
    ctrl.freq_target[0] =
        raw_l0[..3].iter().map(|m| m.cyclic_freq + 0.5).collect();
    ctrl.freq_sigma[0] = vec![1.0; 3];
    ctrl.freq_target[1] =
        raw_l1[..2].iter().map(|m| m.cyclic_freq - 0.4).collect();
    ctrl.freq_sigma[1] = vec![1.0; 2];

    let params = AsteroParamSet {
        initial_mass: model.m_grav[0] / MSUN,
        initial_y: model.controls.initial_y,
        initial_feh: 0.0,
        alpha: model.controls.mixing_length_alpha,
        f_ov: 0.0,
        my_param1: 0.0,
        my_param2: 0.0,
        my_param3: 0.0,
    };
    let sample = eval_model_astero_sample(
        &model, &ctrl, &[0.0; 10], params, 1, &dir.path,
    )
    .expect("eval_model_astero_sample with adipls");
    assert_eq!(sample.model_freqs[0].len(), 3);
    assert_eq!(sample.model_freqs[1].len(), 2);
    assert!(sample.chi2_seismo > 0.0 && sample.chi2_seismo.is_finite());

    // Missing adipls.c.pruned.in MUST error rather than falling back to BVP.
    let missing_dir = TempCaseDir::new("sample_missing");
    let err = eval_model_astero_sample(
        &model,
        &ctrl,
        &[0.0; 10],
        params,
        1,
        &missing_dir.path,
    )
    .expect_err("missing adipls.c.pruned.in must fail loudly");
    assert!(matches!(err, StarError::IoError(_)));
}

#[derive(Debug)]
struct ModeRow {
    el: i32,
    order: i32,
    cyclic_freq_uhz: f64,
    inertia: f64,
}

fn parse_hex_f64(hex: &str) -> f64 {
    let bits = u64::from_str_radix(hex.trim(), 16)
        .unwrap_or_else(|e| panic!("bad hex f64 `{hex}`: {e}"));
    f64::from_bits(bits)
}

fn load_captured_modes_by_call(path: &Path) -> BTreeMap<usize, Vec<ModeRow>> {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut by_call: BTreeMap<usize, Vec<ModeRow>> = BTreeMap::new();
    for (idx, line) in text.lines().enumerate() {
        if idx == 0 || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        let call_id: usize = cols[0].parse().expect("call_id");
        let el: i32 = cols[3].parse().expect("el");
        let order: i32 = cols[4].parse().expect("order");
        let cyclic_freq_uhz = parse_hex_f64(cols[6]);
        let inertia = parse_hex_f64(cols[7]);
        by_call.entry(call_id).or_default().push(ModeRow {
            el,
            order,
            cyclic_freq_uhz,
            inertia,
        });
    }
    by_call
}

fn load_and_hydrate_mod(case_dir: &Path, mod_path: &Path) -> StarModel {
    let bundle = InlistBundle::from_file_in_case_dir_with_options(
        case_dir,
        "inlist",
        InlistLoadOptions {
            allow_unsupported_controls: true,
        },
    )
    .expect("load star inlist");
    let text = fs::read_to_string(mod_path).expect("read captured .mod");
    let mut controls = bundle.controls;
    controls.calculate_brunt_n2 = true;
    if controls.initial_y < 0.0 {
        controls.initial_y = (0.24 + 2.0 * controls.initial_z).clamp(0.0, 1.0);
    }
    let mut model =
        read_mod_str_with_dir(&text, controls, bundle.job, Some(case_dir))
            .expect("read_mod_str_with_dir");
    model.eos_config = bundle.eos_config;
    model.kap_config = bundle.kap_config;
    set_hydro_vars(&mut model, false, true).expect("set_hydro_vars");
    model
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn verify_surface_effects_captures(root: &Path) {
    let case_dir = root.join("surface_effects");
    let model = load_and_hydrate_mod(
        &case_dir,
        &case_dir.join("pulse_tap/models/adipls_eval_000001_step_0000954.mod"),
    );
    let mut ctrl = parse_astero_search_controls_with_options(
        &case_dir,
        InlistLoadOptions {
            allow_unsupported_controls: true,
        },
    )
    .expect("parse astero");
    ctrl.weights.chi2_seismo_fraction = 1.0;

    let cap_modes =
        load_captured_modes_by_call(&case_dir.join("a0tap/modes.tsv"));
    let mut driver = AsteroAdipls::init(&case_dir, &ctrl).expect("init");
    for (l, call_id) in [(0usize, 1usize), (1, 2), (2, 3), (3, 4)] {
        let f_min =
            *ctrl.freq_target[l].first().unwrap() * ctrl.nu_lower_factor;
        let f_max = *ctrl.freq_target[l].last().unwrap() * ctrl.nu_upper_factor;
        let iscan = (ctrl.iscan_factor[l] * (ctrl.nl[l] as f64)) as i32;
        let dnu_arg = if l == 0 { 0.0 } else { 136.543_143_940_618_73 };
        let got = driver
            .get_one_el_info(&model, l, f_min, f_max, iscan, dnu_arg, l == 0)
            .expect("surface_effects get_one_el_info");
        let cap = &cap_modes[&call_id];
        assert_eq!(
            got.len(),
            cap.len(),
            "surface_effects l={l} mode count mismatch"
        );
        for (g, c) in got.iter().zip(cap.iter()) {
            assert_eq!(g.el, c.el);
            assert_eq!(g.order, c.order);
            let rel_nu =
                (g.cyclic_freq - c.cyclic_freq_uhz).abs() / c.cyclic_freq_uhz;
            assert!(
                rel_nu <= 1.0e-3,
                "surface_effects l={l} n={}: got={} cap={} rel={rel_nu:.3e}",
                g.order,
                g.cyclic_freq,
                c.cyclic_freq_uhz
            );
            let rel_e = (g.inertia - c.inertia).abs() / c.inertia;
            assert!(
                rel_e <= 2.0e-2,
                "surface_effects l={l} inertia n={}: got={} cap={} rel={rel_e:.3e}",
                g.order,
                g.inertia,
                c.inertia
            );
        }
    }

    let initial_y = if model.controls.initial_y < 0.0 {
        0.28
    } else {
        model.controls.initial_y
    };
    let sample = eval_model_astero_sample(
        &model,
        &ctrl,
        &[0.0; 10],
        AsteroParamSet {
            initial_mass: model.m_grav[0] / MSUN,
            initial_y,
            initial_feh: 0.0,
            alpha: model.controls.mixing_length_alpha,
            f_ov: 0.0,
            my_param1: 0.0,
            my_param2: 0.0,
            my_param3: 0.0,
        },
        1,
        &case_dir,
    )
    .expect("surface_effects eval_model_astero_sample");
    assert!(sample.chi2.is_finite());
    for l in 0..4 {
        assert_eq!(sample.model_freqs[l].len(), ctrl.nl[l]);
    }
}

fn verify_astero_adipls_captures(root: &Path) {
    let case_dir = root.join("astero_adipls");
    let model = load_and_hydrate_mod(
        &case_dir,
        &case_dir.join("pulse_tap/models/adipls_eval_000001_step_0000050.mod"),
    );
    let mut ctrl = AsteroSearchControls::default();
    ctrl.oscillation_code = String::from("adipls");
    ctrl.add_center_point = true;
    ctrl.keep_surface_point = false;
    ctrl.add_atmosphere = true;
    let mut driver =
        AsteroAdipls::init_with_redistribute(&case_dir, &ctrl, true)
            .expect("astero_adipls init_with_redistribute");
    assert!(driver.redistrb_input().is_some());

    let r_cm = RSUN * eval_photosphere_info(&model).r_rsun;
    let g_cgs = model.cgrav[0];
    let m_g = model.m_grav[0];
    let got = driver
        .do_adipls_get_one_el_info(
            &model, 0, 50.0, 1000.0, 200, r_cm, g_cgs, m_g, 0.0, true, false,
            true, 0, 1.0,
        )
        .expect("astero_adipls do_adipls_get_one_el_info");
    let cap_modes =
        load_captured_modes_by_call(&case_dir.join("a0tap/modes.tsv"));
    let cap = &cap_modes[&1];
    assert_ne!(got.len(), 0);
    assert_eq!(got.len(), cap.len(), "astero_adipls call 1 mode count");
    for (g, c) in got.iter().zip(cap.iter()) {
        assert_eq!(g.el, c.el);
        assert_eq!(g.order, c.order);
        let rel = (g.cyclic_freq - c.cyclic_freq_uhz).abs() / c.cyclic_freq_uhz;
        assert!(
            rel <= 2.0e-3,
            "astero_adipls n={}: got={} cap={} rel={rel:.3e}",
            g.order,
            g.cyclic_freq,
            c.cyclic_freq_uhz
        );
    }
}

#[test]
fn test_adipls_a3_captures_parity_pipeline() {
    let raw = std::env::var("FLEXMESA_ADIPLS_CAPTURES").ok();
    let path_str = raw.as_deref().map_or("", str::trim);
    if path_str.is_empty() {
        eprintln!(
            "INFO: FLEXMESA_ADIPLS_CAPTURES unset; external A3 capture parity not run"
        );
        return;
    }
    let root = PathBuf::from(path_str);
    assert!(
        root.is_dir(),
        "FLEXMESA_ADIPLS_CAPTURES path `{}` is not a directory",
        root.display()
    );

    verify_surface_effects_captures(&root);
    verify_astero_adipls_captures(&root);
}
