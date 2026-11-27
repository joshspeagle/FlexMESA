// SPDX-License-Identifier: GPL-3.0-or-later
//! Invariant and control-coverage tests for `&astero_search_controls`,
//! surface corrections, separation ratios, optimizer drivers, restarts,
//! ADIPLS/GYRE controls, and model/table output invariants (`#162` / `#147`).

#![allow(clippy::too_many_lines)]

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use flexmesa_star::controls::{
    AtmOption, AtmTTauOpacity, AtmTTauRelation, StarControls, StarError,
    StarJobControls,
};
use flexmesa_star::inlist::{InlistBundle, InlistLoadOptions};
use flexmesa_star::inlist_runner::StarLogsOptions;
use flexmesa_star::model::StarModel;
use flexmesa_star::star_lib::star_write_model;
use flexmesa_star::{load_zams_model, set_hydro_vars};

use flexmesa_astero::adipls_support::AsteroAdipls;
use flexmesa_astero::gyre_support::AsteroGyre;
use flexmesa_astero::inlist_astero::{
    AsteroSearchControls, ASTERO_SEARCH_CONTROLS_SUPPORTED,
    ASTERO_SEARCH_CONTROLS_UNUSED, MAX_CONSTRAINTS,
};
use flexmesa_astero::sample_eval::eval_spectro_and_delta_nu_chi2;
use flexmesa_astero::{
    estimate_l0_n_obs_with_numax, eval_frequency_ratios_with_trimming,
    eval_model_astero_sample, eval_r02_model_with_indices,
    eval_r02_obs_with_indices, eval_surf_coefs, get_max_sequence,
    parse_astero_search_controls, run_single_astero_track_with_case_hooks,
    run_star_astero_with_case_hooks, simplex_minimize_with_options,
    supported_astero_control_inventory, AsteroCaseHooks,
};

static TEST_DIR_SEQ: AtomicUsize = AtomicUsize::new(1);

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(tag: &str) -> Self {
        let id = TEST_DIR_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "flexmesa_controls_162_{tag}_{}_{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    fn write(&self, rel: &str, text: &str) {
        let dst = self.path.join(rel);
        if let Some(p) = dst.parent() {
            fs::create_dir_all(p).expect("create parent");
        }
        fs::write(&dst, text).expect("write fixture file");
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

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

const MINIMAL_GYRE_IN: &str = "\
&model
/
&mode
  l = 0
/
&osc
  outer_bound = 'VACUUM'
  variables_set = 'JCD'
/
&rot
/
&num
  diff_scheme = 'COLLOC_GL4'
/
&scan
  grid_type = 'LINEAR'
  freq_min = 1500
  freq_max = 3500
  freq_min_units = 'UHZ'
  freq_max_units = 'UHZ'
  n_freq = 40
/
&grid
/
";

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

fn setup_e2e_star_case(dir: &TempDirGuard, max_steps: usize) -> StarModel {
    let model = hydrated_solar_zams();
    star_write_model(&model, &dir.path.join("start.mod"))
        .expect("write start.mod");
    let inlist = format!(
        "&star_job
  load_saved_model = .true.
  load_model_filename = 'start.mod'
  pgstar_flag = .false.
/ ! end of star_job namelist

&controls
  max_model_number = {}
  max_years_for_timestep = 1.0d6
  calculate_brunt_n2 = .true.
  atm_option = 'T_tau'
  atm_T_tau_relation = 'Eddington'
  atm_T_tau_opacity = 'fixed'
  write_profiles_flag = .false.
/ ! end of controls namelist

&kap
  Zbase = 0.02d0
/ ! end of kap namelist
",
        model.model_number + max_steps
    );
    dir.write("inlist", &inlist);
    model
}

struct AlphaHook;

impl AsteroCaseHooks for AlphaHook {
    fn set_param(
        &mut self,
        bundle: &mut InlistBundle,
        _ctrl: &AsteroSearchControls,
        name: &str,
        val: f64,
    ) -> Result<(), StarError> {
        match name {
            "alpha" | "my_param1" => {
                bundle.controls.mixing_length_alpha = val;
            }
            "initial_mass" => {
                bundle.controls.initial_mass = val;
            }
            other => {
                return Err(StarError::InvalidInput(format!(
                    "unexpected param '{other}'"
                )));
            }
        }
        Ok(())
    }

    fn set_constraint_value(
        &mut self,
        model: &StarModel,
        _ctrl: &AsteroSearchControls,
        name: &str,
    ) -> Result<f64, StarError> {
        match name {
            "Teff" => Ok(model.teff),
            "alpha_obs" => Ok(model.controls.mixing_length_alpha),
            other => Err(StarError::InvalidInput(format!(
                "unexpected constraint '{other}'"
            ))),
        }
    }
}

#[test]
fn test_inventory_all_153_controls_and_roundtrip_save_controls() {
    let inv = supported_astero_control_inventory();
    assert_eq!(inv.supported.len(), 153);
    assert_eq!(ASTERO_SEARCH_CONTROLS_SUPPORTED.len(), 153);
    assert!(inv.unused.is_empty());
    assert_eq!(ASTERO_SEARCH_CONTROLS_UNUSED, [] as [&str; 0]);

    let tmp = TempDirGuard::new("inventory_roundtrip");
    tmp.write(
        "extra_sub.inlist",
        "&astero_search_controls
  gyre_non_ad = .true.
  y_frac_he3 = 1.5d-4
  trace_limits = .true.
  trace_time_in_oscillation_code = .true.
  save_next_best_at_higher_frequency = .true.
  save_next_best_at_lower_frequency = .true.
/
",
    );
    tmp.write(
        "inlist_astero_search_controls",
        "&astero_search_controls
  read_extra_astero_search_inlist(1) = .true.
  extra_astero_search_inlist_name(1) = 'extra_sub.inlist'
  search_type = 'simplex'
  simplex_alpha = 1.1d0
  simplex_beta = 0.45d0
  simplex_gamma = 1.9d0
  simplex_delta = 0.55d0
  restart_simplex_from_file = .false.
  restart_scan_grid_from_file = .false.
  astero_results_dbl_format = '(e24.15)'
  astero_results_int_format = '(i12)'
  astero_results_txt_format = '(a20)'
  surf_coef1_name = 'coef_a'
  surf_coef2_name = 'coef_b'
  write_best_model_data_for_each_sample = .false.
  num_digits = 5
  sample_results_prefix = 'best_s_'
  sample_results_postfix = '.txt'
  model_num_digits = 6
  write_fgong_for_each_model = .true.
  fgong_prefix = 'fg_'
  fgong_postfix = '.fgong'
  write_fgong_for_best_model = .true.
  best_model_fgong_filename = 'best.fgong'
  write_gyre_for_each_model = .true.
  gyre_prefix = 'gy_'
  gyre_postfix = '.gyre'
  max_num_gyre_points = 512
  write_gyre_for_best_model = .true.
  best_model_gyre_filename = 'best.gyre'
  write_profile_for_best_model = .true.
  best_model_profile_filename = 'best.prof'
  save_model_for_best_model = .true.
  best_model_save_model_filename = 'best.mod'
  save_info_for_last_model = .true.
  last_model_save_info_filename = 'last_info.txt'
  save_controls = .true.
  save_controls_filename = 'dumped_controls.txt'
  save_mode_model_number = 7
  save_mode_filename = 'mode7.dat'
  el_to_save = 1
  order_to_save = -2
  adipls_irotkr = 1
  adipls_nprtkr = 2
  adipls_igm1kr = 3
  adipls_npgmkr = 4
  trace_chi2_seismo_delta_nu_info = .true.
  trace_chi2_seismo_nu_max_info = .true.
  trace_chi2_seismo_ratios_info = .true.
  trace_chi2_seismo_frequencies_info = .true.
  trace_chi2_spectro_info = .true.
  nl(0) = 0
  nl(1) = 0
  nl(2) = 0
  nl(3) = 0
/
",
    );

    let ctrl = parse_astero_search_controls(&tmp.path)
        .expect("parse full newly implemented control groups");
    assert!(ctrl.gyre_non_ad);
    assert!((ctrl.y_frac_he3 - 1.5e-4).abs() <= 1.0e-18);
    assert!(ctrl.trace_limits);
    assert!(ctrl.trace_time_in_oscillation_code);
    assert!(ctrl.save_next_best_at_higher_frequency);
    assert!(ctrl.save_next_best_at_lower_frequency);
    assert!((ctrl.simplex_alpha - 1.1).abs() <= 1.0e-15);
    assert!((ctrl.simplex_beta - 0.45).abs() <= 1.0e-15);
    assert!((ctrl.simplex_gamma - 1.9).abs() <= 1.0e-15);
    assert!((ctrl.simplex_delta - 0.55).abs() <= 1.0e-15);
    assert_eq!(ctrl.astero_results_dbl_format, "(e24.15)");
    assert_eq!(ctrl.astero_results_int_format, "(i12)");
    assert_eq!(ctrl.astero_results_txt_format, "(a20)");
    assert_eq!(ctrl.surf_coef1_name, "coef_a");
    assert_eq!(ctrl.surf_coef2_name, "coef_b");
    assert!(!ctrl.write_best_model_data_for_each_sample);
    assert_eq!(ctrl.num_digits, 5);
    assert_eq!(ctrl.sample_results_prefix, "best_s_");
    assert_eq!(ctrl.sample_results_postfix, ".txt");
    assert_eq!(ctrl.model_num_digits, 6);
    assert!(ctrl.write_fgong_for_each_model);
    assert!(ctrl.write_fgong_for_best_model);
    assert!(ctrl.write_gyre_for_each_model);
    assert_eq!(ctrl.max_num_gyre_points, 512);
    assert!(ctrl.write_gyre_for_best_model);
    assert!(ctrl.write_profile_for_best_model);
    assert!(ctrl.save_model_for_best_model);
    assert!(ctrl.save_info_for_last_model);
    assert!(ctrl.save_controls);
    assert_eq!(ctrl.save_mode_model_number, 7);
    assert_eq!(ctrl.el_to_save, 1);
    assert_eq!(ctrl.order_to_save, -2);
    assert_eq!(
        (
            ctrl.adipls_irotkr,
            ctrl.adipls_nprtkr,
            ctrl.adipls_igm1kr,
            ctrl.adipls_npgmkr
        ),
        (1, 2, 3, 4)
    );
    assert!(ctrl.trace_chi2_seismo_delta_nu_info);
    assert!(ctrl.trace_chi2_seismo_nu_max_info);
    assert!(ctrl.trace_chi2_seismo_ratios_info);
    assert!(ctrl.trace_chi2_seismo_frequencies_info);
    assert!(ctrl.trace_chi2_spectro_info);

    // Save controls and verify lossless re-parse into identical fields.
    ctrl.write_controls_if_requested(&tmp.path)
        .expect("write save_controls");
    let dump_path = tmp.path.join("dumped_controls.txt");
    assert!(dump_path.is_file());
    let round_dir = TempDirGuard::new("roundtrip_reparse");
    round_dir.write(
        "inlist_astero_search_controls",
        &fs::read_to_string(&dump_path).expect("read dumped controls"),
    );
    let reparsed = parse_astero_search_controls(&round_dir.path)
        .expect("reparse saved controls");
    assert!((reparsed.simplex_alpha - 1.1).abs() <= 1.0e-14);
    assert!((reparsed.simplex_delta - 0.55).abs() <= 1.0e-14);
    assert_eq!(reparsed.adipls_npgmkr, 4);
    assert_eq!(reparsed.max_num_gyre_points, 512);
    assert_eq!(reparsed.astero_results_dbl_format, "(e24.15)");

    // Invalid format specifiers and missing include files must fail loudly.
    let bad_fmt = TempDirGuard::new("bad_fmt");
    bad_fmt.write(
        "inlist_astero_search_controls",
        "&astero_search_controls\n  astero_results_dbl_format = 'badfmt'\n/\n",
    );
    assert!(parse_astero_search_controls(&bad_fmt.path).is_err());

    let bad_inc = TempDirGuard::new("bad_inc");
    bad_inc.write(
        "inlist_astero_search_controls",
        "&astero_search_controls
  read_extra_astero_search_inlist(2) = .true.
  extra_astero_search_inlist_name(2) = 'does_not_exist.inlist'
/
",
    );
    assert!(parse_astero_search_controls(&bad_inc.path).is_err());
}

#[test]
fn test_kjeldsen_numax_order_gap_and_eval_surf_coefs_exact() {
    // Closed-form `nmax` and skipped-order gap check (`astero_support.f90:655-664`).
    let dnu = 135.0;
    let nu_max = 3090.0;
    let dnu_sun = 135.1;
    let nu_max_sun = 3090.0;
    let l0_with_gap = [2090.0, 2225.0, 2495.0, 2630.0];
    let n_est = estimate_l0_n_obs_with_numax(
        &l0_with_gap,
        dnu,
        nu_max,
        dnu_sun,
        nu_max_sun,
    );
    assert_eq!(n_est.len(), 4);
    // Relative step from f[1]=2225 to f[2]=2495 is +2*dnu (skipped n), NOT +1.
    assert!((n_est[1] - n_est[0] - 1.0).abs() <= 1.0e-12);
    assert!((n_est[2] - n_est[1] - 2.0).abs() <= 1.0e-12);
    assert!((n_est[3] - n_est[2] - 1.0).abs() <= 1.0e-12);
    // `nmax = (3090/135)*(135.1/3090)*22.6 - 1.6 = 21.01674...`, norders = round((2630-2090)/135)+1 = 5.
    let expected_nmax = (nu_max / dnu) * (dnu_sun / nu_max_sun) * 22.6 - 1.6;
    let expected_n1 = (expected_nmax - 2.0_f64).trunc();
    assert!((n_est[0] - expected_n1).abs() <= 1.0e-12);

    // Closed-form `eval_surf_coefs` (`astero_support.f90:1280-1334`).
    let m_numax = 1800.0;
    let s_numax = 3078.0;
    let ratio = 5000.0 * m_numax / s_numax;
    let (c1, c2) = eval_surf_coefs("cubic", -2.5e-12, 0.0, m_numax, s_numax);
    assert!((c1 - (-2.5e-12) * ratio * ratio * ratio).abs() <= 1.0e-12);
    assert_eq!(c2, 0.0);

    let (b1, b2) =
        eval_surf_coefs("combined", 1.2e-11, -3.4e3, m_numax, s_numax);
    assert!((b1 - 1.2e-11 * ratio * ratio * ratio).abs() <= 1.0e-10);
    assert!((b2 - (-3.4e3 / ratio)).abs() <= 1.0e-12);

    let (p1, p2) = eval_surf_coefs("power_law", -4.2, 4.9, m_numax, s_numax);
    assert!((p1 + 4.2).abs() <= 1.0e-15 && (p2 - 4.9).abs() <= 1.0e-15);
}

#[test]
fn test_ratios_max_seq_trim_and_r02_fixed_pairing_identity() {
    // Small variation `< 0.75 * dnu` must NOT break sequence; gap `> 1.5 * dnu` MUST split.
    let dnu = 100.0;
    let freqs = [900.0, 960.0, 1060.0, 1160.0, 1260.0, 1360.0, 1520.0];
    assert_eq!(get_max_sequence(&freqs, dnu), (1, 6));

    // Overhanging l=0 low mode & l=1 high mode trimmed to matching sequence lengths.
    let l0 = [850.0, 950.0, 1050.0, 1150.0, 1250.0, 1350.0];
    let s0 = [0.1; 6];
    let l1 = [900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0];
    let s1 = [0.1; 7];
    let l2 = [846.0, 945.0, 1045.0, 1145.0, 1245.0, 1345.0];
    let s2 = [0.2; 6];
    let r010 = eval_frequency_ratios_with_trimming(
        &l0, &s0, &l1, &s1, dnu, true, 0, 0, 0,
    )
    .expect("eval_frequency_ratios_with_trimming should succeed");
    assert_eq!((r010.n, r010.l0_first, r010.l1_first), (4, 0, 0));

    let obs_r02 = eval_r02_obs_with_indices(
        &l0,
        &s0,
        &l1,
        &s1,
        &l2,
        &s2,
        r010.n,
        r010.l0_first,
        r010.l1_first,
    );
    assert_eq!(obs_r02.i2_for_r02, vec![Some(1), Some(2), Some(3), Some(4)]);
    for &ratio in obs_r02.ratios_r02.iter().take(r010.n) {
        assert!((ratio - 0.05).abs() <= 1.0e-12);
    }

    // Model evaluation must reuse fixed `i2_for_r02` even if mode `j` shifts outside 0.25*dnu.
    let l2_mod = [846.0, 924.0, 1044.0, 1144.0, 1244.0, 1344.0];
    let (r02_mod, _) = eval_r02_model_with_indices(
        &l0,
        &l1,
        &l2_mod,
        r010.l0_first,
        r010.l1_first,
        &obs_r02,
    );
    // (950 - 924)/100 = 0.26 uses fixed index `1` (`924.0`), NOT skipped!
    assert!((r02_mod[0] - 0.26).abs() <= 1.0e-12);
    assert!((r02_mod[1] - 0.06).abs() <= 1.0e-12);
}

#[test]
fn test_simplex_custom_coefficients_converge_to_convex_minimum() {
    let x0 = [2.0_f64, -1.5_f64];
    let lo = [-5.0, -5.0];
    let hi = [5.0, 5.0];
    let (x_best, f_best, _) = simplex_minimize_with_options(
        &x0,
        &lo,
        &hi,
        300,
        1.0e-8,
        1.0e-8,
        1.0e-10,
        0.0,
        true,
        false,
        1,
        |x| {
            flexmesa_math::pow2(x[0] - 0.75)
                + 4.0 * flexmesa_math::pow2(x[1] + 1.25)
        },
    );
    assert!(f_best <= 1.0e-8, "unexpected residual {f_best}");
    assert!((x_best[0] - 0.75).abs() <= 1.0e-4);
    assert!((x_best[1] + 1.25).abs() <= 1.0e-4);
}

#[test]
fn test_eval_sample_unnamed_constraint_and_adipls_alt_and_mode_save() {
    let tmp = TempDirGuard::new("adipls_mode_save");
    tmp.write("adipls.c.pruned.in", ADIPLS_PRUNED_IN);
    let mut model = hydrated_solar_zams();
    model.model_number = 7;

    // Unnamed slot 1 active with sigma > 0 must match MESA (`fast_newuoa` behavior).
    let unnamed_dir = TempDirGuard::new("unnamed_constraint");
    unnamed_dir.write(
        "inlist_astero_search_controls",
        "&astero_search_controls
  constraint_name(1) = ''
  constraint_target(1) = 1.0d0
  constraint_sigma(1) = 1.0d-3
  include_constraint_in_chi2_spectro(1) = .true.
/
",
    );
    let c_unnamed = parse_astero_search_controls(&unnamed_dir.path)
        .expect("parse unnamed constraint");
    let (chi2_sp_unnamed, _, _) = eval_spectro_and_delta_nu_chi2(
        &model,
        &c_unnamed,
        &[0.0; MAX_CONSTRAINTS],
    );
    assert!(
        (chi2_sp_unnamed - 1.0e6).abs() <= 1.0e-6,
        "unnamed constraint(1) with target=1, sigma=1e-3, val=0 must contribute 1e6, got {chi2_sp_unnamed}"
    );

    // Discover real radial & dipole modes first.
    let mut base_ctrl = AsteroSearchControls::default();
    base_ctrl.oscillation_code = String::from("adipls");
    base_ctrl.add_atmosphere = false;
    base_ctrl.add_center_point = true;
    base_ctrl.keep_surface_point = false;
    base_ctrl.do_redistribute_mesh = false;
    let mut drv =
        AsteroAdipls::init(&tmp.path, &base_ctrl).expect("init adipls");
    let raw_l0 = drv
        .get_one_el_info(&model, 0, 1500.0, 3500.0, 120, 135.0, true)
        .expect("l=0");
    let raw_l1 = drv
        .get_one_el_info(&model, 1, 1500.0, 3500.0, 120, 135.0, false)
        .expect("l=1");
    let raw_l2 = drv
        .get_one_el_info(&model, 2, 1500.0, 3500.0, 120, 135.0, false)
        .expect("l=2");
    assert!(raw_l0.len() >= 6 && raw_l1.len() >= 6 && raw_l2.len() >= 5);

    let mut ctrl = base_ctrl;
    ctrl.nl = [6, 6, 5, 0];
    ctrl.freq_target[0] =
        raw_l0[1..7].iter().map(|m| m.cyclic_freq + 0.1).collect();
    ctrl.freq_sigma[0] = vec![1.0; 6];
    ctrl.freq_target[1] =
        raw_l1[1..7].iter().map(|m| m.cyclic_freq - 0.1).collect();
    ctrl.freq_sigma[1] = vec![1.0; 6];
    ctrl.freq_target[2] = raw_l2[1..6].iter().map(|m| m.cyclic_freq).collect();
    ctrl.freq_sigma[2] = vec![1.0; 5];
    ctrl.weights.chi2_seismo_fraction = 1.0;
    ctrl.weights.chi2_seismo_r_010_fraction = 0.25;
    ctrl.weights.chi2_seismo_r_02_fraction = 0.25;
    ctrl.correction_scheme = String::from("combined");
    ctrl.correction_factor = 1.0;
    ctrl.save_next_best_at_higher_frequency = true;
    ctrl.save_next_best_at_lower_frequency = true;
    ctrl.adipls_irotkr = 0;
    ctrl.adipls_nprtkr = 1;
    ctrl.adipls_igm1kr = 0;
    ctrl.adipls_npgmkr = 2;
    ctrl.write_fgong_for_each_model = true;
    ctrl.fgong_prefix = String::from("fg_");
    ctrl.fgong_postfix = String::from(".fgong");
    ctrl.model_num_digits = 5;
    ctrl.save_mode_model_number = model.model_number.cast_signed();
    ctrl.save_mode_filename = String::from("outputs/saved_mode.data");
    ctrl.el_to_save = 0;
    ctrl.order_to_save = raw_l0[1].order;

    let sample = eval_model_astero_sample(
        &model,
        &ctrl,
        &[0.0; MAX_CONSTRAINTS],
        ctrl.first_params(),
        1,
        &tmp.path,
    )
    .expect("eval sample with ratios, alt matches, fgong, and mode save");

    assert_eq!(sample.ratios_n, 4);
    assert_eq!(sample.obs_r01.len(), 4);
    assert_eq!(sample.model_r01.len(), 4);
    assert_eq!(sample.obs_r10.len(), 4);
    assert_eq!(sample.model_r10.len(), 4);
    assert!(sample.chi2_r010.is_finite() && sample.chi2_r010 >= 0.0);
    assert!(sample.chi2_r02.is_finite() && sample.chi2_r02 >= 0.0);
    assert!(sample.surf_coef1.is_finite() && sample.surf_coef2.is_finite());
    assert!(sample.model_freqs_alt_up[1][0] > sample.model_freqs[1][0]);
    assert!(sample.model_freqs_alt_down[1][0] < sample.model_freqs[1][0]);
    assert!(sample.model_freqs_corr_alt_up[1][0] > 0.0);
    assert!(sample.model_freqs_corr_alt_down[1][0] > 0.0);

    let fgong_path = tmp
        .path
        .join("outputs")
        .join(format!("fg_{:05}.fgong", model.model_number));
    assert!(fgong_path.is_file(), "expected {}", fgong_path.display());

    let mode_path = tmp.path.join("outputs/saved_mode.data");
    let mode_txt =
        fs::read_to_string(&mode_path).expect("saved eigenfunction file");
    assert!(mode_txt.contains("x = r/R"));
    assert!(mode_txt.contains("xi_r/R"));
    assert!(mode_txt.contains("freq (microHz)"));

    // Verify `adipls_irotkr = 1` propagates cleanly into `flexmesa-adipls`.
    let mut rot_ctrl = ctrl.clone();
    rot_ctrl.adipls_irotkr = 1;
    let rot_err = eval_model_astero_sample(
        &model,
        &rot_ctrl,
        &[0.0; MAX_CONSTRAINTS],
        rot_ctrl.first_params(),
        1,
        &tmp.path,
    )
    .expect_err("irotkr = 1 must propagate into adipls");
    assert!(rot_err.to_string().contains("irotkr"));

    // Verify AsteroGyre writes both per-model FGONG and GYRE files via store_model.
    tmp.write("gyre.in", MINIMAL_GYRE_IN);
    let mut g_ctrl = AsteroSearchControls::default();
    g_ctrl.oscillation_code = String::from("gyre");
    g_ctrl.gyre_input_file = String::from("gyre.in");
    g_ctrl.write_fgong_for_each_model = true;
    g_ctrl.fgong_prefix = String::from("gy_fg_");
    g_ctrl.write_gyre_for_each_model = true;
    g_ctrl.gyre_prefix = String::from("gy_model_");
    g_ctrl.gyre_non_ad = true;
    g_ctrl.max_num_gyre_points = 300;
    let mut gyre_drv =
        AsteroGyre::init(&tmp.path, &g_ctrl, ".").expect("init gyre");
    assert!(gyre_drv.gyre_non_ad());
    assert_eq!(gyre_drv.max_num_gyre_points(), 300);
    gyre_drv.store_model(&model).expect("gyre store_model");
    assert!(tmp
        .path
        .join(format!("outputs/gy_fg_{:04}.data", model.model_number))
        .is_file());
    assert!(tmp
        .path
        .join(format!("outputs/gy_model_{:04}.data", model.model_number))
        .is_file());
}

#[test]
fn test_search_driver_formats_artifacts_and_scan_grid_and_simplex_restarts() {
    let tmp = TempDirGuard::new("driver_artifacts_and_restarts");
    let model0 = setup_e2e_star_case(&tmp, 2);

    let grid_cfg = TempDirGuard::new("grid_cfg");
    grid_cfg.write(
        "inlist_astero_search_controls",
        &format!(
            "&astero_search_controls
  search_type = 'scan_grid'
  param_name(1) = 'alpha'
  vary_param(1) = .true.
  first_param(1) = {:.12e}
  min_param(1) = 1.90d0
  max_param(1) = 2.00d0
  delta_param(1) = 0.05d0
  constraint_name(1) = 'alpha_obs'
  constraint_target(1) = 1.95d0
  constraint_sigma(1) = 0.1d0
  include_constraint_in_chi2_spectro(1) = .true.
  chi2_seismo_fraction = 0.0d0
  min_age_limit = -1.0d0
  Lnuc_div_L_limit = -1.0d0
  surf_coef1_name = 'sc_one'
  surf_coef2_name = 'sc_two'
  astero_results_dbl_format = '(1pes24.14)'
  astero_results_int_format = '(i12)'
  astero_results_txt_format = '(a20)'
  write_best_model_data_for_each_sample = .true.
  num_digits = 5
  sample_results_prefix = 'best_'
  sample_results_postfix = '.info'
  write_fgong_for_best_model = .true.
  best_model_fgong_filename = 'best.fgong'
  write_gyre_for_best_model = .true.
  best_model_gyre_filename = 'best.gyre'
  write_profile_for_best_model = .true.
  best_model_profile_filename = 'best.profile'
  save_model_for_best_model = .true.
  best_model_save_model_filename = 'best.mod'
  save_info_for_last_model = .true.
  last_model_save_info_filename = 'last_info.txt'
  save_controls = .true.
  save_controls_filename = 'saved_ctrl.inlist'
  scan_grid_output_filename = 'grid.results'
/
",
            model0.controls.mixing_length_alpha
        ),
    );
    let ctrl =
        parse_astero_search_controls(&grid_cfg.path).expect("parse grid ctrl");

    let mut hook = AlphaHook;
    let out1 = tmp.path.join("run1");
    let logs1 = StarLogsOptions {
        output_root: Some(out1.as_path()),
        ..StarLogsOptions::default()
    };
    let report1 = run_star_astero_with_case_hooks(
        &tmp.path,
        "inlist",
        None,
        &ctrl,
        logs1,
        InlistLoadOptions::default(),
        Some(&mut hook),
    )
    .expect("run 3-point scan_grid");
    assert_eq!(report1.samples.len(), 3);
    assert!(report1.best_chi2 < 1.0e-12);
    assert!(out1.join("saved_ctrl.inlist").is_file());
    assert!(out1.join("outputs/best_00001.info").is_file());
    assert!(out1.join("outputs/best.fgong").is_file());
    assert!(out1.join("outputs/best.gyre").is_file());
    assert!(out1.join("outputs/best.profile").is_file());
    assert!(out1.join("outputs/best.mod").is_file());
    let last_info = fs::read_to_string(out1.join("outputs/last_info.txt"))
        .expect("last info");
    assert!(last_info.contains("chi^2 spectro"));
    let table = fs::read_to_string(out1.join("outputs/grid.results"))
        .expect("grid results");
    assert!(table.contains("sc_one"));
    assert!(table.contains("sc_two"));

    // Extend grid to 4 points (`2.05` added) and verify restart skips first 3 samples.
    let grid_res_cfg = TempDirGuard::new("grid_res_cfg");
    grid_res_cfg.write(
        "inlist_astero_search_controls",
        &format!(
            "&astero_search_controls
  search_type = 'scan_grid'
  restart_scan_grid_from_file = .true.
  scan_grid_output_filename = 'grid.results'
  astero_results_dbl_format = '(1pes24.14)'
  astero_results_int_format = '(i12)'
  astero_results_txt_format = '(a20)'
  param_name(1) = 'alpha'
  vary_param(1) = .true.
  first_param(1) = {:.12e}
  min_param(1) = 1.90d0
  max_param(1) = 2.05d0
  delta_param(1) = 0.05d0
  constraint_name(1) = 'alpha_obs'
  constraint_target(1) = 1.95d0
  constraint_sigma(1) = 0.1d0
  include_constraint_in_chi2_spectro(1) = .true.
  chi2_seismo_fraction = 0.0d0
  min_age_limit = -1.0d0
  Lnuc_div_L_limit = -1.0d0
/
",
            model0.controls.mixing_length_alpha
        ),
    );
    let ctrl_restart = parse_astero_search_controls(&grid_res_cfg.path)
        .expect("parse grid restart ctrl");
    let report2 = run_star_astero_with_case_hooks(
        &tmp.path,
        "inlist",
        None,
        &ctrl_restart,
        logs1,
        InlistLoadOptions::default(),
        Some(&mut hook),
    )
    .expect("restart scan_grid from file");
    assert_eq!(report2.samples.len(), 4);
    assert_eq!(report2.total_steps, 2); // Only the 4th grid point ran (2 steps).
    assert!((report2.samples[3].params.alpha - 2.05).abs() <= 1.0e-12);

    // Seed a 1D simplex run and verify restart from `simplex_results.data`.
    let simp_cfg = TempDirGuard::new("simp_cfg");
    simp_cfg.write(
        "inlist_astero_search_controls",
        "&astero_search_controls
  search_type = 'simplex'
  param_name(1) = 'alpha'
  vary_param(1) = .true.
  first_param(1) = 1.92d0
  min_param(1) = 1.80d0
  max_param(1) = 2.10d0
  constraint_name(1) = 'alpha_obs'
  constraint_target(1) = 1.95d0
  constraint_sigma(1) = 0.1d0
  include_constraint_in_chi2_spectro(1) = .true.
  chi2_seismo_fraction = 0.0d0
  min_age_limit = -1.0d0
  Lnuc_div_L_limit = -1.0d0
  simplex_itermax = 3
  simplex_fcn_calls_max = 6
  simplex_alpha = 1.1d0
  simplex_beta = 0.5d0
  simplex_gamma = 2.0d0
  simplex_delta = 0.5d0
  write_best_model_data_for_each_sample = .false.
  simplex_output_filename = 'simplex_results.data'
/
",
    );
    let s_ctrl =
        parse_astero_search_controls(&simp_cfg.path).expect("parse simplex");
    let simp_root = tmp.path.join("simp_run");
    let simp_out = simp_root.join("outputs");
    let simp_logs = StarLogsOptions {
        output_root: Some(simp_root.as_path()),
        ..StarLogsOptions::default()
    };
    let s1 = run_star_astero_with_case_hooks(
        &tmp.path,
        "inlist",
        None,
        &s_ctrl,
        simp_logs,
        InlistLoadOptions::default(),
        Some(&mut hook),
    )
    .expect("initial simplex run");
    assert!(s1.samples.len() >= 2);
    // `write_best_model_data_for_each_sample = false` must NOT write per-sample files.
    assert!(!simp_out.join("sample_0001.data").exists());

    let mut s_restart = s_ctrl.clone();
    s_restart.restart_simplex_from_file = true;
    s_restart.simplex_itermax = 2;
    s_restart.simplex_fcn_calls_max = 4;
    let s2 = run_star_astero_with_case_hooks(
        &tmp.path,
        "inlist",
        None,
        &s_restart,
        simp_logs,
        InlistLoadOptions::default(),
        Some(&mut hook),
    )
    .expect("restart simplex from file");
    assert!(s2.samples.len() >= 2);
    assert!(s2.best_chi2 <= 0.15);

    // Direct `run_single_astero_track_with_case_hooks` + `newuoa` 1st-try failure enforcement.
    let (_s, _st, _m) = run_single_astero_track_with_case_hooks(
        &InlistBundle::from_file_in_case_dir(&tmp.path, "inlist")
            .expect("bundle"),
        &s_ctrl,
        s_ctrl.first_params(),
        1,
        Some(1),
        &tmp.path,
        StarLogsOptions::default(),
        Some(&mut hook),
    )
    .expect("single astero track");

    // Verify NEWUOA/BOBYQA uncapped 1st-try failure (`astero_run_support.f90:607-612`).
    let mut fail_ctrl = s_ctrl;
    fail_ctrl.search_type = String::from("newuoa");
    fail_ctrl.min_age_for_chi2 = Some(1.0e20); // Impossible age -> track 1 produces no chi2.
    let fail_out = tmp.path.join("newuoa_fail");
    let err = run_star_astero_with_case_hooks(
        &tmp.path,
        "inlist",
        None,
        &fail_ctrl,
        StarLogsOptions {
            output_root: Some(fail_out.as_path()),
            ..StarLogsOptions::default()
        },
        InlistLoadOptions::default(),
        Some(&mut hook),
    )
    .expect_err("newuoa 1st-try failure must abort cleanly");
    assert!(err.to_string().contains("failed to find chi^2 on 1st try"));
}
