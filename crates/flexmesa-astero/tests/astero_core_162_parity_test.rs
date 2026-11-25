// SPDX-License-Identifier: GPL-3.0-or-later
//! Issue `#162` parity unit and oracle tests for `flexmesa-astero`.

#![forbid(unsafe_code)]
#![allow(clippy::similar_names, clippy::too_many_lines)]

use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use flexmesa_astero::adipls_support::AsteroAdipls;
use flexmesa_astero::gyre_support::AsteroGyre;
use flexmesa_astero::inlist_astero::{
    parse_astero_search_controls, AsteroSearchControls, MAX_CONSTRAINTS,
};
use flexmesa_astero::sample_eval::{
    eval_spectro_and_delta_nu_chi2, parse_gyre_outer_bound,
};
use flexmesa_astero::{
    eval_astero_chi2, eval_delta_pg_bildsten2012, eval_kjeldsen_corr,
    eval_model_astero_sample, eval_nu_max, eval_r02_model_with_indices,
    eval_r02_obs_with_indices, interpolate_ratio_r02, map_bobyqa_param,
    run_astero_test_suite_case_with_case_hooks,
    run_star_astero_with_case_hooks, simplex_minimize_with_options,
    try_bobyqa_minimize, try_newuoa_minimize,
    try_simplex_minimize_with_options, AsteroCaseHooks, AsteroChi2Weights,
    NU_MAX_SUN, TEFF_SUN_ASTERO,
};
use flexmesa_const::{MSUN, RSUN};
use flexmesa_num::bobyqa;
use flexmesa_star::controls::{
    AtmOption, AtmTTauOpacity, AtmTTauRelation, StarControls, StarError,
    StarJobControls,
};
use flexmesa_star::eval_photosphere_info;
use flexmesa_star::inlist::{InlistBundle, InlistLoadOptions};
use flexmesa_star::inlist_runner::StarLogsOptions;
use flexmesa_star::model::StarModel;
use flexmesa_star::pulse_bvp::OuterBoundKind;
use flexmesa_star::star_lib::star_write_model;
use flexmesa_star::{load_zams_model, set_hydro_vars};

static PARITY_DIR_SEQ: AtomicUsize = AtomicUsize::new(1);

struct TempParityDir {
    path: PathBuf,
}

impl TempParityDir {
    fn new(tag: &str) -> Self {
        let id = PARITY_DIR_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "flexmesa_ast162_parity_{tag}_{}_{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create parity temp dir");
        Self { path }
    }

    fn write(&self, rel: &str, body: &str) {
        let p = self.path.join(rel);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).expect("create parent dir");
        }
        fs::write(&p, body).expect("write temp file");
    }
}

impl Drop for TempParityDir {
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
  outer_bound = 'VACUUM',
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

fn setup_e2e_star_case(dir: &TempParityDir, max_steps: usize) -> StarModel {
    let model = hydrated_solar_zams();
    star_write_model(&model, &dir.path.join("start.mod"))
        .expect("write start.mod");
    let inlist = format!(
        "&star_job
  load_saved_model = .true.
  load_model_filename = 'start.mod'
  pgstar_flag = .false.
/

&controls
  max_model_number = {}
  max_years_for_timestep = 1.0d6
  calculate_brunt_n2 = .true.
  atm_option = 'T_tau'
  atm_T_tau_relation = 'Eddington'
  atm_T_tau_opacity = 'fixed'
  write_profiles_flag = .false.
/

&kap
  Zbase = 0.02d0
/
",
        model.model_number + max_steps
    );
    dir.write("inlist", &inlist);
    model
}

struct ParityHook;

impl AsteroCaseHooks for ParityHook {
    fn set_param(
        &mut self,
        bundle: &mut InlistBundle,
        _ctrl: &AsteroSearchControls,
        name: &str,
        val: f64,
    ) -> Result<(), StarError> {
        match name {
            "alpha" => bundle.controls.mixing_length_alpha = val,
            "initial_mass" => bundle.controls.initial_mass = val,
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
            "alpha_obs" => Ok(model.controls.mixing_length_alpha),
            "Teff" => Ok(model.teff),
            other => Err(StarError::InvalidInput(format!(
                "unexpected constraint '{other}'"
            ))),
        }
    }
}

#[test]
fn test_scaling_nu_max_and_delta_pg_deep_lamb_cut() {
    let nu_sun = eval_nu_max(MSUN, RSUN, TEFF_SUN_ASTERO);
    assert!((nu_sun - NU_MAX_SUN).abs() <= 1.0e-12);
    let nu_neg = eval_nu_max(MSUN, RSUN, -100.0);
    assert!(nu_neg.is_infinite() && nu_neg.is_sign_positive());

    // Synthetic profile with an interior Lamb cutoff (k=2) between two
    // g-cavity cells (k=1 and k=3). `omega2 >= sl2` must `continue`, not break.
    let r = [10.0e9, 8.0e9, 6.0e9, 4.0e9];
    let r_mid = [9.0e9, 7.0e9, 5.0e9, 3.0e9];
    let brunt_n2 = [0.0, 1.0e-5, 9.0e-5, 4.0e-5];
    let csound_face = [1.0e8, 1.0e8, 1.0e3, 1.0e8];
    let dpg =
        eval_delta_pg_bildsten2012(&r, &r_mid, &brunt_n2, &csound_face, 100.0);
    let integral_expected =
        1.0e-5_f64.sqrt() * 2.0e9 / 8.0e9 + 4.0e-5_f64.sqrt() * 2.0e9 / 4.0e9;
    let dpg_expected = 2.0_f64.sqrt() * PI * PI / integral_expected;
    assert!((dpg - dpg_expected).abs() <= 1.0e-12);
}

#[test]
fn test_kjeldsen_negative_radial_target_preserves_nonradial_correction() {
    let l0_obs = [2003.0, -1.0, 2260.0];
    let l0_mod = [2005.0, 2140.0, 2275.0];
    let l0_in = [1.0, 1.0, 1.0];
    let l0_n = [15.0, 16.0, 17.0];

    let all_nonradial_at_1 = [2005.0, 2100.0, 2275.0];
    let all_in = [1.0, 0.5, 1.0];
    let (corr_nr, a_div_r, _) = eval_kjeldsen_corr(
        &l0_obs,
        &l0_mod,
        &l0_in,
        &l0_n,
        128.5,
        &all_nonradial_at_1,
        &all_in,
        2150.0,
        4.9,
        1.0,
    );
    assert!(a_div_r.abs() > 0.0);
    assert!(
        (corr_nr[1] - all_nonradial_at_1[1]).abs() > 1.0e-6,
        "non-radial mode at index 1 must still be corrected"
    );

    let (corr_rad, _, _) = eval_kjeldsen_corr(
        &l0_obs, &l0_mod, &l0_in, &l0_n, 128.5, &l0_mod, &l0_in, 2150.0, 4.9,
        1.0,
    );
    assert_eq!(corr_rad[1], l0_mod[1]);
}

#[test]
fn test_astero_chi2_clamps_out_of_range_seismo_subfractions() {
    let weights = AsteroChi2Weights {
        chi2_seismo_fraction: 1.0,
        chi2_seismo_freq_fraction: None,
        chi2_seismo_delta_nu_fraction: 1.5,
        chi2_seismo_nu_max_fraction: 0.0,
        chi2_seismo_r_010_fraction: -0.5,
        chi2_seismo_r_02_fraction: -0.2,
        normalize_chi2_seismo_frequencies: true,
        normalize_chi2_seismo_r_010: true,
        normalize_chi2_seismo_r_02: true,
        normalize_chi2_spectro: true,
    };
    let (chi2, chi2_seismo, _, _, _, _) = eval_astero_chi2(
        &weights,
        &[2000.0],
        &[0.5],
        &[2010.0],
        &[0.1],
        &[0.01],
        &[0.5],
        &[0.2],
        &[0.02],
        &[0.4],
        135.0,
        0.5,
        136.0,
        3078.0,
        20.0,
        3078.0,
        &[],
        &[],
        &[],
    );
    assert!((chi2_seismo - 4.0).abs() <= 1.0e-12);
    assert!((chi2 - 4.0).abs() <= 1.0e-12);
}

#[test]
fn test_simplex_mod_simplex_tolerances_bounds_ars_and_init_retry() {
    // 1. Immediate stop when `f(x_first) < chi2_tol`.
    let (x1, f1, n1) = simplex_minimize_with_options(
        &[0.5],
        &[0.0],
        &[1.0],
        50,
        1.0e-10,
        1.0e-10,
        1.0,
        0.0,
        true,
        false,
        1,
        |x| (x[0] - 0.5) * (x[0] - 0.5) + 0.5,
    );
    assert_eq!(n1, 1);
    assert!((f1 - 0.5).abs() <= 1.0e-15);
    assert!((x1[0] - 0.5).abs() <= 1.0e-15);

    // 2. `enforce_bounds = true` penalizes OOB reflect with `1e99` instead of
    // clamping onto the boundary and calling `eval_fn` at `1.0`.
    let mut seen = Vec::new();
    let (_, _, _) = simplex_minimize_with_options(
        &[0.99],
        &[0.0],
        &[1.0],
        10,
        1.0e-10,
        1.0e-10,
        0.0,
        0.0,
        true,
        false,
        1,
        |x| {
            seen.push(x[0]);
            (x[0] - 2.0) * (x[0] - 2.0)
        },
    );
    assert!(seen.iter().all(|&v| (0.0..=1.0).contains(&v)));
    assert!(seen.iter().all(|&v| (v - 1.0).abs() > 1.0e-12));

    // 3. Do not exit early on a flat plateau (`f_tol` removed); step until
    // `x_atol`/`x_rtol` or `max_evals`.
    let (_, _, n_plat) = simplex_minimize_with_options(
        &[0.5],
        &[0.0],
        &[1.0],
        25,
        1.0e-8,
        1.0e-8,
        0.0,
        0.0,
        true,
        false,
        1,
        |x| 10.0 + 1.0e-12 * x[0] * x[0],
    );
    assert!(n_plat > 10);

    // 4. `adaptive_random_search` runs up to 100 trials inside one contraction
    // failure and accepts `f_ars <= f[h]`.
    let (_, f_ars, n_ars) = simplex_minimize_with_options(
        &[0.80],
        &[0.0],
        &[1.0],
        120,
        1.0e-10,
        1.0e-10,
        0.0,
        0.0,
        false,
        true,
        1_074_698_122,
        |x| {
            if (0.05..=0.20).contains(&x[0]) {
                1.0
            } else {
                10.0
            }
        },
    );
    assert!((f_ars - 1.0).abs() <= 1.0e-12);
    assert!(n_ars <= 125);
    let (_, f_ars_hit, n_ars_hit) = simplex_minimize_with_options(
        &[0.80],
        &[0.0],
        &[1.0],
        120,
        1.0e-10,
        1.0e-10,
        1.5,
        0.0,
        false,
        true,
        1_074_698_122,
        |x| {
            if (0.05..=0.20).contains(&x[0]) {
                1.0
            } else {
                10.0
            }
        },
    );
    assert!((f_ars_hit - 1.0).abs() <= 1.0e-12);
    assert!((5..=104).contains(&n_ars_hit));

    // 5. Initial displaced vertex halves toward `x_first` up to 20 times on Err.
    let (x_retry, f_retry, _) = try_simplex_minimize_with_options(
        &[0.50],
        &[0.0],
        &[1.0],
        80,
        1.0e-8,
        1.0e-8,
        1.0e-12,
        0.0,
        true,
        false,
        1,
        |x| {
            if x[0] > 0.65 {
                Err(StarError::ConvergenceFailure(String::from("oob")))
            } else {
                Ok((x[0] - 0.52) * (x[0] - 0.52))
            }
        },
    )
    .expect("displaced simplex retry succeeds");
    assert!((x_retry[0] - 0.52).abs() <= 1.0e-5);
    assert!(f_retry < 1.0e-8);
}

#[test]
fn test_bobyqa_and_newuoa_extrapolation_and_max_valid_value() {
    assert!((map_bobyqa_param(1.5, 1.0, 0.8, 1.2) - 1.3).abs() <= 1.0e-14);
    assert!((map_bobyqa_param(-1.5, 1.0, 0.8, 1.2) - 0.7).abs() <= 1.0e-14);

    // Unclamped `map_bobyqa_param` lets `NEWUOA` reach a target outside bounds.
    let (x_ext, f_ext, _) = try_newuoa_minimize(
        &[1.0, 0.25],
        &[0.8, 0.20],
        &[1.2, 0.30],
        200,
        0.45,
        1.0e-5,
        |x| Ok((x[0] - 1.25) * (x[0] - 1.25) + (x[1] - 0.25) * (x[1] - 0.25)),
    )
    .expect("newuoa extrapolate");
    assert!(x_ext[0] > 1.22);
    assert!(f_ext < 1.0e-4);

    // `rhoend >= rhobeg` uses `max_evals` directly instead of truncating at npt+1.
    let (_, _, n_eq) = try_newuoa_minimize(
        &[1.0, 0.25],
        &[0.8, 0.20],
        &[1.2, 0.30],
        30,
        0.45,
        0.45,
        |x| Ok((x[0] - 1.15) * (x[0] - 1.15) + (x[1] - 0.29) * (x[1] - 0.29)),
    )
    .expect("newuoa rhoend == rhobeg");
    assert!((5..=30).contains(&n_eq));

    // `try_bobyqa_minimize` clips objective values at `1.0e6` (not `1.0e10`).
    let x_first = [1.0, 0.25];
    let x_lower = [0.8, 0.20];
    let x_upper = [1.2, 0.30];
    let spike_obj = |x: &[f64]| -> f64 {
        if x[0] > 1.05 {
            1.0e8
        } else {
            (x[0] - 0.90) * (x[0] - 0.90) + (x[1] - 0.25) * (x[1] - 0.25)
        }
    };
    let (xb_wrap, fb_wrap, _) = try_bobyqa_minimize(
        &x_first,
        &x_lower,
        &x_upper,
        6,
        0.45,
        1.0e-4,
        |x| Ok(spike_obj(x)),
    )
    .expect("bobyqa 1e6 clip");
    let mut u_oracle = [0.0_f64; 2];
    let res_1e6 = bobyqa(
        2,
        5,
        &mut u_oracle,
        &[-1.0, -1.0],
        &[1.0, 1.0],
        0.45,
        1.0e-4,
        6,
        1.0e6,
        |u| {
            let p = [
                map_bobyqa_param(u[0], x_first[0], x_lower[0], x_upper[0]),
                map_bobyqa_param(u[1], x_first[1], x_lower[1], x_upper[1]),
            ];
            Ok(spike_obj(&p))
        },
    )
    .expect("oracle 1e6");
    let x_oracle = [
        map_bobyqa_param(res_1e6.x_min[0], x_first[0], x_lower[0], x_upper[0]),
        map_bobyqa_param(res_1e6.x_min[1], x_first[1], x_lower[1], x_upper[1]),
    ];
    assert!((xb_wrap[0] - x_oracle[0]).abs() <= 1.0e-14);
    assert!((xb_wrap[1] - x_oracle[1]).abs() <= 1.0e-14);
    assert!((fb_wrap - res_1e6.f_min).abs() <= 1.0e-12);
}

#[test]
fn test_sample_eval_numax_validation_need_modes_r02_and_kjeldsen() {
    let tmp = TempParityDir::new("sample_eval");
    tmp.write("adipls.c.pruned.in", ADIPLS_PRUNED_IN);
    tmp.write("gyre.in", MINIMAL_GYRE_IN);
    assert_eq!(
        parse_gyre_outer_bound(&tmp.path, "gyre.in").expect("parse gyre"),
        OuterBoundKind::Vacuum
    );
    tmp.write("gyre_dziem.in", "&osc\n  outer_bound = 'DZIEM',\n/\n");
    assert_eq!(
        parse_gyre_outer_bound(&tmp.path, "gyre_dziem.in")
            .expect("parse dziem"),
        OuterBoundKind::AdiplsIsothermal
    );

    let mut model = hydrated_solar_zams();
    model.controls.output.get_delta_nu_from_scaled_solar = true;
    model.controls.output.delta_nu_sun = 140.0;
    model.controls.output.nu_max_sun = 3150.0;
    model.controls.output.astero_teff_sun = 5800.0;
    let mut ctrl = AsteroSearchControls::default();
    ctrl.delta_nu = 140.0;
    ctrl.delta_nu_sigma = 1.0;
    let (_, _, dnu_scaled) =
        eval_spectro_and_delta_nu_chi2(&model, &ctrl, &[0.0; MAX_CONSTRAINTS]);
    let phot = eval_photosphere_info(&model);
    let dnu_expected = 140.0
        * (model.mstar / MSUN).sqrt()
        * flexmesa_math::pow3(model.teff / 5800.0)
        / flexmesa_math::pow(phot.l_lsun, 0.75);
    assert!((dnu_scaled - dnu_expected).abs() <= 1.0e-10);

    // `chi2_seismo_nu_max_fraction > 0` without `nu_max` / `nu_max_sigma` errors.
    let mut bad_numax = AsteroSearchControls::default();
    bad_numax.weights.chi2_seismo_fraction = 1.0;
    bad_numax.weights.chi2_seismo_nu_max_fraction = 0.5;
    let err = eval_model_astero_sample(
        &model,
        &bad_numax,
        &[0.0; MAX_CONSTRAINTS],
        bad_numax.first_params(),
        1,
        &tmp.path,
    )
    .expect_err("missing nu_max must fail");
    assert!(matches!(err, StarError::InvalidInput(_)));

    // `need_modes` skips pulsation when only delta_nu + nu_max are active.
    let mut no_modes = AsteroSearchControls::default();
    no_modes.nl = [2, 0, 0, 0];
    no_modes.freq_target[0] = vec![99999.0, 99999.5];
    no_modes.freq_sigma[0] = vec![1.0, 1.0];
    no_modes.delta_nu = 135.0;
    no_modes.delta_nu_sigma = 5.0;
    no_modes.nu_max = 3078.0;
    no_modes.nu_max_sigma = 100.0;
    no_modes.weights.chi2_seismo_fraction = 1.0;
    no_modes.weights.chi2_seismo_delta_nu_fraction = 0.6;
    no_modes.weights.chi2_seismo_nu_max_fraction = 0.4;
    let res_no_modes = eval_model_astero_sample(
        &model,
        &no_modes,
        &[0.0; MAX_CONSTRAINTS],
        no_modes.first_params(),
        1,
        &tmp.path,
    )
    .expect("delta_nu + nu_max only succeeds");
    assert_eq!(res_no_modes.model_freqs[0], [] as [f64; 0]);
    assert!(res_no_modes.chi2.is_finite() && res_no_modes.chi2 < 1.0e90);

    // Discover ZAMS radial/dipole/quadrupole modes via ADIPLS.
    let mut base = AsteroSearchControls::default();
    base.oscillation_code = String::from("adipls");
    base.add_atmosphere = false;
    base.add_center_point = true;
    base.keep_surface_point = false;
    base.do_redistribute_mesh = false;
    let mut drv = AsteroAdipls::init(&tmp.path, &base).expect("init adipls");
    let raw_l0 = drv
        .get_one_el_info(&model, 0, 1500.0, 3500.0, 120, 135.0, true)
        .expect("l=0");
    let raw_l1 = drv
        .get_one_el_info(&model, 1, 1500.0, 3500.0, 120, 135.0, false)
        .expect("l=1");
    let raw_l2 = drv
        .get_one_el_info(&model, 2, 1500.0, 3500.0, 120, 135.0, false)
        .expect("l=2");

    // Empty `r02` pairs fail with `1e99` rather than scoring `chi2_r02 = 0.0`.
    let mut r02_fail = base.clone();
    r02_fail.nl = [6, 6, 1, 0];
    r02_fail.freq_target[0] =
        raw_l0[1..7].iter().map(|m| m.cyclic_freq).collect();
    r02_fail.freq_sigma[0] = vec![1.0; 6];
    r02_fail.freq_target[1] =
        raw_l1[1..7].iter().map(|m| m.cyclic_freq).collect();
    r02_fail.freq_sigma[1] = vec![1.0; 6];
    r02_fail.freq_target[2] = vec![9999.0];
    r02_fail.freq_sigma[2] = vec![1.0];
    r02_fail.weights.chi2_seismo_fraction = 1.0;
    r02_fail.weights.chi2_seismo_r_02_fraction = 0.5;
    let r02_fail_res = eval_model_astero_sample(
        &model,
        &r02_fail,
        &[0.0; MAX_CONSTRAINTS],
        r02_fail.first_params(),
        1,
        &tmp.path,
    )
    .expect("empty r02 pairs returns 1e99");
    assert_eq!(r02_fail_res.chi2, 1.0e99);

    // Valid `r02` interpolates ratios on uncorrected `model_freqs[0]` axis.
    let mut r02_ok = r02_fail.clone();
    r02_ok.nl[2] = 5;
    r02_ok.freq_target[0] =
        raw_l0[1..7].iter().map(|m| m.cyclic_freq + 2.0).collect();
    r02_ok.freq_target[2] =
        raw_l2[1..6].iter().map(|m| m.cyclic_freq).collect();
    r02_ok.freq_sigma[2] = vec![1.0; 5];
    r02_ok.correction_scheme = String::from("cubic");
    r02_ok.correction_factor = 1.0;
    let r02_res = eval_model_astero_sample(
        &model,
        &r02_ok,
        &[0.0; MAX_CONSTRAINTS],
        r02_ok.first_params(),
        1,
        &tmp.path,
    )
    .expect("valid r02");
    let obs_r2 = eval_r02_obs_with_indices(
        &r02_ok.freq_target[0],
        &r02_ok.freq_sigma[0],
        &r02_ok.freq_target[1],
        &r02_ok.freq_sigma[1],
        &r02_ok.freq_target[2],
        &r02_ok.freq_sigma[2],
        r02_res.ratios_n,
        r02_res.ratios_l0_first,
        r02_res.ratios_l1_first,
    );
    let (rm, _) = eval_r02_model_with_indices(
        &r02_res.model_freqs_corr[0],
        &r02_res.model_freqs_corr[1],
        &r02_res.model_freqs_corr[2],
        r02_res.ratios_l0_first,
        r02_res.ratios_l1_first,
        &obs_r2,
    );
    let (_, fm) = eval_r02_model_with_indices(
        &r02_res.model_freqs[0],
        &r02_res.model_freqs_corr[1],
        &r02_res.model_freqs_corr[2],
        r02_res.ratios_l0_first,
        r02_res.ratios_l1_first,
        &obs_r2,
    );
    let mut checked_r02 = 0usize;
    for k in 0..obs_r2.ratios_r02.len() {
        if obs_r2.sigmas_r02[k] > 0.0 {
            let expected = interpolate_ratio_r02(obs_r2.r02_freqs[k], &fm, &rm);
            assert!((r02_res.model_r02[k] - expected).abs() <= 1.0e-12);
            checked_r02 += 1;
        }
    }
    assert!(checked_r02 > 0);

    // `cubic` with `correction_factor = 0.0` still fits `surf_coef1` while
    // applying zero shift, and `obs = 0.0` still contributes to `chi2_radial`.
    let mut cubic_zero = r02_ok.clone();
    cubic_zero.nl = [4, 0, 0, 0];
    cubic_zero.freq_target[0] = vec![
        0.0,
        raw_l0[2].cyclic_freq,
        raw_l0[3].cyclic_freq,
        raw_l0[4].cyclic_freq,
    ];
    cubic_zero.freq_sigma[0] = vec![1000.0; 4];
    cubic_zero.correction_factor = 0.0;
    cubic_zero.weights.chi2_seismo_r_02_fraction = 0.0;
    let cubic_zero_res = eval_model_astero_sample(
        &model,
        &cubic_zero,
        &[0.0; MAX_CONSTRAINTS],
        cubic_zero.first_params(),
        1,
        &tmp.path,
    )
    .expect("cubic factor 0");
    assert!(cubic_zero_res.surf_coef1.abs() > 0.0);
    assert_eq!(
        cubic_zero_res.model_freqs_corr[0],
        cubic_zero_res.model_freqs[0]
    );
    assert!(cubic_zero_res.chi2_radial > 0.0);

    // `kjeldsen` with `nu_max <= 0` does not fall back to `nu_max_uhz`.
    let mut kj_no_numax = base.clone();
    kj_no_numax.nl = [4, 2, 0, 0];
    let kj_l0_targets: Vec<f64> = raw_l0[1..5]
        .iter()
        .map(|m| {
            m.cyclic_freq
                - 20.0 * flexmesa_math::pow(m.cyclic_freq / 3078.0, 4.9)
        })
        .collect();
    let kj_dnu_obs = (kj_l0_targets[3] - kj_l0_targets[0]) / 3.0;
    kj_no_numax.freq_target[0] = kj_l0_targets;
    kj_no_numax.freq_sigma[0] = vec![1.0; 4];
    kj_no_numax.freq_target[1] = vec![-1.0, raw_l1[2].cyclic_freq];
    kj_no_numax.freq_sigma[1] = vec![1.0; 2];
    kj_no_numax.delta_nu = kj_dnu_obs;
    kj_no_numax.nu_max = -1.0;
    kj_no_numax.correction_scheme = String::from("kjeldsen");
    kj_no_numax.correction_factor = 1.0;
    kj_no_numax.weights.chi2_seismo_fraction = 1.0;
    let kj_none = eval_model_astero_sample(
        &model,
        &kj_no_numax,
        &[0.0; MAX_CONSTRAINTS],
        kj_no_numax.first_params(),
        1,
        &tmp.path,
    )
    .expect("kjeldsen no nu_max");
    assert_eq!(kj_none.a_div_r, 0.0);
    assert_eq!(kj_none.model_freqs_corr[0], kj_none.model_freqs[0]);

    // `kjeldsen` with `nu_max > 0` leaves negative `l1` target uncorrected
    // while shifting valid non-radial modes.
    let mut kj_with_numax = kj_no_numax;
    kj_with_numax.nu_max = 3078.0;
    let kj_neg_l1 = eval_model_astero_sample(
        &model,
        &kj_with_numax,
        &[0.0; MAX_CONSTRAINTS],
        kj_with_numax.first_params(),
        1,
        &tmp.path,
    )
    .expect("kjeldsen neg l1");
    assert!(kj_neg_l1.a_div_r.abs() > 0.0);
    assert_eq!(
        kj_neg_l1.model_freqs_corr[1][0],
        kj_neg_l1.model_freqs[1][0]
    );
    assert!(
        (kj_neg_l1.model_freqs_corr[1][1] - kj_neg_l1.model_freqs[1][1]).abs()
            > 1.0e-6
    );

    // `gyre_non_ad = true` no longer bypasses `AsteroGyre`.
    let mut gy_probe = AsteroSearchControls::default();
    gy_probe.oscillation_code = String::from("gyre");
    gy_probe.gyre_input_file = String::from("gyre.in");
    let mut g_drv =
        AsteroGyre::init(&tmp.path, &gy_probe, ".").expect("gyre init");
    let mut gy_modes =
        g_drv.get_modes(&model, 0, true).expect("gyre l=0 modes");
    gy_modes.sort_by(|a, b| a.cyclic_freq.total_cmp(&b.cyclic_freq));
    let mut gy_ctrl = gy_probe;
    gy_ctrl.gyre_non_ad = true;
    gy_ctrl.write_gyre_for_each_model = true;
    gy_ctrl.gyre_prefix = String::from("gy_nad_");
    gy_ctrl.nl = [2, 0, 0, 0];
    gy_ctrl.freq_target[0] =
        vec![gy_modes[0].cyclic_freq, gy_modes[1].cyclic_freq];
    gy_ctrl.freq_sigma[0] = vec![1.0, 1.0];
    gy_ctrl.weights.chi2_seismo_fraction = 1.0;
    let gy_res = eval_model_astero_sample(
        &model,
        &gy_ctrl,
        &[0.0; MAX_CONSTRAINTS],
        gy_ctrl.first_params(),
        1,
        &tmp.path,
    )
    .expect("gyre_non_ad sample eval");
    assert_eq!(gy_res.model_freqs[0].len(), 2);
    assert!(tmp
        .path
        .join(format!("outputs/gy_nad_{:04}.data", model.model_number))
        .is_file());
}

#[test]
fn test_inlist_parser_nl_zero_ascending_ranges_and_ols_delta_nu() {
    let tmp = TempParityDir::new("parser");
    tmp.write(
        "inlist_astero_search_controls",
        "&astero_search_controls
  nl(0) = 0
  freq_target(0, 1) = 1000.0d0
  freq_target(0, 2) = 1100.0d0
  l0_n_obs(1:2) = -1.0d0
/
",
    );
    let ctrl_zero = parse_astero_search_controls(&tmp.path)
        .expect("explicit nl(0) = 0 must not be overridden");
    assert_eq!(ctrl_zero.nl[0], 0);
    assert_eq!(ctrl_zero.freq_target[0], [] as [f64; 0]);

    let bad = TempParityDir::new("parser_desc");
    bad.write(
        "inlist_astero_search_controls",
        "&astero_search_controls
  nl(0) = 2
  freq_target(0, 1) = 1100.0d0
  freq_target(0, 2) = 1000.0d0
/
",
    );
    assert!(matches!(
        parse_astero_search_controls(&bad.path),
        Err(StarError::InvalidInput(_))
    ));

    let ols = TempParityDir::new("parser_ols");
    ols.write(
        "inlist_astero_search_controls",
        "&astero_search_controls
  nl(0) = 3
  nl(1:3) = 0
  l0_n_obs(1) = 20.0d0
  l0_n_obs(2) = 21.0d0
  l0_n_obs(3) = 22.0d0
  freq_target(0, 1) = 2700.0d0
  freq_target(0, 2) = 2835.0d0
  freq_target(0, 3) = 2970.0d0
  freq_sigma(0, 1:3) = 0.5d0
  delta_nu = -1.0d0
  delta_nu_sigma = -1.0d0
/
",
    );
    let ctrl_ols =
        parse_astero_search_controls(&ols.path).expect("ols delta_nu");
    assert!((ctrl_ols.delta_nu - 135.0).abs() <= 1.0e-12);
    let expected_sig = 0.5 / 2.0_f64.sqrt();
    assert!((ctrl_ols.delta_nu_sigma - expected_sig).abs() <= 1.0e-12);
}

#[test]
fn test_search_driver_param_order_validation_table_r02_and_e2e_zero_chi2() {
    let tmp = TempParityDir::new("driver_e2e");
    tmp.write("adipls.c.pruned.in", ADIPLS_PRUNED_IN);
    let model = setup_e2e_star_case(&tmp, 2);

    // 1. Varied parameter with `min >= max` fails loudly.
    tmp.write(
        "bad_bounds/inlist_astero_search_controls",
        "&astero_search_controls
  search_type = 'simplex'
  param_name(1) = 'alpha'
  vary_param(1) = .true.
  first_param(1) = 1.9d0
  min_param(1) = 2.0d0
  max_param(1) = 1.8d0
  delta_param(1) = 0.1d0
/
",
    );
    let bad_bounds = parse_astero_search_controls(&tmp.path.join("bad_bounds"))
        .expect("parse bad bounds controls");
    let mut hook = ParityHook;
    let err = run_star_astero_with_case_hooks(
        &tmp.path,
        "inlist",
        Some(1),
        &bad_bounds,
        StarLogsOptions::default(),
        InlistLoadOptions::default(),
        Some(&mut hook),
    )
    .expect_err("min >= max must fail loudly");
    assert!(err.to_string().contains("min_param >= max_param"));

    // 2. Discover real ZAMS frequencies so sample 2 produces `r01`/`r02` ratios.
    let mut drv_ctrl = AsteroSearchControls::default();
    drv_ctrl.oscillation_code = String::from("adipls");
    drv_ctrl.add_atmosphere = false;
    drv_ctrl.add_center_point = true;
    drv_ctrl.keep_surface_point = false;
    drv_ctrl.do_redistribute_mesh = false;
    let mut drv =
        AsteroAdipls::init(&tmp.path, &drv_ctrl).expect("init adipls");
    let raw_l0 = drv
        .get_one_el_info(&model, 0, 1500.0, 3500.0, 120, 135.0, true)
        .expect("l=0");
    let raw_l1 = drv
        .get_one_el_info(&model, 1, 1500.0, 3500.0, 120, 135.0, false)
        .expect("l=1");
    let raw_l2 = drv
        .get_one_el_info(&model, 2, 1500.0, 3500.0, 120, 135.0, false)
        .expect("l=2");

    // Reversed `param_name` ordering (`alpha` before `initial_mass`),
    // `r01_obs_` & `r02_obs_` table serialization, omitted `total`, and
    // `restart_simplex_from_file` round-tripping `r01`/`r02`.
    tmp.write(
        "simp_cfg/inlist_astero_search_controls",
        "&astero_search_controls
  search_type = 'simplex'
  oscillation_code = 'adipls'
  add_atmosphere = .false.
  add_center_point = .true.
  keep_surface_point = .false.
  do_redistribute_mesh = .false.
  min_age_limit = -1.0d0
  Lnuc_div_L_limit = -1.0d0
  max_yrs_dt_when_warm = 1.0d9
  max_yrs_dt_when_hot = 1.0d9
  simplex_itermax = 1
  simplex_fcn_calls_max = 2
  param_name(1) = 'alpha'
  vary_param(1) = .true.
  first_param(1) = 2.0d0
  min_param(1) = 1.9d0
  max_param(1) = 2.1d0
  delta_param(1) = 0.05d0
  param_name(2) = 'initial_mass'
  vary_param(2) = .true.
  first_param(2) = 1.0d0
  min_param(2) = 0.9d0
  max_param(2) = 1.1d0
  delta_param(2) = 0.05d0
  correction_scheme = 'none'
/
",
    );
    let mut simp_ctrl =
        parse_astero_search_controls(&tmp.path.join("simp_cfg"))
            .expect("parse simplex controls");
    simp_ctrl.nl = [6, 6, 5, 0];
    simp_ctrl.freq_target[0] =
        raw_l0[1..7].iter().map(|m| m.cyclic_freq).collect();
    simp_ctrl.freq_sigma[0] = vec![50.0; 6];
    simp_ctrl.freq_target[1] =
        raw_l1[1..7].iter().map(|m| m.cyclic_freq).collect();
    simp_ctrl.freq_sigma[1] = vec![50.0; 6];
    simp_ctrl.freq_target[2] =
        raw_l2[1..6].iter().map(|m| m.cyclic_freq).collect();
    simp_ctrl.freq_sigma[2] = vec![50.0; 5];
    simp_ctrl.weights.chi2_seismo_fraction = 0.5;
    simp_ctrl.weights.chi2_seismo_r_010_fraction = 0.25;
    simp_ctrl.weights.chi2_seismo_r_02_fraction = 0.25;

    let run_root = tmp.path.join("simp_table");
    let logs = StarLogsOptions {
        output_root: Some(run_root.as_path()),
        ..StarLogsOptions::default()
    };
    let rep1 = run_star_astero_with_case_hooks(
        &tmp.path,
        "inlist",
        None,
        &simp_ctrl,
        logs,
        InlistLoadOptions::default(),
        Some(&mut hook),
    )
    .expect("simplex run with reordered params and ratios");
    assert_eq!(rep1.samples.len(), 3);
    assert!((rep1.samples[1].params.alpha - 2.0).abs() > 1.0e-6);
    assert!((rep1.samples[1].params.initial_mass - 1.0).abs() <= 1.0e-15);
    assert!((rep1.samples[2].params.initial_mass - 1.0).abs() > 1.0e-6);
    assert!(rep1.samples.iter().all(|s| s.ratios_n > 0));
    let table_txt =
        fs::read_to_string(run_root.join("outputs/simplex_results.data"))
            .expect("read simplex_results.data");
    assert!(table_txt.contains("r01_obs_1"));
    assert!(table_txt.contains("r02_obs_1"));
    assert!(!table_txt.lines().nth(1).unwrap_or("").contains("total"));

    let mut restart_ctrl = simp_ctrl;
    restart_ctrl.restart_simplex_from_file = true;
    restart_ctrl.simplex_fcn_calls_max = 1;
    let rep2 = run_star_astero_with_case_hooks(
        &tmp.path,
        "inlist",
        None,
        &restart_ctrl,
        logs,
        InlistLoadOptions::default(),
        Some(&mut hook),
    )
    .expect("simplex restart parses r01/r02");
    assert!(rep2.samples.len() >= 4);
    assert!(rep2.samples[..3].iter().all(|s| !s.obs_r01.is_empty()));
    assert!(rep2.samples[..3].iter().all(|s| !s.obs_r02.is_empty()));

    // 3. `e2e` runner accepts `best.chi2 == 0.0` (`b.chi2 >= 0.0`).
    let e2e_case = tmp.path.join("zero_chi2_case");
    fs::create_dir_all(&e2e_case).expect("create e2e case");
    star_write_model(&model, &e2e_case.join("start.mod"))
        .expect("write start.mod");
    fs::write(
        e2e_case.join("inlist"),
        fs::read_to_string(tmp.path.join("inlist")).expect("read inlist"),
    )
    .expect("write e2e inlist");
    fs::write(
        e2e_case.join("inlist_astero_search_controls"),
        format!(
            "&astero_search_controls
  search_type = 'use_first_values'
  chi2_seismo_fraction = 0.0d0
  min_age_limit = -1.0d0
  Lnuc_div_L_limit = -1.0d0
  param_name(1) = 'alpha'
  first_param(1) = {:.16e}
  constraint_name(1) = 'alpha_obs'
  constraint_target(1) = {:.16e}
  constraint_sigma(1) = 0.1d0
  include_constraint_in_chi2_spectro(1) = .true.
/
",
            model.controls.mixing_length_alpha,
            model.controls.mixing_length_alpha,
        ),
    )
    .expect("write e2e astero controls");
    let e2e_rep = run_astero_test_suite_case_with_case_hooks(
        &e2e_case,
        true,
        None,
        false,
        StarLogsOptions::default(),
        InlistLoadOptions::default(),
        Some(&mut hook),
    )
    .expect("zero chi2 e2e run");
    assert!(e2e_rep.assertions_passed && e2e_rep.passed);
}
