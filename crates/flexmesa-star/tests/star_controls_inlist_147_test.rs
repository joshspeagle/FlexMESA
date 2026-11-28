// SPDX-License-Identifier: GPL-3.0-or-later
//! Issue `#147` tests for `steps_to_take_before_terminate` execution
//! (`inlist_runner.rs`), `num_accretion_species` bounds, `predictive_*` and
//! `diffusion_isolve_solver` `trim_end` matching, bounded positional `xa_*`
//! limit arrays (`NUM_XA_LIMITS = 9`), and explicit slice rejection on
//! user/micro/diffusion/abundance controls.

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;

use flexmesa_star::inlist::InlistBundle;
use flexmesa_star::inlist_runner::{
    run_inlist_stage, DefaultStarExtras, StageRunReport,
};
use flexmesa_star::winds::get_xa_for_accretion;
use flexmesa_star::{init_or_update_model_from_bundle, StarError, StarModel};

struct TempCaseGuard {
    root: PathBuf,
}

impl TempCaseGuard {
    fn new(tag: &str) -> Self {
        static SEQ: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let id = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "flexmesa_star_147_{tag}_{}_{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create temp test dir");
        Self { root }
    }
}

impl Drop for TempCaseGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn run_steps_before_terminate_case(
    tag: &str,
    extra_star_job: &str,
    steps: isize,
    cap: Option<usize>,
) -> (StageRunReport, StarModel, TempCaseGuard) {
    let tmp = TempCaseGuard::new(tag);
    let inlist_path = tmp.root.join("inlist_steps");
    fs::write(
        &inlist_path,
        format!(
            "&star_job\n\
               {extra_star_job}\n\
               steps_to_take_before_terminate = {steps}\n\
             /\n\
             &kap\n\
               Zbase = 0.02d0\n\
             /\n\
             &controls\n\
               initial_mass = 1.0d0\n\
               photo_interval = -1\n\
             /\n"
        ),
    )
    .expect("write inlist_steps");
    let bundle = InlistBundle::from_file(&inlist_path).expect("parse inlist");
    let mut extras = DefaultStarExtras;
    let mut model =
        init_or_update_model_from_bundle(&bundle, None, &mut extras)
            .expect("init model");
    let report =
        run_inlist_stage(&mut model, &bundle, &mut extras, cap).expect("run");
    (report, model, tmp)
}

/// Verifies `steps_to_take_before_terminate >= 0` matches MESA `26.4.1`:
/// sets `max_model_number = model_number + steps_to_take_before_terminate`,
/// executes at least 1 step when `steps_to_take_before_terminate == 0`,
/// emits `"max_model_number"` (never `"steps_to_take_before_terminate"`),
/// and honors caller `max_steps_cap` for smoke runs
/// (`star/job/run_star_support.f90:108-116, 2014-2019`;
/// `star/private/do_one_utils.f90:798-803`).
#[test]
fn test_147_steps_to_take_before_terminate() {
    // 1. Positive steps (= 2) terminates with `"max_model_number"` and
    //    honors `required_termination_code_string`.
    let (r2, m2, tmp2) = run_steps_before_terminate_case(
        "steps_pos",
        "save_model_when_terminate = .true.\n\
         save_model_filename = 'done.mod'\n\
         required_termination_code_string(1) = 'max_model_number'",
        2,
        None,
    );
    assert_eq!(r2.steps_taken, 2);
    assert_eq!(r2.termination_code, "max_model_number");
    assert_eq!(m2.controls.max_model_number, Some(2));
    assert!(
        tmp2.root.join("done.mod").is_file(),
        "save_model_when_terminate must write model for max_model_number"
    );

    // 2. Zero steps (`steps_to_take_before_terminate = 0`) still enters
    //    evolve_loop once -> 1 step, `"max_model_number"`.
    let (r0, m0, _tmp0) =
        run_steps_before_terminate_case("steps_zero", "", 0, None);
    assert_eq!(r0.steps_taken, 1);
    assert_eq!(r0.termination_code, "max_model_number");
    assert_eq!(m0.controls.max_model_number, Some(0));

    // 3. Offset initial model_number = 5 + steps = 1 -> max_model_number = 6.
    let (roff, moff, _tmp_off) = run_steps_before_terminate_case(
        "steps_offset",
        "set_initial_model_number = .true.\ninitial_model_number = 5",
        1,
        None,
    );
    assert_eq!(roff.steps_taken, 1);
    assert_eq!(moff.model_number, 6);
    assert_eq!(moff.controls.max_model_number, Some(6));
    assert_eq!(roff.termination_code, "max_model_number");

    // 4. Caller smoke cap (`Some(1)`) stops before steps = 4.
    let (rcap, mcap, _tmp_cap) =
        run_steps_before_terminate_case("steps_cap", "", 4, Some(1));
    assert_eq!(rcap.steps_taken, 1);
    assert_eq!(rcap.termination_code, "max_steps_cap");
    assert_eq!(mcap.controls.max_model_number, Some(4));
}

/// Verifies 1-based positional indexing, chronological overrides (including
/// scalar slot-1 overrides and `extra_controls_inlist`), and alignment of
/// `xa_*` and `accretion_species_*` arrays (`NUM_XA_LIMITS = 9`).
#[test]
fn test_147_positional_xa_limits_and_accretion() {
    let tmp = TempCaseGuard::new("xa_pos");
    let main_p = tmp.root.join("inlist_main");
    let extra_p = tmp.root.join("inlist_extra_xa");

    fs::write(
        &main_p,
        "&kap\n\
           Zbase = 0.02d0\n\
         /\n\
         &controls\n\
           initial_mass = 1.0d0\n\
           accrete_same_as_surface = .false.\n\
           accrete_given_mass_fractions = .true.\n\
           num_accretion_species = 2\n\
           accretion_species_id(1) = 'h1'\n\
           accretion_species_id(2) = 'he4'\n\
           accretion_species_xa(2) = 1.0d0\n\
           xa_central_lower_limit_species(1) = 'h1'\n\
           xa_central_lower_limit_species(2) = 'he4'\n\
           xa_central_lower_limit = 0.15d0\n\
           xa_central_lower_limit(1) = 0.1d0\n\
           xa_central_lower_limit(2) = 0.05d0\n\
           xa_central_upper_limit_species(2) = 'he4'\n\
           xa_central_upper_limit(2) = 0.5d0\n\
           xa_surface_lower_limit_species(9) = 'o16'\n\
           xa_surface_lower_limit(9) = 0.15d0\n\
           xa_average_upper_limit_species(9) = 'c12'\n\
           xa_average_upper_limit(9) = 0.8d0\n\
           read_extra_controls_inlist(1) = .true.\n\
           extra_controls_inlist_name(1) = 'inlist_extra_xa'\n\
         /\n",
    )
    .expect("write main xa inlist");
    fs::write(
        &extra_p,
        "&controls\n\
           xa_central_lower_limit(1) = 0.2d0\n\
         /\n",
    )
    .expect("write extra xa inlist");

    let bundle = InlistBundle::from_file(&main_p).expect("load sparse xa");
    assert_eq!(
        bundle.controls.xa_central_lower_limit_species,
        ["h1", "he4"]
    );
    assert_eq!(bundle.controls.xa_central_lower_limit.len(), 2);
    assert!((bundle.controls.xa_central_lower_limit[0] - 0.2).abs() < 1.0e-15);
    assert!((bundle.controls.xa_central_lower_limit[1] - 0.05).abs() < 1.0e-15);

    assert_eq!(bundle.controls.xa_central_upper_limit_species, ["", "he4"]);
    assert_eq!(bundle.controls.xa_central_upper_limit.len(), 2);
    assert!((bundle.controls.xa_central_upper_limit[0] - 0.0).abs() < 1.0e-15);
    assert!((bundle.controls.xa_central_upper_limit[1] - 0.5).abs() < 1.0e-15);

    let ts = &bundle.controls.ts_stop;
    assert_eq!(ts.xa_surface_lower_limit_species.len(), 9);
    assert_eq!(ts.xa_surface_lower_limit_species[0], "");
    assert_eq!(ts.xa_surface_lower_limit_species[8], "o16");
    assert_eq!(ts.xa_surface_lower_limit.len(), 9);
    assert!((ts.xa_surface_lower_limit[8] - 0.15).abs() < 1.0e-15);
    assert_eq!(ts.xa_average_upper_limit_species[8], "c12");
    assert_eq!(ts.xa_average_upper_limit.len(), 9);
    assert!((ts.xa_average_upper_limit[8] - 0.8).abs() < 1.0e-15);

    // Sparse accretion_species_xa(2) = 1.0 stays aligned at slot 2 (`he4`).
    assert_eq!(bundle.controls.accretion_species_id, ["h1", "he4"]);
    assert_eq!(bundle.controls.accretion_species_xa.len(), 2);
    assert!((bundle.controls.accretion_species_xa[0] - 0.0).abs() < 1.0e-15);
    assert!((bundle.controls.accretion_species_xa[1] - 1.0).abs() < 1.0e-15);

    let mut extras = DefaultStarExtras;
    let model = init_or_update_model_from_bundle(&bundle, None, &mut extras)
        .expect("init accretion model");
    let xa_acc = get_xa_for_accretion(&model).expect("accretion xa");
    let ih1 = model
        .iso_names
        .iter()
        .position(|n| n == "h1")
        .expect("h1 slot");
    let ihe4 = model
        .iso_names
        .iter()
        .position(|n| n == "he4")
        .expect("he4 slot");
    assert!((xa_acc[ih1] - 0.0).abs() < 1.0e-15);
    assert!((xa_acc[ihe4] - 1.0).abs() < 1.0e-15);
}

/// Verifies `NUM_XA_LIMITS = 9` bounds (`> 9` and `0`) and explicit slice
/// subscript rejection across `xa_*` abundance-limit arrays.
#[test]
fn test_147_xa_bounds_and_slice_rejection() {
    let tmp = TempCaseGuard::new("xa_bounds");
    for (idx, body) in [
        "&controls\n  xa_central_lower_limit(10) = 0.1d0\n/\n",
        "&controls\n  xa_central_upper_limit_species(10) = 'he4'\n/\n",
        "&controls\n  xa_surface_lower_limit(10) = 0.1d0\n/\n",
        "&controls\n  xa_average_upper_limit(10) = 0.8d0\n/\n",
        "&controls\n  xa_central_lower_limit(0) = 0.1d0\n/\n",
        "&controls\n  xa_surface_lower_limit(1:2) = 0.15d0\n/\n",
        "&controls\n  xa_average_upper_limit(:2) = 0.8d0\n/\n",
        "&controls\n  xa_central_lower_limit(:) = 0.1d0\n/\n",
    ]
    .into_iter()
    .enumerate()
    {
        let bad_p = tmp.root.join(format!("inlist_xa_bad_{idx}"));
        fs::write(&bad_p, body).expect("write bad xa");
        let err = InlistBundle::from_file(&bad_p)
            .expect_err("invalid xa subscript or slice must fail loudly");
        assert!(matches!(err, StarError::InvalidInput(_)));
    }
}

/// Verifies `predictive_*` and `diffusion_isolve_solver` `trim_end` semantics
/// (trailing blanks accepted; leading blanks and wrong case rejected) and
/// `num_accretion_species > 250` loading (upstream reads it as a scalar,
/// `ctrls_io.f90:1408`; the error is deferred to `get_xa_for_accretion`,
/// `DECISIONS #43`, `#566`).
#[test]
fn test_147_trim_end_and_num_accretion_species_bound() {
    let tmp = TempCaseGuard::new("trim_acc");

    let ok_p = tmp.root.join("inlist_ok");
    fs::write(
        &ok_p,
        "&controls\n\
           predictive_mix(1) = .true.\n\
           predictive_zone_type(1) = 'burn_H  '\n\
           predictive_zone_loc(1) = 'core  '\n\
           predictive_bdy_loc(1) = 'top  '\n\
           diffusion_isolve_solver = 'ros2_solver  '\n\
           num_accretion_species = 250\n\
         /\n",
    )
    .expect("write ok inlist");
    let b = InlistBundle::from_file(&ok_p).expect("valid trim_end & 250");
    assert!(b.controls.mix_wind.predictive_sets[0].predictive_mix);
    assert_eq!(b.controls.mix_wind.num_accretion_species, 250);
    assert_eq!(b.controls.diffusion.diffusion_isolve_solver, "ros2_solver");

    // Upstream copies `diffusion_isolve_solver` unchecked at load
    // (`star/private/ctrls_io.f90:1807`) and rejects unknown names only when
    // the isolve path runs (`star/private/diffusion.f90:482`).
    for (idx, body) in [
        "&controls\n  diffusion_isolve_solver = ' ros2_solver'\n/\n",
        "&controls\n  diffusion_isolve_solver = 'ROS2_SOLVER'\n/\n",
    ]
    .into_iter()
    .enumerate()
    {
        let p = tmp.root.join(format!("inlist_solver_{idx}"));
        fs::write(&p, body).expect("write solver inlist");
        let b = InlistBundle::from_file(&p).expect("loads unchecked");
        let name = &b.controls.diffusion.diffusion_isolve_solver;
        assert!(
            flexmesa_star::diffusion::IsolveSolver::from_name(name).is_err(),
            "leading blank or wrong case must fail at use"
        );
    }

    for (idx, body) in [
        "&controls\n  predictive_mix(1) = .true.\n  predictive_zone_type(1) = ' burn_H'\n/\n",
        "&controls\n  predictive_mix(1) = .true.\n  predictive_zone_loc(1) = ' core'\n/\n",
        "&controls\n  predictive_mix(1) = .true.\n  predictive_bdy_loc(1) = ' top'\n/\n",
    ]
    .into_iter()
    .enumerate()
    {
        let bad_p = tmp.root.join(format!("inlist_bad_{idx}"));
        fs::write(&bad_p, body).expect("write bad inlist");
        let err = InlistBundle::from_file(&bad_p)
            .expect_err("leading blank, wrong case, or >250 must fail");
        assert!(matches!(err, StarError::InvalidInput(_)));
    }

    // `num_accretion_species = 251` loads (upstream scalar read,
    // `ctrls_io.f90:1408`); only the `get_xa_for_accretion` loop errors.
    let big_p = tmp.root.join("inlist_251");
    fs::write(&big_p, "&controls\n  num_accretion_species = 251\n/\n")
        .expect("write 251 inlist");
    let b = InlistBundle::from_file(&big_p).expect("251 species must load");
    assert_eq!(b.controls.mix_wind.num_accretion_species, 251);
}

/// Verifies that slice subscripts on `x_*_ctrl`, `&eos`/`&kap` arrays, and
/// `diffusion_class_*` remain explicitly rejected (no slice broadcasting).
#[test]
fn test_147_slice_subscripts_rejected_on_user_micro_and_diffusion_arrays() {
    let tmp = TempCaseGuard::new("reject_slices");

    for (idx, body) in [
        "&controls\n  x_ctrl(2:3) = 7.5d0\n/\n",
        "&controls\n  x_ctrl(:) = 1.0d0\n/\n",
        "&controls\n  x_integer_ctrl(:2) = 9\n/\n",
        "&controls\n  x_logical_ctrl(1:2) = .true.\n/\n",
        "&controls\n  x_character_ctrl(1:) = 'a'\n/\n",
        "&controls\n  diffusion_class_A_max(1:2) = 2.5d0\n/\n",
        "&eos\n  eos_ctrl(2:4) = 3.25d0\n/\n",
        "&kap\n  kap_ctrl(1:3) = 0.5d0\n/\n",
    ]
    .into_iter()
    .enumerate()
    {
        let p = tmp.root.join(format!("inlist_slice_{idx}"));
        fs::write(&p, body).expect("write slice inlist");
        let err = InlistBundle::from_file(&p)
            .expect_err("slice subscript must error explicitly");
        assert!(matches!(err, StarError::InvalidInput(_)));
    }
}
