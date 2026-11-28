// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration invariants for `#618` single-star step-driver parity (`evolve.rs`):
//! - `force_timestep > 0` bypassing `max_years_for_timestep` and `rsp.max_timestep`
//!   before `dt_next_unclipped` capture (`timestep.f90:49-53`,
//!   `evolve.f90:2077-2143`).
//! - `ExtrasDirective::Terminate` and `LimitCheckOutcome::Terminate` ordering
//!   relative to `timestep_controller_with_extras`
//!   (`run_star_support.f90:210-265, 781`, `do_one_utils.f90:721`,
//!   `timestep.f90:132-138`).
//! - `star_evolve_step_part1` `!first_try` skipping `dt_next` checks and
//!   `max_timestep` clamping (`evolve.f90:282-469, 1811-1958, 1992-2034`).
//! - `implicit_mdot_loop` mass-loss redos not incrementing `num_redos`
//!   (`evolve.f90:605-690`).
//! - Loud errors for unported `#618` controls (`nonlocal_nico_decay_heat`,
//!   `use_other_pressure`, `solver_save_photo_call_number`).

#![forbid(unsafe_code)]
#![allow(clippy::float_cmp)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use flexmesa_const::{M_SUN, SECYER};
use flexmesa_star::evolve::{
    star_evolve_step, star_evolve_step_part1, star_evolve_step_part2,
    star_evolve_step_with_extras, star_finalize_terminated_step,
    StarStepAttemptOutcome,
};
use flexmesa_star::inlist::InlistBundle;
use flexmesa_star::inlist_runner::{
    run_inlist_stage, DefaultStarExtras, ExtrasDirective, StarExtras,
};
use flexmesa_star::{
    load_zams_model, set_hydro_vars, StarControls, StarError, StarJobControls,
    StarModel,
};

struct TempTestDir(PathBuf);

impl TempTestDir {
    fn new(label: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "flexmesa_618_evolve_{label}_{}_{n}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp test dir");
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempTestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn load_1m_zams_with(mut controls: StarControls) -> StarModel {
    controls.initial_mass = 1.0;
    controls.initial_z = 0.02;
    controls.zbase = 0.02;
    let mut model =
        load_zams_model(controls, StarJobControls::default()).expect("zams");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model
}

struct TerminateTlimProbe {
    term_after_check_calls: Option<usize>,
    tlim_is_redo: bool,
    tlim_ratio: f64,
    check_calls: AtomicUsize,
    other_tlim_calls: AtomicUsize,
    pre_term_dt_next_bits: AtomicU64,
}

impl TerminateTlimProbe {
    fn new(
        term_after_check_calls: Option<usize>,
        tlim_is_redo: bool,
        tlim_ratio: f64,
    ) -> Self {
        Self {
            term_after_check_calls,
            tlim_is_redo,
            tlim_ratio,
            check_calls: AtomicUsize::new(0),
            other_tlim_calls: AtomicUsize::new(0),
            pre_term_dt_next_bits: AtomicU64::new(0),
        }
    }
}

impl StarExtras for TerminateTlimProbe {
    fn extras_check_model(&mut self, m: &mut StarModel) -> ExtrasDirective {
        let idx = self.check_calls.fetch_add(1, Ordering::SeqCst);
        match self.term_after_check_calls {
            Some(n) if idx >= n => {
                self.pre_term_dt_next_bits
                    .store(m.dt_next.to_bits(), Ordering::SeqCst);
                ExtrasDirective::Terminate(String::from("custom_extras_term"))
            }
            _ => ExtrasDirective::KeepGoing,
        }
    }

    fn other_timestep_limit(
        &mut self,
        _m: &mut StarModel,
        _skip_hard_limit: bool,
        _dt: f64,
    ) -> (ExtrasDirective, f64) {
        self.other_tlim_calls.fetch_add(1, Ordering::SeqCst);
        let dir = if self.tlim_is_redo {
            ExtrasDirective::Redo
        } else {
            ExtrasDirective::KeepGoing
        };
        (dir, self.tlim_ratio)
    }
}

struct ImplicitMdotExtras {
    calls: AtomicUsize,
}

impl StarExtras for ImplicitMdotExtras {
    fn other_wind(
        &self,
        m: &StarModel,
        _l: f64,
        _ms: f64,
        _r: f64,
        _t: f64,
    ) -> Result<Option<f64>, StarError> {
        let idx = self.calls.fetch_add(1, Ordering::SeqCst);
        match idx {
            0 => Ok(Some(1.0e-8)),
            1 => Ok(Some(2.0e-8)),
            _ => Ok(Some((-m.mstar_dot * SECYER / M_SUN).max(0.0))),
        }
    }
}

#[test]
fn test_618_force_timestep_bypasses_max_years_and_rsp_max_dt() {
    let mut controls = StarControls {
        max_years_for_timestep: 0.25,
        ..StarControls::default()
    };
    controls.rsp.max_timestep = 0.5 * SECYER;
    controls.ts_stop.force_timestep = 100.0 * SECYER;

    let mut model = load_1m_zams_with(controls);
    model.dt_next = 10.0 * SECYER;
    model.controls.max_age = model.star_age + 120.0;

    let rep = star_evolve_step(&mut model, true, true, true)
        .expect("forced step must converge");
    assert!(rep.converged);
    assert!(
        (rep.dt - 100.0 * SECYER).abs() <= 1.0e-3,
        "expected dt = 100 yr, got {} s",
        rep.dt
    );
    assert!(
        (model.aux.step_hist.dt_next_unclipped - 100.0 * SECYER).abs()
            <= 1.0e-3 * SECYER,
        "force_timestep > 0 must bypass max_years_for_timestep and rsp.max_timestep before dt_next_unclipped (got {} yr)",
        model.aux.step_hist.dt_next_unclipped / SECYER
    );
    assert!(
        (model.dt_next - 20.0 * SECYER).abs() <= 1.0e-3 * SECYER,
        "apply_max_age_to_dt_next should still clip dt_next to 20 yr, got {} yr",
        model.dt_next / SECYER
    );
    assert!((rep.dt_next - model.dt_next).abs() <= 1.0e-6);

    // Contrast: when force_timestep == 0, both caps still clamp dt_next_unclipped.
    let mut cap_controls = StarControls {
        max_years_for_timestep: 0.25,
        ..StarControls::default()
    };
    cap_controls.rsp.max_timestep = 0.5 * SECYER;
    let mut cap_model = load_1m_zams_with(cap_controls);
    cap_model.dt_next = 10.0 * SECYER;
    let cap_rep = star_evolve_step(&mut cap_model, true, true, true)
        .expect("capped step must converge");
    assert!(cap_rep.converged);
    assert!(
        cap_model.aux.step_hist.dt_next_unclipped <= 0.25 * SECYER + 1.0e-3,
        "expected unclipped dt_next <= 0.25 yr when force_timestep == 0, got {} yr",
        cap_model.aux.step_hist.dt_next_unclipped / SECYER
    );
}

#[test]
fn test_618_extras_terminate_runs_other_timestep_limit_without_overriding_termination(
) {
    let dir = TempTestDir::new("extras_term");
    let inlist_path = dir.path().join("inlist");
    fs::write(
        &inlist_path,
        "&star_job\n\
           load_saved_model = .false.\n\
         /\n\
         &kap\n\
           Zbase = 0.02d0\n\
         /\n\
         &controls\n\
           initial_mass = 1.0d0\n\
           initial_z = 0.02d0\n\
           use_other_timestep_limit = .true.\n\
           redo_limit = 1\n\
         /\n",
    )
    .expect("write temp inlist");

    // 1. 2-step run_inlist_stage: step 1 clears doing_first_model_of_run; step 2
    // terminates via ExtrasDirective::Terminate, invokes other_timestep_limit
    // (returning Redo), keeps step 1's dt_next untouched, and stops cleanly.
    let bundle = InlistBundle::from_file(&inlist_path).expect("parse inlist");
    let mut model = load_1m_zams_with(bundle.controls.clone());
    model.job = bundle.job.clone();
    let mut extras = TerminateTlimProbe::new(Some(1), true, 1.0);
    let report = run_inlist_stage(&mut model, &bundle, &mut extras, Some(3))
        .expect("ExtrasDirective::Terminate should stop cleanly");
    assert_eq!(report.steps_taken, 2);
    assert_eq!(report.termination_code, "custom_extras_term");
    assert_eq!(extras.other_tlim_calls.load(Ordering::SeqCst), 1);
    assert_eq!(model.num_redos, 0);
    assert_eq!(model.num_retries, 0);
    let pre_bits = extras.pre_term_dt_next_bits.load(Ordering::SeqCst);
    assert_ne!(pre_bits, 0);
    assert_eq!(
        model.dt_next.to_bits(),
        pre_bits,
        "other_timestep_limit Redo during Terminate must not overwrite dt_next"
    );

    // 2. Direct step on !doing_first_model_of_run with KeepGoing
    // other_timestep_limit applies dt_next scaling while keeping custom reason.
    let sentinel_dt_next = 1.0e4 * SECYER;
    let mut kg_controls = StarControls::default();
    kg_controls.ts_stop.use_other_timestep_limit = true;
    kg_controls.ts_stop.redo_limit = 1;
    let mut kg_model = load_1m_zams_with(kg_controls);
    kg_model.doing_first_model_of_run = false;
    kg_model.dt = 1.0 * SECYER;
    kg_model.dt_next = sentinel_dt_next;
    let mut kg_extras = TerminateTlimProbe::new(Some(0), false, 10.0);
    let kg_rep = star_evolve_step_with_extras(
        &mut kg_model,
        false,
        true,
        true,
        &mut kg_extras,
    )
    .expect("Terminate with KeepGoing other_timestep_limit must stop cleanly");
    assert_eq!(kg_extras.other_tlim_calls.load(Ordering::SeqCst), 1);
    assert_eq!(kg_model.num_redos, 0);
    assert_eq!(kg_model.num_retries, 0);
    assert!(
        kg_model.dt_next < sentinel_dt_next
            && kg_rep.dt_next < sentinel_dt_next,
        "KeepGoing other_timestep_limit must update dt_next (got {})",
        kg_model.dt_next
    );

    // 3. Hard-limit Retry in pick_next_timestep must not retry or overwrite
    // dt_next on ExtrasDirective::Terminate.
    let mut retry_controls = StarControls::default();
    retry_controls.ts_stop.use_other_timestep_limit = true;
    retry_controls.ts_stop.redo_limit = 1;
    retry_controls.ts_stop.varcontrol_dt_limit_ratio_hard_max = 1.0e-20;
    let mut retry_model = load_1m_zams_with(retry_controls);
    retry_model.doing_first_model_of_run = false;
    retry_model.dt_next = sentinel_dt_next;
    let mut retry_extras = TerminateTlimProbe::new(Some(0), true, 1.0);
    let rep = star_evolve_step_with_extras(
        &mut retry_model,
        false,
        true,
        true,
        &mut retry_extras,
    )
    .expect("ExtrasDirective::Terminate must beat hard-limit Retry");
    assert_eq!(rep.retries, 0);
    assert_eq!(retry_model.num_retries, 0);
    assert_eq!(retry_extras.other_tlim_calls.load(Ordering::SeqCst), 0);
    assert!((rep.dt - sentinel_dt_next).abs() <= 1.0e-6);
    assert!(
        (retry_model.dt_next - sentinel_dt_next).abs() <= 1.0e-6,
        "hard-limit Retry must not overwrite dt_next on ExtrasDirective::Terminate"
    );
}

#[test]
fn test_618_limit_check_terminate_runs_other_timestep_limit() {
    let sentinel_dt_next = 1.0e4 * SECYER;
    for (tlim_is_redo, tlim_ratio) in [(false, 10.0), (true, 1.0)] {
        let mut controls = StarControls {
            max_model_number: Some(0),
            ..StarControls::default()
        };
        controls.ts_stop.use_other_timestep_limit = true;
        let mut model = load_1m_zams_with(controls);
        model.doing_first_model_of_run = false;
        model.dt = 1.0 * SECYER;
        model.dt_next = sentinel_dt_next;
        model.controls.max_model_number = Some(model.model_number + 1);
        let mut extras =
            TerminateTlimProbe::new(None, tlim_is_redo, tlim_ratio);
        let _ = star_evolve_step_with_extras(
            &mut model,
            false,
            true,
            true,
            &mut extras,
        )
        .expect("LimitCheckOutcome::Terminate must finish cleanly");
        assert_eq!(extras.other_tlim_calls.load(Ordering::SeqCst), 1);
        assert_eq!(model.num_redos, 0);
        assert_eq!(
            model.aux.step_hist.termination_code.as_deref(),
            Some("max_model_number")
        );
        if tlim_is_redo {
            assert!((model.dt_next - sentinel_dt_next).abs() <= 1.0e-6);
        } else {
            assert!(model.dt_next < sentinel_dt_next);
        }
    }
}

#[test]
fn test_618_limit_check_terminate_preserves_code_and_dt_next() {
    let sentinel_dt_next = 1.0e4 * SECYER;

    // 1. max_model_number with hard-limit Retry in timestep_controller:
    // preserves termination_code, skips retries, leaves dt_next intact.
    let mut retry_controls = StarControls::default();
    retry_controls.ts_stop.varcontrol_dt_limit_ratio_hard_max = 1.0e-20;
    let mut retry_model = load_1m_zams_with(retry_controls);
    retry_model.dt_next = sentinel_dt_next;
    retry_model.controls.max_model_number = Some(retry_model.model_number + 1);
    let rep_retry = star_evolve_step(&mut retry_model, true, true, true)
        .expect("LimitCheckOutcome::Terminate must not retry");
    assert_eq!(rep_retry.retries, 0);
    assert_eq!(retry_model.num_retries, 0);
    assert_eq!(
        retry_model.aux.step_hist.termination_code.as_deref(),
        Some("max_model_number")
    );
    assert!((retry_model.dt_next - sentinel_dt_next).abs() <= 1.0e-6);

    // 2. max_model_number with KeepGoing timestep_controller updates dt_next.
    let ok_controls = StarControls {
        max_years_for_timestep: 0.25,
        ..StarControls::default()
    };
    let mut ok_model = load_1m_zams_with(ok_controls);
    ok_model.dt_next = sentinel_dt_next;
    ok_model.controls.max_model_number = Some(ok_model.model_number + 1);
    let rep_ok = star_evolve_step(&mut ok_model, true, true, true)
        .expect("normal LimitCheckOutcome::Terminate");
    assert_eq!(
        ok_model.aux.step_hist.termination_code.as_deref(),
        Some("max_model_number")
    );
    assert!(ok_model.dt_next <= 0.25 * SECYER + 1.0e-3);
    assert!((rep_ok.dt_next - ok_model.dt_next).abs() <= 1.0e-6);

    // 3. exceeded max abs rel_run_E_err skips timestep_controller entirely
    // (`do_one_utils.f90:721`, `run_star_support.f90:254`).
    let mut e_controls = StarControls::default();
    e_controls.ts_stop.max_abs_rel_run_e_err = 1.0e-30;
    let mut e_model = load_1m_zams_with(e_controls);
    e_model.dt_next = sentinel_dt_next;
    e_model.aux.cumulative_energy_error = 1.0e40;
    e_model.aux.step_hist.cumulative_energy_error_old = 1.0e40;
    let rep_e = star_evolve_step(&mut e_model, true, true, true)
        .expect("max_abs_rel_run_e_err must terminate cleanly");
    assert_eq!(
        e_model.aux.step_hist.termination_code.as_deref(),
        Some("exceeded max abs rel_run_E_err")
    );
    assert_eq!(rep_e.varcontrol, 0.0);
    assert!((e_model.dt_next - sentinel_dt_next).abs() <= 1.0e-6);

    // 4. star_finalize_terminated_step also avoids overwriting dt_next when
    // timestep_controller returns Retry.
    let mut fin_controls = StarControls::default();
    fin_controls.ts_stop.varcontrol_dt_limit_ratio_hard_max = 1.0e-20;
    let mut fin_model = load_1m_zams_with(fin_controls);
    fin_model.dt_next = sentinel_dt_next;
    let mut extras = DefaultStarExtras;
    let p1 =
        star_evolve_step_part1(&mut fin_model, true, true, true, &mut extras)
            .expect("part1");
    let conv = match star_evolve_step_part2(
        &mut fin_model,
        p1,
        true,
        true,
        true,
        &mut extras,
    )
    .expect("part2")
    {
        StarStepAttemptOutcome::Converged(c) => c,
        StarStepAttemptOutcome::Retry { reason, .. } => {
            panic!("unexpected solver retry: {reason}")
        }
    };
    let fin_rep = star_finalize_terminated_step(&mut fin_model, &conv, 0);
    assert!((fin_model.dt_next - sentinel_dt_next).abs() <= 1.0e-6);
    assert!((fin_rep.dt_next - sentinel_dt_next).abs() <= 1.0e-6);
}

#[test]
fn test_618_part1_not_first_try_skips_dt_next_checks_and_clamps() {
    let mut controls = StarControls {
        max_years_for_timestep: 1.0e-5,
        min_timestep_limit: 100.0,
        ..StarControls::default()
    };
    controls.ts_stop.max_timestep = 500.0;
    controls.rsp.max_timestep = 400.0;
    controls.ts_stop.force_timestep = 200.0;

    // 1. !first_try ignores dt_next = 0 and all max_timestep / force_timestep overrides.
    let mut m_zero_next = load_1m_zams_with(controls.clone());
    m_zero_next.dt = 1.0e6;
    m_zero_next.dt_next = 0.0;
    let mut extras = DefaultStarExtras;
    let state = star_evolve_step_part1(
        &mut m_zero_next,
        false,
        true,
        false,
        &mut extras,
    )
    .expect("!first_try must ignore dt_next = 0");
    assert_eq!(m_zero_next.dt, 1.0e6);
    match star_evolve_step_part2(
        &mut m_zero_next,
        state,
        false,
        true,
        true,
        &mut extras,
    )
    .expect("part2")
    {
        StarStepAttemptOutcome::Converged(c) => assert_eq!(c.dt, 1.0e6),
        StarStepAttemptOutcome::Retry { reason, .. } => {
            panic!("unexpected retry: {reason}")
        }
    }

    // 2. !first_try ignores dt_next < min_timestep_limit and preserves model.dt.
    let mut m_small_next = load_1m_zams_with(controls.clone());
    m_small_next.dt = 1.0e6;
    m_small_next.dt_next = 5.0;
    let state_small = star_evolve_step_part1(
        &mut m_small_next,
        false,
        true,
        false,
        &mut extras,
    )
    .expect("!first_try must ignore dt_next < min_timestep_limit");
    assert_eq!(m_small_next.dt, 1.0e6);
    let _ = star_evolve_step_part2(
        &mut m_small_next,
        state_small,
        false,
        true,
        true,
        &mut extras,
    )
    .expect("part2 small dt_next");

    // 3. !first_try with model.dt <= 0 passes part1 and terminates in part2.
    let mut m_zero_dt = load_1m_zams_with(controls);
    let start_mod = m_zero_dt.model_number;
    m_zero_dt.dt = 0.0;
    m_zero_dt.dt_next = 1.0e5;
    let state_zero =
        star_evolve_step_part1(&mut m_zero_dt, false, true, false, &mut extras)
            .expect("!first_try part1 should not reject on dt_next");
    let err = star_evolve_step_part2(
        &mut m_zero_dt,
        state_zero,
        false,
        true,
        true,
        &mut extras,
    )
    .expect_err("part2 must terminate on dt_try <= 0");
    assert!(
        matches!(&err, StarError::Terminated { reason, .. } if reason == "dt_is_zero"),
        "expected dt_is_zero termination from part2, got {err:?}"
    );
    assert_eq!(m_zero_dt.model_number, start_mod);
}

#[test]
fn test_618_mdot_redo_does_not_increment_num_redos() {
    let mut controls = StarControls {
        use_other_wind: true,
        max_tries_for_implicit_wind: 5,
        iwind_tolerance: 0.01,
        iwind_lambda: 0.5,
        ..StarControls::default()
    };
    controls.mix_wind.max_mdot_redo_cnt = 5;

    // 1. Single-star full step driver.
    let mut model = load_1m_zams_with(controls.clone());
    model.dt_next = 1.0e4 * SECYER;
    let mut extras = ImplicitMdotExtras {
        calls: AtomicUsize::new(0),
    };
    let rep =
        star_evolve_step_with_extras(&mut model, true, true, true, &mut extras)
            .expect("mdot redo step must converge");
    assert!(rep.converged);
    assert!(
        extras.calls.load(Ordering::SeqCst) >= 3,
        "expected at least one implicit mdot redo"
    );
    assert_eq!(
        model.num_redos, 0,
        "implicit_mdot_loop must not increment model.num_redos"
    );

    // 2. Split part1/part2 path.
    let mut split_model = load_1m_zams_with(controls);
    split_model.dt_next = 1.0e4 * SECYER;
    let mut split_extras = ImplicitMdotExtras {
        calls: AtomicUsize::new(0),
    };
    let p1 = star_evolve_step_part1(
        &mut split_model,
        true,
        true,
        true,
        &mut split_extras,
    )
    .expect("part1");
    let out = star_evolve_step_part2(
        &mut split_model,
        p1,
        true,
        true,
        true,
        &mut split_extras,
    )
    .expect("part2");
    assert!(matches!(out, StarStepAttemptOutcome::Converged(_)));
    assert!(split_extras.calls.load(Ordering::SeqCst) >= 3);
    assert_eq!(split_model.num_redos, 0);
}

#[test]
fn test_618_unported_evolve_controls_fail_loudly() {
    // 1. Load-time rejection for unported &controls keys (`inlist_inventory.rs`).
    let dir = TempTestDir::new("unported_controls");
    for key_line in [
        "use_other_pressure = .true.",
        "solver_save_photo_call_number = 5",
    ] {
        let key_name = key_line.split('=').next().expect("key").trim();
        let path = dir.path().join(format!("inlist_{key_name}"));
        fs::write(&path, format!("&star_job\n/\n&controls\n  {key_line}\n/\n"))
            .expect("write bad inlist");
        let err = InlistBundle::from_file(&path)
            .expect_err("unported control must fail at load");
        assert!(
            matches!(&err, StarError::InvalidInput(msg) if msg.contains(key_name)),
            "expected InvalidInput mentioning {key_name}, got {err:?}"
        );
    }

    // 2. Step-time rejection for nonlocal_nico_decay_heat outward deposit.
    let controls = StarControls {
        nonlocal_nico_decay_heat: true,
        ..StarControls::default()
    };

    let mut model = load_1m_zams_with(controls.clone());
    model.dt_next = 1.0e4 * SECYER;
    let start_mod = model.model_number;
    let err = star_evolve_step(&mut model, true, true, true)
        .expect_err("nonlocal_nico_decay_heat must error");
    assert!(
        matches!(&err, StarError::InvalidInput(msg) if msg.contains("nonlocal_nico_decay_heat")),
        "expected InvalidInput for nonlocal_nico_decay_heat, got {err:?}"
    );
    assert_eq!(model.num_retries, 0);
    assert_eq!(model.model_number, start_mod);

    let mut split_model = load_1m_zams_with(controls);
    split_model.dt_next = 1.0e4 * SECYER;
    let split_start_mod = split_model.model_number;
    let mut extras = DefaultStarExtras;
    let p1 =
        star_evolve_step_part1(&mut split_model, true, true, true, &mut extras)
            .expect("part1");
    let err_p2 = star_evolve_step_part2(
        &mut split_model,
        p1,
        true,
        true,
        true,
        &mut extras,
    )
    .expect_err("part2 nonlocal_nico_decay_heat must error");
    assert!(
        matches!(&err_p2, StarError::InvalidInput(msg) if msg.contains("nonlocal_nico_decay_heat")),
        "expected InvalidInput in part2, got {err_p2:?}"
    );
    assert_eq!(split_model.model_number, split_start_mod);
}
