// SPDX-License-Identifier: GPL-3.0-or-later
//! Issue `#619` invariant tests for the coupled binary step/timestep driver,
//! implicit RLOF retry/redo bounds, unported control rejections,
//! `adjust_mdots` accounting, and `xa_removed` accretion composition.

#![forbid(unsafe_code)]

use flexmesa_binary::{
    binary_dt_why_str, ensure_no_unported_binary_driver_controls,
    prepare_coupled_mdots, prepare_coupled_mdots_with_extras,
    set_accretion_composition, star_skips_hard_limits, BinaryCompanion,
    BinaryExtras, BinaryExtrasAction, BinaryInlistControls, BinarySystem,
    DefaultBinaryExtras, B_NUM_TLIM, B_TLIM_COMP, B_TLIM_DM, B_TLIM_ECC,
    B_TLIM_ENV, B_TLIM_JORB, B_TLIM_ROCHE, B_TLIM_SEP,
    MAX_NUM_ACCRETION_SPECIES,
};
use flexmesa_const::{MSUN, SECYER};
use flexmesa_star::{
    do_adjust_mass_step, load_zams_model, InlistLoadOptions, StarControls,
    StarError, StarJobControls, StarModel, TLIM_BINARY,
};

struct TempDir619 {
    root: std::path::PathBuf,
}

impl TempDir619 {
    fn new(tag: &str) -> Self {
        static SEQ: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let id = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "flexmesa_bin_619_{tag}_{}_{id}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("create temp 619 dir");
        Self { root }
    }
}

impl Drop for TempDir619 {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn detached_pm_binary(
    max_tries: i64,
    period_days: f64,
    donor_mass_msun: f64,
) -> BinarySystem {
    let donor = load_zams_model(
        StarControls {
            initial_mass: donor_mass_msun,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("ZAMS donor");
    let scheme = if max_tries > 0 {
        "roche_lobe"
    } else {
        "Ritter"
    };
    let bctrl = BinaryInlistControls {
        m1_msun: donor_mass_msun,
        m2_msun: 0.8,
        initial_period_in_days: period_days,
        mdot_scheme: String::from(scheme),
        max_tries_to_achieve: max_tries,
        terminate_if_initial_overflow: false,
        ..BinaryInlistControls::default()
    };
    BinarySystem::from_controls_point_mass(donor, bctrl)
        .expect("point-mass binary")
}

fn two_star_binary(
    m1_msun: f64,
    m2_msun: f64,
    period_days: f64,
    max_tries: i64,
) -> BinarySystem {
    let mut s1 = load_zams_model(
        StarControls {
            initial_mass: m1_msun,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("star 1 ZAMS");
    let mut s2 = load_zams_model(
        StarControls {
            initial_mass: m2_msun,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("star 2 ZAMS");
    s1.controls.use_other_wind = true;
    s2.controls.use_other_wind = true;
    let scheme = if max_tries > 0 {
        "roche_lobe"
    } else {
        "Ritter"
    };
    let bctrl = BinaryInlistControls {
        evolve_both_stars: true,
        m1_msun,
        m2_msun,
        initial_period_in_days: period_days,
        mdot_scheme: String::from(scheme),
        max_tries_to_achieve: max_tries,
        terminate_if_initial_overflow: false,
        ..BinaryInlistControls::default()
    };
    BinarySystem::from_controls_two_stars(s1, s2, bctrl)
        .expect("two-star binary")
}

#[test]
fn test_dt_why_str_binary_and_star_why_tlim_on_set_star_timesteps() {
    struct RetryWhyProbe {
        calls: usize,
        seen_str: Vec<String>,
        seen_tlim: Vec<usize>,
        seen_neg_x: Vec<bool>,
    }
    impl BinaryExtras for RetryWhyProbe {
        fn check_model(
            &mut self,
            bin: &mut BinarySystem,
        ) -> BinaryExtrasAction {
            self.calls += 1;
            self.seen_str.push(bin.dt_why_str_binary.clone());
            self.seen_tlim.push(bin.donor.why_tlim);
            self.seen_neg_x.push(bin.donor.why_tlim_neg_x);
            if self.calls == 1 {
                // Retry restores `pre_step_snap` (`why_tlim_neg_x = true`
                // and uncapped `dt_next`); top-of-loop `set_star_timesteps`
                // must re-apply `TLIM_BINARY` and clear `why_tlim_neg_x`.
                BinaryExtrasAction::Retry
            } else {
                BinaryExtrasAction::KeepGoing
            }
        }
    }

    assert_eq!(B_NUM_TLIM, B_TLIM_DM);
    assert_eq!(binary_dt_why_str(0), " ");
    assert_eq!(binary_dt_why_str(B_TLIM_COMP), "b_companion");
    assert_eq!(binary_dt_why_str(B_TLIM_ROCHE), "b_RL");
    assert_eq!(binary_dt_why_str(B_TLIM_JORB), "b_jorb");
    assert_eq!(binary_dt_why_str(B_TLIM_ENV), "b_envelope");
    assert_eq!(binary_dt_why_str(B_TLIM_SEP), "b_separation");
    assert_eq!(binary_dt_why_str(B_TLIM_ECC), "b_eccentricity");
    assert_eq!(binary_dt_why_str(B_TLIM_DM), "b_deltam");
    assert_eq!(binary_dt_why_str(B_NUM_TLIM + 1), " ");

    let mut sys = detached_pm_binary(0, 5.0, 1.0);
    sys.donor.why_tlim_neg_x = true;
    sys.donor.dt_next = 1.0e4 * SECYER;
    sys.max_timestep_sec = 2.0e3 * SECYER;
    sys.dt_why_reason = B_TLIM_JORB;
    let mut probe = RetryWhyProbe {
        calls: 0,
        seen_str: Vec::new(),
        seen_tlim: Vec::new(),
        seen_neg_x: Vec::new(),
    };
    let _ = sys
        .try_step_with_extras(500.0 * SECYER, &mut probe)
        .expect("step succeeds after outer retry");
    assert_eq!(probe.calls, 2);
    assert_eq!(probe.seen_str, vec!["b_jorb", "b_jorb"]);
    assert_eq!(probe.seen_tlim, vec![TLIM_BINARY, TLIM_BINARY]);
    assert_eq!(probe.seen_neg_x, vec![false, false]);
}

#[test]
fn test_fj_hard_retry_applies_j_reduction_after_dt_floor_check() {
    struct JRetryProbe {
        seen_dts: Vec<f64>,
    }
    impl BinaryExtras for JRetryProbe {
        fn check_model(
            &mut self,
            bin: &mut BinarySystem,
        ) -> BinaryExtrasAction {
            self.seen_dts.push(bin.donor.dt);
            if self.seen_dts.len() >= 2 {
                // Allow the retry attempt at `0.5 * dt_floor` to pass
                // `pick_binary_timestep` once we observed its trial `dt`.
                bin.ignore_hard_limits_this_step = true;
            }
            BinaryExtrasAction::KeepGoing
        }
    }

    // Steps 1-2 advance past `doing_first_model_of_run`, clear
    // `ignore_hard_limits_this_step`, and settle the initial ZAMS remesh.
    let mut sys = detached_pm_binary(0, 2.0, 1.0);
    for _ in 0..2 {
        let _ = sys
            .try_step_with_extras(500.0 * SECYER, &mut DefaultBinaryExtras)
            .expect("warmup step");
    }
    assert!(!sys.ignore_hard_limits_this_step);

    // `cur_dt = 500`, `retry_factor = 0.5 -> 250 > dt_floor = 100`, then
    // `dt_reduction_factor_for_j = 0.2` reduces trial 2 to `50 * SECYER`.
    sys.controls.fj = 1.0e-3;
    sys.controls.fj_hard = 1.0e-12;
    sys.donor.controls.min_timestep_limit = 100.0 * SECYER;
    sys.donor.controls.ts_stop.timestep_factor_for_retries = 0.5;
    sys.controls.dt_reduction_factor_for_j = 0.2;

    let mut sys_fail = sys.clone();
    let mut probe = JRetryProbe {
        seen_dts: Vec::new(),
    };
    let _ = sys
        .try_step_with_extras(500.0 * SECYER, &mut probe)
        .expect("fj_hard retry below dt_floor via j-factor must succeed");
    assert_eq!(probe.seen_dts.len(), 2);
    assert!((probe.seen_dts[0] - 500.0 * SECYER).abs() < 1.0e-6);
    assert!((probe.seen_dts[1] - 50.0 * SECYER).abs() < 1.0e-6);
    assert_eq!(sys.counters.num_retries, 1);
    assert_eq!(sys.counters.num_hard_limit_retries, 1);
    assert!(!sys.have_to_reduce_timestep_due_to_j);

    // Contrasting case: when `cur_dt * retry_factor <= dt_floor`,
    // `prepare_outer_retry` must reject immediately.
    let err = sys_fail
        .try_step_with_extras(150.0 * SECYER, &mut DefaultBinaryExtras)
        .expect_err("retry_factor hitting dt_floor must terminate");
    assert!(err.to_string().contains("exhausted outer timestep retries"));
}

#[test]
fn test_finish_accepted_step_picks_next_dt_after_extras_finish_step() {
    struct FinishMaxDtExtras;
    impl BinaryExtras for FinishMaxDtExtras {
        fn finish_step(
            &mut self,
            bin: &mut BinarySystem,
        ) -> BinaryExtrasAction {
            bin.max_timestep_sec = 42.0 * SECYER;
            bin.dt_why_reason = B_TLIM_ROCHE;
            BinaryExtrasAction::KeepGoing
        }
    }

    struct FinishStarDtExtras;
    impl BinaryExtras for FinishStarDtExtras {
        fn finish_step(
            &mut self,
            bin: &mut BinarySystem,
        ) -> BinaryExtrasAction {
            bin.donor.dt_next = 17.0 * SECYER;
            bin.max_timestep_sec = 100.0 * SECYER;
            BinaryExtrasAction::KeepGoing
        }
    }

    let mut sys1 = detached_pm_binary(0, 5.0, 1.0);
    let (dt1, reason1) = sys1
        .try_step_with_extras(500.0 * SECYER, &mut FinishMaxDtExtras)
        .expect("finish_step max_timestep");
    assert_eq!(dt1, 42.0 * SECYER);
    assert_eq!(reason1, B_TLIM_ROCHE);

    let mut sys2 = detached_pm_binary(0, 5.0, 1.0);
    let (dt2, reason2) = sys2
        .try_step_with_extras(500.0 * SECYER, &mut FinishStarDtExtras)
        .expect("finish_step donor dt_next");
    assert_eq!(dt2, 17.0 * SECYER);
    assert_eq!(reason2, B_TLIM_COMP);
}

#[test]
fn test_implicit_first_try_runs_unclamped_mtransfer_rate_before_clamp() {
    struct MdotProbe {
        seen_rates: Vec<f64>,
    }
    impl BinaryExtras for MdotProbe {
        fn check_model(
            &mut self,
            bin: &mut BinarySystem,
        ) -> BinaryExtrasAction {
            self.seen_rates.push(bin.step_mtransfer_rate);
            BinaryExtrasAction::KeepGoing
        }
    }

    let mut sys = detached_pm_binary(5, 0.25, 1.0);
    assert!(
        sys.rl_relative_gap[0] > 0.0,
        "short period should overflow RL for unclamped try-1 test"
    );
    let raw_rate = -1.0e-3 * MSUN / SECYER;
    let cap_msun_yr = 1.0e-5;
    let cap_rate = -cap_msun_yr * MSUN / SECYER;
    sys.controls.max_implicit_abs_mdot = cap_msun_yr;
    sys.mtransfer_rate = raw_rate;

    let mut probe = MdotProbe {
        seen_rates: Vec::new(),
    };
    let _ = sys
        .try_step_with_extras(SECYER, &mut probe)
        .expect("converges at -max_implicit_abs_mdot on try 2");
    assert_eq!(probe.seen_rates.len(), 2);
    assert_eq!(probe.seen_rates[0].to_bits(), raw_rate.to_bits());
    assert_eq!(probe.seen_rates[1].to_bits(), cap_rate.to_bits());
}

#[test]
fn test_implicit_redo_past_max_tries_and_redo_limit_termination() {
    struct MultiRedoExtras {
        calls: usize,
        stop_after: usize,
    }
    impl BinaryExtras for MultiRedoExtras {
        fn check_model(
            &mut self,
            _bin: &mut BinarySystem,
        ) -> BinaryExtrasAction {
            self.calls += 1;
            if self.calls <= self.stop_after {
                BinaryExtrasAction::Redo
            } else {
                BinaryExtrasAction::KeepGoing
            }
        }
    }

    let mut sys = detached_pm_binary(1, 5.0, 1.0);
    let mut extras = MultiRedoExtras {
        calls: 0,
        stop_after: 3,
    };
    let _ = sys
        .try_step_with_extras(SECYER, &mut extras)
        .expect("detached extras Redo past max_tries must stay redo");
    assert_eq!(extras.calls, 4);
    assert_eq!(sys.counters.num_retries, 0);
    assert_eq!(sys.counters.num_implicit_trials, 4);

    let mut sys_lim = detached_pm_binary(10, 5.0, 1.0);
    sys_lim.donor.controls.ts_stop.redo_limit = 2;
    let mut lim_extras = MultiRedoExtras {
        calls: 0,
        stop_after: 10,
    };
    let err = sys_lim
        .try_step_with_extras(SECYER, &mut lim_extras)
        .expect_err("exceeding redo_limit must terminate");
    match err {
        StarError::Terminated { reason, .. } => {
            assert_eq!(reason, "redo_limit");
        }
        other => panic!("expected Terminated(redo_limit), got {other:?}"),
    }
    assert_eq!(lim_extras.calls, 3);
}

#[test]
fn test_unported_binary_driver_controls_and_ce_flag_fail_explicitly() {
    let base_ctrl = BinaryInlistControls::default();
    assert!(
        ensure_no_unported_binary_driver_controls(&base_ctrl, false).is_ok()
    );

    let ce_err = ensure_no_unported_binary_driver_controls(&base_ctrl, true)
        .expect_err("CE_flag must error");
    assert!(ce_err.to_string().contains("CE_flag"));

    let keys = [
        "ce_begin_at_max_implicit_abs_mdot",
        "use_other_check_implicit_rlo",
        "use_other_implicit_function_to_solve",
        "use_other_rlo_mdot",
        "use_other_adjust_mdots",
        "use_other_mdot_edd",
        "use_other_accreted_material_j",
    ];
    let tmp = TempDir619::new("unported");
    for (idx, key) in keys.iter().enumerate() {
        let on_dir = tmp.root.join(format!("on_{idx}"));
        std::fs::create_dir_all(&on_dir).expect("mkdir");
        std::fs::write(
            on_dir.join("inlist"),
            format!("&binary_controls\n  {key} = .true.\n/\n"),
        )
        .expect("write inlist");
        let err = BinaryInlistControls::from_case_dir_with_options(
            &on_dir,
            InlistLoadOptions {
                allow_unsupported_controls: true,
            },
        )
        .expect_err("enabled unported control must fail even when allowed");
        assert!(matches!(err, StarError::InvalidInput(_)));
        assert!(
            err.to_string().to_ascii_lowercase().contains(key),
            "error for {key} missing name: {err}"
        );

        let off_dir = tmp.root.join(format!("off_{idx}"));
        std::fs::create_dir_all(&off_dir).expect("mkdir");
        std::fs::write(
            off_dir.join("inlist"),
            format!("&binary_controls\n  {key} = .false.\n/\n"),
        )
        .expect("write inlist");
        assert!(BinaryInlistControls::from_case_dir_with_options(
            &off_dir,
            InlistLoadOptions {
                allow_unsupported_controls: true,
            },
        )
        .is_ok());
    }

    let mut sys = detached_pm_binary(0, 5.0, 1.0);
    sys.ce_flag = true;
    let step_err = sys
        .try_step_with_extras(SECYER, &mut DefaultBinaryExtras)
        .expect_err("CE_flag step must fail loudly");
    assert!(step_err.to_string().contains("CE_flag"));
}

#[test]
fn test_implicit_wind_limit_skips_accretor_on_implicit_redo() {
    struct WindSkipProbe {
        wind_calls: [usize; 2],
        try1_winds: [usize; 2],
        check_calls: usize,
    }
    impl BinaryExtras for WindSkipProbe {
        fn other_wind_cgs(
            &mut self,
            _star: &StarModel,
            star_idx: usize,
            _ignore_rlof: bool,
        ) -> Result<Option<f64>, StarError> {
            self.wind_calls[star_idx - 1] += 1;
            Ok(Some(-1.0e-10 * MSUN / SECYER))
        }

        fn check_model(
            &mut self,
            bin: &mut BinarySystem,
        ) -> BinaryExtrasAction {
            self.check_calls += 1;
            if self.check_calls == 1 {
                self.try1_winds = self.wind_calls;
                if let BinaryCompanion::Star(ref mut s2) = bin.companion {
                    s2.aux.step_hist.was_in_implicit_wind_limit = true;
                }
                BinaryExtrasAction::Redo
            } else {
                BinaryExtrasAction::KeepGoing
            }
        }
    }

    let mut sys = two_star_binary(1.2, 1.0, 5.0, 5);
    let mut probe = WindSkipProbe {
        wind_calls: [0, 0],
        try1_winds: [0, 0],
        check_calls: 0,
    };
    let _ = sys
        .try_step_with_extras(10.0 * SECYER, &mut probe)
        .expect("step succeeds with accretor skip on try 2");
    assert_eq!(probe.check_calls, 2);
    // Try 1 evaluates `other_wind_cgs` for both stars (`part1` + coupled
    // mdot prep = 2 calls each); try 2 skips the accretor (`[+2, +0]`).
    assert_eq!(probe.try1_winds, [2, 2]);
    assert_eq!(probe.wind_calls, [4, 2]);
}

#[test]
fn test_wind_accreting_donor_swaps_part2_order_and_transfers_xa_removed() {
    struct WindSwapExtras;
    impl BinaryExtras for WindSwapExtras {
        fn other_wind_cgs(
            &mut self,
            _star: &StarModel,
            star_idx: usize,
            _ignore_rlof: bool,
        ) -> Result<Option<f64>, StarError> {
            let rate = if star_idx == 1 { -1.0e-12 } else { -1.0e-4 };
            Ok(Some(rate * MSUN / SECYER))
        }
    }

    let mut sys = two_star_binary(1.0, 2.0, 3.0, 0);
    sys.controls.do_wind_mass_transfer_2 = true;
    sys.controls.wind_bh_alpha_2 = 10.0;
    sys.controls.max_wind_transfer_fraction_2 = 0.9;

    // Enrich star 2 surface c12 so `s2.aux.xa_removed` is distinct from star 1.
    let c12_idx = sys
        .donor
        .iso_names
        .iter()
        .position(|n| n.eq_ignore_ascii_case("c12"))
        .expect("c12 iso");
    if let BinaryCompanion::Star(ref mut s2) = sys.companion {
        let sp = s2.species;
        for x in s2.xa.iter_mut().take(sp) {
            *x *= 0.9;
        }
        s2.xa[c12_idx] += 0.1;
        let sum0: f64 = s2.xa[..sp].iter().sum();
        for x in s2.xa.iter_mut().take(sp) {
            *x /= sum0;
        }
    }
    let donor_c12_before = sys.donor.cell_xa(0)[c12_idx];

    let _ = sys
        .try_step_with_extras(1.0e3 * SECYER, &mut WindSwapExtras)
        .expect("accretor-first swap populates s2.xa_removed for donor");
    assert!(sys.donor.mstar_dot > 0.0);
    assert!(sys.donor.cell_xa(0)[c12_idx] > donor_c12_before);
    assert_eq!(
        sys.donor.controls.accretion_species_xa.len(),
        sys.donor.species
    );
    let sum_acc: f64 = sys.donor.controls.accretion_species_xa.iter().sum();
    assert!((sum_acc - 1.0).abs() < 1.0e-12);
    if let BinaryCompanion::Star(ref s2) = sys.companion {
        assert_eq!(sys.donor.controls.accretion_species_xa, s2.aux.xa_removed);
    }
}

#[test]
fn test_adjust_mdots_unclamped_fixed_xfer_and_unconditional_do_j_accretion() {
    struct ZeroWind;
    impl BinaryExtras for ZeroWind {
        fn other_wind_cgs(
            &mut self,
            _star: &StarModel,
            star_idx: usize,
            _ignore_rlof: bool,
        ) -> Result<Option<f64>, StarError> {
            let rate = if star_idx == 1 { 0.0 } else { -1.0e-5 };
            Ok(Some(rate * MSUN / SECYER))
        }
    }

    let mut sys = two_star_binary(1.5, 1.2, 5.0, 0);
    sys.controls.mass_transfer_alpha = 0.6;
    sys.controls.mass_transfer_beta = 0.5;
    sys.controls.mass_transfer_delta = 0.1;
    sys.controls.do_j_accretion = true;
    let mtr = -1.0e-7 * MSUN / SECYER;

    let acc = match sys.companion {
        BinaryCompanion::Star(ref mut s2) => Some(&mut **s2),
        BinaryCompanion::PointMass { .. } => None,
    };
    let acct = prepare_coupled_mdots_with_extras(
        &sys.controls,
        &mut sys.donor,
        acc,
        None,
        sys.donor_i,
        sys.rl,
        sys.separation,
        sys.eccentricity,
        mtr,
        false,
        &mut ZeroWind,
    )
    .expect("prepare_coupled_mdots");

    // `fixed_xfer = 1 - 0.6 - 0.5 - 0.1 = -0.2` (unclamped).
    let expected_actual = mtr * (-0.2);
    let tol = 1.0e-12 * expected_actual.abs();
    assert!((acct.actual_mtransfer_rate - expected_actual).abs() < tol);
    assert!(acct.component_mdot[1] < 0.0);
    assert!(acct.accreted_material_j > 0.0);
    if let BinaryCompanion::Star(ref s2) = sys.companion {
        assert!(s2.controls.mix_wind.use_accreted_material_j);
        assert_eq!(
            s2.controls.mix_wind.accreted_material_j.to_bits(),
            acct.accreted_material_j.to_bits()
        );
    }

    sys.donor.controls.use_other_wind = false;
    let _ = prepare_coupled_mdots(
        &sys.controls,
        &mut sys.donor,
        None,
        Some((1.4 * MSUN, 1.4 * MSUN)),
        1,
        sys.rl,
        sys.separation,
        sys.eccentricity,
        0.0,
        true,
    )
    .expect("point-mass prepare_coupled_mdots");
}

#[test]
fn test_xa_removed_multi_cell_and_set_accretion_composition_checks() {
    let mut donor = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("donor ZAMS");
    let mut acc = donor.clone();
    let sp = donor.species;

    // Distinct composition in zone 0 vs zone 1 so average differs from cell 0.
    let mut x0 = vec![0.0; sp];
    let mut x1 = vec![0.0; sp];
    x0[0] = 0.8;
    x0[sp - 1] = 0.2;
    x1[0] = 0.2;
    x1[sp - 1] = 0.8;
    donor.xa[..sp].copy_from_slice(&x0);
    donor.xa[sp..2 * sp].copy_from_slice(&x1);

    let dm0 = donor.dm[0];
    let dm1_half = 0.5 * donor.dm[1];
    let removed = dm0 + dm1_half;
    let dt = 1.0e3 * SECYER;
    donor.mstar_dot = -removed / dt;
    do_adjust_mass_step(&mut donor, dt).expect("multi-cell mass removal");

    let exp0 = (dm0 * x0[0] + dm1_half * x1[0]) / removed;
    let exp_last = (dm0 * x0[sp - 1] + dm1_half * x1[sp - 1]) / removed;
    assert!((donor.aux.xa_removed[0] - exp0).abs() < 1.0e-14);
    assert!((donor.aux.xa_removed[sp - 1] - exp_last).abs() < 1.0e-14);
    assert!((donor.aux.xa_removed[0] - x0[0]).abs() > 0.05);

    set_accretion_composition(&mut donor, &mut acc, true)
        .expect("set accretion onto accretor");
    assert_eq!(acc.controls.accretion_species_xa, donor.aux.xa_removed);

    // Incompatible donor species vs accretor must fail during accretion instead
    // of silently skipping `accretion_species_*`.
    donor.iso_names = (0..sp).map(|j| format!("bogus_iso_{j}")).collect();
    donor.species = sp;
    set_accretion_composition(&mut donor, &mut acc, true)
        .expect("copies mismatched isotope list");
    acc.mstar_dot = 1.0e-8 * MSUN / SECYER;
    let acc_err = do_adjust_mass_step(&mut acc, dt)
        .expect_err("incompatible accretion species must fail in adjust_mass");
    assert!(matches!(acc_err, StarError::InvalidInput(_)));

    // Oversized network must fail `MAX_NUM_ACCRETION_SPECIES` check.
    donor.species = MAX_NUM_ACCRETION_SPECIES + 1;
    let max_err = set_accretion_composition(&mut donor, &mut acc, true)
        .expect_err("exceeding max_num_accretion_species must fail");
    assert!(matches!(max_err, StarError::InvalidInput(_)));
}

#[test]
fn test_star_skips_hard_limits_requires_positive_timestep_hold() {
    let mut star = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("ZAMS");
    star.controls.ts_stop.never_skip_hard_limits = false;
    star.model_number = 0;
    star.aux.step_hist.timestep_hold = 0;
    assert!(!star_skips_hard_limits(&star));

    star.model_number = 1;
    star.aux.step_hist.timestep_hold = 1;
    assert!(star_skips_hard_limits(&star));

    star.model_number = 2;
    assert!(!star_skips_hard_limits(&star));

    star.model_number = 1;
    star.controls.ts_stop.never_skip_hard_limits = true;
    assert!(!star_skips_hard_limits(&star));
}
