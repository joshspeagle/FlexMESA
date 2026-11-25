// SPDX-License-Identifier: GPL-3.0-or-later
//! Zero-`$MESA_DIR` MESA `26.4.1` parity tests for timestep selection and
//! termination checks (`timestep.rs`, `timestep_limits.rs`, `termination.rs`;
//! tracking `#152` and `#170`).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::too_many_lines
)]

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{CLIGHT, LN10, M_SUN, R_SUN, SECDAY, SECYER};
use flexmesa_star::core_boundaries::compute_core_masses;
use flexmesa_star::report_utils::eval_astero_report_scalars;
use flexmesa_star::{
    do_check_limits, eval_core_infall_and_rebound_cms, eval_h211_factor,
    eval_varcontrol, filter_limit, load_zams_model, pick_next_timestep,
    set_hydro_vars, timestep_controller, LimitCheckOutcome, StarControls,
    StarJobControls, StarModel, StepReportContext, TimestepAction,
    TimestepLimitReason, PHASE_STARTING, PHASE_ZAMS,
};

fn zams_model(mass_msun: f64) -> StarModel {
    let controls = StarControls {
        initial_mass: mass_msun,
        initial_z: 0.02,
        zbase: 0.02,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(controls, StarJobControls::default()).expect("zams");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model.save_start_values().expect("save_start_values");
    model.save_old_values();
    model
}

#[test]
fn test_eval_varcontrol_and_h211_closed_form_oracles_zero_mesa_dir() {
    let mut model = StarModel::new(
        6,
        vec![1],
        StarJobControls::default(),
        StarControls::default(),
    );
    model.lnd_old = vec![2.0; 6];
    model.lnd = vec![2.0; 6];
    model.ln_r_old = vec![25.0; 6];
    model.ln_r = vec![25.0; 6];
    model.ln_t_old = vec![15.0; 6];
    let delta = 0.03125_f64;
    model.ln_t = vec![15.0 + delta; 6];

    // Upstream `eval_varcontrol` (`timestep.f90:2332-2357`): for `nz = 6`,
    // `sum_lnt` accumulates interior cells `k = 2, 3`, surface `k = 0`,
    // center `k = 5`, and `k = 1`, while omitting the `k = 4` dead store.
    let expected_sumscales = 2.0 + 15.0 + 25.0;
    let expected_vc = (5.0 * delta) / expected_sumscales / 18.0;
    let vc = eval_varcontrol(&model);
    assert!(
        (vc - expected_vc).abs() / expected_vc <= 1.0e-14,
        "expected varcontrol={expected_vc:.16e}, got {vc:.16e}"
    );

    // Closed-form arctangent limiter and H211 fixed point.
    assert_eq!(filter_limit(1.0), 1.0);
    let x = 1.4_f64;
    let expected_lim = 1.0 + 10.0 * flexmesa_math::atan((x - 1.0) / 10.0);
    assert!((filter_limit(x) - expected_lim).abs() <= 1.0e-15);

    assert_eq!(
        eval_h211_factor(1.0, 1.0, 100.0, 100.0, 1.2, 0.8),
        (1.0, false, false)
    );
    assert_eq!(
        eval_h211_factor(0.5, 0.0, 100.0, 100.0, 1.2, 0.8),
        (1.2, true, false)
    );

    let r_n = filter_limit(1.0 / 0.8);
    let r_prev = filter_limit(1.0 / 1.25);
    let expected_h211 = filter_limit(
        flexmesa_math::pow(r_n, 0.25)
            * flexmesa_math::pow(r_prev, 0.25)
            * flexmesa_math::pow(200.0 / 100.0, -0.25),
    );
    let (h211, hit_max, hit_min) =
        eval_h211_factor(0.8, 1.25, 200.0, 100.0, 1.5, 0.5);
    assert!(!hit_max && !hit_min);
    assert!((h211 - expected_h211).abs() <= 1.0e-15);

    // `pick_next_timestep` floors `varcontrol_target` at `1e-99` (`timestep.f90:2272`).
    model.controls.varcontrol_target = 1.0e-30;
    let rep = pick_next_timestep(&mut model, 1.2, 0.5, 0.0);
    let expected_ratio = expected_vc / 1.0e-30;
    assert!(
        (rep.dt_limit_ratio - expected_ratio).abs() / expected_ratio <= 1.0e-14,
        "expected ratio={expected_ratio:.16e}, got {:.16e}",
        rep.dt_limit_ratio
    );
}

#[test]
fn test_timestep_hold_zero_sentinel_does_not_skip_hard_limits_at_model_zero() {
    let mut model = zams_model(1.0);
    model.model_number = 0;
    model.doing_first_model_of_run = false;
    model.aux.step_hist.timestep_hold = 0;
    model.controls.ts_stop.never_skip_hard_limits = false;
    model
        .controls
        .ts_stop
        .delta_lgt_cntr_limit_only_after_near_zams = false;
    model.controls.ts_stop.delta_lgt_cntr_hard_limit = 0.001;

    let kc = model.nz - 1;
    model.ln_t[kc] = model.ln_t_start[kc] + 0.05;

    // Inactive sentinel (`timestep_hold == 0`) at `model_number == 0` must NOT
    // disable hard limits.
    let (act_zero, _) = timestep_controller(&mut model);
    assert!(
        matches!(act_zero, TimestepAction::Retry(ref msg) if msg.contains("delta_lgT_cntr")),
        "expected hard-limit Retry when timestep_hold == 0 at model_number == 0, got {act_zero:?}"
    );

    // An active hold (`timestep_hold == 1 >= model_number`) skips the hard limit
    // when `never_skip_hard_limits = false` and caps growth at `dt`.
    model.aux.step_hist.timestep_hold = 1;
    model.controls.ts_stop.delta_lgt_cntr_limit = -1.0;
    let (act_hold, rep_hold) = timestep_controller(&mut model);
    assert_eq!(act_hold, TimestepAction::KeepGoing);
    assert!(rep_hold.dt_next <= model.dt);

    // `never_skip_hard_limits = true` overrides `timestep_hold`.
    model.controls.ts_stop.never_skip_hard_limits = true;
    let (act_never_skip, _) = timestep_controller(&mut model);
    assert!(
        matches!(act_never_skip, TimestepAction::Retry(ref msg) if msg.contains("delta_lgT_cntr")),
        "expected Retry with never_skip_hard_limits = true, got {act_never_skip:?}"
    );
}

#[test]
fn test_force_timestep_vs_force_timestep_years_and_hold_precedence() {
    let mut model = zams_model(1.0);
    model.model_number = 5;
    model.doing_first_model_of_run = false;
    model.dt = 100.0;
    model.controls.varcontrol_target = 1.0;
    model.controls.ts_stop.use_dt_low_pass_controller = false;
    model.controls.ts_stop.min_timestep_factor = 0.0;
    model.controls.ts_stop.max_timestep_factor = 1000.0;

    // `timestep.f90:49-69`: `timestep_controller` checks only `force_timestep > 0`;
    // `force_timestep_years` is applied at step start in `prepare_for_new_step`.
    model.controls.ts_stop.force_timestep = 0.0;
    model.controls.ts_stop.force_timestep_years = 10.0;
    model.controls.ts_stop.max_timestep = 50.0;
    let (act_cap, rep_cap) = timestep_controller(&mut model);
    assert_eq!(act_cap, TimestepAction::KeepGoing);
    assert_eq!(rep_cap.dt_next, 50.0);
    assert_eq!(rep_cap.reason, TimestepLimitReason::MaxTimestep);

    // Without `force_timestep > 0`, `timestep_hold > model_number` still caps `dt_next` at `dt`.
    model.controls.ts_stop.max_timestep = 0.0;
    model.aux.step_hist.timestep_hold = 10;
    let (act_hold, rep_hold) = timestep_controller(&mut model);
    assert_eq!(act_hold, TimestepAction::KeepGoing);
    assert_eq!(rep_hold.dt_next, 100.0);
    assert_eq!(rep_hold.reason, TimestepLimitReason::TimestepHold);

    // Explicit `force_timestep > 0` bypasses both `max_timestep` and `timestep_hold`.
    model.controls.ts_stop.max_timestep = 50.0;
    model.controls.ts_stop.force_timestep = 250.0;
    let (act_force, rep_force) = timestep_controller(&mut model);
    assert_eq!(act_force, TimestepAction::KeepGoing);
    assert_eq!(rep_force.dt_next, 250.0);
    assert_eq!(rep_force.reason, TimestepLimitReason::MaxTimestep);
}

#[test]
fn test_mdot_zero_guard_high_t_slice_and_tlim_tie_breaking_order() {
    let mut base = zams_model(1.0);
    base.model_number = 5;
    base.doing_first_model_of_run = false;
    base.dt = 100.0;
    base.controls.varcontrol_target = 1.0;
    base.controls.ts_stop.use_dt_low_pass_controller = false;
    base.controls.ts_stop.min_timestep_factor = 0.0;
    base.controls.ts_stop.max_timestep_factor = 0.0;

    // Zero `mdot == mdot_old` with zero tolerances must not yield `NaN` (`0/0`).
    let mut m_mdot = base.clone();
    m_mdot.controls.mix_wind.delta_mdot_atol = 0.0;
    m_mdot.controls.mix_wind.delta_mdot_rtol = 0.0;
    m_mdot.controls.mix_wind.delta_mdot_limit = 0.1;
    m_mdot.controls.mix_wind.delta_mdot_hard_limit = 0.2;
    m_mdot.mstar_dot = 0.0;
    m_mdot.aux.step_hist.mstar_dot_old = 0.0;
    let (act_z, rep_z) = timestep_controller(&mut m_mdot);
    assert_eq!(act_z, TimestepAction::KeepGoing);
    assert!(rep_z.dt_next.is_finite() && rep_z.dt_next > 0.0);

    // Exact Fortran left-to-right `(atol * M_SUN) / SECYER` denominator.
    m_mdot.controls.mix_wind.delta_mdot_atol = 1.0e-3;
    m_mdot.controls.mix_wind.delta_mdot_rtol = 0.0;
    m_mdot.controls.mix_wind.delta_mdot_limit = 0.5;
    m_mdot.controls.mix_wind.delta_mdot_hard_limit = -1.0;
    m_mdot.mstar_dot = -2.0e-3 * M_SUN / SECYER;
    m_mdot.aux.step_hist.mstar_dot_old = 0.0;
    let (act_m, rep_m) = timestep_controller(&mut m_mdot);
    assert_eq!(act_m, TimestepAction::KeepGoing);
    assert_eq!(rep_m.reason, TimestepLimitReason::MassChange);
    let expected_denom = (1.0e-3 * M_SUN) / SECYER;
    let expected_ratio =
        ((m_mdot.mstar_dot - 0.0).abs() / expected_denom) / 0.5;
    assert_eq!(rep_m.dt_limit_ratio, expected_ratio);

    // Trailing `ln_t` elements beyond `nz` must be ignored when checking `log_max_t`.
    let mut m_hight = base.clone();
    m_hight.controls.ts_stop.max_timestep_factor = 1.3;
    m_hight.controls.ts_stop.max_timestep_factor_at_high_t = 1.05;
    m_hight
        .controls
        .ts_stop
        .min_logt_for_max_timestep_factor_at_high_t = 9.0;
    m_hight.ln_t.push(10.0 * LN10);
    let (act_ht, rep_ht) = timestep_controller(&mut m_hight);
    assert_eq!(act_ht, TimestepAction::KeepGoing);
    assert!(
        (rep_ht.dt_next - 1.3 * m_hight.dt).abs() <= 1.0e-12,
        "expected non-high-T factor 1.3, got {}",
        rep_ht.dt_next / m_hight.dt
    );

    // When an in-range cell exceeds `min_logT_for_max_timestep_factor_at_high_T`,
    // `max_timestep_factor_at_high_t` engages.
    let mut m_hight2 = base.clone();
    m_hight2.controls.ts_stop.delta_lgt_limit = -1.0;
    m_hight2.controls.ts_stop.max_timestep_factor = 1.3;
    m_hight2.controls.ts_stop.max_timestep_factor_at_high_t = 1.05;
    m_hight2
        .controls
        .ts_stop
        .min_logt_for_max_timestep_factor_at_high_t = 9.0;
    m_hight2.ln_t[0] = 10.0 * LN10;
    m_hight2.ln_t_old[0] = 10.0 * LN10;
    m_hight2.ln_t_start[0] = 10.0 * LN10;
    let (act_ht2, rep_ht2) = timestep_controller(&mut m_hight2);
    assert_eq!(act_ht2, TimestepAction::KeepGoing);
    assert!(
        (rep_ht2.dt_next - 1.05 * m_hight2.dt).abs() <= 1.0e-12,
        "expected high-T factor 1.05, got {}",
        rep_ht2.dt_next / m_hight2.dt
    );

    // Tie-breaking parity (`star_data_def.inc:43-103`):
    // `Tlim_XH_cntr` < `Tlim_adjust_J_q` < `Tlim_error_in_energy_conservation`
    // using exact power-of-two ratio `0.25 / 0.125 == 2.0`.
    let mut m_tie = base;
    m_tie.aux.step_hist.total_energy_end = -1.0;
    m_tie.aux.step_hist.error_in_energy_conservation = 0.25;
    m_tie
        .controls
        .ts_stop
        .limit_for_rel_error_in_energy_conservation = 0.125;
    m_tie
        .controls
        .ts_stop
        .hard_limit_for_rel_error_in_energy_conservation = 1.0;
    let (act_e, rep_e) = timestep_controller(&mut m_tie);
    assert_eq!(act_e, TimestepAction::KeepGoing);
    assert_eq!(rep_e.dt_limit_ratio, 2.0);
    assert_eq!(rep_e.reason, TimestepLimitReason::EnergyConservation);

    // Add `r_jq == 2.0` -> `StructuralChange` beats `EnergyConservation`.
    m_tie.controls.rotation_flag = true;
    m_tie.controls.mix_wind.do_adjust_j_lost = true;
    m_tie.mstar_dot = -1.0e15;
    m_tie.controls.ts_stop.adjust_j_q_limit = 0.875;
    m_tie.controls.ts_stop.adjust_j_q_hard_limit = 0.5;
    m_tie.adjust_j_q = 0.75;
    let (act_jq, rep_jq) = timestep_controller(&mut m_tie);
    assert_eq!(act_jq, TimestepAction::KeepGoing);
    assert_eq!(rep_jq.dt_limit_ratio, 2.0);
    assert_eq!(rep_jq.reason, TimestepLimitReason::StructuralChange);

    // Add `r_cntr_x == 2.0` (`XH_cntr`, disabling `dX_div_X_cntr` and `lg_XH_cntr`)
    // -> `AbundanceChange` beats `StructuralChange`.
    let ih1 = m_tie
        .iso_names
        .iter()
        .position(|n| n == "h1")
        .expect("h1 slot");
    let c_off = (m_tie.nz - 1) * m_tie.species + ih1;
    m_tie.xa_old[c_off] = 0.75;
    m_tie.xa[c_off] = 0.50;
    m_tie.controls.ts_stop.delta_dx_div_x_cntr_limit = -1.0;
    m_tie.controls.ts_stop.delta_lg_xh_cntr_limit = -1.0;
    m_tie.controls.ts_stop.delta_xh_cntr_limit = 0.125;
    let (act_xh, rep_xh) = timestep_controller(&mut m_tie);
    assert_eq!(act_xh, TimestepAction::KeepGoing);
    assert_eq!(rep_xh.dt_limit_ratio, 2.0);
    assert_eq!(rep_xh.reason, TimestepLimitReason::AbundanceChange);
}

#[test]
fn test_termination_xa_invalid_species_empty_code_and_delta_pg_at_zero_l() {
    let base = zams_model(1.0);

    // Invalid species in any `xa_*` block terminates with empty `termination_code`
    // (`s% termination_code = 0` in `do_one_utils.f90:1273-1411`).
    for slot in 0..6 {
        let mut m_inv = base.clone();
        match slot {
            0 => {
                m_inv.controls.xa_central_lower_limit_species =
                    vec![String::from("not_an_isotope")];
                m_inv.controls.xa_central_lower_limit = vec![0.1];
            }
            1 => {
                m_inv.controls.xa_central_upper_limit_species =
                    vec![String::from("not_an_isotope")];
                m_inv.controls.xa_central_upper_limit = vec![0.1];
            }
            2 => {
                m_inv.controls.ts_stop.xa_surface_lower_limit_species =
                    vec![String::from("not_an_isotope")];
                m_inv.controls.ts_stop.xa_surface_lower_limit = vec![0.1];
            }
            3 => {
                m_inv.controls.ts_stop.xa_surface_upper_limit_species =
                    vec![String::from("not_an_isotope")];
                m_inv.controls.ts_stop.xa_surface_upper_limit = vec![0.1];
            }
            4 => {
                m_inv.controls.ts_stop.xa_average_lower_limit_species =
                    vec![String::from("not_an_isotope")];
                m_inv.controls.ts_stop.xa_average_lower_limit = vec![0.1];
            }
            _ => {
                m_inv.controls.ts_stop.xa_average_upper_limit_species =
                    vec![String::from("not_an_isotope")];
                m_inv.controls.ts_stop.xa_average_upper_limit = vec![0.1];
            }
        }
        let out_inv =
            do_check_limits(&mut m_inv, None, false).expect("invalid iso");
        assert!(
            matches!(
                out_inv,
                LimitCheckOutcome::Terminate { ref code, ref reason }
                    if code.is_empty() && reason.contains("not_an_isotope")
            ),
            "slot {slot}: expected empty termination_code for invalid species, got {out_inv:?}"
        );
    }

    // Valid nuclide absent from active net cycles without error (`do_one_utils.f90:1282`).
    let mut m_abs = base.clone();
    m_abs.controls.xa_central_lower_limit_species = vec![String::from("fe56")];
    m_abs.controls.xa_central_lower_limit = vec![0.1];
    assert_eq!(
        do_check_limits(&mut m_abs, None, false).expect("absent iso"),
        LimitCheckOutcome::KeepGoing
    );

    // When `L_phot == 0`, `do_one_utils.f90:658-670` zeros `delta_nu`, while
    // `delta_Pg` from `report.f90:353-363` remains evaluated.
    let mut m_ast = base;
    m_ast.controls.calculate_brunt_n2 = true;
    m_ast.brunt_n2.fill(1.0);
    let mut ctx = StepReportContext::compute(&m_ast);
    ctx.phot.l_lsun = 0.0;
    let (_, _, _, dpg_ref) = eval_astero_report_scalars(&m_ast, &ctx.phot);
    assert!(dpg_ref > 0.0 && dpg_ref.is_finite());

    m_ast.controls.ts_stop.delta_pg_lower_limit = 2.0 * dpg_ref;
    let out_pg =
        do_check_limits(&mut m_ast, Some(&ctx), false).expect("delta_Pg");
    assert!(
        matches!(out_pg, LimitCheckOutcome::Terminate { ref code, .. } if code == "delta_Pg_lower_limit"),
        "expected delta_Pg_lower_limit even with L_phot == 0, got {out_pg:?}"
    );

    m_ast.controls.ts_stop.delta_nu_lower_limit = 1.0;
    let out_nu =
        do_check_limits(&mut m_ast, Some(&ctx), false).expect("delta_nu");
    assert!(
        matches!(out_nu, LimitCheckOutcome::Terminate { ref code, .. } if code == "delta_nu_lower_limit"),
        "expected delta_nu == 0 <= 1.0 when L_phot == 0, got {out_nu:?}"
    );
}

#[test]
fn test_termination_cascade_max_model_number_phase_first_model_and_redos() {
    let base = zams_model(1.0);

    // 1. `max_model_number`: KeepGoing below limit, Terminate at and above limit.
    let mut m_mod = base.clone();
    m_mod.controls.max_model_number = Some(10);
    m_mod.model_number = 9;
    assert_eq!(
        do_check_limits(&mut m_mod, None, false).expect("mod 9"),
        LimitCheckOutcome::KeepGoing
    );
    for n in [10_usize, 11_usize] {
        m_mod.model_number = n;
        assert!(matches!(
            do_check_limits(&mut m_mod, None, false).expect("mod limit"),
            LimitCheckOutcome::Terminate { ref code, .. } if code == "max_model_number"
        ));
    }

    // 2. Cascade precedence (`do_one_utils.f90:713-811`):
    //    `max_abs_rel_run_E_err` > `retry_for_v_above_clight` > `max_age` >
    //    `omega_div_omega_crit_limit` > `max_model_number` > `max_period_number`.
    let mut m_cas = base.clone();
    m_cas.controls.rsp_flag = true;
    m_cas.controls.rsp_max_num_periods = 5;
    m_cas.aux.rsp.num_periods = 5;
    assert!(matches!(
        do_check_limits(&mut m_cas, None, false).expect("periods"),
        LimitCheckOutcome::Terminate { ref code, .. } if code == "reached max number of periods"
    ));

    m_cas.controls.max_model_number = Some(1);
    m_cas.model_number = 1;
    assert!(matches!(
        do_check_limits(&mut m_cas, None, false).expect("max_mod beats period"),
        LimitCheckOutcome::Terminate { ref code, .. } if code == "max_model_number"
    ));

    m_cas.controls.rotation_flag = true;
    m_cas.controls.ts_stop.omega_div_omega_crit_limit = 0.5;
    let wc0 = flexmesa_star::eval_upstream_omega_crit(&m_cas, 0);
    m_cas.omega[0] = wc0;
    assert!(matches!(
        do_check_limits(&mut m_cas, None, false).expect("omega beats max_mod"),
        LimitCheckOutcome::Terminate { ref code, .. } if code == "max_omega_div_omega_crit"
    ));

    m_cas.controls.max_age = 1.0;
    m_cas.star_age = 1.0;
    m_cas.aux.step_hist.time_sec = SECYER;
    assert!(matches!(
        do_check_limits(&mut m_cas, None, false).expect("max_age beats omega"),
        LimitCheckOutcome::Terminate { ref code, .. } if code == "max_age"
    ));

    m_cas.controls.max_age = 1.0e36;
    m_cas.controls.ts_stop.max_age_in_days = 1.0;
    m_cas.star_age = 2.0 * SECDAY / SECYER;
    m_cas.aux.step_hist.time_sec = 2.0 * SECDAY;
    m_cas.controls.max_age_in_seconds = Some(1.0);
    assert!(matches!(
        do_check_limits(&mut m_cas, None, false).expect("max_age_in_days beats sec"),
        LimitCheckOutcome::Terminate { ref code, .. } if code == "max_age"
    ));

    m_cas.controls.v_flag = true;
    m_cas.controls.ts_stop.retry_for_v_above_clight = true;
    m_cas.v[0] = 1.1 * CLIGHT;
    assert!(matches!(
        do_check_limits(&mut m_cas, None, false).expect("clight beats age"),
        LimitCheckOutcome::Retry { .. }
    ));

    m_cas.controls.ts_stop.max_abs_rel_run_e_err = 0.1;
    m_cas.aux.total_energy = -1.0e48;
    m_cas.aux.cumulative_energy_error = 2.0e47;
    assert!(matches!(
        do_check_limits(&mut m_cas, None, false).expect("rel_run_E_err first"),
        LimitCheckOutcome::Terminate { ref code, .. }
            if code == "exceeded max abs rel_run_E_err"
    ));

    // 3. `doing_first_model_of_run = true` keeps `PHASE_STARTING`.
    let mut m_ph = base.clone();
    let m_tot_g: f64 = m_ph.dm.iter().sum();
    m_ph.eps_nuc.fill(10.0 * m_ph.lum[0] / m_tot_g);
    m_ph.controls.ts_stop.stop_at_phase_zams = true;
    assert_eq!(
        do_check_limits(&mut m_ph, None, true).expect("first model"),
        LimitCheckOutcome::KeepGoing
    );
    assert_eq!(m_ph.aux.step_hist.phase_of_evolution, PHASE_STARTING);
    assert!(matches!(
        do_check_limits(&mut m_ph, None, false).expect("second model"),
        LimitCheckOutcome::Terminate { ref code, .. } if code == "phase_ZAMS"
    ));
    assert_eq!(m_ph.aux.step_hist.phase_of_evolution, PHASE_ZAMS);

    // 4. `compare_to_target` target-hitting `Redo` vs `Terminate`.
    let mut m_tgt = base.clone();
    m_tgt.dt = 1000.0;
    m_tgt.controls.star_mass_min_limit = 0.9;
    m_tgt.controls.ts_stop.when_to_stop_atol = 1.0e-6;
    m_tgt.controls.ts_stop.when_to_stop_rtol = 1.0e-6;
    m_tgt.mstar = 0.8 * M_SUN;
    assert!(matches!(
        do_check_limits(&mut m_tgt, None, false).expect("overshoot"),
        LimitCheckOutcome::Redo { dt_new, .. } if (dt_new - 500.0).abs() <= 1.0e-12
    ));
    m_tgt.mstar = (0.9 - 1.0e-7) * M_SUN;
    assert!(matches!(
        do_check_limits(&mut m_tgt, None, false).expect("within tol"),
        LimitCheckOutcome::Terminate { ref code, .. } if code == "star_mass_min_limit"
    ));

    // 5. `fe_core_infall_limit` with uninitialized zero energies and disabled
    //    `hard_limit_for_rel_error_in_energy_conservation <= 0` must not terminate,
    //    whereas a satisfied relative energy error bound triggers `fe_core_infall_limit`.
    let cat = NuclideCatalog::builtin_ref();
    let ih1 = cat.get_nuclide_index("h1").expect("h1");
    let ife56 = cat.get_nuclide_index("fe56").expect("fe56");
    let nz_fe = 20_usize;
    let mut ctrl_fe = StarControls {
        initial_mass: 1.0,
        v_flag: true,
        zbase: 0.02,
        ..StarControls::default()
    };
    ctrl_fe.ts_stop.fe_core_boundary_si28_fraction = -1.0;
    let mut m_fe = StarModel::new(
        nz_fe,
        vec![ih1, ife56],
        StarJobControls::default(),
        ctrl_fe,
    );
    m_fe.teff = 5000.0;
    m_fe.dt = 10.0;
    for k in 0..nz_fe {
        let frac = (nz_fe - k) as f64 / (nz_fe as f64);
        m_fe.dq[k] = 1.0 / (nz_fe as f64);
        m_fe.q[k] = frac;
        m_fe.dm[k] = M_SUN / (nz_fe as f64);
        m_fe.m[k] = frac * M_SUN;
        m_fe.m_grav[k] = m_fe.m[k];
        m_fe.r[k] = frac * R_SUN;
        m_fe.rmid[k] = (frac - 0.5 / (nz_fe as f64)).max(1.0e-3) * R_SUN;
        m_fe.ln_r[k] = flexmesa_math::log(m_fe.r[k]);
        m_fe.ln_t[k] = 15.0;
        m_fe.t[k] = flexmesa_math::exp(15.0);
        m_fe.lnd[k] = 2.0;
        m_fe.rho[k] = flexmesa_math::exp(2.0);
        m_fe.peos[k] = 1.0e12;
        m_fe.pgas[k] = 1.0e12;
        m_fe.ln_peos[k] = flexmesa_math::log(1.0e12);
        m_fe.ln_pgas[k] = flexmesa_math::log(1.0e12);
        m_fe.gamma1[k] = 5.0 / 3.0;
        m_fe.gam[k] = 1.0;
        m_fe.ye[k] = 0.5;
        m_fe.csound[k] = 1.0e7;
        m_fe.opacity[k] = 1.0;
        m_fe.grav[k] = 1.0e4;
        m_fe.lum[k] = 1.0e33;
        if k < nz_fe / 2 {
            m_fe.cell_xa_mut(k)[0] = 1.0;
            m_fe.cell_xa_mut(k)[1] = 0.0;
            m_fe.xh_mass[k] = 1.0;
        } else {
            m_fe.cell_xa_mut(k)[0] = 0.0;
            m_fe.cell_xa_mut(k)[1] = 1.0;
            m_fe.v[k] = -5.0e7;
        }
    }
    let cores_fe = compute_core_masses(&m_fe);
    assert!(cores_fe.fe_core_mass > 0.1);
    let (fe_inf, _, _) = eval_core_infall_and_rebound_cms(&m_fe, &cores_fe);
    assert!((fe_inf - 5.0e7).abs() <= 1.0e-6);

    m_fe.aux.step_hist.total_energy_end = 0.0;
    m_fe.aux.step_hist.error_in_energy_conservation = 0.0;
    m_fe.controls
        .ts_stop
        .hard_limit_for_rel_error_in_energy_conservation = -1.0;
    assert_eq!(
        do_check_limits(&mut m_fe, None, false).expect("fe negative hard lim"),
        LimitCheckOutcome::KeepGoing
    );

    m_fe.aux.step_hist.total_energy_end = -1.0e48;
    m_fe.aux.step_hist.error_in_energy_conservation = 1.0e40;
    m_fe.controls
        .ts_stop
        .hard_limit_for_rel_error_in_energy_conservation = 1.0e-3;
    assert!(matches!(
        do_check_limits(&mut m_fe, None, false).expect("fe infall stops"),
        LimitCheckOutcome::Terminate { ref code, .. }
            if code == "fe_core_infall_limit"
    ));
}
