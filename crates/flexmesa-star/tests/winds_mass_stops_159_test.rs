// SPDX-License-Identifier: LGPL-3.0-or-later
//! Comprehensive Issue `#159` unit & integration tests covering:
//! 1. Super-Eddington wind (`0.5 * vesc2`, `Leff`, `prev_Ledd`, `max_wind` override).
//! 2. Wind schemes (`WindScheme::Other`, `"none"` rejection), `no_wind_if_no_rotation`,
//!    `max_T_center_for_{any,full}_mass_loss` taper, envelope/He-layer limits,
//!    `mass_change_full_{on,off}_dt` ramp, and `eval_rlo_wind` + `xfer_ratio`.
//! 3. Rotational mass-loss surface-average alignment (`v_div_v_crit_avg_surf`) and caps.
//! 4. Mass adjustment (`min_star_mass_for_loss`, `max_star_mass_for_gain`, `#263`
//!    crossing clamp, old-surface-`omega` default vs `use_accreted_material_j`,
//!    and `smooth_outer_xa_{big,small}`).
//! 5. `get_xa_for_accretion`, `compute_power_categories` (`log_LH`), and `eps_mdot`.
//! 6. `set_phase_of_evolution` across all 13 `stop_at_phase_*` phases + `RELAX`/`STARTING`.
//! 7. `do_check_limits` across scalar, core, velocity, abundance, RSP, `#264`, and `#265`.
//! 8. `when_to_stop_{rtol,atol}` target-hitting redos, `redo_limit`, and `max_age`
//!    exact/graded approach (`F1-10`).

#![allow(
    clippy::cast_precision_loss,
    clippy::manual_midpoint,
    clippy::similar_names,
    clippy::too_many_lines
)]

use flexmesa_const::{
    MixingType, CLIGHT, LN10, L_SUN, M_SUN, R_SUN, SECDAY, SECYER,
    STANDARD_CGRAV,
};
use flexmesa_star::{
    check_termination_limits_with_history, compute_delta_m,
    compute_power_categories, do_adjust_mass_step, do_check_limits,
    do_set_mdot, do_set_mdot_with_extras, eval_bjorklund_wind,
    eval_blocker_wind, eval_cell_specific_total_energy,
    eval_core_infall_and_rebound_cms, eval_de_jager_wind, eval_dutch_wind,
    eval_min_gamma1, eval_nieuwenhuijzen_wind, eval_nugis_lamers_wind,
    eval_phase_of_evolution, eval_reimers_wind, eval_rlo_wind,
    eval_super_eddington_wind, eval_surf_avg_rotation_info,
    eval_surf_avg_w_div_w_crit, eval_van_loon_wind, eval_vink_wind,
    evaluate_star_history_row, get_ejecta_mass, get_remnant_mass,
    get_shock_mass_msun, get_xa_for_accretion, load_zams_model,
    parse_star_history_columns, run_inlist_stage, set_hydro_vars,
    set_phase_of_evolution, star_evolve, star_evolve_step, DutchLowTScheme,
    ExtrasDirective, InlistBundle, LimitCheckOutcome, MesaCellValue,
    StarControls, StarError, StarExtras, StarJobControls, StarModel,
    StepReportContext, WindScheme, PHASE_C_BURN, PHASE_HE_BURN, PHASE_IAMS,
    PHASE_NE_BURN, PHASE_O_BURN, PHASE_PREMS, PHASE_RELAX, PHASE_SI_BURN,
    PHASE_STARTING, PHASE_TACHEB, PHASE_TAMS, PHASE_TP_AGB, PHASE_WDCS,
    PHASE_ZACHEB, PHASE_ZAMS,
};

fn zams_model(mass_msun: f64) -> StarModel {
    let ctrl = StarControls {
        initial_mass: mass_msun,
        zbase: 0.02,
        initial_z: 0.02,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(ctrl, StarJobControls::default()).expect("load zams");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model.save_old_values();
    model
}

fn rel_err(actual: f64, expected: f64) -> f64 {
    let denom = expected.abs().max(1.0e-30);
    (actual - expected).abs() / denom
}

struct FixedOtherWindExtras {
    wind_msun_yr: f64,
}

impl StarExtras for FixedOtherWindExtras {
    fn other_wind(
        &self,
        _m: &StarModel,
        _l: f64,
        _ms: f64,
        _r: f64,
        _t: f64,
    ) -> Result<Option<f64>, StarError> {
        Ok(Some(self.wind_msun_yr))
    }
}

#[test]
fn test_super_eddington_wind_and_boost_use_prev_ledd_and_half_vesc2() {
    let mut model = zams_model(10.0);
    let l1 = model.lum[0];
    let m1 = model.m_grav[0];
    let r1 = model.r[0];
    let cgrav = model.cgrav[0];

    // Configure prev_Ledd below Leff so super-Eddington wind fires.
    let ledd_factor = 1.25;
    let prev_ledd = 0.25 * l1 / ledd_factor;
    model.controls.super_eddington_scaling_factor = 0.7;
    model.controls.mix_wind.super_eddington_wind_ledd_factor = ledd_factor;
    model.aux.step_hist.prev_ledd = Some(prev_ledd);

    // Perturb surface opacity wildly; eval_super_eddington_wind must use prev_Ledd,
    // NOT surface-cell 4*pi*c*G*M/kap[0].
    model.opacity[0] = 1.0e-12;
    let mdot_sedd = eval_super_eddington_wind(&model, l1, m1, r1, prev_ledd);
    let leff = l1 / ledd_factor;
    let vesc2 = 2.0 * cgrav * m1 / r1;
    let expected = 0.7 * (leff - prev_ledd) / (0.5 * vesc2);
    assert!(
        rel_err(mdot_sedd, expected) < 1.0e-13,
        "super-Eddington wind mismatch: got {mdot_sedd:.16e}, expected {expected:.16e}"
    );
    // Confirm 2x factor versus buggy `vesc2` denominator.
    let buggy = 0.7 * (leff - prev_ledd) / vesc2;
    assert!((mdot_sedd / buggy - 2.0).abs() < 1.0e-13);

    // Super-Eddington wind is stored in `super_eddington_wind_mdot` during
    // `do_set_mdot` and applied as an override capped by `max_wind` inside
    // `compute_delta_m` (`adjust_mass.f90:155-190`).
    model.controls.mix_wind.hot_wind_scheme = String::new();
    model.controls.cool_wind_rgb_scheme = String::new();
    model.controls.cool_wind_agb_scheme = String::new();
    model.controls.mix_wind.super_eddington_wind_max_boost = 1.0;
    model.controls.mix_wind.max_wind = 1.0e-5; // Msun/yr cap
    do_set_mdot(&mut model).expect("do_set_mdot");
    assert!(
        rel_err(model.aux.step_hist.super_eddington_wind_mdot, expected)
            < 1.0e-13
    );
    let dt = 1.0e5;
    let dm_capped = compute_delta_m(&mut model, dt).expect("compute_delta_m");
    let cap_cgs = 1.0e-5 * M_SUN / SECYER;
    assert!(
        rel_err(dm_capped, -cap_cgs * dt) < 1.0e-13,
        "expected max_wind cap {cap_cgs:.6e}, got {:.6e}",
        -dm_capped / dt
    );

    // Super-Eddington boost scales existing wind via prev_Ledd (`winds.f90:216-236`):
    // linear ramp from `1.0` at `full_off` to `max_boost` at `full_on`, saturating at `max_boost`.
    model.controls.super_eddington_scaling_factor = 0.0;
    model.controls.mix_wind.super_eddington_wind_max_boost = 5.0;
    model.controls.mix_wind.wind_boost_full_off_l_div_ledd = 1.0;
    model.controls.mix_wind.wind_boost_full_on_l_div_ledd = 2.0;
    model.controls.mix_wind.max_wind = 0.0;
    model.controls.mix_wind.hot_wind_scheme = String::from("other");
    model.controls.cool_wind_rgb_scheme = String::from("other");
    model.controls.cool_wind_agb_scheme = String::from("other");
    model.controls.use_other_wind = true;
    let base_cgs = -1.0e-6 * M_SUN / SECYER;
    let extras = FixedOtherWindExtras {
        wind_msun_yr: 1.0e-6,
    };
    // Sub-Eddington (`L / prev_Ledd = 0.5 <= full_off = 1.0`) -> boost = 1.0.
    model.aux.step_hist.prev_ledd = Some(2.0 * l1);
    let mdot_sub = do_set_mdot_with_extras(&mut model, &extras).expect("sub");
    assert!(rel_err(mdot_sub, base_cgs) < 1.0e-13);

    // Mid-ramp (`L / prev_Ledd = 1.5`) -> boost = 1 + (5 - 1) * (1.5 - 1) / (2 - 1) = 3.0.
    model.aux.step_hist.prev_ledd = Some(l1 / 1.5);
    let mdot_mid = do_set_mdot_with_extras(&mut model, &extras).expect("mid");
    assert!(
        rel_err(mdot_mid, 3.0 * base_cgs) < 1.0e-13,
        "mid-ramp boost mismatch: got {mdot_mid:.12e}, expected {:.12e}",
        3.0 * base_cgs
    );

    // Saturated (`L / prev_Ledd = 3.0 >= full_on = 2.0`) -> boost = max_boost = 5.0.
    model.aux.step_hist.prev_ledd = Some(l1 / 3.0);
    let mdot_sat = do_set_mdot_with_extras(&mut model, &extras).expect("sat");
    assert!(
        rel_err(mdot_sat, 5.0 * base_cgs) < 1.0e-13,
        "saturated boost mismatch: got {mdot_sat:.12e}, expected {:.12e}",
        5.0 * base_cgs
    );
}

#[test]
fn test_wind_schemes_gates_t_center_taper_envelope_limits_dt_ramp_and_rlo_wind() {
    // WindScheme::parse_name accepts "" and "other", rejects "none" and typos.
    assert_eq!(WindScheme::parse_name(""), Some(WindScheme::None));
    assert_eq!(WindScheme::parse_name("other"), Some(WindScheme::Other));
    assert_eq!(WindScheme::parse_name("OTHER"), Some(WindScheme::Other));
    assert_eq!(WindScheme::parse_name("none"), None);
    assert_eq!(WindScheme::parse_name("bogus"), None);

    let mut model = zams_model(5.0);
    model.controls.mix_wind.hot_wind_scheme = String::from("none");
    model.controls.cool_wind_rgb_scheme = String::from("none");
    model.controls.cool_wind_agb_scheme = String::from("none");
    assert!(
        matches!(do_set_mdot(&mut model), Err(StarError::InvalidInput(_))),
        "\"none\" must be rejected loudly without silent fallback"
    );

    // Blank scheme must return exact 0 even if reimers_scaling_factor > 0.
    model.controls.mix_wind.hot_wind_scheme = String::new();
    model.controls.cool_wind_rgb_scheme = String::new();
    model.controls.cool_wind_agb_scheme = String::new();
    model.controls.reimers_scaling_factor = 1.0;
    model.controls.dutch_scaling_factor = 1.0;
    assert_eq!(do_set_mdot(&mut model).expect("blank"), 0.0);

    // `WindScheme::Other` with `use_other_wind = true`.
    model.controls.mix_wind.hot_wind_scheme = String::from("other");
    model.controls.cool_wind_rgb_scheme = String::from("other");
    model.controls.cool_wind_agb_scheme = String::from("other");
    model.controls.use_other_wind = true;
    let extras = FixedOtherWindExtras {
        wind_msun_yr: 2.0e-7,
    };
    let base_cgs = -2.0e-7 * M_SUN / SECYER;
    let got_base = do_set_mdot_with_extras(&mut model, &extras).expect("other");
    assert!(rel_err(got_base, base_cgs) < 1.0e-13);

    // `no_wind_if_no_rotation`.
    model.controls.rotation_flag = false;
    model.controls.mix_wind.no_wind_if_no_rotation = true;
    assert_eq!(
        do_set_mdot_with_extras(&mut model, &extras).expect("no rot"),
        0.0
    );
    model.controls.mix_wind.no_wind_if_no_rotation = false;

    // `mass_change_full_on_dt` / `mass_change_full_off_dt` linear ramp.
    model.controls.mix_wind.mass_change_full_off_dt = 1.0e3;
    model.controls.mix_wind.mass_change_full_on_dt = 3.0e3;
    model.dt = 1.0e3;
    assert_eq!(
        do_set_mdot_with_extras(&mut model, &extras).expect("dt off"),
        0.0
    );
    model.dt = 2.0e3;
    let mid_dt = do_set_mdot_with_extras(&mut model, &extras).expect("dt mid");
    assert!(
        rel_err(mid_dt, 0.5 * base_cgs) < 1.0e-13,
        "dt midpoint ramp: got {mid_dt:.6e}, expected {:.6e}",
        0.5 * base_cgs
    );
    model.dt = 5.0e3;
    assert!(
        rel_err(
            do_set_mdot_with_extras(&mut model, &extras).expect("dt on"),
            base_cgs
        ) < 1.0e-13
    );
    model.controls.mix_wind.mass_change_full_off_dt = -1.0;
    model.controls.mix_wind.mass_change_full_on_dt = -1.0;

    // `max_T_center_for_any_mass_loss` / `max_T_center_for_full_mass_loss` taper.
    let tc = model.t[model.nz - 1];
    model.aux.step_hist.starting_t_center = Some(tc);
    model.controls.mix_wind.max_t_center_for_full_mass_loss = 0.5 * tc;
    model.controls.mix_wind.max_t_center_for_any_mass_loss = 1.5 * tc;
    let tc_mid = do_set_mdot_with_extras(&mut model, &extras).expect("tc mid");
    assert!(
        rel_err(tc_mid, 0.5 * base_cgs) < 1.0e-13,
        "T_center midpoint taper: got {tc_mid:.6e}, expected {:.6e}",
        0.5 * base_cgs
    );
    model.controls.mix_wind.max_t_center_for_any_mass_loss = 0.9 * tc;
    assert_eq!(
        do_set_mdot_with_extras(&mut model, &extras).expect("tc off"),
        0.0
    );
    model.controls.mix_wind.max_t_center_for_any_mass_loss = 1.0e99;
    model.controls.mix_wind.max_t_center_for_full_mass_loss = 1.0e99;

    // Envelope limit (`wind_H_envelope_limit`).
    model.controls.mix_wind.wind_h_envelope_limit = model.mstar / M_SUN + 1.0;
    assert_eq!(
        do_set_mdot_with_extras(&mut model, &extras).expect("env lim"),
        0.0
    );
    model.controls.mix_wind.wind_h_envelope_limit = 0.0;

    // Single-star RLO wind (`eval_rlo_wind`) above and below Roche lobe radius.
    let mut rlo_model = zams_model(2.0);
    rlo_model.controls.mix_wind.rlo_scaling_factor = 1.0;
    rlo_model.controls.mix_wind.rlo_wind_min_l = 0.0;
    rlo_model.controls.mix_wind.rlo_wind_max_teff = 1.0e6;
    rlo_model.controls.mix_wind.rlo_wind_base_mdot = 1.0e-8;
    rlo_model.controls.mix_wind.rlo_wind_scale_height = 0.1;
    rlo_model.controls.mix_wind.rlo_wind_roche_lobe_radius = 10.0;
    rlo_model.controls.mix_wind.roche_lobe_xfer_full_on = 0.8;
    rlo_model.controls.mix_wind.roche_lobe_xfer_full_off = 1.0;

    let mut xfer = 1.0;
    let w_below = eval_rlo_wind(&rlo_model, 10.0, 9.0, 5000.0, &mut xfer);
    assert_eq!(w_below, 0.0);
    // ratio = 0.9 -> f = (1.0 - 0.9)/(1.0 - 0.8) = 0.5 -> 0.5*(1 - cos(pi/2)) = 0.5
    assert!((xfer - 0.5).abs() < 1.0e-14, "xfer_ratio = {xfer}");

    let mut xfer_above = 1.0;
    let w_above = eval_rlo_wind(&rlo_model, 10.0, 10.2, 5000.0, &mut xfer_above);
    let expected_rlo = 1.0e-8 * flexmesa_math::exp((10.2 - 10.0) / 0.1);
    assert!(
        rel_err(w_above, expected_rlo) < 1.0e-13,
        "RLO wind mismatch: got {w_above:.16e}, expected {expected_rlo:.16e}"
    );
}

#[test]
fn test_rotational_mass_loss_surface_avg_alignment_and_caps() {
    let mut model = zams_model(10.0);
    model.controls.rotation_flag = true;
    model.controls.mix_wind.hot_wind_scheme = String::from("other");
    model.controls.cool_wind_rgb_scheme = String::from("other");
    model.controls.cool_wind_agb_scheme = String::from("other");
    model.controls.use_other_wind = true;
    let base_msun_yr = 1.0e-7;
    let extras = FixedOtherWindExtras {
        wind_msun_yr: base_msun_yr,
    };

    // Set a non-uniform outer omega profile so cell 0 differs from the
    // surface-averaged `v_div_v_crit_avg_surf`.
    let w_crit_0 = flexmesa_star::eval_upstream_omega_crit(&model, 0);
    for k in 0..model.nz {
        let w_c = flexmesa_star::eval_upstream_omega_crit(&model, k);
        model.omega[k] = if k == 0 { 0.15 * w_c } else { 0.65 * w_c };
    }
    assert!(model.omega[0] / w_crit_0 < 0.2);

    let surf_rot = eval_surf_avg_rotation_info(&model);
    let w_ratio_mdot_redo = eval_surf_avg_w_div_w_crit(&mut model);
    assert!(
        surf_rot.v_div_v_crit > 0.5,
        "expected surface average > 0.5, got {}",
        surf_rot.v_div_v_crit
    );
    assert!(w_ratio_mdot_redo > 0.5);

    // Compute exact expected Heger et al. (2000) boost from surface-averaged v/v_crit.
    let exp_xi = model.controls.mix_wind.mdot_omega_power;
    let boost_expected =
        flexmesa_math::pow(1.0 / (1.0 - surf_rot.v_div_v_crit), exp_xi);
    model.controls.mix_wind.max_rotational_mdot_boost = 1.0e4;
    model.controls.mix_wind.rotational_mdot_kh_fac = 1.0e10;
    model.controls.mix_wind.max_mdot_jump_for_rotation = 1.0e10;
    model.aux.step_hist.rotational_mdot_boost_old = 1.0;

    let mdot = do_set_mdot_with_extras(&mut model, &extras).expect("rot boost");
    let expected_cgs = -(base_msun_yr * boost_expected) * M_SUN / SECYER;
    assert!(
        rel_err(mdot, expected_cgs) < 1.0e-12,
        "rotational boost mismatch: got {mdot:.12e}, expected {expected_cgs:.12e}"
    );

    // Cap with `max_rotational_mdot_boost`.
    model.controls.mix_wind.max_rotational_mdot_boost = 1.2;
    let mdot_cap =
        do_set_mdot_with_extras(&mut model, &extras).expect("rot boost cap");
    assert!(rel_err(mdot_cap, -(base_msun_yr * 1.2) * M_SUN / SECYER) < 1.0e-12);
}

#[test]
fn test_adjust_mass_limits_crossing_clamp_263_omega_and_smooth_outer_xa() {
    // 1. `min_star_mass_for_loss` crossing-step clamp (reproducing #263 verbatim).
    let mut loss_model = zams_model(2.0);
    let m0 = loss_model.mstar;
    let m0_msun = m0 / M_SUN;
    let lim_msun = m0_msun - 1.0e-4;
    loss_model.controls.mix_wind.min_star_mass_for_loss = lim_msun;
    loss_model.mstar_dot = -1.0e-3 * M_SUN / SECYER;
    let dt = SECYER;
    let unclamped_mstar = m0 + loss_model.mstar_dot * dt;
    do_adjust_mass_step(&mut loss_model, dt).expect("adjust mass loss clamp");
    // Upstream clamps `mstar_dot = (lim*Msun - old_mstar)/dt` on the crossing
    // step, while omitting the `new_xmstar` update (`adjust_mass.f90:252-265`, #263).
    assert!(
        rel_err(loss_model.mstar_dot, (lim_msun * M_SUN - m0) / dt) < 1.0e-13,
        "crossing step should clamp mstar_dot to min_star_mass_for_loss"
    );
    assert!(
        rel_err(loss_model.mstar, unclamped_mstar) < 1.0e-14,
        "reproducing #263 verbatim: mstar = xmstar + M_center uses unclamped new_xmstar"
    );

    // Already at/below `min_star_mass_for_loss` -> zero delta_m on subsequent step.
    loss_model.mstar = (lim_msun - 1.0e-5) * M_SUN;
    loss_model.xmstar = loss_model.mstar - loss_model.m_center;
    loss_model.save_old_values();
    assert_eq!(
        compute_delta_m(&mut loss_model, dt).expect("below loss min"),
        0.0
    );

    // 2. `max_star_mass_for_gain` already above limit -> zero delta_m.
    let mut gain_model = zams_model(2.0);
    gain_model.controls.mix_wind.max_star_mass_for_gain =
        gain_model.mstar / M_SUN - 1.0e-4;
    gain_model.mstar_dot = 1.0e-3 * M_SUN / SECYER;
    assert_eq!(
        compute_delta_m(&mut gain_model, dt).expect("above gain max"),
        0.0
    );

    // 3. Accreted angular momentum default (`use_accreted_material_j = false`):
    //    uses `omega[k_below_just_added]` with verbatim upstream `set1_irot`
    //    (`adjust_mass.f90:1199-1205, 1232-1257`; `hydro_rotation.f90:55-226`),
    //    NOT old surface `j_rot`.
    let mut rot_acc = zams_model(2.0);
    rot_acc.controls.rotation_flag = true;
    rot_acc.controls.mix_wind.use_accreted_material_j = false;
    let omega_surf_old = 1.2e-5;
    rot_acc.omega.fill(omega_surf_old);
    flexmesa_star::update_rotation_distortion_factors(&mut rot_acc);
    let r0_before = flexmesa_math::exp(rot_acc.ln_r[0]);
    let m0_after = rot_acc.mstar + 5.0e-4 * M_SUN * 0.5;
    // Deliberately make `j_rot[0]` carry a stale/huge value while keeping
    // `omega = omega_surf_old` everywhere so copying old surface `j_rot[0]`
    // would produce a huge error whereas copying `omega[k_below_just_added]`
    // sets `j_rot[0] = i_rot * omega_surf_old`.
    rot_acc.j_rot[0] = 9.99e19;
    rot_acc.save_old_values();
    rot_acc.mstar_dot = 5.0e-4 * M_SUN / SECYER;
    do_adjust_mass_step(&mut rot_acc, 0.5 * SECYER).expect("accrete default omega");
    // Independent Paxton et al. (2019) `eval_i_rot(r0_before, w_roche)` check:
    // at slow rotation `w = omega / sqrt(G*M/r^3) << 1`, `rpsi/re ~ 1 - w^2/6`
    // so `w_roche ~ omega / sqrt(G*M/r^3)` and `i_rot ~ (2/3)*r0^2*(1 + (8/15)*w^2)`.
    // The neglected terms are of order `w^4` (`w ~ 0.028` here); the measured
    // difference is `5.8e-6`.
    let w_approx = omega_surf_old
        / (rot_acc.cgrav[0] * m0_after / (r0_before * r0_before * r0_before))
            .sqrt();
    let i_rot_approx = (2.0 / 3.0)
        * r0_before
        * r0_before
        * (1.0 + (8.0 / 15.0) * w_approx * w_approx);
    assert!(
        rel_err(rot_acc.j_rot[0], i_rot_approx * omega_surf_old) < 1.0e-5,
        "default accretion should give j_rot[0] = i_rot * omega_old ({:.6e}), got {:.6e}",
        i_rot_approx * omega_surf_old,
        rot_acc.j_rot[0]
    );
    assert!(
        (rot_acc.j_rot[0] - 9.99e19).abs() / 9.99e19 > 0.5,
        "default accretion must not copy stale old surface j_rot"
    );

    // 4. `do_adjust_j_lost = true` uses entry `model.j_rot` (NOT stale `j_rot_old`)
    //    and conserves `J_after == J_before - actual_j_lost` (`adjust_mass.f90:1476-1571`).
    let mut m_jloss = zams_model(2.0);
    m_jloss.controls.rotation_flag = true;
    m_jloss.controls.mix_wind.do_adjust_j_lost = true;
    m_jloss.controls.mix_wind.adjust_j_fraction = 1.0;
    m_jloss.controls.mix_wind.min_q_for_adjust_j_lost = 0.99;
    m_jloss.controls.mix_wind.min_j_div_delta_j = 10.0;
    // Set an inward-varying omega so starting_j_rot_surf differs from deeper j_rot,
    // and deliberately zero `j_rot_old` (step-1 state) to verify entry `model.j_rot` is used.
    for k in 0..m_jloss.nz {
        m_jloss.omega[k] = 1.0e-5 * (1.0 + 0.5 * (k as f64 / m_jloss.nz as f64));
    }
    flexmesa_star::update_rotation_distortion_factors(&mut m_jloss);
    m_jloss.aux.j_rot_old.fill(0.0);
    let starting_j_surf = m_jloss.j_rot[0];
    let dmbar_before = |m: &StarModel| -> f64 {
        let mut tot = 0.0_f64;
        let mut dm00 = 0.0_f64;
        for k in 0..m.nz {
            let dmm1 = dm00;
            dm00 = m.dm[k];
            let mut dmb = 0.5 * (dmm1 + dm00);
            if k + 1 == m.nz {
                dmb += 0.5 * dm00;
            }
            tot += dmb * m.j_rot[k];
        }
        tot
    };
    let j_tot_before = dmbar_before(&m_jloss);
    let dm_loss = 5.0e-4 * M_SUN;
    m_jloss.mstar_dot = -dm_loss / SECYER;
    do_adjust_mass_step(&mut m_jloss, SECYER).expect("adjust_j_lost");
    let j_tot_after = dmbar_before(&m_jloss);
    let expected_j_lost = dm_loss * starting_j_surf;
    assert!(
        rel_err(j_tot_before - j_tot_after, expected_j_lost) < 1.0e-3,
        "adjust_j_lost should remove ~{expected_j_lost:.6e}, removed {:.6e}",
        j_tot_before - j_tot_after
    );

    // 5. `smooth_outer_xa_big` / `smooth_outer_xa_small` conserve total species mass.
    let mut sm = zams_model(2.0);
    sm.controls.mix_wind.smooth_outer_xa_big = 0.05;
    sm.controls.mix_wind.smooth_outer_xa_small = 0.01;
    // Introduce a composition jump near the surface.
    let h1_idx = sm.iso_names.iter().position(|n| n == "h1").expect("h1");
    let he4_idx = sm.iso_names.iter().position(|n| n == "he4").expect("he4");
    for k in 0..5 {
        let mut x = sm.cell_xa(k).to_vec();
        let delta = 0.05 * x[h1_idx];
        x[h1_idx] -= delta;
        x[he4_idx] += delta;
        sm.cell_xa_mut(k).copy_from_slice(&x);
    }
    sm.save_old_values();
    sm.mstar_dot = 1.0e-6 * M_SUN / SECYER;
    let m_h1_before: f64 = (0..sm.nz)
        .map(|k| sm.dm[k] * sm.cell_xa(k)[h1_idx])
        .sum();
    let acc_h1 = sm.cell_xa(0)[h1_idx];
    let dm_add = sm.mstar_dot * (0.1 * SECYER);
    do_adjust_mass_step(&mut sm, 0.1 * SECYER).expect("smooth outer xa");
    let m_h1_after: f64 = (0..sm.nz)
        .map(|k| sm.dm[k] * sm.cell_xa(k)[h1_idx])
        .sum();
    assert!(
        rel_err(m_h1_after, m_h1_before + dm_add * acc_h1) < 1.0e-6,
        "smooth_outer_xa must conserve H1 mass"
    );
}

#[test]
fn test_get_xa_for_accretion_modes_power_categories_and_eps_mdot() {
    let mut model = zams_model(2.0);
    let h1_idx = model.iso_names.iter().position(|n| n == "h1").expect("h1");
    let he4_idx = model.iso_names.iter().position(|n| n == "he4").expect("he4");

    // 1. accrete_same_as_surface = true.
    model.controls.accrete_same_as_surface = true;
    let xa_surf = get_xa_for_accretion(&model).expect("same as surface");
    assert_eq!(xa_surf, model.cell_xa(0));

    // 2. accrete_given_mass_fractions valid & out-of-tolerance rejection.
    model.controls.accrete_same_as_surface = false;
    model.controls.accrete_given_mass_fractions = true;
    model.controls.mix_wind.num_accretion_species = 2;
    model.controls.accretion_species_id = vec![String::from("h1"), String::from("he4")];
    model.controls.accretion_species_xa = vec![0.75, 0.25];
    let xa_given = get_xa_for_accretion(&model).expect("given xa");
    assert!((xa_given[h1_idx] - 0.75).abs() < 1.0e-14);
    assert!((xa_given[he4_idx] - 0.25).abs() < 1.0e-14);

    model.controls.accretion_species_xa = vec![0.5, 0.1]; // sum != 1
    assert!(
        matches!(get_xa_for_accretion(&model), Err(StarError::InvalidInput(_))),
        "out-of-tolerance accretion_species_xa sum must error loudly"
    );

    // 3. Standard metals (`accretion_zfracs = 3` GS98) and custom (`accretion_zfracs = 0`).
    model.controls.accrete_given_mass_fractions = false;
    model.controls.accretion_h1 = 0.70;
    model.controls.accretion_he4 = 0.28;
    model.controls.mix_wind.accretion_h2 = 0.0;
    model.controls.mix_wind.accretion_he3 = 0.0;
    model.controls.mix_wind.accretion_zfracs = 3;
    let xa_gs98 = get_xa_for_accretion(&model).expect("GS98 metals");
    assert!((xa_gs98.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
    assert!((xa_gs98[h1_idx] - 0.70).abs() < 1.0e-12);
    assert!((xa_gs98[he4_idx] - 0.28).abs() < 1.0e-12);

    model.controls.mix_wind.accretion_zfracs = 0;
    model.controls.mix_wind.z_fraction_c = 0.25;
    model.controls.mix_wind.z_fraction_n = 0.10;
    model.controls.mix_wind.z_fraction_o = 0.65;
    let xa_custom = get_xa_for_accretion(&model).expect("custom z_fraction");
    assert!((xa_custom.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);

    // 4. `compute_power_categories` preserves `L_by_category` when `eps_nuc_factor == 0`
    //    and separates `power_h_burn = PP + CNO` from total nuclear power.
    let mut p_model = zams_model(5.0);
    let nz = p_model.nz;
    p_model.aux.eps_nuc_categories = vec![[0.0; flexmesa_chem::NUM_CATEGORIES]; nz];
    p_model.eps_nuc.fill(0.0);
    for k in 0..nz {
        p_model.aux.eps_nuc_categories[k][0] = 10.0; // pp
        p_model.aux.eps_nuc_categories[k][1] = 20.0; // cno
        p_model.aux.eps_nuc_categories[k][2] = 70.0; // tri_alpha
        p_model.eps_nuc[k] = 100.0;
    }
    p_model.controls.eps_nuc_factor = 0.0;
    let p_zero = compute_power_categories(&p_model);
    let m_tot_g: f64 = p_model.dm.iter().sum();
    let expected_h = 30.0 * m_tot_g / L_SUN;
    let expected_he = 70.0 * m_tot_g / L_SUN;
    assert!(
        rel_err(
            p_zero.l_by_category_lsun[0] + p_zero.l_by_category_lsun[1],
            expected_h
        ) < 1.0e-13
    );
    assert!(rel_err(p_zero.l_by_category_lsun[2], expected_he) < 1.0e-13);
    assert_eq!(p_zero.power_nuc_burn, 0.0);
    assert_eq!(p_zero.power_h_burn, 0.0);

    p_model.controls.eps_nuc_factor = 1.0;
    let p_one = compute_power_categories(&p_model);
    assert!(rel_err(p_one.power_h_burn, expected_h) < 1.0e-13);
    assert!(rel_err(p_one.power_he_burn, expected_he) < 1.0e-13);
    assert!(rel_err(p_one.power_nuc_burn, 100.0 * m_tot_g / L_SUN) < 1.0e-13);

    // 5. `eval_cell_specific_total_energy` includes face-averaged `0.25*(v_0^2 + v_1^2)`
    //    when `v_flag = true` and accounts for `mass_correction`.
    let mut v_model = zams_model(2.0);
    v_model.controls.v_flag = false;
    let e_no_v = eval_cell_specific_total_energy(&v_model, 0);
    v_model.controls.v_flag = true;
    v_model.v[0] = 1.0e7; // outer face != inner face to distinguish 0.25*(v0^2+v1^2) from 0.5*v0^2
    v_model.v[1] = 3.0e7;
    let e_with_v = eval_cell_specific_total_energy(&v_model, 0);
    let expected_ke = 0.25 * (1.0e14 + 9.0e14);
    assert!(
        rel_err(e_with_v - e_no_v, expected_ke) < 1.0e-12,
        "v_flag face-averaged KE mismatch: got {:.6e}, expected {expected_ke:.6e}",
        e_with_v - e_no_v
    );
    v_model.controls.use_mass_corrections = true;
    v_model.aux.mass_correction[0] = 1.02;
    let e_with_mc = eval_cell_specific_total_energy(&v_model, 0);
    assert!(
        (e_with_mc - e_with_v).abs() > 0.0,
        "use_mass_corrections = true must scale gravitational potential term"
    );
}

#[test]
fn test_set_phase_of_evolution_all_13_stop_phases_and_history_column() {
    let base = zams_model(5.0);
    let h1_idx = base.iso_names.iter().position(|n| n == "h1").expect("h1");
    let he4_idx = base.iso_names.iter().position(|n| n == "he4").expect("he4");

    // 1. doing_first_model_of_run -> PHASE_STARTING.
    let mut m = base.clone();
    assert_eq!(set_phase_of_evolution(&mut m, true), PHASE_STARTING);
    assert_eq!(m.aux.step_hist.phase_of_evolution, PHASE_STARTING);

    // 2. doing_relax -> PHASE_RELAX.
    m.aux.doing_relax = true;
    assert_eq!(set_phase_of_evolution(&mut m, false), PHASE_RELAX);

    // Helper to build a clean baseline with zero burning and controllable center X/Y.
    let make_clean = |h1_c: f64, he4_c: f64| -> StarModel {
        let mut s = base.clone();
        s.aux.doing_relax = false;
        s.eps_nuc.fill(0.0);
        s.aux.eps_nuc_categories =
            vec![[0.0; flexmesa_chem::NUM_CATEGORIES]; s.nz];
        for k in 0..s.nz {
            let mut x = vec![0.0; s.species];
            x[h1_idx] = h1_c;
            x[he4_idx] = he4_c;
            let rem = (1.0 - h1_c - he4_c).max(0.0);
            if s.species > 2 {
                let c_slot = (0..s.species)
                    .find(|&j| j != h1_idx && j != he4_idx)
                    .unwrap_or(0);
                x[c_slot] = rem;
            }
            s.cell_xa_mut(k).copy_from_slice(&x);
        }
        s
    };

    // 3. PreMS (`log_center_T > 5`, `Lnuc < L * lnuc_div_l_zams_limit`).
    let mut m_prems = make_clean(0.7, 0.28);
    m_prems.controls.ts_stop.stop_at_phase_prems = true;
    let out = do_check_limits(&mut m_prems, None, false).expect("prems");
    assert_eq!(m_prems.aux.step_hist.phase_of_evolution, PHASE_PREMS);
    assert!(matches!(out, LimitCheckOutcome::Terminate { ref code, .. } if code == "phase_PreMS"));

    // 4. ZAMS (`power_nuc_burn >= L * lnuc_div_l_zams_limit`, center_h1 > 0.3).
    let mut m_zams = make_clean(0.7, 0.28);
    let m_tot_g: f64 = m_zams.dm.iter().sum();
    let eps_target = 10.0 * m_zams.lum[0] / m_tot_g;
    m_zams.eps_nuc.fill(eps_target);
    m_zams.controls.ts_stop.stop_at_phase_zams = true;
    let out = do_check_limits(&mut m_zams, None, false).expect("zams");
    assert_eq!(m_zams.aux.step_hist.phase_of_evolution, PHASE_ZAMS);
    assert!(matches!(out, LimitCheckOutcome::Terminate { ref code, .. } if code == "phase_ZAMS"));

    // 5. IAMS (`center_h1 <= 0.3`, `> 1e-6`).
    let mut m_iams = make_clean(0.2, 0.78);
    m_iams.controls.ts_stop.stop_at_phase_iams = true;
    let out = do_check_limits(&mut m_iams, None, false).expect("iams");
    assert_eq!(m_iams.aux.step_hist.phase_of_evolution, PHASE_IAMS);
    assert!(matches!(out, LimitCheckOutcome::Terminate { ref code, .. } if code == "phase_IAMS"));

    // 6. TAMS (`center_h1 <= 1e-6`).
    let mut m_tams = make_clean(1.0e-7, 0.98);
    m_tams.controls.ts_stop.stop_at_phase_tams = true;
    let out = do_check_limits(&mut m_tams, None, false).expect("tams");
    assert_eq!(m_tams.aux.step_hist.phase_of_evolution, PHASE_TAMS);
    assert!(matches!(out, LimitCheckOutcome::Terminate { ref code, .. } if code == "phase_TAMS"));

    // 7. He_Burn (`cat_lum_lsun[CAT_3ALF] > 1e2`, `center_eps_burn[CAT_3ALF] <= 1e2`).
    let mut m_he = make_clean(0.0, 0.98);
    let eps_shell = 500.0 * L_SUN / m_he.dm[10];
    m_he.aux.eps_nuc_categories[10][2] = eps_shell;
    m_he.controls.ts_stop.stop_at_phase_he_burn = true;
    let out = do_check_limits(&mut m_he, None, false).expect("he_burn");
    assert_eq!(m_he.aux.step_hist.phase_of_evolution, PHASE_HE_BURN);
    assert!(matches!(out, LimitCheckOutcome::Terminate { ref code, .. } if code == "phase_He_Burn"));

    // 8. ZACHeB (`center_eps_burn[CAT_3ALF] > 1e2`).
    let mut m_zacheb = make_clean(0.0, 0.98);
    for k in (m_zacheb.nz - 5)..m_zacheb.nz {
        m_zacheb.aux.eps_nuc_categories[k][2] = 500.0;
    }
    m_zacheb.controls.ts_stop.stop_at_phase_zacheb = true;
    let out = do_check_limits(&mut m_zacheb, None, false).expect("zacheb");
    assert_eq!(m_zacheb.aux.step_hist.phase_of_evolution, PHASE_ZACHEB);
    assert!(matches!(out, LimitCheckOutcome::Terminate { ref code, .. } if code == "phase_ZACHeB"));

    // 9. TACHeB (`center_he4 <= 1e-4`, no He-burn convective region).
    let mut m_tacheb = make_clean(0.0, 1.0e-5);
    m_tacheb.mixing_type.fill(MixingType::NoMixing);
    m_tacheb.controls.ts_stop.stop_at_phase_tacheb = true;
    let out = do_check_limits(&mut m_tacheb, None, false).expect("tacheb");
    assert_eq!(m_tacheb.aux.step_hist.phase_of_evolution, PHASE_TACHEB);
    assert!(matches!(out, LimitCheckOutcome::Terminate { ref code, .. } if code == "phase_TACHeB"));

    // 10. TP_AGB (`center_he4 < 1e-4`, `he_core_mass - co_core_mass <= 0.1`, burn_he_conv_region).
    let mut m_tpagb = make_clean(0.0, 1.0e-5);
    m_tpagb.mixing_type.fill(MixingType::NoMixing);
    m_tpagb.xh_mass.fill(0.0);
    m_tpagb.yhe_mass.fill(0.5);
    for k in 5..10 {
        m_tpagb.mixing_type[k] = MixingType::ConvectiveMixing;
        m_tpagb.ln_t[k] = 8.1 * LN10; // > burn_he_mix_region_logt (7.9) and < burn_z (8.5)
        m_tpagb.t[k] = flexmesa_math::exp(m_tpagb.ln_t[k]);
    }
    m_tpagb.controls.stop_at_phase_tp_agb = true;
    let out = do_check_limits(&mut m_tpagb, None, false).expect("tp_agb");
    assert_eq!(m_tpagb.aux.step_hist.phase_of_evolution, PHASE_TP_AGB);
    assert!(matches!(out, LimitCheckOutcome::Terminate { ref code, .. } if code == "phase_TP_AGB"));

    // 11. C_Burn (`cat_lum_lsun[CAT_CC=16] > 1e2 && center_he4 < 1e-4`).
    let mut m_cburn = make_clean(0.0, 1.0e-5);
    let nz_last = m_cburn.nz - 1;
    m_cburn.aux.eps_nuc_categories[nz_last][16] =
        500.0 * L_SUN / m_cburn.dm[nz_last];
    m_cburn.controls.stop_at_phase_c_burn = true;
    let out = do_check_limits(&mut m_cburn, None, false).expect("c_burn");
    assert_eq!(m_cburn.aux.step_hist.phase_of_evolution, PHASE_C_BURN);
    assert!(matches!(out, LimitCheckOutcome::Terminate { ref code, .. } if code == "phase_C_Burn"));

    // 12. Ne_Burn (`cat_lum_lsun[CAT_BURN_NE=6] > 1e2 && center_he4 < 1e-4`).
    let mut m_ne = make_clean(0.0, 1.0e-5);
    m_ne.aux.eps_nuc_categories[nz_last][6] = 500.0 * L_SUN / m_ne.dm[nz_last];
    m_ne.controls.stop_at_phase_ne_burn = true;
    let out = do_check_limits(&mut m_ne, None, false).expect("ne_burn");
    assert_eq!(m_ne.aux.step_hist.phase_of_evolution, PHASE_NE_BURN);
    assert!(matches!(out, LimitCheckOutcome::Terminate { ref code, .. } if code == "phase_Ne_Burn"));

    // 13. O_Burn (`cat_lum_lsun[CAT_OO=18] + CAT_CO=17 > 1e2 && center_he4 < 1e-4`).
    let mut m_oburn = make_clean(0.0, 1.0e-5);
    m_oburn.aux.eps_nuc_categories[nz_last][18] =
        500.0 * L_SUN / m_oburn.dm[nz_last];
    m_oburn.controls.stop_at_phase_o_burn = true;
    let out = do_check_limits(&mut m_oburn, None, false).expect("o_burn");
    assert_eq!(m_oburn.aux.step_hist.phase_of_evolution, PHASE_O_BURN);
    assert!(matches!(out, LimitCheckOutcome::Terminate { ref code, .. } if code == "phase_O_Burn"));

    // 14. Si_Burn (`cat_lum_lsun[CAT_BURN_SI=9] > 1e2`).
    let mut m_si = make_clean(0.0, 1.0e-5);
    m_si.aux.eps_nuc_categories[nz_last][9] = 500.0 * L_SUN / m_si.dm[nz_last];
    m_si.controls.ts_stop.stop_at_phase_si_burn = true;
    let out = do_check_limits(&mut m_si, None, false).expect("si_burn");
    assert_eq!(m_si.aux.step_hist.phase_of_evolution, PHASE_SI_BURN);
    assert!(matches!(out, LimitCheckOutcome::Terminate { ref code, .. } if code == "phase_Si_Burn"));

    // 15. WDCS (`phot.logg > 6.0`).
    let mut m_wd = make_clean(0.0, 1.0e-5);
    let r_wd = 0.01 * R_SUN;
    for k in 0..m_wd.nz {
        m_wd.r[k] = r_wd * (1.0 - 0.5 * (k as f64 / m_wd.nz as f64));
        m_wd.ln_r[k] = flexmesa_math::log(m_wd.r[k]);
    }
    m_wd.controls.ts_stop.stop_at_phase_wdcs = true;
    let out = do_check_limits(&mut m_wd, None, false).expect("wdcs");
    assert_eq!(m_wd.aux.step_hist.phase_of_evolution, PHASE_WDCS);
    assert!(matches!(out, LimitCheckOutcome::Terminate { ref code, .. } if code == "phase_WDCS"));

    // Verify `phase_of_evolution` history column reports exact integer code.
    let specs = parse_star_history_columns("phase_of_evolution\n", None, &m_wd)
        .expect("parse phase_of_evolution column");
    let row = evaluate_star_history_row(&m_wd, &specs);
    assert_eq!(row.len(), 1);
    assert_eq!(row[0].1, MesaCellValue::Int(i64::from(PHASE_WDCS)));
    let ctx = StepReportContext::compute(&m_wd);
    assert_eq!(eval_phase_of_evolution(&m_wd, &ctx, false), PHASE_WDCS);
}

#[test]
fn test_do_check_limits_all_categories_and_deferred_bugs_264_265() {
    let base = zams_model(5.0);

    // 1. `Lnuc_div_L_zams_limit` inlist parsing does NOT auto-enable `stop_near_zams`.
    let tmp = std::env::temp_dir().join("flexmesa_159_zams_lim.inlist");
    std::fs::write(&tmp, "&controls\n  Lnuc_div_L_zams_limit = 0.4d0\n/\n")
        .expect("write");
    let bundle = InlistBundle::from_file(&tmp).expect("parse");
    let _ = std::fs::remove_file(&tmp);
    assert!(!bundle.controls.stop_near_zams);
    assert!((bundle.controls.lnuc_div_l_zams_limit - 0.4).abs() < 1.0e-15);

    // 2. `center_R_lower_limit` is compared against `R_center` in cm (NOT Rsun).
    let mut m_rc = base.clone();
    m_rc.r_center = 1.0e8; // cm
    m_rc.controls.center_r_lower_limit = Some(2.0e8); // cm
    let out = do_check_limits(&mut m_rc, None, false).expect("center_R");
    assert!(matches!(out, LimitCheckOutcome::Terminate { ref code, .. } if code == "center_R_lower_limit"));
    m_rc.controls.center_r_lower_limit = Some(0.5e8);
    assert_eq!(
        do_check_limits(&mut m_rc, None, false).expect("center_R ok"),
        LimitCheckOutcome::KeepGoing
    );

    // 3. `max_age_in_days` and `max_age_in_seconds` both emit `"max_age"`.
    let mut m_days = base.clone();
    m_days.aux.step_hist.time_sec = 5.0 * SECDAY;
    m_days.star_age = m_days.aux.step_hist.time_sec / SECYER;
    m_days.controls.ts_stop.max_age_in_days = 4.0;
    assert!(matches!(
        do_check_limits(&mut m_days, None, false).expect("days"),
        LimitCheckOutcome::Terminate { ref code, .. } if code == "max_age"
    ));

    let mut m_sec = base.clone();
    m_sec.aux.step_hist.time_sec = 100.0;
    m_sec.star_age = 100.0 / SECYER;
    m_sec.controls.max_age_in_seconds = Some(50.0);
    assert!(matches!(
        do_check_limits(&mut m_sec, None, false).expect("secs"),
        LimitCheckOutcome::Terminate { ref code, .. } if code == "max_age"
    ));

    // 4. `retry_for_v_above_clight` returns `LimitCheckOutcome::Retry` on signed `max_abs_vel > CLIGHT`
    //    (`do_one_utils.f90:726-736`), and does NOT retry on negative infall `v < -CLIGHT`.
    let mut m_v = base.clone();
    m_v.controls.v_flag = true;
    m_v.controls.ts_stop.retry_for_v_above_clight = true;
    m_v.v[0] = -1.1 * CLIGHT;
    assert_eq!(
        do_check_limits(&mut m_v, None, false).expect("v < -c"),
        LimitCheckOutcome::KeepGoing
    );
    m_v.v[0] = 1.1 * CLIGHT;
    assert!(matches!(
        do_check_limits(&mut m_v, None, false).expect("v > c"),
        LimitCheckOutcome::Retry { .. }
    ));

    // 5. Deferred upstream bug `#264`: `peak_burn_vconv_div_cs > 0.75 * limit`
    //    aborts loudly (`mesa_error` at `do_one_utils.f90:738-745`).
    let mut m_264 = base.clone();
    let k_pk = m_264.nz - 2;
    m_264.eps_nuc.fill(0.0);
    m_264.eps_nuc[k_pk] = 1.0e10;
    m_264.conv_vel[k_pk] = 0.8 * m_264.csound[k_pk];
    m_264.controls.ts_stop.peak_burn_vconv_div_cs_limit = 1.0;
    assert!(
        matches!(
            do_check_limits(&mut m_264, None, false),
            Err(StarError::InvalidInput(_))
        ),
        "upstream #264 aborts when peak_burn_vconv_div_cs > 0.75 * limit"
    );

    // 6. Deferred upstream bug `#265`: `v_surf_div_v_kh` uses mixed units
    //    `v_surf / (R_phot_Rsun / kh_timescale_yr)` (`do_one_utils.f90:635`).
    let mut m_265 = base.clone();
    m_265.controls.v_flag = true;
    m_265.v[0] = 1.0e5;
    let ctx_265 = StepReportContext::compute(&m_265);
    let mixed_ratio =
        m_265.v[0] / (ctx_265.phot.r_rsun / ctx_265.kh_timescale_yr.max(1.0e-99));
    m_265.controls.v_surf_div_v_kh_upper_limit = Some(0.5 * mixed_ratio);
    assert!(matches!(
        do_check_limits(&mut m_265, Some(&ctx_265), false).expect("265"),
        LimitCheckOutcome::Terminate { ref code, .. }
            if code == "v_surf_div_v_kh_upper_limit"
    ));

    // 7. `gamma1_limit` (respecting `gamma1_limit_max_q = 0.95`), `Pgas_div_P_limit`,
    //    `max_abs_rel_run_E_err`, `stop_when_reach_this_cumulative_extra_heating`,
    //    `RSP_GREKM_avg_abs_limit`.
    let mut m_g1 = base.clone();
    let k_mid = m_g1.nz / 2;
    assert!(m_g1.q[k_mid] <= m_g1.controls.output.gamma1_limit_max_q);
    m_g1.gamma1[k_mid] = 1.20;
    assert!((eval_min_gamma1(&m_g1) - 1.20).abs() < 1.0e-14);
    m_g1.controls.ts_stop.gamma1_limit = 1.30;
    assert!(matches!(
        do_check_limits(&mut m_g1, None, false).expect("gamma1"),
        LimitCheckOutcome::Terminate { ref code, .. } if code == "gamma1_limit"
    ));

    let mut m_pg = base.clone();
    m_pg.pgas[1] = 0.1 * m_pg.peos[1];
    m_pg.ln_pgas[1] = flexmesa_math::log(m_pg.pgas[1]);
    m_pg.controls.ts_stop.pgas_div_p_limit = 0.2;
    m_pg.controls.ts_stop.pgas_div_p_limit_max_q = 1.0;
    assert!(matches!(
        do_check_limits(&mut m_pg, None, false).expect("pgas_div_p"),
        LimitCheckOutcome::Terminate { ref code, .. } if code == "Pgas_div_P_limit"
    ));

    let mut m_eerr = base.clone();
    m_eerr.aux.total_energy = -1.0e48;
    m_eerr.aux.cumulative_energy_error = 2.0e47;
    m_eerr.controls.ts_stop.max_abs_rel_run_e_err = 0.1;
    assert!(matches!(
        do_check_limits(&mut m_eerr, None, false).expect("e_err"),
        LimitCheckOutcome::Terminate { ref code, .. }
            if code == "exceeded max abs rel_run_E_err"
    ));

    let mut m_eh = base.clone();
    m_eh.aux.step_hist.cumulative_extra_heating = 5.0e45;
    m_eh.controls.ts_stop.stop_when_reach_this_cumulative_extra_heating = 1.0e45;
    assert!(matches!(
        do_check_limits(&mut m_eh, None, false).expect("extra_heat"),
        LimitCheckOutcome::Terminate { ref code, .. }
            if code == "cumulative extra heating"
    ));

    let mut m_rsp = base.clone();
    m_rsp.controls.rsp_flag = true;
    m_rsp.aux.rsp.num_periods = 9;
    m_rsp.aux.rsp.grekm_avg_abs = 1.0e-8;
    m_rsp.controls.rsp.grekm_avg_abs_limit = 1.0e-6;
    assert_eq!(
        do_check_limits(&mut m_rsp, None, false).expect("rsp < 10"),
        LimitCheckOutcome::KeepGoing
    );
    m_rsp.aux.rsp.num_periods = 10;
    assert!(matches!(
        do_check_limits(&mut m_rsp, None, false).expect("rsp >= 10"),
        LimitCheckOutcome::Terminate { ref code, .. }
            if code == "RSP_GREKM_avg_abs_limit"
    ));

    // 8. `get_ejecta_mass`, `get_remnant_mass`, `get_shock_mass_msun`, and
    //    `eval_core_infall_and_rebound_cms`.
    let mut m_ej = base.clone();
    m_ej.controls.v_flag = true;
    m_ej.v.fill(0.0);
    // Make outer 5 zones unbound (0.5*v^2 > G*m/r - e) and supersonic shock at k=5.
    for v_k in m_ej.v.iter_mut().take(5) {
        *v_k = 1.0e9;
    }
    let ej = get_ejecta_mass(&m_ej);
    let rem = get_remnant_mass(&m_ej);
    assert!(ej > 0.0);
    assert!(rel_err(ej + rem, m_ej.m[0]) < 1.0e-13);
    m_ej.controls.ts_stop.ejecta_mass_max_limit = 0.5 * (ej / M_SUN);
    assert!(matches!(
        do_check_limits(&mut m_ej, None, false).expect("ejecta_mass"),
        LimitCheckOutcome::Terminate { ref code, .. }
            if code == "ejecta_mass_max_limit"
    ));
    m_ej.controls.ts_stop.ejecta_mass_max_limit = -1.0;
    m_ej.controls.remnant_mass_min_limit = 1.01 * (rem / M_SUN);
    assert!(matches!(
        do_check_limits(&mut m_ej, None, false).expect("remnant_mass"),
        LimitCheckOutcome::Terminate { ref code, .. }
            if code == "remnant_mass_min_limit"
    ));

    let mut m_sh = base.clone();
    m_sh.controls.v_flag = true;
    m_sh.v.fill(0.0);
    for k in 5..m_sh.nz {
        m_sh.v[k] = 2.0 * m_sh.csound[k];
    }
    let m_shock = get_shock_mass_msun(&m_sh);
    // Exact linear-interpolation oracle: Mach crosses 1.0 between k=4 (Mach=0) and k=5 (Mach=2)
    // at r_shock = 0.5*(r[4] + r[5]), so m_shock = 0.5*(m[4] + m[5]) / M_SUN.
    let expected_m_shock = 0.5 * (m_sh.m[4] + m_sh.m[5]) / M_SUN;
    assert!(
        rel_err(m_shock, expected_m_shock) < 1.0e-13,
        "shock mass mismatch: got {m_shock:.12e}, expected {expected_m_shock:.12e}"
    );

    // Core infall & rebound oracle (`report.f90:398-438`): construct a He core with
    // negative infall velocities inside the He core and verify `non_fe_infall == 2.5e8`.
    let mut m_inf = base.clone();
    m_inf.controls.v_flag = true;
    let h1_slot = m_inf.iso_names.iter().position(|n| n == "h1").expect("h1");
    let he4_slot = m_inf.iso_names.iter().position(|n| n == "he4").expect("he4");
    let k_core = m_inf.nz / 2;
    for k in k_core..m_inf.nz {
        let mut x = vec![0.0; m_inf.species];
        x[he4_slot] = 1.0;
        x[h1_slot] = 0.0;
        m_inf.cell_xa_mut(k).copy_from_slice(&x);
    }
    m_inf.v.fill(0.0);
    for k in (k_core + 1)..m_inf.nz {
        m_inf.v[k] = -2.5e8;
    }
    let ctx_inf = StepReportContext::compute(&m_inf);
    assert!(ctx_inf.cores.he_core_mass > 0.0);
    let (fe_inf, non_fe_inf, non_fe_reb) =
        eval_core_infall_and_rebound_cms(&m_inf, &ctx_inf.cores);
    assert_eq!(fe_inf, 0.0, "no Fe core -> fe_core_infall == 0");
    assert!(
        rel_err(non_fe_inf, 2.5e8) < 1.0e-13,
        "expected non_fe_core_infall = 2.5e8, got {non_fe_inf:.6e}"
    );
    assert_eq!(
        non_fe_reb,
        -f64::MAX,
        "absent Fe core yields empty slice maxval == -huge(1d0)"
    );

    // 9. Surface & average abundance limits (`xa_surface_*`, `xa_average_*`).
    let mut m_xa = base.clone();
    m_xa.controls.ts_stop.xa_surface_lower_limit_species =
        vec![String::from("h1")];
    m_xa.controls.ts_stop.xa_surface_lower_limit = vec![0.99];
    assert!(matches!(
        do_check_limits(&mut m_xa, None, false).expect("xa_surf_lo"),
        LimitCheckOutcome::Terminate { ref code, .. }
            if code == "xa_surface_lower_limit"
    ));

    let mut m_xa_avg = base.clone();
    m_xa_avg.controls.ts_stop.xa_average_upper_limit_species =
        vec![String::from("h1")];
    m_xa_avg.controls.ts_stop.xa_average_upper_limit = vec![0.01];
    assert!(matches!(
        do_check_limits(&mut m_xa_avg, None, false).expect("xa_avg_hi"),
        LimitCheckOutcome::Terminate { ref code, .. }
            if code == "xa_average_upper_limit"
    ));
}

#[test]
fn test_when_to_stop_rtol_atol_redo_convergence_and_max_age_graded_approach() {
    // 1. Target-hitting `when_to_stop_rtol`/`atol` triggers `Redo` inside
    //    `star_evolve_step` (halving `dt` without burning retries) and converges
    //    onto the target.
    let mut probe = zams_model(2.0);
    probe.dt = 1.0e5 * SECYER;
    probe.dt_next = probe.dt;
    let h1_init = flexmesa_star::center_avg_x_by_name(&probe, "h1");
    star_evolve_step(&mut probe, false, false, true).expect("probe step");
    let h1_after_full = flexmesa_star::center_avg_x_by_name(&probe, "h1");
    let drop = h1_init - h1_after_full;
    assert!(
        drop > 0.0,
        "expected central H1 drop over 1e5 yr on 2 Msun ZAMS, got {drop:.3e}"
    );

    // Place `xa_central_lower_limit` halfway through the single-step H1 drop
    // with tolerance smaller than `0.25 * drop` so the first full step overshoots
    // and triggers `Redo` -> `dt = 0.5 * dt`, landing within tolerance.
    let target_h1 = h1_init - 0.45 * drop;
    let mut m_redo = zams_model(2.0);
    m_redo.dt = 1.0e5 * SECYER;
    m_redo.dt_next = m_redo.dt;
    m_redo.controls.xa_central_lower_limit_species = vec![String::from("h1")];
    m_redo.controls.xa_central_lower_limit = vec![target_h1];
    m_redo.controls.ts_stop.when_to_stop_atol = 0.15 * drop;
    m_redo.controls.ts_stop.when_to_stop_rtol = 0.0;

    let rep = star_evolve_step(&mut m_redo, false, false, true)
        .expect("target-hitting step should succeed via redo");
    assert!(
        m_redo.num_redos >= 1,
        "expected at least 1 limit-hitting redo, got {}",
        m_redo.num_redos
    );
    assert_eq!(rep.retries, 0, "limit-hitting redo must not consume retries");
    assert_eq!(
        m_redo.aux.step_hist.timestep_hold, 0,
        "limit-hitting redo must not engage timestep_hold"
    );
    let h1_final = flexmesa_star::center_avg_x_by_name(&m_redo, "h1");
    assert!(
        (h1_final - target_h1).abs() <= 0.15 * drop + 1.0e-15 || h1_final > target_h1,
        "after redo, h1_final={h1_final:.12e} vs target={target_h1:.12e}"
    );

    // 2. `redo_limit = 1` terminates with `StarError::Terminated { reason: "redo_limit" }`
    //    when tolerance requires > 1 halving, and restores pre-attempt state.
    let mut m_rl = zams_model(2.0);
    m_rl.dt = 1.0e5 * SECYER;
    m_rl.dt_next = m_rl.dt;
    m_rl.controls.xa_central_lower_limit_species = vec![String::from("h1")];
    m_rl.controls.xa_central_lower_limit = vec![h1_init - 0.1 * drop];
    m_rl.controls.ts_stop.when_to_stop_atol = 1.0e-12;
    m_rl.controls.ts_stop.when_to_stop_rtol = 0.0;
    m_rl.controls.ts_stop.redo_limit = 1;
    let mod0 = m_rl.model_number;
    let err = star_evolve_step(&mut m_rl, false, false, true)
        .expect_err("must hit redo_limit");
    assert!(
        matches!(err, StarError::Terminated { ref reason, .. } if reason == "redo_limit"),
        "expected redo_limit termination, got {err:?}"
    );
    assert_eq!(m_rl.model_number, mod0, "redo_limit must restore model_number");

    // 3. Exact `max_age` landing (no `1e-12` premature stop) and graded approach
    //    (`num_adjusted_dt_steps_before_max_age`, `evolve.f90:2089-2142`).
    let mut m_age = zams_model(2.0);
    m_age.star_age = 0.0;
    m_age.aux.step_hist.time_sec = 0.0;
    m_age.aux.step_hist.time_old_sec = 0.0;
    m_age.dt = 2.0e4 * SECYER;
    m_age.dt_next = m_age.dt;
    m_age.controls.max_years_for_timestep = 4.0e4;
    m_age.controls.max_age = 1.0e5;
    m_age.controls.ts_stop.num_adjusted_dt_steps_before_max_age = 4;
    m_age.controls.ts_stop.dt_years_for_steps_before_max_age = 1.5e4;
    m_age.controls.ts_stop.reduction_factor_for_max_timestep = 0.5;

    let reps = star_evolve(&mut m_age, 25, false, false, true)
        .expect("graded max_age evolve");
    assert!(reps.len() >= 3, "expected graded multi-step approach");
    assert!(
        m_age.aux.step_hist.using_revised_max_yr_dt,
        "graded approach must set using_revised_max_yr_dt"
    );
    assert!(
        rel_err(m_age.star_age, 1.0e5) < 1.0e-14,
        "must land exactly at max_age=1e5 yr, got {:.16e}",
        m_age.star_age
    );
    assert!(
        rel_err(m_age.aux.step_hist.time_sec, 1.0e5 * SECYER) < 1.0e-14,
        "time_sec must match exact max_age * SECYER"
    );

    // 4. `max_age_in_days` still clips `dt` in `prepare_for_new_step` even when
    //    `max_age_in_seconds = Some(-1.0)` (`evolve.f90:1908-1913`).
    let mut m_days = zams_model(2.0);
    m_days.star_age = 0.0;
    m_days.aux.step_hist.time_sec = 0.0;
    m_days.aux.step_hist.time_old_sec = 0.0;
    m_days.dt = 100.0 * SECDAY;
    m_days.dt_next = m_days.dt;
    m_days.controls.max_age = -1.0;
    m_days.controls.max_age_in_seconds = Some(-1.0);
    m_days.controls.ts_stop.max_age_in_days = 10.0;
    let rep_days = star_evolve_step(&mut m_days, false, false, true)
        .expect("max_age_in_days clip");
    assert!(
        rel_err(rep_days.dt, 10.0 * SECDAY) < 1.0e-14,
        "expected dt clipped to 10 days ({:.6e} s), got {:.6e} s",
        10.0 * SECDAY,
        rep_days.dt
    );
}

#[test]
fn test_log_l_limits_use_surface_l_while_lnuc_div_l_uses_photosphere_l() {
    // Upstream `star/private/star_utils.f90:1110-1119` separates
    // `L_phot = photosphere_L` (used for `Lnuc_div_L`) from
    // `log_surface_luminosity = log10(max(1d-99, L(1)/Lsun))` (used for
    // `log_L_{lower,upper}_limit` and `abs_diff_lg_LH_lg_Ls_limit`).
    let mut m = zams_model(1.0);
    m.controls.ts_stop.when_to_stop_rtol = 1.0e99;
    m.controls.ts_stop.when_to_stop_atol = 1.0e99;

    // 1. Explicit `StepReportContext` where `photosphere_L != L(1)/Lsun`.
    let mut ctx = StepReportContext::compute(&m);
    m.lum[0] = 4.0 * L_SUN;
    ctx.phot.l_lsun = 1.0;

    // `log10(photosphere_L) = 0.0 < 0.3 < log10(L(1)/Lsun) = 0.60206`.
    m.controls.log_l_upper_limit = Some(0.3);
    let out_log_l =
        do_check_limits(&mut m, Some(&ctx), false).expect("log_L_upper_limit");
    assert!(
        matches!(out_log_l, LimitCheckOutcome::Terminate { ref code, .. } if code == "log_L_upper_limit"),
        "log_L_upper_limit must track L(1)/Lsun, not photosphere_L; got {out_log_l:?}"
    );
    m.controls.log_l_upper_limit = None;

    // `power_nuc / L(1) = power_nuc / 4 < mid < power_nuc / photosphere_L`.
    let power_nuc = ctx.powers.power_nuc_burn;
    assert!(power_nuc > 0.0, "ZAMS model must have positive nuclear power");
    let mid_ratio = 0.5 * (power_nuc / 4.0 + power_nuc / 1.0);
    m.controls.ts_stop.lnuc_div_l_upper_limit = mid_ratio;
    let out_lnuc = do_check_limits(&mut m, Some(&ctx), false)
        .expect("Lnuc_div_L_upper_limit");
    assert!(
        matches!(out_lnuc, LimitCheckOutcome::Terminate { ref code, .. } if code == "Lnuc_div_L_upper_limit"),
        "Lnuc_div_L_upper_limit must track photosphere_L, not L(1)/Lsun; got {out_lnuc:?}"
    );
    m.controls.ts_stop.lnuc_div_l_upper_limit = 1.0e99;

    // 2. Default `ctx = None` where `L(1) = 0` gives `log_surface_luminosity = -99`
    //    while `eval_photosphere_info` clamps `l_erg >= 1 erg/s` (`log10 ~ -33.58`).
    m.lum[0] = 0.0;
    m.controls.log_l_lower_limit = Some(-50.0);
    let out_zero_l =
        do_check_limits(&mut m, None, false).expect("log_L_lower_limit");
    assert!(
        matches!(out_zero_l, LimitCheckOutcome::Terminate { ref code, .. } if code == "log_L_lower_limit"),
        "zero surface luminosity must trigger log_L_lower_limit at -50; got {out_zero_l:?}"
    );
}

struct ClearLimitOnFinishExtras {
    check_calls: usize,
    finish_calls: usize,
}

impl StarExtras for ClearLimitOnFinishExtras {
    fn extras_check_model(&mut self, _model: &mut StarModel) -> ExtrasDirective {
        self.check_calls += 1;
        ExtrasDirective::KeepGoing
    }

    fn extras_finish_step(&mut self, model: &mut StarModel) -> ExtrasDirective {
        self.finish_calls += 1;
        // Clear the active limit before `inlist_runner` refreshes hydro/mixing
        // state, verifying that the in-step `do_check_limits` termination code
        // is propagated directly rather than dropped.
        model.controls.star_mass_max_limit = 0.0;
        ExtrasDirective::KeepGoing
    }
}

#[test]
fn test_in_step_termination_propagates_across_post_step_refresh() {
    // 1. Direct `star_evolve_step` records `termination_code` on acceptance
    //    and clears it at the start of any subsequent step.
    let mut m = zams_model(2.0);
    m.dt = 1.0e5 * SECYER;
    m.dt_next = m.dt;
    m.controls.star_mass_max_limit = 1.0;
    m.controls.ts_stop.when_to_stop_rtol = 1.0e99;
    m.controls.ts_stop.when_to_stop_atol = 1.0e99;

    star_evolve_step(&mut m, false, false, true).expect("terminating step");
    assert_eq!(
        m.aux.step_hist.termination_code.as_deref(),
        Some("star_mass_max_limit"),
        "star_evolve_step must record in-step termination_code"
    );

    m.controls.star_mass_max_limit = 0.0;
    star_evolve_step(&mut m, false, false, true).expect("normal follow-up step");
    assert_eq!(
        m.aux.step_hist.termination_code, None,
        "subsequent non-terminating step must clear termination_code"
    );

    // 2. `run_inlist_stage` honors in-step `termination_code` even when
    //    `extras_finish_step` / post-step refresh clears the limit, skips
    //    `extras_check_model` (`run_star_support.f90:211-212`), and still
    //    invokes `extras_finish_step` once (`run_star_support.f90:734-738`).
    let mut m_stage = zams_model(2.0);
    m_stage.dt = 1.0e5 * SECYER;
    m_stage.dt_next = m_stage.dt;
    m_stage.controls.star_mass_max_limit = 1.0;
    m_stage.controls.ts_stop.when_to_stop_rtol = 1.0e99;
    m_stage.controls.ts_stop.when_to_stop_atol = 1.0e99;
    let tmp = std::env::temp_dir().join("flexmesa_159_stage_term.inlist");
    std::fs::write(&tmp, "&star_job\n/\n&controls\n/\n").expect("write inlist");
    let bundle = InlistBundle::from_file(&tmp).expect("parse empty inlist");
    let _ = std::fs::remove_file(&tmp);
    let mut extras = ClearLimitOnFinishExtras {
        check_calls: 0,
        finish_calls: 0,
    };
    let rep = run_inlist_stage(&mut m_stage, &bundle, &mut extras, Some(3))
        .expect("run_inlist_stage");
    assert_eq!(
        rep.steps_taken, 1,
        "inlist_runner must stop on the step terminated by in-step do_check_limits"
    );
    assert_eq!(rep.termination_code, "star_mass_max_limit");
    assert_eq!(
        extras.check_calls, 0,
        "extras_check_model must be skipped once do_check_limits terminates"
    );
    assert_eq!(
        extras.finish_calls, 1,
        "extras_finish_step must still run once on the terminating step"
    );
}

#[test]
fn test_all_wind_schemes_match_independent_upstream_formulas() {
    // Independent numeric oracles evaluated directly in Python from
    // `star/private/winds.f90:323, 347-400, 479-632` using MESA `26.4.1`
    // `const_def.f90` constants (`Lsun=3.828e33`, `Msun=1.3271244e26/6.67430e-8`,
    // `Rsun=6.957e10`, `clight=2.99792458e10`, `standard_cgrav=6.67430e-8`):
    //   L = 1e5 Lsun, M = 20 Msun, R = 10 Rsun, Z_low = 0.005, Y_wr = 0.98.
    let l_cgs = 1.0e5 * L_SUN;
    let m_cgs = 20.0 * M_SUN;
    let r_cgs = 10.0 * R_SUN;

    // 1. Reimers (`winds.f90:323, 363-364`), eta = 0.5
    let w_reimers = eval_reimers_wind(l_cgs, m_cgs, r_cgs, 0.5);
    assert!(
        rel_err(w_reimers, 9.999_999_999_999_999e-9) < 1.0e-13,
        "Reimers mismatch: {w_reimers:.16e}"
    );
    assert_eq!(eval_reimers_wind(l_cgs, m_cgs, r_cgs, 0.0), 0.0);

    // 2. Blocker (`winds.f90:382-384, 556-561`), eta = 0.1
    let w_blocker = eval_blocker_wind(l_cgs, m_cgs, r_cgs, 0.1);
    assert!(
        rel_err(w_blocker, 5.659_970_081_494_931e-7) < 1.0e-13,
        "Blocker mismatch: {w_blocker:.16e}"
    );

    // 3. de Jager (`winds.f90:385-388, 596-606`), T = 4000 K, eta = 0.8
    let w_dj = eval_de_jager_wind(l_cgs, 4000.0, 0.8);
    assert!(
        rel_err(w_dj, 3.573_036_407_928_932e-6) < 1.0e-13,
        "de Jager mismatch: {w_dj:.16e}"
    );

    // 4. van Loon (`winds.f90:389-392, 609-616`), T = 3500 K, eta = 0.7
    let w_vl = eval_van_loon_wind(l_cgs, 3500.0, 0.7);
    assert!(
        rel_err(w_vl, 1.758_320_502_056_704e-5) < 1.0e-13,
        "van Loon mismatch: {w_vl:.16e}"
    );

    // 5. Nieuwenhuijzen (`winds.f90:393-396, 619-632`), eta = 0.6
    let w_nj = eval_nieuwenhuijzen_wind(l_cgs, m_cgs, r_cgs, 0.6);
    assert!(
        rel_err(w_nj, 9.469_260_183_906_717e-8) < 1.0e-13,
        "Nieuwenhuijzen mismatch: {w_nj:.16e}"
    );

    // 6. Vink (`winds.f90:377-381, 497-553`): hot side (T=35000 K, Z=0.005, eta=0.9),
    //    cool side (T=18000 K, Z=0.005, eta=0.9), and exact bi-stability jump
    //    midpoint (Z=0.019 -> Teff_jump=25882.76 K, alfa=0.5, eta=1.0).
    let w_vh = eval_vink_wind(l_cgs, m_cgs, 35_000.0, 0.005, 0.9);
    assert!(
        rel_err(w_vh, 7.203_017_482_239_952e-8) < 1.0e-13,
        "Vink hot mismatch: {w_vh:.16e}"
    );
    let w_vc = eval_vink_wind(l_cgs, m_cgs, 18_000.0, 0.005, 0.9);
    assert!(
        rel_err(w_vc, 2.401_309_888_448_109e-7) < 1.0e-13,
        "Vink cool mismatch: {w_vc:.16e}"
    );
    let w_vj = eval_vink_wind(l_cgs, m_cgs, 25_882.76, 0.019, 1.0);
    assert!(
        rel_err(w_vj, 4.971_923_061_901_877e-7) < 1.0e-13,
        "Vink jump midpoint mismatch: {w_vj:.16e}"
    );

    // 7. Bjorklund (`winds.f90:397-400, 479-495`): L=1e6 Lsun, M=45 Msun, T=45000 K, Z=0.014, eta=0.8
    let w_bj = eval_bjorklund_wind(
        1.0e6 * L_SUN,
        45.0 * M_SUN,
        45_000.0,
        0.014,
        STANDARD_CGRAV,
        0.8,
    );
    assert!(
        rel_err(w_bj, 8.677_105_553_773_57e-6) < 1.0e-13,
        "Bjorklund mismatch: {w_bj:.16e}"
    );

    // 8. Nugis & Lamers WR (`winds.f90:567-570`) and composite Dutch (`winds.f90:347-362, 564-593`):
    let w_nl = eval_nugis_lamers_wind(l_cgs, 0.98, 0.02, 0.8);
    assert!(
        rel_err(w_nl, 3.080_983_081_532_222e-6) < 1.0e-13,
        "Nugis-Lamers WR mismatch: {w_nl:.16e}"
    );
    // Dutch eta = 0 -> 0.0
    assert_eq!(
        eval_dutch_wind(
            l_cgs,
            m_cgs,
            r_cgs,
            35_000.0,
            0.70,
            0.28,
            0.02,
            DutchLowTScheme::DeJager,
            0.0
        ),
        0.0
    );
    // Dutch high-T with X < 0.4 -> Nugis-Lamers WR
    let w_dutch_wr = eval_dutch_wind(
        l_cgs,
        m_cgs,
        r_cgs,
        15_000.0,
        0.0,
        0.98,
        0.02,
        DutchLowTScheme::DeJager,
        0.8,
    );
    assert!(rel_err(w_dutch_wr, 3.080_983_081_532_222e-6) < 1.0e-13);
    // Dutch high-T with X >= 0.4 -> Vink
    let w_dutch_vink = eval_dutch_wind(
        l_cgs,
        m_cgs,
        r_cgs,
        35_000.0,
        0.70,
        0.295,
        0.005,
        DutchLowTScheme::DeJager,
        0.9,
    );
    assert!(rel_err(w_dutch_vink, 7.203_017_482_239_952e-8) < 1.0e-13);
    // Dutch low-T branches (`de Jager`, `van Loon`, `Nieuwenhuijzen`)
    let w_dutch_dj = eval_dutch_wind(
        l_cgs,
        m_cgs,
        r_cgs,
        4000.0,
        0.70,
        0.28,
        0.02,
        DutchLowTScheme::DeJager,
        0.8,
    );
    assert!(rel_err(w_dutch_dj, 3.573_036_407_928_932e-6) < 1.0e-13);
    let w_dutch_vl = eval_dutch_wind(
        l_cgs,
        m_cgs,
        r_cgs,
        3500.0,
        0.70,
        0.28,
        0.02,
        DutchLowTScheme::VanLoon,
        0.7,
    );
    assert!(rel_err(w_dutch_vl, 1.758_320_502_056_704e-5) < 1.0e-13);
    let w_dutch_nj = eval_dutch_wind(
        l_cgs,
        m_cgs,
        r_cgs,
        4000.0,
        0.70,
        0.28,
        0.02,
        DutchLowTScheme::Nieuwenhuijzen,
        0.6,
    );
    assert!(rel_err(w_dutch_nj, 9.469_260_183_906_717e-8) < 1.0e-13);
    // Dutch transition midpoint T = 10500 K (`alfa = 0.5`) between de Jager and Nugis-Lamers
    let w_dutch_mid = eval_dutch_wind(
        l_cgs,
        m_cgs,
        r_cgs,
        10_500.0,
        0.0,
        0.98,
        0.02,
        DutchLowTScheme::DeJager,
        0.8,
    );
    assert!(
        rel_err(w_dutch_mid, 1.894_934_803_826_664e-6) < 1.0e-13,
        "Dutch 10500 K midpoint mismatch: {w_dutch_mid:.16e}"
    );
}

#[test]
fn test_wind_driver_dispatch_gates_caps_and_rotation_limits() {
    let mut model = zams_model(5.0);

    // 1. `hot_wind_full_on_T < cool_wind_full_on_T` errors loudly (`winds.f90:144-148`).
    model.controls.hot_wind_full_on_t = 5_000.0;
    model.controls.cool_wind_full_on_t = 10_000.0;
    assert!(matches!(
        do_set_mdot(&mut model),
        Err(StarError::InvalidInput(_))
    ));

    // 2. Invalid `Dutch_wind_lowT_scheme` errors loudly (`winds.f90:588-592`).
    model.controls.hot_wind_full_on_t = 12_000.0;
    model.controls.cool_wind_full_on_t = 8_000.0;
    model.controls.mix_wind.dutch_wind_lowt_scheme = String::from("bogus");
    assert!(matches!(
        do_set_mdot(&mut model),
        Err(StarError::InvalidInput(_))
    ));
    model.controls.mix_wind.dutch_wind_lowt_scheme = String::from("de Jager");

    // 3. Hot/cool temperature dispatch, RGB vs AGB switch, and midpoint linear blend
    //    (`winds.f90:150-184`): set Teff = 10000 K (exact midpoint between 8000 K and 12000 K).
    model.teff = 10_000.0;
    model.controls.mix_wind.hot_wind_scheme = String::from("Reimers");
    model.controls.cool_wind_rgb_scheme = String::from("de Jager");
    model.controls.cool_wind_agb_scheme = String::from("van Loon");
    model.controls.reimers_scaling_factor = 1.0;
    model.controls.mix_wind.de_jager_scaling_factor = 1.0;
    model.controls.mix_wind.van_loon_scaling_factor = 1.0;
    let phot = flexmesa_star::eval_photosphere_info(&model);
    let l1 = phot.l_lsun * L_SUN;
    let m1 = model.mstar;
    let r1 = (l1
        / (flexmesa_const::PI
            * flexmesa_const::CRAD
            * CLIGHT
            * flexmesa_math::pow4(10_000.0)))
    .sqrt();
    let expected_hot = 4.0e-13 * (l1 * r1 / m1) / (L_SUN * R_SUN / M_SUN);
    let expected_cool_rgb = flexmesa_math::exp10(
        1.769 * flexmesa_math::log10(l1 / L_SUN)
            - 1.676 * flexmesa_math::log10(10_000.0)
            - 8.158,
    );
    let expected_blend = 0.5 * expected_hot + 0.5 * expected_cool_rgb;
    let mdot_blend = do_set_mdot(&mut model).expect("hot/cool blend");
    assert!(
        rel_err(mdot_blend, -expected_blend * M_SUN / SECYER) < 1.0e-12,
        "hot/cool midpoint blend mismatch: got {mdot_blend:.12e}, expected {:.12e}",
        -expected_blend * M_SUN / SECYER
    );

    // Pure cool RGB vs AGB switch (`center_h1 < 0.01 && center_he4 < rgb_to_agb_wind_switch`).
    model.teff = 4_000.0;
    let h1_idx = model.iso_names.iter().position(|n| n == "h1").expect("h1");
    let he4_idx = model.iso_names.iter().position(|n| n == "he4").expect("he4");
    let nz = model.nz;
    model.cell_xa_mut(nz - 1)[h1_idx] = 0.0;
    model.cell_xa_mut(nz - 1)[he4_idx] = 1.0e-5;
    model.controls.mix_wind.rgb_to_agb_wind_switch = 1.0e-4;
    let expected_agb = flexmesa_math::exp10(
        -5.65 + 1.05 * flexmesa_math::log10(l1 / (1.0e4 * L_SUN))
            - 6.3 * flexmesa_math::log10(4_000.0 / 3_500.0),
    );
    let mdot_agb = do_set_mdot(&mut model).expect("cool AGB switch");
    assert!(
        rel_err(mdot_agb, -expected_agb * M_SUN / SECYER) < 1.0e-12,
        "AGB switch mismatch: got {mdot_agb:.12e}, expected {:.12e}",
        -expected_agb * M_SUN / SECYER
    );

    // 4. `mass_change` positive accretion + `xfer_ratio` reduction, negative loss,
    //    `min_wind`, `max_wind`, `wind_H_He_envelope_limit`, `wind_He_layer_limit`
    //    (`winds.f90:133-141, 240-252, 271-285`).
    let mut mc = zams_model(2.0);
    mc.controls.mix_wind.hot_wind_scheme = String::new();
    mc.controls.cool_wind_rgb_scheme = String::new();
    mc.controls.cool_wind_agb_scheme = String::new();
    mc.controls.mass_change = 1.0e-6; // accretion -> positive mstar_dot
    let acc_full = do_set_mdot(&mut mc).expect("mass_change accretion");
    assert!(rel_err(acc_full, 1.0e-6 * M_SUN / SECYER) < 1.0e-13);

    // Reduce positive `mass_change` by `xfer_ratio = 0.5` from `eval_rlo_wind` (`ratio = 0.9`).
    let phot_mc = flexmesa_star::eval_photosphere_info(&mc);
    let r_phot_rsun = (phot_mc.l_lsun * L_SUN
        / (flexmesa_const::PI
            * flexmesa_const::CRAD
            * CLIGHT
            * flexmesa_math::pow4(mc.teff)))
    .sqrt()
        / R_SUN;
    mc.controls.mix_wind.rlo_scaling_factor = 1.0;
    mc.controls.mix_wind.rlo_wind_min_l = 0.0;
    mc.controls.mix_wind.rlo_wind_max_teff = 1.0e6;
    mc.controls.mix_wind.rlo_wind_roche_lobe_radius = r_phot_rsun / 0.9;
    mc.controls.mix_wind.roche_lobe_xfer_full_on = 0.8;
    mc.controls.mix_wind.roche_lobe_xfer_full_off = 1.0;
    let acc_red = do_set_mdot(&mut mc).expect("accretion xfer_ratio");
    assert!(
        rel_err(acc_red, 0.5 * 1.0e-6 * M_SUN / SECYER) < 1.0e-12,
        "xfer_ratio=0.5 reduction mismatch: got {acc_red:.12e}"
    );
    mc.controls.mix_wind.rlo_scaling_factor = 0.0;

    // Negative `mass_change`, `min_wind`, `max_wind`.
    mc.controls.mass_change = -2.0e-7;
    assert!(
        rel_err(
            do_set_mdot(&mut mc).expect("neg mass_change"),
            -2.0e-7 * M_SUN / SECYER
        ) < 1.0e-13
    );
    mc.controls.mass_change = 0.0;
    mc.controls.mix_wind.min_wind = 3.0e-7;
    assert!(
        rel_err(
            do_set_mdot(&mut mc).expect("min_wind"),
            -3.0e-7 * M_SUN / SECYER
        ) < 1.0e-13
    );
    mc.controls.mix_wind.max_wind = 1.0e-7;
    assert!(
        rel_err(
            do_set_mdot(&mut mc).expect("max_wind"),
            -1.0e-7 * M_SUN / SECYER
        ) < 1.0e-13
    );
    mc.controls.mix_wind.max_wind = 0.0;

    // `wind_H_He_envelope_limit` and `wind_He_layer_limit` shut off wind.
    mc.controls.mix_wind.wind_h_he_envelope_limit = mc.mstar / M_SUN + 1.0;
    assert_eq!(do_set_mdot(&mut mc).expect("H_He env lim"), 0.0);
    mc.controls.mix_wind.wind_h_he_envelope_limit = 0.0;
    mc.controls.mix_wind.wind_he_layer_limit = 0.1; // ZAMS has he_core_mass == co_core_mass == 0
    assert_eq!(do_set_mdot(&mut mc).expect("He layer lim"), 0.0);
    mc.controls.mix_wind.wind_he_layer_limit = 0.0;

    // 5. `rotation_enhancement` caps (`winds.f90:412-476`):
    //    `rotational_mdot_boost_fac`, `rotational_mdot_kh_fac`,
    //    `max_mdot_jump_for_rotation`, and post-rotation `max_wind`.
    let mut rot = zams_model(10.0);
    rot.controls.rotation_flag = true;
    rot.controls.mix_wind.hot_wind_scheme = String::from("other");
    rot.controls.cool_wind_rgb_scheme = String::from("other");
    rot.controls.cool_wind_agb_scheme = String::from("other");
    rot.controls.use_other_wind = true;
    let base_msun_yr = 1.0e-6;
    let base_cgs = base_msun_yr * M_SUN / SECYER;
    let extras = FixedOtherWindExtras {
        wind_msun_yr: base_msun_yr,
    };
    for k in 0..rot.nz {
        let w_c = flexmesa_star::eval_upstream_omega_crit(&rot, k);
        rot.omega[k] = 0.9 * w_c;
    }
    rot.controls.mix_wind.max_rotational_mdot_boost = 1.0e6;
    rot.controls.mix_wind.rotational_mdot_kh_fac = 1.0e10;
    rot.controls.mix_wind.rotational_mdot_boost_fac = 2.0;
    rot.controls.mix_wind.max_mdot_jump_for_rotation = 1.0e10;
    rot.aux.step_hist.mstar_dot_old = -base_cgs;
    let mdot_bfac =
        do_set_mdot_with_extras(&mut rot, &extras).expect("boost_fac cap");
    assert!(
        rel_err(mdot_bfac, -2.0 * base_cgs) < 1.0e-12,
        "rotational_mdot_boost_fac=2 cap mismatch: {mdot_bfac:.12e}"
    );

    // `rotational_mdot_kh_fac`: independent `kh_timescale = 0.75 * G * M^2 / (R * L)`.
    let phot_r = flexmesa_star::eval_photosphere_info(&rot);
    let l1_r = phot_r.l_lsun * L_SUN;
    let r1_r = (l1_r
        / (flexmesa_const::PI
            * flexmesa_const::CRAD
            * CLIGHT
            * flexmesa_math::pow4(rot.teff)))
    .sqrt();
    let kh_s = 0.75 * rot.cgrav[0] * rot.mstar * rot.mstar / (r1_r * l1_r);
    let kh_fac = 1.0e-4;
    let expected_kh_cap = kh_fac * rot.mstar / kh_s;
    rot.controls.mix_wind.rotational_mdot_boost_fac = 1.0e10;
    rot.controls.mix_wind.rotational_mdot_kh_fac = kh_fac;
    let mdot_kh = do_set_mdot_with_extras(&mut rot, &extras).expect("kh cap");
    assert!(
        rel_err(mdot_kh, -expected_kh_cap) < 1.0e-12,
        "rotational_mdot_kh_fac cap mismatch: got {mdot_kh:.12e}, expected {:.12e}",
        -expected_kh_cap
    );

    // `max_mdot_jump_for_rotation`: caps jump relative to `-mstar_dot_old`.
    rot.controls.mix_wind.rotational_mdot_kh_fac = 1.0e10;
    rot.controls.mix_wind.max_mdot_jump_for_rotation = 1.5;
    rot.aux.step_hist.mstar_dot_old = -base_cgs;
    let mdot_jump =
        do_set_mdot_with_extras(&mut rot, &extras).expect("jump cap");
    assert!(
        rel_err(mdot_jump, -1.5 * base_cgs) < 1.0e-12,
        "max_mdot_jump_for_rotation cap mismatch: {mdot_jump:.12e}"
    );

    // Post-rotation `max_wind` recheck (`winds.f90:467-471`).
    rot.controls.mix_wind.max_mdot_jump_for_rotation = 1.0e10;
    rot.controls.mix_wind.max_wind = 1.25 * base_msun_yr;
    let mdot_mw =
        do_set_mdot_with_extras(&mut rot, &extras).expect("post-rot max_wind");
    assert!(
        rel_err(mdot_mw, -1.25 * base_cgs) < 1.0e-12,
        "post-rotation max_wind cap mismatch: {mdot_mw:.12e}"
    );
}

#[test]
fn test_mass_stopping_limits_check_termination_outcomes_and_set1_irot() {
    let base = zams_model(5.0);

    // 1. Mass stopping limits (`star/private/do_one_utils.f90:877-935`):
    //    `star_mass_min_limit`, `star_mass_max_limit`,
    //    `star_species_mass_{min,max}_limit`, `xmstar_{min,max}_limit`,
    //    `envelope_mass_limit`, `envelope_fraction_left_limit`,
    //    `he_core_mass_limit`, `co_core_mass_limit`.
    let mut m = base.clone();
    m.controls.ts_stop.when_to_stop_atol = 1.0e99;
    m.controls.ts_stop.when_to_stop_rtol = 1.0e99;

    m.controls.star_mass_min_limit = 6.0;
    assert!(matches!(
        do_check_limits(&mut m, None, false).expect("star_mass_min"),
        LimitCheckOutcome::Terminate { ref code, .. } if code == "star_mass_min_limit"
    ));
    m.controls.star_mass_min_limit = 0.0;

    m.controls.star_mass_max_limit = 4.0;
    assert!(matches!(
        do_check_limits(&mut m, None, false).expect("star_mass_max"),
        LimitCheckOutcome::Terminate { ref code, .. } if code == "star_mass_max_limit"
    ));
    m.controls.star_mass_max_limit = 0.0;

    m.controls.star_species_mass_min_limit_iso = String::from("h1");
    m.controls.star_species_mass_min_limit = 4.9;
    assert!(matches!(
        do_check_limits(&mut m, None, false).expect("species_min"),
        LimitCheckOutcome::Terminate { ref code, .. } if code == "star_species_mass_min_limit"
    ));
    m.controls.star_species_mass_min_limit = -1.0;

    m.controls.star_species_mass_max_limit_iso = String::from("h1");
    m.controls.star_species_mass_max_limit = 0.1;
    assert!(matches!(
        do_check_limits(&mut m, None, false).expect("species_max"),
        LimitCheckOutcome::Terminate { ref code, .. } if code == "star_species_mass_max_limit"
    ));
    m.controls.star_species_mass_max_limit = 1.0e99;

    m.controls.ts_stop.xmstar_min_limit = m.xmstar + 1.0;
    assert!(matches!(
        do_check_limits(&mut m, None, false).expect("xmstar_min"),
        LimitCheckOutcome::Terminate { ref code, .. } if code == "xmstar_min_limit"
    ));
    m.controls.ts_stop.xmstar_min_limit = -1.0;

    m.controls.ts_stop.xmstar_max_limit = m.xmstar - 1.0;
    assert!(matches!(
        do_check_limits(&mut m, None, false).expect("xmstar_max"),
        LimitCheckOutcome::Terminate { ref code, .. } if code == "xmstar_max_limit"
    ));
    m.controls.ts_stop.xmstar_max_limit = -1.0;

    m.controls.envelope_mass_limit = 6.0;
    assert!(matches!(
        do_check_limits(&mut m, None, false).expect("envelope_mass"),
        LimitCheckOutcome::Terminate { ref code, .. } if code == "envelope_mass_limit"
    ));
    m.controls.envelope_mass_limit = -1.0;

    m.controls.ts_stop.envelope_fraction_left_limit = 1.5;
    assert!(matches!(
        do_check_limits(&mut m, None, false).expect("envelope_frac"),
        LimitCheckOutcome::Terminate { ref code, .. } if code == "envelope_fraction_left_limit"
    ));
    m.controls.ts_stop.envelope_fraction_left_limit = -1.0;

    // Construct an inner He + CO core to exercise `he_core_mass_limit` and `co_core_mass_limit`.
    let h1_idx = m.iso_names.iter().position(|n| n == "h1").expect("h1");
    let he4_idx = m.iso_names.iter().position(|n| n == "he4").expect("he4");
    let c12_idx = m.iso_names.iter().position(|n| n == "c12").expect("c12");
    let k_he = m.nz / 2;
    let k_co = 3 * m.nz / 4;
    for k in k_he..m.nz {
        let mut x = vec![0.0; m.species];
        if k >= k_co {
            x[c12_idx] = 1.0;
        } else {
            x[he4_idx] = 1.0;
        }
        x[h1_idx] = 0.0;
        m.cell_xa_mut(k).copy_from_slice(&x);
    }
    m.controls.he_core_mass_limit = Some(0.1);
    assert!(matches!(
        do_check_limits(&mut m, None, false).expect("he_core_mass"),
        LimitCheckOutcome::Terminate { ref code, .. } if code == "he_core_mass_limit"
    ));
    m.controls.he_core_mass_limit = None;

    m.controls.co_core_mass_limit = Some(0.05);
    assert!(matches!(
        do_check_limits(&mut m, None, false).expect("co_core_mass"),
        LimitCheckOutcome::Terminate { ref code, .. } if code == "co_core_mass_limit"
    ));
    m.controls.co_core_mass_limit = None;

    // 2. `check_termination_limits_with_history` preserves `KeepGoing`, `Redo`,
    //    `Retry`, `Terminate`, and `Err` verbatim (`termination.rs`, `#159`).
    let mut chk = base.clone();
    assert_eq!(
        check_termination_limits_with_history(&chk, None).expect("keep_going"),
        LimitCheckOutcome::KeepGoing
    );
    chk.dt = 1.0e4;
    chk.star_age = 100.0;
    chk.controls.max_age = 50.0;
    chk.controls.ts_stop.when_to_stop_atol = 1.0;
    chk.controls.ts_stop.when_to_stop_rtol = 0.0;
    assert!(matches!(
        check_termination_limits_with_history(&chk, None).expect("redo"),
        LimitCheckOutcome::Redo { dt_new, .. } if (dt_new - 5.0e3).abs() < 1.0e-12
    ));
    chk.controls.ts_stop.when_to_stop_atol = 100.0;
    assert!(matches!(
        check_termination_limits_with_history(&chk, None).expect("terminate"),
        LimitCheckOutcome::Terminate { ref code, .. } if code == "max_age"
    ));
    chk.controls.max_age = -1.0;
    chk.controls.v_flag = true;
    chk.controls.ts_stop.retry_for_v_above_clight = true;
    chk.v[0] = 1.1 * CLIGHT;
    assert!(matches!(
        check_termination_limits_with_history(&chk, None).expect("retry"),
        LimitCheckOutcome::Retry { .. }
    ));
    chk.v[0] = 0.0;
    chk.controls.v_flag = false;
    let k_pk = chk.nz - 2;
    chk.eps_nuc.fill(0.0);
    chk.eps_nuc[k_pk] = 1.0e10;
    chk.conv_vel[k_pk] = 0.8 * chk.csound[k_pk];
    chk.controls.ts_stop.peak_burn_vconv_div_cs_limit = 1.0;
    assert!(matches!(
        check_termination_limits_with_history(&chk, None),
        Err(StarError::InvalidInput(_))
    ));

    // 3. `set1_irot` inside `do_adjust_mass_step` reproduces upstream
    //    `adjust_mass.f90:1232-1257` + `hydro_rotation.f90:55-226`:
    //    (a) at zero rotation (`omega = 0`), `w_roche = 0` so `i_rot = (2/3)*r^2`;
    //    (b) at rapid rotation (`omega = 2e-4 rad/s`), oblate distortion raises
    //        `i_rot` strictly above `(2/3)*r^2` following Paxton et al. (2019).
    let mut m_rot0 = zams_model(2.0);
    m_rot0.controls.rotation_flag = true;
    m_rot0.controls.mix_wind.use_accreted_material_j = false;
    m_rot0.omega.fill(0.0);
    m_rot0.j_rot.fill(0.0);
    // `do_adjust_mass_core` reconstructs `ln_r` from `(lnd, dm)` before calling
    // `set_omega_adjust_mass` / `set1_irot` (`adjust_mass.rs:402, 535`).
    m_rot0.reconstruct_ln_r_from_lnd_and_dm();
    m_rot0.save_old_values();
    let r0_before_zero = flexmesa_math::exp(m_rot0.ln_r[0]);
    m_rot0.mstar_dot = 5.0e-4 * M_SUN / SECYER;
    do_adjust_mass_step(&mut m_rot0, 0.5 * SECYER).expect("adjust mass zero omega");
    let spherical_0 = (2.0 / 3.0) * (r0_before_zero * r0_before_zero);
    assert!(
        rel_err(m_rot0.aux.i_rot[0], spherical_0) < 1.0e-13,
        "set1_irot zero-omega expected spherical (2/3)r^2={spherical_0:.12e}, got {:.12e}",
        m_rot0.aux.i_rot[0]
    );

    let mut m_rot = zams_model(2.0);
    m_rot.controls.rotation_flag = true;
    m_rot.controls.mix_wind.use_accreted_material_j = false;
    m_rot.omega.fill(2.0e-4);
    flexmesa_star::update_rotation_distortion_factors(&mut m_rot);
    m_rot.save_old_values();
    m_rot.mstar_dot = 5.0e-4 * M_SUN / SECYER;
    do_adjust_mass_step(&mut m_rot, 0.5 * SECYER).expect("adjust mass set1_irot");
    let r00 = flexmesa_math::exp(m_rot.ln_r[0]);
    assert!(
        m_rot.aux.i_rot[0] > (2.0 / 3.0) * r00 * r00,
        "rapidly rotating set1_irot must exceed spherical (2/3)r^2"
    );
}
