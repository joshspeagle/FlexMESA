// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration tests for Phase 5b (`#113`): adaptive remeshing (`do_adjust_mesh`),
//! Björklund/Dutch/rotational winds (`do_set_mdot`), 3-region mass adjustment
//! and `eps_mdot` (`do_adjust_mass_step`), 16-set overshooting + Péclet damping,
//! `do_conv_premix` + predictive mixing + extra heat + `star_evolve_step` energy
//! conservation, and `timestep_controller` retry / `star_evolve` termination.

use flexmesa_const::{
    MixingType, CLIGHT, FOUR_PI, L_SUN, M_SUN, SECYER, STANDARD_CGRAV,
};
use flexmesa_star::{
    add_overshooting, apply_predictive_mixing_from_controls,
    apply_step_pre_physics, check_termination_limits, compute_core_masses,
    do_adjust_mass_step, do_adjust_mesh, do_conv_premix, do_element_diffusion,
    do_kap, do_set_mdot, eval_bjorklund_wind, eval_de_jager_wind,
    eval_dutch_wind, eval_nieuwenhuijzen_wind, eval_nugis_lamers_wind,
    eval_omega_crit, eval_van_loon_wind, eval_vink_wind, find_mass_flux,
    load_zams_model, locate_convection_boundaries, set_basic_vars,
    set_hydro_vars, star_evolve, star_evolve_step, timestep_controller,
    DiffusionScheme, DutchLowTScheme, EnergyEqnOption, MeshFunctionConfig,
    StarControls, StarError, StarJobControls, StarModel, TimestepAction,
};

fn rel_err(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1.0e-30)
}

fn species_total_masses(model: &StarModel) -> Vec<f64> {
    let nc = model.species;
    let mut totals = vec![0.0_f64; nc];
    for k in 0..model.nz {
        for j in 0..nc {
            totals[j] += model.dm[k] * model.xa[k * nc + j];
        }
    }
    totals
}

#[test]
fn test_do_adjust_mesh_splits_and_merges_conserving_mass_and_isotopes() {
    let mut model = load_zams_model(
        StarControls {
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 1 Msun ZAMS");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");

    let cfg_default = MeshFunctionConfig::from_controls(&model.controls);
    assert!(
        rel_err(cfg_default.max_dq, model.controls.mix_wind.max_dq) < 1.0e-15
    );
    assert!(
        rel_err(
            cfg_default.p_function_weight,
            model.controls.mix_wind.p_function_weight
        ) < 1.0e-15
    );
    assert!(
        rel_err(
            cfg_default.t_function1_weight,
            model.controls.mix_wind.t_function1_weight
        ) < 1.0e-15
    );

    let nz_init = model.nz;
    let m_species_0 = species_total_masses(&model);

    let mut cfg_fine = cfg_default;
    cfg_fine.p_function_weight = 600.0;
    cfg_fine.t_function1_weight = 1200.0;
    cfg_fine.max_dq = 0.0008;
    // Note: in MESA `26.4.1` (`star/private/mesh_plan.f90:891`), `max_center_cell_dq`
    // terminates interior zoning once `1 - xq_new < max_dq_cntr`, so keeping the
    // upstream default `max_center_cell_dq = 1e-7` resolves the full interior.
    let (splits, _) =
        do_adjust_mesh(&mut model, &cfg_fine).expect("fine remesh");
    assert!(splits > 0, "expected splits on finer mesh weights");
    assert!(model.nz > nz_init, "nz should grow after split remesh");

    let dq_sum: f64 = model.dq.iter().sum();
    assert!((dq_sum - 1.0).abs() < 1.0e-13);
    for k in 0..model.nz {
        let x_sum: f64 = model.cell_xa(k).iter().sum();
        assert!((x_sum - 1.0).abs() < 1.0e-12);
        if k + 1 < model.nz {
            assert!(model.ln_r[k] > model.ln_r[k + 1]);
        }
    }
    let m_species_split = species_total_masses(&model);
    for (a, b) in m_species_split.iter().zip(&m_species_0) {
        if *b > 1.0e-12 * M_SUN {
            assert!(rel_err(*a, *b) < 1.0e-12);
        }
    }

    let nz_split = model.nz;
    let mut cfg_coarse = cfg_default;
    cfg_coarse.p_function_weight = 1.0;
    cfg_coarse.t_function1_weight = 1.0;
    cfg_coarse.grad_function_weight = 0.0;
    cfg_coarse.xa_function_weight = 0.0;
    cfg_coarse.max_dq = 0.25;
    cfg_coarse.max_surface_cell_dq = 1.0e-2;
    cfg_coarse.max_center_cell_dq = 1.0e-2;
    let (_, merges) =
        do_adjust_mesh(&mut model, &cfg_coarse).expect("coarse remesh");
    assert!(merges > 0, "expected merges on coarser mesh weights");
    assert!(model.nz < nz_split, "nz should shrink after merge remesh");

    let dq_sum_merge: f64 = model.dq.iter().sum();
    assert!((dq_sum_merge - 1.0).abs() < 1.0e-13);
    let m_species_merge = species_total_masses(&model);
    for (a, b) in m_species_merge.iter().zip(&m_species_0) {
        if *b > 1.0e-12 * M_SUN {
            assert!(rel_err(*a, *b) < 1.0e-12);
        }
    }
}

#[test]
fn test_do_set_mdot_bjorklund_dutch_and_rotational_boost_match_analytic_laws() {
    let l_surf = 1.0e6 * L_SUN;
    let m_surf = 45.0 * M_SUN;
    let t_surf = 4.5e4;
    let z_surf = 0.014;
    let gamma_e = 0.34 * l_surf / (FOUR_PI * CLIGHT * STANDARD_CGRAV * m_surf);
    let expected_bjork =
        flexmesa_math::exp10(-5.52) * flexmesa_math::pow(1.0 - gamma_e, -1.48);
    let got_bjork = eval_bjorklund_wind(
        l_surf,
        m_surf,
        t_surf,
        z_surf,
        STANDARD_CGRAV,
        1.0,
    );
    assert!(rel_err(got_bjork, expected_bjork) < 1.0e-12);

    let r_surf = 100.0 * flexmesa_const::R_SUN;
    let dutch = |t, x, y, sch| {
        eval_dutch_wind(l_surf, m_surf, r_surf, t, x, y, 0.02, sch, 1.0)
    };
    for (scheme, low_ref) in [
        (
            DutchLowTScheme::DeJager,
            eval_de_jager_wind(l_surf, 4000.0, 1.0),
        ),
        (
            DutchLowTScheme::VanLoon,
            eval_van_loon_wind(l_surf, 4000.0, 1.0),
        ),
        (
            DutchLowTScheme::Nieuwenhuijzen,
            eval_nieuwenhuijzen_wind(l_surf, m_surf, r_surf, 1.0),
        ),
    ] {
        assert!(rel_err(dutch(4000.0, 0.70, 0.28, scheme), low_ref) < 1.0e-13);
    }

    let got_nl = dutch(25_000.0, 0.20, 0.78, DutchLowTScheme::DeJager);
    let ref_nl = eval_nugis_lamers_wind(l_surf, 0.78, 0.02, 1.0);
    assert!(rel_err(got_nl, ref_nl) < 1.0e-13);

    let got_vink = dutch(25_000.0, 0.70, 0.28, DutchLowTScheme::DeJager);
    let ref_vink = eval_vink_wind(l_surf, m_surf, 25_000.0, 0.02, 1.0);
    assert!(rel_err(got_vink, ref_vink) < 1.0e-13);

    let got_blend = dutch(10_500.0, 0.70, 0.28, DutchLowTScheme::VanLoon);
    let w_low_mid = eval_van_loon_wind(l_surf, 10_500.0, 1.0);
    let w_hi_mid = eval_vink_wind(l_surf, m_surf, 10_500.0, 0.02, 1.0);
    assert!(rel_err(got_blend, f64::midpoint(w_low_mid, w_hi_mid)) < 1.0e-13);

    let mut ctrl = StarControls {
        initial_mass: 4.0,
        hot_wind_full_on_t: 1.0e4,
        cool_wind_full_on_t: 0.5e4,
        zbase: 0.02,
        ..Default::default()
    };
    ctrl.mix_wind.hot_wind_scheme = String::from("Bjorklund");
    ctrl.mix_wind.bjorklund_scaling_factor = 1.0;
    let mut model = load_zams_model(ctrl, StarJobControls::default())
        .expect("load 4 Msun ZAMS");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");

    // `min_star_mass_for_loss` is enforced in `compute_delta_m` / `do_adjust_mass_step`
    // (`star/private/adjust_mass.f90:49-54`), not inside `do_set_mdot`.
    let mdot_no_rot = do_set_mdot(&mut model).expect("non-rot mdot");
    assert!(mdot_no_rot < 0.0 && mdot_no_rot.is_finite());

    let mut m_gated = model.clone();
    m_gated.controls.mix_wind.min_star_mass_for_loss = 5.0;
    let mdot_before_gate = do_set_mdot(&mut m_gated).expect("mdot before gate");
    assert!(mdot_before_gate < 0.0);
    do_adjust_mass_step(&mut m_gated, 1.0e3 * SECYER)
        .expect("gated adjust_mass");
    assert_eq!(m_gated.mstar_dot, 0.0);
    assert_eq!(m_gated.mstar, model.mstar);

    let omega_crit = eval_omega_crit(
        model.cgrav[0],
        model.m_grav[0],
        model.r[0],
        model.lum[0],
        model.opacity[0],
        0.999_999,
    );
    model.controls.rotation_flag = true;
    model.controls.mix_wind.mdot_omega_power = 0.43;
    model.omega.fill(0.5 * omega_crit);
    let surf_rot =
        flexmesa_star::report_utils::eval_surf_avg_rotation_info(&model);
    let expected_boost =
        flexmesa_math::pow((1.0 - surf_rot.v_div_v_crit).max(1.0e-22), -0.43);
    let mdot_rot = do_set_mdot(&mut model).expect("rotational mdot");
    assert!(mdot_rot < mdot_no_rot);
    assert!(rel_err(mdot_rot / mdot_no_rot, expected_boost) < 1.0e-12);
    assert!(
        rel_err(
            model.aux.step_hist.rotational_mdot_boost_old,
            expected_boost
        ) < 1.0e-12
    );
}

#[test]
fn test_do_adjust_mass_step_and_eps_mdot_conserve_accreted_species_and_angular_momentum(
) {
    let mut ctrl = StarControls {
        mass_change: 1.0e-6,
        accrete_same_as_surface: false,
        accrete_given_mass_fractions: false,
        accretion_h1: 0.40,
        accretion_he4: 0.58,
        rotation_flag: true,
        zbase: 0.02,
        ..Default::default()
    };
    ctrl.mix_wind.accretion_zfracs = 3; // GS98
    ctrl.mix_wind.use_accreted_material_j = true;
    ctrl.mix_wind.accreted_material_j = 1.0e17;
    ctrl.mix_wind.eps_mdot_factor = 1.0;

    let mut model = load_zams_model(ctrl, StarJobControls::default())
        .expect("load 1 Msun ZAMS");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model.omega.fill(1.0e-5);
    for k in 0..model.nz {
        let i_k = ((2.0 / 3.0) * model.r[k] * model.r[k]).max(1.0e-99);
        model.aux.i_rot[k] = i_k;
        model.j_rot[k] = model.omega[k] * i_k;
    }

    let mstar_before = model.mstar;
    let x_surf_before = model.cell_xa(0)[0];
    let dm_before = model.dm.clone();
    let m_h1_before = species_total_masses(&model)[0];
    let j_tot_before: f64 = model
        .dm_bar
        .iter()
        .zip(&model.j_rot)
        .map(|(dm_b, j)| dm_b * j)
        .sum();

    let mdot = do_set_mdot(&mut model).expect("set accretion mdot");
    assert!(mdot > 0.0);
    let dt = 1.0e3 * SECYER;
    do_adjust_mass_step(&mut model, dt).expect("do_adjust_mass_step");

    let dm_total = mdot * dt;
    assert!(rel_err(model.mstar - mstar_before, dm_total) < 1.0e-12);
    assert!(
        model.cell_xa(0)[0] < x_surf_before,
        "surface H1 should decrease toward accreted 0.40"
    );
    let m_h1_after = species_total_masses(&model)[0];
    let j_tot_after: f64 = model
        .dm_bar
        .iter()
        .zip(&model.j_rot)
        .map(|(dm_b, j)| dm_b * j)
        .sum();
    assert!(rel_err(m_h1_after - m_h1_before, 0.40 * dm_total) < 1.0e-11);
    assert!(rel_err(j_tot_after - j_tot_before, 1.0e17 * dm_total) < 1.0e-11);

    let d_dm: Vec<f64> = model
        .dm
        .iter()
        .zip(&dm_before)
        .map(|(a, b)| a - b)
        .collect();
    let flux = find_mass_flux(&d_dm);
    assert!(rel_err(flux[0], dm_total) < 1.0e-11);
    assert_eq!(flux[model.nz], 0.0);

    assert!(model.eps_mdot.iter().all(|v| v.is_finite()));
    assert!(model.eps_mdot.iter().any(|&v| v.abs() > 0.0));
    assert!(model.j_rot.iter().all(|&j| j.is_finite() && j >= 0.0));
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_overshoot_16_param_sets_and_brunt_b_cutoff_follow_mesa_rules() {
    let ctrl = StarControls {
        initial_mass: 4.0,
        zbase: 0.02,
        ..Default::default()
    };
    let mut base = load_zams_model(ctrl, StarJobControls::default())
        .expect("load 4 Msun ZAMS");
    set_hydro_vars(&mut base, false, false).expect("set_hydro_vars");
    let n_bdy = locate_convection_boundaries(&mut base)
        .expect("locate_convection_boundaries");
    assert!(n_bdy > 0, "4 Msun ZAMS must have convective core");

    let mut m_exp = base.clone();
    {
        let s0 = &mut m_exp.controls.mix_wind.overshoot_sets[0];
        s0.scheme = String::from("exponential");
        s0.zone_type = String::from("burn_H");
        s0.zone_loc = String::from("core");
        s0.bdy_loc = String::from("top");
        s0.f = 0.02;
        s0.f0 = 0.005;
        s0.delta0 = 1.0;

        let s1 = &mut m_exp.controls.mix_wind.overshoot_sets[1];
        s1.scheme = String::from("step");
        s1.zone_type = String::from("any");
        s1.zone_loc = String::from("core");
        s1.bdy_loc = String::from("top");
        s1.f = 0.05;
        s1.f0 = 0.005;
        s1.delta0 = 1.0;
    }
    let mut m_short_circuit = m_exp.clone();
    m_short_circuit.controls.mix_wind.overshoot_sets[0].scheme =
        String::from("none");

    let mut m_bmax = m_exp.clone();
    m_bmax.controls.mix_wind.overshoot_brunt_b_max = 1.0e-6;
    m_bmax.aux.mixing.unsmoothed_brunt_b.fill(1.0);

    add_overshooting(&mut m_exp).expect("exp overshoot");
    add_overshooting(&mut m_short_circuit).expect("short-circuit overshoot");
    add_overshooting(&mut m_bmax).expect("bmax overshoot");

    let n_ov_exp = m_exp
        .mixing_type
        .iter()
        .filter(|&&mt| mt == MixingType::OvershootMixing)
        .count();
    let n_ov_short = m_short_circuit
        .mixing_type
        .iter()
        .filter(|&&mt| mt == MixingType::OvershootMixing)
        .count();
    let n_ov_bmax = m_bmax
        .mixing_type
        .iter()
        .filter(|&&mt| mt == MixingType::OvershootMixing)
        .count();

    assert!(n_ov_exp > 0, "expected OvershootMixing zones from set 0");
    let hp_cb =
        flexmesa_star::eval_conv_bdy_hp(&base, 0).expect("eval_conv_bdy_hp");
    let (_k_ob, r_ob, d_ob, _vc_ob) =
        flexmesa_star::eval_over_bdy_params(&base, 0, 0.005)
            .expect("eval_over_bdy_params");
    let ov_face_idx = m_exp
        .mixing_type
        .iter()
        .position(|&mt| mt == MixingType::OvershootMixing)
        .expect("first overshoot face");
    let dr_ov = m_exp.r[ov_face_idx] - r_ob;
    let expected_d_ov =
        d_ob * flexmesa_math::exp(-2.0 * dr_ov / (0.02 * hp_cb));
    assert!(rel_err(m_exp.d_mix[ov_face_idx], expected_d_ov) < 1.0e-11);
    assert_eq!(n_ov_short, 0, "'none' in set 0 must short-circuit set 1");
    assert_eq!(
        n_ov_bmax, 0,
        "unsmoothed_brunt_B > overshoot_brunt_B_max must stop overshooting"
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_conv_premix_predictive_mixing_and_extra_heat_in_evolve_step() {
    let ctrl = StarControls {
        initial_mass: 4.0,
        do_conv_premix: true,
        do_premix_heating: true,
        zbase: 0.02,
        ..Default::default()
    };
    let mut m_pm = load_zams_model(ctrl, StarJobControls::default())
        .expect("load 4 Msun ZAMS");
    set_hydro_vars(&mut m_pm, false, false).expect("set_hydro_vars");
    let mut m_pred = m_pm.clone();

    let _absorbed = do_conv_premix(&mut m_pm, 5, true).expect("do_conv_premix");
    assert!(m_pm.aux.eps_pre_mix.iter().all(|v| v.is_finite()));
    for k in 0..m_pm.nz {
        let sum_x: f64 = m_pm.cell_xa(k).iter().sum();
        assert!((sum_x - 1.0).abs() < 1.0e-12);
    }

    assert_eq!(
        apply_predictive_mixing_from_controls(&mut m_pm)
            .expect("disabled predictive mixing returns 0"),
        0
    );
    // Construct a realistic multi-cell H/He composition gradient inside the
    // Schwarzschild core under the Ledoux criterion so `gradL` stabilizes a
    // thick gradient zone above the convective core, and verbatim predictive
    // mixing (`predictive_mix.f90`) expands the core outward under the
    // trial-mixed Schwarzschild criterion.
    m_pred.controls.do_conv_premix = false;
    m_pred.controls.use_ledoux_criterion = true;
    m_pred
        .controls
        .mix_wind
        .num_cells_for_smooth_gradl_composition_term = 0;
    m_pred.controls.mix_wind.num_cells_for_smooth_brunt_b = 0;
    let _ = locate_convection_boundaries(&mut m_pred)
        .expect("locate_convection_boundaries");
    let k_bdy = m_pred.aux.mixing.conv_bdy_loc[0];
    let idx_h1 = m_pred
        .iso_names
        .iter()
        .position(|s| s == "h1")
        .expect("h1 present");
    let idx_he4 = m_pred
        .iso_names
        .iter()
        .position(|s| s == "he4")
        .expect("he4 present");
    let sp = m_pred.species;
    let k_grad_top = (k_bdy + 30).min(m_pred.nz - 25);
    let k_grad_bot = (k_grad_top + 12).min(m_pred.nz - 10);
    let span = f64::from(
        u32::try_from((k_grad_bot - k_grad_top).max(1)).expect("small span"),
    );
    for k in k_grad_top..m_pred.nz {
        let frac = if k < k_grad_bot {
            f64::from(u32::try_from(k - k_grad_top).expect("small offset"))
                / span
        } else {
            1.0
        };
        let delta = (0.015 * frac).min(m_pred.xa[k * sp + idx_h1]);
        m_pred.xa[k * sp + idx_h1] -= delta;
        m_pred.xa[k * sp + idx_he4] += delta;
    }
    set_hydro_vars(&mut m_pred, false, false).expect("refresh before pred");
    let ps0 = &mut m_pred.controls.mix_wind.predictive_sets[0];
    ps0.predictive_mix = true;
    ps0.zone_type = String::from("any");
    ps0.zone_loc = String::from("core");
    ps0.bdy_loc = String::from("top");
    ps0.superad_thresh = 0.0;
    let extended = apply_predictive_mixing_from_controls(&mut m_pred)
        .expect("predictive mixing succeeds");
    assert!(
        extended > 0,
        "predictive mixing must extend Ledoux-stabilized convective core"
    );
    assert!(m_pred.d_mix.iter().all(|v| v.is_finite()));
    assert!(m_pred.conv_vel.iter().all(|v| v.is_finite()));

    let mut m_heat = load_zams_model(
        StarControls {
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 1 Msun ZAMS");
    set_hydro_vars(&mut m_heat, false, false).expect("set_hydro_vars");
    m_heat.controls.mix_wind.inject_uniform_extra_heat = 1.0e2;
    m_heat.controls.mix_wind.min_q_for_uniform_extra_heat = 0.2;
    m_heat.controls.mix_wind.max_q_for_uniform_extra_heat = 0.8;

    apply_step_pre_physics(&mut m_heat, 1.0e10).expect("pre_physics");
    let mut total_extra = 0.0_f64;
    for k in 0..m_heat.nz {
        let q_bot = if k + 1 < m_heat.nz {
            m_heat.q[k + 1]
        } else {
            0.0
        };
        let overlap = (m_heat.q[k].min(0.8) - q_bot.max(0.2)).max(0.0);
        let expected = 1.0e2 * overlap / m_heat.dq[k].max(1.0e-30);
        assert!((m_heat.aux.extra_heat[k] - expected).abs() < 1.0e-11);
        total_extra += m_heat.aux.extra_heat[k] * m_heat.dm[k];
    }
    assert!(rel_err(total_extra, 1.0e2 * m_heat.xmstar * 0.6) < 1.0e-12);

    let rep_dedt = star_evolve_step(&mut m_heat, false, false, true)
        .expect("evolve_step Dedt");
    assert!(rep_dedt.converged);
    assert!(m_heat.aux.total_energy.is_finite());
    assert!(m_heat
        .aux
        .step_hist
        .error_in_energy_conservation
        .is_finite());

    m_heat.controls.energy_eqn_option = EnergyEqnOption::EpsGrav;
    let rep_eps = star_evolve_step(&mut m_heat, false, false, true)
        .expect("evolve_step EpsGrav");
    assert!(rep_eps.converged);
    // `#151` (`star/private/evolve.f90:1356-1357`): `eps_grav` mode computes
    // the genuine global total-energy conservation error rather than `0.0`.
    assert!(m_heat
        .aux
        .step_hist
        .error_in_energy_conservation
        .is_finite());
    assert_ne!(m_heat.aux.step_hist.error_in_energy_conservation, 0.0);
}

#[test]
fn test_timestep_controller_hard_limit_triggers_retry_and_termination_stops_evolve(
) {
    let mut model = load_zams_model(
        StarControls {
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 1 Msun ZAMS");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    star_evolve_step(&mut model, false, false, true).expect("initial step");

    let mut m_retry = model.clone();
    m_retry
        .controls
        .ts_stop
        .hard_limit_for_rel_error_in_energy_conservation = 1.0e-12;
    m_retry.aux.step_hist.error_in_energy_conservation = 1.0e40;
    m_retry.aux.total_energy = 1.0e48;
    let (action, _) = timestep_controller(&mut m_retry);
    assert!(matches!(action, TimestepAction::Retry(_)));

    let mut m_lgt_retry = model.clone();
    m_lgt_retry.controls.ts_stop.delta_lgt_cntr_hard_limit = 0.001;
    let kc = m_lgt_retry.nz - 1;
    let nc = m_lgt_retry.species;
    m_lgt_retry.xa_old[kc * nc] = 0.30;
    m_lgt_retry.xa[kc * nc] = 0.25;
    m_lgt_retry.ln_t[kc] = m_lgt_retry.ln_t_old[kc] + 0.05;
    let (action_t, _) = timestep_controller(&mut m_lgt_retry);
    assert!(matches!(action_t, TimestepAction::Retry(_)));

    let mut m_term = model;
    let start_num = m_term.model_number;
    m_term.controls.max_model_number = Some(start_num + 2);
    let reports =
        star_evolve(&mut m_term, 6, false, false, true).expect("star_evolve");
    assert_eq!(reports.len(), 2);
    assert!(matches!(
        check_termination_limits(&m_term),
        Ok(flexmesa_star::LimitCheckOutcome::Terminate { .. })
    ));
}

#[test]
fn test_element_diffusion_pure_h_envelope_preserves_unit_composition_and_do_kap(
) {
    let mut m = load_zams_model(
        StarControls {
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 1 Msun ZAMS");
    set_hydro_vars(&mut m, false, false).expect("set_hydro_vars");
    let nc = m.species;
    for k in 0..5.min(m.nz) {
        m.xa[k * nc..(k + 1) * nc].fill(0.0);
        m.xa[k * nc] = 1.0 - 5.0e-15;
        m.xa[k * nc + 1] = 5.0e-15;
    }
    let _ =
        do_element_diffusion(&mut m, 1.0e8, DiffusionScheme::StantonMurillo)
            .expect("do_element_diffusion");
    assert!(m.xa.iter().all(|&x| (0.0..=1.0).contains(&x)));
    for k in 0..m.nz {
        assert!((m.cell_xa(k).iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
    }
    set_basic_vars(&mut m).expect("set_basic_vars");
    do_kap(&mut m).expect("do_kap");
    assert!(m.opacity.iter().all(|v| v.is_finite() && *v > 0.0));
}

#[test]
fn test_compute_core_masses_matches_subcell_find0_oracles() {
    let mut m = load_zams_model(
        StarControls {
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 1 Msun ZAMS");
    set_hydro_vars(&mut m, false, false).expect("set_hydro_vars");
    let nc = m.species;
    let j_he4 = m.iso_names.iter().position(|n| n == "he4").unwrap();
    let j_c12 = m.iso_names.iter().position(|n| n == "c12").unwrap();
    m.controls.ts_stop.he_core_boundary_h1_fraction = 0.01;
    m.controls.ts_stop.co_core_boundary_he4_fraction = 0.10;
    m.controls.ts_stop.neutron_rich_core_boundary_ye_max = 0.49;
    let (k_he, k_co, k_nr) = (m.nz / 3, m.nz / 2, (3 * m.nz) / 4);
    for k in 0..m.nz {
        m.xa[k * nc..(k + 1) * nc].fill(0.0);
        if k + 1 < k_he {
            m.xa[k * nc] = 0.70;
            m.xa[k * nc + j_he4] = 0.30;
        } else if k + 1 == k_he {
            // `jm1 == j00 == j_he4` between `k_he - 1` and `k_he` -> midpoint `0.5*(qm1 + q00)`
            // per `set_core_info` (`star/private/report.f90:1232-1243`).
            m.xa[k * nc] = 0.03;
            m.xa[k * nc + j_he4] = 0.97;
        } else if k < k_co - 1 {
            m.xa[k * nc] = 0.005;
            m.xa[k * nc + j_he4] = 0.995;
        } else if k == k_co - 1 {
            // `jm1 = j_he4 (0.60 > 0.40)` at `k_co - 1`, `j00 = j_c12 (0.95 > 0.05)` at `k_co`
            // -> `find0(qm1, dm1, q00, d00)` crossover (`report.f90:1232-1250`).
            m.xa[k * nc + j_he4] = 0.60;
            m.xa[k * nc + j_c12] = 0.40;
        } else {
            m.xa[k * nc + j_he4] = 0.05;
            m.xa[k * nc + j_c12] = 0.95;
        }
        m.ye[k] = match (k + 1).cmp(&k_nr) {
            std::cmp::Ordering::Less => 0.50,
            std::cmp::Ordering::Equal => 0.496,
            std::cmp::Ordering::Greater => 0.486,
        };
    }
    let cores = compute_core_masses(&m);
    let q_mid = |k: usize| m.q[k] - 0.5 * m.dq[k];
    let q_he = f64::midpoint(q_mid(k_he - 1), q_mid(k_he));
    let delta_outer_co = 0.40 - 0.60;
    let delta_inner_co = 0.95 - 0.05;
    let q_co = q_mid(k_co - 1)
        + (-delta_outer_co / (delta_inner_co - delta_outer_co))
            * (q_mid(k_co) - q_mid(k_co - 1));
    let q_nr = f64::midpoint(q_mid(k_nr - 1), q_mid(k_nr));
    assert_eq!(
        cores.he_bdy.kbdy_1based,
        flexmesa_star::core_boundaries::k_for_q_1based(&m, q_he) + 1
    );
    assert_eq!(
        cores.co_bdy.kbdy_1based,
        flexmesa_star::core_boundaries::k_for_q_1based(&m, q_co) + 1
    );
    assert_eq!(
        cores.neutron_rich_bdy.kbdy_1based,
        flexmesa_star::core_boundaries::k_for_q_1based(&m, q_nr) + 1
    );
    let m_he = (m.m_center + m.xmstar * q_he) / M_SUN;
    let m_co = (m.m_center + m.xmstar * q_co) / M_SUN;
    let m_nr = (m.m_center + m.xmstar * q_nr) / M_SUN;
    assert!((cores.he_core_mass - m_he).abs() < 1.0e-12);
    assert!((cores.co_core_mass - m_co).abs() < 1.0e-12);
    assert!((cores.neutron_rich_core_mass - m_nr).abs() < 1.0e-12);
}

#[test]
fn test_star_evolve_step_terminate_rolls_back_state_and_eps_mdot() {
    let mut ctrl = StarControls {
        mass_change: -1.0e-7,
        zbase: 0.02,
        ..Default::default()
    };
    ctrl.mix_wind.eps_mdot_factor = 1.0;
    let mut m = load_zams_model(ctrl, StarJobControls::default())
        .expect("load 1 Msun ZAMS");
    set_hydro_vars(&mut m, false, false).expect("set_hydro_vars");
    star_evolve_step(&mut m, false, false, true).expect("baseline step");
    assert!(m.eps_mdot.iter().any(|&v| v.abs() > 0.0));
    let (mod0, age0, eps0) = (m.model_number, m.star_age, m.eps_mdot.clone());
    m.controls.mass_change = 1.0e-6;
    m.controls.varcontrol_target = 1.0e-12;
    m.controls.ts_stop.min_allowed_varcontrol_target = 1.0e-6;
    let err = star_evolve_step(&mut m, false, false, true)
        .expect_err("timestep controller terminate must error");
    assert!(
        matches!(
            err,
            StarError::Terminated { model_number, .. } if model_number == mod0 + 1
        ),
        "expected StarError::Terminated at {}, got {err:?}",
        mod0 + 1
    );
    assert_eq!(m.model_number, mod0);
    assert_eq!(m.star_age, age0);
    assert_eq!(m.eps_mdot, eps0);
}

/// A mass change that would remove more than the envelope is a retriable
/// failure, as upstream's `ierr = -1` (`adjust_mass.f90:260-263`) leads to a
/// retry with a smaller timestep (`evolve.f90:591`), not a fatal input error.
#[test]
fn test_oversized_mass_loss_is_a_retriable_failure() {
    let mut model = load_zams_model(
        StarControls {
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 1 Msun ZAMS");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    let dt = 1.0e3 * SECYER;
    model.mstar_dot = -2.0 * model.xmstar / dt;
    let err = do_adjust_mass_step(&mut model, dt)
        .expect_err("removing twice the envelope mass must fail");
    assert!(err.is_retriable(), "expected a retriable error, got {err}");
}
