// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration tests for Phase 5b (`#113`): adaptive remeshing (`do_adjust_mesh`),
//! Björklund/Dutch/rotational winds (`do_set_mdot`), 3-region mass adjustment
//! and `eps_mdot` (`do_adjust_mass_step`), 16-set overshooting + Péclet damping,
//! `do_conv_premix` + predictive mixing + extra heat + `star_evolve_step` energy
//! conservation, and `timestep_controller` retry / `star_evolve` termination.

use flexmesa_const::{
    MixingType, CLIGHT, FOUR_PI, L_SUN, M_SUN, SECYER, STANDARD_CGRAV,
};
use flexmesa_star::{
    apply_overshooting, apply_predictive_mixing_from_controls,
    apply_step_pre_physics, check_termination_limits, do_adjust_mass_step,
    do_adjust_mesh, do_conv_premix, do_set_mdot, eval_bjorklund_wind,
    eval_de_jager_wind, eval_dutch_wind, eval_nieuwenhuijzen_wind,
    eval_nugis_lamers_wind, eval_omega_crit, eval_van_loon_wind,
    eval_vink_wind, find_mass_flux, load_zams_model,
    locate_convection_boundaries, set_hydro_vars, star_evolve,
    star_evolve_step, timestep_controller, DutchLowTScheme, EnergyEqnOption,
    MeshFunctionConfig, StarControls, StarJobControls, StarModel,
    TimestepAction,
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
    let mut model =
        load_zams_model(StarControls::default(), StarJobControls::default())
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
    cfg_fine.max_surface_cell_dq = 1.0;
    cfg_fine.max_center_cell_dq = 1.0;
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
    cfg_coarse.max_surface_cell_dq = 1.0;
    cfg_coarse.max_center_cell_dq = 1.0;
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
    let expected_bjork = 10.0_f64.powf(-5.52) * (1.0 - gamma_e).powf(-1.48);
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
        ..Default::default()
    };
    ctrl.mix_wind.hot_wind_scheme = String::from("Bjorklund");
    ctrl.mix_wind.bjorklund_scaling_factor = 1.0;
    let mut model = load_zams_model(ctrl, StarJobControls::default())
        .expect("load 4 Msun ZAMS");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");

    model.controls.mix_wind.min_star_mass_for_loss = 5.0;
    let mdot_gated = do_set_mdot(&mut model).expect("gated mdot");
    assert_eq!(mdot_gated, 0.0);

    model.controls.mix_wind.min_star_mass_for_loss = -1.0;
    let mdot_no_rot = do_set_mdot(&mut model).expect("non-rot mdot");
    assert!(mdot_no_rot < 0.0 && mdot_no_rot.is_finite());

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
    model.omega[0] = 0.5 * omega_crit;
    let mdot_rot = do_set_mdot(&mut model).expect("rotational mdot");
    let expected_boost = 0.5_f64.powf(-0.43);
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
        accrete_given_mass_fractions: true,
        accretion_h1: 0.40,
        accretion_he4: 0.58,
        rotation_flag: true,
        ..Default::default()
    };
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
fn test_overshoot_16_param_sets_and_peclet_damping_follow_mesa_rules() {
    let ctrl = StarControls {
        initial_mass: 4.0,
        ..Default::default()
    };
    let mut base = load_zams_model(ctrl, StarJobControls::default())
        .expect("load 4 Msun ZAMS");
    set_hydro_vars(&mut base, false, false).expect("set_hydro_vars");
    let (_, boundaries) = locate_convection_boundaries(&base);
    assert!(
        !boundaries.is_empty(),
        "4 Msun ZAMS must have convective core"
    );

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

    let mut m_peclet = m_exp.clone();
    m_peclet.controls.mix_wind.peclet_function_full_off = 1.0e30;
    m_peclet.controls.mix_wind.peclet_function_full_on = 1.0e31;

    apply_overshooting(&mut m_exp, &boundaries, &[]);
    apply_overshooting(&mut m_short_circuit, &boundaries, &[]);
    apply_overshooting(&mut m_peclet, &boundaries, &[]);

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
    let n_ov_peclet = m_peclet
        .mixing_type
        .iter()
        .filter(|&&mt| mt == MixingType::OvershootMixing)
        .count();

    assert!(n_ov_exp > 0, "expected OvershootMixing zones from set 0");
    assert_eq!(n_ov_short, 0, "'none' in set 0 must short-circuit set 1");
    assert_eq!(
        n_ov_peclet, 0,
        "Pe <= peclet_function_full_off must damp overshoot diffusivity to zero"
    );
}

#[test]
fn test_conv_premix_predictive_mixing_and_extra_heat_in_evolve_step() {
    let ctrl = StarControls {
        initial_mass: 4.0,
        do_conv_premix: true,
        do_premix_heating: true,
        ..Default::default()
    };
    let mut m_pm = load_zams_model(ctrl, StarJobControls::default())
        .expect("load 4 Msun ZAMS");
    set_hydro_vars(&mut m_pm, false, false).expect("set_hydro_vars");

    let _absorbed = do_conv_premix(&mut m_pm, 5, true);
    assert!(m_pm.aux.eps_pre_mix.iter().all(|v| v.is_finite()));
    for k in 0..m_pm.nz {
        let sum_x: f64 = m_pm.cell_xa(k).iter().sum();
        assert!((sum_x - 1.0).abs() < 1.0e-12);
    }

    assert_eq!(apply_predictive_mixing_from_controls(&mut m_pm), 0);
    m_pm.controls.mix_wind.predictive_sets[0].predictive_mix = true;
    let _extended = apply_predictive_mixing_from_controls(&mut m_pm);
    assert!(m_pm.d_mix.iter().all(|v| v.is_finite()));
    assert!(m_pm.conv_vel.iter().all(|v| v.is_finite()));

    let mut m_heat =
        load_zams_model(StarControls::default(), StarJobControls::default())
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
    assert_eq!(m_heat.aux.step_hist.error_in_energy_conservation, 0.0);
}

#[test]
fn test_timestep_controller_hard_limit_triggers_retry_and_termination_stops_evolve(
) {
    let mut model =
        load_zams_model(StarControls::default(), StarJobControls::default())
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
    assert!(check_termination_limits(&m_term).is_some());
}
