// SPDX-License-Identifier: LGPL-3.0-or-later
//! Phase 5a Henyey convergence, variable-temperature `op_split_burn`, `.photo`
//! RTI/Riemann serialization, and `sizequ`/`sizeB` norm filter verification
//! (`Issue #112`).

use flexmesa_const::SECYER;
use flexmesa_star::photo_io::{read_photo_bytes, write_photo_bytes};
use flexmesa_star::solver_norms::compute_residual_norms;
use flexmesa_star::{
    active_solver_tolerances, change_model_network,
    compute_model_correction_norms, do_struct_burn_mix, load_zams_model,
    set_hydro_vars, solve_henyey_step, StarControls, StarJobControls,
    StarModel,
};

fn usize_to_f64(n: usize) -> f64 {
    f64::from(u32::try_from(n).expect("small test zone index fits in u32"))
}

fn build_compact_zams(nz_target: usize, controls: StarControls) -> StarModel {
    let full = load_zams_model(controls, StarJobControls::default())
        .expect("1 Msun ZAMS model should load");
    let nz = nz_target.max(4).min(full.nz);
    let lnp_min = full.ln_peos[0];
    let lnp_max = full.ln_peos[full.nz - 1];
    let mut bounds = Vec::with_capacity(nz + 1);
    bounds.push(0);
    let mut prev = 0_usize;
    for b in 1..nz {
        let target_lnp =
            lnp_min + usize_to_f64(b) * (lnp_max - lnp_min) / usize_to_f64(nz);
        let max_idx = full.nz - (nz - b);
        let idx = (prev + 1..=max_idx)
            .find(|&s| full.ln_peos[s] >= target_lnp)
            .unwrap_or(max_idx)
            .clamp(prev + 1, max_idx);
        bounds.push(idx);
        prev = idx;
    }
    bounds.push(full.nz);

    let mut m = StarModel::new(
        nz,
        full.chem_id.clone(),
        full.job.clone(),
        full.controls.clone(),
    );
    m.mstar = full.mstar;
    m.mstar_start = full.mstar;
    m.mstar_old = full.mstar;
    m.xmstar = full.xmstar;
    m.r_center = full.r_center;
    m.m_center = full.m_center;
    m.l_center = full.l_center;
    m.dt = 1.0e2 * SECYER;
    m.dt_next = m.dt;
    m.dt_old = m.dt;
    for dst_k in 0..nz {
        let k0 = bounds[dst_k];
        let k1 = bounds[dst_k + 1];
        let cell_dq_total: f64 = full.dq[k0..k1].iter().sum();
        let cell_mass_total: f64 = full.dm[k0..k1].iter().sum();
        let vol_sum: f64 = (k0..k1)
            .map(|s| full.dm[s] / full.lnd[s].exp().max(1.0e-30))
            .sum();
        let lnt_sum: f64 = (k0..k1).map(|s| full.dm[s] * full.ln_t[s]).sum();
        m.dq[dst_k] = cell_dq_total;
        m.lnd[dst_k] = (cell_mass_total / vol_sum.max(1.0e-30)).ln();
        m.ln_t[dst_k] = lnt_sum / cell_mass_total.max(1.0e-30);
        m.ln_r[dst_k] = full.ln_r[k0];
        m.lum[dst_k] = full.lum[k0];
        for j in 0..m.species {
            let x_sum: f64 = (k0..k1)
                .map(|s| full.dm[s] * full.xa[s * full.species + j])
                .sum();
            m.xa[dst_k * m.species + j] = x_sum / cell_mass_total.max(1.0e-30);
        }
    }
    m.sync_mesh_and_pack().expect("sync mesh should succeed");
    m.reconstruct_ln_r_from_lnd_and_dm();
    m.finish_init_model().expect("compact init should succeed");
    set_hydro_vars(&mut m, false, false).expect("initial set_hydro_vars");
    m.save_start_values();
    m.save_old_values();
    m
}

#[test]
fn test_coupled_u_flag_hllc_henyey_step_converges_to_strict_residual_tolerances(
) {
    let controls_full = StarControls {
        u_flag: true,
        v_flag: false,
        use_compression_outer_bc: true,
        use_dprad_dm_form_of_t_gradient_eqn: true,
        velocity_q_upper_bound: 2.0,
        velocity_tau_lower_bound: -1.0,
        velocity_logt_lower_bound: -1.0,
        ..StarControls::default()
    };
    let mut full_model =
        load_zams_model(controls_full, StarJobControls::default())
            .expect("full 1 Msun ZAMS model should load");
    full_model.dt = 0.5;
    let r_surf = full_model.r[0].max(1.0);
    for k in 0..full_model.nz {
        full_model.u[k] = 5.0e3 * (full_model.r[k] / r_surf);
    }
    set_hydro_vars(&mut full_model, false, true)
        .expect("u_flag set_hydro_vars");
    full_model.save_start_values();

    let report_full = solve_henyey_step(&mut full_model, false, true)
        .expect("full-star coupled u_flag HLLC Henyey solve should succeed");
    assert!(
        report_full.converged,
        "full u_flag report = {report_full:?}"
    );
    assert!(report_full.max_residual <= 1.0e-3);
    assert!(report_full.residual_norm <= 1.0e-6);
    assert!(full_model.u.iter().all(|u| u.is_finite()));
    assert!(full_model.aux.p_face_ad.iter().all(|p| p.val > 0.0));

    let controls_mom = StarControls {
        u_flag: true,
        v_flag: false,
        use_momentum_outer_bc: true,
        velocity_q_upper_bound: 2.0,
        velocity_tau_lower_bound: -1.0,
        velocity_logt_lower_bound: -1.0,
        ..StarControls::default()
    };
    let mut mom_model = build_compact_zams(24, controls_mom);
    mom_model.dt = 0.5;
    let r_surf_c = mom_model.r[0].max(1.0);
    for k in 0..mom_model.nz {
        mom_model.u[k] = 5.0e3 * (mom_model.r[k] / r_surf_c);
    }
    set_hydro_vars(&mut mom_model, false, true)
        .expect("compact u_flag set_hydro_vars");
    mom_model.save_start_values();

    let report_mom = solve_henyey_step(&mut mom_model, false, true)
        .expect("momentum-BC u_flag HLLC Henyey solve should succeed");
    assert!(report_mom.converged, "mom u_flag report = {report_mom:?}");
    assert!(report_mom.max_residual <= 1.0e-3);
    assert!(report_mom.residual_norm <= 1.0e-6);
}

#[test]
fn test_multi_isotope_approx21_coupled_and_variable_t_const_density_op_split_burn_converge(
) {
    let mut model =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("1 Msun ZAMS model should load");
    change_model_network(&mut model, "approx21.net", None)
        .expect("switching to 21-isotope approx21.net should succeed");
    assert_eq!(model.species, 21);
    set_hydro_vars(&mut model, false, false).expect("approx21 set_hydro_vars");
    model.save_start_values();
    model.save_old_values();

    // 1. Coupled 21-species Henyey solve (`op_split_burn = false`).
    let mut m_coupled = model.clone();
    m_coupled.controls.op_split_burn = false;
    let dt = 10.0 * SECYER;
    let rep_coupled = do_struct_burn_mix(&mut m_coupled, dt, false, true, &[])
        .expect("coupled 21-isotope do_struct_burn_mix should succeed");
    assert!(rep_coupled.converged, "coupled approx21 = {rep_coupled:?}");
    for k in 0..m_coupled.nz {
        let sum_x: f64 = m_coupled.cell_xa(k).iter().sum();
        assert!((sum_x - 1.0).abs() < 1.0e-12);
    }

    // 2. Variable-T `burn_1_zone_const_density` (`T_c ~ 1.5e7 > 1.2e7` K).
    let mut m_var_t = model;
    m_var_t.controls.op_split_burn = true;
    m_var_t.controls.op_split_burn_min_t = 1.0e7;
    m_var_t.controls.op_split_burn_min_t_for_variable_t_solver = 1.2e7;
    assert!(m_var_t.ln_t[m_var_t.nz - 1].exp() > 1.2e7);
    let rep_var_t = do_struct_burn_mix(&mut m_var_t, dt, true, true, &[])
        .expect("variable-T op_split_burn approx21 step should succeed");
    assert!(rep_var_t.converged, "var-T op_split = {rep_var_t:?}");
    assert!(m_var_t.aux.burn_avg_epsnuc[m_var_t.nz - 1] > 0.0);
}

#[test]
fn test_photo_rti_and_u_flag_roundtrip_bit_parity_and_buffer_validation() {
    let controls = StarControls {
        u_flag: true,
        v_flag: false,
        rti_flag: true,
        use_momentum_outer_bc: true,
        ..StarControls::default()
    };
    let mut model = build_compact_zams(8, controls);
    for k in 0..model.nz {
        model.u[k] = 3.0e5 * usize_to_f64(k + 1);
        model.aux.alpha_rti[k] = 0.04 + 0.01 * usize_to_f64(k);
    }
    set_hydro_vars(&mut model, false, true).expect("u_flag + rti hydro vars");
    model.save_start_values();
    model.aux.alpha_rti[2] = 0.19;
    model.pack_xh();

    let bytes = write_photo_bytes(&model);
    let restored =
        read_photo_bytes(&bytes, model.controls.clone(), model.job.clone())
            .expect("photo round-trip should succeed");
    assert_eq!(restored.layout.nvar_hydro, model.layout.nvar_hydro);
    for k in 0..model.nz {
        assert_eq!(
            restored.aux.alpha_rti[k].to_bits(),
            model.aux.alpha_rti[k].to_bits()
        );
        assert_eq!(
            restored.aux.alpha_rti_start[k].to_bits(),
            model.aux.alpha_rti_start[k].to_bits()
        );
        assert_eq!(
            restored.aux.p_face_start[k].to_bits(),
            model.aux.p_face_start[k].to_bits()
        );
    }

    let mut trailing = bytes.clone();
    trailing.push(0xFF);
    assert!(
        read_photo_bytes(&trailing, model.controls.clone(), model.job.clone())
            .is_err(),
        "trailing garbage bytes must fail read_photo_bytes"
    );
    assert!(
        read_photo_bytes(
            &bytes[..bytes.len() - 4],
            model.controls.clone(),
            model.job.clone()
        )
        .is_err(),
        "truncated photo bytes must fail read_photo_bytes"
    );
}

#[test]
fn test_sizeb_filter_flags_nan_propagation_and_strict_unconverged_rejection() {
    let mut model = build_compact_zams(
        6,
        StarControls {
            rti_flag: true,
            ..StarControls::default()
        },
    );
    let nvar = model.layout.nvar(true);
    let x_scale = vec![1.0; model.nz * nvar];
    let mut dx_scaled = vec![0.0; model.nz * nvar];

    // 1. Trace species (`xa_start < 5e-3`) filtered from `sizeB`.
    let trace_species = (0..model.species)
        .find(|&j| model.xa_start[j] < model.controls.correction_xa_limit)
        .expect("ZAMS should have at least one trace species < 5e-3");
    dx_scaled[model.layout.chem_var_index(trace_species)] = 0.8;
    let (cn_filtered, mc_filtered) = compute_model_correction_norms(
        &model, model.nz, nvar, &dx_scaled, &x_scale,
    );
    assert_eq!(cn_filtered, 0.0);
    assert_eq!(mc_filtered, 0.0);

    // 2. `NaN` in a filtered trace-species slot MUST still propagate!
    dx_scaled[model.layout.chem_var_index(trace_species)] = f64::NAN;
    let (cn_nan, mc_nan) = compute_model_correction_norms(
        &model, model.nz, nvar, &dx_scaled, &x_scale,
    );
    assert!(cn_nan.is_nan() && mc_nan.is_nan());
    dx_scaled[model.layout.chem_var_index(trace_species)] = 0.0;

    // 3. `ignore_species_in_max_correction` & `include_l_in_correction_limits`.
    dx_scaled[model.layout.chem_var_index(0)] = 0.8;
    model.controls.ignore_species_in_max_correction = true;
    let (cn_no_sp, mc_no_sp) = compute_model_correction_norms(
        &model, model.nz, nvar, &dx_scaled, &x_scale,
    );
    assert!(cn_no_sp > 0.0 && mc_no_sp.abs() <= 1.0e-30);
    model.controls.ignore_species_in_max_correction = false;
    dx_scaled[model.layout.chem_var_index(0)] = 0.0;
    dx_scaled[model.layout.i_lum] = 1.0e33;
    model.controls.include_l_in_correction_limits = false;
    let (_, mc_l_off) = compute_model_correction_norms(
        &model, model.nz, nvar, &dx_scaled, &x_scale,
    );
    assert!(mc_l_off.abs() <= 1.0e-30);
    model.controls.include_l_in_correction_limits = true;
    let (_, mc_l_on) = compute_model_correction_norms(
        &model, model.nz, nvar, &dx_scaled, &x_scale,
    );
    assert!(mc_l_on > 0.0);
    model.controls.include_l_in_correction_limits = false;
    dx_scaled[model.layout.i_lum] = 0.0;

    // 4. `compute_residual_norms` `NaN` propagation through skipped `i_equ_l`.
    let mut equ = vec![0.0; model.nz * nvar];
    model.controls.convergence_ignore_equl_residuals = true;
    equ[model.layout.i_equ_l] = f64::NAN;
    let (res_norm_nan, max_res_nan) =
        compute_residual_norms(&model, nvar, &equ);
    assert!(res_norm_nan.is_nan() && max_res_nan.is_nan());
    model.controls.convergence_ignore_equl_residuals = false;

    // 5. 3-stage `gold2` schedule and strict rejection of unconverged 1-iter solve.
    model.model_number = 200;
    model.controls.use_gold_tolerances = false;
    model.controls.use_gold2_tolerances = false;
    model.controls.steps_before_use_gold2_tolerances = 100;
    let (rn1, mr1, _, _) = active_solver_tolerances(&model, 1);
    let (rn2, mr2, _, _) = active_solver_tolerances(&model, 7);
    let (rn3, mr3, _, _) = active_solver_tolerances(&model, 15);
    assert!(
        (rn1 - model.controls.gold2_tol_residual_norm1).abs() <= 1.0e-25
            && (mr1 - model.controls.gold2_tol_max_residual1).abs() <= 1.0e-25
    );
    assert!(
        (rn1 - 1.0e-11).abs() <= 1.0e-25
            && rn1 < rn2
            && rn2 < rn3
            && mr1 < mr2
            && mr2 < mr3
    );

    let mut perturbed = model;
    perturbed.controls.max_newton_iters = 1;
    perturbed.controls.use_gold_tolerances = true;
    perturbed.lnd[2] += 0.15;
    perturbed.pack_xh();
    let rep = solve_henyey_step(&mut perturbed, false, true).expect("1-iter");
    assert!(
        !rep.converged,
        "unconverged 1-iter solve must fail: {rep:?}"
    );
}
