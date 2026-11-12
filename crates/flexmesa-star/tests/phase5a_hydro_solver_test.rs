// SPDX-License-Identifier: LGPL-3.0-or-later
//! Phase 5a verification suite (`Issue #112`):
//! 1. Central finite-difference vs. analytical block-tridiagonal Jacobian
//!    verification across coupled `(lnd, lnT, lnR, L, X_j)` (`Dedt` & `EpsGrav`
//!    with EOS `d_dxa` composition partials), coupled `u_flag` HLLC Riemann
//!    `(lnd, lnT, lnR, L, u)`, and coupled `v_flag + rti_flag`
//!    `(lnd, lnT, lnR, L, v, alpha_rti)`.
//! 2. Coupled `u_flag` HLLC Riemann Henyey step convergence.
//! 3. `approx21.net` (21-species) coupled Henyey step and thresholded
//!    `op_split_burn` (`op_split_burn_min_t = 2e9` K) step convergence.
//! 4. `sizeB` trace-species (`correction_xa_limit`) filtering, 3-stage
//!    `gold`/`gold2`/standard residual tolerance schedule, and strict rejection
//!    of unconverged states (`mean_norm_ok` deleted).

use flexmesa_const::SECYER;
use flexmesa_star::{
    active_solver_tolerances, change_model_network,
    compute_model_correction_norms, do_struct_burn_mix, eval_henyey_equations,
    load_zams_model, set_hydro_vars, solve_henyey_step, EnergyEqnOption,
    StarControls, StarJobControls, StarModel,
};

fn usize_to_f64(n: usize) -> f64 {
    f64::from(u32::try_from(n).expect("small test zone index fits in u32"))
}

fn build_compact_zams(
    nz_target: usize,
    controls: StarControls,
    job: StarJobControls,
) -> StarModel {
    let full =
        load_zams_model(controls, job).expect("1 Msun ZAMS model should load");
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
        let mut idx = (prev + 1..=max_idx)
            .find(|&s| full.ln_peos[s] >= target_lnp)
            .unwrap_or(max_idx);
        idx = idx.clamp(prev + 1, max_idx);
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

fn lookup_block_entry(
    sys: &flexmesa_star::HenyeySystem,
    k_equ: usize,
    i_equ: usize,
    k_var: usize,
    j_var: usize,
) -> f64 {
    let blk_sz = sys.nvar * sys.nvar;
    let idx = k_equ * blk_sz + i_equ * sys.nvar + j_var;
    if k_var == k_equ {
        sys.dblk[idx]
    } else if k_var + 1 == k_equ {
        sys.lblk[idx]
    } else {
        sys.ublk[idx]
    }
}

fn assert_hydro_columns_match_fd(
    base_model: &StarModel,
    m0: &StarModel,
    sys0: &flexmesa_star::HenyeySystem,
    k_var: usize,
    include_chem: bool,
    rel_tol: f64,
    abs_tol: f64,
) {
    let nz = sys0.nz;
    let nvar = sys0.nvar;
    let nvar_hydro = m0.layout.nvar_hydro;
    // For `i_lum`, primal `Zeta = exp(3*ln(Gamma) - ln(B^3))` (`mlt.f90:156`)
    // in super-opaque ionization zones (`gradr ~ 1e5, ln(B^3) ~ 55`) has
    // `~2e-8` IEEE-754 ULP noise after multiplying `(1 - Zeta) * gradr`, so
    // `eps_scaled = 1e-3` balances `O(2e-7)` cubic-FD truncation against ULP noise.
    for j_var in 0..nvar_hydro {
        let x_sc = sys0.x_scale[k_var * nvar + j_var];
        let eps_scaled = if j_var == m0.layout.i_lum {
            1.0e-3
        } else {
            1.0e-6
        };
        let delta = eps_scaled * x_sc;

        let mut mp = base_model.clone();
        mp.pack_xh();
        mp.xh[k_var * nvar_hydro + j_var] += delta;
        mp.unpack_xh();
        mp.set_m_grav_and_grav();
        let sys_plus = eval_henyey_equations(&mut mp, include_chem)
            .expect("plus perturbed eqns");

        let mut mm = base_model.clone();
        mm.pack_xh();
        mm.xh[k_var * nvar_hydro + j_var] -= delta;
        mm.unpack_xh();
        mm.set_m_grav_and_grav();
        let sys_minus = eval_henyey_equations(&mut mm, include_chem)
            .expect("minus perturbed eqns");

        for k_equ in 0..nz {
            if k_equ.abs_diff(k_var) > 1 {
                continue;
            }
            for i_equ in 0..nvar {
                let fd = (sys_plus.equ[k_equ * nvar + i_equ]
                    - sys_minus.equ[k_equ * nvar + i_equ])
                    / (2.0 * eps_scaled);
                let analytic =
                    lookup_block_entry(sys0, k_equ, i_equ, k_var, j_var);
                let diff = (analytic - fd).abs();
                let scale = analytic.abs().max(fd.abs()).max(1.0);
                assert!(
                    diff <= abs_tol || diff / scale <= rel_tol,
                    "Hydro Jacobian mismatch at k_equ={k_equ}, i_equ={i_equ}, k_var={k_var}, j_var={j_var}: analytic={analytic:.8e}, fd={fd:.8e}, diff={diff:.3e}"
                );
            }
        }
    }
}

fn assert_species_columns_match_fd(
    base_model: &StarModel,
    m0: &StarModel,
    sys0: &flexmesa_star::HenyeySystem,
    k_var: usize,
    rel_tol: f64,
    abs_tol: f64,
) {
    let use_surf_mom_bc = m0.controls.use_momentum_outer_bc
        || m0.controls.use_zero_pgas_outer_bc
        || m0.controls.use_fixed_psurf_outer_bc;
    let idx_he4 = m0.iso_names.iter().position(|n| n == "he4").unwrap_or(0);
    let nz = sys0.nz;
    let nvar = sys0.nvar;
    let nvar_hydro = m0.layout.nvar_hydro;
    let j_var_he4 = nvar_hydro + idx_he4;
    let x_sc_he4 = sys0.x_scale[k_var * nvar + j_var_he4];
    for j_sp in 0..m0.species {
        let j_var = nvar_hydro + j_sp;
        let x_sc = sys0.x_scale[k_var * nvar + j_var];
        let eps_scaled = 1.0e-6;
        let delta = eps_scaled * x_sc;

        let mut mp = base_model.clone();
        mp.xa[k_var * mp.species + j_sp] += delta;
        let sys_plus =
            eval_henyey_equations(&mut mp, true).expect("chem coord plus eqns");
        let mut mm = base_model.clone();
        mm.xa[k_var * mm.species + j_sp] -= delta;
        let sys_minus = eval_henyey_equations(&mut mm, true)
            .expect("chem coord minus eqns");

        for k_equ in 0..nz {
            if k_equ.abs_diff(k_var) > 1 {
                continue;
            }
            for i_equ in nvar_hydro..nvar {
                let fd = (sys_plus.equ[k_equ * nvar + i_equ]
                    - sys_minus.equ[k_equ * nvar + i_equ])
                    / (2.0 * eps_scaled);
                let analytic =
                    lookup_block_entry(sys0, k_equ, i_equ, k_var, j_var);
                let diff = (analytic - fd).abs();
                let scale = analytic.abs().max(fd.abs()).max(1.0);
                assert!(
                    diff <= abs_tol || diff / scale <= rel_tol,
                    "Chem Jacobian mismatch at k_equ={k_equ}, i_equ={i_equ}, k_var={k_var}, j_sp={j_sp}: analytic={analytic:.8e}, fd={fd:.8e}, diff={diff:.3e}"
                );
            }
        }

        if j_sp == idx_he4 {
            continue;
        }
        let mut msp = base_model.clone();
        msp.xa[k_var * msp.species + j_sp] += delta;
        msp.xa[k_var * msp.species + idx_he4] -= delta;
        let sys_simplex_plus =
            eval_henyey_equations(&mut msp, true).expect("simplex plus eqns");
        let mut msm = base_model.clone();
        msm.xa[k_var * msm.species + j_sp] -= delta;
        msm.xa[k_var * msm.species + idx_he4] += delta;
        let sys_simplex_minus =
            eval_henyey_equations(&mut msm, true).expect("simplex minus eqns");

        for k_equ in 0..nz {
            if k_equ.abs_diff(k_var) > 1 {
                continue;
            }
            for i_equ in 0..nvar {
                let a_j = lookup_block_entry(sys0, k_equ, i_equ, k_var, j_var);
                let a_he4 =
                    lookup_block_entry(sys0, k_equ, i_equ, k_var, j_var_he4);
                let upstream_omits_dxa = i_equ == m0.layout.i_equ_l
                    || (k_equ == 0
                        && i_equ == m0.layout.i_dv_dt
                        && !use_surf_mom_bc);
                if upstream_omits_dxa {
                    assert_eq!(a_j, 0.0);
                    assert_eq!(a_he4, 0.0);
                    continue;
                }
                let fd_dir = (sys_simplex_plus.equ[k_equ * nvar + i_equ]
                    - sys_simplex_minus.equ[k_equ * nvar + i_equ])
                    / (2.0 * eps_scaled);
                let analytic_dir = a_j - a_he4 * (x_sc / x_sc_he4);
                let diff = (analytic_dir - fd_dir).abs();
                let scale = analytic_dir.abs().max(fd_dir.abs()).max(1.0);
                assert!(
                    diff <= abs_tol || diff / scale <= rel_tol,
                    "Simplex dxa Jacobian mismatch at k_equ={k_equ}, i_equ={i_equ}, k_var={k_var}, j_sp={j_sp}: analytic={analytic_dir:.8e}, fd={fd_dir:.8e}, diff={diff:.3e}"
                );
            }
        }
    }
}

fn assert_tridiagonal_jacobian_matches_fd(
    base_model: &StarModel,
    include_chem: bool,
    rel_tol: f64,
    abs_tol: f64,
) {
    let mut m0 = base_model.clone();
    m0.pack_xh();
    let sys0 =
        eval_henyey_equations(&mut m0, include_chem).expect("baseline eqns");

    for k_var in 0..sys0.nz {
        assert_hydro_columns_match_fd(
            base_model,
            &m0,
            &sys0,
            k_var,
            include_chem,
            rel_tol,
            abs_tol,
        );
        if include_chem {
            assert_species_columns_match_fd(
                base_model, &m0, &sys0, k_var, rel_tol, abs_tol,
            );
        }
    }
}

#[test]
fn test_coupled_hydro_chem_jacobian_matches_central_finite_differences_with_eos_dxa_partials(
) {
    for opt in [EnergyEqnOption::Dedt, EnergyEqnOption::EpsGrav] {
        let controls = StarControls {
            energy_eqn_option: opt,
            include_composition_in_eps_grav: true,
            op_split_burn: false,
            redo_conv_for_dr_lt_mixing_length: false,
            ..StarControls::default()
        };
        let mut model =
            build_compact_zams(8, controls, StarJobControls::default());
        // Perturb trial state slightly away from start-of-step so `d(1/rho)/dt != 0`
        // and `ds/dt != 0`, activating `dln_peos_dxa_for_partials` and `d_eps_grav_dx`.
        for k in 0..model.nz {
            model.lnd[k] += 2.0e-3 * (usize_to_f64(k) * 0.2 - 0.5);
            model.ln_t[k] += 1.5e-3 * (0.4 - usize_to_f64(k) * 0.1);
        }
        model.reconstruct_ln_r_from_lnd_and_dm();
        model.set_m_grav_and_grav();
        model.pack_xh();

        assert_tridiagonal_jacobian_matches_fd(&model, true, 5.0e-3, 1.0e-6);
    }
}

#[test]
fn test_coupled_u_flag_hllc_riemann_jacobian_matches_central_finite_differences(
) {
    let controls = StarControls {
        u_flag: true,
        v_flag: false,
        use_momentum_outer_bc: true,
        using_velocity_time_centering: true,
        include_p_in_velocity_time_centering: true,
        redo_conv_for_dr_lt_mixing_length: false,
        velocity_q_upper_bound: 2.0,
        velocity_tau_lower_bound: -1.0,
        velocity_logt_lower_bound: -1.0,
        ..StarControls::default()
    };
    let mut model =
        build_compact_zams(24, controls, StarJobControls::default());
    model.dt = 5.0;
    for k in 0..model.nz {
        model.u[k] =
            2.0e5 * usize_to_f64(model.nz - k) / usize_to_f64(model.nz);
    }
    set_hydro_vars(&mut model, false, true).expect("u_flag hydro vars");
    model.save_start_values();
    for k in 0..model.nz {
        model.u[k] *= 1.02;
        model.lnd[k] += 1.0e-3;
        model.ln_r[k] += 3.0e-4;
    }
    model.set_m_grav_and_grav();
    model.pack_xh();

    assert_tridiagonal_jacobian_matches_fd(&model, false, 3.0e-3, 1.0e-6);
}

#[test]
fn test_coupled_v_flag_rti_alpha_jacobian_matches_central_finite_differences() {
    let controls = StarControls {
        v_flag: true,
        rti_flag: true,
        use_momentum_outer_bc: true,
        redo_conv_for_dr_lt_mixing_length: false,
        velocity_q_upper_bound: 2.0,
        velocity_tau_lower_bound: -1.0,
        velocity_logt_lower_bound: -1.0,
        ..StarControls::default()
    };
    let mut model =
        build_compact_zams(10, controls, StarJobControls::default());
    assert!(model.layout.i_alpha_rti.is_some());
    assert_eq!(model.layout.nvar_hydro, 6);
    model.dt = 10.0;
    for k in 0..model.nz {
        model.v[k] = 1.0e6 * usize_to_f64(k + 1);
        model.aux.alpha_rti[k] = 0.02 + 0.005 * usize_to_f64(k);
    }
    set_hydro_vars(&mut model, false, true).expect("rti hydro vars");
    model.save_start_values();
    // Seed a baroclinic Rayleigh-Taylor unstable layer (`dPdr * drhodr < 0`)
    model.aux.dpdr_drhodr_info[3] = -1.0e-4;
    model.aux.dpdr_drhodr_info[4] = -2.5e-4;
    for k in 0..model.nz {
        model.aux.alpha_rti[k] *= 1.05;
        model.v[k] *= 1.01;
    }
    model.pack_xh();

    assert_tridiagonal_jacobian_matches_fd(&model, false, 3.0e-3, 1.0e-6);
}

#[test]
fn test_coupled_u_flag_hllc_henyey_step_converges_to_strict_residual_tolerances(
) {
    // 1. Full 1000-zone ZAMS model with compression outer BC (`hse_riemann`)
    //    and homologous radial velocity profile `u(r) = u_surf * (r / R_*)`.
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

    // 2. Momentum outer BC (`do_surf_Riemann_dudt_eqn`) on 24-zone compact ZAMS.
    let controls_mom = StarControls {
        u_flag: true,
        v_flag: false,
        use_momentum_outer_bc: true,
        velocity_q_upper_bound: 2.0,
        velocity_tau_lower_bound: -1.0,
        velocity_logt_lower_bound: -1.0,
        ..StarControls::default()
    };
    let mut mom_model =
        build_compact_zams(24, controls_mom, StarJobControls::default());
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
    assert!(mom_model.aux.p_face_ad.iter().all(|p| p.val > 0.0));
}

#[test]
fn test_multi_isotope_approx21_coupled_and_thresholded_op_split_burn_converge()
{
    let mut model =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("full 1 Msun ZAMS model should load");
    change_model_network(&mut model, "approx21.net", None)
        .expect("switching to 21-isotope approx21.net should succeed");
    assert_eq!(model.species, 21);
    set_hydro_vars(&mut model, false, false).expect("approx21 set_hydro_vars");
    model.save_start_values();
    model.save_old_values();

    // 1. Coupled 21-species Henyey solve (`op_split_burn = false`)
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

    // 2. Thresholded `op_split_burn = true` with low threshold so hot inner zones
    // execute `burn_1_zone_const_density` while cool outer zones stay coupled.
    let mut m_split = model;
    m_split.controls.op_split_burn = true;
    m_split.controls.op_split_burn_min_t = 1.0e7;
    let rep_split = do_struct_burn_mix(&mut m_split, dt, true, true, &[])
        .expect("thresholded op_split_burn approx21 step should succeed");
    assert!(rep_split.converged, "op_split approx21 = {rep_split:?}");
    assert!(m_split.aux.burn_avg_epsnuc[m_split.nz - 1] > 0.0);
}

#[test]
fn test_sizeb_trace_species_filter_and_strict_rejection_of_unconverged_residuals(
) {
    let mut model = build_compact_zams(
        6,
        StarControls::default(),
        StarJobControls::default(),
    );
    let nvar = model.layout.nvar(true);
    let x_scale = vec![1.0; model.nz * nvar];
    let mut dx_scaled = vec![0.0; model.nz * nvar];

    // Put a huge scaled correction (`0.8`) on a trace species (`xa_start < 5e-3`)
    let trace_species = (0..model.species)
        .find(|&j| model.xa_start[j] < model.controls.correction_xa_limit)
        .expect("ZAMS should have at least one trace species < 5e-3");
    dx_scaled[model.layout.chem_var_index(trace_species)] = 0.8;
    let (cn_filtered, mc_filtered) = compute_model_correction_norms(
        &model, model.nz, nvar, &dx_scaled, &x_scale,
    );
    assert_eq!(cn_filtered, 0.0);
    assert_eq!(mc_filtered, 0.0);

    // Put the same correction on abundant H1 (`xa_start > 0.5 >= 5e-3`)
    dx_scaled[model.layout.chem_var_index(0)] = 0.8;
    let (_cn_abundant, mc_abundant) = compute_model_correction_norms(
        &model, model.nz, nvar, &dx_scaled, &x_scale,
    );
    assert!((mc_abundant - 0.8).abs() < 1.0e-14);

    // Check 3-stage `gold` / `gold2` / non-gold tolerance schedules
    model.model_number = 200;
    model.controls.use_gold_tolerances = false;
    model.controls.use_gold2_tolerances = false;
    model.controls.steps_before_use_gold2_tolerances = 100;
    let (rn1, mr1, _, _) = active_solver_tolerances(&model, 1);
    let (rn2, mr2, _, _) = active_solver_tolerances(&model, 7);
    let (rn3, mr3, _, _) = active_solver_tolerances(&model, 15);
    assert_eq!(rn1, model.controls.gold2_tol_residual_norm1);
    assert_eq!(mr1, model.controls.gold2_tol_max_residual1);
    assert_eq!(rn2, model.controls.gold2_tol_residual_norm2);
    assert_eq!(mr2, model.controls.gold2_tol_max_residual2);
    assert_eq!(rn3, model.controls.gold2_tol_residual_norm3);
    assert_eq!(mr3, model.controls.gold2_tol_max_residual3);

    model.controls.steps_before_use_gold2_tolerances = -1;
    let (rn_std2, _, _, _) = active_solver_tolerances(&model, 8);
    assert!(rn_std2 > 1.0e50);

    // Verify `mean_norm_ok` (`max_residual <= 0.1`) is deleted: a 1-iteration
    // solve on a perturbed model with `max_residual ~ 1e-2` must NOT claim convergence.
    let mut perturbed = model;
    perturbed.controls.max_newton_iters = 1;
    perturbed.controls.use_gold2_tolerances = false;
    perturbed.controls.use_gold_tolerances = true;
    perturbed.lnd[2] += 0.15;
    perturbed.pack_xh();
    let rep = solve_henyey_step(&mut perturbed, false, true)
        .expect("1-iter solve should return report");
    assert!(
        !rep.converged,
        "unconverged 1-iter solve must not pass: {rep:?}"
    );
}
