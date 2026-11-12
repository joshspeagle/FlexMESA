// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration tests for `RSP2` time-dependent turbulent convection kernels,
//! quintic $C^2$ enthalpy flux limiter, coupled profile residuals, pre-solver
//! stationary adjustment, and two-zone anchor-temperature remeshing
//! (`rsp2.rs`, `rsp2_eqns.rs`, and `rsp2_remesh.rs`, resolving Issue #5).

use std::path::Path;

use flexmesa_const::{BOLTZ_SIGMA, PI, PI4};
use flexmesa_star::zams::load_zams_model;
use flexmesa_star::{
    compute_rsp2_uq_face, eval_henyey_equations, eval_rsp2_face,
    eval_rsp2_hp_face_zone, eval_rsp2_profile_equations, eval_rsp2_zone,
    read_mod_str, remesh_for_rsp2, rsp2_adjust_vars_before_call_solver,
    rsp2_flux_limiter, rsp_total_energy_integrals, set_hydro_vars,
    solve_stationary_rsp2_w, star_evolve_step, Rsp2Config, StarControls,
    StarJobControls, StarModel, VarLayout, X_ALFAS, X_GAMMAR,
};

#[test]
fn test_rsp2_flux_limiter_is_c2_continuous_across_transition_endpoints() {
    let (fl_lo, dfl_lo) = rsp2_flux_limiter(0.95 - 1.0e-12);
    let (fl_at0, dfl_at0) = rsp2_flux_limiter(0.95);
    assert!((fl_at0 - 0.95).abs() <= 1.0e-15);
    assert!((dfl_at0 - 1.0).abs() <= 1.0e-15);
    assert!((fl_lo - fl_at0).abs() <= 2.0e-12);
    assert!((dfl_lo - dfl_at0).abs() <= 2.0e-11);

    let (fl_at1, dfl_at1) = rsp2_flux_limiter(1.0);
    let (fl_hi, dfl_hi) = rsp2_flux_limiter(1.05);
    let (fl_just_below1, dfl_just_below1) = rsp2_flux_limiter(1.0 - 1.0e-7);
    assert!((fl_at1 - 1.0).abs() <= 1.0e-15);
    assert!(dfl_at1.abs() <= 1.0e-15);
    assert!((fl_hi - 1.0).abs() <= 1.0e-15 && dfl_hi == 0.0);
    assert!((fl_just_below1 - 1.0).abs() <= 1.0e-12);
    assert!(dfl_just_below1.abs() <= 1.0e-10);

    // Numerical second derivative FL''(X) = d(FL')/dX vanishes at X = 0.95^+ and X = 1.0^-
    // (with exact Taylor remainder |0.5 * FL''' * h| <= 1.2e4 * 1e-8 = 1.2e-4).
    let h = 1.0e-8;
    let (_, d_near_095) = rsp2_flux_limiter(0.95 + h);
    let (_, d_near_100) = rsp2_flux_limiter(1.0 - h);
    let d2fl_095 = (d_near_095 - dfl_at0) / h;
    let d2fl_100 = (dfl_at1 - d_near_100) / h;
    assert!(d2fl_095.abs() <= 2.0e-4);
    assert!(d2fl_100.abs() <= 2.0e-4);
}

#[test]
fn test_rsp2_zone_chi_dr_eq_and_uq_match_upstream_hydro_rsp2_equations() {
    let cfg = Rsp2Config::default();
    let w = 2.5e5_f64;
    let t = 4.2e4_f64;
    let rho = 2.0e-6_f64;
    let p = 1.1e7_f64;
    let cp = 2.8e8_f64;
    let grad_ad = 0.38_f64;
    let delta = 1.35_f64;
    let kappa = 1.8_f64;
    let hp_cell = 8.0e9_f64;
    let grad_t = 0.62_f64;
    let r_out = 3.02e12_f64;
    let r_in = 3.00e12_f64;
    let u_out = -1.2e6_f64;
    let u_in = -1.0e6_f64;
    let dm = 4.0e28_f64;

    let zone_out = eval_rsp2_zone(
        w, t, rho, p, cp, grad_ad, delta, kappa, hp_cell, grad_t, r_out, r_in,
        u_out, u_in, dm, 0.0, &cfg,
    );

    // Exact oracle for compute_Chi_cell (`hydro_rsp2.f90:717-757`)
    let du_over_r = (u_out / r_out) - (u_in / r_in);
    let r6 = f64::midpoint(r_out.powi(6), r_in.powi(6));
    let chi_oracle = (16.0 / 3.0)
        * PI
        * (cfg.alfam * cfg.alfa / dm)
        * rho
        * rho
        * r6
        * du_over_r
        * hp_cell
        * w;
    assert!(
        (zone_out.chi_turb - chi_oracle).abs() / chi_oracle.abs() <= 1.0e-15
    );

    // Exact oracle for compute_Eq_cell (`hydro_rsp2.f90:820-839`)
    let eq_oracle = PI4 * chi_oracle * du_over_r / dm;
    assert!((zone_out.eq_visc - eq_oracle).abs() / eq_oracle.abs() <= 1.0e-15);

    // Exact oracle for compute_Dr (`hydro_rsp2.f90:907-934`, single Cp, no delta)
    let dr_oracle = w
        * w
        * 4.0
        * BOLTZ_SIGMA
        * (X_GAMMAR * cfg.gammar / cfg.alfa).powi(2)
        * t.powi(3)
        / (rho * rho * cp * kappa * hp_cell * hp_cell);
    assert!((zone_out.sink_dr - dr_oracle).abs() / dr_oracle <= 1.0e-15);

    // Exact oracle for compute_PII_face (`hydro_rsp2.f90:546-608`)
    let face_out = eval_rsp2_face(
        3.01e12,
        1.0e34,
        6.67430e-8,
        p,
        rho,
        t,
        cp,
        grad_ad,
        kappa,
        w,
        0.9 * w,
        (1.02 * t).powi(4),
        (0.98 * t).powi(4),
        16.3,
        16.1,
        zone_out.chi_turb,
        du_over_r,
        dm,
        &cfg,
    );
    let pi_oracle = X_ALFAS * cfg.alfa * cp * face_out.y_face;
    assert!((face_out.pi_face - pi_oracle).abs() / pi_oracle.abs() <= 1.0e-15);

    // Exact oracle for compute_Uq_face (`hydro_rsp2.f90:785-817`)
    let uq = compute_rsp2_uq_face(1.2 * chi_oracle, chi_oracle, 3.01e12, dm);
    let uq_oracle = PI4 * (0.2 * chi_oracle) / (3.01e12 * dm);
    assert!((uq - uq_oracle).abs() / uq_oracle.abs() <= 1.0e-15);

    // Stationary root satisfies S(w) - D(w) - D_r(w) == 0 to < 1e-14
    let w_eq = solve_stationary_rsp2_w(
        t, rho, p, cp, grad_ad, delta, kappa, hp_cell, grad_t, &cfg,
    );
    let z_eq = eval_rsp2_zone(
        w_eq, t, rho, p, cp, grad_ad, delta, kappa, hp_cell, grad_t, r_out,
        r_in, 0.0, 0.0, dm, 0.0, &cfg,
    );
    assert!(z_eq.de_t_dt.abs() / z_eq.source_s.abs().max(1.0) <= 1.0e-14);
}

#[test]
fn test_remesh_for_rsp2_and_profile_equations_preserve_mass_and_monotonic_radius(
) {
    let mut controls = StarControls {
        rsp2_flag: true,
        ..Default::default()
    };
    controls.rsp2.nz = 120;
    controls.rsp2.nz_outer = 30;
    controls.rsp2.t_anchor = 2.5e4;
    controls.rsp2.dq_1_factor = 2.0;
    controls.rsp2.use_tdc_enthalpy_flux_limiter = true;

    let mut model =
        load_zams_model(controls, StarJobControls::default()).expect("zams");
    let mstar_before = model.mstar;
    let xmstar_before = model.xmstar;
    assert!(model.nz > 120);

    remesh_for_rsp2(&mut model).expect("remesh_for_rsp2");
    assert_eq!(model.nz, 120);
    assert!((model.mstar - mstar_before).abs() / mstar_before <= 1.0e-14);
    assert!((model.xmstar - xmstar_before).abs() / xmstar_before <= 1.0e-14);

    // Verify uniform outer zone mass spacing (`dm[1..30]` equal, `dm[0] == 2 * dm[1]`)
    assert!((model.dm[0] - 2.0 * model.dm[1]).abs() / model.dm[1] <= 1.0e-12);
    for k in 1..30 {
        assert!((model.dm[k] - model.dm[1]).abs() / model.dm[1] <= 1.0e-12);
    }

    // Verify strictly decreasing radius from surface to center and normalized species
    for k in 0..model.nz {
        let r_next = if k + 1 < model.nz {
            model.r[k + 1]
        } else {
            model.r_center
        };
        assert!(model.r[k] > r_next);
        let sum_xa: f64 = model.cell_xa(k).iter().sum();
        assert!((sum_xa - 1.0).abs() <= 1.0e-14);
        assert!(eval_rsp2_hp_face_zone(&model, k) > 0.0);
    }

    model.dt = 100.0;
    set_hydro_vars(&mut model, false, false).expect("hydro_vars");
    rsp2_adjust_vars_before_call_solver(&mut model);
    let res = eval_rsp2_profile_equations(&mut model);
    assert_eq!(res.equ_l.len(), 120);
    assert_eq!(res.equ_hp.len(), 120);
    assert_eq!(res.equ_etrb.len(), 120);
    assert!(res.equ_l.iter().all(|x| x.is_finite()));
    assert!(res.equ_hp.iter().all(|x| x.abs() <= 1.0e-14));
    assert!(res.equ_etrb.iter().all(|x| x.is_finite()));
    assert!(model.aux.rsp.et_rsp.iter().any(|&et| et > 0.0));
}

fn build_rsp2_jacobian_test_model() -> StarModel {
    let mut controls = StarControls {
        v_flag: true,
        rsp2_flag: true,
        mixing_length_alpha: 1.5,
        use_zero_pgas_outer_bc: true,
        using_velocity_time_centering: true,
        include_p_in_velocity_time_centering: true,
        include_l_in_velocity_time_centering: true,
        p_theta_for_velocity_time_centering: 0.6,
        l_theta_for_velocity_time_centering: 0.7,
        max_logt_for_include_p_and_l_in_velocity_time_centering: 99.0,
        ..Default::default()
    };
    controls.rsp2.nz = 60;
    controls.rsp2.nz_outer = 20;
    controls.rsp2.t_anchor = 2.5e4;
    controls.rsp2.alfap = 1.0;
    controls.rsp2.alfam = 0.25;
    controls.rsp2.alfat = 0.01;
    controls.rsp2.alfad = 1.0;
    controls.rsp2.alfar = 1.0;
    controls.rsp2.use_tdc_enthalpy_flux_limiter = true;
    controls.rsp2.num_outermost_cells_forced_nonturbulent = 2;
    controls.rsp2.nz_div_ibotom = 6.0;

    let mut model =
        load_zams_model(controls, StarJobControls::default()).expect("zams");
    remesh_for_rsp2(&mut model).expect("remesh");
    assert_eq!(model.layout.nvar_hydro, 7);
    assert!(model.aux.rsp.et_rsp.iter().any(|&et| et > 0.0));

    let nz = model.nz;
    let nz_f = f64::from(u32::try_from(nz).expect("nz fits u32"));
    for k in 0..nz {
        let phase = f64::from(u32::try_from(k).expect("k fits u32")) / nz_f;
        model.v[k] = 4.0e5 * (1.0 - phase) * (2.5 * phase).cos();
        let w_k = 1.8e5 * (PI * phase).sin().max(0.05);
        model.aux.rsp.et_rsp[k] = w_k * w_k;
    }
    model.ln_t[15] = model.ln_t[14] + 0.35;
    model.dt = 25.0;
    set_hydro_vars(&mut model, false, false).expect("hydro_vars");
    assert!(model.d_mix.iter().any(|&d| d > 0.0));
    for k in 0..nz {
        model.aux.rsp.hp_face[k] =
            1.03 * eval_rsp2_hp_face_zone(&model, k).max(1.0);
    }
    model.save_start_values();
    for k in 0..nz {
        model.v[k] *= 1.04;
        model.ln_r[k] += 1.0e-4;
    }
    model.pack_xh();
    model
}

#[test]
fn test_rsp2_24slot_analytical_jacobian_matches_finite_differences() {
    let mut model = build_rsp2_jacobian_test_model();
    let nz = model.nz;
    let base_sys = eval_henyey_equations(&mut model, false).expect("base_sys");
    let nvar = base_sys.nvar;
    assert_eq!(nvar, 7);
    let xh_0 = model.xh.clone();

    let i_etrb = model.layout.i_turbulent_energy.expect("i_turbulent_energy");
    let i_equ_hp = model.layout.i_equ_hp.expect("i_equ_hp");
    let i_equ_l = model.layout.i_equ_l;
    let i_dv_dt = model.layout.i_dv_dt;
    let i_dlne = model.layout.i_dlne_dt;

    for &k_row in &[0_usize, 15, nz - 1] {
        for dk in -1_i32..=1 {
            if (k_row == 0 && dk == -1) || (k_row + 1 == nz && dk == 1) {
                continue;
            }
            let col_zone = match dk {
                -1 => k_row - 1,
                0 => k_row,
                1 => k_row + 1,
                _ => unreachable!(),
            };
            for j_var in 0..nvar {
                let flat_idx = col_zone * nvar + j_var;
                let x_sc = base_sys.x_scale[flat_idx];
                let eps_scaled = 1.0e-6;
                let dx = eps_scaled * x_sc;

                model.xh.clone_from(&xh_0);
                model.xh[flat_idx] += dx;
                model.unpack_xh();
                let sys_hi =
                    eval_henyey_equations(&mut model, false).expect("sys_hi");

                model.xh.clone_from(&xh_0);
                model.xh[flat_idx] -= dx;
                model.unpack_xh();
                let sys_lo =
                    eval_henyey_equations(&mut model, false).expect("sys_lo");

                for &i_equ in &[i_equ_l, i_equ_hp, i_etrb, i_dv_dt, i_dlne] {
                    let fd = (sys_hi.equ[k_row * nvar + i_equ]
                        - sys_lo.equ[k_row * nvar + i_equ])
                        / (2.0 * eps_scaled);
                    let blk_slice = match dk {
                        -1 => &base_sys.lblk,
                        0 => &base_sys.dblk,
                        1 => &base_sys.ublk,
                        _ => unreachable!(),
                    };
                    let ana = blk_slice[(k_row * nvar + i_equ) * nvar + j_var];
                    let scale = ana.abs().max(fd.abs()).max(1.0e-6);
                    assert!(
                        (ana - fd).abs() <= 2.0e-4 * scale,
                        "Jacobian mismatch at k={k_row}, dk={dk}, equ={i_equ}, var={j_var}: ana={ana:.8e}, fd={fd:.8e}"
                    );
                }
            }
        }
    }
}

#[test]
fn test_rsp2_coupled_henyey_evolution_on_blap_conserves_energy() {
    let mod_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/golden/rsp/rsp_BLAP_cycle0.mod");
    let text = std::fs::read_to_string(&mod_path).expect("rsp_BLAP_cycle0.mod");

    let mut controls = StarControls {
        v_flag: true,
        rsp2_flag: true,
        op_split_burn: true,
        do_burn: false,
        do_mix: false,
        dxdt_nuc_factor: 0.0,
        eps_nuc_factor: 0.0,
        non_nuc_neu_factor: 0.0,
        mix_factor: 0.0,
        mixing_length_alpha: 1.5,
        use_zero_pgas_outer_bc: true,
        use_pvsc_art_visc: true,
        pvsc_cq: 4.0,
        pvsc_zshock: 0.1,
        using_velocity_time_centering: true,
        include_p_in_velocity_time_centering: true,
        include_l_in_velocity_time_centering: true,
        use_p_d_1_div_rho_form_of_work_when_time_centering_velocity: true,
        ..Default::default()
    };
    controls.rsp2.change_rsp_flag = true;
    controls.rsp2.new_rsp_flag = false;
    controls.rsp2.change_rsp2_flag = true;
    controls.rsp2.new_rsp2_flag = true;
    controls.rsp2.alfap = 1.0;
    controls.rsp2.alfad = 1.0;
    controls.rsp2.alfam = 0.25;
    controls.rsp2.alfar = 1.0;
    controls.rsp2.alfat = 0.01;
    controls.rsp2.nz = 150;
    controls.rsp2.nz_outer = 40;
    controls.rsp2.t_anchor = 5.0e4;
    controls.rsp2.dq_1_factor = 1.0;

    let mut model =
        read_mod_str(&text, controls, StarJobControls::default()).expect("mod");
    model.layout = VarLayout::with_rsp2(model.species, true, false, true);
    remesh_for_rsp2(&mut model).expect("remesh BLAP");
    assert_eq!(model.nz, 150);

    model.dt = 2.0;
    model.dt_next = 2.0;
    model.controls.rsp.max_timestep = 5.0;
    set_hydro_vars(&mut model, false, false).expect("hydro_vars");
    let (e0, _) = rsp_total_energy_integrals(&mut model);
    model.aux.total_energy = e0;
    model.aux.rsp.total_energy_old = e0;
    model.aux.cumulative_energy_error = 0.0;
    model.save_start_values();
    model.save_old_values();

    for _ in 0..5 {
        let rep =
            star_evolve_step(&mut model, false, true, true).expect("rsp2 step");
        assert!(rep.converged);
    }
    let rel_e_err =
        (model.aux.cumulative_energy_error / model.aux.total_energy).abs();
    assert!(
        rel_e_err.is_finite() && rel_e_err <= 1.0e-4,
        "unexpected RSP2 cumulative relative energy error: {rel_e_err:.3e}"
    );
    assert!(model.v[0].is_finite() && model.v[0].abs() > 0.0);
    assert!(model.aux.rsp.et_rsp.iter().any(|&et| et > 0.0));
}
