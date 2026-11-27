// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone zero-`$MESA_DIR` invariant tests for Newton solver
//! line search & norms (`solver.rs`, `solver_norms.rs`), timestep limits &
//! checks (`timestep.rs`, `timestep_checks.rs`, `timestep_limits.rs`), and
//! Duffell (2016) Rayleigh-Taylor instability (`hydro_rti.rs`).

#![forbid(unsafe_code)]

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{MixingType, FOUR_PI, LN10, L_SUN, M_SUN};
use flexmesa_star::hydro_eqns::HenyeySystem;
use flexmesa_star::report_utils::{
    csound_face_at, eval_photosphere_info, eval_shock_info,
};
use flexmesa_star::solver::compute_corr_coeff;
use flexmesa_star::solver_norms::compute_model_correction_norms;
use flexmesa_star::{
    do1_dalpha_rti_dt_eqn, eval_alpha_rti_sources, eval_h211_factor,
    eval_varcontrol, load_zams_model, set_dpdr_drhodr_info, set_hydro_vars,
    set_rti_mixing_info, solve_henyey_step, timestep_controller, StarControls,
    StarJobControls, StarModel, TimestepAction,
};

fn usize_to_f64(n: usize) -> f64 {
    f64::from(u32::try_from(n).expect("test zone index fits in u32"))
}

fn test_chem_ids() -> Vec<usize> {
    let cat = NuclideCatalog::builtin_ref();
    vec![
        cat.get_nuclide_index("h1").expect("h1"),
        cat.get_nuclide_index("he4").expect("he4"),
    ]
}

fn setup_zams_for_timestep_test() -> StarModel {
    let mut model = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("1.0 Msun ZAMS should load");
    set_hydro_vars(&mut model, false, false)
        .expect("set_hydro_vars on 1.0 Msun ZAMS");
    model.save_start_values().expect("save_start_values");
    model.dt = 1.0e8;
    model.dt_next = 1.0e8;
    model.model_number = 20;
    model.doing_first_model_of_run = false;
    model.mstar_old = model.mstar;
    model.xmstar_old = model.xmstar;
    model.lnd_old.clone_from(&model.lnd);
    model.ln_t_old.clone_from(&model.ln_t);
    model.ln_r_old.clone_from(&model.ln_r);
    model.lum_old.clone_from(&model.lum);
    model.xa_old.clone_from(&model.xa);
    model.aux.step_hist.teff_old = model.teff;
    model.aux.step_hist.l_surf_old = model.surface_l_lsun();
    model.aux.step_hist.last_solver_iterations = 1;
    model.aux.step_hist.timestep_hold = 0;

    // Disable unrelated defaults so each test isolates one MESA limiter path.
    let ts = &mut model.controls.ts_stop;
    ts.never_skip_hard_limits = true;
    ts.dx_nuc_drop_limit = -1.0;
    ts.dx_nuc_drop_hard_limit = -1.0;
    ts.delta_lgl_he_limit = -1.0;
    ts.delta_lgteff_limit = -1.0;
    ts.delta_lgl_limit = -1.0;
    ts.delta_lgt_cntr_limit = -1.0;
    model.controls.varcontrol_target = 1.0;
    model
}

fn make_rti_test_model(nz: usize, v_flag: bool, u_flag: bool) -> StarModel {
    let controls = StarControls {
        rti_flag: true,
        v_flag,
        u_flag,
        do_mix: true,
        mix_factor: 1.0,
        sig_term_limit: 1.0e13,
        rti_a: 1.0e-3,
        rti_b: 2.5,
        rti_c: 0.2,
        rti_d: 3.0,
        rti_max_alpha: 0.5,
        alpha_rti_diffusion_factor: 1.0,
        alpha_rti_src_min_v_div_cs: 0.1,
        alpha_rti_src_min_q: 0.0,
        alpha_rti_src_max_q: 1.0,
        ..StarControls::default()
    };
    let mut model = StarModel::new(
        nz,
        test_chem_ids(),
        StarJobControls::default(),
        controls,
    );
    model.dt = 10.0;
    model.dt_next = 10.0;
    model.m_center = 0.0;
    model.r_center = 1.0e8;
    model.mstar = 2.0 * M_SUN;
    model.xmstar = model.mstar;
    for k in 0..nz {
        let q_k = usize_to_f64(nz - k) / usize_to_f64(nz);
        let dq_k = 1.0 / usize_to_f64(nz);
        model.q[k] = q_k;
        model.dq[k] = dq_k;
        model.dm[k] = model.xmstar * dq_k;
        model.dm_bar[k] = model.dm[k];
        model.m[k] = model.m_center + model.xmstar * q_k;
        let r_k = model.r_center + 1.0e9 * usize_to_f64(nz - k);
        let r_in = model.r_center + 1.0e9 * usize_to_f64(nz - k - 1);
        model.r[k] = r_k;
        model.rmid[k] = f64::midpoint(r_k, r_in);
        model.ln_r[k] = flexmesa_math::log(r_k);
        model.rho[k] = 1.0e-2;
        model.lnd[k] = flexmesa_math::log(model.rho[k]);
        model.peos[k] = 1.0e10 * usize_to_f64(k + 1);
        model.csound[k] = 1.0e6;
    }
    model.ln_r_start.clone_from(&model.ln_r);
    model.lnd_start.clone_from(&model.lnd);
    model.aux.csound_start.clone_from(&model.csound);
    model.aux.alpha_rti_start = vec![0.0; nz];
    model
}

#[test]
fn test_solver_norms_gate_chem_on_burn_or_mix_and_exclude_rsp2_w() {
    let nz = 2_usize;
    let controls = StarControls {
        v_flag: true,
        rsp2_flag: true,
        do_burn: false,
        do_mix: false,
        include_w_in_correction_limits: false,
        ..StarControls::default()
    };
    let mut model = StarModel::new(
        nz,
        test_chem_ids(),
        StarJobControls::default(),
        controls,
    );
    let nvar_hydro = model.layout.nvar_hydro;
    let nvar = model.layout.nvar_total;
    assert_eq!(nvar_hydro, 7);
    assert_eq!(nvar, 9);

    model.aux.rsp.et_rsp = vec![1.0e4; nz];
    model.xa = vec![0.5; nz * model.species];
    model.xa_start = model.xa.clone();

    let mut dx = vec![0.0_f64; nz * nvar];
    let x_scale = vec![1.0_f64; nz * nvar];
    let i_w = model.layout.i_w.expect("i_w");
    let i_hp = model.layout.i_hp.expect("i_hp");
    for k in 0..nz {
        let base = k * nvar;
        dx[base + model.layout.i_lnd] = 0.01;
        dx[base + model.layout.i_lnt] = 0.01;
        dx[base + model.layout.i_lnr] = 0.01;
        dx[base + model.layout.i_lum] = 0.0;
        dx[base + model.layout.i_v.expect("i_v")] = 0.0;
        // w_k = sqrt(1e4) = 100 -> weight = 1 / (1e3 + 100) = 1 / 1100.
        dx[base + i_w] = 5.5e4;
        dx[base + i_hp] = 999.0;
        dx[base + nvar_hydro] = 100.0;
        dx[base + nvar_hydro + 1] = 100.0;
    }

    // Both `chem` (`!do_burn && !do_mix`) and `w` (`!include_w_in_correction_limits`)
    // must be skipped (`solver_support.f90:368-379, 400, 466`).
    let (cn0, mc0) =
        compute_model_correction_norms(&model, nz, nvar, &dx, &x_scale);
    assert!((mc0 - 0.01).abs() <= 1.0e-14, "mc0={mc0}");
    assert!((cn0 - 0.06 / 10.0).abs() <= 1.0e-14, "cn0={cn0}");

    // Enabling `do_burn` includes species perturbations.
    model.controls.do_burn = true;
    let (cn_burn, mc_burn) =
        compute_model_correction_norms(&model, nz, nvar, &dx, &x_scale);
    assert!((mc_burn - 100.0).abs() <= 1.0e-12, "mc_burn={mc_burn}");
    assert!(cn_burn > 10.0, "cn_burn={cn_burn}");

    // Enabling `do_mix` alone also includes species perturbations.
    model.controls.do_burn = false;
    model.controls.do_mix = true;
    let (_, mc_mix) =
        compute_model_correction_norms(&model, nz, nvar, &dx, &x_scale);
    assert!((mc_mix - 100.0).abs() <= 1.0e-12, "mc_mix={mc_mix}");

    // Enabling `include_w_in_correction_limits` includes w = 5.5e4 / 1100 = 50.
    model.controls.do_mix = false;
    model.controls.include_w_in_correction_limits = true;
    let (cn_w, mc_w) =
        compute_model_correction_norms(&model, nz, nvar, &dx, &x_scale);
    assert!((mc_w - 50.0).abs() <= 1.0e-12, "mc_w={mc_w}");
    assert!(
        (cn_w - (0.06 + 100.0) / 12.0).abs() <= 1.0e-12,
        "cn_w={cn_w}"
    );
}

#[test]
fn test_solver_corr_coeff_and_henyey_step_converge_without_mesa_dir() {
    let mut model = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("1.0 Msun ZAMS should load");
    model.dt = 1.0e5;
    model.controls.scale_correction_norm = 0.5;
    model.controls.scale_max_correction = 2.0;
    model.controls.corr_coeff_limit = 0.25;

    let nvar = model.layout.nvar_hydro;
    let x_scale = vec![1.0_f64; model.nz * nvar];
    let mut dx = vec![0.0_f64; model.nz * nvar];
    dx[model.layout.i_lnt] = 4.0;
    // max_correction = 4.0 > scale_max_correction (2.0) -> coeff = 2/4 = 0.5.
    let mid_factor = compute_corr_coeff(&model, model.nz, nvar, &dx, &x_scale);
    assert!(
        (mid_factor - 0.5).abs() <= 1.0e-14,
        "mid_factor={mid_factor}"
    );

    dx[model.layout.i_lnt] = 100.0;
    // max_correction = 100.0 -> 2/100 = 0.02, clamped to corr_coeff_limit = 0.25.
    let floor_factor =
        compute_corr_coeff(&model, model.nz, nvar, &dx, &x_scale);
    assert!(
        (floor_factor - 0.25).abs() <= 1.0e-14,
        "floor_factor={floor_factor}"
    );

    model.controls = StarControls {
        initial_mass: 1.0,
        zbase: 0.02,
        ..StarControls::default()
    };
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model.save_start_values().expect("save_start_values");
    flexmesa_star::record_start_of_step_energy_integrals(&mut model);
    for k in 0..model.nz {
        model.ln_t[k] += 1.0e-4;
    }
    model.pack_xh();
    let rep = solve_henyey_step(&mut model, false, false)
        .expect("solve_henyey_step should execute");
    assert!(rep.converged, "ZAMS step failed to converge: {rep:?}");
    assert!(
        (1..=20).contains(&rep.iterations),
        "iterations={}",
        rep.iterations
    );
    assert!(rep.residual_norm.is_finite());
}

#[test]
fn test_timestep_mixing_boundary_subcell_dq_excludes_nearby_dlog_x_zone() {
    let mut model = setup_zams_for_timestep_test();
    let nz = model.nz;
    model.controls.do_mix = true;
    model.controls.mix_factor = 1.0;

    // Construct uniform q grid with an inner convective core (k >= k_bdy + 1)
    // and outer convective envelope (k <= 4) so mix_bdy_q has 3 entries.
    let k_bdy = nz / 2;
    let k_test = k_bdy - 1;
    let dq = 1.0 / usize_to_f64(nz);
    model.m_center = 0.0;
    model.mstar = M_SUN;
    model.xmstar = M_SUN;
    model.dq.fill(dq);
    for k in 0..nz {
        model.q[k] = usize_to_f64(nz - k) * dq;
        model.dm[k] = model.xmstar * dq;
        model.dm_bar[k] = model.dm[k];
        model.m[k] = model.xmstar * model.q[k];
    }

    for k in 0..nz {
        if k <= 4 || k > k_bdy {
            model.mixing_type[k] = MixingType::ConvectiveMixing;
            model.d_mix[k] = 1.0e10;
        } else {
            model.mixing_type[k] = MixingType::NoMixing;
            model.d_mix[k] = 0.0;
        }
    }
    model.aux.mixing.cz_bdy_dq = vec![0.0; nz];

    // Place dx_mix_dist_limit strictly between |q[k_test] - q[k_bdy]| and
    // |q[k_test] - (q[k_bdy] - cz_bdy_dq[k_bdy])|.
    let raw_dist = (model.q[k_test] - model.q[k_bdy]).abs();
    let shift = 2.0 * raw_dist;
    let ts = &mut model.controls.ts_stop;
    ts.dx_mix_dist_limit = raw_dist + 0.5 * shift;
    ts.dx_limit_species[0] = "h1".to_string();
    ts.dx_limit[0] = 0.01;
    ts.dx_hard_limit[0] = 0.02;
    ts.dx_limit_min_x[0] = 0.0;
    ts.dx_decreases_only[0] = false;

    let ih1 = model.net_iso[test_chem_ids()[0]].expect("h1 slot");
    let sp = model.species;
    model.xa_start = model.xa.clone();
    model.xa[k_test * sp + ih1] += 0.1;

    // Without cz_bdy_dq offset, k_test lies inside dx_mix_dist_limit -> skipped.
    let (act0, _) = timestep_controller(&mut model);
    assert!(
        matches!(act0, TimestepAction::KeepGoing),
        "expected KeepGoing inside mix boundary exclusion, got {act0:?}"
    );

    // Subtracting `cz_bdy_dq[k_bdy]` shifts the sub-cell boundary inward so
    // `k_test` now exceeds `dx_mix_dist_limit` and triggers a retry.
    model.aux.mixing.cz_bdy_dq[k_bdy] = shift;
    let (act1, _) = timestep_controller(&mut model);
    assert!(
        matches!(act1, TimestepAction::Retry(_)),
        "expected Retry once cz_bdy_dq shifts boundary, got {act1:?}"
    );
}

#[test]
fn test_timestep_before_near_zams_and_delta_hr_use_photosphere_luminosity() {
    let mut model = setup_zams_for_timestep_test();
    let nz = model.nz;

    // Force `eval_photosphere_info` onto its `tau_factor < 1` center fallback
    // (`star_utils.f90:1155-1206`) so `L_phot` (`l_center / L_SUN`) differs
    // from `surface_l_lsun()` (`lum[0] / L_SUN`).
    model.controls.tau_factor = 0.5;
    model.tau_base = 1.0e99;
    model.lum[0] = 1.0e-4 * L_SUN;
    model.l_center = 10.0 * L_SUN;
    let l_phot = eval_photosphere_info(&model).l_lsun;
    assert!((l_phot - 10.0).abs() <= 1.0e-12, "l_phot={l_phot}");
    assert!(model.surface_l_lsun() < 1.0e-3);

    // `check_delta_HR` must run even during `doing_relax = true` and use
    // `L_phot` (`timestep.f90:1990-2020`), ignoring identical `surface_l_lsun()`.
    model.aux.doing_relax = true;
    model.aux.step_hist.l_surf_old = model.surface_l_lsun();
    model.aux.step_hist.teff_old = model.teff;
    model.controls.ts_stop.delta_hr_limit = 0.05;
    model.controls.ts_stop.delta_hr_hard_limit = 0.1;
    let (act_hr, _) = timestep_controller(&mut model);
    assert!(
        matches!(act_hr, TimestepAction::Retry(_)),
        "expected delta_HR retry from L_phot even with doing_relax, got {act_hr:?}"
    );

    // `before_near_zams` compares `L_nuc / L_phot` against `lnuc_div_l_zams_limit`
    // (`timestep.f90:1324-1325, 1380-1381`). With huge `L_phot`, `L_nuc / L_phot`
    // stays below `lnuc_div_l_zams_limit` so `before_near_zams` skips
    // `delta_lgT_cntr`; dropping `L_phot` activates it.
    model.aux.doing_relax = false;
    model.controls.ts_stop.delta_hr_limit = -1.0;
    model.controls.ts_stop.delta_hr_hard_limit = -1.0;
    model
        .controls
        .ts_stop
        .delta_lgt_cntr_limit_only_after_near_zams = true;
    model.controls.ts_stop.delta_lgt_cntr_hard_limit = 0.05;
    model.controls.lnuc_div_l_zams_limit = 0.8;
    model.ln_t[nz - 1] = model.ln_t_start[nz - 1] + 0.5 * LN10;

    model.l_center = 1.0e6 * L_SUN;
    let (act_before, _) = timestep_controller(&mut model);
    assert!(
        matches!(act_before, TimestepAction::KeepGoing),
        "expected before_near_zams to skip delta_lgT_cntr, got {act_before:?}"
    );

    model.l_center = 1.0e-2 * L_SUN;
    let (act_after, _) = timestep_controller(&mut model);
    assert!(
        matches!(act_after, TimestepAction::Retry(_)),
        "expected small L_phot to enable delta_lgT_cntr retry, got {act_after:?}"
    );
}

#[test]
fn test_timestep_dlog_eps_nuc_and_burn_lgl_safe_log10_limits() {
    let mut model = setup_zams_for_timestep_test();
    let nz = model.nz;
    let k_test = nz / 2;
    model.mixing_type.fill(MixingType::NoMixing);
    model.eps_nuc_start = vec![0.0; nz];
    model.eps_nuc = vec![0.0; nz];
    model.eps_nuc_start[k_test] = 10.0;
    model.eps_nuc[k_test] = 40.0;

    // Only hard limit active (`delta_log_eps_nuc_limit = -1.0`), `log10(4) > 0.25`.
    model.controls.ts_stop.delta_log_eps_nuc_limit = -1.0;
    model.controls.ts_stop.delta_log_eps_nuc_hard_limit = 0.25;
    let (act_retry, _) = timestep_controller(&mut model);
    assert!(
        matches!(act_retry, TimestepAction::Retry(_)),
        "expected delta_log_eps_nuc hard limit retry, got {act_retry:?}"
    );

    // Convective cell must be skipped cleanly.
    model.mixing_type[k_test] = MixingType::ConvectiveMixing;
    let (act_conv, _) = timestep_controller(&mut model);
    assert!(
        matches!(act_conv, TimestepAction::KeepGoing),
        "expected convective cell to skip delta_log_eps_nuc, got {act_conv:?}"
    );

    // Tiny positive power & Teff must stay finite via `safe_log10`, and H211/varcontrol
    // smoke cleanly without `$MESA_DIR`.
    model.controls.ts_stop.delta_log_eps_nuc_hard_limit = -1.0;
    model.controls.ts_stop.delta_lgl_h_limit = 0.1;
    model.controls.ts_stop.lgl_h_burn_min = -5.0;
    model.controls.ts_stop.delta_lgteff_limit = 0.1;
    model
        .aux
        .eps_nuc_categories
        .fill([0.0; flexmesa_chem::NUM_CATEGORIES]);
    model.aux.step_hist.power_h_burn_old = 1.0e-100;
    let (act_safe, rep_safe) = timestep_controller(&mut model);
    assert!(matches!(act_safe, TimestepAction::KeepGoing));
    assert!(rep_safe.dt_next.is_finite() && rep_safe.dt_next > 0.0);
    assert_eq!(eval_varcontrol(&model).expect("varcontrol"), 0.0);
    let (h211_fac, _, _) = eval_h211_factor(1.0, 1.0, 1.0e5, 1.0e5, 2.0, 0.5);
    assert!((h211_fac - 1.0).abs() <= 1.0e-12);
}

#[test]
fn test_rti_dpdr_drhodr_defaults_to_unsmoothed_and_guards_nonpositive_cs() {
    let mut model = make_rti_test_model(6, false, true);
    model.peos = vec![1.0e10, 2.0e10, 3.0e10, 4.0e10, 5.0e10, 6.0e10];
    model.rho = vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0];
    model.lnd = model.rho.iter().map(|&r| flexmesa_math::log(r)).collect();
    model.lnd_start.clone_from(&model.lnd);

    // Zone 1: `csound <= 0` guard when `alpha_rti_start == 0`.
    model.csound[1] = 0.0;
    model.u[1] = 1.0e5;
    // Zone 3: supersonic/above-min-Mach baroclinic unstable cell.
    model.u[3] = 0.5 * model.csound[3];
    // Zone 4: below `alpha_rti_src_min_v_div_cs` with `alpha_rti_start == 0`.
    model.u[4] = 0.0;

    set_dpdr_drhodr_info(&mut model);

    // Exact unsmoothed indicator in zone 3:
    // `p_face[3]=3.5e10`, `p_face[4]=4.5e10`, `rho_face[3]=2.0`, `rho_face[4]=1.5`.
    let rmid_3 = f64::midpoint(model.r[3], model.r[4]);
    let dr_3 = model.dm[3] / (FOUR_PI * rmid_3 * rmid_3 * model.rho[3]);
    let expected_3 = (-1.0e10 / dr_3) * (0.5 / dr_3);
    let got_3 = model.aux.dpdr_drhodr_info[3];
    assert!(got_3 < 0.0, "got_3={got_3}");
    assert!(
        (got_3 - expected_3).abs() <= 1.0e-13 * expected_3.abs(),
        "got_3={got_3} expected_3={expected_3}"
    );
    // Unsmoothed default (`RTI_smooth_fraction = 1.0`) leaves neighbors 0.0.
    assert_eq!(model.aux.dpdr_drhodr_info[0], 0.0);
    assert_eq!(model.aux.dpdr_drhodr_info[1], 0.0);
    assert_eq!(model.aux.dpdr_drhodr_info[2], 0.0);
    assert_eq!(model.aux.dpdr_drhodr_info[4], 0.0);
    assert_eq!(model.aux.dpdr_drhodr_info[5], 0.0);
}

#[test]
fn test_rti_mixing_info_mass_boost_center_floor_shock_and_v_flag() {
    // 1. Verify mass boost across >5 Msun, 4.5 Msun, <=4 Msun and
    // current `alpha_rti` + `csound_face_at` (ignoring stale start arrays).
    let mut model = make_rti_test_model(6, true, false);
    model.m_center = 0.0;
    model.mstar = 6.0 * M_SUN;
    model.xmstar = 6.0 * M_SUN;
    let m_grid = [6.0, 5.5, 4.5, 4.0, 2.0, 1.0];
    for k in 0..6 {
        model.m[k] = m_grid[k] * M_SUN;
        model.q[k] = m_grid[k] / 6.0;
        let m_in = if k + 1 < 6 { m_grid[k + 1] } else { 0.0 };
        model.dq[k] = (m_grid[k] - m_in) / 6.0;
        model.dm[k] = model.xmstar * model.dq[k];
    }
    for k in 1..6 {
        model.dm_bar[k] = f64::midpoint(model.dm[k - 1], model.dm[k]);
    }
    model.aux.alpha_rti = vec![0.4; 6];
    model.aux.alpha_rti_start = vec![0.0; 6];
    model.csound = vec![1.0e6; 6];
    model.aux.csound_start = vec![9.0e8; 6];

    set_rti_mixing_info(&mut model);
    let eta1 = 0.2 * 0.4 * csound_face_at(&model, 1) * model.r[1];
    let eta2 = 0.2
        * flexmesa_math::exp10(1.5)
        * 0.4
        * csound_face_at(&model, 2)
        * model.r[2];
    let eta3 = 0.2 * 1.0e3 * 0.4 * csound_face_at(&model, 3) * model.r[3];
    assert!((model.aux.eta_rti[1] - eta1).abs() <= 1.0e-12 * eta1);
    assert!((model.aux.eta_rti[2] - eta2).abs() <= 1.0e-12 * eta2);
    assert!((model.aux.eta_rti[3] - eta3).abs() <= 1.0e-12 * eta3);

    // 2. Verify `RTI_dm_for_center_eta_nondecreasing = 0.02 M_sun` floor.
    let mut core_m = make_rti_test_model(6, false, false);
    core_m.m_center = 1.0 * M_SUN;
    core_m.mstar = 1.05 * M_SUN;
    core_m.xmstar = 0.05 * M_SUN;
    let core_masses = [1.05, 1.015, 1.010, 1.005, 1.002, 1.001];
    for k in 0..6 {
        core_m.m[k] = core_masses[k] * M_SUN;
        core_m.q[k] = (core_masses[k] - 1.0) / 0.05;
        let m_in = if k + 1 < 6 { core_masses[k + 1] } else { 1.0 };
        core_m.dq[k] = (core_masses[k] - m_in) / 0.05;
        core_m.dm[k] = core_m.xmstar * core_m.dq[k];
    }
    for k in 1..6 {
        core_m.dm_bar[k] = f64::midpoint(core_m.dm[k - 1], core_m.dm[k]);
    }
    core_m.aux.alpha_rti = vec![0.2, 0.2, 0.0, 0.0, 0.0, 0.0];
    set_rti_mixing_info(&mut core_m);
    assert!(core_m.aux.eta_rti[1] > 0.0);
    assert_eq!(core_m.aux.eta_rti[3], core_m.aux.eta_rti[1]);
    assert_eq!(core_m.aux.eta_rti[4], core_m.aux.eta_rti[1]);
    assert!(core_m.aux.etamid_rti[4] >= core_m.aux.eta_rti[1]);
    // With `v_flag = false`, `sigmid_rti` must remain zero.
    assert!(core_m.aux.sigmid_rti.iter().all(|&s| s == 0.0));

    // 3. Verify `shock_q` gating and `v_flag = true` `sigmid_rti` + `sig_term_limit`.
    let mut shock_m = make_rti_test_model(6, true, false);
    shock_m.aux.alpha_rti = vec![0.4; 6];
    shock_m.v = vec![0.5e6, 0.5e6, 2.0e6, 2.0e6, 2.0e6, 2.0e6];
    shock_m.rho = vec![10.0; 6];
    shock_m.lnd = shock_m.rho.iter().map(|&r| flexmesa_math::log(r)).collect();
    shock_m.controls.sig_term_limit = 1.0e-2;
    let s_info = eval_shock_info(&shock_m);
    assert!(s_info.q > shock_m.q[2] && shock_m.q[1] > s_info.q);
    set_rti_mixing_info(&mut shock_m);
    assert_eq!(shock_m.aux.sig_rti[1], 0.0, "ahead of shock must gate sig");
    assert!(shock_m.aux.sig_rti[2] > 0.0);
    let sig_lim_2 =
        shock_m.controls.sig_term_limit * shock_m.dm_bar[2] / shock_m.dt;
    assert!(shock_m.aux.sig_rti[2] <= sig_lim_2 + 1.0e-12 * sig_lim_2);
    assert_eq!(shock_m.aux.sigmid_rti[0], 0.0);
    assert!(shock_m.aux.sigmid_rti[2] > 0.0);
    let mid_lim_2 =
        shock_m.controls.sig_term_limit * shock_m.dm[2] / shock_m.dt;
    assert!(shock_m.aux.sigmid_rti[2] <= mid_lim_2 + 1.0e-12 * mid_lim_2);
}

#[test]
fn test_rti_eqn_active_without_do_mix_uses_i_equ_scale_and_matches_fd() {
    let mut model = make_rti_test_model(6, true, false);
    model.controls.do_mix = false;
    model.peos = vec![1.0e10, 2.0e10, 3.0e10, 4.0e10, 5.0e10, 6.0e10];
    model.rho = vec![1.0, 2.0, 2.5, 1.5, 1.0, 1.0];
    model.lnd = model.rho.iter().map(|&r| flexmesa_math::log(r)).collect();
    model.lnd_start.clone_from(&model.lnd);
    // Keep `v < csound` so shock_q stays inactive and test both supersaturation
    // (`alpha_rti > rti_max_alpha`) and subsaturation (`< rti_max_alpha`).
    model.v = vec![0.5e6; 6];
    model.aux.alpha_rti_start = vec![0.1, 0.2, 0.2, 0.2, 0.2, 0.1];
    model.aux.alpha_rti = vec![0.2, 0.3, 0.8, 0.25, 0.4, 0.1];

    set_dpdr_drhodr_info(&mut model);
    set_rti_mixing_info(&mut model);
    assert!(model.aux.dpdr_drhodr_info[2] < 0.0);
    assert!(model.aux.sig_rti[2] > 0.0);
    assert!(model.aux.sig_rti[3] > 0.0);

    let nvar = model.layout.nvar_hydro;
    let i_var = model.layout.i_alpha_rti.expect("i_alpha_rti");
    let i_equ = model.layout.i_dalpha_rti_dt.expect("i_dalpha_rti_dt");
    assert_eq!(i_var, i_equ);

    let mut x_scale = vec![1.0_f64; model.nz * nvar];
    for k in 0..model.nz {
        x_scale[k * nvar + i_equ] = 200.0;
    }

    let k = 2_usize;
    let mut sys = HenyeySystem::new(model.nz, nvar, x_scale.clone());
    do1_dalpha_rti_dt_eqn(&model, k, &mut sys);

    // Check `do_mix = false` still couples `(k-1, k, k+1)` and matches central FD.
    let blk_sz = nvar * nvar;
    let row = k * blk_sz + i_equ * nvar;
    assert_ne!(sys.lblk[row + i_var], 0.0);
    assert_ne!(sys.dblk[row + i_var], 0.0);
    assert_ne!(sys.ublk[row + i_var], 0.0);

    let eps = 1.0e-7;
    for (kv, blk_val) in [
        (k - 1, sys.lblk[row + i_var]),
        (k, sys.dblk[row + i_var]),
        (k + 1, sys.ublk[row + i_var]),
    ] {
        let mut m_p = model.clone();
        m_p.aux.alpha_rti[kv] += eps;
        let mut sys_p = HenyeySystem::new(m_p.nz, nvar, x_scale.clone());
        do1_dalpha_rti_dt_eqn(&m_p, k, &mut sys_p);

        let mut m_m = model.clone();
        m_m.aux.alpha_rti[kv] -= eps;
        let mut sys_m = HenyeySystem::new(m_m.nz, nvar, x_scale.clone());
        do1_dalpha_rti_dt_eqn(&m_m, k, &mut sys_m);

        let fd = (sys_p.equ[k * nvar + i_equ] - sys_m.equ[k * nvar + i_equ])
            / (2.0 * eps);
        let analytical = blk_val / x_scale[kv * nvar + i_var];
        assert!(
            (fd - analytical).abs() <= 1.0e-8 * fd.abs().max(1.0),
            "kv={kv}: fd={fd} analytical={analytical}"
        );
    }

    // Verify `eqn_scale` uses `x_scale[i_equ]` rather than `x_scale[i_var]`
    // (`hydro_alpha_rti_eqns.f90:157`) when the two slots are distinct.
    let mut split_model = model.clone();
    let alt_equ = i_var - 1;
    split_model.layout.i_dalpha_rti_dt = Some(alt_equ);
    let mut split_scale = vec![1.0_f64; split_model.nz * nvar];
    for zone in 0..split_model.nz {
        split_scale[zone * nvar + i_var] = 3.0;
        split_scale[zone * nvar + alt_equ] = 200.0;
    }
    let mut sys_split =
        HenyeySystem::new(split_model.nz, nvar, split_scale.clone());
    do1_dalpha_rti_dt_eqn(&split_model, k, &mut sys_split);
    assert!(
        (sys_split.equ[k * nvar + alt_equ] - sys.equ[k * nvar + i_equ]).abs()
            <= 1.0e-14
    );

    let (sp, _, sm, _) = eval_alpha_rti_sources(&model, k);
    assert!(sp > 0.0 && sm > 0.0);
}
