// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration and unit tests for Issue #151: MESA `26.4.1` total-energy
//! accounting (`eval_total_energy_integrals`, `okay_energy_conservation`,
//! `check_rel_error_in_energy`, `eps_grav` energy audit, rotational/TDC/RSP/RSP2
//! integrals, and `history.f90` energy columns).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::manual_midpoint,
    clippy::similar_names,
    clippy::too_many_lines
)]

use flexmesa_const::{M_SUN, SECYER};
use flexmesa_eos::I_DE_DRHO;
use flexmesa_star::inlist::InlistBundle;
use flexmesa_star::inlist_runner::DefaultStarExtras;
use flexmesa_star::{
    cell_specific_ke, cell_specific_pe, cell_specific_rotational_energy,
    cell_specific_total_energy, do_adjust_mass_step, do_struct_burn_mix,
    eval_delta_m_total_energy_integrals, eval_total_energy_integrals,
    evaluate_star_history_row, init_or_update_model_from_bundle,
    load_photo_bytes, load_zams_model, record_start_of_step_energy_integrals,
    set_hydro_vars, star_evolve_step, update_step_energy_conservation,
    verify_photo_restart_parity, write_mod_string, write_photo_bytes,
    EnergyEqnOption, HistoryColumnSpec, MesaCellValue, PrimaryStateBuffer,
    StarControls, StarError, StarJobControls, StarModel, PHOTO_FORMAT_VERSION,
};
use flexmesa_turb::MltOption;

fn rel_err(actual: f64, expected: f64) -> f64 {
    if actual == expected {
        return 0.0;
    }
    let denom = expected.abs().max(1.0e-99);
    (actual - expected).abs() / denom
}

fn history_float(model: &StarModel, col: &'static str) -> f64 {
    let row =
        evaluate_star_history_row(model, &[HistoryColumnSpec::Scalar(col)])
            .expect("history row");
    match row.first().map(|(_, v)| v) {
        Some(MesaCellValue::Float(v)) => *v,
        other => {
            panic!("expected Float for history column {col}, got {other:?}")
        }
    }
}

fn solar_zams(energy_eqn_option: EnergyEqnOption) -> StarModel {
    let controls = StarControls {
        initial_mass: 1.0,
        initial_z: 0.02,
        zbase: 0.02,
        energy_eqn_option,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(controls, StarJobControls::default()).expect("ZAMS");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model
}

/// Verifies `eval_total_energy_integrals`, `eval_delta_m_total_energy_integrals`,
/// `cell_specific_rotational_energy`, and `cell_specific_total_energy` against
/// MESA `26.4.1` (`star/private/star_utils.f90:2328-2612`) across `Quad` PE/KE,
/// rotational face averaging and `include_rotation_in_total_energy` gating,
/// `TDC` turbulent energy, `RSP2` `w^2`, and `RSP` `RSP_Et`.
#[test]
fn test_eval_total_energy_integrals_quad_pe_ke_rotation_tdc_rsp_and_rsp2() {
    let nz = 4;
    let controls = StarControls {
        v_flag: true,
        u_flag: false,
        rotation_flag: true,
        include_rotation_in_total_energy: false,
        mlt_option: MltOption::Tdc,
        tdc_include_eturb_in_energy_equation: true,
        ..StarControls::default()
    };
    let mut m =
        StarModel::new(nz, vec![1], StarJobControls::default(), controls);
    m.m_center = 0.0;
    m.r_center = 0.0;
    m.v_center = 1.0e4;
    m.dm = vec![1.0e30, 2.0e30, 3.0e30, 4.0e30];
    m.mstar = m.dm.iter().sum();
    m.xmstar = m.mstar;
    m.cgrav = vec![6.6743e-8; nz];
    m.m_grav = vec![1.0e31, 9.0e30, 7.0e30, 4.0e30];
    // Include a sub-centimeter radius at the innermost zone and check its
    // analytic PE directly (`-0.5 * cgrav * m_grav / r`) so no spurious
    // `r.max(1.0)` clamp can hide behind a self-check.
    m.r = vec![4.0e10, 3.0e10, 2.0e10, 0.5];
    m.energy = vec![1.5e14, 2.5e14, 3.5e14, 4.5e14];
    m.v = vec![5.0e5, 4.0e5, 3.0e5, 2.0e5];
    m.omega = vec![1.0e-4, 2.0e-4, 3.0e-4, 4.0e-4];
    m.aux.i_rot = vec![1.0e20, 2.0e20, 3.0e20, 4.0e20];
    m.mlt_vc = vec![1.0e4, 2.0e4, 3.0e4, 4.0e4];

    // Check `cell_specific_rotational_energy` matches `star_utils.f90:2328-2339`:
    // `0.5 * (e_p1 + e_00)` with `e = i_rot * omega^2` (and `e_p1 = 0` at center).
    let expected_rot_0 = 0.5
        * (m.aux.i_rot[1] * m.omega[1] * m.omega[1]
            + m.aux.i_rot[0] * m.omega[0] * m.omega[0]);
    let expected_rot_last =
        0.5 * (0.0 + m.aux.i_rot[nz - 1] * m.omega[nz - 1] * m.omega[nz - 1]);
    assert_eq!(cell_specific_rotational_energy(&m, 0), expected_rot_0);
    assert_eq!(
        cell_specific_rotational_energy(&m, nz - 1),
        expected_rot_last
    );

    let analytic_pe_cell = |k: usize| -> f64 {
        let g0 = m.cgrav[k];
        let grav00 = -(g0 * m.m_grav[k] / m.r[k]);
        let gravp1 = if k + 1 == nz {
            0.0
        } else {
            -(m.cgrav[k + 1] * m.m_grav[k + 1] / m.r[k + 1])
        };
        0.5 * (gravp1 + grav00)
    };
    let analytic_ke_cell = |k: usize| -> f64 {
        let v0 = m.v[k];
        let v1 = if k + 1 < nz { m.v[k + 1] } else { m.v_center };
        0.25 * (v0 * v0 + v1 * v1)
    };
    let analytic_rot_cell = |k: usize| -> f64 {
        let e0 = m.aux.i_rot[k] * m.omega[k] * m.omega[k];
        let e1 = if k + 1 < nz {
            m.aux.i_rot[k + 1] * m.omega[k + 1] * m.omega[k + 1]
        } else {
            0.0
        };
        0.5 * (e1 + e0)
    };
    let inner_pe = -0.5 * 6.6743e-8 * 4.0e30 / 0.5;
    assert!(rel_err(cell_specific_pe(&m, nz - 1), inner_pe) < 1.0e-14);
    assert!(rel_err(cell_specific_ke(&m, 0), analytic_ke_cell(0)) < 1.0e-14);

    let expected_ie: f64 = (0..nz).map(|k| m.dm[k] * m.energy[k]).sum();
    let expected_pe: f64 = (0..nz).map(|k| m.dm[k] * analytic_pe_cell(k)).sum();
    let expected_ke: f64 = (0..nz).map(|k| m.dm[k] * analytic_ke_cell(k)).sum();
    let expected_rot: f64 =
        (0..nz).map(|k| m.dm[k] * analytic_rot_cell(k)).sum();
    let expected_tdc: f64 = (0..nz)
        .map(|k| {
            let e_k = if k + 1 < nz {
                0.75 * (m.mlt_vc[k] * m.mlt_vc[k]
                    + m.mlt_vc[k + 1] * m.mlt_vc[k + 1])
            } else {
                0.75 * m.mlt_vc[k] * m.mlt_vc[k]
            };
            m.dm[k] * e_k
        })
        .sum();

    // Default `include_rotation_in_total_energy = false`: `e_rot` is reported
    // in slot 3 but omitted from `e_tot` (`star_utils.f90:2484-2489, 2519-2520`).
    let (ie, pe, ke, rot, turb, tot_no_rot) = eval_total_energy_integrals(&m);
    assert_eq!(ie, expected_ie);
    assert!(rel_err(pe, expected_pe) < 1.0e-14);
    assert!(rel_err(ke, expected_ke) < 1.0e-14);
    assert_eq!(rot, expected_rot);
    assert_eq!(turb, expected_tdc);
    assert!(
        rel_err(
            tot_no_rot,
            expected_ie + expected_pe + expected_ke + expected_tdc
        ) < 1.0e-14
    );

    // Enable `include_rotation_in_total_energy = true`: `e_rot` enters `e_tot`
    // and `cell_specific_total_energy`.
    m.controls.include_rotation_in_total_energy = true;
    let (_, _, _, rot2, _, tot_with_rot) = eval_total_energy_integrals(&m);
    assert_eq!(rot2, expected_rot);
    assert!(rel_err(tot_with_rot, tot_no_rot + expected_rot) < 1.0e-14);
    let cell0_with_rot = cell_specific_total_energy(&m, 0);
    let expected_cell0 = m.energy[0]
        + analytic_ke_cell(0)
        + analytic_pe_cell(0)
        + analytic_rot_cell(0);
    assert!(rel_err(cell0_with_rot, expected_cell0) < 1.0e-14);

    // `RSP2` overrides `TDC` turbulent energy (`star_utils.f90:2490-2505`),
    // and `RSP` adds `RSP_Et` independently (`star_utils.f90:2506-2510`).
    m.controls.rsp2_flag = true;
    m.controls.rsp_flag = false;
    m.aux.etrb = vec![1.0e11, 2.0e11, 3.0e11, 4.0e11];
    let (prof_rsp2, (_, _, _, _, turb_rsp2, _)) =
        eval_delta_m_total_energy_integrals(&m, 0, nz, m.mstar, true);
    let expected_rsp2: f64 = (0..nz).map(|k| m.dm[k] * m.aux.etrb[k]).sum();
    assert_eq!(turb_rsp2, expected_rsp2);
    let expected_prof0 = m.dm[0] * m.energy[0]
        + m.dm[0] * analytic_ke_cell(0)
        + m.dm[0] * analytic_pe_cell(0)
        + m.dm[0] * analytic_rot_cell(0)
        + m.dm[0] * m.aux.etrb[0];
    assert!(rel_err(prof_rsp2[0], expected_prof0) < 1.0e-14);

    m.controls.rsp2_flag = false;
    m.controls.tdc_include_eturb_in_energy_equation = false;
    m.controls.rsp_flag = true;
    m.aux.rsp.et_rsp = vec![5.0e10, 6.0e10, 7.0e10, 8.0e10];
    let (_, _, _, _, turb_rsp, _) = eval_total_energy_integrals(&m);
    let expected_rsp: f64 =
        (0..nz).map(|k| m.dm[k] * m.aux.rsp.et_rsp[k]).sum();
    assert_eq!(turb_rsp, expected_rsp);
    let cell0_rsp = cell_specific_total_energy(&m, 0);
    let expected_cell0_rsp = m.energy[0]
        + analytic_ke_cell(0)
        + analytic_pe_cell(0)
        + analytic_rot_cell(0)
        + m.aux.rsp.et_rsp[0];
    assert!(rel_err(cell0_rsp, expected_cell0_rsp) < 1.0e-14);
}

/// Verifies that an unaccounted non-solver energy perturbation between Phase 1
/// start and Phase 2 solve is measured accurately by `error_in_energy_conservation`,
/// `rel_E_err`, and `log_rel_run_E_err` rather than hidden by summing solver
/// residuals.
#[test]
fn test_non_solver_energy_perturbation_detected_accurately() {
    let mut base = solar_zams(EnergyEqnOption::Dedt);
    base.dt = 1.0e5 * SECYER;
    base.dt_next = base.dt;
    let dt = base.dt;

    // 1. Clean baseline step: measure the genuine total-energy balance residual.
    let mut clean = base.clone();
    clean.save_old_values();
    record_start_of_step_energy_integrals(&mut clean);
    let rep_clean = do_struct_burn_mix(&mut clean, dt, false, true, &[])
        .expect("clean sbm");
    assert!(rep_clean.converged);
    update_step_energy_conservation(&mut clean, dt);
    let err_clean = clean.aux.step_hist.error_in_energy_conservation;
    let e_end_clean = clean.aux.step_hist.total_energy_end;
    assert!(
        (err_clean / e_end_clean).abs() < 1.0e-5,
        "clean dedt step rel_E_err unexpectedly large: {:.3e}",
        err_clean / e_end_clean
    );

    // 2. Inject a known unaccounted Phase 1 internal-energy perturbation before
    //    `do_struct_burn_mix` (simulating a composition clamp or operator-split
    //    drift that modifies `energy` without registering a Phase 1 source/sink).
    //    Because `do_struct_burn_mix` snapshots `energy_start` from the perturbed
    //    state, the Henyey solver residual (`tot_E_equ_err`) remains near zero,
    //    while true global accounting `total_energy_end - (total_energy_old + S)`
    //    must detect the exact injected perturbation!
    let mut pert = base;
    pert.save_old_values();
    record_start_of_step_energy_integrals(&mut pert);
    let e_old = pert.aux.step_hist.total_energy_old;
    // Shift `ln_t` slightly across an interior shell band so `set_hydro_vars`
    // inside `do_struct_burn_mix` recomputes a known `delta_e_inj` > 0.
    let k_lo = pert.nz / 4;
    let k_hi = pert.nz / 2;
    for ln_t in &mut pert.ln_t[k_lo..k_hi] {
        *ln_t += 2.0e-3;
    }
    set_hydro_vars(&mut pert, false, false).expect("perturbed hydro_vars");
    let e_after_pert = eval_total_energy_integrals(&pert).5;
    let delta_e_inj = e_after_pert - e_old;
    assert!(
        (delta_e_inj / e_old).abs() > 1.0e-4,
        "injected perturbation should be clearly measurable: {:.3e}",
        delta_e_inj / e_old
    );

    let rep_pert =
        do_struct_burn_mix(&mut pert, dt, false, true, &[]).expect("pert sbm");
    assert!(rep_pert.converged);
    update_step_energy_conservation(&mut pert, dt);

    let err_pert = pert.aux.step_hist.error_in_energy_conservation;
    let e_end_pert = pert.aux.step_hist.total_energy_end;
    let equ_err = pert.aux.step_hist.tot_e_equ_err;
    // Solver residual alone is tiny, proving the old solver-residual shortcut
    // would have missed `delta_e_inj`.
    assert!(
        (equ_err / e_end_pert).abs() < 1.0e-6,
        "solver equation residual should stay small: {:.3e}",
        equ_err / e_end_pert
    );
    // Global energy accounting captures `delta_e_inj` accurately.
    assert!(
        rel_err(err_pert, delta_e_inj) < 2.0e-2,
        "global accounting must measure injected delta_E={delta_e_inj:.6e}, \
         got err={err_pert:.6e} (rel diff {:.3e})",
        rel_err(err_pert, delta_e_inj)
    );

    // Verify `mesa_history.rs` energy columns match `history.f90:1901-1936`.
    let rel_e = history_float(&pert, "rel_E_err");
    let abs_rel_e = history_float(&pert, "abs_rel_E_err");
    let log_rel_e = history_float(&pert, "log_rel_E_err");
    let rel_run_e = history_float(&pert, "rel_run_E_err");
    let log_rel_run_e = history_float(&pert, "log_rel_run_E_err");
    assert_eq!(rel_e, err_pert / e_end_pert.abs());
    assert_eq!(abs_rel_e, (err_pert / e_end_pert).abs());
    assert_eq!(
        log_rel_e,
        flexmesa_math::log10((err_pert / e_end_pert).abs())
    );
    assert_eq!(rel_run_e, pert.aux.cumulative_energy_error / e_end_pert);
    assert_eq!(
        log_rel_run_e,
        flexmesa_math::log10(
            (pert.aux.cumulative_energy_error / e_end_pert).abs()
        )
    );
}

/// Verifies `hard_limit_for_rel_error_in_energy_conservation` forces a step
/// retry when global conservation error exceeds the hard limit (for steps
/// after `doing_first_model_of_run`, matching `star/private/timestep.f90:132,
/// 322-324`), and confirms `cumulative_energy_error` is rolled back across the
/// rejected attempt.
#[test]
fn test_hard_limit_rel_energy_error_forces_retry_and_rolls_back_cumulative_err()
{
    // Advance past `doing_first_model_of_run` (`timestep.f90:132`) so
    // `check_rel_error_in_energy` (`timestep.f90:322-324`) is active.
    let mut base = solar_zams(EnergyEqnOption::EpsGrav);
    base.controls
        .ts_stop
        .hard_limit_for_rel_error_in_energy_conservation = 1.0;
    base.controls
        .ts_stop
        .limit_for_rel_error_in_energy_conservation = 1.0;
    base.dt = 1.0e5 * SECYER;
    base.dt_next = base.dt;
    let warm =
        star_evolve_step(&mut base, false, false, true).expect("warmup step");
    assert!(warm.converged);
    assert!(!base.doing_first_model_of_run);

    // Measure the genuine single-attempt `|rel_E_err|` of an `EpsGrav` step
    // at `dt0` without triggering energy retries.
    let dt0 = 5.0e7 * SECYER;
    let mut probe = base.clone();
    probe.dt = dt0;
    probe.dt_next = dt0;
    let rep_probe =
        star_evolve_step(&mut probe, false, false, true).expect("probe step");
    assert_eq!(rep_probe.retries, 0);
    let probe_rel_err = (probe.aux.step_hist.error_in_energy_conservation
        / probe.aux.step_hist.total_energy_end)
        .abs();
    assert!(
        probe_rel_err > 0.0 && probe_rel_err.is_finite(),
        "probe eps_grav step must have positive finite rel_E_err: {probe_rel_err:.3e}"
    );

    // Set `hard_limit_for_rel_error_in_energy_conservation` strictly below
    // `probe_rel_err` on a fresh copy with a known `cumulative_energy_error`.
    // Attempt 0 must be rejected by `check_rel_error_in_energy`, shrinking `dt`
    // until `|rel_E_err| <= hard_limit`, and rolling back the rejected attempt's
    // error from `cumulative_energy_error`.
    let hard_lim = 0.5 * probe_rel_err;
    let mut model = base;
    model
        .controls
        .ts_stop
        .hard_limit_for_rel_error_in_energy_conservation = hard_lim;
    model
        .controls
        .ts_stop
        .limit_for_rel_error_in_energy_conservation = 0.5 * hard_lim;
    model.dt = dt0;
    model.dt_next = dt0;
    let initial_cum_err = 1.25e42;
    model.aux.cumulative_energy_error = initial_cum_err;

    let rep = star_evolve_step(&mut model, false, false, true)
        .expect("step must succeed after energy-conservation retry");

    assert!(rep.converged);
    assert!(
        rep.retries >= 1,
        "hard_limit_for_rel_error_in_energy_conservation={hard_lim:.3e} \
         (< probe {probe_rel_err:.3e}) must trigger retry, got retries={}",
        rep.retries
    );
    assert!(
        rep.dt < dt0,
        "retry must reduce dt (got {:.3e} vs initial {:.3e})",
        rep.dt,
        dt0
    );
    let accepted_err = model.aux.step_hist.error_in_energy_conservation;
    let accepted_e_end = model.aux.step_hist.total_energy_end;
    assert!(
        (accepted_err / accepted_e_end).abs() <= hard_lim,
        "accepted step |rel_E_err|={:.3e} must satisfy hard_limit={hard_lim:.3e}",
        (accepted_err / accepted_e_end).abs()
    );
    // Cumulative energy error must equal `initial_cum_err + |accepted_err|`
    // (rejected attempt's error was rolled back cleanly).
    assert_eq!(
        model.aux.cumulative_energy_error,
        initial_cum_err + accepted_err.abs()
    );
}

/// Verifies `EnergyEqnOption::EpsGrav` reports genuine non-zero conservation
/// error (not hardcoded `0.0`), obeys `absolute_cumulative_energy_err`, and
/// populates all `history.f90` diagnostic columns cleanly.
#[test]
fn test_eps_grav_reports_genuine_conservation_error_and_updates_cumulative_err()
{
    let mut model = solar_zams(EnergyEqnOption::EpsGrav);
    model
        .controls
        .ts_stop
        .hard_limit_for_rel_error_in_energy_conservation = 1.0;
    model.dt = 5.0e6 * SECYER;
    model.dt_next = model.dt;

    let mut expected_cum = 0.0_f64;
    for step in 0..3 {
        let rep = star_evolve_step(&mut model, false, false, true)
            .unwrap_or_else(|e| panic!("eps_grav step {step} failed: {e}"));
        assert!(rep.converged);
        let sh = &model.aux.step_hist;
        assert!(sh.error_in_energy_conservation.is_finite());
        assert_ne!(
            sh.error_in_energy_conservation, 0.0,
            "eps_grav step {step} must report genuine non-zero conservation error"
        );
        // Exact upstream balance identity (`star/private/evolve.f90:1356-1357`).
        assert_eq!(
            sh.error_in_energy_conservation,
            sh.total_energy_end
                - (sh.total_energy_old + sh.total_energy_sources_and_sinks)
        );
        expected_cum += sh.error_in_energy_conservation.abs();
        assert_eq!(model.aux.cumulative_energy_error, expected_cum);
    }

    assert!(history_float(&model, "total_energy_sources_and_sinks").is_finite());
    assert!(history_float(&model, "total_eps_grav").is_finite());
    assert!(history_float(&model, "total_nuclear_heating") > 0.0);
    assert!(history_float(&model, "log_rel_run_E_err").is_finite());

    // Signed cumulative mode (`absolute_cumulative_energy_err = false`).
    model.controls.absolute_cumulative_energy_err = false;
    let prev_cum = model.aux.cumulative_energy_error;
    let rep = star_evolve_step(&mut model, false, false, true)
        .expect("signed cumulative eps_grav step");
    assert!(rep.converged);
    assert_eq!(
        model.aux.cumulative_energy_error,
        prev_cum + model.aux.step_hist.error_in_energy_conservation
    );
}

/// Verifies mass-change energy accounting (`adjust_mass` + `calculate_eps_mdot`)
/// populates `total_energy_before_adjust_mass`, `total_energy_after_adjust_mass`,
/// `energy_change_from_do_adjust_mass_and_calculate_eps_mdot`,
/// `mdot_acoustic_surface`, and `mdot_adiabatic_surface` even when
/// `eps_mdot_factor == 0`, matching `adjust_mass.f90:75-95`,
/// `eps_mdot.f90:494-678`, and `evolve.f90:599-604`.
#[test]
fn test_adjust_mass_records_phase1_mdot_energy_accounting() {
    let mut model = solar_zams(EnergyEqnOption::Dedt);
    model.controls.mix_wind.eps_mdot_factor = 0.0;
    model.mstar_dot = -1.0e-7 * M_SUN / SECYER;
    let dt = 1.0e4 * SECYER;
    model.save_old_values();
    record_start_of_step_energy_integrals(&mut model);
    let e_old = model.aux.step_hist.total_energy_old;

    do_adjust_mass_step(&mut model, dt).expect("do_adjust_mass_step");
    let sh = &model.aux.step_hist;
    assert_eq!(sh.total_energy_before_adjust_mass, e_old);
    assert!(sh.total_energy_after_adjust_mass.is_finite());
    assert_ne!(sh.total_energy_after_adjust_mass, 0.0);
    assert_eq!(
        sh.energy_change_from_do_adjust_mass_and_calculate_eps_mdot,
        sh.total_energy_after_adjust_mass - sh.total_energy_before_adjust_mass
    );
    assert!(sh.mdot_acoustic_surface.is_finite());
    assert!(sh.mdot_adiabatic_surface.is_finite());
    assert!(model.eps_mdot.iter().all(|&v| v == 0.0));
}

/// Verifies `.photo` v30 roundtrip preserves `have_initial_energy_integrals`,
/// `total_energy_initial`, `cumulative_energy_error`, and bit-exact restart
/// consistency (`photo_out.f90:69-70`).
#[test]
fn test_photo_v30_preserves_initial_energy_integrals_and_restart_identity() {
    assert_eq!(PHOTO_FORMAT_VERSION, 30);
    let mut model = solar_zams(EnergyEqnOption::Dedt);
    model.dt = 1.0e5 * SECYER;
    model.dt_next = model.dt;
    let rep = star_evolve_step(&mut model, false, false, true).expect("step 1");
    assert!(rep.converged);
    assert!(model.aux.step_hist.have_initial_energy_integrals);
    assert_ne!(model.aux.step_hist.total_energy_initial, 0.0);

    let bytes = write_photo_bytes(&model);
    let loaded = load_photo_bytes(&bytes).expect("load_photo_bytes v30");
    assert!(loaded.aux.step_hist.have_initial_energy_integrals);
    assert_eq!(
        loaded.aux.step_hist.total_energy_initial.to_bits(),
        model.aux.step_hist.total_energy_initial.to_bits()
    );
    assert_eq!(
        loaded.aux.cumulative_energy_error.to_bits(),
        model.aux.cumulative_energy_error.to_bits()
    );
    let (c1, c2) =
        verify_photo_restart_parity(&model, 1).expect("restart parity");
    assert_eq!(c1, c2);
}

struct TempInlistGuard {
    root: std::path::PathBuf,
}

impl TempInlistGuard {
    fn new(tag: &str) -> Self {
        static SEQ: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let id = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "flexmesa_star_151_{tag}_{}_{id}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("create temp inlist dir");
        Self { root }
    }
}

impl Drop for TempInlistGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// Verifies `&controls` namelist wiring for
/// `include_rotation_in_total_energy` and `absolute_cumulative_energy_err`
/// plus strict rejection of malformed boolean inputs.
#[test]
fn test_inlist_loads_energy_accounting_controls() {
    let tmp = TempInlistGuard::new("controls");
    let valid = tmp.root.join("inlist_energy_controls");
    std::fs::write(
        &valid,
        "&kap\n\
           Zbase = 0.02d0\n\
         /\n\
         &controls\n\
           initial_mass = 1.0d0\n\
           include_rotation_in_total_energy = .true.\n\
           absolute_cumulative_energy_err = .false.\n\
         /\n",
    )
    .expect("write valid inlist");
    let bundle = InlistBundle::from_file(&valid).expect("parse valid inlist");
    assert_eq!(bundle.validation_report.unsupported_controls.len(), 0);
    let mut extras = DefaultStarExtras;
    let model = init_or_update_model_from_bundle(&bundle, None, &mut extras)
        .expect("init model from bundle");
    assert!(model.controls.include_rotation_in_total_energy);
    assert!(!model.controls.absolute_cumulative_energy_err);

    for (tag, key) in [
        ("bad_rot", "include_rotation_in_total_energy"),
        ("bad_abs", "absolute_cumulative_energy_err"),
    ] {
        let bad = tmp.root.join(format!("inlist_{tag}"));
        std::fs::write(&bad, format!("&controls\n  {key} = 'not_a_bool'\n/\n"))
            .expect("write malformed inlist");
        let err = InlistBundle::from_file(&bad)
            .expect_err("malformed boolean control must fail");
        assert!(matches!(err, StarError::InvalidInput(_)));
    }
}

/// Verifies `&star_job::set_initial_cumulative_energy_error` &
/// `new_cumulative_energy_error` reset `cumulative_energy_error` on a `.mod`
/// load and on a model carried over from an earlier stage, which is not a
/// photo restart (`star/job/run_star_support.f90:433-442, 2762-2767`).
#[test]
fn test_set_initial_cumulative_energy_error_startup_and_restart() {
    let tmp = TempInlistGuard::new("cum_e_err");
    let mut seed = solar_zams(EnergyEqnOption::Dedt);
    seed.aux.cumulative_energy_error = 1.25e42;
    let mod_path = tmp.root.join("seed.mod");
    std::fs::write(
        &mod_path,
        write_mod_string(&seed).expect("write_mod_string"),
    )
    .expect("write seed.mod");

    let inlist_fresh = tmp.root.join("inlist_fresh");
    std::fs::write(
        &inlist_fresh,
        "&star_job\n\
           load_saved_model = .true.\n\
           load_model_filename = 'seed.mod'\n\
           set_initial_cumulative_energy_error = .true.\n\
           new_cumulative_energy_error = -4.5d38\n\
         /\n\
         &kap\n\
           Zbase = 0.02d0\n\
         /\n",
    )
    .expect("write inlist_fresh");
    let bundle_fresh =
        InlistBundle::from_file(&inlist_fresh).expect("parse fresh inlist");
    let mut extras = DefaultStarExtras;
    let mut model =
        init_or_update_model_from_bundle(&bundle_fresh, None, &mut extras)
            .expect("fresh init");
    assert_eq!(model.aux.cumulative_energy_error, -4.5e38);
    assert_eq!(model.aux.step_hist.cumulative_energy_error_old, -4.5e38);

    // A model carried over from an earlier stage (`existing_model = Some(prev)`)
    // stands for the `.mod` file upstream reloads, so the reset applies.
    model.aux.cumulative_energy_error = 9.0e40;
    let restarted = init_or_update_model_from_bundle(
        &bundle_fresh,
        Some(model),
        &mut extras,
    )
    .expect("restart init");
    assert_eq!(restarted.aux.cumulative_energy_error, -4.5e38);
}

/// Verifies non-finite global energy accounting aborts via
/// `StarError::Terminated` instead of retrying (`evolve.f90:1337-1354, 1548-1595`).
#[test]
fn test_non_finite_energy_error_terminates_without_retry() {
    let mut model = solar_zams(EnergyEqnOption::Dedt);
    model.dt = 1.0e4 * SECYER;
    model.dt_next = model.dt;
    model.aux.cumulative_energy_error = f64::NAN;
    let err = star_evolve_step(&mut model, false, false, true)
        .expect_err("non-finite energy error must terminate");
    match err {
        StarError::Terminated { reason, .. } => {
            assert!(
                reason.contains("okay_energy_conservation: non-finite"),
                "unexpected termination reason: {reason}"
            );
        }
        other => panic!("expected StarError::Terminated, got {other:?}"),
    }
    assert_eq!(model.num_retries, 0);
}

/// Verifies `op_split_burn = true` with `dxdt_nuc_factor == 0` zeroes
/// hot-zone `eps_nuc` and leaves a zero `burn_avg_epsnuc` untouched, because
/// `do_burn` is skipped (`struct_burn_mix.f90:79-98, 100`). Upstream does not
/// clear `burn_avg_epsnuc` in that block, so a stale nonzero value is not
/// cleared and the earlier precondition of `123.0` is replaced by zeros.
#[test]
fn test_op_split_burn_with_zero_dxdt_nuc_factor_keeps_zero_burn_avg_and_eps_nuc(
) {
    let mut model = solar_zams(EnergyEqnOption::Dedt);
    let nz = model.nz;
    model.controls.op_split_burn = true;
    model.controls.op_split_burn_min_t = 1.0e5;
    model.controls.dxdt_nuc_factor = 0.0;
    model.aux.burn_avg_epsnuc = vec![0.0; nz];
    model.save_start_values().expect("save_start_values");
    flexmesa_star::record_start_of_step_energy_integrals(&mut model);
    model.save_old_values();
    let rep = do_struct_burn_mix(&mut model, 10.0 * SECYER, true, true, &[])
        .expect("op_split_burn step with dxdt_nuc_factor = 0");
    assert!(rep.converged);
    for k in 0..nz {
        assert_eq!(model.aux.burn_avg_epsnuc[k], 0.0);
        if model.t[k] >= model.controls.op_split_burn_min_t {
            assert_eq!(model.eps_nuc[k], 0.0);
        }
    }
}

/// Verifies `is_velocity_time_centered` stays step-consistent across
/// `steps_before_use_velocity_time_centering` between solver and energy accounting.
#[test]
fn test_velocity_time_centering_transition_step_is_consistent() {
    let setup = |steps_before: isize| {
        let mut m = solar_zams(EnergyEqnOption::Dedt);
        m.controls.v_flag = false;
        m.controls.include_l_in_velocity_time_centering = true;
        m.controls.l_theta_for_velocity_time_centering = 0.5;
        m.controls.steps_before_use_velocity_time_centering = steps_before;
        m.controls
            .max_logt_for_include_p_and_l_in_velocity_time_centering = 99.0;
        m.model_number = 4;
        m.dt = 1.0e5 * SECYER;
        m.dt_next = m.dt;
        m
    };
    let mut m_transition = setup(5);
    let mut m_control = setup(-1);
    let rep_t =
        star_evolve_step(&mut m_transition, false, false, true).expect("step");
    let rep_c =
        star_evolve_step(&mut m_control, false, false, true).expect("ctrl");
    assert!(rep_t.converged && rep_c.converged);
    assert_eq!(m_transition.model_number, 5);
    assert_eq!(
        m_transition
            .aux
            .step_hist
            .error_in_energy_conservation
            .to_bits(),
        m_control
            .aux
            .step_hist
            .error_in_energy_conservation
            .to_bits()
    );
}

/// Verifies `HydroAuxState` captures `cv_start` and `de_drho_start` in
/// `save_start_values`, populates `ergs_error` matching `tot_e_equ_err` in
/// `assemble_henyey_system`, resizes all three in `resize_zones`, and leaves
/// the solver trajectory bit-identical (`#152`).
#[test]
fn test_cv_de_drho_start_ergs_error_snapshot_resize_and_step_accounting() {
    let mut model = solar_zams(EnergyEqnOption::EpsGrav);
    model.dt = 1.0e5 * SECYER;
    model.dt_next = model.dt;
    model.save_start_values().expect("save_start_values");

    let nz = model.nz;
    let init_cv = model.cv.clone();
    let init_de_drho: Vec<f64> = model
        .eos_res
        .iter()
        .take(nz)
        .map(|r| r[I_DE_DRHO])
        .collect();
    assert_eq!(model.aux.cv_start, init_cv);
    assert_eq!(model.aux.de_drho_start, init_de_drho);
    assert_eq!(model.aux.ergs_error.len(), nz);

    let mut perturbed = model.clone();
    perturbed.aux.cv_start.fill(1.23e45);
    perturbed.aux.de_drho_start.fill(-4.56e12);
    perturbed.aux.ergs_error.fill(7.89e33);

    let rep_a =
        star_evolve_step(&mut model, false, false, true).expect("step base");
    let rep_b = star_evolve_step(&mut perturbed, false, false, true)
        .expect("step perturbed");
    assert!(rep_a.converged && rep_b.converged);
    assert_eq!(model.nz, perturbed.nz);

    // Pre-existing snapshot/residual values have zero trajectory feedback.
    assert_eq!(
        model.aux.step_hist.total_energy_end.to_bits(),
        perturbed.aux.step_hist.total_energy_end.to_bits()
    );
    assert_eq!(
        model.aux.step_hist.tot_e_equ_err.to_bits(),
        perturbed.aux.step_hist.tot_e_equ_err.to_bits()
    );
    for k in 0..model.nz {
        assert_eq!(model.lnd[k].to_bits(), perturbed.lnd[k].to_bits());
        assert_eq!(model.ln_t[k].to_bits(), perturbed.ln_t[k].to_bits());
        assert_eq!(
            model.aux.cv_start[k].to_bits(),
            perturbed.aux.cv_start[k].to_bits()
        );
        assert_eq!(
            model.aux.de_drho_start[k].to_bits(),
            perturbed.aux.de_drho_start[k].to_bits()
        );
        assert_eq!(
            model.aux.ergs_error[k].to_bits(),
            perturbed.aux.ergs_error[k].to_bits()
        );
    }

    // `ergs_error` sums strictly to `tot_e_equ_err` and all fields match nz.
    assert_eq!(model.aux.ergs_error.len(), model.nz);
    assert_eq!(model.aux.cv_start.len(), model.nz);
    assert_eq!(model.aux.de_drho_start.len(), model.nz);
    assert!(model.aux.ergs_error.iter().all(|v| v.is_finite()));
    let sum_ergs: f64 = model.aux.ergs_error.iter().sum();
    assert_eq!(
        sum_ergs.to_bits(),
        model.aux.step_hist.tot_e_equ_err.to_bits()
    );
    assert!(model
        .aux
        .cv_start
        .iter()
        .zip(&model.cv)
        .any(|(a, b)| a != b));
    assert!(model
        .aux
        .de_drho_start
        .iter()
        .zip(model.eos_res.iter().take(model.nz))
        .any(|(&a, row)| a != row[I_DE_DRHO]));

    // `resize_zones` propagates cleanly to all three arrays.
    let new_nz = model.nz + 5;
    model.resize_zones(new_nz);
    assert_eq!(model.aux.cv_start.len(), new_nz);
    assert_eq!(model.aux.de_drho_start.len(), new_nz);
    assert_eq!(model.aux.ergs_error.len(), new_nz);

    // `.photo` restart preserves `cv_start`, `de_drho_start`, and `ergs_error`.
    let loaded = load_photo_bytes(&write_photo_bytes(&perturbed))
        .expect("photo roundtrip");
    assert_eq!(loaded.aux.cv_start, perturbed.aux.cv_start);
    assert_eq!(loaded.aux.de_drho_start, perturbed.aux.de_drho_start);
    assert_eq!(loaded.aux.ergs_error, perturbed.aux.ergs_error);

    // `PrimaryStateBuffer` snapshot/restore preserves all three fields.
    let snap = PrimaryStateBuffer::from_model(&perturbed);
    let mut rolled_back = perturbed.clone();
    rolled_back.aux.cv_start.fill(-1.0);
    rolled_back.aux.de_drho_start.fill(-2.0);
    rolled_back.aux.ergs_error.fill(-3.0);
    snap.restore_into(&mut rolled_back)
        .expect("PrimaryStateBuffer::restore_into");
    assert_eq!(rolled_back.aux.cv_start, perturbed.aux.cv_start);
    assert_eq!(rolled_back.aux.de_drho_start, perturbed.aux.de_drho_start);
    assert_eq!(rolled_back.aux.ergs_error, perturbed.aux.ergs_error);
}
