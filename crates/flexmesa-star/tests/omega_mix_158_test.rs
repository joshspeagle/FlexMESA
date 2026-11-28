// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration and numerical-oracle tests for Issue `#158`
//! (`star/private/solve_omega_mix.f90` `Quad` port + `#153` rotation remap
//! handoff verification).

#![forbid(unsafe_code)]
#![allow(clippy::similar_names, clippy::too_many_lines)]

use std::collections::HashMap;
use std::path::PathBuf;

use flexmesa_const::{M_SUN, R_SUN, SECYER, STANDARD_CGRAV, TWO_THIRDS};
use flexmesa_star::hydro_rotation::{
    eval_fp_ft_single, eval_fp_ft_single_dual, eval_i_rot_dual, roche_a,
    roche_c, set1_i_rot, w_div_w_roche_omega, LOG_TERM_POWER,
};
use flexmesa_star::report_utils::equatorial_and_polar_r_cm;
use flexmesa_star::{
    do_adjust_mesh, do_solve_omega_mix, do_solve_omega_mix_with_extras,
    eval_i_rot, eval_photosphere_info, get_rotation_sigmas,
    init_or_update_model_from_bundle, load_zams_model, remesh_split_merge,
    set_hydro_vars, set_rotation_info, solve_omega_mix, star_evolve_step,
    star_evolve_step_with_extras, total_angular_momentum,
    use_xh_to_update_i_rot_and_j_rot, w_div_w_roche_jrot,
    BinaryOtherTorqueHandle, BinaryOtherTorqueHook, DefaultStarExtras,
    InlistBundle, InlistLoadOptions, MeshFunctionConfig, OmegaMixOutcome,
    PrimaryStateBuffer, StarControls, StarError, StarExtras, StarJobControls,
    StarModel,
};

fn star_suite_dir() -> PathBuf {
    let base = std::env::var("MESA_DIR").map_or_else(
        |_| {
            let home = std::env::var("HOME").unwrap_or_default();
            PathBuf::from(home)
                .join(".cache")
                .join("flexmesa")
                .join("mesa_test_inputs")
        },
        PathBuf::from,
    );
    base.join("star").join("test_suite")
}

fn frac_u(k: usize, nz: usize) -> f64 {
    let denom = u32::try_from(nz.saturating_sub(1).max(1)).unwrap_or(1);
    f64::from(u32::try_from(k).unwrap_or(0)) / f64::from(denom)
}

fn build_rotating_4msun(omega_fn: impl Fn(usize, usize) -> f64) -> StarModel {
    let mut ctrl = StarControls {
        initial_mass: 4.0,
        rotation_flag: true,
        zbase: 0.02,
        ..StarControls::default()
    };
    ctrl.mix_wind.am_nu_factor = 1.0;
    ctrl.mix_wind.am_d_mix_factor = 0.033_333_3;
    ctrl.mix_wind.premix_omega = false;
    let mut model =
        load_zams_model(ctrl, StarJobControls::default()).expect("4 Msun ZAMS");
    let nz = model.nz;
    for (k, om) in model.omega.iter_mut().enumerate().take(nz) {
        *om = omega_fn(k, nz);
    }
    use_xh_to_update_i_rot_and_j_rot(&mut model);
    set_rotation_info(&mut model, false);
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model
}

#[test]
fn test_quad_solve_omega_mix_conserves_j_at_1e13_stiffness_without_rescaling() {
    let mut model = build_rotating_4msun(|k, nz| {
        let frac = frac_u(k, nz);
        1.0e-5 * (1.0 + 4.0 * frac)
    });
    let nz = model.nz;
    let dt = 1.0e5 * SECYER;
    model.controls.mix_wind.am_sig_term_limit = 1.0e13;
    model.aux.am_nu_omega.fill(1.0e26);
    model.aux.am_nu_j.fill(0.0);

    // Verify `get_rotation_sigmas` allows stiffness up to `1e13` (no `1e10` cap).
    get_rotation_sigmas(&mut model, 0, nz, dt);
    let max_stiff = (0..(nz - 1))
        .map(|k| {
            let dmbar = model.xmstar * model.dq[k].min(model.dq[k + 1]);
            model.aux.am_sig_omega[k] * dt / dmbar
        })
        .fold(0.0_f64, f64::max);
    assert!(
        max_stiff > 1.0e12 && max_stiff <= 1.000_001e13,
        "expected am_sig_term_limit=1e13 without 1e10 cap, got max_stiff={max_stiff:.3e}"
    );

    let j0 = total_angular_momentum(&model);
    let omega0_before = model.omega[0];
    let shear0 = (model.omega[0] - model.omega[nz - 1]).abs();
    model.aux.step_hist.num_rotation_solver_steps = 0;

    let outcome =
        do_solve_omega_mix(&mut model, dt).expect("do_solve_omega_mix");
    assert_eq!(outcome, OmegaMixOutcome::KeepGoing);
    assert!(
        (4..=20).contains(&model.aux.step_hist.num_rotation_solver_steps),
        "expected num_rotation_solver_steps in 4..=20, got {}",
        model.aux.step_hist.num_rotation_solver_steps
    );

    let j1 = total_angular_momentum(&model);
    let rel_err = (j1 - j0).abs() / j0.abs();
    // Upstream `solve_omega_mix.f90:233-238` updates `k = 2..nz` and sets
    // `omega(1) = omega(2)` (leaving `dX(1) = 0`), so when a large single-step
    // jump triggers `num_iters >= 2` the conservation residual is bounded by
    // the surface half-shell term `dm_bar[0] * i_rot[0] * |Delta omega[0]| / J0`
    // (~1.6e-12 on 4 Msun ZAMS where `dq[0] = 4e-13`).
    let surf_bound = (model.dm_bar[0]
        * model.aux.i_rot[0]
        * (model.omega[0] - omega0_before).abs())
        / j0.abs()
        + 1.0e-14;
    assert!(
        rel_err <= surf_bound && rel_err < 1.0e-11,
        "Quad solve_omega_mix J_tot error {rel_err:.3e} exceeded upstream surface bound {surf_bound:.3e}"
    );
    let shear1 = (model.omega[0] - model.omega[nz - 1]).abs();
    assert!(
        shear1 < 1.0e-3 * shear0,
        "expected strong diffusion to flatten shear: {shear0:.3e} -> {shear1:.3e}"
    );

    // In the `num_iters == 1` regime (`avg_del <= 1e-7`) at the full `1e13`
    // stiffness limit, `Quad` conserves `J_tot` to `< 1e-14` without rescaling.
    let mut m_small = build_rotating_4msun(|k, nz_s| {
        let frac = frac_u(k, nz_s);
        1.0e-7 * (1.0 + 0.5 * frac)
    });
    m_small.controls.mix_wind.am_sig_term_limit = 1.0e13;
    m_small.aux.am_nu_omega.fill(1.0e26);
    m_small.aux.am_nu_j.fill(0.0);
    let j_s0 = total_angular_momentum(&m_small);
    assert_eq!(
        do_solve_omega_mix(&mut m_small, dt).expect("small-del 1e13"),
        OmegaMixOutcome::KeepGoing
    );
    let j_s1 = total_angular_momentum(&m_small);
    let rel_s = (j_s1 - j_s0).abs() / j_s0.abs();
    assert!(
        rel_s < 1.0e-14,
        "Quad solve_omega_mix at 1e13 stiffness (num_iters=1) must conserve J_tot to <1e-14: rel={rel_s:.3e}"
    );

    // Standalone `solve_omega_mix` wrapper on a non-rotating ZAMS model
    // (`rotation_flag = false`, `i_rot` initially 0, matching `run_omega_mix`).
    let mut non_rot = load_zams_model(
        StarControls {
            initial_mass: 4.0,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("4 Msun non-rot ZAMS");
    let nz_nr = non_rot.nz;
    for (k, om) in non_rot.omega.iter_mut().enumerate().take(nz_nr) {
        let frac = frac_u(k, nz_nr);
        *om = 1.0e-5 + 4.0e-5 * frac;
    }
    let j_nr0: f64 = (0..nz_nr)
        .map(|k| {
            non_rot.dm_bar[k]
                * TWO_THIRDS
                * non_rot.r[k]
                * non_rot.r[k]
                * non_rot.omega[k]
        })
        .sum();
    let nu = vec![1.0e12; nz_nr];
    solve_omega_mix(&mut non_rot, 1.0e7, &nu, 4);
    let j_nr1: f64 = (0..nz_nr)
        .map(|k| non_rot.dm_bar[k] * non_rot.j_rot[k])
        .sum();
    let rel_nr = (j_nr1 - j_nr0).abs() / j_nr0.abs();
    assert!(
        rel_nr < 1.0e-12,
        "standalone solve_omega_mix must conserve J_tot within 1e-12: rel={rel_nr:.3e}"
    );
}

#[test]
fn test_am_sig_j_d2irot_pure_j_diffusion_relaxes_toward_uniform_j_rot() {
    // Uniform omega has zero `d2omega` curvature; only the `am_sig_j` `d2irot`
    // term drives transport when `am_nu_omega == 0` and `am_nu_j > 0`.
    let mut model = build_rotating_4msun(|_, _| 2.0e-5);
    let nz = model.nz;
    let dt = 1.0e2 * SECYER;

    model.controls.mix_wind.am_nu_omega_non_rot_factor = 0.0;
    model.controls.mix_wind.am_nu_omega_rot_factor = 0.0;
    model.controls.mix_wind.am_nu_j_non_rot_factor = 1.0;
    model.controls.mix_wind.am_nu_j_rot_factor = 0.0;
    model.aux.am_nu_omega.fill(0.0);
    model.aux.am_nu_j.fill(1.0e14);

    let j0 = total_angular_momentum(&model);
    let j_rot_before = model.j_rot.clone();
    let omega_core_before = model.omega[nz - 1];
    let omega_surf_before = model.omega[0];

    let outcome =
        do_solve_omega_mix(&mut model, dt).expect("do_solve_omega_mix");
    assert_eq!(outcome, OmegaMixOutcome::KeepGoing);
    assert!(
        model.aux.am_sig_omega.iter().all(|&s| s == 0.0),
        "am_sig_omega must remain identically zero for pure-j diffusion"
    );
    assert!(
        model.aux.am_sig_j.iter().any(|&s| s > 0.0),
        "am_sig_j must be positive"
    );

    let j1 = total_angular_momentum(&model);
    let rel_err = (j1 - j0).abs() / j0.abs();
    let surf_bound = (model.dm_bar[0]
        * model.aux.i_rot[0]
        * (model.omega[0] - omega_surf_before).abs())
        / j0.abs()
        + 1.0e-14;
    assert!(
        rel_err <= surf_bound && rel_err < 1.0e-11,
        "pure-j diffusion J_tot error {rel_err:.3e} exceeded upstream surface bound {surf_bound:.3e}"
    );

    // Since `j_rot = i_rot * omega` increases outward (`r` larger at surface),
    // `am_sig_j` transports `j` inward: core spins up, surface spins down, and
    // total `j_rot` span shrinks.
    assert!(
        model.omega[nz - 1] > omega_core_before,
        "core must spin up under pure-j inward transport: {} -> {}",
        omega_core_before,
        model.omega[nz - 1]
    );
    assert!(
        model.omega[0] < omega_surf_before,
        "surface must spin down under pure-j inward transport: {} -> {}",
        omega_surf_before,
        model.omega[0]
    );
    let span_before = (j_rot_before[0] - j_rot_before[nz - 1]).abs();
    let span_after = (model.j_rot[0] - model.j_rot[nz - 1]).abs();
    assert!(
        span_after < span_before,
        "j_rot span must shrink under pure-j diffusion: {span_before:.3e} -> {span_after:.3e}"
    );
}

#[test]
fn test_adaptive_substepping_and_recalc_mixing_info_each_substep() {
    let base = build_rotating_4msun(|k, nz| {
        let frac = frac_u(k, nz);
        3.0e-5 * (1.0 + 3.0 * frac * frac)
    });
    let dt = 1.0e4 * SECYER;

    // 1. Mild transport enforces `min_steps = 4`.
    let mut mild = base.clone();
    mild.aux.am_nu_omega.fill(1.0e4);
    mild.aux.am_nu_j.fill(0.0);
    mild.aux.step_hist.num_rotation_solver_steps = 0;
    let out_mild = do_solve_omega_mix(&mut mild, 1.0e3).expect("mild mix");
    assert_eq!(out_mild, OmegaMixOutcome::KeepGoing);
    assert_eq!(mild.aux.step_hist.num_rotation_solver_steps, 4);

    // 2. Compare `recalc_mixing_info_each_substep = false` vs `true` using
    // rotational shear-driven diffusivities (`am_nu_non_rotation_factor = 0`).
    let mut no_recalc = base.clone();
    no_recalc.controls.mix_wind.am_nu_non_rotation_factor = 0.0;
    no_recalc.controls.mix_wind.am_nu_factor = 1.0;
    no_recalc.controls.mix_wind.d_dsi_factor = 1.0;
    no_recalc.controls.mix_wind.d_ssi_factor = 1.0;
    no_recalc.controls.mix_wind.d_es_factor = 1.0;
    no_recalc.controls.mix_wind.d_gsf_factor = 1.0;
    no_recalc.controls.mix_wind.recalc_mixing_info_each_substep = false;
    set_hydro_vars(&mut no_recalc, false, false).expect("set_hydro_vars");

    let mut with_recalc = no_recalc.clone();
    with_recalc
        .controls
        .mix_wind
        .recalc_mixing_info_each_substep = true;

    let j0 = total_angular_momentum(&no_recalc);
    assert_eq!(
        do_solve_omega_mix(&mut no_recalc, dt).expect("no_recalc"),
        OmegaMixOutcome::KeepGoing
    );
    assert_eq!(
        do_solve_omega_mix(&mut with_recalc, dt).expect("with_recalc"),
        OmegaMixOutcome::KeepGoing
    );

    let j_no = total_angular_momentum(&no_recalc);
    let j_yes = total_angular_momentum(&with_recalc);
    assert!((j_no - j0).abs() / j0.abs() < 1.0e-11);
    assert!((j_yes - j0).abs() / j0.abs() < 1.0e-11);

    assert!(
        no_recalc.aux.step_hist.num_rotation_solver_steps >= 4
            && with_recalc.aux.step_hist.num_rotation_solver_steps >= 4
    );
    let max_omega_diff = no_recalc
        .omega
        .iter()
        .zip(&with_recalc.omega)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_omega_diff > 0.0,
        "recalc_mixing_info_each_substep=true must update rotational sigmas across substeps"
    );
}

#[test]
fn test_explicit_other_torque_extra_jdot_extra_omegadot_and_binary_hook() {
    struct ConstJdotExtras(f64);
    impl StarExtras for ConstJdotExtras {
        fn other_torque(&self, m: &StarModel) -> Option<Vec<f64>> {
            Some(vec![self.0; m.nz])
        }
    }

    struct ConstOmegaDotExtras(f64);
    impl StarExtras for ConstOmegaDotExtras {
        fn apply_other_torque(
            &mut self,
            m: &mut StarModel,
        ) -> Result<(), StarError> {
            m.aux.extra_omegadot.fill(self.0);
            Ok(())
        }
    }

    struct BinaryConstJdot(f64);
    impl BinaryOtherTorqueHook for BinaryConstJdot {
        fn apply(&self, model: &mut StarModel) -> Result<(), StarError> {
            for v in &mut model.aux.extra_jdot {
                *v += self.0;
            }
            Ok(())
        }
    }

    let base = build_rotating_4msun(|_, _| 1.0e-5);
    let nz = base.nz;
    let dt = 1.0e6;

    // 1. Constant `extra_jdot` via `StarExtras::other_torque` with zero diffusion.
    let mut m_jdot = base.clone();
    m_jdot.controls.use_other_torque = true;
    m_jdot.aux.am_nu_omega.fill(0.0);
    m_jdot.aux.am_nu_j.fill(0.0);
    let jdot_val = 2.5e9;
    let omega_before = m_jdot.omega.clone();
    let mut ex_j = ConstJdotExtras(jdot_val);
    assert_eq!(
        do_solve_omega_mix_with_extras(&mut m_jdot, dt, &mut ex_j)
            .expect("jdot"),
        OmegaMixOutcome::KeepGoing
    );
    for k in 1..nz {
        let expected = omega_before[k] + dt * jdot_val / m_jdot.aux.i_rot[k];
        let rel = (m_jdot.omega[k] - expected).abs() / expected.abs();
        assert!(
            rel < 1.0e-13,
            "extra_jdot closed-form mismatch at k={k}: got {:.16e}, expected {:.16e}",
            m_jdot.omega[k],
            expected
        );
    }

    // 2. Constant `extra_omegadot` via `StarExtras::apply_other_torque`.
    let mut m_omdot = base.clone();
    m_omdot.controls.use_other_torque = true;
    m_omdot.aux.am_nu_omega.fill(0.0);
    m_omdot.aux.am_nu_j.fill(0.0);
    let omdot_val = -1.0e-12;
    let mut ex_om = ConstOmegaDotExtras(omdot_val);
    assert_eq!(
        do_solve_omega_mix_with_extras(&mut m_omdot, dt, &mut ex_om)
            .expect("omdot"),
        OmegaMixOutcome::KeepGoing
    );
    for k in 0..nz {
        let expected = omega_before[k] + dt * omdot_val;
        let rel = (m_omdot.omega[k] - expected).abs() / expected.abs();
        assert!(
            rel < 1.0e-13,
            "extra_omegadot closed-form mismatch at k={k}: got {:.16e}, expected {:.16e}",
            m_omdot.omega[k],
            expected
        );
    }

    // 3. `binary_other_torque` hook plus `PrimaryStateBuffer` survival.
    let mut m_bin = base;
    m_bin.aux.am_nu_omega.fill(0.0);
    m_bin.aux.am_nu_j.fill(0.0);
    m_bin.aux.binary_other_torque =
        Some(BinaryOtherTorqueHandle::new(BinaryConstJdot(-1.5e9)));
    let snap = PrimaryStateBuffer::from_model(&m_bin);
    m_bin.aux.binary_other_torque = None;
    snap.restore_into(&mut m_bin).expect("restore_into");
    assert!(
        m_bin.aux.binary_other_torque.is_some(),
        "PrimaryStateBuffer must preserve binary_other_torque"
    );
    m_bin.aux.am_nu_omega.fill(0.0);
    m_bin.aux.am_nu_j.fill(0.0);
    assert_eq!(
        do_solve_omega_mix(&mut m_bin, dt).expect("binary torque"),
        OmegaMixOutcome::KeepGoing
    );
    for k in 1..nz {
        let expected = omega_before[k] - dt * 1.5e9 / m_bin.aux.i_rot[k];
        let rel = (m_bin.omega[k] - expected).abs() / expected.abs();
        assert!(
            rel < 1.0e-13,
            "binary_other_torque closed-form mismatch at k={k}: rel={rel:.3e}"
        );
    }
}

#[test]
fn test_implicit_other_torque_jacobian_and_retry_on_error() {
    // Stiff linear relaxation `extra_omegadot = -lambda * (omega - target)`
    // with exact diagonal derivative `d_extra_omegadot_domega_00 = -lambda`.
    struct StiffDragExtras {
        lambda: f64,
        target: f64,
        calls: usize,
    }
    impl StarExtras for StiffDragExtras {
        fn other_torque_implicit(
            &mut self,
            m: &mut StarModel,
        ) -> Result<(), StarError> {
            self.calls += 1;
            for k in 0..m.nz {
                m.aux.extra_omegadot[k] =
                    -self.lambda * (m.omega[k] - self.target);
                m.aux.d_extra_omegadot_domega_00[k] = -self.lambda;
            }
            Ok(())
        }
    }

    struct FailingImplicitExtras;
    impl StarExtras for FailingImplicitExtras {
        fn other_torque_implicit(
            &mut self,
            _m: &mut StarModel,
        ) -> Result<(), StarError> {
            Err(StarError::ConvergenceFailure(String::from(
                "synthetic implicit torque failure",
            )))
        }
    }

    let mut model = build_rotating_4msun(|_, _| 5.0e-5);
    model.controls.use_other_torque_implicit = true;
    model.aux.am_nu_omega.fill(0.0);
    model.aux.am_nu_j.fill(0.0);
    let dt = 1.0e6;
    let lambda = 1.0e-4; // `lambda * dt = 100` (stiff for explicit Euler)
    let target = 1.0e-5;
    let mut extras = StiffDragExtras {
        lambda,
        target,
        calls: 0,
    };

    let outcome = do_solve_omega_mix_with_extras(&mut model, dt, &mut extras)
        .expect("implicit torque");
    assert_eq!(outcome, OmegaMixOutcome::KeepGoing);
    assert!(extras.calls >= 4);
    for (k, &om) in model.omega.iter().enumerate() {
        assert!(
            (om - target).abs() < 1.0e-8,
            "stiff implicit drag must relax face {k} smoothly to target {target:.3e}, got {om:.6e}"
        );
        assert!(om >= target - 1.0e-15, "implicit drag must never overshoot");
    }

    // Error returned from `other_torque_implicit` triggers `OmegaMixOutcome::Retry`.
    let mut fail_m = build_rotating_4msun(|_, _| 2.0e-5);
    fail_m.controls.use_other_torque_implicit = true;
    fail_m.aux.am_nu_omega.fill(0.0);
    fail_m.aux.am_nu_j.fill(0.0);
    let mut fail_ex = FailingImplicitExtras;
    let retry_out =
        do_solve_omega_mix_with_extras(&mut fail_m, 1.0e4, &mut fail_ex)
            .expect("retry outcome");
    assert!(
        matches!(retry_out, OmegaMixOutcome::Retry(ref msg) if msg.contains("other_torque_implicit")),
        "expected Retry from failing other_torque_implicit, got {retry_out:?}"
    );

    // End-to-end step driver propagation: `other_torque_implicit` retry exhausts
    // `max_number_retries = 0` and terminates cleanly with `max_number_retries`.
    let mut step_m = build_rotating_4msun(|_, _| 2.0e-5);
    step_m.doing_first_model_of_run = false;
    step_m.controls.use_other_torque_implicit = true;
    step_m.controls.max_number_retries = 0;
    step_m.save_start_values().expect("save_start_values");
    step_m.save_old_values();
    let err = star_evolve_step_with_extras(
        &mut step_m,
        false,
        false,
        true,
        &mut fail_ex,
    )
    .expect_err(
        "step must fail on implicit torque retry with max_number_retries=0",
    );
    assert!(
        matches!(
            err,
            StarError::Terminated { ref reason, .. } if reason == "max_number_retries"
        ),
        "expected Terminated(max_number_retries), got {err:?}"
    );
}

#[test]
fn test_angular_momentum_error_retry_returns_retry_without_rescaling() {
    // Construct a controlled surface jump (`omega[0] != omega[1]`) with zero
    // diffusivity: upstream `s% omega(1) = s% omega(2)` produces a finite
    // `|J1 - J0| / |J0|` equal to `dm_bar[0] * i_rot[0] * |omega[1] - omega[0]| / J0`.
    let mut model = build_rotating_4msun(|_, _| 2.0e-5);
    model.aux.am_nu_omega.fill(0.0);
    model.aux.am_nu_j.fill(0.0);
    model.omega[0] = 8.0e-5;
    model.j_rot[0] = model.aux.i_rot[0] * model.omega[0];
    let j0 = total_angular_momentum(&model);
    let expected_rel = (model.dm_bar[0]
        * model.aux.i_rot[0]
        * (model.omega[1] - model.omega[0]).abs())
        / j0.abs();
    assert!(expected_rel > 0.0);

    // Tight threshold below `expected_rel` must return `OmegaMixOutcome::Retry`.
    let mut m_tight = model.clone();
    m_tight.controls.mix_wind.angular_momentum_error_retry = 0.5 * expected_rel;
    let out_retry =
        do_solve_omega_mix(&mut m_tight, 1.0e5).expect("tight conservation");
    assert!(
        matches!(out_retry, OmegaMixOutcome::Retry(ref msg) if msg.contains("conserve angular momentum")),
        "expected angular momentum retry, got {out_retry:?}"
    );

    // Threshold above `expected_rel` succeeds and does NOT globally rescale `omega[nz - 1]`.
    let mut m_loose = model;
    let core_omega_before = m_loose.omega[m_loose.nz - 1];
    m_loose.controls.mix_wind.angular_momentum_error_retry = 2.0 * expected_rel;
    let out_ok =
        do_solve_omega_mix(&mut m_loose, 1.0e5).expect("loose conservation");
    assert_eq!(out_ok, OmegaMixOutcome::KeepGoing);
    assert_eq!(
        m_loose.omega[m_loose.nz - 1],
        core_omega_before,
        "solve_omega_mix must not globally rescale omega to hide J drift"
    );
}

#[test]
fn test_153_handoff_adjust1_omega_and_remesh_split_merge_rapid_rotation() {
    // Rapid non-uniform rotation (`w_div_wc` up to ~0.65) where Paxton (2019)
    // `eval_i_rot` deviates significantly from spherical `(2/3) * r^2`.
    let mut base = build_rotating_4msun(|k, nz| {
        let frac = frac_u(k, nz);
        1.2e-4 * (1.0 + 1.5 * frac)
    });
    set_rotation_info(&mut base, false);
    let max_w = base
        .aux
        .w_div_wc
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    assert!(
        max_w > 0.3,
        "test must exercise rapid Roche distortion (w_div_wc > 0.3), got {max_w:.3}"
    );

    // 1. Standard `do_adjust_mesh` (`adjust1_omega` in `mesh_remap.rs`).
    let mut m_std = base.clone();
    m_std.controls.mesh_delta_coeff = 1.5;
    let j0_std = total_angular_momentum(&m_std);
    let cfg = MeshFunctionConfig::from_controls(&m_std.controls);
    let (splits_std, merges_std) =
        do_adjust_mesh(&mut m_std, &cfg).expect("do_adjust_mesh rotating");
    assert!(
        splits_std + merges_std > 0,
        "standard remesh must split or merge zones"
    );
    let j1_std = total_angular_momentum(&m_std);
    let rel_j_std = (j1_std - j0_std).abs() / j0_std.abs();
    assert!(
        rel_j_std < 1.0e-12,
        "standard remesh J_tot drift too large: rel={rel_j_std:.3e}"
    );
    assert_roche_i_rot_and_omega_consistent(&m_std, "do_adjust_mesh");

    // 2. `remesh_split_merge` (`adjust_mesh_split_merge.rs`).
    let mut m_amr = base;
    m_amr.controls.mix_wind.use_split_merge_amr = true;
    m_amr.controls.mix_wind.split_merge_amr_nz_baseline = 500;
    m_amr.controls.mix_wind.split_merge_amr_max_iters = 40;
    m_amr.controls.mix_wind.split_merge_amr_log_zoning = true;
    let j0_amr = total_angular_momentum(&m_amr);
    let (splits_amr, merges_amr) =
        remesh_split_merge(&mut m_amr).expect("remesh_split_merge rotating");
    assert!(
        splits_amr + merges_amr > 0,
        "split-merge AMR must split or merge zones"
    );
    let j1_amr = total_angular_momentum(&m_amr);
    let rel_j_amr = (j1_amr - j0_amr).abs() / j0_amr.abs();
    assert!(
        rel_j_amr < 1.0e-13,
        "remesh_split_merge J_tot drift too large: rel={rel_j_amr:.3e}"
    );
    assert_roche_i_rot_and_omega_consistent(&m_amr, "remesh_split_merge");
}

fn assert_roche_i_rot_and_omega_consistent(model: &StarModel, label: &str) {
    let mw = &model.controls.mix_wind;
    let mut any_nontrivial_distortion = false;
    for k in 0..model.nz {
        let w_expected = w_div_w_roche_jrot(
            model.r[k],
            model.m_grav[k],
            model.j_rot[k],
            model.cgrav[k],
            mw.w_div_wcrit_max,
            mw.w_div_wcrit_max2,
            false,
        );
        let i_expected =
            eval_i_rot(model.r[k], w_expected, mw.simple_i_rot_flag);
        let i_sph = TWO_THIRDS * model.r[k] * model.r[k];
        if (i_expected - i_sph).abs() > 0.01 * i_sph {
            any_nontrivial_distortion = true;
        }
        assert!(
            (model.aux.w_div_wc[k] - w_expected).abs() <= 1.0e-12,
            "{label}: w_div_wc[{k}] mismatch: got {}, expected {w_expected}",
            model.aux.w_div_wc[k]
        );
        let rel_i =
            (model.aux.i_rot[k] - i_expected).abs() / i_expected.max(1.0);
        assert!(
            rel_i <= 1.0e-13,
            "{label}: i_rot[{k}] must match Paxton 2019 eval_i_rot: got {:.16e}, expected {:.16e} (rel={rel_i:.3e})",
            model.aux.i_rot[k],
            i_expected
        );
        let expected_om = model.j_rot[k] / model.aux.i_rot[k];
        let rel_om = (model.omega[k] - expected_om).abs()
            / expected_om.abs().max(1.0e-30);
        assert!(
            rel_om <= 1.0e-13,
            "{label}: omega[{k}] must equal j_rot / i_rot: rel={rel_om:.3e}"
        );
    }
    assert!(
        any_nontrivial_distortion,
        "{label}: expected at least one zone with >1% Roche i_rot distortion"
    );
}

struct AccretorExtras;

impl StarExtras for AccretorExtras {
    fn other_adjust_mdot(
        &mut self,
        m: &mut StarModel,
        current_mdot: f64,
    ) -> Result<Option<f64>, StarError> {
        let x_ctrl_1 = m.controls.x_ctrl[0];
        let x_ctrl_2 = m.controls.x_ctrl[1];
        let phot_r_rsun = eval_photosphere_info(m).r_rsun;
        m.controls.mix_wind.accreted_material_j =
            x_ctrl_1 * (((m.cgrav[0] * m.mstar) * phot_r_rsun) * R_SUN).sqrt();
        let mdot_add_cgs = flexmesa_math::exp10(x_ctrl_2) * M_SUN / SECYER;
        Ok(Some(current_mdot + mdot_add_cgs))
    }
}

#[test]
fn test_rotating_star_smoke_conserve_angular_momentum() {
    let cons_dir = star_suite_dir().join("conserve_angular_momentum");
    assert!(
        cons_dir.is_dir(),
        "missing conserve_angular_momentum test_suite inputs at {}",
        cons_dir.display()
    );
    let mut cons_bundle = InlistBundle::from_file_with_options(
        &cons_dir.join("inlist_conserve_J_header"),
        InlistLoadOptions {
            allow_unsupported_controls: true,
        },
    )
    .expect("parse conserve_angular_momentum inlist (diagnostic)");
    cons_bundle.job.saved_model_name = String::from("standard_zams.mod");
    let mut default_ex = DefaultStarExtras;
    let mut m_cons =
        init_or_update_model_from_bundle(&cons_bundle, None, &mut default_ex)
            .expect("init conserve_angular_momentum");
    let j_init_cons = total_angular_momentum(&m_cons);
    assert!(j_init_cons > 0.0, "rotating star must have positive J_tot");

    let mut max_rel_j_cons = 0.0_f64;
    for step in 1..=20 {
        let rep = star_evolve_step(&mut m_cons, true, false, true)
            .expect("conserve_angular_momentum step");
        assert!(rep.converged);
        let j_now = total_angular_momentum(&m_cons);
        let rel = (j_now - j_init_cons).abs() / j_init_cons.abs();
        max_rel_j_cons = max_rel_j_cons.max(rel);
        assert!(
            rel < 1.0e-9,
            "conserve_angular_momentum step {step} drifted past 1e-9: rel={rel:.3e}"
        );
    }
    println!(
        "diagnostic conserve_angular_momentum 20 steps: nz={}, log_J={:.12}, max_rel_J_err={:.3e}, age={:.6e} yr, rot_steps={}",
        m_cons.nz,
        flexmesa_math::log10(total_angular_momentum(&m_cons)),
        max_rel_j_cons,
        m_cons.star_age,
        m_cons.aux.step_hist.num_rotation_solver_steps
    );
}

#[test]
fn test_rotating_star_smoke_accreted_material_j() {
    let acc_dir = star_suite_dir().join("accreted_material_j");
    assert!(
        acc_dir.is_dir(),
        "missing accreted_material_j test_suite inputs at {}",
        acc_dir.display()
    );
    let mut acc_bundle = InlistBundle::from_file_with_options(
        &acc_dir.join("inlist_accreted_material_j_header"),
        InlistLoadOptions {
            allow_unsupported_controls: true,
        },
    )
    .expect("parse accreted_material_j inlist (diagnostic)");
    acc_bundle.job.saved_model_name = String::from("standard_zams.mod");
    let mut default_ex = DefaultStarExtras;
    let mut m_acc =
        init_or_update_model_from_bundle(&acc_bundle, None, &mut default_ex)
            .expect("init accreted_material_j");
    assert!(m_acc.controls.mix_wind.premix_omega);
    assert!(m_acc.controls.mix_wind.use_accreted_material_j);

    let mut extras_acc = AccretorExtras;
    let mut max_rel_acc_j = 0.0_f64;
    for step in 1..=10 {
        let j_before = total_angular_momentum(&m_acc);
        let rep = star_evolve_step_with_extras(
            &mut m_acc,
            true,
            false,
            true,
            &mut extras_acc,
        )
        .expect("accreted_material_j step");
        assert!(rep.converged);
        let j_after = total_angular_momentum(&m_acc);
        let expected_dj = m_acc.controls.mix_wind.accreted_material_j
            * m_acc.mstar_dot
            * rep.dt;
        if expected_dj > 1.0e-5 * j_after {
            let am_err =
                (expected_dj - (j_after - j_before)).abs() / expected_dj;
            max_rel_acc_j = max_rel_acc_j.max(am_err);
            assert!(
                am_err < 1.0e-5,
                "accreted_material_j step {step} exceeded 1e-5 accreted-J error: am_err={am_err:.3e}"
            );
        }
    }
    println!(
        "diagnostic accreted_material_j 10 steps: nz={}, M={:.6} Msun, log_J={:.12}, max_rel_acc_J_err={:.3e}, age={:.6} yr, rot_steps={}",
        m_acc.nz,
        m_acc.mstar / M_SUN,
        flexmesa_math::log10(total_angular_momentum(&m_acc)),
        max_rel_acc_j,
        m_acc.star_age,
        m_acc.aux.step_hist.num_rotation_solver_steps
    );
}

#[test]
fn test_other_adjust_mdot_mutation_none_preservation_and_retry_restore() {
    struct MutateWithNoneExtras;
    impl StarExtras for MutateWithNoneExtras {
        fn other_adjust_mdot(
            &mut self,
            m: &mut StarModel,
            _current_mdot: f64,
        ) -> Result<Option<f64>, StarError> {
            m.mstar_dot = 4.56e17;
            m.controls.mix_wind.accreted_material_j = 7.89e18;
            Ok(None)
        }
    }

    struct RetryAndAbandonExtras {
        observed_before_hook: Vec<f64>,
        fail_first_call: bool,
    }
    impl StarExtras for RetryAndAbandonExtras {
        fn other_adjust_mdot(
            &mut self,
            m: &mut StarModel,
            current_mdot: f64,
        ) -> Result<Option<f64>, StarError> {
            self.observed_before_hook
                .push(m.controls.mix_wind.accreted_material_j);
            let call_num = f64::from(
                u32::try_from(self.observed_before_hook.len()).unwrap_or(0),
            );
            m.controls.mix_wind.accreted_material_j = 1.0e18 * call_num;
            Ok(Some(current_mdot + 1.0e17))
        }

        fn other_torque_implicit(
            &mut self,
            _m: &mut StarModel,
        ) -> Result<(), StarError> {
            if self.fail_first_call && self.observed_before_hook.len() == 1 {
                return Err(StarError::ConvergenceFailure(String::from(
                    "synthetic first-try implicit torque failure",
                )));
            }
            Ok(())
        }
    }

    let acc_dir = star_suite_dir().join("accreted_material_j");
    let mut acc_bundle = InlistBundle::from_file_with_options(
        &acc_dir.join("inlist_accreted_material_j_header"),
        InlistLoadOptions {
            allow_unsupported_controls: true,
        },
    )
    .expect("parse accreted_material_j inlist");
    acc_bundle.job.saved_model_name = String::from("standard_zams.mod");
    let mut default_ex = DefaultStarExtras;
    let base =
        init_or_update_model_from_bundle(&acc_bundle, None, &mut default_ex)
            .expect("init accreted_material_j");

    // 1. Returning `Ok(None)` without mutating `m.mstar_dot` preserves
    //    `explicit_step_mdot` (`binary/system_rlof` contract).
    let mut m_none = base.clone();
    let explicit_mdot = -1.23e17;
    m_none.controls.use_other_adjust_mdot = true;
    m_none.mstar_dot = explicit_mdot;
    m_none.controls.mix_wind.accreted_material_j = 3.21e17;
    let rep_none = star_evolve_step_with_extras(
        &mut m_none,
        false,
        false,
        true,
        &mut default_ex,
    )
    .expect("step with DefaultStarExtras");
    assert!(rep_none.converged);
    assert_eq!(m_none.mstar_dot.to_bits(), explicit_mdot.to_bits());
    assert_eq!(
        m_none.controls.mix_wind.accreted_material_j.to_bits(),
        3.21e17_f64.to_bits()
    );

    // 2. Returning `Ok(None)` while mutating `m.mstar_dot` and
    //    `m.controls.mix_wind.accreted_material_j` preserves both mutations.
    let mut m_mut = base.clone();
    m_mut.controls.use_other_adjust_mdot = true;
    m_mut.mstar_dot = explicit_mdot;
    let mut mut_ex = MutateWithNoneExtras;
    let rep_mut = star_evolve_step_with_extras(
        &mut m_mut,
        false,
        false,
        true,
        &mut mut_ex,
    )
    .expect("step with MutateWithNoneExtras");
    assert!(rep_mut.converged);
    assert_eq!(m_mut.mstar_dot.to_bits(), 4.56e17_f64.to_bits());
    assert_eq!(
        m_mut.controls.mix_wind.accreted_material_j.to_bits(),
        7.89e18_f64.to_bits()
    );

    // 3. `PreAttemptScalars` restores `accreted_material_j` on both step
    //    abandonment (`max_number_retries = 0`) and retry (`max_number_retries = 1`).
    let init_acc_j = 0.5e18;
    let mut m_abort = base.clone();
    m_abort.controls.use_other_adjust_mdot = true;
    m_abort.controls.use_other_torque_implicit = true;
    m_abort.controls.max_number_retries = 0;
    m_abort.controls.mix_wind.accreted_material_j = init_acc_j;
    let mut abort_ex = RetryAndAbandonExtras {
        observed_before_hook: Vec::new(),
        fail_first_call: true,
    };
    let err = star_evolve_step_with_extras(
        &mut m_abort,
        false,
        false,
        true,
        &mut abort_ex,
    )
    .expect_err(
        "max_number_retries=0 must abandon step on implicit torque error",
    );
    assert!(
        matches!(
            err,
            StarError::Terminated { ref reason, .. }
                if reason == "max_number_retries"
        ),
        "expected max_number_retries termination, got {err:?}"
    );
    assert_eq!(
        m_abort.controls.mix_wind.accreted_material_j.to_bits(),
        init_acc_j.to_bits()
    );

    let mut m_retry = base;
    m_retry.controls.use_other_adjust_mdot = true;
    m_retry.controls.use_other_torque_implicit = true;
    m_retry.controls.max_number_retries = 2;
    m_retry.controls.ts_stop.retry_limit = 2;
    m_retry.controls.mix_wind.accreted_material_j = init_acc_j;
    let mut retry_ex = RetryAndAbandonExtras {
        observed_before_hook: Vec::new(),
        fail_first_call: true,
    };
    let rep_retry = star_evolve_step_with_extras(
        &mut m_retry,
        false,
        false,
        true,
        &mut retry_ex,
    )
    .expect("retry_limit=1 must recover on second try");
    assert_eq!(rep_retry.retries, 1);
    assert_eq!(
        retry_ex
            .observed_before_hook
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        vec![init_acc_j.to_bits(), init_acc_j.to_bits()]
    );
    assert_eq!(
        m_retry.controls.mix_wind.accreted_material_j.to_bits(),
        2.0e18_f64.to_bits()
    );
}

// ---------------------------------------------------------------------------
// MESA 26.4.1 Fortran Oracle Parity Tests (`mesa_26_4_1_omega_mix_oracles.tsv`)
// ---------------------------------------------------------------------------

type OracleMap = HashMap<(String, String, String, usize, usize), f64>;

fn load_omega_mix_oracles() -> OracleMap {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../benchmarks/fortran_mesa/omega_mix_oracles/mesa_26_4_1_omega_mix_oracles.tsv",
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("failed to read {}: {e}", path.display());
    });
    let mut has_mesa_ver = false;
    let mut has_sdk_ver = false;
    let mut has_compiler = false;
    let mut has_backend = false;
    let mut map = HashMap::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("# mesa_version:") {
            assert!(
                trimmed.contains("26.4.1"),
                "unexpected mesa_version header: {trimmed}"
            );
            has_mesa_ver = true;
            continue;
        }
        if trimmed.starts_with("# mesasdk_version:") {
            assert!(
                trimmed.contains("x86_64-linux-26.6.1"),
                "unexpected mesasdk_version header: {trimmed}"
            );
            has_sdk_ver = true;
            continue;
        }
        if trimmed.starts_with("# compiler:") {
            assert!(
                trimmed.contains("GCC version 15.2.0"),
                "unexpected compiler header: {trimmed}"
            );
            has_compiler = true;
            continue;
        }
        if trimmed.starts_with("# math_backend:") {
            assert!(
                trimmed.contains("CRMATH"),
                "unexpected math_backend header: {trimmed}"
            );
            has_backend = true;
            continue;
        }
        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.starts_with("section\t")
        {
            continue;
        }
        let cols: Vec<&str> = trimmed.split('\t').collect();
        assert_eq!(cols.len(), 7, "unexpected oracle TSV row: {trimmed}");
        let sec = cols[0].to_string();
        let cid = cols[1].to_string();
        let fld = cols[2].to_string();
        let ii: usize = cols[3].parse().expect("parse i");
        let jj: usize = cols[4].parse().expect("parse j");
        let hex = cols[5]
            .strip_prefix("0x")
            .unwrap_or_else(|| panic!("missing 0x prefix: {}", cols[5]));
        let bits = u64::from_str_radix(hex, 16)
            .unwrap_or_else(|e| panic!("invalid hex {}: {e}", cols[5]));
        map.insert((sec, cid, fld, ii, jj), f64::from_bits(bits));
    }
    assert!(has_mesa_ver, "oracle TSV missing # mesa_version header");
    assert!(has_sdk_ver, "oracle TSV missing # mesasdk_version header");
    assert!(has_compiler, "oracle TSV missing # compiler header");
    assert!(has_backend, "oracle TSV missing # math_backend header");
    assert!(
        map.len() >= 4500,
        "expected >= 4500 oracle rows, got {}",
        map.len()
    );
    map
}

fn oracle_val(
    map: &OracleMap,
    sec: &str,
    cid: &str,
    fld: &str,
    i: usize,
    j: usize,
) -> f64 {
    *map.get(&(sec.to_string(), cid.to_string(), fld.to_string(), i, j))
        .unwrap_or_else(|| {
            panic!("missing oracle key ({sec}, {cid}, {fld}, {i}, {j})")
        })
}

fn ordered_f64_key(x: f64) -> u64 {
    let bits = x.to_bits();
    let sign_bit = 1_u64 << 63;
    if (bits & sign_bit) != 0 {
        sign_bit.wrapping_sub(bits & !sign_bit)
    } else {
        sign_bit + bits
    }
}

fn ulp_distance(a: f64, b: f64) -> u64 {
    assert!(
        a.is_finite() && b.is_finite(),
        "non-finite ULP comparison: {a:?} vs {b:?}"
    );
    ordered_f64_key(a).abs_diff(ordered_f64_key(b))
}

fn rel_err(got: f64, want: f64) -> f64 {
    let diff = (got - want).abs();
    let denom = want.abs().max(got.abs());
    if denom == 0.0 {
        0.0
    } else {
        diff / denom
    }
}

#[derive(Default)]
struct ErrorTracker {
    max_ulp: u64,
    max_rel: f64,
    worst_label: String,
    count: usize,
}

impl ErrorTracker {
    fn record(&mut self, label: &str, got: f64, want: f64) {
        let u = ulp_distance(got, want);
        let r = rel_err(got, want);
        self.count += 1;
        if u > self.max_ulp || (u == self.max_ulp && r > self.max_rel) {
            self.max_ulp = u;
            self.max_rel = r;
            self.worst_label = format!(
                "{label} (got={got:.17e} 0x{:016X}, want={want:.17e} 0x{:016X}, ulp={u}, rel={r:.3e})",
                got.to_bits(),
                want.to_bits()
            );
        }
    }
}

fn make_blank_rotating_model(nz: usize, simple_i_rot: bool) -> StarModel {
    let mut ctrl = StarControls {
        rotation_flag: true,
        ..StarControls::default()
    };
    ctrl.mix_wind.simple_i_rot_flag = simple_i_rot;
    ctrl.mix_wind.w_div_wcrit_max = 0.89;
    ctrl.mix_wind.w_div_wcrit_max2 = 0.90;
    ctrl.mix_wind.recalc_mixing_info_each_substep = false;
    ctrl.mix_wind.am_sig_term_limit = 1.0e13;
    ctrl.mix_wind.angular_momentum_error_retry = 1.0e-6;
    ctrl.mix_wind.angular_momentum_error_warn = 5.0e-6;
    let mut m = StarModel::new(nz, vec![1], StarJobControls::default(), ctrl);
    m.model_number = 10;
    m.cgrav.fill(STANDARD_CGRAV);
    m
}

#[test]
fn test_mesa_26_4_1_oracle_eval_i_rot_fp_ft_and_roche_factors() {
    let oracles = load_omega_mix_oracles();
    let nz = 13_usize;

    let mut t_irot = ErrorTracker::default();
    let mut t_dirot_dlnr = ErrorTracker::default();
    let mut t_dirot_dw = ErrorTracker::default();
    let mut t_fp = ErrorTracker::default();
    let mut t_ft = ErrorTracker::default();
    let mut t_dfp_dw = ErrorTracker::default();
    let mut t_dft_dw = ErrorTracker::default();
    let mut t_rpol = ErrorTracker::default();
    let mut t_req = ErrorTracker::default();

    for &(cid, simple_flag) in
        &[("roche_paxton", false), ("roche_simple", true)]
    {
        let mut model = make_blank_rotating_model(nz, simple_flag);
        for k in 0..nz {
            let kf = k + 1;
            model.r[k] =
                oracle_val(&oracles, "eval_i_rot_fp_ft", cid, "r", kf, 0);
            model.aux.w_div_wc[k] = oracle_val(
                &oracles,
                "eval_i_rot_fp_ft",
                cid,
                "w_div_wc",
                kf,
                0,
            );
            model.j_rot[k] = 1.0e18;
            model.m[k] = 4.0 * M_SUN;
            model.m_grav[k] = 4.0 * M_SUN;
            model.rho[k] = 1.0;
        }
        set_rotation_info(&mut model, true);

        for k in 0..nz {
            let kf = k + 1;
            let r_k = model.r[k];
            let w_k = model.aux.w_div_wc[k];

            let want_irot =
                oracle_val(&oracles, "eval_i_rot_fp_ft", cid, "i_rot", kf, 0);
            let want_dirot_dlnr = oracle_val(
                &oracles,
                "eval_i_rot_fp_ft",
                cid,
                "d_i_rot_dlnR",
                kf,
                0,
            );
            let want_dirot_dw = oracle_val(
                &oracles,
                "eval_i_rot_fp_ft",
                cid,
                "d_i_rot_dw_div_wc",
                kf,
                0,
            );
            let want_fp =
                oracle_val(&oracles, "eval_i_rot_fp_ft", cid, "fp_rot", kf, 0);
            let want_ft =
                oracle_val(&oracles, "eval_i_rot_fp_ft", cid, "ft_rot", kf, 0);
            let want_dfp_dw = oracle_val(
                &oracles,
                "eval_i_rot_fp_ft",
                cid,
                "d_fp_dw_div_wc",
                kf,
                0,
            );
            let want_dft_dw = oracle_val(
                &oracles,
                "eval_i_rot_fp_ft",
                cid,
                "d_ft_dw_div_wc",
                kf,
                0,
            );
            let want_rpol =
                oracle_val(&oracles, "eval_i_rot_fp_ft", cid, "r_polar", kf, 0);
            let want_req = oracle_val(
                &oracles,
                "eval_i_rot_fp_ft",
                cid,
                "r_equatorial",
                kf,
                0,
            );

            // Direct `eval_i_rot`, `eval_i_rot_dual`, and `set1_i_rot` must agree bit-for-bit.
            let got_irot_direct = eval_i_rot(r_k, w_k, simple_flag);
            let got_irot_dual = eval_i_rot_dual(r_k, w_k, simple_flag);
            assert_eq!(got_irot_direct.to_bits(), model.aux.i_rot[k].to_bits());
            assert_eq!(
                got_irot_dual.val.to_bits(),
                model.aux.i_rot[k].to_bits()
            );
            assert_eq!(
                got_irot_dual.d1[1].to_bits(),
                model.aux.d_i_rot_dlnr[k].to_bits()
            );
            let mut m_one = model.clone();
            set1_i_rot(&mut m_one, k, r_k);
            assert_eq!(
                m_one.aux.i_rot[k].to_bits(),
                model.aux.i_rot[k].to_bits()
            );

            t_irot.record(
                &format!("{cid}.i_rot[{kf}]"),
                model.aux.i_rot[k],
                want_irot,
            );
            t_dirot_dlnr.record(
                &format!("{cid}.d_i_rot_dlnR[{kf}]"),
                model.aux.d_i_rot_dlnr[k],
                want_dirot_dlnr,
            );
            t_dirot_dw.record(
                &format!("{cid}.d_i_rot_dw_div_wc[{kf}]"),
                got_irot_dual.d1[0],
                want_dirot_dw,
            );

            let (fp_direct, ft_direct) = eval_fp_ft_single(w_k);
            let (fp_dual, ft_dual) = eval_fp_ft_single_dual(w_k);
            assert_eq!(fp_direct.to_bits(), model.aux.fp_rot[k].to_bits());
            assert_eq!(ft_direct.to_bits(), model.aux.ft_rot[k].to_bits());
            t_fp.record(&format!("{cid}.fp_rot[{kf}]"), fp_direct, want_fp);
            t_ft.record(&format!("{cid}.ft_rot[{kf}]"), ft_direct, want_ft);
            if !simple_flag {
                // `roche_paxton` ran with `w_div_wc_flag = .true.` in MESA.
                t_dfp_dw.record(
                    &format!("{cid}.d_fp_dw_div_wc[{kf}]"),
                    fp_dual.d1[0],
                    want_dfp_dw,
                );
                t_dft_dw.record(
                    &format!("{cid}.d_ft_dw_div_wc[{kf}]"),
                    ft_dual.d1[0],
                    want_dft_dw,
                );
            }

            let (got_req, got_rpol) = equatorial_and_polar_r_cm(&model, k);
            t_req.record(
                &format!("{cid}.r_equatorial[{kf}]"),
                got_req,
                want_req,
            );
            t_rpol.record(&format!("{cid}.r_polar[{kf}]"), got_rpol, want_rpol);
        }
    }

    // Section 2: `roche_a` and `roche_c`
    let mut t_ac = ErrorTracker::default();
    for kf in 1..=12_usize {
        let w = oracle_val(&oracles, "roche_factors", "fabry2022", "w", kf, 0);
        let want_a = oracle_val(
            &oracles,
            "roche_factors",
            "fabry2022",
            "roche_A",
            kf,
            0,
        );
        let want_c = oracle_val(
            &oracles,
            "roche_factors",
            "fabry2022",
            "roche_C",
            kf,
            0,
        );
        let w2 = flexmesa_math::pow2(w);
        let w4 = flexmesa_math::pow4(w);
        let w6 = flexmesa_math::pow6(w);
        let w_log =
            flexmesa_math::log(1.0 - flexmesa_math::pow(w, LOG_TERM_POWER));
        t_ac.record(&format!("roche_A[{kf}]"), roche_a(w4, w6, w_log), want_a);
        t_ac.record(
            &format!("roche_C[{kf}]"),
            roche_c(w2, w4, w6, w_log),
            want_c,
        );
    }

    println!(
        "ORACLE eval_i_rot_fp_ft & roche_factors:\n  \
         i_rot          : max_ulp={} max_rel={:.3e} [{}]\n  \
         d_i_rot_dlnR   : max_ulp={} max_rel={:.3e} [{}]\n  \
         d_i_rot_dw     : max_ulp={} max_rel={:.3e} [{}]\n  \
         fp_rot         : max_ulp={} max_rel={:.3e} [{}]\n  \
         ft_rot         : max_ulp={} max_rel={:.3e} [{}]\n  \
         d_fp_dw        : max_ulp={} max_rel={:.3e} [{}]\n  \
         d_ft_dw        : max_ulp={} max_rel={:.3e} [{}]\n  \
         r_equatorial   : max_ulp={} max_rel={:.3e} [{}]\n  \
         r_polar        : max_ulp={} max_rel={:.3e} [{}]\n  \
         roche_A/C      : max_ulp={} max_rel={:.3e} [{}]",
        t_irot.max_ulp,
        t_irot.max_rel,
        t_irot.worst_label,
        t_dirot_dlnr.max_ulp,
        t_dirot_dlnr.max_rel,
        t_dirot_dlnr.worst_label,
        t_dirot_dw.max_ulp,
        t_dirot_dw.max_rel,
        t_dirot_dw.worst_label,
        t_fp.max_ulp,
        t_fp.max_rel,
        t_fp.worst_label,
        t_ft.max_ulp,
        t_ft.max_rel,
        t_ft.worst_label,
        t_dfp_dw.max_ulp,
        t_dfp_dw.max_rel,
        t_dfp_dw.worst_label,
        t_dft_dw.max_ulp,
        t_dft_dw.max_rel,
        t_dft_dw.worst_label,
        t_req.max_ulp,
        t_req.max_rel,
        t_req.worst_label,
        t_rpol.max_ulp,
        t_rpol.max_rel,
        t_rpol.worst_label,
        t_ac.max_ulp,
        t_ac.max_rel,
        t_ac.worst_label,
    );

    assert_eq!(
        t_irot.max_ulp, 0,
        "i_rot ULP mismatch: {}",
        t_irot.worst_label
    );
    assert_eq!(
        t_dirot_dlnr.max_ulp, 0,
        "d_i_rot_dlnR ULP mismatch: {}",
        t_dirot_dlnr.worst_label
    );
    assert_eq!(
        t_dirot_dw.max_ulp, 0,
        "d_i_rot_dw_div_wc ULP mismatch: {}",
        t_dirot_dw.worst_label
    );
    assert_eq!(t_fp.max_ulp, 0, "fp_rot ULP mismatch: {}", t_fp.worst_label);
    assert_eq!(t_ft.max_ulp, 0, "ft_rot ULP mismatch: {}", t_ft.worst_label);
    assert_eq!(
        t_dfp_dw.max_ulp, 0,
        "d_fp_dw ULP mismatch: {}",
        t_dfp_dw.worst_label
    );
    assert_eq!(
        t_dft_dw.max_ulp, 0,
        "d_ft_dw ULP mismatch: {}",
        t_dft_dw.worst_label
    );
    assert_eq!(
        t_req.max_ulp, 0,
        "r_equatorial ULP mismatch: {}",
        t_req.worst_label
    );
    assert_eq!(
        t_rpol.max_ulp, 0,
        "r_polar ULP mismatch: {}",
        t_rpol.worst_label
    );
    assert_eq!(
        t_ac.max_ulp, 0,
        "roche_A/C ULP mismatch: {}",
        t_ac.worst_label
    );
}

#[test]
fn test_mesa_26_4_1_oracle_w_div_w_roche_and_rotation_sigmas() {
    let oracles = load_omega_mix_oracles();

    // Section 3: w_div_w_roche_omega and w_div_w_roche_jrot
    let mut t_w_om = ErrorTracker::default();
    let mut t_w_jr = ErrorTracker::default();
    for &(cid, w_flag) in &[("hard_cap", false), ("sigmoid_cap", true)] {
        let rpsi = oracle_val(&oracles, "w_div_w_roche", cid, "rpsi", 0, 0);
        let m_phi = oracle_val(&oracles, "w_div_w_roche", cid, "m_phi", 0, 0);
        let cgrav = oracle_val(&oracles, "w_div_w_roche", cid, "cgrav", 0, 0);
        let max_w = oracle_val(&oracles, "w_div_w_roche", cid, "max_w", 0, 0);
        let max_w2 = oracle_val(&oracles, "w_div_w_roche", cid, "max_w2", 0, 0);

        for kf in 1..=10_usize {
            let om_in =
                oracle_val(&oracles, "w_div_w_roche", cid, "omega_in", kf, 0);
            let want_om = oracle_val(
                &oracles,
                "w_div_w_roche",
                cid,
                "w_from_omega",
                kf,
                0,
            );
            let got_om = w_div_w_roche_omega(
                rpsi, m_phi, om_in, cgrav, max_w, max_w2, w_flag,
            );
            t_w_om.record(
                &format!("{cid}.w_from_omega[{kf}]"),
                got_om,
                want_om,
            );

            let jr_in =
                oracle_val(&oracles, "w_div_w_roche", cid, "jrot_in", kf, 0);
            let want_jr = oracle_val(
                &oracles,
                "w_div_w_roche",
                cid,
                "w_from_jrot",
                kf,
                0,
            );
            let got_jr = w_div_w_roche_jrot(
                rpsi, m_phi, jr_in, cgrav, max_w, max_w2, w_flag,
            );
            t_w_jr.record(&format!("{cid}.w_from_jrot[{kf}]"), got_jr, want_jr);
        }
    }

    // Section 4: get_rotation_sigmas
    let mut t_sig_j = ErrorTracker::default();
    let mut t_sig_om = ErrorTracker::default();
    let nz = 12_usize;
    for &cid in &["sigmas_normal", "sigmas_capped"] {
        let mut model = make_blank_rotating_model(nz, false);
        model.xmstar =
            oracle_val(&oracles, "rotation_sigmas", cid, "xmstar", 0, 0);
        model.mstar = model.xmstar;
        let dt = oracle_val(&oracles, "rotation_sigmas", cid, "dt", 0, 0);
        model.controls.mix_wind.am_sig_term_limit = oracle_val(
            &oracles,
            "rotation_sigmas",
            cid,
            "am_sig_term_limit",
            0,
            0,
        );
        for k in 0..nz {
            let kf = k + 1;
            model.r[k] =
                oracle_val(&oracles, "rotation_sigmas", cid, "r", kf, 0);
            model.rho[k] =
                oracle_val(&oracles, "rotation_sigmas", cid, "rho", kf, 0);
            model.dm[k] =
                oracle_val(&oracles, "rotation_sigmas", cid, "dm", kf, 0);
            model.dq[k] =
                oracle_val(&oracles, "rotation_sigmas", cid, "dq", kf, 0);
            model.aux.am_nu_omega[k] = oracle_val(
                &oracles,
                "rotation_sigmas",
                cid,
                "am_nu_omega",
                kf,
                0,
            );
            model.aux.am_nu_j[k] =
                oracle_val(&oracles, "rotation_sigmas", cid, "am_nu_j", kf, 0);
        }
        get_rotation_sigmas(&mut model, 0, nz, dt);
        for k in 0..nz {
            let kf = k + 1;
            let want_sj =
                oracle_val(&oracles, "rotation_sigmas", cid, "am_sig_j", kf, 0);
            let want_so = oracle_val(
                &oracles,
                "rotation_sigmas",
                cid,
                "am_sig_omega",
                kf,
                0,
            );
            t_sig_j.record(
                &format!("{cid}.am_sig_j[{kf}]"),
                model.aux.am_sig_j[k],
                want_sj,
            );
            t_sig_om.record(
                &format!("{cid}.am_sig_omega[{kf}]"),
                model.aux.am_sig_omega[k],
                want_so,
            );
        }
    }

    println!(
        "ORACLE w_div_w_roche & rotation_sigmas:\n  \
         w_from_omega: max_ulp={} max_rel={:.3e} [{}]\n  \
         w_from_jrot : max_ulp={} max_rel={:.3e} [{}]\n  \
         am_sig_j    : max_ulp={} max_rel={:.3e} [{}]\n  \
         am_sig_omega: max_ulp={} max_rel={:.3e} [{}]",
        t_w_om.max_ulp,
        t_w_om.max_rel,
        t_w_om.worst_label,
        t_w_jr.max_ulp,
        t_w_jr.max_rel,
        t_w_jr.worst_label,
        t_sig_j.max_ulp,
        t_sig_j.max_rel,
        t_sig_j.worst_label,
        t_sig_om.max_ulp,
        t_sig_om.max_rel,
        t_sig_om.worst_label,
    );

    assert_eq!(
        t_w_om.max_ulp, 0,
        "w_div_w_roche_omega ULP mismatch: {}",
        t_w_om.worst_label
    );
    assert_eq!(
        t_w_jr.max_ulp, 0,
        "w_div_w_roche_jrot ULP mismatch: {}",
        t_w_jr.worst_label
    );
    assert_eq!(
        t_sig_j.max_ulp, 0,
        "am_sig_j ULP mismatch: {}",
        t_sig_j.worst_label
    );
    assert_eq!(
        t_sig_om.max_ulp, 0,
        "am_sig_omega ULP mismatch: {}",
        t_sig_om.worst_label
    );
}

#[test]
fn test_mesa_26_4_1_oracle_do_solve_omega_mix_all_regimes() {
    // Mirror the exact Fortran hook modules from `gen_omega_mix_oracles.f90`.
    struct OracleExplicitJdot;
    impl StarExtras for OracleExplicitJdot {
        fn apply_other_torque(
            &mut self,
            m: &mut StarModel,
        ) -> Result<(), StarError> {
            for k in 0..m.nz {
                let kf = f64::from(u32::try_from(k).unwrap_or(0));
                m.aux.extra_jdot[k] = -2.0e6 * (1.0 - 0.025 * kf);
                m.aux.extra_omegadot[k] = 0.0;
            }
            Ok(())
        }
    }

    struct OracleExplicitOmegadot;
    impl StarExtras for OracleExplicitOmegadot {
        fn apply_other_torque(
            &mut self,
            m: &mut StarModel,
        ) -> Result<(), StarError> {
            for k in 0..m.nz {
                let kf = f64::from(u32::try_from(k).unwrap_or(0));
                m.aux.extra_jdot[k] = 0.0;
                m.aux.extra_omegadot[k] = -1.5e-16 * (1.0 + 0.08 * kf);
            }
            Ok(())
        }
    }

    struct OracleImplicitTridiag {
        calls: usize,
    }
    impl StarExtras for OracleImplicitTridiag {
        fn other_torque_implicit(
            &mut self,
            m: &mut StarModel,
        ) -> Result<(), StarError> {
            const OM_EQ: f64 = 3.0e-5;
            const C_J_DRAG: f64 = 2.0e-11;
            const C_J_CPL: f64 = 5.0e-12;
            const C_OM_DRAG: f64 = 1.5e-11;
            const C_OM_CPL: f64 = 4.0e-12;
            self.calls += 1;
            let nz = m.nz;
            m.aux.extra_jdot[..nz].fill(0.0);
            m.aux.extra_omegadot[..nz].fill(0.0);
            m.aux.d_extra_jdot_domega_m1[..nz].fill(0.0);
            m.aux.d_extra_jdot_domega_00[..nz].fill(0.0);
            m.aux.d_extra_jdot_domega_p1[..nz].fill(0.0);
            m.aux.d_extra_omegadot_domega_m1[..nz].fill(0.0);
            m.aux.d_extra_omegadot_domega_00[..nz].fill(0.0);
            m.aux.d_extra_omegadot_domega_p1[..nz].fill(0.0);

            for k in 1..nz {
                let dw = m.omega[k] - OM_EQ;
                let irot_k = m.aux.i_rot[k];

                m.aux.extra_jdot[k] =
                    -C_J_DRAG * irot_k * (dw + 1.0e8 * dw * dw * dw);
                m.aux.d_extra_jdot_domega_00[k] =
                    -C_J_DRAG * irot_k * (1.0 + 3.0e8 * dw * dw);

                m.aux.extra_jdot[k] +=
                    C_J_CPL * irot_k * (m.omega[k - 1] - m.omega[k]);
                m.aux.d_extra_jdot_domega_m1[k] = C_J_CPL * irot_k;
                m.aux.d_extra_jdot_domega_00[k] -= C_J_CPL * irot_k;
                if k + 1 < nz {
                    m.aux.extra_jdot[k] +=
                        C_J_CPL * irot_k * (m.omega[k + 1] - m.omega[k]);
                    m.aux.d_extra_jdot_domega_p1[k] = C_J_CPL * irot_k;
                    m.aux.d_extra_jdot_domega_00[k] -= C_J_CPL * irot_k;
                }

                m.aux.extra_omegadot[k] = -C_OM_DRAG * (dw + 2.0e4 * dw * dw);
                m.aux.d_extra_omegadot_domega_00[k] =
                    -C_OM_DRAG * (1.0 + 4.0e4 * dw);

                m.aux.extra_omegadot[k] +=
                    C_OM_CPL * (m.omega[k - 1] - m.omega[k]);
                m.aux.d_extra_omegadot_domega_m1[k] = C_OM_CPL;
                m.aux.d_extra_omegadot_domega_00[k] -= C_OM_CPL;
                if k + 1 < nz {
                    m.aux.extra_omegadot[k] +=
                        C_OM_CPL * (m.omega[k + 1] - m.omega[k]);
                    m.aux.d_extra_omegadot_domega_p1[k] = C_OM_CPL;
                    m.aux.d_extra_omegadot_domega_00[k] -= C_OM_CPL;
                }
            }
            Ok(())
        }
    }

    struct OracleImplicitNonconverging {
        calls: usize,
    }
    impl StarExtras for OracleImplicitNonconverging {
        fn other_torque_implicit(
            &mut self,
            m: &mut StarModel,
        ) -> Result<(), StarError> {
            self.calls += 1;
            let nz = m.nz;
            for k in 0..nz {
                m.aux.extra_jdot[k] = 0.0;
                m.aux.extra_omegadot[k] = -1.0e-5 * (m.omega[k] - 1.0e-5);
                m.aux.d_extra_jdot_domega_m1[k] = 0.0;
                m.aux.d_extra_jdot_domega_00[k] = 0.0;
                m.aux.d_extra_jdot_domega_p1[k] = 0.0;
                m.aux.d_extra_omegadot_domega_m1[k] = 0.0;
                m.aux.d_extra_omegadot_domega_00[k] = 0.0;
                m.aux.d_extra_omegadot_domega_p1[k] = 0.0;
            }
            Ok(())
        }
    }

    let oracles = load_omega_mix_oracles();
    let nz = 24_usize;
    let cases = [
        ("diffusive_omega_shear", false),
        ("diffusive_pure_j", false),
        ("diffusive_mixed_omega_j", false),
        ("diffusive_stiff_1e13", false),
        ("explicit_extra_jdot", false),
        ("explicit_extra_omegadot", false),
        ("implicit_torque_tridiag", false),
        ("adaptive_substepping", false),
        ("max_iters_retry", false),
        ("conservation_retry_triggered", false),
        ("conservation_retry_passed", false),
        ("simple_i_rot_shear", true),
    ];

    let mut t_irot = ErrorTracker::default();
    let mut t_omega = ErrorTracker::default();
    let mut t_jrot = ErrorTracker::default();
    let mut t_sig_om = ErrorTracker::default();
    let mut t_sig_j = ErrorTracker::default();
    let mut t_jtot = ErrorTracker::default();

    for &(cid, simple_flag) in &cases {
        let mut model = make_blank_rotating_model(nz, simple_flag);
        model.xmstar =
            oracle_val(&oracles, "solve_omega_mix", cid, "xmstar", 0, 0);
        model.mstar = model.xmstar;
        let dt_total =
            oracle_val(&oracles, "solve_omega_mix", cid, "dt_total", 0, 0);
        model.controls.mix_wind.am_sig_term_limit = oracle_val(
            &oracles,
            "solve_omega_mix",
            cid,
            "am_sig_term_limit",
            0,
            0,
        );
        model.controls.mix_wind.angular_momentum_error_retry = oracle_val(
            &oracles,
            "solve_omega_mix",
            cid,
            "angular_momentum_error_retry",
            0,
            0,
        );
        for k in 0..nz {
            let kf = k + 1;
            model.r[k] =
                oracle_val(&oracles, "solve_omega_mix", cid, "r", kf, 0);
            model.m[k] =
                oracle_val(&oracles, "solve_omega_mix", cid, "m", kf, 0);
            model.m_grav[k] = model.m[k];
            model.dm[k] =
                oracle_val(&oracles, "solve_omega_mix", cid, "dm", kf, 0);
            model.dq[k] =
                oracle_val(&oracles, "solve_omega_mix", cid, "dq", kf, 0);
            model.dm_bar[k] =
                oracle_val(&oracles, "solve_omega_mix", cid, "dm_bar", kf, 0);
            model.rho[k] =
                oracle_val(&oracles, "solve_omega_mix", cid, "rho", kf, 0);
            model.j_rot[k] = oracle_val(
                &oracles,
                "solve_omega_mix",
                cid,
                "j_rot_init",
                kf,
                0,
            );
            model.aux.am_nu_omega[k] = oracle_val(
                &oracles,
                "solve_omega_mix",
                cid,
                "am_nu_omega",
                kf,
                0,
            );
            model.aux.am_nu_j[k] =
                oracle_val(&oracles, "solve_omega_mix", cid, "am_nu_j", kf, 0);
        }

        let want_j0 =
            oracle_val(&oracles, "solve_omega_mix", cid, "J_tot0", 0, 0);
        let got_j0 = total_angular_momentum(&model);
        t_jtot.record(&format!("{cid}.J_tot0"), got_j0, want_j0);

        model.aux.step_hist.num_rotation_solver_steps = 0;
        let (outcome, got_imp_calls) = match cid {
            "explicit_extra_jdot" => {
                model.controls.use_other_torque = true;
                let mut ex = OracleExplicitJdot;
                (
                    do_solve_omega_mix_with_extras(
                        &mut model, dt_total, &mut ex,
                    )
                    .expect(cid),
                    0_usize,
                )
            }
            "explicit_extra_omegadot" => {
                model.controls.use_other_torque = true;
                let mut ex = OracleExplicitOmegadot;
                (
                    do_solve_omega_mix_with_extras(
                        &mut model, dt_total, &mut ex,
                    )
                    .expect(cid),
                    0_usize,
                )
            }
            "implicit_torque_tridiag" => {
                model.controls.use_other_torque_implicit = true;
                let mut ex = OracleImplicitTridiag { calls: 0 };
                let out = do_solve_omega_mix_with_extras(
                    &mut model, dt_total, &mut ex,
                )
                .expect(cid);
                (out, ex.calls)
            }
            "max_iters_retry" => {
                model.controls.use_other_torque_implicit = true;
                let mut ex = OracleImplicitNonconverging { calls: 0 };
                let out = do_solve_omega_mix_with_extras(
                    &mut model, dt_total, &mut ex,
                )
                .expect(cid);
                (out, ex.calls)
            }
            _ => (
                do_solve_omega_mix(&mut model, dt_total).expect(cid),
                0_usize,
            ),
        };

        let want_outcome =
            oracle_val(&oracles, "solve_omega_mix", cid, "outcome", 0, 0);
        let got_outcome_code = match outcome {
            OmegaMixOutcome::KeepGoing => 0.0,
            OmegaMixOutcome::Retry(_) => 2.0,
            OmegaMixOutcome::Terminate(_) => 3.0,
        };
        assert_eq!(
            got_outcome_code, want_outcome,
            "{cid}: outcome mismatch (got {outcome:?}, want code {want_outcome})"
        );

        let want_steps =
            oracle_val(&oracles, "solve_omega_mix", cid, "steps_used", 0, 0);
        assert_eq!(
            f64::from(
                u32::try_from(model.aux.step_hist.num_rotation_solver_steps)
                    .unwrap_or(0)
            ),
            want_steps,
            "{cid}: steps_used mismatch"
        );

        let want_imp = oracle_val(
            &oracles,
            "solve_omega_mix",
            cid,
            "implicit_calls",
            0,
            0,
        );
        assert_eq!(
            f64::from(u32::try_from(got_imp_calls).unwrap_or(0)),
            want_imp,
            "{cid}: implicit_calls mismatch"
        );

        let want_j1 =
            oracle_val(&oracles, "solve_omega_mix", cid, "J_tot1", 0, 0);
        let got_j1 = total_angular_momentum(&model);
        t_jtot.record(&format!("{cid}.J_tot1"), got_j1, want_j1);

        let mut case_omega = ErrorTracker::default();
        for k in 0..nz {
            let kf = k + 1;
            let want_ir = oracle_val(
                &oracles,
                "solve_omega_mix",
                cid,
                "i_rot_final",
                kf,
                0,
            );
            let want_om = oracle_val(
                &oracles,
                "solve_omega_mix",
                cid,
                "omega_final",
                kf,
                0,
            );
            let want_jr = oracle_val(
                &oracles,
                "solve_omega_mix",
                cid,
                "j_rot_final",
                kf,
                0,
            );
            let want_so = oracle_val(
                &oracles,
                "solve_omega_mix",
                cid,
                "am_sig_omega",
                kf,
                0,
            );
            let want_sj =
                oracle_val(&oracles, "solve_omega_mix", cid, "am_sig_j", kf, 0);

            t_irot.record(
                &format!("{cid}.i_rot_final[{kf}]"),
                model.aux.i_rot[k],
                want_ir,
            );
            t_omega.record(
                &format!("{cid}.omega_final[{kf}]"),
                model.omega[k],
                want_om,
            );
            case_omega.record(
                &format!("{cid}.omega_final[{kf}]"),
                model.omega[k],
                want_om,
            );
            t_jrot.record(
                &format!("{cid}.j_rot_final[{kf}]"),
                model.j_rot[k],
                want_jr,
            );
            t_sig_om.record(
                &format!("{cid}.am_sig_omega[{kf}]"),
                model.aux.am_sig_omega[k],
                want_so,
            );
            t_sig_j.record(
                &format!("{cid}.am_sig_j[{kf}]"),
                model.aux.am_sig_j[k],
                want_sj,
            );
        }
        println!(
            "  case {cid:<28}: outcome={got_outcome_code:.0} steps={:<2} imp={:<2} max_omega_ulp={} max_omega_rel={:.3e}",
            model.aux.step_hist.num_rotation_solver_steps,
            got_imp_calls,
            case_omega.max_ulp,
            case_omega.max_rel
        );
    }

    println!(
        "ORACLE solve_omega_mix summary (12 cases, 24 zones each):\n  \
         i_rot_final : max_ulp={} max_rel={:.3e} [{}]\n  \
         omega_final : max_ulp={} max_rel={:.3e} [{}]\n  \
         j_rot_final : max_ulp={} max_rel={:.3e} [{}]\n  \
         am_sig_omega: max_ulp={} max_rel={:.3e} [{}]\n  \
         am_sig_j    : max_ulp={} max_rel={:.3e} [{}]\n  \
         J_tot0/1    : max_ulp={} max_rel={:.3e} [{}]",
        t_irot.max_ulp,
        t_irot.max_rel,
        t_irot.worst_label,
        t_omega.max_ulp,
        t_omega.max_rel,
        t_omega.worst_label,
        t_jrot.max_ulp,
        t_jrot.max_rel,
        t_jrot.worst_label,
        t_sig_om.max_ulp,
        t_sig_om.max_rel,
        t_sig_om.worst_label,
        t_sig_j.max_ulp,
        t_sig_j.max_rel,
        t_sig_j.worst_label,
        t_jtot.max_ulp,
        t_jtot.max_rel,
        t_jtot.worst_label,
    );

    assert_eq!(
        t_irot.max_ulp, 0,
        "i_rot_final ULP mismatch: {}",
        t_irot.worst_label
    );
    assert_eq!(
        t_sig_om.max_ulp, 0,
        "am_sig_omega ULP mismatch: {}",
        t_sig_om.worst_label
    );
    assert_eq!(
        t_sig_j.max_ulp, 0,
        "am_sig_j ULP mismatch: {}",
        t_sig_j.worst_label
    );
    assert_eq!(
        t_omega.max_ulp, 0,
        "omega_final ULP mismatch: {}",
        t_omega.worst_label
    );
    assert_eq!(
        t_jrot.max_ulp, 0,
        "j_rot_final ULP mismatch: {}",
        t_jrot.worst_label
    );
    assert_eq!(
        t_jtot.max_ulp, 0,
        "J_tot ULP mismatch: {}",
        t_jtot.worst_label
    );
}
