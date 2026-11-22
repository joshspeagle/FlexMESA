// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration and numerical-oracle tests for Issue `#158`
//! (`star/private/solve_omega_mix.f90` `Quad` port + `#153` rotation remap
//! handoff verification).

#![forbid(unsafe_code)]

use std::path::PathBuf;

use flexmesa_const::{M_SUN, R_SUN, SECYER, TWO_THIRDS};
use flexmesa_star::{
    BinaryOtherTorqueHandle, BinaryOtherTorqueHook, DefaultStarExtras,
    InlistBundle, InlistLoadOptions, MeshFunctionConfig, OmegaMixOutcome,
    PrimaryStateBuffer, StarControls, StarError, StarExtras, StarJobControls,
    StarModel, do_adjust_mesh, do_solve_omega_mix,
    do_solve_omega_mix_with_extras, eval_i_rot, eval_photosphere_info,
    get_rotation_sigmas, init_or_update_model_from_bundle, load_zams_model,
    remesh_split_merge, set_hydro_vars, set_rotation_info, solve_omega_mix,
    star_evolve_step, star_evolve_step_with_extras, total_angular_momentum,
    update_rotation_distortion_factors, w_div_w_roche_jrot,
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
    update_rotation_distortion_factors(&mut model);
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

    let outcome = do_solve_omega_mix(&mut model, dt).expect("do_solve_omega_mix");
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
        .map(|k| non_rot.dm_bar[k] * TWO_THIRDS * non_rot.r[k] * non_rot.r[k] * non_rot.omega[k])
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

    let outcome = do_solve_omega_mix(&mut model, dt).expect("do_solve_omega_mix");
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
    with_recalc.controls.mix_wind.recalc_mixing_info_each_substep = true;

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
        fn apply_other_torque(&mut self, m: &mut StarModel) -> Result<(), StarError> {
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
                m.aux.extra_omegadot[k] = -self.lambda * (m.omega[k] - self.target);
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
    .expect_err("step must fail on implicit torque retry with max_number_retries=0");
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
        let i_expected = eval_i_rot(model.r[k], w_expected, mw.simple_i_rot_flag);
        let i_sph = TWO_THIRDS * model.r[k] * model.r[k];
        if (i_expected - i_sph).abs() > 0.01 * i_sph {
            any_nontrivial_distortion = true;
        }
        assert!(
            (model.aux.w_div_wc[k] - w_expected).abs() <= 1.0e-12,
            "{label}: w_div_wc[{k}] mismatch: got {}, expected {w_expected}",
            model.aux.w_div_wc[k]
        );
        let rel_i = (model.aux.i_rot[k] - i_expected).abs() / i_expected.max(1.0);
        assert!(
            rel_i <= 1.0e-13,
            "{label}: i_rot[{k}] must match Paxton 2019 eval_i_rot: got {:.16e}, expected {:.16e} (rel={rel_i:.3e})",
            model.aux.i_rot[k],
            i_expected
        );
        let expected_om = model.j_rot[k] / model.aux.i_rot[k];
        let rel_om =
            (model.omega[k] - expected_om).abs() / expected_om.abs().max(1.0e-30);
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

struct AccretorExtras {
    mdot_add_cgs: f64,
}

impl StarExtras for AccretorExtras {
    fn other_adjust_mdot(
        &self,
        _m: &StarModel,
        current_mdot: f64,
    ) -> Result<Option<f64>, StarError> {
        Ok(Some(current_mdot + self.mdot_add_cgs))
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

    let x_ctrl_1 = m_acc.controls.x_ctrl[0];
    let x_ctrl_2 = m_acc.controls.x_ctrl[1];
    let mut extras_acc = AccretorExtras {
        mdot_add_cgs: flexmesa_math::exp10(x_ctrl_2) * M_SUN / SECYER,
    };
    let mut max_rel_acc_j = 0.0_f64;
    for step in 1..=10 {
        let j_before = total_angular_momentum(&m_acc);
        let phot_r_rsun = eval_photosphere_info(&m_acc).r_rsun;
        m_acc.controls.mix_wind.accreted_material_j = x_ctrl_1
            * (m_acc.cgrav[0] * m_acc.mstar * phot_r_rsun * R_SUN).sqrt();
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
        let expected_dj =
            m_acc.controls.mix_wind.accreted_material_j * m_acc.mstar_dot * rep.dt;
        if expected_dj > 1.0e-5 * j_after {
            let am_err = (expected_dj - (j_after - j_before)).abs() / expected_dj;
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
