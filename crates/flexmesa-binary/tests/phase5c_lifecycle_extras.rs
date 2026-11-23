// SPDX-License-Identifier: LGPL-3.0-or-later
//! Unit and integration tests for Issue `#137` coupled binary step-lifecycle
//! parity (`xa_sub_xa_start`, `1e-12` species normalization guard, symmetric
//! Strang-split `premix_omega`, `am_d_mix_factor == 0` transport,
//! `op_split_burn` + `save_start_values` ordering, and `BinaryExtras` custom
//! hooks).

#![forbid(unsafe_code)]

use flexmesa_binary::{
    BinaryExtras, BinaryExtrasAction, BinaryInlistControls, BinarySystem,
};
use flexmesa_const::{MSUN, SECYER};
use flexmesa_star::{
    StarControls, StarError, StarJobControls, StarModel,
    apply_step_pre_physics, do_struct_burn_mix,
    load_zams_model, set_hydro_vars,
    solve_henyey_step, star_evolve_step, sync_post_henyey_rotation,
    update_rotation_distortion_factors,
};

#[test]
fn test_default_am_d_mix_zero_omega_transport_and_gold2_chem_increment_convergence()
 {
    let mut ctrl = StarControls {
        initial_mass: 4.0,
        rotation_flag: true,
        use_gold2_tolerances: true,
        zbase: 0.02,
        ..StarControls::default()
    };
    ctrl.mix_wind.premix_omega = false;
    ctrl.mix_wind.am_nu_factor = 1.0;
    ctrl.mix_wind.am_d_mix_factor = 0.0;
    let mut star =
        load_zams_model(ctrl, StarJobControls::default()).expect("4 Msun ZAMS");
    let nz = star.nz;
    let denom =
        f64::from(u32::try_from(nz.saturating_sub(1).max(1)).unwrap_or(1));
    for (k, om) in star.omega.iter_mut().enumerate().take(nz) {
        let frac = f64::from(u32::try_from(k).unwrap_or(0)) / denom;
        *om = 1.0e-5 * (1.0 + 0.5 * frac);
    }
    update_rotation_distortion_factors(&mut star);
    set_hydro_vars(&mut star, false, false).expect("set_hydro_vars");
    star.d_mix.fill(1.0e16);
    star.mixing_type
        .fill(flexmesa_const::MixingType::ConvectiveMixing);
    let shear_before = (star.omega[0] - star.omega[nz - 1]).abs();
    sync_post_henyey_rotation(&mut star, 1.0e4 * SECYER);
    let shear_after = (star.omega[0] - star.omega[nz - 1]).abs();
    assert!(
        shear_after < 0.1 * shear_before,
        "am_d_mix_factor=0 must still transport omega: {shear_before} -> {shear_after}"
    );

    // Equilibrate the newly spun-up 4 Msun ZAMS star onto hydrostatic/thermal
    // balance at its ZAMS timestep before injecting the stiff 50 s core test.
    star.pack_xh();
    star.save_start_values().expect("save_start_values");
    solve_henyey_step(&mut star, false, true)
        .expect("equilibrate spun-up ZAMS");

    // Stiff gold2 convective core (dt * sig / dm = 1e13 at dt = 50 s) with
    // 1-ULP start-of-step normalization jitter across core cells.
    star.pack_xh();
    star.save_start_values().expect("save_start_values");
    star.dt = 50.0;
    let k_core = nz.saturating_sub(4);
    assert!(k_core >= 2 && star.species > 0);
    for k in k_core..nz {
        star.sig[k] = 1.0e13 * star.dm[k] / star.dt;
        if k % 2 == 1 {
            let idx = k * star.species;
            star.xa_start[idx] += f64::EPSILON * star.xa_start[idx];
            star.xa[idx] = star.xa_start[idx];
        }
    }
    let rep = solve_henyey_step(&mut star, true, true)
        .expect("gold2 stiff chem step");
    assert!(
        rep.converged,
        "gold2 stiff chem failed to converge: {rep:?}"
    );
    assert!(
        rep.max_residual <= star.controls.gold2_tol_max_residual1,
        "expected gold2 stage-1 residual convergence, got {rep:?}"
    );
    for k in k_core..nz {
        let sum_x: f64 = star.cell_xa(k).iter().sum();
        assert!(
            (sum_x - 1.0).abs() <= 1.0e-12,
            "core cell {k} sum_x drifted outside 1e-12 guard: {sum_x}"
        );
    }
}

#[test]
fn test_premix_omega_symmetric_strang_split_pre_and_post_henyey() {
    let mut base_ctrl = StarControls {
        initial_mass: 3.0,
        rotation_flag: true,
        zbase: 0.02,
        ..StarControls::default()
    };
    base_ctrl.mix_wind.am_nu_factor = 1.0;
    base_ctrl.mix_wind.am_d_mix_factor = 0.033_333_3;
    let mut star_premix =
        load_zams_model(base_ctrl.clone(), StarJobControls::default())
            .expect("3 Msun ZAMS");
    let nz = star_premix.nz;
    let denom =
        f64::from(u32::try_from(nz.saturating_sub(1).max(1)).unwrap_or(1));
    for (k, om) in star_premix.omega.iter_mut().enumerate().take(nz) {
        let frac = f64::from(u32::try_from(k).unwrap_or(0)) / denom;
        *om = 2.0e-5 * (1.0 + 0.8 * frac);
    }
    update_rotation_distortion_factors(&mut star_premix);
    set_hydro_vars(&mut star_premix, false, false).expect("set_hydro_vars");
    star_premix.save_start_values().expect("save_start_values");
    star_premix.save_old_values();

    let mut star_full = star_premix.clone();
    star_premix.controls.mix_wind.premix_omega = true;
    star_full.controls.mix_wind.premix_omega = false;

    let dt = 1.0e2 * SECYER;
    let mut pre_only = star_premix.clone();
    pre_only.dt = dt;
    let shear_0 = (pre_only.omega[0] - pre_only.omega[nz - 1]).abs();
    apply_step_pre_physics(&mut pre_only, dt).expect("apply_step_pre_physics");
    let shear_half = (pre_only.omega[0] - pre_only.omega[nz - 1]).abs();
    assert!(
        shear_half < shear_0,
        "premix_omega must diffuse omega during apply_step_pre_physics: {shear_0} -> {shear_half}"
    );

    star_premix.dt = dt;
    star_full.dt = dt;
    let rep_premix = star_evolve_step(&mut star_premix, false, false, true)
        .expect("premix step");
    let rep_full = star_evolve_step(&mut star_full, false, false, true)
        .expect("full step");
    assert_eq!(rep_premix.dt, rep_full.dt);
    let shear_premix = (star_premix.omega[0] - star_premix.omega[nz - 1]).abs();
    let shear_full = (star_full.omega[0] - star_full.omega[nz - 1]).abs();
    let rel_diff = (shear_premix - shear_full).abs() / shear_0.max(1.0e-30);
    assert!(
        rel_diff < 5.0e-2,
        "Strang-split premix_omega (0.5+0.5)*dt should match 1.0*dt transport: premix={shear_premix}, full={shear_full}, rel_diff={rel_diff}"
    );
}

#[test]
fn test_op_split_burn_runs_before_save_start_values_preserving_pre_burn_t_start()
 {
    let ctrl = StarControls {
        initial_mass: 15.0,
        op_split_burn: true,
        op_split_burn_min_t: 1.0e7,
        zbase: 0.02,
        ..StarControls::default()
    };
    let mut star = load_zams_model(ctrl, StarJobControls::default())
        .expect("15 Msun ZAMS");
    set_hydro_vars(&mut star, false, false).expect("set_hydro_vars");
    star.save_start_values().expect("save_start_values");
    star.save_old_values();

    let c_idx = star.nz - 1;
    let h1_j = star
        .iso_names
        .iter()
        .position(|n| n.eq_ignore_ascii_case("h1"))
        .expect("h1 index");
    let h1_init = star.cell_xa(c_idx)[h1_j];
    let ln_t_init = star.ln_t[c_idx];

    let dt = 2.0e4 * SECYER;
    let rep = do_struct_burn_mix(&mut star, dt, true, true, &[])
        .expect("op_split_burn struct_burn_mix");
    assert!(rep.converged);

    let h1_start_saved = star.xa_start[c_idx * star.species + h1_j];
    let h1_final = star.cell_xa(c_idx)[h1_j];
    assert!(
        h1_start_saved < h1_init,
        "xa_start must capture post-op_split_burn core H1 depletion: init={h1_init}, xa_start={h1_start_saved}"
    );
    assert!(
        h1_final < h1_init,
        "final core H1 must remain depleted after Henyey mixing: init={h1_init}, final={h1_final}"
    );
    assert!(
        (star.ln_t_start[c_idx] - ln_t_init).abs() < 1.0e-14,
        "ln_t_start must preserve pre-burn temperature: expected {ln_t_init}, got {}",
        star.ln_t_start[c_idx]
    );
}

struct FixedWindAndTerminateExtras {
    wind_msun_yr: f64,
    wind_calls: usize,
    stop_at_step: usize,
}

impl BinaryExtras for FixedWindAndTerminateExtras {
    fn other_wind_cgs(
        &mut self,
        _star: &StarModel,
        _star_idx: usize,
        ignore_rlof_flag: bool,
    ) -> Result<Option<f64>, StarError> {
        self.wind_calls += 1;
        if ignore_rlof_flag {
            return Ok(Some(0.0));
        }
        Ok(Some(-self.wind_msun_yr * (MSUN / SECYER)))
    }

    fn finish_step(&mut self, bin: &mut BinarySystem) -> BinaryExtrasAction {
        if bin.model_number >= self.stop_at_step {
            BinaryExtrasAction::Terminate(String::from("CustomExtrasStop"))
        } else {
            BinaryExtrasAction::KeepGoing
        }
    }
}

#[test]
fn test_binary_extras_custom_other_wind_and_finish_step_termination() {
    let c1 = StarControls {
        initial_mass: 2.0,
        use_other_wind: true,
        zbase: 0.02,
        ..StarControls::default()
    };
    let s1 =
        load_zams_model(c1, StarJobControls::default()).expect("2 Msun ZAMS");
    let bctrl = BinaryInlistControls {
        evolve_both_stars: false,
        m1_msun: 2.0,
        m2_msun: 1.4,
        initial_period_in_days: 5.0,
        terminate_if_initial_overflow: false,
        ..BinaryInlistControls::default()
    };
    let mut bin = BinarySystem::from_controls_point_mass(s1, bctrl)
        .expect("point-mass binary");
    let mut extras = FixedWindAndTerminateExtras {
        wind_msun_yr: 1.0e-7,
        wind_calls: 0,
        stop_at_step: 2,
    };

    let m1_before = bin.donor.mstar;
    let (dt1, _) = bin
        .try_step_with_extras(100.0 * SECYER, &mut extras)
        .expect("step 1");
    assert!(extras.wind_calls >= 1);
    assert!(
        bin.donor.mstar < m1_before,
        "custom other_wind must remove mass"
    );
    assert!(bin.last_termination_reason.is_none());
    // Verify finalize_star_post_binary_torques did NOT overwrite solved ln_t_start with ln_t.
    let diff_ln_t: f64 = bin
        .donor
        .ln_t
        .iter()
        .zip(&bin.donor.ln_t_start)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        diff_ln_t > 0.0,
        "post-trial finalize must preserve start-of-step ln_t_start rather than overwriting with ln_t"
    );

    let _ = bin.try_step_with_extras(dt1, &mut extras).expect("step 2");
    assert_eq!(
        bin.last_termination_reason.as_deref(),
        Some("CustomExtrasStop")
    );
}
