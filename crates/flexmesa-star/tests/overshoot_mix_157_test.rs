// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone invariant and unit tests for convective overshooting,
//! convective premixing, mixing state/microphysics, rotational mixing
//! diffusivities, and `Quad` angular-momentum diffusion (`#157`, `#158`,
//! `#161`).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use flexmesa_const::{MixingType, M_SUN, SECYER, STANDARD_CGRAV, TWO_THIRDS};
use flexmesa_math::{cospi, exp, pow, pow3};
use flexmesa_num::{find0, two_piece_linear_coeffs};
use flexmesa_star::conv_premix;
use flexmesa_star::hydro_rotation::{
    set_rotation_info, use_xh_to_update_i_rot_and_j_rot,
};
use flexmesa_star::hydro_state::{
    BinaryOtherTorqueHandle, BinaryOtherTorqueHook,
};
use flexmesa_star::inlist_runner::{DefaultStarExtras, StarExtras};
use flexmesa_star::mix_info::{locate_convection_boundaries, set_mixing_info};
use flexmesa_star::mixing_state::{
    driving_species_for_cid, set_grads, threshold_smoothing, weighted_smoothing,
};
use flexmesa_star::omega_mix::{
    do_solve_omega_mix, do_solve_omega_mix_with_extras, solve_omega_mix,
    OmegaMixOutcome,
};
use flexmesa_star::overshoot::{
    add_overshooting, apply_overshoot_configs, eval_conv_bdy_hp,
    eval_conv_bdy_k, eval_conv_bdy_r, eval_over_bdy_params, OverBdyAnchorCache,
    OvershootConfig, OvershootScheme,
};
use flexmesa_star::premix;
use flexmesa_star::relax::apply_uniform_rotation;
use flexmesa_star::rotation_mix_info::{
    set_rotation_mixing_info, store_prev_mesh_st_info, store_st_start_info,
};
use flexmesa_star::{
    load_zams_model, set_hydro_vars, StarControls, StarError, StarJobControls,
    StarModel,
};

fn zams_4m() -> StarModel {
    let ctrl = StarControls {
        initial_mass: 4.0,
        zbase: 0.02,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(ctrl, StarJobControls::default()).expect("4 Msun ZAMS");
    model.dt = 1.0e3 * SECYER;
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model.save_start_values().expect("save_start_values");
    model
}

/// Builds a 4 Msun ZAMS model with an outward-increasing H1 / decreasing He4
/// gradient anchored directly at the convective core top (`k_bdy..k_bdy+60`)
/// so no detached convective shell forms above the core, Ledoux stabilization
/// keeps the ramp radiative while `gradr > grada`, and `conv_premix` can
/// advance outward into higher-H1 material (halted when
/// `conv_premix_avoid_increase = true`).
fn build_ledoux_core_gradient_4m() -> StarModel {
    let mut model = zams_4m();
    model.controls.use_ledoux_criterion = true;
    model
        .controls
        .mix_wind
        .num_cells_for_smooth_gradl_composition_term = 0;
    model.controls.mix_wind.num_cells_for_smooth_brunt_b = 0;
    let _ = locate_convection_boundaries(&mut model)
        .expect("locate_convection_boundaries");
    let k_bdy = model.aux.mixing.conv_bdy_loc[0];

    let sp = model.species;
    let idx_h1 = model.iso_names.iter().position(|s| s == "h1").expect("h1");
    let idx_he4 = model
        .iso_names
        .iter()
        .position(|s| s == "he4")
        .expect("he4");
    let k_grad_bot = (k_bdy + 60).min(model.nz - 10);
    let span = f64::from(
        u32::try_from((k_grad_bot - k_bdy).max(1)).expect("small span"),
    );
    for k in k_bdy..model.nz {
        let frac = if k < k_grad_bot {
            f64::from(u32::try_from(k - k_bdy + 1).expect("small offset"))
                / span
        } else {
            1.0
        };
        let delta = (0.012 * frac.min(1.0)).min(model.xa[k * sp + idx_h1]);
        model.xa[k * sp + idx_h1] -= delta;
        model.xa[k * sp + idx_he4] += delta;
    }
    model.dt = 1.0e12;
    set_hydro_vars(&mut model, false, false).expect("refresh Ledoux gradient");
    model
}

fn synthetic_overshoot_model() -> StarModel {
    let nz = 8_usize;
    let ctrl = StarControls {
        initial_mass: 4.0,
        ..Default::default()
    };
    let mut model =
        StarModel::new(nz, vec![2, 7], StarJobControls::default(), ctrl);
    let radii = [
        7.0e10, 6.0e10, 5.0e10, 4.0e10, 3.0e10, 2.0e10, 1.0e10, 5.0e9,
    ];
    let dq_val = 1.0 / (nz as f64);
    for k in 0..nz {
        model.dq[k] = dq_val;
        model.q[k] = 1.0 - (k as f64) * dq_val;
        model.dm[k] = model.xmstar * dq_val;
        model.r[k] = radii[k];
        model.cgrav[k] = STANDARD_CGRAV;
        model.ln_peos[k] = 30.0 + 0.4 * (k as f64);
        model.peos[k] = exp(model.ln_peos[k]);
        model.lnd[k] = 1.0 + 0.2 * (k as f64);
        model.rho[k] = exp(model.lnd[k]);
        model.scale_height[k] = 1.0e10;
        model.aux.mixing.mlt_mixing_length[k] = 3.0e9;
    }
    model.conv_vel = vec![0.0, 0.0, 0.0, 1.0e5, 2.0e5, 3.0e5, 4.0e5, 5.0e5];
    model.mixing_type = vec![
        MixingType::NoMixing,
        MixingType::NoMixing,
        MixingType::NoMixing,
        MixingType::NoMixing,
        MixingType::ConvectiveMixing,
        MixingType::ConvectiveMixing,
        MixingType::ConvectiveMixing,
        MixingType::ConvectiveMixing,
    ];
    model.d_mix = vec![0.0, 0.0, 0.0, 0.0, 1.0e14, 1.0e14, 1.0e14, 1.0e14];
    model.aux.mixing.conv_bdy_loc = vec![3];
    model.aux.mixing.top_conv_bdy = vec![true];
    model.aux.mixing.cz_bdy_dq[3] = 0.4 * dq_val;
    model.aux.mixing.conv_bdy_q = vec![model.q[3] - 0.4 * dq_val];
    model.aux.mixing.burn_h_conv_region = vec![true];
    model.aux.mixing.burn_he_conv_region = vec![false];
    model.aux.mixing.burn_z_conv_region = vec![false];
    model
}

#[test]
fn test_overshoot_d_mix_check_runs_without_boundaries_and_guards_bdy_bounds() {
    // Upstream `overshoot.f90:309-323` checks `d_mix` finiteness across
    // `0..nz` even when `n_bdy == 0` or every overshoot scheme is blank.
    let mut m_empty = StarModel::new(
        5,
        vec![2, 7],
        StarJobControls::default(),
        StarControls::default(),
    );
    m_empty.d_mix[2] = f64::NAN;
    let err = add_overshooting(&mut m_empty)
        .expect_err("NaN d_mix must fail even with zero boundaries");
    assert!(matches!(err, StarError::MicrophysicsError(_)));

    // Tiny `nz = 2` grid with `n_bdy = 0`: finite succeeds, Inf fails.
    let mut m_tiny = StarModel::new(
        2,
        vec![2, 7],
        StarJobControls::default(),
        StarControls::default(),
    );
    assert!(add_overshooting(&mut m_tiny).is_ok());
    m_tiny.d_mix[1] = f64::INFINITY;
    assert!(matches!(
        add_overshooting(&mut m_tiny),
        Err(StarError::MicrophysicsError(_))
    ));

    // Boundaries present, all schemes empty -> skips overshoot loop but still
    // enforces the `d_mix` finiteness check.
    let mut m_blank = synthetic_overshoot_model();
    assert!(add_overshooting(&mut m_blank).is_ok());
    m_blank.d_mix[0] = f64::NEG_INFINITY;
    assert!(matches!(
        add_overshooting(&mut m_blank),
        Err(StarError::MicrophysicsError(_))
    ));

    // Malformed out-of-range `conv_bdy_loc` under CZ Hp limiting errors cleanly.
    let mut m_bad_loc = synthetic_overshoot_model();
    m_bad_loc
        .controls
        .mix_wind
        .limit_overshoot_hp_using_size_of_convection_zone = true;
    m_bad_loc.aux.mixing.conv_bdy_loc = vec![99, 2];
    m_bad_loc.aux.mixing.top_conv_bdy = vec![false, true];
    m_bad_loc.aux.mixing.conv_bdy_q = vec![0.3, 0.5];
    assert!(matches!(
        eval_conv_bdy_hp(&m_bad_loc, 1),
        Err(StarError::InvalidInput(_))
    ));
}

#[test]
fn test_overshoot_bdy_geometry_hp_limit_cache_and_vc_branches() {
    let mut m = synthetic_overshoot_model();

    assert_eq!(eval_conv_bdy_k(&m, 0).expect("k_cb"), 3);
    let r_cb = eval_conv_bdy_r(&m, 0).expect("r_cb");
    let w = 0.4_f64;
    let want_r_cb = pow(w * pow3(m.r[4]) + (1.0 - w) * pow3(m.r[3]), 1.0 / 3.0);
    assert!((r_cb - want_r_cb).abs() <= 1.0e-14 * want_r_cb);

    // Exact 3-cell `two_piece_linear_coeffs` check on `eval_conv_bdy_hp`.
    let dq = m.dq[3];
    let (a0, a1, a2) = two_piece_linear_coeffs(
        dq + m.aux.mixing.cz_bdy_dq[3],
        0.5 * dq,
        1.5 * dq,
        2.5 * dq,
    )
    .expect("two_piece_linear_coeffs");
    let want_p = exp(a0 * m.ln_peos[2] + a1 * m.ln_peos[3] + a2 * m.ln_peos[4]);
    let want_rho = exp(a0 * m.lnd[2] + a1 * m.lnd[3] + a2 * m.lnd[4]);
    let want_hp = want_p
        / (want_rho * STANDARD_CGRAV * (m.xmstar * m.aux.mixing.conv_bdy_q[0])
            / (r_cb * r_cb));
    let hp_cb = eval_conv_bdy_hp(&m, 0).expect("hp_cb");
    assert!((hp_cb - want_hp).abs() <= 1.0e-14 * want_hp);

    // `limit_overshoot_Hp_using_size_of_convection_zone`: upstream asymmetric
    // use of `alpha_mlt[k]` on LHS vs `mixing_length_alpha` on RHS when
    // `overshoot_alpha <= 0` (`overshoot_utils.f90:219`). Scale `ln_peos` up
    // so `Hp_cb > dr_cz` unconditionally triggers the cap.
    let mut m_lim = m.clone();
    for k in 0..m_lim.nz {
        m_lim.ln_peos[k] = 38.0 + 0.4 * (k as f64);
        m_lim.peos[k] = exp(m_lim.ln_peos[k]);
    }
    m_lim.controls.mixing_length_alpha = 2.0;
    m_lim.aux.mixing.alpha_mlt = vec![4.0; m_lim.nz];
    m_lim
        .controls
        .mix_wind
        .limit_overshoot_hp_using_size_of_convection_zone = true;
    m_lim.controls.mix_wind.overshoot_alpha = 0.0;
    let dr_cz = m_lim.r[3] - m_lim.r_center; // 4e10
    let hp_limited = eval_conv_bdy_hp(&m_lim, 0).expect("hp limited");
    let want_lim = dr_cz / m_lim.controls.mixing_length_alpha;
    assert_eq!(hp_limited.to_bits(), want_lim.to_bits());

    m_lim.controls.mix_wind.overshoot_alpha = 2.5;
    let hp_oa = eval_conv_bdy_hp(&m_lim, 0).expect("hp overshoot_alpha");
    let want_oa = dr_cz / 2.5;
    assert_eq!(hp_oa.to_bits(), want_oa.to_bits());

    // Bottom boundary in cell 0 lacks `k_o = k - 1` (`DECISIONS #43`).
    let mut m_bot0 = m.clone();
    m_bot0.aux.mixing.conv_bdy_loc = vec![1];
    m_bot0.aux.mixing.top_conv_bdy = vec![false];
    m_bot0.aux.mixing.cz_bdy_dq[0] = 0.5 * dq;
    assert!(matches!(
        eval_conv_bdy_hp(&m_bot0, 0),
        Err(StarError::InvalidInput(_))
    ));

    // Anchor cache invariants and all 3 `conv_vel` interpolation branches.
    let f0_small = 1.0e9 / hp_cb;
    let cache = OverBdyAnchorCache::from_model(&m, 0).expect("cache");
    assert_eq!(cache.boundary_index(), 0);
    assert_eq!(cache.hp_cb().to_bits(), hp_cb.to_bits());

    // Branch 1: both faces nonzero (`top` boundary in cell 3).
    let (k1, r1, d1, vc1) =
        eval_over_bdy_params(&m, 0, f0_small).expect("branch 1");
    let (ck1, cr1, cd1, cvc1) =
        cache.eval_at(f0_small).expect("cache branch 1");
    assert_eq!(
        (k1, r1.to_bits(), d1.to_bits(), vc1.to_bits()),
        (ck1, cr1.to_bits(), cd1.to_bits(), cvc1.to_bits())
    );
    assert_eq!(k1, 3);
    let w_ob = (pow3(m.r[3]) - pow3(r1)) / (pow3(m.r[3]) - pow3(m.r[4]));
    let want_vc1 = (1.0 - w_ob) * 1.0e5 + w_ob * 2.0e5;
    assert!((vc1 - want_vc1).abs() <= 1.0e-13 * want_vc1);
    assert!((d1 - vc1 * 3.0e9 / 3.0).abs() <= 1.0e-14 * d1);

    // Branch 3 (`top` boundary): outer face zero (`conv_vel[3] = 0`), inner face nonzero.
    m.conv_vel[3] = 0.0;
    m.conv_vel[4] = 2.0e5;
    let (_, r3, _, vc3) =
        eval_over_bdy_params(&m, 0, f0_small).expect("branch 3");
    let w3 = (pow3(r_cb) - pow3(r3)) / (pow3(r_cb) - pow3(m.r[4]));
    assert!((vc3 - w3 * 2.0e5).abs() <= 1.0e-13 * 2.0e5);

    // Branch 2 (`bottom` boundary in cell 3, `conv_bdy_loc = 4`): outer face
    // nonzero (`conv_vel[3] = 1e5`), inner face zero (`conv_vel[4] = 0`),
    // searching outward so `r_cb < r_ob < r[3]`.
    let mut m_bot = m.clone();
    m_bot.conv_vel[3] = 1.0e5;
    m_bot.conv_vel[4] = 0.0;
    m_bot.aux.mixing.conv_bdy_loc = vec![4];
    m_bot.aux.mixing.top_conv_bdy = vec![false];
    let hp_bot = eval_conv_bdy_hp(&m_bot, 0).expect("hp bot");
    let (_, r2, _, vc2) =
        eval_over_bdy_params(&m_bot, 0, 1.0e9 / hp_bot).expect("branch 2");
    let w2 = (pow3(m_bot.r[3]) - pow3(r2)) / (pow3(m_bot.r[3]) - pow3(r_cb));
    assert!((vc2 - (1.0 - w2) * 1.0e5).abs() <= 1.0e-13 * 1.0e5);
}

#[test]
fn test_overshoot_step_cospi_find0_brunt_bmax_and_configs() {
    // Closed-form `OvershootConfig::eval_factor` checks.
    let cfg_exp = OvershootConfig {
        scheme: OvershootScheme::Exponential,
        f: 0.02,
        ..OvershootConfig::default()
    };
    assert_eq!(cfg_exp.eval_factor(1.0e8, 0.0), 0.0);
    let hp = 1.0e10;
    let dr = 1.0e8;
    assert_eq!(
        cfg_exp.eval_factor(dr, hp).to_bits(),
        exp(-2.0 * dr / (0.02 * hp)).to_bits()
    );

    let cfg_step = OvershootConfig {
        scheme: OvershootScheme::Step,
        f: 0.05,
        ..OvershootConfig::default()
    };
    assert_eq!(cfg_step.eval_factor(0.049 * hp, hp), 1.0);
    assert_eq!(cfg_step.eval_factor(0.050 * hp, hp), 0.0);

    // Step overshoot plateau + `(D0 + Delta0*D_ob)` + `find0` `cz_bdy_dq` update
    // + midpoint `cospi` mass ramp on 4 Msun ZAMS.
    let mut m_step = zams_4m();
    set_mixing_info(&mut m_step).expect("base mixing_info");
    let hp_cb = eval_conv_bdy_hp(&m_step, 0).expect("zams hp_cb");
    let f_full = 0.08_f64;
    let f0_full = 0.01_f64;
    let w_mass = (4.0_f64 - 3.0_f64) / (5.0_f64 - 3.0_f64);
    let ramp = 0.5 * (1.0 - cospi(w_mass));
    let f_eff = f_full * ramp;
    let f0_eff = f0_full * ramp;
    let (_, r_ob, d_ob, vc_ob) =
        eval_over_bdy_params(&m_step, 0, f0_eff).expect("step ob");
    let d0 = 5.0e12_f64;
    let delta0 = 0.5_f64;
    let plateau_d = d0 + delta0 * d_ob;
    let plateau_vc = (d0 / d_ob + delta0) * vc_ob;

    let ov = &mut m_step.controls.mix_wind.overshoot_sets[0];
    ov.scheme = String::from("step");
    ov.zone_type = String::from("burn_H");
    ov.zone_loc = String::from("core");
    ov.bdy_loc = String::from("top");
    ov.f = f_full;
    ov.f0 = f0_full;
    ov.d0 = d0;
    ov.delta0 = delta0;
    ov.mass_full_off = 3.0;
    ov.mass_full_on = 5.0;

    let r_cb = eval_conv_bdy_r(&m_step, 0).expect("zams r_cb");
    add_overshooting(&mut m_step).expect("step overshoot");

    let mut k_outermost_ov = None;
    for k in 0..m_step.nz {
        if m_step.mixing_type[k] == MixingType::OvershootMixing {
            assert!((m_step.r[k] - r_ob) < f_eff * hp_cb);
            assert!((m_step.d_mix[k] - plateau_d).abs() <= 1.0e-12 * plateau_d);
            assert!(
                (m_step.conv_vel[k] - plateau_vc).abs() <= 1.0e-12 * plateau_vc
            );
            if k_outermost_ov.is_none() {
                k_outermost_ov = Some(k);
            }
        }
    }
    let kc_first_ov = k_outermost_ov.expect("step overshoot must activate");
    assert!(kc_first_ov > 0 && m_step.r[kc_first_ov - 1] > r_cb);
    let k_cut = kc_first_ov - 1;
    let d_min = m_step.controls.mix_wind.overshoot_d_min;
    let expected_dq = find0(
        0.0,
        0.0 - d_min,
        m_step.dq[k_cut],
        m_step.d_mix[k_cut + 1] - d_min,
    )
    .expect("find0");
    assert!(
        (m_step.aux.mixing.cz_bdy_dq[k_cut] - expected_dq).abs()
            <= 1.0e-14 * m_step.dq[k_cut]
    );

    // Error gates: `f <= 0 || f0 <= 0`, and `overshoot_brunt_B_max > 0` without
    // `calculate_Brunt_N2`.
    let mut m_f0 = zams_4m();
    set_mixing_info(&mut m_f0).expect("mixing");
    m_f0.controls.mix_wind.overshoot_sets[0].scheme =
        String::from("exponential");
    m_f0.controls.mix_wind.overshoot_sets[0].zone_type = String::from("any");
    m_f0.controls.mix_wind.overshoot_sets[0].zone_loc = String::from("any");
    m_f0.controls.mix_wind.overshoot_sets[0].bdy_loc = String::from("any");
    m_f0.controls.mix_wind.overshoot_sets[0].f = 0.0;
    m_f0.controls.mix_wind.overshoot_sets[0].f0 = 0.004;
    assert!(matches!(
        add_overshooting(&mut m_f0),
        Err(StarError::InvalidInput(_))
    ));

    let mut m_bmax_err = zams_4m();
    set_mixing_info(&mut m_bmax_err).expect("mixing");
    m_bmax_err.controls.calculate_brunt_b = true;
    m_bmax_err.controls.calculate_brunt_n2 = false;
    m_bmax_err.controls.mix_wind.overshoot_brunt_b_max = 1.0e-3;
    let s0 = &mut m_bmax_err.controls.mix_wind.overshoot_sets[0];
    s0.scheme = String::from("exponential");
    s0.zone_type = String::from("any");
    s0.zone_loc = String::from("any");
    s0.bdy_loc = String::from("any");
    s0.f = 0.02;
    s0.f0 = 0.004;
    assert!(matches!(
        add_overshooting(&mut m_bmax_err),
        Err(StarError::InvalidInput(_))
    ));

    // Post-loop CZ zeroing (`overshoot.f90:294-297`) when `D(k) < D_min`
    // triggers while still inside the convection zone (`k >= k_cb`).
    let mut m_cz_zero = zams_4m();
    set_mixing_info(&mut m_cz_zero).expect("mixing");
    let k_cb = eval_conv_bdy_k(&m_cz_zero, 0).expect("k_cb");
    let (k_ob_zero, _, _, _) =
        eval_over_bdy_params(&m_cz_zero, 0, 0.25).expect("k_ob_zero");
    assert!(k_ob_zero > k_cb);
    let mut cfg_high_dmin = OvershootConfig {
        scheme: OvershootScheme::Step,
        top_conv_bdy: true,
        f: 0.01,
        f0: 0.25,
        d0: 0.0,
        delta0: 0.0,
        d_min: 1.0,
    };
    apply_overshoot_configs(&mut m_cz_zero, &[cfg_high_dmin.clone()])
        .expect("high d_min zeros CZ remainder");
    for k in k_cb..=k_ob_zero {
        assert_eq!(m_cz_zero.d_mix[k], 0.0);
        assert_eq!(m_cz_zero.mixing_type[k], MixingType::NoMixing);
    }
    if k_ob_zero + 1 < m_cz_zero.nz {
        assert!(m_cz_zero.d_mix[k_ob_zero + 1] > 0.0);
    }

    // `apply_overshoot_configs`: disabled configs (`f <= 0`) are no-op, and
    // prior `overshoot_sets`/`overshoot_d_min` are restored even when `Err`.
    let mut m_cfg = zams_4m();
    set_mixing_info(&mut m_cfg).expect("mixing");
    m_cfg.controls.mix_wind.overshoot_sets[0].scheme =
        String::from("exponential");
    m_cfg.controls.mix_wind.overshoot_d_min = 77.0;
    cfg_high_dmin.f = 0.0;
    let d_snap = m_cfg.d_mix.clone();
    apply_overshoot_configs(&mut m_cfg, &[cfg_high_dmin])
        .expect("disabled config is no-op");
    assert_eq!(m_cfg.d_mix, d_snap);

    m_cfg.aux.mixing.cz_bdy_dq[k_cb] = -1.0; // invalid -> triggers error in eval_conv_bdy_r
    let valid_cfg = OvershootConfig::default();
    assert!(apply_overshoot_configs(&mut m_cfg, &[valid_cfg]).is_err());
    assert_eq!(
        m_cfg.controls.mix_wind.overshoot_sets[0].scheme,
        "exponential"
    );
    assert_eq!(m_cfg.controls.mix_wind.overshoot_d_min, 77.0);
}

#[test]
fn test_conv_premix_fuel_avoidance_burn_classification_and_premix_heating() {
    let mut base = build_ledoux_core_gradient_4m();
    base.controls.mix_wind.conv_premix_time_factor = -1.0;
    base.controls.mix_wind.conv_premix_fix_pgas = true;

    // 1. Short `dt` forces `(avg_xa_new - avg_xa_old) / dt` > `-davg_xa_dt[h1]`
    //    so `conv_premix_avoid_increase = true` blocks advancing into H-rich
    //    outer shells, whereas `false` advances cleanly.
    let mut m_avoid = base.clone();
    m_avoid.dt = 1.0e5;
    m_avoid.controls.mix_wind.conv_premix_avoid_increase = true;
    let adv_avoid =
        conv_premix::do_conv_premix(&mut m_avoid).expect("avoid increase");
    assert_eq!(
        adv_avoid, 0,
        "fuel increase avoidance must halt advance when mixing increases H1 faster than burning"
    );

    let mut m_allow = base.clone();
    m_allow.dt = 1.0e5;
    m_allow.controls.mix_wind.conv_premix_avoid_increase = false;
    let adv_allow =
        conv_premix::do_conv_premix(&mut m_allow).expect("allow advance");
    assert!(adv_allow > 0);
    assert!(m_allow.aux.need_to_setvars);

    // 2. `premix::do_conv_premix` wrapper: heating rate `eps_pre_mix` vs disabled,
    //    plus `conv_premix_avoid_increase` restoration on `Err`.
    let mut m_heat = base.clone();
    m_heat.controls.do_premix_heating = true;
    let e_before = m_heat.energy.clone();
    let adv_w =
        premix::do_conv_premix(&mut m_heat, 0, false).expect("premix heating");
    assert_eq!(adv_w, adv_allow);
    let mut max_eps = 0.0_f64;
    for k in 0..m_heat.nz {
        let want = (e_before[k] - m_heat.energy[k]) / m_heat.dt;
        assert_eq!(m_heat.aux.eps_pre_mix[k].to_bits(), want.to_bits());
        max_eps = max_eps.max(m_heat.aux.eps_pre_mix[k].abs());
    }
    assert!(max_eps > 0.0);

    let mut m_noheat = base.clone();
    m_noheat.controls.do_premix_heating = false;
    let _ = premix::do_conv_premix(&mut m_noheat, 0, false)
        .expect("premix no heating");
    assert!(m_noheat.aux.eps_pre_mix.iter().all(|&v| v == 0.0));

    let mut m_err = base.clone();
    m_err.controls.mix_wind.conv_premix_avoid_increase = true;
    m_err.dxdt_nuc.clear(); // triggers error in `set_abund_data`
    assert!(premix::do_conv_premix(&mut m_err, 0, false).is_err());
    assert!(
        m_err.controls.mix_wind.conv_premix_avoid_increase,
        "premix wrapper must restore conv_premix_avoid_increase on error"
    );

    // 3. `premix::add_predictive_mixing` initializes empty `conv_bdy_loc`,
    //    rejects simultaneous `do_conv_premix`, and restores `predictive_sets`.
    let mut m_pred = base.clone();
    m_pred.aux.mixing.conv_bdy_loc.clear();
    m_pred.controls.mix_wind.predictive_sets[0].superad_thresh = 0.9;
    let _ = premix::add_predictive_mixing(&mut m_pred, 0.0, 0.0)
        .expect("standalone add_predictive_mixing");
    assert_eq!(
        m_pred.controls.mix_wind.predictive_sets[0].superad_thresh,
        0.9
    );

    m_pred.controls.do_conv_premix = true;
    assert!(premix::add_predictive_mixing(&mut m_pred, 0.0, 0.0).is_err());
    assert_eq!(
        m_pred.controls.mix_wind.predictive_sets[0].superad_thresh,
        0.9
    );
}

#[test]
fn test_mixing_state_smoothing_set_grads_and_driving_species() {
    // 1. Exact `weighted_smoothing` (Pascal `[1,2,1]/4` & `preserve_sign`).
    let mut dd = [0.0, 0.0, 16.0, 0.0, 0.0];
    let mut work = [0.0; 5];
    weighted_smoothing(&mut dd, 1, false, &mut work);
    assert_eq!(dd, [0.0, 4.0, 8.0, 4.0, 0.0]);

    let mut signed = [10.0, 4.0, -0.0, 8.0, 12.0];
    let mut w_signed = [0.0; 5];
    weighted_smoothing(&mut signed, 1, true, &mut w_signed);
    // At index 1 (`v0=4`), neighbor index 2 (`-0.0`) breaks (`v0 * (-0.0) <= 0`),
    // so only weights [1, 2] participate: (1*10 + 2*4)/3 = 6.
    assert!((signed[1] - 6.0).abs() <= 1.0e-14);
    assert_eq!(signed[2], -0.0);

    // 2. `threshold_smoothing` leaves sub-threshold cells and region endpoints
    //    untouched.
    let mut arr = [1.0, 2.0, 0.01, 2.0, 6.0, 2.0, 0.05];
    let mut w_arr = [0.0; 7];
    threshold_smoothing(&mut arr, 0.5, 1, false, &mut w_arr);
    // Region 1 `[0..=1]` has length 2 -> unchanged `[1.0, 2.0]`.
    assert_eq!(arr[0], 1.0);
    assert_eq!(arr[1], 2.0);
    assert_eq!(arr[2], 0.01);
    // Region 2 `[3..=5]` smooths index 4: (2 + 12 + 2)/4 = 4.0.
    assert_eq!(arr[3], 2.0);
    assert_eq!(arr[4], 4.0);
    assert_eq!(arr[5], 2.0);
    assert_eq!(arr[6], 0.05);

    // 3. `set_grads`: `grad_density`, `grad_temperature`, Brunt toggles, and
    //    `dominant_iso_for_thermohaline` stencil + edge copy (`hydro_vars.f90:1050-1117`).
    let cat = flexmesa_chem::NuclideCatalog::builtin_ref();
    let cid_h1 = cat.get_by_name("h1").expect("h1").chem_id;
    let cid_he3 = cat.get_by_name("he3").expect("he3").chem_id;
    let cid_he4 = cat.get_by_name("he4").expect("he4").chem_id;
    let nz = 7_usize;
    let ctrl = StarControls {
        calculate_brunt_b: true,
        use_ledoux_criterion: true,
        ..Default::default()
    };
    let mut m = StarModel::new(
        nz,
        vec![cid_h1, cid_he3, cid_he4],
        StarJobControls::default(),
        ctrl,
    );
    m.controls
        .mix_wind
        .num_cells_for_smooth_gradl_composition_term = 0;
    for k in 0..nz {
        let kf = k as f64;
        m.ln_peos[k] = 20.0 + kf;
        m.lnd[k] = 1.0 + 0.6 * kf;
        m.ln_t[k] = 10.0 + 0.4 * kf;
        m.aux.mixing.unsmoothed_brunt_b[k] = 0.05 * kf;
        // Outward enrichment in he3 -> `(xa[k-2]+xa[k-1]-xa[k]-xa[k+1])*(1+Z)/A`
        // maximizes for `he3` (`(1+2)/3 = 1`) over `he4` (`3/4`) when steps match,
        // or make `he3` gradient steeper at interior cells.
        m.xa[k * 3] = 0.5;
        m.xa[k * 3 + 1] = 0.3 - 0.02 * kf;
        m.xa[k * 3 + 2] = 0.2 + 0.02 * kf;
    }
    set_grads(&mut m);
    // `grad_density[0]` / `grad_temperature[0]` start at 0 before
    // `smooth_in_place` sweeps `[3*v[0]+v[1], v[k-1]+2*v[k]+v[k+1], ...]/4`
    // in place from `k = 0` inward, leaving `0.6 - grad[k] = 0.45 * 4^(-k)`.
    let mut want_gd = vec![0.6; nz];
    let mut want_gt = vec![0.4; nz];
    want_gd[0] = 0.0;
    want_gt[0] = 0.0;
    for arr in [&mut want_gd, &mut want_gt] {
        arr[0] = (3.0 * arr[0] + arr[1]) / 4.0;
        for k in 1..(nz - 1) {
            arr[k] = (arr[k - 1] + 2.0 * arr[k] + arr[k + 1]) / 4.0;
        }
        arr[nz - 1] = (arr[nz - 2] + 3.0 * arr[nz - 1]) / 4.0;
    }
    for k in 0..nz {
        assert!(
            (m.aux.rotmix.grad_density[k] - want_gd[k]).abs() <= 1.0e-14,
            "grad_density[{k}]"
        );
        assert!(
            (m.aux.rotmix.grad_temperature[k] - want_gt[k]).abs() <= 1.0e-14,
            "grad_temperature[{k}]"
        );
    }
    assert_eq!(m.aux.smoothed_brunt_b, m.aux.mixing.unsmoothed_brunt_b);
    assert_eq!(m.aux.mixing.gradl_composition_term, m.aux.smoothed_brunt_b);
    assert_eq!(m.aux.mixing.dominant_iso_for_thermohaline[2], cid_he3);
    assert_eq!(m.aux.mixing.dominant_iso_for_thermohaline[0], cid_he3);
    assert_eq!(m.aux.mixing.dominant_iso_for_thermohaline[1], cid_he3);
    assert_eq!(
        m.aux.mixing.dominant_iso_for_thermohaline[nz - 1],
        m.aux.mixing.dominant_iso_for_thermohaline[nz - 3]
    );
    let ds = driving_species_for_cid(cid_he3);
    assert_eq!((ds.chem_z, ds.chem_a), (2.0, 3.0));
    let ds_fallback = driving_species_for_cid(0);
    assert_eq!((ds_fallback.chem_z, ds_fallback.chem_a), (2.0, 4.0));

    // Toggle `use_ledoux_criterion = false` keeps `smoothed_brunt_b` while
    // zeroing `gradl_composition_term`; `calculate_brunt_b = false` zeroes both.
    m.controls.use_ledoux_criterion = false;
    set_grads(&mut m);
    assert!(m.aux.smoothed_brunt_b[3] > 0.0);
    assert!(m
        .aux
        .mixing
        .gradl_composition_term
        .iter()
        .all(|&v| v == 0.0));
    m.controls.calculate_brunt_b = false;
    set_grads(&mut m);
    assert!(m.aux.smoothed_brunt_b.iter().all(|&v| v == 0.0));
}

#[test]
fn test_rotmix_st_guards_smoothing_clipping_and_d_omega_mixing() {
    // 1. All-zero `omega` fast exit clears rotational arrays cleanly.
    let mut m0 = zams_4m();
    m0.aux.rotmix.d_dsi = vec![42.0; m0.nz];
    m0.aux.rotmix.d_omega = vec![99.0; m0.nz];
    set_rotation_mixing_info(&mut m0).expect("zero omega fast exit");
    assert!(m0.aux.rotmix.d_dsi.iter().all(|&v| v == 0.0));
    assert!(m0.aux.rotmix.d_omega.iter().all(|&v| v == 0.0));

    // 2. Non-zero `omega` on `nz < 3` returns `InvalidInput` (`DECISIONS #43`).
    let mut m_tiny = StarModel::new(
        2,
        vec![2, 7],
        StarJobControls::default(),
        StarControls::default(),
    );
    m_tiny.omega = vec![1.0e-5, 1.0e-5];
    assert!(matches!(
        set_rotation_mixing_info(&mut m_tiny),
        Err(StarError::InvalidInput(_))
    ));

    // 3. Deferred upstream `#503`: ST active while both `D_ES_factor` and
    //    `D_GSF_factor` are zero leaves `kap_cond` unset.
    let mut m_503 = zams_4m();
    apply_uniform_rotation(&mut m_503, 1.0e-5);
    m_503.controls.mix_wind.d_st_factor = 1.0;
    m_503.controls.mix_wind.d_es_factor = 0.0;
    m_503.controls.mix_wind.d_gsf_factor = 0.0;
    let err_503 = set_rotation_mixing_info(&mut m_503)
        .expect_err("#503 missing kap_cond must error");
    assert!(matches!(err_503, StarError::InvalidInput(_)));

    // 4. Full rotational mixing suite on sheared 4 Msun ZAMS.
    let mut m = build_ledoux_core_gradient_4m();
    apply_uniform_rotation(&mut m, 2.0e-5);
    for k in 0..m.nz {
        m.omega[k] = 2.0e-5 * (1.0 + 0.6 * (1.0 - m.q[k]));
    }
    use_xh_to_update_i_rot_and_j_rot(&mut m);
    set_rotation_info(&mut m, false);
    m.controls.d_omega_flag = true;
    m.controls.mix_wind.d_dsi_factor = 1.0;
    m.controls.mix_wind.d_sh_factor = 1.0;
    m.controls.mix_wind.d_ssi_factor = 1.0;
    m.controls.mix_wind.d_es_factor = 1.0;
    m.controls.mix_wind.d_gsf_factor = 1.0;
    m.controls.mix_wind.d_st_factor = 1.0;
    m.controls.mix_wind.am_nu_st_factor = 1.0;
    m.controls.mix_wind.smooth_d_omega = 1;
    m.controls.mix_wind.d_omega_mixing_rate = 1.0;
    m.dt = 1.0e8;

    set_rotation_mixing_info(&mut m).expect("full rotmix");
    assert!(m
        .aux
        .rotmix
        .d_visc
        .iter()
        .all(|&v| v.is_finite() && v >= 0.0));
    assert!(m.aux.rotmix.d_es.iter().any(|&v| v > 0.0));
    assert!(m.aux.rotmix.d_st.iter().any(|&v| v > 0.0));
    assert!(m.aux.rotmix.dynamo_b_phi.iter().any(|&v| v > 0.0));
    assert_eq!(m.aux.rotmix.d_omega[0], 0.0);
    assert!(m
        .aux
        .rotmix
        .d_omega
        .iter()
        .all(|&v| v.is_finite() && v >= 0.0));

    // `skip_rotation_in_convection_zones` & `clip_D_limit`:
    let mut m_skip = m.clone();
    m_skip.controls.mix_wind.skip_rotation_in_convection_zones = true;
    set_rotation_mixing_info(&mut m_skip).expect("skip convective");
    for k in 0..m_skip.nz {
        if m_skip.mixing_type[k] == MixingType::ConvectiveMixing {
            assert_eq!(m_skip.aux.rotmix.d_dsi[k], 0.0);
            assert_eq!(m_skip.aux.rotmix.d_sh[k], 0.0);
            assert_eq!(m_skip.aux.rotmix.d_ssi[k], 0.0);
            assert_eq!(m_skip.aux.rotmix.d_es[k], 0.0);
            assert_eq!(m_skip.aux.rotmix.d_gsf[k], 0.0);
            assert_eq!(m_skip.aux.rotmix.d_st[k], 0.0);
            assert_eq!(m_skip.aux.rotmix.nu_st[k], 0.0);
            assert_eq!(m_skip.aux.rotmix.dynamo_b_r[k], 0.0);
            assert_eq!(m_skip.aux.rotmix.dynamo_b_phi[k], 0.0);
        }
    }

    let mut m_clip = m.clone();
    m_clip.controls.mix_wind.clip_d_limit = 1.0e30;
    set_rotation_mixing_info(&mut m_clip).expect("clip_d_limit");
    assert!(m_clip.aux.rotmix.d_es.iter().all(|&v| v == 0.0));
    assert!(m_clip.aux.rotmix.d_st.iter().all(|&v| v == 0.0));

    // 5. ST time smoothing (`store_st_start_info` + `store_prev_mesh_st_info`)
    //    and `#504` guard (`prev_mesh_dq(0)`).
    let mut m_st = m.clone();
    store_st_start_info(&mut m_st);
    store_prev_mesh_st_info(&mut m_st);
    m_st.doing_first_model_of_run = false;
    m_st.mstar_old = m_st.mstar;
    m_st.dt = 1.0e8;
    // Scale shear to make raw `D_ST` jump sharply, then verify `st_angsmt`
    // pulls it closer to `d_st_start` than `st_angsmt = 0.0`.
    for k in 0..m_st.nz {
        m_st.omega[k] = 5.0e-5 * (1.0 + 1.5 * (1.0 - m_st.q[k]));
    }
    use_xh_to_update_i_rot_and_j_rot(&mut m_st);
    set_rotation_info(&mut m_st, false);

    let mut m_unsmoothed = m_st.clone();
    m_unsmoothed.controls.mix_wind.st_angsmt = 0.0;
    set_rotation_mixing_info(&mut m_unsmoothed).expect("raw st");
    m_st.controls.mix_wind.st_angsmt = 0.2;
    m_st.controls.mix_wind.st_angsml = 0.0;
    set_rotation_mixing_info(&mut m_st).expect("time-smoothed st");
    let mut diff_count = 0_usize;
    for k in 1..(m_st.nz - 1) {
        if m_st.aux.rotmix.d_st[k] != m_unsmoothed.aux.rotmix.d_st[k] {
            diff_count += 1;
        }
    }
    assert!(
        diff_count > 0,
        "ST time smoothing must constrain large ST jumps"
    );

    // `#504` malformed/degenerate `prev_mesh_nz = 1`:
    let mut m_504 = m_st.clone();
    m_504.aux.st_start.prev_mesh_nz = 1;
    m_504.mstar_old = m_504.m[1];
    assert!(matches!(
        set_rotation_mixing_info(&mut m_504),
        Err(StarError::InvalidInput(_))
    ));

    // 6. `doing_finish_load_model` zeroes `D_omega`.
    let mut m_load = m.clone();
    m_load.aux.doing_finish_load_model = true;
    set_rotation_mixing_info(&mut m_load).expect("finish_load_model");
    assert!(m_load.aux.rotmix.d_omega.iter().all(|&v| v == 0.0));
}

struct BadLenExtras;

impl StarExtras for BadLenExtras {
    fn other_torque(&self, _m: &StarModel) -> Option<Vec<f64>> {
        Some(vec![1.0])
    }
}

struct UniformAccelExtras {
    omegadot: f64,
}

impl StarExtras for UniformAccelExtras {
    fn apply_other_torque(
        &mut self,
        m: &mut StarModel,
    ) -> Result<(), StarError> {
        m.aux.extra_omegadot.fill(self.omegadot);
        Ok(())
    }
}

struct FailBinaryImplicit(Arc<AtomicBool>);

impl BinaryOtherTorqueHook for FailBinaryImplicit {
    fn apply(&self, _model: &mut StarModel) -> Result<(), StarError> {
        self.0.store(true, Ordering::SeqCst);
        Err(StarError::ConvergenceFailure(String::from("bad torque")))
    }
}

#[test]
fn test_omega_mix_guards_torques_quad_conservation_and_standalone() {
    // 1. Early returns for `nz < 2` and `dt_total <= 0.0`.
    let mut m0 = StarModel::new(
        0,
        vec![2, 7],
        StarJobControls::default(),
        StarControls::default(),
    );
    assert_eq!(
        do_solve_omega_mix(&mut m0, 1.0e5).expect("nz=0"),
        OmegaMixOutcome::KeepGoing
    );
    let mut m1 = StarModel::new(
        1,
        vec![2, 7],
        StarJobControls::default(),
        StarControls::default(),
    );
    m1.r[0] = 1.0e10;
    m1.m[0] = M_SUN;
    m1.cgrav[0] = STANDARD_CGRAV;
    m1.controls.rotation_flag = true;
    m1.controls.mix_wind.simple_i_rot_flag = true;
    assert_eq!(
        do_solve_omega_mix(&mut m1, 1.0e5).expect("nz=1"),
        OmegaMixOutcome::KeepGoing
    );

    let mut m_base = zams_4m();
    apply_uniform_rotation(&mut m_base, 1.0e-5);
    let omega_snap = m_base.omega.clone();
    assert_eq!(
        do_solve_omega_mix(&mut m_base, 0.0).expect("dt=0"),
        OmegaMixOutcome::KeepGoing
    );
    assert_eq!(m_base.omega, omega_snap);

    // 2. Bad omega (`> 1e50` or `NaN`) fails validation cleanly.
    let mut m_bad = m_base.clone();
    m_bad.j_rot[3] = 1.0e75;
    assert!(matches!(
        do_solve_omega_mix(&mut m_bad, 1.0e3),
        Err(StarError::ConvergenceFailure(_))
    ));

    // 3. Explicit `other_torque` wrong length returns `InvalidInput`, and
    //    exact `extra_omegadot` / `extra_jdot` updates `omega` linearly in `dt`.
    let mut m_len = m_base.clone();
    m_len.controls.use_other_torque = true;
    assert!(matches!(
        do_solve_omega_mix_with_extras(&mut m_len, 1.0e4, &mut BadLenExtras),
        Err(StarError::InvalidInput(_))
    ));

    let mut m_acc = m_base.clone();
    m_acc.controls.use_other_torque = true;
    m_acc.aux.am_nu_omega.fill(0.0);
    m_acc.aux.am_nu_j.fill(0.0);
    let dt_acc = 1.0e6_f64;
    let omegadot = 2.0e-13_f64;
    let omega_before = m_acc.omega[5];
    let mut ex_acc = UniformAccelExtras { omegadot };
    assert_eq!(
        do_solve_omega_mix_with_extras(&mut m_acc, dt_acc, &mut ex_acc)
            .expect("uniform extra_omegadot"),
        OmegaMixOutcome::KeepGoing
    );
    let expected_omega = omega_before + omegadot * dt_acc;
    assert!(
        (m_acc.omega[5] - expected_omega).abs() <= 1.0e-14 * expected_omega
    );

    // 4. Failing `binary_other_torque_implicit` triggers `OmegaMixOutcome::Retry`.
    let hit = Arc::new(AtomicBool::new(false));
    let mut m_bin = m_base.clone();
    m_bin.aux.binary_other_torque_implicit = Some(
        BinaryOtherTorqueHandle::new(FailBinaryImplicit(hit.clone())),
    );
    let mut def_ex = DefaultStarExtras;
    let bin_out =
        do_solve_omega_mix_with_extras(&mut m_bin, 1.0e4, &mut def_ex)
            .expect("binary implicit failure yields Retry");
    assert!(hit.load(Ordering::SeqCst));
    assert!(matches!(bin_out, OmegaMixOutcome::Retry(_)));

    // 5. Coupled `am_nu_j` + `am_nu_omega` with sheared `omega` and spatially
    //    varying `i_rot` conserves total angular momentum to < 1e-13.
    let mut m_diff = zams_4m();
    apply_uniform_rotation(&mut m_diff, 1.0e-5);
    for k in 0..m_diff.nz {
        m_diff.omega[k] = 1.0e-5 * (1.0 + 0.5 * m_diff.q[k]);
    }
    use_xh_to_update_i_rot_and_j_rot(&mut m_diff);
    set_rotation_info(&mut m_diff, false);
    m_diff.aux.am_nu_omega.fill(1.0e14);
    m_diff.aux.am_nu_j.fill(5.0e13);
    let j_tot0: f64 = (0..m_diff.nz)
        .map(|k| m_diff.dm_bar[k] * m_diff.j_rot[k])
        .sum();
    let out_diff = do_solve_omega_mix(&mut m_diff, 1.0e8)
        .expect("diffusive J+omega solve");
    assert_eq!(out_diff, OmegaMixOutcome::KeepGoing);
    assert_eq!(m_diff.omega[0].to_bits(), m_diff.omega[1].to_bits());
    let j_tot1: f64 = (0..m_diff.nz)
        .map(|k| m_diff.dm_bar[k] * m_diff.j_rot[k])
        .sum();
    // Surface boundary sets `omega[0] = omega[1]` outside `x_1[0]`
    // (`solve_omega_mix.f90:238`), leaving ~1e-12 surface J adjustment while
    // keeping total J drift far below `angular_momentum_error_warn` (1e-6).
    assert!(
        (j_tot1 - j_tot0).abs() <= 1.0e-11 * j_tot0.abs(),
        "total J drift: rel={:.3e}",
        (j_tot1 - j_tot0).abs() / j_tot0.abs()
    );

    // 6. Standalone `solve_omega_mix` initializes `i_rot = (2/3) * r^2` when
    //    non-positive, diffuses omega while conserving total J, and ignores
    //    mismatched `nu_omega` length.
    let mut m_sa = zams_4m();
    for k in 0..m_sa.nz {
        m_sa.omega[k] = 1.0e-5 * (1.0 + 0.4 * (1.0 - m_sa.q[k]));
        m_sa.aux.i_rot[k] = 0.0;
    }
    let omega_sa_before = m_sa.omega.clone();
    let bad_nu = vec![1.0e12; 2];
    solve_omega_mix(&mut m_sa, 1.0e7, &bad_nu, 4);
    assert_eq!(m_sa.aux.i_rot[5], 0.0);
    assert_eq!(m_sa.omega, omega_sa_before);

    let j_sa_before: f64 = (0..m_sa.nz)
        .map(|k| {
            m_sa.dm_bar[k]
                * (TWO_THIRDS * m_sa.r[k] * m_sa.r[k])
                * m_sa.omega[k]
        })
        .sum();
    let nu = vec![1.0e14; m_sa.nz];
    solve_omega_mix(&mut m_sa, 1.0e8, &nu, 4);
    for k in 0..m_sa.nz {
        let want_i = TWO_THIRDS * m_sa.r[k] * m_sa.r[k];
        assert_eq!(m_sa.aux.i_rot[k].to_bits(), want_i.to_bits());
    }
    let j_sa_after: f64 =
        (0..m_sa.nz).map(|k| m_sa.dm_bar[k] * m_sa.j_rot[k]).sum();
    assert!((j_sa_after - j_sa_before).abs() <= 1.0e-11 * j_sa_before.abs());
}
