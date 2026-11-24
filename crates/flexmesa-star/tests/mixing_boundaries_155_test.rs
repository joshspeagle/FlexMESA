// SPDX-License-Identifier: GPL-3.0-or-later
//! Unit and integration tests for Issue `#155` (Audit C4):
//! thresholded composition-gradient smoothing, Brunt $B$/$N^2$, MLT call-site
//! gates, `mix_info`, `overshoot`, `predictive_mix`, and `conv_premix`.

#![forbid(unsafe_code)]

use flexmesa_const::{MixingType, M_SUN};
use flexmesa_star::conv_premix;
use flexmesa_star::mix_info::{
    clean_mixing_glitches, compute_convection_sigmas, set_mixing_info,
    set_mixing_info_with_skip,
};
use flexmesa_star::overshoot::add_overshooting;
use flexmesa_star::predictive_mix;
use flexmesa_star::{
    do_brunt_b, do_brunt_n2, load_photo_bytes, load_zams_model,
    locate_convection_boundaries, set_hydro_vars, set_mlt_vars,
    threshold_smoothing, update_sig_from_d_mix, weighted_smoothing,
    write_photo_bytes, StarControls, StarJobControls, StarModel,
};

fn zams_4m() -> StarModel {
    let ctrl = StarControls {
        initial_mass: 4.0,
        zbase: 0.02,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(ctrl, StarJobControls::default()).expect("4 Msun ZAMS");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model
}

/// Constructs an evolved 4 Msun core with a thick 20-cell H/He composition
/// gradient inside the original Schwarzschild core so `use_ledoux_criterion`
/// stabilizes a genuine multi-cell semiconvective/radiative gradient zone
/// above the convective core.
fn build_ledoux_core_gradient_4m() -> StarModel {
    let mut model = zams_4m();
    model.controls.use_ledoux_criterion = true;
    model.controls.mix_wind.num_cells_for_smooth_gradl_composition_term = 0;
    model.controls.mix_wind.num_cells_for_smooth_brunt_b = 0;
    let _ = locate_convection_boundaries(&mut model)
        .expect("locate_convection_boundaries");
    let k_bdy = model.aux.mixing.conv_bdy_loc[0];

    // Place a modest H/He gradient deep inside the convective core where
    // `gradr >> grada` so `gradr > grada` holds across the gradient while
    // `gradL = grada + B > gradr` stabilizes it under Ledoux.
    let sp = model.species;
    let idx_h1 = model.iso_names.iter().position(|s| s == "h1").expect("h1");
    let idx_he4 = model.iso_names.iter().position(|s| s == "he4").expect("he4");
    let k_grad_top = (k_bdy + 30).min(model.nz - 25);
    let k_grad_bot = (k_grad_top + 12).min(model.nz - 10);
    let span = f64::from(u32::try_from((k_grad_bot - k_grad_top).max(1)).expect("small span"));
    for k in k_grad_top..model.nz {
        let frac = if k < k_grad_bot {
            f64::from(u32::try_from(k - k_grad_top).expect("small offset"))
                / span
        } else {
            1.0
        };
        let delta = (0.015 * frac).min(model.xa[k * sp + idx_h1]);
        model.xa[k * sp + idx_h1] -= delta;
        model.xa[k * sp + idx_he4] += delta;
    }
    model.dt = 1.0e12;
    set_hydro_vars(&mut model, false, false).expect("refresh Ledoux gradient");
    let k_top = model.aux.mixing.conv_bdy_loc[0];
    assert!(
        model.gradr[k_top] > model.aux.mixing.grada_face[k_top]
            && model.gradr[k_top] <= model.gradl[k_top],
        "core top face k={k_top} must be Schwarzschild-unstable and Ledoux-stable: \
         gradr={}, grada={}, gradl={}",
        model.gradr[k_top],
        model.aux.mixing.grada_face[k_top],
        model.gradl[k_top]
    );
    model
}

#[test]
fn test_weighted_and_threshold_smoothing_match_binomial_weights_and_isolate_regions(
) {
    // 1. Exact binomial weights for ns = 1, 2, 3 on an interior impulse.
    for (ns, expected_row) in [
        (1_usize, vec![0.25, 0.5, 0.25]),
        (
            2_usize,
            vec![1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0],
        ),
        (
            3_usize,
            vec![
                1.0 / 64.0,
                6.0 / 64.0,
                15.0 / 64.0,
                20.0 / 64.0,
                15.0 / 64.0,
                6.0 / 64.0,
                1.0 / 64.0,
            ],
        ),
    ] {
        let n = 15_usize;
        let center = 7_usize;
        let mut dd = vec![0.0; n];
        let mut work = vec![0.0; n];
        dd[center] = 1.0;
        dd[0] = 3.5;
        dd[n - 1] = -2.25;
        weighted_smoothing(&mut dd, ns, false, &mut work);

        // Endpoints must remain strictly untouched (`star_utils.f90:2790`).
        assert_eq!(dd[0].to_bits(), 3.5_f64.to_bits());
        assert_eq!(dd[n - 1].to_bits(), (-2.25_f64).to_bits());
        for (offset, &w) in expected_row.iter().enumerate() {
            let idx = center - ns + offset;
            assert!(
                (dd[idx] - w).abs() < 1.0e-14,
                "ns={ns} idx={idx}: got={}, want={w}",
                dd[idx]
            );
        }
    }

    // 2. `preserve_sign = true` keeps original value when smoothing would flip sign
    //    (`star_utils.f90:2820`).
    let mut signed = vec![0.0, 10.0, 10.0, -0.01, 10.0, 10.0, 0.0];
    let mut work_s = vec![0.0; signed.len()];
    weighted_smoothing(&mut signed, 1, true, &mut work_s);
    assert_eq!(
        signed[3].to_bits(),
        (-0.01_f64).to_bits(),
        "preserve_sign must retain original value when smoothed value flips sign"
    );

    // 3. `threshold_smoothing` isolates regions above `thresh` with zero leakage.
    let mut arr = vec![
        0.0, 0.0, 1.0e-2, 2.0e-2, 1.0e-2, 1.0e-5, 0.0, 1.0e-5, 5.0e-2, 1.0e-1,
        5.0e-2, 0.0, 0.0,
    ];
    let mut work_t = vec![0.0; arr.len()];
    let mut ref_left = arr[2..=4].to_vec();
    let mut wl = vec![0.0; ref_left.len()];
    weighted_smoothing(&mut ref_left, 1, false, &mut wl);
    let mut ref_right = arr[8..=10].to_vec();
    let mut wr = vec![0.0; ref_right.len()];
    weighted_smoothing(&mut ref_right, 1, false, &mut wr);

    threshold_smoothing(&mut arr, 1.0e-3, 1, false, &mut work_t);
    assert_eq!(arr[5].to_bits(), 1.0e-5_f64.to_bits());
    assert_eq!(arr[6].to_bits(), 0.0_f64.to_bits());
    assert_eq!(arr[7].to_bits(), 1.0e-5_f64.to_bits());
    for (i, &want) in ref_left.iter().enumerate() {
        assert!((arr[2 + i] - want).abs() < 1.0e-15);
    }
    for (i, &want) in ref_right.iter().enumerate() {
        assert!((arr[8 + i] - want).abs() < 1.0e-15);
    }

    // 4. `thresh = 0.0` matches full-array `weighted_smoothing`.
    let mut a1 = vec![0.1, 0.4, 0.2, 0.9, 0.3, 0.5, 0.1];
    let mut a2 = a1.clone();
    let mut w1 = vec![0.0; a1.len()];
    let mut w2 = vec![0.0; a2.len()];
    threshold_smoothing(&mut a1, 0.0, 2, false, &mut w1);
    weighted_smoothing(&mut a2, 2, false, &mut w2);
    for (x, y) in a1.iter().zip(a2.iter()) {
        assert!((x - y).abs() < 1.0e-15);
    }
}

#[test]
fn test_brunt_b_preserves_unsmoothed_b_and_brunt_n2_respects_gates() {
    let mut model = zams_4m();
    let sp = model.species;
    let idx_h1 = model.iso_names.iter().position(|s| s == "h1").expect("h1");
    let idx_he4 = model.iso_names.iter().position(|s| s == "he4").expect("he4");

    // Inject a sharp step in He inside the interior so `brunt_B` has a sharp peak.
    let k_step = model.nz / 2;
    for k in k_step..model.nz {
        let d = 0.2_f64.min(model.xa[k * sp + idx_h1]);
        model.xa[k * sp + idx_h1] -= d;
        model.xa[k * sp + idx_he4] += d;
    }
    model.controls.calculate_brunt_b = true;
    model.controls.calculate_brunt_n2 = true;
    model.controls.mix_wind.num_cells_for_smooth_brunt_b = 2;
    model.controls.mix_wind.threshold_for_smooth_brunt_b = 0.0;
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");

    let peak_unsmoothed = model.aux.mixing.unsmoothed_brunt_b[k_step];
    let peak_smoothed = model.brunt_b[k_step];
    assert!(
        peak_unsmoothed > 0.0,
        "unsmoothed_brunt_b must capture sharp composition jump"
    );
    assert!(
        peak_smoothed < peak_unsmoothed,
        "threshold_smoothing must smooth brunt_b while preserving unsmoothed_brunt_b"
    );

    // `brunt_n2_coefficient` scales both `brunt_n2` and `brunt_n2_composition_term`.
    let n2_orig = model.brunt_n2[k_step];
    let n2_comp_orig = model.aux.brunt_n2_composition_term[k_step];
    assert!(n2_orig.is_finite() && n2_comp_orig > 0.0);
    model.controls.mix_wind.brunt_n2_coefficient = 2.5;
    do_brunt_n2(&mut model);
    assert!(
        (model.brunt_n2[k_step] - 2.5 * n2_orig).abs()
            <= 1.0e-12 * n2_orig.abs().max(1.0)
    );
    assert!(
        (model.aux.brunt_n2_composition_term[k_step] - 2.5 * n2_comp_orig).abs()
            <= 1.0e-12 * n2_comp_orig.abs().max(1.0)
    );

    // `min_magnitude_brunt_b` clips sub-threshold `brunt_B` and `brunt_N2` to 0.
    model.controls.mix_wind.brunt_n2_coefficient = 1.0;
    model.controls.mix_wind.min_magnitude_brunt_b = peak_smoothed * 2.0;
    do_brunt_b(&mut model).expect("do_brunt_b");
    do_brunt_n2(&mut model);
    assert_eq!(model.brunt_b[k_step], 0.0);
    assert_eq!(model.brunt_n2[k_step], 0.0);
}

#[test]
fn test_mlt_call_site_gates_t_tau_gradr_and_mlt_vs_mixing_separation() {
    let mut model = zams_4m();

    // 1. Default `mlt_make_surface_no_mixing = false` and `use_t_tau_gradr_factor = false`
    //    (F2-19 removed). Enabling `mlt_make_surface_no_mixing = true` forces face 0
    //    to NoMixing with `gradt[0] == gradr[0]`.
    assert!(!model.controls.mix_wind.mlt_make_surface_no_mixing);
    assert!(!model.controls.mix_wind.use_t_tau_gradr_factor);
    assert_eq!(model.aux.mixing.gradr_factor[0], 1.0);
    let mut m_surf = model.clone();
    m_surf.controls.mix_wind.mlt_make_surface_no_mixing = true;
    set_mlt_vars(&mut m_surf).expect("set_mlt_vars surface no mixing");
    assert_eq!(m_surf.aux.mixing.mlt_mixing_type[0], MixingType::NoMixing);
    assert_eq!(m_surf.gradt[0].to_bits(), m_surf.gradr[0].to_bits());

    // 2. `use_t_tau_gradr_factor = true` scales `gradr` by `1 + dQ/dtau`.
    let gradr0_plain = model.gradr[0];
    let mut m_ttau = model.clone();
    m_ttau.controls.mix_wind.use_t_tau_gradr_factor = true;
    set_hydro_vars(&mut m_ttau, false, false).expect("set_hydro_vars t_tau");
    let expected_fac = 1.0
        + flexmesa_atm::eval_t_tau_dq_dtau(
            m_ttau.controls.atm_t_tau_relation.id(),
            m_ttau.tau[0],
        )
        .expect("t_tau dq_dtau");
    assert!((m_ttau.aux.mixing.gradr_factor[0] - expected_fac).abs() < 1.0e-14);
    assert!(
        (m_ttau.gradr[0] - gradr0_plain * expected_fac).abs()
            <= 1.0e-12 * gradr0_plain.abs().max(1.0)
    );

    // 3. `mlt_gradt_fraction` blends `gradt = f * grada_face + (1 - f) * gradr`.
    let mut m_gf = model.clone();
    m_gf.controls.mix_wind.mlt_gradt_fraction = 0.35;
    set_mlt_vars(&mut m_gf).expect("set_mlt_vars");
    let k_mid = m_gf.nz / 2;
    let want_gt = 0.35 * m_gf.aux.mixing.grada_face[k_mid]
        + 0.65 * m_gf.gradr[k_mid];
    assert!((m_gf.gradt[k_mid] - want_gt).abs() < 1.0e-13);

    // 4. `set_mlt_vars` and `set_hydro_vars` with `okay_to_set_mixing_info = false`
    //    must NOT clobber `mixing_type`, `d_mix`, or `conv_vel`.
    let k_rad = 10_usize.min(model.nz - 2);
    model.mixing_type[k_rad] = MixingType::OvershootMixing;
    model.d_mix[k_rad] = 1.2345e11;
    model.conv_vel[k_rad] = 9.876e3;
    model.aux.mixing.okay_to_set_mixing_info = false;
    set_hydro_vars(&mut model, false, false).expect("hydro vars skip mixing");
    assert_eq!(model.mixing_type[k_rad], MixingType::OvershootMixing);
    assert_eq!(model.d_mix[k_rad].to_bits(), 1.2345e11_f64.to_bits());
    assert_eq!(model.conv_vel[k_rad].to_bits(), 9.876e3_f64.to_bits());
}

#[test]
fn test_mix_info_glitches_prune_bad_cz_min_d_mix_and_high_tc_sigma_taper() {
    let mut model = zams_4m();
    let nz = model.nz;

    // 1. `remove_tiny_mixing` clears tiny `d_mix` without altering `gradt`.
    let k_tiny = 15_usize;
    let gradt_before = model.gradt[k_tiny];
    model.mixing_type[k_tiny] = MixingType::ConvectiveMixing;
    model.d_mix[k_tiny] = 1.0e-12;
    model.conv_vel[k_tiny] = 1.0;
    clean_mixing_glitches(&mut model, 1.0e-5, 0.0);
    assert_eq!(model.mixing_type[k_tiny], MixingType::NoMixing);
    assert_eq!(model.d_mix[k_tiny], 0.0);
    assert_eq!(model.gradt[k_tiny].to_bits(), gradt_before.to_bits());

    // 2. `set_min_d_mix = true` + `min_d_mix` floors interior diffusivity and
    //    sets `MinimumMixing`.
    let mut m_min = zams_4m();
    m_min.controls.mix_wind.set_min_d_mix = true;
    m_min.controls.min_d_mix = 1.0e4;
    set_mixing_info(&mut m_min).expect("set_mixing_info min_d_mix");
    let k_rad = 12_usize;
    assert_eq!(m_min.mixing_type[k_rad], MixingType::MinimumMixing);
    assert!(m_min.d_mix[k_rad] >= 1.0e4);

    // 3. `d_mix_zero_region_top_q` / `bottom_q` zeroes `d_mix` inside [bot, top].
    let mut m_zero = m_min.clone();
    m_zero.controls.mix_wind.d_mix_zero_region_bottom_q = 0.4;
    m_zero.controls.mix_wind.d_mix_zero_region_top_q = 0.6;
    set_mixing_info(&mut m_zero).expect("set_mixing_info zero region");
    for k in 1..nz {
        if m_zero.q[k] >= 0.4 && m_zero.q[k] <= 0.6 {
            assert_eq!(m_zero.d_mix[k], 0.0);
            assert_eq!(m_zero.mixing_type[k], MixingType::NoMixing);
        }
    }

    // 4. `sig_min_factor_for_high_tcenter` tapers `sig` at high T_c (Kelvin)
    //    near mixing boundaries when composition gradients are present
    //    (`mix_info.f90:1447-1555`).
    let mut m_hi = zams_4m();
    let k_core = nz - 5;
    let sp = m_hi.species;
    let idx_h1 = m_hi.iso_names.iter().position(|s| s == "h1").expect("h1");
    let idx_he4 = m_hi.iso_names.iter().position(|s| s == "he4").expect("he4");
    m_hi.xa[(k_core - 1) * sp + idx_h1] += 0.1;
    m_hi.xa[(k_core - 1) * sp + idx_he4] -= 0.1;
    m_hi.dt = 1.0e12;
    m_hi.controls.mix_wind.sig_min_factor_for_high_tcenter = 1.0e-4;
    m_hi.controls.mix_wind.tcenter_max_for_sig_min_factor_full_off = 1.0e6;
    m_hi.controls.mix_wind.tcenter_min_for_sig_min_factor_full_on = 1.0e7;
    m_hi.controls.mix_wind.delta_m_lower_for_sig_min_factor = 1.0e35;
    m_hi.controls.mix_wind.delta_m_upper_for_sig_min_factor = 2.0e35;
    m_hi.controls.mix_wind.max_delta_m_to_bdy_for_sig_min_factor = 1.0;
    m_hi.controls.sig_term_limit = 1.0e99;
    let mut m_lo = m_hi.clone();
    m_lo.controls.mix_wind.sig_min_factor_for_high_tcenter = 1.0;
    update_sig_from_d_mix(&mut m_hi).unwrap();
    update_sig_from_d_mix(&mut m_lo).unwrap();
    assert!(m_lo.sig[k_core] > 0.0);
    assert!(
        (m_hi.sig[k_core] / m_lo.sig[k_core] - 1.0e-4).abs() < 1.0e-6,
        "high-Tc taper ratio got {}",
        m_hi.sig[k_core] / m_lo.sig[k_core]
    );
}

#[test]
fn test_overshoot_cospi_ramp_other_scheme_error_and_unsmoothed_brunt_b() {
    let mut model = zams_4m();
    model.mstar = 4.0 * M_SUN;

    // 1. Exponential overshoot populates `OvershootMixing` above the core.
    let ov = &mut model.controls.mix_wind.overshoot_sets[0];
    ov.scheme = String::from("exponential");
    ov.zone_type = String::from("any");
    ov.zone_loc = String::from("core");
    ov.bdy_loc = String::from("top");
    ov.f = 0.016;
    ov.f0 = 0.004;
    ov.mass_full_off = 3.0;
    ov.mass_full_on = 5.0; // M = 4.0 is exact midpoint -> cospi ramp = 0.5

    set_mixing_info(&mut model).expect("set_mixing_info with cospi overshoot");
    let n_ov_mid = model
        .mixing_type
        .iter()
        .filter(|&&mt| mt == MixingType::OvershootMixing)
        .count();
    assert!(
        n_ov_mid > 0,
        "cospi midpoint must still produce overshoot faces"
    );

    // 2. Below `mass_full_off`, cospi ramp zeroes overshoot completely.
    let mut m_off = zams_4m();
    m_off.mstar = 4.0 * M_SUN;
    let ov_off = &mut m_off.controls.mix_wind.overshoot_sets[0];
    ov_off.scheme = String::from("exponential");
    ov_off.zone_type = String::from("any");
    ov_off.zone_loc = String::from("core");
    ov_off.bdy_loc = String::from("top");
    ov_off.f = 0.016;
    ov_off.f0 = 0.004;
    ov_off.mass_full_off = 5.0;
    ov_off.mass_full_on = 6.0;
    set_mixing_info(&mut m_off).expect("set_mixing_info ramp off");
    let n_ov_off = m_off
        .mixing_type
        .iter()
        .filter(|&&mt| mt == MixingType::OvershootMixing)
        .count();
    assert_eq!(n_ov_off, 0, "mass <= mass_full_off must disable overshoot");

    // 3. `overshoot_scheme = "other"` without a user hook must return an error
    //    matching MESA `26.4.1` `null_other_overshooting_scheme`.
    let mut m_other = zams_4m();
    let ov_other = &mut m_other.controls.mix_wind.overshoot_sets[0];
    ov_other.scheme = String::from("other");
    ov_other.zone_type = String::from("any");
    ov_other.zone_loc = String::from("core");
    ov_other.bdy_loc = String::from("top");
    ov_other.f = 0.016;
    ov_other.f0 = 0.004;
    assert!(
        add_overshooting(&mut m_other).is_err(),
        "'other' overshoot scheme without hook must fail loudly"
    );
}

#[test]
fn test_predictive_mix_trial_schwarzschild_restore_and_stop_controls() {
    let mut model = build_ledoux_core_gradient_4m();
    let xa_before = model.xa.clone();
    let lnd_before = model.lnd.clone();
    let kap_before = model.opacity.clone();

    let ps0 = &mut model.controls.mix_wind.predictive_sets[0];
    ps0.predictive_mix = true;
    ps0.zone_type = String::from("any");
    ps0.zone_loc = String::from("core");
    ps0.bdy_loc = String::from("top");
    ps0.superad_thresh = 0.0;

    let n_ext =
        predictive_mix::add_predictive_mixing(&mut model).expect("pred mix");
    assert!(
        n_ext > 0,
        "verbatim predictive_mix must expand Ledoux-stabilized convective core; \
         bdy_loc={:?} top={:?} q={:?} r_center={} n_ext={n_ext}",
        model.aux.mixing.conv_bdy_loc,
        model.aux.mixing.top_conv_bdy,
        model.aux.mixing.conv_bdy_q,
        model.r_center
    );

    // Cell compositions, densities, and opacities must be restored bit-exactly.
    assert_eq!(model.xa, xa_before);
    assert_eq!(model.lnd, lnd_before);
    assert_eq!(model.opacity, kap_before);

    // Enabling both `do_conv_premix` and `predictive_mix` must error loudly.
    let mut m_conflict = model.clone();
    m_conflict.controls.do_conv_premix = true;
    assert!(predictive_mix::add_predictive_mixing(&mut m_conflict).is_err());

    // Invalid selector string must error loudly.
    let mut m_bad = model.clone();
    m_bad.controls.mix_wind.predictive_sets[0].zone_type =
        String::from("invalid_zone");
    assert!(predictive_mix::add_predictive_mixing(&mut m_bad).is_err());
}

#[test]
fn test_conv_premix_ingestion_conservation_time_factor_and_fix_pgas_modes() {
    let base = build_ledoux_core_gradient_4m();

    let species_mass = |m: &StarModel| -> Vec<f64> {
        let mut sums = vec![0.0; m.species];
        for k in 0..m.nz {
            for l in 0..m.species {
                sums[l] += m.dm[k] * m.xa[k * m.species + l];
            }
        }
        sums
    };
    let mass_before = species_mass(&base);

    // 1. Tiny `conv_premix_time_factor` stops advancement at 0 cells.
    let mut m_zero_time = base.clone();
    m_zero_time.controls.mix_wind.conv_premix_time_factor = 1.0e-20;
    let n_zero = conv_premix::do_conv_premix(&mut m_zero_time)
        .expect("conv_premix tiny time_factor");
    assert_eq!(n_zero, 0, "tiny conv_premix_time_factor must block advance");

    // 2. Unbounded clock (`time_factor = -1.0`), `fix_pgas = true`: advances
    //    into Ledoux-stabilized region, conserves mass of every isotope, and
    //    preserves gas pressure across modified cells.
    let mut fixed_pgas = base.clone();
    fixed_pgas.controls.mix_wind.conv_premix_time_factor = -1.0;
    fixed_pgas.controls.mix_wind.conv_premix_fix_pgas = true;
    fixed_pgas.controls.mix_wind.conv_premix_avoid_increase = false;
    let pgas_before = fixed_pgas.pgas.clone();
    let absorbed_pgas = conv_premix::do_conv_premix(&mut fixed_pgas)
        .expect("conv_premix PT");
    assert!(
        absorbed_pgas > 0,
        "conv_premix must absorb Ledoux-stabilized cells"
    );
    let mass_after_pgas = species_mass(&fixed_pgas);
    for (a, b) in mass_before.iter().zip(mass_after_pgas.iter()) {
        assert!(
            (a - b).abs() <= 1.0e-12 * a.abs().max(1.0e20),
            "species mass drift: before={a:.12e}, after={b:.12e}"
        );
    }
    for k in 0..fixed_pgas.nz {
        let rel_dp = (fixed_pgas.pgas[k] - pgas_before[k]).abs()
            / pgas_before[k].abs().max(1.0);
        assert!(
            rel_dp < 1.0e-6,
            "fix_pgas=true must hold Pgas fixed at cell {k}: rel_dp={rel_dp:.3e}"
        );
    }

    // 3. `fix_pgas = false`: holds density (`lnd`) bit-identical on all cells.
    let mut fixed_rho = base.clone();
    fixed_rho.controls.mix_wind.conv_premix_time_factor = -1.0;
    fixed_rho.controls.mix_wind.conv_premix_fix_pgas = false;
    fixed_rho.controls.mix_wind.conv_premix_avoid_increase = false;
    let lnd_before = fixed_rho.lnd.clone();
    let absorbed_rho = conv_premix::do_conv_premix(&mut fixed_rho)
        .expect("conv_premix DT");
    assert!(absorbed_rho > 0);
    assert_eq!(
        fixed_rho.lnd, lnd_before,
        "fix_pgas=false must keep lnd bit-identical"
    );
}

#[test]
fn test_find0_clamps_to_bracket_on_steep_overshoot_boundary() {
    // Reproduces the +1 ULP overshoot case from MESA `num_lib.f90:530-533`.
    let dq = 0.000_435_829_416_274_917_85_f64;
    let d_val = 7.034_117_503_949_756e15_f64;
    let d_cutoff = 1.0e-2_f64;
    let x0 = flexmesa_num::find0(0.0, d_val - d_cutoff, dq, -d_cutoff)
        .expect("find0 steep bracket");
    assert!(
        (0.0..=dq).contains(&x0),
        "find0 must clamp x0={x0:.17e} within [0, {dq:.17e}]"
    );
}

#[test]
fn test_retry_rollback_preserves_alpha_mlt_and_boundaries_across_nz_change() {
    use flexmesa_star::{compute_convection_sigmas, PrimaryStateBuffer};

    let mut model = zams_4m();
    let orig_nz = model.nz;
    let orig_alpha = model.aux.mixing.alpha_mlt.clone();
    let orig_mix_bdy = model.aux.mixing.mix_bdy_loc.clone();
    assert_ne!(orig_mix_bdy.len(), 0);

    let buf = PrimaryStateBuffer::from_model(&model);

    // Simulate a failed attempt that split zones (`nz` grew) and left
    // boundary indices beyond `orig_nz`, then rolled back.
    model.resize_zones(orig_nz + 40);
    model.aux.mixing.mix_bdy_loc.push(orig_nz + 25);
    buf.restore_into(&mut model).expect("rollback after grow");
    assert_eq!(model.nz, orig_nz);
    assert_eq!(model.aux.mixing.mix_bdy_loc, orig_mix_bdy);
    assert!(model.aux.mixing.mix_bdy_loc.iter().all(|&k| k < orig_nz));

    // Trigger `Tc > 2.8e9` path in `compute_convection_sigmas`; must not OOB.
    model.t[orig_nz - 1] = 3.0e9;
    compute_convection_sigmas(&mut model, 1.0e8, 1.0, 1.0e10).unwrap();

    // Simulate a failed attempt that merged zones (`nz` shrank) then rolled
    // back; `alpha_mlt` in the restored tail must match pre-attempt values.
    model.resize_zones(orig_nz - 20);
    buf.restore_into(&mut model).expect("rollback after shrink");
    assert_eq!(model.aux.mixing.alpha_mlt, orig_alpha);
}

#[test]
fn test_set_mixing_info_rotation_flag_clips_d_mix_and_refreshes_cdc() {
    let mut model = zams_4m();
    model.controls.rotation_flag = true;
    model.controls.mix_wind.am_d_mix_factor = 0.0;
    model.controls.mix_wind.d_visc_factor = 0.0;
    model.controls.mix_wind.set_min_d_mix = true;
    model.controls.min_d_mix = 1.0e-12;
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");

    // Upstream `mix_info.f90:287-301`: even with `am_d_mix_factor = 0`,
    // `rotation_flag` clips `D_mix < 1e-10` to `0.0` and refreshes `cdc`.
    for k in 1..model.nz {
        assert!(
            model.d_mix[k] == 0.0 || model.d_mix[k] >= 1.0e-10,
            "face {k} retained sub-threshold d_mix={}",
            model.d_mix[k]
        );
        if model.d_mix[k] == 0.0 {
            assert_eq!(model.aux.mixing.cdc[k], 0.0);
            assert_ne!(model.mixing_type[k], MixingType::RotationMixing);
        }
    }
    assert_eq!(model.aux.mixing.cdc[0], model.aux.mixing.cdc[1]);
}

#[test]
fn test_premix_add_predictive_mixing_fails_loudly_on_conv_premix_conflict() {
    let mut model = zams_4m();
    model.controls.do_conv_premix = true;
    let err = flexmesa_star::add_predictive_mixing(&mut model, 0.1, 0.005)
        .expect_err("must fail loudly when do_conv_premix is active");
    assert!(
        err.to_string().contains("convective premixing"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_locate_convection_boundaries_matches_set_mixing_info() {
    let mut model = zams_4m();
    let expected_loc = model.aux.mixing.conv_bdy_loc.clone();
    let expected_top = model.aux.mixing.top_conv_bdy.clone();
    let expected_dq = model.aux.mixing.cz_bdy_dq.clone();
    let n_bdy = locate_convection_boundaries(&mut model)
        .expect("locate_convection_boundaries");
    assert_eq!(n_bdy, expected_loc.len());
    assert_eq!(model.aux.mixing.conv_bdy_loc, expected_loc);
    assert_eq!(model.aux.mixing.top_conv_bdy, expected_top);
    assert_eq!(model.aux.mixing.cz_bdy_dq, expected_dq);
}

#[test]
fn test_overshoot_selectors_compare_case_sensitively_matching_upstream() {
    // Exact MESA 26.4.1 case (`'burn_H'`, `'core'`, `'top'`, `'exponential'`)
    // succeeds (`star/private/overshoot.f90:101-177`).
    let mut m_ok = zams_4m();
    let ov = &mut m_ok.controls.mix_wind.overshoot_sets[0];
    ov.scheme = String::from("exponential");
    ov.zone_type = String::from("burn_H");
    ov.zone_loc = String::from("core");
    ov.bdy_loc = String::from("top");
    ov.f = 0.016;
    ov.f0 = 0.004;
    add_overshooting(&mut m_ok).expect("exact MESA case must succeed");

    // Wrong case on any selector must return `Err(StarError::InvalidInput)`.
    for (label, scheme, zone_type, zone_loc, bdy_loc) in [
        ("zone_type", "exponential", "burn_h", "core", "top"),
        ("zone_loc", "exponential", "burn_H", "CORE", "top"),
        ("bdy_loc", "exponential", "burn_H", "core", "Top"),
        ("scheme", "Exponential", "burn_H", "core", "top"),
        ("none_case", "NONE", "any", "any", "any"),
    ] {
        let mut m_bad = zams_4m();
        let s0 = &mut m_bad.controls.mix_wind.overshoot_sets[0];
        s0.scheme = String::from(scheme);
        s0.zone_type = String::from(zone_type);
        s0.zone_loc = String::from(zone_loc);
        s0.bdy_loc = String::from(bdy_loc);
        s0.f = 0.016;
        s0.f0 = 0.004;
        assert!(
            add_overshooting(&mut m_bad).is_err(),
            "wrong case on {label} must fail loudly"
        );
    }
}

#[test]
fn test_set_mixing_info_skip_cz_bdy_mass_and_have_new_generation() {
    let mut model = zams_4m();
    model.aux.mixing.cz_top_mass.clear();
    model.aux.mixing.cz_bot_mass.clear();
    model.aux.mixing.have_new_cz_bdy_info = false;

    set_mixing_info_with_skip(&mut model, true)
        .expect("set_mixing_info_with_skip(true)");
    assert!(!model.aux.mixing.have_new_cz_bdy_info);
    assert_eq!(model.aux.mixing.cz_top_mass.len(), 0);

    // `hydro_vars.f90:81`: `skip_set_cz_bdy_mass = .not. have_new_generation`.
    model.aux.mixing.have_new_generation = false;
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    assert!(!model.aux.mixing.have_new_cz_bdy_info);
    assert_eq!(model.aux.mixing.cz_top_mass.len(), 0);

    model.aux.mixing.have_new_generation = true;
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    assert!(model.aux.mixing.have_new_cz_bdy_info);
    assert_ne!(model.aux.mixing.cz_top_mass.len(), 0);

    // `FMPHOTO2` v27 wire compatibility: `have_new_generation` is skipped in
    // serialization and deserializes back to `true` via
    // `default_have_new_generation`.
    let bytes = write_photo_bytes(&model);
    let restored = load_photo_bytes(&bytes).expect("load_photo_bytes");
    assert!(restored.aux.mixing.have_new_generation);
}

#[test]
fn test_set_mixing_info_rsp2_and_rti_branches_match_upstream() {
    // 1. RSP2 branch (`mix_info.f90:126-142`): computes `conv_vel`, `d_mix`,
    //    `cdc`, and mixing_type thresholds from `et_rsp`/`hp_face`/`lt`/`lc`.
    let mut m_rsp2 = zams_4m();
    let nz = m_rsp2.nz;
    m_rsp2.controls.rsp2_flag = true;
    m_rsp2.aux.rsp.et_rsp = vec![1.0e10; nz];
    m_rsp2.aux.rsp.hp_face = vec![1.0e9; nz];
    m_rsp2.aux.rsp.lc_rsp = vec![0.0; nz];
    m_rsp2.aux.rsp.lt_rsp = vec![0.0; nz];
    let k_overshoot = nz / 3;
    let k_conv = nz / 2;
    let lum_overshoot = m_rsp2.lum[k_overshoot].abs().max(1.0);
    let lum_conv = m_rsp2.lum[k_conv].abs().max(1.0);
    m_rsp2.aux.rsp.lt_rsp[k_overshoot] = 1.0e-2 * lum_overshoot;
    m_rsp2.aux.rsp.lc_rsp[k_conv] = 1.0e-2 * lum_conv;
    set_mixing_info(&mut m_rsp2).expect("set_mixing_info rsp2");
    assert_eq!(m_rsp2.mixing_type[k_overshoot], MixingType::OvershootMixing);
    assert_eq!(m_rsp2.mixing_type[k_conv], MixingType::ConvectiveMixing);
    assert!(m_rsp2.d_mix[k_conv] > 0.0);

    // Unsized RSP2 arrays must fail loudly rather than silently skipping.
    let mut m_bad_rsp2 = zams_4m();
    m_bad_rsp2.controls.rsp2_flag = true;
    m_bad_rsp2.aux.rsp.et_rsp.clear();
    assert!(set_mixing_info(&mut m_bad_rsp2).is_err());

    // 2. RTI branch (`mix_info.f90:79-84, 230-231, 2373-2436`): runs
    //    `set_rti_mixing_info`, `set_dpdr_drhodr_info`, and `add_rti_turbulence`.
    let mut m_rti = zams_4m();
    let nz_r = m_rti.nz;
    m_rti.controls.rti_flag = true;
    m_rti.controls.composition_rti_diffusion_factor = 1.0;
    m_rti.aux.alpha_rti = vec![0.05; nz_r];
    m_rti.aux.alpha_rti_start = vec![0.05; nz_r];
    let k_rad = (2..(nz_r - 2))
        .find(|&k| m_rti.aux.mixing.mlt_mixing_type[k] == MixingType::NoMixing)
        .expect("radiative interior face");
    set_mixing_info(&mut m_rti).expect("set_mixing_info rti");
    assert!(m_rti.aux.eta_rti[k_rad] > 0.0);
    assert!(m_rti.d_mix[k_rad] >= m_rti.aux.eta_rti[k_rad]);
    assert_eq!(m_rti.mixing_type[k_rad], MixingType::RayleighTaylorMixing);
}

#[test]
fn test_compute_convection_sigmas_errors_on_inconsistent_mix_boundaries() {
    let make_high_tc = || {
        let mut m = zams_4m();
        let nz = m.nz;
        m.t[nz - 1] = 1.0e9;
        m.controls.mix_wind.sig_min_factor_for_high_tcenter = 1.0e-4;
        m.controls.mix_wind.tcenter_max_for_sig_min_factor_full_off = 1.0e6;
        m.controls.mix_wind.tcenter_min_for_sig_min_factor_full_on = 1.0e7;
        m
    };

    // Case 1 (`mix_info.f90:1474`): bottom boundary with no matching top.
    let mut m1 = make_high_tc();
    m1.aux.mixing.mix_bdy_loc = vec![m1.nz / 2];
    m1.aux.mixing.top_mix_bdy = vec![false];
    let err1 = compute_convection_sigmas(&mut m1, 1.0e8, 1.0, 1.0e10)
        .expect_err("bad mix bdy info 1 must error");
    assert!(err1.to_string().contains("bad mix bdy info 1"), "{err1}");

    // Case 2 (`mix_info.f90:1482`): consecutive top boundaries.
    let mut m2 = make_high_tc();
    m2.aux.mixing.mix_bdy_loc = vec![m2.nz / 2, m2.nz / 3];
    m2.aux.mixing.top_mix_bdy = vec![true, true];
    let err2 = compute_convection_sigmas(&mut m2, 1.0e8, 1.0, 1.0e10)
        .expect_err("bad mix bdy info 2 must error");
    assert!(err2.to_string().contains("bad mix bdy info 2"), "{err2}");

    // Case 3 (`mix_info.f90:1488`): second bottom boundary without matching top.
    let mut m3 = make_high_tc();
    m3.aux.mixing.mix_bdy_loc = vec![m3.nz / 2, m3.nz / 3];
    m3.aux.mixing.top_mix_bdy = vec![true, false];
    let err3 = compute_convection_sigmas(&mut m3, 1.0e8, 1.0, 1.0e10)
        .expect_err("bad mix bdy info 3 must error");
    assert!(err3.to_string().contains("bad mix bdy info 3"), "{err3}");
}
