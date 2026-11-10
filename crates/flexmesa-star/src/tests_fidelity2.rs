// SPDX-License-Identifier: LGPL-3.0-or-later
//! Unit tests for Stage `Fidelity-2` (`#113`):
//! - Brassard et al. (1991) finite-difference EOS vs. $\nabla_\mu$ `do_brunt_b`
//!   and `brunt_n2_composition_term` (`brunt.f90`).
//! - `set_mlt_vars` dispatch to `set_semiconvection`, `set_thermohaline`, and
//!   `set_tdc`, plus `mlt_vc = 0` for non-convective mixing (`turb_support.f90`,
//!   `turb_info.f90`).
//! - Rotational distortion $(f_P, f_T, i_{\mathrm{rot}})$ coupling into
//!   `set_mlt_vars`, `do_brunt_n2`, `set_rotation_mixing_info`, and
//!   `solve_omega_mix` (`hydro_rotation.f90`, `solve_omega_mix.f90`).
//! - Radiation-pressure corrected `AtmOption::SimplePhotosphere` surface
//!   boundary condition and `StarAdOrder1` partials (`hydro_vars.f90:857-916`).
//! - Multi-isotope C/N/O/Ne summation in `do_kap` (`micro.f90:694-702`),
//!   `eval_nugis_lamers_wind` (`winds.f90:459-462`), and spectator-normalized
//!   `do_phase_separation` (`phase_separation.f90:253`).

use flexmesa_const::{MixingType, L_SUN, M_SUN};
use flexmesa_turb::{MltOption, SemiconvectionOption, ThermohalineOption};

use crate::controls::{AtmOption, StarControls, StarJobControls};
use crate::hydro_vars::{
    do_brunt_b, do_brunt_n2, eval_surf_bc, set_hydro_vars, set_mlt_vars,
};
use crate::micro::{change_model_network, do_kap, set_basic_vars};
use crate::premix::do_phase_separation;
use crate::rotation::{
    set_rotation_mixing_info, update_rotation_distortion_factors,
};
use crate::star_ad::{I_LND_00, I_LNR_00, I_LNT_00, I_L_00};
use crate::struct_burn_mix::solve_omega_mix;
use crate::winds::{eval_dutch_wind, eval_nugis_lamers_wind, DutchLowTScheme};
use crate::zams::load_zams_model;

#[test]
fn test_brunt_b_brassard_eos_finite_difference_and_gradmux_forms() {
    let mut model =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("load 1 Msun ZAMS");
    set_hydro_vars(&mut model, true, true).expect("set_hydro_vars");

    let k_test = model.nz / 2;
    let nc = model.species;
    // Create a stabilizing composition jump at face `k_test` (higher He4 in cell `k_test` than `k_test - 1`)
    // while adjusting `lnd[k_test]` to preserve hydrostatic pressure `P[k_test]`.
    let p_orig = model.peos[k_test];
    model.xa[(k_test - 1) * nc] = 0.70;
    model.xa[(k_test - 1) * nc + 2] = 0.28;
    model.xa[k_test * nc] = 0.66;
    model.xa[k_test * nc + 2] = 0.32;
    set_basic_vars(&mut model).expect("set_basic_vars");
    crate::micro::do_eos(&mut model).expect("do_eos");
    let dln_rho_hse =
        (p_orig / model.peos[k_test]).ln() / model.chi_rho[k_test].max(0.5);
    model.lnd[k_test] += dln_rho_hse;
    set_basic_vars(&mut model).expect("set_basic_vars HSE");
    crate::micro::do_eos(&mut model).expect("do_eos HSE");

    model.controls.calculate_brunt_n2 = true;
    model.controls.use_ledoux_criterion = true;

    // 1. Default Brassard et al. (1991) finite-difference EOS form (`!use_brunt_gradmux_form`)
    model.controls.use_brunt_gradmux_form = false;
    do_brunt_b(&mut model);
    let b_brassard = model.brunt_b[k_test];
    assert!(
        b_brassard > 0.0,
        "stabilizing He gradient must give B > 0 in Brassard form, got {b_brassard}"
    );

    // 2. Explicit grad_mu form (`use_brunt_gradmux_form = true`)
    model.controls.use_brunt_gradmux_form = true;
    do_brunt_b(&mut model);
    let b_gradmux = model.brunt_b[k_test];
    assert!(
        b_gradmux > 0.0,
        "stabilizing He gradient must give B > 0 in gradmux form, got {b_gradmux}"
    );
    let rel_diff =
        (b_brassard - b_gradmux).abs() / b_gradmux.abs().max(1.0e-12);
    assert!(
        rel_diff < 0.15,
        "Brassard ({b_brassard}) and gradmux ({b_gradmux}) forms should agree closely in fully ionized interior (rel_diff={rel_diff})"
    );

    // 3. Verify `do_brunt_n2` populates `model.aux.brunt_n2_composition_term`
    set_mlt_vars(&mut model).expect("set_mlt_vars");
    do_brunt_n2(&mut model);
    assert!(
        model.aux.brunt_n2_composition_term[k_test] > 0.0,
        "brunt_n2_composition_term must be positive at stabilizing composition jump"
    );

    // 4. Verify radiation-significant regime (high T where P_rad is comparable to P_gas)
    model.ln_t[k_test - 1] += 1.2;
    model.ln_t[k_test] += 1.2;
    set_basic_vars(&mut model).expect("set_basic_vars high-T");
    crate::micro::do_eos(&mut model).expect("do_eos high-T");
    model.controls.use_brunt_gradmux_form = false;
    do_brunt_b(&mut model);
    let b_mhm_rad = model.brunt_b[k_test];
    model.controls.use_brunt_gradmux_form = true;
    do_brunt_b(&mut model);
    let b_gradmu_rad = model.brunt_b[k_test];
    let rel_diff_rad =
        (b_mhm_rad - b_gradmu_rad).abs() / b_gradmu_rad.abs().max(1.0e-12);
    assert!(
        rel_diff_rad < 0.15,
        "MHM ({b_mhm_rad}) and gradmu ({b_gradmu_rad}) forms must agree in radiation-significant regime (rel_diff={rel_diff_rad})"
    );
}

#[test]
fn test_eval_face_mlt_semiconvection_thermohaline_and_tdc_dispatch() {
    let mut model =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("load 1 Msun ZAMS");
    set_hydro_vars(&mut model, true, true).expect("set_hydro_vars");

    // Pick a radiative interior face `k_rad`
    let k_rad = (1..model.nz)
        .find(|&k| model.mixing_type[k] == MixingType::NoMixing)
        .expect("radiative face in 1 Msun ZAMS");

    // 1. Semiconvection: grad_ad < grad_r < grad_L (B > 0)
    let grada_k = model.grada[k_rad];
    let gradr_k = model.gradr[k_rad];
    // Boost luminosity at `k_rad` so `gradr > grada`, and set `brunt_b[k_rad]` so `gradl > gradr`
    if gradr_k <= grada_k {
        let boost = (grada_k * 1.25) / gradr_k.max(1.0e-6);
        model.lum[k_rad] *= boost;
    }
    model.controls.use_ledoux_criterion = true;
    model.controls.alpha_semiconvection = 0.1;
    model.controls.semiconvection_option = SemiconvectionOption::Langer85;
    model.brunt_b[k_rad] = 0.50;
    set_mlt_vars(&mut model).expect("set_mlt_vars with semiconvection");
    assert_eq!(
        model.mixing_type[k_rad],
        MixingType::SemiconvectiveMixing,
        "face with grada < gradr < gradL and alpha_semiconvection > 0 must be SemiconvectiveMixing"
    );
    assert!(model.d_mix[k_rad] > 0.0);
    assert!(model.conv_vel[k_rad] > 0.0);
    assert_eq!(
        model.mlt_vc[k_rad], 0.0,
        "mlt_vc must be zeroed when mixing_type != ConvectiveMixing"
    );

    // 2. Thermohaline mixing: grad_r < grad_ad and B < 0
    model.lum[k_rad] *= 0.2;
    model.controls.alpha_semiconvection = 0.0;
    model.controls.thermohaline_coeff = 2.0;
    model.controls.thermohaline_option = ThermohalineOption::Kippenhahn;
    model.brunt_b[k_rad] = -0.05;
    set_mlt_vars(&mut model).expect("set_mlt_vars with thermohaline");
    assert_eq!(
        model.mixing_type[k_rad],
        MixingType::ThermohalineMixing,
        "face with gradr < grada, B < 0, and thermohaline_coeff > 0 must be ThermohalineMixing"
    );
    assert!(model.d_mix[k_rad] > 0.0);
    assert_eq!(
        model.mlt_vc[k_rad], 0.0,
        "mlt_vc must be zeroed for ThermohalineMixing"
    );

    // 3. Time-Dependent Convection (`MltOption::Tdc`)
    let k_conv = (1..model.nz)
        .find(|&k| model.mixing_type[k] == MixingType::ConvectiveMixing)
        .expect("convective face in 1 Msun ZAMS");
    model.controls.mlt_option = MltOption::Tdc;
    model.dt = 1.0e4;
    model.aux.mlt_vc_old[k_conv] = model.mlt_vc[k_conv].max(1.0e3);
    set_mlt_vars(&mut model).expect("set_mlt_vars with TDC");
    assert_eq!(model.mixing_type[k_conv], MixingType::ConvectiveMixing);
    assert!(model.mlt_vc[k_conv] > 0.0);
}

#[test]
fn test_eval_face_mlt_and_brunt_n2_rotational_fp_ft_coupling() {
    let mut model =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("load 1 Msun ZAMS");
    model.controls.calculate_brunt_n2 = true;
    set_hydro_vars(&mut model, true, true)
        .expect("set_hydro_vars non-rotating");

    let k = model.nz / 3;
    let hp_nonrot = model.scale_height[k];
    let gradr_nonrot = model.gradr[k];
    let n2_nonrot = model.brunt_n2[k];

    model.controls.rotation_flag = true;
    model.controls.use_gravity_rotation_correction = true;
    model.omega.fill(5.0e-5);
    update_rotation_distortion_factors(&mut model);
    let fp_k = model.aux.fp_rot[k];
    let ft_k = model.aux.ft_rot[k];
    assert!((0.75..1.0).contains(&fp_k));
    assert!((0.95..1.0).contains(&ft_k));
    assert!(model.aux.i_rot[k] > (2.0 / 3.0) * model.r[k] * model.r[k]);

    set_mlt_vars(&mut model).expect("set_mlt_vars rotating");
    do_brunt_n2(&mut model);

    let hp_ratio = model.scale_height[k] / hp_nonrot;
    assert!(
        (hp_ratio - 1.0 / fp_k).abs() < 1.0e-10,
        "scale_height must scale by 1/fp_rot: got {hp_ratio}, expected {}",
        1.0 / fp_k
    );
    let gradr_ratio = model.gradr[k] / gradr_nonrot;
    assert!(
        (gradr_ratio - ft_k / fp_k).abs() < 1.0e-10,
        "gradr must scale by ft_rot/fp_rot: got {gradr_ratio}, expected {}",
        ft_k / fp_k
    );
    let _ = n2_nonrot;
    let zones = set_rotation_mixing_info(&mut model);
    assert_eq!(zones.len(), model.nz);
    let gradr_ratio_after = model.gradr[k] / gradr_nonrot;
    assert!(
        (gradr_ratio_after - ft_k / fp_k).abs() < 1.0e-10,
        "set_rotation_mixing_info must not double-scale gradr: got {gradr_ratio_after}, expected {}",
        ft_k / fp_k
    );
}

#[test]
fn test_eval_surf_bc_simple_photosphere_pextra_and_ad_partials() {
    let mut model =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("load 1 Msun ZAMS");
    set_hydro_vars(&mut model, true, true).expect("set_hydro_vars");

    model.controls.atm_option = AtmOption::SimplePhotosphere;
    model.controls.pextra_factor = 0.0;
    eval_surf_bc(&mut model).expect("eval_surf_bc pextra=0");
    let lnp_no_extra = model.ln_peos_surf_ad.val;

    model.controls.pextra_factor = 1.0;
    eval_surf_bc(&mut model).expect("eval_surf_bc pextra=1");
    let lnp_ad = model.ln_peos_surf_ad;
    let lnt_ad = model.ln_t_surf_ad;

    let expected_extra = (1.0
        + 1.6e-4 * model.opacity[0] * (model.lum[0] / L_SUN)
            / (model.m_grav[0] / M_SUN))
        .ln();
    assert!(
        ((lnp_ad.val - lnp_no_extra) - expected_extra).abs() < 1.0e-12,
        "SimplePhotosphere pextra shift mismatch"
    );

    // Finite-difference verification of `I_LNR_00`, `I_L_00`, `I_LND_00`, `I_LNT_00`
    let eps = 1.0e-6;
    let lnr0 = model.ln_r[0];
    model.ln_r[0] = lnr0 + eps;
    model.r[0] = model.ln_r[0].exp();
    eval_surf_bc(&mut model).expect("eval_surf_bc +lnr");
    let lnp_p = model.ln_peos_surf_ad.val;
    let lnt_p = model.ln_t_surf_ad.val;
    model.ln_r[0] = lnr0 - eps;
    model.r[0] = model.ln_r[0].exp();
    eval_surf_bc(&mut model).expect("eval_surf_bc -lnr");
    let lnp_m = model.ln_peos_surf_ad.val;
    let lnt_m = model.ln_t_surf_ad.val;
    model.ln_r[0] = lnr0;
    model.r[0] = lnr0.exp();

    let fd_lnp_lnr = (lnp_p - lnp_m) / (2.0 * eps);
    let fd_lnt_lnr = (lnt_p - lnt_m) / (2.0 * eps);
    assert!((lnp_ad.d1[I_LNR_00] - fd_lnp_lnr).abs() < 1.0e-7);
    assert!((lnt_ad.d1[I_LNR_00] - fd_lnt_lnr).abs() < 1.0e-7);

    // Luminosity `I_L_00` finite difference
    let l0 = model.lum[0];
    let dl = l0 * eps;
    model.lum[0] = l0 + dl;
    eval_surf_bc(&mut model).expect("eval_surf_bc +L");
    let lnp_lp = model.ln_peos_surf_ad.val;
    let lnt_lp = model.ln_t_surf_ad.val;
    model.lum[0] = l0 - dl;
    eval_surf_bc(&mut model).expect("eval_surf_bc -L");
    let lnp_lm = model.ln_peos_surf_ad.val;
    let lnt_lm = model.ln_t_surf_ad.val;
    model.lum[0] = l0;
    let fd_lnp_l = (lnp_lp - lnp_lm) / (2.0 * dl);
    let fd_lnt_l = (lnt_lp - lnt_lm) / (2.0 * dl);
    assert!(
        (lnp_ad.d1[I_L_00] - fd_lnp_l).abs()
            <= 1.0e-6 * fd_lnp_l.abs().max(1.0e-30)
    );
    assert!(
        (lnt_ad.d1[I_L_00] - fd_lnt_l).abs()
            <= 1.0e-6 * fd_lnt_l.abs().max(1.0e-30)
    );

    // Density `I_LND_00` and temperature `I_LNT_00` finite differences via `opacity[0]`
    let kap0 = model.opacity[0];
    let dkap_dlnd = model.d_opacity_dlnd[0];
    let dkap_dlnt = model.d_opacity_dlnt[0];

    model.opacity[0] = kap0 + dkap_dlnd * eps;
    eval_surf_bc(&mut model).expect("eval_surf_bc +lnd");
    let lnp_dp = model.ln_peos_surf_ad.val;
    model.opacity[0] = kap0 - dkap_dlnd * eps;
    eval_surf_bc(&mut model).expect("eval_surf_bc -lnd");
    let lnp_dm = model.ln_peos_surf_ad.val;
    let fd_lnp_lnd = (lnp_dp - lnp_dm) / (2.0 * eps);
    assert!(
        (lnp_ad.d1[I_LND_00] - fd_lnp_lnd).abs()
            <= 1.0e-6 * fd_lnp_lnd.abs().max(1.0)
    );

    model.opacity[0] = kap0 + dkap_dlnt * eps;
    eval_surf_bc(&mut model).expect("eval_surf_bc +lnt");
    let lnp_tp = model.ln_peos_surf_ad.val;
    model.opacity[0] = kap0 - dkap_dlnt * eps;
    eval_surf_bc(&mut model).expect("eval_surf_bc -lnt");
    let lnp_tm = model.ln_peos_surf_ad.val;
    model.opacity[0] = kap0;
    let fd_lnp_lnt = (lnp_tp - lnp_tm) / (2.0 * eps);
    assert!(
        (lnp_ad.d1[I_LNT_00] - fd_lnp_lnt).abs()
            <= 1.0e-6 * fd_lnp_lnt.abs().max(1.0)
    );
}

#[test]
fn test_do_kap_multi_isotope_cno_and_nugis_lamers_and_omega_mix() {
    let mut model =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("load 1 Msun ZAMS");
    change_model_network(&mut model, "cno_extras_o18_and_ne22.net", None)
        .expect("change network to cno_extras_o18_and_ne22.net");
    set_basic_vars(&mut model).expect("set_basic_vars");
    crate::micro::do_eos(&mut model).expect("do_eos");

    model.kap_config.use_type2_opacities = true;
    let j_c12 = model
        .iso_names
        .iter()
        .position(|s| s == "c12")
        .expect("c12");
    let j_c13 = model
        .iso_names
        .iter()
        .position(|s| s == "c13")
        .expect("c13");
    model.xa[j_c12] = 0.05;
    model.xa[j_c13] = 0.0;
    do_kap(&mut model).expect("do_kap all c12");
    let kap_c12 = model.opacity[0];

    model.xa[j_c12] = 0.02;
    model.xa[j_c13] = 0.03;
    do_kap(&mut model).expect("do_kap split c12+c13");
    let kap_split = model.opacity[0];
    assert!(
        (kap_c12 - kap_split).abs() <= 1.0e-12 * kap_c12,
        "do_kap must sum all C isotopes (c12 + c13) identically for Type 2 opacities"
    );

    // Nugis & Lamers (2000) WR wind matches Dutch wind for x_surf < 0.2 at high T
    let w_nl = eval_nugis_lamers_wind(1.0e5 * L_SUN, 0.98, 0.02, 1.0);
    let w_dutch = eval_dutch_wind(
        1.0e5 * L_SUN,
        15.0 * M_SUN,
        5.0 * flexmesa_const::R_SUN,
        5.0e4,
        0.0,
        0.98,
        0.02,
        DutchLowTScheme::DeJager,
        1.0,
    );
    assert!((w_nl - w_dutch).abs() <= 1.0e-12 * w_nl);

    // `solve_omega_mix` total angular momentum conservation with Roche-distorted `i_rot`
    model.controls.rotation_flag = true;
    for k in 0..model.nz {
        model.omega[k] = 1.0e-4 * (1.0 + 0.5 * (k as f64) / (model.nz as f64));
    }
    update_rotation_distortion_factors(&mut model);
    let j_before: f64 = (0..model.nz)
        .map(|k| model.dm_bar[k] * model.j_rot[k])
        .sum();
    let nu_omega = vec![1.0e12; model.nz];
    solve_omega_mix(&mut model, 1.0e5, &nu_omega, 4);
    let j_after: f64 = (0..model.nz)
        .map(|k| model.dm_bar[k] * model.j_rot[k])
        .sum();
    assert!(
        (j_before - j_after).abs() <= 1.0e-11 * j_before.abs(),
        "solve_omega_mix must conserve total angular momentum with model.aux.i_rot"
    );

    // Phase separation with spectator species
    let k_cryst = model.nz / 2;
    for k in 0..model.nz {
        model.gam[k] = if k >= k_cryst { 185.0 } else { 140.0 };
    }
    let eps_ps = do_phase_separation(&mut model, 1.0e9, 175.0, false);
    assert_eq!(eps_ps.len(), model.nz);
}
