// SPDX-License-Identifier: GPL-3.0-or-later
//! Unit tests for Stage `Fidelity-2` (`#113`):
//! - Brassard et al. (1991) finite-difference EOS vs. $\nabla_\mu$ `do_brunt_b`
//!   and `brunt_n2_composition_term` (`brunt.f90`).
//! - `set_mlt_vars` dispatch to `set_semiconvection`, `set_thermohaline`, and
//!   `set_tdc`, plus `mlt_vc = 3 * D / Lambda` for semiconvective and
//!   thermohaline faces (`turb/private/semiconvection.f90:82, 120`,
//!   `turb/public/turb.f90:74`, `star/private/turb_info.f90:263`).
//! - Rotational distortion $(f_P, f_T, i_{\mathrm{rot}})$ coupling into
//!   `set_mlt_vars`, `do_brunt_n2`, `set_rotation_mixing_info`, and
//!   `solve_omega_mix` (`hydro_rotation.f90`, `solve_omega_mix.f90`).
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
use crate::rotation::update_rotation_distortion_factors;
use crate::rotation_mix_info::set_rotation_mixing_info;
use crate::star_ad::{I_LND_00, I_LNR_00, I_LNT_00, I_L_00};
use crate::struct_burn_mix::solve_omega_mix;
use crate::winds::{eval_dutch_wind, eval_nugis_lamers_wind, DutchLowTScheme};
use crate::zams::load_zams_model;

fn solar_zams_controls() -> StarControls {
    StarControls {
        zbase: 0.02,
        ..StarControls::default()
    }
}

#[test]
fn test_brunt_b_brassard_eos_finite_difference_and_gradmux_forms() {
    let mut model =
        load_zams_model(solar_zams_controls(), StarJobControls::default())
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
    let dln_rho_hse = flexmesa_math::log(p_orig / model.peos[k_test])
        / model.chi_rho[k_test].max(0.5);
    model.lnd[k_test] += dln_rho_hse;
    set_basic_vars(&mut model).expect("set_basic_vars HSE");
    crate::micro::do_eos(&mut model).expect("do_eos HSE");

    model.controls.calculate_brunt_n2 = true;
    model.controls.use_ledoux_criterion = true;

    // 1. Default Brassard et al. (1991) finite-difference EOS form (`!use_brunt_gradmux_form`)
    model.controls.use_brunt_gradmux_form = false;
    do_brunt_b(&mut model).expect("do_brunt_b Brassard");
    let b_brassard = model.brunt_b[k_test];
    assert!(
        b_brassard > 0.0,
        "stabilizing He gradient must give B > 0 in Brassard form, got {b_brassard}"
    );

    // 2. Explicit grad_mu form (`use_brunt_gradmux_form = true`)
    model.controls.use_brunt_gradmux_form = true;
    do_brunt_b(&mut model).expect("do_brunt_b gradmux");
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
    do_brunt_n2(&mut model).expect("do_brunt_n2");
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
    do_brunt_b(&mut model).expect("do_brunt_b Brassard high-T");
    let b_mhm_rad = model.brunt_b[k_test];
    model.controls.use_brunt_gradmux_form = true;
    do_brunt_b(&mut model).expect("do_brunt_b gradmux high-T");
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
        load_zams_model(solar_zams_controls(), StarJobControls::default())
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
    model.aux.mixing.gradl_composition_term[k_rad] = 0.50;
    set_mlt_vars(&mut model).expect("set_mlt_vars with semiconvection");
    assert_eq!(
        model.aux.mixing.mlt_mixing_type[k_rad],
        MixingType::SemiconvectiveMixing,
        "face with grada < gradr < gradL and alpha_semiconvection > 0 must be SemiconvectiveMixing"
    );
    assert!(model.mlt_d[k_rad] > 0.0);
    // Literal provenance (#155): pre-#262 asserted `mlt_vc == 0.0`, which
    // departed from upstream MESA 26.4.1. Upstream sets `conv_vel = 3*D/Lambda`
    // (`turb/private/semiconvection.f90:82, 120`) and stores it in `mlt_vc`
    // (`star/private/turb_info.f90:263`).
    let expected_sc_vc =
        3.0 * model.mlt_d[k_rad] / model.aux.mixing.mlt_mixing_length[k_rad];
    assert!(
        (model.mlt_vc[k_rad] - expected_sc_vc).abs()
            <= 1.0e-12 * expected_sc_vc.abs().max(1.0),
        "semiconvection mlt_vc must equal 3*D/Lambda: got {}, want {expected_sc_vc}",
        model.mlt_vc[k_rad]
    );

    // 2. Thermohaline mixing: grad_r < grad_ad and B < 0
    model.lum[k_rad] *= 0.2;
    model.controls.alpha_semiconvection = 0.0;
    model.controls.thermohaline_coeff = 2.0;
    model.controls.thermohaline_option = ThermohalineOption::Kippenhahn;
    model.aux.mixing.gradl_composition_term[k_rad] = -0.05;
    set_mlt_vars(&mut model).expect("set_mlt_vars with thermohaline");
    assert_eq!(
        model.aux.mixing.mlt_mixing_type[k_rad],
        MixingType::ThermohalineMixing,
        "face with gradr < grada, B < 0, and thermohaline_coeff > 0 must be ThermohalineMixing"
    );
    assert!(model.mlt_d[k_rad] > 0.0);
    // Literal provenance (#155): pre-#262 asserted `mlt_vc == 0.0` and #262
    // dropped the check. Upstream `set_thermohaline` sets `conv_vel = 3*D/Lambda`
    // (`turb/public/turb.f90:74`, stored at `star/private/turb_info.f90:263`).
    let expected_th_vc =
        3.0 * model.mlt_d[k_rad] / model.aux.mixing.mlt_mixing_length[k_rad];
    assert!(
        (model.mlt_vc[k_rad] - expected_th_vc).abs()
            <= 1.0e-12 * expected_th_vc.abs().max(1.0),
        "thermohaline mlt_vc must equal 3*D/Lambda: got {}, want {expected_th_vc}",
        model.mlt_vc[k_rad]
    );

    // 3. Time-Dependent Convection (`MltOption::Tdc`)
    let k_conv = (1..model.nz)
        .find(|&k| model.aux.mixing.mlt_mixing_type[k] == MixingType::ConvectiveMixing)
        .expect("convective face in 1 Msun ZAMS");
    model.controls.mlt_option = MltOption::Tdc;
    model.dt = 1.0e4;
    model.aux.have_mlt_vc = true;
    model.aux.mlt_vc_old[k_conv] = model.mlt_vc[k_conv].max(1.0e3);
    set_mlt_vars(&mut model).expect("set_mlt_vars with TDC");
    assert_eq!(
        model.aux.mixing.mlt_mixing_type[k_conv],
        MixingType::ConvectiveMixing
    );
    assert!(model.mlt_vc[k_conv] > 0.0);
}

#[test]
fn test_eval_face_mlt_and_brunt_n2_rotational_fp_ft_coupling() {
    let mut model =
        load_zams_model(solar_zams_controls(), StarJobControls::default())
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
    do_brunt_n2(&mut model).expect("do_brunt_n2");

    // Literal provenance (#155): pre-#262 asserted `hp_ratio == 1.0 / fp_k`,
    // which departed from upstream MESA 26.4.1. Upstream `get_scale_height_face`
    // (`star/private/star_utils.f90:4001`) and `do_brunt_N2`
    // (`star/private/brunt.f90:160`) use spherical `grav = G*m_grav/r^2`;
    // only `gradr` scales by `ft_rot/fp_rot` (`star/private/turb_info.f90:125-126`).
    let hp_ratio = model.scale_height[k] / hp_nonrot;
    assert!(
        (hp_ratio - 1.0).abs() < 1.0e-12,
        "scale_height must match upstream spherical gravity (star_utils.f90:4001): got {hp_ratio}"
    );
    let gradr_ratio = model.gradr[k] / gradr_nonrot;
    assert!(
        (gradr_ratio - ft_k / fp_k).abs() < 1.0e-10,
        "gradr must scale by ft_rot/fp_rot: got {gradr_ratio}, expected {}",
        ft_k / fp_k
    );
    let _ = n2_nonrot;
    set_rotation_mixing_info(&mut model).expect("set_rotation_mixing_info");
    assert_eq!(model.aux.rotmix.d_st.len(), model.nz);
    let gradr_ratio_after = model.gradr[k] / gradr_nonrot;
    assert!(
        (gradr_ratio_after - ft_k / fp_k).abs() < 1.0e-10,
        "set_rotation_mixing_info must not double-scale gradr: got {gradr_ratio_after}, expected {}",
        ft_k / fp_k
    );
}

#[test]
fn test_do_kap_multi_isotope_cno_and_nugis_lamers_and_omega_mix() {
    let mut model =
        load_zams_model(solar_zams_controls(), StarJobControls::default())
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
    let eps_ps = do_phase_separation(&mut model, 1.0e9, 175.0, false)
        .expect("do_phase_separation");
    assert_eq!(eps_ps.len(), model.nz);
}

#[test]
fn test_eval_surf_bc_t_tau_opacity_fixed_iterated_varying_and_inlist() {
    use crate::controls::{AtmTTauOpacity, StarError};
    use crate::inlist::InlistBundle;

    // 1. Inlist parsing accepts fixed/iterated/varying + tuning controls with 0
    //    unsupported controls, and rejects invalid `atm_T_tau_opacity` loudly.
    let tmp_dir = std::env::temp_dir().join(format!(
        "flexmesa_atm_t_tau_opacity_test_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).expect("create temp dir");

    for (raw, expected) in [
        ("fixed", AtmTTauOpacity::Fixed),
        ("iterated", AtmTTauOpacity::Iterated),
        ("varying", AtmTTauOpacity::Varying),
    ] {
        let p = tmp_dir.join(format!("inlist_{raw}"));
        std::fs::write(
            &p,
            format!(
                "&controls\n\
                   atm_option = 'T_tau'\n\
                   atm_T_tau_relation = 'Eddington'\n\
                   atm_T_tau_opacity = '{raw}'\n\
                   atm_T_tau_errtol = 1.0d-8\n\
                   atm_T_tau_max_iters = 60\n\
                   atm_T_tau_max_steps = 400\n\
                 /\n"
            ),
        )
        .expect("write inlist");
        let bundle = InlistBundle::from_file(&p)
            .expect("atm_T_tau_opacity inlist must pass strict validation");
        assert_eq!(bundle.validation_report.unsupported_controls.len(), 0);
        assert_eq!(bundle.controls.atm_t_tau_opacity, expected);
        assert!((bundle.controls.atm_t_tau_errtol - 1.0e-8).abs() <= 1.0e-20);
        assert_eq!(bundle.controls.atm_t_tau_max_iters, 60);
        assert_eq!(bundle.controls.atm_t_tau_max_steps, 400);
    }

    let bad = tmp_dir.join("inlist_bad");
    std::fs::write(&bad, "&controls\n  atm_T_tau_opacity = 'bogus'\n/\n")
        .expect("write bad inlist");
    let err = InlistBundle::from_file(&bad)
        .expect_err("invalid atm_T_tau_opacity must fail loudly");
    assert!(matches!(err, StarError::InvalidInput(_)));
    let _ = std::fs::remove_dir_all(&tmp_dir);

    // 2. Dispatch across Fixed, Iterated, and Varying on a 1 Msun ZAMS model.
    let mut model =
        load_zams_model(solar_zams_controls(), StarJobControls::default())
            .expect("load 1 Msun ZAMS");
    model.controls.atm_option = AtmOption::TTau;
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");

    // Fixed mode: respects `pextra_factor` and exposes explicit opacity partials.
    model.controls.atm_t_tau_opacity = AtmTTauOpacity::Fixed;
    model.controls.pextra_factor = 1.0;
    eval_surf_bc(&mut model).expect("eval_surf_bc fixed pextra=1");
    let lnp_fixed_p1 = model.ln_peos_surf_ad.val;
    let lnt_fixed = model.ln_t_surf_ad.val;
    assert!(model.ln_peos_surf_ad.d1[I_LND_00].abs() > 0.0);
    assert!(model.ln_peos_surf_ad.d1[I_LNT_00].abs() > 0.0);

    model.controls.pextra_factor = 200.0;
    eval_surf_bc(&mut model).expect("eval_surf_bc fixed pextra=200");
    assert!(
        model.ln_peos_surf_ad.val > lnp_fixed_p1,
        "larger pextra_factor must increase fixed T(tau) surface pressure"
    );
    model.controls.pextra_factor = 1.0;

    // Perturb `opacity[0]` so Fixed and Iterated diverge sharply.
    let kap_orig = model.opacity[0];
    model.opacity[0] = kap_orig * 3.0;
    eval_surf_bc(&mut model).expect("eval_surf_bc fixed perturbed kap");
    let lnp_fixed_perturbed = model.ln_peos_surf_ad.val;

    // Iterated mode: solves kap self-consistently, zeroing direct (lnd, lnT)
    // cell-0 opacity partials, and matches `ln_r`/`L` central finite differences.
    model.controls.atm_t_tau_opacity = AtmTTauOpacity::Iterated;
    model.aux.opacity_start_surf = kap_orig;
    model.aux.doing_struct_burn_mix = true;
    model.aux.solver_iter = 1;
    eval_surf_bc(&mut model).expect("eval_surf_bc iterated in solver iter");
    let lnp_iter_ad = model.ln_peos_surf_ad;
    let lnt_iter_ad = model.ln_t_surf_ad;
    assert_eq!(lnp_iter_ad.d1[I_LND_00], 0.0);
    assert_eq!(lnp_iter_ad.d1[I_LNT_00], 0.0);
    assert!((lnt_iter_ad.val - lnt_fixed).abs() <= 1.0e-12);
    assert!(
        (lnp_iter_ad.val - lnp_fixed_perturbed).abs() > 0.5,
        "iterated T(tau) must converge to self-consistent kap instead of perturbed opacity[0]"
    );

    let eps = 1.0e-5;
    let lnr0 = model.ln_r[0];
    model.ln_r[0] = lnr0 + eps;
    model.r[0] = flexmesa_math::exp(model.ln_r[0]);
    eval_surf_bc(&mut model).expect("iterated +lnr");
    let lnp_rp = model.ln_peos_surf_ad.val;
    model.ln_r[0] = lnr0 - eps;
    model.r[0] = flexmesa_math::exp(model.ln_r[0]);
    eval_surf_bc(&mut model).expect("iterated -lnr");
    let lnp_rm = model.ln_peos_surf_ad.val;
    model.ln_r[0] = lnr0;
    model.r[0] = flexmesa_math::exp(lnr0);
    let fd_iter_lnr = (lnp_rp - lnp_rm) / (2.0 * eps);
    assert!(
        (lnp_iter_ad.d1[I_LNR_00] - fd_iter_lnr).abs()
            <= 1.0e-4 * fd_iter_lnr.abs().max(1.0),
        "iterated dlnP/dlnR analytic={} vs fd={}",
        lnp_iter_ad.d1[I_LNR_00],
        fd_iter_lnr
    );

    // Varying mode: integrates dP/dtau = g/kap(P, T) via `eval_t_tau_varying`.
    // Upstream `atm_t_tau_varying.f90:173-186` finite-differences in `lnTeff`
    // at fixed `g`, so `dlnP/dL`, `dlnT/dL`, and `dlnT/dlnR` match central
    // finite differences while `dlnP/dlnR == -2 * L * dlnP/dL`.
    model.aux.doing_struct_burn_mix = false;
    model.aux.solver_iter = 0;
    model.opacity[0] = kap_orig;
    model.controls.atm_t_tau_opacity = AtmTTauOpacity::Varying;
    eval_surf_bc(&mut model).expect("eval_surf_bc varying");
    let lnp_var_ad = model.ln_peos_surf_ad;
    let lnt_var_ad = model.ln_t_surf_ad;
    assert_eq!(lnp_var_ad.d1[I_LND_00], 0.0);
    assert_eq!(lnp_var_ad.d1[I_LNT_00], 0.0);
    assert!(lnp_var_ad.val.is_finite());
    assert!(
        (lnp_var_ad.val - lnp_iter_ad.val).abs() > 1.0e-3,
        "varying opacity integration must differ from uniform iterated opacity"
    );

    let l0 = model.lum[0];
    let dl = l0 * eps;
    model.lum[0] = l0 + dl;
    eval_surf_bc(&mut model).expect("varying +L");
    let lnp_lp = model.ln_peos_surf_ad.val;
    let lnt_lp = model.ln_t_surf_ad.val;
    model.lum[0] = l0 - dl;
    eval_surf_bc(&mut model).expect("varying -L");
    let lnp_lm = model.ln_peos_surf_ad.val;
    let lnt_lm = model.ln_t_surf_ad.val;
    model.lum[0] = l0;
    let fd_var_lnp_l = (lnp_lp - lnp_lm) / (2.0 * dl);
    let fd_var_lnt_l = (lnt_lp - lnt_lm) / (2.0 * dl);
    assert!(
        (lnp_var_ad.d1[I_L_00] - fd_var_lnp_l).abs()
            <= 1.0e-3 * fd_var_lnp_l.abs().max(1.0e-30),
        "varying dlnP/dL analytic={} vs fd={}",
        lnp_var_ad.d1[I_L_00],
        fd_var_lnp_l
    );
    assert!(
        (lnt_var_ad.d1[I_L_00] - fd_var_lnt_l).abs()
            <= 1.0e-4 * fd_var_lnt_l.abs().max(1.0e-30),
        "varying dlnT/dL analytic={} vs fd={}",
        lnt_var_ad.d1[I_L_00],
        fd_var_lnt_l
    );
    let exp_dlnp_dlnr = -2.0 * l0 * lnp_var_ad.d1[I_L_00];
    assert!(
        (lnp_var_ad.d1[I_LNR_00] - exp_dlnp_dlnr).abs()
            <= 1.0e-12 * exp_dlnp_dlnr.abs().max(1.0),
        "varying dlnP/dlnR={} must equal -2*L*dlnP/dL={}",
        lnp_var_ad.d1[I_LNR_00],
        exp_dlnp_dlnr
    );
}
