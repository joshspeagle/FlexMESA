// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone zero-`$MESA_DIR` MESA `26.4.1` stellar-wind parity tests (`#159`,
//! `#161`), covering exact closed-form scheme limits, `other_wind` hook
//! precedence with custom scheme names, `hot_cool_wind` and `7M_prems_to_AGB`
//! driver pathways, rotational `w / w_crit` enhancement + caps, and single-star
//! RLO / super-Eddington / envelope-limit gates (`star/private/winds.f90`).

#![forbid(unsafe_code)]

use std::cell::RefCell;

use flexmesa_const::{
    CLIGHT, CRAD, FOUR_PI, LN10, L_SUN, M_SUN, PI, R_SUN, SECYER,
    STANDARD_CGRAV,
};
use flexmesa_star::{
    compute_core_masses, do_set_mdot, do_set_mdot_with_extras,
    eval_bjorklund_wind, eval_blocker_wind, eval_de_jager_wind,
    eval_dutch_wind, eval_nieuwenhuijzen_wind, eval_nugis_lamers_wind,
    eval_photosphere_info, eval_reimers_wind, eval_rlo_wind,
    eval_super_eddington_wind, eval_surf_avg_rotation_info,
    eval_upstream_omega_crit, eval_van_loon_wind, eval_vink_wind,
    load_zams_model, set_hydro_vars, set_mdot, DutchLowTScheme, StarControls,
    StarError, StarExtras, StarJobControls, StarModel, WindScheme,
};

fn rel_err(got: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        got.abs()
    } else {
        (got - expected).abs() / expected.abs()
    }
}

fn zams_model(mass_msun: f64) -> StarModel {
    let ctrl = StarControls {
        initial_mass: mass_msun,
        zbase: 0.02,
        initial_z: 0.02,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(ctrl, StarJobControls::default()).expect("load zams");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model
}

fn set_species_zone(model: &mut StarModel, k: usize, name: &str, val: f64) {
    if let Some(j) = model
        .iso_names
        .iter()
        .position(|iso| iso.eq_ignore_ascii_case(name))
    {
        model.cell_xa_mut(k)[j] = val;
    }
}

fn clear_wind_schemes(model: &mut StarModel) {
    model.controls.mix_wind.hot_wind_scheme.clear();
    model.controls.cool_wind_rgb_scheme.clear();
    model.controls.cool_wind_agb_scheme.clear();
    model.controls.reimers_scaling_factor = 0.0;
    model.controls.blocker_scaling_factor = 0.0;
    model.controls.vink_scaling_factor = 0.0;
    model.controls.dutch_scaling_factor = 0.0;
    model.controls.mix_wind.nieuwenhuijzen_scaling_factor = 0.0;
    model.controls.mix_wind.de_jager_scaling_factor = 0.0;
    model.controls.mix_wind.van_loon_scaling_factor = 0.0;
    model.controls.mix_wind.bjorklund_scaling_factor = 0.0;
}

struct RecordingOtherWindExtras {
    wind_msun_yr: f64,
    calls: RefCell<Vec<(f64, f64, f64, f64)>>,
}

impl RecordingOtherWindExtras {
    fn new(wind_msun_yr: f64) -> Self {
        Self {
            wind_msun_yr,
            calls: RefCell::new(Vec::new()),
        }
    }
}

impl StarExtras for RecordingOtherWindExtras {
    fn other_wind(
        &self,
        _model: &StarModel,
        l_surf_erg_s: f64,
        m_surf_g: f64,
        r_surf_cm: f64,
        teff_k: f64,
    ) -> Result<Option<f64>, StarError> {
        self.calls.borrow_mut().push((
            l_surf_erg_s,
            m_surf_g,
            r_surf_cm,
            teff_k,
        ));
        Ok(Some(self.wind_msun_yr))
    }
}

#[test]
fn winds_159_v2_closed_form_scheme_oracles() {
    // 1. Reimers (1975): exact unity point at (L_SUN, M_SUN, R_SUN).
    let w_reimers = eval_reimers_wind(L_SUN, M_SUN, R_SUN, 0.5);
    assert!(
        rel_err(w_reimers, 2.0e-13) < 1.0e-15,
        "reimers unity mismatch: {w_reimers:.16e}"
    );
    assert_eq!(eval_reimers_wind(L_SUN, M_SUN, R_SUN, -0.1).to_bits(), 0);
    assert_eq!(eval_reimers_wind(0.0, M_SUN, R_SUN, 1.0).to_bits(), 0);

    // 2. Blocker (1995): unity point gives 4e-13 * 4.83e-9 = 1.932e-21.
    let w_blocker = eval_blocker_wind(L_SUN, M_SUN, R_SUN, 1.0);
    assert!(
        rel_err(w_blocker, 4.0e-13 * 4.83e-9) < 1.0e-15,
        "blocker unity mismatch: {w_blocker:.16e}"
    );

    // 3. de Jager (1988): at L = L_SUN, T = 10^4 K -> log10 w = -1.676*4 - 8.158.
    let w_dejager = eval_de_jager_wind(L_SUN, 1.0e4, 0.75);
    let exp_dejager = 0.75 * flexmesa_math::exp10(-1.676 * 4.0 - 8.158);
    assert_eq!(w_dejager.to_bits(), exp_dejager.to_bits());

    // 4. van Loon (2005): at L = 1e4 L_SUN, T = 3500 K -> log10 w = -5.65.
    let w_vanloon = eval_van_loon_wind(1.0e4 * L_SUN, 3.5e3, 1.25);
    let exp_vanloon = 1.25 * flexmesa_math::exp10(-5.65);
    assert_eq!(w_vanloon.to_bits(), exp_vanloon.to_bits());

    // 5. Nieuwenhuijzen & de Jager (1990): unity point -> log10 w = -14.02.
    let w_nieuw = eval_nieuwenhuijzen_wind(L_SUN, M_SUN, R_SUN, 0.4);
    let exp_nieuw = 0.4 * flexmesa_math::exp10(-14.02);
    assert_eq!(w_nieuw.to_bits(), exp_nieuw.to_bits());

    // 6. Nugis & Lamers (2000): L = L_SUN, Y = 1, Z = 0.04 -> 2e-12 Msun/yr.
    let w_nugis = eval_nugis_lamers_wind(L_SUN, 1.0, 0.04, 1.0);
    assert!(
        rel_err(w_nugis, 2.0e-12) < 1.0e-15,
        "nugis-lamers unity mismatch: {w_nugis:.16e}"
    );

    // 7. Vink (2001): hot/cool exact logs & bistability jump ramp at Z = 0.019.
    let z_solar = 0.019;
    let w_vink_hot =
        eval_vink_wind(1.0e5 * L_SUN, 30.0 * M_SUN, 4.0e4, z_solar, 1.0);
    let expected_hot =
        flexmesa_math::exp10(-6.697 - 1.226 * flexmesa_math::log10(2.6 / 2.0));
    assert_eq!(w_vink_hot.to_bits(), expected_hot.to_bits());

    let w_vink_cool =
        eval_vink_wind(1.0e5 * L_SUN, 30.0 * M_SUN, 2.0e4, z_solar, 1.0);
    let expected_cool =
        flexmesa_math::exp10(-6.688 - 1.601 * flexmesa_math::log10(1.3 / 2.0));
    assert_eq!(w_vink_cool.to_bits(), expected_cool.to_bits());

    let t_jump = 1.0e3 * (61.2 + 2.59 * (-13.636));
    let w_bistable =
        eval_vink_wind(1.0e5 * L_SUN, 30.0 * M_SUN, t_jump, z_solar, 1.0);
    let log_t4e4 = flexmesa_math::log10(t_jump / 4.0e4);
    let hot_bistable = flexmesa_math::exp10(
        -6.697 - 1.226 * flexmesa_math::log10(2.6 / 2.0) + 0.933 * log_t4e4
            - 10.92 * flexmesa_math::pow2(log_t4e4),
    );
    let cool_bistable = flexmesa_math::exp10(
        -6.688 - 1.601 * flexmesa_math::log10(1.3 / 2.0)
            + 1.07 * flexmesa_math::log10(t_jump / 2.0e4),
    );
    assert!(
        rel_err(w_bistable, f64::midpoint(hot_bistable, cool_bistable))
            < 1.0e-14,
        "vink bistability midpoint mismatch: got {w_bistable:.16e}"
    );
    let w_above = eval_vink_wind(
        1.0e5 * L_SUN,
        30.0 * M_SUN,
        t_jump + 101.0,
        z_solar,
        1.0,
    );
    let w_below = eval_vink_wind(
        1.0e5 * L_SUN,
        30.0 * M_SUN,
        t_jump - 101.0,
        z_solar,
        1.0,
    );
    assert!(
        w_above < w_below,
        "cool side should exceed hot side around jump"
    );
}

#[test]
fn winds_159_v2_bjorklund_cross_term_dutch_and_set_mdot_guards() {
    // 1. Bjorklund et al. (2023): exact M_eff = 45 M_SUN cancellation point and
    //    (T, Z) cross-term slope (0.75 - 1.87 * log10(T / 4.5e4)).
    let l_bj = 1.0e6 * L_SUN;
    let m_rad = 0.34 * l_bj / (FOUR_PI * CLIGHT * STANDARD_CGRAV);
    let m_bj = 45.0 * M_SUN + m_rad;
    let w_bjork =
        eval_bjorklund_wind(l_bj, m_bj, 4.5e4, 0.014, STANDARD_CGRAV, 0.5);
    let exp_bjork = 0.5 * flexmesa_math::exp10(-5.52);
    assert!(
        rel_err(w_bjork, exp_bjork) < 1.0e-13,
        "bjorklund M_eff=45 Ms point mismatch: got {w_bjork:.16e}, exp {exp_bjork:.16e}"
    );
    let w_bj_z1 =
        eval_bjorklund_wind(l_bj, m_bj, 4.5e5, 0.014, STANDARD_CGRAV, 1.0);
    let w_bj_z2 =
        eval_bjorklund_wind(l_bj, m_bj, 4.5e5, 0.028, STANDARD_CGRAV, 1.0);
    let z_slope =
        flexmesa_math::log10(w_bj_z2 / w_bj_z1) / flexmesa_math::log10(2.0);
    assert!(
        rel_err(z_slope, 0.75 - 1.87) < 1.0e-12,
        "bjorklund high-T Z slope mismatch: {z_slope:.15e}"
    );

    // 2. Dutch composite wind: transition midpoint, X < 0.4 WR cutoff, and
    //    consistent `eta <= 0.0` guard matching sibling evaluators + `set_mdot`.
    let w_dutch_low = eval_dutch_wind(
        1.0e5 * L_SUN,
        15.0 * M_SUN,
        50.0 * R_SUN,
        10_000.0,
        0.70,
        0.28,
        0.02,
        DutchLowTScheme::DeJager,
        1.0,
    );
    assert_eq!(
        w_dutch_low.to_bits(),
        eval_de_jager_wind(1.0e5 * L_SUN, 10_000.0, 1.0).to_bits()
    );
    let w_dutch_high_vink = eval_dutch_wind(
        1.0e5 * L_SUN,
        15.0 * M_SUN,
        50.0 * R_SUN,
        11_000.0,
        0.40,
        0.58,
        0.02,
        DutchLowTScheme::DeJager,
        1.0,
    );
    assert_eq!(
        w_dutch_high_vink.to_bits(),
        eval_vink_wind(1.0e5 * L_SUN, 15.0 * M_SUN, 11_000.0, 0.02, 1.0)
            .to_bits()
    );
    let w_dutch_high_wr = eval_dutch_wind(
        1.0e5 * L_SUN,
        15.0 * M_SUN,
        50.0 * R_SUN,
        11_000.0,
        0.3999,
        0.5801,
        0.02,
        DutchLowTScheme::DeJager,
        1.0,
    );
    assert_eq!(
        w_dutch_high_wr.to_bits(),
        eval_nugis_lamers_wind(1.0e5 * L_SUN, 0.5801, 0.02, 1.0).to_bits()
    );
    let w_dutch_mid = eval_dutch_wind(
        1.0e5 * L_SUN,
        15.0 * M_SUN,
        50.0 * R_SUN,
        10_250.0,
        0.70,
        0.28,
        0.02,
        DutchLowTScheme::DeJager,
        0.25,
    );
    let exp_mid_low = eval_de_jager_wind(1.0e5 * L_SUN, 10_250.0, 1.0);
    let exp_mid_high =
        eval_vink_wind(1.0e5 * L_SUN, 15.0 * M_SUN, 10_250.0, 0.02, 1.0);
    assert_eq!(
        w_dutch_mid.to_bits(),
        (0.25 * (0.75 * exp_mid_low + 0.25 * exp_mid_high)).to_bits()
    );
}

#[test]
fn winds_159_v2_dutch_guards_and_set_mdot_cross_check() {
    assert_eq!(
        eval_dutch_wind(
            1.0e5 * L_SUN,
            15.0 * M_SUN,
            50.0 * R_SUN,
            15_000.0,
            0.70,
            0.28,
            0.02,
            DutchLowTScheme::VanLoon,
            -1.0,
        )
        .to_bits(),
        0
    );

    // Cross-check `set_mdot` direct helper vs `eval_reimers_wind`.
    let mut m = zams_model(1.0);
    let got_mdot =
        set_mdot(&mut m, WindScheme::Reimers, DutchLowTScheme::DeJager, 0.5);
    let l0 = m.lum[0].max(1.0);
    let m0 = m.m_grav[0].max(1.0);
    let r0 = m.r[0].max(1.0);
    let expected_cgs = -eval_reimers_wind(l0, m0, r0, 0.5) * M_SUN / SECYER;
    assert_eq!(got_mdot.to_bits(), expected_cgs.to_bits());
    assert_eq!(m.mstar_dot.to_bits(), expected_cgs.to_bits());
}

#[test]
fn winds_159_v2_other_wind_precedence_and_error_guards() {
    let base = zams_model(10.0);

    // 1. Upstream `winds.f90:319-406` evaluates `use_other_wind` BEFORE rejecting
    //    unknown scheme labels, allowing descriptive names like `'Kudritzki'`,
    //    `'Stern51'`, or `'Grafener'`.
    let mut m = base.clone();
    m.controls.use_other_wind = true;
    m.controls.hot_wind_full_on_t = 12_000.0;
    m.controls.cool_wind_full_on_t = 8_000.0;
    m.controls.mix_wind.hot_wind_scheme = String::from("Kudritzki");
    m.controls.cool_wind_rgb_scheme = String::from("Stern51");
    m.controls.cool_wind_agb_scheme = String::from("Grafener");
    m.controls.mix_wind.rgb_to_agb_wind_switch = 1.0e-4;
    m.controls.mix_wind.dutch_wind_lowt_scheme = String::from("invalid_unused");

    // Hot branch (T > hot_wind_full_on_T): single hook call, exact L1/M1/R1/T1.
    m.teff = 20_000.0;
    let extras_hot = RecordingOtherWindExtras::new(2.5e-6);
    let got_hot = do_set_mdot_with_extras(&mut m, &extras_hot)
        .expect("use_other_wind with custom hot_wind_scheme must succeed");
    assert_eq!(got_hot.to_bits(), (-2.5e-6 * M_SUN / SECYER).to_bits());
    let calls_hot = extras_hot.calls.borrow();
    assert_eq!(calls_hot.len(), 1);
    let (l1, m1, r1, t1) = calls_hot[0];
    let phot = eval_photosphere_info(&m);
    assert_eq!(l1.to_bits(), (phot.l_lsun * L_SUN).to_bits());
    assert_eq!(m1.to_bits(), m.mstar.to_bits());
    assert_eq!(t1.to_bits(), 20_000.0_f64.to_bits());
    let expected_r1 = (l1 / (PI * CRAD * CLIGHT * t1 * t1 * t1 * t1)).sqrt();
    assert_eq!(r1.to_bits(), expected_r1.to_bits());

    // Cool RGB branch:
    m.teff = 5_000.0;
    let extras_cool = RecordingOtherWindExtras::new(1.2e-7);
    let got_cool = do_set_mdot_with_extras(&mut m, &extras_cool)
        .expect("use_other_wind with custom cool_wind_rgb_scheme must succeed");
    assert_eq!(got_cool.to_bits(), (-1.2e-7 * M_SUN / SECYER).to_bits());
    assert_eq!(extras_cool.calls.borrow().len(), 1);

    // Cool AGB branch (`center_h1 < 0.01 && center_he4 < switch`):
    let nz = m.nz;
    set_species_zone(&mut m, nz - 1, "h1", 0.0);
    set_species_zone(&mut m, nz - 1, "he4", 0.0);
    let extras_agb = RecordingOtherWindExtras::new(3.4e-7);
    let got_agb = do_set_mdot_with_extras(&mut m, &extras_agb)
        .expect("use_other_wind with custom cool_wind_agb_scheme must succeed");
    assert_eq!(got_agb.to_bits(), (-3.4e-7 * M_SUN / SECYER).to_bits());
    assert_eq!(extras_agb.calls.borrow().len(), 1);

    // Blend region (`cool_on <= T <= hot_on`): invokes hook for both branches.
    m.teff = 10_000.0;
    let extras_blend = RecordingOtherWindExtras::new(5.0e-7);
    let got_blend = do_set_mdot_with_extras(&mut m, &extras_blend)
        .expect("use_other_wind blend region must succeed");
    assert_eq!(extras_blend.calls.borrow().len(), 2);
    assert!(rel_err(got_blend, -5.0e-7 * M_SUN / SECYER) < 1.0e-15);

    // 2. `scheme = 'Other'` works even when `use_other_wind = false`.
    let mut m_sch = base.clone();
    clear_wind_schemes(&mut m_sch);
    m_sch.controls.use_other_wind = false;
    m_sch.controls.mix_wind.hot_wind_scheme = String::from("Other");
    m_sch.teff = 20_000.0;
    let extras_sch = RecordingOtherWindExtras::new(4.0e-8);
    let got_sch = do_set_mdot_with_extras(&mut m_sch, &extras_sch)
        .expect("scheme='Other' invokes other_wind");
    assert_eq!(got_sch.to_bits(), (-4.0e-8 * M_SUN / SECYER).to_bits());

    // 3. Missing hook with `use_other_wind = true` still fails loudly.
    let err_missing = do_set_mdot(&mut m)
        .expect_err("use_other_wind without hook must fail loudly");
    assert!(err_missing.to_string().contains("use_other_wind"));

    // 4. Unknown scheme with `use_other_wind = false` still fails loudly.
    m.controls.use_other_wind = false;
    m.teff = 20_000.0;
    let err_unknown = do_set_mdot(&mut m)
        .expect_err("unknown scheme without use_other_wind must fail loudly");
    assert!(err_unknown
        .to_string()
        .contains("unknown name for wind scheme"));

    // 5. Inverted temperature thresholds fail loudly (`winds.f90:144-148`).
    m.controls.cool_wind_full_on_t = 15_000.0;
    m.controls.hot_wind_full_on_t = 12_000.0;
    let err_temp = do_set_mdot(&mut m)
        .expect_err("hot_wind_full_on_T < cool_wind_full_on_T must fail");
    assert!(matches!(err_temp, StarError::InvalidInput(_)));
}

#[test]
fn winds_159_v2_hot_cool_wind_and_7m_prems_to_agb_control_paths() {
    // 1. `star/test_suite/hot_cool_wind` wind setup (`Zbase = 0.02`,
    //    `cool_wind_full_on_T = 8e3`, `hot_wind_full_on_T = 1.2e4`,
    //    `cool_wind_RGB_scheme = 'Reimers'`, `RGB_to_AGB_wind_switch = -1`,
    //    `hot_wind_scheme = 'Vink'`).
    let mut hc = zams_model(5.0);
    clear_wind_schemes(&mut hc);
    hc.controls.hot_wind_full_on_t = 1.2e4;
    hc.controls.cool_wind_full_on_t = 8.0e3;
    hc.controls.mix_wind.hot_wind_scheme = String::from("Vink");
    hc.controls.vink_scaling_factor = 1.0;
    hc.controls.cool_wind_rgb_scheme = String::from("Reimers");
    hc.controls.cool_wind_agb_scheme = String::from("Blocker");
    hc.controls.mix_wind.rgb_to_agb_wind_switch = -1.0;
    hc.controls.reimers_scaling_factor = 1.0;
    hc.controls.blocker_scaling_factor = 1.0;
    hc.kap_config.zbase = 0.02;

    let phot = eval_photosphere_info(&hc);
    let l1 = phot.l_lsun * L_SUN;
    let m1 = hc.mstar;
    let nz = hc.nz;

    // With `rgb_to_agb_wind_switch = -1`, even depleted center H1=He4=0 never
    // enters `cool_wind_agb_scheme` (`Blocker`).
    set_species_zone(&mut hc, nz - 1, "h1", 0.0);
    set_species_zone(&mut hc, nz - 1, "he4", 0.0);
    hc.teff = 6_000.0;
    let r_6k = (l1
        / (PI * CRAD * CLIGHT * 6_000.0 * 6_000.0 * 6_000.0 * 6_000.0))
        .sqrt();
    let exp_rei_6k = -eval_reimers_wind(l1, m1, r_6k, 1.0) * M_SUN / SECYER;
    assert_eq!(
        do_set_mdot(&mut hc).expect("hot_cool cool").to_bits(),
        exp_rei_6k.to_bits()
    );

    // Transition quarter-point T = 9000 K: beta = (12000 - 9000)/4000 = 0.75.
    hc.teff = 9_000.0;
    let r_9k = (l1
        / (PI * CRAD * CLIGHT * 9_000.0 * 9_000.0 * 9_000.0 * 9_000.0))
        .sqrt();
    let w_hot_9k = eval_vink_wind(l1, m1, 9_000.0, 0.02, 1.0);
    let w_cool_9k = eval_reimers_wind(l1, m1, r_9k, 1.0);
    let exp_9k = -(0.25 * w_hot_9k + 0.75 * w_cool_9k) * M_SUN / SECYER;
    assert_eq!(
        do_set_mdot(&mut hc)
            .expect("hot_cool blend 9000 K")
            .to_bits(),
        exp_9k.to_bits()
    );

    // Hot regime T = 20000 K: `zbase >= 0` overrides present-day `1 - (X + Y)`
    // even if surface H/He changes; `zbase < 0` falls back to `1 - (X + Y)`.
    set_species_zone(&mut hc, 0, "h1", 0.60);
    set_species_zone(&mut hc, 0, "he4", 0.30);
    hc.teff = 20_000.0;
    let exp_zbase =
        -eval_vink_wind(l1, m1, 20_000.0, 0.02, 1.0) * M_SUN / SECYER;
    assert_eq!(
        do_set_mdot(&mut hc).expect("hot_cool zbase").to_bits(),
        exp_zbase.to_bits()
    );

    hc.kap_config.zbase = -1.0;
    let z_present = 1.0 - (0.60 + 0.30);
    let exp_zpresent =
        -eval_vink_wind(l1, m1, 20_000.0, z_present, 1.0) * M_SUN / SECYER;
    assert_eq!(
        do_set_mdot(&mut hc).expect("hot_cool z_present").to_bits(),
        exp_zpresent.to_bits()
    );
}

#[test]
fn winds_159_v2_seven_m_prems_to_agb_control_path() {
    // `star/test_suite/7M_prems_to_AGB` wind setup (`Zbase = 0.001`,
    // `cool_wind_full_on_T = 9.99d9`, `hot_wind_full_on_T = 1d10`,
    // `Reimers = 0.5`, `Blocker = 0.0003`, `RGB_to_AGB_wind_switch = 1d-4`).
    let mut m7 = zams_model(7.0);
    clear_wind_schemes(&mut m7);
    m7.controls.cool_wind_full_on_t = 9.99e9;
    m7.controls.hot_wind_full_on_t = 1.0e10;
    m7.controls.cool_wind_rgb_scheme = String::from("Reimers");
    m7.controls.reimers_scaling_factor = 0.5;
    m7.controls.cool_wind_agb_scheme = String::from("Blocker");
    m7.controls.blocker_scaling_factor = 0.0003;
    m7.controls.mix_wind.rgb_to_agb_wind_switch = 1.0e-4;
    m7.kap_config.zbase = 0.001;

    let phot7 = eval_photosphere_info(&m7);
    let l7 = phot7.l_lsun * L_SUN;
    let m7_g = m7.mstar;
    let t7 = m7.teff;
    let r7 = (l7 / (PI * CRAD * CLIGHT * t7 * t7 * t7 * t7)).sqrt();
    let nz7 = m7.nz;

    // Core He burning (`center_h1 < 0.01`, `center_he4 >= 1e-4`): stays Reimers.
    set_species_zone(&mut m7, nz7 - 1, "h1", 0.0);
    set_species_zone(&mut m7, nz7 - 1, "he4", 2.0e-4);
    let reimers_rgb_cgs =
        -eval_reimers_wind(l7, m7_g, r7, 0.5) * M_SUN / SECYER;
    assert_eq!(
        do_set_mdot(&mut m7).expect("7M RGB/CHeB").to_bits(),
        reimers_rgb_cgs.to_bits()
    );

    // AGB (`center_h1 < 0.01 && center_he4 < 1e-4`): switches to Blocker.
    set_species_zone(&mut m7, nz7 - 1, "h1", 0.005);
    set_species_zone(&mut m7, nz7 - 1, "he4", 5.0e-5);
    let blocker_agb_cgs =
        -eval_blocker_wind(l7, m7_g, r7, 0.0003) * M_SUN / SECYER;
    assert_eq!(
        do_set_mdot(&mut m7).expect("7M AGB Blocker").to_bits(),
        blocker_agb_cgs.to_bits()
    );
}

#[test]
fn winds_159_v2_rotation_threshold_boost_and_caps() {
    let base = zams_model(10.0);

    // 1. Without a wind scheme (`using_wind_scheme == false`) and `min_wind == 0`,
    //    `rotation_enhancement` triggers ONLY when `v_div_v_crit_avg_surf > 0.8`
    //    (`winds.f90:289-292`).
    let mut m_rot = base.clone();
    clear_wind_schemes(&mut m_rot);
    m_rot.controls.rotation_flag = true;
    m_rot.controls.mix_wind.min_wind = 0.0;
    m_rot.controls.mass_change = -1.0e-6;
    m_rot.controls.mix_wind.mdot_omega_power = 0.43;
    m_rot.controls.mix_wind.max_rotational_mdot_boost = 1.0e4;
    m_rot.controls.mix_wind.rotational_mdot_boost_fac = 1.0e5;
    m_rot.controls.mix_wind.rotational_mdot_kh_fac = 1.0e5;
    m_rot.controls.mix_wind.max_mdot_jump_for_rotation = 1.0e5;

    for k in 0..m_rot.nz {
        m_rot.omega[k] = 0.5 * eval_upstream_omega_crit(&m_rot, k);
    }
    let avg_below = eval_surf_avg_rotation_info(&m_rot).v_div_v_crit;
    assert!(avg_below <= 0.8, "expected v/vc <= 0.8, got {avg_below}");
    m_rot.aux.step_hist.rotational_mdot_boost_old = 1.0;
    let base_cgs = -1.0e-6 * M_SUN / SECYER;
    let got_below = do_set_mdot(&mut m_rot).expect("v/vc <= 0.8");
    assert_eq!(got_below.to_bits(), base_cgs.to_bits());
    assert_eq!(
        m_rot.aux.step_hist.rotational_mdot_boost_old.to_bits(),
        1.0_f64.to_bits()
    );

    for k in 0..m_rot.nz {
        m_rot.omega[k] = 0.85 * eval_upstream_omega_crit(&m_rot, k);
    }
    let avg_above = eval_surf_avg_rotation_info(&m_rot).v_div_v_crit;
    assert!(avg_above > 0.8, "expected v/vc > 0.8, got {avg_above}");
    let got_above = do_set_mdot(&mut m_rot).expect("v/vc > 0.8");
    let expected_boost = flexmesa_math::pow(1.0 - avg_above, -0.43);
    assert!(rel_err(got_above, base_cgs * expected_boost) < 1.0e-12);
    assert_eq!(
        m_rot.aux.step_hist.rotational_mdot_boost_old.to_bits(),
        expected_boost.to_bits()
    );

    // 2. Super-critical clamp `(1 - v/vc).max(1e-22)` stays finite & obeys
    //    `max_rotational_mdot_boost` (`winds.f90:438-442`).
    for k in 0..m_rot.nz {
        m_rot.omega[k] = 1.05 * eval_upstream_omega_crit(&m_rot, k);
    }
    m_rot.controls.mix_wind.max_rotational_mdot_boost = 500.0;
    let got_crit = do_set_mdot(&mut m_rot).expect("v/vc >= 1 clamp");
    assert!(got_crit.is_finite());
    assert_eq!(
        m_rot.aux.step_hist.rotational_mdot_boost_old.to_bits(),
        500.0_f64.to_bits()
    );
    assert!(rel_err(got_crit, base_cgs * 500.0) < 1.0e-12);

    // 3. `rotational_mdot_boost` records pre-`jump` / pre-`max_wind` boost,
    //    while `mdot_omega_power <= 0` or `!rotation_flag` skips boost.
    m_rot.aux.step_hist.mstar_dot_old = base_cgs;
    m_rot.controls.mix_wind.max_mdot_jump_for_rotation = 2.0;
    m_rot.controls.mix_wind.max_wind = 1.5e-6;
    let got_capped = do_set_mdot(&mut m_rot).expect("post-rot max_wind cap");
    assert_eq!(got_capped.to_bits(), (-1.5e-6 * M_SUN / SECYER).to_bits());
    assert_eq!(
        m_rot.aux.step_hist.rotational_mdot_boost_old.to_bits(),
        500.0_f64.to_bits()
    );

    m_rot.controls.mix_wind.max_wind = -1.0;
    m_rot.controls.mix_wind.mdot_omega_power = 0.0;
    let got_pow0 = do_set_mdot(&mut m_rot).expect("mdot_omega_power == 0");
    assert_eq!(got_pow0.to_bits(), base_cgs.to_bits());

    m_rot.controls.mix_wind.mdot_omega_power = 0.43;
    m_rot.controls.rotation_flag = false;
    let got_norot = do_set_mdot(&mut m_rot).expect("rotation_flag == false");
    assert_eq!(got_norot.to_bits(), base_cgs.to_bits());
}

#[test]
fn winds_159_v2_rlo_scale_height_and_super_eddington() {
    let base = zams_model(5.0);

    // 1. `eval_rlo_wind`: hydrostatic scale_height <= 0, 6 * ln10 cap,
    //    `rlo_wind_min_l` / `rlo_wind_max_teff`, and cosine `xfer_ratio` +
    //    positive `mass_change` accretion modulation (`winds.f90:133-141, 661-703`).
    let mut mr = base.clone();
    clear_wind_schemes(&mut mr);
    mr.controls.mix_wind.rlo_scaling_factor = 2.0;
    mr.controls.mix_wind.rlo_wind_base_mdot = 1.0e-8;
    mr.controls.mix_wind.rlo_wind_min_l = 1.0;
    mr.controls.mix_wind.rlo_wind_max_teff = 1.0e6;
    mr.controls.mix_wind.rlo_wind_scale_height = 0.0;
    mr.controls.mix_wind.roche_lobe_xfer_full_on = 0.90;
    mr.controls.mix_wind.roche_lobe_xfer_full_off = 1.00;

    let h_expected = mr.peos[0]
        / (mr.cgrav[0] * mr.m[0] * mr.rho[0] / (mr.r[0] * mr.r[0]))
        / R_SUN;
    mr.controls.mix_wind.rlo_wind_roche_lobe_radius = 1.0e-3;
    let mut xr = 1.0;
    let w_cap = eval_rlo_wind(&mr, 1.0e3, 10.0, 5.0e3, &mut xr);
    let exp_cap = 2.0 * 1.0e-8 * flexmesa_math::exp(6.0 * LN10);
    assert!(
        rel_err(w_cap, exp_cap) < 1.0e-13,
        "rlo 6*ln10 cap mismatch: got {w_cap:.12e}, exp {exp_cap:.12e}"
    );
    mr.controls.mix_wind.rlo_wind_roche_lobe_radius = 10.0 - h_expected;
    let w_one_h = eval_rlo_wind(&mr, 1.0e3, 10.0, 5.0e3, &mut xr);
    assert!(
        rel_err(w_one_h, 2.0 * 1.0e-8 * flexmesa_math::exp(1.0)) < 1.0e-12,
        "rlo hydrostatic scale_height mismatch"
    );

    // Check `min_l` and `max_teff` cutoffs:
    assert_eq!(eval_rlo_wind(&mr, 0.5, 10.0, 5.0e3, &mut xr).to_bits(), 0);
    assert_eq!(eval_rlo_wind(&mr, 1.0e3, 10.0, 2.0e6, &mut xr).to_bits(), 0);

    // Verify `do_set_mdot` applies midpoint `xfer_ratio = 0.5` to `mass_change > 0`:
    let phot_r = eval_photosphere_info(&mr);
    let l1_r = phot_r.l_lsun * L_SUN;
    let t_r = mr.teff;
    let r1_r = (l1_r / (PI * CRAD * CLIGHT * t_r * t_r * t_r * t_r)).sqrt();
    let r_rsun_r = r1_r / R_SUN;
    mr.controls.mix_wind.rlo_wind_roche_lobe_radius = r_rsun_r / 0.95;
    mr.controls.mass_change = 1.0e-5;
    let got_xfer = do_set_mdot(&mut mr).expect("rlo xfer midpoint accretion");
    assert!(
        rel_err(got_xfer, 0.5 * 1.0e-5 * M_SUN / SECYER) < 1.0e-12,
        "xfer midpoint mismatch: {got_xfer:.12e}"
    );
    assert!(!mr.aux.step_hist.doing_rlo_wind);

    // 2. Super-Eddington wind vs super-Eddington boost exclusion
    //    (`winds.f90:92-102, 216-236, 637-658`).
    let mut ms = base;
    clear_wind_schemes(&mut ms);
    let l_s = eval_photosphere_info(&ms).l_lsun * L_SUN;
    let t_s = ms.teff;
    let r_s = (l_s / (PI * CRAD * CLIGHT * t_s * t_s * t_s * t_s)).sqrt();
    let ledd_s = 0.4 * l_s;
    ms.aux.step_hist.prev_ledd = Some(ledd_s);
    ms.controls.super_eddington_scaling_factor = 0.0;
    ms.controls.mix_wind.hot_wind_scheme = String::from("Reimers");
    ms.controls.reimers_scaling_factor = 1.0;
    ms.controls.cool_wind_full_on_t = 1.0e3;
    ms.controls.hot_wind_full_on_t = 2.0e3;
    ms.controls.mix_wind.wind_boost_full_off_l_div_ledd = 1.0;
    ms.controls.mix_wind.wind_boost_full_on_l_div_ledd = 2.0;
    ms.controls.mix_wind.super_eddington_wind_max_boost = 10.0;
    let base_rei_cgs =
        eval_reimers_wind(l_s, ms.mstar, r_s, 1.0) * M_SUN / SECYER;
    let got_boost = do_set_mdot(&mut ms).expect("super-edd wind boost");
    assert_eq!(ms.aux.step_hist.super_eddington_wind_mdot.to_bits(), 0);
    assert!(rel_err(got_boost, -10.0 * base_rei_cgs) < 1.0e-13);

    // With `super_eddington_scaling_factor > 0`, boost is skipped and direct
    // super-Eddington wind is used when `Leff > prev_Ledd`.
    ms.controls.super_eddington_scaling_factor = 1.0;
    ms.controls.mix_wind.super_eddington_wind_ledd_factor = 1.0;
    let vesc2 = 2.0 * ms.cgrav[0] * ms.mstar / r_s;
    let exp_sedd = (l_s - ledd_s) / (0.5 * vesc2);
    let got_sedd = do_set_mdot(&mut ms).expect("direct super-edd wind");
    assert_eq!(
        ms.aux.step_hist.super_eddington_wind_mdot.to_bits(),
        exp_sedd.to_bits()
    );
    assert_eq!(got_sedd.to_bits(), (-exp_sedd).to_bits());
    let direct_sedd =
        eval_super_eddington_wind(&ms, l_s, ms.mstar, r_s, ledd_s);
    assert_eq!(direct_sedd.to_bits(), exp_sedd.to_bits());
}

#[test]
fn winds_159_v2_wind_gates_taper_and_envelope_limits() {
    // `no_wind_if_no_rotation` and `mass_change_full_off_dt` return 0 even
    // when `min_wind > 0` (`winds.f90:194-244`), whereas `starting_T_center`
    // and envelope limits zero `wind_mdot` after `min_wind`.
    let mut mg = zams_model(5.0);
    clear_wind_schemes(&mut mg);
    mg.controls.cool_wind_full_on_t = 1.0e3;
    mg.controls.hot_wind_full_on_t = 2.0e3;
    mg.controls.mix_wind.hot_wind_scheme = String::from("Reimers");
    mg.controls.reimers_scaling_factor = 1.0;
    mg.controls.mix_wind.min_wind = 1.0e-5;
    mg.controls.rotation_flag = false;
    mg.controls.mix_wind.no_wind_if_no_rotation = true;
    assert_eq!(
        do_set_mdot(&mut mg)
            .expect("no_wind_if_no_rotation overrides min_wind")
            .to_bits(),
        0
    );

    mg.controls.mix_wind.no_wind_if_no_rotation = false;
    mg.dt = 10.0;
    mg.controls.mix_wind.mass_change_full_off_dt = 10.0;
    mg.controls.mix_wind.mass_change_full_on_dt = 20.0;
    assert_eq!(
        do_set_mdot(&mut mg)
            .expect("dt <= full_off overrides min_wind")
            .to_bits(),
        0
    );

    mg.dt = 100.0;
    mg.controls.mix_wind.max_t_center_for_full_mass_loss = 5.0e8;
    mg.controls.mix_wind.max_t_center_for_any_mass_loss = 1.0e9;
    mg.aux.step_hist.starting_t_center = Some(1.0e7);
    let full_mg =
        do_set_mdot(&mut mg).expect("active wind below T_c threshold");
    assert!(full_mg < 0.0, "expected active mass loss below threshold");

    mg.aux.step_hist.starting_t_center = Some(7.5e8);
    let mid_mg = do_set_mdot(&mut mg).expect("T_c midpoint taper");
    assert!(rel_err(mid_mg, 0.5 * full_mg) < 1.0e-13);

    mg.aux.step_hist.starting_t_center = Some(2.0e9);
    assert_eq!(
        do_set_mdot(&mut mg)
            .expect("starting_T_center limit")
            .to_bits(),
        (-0.0_f64).to_bits()
    );

    mg.aux.step_hist.starting_t_center = Some(1.0e7);
    let cores = compute_core_masses(&mg);
    let h_he_env = mg.mstar / M_SUN - cores.co_core_mass;
    mg.controls.mix_wind.wind_h_he_envelope_limit = h_he_env + 0.05;
    assert_eq!(
        do_set_mdot(&mut mg).expect("H+He envelope limit").to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        mg.aux.step_hist.explicit_mstar_dot.to_bits(),
        (-0.0_f64).to_bits()
    );
}
