// SPDX-License-Identifier: GPL-3.0-or-later
//! Pinned MESA `26.4.1` IEEE-754 `0-ULP` parity tests for `winds.rs` and
//! `winds_driver.rs` (`#159`, `#140`), covering `star/hot_cool_wind`,
//! `star/high_z`, `star/7M_prems_to_AGB`, all wind schemes, super-Eddington,
//! RLO, and wind-driver controls without runtime `$MESA_DIR`.

#![forbid(unsafe_code)]
#![allow(clippy::float_cmp, clippy::unreadable_literal, clippy::too_many_lines)]

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{
    CLIGHT, CRAD, L_SUN, M_SUN, PI, R_SUN, SECYER, STANDARD_CGRAV,
};
use flexmesa_math::{with_numerics_mode, NumericsMode};
use flexmesa_star::{
    do_set_mdot, eval_bjorklund_wind, eval_blocker_wind, eval_de_jager_wind,
    eval_dutch_wind, eval_nieuwenhuijzen_wind, eval_nugis_lamers_wind,
    eval_reimers_wind, eval_rlo_wind, eval_super_eddington_wind,
    eval_van_loon_wind, eval_vink_wind, DutchLowTScheme, StarControls,
    StarJobControls, StarModel,
};

fn ulp_distance(a: f64, b: f64) -> u64 {
    if a == b {
        return 0;
    }
    let ua = a.to_bits().cast_signed();
    let ub = b.to_bits().cast_signed();
    if (ua < 0) != (ub < 0) {
        return u64::MAX;
    }
    (ua - ub).unsigned_abs()
}

/// MESA `26.4.1` `winds::set_mdot` reference rows (`case`, `role`, `field`,
/// hex bits) written by
/// `benchmarks/fortran_mesa/winds_oracles/gen_winds_oracles.f90`.
const WINDS_ORACLES: &str = include_str!(
    "../../../benchmarks/fortran_mesa/winds_oracles/mesa_26_4_1_winds_oracles.tsv"
);

/// Return the MESA bit pattern of `field` (role `in` or `out`) in `case`.
fn mesa_bits(case: &str, role: &str, field: &str) -> u64 {
    for line in WINDS_ORACLES.lines() {
        if line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() >= 5
            && cols[0] == case
            && cols[1] == role
            && cols[2] == field
        {
            let hex = cols[4].trim_start_matches("0x");
            return u64::from_str_radix(hex, 16).expect("hex column");
        }
    }
    panic!("no MESA oracle row for {case} {role} {field}");
}

/// MESA input value of `field` in `case`.
fn mesa_in(case: &str, field: &str) -> f64 {
    f64::from_bits(mesa_bits(case, "in", field))
}

/// MESA output value of `field` in `case`.
fn mesa_out(case: &str, field: &str) -> f64 {
    f64::from_bits(mesa_bits(case, "out", field))
}

/// Assert that the test's photospheric inputs are the ones the MESA
/// generator evaluated (`L_phot`, `M_phot`, `T_phot`).
fn assert_inputs(case: &str, l: f64, m: f64, t: f64) {
    assert_eq!(
        l.to_bits(),
        mesa_bits(case, "in", "L_phot"),
        "{case} L_phot"
    );
    assert_eq!(
        m.to_bits(),
        mesa_bits(case, "in", "M_phot"),
        "{case} M_phot"
    );
    assert_eq!(
        t.to_bits(),
        mesa_bits(case, "in", "T_phot"),
        "{case} T_phot"
    );
}

/// Assert `got` equals MESA's `s% mstar_dot` for `case`, bit for bit.
fn assert_mesa(case: &str, got: f64) {
    assert_zero_ulp(case, got, mesa_bits(case, "out", "mstar_dot"));
}

fn assert_zero_ulp(label: &str, got: f64, expected_bits: u64) {
    let expected = f64::from_bits(expected_bits);
    let ulps = ulp_distance(got, expected);
    assert_eq!(
        got.to_bits(),
        expected_bits,
        "{label}: got {got:.17e} (0x{:016X}), expected {expected:.17e} (0x{expected_bits:016X}), ulp_dist={ulps}",
        got.to_bits(),
    );
}

#[allow(clippy::too_many_arguments)]
fn make_wind_model(
    case: &str,
    l_phot: f64,
    m_phot: f64,
    t_phot: f64,
    z_base: f64,
    surf_h1: f64,
    surf_he4: f64,
    cen_h1: f64,
    cen_he4: f64,
) -> StarModel {
    assert_inputs(case, l_phot, m_phot, t_phot);
    for (field, value) in [
        ("Zbase", z_base),
        ("surface_h1", surf_h1),
        ("surface_he4", surf_he4),
        ("center_h1", cen_h1),
        ("center_he4", cen_he4),
    ] {
        assert_eq!(
            value.to_bits(),
            mesa_bits(case, "in", field),
            "{case} {field}"
        );
    }
    let mut ctrl = StarControls {
        tau_factor: 1.0,
        cool_wind_rgb_scheme: String::new(),
        cool_wind_agb_scheme: String::new(),
        reimers_scaling_factor: 0.0,
        blocker_scaling_factor: 0.0,
        vink_scaling_factor: 0.0,
        dutch_scaling_factor: 0.0,
        super_eddington_scaling_factor: 0.0,
        mass_change: 0.0,
        rotation_flag: false,
        zbase: z_base,
        ..StarControls::default()
    };
    ctrl.mix_wind.hot_wind_scheme = String::new();
    ctrl.mix_wind.de_jager_scaling_factor = 0.0;
    ctrl.mix_wind.van_loon_scaling_factor = 0.0;
    ctrl.mix_wind.nieuwenhuijzen_scaling_factor = 0.0;
    ctrl.mix_wind.bjorklund_scaling_factor = 0.0;
    ctrl.mix_wind.dutch_wind_lowt_scheme = String::from("de Jager");
    ctrl.mix_wind.rlo_scaling_factor = 0.0;
    ctrl.mix_wind.no_wind_if_no_rotation = false;
    ctrl.mix_wind.mass_change_full_on_dt = 0.0;
    ctrl.mix_wind.mass_change_full_off_dt = 0.0;
    ctrl.mix_wind.min_wind = 0.0;
    ctrl.mix_wind.max_wind = 0.0;
    ctrl.mix_wind.wind_boost_full_off_l_div_ledd = 1.0;
    ctrl.mix_wind.wind_boost_full_on_l_div_ledd = 2.0;
    ctrl.mix_wind.super_eddington_wind_max_boost = 1.0;
    ctrl.mix_wind.max_t_center_for_any_mass_loss = 1.0e99;
    ctrl.mix_wind.max_t_center_for_full_mass_loss = 1.0e99;
    ctrl.mix_wind.wind_h_envelope_limit = 0.0;
    ctrl.mix_wind.wind_h_he_envelope_limit = 0.0;
    ctrl.mix_wind.wind_he_layer_limit = 0.0;

    let catalog = NuclideCatalog::builtin_ref();
    let h1_id = catalog.get_nuclide_index("h1").expect("h1 nuclide");
    let he4_id = catalog.get_nuclide_index("he4").expect("he4 nuclide");
    let mut m = StarModel::new(
        2,
        vec![h1_id, he4_id],
        StarJobControls::default(),
        ctrl,
    );
    m.kap_config.zbase = z_base;
    m.dt = 1.0e10;
    m.mstar = m_phot;
    m.xmstar = m_phot;
    m.m_center = 0.0;
    m.teff = t_phot;
    let r_bb = (l_phot
        / (PI * CRAD * CLIGHT * t_phot * t_phot * t_phot * t_phot))
        .sqrt();
    m.lum = vec![l_phot, l_phot];
    m.m_grav = vec![m_phot, 0.5 * m_phot];
    m.m = vec![m_phot, 0.5 * m_phot];
    m.q = vec![1.0, 0.5];
    m.dq = vec![0.5, 0.5];
    m.dm = vec![0.5 * m_phot, 0.5 * m_phot];
    m.r = vec![r_bb, 0.5 * r_bb];
    m.cgrav = vec![STANDARD_CGRAV, STANDARD_CGRAV];
    m.t = vec![t_phot, 1.0e7];
    m.rho = vec![1.0e-9, 1.0];
    m.peos = vec![1.0e3, 1.0e15];
    m.xa = vec![surf_h1, surf_he4, cen_h1, cen_he4];
    m.aux.step_hist.prev_ledd = Some(1.0e40);
    m.aux.step_hist.starting_t_center = Some(1.0e7);
    m
}

#[test]
fn test_mesa_26_4_1_target_cases_step1_and_blends_zero_ulp() {
    with_numerics_mode(NumericsMode::Strict, || {
        // 1. `star/hot_cool_wind` step 1 (Vink cool branch, Zbase = 0.02).
        let mut m1 = make_wind_model(
            "hot_cool_wind_step1",
            f64::from_bits(0x4795BD6A8B756743),
            f64::from_bits(0x470572069D3FA0C4),
            f64::from_bits(0x40D4769F25A30032),
            0.02,
            0.70,
            0.28,
            0.70,
            0.28,
        );
        m1.controls.cool_wind_full_on_t = 8.0e3;
        m1.controls.hot_wind_full_on_t = 1.2e4;
        m1.controls.cool_wind_rgb_scheme = String::from("Reimers");
        m1.controls.mix_wind.rgb_to_agb_wind_switch = -1.0;
        m1.controls.reimers_scaling_factor = 1.0;
        m1.controls.mix_wind.hot_wind_scheme = String::from("Vink");
        m1.controls.vink_scaling_factor = 1.0;
        let mdot1 = do_set_mdot(&mut m1).expect("hot_cool_wind step1");
        assert_mesa("hot_cool_wind_step1", mdot1);

        // 2. `star/hot_cool_wind` hot/cool blend midpoint T=10000 K.
        let mut m2 = make_wind_model(
            "hot_cool_wind_blend",
            f64::from_bits(0x47D932E7005377AA),
            f64::from_bits(0x47056A2ED5580A3B),
            f64::from_bits(0x40C3880000000000),
            0.02,
            0.70,
            0.28,
            0.70,
            0.28,
        );
        m2.controls = m1.controls.clone();
        let mdot2 = do_set_mdot(&mut m2).expect("hot_cool_wind blend");
        assert_mesa("hot_cool_wind_blend", mdot2);

        // 3. `star/high_z` Zbase = 0.07 Vink wind.
        let mut m3 = make_wind_model(
            "high_z_z07",
            f64::from_bits(0x47A118695D5CED92),
            f64::from_bits(0x470572069D3FA0C4),
            f64::from_bits(0x40D202C000000000),
            0.07,
            0.55,
            0.38,
            0.55,
            0.38,
        );
        m3.controls = m1.controls.clone();
        let mdot3 = do_set_mdot(&mut m3).expect("high_z z=0.07");
        assert_mesa("high_z_z07", mdot3);

        // 4. `star/7M_prems_to_AGB` pre-MS Reimers (Zbase = 0.001, eta = 0.5).
        let mut m4 = make_wind_model(
            "prems_7m_reimers",
            f64::from_bits(0x479A1614A8ED5870),
            f64::from_bits(0x470572069D3FA0C4),
            f64::from_bits(0x40B2809431B74921),
            0.001,
            0.757,
            0.242,
            0.757,
            0.242,
        );
        m4.controls.cool_wind_full_on_t = 9.99e9;
        m4.controls.hot_wind_full_on_t = 1.0e10;
        m4.controls.cool_wind_rgb_scheme = String::from("Reimers");
        m4.controls.cool_wind_agb_scheme = String::from("Blocker");
        m4.controls.mix_wind.rgb_to_agb_wind_switch = 1.0e-4;
        m4.controls.reimers_scaling_factor = 0.5;
        m4.controls.blocker_scaling_factor = 0.0003;
        let mdot4 = do_set_mdot(&mut m4).expect("7M prems Reimers");
        assert_mesa("prems_7m_reimers", mdot4);

        // 5. `star/7M_prems_to_AGB` AGB Blocker switch (center_h1 < 0.01, center_he4 < 1e-4).
        let mut m5 = make_wind_model(
            "agb_7m_blocker",
            f64::from_bits(0x47CCCC75B73AD1E7),
            f64::from_bits(0x470533485E02EC76),
            f64::from_bits(0x40B0040000000000),
            0.001,
            0.757,
            0.242,
            0.0,
            1.0e-5,
        );
        m5.controls = m4.controls.clone();
        let mdot5 = do_set_mdot(&mut m5).expect("7M AGB Blocker");
        assert_mesa("agb_7m_blocker", mdot5);
    });
}

#[test]
fn test_mesa_26_4_1_vink_and_bjorklund_zero_ulp() {
    with_numerics_mode(NumericsMode::Strict, || {
        let l1 = f64::from_bits(0x47F1FFC99284C330); // 1e5 * L_SUN
        let m1 = f64::from_bits(0x4726FA2BA87B07AE); // 30 * M_SUN

        let to_cgs = |w: f64| -w * M_SUN / SECYER;

        // Inputs the MESA generator evaluated (`T_phot`, `Zbase`, scaling).
        for (case, t, z, eta) in [
            ("vink_hot_35k", 0x40E1170000000000_u64, 0.005, 0.9),
            ("vink_hot_30k", 0x40DD4C0000000000, 0.005, 0.9),
            ("vink_cool_18k", 0x40D1940000000000, 0.005, 0.9),
            ("vink_jump_mid", 0x40D946B0A3D70A3D, 0.019, 1.0),
            ("vink_jump_25950", 0x40D9578000000000, 0.019, 1.0),
            ("vink_lowz_1e_12", 0x40E1170000000000, 1.0e-12, 1.0),
        ] {
            assert_inputs(case, l1, m1, f64::from_bits(t));
            assert_eq!(mesa_in(case, "Zbase"), z, "{case} Zbase");
            assert_eq!(mesa_in(case, "Vink_scaling_factor"), eta, "{case} eta");
        }

        // Vink hot side (T = 35000 K and T = 30000 K, Z = 0.005, eta = 0.9) — hits pow2(log10(T1/4e4)).
        assert_mesa(
            "vink_hot_35k",
            to_cgs(eval_vink_wind(
                l1,
                m1,
                f64::from_bits(0x40E1170000000000),
                0.005,
                0.9,
            )),
        );
        assert_mesa(
            "vink_hot_30k",
            to_cgs(eval_vink_wind(
                l1,
                m1,
                f64::from_bits(0x40DD4C0000000000),
                0.005,
                0.9,
            )),
        );
        // Vink cool side (T = 18000 K, Z = 0.005, eta = 0.9).
        assert_mesa(
            "vink_cool_18k",
            to_cgs(eval_vink_wind(
                l1,
                m1,
                f64::from_bits(0x40D1940000000000),
                0.005,
                0.9,
            )),
        );
        // Vink jump ramp midpoint (T = 25882.76 K, Z = 0.019, eta = 1.0).
        assert_mesa(
            "vink_jump_mid",
            to_cgs(eval_vink_wind(
                l1,
                m1,
                f64::from_bits(0x40D946B0A3D70A3D),
                0.019,
                1.0,
            )),
        );
        // Vink jump ramp edge (T = 25950 K, Z = 0.019, eta = 1.0).
        assert_mesa(
            "vink_jump_25950",
            to_cgs(eval_vink_wind(
                l1,
                m1,
                f64::from_bits(0x40D9578000000000),
                0.019,
                1.0,
            )),
        );
        // Vink ultra-low Z = 1e-12 (< 1e-10 * Z_solar): verifies unclamped Z/Zsolar!
        assert_mesa(
            "vink_lowz_1e_12",
            to_cgs(eval_vink_wind(
                l1,
                m1,
                f64::from_bits(0x40E1170000000000),
                1.0e-12,
                1.0,
            )),
        );

        // Bjorklund standard Z = 0.014 and ultra-low Z = 1e-13 (< 1e-10 * Z_bjork).
        let l_bj = f64::from_bits(0x48267FBBF725F3FC); // 1e6 * L_SUN
        let m_bj = f64::from_bits(0x47313BA0BE5C45C2); // 45 * M_SUN
        let t_bj = f64::from_bits(0x40E5F90000000000); // 45000 K
        for (case, z) in [("bjorklund_std", 0.014), ("bjorklund_lowz", 1.0e-13)]
        {
            assert_inputs(case, l_bj, m_bj, t_bj);
            assert_eq!(mesa_in(case, "Zbase"), z, "{case} Zbase");
            assert_eq!(mesa_in(case, "Bjorklund_scaling_factor"), 0.8);
            assert_eq!(mesa_in(case, "cgrav1"), STANDARD_CGRAV);
        }
        assert_mesa(
            "bjorklund_std",
            to_cgs(eval_bjorklund_wind(
                l_bj,
                m_bj,
                t_bj,
                0.014,
                STANDARD_CGRAV,
                0.8,
            )),
        );
        assert_mesa(
            "bjorklund_lowz",
            to_cgs(eval_bjorklund_wind(
                l_bj,
                m_bj,
                t_bj,
                1.0e-13,
                STANDARD_CGRAV,
                0.8,
            )),
        );
    });
}

#[test]
fn test_mesa_26_4_1_dutch_cool_sedd_and_rlo_zero_ulp() {
    with_numerics_mode(NumericsMode::Strict, || {
        let l1 = f64::from_bits(0x47F1FFC99284C330); // 1e5 * L_SUN
        let m30 = f64::from_bits(0x4726FA2BA87B07AE); // 30 * M_SUN
        let m25 = f64::from_bits(0x472325CF0C668666); // 25 * M_SUN
        let m15 = f64::from_bits(0x4716FA2BA87B07AE); // 15 * M_SUN
        let r_of =
            |l: f64, t: f64| (l / (PI * CRAD * CLIGHT * t * t * t * t)).sqrt();
        let to_cgs = |w: f64| -w * M_SUN / SECYER;

        // Dutch high-T Vink & WR Nugis-Lamers.
        let t25k = f64::from_bits(0x40D86A0000000000);
        assert_mesa(
            "dutch_vink",
            to_cgs(eval_dutch_wind(
                l1,
                m30,
                r_of(l1, t25k),
                t25k,
                0.70,
                0.28,
                0.02,
                DutchLowTScheme::DeJager,
                0.8,
            )),
        );
        assert_mesa(
            "dutch_wr_nl",
            to_cgs(eval_dutch_wind(
                l1,
                m15,
                r_of(l1, t25k),
                t25k,
                0.20,
                0.78,
                0.02,
                DutchLowTScheme::DeJager,
                0.8,
            )),
        );

        // Dutch low-T (T = 4000 K) variants.
        let t4k = f64::from_bits(0x40AF400000000000);
        assert_mesa(
            "dutch_low_dejager",
            to_cgs(eval_dutch_wind(
                l1,
                m25,
                r_of(l1, t4k),
                t4k,
                0.70,
                0.28,
                0.02,
                DutchLowTScheme::DeJager,
                0.8,
            )),
        );
        assert_mesa(
            "dutch_low_vanloon",
            to_cgs(eval_dutch_wind(
                l1,
                m25,
                r_of(l1, t4k),
                t4k,
                0.70,
                0.28,
                0.02,
                DutchLowTScheme::VanLoon,
                0.8,
            )),
        );
        assert_mesa(
            "dutch_low_nieuwenhuijzen",
            to_cgs(eval_dutch_wind(
                l1,
                m25,
                r_of(l1, t4k),
                t4k,
                0.70,
                0.28,
                0.02,
                DutchLowTScheme::Nieuwenhuijzen,
                0.8,
            )),
        );

        // Dutch transition blend (T = 10500 K, van Loon).
        let t10500 = f64::from_bits(0x40C4820000000000);
        assert_mesa(
            "dutch_blend_10500",
            to_cgs(eval_dutch_wind(
                l1,
                m25,
                r_of(l1, t10500),
                t10500,
                0.70,
                0.28,
                0.02,
                DutchLowTScheme::VanLoon,
                0.8,
            )),
        );

        // Direct de Jager, van Loon, Nieuwenhuijzen at T = 15000 K.
        let t15k = f64::from_bits(0x40CD4C0000000000);

        // Inputs the MESA generator evaluated for the scheme cases above.
        for (case, m, t, x, y, eta_field, eta) in [
            (
                "dutch_vink",
                m30,
                t25k,
                0.70,
                0.28,
                "Dutch_scaling_factor",
                0.8,
            ),
            (
                "dutch_wr_nl",
                m15,
                t25k,
                0.20,
                0.78,
                "Dutch_scaling_factor",
                0.8,
            ),
            (
                "dutch_low_dejager",
                m25,
                t4k,
                0.70,
                0.28,
                "Dutch_scaling_factor",
                0.8,
            ),
            (
                "dutch_low_vanloon",
                m25,
                t4k,
                0.70,
                0.28,
                "Dutch_scaling_factor",
                0.8,
            ),
            (
                "dutch_low_nieuwenhuijzen",
                m25,
                t4k,
                0.70,
                0.28,
                "Dutch_scaling_factor",
                0.8,
            ),
            (
                "dutch_blend_10500",
                m25,
                t10500,
                0.70,
                0.28,
                "Dutch_scaling_factor",
                0.8,
            ),
            (
                "direct_dejager",
                m30,
                t15k,
                0.70,
                0.28,
                "de_Jager_scaling_factor",
                0.7,
            ),
            (
                "direct_vanloon",
                m30,
                t15k,
                0.70,
                0.28,
                "van_Loon_scaling_factor",
                0.6,
            ),
            (
                "direct_nieuwenhuijzen",
                m25,
                t15k,
                0.70,
                0.28,
                "Nieuwenhuijzen_scaling_factor",
                0.5,
            ),
            (
                "direct_reimers",
                m25,
                t4k,
                0.70,
                0.28,
                "Reimers_scaling_factor",
                0.5,
            ),
            (
                "direct_blocker",
                m25,
                t4k,
                0.70,
                0.28,
                "Blocker_scaling_factor",
                0.5,
            ),
        ] {
            assert_inputs(case, l1, m, t);
            assert_eq!(mesa_in(case, "Zbase"), 0.02, "{case} Zbase");
            assert_eq!(mesa_in(case, "surface_h1"), x, "{case} surface_h1");
            assert_eq!(mesa_in(case, "surface_he4"), y, "{case} surface_he4");
            assert_eq!(mesa_in(case, eta_field), eta, "{case} {eta_field}");
        }
        assert_mesa(
            "direct_dejager",
            to_cgs(eval_de_jager_wind(l1, t15k, 0.7)),
        );
        assert_mesa(
            "direct_vanloon",
            to_cgs(eval_van_loon_wind(l1, t15k, 0.6)),
        );
        assert_mesa(
            "direct_nieuwenhuijzen",
            to_cgs(eval_nieuwenhuijzen_wind(l1, m25, r_of(l1, t15k), 0.5)),
        );

        // Direct Reimers and Blocker (scaling factor 0.5) against MESA; the
        // Nugis-Lamers helper is checked through `dutch_wr_nl` above and only
        // sanity-checked here.
        assert_mesa(
            "direct_reimers",
            to_cgs(eval_reimers_wind(l1, m25, r_of(l1, t4k), 0.5)),
        );
        assert_mesa(
            "direct_blocker",
            to_cgs(eval_blocker_wind(l1, m25, r_of(l1, t4k), 0.5)),
        );
        assert!(eval_nugis_lamers_wind(l1, 0.78, 0.02, 0.8) > 0.0);

        // Super-Eddington wind (L = 1e6 Lsun, M = 20 Msun, T = 30000 K, prev_Ledd = 2e5 Lsun).
        let l_se = f64::from_bits(0x48267FBBF725F3FC);
        let m20 = f64::from_bits(0x471EA2E4E0A40A3D);
        let t30k = f64::from_bits(0x40DD4C0000000000);
        let mut ms = make_wind_model(
            "super_edd_wind",
            l_se,
            m20,
            t30k,
            0.02,
            0.70,
            0.28,
            0.70,
            0.28,
        );
        ms.controls.super_eddington_scaling_factor = 0.7;
        ms.controls.mix_wind.super_eddington_wind_ledd_factor = 1.2;
        ms.aux.step_hist.prev_ledd = Some(2.0e5 * L_SUN);
        let mdot_se = do_set_mdot(&mut ms).expect("super_edd_wind");
        assert_mesa("super_edd_wind", mdot_se);
        assert_zero_ulp(
            "super_edd_wind_record",
            ms.aux.step_hist.super_eddington_wind_mdot,
            mesa_bits("super_edd_wind", "out", "super_eddington_wind_mdot"),
        );
        let sedd_direct = eval_super_eddington_wind(
            &ms,
            l_se,
            m20,
            r_of(l_se, t30k),
            2.0e5 * L_SUN,
        );
        assert_zero_ulp(
            "super_edd_direct",
            sedd_direct,
            mesa_bits("super_edd_wind", "out", "super_eddington_wind_mdot"),
        );

        // Super-Eddington boost to Vink wind.
        let l_bst = f64::from_bits(0x4801FFC99284C330); // 2e5 * L_SUN
        let mut mb = make_wind_model(
            "super_edd_boost",
            l_bst,
            m20,
            t30k,
            0.02,
            0.70,
            0.28,
            0.70,
            0.28,
        );
        mb.controls.cool_wind_full_on_t = 8.0e3;
        mb.controls.hot_wind_full_on_t = 1.2e4;
        mb.controls.mix_wind.hot_wind_scheme = String::from("Vink");
        mb.controls.vink_scaling_factor = 1.0;
        mb.controls.super_eddington_scaling_factor = 0.0;
        mb.controls.mix_wind.wind_boost_full_off_l_div_ledd = 1.0;
        mb.controls.mix_wind.wind_boost_full_on_l_div_ledd = 3.0;
        mb.controls.mix_wind.super_eddington_wind_max_boost = 5.0;
        mb.aux.step_hist.prev_ledd = Some(1.0e5 * L_SUN);
        let mdot_bst = do_set_mdot(&mut mb).expect("super_edd_boost");
        assert_mesa("super_edd_boost", mdot_bst);

        // RLO overflow and xfer_ratio scaling.
        let l_rlo = f64::from_bits(0x479D4AAD50AAAA71);
        let m5 = f64::from_bits(0x46FEA2E4E0A40A3D);
        let mut mr = make_wind_model(
            "rlo_overflow",
            l_rlo,
            m5,
            t4k,
            0.02,
            0.70,
            0.28,
            0.70,
            0.28,
        );
        mr.controls.mix_wind.rlo_scaling_factor = 0.8;
        mr.controls.mix_wind.rlo_wind_min_l = 10.0;
        mr.controls.mix_wind.rlo_wind_max_teff = 1.0e4;
        mr.controls.mix_wind.rlo_wind_scale_height = 0.0;
        mr.controls.mix_wind.rlo_wind_roche_lobe_radius = 100.0;
        mr.controls.mix_wind.rlo_wind_base_mdot = 1.0e-8;
        mr.r[0] = 105.0 * R_SUN;
        let mdot_rlo = do_set_mdot(&mut mr).expect("rlo_overflow");
        assert_mesa("rlo_overflow", mdot_rlo);

        let l_xfer = f64::from_bits(0x4797FA5F6F4FEA26);
        let mut mx = make_wind_model(
            "rlo_xfer_ratio",
            l_xfer,
            m5,
            t4k,
            0.02,
            0.70,
            0.28,
            0.70,
            0.28,
        );
        mx.controls.mix_wind = mr.controls.mix_wind.clone();
        mx.controls.mix_wind.roche_lobe_xfer_full_on = 0.90;
        mx.controls.mix_wind.roche_lobe_xfer_full_off = 1.00;
        mx.controls.mass_change = 1.0e-6;
        mx.r[0] = 95.0 * R_SUN;
        let mut xfer = 1.0;
        let rlo_sub = eval_rlo_wind(
            &mx,
            l_xfer / L_SUN,
            r_of(l_xfer, t4k) / R_SUN,
            t4k,
            &mut xfer,
        );
        assert_eq!(rlo_sub, 0.0);
        // MESA: `eval_rlo_wind` returned zero (`doing_rlo_wind` false).
        assert_eq!(mesa_out("rlo_xfer_ratio", "doing_rlo_wind"), 0.0);
        let mdot_xfer = do_set_mdot(&mut mx).expect("rlo_xfer_ratio");
        assert_mesa("rlo_xfer_ratio", mdot_xfer);

        // Timestep ramp (`dt_ramp_mid`).
        let mut mdt = make_wind_model(
            "dt_ramp_mid",
            f64::from_bits(0x47F1FFC99284C330),
            f64::from_bits(0x4726FA2BA87B07AE),
            f64::from_bits(0x40E1170000000000),
            0.02,
            0.70,
            0.28,
            0.70,
            0.28,
        );
        mdt.controls.cool_wind_full_on_t = 8.0e3;
        mdt.controls.hot_wind_full_on_t = 1.2e4;
        mdt.controls.mix_wind.hot_wind_scheme = String::from("Vink");
        mdt.controls.vink_scaling_factor = 1.0;
        mdt.controls.mix_wind.mass_change_full_off_dt = 1.0e6;
        mdt.controls.mix_wind.mass_change_full_on_dt = 5.0e6;
        mdt.dt = 2.5e6;
        let mdot_dt = do_set_mdot(&mut mdt).expect("dt_ramp_mid");
        assert_mesa("dt_ramp_mid", mdot_dt);

        // Central T taper + Zbase < 0 (`1 - (X + Y)` fallback).
        let mut mtc = make_wind_model(
            "tcenter_taper_zneg",
            f64::from_bits(0x47F1FFC99284C330),
            f64::from_bits(0x4726FA2BA87B07AE),
            f64::from_bits(0x40E1170000000000),
            -1.0,
            0.68,
            0.29,
            0.68,
            0.29,
        );
        mtc.controls.cool_wind_full_on_t = 8.0e3;
        mtc.controls.hot_wind_full_on_t = 1.2e4;
        mtc.controls.mix_wind.hot_wind_scheme = String::from("Vink");
        mtc.controls.vink_scaling_factor = 1.0;
        mtc.aux.step_hist.starting_t_center = Some(1.5e9);
        mtc.controls.mix_wind.max_t_center_for_full_mass_loss = 1.0e9;
        mtc.controls.mix_wind.max_t_center_for_any_mass_loss = 2.0e9;
        let mdot_tc = do_set_mdot(&mut mtc).expect("tcenter_taper_zneg");
        assert_mesa("tcenter_taper_zneg", mdot_tc);

        // `hot_wind_full_on_T == cool_wind_full_on_T` tie.
        let mut mtie = make_wind_model(
            "hot_cool_tie",
            f64::from_bits(0x47F1FFC99284C330),
            f64::from_bits(0x471EA2E4E0A40A3D),
            f64::from_bits(0x40C3880000000000),
            0.02,
            0.70,
            0.28,
            0.70,
            0.28,
        );
        mtie.controls.cool_wind_full_on_t = 1.0e4;
        mtie.controls.hot_wind_full_on_t = 1.0e4;
        mtie.controls.mix_wind.hot_wind_scheme = String::from("Vink");
        mtie.controls.vink_scaling_factor = 1.0;
        mtie.controls.cool_wind_rgb_scheme = String::from("Reimers");
        mtie.controls.reimers_scaling_factor = 1.0;
        let mdot_tie = do_set_mdot(&mut mtie).expect("hot_cool_tie");
        assert_mesa("hot_cool_tie", mdot_tie);

        // `min_wind` floor and `max_wind` cap.
        let mut mmm = make_wind_model(
            "min_wind_floor",
            f64::from_bits(0x47F1FFC99284C330),
            f64::from_bits(0x471EA2E4E0A40A3D),
            f64::from_bits(0x40CD4C0000000000),
            0.02,
            0.70,
            0.28,
            0.70,
            0.28,
        );
        mmm.controls.mix_wind.hot_wind_scheme = String::from("Vink");
        mmm.controls.vink_scaling_factor = 1.0;
        mmm.controls.mix_wind.min_wind = 1.0e-4;
        let mdot_min = do_set_mdot(&mut mmm).expect("min_wind_floor");
        assert_mesa("min_wind_floor", mdot_min);
        mmm.controls.mix_wind.min_wind = 0.0;
        mmm.controls.mix_wind.max_wind = 1.0e-9;
        mmm.teff = f64::from_bits(0x40E1170000000000);
        assert_inputs(
            "max_wind_cap",
            f64::from_bits(0x47F1FFC99284C330),
            f64::from_bits(0x471EA2E4E0A40A3D),
            mmm.teff,
        );
        let mdot_max = do_set_mdot(&mut mmm).expect("max_wind_cap");
        assert_mesa("max_wind_cap", mdot_max);

        // Envelope mass wind cutoff (`wind_h_envelope_limit`) with 3-zone core boundary.
        let l_env = f64::from_bits(0x47F1FFC99284C330);
        let m7 = f64::from_bits(0x470572069D3FA0C4);
        let t_env = f64::from_bits(0x40D3880000000000);
        let catalog = NuclideCatalog::builtin_ref();
        let h1_id = catalog.get_nuclide_index("h1").expect("h1 nuclide");
        let he4_id = catalog.get_nuclide_index("he4").expect("he4 nuclide");
        let mut env_ctrl = mr.controls.clone();
        env_ctrl.mix_wind.rlo_scaling_factor = 0.0;
        env_ctrl.mix_wind.hot_wind_scheme = String::from("Vink");
        env_ctrl.vink_scaling_factor = 1.0;
        env_ctrl.mix_wind.wind_h_envelope_limit = 0.1;
        env_ctrl.tau_factor = 1.0;
        env_ctrl.zbase = 0.02;
        let mut menv = StarModel::new(
            3,
            vec![h1_id, he4_id],
            StarJobControls::default(),
            env_ctrl,
        );
        menv.kap_config.zbase = 0.02;
        menv.dt = 1.0e10;
        menv.mstar = m7;
        menv.xmstar = m7;
        menv.teff = t_env;
        let r_env = r_of(l_env, t_env);
        assert_inputs("h_env_limit_zero", l_env, m7, t_env);
        menv.lum = vec![l_env, l_env, l_env];
        menv.r = vec![r_env, 0.99 * r_env, 0.5 * r_env];
        menv.q = vec![1.0, 6.99 / 7.0, 6.98 / 7.0];
        menv.dq = vec![0.01 / 7.0, 0.01 / 7.0, 6.98 / 7.0];
        menv.dm = vec![0.01 * M_SUN, 0.01 * M_SUN, 6.98 * M_SUN];
        menv.m = vec![7.0 * M_SUN, 6.99 * M_SUN, 6.98 * M_SUN];
        menv.m_grav = menv.m.clone();
        menv.cgrav = vec![STANDARD_CGRAV; 3];
        menv.t = vec![t_env, 1.0e6, 1.0e7];
        menv.ln_t = menv.t.iter().map(|v| flexmesa_math::log(*v)).collect();
        menv.rho = vec![1.0e-9, 1.0e-3, 1.0];
        menv.lnd = menv.rho.iter().map(|v| flexmesa_math::log(*v)).collect();
        menv.peos = vec![1.0e3, 1.0e9, 1.0e15];
        menv.xa = vec![0.70, 0.28, 0.0, 0.98, 0.0, 0.98];
        menv.aux.step_hist.prev_ledd = Some(1.0e40);
        menv.aux.step_hist.starting_t_center = Some(1.0e7);
        let mdot_env = do_set_mdot(&mut menv).expect("h_env_limit_zero");
        assert_eq!(mdot_env, 0.0);
        assert_eq!(mesa_out("h_env_limit_zero", "mstar_dot"), 0.0);

        // `no_wind_if_no_rotation` and `dt <= mass_change_full_off_dt` gates.
        let mut mgate = make_wind_model(
            "no_wind_if_no_rotation",
            l1,
            m30,
            t15k,
            0.02,
            0.70,
            0.28,
            0.70,
            0.28,
        );
        mgate.controls.mix_wind.hot_wind_scheme = String::from("Vink");
        mgate.controls.vink_scaling_factor = 1.0;
        mgate.controls.mix_wind.no_wind_if_no_rotation = true;
        assert_eq!(
            do_set_mdot(&mut mgate).expect("no_wind_if_no_rotation"),
            mesa_out("no_wind_if_no_rotation", "mstar_dot")
        );
        mgate.controls.mix_wind.no_wind_if_no_rotation = false;
        mgate.controls.mix_wind.mass_change_full_off_dt = 1.0e6;
        mgate.controls.mix_wind.mass_change_full_on_dt = 5.0e6;
        mgate.dt = 1.0e6;
        assert_eq!(
            do_set_mdot(&mut mgate).expect("dt_full_off"),
            mesa_out("dt_full_off", "mstar_dot")
        );
    });
}
