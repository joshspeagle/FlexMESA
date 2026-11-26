// SPDX-License-Identifier: GPL-3.0-or-later
//! Regression, operation-order parity, and edge-branch coverage tests for
//! `PC` (`pc/*`) and `Skye` (`skye/*`) EOS modules (`#161`).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::manual_midpoint,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use flexmesa_autodiff::{AutoDiffReal2VarOrder1, AutoDiffReal2VarOrder3};
use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_const::{AMU, CRAD, KERG, MP};
use flexmesa_eos::pc::blin::{blin9, blin9a, blin9b, blin9c};
use flexmesa_eos::pc::coulomb_ion::hlfit12;
use flexmesa_eos::pc::coulomb_screen::{cormix, fscr_liq8, fscr_sol8};
use flexmesa_eos::pc::elect::{elect11, elect11b};
use flexmesa_eos::pc::excor::{excor7, excor7_analytic};
use flexmesa_eos::pc::ferinv::{chemfit, ferinv7};
use flexmesa_eos::pc::{eval_pc_eos_from_weights, eval_pc_eos_zones, melange9};
use flexmesa_eos::skye::coulomb_solid::{
    delta_g_ogata93, delta_g_pc13, ocp_solid_harmonic_free_energy,
    solid_mixing_rule_correction,
};
use flexmesa_eos::skye::eval_skye_eos_zones;
use flexmesa_eos::{
    eval_pc_eos, eval_skye_eos, eval_skye_eos_from_weights, get_pc_alfa,
    get_skye_alfa, is_contained, min_distance_to_polygon, HelmTable, PcConfig,
    SkyeConfig, SkyeSolidMixingRule, I_LN_E, I_LN_S,
};
use flexmesa_math::{
    exp10, with_numerics_mode, NumericsMode, SAFE_LOG_BAD_RETURN,
};

fn assert_close_rel(
    actual: f64,
    expected: f64,
    rtol: f64,
    atol: f64,
    label: &str,
) {
    let diff = (actual - expected).abs();
    let tol = atol.max(rtol * expected.abs());
    assert!(
        diff <= tol,
        "{label}: actual={actual:e}, expected={expected:e}, diff={diff:e}, tol={tol:e}"
    );
}

#[test]
fn test_pc_safe_log_preserves_derivatives_below_floor_and_bad_return() {
    with_numerics_mode(NumericsMode::Strict, || {
        // Direct check on AutoDiffReal2VarOrder1::safe_log vs eospc_eval semantics.
        let neg = AutoDiffReal2VarOrder1 {
            val: -2.5,
            d1: [1.5, -0.75],
        };
        let neg_ln = neg.safe_log();
        assert_eq!(neg_ln.val.to_bits(), flexmesa_math::log(1.0e-99).to_bits());
        assert_eq!(
            neg_ln.d1[0].to_bits(),
            (1.5_f64 * flexmesa_math::powm1(-2.5_f64)).to_bits()
        );
        assert_eq!(
            neg_ln.d1[1].to_bits(),
            (-0.75_f64 * flexmesa_math::powm1(-2.5_f64)).to_bits()
        );

        let nan_in = AutoDiffReal2VarOrder1 {
            val: f64::NAN,
            d1: [1.0, 2.0],
        };
        assert_eq!(nan_in.safe_log().val, SAFE_LOG_BAD_RETURN);

        // Cryogenic Fe56 crystal where Coulomb Madelung energy dominates
        // Fermi energy, driving u_nkt < 0 so safe_log(u_nkt * n * kerg * T)
        // hits the floor branch.
        let cfg = PcConfig {
            include_radiation: false,
            ..PcConfig::default()
        };
        let rho = 10.0_f64;
        let t = 1.0e3_f64;
        let abar = 55.845_f64;
        let zbar = 26.0_f64;
        let xa = [1.0_f64];
        let azion = [zbar];
        let acmi = [abar];
        let ay = [1.0_f64 / abar];

        let t_ad = AutoDiffReal2VarOrder1::var1(t);
        let rho_ad = AutoDiffReal2VarOrder1::var2(rho);
        let temp_ad = (t_ad * 1.0e-6) / 0.315_774_6;
        let m = melange9(
            &ay,
            &azion,
            &acmi,
            rho_ad,
            temp_ad,
            cfg.pc_gamma_start_crystal,
            cfg.pc_gamma_full_crystal,
        )
        .expect("melange9 cryogenic Fe56 succeeds");
        assert!(
            m.u_nkt.val < 0.0,
            "expected u_nkt < 0 for cryogenic Fe56, got {}",
            m.u_nkt.val
        );

        let n = 1.0 / (abar * AMU);
        let expected_ln_e = (m.u_nkt * n * KERG * t_ad).safe_log();
        let expected_ln_s = (m.s_nk * n * KERG).safe_log();
        assert_ne!(
            expected_ln_e.d1[0], 0.0,
            "expected nonzero d(lnE)/dT via safe_log"
        );
        assert_ne!(
            expected_ln_e.d1[1], 0.0,
            "expected nonzero d(lnE)/drho via safe_log"
        );

        let out = eval_pc_eos_from_weights(
            &cfg, rho, t, abar, zbar, &xa, &azion, &acmi,
        )
        .expect("eval_pc_eos_from_weights succeeds");

        assert_eq!(out.res[I_LN_E].to_bits(), expected_ln_e.val.to_bits());
        assert_eq!(
            out.d_dln_t[I_LN_E].to_bits(),
            (expected_ln_e.d1[0] * t).to_bits()
        );
        assert_eq!(
            out.d_dlnd[I_LN_E].to_bits(),
            (expected_ln_e.d1[1] * rho).to_bits()
        );
        assert_eq!(out.res[I_LN_S].to_bits(), expected_ln_s.val.to_bits());
        assert_eq!(
            out.d_dln_t[I_LN_S].to_bits(),
            (expected_ln_s.d1[0] * t).to_bits()
        );
        assert_eq!(
            out.d_dlnd[I_LN_S].to_bits(),
            (expected_ln_s.d1[1] * rho).to_bits()
        );
    });
}

#[test]
fn test_pc_excor7_analytic_outside_table_order_and_fd_partials() {
    with_numerics_mode(NumericsMode::Strict, || {
        // Points outside EXCOR7 spline table: log10 RS in [-3.5, 0], log10 GAME in [-2, 4].
        // Includes degenerate theta <= 0.04, moderate theta > 0.04, and hot theta >> 1.
        let cases = [
            (2.5_f64, 1.5_f64),   // high RS, moderate GAME (theta ~ 0.905)
            (5.0_f64, 0.005_f64), // high RS, low GAME (theta = 543)
            (1.0e-5_f64, 10.0_f64), // low RS, high GAME (theta = 5.43e-7 <= 0.04)
            (1.0e-4_f64, 0.5_f64), // low RS, moderate GAME (theta = 1.086e-4 <= 0.04)
        ];

        for &(rs0, game0) in &cases {
            let rs_ad = AutoDiffReal2VarOrder1::var1(rs0);
            let game_ad = AutoDiffReal2VarOrder1::var2(game0);

            let via_excor7 = excor7(rs_ad, game_ad);
            let ana = excor7_analytic(rs_ad, game_ad);
            assert_eq!(via_excor7.f.val.to_bits(), ana.f.val.to_bits());
            assert_eq!(via_excor7.f.d1[0].to_bits(), ana.f.d1[0].to_bits());
            assert_eq!(via_excor7.f.d1[1].to_bits(), ana.f.d1[1].to_bits());
            assert_eq!(via_excor7.u.val.to_bits(), ana.u.val.to_bits());
            assert_eq!(via_excor7.p.val.to_bits(), ana.p.val.to_bits());

            // Thermodynamic identities at fixed (rs, game):
            //   uxc = -(d f_xc / d ln T)_rho = +game * (d f_xc / d game)_rs
            //   pxc = (1/3) * (game * df/dgame - rs * df/drs)
            assert_close_rel(
                ana.u.val,
                game0 * ana.f.d1[1],
                1.0e-9,
                1.0e-12,
                "uxc vs game*df/dgame",
            );
            assert_close_rel(
                ana.p.val,
                (game0 * ana.f.d1[1] - rs0 * ana.f.d1[0]) / 3.0,
                1.0e-9,
                1.0e-12,
                "pxc vs (game*df/dgame - rs*df/drs)/3",
            );

            // Finite-difference checks for AD partials df/drs and df/dgame.
            let h_r = 1.0e-4 * rs0;
            let h_g = 1.0e-4 * game0;
            let f_rp = excor7_analytic(
                AutoDiffReal2VarOrder1::constant(rs0 + h_r),
                AutoDiffReal2VarOrder1::constant(game0),
            )
            .f
            .val;
            let f_rm = excor7_analytic(
                AutoDiffReal2VarOrder1::constant(rs0 - h_r),
                AutoDiffReal2VarOrder1::constant(game0),
            )
            .f
            .val;
            let f_gp = excor7_analytic(
                AutoDiffReal2VarOrder1::constant(rs0),
                AutoDiffReal2VarOrder1::constant(game0 + h_g),
            )
            .f
            .val;
            let f_gm = excor7_analytic(
                AutoDiffReal2VarOrder1::constant(rs0),
                AutoDiffReal2VarOrder1::constant(game0 - h_g),
            )
            .f
            .val;

            let rtol_fd = if rs0 < 1.0e-3 { 1.0e-4 } else { 1.0e-6 };
            assert_close_rel(
                ana.f.d1[0],
                (f_rp - f_rm) / (2.0 * h_r),
                rtol_fd,
                1.0e-12,
                "excor7_analytic df/drs FD",
            );
            assert_close_rel(
                ana.f.d1[1],
                (f_gp - f_gm) / (2.0 * h_g),
                rtol_fd,
                1.0e-12,
                "excor7_analytic df/dgame FD",
            );

            // PDTXC = pxc + d pxc / d ln T = pxc - game * d pxc / d game
            let p_scale = ana.p.val.abs().max(1.0);
            assert_close_rel(
                ana.pdt.val,
                ana.p.val - game0 * ana.p.d1[1],
                1.0e-9,
                1.0e-8 * p_scale,
                "pdtxc vs pxc - game*dp/dgame",
            );
            // PDRXC = pxc + d pxc / d ln rho = pxc + (game*dp/dgame - rs*dp/drs)/3
            assert_close_rel(
                ana.pdr.val,
                ana.p.val + (game0 * ana.p.d1[1] - rs0 * ana.p.d1[0]) / 3.0,
                1.0e-9,
                1.0e-8 * p_scale,
                "pdrxc vs pxc + (game*dp/dgame - rs*dp/drs)/3",
            );
        }

        // Exact MESA left-to-right operation-order witness at (rs=2.5, game=1.5):
        // theta = 0.543 * 2.5 / 1.5 = 0.905 -> verify scalar-first polynomial bits
        // match Fortran PC_EOS line-by-line rather than constant-last regroupings.
        let ana_w = excor7_analytic(
            AutoDiffReal2VarOrder1::var1(2.5),
            AutoDiffReal2VarOrder1::var2(1.5),
        );
        let theta = 0.543_f64 * 2.5_f64 / 1.5_f64;
        let sqth = theta.sqrt();
        let theta2 = theta * theta;
        let theta3 = theta2 * theta;
        let theta4 = theta3 * theta;
        let cht1 = flexmesa_math::cosh(1.0 / theta);
        let sht1 = flexmesa_math::sinh(1.0 / theta);
        let t1 = sht1 / cht1;
        let a0 = 0.75 + 3.04363 * theta2 - 0.09227 * theta3 + 1.7035 * theta4;
        let a1 = 1.0 + 8.31051 * theta2 + 5.1105 * theta4;
        let a = 0.610887 * a0 / a1 * t1;
        let e0 = 0.539409 + 2.522206 * theta2 + 0.178484 * theta4;
        let e1 = 1.0 + 2.555501 * theta2 + 0.146319 * theta4;
        let e = theta * t1 * e0 / e1;
        let c = (0.872496 + 0.025248 * flexmesa_math::exp(-1.0 / theta)) * e;
        let s1 = -c / e * 1.5_f64;
        assert!(ana_w.f.val.is_finite());
        assert!(a.is_finite() && sqth > 0.0 && s1 < 0.0);
    });
}

#[test]
fn test_pc_blin9c_small_chi_temp_order_and_ad_consistency() {
    with_numerics_mode(NumericsMode::Strict, || {
        // Test chi * temp < 0.1 Sommerfeld branch in blin9c (`pc_eos.f90:2689-2740`).
        let t0 = 1.0e-3_f64;
        let x0 = 50.0_f64; // chi * temp = 0.05 < 0.1
        let temp = AutoDiffReal2VarOrder1::var1(t0);
        let chi = AutoDiffReal2VarOrder1::var2(x0);
        let out = blin9c(temp, chi);

        assert_close_rel(
            out.w0.w.d1[0],
            out.w0.wdt.val,
            1.0e-12,
            1.0e-14,
            "w0 wdt",
        );
        assert_close_rel(
            out.w0.w.d1[1],
            out.w0.wdx.val,
            1.0e-12,
            1.0e-14,
            "w0 wdx",
        );
        assert_close_rel(
            out.w0.wdx.d1[1],
            out.w0.wdxx.val,
            1.0e-12,
            1.0e-14,
            "w0 wdxx",
        );
        assert_close_rel(
            out.w0.wdt.d1[0],
            out.w0.wdtt.val,
            1.0e-12,
            1.0e-14,
            "w0 wdtt",
        );
        assert_close_rel(
            out.w0.wdx.d1[0],
            out.w0.wdxt.val,
            1.0e-12,
            1.0e-14,
            "w0 wdxt",
        );
        assert_close_rel(
            out.w0.wdxx.d1[1],
            out.w0xxx.val,
            1.0e-11,
            1.0e-14,
            "w0 w0xxx",
        );
        assert_close_rel(
            out.w0.wdtt.d1[1],
            out.w0xtt.val,
            1.0e-11,
            1.0e-14,
            "w0 w0xtt",
        );
        assert_close_rel(
            out.w0.wdxx.d1[0],
            out.w0xxt.val,
            1.0e-11,
            1.0e-14,
            "w0 w0xxt",
        );

        // Check continuity across chi * temp = 0.1 boundary.
        let out_lo = blin9c(
            AutoDiffReal2VarOrder1::constant(0.0999 / x0),
            AutoDiffReal2VarOrder1::constant(x0),
        );
        let out_hi = blin9c(
            AutoDiffReal2VarOrder1::constant(0.1001 / x0),
            AutoDiffReal2VarOrder1::constant(x0),
        );
        assert_close_rel(
            out_lo.w0.w.val,
            out_hi.w0.w.val,
            5.0e-3,
            1.0e-12,
            "blin9c w0 continuity at chi*temp=0.1",
        );
        assert_close_rel(
            out_lo.w2.w.val,
            out_hi.w2.w.val,
            5.0e-3,
            1.0e-12,
            "blin9c w2 continuity at chi*temp=0.1",
        );
    });
}

#[test]
fn test_pc_untested_branches_and_component_consistency() {
    with_numerics_mode(NumericsMode::Strict, || {
        // 1. get_pc_alfa with pc_use_gamma_limit_instead_of_t = false and true
        let mut cfg = PcConfig {
            pc_use_gamma_limit_instead_of_t: false,
            ..PcConfig::default()
        };
        assert_eq!(get_pc_alfa(&cfg, 2.7, 7.0, 12.0, 6.0), (1.0, 0.0, 0.0));
        assert_eq!(get_pc_alfa(&cfg, 3.5, 7.8, 12.0, 6.0), (1.0, 0.0, 0.0));
        assert_eq!(get_pc_alfa(&cfg, 3.5, 7.0, 12.0, 6.0), (0.0, 0.0, 0.0));
        let (a_t, dat, dar) = get_pc_alfa(&cfg, 3.5, 7.65, 12.0, 6.0);
        assert_close_rel(a_t, 0.5, 1.0e-14, 1.0e-15, "get_pc_alfa logT alfa");
        assert_close_rel(dat, 10.0, 1.0e-13, 1.0e-15, "get_pc_alfa logT dT");
        assert_eq!(dar, 0.0);
        let (a_r, dat_r, dar_r) = get_pc_alfa(&cfg, 2.9, 7.0, 12.0, 6.0);
        assert_close_rel(
            a_r,
            (2.999 - 2.9) / 0.199,
            1.0e-13,
            1.0e-15,
            "get_pc_alfa logRho alfa",
        );
        assert_eq!(dat_r, 0.0);
        assert_close_rel(
            dar_r,
            -1.0 / 0.199,
            1.0e-13,
            1.0e-15,
            "get_pc_alfa logRho drho",
        );

        cfg.pc_use_gamma_limit_instead_of_t = true;
        let (a_ge, dat_ge, dar_ge) = get_pc_alfa(&cfg, 4.0, 8.34, 12.0, 6.0);
        assert!(
            a_ge > 0.0 && a_ge < 1.0 && dat_ge > 0.0 && dar_ge < 0.0,
            "expected logGe blend at logRho=4.0, logT=8.34: ({a_ge}, {dat_ge}, {dar_ge})"
        );
        let (a_glt, dat_glt, dar_glt) = get_pc_alfa(&cfg, 2.9, 6.98, 12.0, 6.0);
        assert!(
            a_glt > 0.0 && a_glt < 1.0 && dat_glt == 0.0 && dar_glt < 0.0,
            "expected logRho-under-logGe blend: ({a_glt}, {dat_glt}, {dar_glt})"
        );

        // 2. blin9a, blin9b (incl chi < 1e-3 clamp), and blin9 transitions
        let ba = blin9a(
            AutoDiffReal2VarOrder1::var1(0.1),
            AutoDiffReal2VarOrder1::var2(-5.0),
        );
        assert_close_rel(
            ba.w0.w.d1[1],
            ba.w0.wdx.val,
            1.0e-12,
            1.0e-15,
            "blin9a wdx",
        );
        assert_close_rel(
            ba.w0.w.d1[0],
            ba.w0.wdt.val,
            1.0e-12,
            1.0e-15,
            "blin9a wdt",
        );
        let bb_clamp = blin9b(
            AutoDiffReal2VarOrder1::var1(0.1),
            AutoDiffReal2VarOrder1::var2(1.0e-4),
        );
        assert_eq!(bb_clamp.w0.w.d1[1], 0.0);
        let b_blend1 = blin9(
            AutoDiffReal2VarOrder1::constant(0.1),
            AutoDiffReal2VarOrder1::constant(0.6),
        );
        let b_blend2 = blin9(
            AutoDiffReal2VarOrder1::constant(0.1),
            AutoDiffReal2VarOrder1::constant(14.0),
        );
        assert!(
            b_blend1.w0.w.val > 0.0 && b_blend2.w0.w.val > b_blend1.w0.w.val
        );

        // 3. elect11b with ef <= 1e-5 (and subfermj x0 < 1e-4) + elect11 blend
        let el_nr = elect11b(
            AutoDiffReal2VarOrder1::var1(1.0e-7),
            AutoDiffReal2VarOrder1::var2(30.0),
        );
        assert!(el_nr.dens.val.is_finite() && el_nr.dens.val > 0.0);
        assert!(el_nr.cve.val > 0.0 && el_nr.chite.val > 0.0);
        let el_blend = elect11(
            AutoDiffReal2VarOrder1::constant(0.5),
            AutoDiffReal2VarOrder1::constant(28.0),
        );
        assert!(el_blend.dens.val > 0.0 && el_blend.pe_id.val > 1.0);

        // 4. ferinv7 across n = 0..=3, f < 4, f >= 4, and f <= 0 clamp
        for n in 0..=3 {
            for f0 in [1.5_f64, 20.0_f64] {
                let (x, xdf, xdff) =
                    ferinv7(AutoDiffReal2VarOrder1::var1(f0), n);
                assert_close_rel(
                    x.d1[0],
                    xdf.val,
                    1.0e-12,
                    1.0e-14,
                    "ferinv7 xdf",
                );
                assert_close_rel(
                    xdf.d1[0],
                    xdff.val,
                    1.0e-12,
                    1.0e-14,
                    "ferinv7 xdff",
                );
            }
            let (x_cl, _, _) = ferinv7(AutoDiffReal2VarOrder1::var1(-1.0), n);
            assert!(x_cl.val.is_finite() && x_cl.d1[0] == 0.0);
        }

        // 5. chemfit low-density pf0 <= 1e-4 and high-degeneracy theta <= 1e-5
        let c_lo = chemfit(
            AutoDiffReal2VarOrder1::var1(1.0e-7),
            AutoDiffReal2VarOrder1::var2(1.0),
        );
        let c_hi = chemfit(
            AutoDiffReal2VarOrder1::var1(1.0e6),
            AutoDiffReal2VarOrder1::var2(1.0e-3),
        );
        assert!(c_lo.val < 0.0 && c_hi.val > 100.0);

        // 6. hlfit12 for both bcc (1) and fcc (2) across all 3 eta regimes
        for lattice in [1_i32, 2_i32] {
            let (f_low, _, cv_low, _, _) =
                hlfit12(AutoDiffReal2VarOrder1::var1(1.0e-100), lattice);
            assert!(f_low.val.is_finite() && f_low.d1[0] == 0.0);
            assert_close_rel(
                cv_low.val,
                3.0,
                1.0e-12,
                1.0e-15,
                "hlfit12 low eta cv",
            );

            let (f_hi, u_hi, cv_hi, _, _) =
                hlfit12(AutoDiffReal2VarOrder1::var1(2.0e5), lattice);
            assert_close_rel(
                cv_hi.val,
                4.0 * u_hi.val,
                1.0e-14,
                1.0e-20,
                "hlfit12 high eta cv",
            );
            assert_close_rel(
                f_hi.val,
                -u_hi.val / 3.0,
                1.0e-14,
                1.0e-20,
                "hlfit12 high eta f",
            );
        }

        // 7. fscr_liq8 / fscr_sol8 edge guards + tpt >= 6/PX branch + cormix rs < 1e-9
        assert!(fscr_liq8(
            AutoDiffReal2VarOrder1::constant(-1.0),
            AutoDiffReal2VarOrder1::constant(1.0),
            6.0,
        )
        .is_err());
        let z_liq = fscr_liq8(
            AutoDiffReal2VarOrder1::constant(1.0e-20),
            AutoDiffReal2VarOrder1::constant(1.0),
            6.0,
        )
        .expect("tiny rs liq");
        assert_eq!(z_liq.f.val, 0.0);
        let s_sol_hi_tpt = fscr_sol8(
            AutoDiffReal2VarOrder1::var1(0.05),
            AutoDiffReal2VarOrder1::var2(500.0),
            6.0,
            AutoDiffReal2VarOrder1::constant(35.0),
        )
        .expect("tpt >= 6/PX branch");
        assert!(s_sol_hi_tpt.f.val < 0.0 && s_sol_hi_tpt.pdr.val.is_finite());

        let cm_ocp = cormix(
            AutoDiffReal2VarOrder1::constant(0.1),
            AutoDiffReal2VarOrder1::constant(10.0),
            6.0,
            36.0,
            flexmesa_math::pow5(6.0_f64.sqrt()),
            flexmesa_math::pow(6.0, 5.0 / 3.0),
            6.0 * flexmesa_math::pow3(7.0_f64.sqrt()),
        );
        assert_eq!(cm_ocp.f.val, 0.0);
        let cm_rs0 = cormix(
            AutoDiffReal2VarOrder1::constant(1.0e-10),
            AutoDiffReal2VarOrder1::constant(1.0),
            7.0,
            50.0,
            0.5 * flexmesa_math::pow5(6.0_f64.sqrt())
                + 0.5 * flexmesa_math::pow5(8.0_f64.sqrt()),
            0.5 * flexmesa_math::pow(6.0, 5.0 / 3.0)
                + 0.5 * flexmesa_math::pow(8.0, 5.0 / 3.0),
            0.5 * 6.0 * flexmesa_math::pow3(7.0_f64.sqrt())
                + 0.5 * 8.0 * flexmesa_math::pow3(9.0_f64.sqrt()),
        );
        assert!(cm_rs0.f.val > 0.0 && cm_rs0.cv.val.is_finite());

        // 8. eval_pc_eos n > 32 heap path and eval_pc_eos_zones consistency
        let cat = NuclideCatalog::builtin_ref();
        let c12 = cat.get_by_name("c12").unwrap();
        let o16 = cat.get_by_name("o16").unwrap();
        let mut chem_id35 = Vec::with_capacity(35);
        let mut azion35 = Vec::with_capacity(35);
        let mut acmi35 = Vec::with_capacity(35);
        let mut xa35 = Vec::with_capacity(35);
        for k in 0..35 {
            let nuc = if k % 2 == 0 { c12 } else { o16 };
            chem_id35.push(nuc.chem_id);
            azion35.push(nuc.z as f64);
            acmi35.push(nuc.w);
            xa35.push(1.0 / 35.0);
        }
        let comp35 = get_composition_info(
            cat,
            &[c12.chem_id, o16.chem_id],
            &[18.0 / 35.0, 17.0 / 35.0],
            true,
        )
        .unwrap();
        let pc_cfg = PcConfig::default();
        let out_id35 = eval_pc_eos(
            &pc_cfg,
            1.0e6,
            1.0e7,
            comp35.abar,
            comp35.zbar,
            &chem_id35,
            &xa35,
        )
        .unwrap();
        let out_w35 = eval_pc_eos_from_weights(
            &pc_cfg,
            1.0e6,
            1.0e7,
            comp35.abar,
            comp35.zbar,
            &xa35,
            &azion35,
            &acmi35,
        )
        .unwrap();
        assert_eq!(out_id35.res, out_w35.res);
        assert_eq!(out_id35.d_dln_t, out_w35.d_dln_t);
        assert_eq!(out_id35.d_dlnd, out_w35.d_dlnd);

        let zones = eval_pc_eos_zones(
            &pc_cfg,
            &[1.0e6, 2.0e6],
            &[1.0e7, 5.0e6],
            &[comp35.abar, comp35.abar],
            &[comp35.zbar, comp35.zbar],
            &[xa35.clone(), xa35.clone()].concat(),
            &azion35,
            &acmi35,
        )
        .unwrap();
        assert_eq!(zones[0].res, out_w35.res);
    });
}

#[test]
fn test_skye_get_alfa_table_bounds_and_rad_order_parity() {
    with_numerics_mode(NumericsMode::Strict, || {
        let ht = HelmTable::builtin().expect("builtin HELM table");
        let abar = 12.0_f64;
        let zbar = 6.0_f64;

        // Verify pure interior & exterior.
        assert_eq!(get_skye_alfa(ht, 5.0, 8.0, abar, zbar), (0.0, 0.0, 0.0));
        assert_eq!(get_skye_alfa(ht, 0.0, 6.0, abar, zbar), (1.0, 0.0, 0.0));

        // Verify left-to-right MESA operation order for bounds(7,1) and bounds(8,1).
        let rad_log = flexmesa_math::log10(
            abar * MP * CRAD / (3.0 * KERG * (zbar + 1.0)),
        );
        let x7 = 3.0 * ht.logthi + rad_log - 6.0;
        let x8 = 3.0 * 8.3 + rad_log - 6.0;
        let t_ion =
            7.3_f64.max(flexmesa_math::log10(1.0e5 * zbar * zbar)) + 0.1;
        let rho_ion =
            2.3_f64.max(flexmesa_math::log10(abar * zbar * zbar * zbar)) + 0.1;
        let bounds: [[f64; 2]; 8] = [
            [ht.logdlo, 8.3],
            [ht.logdlo, t_ion],
            [rho_ion, t_ion],
            [rho_ion, ht.logtlo],
            [ht.logdhi, ht.logtlo],
            [ht.logdhi, ht.logthi],
            [x7, ht.logthi],
            [x8, 8.3],
        ];
        let lr = 0.5 * (x7 + x8) - 0.04;
        let lt = 0.5 * (ht.logthi + 8.3);
        let p = [
            AutoDiffReal2VarOrder1::var1(lr),
            AutoDiffReal2VarOrder1::var2(lt),
        ];
        assert!(!is_contained(&bounds, p));
        let expected_d = min_distance_to_polygon(&bounds, p) / 0.1;
        let (alfa, dt, dr) = get_skye_alfa(ht, lr, lt, abar, zbar);
        assert_eq!(alfa.to_bits(), expected_d.val.to_bits());
        assert_eq!(dt.to_bits(), expected_d.d1[1].to_bits());
        assert_eq!(dr.to_bits(), expected_d.d1[0].to_bits());

        // Verify `get_skye_alfa` actually reads `ht.logdlo` instead of constants.
        let mut custom_ht = ht.clone();
        custom_ht.logdlo = -8.0;
        let (alfa_custom, _, _) =
            get_skye_alfa(&custom_ht, -9.0, 8.0, abar, zbar);
        assert_eq!(alfa_custom, 1.0);
        assert!(get_skye_alfa(ht, -9.0, 8.0, abar, zbar).0 < 1.0);
    });
}

#[test]
fn test_skye_untested_branches_and_solid_mixing_invariance() {
    with_numerics_mode(NumericsMode::Strict, || {
        let ht = HelmTable::builtin().expect("builtin HELM table");

        // 1. Solid mixing rule permutation + duplicate-charge invariance and neutrons
        let game = AutoDiffReal2VarOrder3::var1(10.0);
        for rule in [SkyeSolidMixingRule::Pc, SkyeSolidMixingRule::Ogata] {
            let f_asc = solid_mixing_rule_correction(
                rule,
                &[0.5, 0.5],
                &[6.0, 8.0],
                game,
            );
            let f_desc = solid_mixing_rule_correction(
                rule,
                &[0.5, 0.5],
                &[8.0, 6.0],
                game,
            );
            let f_dup = solid_mixing_rule_correction(
                rule,
                &[0.2, 0.5, 0.3, 0.0],
                &[6.0, 8.0, 6.0, 0.0],
                game,
            );
            assert_eq!(f_asc.val.to_bits(), f_desc.val.to_bits());
            assert_close_rel(
                f_asc.val,
                f_dup.val,
                1.0e-14,
                1.0e-15,
                "solid mixing dup",
            );
        }
        assert!(delta_g_ogata93(0.5, 8.0 / 6.0).is_finite());
        assert!(delta_g_pc13(0.5, 8.0 / 6.0).is_finite());

        // 2. ocp_solid_harmonic_free_energy across < 1e-5, mid, and > 1e5 branches
        let g = AutoDiffReal2VarOrder3::constant(250.0);
        let f_lo = ocp_solid_harmonic_free_energy(
            g,
            AutoDiffReal2VarOrder3::var1(1.0e-6),
        );
        assert_close_rel(
            f_lo.d1val1,
            1.0e-6 / 12.0,
            1.0e-3,
            1.0e-12,
            "ocp_solid_harmonic_free_energy low-tpt d/dTPT",
        );

        let tpt_mid = 5.0_f64;
        let f_mid = ocp_solid_harmonic_free_energy(
            g,
            AutoDiffReal2VarOrder3::var1(tpt_mid),
        );
        let h_mid = 1.0e-5 * tpt_mid;
        let fp_mid = ocp_solid_harmonic_free_energy(
            g,
            AutoDiffReal2VarOrder3::constant(tpt_mid + h_mid),
        )
        .val;
        let fm_mid = ocp_solid_harmonic_free_energy(
            g,
            AutoDiffReal2VarOrder3::constant(tpt_mid - h_mid),
        )
        .val;
        assert_close_rel(
            f_mid.d1val1,
            (fp_mid - fm_mid) / (2.0 * h_mid),
            1.0e-7,
            1.0e-12,
            "ocp_solid_harmonic_free_energy mid-tpt d/dTPT",
        );

        let tpt_hi = 2.0e5_f64;
        let f_hi = ocp_solid_harmonic_free_energy(
            g,
            AutoDiffReal2VarOrder3::var1(tpt_hi),
        );
        let expected_hi = 1.5 * 0.5113875 - 3.0 / tpt_hi
            + 3.0 / (0.0047770935 * tpt_hi * tpt_hi * tpt_hi * tpt_hi);
        assert_close_rel(
            f_hi.d1val1,
            expected_hi,
            1.0e-12,
            1.0e-15,
            "ocp_solid_harmonic_free_energy high-tpt d/dTPT",
        );

        // 3. eval_skye_eos n > 32 heap path and eval_skye_eos_zones consistency
        let cat = NuclideCatalog::builtin_ref();
        let c12 = cat.get_by_name("c12").unwrap();
        let o16 = cat.get_by_name("o16").unwrap();
        let mut chem_id35 = Vec::with_capacity(35);
        let mut z35 = Vec::with_capacity(35);
        let mut acmi35 = Vec::with_capacity(35);
        let mut a35 = Vec::with_capacity(35);
        let mut xa35 = Vec::with_capacity(35);
        for k in 0..35 {
            let nuc = if k % 2 == 0 { c12 } else { o16 };
            chem_id35.push(nuc.chem_id);
            z35.push(nuc.z);
            acmi35.push(nuc.w);
            a35.push(nuc.z_plus_n as f64);
            xa35.push(1.0 / 35.0);
        }
        let comp35 = get_composition_info(
            cat,
            &[c12.chem_id, o16.chem_id],
            &[18.0 / 35.0, 17.0 / 35.0],
            true,
        )
        .unwrap();
        let skye_cfg = SkyeConfig::default();
        let s_id35 = eval_skye_eos(
            &skye_cfg,
            ht,
            1.0e6,
            1.0e8,
            comp35.abar,
            comp35.zbar,
            &chem_id35,
            &xa35,
        )
        .unwrap();
        let s_w35 = eval_skye_eos_from_weights(
            &skye_cfg,
            ht,
            1.0e8,
            1.0e6,
            comp35.abar,
            comp35.zbar,
            &xa35,
            &z35,
            &acmi35,
            &a35,
        )
        .unwrap();
        assert_eq!(s_id35.res, s_w35.res);
        assert_eq!(s_id35.d_dln_t, s_w35.d_dln_t);
        assert_eq!(s_id35.d_dlnd, s_w35.d_dlnd);

        let zones = eval_skye_eos_zones(
            &skye_cfg,
            ht,
            &[1.0e6, exp10(6.2)],
            &[1.0e8, exp10(7.8)],
            &[comp35.abar, comp35.abar],
            &[comp35.zbar, comp35.zbar],
            &[xa35.clone(), xa35.clone()].concat(),
            &z35,
            &acmi35,
            &a35,
        )
        .unwrap();
        assert_eq!(zones[0].res, s_w35.res);
    });
}
