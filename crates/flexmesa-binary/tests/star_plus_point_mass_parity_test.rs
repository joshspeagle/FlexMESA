// SPDX-License-Identifier: LGPL-3.0-or-later
//! Regression and MESA `26.4.1` parity tests for `star_plus_point_mass` and
//! `star_plus_point_mass_explicit_mdot` allowlisted binary physics (`#446`).
//!
//! Verifies that optically thin RLOF rates (`eval_ritter_mdot_thin`,
//! `eval_ritter_mdot_eccentric`, `eval_arras_mdot_thin`) evaluate
//! `exp(exponent)` verbatim to `binary/private/binary_mdot.f90` without the
//! artificial `-100.0` sub-threshold cutoff or `+60.0`/`+80.0` upper caps,
//! and pins orbital/gravitational-wave/Roche-lobe invariants at MESA `26.4.1`
//! `star_plus_point_mass` model 100.

#![forbid(unsafe_code)]

use flexmesa_binary::{
    binary_pick_next_timestep, eval_arras_mdot_thin, eval_j_orb, eval_jdot_gr,
    eval_ritter_mdot_eccentric, eval_ritter_mdot_thin, eval_rl_relative_gap,
    eval_rlobe, period_from_separation, separation_from_period,
    BinaryTimestepControls, B_TLIM_ROCHE,
};
use flexmesa_const::{
    CLIGHT, KERG, MP, MSUN, ONE_THIRD, PI, RSUN, SECDAY, SECYER, STANDARD_CGRAV,
};
use flexmesa_math::{exp, log10, pow, with_numerics_mode, NumericsMode};

#[test]
fn eval_ritter_mdot_thin_evaluates_deep_sub_rlof_and_overflow_without_clamps() {
    with_numerics_mode(NumericsMode::Strict, || {
        // Representative ZAMS 5 Msun donor surface state near step ~45-50 of
        // `star_plus_point_mass` (`R ~ 2.97 Rsun`, `RL ~ 3.32 Rsun`, `H_R ~ 2.5e-3 RL`).
        let m_donor = 5.0 * MSUN;
        let m_accretor = 10.0 * MSUN;
        let r_donor = 2.968 * RSUN;
        let rho_surf = 8.0e-10_f64;
        let p_surf = 1.0e3_f64;
        let t_surf = 1.6e4_f64;
        let mu_surf = 0.62_f64;

        let grav = STANDARD_CGRAV * m_donor / (r_donor * r_donor);
        let hp = p_surf / (grav * rho_surf);
        let q = m_accretor / m_donor;
        let q1 = q.clamp(0.5, 10.0);
        let f1 = 1.23 + 0.5 * log10(q1);
        let q2 = q.clamp(0.04, 20.0);
        let log_q2 = log10(q2);
        let gamma_q = if q2 < 1.0 {
            0.954 + 0.025 * log_q2 - 0.038 * log_q2 * log_q2
        } else {
            0.954 + 0.039 * log_q2 + 0.114 * log_q2 * log_q2
        };
        let hr = hp / gamma_q;
        let v_th = (KERG * t_surf / (MP * mu_surf)).sqrt();

        // Deep sub-RLOF where `ritter_exponent < -100` (previously clamped to 0.0,
        // forcing `lg_mtransfer_rate = -99.0` instead of ~`-85` to `-63` dex).
        for &target_exp in &[-160.0_f64, -115.0_f64, -100.5_f64] {
            let roche_r = r_donor - target_exp * hr;
            let rl3 = roche_r * roche_r * roche_r;
            let expected_mdot0 =
                (2.0 * PI / exp(0.5)) * (v_th * v_th * v_th) * rl3
                    / (STANDARD_CGRAV * m_donor)
                    * rho_surf
                    * f1;
            let expected = -expected_mdot0 * exp(target_exp);

            let actual_ritter = eval_ritter_mdot_thin(
                m_donor, m_accretor, r_donor, roche_r, rho_surf, p_surf,
                t_surf, mu_surf, false,
            );
            let actual_kolb = eval_ritter_mdot_thin(
                m_donor, m_accretor, r_donor, roche_r, rho_surf, p_surf,
                t_surf, mu_surf, true,
            );

            assert!(
                actual_ritter < 0.0 && actual_ritter.is_finite(),
                "sub-threshold Ritter exponent {target_exp} must stay negative finite: got {actual_ritter:e}"
            );
            let rel_err = (actual_ritter - expected).abs() / expected.abs();
            assert!(
                rel_err < 1.0e-12,
                "exponent {target_exp}: actual={actual_ritter:.15e}, expected={expected:.15e}"
            );
            assert_eq!(actual_ritter.to_bits(), actual_kolb.to_bits());
            let lg_msun_yr = log10(actual_ritter.abs() / MSUN * SECYER);
            assert!(
                lg_msun_yr > -99.0,
                "expected observable non-floored log10 rate at exponent {target_exp}, got {lg_msun_yr}"
            );
        }

        // Overflow with `ritter_exponent > 60.0`: Ritter uses `exp(exponent)` directly,
        // while Kolb caps the thin contribution at `-mdot0`.
        let overflow_exp = 75.0_f64;
        let rl_overflow = r_donor - overflow_exp * hr;
        let rl3_over = rl_overflow * rl_overflow * rl_overflow;
        let mdot0_over =
            (2.0 * PI / exp(0.5)) * (v_th * v_th * v_th) * rl3_over
                / (STANDARD_CGRAV * m_donor)
                * rho_surf
                * f1;
        let expected_ritter_over = -mdot0_over * exp(overflow_exp);

        let ritter_over = eval_ritter_mdot_thin(
            m_donor,
            m_accretor,
            r_donor,
            rl_overflow,
            rho_surf,
            p_surf,
            t_surf,
            mu_surf,
            false,
        );
        let kolb_over = eval_ritter_mdot_thin(
            m_donor,
            m_accretor,
            r_donor,
            rl_overflow,
            rho_surf,
            p_surf,
            t_surf,
            mu_surf,
            true,
        );
        assert!(
            (ritter_over - expected_ritter_over).abs()
                / expected_ritter_over.abs()
                < 1.0e-12
        );
        assert!((kolb_over - (-mdot0_over)).abs() / mdot0_over < 1.0e-14);
    });
}

#[test]
fn eval_ritter_mdot_eccentric_stays_nonzero_below_minus_100_scale_heights() {
    with_numerics_mode(NumericsMode::Strict, || {
        let m_donor = 5.0 * MSUN;
        let m_accretor = 10.0 * MSUN;
        let r_donor = 3.0 * RSUN;
        let rho_surf = 8.0e-10_f64;
        let p_surf = 1.0e3_f64;
        let t_surf = 1.6e4_f64;
        let mu_surf = 0.62_f64;

        let grav = STANDARD_CGRAV * m_donor / (r_donor * r_donor);
        let hp = p_surf / (grav * rho_surf);
        let q = m_accretor / m_donor;
        let log_q2 = log10(q.clamp(0.04, 20.0));
        let gamma_q = 0.954 + 0.039 * log_q2 + 0.114 * log_q2 * log_q2;
        let hr = hp / gamma_q;

        // Set periastron `RL_p = r_donor + 140 * H_R` with eccentricity `e = 0.2`.
        let e = 0.2_f64;
        let rl_peri = r_donor + 140.0 * hr;
        let rl_mean = rl_peri / (1.0 - e);
        let dm = eval_ritter_mdot_eccentric(
            m_donor, m_accretor, r_donor, rl_mean, e, rho_surf, p_surf, t_surf,
            mu_surf, false, 100,
        );
        assert!(
            dm < 0.0 && dm.is_finite(),
            "eccentric deep sub-RLOF got {dm:e}"
        );
    });
}

#[test]
fn eval_arras_mdot_thin_matches_mesa_fortran_without_exponent_cap() {
    with_numerics_mode(NumericsMode::Strict, || {
        let m_donor = 5.0 * MSUN;
        let m_accretor = 10.0 * MSUN;
        let separation = 10.35 * RSUN;
        let rho_surf = 8.0e-10_f64;
        let t_surf = 1.6e4_f64;
        let mu_surf = 0.62_f64;
        let v_th = (KERG * t_surf / (MP * mu_surf)).sqrt();
        let omega = (STANDARD_CGRAV * (m_donor + m_accretor)
            / (separation * separation * separation))
            .sqrt();
        let q = m_accretor / m_donor;
        let q13 = pow(q, ONE_THIRD);
        let asl = 4.0 + 4.16 / (-0.96 + q13 + 1.0 / q13);
        let area = 2.0 * PI * (v_th / omega) * (v_th / omega)
            / (asl * (asl - 1.0)).sqrt();

        let eval_phi = |rv: f64| {
            let rfac = rv / separation;
            let rfac3 = rfac * rfac * rfac;
            -(STANDARD_CGRAV * m_donor / rv)
                * (1.0
                    + (1.0 + q) * rfac3 / 3.0
                    + 4.0 * (1.0 + 5.0 * q + 13.0 * q * q) * rfac3 * rfac3
                        / 45.0)
        };
        let rv_l1 = eval_rlobe(m_donor, m_accretor, separation);
        let r_donor = rv_l1 * 1.15;
        let exp_val = -(eval_phi(rv_l1) - eval_phi(r_donor)) / (v_th * v_th);
        assert!(exp_val > 80.0, "test radius must exceed prior 80.0 clamp");

        let expected =
            -(rho_surf / exp(1.0).sqrt()) * exp(exp_val) * v_th * area;
        let actual = eval_arras_mdot_thin(
            m_donor, m_accretor, separation, r_donor, rho_surf, t_surf, mu_surf,
        );
        let rel_err = (actual - expected).abs() / expected.abs();
        assert!(
            rel_err < 1.0e-14,
            "Arras exponent {exp_val}: actual={actual:.15e}, expected={expected:.15e}"
        );
    });
}

#[test]
fn star_plus_point_mass_model_100_binary_primitives_match_mesa_26_4_1() {
    with_numerics_mode(NumericsMode::Strict, || {
        // MESA 26.4.1 `binary/star_plus_point_mass` `binary_history.data` model 100.
        let m1 = 4.999_996_518_448_367 * MSUN;
        let m2 = 1.000_000_348_155_163_4e1 * MSUN;
        let period_sec = 9.940_122_515_806_085e-1 * SECDAY;
        let sep_ref = 1.033_703_615_482_002_3e1 * RSUN;
        let rl1_ref = 3.315_997_526_825_342 * RSUN;
        let rl2_ref = 4.548_340_672_954_489 * RSUN;
        let j_orb_ref = 2.507_811_966_037_817_2e53_f64;
        let jdot_gr_ref = -4.343_937_364_961_423_4e35_f64;
        let gap1_ref = -3.934_319_250_341_292e-3_f64;

        let sep = separation_from_period(m1, m2, period_sec);
        let period_from_sep = period_from_separation(m1, m2, sep_ref);
        let period_rt = period_from_separation(m1, m2, sep);
        let rl1 = eval_rlobe(m1, m2, sep_ref);
        let rl2 = eval_rlobe(m2, m1, sep_ref);
        let j_orb = eval_j_orb(m1, m2, sep_ref, 0.0);
        let jdot_gr = eval_jdot_gr(m1, m2, sep_ref, j_orb);

        assert!((sep - sep_ref).abs() / sep_ref < 2.0e-14);
        assert!((period_from_sep - period_sec).abs() / period_sec < 1.0e-15);
        assert!((period_rt - period_sec).abs() / period_sec < 2.0e-14);
        assert!((rl1 - rl1_ref).abs() / rl1_ref < 1.0e-15);
        assert!((rl2 - rl2_ref).abs() / rl2_ref < 1.0e-15);
        assert!((j_orb - j_orb_ref).abs() / j_orb_ref < 1.0e-15);
        // `binary_evolve_step` evaluates `get_jdot` at `(a_old, j_old)` before
        // advancing `(j, a)` to model 100 (`4 * da/a ~ 2.7e-5`), while matching
        // the closed-form GW torque at `(sep_ref, j_orb)` to `< 1e-15`.
        let jdot_gr_exact = -32.0
            * (STANDARD_CGRAV * STANDARD_CGRAV * STANDARD_CGRAV)
            * m1
            * m2
            * (m1 + m2)
            / (5.0
                * CLIGHT
                * CLIGHT
                * CLIGHT
                * CLIGHT
                * CLIGHT
                * (sep_ref * sep_ref * sep_ref * sep_ref))
            * j_orb;
        assert!(
            (jdot_gr - jdot_gr_exact).abs() / jdot_gr_exact.abs() < 1.0e-15
        );
        assert!((jdot_gr - jdot_gr_ref).abs() / jdot_gr_ref.abs() < 3.0e-5);

        let r1 = rl1_ref * (1.0 + gap1_ref);
        let gap1 = eval_rl_relative_gap(r1, rl1, 0.0);
        assert!((gap1 - gap1_ref).abs() < 1.0e-14);

        // `binary_pick_next_timestep` with `star_plus_point_mass` controls
        // (`fr = 0.05`, `fr_limit = 1e-2`) limits step 100 via `b_Tlim_roche`.
        let ts_ctrl = BinaryTimestepControls {
            donor_time_step_yr: 7.406_842e4,
            max_timestep_sec: 1.0e99,
            time_delta_coeff: 1.0,
            dt_softening_factor: 0.5,
            rl_relative_gap: [gap1_ref, -1.0],
            rl_relative_gap_old: [-4.436e-3, -1.0],
            angular_momentum_j: j_orb,
            angular_momentum_j_old: j_orb_ref * 1.000_002,
            separation: sep,
            separation_old: sep,
            eccentricity: 0.0,
            eccentricity_old: 0.0,
            env_donor: 4.999_996_5,
            env_donor_old: 4.999_998,
            m: [m1, m2],
            m_old: [m1, m2],
            is_point_mass_companion: true,
            doing_first_model_of_run: false,
            ignore_hard_limits_this_step: false,
            fm: 0.01,
            fm_limit: 1.0e30,
            fr: 0.05,
            fr_limit: 1.0e-2,
            fr_dt_limit: 10.0,
            fj: 0.01,
            fa: 0.01,
            fe: 0.01,
            fe_limit: 1.0e-2,
            fdm: 0.005,
        };
        let (next_dt_sec, reason) = binary_pick_next_timestep(&ts_ctrl);
        assert_eq!(reason, B_TLIM_ROCHE);
        assert!(next_dt_sec > 0.0 && next_dt_sec.is_finite());
    });
}
