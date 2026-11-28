// SPDX-License-Identifier: GPL-3.0-or-later
//! Invariant unit tests for `#581` `powN` multiplication grouping in
//! `flexmesa-binary` (`mdot`, `edd_wind`, `disk`).

#![forbid(unsafe_code)]
#![allow(
    clippy::excessive_precision,
    clippy::similar_names,
    clippy::unreadable_literal
)]

use flexmesa_binary::{
    bh_initial_equatorial_mass, eval_arras_mdot_thin, eval_bh_spin,
    eval_l2_mass_loss_fraction, eval_ritter_mdot_thin, eval_rlobe,
};
use flexmesa_const::{KERG, MP, MSUN, ONE_THIRD, PI, RSUN, STANDARD_CGRAV};
use flexmesa_math::{exp, log10, pow, pow2, pow3, pow4, pow6};

#[test]
fn test_eval_ritter_mdot_thin_gamma_q_uses_pow2_grouping() {
    for (m_donor, m_accretor) in
        [(10.0 * MSUN, 0.56 * MSUN), (1.0 * MSUN, 9.0 * MSUN)]
    {
        let r_donor = 2.0 * RSUN;
        let rl_donor = eval_rlobe(m_donor, m_accretor, 6.0 * RSUN);
        let rho_surf = 1.0e-7;
        let p_surf = 1.0e5;
        let t_surf = 1.2e4;
        let mu_surf = 0.62;

        let actual = eval_ritter_mdot_thin(
            m_donor, m_accretor, r_donor, rl_donor, rho_surf, p_surf, t_surf,
            mu_surf, false,
        );

        let grav = STANDARD_CGRAV * m_donor / (r_donor * r_donor);
        let hp = p_surf / (grav * rho_surf);
        let v_th = (KERG * t_surf / (MP * mu_surf)).sqrt();
        let q = m_accretor / m_donor;
        let q1 = q.clamp(0.5, 10.0);
        let f1 = 1.23 + 0.5 * log10(q1);
        // `binary/private/binary_mdot.f90:704-706`: `rl3 = rl*rl*rl` and
        // `(2.0D0*pi/exp(0.5d0)) * v_th*v_th*v_th * rl3/(G*m) * rho * F1`,
        // left to right.
        let rl3 = rl_donor * rl_donor * rl_donor;
        let mdot0 = (2.0 * PI / exp(0.5)) * v_th * v_th * v_th * rl3
            / (STANDARD_CGRAV * m_donor)
            * rho_surf
            * f1;
        let q2 = q.clamp(0.04, 20.0);
        let log_q2 = log10(q2);
        let gamma_q_pow2 = if q2 < 1.0 {
            0.954 + 0.025 * log_q2 - 0.038 * pow2(log_q2)
        } else {
            0.954 + 0.039 * log_q2 + 0.114 * pow2(log_q2)
        };
        let gamma_q_l2r = if q2 < 1.0 {
            0.954 + 0.025 * log_q2 - 0.038 * log_q2 * log_q2
        } else {
            0.954 + 0.039 * log_q2 + 0.114 * log_q2 * log_q2
        };
        assert_ne!(
            gamma_q_pow2.to_bits(),
            gamma_q_l2r.to_bits(),
            "q={q} must distinguish pow2 vs left-to-right gamma_q"
        );
        let expected = -mdot0 * exp((r_donor - rl_donor) / (hp / gamma_q_pow2));
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn test_eval_arras_mdot_thin_uses_pow2_pow3_and_pow6_grouping() {
    let m_donor = 1.4 * MSUN;
    let m_accretor = 0.7 * MSUN;
    let separation = 10.0 * RSUN;
    let r_donor = 2.3 * RSUN;
    let rho_surf = 1.0e-7;
    let t_surf = 1.0e4;
    let mu_surf = 0.62;

    let actual = eval_arras_mdot_thin(
        m_donor, m_accretor, separation, r_donor, rho_surf, t_surf, mu_surf,
    );

    let q = m_accretor / m_donor;
    let v_th = (KERG * t_surf / (MP * mu_surf)).sqrt();
    let omega = (STANDARD_CGRAV * (m_donor + m_accretor)
        / (separation * separation * separation))
        .sqrt();
    let q13 = pow(q, ONE_THIRD);
    let asl = 4.0 + 4.16 / (-0.96 + q13 + 1.0 / q13);
    let v_om = v_th / omega;
    let area = 2.0 * PI * pow2(v_om) / (asl * (asl - 1.0)).sqrt();
    let mfac1 = 1.0 + q;
    let mfac2 = 1.0 + 5.0 * q + 13.0 * q * q;
    let eval_phi = |rv: f64| {
        let rfac = rv / separation;
        -(STANDARD_CGRAV * m_donor / rv)
            * (1.0 + mfac1 * pow3(rfac) / 3.0 + 4.0 * mfac2 * pow6(rfac) / 45.0)
    };
    let rv_l1 = eval_rlobe(m_donor, m_accretor, separation);
    let phi_l1 = eval_phi(rv_l1);
    let phi = eval_phi(r_donor);
    let rho_l1 =
        (rho_surf / exp(1.0).sqrt()) * exp(-(phi_l1 - phi) / (v_th * v_th));
    let expected = -rho_l1 * v_th * area;

    let rfac_witness = 0.23_f64;
    let rfac3 = rfac_witness * rfac_witness * rfac_witness;
    assert_ne!(
        (4.0 * mfac2 * pow6(rfac_witness) / 45.0).to_bits(),
        (4.0 * mfac2 * rfac3 * rfac3 / 45.0).to_bits()
    );
    assert_eq!(actual.to_bits(), expected.to_bits());
}

#[test]
fn test_bh_spin_and_initial_mass_use_pow2_grouping() {
    let a = 0.78_f64;
    let m_bh_init = 8.0 * MSUN;
    let actual_init = bh_initial_equatorial_mass(m_bh_init, a);
    let one_m_a2 = (1.0 - a * a).max(0.0);
    let z1 = 1.0
        + pow(one_m_a2, ONE_THIRD)
            * (pow(1.0 + a, ONE_THIRD) + pow(1.0 - a, ONE_THIRD));
    let z2_pow2 = (3.0 * pow2(a) + pow2(z1)).sqrt();
    let z2_l2r = (3.0 * a * a + z1 * z1).sqrt();
    assert_ne!(z2_pow2.to_bits(), z2_l2r.to_bits());
    let r_isco = 3.0 + z2_pow2
        - ((3.0 - z1) * (3.0 + z1 + 2.0 * z2_pow2)).max(0.0).sqrt();
    let expected_init = m_bh_init * (r_isco / 6.0).sqrt();
    assert_eq!(actual_init.to_bits(), expected_init.to_bits());

    let ratio = 0.4096_f64;
    let m_bh_zero_eq = ratio;
    let m_bh = 1.0_f64;
    let actual_spin = eval_bh_spin(m_bh, m_bh_zero_eq);
    let expected_spin = ((2.0 / 3.0_f64).sqrt()
        * ratio
        * (4.0 - (18.0 * pow2(ratio) - 2.0).max(0.0).sqrt()))
    .clamp(0.0, 1.0);
    let legacy_spin = ((2.0 / 3.0_f64).sqrt()
        * ratio
        * (4.0 - (18.0 * ratio * ratio - 2.0).max(0.0).sqrt()))
    .clamp(0.0, 1.0);
    assert_ne!(expected_spin.to_bits(), legacy_spin.to_bits());
    assert_eq!(actual_spin.to_bits(), expected_spin.to_bits());
}

#[test]
fn test_disk_l2_fraction_and_lagrange_polynomial_pow2_pow4_bounds() {
    let fl2 =
        eval_l2_mass_loss_fraction(20.0, 10.0, 0.001, 2.0, 0.1, 1.3 / 2.4);
    assert!(fl2.is_finite() && (0.0..=1.0).contains(&fl2));

    let log_q = log10(0.35);
    assert_ne!(
        (-0.0355 * pow2(log_q)).to_bits(),
        (-0.0355 * log_q * log_q).to_bits()
    );
    let xl1 = -0.0355 * pow2(log_q) + 0.251 * log_q.abs() + 0.500;
    let one_m_xl1 = 1.0 - xl1;
    let mu = 0.35 / 1.35;
    let rd_over_a_pow4 = pow4(one_m_xl1) / mu;
    let rd_over_a_l2r =
        (((one_m_xl1 * one_m_xl1) * one_m_xl1) * one_m_xl1) / mu;
    let rd_over_a_p2p2 = pow2(pow2(one_m_xl1)) / mu;
    assert_eq!(rd_over_a_pow4.to_bits(), rd_over_a_l2r.to_bits());
    assert_ne!(rd_over_a_pow4.to_bits(), rd_over_a_p2p2.to_bits());
}
