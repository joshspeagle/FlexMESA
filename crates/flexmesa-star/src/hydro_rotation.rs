// SPDX-License-Identifier: LGPL-3.0-or-later
//! Roche-lobe rotational distortion helpers (`w_div_w_roche_omega`,
//! `w_div_w_roche_jrot`, `eval_i_rot`, and the Fabry, Marchant & Sana 2022
//! polynomial fits `rpsi_from_re_factor`, `re_from_rpsi_factor`, `A`, `B`, `C`,
//! and `sigmoid`) matching `star/private/hydro_rotation.f90:51-226, 874-920`.

use flexmesa_const::{ONE_SIXTH, ONE_THIRD, TWO_THIRDS};

/// Exponent in `log(1 - w^5.626)` (`log_term_power` in `hydro_rotation.f90:51`).
pub const LOG_TERM_POWER: f64 = 5.626;

/// Upstream default for `w_div_wcrit_max2` (`star/defaults/controls.defaults:4188`).
pub const DEFAULT_W_DIV_WCRIT_MAX2: f64 = 0.90;

/// Volume-equivalent radius ratio $r_\psi / r_e$ (`rpsi_from_re_factor` in
/// `star/private/hydro_rotation.f90:874-880`; Fabry et al. 2022, Eq. A.2).
#[must_use]
#[inline]
pub fn rpsi_from_re_factor(o2: f64, o4: f64, o6: f64) -> f64 {
    1.0 - ONE_SIXTH * o2 + 0.02025 * o4 - 0.03870 * o6
}

/// Equatorial radius ratio $r_e / r_\psi$ (`re_from_rpsi_factor` in
/// `star/private/hydro_rotation.f90:882-888`; Fabry et al. 2022, Eq. A.3).
#[must_use]
#[inline]
pub fn re_from_rpsi_factor(o2: f64, o4: f64, o6: f64) -> f64 {
    1.0 + ONE_SIXTH * o2 - 0.005124 * o4 + 0.06562 * o6
}

/// Roche surface-integral factor `A` (`star/private/hydro_rotation.f90:890-896`;
/// Fabry et al. 2022, Eq. A.6).
#[must_use]
#[inline]
pub fn roche_a(o4: f64, o6: f64, o_log_term: f64) -> f64 {
    1.0 + 0.3293 * o4 - 0.4926 * o6 - 0.5560 * o_log_term
}

/// Roche moment-of-inertia factor `B` (`star/private/hydro_rotation.f90:898-905`;
/// Fabry et al. 2022, Eq. A.7).
#[must_use]
#[inline]
pub fn roche_b(o2: f64, o4: f64, o6: f64, o_log_term: f64) -> f64 {
    1.0 + o2 / 5.0 + 0.4140 * o4 - 0.8650 * o6 - 0.8370 * o_log_term
}

/// Roche specific-angular-momentum factor `C`
/// (`star/private/hydro_rotation.f90:907-914`; Fabry et al. 2022, Eq. A.12).
#[must_use]
#[inline]
pub fn roche_c(o2: f64, o4: f64, o6: f64, o_log_term: f64) -> f64 {
    1.0 + (17.0 / 60.0) * o2 + 0.4010 * o4 - 0.8606 * o6 - 0.9305 * o_log_term
}

/// Smoothly maps $|x| \in [x_{\max 1}, \infty)$ onto $[x_{\max 1}, x_{\max 2})$
/// (`sigmoid` in `star/private/hydro_rotation.f90:916-920`).
#[must_use]
#[inline]
pub fn roche_sigmoid(x: f64, xmax1: f64, xmax2: f64) -> f64 {
    2.0 * (xmax2 - xmax1)
        / (1.0 + flexmesa_math::exp(-2.0 * (x.abs() - xmax1) / (xmax2 - xmax1)))
        - xmax2
        + 2.0 * xmax1
}

/// Computes `w_div_w_roche` for a known angular frequency `omega`
/// ($\mathrm{rad\,s^{-1}}$), volume-equivalent radius `rpsi` ($\mathrm{cm}$),
/// and enclosed mass `m_phi` ($\mathrm{g}$) matching `w_div_w_roche_omega` in
/// `star/private/hydro_rotation.f90:55-108`.
#[must_use]
pub fn w_div_w_roche_omega(
    rpsi: f64,
    m_phi: f64,
    omega: f64,
    cgrav: f64,
    max_w: f64,
    max_w2: f64,
    w_div_wc_flag: bool,
) -> f64 {
    if omega == 0.0 {
        return 0.0;
    }

    let mut dimless_rpsi = rpsi
        * flexmesa_math::pow(omega.abs(), TWO_THIRDS)
        / flexmesa_math::pow(cgrav * m_phi, ONE_THIRD);

    let mut wr;
    if w_div_wc_flag {
        wr = max_w;
        let rpsi_lim1 = flexmesa_math::pow(wr, TWO_THIRDS)
            * rpsi_from_re_factor(
                flexmesa_math::pow2(wr),
                flexmesa_math::pow4(wr),
                flexmesa_math::pow6(wr),
            );

        wr = max_w2;
        let rpsi_lim2 = flexmesa_math::pow(wr, TWO_THIRDS)
            * rpsi_from_re_factor(
                flexmesa_math::pow2(wr),
                flexmesa_math::pow4(wr),
                flexmesa_math::pow6(wr),
            );

        if dimless_rpsi.abs() > rpsi_lim1 {
            dimless_rpsi = roche_sigmoid(dimless_rpsi.abs(), rpsi_lim1, rpsi_lim2);
        }
    } else {
        wr = max_w;
        let new_dimless_rpsi = flexmesa_math::pow(wr, TWO_THIRDS)
            * rpsi_from_re_factor(
                flexmesa_math::pow2(wr),
                flexmesa_math::pow4(wr),
                flexmesa_math::pow6(wr),
            );
        if dimless_rpsi > new_dimless_rpsi {
            return wr;
        }
    }

    let mut wr_high = wr;
    let mut wr_low = 0.0_f64;
    while wr_high - wr_low > 1.0e-6 {
        wr = 0.5 * (wr_high + wr_low);
        let new_dimless_rpsi = flexmesa_math::pow(wr, TWO_THIRDS)
            * rpsi_from_re_factor(
                flexmesa_math::pow2(wr),
                flexmesa_math::pow4(wr),
                flexmesa_math::pow6(wr),
            );
        if dimless_rpsi > new_dimless_rpsi {
            wr_low = wr;
        } else {
            wr_high = wr;
        }
    }
    let mut w_roche = 0.5 * (wr_high + wr_low);
    if omega < 0.0 {
        w_roche = -w_roche;
    }
    w_roche
}

/// Computes `w_div_w_roche` for a known specific angular momentum `jrot`
/// ($\mathrm{cm^2\,s^{-1}}$), volume-equivalent radius `rpsi` ($\mathrm{cm}$),
/// and enclosed mass `m_phi` ($\mathrm{g}$) matching `w_div_w_roche_jrot` in
/// `star/private/hydro_rotation.f90:111-179`.
#[must_use]
pub fn w_div_w_roche_jrot(
    rpsi: f64,
    m_phi: f64,
    jrot: f64,
    cgrav: f64,
    max_w: f64,
    max_w2: f64,
    w_div_wc_flag: bool,
) -> f64 {
    if jrot == 0.0 {
        return 0.0;
    }

    let mut dimless_factor = jrot.abs() / (cgrav * m_phi * rpsi).sqrt();

    let mut wr;
    if w_div_wc_flag {
        wr = max_w;
        let w4 = flexmesa_math::pow4(wr);
        let w6 = flexmesa_math::pow6(wr);
        let w_log_term =
            flexmesa_math::log(1.0 - flexmesa_math::pow(wr, LOG_TERM_POWER));
        let jr_lim1 = TWO_THIRDS
            * wr
            * roche_c(flexmesa_math::pow2(wr), w4, w6, w_log_term)
            / roche_a(w4, w6, w_log_term);

        wr = max_w2;
        let w4 = flexmesa_math::pow4(wr);
        let w6 = flexmesa_math::pow6(wr);
        let w_log_term =
            flexmesa_math::log(1.0 - flexmesa_math::pow(wr, LOG_TERM_POWER));
        let jr_lim2 = TWO_THIRDS
            * wr
            * roche_c(flexmesa_math::pow2(wr), w4, w6, w_log_term)
            / roche_a(w4, w6, w_log_term);

        if dimless_factor.abs() > jr_lim1 {
            dimless_factor =
                roche_sigmoid(dimless_factor.abs(), jr_lim1, jr_lim2);
        }
    } else {
        wr = max_w;
        let w2 = flexmesa_math::pow2(wr);
        let w4 = flexmesa_math::pow4(wr);
        let w6 = flexmesa_math::pow6(wr);
        let w_log_term =
            flexmesa_math::log(1.0 - flexmesa_math::pow(wr, LOG_TERM_POWER));
        let new_dimless_factor =
            TWO_THIRDS * wr * roche_c(w2, w4, w6, w_log_term)
                / roche_a(w4, w6, w_log_term);
        if dimless_factor > new_dimless_factor {
            return wr;
        }
    }

    let mut wr_high = wr;
    let mut wr_low = 0.0_f64;
    while wr_high - wr_low > 1.0e-6 {
        wr = 0.5 * (wr_high + wr_low);
        let w2 = flexmesa_math::pow2(wr);
        let w4 = flexmesa_math::pow4(wr);
        let w6 = flexmesa_math::pow6(wr);
        let w_log_term =
            flexmesa_math::log(1.0 - flexmesa_math::pow(wr, LOG_TERM_POWER));
        let new_dimless_factor =
            TWO_THIRDS * wr * roche_c(w2, w4, w6, w_log_term)
                / roche_a(w4, w6, w_log_term);
        if dimless_factor > new_dimless_factor {
            wr_low = wr;
        } else {
            wr_high = wr;
        }
    }
    let mut w_roche = 0.5 * (wr_high + wr_low);
    if jrot < 0.0 {
        w_roche = -w_roche;
    }
    w_roche
}

/// Evaluates the shell specific moment of inertia $i_{\mathrm{rot}}$
/// ($\mathrm{cm^2}$) following Paxton et al. (2019, ApJS 243, 10), matching
/// `eval_i_rot` in `star/private/hydro_rotation.f90:181-226`.
#[must_use]
pub fn eval_i_rot(
    r00: f64,
    w_div_w_crit_roche: f64,
    simple_i_rot_flag: bool,
) -> f64 {
    if simple_i_rot_flag {
        TWO_THIRDS * r00 * r00
    } else {
        let w = w_div_w_crit_roche;
        let w2 = flexmesa_math::pow2(w);
        let w4 = flexmesa_math::pow4(w);
        let w6 = flexmesa_math::pow6(w);
        let lg_one_sub_w4 = flexmesa_math::log(1.0 - w4);
        let re = r00 * (1.0 + w2 / 6.0 - 0.0002507 * w4 + 0.06075 * w6);
        let b = 1.0 + w2 / 5.0 - 0.2735 * w4 - 0.4327 * w6
            - (3.0 / 2.0) * 0.5583 * lg_one_sub_w4;
        let a = 1.0 - 0.1076 * w4 - 0.2336 * w6 - 0.5583 * lg_one_sub_w4;
        TWO_THIRDS * flexmesa_math::pow2(re) * b / a
    }
}

#[cfg(test)]
mod tests {
    use flexmesa_const::{M_SUN, R_SUN, STANDARD_CGRAV, TWO_THIRDS};

    use super::{eval_i_rot, w_div_w_roche_jrot, w_div_w_roche_omega};

    // Expected values below were evaluated independently in Python directly
    // from the upstream formulas in `star/private/hydro_rotation.f90:55-226,
    // 874-920` using MESA `26.4.1` `const_def.f90` constants:
    //   w = 0.6 -> j_exact = (2/3)*w*C/A * sqrt(G*Msun*Rsun)
    //                      = 1.3493170707419899e18 cm^2/s
    //   w = 0.6 -> om_exact = w * rpsi_from_re_factor(w)^(3/2) * sqrt(G*Msun/Rsun^3)
    //                       = 3.4374235810367103e-4 rad/s
    //   eval_i_rot(Rsun, 0.6, false) = 3.9215030601732740e21 cm^2

    #[test]
    fn test_w_div_w_roche_zero_inputs_return_exact_zero() {
        assert_eq!(
            w_div_w_roche_jrot(R_SUN, M_SUN, 0.0, STANDARD_CGRAV, 0.89, 0.90, false),
            0.0
        );
        assert_eq!(
            w_div_w_roche_omega(R_SUN, M_SUN, 0.0, STANDARD_CGRAV, 0.89, 0.90, false),
            0.0
        );
    }

    #[test]
    fn test_w_div_w_roche_bisection_matches_independent_upstream_formulas() {
        let j_06 = 1.349_317_070_741_989_9e18;
        let w_j = w_div_w_roche_jrot(
            R_SUN,
            M_SUN,
            j_06,
            STANDARD_CGRAV,
            0.89,
            0.90,
            false,
        );
        assert!(
            (w_j - 0.6).abs() <= 5.0e-7,
            "w_div_w_roche_jrot expected 0.6 within bisection tol 5e-7, got {w_j:.16e}"
        );
        // Sign symmetry for negative jrot below cap (`hydro_rotation.f90:175-177`).
        let w_j_neg = w_div_w_roche_jrot(
            R_SUN,
            M_SUN,
            -j_06,
            STANDARD_CGRAV,
            0.89,
            0.90,
            false,
        );
        assert_eq!(w_j_neg, -w_j);

        let om_06 = 3.437_423_581_036_710_3e-4;
        let w_om = w_div_w_roche_omega(
            R_SUN,
            M_SUN,
            om_06,
            STANDARD_CGRAV,
            0.89,
            0.90,
            false,
        );
        assert!(
            (w_om - 0.6).abs() <= 5.0e-7,
            "w_div_w_roche_omega expected 0.6 within bisection tol 5e-7, got {w_om:.16e}"
        );
        let w_om_neg = w_div_w_roche_omega(
            R_SUN,
            M_SUN,
            -om_06,
            STANDARD_CGRAV,
            0.89,
            0.90,
            false,
        );
        assert_eq!(w_om_neg, -w_om);
    }

    #[test]
    fn test_w_div_w_roche_hard_cap_and_sigmoid_asymptote_match_upstream() {
        // Hard cap when `w_div_wc_flag = false` returns exact `max_w`.
        assert_eq!(
            w_div_w_roche_jrot(R_SUN, M_SUN, 1.0e22, STANDARD_CGRAV, 0.89, 0.90, false),
            0.89
        );
        assert_eq!(
            w_div_w_roche_omega(R_SUN, M_SUN, 1.0e-1, STANDARD_CGRAV, 0.89, 0.90, false),
            0.89
        );

        // Smooth sigmoid cap when `w_div_wc_flag = true` approaches `max_w2 = 0.90`.
        let w_sig_j = w_div_w_roche_jrot(
            R_SUN,
            M_SUN,
            1.0e22,
            STANDARD_CGRAV,
            0.89,
            0.90,
            true,
        );
        assert!(
            (w_sig_j - 0.90).abs() <= 1.0e-6,
            "sigmoid jrot asymptote expected 0.90, got {w_sig_j:.16e}"
        );
        let w_sig_om = w_div_w_roche_omega(
            R_SUN,
            M_SUN,
            1.0e-1,
            STANDARD_CGRAV,
            0.89,
            0.90,
            true,
        );
        assert!(
            (w_sig_om - 0.90).abs() <= 1.0e-6,
            "sigmoid omega asymptote expected 0.90, got {w_sig_om:.16e}"
        );
    }

    #[test]
    fn test_eval_i_rot_closed_forms_and_paxton_2019_polynomial() {
        assert_eq!(eval_i_rot(R_SUN, 0.6, true), TWO_THIRDS * R_SUN * R_SUN);
        assert_eq!(
            eval_i_rot(R_SUN, 0.0, false),
            TWO_THIRDS * (R_SUN * R_SUN)
        );

        let expected_06 = 3.921_503_060_173_274e21;
        let got_06 = eval_i_rot(R_SUN, 0.6, false);
        let rel = (got_06 - expected_06).abs() / expected_06;
        assert!(
            rel < 1.0e-14,
            "eval_i_rot(R_SUN, 0.6) mismatch: got {got_06:.16e}, expected {expected_06:.16e}"
        );
    }
}
