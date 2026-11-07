// SPDX-License-Identifier: LGPL-3.0-or-later
//! Coulomb chemical-potential corrections (`DCW1973`, `SLR1982`, `PCR2009`),
//! electron screening shift (`Throwe1983`, `Itoh2002`), and on-the-fly Fermi-Dirac
//! phase-space weak rates (`eval_ecapture`, `eval_psi`, `coulomb.f90`).

use flexmesa_const::{
    CLIGHT, FINE, KERG, KEV, LN2, ME, MEV_TO_ERGS, ONE_THIRD, PI, PI2,
};
use flexmesa_eos::dfermi;

use crate::loader::RatesCatalog;
use crate::types::{
    CoulombMuiOption, CoulombVsOption, RatesError, WeakRateEval,
};

/// Evaluates the dimensionless ion excess Coulomb chemical potential
/// $\mu_{\mathrm{I}} / (k_{\mathrm{B}} T)$ (`do_mui_coulomb` in `rates/private/coulomb.f90`).
#[must_use]
pub fn eval_mui_coulomb(
    which_mui: CoulombMuiOption,
    temp: f64,
    den: f64,
    zbar: f64,
    abar: f64,
    z: f64,
) -> f64 {
    if which_mui == CoulombMuiOption::None || z <= 0.0 {
        return 0.0;
    }
    let ye = zbar / abar;
    let game = 2.275e5 * (den * ye).powf(ONE_THIRD) / temp;
    let rs = 1.388 * (den * ye).powf(-ONE_THIRD);

    match which_mui {
        CoulombMuiOption::None => 0.0,
        CoulombMuiOption::Dcw1973 => {
            let gam = z.powf(2.0 / 3.0) * zbar * game;
            let zr = z / zbar;
            -zr * (gam
                * (0.9 + 0.2843 * zr.powf(-ONE_THIRD)
                    - 0.054 * zr.powf(-2.0 / 3.0))
                - (9.0 / 16.0)
                + 0.460 * zr.powf(-ONE_THIRD))
        }
        CoulombMuiOption::Slr1982 => {
            let gam = z.powf(5.0 / 3.0) * game;
            -0.898004 * gam + 4.0 * 0.96786 * gam.powf(0.25)
                - 4.0 * 0.220703 * gam.powf(-0.25)
                - 0.86097 * gam.ln()
                - 2.52692
        }
        CoulombMuiOption::Pcr2009 => {
            let gam = z.powf(5.0 / 3.0) * game;
            let a1 = -0.907347_f64;
            let a2 = 0.62849_f64;
            let a3 = -3.0_f64.sqrt() / 2.0 - a1 / a2.sqrt();
            let fii = a1
                * ((gam * (a2 + gam)).sqrt()
                    - a2 * ((gam / a2).sqrt() + (1.0 + gam / a2).sqrt()).ln())
                + 2.0 * a3 * (gam.sqrt() - gam.sqrt().atan());
            let a = 1.11 * z.powf(0.475);
            let b = 0.2 + 0.078 * z.ln().powi(2);
            let nu = 1.16 + 0.08 * z.ln();
            let cdh = (z / 3.0_f64.sqrt())
                * ((1.0 + z).powf(1.5) - 1.0 - z.powf(1.5));
            let ctf = 0.2513
                * z.powf(7.0 / 3.0)
                * (1.0 - z.powf(-ONE_THIRD) + 0.2 * z.powf(-0.5));
            let g1 = 1.0
                + (0.78 / (21.0 + game * (z / rs).powi(3))) * (game / z).sqrt();
            let g2 = 1.0
                + ((z - 1.0) / 9.0)
                    * (1.0 + 1.0 / (0.001 * z * z + 2.0 * game))
                    * rs.powi(3)
                    / (1.0 + 6.0 * rs * rs);
            let x = 0.014005 / rs;
            let h1 = 1.0
                / (1.0
                    + (x / (1.0 + x * x).sqrt()).powi(6) * z.powf(-ONE_THIRD));
            let h2 = 1.0 / (1.0 + x * x).sqrt();
            let fie = -game
                * (cdh * game.sqrt() + ctf * a * game.powf(nu) * g1 * h1)
                / (1.0 + (b * game.sqrt() + a * g2 * game.powf(nu) / rs) * h2);
            fii + fie
        }
    }
}

/// Evaluates the dimensionless electron screening potential shift $V_s$
/// (`do_vs_coulomb` in `rates/private/coulomb.f90`).
#[must_use]
pub fn eval_vs_coulomb(
    which_vs: CoulombVsOption,
    _temp: f64,
    den: f64,
    zbar: f64,
    abar: f64,
    z: f64,
) -> f64 {
    if which_vs == CoulombVsOption::None || z <= 0.0 {
        return 0.0;
    }
    let ye = zbar / abar;
    let rs = 1.388 * (den * ye).powf(-ONE_THIRD);
    match which_vs {
        CoulombVsOption::None => 0.0,
        CoulombVsOption::Throwe1983 => 2.0 * z * FINE * (FINE / PI).sqrt(),
        CoulombVsOption::Itoh2002 => {
            let s = 0.5 * (rs.max(1.0e-5).log10() + 3.0);
            let coeffs = [
                0.0450861,
                0.00113078,
                0.00312104,
                0.000864302,
                0.0157214,
                0.0816962,
                0.0784921,
                -0.0680863,
                -0.0979967,
                0.0204907,
                0.0366713,
            ];
            let fj: f64 = coeffs
                .iter()
                .enumerate()
                .map(|(i, &c)| c * s.powi(i as i32))
                .sum();
            1.46e-2 * z * fj
        }
    }
}

#[inline]
fn fd_complete(k: usize, y: f64) -> f64 {
    dfermi(k as f64, y, 0.0).0
}

/// Evaluates on-the-fly electron-capture and $\beta^-$-decay rates via exact
/// relativistic Fermi-Dirac phase-space integrals $I_{\mathrm{ec}}, J_{\mathrm{ec}},
/// I_{\mathrm{ee}}, J_{\mathrm{ee}}$ (`eval_ecapture.f90` and `eval_psi.f90`).
///
/// # Errors
/// Returns [`RatesError::UnknownWeakPair`] if `(lhs_name, rhs_name)` is not in
/// the special transitions catalog.
#[allow(clippy::too_many_arguments)]
pub fn eval_ecapture_pair(
    lhs_name: &str,
    rhs_name: &str,
    t9: f64,
    ye_rho: f64,
    eta_k: f64,
    which_mui: CoulombMuiOption,
    which_vs: CoulombVsOption,
    zbar: f64,
    abar: f64,
) -> Result<WeakRateEval, RatesError> {
    let cat = RatesCatalog::builtin();
    let (states_i, states_f, trans, z_lhs, z_rhs, q_tot) =
        cat.special_ecapture_data(lhs_name, rhs_name)?;

    let kbt = 1.0e3 * KEV * t9; // MeV
    let mec2 = ME * CLIGHT * CLIGHT / MEV_TO_ERGS; // MeV
    let beta = mec2 / kbt;
    let eta = eta_k + beta;
    if eta < 2.0 * beta {
        return Ok(WeakRateEval {
            lambda: 0.0,
            dlambda_dln_t: 0.0,
            dlambda_dln_rho: 0.0,
            q: q_tot,
            dq_dln_t: 0.0,
            dq_dln_rho: 0.0,
            qneu: 0.0,
            dqneu_dln_t: 0.0,
            dqneu_dln_rho: 0.0,
        });
    }

    let temp = t9 * 1.0e9;
    let ye = zbar / abar;
    let den = ye_rho / ye;
    let dmu = eval_mui_coulomb(which_mui, temp, den, zbar, abar, z_lhs)
        - eval_mui_coulomb(which_mui, temp, den, zbar, abar, z_rhs);
    let dq = dmu * kbt;
    let f_mue = 1.0 - eval_vs_coulomb(which_vs, temp, den, zbar, abar, z_lhs);
    let eta_tilde = f_mue * eta;

    let zpart: f64 = states_i
        .iter()
        .map(|&(e, j)| (2.0 * j + 1.0) * (-e / kbt).exp())
        .sum();

    let is_ec = (z_lhs - z_rhs) > 0.0;
    let (qx, ggm) = if is_ec {
        (q_tot - mec2, (PI * FINE * z_lhs).exp())
    } else {
        (q_tot + mec2, (PI * FINE * z_rhs).exp())
    };

    let mut lam = 0.0;
    let mut eps_nu = 0.0;

    for &(si, sf, logft) in trans {
        let (ei, ji) = states_i[si - 1];
        let ef = states_f[sf - 1].0;
        let zeta = (qx + dq - ef + ei) / kbt;
        let (ie, je) = if is_ec {
            if zeta > -beta {
                (0.0, 0.0)
            } else {
                let y = zeta + eta_tilde;
                let (f2, f3, f4, f5) = (
                    fd_complete(2, y),
                    fd_complete(3, y),
                    fd_complete(4, y),
                    fd_complete(5, y),
                );
                let ie_v =
                    (f4 - 2.0 * zeta * f3 + zeta * zeta * f2) / beta.powi(5);
                let je_v =
                    (f5 - 2.0 * zeta * f4 + zeta * zeta * f3) / beta.powi(6);
                (ie_v, je_v)
            }
        } else if zeta < beta {
            (0.0, 0.0)
        } else {
            let y1 = zeta - eta_tilde;
            let y2 = beta - eta_tilde;
            let (f2_1, f3_1, f4_1, f5_1) = (
                fd_complete(2, y1),
                fd_complete(3, y1),
                fd_complete(4, y1),
                fd_complete(5, y1),
            );
            let (f0_2, f1_2, f2_2, f3_2, f4_2, f5_2) = (
                fd_complete(0, y2),
                fd_complete(1, y2),
                fd_complete(2, y2),
                fd_complete(3, y2),
                fd_complete(4, y2),
                fd_complete(5, y2),
            );
            let c3 = 2.0 * zeta - 4.0 * beta;
            let c2 = 6.0 * beta * beta - 6.0 * beta * zeta + zeta * zeta;
            let c1 = -2.0
                * beta
                * (2.0 * beta * beta - 3.0 * beta * zeta + zeta * zeta);
            let c0 = beta * beta * (beta - zeta).powi(2);
            let ie_v = ((f4_1 - 2.0 * zeta * f3_1 + zeta * zeta * f2_1)
                - (f4_2 + c3 * f3_2 + c2 * f2_2 + c1 * f1_2 + c0 * f0_2))
                / beta.powi(5);

            let d4 = 3.0 * zeta - 5.0 * beta;
            let d3 =
                10.0 * beta * beta - 12.0 * beta * zeta + 3.0 * zeta * zeta;
            let d2 = zeta.powi(3) - 9.0 * beta * zeta * zeta
                + 18.0 * beta * beta * zeta
                - 10.0 * beta.powi(3);
            let d1 = beta * (5.0 * beta - 2.0 * zeta) * (beta - zeta).powi(2);
            let d0 = -beta * beta * (beta - zeta).powi(3);
            let je_v = ((f5_1 - 2.0 * zeta * f4_1 + zeta * zeta * f3_1)
                - (f5_2
                    + d4 * f4_2
                    + d3 * f3_2
                    + d2 * f2_2
                    + d1 * f1_2
                    + d0 * f0_2))
                / beta.powi(6);
            (ie_v, je_v)
        };

        if ie > 0.0 && je > 0.0 {
            let prob_i = (2.0 * ji + 1.0) * (-ei / kbt).exp() / zpart;
            let w = ggm * LN2 * 10.0_f64.powf(-logft) * prob_i;
            lam += ie * w;
            eps_nu += mec2 * je * w;
        }
    }

    let qneu = if lam > 0.0 { eps_nu / lam } else { 0.0 };
    Ok(WeakRateEval {
        lambda: lam,
        dlambda_dln_t: 0.0,
        dlambda_dln_rho: 0.0,
        q: q_tot,
        dq_dln_t: 0.0,
        dq_dln_rho: 0.0,
        qneu,
        dqneu_dln_t: 0.0,
        dqneu_dln_rho: 0.0,
    })
}

/// Evaluates proton electron-capture (`rpen`, `spen`) and neutron positron-capture
/// (`rnep`, `snep`) rates and neutrino losses (`ecapnuc` in `rates/private/ratelib.f90`).
#[must_use]
pub fn eval_ecapnuc_rate(
    etakep: f64,
    temp: f64,
    rho: f64,
) -> (f64, f64, f64, f64) {
    const C2ME: f64 = ME * CLIGHT * CLIGHT;
    const K_DIV_C2ME: f64 = KERG / C2ME;
    const CMK5: f64 =
        K_DIV_C2ME * K_DIV_C2ME * K_DIV_C2ME * K_DIV_C2ME * K_DIV_C2ME;
    const CMK6: f64 = CMK5 * K_DIV_C2ME;
    const QN1: f64 = -2.071_644_6e-6;
    const QN2: f64 = 2.071_644_6e-6;
    const FT: f64 = 1_083.926_9;

    let t9 = temp * 1.0e-9;
    if t9 < 1.0e-3 || rho < 1.0e-9 || etakep < -50.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let eval_branch = |qn: f64, etael: f64, etaef: f64| -> (f64, f64) {
        let t5 = temp * temp * temp * temp * temp;
        let zetan = qn / KERG / temp;
        let eta = etaef - etael;
        let exeta = if eta <= 6.8e2 { eta.exp() } else { 0.0 };
        let etael2 = etael * etael;
        let etael3 = etael2 * etael;
        let etael4 = etael3 * etael;
        let etael5 = etael4 * etael;
        let zetan2 = zetan * zetan;
        let f0 = if eta <= 6.8e2 { exeta.ln_1p() } else { eta };

        let f1l = exeta;
        let f2l = 2.0 * f1l;
        let f3l = 6.0 * f1l;
        let f4l = 24.0 * f1l;
        let f5l = 120.0 * f1l;

        let (f1g, f2g, f3g, f4g, f5g) = if eta > 0.0 {
            let exmeta = (-eta).exp();
            let eta2 = eta * eta;
            let eta3 = eta2 * eta;
            let eta4 = eta3 * eta;
            (
                0.5 * eta2 + 2.0 - exmeta,
                eta3 / 3.0 + 4.0 * eta + 2.0 * exmeta,
                0.25 * eta4 + 0.5 * PI2 * eta2 + 12.0 - 6.0 * exmeta,
                0.2 * eta4 * eta
                    + 2.0 * PI2 / 3.0 * eta3
                    + 48.0 * eta
                    + 24.0 * exmeta,
                eta4 * eta2 / 6.0
                    + 5.0 / 6.0 * PI2 * eta4
                    + 7.0 / 6.0 * PI2 * eta2
                    + 240.0
                    - 120.0 * exmeta,
            )
        } else {
            (0.0, 0.0, 0.0, 0.0, 0.0)
        };

        let fac3 = 2.0 * zetan + 4.0 * etael;
        let fac2 = 6.0 * etael2 + 6.0 * etael * zetan + zetan2;
        let fac1 = 4.0 * etael3 + 6.0 * etael2 * zetan + 2.0 * etael * zetan2;
        let fac0 = etael4 + 2.0 * zetan * etael3 + etael2 * zetan2;

        let rie1 = f4l + fac3 * f3l + fac2 * f2l + fac1 * f1l + fac0 * f0;
        let rie2 = f4g + fac3 * f3g + fac2 * f2g + fac1 * f1g + fac0 * f0;

        let facv4 = 5.0 * etael + 3.0 * zetan;
        let facv3 = 10.0 * etael2 + 12.0 * etael * zetan + 3.0 * zetan2;
        let facv2 = 10.0 * etael3
            + 18.0 * etael2 * zetan
            + 9.0 * etael * zetan2
            + zetan2 * zetan;
        let facv1 = 5.0 * etael4
            + 12.0 * etael3 * zetan
            + 9.0 * etael2 * zetan2
            + 2.0 * etael * zetan2 * zetan;
        let facv0 = etael5
            + 3.0 * etael4 * zetan
            + 3.0 * etael3 * zetan2
            + etael2 * zetan2 * zetan;
        let rjv1 = f5l
            + facv4 * f4l
            + facv3 * f3l
            + facv2 * f2l
            + facv1 * f1l
            + facv0 * f0;
        let rjv2 = f5g
            + facv4 * f4g
            + facv3 * f3g
            + facv2 * f2g
            + facv1 * f1g
            + facv0 * f0;

        if eta > 0.0 {
            (
                LN2 * CMK5 * t5 * rie2 / FT,
                LN2 * CMK6 * t5 * temp * rjv2 / FT,
            )
        } else {
            (
                LN2 * CMK5 * t5 * rie1 / FT,
                LN2 * CMK6 * t5 * temp * rjv1 / FT,
            )
        }
    };

    let etaef = etakep + C2ME / KERG / temp;
    let (rpen, spen) = eval_branch(QN1, QN2 / KERG / temp, etaef);
    let (rnep, snep) = eval_branch(QN2, C2ME / KERG / temp, -etaef);
    (rpen, rnep, spen, snep)
}
