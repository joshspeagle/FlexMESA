// SPDX-License-Identifier: GPL-3.0-or-later
//! Coulomb chemical-potential corrections (`DCW1973`, `SLR1982`, `PCR2009`),
//! electron screening shift (`Throwe1983`, `Itoh2002`), and on-the-fly Fermi-Dirac
//! phase-space weak rates (`eval_ecapture`, `eval_psi`, `coulomb.f90`).

use flexmesa_autodiff::AutoDiffReal2VarOrder1;
use flexmesa_const::{
    CLIGHT, FINE, KERG, KEV, LN2, ME, MEV_TO_ERGS, ONE_THIRD, PI, PI2,
};
use flexmesa_eos::gauss_fermi::dfermi_order1;

use crate::loader::{EcaptureTables, RatesCatalog};
use crate::types::{
    CoulombMuiOption, CoulombVsOption, RatesError, WeakRateEval,
};

/// Matches MESA `auto_diff_real_2var_order1` division (`x * powm1(y)` in
/// `auto_diff/private/auto_diff_real_2var_order1_module.f90:1214-1245`) in
/// `NumericsMode::Strict`.
#[inline]
fn ad_div(lhs: f64, rhs: f64) -> f64 {
    if flexmesa_math::numerics_mode() == flexmesa_math::NumericsMode::Strict {
        lhs * flexmesa_math::powm1(rhs)
    } else {
        lhs / rhs
    }
}

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
    let game = ad_div(2.275e5 * flexmesa_math::pow(ye * den, ONE_THIRD), temp);
    let rs = 1.388 * flexmesa_math::pow(ye * den, -ONE_THIRD);

    match which_mui {
        CoulombMuiOption::None => 0.0,
        CoulombMuiOption::Dcw1973 => {
            // `rates/private/eval_coulomb.f90:95`:
            // `gamma = pow(Z,2d0/3d0) * cc% zbar * cc% gamma_e`, in that
            // order.
            let gam = flexmesa_math::pow(z, 2.0 / 3.0) * zbar * game;
            let zr = z / zbar;
            let zr_m13 = flexmesa_math::pow(zr, -ONE_THIRD);
            -zr * (gam * (0.9 + (0.2843 - 0.054 * zr_m13) * zr_m13)
                + (-9.0 / 16.0 + 0.460 * zr_m13))
        }
        CoulombMuiOption::Slr1982 => {
            // `eval_coulomb.f90:140-145`
            let gam = game * flexmesa_math::pow(z, 5.0 / 3.0);
            let gam14 = flexmesa_math::pow(gam, 0.25);
            -0.898004 * gam + 4.0 * 0.96786 * gam14
                - ad_div(4.0 * 0.220703, gam14)
                - 0.86097 * flexmesa_math::log(gam)
                - 2.52692
        }
        CoulombMuiOption::Pcr2009 => {
            // `eval_coulomb.f90:173-250`
            let gam = game * flexmesa_math::pow(z, 5.0 / 3.0);
            let a1 = -0.907347_f64;
            let a2 = 0.62849_f64;
            let a3 = -3.0_f64.sqrt() / 2.0 - a1 / a2.sqrt();
            let ga2 = ad_div(gam, a2);
            let fii = a1
                * ((gam * (a2 + gam)).sqrt()
                    - a2 * flexmesa_math::log(ga2.sqrt() + (1.0 + ga2).sqrt()))
                + 2.0 * a3 * (gam.sqrt() - flexmesa_math::atan(gam.sqrt()));
            let log_z = flexmesa_math::log(z);
            let a = 1.11 * flexmesa_math::pow(z, 0.475);
            let b = 0.2 + 0.078 * log_z * log_z;
            let nu = 1.16 + 0.08 * log_z;
            let cdh = (z / 3.0_f64.sqrt())
                * (flexmesa_math::pow(1.0 + z, 1.5)
                    - 1.0
                    - flexmesa_math::pow(z, 1.5));
            let ctf = 0.2513
                * flexmesa_math::pow(z, 7.0 / 3.0)
                * (1.0 - flexmesa_math::pow(z, -ONE_THIRD)
                    + 0.2 * flexmesa_math::pow(z, -0.5));
            let z_over_rs = ad_div(z, rs);
            let g1 = 1.0
                + ad_div(
                    0.78,
                    21.0 + game * flexmesa_math::pow(z_over_rs, 3.0),
                ) * ad_div(game, z).sqrt();
            let g2 = 1.0
                + ad_div(
                    ((z - 1.0) / 9.0)
                        * (1.0 + ad_div(1.0, 0.001 * z * z + 2.0 * game))
                        * flexmesa_math::pow(rs, 3.0),
                    1.0 + 6.0 * rs * rs,
                );
            let x = ad_div(0.014005, rs);
            let sq1x2 = (1.0 + x * x).sqrt();
            let h1 = ad_div(
                1.0,
                1.0 + flexmesa_math::pow(ad_div(x, sq1x2), 6.0)
                    * flexmesa_math::pow(z, -ONE_THIRD),
            );
            let h2 = ad_div(1.0, sq1x2);
            let fie = ad_div(
                -game
                    * (cdh * game.sqrt()
                        + ctf * a * flexmesa_math::pow(game, nu) * g1 * h1),
                1.0 + (b * game.sqrt()
                    + ad_div(a * g2 * flexmesa_math::pow(game, nu), rs))
                    * h2,
            );
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
    let rs = 1.388 * flexmesa_math::pow(ye * den, -ONE_THIRD);
    match which_vs {
        CoulombVsOption::None => 0.0,
        CoulombVsOption::Throwe1983 => 2.0 * z * FINE * (FINE / PI).sqrt(),
        CoulombVsOption::Itoh2002 => {
            let s = ad_div(flexmesa_math::log10(rs.max(1.0e-5)) + 3.0, 2.0);
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
            let fj = if s == 0.0 {
                coeffs[0]
            } else {
                let mut acc = 0.0_f64;
                for (i, &c) in coeffs.iter().enumerate() {
                    acc += c * flexmesa_math::pow(s, i as f64);
                }
                acc
            };
            1.46e-2 * z * fj
        }
    }
}

#[inline]
fn coulomb_rs_ad(den: f64, zbar: f64, abar: f64) -> AutoDiffReal2VarOrder1 {
    let den_ad = AutoDiffReal2VarOrder1::new(den, [0.0, den]);
    let ye = zbar / abar;
    1.388 * (ye * den_ad).powf(-ONE_THIRD)
}

#[inline]
fn coulomb_plasma_params_ad(
    temp: f64,
    den: f64,
    zbar: f64,
    abar: f64,
) -> (AutoDiffReal2VarOrder1, AutoDiffReal2VarOrder1) {
    let temp_ad = AutoDiffReal2VarOrder1::new(temp, [temp, 0.0]);
    let den_ad = AutoDiffReal2VarOrder1::new(den, [0.0, den]);
    let ye = zbar / abar;
    let ye_den = ye * den_ad;
    let game = (2.275e5 * ye_den.powf(ONE_THIRD)) / temp_ad;
    let rs = 1.388 * ye_den.powf(-ONE_THIRD);
    (game, rs)
}

#[inline]
fn eval_mui_coulomb_ad(
    which_mui: CoulombMuiOption,
    game: AutoDiffReal2VarOrder1,
    rs: AutoDiffReal2VarOrder1,
    zbar: f64,
    z: f64,
) -> AutoDiffReal2VarOrder1 {
    if which_mui == CoulombMuiOption::None || z <= 0.0 {
        return AutoDiffReal2VarOrder1::zero();
    }
    match which_mui {
        CoulombMuiOption::None => AutoDiffReal2VarOrder1::zero(),
        CoulombMuiOption::Dcw1973 => {
            let gam = flexmesa_math::pow(z, 2.0 / 3.0) * zbar * game;
            let zr = z / zbar;
            let zr_m13 = flexmesa_math::pow(zr, -ONE_THIRD);
            -zr * (gam * (0.9 + (0.2843 - 0.054 * zr_m13) * zr_m13)
                + (-9.0 / 16.0 + 0.460 * zr_m13))
        }
        CoulombMuiOption::Slr1982 => {
            let gam = game * flexmesa_math::pow(z, 5.0 / 3.0);
            let gam14 = gam.powf(0.25);
            -0.898004 * gam + 4.0 * 0.96786 * gam14
                - (4.0 * 0.220703) / gam14
                - 0.86097 * gam.log()
                - 2.52692
        }
        CoulombMuiOption::Pcr2009 => {
            let gam = game * flexmesa_math::pow(z, 5.0 / 3.0);
            let a1 = -0.907347_f64;
            let a2 = 0.62849_f64;
            let a3 = -3.0_f64.sqrt() / 2.0 - a1 / a2.sqrt();
            let ga2 = gam / a2;
            let fii = a1
                * ((gam * (a2 + gam)).sqrt()
                    - a2 * (ga2.sqrt() + (1.0 + ga2).sqrt()).log())
                + 2.0 * a3 * (gam.sqrt() - gam.sqrt().atan());
            let log_z = flexmesa_math::log(z);
            let a = 1.11 * flexmesa_math::pow(z, 0.475);
            let b = 0.2 + 0.078 * log_z * log_z;
            let nu = 1.16 + 0.08 * log_z;
            let cdh = (z / 3.0_f64.sqrt())
                * (flexmesa_math::pow(1.0 + z, 1.5)
                    - 1.0
                    - flexmesa_math::pow(z, 1.5));
            let ctf = 0.2513
                * flexmesa_math::pow(z, 7.0 / 3.0)
                * (1.0 - flexmesa_math::pow(z, -ONE_THIRD)
                    + 0.2 * flexmesa_math::pow(z, -0.5));
            let z_over_rs = z / rs;
            let g1 = 1.0
                + (0.78 / (21.0 + game * z_over_rs.powf(3.0)))
                    * (game / z).sqrt();
            let g2 = 1.0
                + (((z - 1.0) / 9.0)
                    * (1.0 + 1.0 / (0.001 * z * z + 2.0 * game))
                    * rs.powf(3.0))
                    / (1.0 + 6.0 * rs * rs);
            let x = 0.014005 / rs;
            let sq1x2 = (1.0 + x * x).sqrt();
            let h1 = 1.0
                / (1.0
                    + (x / sq1x2).powf(6.0)
                        * flexmesa_math::pow(z, -ONE_THIRD));
            let h2 = 1.0 / sq1x2;
            let fie = (-game
                * (cdh * game.sqrt() + ctf * a * game.powf(nu) * g1 * h1))
                / (1.0
                    + (b * game.sqrt() + (a * g2 * game.powf(nu)) / rs) * h2);
            fii + fie
        }
    }
}

#[inline]
fn eval_vs_coulomb_ad(
    which_vs: CoulombVsOption,
    rs: AutoDiffReal2VarOrder1,
    z: f64,
) -> AutoDiffReal2VarOrder1 {
    if which_vs == CoulombVsOption::None || z <= 0.0 {
        return AutoDiffReal2VarOrder1::zero();
    }
    match which_vs {
        CoulombVsOption::None => AutoDiffReal2VarOrder1::zero(),
        CoulombVsOption::Throwe1983 => AutoDiffReal2VarOrder1::constant(
            2.0 * z * FINE * (FINE / PI).sqrt(),
        ),
        CoulombVsOption::Itoh2002 => {
            let rs_clamped = if rs.val < 1.0e-5 {
                AutoDiffReal2VarOrder1::constant(1.0e-5)
            } else {
                rs
            };
            let s = (rs_clamped.log10() + 3.0) / 2.0;
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
            let fj = if s.val == 0.0 {
                AutoDiffReal2VarOrder1::constant(coeffs[0])
            } else {
                let mut acc = AutoDiffReal2VarOrder1::zero();
                for (i, &c) in coeffs.iter().enumerate() {
                    acc = c * s.powf(i as f64) + acc;
                }
                acc
            };
            1.46e-2 * z * fj
        }
    }
}

/// Evaluates the dimensionless ion excess Coulomb chemical potential
/// $\mu_{\mathrm{I}} / (k_{\mathrm{B}} T)$ and its logarithmic temperature and
/// density derivatives $(\mu_{\mathrm{I}}, \partial\mu_{\mathrm{I}}/\partial\ln T,
/// \partial\mu_{\mathrm{I}}/\partial\ln\rho)$ (`do_muI_coulomb` in
/// `rates/private/eval_coulomb.f90`).
#[must_use]
pub fn eval_mui_coulomb_with_derivs(
    which_mui: CoulombMuiOption,
    temp: f64,
    den: f64,
    zbar: f64,
    abar: f64,
    z: f64,
) -> (f64, f64, f64) {
    if which_mui == CoulombMuiOption::None || z <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let (game, rs) = coulomb_plasma_params_ad(temp, den, zbar, abar);
    let mu = eval_mui_coulomb_ad(which_mui, game, rs, zbar, z);
    (mu.val, mu.d1[0], mu.d1[1])
}

/// Evaluates the dimensionless electron screening potential shift $V_s$ and its
/// logarithmic temperature and density derivatives $(V_s, \partial V_s/\partial\ln T,
/// \partial V_s/\partial\ln\rho)$ (`do_Vs_coulomb` in `rates/private/eval_coulomb.f90`).
#[must_use]
pub fn eval_vs_coulomb_with_derivs(
    which_vs: CoulombVsOption,
    _temp: f64,
    den: f64,
    zbar: f64,
    abar: f64,
    z: f64,
) -> (f64, f64, f64) {
    if which_vs == CoulombVsOption::None || z <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let rs = coulomb_rs_ad(den, zbar, abar);
    let vs = eval_vs_coulomb_ad(which_vs, rs, z);
    (vs.val, vs.d1[0], vs.d1[1])
}

#[inline]
fn fd_complete_ad(
    k: usize,
    y: AutoDiffReal2VarOrder1,
) -> AutoDiffReal2VarOrder1 {
    dfermi_order1(k as f64, y, AutoDiffReal2VarOrder1::zero())
}

#[allow(clippy::too_many_arguments)]
fn eval_ecapture_with_data(
    states_i: &[(f64, f64)],
    states_f: &[(f64, f64)],
    trans: &[(usize, usize, f64)],
    z_lhs: f64,
    z_rhs: f64,
    q_tot: f64,
    t9: f64,
    ye_rho: f64,
    eta_k: f64,
    d_etak_dln_t: f64,
    d_etak_dln_rho: f64,
    which_mui: CoulombMuiOption,
    which_vs: CoulombVsOption,
    zbar: f64,
    abar: f64,
) -> Result<WeakRateEval, RatesError> {
    let kbt_val = (1.0e3 * KEV) * t9; // MeV
    let kbt = AutoDiffReal2VarOrder1::new(kbt_val, [kbt_val, 0.0]);
    let mec2 = ME * CLIGHT * CLIGHT / MEV_TO_ERGS; // MeV
    let beta = mec2 / kbt;
    let eta = AutoDiffReal2VarOrder1::new(
        eta_k + beta.val,
        [d_etak_dln_t + beta.d1[0], d_etak_dln_rho + beta.d1[1]],
    );
    if eta.val < 2.0 * beta.val {
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
    let (game, rs) = coulomb_plasma_params_ad(temp, den, zbar, abar);
    let dmu = eval_mui_coulomb_ad(which_mui, game, rs, zbar, z_lhs)
        - eval_mui_coulomb_ad(which_mui, game, rs, zbar, z_rhs);
    let dq = dmu * kbt;
    let f_mue = 1.0 - eval_vs_coulomb_ad(which_vs, rs, z_lhs);
    let eta_tilde = f_mue * eta;

    let bf: Vec<AutoDiffReal2VarOrder1> = states_i
        .iter()
        .map(|&(e, j)| (2.0 * j + 1.0) * (-e / kbt).exp())
        .collect();
    let mut zpart = AutoDiffReal2VarOrder1::zero();
    for &b in &bf {
        zpart = zpart + b;
    }
    let ppi: Vec<AutoDiffReal2VarOrder1> =
        bf.iter().map(|&b| b / zpart).collect();

    let is_ec = (z_lhs - z_rhs) > 0.0;
    let (qx, ggm) = if is_ec {
        (q_tot - mec2, flexmesa_math::exp(PI * FINE * z_lhs))
    } else {
        (q_tot + mec2, flexmesa_math::exp(PI * FINE * z_rhs))
    };
    let q_out = if is_ec { qx + mec2 } else { qx - mec2 };

    let mut lam = AutoDiffReal2VarOrder1::zero();
    let mut eps_nu = AutoDiffReal2VarOrder1::zero();

    for &(si, sf, logft) in trans {
        let ei = states_i
            .get(si.saturating_sub(1))
            .filter(|_| si > 0)
            .ok_or_else(|| {
                RatesError::InvalidTableData(format!(
                    "ecapture initial state index {si} out of range (len={})",
                    states_i.len()
                ))
            })?
            .0;
        let ef = states_f
            .get(sf.saturating_sub(1))
            .filter(|_| sf > 0)
            .ok_or_else(|| {
                RatesError::InvalidTableData(format!(
                    "ecapture final state index {sf} out of range (len={})",
                    states_f.len()
                ))
            })?
            .0;
        let zeta = (qx + dq - ef + ei) / kbt;
        let zero = AutoDiffReal2VarOrder1::zero();
        let (ie, je) = if is_ec {
            if zeta.val > -beta.val {
                (zero, zero)
            } else {
                let y = zeta + eta_tilde;
                let (f2, f3, f4, f5) = (
                    fd_complete_ad(2, y),
                    fd_complete_ad(3, y),
                    fd_complete_ad(4, y),
                    fd_complete_ad(5, y),
                );
                let c3 = -2.0 * zeta;
                let c2 = zeta * zeta;
                let ie_v = (f4 + c3 * f3 + c2 * f2) / beta.pow5();
                let je_v = (f5 + c3 * f4 + c2 * f3) / beta.pow6();
                (ie_v, je_v)
            }
        } else if zeta.val < beta.val {
            (zero, zero)
        } else {
            let y1 = zeta - eta_tilde;
            let y2 = beta - eta_tilde;
            let (f2_1, f3_1, f4_1, f5_1) = (
                fd_complete_ad(2, y1),
                fd_complete_ad(3, y1),
                fd_complete_ad(4, y1),
                fd_complete_ad(5, y1),
            );
            let (f0_2, f1_2, f2_2, f3_2, f4_2, f5_2) = (
                fd_complete_ad(0, y2),
                fd_complete_ad(1, y2),
                fd_complete_ad(2, y2),
                fd_complete_ad(3, y2),
                fd_complete_ad(4, y2),
                fd_complete_ad(5, y2),
            );
            let c3_1 = -2.0 * zeta;
            let c2_1 = zeta * zeta;
            let ie_base = f4_1 + c3_1 * f3_1 + c2_1 * f2_1;
            let je_base = f5_1 + c3_1 * f4_1 + c2_1 * f3_1;

            let c3 = 2.0 * zeta - 4.0 * beta;
            let c2 = 6.0 * beta * beta - 6.0 * beta * zeta + zeta * zeta;
            let c1 = -2.0
                * beta
                * (2.0 * beta * beta - 3.0 * beta * zeta + zeta * zeta);
            let c0 = beta * beta * (beta - zeta) * (beta - zeta);
            let ie_v = (ie_base
                - (f4_2 + c3 * f3_2 + c2 * f2_2 + c1 * f1_2 + c0 * f0_2))
                / beta.pow5();

            let d4 = 3.0 * zeta - 5.0 * beta;
            let d3 =
                10.0 * beta * beta - 12.0 * beta * zeta + 3.0 * zeta * zeta;
            let d2 = zeta * zeta * zeta - 9.0 * beta * zeta * zeta
                + 18.0 * beta * beta * zeta
                - 10.0 * beta * beta * beta;
            let d1 = beta
                * (5.0 * beta - 2.0 * zeta)
                * (beta - zeta)
                * (beta - zeta);
            let d0 = -beta * beta * (beta - zeta).pow3();
            let je_v = (je_base
                - (f5_2
                    + d4 * f4_2
                    + d3 * f3_2
                    + d2 * f2_2
                    + d1 * f1_2
                    + d0 * f0_2))
                / beta.pow6();
            (ie_v, je_v)
        };

        let ie_gm = ggm * ie;
        let je_gm = ggm * je;
        if ie_gm.val > 0.0 && je_gm.val > 0.0 {
            let prob_i = ppi[si - 1];
            let ln2ft = LN2 * flexmesa_math::exp10(-logft);
            lam = lam + ie_gm * ln2ft * prob_i;
            eps_nu = eps_nu + mec2 * je_gm * ln2ft * prob_i;
        }
    }

    let qneu = if lam.val > 1.0e-30 {
        eps_nu / lam
    } else {
        AutoDiffReal2VarOrder1::zero()
    };
    Ok(WeakRateEval {
        lambda: lam.val,
        dlambda_dln_t: lam.d1[0],
        dlambda_dln_rho: lam.d1[1],
        q: q_out,
        dq_dln_t: 0.0,
        dq_dln_rho: 0.0,
        qneu: qneu.val,
        dqneu_dln_t: qneu.d1[0],
        dqneu_dln_rho: qneu.d1[1],
    })
}

/// Evaluates on-the-fly electron-capture and $\beta^-$-decay rates via exact
/// relativistic Fermi-Dirac phase-space integrals $I_{\mathrm{ec}}, J_{\mathrm{ec}},
/// I_{\mathrm{ee}}, J_{\mathrm{ee}}$ (`eval_ecapture.f90` and `eval_psi.f90`),
/// assuming zero kinetic degeneracy derivatives (`d_etak_dln_t = d_etak_dln_rho = 0`).
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
    eval_ecapture_pair_with_derivs(
        lhs_name, rhs_name, t9, ye_rho, eta_k, 0.0, 0.0, which_mui, which_vs,
        zbar, abar,
    )
}

/// Evaluates on-the-fly electron-capture and $\beta^-$-decay rates with full
/// logarithmic temperature and density derivatives using custom nuclear state
/// and transition tables ([`EcaptureTables`]).
///
/// # Errors
/// Returns [`RatesError::UnknownWeakPair`] if `(lhs_name, rhs_name)` is not in
/// `tables`, or [`RatesError::InvalidTableData`] if an isotope or state index is
/// invalid.
#[allow(clippy::too_many_arguments)]
pub fn eval_ecapture_pair_with_tables(
    tables: &EcaptureTables,
    lhs_name: &str,
    rhs_name: &str,
    t9: f64,
    ye_rho: f64,
    eta_k: f64,
    d_etak_dln_t: f64,
    d_etak_dln_rho: f64,
    which_mui: CoulombMuiOption,
    which_vs: CoulombVsOption,
    zbar: f64,
    abar: f64,
) -> Result<WeakRateEval, RatesError> {
    let (states_i, states_f, trans, z_lhs, z_rhs, q_tot) =
        tables.pair_data(lhs_name, rhs_name)?;
    eval_ecapture_with_data(
        states_i,
        states_f,
        trans,
        z_lhs,
        z_rhs,
        q_tot,
        t9,
        ye_rho,
        eta_k,
        d_etak_dln_t,
        d_etak_dln_rho,
        which_mui,
        which_vs,
        zbar,
        abar,
    )
}

/// Evaluates on-the-fly electron-capture and $\beta^-$-decay rates including full
/// logarithmic temperature and density derivatives (`dlambda_dln_t`,
/// `dlambda_dln_rho`, `dqneu_dln_t`, `dqneu_dln_rho`) using upstream MESA's
/// first-order two-variable dual arithmetic (`eval_ecapture.f90` and `eval_psi.f90`).
///
/// # Errors
/// Returns [`RatesError::UnknownWeakPair`] if `(lhs_name, rhs_name)` is not in
/// the built-in ecapture transitions catalog, or [`RatesError::InvalidTableData`]
/// if an isotope or state index is invalid.
#[allow(clippy::too_many_arguments)]
pub fn eval_ecapture_pair_with_derivs(
    lhs_name: &str,
    rhs_name: &str,
    t9: f64,
    ye_rho: f64,
    eta_k: f64,
    d_etak_dln_t: f64,
    d_etak_dln_rho: f64,
    which_mui: CoulombMuiOption,
    which_vs: CoulombVsOption,
    zbar: f64,
    abar: f64,
) -> Result<WeakRateEval, RatesError> {
    let cat = RatesCatalog::builtin();
    let (states_i, states_f, trans, z_lhs, z_rhs, q_tot) =
        cat.special_ecapture_data(lhs_name, rhs_name)?;
    eval_ecapture_with_data(
        states_i,
        states_f,
        trans,
        z_lhs,
        z_rhs,
        q_tot,
        t9,
        ye_rho,
        eta_k,
        d_etak_dln_t,
        d_etak_dln_rho,
        which_mui,
        which_vs,
        zbar,
        abar,
    )
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
        let exeta = if eta <= 6.8e2 {
            flexmesa_math::exp(eta)
        } else {
            0.0
        };
        let etael2 = etael * etael;
        let etael3 = etael2 * etael;
        let etael4 = etael3 * etael;
        let etael5 = etael4 * etael;
        let zetan2 = zetan * zetan;
        let f0 = if eta <= 6.8e2 {
            flexmesa_math::log1p(exeta)
        } else {
            eta
        };

        let f1l = exeta;
        let f2l = 2.0 * f1l;
        let f3l = 6.0 * f1l;
        let f4l = 24.0 * f1l;
        let f5l = 120.0 * f1l;

        let (f1g, f2g, f3g, f4g, f5g) = if eta > 0.0 {
            let exmeta = flexmesa_math::exp(-eta);
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
