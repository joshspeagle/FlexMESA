//! Electron exchange-correlation (`EXCOR7`) for the Potekhin-Chabrier (`PC`)
//! equation of state.
//!
//! Reference: Tanaka & Ichimaru (1985-1987), Ichimaru (1993); upstream
//! Fortran MESA `eos/private/pc_eos.f90` lines 1844-2162 (`use_EXCOR7_table = .true.`)
//! and `eos/private/pc_support.f90` (`get_EXCOR7`).

use flexmesa_autodiff::AutoDiffReal2VarOrder1;

use crate::pc::coulomb_ion::PcIonContrib;
use crate::pc::excor_table::{
    get_excor7, J_CVXC, J_FXC, J_PDRXC, J_PDTXC, J_PXC, J_SXC, J_UXC,
};

/// Evaluate `EXCOR7(RS, GAME)` for the electron exchange-correlation
/// contribution per electron (`pc_eos.f90:1895-2135`).
///
/// Mirrors MESA `26.4.1`'s `use_EXCOR7_table = .true.`: queries the cached
/// `351 x 601` bicubic spline table (`get_excor7`) and falls back to the
/// analytical Tanaka & Ichimaru formula (`excor7_analytic`) when `(ln RS,
/// ln GAME)` lies outside the table domain (`skip = .true.`).
#[must_use]
pub fn excor7(
    rs: AutoDiffReal2VarOrder1,
    game: AutoDiffReal2VarOrder1,
) -> PcIonContrib {
    if let Some(interp) = get_excor7(rs.val, game.val) {
        let dln_rs_dt = rs.d1[0] / rs.val;
        let dln_game_dt = game.d1[0] / game.val;
        let dln_rs_drho = rs.d1[1] / rs.val;
        let dln_game_drho = game.d1[1] / game.val;

        let make_ad = |k: usize| AutoDiffReal2VarOrder1 {
            val: interp.fval[k],
            d1: [
                interp.df_dln_rs[k] * dln_rs_dt
                    + interp.df_dln_game[k] * dln_game_dt,
                interp.df_dln_rs[k] * dln_rs_drho
                    + interp.df_dln_game[k] * dln_game_drho,
            ],
        };

        return PcIonContrib {
            f: make_ad(J_FXC),
            u: make_ad(J_UXC),
            p: make_ad(J_PXC),
            s: make_ad(J_SXC),
            cv: make_ad(J_CVXC),
            pdt: make_ad(J_PDTXC),
            pdr: make_ad(J_PDRXC),
        };
    }

    excor7_analytic(rs, game)
}

/// Evaluate the analytical Tanaka & Ichimaru `EXCOR7(RS, GAME)` fallback used
/// when `(ln RS, ln GAME)` lies outside the `EXCOR7` spline table bounds
/// (`pc_eos.f90:1913-2083`, `THETA > 0.04`).
#[must_use]
pub fn excor7_analytic(
    rs: AutoDiffReal2VarOrder1,
    game: AutoDiffReal2VarOrder1,
) -> PcIonContrib {
    let theta = 0.543 * rs / game;
    let sqth = theta.sqrt();
    let theta2 = theta * theta;
    let theta3 = theta2 * theta;
    let theta4 = theta3 * theta;
    let (t1, t2, t1dh, t1dhh, t2dh, t2dhh) = if theta.val > 0.04 {
        let inv_th = 1.0 / theta;
        let inv_sq = 1.0 / sqth;
        let cht1 = inv_th.cosh();
        let sht1 = inv_th.sinh();
        let cht2 = inv_sq.cosh();
        let sht2 = inv_sq.sinh();
        let t1 = sht1 / cht1;
        let t2 = sht2 / cht2;
        let th_cht1 = theta * cht1;
        let th_cht2 = theta * cht2;
        let t1dh = -1.0 / (th_cht1 * th_cht1);
        let t1dhh = 2.0 / th_cht1.pow3() * (cht1 - sht1 / theta);
        let t2dh = -0.5 * sqth / (th_cht2 * th_cht2);
        let t2dhh = (0.75 * sqth * cht2 - 0.5 * sht2) / th_cht2.pow3();
        (t1, t2, t1dh, t1dhh, t2dh, t2dhh)
    } else {
        let one = AutoDiffReal2VarOrder1::constant(1.0);
        let zero = AutoDiffReal2VarOrder1::zero();
        (one, one, zero, zero, zero, zero)
    };
    let a0 = 0.75 + 3.04363 * theta2 - 0.09227 * theta3 + 1.7035 * theta4;
    let a0dh = 6.08726 * theta - 0.27681 * theta2 + 6.814 * theta3;
    let a0dhh = 6.08726 - 0.55362 * theta + 20.442 * theta2;
    let a1 = 1.0 + 8.31051 * theta2 + 5.1105 * theta4;
    let a1dh = 16.62102 * theta + 20.442 * theta3;
    let a1dhh = 16.62102 + 61.326 * theta2;
    let a = 0.610887 * a0 / a1 * t1;
    let ah = a0dh / a0 - a1dh / a1 + t1dh / t1;
    let adh = a * ah;
    let adhh = adh * ah
        + a * (a0dhh / a0 - (a0dh / a0).pow2() - a1dhh / a1
            + (a1dh / a1).pow2()
            + t1dhh / t1
            - (t1dh / t1).pow2());

    let b0 = 0.341308 + 12.070873 * theta2 + 1.148889 * theta4;
    let b0dh = 24.141746 * theta + 4.595556 * theta3;
    let b0dhh = 24.141746 + 13.786668 * theta2;
    let b1 = 1.0 + 10.495346 * theta2 + 1.326623 * theta4;
    let b1dh = 20.990692 * theta + 5.306492 * theta3;
    let b1dhh = 20.990692 + 15.919476 * theta2;
    let b = sqth * t2 * b0 / b1;
    let bh = 0.5 / theta + t2dh / t2 + b0dh / b0 - b1dh / b1;
    let bdh = b * bh;
    let bdhh = bdh * bh
        + b * (-0.5 / theta2 + t2dhh / t2 - (t2dh / t2).pow2() + b0dhh / b0
            - (b0dh / b0).pow2()
            - b1dhh / b1
            + (b1dh / b1).pow2());

    let d0 = 0.614925 + 16.996055 * theta2 + 1.489056 * theta4;
    let d0dh = 33.99211 * theta + 5.956224 * theta3;
    let d0dhh = 33.99211 + 17.868672 * theta2;
    let d1 = 1.0 + 10.10935 * theta2 + 1.22184 * theta4;
    let d1dh = 20.2187 * theta + 4.88736 * theta3;
    let d1dhh = 20.2187 + 14.66208 * theta2;
    let d = sqth * t2 * d0 / d1;
    let dh = 0.5 / theta + t2dh / t2 + d0dh / d0 - d1dh / d1;
    let ddh = d * dh;
    let ddhh = ddh * dh
        + d * (-0.5 / theta2 + t2dhh / t2 - (t2dh / t2).pow2() + d0dhh / d0
            - (d0dh / d0).pow2()
            - d1dhh / d1
            + (d1dh / d1).pow2());

    let e0 = 0.539409 + 2.522206 * theta2 + 0.178484 * theta4;
    let e0dh = 5.044412 * theta + 0.713936 * theta3;
    let e0dhh = 5.044412 + 2.141808 * theta2;
    let e1 = 1.0 + 2.555501 * theta2 + 0.146319 * theta4;
    let e1dh = 5.111002 * theta + 0.585276 * theta3;
    let e1dhh = 5.111002 + 1.755828 * theta2;
    let e = theta * t1 * e0 / e1;
    let eh = 1.0 / theta + t1dh / t1 + e0dh / e0 - e1dh / e1;
    let edh = e * eh;
    let edhh = edh * eh
        + e * (t1dhh / t1 - (t1dh / t1).pow2() + e0dhh / e0
            - (e0dh / e0).pow2()
            - e1dhh / e1
            + (e1dh / e1).pow2()
            - 1.0 / theta2);

    let exp1th = (-1.0 / theta).exp();
    let c = (0.872496 + 0.025248 * exp1th) * e;
    let cdh = 0.025248 * exp1th / theta2 * e + c * edh / e;
    let cdhh = 0.025248 * exp1th / theta2
        * (edh + (1.0 - 2.0 * theta) / theta2 * e)
        + cdh * edh / e
        + c * edhh / e
        - c * (edh / e).pow2();

    let discr = (4.0 * e - d * d).sqrt();
    let didh = 0.5 / discr * (4.0 * edh - 2.0 * d * ddh);
    let didhh = (-((2.0 * edh - d * ddh) / discr).pow2() + 2.0 * edhh
        - ddh * ddh
        - d * ddhh)
        / discr;

    let s1 = -c / e * game;
    let s1h = cdh / c - edh / e;
    let s1dh = s1 * s1h;
    let s1dhh = s1dh * s1h
        + s1 * (cdhh / c - (cdh / c).pow2() - edhh / e + (edh / e).pow2());
    let s1dg = -c / e;
    let s1dhg = s1dg * (cdh / c - edh / e);

    let b2 = b - c * d / e;
    let b2dh = bdh - (cdh * d + c * ddh) / e + c * d * edh / (e * e);
    let b2dhh = bdhh - (cdhh * d + 2.0 * cdh * ddh + c * ddhh) / e
        + (2.0 * (cdh * d + c * ddh - c * d * edh / e) * edh + c * d * edhh)
            / (e * e);

    let sqge = game.sqrt();
    let s2 = -2.0 / e * b2 * sqge;
    let s2h = b2dh / b2 - edh / e;
    let s2dh = s2 * s2h;
    let s2dhh = s2dh * s2h
        + s2 * (b2dhh / b2 - (b2dh / b2).pow2() - edhh / e + (edh / e).pow2());
    let s2dg = 0.5 * s2 / game;
    let s2dgg = -0.5 * s2dg / game;
    let s2dhg = 0.5 * s2dh / game;

    let r3 = e * game + d * sqge + 1.0;
    let r3dh = edh * game + ddh * sqge;
    let r3dhh = edhh * game + ddhh * sqge;
    let r3dg = e + 0.5 * d / sqge;
    let r3dgg = -0.25 * d / (game * sqge);
    let r3dhg = edh + 0.5 * ddh / sqge;

    let b3 = a - c / e;
    let b3dh = adh - cdh / e + c * edh / (e * e);
    let b3dhh = adhh - cdhh / e + (2.0 * cdh * edh + c * edhh) / (e * e)
        - 2.0 * c * edh * edh / e.pow3();

    let c3 = (d / e * b2 - b3) / e;
    let c3dh = (ddh * b2 + d * b2dh + b3 * edh) / (e * e)
        - 2.0 * d * b2 * edh / e.pow3()
        - b3dh / e;
    let c3dhh = (-b3dhh
        + (ddhh * b2
            + 2.0 * ddh * b2dh
            + d * b2dhh
            + b3dh * edh
            + b3 * edhh
            + b3dh * edh)
            / e
        - 2.0
            * ((ddh * b2 + d * b2dh + b3 * edh + ddh * b2 + d * b2dh) * edh
                + d * b2 * edhh)
            / (e * e)
        + 6.0 * d * b2 * edh * edh / e.pow3())
        / e;

    let s3 = c3 * r3.log();
    let s3dh = s3 * c3dh / c3 + c3 * r3dh / r3;
    let s3dhh = (s3dh * c3dh + s3 * c3dhh) / c3 - s3 * (c3dh / c3).pow2()
        + (c3dh * r3dh + c3 * r3dhh) / r3
        - c3 * (r3dh / r3).pow2();
    let s3dg = c3 * r3dg / r3;
    let s3dgg = c3 * (r3dgg / r3 - (r3dg / r3).pow2());
    let s3dhg = (c3dh * r3dg + c3 * r3dhg) / r3 - c3 * r3dg * r3dh / (r3 * r3);

    let b4 = 2.0 - d * d / e;
    let b4dh = edh * (d / e) * (d / e) - 2.0 * d * ddh / e;
    let b4dhh = edhh * (d / e) * (d / e)
        + 2.0 * edh * (d / e) * (d / e) * (ddh / d - edh / e)
        - 2.0 * (ddh * ddh + d * ddhh) / e
        + 2.0 * d * ddh * edh / (e * e);

    let c4 = 2.0 * e * sqge + d;
    let c4dh = 2.0 * edh * sqge + ddh;
    let c4dhh = 2.0 * edhh * sqge + ddhh;
    let c4dg = e / sqge;
    let c4dgg = -0.5 * e / (game * sqge);
    let c4dhg = edh / sqge;

    let s4a = 2.0 / e / discr;
    let s4ah = edh / e + didh / discr;
    let s4adh = -s4a * s4ah;
    let s4adhh = -s4adh * s4ah
        - s4a
            * (edhh / e - (edh / e) * (edh / e) + didhh / discr
                - (didh / discr).pow2());

    let s4b = d * b3 + b4 * b2;
    let s4bdh = ddh * b3 + d * b3dh + b4dh * b2 + b4 * b2dh;
    let s4bdhh = ddhh * b3
        + 2.0 * ddh * b3dh
        + d * b3dhh
        + b4dhh * b2
        + 2.0 * b4dh * b2dh
        + b4 * b2dhh;

    let s4c = (c4 / discr).atan() - (d / discr).atan();
    let up1 = c4dh * discr - c4 * didh;
    let dn1 = discr * discr + c4 * c4;
    let up2 = ddh * discr - d * didh;
    let dn2 = discr * discr + d * d;
    let s4cdh = up1 / dn1 - up2 / dn2;
    let s4cdhh = (c4dhh * discr - c4 * didhh) / dn1
        - up1 * 2.0 * (discr * didh + c4 * c4dh) / (dn1 * dn1)
        - (ddhh * discr - d * didhh) / dn2
        + up2 * 2.0 * (discr * didh + d * ddh) / (dn2 * dn2);
    let s4cdg = c4dg * discr / dn1;
    let s4cdgg = c4dgg * discr / dn1 - 2.0 * c4 * discr * (c4dg / dn1).pow2();
    let s4cdhg = (c4dhg * discr + c4dg * didh
        - c4dg * discr / dn1 * 2.0 * (discr * didh + c4 * c4dh))
        / dn1;

    let s4 = s4a * s4b * s4c;
    let s4dh = s4adh * s4b * s4c + s4a * s4bdh * s4c + s4a * s4b * s4cdh;
    let s4dhh = s4adhh * s4b * s4c
        + s4a * s4bdhh * s4c
        + s4a * s4b * s4cdhh
        + 2.0
            * (s4adh * s4bdh * s4c + s4adh * s4b * s4cdh + s4a * s4bdh * s4cdh);
    let s4dg = s4a * s4b * s4cdg;
    let s4dgg = s4a * s4b * s4cdgg;
    let s4dhg = s4a * s4b * s4cdhg + s4cdg * (s4adh * s4b + s4a * s4bdh);

    let fxc = s1 + s2 + s3 + s4;
    let fxcdh = s1dh + s2dh + s3dh + s4dh;
    let fxcdg = s1dg + s2dg + s3dg + s4dg;
    let fxcdhh = s1dhh + s2dhh + s3dhh + s4dhh;
    let fxcdgg = s2dgg + s3dgg + s4dgg;
    let fxcdhg = s1dhg + s2dhg + s3dhg + s4dhg;

    let pxc = (game * fxcdg - 2.0 * theta * fxcdh) / 3.0;
    let uxc = game * fxcdg - theta * fxcdh;
    let mut sxc = (game * s2dg - s2 + game * s3dg - s3
        + s4a * s4b * (game * s4cdg - s4c))
        - theta * fxcdh;
    if sxc.val.abs() < 1.0e-9 * (theta.val * fxcdh.val).abs() {
        sxc = AutoDiffReal2VarOrder1::zero();
    }
    let mut cvxc = 2.0 * theta * (game * fxcdhg - fxcdh)
        - theta * theta * fxcdhh
        - game * game * fxcdgg;
    if cvxc.val.abs() < 1.0e-9 * (game.val * game.val * fxcdgg.val).abs() {
        cvxc = AutoDiffReal2VarOrder1::zero();
    }
    let pdlh =
        theta * (game * fxcdhg - 2.0 * fxcdh - 2.0 * theta * fxcdhh) / 3.0;
    let pdlg = game * (fxcdg + game * fxcdgg - 2.0 * theta * fxcdhg) / 3.0;
    let pdrxc = pxc + (pdlg - 2.0 * pdlh) / 3.0;
    let pdtxc = game * (theta * fxcdhg - game * fxcdgg / 3.0)
        - theta * (fxcdh / 0.75 + theta * fxcdhh / 1.5);

    PcIonContrib {
        f: fxc,
        u: uxc,
        p: pxc,
        s: sxc,
        cv: cvxc,
        pdt: pdtxc,
        pdr: pdrxc,
    }
}
