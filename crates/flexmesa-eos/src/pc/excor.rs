//! Electron exchange-correlation (`EXCOR7`) for the Potekhin-Chabrier (`PC`)
//! equation of state.
//!
//! Reference: Tanaka & Ichimaru (1985-1987), Ichimaru (1993); upstream
//! Fortran MESA `eos/private/pc_eos.f90` lines 1844-2083.

use flexmesa_autodiff::AutoDiffReal2VarOrder1;

use crate::pc::coulomb_ion::PcIonContrib;

/// Evaluate `EXCOR7(RS, GAME)` for the electron exchange-correlation
/// contribution per electron (`pc_eos.f90` lines 1844-2083).
#[must_use]
pub fn excor7(
    rs: AutoDiffReal2VarOrder1,
    game: AutoDiffReal2VarOrder1,
) -> PcIonContrib {
    let theta = rs * 0.543 / game;
    let sqth = theta.sqrt();
    let theta2 = theta * theta;
    let theta3 = theta2 * theta;
    let theta4 = theta3 * theta;
    let (t1, t2, t1dh, t1dhh, t2dh, t2dhh) = if theta.val > 0.04 {
        let inv_th = theta.recip();
        let inv_sq = sqth.recip();
        let cht1 = inv_th.cosh();
        let sht1 = inv_th.sinh();
        let cht2 = inv_sq.cosh();
        let sht2 = inv_sq.sinh();
        let t1 = sht1 / cht1;
        let t2 = sht2 / cht2;
        let th_cht1 = theta * cht1;
        let th_cht2 = theta * cht2;
        let t1dh = -th_cht1.pow2().recip();
        let t1dhh = (cht1 - sht1 / theta) * 2.0 / th_cht1.pow3();
        let t2dh = -sqth * 0.5 / th_cht2.pow2();
        let t2dhh = (sqth * cht2 * 0.75 - sht2 * 0.5) / th_cht2.pow3();
        (t1, t2, t1dh, t1dhh, t2dh, t2dhh)
    } else {
        let one = AutoDiffReal2VarOrder1::constant(1.0);
        let zero = AutoDiffReal2VarOrder1::zero();
        (one, one, zero, zero, zero, zero)
    };
    let a0 = theta2 * 3.04363 - theta3 * 0.09227 + theta4 * 1.7035 + 0.75;
    let a0dh = theta * 6.08726 - theta2 * 0.27681 + theta3 * 6.814;
    let a0dhh = -theta * 0.55362 + theta2 * 20.442 + 6.08726;
    let a1 = theta2 * 8.31051 + theta4 * 5.1105 + 1.0;
    let a1dh = theta * 16.62102 + theta3 * 20.442;
    let a1dhh = theta2 * 61.326 + 16.62102;
    let a = a0 / a1 * t1 * 0.610887;
    let ah = a0dh / a0 - a1dh / a1 + t1dh / t1;
    let adh = a * ah;
    let adhh = adh * ah
        + a * (a0dhh / a0 - (a0dh / a0).pow2() - a1dhh / a1
            + (a1dh / a1).pow2()
            + t1dhh / t1
            - (t1dh / t1).pow2());

    let b0 = theta2 * 12.070873 + theta4 * 1.148889 + 0.341308;
    let b0dh = theta * 24.141746 + theta3 * 4.595556;
    let b0dhh = theta2 * 13.786668 + 24.141746;
    let b1 = theta2 * 10.495346 + theta4 * 1.326623 + 1.0;
    let b1dh = theta * 20.990692 + theta3 * 5.306492;
    let b1dhh = theta2 * 15.919476 + 20.990692;
    let b = sqth * t2 * b0 / b1;
    let bh = theta.recip() * 0.5 + t2dh / t2 + b0dh / b0 - b1dh / b1;
    let bdh = b * bh;
    let bdhh = bdh * bh
        + b * (-theta2.recip() * 0.5 + t2dhh / t2 - (t2dh / t2).pow2()
            + b0dhh / b0
            - (b0dh / b0).pow2()
            - b1dhh / b1
            + (b1dh / b1).pow2());

    let d0 = theta2 * 16.996055 + theta4 * 1.489056 + 0.614925;
    let d0dh = theta * 33.99211 + theta3 * 5.956224;
    let d0dhh = theta2 * 17.868672 + 33.99211;
    let d1 = theta2 * 10.10935 + theta4 * 1.22184 + 1.0;
    let d1dh = theta * 20.2187 + theta3 * 4.88736;
    let d1dhh = theta2 * 14.66208 + 20.2187;
    let d = sqth * t2 * d0 / d1;
    let dh = theta.recip() * 0.5 + t2dh / t2 + d0dh / d0 - d1dh / d1;
    let ddh = d * dh;
    let ddhh = ddh * dh
        + d * (-theta2.recip() * 0.5 + t2dhh / t2 - (t2dh / t2).pow2()
            + d0dhh / d0
            - (d0dh / d0).pow2()
            - d1dhh / d1
            + (d1dh / d1).pow2());

    let e0 = theta2 * 2.522206 + theta4 * 0.178484 + 0.539409;
    let e0dh = theta * 5.044412 + theta3 * 0.713936;
    let e0dhh = theta2 * 2.141808 + 5.044412;
    let e1 = theta2 * 2.555501 + theta4 * 0.146319 + 1.0;
    let e1dh = theta * 5.111002 + theta3 * 0.585276;
    let e1dhh = theta2 * 1.755828 + 5.111002;
    let e = theta * t1 * e0 / e1;
    let eh = theta.recip() + t1dh / t1 + e0dh / e0 - e1dh / e1;
    let edh = e * eh;
    let edhh = edh * eh
        + e * (t1dhh / t1 - (t1dh / t1).pow2() + e0dhh / e0
            - (e0dh / e0).pow2()
            - e1dhh / e1
            + (e1dh / e1).pow2()
            - theta2.recip());

    let exp1th = (-theta.recip()).exp();
    let c = (exp1th * 0.025248 + 0.872496) * e;
    let cdh = exp1th * 0.025248 / theta2 * e + c * edh / e;
    let cdhh = exp1th * 0.025248 / theta2
        * (edh + (-theta * 2.0 + 1.0) / theta2 * e)
        + cdh * edh / e
        + c * edhh / e
        - c * (edh / e).pow2();

    let discr = (e * 4.0 - d * d).sqrt();
    let didh = (edh * 4.0 - d * ddh * 2.0) * 0.5 / discr;
    let didhh = (-((edh * 2.0 - d * ddh) / discr).pow2() + edhh * 2.0
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
    let b2dhh = bdhh - (cdhh * d + cdh * ddh * 2.0 + c * ddhh) / e
        + ((cdh * d + c * ddh - c * d * edh / e) * edh * 2.0 + c * d * edhh)
            / (e * e);

    let sqge = game.sqrt();
    let s2 = -b2 * sqge * 2.0 / e;
    let s2h = b2dh / b2 - edh / e;
    let s2dh = s2 * s2h;
    let s2dhh = s2dh * s2h
        + s2 * (b2dhh / b2 - (b2dh / b2).pow2() - edhh / e + (edh / e).pow2());
    let s2dg = s2 * 0.5 / game;
    let s2dgg = -s2dg * 0.5 / game;
    let s2dhg = s2dh * 0.5 / game;

    let r3 = e * game + d * sqge + 1.0;
    let r3dh = edh * game + ddh * sqge;
    let r3dhh = edhh * game + ddhh * sqge;
    let r3dg = e + d * 0.5 / sqge;
    let r3dgg = -d * 0.25 / (game * sqge);
    let r3dhg = edh + ddh * 0.5 / sqge;

    let b3 = a - c / e;
    let b3dh = adh - cdh / e + c * edh / (e * e);
    let b3dhh = adhh - cdhh / e + (cdh * edh * 2.0 + c * edhh) / (e * e)
        - c * edh * edh * 2.0 / e.pow3();

    let c3 = (d / e * b2 - b3) / e;
    let c3dh = (ddh * b2 + d * b2dh + b3 * edh) / (e * e)
        - d * b2 * edh * 2.0 / e.pow3()
        - b3dh / e;
    let c3dhh = (-b3dhh
        + (ddhh * b2
            + ddh * b2dh * 2.0
            + d * b2dhh
            + b3dh * edh * 2.0
            + b3 * edhh)
            / e
        - ((ddh * b2 * 2.0 + d * b2dh * 2.0 + b3 * edh) * edh + d * b2 * edhh)
            * 2.0
            / (e * e)
        + d * b2 * edh * edh * 6.0 / e.pow3())
        / e;

    let s3 = c3 * r3.ln();
    let s3dh = s3 * c3dh / c3 + c3 * r3dh / r3;
    let s3dhh = (s3dh * c3dh + s3 * c3dhh) / c3 - s3 * (c3dh / c3).pow2()
        + (c3dh * r3dh + c3 * r3dhh) / r3
        - c3 * (r3dh / r3).pow2();
    let s3dg = c3 * r3dg / r3;
    let s3dgg = c3 * (r3dgg / r3 - (r3dg / r3).pow2());
    let s3dhg = (c3dh * r3dg + c3 * r3dhg) / r3 - c3 * r3dg * r3dh / (r3 * r3);

    let b4 = -d * d / e + 2.0;
    let b4dh = edh * (d / e) * (d / e) - d * ddh * 2.0 / e;
    let b4dhh = edhh * (d / e) * (d / e)
        + edh * (d / e) * (d / e) * (ddh / d - edh / e) * 2.0
        - (ddh * ddh + d * ddhh) * 2.0 / e
        + d * ddh * edh * 2.0 / (e * e);

    let c4 = e * sqge * 2.0 + d;
    let c4dh = edh * sqge * 2.0 + ddh;
    let c4dhh = edhh * sqge * 2.0 + ddhh;
    let c4dg = e / sqge;
    let c4dgg = -e * 0.5 / (game * sqge);
    let c4dhg = edh / sqge;

    let s4a = (e * discr).recip() * 2.0;
    let s4ah = edh / e + didh / discr;
    let s4adh = -s4a * s4ah;
    let s4adhh = -s4adh * s4ah
        - s4a
            * (edhh / e - (edh / e).pow2() + didhh / discr
                - (didh / discr).pow2());

    let s4b = d * b3 + b4 * b2;
    let s4bdh = ddh * b3 + d * b3dh + b4dh * b2 + b4 * b2dh;
    let s4bdhh = ddhh * b3
        + ddh * b3dh * 2.0
        + d * b3dhh
        + b4dhh * b2
        + b4dh * b2dh * 2.0
        + b4 * b2dhh;

    let s4c = (c4 / discr).atan() - (d / discr).atan();
    let up1 = c4dh * discr - c4 * didh;
    let dn1 = discr * discr + c4 * c4;
    let up2 = ddh * discr - d * didh;
    let dn2 = discr * discr + d * d;
    let s4cdh = up1 / dn1 - up2 / dn2;
    let s4cdhh = (c4dhh * discr - c4 * didhh) / dn1
        - up1 * (discr * didh + c4 * c4dh) * 2.0 / (dn1 * dn1)
        - (ddhh * discr - d * didhh) / dn2
        + up2 * (discr * didh + d * ddh) * 2.0 / (dn2 * dn2);
    let s4cdg = c4dg * discr / dn1;
    let s4cdgg = c4dgg * discr / dn1 - c4 * discr * (c4dg / dn1).pow2() * 2.0;
    let s4cdhg = (c4dhg * discr + c4dg * didh
        - c4dg * discr / dn1 * (discr * didh + c4 * c4dh) * 2.0)
        / dn1;

    let s4 = s4a * s4b * s4c;
    let s4dh = s4adh * s4b * s4c + s4a * s4bdh * s4c + s4a * s4b * s4cdh;
    let s4dhh = s4adhh * s4b * s4c
        + s4a * s4bdhh * s4c
        + s4a * s4b * s4cdhh
        + (s4adh * s4bdh * s4c + s4adh * s4b * s4cdh + s4a * s4bdh * s4cdh)
            * 2.0;
    let s4dg = s4a * s4b * s4cdg;
    let s4dgg = s4a * s4b * s4cdgg;
    let s4dhg = s4a * s4b * s4cdhg + s4cdg * (s4adh * s4b + s4a * s4bdh);

    let fxc = s1 + s2 + s3 + s4;
    let fxcdh = s1dh + s2dh + s3dh + s4dh;
    let fxcdg = s1dg + s2dg + s3dg + s4dg;
    let fxcdhh = s1dhh + s2dhh + s3dhh + s4dhh;
    let fxcdgg = s2dgg + s3dgg + s4dgg;
    let fxcdhg = s1dhg + s2dhg + s3dhg + s4dhg;

    let pxc = (game * fxcdg - theta * fxcdh * 2.0) / 3.0;
    let uxc = game * fxcdg - theta * fxcdh;
    let mut sxc = (game * s2dg - s2 + game * s3dg - s3
        + s4a * s4b * (game * s4cdg - s4c))
        - theta * fxcdh;
    if sxc.val.abs() < 1.0e-9 * (theta.val * fxcdh.val).abs() {
        sxc = AutoDiffReal2VarOrder1::zero();
    }
    let mut cvxc = theta * (game * fxcdhg - fxcdh) * 2.0
        - theta * theta * fxcdhh
        - game * game * fxcdgg;
    if cvxc.val.abs() < 1.0e-9 * (game.val * game.val * fxcdgg.val).abs() {
        cvxc = AutoDiffReal2VarOrder1::zero();
    }
    let pdlh =
        theta * (game * fxcdhg - fxcdh * 2.0 - theta * fxcdhh * 2.0) / 3.0;
    let pdlg = game * (fxcdg + game * fxcdgg - theta * fxcdhg * 2.0) / 3.0;
    let pdrxc = pxc + (pdlg - pdlh * 2.0) / 3.0;
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
