//! Electron-ion liquid/solid screening (`FSCRliq8`, `FSCRsol8`), linear mixing
//! corrections (`CORMIX`), and single-species non-ideal plasma combiner
//! (`EOSFI8`) for the Potekhin-Chabrier (`PC`) equation of state.
//!
//! Reference: Potekhin & Chabrier (2000, 2009, 2010, 2013); upstream Fortran
//! MESA `eos/private/pc_eos.f90`.

use flexmesa_autodiff::AutoDiffReal2VarOrder1;

use crate::pc::coulomb_ion::PcIonContrib;
pub use crate::pc::eosfi::eosfi8;
use crate::EosError;

/// Evaluate `FSCRliq8(RS, GAME, Zion)` for electron-ion screening in the
/// Coulomb liquid (`pc_eos.f90` lines 781-960).
pub fn fscr_liq8(
    rs: AutoDiffReal2VarOrder1,
    game: AutoDiffReal2VarOrder1,
    zion: f64,
) -> Result<PcIonContrib, EosError> {
    const XRS: f64 = 0.0140047;
    const TINY: f64 = 1.0e-19;
    if rs.val < 0.0 {
        return Err(EosError::InvalidState {
            context: "fscr_liq8",
            detail: "RS < 0".to_string(),
        });
    }
    if rs.val < TINY {
        let z = AutoDiffReal2VarOrder1::zero();
        return Ok(PcIonContrib {
            f: z,
            u: z,
            p: z,
            s: z,
            cv: z,
            pdt: z,
            pdr: z,
        });
    }
    let zion_ad = AutoDiffReal2VarOrder1::constant(zion);
    let sqg = game.sqrt();
    let sqz1 = (1.0 + zion_ad).sqrt();
    let sqz = zion_ad.sqrt();
    let cdh0 = zion_ad / 1.73205;
    let cdh = cdh0 * (sqz1 * sqz1 * sqz1 - sqz * sqz * sqz - 1.0);
    let zln = zion_ad.ln();
    let z13 = (zln / 3.0).exp();
    let x = XRS / rs;
    let ctf = zion_ad * zion_ad * 0.2513 * (z13 - 1.0 + 0.2 / z13.sqrt());
    let p01 = 1.11 * (0.475 * zln).exp();
    let p03 = 0.2 + 0.078 * zln * zln;
    let ptx = 1.16 + 0.08 * zln;
    let tx = game.pow(ptx);
    let txdg = ptx * tx / game;
    let txdgg = (ptx - 1.0) * txdg / game;
    let ty1 = 1.0 / (1.0e-3 * zion_ad * zion_ad + 2.0 * game);
    let ty1dg = -2.0 * ty1 * ty1;
    let ty1dgg = -4.0 * ty1 * ty1dg;
    let ty2 = 1.0 + 6.0 * rs * rs;
    let ty2dx = -12.0 * rs * rs / x;
    let ty2dxx = -3.0 * ty2dx / x;
    let ty = rs * rs * rs / ty2 * (1.0 + ty1);
    let tyx = 3.0 / x + ty2dx / ty2;
    let tydx = -ty * tyx;
    let tydg = rs * rs * rs * ty1dg / ty2;
    let p1 = (zion_ad - 1.0) / 9.0;
    let cor1 = 1.0 + p1 * ty;
    let cor1dx = p1 * tydx;
    let cor1dg = p1 * tydg;
    let cor1dxx = p1
        * (ty * (3.0 / (x * x) + (ty2dx / ty2) * (ty2dx / ty2) - ty2dxx / ty2)
            - tydx * tyx);
    let cor1dgg = p1 * rs * rs * rs * ty1dgg / ty2;
    let cor1dxg = -p1 * tydg * tyx;
    let u0 = 0.78 * (game / zion_ad).sqrt() * rs * rs * rs;
    let u0dx = -3.0 * u0 / x;
    let u0dg = 0.5 * u0 / game;
    let u0dxx = -4.0 * u0dx / x;
    let u0dgg = -0.5 * u0dg / game;
    let u0dxg = -3.0 * u0dg / x;
    let d0dg = zion_ad * zion_ad * zion_ad;
    let d0 = game * d0dg + 21.0 * rs * rs * rs;
    let d0dx = -63.0 * rs * rs * rs / x;
    let d0dxx = 252.0 * rs * rs * rs / (x * x);
    let cor0 = 1.0 + u0 / d0;
    let cor0dx = (u0dx - u0 * d0dx / d0) / d0;
    let cor0dg = (u0dg - u0 * d0dg / d0) / d0;
    let cor0dxx = (u0dxx - (2.0 * u0dx * d0dx + u0 * d0dxx) / d0
        + 2.0 * (d0dx / d0) * (d0dx / d0))
        / d0;
    let cor0dgg = (u0dgg - 2.0 * u0dg * d0dg / d0
        + 2.0 * u0 * (d0dg / d0) * (d0dg / d0))
        / d0;
    let cor0dxg = (u0dxg - (u0dx * d0dg + u0dg * d0dx) / d0
        + 2.0 * u0 * d0dx * d0dg / (d0 * d0))
        / d0;
    let rele = (1.0 + x * x).sqrt();
    let q1 = 0.18 / zion_ad.sqrt().sqrt();
    let q2 = 0.2 + 0.37 / zion_ad.sqrt();
    let h1u = 1.0 + x * x / 5.0;
    let h1d = 1.0 + q1 * x + q2 * x * x;
    let h1 = h1u / h1d;
    let h1x = 0.4 * x / h1u - (q1 + 2.0 * q2 * x) / h1d;
    let h1dx = h1 * h1x;
    let h1dxx = h1dx * h1x
        + h1 * (0.4 / h1u - (0.4 * x / h1u) * (0.4 * x / h1u) - 2.0 * q2 / h1d
            + ((q1 + 2.0 * q2 * x) / h1d).pow2());
    let up = cdh * sqg + p01 * ctf * tx * cor0 * h1;
    let updx = p01 * ctf * tx * (cor0dx * h1 + cor0 * h1dx);
    let updg = 0.5 * cdh / sqg + p01 * ctf * (txdg * cor0 + tx * cor0dg) * h1;
    let updxx =
        p01 * ctf * tx * (cor0dxx * h1 + 2.0 * cor0dx * h1dx + cor0 * h1dxx);
    let updgg = -0.25 * cdh / (sqg * game)
        + p01 * ctf * (txdgg * cor0 + 2.0 * txdg * cor0dg + tx * cor0dgg) * h1;
    let updxg = p01
        * ctf
        * (txdg * (cor0dx * h1 + cor0 * h1dx)
            + tx * (cor0dxg * h1 + cor0dg * h1dx));
    let dn1 = p03 * sqg + p01 / rs * tx * cor1;
    let dn1dx = p01 * tx * (cor1 / XRS + cor1dx / rs);
    let dn1dg = 0.5 * p03 / sqg + p01 / rs * (txdg * cor1 + tx * cor1dg);
    let dn1dxx = p01 * tx / XRS * (2.0 * cor1dx + x * cor1dxx);
    let dn1dgg = -0.25 * p03 / (game * sqg)
        + p01 / rs * (txdgg * cor1 + 2.0 * txdg * cor1dg + tx * cor1dgg);
    let dn1dxg = p01
        * (txdg * (cor1 / XRS + cor1dx / rs)
            + tx * (cor1dg / XRS + cor1dxg / rs));
    let dn = 1.0 + dn1 / rele;
    let dndx = dn1dx / rele - x * dn1 / (rele * rele * rele);
    let dndxx = (dn1dxx
        - ((2.0 * x * dn1dx + dn1) - 3.0 * x * x * dn1 / (rele * rele))
            / (rele * rele))
        / rele;
    let dndg = dn1dg / rele;
    let dndgg = dn1dgg / rele;
    let dndxg = dn1dxg / rele - x * dn1dg / (rele * rele * rele);
    let fscr = -up / dn * game;
    let fx = (up * dndx / dn - updx) / dn;
    let fxdg = ((updg * dndx + updx * dndg + up * dndxg
        - 2.0 * up * dndx * dndg / dn)
        / dn
        - updxg)
        / dn;
    let fdx = fx * game;
    let fg = (up * dndg / dn - updg) / dn;
    let fdg = fg * game - up / dn;
    let fdxx = ((up * dndxx + 2.0 * (updx * dndx - up * dndx * dndx / dn))
        / dn
        - updxx)
        / dn
        * game;
    let fdgg = 2.0 * fg
        + game
            * ((2.0 * dndg * (updg - up * dndg / dn) + up * dndgg) / dn
                - updgg)
            / dn;
    let fdxg = fx + game * fxdg;
    let uscr = game * fdg;
    let cvscr = -game * game * fdgg;
    let pscr = (x * fdx + game * fdg) / 3.0;
    let pdtscr = -game * game * (x * fxdg + fdgg) / 3.0;
    let pdrscr = (12.0 * pscr
        + x * x * fdxx
        + 2.0 * x * game * fdxg
        + game * game * fdgg)
        / 9.0;
    let sscr = uscr - fscr;
    Ok(PcIonContrib {
        f: fscr,
        u: uscr,
        p: pscr,
        s: sscr,
        cv: cvscr,
        pdt: pdtscr,
        pdr: pdrscr,
    })
}

/// Evaluate `FSCRsol8(RS, GAMI, Zion, TPT)` for electron-ion screening in the
/// Coulomb crystal (`pc_eos.f90` lines 1034-1200).
pub fn fscr_sol8(
    rs: AutoDiffReal2VarOrder1,
    gami: AutoDiffReal2VarOrder1,
    zion: f64,
    tpt: AutoDiffReal2VarOrder1,
) -> Result<PcIonContrib, EosError> {
    const AP: [f64; 4] = [1.1866, 0.684, 17.9, 41.5];
    const ENAT: f64 = 2.7182818285;
    const TINY: f64 = 1.0e-19;
    const PX: f64 = 0.205;
    if rs.val < 0.0 {
        return Err(EosError::InvalidState {
            context: "fscr_sol8",
            detail: "RS < 0".to_string(),
        });
    }
    if rs.val < TINY {
        let z = AutoDiffReal2VarOrder1::zero();
        return Ok(PcIonContrib {
            f: z,
            u: z,
            p: z,
            s: z,
            cv: z,
            pdt: z,
            pdr: z,
        });
    }
    let xsr = 0.0140047 / rs;
    let z13 = flexmesa_math::pow(zion, 1.0 / 3.0);
    let p1 = AutoDiffReal2VarOrder1::constant(
        0.00352 * (1.0 - AP[0] / flexmesa_math::pow(zion, 0.267) + 0.27 / zion),
    );
    let p2 = AutoDiffReal2VarOrder1::constant(
        1.0 + 2.25 / z13
            * (1.0
                + AP[1] * flexmesa_math::pow5(zion)
                + 0.222 * flexmesa_math::pow6(zion))
            / (1.0 + 0.222 * flexmesa_math::pow6(zion)),
    );
    let zln = flexmesa_math::log(zion);
    let finf = (p2 / (xsr * xsr) + 1.0).sqrt() * z13 * z13 * p1;
    let finfx = -p2 / ((p2 + xsr * xsr) * xsr);
    let finfdx = finf * finfx;
    let finfdxx = finfdx * finfx
        - finfdx * (p2 + 3.0 * xsr * xsr) / ((p2 + xsr * xsr) * xsr);
    let r1 = AP[3] / (1.0 + zln);
    let r2 = 0.395 * zln + 0.347 / zion / zion.sqrt();
    let r3 = 1.0 / (1.0 + zln * zln.sqrt() * 0.01 + 0.097 / (zion * zion));
    let q1u = r1 + AP[2] * xsr * xsr;
    let q1d = 1.0 + r2 * xsr * xsr;
    let q1 = q1u / q1d;
    let q1x = 2.0 * xsr * (AP[2] / q1u - r2 / q1d);
    let q1xdx = q1x / xsr
        + 4.0
            * xsr
            * xsr
            * ((r2 / q1d) * (r2 / q1d) - (AP[2] / q1u) * (AP[2] / q1u));
    let q1dx = q1 * q1x;
    let q1dxx = q1dx * q1x + q1 * q1xdx;

    let (sup, supdx, supdg, supdxx, supdgg, supdxg) = if tpt.val < 6.0 / PX {
        let y0 = (PX * tpt) * (PX * tpt);
        let y0dx = y0 / xsr;
        let y0dg = 2.0 * y0 / gami;
        let y0dxx = AutoDiffReal2VarOrder1::zero();
        let y0dgg = y0dg / gami;
        let y0dxg = y0dg / xsr;
        let y1 = y0.exp();
        let y1dx = y1 * y0dx;
        let y1dg = y1 * y0dg;
        let y1dxx = y1 * (y0dx * y0dx + y0dxx);
        let y1dgg = y1 * (y0dg * y0dg + y0dgg);
        let y1dxg = y1 * (y0dx * y0dg + y0dxg);
        let sa = 1.0 + y1;
        let supa = sa.ln();
        let supadx = y1dx / sa;
        let supadg = y1dg / sa;
        let supadxx = (y1dxx - y1dx * y1dx / sa) / sa;
        let supadgg = (y1dgg - y1dg * y1dg / sa) / sa;
        let supadxg = (y1dxg - y1dx * y1dg / sa) / sa;
        let em2 = AutoDiffReal2VarOrder1::constant(ENAT - 2.0);
        let sb = ENAT - em2 / y1;
        let supb = sb.ln();
        let em2y1 = em2 / (y1 * y1 * sb);
        let supbdx = em2y1 * y1dx;
        let supbdg = em2y1 * y1dg;
        let supbdxx = em2y1 * (y1dxx - 2.0 * y1dx * y1dx / y1 - y1dx * supbdx);
        let supbdgg = em2y1 * (y1dgg - 2.0 * y1dg * y1dg / y1 - y1dg * supbdg);
        let supbdxg = em2y1 * (y1dxg - 2.0 * y1dx * y1dg / y1 - y1dg * supbdx);
        let sup = (supa / supb).sqrt();
        let supx = 0.5 * (supadx / supa - supbdx / supb);
        let supdx = sup * supx;
        let supg = 0.5 * (supadg / supa - supbdg / supb);
        let supdg = sup * supg;
        let supdxx = supdx * supx
            + sup
                * 0.5
                * (supadxx / supa
                    - (supadx / supa) * (supadx / supa)
                    - supbdxx / supb
                    + (supbdx / supb) * (supbdx / supb));
        let supdgg = supdg * supg
            + sup
                * 0.5
                * (supadgg / supa
                    - (supadg / supa) * (supadg / supa)
                    - supbdgg / supb
                    + (supbdg / supb) * (supbdg / supb));
        let supdxg = supdx * supg
            + sup
                * 0.5
                * ((supadxg - supadx * supadg / supa) / supa
                    - (supbdxg - supbdx * supbdg / supb) / supb);
        (sup, supdx, supdg, supdxx, supdgg, supdxg)
    } else {
        let sup = PX * tpt;
        let supdx = 0.5 * PX * tpt / xsr;
        let supdg = PX * tpt / gami;
        let supdxx = -0.5 * supdx / xsr;
        let supdgg = AutoDiffReal2VarOrder1::zero();
        let supdxg = supdx / gami;
        (sup, supdx, supdg, supdxx, supdgg, supdxg)
    };

    let gr3 = (gami / sup).powf(r3);
    let gr3x = -r3 * supdx / sup;
    let gr3dx = gr3 * gr3x;
    let gr3dxx = gr3dx * gr3x
        - r3 * gr3 * (supdxx / sup - (supdx / sup) * (supdx / sup));
    let gr3g = r3 * (1.0 / gami - supdg / sup);
    let gr3dg = gr3 * gr3g;
    let gr3dgg = gr3dg * gr3g
        + gr3
            * r3
            * ((supdg / sup) * (supdg / sup)
                - supdgg / sup
                - 1.0 / (gami * gami));
    let gr3dxg =
        gr3dg * gr3x + gr3 * r3 * (supdx * supdg / (sup * sup) - supdxg / sup);

    let w = 1.0 + q1 / gr3;
    let wdx = q1dx / gr3 - q1 * gr3dx / (gr3 * gr3 * gr3);
    let wdg = -q1 * gr3dg / (gr3 * gr3);
    let wdxx = q1dxx / gr3
        - (2.0 * q1dx * gr3dx + q1 * (gr3dxx - 2.0 * gr3dx * gr3dx / gr3))
            / (gr3 * gr3);
    let wdgg = q1 * (2.0 * gr3dg * gr3dg / gr3 - gr3dgg) / (gr3 * gr3);
    let wdxg = -(q1dx * gr3dg + q1 * (gr3dxg - 2.0 * gr3dx * gr3dg / gr3))
        / (gr3 * gr3);

    let fscr = -gami * finf * w;
    let fdx = -gami * (finfdx * w + finf * wdx);
    let fdxx = -gami * (finfdxx * w + 2.0 * finfdx * wdx + finf * wdxx);
    let fdg = -finf * w - gami * finf * wdg;
    let fdgg = -2.0 * finf * wdg - gami * finf * wdgg;
    let fdxg = -finfdx * w - finf * wdx - gami * (finfdx * wdg + finf * wdxg);
    let s_scr = -gami * gami * finf * wdg;
    let uscr = s_scr + fscr;
    let cvscr = -gami * gami * fdgg;
    let pscr = (xsr * fdx + gami * fdg) / 3.0;
    let pdtscr = gami * gami * (xsr * finf * (finfx * wdg + wdxg) - fdgg) / 3.0;
    let pdrscr = (12.0 * pscr
        + xsr * xsr * fdxx
        + 2.0 * xsr * gami * fdxg
        + gami * gami * fdgg)
        / 9.0;
    Ok(PcIonContrib {
        f: fscr,
        u: uscr,
        p: pscr,
        s: s_scr,
        cv: cvscr,
        pdt: pdtscr,
        pdr: pdrscr,
    })
}

/// Evaluate `CORMIX` for the liquid linear-mixing-rule correction (`pc_eos.f90`
/// lines 1414-1476).
#[must_use]
pub fn cormix(
    rs: AutoDiffReal2VarOrder1,
    game: AutoDiffReal2VarOrder1,
    zmean: f64,
    z2mean: f64,
    z52: f64,
    z53: f64,
    z321: f64,
) -> PcIonContrib {
    const TINY: f64 = 1.0e-9;
    let d = z2mean / (zmean * zmean);
    if (d - 1.0).abs() < TINY {
        let z = AutoDiffReal2VarOrder1::zero();
        return PcIonContrib {
            f: z,
            u: z,
            p: z,
            s: z,
            cv: z,
            pdt: z,
            pdr: z,
        };
    }
    let gami_mean = game * z53;
    let dif0 = if rs.val < TINY {
        z52 - (z2mean * z2mean * z2mean / zmean).sqrt()
    } else {
        let zm = z2mean + zmean;
        z321 - (zm * zm * zm / zmean).sqrt()
    };
    let dif_r = dif0 / z52;
    let dif_fdh = dif0 * game * (game / 3.0).sqrt();
    let p3 = flexmesa_math::pow(d, -0.2);
    let d0 = (2.6 * dif_r + 14.0 * dif_r * dif_r * dif_r) / (1.0 - p3);
    let gp = d0 * gami_mean.powf(p3);
    let fmix0 = dif_fdh / (1.0 + gp);
    let q = d * d * 0.0117;
    let r = 1.5 / p3 - 1.0;
    let gq = q * gp;
    let fmix = fmix0 / (1.0 + gq).powf(r);
    let g = 1.5 - p3 * gp / (1.0 + gp) - r * p3 * gq / (1.0 + gq);
    let umix = fmix * g;
    let pmix = umix / 3.0;
    let gdg = -p3
        * p3
        * (gp / ((1.0 + gp) * (1.0 + gp)) + r * gq / ((1.0 + gq) * (1.0 + gq)));
    let udg = umix * g + fmix * gdg;
    let cvmix = umix - udg;
    let pdtmix = pmix - udg / 3.0;
    let pdrmix = pmix + udg / 9.0;
    let smix = umix - fmix;
    PcIonContrib {
        f: fmix,
        u: umix,
        p: pmix,
        s: smix,
        cv: cvmix,
        pdt: pdtmix,
        pdr: pdrmix,
    }
}
