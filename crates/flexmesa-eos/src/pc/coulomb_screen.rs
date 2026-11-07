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
    let sqg = game.sqrt();
    let sqz1 = (1.0 + zion).sqrt();
    let sqz = zion.sqrt();
    let cdh0 = zion / 1.73205;
    let cdh = cdh0 * (sqz1 * sqz1 * sqz1 - sqz * sqz * sqz - 1.0);
    let zln = zion.ln();
    let z13 = (zln / 3.0).exp();
    let x = rs.recip() * XRS;
    let ctf = zion * zion * 0.2513 * (z13 - 1.0 + 0.2 / z13.sqrt());
    let p01 = 1.11 * (0.475 * zln).exp();
    let p03 = 0.2 + 0.078 * zln * zln;
    let ptx = 1.16 + 0.08 * zln;
    let tx = game.powf(ptx);
    let txdg = tx * ptx / game;
    let txdgg = txdg * (ptx - 1.0) / game;
    let ty1 = (game * 2.0 + 1.0e-3 * zion * zion).recip();
    let ty1dg = -ty1 * ty1 * 2.0;
    let ty1dgg = -ty1 * ty1dg * 4.0;
    let rs2 = rs * rs;
    let rs3 = rs2 * rs;
    let ty2 = rs2 * 6.0 + 1.0;
    let ty2dx = -rs2 * 12.0 / x;
    let ty2dxx = -ty2dx * 3.0 / x;
    let ty = rs3 / ty2 * (ty1 + 1.0);
    let tyx = x.recip() * 3.0 + ty2dx / ty2;
    let tydx = -ty * tyx;
    let tydg = rs3 * ty1dg / ty2;
    let p1 = (zion - 1.0) / 9.0;
    let cor1 = ty * p1 + 1.0;
    let cor1dx = tydx * p1;
    let cor1dg = tydg * p1;
    let cor1dxx = (ty
        * ((x * x).recip() * 3.0 + (ty2dx / ty2).pow2() - ty2dxx / ty2)
        - tydx * tyx)
        * p1;
    let cor1dgg = rs3 * ty1dgg / ty2 * p1;
    let cor1dxg = -tydg * tyx * p1;
    let u0 = (game / zion).sqrt() * rs3 * 0.78;
    let u0dx = -u0 * 3.0 / x;
    let u0dg = u0 * 0.5 / game;
    let u0dxx = -u0dx * 4.0 / x;
    let u0dgg = -u0dg * 0.5 / game;
    let u0dxg = -u0dg * 3.0 / x;
    let d0dg = zion * zion * zion;
    let d0 = game * d0dg + rs3 * 21.0;
    let d0dx = -rs3 * 63.0 / x;
    let d0dxx = rs3 * 252.0 / (x * x);
    let cor0 = u0 / d0 + 1.0;
    let cor0dx = (u0dx - u0 * d0dx / d0) / d0;
    let cor0dg = (u0dg - u0 * d0dg / d0) / d0;
    let cor0dxx = (u0dxx - (u0dx * d0dx * 2.0 + u0 * d0dxx) / d0
        + (d0dx / d0).pow2() * 2.0)
        / d0;
    let cor0dgg = (u0dgg - u0dg * (2.0 * d0dg) / d0
        + u0 * 2.0 * (d0dg / d0) * (d0dg / d0))
        / d0;
    let cor0dxg = (u0dxg - (u0dx * d0dg + u0dg * d0dx) / d0
        + u0 * d0dx * (2.0 * d0dg) / (d0 * d0))
        / d0;
    let rele = (x * x + 1.0).sqrt();
    let q1 = 0.18 / zion.sqrt().sqrt();
    let q2 = 0.2 + 0.37 / zion.sqrt();
    let h1u = x * x / 5.0 + 1.0;
    let h1d = x * q1 + x * x * q2 + 1.0;
    let h1 = h1u / h1d;
    let h1x = x * 0.4 / h1u - (x * (2.0 * q2) + q1) / h1d;
    let h1dx = h1 * h1x;
    let h1dxx = h1dx * h1x
        + h1 * (h1u.recip() * 0.4
            - (x * 0.4 / h1u).pow2()
            - h1d.recip() * (2.0 * q2)
            + ((x * (2.0 * q2) + q1) / h1d).pow2());
    let up = sqg * cdh + tx * cor0 * h1 * (p01 * ctf);
    let updx = tx * (cor0dx * h1 + cor0 * h1dx) * (p01 * ctf);
    let updg = sqg.recip() * (0.5 * cdh)
        + (txdg * cor0 + tx * cor0dg) * h1 * (p01 * ctf);
    let updxx =
        tx * (cor0dxx * h1 + cor0dx * h1dx * 2.0 + cor0 * h1dxx) * (p01 * ctf);
    let updgg = -(sqg * game).recip() * (0.25 * cdh)
        + (txdgg * cor0 + txdg * cor0dg * 2.0 + tx * cor0dgg)
            * h1
            * (p01 * ctf);
    let updxg = (txdg * (cor0dx * h1 + cor0 * h1dx)
        + tx * (cor0dxg * h1 + cor0dg * h1dx))
        * (p01 * ctf);
    let dn1 = sqg * p03 + tx * cor1 * p01 / rs;
    let dn1dx = tx * (cor1 / XRS + cor1dx / rs) * p01;
    let dn1dg =
        sqg.recip() * (0.5 * p03) + (txdg * cor1 + tx * cor1dg) * p01 / rs;
    let dn1dxx = tx * (cor1dx * 2.0 + x * cor1dxx) * (p01 / XRS);
    let dn1dgg = -(game * sqg).recip() * (0.25 * p03)
        + (txdgg * cor1 + txdg * cor1dg * 2.0 + tx * cor1dgg) * p01 / rs;
    let dn1dxg = (txdg * (cor1 / XRS + cor1dx / rs)
        + tx * (cor1dg / XRS + cor1dxg / rs))
        * p01;
    let rele2 = rele * rele;
    let rele3 = rele2 * rele;
    let dn = dn1 / rele + 1.0;
    let dndx = dn1dx / rele - x * dn1 / rele3;
    let dndxx = (dn1dxx
        - (x * dn1dx * 2.0 + dn1 - x * x * dn1 * 3.0 / rele2) / rele2)
        / rele;
    let dndg = dn1dg / rele;
    let dndgg = dn1dgg / rele;
    let dndxg = dn1dxg / rele - x * dn1dg / rele3;
    let fscr = -up / dn * game;
    let fx = (up * dndx / dn - updx) / dn;
    let fxdg = ((updg * dndx + updx * dndg + up * dndxg
        - up * dndx * dndg * 2.0 / dn)
        / dn
        - updxg)
        / dn;
    let fdx = fx * game;
    let fg = (up * dndg / dn - updg) / dn;
    let fdg = fg * game - up / dn;
    let fdxx = ((up * dndxx + (updx * dndx - up * dndx * dndx / dn) * 2.0)
        / dn
        - updxx)
        / dn
        * game;
    let fdgg = fg * 2.0
        + game
            * ((dndg * (updg - up * dndg / dn) * 2.0 + up * dndgg) / dn
                - updgg)
            / dn;
    let fdxg = fx + game * fxdg;
    let uscr = game * fdg;
    let cvscr = -game * game * fdgg;
    let pscr = (x * fdx + game * fdg) / 3.0;
    let pdtscr = -game * game * (x * fxdg + fdgg) / 3.0;
    let pdrscr = (pscr * 12.0
        + x * x * fdxx
        + x * game * fdxg * 2.0
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
    let xsr = rs.recip() * 0.0140047;
    let z13 = zion.powf(1.0 / 3.0);
    let p1 = 0.00352 * (1.0 - AP[0] / zion.powf(0.267) + 0.27 / zion);
    let z5 = zion.powi(5);
    let z6 = z5 * zion;
    let p2 =
        1.0 + 2.25 / z13 * (1.0 + AP[1] * z5 + 0.222 * z6) / (1.0 + 0.222 * z6);
    let zln = zion.ln();
    let xsr2 = xsr * xsr;
    let finf = (xsr2.recip() * p2 + 1.0).sqrt() * (z13 * z13 * p1);
    let finfx = -((xsr2 + p2) * xsr).recip() * p2;
    let finfdx = finf * finfx;
    let finfdxx =
        finfdx * finfx - finfdx * (xsr2 * 3.0 + p2) / ((xsr2 + p2) * xsr);
    let r1 = AP[3] / (1.0 + zln);
    let r2 = 0.395 * zln + 0.347 / zion / zion.sqrt();
    let r3 = 1.0 / (1.0 + zln * zln.sqrt() * 0.01 + 0.097 / (zion * zion));
    let q1u = xsr2 * AP[2] + r1;
    let q1d = xsr2 * r2 + 1.0;
    let q1 = q1u / q1d;
    let q1x = xsr * 2.0 * (q1u.recip() * AP[2] - q1d.recip() * r2);
    let q1xdx = q1x / xsr
        + xsr2
            * 4.0
            * ((q1d.recip() * r2).pow2() - (q1u.recip() * AP[2]).pow2());
    let q1dx = q1 * q1x;
    let q1dxx = q1dx * q1x + q1 * q1xdx;

    let (sup, supdx, supdg, supdxx, supdgg, supdxg) = if tpt.val < 6.0 / PX {
        let px_tpt = tpt * PX;
        let y0 = px_tpt * px_tpt;
        let y0dx = y0 / xsr;
        let y0dg = y0 * 2.0 / gami;
        let y0dxx = AutoDiffReal2VarOrder1::zero();
        let y0dgg = y0dg / gami;
        let y0dxg = y0dg / xsr;
        let y1 = y0.exp();
        let y1dx = y1 * y0dx;
        let y1dg = y1 * y0dg;
        let y1dxx = y1 * (y0dx * y0dx + y0dxx);
        let y1dgg = y1 * (y0dg * y0dg + y0dgg);
        let y1dxg = y1 * (y0dx * y0dg + y0dxg);
        let sa = y1 + 1.0;
        let supa = sa.ln();
        let supadx = y1dx / sa;
        let supadg = y1dg / sa;
        let supadxx = (y1dxx - y1dx * y1dx / sa) / sa;
        let supadgg = (y1dgg - y1dg * y1dg / sa) / sa;
        let supadxg = (y1dxg - y1dx * y1dg / sa) / sa;
        let em2 = ENAT - 2.0;
        let sb = -y1.recip() * em2 + ENAT;
        let supb = sb.ln();
        let em2y1 = (y1 * y1 * sb).recip() * em2;
        let supbdx = em2y1 * y1dx;
        let supbdg = em2y1 * y1dg;
        let supbdxx = em2y1 * (y1dxx - y1dx * y1dx * 2.0 / y1 - y1dx * supbdx);
        let supbdgg = em2y1 * (y1dgg - y1dg * y1dg * 2.0 / y1 - y1dg * supbdg);
        let supbdxg = em2y1 * (y1dxg - y1dx * y1dg * 2.0 / y1 - y1dg * supbdx);
        let sup = (supa / supb).sqrt();
        let supx = (supadx / supa - supbdx / supb) * 0.5;
        let supdx = sup * supx;
        let supg = (supadg / supa - supbdg / supb) * 0.5;
        let supdg = sup * supg;
        let supdxx = supdx * supx
            + sup
                * 0.5
                * (supadxx / supa - (supadx / supa).pow2() - supbdxx / supb
                    + (supbdx / supb).pow2());
        let supdgg = supdg * supg
            + sup
                * 0.5
                * (supadgg / supa - (supadg / supa).pow2() - supbdgg / supb
                    + (supbdg / supb).pow2());
        let supdxg = supdx * supg
            + sup
                * 0.5
                * ((supadxg - supadx * supadg / supa) / supa
                    - (supbdxg - supbdx * supbdg / supb) / supb);
        (sup, supdx, supdg, supdxx, supdgg, supdxg)
    } else {
        let sup = tpt * PX;
        let supdx = tpt * (0.5 * PX) / xsr;
        let supdg = tpt * PX / gami;
        let supdxx = -supdx * 0.5 / xsr;
        let supdgg = AutoDiffReal2VarOrder1::zero();
        let supdxg = supdx / gami;
        (sup, supdx, supdg, supdxx, supdgg, supdxg)
    };

    let gr3 = (gami / sup).powf(r3);
    let gr3x = -supdx / sup * r3;
    let gr3dx = gr3 * gr3x;
    let gr3dxx =
        gr3dx * gr3x - gr3 * (supdxx / sup - (supdx / sup).pow2()) * r3;
    let gr3g = (gami.recip() - supdg / sup) * r3;
    let gr3dg = gr3 * gr3g;
    let gr3dgg = gr3dg * gr3g
        + gr3
            * ((supdg / sup).pow2() - supdgg / sup - (gami * gami).recip())
            * r3;
    let gr3dxg =
        gr3dg * gr3x + gr3 * (supdx * supdg / (sup * sup) - supdxg / sup) * r3;

    let gr3_2 = gr3 * gr3;
    let gr3_3 = gr3_2 * gr3;
    let w = q1 / gr3 + 1.0;
    let wdx = q1dx / gr3 - q1 * gr3dx / gr3_3;
    let wdg = -q1 * gr3dg / gr3_2;
    let wdxx = q1dxx / gr3
        - (q1dx * gr3dx * 2.0 + q1 * (gr3dxx - gr3dx * gr3dx * 2.0 / gr3))
            / gr3_2;
    let wdgg = q1 * (gr3dg * gr3dg * 2.0 / gr3 - gr3dgg) / gr3_2;
    let wdxg =
        -(q1dx * gr3dg + q1 * (gr3dxg - gr3dx * gr3dg * 2.0 / gr3)) / gr3_2;

    let fscr = -gami * finf * w;
    let fdx = -gami * (finfdx * w + finf * wdx);
    let fdxx = -gami * (finfdxx * w + finfdx * wdx * 2.0 + finf * wdxx);
    let fdg = -finf * w - gami * finf * wdg;
    let fdgg = -finf * wdg * 2.0 - gami * finf * wdgg;
    let fdxg = -finfdx * w - finf * wdx - gami * (finfdx * wdg + finf * wdxg);
    let s_scr = -gami * gami * finf * wdg;
    let uscr = s_scr + fscr;
    let cvscr = -gami * gami * fdgg;
    let pscr = (xsr * fdx + gami * fdg) / 3.0;
    let pdtscr = gami * gami * (xsr * finf * (finfx * wdg + wdxg) - fdgg) / 3.0;
    let pdrscr = (pscr * 12.0
        + xsr * xsr * fdxx
        + xsr * gami * fdxg * 2.0
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
    let dif_fdh = game * (game / 3.0).sqrt() * dif0;
    let p3 = d.powf(-0.2);
    let d0 = (2.6 * dif_r + 14.0 * dif_r * dif_r * dif_r) / (1.0 - p3);
    let gp = gami_mean.powf(p3) * d0;
    let fmix0 = dif_fdh / (gp + 1.0);
    let q = d * d * 0.0117;
    let r = 1.5 / p3 - 1.0;
    let gq = gp * q;
    let fmix = fmix0 / (gq + 1.0).powf(r);
    let g = -gp * p3 / (gp + 1.0) - gq * (r * p3) / (gq + 1.0) + 1.5;
    let umix = fmix * g;
    let pmix = umix / 3.0;
    let gdg =
        -(gp / (gp + 1.0).pow2() + gq * r / (gq + 1.0).pow2()) * (p3 * p3);
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
