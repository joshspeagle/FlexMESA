// SPDX-License-Identifier: LGPL-3.0-or-later
//! Single-species non-ideal plasma combiner (`EOSFI8`) for the Potekhin-Chabrier
//! (`PC`) equation of state (`eos/private/pc_eos.f90` lines 484-627).

use flexmesa_autodiff::AutoDiffReal2VarOrder1;

use crate::pc::coulomb_ion::{anharm8, fharm12, fition9, PcIonContrib};
use crate::pc::coulomb_screen::{fscr_liq8, fscr_sol8};
use crate::pc::excor::excor7;
use crate::EosError;

/// Evaluate `EOSFI8` (`pc_eos.f90` lines 484-627), returning `(FC1..PDR1, FC2..PDR2)`
/// for a single ion species in liquid (`liqsol == 0`) or solid (`liqsol == 1`) phase.
pub fn eosfi8(
    liqsol: i32,
    cmi: f64,
    zion: f64,
    rs: AutoDiffReal2VarOrder1,
    gami: AutoDiffReal2VarOrder1,
) -> Result<(PcIonContrib, PcIonContrib), EosError> {
    const AUM: f64 = 1822.888;
    if (liqsol != 0 && liqsol != 1)
        || cmi <= 0.1
        || zion <= 0.1
        || rs.val <= 0.0
        || gami.val <= 0.0
    {
        return Err(EosError::InvalidState {
            context: "eosfi8",
            detail: "invalid input".to_string(),
        });
    }
    let game = gami / zion.powf(5.0 / 3.0);
    let xc = excor7(rs, game);
    let cotpt = (3.0 / AUM / cmi).sqrt() / zion.powf(7.0 / 6.0);
    let tpt = gami / rs.sqrt() * cotpt;
    let fid_ion = (tpt * tpt / gami).ln() * 1.5 - 1.323515;

    let (
        fion,
        uion,
        pion,
        cvii,
        pdtii,
        pdrii,
        fitot,
        uitot,
        pitot,
        cvitot,
        scitot,
        pdti,
        pdri,
    ) = if liqsol == 0 {
        let ii = fition9(gami);
        let fitot = ii.f + fid_ion;
        let uitot = ii.u + 1.5;
        let pitot = ii.p + 1.0;
        let cvitot = ii.cv + 1.5;
        let scitot = uitot - fitot;
        let pdti = ii.pdt + 1.0;
        let pdri = ii.pdr + 1.0;
        (
            ii.f, ii.u, ii.p, ii.cv, ii.pdt, ii.pdr, fitot, uitot, pitot,
            cvitot, scitot, pdti, pdri,
        )
    } else {
        let harm = fharm12(gami, tpt);
        let ah = anharm8(gami, tpt);
        let fitot = harm.f + ah.f;
        let fion = fitot - fid_ion;
        let uitot = harm.u + ah.u;
        let uion = uitot - 1.5;
        let pitot = harm.p + ah.p;
        let pion = pitot - 1.0;
        let pdti = harm.pdt + ah.pdt;
        let pdri = harm.pdr + ah.pdr;
        let pdtii = pdti - 1.0;
        let pdrii = pdri - 1.0;
        let cvitot = harm.cv + ah.cv;
        let scitot = harm.s + ah.u - ah.f;
        let cvii = cvitot - 1.5;
        (
            fion, uion, pion, cvii, pdtii, pdrii, fitot, uitot, pitot, cvitot,
            scitot, pdti, pdri,
        )
    };

    let scr = if liqsol == 1 {
        fscr_sol8(rs, gami, zion, tpt)?
    } else {
        fscr_liq8(rs, game, zion)?
    };

    let fc0 = scr.f + xc.f * zion;
    let uc0 = scr.u + xc.u * zion;
    let pc0 = scr.p + xc.p * zion;
    let sc0 = scr.s + xc.s * zion;
    let cv0 = scr.cv + xc.cv * zion;
    let pdt0 = scr.pdt + xc.pdt * zion;
    let pdr0 = scr.pdr + xc.pdr * zion;

    let c1 = PcIonContrib {
        f: fion + fc0,
        u: uion + uc0,
        p: pion + pc0,
        s: (uion - fion) + sc0,
        cv: cvii + cv0,
        pdt: pdtii + pdt0,
        pdr: pdrii + pdr0,
    };
    let c2 = PcIonContrib {
        f: fitot + fc0,
        u: uitot + uc0,
        p: pitot + pc0,
        s: scitot + sc0,
        cv: cvitot + cv0,
        pdt: pdti + pdt0,
        pdr: pdri + pdr0,
    };
    Ok((c1, c2))
}
