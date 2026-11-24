//! Ideal relativistic Fermi-Dirac electron gas (`ELECT11`, `ELECT11a`,
//! `ELECT11b`, `SOMMERF`, `SUBFERMJ`) for the Potekhin-Chabrier (`PC`)
//! equation of state.
//!
//! Reference: Potekhin & Chabrier (2010), Contrib. Plasma Phys., 50, 82;
//! upstream Fortran MESA `eos/private/pc_eos.f90`.

use flexmesa_autodiff::AutoDiffReal2VarOrder1;
use flexmesa_const::PI;

use crate::pc::blin::{blin9, Blin9Output, BlinSeries};
use crate::pc::ferinv::fermi10;

/// Output of `ELECT11`: ideal relativistic electron gas thermodynamics.
#[derive(Debug, Clone, Copy)]
pub struct Elect11Output {
    /// Electron number density `n_e` in atomic units (`a_0^{-3}`).
    pub dens: AutoDiffReal2VarOrder1,
    /// Ideal electron free energy `F_{e,id} / (N_e k_B T)`.
    pub fe_id: AutoDiffReal2VarOrder1,
    /// Ideal electron pressure `P_{e,id} / (n_e k_B T)`.
    pub pe_id: AutoDiffReal2VarOrder1,
    /// Ideal electron internal energy `U_{e,id} / (N_e k_B T)`.
    pub ue_id: AutoDiffReal2VarOrder1,
    /// Ideal electron entropy `S_{e,id} / (N_e k_B)`.
    pub se_id: AutoDiffReal2VarOrder1,
    /// Ideal electron heat capacity `C_{V,e} / (N_e k_B)`.
    pub cve: AutoDiffReal2VarOrder1,
    /// Logarithmic temperature derivative `\chi_{T,e} = (\partial \ln P_e / \partial \ln T)_V`.
    pub chite: AutoDiffReal2VarOrder1,
    /// Logarithmic density derivative `\chi_{\rho,e} = (\partial \ln P_e / \partial \ln n_e)_T`.
    pub chire: AutoDiffReal2VarOrder1,
    /// `(\partial \ln n_e / \partial \chi)_T`.
    pub dln_dh: AutoDiffReal2VarOrder1,
    /// `(\partial \ln n_e / \partial \ln T)_\chi`.
    pub dln_dt: AutoDiffReal2VarOrder1,
    /// `(\partial^2 \ln n_e / \partial \chi^2)_T`.
    pub dln_dhh: AutoDiffReal2VarOrder1,
    /// `(\partial^2 \ln n_e / \partial \ln T^2)_\chi`.
    pub dln_dtt: AutoDiffReal2VarOrder1,
    /// `\partial^2 \ln n_e / (\partial \chi \partial \ln T)`.
    pub dln_dht: AutoDiffReal2VarOrder1,
}

/// Evaluate `ELECT11(TEMP, CHI)` across non-degenerate, moderately
/// degenerate, and strongly degenerate regimes (`pc_eos.f90` lines 1479-1555).
///
/// # Arguments
/// * `temp` - Temperature in atomic units (`2 Ry = 3.157746e5 K`).
/// * `chi` - Dimensionless electron chemical potential `\mu_e / (k_B T)`.
#[must_use]
pub fn elect11(
    temp: AutoDiffReal2VarOrder1,
    chi: AutoDiffReal2VarOrder1,
) -> Elect11Output {
    let chi1 = AutoDiffReal2VarOrder1::constant(0.6_f64);
    let chi2 = AutoDiffReal2VarOrder1::constant(28.0_f64);
    let xmax = AutoDiffReal2VarOrder1::constant(20.0_f64);
    let dchi1 = AutoDiffReal2VarOrder1::constant(0.1_f64);
    let dchi2 = chi2 - chi1 - dchi1;
    let xscal2 = xmax / dchi2;
    let x2 = (chi - chi2) * xscal2;
    if x2.val < -xmax.val {
        elect11a(temp, chi)
    } else if x2.val > xmax.val {
        elect11b(temp, chi)
    } else {
        let a = elect11a(temp, chi);
        let b = elect11b(temp, chi);
        let (fp, fm) = fermi10(x2, xmax.val);
        Elect11Output {
            dens: a.dens * fp + b.dens * fm,
            fe_id: a.fe_id * fp + b.fe_id * fm,
            pe_id: a.pe_id * fp + b.pe_id * fm,
            ue_id: a.ue_id * fp + b.ue_id * fm,
            se_id: a.se_id * fp + b.se_id * fm,
            cve: a.cve * fp + b.cve * fm,
            chite: a.chite * fp + b.chite * fm,
            chire: a.chire * fp + b.chire * fm,
            dln_dh: a.dln_dh * fp + b.dln_dh * fm,
            dln_dt: a.dln_dt * fp + b.dln_dt * fm,
            dln_dhh: a.dln_dhh * fp + b.dln_dhh * fm,
            dln_dtt: a.dln_dtt * fp + b.dln_dtt * fm,
            dln_dht: a.dln_dht * fp + b.dln_dht * fm,
        }
    }
}

/// Evaluate `ELECT11a(TEMP, CHI)` via `BLIN9` (`pc_eos.f90` lines 1557-1620).
#[must_use]
pub fn elect11a(
    temp: AutoDiffReal2VarOrder1,
    chi: AutoDiffReal2VarOrder1,
) -> Elect11Output {
    let bohr = AutoDiffReal2VarOrder1::constant(137.036_f64);
    let pi2 = AutoDiffReal2VarOrder1::constant(PI * PI);
    let bohr2 = bohr * bohr;
    let bohr3 = bohr2 * bohr;
    let temr = temp / bohr2;
    let b = blin9(temr, chi);
    let tpi = temr * (2.0 * temr).sqrt() / pi2;
    let denr = tpi * (b.w1.w * temr + b.w0.w);
    let pr = temr * tpi / 3.0 * (b.w2.w * temr + 2.0 * b.w1.w);
    let u = temr * tpi * (b.w2.w * temr + b.w1.w);
    let pe_id = pr / (denr * temr);
    let ue_id = u / (denr * temr);
    let fe_id = chi - pe_id;
    let dens = denr * bohr3;
    let se_id = ue_id - fe_id;
    let dndt =
        tpi * (1.5 * b.w0.w / temr + 2.5 * b.w1.w + b.w0.wdt + temr * b.w1.wdt);
    let dpdt = tpi / 3.0
        * (5.0 * b.w1.w
            + 2.0 * temr * b.w1.wdt
            + 3.5 * temr * b.w2.w
            + temr * temr * b.w2.wdt);
    let dudt = tpi
        * (2.5 * b.w1.w
            + temr * b.w1.wdt
            + 3.5 * temr * b.w2.w
            + temr * temr * b.w2.wdt);
    let dndh = tpi * (b.w0.wdx + temr * b.w1.wdx);
    let dndhh = tpi * (b.w0.wdxx + temr * b.w1.wdxx);
    let dndtt = tpi
        * (0.75 * b.w0.w / temr.pow2()
            + 3.0 * b.w0.wdt / temr
            + b.w0.wdtt
            + 3.75 * b.w1.w / temr
            + 5.0 * b.w1.wdt
            + temr * b.w1.wdtt);
    let dndht = tpi
        * (1.5 * b.w0.wdx / temr
            + b.w0.wdxt
            + 2.5 * b.w1.wdx
            + temr * b.w1.wdxt);
    let dln_dh = dndh / denr;
    let dln_dt = dndt * temr / denr;
    let dln_dhh = dndhh / denr - dln_dh.pow2();
    let dln_dtt = temr.pow2() / denr * dndtt + dln_dt - dln_dt.pow2();
    let dln_dht = temr / denr * (dndht - dndt * dln_dh);
    let dpdh = tpi / 3.0 * temr * (2.0 * b.w1.wdx + temr * b.w2.wdx);
    let dudh = tpi * temr * (b.w1.wdx + temr * b.w2.wdx);
    let cve = (dudt - dudh * dndt / dndh) / denr;
    let chite = temr / pr * (dpdt - dpdh * dndt / dndh);
    let chire = denr / pr * dpdh / dndh;
    Elect11Output {
        dens,
        fe_id,
        pe_id,
        ue_id,
        se_id,
        cve,
        chite,
        chire,
        dln_dh,
        dln_dt,
        dln_dhh,
        dln_dtt,
        dln_dht,
    }
}

/// Evaluate `ELECT11b(TEMP, CHI)` via the Sommerfeld asymptotic expansion
/// (`pc_eos.f90` lines 1622-1705).
#[must_use]
pub fn elect11b(
    temp: AutoDiffReal2VarOrder1,
    chi: AutoDiffReal2VarOrder1,
) -> Elect11Output {
    let bohr = AutoDiffReal2VarOrder1::constant(137.036_f64);
    let pi2 = AutoDiffReal2VarOrder1::constant(PI * PI);
    let bohr2 = bohr * bohr;
    let bohr3 = bohr2 * bohr;
    let temr = temp / bohr2;
    let ef0 = chi * temr;
    let delta_ef = pi2 * temr * temr / 6.0 * (1.0 + 2.0 * ef0 * (2.0 + ef0))
        / (ef0 * (1.0 + ef0) * (2.0 + ef0));
    let ef = ef0 + delta_ef;
    let g = 1.0 + ef;
    let (pf, f0, df, p0, del_p, mut cve) = if ef.val > 1.0e-5 {
        let pf = (g * g - 1.0).sqrt();
        let f0 = (pf * (1.0 + 2.0 * pf * pf) * g
            - pf * pf * pf / 0.375
            - (pf + g).ln())
            / 8.0
            / pi2;
        let df = -temr * temr * pf * g / 6.0;
        let p0 = (pf * g * (pf * pf / 1.5 - 1.0) + (pf + g).ln()) / 8.0 / pi2;
        let del_p = temr * temr * pf * (pf * pf + 2.0) / g / 18.0;
        let cve = pi2 * temr * g / (pf * pf);
        (pf, f0, df, p0, del_p, cve)
    } else {
        let pf = (2.0 * ef).sqrt();
        let f0 = pf.pow5() * 0.1 / pi2;
        let df = -temr * temr * pf / 6.0;
        let p0 = f0 / 1.5;
        let del_p = temr * temr * pf / 9.0;
        let cve = pi2 * temr / ef / 2.0;
        (pf, f0, df, p0, del_p, cve)
    };
    let f = f0 + df;
    let p = p0 + del_p;
    let s = -2.0 * df;
    let u = f + s;
    let chire = pf.pow5() / (9.0 * pi2 * p * g);
    let mut chite = 2.0 * del_p / p;
    let denr = pf * pf * pf / 3.0 / pi2;
    let dens = denr * bohr3;

    let b = sommerf(temr, chi);
    let tpi = temr * (2.0 * temr).sqrt() / pi2;
    let dndh = tpi * (b.w0.wdx + temr * b.w1.wdx);
    let dndt =
        tpi * (1.5 * b.w0.w / temr + 2.5 * b.w1.w + b.w0.wdt + temr * b.w1.wdt);
    let dndhh = tpi * (b.w0.wdxx + temr * b.w1.wdxx);
    let dndtt = tpi
        * (0.75 * b.w0.w / temr.pow2()
            + 3.0 * b.w0.wdt / temr
            + b.w0.wdtt
            + 3.75 * b.w1.w / temr
            + 5.0 * b.w1.wdt
            + temr * b.w1.wdtt);
    let dndht = tpi
        * (1.5 * b.w0.wdx / temr
            + b.w0.wdxt
            + 2.5 * b.w1.wdx
            + temr * b.w1.wdxt);
    let dln_dh = dndh / denr;
    let dln_dt = dndt * temr / denr;
    let dln_dhh = dndhh / denr - dln_dh.pow2();
    let dln_dtt = temr.pow2() / denr * dndtt + dln_dt - dln_dt.pow2();
    let dln_dht = temr / denr * (dndht - dndt * dln_dh);
    let dt = denr * temr;
    let pe_id = p / dt;
    let ue_id = u / dt;
    let fe_id = f / dt;
    let mut se_id = s / dt;
    // Empirical corrections of 16.02.09 (`pc_eos.f90` lines 1699-1703):
    let d1 = delta_ef / ef;
    let d2 = d1 * (4.0 - 2.0 * (pf / g));
    cve = cve / (1.0 + d2);
    se_id = se_id / (1.0 + d1);
    chite = chite / (1.0 + d2);
    Elect11Output {
        dens,
        fe_id,
        pe_id,
        ue_id,
        se_id,
        cve,
        chite,
        chire,
        dln_dh,
        dln_dt,
        dln_dhh,
        dln_dtt,
        dln_dht,
    }
}

/// Evaluate `SOMMERF` (`pc_eos.f90` lines 1707-1770) and `SUBFERMJ`
/// (`pc_eos.f90` lines 1772-1816).
#[must_use]
pub fn sommerf(
    temr: AutoDiffReal2VarOrder1,
    chi: AutoDiffReal2VarOrder1,
) -> Blin9Output {
    let cmu1 = chi * temr;
    let cmu = cmu1 + 1.0;
    let x0 = (cmu1 * (cmu1 + 2.0)).sqrt();
    let x3 = x0.pow3();
    let x5 = x0.pow5();
    let (cj00, cj10, cj20) = if x0.val < 1.0e-4 {
        (x3 / 3.0, 0.1 * x5, x0.pow7() / 28.0)
    } else {
        let cl = (x0 + cmu).ln();
        let cj00 = 0.5 * (x0 * cmu - cl);
        let cj10 = x3 / 3.0 - cj00;
        let cj20 = (0.75 * cmu - 2.0) / 3.0 * x3 + 1.25 * cj00;
        (cj00, cj10, cj20)
    };
    let cj01 = x0;
    let cj11 = cj01 * cmu1;
    let cj21 = cj11 * cmu1;
    let cj02 = cmu / x0;
    let cj12 = cmu1 / x0 * (3.0 + 2.0 * cmu1);
    let cj22 = cmu1.pow2() / x0 * (5.0 + 3.0 * cmu1);
    let cj03 = -1.0 / x3;
    let cj13 = cmu1 / x3 * (2.0 * cmu1.pow2() + 6.0 * cmu1 + 3.0);
    let cj23 = cmu1.pow2() / x3 * (6.0 * cmu1.pow2() + 20.0 * cmu1 + 15.0);
    let cj04 = 3.0 * cmu / x5;
    let cj14 = -3.0 * cmu1 / x5;
    let cj24 = cmu1.pow2() / x5
        * (6.0 * cmu1.pow3() + 30.0 * cmu1.pow2() + 45.0 * cmu1 + 15.0);
    let cj05 = (-12.0 * cmu1.pow2() - 24.0 * cmu1 - 15.0) / (x5 * x0.pow2());

    let pit26 = (PI * temr).pow2() / 6.0;
    let cn0 = (0.5 / temr).sqrt() / temr;
    let cn1 = cn0 / temr;
    let cn2 = cn1 / temr;
    let w0 = BlinSeries {
        w: cn0 * (cj00 + pit26 * cj02),
        wdx: cn0 * temr * (cj01 + pit26 * cj03),
        wdt: cn1
            * (cmu1 * cj01 - 1.5 * cj00 + pit26 * (cmu1 * cj03 + 0.5 * cj02)),
        wdxx: cn0 * temr.pow2() * (cj02 + pit26 * cj04),
        wdxt: cn0
            * (cmu1 * cj02 - 0.5 * cj01 + pit26 * (cmu1 * cj04 + 1.5 * cj03)),
        wdtt: cn2
            * (3.75 * cj00 - 3.0 * cmu1 * cj01
                + cmu1.pow2() * cj02
                + pit26 * (-0.25 * cj02 + cmu1 * cj03 + cmu1.pow2() * cj04)),
    };
    let w1 = BlinSeries {
        w: cn1 * (cj10 + pit26 * cj12),
        wdx: cn0 * (cj11 + pit26 * cj13),
        wdt: cn2
            * (cmu1 * cj11 - 2.5 * cj10 + pit26 * (cmu1 * cj13 - 0.5 * cj12)),
        wdxx: cn0 * temr * (cj12 + pit26 * cj14),
        wdxt: cn1
            * (cmu1 * cj12 - 1.5 * cj11 + pit26 * (cmu1 * cj14 + 0.5 * cj13)),
        wdtt: cn2 / temr
            * (8.75 * cj10 - 5.0 * cmu1 * cj11
                + cmu1.pow2() * cj12
                + pit26 * (0.75 * cj12 - cmu1 * cj13 + cmu1.pow2() * cj14)),
    };
    let w2 = BlinSeries {
        w: cn2 * (cj20 + pit26 * cj22),
        wdx: cn1 * (cj21 + pit26 * cj23),
        wdt: cn2 / temr
            * (cmu1 * cj21 - 3.5 * cj20 + pit26 * (cmu1 * cj23 - 1.5 * cj22)),
        wdxx: cn0 * (cj22 + pit26 * cj24),
        wdxt: cn2
            * (cmu1 * cj22 - 2.5 * cj21 + pit26 * (cmu1 * cj24 - 0.5 * cj23)),
        wdtt: cn2 / temr.pow2()
            * (15.75 * cj20 - 7.0 * cmu1 * cj21
                + cmu1.pow2() * cj22
                + pit26
                    * (3.75 * cj22 - 3.0 * cmu1 * cj23 + cmu1.pow2() * cj24)),
    };
    let w0xxx = cn0 * temr.pow3() * (cj03 + pit26 * cj05);
    let w0xxt = cn0
        * temr
        * (cmu1 * cj03 + 0.5 * cj02 + pit26 * (cmu1 * cj05 + 2.5 * cj04));
    let w0xtt = cn1
        * (0.75 * cj01 - cmu1 * cj02
            + cmu1.pow2() * cj03
            + pit26 * (0.75 * cj03 + 3.0 * cmu1 * cj04 + cmu1.pow2() * cj05));
    Blin9Output {
        w0,
        w1,
        w2,
        w0xxx,
        w0xtt,
        w0xxt,
    }
}
