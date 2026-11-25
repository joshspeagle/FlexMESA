//! Ion-ion Coulomb liquid (`FITION9`), harmonic crystal (`FHARM12`, `HLfit12`),
//! and anharmonic crystal (`ANHARM8`) thermodynamic contributions for the
//! Potekhin-Chabrier (`PC`) equation of state.
//!
//! Reference: Potekhin & Chabrier (2000, 2010), Baiko, Potekhin & Yakovlev
//! (2001), Farouki & Hamaguchi (1993); upstream Fortran MESA
//! `eos/private/pc_eos.f90`.

use flexmesa_autodiff::AutoDiffReal2VarOrder1;

/// Reduced ion or screening thermodynamic contribution per ion.
#[derive(Debug, Clone, Copy)]
pub struct PcIonContrib {
    /// Free energy per ion divided by `k_B T`.
    pub f: AutoDiffReal2VarOrder1,
    /// Internal energy per ion divided by `k_B T`.
    pub u: AutoDiffReal2VarOrder1,
    /// Pressure divided by `n_i k_B T`.
    pub p: AutoDiffReal2VarOrder1,
    /// Entropy per ion divided by `k_B`.
    pub s: AutoDiffReal2VarOrder1,
    /// Heat capacity at constant volume per ion divided by `k_B`.
    pub cv: AutoDiffReal2VarOrder1,
    /// `p + (\partial p / \partial \ln T)_\rho = (n_i k_B T)^{-1} (\partial P / \partial \ln T)_\rho`.
    pub pdt: AutoDiffReal2VarOrder1,
    /// `p + (\partial p / \partial \ln \rho)_T = (n_i k_B T)^{-1} (\partial P / \partial \ln \rho)_T`.
    pub pdr: AutoDiffReal2VarOrder1,
}

/// Evaluate `FITION9(GAMI)` for the classical one-component plasma (OCP)
/// liquid (`pc_eos.f90` lines 630-690).
///
/// # Arguments
/// * `gami` - Ion-ion Coulomb coupling parameter `\Gamma_i`.
#[must_use]
pub fn fition9(gami: AutoDiffReal2VarOrder1) -> PcIonContrib {
    const A1: f64 = -0.907347;
    const A2: f64 = 0.62849;
    const C1: f64 = 0.004500;
    const G1: f64 = 170.0;
    const C2: f64 = -8.4e-5;
    const G2: f64 = 0.0037;
    const SQ32: f64 = 0.8660254038;

    let a3 = -SQ32 - A1 / A2.sqrt();
    let f0 = A1
        * ((gami * (A2 + gami)).sqrt()
            - A2 * ((gami / A2).sqrt() + (1.0 + gami / A2).sqrt()).ln())
        + 2.0 * a3 * (gami.sqrt() - gami.sqrt().atan());
    let u0 = gami.sqrt().pow3() * (A1 / (A2 + gami).sqrt() + a3 / (1.0 + gami));
    let u = u0
        + C1 * gami * gami / (G1 + gami)
        + C2 * gami * gami / (G2 + gami * gami);
    let f = f0
        + C1 * (gami - G1 * (1.0 + gami / G1).ln())
        + C2 / 2.0 * (1.0 + gami * gami / G2).ln();
    let cv = -0.5
        * gami.sqrt().pow3()
        * (A1 * A2 / (A2 + gami).sqrt().pow3()
            + a3 * (1.0 - gami) / (1.0 + gami).pow2())
        - gami
            * gami
            * (C1 * G1 / (G1 + gami).pow2()
                + C2 * (G2 - gami * gami) / (G2 + gami * gami).pow2());
    let p = u / 3.0;
    let pdr = (4.0 * u - cv) / 9.0;
    let pdt = cv / 3.0;
    let s = u - f;
    PcIonContrib {
        f,
        u,
        p,
        s,
        cv,
        pdt,
        pdr,
    }
}

/// Evaluate `HLfit12(eta, LATTICE)` for the thermal phonon excitations of a
/// harmonic crystal (`pc_eos.f90` lines 1294-1412).
///
/// # Arguments
/// * `eta_in` - Ion quantum parameter `\eta = T_p / T`.
/// * `lattice` - `1` for bcc lattice, `2` for fcc lattice.
#[must_use]
pub fn hlfit12(
    eta_in: AutoDiffReal2VarOrder1,
    lattice: i32,
) -> (
    AutoDiffReal2VarOrder1,
    AutoDiffReal2VarOrder1,
    AutoDiffReal2VarOrder1,
    AutoDiffReal2VarOrder1,
    AutoDiffReal2VarOrder1,
) {
    const EPS: f64 = 1.0e-5;
    const TINY: f64 = 1.0e-99;

    let (
        clm,
        u1,
        alpha,
        beta,
        gamma,
        a1,
        a2,
        a3,
        a4,
        a6,
        a8,
        b0,
        b2,
        b4,
        b5,
        b6,
        b7,
        c9,
        c11,
    ) = if lattice == 2 {
        (
            -2.45373,
            0.513194,
            0.257591,
            0.365284,
            0.9167070,
            0.0,
            0.532535,
            0.0,
            3.76545e-4,
            2.63013e-7,
            6.6318e-11,
            303.20,
            7.7255,
            0.0439597,
            0.000114295,
            5.63434e-5,
            1.36488e-6,
            0.00492387,
            0.00437506,
        )
    } else {
        (
            -2.49389,
            0.5113875,
            0.265764,
            0.334547,
            0.932446,
            0.1839,
            0.593586,
            0.0054814,
            5.01813e-4,
            3.9247e-7,
            5.8356e-11,
            261.66,
            7.07997,
            0.0409484,
            0.000397355,
            5.11148e-5,
            2.19749e-6,
            0.004757014,
            0.0047770935,
        )
    };

    let mut eta = eta_in;
    let u1_ad = AutoDiffReal2VarOrder1::constant(u1);
    let (f, u, cv) = if eta.val > 1.0 / EPS {
        let u = 3.0 / (c11 * eta * eta * eta);
        let f = -u / 3.0;
        let cv = 4.0 * u;
        (f, u, cv)
    } else if eta.val < EPS {
        if eta.val < TINY {
            eta = AutoDiffReal2VarOrder1::constant(TINY);
        }
        let f = 3.0 * eta.ln() + clm - 1.5 * u1_ad * eta + eta * eta / 24.0;
        let u = 3.0 - 1.5 * u1_ad * eta + eta * eta / 12.0;
        let cv = 3.0 - eta * eta / 12.0;
        (f, u, cv)
    } else {
        let b9 = a6 * c9;
        let b11 = a8 * c11;
        let up = 1.0
            + eta
                * (a1
                    + eta
                        * (a2
                            + eta
                                * (a3
                                    + eta
                                        * (a4
                                            + eta
                                                * eta
                                                * (a6 + eta * eta * a8)))));
        let dn = b0
            + eta
                * eta
                * (b2
                    + eta
                        * eta
                        * (b4
                            + eta
                                * (b5
                                    + eta
                                        * (b6
                                            + eta
                                                * (b7
                                                    + eta
                                                        * eta
                                                        * (b9
                                                            + eta
                                                                * eta
                                                                * b11))))));
        let ea = (-alpha * eta).exp();
        let eb = (-beta * eta).exp();
        let eg = (-gamma * eta).exp();
        let f = (1.0 - ea).ln() + (1.0 - eb).ln() + (1.0 - eg).ln() - up / dn;
        let up1 = a1
            + eta
                * (2.0 * a2
                    + eta
                        * (3.0 * a3
                            + eta
                                * (4.0 * a4
                                    + eta
                                        * eta
                                        * (6.0 * a6 + eta * eta * 8.0 * a8))));
        let up2 = 2.0 * a2
            + eta
                * (6.0 * a3
                    + eta
                        * (12.0 * a4
                            + eta * eta * (30.0 * a6 + eta * eta * 56.0 * a8)));
        let dn1 = eta
            * (2.0 * b2
                + eta
                    * eta
                    * (4.0 * b4
                        + eta
                            * (5.0 * b5
                                + eta
                                    * (6.0 * b6
                                        + eta
                                            * (7.0 * b7
                                                + eta
                                                    * eta
                                                    * (9.0 * b9
                                                        + eta
                                                            * eta
                                                            * 11.0
                                                            * b11))))));
        let dn2 = 2.0 * b2
            + eta
                * eta
                * (12.0 * b4
                    + eta
                        * (20.0 * b5
                            + eta
                                * (30.0 * b6
                                    + eta
                                        * (42.0 * b7
                                            + eta
                                                * eta
                                                * (72.0 * b9
                                                    + eta
                                                        * eta
                                                        * 110.0
                                                        * b11)))));
        let u_over_eta = alpha * ea / (1.0 - ea)
            + beta * eb / (1.0 - eb)
            + gamma * eg / (1.0 - eg)
            - (up1 * dn - dn1 * up) / (dn * dn);
        let cv_over_eta2 = alpha * alpha * ea / ((1.0 - ea) * (1.0 - ea))
            + beta * beta * eb / ((1.0 - eb) * (1.0 - eb))
            + gamma * gamma * eg / ((1.0 - eg) * (1.0 - eg))
            + ((up2 * dn - dn2 * up) * dn - 2.0 * (up1 * dn - dn1 * up) * dn1)
                / (dn * dn * dn);
        let u = u_over_eta * eta;
        let cv = cv_over_eta2 * eta * eta;
        (f, u, cv)
    };
    let s = u - f;
    (f, u, cv, s, u1_ad)
}

/// Evaluate `FHARM12(GAMI, TPT)` for the harmonic bcc Coulomb crystal
/// including static Madelung and zero-point lattice energy (`pc_eos.f90`
/// lines 1261-1292).
#[must_use]
pub fn fharm12(
    gami: AutoDiffReal2VarOrder1,
    tpt: AutoDiffReal2VarOrder1,
) -> PcIonContrib {
    const CM: f64 = 0.895929256;
    let (f, u, cv_th, s_th, u1) = hlfit12(tpt, 1);
    let u0 = -CM * gami;
    let e0 = 1.5 * u1 * tpt;
    let u_th = u + e0;
    let f_th = f + e0;
    let u_harm = u0 + u_th;
    let f_harm = u0 + f_th;
    let s_harm = s_th;
    let p_harm = u0 / 3.0 + u_th / 2.0;
    let pdt_harm = 0.5 * cv_th;
    let pdr_harm = u0 / 2.25 + 0.75 * u_th - 0.25 * cv_th;
    PcIonContrib {
        f: f_harm,
        u: u_harm,
        p: p_harm,
        s: s_harm,
        cv: cv_th,
        pdt: pdt_harm,
        pdr: pdr_harm,
    }
}

/// Evaluate `ANHARM8(GAMI, TPT)` for the anharmonic crystal free energy and
/// derivatives (`pc_eos.f90` lines 1202-1259).
#[must_use]
pub fn anharm8(
    gami: AutoDiffReal2VarOrder1,
    tpt: AutoDiffReal2VarOrder1,
) -> PcIonContrib {
    const AA: [f64; 3] = [10.9, 247.0, 1.765e5];

    let b1 = AutoDiffReal2VarOrder1::constant(0.12);
    let ck = b1 / AA[0];
    let tpt2 = tpt * tpt;
    let tpt4 = tpt2 * tpt2;
    let tq = b1 * tpt2 / gami;
    let tk2 = ck * tpt2;
    let sup = (-tk2).exp();
    let mut f_ah = AutoDiffReal2VarOrder1::zero();
    let mut u_ah = AutoDiffReal2VarOrder1::zero();
    let mut p_ah = AutoDiffReal2VarOrder1::zero();
    let mut cv_ah = AutoDiffReal2VarOrder1::zero();
    let mut pdt_ah = AutoDiffReal2VarOrder1::zero();
    let mut pdr_ah = AutoDiffReal2VarOrder1::zero();
    let mut sup_gn = sup;

    for (idx, &acn_val) in AA.iter().enumerate() {
        let cn = AutoDiffReal2VarOrder1::constant((idx + 1) as f64);
        sup_gn = sup_gn / gami;
        let acn = AutoDiffReal2VarOrder1::constant(acn_val);
        f_ah = f_ah - acn / cn * sup_gn;
        u_ah = u_ah + (acn * (1.0 + 2.0 * tk2 / cn)) * sup_gn;
        let pn = acn_val / 3.0 + tk2 * acn_val / cn;
        p_ah = p_ah + pn * sup_gn;
        cv_ah = cv_ah
            + ((cn + 1.0) * acn_val
                + (4.0 - 2.0 / cn) * acn_val * tk2
                + 4.0 * acn_val * ck * ck / cn * tpt4)
                * sup_gn;
        pdt_ah = pdt_ah
            + (pn * (1.0 + cn + 2.0 * tk2) - 2.0 / cn * acn_val * tk2) * sup_gn;
        pdr_ah = pdr_ah
            + (pn * (1.0 - cn / 3.0 - tk2) + acn_val / cn * tk2) * sup_gn;
    }
    f_ah = f_ah - tq;
    u_ah = u_ah - tq;
    p_ah = p_ah - tq / 1.5;
    pdr_ah = pdr_ah - tq / 4.5;
    let s_ah = u_ah - f_ah;
    PcIonContrib {
        f: f_ah,
        u: u_ah,
        p: p_ah,
        s: s_ah,
        cv: cv_ah,
        pdt: pdt_ah,
        pdr: pdr_ah,
    }
}
