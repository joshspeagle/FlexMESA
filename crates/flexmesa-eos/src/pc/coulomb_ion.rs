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
    let sq_g = gami.sqrt();
    let f0 = ((gami * (gami + A2)).sqrt()
        - ((gami / A2).sqrt() + (gami / A2 + 1.0).sqrt()).ln() * A2)
        * A1
        + (sq_g - sq_g.atan()) * (2.0 * a3);
    let u0 = sq_g.pow3()
        * ((gami + A2).sqrt().recip() * A1 + (gami + 1.0).recip() * a3);
    let gami2 = gami.pow2();
    let u = u0 + gami2 * C1 / (gami + G1) + gami2 * C2 / (gami2 + G2);
    let f = f0
        + (gami - (gami / G1 + 1.0).ln() * G1) * C1
        + (gami2 / G2 + 1.0).ln() * (0.5 * C2);
    let cv = -sq_g.pow3()
        * 0.5
        * ((gami + A2).sqrt().pow3().recip() * (A1 * A2)
            + (-gami + 1.0) / (gami + 1.0).pow2() * a3)
        - gami2
            * ((gami + G1).pow2().recip() * (C1 * G1)
                + (-gami2 + G2) / (gami2 + G2).pow2() * C2);
    let p = u / 3.0;
    let pdr = (u * 4.0 - cv) / 9.0;
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
    f64,
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
    let (f, u, cv) = if eta.val > 1.0 / EPS {
        let u = (eta.pow3() * c11).recip() * 3.0;
        let f = -u / 3.0;
        let cv = u * 4.0;
        (f, u, cv)
    } else if eta.val < EPS {
        if eta.val < TINY {
            eta = AutoDiffReal2VarOrder1::constant(TINY);
        }
        let f = eta.ln() * 3.0 + clm - eta * (1.5 * u1) + eta.pow2() / 24.0;
        let u = -eta * (1.5 * u1) + eta.pow2() / 12.0 + 3.0;
        let cv = -eta.pow2() / 12.0 + 3.0;
        (f, u, cv)
    } else {
        let b9 = a6 * c9;
        let b11 = a8 * c11;
        let eta2 = eta.pow2();
        let up = eta
            * (eta * (eta * (eta * (eta2 * (eta2 * a8 + a6) + a4) + a3) + a2)
                + a1)
            + 1.0;
        let dn = eta2
            * (eta2
                * (eta
                    * (eta * (eta * (eta2 * (eta2 * b11 + b9) + b7) + b6)
                        + b5)
                    + b4)
                + b2)
            + b0;
        let ea = (-eta * alpha).exp();
        let eb = (-eta * beta).exp();
        let eg = (-eta * gamma).exp();
        let f =
            (-ea + 1.0).ln() + (-eb + 1.0).ln() + (-eg + 1.0).ln() - up / dn;
        let up1 = eta
            * (eta
                * (eta * (eta2 * (eta2 * (8.0 * a8) + 6.0 * a6) + 4.0 * a4)
                    + 3.0 * a3)
                + 2.0 * a2)
            + a1;
        let up2 = eta
            * (eta * (eta2 * (eta2 * (56.0 * a8) + 30.0 * a6) + 12.0 * a4)
                + 6.0 * a3)
            + 2.0 * a2;
        let dn1 = eta
            * (eta2
                * (eta
                    * (eta
                        * (eta
                            * (eta2 * (eta2 * (11.0 * b11) + 9.0 * b9)
                                + 7.0 * b7)
                            + 6.0 * b6)
                        + 5.0 * b5)
                    + 4.0 * b4)
                + 2.0 * b2);
        let dn2 = eta2
            * (eta
                * (eta
                    * (eta
                        * (eta2 * (eta2 * (110.0 * b11) + 72.0 * b9)
                            + 42.0 * b7)
                        + 30.0 * b6)
                    + 20.0 * b5)
                + 12.0 * b4)
            + 2.0 * b2;
        let u_over_eta = ea * alpha / (-ea + 1.0)
            + eb * beta / (-eb + 1.0)
            + eg * gamma / (-eg + 1.0)
            - (up1 * dn - dn1 * up) / dn.pow2();
        let cv_over_eta2 = ea * (alpha * alpha) / (-ea + 1.0).pow2()
            + eb * (beta * beta) / (-eb + 1.0).pow2()
            + eg * (gamma * gamma) / (-eg + 1.0).pow2()
            + ((up2 * dn - dn2 * up) * dn - (up1 * dn - dn1 * up) * dn1 * 2.0)
                / dn.pow3();
        (f, u_over_eta * eta, cv_over_eta2 * eta2)
    };
    let s = u - f;
    (f, u, cv, s, u1)
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
    let u0 = -gami * CM;
    let e0 = tpt * (1.5 * u1);
    let u_th = u + e0;
    let f_th = f + e0;
    let u_harm = u0 + u_th;
    let f_harm = u0 + f_th;
    let s_harm = s_th;
    let p_harm = u0 / 3.0 + u_th / 2.0;
    let pdt_harm = cv_th * 0.5;
    let pdr_harm = u0 / 2.25 + u_th * 0.75 - cv_th * 0.25;
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
    const B1: f64 = 0.12;
    const AA: [f64; 3] = [10.9, 247.0, 1.765e5];

    let ck = B1 / AA[0];
    let tpt2 = tpt.pow2();
    let tpt4 = tpt2.pow2();
    let tq = tpt2 * B1 / gami;
    let tk2 = tpt2 * ck;
    let sup = (-tk2).exp();
    let mut f_ah = AutoDiffReal2VarOrder1::zero();
    let mut u_ah = AutoDiffReal2VarOrder1::zero();
    let mut p_ah = AutoDiffReal2VarOrder1::zero();
    let mut cv_ah = AutoDiffReal2VarOrder1::zero();
    let mut pdt_ah = AutoDiffReal2VarOrder1::zero();
    let mut pdr_ah = AutoDiffReal2VarOrder1::zero();
    let mut sup_gn = sup;

    for (idx, &acn) in AA.iter().enumerate() {
        let cn = (idx + 1) as f64;
        sup_gn = sup_gn / gami;
        f_ah = f_ah - sup_gn * (acn / cn);
        u_ah = u_ah + (tk2 * (2.0 / cn) + 1.0) * acn * sup_gn;
        let pn = tk2 * (acn / cn) + acn / 3.0;
        p_ah = p_ah + pn * sup_gn;
        cv_ah = cv_ah
            + (tk2 * ((4.0 - 2.0 / cn) * acn)
                + tpt4 * (4.0 * acn * ck * ck / cn)
                + (cn + 1.0) * acn)
                * sup_gn;
        pdt_ah = pdt_ah
            + (pn * (tk2 * 2.0 + cn + 1.0) - tk2 * (2.0 * acn / cn)) * sup_gn;
        pdr_ah = pdr_ah
            + (pn * (-tk2 + (1.0 - cn / 3.0)) + tk2 * (acn / cn)) * sup_gn;
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
