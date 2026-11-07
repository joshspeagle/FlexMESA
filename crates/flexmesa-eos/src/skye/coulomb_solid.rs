//! Harmonic/anharmonic bcc Coulomb solid free energy, electron-ion screening,
//! and Ogata et al. (1993) / Potekhin & Chabrier (2013) solid mixing-rule
//! corrections for the `Skye` equation of state.
//!
//! Reference: Jermyn et al. (2021), ApJ, 913, 72; Baiko, Potekhin & Yakovlev
//! (2001), Farouki & Hamaguchi (1993), Ogata et al. (1993), Potekhin &
//! Chabrier (2013); upstream Fortran MESA `eos/private/skye_coulomb_solid.f90`.

use flexmesa_autodiff::AutoDiffReal2VarOrder3;
use flexmesa_const::{EULERNUM, FINE, PI};

use crate::skye::coulomb_liquid::{
    ocp_liquid_screening_free_energy_correction, ME_IN_AMU,
};

/// Prescription for multi-component solid linear-mixing-rule corrections in
/// `Skye` (`Skye_solid_mixing_rule`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SkyeSolidMixingRule {
    /// Potekhin & Chabrier (2013) analytical fit (`"PC"`, MESA default).
    #[default]
    Pc,
    /// Ogata, Iyetomi & Ichimaru (1993) fit (`"Ogata"`).
    Ogata,
}

/// Classical and quantum anharmonic non-ideal free energy per ion per `k_B T`
/// in the solid phase (`skye_coulomb_solid.f90` lines 46-62).
#[must_use]
pub fn ocp_solid_anharmonic_free_energy(
    gami: AutoDiffReal2VarOrder3,
    tpt: AutoDiffReal2VarOrder3,
) -> AutoDiffReal2VarOrder3 {
    const A1: f64 = 10.9;
    const A2: f64 = 247.0;
    const A3: f64 = 1.765e5;
    const B1: f64 = 0.12;

    let tpt2 = tpt * tpt;
    let f = -(A1 / gami + A2 / (2.0 * gami.pow2()) + A3 / (3.0 * gami.pow3()));
    f * (-(B1 / A1) * tpt2).exp() - B1 * tpt2 / gami
}

/// Harmonic non-ideal bcc solid free energy per ion per `k_B T` subtracting
/// the ideal-gas contribution (`skye_coulomb_solid.f90` lines 77-145).
#[must_use]
pub fn ocp_solid_harmonic_free_energy(
    gami: AutoDiffReal2VarOrder3,
    tpt_in: AutoDiffReal2VarOrder3,
) -> AutoDiffReal2VarOrder3 {
    const CM: f64 = 0.895929256;
    const EPS: f64 = 1.0e-5;
    const CLM: f64 = -2.49389;
    const U1: f64 = 0.5113875;
    const ALPHA: f64 = 0.265764;
    const BETA: f64 = 0.334547;
    const GAMMA: f64 = 0.932446;
    const A1: f64 = 0.1839;
    const A2: f64 = 0.593586;
    const A3: f64 = 0.0054814;
    const A4: f64 = 5.01813e-4;
    const A6: f64 = 3.9247e-7;
    const A8: f64 = 5.8356e-11;
    const B0: f64 = 261.66;
    const B2: f64 = 7.07997;
    const B4: f64 = 0.0409484;
    const B5: f64 = 0.000397355;
    const B6: f64 = 5.11148e-5;
    const B7: f64 = 2.19749e-6;
    const C9: f64 = 0.004757014;
    const C11: f64 = 0.0047770935;
    const B9: f64 = A6 * C9;
    const B11: f64 = A8 * C11;

    let tpt = tpt_in;
    let f_therm = if tpt.val > 1.0 / EPS {
        -1.0 / (C11 * tpt * tpt * tpt)
    } else if tpt.val < EPS {
        3.0 * tpt.log() + CLM - (1.5 * U1) * tpt + tpt * tpt / 24.0
    } else {
        let tpt2 = tpt * tpt;
        let up = 1.0
            + tpt
                * (A1
                    + tpt
                        * (A2
                            + tpt
                                * (A3 + tpt * (A4 + tpt2 * (A6 + tpt2 * A8)))));
        let dn = B0
            + tpt2
                * (B2
                    + tpt2
                        * (B4
                            + tpt
                                * (B5
                                    + tpt
                                        * (B6
                                            + tpt
                                                * (B7
                                                    + tpt2
                                                        * (B9
                                                            + tpt2 * B11))))));
        let ea = (-ALPHA * tpt).exp();
        let eb = (-BETA * tpt).exp();
        let eg = (-GAMMA * tpt).exp();
        (1.0 - ea).log() + (1.0 - eb).log() + (1.0 - eg).log() - up / dn
    };
    let u0 = -CM * gami;
    let e0 = (1.5 * U1) * tpt;
    let f = u0 + f_therm + e0;
    f - 3.0 * tpt.log() + 1.5 * gami.log() + 1.323515
}

#[inline]
fn safe_exp(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    x.min_scalar(100.0).max_scalar(-100.0).exp()
}

#[inline]
fn safe_tanh(x: AutoDiffReal2VarOrder3) -> AutoDiffReal2VarOrder3 {
    x.min_scalar(100.0).max_scalar(-100.0).tanh()
}

/// Electron-ion screening free-energy correction per ion per `k_B T` in the
/// solid phase (`skye_coulomb_solid.f90` lines 175-214).
#[must_use]
pub fn ocp_solid_screening_free_energy_correction(
    z_in: f64,
    mi: f64,
    ge: AutoDiffReal2VarOrder3,
    rs: AutoDiffReal2VarOrder3,
) -> AutoDiffReal2VarOrder3 {
    const ATF: f64 = 0.00352;
    let zi = (z_in as i64) as f64;
    let z2 = zi * zi;
    let z3 = z2 * zi;
    let z4 = z3 * zi;
    let z5 = z4 * zi;
    let z6 = z5 * zi;
    let logz = zi.ln();
    let logz_3_2 = logz.powf(1.5);
    let zm0p267 = zi.powf(-0.267);
    let zm1_3 = zi.powf(-1.0 / 3.0);
    let zm3_2 = zi.powf(-1.5);
    let z5_3 = zi.powf(5.0 / 3.0);
    let z7_6 = zi.powf(7.0 / 6.0);
    let z2_3 = zi.powf(2.0 / 3.0);

    let s = 1.0 / (1.0 + 1.0e-2 * logz_3_2 + 0.097 / z2);
    let b1 = 1.0 - 1.1866 * zm0p267 + 0.27 / z_in;
    let b2 = 1.0
        + (2.25 * zm1_3) * (1.0 + 0.684 * z5 + 0.222 * z6) / (1.0 + 0.222 * z6);
    let b3 = 41.5 / (1.0 + logz);
    let b4 = 0.395 * logz + 0.347 * zm3_2;

    let g = ge * z5_3;
    let cotpt = (3.0 * ME_IN_AMU / mi).sqrt() / z7_6;
    let tpt = g * cotpt / rs.sqrt();
    let tpt_scaled2 = (0.205 * tpt).pow2();
    let supp = safe_exp(-tpt_scaled2);
    let q = ((tpt_scaled2 + (1.0 + supp).log())
        / (EULERNUM - (EULERNUM - 2.0) * supp).log())
    .sqrt();

    let xr = ((9.0 * PI / 4.0).powf(1.0 / 3.0) * FINE) / rs;
    let xr2 = xr.pow2();
    let a = (b3 + 17.9 * xr2) / (1.0 + b4 * xr2);
    let f_inf = (ATF * z2_3 * b1) * (1.0 + b2 / xr2).sqrt();
    let f_sol = -f_inf * g * (1.0 + a * (q / g).powf(s));

    let gr = (1.0 + xr2).sqrt();
    let alpha = (3.0 * (4.0 / (9.0 * PI)).powf(2.0 / 3.0)) * (rs / ge) * gr;
    let f_liq = ocp_liquid_screening_free_energy_correction(z_in, mi, ge, rs);
    let switch = safe_tanh(2.0 * alpha).pow3();
    switch * f_liq + (1.0 - switch) * f_sol
}

/// Ogata, Iyetomi & Ichimaru (1993) two-component solid mixing factor
/// `\Delta G(x_2, R_Z)` (`skye_coulomb_solid.f90` lines 226-239).
#[must_use]
pub fn delta_g_ogata93(x2: f64, rz: f64) -> f64 {
    let rm1 = rz - 1.0;
    let cr =
        0.05 * (rm1 * rm1) / ((1.0 + 0.64 * rm1) * (1.0 + 0.5 * rm1 * rm1));
    let sq = x2.sqrt();
    cr / (1.0
        + (sq * (sq - 0.3) * (sq - 0.7) * (sq - 1.0)) * 27.0 * rm1
            / (1.0 + 0.1 * rm1))
}

/// Potekhin & Chabrier (2013) two-component solid mixing factor
/// `\Delta G(x_2, R_Z)` (`skye_coulomb_solid.f90` lines 248-259).
#[must_use]
pub fn delta_g_pc13(x2: f64, rz: f64) -> f64 {
    let x = x2 / rz + (1.0 - 1.0 / rz) * x2.powf(rz);
    0.012
        * ((x * (1.0 - x)) / (x2 * (1.0 - x2)))
        * (1.0 - 1.0 / (rz * rz))
        * (1.0 - x2 + x2 * rz.powf(5.0 / 3.0))
}

/// Multi-component Coulomb solid linear-mixing-rule correction
/// (`skye_coulomb_solid.f90` lines 270-345).
#[must_use]
pub fn solid_mixing_rule_correction(
    rule: SkyeSolidMixingRule,
    ay: &[f64],
    azion: &[f64],
    game: AutoDiffReal2VarOrder3,
) -> AutoDiffReal2VarOrder3 {
    const EPS: f64 = 1.0e-40;
    let n = ay.len().min(azion.len());
    let mut unique_charges: Vec<f64> = Vec::with_capacity(n);
    let mut charge_abundances: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        if let Some(idx) = unique_charges.iter().position(|&z| z == azion[i]) {
            charge_abundances[idx] += ay[i];
        } else {
            unique_charges.push(azion[i]);
            charge_abundances.push(ay[i]);
        }
    }

    let num_unique = unique_charges.len();
    let mut f = AutoDiffReal2VarOrder3::constant(0.0);
    for i in 0..num_unique {
        let zi = unique_charges[i];
        if zi == 0.0 {
            continue;
        }
        for j in 0..num_unique {
            let zj = unique_charges[j];
            if zj == 0.0 || zj < zi {
                continue;
            }
            let rz = zj / zi;
            let aj = charge_abundances[j]
                / EPS.max(charge_abundances[i] + charge_abundances[j]);
            let dg = match rule {
                SkyeSolidMixingRule::Ogata => delta_g_ogata93(aj, rz),
                SkyeSolidMixingRule::Pc => delta_g_pc13(aj, rz),
            };
            let gami = game * zi.powf(5.0 / 3.0);
            f = f + gami * (charge_abundances[i] * charge_abundances[j] * dg);
        }
    }
    f
}
