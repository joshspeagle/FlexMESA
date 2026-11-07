//! Classical/quantum OCP liquid free energy, electron-ion screening, liquid
//! linear-mixing correction, and third-order electron exchange-correlation
//! (`EXCOR7`) for the `Skye` equation of state.
//!
//! Reference: Jermyn et al. (2021), ApJ, 913, 72; Potekhin & Chabrier (2000,
//! 2013), Baiko & Yakovlev (2019); upstream Fortran MESA
//! `eos/private/skye_coulomb_liquid.f90` and `eos/private/skye_coulomb.f90`.

use flexmesa_autodiff::AutoDiffReal2VarOrder3;
use flexmesa_const::{AMU, FINE, ME, PI};

/// Electron mass in atomic mass units (`m_e / \mathrm{amu}`).
pub const ME_IN_AMU: f64 = ME / AMU;

/// Classical one-component plasma (OCP) liquid free energy per ion per `k_B T`
/// (`skye_coulomb_liquid.f90` lines 42-63).
#[must_use]
pub fn classical_ocp_liquid_free_energy(
    g: AutoDiffReal2VarOrder3,
) -> AutoDiffReal2VarOrder3 {
    let sq32 = 3.0_f64.sqrt() / 2.0;
    let a1 = -0.9070_f64;
    let a2 = 0.62954_f64;
    let a3 = -sq32 - a1 / a2.sqrt();
    let b1 = 4.56e-3_f64;
    let b2 = 211.6_f64;
    let b3 = -1.0e-4_f64;
    let b4 = 4.62e-3_f64;

    let sq_g = g.sqrt();
    let fa = a1
        * ((g * (a2 + g)).sqrt()
            - a2 * ((g / a2).sqrt() + (1.0 + g / a2).sqrt()).log())
        + 2.0 * a3 * (sq_g - sq_g.atan());
    let fb = b1 * (g - b2 * (1.0 + g / b2).log())
        + 0.5 * b3 * (1.0 + g.pow2() / b4).log();
    fa + fb
}

/// Quantum OCP liquid free-energy correction per ion per `k_B T` from Baiko &
/// Yakovlev (2019) (`skye_coulomb_liquid.f90` lines 72-85).
#[must_use]
pub fn quantum_ocp_liquid_free_energy_correction(
    tpt: AutoDiffReal2VarOrder3,
) -> AutoDiffReal2VarOrder3 {
    let q1 = 5.994_f64;
    let q2 = 70.3_f64;
    let q4 = 22.7_f64;
    let q3 = (0.25 - q1 / q2) * q4;
    let eta = tpt / 3.0_f64.sqrt();
    q1 * eta - (q1 * q2) * (1.0 + eta / q2).log()
        + (0.5 * q3) * (1.0 + eta.pow2() / q4).log()
}

/// Electron-ion screening free-energy correction per ion per `k_B T` in the
/// liquid phase (`skye_coulomb_liquid.f90` lines 95-124).
#[must_use]
pub fn ocp_liquid_screening_free_energy_correction(
    z_in: f64,
    _mi: f64,
    ge: AutoDiffReal2VarOrder3,
    rs: AutoDiffReal2VarOrder3,
) -> AutoDiffReal2VarOrder3 {
    let zi = (z_in as i64) as f64;
    let z2 = zi * zi;
    let logz = zi.ln();
    let sqlogz = logz * logz;
    let z0p475 = zi.powf(0.475);
    let zp1_3_2 = (1.0 + zi).powf(1.5);
    let z3_2 = zi.powf(1.5);
    let z7_3 = zi.powf(7.0 / 3.0);
    let zm1_3 = zi.powf(-1.0 / 3.0);
    let zm1_2 = zi.powf(-0.5);
    let zm1_4 = zi.powf(-0.25);

    let a = 1.11 * z0p475;
    let b = 0.2 + 0.078 * sqlogz;
    let nu = 1.16 + 0.08 * logz;
    let cdh = (z_in / 3.0_f64.sqrt()) * (zp1_3_2 - 1.0 - z3_2);
    let ctf = (18.0 / 175.0)
        * (12.0 / PI).powf(2.0 / 3.0)
        * z7_3
        * (1.0 - zm1_3 + 0.2 * zm1_2);

    let xr = ((9.0 * PI / 4.0).powf(1.0 / 3.0) * FINE) / rs;
    let xr2 = xr.pow2();
    let gr = (1.0 + xr2).sqrt();
    let g1 = 1.0 + 0.78 * (ge / z_in).sqrt() / (21.0 + ge * (z_in / rs).pow3());
    let g2 = 1.0
        + ((z_in - 1.0) / 9.0)
            * (1.0 + 1.0 / (0.001 * z2 + 2.0 * ge))
            * (rs.pow3() / (1.0 + 6.0 * rs.pow2()));
    let h = (1.0 + 0.2 * xr2)
        / (1.0 + (0.18 * zm1_4) * xr + (0.37 * zm1_2) * xr2 + 0.2 * xr2);
    let sq_ge = ge.sqrt();
    let ge_nu = ge.powf(nu);
    -ge * (cdh * sq_ge + (ctf * a) * ge_nu * g1 * h)
        / (1.0 + (b * sq_ge + a * g2 * ge_nu / rs) / gr)
}

/// Correction to the linear mixing rule for a Coulomb liquid mixture
/// (`skye_coulomb_liquid.f90` lines 138-178).
#[must_use]
pub fn liquid_mixing_rule_correction(
    rs: AutoDiffReal2VarOrder3,
    game: AutoDiffReal2VarOrder3,
    zmean: f64,
    z2mean: f64,
    z52: f64,
    z53: f64,
    z321: f64,
) -> AutoDiffReal2VarOrder3 {
    const TINY: f64 = 1.0e-9;
    if zmean == 0.0 {
        return AutoDiffReal2VarOrder3::constant(0.0);
    }
    let d = z2mean / (zmean * zmean);
    if (d - 1.0).abs() < TINY {
        return AutoDiffReal2VarOrder3::constant(0.0);
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
    let p3 = d.powf(-0.2);
    let d0 = (2.6 * dif_r + 14.0 * dif_r * dif_r * dif_r) / (1.0 - p3);
    let gp = d0 * gami_mean.powf(p3);
    let fmix0 = dif_fdh / (1.0 + gp);
    let q = d * d * 0.0117;
    let r = 1.5 / p3 - 1.0;
    let gq = q * gp;
    fmix0 / (1.0 + gq).powf(r)
}

/// Electron exchange-correlation free energy per electron per `k_B T` in
/// [`AutoDiffReal2VarOrder3`] (`skye_coulomb.f90` lines 404-484).
#[must_use]
pub fn excor7_order3(
    rs: AutoDiffReal2VarOrder3,
    game: AutoDiffReal2VarOrder3,
) -> AutoDiffReal2VarOrder3 {
    let theta = 0.543 * rs / game;
    let sqth = theta.sqrt();
    let theta2 = theta * theta;
    let theta3 = theta2 * theta;
    let theta4 = theta3 * theta;
    let (t1, t2) = if theta.val > 0.007 {
        let inv_th = 1.0 / theta;
        let inv_sq = 1.0 / sqth;
        let cht1 = inv_th.cosh();
        let sht1 = inv_th.sinh();
        let cht2 = inv_sq.cosh();
        let sht2 = inv_sq.sinh();
        (sht1 / cht1, sht2 / cht2)
    } else {
        let one = AutoDiffReal2VarOrder3::constant(1.0);
        (one, one)
    };
    let a0 = 0.75 + 3.04363 * theta2 - 0.09227 * theta3 + 1.7035 * theta4;
    let a1 = 1.0 + 8.31051 * theta2 + 5.1105 * theta4;
    let a = 0.610887 * a0 / a1 * t1;

    let b0 = 0.341308 + 12.070873 * theta2 + 1.148889 * theta4;
    let b1 = 1.0 + 10.495346 * theta2 + 1.326623 * theta4;
    let b = sqth * t2 * b0 / b1;

    let d0 = 0.614925 + 16.996055 * theta2 + 1.489056 * theta4;
    let d1 = 1.0 + 10.10935 * theta2 + 1.22184 * theta4;
    let d = sqth * t2 * d0 / d1;

    let e0 = 0.539409 + 2.522206 * theta2 + 0.178484 * theta4;
    let e1 = 1.0 + 2.555501 * theta2 + 0.146319 * theta4;
    let e = theta * t1 * e0 / e1;

    let exp1th = (-1.0 / theta).exp();
    let c = (0.872496 + 0.025248 * exp1th) * e;
    let discr = (4.0 * e - d * d).sqrt();
    let s1 = -c / e * game;
    let b2 = b - c * d / e;
    let sqge = game.sqrt();
    let s2 = -2.0 / e * b2 * sqge;
    let r3 = e * game + d * sqge + 1.0;
    let b3 = a - c / e;
    let c3 = (d / e * b2 - b3) / e;
    let s3 = c3 * r3.log();
    let b4 = 2.0 - d * d / e;
    let c4 = 2.0 * e * sqge + d;
    let s4a = 2.0 / e / discr;
    let s4b = d * b3 + b4 * b2;
    let s4c = (c4 / discr).atan() - (d / discr).atan();
    let s4 = s4a * s4b * s4c;
    s1 + s2 + s3 + s4
}
