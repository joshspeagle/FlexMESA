//! Uniform-background Coulomb corrections (`pcoul`, `ecoul`, `scoul`) for the
//! `HELM` equation of state.
//!
//! This module ports `eos/private/helm_coulomb2.dek` (Yakovlev & Shalybkov
//! 1989, Sov. Sci. Rev. E Astrophys. Space Phys., 7, 311; Timmes & Swesty
//! 2000).
//!
//! Given the ion number density $n_{\mathrm{ion}}$ (`xni`), ideal-ion pressure
//! $P_{\mathrm{ion}}$ (`pion`), temperature $T$, mass density $\rho$, and
//! composition moments $(\bar{A}, \bar{Z})$, the routine computes the ion
//! coupling parameter
//! $$\Gamma = \frac{\bar{Z}^{5/3} e^2}{a_e k_B T}, \quad
//! a_e = \left(\frac{3}{4\pi \bar{Z} n_{\mathrm{ion}}}\right)^{1/3}$$
//! and evaluates the strong-coupling ($\Gamma \ge 1$) or weak-coupling
//! ($\Gamma < 1$) free-energy fits together with all first and second partial
//! derivatives with respect to $(\rho, T, \bar{A}, \bar{Z})$.

use flexmesa_const::{AVO, KERG, PI, QE};

use crate::helm::components::HelmThermoVar;

const A1: f64 = -0.898004;
const B1: f64 = 0.96786;
const C1: f64 = 0.220703;
const D1: f64 = -0.86097;
const E1: f64 = 2.5269;
const A2: f64 = 0.29561;
const B2: f64 = 1.9885;
const C2: f64 = 0.288675;

/// Evaluates `HELM` Coulomb pressure, specific energy, specific entropy, and
/// ion coupling parameter $\Gamma$ (`plasg`).
///
/// # Errors
/// Returns `Err(-1)` if `plasg < 0.0`.
#[allow(clippy::too_many_arguments)]
pub fn eval_helm_coulomb(
    temp: f64,
    den: f64,
    abar: f64,
    zbar: f64,
    xni: &HelmThermoVar,
    pion: &HelmThermoVar,
    coulomb_temp_cut: f64,
    coulomb_den_cut: f64,
) -> Result<(HelmThermoVar, HelmThermoVar, HelmThermoVar, f64), i32> {
    let third = 1.0 / 3.0;
    let forth = 4.0 / 3.0;
    let fiveth = 5.0 / 3.0;
    let teninth = 10.0 / 9.0;
    let forthpi = forth * PI;
    let esqu = QE * QE;
    let tempinv = 1.0 / temp;
    let deninv = 1.0 / den;
    let ytot1 = 1.0 / abar;
    let ktinv = 1.0 / (KERG * temp);

    let y_s = forthpi * zbar;
    let s_val = y_s * xni.val;
    let sinv = 1.0 / s_val;
    let dsdd = y_s * xni.dd;
    let dsda = y_s * xni.da;
    let dsdz = forthpi * xni.val;
    let dsdda = y_s * xni.dda;
    let dsddz = forthpi * xni.dd;
    let dsdaa = y_s * xni.daa;
    let dsdaz = forthpi * xni.da;

    let aele_val = sinv.powf(third);
    let aeleinv = 1.0 / aele_val;
    let z_a = -third * aele_val * sinv;
    let y_a = -forth * z_a * sinv;
    let daeledd = z_a * dsdd;
    let daeleda = z_a * dsda;
    let daeledz = z_a * dsdz;
    let daeleddd = y_a * dsdd * dsdd;
    let daeledda = y_a * dsda * dsdd + z_a * dsdda;
    let daeleddz = y_a * dsdz * dsdd + z_a * dsddz;
    let daeledaa = y_a * dsda * dsda + z_a * dsdaa;
    let daeledaz = y_a * dsdz * dsda + z_a * dsdaz;
    let daeledzz = y_a * dsdz * dsdz;

    let eplasg = esqu * ktinv * aeleinv;
    let z_e = -eplasg * aeleinv;
    let y_e = -2.0 * z_e * aeleinv;
    let deplasgdd = z_e * daeledd;
    let deplasgdt = -eplasg * tempinv;
    let deplasgda = z_e * daeleda;
    let deplasgdz = z_e * daeledz;
    let deplasgddd = y_e * daeledd * daeledd + z_e * daeleddd;
    let deplasgddt = -deplasgdd * tempinv;
    let deplasgdda = y_e * daeleda * daeledd + z_e * daeledda;
    let deplasgddz = y_e * daeledz * daeledd + z_e * daeleddz;
    let deplasgdtt = 2.0 * eplasg * tempinv * tempinv;
    let deplasgdta = -deplasgda * tempinv;
    let deplasgdtz = -deplasgdz * tempinv;
    let deplasgdaa = y_e * daeleda * daeleda + z_e * daeledaa;
    let deplasgdaz = y_e * daeledz * daeleda + z_e * daeledaz;
    let deplasgdzz = y_e * daeledz * daeledz + z_e * daeledzz;

    let x_z = zbar.powf(third);
    let z_z = x_z.powi(5);
    let ww_z = fiveth * x_z * x_z;
    let plasg_val = z_z * eplasg;
    if plasg_val < 0.0 {
        return Err(-1);
    }
    let plasginv = 1.0 / plasg_val;
    let plasg = HelmThermoVar {
        val: plasg_val,
        dd: z_z * deplasgdd,
        dt: z_z * deplasgdt,
        da: z_z * deplasgda,
        dz: z_z * deplasgdz + ww_z * eplasg,
        ddd: z_z * deplasgddd,
        ddt: z_z * deplasgddt,
        dda: z_z * deplasgdda,
        ddz: z_z * deplasgddz + ww_z * deplasgdd,
        dtt: z_z * deplasgdtt,
        dta: z_z * deplasgdta,
        dtz: z_z * deplasgdtz + ww_z * deplasgdt,
        daa: z_z * deplasgdaa,
        daz: z_z * deplasgdaz + ww_z * deplasgda,
        dzz: z_z * deplasgdzz
            + 2.0 * ww_z * deplasgdz
            + (teninth / x_z) * eplasg,
    };

    let (mut pcoul, mut ecoul, mut scoul) = if plasg_val >= 1.0 {
        eval_strong_coulomb(den, deninv, ytot1, pion, &plasg, plasginv)
    } else {
        eval_weak_coulomb(deninv, ytot1, pion, &plasg, plasginv)
    };

    if temp <= coulomb_temp_cut && den <= coulomb_den_cut {
        let (coulfactor, dcfdt) = if temp <= 0.1 * coulomb_temp_cut {
            (0.0, 0.0)
        } else {
            let ratio = coulomb_temp_cut / temp;
            let cf = (1.0 - ratio * ratio).exp();
            (cf, cf * 2.0 * ratio * ratio / temp)
        };
        for var in [&mut pcoul, &mut ecoul, &mut scoul] {
            var.dd *= coulfactor;
            var.dt = coulfactor * var.dt + dcfdt * var.val;
            var.da *= coulfactor;
            var.dz *= coulfactor;
            var.val *= coulfactor;
        }
    }

    Ok((pcoul, ecoul, scoul, plasg_val))
}

fn eval_strong_coulomb(
    den: f64,
    deninv: f64,
    ytot1: f64,
    pion: &HelmThermoVar,
    plasg: &HelmThermoVar,
    plasginv: f64,
) -> (HelmThermoVar, HelmThermoVar, HelmThermoVar) {
    let third = 1.0 / 3.0;
    let p1 = plasg.val.sqrt().sqrt();
    let p2 = 1.0 / p1;
    let p3 = p1 * plasginv;
    let p4 = p2 * plasginv;
    let p5 = p3 * plasginv;
    let p6 = p4 * plasginv;

    let u0 = A1 * plasg.val + B1 * p1 + C1 * p2 + D1;
    let du0 = A1 + 0.25 * B1 * p3 - 0.25 * C1 * p4;
    let ddu0 = -0.1875 * B1 * p5 + 0.3125 * C1 * p6;

    let z = pion.val * deninv;
    let ecoul_val = z * u0;
    let x = deninv * u0;
    let y = deninv * du0;
    let ww = z * du0;
    let dfk = z * ddu0;

    let ecoul = HelmThermoVar {
        val: ecoul_val,
        dd: pion.dd * x - z * x + ww * plasg.dd,
        dt: pion.dt * x + ww * plasg.dt,
        da: pion.da * x + ww * plasg.da,
        dz: pion.dz * x + ww * plasg.dz,
        ddd: pion.ddd * x
            + 2.0 * pion.dd * (y * plasg.dd - deninv * x)
            + z * (2.0 * deninv * x
                + (ddu0 * plasg.dd - 2.0 * y) * plasg.dd
                + du0 * plasg.ddd),
        ddt: pion.ddt * x + pion.dd * y * plasg.dt
            - pion.dt * deninv * x
            - z * y * plasg.dt
            + (pion.dt * y + dfk * plasg.dt) * plasg.dd
            + ww * plasg.ddt,
        dda: pion.dda * x + pion.dd * y * plasg.da
            - pion.da * deninv * x
            - z * y * plasg.da
            + (pion.da * y + dfk * plasg.da) * plasg.dd
            + ww * plasg.dda,
        ddz: pion.ddz * x + pion.dd * y * plasg.dz
            - pion.dz * deninv * x
            - z * y * plasg.dz
            + (pion.dz * y + dfk * plasg.dz) * plasg.dd
            + ww * plasg.ddz,
        dtt: pion.dtt * x
            + (2.0 * pion.dt * y + dfk * plasg.dt) * plasg.dt
            + ww * plasg.dtt,
        dta: pion.dta * x
            + pion.dt * y * plasg.da
            + (pion.da * y + dfk * plasg.da) * plasg.dt
            + ww * plasg.dta,
        dtz: pion.dtz * x
            + pion.dt * y * plasg.dz
            + (pion.dz * y + dfk * plasg.dz) * plasg.dt
            + ww * plasg.dtz,
        daa: pion.daa * x
            + (2.0 * pion.da * y + dfk * plasg.da) * plasg.da
            + ww * plasg.daa,
        daz: pion.daz * x
            + pion.da * y * plasg.dz
            + (pion.dz * y + dfk * plasg.dz) * plasg.da
            + ww * plasg.daz,
        dzz: pion.dzz * x
            + (2.0 * pion.dz * y + dfk * plasg.dz) * plasg.dz
            + ww * plasg.dzz,
    };

    let y_p = third * den;
    let pcoul = HelmThermoVar {
        val: y_p * ecoul.val,
        dd: third * ecoul.val + y_p * ecoul.dd,
        dt: y_p * ecoul.dt,
        da: y_p * ecoul.da,
        dz: y_p * ecoul.dz,
        ddd: 2.0 * third * ecoul.dd + y_p * ecoul.ddd,
        ddt: third * ecoul.dt + y_p * ecoul.ddt,
        dda: third * ecoul.da + y_p * ecoul.dda,
        ddz: third * ecoul.dz + y_p * ecoul.ddz,
        dtt: y_p * ecoul.dtt,
        dta: y_p * ecoul.dta,
        dtz: y_p * ecoul.dtz,
        daa: y_p * ecoul.daa,
        daz: y_p * ecoul.daz,
        dzz: y_p * ecoul.dzz,
    };

    let u0_s = 3.0 * B1 * p1 - 5.0 * C1 * p2 + D1 * (plasg.val.ln() - 1.0) - E1;
    let du0_s = 0.75 * B1 * p3 + 1.25 * C1 * p4 + D1 * plasginv;
    let ddu0_s =
        -0.5625 * B1 * p5 - 1.5625 * C1 * p6 - D1 * plasginv * plasginv;
    let scoul = build_scoul(u0_s, du0_s, ddu0_s, ytot1, plasg, true);

    (pcoul, ecoul, scoul)
}

fn build_scoul(
    u0_s: f64,
    du0_s: f64,
    ddu0_s: f64,
    ytot1: f64,
    plasg: &HelmThermoVar,
    strong_branch: bool,
) -> HelmThermoVar {
    let z_s = -AVO * ytot1 * KERG;
    let scoul_val = z_s * u0_s;
    let ww_s = z_s * du0_s;
    let x_s = z_s * ddu0_s;
    let lead = if strong_branch { x_s } else { ww_s };
    HelmThermoVar {
        val: scoul_val,
        dd: ww_s * plasg.dd,
        dt: ww_s * plasg.dt,
        da: ww_s * plasg.da - scoul_val * ytot1,
        dz: ww_s * plasg.dz,
        ddd: x_s * plasg.dd * plasg.dd + ww_s * plasg.ddd,
        ddt: x_s * plasg.dt * plasg.dd + ww_s * plasg.ddt,
        dda: x_s * plasg.da * plasg.dd + ww_s * plasg.dda
            - lead * ytot1 * plasg.dd,
        ddz: x_s * plasg.dz * plasg.dd + ww_s * plasg.ddz,
        dtt: x_s * plasg.dt * plasg.dt + ww_s * plasg.dtt,
        dta: x_s * plasg.da * plasg.dt + ww_s * plasg.dta
            - lead * ytot1 * plasg.dt,
        dtz: x_s * plasg.dz * plasg.dt + ww_s * plasg.dtz,
        daa: x_s * plasg.da * plasg.da + ww_s * plasg.daa
            - lead * ytot1 * plasg.da
            - ww_s * plasg.da * ytot1
            + 2.0 * scoul_val * ytot1 * ytot1,
        daz: x_s * plasg.dz * plasg.da + ww_s * plasg.daz
            - ww_s * plasg.dz * ytot1,
        dzz: x_s * plasg.dz * plasg.dz + ww_s * plasg.dzz,
    }
}

fn eval_weak_coulomb(
    deninv: f64,
    ytot1: f64,
    pion: &HelmThermoVar,
    plasg: &HelmThermoVar,
    plasginv: f64,
) -> (HelmThermoVar, HelmThermoVar, HelmThermoVar) {
    let third = 1.0 / 3.0;
    let x_p = plasg.val.sqrt();
    let p1 = plasg.val * x_p;
    let p2 = plasg.val.powf(B2);
    let p3 = x_p;
    let p4 = p2 * plasginv;
    let p5 = p3 * plasginv;
    let p6 = p4 * plasginv;

    let u0 = C2 * p1 - third * A2 * p2;
    let du0 = 1.5 * C2 * p3 - third * A2 * B2 * p4;
    let ddu0 = 0.75 * C2 * p5 - third * A2 * B2 * (B2 - 1.0) * p6;

    let pcoul_val = -pion.val * u0;
    let x = pion.val * du0;
    let y = pion.val * ddu0;
    let pcoul = HelmThermoVar {
        val: pcoul_val,
        dd: -pion.dd * u0 - x * plasg.dd,
        dt: -pion.dt * u0 - x * plasg.dt,
        da: -pion.da * u0 - x * plasg.da,
        dz: -pion.dz * u0 - x * plasg.dz,
        ddd: -pion.ddd * u0
            - (2.0 * pion.dd * du0 + y * plasg.dd) * plasg.dd
            - x * plasg.ddd,
        ddt: -pion.ddt * u0
            - pion.dd * du0 * plasg.dt
            - (pion.dt * du0 + y * plasg.dt) * plasg.dd
            - x * plasg.ddt,
        dda: -pion.dda * u0
            - pion.dd * du0 * plasg.da
            - (pion.da * du0 + y * plasg.da) * plasg.dd
            - x * plasg.dda,
        ddz: -pion.ddz * u0
            - pion.dd * du0 * plasg.dz
            - (pion.dz * du0 + y * plasg.dz) * plasg.dd
            - x * plasg.ddz,
        dtt: -pion.dtt * u0
            - (2.0 * pion.dt * du0 + y * plasg.dt) * plasg.dt
            - x * plasg.dtt,
        dta: -pion.dta * u0
            - pion.dt * du0 * plasg.da
            - (pion.da * du0 + y * plasg.da) * plasg.dt
            - x * plasg.dta,
        dtz: -pion.dtz * u0
            - pion.dt * du0 * plasg.dz
            - (pion.dz * du0 + y * plasg.dz) * plasg.dt
            - x * plasg.dtz,
        daa: -pion.daa * u0
            - (2.0 * pion.da * du0 + y * plasg.da) * plasg.da
            - x * plasg.daa,
        daz: -pion.daz * u0
            - pion.da * du0 * plasg.dz
            - (pion.dz * du0 + y * plasg.dz) * plasg.da
            - x * plasg.daz,
        dzz: -pion.dzz * u0
            - (2.0 * pion.dz * du0 + y * plasg.dz) * plasg.dz
            - x * plasg.dzz,
    };

    let z_e = 3.0 * deninv;
    let y_e = -z_e * deninv;
    let ecoul_val = z_e * pcoul.val;
    let x_e = deninv * deninv;
    let ecoul = HelmThermoVar {
        val: ecoul_val,
        dd: z_e * pcoul.dd - ecoul_val * deninv,
        dt: z_e * pcoul.dt,
        da: z_e * pcoul.da,
        dz: z_e * pcoul.dz,
        ddd: z_e * pcoul.ddd
            + y_e * pcoul.dd
            + (2.0 * ecoul_val - 3.0 * pcoul.dd) * x_e,
        ddt: z_e * pcoul.ddt - 3.0 * pcoul.dt * x_e,
        dda: z_e * pcoul.dda - 3.0 * pcoul.da * x_e,
        ddz: z_e * pcoul.ddz - 3.0 * pcoul.dz * x_e,
        dtt: z_e * pcoul.dtt,
        dta: z_e * pcoul.dta,
        dtz: z_e * pcoul.dtz,
        daa: z_e * pcoul.daa,
        daz: z_e * pcoul.daz,
        dzz: z_e * pcoul.dzz,
    };

    let u0_s = C2 * p1 - (A2 / B2) * (B2 - 1.0) * p2;
    let du0_s = 1.5 * C2 * p3 - A2 * (B2 - 1.0) * p4;
    let ddu0_s = 0.75 * C2 * p5 - A2 * (B2 - 1.0) * (B2 - 1.0) * p6;
    let scoul = build_scoul(u0_s, du0_s, ddu0_s, ytot1, plasg, false);

    (pcoul, ecoul, scoul)
}
