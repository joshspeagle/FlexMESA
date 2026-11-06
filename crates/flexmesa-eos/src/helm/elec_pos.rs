//! Electron-positron biquintic/bicubic Helmholtz free-energy evaluation
//! (`eos/private/helm_electron_positron.dek` and
//! `eos/private/skye_ideal.f90::compute_ideal_ele`).

use flexmesa_autodiff::AutoDiffReal2VarOrder3;
use flexmesa_const::{KERG, MP};

use crate::helm::components::HelmThermoVar;
use crate::helm::polynomials::{CubicBasis1D, QuinticBasis1D, h3, h5};
use crate::helm::table::HelmTable;

/// Evaluates electron-positron pressure, specific energy, specific entropy,
/// chemical potential, and number density
/// (`eos/private/helm_electron_positron.dek`).
///
/// # Errors
/// Returns `Err(-1)` if `dsepdt < 0.0` (unphysical negative entropy
/// derivative).
#[allow(clippy::too_many_lines)]
pub fn eval_electron_positron(
    table: &HelmTable,
    temp: f64,
    log_temp: f64,
    den: f64,
    abar: f64,
    zbar: f64,
    include_elec_pos: bool,
) -> Result<
    (
        HelmThermoVar,
        HelmThermoVar,
        HelmThermoVar,
        HelmThermoVar,
        HelmThermoVar,
    ),
    i32,
> {
    if !include_elec_pos {
        return Ok((
            HelmThermoVar::default(),
            HelmThermoVar::default(),
            HelmThermoVar::default(),
            HelmThermoVar::default(),
            HelmThermoVar::default(),
        ));
    }

    let ytot1 = 1.0 / abar;
    let ye = ytot1 * zbar;
    let din = ye * den;
    let dindd = ye;
    let dinda = -din * ytot1;
    let dindz = den * ytot1;
    let dindda = -ye * ytot1;
    let dinddz = ytot1;
    let ww_a = ytot1 * ytot1;
    let dindaa = 2.0 * din * ww_a;
    let dindaz = -den * ww_a;
    let dinddaz = -ww_a;

    let c = table.cell(log_temp, din);
    let xt = ((temp - c.t_lo) * c.dti).max(0.0);
    let xd = ((din - c.d_lo) * c.ddi).max(0.0);
    let qt = QuinticBasis1D::eval(xt, c.dth, c.dt2, c.dti, c.dt2i, c.dt3i);
    let qd = QuinticBasis1D::eval(xd, c.dd, c.dd2, c.ddi, c.dd2i, c.dd3i);
    let ct = CubicBasis1D::eval(xt, c.dth, c.dti, c.dt2i, c.dt3i);
    let cd = CubicBasis1D::eval(xd, c.dd, c.ddi, c.dd2i, c.dd3i);

    let free = h5(&c.fi_f, &qt.w0, &qd.w0);
    let df_d = h5(&c.fi_f, &qt.w0, &qd.w1);
    let df_t = h5(&c.fi_f, &qt.w1, &qd.w0);
    let df_dd = h5(&c.fi_f, &qt.w0, &qd.w2);
    let df_tt = h5(&c.fi_f, &qt.w2, &qd.w0);
    let df_dt = h5(&c.fi_f, &qt.w1, &qd.w1);
    let df_ddd = h5(&c.fi_f, &qt.w0, &qd.w3);
    let df_ddt = h5(&c.fi_f, &qt.w1, &qd.w2);
    let df_dtt = h5(&c.fi_f, &qt.w2, &qd.w1);
    let df_ttt = h5(&c.fi_f, &qt.w3, &qd.w0);

    let dpepdd_in = h3(&c.fi_dpdf, &ct.w0, &cd.w0).max(1.0e-30);
    let dpepddd_in = h3(&c.fi_dpdf, &ct.w0, &cd.w1);
    let dpepddt_in = h3(&c.fi_dpdf, &ct.w1, &cd.w0);

    let etaele_val = h3(&c.fi_ef, &ct.w0, &cd.w0);
    let x_ef = h3(&c.fi_ef, &ct.w0, &cd.w1);
    let y_ef = h3(&c.fi_ef, &ct.w0, &cd.w2);
    let z_ef = h3(&c.fi_ef, &ct.w1, &cd.w1);
    let etaele = HelmThermoVar {
        val: etaele_val,
        dd: ye * x_ef,
        dt: h3(&c.fi_ef, &ct.w1, &cd.w0),
        da: -x_ef * din * ytot1,
        dz: x_ef * den * ytot1,
        ddd: ye * ye * y_ef,
        ddt: ye * z_ef,
        dda: -ye * ytot1 * x_ef + ye * y_ef * dinda,
        ddz: ytot1 * x_ef + ye * y_ef * dindz,
        dtt: h3(&c.fi_ef, &ct.w2, &cd.w0),
        dta: z_ef * dinda,
        dtz: z_ef * dindz,
        daa: (y_ef * din + 2.0 * x_ef) * din * ww_a,
        daz: -(y_ef * dindz * din + x_ef * den * ytot1) * ytot1,
        dzz: y_ef * dindz * den * ytot1,
    };

    let xnefer_val = h3(&c.fi_xf, &ct.w0, &cd.w0);
    let x_xf = h3(&c.fi_xf, &ct.w0, &cd.w1).max(1.0e-30);
    let y_xf = h3(&c.fi_xf, &ct.w0, &cd.w2);
    let z_xf = h3(&c.fi_xf, &ct.w1, &cd.w1);
    let xnefer = HelmThermoVar {
        val: xnefer_val,
        dd: ye * x_xf,
        dt: h3(&c.fi_xf, &ct.w1, &cd.w0),
        da: -x_xf * din * ytot1,
        dz: x_xf * den * ytot1,
        ddd: ye * ye * y_xf,
        ddt: ye * z_xf,
        dda: -ye * ytot1 * x_xf + ye * y_xf * dinda,
        ddz: ytot1 * x_xf + ye * y_xf * dindz,
        dtt: h3(&c.fi_xf, &ct.w2, &cd.w0),
        dta: z_xf * dinda,
        dtz: z_xf * dindz,
        daa: (y_xf * din + 2.0 * x_xf) * din * ww_a,
        daz: -(y_xf * dindz * din + x_xf * den * ytot1) * ytot1,
        dzz: y_xf * dindz * den * ytot1,
    };

    let din2 = din * din;
    let pele_val = din2 * df_d;
    let (ww, zz, y_p) = if pele_val / (din * KERG * temp / (abar * MP)) < 1.0e6
    {
        (
            din2 * df_dd + 2.0 * din * df_d,
            din2 * df_ddd + 4.0 * din * df_dd + 2.0 * df_d,
            din2 * df_ddt + 2.0 * din * df_dt,
        )
    } else {
        (dpepdd_in, dpepddd_in, dpepddt_in)
    };
    let pele = HelmThermoVar {
        val: pele_val,
        dd: ww * dindd,
        dt: din2 * df_dt,
        da: ww * dinda,
        dz: ww * dindz,
        ddd: zz * dindd * dindd,
        ddt: y_p * dindd,
        dda: (zz * dindd + ww) * dindda,
        ddz: (zz * dindd + ww) * dinddz,
        dtt: din2 * df_dtt,
        dta: y_p * dinda,
        dtz: y_p * dindz,
        daa: (zz * dinda + ww) * dindaa,
        daz: (zz * dinda + ww) * dindaz,
        dzz: zz * dindz * dindz,
    };

    let ye2 = ye * ye;
    let y_s = ye * ytot1;
    let dsepdt = -df_tt * ye;
    if dsepdt < 0.0 {
        return Err(-1);
    }
    let sele = HelmThermoVar {
        val: -df_t * ye,
        dd: -df_dt * ye2,
        dt: dsepdt,
        da: -df_dt * dinda * ye + df_t * y_s,
        dz: -df_dt * dindz * ye - df_t * ytot1,
        ddd: -df_ddt * ye2 * ye,
        ddt: -df_dtt * ye2,
        dda: -df_ddt * dinda * ye2 + 2.0 * df_dt * ye2 * ytot1,
        ddz: -df_ddt * dindz * ye2 - 2.0 * df_dt * y_s,
        dtt: -df_ttt * ye,
        dta: -df_dtt * dinda * ye + df_tt * y_s,
        dtz: -df_dtt * dindz * ye - df_tt * ytot1,
        daa: -df_ddt * dinda * dinda * ye
            - df_dt * (dindaa * ye - 2.0 * dinda * ye * ytot1)
            - 2.0 * df_t * y_s * ytot1,
        daz: -df_ddt * dindz * dinda * ye
            - df_dt * (dindaz * ye + dinda * ytot1 + dindz * dindda)
            - df_t * dinddaz,
        dzz: -df_ddt * dindz * dindz * ye
            - df_dt * (dindz * ytot1 + dindz * dinddz),
    };

    let eele = HelmThermoVar {
        val: ye * free + temp * sele.val,
        dd: ye * df_d * dindd + temp * sele.dd,
        dt: temp * sele.dt,
        da: -y_s * free + ye * df_d * dinda + temp * sele.da,
        dz: ytot1 * free + ye * df_d * dindz + temp * sele.dz,
        ddd: ye * df_dd * dindd * dindd + temp * sele.ddd,
        ddt: ye * df_dt * dindd + sele.dd + temp * sele.ddt,
        dda: -y_s * df_d * dindd
            + ye * df_dd * dinda * dindd
            + ye * df_d * dindda
            + temp * sele.dda,
        ddz: ytot1 * df_d * dindd
            + ye * df_dd * dindz * dindd
            + ye * df_d * dinddz
            + temp * sele.ddz,
        dtt: sele.dt + temp * sele.dtt,
        dta: temp * sele.dta,
        dtz: temp * sele.dtz,
        daa: 2.0 * y_s * (ytot1 * free - df_d * dinda)
            + ye * (df_d * dindaa + df_dd * dinda * dinda)
            + temp * sele.daa,
        daz: -ytot1 * ytot1 * free - y_s * df_d * dindz
            + ytot1 * df_d * dinda
            + ye * df_dd * dindz * dinda
            + ye * df_d * dindaz
            + temp * sele.daz,
        dzz: (2.0 * ytot1 * df_d + ye * df_dd * dindz) * dindz
            + temp * sele.dzz,
    };

    Ok((pele, eele, sele, etaele, xnefer))
}

impl HelmTable {
    /// Evaluates the `HELM` electron-positron free energy $F_{\mathrm{ele}}$,
    /// degeneracy parameter $\eta_{\mathrm{ele}}$, and electron-positron number
    /// density $n_e$ as `AutoDiffReal2VarOrder3` with `val1 = T`, `val2 = Rho`,
    /// matching `eos/private/skye_ideal.f90::compute_ideal_ele`.
    #[must_use]
    pub fn eval_electron_positron_order3(
        &self,
        temp: f64,
        log_temp: f64,
        den: f64,
        abar: f64,
        zbar: f64,
    ) -> (
        AutoDiffReal2VarOrder3,
        AutoDiffReal2VarOrder3,
        AutoDiffReal2VarOrder3,
    ) {
        let ye = zbar / abar;
        let din = ye * den;
        let c = self.cell(log_temp, din);
        let xt = ((temp - c.t_lo) * c.dti).max(0.0);
        let xd = ((din - c.d_lo) * c.ddi).max(0.0);

        let qt = QuinticBasis1D::eval(xt, c.dth, c.dt2, c.dti, c.dt2i, c.dt3i);
        let qd = QuinticBasis1D::eval(xd, c.dd, c.dd2, c.ddi, c.dd2i, c.dd3i);
        let ct = CubicBasis1D::eval(xt, c.dth, c.dti, c.dt2i, c.dt3i);
        let cd = CubicBasis1D::eval(xd, c.dd, c.ddi, c.dd2i, c.dd3i);

        let ye2 = ye * ye;
        let ye3 = ye2 * ye;
        let ye4 = ye2 * ye2;

        let f_ele = AutoDiffReal2VarOrder3 {
            val: h5(&c.fi_f, &qt.w0, &qd.w0) * ye,
            d1val1: h5(&c.fi_f, &qt.w1, &qd.w0) * ye,
            d1val2: h5(&c.fi_f, &qt.w0, &qd.w1) * ye2,
            d2val1: h5(&c.fi_f, &qt.w2, &qd.w0) * ye,
            d1val1_d1val2: h5(&c.fi_f, &qt.w1, &qd.w1) * ye2,
            d2val2: h5(&c.fi_f, &qt.w0, &qd.w2) * ye3,
            d3val1: h5(&c.fi_f, &qt.w3, &qd.w0) * ye,
            d2val1_d1val2: h5(&c.fi_f, &qt.w2, &qd.w1) * ye2,
            d1val1_d2val2: h5(&c.fi_f, &qt.w1, &qd.w2) * ye3,
            d3val2: h5(&c.fi_f, &qt.w0, &qd.w3) * ye4,
        };

        let eta_ele = AutoDiffReal2VarOrder3 {
            val: h3(&c.fi_ef, &ct.w0, &cd.w0),
            d1val1: h3(&c.fi_ef, &ct.w1, &cd.w0),
            d1val2: h3(&c.fi_ef, &ct.w0, &cd.w1) * ye,
            d2val1: h3(&c.fi_ef, &ct.w2, &cd.w0),
            d1val1_d1val2: h3(&c.fi_ef, &ct.w1, &cd.w1) * ye,
            d2val2: h3(&c.fi_ef, &ct.w0, &cd.w2) * ye2,
            d3val1: h3(&c.fi_ef, &ct.w3, &cd.w0),
            d2val1_d1val2: h3(&c.fi_ef, &ct.w2, &cd.w1) * ye,
            d1val1_d2val2: h3(&c.fi_ef, &ct.w1, &cd.w2) * ye2,
            d3val2: h3(&c.fi_ef, &ct.w0, &cd.w3) * ye3,
        };

        let dxne_ddin = h3(&c.fi_xf, &ct.w0, &cd.w1).max(1.0e-30);
        let xne_fer = AutoDiffReal2VarOrder3 {
            val: h3(&c.fi_xf, &ct.w0, &cd.w0),
            d1val1: h3(&c.fi_xf, &ct.w1, &cd.w0),
            d1val2: dxne_ddin * ye,
            d2val1: h3(&c.fi_xf, &ct.w2, &cd.w0),
            d1val1_d1val2: h3(&c.fi_xf, &ct.w1, &cd.w1) * ye,
            d2val2: h3(&c.fi_xf, &ct.w0, &cd.w2) * ye2,
            d3val1: h3(&c.fi_xf, &ct.w3, &cd.w0),
            d2val1_d1val2: h3(&c.fi_xf, &ct.w2, &cd.w1) * ye,
            d1val1_d2val2: h3(&c.fi_xf, &ct.w1, &cd.w2) * ye2,
            d3val2: h3(&c.fi_xf, &ct.w0, &cd.w3) * ye3,
        };

        (f_ele, eta_ele, xne_fer)
    }
}
