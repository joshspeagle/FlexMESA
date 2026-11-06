//! Radiation and ideal-ion thermodynamic components for the `HELM` equation of
//! state (`eos/private/helm_radiation.dek` and
//! `eos/private/helm_ideal_ions.dek`).
//!
//! Each thermodynamic quantity (`P`, `E`, `S`, `xni`, `etaele`, `xnefer`) is
//! represented by [`HelmThermoVar`], storing its scalar value together with all
//! 4 first partial derivatives and 10 second partial derivatives with respect
//! to $(\rho, T, \bar{A}, \bar{Z})$.

use flexmesa_const::{AVO, CRAD, KERG};

/// Sackur-Tetrode constant `sifac = 8.6322745944370191e-45` from
/// `helm_initialize_local_variables.dek`.
pub const HELM_SIFAC: f64 = 8.6322745944370191e-45;

/// A scalar thermodynamic variable together with its first and second partial
/// derivatives with respect to $(\rho, T, \bar{A}, \bar{Z})$.
#[derive(Debug, Clone, Copy, Default)]
pub struct HelmThermoVar {
    /// Function value.
    pub val: f64,
    /// First derivative $\partial / \partial \rho$.
    pub dd: f64,
    /// First derivative $\partial / \partial T$.
    pub dt: f64,
    /// First derivative $\partial / \partial \bar{A}$.
    pub da: f64,
    /// First derivative $\partial / \partial \bar{Z}$.
    pub dz: f64,
    /// Second derivative $\partial^2 / \partial \rho^2$.
    pub ddd: f64,
    /// Second derivative $\partial^2 / \partial \rho \partial T$.
    pub ddt: f64,
    /// Second derivative $\partial^2 / \partial \rho \partial \bar{A}$.
    pub dda: f64,
    /// Second derivative $\partial^2 / \partial \rho \partial \bar{Z}$.
    pub ddz: f64,
    /// Second derivative $\partial^2 / \partial T^2$.
    pub dtt: f64,
    /// Second derivative $\partial^2 / \partial T \partial \bar{A}$.
    pub dta: f64,
    /// Second derivative $\partial^2 / \partial T \partial \bar{Z}$.
    pub dtz: f64,
    /// Second derivative $\partial^2 / \partial \bar{A}^2$.
    pub daa: f64,
    /// Second derivative $\partial^2 / \partial \bar{A} \partial \bar{Z}$.
    pub daz: f64,
    /// Second derivative $\partial^2 / \partial \bar{Z}^2$.
    pub dzz: f64,
}

impl HelmThermoVar {
    /// Elementwise sum of two `HelmThermoVar` instances
    /// (`helm_sum_totals.dek`).
    #[inline]
    #[must_use]
    pub fn add(&self, rhs: &Self) -> Self {
        Self {
            val: self.val + rhs.val,
            dd: self.dd + rhs.dd,
            dt: self.dt + rhs.dt,
            da: self.da + rhs.da,
            dz: self.dz + rhs.dz,
            ddd: self.ddd + rhs.ddd,
            ddt: self.ddt + rhs.ddt,
            dda: self.dda + rhs.dda,
            ddz: self.ddz + rhs.ddz,
            dtt: self.dtt + rhs.dtt,
            dta: self.dta + rhs.dta,
            dtz: self.dtz + rhs.dtz,
            daa: self.daa + rhs.daa,
            daz: self.daz + rhs.daz,
            dzz: self.dzz + rhs.dzz,
        }
    }

    /// Scales the value and all derivatives by a constant `c`.
    #[inline]
    #[must_use]
    pub fn scale(&self, c: f64) -> Self {
        Self {
            val: self.val * c,
            dd: self.dd * c,
            dt: self.dt * c,
            da: self.da * c,
            dz: self.dz * c,
            ddd: self.ddd * c,
            ddt: self.ddt * c,
            dda: self.dda * c,
            ddz: self.ddz * c,
            dtt: self.dtt * c,
            dta: self.dta * c,
            dtz: self.dtz * c,
            daa: self.daa * c,
            daz: self.daz * c,
            dzz: self.dzz * c,
        }
    }
}

/// Evaluates blackbody radiation pressure, energy, and entropy
/// (`eos/private/helm_radiation.dek`).
#[must_use]
pub fn eval_radiation(
    temp: f64,
    den: f64,
    include_radiation: bool,
) -> (HelmThermoVar, HelmThermoVar, HelmThermoVar) {
    if !include_radiation {
        return (
            HelmThermoVar::default(),
            HelmThermoVar::default(),
            HelmThermoVar::default(),
        );
    }
    let third = 1.0 / 3.0;
    let tempinv = 1.0 / temp;
    let deninv = 1.0 / den;

    let prad_val = CRAD * third * temp.powi(4);
    let dpraddt = 4.0 * prad_val * tempinv;
    let dpraddtt = 4.0 * CRAD * temp * temp;
    let prad = HelmThermoVar {
        val: prad_val,
        dt: dpraddt,
        dtt: dpraddtt,
        ..HelmThermoVar::default()
    };

    let erad_val = 3.0 * prad_val * deninv;
    let deraddd = -erad_val * deninv;
    let deraddt = 3.0 * dpraddt * deninv;
    let deradddd = -2.0 * deraddd * deninv;
    let deradddt = -deraddt * deninv;
    let deraddtt = 3.0 * dpraddtt * deninv;
    let erad = HelmThermoVar {
        val: erad_val,
        dd: deraddd,
        dt: deraddt,
        ddd: deradddd,
        ddt: deradddt,
        dtt: deraddtt,
        ..HelmThermoVar::default()
    };

    let srad_val = (prad_val * deninv + erad_val) * tempinv;
    let dsraddd = (-prad_val * deninv * deninv + deraddd) * tempinv;
    let dsraddt = (dpraddt * deninv + deraddt - srad_val) * tempinv;
    let dsradddd =
        (2.0 * prad_val * deninv * deninv * deninv + deradddd) * tempinv;
    let dsradddt =
        (-dpraddt * deninv * deninv + deradddt - dsraddd) * tempinv;
    let dsraddtt =
        (dpraddtt * deninv + deraddtt - 2.0 * dsraddt) * tempinv;
    let srad = HelmThermoVar {
        val: srad_val,
        dd: dsraddd,
        dt: dsraddt,
        ddd: dsradddd,
        ddt: dsradddt,
        dtt: dsraddtt,
        ..HelmThermoVar::default()
    };
    (prad, erad, srad)
}

/// Evaluates ideal-ion pressure, specific energy, specific entropy, ion number
/// density, and ion degeneracy (`eos/private/helm_ideal_ions.dek`).
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn eval_ideal_ions(
    temp: f64,
    den: f64,
    abar: f64,
) -> (
    HelmThermoVar,
    HelmThermoVar,
    HelmThermoVar,
    HelmThermoVar,
    HelmThermoVar,
) {
    let tempinv = 1.0 / temp;
    let deninv = 1.0 / den;
    let ytot1 = 1.0 / abar;
    let kt = KERG * temp;
    let ktinv = 1.0 / kt;
    let kergavo = KERG * AVO;

    let xni_val = AVO * ytot1 * den;
    let dxnidd = AVO * ytot1;
    let dxnida = -xni_val * ytot1;
    let dxnidda = -dxnidd * ytot1;
    let dxnidaa = -2.0 * dxnida * ytot1;
    let xni = HelmThermoVar {
        val: xni_val,
        dd: dxnidd,
        da: dxnida,
        dda: dxnidda,
        daa: dxnidaa,
        ..HelmThermoVar::default()
    };

    let pion_val = (xni_val * kt).max(1.0e-99);
    let dpiondd = dxnidd * kt;
    let dpiondt = xni_val * KERG;
    let dpionda = -pion_val * ytot1;
    let dpionddt = dxnidd * KERG;
    let dpiondda = dxnidda * kt;
    let dpiondta = dxnida * KERG;
    let dpiondaa = -2.0 * dpionda * ytot1;
    let pion = HelmThermoVar {
        val: pion_val,
        dd: dpiondd,
        dt: dpiondt,
        da: dpionda,
        ddt: dpionddt,
        dda: dpiondda,
        dta: dpiondta,
        daa: dpiondaa,
        ..HelmThermoVar::default()
    };

    let y_e = 1.5 * deninv;
    let eion_val = y_e * pion_val;
    let deiondt = y_e * dpiondt;
    let deionda = y_e * dpionda;
    let deiondda = y_e * dpiondda - deionda * deninv;
    let deiondta = y_e * dpiondta;
    let deiondaa = y_e * dpiondaa;
    let eion = HelmThermoVar {
        val: eion_val,
        dt: deiondt,
        da: deionda,
        dda: deiondda,
        dta: deiondta,
        daa: deiondaa,
        ..HelmThermoVar::default()
    };

    let y_s = ktinv * ytot1;
    let yy_s = y_s * y_s.sqrt();
    let z_s = xni_val * HELM_SIFAC * yy_s;
    let etaion_val = z_s.ln();
    let xx = 1.0 / xni_val;
    let detaidd = dxnidd * xx;
    let detaidt = -1.5 * tempinv;
    let detaida = dxnida * xx - 1.5 * ytot1;
    let detaiddd = -detaidd * detaidd;
    let detaidda = dxnidda * xx - dxnidd * dxnida * xx * xx;
    let detaidtt = -detaidt * tempinv;
    let detaidaa =
        (dxnidaa - dxnida * dxnida * xx) * xx + 1.5 * ytot1 * ytot1;
    let etaion = HelmThermoVar {
        val: etaion_val,
        dd: detaidd,
        dt: detaidt,
        da: detaida,
        ddd: detaiddd,
        dda: detaidda,
        dtt: detaidtt,
        daa: detaidaa,
        ..HelmThermoVar::default()
    };

    let y = kergavo * ytot1;
    let z = y * ytot1;
    let mut sion_val =
        (eion_val + pion_val * deninv) * tempinv - etaion_val * y;
    let mut dsiondd =
        (dpiondd - pion_val * deninv) * deninv * tempinv - detaidd * y;
    let mut dsiondt = ((deiondt + dpiondt * deninv)
        - (eion_val + pion_val * deninv) * tempinv)
        * tempinv
        - detaidt * y;
    let mut dsionda =
        (deionda + dpionda * deninv) * tempinv + etaion_val * z - detaida * y;
    let mut dsionddd =
        2.0 * (pion_val * deninv - dpiondd) * deninv * deninv * tempinv
            - detaiddd * y;
    let mut dsionddt = 0.0;
    let mut dsiondda =
        (deiondda + (dpiondda - dpionda * deninv) * deninv) * tempinv
            + detaidd * z
            - detaidda * y;
    let mut dsiondtt = -2.0 * (deiondt + dpiondt * deninv) * tempinv * tempinv
        + 2.0 * (eion_val + pion_val * deninv) * tempinv * tempinv * tempinv
        - detaidtt * y;
    let mut dsiondta = ((deiondta + dpiondta * deninv)
        - (deionda + dpionda * deninv) * tempinv)
        * tempinv
        + detaidt * z;
    let mut dsiondaa = (deiondaa + dpiondaa * deninv) * tempinv
        - 2.0 * etaion_val * z * ytot1
        + 2.0 * detaida * z
        - detaidaa * y;

    if sion_val < 0.0 {
        let s0 = 1.0e3;
        let denom = s0 + sion_val;
        let denom2 = denom * denom;
        let denom3 = denom2 * denom;
        let s0_sq = s0 * s0;
        sion_val = s0 * sion_val / denom;
        dsionddd =
            s0_sq * (dsionddd / denom2 - 2.0 * dsiondd * dsiondd / denom3);
        dsionddt =
            s0_sq * (dsionddt / denom2 - 2.0 * dsiondd * dsiondt / denom3);
        dsiondda =
            s0_sq * (dsiondda / denom2 - 2.0 * dsiondd * dsionda / denom3);
        dsiondtt =
            s0_sq * (dsiondtt / denom2 - 2.0 * dsiondt * dsiondt / denom3);
        dsiondta =
            s0_sq * (dsiondta / denom2 - 2.0 * dsiondt * dsionda / denom3);
        dsiondaa =
            s0_sq * (dsiondaa / denom2 - 2.0 * dsionda * dsionda / denom3);
        dsiondd = s0_sq * dsiondd / denom2;
        dsiondt = s0_sq * dsiondt / denom2;
        dsionda = s0_sq * dsionda / denom2;
    }

    let sion = HelmThermoVar {
        val: sion_val,
        dd: dsiondd,
        dt: dsiondt,
        da: dsionda,
        ddd: dsionddd,
        ddt: dsionddt,
        dda: dsiondda,
        dtt: dsiondtt,
        dta: dsiondta,
        daa: dsiondaa,
        ..HelmThermoVar::default()
    };

    (pion, eion, sion, xni, etaion)
}
