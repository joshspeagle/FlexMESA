// SPDX-License-Identifier: LGPL-3.0-or-later
//! Precomputed thermodynamic and composition powers (`NeuInputs`) and weak
//! interaction coupling constants for `flexmesa-neu`.
//!
//! Ports `type(inputs)` and `set_inputs` from `neu/private/mod_neu.f90`.
//!
//! # Precomputed Powers
//!
//! Computing powers of dimensionless temperature $\lambda = T / (5.9302 \times
//! 10^9\,\mathrm{K})$ and electron degeneracy parameter $\zeta = (10^{-9}\rho
//! Y_e)^{1/3} / \lambda$ once in [`NeuInputs::new`] avoids redundant divisions
//! and fractional powers across the five neutrino channels (`pair`, `plas`,
//! `phot`, `brem`, `reco`).

use flexmesa_const::{NUM_NEU_FAM, ONE_THIRD, WEINBERG_THETA};
use flexmesa_math::pow;

/// Minimum temperature ($T_{\min} = 10^7\,\mathrm{K}$) below which thermal
/// neutrino losses vanish identically (`Tmin_neu` in `neu_def.f90`).
pub const TMIN_NEU: f64 = 1.0e7;

/// Base-10 logarithm of [`TMIN_NEU`] (`log10Tmin_neu = 7.0` in `neu_def.f90`).
pub const LOG10_TMIN_NEU: f64 = 7.0;

/// Dimensionless relativistic electron temperature scale factor
/// $1 / 5.9302$ (`con1` in `mod_neu.f90`).
pub(crate) const CON1: f64 = 1.0 / 5.9302;

/// Vector weak coupling $C_V = \tfrac{1}{2} + 2 \sin^2\theta_W$.
pub(crate) const CV: f64 = 0.5 + 2.0 * WEINBERG_THETA;

/// Neutral-current vector coupling for $\mu, \tau$ neutrinos ($1 - C_V$).
pub(crate) const CVP: f64 = 1.0 - CV;

/// Axial-vector weak coupling $C_A = \tfrac{1}{2}$.
pub(crate) const CA: f64 = 0.5;

/// Neutral-current axial coupling for $\mu, \tau$ neutrinos ($1 - C_A$).
pub(crate) const CAP: f64 = 1.0 - CA;

/// Weak coupling combination $C_V^2 + (N_\nu - 1) C_V'^2$ (`tfac0`).
pub(crate) const TFAC0: f64 = CV * CV + (NUM_NEU_FAM - 1.0) * (CVP * CVP);

/// Weak coupling combination $C_V^2 + C_A^2 + (N_\nu - 1)(C_V'^2 + C_A'^2)$.
pub(crate) const TFAC1: f64 =
    CV * CV + CA * CA + (NUM_NEU_FAM - 1.0) * (CVP * CVP + CAP * CAP);

/// Weak coupling combination $C_V^2 - C_A^2 + (N_\nu - 1)(C_V'^2 - C_A'^2)$.
pub(crate) const TFAC2: f64 =
    CV * CV - CA * CA + (NUM_NEU_FAM - 1.0) * (CVP * CVP - CAP * CAP);

/// Ratio `TFAC2 / TFAC1` (`tfac3`).
pub(crate) const TFAC3: f64 = TFAC2 / TFAC1;

/// Half of `TFAC1` (`tfac4`).
pub(crate) const TFAC4: f64 = 0.5 * TFAC1;

/// Half of `TFAC2` (`tfac5`).
pub(crate) const TFAC5: f64 = 0.5 * TFAC2;

/// Recombination weak coupling combination (`tfac6`).
pub(crate) const TFAC6: f64 = CV * CV
    + 1.5 * CA * CA
    + (NUM_NEU_FAM - 1.0) * (CVP * CVP + 1.5 * CAP * CAP);

/// Precomputed thermodynamic and composition powers (`type(inputs)` in
/// `mod_neu.f90`).
#[derive(Debug, Clone, Copy)]
pub struct NeuInputs {
    /// Temperature $T$ ($\mathrm{K}$).
    pub temp: f64,
    /// Base-10 log of temperature $\log_{10}(T/\mathrm{K})$.
    pub logtemp: f64,
    /// Mass density $\rho$ ($\mathrm{g\,cm^{-3}}$).
    pub den: f64,
    /// Base-10 log of density $\log_{10}(\rho/\mathrm{g\,cm^{-3}})$.
    pub logden: f64,
    /// Scaled density $\rho_6 = 10^{-6}\rho$.
    pub den6: f64,
    /// Electron mole fraction $Y_e = \bar{Z}/\bar{A}$.
    pub ye: f64,
    /// Inverse density $1/\rho$.
    pub deni: f64,
    /// Inverse temperature $1/T$.
    pub tempi: f64,
    /// Inverse mean atomic weight $1/\bar{A}$.
    pub abari: f64,
    /// Inverse mean atomic charge $1/\bar{Z}$.
    pub zbari: f64,
    /// Mean atomic weight $\bar{A}$.
    pub abar: f64,
    /// Mean ionic charge $\bar{Z}$.
    pub zbar: f64,
    /// Scaled temperature $T_9 = 10^{-9} T$.
    pub t9: f64,
    /// Dimensionless temperature $\lambda = T_9 / 5.9302$.
    pub xl: f64,
    /// Derivative $d\lambda / dT$.
    pub xldt: f64,
    /// Square root $\lambda^{1/2}$.
    pub xlp5: f64,
    /// $\lambda^2$.
    pub xl2: f64,
    /// $\lambda^3$.
    pub xl3: f64,
    /// $\lambda^4$.
    pub xl4: f64,
    /// $\lambda^5$.
    pub xl5: f64,
    /// $\lambda^6$.
    pub xl6: f64,
    /// $\lambda^7$.
    pub xl7: f64,
    /// $\lambda^8$.
    pub xl8: f64,
    /// $\lambda^9$.
    pub xl9: f64,
    /// $\lambda^{-1/2}$.
    pub xlmp5: f64,
    /// $\lambda^{-1}$.
    pub xlm1: f64,
    /// $\lambda^{-2}$.
    pub xlm2: f64,
    /// $\lambda^{-3}$.
    pub xlm3: f64,
    /// $\lambda^{-4}$.
    pub xlm4: f64,
    /// Electron mass density $\rho Y_e$ (`rm`).
    pub rm: f64,
    /// $\partial(\rho Y_e)/\partial\rho = Y_e$.
    pub rmdd: f64,
    /// $\partial(\rho Y_e)/\partial\bar{A} = -\rho Y_e / \bar{A}$.
    pub rmda: f64,
    /// $\partial(\rho Y_e)/\partial\bar{Z} = \rho / \bar{A}$.
    pub rmdz: f64,
    /// Inverse $1 / (\rho Y_e)$.
    pub rmi: f64,
    /// Degeneracy variable $\zeta = (10^{-9}\rho Y_e)^{1/3} / \lambda$.
    pub zeta: f64,
    /// $\partial\zeta/\partial T$.
    pub zetadt: f64,
    /// $\partial\zeta/\partial\rho$.
    pub zetadd: f64,
    /// $\partial\zeta/\partial\bar{A}$.
    pub zetada: f64,
    /// $\partial\zeta/\partial\bar{Z}$.
    pub zetadz: f64,
    /// $\zeta^2$.
    pub zeta2: f64,
    /// $\zeta^3$.
    pub zeta3: f64,
}

impl NeuInputs {
    /// Populates [`NeuInputs`] from $(T, \log_{10} T, \rho, \log_{10} \rho,
    /// \bar{A}, \bar{Z})$ (`set_inputs` in `mod_neu.f90`).
    #[must_use]
    pub fn new(
        temp: f64,
        logtemp: f64,
        den: f64,
        logden: f64,
        abar: f64,
        zbar: f64,
    ) -> Self {
        let den6 = den * 1.0e-6;
        let deni = 1.0 / den;
        let tempi = 1.0 / temp;
        let abari = 1.0 / abar;
        let zbari = 1.0 / zbar;
        let ye = zbar * abari;
        let t9 = temp * 1.0e-9;
        let xl = t9 * CON1;
        let xldt = 1.0e-9 * CON1;
        let xlp5 = xl.sqrt();
        let xl2 = xl * xl;
        let xl3 = xl2 * xl;
        let xl4 = xl3 * xl;
        let xl5 = xl4 * xl;
        let xl6 = xl5 * xl;
        let xl7 = xl6 * xl;
        let xl8 = xl7 * xl;
        let xl9 = xl8 * xl;
        let xlmp5 = 1.0 / xlp5;
        let xlm1 = 1.0 / xl;
        let xlm2 = xlm1 * xlm1;
        let xlm3 = xlm1 * xlm2;
        let xlm4 = xlm1 * xlm3;
        let rm = den * ye;
        let rmdd = ye;
        let rmda = -rm * abari;
        let rmdz = den * abari;
        let rmi = 1.0 / rm;
        let a0 = rm * 1.0e-9;
        let a1 = pow(a0, ONE_THIRD);
        let zeta = a1 * xlm1;
        let zetadt = -a1 * xlm2 * xldt;
        let a2 = ONE_THIRD * a1 * rmi * xlm1;
        let zetadd = a2 * rmdd;
        let zetada = a2 * rmda;
        let zetadz = a2 * rmdz;
        let zeta2 = zeta * zeta;
        let zeta3 = zeta2 * zeta;
        Self {
            temp,
            logtemp,
            den,
            logden,
            den6,
            ye,
            deni,
            tempi,
            abari,
            zbari,
            abar,
            zbar,
            t9,
            xl,
            xldt,
            xlp5,
            xl2,
            xl3,
            xl4,
            xl5,
            xl6,
            xl7,
            xl8,
            xl9,
            xlmp5,
            xlm1,
            xlm2,
            xlm3,
            xlm4,
            rm,
            rmdd,
            rmda,
            rmdz,
            rmi,
            zeta,
            zetadt,
            zetadd,
            zetada,
            zetadz,
            zeta2,
            zeta3,
        }
    }
}
