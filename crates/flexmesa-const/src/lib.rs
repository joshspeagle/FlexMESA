// SPDX-License-Identifier: LGPL-3.0-or-later
//! Physical, mathematical, and astronomical constants in CGS units.
//!
//! This crate ports upstream MESA's `const_def` module in IEEE-754 double
//! precision (`f64`). Fundamental physical constants follow CODATA 2018 exact
//! definitions, and solar/planetary reference values follow the IAU 2012 and
//! IAU 2015 Resolution B3 nominal conversion constants.

#![allow(clippy::approx_constant, clippy::excessive_precision)]

// Mathematical constants (`f64`, dimensionless)

/// Ratio of a circle's circumference to its diameter ($\pi$).
pub const PI: f64 = 3.141_592_653_589_793_238_462_643_383_279_502_884_197_169_399_375_1;

/// Twice $\pi$ ($2\pi$).
pub const TWO_PI: f64 = 2.0 * PI;

/// Square of $\pi$ ($\pi^2$), matching upstream `const_def::pi2`.
pub const PI2: f64 = PI * PI;

/// Four $\pi$ ($4\pi$), matching upstream `const_def::pi4`.
pub const PI4: f64 = 4.0 * PI;

/// Alias for $4\pi$.
pub const FOUR_PI: f64 = PI4;

/// Euler-Mascheroni constant ($\gamma$), matching upstream `const_def::eulercon`.
pub const EULERCON: f64 = 0.577_215_664_901_532_861;

/// Base of the natural logarithm ($e$), matching upstream `const_def::eulernum`.
pub const EULERNUM: f64 = 2.718_281_828_459_045_235_360_287_471_352_662_49;

/// Natural logarithm of 2 ($\ln 2$).
pub const LN2: f64 = 6.931_471_805_599_452_9e-1;

/// Natural logarithm of 3 ($\ln 3$).
pub const LN3: f64 = 1.098_612_288_668_109_6;

/// Natural logarithm of $\pi$ ($\ln \pi$).
pub const LNPI: f64 = 1.144_729_885_849_400_174_143_43;

/// Natural logarithm of 10 ($\ln 10$).
pub const LN10: f64 = 2.302_585_092_994_045_5;

/// Reciprocal natural logarithm of 10 ($1 / \ln 10$).
pub const ILN10: f64 = 0.434_294_481_903_251_87;

/// Degrees-to-radians conversion factor ($\pi / 180$).
pub const A2RAD: f64 = PI / 180.0;

/// Radians-to-degrees conversion factor ($180 / \pi$).
pub const RAD2A: f64 = 180.0 / PI;

/// Alias for degrees-to-radians conversion factor.
pub const DEG2RAD: f64 = A2RAD;

/// Alias for radians-to-degrees conversion factor.
pub const RAD2DEG: f64 = RAD2A;

/// One third ($1/3$).
pub const ONE_THIRD: f64 = 1.0 / 3.0;

/// Two thirds ($2/3$).
pub const TWO_THIRDS: f64 = 2.0 / 3.0;

/// Four thirds ($4/3$).
pub const FOUR_THIRDS: f64 = 4.0 / 3.0;

/// Five thirds ($5/3$).
pub const FIVE_THIRDS: f64 = 5.0 / 3.0;

/// One sixth ($1/6$).
pub const ONE_SIXTH: f64 = 1.0 / 6.0;

/// Four thirds $\pi$ ($\frac{4}{3}\pi$), used for spherical volume factors.
pub const FOUR_THIRDS_PI: f64 = FOUR_THIRDS * PI;

/// Natural logarithm of $\frac{4}{3}\pi$ ($\ln(4\pi/3)$).
pub const LN4PI3: f64 = 1.432_411_958_301_181_0;

/// Cube root of 2 ($2^{1/3}$).
pub const TWO_13: f64 = 1.259_921_049_894_873_0;

/// Cube root of 4 ($4^{1/3}$).
pub const FOUR_13: f64 = 1.587_401_051_968_199_4;

/// Square root of 2 ($\sqrt{2}$).
pub const SQRT2: f64 = 1.414_213_562_373_095;

/// Square root of $2/3$ ($\sqrt{2/3}$).
pub const SQRT_2_DIV_3: f64 = 0.816_496_580_927_726;

// Exact SI / CODATA 2018 physical constants in CGS units

/// Avogadro constant $N_{\mathrm{A}}$ in $\mathrm{mol^{-1}}$.
pub const AVO: f64 = 6.022_140_76e23;

/// Atomic mass unit $m_{\mathrm{u}} = 1 / N_{\mathrm{A}}$ in $\mathrm{g}$.
pub const AMU: f64 = 1.0 / AVO;

/// Speed of light in vacuum $c$ in $\mathrm{cm\,s^{-1}}$.
pub const CLIGHT: f64 = 2.997_924_58e10;

/// Elementary electron charge ($q_{\mathrm{e}}$) in electrostatic units
/// ($\mathrm{esu}$).
pub const QE: f64 = (CLIGHT / 10.0) * 1.602_176_634e-19;

/// Boltzmann constant $k_{\mathrm{B}}$ in $\mathrm{erg\,K^{-1}}$.
pub const KERG: f64 = 1.380_649e-16;

/// Alias for Boltzmann constant $k_{\mathrm{B}}$ in $\mathrm{erg\,K^{-1}}$.
pub const BOLTZM: f64 = KERG;

/// Planck constant $h$ in $\mathrm{erg\,s}$.
pub const PLANCK_H: f64 = 6.626_070_15e-27;

/// Reduced Planck constant $\hbar = h / (2\pi)$ in $\mathrm{erg\,s}$.
pub const HBAR: f64 = PLANCK_H / (2.0 * PI);

/// Ideal gas constant $\mathcal{R} = k_{\mathrm{B}} N_{\mathrm{A}}$ in
/// $\mathrm{erg\,K^{-1}\,mol^{-1}}$.
pub const CGAS: f64 = BOLTZM * AVO;

/// Electron volt in $\mathrm{erg}$ ($1\;\mathrm{eV}$).
pub const EV2ERG: f64 = 1.602_176_634e-12;

/// Mega-electron-volt in $\mathrm{erg}$ ($1\;\mathrm{MeV}$).
pub const MEV_TO_ERGS: f64 = 1.0e6 * EV2ERG;

/// Mega-electron-volt per atomic mass unit ($\mathrm{MeV} / m_{\mathrm{u}}$).
pub const MEV_AMU: f64 = MEV_TO_ERGS / AMU;

/// Mega-electron-volt equivalent mass in $\mathrm{g}$ ($\mathrm{MeV} / c^2$).
pub const MEV2GR: f64 = (1.0e6 * EV2ERG) / (CLIGHT * CLIGHT);

/// Alias for mega-electron-volt equivalent mass in $\mathrm{g}$.
pub const MEV_TO_GRAMS: f64 = MEV2GR;

/// Nuclear energy conversion factor ($\mathrm{MeV} \cdot N_{\mathrm{A}}$ in
/// $\mathrm{erg\,mol^{-1}}$).
pub const QCONV: f64 = MEV_TO_ERGS * AVO;

/// Temperature-to-eV conversion factor $k_{\mathrm{B}} / \mathrm{eV}$ in
/// $\mathrm{eV\,K^{-1}}$.
pub const KEV: f64 = KERG / EV2ERG;

/// Stefan-Boltzmann constant $\sigma_{\mathrm{SB}} = \frac{\pi^2 k_{\mathrm{B}}^4}{60\,\hbar^3 c^2}$
/// in $\mathrm{erg\,cm^{-2}\,s^{-1}\,K^{-4}}$.
pub const BOLTZ_SIGMA: f64 =
    (PI * PI * BOLTZM * BOLTZM * BOLTZM * BOLTZM) / (60.0 * HBAR * HBAR * HBAR * CLIGHT * CLIGHT);

/// Radiation density constant $a_{\mathrm{rad}} = 4\sigma_{\mathrm{SB}} / c$ in
/// $\mathrm{erg\,cm^{-3}\,K^{-4}}$.
pub const CRAD: f64 = (BOLTZ_SIGMA * 4.0) / CLIGHT;

// Astronomical constants in CGS units (IAU 2012 / 2015 Resolution B3)

/// Astronomical unit in $\mathrm{cm}$ (IAU 2012 exact).
pub const AU: f64 = 1.495_978_707_00e13;

/// Parsec in $\mathrm{cm}$ ($3600 \cdot (180/\pi)\;\mathrm{au}$).
pub const PC: f64 = (3.600e3 * RAD2A) * AU;

/// Days per Julian year.
pub const DAYYER: f64 = 365.25;

/// Seconds per day ($86\,400\;\mathrm{s}$).
pub const SECDAY: f64 = 86_400.0;

/// Seconds per Julian year ($365.25 \times 86\,400\;\mathrm{s}$).
pub const SECYER: f64 = SECDAY * DAYYER;

/// Light-year in $\mathrm{cm}$ ($c \cdot \mathrm{yr}$).
pub const LY: f64 = CLIGHT * SECYER;

// Measured particle and atomic constants (CODATA 2018)

/// Neutron mass ($m_{\mathrm{n}}$) in $\mathrm{g}$ (CODATA 2018).
pub const MN: f64 = 1.674_927_498_04e-24;

/// Proton mass ($m_{\mathrm{p}}$) in $\mathrm{g}$ (CODATA 2018).
pub const MP: f64 = 1.672_621_923_69e-24;

/// Electron mass ($m_{\mathrm{e}}$) in $\mathrm{g}$ (CODATA 2018).
pub const ME: f64 = 9.109_383_701_5e-28;

/// Bohr radius ($a_{0}$) in $\mathrm{cm}$ (CODATA 2018).
pub const RBOHR: f64 = 5.291_772_109_03e-9;

/// Fine-structure constant $\alpha$ (dimensionless, CODATA 2018).
pub const FINE: f64 = 7.297_352_569_3e-3;

/// Hydrogen ionization potential (Rydberg energy) in $\mathrm{eV}$.
pub const HION: f64 = 13.605_693_122_994;

/// Thomson scattering cross section $\sigma_{\mathrm{T}}$ in $\mathrm{cm^2}$
/// (CODATA 2018).
pub const SIGE: f64 = 6.652_458_732_1e-25;

/// Weak mixing angle $\sin^2 \theta_{\mathrm{W}}$.
pub const WEINBERG_THETA: f64 = 0.22290;

/// Effective number of light neutrino families $N_{\nu}$.
pub const NUM_NEU_FAM: f64 = 3.0;

// Solar and planetary reference constants (IAU 2015 Resolution B3)

/// Standard Newtonian gravitational constant $G$ in
/// $\mathrm{cm^3\,g^{-1}\,s^{-2}}$ (CODATA 2018).
pub const STANDARD_CGRAV: f64 = 6.674_30e-8;

/// Nominal solar mass parameter $\mu_{\odot} = G M_{\odot}$ in
/// $\mathrm{cm^3\,s^{-2}}$ (IAU 2015 Resolution B3).
pub const MU_SUN: f64 = 1.327_124_4e26;

/// Nominal terrestrial mass parameter $\mu_{\oplus} = G M_{\oplus}$ in
/// $\mathrm{cm^3\,s^{-2}}$ (IAU 2015 Resolution B3).
pub const MU_EARTH: f64 = 3.986_004e20;

/// Nominal Jovian mass parameter $\mu_{\mathrm{J}} = G M_{\mathrm{J}}$ in
/// $\mathrm{cm^3\,s^{-2}}$ (IAU 2015 Resolution B3).
pub const MU_JUPITER: f64 = 1.266_865_3e23;

/// Solar age in Julian years (Bahcall et al. 2005, `ApJ`, 618, 1049).
pub const AGESUN: f64 = 4.57e9;

/// Solar mass $M_{\odot} = \mu_{\odot} / G$ in $\mathrm{g}$.
pub const MSUN: f64 = MU_SUN / STANDARD_CGRAV;

/// Alias for solar mass $M_{\odot}$ in $\mathrm{g}$.
pub const M_SUN: f64 = MSUN;

/// Nominal solar radius $R_{\odot}$ in $\mathrm{cm}$ (IAU 2015 Resolution B3).
pub const RSUN: f64 = 6.957e10;

/// Alias for solar radius $R_{\odot}$ in $\mathrm{cm}$.
pub const R_SUN: f64 = RSUN;

/// Nominal solar luminosity $L_{\odot}$ in $\mathrm{erg\,s^{-1}}$
/// (IAU 2015 Resolution B3).
pub const LSUN: f64 = 3.828e33;

/// Alias for solar luminosity $L_{\odot}$ in $\mathrm{erg\,s^{-1}}$.
pub const L_SUN: f64 = LSUN;

/// Nominal solar effective temperature $T_{\mathrm{eff},\odot}$ in $\mathrm{K}$
/// (IAU 2015 Resolution B3).
pub const TEFFSUN: f64 = 5772.0;

/// Base-10 logarithm of nominal solar surface gravity $\log_{10}(g_{\odot})$
/// in CGS ($\mathrm{cm\,s^{-2}}$).
pub const LOGGSUN: f64 = 4.438_067_627_303_133_2;

/// Nominal solar absolute bolometric magnitude $M_{\mathrm{bol},\odot}$
/// (IAU 2015 Resolution B2).
pub const MBOLSUN: f64 = 4.74;

/// Earth mass $M_{\oplus} = \mu_{\oplus} / G$ in $\mathrm{g}$.
pub const M_EARTH: f64 = MU_EARTH / STANDARD_CGRAV;

/// Nominal Earth equatorial radius $R_{\oplus}$ in $\mathrm{cm}$
/// (IAU 2015 Resolution B3).
pub const R_EARTH: f64 = 6.3781e8;

/// Nominal Earth polar radius in $\mathrm{cm}$ (IAU 2015 Resolution B3).
pub const R_EARTH_POLAR: f64 = 6.3568e8;

/// Jupiter mass $M_{\mathrm{J}} = \mu_{\mathrm{J}} / G$ in $\mathrm{g}$.
pub const M_JUPITER: f64 = MU_JUPITER / STANDARD_CGRAV;

/// Nominal Jupiter equatorial radius $R_{\mathrm{J}}$ in $\mathrm{cm}$
/// (IAU 2015 Resolution B3).
pub const R_JUPITER: f64 = 7.1492e9;

/// Nominal Jupiter polar radius in $\mathrm{cm}$ (IAU 2015 Resolution B3).
pub const R_JUPITER_POLAR: f64 = 6.6854e9;

/// Jupiter orbital semimajor axis in $\mathrm{cm}$.
pub const SEMIMAJOR_AXIS_JUPITER: f64 = 7.7857e13;

// Sentinel values and mixing-type definitions

/// Upstream MESA sentinel for an omitted optional scalar argument (`-9e99`).
pub const ARG_NOT_PROVIDED: f64 = -9.0e99;

/// Upstream MESA sentinel for a missing scalar value (`-9e99`).
pub const MISSING_VALUE: f64 = ARG_NOT_PROVIDED;

/// Number of non-negative mixing-type categories (`phase_separation_mixing + 1`).
pub const NUMBER_OF_MIXING_TYPES: usize = 11;

/// Mixing process classification matching upstream MESA `const_def`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum MixingType {
    /// Solidified / crystallized interior zone (`-1`).
    Crystallized = -1,
    /// Radiative zone with no active mixing (`0`).
    NoMixing = 0,
    /// Dynamically unstable convective zone (`1`).
    ConvectiveMixing = 1,
    /// Convective overshoot boundary zone (`2`).
    OvershootMixing = 2,
    /// Semiconvective zone (`3`).
    SemiconvectiveMixing = 3,
    /// Thermohaline (salt-finger) mixing zone (`4`).
    ThermohalineMixing = 4,
    /// Rotationally induced mixing zone (`5`).
    RotationMixing = 5,
    /// Rayleigh-Taylor instability mixing zone (`6`).
    RayleighTaylorMixing = 6,
    /// Floor / minimum background diffusive mixing (`7`).
    MinimumMixing = 7,
    /// Unclassified / user-injected diffusive mixing (`8`).
    AnonymousMixing = 8,
    /// Residual / leftover convective mixing (`9`).
    LeftoverConvectiveMixing = 9,
    /// Chemical phase-separation mixing (`10`).
    PhaseSeparationMixing = 10,
}

impl MixingType {
    /// Convert an integer mixing code from upstream MESA into a [`MixingType`].
    #[must_use]
    pub const fn from_code(code: i32) -> Option<Self> {
        match code {
            -1 => Some(Self::Crystallized),
            0 => Some(Self::NoMixing),
            1 => Some(Self::ConvectiveMixing),
            2 => Some(Self::OvershootMixing),
            3 => Some(Self::SemiconvectiveMixing),
            4 => Some(Self::ThermohalineMixing),
            5 => Some(Self::RotationMixing),
            6 => Some(Self::RayleighTaylorMixing),
            7 => Some(Self::MinimumMixing),
            8 => Some(Self::AnonymousMixing),
            9 => Some(Self::LeftoverConvectiveMixing),
            10 => Some(Self::PhaseSeparationMixing),
            _ => None,
        }
    }

    /// Return the integer mixing code matching upstream MESA `const_def`.
    #[must_use]
    pub const fn code(self) -> i32 {
        self as i32
    }
}
