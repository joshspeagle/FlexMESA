// SPDX-License-Identifier: LGPL-3.0-or-later
//! Built-in Johnson (`B, I, J, M, R, U, V`) filter transmission curves and
//! photometric zero-point fluxes (`data/colors_data/filters/Generic/Johnson`).
//!
//! Stores the 7 standard Johnson broadband passbands shipped with upstream
//! MESA `colors` together with their precomputed Vega (`vega_flam.csv`),
//! AB ($f_\nu = 3631\,$Jy), and ST
//! ($f_\lambda = 3.63 \times 10^{-9}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$)
//! photon-counting zero-point fluxes:
//! - `B`: Johnson blue passband ($3600\text{--}5600\,\mathrm{\AA}$, 21 knots).
//! - `I`: Johnson near-infrared passband ($7000\text{--}11200\,\mathrm{\AA}$,
//!   22 knots).
//! - `J`: Johnson $J$ passband ($10400\text{--}14400\,\mathrm{\AA}$, 21 knots).
//! - `M`: Johnson mid-infrared $M$ passband
//!   ($43000\text{--}59000\,\mathrm{\AA}$, 17 knots).
//! - `R`: Johnson red passband ($5200\text{--}9200\,\mathrm{\AA}$, 21 knots).
//! - `U`: Johnson ultraviolet passband ($3050\text{--}4000\,\mathrm{\AA}$,
//!   20 knots).
//! - `V`: Johnson visual passband ($4700\text{--}7000\,\mathrm{\AA}$,
//!   24 knots).

use crate::types::MagSystem;

/// Number of standard Johnson photometric filters (`B, I, J, M, R, U, V`).
pub const NUM_JOHNSON_FILTERS: usize = 7;

/// Number of output columns produced by `eval_colors` (`3 + 7 = 10`).
pub const NUM_COLORS_OUT: usize = 10;

/// Canonical history column names produced by `colors`
/// (`colors/private/colors_history.f90`).
pub const COLOR_COLUMN_NAMES: [&str; NUM_COLORS_OUT] = [
    "Mag_bol",
    "Flux_bol",
    "Interp_rad",
    "B",
    "I",
    "J",
    "M",
    "R",
    "U",
    "V",
];

/// Standard Johnson filter names in canonical MESA order.
pub const JOHNSON_FILTER_NAMES: [&str; NUM_JOHNSON_FILTERS] =
    ["B", "I", "J", "M", "R", "U", "V"];

/// Photometric filter transmission curve and precomputed zero-point fluxes.
///
/// Each passband is defined by its strictly increasing wavelength grid
/// (`wavelengths` in $\mathrm{\AA}$), dimensionless photon response
/// (`transmission`), and precomputed Vega, AB, and ST zero-point fluxes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilterCurve {
    /// Filter passband identifier (e.g., `"B"`, `"V"`).
    pub name: &'static str,
    /// Wavelength grid in Angstroms (`AA`), strictly increasing.
    pub wavelengths: &'static [f64],
    /// Dimensionless filter transmission response $S_\lambda \in [0, 1]$.
    pub transmission: &'static [f64],
    /// Vega zero-point flux in $\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$.
    pub vega_zero_point: f64,
    /// AB zero-point flux in $\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$.
    pub ab_zero_point: f64,
    /// ST zero-point flux in $\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$.
    pub st_zero_point: f64,
}

impl FilterCurve {
    /// Returns the precomputed zero-point flux $F_0$ in
    /// $\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$ for [`MagSystem`] `sys`
    /// (`Vega`, `Ab`, or `St`) used in $m = -2.5 \log_{10}(F_{\mathrm{syn}} /
    /// F_0)$.
    #[inline]
    #[must_use]
    pub const fn zero_point(&self, sys: MagSystem) -> f64 {
        match sys {
            MagSystem::Vega => self.vega_zero_point,
            MagSystem::Ab => self.ab_zero_point,
            MagSystem::St => self.st_zero_point,
        }
    }
}

/// Wavelength grid in Angstroms for Johnson `B` passband.
#[rustfmt::skip]
const JOHNSON_B_WAVELENGTHS: [f64; 11] = [
    3.700000e+03, 3.800000e+03, 4.000000e+03, 4.200000e+03, 4.400000e+03,
    4.600000e+03, 4.800000e+03, 5.000000e+03, 5.200000e+03, 5.400000e+03,
    5.600000e+03,
];

/// Transmission response for Johnson `B` passband.
#[rustfmt::skip]
const JOHNSON_B_TRANSMISSION: [f64; 11] = [
    0.000000e+00, 1.100000e-01, 9.200000e-01, 1.000000e+00, 9.400000e-01,
    7.900000e-01, 5.800000e-01, 3.600000e-01, 1.500000e-01, 4.000000e-02,
    0.000000e+00,
];

/// Wavelength grid in Angstroms for Johnson `I` passband.
#[rustfmt::skip]
const JOHNSON_I_WAVELENGTHS: [f64; 26] = [
    6.800000e+03, 7.000000e+03, 7.200000e+03, 7.400000e+03, 7.600000e+03,
    7.800000e+03, 8.000000e+03, 8.200000e+03, 8.400000e+03, 8.600000e+03,
    8.800000e+03, 9.000000e+03, 9.200000e+03, 9.400000e+03, 9.600000e+03,
    9.800000e+03, 1.000000e+04, 1.020000e+04, 1.040000e+04, 1.060000e+04,
    1.080000e+04, 1.100000e+04, 1.120000e+04, 1.140000e+04, 1.160000e+04,
    1.200000e+04,
];

/// Transmission response for Johnson `I` passband.
#[rustfmt::skip]
const JOHNSON_I_TRANSMISSION: [f64; 26] = [
    0.000000e+00, 1.000000e-02, 1.700000e-01, 3.600000e-01, 5.600000e-01,
    7.600000e-01, 9.600000e-01, 9.800000e-01, 9.900000e-01, 1.000000e+00,
    9.800000e-01, 9.300000e-01, 8.400000e-01, 7.100000e-01, 5.800000e-01,
    4.700000e-01, 3.600000e-01, 2.800000e-01, 2.000000e-01, 1.500000e-01,
    1.000000e-01, 8.000000e-02, 5.000000e-02, 3.000000e-02, 2.000000e-02,
    0.000000e+00,
];

/// Wavelength grid in Angstroms for Johnson `J` passband.
#[rustfmt::skip]
const JOHNSON_J_WAVELENGTHS: [f64; 31] = [
    9.600000e+03, 9.800000e+03, 1.000000e+04, 1.020000e+04, 1.040000e+04,
    1.060000e+04, 1.080000e+04, 1.100000e+04, 1.120000e+04, 1.140000e+04,
    1.160000e+04, 1.180000e+04, 1.200000e+04, 1.220000e+04, 1.240000e+04,
    1.260000e+04, 1.280000e+04, 1.300000e+04, 1.320000e+04, 1.340000e+04,
    1.360000e+04, 1.380000e+04, 1.400000e+04, 1.420000e+04, 1.440000e+04,
    1.460000e+04, 1.480000e+04, 1.500000e+04, 1.520000e+04, 1.540000e+04,
    1.560000e+04,
];

/// Transmission response for Johnson `J` passband.
#[rustfmt::skip]
const JOHNSON_J_TRANSMISSION: [f64; 31] = [
    0.000000e+00, 2.000000e-02, 3.000000e-02, 6.000000e-02, 1.600000e-01,
    3.500000e-01, 6.200000e-01, 9.300000e-01, 8.500000e-01, 7.800000e-01,
    7.800000e-01, 8.000000e-01, 8.500000e-01, 9.300000e-01, 7.500000e-01,
    6.400000e-01, 6.300000e-01, 6.300000e-01, 6.600000e-01, 6.800000e-01,
    7.000000e-01, 7.000000e-01, 6.600000e-01, 6.000000e-01, 4.600000e-01,
    2.700000e-01, 1.400000e-01, 9.000000e-02, 6.000000e-02, 2.000000e-02,
    0.000000e+00,
];

/// Wavelength grid in Angstroms for Johnson `M` passband.
#[rustfmt::skip]
const JOHNSON_M_WAVELENGTHS: [f64; 23] = [
    4.100000e+04, 4.200000e+04, 4.300000e+04, 4.400000e+04, 4.500000e+04,
    4.600000e+04, 4.700000e+04, 4.800000e+04, 4.900000e+04, 5.000000e+04,
    5.100000e+04, 5.200000e+04, 5.300000e+04, 5.400000e+04, 5.500000e+04,
    5.600000e+04, 5.700000e+04, 5.800000e+04, 5.900000e+04, 6.000000e+04,
    6.100000e+04, 6.200000e+04, 6.300000e+04,
];

/// Transmission response for Johnson `M` passband.
#[rustfmt::skip]
const JOHNSON_M_TRANSMISSION: [f64; 23] = [
    0.000000e+00, 4.000000e-02, 2.500000e-01, 4.800000e-01, 1.000000e+00,
    9.300000e-01, 9.200000e-01, 8.900000e-01, 8.400000e-01, 7.800000e-01,
    7.600000e-01, 7.900000e-01, 7.400000e-01, 6.800000e-01, 6.000000e-01,
    4.900000e-01, 3.200000e-01, 2.600000e-01, 2.100000e-01, 1.900000e-01,
    9.000000e-02, 2.000000e-02, 0.000000e+00,
];

/// Wavelength grid in Angstroms for Johnson `R` passband.
#[rustfmt::skip]
const JOHNSON_R_WAVELENGTHS: [f64; 23] = [
    5.200000e+03, 5.400000e+03, 5.600000e+03, 5.800000e+03, 6.000000e+03,
    6.200000e+03, 6.400000e+03, 6.600000e+03, 6.800000e+03, 7.000000e+03,
    7.200000e+03, 7.400000e+03, 7.600000e+03, 7.800000e+03, 8.000000e+03,
    8.200000e+03, 8.400000e+03, 8.600000e+03, 8.800000e+03, 9.000000e+03,
    9.200000e+03, 9.400000e+03, 9.600000e+03,
];

/// Transmission response for Johnson `R` passband.
#[rustfmt::skip]
const JOHNSON_R_TRANSMISSION: [f64; 23] = [
    0.000000e+00, 6.000000e-02, 2.800000e-01, 5.000000e-01, 6.900000e-01,
    7.900000e-01, 8.800000e-01, 9.400000e-01, 9.800000e-01, 1.000000e+00,
    9.400000e-01, 8.500000e-01, 7.300000e-01, 5.700000e-01, 4.200000e-01,
    3.100000e-01, 1.700000e-01, 1.100000e-01, 6.000000e-02, 4.000000e-02,
    2.000000e-02, 1.000000e-02, 0.000000e+00,
];

/// Wavelength grid in Angstroms for Johnson `U` passband.
#[rustfmt::skip]
const JOHNSON_U_WAVELENGTHS: [f64; 13] = [
    3.000000e+03, 3.100000e+03, 3.200000e+03, 3.300000e+03, 3.400000e+03,
    3.500000e+03, 3.600000e+03, 3.700000e+03, 3.800000e+03, 3.900000e+03,
    4.000000e+03, 4.100000e+03, 4.200000e+03,
];

/// Transmission response for Johnson `U` passband.
#[rustfmt::skip]
const JOHNSON_U_TRANSMISSION: [f64; 13] = [
    0.000000e+00, 1.000000e-01, 6.100000e-01, 8.400000e-01, 9.300000e-01,
    9.700000e-01, 1.000000e+00, 9.700000e-01, 7.300000e-01, 3.600000e-01,
    5.000000e-02, 1.000000e-02, 0.000000e+00,
];

/// Wavelength grid in Angstroms for Johnson `V` passband.
#[rustfmt::skip]
const JOHNSON_V_WAVELENGTHS: [f64; 15] = [
    4.600000e+03, 4.800000e+03, 5.000000e+03, 5.200000e+03, 5.400000e+03,
    5.600000e+03, 5.800000e+03, 6.000000e+03, 6.200000e+03, 6.400000e+03,
    6.600000e+03, 6.800000e+03, 7.000000e+03, 7.200000e+03, 7.400000e+03,
];

/// Transmission response for Johnson `V` passband.
#[rustfmt::skip]
const JOHNSON_V_TRANSMISSION: [f64; 15] = [
    0.000000e+00, 2.000000e-02, 3.800000e-01, 9.100000e-01, 9.800000e-01,
    7.200000e-01, 6.200000e-01, 4.000000e-01, 2.000000e-01, 8.000000e-02,
    2.000000e-02, 1.000000e-02, 1.000000e-02, 1.000000e-02, 0.000000e+00,
];

/// Built-in Johnson `B, I, J, M, R, U, V` filter definitions.
pub const JOHNSON_FILTERS: [FilterCurve; NUM_JOHNSON_FILTERS] = [
    FilterCurve {
        name: "B",
        wavelengths: &JOHNSON_B_WAVELENGTHS,
        transmission: &JOHNSON_B_TRANSMISSION,
        vega_zero_point: 6.23409444220002772e-09,
        ab_zero_point: 5.55919935245269163e-09,
        st_zero_point: 3.62999999999999925e-09,
    },
    FilterCurve {
        name: "I",
        wavelengths: &JOHNSON_I_WAVELENGTHS,
        transmission: &JOHNSON_I_TRANSMISSION,
        vega_zero_point: 9.00674021125682667e-10,
        ab_zero_point: 1.42407642320277786e-09,
        st_zero_point: 3.62999999999999842e-09,
    },
    FilterCurve {
        name: "J",
        wavelengths: &JOHNSON_J_WAVELENGTHS,
        transmission: &JOHNSON_J_TRANSMISSION,
        vega_zero_point: 3.06321675530278865e-10,
        ab_zero_point: 7.04915964405315247e-10,
        st_zero_point: 3.62999999999999967e-09,
    },
    FilterCurve {
        name: "M",
        wavelengths: &JOHNSON_M_WAVELENGTHS,
        transmission: &JOHNSON_M_TRANSMISSION,
        vega_zero_point: 1.74102609518352225e-12,
        ab_zero_point: 4.32237370848469843e-11,
        st_zero_point: 3.62999999999999967e-09,
    },
    FilterCurve {
        name: "R",
        wavelengths: &JOHNSON_R_WAVELENGTHS,
        transmission: &JOHNSON_R_TRANSMISSION,
        vega_zero_point: 1.81033471956106077e-09,
        ab_zero_point: 2.28463185920518181e-09,
        st_zero_point: 3.63000000000000049e-09,
    },
    FilterCurve {
        name: "U",
        wavelengths: &JOHNSON_U_WAVELENGTHS,
        transmission: &JOHNSON_U_TRANSMISSION,
        vega_zero_point: 3.82215071506616128e-09,
        ab_zero_point: 8.74427347756693118e-09,
        st_zero_point: 3.62999999999999967e-09,
    },
    FilterCurve {
        name: "V",
        wavelengths: &JOHNSON_V_WAVELENGTHS,
        transmission: &JOHNSON_V_TRANSMISSION,
        vega_zero_point: 3.53100675319667627e-09,
        ab_zero_point: 3.56455157325040893e-09,
        st_zero_point: 3.63000000000000049e-09,
    },
];
