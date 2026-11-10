//! Safe Rust ionic charge tables, 4D $(Z, X, \log Q, \log T)$ partial-pressure
//! and neutral-fraction interpolators, and Dupuis/Paquette diffusion charges
//! (`flexmesa-ionization`).
//!
//! Part of the Wave 2b (`ionization`) microphysics layer in `FlexMESA`.
//! Verified against 100% of upstream MESA `ionization/test/test_output`
//! (`test_fe56_in_he4` and `do_test_eval_ionization`) and `mod_typical_charge`.
//! Enforces `#![forbid(unsafe_code)]` and IEEE-754 `f64` precision.
//!
//! Corresponds to upstream MESA `ionization` (`ionization_def.f90`,
//! `ionization_lib.f90`, `mod_ionization.f90`, `ion_tables_load.f90`,
//! `ion_tables_eval.f90`) and `star/private/mod_typical_charge.f90`:
//! - **Fe56 in He4 (`fe56_he4`)**: 2D not-a-knot bicubic spline over the
//!   $105 \times 30$ Thomas-Fermi-Dirac-Hubbard $(\log_{10} n_e, \log_{10} T)$
//!   grid (`eval_charge_of_fe56_in_he4`, `Fe56He4Table`).
//! - **4D Ionization Tables (`ion_table`, `ion_table_load`)**: 3-stage
//!   interpolator in $(Z, X, \log Q, \log T)$ over 28 output quantities
//!   (`logPgas`, `logpp_H..Fe`, `Z_H..Fe`, `fneut_H..Fe`), monomorphized over
//!   [`IonScalar`] (`f64` and [`DualOrder1<N>`]) and `rayon` parallel zones.
//! - **Diffusion Typical Charge (`typical_charge`)**: Paquette et al. (1986,
//!   Eq. 20) + Dupuis et al. (1992, Eq. 3) continuum depression (`chi_info`,
//!   `chi_effective`) and Allen (1973) ionization potentials (`Z = 1..=30`,
//!   `get_typical_charge`, `eval_typical_charge`).
//!
//! # Index and Memory-Order Contract
//! - Output channel indices (`ION_ILOG_PGAS = 1 .. ION_IFNEUT_FE = 28`)
//!   preserve 1-based MESA constants in `ionization_def.f90`, while 0-based
//!   Rust slices `[f64; 28]` index channels `0..28`.
//! - Multi-zone output arrays use row-major `(n_zones, 28)` layout.

#![forbid(unsafe_code)]
#![allow(
    clippy::approx_constant,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::similar_names,
    clippy::unreadable_literal
)]

pub mod fe56_he4;
pub mod fe56_he4_data_a;
pub mod fe56_he4_data_b;
pub mod fe56_he4_data_c;
pub mod ion_table;
pub mod ion_table_load;
pub mod potentials_data;
pub mod typical_charge;

use std::error::Error;
use std::fmt::{self, Debug, Display};
use std::ops::{Add, Div, Mul, Sub};

pub use flexmesa_autodiff::DualOrder1;
use flexmesa_math::{cospi, log10, pow};

pub use fe56_he4::{
    eval_charge_of_fe56_in_he4, Fe56He4Table, PARALLEL_ZONE_THRESHOLD,
};
pub use fe56_he4_data_a::{
    LOG_NE_FE56_HE4, LOG_T_FE56_HE4, NUM_LOG_NE_FE56_HE4, NUM_LOG_T_FE56_HE4,
};
pub use ion_table::eval_ionization;
pub use ion_table_load::IonTableGrid;
pub use potentials_data::{IONIZATION_POTENTIALS, MAX_TYPICAL_CHARGE_Z};
pub use typical_charge::{
    chi_effective, chi_info, eval_typical_charge, get_typical_charge,
    ionization_potential,
};

/// Number of output quantities returned by [`eval_ionization`] (`28`).
pub const NUM_ION_VALS: usize = 28;

/// Number of metallicity grid tables `ion_Zs` (`5`).
pub const NUM_ION_ZS: usize = 5;

/// Number of hydrogen mass-fraction grid tables `ion_Xs` (`6`).
pub const NUM_ION_XS: usize = 6;

/// Tabulated metallicity grid $Z \in \{0.00, 0.02, 0.04, 0.20, 1.00\}$.
pub const ION_ZS: [f64; NUM_ION_ZS] = [0.00, 0.02, 0.04, 0.20, 1.00];

/// Number of valid $X$ tables for each $Z$ in [`ION_ZS`].
pub const NUM_ION_XS_FOR_Z: [usize; NUM_ION_ZS] = [6, 5, 5, 5, 1];

/// Tabulated hydrogen mass-fraction grid $X \in \{0.0, 0.2, 0.4, 0.6, 0.8,
/// 1.0\}$.
pub const ION_XS: [f64; NUM_ION_XS] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

/// Ordered names of the 28 ionization output channels (`ion_result_names`).
pub const ION_RESULT_NAMES: [&str; NUM_ION_VALS] = [
    "logPgas", "logpp_H", "logpp_He", "logpp_C", "logpp_N", "logpp_O",
    "logpp_Ne", "logpp_Mg", "logpp_Si", "logpp_Fe", "Z_H", "Z_He", "Z_C",
    "Z_N", "Z_O", "Z_Ne", "Z_Mg", "Z_Si", "Z_Fe", "fneut_H", "fneut_He",
    "fneut_C", "fneut_N", "fneut_O", "fneut_Ne", "fneut_Mg", "fneut_Si",
    "fneut_Fe",
];

/// Errors returned by [`flexmesa_ionization`](crate) routines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IonError {
    /// Invalid input parameter or out-of-domain coordinate.
    InvalidInput {
        /// Calling routine context.
        context: &'static str,
        /// Diagnostic description.
        detail: String,
    },
    /// 2D bicubic spline setup or evaluation error.
    Interp(String),
    /// Malformed ASCII ionization table file.
    TableParse {
        /// File path or table identifier.
        path: String,
        /// Parse failure detail.
        detail: String,
    },
    /// I/O error reading an external table file.
    Io(String),
}

impl Display for IonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { context, detail } => {
                write!(f, "{context}: {detail}")
            }
            Self::Interp(msg) => write!(f, "interpolation error: {msg}"),
            Self::TableParse { path, detail } => {
                write!(f, "failed to parse {path}: {detail}")
            }
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl Error for IonError {}

/// Trait abstracting over primal IEEE-754 scalars (`f64`) and first-order
/// forward-mode dual numbers ([`DualOrder1<N>`]) for ionization kernels.
pub trait IonScalar:
    Copy
    + Clone
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    /// Embed a constant `f64` scalar with zero derivatives.
    #[must_use]
    fn from_f64(v: f64) -> Self;

    /// Extract the primal `f64` value.
    #[must_use]
    fn val(self) -> f64;

    /// Square root $\sqrt{x}$.
    #[must_use]
    fn sqrt(self) -> Self;

    /// Base-10 logarithm $\log_{10}(x)$.
    #[must_use]
    fn log10(self) -> Self;

    /// Real power $x^p$.
    #[must_use]
    fn powf(self, p: f64) -> Self;

    /// Half-period cosine $\cos(\pi x)$.
    #[must_use]
    fn cospi(self) -> Self;
}

/// Scalar implementation of [`IonScalar`] for native 64-bit IEEE-754 floats.
impl IonScalar for f64 {
    /// Wrap a constant `f64` scalar.
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }

    /// Return the primal `f64` value.
    #[inline]
    fn val(self) -> f64 {
        self
    }

    /// Compute the IEEE-754 square root $\sqrt{x}$.
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    /// Compute the base-10 logarithm $\log_{10}(x)$.
    #[inline]
    fn log10(self) -> Self {
        log10(self)
    }

    /// Compute the real power $x^p$.
    #[inline]
    fn powf(self, p: f64) -> Self {
        pow(self, p)
    }

    /// Compute the half-period cosine $\cos(\pi x)$.
    #[inline]
    fn cospi(self) -> Self {
        cospi(self)
    }
}

/// Forward-mode automatic differentiation implementation of [`IonScalar`] for
/// [`DualOrder1<N>`].
impl<const N: usize> IonScalar for DualOrder1<N> {
    /// Embed a constant `f64` with zero derivative tangents.
    #[inline]
    fn from_f64(v: f64) -> Self {
        Self::constant(v)
    }

    /// Extract the primal `f64` value.
    #[inline]
    fn val(self) -> f64 {
        self.val
    }

    /// Propagate first-order derivatives through $\sqrt{x}$.
    #[inline]
    fn sqrt(self) -> Self {
        Self::sqrt(self)
    }

    /// Propagate first-order derivatives through $\log_{10}(x)$.
    #[inline]
    fn log10(self) -> Self {
        Self::log10(self)
    }

    /// Propagate first-order derivatives through $x^p$.
    #[inline]
    fn powf(self, p: f64) -> Self {
        Self::powf(self, p)
    }

    /// Propagate first-order derivatives through $\cos(\pi x)$.
    #[inline]
    fn cospi(self) -> Self {
        Self::cospi(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify `eval_charge_of_fe56_in_he4` against `test_output`.
    ///
    /// Checks all three golden $(\log_{10} n_e, \log_{10} T)$ points from
    /// upstream MESA `ionization/test/src/test_ionization.f90`.
    #[test]
    fn test_fe56_in_he4_matches_mesa_golden_output() {
        let cases = [
            (30.5, 7.9, 25.895_386_9),
            (26.0, 7.0, 19.099_324_7),
            (24.5, 6.1, 11.231_852_4),
        ];
        for (lne, lt, expected) in cases {
            let z = eval_charge_of_fe56_in_he4(lne, lt).expect("eval");
            assert!(
                (z - expected).abs() < 1e-7,
                "at ({lne}, {lt}): got {z}, expected {expected}"
            );
        }
    }

    /// Verify `eval_ionization` against all 28 golden values in `test_output`.
    #[test]
    fn test_eval_ionization_matches_mesa_golden_output() {
        let z = 0.0188;
        let x = 0.725;
        let t = 3.234_552_959_158_798_9e4;
        let rho = 9.076_877_520_606_785_8e-6;
        let res = eval_ionization(z, x, Some(rho), None, Some(t), None)
            .expect("eval_ionization");

        let expected: [f64; 28] = [
            7.571_288_983_310_772_4e0,
            7.249_798_734_689_492_5e0,
            6.295_307_644_551_021_7e0,
            2.327_522_341_939_101_6e0,
            1.763_835_741_518_751_8e0,
            2.633_371_341_521_088_2e0,
            1.869_281_541_521_129_7e0,
            1.438_824_941_517_523_6e0,
            1.589_669_941_532_439_6e0,
            1.323_525_141_524_150_9e0,
            9.634_304_690_432_556_9e-1,
            6.715_630_933_782_363_9e-1,
            1.142_408_918_941_255_8e0,
            1.089_862_068_046_478_6e0,
            1.008_894_828_491_671_4e0,
            2.483_115_534_518_743_7e0,
            1.940_356_414_576_985_2e0,
            1.839_216_924_010_597_8e0,
            1.971_759_495_053_324_6e0,
            3.656_953_095_671_257_3e-2,
            3.284_281_731_970_192_2e-1,
            1.032_730_409_056_921_6e-2,
            1.050_463_337_973_329_8e-2,
            3.903_958_595_612_055_2e-2,
            2.486_397_698_827_527_5e-3,
            1.001_464_231_564_128_2e-5,
            9.654_071_204_560_319_2e-4,
            4.446_083_247_970_579_0e-6,
        ];
        for (k, (&got, &exp)) in res.iter().zip(expected.iter()).enumerate() {
            let rel = (got - exp).abs() / exp.abs().max(1e-30);
            assert!(
                rel < 1e-14,
                "channel {} ({}): got {got:.16e}, expected {exp:.16e} \
                 (rel={rel:.2e})",
                k,
                ION_RESULT_NAMES[k]
            );
        }
    }

    /// Verify `chi_info`, `chi_effective`, and `eval_typical_charge` on C12 and
    /// Fe56 across solar-core and white-dwarf conditions, plus `DualOrder1` AD.
    #[test]
    fn test_typical_charge_and_dual_autodiff() {
        let cat = flexmesa_chem::NuclideCatalog::builtin();
        let ic12 = cat.get_nuclide_index("c12").expect("c12");
        let ife56 = cat.get_nuclide_index("fe56").expect("fe56");

        // Hot dense plasma (T = 1e7 K, rho = 100 g/cm^3): C12 fully ionized (6)
        let z_c12 =
            eval_typical_charge(&cat, ic12, 1.3, 0.85, 1.0e7, 7.0, 100.0, 2.0)
                .expect("c12 charge");
        assert!((z_c12 - 6.0).abs() < 1e-15);

        // Cool envelope (T = 1e4 K, rho = 1e-6 g/cm^3): Fe56 singly ionized (1)
        let z_fe56 = eval_typical_charge(
            &cat, ife56, 1.3, 0.5, 1.0e4, 4.0, 1.0e-6, -6.0,
        )
        .expect("fe56 charge");
        assert!((z_fe56 - 1.0).abs() < 1e-15);

        // Verify DualOrder1<2> forward-mode AD on chi_info / chi_effective
        let t_d = DualOrder1::<2>::independent(1.0e6, 0);
        let lt_d = t_d.log10();
        let rho_d = DualOrder1::<2>::independent(10.0, 1);
        let lrho_d = rho_d.log10();
        let a1 = DualOrder1::<2>::constant(4.0);
        let z1 = DualOrder1::<2>::constant(2.0);
        let z2 = DualOrder1::<2>::constant(6.0);
        let (chi, c0, c1, c2) = chi_info(a1, z1, t_d, lt_d, rho_d, lrho_d);
        let ceff = chi_effective(chi, c0, c1, c2, z1, z2);
        assert!(
            ceff.val > 0.0 && ceff.d1[0].is_finite() && ceff.d1[1].is_finite()
        );
    }

    /// Verify full $175 \times 102$ $(\log Q, \log T)$ grid coverage across all
    /// $(Z, X)$ tables without $2 \times 2$ cell clamping, and parity between
    /// embedded binary and ASCII/cached tables.
    #[test]
    fn test_full_175x102_grid_and_embedded_parity() {
        let grid = IonTableGrid::builtin();
        assert_eq!(grid.num_log_qs, 175);
        assert_eq!(grid.num_log_ts, 102);
        assert!((grid.log_q_min - (-10.0)).abs() < 1e-14);
        assert!((grid.log_t_min - 2.10).abs() < 1e-14);

        let embedded = IonTableGrid::from_embedded_full();
        assert_eq!(embedded.num_log_qs, 175);
        assert_eq!(embedded.num_log_ts, 102);
        for iz in 0..NUM_ION_ZS {
            for ix in 0..NUM_ION_XS {
                assert_eq!(grid.coeffs[iz][ix].len(), NUM_ION_VALS);
                for v in 0..NUM_ION_VALS {
                    assert_eq!(
                        grid.coeffs[iz][ix][v].len(),
                        175 * 102 * 4
                    );
                    for (&a, &b) in grid.coeffs[iz][ix][v]
                        .iter()
                        .zip(embedded.coeffs[iz][ix][v].iter())
                    {
                        assert!((a - b).abs() <= 1e-14);
                    }
                }
            }
        }

        // Cool vs hot plasma must yield distinct ionization states (no 2x2 clamping)
        let cool = eval_ionization(0.02, 0.70, None, Some(-7.0), None, Some(3.5))
            .expect("cool eval");
        let hot = eval_ionization(0.02, 0.70, None, Some(-1.0), None, Some(7.5))
            .expect("hot eval");
        assert!(cool[10] < 0.05, "H should be mostly neutral at logT=3.5: {}", cool[10]);
        assert!((hot[10] - 1.0).abs() < 1e-2, "H should be ionized at logT=7.5: {}", hot[10]);
        assert!(hot[18] > 20.0, "Fe should be highly ionized at logT=7.5: {}", hot[18]);
    }
}
