//! Safe Rust chemical element tables, nuclide catalogs, solar abundance scales,
//! reaction $Q$-values, and composition moments (`flexmesa-chem`).
//!
//! Part of the Wave 2a (`chem`) microphysics layer in `FlexMESA`.
//! Verified against 100% of upstream MESA `chem/test/test_output`
//! (`o16`, `AG89`, `Lodders 2003`, `composition_info`, `reaction_Qtotal`).
//! Enforces `#![forbid(unsafe_code)]` and IEEE-754 `f64` precision.
//!
//! Corresponds to upstream MESA `chem` (`chem_def.f90`, `chem_lib.f90`,
//! `chem_isos_io.f90`, and `lodders_mod.f90`):
//! - **Tier A (Static Catalogs & Queries)**: Periodic table symbols ($Z \in
//!   [0, 112]$), 24 nuclear reaction categories, 286 Anders & Grevesse (1989)
//!   and Lodders (2003) solar isotopes, 10 solar elemental abundance scales
//!   (`AG89` through `AAG21`), and [`NuclideCatalog`] (347 built-in reference
//!   nuclides + full `isotopes.data` parser).
//! - **Tier B (Differentiable Kernels)**: [`get_composition_info`],
//!   [`eval_composition_moments_generic`] (monomorphized over `f64` and
//!   [`DualOrder1<N>`]), and multi-zone `rayon`-parallel
//!   [`get_composition_info_zones`].
//!
//! # Index and Memory-Order Contract
//! - Element IDs (`e_h = 1 .. e_cn = 112`) and nuclide IDs (`chem_id >= 1`)
//!   preserve upstream 1-based MESA integer identifiers (`0` for `neut`).
//! - Solar metal mass-fraction arrays (`[f64; 112]`) use 0-based index `z - 1`
//!   for atomic number $Z \in [1, 112]$.
//! - Multi-zone abundance buffers (`x_zones`) use row-major `(n_zones,
//!   n_species)` layout.
//!
//! # Modules
//! - [`elements`]: Periodic table symbols, full names, dominant isotopes,
//!   nuclear reaction categories (`1..=24`), and partition function grid.
//! - [`solar_ag89`] / [`solar_ag89_data`]: Anders & Grevesse (1989) and
//!   Lodders (2003) 286-isotope solar catalog and beta-decay mapping.
//! - [`solar_scales`] / [`solar_scales_data`]: 10 solar elemental abundance
//!   scales (`AG89`, `GN93`, `GS98`, `L03`, `AGS05`, `AGSS09`, `L09`,
//!   `A09_Prz`, `MB22`, `AAG21`) and `[M/H]` conversion.
//! - [`nuclides`]: 347-nuclide reference catalog, `isotopes.data` parser,
//!   and nuclear reaction $Q$-value evaluation (`reaction_Qtotal`).
//! - [`composition`]: Composition moments and analytical partial derivatives.

#![forbid(unsafe_code)]
#![allow(
    clippy::approx_constant,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::similar_names,
    clippy::unreadable_literal
)]

pub mod composition;
pub mod elements;
pub mod nuclides;
pub mod nuclides_data_a;
pub mod nuclides_data_b;
pub mod nuclides_data_c;
pub mod solar_ag89;
pub mod solar_ag89_data;
pub mod solar_scales;
pub mod solar_scales_data;

use std::error::Error;
use std::fmt::{self, Debug, Display};
use std::ops::{Add, Div, Mul, Sub};

pub use flexmesa_autodiff::DualOrder1;

pub use composition::{
    eval_composition_moments_generic, get_composition_info,
    get_composition_info_from_props, get_composition_info_zones,
    CompositionMoments, CompositionResult, SpeciesProps,
    PARALLEL_ZONE_THRESHOLD,
};
pub use elements::{
    category_id, category_name, chem_get_element_id, element_long_name,
    element_main_iso_name, element_name, generate_long_nuclide_names,
    generate_nuclide_names, get_partition_fcn_indx, CATEGORY_NAME,
    EL_LONG_NAME, EL_NAME, MAIN_ISO_NAME, MAX_EL_Z, NPART, NUM_CATEGORIES,
    NUM_CHEM_ELEMENTS, T_PART,
};
pub use nuclides::{
    builtin_nuclide_rows, NuclideCatalog, NuclideEntry, DEL_MN, DEL_MP,
};
pub use solar_ag89::{
    ag89_yesol, ag89_zsol, chem_get_solar, chem_xsol, get_stable_mass_frac,
    lodders03_element_atom_percent,
};
pub use solar_ag89_data::{Ag89Row, AG89_TABLE, SOLSIZ};
pub use solar_scales::{
    chem_m_div_h, element_atomic_weights, solar_cno_fraction,
    solar_element_zfrac, SolarScale,
};

/// Errors returned by [`flexmesa_chem`](crate) routines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChemError {
    /// Invalid input parameter or unknown nuclide/scale identifier.
    InvalidInput {
        /// Calling routine context.
        context: &'static str,
        /// Diagnostic description.
        detail: String,
    },
    /// Malformed `isotopes.data` or `lodders03.data` table entry.
    TableParse {
        /// File or table identifier.
        path: String,
        /// Parse failure detail.
        detail: String,
    },
    /// I/O error reading an external table file.
    Io(String),
}

impl Display for ChemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { context, detail } => {
                write!(f, "{context}: {detail}")
            }
            Self::TableParse { path, detail } => {
                write!(f, "failed to parse {path}: {detail}")
            }
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl Error for ChemError {}

/// Trait abstracting over primal IEEE-754 scalars (`f64`) and first-order
/// forward-mode dual numbers ([`DualOrder1<N>`]) for composition kernels.
pub trait ChemScalar:
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
}

impl ChemScalar for f64 {
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }

    #[inline]
    fn val(self) -> f64 {
        self
    }
}

impl<const N: usize> ChemScalar for DualOrder1<N> {
    #[inline]
    fn from_f64(v: f64) -> Self {
        Self::constant(v)
    }

    #[inline]
    fn val(self) -> f64 {
        self.val
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify `o16` nuclear properties and `he` element lookup against MESA.
    #[test]
    fn test_o16_properties_match_mesa_golden_output() {
        let cat = NuclideCatalog::builtin();
        let io16 = cat.get_nuclide_index("o16").expect("o16 present");
        let o16 = cat.get_by_id(io16).expect("o16 entry");
        assert_eq!(o16.name, "o16");
        assert_eq!(o16.z, 8);
        assert!((o16.w - 1.599_491_462_007_534_9e1).abs() < 1e-14);
        assert!((o16.binding_energy - 1.276_193_022_100_000_0e2).abs() < 1e-13);
        assert_eq!(chem_get_element_id("he"), Some(2));
        assert_eq!(element_name(2), Some("he"));
    }

    /// Verify `zsol`, `yesol`, and `CNO/Z` across solar abundance scales.
    #[test]
    fn test_ag89_and_solar_cno_fractions_match_mesa_golden_output() {
        let cat = NuclideCatalog::builtin();
        let zsol = ag89_zsol();
        let yesol = ag89_yesol();
        assert!((zsol - 1.898_710_963_189_181_9e-2).abs() < 1e-16);
        assert!((yesol - 8.528_019_090_202_733_1e-1).abs() < 1e-15);

        let cno_ag89 =
            (chem_xsol("c12") + chem_xsol("n14") + chem_xsol("o16")) / zsol;
        assert!((cno_ag89 - 7.230_747_736_843_437_0e-1).abs() < 1e-15);

        let cno_gn93 = solar_cno_fraction(SolarScale::Gn93, &cat);
        let cno_gs98 = solar_cno_fraction(SolarScale::Gs98, &cat);
        let cno_l03 = solar_cno_fraction(SolarScale::L03, &cat);
        let cno_ags05 = solar_cno_fraction(SolarScale::Ags05, &cat);
        assert!((cno_gn93 - 7.104_573_542_889_379_7e-1).abs() < 1e-15);
        assert!((cno_gs98 - 6.918_876_873_434_645_1e-1).abs() < 1e-15);
        assert!((cno_l03 - 6.630_257_046_211_597_3e-1).abs() < 1e-15);
        assert!((cno_ags05 - 6.677_095_411_699_899_3e-1).abs() < 1e-15);
    }

    /// Verify 50/50 `c12`-`fe56` composition moments and `DualOrder1` AD.
    #[test]
    fn test_composition_info_c12_fe56_and_dual_autodiff_parity() {
        let cat = NuclideCatalog::builtin();
        let ic12 = cat.get_nuclide_index("c12").expect("c12");
        let ife56 = cat.get_nuclide_index("fe56").expect("fe56");
        let ids = [ic12, ife56];
        let x = [0.5_f64, 0.5_f64];
        let res =
            get_composition_info(&cat, &ids, &x, false).expect("comp info");

        assert!((res.abar - 336.0 / 17.0).abs() < 1e-14);
        assert!((res.zbar - 162.0 / 17.0).abs() < 1e-14);
        assert!((res.z2bar - 2532.0 / 17.0).abs() < 1e-13);
        assert!((res.ye - 27.0 / 56.0).abs() < 1e-15);
        assert!((res.mass_correction - 0.999_419).abs() < 1e-6);

        let props = SpeciesProps::from_catalog(&cat, &ids).expect("props");
        let x_dual = [
            DualOrder1::<2>::independent(0.5, 0),
            DualOrder1::<2>::independent(0.5, 1),
        ];
        let m_dual = eval_composition_moments_generic(&props, &x_dual);
        assert!((m_dual.abar.val() - res.abar).abs() < 1e-15);
        assert!((m_dual.abar.d1[0] - res.dabar_dx[0]).abs() < 1e-14);
        assert!((m_dual.abar.d1[1] - res.dabar_dx[1]).abs() < 1e-14);
    }

    /// Verify `reaction_qtotal` on `2 he3 -> he4 + 2 h1`.
    #[test]
    fn test_reaction_qtotal_two_he3_to_he4_plus_two_h1() {
        let cat = NuclideCatalog::builtin();
        let ihe3 = cat.get_nuclide_index("he3").expect("he3");
        let ihe4 = cat.get_nuclide_index("he4").expect("he4");
        let ih1 = cat.get_nuclide_index("h1").expect("h1");
        let reactants = [ihe3, ihe3, ihe4, ih1, ih1];
        let qtotal = cat
            .reaction_qtotal(2, 3, &reactants)
            .expect("reaction_qtotal");
        assert!((qtotal - 1.285_957_285_000_000_5e1).abs() < 1e-14);
    }

    /// Verify Lodders (2003) atom percentages for `h1`, `fe56`, and `u238`.
    #[test]
    fn test_lodders03_atom_percent_lookup() {
        assert!(
            (lodders03_element_atom_percent("h1") - 99.998_06).abs() < 1e-6
        );
        assert!((lodders03_element_atom_percent("fe56") - 91.754).abs() < 1e-6);
        assert!(
            (lodders03_element_atom_percent("u238") - 99.2745).abs() < 1e-6
        );
    }
}
