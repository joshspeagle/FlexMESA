//! Public types, options, configuration, and output structures for `flexmesa-kap`.
//!
//! Mirrors the constants and control structures in upstream Fortran MESA's
//! `kap/public/kap_def.f90` and `kap/defaults/kap.defaults`.

use std::fmt;

/// Index of the low-temperature radiative opacity fraction in `kap_fracs`.
pub const I_FRAC_LOW_T: usize = 0;

/// Index of the high-temperature radiative opacity fraction in `kap_fracs`.
pub const I_FRAC_HIGH_T: usize = 1;

/// Index of the Type 2 (C/O-enhanced) high-temperature fraction in `kap_fracs`.
pub const I_FRAC_TYPE2: usize = 2;

/// Index of the Compton-scattering opacity fraction in `kap_fracs`.
pub const I_FRAC_COMPTON: usize = 3;

/// Total number of opacity regime blend fractions (`4`).
pub const NUM_KAP_FRACS: usize = 4;

/// High-temperature fixed-metal (Type 1) opacity table option (`kap_file_prefix`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KapOption {
    /// Grevesse & Noels (1993) solar mixture (`gn93`).
    Gn93,
    /// Grevesse & Sauval (1998) solar mixture (`gs98`, default).
    Gs98,
    /// Asplund et al. (2009) solar mixture (`a09`).
    A09,
    /// Opacity Project with Grevesse & Sauval (1998) solar mixture (`OP_gs98`).
    OpGs98,
    /// Opacity Project with Asplund et al. (2009) solar mixture (`OP_a09`).
    OpA09,
}

impl KapOption {
    /// Return the MESA table prefix string for this fixed-metal option.
    #[must_use]
    pub const fn prefix(self) -> &'static str {
        match self {
            Self::Gn93 => "gn93",
            Self::Gs98 => "gs98",
            Self::A09 => "a09",
            Self::OpGs98 => "OP_gs98",
            Self::OpA09 => "OP_a09_nans_removed_by_hand",
        }
    }

    /// Parse a MESA `kap_file_prefix` string into a [`KapOption`].
    #[must_use]
    pub fn from_prefix(s: &str) -> Option<Self> {
        match s.trim() {
            "gn93" => Some(Self::Gn93),
            "gs98" => Some(Self::Gs98),
            "a09" => Some(Self::A09),
            "OP_gs98" => Some(Self::OpGs98),
            "OP_a09" | "OP_a09_nans_removed_by_hand" => Some(Self::OpA09),
            _ => None,
        }
    }
}

/// High-temperature C/O-enhanced (Type 2) opacity table option (`kap_CO_prefix`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KapCoOption {
    /// Grevesse & Noels (1993) C/O-enhanced mixture (`gn93_co`).
    Gn93Co,
    /// Grevesse & Sauval (1998) C/O-enhanced mixture (`gs98_co`, default).
    Gs98Co,
    /// Asplund et al. (2009) C/O-enhanced mixture (`a09_co`).
    A09Co,
}

impl KapCoOption {
    /// Return the MESA table prefix string for this C/O-enhanced option.
    #[must_use]
    pub const fn prefix(self) -> &'static str {
        match self {
            Self::Gn93Co => "gn93_co",
            Self::Gs98Co => "gs98_co",
            Self::A09Co => "a09_co",
        }
    }

    /// Return the reference metal mass fractions `(fC, fN, fO, fNe)` for this mixture.
    #[must_use]
    pub const fn ref_zfracs(self) -> (f64, f64, f64, f64) {
        match self {
            Self::Gn93Co => (0.173_285, 0.053_152, 0.482_273, 0.098_668),
            Self::Gs98Co => (0.165_659, 0.048_844, 0.453_345, 0.114_576),
            Self::A09Co => (0.176_246, 0.050_427, 0.428_279, 0.093_860),
        }
    }

    /// Parse a MESA `kap_CO_prefix` string into a [`KapCoOption`].
    #[must_use]
    pub fn from_prefix(s: &str) -> Option<Self> {
        match s.trim() {
            "gn93_co" => Some(Self::Gn93Co),
            "gs98_co" => Some(Self::Gs98Co),
            "a09_co" => Some(Self::A09Co),
            _ => None,
        }
    }
}

/// Low-temperature molecular/dust opacity table option (`kap_lowT_prefix`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KapLowTOption {
    /// Ferguson et al. (2005) with Grevesse & Sauval (1998) mixture (`lowT_fa05_gs98`, default).
    LowTFa05Gs98,
    /// Ferguson et al. (2005) with Grevesse & Noels (1993) mixture (`lowT_fa05_gn93`).
    LowTFa05Gn93,
    /// Ferguson et al. (2005) with Asplund et al. (2009) mixture (`lowT_fa05_a09p`).
    LowTFa05A09p,
    /// Freedman et al. (2011) pressure-broadened low-T tables (`lowT_Freedman11`).
    LowTFreedman11,
    /// Lederer & Aringer (2009) C/N-variable low-T tables (`kapCN`).
    KapCn,
    /// Marigo & Aringer (2009) AESOPUS C/N/O-variable low-T tables (`AESOPUS`).
    Aesopus,
}

impl KapLowTOption {
    /// Return the MESA table prefix string for this low-temperature option.
    #[must_use]
    pub const fn prefix(self) -> &'static str {
        match self {
            Self::LowTFa05Gs98 => "lowT_fa05_gs98",
            Self::LowTFa05Gn93 => "lowT_fa05_gn93",
            Self::LowTFa05A09p => "lowT_fa05_a09p",
            Self::LowTFreedman11 => "lowT_Freedman11",
            Self::KapCn => "kapCN",
            Self::Aesopus => "AESOPUS",
        }
    }

    /// Parse a MESA `kap_lowT_prefix` string into a [`KapLowTOption`].
    #[must_use]
    pub fn from_prefix(s: &str) -> Option<Self> {
        match s.trim() {
            "lowT_fa05_gs98" => Some(Self::LowTFa05Gs98),
            "lowT_fa05_gn93" => Some(Self::LowTFa05Gn93),
            "lowT_fa05_a09p" => Some(Self::LowTFa05A09p),
            "lowT_Freedman11" => Some(Self::LowTFreedman11),
            "kapCN" => Some(Self::KapCn),
            "AESOPUS" => Some(Self::Aesopus),
            _ => None,
        }
    }
}

/// Control configuration for opacity evaluation (`Kap_General_Info` in `kap_def.f90`).
#[derive(Clone, Debug)]
pub struct KapConfig {
    /// Base metallicity $Z_{\mathrm{base}}$ (`-1.0` if unset).
    pub zbase: f64,
    /// Fixed-metal Type 1 opacity option (`gs98` by default).
    pub kap_option: KapOption,
    /// C/O-enhanced Type 2 opacity option (`gs98_co` by default).
    pub kap_co_option: KapCoOption,
    /// Low-temperature opacity option (`lowT_fa05_gs98` by default).
    pub kap_low_t_option: KapLowTOption,
    /// Lower $\log_{10}(T/\mathrm{K})$ boundary for the low-$T$ to high-$T$ blend (`3.80`).
    pub kap_blend_log_t_lower_bdy: f64,
    /// Upper $\log_{10}(T/\mathrm{K})$ boundary for the low-$T$ to high-$T$ blend (`3.88`).
    pub kap_blend_log_t_upper_bdy: f64,
    /// Use piecewise-monotonic cubic interpolation in $X$ (`false` by default).
    pub cubic_interpolation_in_x: bool,
    /// Use piecewise-monotonic cubic interpolation in $Z$ (`false` by default).
    pub cubic_interpolation_in_z: bool,
    /// Include electron thermal conduction opacity (`true` by default).
    pub include_electron_conduction: bool,
    /// Apply Blouin et al. (2020) moderate-coupling H/He conduction correction (`true`).
    pub use_blouin_conductive_opacities: bool,
    /// Evaluate Type 1 tables at `zbase` rather than local `Z` (`false` by default).
    pub use_zbase_for_type1: bool,
    /// Enable Type 2 C/O-enhanced opacity evaluation (`false` by default).
    pub use_type2_opacities: bool,
    /// Hydrogen mass fraction $X$ below which Type 2 opacities are fully enabled (`1e-6`).
    pub kap_type2_full_on_x: f64,
    /// Hydrogen mass fraction $X$ above which Type 2 opacities are fully disabled (`1e-3`).
    pub kap_type2_full_off_x: f64,
    /// Metallicity enhancement $Z - Z_{\mathrm{base}}$ below which Type 2 is disabled (`1e-3`).
    pub kap_type2_full_off_dz: f64,
    /// Metallicity enhancement $Z - Z_{\mathrm{base}}$ above which Type 2 is enabled (`1e-2`).
    pub kap_type2_full_on_dz: f64,
    /// Upper $\log_{10}(T/\mathrm{K})$ boundary where Compton opacity reaches full weight (`8.69`).
    pub log_t_compton_blend_hi: f64,
    /// Lower $\log_{10} R$ boundary where Compton opacity reaches full weight (`-7.99`).
    pub log_r_compton_blend_lo: f64,
}

impl Default for KapConfig {
    fn default() -> Self {
        Self {
            zbase: -1.0,
            kap_option: KapOption::Gs98,
            kap_co_option: KapCoOption::Gs98Co,
            kap_low_t_option: KapLowTOption::LowTFa05Gs98,
            kap_blend_log_t_lower_bdy: 3.80,
            kap_blend_log_t_upper_bdy: 3.88,
            cubic_interpolation_in_x: false,
            cubic_interpolation_in_z: false,
            include_electron_conduction: true,
            use_blouin_conductive_opacities: true,
            use_zbase_for_type1: false,
            use_type2_opacities: false,
            kap_type2_full_on_x: 1.0e-6,
            kap_type2_full_off_x: 1.0e-3,
            kap_type2_full_off_dz: 0.001,
            kap_type2_full_on_dz: 0.01,
            log_t_compton_blend_hi: 8.69,
            log_r_compton_blend_lo: -7.99,
        }
    }
}

/// Complete single-zone opacity evaluation output (`kap_get` / `Get_kap_Results`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KapEvalOutput {
    /// Total Rosseland mean opacity $\kappa$ in $\mathrm{cm^2\,g^{-1}}$.
    pub kap: f64,
    /// Base-10 logarithm $\log_{10}(\kappa / \mathrm{cm^2\,g^{-1}})$.
    pub log10_kap: f64,
    /// Logarithmic density derivative $(\partial \ln \kappa / \partial \ln \rho)_T$.
    pub dlnkap_dln_rho: f64,
    /// Logarithmic temperature derivative $(\partial \ln \kappa / \partial \ln T)_\rho$.
    pub dlnkap_dln_t: f64,
    /// Regime blend fractions `[frac_lowT, frac_highT, frac_Type2, frac_Compton]`.
    pub kap_fracs: [f64; NUM_KAP_FRACS],
}

/// Errors returned by opacity table loading and evaluation routines.
#[derive(Clone, Debug, PartialEq)]
pub enum KapError {
    /// Requested temperature or density lies outside table boundaries.
    OutOfBounds(&'static str),
    /// Invalid composition or missing `zbase` for Type 2 evaluation.
    InvalidComposition(&'static str),
    /// Requested table slice is not available.
    MissingTable(&'static str),
    /// Numerical failure (NaN/Inf or singular interpolation).
    NumericalFailure(&'static str),
}

impl fmt::Display for KapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds(msg) => write!(f, "KapError::OutOfBounds({msg})"),
            Self::InvalidComposition(msg) => {
                write!(f, "KapError::InvalidComposition({msg})")
            }
            Self::MissingTable(msg) => {
                write!(f, "KapError::MissingTable({msg})")
            }
            Self::NumericalFailure(msg) => {
                write!(f, "KapError::NumericalFailure({msg})")
            }
        }
    }
}

impl std::error::Error for KapError {}
