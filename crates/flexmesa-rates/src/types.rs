// SPDX-License-Identifier: LGPL-3.0-or-later
//! Core nuclear reaction rate (`rates`) types, options, reaction metadata, and
//! error handling (`rates/public/rates_def.f90`).

use std::error::Error;
use std::fmt::{self, Display};

/// Number of predefined MESA reactions (`1..=307`) in
/// `rates/public/rates_def.f90`.
pub const NUM_PREDEFINED_REACTIONS: usize = 307;

/// Total number of bundled reaction slots (`307` predefined + `r1..r9` generic
/// rate tables = `316`).
pub const NUM_BUNDLED_REACTIONS: usize = 316;

/// Minimum temperature $T_9 = 10^{-2}$ ($\log_{10} T = 7.0$) below which strong
/// REACLIB rates are zeroed (`reaclib_min_T9 = 1d-2` in `rates_def.f90`).
pub const DEFAULT_REACLIB_MIN_T9: f64 = 1.0e-2;

/// Minimum temperature $T_9 = 10^{-5}$ below which analytic `pp`/`CNO`
/// `ratelib` formulas are zeroed (`low_T9pp_cutoff = 1d-5` in `ratelib.f90`).
pub const LOW_T9_PP_CUTOFF: f64 = 1.0e-5;

/// Maximum $\ln \lambda$ clamp before exponentiation in REACLIB evaluations
/// (`max_ln_lambda = 227.955924206411d0`, $\approx 99 \ln 10$).
pub const MAX_LN_LAMBDA: f64 = 227.955_924_206_411;

/// Plasma electrostatic screening formulation (`rates_def.f90`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScreeningMode {
    /// Unscreened reaction rates (`no_screening = 0`, $F_{\mathrm{sc}} = 1$).
    NoScreening = 0,
    /// Salpeter (1954) weak plasma screening (`salpeter_screening = 1`).
    Salpeter = 1,
    /// Graboske et al. (1973), Alastuey & Jancovici (1978), and Itoh et al.
    /// (1979) weak-intermediate-strong screening (`extended_screening = 2`).
    Extended = 2,
    /// Chugunov, DeWitt & Yakovlev (2007) one-component plasma screening
    /// (`chugunov_screening = 3`).
    Chugunov = 3,
}

impl ScreeningMode {
    /// Parses a screening mode from a case-insensitive string name or integer
    /// code (`"no_screening"`, `"salpeter"`, `"extended"`, `"chugunov"`).
    ///
    /// # Errors
    /// Returns [`RatesError::InvalidOption`] if `name` is unrecognized.
    pub fn from_name(name: &str) -> Result<Self, RatesError> {
        match name.trim().to_ascii_lowercase().as_str() {
            "0" | "none" | "no_screening" | "off" => Ok(Self::NoScreening),
            "1" | "salpeter" | "salpeter_screening" => Ok(Self::Salpeter),
            "2" | "extended" | "extended_screening" => Ok(Self::Extended),
            "3" | "chugunov" | "chugunov_screening" => Ok(Self::Chugunov),
            other => Err(RatesError::InvalidOption(format!(
                "unknown screening mode '{other}'"
            ))),
        }
    }
}

/// Ion chemical-potential Coulomb shift formulation (`which_mui_coulomb` in
/// `rates_def.f90`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoulombMuiOption {
    /// Zero Coulomb shift (`No_Coulomb = 0`).
    None = 0,
    /// DeWitt, Graboske & Cooper (1973) two-component fit (`DCW1973 = 1`).
    Dcw1973 = 1,
    /// Slattery, Doolen & DeWitt (1982) liquid OCP fit (`SLR1982 = 2`).
    Slr1982 = 2,
    /// Potekhin, Chabrier & Rogers (2009) electron-ion + ion-ion fit
    /// (`PCR2009 = 3`).
    Pcr2009 = 3,
}

impl CoulombMuiOption {
    /// Parses a [`CoulombMuiOption`] from a case-insensitive name or integer
    /// code (`"none"`, `"dcw1973"`, `"slr1982"`, `"pcr2009"`).
    ///
    /// # Errors
    /// Returns [`RatesError::InvalidOption`] if `name` is unrecognized.
    pub fn from_name(name: &str) -> Result<Self, RatesError> {
        match name.trim().to_ascii_lowercase().as_str() {
            "0" | "none" | "no_coulomb" => Ok(Self::None),
            "1" | "dcw1973" => Ok(Self::Dcw1973),
            "2" | "slr1982" => Ok(Self::Slr1982),
            "3" | "pcr2009" => Ok(Self::Pcr2009),
            other => Err(RatesError::InvalidOption(format!(
                "unknown CoulombMuiOption '{other}'"
            ))),
        }
    }
}

/// Electron screening potential shift $V_s$ formulation (`which_vs_coulomb` in
/// `rates_def.f90`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoulombVsOption {
    /// Zero electron screening shift (`No_Screening = 0`).
    None = 0,
    /// Throwe (1983) Thomas-Fermi approximation (`Throwe1983 = 1`).
    Throwe1983 = 1,
    /// Itoh et al. (2002) linear-response screening shift (`Itoh2002 = 2`).
    Itoh2002 = 2,
}

impl CoulombVsOption {
    /// Parses a [`CoulombVsOption`] from a case-insensitive name or integer
    /// code (`"none"`, `"throwe1983"`, `"itoh2002"`).
    ///
    /// # Errors
    /// Returns [`RatesError::InvalidOption`] if `name` is unrecognized.
    pub fn from_name(name: &str) -> Result<Self, RatesError> {
        match name.trim().to_ascii_lowercase().as_str() {
            "0" | "none" | "no_screening" => Ok(Self::None),
            "1" | "throwe1983" => Ok(Self::Throwe1983),
            "2" | "itoh2002" => Ok(Self::Itoh2002),
            other => Err(RatesError::InvalidOption(format!(
                "unknown CoulombVsOption '{other}'"
            ))),
        }
    }
}

/// Metadata and stoichiometry for a single nuclear reaction (`1..=316`).
#[derive(Debug, Clone)]
pub struct ReactionInfo {
    /// 1-based MESA reaction identifier (`ir`).
    pub id: usize,
    /// Reaction handle name (e.g., `"r1212"`, `"r_c12_ag_o16"`).
    pub name: String,
    /// Evaluation opcode (`0` = default REACLIB, `1` = `interp_pm` table,
    /// `2` = `-1.0`, `3..=28` = analytic `ratelib`, `29..=31` = JINA).
    pub op_code: u8,
    /// Whether this reaction evaluates the reverse (`1`) or forward (`0`)
    /// channel.
    pub is_reverse: bool,
    /// Target REACLIB handle used when `op_code` is `0` or `29`.
    pub target_handle: String,
    /// Number of distinct reactant species (`0..=3`).
    pub num_in: usize,
    /// Stoichiometric counts of reactants (`[u8; 3]`).
    pub in_num: [u8; 3],
    /// 1-based `chem_id` identifiers of reactants (`[u16; 3]`).
    pub in_iso: [u16; 3],
    /// Number of distinct product species (`0..=3`).
    pub num_out: usize,
    /// Stoichiometric counts of products (`[u8; 3]`).
    pub out_num: [u8; 3],
    /// 1-based `chem_id` identifiers of products (`[u16; 3]`).
    pub out_iso: [u16; 3],
    /// Exponent of $Y_e$ in density scaling (`0`, `1`, or `2`).
    pub ye_exp: i8,
    /// Exponent of $\rho$ in density scaling (`N_in - 1`, `0..=4`).
    pub rho_exp: i8,
    /// 1-based MESA energy category (`1..=22`, `ipp = 1`, `icno = 2`, etc.).
    pub category: u8,
    /// 1-based `chem_id` identifiers of screening pair (`[sc1, sc2, sc3]`).
    pub screening_isos: [u16; 3],
    /// 1-based `chem_id` identifiers of weak reaction pair (`[lhs, rhs]`).
    pub weak_isos: [u16; 2],
    /// Standard total nuclear $Q$-value (in $\mathrm{MeV}$).
    pub std_q: f64,
    /// Standard average neutrino energy loss $Q_\nu$ (in $\mathrm{MeV}$).
    pub std_qneu: f64,
    /// Low-temperature laboratory decay rate $\lambda_{\mathrm{low}\,T}$
    /// ($\mathrm{s^{-1}}$).
    pub weak_lowt_rate: f64,
}

/// A single 7-parameter JINA REACLIB polynomial term and its precomputed
/// detailed-balance reverse parameters.
#[derive(Debug, Clone)]
pub struct ReaclibTerm {
    /// REACLIB chapter (`1..=11`).
    pub chapter: u8,
    /// Reaction type flag (`b'-'`, `b'r'`, `b'w'`, `b'e'`, `b'n'`, `b'v'`).
    pub rflag: u8,
    /// Detailed-balance exponent
    /// $\Delta N = N_{\mathrm{in}} - N_{\mathrm{out}}$.
    pub inv_exp: i8,
    /// Seven REACLIB polynomial coefficients $a_0 \dots a_6$.
    pub coeffs: [f64; 7],
    /// First reverse detailed-balance coefficient
    /// $c_1 = \ln(\text{prefactor})$.
    pub inv_c1: f64,
    /// Second reverse detailed-balance coefficient
    /// $c_2 = (B_{\mathrm{in}} - B_{\mathrm{out}}) / (k_{\mathrm{B}} \cdot 10^9\,\mathrm{K})$.
    pub inv_c2: f64,
    /// Ratio of nuclear partition functions
    /// $G_{\mathrm{in}}(T_9) / G_{\mathrm{out}}(T_9)$ across the 24-point grid.
    pub inv_part: [f64; 24],
}

/// Output of plasma electrostatic screening `screen_pair`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenEval {
    /// Multiplicative screening enhancement factor $F_{\mathrm{sc}} = \exp(h_{12})$.
    pub scor: f64,
    /// Partial derivative $\partial F_{\mathrm{sc}} / \partial T$
    /// ($\mathrm{K^{-1}}$).
    pub dscor_dt: f64,
    /// Partial derivative $\partial F_{\mathrm{sc}} / \partial \rho$
    /// ($\mathrm{cm^3\,g^{-1}}$).
    pub dscor_dd: f64,
}

impl ScreenEval {
    /// Unscreened identity result `(scor = 1.0, dscor_dt = 0.0, dscor_dd = 0.0)`.
    pub const UNSCREENED: Self = Self {
        scor: 1.0,
        dscor_dt: 0.0,
        dscor_dd: 0.0,
    };
}

/// Output of a weak reaction evaluation (`weaklib`, `Suzuki2016`, `ecapture`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeakRateEval {
    /// Total weak reaction rate
    /// $\lambda = \lambda_{\mathrm{decay}} + \lambda_{\mathrm{capture}}$
    /// ($\mathrm{s^{-1}}$).
    pub lambda: f64,
    /// Derivative $\partial \lambda / \partial \ln T$ ($\mathrm{s^{-1}}$).
    pub dlambda_dln_t: f64,
    /// Derivative $\partial \lambda / \partial \ln \rho$ ($\mathrm{s^{-1}}$).
    pub dlambda_dln_rho: f64,
    /// Total nuclear $Q$-value (in $\mathrm{MeV}$).
    pub q: f64,
    /// Derivative $\partial Q / \partial \ln T$ (in $\mathrm{MeV}$).
    pub dq_dln_t: f64,
    /// Derivative $\partial Q / \partial \ln \rho$ (in $\mathrm{MeV}$).
    pub dq_dln_rho: f64,
    /// Average neutrino energy loss $Q_\nu$ per reaction (in $\mathrm{MeV}$).
    pub qneu: f64,
    /// Derivative $\partial Q_\nu / \partial \ln T$ (in $\mathrm{MeV}$).
    pub dqneu_dln_t: f64,
    /// Derivative $\partial Q_\nu / \partial \ln \rho$ (in $\mathrm{MeV}$).
    pub dqneu_dln_rho: f64,
}

/// Error type for nuclear reaction rate (`rates`) operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RatesError {
    /// Reaction name or 1-based ID was not found in the catalog.
    UnknownReaction(String),
    /// Weak reaction pair `(lhs, rhs)` was not found in the weak catalog.
    UnknownWeakPair(String, String),
    /// Invalid screening or Coulomb option.
    InvalidOption(String),
    /// Corrupted binary table bundle.
    InvalidTableData(String),
}

impl Display for RatesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownReaction(r) => write!(f, "unknown reaction '{r}'"),
            Self::UnknownWeakPair(l, r) => {
                write!(f, "unknown weak reaction pair '{l} -> {r}'")
            }
            Self::InvalidOption(msg) | Self::InvalidTableData(msg) => {
                write!(f, "{msg}")
            }
        }
    }
}

impl Error for RatesError {}
