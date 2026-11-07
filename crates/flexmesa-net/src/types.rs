// SPDX-License-Identifier: LGPL-3.0-or-later
//! Core nuclear reaction network (`net`) types, constants, reaction-kind
//! classifications, and evaluation results (`net/public/net_def.f90`).

use std::error::Error;
use std::fmt::{self, Display};
use std::sync::Arc;

pub use flexmesa_chem::NUM_CATEGORIES;
use flexmesa_interp1d::CubicCoeffs;
use flexmesa_rates::ReactionInfo;

/// Number of rate-value slots (`[val, d_dT, d_dRho] = 3`).
pub const NUM_RVS: usize = 3;

/// Lower temperature bound $T = 10^{\mathrm{rattab\_tlo}} = 2 \times 10^5\,\mathrm{K}$.
pub const RATTAB_TEMP_LO: f64 = 2.0e5;

/// Lower $\log_{10} T$ bound of the precomputed raw rate table
/// (`rattab_tlo = 5.30102999566398d0` in `rates/public/rates_def.f90`).
pub const RATTAB_TLO: f64 = 5.301_029_995_663_98;

/// Upper $\log_{10} T$ bound of the precomputed raw rate table
/// (`rattab_thi = 10.301029995664d0` in `rates/public/rates_def.f90`).
pub const RATTAB_THI: f64 = 10.301_029_995_664;

/// Number of points per decade in the raw rate table (`2000`).
pub const RATTAB_POINTS_PER_DECADE: usize = 2000;

/// Total number of grid points in the precomputed raw rate table (`10001`).
pub const NRATTAB: usize = 10001;

/// Step size $\Delta \log_{10} T$ in the precomputed raw rate table.
pub const RATTAB_TSTP: f64 = (RATTAB_THI - RATTAB_TLO) / ((NRATTAB - 1) as f64);

/// Maximum $\log_{10} T$ above which raw reaction rates are clamped
/// (`max_safe_logT_for_rates = 10.30102999d0` in `rates_def.f90`).
pub const MAX_SAFE_LOGT_FOR_RATES: f64 = 10.301_029_99;

/// Number of reactions in `approx21.net` (`116`).
pub const APPROX21_NUM_REACTIONS: usize = 116;

/// Number of MESA reactions in `approx21.net` (`93`).
pub const APPROX21_NUM_MESA_REACTIONS: usize = 93;

/// Number of reactions in `approx21_plus_co56.net` (`117`).
pub const APPROX21_CO56_NUM_REACTIONS: usize = 117;

/// Number of MESA reactions in `approx21_plus_co56.net` (`94`).
pub const APPROX21_CO56_NUM_MESA_REACTIONS: usize = 94;

/// 1-based `chem_id` of `neut` (`1`).
pub const INEUT: usize = 1;
/// 1-based `chem_id` of `h1` (`2`).
pub const IH1: usize = 2;
/// 1-based `chem_id` of `prot` (`3`).
pub const IPROTON: usize = 3;
/// 1-based `chem_id` of `h2` (`4`).
pub const IH2: usize = 4;
/// 1-based `chem_id` of `he3` (`6`).
pub const IHE3: usize = 6;
/// 1-based `chem_id` of `he4` (`7`).
pub const IHE4: usize = 7;
/// 1-based `chem_id` of `c12` (`40`).
pub const IC12: usize = 40;
/// 1-based `chem_id` of `n14` (`53`).
pub const IN14: usize = 53;
/// 1-based `chem_id` of `o16` (`66`).
pub const IO16: usize = 66;
/// 1-based `chem_id` of `ne20` (`102`).
pub const INE20: usize = 102;
/// 1-based `chem_id` of `mg24` (`156`).
pub const IMG24: usize = 156;
/// 1-based `chem_id` of `si28` (`219`).
pub const ISI28: usize = 219;
/// 1-based `chem_id` of `s32` (`289`).
pub const IS32: usize = 289;
/// 1-based `chem_id` of `ar36` (`366`).
pub const IAR36: usize = 366;
/// 1-based `chem_id` of `ca40` (`450`).
pub const ICA40: usize = 450;
/// 1-based `chem_id` of `ti44` (`539`).
pub const ITI44: usize = 539;
/// 1-based `chem_id` of `cr48` (`634`).
pub const ICR48: usize = 634;
/// 1-based `chem_id` of `fe52` (`733`).
pub const IFE52: usize = 733;
/// 1-based `chem_id` of `fe54` (`735`).
pub const IFE54: usize = 735;
/// 1-based `chem_id` of `fe56` (`737`).
pub const IFE56: usize = 737;
/// 1-based `chem_id` of `co55` (`785`).
pub const ICO55: usize = 785;
/// 1-based `chem_id` of `co56` (`786`).
pub const ICO56: usize = 786;
/// 1-based `chem_id` of `co57` (`787`).
pub const ICO57: usize = 787;
/// 1-based `chem_id` of `ni56` (`837`).
pub const INI56: usize = 837;

/// REACLIB reaction pairing classification (`net_def.f90` lines 53-63).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ReaclibKind {
    /// Basic $(n, \gamma)$ forward/reverse pair (`ng_kind = 1`).
    Ng = 1,
    /// Basic $(p, n)$ forward/reverse pair (`pn_kind = 2`).
    Pn = 2,
    /// Basic $(p, \gamma)$ forward/reverse pair (`pg_kind = 3`).
    Pg = 3,
    /// Basic $(\alpha, p)$ forward/reverse pair (`ap_kind = 4`).
    Ap = 4,
    /// Basic $(\alpha, n)$ forward/reverse pair (`an_kind = 5`).
    An = 5,
    /// Basic $(\alpha, \gamma)$ forward/reverse pair (`ag_kind = 6`).
    Ag = 6,
    /// General $1 \leftrightarrow 1$ stoichiometric pair (`general_one_one_kind = 7`).
    GeneralOneOne = 7,
    /// General $2 \leftrightarrow 1$ stoichiometric pair (`general_two_one_kind = 8`).
    GeneralTwoOne = 8,
    /// General $2 \leftrightarrow 2$ stoichiometric pair (`general_two_two_kind = 9`).
    GeneralTwoTwo = 9,
    /// Unpaired or reverse partner reaction (`other_kind = 10`).
    Other = 10,
}

/// Broad nuclear reaction kind (`net_def.f90` lines 47-50).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ReactionKind {
    /// Weak decay or electron capture (`weak_kind = 1`).
    Weak = 1,
    /// Neutron-involving strong reaction without protons (`neut_kind = 2`).
    Neut = 2,
    /// Proton-involving strong reaction (`prot_kind = 3`).
    Prot = 3,
    /// Other strong reaction (`other_strong_kind = 4`).
    OtherStrong = 4,
}

/// Precomputed pair-screening charge parameters (`AZ_info` in `net_screen.f90`).
#[derive(Debug, Clone, Copy, Default)]
pub struct ScreenPairParams {
    /// $(Z_1 + Z_2)^{1/3}$.
    pub zs13: f64,
    /// $(Z_1 + Z_2)^{5/3} - Z_1^{5/3} - Z_2^{5/3}$.
    pub zhat: f64,
    /// $(Z_1 + Z_2)^{5/12} - Z_1^{5/12} - Z_2^{5/12}$.
    pub zhat2: f64,
    /// $\tfrac{5}{3} \ln(Z_1 Z_2 / (Z_1 + Z_2))$.
    pub lzav: f64,
    /// $(Z_1^2 Z_2^2 A_1 A_2 / (A_1 + A_2))^{1/3}$.
    pub aznut: f64,
    /// $1 / (Z_1 + Z_2)^{1/3}$.
    pub zs13inv: f64,
}

/// Complete compiled nuclear reaction network definition (`Net_General_Info`).
#[derive(Debug, Clone)]
pub struct NetGeneralInfo {
    /// Network filename (e.g., `"basic.net"`, `"approx21.net"`).
    pub net_filename: String,
    /// Number of active isotopes (`species`).
    pub num_isos: usize,
    /// Number of MESA reactions in the network (`num_reactions`).
    pub num_reactions: usize,
    /// Number of weak reactions in the network (`num_wk_reactions`).
    pub num_wk_reactions: usize,
    /// 1-based `chem_id` for each network species (`len = num_isos`).
    pub chem_id: Vec<usize>,
    /// Lookup table mapping 1-based `chem_id` (`1..=num_chem_isos`) to 1-based
    /// network species index (`0` if not in net).
    pub net_iso: Vec<usize>,
    /// Metadata for each reaction in the network (`len = num_reactions`).
    pub reactions: Vec<ReactionInfo>,
    /// Lookup from reaction name to 1-based network reaction index (`1..=num_reactions`).
    pub net_reaction_by_name: std::collections::HashMap<String, usize>,
    /// Lookup from 1-based `rates` reaction ID to 1-based network reaction index.
    pub net_reaction_by_id: std::collections::HashMap<usize, usize>,
    /// Pairing classification for each reaction (`len = num_reactions`).
    pub reaction_reaclib_kind: Vec<ReaclibKind>,
    /// Broad reaction classification (`len = num_reactions`).
    pub reaction_kind: Vec<ReactionKind>,
    /// Reverse reaction ID for paired reactions (`len = num_reactions`).
    pub reverse_id_for_kind_ne_other: Vec<usize>,
    /// 1-based weak reaction index (`0` if not weak, `len = num_reactions`).
    pub weak_reaction_index: Vec<usize>,
    /// 1-based network reaction index for each weak reaction (`len = num_wk_reactions`).
    pub weak_reaction_num: Vec<usize>,
    /// 1-based `WeakTable` ID (`0` if not in `weakreactions.tables`, `len = num_wk_reactions`).
    pub weaklib_ids: Vec<usize>,
    /// Maximum atomic number $Z$ among reactants/products (`len = num_reactions`).
    pub reaction_max_z: Vec<i32>,
    /// Mass number $A$ of the highest-$Z$ reactant/product (`len = num_reactions`).
    pub reaction_max_z_plus_n: Vec<i32>,
    /// Precomputed $Z_i^{1.58}$ for each species (`len = num_isos`).
    pub z158: Vec<f64>,
    /// Precomputed $Z_i^{2.5}$ for each species (`len = num_isos`).
    pub z52: Vec<f64>,
    /// Precomputed ion mass excess in grams (`len = num_isos`).
    pub mion: Vec<f64>,
    /// Precomputed screening parameters for each reaction (`len = num_reactions`).
    pub screen_params: Vec<ScreenPairParams>,
    /// Precomputed 10001-point `interp_m3q` cubic spline coefficients on `logttab`
    /// for each reaction (`len = num_reactions`).
    pub rattab_splines: Vec<Arc<Vec<CubicCoeffs>>>,
    /// Lower $\log_{10} T$ cutoff (`rattab_tlo`).
    pub log_tcut_lo: f64,
    /// Blending $\log_{10} T$ cutoff (`rattab_tlo + 0.1`).
    pub log_tcut_lim: f64,
    /// Whether this network uses the hardwired `approx21` reduced formalism.
    pub doing_approx21: bool,
    /// Whether `co56` is included in `approx21` (`approx21_plus_co56`).
    pub add_co56_to_approx21: bool,
    /// 1-based `chem_id` of the neutron-rich $Y_e$ sink isotope (`cr56`, `cr60`, etc.).
    pub approx21_ye_iso: usize,
    /// Effective neutron count $(N_{\mathrm{ye}} - N_{{}^{56}\mathrm{Fe}}) - (Z_{{}^{56}\mathrm{Fe}} - Z_{\mathrm{ye}})$.
    pub fe56ec_n_neut: i32,
    /// Fake $T_9$ factor for `fe56ec` in `approx21` (`1e-4`).
    pub fe56ec_fake_factor: f64,
    /// Minimum $T$ for `fe56ec_fake_factor` (`3e9`).
    pub min_t_for_fe56ec_fake_factor: f64,
}

impl NetGeneralInfo {
    /// Returns the 1-based network reaction index (`1..=num_reactions`) for `name`.
    #[inline]
    #[must_use]
    pub fn reaction_index_by_name(&self, name: &str) -> Option<usize> {
        self.net_reaction_by_name.get(name.trim()).copied()
    }
}

/// Complete output of a single-point nuclear network evaluation (`net_get`).
#[derive(Debug, Clone)]
pub struct NetEvalResult {
    /// Net nuclear energy generation rate $\epsilon_{\mathrm{nuc}}$
    /// ($\mathrm{erg\,g^{-1}\,s^{-1}}$, excluding neutrino losses).
    pub eps_nuc: f64,
    /// Partial derivative $\partial \epsilon_{\mathrm{nuc}} / \partial \rho$ at
    /// constant $T, X_i$.
    pub d_eps_nuc_drho: f64,
    /// Partial derivative $\partial \epsilon_{\mathrm{nuc}} / \partial T$ at
    /// constant $\rho, X_i$.
    pub d_eps_nuc_dt: f64,
    /// Partial derivatives $\partial \epsilon_{\mathrm{nuc}} / \partial X_i$
    /// (`len = num_isos`).
    pub d_eps_nuc_dx: Vec<f64>,
    /// Mass-fraction time derivatives $\dot{X}_i = dX_i/dt$ ($\mathrm{s^{-1}}$,
    /// `len = num_isos`).
    pub dxdt: Vec<f64>,
    /// Partial derivatives $\partial \dot{X}_i / \partial \rho$ (`len = num_isos`).
    pub d_dxdt_drho: Vec<f64>,
    /// Partial derivatives $\partial \dot{X}_i / \partial T$ (`len = num_isos`).
    pub d_dxdt_dt: Vec<f64>,
    /// Dense Jacobian $\partial \dot{X}_i / \partial X_j$ in row-major
    /// `(num_isos, num_isos)` layout (`d_dxdt_dx[i * num_isos + j]`).
    pub d_dxdt_dx: Vec<f64>,
    /// Nuclear energy generation split across the 24 `chem_def` categories
    /// (`[f64; 24]`).
    pub eps_nuc_categories: [f64; NUM_CATEGORIES],
    /// Total nuclear reaction neutrino energy loss rate $\epsilon_\nu$
    /// ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
    pub eps_neu_total: f64,
    /// Unscreened reaction rates (`len = num_reactions`, or `99`/`103` for `approx21`).
    pub rate_raw: Vec<f64>,
    /// Screened reaction rates (`len = num_reactions`, or `99`/`103` for `approx21`).
    pub rate_screened: Vec<f64>,
}

/// Error type for nuclear reaction network (`net`) operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetError {
    /// Unknown `.net` file name.
    UnknownNetFile(String),
    /// Syntax or semantic error while parsing a `.net` definition.
    ParseError(String),
    /// Invalid or unknown isotope or reaction name.
    InvalidReactionOrIso(String),
    /// Missing required reaction in network.
    MissingReaction(String),
    /// Failure during rate interpolation, EOS lookup, or stiff ODE integration.
    IntegrationError(String),
}

impl Display for NetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownNetFile(s) => write!(f, "unknown net file '{s}'"),
            Self::MissingReaction(s) => write!(f, "missing reaction '{s}'"),
            Self::ParseError(s)
            | Self::InvalidReactionOrIso(s)
            | Self::IntegrationError(s) => write!(f, "{s}"),
        }
    }
}

impl Error for NetError {}
