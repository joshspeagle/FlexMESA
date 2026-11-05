// SPDX-License-Identifier: LGPL-3.0-or-later
//! Thermal neutrino energy loss rates (`neu`).
//!
//! Ports MESA's `neu` module (`neu/public/neu_def.f90`,
//! `neu/public/neu_lib.f90`, and `neu/private/mod_neu.f90`) evaluating Itoh et
//! al. (1996, ApJS, 102, 411) fitting formulas for specific thermal neutrino
//! energy loss rates ($\mathrm{erg\,g^{-1}\,s^{-1}}$) and their analytical
//! partial derivatives with respect to $(T, \rho, \bar{A}, \bar{Z})$ across
//! five channels:
//!
//! 1. **Pair production (`pair_neu`)**: $e^+ + e^- \to \nu + \bar{\nu}$.
//! 2. **Plasmon decay (`plas_neu`)**: $\tilde{\gamma} \to \nu + \bar{\nu}$.
//! 3. **Photoneutrino (`phot_neu`)**: $e^\pm + \gamma \to e^\pm + \nu +
//!    \bar{\nu}$.
//! 4. **Bremsstrahlung (`brem_neu`)**: $e^- + (Z, A) \to e^- + (Z, A) + \nu +
//!    \bar{\nu}$.
//! 5. **Recombination (`reco_neu`)**: $e^-_{\mathrm{cont}} \to
//!    e^-_{\mathrm{bound}} + \nu + \bar{\nu}$.
//!
//! # Index and Memory-Order Contract
//!
//! - Upstream 1-based channel indices (`pair_neu_type = 1 .. reco_neu_type =
//!   5`) map to 0-based [`NeuChannel`] (`Pair = 0 .. Reco = 4`).
//! - Upstream 1-based derivative indices (`ineu = 1 .. idneu_dzbar = 5`) map to
//!   [`NeuRate`] fields (`val`, `d_t`, `d_rho`, `d_abar`, `d_zbar`) and 0-based
//!   row-major `sources: [[f64; 5]; 5]` (`sources[channel][deriv]`).

#![forbid(unsafe_code)]
#![allow(
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::suboptimal_flops,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

pub mod brem;
pub mod inputs;
pub mod pair_plas;
pub mod phot;
pub mod reco;

use flexmesa_const::{LN10, PI};
use flexmesa_math::{cospi, exp10, log10, sinpi};
use flexmesa_utils::is_bad;
use std::fmt;

pub use brem::brem_neu;
pub use inputs::{NeuInputs, LOG10_TMIN_NEU, TMIN_NEU};
pub use pair_plas::{pair_neu, plas_neu};
pub use phot::phot_neu;
pub use reco::{ifermi12, reco_neu, zfermim12};

/// Errors returned by `flexmesa-neu`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NeuError {
    /// Neither linear nor logarithmic temperature was provided (`info = -2`).
    MissingTemperature,
    /// Neither linear nor logarithmic density was provided (`info = -3`).
    MissingDensity,
    /// Out-of-range or non-finite thermodynamic/composition inputs (`info =
    /// -1`).
    InvalidInput(&'static str),
}

impl fmt::Display for NeuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingTemperature => {
                write!(f, "neither T nor log10_T was provided to neu_get")
            }
            Self::MissingDensity => {
                write!(f, "neither Rho nor log10_Rho was provided to neu_get")
            }
            Self::InvalidInput(msg) => write!(f, "invalid neu input: {msg}"),
        }
    }
}

impl std::error::Error for NeuError {}

/// 0-based thermal neutrino reaction channel index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum NeuChannel {
    /// Pair production ($e^+ + e^- \to \nu + \bar{\nu}$).
    Pair = 0,
    /// Plasmon decay ($\tilde{\gamma} \to \nu + \bar{\nu}$).
    Plas = 1,
    /// Photoneutrino ($e^\pm + \gamma \to e^\pm + \nu + \bar{\nu}$).
    Phot = 2,
    /// Bremsstrahlung ($e^- + (Z, A) \to e^- + (Z, A) + \nu + \bar{\nu}$).
    Brem = 3,
    /// Recombination ($e^-_{\mathrm{cont}} \to e^-_{\mathrm{bound}} + \nu +
    /// \bar{\nu}$).
    Reco = 4,
}

impl NeuChannel {
    /// Ordered array of all five thermal neutrino reaction channels.
    pub const ALL: [Self; 5] =
        [Self::Pair, Self::Plas, Self::Phot, Self::Brem, Self::Reco];
}

/// Per-channel enable flags (`flags(num_neu_types)` in `neu_def.f90`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeuFlags {
    /// Enable pair-production neutrino losses (`pair_neu_type`).
    pub pair: bool,
    /// Enable plasmon-decay neutrino losses (`plas_neu_type`).
    pub plas: bool,
    /// Enable photoneutrino losses (`phot_neu_type`).
    pub phot: bool,
    /// Enable bremsstrahlung neutrino losses (`brem_neu_type`).
    pub brem: bool,
    /// Enable recombination neutrino losses (`reco_neu_type`).
    pub reco: bool,
}

impl NeuFlags {
    /// Enables all five neutrino channels (`[true; 5]`).
    #[must_use]
    pub const fn all() -> Self {
        Self {
            pair: true,
            plas: true,
            phot: true,
            brem: true,
            reco: true,
        }
    }

    /// Disables all five neutrino channels (`[false; 5]`).
    #[must_use]
    pub const fn none() -> Self {
        Self {
            pair: false,
            plas: false,
            phot: false,
            brem: false,
            reco: false,
        }
    }

    /// Standard stellar-evolution default (`reco = false`, all others `true`).
    #[must_use]
    pub const fn default_stellar() -> Self {
        Self {
            pair: true,
            plas: true,
            phot: true,
            brem: true,
            reco: false,
        }
    }
}

impl Default for NeuFlags {
    fn default() -> Self {
        Self::all()
    }
}

/// Specific neutrino energy loss rate ($\mathrm{erg\,g^{-1}\,s^{-1}}$) and its
/// four partial derivatives with respect to $(T, \rho, \bar{A}, \bar{Z})$.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct NeuRate {
    /// Energy loss rate $\epsilon_\nu$ ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
    pub val: f64,
    /// Partial derivative $\partial \epsilon_\nu / \partial T$
    /// ($\mathrm{erg\,g^{-1}\,s^{-1}\,K^{-1}}$).
    pub d_t: f64,
    /// Partial derivative $\partial \epsilon_\nu / \partial \rho$
    /// ($\mathrm{erg\,g^{-1}\,s^{-1}\,cm^3\,g^{-1}}$).
    pub d_rho: f64,
    /// Partial derivative $\partial \epsilon_\nu / \partial \bar{A}$.
    pub d_abar: f64,
    /// Partial derivative $\partial \epsilon_\nu / \partial \bar{Z}$.
    pub d_zbar: f64,
}

impl NeuRate {
    /// Zero rate and zero partial derivatives.
    pub const ZERO: Self = Self {
        val: 0.0,
        d_t: 0.0,
        d_rho: 0.0,
        d_abar: 0.0,
        d_zbar: 0.0,
    };

    /// Converts volumetric rate ($Q$ in $\mathrm{erg\,cm^{-3}\,s^{-1}}$) to
    /// specific rate ($\epsilon = Q / \rho$ in $\mathrm{erg\,g^{-1}\,s^{-1}}$).
    #[must_use]
    pub fn to_specific(self, deni: f64) -> Self {
        let val = self.val * deni;
        Self {
            val,
            d_t: self.d_t * deni,
            d_rho: self.d_rho * deni - val * deni,
            d_abar: self.d_abar * deni,
            d_zbar: self.d_zbar * deni,
        }
    }

    /// Applies the smooth low-temperature cosine cutoff factor $f(T)$ and its
    /// temperature derivative $df/dT$.
    #[must_use]
    pub fn with_cutoff(self, f: f64, df_dt: f64) -> Self {
        Self {
            val: f * self.val,
            d_t: f * self.d_t + df_dt * self.val,
            d_rho: f * self.d_rho,
            d_abar: f * self.d_abar,
            d_zbar: f * self.d_zbar,
        }
    }

    /// Converts to a 5-element array `[val, d_t, d_rho, d_abar, d_zbar]`.
    #[must_use]
    pub const fn to_array(self) -> [f64; 5] {
        [self.val, self.d_t, self.d_rho, self.d_abar, self.d_zbar]
    }
}

/// Complete thermal neutrino loss output (`loss` and `sources` in `neu_get`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct NeuResult {
    /// Total specific neutrino energy loss rate and partial derivatives.
    pub total: NeuRate,
    /// Pair-production channel rate and partial derivatives.
    pub pair: NeuRate,
    /// Plasmon-decay channel rate and partial derivatives.
    pub plas: NeuRate,
    /// Photoneutrino channel rate and partial derivatives.
    pub phot: NeuRate,
    /// Bremsstrahlung channel rate and partial derivatives.
    pub brem: NeuRate,
    /// Recombination channel rate and partial derivatives.
    pub reco: NeuRate,
    /// Total loss array `[val, d_t, d_rho, d_abar, d_zbar]`.
    pub loss: [f64; 5],
    /// Row-major `(5, 5)` channel array `sources[channel][deriv]`.
    pub sources: [[f64; 5]; 5],
}

impl NeuResult {
    /// Returns the [`NeuRate`] for the specified [`NeuChannel`].
    #[must_use]
    pub const fn channel(&self, ch: NeuChannel) -> NeuRate {
        match ch {
            NeuChannel::Pair => self.pair,
            NeuChannel::Plas => self.plas,
            NeuChannel::Phot => self.phot,
            NeuChannel::Brem => self.brem,
            NeuChannel::Reco => self.reco,
        }
    }
}

/// Evaluates specific thermal neutrino energy loss rates and analytical partial
/// derivatives (`neu_get` in `neu_lib.f90` / `neutrinos` in `mod_neu.f90`).
///
/// Accepts `Some(T)` and/or `Some(log10_T)` (computing whichever is `None` via
/// `log10` or `exp10`), and similarly `Some(Rho)` and/or `Some(log10_Rho)`.
///
/// # Errors
///
/// Returns [`NeuError`] if both linear and log inputs are `None` or if
/// thermodynamic/composition inputs are non-positive or out of bounds.
pub fn neu_get(
    t_opt: Option<f64>,
    log10_t_opt: Option<f64>,
    rho_opt: Option<f64>,
    log10_rho_opt: Option<f64>,
    abar: f64,
    zbar: f64,
    log10_tlim: f64,
    flags: NeuFlags,
) -> Result<NeuResult, NeuError> {
    let temp = match (t_opt, log10_t_opt) {
        (Some(t), _) => t,
        (None, Some(lt)) => exp10(lt),
        (None, None) => return Err(NeuError::MissingTemperature),
    };
    if temp <= 0.0 || is_bad(temp) {
        return Err(NeuError::InvalidInput("T must be positive and finite"));
    }
    let logtemp = log10_t_opt.unwrap_or_else(|| log10(temp));
    if logtemp > 20.0 || is_bad(logtemp) {
        return Err(NeuError::InvalidInput("log10_T exceeds 20 or is NaN"));
    }
    let den = match (rho_opt, log10_rho_opt) {
        (Some(r), _) => r,
        (None, Some(lr)) => exp10(lr),
        (None, None) => return Err(NeuError::MissingDensity),
    };
    if den <= 0.0 || is_bad(den) {
        return Err(NeuError::InvalidInput("Rho must be positive and finite"));
    }
    let logden = log10_rho_opt.unwrap_or_else(|| log10(den));
    if logden > 20.0 || is_bad(logden) {
        return Err(NeuError::InvalidInput("log10_Rho exceeds 20 or is NaN"));
    }
    if abar <= 0.0
        || zbar <= 0.0
        || is_bad(abar)
        || is_bad(zbar)
        || is_bad(log10_tlim)
    {
        return Err(NeuError::InvalidInput(
            "abar, zbar, and log10_tlim must be valid",
        ));
    }
    if temp <= TMIN_NEU || logtemp <= LOG10_TMIN_NEU {
        return Ok(NeuResult::default());
    }

    let input = NeuInputs::new(temp, logtemp, den, logden, abar, zbar);
    let mut pair = if flags.pair {
        pair_neu(&input).to_specific(input.deni)
    } else {
        NeuRate::ZERO
    };
    let mut plas = if flags.plas {
        plas_neu(&input).to_specific(input.deni)
    } else {
        NeuRate::ZERO
    };
    let mut phot = if flags.phot {
        phot_neu(&input).to_specific(input.deni)
    } else {
        NeuRate::ZERO
    };
    let mut brem = if flags.brem {
        brem_neu(&input).to_specific(input.deni)
    } else {
        NeuRate::ZERO
    };
    let mut reco = if flags.reco {
        reco_neu(&input).to_specific(input.deni)
    } else {
        NeuRate::ZERO
    };

    if input.logtemp <= log10_tlim && log10_tlim > LOG10_TMIN_NEU {
        let dtlim = log10_tlim - LOG10_TMIN_NEU;
        let phase = (input.logtemp - LOG10_TMIN_NEU) / dtlim;
        let f_cut = 0.5 * (1.0 - cospi(phase));
        let df_dt = 0.5 * PI * sinpi(phase) * (1.0 / (dtlim * temp * LN10));
        plas = plas.with_cutoff(f_cut, df_dt);
        pair = pair.with_cutoff(f_cut, df_dt);
        phot = phot.with_cutoff(f_cut, df_dt);
        brem = brem.with_cutoff(f_cut, df_dt);
        reco = reco.with_cutoff(f_cut, df_dt);
    }

    let total = NeuRate {
        val: plas.val + pair.val + phot.val + brem.val + reco.val,
        d_t: plas.d_t + pair.d_t + phot.d_t + brem.d_t + reco.d_t,
        d_rho: plas.d_rho + pair.d_rho + phot.d_rho + brem.d_rho + reco.d_rho,
        d_abar: plas.d_abar
            + pair.d_abar
            + phot.d_abar
            + brem.d_abar
            + reco.d_abar,
        d_zbar: plas.d_zbar
            + pair.d_zbar
            + phot.d_zbar
            + brem.d_zbar
            + reco.d_zbar,
    };
    if is_bad(total.val) {
        return Err(NeuError::InvalidInput("non-finite total neutrino loss"));
    }
    Ok(NeuResult {
        total,
        pair,
        plas,
        phot,
        brem,
        reco,
        loss: total.to_array(),
        sources: [
            pair.to_array(),
            plas.to_array(),
            phot.to_array(),
            brem.to_array(),
            reco.to_array(),
        ],
    })
}
