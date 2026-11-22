// SPDX-License-Identifier: LGPL-3.0-or-later
//! Embedded binary table loader (`kap_tables.bin`) for `flexmesa-kap`.
//!
//! Lazily unpacks and fits 2D PSPLINE bicubic splines on first access per
//! `(Z, X)` or `(Z, X, dXC, dXO)` slice using thread-safe [`OnceLock`] cells.

use crate::loader_ascii::load_or_build_kap_tables;
use crate::spline::fit_bicubic_not_a_knot;
use crate::types::{KapConfig, KapError, KapLowTOption};
use flexmesa_utils::is_bad;
use std::sync::OnceLock;

/// Reference $\delta X_{\mathrm{C}}, \delta X_{\mathrm{O}}$ grid in Type 2 tables (`kap_def.f90`).
pub const KAP_CO_DXS: [f64; 8] =
    [0.00, 0.01, 0.03, 0.10, 0.20, 0.40, 0.60, 1.00];

/// Compute $\log_{10}(Z + 10^{-3} + \delta X)$ (`get_dX_lookup` in `load_co_kap.f90`).
#[must_use]
pub fn get_dx_lookup(dx: f64, z: f64) -> f64 {
    flexmesa_math::log10(z + 1.0e-3 + dx)
}

/// Lazily fitted 2D bicubic spline cache cell.
pub(crate) type SplineCell =
    OnceLock<Result<(Vec<[f64; 4]>, bool, bool), KapError>>;

fn fit_kap_spline(
    log_rs: &[f64],
    log_ts: &[f64],
    vals: &[f64],
    context: &'static str,
    table_label: &str,
) -> Result<(Vec<[f64; 4]>, bool, bool), KapError> {
    let n_t = log_ts.len();
    for (idx, &val) in vals.iter().enumerate() {
        if is_bad(val) {
            let i_r = idx.checked_div(n_t).unwrap_or(idx);
            let j_t = idx.checked_rem(n_t).unwrap_or(0);
            return Err(KapError::TableIo(format!(
                "non-finite input log_kap ({val}) in {table_label} cell=({i_r}, {j_t})"
            )));
        }
    }
    let fitted = fit_bicubic_not_a_knot(log_rs, log_ts, vals)
        .map_err(|_| KapError::NumericalFailure(context))?;
    for (cell, c4) in fitted.0.iter().enumerate() {
        let i_r = cell.checked_div(n_t).unwrap_or(cell);
        let j_t = cell.checked_rem(n_t).unwrap_or(0);
        for (m, &c) in c4.iter().enumerate() {
            if is_bad(c) {
                return Err(KapError::TableIo(format!(
                    "non-finite spline coefficient ({c}) in {table_label} cell=({i_r}, {j_t}) coeff={m}"
                )));
            }
        }
    }
    Ok(fitted)
}

/// Single 2D $(\log_{10} R, \log_{10} T)$ fixed-metal or low-$T$ opacity table.
pub struct KapXTable {
    /// Hydrogen mass fraction $X$.
    pub x: f64,
    /// Metallicity $Z$.
    pub z: f64,
    /// Minimum $\log_{10} R$.
    pub log_r_min: f64,
    /// Maximum $\log_{10} R$.
    pub log_r_max: f64,
    /// Minimum $\log_{10}(T/\mathrm{K})$.
    pub log_t_min: f64,
    /// Maximum $\log_{10}(T/\mathrm{K})$.
    pub log_t_max: f64,
    /// Tabulated $\log_{10} R$ grid.
    pub log_rs: Vec<f64>,
    /// Tabulated $\log_{10}(T/\mathrm{K})$ grid.
    pub log_ts: Vec<f64>,
    /// Raw $1000 \times \log_{10} \kappa$ integer values (`None` if not embedded).
    pub raw_i16: Option<Vec<i16>>,
    /// Lazily fitted `(coeffs, ili_log_rs, ili_log_ts)` spline coefficients.
    pub(crate) spline: SplineCell,
}

impl KapXTable {
    /// Return the fitted bicubic spline `(coeffs, ili_log_rs, ili_log_ts)` for this table.
    ///
    /// # Errors
    /// Returns [`KapError::MissingTable`] if this slice is not embedded, or
    /// [`KapError::NumericalFailure`] if bicubic spline fitting fails.
    pub fn spline_data(
        &self,
    ) -> Result<&(Vec<[f64; 4]>, bool, bool), KapError> {
        let raw = self.raw_i16.as_ref().ok_or(KapError::MissingTable(
            "fixed X table slice not embedded",
        ))?;
        self.spline
            .get_or_init(|| {
                let vals: Vec<f64> =
                    raw.iter().map(|&v| f64::from(v) / 1000.0).collect();
                let label = format!("fixed(X={}, Z={})", self.x, self.z);
                fit_kap_spline(
                    &self.log_rs,
                    &self.log_ts,
                    &vals,
                    "bicubic spline fit failed on fixed kap table",
                    &label,
                )
            })
            .as_ref()
            .map_err(Clone::clone)
    }
}

/// Set of $X$ tables at a single metallicity $Z$.
pub struct KapZTable {
    /// Metallicity $Z$.
    pub z: f64,
    /// $X$ tables in order of increasing $X$.
    pub x_tables: Vec<KapXTable>,
}

/// Named collection of fixed-metal or low-$T$ `KapZTable`s (`gs98`, `a09`, `lowT_fa05_gs98`).
pub struct KapFixedSet {
    /// Prefix name (e.g. `"gs98"`).
    pub prefix: String,
    /// Tables ordered by increasing $Z$.
    pub z_tables: Vec<KapZTable>,
}

/// Single $(\delta X_{\mathrm{C}}, \delta X_{\mathrm{O}})$ sub-table inside a `KapCoXTable`.
pub struct KapCoSubTable {
    /// Carbon enhancement $\delta X_{\mathrm{C}}$.
    pub dxc: f64,
    /// Oxygen enhancement $\delta X_{\mathrm{O}}$.
    pub dxo: f64,
    /// Precomputed $\log_{10}(Z + 10^{-3} + \delta X_{\mathrm{C}})$.
    pub dxc_lookup: f64,
    /// Precomputed $\log_{10}(Z + 10^{-3} + \delta X_{\mathrm{O}})$.
    pub dxo_lookup: f64,
    /// Raw $1000 \times \log_{10} \kappa$ values (`None` if not embedded).
    pub raw_i16: Option<Vec<i16>>,
    /// Lazily fitted bicubic spline `(coeffs, ili_log_rs, ili_log_ts)`.
    pub(crate) spline: SplineCell,
}

impl KapCoSubTable {
    /// Return the fitted bicubic spline for this $(\delta X_{\mathrm{C}}, \delta X_{\mathrm{O}})$ slice.
    ///
    /// # Errors
    /// Returns [`KapError::MissingTable`] if this sub-table is not embedded, or
    /// [`KapError::NumericalFailure`] if bicubic spline fitting fails.
    pub fn spline_data(
        &self,
        log_rs: &[f64],
        log_ts: &[f64],
    ) -> Result<&(Vec<[f64; 4]>, bool, bool), KapError> {
        let raw = self
            .raw_i16
            .as_ref()
            .ok_or(KapError::MissingTable("CO subtable not embedded"))?;
        self.spline
            .get_or_init(|| {
                let vals: Vec<f64> =
                    raw.iter().map(|&v| f64::from(v) / 1000.0).collect();
                let label = format!("CO(dXC={}, dXO={})", self.dxc, self.dxo);
                fit_kap_spline(
                    log_rs,
                    log_ts,
                    &vals,
                    "bicubic spline fit failed on CO kap table",
                    &label,
                )
            })
            .as_ref()
            .map_err(Clone::clone)
    }
}

/// Set of `58` C/O-enhanced sub-tables at fixed $(Z_{\mathrm{base}}, X)$.
pub struct KapCoXTable {
    /// Hydrogen mass fraction $X$.
    pub x: f64,
    /// Base metallicity $Z_{\mathrm{base}}$.
    pub z: f64,
    /// Whether this $(Z_{\mathrm{base}}, X)$ table was embedded.
    pub present: bool,
    /// Number of sub-tables (`58`).
    pub num_co_tables: usize,
    /// Number of sub-tables with $\delta X_{\mathrm{C}} > \delta X_{\mathrm{O}}$ (`26`).
    pub num_dxc_gt_dxo: usize,
    /// Minimum $\log_{10} R$.
    pub log_r_min: f64,
    /// Maximum $\log_{10} R$.
    pub log_r_max: f64,
    /// Minimum $\log_{10}(T/\mathrm{K})$.
    pub log_t_min: f64,
    /// Maximum $\log_{10}(T/\mathrm{K})$.
    pub log_t_max: f64,
    /// Tabulated $\log_{10} R$ grid.
    pub log_rs: Vec<f64>,
    /// Tabulated $\log_{10}(T/\mathrm{K})$ grid.
    pub log_ts: Vec<f64>,
    /// 1-based sub-table index lookup `co_table_numbers[idxc][idxo]` (`-1` if none).
    pub co_table_numbers: [[i32; 8]; 8],
    /// 1-based next-$\delta X_{\mathrm{O}}$ table index for each sub-table.
    pub next_dxo_table: Vec<i32>,
    /// 1-based next-$\delta X_{\mathrm{C}}$ table index for each sub-table.
    pub next_dxc_table: Vec<i32>,
    /// Sub-tables (`0..num_co_tables`).
    pub co_tables: Vec<KapCoSubTable>,
}

/// C/O-enhanced tables at a single $Z_{\mathrm{base}}$.
pub struct KapCoZTable {
    /// Base metallicity $Z_{\mathrm{base}}$.
    pub zbase: f64,
    /// Reference carbon fraction in $Z_{\mathrm{base}}$.
    pub zfrac_c: f64,
    /// Reference nitrogen fraction in $Z_{\mathrm{base}}$.
    pub zfrac_n: f64,
    /// Reference oxygen fraction in $Z_{\mathrm{base}}$.
    pub zfrac_o: f64,
    /// Reference neon fraction in $Z_{\mathrm{base}}$.
    pub zfrac_ne: f64,
    /// $X$ tables (`[0.0, 0.03, 0.10, 0.35, 0.70]`).
    pub x_tables: Vec<KapCoXTable>,
}

/// Named collection of C/O-enhanced `KapCoZTable`s (`gn93_co`, `a09_co`).
pub struct KapCoSet {
    /// Prefix name (`"gn93_co"` or `"a09_co"`).
    pub prefix: String,
    /// Tables ordered by increasing $Z_{\mathrm{base}}$.
    pub z_tables: Vec<KapCoZTable>,
}

/// Single 2D slice of `AESOPUS_AGSS09`.
pub struct AesopusSlice {
    /// Raw `f32` values (`num_log_rs * num_log_ts`).
    pub raw_f32: Option<Vec<f32>>,
    /// Lazily fitted bicubic spline `(coeffs, ili_log_rs, ili_log_ts)`.
    pub(crate) spline: SplineCell,
}

impl AesopusSlice {
    /// Return the fitted bicubic spline for this AESOPUS slice.
    ///
    /// # Errors
    /// Returns [`KapError::MissingTable`] if this slice is not embedded, or
    /// [`KapError::NumericalFailure`] if bicubic spline fitting fails.
    pub fn spline_data(
        &self,
        log_rs: &[f64],
        log_ts: &[f64],
    ) -> Result<&(Vec<[f64; 4]>, bool, bool), KapError> {
        let raw = self
            .raw_f32
            .as_ref()
            .ok_or(KapError::MissingTable("AESOPUS slice not embedded"))?;
        self.spline
            .get_or_init(|| {
                let vals: Vec<f64> =
                    raw.iter().map(|&v| f64::from(v)).collect();
                fit_kap_spline(
                    log_rs,
                    log_ts,
                    &vals,
                    "bicubic spline fit failed on AESOPUS slice",
                    "AESOPUS",
                )
            })
            .as_ref()
            .map_err(Clone::clone)
    }
}

/// AESOPUS table set for one reference metallicity $Z_{\mathrm{ref}}$.
pub struct AesopusZSet {
    /// Tabulated $X$ grid.
    pub xs: Vec<f64>,
    /// Tabulated $f_{\mathrm{CO}}$ grid.
    pub fcos: Vec<f64>,
    /// Tabulated $f_{\mathrm{C}}$ grid.
    pub fcs: Vec<f64>,
    /// Tabulated $f_{\mathrm{N}}$ grid.
    pub fns: Vec<f64>,
    /// Flat slices indexed by `((ix * num_fcos + ico) * num_fcs + ic) * num_fns + in_`.
    pub slices: Vec<AesopusSlice>,
}

/// Complete `AESOPUS_AGSS09` data structure.
pub struct AesopusData {
    /// Solar metallicity $Z_\odot$.
    pub zsun: f64,
    /// Reference $\log_{10}(X_{\mathrm{C}} / X_{\mathrm{O}})_{\mathrm{ref}}$.
    pub fco_ref: f64,
    /// Reference $\log_{10}(X_{\mathrm{C}} / Z)_{\mathrm{ref}}$.
    pub fc_ref: f64,
    /// Reference $\log_{10}(X_{\mathrm{N}} / Z)_{\mathrm{ref}}$.
    pub fn_ref: f64,
    /// Tabulated $\log_{10} R$ grid (`19` points).
    pub log_rs: Vec<f64>,
    /// Tabulated $\log_{10}(T/\mathrm{K})$ grid (`42` points).
    pub log_ts: Vec<f64>,
    /// Reference metallicities $Z_{\mathrm{ref}}$ (`13` points).
    pub zs: Vec<f64>,
    /// Precomputed $\log_{10} Z_{\mathrm{ref}}$.
    pub log_zs: Vec<f64>,
    /// Per-$Z$ table sets.
    pub ts: Vec<AesopusZSet>,
}

/// Single `kapCN` $Z$ set (`63` sub-tables).
pub struct KapCnZSet {
    /// Base metallicity $Z$.
    pub z: f64,
    /// Carbon enhancement factors $f_{\mathrm{C}}$ (`7` points).
    pub fcs: [f64; 7],
    /// Nitrogen enhancement factors $f_{\mathrm{N}}$ (`3` points).
    pub fns: [f64; 3],
    /// Raw $1000 \times \log_{10} \kappa$ grids for each of the `63` sub-tables.
    pub raw_i16: Vec<Vec<i16>>,
    /// Lazily fitted bicubic splines for each of the `63` sub-tables.
    pub(crate) splines: Vec<SplineCell>,
}

impl KapCnZSet {
    /// Return the fitted bicubic spline for 0-based `tbl_idx` (`0..63`).
    ///
    /// # Errors
    /// Returns [`KapError::MissingTable`] if `tbl_idx` is out of range, or
    /// [`KapError::NumericalFailure`] if bicubic spline fitting fails.
    pub fn spline_data(
        &self,
        tbl_idx: usize,
        log_rs: &[f64],
        log_ts: &[f64],
    ) -> Result<&(Vec<[f64; 4]>, bool, bool), KapError> {
        let cell = self.splines.get(tbl_idx).ok_or(KapError::MissingTable(
            "kapCN subtable index out of range",
        ))?;
        let raw = self
            .raw_i16
            .get(tbl_idx)
            .ok_or(KapError::MissingTable("kapCN subtable data missing"))?;
        cell.get_or_init(|| {
            let vals: Vec<f64> =
                raw.iter().map(|&v| f64::from(v) / 1000.0).collect();
            let label = format!("kapCN(Z={}, subtable={tbl_idx})", self.z);
            fit_kap_spline(
                log_rs,
                log_ts,
                &vals,
                "bicubic spline fit failed on kapCN table",
                &label,
            )
        })
        .as_ref()
        .map_err(Clone::clone)
    }
}

/// Complete `kapCN` Lederer & Aringer (2009) table collection.
pub struct KapCnData {
    /// Tabulated $\log_{10} R$ grid (`17` points).
    pub log_rs: Vec<f64>,
    /// Tabulated $\log_{10}(T/\mathrm{K})$ grid (`18` points).
    pub log_ts: Vec<f64>,
    /// Tabulated $X$ grid (`[0.5, 0.7, 0.8]`).
    pub xs: [f64; 3],
    /// Tabulated $Z$ grid (`14` points).
    pub zs: Vec<f64>,
    /// Per-$Z$ table sets (`14` sets).
    pub z_sets: Vec<KapCnZSet>,
}

/// Master opacity table collection loaded from `kap_tables.bin`.
pub struct KapTables {
    /// Fixed-metal and low-$T$ table sets (`gs98`, `lowT_fa05_gs98`, `a09`).
    pub fixed_sets: Vec<KapFixedSet>,
    /// C/O-enhanced Type 2 table sets (`gn93_co`, `a09_co`).
    pub co_sets: Vec<KapCoSet>,
    /// `AESOPUS_AGSS09` low-$T$ table set.
    pub aesopus: AesopusData,
    /// `kapCN` low-$T$ table set.
    pub kap_cn: KapCnData,
}

impl KapTables {
    /// Compute `(logT_Compton_blend_hi, logR_Compton_blend_lo)` from the loaded
    /// Type 1 and Type 2 table headers (`setup_kap_tables` in `kap/private/load_kap.f90:65-70`).
    ///
    /// # Errors
    /// Returns [`KapError::MissingTable`] if the selected Type 1 or Type 2 table set is missing.
    pub fn compton_blend_bounds(
        &self,
        cfg: &KapConfig,
    ) -> Result<(f64, f64), KapError> {
        let fixed_prefix = cfg.kap_option.prefix();
        let t1_x0 = self
            .fixed_sets
            .iter()
            .find(|s| s.prefix == fixed_prefix)
            .and_then(|s| s.z_tables.first())
            .and_then(|z| z.x_tables.first())
            .ok_or(KapError::MissingTable(
                "missing Type 1 x_table for Compton blend bounds",
            ))?;
        let co_prefix = cfg.kap_co_option.prefix();
        let co_x0 = self
            .co_sets
            .iter()
            .find(|s| s.prefix == co_prefix)
            .and_then(|s| s.z_tables.first())
            .and_then(|z| z.x_tables.first())
            .ok_or(KapError::MissingTable(
                "missing Type 2 CO x_table for Compton blend bounds",
            ))?;
        let log_t_compton_blend_hi =
            (t1_x0.log_t_max - 0.01).min(co_x0.log_t_max - 0.01);
        let log_r_compton_blend_lo = t1_x0.log_r_min + 0.01;
        Ok((log_t_compton_blend_hi, log_r_compton_blend_lo))
    }

    /// Compute `(highT_logT_min, lowT_logT_max)` from the loaded table headers
    /// (`Get_kap_Results_blend_T` in `kap/private/kap_eval.f90:287-318`).
    ///
    /// # Errors
    /// Returns [`KapError::MissingTable`] if the selected table set is missing.
    pub fn temp_blend_bounds(
        &self,
        cfg: &KapConfig,
    ) -> Result<(f64, f64), KapError> {
        let high_t_min = if cfg.use_type2_opacities {
            let co_prefix = cfg.kap_co_option.prefix();
            self.co_sets
                .iter()
                .find(|s| s.prefix == co_prefix)
                .and_then(|s| s.z_tables.first())
                .and_then(|z| z.x_tables.first())
                .ok_or(KapError::MissingTable(
                    "missing Type 2 CO x_table for blend lower bound",
                ))?
                .log_t_min
        } else {
            let fixed_prefix = cfg.kap_option.prefix();
            self.fixed_sets
                .iter()
                .find(|s| s.prefix == fixed_prefix)
                .and_then(|s| s.z_tables.first())
                .and_then(|z| z.x_tables.first())
                .ok_or(KapError::MissingTable(
                    "missing Type 1 x_table for blend lower bound",
                ))?
                .log_t_min
        };

        let low_t_max = match cfg.kap_low_t_option {
            KapLowTOption::KapCn => *self
                .kap_cn
                .log_ts
                .last()
                .ok_or(KapError::MissingTable("empty kapCN logT grid"))?,
            KapLowTOption::Aesopus => {
                let ka =
                    crate::aesopus_h5::resolve_aesopus_data(&cfg.aesopus_filename)?;
                *ka.log_ts
                    .last()
                    .ok_or(KapError::MissingTable("empty AESOPUS logT grid"))?
            }
            _ => {
                let lt_prefix = cfg.kap_low_t_option.prefix();
                self.fixed_sets
                    .iter()
                    .find(|s| s.prefix == lt_prefix)
                    .and_then(|s| s.z_tables.first())
                    .and_then(|z| z.x_tables.first())
                    .ok_or(KapError::MissingTable(
                        "missing lowT x_table for blend upper bound",
                    ))?
                    .log_t_max
            }
        };
        Ok((high_t_min, low_t_max))
    }
}

static KAP_TABLES: OnceLock<Result<KapTables, KapError>> = OnceLock::new();

/// Return a reference to the lazily parsed master [`KapTables`].
///
/// # Errors
/// Returns [`KapError::TableIo`] if the opacity table bundle is missing,
/// unreadable, or malformed.
pub fn kap_tables() -> Result<&'static KapTables, KapError> {
    KAP_TABLES
        .get_or_init(load_or_build_kap_tables)
        .as_ref()
        .map_err(Clone::clone)
}
