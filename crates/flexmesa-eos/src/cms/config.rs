// SPDX-License-Identifier: LGPL-3.0-or-later
//! Configuration and blend-fraction (`get_cms_alfa`) for the Chabrier-Mazevet-Soubiran
//! (`CMS`) equation of state (`eos/private/eoscms_eval.f90`).

use flexmesa_autodiff::AutoDiffReal2VarOrder1;

/// Number of hydrogen mass fraction tables in `CMS` (`num_eosCMS_Xs = 11`).
pub const CMS_NUM_XS: usize = 11;

/// Hydrogen mass fractions for the 11 `CMS` tables (`0.00, 0.10, ..., 1.00`).
pub const CMS_XVALS: [f64; CMS_NUM_XS] =
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

/// Three-digit filename tags for the 11 `CMS` tables (`000` through `100`).
pub const CMS_XSTR: [&str; CMS_NUM_XS] = [
    "000", "010", "020", "030", "040", "050", "060", "070", "080", "090", "100",
];

/// Configuration for the `CMS` equation of state (`eos.defaults`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CmsConfig {
    /// Use a fixed composition index (`0..=10`) instead of interpolating in $X$ (default `false`).
    pub use_fixed_composition: bool,
    /// Fixed composition index (`0..=10`) when `use_fixed_composition` is `true` (default `-1`).
    pub fixed_composition_index: isize,
    /// Use 4-point piecewise monotonic cubic interpolation in $X$ (default `false`).
    pub cubic_in_x: bool,
    /// Include blackbody radiation pressure/energy/entropy in `CMS` (default `false`).
    pub include_radiation: bool,
    /// Metallicity $Z$ above which `CMS` has zero weight (default `-1.0`, disabled).
    pub max_z_for_any_cms: f64,
    /// Metallicity $Z$ below which `CMS` has full weight (default `-1.0`, disabled).
    pub max_z_for_all_cms: f64,
    /// Lower $\log_{10} Q$ cutoff for any `CMS` weight (default `-100.0`).
    pub logq_min_for_any_cms: f64,
    /// Lower $\log_{10} Q$ threshold for full `CMS` weight (default `-100.0`).
    pub logq_min_for_all_cms: f64,
    /// Upper $\log_{10} Q$ threshold for full `CMS` weight (default `100.0`).
    pub logq_max_for_all_cms: f64,
    /// Upper $\log_{10} Q$ cutoff for any `CMS` weight (default `100.0`).
    pub logq_max_for_any_cms: f64,
    /// Lower $\log_{10} \rho$ cutoff for any `CMS` weight (default `-8.0`).
    pub logrho_min_for_any_cms: f64,
    /// Lower $\log_{10} \rho$ threshold for full `CMS` weight (default `-7.9`).
    pub logrho_min_for_all_cms: f64,
    /// Upper $\log_{10} \rho$ threshold for full `CMS` weight (default `5.9`).
    pub logrho_max_for_all_cms: f64,
    /// Upper $\log_{10} \rho$ cutoff for any `CMS` weight (default `6.0`).
    pub logrho_max_for_any_cms: f64,
    /// Lower $\log_{10} T$ cutoff for any `CMS` weight (default `2.0`).
    pub logt_min_for_any_cms: f64,
    /// Lower $\log_{10} T$ threshold for full `CMS` weight (default `2.1`).
    pub logt_min_for_all_cms: f64,
    /// Upper $\log_{10} T$ threshold for full `CMS` weight (default `7.9`).
    pub logt_max_for_all_cms: f64,
    /// Upper $\log_{10} T$ cutoff for any `CMS` weight (default `8.0`).
    pub logt_max_for_any_cms: f64,
    /// Upper $\log_{10} T$ threshold for full `CMS` in the lower-right corner (default `5.5`).
    pub logt_low_max_for_all_cms: f64,
    /// Upper $\log_{10} T$ cutoff for any `CMS` in the lower-right corner (default `5.6`).
    pub logt_low_max_for_any_cms: f64,
    /// Lower $\log_{10} \rho$ threshold for the high-$T$ region (default `2.2`).
    pub logrho_high_min_for_all_cms: f64,
    /// Lower $\log_{10} \rho$ cutoff for the high-$T$ region (default `2.1`).
    pub logrho_high_min_for_any_cms: f64,
}

impl Default for CmsConfig {
    fn default() -> Self {
        Self {
            use_fixed_composition: false,
            fixed_composition_index: -1,
            cubic_in_x: false,
            include_radiation: false,
            max_z_for_any_cms: -1.0,
            max_z_for_all_cms: -1.0,
            logq_min_for_any_cms: -100.0,
            logq_min_for_all_cms: -100.0,
            logq_max_for_all_cms: 100.0,
            logq_max_for_any_cms: 100.0,
            logrho_min_for_any_cms: -8.0,
            logrho_min_for_all_cms: -7.9,
            logrho_max_for_all_cms: 5.9,
            logrho_max_for_any_cms: 6.0,
            logt_min_for_any_cms: 2.0,
            logt_min_for_all_cms: 2.1,
            logt_max_for_all_cms: 7.9,
            logt_max_for_any_cms: 8.0,
            logt_low_max_for_all_cms: 5.5,
            logt_low_max_for_any_cms: 5.6,
            logrho_high_min_for_all_cms: 2.2,
            logrho_high_min_for_any_cms: 2.1,
        }
    }
}

fn cms_trapezoid(
    x: AutoDiffReal2VarOrder1,
    min_any: f64,
    min_all: f64,
    max_all: f64,
    max_any: f64,
) -> AutoDiffReal2VarOrder1 {
    if x < min_any {
        AutoDiffReal2VarOrder1::zero()
    } else if x <= min_all {
        (x - min_any) / (min_all - min_any)
    } else if x <= max_all {
        AutoDiffReal2VarOrder1::constant(1.0)
    } else if x < max_any {
        (x - max_any) / (max_all - max_any)
    } else {
        AutoDiffReal2VarOrder1::zero()
    }
}

/// Computes the raw `CMS` blend fraction `(alfa, d_alfa_dlogT, d_alfa_dlogRho)`.
///
/// Mirrors `Get_CMS_alfa` in `eos/private/eoscms_eval.f90` (lines 38..149).
#[must_use]
pub fn get_cms_alfa(
    cfg: &CmsConfig,
    log_rho: f64,
    log_t: f64,
    z: f64,
) -> (f64, f64, f64) {
    let log_rho_ad = AutoDiffReal2VarOrder1::var1(log_rho);
    let log_t_ad = AutoDiffReal2VarOrder1::var2(log_t);
    let log_q_ad = log_rho_ad - 2.0 * log_t_ad + 12.0;

    let blend_log_t = cms_trapezoid(
        log_t_ad,
        cfg.logt_min_for_any_cms,
        cfg.logt_min_for_all_cms,
        cfg.logt_max_for_all_cms,
        cfg.logt_max_for_any_cms,
    );
    let blend_log_rho = cms_trapezoid(
        log_rho_ad,
        cfg.logrho_min_for_any_cms,
        cfg.logrho_min_for_all_cms,
        cfg.logrho_max_for_all_cms,
        cfg.logrho_max_for_any_cms,
    );
    let blend_log_q = cms_trapezoid(
        log_q_ad,
        cfg.logq_min_for_any_cms,
        cfg.logq_min_for_all_cms,
        cfg.logq_max_for_all_cms,
        cfg.logq_max_for_any_cms,
    );

    let blend_z = if cfg.max_z_for_any_cms <= 0.0
        || cfg.max_z_for_all_cms <= 0.0
        || z <= cfg.max_z_for_all_cms
    {
        AutoDiffReal2VarOrder1::constant(1.0)
    } else if z >= cfg.max_z_for_any_cms {
        AutoDiffReal2VarOrder1::zero()
    } else {
        AutoDiffReal2VarOrder1::constant(
            (z - cfg.max_z_for_any_cms)
                / (cfg.max_z_for_all_cms - cfg.max_z_for_any_cms),
        )
    };

    let blend = blend_z * blend_log_t * blend_log_rho * blend_log_q;
    (1.0 - blend.val, -blend.d1[1], -blend.d1[0])
}
