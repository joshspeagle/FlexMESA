//! Whole-number ionic charges and continuum depression (`mod_typical_charge`
//! and `mod_ionization.f90`).
//!
//! Implements the Paquette et al. (1986, ApJS 61, 197, Eq. 20) and Dupuis et
//! al. (1992, ApJS 82, 505, Eq. 3) continuum-depression effective ionization
//! threshold:
//! - [`chi_info`]: Base threshold $\chi$ and continuum-depression coefficients
//!   $(c_0, c_1, c_2)$ in $\mathrm{eV}$, monomorphized over [`IonScalar`]
//!   (`f64` and [`flexmesa_autodiff::DualOrder1<N>`]).
//! - [`chi_effective`]: Effective ionization threshold $\chi_{\mathrm{eff}}$
//!   in $\mathrm{eV}$, monomorphized over [`IonScalar`].
//! - [`get_typical_charge`] / [`eval_typical_charge`]: Whole-number typical
//!   ionic charge against the Allen (1973) ionization potentials ($Z \le 30$).

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{ONE_THIRD, TWO_THIRDS};

use crate::potentials_data::{IONIZATION_POTENTIALS, MAX_TYPICAL_CHARGE_Z};
use crate::{IonError, IonScalar};

/// Look up the Allen (1973) ionization potential $I_{Z, k}$ in $\mathrm{eV}$
/// for atomic number `z` ($1 \le Z \le 30$) and 1-based stage `stage`
/// ($1 \le k \le Z$).
#[must_use]
pub fn ionization_potential(z: usize, stage: usize) -> Option<f64> {
    if (1..=MAX_TYPICAL_CHARGE_Z).contains(&z) && (1..=z).contains(&stage) {
        Some(IONIZATION_POTENTIALS[z - 1][stage - 1])
    } else {
        None
    }
}

/// Compute the Paquette et al. (1986, Eq. 20) and Dupuis et al. (1992, Eq. 3)
/// ionization threshold parameters $(\chi, c_0, c_1, c_2)$ in $\mathrm{eV}$.
///
/// Matches `chi_info` in `mod_ionization.f90` and `mod_typical_charge.f90`:
/// $$\chi = 1.987 \times 10^{-4}\,T\,\left(-8.392 - \log_{10}\rho +
///   1.5\log_{10} T - \log_{10}(z_1 / a_1)\right),$$
/// $$c_0 = 1.085 \times 10^{-4}\,\frac{\rho\,T}{a_1}, \quad
///   c_1 = 1.617 \times 10^{4}\sqrt{\frac{\rho\,(z_1^2 + z_1)}{T\,a_1}}, \quad
///   c_2 = 29.38\,z_1\,\left(\frac{\rho}{a_1}\right)^{1/3}.$$
#[must_use]
pub fn chi_info<T: IonScalar>(
    a1: T,
    z1: T,
    t: T,
    log10_t: T,
    rho: T,
    log10_rho: T,
) -> (T, T, T, T) {
    let c_chi = T::from_f64(1.987e-4);
    let offset = T::from_f64(-8.392);
    let one_half = T::from_f64(1.5);
    let log_z_over_a = (z1 / a1).log10();
    let chi =
        c_chi * t * (offset - log10_rho + one_half * log10_t - log_z_over_a);

    let c0 = T::from_f64(1.085e-4) * rho * t / a1;
    let c1 = T::from_f64(1.617e4) * (rho * (z1 * z1 + z1) / (t * a1)).sqrt();
    let c2 = T::from_f64(29.38) * z1 * (rho / a1).powf(ONE_THIRD);
    (chi, c0, c1, c2)
}

/// Compute the effective ionization threshold $\chi_{\mathrm{eff}}$ in
/// $\mathrm{eV}$ (`chi_effective` in `mod_ionization.f90`):
/// $$\chi_{\mathrm{eff}} = \chi + \frac{c_0}{z_2^3} +
///   \min\!\left(c_1 z_2,\; c_2\left[\left(\frac{z_2}{z_1}\right)^{2/3} +
///   0.6\right]\right).$$
#[must_use]
pub fn chi_effective<T: IonScalar>(
    chi: T,
    c0: T,
    c1: T,
    c2: T,
    z1: T,
    z2: T,
) -> T {
    let term1 = c0 / (z2 * z2 * z2);
    let branch_a = c1 * z2;
    let branch_b = c2 * ((z2 / z1).powf(TWO_THIRDS) + T::from_f64(0.6));
    let dep = if branch_a.val() < branch_b.val() {
        branch_a
    } else {
        branch_b
    };
    chi + term1 + dep
}

/// Compute the whole-number typical ionic charge $Z_{\mathrm{typ}} \in
/// [1, Z_{\max}]$ for an ion of atomic number `izmax` in a background plasma of
/// mean nucleon number `a1` and mean background charge `z1 = a1 * free_e`.
///
/// Matches `get_typical_charge` in `star/private/mod_typical_charge.f90`:
/// - Never returns `0.0`; returns `izmax as f64` directly when `izmax > 30` or
///   `izmax <= 1`.
/// - Loops `i = 1 .. izmax - 1`, evaluating `chi_effective(..., z2 = i + 1)`
///   and comparing against `ip(izmax, i + 1)`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn get_typical_charge(
    izmax: usize,
    a1: f64,
    z1: f64,
    t: f64,
    log10_t: f64,
    rho: f64,
    log10_rho: f64,
) -> f64 {
    if izmax == 0 {
        return 0.0;
    }
    let z_full = izmax as f64;
    if izmax > MAX_TYPICAL_CHARGE_Z || izmax == 1 {
        return z_full;
    }
    let (chi, c0, c1, c2) = chi_info(a1, z1, t, log10_t, rho, log10_rho);
    let row = &IONIZATION_POTENTIALS[izmax - 1];
    for (i, &ip_val) in row.iter().enumerate().take(izmax).skip(1) {
        let z2 = i as f64;
        let chi_eff = chi_effective(chi, c0, c1, c2, z1, z2 + 1.0);
        if chi_eff < ip_val {
            return z2;
        }
    }
    z_full
}

/// Evaluate `eval_typical_charge` for a 1-based `chem_id` (`cid`), mean
/// atomic weight `abar`, and free electrons per nucleon `free_e`
/// (`z1 = abar * free_e`).
///
/// # Errors
/// Returns [`IonError::InvalidInput`] if `cid` is not in `catalog`.
#[allow(clippy::too_many_arguments)]
pub fn eval_typical_charge(
    catalog: &NuclideCatalog,
    cid: usize,
    abar: f64,
    free_e: f64,
    t: f64,
    log10_t: f64,
    rho: f64,
    log10_rho: f64,
) -> Result<f64, IonError> {
    let iso = catalog
        .get_by_id(cid)
        .ok_or_else(|| IonError::InvalidInput {
            context: "eval_typical_charge",
            detail: format!("unknown chem_id {cid}"),
        })?;
    Ok(get_typical_charge(
        iso.z,
        abar,
        abar * free_e,
        t,
        log10_t,
        rho,
        log10_rho,
    ))
}
