// SPDX-License-Identifier: LGPL-3.0-or-later
//! Neutral ground-state ionization energy offset
//! (`eos/private/ion_offset.f90`).
//!
//! Computes the specific energy $\Delta E_{\mathrm{ion}}$ (in
//! $\mathrm{erg\,g^{-1}}$) required to strip all bound electrons from a neutral
//! mixture using the NIST Atomic Spectra Database total binding energies for
//! $1 \le Z \le 28$ ($\mathrm{H}$ through $\mathrm{Ni}$).

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{AMU, EV2ERG};

use crate::types::EosError;

/// Total atomic binding energies (in $\mathrm{eV}$) for elements $Z = 1..=28$
/// ($\mathrm{H}$ through $\mathrm{Ni}$) from the NIST Atomic Spectra Database,
/// matching `ionization_table(1:28)` in `eos/private/ion_offset.f90`.
pub const NIST_IONIZATION_TABLE_EV: [f64; 28] = [
    13.598_434_599_702,
    79.005_154_5,
    203.486_171_1,
    399.148_64,
    670.980_9,
    1_030.108_5,
    1_486.058,
    2_043.842_8,
    2_715.89,
    3_511.696,
    4_420.0,
    5_451.06,
    6_604.95,
    7_888.53,
    9_305.8,
    10_859.7,
    12_556.4,
    14_400.8,
    16_382.0,
    18_510.0,
    20_788.0,
    23_221.0,
    25_820.0,
    28_582.0,
    31_514.0,
    34_619.0,
    37_899.0,
    41_356.0,
];

/// Compute the specific ionization energy offset $\Delta E_{\mathrm{ion}}$ in
/// $\mathrm{erg\,g^{-1}}$ directly from species mass fractions `xa`, atomic
/// numbers `z`, and baryon numbers `a`.
///
/// Matches `compute_ion_offset` in `eos/private/ion_offset.f90`:
/// $$\Delta E_{\mathrm{ion}} = \frac{\mathrm{eV2erg}}{\mathrm{amu}}
/// \sum_{1 \le Z_k \le 28} E_{\mathrm{bind}}(Z_k)\,\frac{X_k}{A_k}.$$
#[must_use]
pub fn compute_ion_offset_from_za(xa: &[f64], z: &[usize], a: &[f64]) -> f64 {
    let n = xa.len().min(z.len()).min(a.len());
    let mut offset_ev_per_amu = 0.0_f64;
    for k in 0..n {
        let zk = z[k];
        if (1..=28).contains(&zk) && a[k] > 0.0 {
            offset_ev_per_amu +=
                NIST_IONIZATION_TABLE_EV[zk - 1] * xa[k] / a[k];
        }
    }
    offset_ev_per_amu * EV2ERG / AMU
}

/// Compute the specific ionization energy offset $\Delta E_{\mathrm{ion}}$ in
/// $\mathrm{erg\,g^{-1}}$ from species mass fractions `xa` and 1-based MESA
/// `chem_id` identifiers using the built-in [`NuclideCatalog`].
///
/// # Errors
/// Returns [`EosError::InvalidState`] if `xa` and `chem_id` have different
/// lengths or if any `chem_id` is not found in the nuclide catalog.
pub fn compute_ion_offset(
    xa: &[f64],
    chem_id: &[usize],
) -> Result<f64, EosError> {
    if xa.len() != chem_id.len() {
        return Err(EosError::InvalidState {
            context: "compute_ion_offset",
            detail: format!(
                "xa length ({}) != chem_id length ({})",
                xa.len(),
                chem_id.len()
            ),
        });
    }
    let catalog = NuclideCatalog::builtin();
    let mut offset_ev_per_amu = 0.0_f64;
    for (&xk, &cid) in xa.iter().zip(chem_id.iter()) {
        let entry =
            catalog
                .get_by_id(cid)
                .ok_or_else(|| EosError::InvalidState {
                    context: "compute_ion_offset",
                    detail: format!("unknown chem_id {cid}"),
                })?;
        if (1..=28).contains(&entry.z) {
            offset_ev_per_amu += NIST_IONIZATION_TABLE_EV[entry.z - 1] * xk
                / (entry.z_plus_n as f64);
        }
    }
    Ok(offset_ev_per_amu * EV2ERG / AMU)
}
