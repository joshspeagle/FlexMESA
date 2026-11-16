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
/// reproducing `ionization_table(1:28)` in `eos/private/ion_offset.f90:49-52`
/// at tag `26.4.1`, where unsuffixed Fortran literals are evaluated in default
/// single precision (`real(4)`) before widening to `real(dp)` (deferred bug `#192`).
#[allow(clippy::excessive_precision, clippy::cast_lossless)]
pub const NIST_IONIZATION_TABLE_EV: [f64; 28] = [
    13.598_434_599_702_f32 as f64,
    79.005_154_5_f32 as f64,
    203.486_171_1_f32 as f64,
    399.148_64_f32 as f64,
    670.980_9_f32 as f64,
    1_030.108_5_f32 as f64,
    1_486.058_f32 as f64,
    2_043.842_8_f32 as f64,
    2_715.89_f32 as f64,
    3_511.696_f32 as f64,
    4_420.0_f32 as f64,
    5_451.06_f32 as f64,
    6_604.95_f32 as f64,
    7_888.53_f32 as f64,
    9_305.8_f32 as f64,
    10_859.7_f32 as f64,
    12_556.4_f32 as f64,
    14_400.8_f32 as f64,
    16_382.0_f32 as f64,
    18_510.0_f32 as f64,
    20_788.0_f32 as f64,
    23_221.0_f32 as f64,
    25_820.0_f32 as f64,
    28_582.0_f32 as f64,
    31_514.0_f32 as f64,
    34_619.0_f32 as f64,
    37_899.0_f32 as f64,
    41_356.0_f32 as f64,
];

/// Compute the specific ionization energy offset $\Delta E_{\mathrm{ion}}$ in
/// $\mathrm{erg\,g^{-1}}$ directly from species mass fractions `xa`, atomic
/// numbers `z`, and baryon numbers `a`.
///
/// Reproduces `compute_ion_offset` in `eos/private/ion_offset.f90:58-81` at tag
/// `26.4.1`, including the upstream normalization of `ya(k) = xa(k) / A(k)` by
/// `norm = sum_j(xa(j) / A(j))` (logged as deferred upstream bug `#192`).
#[must_use]
pub fn compute_ion_offset_from_za(xa: &[f64], z: &[usize], a: &[f64]) -> f64 {
    let n = xa.len().min(z.len()).min(a.len());
    let mut norm = 0.0_f64;
    for k in 0..n {
        if a[k] > 0.0 {
            norm += xa[k] / a[k];
        }
    }
    if norm == 0.0 {
        return 0.0;
    }
    let mut offset_ev = 0.0_f64;
    for k in 0..n {
        let zk = z[k];
        if (1..=28).contains(&zk) && a[k] > 0.0 {
            // Reproduces eos/private/ion_offset.f90:67-77 (see #192).
            let ya_k = (xa[k] / a[k]) / norm;
            offset_ev += NIST_IONIZATION_TABLE_EV[zk - 1] * ya_k;
        }
    }
    offset_ev * EV2ERG / AMU
}

/// Compute the specific ionization energy offset $\Delta E_{\mathrm{ion}}$ in
/// $\mathrm{erg\,g^{-1}}$ from species mass fractions `xa` and 1-based MESA
/// `chem_id` identifiers using the built-in [`NuclideCatalog`].
///
/// Reproduces `compute_ion_offset` in `eos/private/ion_offset.f90:58-81` at tag
/// `26.4.1` (deferred upstream bug `#192`).
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
    let catalog = NuclideCatalog::builtin_ref();
    let mut norm = 0.0_f64;
    for (&xk, &cid) in xa.iter().zip(chem_id.iter()) {
        let entry =
            catalog
                .get_by_id(cid)
                .ok_or_else(|| EosError::InvalidState {
                    context: "compute_ion_offset",
                    detail: format!("unknown chem_id {cid}"),
                })?;
        norm += xk / (entry.z_plus_n as f64);
    }
    if norm == 0.0 {
        return Ok(0.0);
    }
    let mut offset_ev = 0.0_f64;
    for (&xk, &cid) in xa.iter().zip(chem_id.iter()) {
        if let Some(entry) = catalog.get_by_id(cid) {
            if (1..=28).contains(&entry.z) {
                let ya_k = xk / (entry.z_plus_n as f64);
                // Reproduces eos/private/ion_offset.f90:67-77 (see #192).
                offset_ev += NIST_IONIZATION_TABLE_EV[entry.z - 1] * (ya_k / norm);
            }
        }
    }
    Ok(offset_ev * EV2ERG / AMU)
}
