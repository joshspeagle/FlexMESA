// SPDX-License-Identifier: LGPL-3.0-or-later
//! Neutral ideal-gas plus Planckian radiation equation of state
//! (`eos/private/ideal.f90`).
//!
//! Evaluates `compute_F_ideal_ion` and `compute_F_rad` in
//! [`AutoDiffReal2VarOrder3`], reduces the free energy via
//! [`crate::free_energy::pack_for_export`], and sets `res[I_MU] = abar` and
//! `res[I_FRAC_IDEAL] = 1.0` matching `get_ideal_for_eosdt` in `ideal.f90`.

use flexmesa_autodiff::AutoDiffReal2VarOrder3;
use flexmesa_chem::NuclideCatalog;

use crate::free_energy::{compute_f_ideal_ion, pack_for_export};
use crate::types::{EosError, EosEvalOutput, I_FRAC_IDEAL, I_MU};

/// Minimum species mass fraction included in `ideal_eos` (`1e-10`).
pub const IDEAL_MASS_FRACTION_LIMIT: f64 = 1.0e-10;

/// Evaluate the neutral ideal-gas + radiation EOS directly from species mass
/// fractions `xa`, atomic weights `weights` ($\mathrm{amu}$), and baryon
/// numbers `a` (`ideal_eos` in `eos/private/ideal.f90`).
///
/// # Errors
/// Returns [`EosError::InvalidState`] if no species exceed
/// [`IDEAL_MASS_FRACTION_LIMIT`] or if $T \le 0$, $\rho \le 0$, or $\bar{A}
/// \le 0$.
pub fn eval_ideal_eos_from_weights(
    temp_in: f64,
    den_in: f64,
    abar: f64,
    zbar: f64,
    xa: &[f64],
    weights: &[f64],
    a: &[f64],
) -> Result<EosEvalOutput, EosError> {
    if temp_in <= 0.0 || den_in <= 0.0 || abar <= 0.0 {
        return Err(EosError::InvalidState {
            context: "eval_ideal_eos_from_weights",
            detail: format!(
                "invalid state (T={temp_in:e}, rho={den_in:e}, abar={abar:e})"
            ),
        });
    }
    let species = xa.len().min(weights.len()).min(a.len());
    let mut acmi = Vec::with_capacity(species);
    let mut a_sel = Vec::with_capacity(species);
    let mut select_xa = Vec::with_capacity(species);
    let mut norm = 0.0_f64;

    for j in 0..species {
        if xa[j] > IDEAL_MASS_FRACTION_LIMIT {
            acmi.push(weights[j]);
            a_sel.push(a[j]);
            select_xa.push(xa[j]);
            norm += xa[j];
        }
    }
    if select_xa.is_empty() || norm <= 0.0 {
        return Err(EosError::InvalidState {
            context: "eval_ideal_eos_from_weights",
            detail: "no species with xa > 1e-10".to_string(),
        });
    }

    for x in &mut select_xa {
        *x /= norm;
    }

    let mut ya = Vec::with_capacity(select_xa.len());
    let mut ya_norm = 0.0_f64;
    for (&x, &aj) in select_xa.iter().zip(a_sel.iter()) {
        let yj = x / aj;
        ya.push(yj);
        ya_norm += yj;
    }
    for y in &mut ya {
        *y /= ya_norm;
    }

    let temp = AutoDiffReal2VarOrder3::var1(temp_in);
    let den = AutoDiffReal2VarOrder3::var2(den_in);

    let f_ideal_ion = compute_f_ideal_ion(temp, den, abar, &acmi, &ya);
    let zero = AutoDiffReal2VarOrder3::constant(0.0);
    let etaele = AutoDiffReal2VarOrder3::constant(-1.0e99);
    let xnefer = AutoDiffReal2VarOrder3::constant(1.0e-20);

    let mut out = pack_for_export(
        f_ideal_ion,
        zero,
        zero,
        zero,
        temp,
        den,
        xnefer,
        etaele,
        abar,
        zbar,
        zero,
        zero,
        zero,
        xa.len(),
    )?;
    out.res[I_MU] = abar;
    out.set_single_component_fraction(I_FRAC_IDEAL);
    Ok(out)
}

/// Evaluate the neutral ideal-gas + radiation EOS from species mass fractions
/// `xa` and 1-based MESA `chem_id` identifiers (`get_ideal_for_eosdt` in
/// `eos/private/ideal.f90`).
///
/// # Errors
/// Returns [`EosError::InvalidState`] if `xa` and `chem_id` have mismatched
/// lengths, any `chem_id` is unknown, or thermodynamic inputs are non-positive.
pub fn eval_ideal_eos(
    temp_in: f64,
    den_in: f64,
    abar: f64,
    zbar: f64,
    xa: &[f64],
    chem_id: &[usize],
) -> Result<EosEvalOutput, EosError> {
    if xa.len() != chem_id.len() {
        return Err(EosError::InvalidState {
            context: "eval_ideal_eos",
            detail: format!(
                "xa length ({}) != chem_id length ({})",
                xa.len(),
                chem_id.len()
            ),
        });
    }
    let catalog = NuclideCatalog::builtin();
    let mut weights = Vec::with_capacity(xa.len());
    let mut a = Vec::with_capacity(xa.len());
    for &cid in chem_id {
        let entry =
            catalog
                .get_by_id(cid)
                .ok_or_else(|| EosError::InvalidState {
                    context: "eval_ideal_eos",
                    detail: format!("unknown chem_id {cid}"),
                })?;
        weights.push(entry.w);
        a.push(entry.z_plus_n as f64);
    }
    eval_ideal_eos_from_weights(temp_in, den_in, abar, zbar, xa, &weights, &a)
}
