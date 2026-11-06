// SPDX-License-Identifier: LGPL-3.0-or-later
//! Third-order Helmholtz free-energy thermodynamic reduction shared by `ideal`
//! and `Skye` (`eos/private/skye_ideal.f90` and
//! `eos/private/skye_thermodynamics.f90`).
//!
//! Evaluates radiation and ideal-ion specific Helmholtz free energies in
//! [`AutoDiffReal2VarOrder3`] (with variable 1 $= T$ and variable 2 $= \rho$)
//! and reduces any total free energy $F_{\mathrm{gas}}(T, \rho)$ into the full
//! 26-channel MESA output state with guaranteed Maxwell consistency.

use flexmesa_autodiff::AutoDiffReal2VarOrder3;
use flexmesa_const::{AMU, AVO, CLIGHT, CRAD, KERG, PI, PLANCK_H};
use flexmesa_math::pow;

use crate::types::{
    EosError, EosEvalOutput, I_CHI_RHO, I_CHI_T, I_CP, I_CV, I_DE_DRHO,
    I_DS_DRHO, I_DS_DT, I_ETA, I_GAMMA1, I_GAMMA3, I_GRAD_AD, I_LATENT_DDLNRHO,
    I_LATENT_DDLNT, I_LN_E, I_LN_FREE_E, I_LN_PGAS, I_LN_S, I_MU, I_PHASE,
};

/// Quantum concentration constant
/// $\mathrm{sifac} = h^{3} / (2 \pi m_{u})^{3/2}$ matching `skye_ideal.f90`.
#[must_use]
#[inline]
pub fn sifac_constant() -> f64 {
    let two_pi_amu = 2.0 * PI * AMU;
    PLANCK_H * PLANCK_H * PLANCK_H / (two_pi_amu * two_pi_amu.sqrt())
}

/// Compute the specific radiation Helmholtz free energy
/// $F_{\mathrm{rad}}(T, \rho) = -\frac{1}{3} a_{\mathrm{rad}} T^{4} / \rho$ in
/// $\mathrm{erg\,g^{-1}}$ (`compute_F_rad` in `skye_ideal.f90`).
#[must_use]
#[inline]
pub fn compute_f_rad(
    temp: AutoDiffReal2VarOrder3,
    den: AutoDiffReal2VarOrder3,
) -> AutoDiffReal2VarOrder3 {
    -CRAD * temp.pow4() / (3.0 * den)
}

/// Compute the specific ideal-ion Helmholtz free energy
/// $F_{\mathrm{ideal,ion}}(T, \rho)$ in $\mathrm{erg\,g^{-1}}$
/// (`compute_F_ideal_ion` in `skye_ideal.f90`).
///
/// # Arguments
/// * `temp` - Temperature $T$ in $\mathrm{K}$ seeded as variable 1.
/// * `den` - Mass density $\rho$ in $\mathrm{g\,cm^{-3}}$ seeded as variable 2.
/// * `abar` - Mean baryon number $\bar{A}$.
/// * `weights` - Atomic weights $W_j$ in $\mathrm{amu}$ per active species.
/// * `ya` - Normalized number fractions $y_j$ ($\sum_j y_j = 1$) per active
///   species.
#[must_use]
pub fn compute_f_ideal_ion(
    temp: AutoDiffReal2VarOrder3,
    den: AutoDiffReal2VarOrder3,
    abar: f64,
    weights: &[f64],
    ya: &[f64],
) -> AutoDiffReal2VarOrder3 {
    let kt = KERG * temp;
    let n = den / (AMU * abar);
    let n_q = kt.powf(1.5) / sifac_constant();

    let mut f_ideal_ion = AutoDiffReal2VarOrder3::constant(0.0);
    for (&w_j, &y_j) in weights.iter().zip(ya.iter()) {
        let n_j = y_j * n;
        let n_qj = n_q * pow(w_j, 1.5);
        f_ideal_ion = f_ideal_ion + y_j * ((n_j / n_qj).log() - 1.0);
    }
    f_ideal_ion * kt / (AMU * abar)
}

/// Compute the free electron number density $n_e = N_{\mathrm{A}} (\bar{Z} /
/// \bar{A}) \rho$ in $\mathrm{cm^{-3}}$ (`compute_xne` in `skye_ideal.f90`).
#[must_use]
#[inline]
pub fn compute_xne(
    den: AutoDiffReal2VarOrder3,
    ytot1: f64,
    zbar: f64,
) -> AutoDiffReal2VarOrder3 {
    let xni = AVO * ytot1 * den;
    xni * zbar
}

/// Compute specific entropy $S = -(\partial F / \partial T)_\rho$, specific
/// internal energy $E = F + T S$, and gas pressure
/// $P = \rho^{2} (\partial F / \partial \rho)_T$ from a specific Helmholtz free
/// energy $F(T, \rho)$ (`thermodynamics_from_free_energy` in
/// `skye_thermodynamics.f90`).
#[must_use]
#[inline]
pub fn thermodynamics_from_free_energy(
    f: AutoDiffReal2VarOrder3,
    temp: AutoDiffReal2VarOrder3,
    den: AutoDiffReal2VarOrder3,
) -> (
    AutoDiffReal2VarOrder3,
    AutoDiffReal2VarOrder3,
    AutoDiffReal2VarOrder3,
) {
    let s = -f.differentiate_1();
    let p = den.pow2() * f.differentiate_2();
    let e = f + temp * s;
    (s, e, p)
}

/// Secondary thermodynamic quantities derived from $(T, \rho, S, E, P)$.
#[derive(Clone, Copy, Debug)]
pub struct DerivedThermodynamics {
    /// Isochoric specific heat $C_V = (\partial E / \partial T)_\rho$.
    pub cv: AutoDiffReal2VarOrder3,
    /// Isobaric specific heat $C_P$.
    pub cp: AutoDiffReal2VarOrder3,
    /// Temperature susceptibility
    /// $\chi_T = (\partial\ln P / \partial\ln T)_\rho$.
    pub chit: AutoDiffReal2VarOrder3,
    /// Density susceptibility
    /// $\chi_\rho = (\partial\ln P / \partial\ln\rho)_T$.
    pub chid: AutoDiffReal2VarOrder3,
    /// First adiabatic index $\Gamma_1$.
    pub gam1: AutoDiffReal2VarOrder3,
    /// Second adiabatic index `(gam2 = 1 - nabad)` in MESA notation.
    pub gam2: AutoDiffReal2VarOrder3,
    /// Third adiabatic index $\Gamma_3$.
    pub gam3: AutoDiffReal2VarOrder3,
    /// Adiabatic temperature gradient
    /// $\nabla_{\mathrm{ad}} = (\Gamma_3 - 1) / \Gamma_1$.
    pub nabad: AutoDiffReal2VarOrder3,
    /// Relativistic sound speed $c_s$ in $\mathrm{cm\,s^{-1}}$.
    pub cs: AutoDiffReal2VarOrder3,
}

/// Compute susceptibilities, specific heats, adiabatic exponents, and sound
/// speed (`compute_derived_quantities` in `skye_thermodynamics.f90`).
#[must_use]
pub fn compute_derived_quantities(
    temp: AutoDiffReal2VarOrder3,
    dens: AutoDiffReal2VarOrder3,
    _s: AutoDiffReal2VarOrder3,
    e: AutoDiffReal2VarOrder3,
    p: AutoDiffReal2VarOrder3,
) -> DerivedThermodynamics {
    let chit = p.differentiate_1() * temp / p;
    let chid = p.differentiate_2() * dens / p;
    let cv = e.differentiate_1();
    let gam3 = 1.0 + (p / dens) * chit / (temp * cv);
    let gam1 = chid + (gam3 - 1.0) * chit;
    let nabad = (gam3 - 1.0) / gam1;
    let gam2 = 1.0 - nabad;
    let cp = cv * gam1 / chid;
    let cs =
        CLIGHT * (gam1 / (1.0 + (dens / p) * (e + CLIGHT * CLIGHT))).sqrt();
    DerivedThermodynamics {
        cv,
        cp,
        chit,
        chid,
        gam1,
        gam2,
        gam3,
        nabad,
        cs,
    }
}

/// Assemble the 26-channel MESA [`EosEvalOutput`] from free-energy components
/// (`pack_for_export` in `skye_thermodynamics.f90`).
///
/// # Errors
/// Returns [`EosError::InvalidState`] if total entropy $S < 0$, total internal
/// energy $E < 0$, or gas pressure $P_{\mathrm{gas}} < 0$.
#[allow(clippy::too_many_arguments)]
pub fn pack_for_export(
    f_ideal_ion: AutoDiffReal2VarOrder3,
    f_coul: AutoDiffReal2VarOrder3,
    _f_rad: AutoDiffReal2VarOrder3,
    f_ele: AutoDiffReal2VarOrder3,
    temp: AutoDiffReal2VarOrder3,
    dens: AutoDiffReal2VarOrder3,
    xnefer: AutoDiffReal2VarOrder3,
    etaele: AutoDiffReal2VarOrder3,
    abar: f64,
    zbar: f64,
    phase: AutoDiffReal2VarOrder3,
    latent_ddlnt: AutoDiffReal2VarOrder3,
    latent_ddlnrho: AutoDiffReal2VarOrder3,
    species: usize,
) -> Result<EosEvalOutput, EosError> {
    let f_gas = f_ideal_ion + f_coul + f_ele;
    let (sgas, egas, pgas) = thermodynamics_from_free_energy(f_gas, temp, dens);

    // Explicit radiation terms avoiding rho^2 / rho^2 cancellation in chid.
    let prad = CRAD * temp.pow4() / 3.0;
    let erad = CRAD * temp.pow4() / dens;
    let srad = 4.0 * CRAD * temp.pow3() / (3.0 * dens);

    let p = prad + pgas;
    let e = erad + egas;
    let s = srad + sgas;

    if s.val < 0.0 || e.val < 0.0 || pgas.val < 0.0 {
        return Err(EosError::InvalidState {
            context: "pack_for_export",
            detail: format!(
                "non-positive state (S={:e}, E={:e}, Pgas={:e})",
                s.val, e.val, pgas.val
            ),
        });
    }

    let ln_s = s.log();
    let ln_e = e.log();
    let ln_pgas = pgas.log();
    let mu = AutoDiffReal2VarOrder3::constant(abar / (1.0 + zbar));
    let ln_free_e = (xnefer.max_scalar(1e-99) / (AVO * dens)).log();

    let d = compute_derived_quantities(temp, dens, s, e, p);
    let mut out = EosEvalOutput::zeros(species);
    let t_val = temp.val;
    let rho_val = dens.val;
    let channels: [(usize, AutoDiffReal2VarOrder3); 19] = [
        (I_LN_S, ln_s),
        (I_LN_E, ln_e),
        (I_LN_PGAS, ln_pgas),
        (I_MU, mu),
        (I_GRAD_AD, d.nabad),
        (I_CHI_RHO, d.chid),
        (I_CHI_T, d.chit),
        (I_CP, d.cp),
        (I_CV, d.cv),
        (I_DE_DRHO, e.differentiate_2()),
        (I_DS_DT, s.differentiate_1()),
        (I_DS_DRHO, s.differentiate_2()),
        (I_LN_FREE_E, ln_free_e),
        (I_GAMMA1, d.gam1),
        (I_GAMMA3, d.gam3),
        (I_ETA, etaele),
        (I_PHASE, phase),
        (I_LATENT_DDLNT, latent_ddlnt),
        (I_LATENT_DDLNRHO, latent_ddlnrho),
    ];
    for (idx, q) in channels {
        out.res[idx] = q.val;
        out.d_dln_t[idx] = q.d1val1 * t_val;
        out.d_dlnd[idx] = q.d1val2 * rho_val;
    }

    Ok(out)
}
