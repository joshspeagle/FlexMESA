# SPDX-License-Identifier: LGPL-3.0-or-later
"""Radiative, conductive, and Compton opacity (`kap`) kernels for FlexMESA.

Ports upstream Fortran MESA's `kap` module (`kap_def.f90`, `kap_lib.f90`,
`kap_eval.f90`, `kap_eval_fixed.f90`, `kap_eval_co.f90`, `condint.f90`,
`kap_aesopus.f90`, and `kapcn.f90`) with Safe Rust execution via
`flexmesa._rust` and `@jax.jit` / `jax.custom_jvp` differentiable kernels.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from flexmesa import _dispatch, _rust
from flexmesa.kap import jax_kernels, submodels

I_FRAC_LOW_T: int = 0
I_FRAC_HIGH_T: int = 1
I_FRAC_TYPE2: int = 2
I_FRAC_COMPTON: int = 3
NUM_KAP_FRACS: int = 4

combine_rad_with_conduction_jax = jax_kernels.combine_rad_with_conduction_jax
compton_opacity_jax = jax_kernels.compton_opacity_jax
make_differentiable_kap = jax_kernels.make_differentiable_kap
kap_get_elect_cond_opacity = submodels.kap_get_elect_cond_opacity
kap_get_compton_opacity = submodels.kap_get_compton_opacity
kap_get_fixed_metal = submodels.kap_get_fixed_metal
kap_get_co = submodels.kap_get_co
kap_get_aesopus = submodels.kap_get_aesopus
kap_get_kapcn = submodels.kap_get_kapcn


@dataclasses.dataclass(frozen=True)
class KapEvalOutput:
    """Single-zone opacity evaluation output (`kap_get` / `Get_kap_Results`).

    Attributes:
        kap: Total Rosseland mean opacity in `cm^2 g^-1`.
        log10_kap: Base-10 logarithm of total opacity `log10(kap / cm^2 g^-1)`.
        dlnkap_dln_rho: Partial derivative `(d ln kap / d ln rho)_T`.
        dlnkap_dln_t: Partial derivative `(d ln kap / d ln T)_rho`.
        kap_fracs: 4-element `float64` array `[frac_lowT, frac_highT,
            frac_Type2, frac_Compton]`.
    """

    kap: float
    log10_kap: float
    dlnkap_dln_rho: float
    dlnkap_dln_t: float
    kap_fracs: np.ndarray


def _pack_kap_output(
    raw: tuple[float, float, float, list[float]],
) -> KapEvalOutput:
    """Convert raw `(kap, dln_dr, dln_dt, fracs)` tuple to [`KapEvalOutput`]."""
    kap, dln_dr, dln_dt, fracs = raw
    return KapEvalOutput(
        kap=float(kap),
        log10_kap=float(np.log10(kap)),
        dlnkap_dln_rho=float(dln_dr),
        dlnkap_dln_t=float(dln_dt),
        kap_fracs=np.asarray(fracs, dtype=np.float64),
    )


def kap_get_results(
    z: float,
    x: float,
    zbase: float,
    xc: float = 0.0,
    xn: float = 0.0,
    xo: float = 0.0,
    xne: float = 0.0,
    log_rho: float = 0.0,
    log_t: float = 6.0,
    lnfree_e: float = 0.0,
    d_lnfree_e_dln_rho: float = 0.0,
    d_lnfree_e_dln_t: float = 0.0,
    eta: float = 0.0,
    d_eta_dln_rho: float = 0.0,
    d_eta_dln_t: float = 0.0,
    zbar: float = 1.0,
    *,
    kap_option: str = "gs98",
    kap_co_option: str = "gs98_co",
    kap_low_t_option: str = "lowT_fa05_gs98",
    kap_blend_log_t_upper_bdy: float = 3.88,
    kap_blend_log_t_lower_bdy: float = 3.80,
    cubic_interpolation_in_x: bool = False,
    cubic_interpolation_in_z: bool = False,
    include_electron_conduction: bool = True,
    use_blouin_conductive_opacities: bool = True,
    use_zbase_for_type1: bool = False,
    use_type2_opacities: bool = False,
    kap_type2_full_off_x: float = 1.0e-3,
    kap_type2_full_on_x: float = 1.0e-6,
    kap_type2_full_off_dz: float = 0.001,
    kap_type2_full_on_dz: float = 0.01,
    backend: _dispatch.Backend = "auto",
) -> KapEvalOutput:
    """Evaluate single-zone radiative + conductive + Compton opacity.

    Mirrors `Get_kap_Results` in `kap/private/kap_eval.f90`.

    Args:
        z: Total metal mass fraction `Z`.
        x: Hydrogen mass fraction `X`.
        zbase: Base metallicity `Zbase`.
        xc: Carbon mass fraction `XC`.
        xn: Nitrogen mass fraction `XN`.
        xo: Oxygen mass fraction `XO`.
        xne: Neon mass fraction `XNe`.
        log_rho: Base-10 log mass density `log10(rho / g cm^-3)`.
        log_t: Base-10 log temperature `log10(T / K)`.
        lnfree_e: Natural log of free electrons per nucleon.
        d_lnfree_e_dln_rho: Partial `(d ln free_e / d ln rho)_T`.
        d_lnfree_e_dln_t: Partial `(d ln free_e / d ln T)_rho`.
        eta: Electron degeneracy parameter.
        d_eta_dln_rho: Partial `(d eta / d ln rho)_T`.
        d_eta_dln_t: Partial `(d eta / d ln T)_rho`.
        zbar: Mean ionic charge `Zbar`.
        kap_option: High-T Type 1 table prefix (`"gs98"`, `"a09"`, etc.).
        kap_co_option: High-T Type 2 table prefix (`"gn93_co"`, `"a09_co"`).
        kap_low_t_option: Low-T table prefix (`"lowT_fa05_gs98"`, `"AESOPUS"`,
            `"kapCN"`).
        kap_blend_log_t_upper_bdy: Upper `log10(T/K)` for low-T/high-T blend.
        kap_blend_log_t_lower_bdy: Lower `log10(T/K)` for low-T/high-T blend.
        cubic_interpolation_in_x: Use cubic interpolation in `X`.
        cubic_interpolation_in_z: Use cubic interpolation in `Z`.
        include_electron_conduction: Include electron thermal conduction.
        use_blouin_conductive_opacities: Use Blouin et al. (2020) H/He
            conduction.
        use_zbase_for_type1: Use `min(Z, Zbase)` for Type 1 when Type 2 active.
        use_type2_opacities: Enable Type 2 C/O-enhanced tables.
        kap_type2_full_off_x: `X` threshold above which Type 2 is off.
        kap_type2_full_on_x: `X` threshold below which Type 2 is fully on.
        kap_type2_full_off_dz: `Z - Zbase` threshold below which Type 2 is off.
        kap_type2_full_on_dz: `Z - Zbase` threshold above which Type 2 is on.
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        [`KapEvalOutput`] containing `kap`, `log10_kap`, `dlnkap_dln_rho`,
        `dlnkap_dln_t`, and `kap_fracs`.
    """
    _ = _dispatch.resolve_backend(backend, z, x, zbase, log_rho, log_t)
    return _pack_kap_output(
        _rust.kap_get_results_rust(
            float(z),
            float(x),
            float(zbase),
            float(xc),
            float(xn),
            float(xo),
            float(xne),
            float(log_rho),
            float(log_t),
            float(lnfree_e),
            float(d_lnfree_e_dln_rho),
            float(d_lnfree_e_dln_t),
            float(eta),
            float(d_eta_dln_rho),
            float(d_eta_dln_t),
            zbar=float(zbar),
            kap_option=kap_option,
            kap_co_option=kap_co_option,
            kap_low_t_option=kap_low_t_option,
            kap_blend_log_t_upper_bdy=float(kap_blend_log_t_upper_bdy),
            kap_blend_log_t_lower_bdy=float(kap_blend_log_t_lower_bdy),
            cubic_interpolation_in_x=bool(cubic_interpolation_in_x),
            cubic_interpolation_in_z=bool(cubic_interpolation_in_z),
            include_electron_conduction=bool(include_electron_conduction),
            use_blouin_conductive_opacities=bool(
                use_blouin_conductive_opacities
            ),
            use_zbase_for_type1=bool(use_zbase_for_type1),
            use_type2_opacities=bool(use_type2_opacities),
            kap_type2_full_off_x=float(kap_type2_full_off_x),
            kap_type2_full_on_x=float(kap_type2_full_on_x),
            kap_type2_full_off_dz=float(kap_type2_full_off_dz),
            kap_type2_full_on_dz=float(kap_type2_full_on_dz),
        )
    )


def kap_get(
    chem_id: np.ndarray,
    xa: np.ndarray,
    log_rho: float,
    log_t: float,
    lnfree_e: float = 0.0,
    d_lnfree_e_dln_rho: float = 0.0,
    d_lnfree_e_dln_t: float = 0.0,
    eta: float = 0.0,
    d_eta_dln_rho: float = 0.0,
    d_eta_dln_t: float = 0.0,
    zbase: float = 0.02,
    **kwargs: Any,
) -> KapEvalOutput:
    """Evaluate `kap_get` from 1-based `chem_id` and mass-fraction vector `xa`.

    Args:
        chem_id: 1-based nuclide indices, shape `(N_spec,)`.
        xa: Mass fractions, shape `(N_spec,)`.
        log_rho: Base-10 log mass density `log10(rho / g cm^-3)`.
        log_t: Base-10 log temperature `log10(T / K)`.
        lnfree_e: Natural log of free electrons per nucleon.
        d_lnfree_e_dln_rho: Partial `(d ln free_e / d ln rho)_T`.
        d_lnfree_e_dln_t: Partial `(d ln free_e / d ln T)_rho`.
        eta: Electron degeneracy parameter.
        d_eta_dln_rho: Partial `(d eta / d ln rho)_T`.
        d_eta_dln_t: Partial `(d eta / d ln T)_rho`.
        zbase: Base metallicity `Zbase`.
        **kwargs: Optional `KapConfig` overrides.

    Returns:
        [`KapEvalOutput`] for the given isotopic composition and state.
    """
    cid_arr = np.ascontiguousarray(chem_id, dtype=np.uintp)
    xa_arr = np.ascontiguousarray(xa, dtype=np.float64)
    return _pack_kap_output(
        _rust.kap_get_rust(
            cid_arr,
            xa_arr,
            float(log_rho),
            float(log_t),
            float(lnfree_e),
            float(d_lnfree_e_dln_rho),
            float(d_lnfree_e_dln_t),
            float(eta),
            float(d_eta_dln_rho),
            float(d_eta_dln_t),
            zbase=float(zbase),
            **kwargs,
        )
    )


def kap_get_zones(
    chem_id: np.ndarray,
    xa_zones: np.ndarray,
    log_rho_zones: np.ndarray,
    log_t_zones: np.ndarray,
    lnfree_e_zones: np.ndarray,
    d_lnfree_e_dln_rho_zones: np.ndarray,
    d_lnfree_e_dln_t_zones: np.ndarray,
    eta_zones: np.ndarray,
    d_eta_dln_rho_zones: np.ndarray,
    d_eta_dln_t_zones: np.ndarray,
    zbase: float = 0.02,
    *,
    parallel: bool = True,
    **kwargs: Any,
) -> np.ndarray:
    """Evaluate `kap_get` across `N_zones` zones in parallel.

    Args:
        chem_id: 1-based nuclide indices, shape `(N_spec,)`.
        xa_zones: Zone mass fractions, shape `(N_zones, N_spec)`.
        log_rho_zones: Zone `log10(rho / g cm^-3)`, shape `(N_zones,)`.
        log_t_zones: Zone `log10(T / K)`, shape `(N_zones,)`.
        lnfree_e_zones: Zone `ln(free_e)`, shape `(N_zones,)`.
        d_lnfree_e_dln_rho_zones: Zone `(d ln free_e / d ln rho)_T`,
            shape `(N_zones,)`.
        d_lnfree_e_dln_t_zones: Zone `(d ln free_e / d ln T)_rho`,
            shape `(N_zones,)`.
        eta_zones: Zone electron degeneracy parameter, shape `(N_zones,)`.
        d_eta_dln_rho_zones: Zone `(d eta / d ln rho)_T`, shape `(N_zones,)`.
        d_eta_dln_t_zones: Zone `(d eta / d ln T)_rho`, shape `(N_zones,)`.
        zbase: Base metallicity `Zbase`.
        parallel: Dispatch across the persistent 4-thread worker pool.
        **kwargs: Optional `KapConfig` overrides.

    Returns:
        Array of shape `(N_zones, 7)` with columns `[kap, dlnkap_dlnRho,
        dlnkap_dlnT, frac_lowT, frac_highT, frac_Type2, frac_Compton]`.
    """
    return _rust.kap_get_zones_rust(
        np.ascontiguousarray(chem_id, dtype=np.uintp),
        np.ascontiguousarray(xa_zones, dtype=np.float64),
        np.ascontiguousarray(log_rho_zones, dtype=np.float64),
        np.ascontiguousarray(log_t_zones, dtype=np.float64),
        np.ascontiguousarray(lnfree_e_zones, dtype=np.float64),
        np.ascontiguousarray(d_lnfree_e_dln_rho_zones, dtype=np.float64),
        np.ascontiguousarray(d_lnfree_e_dln_t_zones, dtype=np.float64),
        np.ascontiguousarray(eta_zones, dtype=np.float64),
        np.ascontiguousarray(d_eta_dln_rho_zones, dtype=np.float64),
        np.ascontiguousarray(d_eta_dln_t_zones, dtype=np.float64),
        zbase=float(zbase),
        parallel=bool(parallel),
        **kwargs,
    )
