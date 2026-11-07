# SPDX-License-Identifier: LGPL-3.0-or-later
"""Sub-model opacity evaluation wrappers (`flexmesa.kap.submodels`).

Exposes direct access to Type 1 fixed-metal tables (`kap_get_fixed_metal`),
Type 2 C/O-enhanced tables (`kap_get_co`), Marigo & Aringer (2009) `AESOPUS`
low-temperature molecular tables (`kap_get_aesopus`), Lederer & Aringer (2009)
`kapCN` C/N-varying molecular tables (`kap_get_kapcn`), Potekhin/Blouin
electron thermal conduction (`kap_get_elect_cond_opacity`), and Poutanen (2017)
relativistic/degenerate Compton scattering (`kap_get_compton_opacity`).
"""

from __future__ import annotations

import jax

from flexmesa import _dispatch, _rust
from flexmesa.kap import jax_kernels


def kap_get_elect_cond_opacity(
    zbar: float,
    log_rho: float,
    log_t: float,
    *,
    use_blouin: bool = True,
) -> tuple[float, float, float]:
    """Evaluate Potekhin / Blouin electron thermal conduction opacity.

    Args:
        zbar: Mean ionic charge `Zbar`.
        log_rho: Base-10 log mass density `log10(rho / g cm^-3)`.
        log_t: Base-10 log temperature `log10(T / K)`.
        use_blouin: Apply Blouin et al. (2020) H/He moderate-coupling
            correction.

    Returns:
        Tuple `(kap, dlnkap_dln_rho, dlnkap_dln_t)` where `kap` has units
        `cm^2 g^-1`.
    """
    return _rust.kap_get_elect_cond_opacity_rust(
        float(zbar), float(log_rho), float(log_t), use_blouin=bool(use_blouin)
    )


def kap_get_compton_opacity(
    rho: jax.Array | float,
    t: jax.Array | float,
    lnfree_e: jax.Array | float,
    d_lnfree_e_dln_rho: jax.Array | float,
    d_lnfree_e_dln_t: jax.Array | float,
    eta: jax.Array | float,
    d_eta_dln_rho: jax.Array | float,
    d_eta_dln_t: jax.Array | float,
    *,
    backend: _dispatch.Backend = "auto",
) -> tuple[jax.Array | float, jax.Array | float, jax.Array | float]:
    """Evaluate Poutanen (2017) relativistic/degenerate Compton opacity.

    Args:
        rho: Mass density in `g cm^-3`, shape `(*batch,)`.
        t: Temperature in `K`, shape `(*batch,)`.
        lnfree_e: Natural log of free electrons per nucleon, shape `(*batch,)`.
        d_lnfree_e_dln_rho: Partial `(d ln free_e / d ln rho)_T`.
        d_lnfree_e_dln_t: Partial `(d ln free_e / d ln T)_rho`.
        eta: Electron degeneracy parameter.
        d_eta_dln_rho: Partial `(d eta / d ln rho)_T`.
        d_eta_dln_t: Partial `(d eta / d ln T)_rho`.
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Tuple `(kap, dlnkap_dln_rho, dlnkap_dln_t)`. Returns `jax.Array`
        outputs when any input is a `jax.Array` or `backend="jax"` under
        JAX tracing, and Python `float` scalars otherwise.
    """
    args = (
        rho,
        t,
        lnfree_e,
        d_lnfree_e_dln_rho,
        d_lnfree_e_dln_t,
        eta,
        d_eta_dln_rho,
        d_eta_dln_t,
    )
    resolved = _dispatch.resolve_backend(backend, *args)
    if resolved == "jax":
        k, dr, dt = jax_kernels.compton_opacity_jax(*args)
        if _dispatch.is_jax_input(*args):
            return k, dr, dt
        return float(k), float(dr), float(dt)
    return _rust.kap_get_compton_opacity_rust(
        float(rho),
        float(t),
        float(lnfree_e),
        float(d_lnfree_e_dln_rho),
        float(d_lnfree_e_dln_t),
        float(eta),
        float(d_eta_dln_rho),
        float(d_eta_dln_t),
    )


def kap_get_fixed_metal(
    z: float,
    x: float,
    log_rho: float,
    log_t: float,
    *,
    prefix: str = "gs98",
    cubic_x: bool = False,
    cubic_z: bool = False,
) -> tuple[float, float, float]:
    """Evaluate Type 1 fixed-metal opacity.

    Args:
        z: Total metal mass fraction `Z`.
        x: Hydrogen mass fraction `X`.
        log_rho: Base-10 log mass density `log10(rho / g cm^-3)`.
        log_t: Base-10 log temperature `log10(T / K)`.
        prefix: Fixed-metal table prefix (`"gs98"`, `"a09"`, etc.).
        cubic_x: Use cubic interpolation in `X`.
        cubic_z: Use cubic interpolation in `Z`.

    Returns:
        Tuple `(log10_kap, dlnkap_dln_rho, dlnkap_dln_t)`.
    """
    return _rust.kap_get_fixed_metal_rust(
        float(z),
        float(x),
        float(log_rho),
        float(log_t),
        prefix=prefix,
        cubic_x=bool(cubic_x),
        cubic_z=bool(cubic_z),
    )


def kap_get_co(
    zbase: float,
    x: float,
    dxc: float,
    dxo: float,
    log_rho: float,
    log_t: float,
    *,
    kap_co_option: str = "gn93_co",
    cubic_x: bool = False,
    cubic_z: bool = False,
) -> tuple[float, float, float]:
    """Evaluate Type 2 C/O-enhanced opacity.

    Args:
        zbase: Base metallicity `Zbase`.
        x: Hydrogen mass fraction `X`.
        dxc: Carbon enhancement `dXC`.
        dxo: Oxygen enhancement `dXO`.
        log_rho: Base-10 log mass density `log10(rho / g cm^-3)`.
        log_t: Base-10 log temperature `log10(T / K)`.
        kap_co_option: Type 2 table prefix (`"gn93_co"`, `"a09_co"`).
        cubic_x: Use cubic interpolation in `X`.
        cubic_z: Use cubic interpolation in `Z`.

    Returns:
        Tuple `(log10_kap, dlnkap_dln_rho, dlnkap_dln_t)`.
    """
    return _rust.kap_get_co_rust(
        float(zbase),
        float(x),
        float(dxc),
        dxo=float(dxo),
        log_rho=float(log_rho),
        log_t=float(log_t),
        kap_co_option=kap_co_option,
        cubic_x=bool(cubic_x),
        cubic_z=bool(cubic_z),
    )


def kap_get_aesopus(
    zbase: float,
    x: float,
    xc: float,
    xn: float,
    xo: float,
    log_rho: float,
    log_t: float,
) -> tuple[float, float, float]:
    """Evaluate Marigo & Aringer (2009) `AESOPUS_AGSS09` low-T opacity.

    Args:
        zbase: Reference metallicity `Zref`.
        x: Hydrogen mass fraction `X`.
        xc: Carbon mass fraction `XC`.
        xn: Nitrogen mass fraction `XN`.
        xo: Oxygen mass fraction `XO`.
        log_rho: Base-10 log mass density `log10(rho / g cm^-3)`.
        log_t: Base-10 log temperature `log10(T / K)`.

    Returns:
        Tuple `(kap, dlnkap_dln_rho, dlnkap_dln_t)`.
    """
    return _rust.kap_get_aesopus_rust(
        float(zbase),
        float(x),
        float(xc),
        float(xn),
        float(xo),
        float(log_rho),
        float(log_t),
    )


def kap_get_kapcn(
    zbase: float,
    x: float,
    fc: float,
    fn_val: float,
    log_rho: float,
    log_t: float,
) -> tuple[float, float, float]:
    """Evaluate Lederer & Aringer (2009) `kapCN` low-T opacity.

    Args:
        zbase: Base metallicity `Zbase`.
        x: Hydrogen mass fraction `X`.
        fc: Dimensionless carbon enhancement factor
            `fC = (X_C / Z) / (X_C / Z)_solar`.
        fn_val: Dimensionless nitrogen enhancement factor
            `fN = (X_N / Z) / (X_N / Z)_solar`.
        log_rho: Base-10 log mass density `log10(rho / g cm^-3)`.
        log_t: Base-10 log temperature `log10(T / K)`.

    Returns:
        Tuple `(kap, dlnkap_dln_rho, dlnkap_dln_t)`.
    """
    return _rust.kap_get_kapcn_rust(
        float(zbase),
        float(x),
        float(fc),
        float(fn_val),
        float(log_rho),
        float(log_t),
    )
