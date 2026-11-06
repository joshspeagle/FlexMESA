# SPDX-License-Identifier: LGPL-3.0-or-later
"""Bolometric, synthetic filter, and zero-point photometry for ``colors``.

Ports upstream MESA ``colors/private/bolometric.f90`` and ``synthetic.f90``
into unified Safe Rust (``flexmesa._rust``) and ``@jax.jit`` (``float64``)
evaluators.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from flexmesa import _dispatch, _rust, constants
from flexmesa.colors import utils

type ArrayLike = npt.NDArray[np.float64] | jax.Array


def flux_to_magnitude(flux: float | jax.Array) -> float | jax.Array:
    """Convert integrated flux ``F > 0`` to Pogson magnitude ``-2.5 log10(F)``.

    Args:
        flux: Integrated flux ``F`` in ``erg s^-1 cm^-2``.

    Returns:
        Magnitude ``-2.5 log10(F)``, or ``99.0`` when ``F <= 0``.
    """
    if isinstance(flux, jax.Array):
        safe = jnp.where(flux > 0.0, flux, 1.0)
        return jnp.where(flux > 0.0, -2.5 * jnp.log10(safe), 99.0)
    f_val = float(flux)
    return 99.0 if f_val <= 0.0 else float(-2.5 * np.log10(f_val))


def clean_sed_fluxes_jax(
    wavelengths: jax.Array,
    fluxes: jax.Array,
) -> jax.Array:
    """Zero out non-positive interior wavelengths or negative SED fluxes.

    Matches the interior sanitization loop `1..n-1` in upstream MESA
    `bolometric.f90` and `synthetic.f90`.
    """
    n = wavelengths.shape[0]
    mask = (jnp.arange(n) < n - 1) & ((wavelengths <= 0.0) | (fluxes < 0.0))
    return jnp.where(mask, 0.0, fluxes)


@jax.jit
def calculate_bolometric_phot_jax(
    wavelengths: jax.Array,
    fluxes: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Integrate bolometric flux and magnitude in JAX ``float64``.

    Args:
        wavelengths: 1D wavelength grid in ``Angstrom`` of shape ``(n,)``.
        fluxes: 1D flux density in ``erg s^-1 cm^-2 Angstrom^-1`` of shape
            ``(n,)``.

    Returns:
        Tuple ``(bol_mag, bol_flux)``.
    """
    w = jnp.asarray(wavelengths, dtype=jnp.float64)
    f = jnp.asarray(fluxes, dtype=jnp.float64)
    cleaned = clean_sed_fluxes_jax(w, f)
    bol_flux = utils.simpson_integration_jax(w, cleaned)
    bol_mag = jnp.asarray(flux_to_magnitude(bol_flux), dtype=jnp.float64)
    return bol_mag, bol_flux


def calculate_bolometric_phot(
    wavelengths: ArrayLike,
    fluxes: ArrayLike,
    *,
    backend: _dispatch.Backend = "auto",
) -> tuple[float | jax.Array, float | jax.Array]:
    """Integrate bolometric flux and magnitude over ``(wavelengths, fluxes)``.

    Args:
        wavelengths: 1D wavelength grid in ``Angstrom`` of shape ``(n,)``.
        fluxes: 1D flux density in ``erg s^-1 cm^-2 Angstrom^-1`` of shape
            ``(n,)``.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        Tuple ``(bol_mag, bol_flux)``.
    """
    resolved = _dispatch.resolve_backend(backend, wavelengths, fluxes)
    if resolved == "rust":
        w_arr = _dispatch.as_c_f64(wavelengths)
        f_arr = _dispatch.as_c_f64(fluxes)
        return _rust.py_colors_calculate_bolometric_phot(w_arr, f_arr)
    return calculate_bolometric_phot_jax(
        jnp.asarray(wavelengths, dtype=jnp.float64),
        jnp.asarray(fluxes, dtype=jnp.float64),
    )


@jax.jit
def calculate_synthetic_flux_jax(
    wavelengths: jax.Array,
    convolved_flux: jax.Array,
    filter_on_sed_grid: jax.Array,
) -> jax.Array:
    """Compute photon-counting synthetic filter flux in JAX ``float64``.

    Args:
        wavelengths: 1D wavelength grid in ``Angstrom`` of shape ``(n,)``.
        convolved_flux: 1D product ``F_lambda * S_lambda`` of shape ``(n,)``.
        filter_on_sed_grid: 1D filter transmission ``S_lambda`` of shape
            ``(n,)``.

    Returns:
        Photon-counting mean flux
        ``int F S lambda dlambda / int S lambda dlambda``.
    """
    w = jnp.asarray(wavelengths, dtype=jnp.float64)
    cf = jnp.asarray(convolved_flux, dtype=jnp.float64)
    fg = jnp.asarray(filter_on_sed_grid, dtype=jnp.float64)
    num = utils.simpson_integration_jax(w, cf * w)
    den = utils.simpson_integration_jax(w, fg * w)
    return num / den


@jax.jit
def calculate_synthetic_jax(
    wavelengths: jax.Array,
    fluxes: jax.Array,
    filter_wavelengths: jax.Array,
    filter_trans: jax.Array,
    zero_point_flux: jax.Array | float,
) -> jax.Array:
    """Compute synthetic filter magnitude in JAX ``float64``.

    Args:
        wavelengths: 1D SED wavelength grid in ``Angstrom``.
        fluxes: 1D SED flux density in ``erg s^-1 cm^-2 Angstrom^-1``.
        filter_wavelengths: 1D filter wavelength grid in ``Angstrom``.
        filter_trans: 1D filter transmission curve.
        zero_point_flux: Zero-point flux ``F_0`` in
            ``erg s^-1 cm^-2 Angstrom^-1``.

    Returns:
        Synthetic magnitude ``-2.5 log10(F_syn / F_0)``.
    """
    w = jnp.asarray(wavelengths, dtype=jnp.float64)
    f = jnp.asarray(fluxes, dtype=jnp.float64)
    fw = jnp.asarray(filter_wavelengths, dtype=jnp.float64)
    ft = jnp.asarray(filter_trans, dtype=jnp.float64)
    zp = jnp.asarray(zero_point_flux, dtype=jnp.float64)
    cleaned = clean_sed_fluxes_jax(w, f)
    f_on_sed = utils.interpolate_array_jax(fw, ft, w)
    syn_flux = calculate_synthetic_flux_jax(w, cleaned * f_on_sed, f_on_sed)
    return -2.5 * jnp.log10(syn_flux / zp)


def calculate_synthetic(
    wavelengths: ArrayLike,
    fluxes: ArrayLike,
    filter_wavelengths: ArrayLike,
    filter_trans: ArrayLike,
    zero_point_flux: float | jax.Array,
    *,
    backend: _dispatch.Backend = "auto",
) -> float | jax.Array:
    """Compute synthetic filter magnitude from an SED and filter curve.

    Args:
        wavelengths: 1D SED wavelength grid in ``Angstrom``.
        fluxes: 1D SED flux density in ``erg s^-1 cm^-2 Angstrom^-1``.
        filter_wavelengths: 1D filter wavelength grid in ``Angstrom``.
        filter_trans: 1D filter transmission curve.
        zero_point_flux: Zero-point flux ``F_0``.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        Synthetic magnitude ``-2.5 log10(F_syn / F_0)``.
    """
    resolved = _dispatch.resolve_backend(
        backend,
        wavelengths,
        fluxes,
        filter_wavelengths,
        filter_trans,
        zero_point_flux,
    )
    if resolved == "rust":
        return _rust.py_colors_calculate_synthetic(
            _dispatch.as_c_f64(wavelengths),
            _dispatch.as_c_f64(fluxes),
            _dispatch.as_c_f64(filter_wavelengths),
            _dispatch.as_c_f64(filter_trans),
            float(zero_point_flux),
        )
    return calculate_synthetic_jax(
        jnp.asarray(wavelengths, dtype=jnp.float64),
        jnp.asarray(fluxes, dtype=jnp.float64),
        jnp.asarray(filter_wavelengths, dtype=jnp.float64),
        jnp.asarray(filter_trans, dtype=jnp.float64),
        zero_point_flux,
    )


def compute_vega_zero_point(
    vega_wave: ArrayLike,
    vega_flux: ArrayLike,
    filt_wave: ArrayLike,
    filt_trans: ArrayLike,
    *,
    backend: _dispatch.Backend = "auto",
) -> float | jax.Array:
    """Compute the Vega photon-counting zero-point flux for a filter curve.

    Args:
        vega_wave: 1D Vega wavelength grid in ``Angstrom``.
        vega_flux: 1D Vega flux spectrum in ``erg s^-1 cm^-2 Angstrom^-1``.
        filt_wave: 1D filter wavelength grid in ``Angstrom``.
        filt_trans: 1D filter transmission curve.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        Vega zero-point flux in ``erg s^-1 cm^-2 Angstrom^-1``.
    """
    resolved = _dispatch.resolve_backend(
        backend, vega_wave, vega_flux, filt_wave, filt_trans
    )
    if resolved == "rust":
        return _rust.py_colors_compute_vega_zero_point(
            _dispatch.as_c_f64(vega_wave),
            _dispatch.as_c_f64(vega_flux),
            _dispatch.as_c_f64(filt_wave),
            _dispatch.as_c_f64(filt_trans),
        )
    vw = jnp.asarray(vega_wave, dtype=jnp.float64)
    vf = jnp.asarray(vega_flux, dtype=jnp.float64)
    fw = jnp.asarray(filt_wave, dtype=jnp.float64)
    ft = jnp.asarray(filt_trans, dtype=jnp.float64)
    f_on_vega = utils.interpolate_array_jax(fw, ft, vw)
    return calculate_synthetic_flux_jax(vw, vf * f_on_vega, f_on_vega)


def compute_ab_zero_point(
    filt_wave: ArrayLike,
    filt_trans: ArrayLike,
    *,
    backend: _dispatch.Backend = "auto",
) -> float | jax.Array:
    """Compute the AB photon-counting zero-point flux ($f_\\nu = 3631$ Jy).

    Args:
        filt_wave: 1D filter wavelength grid in ``Angstrom``.
        filt_trans: 1D filter transmission curve.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        AB zero-point flux in ``erg s^-1 cm^-2 Angstrom^-1``.
    """
    resolved = _dispatch.resolve_backend(backend, filt_wave, filt_trans)
    if resolved == "rust":
        return _rust.py_colors_compute_ab_zero_point(
            _dispatch.as_c_f64(filt_wave),
            _dispatch.as_c_f64(filt_trans),
        )
    fw = jnp.asarray(filt_wave, dtype=jnp.float64)
    ft = jnp.asarray(filt_trans, dtype=jnp.float64)
    ab_f = jnp.where(
        fw > 0.0, 3.631e-20 * ((constants.CLIGHT * 1.0e8) / (fw * fw)), 0.0
    )
    return calculate_synthetic_flux_jax(fw, ab_f * ft, ft)


def compute_st_zero_point(
    filt_wave: ArrayLike,
    filt_trans: ArrayLike,
    *,
    backend: _dispatch.Backend = "auto",
) -> float | jax.Array:
    """Compute the ST photon-counting zero-point flux ($f_\\lambda = 3.63e-9$).

    Args:
        filt_wave: 1D filter wavelength grid in ``Angstrom``.
        filt_trans: 1D filter transmission curve.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        ST zero-point flux in ``erg s^-1 cm^-2 Angstrom^-1``.
    """
    resolved = _dispatch.resolve_backend(backend, filt_wave, filt_trans)
    if resolved == "rust":
        return _rust.py_colors_compute_st_zero_point(
            _dispatch.as_c_f64(filt_wave),
            _dispatch.as_c_f64(filt_trans),
        )
    fw = jnp.asarray(filt_wave, dtype=jnp.float64)
    ft = jnp.asarray(filt_trans, dtype=jnp.float64)
    return calculate_synthetic_flux_jax(fw, 3.63e-9 * ft, ft)
