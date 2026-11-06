# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Continuum depression and typical diffusion charges (`mod_typical_charge`).

Implements the Three-Tier Hybrid (`Tier B`) continuum-depression and typical
ionic charge routines from `mod_ionization.f90` and
`star/private/mod_typical_charge.f90`:
- `chi_info`, `chi_info_jax`, `chi_effective`, `chi_effective_jax`: Paquette et
  al. (1986, Eq. 20) and Dupuis et al. (1992, Eq. 3) continuum-depression
  ionization threshold in $\mathrm{eV}$.
- `eval_typical_charge`, `ionization_potential`: Whole-number typical ionic
  charge against the Allen (1973) ionization potentials ($Z \in [1, 30]$).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from flexmesa import _dispatch, _rust
from flexmesa.chem import catalog as _cat


@functools.lru_cache(maxsize=1)
def _potentials_np() -> npt.NDArray[np.float64]:
    r"""Load and cache the `(30, 30)` Allen (1973) table as a NumPy array."""
    return np.asarray(_rust.py_ion_potentials_table(), dtype=np.float64)


@functools.lru_cache(maxsize=1)
def _potentials_jax() -> jax.Array:
    r"""Load and cache the `(30, 30)` Allen (1973) ionization potential table.

    Returns:
        2D `float64` `jax.Array` of shape `(30, 30)` in $\mathrm{eV}$.
    """
    return jnp.asarray(_potentials_np(), dtype=jnp.float64)


def ionization_potential(z: int, stage: int) -> float:
    r"""Return the Allen (1973) ionization potential $I_{Z, k}$ ($\mathrm{eV}$).

    Args:
        z: Atomic number $Z \in [1, 30]$.
        stage: 1-based ionization stage $k \in [1, Z]$.

    Returns:
        Ionization potential $I_{Z, k}$ in $\mathrm{eV}$.

    Raises:
        ValueError: If `z` is outside `[1, 30]` or `stage` is outside `[1, z]`.
    """
    zi, ki = int(z), int(stage)
    if not (1 <= zi <= 30 and 1 <= ki <= zi):
        raise ValueError(
            f"Require 1 <= z <= 30 and 1 <= stage <= z (got z={z}, "
            f"stage={stage})."
        )
    return float(_potentials_np()[zi - 1, ki - 1])


@jax.jit  # type: ignore[untyped-decorator]
def chi_info_jax(
    a1: jax.Array | float,
    z1: jax.Array | float,
    t: jax.Array | float,
    log10_t: jax.Array | float,
    rho: jax.Array | float,
    log10_rho: jax.Array | float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Compute $(\chi, c_0, c_1, c_2)$ in $\mathrm{eV}$ in JAX (`chi_info`).

    Args:
        a1: Mean background nucleon number $\bar{A}$ (dimensionless).
        z1: Mean background charge $\bar{Z} = \bar{A} Y_e$ (dimensionless).
        t: Temperature $T$ in $\mathrm{K}$.
        log10_t: Base-10 log temperature $\log_{10}(T / \mathrm{K})$.
        rho: Mass density $\rho$ in $\mathrm{g\,cm^{-3}}$.
        log10_rho: Base-10 log mass density $\log_{10}(\rho /
            \mathrm{g\,cm^{-3}})$.

    Returns:
        Tuple `(chi, c0, c1, c2)` of `float64` `jax.Array` tensors in
        $\mathrm{eV}$.
    """
    a1_j = jnp.asarray(a1, dtype=jnp.float64)
    z1_j = jnp.asarray(z1, dtype=jnp.float64)
    t_j = jnp.asarray(t, dtype=jnp.float64)
    lt_j = jnp.asarray(log10_t, dtype=jnp.float64)
    rho_j = jnp.asarray(rho, dtype=jnp.float64)
    lrho_j = jnp.asarray(log10_rho, dtype=jnp.float64)
    chi = (
        1.987e-4 * t_j * (-8.392 - lrho_j + 1.5 * lt_j - jnp.log10(z1_j / a1_j))
    )
    c0 = 1.085e-4 * rho_j * t_j / a1_j
    c1 = 1.617e4 * jnp.sqrt(rho_j * (z1_j * z1_j + z1_j) / (t_j * a1_j))
    c2 = 29.38 * z1_j * jnp.power(rho_j / a1_j, 1.0 / 3.0)
    return chi, c0, c1, c2


@jax.jit  # type: ignore[untyped-decorator]
def chi_effective_jax(
    chi: jax.Array | float,
    c0: jax.Array | float,
    c1: jax.Array | float,
    c2: jax.Array | float,
    z1: jax.Array | float,
    z2: jax.Array | float,
) -> jax.Array:
    r"""Compute $\chi_{\mathrm{eff}}$ in $\mathrm{eV}$ in JAX (`chi_effective`).

    Args:
        chi: Base ionization threshold $\chi$ in $\mathrm{eV}$.
        c0: Continuum-depression coefficient $c_0$ in $\mathrm{eV}$.
        c1: Continuum-depression coefficient $c_1$ in $\mathrm{eV}$.
        c2: Continuum-depression coefficient $c_2$ in $\mathrm{eV}$.
        z1: Background charge $z_1$ (dimensionless).
        z2: Trial ion charge $z_2$ (dimensionless).

    Returns:
        Effective ionization threshold $\chi_{\mathrm{eff}}$ in $\mathrm{eV}$.
    """
    chi_j = jnp.asarray(chi, dtype=jnp.float64)
    c0_j = jnp.asarray(c0, dtype=jnp.float64)
    c1_j = jnp.asarray(c1, dtype=jnp.float64)
    c2_j = jnp.asarray(c2, dtype=jnp.float64)
    z1_j = jnp.asarray(z1, dtype=jnp.float64)
    z2_j = jnp.asarray(z2, dtype=jnp.float64)
    dep = jnp.minimum(
        c1_j * z2_j, c2_j * (jnp.power(z2_j / z1_j, 2.0 / 3.0) + 0.6)
    )
    return chi_j + c0_j / (z2_j * z2_j * z2_j) + dep


def chi_info(
    a1: float,
    z1: float,
    t: float,
    log10_t: float,
    rho: float,
    log10_rho: float,
    *,
    backend: _dispatch.Backend = "auto",
) -> tuple[float, float, float, float]:
    r"""Compute $(\chi, c_0, c_1, c_2)$ in $\mathrm{eV}$ (`chi_info`).

    Args:
        a1: Mean background nucleon number $\bar{A}$ (dimensionless).
        z1: Mean background charge $\bar{Z} = \bar{A} Y_e$ (dimensionless).
        t: Temperature $T$ in $\mathrm{K}$.
        log10_t: Base-10 log temperature $\log_{10}(T / \mathrm{K})$.
        rho: Mass density $\rho$ in $\mathrm{g\,cm^{-3}}$.
        log10_rho: Base-10 log mass density $\log_{10}(\rho /
            \mathrm{g\,cm^{-3}})$.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        Tuple `(chi, c0, c1, c2)` in $\mathrm{eV}$.
    """
    resolved = _dispatch.resolve_backend(backend, a1, z1, t, rho)
    if resolved == "jax":
        c, c0, c1, c2 = chi_info_jax(a1, z1, t, log10_t, rho, log10_rho)
        return float(c), float(c0), float(c1), float(c2)
    return _rust.py_ion_chi_info(
        float(a1),
        float(z1),
        float(t),
        float(log10_t),
        float(rho),
        float(log10_rho),
    )


def chi_effective(
    chi: float,
    c0: float,
    c1: float,
    c2: float,
    z1: float,
    z2: float,
    *,
    backend: _dispatch.Backend = "auto",
) -> float:
    r"""Compute $\chi_{\mathrm{eff}}$ in $\mathrm{eV}$ (`chi_effective`).

    Args:
        chi: Base threshold $\chi$ in $\mathrm{eV}$.
        c0: Coefficient $c_0$ in $\mathrm{eV}$.
        c1: Coefficient $c_1$ in $\mathrm{eV}$.
        c2: Coefficient $c_2$ in $\mathrm{eV}$.
        z1: Background charge $z_1$ (dimensionless).
        z2: Trial ion charge $z_2$ (dimensionless).
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        Effective ionization threshold $\chi_{\mathrm{eff}}$ in $\mathrm{eV}$.
    """
    resolved = _dispatch.resolve_backend(backend, chi, c0, c1, c2, z1, z2)
    if resolved == "jax":
        return float(chi_effective_jax(chi, c0, c1, c2, z1, z2))
    return float(
        _rust.py_ion_chi_effective(
            float(chi),
            float(c0),
            float(c1),
            float(c2),
            float(z1),
            float(z2),
        )
    )


def eval_typical_charge(
    cid: int | str,
    abar: float,
    free_e: float,
    t: float,
    log10_t: float,
    rho: float,
    log10_rho: float,
    *,
    backend: _dispatch.Backend = "auto",
) -> float:
    r"""Compute the whole-number typical ionic charge (`eval_typical_charge`).

    Args:
        cid: 1-based `chem_id` or nuclide name (e.g. ``"c12"``, ``"fe56"``).
        abar: Mean background nucleon number $\bar{A}$ (dimensionless).
        free_e: Free electrons per nucleon $Y_e$ (dimensionless).
        t: Temperature $T$ in $\mathrm{K}$.
        log10_t: Base-10 log temperature $\log_{10}(T / \mathrm{K})$.
        rho: Mass density $\rho$ in $\mathrm{g\,cm^{-3}}$.
        log10_rho: Base-10 log mass density $\log_{10}(\rho /
            \mathrm{g\,cm^{-3}})$.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        Whole-number ionic charge $Z_{\mathrm{typ}} \in [1, Z_{\max}]$.

    Raises:
        ValueError: If `cid` is not a recognized nuclide.
    """
    resolved_cid = (
        _cat.get_nuclide_index(cid) if isinstance(cid, str) else int(cid)
    )
    resolved = _dispatch.resolve_backend(backend, abar, free_e, t, rho)
    if resolved == "jax":
        info = _cat.get_nuclide_info(resolved_cid)
        if info.z <= 1 or info.z > 30:
            return float(info.z)
        z1 = jnp.asarray(abar, dtype=jnp.float64) * jnp.asarray(
            free_e, dtype=jnp.float64
        )
        chi, c0, c1, c2 = chi_info_jax(abar, z1, t, log10_t, rho, log10_rho)
        stages = jnp.arange(1, info.z, dtype=jnp.float64)
        ceff = chi_effective_jax(chi, c0, c1, c2, z1, stages + 1.0)
        row = _potentials_jax()[info.z - 1, 1 : info.z]
        mask = ceff < row
        z_out = jnp.where(
            jnp.any(mask), stages[jnp.argmax(mask)], float(info.z)
        )
        return float(z_out)
    return float(
        _rust.py_ion_eval_typical_charge(
            resolved_cid,
            float(abar),
            float(free_e),
            float(t),
            float(log10_t),
            float(rho),
            float(log10_rho),
        )
    )
