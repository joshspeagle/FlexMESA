# SPDX-License-Identifier: LGPL-3.0-or-later
"""Potekhin-Chabrier (`PC`) electron-ion plasma EOS (`eospc_eval.f90`).

Provides the Tier B dual-track (`backend="auto" | "rust" | "jax"`) Python and
JAX interfaces for `get_pc_alfa`, `eval_pc_eos`, and `eval_pc_eos_zones`.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import _dispatch, _rust, chem
from flexmesa.eos import _pc_jax, core

eval_pc_eos_jax = _pc_jax.eval_pc_eos_jax
excor7_jax = _pc_jax.excor7_jax
fharm12_anharm8_jax = _pc_jax.fharm12_anharm8_jax
fition9_jax = _pc_jax.fition9_jax
get_pc_alfa_jax = _pc_jax.get_pc_alfa_jax


def get_pc_alfa(
    log_rho: float | jax.Array,
    log_t: float | jax.Array,
    abar: float | jax.Array,
    zbar: float | jax.Array,
    *,
    use_gamma_limit: bool = True,
    log_rho1: float = 2.999,
    log_rho2: float = 2.8,
    log_t1: float = 7.6,
    log_t2: float = 7.7,
    log_gamma_e_all_helm: float = -1.8,
    log_gamma_e_all_pc: float = -1.7,
    backend: _dispatch.Backend = "auto",
) -> tuple[float | jax.Array, float | jax.Array, float | jax.Array]:
    """Evaluate `Get_PC_alfa` (`eospc_eval.f90` lines 31-96).

    Args:
        log_rho: Base-10 log mass density $\\log_{10} \\rho$ in `g cm^-3`.
        log_t: Base-10 log temperature $\\log_{10} T$ in `K`.
        abar: Mean atomic weight $\\bar{A}$.
        zbar: Mean atomic charge $\\bar{Z}$.
        use_gamma_limit: Whether to use $\\log_{10} \\Gamma_e$ thresholds.
        log_rho1: Upper $\\log_{10} \\rho$ threshold for pure `PC`.
        log_rho2: Lower $\\log_{10} \\rho$ threshold below which `PC` is zero.
        log_t1: Lower $\\log_{10} T$ threshold when `T` limits are used.
        log_t2: Upper $\\log_{10} T$ threshold when `T` limits are used.
        log_gamma_e_all_helm: $\\log_{10} \\Gamma_e$ threshold for pure `HELM`.
        log_gamma_e_all_pc: $\\log_{10} \\Gamma_e$ threshold for pure `PC`.
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Tuple `(alfa, d_alfa_dlogT, d_alfa_dlogRho)`.
    """
    resolved = _dispatch.resolve_backend(backend, log_rho, log_t, abar, zbar)
    if resolved == "jax":
        return get_pc_alfa_jax(
            jnp.asarray(log_rho, dtype=jnp.float64),
            jnp.asarray(log_t, dtype=jnp.float64),
            jnp.asarray(abar, dtype=jnp.float64),
            jnp.asarray(zbar, dtype=jnp.float64),
            use_gamma_limit,
            log_rho1,
            log_rho2,
            log_t1,
            log_t2,
            log_gamma_e_all_helm,
            log_gamma_e_all_pc,
        )
    return _rust.eos_get_pc_alfa(
        float(log_rho),
        float(log_t),
        float(abar),
        float(zbar),
        bool(use_gamma_limit),
        float(log_rho1),
        float(log_rho2),
        float(log_t1),
        float(log_t2),
        float(log_gamma_e_all_helm),
        float(log_gamma_e_all_pc),
    )


def eval_pc_eos(
    rho: float | jax.Array,
    temp: float | jax.Array,
    abar: float | jax.Array,
    zbar: float | jax.Array,
    chem_id: Sequence[int],
    xa: np.ndarray | jax.Array,
    *,
    include_radiation: bool = True,
    pc_gamma_start_crystal: float = 170.0,
    pc_gamma_full_crystal: float = 180.0,
    mass_fraction_limit: float = 1.0e-4,
    backend: _dispatch.Backend = "auto",
) -> core.EosEvalOutput:
    """Evaluate the 26-channel Potekhin-Chabrier (`PC`) equation of state.

    Args:
        rho: Mass density $\\rho$ in `g cm^-3` (scalar).
        temp: Temperature $T$ in `K` (scalar).
        abar: Mean atomic weight $\\bar{A}$.
        zbar: Mean atomic charge $\\bar{Z}$.
        chem_id: 1-based MESA nuclide identifiers of length `N_species`.
        xa: Species mass fractions of shape `(N_species,)`.
        include_radiation: Whether to include photon radiation pressure/energy.
        pc_gamma_start_crystal: $\\Gamma_i$ onset of crystallization.
        pc_gamma_full_crystal: $\\Gamma_i$ completion of crystallization.
        mass_fraction_limit: Minimum species mass fraction included in `PC`.
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        [`EosEvalOutput`] with `res`, `d_dlnd`, and `d_dln_t` of shape `(26,)`.
    """
    resolved = _dispatch.resolve_backend(backend, rho, temp, abar, zbar, xa)
    if resolved == "jax":
        azion = jnp.asarray(
            [float(chem.get_nuclide_info(int(cid)).z) for cid in chem_id],
            dtype=jnp.float64,
        )
        acmi = jnp.asarray(
            [float(chem.get_nuclide_info(int(cid)).w) for cid in chem_id],
            dtype=jnp.float64,
        )
        res, d_dlnd, d_dln_t = eval_pc_eos_jax(
            jnp.asarray(rho, dtype=jnp.float64),
            jnp.asarray(temp, dtype=jnp.float64),
            jnp.asarray(abar, dtype=jnp.float64),
            jnp.asarray(zbar, dtype=jnp.float64),
            jnp.asarray(xa, dtype=jnp.float64),
            azion,
            acmi,
            include_radiation=include_radiation,
            pc_gamma_start_crystal=pc_gamma_start_crystal,
            pc_gamma_full_crystal=pc_gamma_full_crystal,
            mass_fraction_limit=mass_fraction_limit,
        )
        return core.EosEvalOutput(res=res, d_dlnd=d_dlnd, d_dln_t=d_dln_t)
    res_np, dlnd_np, dlnt_np = _rust.eos_eval_pc(
        float(rho),
        float(temp),
        float(abar),
        float(zbar),
        [int(c) for c in chem_id],
        np.ascontiguousarray(xa, dtype=np.float64),
        bool(include_radiation),
        float(pc_gamma_start_crystal),
        float(pc_gamma_full_crystal),
        float(mass_fraction_limit),
    )
    return core.EosEvalOutput(res=res_np, d_dlnd=dlnd_np, d_dln_t=dlnt_np)


def eval_pc_eos_zones(
    rho: np.ndarray | jax.Array,
    temp: np.ndarray | jax.Array,
    abar: np.ndarray | jax.Array,
    zbar: np.ndarray | jax.Array,
    xa: np.ndarray | jax.Array,
    azion: np.ndarray | jax.Array,
    acmi: np.ndarray | jax.Array,
    *,
    include_radiation: bool = True,
    pc_gamma_start_crystal: float = 170.0,
    pc_gamma_full_crystal: float = 180.0,
    mass_fraction_limit: float = 1.0e-4,
    parallel: bool = True,
    backend: _dispatch.Backend = "auto",
) -> core.EosEvalOutput:
    """Evaluate the 26-channel `PC` EOS across `N_zones` stellar zones.

    Args:
        rho: Zone mass densities $\\rho$ in `g cm^-3` of shape `(N_zones,)`.
        temp: Zone temperatures $T$ in `K` of shape `(N_zones,)`.
        abar: Zone mean atomic weights $\\bar{A}$ of shape `(N_zones,)`.
        zbar: Zone mean atomic charges $\\bar{Z}$ of shape `(N_zones,)`.
        xa: Species mass fractions of shape `(N_zones, N_species)`.
        azion: Species atomic charges $Z_i$ of shape `(N_species,)`.
        acmi: Species atomic weights $A_i$ in `amu` of shape `(N_species,)`.
        include_radiation: Whether to include photon radiation pressure/energy.
        pc_gamma_start_crystal: $\\Gamma_i$ onset of crystallization.
        pc_gamma_full_crystal: $\\Gamma_i$ completion of crystallization.
        mass_fraction_limit: Minimum species mass fraction included in `PC`.
        parallel: Whether to use the Rayon worker pool on `"rust"`.
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        [`EosEvalOutput`] with `res`, `d_dlnd`, and `d_dln_t` of shape
        `(N_zones, 26)`.
    """
    resolved = _dispatch.resolve_backend(backend, rho, temp, abar, zbar, xa)
    if resolved == "jax":
        res, d_dlnd, d_dln_t = eval_pc_eos_jax(
            jnp.asarray(rho, dtype=jnp.float64),
            jnp.asarray(temp, dtype=jnp.float64),
            jnp.asarray(abar, dtype=jnp.float64),
            jnp.asarray(zbar, dtype=jnp.float64),
            jnp.asarray(xa, dtype=jnp.float64),
            jnp.asarray(azion, dtype=jnp.float64),
            jnp.asarray(acmi, dtype=jnp.float64),
            include_radiation=include_radiation,
            pc_gamma_start_crystal=pc_gamma_start_crystal,
            pc_gamma_full_crystal=pc_gamma_full_crystal,
            mass_fraction_limit=mass_fraction_limit,
        )
        return core.EosEvalOutput(res=res, d_dlnd=d_dlnd, d_dln_t=d_dln_t)
    out = _rust.eos_eval_pc_zones(
        np.ascontiguousarray(rho, dtype=np.float64),
        np.ascontiguousarray(temp, dtype=np.float64),
        np.ascontiguousarray(abar, dtype=np.float64),
        np.ascontiguousarray(zbar, dtype=np.float64),
        np.ascontiguousarray(np.asarray(xa, dtype=np.float64).reshape(-1)),
        np.ascontiguousarray(azion, dtype=np.float64),
        np.ascontiguousarray(acmi, dtype=np.float64),
        bool(include_radiation),
        float(pc_gamma_start_crystal),
        float(pc_gamma_full_crystal),
        float(mass_fraction_limit),
        bool(parallel),
    )
    n = core.NUM_EOS_BASIC_RESULTS
    return core.EosEvalOutput(
        res=out[:, :n],
        d_dlnd=out[:, n : 2 * n],
        d_dln_t=out[:, 2 * n : 3 * n],
    )
