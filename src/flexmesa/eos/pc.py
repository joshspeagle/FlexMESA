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

from flexmesa import _dispatch, _rust, constants
from flexmesa.eos import _pc_jax, core

excor7_jax = _pc_jax.excor7_jax
fharm12_anharm8_jax = _pc_jax.fharm12_anharm8_jax
fition9_jax = _pc_jax.fition9_jax


def _eval_pc_ffi_core(
    inc_rad: bool,
    g_start: float,
    g_full: float,
    mf_lim: float,
    r_in: jax.Array,
    t_in: jax.Array,
    a_in: jax.Array,
    z_in: jax.Array,
    x_in: jax.Array,
    az_in: jax.Array,
    ac_in: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Explicit `solver_mode='rust_ffi'` host callback for `PC` EOS."""

    def _host_eval_pc(
        rho_np: np.ndarray,
        temp_np: np.ndarray,
        abar_np: np.ndarray,
        zbar_np: np.ndarray,
        xa_np: np.ndarray,
        azion_np: np.ndarray,
        acmi_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        orig_shape = np.shape(rho_np)
        r_1d = np.ascontiguousarray(np.asarray(rho_np, np.float64).reshape(-1))
        t_1d = np.ascontiguousarray(np.asarray(temp_np, np.float64).reshape(-1))
        nz = r_1d.size
        a_1d = np.ascontiguousarray(
            np.broadcast_to(np.asarray(abar_np, np.float64), orig_shape).ravel()
        )
        z_1d = np.ascontiguousarray(
            np.broadcast_to(np.asarray(zbar_np, np.float64), orig_shape).ravel()
        )
        az_1d = np.ascontiguousarray(np.asarray(azion_np, np.float64).ravel())
        ac_1d = np.ascontiguousarray(np.asarray(acmi_np, np.float64).ravel())
        xa_1d = np.ascontiguousarray(
            np.broadcast_to(
                np.asarray(xa_np, np.float64), (*orig_shape, az_1d.size)
            ).ravel()
        )
        args = (inc_rad, g_start, g_full, mf_lim, True)
        out = _rust.eos_eval_pc_zones(
            r_1d, t_1d, a_1d[:nz], z_1d[:nz], xa_1d, az_1d, ac_1d, *args
        )
        n = core.NUM_EOS_BASIC_RESULTS
        return (
            out[:, :n].reshape((*orig_shape, n)),
            out[:, n : 2 * n].reshape((*orig_shape, n)),
            out[:, 2 * n : 3 * n].reshape((*orig_shape, n)),
        )

    shape = (*r_in.shape, core.NUM_EOS_BASIC_RESULTS)
    vc = jax.ShapeDtypeStruct(shape, jnp.float64)
    cb_args = (r_in, t_in, a_in, z_in, x_in, az_in, ac_in)
    r_c, d_c, t_c = jax.pure_callback(
        _host_eval_pc, (vc, vc, vc), *cb_args, vmap_method="expand_dims"
    )
    return r_c, d_c, t_c


_eval_pc_ffi_jvp_fn = jax.custom_jvp(
    _eval_pc_ffi_core, nondiff_argnums=(0, 1, 2, 3)
)


@_eval_pc_ffi_jvp_fn.defjvp
def _eval_pc_ffi_jvp(
    inc_rad: bool,
    g_start: float,
    g_full: float,
    mf_lim: float,
    primals: tuple[jax.Array, ...],
    tangents: tuple[jax.Array, ...],
) -> tuple[
    tuple[jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array, jax.Array],
]:
    res_0, dlnd_0, dlnt_0 = _eval_pc_ffi_jvp_fn(
        inc_rad, g_start, g_full, mf_lim, *primals
    )
    return core.pack_dt_jvp(
        primals[0],
        primals[1],
        tangents[0],
        tangents[1],
        (res_0, dlnd_0, dlnt_0),
    )


def eval_pc_eos_jax(
    rho: jax.Array,
    temp: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    xa: jax.Array,
    azion: jax.Array,
    acmi: jax.Array,
    *,
    include_radiation: bool = True,
    pc_gamma_start_crystal: float = 170.0,
    pc_gamma_full_crystal: float = 180.0,
    mass_fraction_limit: float = 1.0e-4,
    solver_mode: str = "jax",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate 26-channel `PC` EOS in JAX (`"jax"` native or `"rust_ffi"`)."""
    if solver_mode == "rust_ffi":
        f64 = jnp.float64
        sg = jax.lax.stop_gradient
        res_f, dlnd_f, dlnt_f = _eval_pc_ffi_jvp_fn(
            bool(include_radiation),
            float(pc_gamma_start_crystal),
            float(pc_gamma_full_crystal),
            float(mass_fraction_limit),
            jnp.asarray(rho, f64),
            jnp.asarray(temp, f64),
            sg(jnp.asarray(abar, f64)),
            sg(jnp.asarray(zbar, f64)),
            sg(jnp.asarray(xa, f64)),
            sg(jnp.asarray(azion, f64)).reshape(-1),
            sg(jnp.asarray(acmi, f64)).reshape(-1),
        )
        return res_f, dlnd_f, dlnt_f
    return _pc_jax.eval_pc_eos_jax(
        rho,
        temp,
        abar,
        zbar,
        xa,
        azion,
        acmi,
        include_radiation=include_radiation,
        pc_gamma_start_crystal=pc_gamma_start_crystal,
        pc_gamma_full_crystal=pc_gamma_full_crystal,
        mass_fraction_limit=mass_fraction_limit,
    )


@jax.jit
def get_pc_alfa_jax(
    log_rho: jax.Array,
    log_t: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
    use_gamma_limit: bool = True,
    log_rho1: float = 2.999,
    log_rho2: float = 2.8,
    log_t1: float = 7.6,
    log_t2: float = 7.7,
    log_gamma_e_all_helm: float = -1.8,
    log_gamma_e_all_pc: float = -1.7,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate `Get_PC_alfa` (`eospc_eval.f90` lines 31-96) in JAX."""
    log_rho = jnp.asarray(log_rho, dtype=jnp.float64)
    log_t = jnp.asarray(log_t, dtype=jnp.float64)
    abar = jnp.asarray(abar, dtype=jnp.float64)
    zbar = jnp.asarray(zbar, dtype=jnp.float64)
    log_rho_lo0 = jnp.asarray(log_rho2, dtype=jnp.float64)
    log_rho_hi = jnp.asarray(log_rho1, dtype=jnp.float64)
    log_t1_f = jnp.asarray(log_t1, dtype=jnp.float64)
    log_t2_f = jnp.asarray(log_t2, dtype=jnp.float64)

    log_ge0 = jnp.log10(
        constants.QE
        * constants.QE
        * jnp.power(
            constants.FOUR_THIRDS_PI * constants.AVO * zbar / abar,
            constants.ONE_THIRD,
        )
        / constants.KERG
    )
    log_ge = log_ge0 + log_rho / 3.0 - log_t
    log_ge_lo = jnp.asarray(log_gamma_e_all_helm, dtype=jnp.float64)
    log_ge_hi = jnp.asarray(log_gamma_e_all_pc, dtype=jnp.float64)
    log_t_lo_g = log_ge0 + log_rho_lo0 / 3.0 - log_ge_lo
    log_t_hi_g = log_ge0 + log_rho_hi / 3.0 - log_ge_hi

    span_ge = log_ge_hi - log_ge_lo
    alfa_ge = (log_ge_hi - log_ge) / span_ge
    dt_ge = jnp.ones_like(log_rho) / span_ge
    dr_ge = -(1.0 / 3.0) * jnp.ones_like(log_rho) / span_ge

    log_rho_lo_eff = jnp.where(
        log_t >= log_t_lo_g,
        (log_t_lo_g + log_ge_lo - log_ge0) * 3.0,
        log_rho_lo0,
    )
    span_r = log_rho_hi - log_rho_lo_eff
    alfa_r = (log_rho_hi - log_rho) / span_r
    dr_r = -1.0 / span_r

    out_outside_g = (log_rho <= log_rho_lo0) | (log_ge <= log_ge_lo)
    out_inside_g = (log_rho >= log_rho_hi) & (log_ge >= log_ge_hi)
    use_ge_branch = log_t >= log_t_hi_g

    alfa_g = jnp.where(
        out_outside_g,
        1.0,
        jnp.where(out_inside_g, 0.0, jnp.where(use_ge_branch, alfa_ge, alfa_r)),
    )
    dt_g = jnp.where(
        out_outside_g | out_inside_g,
        0.0,
        jnp.where(use_ge_branch, dt_ge, 0.0),
    )
    dr_g = jnp.where(
        out_outside_g | out_inside_g,
        0.0,
        jnp.where(use_ge_branch, dr_ge, dr_r),
    )

    span_t = log_t2_f - log_t1_f
    alfa_t_br = (log_t - log_t1_f) / span_t
    dt_t_br = jnp.ones_like(log_rho) / span_t
    span_r0 = log_rho_hi - log_rho_lo0
    alfa_r0 = (log_rho_hi - log_rho) / span_r0
    dr_r0 = -jnp.ones_like(log_rho) / span_r0

    out_outside_t = (log_rho <= log_rho_lo0) | (log_t >= log_t2_f)
    out_inside_t = (log_rho >= log_rho_hi) & (log_t <= log_t1_f)
    use_t_branch = log_t >= log_t1_f

    alfa_t = jnp.where(
        out_outside_t,
        1.0,
        jnp.where(
            out_inside_t, 0.0, jnp.where(use_t_branch, alfa_t_br, alfa_r0)
        ),
    )
    dt_t = jnp.where(
        out_outside_t | out_inside_t,
        0.0,
        jnp.where(use_t_branch, dt_t_br, 0.0),
    )
    dr_t = jnp.where(
        out_outside_t | out_inside_t,
        0.0,
        jnp.where(use_t_branch, 0.0, dr_r0),
    )

    use_g = jnp.asarray(use_gamma_limit)
    return (
        jnp.where(use_g, alfa_g, alfa_t),
        jnp.where(use_g, dt_g, dt_t),
        jnp.where(use_g, dr_g, dr_t),
    )


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
        a_j, dt_j, dr_j = get_pc_alfa_jax(
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
        return a_j, dt_j, dr_j
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
    solver_mode: str = "jax",
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
        solver_mode: JAX evaluation mode (`"jax"` native or `"rust_ffi"`).
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        [`EosEvalOutput`] with `res`, `d_dlnd`, and `d_dln_t` of shape `(26,)`.
    """
    resolved = _dispatch.resolve_backend(backend, rho, temp, abar, zbar, xa)
    if resolved == "jax":
        azion, acmi, _ = core.nuclide_arrays_jax(chem_id)
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
            solver_mode=solver_mode,
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
    solver_mode: str = "jax",
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
        solver_mode: JAX evaluation mode (`"jax"` native or `"rust_ffi"`).
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
            solver_mode=solver_mode,
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
