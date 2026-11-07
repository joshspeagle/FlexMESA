# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Top-level `eosDT_get` 7-level cascade, 4D table evaluators, and `PT`/`DE`
inverse solvers (`eos/public`).

Provides the Tier B dual-track (`backend="auto" | "rust" | "jax"`) Python and
JAX interfaces for [`eos_dt_get`], [`eos_pt_get`], [`eval_opal_scvh_eos`],
[`eval_free_eos`], [`eval_cms_eos`], [`eos_dt_get_t`], [`eos_dt_get_rho`],
[`eos_pt_get_t`], [`eos_pt_get_pgas`], and [`eos_gamma_de_get`].
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import _dispatch, _rust
from flexmesa.eos import core, skye


@dataclasses.dataclass(frozen=True)
class EosPtOutput:
    """Result of an inverse `(P_gas, T)` EOS evaluation (`eosPT_get`).

    Attributes:
        rho: Converged mass density $\\rho$ in `g cm^-3`.
        log10_rho: Converged $\\log_{10}(\\rho / \\mathrm{g\\,cm^{-3}})$.
        dln_rho_dln_pgas_const_t: Implicit derivative
            $(\\partial \\ln \\rho / \\partial \\ln P_{\\mathrm{gas}})_T$.
        dln_rho_dln_t_const_pgas: Implicit derivative
            $(\\partial \\ln \\rho / \\partial \\ln T)_{P_{\\mathrm{gas}}}$.
        eos: 26-channel [`EosEvalOutput`] evaluated at `(rho, T)`.
    """

    rho: float | jax.Array
    log10_rho: float | jax.Array
    dln_rho_dln_pgas_const_t: float | jax.Array
    dln_rho_dln_t_const_pgas: float | jax.Array
    eos: core.EosEvalOutput


@dataclasses.dataclass(frozen=True)
class EosInverseOutput:
    """Result of an inverse EOS root-find (`eosDT_get_*` / `eosPT_get_*`).

    Attributes:
        result: Solved linear quantity ($T$ in `K`, $\\rho$ in `g cm^-3`, or
            $P_{\\mathrm{gas}}$ in `dyn cm^-2`).
        log10_result: Solved base-10 logarithm.
        eos: 26-channel [`EosEvalOutput`] evaluated at the converged state.
    """

    result: float | jax.Array
    log10_result: float | jax.Array
    eos: core.EosEvalOutput


def _pack_pt(vals: tuple[Any, ...]) -> EosPtOutput:
    """Pack a 7-tuple from `eos_eval_pt_get` into `EosPtOutput`."""
    r, lr, dp, dt, res, dlnd, dlnt = vals
    return EosPtOutput(r, lr, dp, dt, core.EosEvalOutput(res, dlnd, dlnt))


def _pack_inv(vals: tuple[Any, ...]) -> EosInverseOutput:
    """Pack a 5-tuple from Rust inverse solvers into `EosInverseOutput`."""
    v, lv, res, dlnd, dlnt = vals
    return EosInverseOutput(v, lv, core.EosEvalOutput(res, dlnd, dlnt))


def _prep(
    cid: Sequence[int],
    xa: Any,
    s: bool,
    p: bool,
    g: bool,
    r: str,
    c: bool,
    f: bool,
    o: bool,
) -> tuple[list[int], np.ndarray, bool, bool, bool, str, bool, bool, bool]:
    """Normalize composition and control flags for Rust EOS FFI calls."""
    xa_c = np.ascontiguousarray(xa, dtype=np.float64)
    return (
        [int(x) for x in cid],
        xa_c,
        bool(s),
        bool(p),
        bool(g),
        str(r),
        bool(c),
        bool(f),
        bool(o),
    )


def eos_dt_get_jax(
    rho: jax.Array,
    temp: jax.Array,
    chem_id: Sequence[int],
    xa: jax.Array,
    *,
    use_skye: bool = True,
    use_pc: bool = False,
    pc_use_gamma_limit: bool = True,
    skye_solid_mixing_rule: str = "PC",
    use_cms: bool = False,
    use_free_eos: bool = True,
    use_opal_scvh: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate `eosDT_get` in JAX with exact `custom_jvp` derivative rules."""
    f_args = _prep(
        chem_id,
        xa,
        use_skye,
        use_pc,
        pc_use_gamma_limit,
        skye_solid_mixing_rule,
        use_cms,
        use_free_eos,
        use_opal_scvh,
    )

    def _host_cb(
        r_a: np.ndarray, t_a: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        res_np, dlnd_np, dlnt_np = _rust.eos_eval_dt_get(
            float(r_a), float(t_a), *f_args
        )
        return (
            np.asarray(res_np, dtype=np.float64),
            np.asarray(dlnd_np, dtype=np.float64),
            np.asarray(dlnt_np, dtype=np.float64),
        )

    return skye.eval_host_dt_jvp(rho, temp, _host_cb)


def eos_dt_get(
    rho: float | jax.Array,
    temp: float | jax.Array,
    chem_id: Sequence[int],
    xa: np.ndarray | jax.Array,
    *,
    use_skye: bool = True,
    use_pc: bool = False,
    pc_use_gamma_limit: bool = True,
    skye_solid_mixing_rule: str = "PC",
    use_cms: bool = False,
    use_free_eos: bool = True,
    use_opal_scvh: bool = True,
    backend: _dispatch.Backend = "auto",
) -> core.EosEvalOutput:
    """Evaluate the unified 7-level `eosDT_get` cascade (`eosdt_eval.f90`).

    Args:
        rho: Mass density $\\rho$ in `g cm^-3` (scalar).
        temp: Temperature $T$ in `K` (scalar).
        chem_id: 1-based MESA nuclide identifiers of length `N_species`.
        xa: Species mass fractions of shape `(N_species,)`.
        use_skye: Whether `Skye` is enabled.
        use_pc: Whether `PC` is enabled when `Skye` is disabled.
        pc_use_gamma_limit: Whether `PC` uses $\\log_{10} \\Gamma_e$ limits.
        skye_solid_mixing_rule: Solid mixing rule (`"PC"` or `"Ogata"`).
        use_cms: Whether `CMS` H/He EOS is enabled.
        use_free_eos: Whether `FreeEOS` 4D tables are enabled.
        use_opal_scvh: Whether `OPAL/SCVH` 4D tables are enabled.
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        [`EosEvalOutput`] with `res`, `d_dlnd`, and `d_dln_t` of shape `(26,)`.
    """
    if _dispatch.resolve_backend(backend, rho, temp, xa) == "jax":
        res, d_dlnd, d_dln_t = eos_dt_get_jax(
            jnp.asarray(rho, np.float64),
            jnp.asarray(temp, np.float64),
            chem_id,
            jnp.asarray(xa, np.float64),
            use_skye=use_skye,
            use_pc=use_pc,
            pc_use_gamma_limit=pc_use_gamma_limit,
            skye_solid_mixing_rule=skye_solid_mixing_rule,
            use_cms=use_cms,
            use_free_eos=use_free_eos,
            use_opal_scvh=use_opal_scvh,
        )
        return core.EosEvalOutput(res, d_dlnd, d_dln_t)
    f_args = _prep(
        chem_id,
        xa,
        use_skye,
        use_pc,
        pc_use_gamma_limit,
        skye_solid_mixing_rule,
        use_cms,
        use_free_eos,
        use_opal_scvh,
    )
    return core.EosEvalOutput(
        *_rust.eos_eval_dt_get(float(rho), float(temp), *f_args)
    )


def _eval_4d_table_dt(
    fn: Any,
    rho: float | jax.Array,
    temp: float | jax.Array,
    chem_id: Sequence[int],
    xa: np.ndarray | jax.Array,
    backend: _dispatch.Backend,
) -> core.EosEvalOutput:
    """Shared dual-track wrapper for standalone 4D EOS table evaluators."""
    cids = [int(c) for c in chem_id]
    xa_c = np.ascontiguousarray(xa, dtype=np.float64)
    if _dispatch.resolve_backend(backend, rho, temp, xa) == "jax":

        def _host_cb(
            r_a: np.ndarray, t_a: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            res_np, dlnd_np, dlnt_np = fn(float(r_a), float(t_a), cids, xa_c)
            return (
                np.asarray(res_np, dtype=np.float64),
                np.asarray(dlnd_np, dtype=np.float64),
                np.asarray(dlnt_np, dtype=np.float64),
            )

        res, d_dlnd, d_dln_t = skye.eval_host_dt_jvp(
            jnp.asarray(rho, np.float64),
            jnp.asarray(temp, np.float64),
            _host_cb,
        )
        return core.EosEvalOutput(res, d_dlnd, d_dln_t)
    return core.EosEvalOutput(*fn(float(rho), float(temp), cids, xa_c))


def eval_opal_scvh_eos(
    rho: float | jax.Array,
    temp: float | jax.Array,
    chem_id: Sequence[int],
    xa: np.ndarray | jax.Array,
    *,
    backend: _dispatch.Backend = "auto",
) -> core.EosEvalOutput:
    """Evaluate `OPAL/SCVH` 4D tabulated EOS standalone at `(rho, T)`."""
    return _eval_4d_table_dt(
        _rust.eos_eval_opal_scvh, rho, temp, chem_id, xa, backend
    )


def eval_free_eos(
    rho: float | jax.Array,
    temp: float | jax.Array,
    chem_id: Sequence[int],
    xa: np.ndarray | jax.Array,
    *,
    backend: _dispatch.Backend = "auto",
) -> core.EosEvalOutput:
    """Evaluate `FreeEOS` 4D tabulated EOS standalone at `(rho, T)`."""
    return _eval_4d_table_dt(
        _rust.eos_eval_free_eos, rho, temp, chem_id, xa, backend
    )


def eval_cms_eos(
    rho: float | jax.Array,
    temp: float | jax.Array,
    chem_id: Sequence[int],
    xa: np.ndarray | jax.Array,
    *,
    backend: _dispatch.Backend = "auto",
) -> core.EosEvalOutput:
    """Evaluate Chabrier, Mazevet & Soubiran (`CMS`) EOS standalone."""
    return _eval_4d_table_dt(
        _rust.eos_eval_cms, rho, temp, chem_id, xa, backend
    )


#: Special `which_other` code for $E_{\mathrm{gas}} = E - E_{\mathrm{rad}}$.
WHICH_OTHER_EGAS: int = -1
#: Special `which_other` code for $\log_{10} P_{\mathrm{total}}$.
WHICH_OTHER_LOG_PTOT: int = -2

_SC = jax.ShapeDtypeStruct((), jnp.float64)
_VC = jax.ShapeDtypeStruct((core.NUM_EOS_BASIC_RESULTS,), jnp.float64)
_PT_SPEC = (_SC, _SC, _SC, _SC, _VC, _VC, _VC)


def eos_pt_get_jax(
    pgas: jax.Array,
    temp: jax.Array,
    chem_id: Sequence[int],
    xa: jax.Array,
    *,
    use_skye: bool = True,
    use_pc: bool = False,
    pc_use_gamma_limit: bool = True,
    skye_solid_mixing_rule: str = "PC",
    use_cms: bool = False,
    use_free_eos: bool = True,
    use_opal_scvh: bool = True,
) -> EosPtOutput:
    """Evaluate `eosPT_get` in JAX with Implicit Function Theorem JVP."""
    f_args = _prep(
        chem_id,
        xa,
        use_skye,
        use_pc,
        pc_use_gamma_limit,
        skye_solid_mixing_rule,
        use_cms,
        use_free_eos,
        use_opal_scvh,
    )

    def _host_cb(p_a: np.ndarray, t_a: np.ndarray) -> tuple[np.ndarray, ...]:
        vals = _rust.eos_eval_pt_get(float(p_a), float(t_a), *f_args)
        return tuple(np.asarray(v, dtype=np.float64) for v in vals)

    def _eval_core(p_in: jax.Array, t_in: jax.Array) -> tuple[jax.Array, ...]:
        res: tuple[jax.Array, ...] = jax.pure_callback(
            _host_cb, _PT_SPEC, p_in, t_in
        )
        return res

    _eval = jax.custom_jvp(_eval_core)

    def _eval_jvp(
        primals: tuple[jax.Array, jax.Array],
        tangents: tuple[jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
        p_p, t_p = primals
        dp, dt = tangents
        rho_v, lrho_v, d_dlpg, d_dlt, res, d_dlnd, d_dln_t = _eval_core(
            p_p, t_p
        )
        dln_p, dln_t = dp / p_p, dt / t_p
        dln_rho = (dln_p - d_dln_t[core.I_LN_PGAS] * dln_t) / d_dlnd[
            core.I_LN_PGAS
        ]
        drho, dlrho = rho_v * dln_rho, dln_rho / jnp.log(10.0)
        dres = d_dlnd * dln_rho + d_dln_t * dln_t
        zs, zv = jnp.zeros_like(rho_v), jnp.zeros_like(res)
        return (
            (rho_v, lrho_v, d_dlpg, d_dlt, res, d_dlnd, d_dln_t),
            (drho, dlrho, zs, zs, dres, zv, zv),
        )

    _eval.defjvp(_eval_jvp)
    p_f, t_f = jnp.asarray(pgas, np.float64), jnp.asarray(temp, np.float64)
    return _pack_pt(_eval(p_f, t_f))


def eos_pt_get(
    pgas: float | jax.Array,
    temp: float | jax.Array,
    chem_id: Sequence[int],
    xa: np.ndarray | jax.Array,
    *,
    use_skye: bool = True,
    use_pc: bool = False,
    pc_use_gamma_limit: bool = True,
    skye_solid_mixing_rule: str = "PC",
    use_cms: bool = False,
    use_free_eos: bool = True,
    use_opal_scvh: bool = True,
    backend: _dispatch.Backend = "auto",
) -> EosPtOutput:
    """Evaluate the inverse `(P_gas, T)` solver `eosPT_get` (`eospt_eval.f90`).

    Args:
        pgas: Gas pressure $P_{\\mathrm{gas}}$ in `dyn cm^-2`.
        temp: Temperature $T$ in `K`.
        chem_id: 1-based MESA nuclide identifiers of length `N_species`.
        xa: Species mass fractions of shape `(N_species,)`.
        use_skye: Whether `Skye` is enabled.
        use_pc: Whether `PC` is enabled when `Skye` is disabled.
        pc_use_gamma_limit: Whether `PC` uses $\\log_{10} \\Gamma_e$ limits.
        skye_solid_mixing_rule: Solid mixing rule (`"PC"` or `"Ogata"`).
        use_cms: Whether `CMS` H/He EOS is enabled.
        use_free_eos: Whether `FreeEOS` 4D tables are enabled.
        use_opal_scvh: Whether `OPAL/SCVH` 4D tables are enabled.
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        [`EosPtOutput`] containing `rho`, `log10_rho`, implicit derivatives, and
        the 26-channel [`EosEvalOutput`].
    """
    if _dispatch.resolve_backend(backend, pgas, temp, xa) == "jax":
        return eos_pt_get_jax(
            jnp.asarray(pgas, np.float64),
            jnp.asarray(temp, np.float64),
            chem_id,
            jnp.asarray(xa, np.float64),
            use_skye=use_skye,
            use_pc=use_pc,
            pc_use_gamma_limit=pc_use_gamma_limit,
            skye_solid_mixing_rule=skye_solid_mixing_rule,
            use_cms=use_cms,
            use_free_eos=use_free_eos,
            use_opal_scvh=use_opal_scvh,
        )
    f_args = _prep(
        chem_id,
        xa,
        use_skye,
        use_pc,
        pc_use_gamma_limit,
        skye_solid_mixing_rule,
        use_cms,
        use_free_eos,
        use_opal_scvh,
    )
    return _pack_pt(_rust.eos_eval_pt_get(float(pgas), float(temp), *f_args))


def _call_inv_solver(
    fn: Any,
    val: float,
    which_other: int,
    other_value: float,
    chem_id: Sequence[int],
    xa: np.ndarray,
    log_guess: float,
    use_skye: bool,
    use_pc: bool,
    use_cms: bool,
    use_free_eos: bool,
    use_opal_scvh: bool,
) -> EosInverseOutput:
    """Shared FFI helper for `eosDT_get_*` and `eosPT_get_*` inverse solvers."""
    xa_c = np.ascontiguousarray(xa, np.float64)
    cids = [int(c) for c in chem_id]
    return _pack_inv(
        fn(
            float(val),
            int(which_other),
            float(other_value),
            cids,
            xa_c,
            float(log_guess),
            bool(use_skye),
            bool(use_pc),
            bool(use_cms),
            bool(use_free_eos),
            bool(use_opal_scvh),
        )
    )


def eos_dt_get_t(
    rho: float,
    which_other: int,
    other_value: float,
    chem_id: Sequence[int],
    xa: np.ndarray,
    *,
    log_t_guess: float = 6.0,
    use_skye: bool = True,
    use_pc: bool = False,
    use_cms: bool = False,
    use_free_eos: bool = True,
    use_opal_scvh: bool = True,
) -> EosInverseOutput:
    """Solve for $T$ given $\\rho$ and `res[which_other]`."""
    return _call_inv_solver(
        _rust.eos_eval_dt_get_t,
        rho,
        which_other,
        other_value,
        chem_id,
        xa,
        log_t_guess,
        use_skye,
        use_pc,
        use_cms,
        use_free_eos,
        use_opal_scvh,
    )


def eos_dt_get_rho(
    temp: float,
    which_other: int,
    other_value: float,
    chem_id: Sequence[int],
    xa: np.ndarray,
    *,
    log_rho_guess: float = 0.0,
    use_skye: bool = True,
    use_pc: bool = False,
    use_cms: bool = False,
    use_free_eos: bool = True,
    use_opal_scvh: bool = True,
) -> EosInverseOutput:
    """Solve for $\\rho$ given $T$ and `res[which_other]`."""
    return _call_inv_solver(
        _rust.eos_eval_dt_get_rho,
        temp,
        which_other,
        other_value,
        chem_id,
        xa,
        log_rho_guess,
        use_skye,
        use_pc,
        use_cms,
        use_free_eos,
        use_opal_scvh,
    )


def eos_pt_get_t(
    pgas: float,
    which_other: int,
    other_value: float,
    chem_id: Sequence[int],
    xa: np.ndarray,
    *,
    log_t_guess: float = 6.0,
    use_skye: bool = True,
    use_pc: bool = False,
    use_cms: bool = False,
    use_free_eos: bool = True,
    use_opal_scvh: bool = True,
) -> EosInverseOutput:
    """Solve for $T$ given $P_{\\mathrm{gas}}$ and `res[which_other]`."""
    return _call_inv_solver(
        _rust.eos_eval_pt_get_t,
        pgas,
        which_other,
        other_value,
        chem_id,
        xa,
        log_t_guess,
        use_skye,
        use_pc,
        use_cms,
        use_free_eos,
        use_opal_scvh,
    )


def eos_pt_get_pgas(
    temp: float,
    which_other: int,
    other_value: float,
    chem_id: Sequence[int],
    xa: np.ndarray,
    *,
    log_pgas_guess: float = 10.0,
    use_skye: bool = True,
    use_pc: bool = False,
    use_cms: bool = False,
    use_free_eos: bool = True,
    use_opal_scvh: bool = True,
) -> EosInverseOutput:
    """Solve for $P_{\\mathrm{gas}}$ given $T$ and `res[which_other]`."""
    return _call_inv_solver(
        _rust.eos_eval_pt_get_pgas,
        temp,
        which_other,
        other_value,
        chem_id,
        xa,
        log_pgas_guess,
        use_skye,
        use_pc,
        use_cms,
        use_free_eos,
        use_opal_scvh,
    )


def eos_gamma_de_get(
    abar: float,
    zbar: float,
    rho: float,
    energy: float,
    *,
    gamma: float = 5.0 / 3.0,
) -> EosInverseOutput:
    """Evaluate closed-form ideal $\\gamma$-law `eos_gamma_DE_get`."""
    return _pack_inv(
        _rust.eos_eval_gamma_de_get(
            float(abar), float(zbar), float(rho), float(energy), float(gamma)
        )
    )
