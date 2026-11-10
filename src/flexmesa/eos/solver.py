# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Top-level `eosDT_get` 7-level cascade, 4D table evaluators, and `PT`/`DE`
inverse solvers (`eos/public`).

Provides the Tier B dual-track (`backend="auto" | "rust" | "jax"`) Python and
JAX interfaces for [`eos_dt_get`], [`eos_pt_get`], [`eval_opal_scvh_eos`],
[`eval_free_eos`], [`eval_cms_eos`], [`eos_dt_get_t`], [`eos_dt_get_rho`],
[`eos_pt_get_t`], [`eos_pt_get_pgas`], and [`eos_gamma_de_get`].
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import _dispatch, _rust, chem
from flexmesa.eos import (
    _blend_dt_jax,
    _inverse_solver,
    _tables_4d_jax,
    core,
    skye,
)

EosPtOutput = _inverse_solver.EosPtOutput
EosInverseOutput = _inverse_solver.EosInverseOutput
WHICH_OTHER_EGAS = _inverse_solver.WHICH_OTHER_EGAS
WHICH_OTHER_LOG_PTOT = _inverse_solver.WHICH_OTHER_LOG_PTOT
eos_pt_get_jax = _inverse_solver.eos_pt_get_jax
eos_pt_get = _inverse_solver.eos_pt_get
eos_dt_get_t = _inverse_solver.eos_dt_get_t
eos_dt_get_rho = _inverse_solver.eos_dt_get_rho
eos_pt_get_t = _inverse_solver.eos_pt_get_t
eos_pt_get_pgas = _inverse_solver.eos_pt_get_pgas
eos_gamma_de_get = _inverse_solver.eos_gamma_de_get
_concrete_xa_np = _inverse_solver._concrete_xa_np
_prep = _inverse_solver._prep


def eos_dt_get_jax(
    rho: jax.Array,
    temp: jax.Array,
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
    solver_mode: str = "jax",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate `eosDT_get` in native JAX (or `solver_mode="rust_ffi"`)."""
    if solver_mode == "jax":
        return _blend_dt_jax.eos_dt_get_native_jax(
            jnp.asarray(rho, dtype=jnp.float64),
            jnp.asarray(temp, dtype=jnp.float64),
            chem_id,
            xa,
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
    solver_mode: str = "jax",
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
        solver_mode: JAX mode (`"jax"` for native JAX or `"rust_ffi"`).

    Returns:
        [`EosEvalOutput`] with `res`, `d_dlnd`, and `d_dln_t` of shape `(26,)`.
    """
    if _dispatch.resolve_backend(backend, rho, temp, xa) == "jax":
        res, d_dlnd, d_dln_t = eos_dt_get_jax(
            jnp.asarray(rho, np.float64),
            jnp.asarray(temp, np.float64),
            chem_id,
            xa,
            use_skye=use_skye,
            use_pc=use_pc,
            pc_use_gamma_limit=pc_use_gamma_limit,
            skye_solid_mixing_rule=skye_solid_mixing_rule,
            use_cms=use_cms,
            use_free_eos=use_free_eos,
            use_opal_scvh=use_opal_scvh,
            solver_mode=solver_mode,
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
    kind_code: int,
    rho: float | jax.Array,
    temp: float | jax.Array,
    chem_id: Sequence[int],
    xa: np.ndarray | jax.Array,
    backend: _dispatch.Backend,
    solver_mode: str = "jax",
) -> core.EosEvalOutput:
    """Shared dual-track wrapper for standalone 4D EOS table evaluators."""
    cids = [int(c) for c in chem_id]
    xa_c = _concrete_xa_np(xa)
    if _dispatch.resolve_backend(backend, rho, temp, xa) == "jax":
        r_j = jnp.asarray(rho, dtype=jnp.float64)
        t_j = jnp.asarray(temp, dtype=jnp.float64)
        if solver_mode == "jax":
            comp = chem.get_composition_info(cids, xa_c)
            if kind_code == 2:
                res, dlnd, dlnt = _tables_4d_jax.eval_cms_eos_jax(
                    r_j, t_j, float(comp.xh)
                )
            else:
                res, dlnd, dlnt = _tables_4d_jax.eval_eosdt_table_jax(
                    kind_code, r_j, t_j, float(comp.xh), float(comp.xz)
                )
            return core.EosEvalOutput(res, dlnd, dlnt)

        def _host_cb(
            r_a: np.ndarray, t_a: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            res_np, dlnd_np, dlnt_np = fn(float(r_a), float(t_a), cids, xa_c)
            return (
                np.asarray(res_np, dtype=np.float64),
                np.asarray(dlnd_np, dtype=np.float64),
                np.asarray(dlnt_np, dtype=np.float64),
            )

        res, d_dlnd, d_dln_t = skye.eval_host_dt_jvp(r_j, t_j, _host_cb)
        return core.EosEvalOutput(res, d_dlnd, d_dln_t)
    return core.EosEvalOutput(*fn(float(rho), float(temp), cids, xa_c))


def eval_opal_scvh_eos(
    rho: float | jax.Array,
    temp: float | jax.Array,
    chem_id: Sequence[int],
    xa: np.ndarray | jax.Array,
    *,
    backend: _dispatch.Backend = "auto",
    solver_mode: str = "jax",
) -> core.EosEvalOutput:
    """Evaluate `OPAL/SCVH` 4D tabulated EOS standalone at `(rho, T)`."""
    return _eval_4d_table_dt(
        _rust.eos_eval_opal_scvh,
        0,
        rho,
        temp,
        chem_id,
        xa,
        backend,
        solver_mode,
    )


def eval_free_eos(
    rho: float | jax.Array,
    temp: float | jax.Array,
    chem_id: Sequence[int],
    xa: np.ndarray | jax.Array,
    *,
    backend: _dispatch.Backend = "auto",
    solver_mode: str = "jax",
) -> core.EosEvalOutput:
    """Evaluate `FreeEOS` 4D tabulated EOS standalone at `(rho, T)`."""
    return _eval_4d_table_dt(
        _rust.eos_eval_free_eos,
        1,
        rho,
        temp,
        chem_id,
        xa,
        backend,
        solver_mode,
    )


def eval_cms_eos(
    rho: float | jax.Array,
    temp: float | jax.Array,
    chem_id: Sequence[int],
    xa: np.ndarray | jax.Array,
    *,
    backend: _dispatch.Backend = "auto",
    solver_mode: str = "jax",
) -> core.EosEvalOutput:
    """Evaluate Chabrier, Mazevet & Soubiran (`CMS`) EOS standalone."""
    return _eval_4d_table_dt(
        _rust.eos_eval_cms, 2, rho, temp, chem_id, xa, backend, solver_mode
    )
