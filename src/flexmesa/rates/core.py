# SPDX-License-Identifier: LGPL-3.0-or-later
"""Core nuclear reaction rate (`rates`) lookup and evaluation routines.

Provides reaction catalog queries (`reaction_id_by_name`, `reaction_name_by_id`,
`reaction_info`), REACLIB handle evaluators (`eval_reaclib_handle`), raw
unscreened rate dispatchers (`set_raw_rate`, `set_raw_rates`,
`set_raw_rates_zones`), Fushiki & Lamb (1987) pycnonuclear triple-alpha
(`eval_fl_epsnuc_3alf`), and `interp_pm` rate-table evaluation
(`eval_rate_table`).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from flexmesa import _dispatch, _rust
from flexmesa.rates import jax_kernels

if TYPE_CHECKING:
    import jax
    import numpy.typing as npt


@dataclasses.dataclass(frozen=True)
class ReactionInfo:
    """Metadata for a single MESA nuclear reaction (`1..=316`).

    Attributes:
        id: 1-based reaction ID (`1..=316`).
        name: Canonical MESA reaction handle (e.g. `"r1212"`, `"rpp_to_he3"`).
        op_code: Evaluator dispatch code (`0` = REACLIB default, `1` = table,
            `2` = `-1`, `3..=28` = analytic `ratelib`, `29..=31` = JINA).
        is_reverse: True if evaluating a reverse reaction channel.
        target_handle: Underlying REACLIB reaction handle.
        ye_exp: Electron mole fraction exponent $p$ ($Y_e^p$).
        rho_exp: Density exponent $N_{\\mathrm{in}} - 1$
            ($\\rho^{N_{\\mathrm{in}}-1}$).
        category: 1-based MESA energy-generation category (`1..=22`).
        std_q: Total nuclear $Q$-value in $\\mathrm{MeV}$.
        std_qneu: Standard neutrino energy loss $Q_{\\nu}$ in $\\mathrm{MeV}$.
        weak_lowt_rate: Low-temperature ($T_9 \\le 0.01$) weak decay rate in
            $\\mathrm{s}^{-1}$ (or negative sentinel if strong).
    """

    id: int
    name: str
    op_code: int
    is_reverse: bool
    target_handle: str
    ye_exp: int
    rho_exp: int
    category: int
    std_q: float
    std_qneu: float
    weak_lowt_rate: float


def reaction_id_by_name(name: str) -> int:
    """Look up the 1-based MESA reaction ID (`1..=316`) for `name`.

    Args:
        name: Reaction handle string (e.g. `"rpp_to_he3"`, `"r_c12_ag_o16"`).

    Returns:
        1-based integer reaction ID (`1..=316`).
    """
    return int(_rust.rates_reaction_id_by_name(name))


def reaction_name_by_id(ir: int) -> str:
    """Look up the canonical MESA reaction handle for 1-based ID `ir`.

    Args:
        ir: 1-based reaction ID (`1..=316`).

    Returns:
        Canonical reaction handle string.
    """
    return str(_rust.rates_reaction_name_by_id(int(ir)))


def reaction_info(ir: int | str) -> ReactionInfo:
    """Return [`ReactionInfo`] metadata for a 1-based reaction ID or handle.

    Args:
        ir: 1-based reaction ID (`1..=316`) or reaction name string.

    Returns:
        Populated [`ReactionInfo`] instance.
    """
    rid = reaction_id_by_name(ir) if isinstance(ir, str) else int(ir)
    (
        name,
        op_code,
        is_rev,
        target_handle,
        ye_exp,
        rho_exp,
        category,
        std_q,
        std_qneu,
        weak_lowt_rate,
    ) = _rust.rates_reaction_info(rid)
    return ReactionInfo(
        id=rid,
        name=name,
        op_code=op_code,
        is_reverse=is_rev,
        target_handle=target_handle,
        ye_exp=ye_exp,
        rho_exp=rho_exp,
        category=category,
        std_q=std_q,
        std_qneu=std_qneu,
        weak_lowt_rate=weak_lowt_rate,
    )


def eval_reaclib_handle(
    handle: str,
    t9: float | jax.Array,
    *,
    check_e_flag: bool = True,
    backend: _dispatch.Backend = "auto",
) -> (
    tuple[float, float, float, float]
    | tuple[jax.Array, jax.Array, jax.Array, jax.Array]
):
    """Evaluate a REACLIB reaction handle at $T_9 = T / 10^9\\,\\mathrm{K}$.

    Args:
        handle: REACLIB reaction handle (e.g. `"r_he4_he4_he4_to_c12"`).
        t9: Temperature $T_9$ in $10^9\\,\\mathrm{K}$.
        check_e_flag: Whether to exempt `'e'`-flagged REACLIB terms from the
            low-$T_9$ cutoff.
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Tuple `(forward_rate, dforward_dlnT, reverse_rate, dreverse_dlnT)` as
        Python `float`s on `"rust"` backend or differentiable JAX `float64`
        arrays on `"jax"` backend.
    """
    resolved = _dispatch.resolve_backend(backend, t9)
    if resolved == "jax":
        is_rev, terms = _rust.rates_reaclib_terms(handle)
        rflags = jnp.asarray([t[1] for t in terms], dtype=jnp.int32)
        inv_exps = jnp.asarray([t[2] for t in terms], dtype=jnp.float64)
        coeffs = jnp.asarray([t[3] for t in terms], dtype=jnp.float64)
        inv_c1s = jnp.asarray([t[4] for t in terms], dtype=jnp.float64)
        inv_c2s = jnp.asarray([t[5] for t in terms], dtype=jnp.float64)
        inv_parts = jnp.asarray([t[6] for t in terms], dtype=jnp.float64)
        lam, dlam, rlam, drlam = jax_kernels.eval_reaclib_terms_jax(
            coeffs,
            inv_exps,
            inv_c1s,
            inv_c2s,
            inv_parts,
            rflags,
            jnp.asarray(t9, dtype=jnp.float64),
            jnp.asarray(check_e_flag),
        )
        if is_rev:
            return rlam, drlam, lam, dlam
        return lam, dlam, rlam, drlam
    return _rust.rates_eval_reaclib_handle(
        handle, float(t9), bool(check_e_flag)
    )


def set_raw_rate(
    ir: int | str,
    temp: float,
    *,
    log_t: float | None = None,
) -> float:
    """Evaluate the unscreened reaction rate `set_raw_rate` for reaction `ir`.

    Args:
        ir: 1-based reaction ID (`1..=316`) or reaction name string.
        temp: Temperature $T$ in $\\mathrm{K}$.
        log_t: Optional $\\log_{10}(T / \\mathrm{K})$ matching `tfactors`.

    Returns:
        Unscreened nuclear reaction rate in CGS/molar units.
    """
    rid = reaction_id_by_name(ir) if isinstance(ir, str) else int(ir)
    lt = None if log_t is None else float(log_t)
    return float(_rust.rates_set_raw_rate(rid, float(temp), lt))


def set_raw_rates(
    irs: npt.ArrayLike,
    temp: float,
    *,
    log_t: float | None = None,
) -> npt.NDArray[np.float64]:
    """Evaluate `set_raw_rate` across a 1D sequence of 1-based reaction IDs.

    Args:
        irs: 1D sequence of 1-based reaction IDs `(N_rxns,)` in `1..=316`.
        temp: Temperature $T$ in $\\mathrm{K}$.
        log_t: Optional $\\log_{10}(T / \\mathrm{K})$.

    Returns:
        1D `float64` NumPy array of shape `(N_rxns,)` with raw reaction rates.
    """
    ir_arr = np.ascontiguousarray(irs, dtype=np.uintp)
    lt = None if log_t is None else float(log_t)
    return _rust.rates_set_raw_rates(ir_arr, float(temp), lt)


def set_raw_rates_zones(
    irs: npt.ArrayLike,
    temps: npt.ArrayLike,
    *,
    log_ts: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Evaluate `set_raw_rates` across `N_zones` stellar zones.

    Args:
        irs: 1D sequence of 1-based reaction IDs `(N_rxns,)`.
        temps: 1D array of zone temperatures `(N_zones,)` in $\\mathrm{K}$.
        log_ts: Optional 1D array of $\\log_{10}(T / \\mathrm{K})$ `(N_zones,)`.

    Returns:
        2D `float64` NumPy array of shape `(N_zones, N_rxns)`.
    """
    ir_arr = np.ascontiguousarray(irs, dtype=np.uintp)
    t_arr = np.ascontiguousarray(temps, dtype=np.float64)
    lt_arr = (
        None
        if log_ts is None
        else np.ascontiguousarray(log_ts, dtype=np.float64)
    )
    return _rust.rates_set_raw_rates_zones(ir_arr, t_arr, lt_arr)


def eval_fl_epsnuc_3alf(
    temp: float | jax.Array,
    rho: float | jax.Array,
    y_he4: float | jax.Array,
    mu_e: float | jax.Array = 2.0,
    *,
    backend: _dispatch.Backend = "auto",
) -> float | jax.Array:
    """Evaluate Fushiki & Lamb (1987) pycnonuclear 3-alpha energy rate.

    Args:
        temp: Temperature $T$ in $\\mathrm{K}$.
        rho: Mass density $\\rho$ in $\\mathrm{g\\,cm^{-3}}$.
        y_he4: Helium-4 mass fraction $Y$.
        mu_e: Electron molecular weight $\\mu_e \\approx 1 / Y_e$
            (default `2.0`).
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Specific energy generation rate $\\epsilon_{3\\alpha}$ in
        $\\mathrm{erg\\,g^{-1}\\,s^{-1}}$.
    """
    resolved = _dispatch.resolve_backend(backend, temp, rho, y_he4, mu_e)
    if resolved == "jax":
        return jax_kernels.eval_fl_epsnuc_3alf_jax(
            jnp.asarray(temp),
            jnp.asarray(rho),
            jnp.asarray(y_he4),
            jnp.asarray(mu_e),
        )
    return float(
        _rust.rates_eval_fl_epsnuc_3alf(
            float(temp), float(rho), float(y_he4), float(mu_e)
        )
    )


def eval_rate_table(ir: int | str, temp: float) -> tuple[float, float]:
    """Evaluate a 1D `interp_pm` cubic rate table `(rate, drate_dt)`.

    Args:
        ir: 1-based reaction ID (`1..=316`) or reaction name (e.g. `"r3"`).
        temp: Temperature $T$ in $\\mathrm{K}$.

    Returns:
        Tuple `(rate, drate_dt)` in CGS units and $\\mathrm{K}^{-1}$.
    """
    rid = reaction_id_by_name(ir) if isinstance(ir, str) else int(ir)
    return _rust.rates_eval_rate_table(rid, float(temp))
