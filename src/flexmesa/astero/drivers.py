# SPDX-License-Identifier: LGPL-3.0-or-later
"""Mode matching, $r_{02}$ ratios, composite $\\chi^2$, and optimizers."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from flexmesa import _dispatch, _rust
from flexmesa.astero import jax_kernels


def _is_jax(x: Any, backend: Any) -> bool:
    """Returns `True` when `resolve_backend(backend, x)` selects JAX."""
    return _dispatch.resolve_backend(backend, x) == "jax"


def match_modes_by_degree(
    obs_freqs: Any,
    cand_freqs: Any,
    cand_inertias: Any,
    cand_orders: Any,
) -> tuple[Any, Any, Any]:
    """Matches observed mode frequencies to candidate model modes.

    Args:
        obs_freqs: Observed frequencies (`(M_l,)`) [`uHz`].
        cand_freqs: Candidate model frequencies (`(K_l,)`) [`uHz`].
        cand_inertias: Candidate mode inertias (`(K_l,)`).
        cand_orders: Candidate radial orders (`(K_l,)`, `int32`).

    Returns:
        Tuple `(matched_freqs, matched_inertias, matched_orders)`.
    """
    return _rust.astero_match_modes_by_degree(
        _dispatch.as_c_f64(obs_freqs),
        _dispatch.as_c_f64(cand_freqs),
        _dispatch.as_c_f64(cand_inertias),
        np.ascontiguousarray(cand_orders, dtype=np.int32),
    )


def eval_ratios_r02(
    l0_freqs: Any,
    l0_sigs: Any,
    l1_freqs: Any,
    l1_sigs: Any,
    l2_freqs: Any,
    l2_sigs: Any,
    *,
    backend: str = "auto",
) -> tuple[Any, Any]:
    """Evaluates quadrupole-to-radial frequency separation ratios $r_{02}(n)$.

    Args:
        l0_freqs: Radial ($l=0$) frequencies (`(L,)`) [`uHz`].
        l0_sigs: Radial ($l=0$) uncertainties (`(L,)`) [`uHz`].
        l1_freqs: Dipole ($l=1$) frequencies (`(L,)`) [`uHz`].
        l1_sigs: Dipole ($l=1$) uncertainties (`(L,)`) [`uHz`].
        l2_freqs: Quadrupole ($l=2$) frequencies (`(L,)`) [`uHz`].
        l2_sigs: Quadrupole ($l=2$) uncertainties (`(L,)`) [`uHz`].
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Tuple `(r02, sig_r02)` of shape `(L - 1,)`.
    """
    if _is_jax(l0_freqs, backend):
        r02, sig_r02 = jax_kernels.eval_ratios_r02_jax(
            l0_freqs, l0_sigs, l1_freqs, l1_sigs, l2_freqs, l2_sigs
        )
        return r02, sig_r02
    return _rust.astero_eval_ratios_r02(
        _dispatch.as_c_f64(l0_freqs),
        _dispatch.as_c_f64(l0_sigs),
        _dispatch.as_c_f64(l1_freqs),
        _dispatch.as_c_f64(l1_sigs),
        _dispatch.as_c_f64(l2_freqs),
        _dispatch.as_c_f64(l2_sigs),
    )


def eval_astero_chi2(
    obs_freqs: Any,
    obs_sigmas: Any,
    corr_model_freqs: Any,
    spectro_obs: Any,
    spectro_sigmas: Any,
    spectro_model: Any,
    r010_obs: Any | None = None,
    r010_sigmas: Any | None = None,
    r010_model: Any | None = None,
    r02_obs: Any | None = None,
    r02_sigmas: Any | None = None,
    r02_model: Any | None = None,
    delta_nu_obs: float = 0.0,
    delta_nu_sigma: float = 0.0,
    delta_nu_model: float = 0.0,
    nu_max_obs: float = 0.0,
    nu_max_sigma: float = 0.0,
    nu_max_model: float = 0.0,
    chi2_seismo_fraction: float = 2.0 / 3.0,
    chi2_seismo_delta_nu_fraction: float = 0.0,
    chi2_seismo_nu_max_fraction: float = 0.0,
    chi2_seismo_r_010_fraction: float = 0.0,
    chi2_seismo_r_02_fraction: float = 0.0,
    *,
    backend: str = "auto",
) -> dict[str, Any]:
    """Evaluates composite asteroseismic + spectroscopic $\\chi^2$ (`get_chi2`).

    Args:
        obs_freqs: Observed frequencies (`(M,)`) [`uHz`].
        obs_sigmas: Observed uncertainties (`(M,)`) [`uHz`].
        corr_model_freqs: Surface-corrected model frequencies (`(M,)`) [`uHz`].
        spectro_obs: Spectroscopic targets (`(S,)`).
        spectro_sigmas: Spectroscopic uncertainties (`(S,)`).
        spectro_model: Model spectroscopic observables (`(S,)`).
        r010_obs: Observed $r_{010}$ ratios (`(K,)`, optional).
        r010_sigmas: Observed $r_{010}$ uncertainties (`(K,)`, optional).
        r010_model: Model $r_{010}$ ratios (`(K,)`, optional).
        r02_obs: Observed $r_{02}$ ratios (`(K_2,)`, optional).
        r02_sigmas: Observed $r_{02}$ uncertainties (`(K_2,)`, optional).
        r02_model: Model $r_{02}$ ratios (`(K_2,)`, optional).
        delta_nu_obs: Observed $\\Delta\\nu$ [`uHz`].
        delta_nu_sigma: Uncertainty on $\\Delta\\nu$ [`uHz`].
        delta_nu_model: Model $\\Delta\\nu$ [`uHz`].
        nu_max_obs: Observed $\\nu_{\\max}$ [`uHz`].
        nu_max_sigma: Uncertainty on $\\nu_{\\max}$ [`uHz`].
        nu_max_model: Model $\\nu_{\\max}$ [`uHz`].
        chi2_seismo_fraction: Weight of $\\chi^2_{\\mathrm{seismo}}$ in
            $\\chi^2_{\\mathrm{total}}$.
        chi2_seismo_delta_nu_fraction: Weight of $\\chi^2_{\\Delta\\nu}$.
        chi2_seismo_nu_max_fraction: Weight of $\\chi^2_{\\nu_{\\max}}$.
        chi2_seismo_r_010_fraction: Weight of $\\chi^2_{r_{010}}$.
        chi2_seismo_r_02_fraction: Weight of $\\chi^2_{r_{02}}$.
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        Dictionary of $\\chi^2$ components (`chi2_total`, `chi2_seismo`,
        `chi2_spectro`, `chi2_frequencies`, `chi2_r_010`, `chi2_r_02`).
    """
    empty = np.zeros(0, dtype=np.float64)
    r010_o = empty if r010_obs is None else r010_obs
    r010_s = empty if r010_sigmas is None else r010_sigmas
    r010_m = empty if r010_model is None else r010_model
    r02_o = empty if r02_obs is None else r02_obs
    r02_s = empty if r02_sigmas is None else r02_sigmas
    r02_m = empty if r02_model is None else r02_model

    if _is_jax(corr_model_freqs, backend):
        c_freq = jax_kernels._mean_chi2_jax(
            obs_freqs, obs_sigmas, corr_model_freqs
        )
        c_r010 = jax_kernels._mean_chi2_jax(
            jnp.asarray(r010_o, dtype=jnp.float64),
            jnp.asarray(r010_s, dtype=jnp.float64),
            jnp.asarray(r010_m, dtype=jnp.float64),
        )
        c_r02 = jax_kernels._mean_chi2_jax(
            jnp.asarray(r02_o, dtype=jnp.float64),
            jnp.asarray(r02_s, dtype=jnp.float64),
            jnp.asarray(r02_m, dtype=jnp.float64),
        )
        c_spec = jax_kernels._mean_chi2_jax(
            spectro_obs, spectro_sigmas, spectro_model
        )
        c_dnu = (
            ((delta_nu_obs - delta_nu_model) / delta_nu_sigma) ** 2
            if delta_nu_sigma > 0.0
            else 0.0
        )
        c_numax = (
            ((nu_max_obs - nu_max_model) / nu_max_sigma) ** 2
            if nu_max_sigma > 0.0
            else 0.0
        )
        f_freq = jnp.clip(
            1.0
            - (
                chi2_seismo_r_010_fraction
                + chi2_seismo_r_02_fraction
                + chi2_seismo_delta_nu_fraction
                + chi2_seismo_nu_max_fraction
            ),
            0.0,
            1.0,
        )
        c_seis = (
            chi2_seismo_r_010_fraction * c_r010
            + chi2_seismo_r_02_fraction * c_r02
            + f_freq * c_freq
            + chi2_seismo_delta_nu_fraction * c_dnu
            + chi2_seismo_nu_max_fraction * c_numax
        )
        f_seis = jnp.clip(jnp.float64(chi2_seismo_fraction), 0.0, 1.0)
        c_tot = f_seis * c_seis + (1.0 - f_seis) * c_spec
        return {
            "chi2_total": c_tot,
            "chi2_seismo": c_seis,
            "chi2_spectro": c_spec,
            "chi2_frequencies": c_freq,
            "chi2_r_010": c_r010,
            "chi2_r_02": c_r02,
        }
    return _rust.astero_eval_chi2(
        _dispatch.as_c_f64(obs_freqs),
        _dispatch.as_c_f64(obs_sigmas),
        _dispatch.as_c_f64(corr_model_freqs),
        _dispatch.as_c_f64(r010_o),
        _dispatch.as_c_f64(r010_s),
        _dispatch.as_c_f64(r010_m),
        _dispatch.as_c_f64(r02_o),
        _dispatch.as_c_f64(r02_s),
        _dispatch.as_c_f64(r02_m),
        float(delta_nu_obs),
        float(delta_nu_sigma),
        float(delta_nu_model),
        float(nu_max_obs),
        float(nu_max_sigma),
        float(nu_max_model),
        _dispatch.as_c_f64(spectro_obs),
        _dispatch.as_c_f64(spectro_sigmas),
        _dispatch.as_c_f64(spectro_model),
        float(chi2_seismo_fraction),
        float(chi2_seismo_delta_nu_fraction),
        float(chi2_seismo_nu_max_fraction),
        float(chi2_seismo_r_010_fraction),
        float(chi2_seismo_r_02_fraction),
    )


def nelder_mead_minimize(
    objective: Any,
    x_first: Any,
    x_lower: Any,
    x_upper: Any,
    *,
    max_evals: int = 500,
    x_atol: float = 1.0e-5,
    x_rtol: float = 1.0e-5,
    chi2_tol: float = 0.0,
    centroid_weight_power: float = 0.0,
) -> dict[str, Any]:
    """Minimizes `objective(x)` via Nelder-Mead (`mod_simplex.f90`).

    Args:
        objective: Callable `objective(x_arr) -> float`.
        x_first: Initial guess (`(D,)`).
        x_lower: Lower parameter bounds (`(D,)`).
        x_upper: Upper parameter bounds (`(D,)`).
        max_evals: Maximum objective evaluations.
        x_atol: Absolute simplex convergence tolerance.
        x_rtol: Relative simplex convergence tolerance.
        chi2_tol: Stop early if objective $\\le \\chi^2_{\\mathrm{tol}}$.
        centroid_weight_power: Centroid inverse-objective weight power.

    Returns:
        Dictionary with keys `x_best`, `f_best`, and `num_evals`.
    """
    return _rust.astero_nelder_mead_minimize(
        objective,
        [float(v) for v in x_first],
        [float(v) for v in x_lower],
        [float(v) for v in x_upper],
        int(max_evals),
        float(x_atol),
        float(x_rtol),
        float(chi2_tol),
        float(centroid_weight_power),
    )


def powell_quadratic_minimize(
    objective: Any,
    x_first: Any,
    x_lower: Any,
    x_upper: Any,
    *,
    max_evals: int = 500,
    rhobeg: float = 0.5,
    rhoend: float = 1.0e-4,
) -> dict[str, Any]:
    """Minimizes `objective(x)` via Powell's quadratic trust-region search.

    Args:
        objective: Callable `objective(x_arr) -> float`.
        x_first: Initial guess (`(D,)`).
        x_lower: Lower parameter bounds (`(D,)`).
        x_upper: Upper parameter bounds (`(D,)`).
        max_evals: Maximum objective evaluations.
        rhobeg: Initial normalized trust-region radius.
        rhoend: Final normalized trust-region radius.

    Returns:
        Dictionary with keys `x_best`, `f_best`, and `num_evals`.
    """
    return _rust.astero_powell_quadratic_minimize(
        objective,
        [float(v) for v in x_first],
        [float(v) for v in x_lower],
        [float(v) for v in x_upper],
        int(max_evals),
        float(rhobeg),
        float(rhoend),
    )


def scan_cartesian_grid(
    objective: Any,
    x_lower: Any,
    x_upper: Any,
    x_delta: Any,
) -> dict[str, Any]:
    """Evaluates `objective(x)` across a Cartesian parameter grid.

    Args:
        objective: Callable `objective(x_arr) -> float`.
        x_lower: Lower grid bounds (`(D,)`).
        x_upper: Upper grid bounds (`(D,)`).
        x_delta: Grid step sizes (`(D,)`).

    Returns:
        Dictionary with keys `x_best`, `f_best`, `num_evals`, and `all_values`.
    """
    return _rust.astero_scan_cartesian_grid(
        objective,
        [float(v) for v in x_lower],
        [float(v) for v in x_upper],
        [float(v) for v in x_delta],
    )
