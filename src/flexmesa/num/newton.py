# SPDX-License-Identifier: LGPL-3.0-or-later
"""Damped Newton-Raphson solver for dense, banded, and block-tridiagonal grids.

Ports MESA's Newton-Raphson iteration (``num/private/mod_newton.f90`` and
``num/test/src/test_newton.f90``) with line-search damping across dense,
banded, and block-tridiagonal Jacobians (:mod:`flexmesa.mtx`), plus a
JAX-differentiable implicit Newton solver (:func:`newton_solve_jax`) using
``@jax.custom_vjp``.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from flexmesa.mtx import band as mtx_band
from flexmesa.mtx import block_tridiag as mtx_btd
from flexmesa.mtx import dense as mtx_dense


@dataclasses.dataclass(frozen=True)
class NewtonConfig:
    """Configuration for damped Newton-Raphson solvers.

    Attributes:
        max_iter: Maximum Newton iterations allowed.
        tol_correction_norm: Maximum RMS correction norm.
        tol_max_correction: Maximum supremum correction norm.
        tol_residual_norm: Maximum RMS residual norm.
        min_damping: Minimum line-search damping factor.
    """

    max_iter: int = 50
    tol_correction_norm: float = 1e-9
    tol_max_correction: float = 1e-7
    tol_residual_norm: float = 1e-10
    min_damping: float = 1e-3


@dataclasses.dataclass(frozen=True)
class NewtonResult:
    """Convergence diagnostics returned by Newton-Raphson solvers.

    Attributes:
        y: Converged state vector of shape ``(n,)`` in ``float64``.
        iterations: Number of Newton iterations completed.
        residual_norm: Final RMS residual norm.
        correction_norm: Final RMS correction norm.
    """

    y: npt.NDArray[np.float64]
    iterations: int
    residual_norm: float
    correction_norm: float


def _rms_and_max(vec: npt.NDArray[np.float64]) -> tuple[float, float]:
    """Computes the RMS norm and supremum norm of ``vec``.

    Args:
        vec: 1D ``float64`` array of shape ``(n,)``.

    Returns:
        Tuple ``(rms_norm, max_abs_norm)``.
    """
    if vec.size == 0:
        return 0.0, 0.0
    rms = float(math.sqrt(float(np.mean(vec * vec))))
    max_abs = float(np.max(np.abs(vec)))
    return rms, max_abs


def _newton_driver(
    y_init: npt.ArrayLike,
    config: NewtonConfig,
    eval_res: Callable[[npt.NDArray[np.float64]], npt.ArrayLike],
    solve_step: Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64]],
        npt.NDArray[np.float64],
    ],
) -> NewtonResult:
    """Generic damped Newton-Raphson loop.

    Args:
        y_init: Initial state vector of shape ``(n,)``.
        config: :class:`NewtonConfig` tolerances and limits.
        eval_res: Residual callable ``eval_res(y)`` returning shape ``(n,)``.
        solve_step: Linear step solver ``solve_step(y, neg_res)``.

    Returns:
        :class:`NewtonResult` with converged state and norm diagnostics.

    Raises:
        ValueError: If ``y_init`` is empty.
        RuntimeError: If ``config.max_iter`` is exceeded.
    """
    y = np.asarray(y_init, dtype=np.float64).ravel().copy()
    if y.size == 0:
        raise ValueError("y_init must be non-empty in newton_solve")
    res = np.asarray(eval_res(y), dtype=np.float64).ravel()
    res_norm, _ = _rms_and_max(res)
    if res_norm <= config.tol_residual_norm:
        return NewtonResult(
            y=y,
            iterations=0,
            residual_norm=res_norm,
            correction_norm=0.0,
        )

    for it in range(1, config.max_iter + 1):
        dy = np.asarray(solve_step(y, -res), dtype=np.float64).ravel()
        corr_norm, max_corr = _rms_and_max(dy)

        coeff = 1.0
        y_trial = y
        res_trial = res
        res_trial_norm = res_norm
        while coeff >= config.min_damping:
            y_trial = y + coeff * dy
            res_trial = np.asarray(eval_res(y_trial), dtype=np.float64).ravel()
            rn, _ = _rms_and_max(res_trial)
            res_trial_norm = rn
            if (
                rn <= (1.0 - 0.1 * coeff) * res_norm
                or rn <= config.tol_residual_norm
                or (coeff == 1.0 and corr_norm <= config.tol_correction_norm)
            ):
                break
            coeff *= 0.5
        y = y_trial
        res = res_trial
        res_norm = res_trial_norm

        if (
            corr_norm * coeff <= config.tol_correction_norm
            and max_corr * coeff <= config.tol_max_correction
        ) or res_norm <= config.tol_residual_norm:
            return NewtonResult(
                y=y,
                iterations=it,
                residual_norm=res_norm,
                correction_norm=corr_norm * coeff,
            )

    raise RuntimeError("newton_solve exceeded max_iter")


def newton_solve_dense(
    y_init: npt.ArrayLike,
    eval_res: Callable[[npt.NDArray[np.float64]], npt.ArrayLike],
    eval_jac: Callable[[npt.NDArray[np.float64]], npt.ArrayLike],
    config: NewtonConfig | None = None,
) -> NewtonResult:
    """Solves $F(y) = 0$ with dense $(N, N)$ Jacobian via damped Newton-Raphson.

    Args:
        y_init: Initial state vector of shape ``(n,)``.
        eval_res: Residual callable ``eval_res(y)`` returning shape ``(n,)``.
        eval_jac: Jacobian callable ``eval_jac(y)`` returning shape ``(n, n)``.
        config: Optional :class:`NewtonConfig` settings.

    Returns:
        :class:`NewtonResult` with converged state and norm diagnostics.
    """
    cfg = NewtonConfig() if config is None else config

    def _step(
        y: npt.NDArray[np.float64],
        rhs: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        n = int(y.size)
        j_mat = np.asarray(eval_jac(y), dtype=np.float64).reshape((n, n))
        return mtx_dense.solve_dense(j_mat, rhs)

    return _newton_driver(y_init, cfg, eval_res, _step)


def newton_solve_banded(
    kl: int,
    ku: int,
    y_init: npt.ArrayLike,
    eval_res: Callable[[npt.NDArray[np.float64]], npt.ArrayLike],
    eval_jac_band: Callable[[npt.NDArray[np.float64]], npt.ArrayLike],
    config: NewtonConfig | None = None,
) -> NewtonResult:
    """Solves $F(y) = 0$ with banded Jacobian ``(2*kl + ku + 1, n)``.

    Args:
        kl: Number of sub-diagonals ($k_l \\ge 0$).
        ku: Number of super-diagonals ($k_u \\ge 0$).
        y_init: Initial state vector of shape ``(n,)``.
        eval_res: Residual callable ``eval_res(y)`` returning shape ``(n,)``.
        eval_jac_band: Banded Jacobian callable returning ``(ldab, n)``.
        config: Optional :class:`NewtonConfig` settings.

    Returns:
        :class:`NewtonResult` with converged state and norm diagnostics.
    """
    cfg = NewtonConfig() if config is None else config

    def _step(
        y: npt.NDArray[np.float64],
        rhs: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        ab = np.asarray(eval_jac_band(y), dtype=np.float64)
        return mtx_band.solve_banded(ab, kl, ku, rhs)

    return _newton_driver(y_init, cfg, eval_res, _step)


def newton_solve_block_tridiagonal(
    nvar: int,
    nz: int,
    y_init: npt.ArrayLike,
    eval_res: Callable[[npt.NDArray[np.float64]], npt.ArrayLike],
    eval_blocks: Callable[
        [npt.NDArray[np.float64]],
        tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike],
    ],
    config: NewtonConfig | None = None,
) -> NewtonResult:
    """Solves $F(y) = 0$ across ``nz`` shells and ``nvar`` variables per shell.

    Args:
        nvar: Variables per shell ($n_{\\mathrm{var}} \\ge 1$).
        nz: Number of shells ($n_z \\ge 1$).
        y_init: Initial state vector of shape ``(nz * nvar,)`` or
            ``(nz, nvar)``.
        eval_res: Residual callable returning shape ``(nz * nvar,)``.
        eval_blocks: Callable returning ``(lblk, dblk, ublk)`` each of shape
            ``(nz, nvar, nvar)``.
        config: Optional :class:`NewtonConfig` settings.

    Returns:
        :class:`NewtonResult` with converged state and norm diagnostics.
    """
    cfg = NewtonConfig() if config is None else config

    def _step(
        y: npt.NDArray[np.float64],
        rhs: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        lblk, dblk, ublk = eval_blocks(y)
        sol = mtx_btd.solve_block_tridiagonal(
            lblk, dblk, ublk, rhs.reshape((nz, nvar))
        )
        return sol.ravel()

    return _newton_driver(y_init, cfg, eval_res, _step)


def newton_solve_jax(
    res_fn: Callable[[jax.Array, jax.Array], jax.Array],
    y_init: jax.Array,
    params: jax.Array,
    max_iter: int = 30,
    tol: float = 1e-12,
) -> jax.Array:
    """Solves $F(y^*(p), p) = 0$ in JAX ``float64`` with implicit VJP.

    Uses Newton-Raphson with ``jax.jacfwd`` for the primal solve and propagates
    reverse-mode cotangents via the implicit function theorem:
    $$J_y^\\top v = g, \\qquad \\bar{p} = -J_p^\\top v,$$
    with $J_y = \\partial F / \\partial y$ and $J_p = \\partial F / \\partial p$
    evaluated at $(y^*, p)$.

    Args:
        res_fn: Differentiable JAX callable ``res_fn(y, params)`` returning
            residual of shape ``(n,)`` in ``float64``.
        y_init: Initial guess of shape ``(n,)`` in ``float64``.
        params: Parameter vector of shape ``(k,)`` in ``float64``.
        max_iter: Maximum Newton iterations.
        tol: Residual norm stopping tolerance.

    Returns:
        Converged state ``y_star`` of shape ``(n,)`` supporting ``jax.grad``
        and ``jax.jacrev``.
    """
    y0 = jnp.asarray(y_init, dtype=jnp.float64)

    @functools.partial(jax.custom_vjp)
    def _solve(p: jax.Array) -> jax.Array:
        jac_y_fn = jax.jacfwd(res_fn, argnums=0)
        y = y0
        for _ in range(max_iter):
            f_val = res_fn(y, p)
            if float(jnp.linalg.norm(f_val)) <= tol:
                break
            j_mat = jac_y_fn(y, p)
            dy = jnp.linalg.solve(j_mat, -f_val)
            y = y + dy
        return y

    def _solve_fwd(
        p: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        y_star = _solve(p)
        return y_star, (y_star, p)

    def _solve_bwd(
        res: tuple[jax.Array, jax.Array],
        g: jax.Array,
    ) -> tuple[jax.Array]:
        y_star, p = res
        j_y = jax.jacfwd(res_fn, argnums=0)(y_star, p)
        j_p = jax.jacfwd(res_fn, argnums=1)(y_star, p)
        v = jnp.linalg.solve(j_y.T, g)
        return (-jnp.dot(v, j_p),)

    _solve.defvjp(_solve_fwd, _solve_bwd)
    return _solve(jnp.asarray(params, dtype=jnp.float64))
