# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Four-dimensional $(Z, X, \log Q, \log T)$ ionization table evaluators.

Implements the Three-Tier Hybrid (`Tier B`) 3-stage interpolator
(`eval_ionization`, `eval_ionization_jax`, `eval_ionization_zones`) matching
upstream MESA `ion_tables_eval.f90`:
1. **Stage 1 ($(\log Q, \log T)$ bicubic spline)**: Evaluates not-a-knot
   bicubic splines with silent boundary clamping (`Locate_logQ`, `Locate_logT`
   with $+10^{-4}$ index bias).
2. **Stage 2 ($X$ interpolation)**: 3-point quadratic Lagrange near edges or
   4-point blended quadratic-cubic on interior intervals.
3. **Stage 3 ($Z$ interpolation)**: 3-point quadratic Lagrange on $Z \in
   [0, 0.04]$ or cosine-smoothed $0.5[1 - \cos(\pi\alpha)]$ blending on
   $Z \in (0.04, 1.0]$.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from flexmesa import _dispatch, _rust

NUM_ION_VALS: int = 28
ION_ZS: tuple[float, ...] = (0.00, 0.02, 0.04, 0.20, 1.00)
ION_XS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
ION_RESULT_NAMES: tuple[str, ...] = tuple(_rust.py_ion_result_names())


@dataclasses.dataclass(frozen=True)
class IonizationResult:
    r"""Structured 28-channel output of `eval_ionization`.

    Attributes:
        values: Raw 1D `float64` NumPy array of shape `(28,)` in
            `ION_RESULT_NAMES` order.
        log_pgas: Base-10 log gas pressure $\log_{10}(P_{\mathrm{gas}} /
            \mathrm{dyn\,cm^{-2}})$.
        logpp: 9-element `float64` array of log partial pressures (`H..Fe`).
        z_ion: 9-element `float64` array of mean ionic charges (`H..Fe`).
        fneut: 9-element `float64` array of neutral fractions (`H..Fe`).
    """

    values: npt.NDArray[np.float64]

    @property
    def log_pgas(self) -> float:
        r"""Base-10 log gas pressure $\log_{10} P_{\mathrm{gas}}$."""
        return float(self.values[0])

    @property
    def logpp(self) -> npt.NDArray[np.float64]:
        r"""Log partial pressures `logpp_H..Fe` of shape `(9,)`."""
        return self.values[1:10]

    @property
    def z_ion(self) -> npt.NDArray[np.float64]:
        r"""Mean ionic charges `Z_H..Fe` of shape `(9,)`."""
        return self.values[10:19]

    @property
    def fneut(self) -> npt.NDArray[np.float64]:
        r"""Neutral fractions `fneut_H..Fe` of shape `(9,)`."""
        return self.values[19:28]

    def __getitem__(self, key: int | str) -> float:
        r"""Index by 0-based integer channel `0..27` or channel name."""
        if isinstance(key, str):
            idx = ION_RESULT_NAMES.index(key)
            return float(self.values[idx])
        return float(self.values[key])


@functools.lru_cache(maxsize=1)
def _builtin_cell_coeffs_jax() -> jax.Array:
    r"""Load and cache the `(5, 6, 28, nq, nt, 4)` built-in 4D spline cell."""
    _, _, nq, _, _, nt, arr = _rust.py_ion_builtin_cell_grid()
    return jnp.asarray(arr, dtype=jnp.float64).reshape(5, 6, 28, nq, nt, 4)


def _uniform_axis_basis(
    val: jax.Array, vmin: float, dv: float, nv: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Locate cell index and cubic basis weights along a uniform 1D axis."""
    iraw = jnp.floor((val - vmin) / dv + 1.0e-4).astype(jnp.int32)
    idx = jnp.clip(iraw, 0, nv - 2)
    v0 = vmin + idx.astype(jnp.float64) * dv
    v1 = v0 + dv
    vclamped = jnp.where(iraw < 0, v0, jnp.where(iraw >= nv - 1, v1, val))
    u = (vclamped - v0) / dv
    ui = 1.0 - u
    return idx, u, ui, u * (u * u - 1.0), ui * (ui * ui - 1.0)


def _eval_all_zx_cells(
    log10_rho: jax.Array, log10_t: jax.Array, coeffs: jax.Array
) -> jax.Array:
    r"""Evaluate Stage 1 $(\log Q, \log T)$ bicubic spline across all $(Z, X)$.

    Args:
        log10_rho: Scalar `float64` $\log_{10}\rho$.
        log10_t: Scalar `float64` $\log_{10} T$.
        coeffs: Array of shape `(5, 6, 28, nq, nt, 4)`.

    Returns:
        Tensor of shape `(5, 6, 28)` giving all 28 channels at every $(Z, X)$
        table node.
    """
    lq_min, dq, nq = -2.08, 0.09, coeffs.shape[3]
    lt_min, dt, nt = 4.50, 0.06, coeffs.shape[4]
    raw_q = log10_rho - 2.0 * log10_t + 12.0
    iq, xp, xpi, cx, cxi = _uniform_axis_basis(raw_q, lq_min, dq, nq)
    jt, yp, ypi, cy, cyi = _uniform_axis_basis(log10_t, lt_min, dt, nt)

    f00, f01 = coeffs[:, :, :, iq, jt, :], coeffs[:, :, :, iq, jt + 1, :]
    f10, f11 = (
        coeffs[:, :, :, iq + 1, jt, :],
        coeffs[:, :, :, iq + 1, jt + 1, :],
    )
    sixth_hx2 = (dq * dq) / 6.0
    sixth_hy2 = (dt * dt) / 6.0
    z36th = (dq * dq * dt * dt) / 36.0

    return (
        xpi * (ypi * f00[..., 0] + yp * f01[..., 0])
        + xp * (ypi * f10[..., 0] + yp * f11[..., 0])
        + sixth_hx2
        * (
            cxi * (ypi * f00[..., 1] + yp * f01[..., 1])
            + cx * (ypi * f10[..., 1] + yp * f11[..., 1])
        )
        + sixth_hy2
        * (
            xpi * (cyi * f00[..., 2] + cy * f01[..., 2])
            + xp * (cyi * f10[..., 2] + cy * f11[..., 2])
        )
        + z36th
        * (
            cxi * (cyi * f00[..., 3] + cy * f01[..., 3])
            + cx * (cyi * f10[..., 3] + cy * f11[..., 3])
        )
    )


def _quad_3pt_uniform(
    dx: float, delta: jax.Array, y0: jax.Array, y1: jax.Array, y2: jax.Array
) -> jax.Array:
    r"""Evaluate uniform-grid 3-point quadratic Lagrange polynomial."""
    denom = 2.0 * dx * dx
    return (
        ((2.0 * dx * dx - 3.0 * dx * delta + delta * delta) / denom) * y0
        + (2.0 * (2.0 * dx - delta) * delta / denom) * y1
        + (delta * (delta - dx) / denom) * y2
    )


def _interp_x_for_iz(tbl_z: jax.Array, x: jax.Array, num_xs: int) -> jax.Array:
    r"""Stage 2 branchless interpolation in $X$ for a single $Z$ slice."""
    if num_xs == 1:
        return tbl_z[0]
    dx = 0.2
    q_lo = _quad_3pt_uniform(dx, x, tbl_z[0], tbl_z[1], tbl_z[2])
    q_hi = _quad_3pt_uniform(
        dx,
        x - 0.2 * (num_xs - 3),
        tbl_z[num_xs - 3],
        tbl_z[num_xs - 2],
        tbl_z[num_xs - 1],
    )
    ix_lo = jnp.where(x <= 0.4, 0, jnp.where(x <= 0.6, 1, 2))
    coef = (x - 0.2 * (ix_lo.astype(jnp.float64) + 1.0)) / dx
    c1 = -coef * (coef - 1.0) * (coef - 1.0) / 2.0
    c2 = (2.0 + coef * coef * (3.0 * coef - 5.0)) / 2.0
    c3 = (coef + coef * coef * (4.0 - 3.0 * coef)) / 2.0
    c4 = coef * coef * (coef - 1.0) / 2.0
    cub = (
        c1 * tbl_z[ix_lo]
        + c2 * tbl_z[ix_lo + 1]
        + c3 * tbl_z[ix_lo + 2]
        + c4 * tbl_z[ix_lo + 3]
    )
    x_hi_thresh = 0.2 * (num_xs - 2)
    return jnp.where(
        x < 1.0e-6,
        tbl_z[0],
        jnp.where(x <= 0.2, q_lo, jnp.where(x >= x_hi_thresh, q_hi, cub)),
    )


def _eval_ionization_single_jax(
    z_in: jax.Array,
    x_in: jax.Array,
    log10_rho: jax.Array,
    log10_t: jax.Array,
    coeffs: jax.Array,
) -> jax.Array:
    r"""Evaluate all 28 ionization channels for a single point in JAX."""
    z = jnp.where(z_in < 1.0e-20, 0.0, z_in)
    x = jnp.where(x_in < 1.0e-20, 0.0, x_in)
    zx_vals = _eval_all_zx_cells(log10_rho, log10_t, coeffs)
    r0 = _interp_x_for_iz(zx_vals[0], x, 6)
    r1 = _interp_x_for_iz(zx_vals[1], x, 5)
    r2 = _interp_x_for_iz(zx_vals[2], x, 5)
    r3 = _interp_x_for_iz(zx_vals[3], x, 5)
    r4 = _interp_x_for_iz(zx_vals[4], x, 1)

    dz = 0.02
    denom = 2.0 * dz * dz
    cz1 = (2.0 * dz * dz - 3.0 * dz * z + z * z) / denom
    cz2 = 2.0 * (2.0 * dz - z) * z / denom
    cz3 = z * (z - dz) / denom
    quad_z = cz1 * r0 + cz2 * r1 + cz3 * r2

    a23 = 0.5 * (1.0 - jnp.cos(jnp.pi * (z - 0.04) / (0.20 - 0.04)))
    cos_23 = (1.0 - a23) * r2 + a23 * r3

    a34 = 0.5 * (1.0 - jnp.cos(jnp.pi * (z - 0.20) / (1.00 - 0.20)))
    cos_34 = (1.0 - a34) * r3 + a34 * r4

    return jnp.where(
        z < 1.0e-6,
        r0,
        jnp.where(
            z <= 0.04,
            quad_z,
            jnp.where(
                z <= 0.20,
                cos_23,
                jnp.where(z <= 1.00 - 1.0e-6, cos_34, r4),
            ),
        ),
    )


_eval_ion_single_jit = jax.jit(_eval_ionization_single_jax)
_eval_ion_batch_jit = jax.jit(
    jax.vmap(_eval_ionization_single_jax, in_axes=(0, 0, 0, 0, None))
)


def eval_ionization_jax(
    z: jax.Array | float,
    x: jax.Array | float,
    log10_rho: jax.Array | float,
    log10_t: jax.Array | float,
) -> jax.Array:
    r"""Evaluate all 28 ionization channels in JAX (`eval_ionization`).

    Args:
        z: Metallicity mass fraction $Z$ of shape `()` or `(n_zones,)`.
        x: Hydrogen mass fraction $X$ of shape `()` or `(n_zones,)`.
        log10_rho: Base-10 log density $\log_{10}(\rho /
            \mathrm{g\,cm^{-3}})$.
        log10_t: Base-10 log temperature $\log_{10}(T / \mathrm{K})$.

    Returns:
        `float64` `jax.Array` of shape `(28,)` for scalar inputs or
        `(n_zones, 28)` for 1D inputs.
    """
    coeffs = _builtin_cell_coeffs_jax()
    z_j = jnp.asarray(z, dtype=jnp.float64)
    x_j = jnp.asarray(x, dtype=jnp.float64)
    lrho_j = jnp.asarray(log10_rho, dtype=jnp.float64)
    lt_j = jnp.asarray(log10_t, dtype=jnp.float64)
    if z_j.ndim == 0:
        out_1d: jax.Array = _eval_ion_single_jit(z_j, x_j, lrho_j, lt_j, coeffs)
        return out_1d
    out_2d: jax.Array = _eval_ion_batch_jit(z_j, x_j, lrho_j, lt_j, coeffs)
    return out_2d


def _resolve_log10_pair(
    lin: float | None, log10_val: float | None, name: str
) -> float:
    r"""Resolve `(linear, log10)` arguments into a finite `log10` value."""
    if lin is not None and (float(lin) <= 0.0 or not np.isfinite(lin)):
        raise ValueError(f"{name} must be positive and finite.")
    if log10_val is not None:
        lv = float(log10_val)
        if not np.isfinite(lv):
            raise ValueError(f"log10_{name} must be finite.")
        return lv
    if lin is not None:
        return float(np.log10(float(lin)))
    raise ValueError(f"Either {name} or log10_{name} must be provided.")


def eval_ionization(
    z: float,
    x: float,
    *,
    rho: float | None = None,
    log10_rho: float | None = None,
    t: float | None = None,
    log10_t: float | None = None,
    backend: _dispatch.Backend = "auto",
) -> IonizationResult:
    r"""Evaluate the 28-channel 4D ionization state (`eval_ionization`).

    Args:
        z: Metallicity mass fraction $Z$ (dimensionless).
        x: Hydrogen mass fraction $X$ (dimensionless).
        rho: Mass density $\rho$ in $\mathrm{g\,cm^{-3}}$ (optional if
            `log10_rho` is supplied).
        log10_rho: Base-10 log density $\log_{10}(\rho /
            \mathrm{g\,cm^{-3}})$.
        t: Temperature $T$ in $\mathrm{K}$ (optional if `log10_t` is supplied).
        log10_t: Base-10 log temperature $\log_{10}(T / \mathrm{K})$.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        `IonizationResult` wrapping the 28 output quantities.

    Raises:
        ValueError: If neither linear nor `log10` density/temperature is
            provided, or if inputs are non-positive or non-finite.
    """
    resolved = _dispatch.resolve_backend(backend, z, x)
    if resolved == "jax":
        zf, xf = float(z), float(x)
        if not (np.isfinite(zf) and np.isfinite(xf)):
            raise ValueError("z and x must be finite.")
        lt_f = _resolve_log10_pair(t, log10_t, "t")
        lrho_f = _resolve_log10_pair(rho, log10_rho, "rho")
        res_j = eval_ionization_jax(zf, xf, lrho_f, lt_f)
        return IonizationResult(np.asarray(res_j, dtype=np.float64))
    raw = _rust.py_ion_eval_ionization(
        float(z),
        float(x),
        None if rho is None else float(rho),
        None if log10_rho is None else float(log10_rho),
        None if t is None else float(t),
        None if log10_t is None else float(log10_t),
    )
    return IonizationResult(np.asarray(raw, dtype=np.float64))


def eval_ionization_zones(
    z: npt.NDArray[np.float64] | jax.Array,
    x: npt.NDArray[np.float64] | jax.Array,
    log10_rho: npt.NDArray[np.float64] | jax.Array,
    log10_t: npt.NDArray[np.float64] | jax.Array,
    *,
    parallel: bool = True,
    backend: _dispatch.Backend = "auto",
) -> npt.NDArray[np.float64] | jax.Array:
    r"""Evaluate all 28 ionization channels across `n_zones` stellar zones.

    Args:
        z: 1D array of metallicities $Z$ of shape `(n_zones,)`.
        x: 1D array of hydrogen mass fractions $X$ of shape `(n_zones,)`.
        log10_rho: 1D array of $\log_{10}(\rho / \mathrm{g\,cm^{-3}})$ of shape
            `(n_zones,)`.
        log10_t: 1D array of $\log_{10}(T / \mathrm{K})$ of shape `(n_zones,)`.
        parallel: When using `"rust"`, evaluate across the persistent 4-thread
            `rayon` worker pool when `n_zones >= 64`.
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        2D `float64` array of shape `(n_zones, 28)` in `ION_RESULT_NAMES`
        column order.

    Raises:
        ValueError: If input arrays are not 1D of equal length.
    """
    resolved = _dispatch.resolve_backend(backend, z, x, log10_rho, log10_t)
    if resolved == "jax":
        z_j = jnp.asarray(z, dtype=jnp.float64)
        x_j = jnp.asarray(x, dtype=jnp.float64)
        lrho_j = jnp.asarray(log10_rho, dtype=jnp.float64)
        lt_j = jnp.asarray(log10_t, dtype=jnp.float64)
        if z_j.ndim != 1 or not (
            x_j.shape == lrho_j.shape == lt_j.shape == z_j.shape
        ):
            raise ValueError(
                "z, x, log10_rho, and log10_t must be 1D arrays of equal shape."
            )
        return eval_ionization_jax(z_j, x_j, lrho_j, lt_j)
    z_np = _dispatch.as_c_f64(z)
    x_np = _dispatch.as_c_f64(x)
    lrho_np = _dispatch.as_c_f64(log10_rho)
    lt_np = _dispatch.as_c_f64(log10_t)
    if z_np.ndim != 1 or not (
        x_np.shape == lrho_np.shape == lt_np.shape == z_np.shape
    ):
        raise ValueError(
            "z, x, log10_rho, and log10_t must be 1D arrays of equal shape."
        )
    return _rust.py_ion_eval_ionization_zones(
        z_np, x_np, lrho_np, lt_np, bool(parallel)
    )
