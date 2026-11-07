# SPDX-License-Identifier: LGPL-3.0-or-later
"""Plasma electrostatic screening (`no_screening`, `salpeter`, `extended`).

Provides unified Safe Rust and JAX (`float64`) dispatchers for single-pair and
multi-zone plasma screening enhancement factors and $(T, \\rho)$ derivatives.
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
class ScreenEval:
    """Plasma electrostatic screening evaluation result.

    Attributes:
        scor: Screening enhancement factor $F_{\\mathrm{sc}} = \\exp(H_{12})$
            (dimensionless).
        dscor_dt: Temperature partial derivative
            $\\partial F_{\\mathrm{sc}} / \\partial T$ ($\\mathrm{K}^{-1}$).
        dscor_dd: Density partial derivative
            $\\partial F_{\\mathrm{sc}} / \\partial \\rho$
            ($\\mathrm{cm^3\\,g^{-1}}$).
    """

    scor: float
    dscor_dt: float
    dscor_dd: float

    def __iter__(self):  # type: ignore[no-untyped-def]
        """Iterate over `(scor, dscor_dt, dscor_dd)`."""
        yield self.scor
        yield self.dscor_dt
        yield self.dscor_dd

    def __getitem__(self, idx: int) -> float:
        """Index into `(scor, dscor_dt, dscor_dd)`."""
        return (self.scor, self.dscor_dt, self.dscor_dd)[idx]


def screen_pair(
    mode: str,
    temp: float | jax.Array,
    den: float | jax.Array,
    zbar: float | jax.Array,
    abar: float | jax.Array,
    z2bar: float | jax.Array,
    a1: float,
    z1: float,
    a2: float,
    z2: float,
    *,
    backend: _dispatch.Backend = "auto",
) -> ScreenEval | tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate plasma screening for reacting pair `(a1, z1)` + `(a2, z2)`.

    Args:
        mode: Screening mode (`"no_screening"`, `"salpeter"`, `"extended"`, or
            `"chugunov"`).
        temp: Temperature $T$ in $\\mathrm{K}$.
        den: Mass density $\\rho$ in $\\mathrm{g\\,cm^{-3}}$.
        zbar: Mean ionic charge $\\bar{Z}$.
        abar: Mean atomic weight $\\bar{A}$.
        z2bar: Mean square charge $\\overline{Z^2}$.
        a1: Reactant 1 atomic weight $A_1$ ($\\mathrm{g\\,mol^{-1}}$).
        z1: Reactant 1 atomic number $Z_1$.
        a2: Reactant 2 atomic weight $A_2$ ($\\mathrm{g\\,mol^{-1}}$).
        z2: Reactant 2 atomic number $Z_2$.
        backend: Execution backend (`"auto"`, `"rust"`, or `"jax"`).

    Returns:
        [`ScreenEval`] on `"rust"` backend, or `(scor, dscor_dt, dscor_dd)` JAX
        `float64` arrays on `"jax"` backend.
    """
    resolved = _dispatch.resolve_backend(backend, temp, den, zbar, abar, z2bar)
    m = mode.strip().lower()
    if resolved == "jax":
        t_j = jnp.asarray(temp, dtype=jnp.float64)
        d_j = jnp.asarray(den, dtype=jnp.float64)
        zb_j = jnp.asarray(zbar, dtype=jnp.float64)
        ab_j = jnp.asarray(abar, dtype=jnp.float64)
        z2b_j = jnp.asarray(z2bar, dtype=jnp.float64)
        if m in ("0", "none", "no_screening", "off"):
            z0 = jnp.zeros_like(t_j)
            return jnp.ones_like(t_j), z0, z0
        if m in ("1", "salpeter", "salpeter_screening"):
            s, dt, dd = jax_kernels.screen_salpeter_jax(
                t_j, d_j, zb_j, ab_j, z2b_j, jnp.asarray(z1), jnp.asarray(z2)
            )
            return s, dt, dd
        if m in ("2", "extended", "extended_screening"):
            s, dt, dd = jax_kernels.screen_extended_jax(
                t_j,
                d_j,
                zb_j,
                ab_j,
                z2b_j,
                jnp.asarray(a1),
                jnp.asarray(z1),
                jnp.asarray(a2),
                jnp.asarray(z2),
            )
            return s, dt, dd
        if m in ("3", "chugunov", "chugunov_screening"):
            s, dt, dd = jax_kernels.screen_chugunov_jax(
                t_j, d_j, zb_j, ab_j, jnp.asarray(z1), jnp.asarray(z2)
            )
            return s, dt, dd
        msg = f"Unknown screening mode: {mode}"
        raise ValueError(msg)

    scor, dscor_dt, dscor_dd = _rust.rates_screen_pair(
        mode,
        float(temp),
        float(den),
        float(zbar),
        float(abar),
        float(z2bar),
        float(a1),
        float(z1),
        float(a2),
        float(z2),
    )
    return ScreenEval(scor=scor, dscor_dt=dscor_dt, dscor_dd=dscor_dd)


def screen_pair_zones(
    mode: str,
    temps: npt.ArrayLike,
    dens: npt.ArrayLike,
    zbars: npt.ArrayLike,
    abars: npt.ArrayLike,
    z2bars: npt.ArrayLike,
    a1: float,
    z1: float,
    a2: float,
    z2: float,
) -> npt.NDArray[np.float64]:
    """Evaluate plasma screening across `N_zones` stellar zones in parallel.

    Args:
        mode: Screening mode (`"no_screening"`, `"salpeter"`, `"extended"`, or
            `"chugunov"`).
        temps: 1D array of temperatures `(N_zones,)` in $\\mathrm{K}$.
        dens: 1D array of densities `(N_zones,)` in $\\mathrm{g\\,cm^{-3}}$.
        zbars: 1D array of mean ionic charges `(N_zones,)`.
        abars: 1D array of mean atomic weights `(N_zones,)`.
        z2bars: 1D array of mean square charges `(N_zones,)`.
        a1: Reactant 1 atomic weight $A_1$.
        z1: Reactant 1 atomic number $Z_1$.
        a2: Reactant 2 atomic weight $A_2$.
        z2: Reactant 2 atomic number $Z_2$.

    Returns:
        2D `float64` NumPy array of shape `(N_zones, 3)` with columns
        `[scor, dscor_dt, dscor_dd]`.
    """
    return _rust.rates_screen_pair_zones(
        mode,
        np.ascontiguousarray(temps, dtype=np.float64),
        np.ascontiguousarray(dens, dtype=np.float64),
        np.ascontiguousarray(zbars, dtype=np.float64),
        np.ascontiguousarray(abars, dtype=np.float64),
        np.ascontiguousarray(z2bars, dtype=np.float64),
        float(a1),
        float(z1),
        float(a2),
        float(z2),
    )
