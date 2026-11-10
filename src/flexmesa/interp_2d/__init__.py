# SPDX-License-Identifier: LGPL-3.0-or-later
"""Two-dimensional bicubic, piecewise-monotonic, and Akima (1996) interpolation.

This subpackage ports upstream MESA ``interp_2d`` into ``float64`` NumPy/JAX
with differentiable bicubic evaluation (`jax.grad`, `jax.hessian`, `jax.vmap`):

- **C^2 Bicubic splines** (`interp_mkbicub_db`, `interp_evbicub_db`,
  `interp_evbicub_mask_db`, `herm2xy_db`, `interp_mkspline_db`, `splinck_db`):
  full 6-component evaluation of $(f, f_x, f_y, f_{xx}, f_{yy}, f_{xy})$ on
  rectilinear grids with configurable boundary conditions (`BicubicBc`).
- **2D Steffen (1990) piecewise-monotonic splines** (`interp_mkbipm_db`,
  `interp_evbipm_db`): tensor-product 1D monotonic interpolation along $x$
  followed by 4-point local monotonic evaluation along $y$.
- **Akima (1996, ACM Algorithm 760) bivariate surface fitting**
  (`interp_rgbi3p_db`, `interp_rgsf3p_db`, `rgpd3p_db`).
"""

from __future__ import annotations

from flexmesa.interp_2d import akima as _akima
from flexmesa.interp_2d import bicub as _bicub
from flexmesa.interp_2d import bipm as _bipm
from flexmesa.interp_2d import renka790 as _renka790

BicubicBc = _bicub.BicubicBc
BicubicEval = _bicub.BicubicEval
Cshep2Interpolant = _renka790.Cshep2Interpolant
Qshep2Interpolant = _renka790.Qshep2Interpolant
cs2hes_db = _renka790.cs2hes_db
herm2xy_db = _bicub.herm2xy_db
interp_cs2grd_db = _renka790.interp_cs2grd_db
interp_cs2val_db = _renka790.interp_cs2val_db
interp_cshep2_db = _renka790.interp_cshep2_db
interp_evbicub_db = _bicub.interp_evbicub_db
interp_evbicub_mask_db = _bicub.interp_evbicub_mask_db
interp_evbipm_db = _bipm.interp_evbipm_db
interp_mkbicub_db = _bicub.interp_mkbicub_db
interp_mkbipm_db = _bipm.interp_mkbipm_db
interp_mkspline_db = _bicub.interp_mkspline_db
interp_rgbi3p_db = _akima.interp_rgbi3p_db
interp_rgsf3p_db = _akima.interp_rgsf3p_db
mkspline_db = _bicub.mkspline_db
qs2grd = _renka790.qs2grd
qs2val = _renka790.qs2val
qshep2 = _renka790.qshep2
rgpd3p_db = _akima.rgpd3p_db
splinck_db = _bicub.splinck_db

__all__ = [
    "BicubicBc",
    "BicubicEval",
    "Cshep2Interpolant",
    "Qshep2Interpolant",
    "cs2hes_db",
    "herm2xy_db",
    "interp_cs2grd_db",
    "interp_cs2val_db",
    "interp_cshep2_db",
    "interp_evbicub_db",
    "interp_evbicub_mask_db",
    "interp_evbipm_db",
    "interp_mkbicub_db",
    "interp_mkbipm_db",
    "interp_mkspline_db",
    "interp_rgbi3p_db",
    "interp_rgsf3p_db",
    "mkspline_db",
    "qs2grd",
    "qs2val",
    "qshep2",
    "rgpd3p_db",
    "splinck_db",
]
