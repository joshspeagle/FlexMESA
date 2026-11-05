# SPDX-License-Identifier: LGPL-3.0-or-later
"""Two-dimensional bicubic, piecewise-monotonic, and Akima (1996) interpolation.

This subpackage ports upstream MESA ``interp_2d`` (`interp_mkbicub_db`,
`interp_evbicub_db`, `interp_mkspline_db`, `interp_mkbipm_db`,
`interp_evbipm_db`, `interp_rgbi3p_db`, `interp_rgsf3p_db`, and `rgpd3p_db`)
into ``float64`` NumPy/JAX with differentiable bicubic evaluation.
"""

from __future__ import annotations

from flexmesa.interp_2d import akima as _akima
from flexmesa.interp_2d import bicub as _bicub
from flexmesa.interp_2d import bipm as _bipm

BicubicBc = _bicub.BicubicBc
BicubicEval = _bicub.BicubicEval
herm2xy_db = _bicub.herm2xy_db
interp_evbicub_db = _bicub.interp_evbicub_db
interp_evbicub_mask_db = _bicub.interp_evbicub_mask_db
interp_evbipm_db = _bipm.interp_evbipm_db
interp_mkbicub_db = _bicub.interp_mkbicub_db
interp_mkbipm_db = _bipm.interp_mkbipm_db
interp_mkspline_db = _bicub.interp_mkspline_db
interp_rgbi3p_db = _akima.interp_rgbi3p_db
interp_rgsf3p_db = _akima.interp_rgsf3p_db
rgpd3p_db = _akima.rgpd3p_db
splinck_db = _bicub.splinck_db

__all__ = [
    "BicubicBc",
    "BicubicEval",
    "herm2xy_db",
    "interp_evbicub_db",
    "interp_evbicub_mask_db",
    "interp_evbipm_db",
    "interp_mkbicub_db",
    "interp_mkbipm_db",
    "interp_mkspline_db",
    "interp_rgbi3p_db",
    "interp_rgsf3p_db",
    "rgpd3p_db",
    "splinck_db",
]
