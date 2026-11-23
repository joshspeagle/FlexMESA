# SPDX-License-Identifier: LGPL-3.0-or-later
"""Upstream `astero/test_suite/*` `set_param` routines from each case's
`run_star_extras.f90` (`#423`).

The search driver calls `set_param(snapshot, name, val)` for every named
search parameter of every track, in index order, before the model is loaded
or built (`astero/private/extras_support.f90:974-988`). `snapshot` holds the
`astero_def` values and the current `s% job% initial_*` fields; the returned
dictionary lists the assignments to make, in upstream order.
"""

from __future__ import annotations

import re
from typing import Any

from flexmesa import _rust

_SET_PARAM_RE = re.compile(r"subroutine set_param\(.*?end subroutine set_param")


def _search_set_param(
    snap: dict[str, Any], name: str, val: float
) -> dict[str, Any]:
    """`set_param` cases shared by every search case.

    `astero/test_suite/fast_newuoa/src/run_star_extras.f90:115-184`, identical
    in `example_astero` (`:115-184`), `fast_scan_grid` (`:121-190`) and
    `fast_simplex` (`:122-191`).

    Raises:
        ValueError: For a name the case does not handle (`ierr = -1`,
            `invalid name in set_param`).
    """
    upd: dict[str, Any] = {}
    if name == "initial_mass":
        upd["new_mass"] = val
        upd["relax_initial_mass"] = True
        upd["initial_mass"] = val
    elif name == "initial_Y":
        he3 = snap["Y_frac_he3"] * val
        upd["initial_he3"] = he3
        upd["initial_he4"] = val - he3
        upd["set_uniform_initial_composition"] = True
    elif name == "initial_FeH":
        # (Z/X) = (Z/X)sun * 10^[Fe/H]
        z_div_x = snap["Z_div_X_solar"] * _rust.exp10(val)
        if snap["Y_depends_on_Z"]:
            dydz, y0 = snap["dYdZ"], snap["Y0"]
            c = 1.0 + z_div_x * (1.0 + dydz)
            x = (1.0 - y0) / c
            y = (y0 + z_div_x * (dydz + y0)) / c
            he3 = snap["Y_frac_he3"] * y
            upd["initial_he3"] = he3
            upd["initial_he4"] = y - he3
        else:
            # Y from the initial_he3 an earlier initial_Y set.
            y = snap["initial_he3"] / snap["Y_frac_he3"]
            x = (1.0 - y) / (1.0 + z_div_x)
        upd["initial_h1"] = x
        upd["initial_h2"] = 0.0
        upd["set_uniform_initial_composition"] = True
    elif name == "alpha":
        upd["mixing_length_alpha"] = val
    elif name == "f_ov":
        if val > 0.0:
            upd["overshoot_scheme(1)"] = "exponential"
            upd["overshoot_zone_type(1)"] = "any"
            upd["overshoot_zone_loc(1)"] = "any"
            upd["overshoot_bdy_loc(1)"] = "any"
            upd["overshoot_f(1)"] = val
            upd["overshoot_f0(1)"] = snap["f0_ov_div_f_ov"] * val
        else:
            upd["overshoot_scheme(1)"] = ""
            upd["overshoot_zone_type(1)"] = ""
            upd["overshoot_zone_loc(1)"] = ""
            upd["overshoot_bdy_loc(1)"] = ""
            upd["overshoot_f(1)"] = 0.0
            upd["overshoot_f0(1)"] = 0.0
    else:
        raise ValueError(f"invalid name in set_param {name}")
    return upd


class SearchSetParamAsteroExtras:
    """`set_param` of `fast_newuoa`, `example_astero`, `fast_scan_grid` and
    `fast_simplex` (see `_search_set_param`)."""

    def set_param(
        self, snap: dict[str, Any], name: str, val: float
    ) -> dict[str, Any]:
        """Return the assignments `set_param(name, val)` makes."""
        return _search_set_param(snap, name, val)


class FastFromFileAsteroExtras:
    """`set_param` of fast_from_file: the shared cases plus `semiconv`,
    `thermohal` and `diffusion`.

    Upstream:
    `astero/test_suite/fast_from_file/src/run_star_extras.f90:122-199`
    (the three extra cases at `:185-193`).
    """

    def set_param(
        self, snap: dict[str, Any], name: str, val: float
    ) -> dict[str, Any]:
        """Return the assignments `set_param(name, val)` makes."""
        if name == "semiconv":
            return {"alpha_semiconvection": val}
        if name == "thermohal":
            return {"thermohaline_coeff": val}
        if name == "diffusion":
            return {"diffusion_class_factor(:)": val}
        return _search_set_param(snap, name, val)


# SHA-256 of the normalized `set_param` subroutine text (see
# `tests.case_extras.normalize_fortran`). The `set_param` of `astero_adipls`,
# `astero_gyre` and `surface_effects` is empty (`ierr = 0`) and those cases
# set `astero_just_call_my_extras_check_model`, so they get no handler.
_SEARCH_SET_PARAM_DIGEST = (
    "ac14ad22a14892d35288d639531e9e47efb717d24af7d5d5d69f9237866c8e1a"
)
_FAST_FROM_FILE_SET_PARAM_DIGEST = (
    "a9e686db34f01d5aeb10baa26a6c21509996e93a94445559fe54625357168a92"
)


def match_astero_extras(norm_star: str, digest: Any) -> Any:
    """Match an astero case's `set_param` from normalized Fortran source.

    Args:
        norm_star: Normalized `run_star_extras.f90` text.
        digest: Callable returning the SHA-256 hex digest of a string.

    Returns:
        A handler implementing `set_param`, or `None`.
    """
    m = _SET_PARAM_RE.search(norm_star)
    if m is None:
        return None
    d = digest(m.group(0))
    if d == _SEARCH_SET_PARAM_DIGEST:
        return SearchSetParamAsteroExtras()
    if d == _FAST_FROM_FILE_SET_PARAM_DIGEST:
        return FastFromFileAsteroExtras()
    return None
