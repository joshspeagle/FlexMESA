# SPDX-License-Identifier: GPL-3.0-or-later
"""Upstream `astero/test_suite/*` `set_param` routines from each case's
`run_star_extras.f90` (`#423`).

The search driver calls `set_param(snapshot, name, val)` for every named
search parameter of every track, in index order, before the model is loaded
or built (`astero/private/extras_support.f90:974-988`). `snapshot` holds the
`astero_def` values and the current `s% job% initial_*` fields; the returned
dictionary lists the assignments to make, in upstream order.
"""

from __future__ import annotations

import math
import os
import pathlib
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

from flexmesa import _rust, constants

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


# `case default` of `set_constraint_value`: the adapter evaluates
# `star_get_history_output(s, name, ierr)` and fails on `ierr /= 0`.
STAR_GET_HISTORY_OUTPUT = "star_get_history_output"

# `convective_mixing` (`const/public/const_def.f90`).
_CONVECTIVE_MIXING = 1


def _search_set_constraint_value(snap: dict[str, Any], name: str) -> Any:
    """`set_constraint_value`, byte-identical in the five search cases.

    `astero/test_suite/fast_newuoa/src/run_star_extras.f90:58-112`, also
    `example_astero` (`:58-112`), `fast_scan_grid` (`:64-118`),
    `fast_simplex` (`:65-119`) and `fast_from_file` (`:65-119`).

    Returns:
        The constraint value, or `STAR_GET_HISTORY_OUTPUT` for the
        `case default` branch.

    Raises:
        ValueError: For `Rcz` when no zone matches: upstream leaves `val`
            (`intent(out)`) unset, which has no defined value
            (`DECISIONS #43`, deferred #438).
    """
    x = max(snap["surface_h1"], 1.0e-10)
    y = snap["surface_he3"] + snap["surface_he4"]
    z = max(1.0e-99, min(1.0, 1 - x - y))
    if name == "M_H":
        return _rust.log10(z / x / snap["Z_div_X_solar"])
    if name == "surface_Z_div_X":
        return z / x
    if name == "surface_He":
        return y
    if name == "Rcz":
        mixing_type, r, rsun = snap["mixing_type"], snap["r"], snap["Rsun"]
        # `do i = 1, s% nz-1`: locate the bottom of the solar convective zone.
        for i in range(len(r) - 1):
            if (
                mixing_type[i + 1] != _CONVECTIVE_MIXING
                and mixing_type[i] == _CONVECTIVE_MIXING
            ):
                if r[i + 1] > 0.25 * rsun and r[i] < 0.9 * rsun:
                    return r[i] / rsun
        raise ValueError(
            "set_constraint_value('Rcz'): no convective-zone base between "
            "0.25 and 0.9 Rsun, so upstream leaves val unset "
            "(astero/test_suite/fast_newuoa/src/run_star_extras.f90:96-105)"
        )
    return STAR_GET_HISTORY_OUTPUT


class SearchAsteroExtras:
    """`set_param` and `set_constraint_value` of `fast_newuoa`,
    `example_astero`, `fast_scan_grid` and `fast_simplex` (see
    `_search_set_param` and `_search_set_constraint_value`).

    Attributes:
        unported_routines: Names of the case's other extras routines that
            hold case logic this handler does not execute.
    """

    def __init__(self, unported_routines: tuple[str, ...] = ()) -> None:
        self.unported_routines = unported_routines

    def set_param(
        self, snap: dict[str, Any], name: str, val: float
    ) -> dict[str, Any]:
        """Return the assignments `set_param(name, val)` makes."""
        return _search_set_param(snap, name, val)

    def set_constraint_value(self, snap: dict[str, Any], name: str) -> Any:
        """Return `set_constraint_value(name)`."""
        return _search_set_constraint_value(snap, name)


# astero/test_suite/example_astero/src/run_star_extras.f90:199-227
_EXAMPLE_ASTERO_AFTER_EVOLVE_DIGEST = (
    "ef3e899136b487763e84db7254b9184e5a4760ce66031a79f2de2bb9b3bd54a8"
)


class ExampleAsteroExtras(SearchAsteroExtras):
    """Ported `extras_after_evolve` for `astero/example_astero`.

    Upstream:
    `astero/test_suite/example_astero/src/run_star_extras.f90:199-227`.
    """

    def __init__(self) -> None:
        super().__init__(unported_routines=())
        self.best_chi2: float | None = None
        self.termination_code_str: str | None = None
        self.testhub_extras: dict[str, float] = {}

    def extras_after_evolve(
        self, snap: dict[str, Any]
    ) -> dict[str, str] | None:
        """Check `best_chi2` bounds and set `t_xtra1` on `(1, 15)`.

        Args:
            snap: Post-evolve snapshot containing `best_chi2` and stellar
                surface state.

        Returns:
            `{"termination_code_slot": "t_xtra1", "termination_code_str":
            "good chi^2"}` when `1.0 < best_chi2 < 15.0`, or `None` otherwise
            after logging the diagnostic branch and recording
            `testhub_extras["chi2"] = best_chi2`.
        """
        best_chi2 = float(snap["best_chi2"])
        self.best_chi2 = best_chi2
        self.testhub_extras = {"chi2": best_chi2}
        self.termination_code_str = None
        print(f"best_chi2 {best_chi2}")
        if 1.0 < best_chi2 < 15.0:
            print("chi square within limit")
            self.termination_code_str = "good chi^2"
            return {
                "termination_code_slot": "t_xtra1",
                "termination_code_str": "good chi^2",
            }
        if best_chi2 <= 0.0:
            print("ERROR: failed to evaluate chi square")
        elif best_chi2 <= 1.0:
            print("bogus chi square ???  value too small??")
        else:
            print("ERROR: chi square too large")
        return None


class FastFromFileAsteroExtras(SearchAsteroExtras):
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


class EmptySetParamAsteroExtras:
    """Fallback `set_param` and `set_constraint_value` handler for unported
    non-search astero cases.

    `astero_adipls`, `astero_gyre`, and `surface_effects` share a
    `set_param` that only sets `ierr = 0`
    (`astero/test_suite/astero_adipls/src/run_star_extras.f90:86-96`,
    `astero_gyre` `:91-100`, `surface_effects` `:92-100`) and a
    `set_constraint_value` with only the `case default` branch that reads the
    history column of that name (`astero_adipls` `:58-83`, `astero_gyre`
    `:63-88`, `surface_effects` `:64-89`). All three cases now resolve to
    full star-hook handlers (`AsteroAdiplsExtras`, `AsteroGyreExtras`,
    `SurfaceEffectsExtras`), leaving this class as the fallback if another
    case shares those empty search routines while other routines remain
    unported.

    Attributes:
        unported_routines: Names of the case's other extras routines that
            hold case logic this handler does not execute.
    """

    def __init__(self, unported_routines: tuple[str, ...] = ()) -> None:
        self.unported_routines = unported_routines

    def set_param(
        self, snap: dict[str, Any], name: str, val: float
    ) -> dict[str, Any]:
        """Return the assignments `set_param(name, val)` makes: none."""
        del snap, name, val
        return {}

    def set_constraint_value(self, snap: dict[str, Any], name: str) -> Any:
        """Return `set_constraint_value(name)`: always the history column."""
        del snap, name
        return STAR_GET_HISTORY_OUTPUT


# Each routine of a `run_star_extras.f90`, from normalized source (see
# `tests.case_extras.normalize_fortran`).
_ROUTINE_RE = re.compile(
    r"(?:integer |logical |real\(dp\) )?(subroutine|function) (\w+)\(.*?"
    r"end \1 \2"
)

# SHA-256 of the normalized `set_param` subroutine.
_SEARCH_SET_PARAM_DIGEST = (
    "ac14ad22a14892d35288d639531e9e47efb717d24af7d5d5d69f9237866c8e1a"
)
_FAST_FROM_FILE_SET_PARAM_DIGEST = (
    "a9e686db34f01d5aeb10baa26a6c21509996e93a94445559fe54625357168a92"
)
# `set_constraint_value`, identical in the five search cases.
_SEARCH_SET_CONSTRAINT_VALUE_DIGEST = (
    "4ff0c53d1c35875075d88a074967cf93e99c756f7ba010b714b9446c0fc4be29"
)
# `set_param` (only `ierr = 0`) and `set_constraint_value` (only `case
# default`), identical in `astero_adipls`, `astero_gyre` and
# `surface_effects`.
_EMPTY_SET_PARAM_DIGEST = (
    "50db057770e0922de5b5f1f710d466230b4c55a0b9ca6061030f8391deb47fe4"
)
_DEFAULT_SET_CONSTRAINT_VALUE_DIGEST = (
    "e7ed93317a3605fd1e41d699c96dae5721ffa3a14a3cecf55e023cef10751e63"
)

# Routines read and found to hold no case logic: pointer set-up, the shared
# `test_suite_startup` / `test_suite_after_evolve` testhub and timing
# helpers, `keep_going`, and zero extra columns or header items
# (`astero/test_suite/fast_newuoa/src/run_star_extras.f90:38-55, 187-211,
# 214-297`; `astero/test_suite/fast_simplex/src/run_star_extras.f90:38-62,
# 322-377`).
_STOCK_ROUTINE_DIGESTS: dict[str, frozenset[str]] = {
    "extras_controls": frozenset(
        {
            "8c4491a51bdb7e6095c4a3aedcf0209de6ed66e7e15d115b09f7191072d59d3b",
            "1bb2eb4578bf6b02d8a40bb39eaa4f7601fa0130d5ede1a9f1c44e66bab2d68f",
        }
    ),
    "extras_startup": frozenset(
        {"945bc8b7e365406c7f0fd5b9586324b868fdc676346914f7394876aab22f30f6"}
    ),
    "extras_after_evolve": frozenset(
        {"46b2cec598812053b1b8bce308bba93fa4d7807fd850038773a78becb15f1b4b"}
    ),
    "extras_check_model": frozenset(
        {
            "0f8ffc1f59541485197e392a04c13581e4cb341d57c3ed32c7ac4659ae43d01d",
            "3c867eb2541d822c6cc5a98b2bd567dea1c7c46041d68251631cac261075796d",
        }
    ),
    "extras_finish_step": frozenset(
        {"68db5ddf03a81fa38d8d47f145337a68c8b64bc2f369ede17c3ff4c7bc1c940b"}
    ),
    "how_many_extra_history_columns": frozenset(
        {"90d8c21456f037612416160a5d81adaf65b71f9d8221fbc5481319c77573dbb3"}
    ),
    "data_for_extra_history_columns": frozenset(
        {"01288ef1cc03754d84fe38a8e7bb503ddceb5305aef9bab2f9adbd74d41db7ec"}
    ),
    "how_many_extra_profile_columns": frozenset(
        {"294e7adcf638b1d7511d2cda6b4c43a441fb2c2f9229210c52c2f19661ac7248"}
    ),
    "data_for_extra_profile_columns": frozenset(
        {"978902d5eebc99f1f38df1a5f1d46c13a57f66533c2b97350580801030de27d7"}
    ),
    "how_many_extra_history_header_items": frozenset(
        {"36dcb7dbcbf792f2fbdf8c3f64ddd00e9910316da76942a32b7d112158ba8736"}
    ),
    "data_for_extra_history_header_items": frozenset(
        {"7d4b010584bd8c9f5fda7536042e31a92d02d513574242f6d2aa3df9acb48a65"}
    ),
    "how_many_extra_profile_header_items": frozenset(
        {"eab88cc387da727099835c262f5b229be401824329952b39ab58ec977d89a634"}
    ),
    "data_for_extra_profile_header_items": frozenset(
        {"45e7b422878397546196a7a2c25d8b29cd75e1973536c9ed5201b1efb136b700"}
    ),
}


# `astero/test_suite/astero_gyre/src/run_star_extras.f90`: digests of the
# routines `AsteroGyreExtras` ports (normalized source, see
# `tests.case_extras.normalize_fortran`).
_ASTERO_GYRE_DIGESTS: dict[str, str] = {
    "extras_check_model": (
        "eaea7a0e2f96c48df19687bc6398a763b2acf74d6a0287de1eb27fad564805cd"
    ),
    "extras_after_evolve": (
        "02d78ef032b87691ce8e77035f532e06727e548b1b9e9455ef092f892b269961"
    ),
    "get_gyre_frequency_info": (
        "5070d937a88fde9888d78e5f12e35e9e761ad1f4ee9c76bd1ee03be13c624aa4"
    ),
    # Never called: with `astero_just_call_my_extras_check_model = .true.`
    # astero runs one track with nothing varied
    # (`astero/private/astero_run_support.f90:171-174`, `:239`), and the case
    # names no search parameters or constraints.
    "set_param": (
        "50db057770e0922de5b5f1f710d466230b4c55a0b9ca6061030f8391deb47fe4"
    ),
    "set_constraint_value": (
        "e7ed93317a3605fd1e41d699c96dae5721ffa3a14a3cecf55e023cef10751e63"
    ),
}


class AsteroGyreExtras:
    """`astero_gyre`'s `run_star_extras.f90` (`:103-228`), over astero's GYRE
    interface (`_rust.AsteroGyre`, `astero/private/gyre_support.f90`).

    The case sets `astero_just_call_my_extras_check_model`, so astero runs a
    plain star run, calls `init_gyre` at startup
    (`astero/private/astero_run_support.f90:109-115`) and hands every step to
    this `extras_check_model`. GYRE is called every 50 models and after the
    run, for radial modes; the run reports whether the `n_pg = 4` mode lies
    within 3% of `x_ctrl(1)`.
    """

    # Every snapshot carries a `star_model` handle for GYRE's `set_model`.
    wants_star_model = True

    def __init__(self) -> None:
        self.gyre: Any = None
        self.expected_freq = 0.0
        self.actual_freq = 0.0
        self.last_results: list[tuple[int, int, int, float, float, float]] = []
        # (model_number, results) of every GYRE call, for evidence scripts.
        self.calls: list[
            tuple[int, list[tuple[int, int, int, float, float, float]]]
        ] = []

    def extras_startup(self, state: dict[str, Any], _restart: bool) -> None:
        """`init_gyre(gyre_input_file)` for the run
        (`astero_run_support.f90:109-115`); the case's `extras_startup`
        (`:103-112`) only calls the output-only `test_suite_startup`."""
        mesa_dir = os.environ.get("MESA_DIR", "") or "mesa"
        self.gyre = _rust.AsteroGyre(str(state["case_dir"]), mesa_dir)

    def _get_gyre_frequency_info(
        self, state: dict[str, Any], check_match: bool
    ) -> bool:
        """`get_gyre_frequency_info(s, check_match, okay, ierr)`
        (`:183-228`)."""
        okay = False
        l_to_match = 0
        order_to_match = 4
        self.expected_freq = float(state["x_ctrl"][0])
        results = self.gyre.get_modes(state["star_model"], l_to_match, True)
        self.last_results = list(results)
        self.calls.append((int(state["model_number"]), list(results)))
        print()
        print(
            f"{'el':>8}{'order':>8}{'freq (microHz)':>20}"
            f"{'inertia':>20}{'growth rate (s)':>20}"
        )
        for _el, order, _em, inertia, freq, growth in results:
            print(
                f"{l_to_match:8d}{order:8d}{freq:20.10f}{inertia:20.10e}{growth:20.10e}"
            )
            if (
                check_match
                and self.expected_freq > 0
                and not okay
                and order == order_to_match
            ):
                self.actual_freq = freq
                okay = abs(self.actual_freq - self.expected_freq) < (
                    self.expected_freq * 3e-2
                )
        return okay

    def extras_check_model(self, state: dict[str, Any]) -> str:
        """`extras_check_model` (`:148-180`): GYRE every 50 models."""
        if float(state["x_ctrl"][0]) > 0.0:
            if int(state["model_number"]) % 50 != 0:
                return "keep_going"
            print("get gyre frequency info")
            self._get_gyre_frequency_info(state, False)
        return "keep_going"

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """`extras_after_evolve` (`:115-144`): final match against
        `x_ctrl(1)`; `test_suite_after_evolve` is output only."""
        if float(state["x_ctrl"][0]) > 0.0:
            okay = self._get_gyre_frequency_info(state, True)
            word = "got ok match" if okay else "ERROR: bad match"
            print(
                f"{word} for expected frequency"
                f"{self.actual_freq:20.2f}{self.expected_freq:20.2f}"
            )


# `astero/test_suite/astero_adipls/src/run_star_extras.f90`: digests of the
# routines `AsteroAdiplsExtras` accounts for (normalized source, see
# `tests.case_extras.normalize_fortran`).
_ASTERO_ADIPLS_DIGESTS: dict[str, str] = {
    "extras_check_model": (
        "ef310f57ef3f02e713e7a384d9d81fee78f4611763b7fd8a69e2f67e76f1a02a"
    ),
    "extras_after_evolve": (
        "7a744898fd7739fe9dd02992308848c1e745875ad82b618ba3e020884ec8bfc5"
    ),
    "get_adipls_frequency_info": (
        "6cabc9b6604361b8220a24a3efa14eec11dc552a4b400b065970cb83421813f3"
    ),
    # Never called: `use_other_adipls_mode_info = .false.` by default.
    "my_other_adipls_mode_info": (
        "59688e148a14901a14d93c9a961edf562b79bd242c77ef1b192a4a3b6c35a248"
    ),
    # Never called: with `astero_just_call_my_extras_check_model = .true.`
    # astero runs one track with nothing varied
    # (`astero/private/astero_run_support.f90:171-174`, `:239`), and the case
    # names no search parameters or constraints.
    "set_param": (
        "50db057770e0922de5b5f1f710d466230b4c55a0b9ca6061030f8391deb47fe4"
    ),
    "set_constraint_value": (
        "e7ed93317a3605fd1e41d699c96dae5721ffa3a14a3cecf55e023cef10751e63"
    ),
}


class AsteroAdiplsExtras:
    """`astero_adipls`'s `run_star_extras.f90` (`:99-282`), over astero's
    ADIPLS interface (`_rust.AsteroAdipls`,
    `astero/private/adipls_support.f90`).

    The case sets `astero_just_call_my_extras_check_model`, so astero runs a
    plain star run, initializes ADIPLS (including `redistrb.c.pruned.in` when
    `do_redistribute_mesh = .true.`, `astero_run_support.f90:117-124`) at
    startup and hands every step to `extras_check_model`. ADIPLS is called
    every 50 models and after the run for radial modes; the run reports whether
    the `n_pg = 4` mode lies within 3% of `x_ctrl(1)`.
    """

    # Every snapshot carries a `star_model` handle for `AsteroAdipls`.
    wants_star_model = True

    def __init__(self) -> None:
        self.adipls: Any = None
        self.expected_freq = 0.0
        self.actual_freq = 0.0
        self.okay = False
        self.last_results: list[tuple[int, int, int, float, float, float]] = []
        # `(model_number, results)` of every ADIPLS call.
        self.calls: list[
            tuple[int, list[tuple[int, int, int, float, float, float]]]
        ] = []

    def extras_startup(self, state: dict[str, Any], _restart: bool) -> None:
        """`run_adipls(s, first_time=.true., ...)` for the run
        (`astero_run_support.f90:117-124`); the case's `extras_startup`
        (`:111-120`) only calls the output-only `test_suite_startup`."""
        self.adipls = _rust.AsteroAdipls(str(state["case_dir"]))

    def _get_adipls_frequency_info(
        self,
        state: dict[str, Any],
        store_for_adipls: bool,
        l_to_match: int,
        order_to_match: int,
        expected_freq: float,
    ) -> tuple[float, bool]:
        """`get_adipls_frequency_info(...)` (`:215-281`)."""
        r_cm = constants.RSUN * float(state["photosphere_r"])
        g_cgs = float(state["cgrav"][0])
        m_g = constants.MSUN * float(state["star_mass"])
        results = self.adipls.get_one_el_info(
            state["star_model"],
            l_to_match,
            50.0,
            1000.0,
            200,
            r_cm,
            g_cgs,
            m_g,
            True,
            False,
            True,
            False,
            store_for_adipls,
        )
        self.last_results = list(results)
        self.calls.append((int(state["model_number"]), list(results)))
        print()
        print(f"{'el':>8}{'order':>8}{'freq (microHz)':>20}{'inertia':>20}")
        okay = expected_freq <= 0.0
        freq = 0.0
        for _el, order, _em, inertia, mode_freq, _growth in results:
            print(
                f"{l_to_match:8d}{order:8d}{mode_freq:20.10f}{inertia:20.10e}"
            )
            if expected_freq > 0.0 and order == order_to_match:
                freq = mode_freq
                okay = abs(freq - expected_freq) < (expected_freq * 3.0e-2)
        return freq, okay

    def extras_check_model(self, state: dict[str, Any]) -> str:
        """`extras_check_model` (`:170-212`): ADIPLS every 50 models."""
        if float(state["x_ctrl"][0]) > 0.0:
            if int(state["model_number"]) % 50 != 0:
                return "keep_going"
            self._get_adipls_frequency_info(
                state,
                store_for_adipls=True,
                l_to_match=0,
                order_to_match=5,
                expected_freq=-1.0,
            )
        return "keep_going"

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """`extras_after_evolve` (`:123-166`): final match against
        `x_ctrl(1)`; `test_suite_after_evolve` is output only."""
        if float(state["x_ctrl"][0]) > 0.0:
            self.expected_freq = float(state["x_ctrl"][0])
            freq, okay = self._get_adipls_frequency_info(
                state,
                store_for_adipls=True,
                l_to_match=0,
                order_to_match=4,
                expected_freq=self.expected_freq,
            )
            self.actual_freq = freq
            self.okay = okay
            word = "got ok match" if okay else "ERROR: bad match"
            print(
                f"{word} for expected frequency"
                f"{self.actual_freq:20.2f}{self.expected_freq:20.2f}"
            )


# `astero/test_suite/surface_effects/src/run_star_extras.f90` (with
# `src/targets.inc` inlined by `load_fortran_source`): digests of every
# non-stock routine `SurfaceEffectsExtras` accounts for.
_SURFACE_EFFECTS_DIGESTS: dict[str, str] = {
    # `extras_startup` (`:103-117`, including `src/targets.inc`).
    "extras_startup": (
        "d80b5d64a4cfd438f062796accd207272dff93d21e805142b00a40e20d698063"
    ),
    # `extras_after_evolve` and contained check helpers (`:120-330`).
    "extras_after_evolve": (
        "da6b3a33ad455a6c62bc986c888d114915f68f7e997de13d624dfb225de6572a"
    ),
    # `extras_check_model` (`:334-360`): dead `.false.` branch, returns
    # `keep_going`.
    "extras_check_model": (
        "e4d8e1a79ef6db17c9314f65b4ba9146a14f7b977024762202be0d48452ed9a1"
    ),
    # `astero_just_call_my_extras_check_model = .true.` runs no search hooks;
    # `_EMPTY_SET_PARAM_DIGEST` and `_DEFAULT_SET_CONSTRAINT_VALUE_DIGEST`.
    "set_param": (
        "50db057770e0922de5b5f1f710d466230b4c55a0b9ca6061030f8391deb47fe4"
    ),
    "set_constraint_value": (
        "e7ed93317a3605fd1e41d699c96dae5721ffa3a14a3cecf55e023cef10751e63"
    ),
}

_TARGETS_LINE_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+?)\s*$")
_REQUIRED_TARGETS = (
    "target_a3",
    "target_b1",
    "target_b3",
    "target_p0",
    "target_p1",
    "target_s0",
    "target_s1",
    "target_a_div_r",
)
_PERFECT_TOL = 1.0e-12
_SOLAR_TOL = 10.0


def _parse_targets_inc(path: pathlib.Path) -> dict[str, float]:
    """Parse `astero/test_suite/surface_effects/src/targets.inc`."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing targets file: {path}")
    targets: dict[str, float] = {}
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for lineno, raw_line in enumerate(raw_lines, start=1):
        code = raw_line.split("!", 1)[0].strip()
        if not code:
            continue
        match = _TARGETS_LINE_RE.match(code)
        if match is None:
            raise ValueError(f"Malformed line {lineno} in {path}: {raw_line!r}")
        key = match.group(1).lower()
        if key not in _REQUIRED_TARGETS:
            raise ValueError(f"Unexpected target key {key!r} in {path}")
        if key in targets:
            raise ValueError(f"Duplicate target key {key!r} in {path}")
        val = float(_rust.parse_fortran_float(match.group(2).strip()))
        if not math.isfinite(val):
            raise ValueError(
                f"Non-finite target value for {key!r} in {path}: {val!r}"
            )
        targets[key] = val
    missing = [key for key in _REQUIRED_TARGETS if key not in targets]
    if missing:
        raise ValueError(f"Missing required targets in {path}: {missing}")
    return targets


def _check_okay(name: str, val: float, tgt: float, tol: float) -> bool:
    """Upstream `check_okay`
    (`astero/test_suite/surface_effects/src/run_star_extras.f90:290-302`)."""
    rel_err = val / tgt - 1.0
    passed = math.isfinite(rel_err) and abs(rel_err) < tol
    tag = "" if passed else "    FAILED"
    print(f"{name:>26}{val:26.16E}{tgt:26.16E}{rel_err:26.16E}{tag}")
    return passed


def _check_negative(name: str, val: float) -> bool:
    """Upstream `check_negative`
    (`astero/test_suite/surface_effects/src/run_star_extras.f90:304-315`)."""
    if not math.isfinite(val) or val > 0.0:
        print(f"{name:>26}{val:26.16E}{'<= 0':>26}{'FAILED':>36}")
        return False
    return True


def _check_positive(name: str, val: float) -> bool:
    """Upstream `check_positive`
    (`astero/test_suite/surface_effects/src/run_star_extras.f90:317-328`)."""
    if not math.isfinite(val) or val < 0.0:
        print(f"{name:>26}{val:26.16E}{'>= 0':>26}{'FAILED':>36}")
        return False
    return True


class SurfaceEffectsExtras:
    """`astero/test_suite/surface_effects/src/run_star_extras.f90:103-360`
    over Safe Rust `AsteroAdipls` and surface corrections (`#569`).

    The case sets `astero_just_call_my_extras_check_model = .true.`, so astero
    runs a single `star` evolution and invokes `extras_startup`,
    `extras_check_model`, and `extras_after_evolve`. After evolution,
    `extras_after_evolve` computes `l = 0..3` ADIPLS modes, updates
    `delta_nu_model` from `l = 0` modes via OLS, matches observed targets,
    writes `freqs.dat` into the run's working directory (`state["case_dir"]`),
    and evaluates both perfect-data parameter recovery checks (`< 1e-12`) and
    solar-data surface correction checks against `src/targets.inc`.
    """

    wants_star_model = True

    def __init__(self) -> None:
        self.adipls: Any = None
        self.ctrl: dict[str, Any] = {}
        self.targets: dict[str, float] = {}
        self.freqs_dat_path: pathlib.Path | None = None
        self.raw_modes: dict[
            int, list[tuple[int, int, int, float, float, float]]
        ] = {}
        self.matched_modes: dict[
            int,
            tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int32]],
        ] = {}
        self.delta_nu_model = 0.0
        self.perfect_results: dict[str, float] = {}
        self.solar_results: dict[str, float] = {}
        self.okay = False

    def extras_startup(self, state: dict[str, Any], _restart: bool) -> None:
        """Initialize controls, ADIPLS solver, and `src/targets.inc`."""
        case_dir = pathlib.Path(str(state["case_dir"]))
        self.ctrl = _rust.astero_parse_search_controls(str(case_dir))
        self.adipls = _rust.AsteroAdipls(str(case_dir))
        self.targets = _parse_targets_inc(case_dir / "src" / "targets.inc")
        self.freqs_dat_path = None
        self.raw_modes = {}
        self.matched_modes = {}
        self.delta_nu_model = 0.0
        self.perfect_results = {}
        self.solar_results = {}
        self.okay = False

    def extras_check_model(self, state: dict[str, Any]) -> str:
        """Upstream `extras_check_model` (`:334-360`): always `keep_going`."""
        del state
        return "keep_going"

    def _extract_and_match_modes(
        self,
        star_model: Any,
        l0_orders: NDArray[np.float64],
    ) -> None:
        """Extract `l = 0..3` ADIPLS modes and match observed frequencies."""
        nl: list[int] = [int(v) for v in self.ctrl["nl"]]
        freq_target: list[list[float]] = self.ctrl["freq_target"]
        freq_sigma: list[list[float]] = self.ctrl["freq_sigma"]
        nu_lo = float(self.ctrl["nu_lower_factor"])
        nu_hi = float(self.ctrl["nu_upper_factor"])
        iscan_factor: list[float] = [
            float(v) for v in self.ctrl["iscan_factor"]
        ]

        for el in range(4):
            nl_el = nl[el]
            if nl_el <= 0:
                raise ValueError(f"Expected nl[{el}] > 0, got {nl_el}")
            if len(freq_target[el]) < nl_el or len(freq_sigma[el]) < nl_el:
                raise ValueError(
                    f"freq_target/freq_sigma too short for nl[{el}]={nl_el}"
                )
            obs_f = np.asarray(freq_target[el][:nl_el], dtype=np.float64)
            nu1 = nu_lo * float(obs_f[0])
            nu2 = nu_hi * float(obs_f[-1])
            iscan = round(iscan_factor[el]) * nl_el
            if iscan < 1:
                raise ValueError(f"Invalid iscan={iscan} for el={el}")
            dnu_arg = 0.0 if el == 0 else self.delta_nu_model
            raw = self.adipls.get_astero_one_el_info(
                star_model, el, nu1, nu2, iscan, dnu_arg, True
            )
            if not raw:
                raise RuntimeError(f"failed to find l={el} modes")
            self.raw_modes[el] = list(raw)
            cand_orders = np.asarray([row[1] for row in raw], dtype=np.int32)
            cand_inertias = np.asarray(
                [row[3] for row in raw], dtype=np.float64
            )
            cand_freqs = np.asarray([row[4] for row in raw], dtype=np.float64)
            mf, mi, mo = _rust.astero_match_modes_by_degree(
                obs_f, cand_freqs, cand_inertias, cand_orders
            )
            if len(mf) != nl_el or len(mi) != nl_el or len(mo) != nl_el:
                raise RuntimeError(
                    f"failed to match frequencies for l={el}: "
                    f"matched {len(mf)} of {nl_el}"
                )
            self.matched_modes[el] = (mf, mi, mo)
            if el == 0:
                self._update_radial_delta_nu_model(obs_f, mf, l0_orders)

    def _update_radial_delta_nu_model(
        self,
        obs_f: NDArray[np.float64],
        mf: NDArray[np.float64],
        l0_orders: NDArray[np.float64],
    ) -> None:
        """Update `delta_nu_model` via OLS with upstream guards
        (`astero/private/astero_support.f90:182-207`, `:677-687`)."""
        corr_fac = float(self.ctrl["correction_factor"])
        nl_0 = int(self.ctrl["nl"][0])
        delta_nu_obs = float(self.ctrl["delta_nu"])
        nu_max = float(self.ctrl["nu_max"])
        b_pow = float(self.ctrl["correction_b"])
        if corr_fac <= 0.0 or nl_0 <= 0 or delta_nu_obs <= 0.0 or nu_max <= 0.0:
            return
        valid = obs_f >= 0.0
        cnt = int(np.count_nonzero(valid))
        if cnt == 0:
            return
        avg_nu_obs = float(np.sum(obs_f[valid])) / float(cnt)
        if avg_nu_obs <= 0.0:
            return
        avg_radial_n = float(np.sum(l0_orders[valid])) / float(cnt)
        avg_nu_model = float(np.sum(mf[valid])) / float(cnt)
        sum_1 = 0.0
        sum_2 = 0.0
        sum_3 = 0.0
        for f_obs_val, f_mod_val, n_val in zip(
            obs_f[valid], mf[valid], l0_orders[valid], strict=True
        ):
            dn = float(n_val) - avg_radial_n
            sum_1 += (float(f_mod_val) - avg_nu_model) * dn
            sum_2 += dn * dn
            sum_3 += float(_rust.pow(float(f_obs_val) / nu_max, b_pow))
        if sum_2 == 0.0 or sum_3 == 0.0:
            return
        dnu_mod = sum_1 / sum_2
        if not math.isfinite(dnu_mod):
            raise RuntimeError(f"Invalid radial delta_nu_model={dnu_mod}")
        self.delta_nu_model = dnu_mod

    def _write_freqs_dat(self, case_dir: pathlib.Path) -> pathlib.Path:
        """Write `freqs.dat` matching
        `astero/test_suite/surface_effects/src/run_star_extras.f90:163-178`."""
        nl: list[int] = [int(v) for v in self.ctrl["nl"]]
        freq_target: list[list[float]] = self.ctrl["freq_target"]
        freq_sigma: list[list[float]] = self.ctrl["freq_sigma"]
        out_path = case_dir / "freqs.dat"
        lines = [
            f"{'l':>5}{'obs':>26}{'obs_sigma':>26}{'freq':>26}{'inertia':>26}\n"
        ]
        for el in range(4):
            mf, mi, _mo = self.matched_modes[el]
            for i in range(nl[el]):
                lines.append(
                    f"{el:5d}"
                    f"{float(freq_target[el][i]):26.16E}"
                    f"{float(freq_sigma[el][i]):26.16E}"
                    f"{float(mf[i]):26.16E}"
                    f"{float(mi[i]):26.16E}\n"
                )
        out_path.write_text("".join(lines), encoding="utf-8")
        return out_path

    def _run_perfect_checks(
        self,
        all_obs_s: NDArray[np.float64],
        all_mod_f: NDArray[np.float64],
        all_mod_i: NDArray[np.float64],
        l0_obs_f: NDArray[np.float64],
        l0_mf: NDArray[np.float64],
        l0_mi: NDArray[np.float64],
        l0_orders: NDArray[np.float64],
    ) -> bool:
        """Run perfect-data checks
        (`astero/test_suite/surface_effects/src/run_star_extras.f90:180-245`)."""
        dnu_obs = float(self.ctrl["delta_nu"])
        nu_max = float(self.ctrl["nu_max"])
        b_pow = float(self.ctrl["correction_b"])
        corr_fac = float(self.ctrl["correction_factor"])

        print()
        print("comparing to perfect data")
        print()
        print(
            f"{'parameter':>26}{'result':>26}{'target':>26}"
            f"{'result/target-1':>26}"
        )
        print()

        # Kjeldsen left commented out in upstream `:189-192`.
        okay = True

        tgt_a3 = self.targets["target_a3"]
        all_mod_f3 = all_mod_f * all_mod_f * all_mod_f
        test_cubic = all_mod_f + tgt_a3 * all_mod_f3 / all_mod_i
        res_cubic = _rust.astero_eval_surface_correction(
            "cubic",
            test_cubic,
            all_obs_s,
            all_mod_f,
            all_mod_i,
            l0_obs_f,
            l0_mf,
            l0_mi,
            l0_orders,
            dnu_obs,
            nu_max,
            b_pow,
            corr_fac,
        )
        a3 = float(res_cubic["p0"])
        okay = _check_okay("a3", a3, tgt_a3, _PERFECT_TOL) and okay

        tgt_b3 = self.targets["target_b3"]
        tgt_b1 = self.targets["target_b1"]
        test_comb = (
            all_mod_f + (tgt_b1 / all_mod_f + tgt_b3 * all_mod_f3) / all_mod_i
        )
        res_comb = _rust.astero_eval_surface_correction(
            "combined",
            test_comb,
            all_obs_s,
            all_mod_f,
            all_mod_i,
            l0_obs_f,
            l0_mf,
            l0_mi,
            l0_orders,
            dnu_obs,
            nu_max,
            b_pow,
            corr_fac,
        )
        b3 = float(res_comb["p0"])
        b1 = float(res_comb["p1"])
        okay = _check_okay("b3", b3, tgt_b3, _PERFECT_TOL) and okay
        okay = _check_okay("b1", b1, tgt_b1, _PERFECT_TOL) and okay

        tgt_p0 = self.targets["target_p0"]
        tgt_p1 = self.targets["target_p1"]
        pl_vals: list[float] = []
        for el in range(4):
            mf, mi, _mo = self.matched_modes[el]
            for f_val, i_val in zip(mf, mi, strict=True):
                f_m = float(f_val)
                shift = tgt_p0 * _rust.pow(f_m / nu_max, tgt_p1)
                if el > 0:
                    e0 = _rust.astero_interpolate_l0_inertia(f_m, l0_mf, l0_mi)
                    shift *= e0 / float(i_val)
                pl_vals.append(f_m + shift)
        test_pl = np.asarray(pl_vals, dtype=np.float64)
        res_pl = _rust.astero_eval_surface_correction(
            "power_law",
            test_pl,
            all_obs_s,
            all_mod_f,
            all_mod_i,
            l0_obs_f,
            l0_mf,
            l0_mi,
            l0_orders,
            dnu_obs,
            nu_max,
            b_pow,
            corr_fac,
        )
        p0 = float(res_pl["p0"])
        p1 = float(res_pl["p1"])
        okay = _check_okay("p0", p0, tgt_p0, _PERFECT_TOL) and okay
        okay = _check_okay("p1", p1, tgt_p1, _PERFECT_TOL) and okay

        tgt_s0 = self.targets["target_s0"]
        tgt_s1 = self.targets["target_s1"]
        sonoi_vals: list[float] = []
        for el in range(4):
            mf, mi, _mo = self.matched_modes[el]
            for f_val, i_val in zip(mf, mi, strict=True):
                f_m = float(f_val)
                shift = (
                    tgt_s0
                    * nu_max
                    * (1.0 - 1.0 / (1.0 + _rust.pow(f_m / nu_max, tgt_s1)))
                )
                if el > 0:
                    e0 = _rust.astero_interpolate_l0_inertia(f_m, l0_mf, l0_mi)
                    shift *= e0 / float(i_val)
                sonoi_vals.append(f_m + shift)
        test_sonoi = np.asarray(sonoi_vals, dtype=np.float64)
        res_sonoi = _rust.astero_eval_surface_correction(
            "sonoi",
            test_sonoi,
            all_obs_s,
            all_mod_f,
            all_mod_i,
            l0_obs_f,
            l0_mf,
            l0_mi,
            l0_orders,
            dnu_obs,
            nu_max,
            b_pow,
            corr_fac,
        )
        s0 = float(res_sonoi["p0"])
        s1 = float(res_sonoi["p1"])
        okay = _check_okay("s0", s0, tgt_s0, _PERFECT_TOL) and okay
        okay = _check_okay("s1", s1, tgt_s1, _PERFECT_TOL) and okay

        self.perfect_results = {
            "a3": a3,
            "b3": b3,
            "b1": b1,
            "p0": p0,
            "p1": p1,
            "s0": s0,
            "s1": s1,
        }
        return okay

    def _run_solar_checks(
        self,
        all_obs_f: NDArray[np.float64],
        all_obs_s: NDArray[np.float64],
        all_mod_f: NDArray[np.float64],
        all_mod_i: NDArray[np.float64],
        l0_obs_f: NDArray[np.float64],
        l0_obs_s: NDArray[np.float64],
        l0_mf: NDArray[np.float64],
        l0_mi: NDArray[np.float64],
        l0_orders: NDArray[np.float64],
        okay: bool,
    ) -> bool:
        """Run solar-data checks
        (`astero/test_suite/surface_effects/src/run_star_extras.f90:247-285`)."""
        dnu_obs = float(self.ctrl["delta_nu"])
        nu_max = float(self.ctrl["nu_max"])
        b_pow = float(self.ctrl["correction_b"])
        corr_fac = float(self.ctrl["correction_factor"])

        print()
        print("comparing to solar data")
        print()
        print(
            f"{'parameter':>26}{'result':>26}{'target':>26}"
            f"{'result/target-1':>26}"
        )
        print()

        res_kj = _rust.astero_eval_surface_correction(
            "kjeldsen",
            l0_obs_f,
            l0_obs_s,
            l0_mf,
            l0_mi,
            l0_obs_f,
            l0_mf,
            l0_mi,
            l0_orders,
            dnu_obs,
            nu_max,
            b_pow,
            corr_fac,
        )
        a_div_r = float(res_kj["p0"])
        okay = (
            _check_okay(
                "a_div_r", a_div_r, self.targets["target_a_div_r"], _SOLAR_TOL
            )
            and okay
        )

        res_cubic = _rust.astero_eval_surface_correction(
            "cubic",
            all_obs_f,
            all_obs_s,
            all_mod_f,
            all_mod_i,
            l0_obs_f,
            l0_mf,
            l0_mi,
            l0_orders,
            dnu_obs,
            nu_max,
            b_pow,
            corr_fac,
        )
        a3_sol = float(res_cubic["p0"])
        _check_okay("a3", a3_sol, self.targets["target_a3"], _SOLAR_TOL)
        okay = _check_negative("a3", a3_sol) and okay

        res_comb = _rust.astero_eval_surface_correction(
            "combined",
            all_obs_f,
            all_obs_s,
            all_mod_f,
            all_mod_i,
            l0_obs_f,
            l0_mf,
            l0_mi,
            l0_orders,
            dnu_obs,
            nu_max,
            b_pow,
            corr_fac,
        )
        b3_sol = float(res_comb["p0"])
        b1_sol = float(res_comb["p1"])
        _check_okay("b3", b3_sol, self.targets["target_b3"], _SOLAR_TOL)
        okay = _check_negative("b3", b3_sol) and okay
        _check_okay("b1", b1_sol, self.targets["target_b1"], _SOLAR_TOL)

        res_pl = _rust.astero_eval_surface_correction(
            "power_law",
            all_obs_f,
            all_obs_s,
            all_mod_f,
            all_mod_i,
            l0_obs_f,
            l0_mf,
            l0_mi,
            l0_orders,
            dnu_obs,
            nu_max,
            b_pow,
            corr_fac,
        )
        p0_sol = float(res_pl["p0"])
        p1_sol = float(res_pl["p1"])
        _check_okay("p0", p0_sol, self.targets["target_p0"], _SOLAR_TOL)
        okay = _check_negative("p0", p0_sol) and okay
        _check_okay("p1", p1_sol, self.targets["target_p1"], _SOLAR_TOL)

        res_sonoi = _rust.astero_eval_surface_correction(
            "sonoi",
            all_obs_f,
            all_obs_s,
            all_mod_f,
            all_mod_i,
            l0_obs_f,
            l0_mf,
            l0_mi,
            l0_orders,
            dnu_obs,
            nu_max,
            b_pow,
            corr_fac,
        )
        s0_sol = float(res_sonoi["p0"])
        s1_sol = float(res_sonoi["p1"])
        _check_okay("s0", s0_sol, self.targets["target_s0"], _SOLAR_TOL)
        okay = _check_negative("s0", s0_sol) and okay
        _check_okay("s1", s1_sol, self.targets["target_s1"], _SOLAR_TOL)
        okay = _check_positive("s1", s1_sol) and okay

        self.solar_results = {
            "a_div_r": a_div_r,
            "a3": a3_sol,
            "b3": b3_sol,
            "b1": b1_sol,
            "p0": p0_sol,
            "p1": p1_sol,
            "s0": s0_sol,
            "s1": s1_sol,
        }
        return okay

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """Extract `l = 0..3` modes, write `freqs.dat`, and check fits.

        Upstream `test_suite_after_evolve` (`:286`) is testhub/timing output
        only.
        """
        if self.adipls is None or not self.ctrl or not self.targets:
            raise RuntimeError(
                "SurfaceEffectsExtras.extras_startup must run before "
                "extras_after_evolve"
            )
        osc_code = str(self.ctrl["oscillation_code"]).strip().lower()
        if osc_code != "adipls":
            raise ValueError(
                "SurfaceEffectsExtras requires oscillation_code='adipls', "
                f"got {osc_code!r}"
            )
        nl: list[int] = [int(v) for v in self.ctrl["nl"]]
        if len(nl) != 4:
            raise ValueError(f"Expected 4 entries in nl, got {len(nl)}")
        l0_n_raw = [float(v) for v in self.ctrl["l0_n_obs"]]
        if len(l0_n_raw) < nl[0]:
            raise ValueError(
                f"l0_n_obs length ({len(l0_n_raw)}) < nl[0] ({nl[0]})"
            )
        l0_orders = np.asarray(l0_n_raw[: nl[0]], dtype=np.float64)
        if not np.all(np.isfinite(l0_orders)) or bool(np.any(l0_orders <= 0.0)):
            raise ValueError(
                "l0_n_obs must contain positive finite orders for all "
                f"nl[0]={nl[0]} radial modes"
            )

        self.raw_modes = {}
        self.matched_modes = {}
        self.delta_nu_model = 0.0
        self._extract_and_match_modes(state["star_model"], l0_orders)

        case_dir = pathlib.Path(str(state["case_dir"]))
        self.freqs_dat_path = self._write_freqs_dat(case_dir)

        freq_target: list[list[float]] = self.ctrl["freq_target"]
        freq_sigma: list[list[float]] = self.ctrl["freq_sigma"]
        all_obs_f = np.concatenate(
            [
                np.asarray(freq_target[el][: nl[el]], dtype=np.float64)
                for el in range(4)
            ]
        )
        all_obs_s = np.concatenate(
            [
                np.asarray(freq_sigma[el][: nl[el]], dtype=np.float64)
                for el in range(4)
            ]
        )
        all_mod_f = np.concatenate(
            [self.matched_modes[el][0] for el in range(4)]
        )
        all_mod_i = np.concatenate(
            [self.matched_modes[el][1] for el in range(4)]
        )
        l0_obs_f = np.asarray(freq_target[0][: nl[0]], dtype=np.float64)
        l0_obs_s = np.asarray(freq_sigma[0][: nl[0]], dtype=np.float64)
        l0_mf, l0_mi, _l0_mo = self.matched_modes[0]

        okay = self._run_perfect_checks(
            all_obs_s,
            all_mod_f,
            all_mod_i,
            l0_obs_f,
            l0_mf,
            l0_mi,
            l0_orders,
        )
        okay = self._run_solar_checks(
            all_obs_f,
            all_obs_s,
            all_mod_f,
            all_mod_i,
            l0_obs_f,
            l0_obs_s,
            l0_mf,
            l0_mi,
            l0_orders,
            okay,
        )
        self.okay = okay
        print()
        if okay:
            print("all tests are within tolerances")
        print()
        # Upstream `call test_suite_after_evolve(s, ierr)`
        # (`astero/test_suite/surface_effects/src/run_star_extras.f90:286`,
        # `star/job/test_suite_extras.inc:94-107`) writes testhub/timing
        # output only with no physical state or termination effect, matching
        # `AsteroGyreExtras` and `AsteroAdiplsExtras`.


def match_astero_extras(norm_star: str, digest: Any) -> Any:
    """Match an astero case's extras from normalized Fortran source.

    A handler is returned when the case's `set_param` and
    `set_constraint_value` are ported. Every other routine whose digest is
    not a reviewed no-logic template is listed in the handler's
    `unported_routines`, so the ledger keeps the case `unported`.

    Args:
        norm_star: Normalized `run_star_extras.f90` text.
        digest: Callable returning the SHA-256 hex digest of a string.

    Returns:
        A handler implementing `set_param` and `set_constraint_value`, or
        `None`.
    """
    routines = {
        m.group(2): digest(m.group(0)) for m in _ROUTINE_RE.finditer(norm_star)
    }
    if all(
        routines.get(name) == d for name, d in _ASTERO_ADIPLS_DIGESTS.items()
    ) and all(
        name in _ASTERO_ADIPLS_DIGESTS
        or d in _STOCK_ROUTINE_DIGESTS.get(name, frozenset())
        for name, d in routines.items()
    ):
        return AsteroAdiplsExtras()
    if all(
        routines.get(name) == d for name, d in _ASTERO_GYRE_DIGESTS.items()
    ) and all(
        name in _ASTERO_GYRE_DIGESTS
        or d in _STOCK_ROUTINE_DIGESTS.get(name, frozenset())
        for name, d in routines.items()
    ):
        return AsteroGyreExtras()
    if all(
        routines.get(name) == d for name, d in _SURFACE_EFFECTS_DIGESTS.items()
    ) and all(
        name in _SURFACE_EFFECTS_DIGESTS
        or d in _STOCK_ROUTINE_DIGESTS.get(name, frozenset())
        for name, d in routines.items()
    ):
        return SurfaceEffectsExtras()
    unported = tuple(
        name
        for name, d in routines.items()
        if name not in ("set_param", "set_constraint_value")
        and d not in _STOCK_ROUTINE_DIGESTS.get(name, frozenset())
    )
    set_param = routines.get("set_param")
    set_constraint_value = routines.get("set_constraint_value")
    if (
        set_param == _EMPTY_SET_PARAM_DIGEST
        and set_constraint_value == _DEFAULT_SET_CONSTRAINT_VALUE_DIGEST
    ):
        return EmptySetParamAsteroExtras(unported)
    if set_constraint_value != _SEARCH_SET_CONSTRAINT_VALUE_DIGEST:
        return None
    if set_param == _SEARCH_SET_PARAM_DIGEST:
        if (
            unported == ("extras_after_evolve",)
            and routines.get("extras_after_evolve")
            == _EXAMPLE_ASTERO_AFTER_EVOLVE_DIGEST
        ):
            return ExampleAsteroExtras()
        return SearchAsteroExtras(unported)
    if set_param == _FAST_FROM_FILE_SET_PARAM_DIGEST:
        return FastFromFileAsteroExtras(unported)
    return None
