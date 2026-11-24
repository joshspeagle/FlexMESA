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

import os
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
    """`set_param` and `set_constraint_value` of `astero_adipls`,
    `astero_gyre` and `surface_effects`.

    Upstream `set_param` only sets `ierr = 0`
    (`astero/test_suite/astero_adipls/src/run_star_extras.f90:86-96`,
    `astero_gyre` `:91-100`, `surface_effects` `:92-100`), and
    `set_constraint_value` has only the `case default` branch that reads the
    history column of that name (`astero_adipls` `:58-83`, `astero_gyre`
    `:63-88`, `surface_effects` `:64-89`). These cases set
    `astero_just_call_my_extras_check_model`, so the search calls neither
    routine; they are ported so that the case's own versions, not another
    case's, would run if it did.

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
        routines.get(name) == d for name, d in _ASTERO_GYRE_DIGESTS.items()
    ) and all(
        name in _ASTERO_GYRE_DIGESTS
        or d in _STOCK_ROUTINE_DIGESTS.get(name, frozenset())
        for name, d in routines.items()
    ):
        return AsteroGyreExtras()
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
        return SearchAsteroExtras(unported)
    if set_param == _FAST_FROM_FILE_SET_PARAM_DIGEST:
        return FastFromFileAsteroExtras(unported)
    return None
