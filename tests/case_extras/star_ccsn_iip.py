# SPDX-License-Identifier: GPL-3.0-or-later
r"""Upstream Type II-P core-collapse supernova (`ccsn_IIp`) low-density wind,
$^{56}\mathrm{Ni}$/$^{56}\mathrm{Co}$ rescaling, shock `stop_m` locator, RTI
post-check, and STELLA handoff `StarExtras` port (`#231`).

Covers:
- `star/test_suite/ccsn_IIp/src/run_star_extras.f90:31-1065`
- `star/test_suite/ccsn_IIp/src/stella/stella.inc:2-646`
- `star/test_suite/ccsn_IIp/src/stella/stella_controls.inc:1-3`
"""

from __future__ import annotations

import pathlib
from typing import Any

from flexmesa import constants
from flexmesa.chem import catalog as chem_catalog
from tests.case_extras import _helpers, star_core_collapse

_H1_LIMIT: float = 0.1


def _parse_nml_scalar(raw: str) -> Any:
    """Parse a Fortran namelist RHS (`bool`, `int`, `float`, or `str`)."""
    text = raw.strip().rstrip(",").strip()
    low = text.lower()
    if low in (".true.", "t"):
        return True
    if low in (".false.", "f"):
        return False
    if (text.startswith("'") and text.endswith("'")) or (
        text.startswith('"') and text.endswith('"')
    ):
        return text[1:-1]
    if any(ch in low for ch in (".", "d", "e")):
        return float(text.replace("D", "e").replace("d", "e"))
    return int(text)


def set_nico_mass_arrays(
    xa: list[list[float]],
    dm: list[float],
    m_arr: list[float],
    *,
    i_o16_1: int,
    i_ni56_1: int,
    i_co56_1: int,
    new_ni: float,
    m_center: float,
    he_core_mass: float,
    x_ni_mass_start: float,
    x_ni_mass_end: float,
    u_flag: bool = False,
    u_arr: list[float] | None = None,
    min_cntr_u: float = 0.0,
    final_call: bool = False,
    stella_skip_inner_dm: float = 0.0,
    stella_skip_inner_v_limit: float = 5.0e7,
) -> float:
    r"""Port of `set_nico_mass` (`:193-381`) mutating `xa` `(species, nz)` in
    place and returning revised $^{56}\mathrm{Ni}$ mass in $M_\odot$.
    """
    if i_o16_1 <= 0:
        raise RuntimeError("need to have o16 in net for set_nico_mass")
    species = len(xa)
    nz = len(dm)
    o16 = i_o16_1 - 1
    ni56 = i_ni56_1 - 1
    co56 = i_co56_1 - 1
    msun = constants.MSUN

    for k in range(nz):
        for j in range(species):
            xa[j][k] = max(0.0, min(1.0, float(xa[j][k])))
        xsum = sum(xa[j][k] for j in range(species))
        if abs(xsum - 1.0) > 1.0e-12:
            for j in range(species):
                xa[j][k] = xa[j][k] / xsum

    min_m = x_ni_mass_start * msun + m_center
    max_m = x_ni_mass_end * msun + he_core_mass * msun
    if u_flag:
        if u_arr is None:
            raise KeyError("u")
        for k in range(nz - 1, -1, -1):
            if float(u_arr[k]) > min_cntr_u and float(m_arr[k]) > min_m:
                min_m = float(m_arr[k])
                break

    if max_m > 0.0:
        for k in range(nz):
            xa[o16][k] = xa[o16][k] + xa[ni56][k] + xa[co56][k]
            xa[co56][k] = 0.0
            xa[ni56][k] = 0.0
        max_m = min(max_m, float(m_arr[0]))
        if min_m < 0.0:
            min_m = max_m - new_ni * msun
            mp1 = float(m_arr[0])
            for k in range(nz):
                m00 = mp1
                mp1 = m00 - float(dm[k])
                if mp1 >= max_m:
                    continue
                if m00 <= max_m and mp1 >= min_m:
                    for j in range(species):
                        xa[j][k] = 0.0
                    xa[ni56][k] = 1.0
                else:
                    dm_ov = min(m00, max_m) - max(mp1, min_m)
                    f_rem = max(0.0, min(1.0, 1.0 - dm_ov / float(dm[k])))
                    for j in range(species):
                        xa[j][k] = f_rem * xa[j][k]
                    xa[ni56][k] = 0.0
                    xa[ni56][k] = 1.0 - sum(xa[j][k] for j in range(species))
        else:
            min_m = max(min_m, m_center)
            if min_m >= max_m:
                raise RuntimeError("set_nico_mass: min_m >= max_m")
            new_ni_frac = new_ni * msun / (max_m - min_m)
            mp1 = float(m_arr[0])
            for k in range(nz):
                m00 = mp1
                mp1 = m00 - float(dm[k])
                if m00 <= min_m:
                    break
                if mp1 >= max_m:
                    continue
                dm_ov = (
                    float(dm[k])
                    if (m00 <= max_m and mp1 >= min_m)
                    else min(m00, max_m) - max(mp1, min_m)
                )
                xa[ni56][k] = 0.0
                jmax = max(range(species), key=lambda idx: xa[idx][k])
                if xa[jmax][k] < new_ni_frac:
                    raise RuntimeError("set_nico_mass: too much ni for cell")
                xa[ni56][k] = new_ni_frac * dm_ov / float(dm[k])
                xa[jmax][k] -= xa[ni56][k]
        mass_ni56 = sum(float(dm[k]) * xa[ni56][k] for k in range(nz)) / msun
        if abs(mass_ni56 - new_ni) > 1.0e-6 * new_ni:
            raise RuntimeError("set_nico_mass: bad new ni")
        return mass_ni56

    kcut = nz
    if final_call and stella_skip_inner_dm > 0.0:
        if u_arr is None and stella_skip_inner_v_limit >= 0.0:
            raise KeyError("u")
        mcut = m_center + stella_skip_inner_dm * msun
        for k in range(nz - 1, -1, -1):
            v_ok = stella_skip_inner_v_limit < 0.0 or (
                u_arr is not None
                and float(u_arr[k]) >= stella_skip_inner_v_limit
            )
            if float(m_arr[k]) >= mcut and v_ok:
                kcut = k + 1
                break

    for k in range(nz):
        xa[ni56][k] += xa[co56][k]
        xa[co56][k] = 0.0
    mass_ni56 = sum(float(dm[k]) * xa[ni56][k] for k in range(kcut)) / msun
    mass_co56 = sum(float(dm[k]) * xa[co56][k] for k in range(kcut)) / msun
    old_nico = mass_ni56 + mass_co56
    if new_ni < 0.0:
        raise RuntimeError("bad mass set_nico_mass")
    alfa_nico = _helpers.fdiv(new_ni, old_nico)
    for k in range(kcut):
        xo16 = xa[o16][k]
        old_xnico = xa[ni56][k] + xa[co56][k]
        new_xnico = alfa_nico * old_xnico
        dxnico = new_xnico - old_xnico
        xa[co56][k] = 0.0
        xa[ni56][k] = new_xnico
        if dxnico <= xo16:
            xa[o16][k] = xo16 - dxnico
        else:
            jmax = max(range(species), key=lambda idx: xa[idx][k])
            xa[jmax][k] -= dxnico
            if xa[jmax][k] <= 0.0:
                raise RuntimeError("failed in set_nico_mass")
    check_ni56 = sum(float(dm[k]) * xa[ni56][k] for k in range(kcut)) / msun
    if abs(check_ni56 - new_ni) > 1.0e-10 * max(1.0e-10, new_ni):
        raise RuntimeError("bad check mass in set_nico_mass")
    return new_ni


def enable_magnetar(state: dict[str, Any]) -> None:
    """Guard magnetar injection (`:1038-1065`; deferred upstream bug `#245`).

    Upstream declares an uninitialized local `real(dp) :: age_days` at `:1040`
    and branches on it at `:1050-1062` whenever `x_ctrl(55) > 0`.
    """
    l_center = float(state["x_ctrl"][54])
    if l_center <= 0.0:
        return
    raise RuntimeError(
        "ccsn_IIp: enable_magnetar (:1040, 1050-1062) branches on an "
        "uninitialized local age_days when x_ctrl(55) > 0 (#245, #169)"
    )


def compute_adjusted_v_center(state: dict[str, Any]) -> float:
    r"""Evaluate `adjust_v_center` in $\mathrm{cm\,s^{-1}}$ (`:928-938`)."""
    nz = int(state["nz"])
    v_c = float(state["v_center"])
    r_c = float(state["r_center"])
    dt = float(state["dt"])
    v_c = v_c - float(state["cgrav"][nz - 1]) * float(state["m_center"]) / (
        r_c * r_c
    )
    v_c = max(v_c, -float(state["x_ctrl"][0]) * constants.CLIGHT)
    if r_c + v_c * dt < float(state["center_r_lower_limit"]):
        v_c = (float(state["center_r_lower_limit"]) - r_c) / dt
    return v_c


class CcsnIIpStarExtras:
    r"""Type II-P CCSN shock progression, low-density wind, $^{56}\mathrm{Ni}$
    mass rescaling, RTI mixing check, and STELLA export hooks from
    `star/test_suite/ccsn_IIp/src/run_star_extras.f90:31-1065` and
    `star/test_suite/ccsn_IIp/src/stella/stella.inc:2-646`.
    """

    def __init__(self) -> None:
        self.stella_params: dict[str, Any] = {
            "save_stella_data_for_model_number": 0,
            "save_stella_data_when_terminate": True,
            "save_stella_data_filename": "mesa",
            "stella_num_points": 400,
            "stella_nz_extra": 0,
            "stella_min_surf_logrho": -100.0,
            "stella_min_velocity": 0.0,
            "stella_skip_inner_dm": 0.0,
            "stella_skip_inner_v_limit": 5.0e7,
            "stella_mdot_years_for_wind": 1.0,
            "stella_mdot_for_wind": 0.25,
            "stella_v_wind": 10.0,
            "stella_show_headers": False,
        }
        self.initial_nico = 0.0
        self.initial_m_center = 0.0
        self.initial_he_core_mass = 0.0
        self.initial_mass = 0.0
        self.initial_time = 0.0
        self.start_m = 0.0
        self.stop_m = 0.0
        self.max_mass_h = -1.0

    def read_inlist_stella(self, case_dir: pathlib.Path | str) -> None:
        """Read `inlist_stella` (`:2-41` of `stella.inc`)."""
        nml_path = pathlib.Path(case_dir) / "inlist_stella"
        raw = nml_path.read_text(encoding="utf-8")
        in_nml = False
        found = False
        for raw_line in raw.splitlines():
            code = raw_line.split("!", 1)[0].strip()
            if not code:
                continue
            low = code.lower()
            if not in_nml:
                if low.startswith("&stella"):
                    in_nml = True
                    found = True
                continue
            if low == "/" or low.startswith("/"):
                break
            if "=" not in code:
                raise ValueError(f"Malformed &stella line: {code!r}")
            key, val_str = (part.strip() for part in code.split("=", 1))
            key_low = key.lower()
            if key_low not in self.stella_params:
                raise ValueError(f"Unknown &stella control: {key!r}")
            self.stella_params[key_low] = _parse_nml_scalar(val_str)
        if not found:
            raise ValueError(f"Missing &stella namelist in {nml_path}")

    def extras_controls(self, state: dict[str, Any]) -> None:
        """Read `inlist_stella` (`:105-132`)."""
        self.read_inlist_stella(str(state["case_dir"]))

    def other_alpha_mlt(self, state: dict[str, Any]) -> float | None:
        """Guard per-cell `alpha_mlt_routine` (`:138-161`)."""
        return star_core_collapse.require_per_cell_alpha_mlt(
            state, "ccsn_IIp", ":138-161"
        )

    def other_wind(
        self,
        state: dict[str, Any],
        _l_surf: float,
        _m_surf: float,
        _r_surf: float,
        _t_surf: float,
    ) -> float:
        r"""Evaluate `low_density_wind_routine` in $M_\odot\,\mathrm{yr^{-1}}$
        (`:164-190`).
        """
        i_lnd = int(state["i_lnd"])
        if i_lnd <= 0:
            return 0.0
        nz = int(state["nz"])
        lnd = state["lnd"]
        dm = state["dm"]
        lnd_limit = constants.LN10 * (-14.5)
        msum = 0.0
        for k in range(nz):
            if float(lnd[k]) >= lnd_limit:
                break
            msum += float(dm[k])
        if msum == 0.0:
            return 0.0
        return (msum / constants.MSUN) / (float(state["dt"]) / constants.SECYER)

    def find_stop_m(self, state: dict[str, Any]) -> tuple[float, float]:
        r"""Compute `(start_m, stop_m)` in $M_\odot$ for shock parts `1..5`
        (`:484-614`).
        """
        nz = int(state["nz"])
        msun = constants.MSUN
        m_center = float(state["m_center"])
        star_mass = float(state["star_mass"])
        m_arr = state["m"]
        x_ctrl = state["x_ctrl"]
        part = int(state["x_integer_ctrl"][0])

        start_m = float(state["shock_mass"])
        if start_m == 0.0:
            u_arr = state["u"]
            k_max = max(range(nz), key=lambda idx: float(u_arr[idx]))
            start_m = float(m_arr[k_max]) / msun
            if start_m >= 2.0 * m_center / msun:
                start_m = m_center / msun

        force_stop = float(x_ctrl[97])
        default_stop = float(x_ctrl[98])
        below_surf = float(x_ctrl[15])
        stop_m = 0.0
        if force_stop > 0.0:
            stop_m = force_stop
        elif part == 1:
            xa_o16 = state["xa_o16"]
            xa_he4 = state["xa_he4"]
            for k in range(nz):
                if float(xa_o16[k]) > float(xa_he4[k]):
                    m_k = float(m_arr[k])
                    stop_m = (0.15 * m_center + 0.85 * m_k) / msun
                    if stop_m <= start_m:
                        stop_m = (0.05 * m_center + 0.95 * m_k) / msun
                        if stop_m <= start_m:
                            stop_m = (0.01 * m_center + 0.99 * m_k) / msun
                            if stop_m <= start_m:
                                stop_m = 1.001 * start_m
                    break
            if stop_m == 0.0:
                if default_stop <= 0.0:
                    raise RuntimeError("failed to find stop_m")
                stop_m = default_stop
        elif part == 2:
            xa_h1 = state["xa_h1"]
            xa_he4 = state["xa_he4"]
            xa_o16 = state["xa_o16"]
            k1 = -1
            for k in range(nz):
                if float(xa_he4[k]) > float(xa_h1[k]):
                    k1 = k
                    break
            if k1 >= 0:
                for k in range(k1 + 1, nz):
                    if float(xa_o16[k]) > float(xa_he4[k]):
                        f_he = float(x_ctrl[6])
                        stop_m = (
                            (1.0 - f_he) * float(m_arr[k])
                            + f_he * float(m_arr[k1])
                        ) / msun
                        break
            if stop_m == 0.0:
                if default_stop <= 0.0:
                    raise RuntimeError("failed to find stop_m")
                stop_m = default_stop
        elif part == 3:
            xa_h1 = state["xa_h1"]
            xa_he4 = state["xa_he4"]
            for k in range(nz):
                if float(xa_he4[k]) > float(xa_h1[k]):
                    stop_m = 0.95 * float(m_arr[k]) / msun + 0.05 * (
                        star_mass - 0.1
                    )
                    break
            if stop_m == 0.0:
                raise RuntimeError("failed to find stop_m")
        elif part == 4:
            stop_m = star_mass - below_surf
        elif part == 5 and below_surf > 0.0:
            stop_m = star_mass - below_surf

        if below_surf > 0.0:
            stop_m = min(stop_m, star_mass - below_surf)
        if start_m > stop_m and stop_m > 0.0:
            raise RuntimeError("extras_startup: start_m > stop_m")
        return start_m, stop_m

    def check_rti(self, state: dict[str, Any]) -> float:
        """Evaluate `check_rti` (`:764-789`), reproducing double `net_iso` at
        `:775` (`#246`, `#169`).
        """
        nz = int(state["nz"])
        species = int(state["species"])
        net_iso = state["net_iso"]
        m_arr = state["m"]
        xa_matrix = state["xa"]

        ih1 = chem_catalog.get_nuclide_index("h1")
        first_slot = int(net_iso[ih1 - 1])
        if first_slot <= 0 or first_slot > len(net_iso):
            raise IndexError(
                f"ccsn_IIp: check_rti (:775) net_iso(ih1)={first_slot} out of "
                "bounds for double net_iso lookup (#246, #169)"
            )
        # Reproduce `s% net_iso(s% net_iso(ih1))` verbatim (#246, #169).
        second_slot = int(net_iso[first_slot - 1])
        if second_slot <= 0 or second_slot > species:
            raise IndexError(
                "ccsn_IIp: check_rti (:775) double lookup "
                f"net_iso(net_iso(ih1))={second_slot} out of 1..{species} "
                "(#246, #169)"
            )

        h1_mass = -1.0
        for k in range(nz - 1):
            if (
                float(m_arr[k]) >= self.max_mass_h
                and float(m_arr[k + 1]) <= self.max_mass_h
            ):
                h1_mass = float(xa_matrix[second_slot - 1][k])
                break
        if h1_mass < 0.0:
            raise RuntimeError("Error: Could not find h1 limit")
        if h1_mass < _H1_LIMIT * 2.0:
            raise RuntimeError("Error: RTI mxiing did not seem to work")
        return h1_mass

    def extras_startup(
        self, state: dict[str, Any], restart: bool
    ) -> dict[str, Any] | None:
        """Initialize stage state or require `#150`/`#154` (`:383-735`)."""
        if restart:
            raise NotImplementedError(
                "ccsn_IIp: extras_startup(restart=True) requires "
                "extras_photo_read (:1002-1021, #150)"
            )
        if (
            int(state["net_iso_o16"]) <= 0
            or int(state["net_iso_he4"]) <= 0
            or int(state["net_iso_h1"]) <= 0
        ):
            raise RuntimeError("missing o16, he4, or h1")
        if float(state["logrho_min_opal_scvh_limit"]) > -12.0:
            raise RuntimeError(
                "extras_startup: logRho_min_OPAL_SCVH_limit > -12"
            )

        self.initial_nico = 0.0
        self.stop_m = 0.0
        self.initial_m_center = float(state["m_center"])
        self.initial_mass = float(state["star_mass"])
        self.initial_time = float(state["time"])
        self.initial_he_core_mass = float(state["he_core_mass"])
        self.max_mass_h = -1.0

        part = int(state["x_integer_ctrl"][0])
        x_ctrl = state["x_ctrl"]
        if part == -3:
            if float(
                state["total_mass_for_inject_extra_ergs_sec"]
            ) > 0.0 and bool(state["v_flag"]):
                raise NotImplementedError(
                    "ccsn_IIp: extras_startup part -3 zeros s% v(1:nz) "
                    "(:430-435), which StarExtras updates do not expose (#152)"
                )
            return None

        if part == 4:
            nz = int(state["nz"])
            xa_h1 = state["xa_h1"]
            m_arr = state["m"]
            for k in range(nz):
                if float(xa_h1[k]) < _H1_LIMIT:
                    self.max_mass_h = float(m_arr[k])
                    break
            if self.max_mass_h < 0.0:
                raise RuntimeError("Error: Could not find h1 limit")

        if part == 5 and float(x_ctrl[18]) > 0.0:
            raise NotImplementedError(
                "ccsn_IIp: add_csm (:616-733) requires in-place hydro/EOS "
                "cell reconstruction (#154)"
            )
        if (
            float(x_ctrl[46]) > 0.0 and int(state["x_integer_ctrl"][2]) > 0
        ) or (float(x_ctrl[49]) > 0.0 and int(state["x_integer_ctrl"][3]) > 0):
            raise NotImplementedError(
                "ccsn_IIp: smooth_xa_by_boxcar_mass (:459, 476) is not "
                "exposed in flexmesa-star StarExtras (#158)"
            )
        if (
            int(state["net_iso_ni56"]) > 0
            and int(state["net_iso_co56"]) > 0
            and float(x_ctrl[11]) > 0.0
        ):
            raise NotImplementedError(
                "ccsn_IIp: set_nico_mass (:466) mutates s% xa in place, which "
                "flexmesa-star StarExtras updates do not expose (#158)"
            )
        if part == 1:
            raise NotImplementedError(
                "ccsn_IIp: extras_startup part 1 resets "
                "s% cumulative_energy_error = 0 (:481-482), which StarExtras "
                "updates do not expose (#152)"
            )

        self.start_m, self.stop_m = self.find_stop_m(state)
        if self.stop_m > 0.0:
            new_x_ctrl = [float(v) for v in x_ctrl]
            new_x_ctrl[1] = self.stop_m
            return {"x_ctrl": new_x_ctrl}
        return None

    def extras_check_model(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:796-806`)."""
        return "keep_going"

    def extra_profile_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, list[float]]]:
        """Return `du` zero-column in shock part 3 (`:834-873`)."""
        if int(state["x_integer_ctrl"][0]) == 3:
            return [("du", [0.0] * int(state["nz"]))]
        return []

    def extras_start_step(self, state: dict[str, Any]) -> str:
        """Run magnetar/RTI/center-velocity step-start checks (`:877-948`)."""
        part = int(state["x_integer_ctrl"][0])
        if part == 1 and int(state["model_number"]) >= 1000:
            raise NotImplementedError(
                "ccsn_IIp: extras_start_step sets s% max_timestep = 0 "
                "(:891-892), which StarExtras updates do not expose (#152)"
            )
        x_ctrl = state["x_ctrl"]
        age_days = float(state["star_age"]) * 365.25
        if (
            float(x_ctrl[13]) > 0.0
            and age_days >= float(x_ctrl[13])
            and float(state["rti_c"]) > 0.0
        ):
            raise NotImplementedError(
                "ccsn_IIp: turn_off_rti (:896-899, 914-925) mutates RTI "
                "controls, which StarExtras updates do not expose (#152)"
            )
        enable_magnetar(state)
        if age_days >= float(x_ctrl[31]):
            raise NotImplementedError(
                "ccsn_IIp: adjust_delta_lgL (:903-905, 941-946) mutates "
                "s% delta_lgL_limit, which StarExtras updates do not expose "
                "(#152)"
            )
        if bool(state["x_logical_ctrl"][0]) and float(state["dt"]) > 0.0:
            _ = compute_adjusted_v_center(state)
            raise NotImplementedError(
                "ccsn_IIp: adjust_v_center (:907-909, 928-938) mutates "
                "s% v_center, which StarExtras updates do not expose (#152)"
            )
        return "keep_going"

    def extras_finish_step(
        self, state: dict[str, Any]
    ) -> str | tuple[str, str]:
        """Stop when `shock_mass >= x_ctrl(2)` and guard `restore_nico_mass` /
        STELLA output (`:952-1000`).
        """
        x_ctrl = state["x_ctrl"]
        stop_m_target = float(x_ctrl[1])
        if stop_m_target <= 0.0:
            return "keep_going"
        result: str | tuple[str, str] = "keep_going"
        if float(state["shock_mass"]) >= stop_m_target:
            if (
                int(state["net_iso_ni56"]) > 0
                and int(state["net_iso_co56"]) > 0
                and float(x_ctrl[11]) > 0.0
            ):
                raise NotImplementedError(
                    "ccsn_IIp: restore_nico_mass (:972, 986-997) mutates "
                    "s% xa in place (#158)"
                )
            result = ("terminate", "shock has reached target location")
        if int(state["x_integer_ctrl"][0]) == 5 and int(
            state["model_number"]
        ) == int(self.stella_params["save_stella_data_for_model_number"]):
            raise NotImplementedError(
                "ccsn_IIp: write_stella_data (:978-982; "
                "star/test_suite/ccsn_IIp/src/stella/stella.inc:44-646) "
                "is deferred to STELLA (#172)"
            )
        return result

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """Guard part-5 STELLA export (`#172`) and run part-4 `check_rti`
        (`:738-792`).
        """
        part = int(state["x_integer_ctrl"][0])
        if part == 5 and bool(
            self.stella_params["save_stella_data_when_terminate"]
        ):
            raise NotImplementedError(
                "ccsn_IIp: write_stella_data (:748-752; "
                "star/test_suite/ccsn_IIp/src/stella/stella.inc:44-646) "
                "is deferred to STELLA (#172)"
            )
        if part == 4 and bool(state["rti_flag"]):
            self.check_rti(state)
