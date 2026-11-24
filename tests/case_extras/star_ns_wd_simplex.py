# SPDX-License-Identifier: GPL-3.0-or-later
r"""Upstream neutron-star TOV gravity, white-dwarf cooling/nova-burst,
overshoot mesh-refinement, and solar simplex calibration `StarExtras` ports
(`#231`).

Covers:
- `star/test_suite/ns_c/src/run_star_extras.f90:37-151`
- `star/test_suite/ns_c/src/other_cgrav.inc:5-48`
- `star/test_suite/ns_h/src/run_star_extras.f90:37-151`
- `star/test_suite/ns_he/src/run_star_extras.f90:37-151`
- `star/test_suite/wd_cool_0.6M/src/run_star_extras.f90:39-172`
- `star/test_suite/wd_nova_burst/src/run_star_extras.f90:32-194`
- `star/test_suite/1M_pre_ms_to_wd/src/run_star_extras.f90:42-158`
- star/test_suite/1M_pre_ms_to_wd/src/xtra_coeff_os/xtra_coeff_os_controls.inc:4
- `star/test_suite/1M_pre_ms_to_wd/src/xtra_coeff_os/xtra_coeff_os.inc:2-256`
- `star/test_suite/simplex_solar_calibration/src/run_star_extras.f90:41-251`
- `star/test_suite/simplex_solar_calibration/src/simplex_search_run_support.f90`
  lines `600-633` and `1671-1800`
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Any

from flexmesa import constants
from flexmesa import math as fm_math
from tests.case_extras import _helpers, star_mixing_init

_XTRA_COEFF_DEFAULTS: dict[str, float] = {
    "xtra_coef_os_full_on": 1.0e-4,
    "xtra_coef_os_full_off": 0.1,
    "xtra_coef_os_above_nonburn": 1.0,
    "xtra_coef_os_below_nonburn": 1.0,
    "xtra_coef_os_above_burn_h": 1.0,
    "xtra_coef_os_below_burn_h": 1.0,
    "xtra_coef_os_above_burn_he": 1.0,
    "xtra_coef_os_below_burn_he": 1.0,
    "xtra_coef_os_above_burn_z": 1.0,
    "xtra_coef_os_below_burn_z": 1.0,
    "xtra_dist_os_above_nonburn": 0.2,
    "xtra_dist_os_below_nonburn": 0.2,
    "xtra_dist_os_above_burn_h": 0.2,
    "xtra_dist_os_below_burn_h": 0.2,
    "xtra_dist_os_above_burn_he": 0.2,
    "xtra_dist_os_below_burn_he": 0.2,
    "xtra_dist_os_above_burn_z": 0.2,
    "xtra_dist_os_below_burn_z": 0.2,
}


def _parse_fortran_float(raw: str) -> float:
    """Parse a Fortran real literal (`1d-4`, `0.25d0`) to `float`."""
    cleaned = raw.strip().replace("D", "e").replace("d", "e")
    return float(cleaned)


class NeutronStarEnvStarExtras:
    r"""Tolman-Oppenheimer-Volkoff `other_cgrav` hook shared by `ns_c`, `ns_h`,
    and `ns_he` (`star/test_suite/ns_c/src/other_cgrav.inc:5-48`).
    """

    def __init__(self) -> None:
        self._tzo_helper = star_mixing_init.TzoStarExtras()

    def other_cgrav(self, state: dict[str, Any]) -> list[float]:
        r"""Evaluate TOV `cgrav(1:nz)` in $\mathrm{cm^3\,g^{-1}\,s^{-2}}$
        (`:5-48`).
        """
        return self._tzo_helper.other_cgrav(state)


class WdCool06MStarExtras:
    """`zbar_div_abar` and `eosPC_fraction` profile columns from
    `star/test_suite/wd_cool_0.6M/src/run_star_extras.f90:39-172`.
    """

    def __init__(self) -> None:
        self.max_frac = 0.0
        self.kmax_1based = 0

    def extra_profile_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, list[float]]]:
        """Compute `zbar_div_abar` and `eosPC_fraction` `(nz,)` (`:118-160`)."""
        nz = int(state["nz"])
        zbar = state["zbar"]
        abar = state["abar"]
        eos_pc = state["eos_frac_pc"]
        z_div_a: list[float] = []
        pc_col: list[float] = []
        max_frac = 0.0
        kmax = 0
        for k in range(nz):
            z_div_a.append(_helpers.fdiv(float(zbar[k]), float(abar[k])))
            frac = float(eos_pc[k])
            pc_col.append(frac)
            if frac > max_frac:
                max_frac = frac
                kmax = k + 1
        self.max_frac = max_frac
        self.kmax_1based = kmax
        return [("zbar_div_abar", z_div_a), ("eosPC_fraction", pc_col)]


class WdNovaBurstStarExtras:
    r"""Classical-nova $L_{\mathrm{phot}}$ burst counter and termination hook
    from `star/test_suite/wd_nova_burst/src/run_star_extras.f90:32-194`.
    """

    def __init__(self) -> None:
        self.num_bursts = 0
        self.waiting_for_burst = True
        self.l_burst = 1.0e4
        self.l_between = 1.0e3

    def extras_startup(self, _state: dict[str, Any], restart: bool) -> None:
        """Reset burst state or require `other_photo_read` on restart
        (`:40-45, 76-89`).
        """
        if restart:
            raise NotImplementedError(
                "wd_nova_burst: extras_startup(restart=True) requires "
                "other_photo_read (:40-45, 62) in flexmesa-star (#150)"
            )
        self.num_bursts = 0
        self.waiting_for_burst = True

    def extras_check_model(
        self, state: dict[str, Any]
    ) -> str | tuple[str, str]:
        r"""Count $L_{\mathrm{phot}} > 10^4\,L_\odot$ bursts and stop below
        $10^3\,L_\odot$ (`:105-128`).
        """
        l_phot = float(state["l_phot"])
        if l_phot > self.l_burst:
            if self.waiting_for_burst:
                self.num_bursts += 1
                self.waiting_for_burst = False
        elif l_phot < self.l_between:
            done = self.num_bursts >= 1
            self.waiting_for_burst = True
            if done:
                return ("terminate", "have finished burst")
        return "keep_going"

    def extra_profile_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, list[float]]]:
        """Return `zbar_div_abar` `(nz,)` profile column (`:154-182`)."""
        return _helpers.zbar_div_abar_profile_column(state)


class PreMs1MToWdStarExtras:
    """Overshoot `other_mesh_delta_coeff_factor` hook and `&xtra_coeff_os`
    reader (`:42-158` of `1M_pre_ms_to_wd` and `:2-256` of `xtra_coeff_os`).
    """

    def __init__(self) -> None:
        self.params: dict[str, float] = dict(_XTRA_COEFF_DEFAULTS)
        self._loaded_xtra_coeff_os = False

    def read_inlist_xtra_coeff_os(self, case_dir: pathlib.Path | str) -> None:
        """Read `inlist_xtra_coeff_os` namelist (`:2-49` of `xtra_coeff_os`)."""
        self.params = dict(_XTRA_COEFF_DEFAULTS)
        self._loaded_xtra_coeff_os = False
        nml_path = pathlib.Path(case_dir) / "inlist_xtra_coeff_os"
        raw = nml_path.read_text(encoding="utf-8")
        in_nml = False
        found = False
        for raw_line in raw.splitlines():
            code = raw_line.split("!", 1)[0].strip()
            if not code:
                continue
            low = code.lower()
            if not in_nml:
                if low.startswith("&xtra_coeff_os"):
                    in_nml = True
                    found = True
                continue
            if low == "/" or low.startswith("/"):
                break
            if "=" not in code:
                raise ValueError(f"Malformed &xtra_coeff_os line: {code!r}")
            key, val_str = (part.strip() for part in code.split("=", 1))
            key_low = key.lower()
            if key_low not in self.params:
                raise ValueError(f"Unknown &xtra_coeff_os control: {key!r}")
            self.params[key_low] = _parse_fortran_float(val_str.rstrip(","))
        if not found:
            raise ValueError(f"Missing &xtra_coeff_os namelist in {nml_path}")
        self._loaded_xtra_coeff_os = True

    def extras_controls(self, state: dict[str, Any]) -> None:
        """Read `inlist_xtra_coeff_os` when
        `use_other_mesh_delta_coeff_factor` is true (`:42-59`).
        """
        if not bool(state["use_other_mesh_delta_coeff_factor"]):
            return
        self.read_inlist_xtra_coeff_os(str(state["case_dir"]))

    def _select_os_pair(
        self, direction: str, max_eps: float, eps_h: float, eps_he: float
    ) -> tuple[float, float]:
        """Select `(xtra_coef, xtra_dist)` for `'below'` or `'above'`."""
        if max_eps < 1.0:
            suffix = "nonburn"
        elif eps_h > 0.5 * max_eps:
            suffix = "burn_h"
        elif eps_he > 0.5 * max_eps:
            suffix = "burn_he"
        else:
            suffix = "burn_z"
        return (
            self.params[f"xtra_coef_os_{direction}_{suffix}"],
            self.params[f"xtra_dist_os_{direction}_{suffix}"],
        )

    def eval_other_mesh_delta_coeff_factor(
        self, state: dict[str, Any]
    ) -> list[float]:
        """Evaluate `s% mesh_delta_coeff_factor(1:nz)` from snapshot
        `mesh_delta_coeff_factor` (`:52-256` of `xtra_coeff_os.inc`).
        """
        return self.other_mesh_delta_coeff_factor(
            state, state["mesh_delta_coeff_factor"]
        )

    def other_mesh_delta_coeff_factor(
        self,
        state: dict[str, Any],
        current: Sequence[float] | None = None,
    ) -> list[float]:
        """Evaluate `s% mesh_delta_coeff_factor(1:nz)` (`:52-256` of
        `xtra_coeff_os.inc`).
        """
        if not self._loaded_xtra_coeff_os and state.get("case_dir"):
            nml_path = (
                pathlib.Path(str(state["case_dir"])) / "inlist_xtra_coeff_os"
            )
            if nml_path.is_file():
                self.read_inlist_xtra_coeff_os(str(state["case_dir"]))
        nz = int(state["nz"])
        base = state["mesh_delta_coeff_factor"] if current is None else current
        factors = [float(v) for v in base[:nz]]
        coef_keys = [
            k
            for k in self.params
            if k.startswith("xtra_coef_os_a") or k.startswith("xtra_coef_os_b")
        ]
        if all(self.params[k] == 1.0 for k in coef_keys):
            return factors

        he_cntr = float(state["xa_he4"][nz - 1])
        full_off = self.params["xtra_coef_os_full_off"]
        full_on = self.params["xtra_coef_os_full_on"]
        if he_cntr >= full_off:
            alfa_os = 0.0
        elif he_cntr <= full_on:
            alfa_os = 1.0
        else:
            alfa_os = (full_off - he_cntr) / (full_off - full_on)
        if alfa_os == 0.0:
            return factors

        conv_mix = int(constants.MixingType.CONVECTIVE_MIXING)
        ov_mix = int(constants.MixingType.OVERSHOOT_MIXING)
        mix = state["mixing_type"]
        eps_nuc = state["eps_nuc"]
        eps_pp = state["eps_nuc_ipp"]
        eps_cno = state["eps_nuc_icno"]
        eps_3alf = state["eps_nuc_i3alf"]
        peos = state["peos"]
        rho = state["rho"]
        grav = state["grav"]
        r_arr = state["r"]

        # Pass 1: surface to center (`:97-174`).
        in_conv = int(mix[0]) == conv_mix
        k = 1
        max_eps = -1.0e99
        eps_h = max_eps
        eps_he = max_eps
        while k < nz:
            eps = float(eps_nuc[k])
            if in_conv:
                if int(mix[k]) == conv_mix:
                    if eps > max_eps:
                        max_eps = eps
                        eps_h = float(eps_pp[k]) + float(eps_cno[k])
                        eps_he = float(eps_3alf[k])
                else:
                    in_conv = False
                    c_raw, xtra_dist = self._select_os_pair(
                        "below", max_eps, eps_h, eps_he
                    )
                    coef = c_raw * alfa_os + (1.0 - alfa_os)
                    if coef > 0.0 and coef != 1.0:
                        while True:
                            if int(mix[k]) != ov_mix:
                                break
                            if coef < factors[k]:
                                factors[k] = coef
                            if k == nz - 1:
                                break
                            k += 1
                        if xtra_dist > 0.0:
                            hp = float(peos[k]) / (
                                float(rho[k]) * float(grav[k])
                            )
                            r_extra = max(0.0, float(r_arr[k]) - xtra_dist * hp)
                            while True:
                                if float(r_arr[k]) < r_extra:
                                    break
                                if coef < factors[k]:
                                    factors[k] = coef
                                if k == nz - 1:
                                    break
                                k += 1
            elif int(mix[k]) == conv_mix:
                in_conv = True
                max_eps = eps
            k += 1

        # Pass 2: center to surface (`:177-255`).
        in_conv = int(mix[nz - 1]) == conv_mix
        k = nz - 2
        max_eps = -1.0e99
        eps_h = max_eps
        eps_he = max_eps
        while k >= 0:
            eps = float(eps_nuc[k])
            if in_conv:
                if int(mix[k]) == conv_mix:
                    if eps > max_eps:
                        max_eps = eps
                        eps_h = float(eps_pp[k]) + float(eps_cno[k])
                        eps_he = float(eps_3alf[k])
                else:
                    in_conv = False
                    c_raw, xtra_dist = self._select_os_pair(
                        "above", max_eps, eps_h, eps_he
                    )
                    coef = c_raw * alfa_os + (1.0 - alfa_os)
                    if coef > 0.0 and coef != 1.0:
                        while True:
                            if int(mix[k]) != ov_mix:
                                break
                            if coef < factors[k]:
                                factors[k] = coef
                            if k == 0:
                                break
                            k -= 1
                        if xtra_dist > 0.0:
                            hp = float(peos[k]) / (
                                float(rho[k]) * float(grav[k])
                            )
                            r_extra = min(
                                float(r_arr[0]),
                                float(r_arr[k]) + xtra_dist * hp,
                            )
                            while True:
                                if float(r_arr[k]) > r_extra:
                                    break
                                if coef < factors[k]:
                                    factors[k] = coef
                                if k == 0:
                                    break
                                k -= 1
            elif int(mix[k]) == conv_mix:
                in_conv = True
                max_eps = eps
            k -= 1

        return factors


def set_simplex_my_vars(
    state: dict[str, Any], include_my_var1_in_chi2: bool
) -> float | None:
    """Evaluate `my_var1 = s% Teff` when `include_my_var1_in_chi2` is true
    (`star/test_suite/simplex_solar_calibration/src/run_star_extras.f90:66-81`).
    """
    if include_my_var1_in_chi2:
        return float(state["teff"])
    return None


def will_set_simplex_my_param(
    i_param: int, new_value: float, vary_my_param1: bool
) -> float | None:
    """Return updated `mixing_length_alpha` when `i == 1` and `vary_my_param1`
    (`star/test_suite/simplex_solar_calibration/src/run_star_extras.f90:84-101`).
    """
    if i_param == 1 and vary_my_param1:
        return float(new_value)
    return None


def eval_gradr_grada_mesh_fcn(
    state: dict[str, Any],
) -> tuple[str, bool, list[float]]:
    """Evaluate `gradr_grada_mesh_fcn_data` `(nz,)`
    (`star/test_suite/simplex_solar_calibration/src/run_star_extras.f90:211-251`).
    """
    x_ctrl = state["x_ctrl"]
    weight = float(x_ctrl[0])
    width = float(x_ctrl[1])
    center = float(x_ctrl[2])
    nz = int(state["nz"])
    gradr = state["gradr"]
    grada = state["grada"]
    maxval = 700.0
    vals: list[float] = []
    for k in range(nz):
        arg = min(
            maxval,
            _helpers.fdiv(float(gradr[k]) - float(grada[k]) - center, width),
        )
        vals.append(weight * float(fm_math.tanh(arg)) * width)
    return ("gradr_grada_function", False, vals)


def simplex_param(
    x: float,
    first: float,
    p_min: float,
    p_max: float,
    *,
    scale_simplex_params: bool = True,
) -> float:
    """Map normalized simplex coordinate `x` to parameter space
    (`star/test_suite/simplex_solar_calibration/src/simplex_search_run_support.f90:600-611`).
    """
    if not scale_simplex_params:
        return x
    if x > 0.0:
        return first + x * (p_max - first)
    return first + x * (first - p_min)


def simplex_inverse(
    param: float,
    first: float,
    p_min: float,
    p_max: float,
    *,
    scale_simplex_params: bool = True,
) -> float:
    """Map physical parameter `param` to normalized simplex coordinate
    (`star/test_suite/simplex_solar_calibration/src/simplex_search_run_support.f90:614-633`).
    """
    if not scale_simplex_params:
        return param
    if param > first:
        if p_max == first:
            return 1.0
        return (param - first) / (p_max - first)
    if first == p_min:
        return -1.0
    return (param - first) / (first - p_min)


def eval_simplex_chi2(
    terms: list[tuple[bool, float, float, float]],
) -> tuple[float, int]:
    """Evaluate evenly weighted Brandao et al. (2011) `get_chi2`
    (`star/test_suite/simplex_solar_calibration/src/simplex_search_run_support.f90:1671-1800`).

    Args:
        terms: Sequence of `(include_flag, value, target, sigma)` tuples.

    Returns:
        Tuple `(chi2, num_chi2_terms)`.
    """
    chi2_sum = 0.0
    chi2_n = 0
    for include_flag, value, target, sigma in terms:
        if sigma > 0.0 and include_flag:
            diff = (value - target) / sigma
            chi2_sum += diff * diff
            chi2_n += 1
    return (chi2_sum / float(max(1, chi2_n)), chi2_n)


class SimplexSolarCalibrationStarExtras:
    """Simplex solar-calibration callbacks and `gradr_grada_function` mesh
    weight from
    `star/test_suite/simplex_solar_calibration/src/run_star_extras.f90:41-251`.
    """

    def extras_controls(self, _state: dict[str, Any]) -> None:
        """Fail loudly because `star_simplex_procs` (`#162`) and
        `other_mesh_fcn_data` (`#153`) are not exposed (`:41-63`).
        """
        raise NotImplementedError(
            "simplex_solar_calibration: extras_controls registers "
            "star_simplex_procs (:55-59, #162) and "
            "how_many_other_mesh_fcns/other_mesh_fcn_data (:61-62, #153), "
            "which flexmesa-star StarExtras does not expose"
        )
