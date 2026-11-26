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

import math
import pathlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from flexmesa import _rust, constants
from flexmesa import math as fm_math
from flexmesa.data import fetch as fm_fetch
from flexmesa.interp_1d import interp_pm, interp_value
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


_SIMPLEX_SEARCH_DEFAULTS: dict[str, bool | int | float | str] = {
    "just_do_first_values": False,
    "vary_FeH": False,
    "vary_Y": False,
    "vary_mass": False,
    "vary_alpha": False,
    "vary_f_ov": False,
    "first_FeH": 0.0,
    "first_Y": 0.0,
    "first_mass": 0.0,
    "first_alpha": 0.0,
    "first_f_ov": 0.0,
    "min_FeH": 0.0,
    "min_Y": 0.0,
    "min_mass": 0.0,
    "min_alpha": 0.0,
    "min_f_ov": 0.0,
    "max_FeH": 0.0,
    "max_Y": 0.0,
    "max_mass": 0.0,
    "max_alpha": 0.0,
    "max_f_ov": 0.0,
    "vary_my_param1": False,
    "vary_my_param2": False,
    "vary_my_param3": False,
    "first_my_param1": 0.0,
    "first_my_param2": 0.0,
    "first_my_param3": 0.0,
    "min_my_param1": 0.0,
    "min_my_param2": 0.0,
    "min_my_param3": 0.0,
    "max_my_param1": 0.0,
    "max_my_param2": 0.0,
    "max_my_param3": 0.0,
    "my_param1_name": "my_param1",
    "my_param2_name": "my_param2",
    "my_param3_name": "my_param3",
    "Y_depends_on_Z": False,
    "Y0": 0.248,
    "dYdZ": 1.4,
    "f0_ov_div_f_ov": -1.0,
    "include_Teff_in_chi2": False,
    "Teff_target": -1.0,
    "Teff_sigma": -1.0,
    "include_logL_in_chi2": False,
    "logL_target": -1.0,
    "logL_sigma": -1.0,
    "include_logg_in_chi2": False,
    "logg_target": -1.0,
    "logg_sigma": -1.0,
    "include_FeH_in_chi2": True,
    "FeH_target": -1.0,
    "FeH_sigma": -1.0,
    "Z_div_X_solar": 0.02293,
    "include_logR_in_chi2": False,
    "logR_target": -1.0,
    "logR_sigma": -1.0,
    "include_surface_Z_div_X_in_chi2": False,
    "surface_Z_div_X_target": -1.0,
    "surface_Z_div_X_sigma": -1.0,
    "include_surface_He_in_chi2": False,
    "surface_He_target": -1.0,
    "surface_He_sigma": -1.0,
    "include_Rcz_in_chi2": False,
    "Rcz_target": -1.0,
    "Rcz_sigma": -1.0,
    "include_solar_cs_rms_in_chi2": False,
    "solar_cs_rms_target": -1.0,
    "solar_cs_rms_sigma": -1.0,
    "report_solar_cs_rms": False,
    "include_age_in_chi2": False,
    "age_target": -1.0,
    "age_sigma": -1.0,
    "num_smaller_steps_before_age_target": 50,
    "dt_for_smaller_steps_before_age_target": 0.0065e8,
    "include_my_var1_in_chi2": False,
    "my_var1_target": 0.0,
    "my_var1_sigma": 0.0,
    "my_var1_name": "my_var1",
    "include_my_var2_in_chi2": False,
    "my_var2_target": 0.0,
    "my_var2_sigma": 0.0,
    "my_var2_name": "my_var2",
    "include_my_var3_in_chi2": False,
    "my_var3_target": 0.0,
    "my_var3_sigma": 0.0,
    "my_var3_name": "my_var3",
    "trace_chi2_info": False,
    "min_age_limit": 1.0e6,
    "Lnuc_div_L_limit": 0.95,
    "chi2_limit": 1000.0,
    "eval_chi2_at_target_age_only": False,
    "min_age_for_chi2": -1.0,
    "max_age_for_chi2": -1.0,
    "simplex_itermax": 1000,
    "simplex_fcn_calls_max": 10000,
    "simplex_x_atol": 1.0e-10,
    "simplex_x_rtol": 1.0e-10,
    "simplex_chi2_tol": 1.0e-10,
    "chi2_limit_for_small_timesteps": 50.0,
    "max_yrs_dt_chi2_small_limit": 3.0e5,
    "chi2_limit_for_smaller_timesteps": 20.0,
    "max_yrs_dt_chi2_smaller_limit": 1.0e5,
    "chi2_limit_for_smallest_timesteps": 10.0,
    "max_yrs_dt_chi2_smallest_limit": 5.0e4,
    "sigmas_coeff_for_logg_limit": -5.0,
    "sigmas_coeff_for_logL_limit": 5.0,
    "sigmas_coeff_for_Teff_limit": -5.0,
    "sigmas_coeff_for_logR_limit": 0.0,
    "sigmas_coeff_for_surface_Z_div_X_limit": 0.0,
    "sigmas_coeff_for_surface_He_limit": 0.0,
    "sigmas_coeff_for_solar_Rcz_limit": 0.0,
    "sigmas_coeff_for_solar_cs_rms_limit": 0.0,
    "sigmas_coeff_for_my_var1_limit": 0.0,
    "sigmas_coeff_for_my_var2_limit": 0.0,
    "sigmas_coeff_for_my_var3_limit": 0.0,
    "chi2_relative_increase_limit": 2.0,
    "limit_num_chi2_too_big": 20,
    "chi2_search_limit1": 3.0,
    "chi2_search_limit2": 4.0,
    "min_num_samples_for_avg": 2,
    "max_num_samples_for_avg": 10,
    "avg_age_sigma_limit": 10.0,
    "avg_model_number_sigma_limit": 10.0,
    "simplex_centroid_weight_power": 0.0,
    "simplex_enforce_bounds": False,
    "simplex_adaptive_random_search": False,
    "simplex_output_filename": "simplex_results.data",
    "restart_simplex_from_file": False,
    "simplex_seed": 1074698122,
    "simplex_alpha": 1.0,
    "simplex_beta": 2.0,
    "simplex_gamma": 0.5,
    "simplex_delta": 0.5,
    "write_best_model_data_for_each_sample": True,
    "num_digits": 4,
    "sample_results_dir": "outputs",
    "sample_results_prefix": "sample_",
    "sample_results_postfix": ".data",
    "model_num_digits": 4,
    "write_profile_for_best_model": False,
    "best_model_profile_filename": "",
    "save_model_for_best_model": False,
    "best_model_save_model_filename": "",
    "save_info_for_last_model": False,
    "last_model_save_info_filename": "",
    "shell_script_for_each_sample": "",
    "shell_script_num_string_char": "#",
    "simplex_just_call_my_extras_check_model": False,
    "trace_limits": False,
    "save_controls": False,
    "save_controls_filename": "",
    "Y_frac_he3": 1.0e-4,
    "read_extra_simplex_search_inlist1": False,
    "extra_simplex_search_inlist1_name": "undefined",
    "read_extra_simplex_search_inlist2": False,
    "extra_simplex_search_inlist2_name": "undefined",
    "read_extra_simplex_search_inlist3": False,
    "extra_simplex_search_inlist3_name": "undefined",
    "read_extra_simplex_search_inlist4": False,
    "extra_simplex_search_inlist4_name": "undefined",
    "read_extra_simplex_search_inlist5": False,
    "extra_simplex_search_inlist5_name": "undefined",
}
_SIMPLEX_KEY_CANONICAL: dict[str, str] = {
    key.lower(): key for key in _SIMPLEX_SEARCH_DEFAULTS
}
_SOLAR_CSOUND_HEADER_LINES = 11
_SOLAR_CSOUND_NPTS = 79
_SOLAR_CSOUND_MIN_R = 0.094
_SOLAR_CSOUND_MAX_R = 0.94


def _strip_fortran_comment(line: str) -> str:
    """Strip inline `!` comments outside Fortran string literals."""
    quote_char = ""
    out: list[str] = []
    idx = 0
    while idx < len(line):
        ch = line[idx]
        if quote_char:
            out.append(ch)
            if ch == quote_char:
                if idx + 1 < len(line) and line[idx + 1] == quote_char:
                    out.append(line[idx + 1])
                    idx += 2
                    continue
                quote_char = ""
        elif ch in ("'", '"'):
            quote_char = ch
            out.append(ch)
        elif ch == "!":
            break
        else:
            out.append(ch)
        idx += 1
    return "".join(out)


def _extract_simplex_namelist_body(text: str, path: pathlib.Path) -> str:
    """Extract `&simplex_search_controls ... /` body from Fortran inlist."""
    cleaned = "\n".join(
        _strip_fortran_comment(line) for line in text.splitlines()
    )
    marker = "&simplex_search_controls"
    lowered = cleaned.lower()
    start = lowered.find(marker)
    while start != -1:
        end_hdr = start + len(marker)
        if end_hdr == len(cleaned) or cleaned[end_hdr].isspace():
            break
        start = lowered.find(marker, end_hdr)
    if start == -1:
        msg = f"missing &simplex_search_controls namelist in {path}"
        raise ValueError(msg)

    idx = start + len(marker)
    quote_char = ""
    while idx < len(cleaned):
        ch = cleaned[idx]
        if quote_char:
            if ch == quote_char:
                if idx + 1 < len(cleaned) and cleaned[idx + 1] == quote_char:
                    idx += 2
                    continue
                quote_char = ""
        elif ch in ("'", '"'):
            quote_char = ch
        elif ch == "/":
            return cleaned[start + len(marker) : idx]
        idx += 1
    msg = f"unterminated &simplex_search_controls namelist in {path}"
    raise ValueError(msg)


def _split_namelist_assignments(body: str) -> list[tuple[str, str]]:
    """Split namelist body into `(raw_key, raw_value)` assignment pairs."""
    statements: list[str] = []
    buf: list[str] = []
    quote_char = ""
    idx = 0
    while idx < len(body):
        ch = body[idx]
        if quote_char:
            buf.append(ch)
            if ch == quote_char:
                if idx + 1 < len(body) and body[idx + 1] == quote_char:
                    buf.append(body[idx + 1])
                    idx += 2
                    continue
                quote_char = ""
        elif ch in ("'", '"'):
            quote_char = ch
            buf.append(ch)
        elif ch in ("\n", "\r", ","):
            piece = "".join(buf).strip()
            if piece:
                statements.append(piece)
            buf.clear()
        else:
            buf.append(ch)
        idx += 1
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)

    pairs: list[tuple[str, str]] = []
    for stmt in statements:
        if "=" not in stmt:
            msg = f"malformed namelist entry {stmt!r}"
            raise ValueError(msg)
        raw_k, raw_v = stmt.split("=", 1)
        pairs.append((raw_k.strip(), raw_v.strip()))
    return pairs


def _parse_namelist_value(
    key: str,
    raw_val: str,
    template: bool | int | float | str,
) -> bool | int | float | str:
    """Parse one Fortran namelist value according to the default's type."""
    if isinstance(template, bool):
        low = raw_val.lower()
        if low in (".true.", "true", "t"):
            return True
        if low in (".false.", "false", "f"):
            return False
        msg = f"invalid boolean for {key!r}: {raw_val!r}"
        raise ValueError(msg)
    if isinstance(template, int):
        return int(raw_val)
    if isinstance(template, float):
        return _parse_fortran_float(raw_val)
    if (
        len(raw_val) >= 2
        and raw_val[0] in ("'", '"')
        and raw_val[-1] == raw_val[0]
    ):
        q = raw_val[0]
        return raw_val[1:-1].replace(q + q, q)
    msg = f"invalid quoted string for {key!r}: {raw_val!r}"
    raise ValueError(msg)


def _read1_simplex_search_inlist(
    base_dir: pathlib.Path,
    filename: str,
    cfg: dict[str, Any],
    level: int,
) -> None:
    """Recursively read `filename` into `cfg` matching upstream order
    (`star/test_suite/simplex_solar_calibration/src/simplex_search_run_support.f90:1931-2035`).
    """
    if level >= 10:
        msg = "too many levels of nested extra simplex_search inlist files"
        raise ValueError(msg)

    inlist_path = pathlib.Path(filename)
    if not inlist_path.is_absolute():
        inlist_path = base_dir / inlist_path
    body = _extract_simplex_namelist_body(
        inlist_path.read_text(encoding="utf-8"),
        inlist_path,
    )
    for raw_k, raw_v in _split_namelist_assignments(body):
        canonical = _SIMPLEX_KEY_CANONICAL.get(raw_k.lower())
        if canonical is None:
            msg = (
                f"unknown simplex_search_controls key {raw_k!r} "
                f"in {inlist_path}"
            )
            raise ValueError(msg)
        cfg[canonical] = _parse_namelist_value(
            canonical,
            raw_v,
            _SIMPLEX_SEARCH_DEFAULTS[canonical],
        )

    pending: list[str] = []
    for slot in range(1, 6):
        flag_key = f"read_extra_simplex_search_inlist{slot}"
        name_key = f"extra_simplex_search_inlist{slot}_name"
        do_read = bool(cfg[flag_key])
        extra_name = str(cfg[name_key])
        cfg[flag_key] = False
        cfg[name_key] = "undefined"
        if do_read:
            pending.append(extra_name)

    for extra_name in pending:
        _read1_simplex_search_inlist(base_dir, extra_name, cfg, level + 1)


def read_simplex_search_controls(
    case_dir: pathlib.Path | str,
    filename: str = "inlist_simplex_search_controls",
) -> dict[str, Any]:
    """Parse `&simplex_search_controls` and recursive extra includes
    (`star/test_suite/simplex_solar_calibration/src/simplex_search_run_support.f90:1916-2035`).
    """
    cfg: dict[str, Any] = dict(_SIMPLEX_SEARCH_DEFAULTS)
    _read1_simplex_search_inlist(pathlib.Path(case_dir), filename, cfg, 1)
    return cfg


def _load_solar_csound_data() -> tuple[list[float], list[float], list[float]]:
    """Load `data/star_data/solar_csound.data` (`npts = 79`, 11-line header;
    `star/test_suite/simplex_solar_calibration/src/simplex_search_run_support.f90:1540-1588`).
    """
    data_path = (
        fm_fetch.resolve_mesa_test_inputs_dir()
        / "data"
        / "star_data"
        / "solar_csound.data"
    )
    if not data_path.is_file():
        msg = f"missing solar_csound.data at {data_path}"
        raise FileNotFoundError(msg)

    lines = data_path.read_text(encoding="utf-8").splitlines()
    data_r: list[float] = []
    data_csound: list[float] = []
    data_width: list[float] = []
    for line in lines[_SOLAR_CSOUND_HEADER_LINES:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        data_r.append(_parse_fortran_float(parts[0]))
        data_csound.append(_parse_fortran_float(parts[2]))
        data_width.append(_parse_fortran_float(parts[6]))
        if len(data_r) == _SOLAR_CSOUND_NPTS:
            break

    if len(data_r) != _SOLAR_CSOUND_NPTS:
        msg = (
            f"expected {_SOLAR_CSOUND_NPTS} rows in {data_path}, "
            f"got {len(data_r)}"
        )
        raise ValueError(msg)
    return (data_r, data_csound, data_width)


def _check_one_sigma_limit(
    value: float,
    target: float,
    sigma: float,
    sigmas_coeff: float,
) -> bool:
    """Return True if `value` breaches `target + sigma * sigmas_coeff`
    (`star/test_suite/simplex_solar_calibration/src/simplex_search_run_support.f90:1332-1535`).
    """
    if sigmas_coeff == 0.0 or sigma <= 0.0:
        return False
    limit = target + sigma * sigmas_coeff
    if sigmas_coeff > 0.0 and value > limit:
        return True
    return sigmas_coeff < 0.0 and value < limit


class SimplexSolarCalibrationStarExtras:
    """Simplex solar-calibration callbacks and `gradr_grada_function` mesh
    weight from
    `star/test_suite/simplex_solar_calibration/src/run_star_extras.f90:41-251`
    and
    `star/test_suite/simplex_solar_calibration/src/simplex_search_run_support.f90`
    lines `855-1800`.

    Applies single-sample (`just_do_first_values`) startup controls,
    age-target timestep ramp-down, `dt_next` clamp, observable extraction,
    and `chi2` termination checks; multi-sample simplex optimization remains
    unported (`#162`).
    """

    def __init__(self) -> None:
        self._active = False
        self._cfg: dict[str, Any] = {}
        self._initial_max_years_for_timestep = -1.0
        self._simplex_using_revised_max_yr_dt = False
        self._simplex_revised_max_yr_dt = 0.0
        self._simplex_max_dt_next = 1.0e99
        self._best_chi2 = -1.0
        self._num_chi2_too_big = 0
        self._avg_age_top_samples = 1.0e99
        self._avg_age_sigma = 0.0
        self._avg_model_number_top_samples = 1.0e99
        self._avg_model_number_sigma = 0.0
        self._rcz = 0.0
        self._feh = -1.0
        self._logg = 0.0
        self._logr = 0.0
        self._surface_x = 0.0
        self._surface_he = 0.0
        self._surface_z = 0.0
        self._surface_z_div_x = 0.0
        self._solar_cs_rms = 0.0
        self._chi2 = -1.0
        self._my_var1 = 0.0
        self._my_var2 = 0.0
        self._my_var3 = 0.0
        self._solar_data: (
            tuple[list[float], list[float], list[float]] | None
        ) = None

    def _reset_run_state(
        self,
        cfg: dict[str, Any],
        initial_max_years: float,
    ) -> None:
        self._active = True
        self._cfg = cfg
        self._initial_max_years_for_timestep = initial_max_years
        self._simplex_using_revised_max_yr_dt = False
        self._simplex_revised_max_yr_dt = 0.0
        self._simplex_max_dt_next = 1.0e99
        self._best_chi2 = -1.0
        self._num_chi2_too_big = 0
        self._rcz = 0.0
        self._chi2 = -1.0
        self._feh = -1.0
        self._my_var1 = 0.0
        self._my_var2 = 0.0
        self._my_var3 = 0.0

    def calc_current_rms(self, state: dict[str, Any]) -> float:
        """Compute unweighted RMS sound-speed residual vs `solar_csound.data`
        (`star/test_suite/simplex_solar_calibration/src/simplex_search_run_support.f90:1591-1666`).
        """
        if self._solar_data is None:
            self._solar_data = _load_solar_csound_data()
        data_r, data_csound, _data_width = self._solar_data

        nz = int(state["nz"])
        r_np = np.asarray(state["r"], dtype=np.float64)
        dq = [float(v) for v in state["dq"]]
        csound = [float(v) for v in state["csound"]]
        f1: list[float] = [csound[0]]
        for k in range(1, nz):
            f1.append(
                (csound[k] * dq[k - 1] + csound[k - 1] * dq[k])
                / (dq[k - 1] + dq[k])
            )

        coeffs = interp_pm(r_np, np.asarray(f1, dtype=np.float64))
        sum_y2 = 0.0
        sum_dr = 0.0
        for r_pt, cs_obs in zip(data_r, data_csound, strict=True):
            if r_pt < _SOLAR_CSOUND_MIN_R or r_pt > _SOLAR_CSOUND_MAX_R:
                continue
            cs = float(interp_value(r_np, coeffs, constants.RSUN * r_pt))
            sum_dr += 1.0
            rel = (cs - cs_obs) / cs_obs
            sum_y2 += rel * rel
        return math.sqrt(sum_y2 / sum_dr)

    def _update_observables(self, state: dict[str, Any]) -> None:
        """Compute surface composition, gravity, radius, `Rcz`, and sound-speed
        RMS from
        `star/test_suite/simplex_solar_calibration/src/simplex_search_run_support.f90`
        lines `1170-1194`.
        """
        cfg = self._cfg
        surface_x = max(float(state["surface_h1"]), 1.0e-10)
        surface_he = float(state["surface_he3"]) + float(state["surface_he4"])
        surface_z = max(1.0e-99, min(1.0, 1.0 - (surface_x + surface_he)))
        surface_z_div_x = surface_z / surface_x

        self._surface_x = surface_x
        self._surface_he = surface_he
        self._surface_z = surface_z
        self._surface_z_div_x = surface_z_div_x
        self._feh = _rust.log10(surface_z_div_x / float(cfg["Z_div_X_solar"]))
        self._logg = _rust.log10(float(state["grav"][0]))
        self._logr = _rust.log10(float(state["photosphere_r"]))

        if not bool(cfg["include_Rcz_in_chi2"]):
            self._rcz = 0.0
        else:
            conv_mix = int(constants.MixingType.CONVECTIVE_MIXING)
            nz = int(state["nz"])
            mix = state["mixing_type"]
            r_arr = state["r"]
            for i in range(nz - 1):
                if int(mix[i + 1]) != conv_mix and int(mix[i]) == conv_mix:
                    if (
                        float(r_arr[i + 1]) > 0.25 * constants.RSUN
                        and float(r_arr[i]) < 0.9 * constants.RSUN
                    ):
                        self._rcz = float(r_arr[i]) / constants.RSUN
                        break

        if bool(cfg["include_solar_cs_rms_in_chi2"]) or bool(
            cfg["report_solar_cs_rms"]
        ):
            self._solar_cs_rms = self.calc_current_rms(state)
        else:
            self._solar_cs_rms = 0.0

        my1 = set_simplex_my_vars(state, bool(cfg["include_my_var1_in_chi2"]))
        if my1 is not None:
            self._my_var1 = my1

    def _limits_tripped(self, state: dict[str, Any]) -> bool:
        """Check age and sigma-based early termination bounds
        (`star/test_suite/simplex_solar_calibration/src/simplex_search_run_support.f90:1318-1537`).
        """
        cfg = self._cfg
        if float(state["star_age"]) >= float(cfg["max_age_for_chi2"]):
            return True
        checks = (
            (
                float(state["teff"]),
                "Teff_target",
                "Teff_sigma",
                "sigmas_coeff_for_Teff_limit",
            ),
            (
                self._logg,
                "logg_target",
                "logg_sigma",
                "sigmas_coeff_for_logg_limit",
            ),
            (
                float(state["log_surface_luminosity"]),
                "logL_target",
                "logL_sigma",
                "sigmas_coeff_for_logL_limit",
            ),
            (
                self._logr,
                "logR_target",
                "logR_sigma",
                "sigmas_coeff_for_logR_limit",
            ),
            (
                self._surface_z_div_x,
                "surface_Z_div_X_target",
                "surface_Z_div_X_sigma",
                "sigmas_coeff_for_surface_Z_div_X_limit",
            ),
            (
                self._surface_he,
                "surface_He_target",
                "surface_He_sigma",
                "sigmas_coeff_for_surface_He_limit",
            ),
            (
                self._rcz,
                "Rcz_target",
                "Rcz_sigma",
                "sigmas_coeff_for_solar_Rcz_limit",
            ),
            (
                self._solar_cs_rms,
                "solar_cs_rms_target",
                "solar_cs_rms_sigma",
                "sigmas_coeff_for_solar_cs_rms_limit",
            ),
            (
                self._my_var1,
                "my_var1_target",
                "my_var1_sigma",
                "sigmas_coeff_for_my_var1_limit",
            ),
            (
                self._my_var2,
                "my_var2_target",
                "my_var2_sigma",
                "sigmas_coeff_for_my_var2_limit",
            ),
            (
                self._my_var3,
                "my_var3_target",
                "my_var3_sigma",
                "sigmas_coeff_for_my_var3_limit",
            ),
        )
        return any(
            _check_one_sigma_limit(
                val,
                float(cfg[t_k]),
                float(cfg[s_k]),
                float(cfg[c_k]),
            )
            for val, t_k, s_k, c_k in checks
        )

    def _compute_chi2(self, state: dict[str, Any]) -> float:
        """Compute evenly weighted chi-squared over enabled observables
        (`star/test_suite/simplex_solar_calibration/src/simplex_search_run_support.f90:1671-1800`).
        """
        cfg = self._cfg
        my1 = set_simplex_my_vars(state, bool(cfg["include_my_var1_in_chi2"]))
        if my1 is not None:
            self._my_var1 = my1
        terms = [
            (
                bool(cfg["include_Teff_in_chi2"]),
                float(state["teff"]),
                float(cfg["Teff_target"]),
                float(cfg["Teff_sigma"]),
            ),
            (
                bool(cfg["include_logL_in_chi2"]),
                float(state["log_surface_luminosity"]),
                float(cfg["logL_target"]),
                float(cfg["logL_sigma"]),
            ),
            (
                bool(cfg["include_logg_in_chi2"]),
                self._logg,
                float(cfg["logg_target"]),
                float(cfg["logg_sigma"]),
            ),
            (
                bool(cfg["include_FeH_in_chi2"]),
                self._feh,
                float(cfg["FeH_target"]),
                float(cfg["FeH_sigma"]),
            ),
            (
                bool(cfg["include_logR_in_chi2"]),
                self._logr,
                float(cfg["logR_target"]),
                float(cfg["logR_sigma"]),
            ),
            (
                bool(cfg["include_age_in_chi2"]),
                float(state["star_age"]),
                float(cfg["age_target"]),
                float(cfg["age_sigma"]),
            ),
            (
                bool(cfg["include_surface_Z_div_X_in_chi2"]),
                self._surface_z_div_x,
                float(cfg["surface_Z_div_X_target"]),
                float(cfg["surface_Z_div_X_sigma"]),
            ),
            (
                bool(cfg["include_surface_He_in_chi2"]),
                self._surface_he,
                float(cfg["surface_He_target"]),
                float(cfg["surface_He_sigma"]),
            ),
            (
                bool(cfg["include_Rcz_in_chi2"]),
                self._rcz,
                float(cfg["Rcz_target"]),
                float(cfg["Rcz_sigma"]),
            ),
            (
                bool(cfg["include_solar_cs_rms_in_chi2"]),
                self._solar_cs_rms,
                float(cfg["solar_cs_rms_target"]),
                float(cfg["solar_cs_rms_sigma"]),
            ),
            (
                bool(cfg["include_my_var1_in_chi2"]),
                self._my_var1,
                float(cfg["my_var1_target"]),
                float(cfg["my_var1_sigma"]),
            ),
            (
                bool(cfg["include_my_var2_in_chi2"]),
                self._my_var2,
                float(cfg["my_var2_target"]),
                float(cfg["my_var2_sigma"]),
            ),
            (
                bool(cfg["include_my_var3_in_chi2"]),
                self._my_var3,
                float(cfg["my_var3_target"]),
                float(cfg["my_var3_sigma"]),
            ),
        ]
        chi2, _ = eval_simplex_chi2(terms)
        self._chi2 = chi2
        return chi2

    def _check_too_many_bad(self) -> str:
        """Increment bad-chi2 counter and terminate above limit
        (`star/test_suite/simplex_solar_calibration/src/simplex_search_run_support.f90:1276-1286`).
        """
        if self._best_chi2 > 0.0:
            self._num_chi2_too_big += 1
            if self._num_chi2_too_big > int(
                self._cfg["limit_num_chi2_too_big"]
            ):
                return "terminate"
            return "keep_going"
        self._num_chi2_too_big = 0
        return "keep_going"

    def _adjust_age_target_timestep(
        self,
        state: dict[str, Any],
        updates: dict[str, Any],
    ) -> float:
        """Apply `eval_chi2_at_target_age_only` / `include_age_in_chi2`
        timestep ramp-down
        (`star/test_suite/simplex_solar_calibration/src/simplex_search_run_support.f90:1060-1130`).
        """
        cfg = self._cfg
        n_small = int(cfg["num_smaller_steps_before_age_target"])
        dt_small = float(cfg["dt_for_smaller_steps_before_age_target"])
        if n_small <= 0 or dt_small <= 0.0:
            msg = (
                "must set positive num_smaller_steps_before_age_target and "
                "dt_for_smaller_steps_before_age_target"
            )
            raise ValueError(msg)

        star_age = float(state["star_age"])
        age_target = float(cfg["age_target"])
        max_years = float(state["max_years_for_timestep"])

        if age_target > star_age:
            remaining_years = age_target - star_age
            if self._simplex_using_revised_max_yr_dt:
                max_years = self._simplex_revised_max_yr_dt
                updates["max_years_for_timestep"] = max_years

            n = math.floor(remaining_years / max_years + 1.0e-6)
            if remaining_years <= max_years:
                max_years = remaining_years
                updates["max_years_for_timestep"] = max_years
                self._simplex_using_revised_max_yr_dt = True
                self._simplex_revised_max_yr_dt = max_years
                self._simplex_max_dt_next = max_years * constants.SECYER
            elif n <= n_small:
                prev_max_years = max_years
                i_steps = math.floor(remaining_years / dt_small + 1.0e-6)
                if (i_steps + 1.0e-9) * dt_small < remaining_years:
                    max_years = remaining_years / float(i_steps + 1)
                else:
                    max_years = remaining_years / float(i_steps)
                red = float(state["reduction_factor_for_max_timestep"])
                min_max = prev_max_years * red
                if max_years < min_max:
                    max_years = min_max
                self._simplex_using_revised_max_yr_dt = True
                updates["max_years_for_timestep"] = max_years
                self._simplex_revised_max_yr_dt = max_years
                if float(state["dt_next"]) / constants.SECYER > max_years:
                    self._simplex_max_dt_next = max_years * constants.SECYER
        elif bool(cfg["include_age_in_chi2"]):
            if abs(max_years - dt_small) > dt_small * 1.0e-2:
                msg = "bad max_years_for_timestep"
                raise RuntimeError(msg)
        return max_years

    def _run_check_model_body(
        self,
        state: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Execute `do_simplex_extras_check_model` returning `(action, updates)`
        (`star/test_suite/simplex_solar_calibration/src/simplex_search_run_support.f90:1041-1390`).
        """
        cfg = self._cfg
        self._simplex_max_dt_next = 1.0e99
        self._chi2 = -1.0
        self._feh = -1.0
        updates: dict[str, Any] = {}

        star_age = float(state["star_age"])
        checking_age = bool(cfg["eval_chi2_at_target_age_only"]) or bool(
            cfg["include_age_in_chi2"]
        )
        if checking_age:
            self._adjust_age_target_timestep(state, updates)
        elif star_age < float(cfg["min_age_for_chi2"]):
            return ("keep_going", updates)

        if bool(cfg["include_age_in_chi2"]) and star_age < float(
            cfg["min_age_for_chi2"]
        ):
            return ("keep_going", updates)
        if bool(cfg["eval_chi2_at_target_age_only"]) and star_age < float(
            cfg["age_target"]
        ):
            return ("keep_going", updates)

        if float(state["power_nuc_burn"]) < float(state["l_phot"]) * float(
            cfg["Lnuc_div_L_limit"]
        ) or star_age < float(cfg["min_age_limit"]):
            return ("keep_going", updates)

        if not checking_age:
            age_limit = (
                self._avg_age_top_samples
                + float(cfg["avg_age_sigma_limit"]) * self._avg_age_sigma
            )
            if star_age > age_limit:
                return ("terminate", updates)
            model_limit = (
                self._avg_model_number_top_samples
                + float(cfg["avg_model_number_sigma_limit"])
                * self._avg_model_number_sigma
            )
            if float(state["model_number"]) > model_limit:
                return ("terminate", updates)

        self._update_observables(state)
        if self._limits_tripped(state):
            return ("terminate", updates)

        chi2 = self._compute_chi2(state)
        if math.isnan(chi2) or math.isinf(chi2):
            self._chi2 = 1.0e99
            return ("terminate", updates)
        if chi2 > float(cfg["chi2_limit"]):
            return (self._check_too_many_bad(), updates)

        if not checking_age:
            for chi2_key, dt_key in (
                (
                    "chi2_limit_for_smallest_timesteps",
                    "max_yrs_dt_chi2_smallest_limit",
                ),
                (
                    "chi2_limit_for_smaller_timesteps",
                    "max_yrs_dt_chi2_smaller_limit",
                ),
                (
                    "chi2_limit_for_small_timesteps",
                    "max_yrs_dt_chi2_small_limit",
                ),
            ):
                if chi2 <= float(cfg[chi2_key]):
                    dt_lim = float(cfg[dt_key])
                    updates["max_years_for_timestep"] = dt_lim
                    if float(state["dt"]) > dt_lim * constants.SECYER:
                        updates["dt"] = dt_lim * constants.SECYER
                        updates["timestep_hold"] = (
                            int(state["model_number"]) + 10
                        )
                        return ("redo", updates)
                    break

        if self._best_chi2 <= 0.0 or chi2 < self._best_chi2:
            self._best_chi2 = chi2

        directive = "keep_going"
        if bool(cfg["include_age_in_chi2"]) and star_age >= float(
            cfg["max_age_for_chi2"]
        ):
            directive = "terminate"
        if bool(cfg["eval_chi2_at_target_age_only"]) and star_age >= float(
            cfg["age_target"]
        ):
            directive = "terminate"
        if self._best_chi2 > 0.0:
            if self._best_chi2 <= float(
                cfg["chi2_search_limit1"]
            ) and chi2 >= float(cfg["chi2_search_limit2"]):
                return ("terminate", updates)
            if chi2 >= float(cfg["chi2_relative_increase_limit"]) * (
                self._best_chi2
            ):
                self._num_chi2_too_big += 1
                if self._num_chi2_too_big > int(cfg["limit_num_chi2_too_big"]):
                    return ("terminate", updates)
                return (directive, updates)
            self._num_chi2_too_big = 0
        return (directive, updates)

    def extras_controls(self, state: dict[str, Any]) -> dict[str, Any]:
        """Parse `inlist_simplex_search_controls` and return startup updates
        (`star/test_suite/simplex_solar_calibration/src/simplex_search_run_support.f90:775-975`).
        """
        raw_case_dir = state.get("case_dir") if "case_dir" in state else None
        if not raw_case_dir:
            self._active = False
            return {}
        case_dir = pathlib.Path(str(raw_case_dir))
        if not (case_dir / "inlist_simplex_search_controls").is_file():
            self._active = False
            return {}

        cfg = read_simplex_search_controls(case_dir)
        if not bool(cfg["just_do_first_values"]):
            msg = (
                "multi-sample simplex search optimization is not ported (#162)"
            )
            raise NotImplementedError(msg)

        self._reset_run_state(cfg, float(state["max_years_for_timestep"]))
        if bool(cfg["simplex_just_call_my_extras_check_model"]):
            return {}

        alpha = float(cfg["first_alpha"])
        f_ov = float(cfg["first_f_ov"])
        feh = float(cfg["first_FeH"])
        initial_z_div_x = float(cfg["Z_div_X_solar"]) * _rust.exp10(feh)
        if bool(cfg["Y_depends_on_Z"]):
            a_val = initial_z_div_x
            b_val = float(cfg["dYdZ"])
            y0_val = float(cfg["Y0"])
            c_val = 1.0 + a_val * (1.0 + b_val)
            x_val = (1.0 - y0_val) / c_val
            y_val = (y0_val + a_val * (b_val + y0_val)) / c_val
        else:
            y_val = float(cfg["first_Y"])
            x_val = (1.0 - y_val) / (1.0 + initial_z_div_x)

        new_mass = float(cfg["first_mass"])
        he3_val = float(cfg["Y_frac_he3"]) * y_val
        he4_val = y_val - he3_val

        updates: dict[str, Any] = {
            "mixing_length_alpha": alpha,
            "initial_mass": new_mass,
            "job_new_mass": new_mass,
            "job_relax_initial_mass": True,
            "job_initial_h1": x_val,
            "job_initial_h2": 0.0,
            "job_initial_he3": he3_val,
            "job_initial_he4": he4_val,
            "job_set_uniform_initial_composition": True,
        }
        if f_ov != 0.0:
            updates["overshoot_slot_0"] = {
                "scheme": "exponential",
                "zone_type": "any",
                "zone_loc": "any",
                "bdy_loc": "any",
                "f": f_ov,
                "f0": float(cfg["f0_ov_div_f_ov"]) * f_ov,
            }
        return updates

    def extras_startup(self, _state: dict[str, Any], _restart: bool) -> None:
        """No-op test-suite startup
        (`star/test_suite/simplex_solar_calibration/src/run_star_extras.f90:104-113`).
        """
        return None

    def extras_check_model(self, state: dict[str, Any]) -> str | dict[str, Any]:
        """Apply `simplex_extras_check_model` age-target and chi2 checks from
        `star/test_suite/simplex_solar_calibration/src/simplex_search_run_support.f90`
        lines `977-988` and `1041-1390`.
        """
        if not self._active:
            return "keep_going"
        if bool(self._cfg["simplex_just_call_my_extras_check_model"]):
            return "keep_going"
        action, updates = self._run_check_model_body(state)
        if updates:
            return {"action": action, "updates": updates}
        return action

    def extras_finish_step(
        self, state: dict[str, Any]
    ) -> str | tuple[str, dict[str, Any]]:
        """Clamp `dt_next` to `simplex_max_dt_next`
        (`star/test_suite/simplex_solar_calibration/src/simplex_search_run_support.f90:991-1000`).
        """
        if not self._active:
            return "keep_going"
        dt_next = float(state["dt_next"])
        if self._simplex_max_dt_next < dt_next:
            return ("keep_going", {"dt_next": self._simplex_max_dt_next})
        return "keep_going"

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """Evaluate final `chi2` on post-evolve model when at age target
        (`star/test_suite/simplex_solar_calibration/src/simplex_search_run_support.f90:1003-1019`).
        """
        if not self._active:
            return None
        if bool(self._cfg["simplex_just_call_my_extras_check_model"]):
            return None
        cfg = self._cfg
        star_age = float(state["star_age"])
        at_target = (
            bool(cfg["eval_chi2_at_target_age_only"])
            and star_age >= float(cfg["age_target"])
        ) or (
            bool(cfg["include_age_in_chi2"])
            and star_age >= float(cfg["max_age_for_chi2"])
        )
        if at_target or bool(cfg["save_info_for_last_model"]):
            self._run_check_model_body(state)
        return None

    def other_mesh_functions(
        self, state: dict[str, Any]
    ) -> list[tuple[str, bool, list[float]]]:
        """Return `gradr_grada_function` custom mesh weight
        (`star/test_suite/simplex_solar_calibration/src/run_star_extras.f90:211-251`).
        """
        return [eval_gradr_grada_mesh_fcn(state)]
