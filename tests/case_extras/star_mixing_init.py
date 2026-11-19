# SPDX-License-Identifier: LGPL-3.0-or-later
"""Upstream mixing, envelope-builder, and TOV-gravity `StarExtras` ports
(`#231`).

Covers:
- `star/test_suite/extended_convective_penetration/src/run_star_extras.f90`
- `star/test_suite/make_env/src/run_star_extras.f90`
- `star/test_suite/tzo/src/run_star_extras.f90`
"""

from __future__ import annotations

import math
from typing import Any

from flexmesa import _rust, constants, num
from tests.case_extras import _helpers


def evaluate_hp(
    ln_peos: float,
    lnd: float,
    cgrav: float,
    m_center: float,
    xmstar: float,
    q: float,
    r: float,
) -> float:
    """Evaluate local pressure scale height $H_P$ in $\\mathrm{cm}$
    (`extended_convective_penetration/src/run_star_extras.f90:437-450`).
    """
    p_val = _rust.crmath_exp(ln_peos)
    rho_val = _rust.crmath_exp(lnd)
    return p_val / (rho_val * cgrav * (m_center + xmstar * q) / (r * r))


def calc_peclet_number(state: dict[str, Any], k: int) -> float:
    """Evaluate cell Peclet number (dimensionless) at 0-based index `k`
    (`extended_convective_penetration/src/run_star_extras.f90:420-435`).
    """
    t_k = float(state["t"][k])
    rho_k = float(state["rho"][k])
    conductivity = (
        16.0
        * constants.BOLTZ_SIGMA
        * (t_k * t_k * t_k)
        / (
            3.0
            * float(state["opacity"][k])
            * (rho_k * rho_k)
            * float(state["cp"][k])
        )
    )
    hp = evaluate_hp(
        float(state["ln_peos"][k]),
        float(state["lnd"][k]),
        float(state["cgrav"][k]),
        float(state["m_center"]),
        float(state["xmstar"]),
        float(state["q"][k]),
        float(state["r"][k]),
    )
    return (
        float(state["conv_vel"][k])
        * hp
        * float(state["mixing_length_alpha"])
        / conductivity
    )


def _taper_overshoot_params(
    state: dict[str, Any], j_param: int
) -> tuple[float, float, float, float, float]:
    """Extract and mass-taper `(f, f0, f2, d0, delta0)` (`:267-295`)."""
    f = float(state["overshoot_f"][j_param])
    f0 = float(state["overshoot_f0"][j_param])
    f2 = float(state["x_ctrl"][j_param])
    d0 = float(state["overshoot_d0"][j_param])
    delta0 = float(state["overshoot_delta0"][j_param])
    if f < 0.0 or f0 <= 0.0 or f2 < 0.0:
        raise ValueError(
            "for extended convective penetration, must set f0 > 0, and f "
            "and f2 >= 0"
        )
    star_mass = float(state["star_mass"])
    m_on = float(state["overshoot_mass_full_on"][j_param])
    m_off = float(state["overshoot_mass_full_off"][j_param])
    if star_mass < m_on:
        if star_mass > m_off:
            w = (star_mass - m_off) / (m_on - m_off)
            fac = 0.5 * (1.0 - _rust.cospi(w))
            f, f0, f2 = f * fac, f0 * fac, f2 * fac
        else:
            f, f0, f2 = 0.0, 0.0, 0.0
    return f, f0, f2, d0, delta0


def dq_f(r: float, n: int, a1: float) -> tuple[float, float]:
    """Evaluate geometric series residual and derivative
    (`make_env/src/run_star_extras.f90:547-568`).
    """
    sn = 1.0
    r_to_n = _rust.pow(r, float(n))
    f_val = sn * (1.0 - r) - a1 * (1.0 - r_to_n)
    df_dr = -sn + a1 * float(n) * r_to_n / r
    return f_val, df_dr


def calc_dq_factor(n: int, dq1: float) -> float:
    """Solve for geometric cell-mass ratio `dq_factor`
    (`make_env/src/run_star_extras.f90:570-603`).
    """
    root_r, _ = num.safe_root_with_guess(
        lambda rv: dq_f(rv, n, dq1),
        1.08,
        0.0,
        x1_opt=1.01,
        x3_opt=1.05,
        y1_opt=None,
        y3_opt=None,
        newt_imax=0,
        imax=100,
        epsx=1.0e-10,
        epsy=1.0e-10,
    )
    return float(root_r)


def _locate_mass_cells(m_arr: Any, nz: int) -> tuple[int, int, int, int]:
    """Locate 0-based cells crossing `0.8, 0.95, 1.1, 1.5 Msun` (`:168-178`)."""
    k1 = -1
    k2 = -1
    k3 = -1
    k4 = -1
    msun = constants.MSUN
    for k in range(nz - 1, -1, -1):
        m_k = float(m_arr[k])
        if m_k > 0.8 * msun and k1 < 0:
            k1 = k
        if m_k > 0.95 * msun and k2 < 0:
            k2 = k
        if m_k > 1.1 * msun and k3 < 0:
            k3 = k
        if m_k > 1.5 * msun and k4 < 0:
            k4 = k
    # Guard against Python `-1` wraparound where Fortran `k == 0` fails bounds.
    if min(k1, k2, k3, k4) < 0:
        raise IndexError("mass profile does not span 0.8..1.5 Msun")
    return k1, k2, k3, k4


class ExtendedConvectivePenetrationStarExtras:
    """Michielsen et al. (2019) convective penetration and Peclet `gradT` blend.

    Source: `extended_convective_penetration/src/run_star_extras.f90:37-452`.
    """

    def __init__(self) -> None:
        self.checks_okay: bool | None = None

    def extras_controls(self, state: dict[str, Any]) -> None:
        """Skip hook registration during pre-MS creation, else require `#155`
        (`:37-60`).
        """
        if bool(state["create_pre_main_sequence_model"]):
            return
        raise NotImplementedError(
            "extended_convective_penetration: extras_controls registers "
            "other_adjust_mlt_gradT_fraction and other_overshooting_scheme "
            "(:57-58), which flexmesa-star StarExtras does not expose (#155)"
        )

    def other_adjust_mlt_gradt_fraction_peclet(
        self, state: dict[str, Any]
    ) -> list[float]:
        """Compute `adjust_mlt_gradT_fraction` `(nz,)` (`:378-416`)."""
        frac_out = [float(v) for v in state["adjust_mlt_gradt_fraction"]]
        d_mix = state["d_mix"]
        d0 = float(d_mix[0])
        if math.isnan(d0) or math.isinf(d0):
            return frac_out
        if int(state["num_conv_boundaries"]) < 1:
            return frac_out
        nz = int(state["nz"])
        min_d_mix = float(state["min_d_mix"])
        for k in range(nz - 1, -1, -1):
            if float(d_mix[k]) <= min_d_mix:
                break
            pe = calc_peclet_number(state, k)
            if pe >= 100.0:
                frac_out[k] = 1.0
            elif pe <= 0.01:
                frac_out[k] = 0.0
            else:
                frac_out[k] = (_rust.safe_log10(pe) + 2.0) / 4.0
        return frac_out

    def eval_penetration_profile(
        self, state: dict[str, Any], i_bdy: int, j_param: int
    ) -> tuple[int, int, list[float], list[float]]:
        """Evaluate `(k_a, k_b, D, vc)` (0-based indices) for boundary `i_bdy`
        (`:236-372`).
        """
        f, f0, f2, d0, delta0 = _taper_overshoot_params(state, j_param)
        _ = f0
        nz = int(state["nz"])
        hp_cb = float(state["conv_bdy_hp"][i_bdy])
        k_ob = int(state["over_bdy_k"][i_bdy])
        r_ob = float(state["over_bdy_r"][i_bdy])
        d_ob = float(state["over_bdy_d"][i_bdy])
        vc_ob = float(state["over_bdy_vc"][i_bdy])
        outward = bool(state["top_conv_bdy"][i_bdy])
        k_a, k_b, dk = (k_ob, 0, -1) if outward else (k_ob + 1, nz - 1, 1)
        r_step = f * hp_cb if f > 0.0 else 0.0
        d_arr = [0.0] * nz
        vc_arr = [0.0] * nz
        d_min = float(state["overshoot_d_min"])
        r_arr = state["r"]
        k = k_a
        while (k >= k_b) if dk < 0 else (k <= k_b):
            r_k = float(r_arr[k])
            dr = (r_k - r_ob) if outward else (r_ob - r_k)
            if dr < r_step and f > 0.0:
                factor = 1.0
            elif f2 > 0.0:
                factor = _rust.crmath_exp(-2.0 * (dr - r_step) / (f2 * hp_cb))
            else:
                factor = 0.0
            d_arr[k] = (d0 + delta0 * d_ob) * factor
            vc_arr[k] = (
                (d0 / d_ob + delta0) * vc_ob * factor if d_ob != 0.0 else 0.0
            )
            if d_arr[k] < d_min:
                k_b = k
                break
            k += dk
        return k_a, k_b, d_arr, vc_arr

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """Run all 11 post-evolution mixing and `gradT` tolerance checks
        (`:145-232`).
        """
        if bool(state["create_pre_main_sequence_model"]):
            return
        nz = int(state["nz"])
        k1, k2, k3, k4 = _locate_mass_cells(state["m"], nz)
        mix = state["mixing_type"]
        grada = state["grada"]
        gradr = state["gradr"]
        gradt = state["gradt"]
        d_mix = state["d_mix"]
        bdy0 = int(state["conv_bdy_loc"][0])
        pe = calc_peclet_number(state, k3)
        fraction = (_rust.safe_log10(pe) + 2.0) / 4.0
        gradient = fraction * float(grada[k3]) + (1.0 - fraction) * float(
            gradr[k3]
        )
        d_bdy = float(d_mix[bdy0])
        checks = [
            int(mix[k1]) == int(constants.MixingType.CONVECTIVE_MIXING),
            int(mix[k2]) == int(constants.MixingType.OVERSHOOT_MIXING),
            int(mix[k3]) == int(constants.MixingType.OVERSHOOT_MIXING),
            int(mix[k4]) == int(constants.MixingType.MINIMUM_MIXING),
            0.1 <= fraction <= 0.9,
            abs(float(gradt[k1]) - float(grada[k1])) <= 1.0e-3,
            abs(float(gradt[k2]) - float(grada[k2])) <= 1.0e-3,
            abs(float(gradt[k3]) - gradient) <= 1.0e-3,
            abs(float(gradt[k4]) - float(gradr[k4])) <= 1.0e-3,
            0.95 * d_bdy <= float(d_mix[k2]) <= 1.05 * d_bdy,
            1.0e4 <= float(d_mix[k3]) <= 1.0e-2 * d_bdy,
        ]
        self.checks_okay = all(checks)


class MakeEnvStarExtras:
    """Static envelope constructor and geometric mesh-spacing root solver from
    `star/test_suite/make_env/src/run_star_extras.f90:36-605`.
    """

    def extras_controls(self, state: dict[str, Any]) -> None:
        """Return when `x_logical_ctrl(1)` is false; raise `#154` otherwise
        (`:36-57`).
        """
        if not bool(state["x_logical_ctrl"][0]):
            return
        raise NotImplementedError(
            "make_env: create_env (:158-605) requires in-place StarModel "
            "allocation, cell-wise EOS/kap/MLT integration, and "
            "star_write_model in flexmesa-star (#154)"
        )


class TzoStarExtras:
    """Thorne-Zytkow object TOV `other_cgrav` and NS core accretion hooks from
    `star/test_suite/tzo/src/run_star_extras.f90:40-260`.
    """

    def __init__(self) -> None:
        self._use_hydro = False
        self._inlist_part = 0
        self._eta_ledd = 0.0
        self._eta_medd = 0.0
        self._target_mass = 0.0

    def extras_controls(self, state: dict[str, Any]) -> None:
        """Cache `x_logical_ctrl(1)`, `x_integer_ctrl(1)`, and `x_ctrl(1:3)`
        (`:40-69`).
        """
        self._use_hydro = bool(state["x_logical_ctrl"][0])
        self._inlist_part = int(state["x_integer_ctrl"][0])
        x_ctrl = state["x_ctrl"]
        self._eta_ledd = float(x_ctrl[0])
        self._eta_medd = float(x_ctrl[1])
        self._target_mass = float(x_ctrl[2])

    def compute_part4_center_step(
        self, state: dict[str, Any]
    ) -> dict[str, float | bool]:
        """Evaluate part-4 Eddington core accretion quantities (`:108-125`)."""
        nz = int(state["nz"])
        m_center_old = float(state["m_center"])
        kap_center = float(state["opacity"][nz - 1])
        c = constants.CLIGHT
        l_center_edd = (
            4.0 * constants.PI * constants.STANDARD_CGRAV * m_center_old * c
        ) / kap_center
        mdot_center = self._eta_medd * l_center_edd / (c * c)
        m_center_new = m_center_old + mdot_center * float(state["dt"])
        enable_hydro = (
            float(state["star_age"]) > 100.0
            and (not bool(state["v_flag"]))
            and self._use_hydro
        )
        return {
            "enable_v_flag": enable_hydro,
            "m_center": m_center_new,
            "xmstar": float(state["mstar"]) - m_center_new,
            "l_center": self._eta_ledd * l_center_edd,
        }

    def extras_start_step(self, state: dict[str, Any]) -> str:
        """Keep going for parts 1..3; fail loudly in part 4 (`:96-129`)."""
        if self._inlist_part != 4:
            return "keep_going"
        _ = self.compute_part4_center_step(state)
        raise NotImplementedError(
            "tzo: extras_start_step case(4) (:108-125) calls star_set_v_flag "
            "and mutates s% m_center, s% xmstar, and s% L_center, which "
            "flexmesa-star StarExtras updates do not expose (#152)"
        )

    def extras_check_model(
        self, state: dict[str, Any]
    ) -> str | tuple[str, str]:
        """Stop when `m_center / Msun > target_mass` in part 4 (`:133-148`)."""
        m_center_msun = float(state["m_center"]) / constants.MSUN
        if m_center_msun > self._target_mass and self._inlist_part == 4:
            return ("terminate", "PASS: Have reached requested NS mass")
        return "keep_going"

    def other_cgrav(self, state: dict[str, Any]) -> list[float]:
        """Evaluate TOV `cgrav(1:nz)` in $\\mathrm{cm^3\\,g^{-1}\\,s^{-2}}$
        (`:216-259`).
        """
        nz = int(state["nz"])
        r_arr = state["r"]
        m_arr = state["m"]
        rho_arr = state["rho"]
        peos_arr = state["peos"]
        g_const = constants.STANDARD_CGRAV
        c2 = constants.CLIGHT * constants.CLIGHT
        out: list[float] = []
        for k in range(nz):
            r_k = float(r_arr[k])
            m_k = float(m_arr[k])
            if k > 0:
                rho_f = 0.5 * (float(rho_arr[k - 1]) + float(rho_arr[k]))
                p_f = 0.5 * (float(peos_arr[k - 1]) + float(peos_arr[k]))
                f1 = 1.0 + _helpers.fdiv(p_f, rho_f * c2)
                f2 = 1.0 + _helpers.fdiv(
                    4.0 * constants.PI * (r_k * r_k * r_k) * p_f, m_k * c2
                )
            else:
                f1 = 1.0
                f2 = 1.0
            f3 = 1.0 - _helpers.fdiv(2.0 * g_const * m_k, r_k * c2)
            cgrav_k = _helpers.fdiv(g_const * f1 * f2, f3)
            if math.isnan(cgrav_k) or math.isinf(cgrav_k):
                raise RuntimeError(
                    f"my_other_cgrav: non-finite cell {k + 1}: {cgrav_k}"
                )
            out.append(cgrav_k)
        return out
