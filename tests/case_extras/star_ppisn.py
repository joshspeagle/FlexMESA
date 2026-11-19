# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Upstream pulsational pair-instability supernova (`ppisn`) Brott wind,
`mstar_dot` limiter, Fabry+2022 Roche distortion, escape-velocity opacity
override, and 22-column hydro/ejecta history `StarExtras` port (`#231`).

Covers:
- `star/test_suite/ppisn/src/run_star_extras.f90:38-1381`
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from flexmesa import _rust, constants
from flexmesa import math as fm_math
from tests.case_extras import _helpers

_LOG_TERM_POWER: float = 5.626


def compute_gamma1_integral(
    pgas: Sequence[float],
    prad: Sequence[float],
    dm: Sequence[float],
    rho: Sequence[float],
    gamma1: Sequence[float],
) -> float:
    r"""Evaluate pressure-weighted $\langle\Gamma_1 - 4/3\rangle$
    (`star/test_suite/ppisn/src/run_star_extras.f90:934-941, 1104-1111`).
    """
    integral_norm = 0.0
    gamma1_int = 0.0
    four_thirds = 4.0 / 3.0
    for k in range(len(dm)):
        weight = (
            (float(pgas[k]) + float(prad[k])) * float(dm[k]) / float(rho[k])
        )
        integral_norm += weight
        gamma1_int += (float(gamma1[k]) - four_thirds) * weight
    return gamma1_int / max(1.0e-99, integral_norm)


def eval_roche_fp_ft(
    aw: float,
    r: float,
    q_k: float,
    *,
    u_flag: bool,
    w_div_wc_flag: bool,
) -> dict[str, float]:
    r"""Evaluate Fabry et al. (2022) Roche geometry factors (`fp`, `ft`,
    `dfp_daw`, `dft_daw`, `r_polar`, `r_equatorial`)
    (`star/test_suite/ppisn/src/run_star_extras.f90:335-406`).
    """
    w = float(aw)
    w2 = w * w
    w4 = w2 * w2
    w6 = w4 * w2
    w_pow = _rust.pow(w, _LOG_TERM_POWER)
    dw_pow = (
        _LOG_TERM_POWER * _rust.pow(w, _LOG_TERM_POWER - 1.0)
        if w > 0.0
        else 0.0
    )
    one_minus_wpow = 1.0 - w_pow
    w_log_term = float(fm_math.log(one_minus_wpow))
    dw_log_term = _helpers.fdiv(-dw_pow, one_minus_wpow)

    a_omega = 1.0 + 0.3293 * w4 - 0.4926 * w6 - 0.5560 * w_log_term
    da_omega = (
        4.0 * 0.3293 * w * w2 - 6.0 * 0.4926 * w * w4 - 0.5560 * dw_log_term
    )

    two_thirds = constants.TWO_THIRDS
    fp_num = 1.0 - two_thirds * w2 - 0.2133 * w4 - 0.1068 * w6
    dfp_num = (
        -2.0 * two_thirds * w - 4.0 * 0.2133 * w * w2 - 6.0 * 0.1068 * w * w4
    )
    ft_num = 1.0 - 0.07955 * w4 - 0.2322 * w6
    dft_num = -4.0 * 0.07955 * w * w2 - 6.0 * 0.2322 * w * w4

    fp_val = fp_num / a_omega
    dfp_val = (dfp_num * a_omega - fp_num * da_omega) / (a_omega * a_omega)
    ft_val = ft_num / a_omega
    dft_val = (dft_num * a_omega - ft_num * da_omega) / (a_omega * a_omega)

    re_fac = 1.0 + constants.ONE_SIXTH * w2 - 0.005124 * w4 + 0.06562 * w6
    r_eq = r * re_fac
    r_pol = r_eq / (1.0 + 0.5 * w2)
    r_eq = max(r_eq, r)
    r_pol = min(r_pol, r)

    dfp_out = dfp_val if w_div_wc_flag else 0.0
    dft_out = dft_val if w_div_wc_flag else 0.0
    if u_flag:
        if q_k > 0.999:
            fp_val, ft_val = 1.0, 1.0
            dfp_out, dft_out = 0.0, 0.0
        elif q_k > 0.998:
            alpha = 1.0 - (q_k - 0.998) / 1.0e-3
            fp_val = fp_val * alpha + (1.0 - alpha)
            ft_val = ft_val * alpha + (1.0 - alpha)
            dfp_out *= alpha
            dft_out *= alpha

    return {
        "fp": fp_val,
        "ft": ft_val,
        "dfp_daw": dfp_out,
        "dft_daw": dft_out,
        "r_polar": r_pol,
        "r_equatorial": r_eq,
    }


def eval_surface_escape_kap_override(
    state: dict[str, Any], k_1based: int
) -> tuple[float, float, float] | None:
    r"""Evaluate `my_other_kap_get` surface escape-velocity electron-scattering
    override (`star/test_suite/ppisn/src/run_star_extras.f90:408-493`).

    Returns `(kap, dln_kap_dlnRho, dln_kap_dlnT)` when `k_1based == 1`,
    `u_flag` is true, and surface velocity exceeds local escape velocity, or
    `None` when standard `kap_get` applies.
    """
    if k_1based != 1 or not bool(state["u_flag"]):
        return None
    if int(state["solver_iter"]) > 0:
        velocity = float(state["xh_start_u_1"])
        radius = float(state["r_start_1"])
    else:
        velocity = float(state["u"][0])
        radius = float(state["r"][0])
    v_esc = math.sqrt(
        2.0 * float(state["cgrav"][0]) * float(state["m"][0]) / radius
    )
    if velocity > v_esc:
        return (0.2 * (1.0 + float(state["x"][0])), 0.0, 0.0)
    return None


class PpisnStarExtras:
    """Brott et al. (2011) wind (`Z_solar = 0.0142`), `mstar_dot` 5%-per-step
    damper, 22-column ejecta/core history, and pulse-relaxation hooks from
    `star/test_suite/ppisn/src/run_star_extras.f90:38-1381`.
    """

    def __init__(self) -> None:
        self.lx_hydro_on = False
        self.lx_hydro_has_been_on = False
        self.lx_have_saved_gamma_prof = False
        self.lx_have_reached_gamma_limit = False
        self.lx_he_zams = False
        self.x_time_start_pulse = -1.0
        self.x_dyn_time = -1.0
        self.x_gamma_int_bound = -1.0
        self.x_time_since_first_gamma_zero = 0.0
        self.x_star_age_at_relax = -1.0
        self.ix_steps_met_relax_cond = 0
        self.ix_num_relaxations = 0
        self.ix_steps_since_relax = 0
        self.ix_steps_since_hydro_on = 0
        self.stop_100d_after_pulse = False
        self.in_inlist_pulses = False
        self.num_steps_before_relax = 0
        self.ctrl_params: dict[str, float] = {}

    def extras_controls(self, state: dict[str, Any]) -> None:
        """Cache `x_ctrl(1:17)` and guard `other_kap_get` / `other_eval_fp_ft`
        (`:108-168`).
        """
        x_ctrl = state["x_ctrl"]
        self.ctrl_params = {
            "min_gamma_sub_43_for_hydro": float(x_ctrl[0]),
            "max_v_for_pulse": float(x_ctrl[1]),
            "q_for_dyn_ts": float(x_ctrl[2]),
            "num_dyn_ts_for_relax": float(x_ctrl[3]),
            "q_for_relax_check": float(x_ctrl[4]),
            "max_v_for_relax": float(x_ctrl[5]),
            "max_machn_for_relax": float(x_ctrl[6]),
            "max_lneu_for_relax": float(x_ctrl[7]),
            "max_lnuc_for_relax": float(x_ctrl[8]),
            "max_dt_before_pulse": float(x_ctrl[9]),
            "max_lneu_for_mass_loss": float(x_ctrl[10]),
            "delta_lglnuc_limit": float(x_ctrl[11]),
            "max_lphoto_for_lglnuc_limit": float(x_ctrl[12]),
            "max_lphoto_for_lglnuc_limit2": float(x_ctrl[13]),
            "logt_for_v_flag": float(x_ctrl[14]),
            "loglneu_for_v_flag": float(x_ctrl[15]),
            "vsurf_for_fixed_bc": float(x_ctrl[16]),
            "delta_lgrho_cntr_hard_limit": float(
                state["delta_lgrho_cntr_hard_limit"]
            ),
            "dt_div_min_dr_div_cs_limit": float(
                state["dt_div_min_dr_div_cs_limit"]
            ),
        }
        self.num_steps_before_relax = int(state["x_integer_ctrl"][0])
        self.stop_100d_after_pulse = bool(state["x_logical_ctrl"][0])
        self.in_inlist_pulses = bool(state["x_logical_ctrl"][1])
        if bool(state["use_other_kap"]):
            raise NotImplementedError(
                "ppisn: extras_controls registers s% other_kap_get => "
                "my_other_kap_get (:136, 408-493), which flexmesa-star "
                "StarExtras does not expose (#161)"
            )
        if bool(state["use_other_eval_fp_ft"]):
            raise NotImplementedError(
                "ppisn: extras_controls registers s% other_eval_fp_ft => "
                "my_other_eval_fp_ft (:135, 335-406), which flexmesa-star "
                "StarExtras does not expose (#159)"
            )

    def other_wind(
        self,
        state: dict[str, Any],
        l_surf: float,
        m_surf: float,
        r_surf: float,
        t_surf: float,
    ) -> float:
        r"""Evaluate `brott_wind` in $M_\odot\,\mathrm{yr^{-1}}$ using Asplund
        et al. (2009) $Z_\odot = 0.0142$ (`:170-314`).
        """
        lsun = constants.LSUN
        msun = constants.MSUN
        l1 = float(l_surf)
        m1 = float(m_surf)
        r1 = float(r_surf)
        t1 = float(t_surf)
        xs = float(state["surface_h1"])
        _ = float(state["surface_he4"])
        z_div_z_solar = float(state["zbase"]) / 0.0142
        log_z = _rust.log10(z_div_z_solar)
        teff_jump = 1.0e3 * (61.2 + 2.59 * (-13.636 + 0.889 * log_z))

        if t1 > 27500.0:
            alfa = 1.0
        elif t1 < 22500.0:
            alfa = 0.0
        else:
            dt_jump = 100.0
            if t1 > teff_jump + dt_jump:
                alfa = 1.0
            elif t1 < teff_jump - dt_jump:
                alfa = 0.0
            else:
                alfa = (t1 - (teff_jump - dt_jump)) / (2.0 * dt_jump)

        if alfa > 0.0:
            vinf_div_vesc = 2.6 * _rust.pow(z_div_z_solar, 0.13)
            log_t = _rust.log10(t1 / 4.0e4)
            log_mdot = (
                -6.697
                + 2.194 * _rust.log10(l1 / lsun / 1.0e5)
                - 1.313 * _rust.log10(m1 / msun / 30.0)
                - 1.226 * _rust.log10(vinf_div_vesc / 2.0)
                + 0.933 * log_t
                - 10.92 * (log_t * log_t)
                + 0.85 * log_z
            )
            w1 = _rust.exp10(log_mdot)
        else:
            w1 = 0.0

        if alfa < 1.0:
            vinf_div_vesc = 1.3 * _rust.pow(z_div_z_solar, 0.13)
            log_mdot = (
                -6.688
                + 2.210 * _rust.log10(l1 / lsun / 1.0e5)
                - 1.339 * _rust.log10(m1 / msun / 30.0)
                - 1.601 * _rust.log10(vinf_div_vesc / 2.0)
                + 1.07 * _rust.log10(t1 / 2.0e4)
                + 0.85 * log_z
            )
            w2 = _rust.exp10(log_mdot)
        else:
            w2 = 0.0

        vink_wind = alfa * w1 + (1.0 - alfa) * w2
        nieu_wind = _rust.exp10(
            -14.02
            + 1.24 * _rust.log10(l1 / lsun)
            + 0.16 * _rust.log10(m1 / msun)
            + 0.81 * _rust.log10(r1 / constants.RSUN)
            + 0.85 * log_z
        )
        hamann_wind = (
            _rust.exp10(
                -11.95 + 1.5 * _rust.log10(l1 / lsun) - 2.85 * xs + 0.85 * log_z
            )
            / 10.0
        )

        if t1 < teff_jump:
            return max(vink_wind, nieu_wind)
        alfa_hi = 0.0
        if xs > 0.7:
            alfa_hi = 1.0
        elif 0.4 < xs < 0.7:
            alfa_hi = (xs - 0.4) / 0.3
        return alfa_hi * vink_wind + (1.0 - alfa_hi) * hamann_wind

    def other_adjust_mdot(
        self, state: dict[str, Any], current_mdot: float
    ) -> float:
        r"""Clamp `mstar_dot` within $\pm 5\%$ of `mstar_dot_old` when
        `generations > 2` (`:316-333`).
        """
        mdot = float(current_mdot)
        if int(state["generations"]) > 2:
            mdot_old = float(state["mstar_dot_old"])
            if abs(mdot) > 1.05 * abs(mdot_old):
                return 1.05 * mdot_old
            if abs(mdot) < 0.95 * abs(mdot_old):
                return 0.95 * mdot_old
        return mdot

    def extras_startup(self, _state: dict[str, Any], restart: bool) -> None:
        """Initialize `lxtra`/`xtra`/`ixtra` trackers or require `#150`/`#152`
        on restart (`:495-542`).
        """
        if restart:
            raise NotImplementedError(
                "ppisn: extras_startup(restart=True) requires persistent "
                "lxtra/xtra/ixtra photo state (#150) and star_read_controls/"
                "star_set_u_flag (:524-540, #152, #162)"
            )
        self.lx_hydro_on = False
        self.lx_hydro_has_been_on = False
        self.lx_have_saved_gamma_prof = False
        self.lx_have_reached_gamma_limit = False
        self.lx_he_zams = False
        self.x_time_start_pulse = -1.0
        self.x_dyn_time = -1.0
        self.x_gamma_int_bound = -1.0
        self.x_time_since_first_gamma_zero = 0.0
        self.x_star_age_at_relax = -1.0
        self.ix_steps_met_relax_cond = 0
        self.ix_num_relaxations = 0
        self.ix_steps_since_relax = 0
        self.ix_steps_since_hydro_on = 0

    def extras_check_model(self, _state: dict[str, Any]) -> str:
        """Return `keep_going` (`:559-570`)."""
        return "keep_going"

    def extra_history_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, float]]:
        """Evaluate all 22 extra history columns (`:573-738`)."""
        nz = int(state["nz"])
        u_flag = bool(state["u_flag"])
        rot_flag = bool(state["rotation_flag"])
        r_arr = state["r"]
        q_arr = state["q"]
        m_arr = state["m"]
        x_arr = state["x"]
        y_arr = state["y"]
        cgrav = state["cgrav"]
        dm_arr = state["dm"]
        dm_bar = state["dm_bar"]
        energy = state["energy"]
        rsun = constants.RSUN
        msun = constants.MSUN

        u_arr = state["u"] if u_flag else ()
        max_vel = max(float(u_arr[k]) for k in range(nz)) if u_flag else 0.0
        min_vel = min(float(u_arr[k]) for k in range(nz)) if u_flag else 0.0
        yr_since_coll = self.x_time_since_first_gamma_zero / constants.SECYER

        log_r_098 = -100.0
        log_r_095 = -100.0
        log_r_090 = -100.0
        he_core = -100.0
        co_core = 0.0
        log_r_100 = _rust.log10(float(r_arr[0]) / rsun)
        for k in range(nz):
            q_k = float(q_arr[k])
            if q_k < 0.98 and log_r_098 == -100.0:
                log_r_098 = _rust.log10(float(r_arr[k]) / rsun)
            elif q_k < 0.95 and log_r_095 == -100.0:
                log_r_095 = _rust.log10(float(r_arr[k]) / rsun)
            elif q_k < 0.9 and log_r_090 == -100.0:
                log_r_090 = _rust.log10(float(r_arr[k]) / rsun)
            if he_core < 0.0 and float(x_arr[k]) < 0.01:
                he_core = float(m_arr[k]) / msun
            if co_core <= 0.0 and float(y_arr[k]) < 0.01:
                co_core = float(m_arr[k]) / msun

        k_1 = 1
        if u_flag:
            k_scan = 0
            for idx_1 in range(nz, 0, -1):
                idx0 = idx_1 - 1
                v_esc = math.sqrt(
                    2.0
                    * float(cgrav[idx0])
                    * float(m_arr[idx0])
                    / float(r_arr[idx0])
                )
                if float(u_arr[idx0]) > v_esc:
                    k_scan = idx_1
                    break
            k_1 = 1 if k_scan == 0 else k_scan

        if not rot_flag:
            total_j = 0.0
            total_j_bel = 0.0
        else:
            j_rot = state["j_rot"]
            total_j = sum(float(dm_bar[i]) * float(j_rot[i]) for i in range(nz))
            total_j_bel = sum(
                float(dm_bar[i]) * float(j_rot[i]) for i in range(k_1 - 1, nz)
            )

        k0_idx = k_1 - 1
        log_r_vesc = _rust.log10(float(r_arr[k0_idx]) / rsun)
        m_below_vesc = float(m_arr[k0_idx]) / msun
        q_vesc = float(q_arr[k0_idx])
        log_r_vesc_098 = -100.0
        log_r_vesc_095 = -100.0
        log_r_vesc_090 = -100.0
        for i in range(k0_idx, nz):
            ratio = _helpers.fdiv(float(q_arr[i]), q_vesc)
            if ratio < 0.98 and log_r_vesc_098 == -100.0:
                log_r_vesc_098 = _rust.log10(float(r_arr[i]) / rsun)
            elif ratio < 0.95 and log_r_vesc_095 == -100.0:
                log_r_vesc_095 = _rust.log10(float(r_arr[i]) / rsun)
            elif ratio < 0.9 and log_r_vesc_090 == -100.0:
                log_r_vesc_090 = _rust.log10(float(r_arr[i]) / rsun)

        u_ejecta = 0.0
        t_ejecta = 0.0
        omega_ejecta = 0.0
        if k_1 > 1:
            for i in range(k_1):
                dm_i = float(dm_arr[i])
                u_ejecta += dm_i * float(energy[i])
                if u_flag:
                    u_i = float(u_arr[i])
                    t_ejecta += 0.5 * dm_i * u_i * u_i
                omega_ejecta -= (
                    float(dm_bar[i])
                    * float(cgrav[i])
                    * float(m_arr[i])
                    / float(r_arr[i])
                )

        return [
            ("relax_count", float(self.ix_num_relaxations)),
            ("log_R_098", log_r_098),
            ("log_R_095", log_r_095),
            ("log_R_090", log_r_090),
            ("helium_core_mass", he_core),
            ("co_core_mass", co_core),
            ("log_R_100", log_r_100),
            ("log_R_vesc", log_r_vesc),
            ("log_R_vesc_098", log_r_vesc_098),
            ("log_R_vesc_095", log_r_vesc_095),
            ("log_R_vesc_090", log_r_vesc_090),
            ("M_below_vesc", m_below_vesc),
            ("u_flag", 1.0 if u_flag else 0.0),
            ("U_ejecta", u_ejecta),
            ("T_ejecta", t_ejecta),
            ("Omega_ejecta", omega_ejecta),
            ("gamma_integral", self.x_gamma_int_bound),
            ("max_velocity", max_vel),
            ("min_velocity", min_vel),
            ("yr_since_coll", yr_since_coll),
            ("total_J", total_j),
            ("total_J_bel_vesc", total_j_bel),
        ]

    def extra_profile_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, list[float]]]:
        """Evaluate 8 (`u_flag=True`) or 1 (`u_flag=False`) profile columns
        (`:741-811`).
        """
        nz = int(state["nz"])
        u_flag = bool(state["u_flag"])
        rot_flag = bool(state["rotation_flag"])
        m_arr = state["m"]
        spin = [0.0] * nz
        j_inside = 0.0
        if rot_flag:
            j_rot = state["j_rot"]
            dm_bar = state["dm_bar"]
            for k in range(nz - 1, -1, -1):
                j_inside += float(j_rot[k]) * float(dm_bar[k])
                m_k = float(m_arr[k])
                spin[k] = _helpers.fdiv(
                    j_inside * constants.CLIGHT,
                    (m_k * m_k) * constants.STANDARD_CGRAV,
                )
        if not u_flag:
            return [("spin_parameter", spin)]

        cgrav = state["cgrav"]
        r_arr = state["r"]
        u_arr = state["u"]
        t_arr = state["t"]
        mu_arr = state["mu"]
        rho_arr = state["rho"]
        mlt_vc = state["mlt_vc"]
        vesc = [0.0] * nz
        v_div_vesc = [0.0] * nz
        grav_e = [0.0] * nz
        kin_e = [0.0] * nz
        therm_e = [0.0] * nz
        tot_e = [0.0] * nz
        vc_col = [0.0] * nz
        gas_fac = constants.TWO_THIRDS * constants.AVO * constants.KERG
        for k in range(nz - 1, -1, -1):
            c_k = float(cgrav[k])
            m_k = float(m_arr[k])
            r_k = float(r_arr[k])
            u_k = float(u_arr[k])
            t_k = float(t_arr[k])
            rho_k = float(rho_arr[k])
            vesc[k] = math.sqrt(2.0 * c_k * m_k / r_k)
            v_div_vesc[k] = _helpers.fdiv(u_k, vesc[k])
            grav_e[k] = -c_k * m_k / r_k
            kin_e[k] = 0.5 * u_k * u_k
            t2 = t_k * t_k
            therm_e[k] = (
                gas_fac * t_k / (2.0 * float(mu_arr[k]) * rho_k)
                + constants.CRAD * (t2 * t2) / rho_k
            )
            tot_e[k] = grav_e[k] + kin_e[k] + therm_e[k]
            vc_col[k] = float(mlt_vc[k])
        return [
            ("vesc", vesc),
            ("v_div_vesc", v_div_vesc),
            ("specific_grav_e", grav_e),
            ("specific_kin_e", kin_e),
            ("specific_thermal_e", therm_e),
            ("total_specific_e", tot_e),
            ("mlt_vc", vc_col),
            ("spin_parameter", spin),
        ]

    def extras_start_step(self, _state: dict[str, Any]) -> str:
        """Fail loudly on `other_before_struct_burn_mix` and dynamic hydro/relax
        switching (`:814-1228`).
        """
        raise NotImplementedError(
            "ppisn: extras_start_step enables "
            "s% use_other_before_struct_burn_mix (:833, 1225), switches "
            "inlist_hydro_on/inlist_hydro_off via star_read_controls, and "
            "calls star_set_u_flag/star_relax_to_star_cut (:1045-1165), which "
            "flexmesa-star StarExtras does not expose (#152, #154, #162, #231)"
        )

    def extras_finish_step(self, state: dict[str, Any]) -> str:
        """Update pulse/relaxation step counters and guard custom `t_xtra1`
        termination string (`:1344-1381`).
        """
        if self.lx_have_reached_gamma_limit:
            self.x_time_since_first_gamma_zero += float(state["dt"])
        self.ix_steps_since_relax += 1
        self.ix_steps_since_hydro_on += 1
        if (
            self.ix_num_relaxations == 1
            and self.stop_100d_after_pulse
            and float(state["star_age"]) - self.x_star_age_at_relax
            > 100.0 / constants.DAYYER
        ):
            raise NotImplementedError(
                "ppisn: extras_finish_step sets custom termination_code_str "
                "'Successful test: evolved 100 days past first relax' (:1374-"
                "1375), which StarExtras cannot emit (#231)"
            )
        return "keep_going"
