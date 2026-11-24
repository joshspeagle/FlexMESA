# SPDX-License-Identifier: GPL-3.0-or-later
"""Upstream mixing, envelope-builder, and TOV-gravity `StarExtras` ports
(`#231`).

Covers:
- star/test_suite/extended_convective_penetration/src/run_star_extras.f90:37-452
- star/test_suite/make_env/src/run_star_extras.f90:36-605
- star/test_suite/tzo/src/run_star_extras.f90:40-260
"""

from __future__ import annotations

import math
import pathlib
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
    (`:437-450` of `extended_convective_penetration`).
    """
    p_val = _rust.crmath_exp(ln_peos)
    rho_val = _rust.crmath_exp(lnd)
    return p_val / (rho_val * cgrav * (m_center + xmstar * q) / (r * r))


def calc_peclet_number(state: dict[str, Any], k: int) -> float:
    """Evaluate cell Peclet number (dimensionless) at 0-based index `k`
    (`:420-435` of `extended_convective_penetration`).
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
    (`star/test_suite/make_env/src/run_star_extras.f90:547-568`).
    """
    sn = 1.0
    r_to_n = _rust.pow(r, float(n))
    f_val = sn * (1.0 - r) - a1 * (1.0 - r_to_n)
    df_dr = -sn + a1 * float(n) * r_to_n / r
    return f_val, df_dr


def calc_dq_factor(n: int, dq1: float) -> float:
    """Solve for geometric cell-mass ratio `dq_factor`
    (`star/test_suite/make_env/src/run_star_extras.f90:570-603`).
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
    """Michielsen et al. (2019) convective penetration and Peclet `gradT` blend
    (`:37-452` of `extended_convective_penetration` cited in module docstring).
    """

    def __init__(self) -> None:
        self._pre_ms: bool = False
        self.checks_okay: bool | None = None

    def extras_controls(self, state: dict[str, Any]) -> None:
        """Record whether pre-MS skips penetration hooks (`:37-60`)."""
        self._pre_ms = bool(state["create_pre_main_sequence_model"])

    def other_adjust_mlt_gradt_fraction(
        self, state: dict[str, Any]
    ) -> list[float] | None:
        """Compute Peclet `adjust_mlt_gradT_fraction` outside pre-MS (`:57`)."""
        if self._pre_ms:
            return None
        return self.other_adjust_mlt_gradt_fraction_peclet(state)

    def other_overshooting_scheme(
        self, state: dict[str, Any], i_bdy: int, j_param: int
    ) -> tuple[int, int, list[float], list[float]] | None:
        """Evaluate extended convective penetration profile (`:58`)."""
        if self._pre_ms:
            return None
        return self.eval_penetration_profile(state, i_bdy, j_param)

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
        nz = int(state["nz"])
        hp_cb = float(state["conv_bdy_hp"][i_bdy])
        eval_ob = state.get("eval_over_bdy_params")
        if callable(eval_ob):
            k_ob_val, r_ob_val, d_ob_val, vc_ob_val = eval_ob(i_bdy, f0)
            k_ob = int(k_ob_val)
            r_ob = float(r_ob_val)
            d_ob = float(d_ob_val)
            vc_ob = float(vc_ob_val)
        else:
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


# `star/test_suite/make_env/src/run_star_extras.f90`: SHA-256 of each
# normalized routine (`tests.case_extras.normalize_fortran`). `create_env`
# includes its contained routines (`get_initial_guess_for_atm`, `get_atm`,
# `do1_cell`, `dq_f`, `calc_dq_factor`). `extras_after_evolve` (unused local
# `dt`) and `extras_check_model` (`keep_going`) were read and hold no case
# logic; the other routines are the stock template.
_MAKE_ENV_ROUTINE_DIGESTS: dict[str, str] = {
    "extras_controls": (
        "d6977ee7d85b1af4705bc2c397a8e7fa1311a2adc817848535aeefba89ed4449"
    ),
    "extras_startup": (
        "945bc8b7e365406c7f0fd5b9586324b868fdc676346914f7394876aab22f30f6"
    ),
    "extras_after_evolve": (
        "ae173bfc9afe1a5837fdd7c57eac399cf6e5930570d72eb5d96ab23ab0f5b111"
    ),
    "extras_check_model": (
        "1d7939ba515e2da3405b62475ced33624c3f4ca313e009707819879ff2b1f074"
    ),
    "how_many_extra_history_columns": (
        "90d8c21456f037612416160a5d81adaf65b71f9d8221fbc5481319c77573dbb3"
    ),
    "data_for_extra_history_columns": (
        "01288ef1cc03754d84fe38a8e7bb503ddceb5305aef9bab2f9adbd74d41db7ec"
    ),
    "how_many_extra_profile_columns": (
        "294e7adcf638b1d7511d2cda6b4c43a441fb2c2f9229210c52c2f19661ac7248"
    ),
    "data_for_extra_profile_columns": (
        "978902d5eebc99f1f38df1a5f1d46c13a57f66533c2b97350580801030de27d7"
    ),
    "extras_finish_step": (
        "68db5ddf03a81fa38d8d47f145337a68c8b64bc2f369ede17c3ff4c7bc1c940b"
    ),
    "create_env": (
        "ae802ded236964396179a68a295d72d3559841eede28692c94f0f213e2cfb9f7"
    ),
}


def match_make_env_extras(norm_star: str) -> MakeEnvStarExtras | None:
    """Return a `MakeEnvStarExtras` when every routine of the normalized
    `run_star_extras.f90` has the digest recorded for `make_env`.

    Args:
        norm_star: Normalized `run_star_extras.f90` text.

    Returns:
        The handler, or `None` for any other source.
    """
    from tests.case_extras import astero_extras, source_sha256

    routines = {
        m.group(2): source_sha256(m.group(0))
        for m in astero_extras._ROUTINE_RE.finditer(norm_star)
    }
    if routines != _MAKE_ENV_ROUTINE_DIGESTS:
        return None
    return MakeEnvStarExtras()


class MakeEnvStarExtras:
    """Static envelope constructor `create_env` from
    `star/test_suite/make_env/src/run_star_extras.f90:36-605`.

    `extras_controls` builds the envelope cell by cell through the
    `star_lib` subset (`flexmesa._rust.StarLibModel`) and writes `start.mod`,
    which the run then loads (`load_model_filename = 'start.mod'`).
    """

    wants_star_model = True

    def extras_controls(self, state: dict[str, Any]) -> dict[str, Any] | None:
        """Return when `x_logical_ctrl(1)` is false; otherwise call
        `create_env` (`:36-57`).
        """
        if not bool(state["x_logical_ctrl"][0]):
            return None
        return create_env(state)


_LOGRHO_TOL = 1.0e-11
_LOGPGAS_TOL = 1.0e-11
_ONE_THIRD = 1.0 / 3.0
_TWO_THIRDS = 2.0 / 3.0
_FOUR_THIRDS = 4.0 / 3.0


def _pow4(x: float) -> float:
    """`pow4` (`math/private/math_pown.f90:94-101`)."""
    return x * x * x * x


def create_env(state: dict[str, Any]) -> dict[str, Any]:
    """Build the envelope model and write `start.mod`
    (`star/test_suite/make_env/src/run_star_extras.f90:158-337`).

    Cells are 0-based here (upstream's `k - 1`). Returns the `s% tau_factor`
    that `create_env` leaves on the star (`:210`).

    Raises:
        ValueError: When a solve fails, where upstream stops the run.
    """
    x_ctrl = state["x_ctrl"]
    x_logical_ctrl = state["x_logical_ctrl"]
    nz = int(state["x_integer_ctrl"][0])
    max_iters = 100
    net_name = str(state["x_character_ctrl"][0]).strip()
    case_dir = str(state["case_dir"])

    s = _rust.StarLibModel.new_for_build(
        state["star_model"], net_name, nz, search_dir=case_dir
    )
    msun = constants.MSUN
    lsun = constants.LSUN
    pi = constants.PI
    crad = constants.CRAD
    ln10 = constants.LN10
    standard_cgrav = constants.STANDARD_CGRAV

    mstar = float(x_ctrl[0]) * msun
    s.set_scalar("mstar", mstar)
    xmstar = float(x_ctrl[1]) * mstar
    s.set_scalar("xmstar", xmstar)
    s.set_scalar("M_center", mstar - xmstar)

    s.set("m", 0, mstar)
    s.set("m_grav", 0, mstar)
    s.set("cgrav", 0, standard_cgrav)
    tau_factor = float(x_ctrl[5])
    s.set_scalar("tau_factor", tau_factor)

    teff_target = 0.0
    if bool(x_logical_ctrl[1]):  # R and L in cgs units (:219-221)
        s.set("r", 0, float(x_ctrl[2]))
        for k in range(nz):
            s.set("L", k, float(x_ctrl[3]))
    else:  # x_ctrl(3) = Teff; x_ctrl(4) = L/Lsun (:222-238)
        l_val = float(x_ctrl[3]) * lsun
        for k in range(nz):
            s.set("L", k, l_val)
        tau_base = _TWO_THIRDS
        tau_surf = tau_factor * tau_base
        teff_target = float(x_ctrl[2])
        teff4 = _pow4(teff_target)
        t4 = teff4 * 0.75 * (tau_surf + tau_base)
        l1 = s.get("L", 0)
        s.set("r", 0, math.sqrt(l1 / (4.0 * pi * constants.BOLTZ_SIGMA * t4)))

    s.set_scalar("L_center", s.get("L", nz - 1))

    ln_dq1 = float(x_ctrl[4]) * ln10
    dq1 = _rust.crmath_exp(ln_dq1)
    dq_factor = calc_dq_factor(nz, dq1)

    s.set("q", 0, 1.0)
    s.set("dq", 0, dq1)
    for k in range(1, nz):
        s.set("q", k, s.get("q", k - 1) - s.get("dq", k - 1))
        s.set("dq", k, s.get("dq", k - 1) * dq_factor)

    s.normalize_dqs()
    s.set_qs()
    s.set_m_and_dm()
    s.set_dm_bar()
    s.change_to_xa_for_accretion(0, nz - 1)

    for k in range(nz):
        s.set("m_grav", k, s.get("m", k))
        s.basic_composition_info(k)

    s.set_scalar("tau_base", _TWO_THIRDS)
    tau_surf = tau_factor * _TWO_THIRDS
    save_atm = s.get_atm_options()
    save_pextra_factor = s.get_scalar("Pextra_factor")
    s.set_atm_options("T_tau", "Eddington", "iterated")
    s.set_scalar("Pextra_factor", 2.0)
    s.set_scalar("Teff", _rust.atm_teff(s.get("L", 0), s.get("r", 0)))

    # `get_initial_guess_for_atm` (:345-374).
    s.set("opacity", 0, 1.0e-2)
    ln_t_surf, ln_p_surf = s.get_atm_pt(
        tau_surf,
        s.get("L", 0),
        s.get("r", 0),
        s.get("m", 0),
        s.get("cgrav", 0),
        s.get_scalar("Teff"),
    )
    t_surf = _rust.crmath_exp(ln_t_surf)
    p_surf = _rust.crmath_exp(ln_p_surf)
    prad = _ONE_THIRD * crad * _pow4(t_surf)
    pgas = p_surf - prad
    log_t = _rust.log10(t_surf)
    log_rho = s.solve_eos_given_pgas_t_auto(
        0, log_t, _rust.log10(pgas), _LOGRHO_TOL, _LOGPGAS_TOL
    )

    teff_now = s.get_scalar("Teff")
    if not bool(x_logical_ctrl[1]) and abs(teff_target - teff_now) > (
        1.0e-5 * teff_target
    ):  # :291-298
        ratio = teff_now / teff_target
        s.set("r", 0, s.get("r", 0) * (ratio * ratio))

    cell = _EnvCell(s, nz, max_iters)

    # Cell 1, then the final atmosphere (:300-317).
    cell.t_m1, cell.p_m1, cell.log_rho_m1 = t_surf, p_surf, log_rho
    cell.log_rho = log_rho
    cell.do1_cell(0)
    s.set_atm_options(*save_atm)
    s.set_scalar("Pextra_factor", save_pextra_factor)

    # `get_atm` (:403-438).
    opacity1 = s.get("opacity", 0)
    if opacity1 <= 0.0 or not math.isfinite(opacity1):
        raise ValueError(f"make_env get_atm: s% opacity(1) = {opacity1}")
    ln_t_surf, ln_p_surf = s.get_surf_pt()
    t_surf = _rust.crmath_exp(ln_t_surf)
    p_surf = _rust.crmath_exp(ln_p_surf)
    prad = _ONE_THIRD * crad * _pow4(t_surf)
    pgas = p_surf - prad
    log_t = _rust.log10(t_surf)
    log_rho = s.solve_eos_given_pgas_t(
        0,
        log_t,
        _rust.log10(pgas),
        cell.log_rho,
        _LOGRHO_TOL,
        _LOGPGAS_TOL,
    )

    # Update the surface values for the final atmosphere (:324-327).
    cell.t_m1, cell.p_m1, cell.log_rho_m1 = t_surf, p_surf, log_rho
    for k in range(nz):  # :329-337
        cell.do1_cell(k)
        cell.t_m1, cell.p_m1, cell.log_rho_m1 = (
            cell.t_00,
            cell.p_00,
            cell.log_rho,
        )

    s.pack_xh()
    s.model_number = 0
    s.set_scalar("star_age", 0.0)
    s.set_scalar("crystal_core_boundary_mass", -1.0)
    s.write_model(str(pathlib.Path(case_dir) / "start.mod"))
    return {"tau_factor": tau_factor}


class _EnvCell:
    """State of `do1_cell` (:441-545), which reads `r(k)`, `T_m1`, `P_m1`
    and `logRho_m1` and sets `T_00`, `P_00` and `logRho`."""

    def __init__(self, s: Any, nz: int, max_iters: int) -> None:
        self.s = s
        self.nz = nz
        self.max_iters = max_iters
        self.t_m1 = 0.0
        self.p_m1 = 0.0
        self.log_rho_m1 = 0.0
        self.t_00 = 0.0
        self.p_00 = 0.0
        self.log_rho = 0.0

    def do1_cell(self, k: int) -> None:
        """`do1_cell` for 0-based cell `k`."""
        s = self.s
        pi = constants.PI
        crad = constants.CRAD
        ln10 = constants.LN10
        r_00 = s.get("r", k)
        s.set("cgrav", k, constants.STANDARD_CGRAV)
        grav = -s.get("cgrav", k) * s.get("m", k) / (r_00 * r_00)

        if k > 0:
            dm_face = 0.5 * (s.get("dm", k) + s.get("dm", k - 1))
        else:
            dm_face = 0.5 * s.get("dm", k)
        area = 4.0 * pi * (r_00 * r_00)

        p_00 = self.p_m1 - grav * dm_face / area
        p_face = 0.5 * (self.p_m1 + p_00)
        dlnp_face = (self.p_m1 - p_00) / p_face
        t_00 = self.t_m1
        log_rho = self.log_rho
        mixing_length_alpha = s.get_scalar("mixing_length_alpha")
        for it in range(1, self.max_iters + 1):
            prad = _ONE_THIRD * crad * _pow4(t_00)
            pgas = p_00 - prad
            log_t = _rust.log10(t_00)
            log_rho = s.solve_eos_given_pgas_t(
                k,
                log_t,
                _rust.log10(pgas),
                self.log_rho_m1,
                _LOGRHO_TOL,
                _LOGPGAS_TOL,
            )
            s.set("lnd", k, log_rho * ln10)
            s.set("lnT", k, log_t * ln10)
            s.set("rho", k, _rust.exp10(log_rho))
            s.set("T", k, t_00)
            s.do_eos_for_cell(k)
            s.do_kap_for_cell(k)

            s.set("lnT_start", k, s.get("lnT", k))
            s.set("csound_start", k, s.get("csound", k))
            s.set_scalar("mlt_gradT_fraction", -1.0)
            s.set("adjust_mlt_gradT_fraction", k, -1.0)
            s.set("gradL_composition_term", k, 0.0)
            s.set("alpha_mlt", k, mixing_length_alpha)
            s.set("gradr_factor", k, 1.0)
            s.set_mlt_vars(k, k)

            grad_t = s.get("gradT", k)
            d_grad_t_dt = s.d_gradt_dlnt00(k) / t_00
            t_face = 0.5 * (self.t_m1 + t_00)
            d_t_face_dt = 0.5
            t_expected = self.t_m1 - t_face * grad_t * dlnp_face
            d_t_expected_dt = -dlnp_face * (
                d_t_face_dt * grad_t + t_face * d_grad_t_dt
            )
            resid = t_expected - t_00
            d_resid_dt = d_t_expected_dt - 1.0
            d_t = -resid / d_resid_dt
            if abs(d_t / t_00) < 1.0e-4:
                break
            if it == self.max_iters:
                raise ValueError(f"make_env do1_cell: failed to find T at {k}")
            t_00 = t_00 + d_t

        rho_00 = _rust.exp10(log_rho)
        r3 = r_00 * r_00 * r_00
        vol = _FOUR_THIRDS * pi * r3 - s.get("dm", k) / rho_00
        r_p1 = _rust.pow(0.75 * vol / pi, _ONE_THIRD)
        if k == self.nz - 1:
            s.set_scalar("R_center", r_p1)
        else:
            s.set("r", k + 1, r_p1)

        s.set("lnR", k, _rust.crmath_log(r_00))
        s.set("lnd", k, _rust.crmath_log(rho_00))
        s.set("lnT", k, _rust.crmath_log(t_00))
        self.t_00 = t_00
        self.p_00 = p_00
        self.log_rho = log_rho


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
