# SPDX-License-Identifier: LGPL-3.0-or-later
"""Upstream wind, physics-hook, and magnetic-braking `StarExtras` ports
(`#231`).

Covers:
- `star/test_suite/other_physics_hooks/src/run_star_extras.f90` and
  `src/other_winds/other_winds.inc`
- `star/test_suite/magnetic_braking/src/run_star_extras.f90`
"""

from __future__ import annotations

import math
import pathlib
import re
from typing import Any

from flexmesa import _rust, constants
from tests.case_extras import _helpers

_NML_KV_RE = re.compile(r"^([A-Za-z_]\w*)\s*=\s*(.+)$")


def _parse_fortran_float(raw: str) -> float:
    """Parse a Fortran real literal (`1d0`, `1.35d-12`, `-0.5`) to `float64`."""
    cleaned = raw.strip().rstrip(",").strip()
    return float(cleaned.replace("D", "E").replace("d", "e"))


def _cf(u: float, alpha: float, beta: float) -> float:
    """Finite cone-angle correction factor (`other_winds.inc:276-293`)."""
    xl = (beta + 1.0) * u - 1.0
    xl = xl * u / beta
    if abs(xl) <= 1.0e-3:
        return 1.0
    val = 1.0 - _rust.pow(1.0 - xl, 1.0 + alpha)
    val = val / xl
    return val / (1.0 + alpha)


def _phicr(vsquar: float) -> float:
    """Pauldrach `phi` function at `ucrit` (`other_winds.inc:329-341`)."""
    vsq = math.sqrt(vsquar)
    xl = _rust.log10(vsq)
    xl = 0.36 + xl
    xl = 0.3 * xl
    xl = _rust.pow(vsq, xl)
    return 3.0 * xl


def _betcr(alpha: float, delta: float) -> float:
    """Critical-point `beta` selector (`other_winds.inc:358-387`)."""
    if alpha <= 0.7:
        if delta < 0.03:
            return 2.0
        if delta < 0.055:
            return 1.0
        if delta < 0.085:
            return 0.7
        if delta < 0.095:
            return 0.5
        return 0.25
    if delta >= 0.03:
        if delta >= 0.095:
            return 0.7
        if delta <= 0.055:
            return 2.0
        return 1.0
    return 2.0


def _qq(beta: float, delta: float) -> float:
    """Piecewise `q(beta, delta)` helper (`other_winds.inc:390-410`)."""
    b = min(2.5, max(0.2, beta))
    if b >= 1.0:
        a = 15.0 * (b - 1.0) + 7.5
    elif b >= 0.7:
        a = 11.66667 * (b - 0.7) + 4.0
    elif b >= 0.5:
        a = 7.5 * (b - 0.5) + 2.5
    else:
        a = 5.28 * (b - 0.25) + 1.18
    return _rust.pow(a, delta - 1.0)


def _uc(alpha: float, _beta: float, delta: float, vsquar: float) -> float:
    """Reciprocal critical point `uc` (`other_winds.inc:307-326`)."""
    sigma = alpha * alpha * (1.0 - alpha) / vsquar
    xphi = _phicr(vsquar)
    sigma = sigma / (xphi * xphi)
    bbb = _betcr(alpha, delta)
    q = _qq(bbb, delta)
    xc = -2.0 * q / (sigma * 3.0)
    hilf = (1.46 / sigma) * (alpha + 1.0) * _rust.pow(alpha, 0.6)
    xc = _rust.pow(hilf, 1.0 / 3.0) + xc
    if xc < 1.03:
        xc = 1.03
    return 1.0 / xc


def _vaucr(vsound: float, vesc: float, alpha: float) -> float:
    """Critical-point velocity (`other_winds.inc:344-355`)."""
    vsq = vsound * vsound / (vesc * vesc)
    phi = _phicr(vsq)
    hilf = math.sqrt(phi / (1.0 - alpha))
    h = 1.0 - _rust.pow(1.0 - alpha, 2.0 / alpha)
    return vsound * (hilf / h)


def _xintap(alpha: float, beta: float, delta: float, ucrit: float) -> float:
    """Approximate wind velocity integral (`other_winds.inc:413-460`)."""
    q = _qq(beta, delta)
    hilfex = 1.0 / (1.0 - alpha)
    alaaf = _rust.pow(1.0 / (1.0 + alpha), hilfex)
    a0 = (1.0 + beta - alaaf) / beta
    a1 = a0 - alaaf
    b1 = alpha * hilfex / (2.0 * beta)
    b2 = b1 * (beta + 1.0)
    bet1 = 1.0 / (1.0 + beta)
    bet2 = bet1 * bet1
    bet3 = bet2 * bet1
    bet4 = bet3 * bet1
    bet5 = bet4 * bet1
    uc1 = ucrit
    uc2 = uc1 * uc1
    uc3 = uc2 * uc1
    uc4 = uc3 * uc1
    x = a0 * (uc1 - bet1 + hilfex * q * (uc3 - bet3) / 3.0)
    x = x - a1 * ((uc2 - bet2) / 2.0 + hilfex * q * (uc4 - bet4) / 4.0)
    x = x + bet1
    x = x + b1 * (bet2 / 2.0 + hilfex * q * bet4 / 4.0)
    x = x - b2 * (bet3 / 3.0 + hilfex * q * bet5 / 5.0)
    x = x + hilfex * q * bet3 / 3.0
    g = (1.0 / (a0 - a1 * uc1)) * _rust.pow(1.0 / (q * uc2 + 1.0), hilfex)
    z = (2.0 / alpha) * (1.0 - 1.0 / g)
    return g * (1.0 + math.sqrt(z)) * x


def _sigmae(yps: float, xihe: float) -> float:
    """Thomson scattering opacity per unit mass (`other_winds.inc:486-493`)."""
    return ((1.0 + xihe * yps) / (1.0 + 4.0 * yps)) * 0.3978


def _gammma(stlum: float, stmass: float, sig: float) -> float:
    """Eddington factor `Gamma` (`other_winds.inc:496-500`)."""
    return 7.655e-5 * sig * stlum / stmass


def _xmdcak(
    xk: float,
    alpha: float,
    stlum: float,
    stmass: float,
    sig: float,
    ga: float,
    teff: float,
) -> float:
    """Evaluate CAK mass-loss factor (`other_winds.inc:463-483`)."""
    hilfex = 1.0 / alpha
    hilf = _rust.pow(1.2762e23 * xk * stlum * 1.0e-10, hilfex)
    hiover = _rust.pow(1.0e10, hilfex)
    hilf = hilf * alpha / (1.2848e4 * math.sqrt(teff))
    hi = _rust.pow(
        sig * (1.0 - alpha) / (1.6671e27 * stmass * (1.0 - ga)),
        hilfex * (1.0 - alpha),
    )
    return hilf * hi * hiover


def _xmdfc(
    alpha: float,
    beta: float,
    delta: float,
    xint: float,
    ucrit: float,
    rst: float,
    vesc: float,
    yps: float,
    xmdcak_val: float,
) -> float:
    """Evaluate finite-cone-angle mass loss (`other_winds.inc:510-535`)."""
    d_val = (
        (3.92859e1 * (1.0 + 2.0 * yps) / (1.0 + 4.0 * yps))
        / (rst * rst)
        / vesc
        * 1.0e-11
        * math.sqrt((1.0 - alpha) / alpha)
    )
    hiexa = 1.0 / (alpha - delta)
    a_val = _rust.pow(d_val / math.sqrt(xint), hiexa * delta)
    b_val = _rust.pow(
        _cf(ucrit, alpha, beta) * (1.0 + _qq(beta, delta) * ucrit * ucrit),
        hiexa,
    )
    c_val = _rust.pow(xmdcak_val, hiexa * alpha) * 1.58661e-26
    return a_val * b_val * c_val


def eval_kuma_wind(
    xlogl: float, xlteff: float, stmass: float, xsurf: float
) -> tuple[float, float]:
    """Evaluate Kudritzki radiation-driven wind (`other_winds.inc:220-556`).

    Args:
        xlogl: $\\log_{10}(L / L_\\odot)$ (dimensionless).
        xlteff: $\\log_{10}(T_{\\mathrm{eff}} / \\mathrm{K})$ (dimensionless).
        stmass: Stellar mass $M / M_\\odot$ (solar masses).
        xsurf: Surface hydrogen mass fraction $X$ (dimensionless).

    Returns:
        Tuple `(xmdfic, vinfkm)` giving mass-loss rate in
        $M_\\odot\\,\\mathrm{yr^{-1}}$ and terminal velocity in
        $\\mathrm{km\\,s^{-1}}$.
    """
    teff = _rust.exp10(xlteff)
    stlum = _rust.exp10(xlogl)
    ttt = teff / 5780.0
    ttt = ttt * ttt * ttt * ttt
    rst = math.sqrt(stlum / ttt)
    yps = (0.98 - xsurf) / (4.0 * xsurf)
    sig = _sigmae(yps, 2.0)
    ga = _gammma(stlum, stmass, sig)
    vesckm = 617.0 * math.sqrt(stmass * (1.0 - ga) / rst)
    vesc = vesckm * 1.0e5
    vsound = 9.085e3 * math.sqrt(((2.0 + 3.0 * yps) / (1.0 + 4.0 * yps)) * teff)
    vsquar = (vsound / vesc) * (vsound / vesc)

    xk, alpha, beta, delta = 0.085, 0.657, 1.0, 0.095
    ucrit = _uc(alpha, beta, delta, vsquar)
    xx = _xintap(alpha, beta, delta, ucrit)
    vc = _vaucr(vsound, vesc, alpha)
    xx = xx + (1.0 - alpha) * vc * vc / (alpha * vesc * vesc)
    vinfkm = vesckm * math.sqrt(xx) * math.sqrt(alpha / (1.0 - alpha))
    xmdca = _xmdcak(xk, alpha, stlum, stmass, sig, ga, teff)
    xmdfic = _xmdfc(alpha, beta, delta, xx, ucrit, rst, vesc, yps, xmdca)
    return xmdfic, vinfkm


def eval_grafener_wind(
    l_cgs: float,
    m_cgs: float,
    t_surf: float,
    x_surf: float,
    z_surf: float,
    scaling_factor: float = 1.0,
) -> float:
    """Evaluate Grafener & Hamann (2008) WR wind (`other_winds.inc:70-95`).

    Args:
        l_cgs: Surface luminosity $L$ in $\\mathrm{erg\\,s^{-1}}$.
        m_cgs: Stellar mass $M$ in $\\mathrm{g}$.
        t_surf: Surface temperature $T_{\\mathrm{surf}}$ in $\\mathrm{K}$.
        x_surf: Surface hydrogen mass fraction $X$ (dimensionless).
        z_surf: Surface metal mass fraction $Z$ (dimensionless).
        scaling_factor: Multiplicative `Grafener_scaling_factor`.

    Returns:
        Wind mass-loss rate $w$ in $M_\\odot\\,\\mathrm{yr^{-1}}$.
    """
    gamma_edd = (
        _rust.exp10(-4.813)
        * (1.0 + x_surf)
        * (l_cgs / constants.LSUN)
        * (constants.MSUN / m_cgs)
    )
    lg_z = _rust.log10(z_surf / 0.019)
    beta = 1.727 + 0.250 * lg_z
    gammazero = 0.326 - 0.301 * lg_z - 0.045 * lg_z * lg_z
    if gammazero > gamma_edd:
        return 0.0
    log_mdot = (
        10.046
        - 3.5 * _rust.log10(t_surf)
        + beta * _rust.log10(gamma_edd - gammazero)
        + 0.42 * _rust.log10(l_cgs / constants.LSUN)
        - 0.45 * x_surf
    )
    return _rust.exp10(log_mdot) * scaling_factor


def wr_hamann(xsl: float, xhs: float, znow: float) -> float:
    """Evaluate Hamann et al. (1982, 1995) WR wind (`other_winds.inc:192-217`).

    Args:
        xsl: Surface luminosity $L$ in $\\mathrm{erg\\,s^{-1}}$.
        xhs: Surface hydrogen mass fraction $X_{\\mathrm{s}}$ (dimensionless).
        znow: Effective metallicity $Z$ (dimensionless).

    Returns:
        Unscaled WR wind rate in $M_\\odot\\,\\mathrm{yr^{-1}}$.
    """
    xlogl = _rust.log10(xsl / constants.LSUN)
    logz = _rust.log10(max(1.0e-20, znow) / 0.02)
    if xlogl > 4.5:
        xlogmd = 1.5 * xlogl - 2.85 * xhs - 11.95 + 0.86 * logz
    else:
        xlogmd = 6.8 * xlogl - 2.85 * xhs - 35.8 + 0.86 * logz
    return _rust.exp10(xlogmd)


def _stern51_h_rich(
    l_cgs: float,
    m_cgs: float,
    r_cgs: float,
    teff: float,
    x_surf: float,
    xhs: float,
    xhc: float,
    zbase: float,
    z_fe: float,
    z_cno: float,
) -> float:
    """Compute `Stern51` wind for `X_surf > 0.3` (`other_winds.inc:146-175`)."""
    znow = max(
        0.0,
        min(1.0, 0.02 * (0.5 * (z_fe / 1.32e-3) + 0.5 * (z_cno / 0.01868))),
    )
    if xhc < 1.0e-6:
        logmd = (
            1.24 * _rust.log10(l_cgs / constants.LSUN)
            + 0.16 * _rust.log10(m_cgs / constants.MSUN)
            + 0.81 * _rust.log10(r_cgs / constants.RSUN)
            - 14.016
        )
        dms = _rust.exp10(logmd)
    else:
        dms, _ = eval_kuma_wind(
            _rust.log10(l_cgs / constants.LSUN),
            _rust.log10(teff),
            m_cgs / constants.MSUN,
            xhs,
        )
    dms = dms * _rust.pow(znow / 0.02, 0.69)
    if x_surf <= 0.45:
        znow = max(zbase, 1.0e-6)
        dms1 = dms
        dms2 = wr_hamann(l_cgs, xhs, znow) / 10.0
        xxx = (xhs - 0.375) * 12.0
        fff = 0.25 * (2.0 - 3.0 * xxx + xxx * xxx * xxx)
        dms = max(dms1, (1.0 - fff) * dms1 + fff * dms2)
    return dms


def eval_stern51_wind(
    state: dict[str, Any],
    l_cgs: float,
    m_cgs: float,
    r_cgs: float,
    teff: float,
    x_surf: float,
    z_surf: float,
) -> float:
    """Evaluate Stern51 composite wind (`other_winds.inc:114-189`)."""
    zbase = float(state["zbase"])
    if zbase < 0.0:
        raise ValueError("must set Zbase to use stern51 wind")
    z_fe = zbase * 0.07
    z_cno = z_surf - z_fe
    if bool(state["has_h1_iso"]):
        xhs = float(state["surface_h1"])
        xhc = float(state["center_h1"])
    else:
        xhs = x_surf
        xhc = 0.0
    if x_surf > 0.3:
        return _stern51_h_rich(
            l_cgs, m_cgs, r_cgs, teff, x_surf, xhs, xhc, zbase, z_fe, z_cno
        )
    znow = max(zbase, 1.0e-6)
    dms = wr_hamann(l_cgs, xhs, znow) / 10.0
    if z_cno >= znow:
        dms = dms * (1.0 + 19.0 * (z_cno - znow) / (1.0 - znow))
    return dms


class OtherPhysicsHooksStarExtras:
    """Physics-hook test harness from
    `star/test_suite/other_physics_hooks/src/run_star_extras.f90:45-435` and
    `src/other_winds/other_winds.inc:2-556`.
    """

    def __init__(self) -> None:
        self.wind_scheme = ""
        self.kudritzki_scaling_factor = 0.0
        self.grafener_scaling_factor = 0.0
        self.stern51_scaling_factor = 0.0
        self._loaded_inlist = False

    def read_inlist_other_winds(self, case_dir: pathlib.Path | str) -> None:
        """Read `inlist_other_winds` (`other_winds.inc:2-33`)."""
        self.wind_scheme = ""
        self.kudritzki_scaling_factor = 0.0
        self.grafener_scaling_factor = 0.0
        self.stern51_scaling_factor = 0.0
        nml_path = pathlib.Path(case_dir) / "inlist_other_winds"
        text = nml_path.read_text(encoding="utf-8")
        in_block = False
        found_block = False
        for raw_line in text.splitlines():
            line = raw_line.split("!", 1)[0].strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("&other_winds"):
                in_block = True
                found_block = True
                continue
            if in_block and low.startswith("/"):
                break
            if not in_block:
                continue
            m = _NML_KV_RE.match(line)
            if m is None:
                raise ValueError(f"Malformed &other_winds line: {line!r}")
            key = m.group(1).lower()
            val_str = m.group(2).strip().rstrip(",").strip()
            if key == "wind_scheme":
                self.wind_scheme = val_str.strip("'\"")
            elif key == "kudritzki_scaling_factor":
                self.kudritzki_scaling_factor = _parse_fortran_float(val_str)
            elif key == "grafener_scaling_factor":
                self.grafener_scaling_factor = _parse_fortran_float(val_str)
            elif key == "stern51_scaling_factor":
                self.stern51_scaling_factor = _parse_fortran_float(val_str)
            else:
                raise ValueError(f"Unknown &other_winds control: {key!r}")
        if not found_block:
            raise ValueError(f"Missing &other_winds namelist in {nml_path}")
        self._loaded_inlist = True

    def extras_controls(self, _state: dict[str, Any]) -> None:
        """Fail loudly naming unexposed `other_*` callbacks (`:45-92`)."""
        raise NotImplementedError(
            "other_physics_hooks: extras_controls registers "
            "other_overshooting_scheme (:53, #155), other_timestep_limit (:55, "
            "#152), other_mesh_delta_coeff_factor (:59, #153), other_kap_get / "
            "other_eos_results / other_screening / other_rate_get (:62-75, "
            "#161), other_close_gaps (:77-78, #155), and other_pressure "
            "(:80-81, #152), which flexmesa-star StarExtras does not expose"
        )

    def other_wind(
        self,
        state: dict[str, Any],
        l_surf: float,
        m_surf: float,
        r_surf: float,
        t_surf: float,
    ) -> float:
        """Evaluate `other_wind` in $M_\\odot\\,\\mathrm{yr^{-1}}$
        (`other_winds.inc:36-67`).
        """
        if not self._loaded_inlist:
            self.read_inlist_other_winds(str(state["case_dir"]))
        _ = int(state["model_number"])
        x_surf = float(state["surface_h1"])
        y_surf = float(state["surface_he4"])
        zbase = float(state["zbase"])
        # Matches star/private/winds.f90:336-341 for Z passed to s% other_wind.
        z_surf = zbase if zbase >= 0.0 else 1.0 - (x_surf + y_surf)
        if self.wind_scheme == "Stern51":
            return eval_stern51_wind(
                state, l_surf, m_surf, r_surf, t_surf, x_surf, z_surf
            )
        if self.wind_scheme == "Grafener":
            self.grafener_scaling_factor = 1.0
            return eval_grafener_wind(
                l_surf,
                m_surf,
                t_surf,
                x_surf,
                z_surf,
                self.grafener_scaling_factor,
            )
        if self.wind_scheme == "Kudritzki":
            self.kudritzki_scaling_factor = 1.0
            xmdfic, _ = eval_kuma_wind(
                _rust.log10(l_surf / constants.LSUN),
                _rust.log10(t_surf),
                m_surf / constants.MSUN,
                x_surf,
            )
            return xmdfic * self.kudritzki_scaling_factor
        raise ValueError(
            f"other_wind: invalid wind_scheme: <{self.wind_scheme}>"
        )


def _distribute_dipole_torque(
    nz: int, j_dot: float, dt: float, j_rot: Any, dm_bar: Any
) -> list[float]:
    """Distribute `j_dot` across shells with `-j_rot/dt` cap (`:133-150`)."""
    extra_jdot = [0.0] * nz
    residual_jdot = 0.0
    j_capped = 0
    for k in range(nz):
        dm_k = float(dm_bar[k])
        cap_k = _helpers.fdiv(float(j_rot[k]), dt)
        ej_k = _helpers.fdiv(j_dot, float(nz) * dm_k)
        if abs(ej_k) > abs(cap_k):
            residual_jdot -= (abs(ej_k) - abs(cap_k)) * dm_k
            ej_k = -cap_k
            j_capped += 1
        extra_jdot[k] = ej_k
    for k in range(nz):
        dm_k = float(dm_bar[k])
        cap_k = _helpers.fdiv(float(j_rot[k]), dt)
        if abs(extra_jdot[k]) < abs(cap_k):
            extra_jdot[k] += _helpers.fdiv(
                residual_jdot, float(nz - j_capped) * dm_k
            )
    return extra_jdot


class MagneticBrakingStarExtras:
    """Ud-Doula et al. (2008) dipole magnetic-braking torque and spindown limit
    from `star/test_suite/magnetic_braking/src/run_star_extras.f90:38-305`.
    """

    def __init__(self) -> None:
        self._t_spindown = 1.0e100
        self._j_tot = 0.0
        self.spindown_ok: bool | None = None

    def extras_controls(self, state: dict[str, Any]) -> None:
        """Require `other_timestep_limit` when enabled (`:38-57`)."""
        if bool(state["use_other_timestep_limit"]):
            raise NotImplementedError(
                "magnetic_braking: extras_controls registers "
                "other_timestep_limit (:56), which flexmesa-star StarExtras "
                "does not expose (#152)"
            )

    def other_torque(self, state: dict[str, Any]) -> list[float]:
        """Compute `extra_jdot` `(nz,)` in $\\mathrm{cm^2\\,s^{-2}}$
        (`:60-171`).
        """
        nz = int(state["nz"])
        bfield = float(state["x_ctrl"][0])
        v_rot = float(state["v_rot_avg_surf"])
        j_rot = state["j_rot"]
        dm_bar = state["dm_bar"]
        j_tot = 0.0
        for k in range(nz):
            j_tot += float(j_rot[k]) * float(dm_bar[k])
        self._j_tot = j_tot

        extra_jdot = [0.0] * nz
        mstar_dot = float(state["mstar_dot"])
        if mstar_dot != 0.0 and j_tot > 1.0e50 and v_rot > 0.8e5:
            _ = bool(state["x_logical_ctrl"][0])
            r_phot_cm = float(state["photosphere_r"]) * constants.RSUN
            vinf = 1.92 * math.sqrt(
                2.0
                * constants.STANDARD_CGRAV
                * float(state["mstar"])
                / r_phot_cm
            )
            eta = ((r_phot_cm * bfield) * (r_phot_cm * bfield)) / (
                abs(mstar_dot) * vinf
            )
            j_dot = (
                (2.0 / 3.0)
                * mstar_dot
                * float(state["omega_avg_surf"])
                * (r_phot_cm * r_phot_cm)
                * eta
            )
            dt = float(state["dt"])
            self._t_spindown = abs(_helpers.fdiv(j_tot, j_dot))
            return _distribute_dipole_torque(nz, j_dot, dt, j_rot, dm_bar)
        return extra_jdot

    def other_timestep_limit(
        self, state: dict[str, Any], dt_limit_ratio: float
    ) -> tuple[str, float]:
        """Evaluate spindown `dt_limit_ratio` (`:270-292`)."""
        if (
            self._j_tot > 1.0e50
            and float(state["v_rot_avg_surf"]) > 0.8e5
            and self._t_spindown > 0.0
        ):
            dt_limit_ratio = (
                float(state["x_ctrl"][1])
                * float(state["dt"])
                / self._t_spindown
            )
        return "keep_going", dt_limit_ratio

    def extra_profile_columns(
        self, state: dict[str, Any]
    ) -> list[tuple[str, list[float]]]:
        """Return `zbar_div_abar` profile column (`:239-267`)."""
        return _helpers.zbar_div_abar_profile_column(state)

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """Record final surface rotation velocity check (`:186-201`)."""
        self.spindown_ok = float(state["v_rot_avg_surf"]) < 1.0e5
