# SPDX-License-Identifier: LGPL-3.0-or-later
"""Thermohaline (salt-finger) mixing (`set_thermohaline`, `thermohaline.f90`).

Computes compositional diffusivity $D_{\\mathrm{thrm}}$ when the thermal
gradient is Schwarzschild stable ($\\nabla_r < \\nabla_{\\mathrm{ad}}$)
but destabilized by an inverted molecular-weight gradient
($\\nabla_L - \\nabla_{\\mathrm{ad}} < 0$) across three prescriptions:
1. `ThermohalineOption.KIPPENHAHN`: Kippenhahn, Ruschenplatt, & Thomas (1980).
2. `ThermohalineOption.TRAXLER_GARAUD_STELLMACH_11`: Traxler, Garaud, &
   Stellmach (2011, ApJL, 728, L29, eq. 24).
3. `ThermohalineOption.BROWN_GARAUD_STELLMACH_13`: Brown, Garaud, & Stellmach
   (2013, ApJ, 768, 34, eqs. 19, 20, 33).
"""

from __future__ import annotations

import dataclasses
import math

from flexmesa import constants as _const
from flexmesa.turb import types as _types


@dataclasses.dataclass(frozen=True)
class ThermohalineParams:
    """Non-differentiable parameters for `set_thermohaline`.

    Attributes:
        option: Thermohaline prescription (`ThermohalineOption`).
        gradl_composition_term: Dimensionless composition gradient term
            $\\nabla_L - \\nabla_{\\mathrm{ad}}$ (`dlnMu/dlnP * chiY/chiT`).
        xh1: Hydrogen mass fraction $X({}^1\\mathrm{H})$ (dimensionless, in
            $[0, 1]$).
        thermohaline_coeff: Dimensionless multiplicative efficiency factor
            $\\alpha_{\\mathrm{th}}$.
        species: Driving isotope charge and mass number (`DrivingSpecies`).
    """

    option: _types.ThermohalineOption = _types.ThermohalineOption.KIPPENHAHN
    gradl_composition_term: float = -1.0e-4
    xh1: float = 0.7
    thermohaline_coeff: float = 1.0
    species: _types.DrivingSpecies = dataclasses.field(
        default_factory=_types.DrivingSpecies.he3
    )


@dataclasses.dataclass(frozen=True)
class ThermohalineInputs:
    """Differentiable thermodynamic and structure inputs for `set_thermohaline`.

    Attributes:
        lambda_: Mixing length $\\Lambda$ in $\\mathrm{cm}$.
        grada: Adiabatic temperature gradient $\\nabla_{\\mathrm{ad}}$.
        gradr: Radiative temperature gradient $\\nabla_r$.
        t: Temperature $T$ in $\\mathrm{K}$.
        opacity: Rosseland mean opacity $\\kappa$ in $\\mathrm{cm^2\\,g^{-1}}$.
        rho: Mass density $\\rho$ in $\\mathrm{g\\,cm^{-3}}$.
        cp: Specific heat at constant pressure $C_P$ in
            $\\mathrm{erg\\,g^{-1}\\,K^{-1}}$.
    """

    lambda_: _types.ScalarOrDual
    grada: _types.ScalarOrDual
    gradr: _types.ScalarOrDual
    t: _types.ScalarOrDual
    opacity: _types.ScalarOrDual
    rho: _types.ScalarOrDual
    cp: _types.ScalarOrDual


def get_d_thermohaline(
    option: _types.ThermohalineOption,
    grada: float,
    gradr: float,
    t: float,
    opacity: float,
    rho: float,
    cp: float,
    gradl_composition_term: float,
    species: _types.DrivingSpecies,
    xh1: float,
    thermohaline_coeff: float,
) -> float:
    """Compute scalar thermohaline diffusivity $D_{\\mathrm{thrm}}$ in CGS.

    Matches `get_D_thermohaline` in `turb/private/thermohaline.f90`.

    Args:
        option: Thermohaline prescription (`ThermohalineOption`).
        grada: Adiabatic gradient $\\nabla_{\\mathrm{ad}}$ (dimensionless).
        gradr: Radiative gradient $\\nabla_r$ (dimensionless).
        t: Temperature $T$ in $\\mathrm{K}$.
        opacity: Rosseland opacity $\\kappa$ in $\\mathrm{cm^2\\,g^{-1}}$.
        rho: Mass density $\\rho$ in $\\mathrm{g\\,cm^{-3}}$.
        cp: Specific heat $C_P$ in $\\mathrm{erg\\,g^{-1}\\,K^{-1}}$.
        gradl_composition_term: $\\nabla_L - \\nabla_{\\mathrm{ad}}$.
        species: Driving isotope $(Z, A)$ (`DrivingSpecies`).
        xh1: Hydrogen mass fraction $X({}^1\\mathrm{H})$.
        thermohaline_coeff: Multiplicative efficiency factor.

    Returns:
        Thermohaline diffusion coefficient $D_{\\mathrm{thrm}}$ in
        $\\mathrm{cm^2\\,s^{-1}}$.
    """
    dgrad = max(1.0e-40, grada - gradr)
    k_therm = (
        4.0 * _const.CRAD * _const.CLIGHT * (t * t * t) / (3.0 * opacity * rho)
    )
    if option == _types.ThermohalineOption.KIPPENHAHN:
        d_raw = (
            -3.0 * k_therm / (2.0 * rho * cp) * gradl_composition_term / dgrad
        )
    else:
        k_t, k_mu, nu = _get_diff_coeffs(
            k_therm, cp, rho, t, opacity, species, xh1
        )
        r0 = (gradr - grada) / gradl_composition_term
        pr = nu / k_t
        tau = k_mu / k_t
        r_th = (r0 - 1.0) / (1.0 / tau - 1.0)
        if r_th >= 1.0 or pr < 0.0:
            d_raw = 0.0
        elif option == _types.ThermohalineOption.TRAXLER_GARAUD_STELLMACH_11:
            d_raw = (
                101.0
                * math.sqrt(k_mu * nu)
                * math.exp(-3.6 * r_th)
                * math.pow(1.0 - r_th, 1.1)
            )
        else:
            d_raw = k_mu * (_numu(r0, r_th, pr, tau) - 1.0)
    return thermohaline_coeff * d_raw


def set_thermohaline(
    params: ThermohalineParams,
    inp: ThermohalineInputs,
) -> _types.TurbOutput:
    """Evaluate thermohaline mixing outputs (`set_thermohaline` in `turb.f90`).

    Args:
        params: Non-differentiable configuration (`ThermohalineParams`).
        inp: Differentiable thermodynamic inputs (`ThermohalineInputs`).

    Returns:
        `TurbOutput` with `mixing_type = MixingType.THERMOHALINE_MIXING`.
    """
    d_thrm = get_d_thermohaline(
        option=params.option,
        grada=_types.val_of(inp.grada),
        gradr=_types.val_of(inp.gradr),
        t=_types.val_of(inp.t),
        opacity=_types.val_of(inp.opacity),
        rho=_types.val_of(inp.rho),
        cp=_types.val_of(inp.cp),
        gradl_composition_term=params.gradl_composition_term,
        species=params.species,
        xh1=params.xh1,
        thermohaline_coeff=params.thermohaline_coeff,
    )
    proto = inp.gradr
    d = _types.const_like(proto, d_thrm)
    return _types.TurbOutput(
        gamma=_types.const_like(proto, 0.0),
        grad_t=inp.gradr,
        y_face=inp.gradr - inp.grada,
        conv_vel=3.0 * d / inp.lambda_,
        d=d,
        mixing_type=_const.MixingType.THERMOHALINE_MIXING,
    )


def _get_diff_coeffs(
    k_therm: float,
    cp: float,
    rho: float,
    t: float,
    opacity: float,
    species: _types.DrivingSpecies,
    xh1: float,
) -> tuple[float, float, float]:
    """Compute `(kt, kmu, vis)` following Proffitt & Michaud (1993)."""
    kt = k_therm / (cp * rho)
    qe2 = _const.QE * _const.QE
    qe4 = qe2 * qe2
    loglambdah_base = (
        -19.26
        - 0.5 * math.log(rho)
        + 1.5 * math.log(t)
        - 0.5 * math.log(1.0 + 0.5 * (1.0 + xh1))
    )
    t2 = t * t
    nu_rad = (
        4.0
        * _const.CRAD
        * (t2 * t2)
        / (15.0 * _const.CLIGHT * opacity * (rho * rho))
    )
    nu_mol = (
        0.406
        * math.sqrt(_const.AMU)
        * math.pow(_const.BOLTZM * t, 2.5)
        / (qe4 * loglambdah_base * rho)
    )
    vis = nu_mol + nu_rad
    b_coeff = (
        (15.0 / 16.0)
        * math.sqrt(2.0 * _const.AMU / (5.0 * _const.PI))
        * math.pow(_const.BOLTZM, 2.5)
        / qe4
    )
    if species.chem_z > 2.0:
        loglambdacx = loglambdah_base - math.log(species.chem_z)
        loglambdacy = loglambdah_base - math.log(2.0 * species.chem_z)
        ccx = math.log(math.exp(1.2 * loglambdacx) + 1.0) / 1.2
        ccy = math.log(math.exp(1.2 * loglambdacy) + 1.0) / 1.2
        acx = species.chem_a / (1.0 + species.chem_a)
        acy = 4.0 * species.chem_a / (4.0 + species.chem_a)
        kmu = (
            2.0
            * b_coeff
            * math.pow(t, 2.5)
            / (math.sqrt(5.0) * rho * species.chem_z * species.chem_z)
            / (xh1 * math.sqrt(acx) * ccx + (1.0 - xh1) * math.sqrt(acy) * ccy)
        )
    else:
        loglambdah = loglambdah_base - math.log(2.0)
        ccy = math.log(math.exp(1.2 * loglambdah) + 1.0) / 1.2
        kmu = (
            (b_coeff * math.pow(t, 2.5) / (rho * ccy))
            * (3.0 + xh1)
            / ((1.0 + xh1) * (3.0 + 5.0 * xh1) * (0.7 + 0.3 * xh1))
        )
    return kt, kmu, vis


def _numu(r0: float, r_th: float, prandtl: float, diffratio: float) -> float:
    """Evaluate salt-finger Nusselt number $\\mathrm{Nu}_\\mu$ (Brown 2013)."""
    maxl_est, lambda_est = _analytical_estimate_th(r_th, prandtl, diffratio)
    vars_sol = [maxl_est, lambda_est]
    ok = _nr_solve(vars_sol, prandtl, diffratio, r0)
    it = 1
    while it <= 200 and (vars_sol[1] < 0.0 or not ok):
        vars_sol = [maxl_est, lambda_est]
        scale = 1.0 + it * 1.0e-2
        if _nr_solve(vars_sol, prandtl * scale, diffratio, r0 / scale):
            ok = _nr_solve(vars_sol, prandtl, diffratio, r0)
        it += 1
    if vars_sol[1] < 0.0 or not ok:
        maxl2 = maxl_est * maxl_est
        lambdamax = lambda_est
    else:
        maxl2 = vars_sol[0] * vars_sol[0]
        lambdamax = vars_sol[1]
    return 1.0 + 49.0 * lambdamax * lambdamax / (
        diffratio * maxl2 * (lambdamax + diffratio * maxl2)
    )


def _analytical_estimate_th(
    r_th: float,
    prandtl: float,
    diffratio: float,
) -> tuple[float, float]:
    """Analytical starting guess $(l_{\\max}, \\lambda_{\\max})$.

    Follows Brown, Garaud, & Stellmach (2013) Appendix B.
    """
    phi = diffratio / prandtl
    if r_th < 0.5 and r_th <= prandtl:
        maxl = math.sqrt(
            math.sqrt(1.0 / (1.0 + phi))
            - math.sqrt(prandtl) * (1.0 + phi / ((1.0 + phi) ** 2))
        )
        lambdamax = math.sqrt(prandtl) - prandtl * math.sqrt(1.0 + phi)
    else:
        if r_th < 0.5:
            maxl = math.pow(
                (1.0 / (1.0 + phi))
                - 2.0 * math.sqrt(r_th * phi) / math.pow(1.0 + phi, 2.5),
                0.25,
            )
        else:
            omr = 1.0 - r_th
            maxl = math.pow(
                _const.ONE_THIRD * omr + omr * omr * (5.0 - 4.0 * phi) / 27.0,
                0.25,
            )
        l2 = maxl * maxl
        l4 = l2 * l2
        lambdamax = 2.0 * prandtl * phi * l4 * l2 / (1.0 - (1.0 + phi) * l4)
    if lambdamax < 0.0:
        return 0.5, 0.5
    return maxl, lambdamax


def _nr_solve(
    xrk: list[float],
    prandtl: float,
    diffratio: float,
    r0: float,
) -> bool:
    """2D Newton-Raphson solver for Brown et al. (2013) eqs. 19-20."""
    acy = 1.0e-13
    sav0, sav1 = xrk[0], xrk[1]
    err = 1.0e99
    errold = 1.0e99
    it = 0
    while err > acy and it < 20:
        f0, f1, j00, j01, j10, j11 = _thermohaline_rhs(
            xrk[0], xrk[1], prandtl, diffratio, r0
        )
        det = j00 * j11 - j01 * j10
        if det == 0.0 or not math.isfinite(det):
            return False
        dx0 = (j11 * f0 - j01 * f1) / det
        dx1 = (-j10 * f0 + j00 * f1) / det
        it += 1
        err = math.sqrt(dx0 * dx0 + dx1 * dx1)
        if it > 5 and err > errold:
            xrk[0], xrk[1] = sav0, sav1
            return False
        xrk[0] -= dx0
        xrk[1] -= dx1
        errold = err
    return err <= acy


def _thermohaline_rhs(
    x0: float,
    x1: float,
    prandtl: float,
    diffratio: float,
    r0: float,
) -> tuple[float, float, float, float, float, float]:
    """Evaluate `(f0, f1, j00, j01, j10, j11)` for Brown et al. (2013)."""
    b_2 = 1.0 + prandtl + diffratio
    x0_2 = x0 * x0
    x0_3 = x0_2 * x0
    x0_4 = x0_3 * x0
    a_2 = x0_2 * b_2
    myterm = diffratio * prandtl + prandtl + diffratio
    b_1 = 2.0 * x0_2 * myterm
    a_1 = x0_4 * myterm + prandtl * (1.0 - (1.0 / r0))
    b_0 = 3.0 * x0_4 * diffratio * prandtl + prandtl * (diffratio - (1.0 / r0))
    a_0 = x0_4 * x0_2 * diffratio * prandtl + x0_2 * prandtl * (
        diffratio - (1.0 / r0)
    )
    f0 = ((x1 + a_2) * x1 + a_1) * x1 + a_0
    f1 = b_2 * x1 * x1 + b_1 * x1 + b_0
    j00 = (
        2.0 * x0 * b_2 * x1 * x1
        + 4.0 * x0_3 * myterm * x1
        + 6.0 * x0_4 * x0 * diffratio * prandtl
        + 2.0 * x0 * prandtl * (diffratio - (1.0 / r0))
    )
    j01 = 3.0 * x1 * x1 + 2.0 * a_2 * x1 + a_1
    j10 = 4.0 * x0 * myterm * x1 + 12.0 * x0_3 * diffratio * prandtl
    j11 = 2.0 * b_2 * x1 + b_1
    return f0, f1, j00, j01, j10, j11
