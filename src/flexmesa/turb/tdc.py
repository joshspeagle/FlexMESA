# SPDX-License-Identifier: LGPL-3.0-or-later
"""Time-Dependent Convection solver (`set_tdc`, `turb/private/tdc.f90`).

Solves the Kuhfuss (1986) / Smolec time-dependent convection energy balance
equation
$$Q(Y) = (L - L_0 \\nabla_L) - L_0 Y
  - c_0 A_f(Y_{\\mathrm{env}}) Y_{\\mathrm{env}} = 0$$
over $Z = \\ln|Y| \\in [-100, 100]$ using bisection bracketing followed by
damped Newton-Raphson refinement (`bracket_plus_newton_search`). The Newton
iteration runs on `AutoDiffRealTdc` (upstream's `auto_diff_real_tdc`), so the
returned $Y$ carries the structure partials of the last Newton update
(`turb/private/tdc.f90:253-345`).
"""

from __future__ import annotations

import dataclasses

from flexmesa import constants as _const
from flexmesa.turb import mlt as _mlt
from flexmesa.turb import tdc_search as _search
from flexmesa.turb import tdc_support as _support
from flexmesa.turb import types as _types


@dataclasses.dataclass(frozen=True)
class TdcParams:
    """Non-differentiable configuration parameters for `set_tdc`.

    Attributes:
        conv_vel_start: Initial convective speed $v_{\\mathrm{conv,start}}$ in
            $\\mathrm{cm\\,s^{-1}}$.
        mixing_length_alpha: Mixing-length parameter $\\alpha_{\\mathrm{MLT}}$.
        tdc_alpha_d: Turbulent dissipation multiplier $\\alpha_D$.
        tdc_alpha_r: Radiative damping multiplier $\\alpha_R$.
        tdc_alpha_pt: Turbulent pressure work multiplier $\\alpha_{P_t}$.
        tdc_alpha_c: Convective flux multiplier $\\alpha_C$.
        tdc_alpha_s: Buoyant source multiplier $\\alpha_S$.
        dt: Time step $\\Delta t$ in $\\mathrm{s}$.
        cgrav: Gravitational constant $G$ in $\\mathrm{cm^3\\,g^{-1}\\,s^{-2}}$.
        m: Enclosed mass $m$ at the cell face in $\\mathrm{g}$.
        scale: Luminosity scale for relative residual convergence in
            $\\mathrm{erg\\,s^{-1}}$.
        max_conv_vel: Upper cap on convective velocity in
            $\\mathrm{cm\\,s^{-1}}$.
        include_mlt_corr_to_tdc: Include steady-state Cox MLT geometric
            correction $\\Gamma / (1 + \\Gamma)$.
        use_tdc_enthalpy_flux_limiter: Enable Wuchterl & Feuchtinger (1998)
            quintic enthalpy flux limiter.
    """

    conv_vel_start: float = 0.0
    mixing_length_alpha: float = 2.0
    tdc_alpha_d: float = 1.0
    tdc_alpha_r: float = 0.0
    tdc_alpha_pt: float = 0.0
    tdc_alpha_c: float = 1.0
    tdc_alpha_s: float = 1.0
    dt: float = 1.0
    cgrav: float = _const.STANDARD_CGRAV
    m: float = _const.MSUN
    scale: float = _const.LSUN
    max_conv_vel: float = 1.0e99
    include_mlt_corr_to_tdc: bool = True
    use_tdc_enthalpy_flux_limiter: bool = False


@dataclasses.dataclass(frozen=True)
class TdcInputs:
    """Differentiable thermodynamic and structure inputs for `set_tdc`.

    Attributes:
        chi_t: $(\\partial \\ln P / \\partial \\ln T)_\\rho$ (dimensionless).
        chi_rho: $(\\partial \\ln P / \\partial \\ln \\rho)_T$ (dimensionless).
        gradr: Radiative temperature gradient $\\nabla_r$ (dimensionless).
        r: Radial coordinate $r$ at the cell face in $\\mathrm{cm}$.
        p: Total pressure $P$ in $\\mathrm{dyn\\,cm^{-2}}$.
        t: Temperature $T$ in $\\mathrm{K}$.
        rho: Mass density $\\rho$ in $\\mathrm{g\\,cm^{-3}}$.
        dv: Specific volume change $\\Delta(1/\\rho)$ in
            $\\mathrm{cm^3\\,g^{-1}}$.
        cp: Specific heat $C_P$ in $\\mathrm{erg\\,g^{-1}\\,K^{-1}}$.
        opacity: Rosseland mean opacity $\\kappa$ in $\\mathrm{cm^2\\,g^{-1}}$.
        scale_height: Pressure scale height $H_P$ in $\\mathrm{cm}$.
        gradl: Ledoux temperature gradient $\\nabla_L$ (dimensionless).
        grada: Adiabatic temperature gradient $\\nabla_{\\mathrm{ad}}$.
        eq_div_w: Coupled source term $E_q / w$ in $\\mathrm{cm\\,s^{-2}}$.
        grav: Local gravitational acceleration $g$ in $\\mathrm{cm\\,s^{-2}}$.
        energy: Specific internal energy $e$ in $\\mathrm{erg\\,g^{-1}}$.
    """

    chi_t: _types.ScalarOrDual
    chi_rho: _types.ScalarOrDual
    gradr: _types.ScalarOrDual
    r: _types.ScalarOrDual
    p: _types.ScalarOrDual
    t: _types.ScalarOrDual
    rho: _types.ScalarOrDual
    dv: _types.ScalarOrDual
    cp: _types.ScalarOrDual
    opacity: _types.ScalarOrDual
    scale_height: _types.ScalarOrDual
    gradl: _types.ScalarOrDual
    grada: _types.ScalarOrDual
    eq_div_w: _types.ScalarOrDual
    grav: _types.ScalarOrDual
    energy: _types.ScalarOrDual


@dataclasses.dataclass(frozen=True)
class TdcOutput:
    """Output structure returned by `set_tdc`.

    Attributes:
        output: Standard turbulent transport outputs (`TurbOutput`).
        num_iters: Number of Newton iterations taken in the TDC root solver.
    """

    output: _types.TurbOutput
    num_iters: int


def set_tdc(
    params: TdcParams,
    inp: TdcInputs,
) -> TdcOutput:
    """Evaluate Time-Dependent Convection (`set_TDC` in `turb/public/turb.f90`).

    Calls `set_mlt` with Cox MLT unconditionally before packing `TdcInfo`
    (`turb/public/turb.f90:142-145`) and enforces upstream's
    `mixing_length_alpha == 0` / `dt <= 0` guard in `get_TDC_solution`
    (`turb/private/tdc.f90:65-67`).

    Args:
        params: Non-differentiable configuration parameters (`TdcParams`).
        inp: Differentiable thermodynamic inputs (`TdcInputs`) with `float`
            or `DualOrder1` fields.

    Returns:
        `TdcOutput` pairing `TurbOutput` with `num_iters`.

    Raises:
        ValueError: If `mixing_length_alpha == 0` or `dt <= 0`.
        RuntimeError: If the TDC root solver fails to converge.
    """
    proto = inp.gradr
    lum = (
        (64.0 * _const.PI * _const.BOLTZ_SIGMA)
        * _types.s_pow4(inp.t)
        * inp.grav
        * _types.s_pow2(inp.r)
        * inp.gradr
        / (3.0 * inp.p * inp.opacity)
    )

    # Upstream `turb/public/turb.f90:141-145` always calls `set_MLT('Cox', ...)`
    # before packing `TdcInfo`.
    lambda_ = params.mixing_length_alpha * inp.scale_height
    mlt_params = _types.MltParams(
        option=_types.MltOption.COX,
        mixing_length_alpha=params.mixing_length_alpha,
        henyey_nu=0.0,
        henyey_y=0.0,
        max_conv_vel=1.0e99,
    )
    mlt_inp = _types.MltInputs(
        chi_t=inp.chi_t,
        chi_rho=inp.chi_rho,
        cp=inp.cp,
        grav=inp.grav,
        lambda_=lambda_,
        rho=inp.rho,
        p=inp.p,
        t=inp.t,
        opacity=inp.opacity,
        gradr=inp.gradr,
        grada=inp.grada,
        gradl=inp.gradl,
    )
    gamma = _mlt.set_mlt(mlt_params, mlt_inp).gamma

    c0 = (
        (
            params.tdc_alpha_c
            * params.mixing_length_alpha
            * _support.X_ALFAS
            * 4.0
            * _const.PI
        )
        * inp.rho
        * inp.t
        * inp.cp
        * _types.s_pow2(inp.r)
    )
    l0 = (
        (
            (16.0 * _const.PI * _const.CRAD * _const.CLIGHT / 3.0)
            * params.cgrav
            * params.m
        )
        * _types.s_pow4(inp.t)
        / (inp.p * inp.opacity)
    )
    a0 = _types.const_like(proto, params.conv_vel_start / _const.SQRT_2_DIV_3)
    h = inp.energy + inp.p / inp.rho

    info: _support.TdcInfo[_types.ScalarOrDual] = _support.TdcInfo(
        include_mlt_corr_to_tdc=params.include_mlt_corr_to_tdc,
        use_tdc_enthalpy_flux_limiter=params.use_tdc_enthalpy_flux_limiter,
        mixing_length_alpha=params.mixing_length_alpha,
        tdc_alpha_c=params.tdc_alpha_c,
        tdc_alpha_s=params.tdc_alpha_s,
        tdc_alpha_d=params.tdc_alpha_d,
        tdc_alpha_r=params.tdc_alpha_r,
        tdc_alpha_pt=params.tdc_alpha_pt,
        dt=params.dt,
        a0=a0,
        c0=c0,
        lum=lum,
        l0=l0,
        gradl=inp.gradl,
        grada=inp.grada,
        t=inp.t,
        rho=inp.rho,
        dv=inp.dv,
        cp=inp.cp,
        kap=inp.opacity,
        hp=inp.scale_height,
        gamma=gamma,
        eq_div_w=inp.eq_div_w,
        h=h,
    )

    conv_vel, y_face, num_iters = _get_tdc_solution(
        info, params.scale, -100.0, 100.0
    )

    if _types.val_of(conv_vel) > params.max_conv_vel:
        conv_vel = _types.const_like(proto, params.max_conv_vel)
        num = info.lum - info.l0 * info.gradl
        if params.include_mlt_corr_to_tdc:
            y_face = num / (
                info.l0
                + info.c0
                * (_const.SQRT_2_DIV_3 * params.max_conv_vel)
                * (info.gamma / (1.0 + info.gamma))
            )
        else:
            y_face = num / (
                info.l0 + info.c0 * (_const.SQRT_2_DIV_3 * params.max_conv_vel)
            )

    grad_t = y_face + inp.gradl
    d = conv_vel * inp.scale_height * (params.mixing_length_alpha / 3.0)
    mixing_type = (
        _const.MixingType.CONVECTIVE_MIXING
        if _types.val_of(conv_vel) > 0.0
        else _const.MixingType.NO_MIXING
    )
    return TdcOutput(
        output=_types.TurbOutput(
            gamma=gamma,
            grad_t=grad_t,
            y_face=y_face,
            conv_vel=conv_vel,
            d=d,
            mixing_type=mixing_type,
        ),
        num_iters=num_iters,
    )


def _get_tdc_solution(
    info: _support.TdcInfo[_types.ScalarOrDual],
    scale: float,
    zlb: float,
    zub: float,
) -> tuple[_types.ScalarOrDual, _types.ScalarOrDual, int]:
    """Solve for `(conv_vel, y_face, num_iters)` (`get_TDC_solution`)."""
    # Upstream guard in `turb/private/tdc.f90:65-67`.
    if info.mixing_length_alpha == 0.0 or info.dt <= 0.0:
        msg = (
            "bad call to TDC get_TDC_solution: "
            "mixing_length_alpha == 0 or dt <= 0"
        )
        raise ValueError(msg)

    info_p = info.to_primal()
    info_z = info.to_z_dual()
    info_t, has_dual = info.to_tdc()
    q0, _ = _support.compute_q(info_p, 0.0)
    y_is_positive = q0 > 0.0

    num_iters = 0
    y_sol: _types.ScalarOrDual
    if y_is_positive:
        y_tdc, num_iters = _search.bracket_plus_newton_search(
            info_p, info_t, scale, True, zlb, zub
        )
        y_sol = y_tdc.unconvert(is_dual=has_dual)
    else:
        rad_y = (info.lum - info.l0 * info.gradl) / info.l0
        if _types.val_of(info.a0) < 1.0e-10:
            y_sol = rad_y
        else:
            _, af_lb = _support.compute_q(info_p, _support.set_y(False, zlb))
            if af_lb == 0.0:
                y_sol = rad_y
            else:
                _, af_ub = _support.compute_q(
                    info_p, _support.set_y(False, zub)
                )
                if af_ub > 0.0:
                    z0 = zub
                else:
                    z0 = _search.af_bisection_search(info_p, zlb, zub)

                z1 = _search.dq_dz_bisection_search(info_z, zlb, z0)
                if z1 is not None:
                    q1, _ = _support.compute_q(
                        info_p, _support.set_y(False, z1)
                    )
                    if q1 < 0.0:
                        y_sol = rad_y
                    else:
                        try:
                            y_tdc, num_iters = (
                                _search.bracket_plus_newton_search(
                                    info_p, info_t, scale, False, zlb, z1
                                )
                            )
                        except RuntimeError:
                            y_tdc, num_iters = (
                                _search.bracket_plus_newton_search(
                                    info_p, info_t, scale, False, z1, z0
                                )
                            )
                        y_sol = y_tdc.unconvert(is_dual=has_dual)
                else:
                    q_y0, _ = _support.compute_q(
                        info_p, _support.set_y(False, z0)
                    )
                    if q_y0 > 0.0:
                        y_tdc, num_iters = _search.bracket_plus_newton_search(
                            info_p, info_t, scale, False, zlb, z0
                        )
                        y_sol = y_tdc.unconvert(is_dual=has_dual)
                    else:
                        y_sol = rad_y

    _, af = _support.compute_q(info, y_sol)
    conv_vel = _const.SQRT_2_DIV_3 * af
    return conv_vel, y_sol, num_iters
