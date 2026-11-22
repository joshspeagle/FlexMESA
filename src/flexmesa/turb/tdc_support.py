# SPDX-License-Identifier: LGPL-3.0-or-later
"""Time-dependent convection physics kernels (`turb/private/tdc_support.f90`).

Provides the Kuhfuss (1986) / Smolec time-dependent convection velocity
evolution (`eval_xis`, `eval_af`), Wuchterl & Feuchtinger (1998) quintic
smootherstep enthalpy flux limiter (`flux_limiter_function`), and luminosity
residual evaluation (`compute_q`) generic over `float`, `DualOrder1`, and
`AutoDiffRealTdc`.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import math
from typing import Final

from flexmesa import auto_diff as _ad
from flexmesa import constants as _const
from flexmesa.turb import tdc_dual as _tdc_dual
from flexmesa.turb import types as _types

X_ALFAS: Final[float] = 0.5 * _const.SQRT_2_DIV_3
X_CEDE: Final[float] = (8.0 / 3.0) * _const.SQRT_2_DIV_3
X_ALFAP: Final[float] = 2.0 / 3.0


@dataclasses.dataclass(frozen=True)
class TdcInfo[
    S: (
        float,
        _ad.DualOrder1,
        _types.ScalarOrDual,
        _tdc_dual.AutoDiffRealTdc,
    )
]:
    """Internal state container for TDC evaluations independent of $Y$.

    Matches `tdc_info` in `turb/private/tdc_support.f90`.

    Attributes:
        include_mlt_corr_to_tdc: Include steady-state Cox MLT geometric
            correction $\\Gamma / (1 + \\Gamma)$.
        use_tdc_enthalpy_flux_limiter: Enable Wuchterl & Feuchtinger (1998)
            quintic enthalpy flux limiter.
        mixing_length_alpha: Mixing-length parameter $\\alpha_{\\mathrm{MLT}}$.
        tdc_alpha_c: Convective flux multiplier $\\alpha_C$.
        tdc_alpha_s: Buoyant source multiplier $\\alpha_S$.
        tdc_alpha_d: Turbulent drag multiplier $\\alpha_D$.
        tdc_alpha_r: Radiative damping multiplier $\\alpha_R$.
        tdc_alpha_pt: Turbulent pressure work multiplier $\\alpha_{P_t}$.
        dt: Time step $\\Delta t$ in $\\mathrm{s}$.
        a0: Initial scaled convective speed $A_0$ in $\\mathrm{cm\\,s^{-1}}$.
        c0: Convective luminosity prefactor $c_0$ in $\\mathrm{erg\\,cm^{-1}}$.
        lum: Total luminosity $L$ in $\\mathrm{erg\\,s^{-1}}$.
        l0: Radiative luminosity scale $L_0 = L / \\nabla_r$ in
            $\\mathrm{erg\\,s^{-1}}$.
        gradl: Ledoux temperature gradient $\\nabla_L$ (dimensionless).
        grada: Adiabatic temperature gradient $\\nabla_{\\mathrm{ad}}$.
        t: Temperature $T$ in $\\mathrm{K}$.
        rho: Mass density $\\rho$ in $\\mathrm{g\\,cm^{-3}}$.
        dv: Specific volume change $\\Delta(1/\\rho)$ in
            $\\mathrm{cm^3\\,g^{-1}}$.
        cp: Specific heat $C_P$ in $\\mathrm{erg\\,g^{-1}\\,K^{-1}}$.
        kap: Rosseland mean opacity $\\kappa$ in $\\mathrm{cm^2\\,g^{-1}}$.
        hp: Pressure scale height $H_P$ in $\\mathrm{cm}$.
        gamma: Steady-state Cox MLT efficiency $\\Gamma$ (dimensionless).
        eq_div_w: Coupled source term $E_q / w$ in $\\mathrm{cm\\,s^{-2}}$.
        h: Specific enthalpy $h = e + P/\\rho$ in $\\mathrm{erg\\,g^{-1}}$.
    """

    include_mlt_corr_to_tdc: bool
    use_tdc_enthalpy_flux_limiter: bool
    mixing_length_alpha: float
    tdc_alpha_c: float
    tdc_alpha_s: float
    tdc_alpha_d: float
    tdc_alpha_r: float
    tdc_alpha_pt: float
    dt: float
    a0: S
    c0: S
    lum: S
    l0: S
    gradl: S
    grada: S
    t: S
    rho: S
    dv: S
    cp: S
    kap: S
    hp: S
    gamma: S
    eq_div_w: S
    h: S

    def _scalar_fields(self) -> tuple[S, ...]:
        """Return all 15 differentiable scalar fields as a tuple."""
        return (
            self.a0,
            self.c0,
            self.lum,
            self.l0,
            self.gradl,
            self.grada,
            self.t,
            self.rho,
            self.dv,
            self.cp,
            self.kap,
            self.hp,
            self.gamma,
            self.eq_div_w,
            self.h,
        )

    def _map_scalars[
        U: (
            float,
            _ad.DualOrder1,
            _types.ScalarOrDual,
            _tdc_dual.AutoDiffRealTdc,
        )
    ](
        self,
        fn: collections.abc.Callable[[S], U],
    ) -> TdcInfo[U]:
        """Project all 15 differentiable fields via a scalar callable."""
        return TdcInfo(
            include_mlt_corr_to_tdc=self.include_mlt_corr_to_tdc,
            use_tdc_enthalpy_flux_limiter=self.use_tdc_enthalpy_flux_limiter,
            mixing_length_alpha=self.mixing_length_alpha,
            tdc_alpha_c=self.tdc_alpha_c,
            tdc_alpha_s=self.tdc_alpha_s,
            tdc_alpha_d=self.tdc_alpha_d,
            tdc_alpha_r=self.tdc_alpha_r,
            tdc_alpha_pt=self.tdc_alpha_pt,
            dt=self.dt,
            a0=fn(self.a0),
            c0=fn(self.c0),
            lum=fn(self.lum),
            l0=fn(self.l0),
            gradl=fn(self.gradl),
            grada=fn(self.grada),
            t=fn(self.t),
            rho=fn(self.rho),
            dv=fn(self.dv),
            cp=fn(self.cp),
            kap=fn(self.kap),
            hp=fn(self.hp),
            gamma=fn(self.gamma),
            eq_div_w=fn(self.eq_div_w),
            h=fn(self.h),
        )

    def to_primal(self) -> TdcInfo[float]:
        """Project all differentiable fields to primal `float` scalars.

        Returns:
            A `TdcInfo[float]` instance for scalar bisection evaluations.
        """
        return self._map_scalars(_types.val_of)

    def to_z_dual(self) -> TdcInfo[_ad.DualOrder1]:
        """Project all differentiable fields to 1D constant `DualOrder1`.

        Returns:
            A `TdcInfo[DualOrder1]` instance whose scalar fields are 1-variable
            dual constants for 1D differentiation with respect to $Z = \\ln|Y|$.
        """
        return self._map_scalars(_to_const_dual1)

    def to_tdc(self) -> tuple[TdcInfo[_tdc_dual.AutoDiffRealTdc], bool]:
        """Lift every structure field into `AutoDiffRealTdc` (`convert`).

        Returns:
            Two-tuple `(info_tdc, has_dual)` where `info_tdc` carries every
            field as an `AutoDiffRealTdc` of common arity `N` (`N = 0` when all
            fields are plain `float`), and `has_dual` is `True` if any input
            field was a `DualOrder1`.

        Raises:
            ValueError: If two `DualOrder1` fields disagree in arity.
        """
        num_vars = 0
        has_dual = False
        for field in self._scalar_fields():
            if isinstance(field, _ad.DualOrder1):
                n = int(field.d1.shape[-1])
                if has_dual and n != num_vars:
                    msg = (
                        f"Mismatched DualOrder1 arities in TdcInfo: "
                        f"{num_vars} vs {n}"
                    )
                    raise ValueError(msg)
                num_vars = n
                has_dual = True
            elif isinstance(field, _tdc_dual.AutoDiffRealTdc):
                n = field.num_vars
                if has_dual and n != num_vars:
                    msg = (
                        f"Mismatched AutoDiffRealTdc arities in TdcInfo: "
                        f"{num_vars} vs {n}"
                    )
                    raise ValueError(msg)
                num_vars = n
                has_dual = n > 0

        def _lift(x: S) -> _tdc_dual.AutoDiffRealTdc:
            if isinstance(x, _tdc_dual.AutoDiffRealTdc):
                return x
            return _tdc_dual.AutoDiffRealTdc.convert(x, num_vars)

        return self._map_scalars(_lift), has_dual


def _to_const_dual1(x: _types.TdcScalar) -> _ad.DualOrder1:
    """Wrap the primal value of `x` as a 1-variable constant `DualOrder1`."""
    return _ad.DualOrder1.constant(_types.val_of(x), num_vars=1)


def set_y[
    S: (
        float,
        _ad.DualOrder1,
        _types.ScalarOrDual,
        _tdc_dual.AutoDiffRealTdc,
    )
](y_is_positive: bool, z: S) -> S:
    """Map logarithmic coordinate $Z = \\ln|Y|$ to $Y = \\pm \\exp(Z)$.

    Args:
        y_is_positive: Sign flag (`True` for $Y > 0$, `False` for $Y < 0$).
        z: Dimensionless logarithmic superadiabaticity $Z = \\ln|Y|$.

    Returns:
        Dimensionless superadiabaticity $Y = \\pm \\exp(Z)$.
    """
    exp_z = _types.s_exp(z)
    return exp_z if y_is_positive else -exp_z


def safe_tanh[
    S: (
        float,
        _ad.DualOrder1,
        _types.ScalarOrDual,
        _tdc_dual.AutoDiffRealTdc,
    )
](x: S) -> S:
    """Evaluate numerically safe hyperbolic tangent clamping $|x| > 50$.

    Args:
        x: Dimensionless input argument.

    Returns:
        Dimensionless clamped $\\tanh(x)$ matching `safe_tanh` in
        `tdc_support.f90`.
    """
    xv = _types.val_of(x)
    if xv > 50.0:
        return _types.const_like(x, 1.0)
    if xv < -50.0:
        return _types.const_like(x, -1.0)
    return _types.s_tanh(x)


def flux_limiter_function[
    S: (
        float,
        _ad.DualOrder1,
        _types.ScalarOrDual,
        _tdc_dual.AutoDiffRealTdc,
    )
](x: S) -> S:
    """Evaluate Wuchterl & Feuchtinger (1998) $C^2$ quintic smootherstep.

    Args:
        x: Dimensionless ratio of convective source enthalpy to specific
            enthalpy $h$.

    Returns:
        Dimensionless limited ratio (`x` for $x < 0.95$, `1.0` for $x \\ge 1.0$,
        and a quintic blend on $[0.95, 1.0)$).
    """
    x0 = 0.95
    x1 = 1.0
    xv = _types.val_of(x)
    if xv < x0:
        return x
    if xv >= x1:
        return _types.const_like(x, 1.0)
    s = (x - x0) / (x1 - x0)
    p = _types.s_pow3(s) * (10.0 + s * (-15.0 + 6.0 * s))
    return x + (1.0 - x) * p


def eval_xis[
    S: (
        float,
        _ad.DualOrder1,
        _types.ScalarOrDual,
        _tdc_dual.AutoDiffRealTdc,
    )
](
    info: TdcInfo[S],
    y: S,
) -> tuple[S, S, S]:
    """Evaluate $(\\xi_0, \\xi_1, \\xi_2)$ of the TDC velocity equation.

    Args:
        info: Precomputed cell-face TDC state (`TdcInfo`).
        y: Dimensionless envelope-corrected superadiabaticity
            $Y_{\\mathrm{env}}$.

    Returns:
        Three-tuple `(xi0, xi1, xi2)` in $(\\mathrm{cm^2\\,s^{-3}},
        \\mathrm{s^{-1}}, \\mathrm{cm^{-1}})$ for $2\\,dw/dt = \\xi_0 + \\xi_1 w
        + \\xi_2 w^2$, matching `eval_xis` in `tdc_support.f90`.
    """
    x_gammar = 2.0 * math.sqrt(3.0)
    s0_pref = (
        (info.tdc_alpha_s * X_ALFAS * info.mixing_length_alpha)
        * info.cp
        * info.t
        / info.hp
        * info.grada
    )
    if info.use_tdc_enthalpy_flux_limiter:
        scale: S = _types.const_like(y, 1.0)
        if _types.val_of(y) > 0.0:
            factor = (
                info.mixing_length_alpha
                * info.tdc_alpha_s
                * X_ALFAS
                / _const.SQRT_2_DIV_3
            )
            x = (info.cp * info.t / info.h) * factor * y
            fl = flux_limiter_function(x)
            if abs(_types.val_of(x)) >= 0.95:
                scale = fl / x
        s0 = s0_pref * (scale * y) + info.eq_div_w
    else:
        s0 = s0_pref * y + info.eq_div_w

    d0 = (info.tdc_alpha_d * X_CEDE / info.mixing_length_alpha) / info.hp
    gammar_div_alfa = (
        info.tdc_alpha_r * x_gammar / info.mixing_length_alpha
    ) / info.hp
    dr0 = (
        (4.0 * _const.BOLTZ_SIGMA)
        * _types.s_pow2(gammar_div_alfa)
        * _types.s_pow3(info.t)
        / (_types.s_pow2(info.rho) * info.cp * info.kap)
    )
    pt0 = (info.tdc_alpha_pt * X_ALFAP) * info.rho
    dv_dt = info.dv / info.dt
    return s0, -(dr0 + pt0 * dv_dt), -d0


def eval_af[
    S: (
        float,
        _ad.DualOrder1,
        _types.ScalarOrDual,
        _tdc_dual.AutoDiffRealTdc,
    )
](
    dt: float,
    a0: S,
    xi0: S,
    xi1: S,
    xi2: S,
) -> S:
    """Evaluate final scaled convective speed $A_f(\\Delta t)$ (`eval_Af`).

    Args:
        dt: Time step $\\Delta t$ in $\\mathrm{s}$.
        a0: Initial scaled speed $A_0$ in $\\mathrm{cm\\,s^{-1}}$.
        xi0: Buoyant source coefficient $\\xi_0$ in $\\mathrm{cm^2\\,s^{-3}}$.
        xi1: Linear damping coefficient $\\xi_1$ in $\\mathrm{s^{-1}}$.
        xi2: Quadratic drag coefficient $\\xi_2$ in $\\mathrm{cm^{-1}}$.

    Returns:
        Final scaled convective speed $A_f(\\Delta t)$ in
        $\\mathrm{cm\\,s^{-1}}$.
    """
    j2 = _types.s_pow2(xi1) - 4.0 * xi0 * xi2
    j2_val = _types.val_of(j2)
    if j2_val > 0.0:
        j = _types.s_sqrt(_types.s_abs(j2))
        jt4 = (0.25 * dt) * j
        th = safe_tanh(jt4)
        num = th * (2.0 * xi0 + a0 * xi1) + a0 * j
        den = th * (xi1 + 2.0 * a0 * xi2) - j
        af = num / den
        return -af if _types.val_of(af) < 0.0 else af
    if j2_val < 0.0:
        j = _types.s_sqrt(_types.s_abs(j2))
        jt4 = (0.25 * dt) * j
        y_for_atan = xi1 + 2.0 * a0 * xi2
        atan_y = _types.s_atan(y_for_atan / j)
        root = _types.s_atan(xi1 / j) - atan_y
        rv = _types.val_of(root)
        if rv > _const.PI:
            root = root - _const.PI
        elif rv < -_const.PI:
            root = root + 2.0 * _const.PI
        elif rv < 0.0:
            root = root + _const.PI
        if _types.val_of(jt4) < _types.val_of(root):
            num = -xi1 + j * _types.s_tan(jt4 + atan_y)
            den = 2.0 * xi2
            return num / den
        return _types.const_like(xi0, 0.0)
    return a0


def compute_q[
    S: (
        float,
        _ad.DualOrder1,
        _types.ScalarOrDual,
        _tdc_dual.AutoDiffRealTdc,
    )
](
    info: TdcInfo[S],
    y: S,
) -> tuple[S, S]:
    """Evaluate luminosity residual $Q(Y)$ and final speed $A_f$ (`compute_Q`).

    Args:
        info: Precomputed cell-face TDC state (`TdcInfo`).
        y: Superadiabaticity $Y = \\nabla_T - \\nabla_L$.

    Returns:
        Two-tuple `(q, af)` of luminosity residual $Q(Y)$ in
        $\\mathrm{erg\\,s^{-1}}$ and final scaled speed $A_f$ in
        $\\mathrm{cm\\,s^{-1}}$.
    """
    if _types.val_of(y) > 0.0 and info.include_mlt_corr_to_tdc:
        y_env = y * (info.gamma / (1.0 + info.gamma))
    else:
        y_env = y
    xi0, xi1, xi2 = eval_xis(info, y_env)
    af = eval_af(info.dt, info.a0, xi0, xi1, xi2)
    q = (info.lum - info.l0 * info.gradl) - info.l0 * y - info.c0 * af * y_env
    return q, af
