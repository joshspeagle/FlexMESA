# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared data structures and scalar-dual dispatch for `flexmesa.turb`.

Defines configuration parameter dataclasses, thermodynamic input dataclasses,
output structures, and generic mathematical dispatch over IEEE-754 `float`
and first-order forward-mode dual numbers (`flexmesa.auto_diff.DualOrder1`).
"""

from __future__ import annotations

import dataclasses
import enum
import math
import typing

from flexmesa import auto_diff as _ad
from flexmesa import constants as _const
from flexmesa.turb import tdc_dual as _tdc_dual

type ScalarOrDual = float | _ad.DualOrder1
type TdcScalar = float | _ad.DualOrder1 | _tdc_dual.AutoDiffRealTdc


class _OptionEnum(enum.Enum):
    """Base enumeration supporting case-exact whitespace-trimmed parsing."""

    @classmethod
    def _parse(cls, name: str, label: str) -> typing.Self:
        key = name.strip()
        for member in cls:
            if member.value == key:
                return member
        msg = f"Unknown {label} option: {name!r}"
        raise ValueError(msg)


class MltOption(_OptionEnum):
    """Geometric Mixing-Length Theory prescription (`MLT_option` in `mlt.f90`).

    Attributes:
        COX: Cox & Giuli (1968) optically thick prescription ($a_0 = 9/4$).
        TDC: Alias for `COX` used when falling back from TDC to MLT.
        HENYEY: Henyey, Vardya, & Bodenheimer (1965) prescription.
        ML1: Bohm-Vitense (1958) standard ML1 prescription.
        ML2: Bohm & Cassinelli (1971) ML2 prescription.
        MIHALAS: Mihalas (1978) optically thin/thick bridged prescription.
    """

    COX = "Cox"
    TDC = "TDC"
    HENYEY = "Henyey"
    ML1 = "ML1"
    ML2 = "ML2"
    MIHALAS = "Mihalas"

    @classmethod
    def from_str(cls, name: str) -> MltOption:
        """Parse an MLT option string into an `MltOption` enum member.

        Args:
            name: Option string (`"Cox"`, `"TDC"`, `"Henyey"`, `"ML1"`,
                `"ML2"`, or `"Mihalas"`).

        Returns:
            Matching `MltOption` variant.

        Raises:
            ValueError: If `name` is not a recognized MLT prescription.
        """
        return cls._parse(name, "MLT")


class SemiconvectionOption(_OptionEnum):
    """Semiconvective mixing prescription (`semiconvection_option`).

    Attributes:
        LANGER_85_GRADT_IS_GRADR: Langer (1985) diffusivity with
            $\\nabla_T = \\nabla_r$.
        LANGER_85: Full Langer (1985) quadratic solution for $\\nabla_T$.
    """

    LANGER_85_GRADT_IS_GRADR = "Langer_85 mixing; gradT = gradr"
    LANGER_85 = "Langer_85"

    @classmethod
    def from_str(cls, name: str) -> SemiconvectionOption:
        """Parse a semiconvection option string.

        Args:
            name: Option string (`"Langer_85 mixing; gradT = gradr"` or
                `"Langer_85"`).

        Returns:
            Matching `SemiconvectionOption` variant.

        Raises:
            ValueError: If `name` is not recognized.
        """
        return cls._parse(name, "semiconvection")


class ThermohalineOption(_OptionEnum):
    """Thermohaline mixing prescription (`thermohaline_option`).

    Attributes:
        KIPPENHAHN: Kippenhahn, Ruschenplatt, & Thomas (1980) formula.
        TRAXLER_GARAUD_STELLMACH_11: Traxler, Garaud, & Stellmach (2011) fit.
        BROWN_GARAUD_STELLMACH_13: Brown, Garaud, & Stellmach (2013) model.
    """

    KIPPENHAHN = "Kippenhahn"
    TRAXLER_GARAUD_STELLMACH_11 = "Traxler_Garaud_Stellmach_11"
    BROWN_GARAUD_STELLMACH_13 = "Brown_Garaud_Stellmach_13"

    @classmethod
    def from_str(cls, name: str) -> ThermohalineOption:
        """Parse a thermohaline option string.

        Args:
            name: Option string (`"Kippenhahn"`,
                `"Traxler_Garaud_Stellmach_11"`, or
                `"Brown_Garaud_Stellmach_13"`).

        Returns:
            Matching `ThermohalineOption` variant.

        Raises:
            ValueError: If `name` is not recognized.
        """
        return cls._parse(name, "thermohaline")


@dataclasses.dataclass(frozen=True)
class DrivingSpecies:
    """Atomic charge $Z$ and mass number $A$ of a thermohaline-driving species.

    Attributes:
        chem_z: Atomic charge number $Z$ (dimensionless).
        chem_a: Atomic mass number $A = Z + N$ (dimensionless).
    """

    chem_z: float
    chem_a: float

    @classmethod
    def he4(cls) -> DrivingSpecies:
        """Return ${}^4\\mathrm{He}$ ($Z = 2, A = 4$)."""
        return cls(2.0, 4.0)

    @classmethod
    def he3(cls) -> DrivingSpecies:
        """Return ${}^3\\mathrm{He}$ ($Z = 2, A = 3$)."""
        return cls(2.0, 3.0)

    @classmethod
    def c12(cls) -> DrivingSpecies:
        """Return ${}^{12}\\mathrm{C}$ ($Z = 6, A = 12$)."""
        return cls(6.0, 12.0)


@dataclasses.dataclass(frozen=True)
class TurbOutput:
    """Standard output container for turbulent mixing prescriptions.

    Attributes:
        gamma: Convective efficiency $\\Gamma$ (dimensionless).
        grad_t: Actual temperature gradient $\\nabla_T$ (dimensionless).
        y_face: Superadiabaticity $Y_{\\mathrm{face}}$ (dimensionless).
        conv_vel: Convective or effective mixing speed in
            $\\mathrm{cm\\,s^{-1}}$.
        d: Compositional diffusion coefficient $D$ in $\\mathrm{cm^2\\,s^{-1}}$.
        mixing_type: Active `MixingType` classification.
    """

    gamma: ScalarOrDual
    grad_t: ScalarOrDual
    y_face: ScalarOrDual
    conv_vel: ScalarOrDual
    d: ScalarOrDual
    mixing_type: _const.MixingType


@dataclasses.dataclass(frozen=True)
class MltParams:
    """Non-differentiable configuration parameters for `set_mlt`.

    Attributes:
        option: Geometric MLT prescription (`MltOption`).
        mixing_length_alpha: Mixing-length parameter $\\alpha_{\\mathrm{MLT}}$.
        henyey_nu: Henyey velocity parameter $\\nu$.
        henyey_y: Henyey optical-depth parameter $y$.
        max_conv_vel: Maximum convective speed in $\\mathrm{cm\\,s^{-1}}$.
    """

    option: MltOption = MltOption.COX
    mixing_length_alpha: float = 2.0
    henyey_nu: float = 8.0
    henyey_y: float = _const.ONE_THIRD
    max_conv_vel: float = 1.0e99

    @classmethod
    def cox(cls, mixing_length_alpha: float) -> MltParams:
        """Create standard Cox MLT parameters for $\\alpha_{\\mathrm{MLT}}$."""
        return cls(
            option=MltOption.COX,
            mixing_length_alpha=mixing_length_alpha,
        )


@dataclasses.dataclass(frozen=True)
class MltInputs:
    """Differentiable thermodynamic and structure inputs for `set_mlt`.

    Attributes:
        chi_t: $(\\partial \\ln P / \\partial \\ln T)_\\rho$ (dimensionless).
        chi_rho: $(\\partial \\ln P / \\partial \\ln \\rho)_T$ (dimensionless).
        cp: Specific heat $C_P$ in $\\mathrm{erg\\,g^{-1}\\,K^{-1}}$.
        grav: Local gravitational acceleration $g$ in $\\mathrm{cm\\,s^{-2}}$.
        lambda_: Mixing length $\\Lambda$ in $\\mathrm{cm}$.
        rho: Mass density $\\rho$ in $\\mathrm{g\\,cm^{-3}}$.
        p: Total pressure $P$ in $\\mathrm{dyn\\,cm^{-2}}$.
        t: Temperature $T$ in $\\mathrm{K}$.
        opacity: Rosseland mean opacity $\\kappa$ in $\\mathrm{cm^2\\,g^{-1}}$.
        gradr: Radiative temperature gradient $\\nabla_r$ (dimensionless).
        grada: Adiabatic temperature gradient $\\nabla_{\\mathrm{ad}}$.
        gradl: Ledoux temperature gradient $\\nabla_L$ (dimensionless).
    """

    chi_t: ScalarOrDual
    chi_rho: ScalarOrDual
    cp: ScalarOrDual
    grav: ScalarOrDual
    lambda_: ScalarOrDual
    rho: ScalarOrDual
    p: ScalarOrDual
    t: ScalarOrDual
    opacity: ScalarOrDual
    gradr: ScalarOrDual
    grada: ScalarOrDual
    gradl: ScalarOrDual


def val_of(x: TdcScalar) -> float:
    """Extract primal `float` value from a scalar or dual input."""
    if isinstance(x, _tdc_dual.AutoDiffRealTdc):
        return x.val
    if isinstance(x, _ad.DualOrder1):
        return float(x.val)
    return float(x)


@typing.overload
def const_like(proto: float, v: float) -> float: ...


@typing.overload
def const_like(proto: _ad.DualOrder1, v: float) -> _ad.DualOrder1: ...


@typing.overload
def const_like(
    proto: _tdc_dual.AutoDiffRealTdc, v: float
) -> _tdc_dual.AutoDiffRealTdc: ...


@typing.overload
def const_like(proto: ScalarOrDual, v: float) -> ScalarOrDual: ...


@typing.overload
def const_like(proto: TdcScalar, v: float) -> TdcScalar: ...


def const_like(proto: TdcScalar, v: float) -> TdcScalar:
    """Construct a constant matching the type/arity of `proto`."""
    if isinstance(proto, _tdc_dual.AutoDiffRealTdc):
        return _tdc_dual.AutoDiffRealTdc.constant(v, proto.num_vars)
    if isinstance(proto, _ad.DualOrder1):
        return _ad.DualOrder1.constant(v, num_vars=int(proto.d1.shape[-1]))
    return float(v)


@typing.overload
def with_val_like(x: float, new_val: float) -> float: ...


@typing.overload
def with_val_like(x: _ad.DualOrder1, new_val: float) -> _ad.DualOrder1: ...


@typing.overload
def with_val_like(
    x: _tdc_dual.AutoDiffRealTdc, new_val: float
) -> _tdc_dual.AutoDiffRealTdc: ...


@typing.overload
def with_val_like(x: ScalarOrDual, new_val: float) -> ScalarOrDual: ...


@typing.overload
def with_val_like(x: TdcScalar, new_val: float) -> TdcScalar: ...


def with_val_like(x: TdcScalar, new_val: float) -> TdcScalar:
    """Replace the primal value of `x` while preserving its derivatives."""
    if isinstance(x, _tdc_dual.AutoDiffRealTdc):
        return x.with_val(new_val)
    if isinstance(x, _ad.DualOrder1):
        return _ad.DualOrder1(val=float(new_val), d1=x.d1)
    return float(new_val)


@typing.overload
def s_abs(x: float) -> float: ...


@typing.overload
def s_abs(x: _ad.DualOrder1) -> _ad.DualOrder1: ...


@typing.overload
def s_abs(x: _tdc_dual.AutoDiffRealTdc) -> _tdc_dual.AutoDiffRealTdc: ...


@typing.overload
def s_abs(x: ScalarOrDual) -> ScalarOrDual: ...


@typing.overload
def s_abs(x: TdcScalar) -> TdcScalar: ...


def s_abs(x: TdcScalar) -> TdcScalar:
    """Evaluate $|x|$ for `float`, `DualOrder1`, or `AutoDiffRealTdc`."""
    if isinstance(x, (_ad.DualOrder1, _tdc_dual.AutoDiffRealTdc)):
        return x.abs()
    return abs(x)


@typing.overload
def s_sqrt(x: float) -> float: ...


@typing.overload
def s_sqrt(x: _ad.DualOrder1) -> _ad.DualOrder1: ...


@typing.overload
def s_sqrt(x: _tdc_dual.AutoDiffRealTdc) -> _tdc_dual.AutoDiffRealTdc: ...


@typing.overload
def s_sqrt(x: ScalarOrDual) -> ScalarOrDual: ...


@typing.overload
def s_sqrt(x: TdcScalar) -> TdcScalar: ...


def s_sqrt(x: TdcScalar) -> TdcScalar:
    """Evaluate $\\sqrt{x}$ for `float`, `DualOrder1`, or `AutoDiffRealTdc`."""
    if isinstance(x, (_ad.DualOrder1, _tdc_dual.AutoDiffRealTdc)):
        return x.sqrt()
    return math.sqrt(x)


@typing.overload
def s_exp(x: float) -> float: ...


@typing.overload
def s_exp(x: _ad.DualOrder1) -> _ad.DualOrder1: ...


@typing.overload
def s_exp(x: _tdc_dual.AutoDiffRealTdc) -> _tdc_dual.AutoDiffRealTdc: ...


@typing.overload
def s_exp(x: ScalarOrDual) -> ScalarOrDual: ...


@typing.overload
def s_exp(x: TdcScalar) -> TdcScalar: ...


def s_exp(x: TdcScalar) -> TdcScalar:
    """Evaluate $\\exp(x)$ for `float`, `DualOrder1`, or `AutoDiffRealTdc`."""
    if isinstance(x, (_ad.DualOrder1, _tdc_dual.AutoDiffRealTdc)):
        return x.exp()
    return math.exp(x)


@typing.overload
def s_log(x: float) -> float: ...


@typing.overload
def s_log(x: _ad.DualOrder1) -> _ad.DualOrder1: ...


@typing.overload
def s_log(x: _tdc_dual.AutoDiffRealTdc) -> _tdc_dual.AutoDiffRealTdc: ...


@typing.overload
def s_log(x: ScalarOrDual) -> ScalarOrDual: ...


@typing.overload
def s_log(x: TdcScalar) -> TdcScalar: ...


def s_log(x: TdcScalar) -> TdcScalar:
    """Evaluate $\\ln(x)$ for `float`, `DualOrder1`, or `AutoDiffRealTdc`."""
    if isinstance(x, (_ad.DualOrder1, _tdc_dual.AutoDiffRealTdc)):
        return x.log()
    return math.log(x)


@typing.overload
def s_powf(x: float, p: float) -> float: ...


@typing.overload
def s_powf(x: _ad.DualOrder1, p: float) -> _ad.DualOrder1: ...


@typing.overload
def s_powf(
    x: _tdc_dual.AutoDiffRealTdc, p: float
) -> _tdc_dual.AutoDiffRealTdc: ...


@typing.overload
def s_powf(x: ScalarOrDual, p: float) -> ScalarOrDual: ...


@typing.overload
def s_powf(x: TdcScalar, p: float) -> TdcScalar: ...


def s_powf(x: TdcScalar, p: float) -> TdcScalar:
    """Evaluate $x^p$ for `float`, `DualOrder1`, or `AutoDiffRealTdc`."""
    if isinstance(x, (_ad.DualOrder1, _tdc_dual.AutoDiffRealTdc)):
        return x.powf(p)
    return math.pow(x, p)


@typing.overload
def s_pow2(x: float) -> float: ...


@typing.overload
def s_pow2(x: _ad.DualOrder1) -> _ad.DualOrder1: ...


@typing.overload
def s_pow2(x: _tdc_dual.AutoDiffRealTdc) -> _tdc_dual.AutoDiffRealTdc: ...


@typing.overload
def s_pow2(x: ScalarOrDual) -> ScalarOrDual: ...


@typing.overload
def s_pow2(x: TdcScalar) -> TdcScalar: ...


def s_pow2(x: TdcScalar) -> TdcScalar:
    """Evaluate $x^2$ for `float`, `DualOrder1`, or `AutoDiffRealTdc`."""
    if isinstance(x, (_ad.DualOrder1, _tdc_dual.AutoDiffRealTdc)):
        return x.pow2()
    return x * x


@typing.overload
def s_pow3(x: float) -> float: ...


@typing.overload
def s_pow3(x: _ad.DualOrder1) -> _ad.DualOrder1: ...


@typing.overload
def s_pow3(x: _tdc_dual.AutoDiffRealTdc) -> _tdc_dual.AutoDiffRealTdc: ...


@typing.overload
def s_pow3(x: ScalarOrDual) -> ScalarOrDual: ...


@typing.overload
def s_pow3(x: TdcScalar) -> TdcScalar: ...


def s_pow3(x: TdcScalar) -> TdcScalar:
    """Evaluate $x^3$ for `float`, `DualOrder1`, or `AutoDiffRealTdc`."""
    if isinstance(x, (_ad.DualOrder1, _tdc_dual.AutoDiffRealTdc)):
        return x.pow3()
    return x * x * x


@typing.overload
def s_pow4(x: float) -> float: ...


@typing.overload
def s_pow4(x: _ad.DualOrder1) -> _ad.DualOrder1: ...


@typing.overload
def s_pow4(x: _tdc_dual.AutoDiffRealTdc) -> _tdc_dual.AutoDiffRealTdc: ...


@typing.overload
def s_pow4(x: ScalarOrDual) -> ScalarOrDual: ...


@typing.overload
def s_pow4(x: TdcScalar) -> TdcScalar: ...


def s_pow4(x: TdcScalar) -> TdcScalar:
    """Evaluate $x^4$ for `float`, `DualOrder1`, or `AutoDiffRealTdc`."""
    if isinstance(x, (_ad.DualOrder1, _tdc_dual.AutoDiffRealTdc)):
        return x.pow4()
    x2 = x * x
    return x2 * x2


@typing.overload
def s_tanh(x: float) -> float: ...


@typing.overload
def s_tanh(x: _ad.DualOrder1) -> _ad.DualOrder1: ...


@typing.overload
def s_tanh(x: _tdc_dual.AutoDiffRealTdc) -> _tdc_dual.AutoDiffRealTdc: ...


@typing.overload
def s_tanh(x: ScalarOrDual) -> ScalarOrDual: ...


@typing.overload
def s_tanh(x: TdcScalar) -> TdcScalar: ...


def s_tanh(x: TdcScalar) -> TdcScalar:
    """Evaluate $\\tanh(x)$ for `float`, `DualOrder1`, or `AutoDiffRealTdc`."""
    if isinstance(x, (_ad.DualOrder1, _tdc_dual.AutoDiffRealTdc)):
        return x.tanh()
    return math.tanh(x)


@typing.overload
def s_tan(x: float) -> float: ...


@typing.overload
def s_tan(x: _ad.DualOrder1) -> _ad.DualOrder1: ...


@typing.overload
def s_tan(x: _tdc_dual.AutoDiffRealTdc) -> _tdc_dual.AutoDiffRealTdc: ...


@typing.overload
def s_tan(x: ScalarOrDual) -> ScalarOrDual: ...


@typing.overload
def s_tan(x: TdcScalar) -> TdcScalar: ...


def s_tan(x: TdcScalar) -> TdcScalar:
    """Evaluate $\\tan(x)$ for `float`, `DualOrder1`, or `AutoDiffRealTdc`."""
    if isinstance(x, (_ad.DualOrder1, _tdc_dual.AutoDiffRealTdc)):
        return x.tan()
    return math.tan(x)


@typing.overload
def s_atan(x: float) -> float: ...


@typing.overload
def s_atan(x: _ad.DualOrder1) -> _ad.DualOrder1: ...


@typing.overload
def s_atan(x: _tdc_dual.AutoDiffRealTdc) -> _tdc_dual.AutoDiffRealTdc: ...


@typing.overload
def s_atan(x: ScalarOrDual) -> ScalarOrDual: ...


@typing.overload
def s_atan(x: TdcScalar) -> TdcScalar: ...


def s_atan(x: TdcScalar) -> TdcScalar:
    """Evaluate $\\arctan(x)$ for scalar or dual inputs."""
    if isinstance(x, (_ad.DualOrder1, _tdc_dual.AutoDiffRealTdc)):
        return x.atan()
    return math.atan(x)
