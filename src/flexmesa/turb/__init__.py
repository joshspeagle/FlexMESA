# SPDX-License-Identifier: LGPL-3.0-or-later
"""Convective and diffusive turbulent mixing (`flexmesa.turb`).

Ports upstream MESA's `turb` module (`turb/public/turb.f90` and
`turb/private/{mlt,tdc,tdc_support,semiconvection,thermohaline}.f90`) to
Python and JAX (`float64`), supporting IEEE-754 scalars (`float`), first-order
forward-mode dual numbers (`flexmesa.auto_diff.DualOrder1`), and pure-JAX
`float64` kernels (`set_mlt_jax`, `set_semiconvection_jax`).

Provided Submodules and Prescriptions:
- `mlt`: Local Mixing-Length Theory (`Cox`, `TDC`, `Henyey`, `ML1`, `ML2`,
  `Mihalas`) solving the cubic convective efficiency equation.
- `tdc`: Kuhfuss (1986) / Smolec time-dependent convection (`set_tdc`) with
  1D logarithmic root-finding and Implicit Function Theorem derivatives.
- `semiconvection`: Langer, El Eid, & Fricke (1985) semiconvective mixing.
- `thermohaline`: Kippenhahn et al. (1980), Traxler et al. (2011), and
  Brown et al. (2013) salt-finger thermohaline mixing diffusivities.
"""

from __future__ import annotations

from flexmesa import constants as _const
from flexmesa.turb import mlt as _mlt
from flexmesa.turb import semiconvection as _sc
from flexmesa.turb import tdc as _tdc
from flexmesa.turb import thermohaline as _th
from flexmesa.turb import types as _types

DrivingSpecies = _types.DrivingSpecies
MixingType = _const.MixingType
MltInputs = _types.MltInputs
MltOption = _types.MltOption
MltParams = _types.MltParams
SemiconvectionInputs = _sc.SemiconvectionInputs
SemiconvectionOption = _types.SemiconvectionOption
SemiconvectionParams = _sc.SemiconvectionParams
TdcInputs = _tdc.TdcInputs
TdcOutput = _tdc.TdcOutput
TdcParams = _tdc.TdcParams
ThermohalineInputs = _th.ThermohalineInputs
ThermohalineOption = _types.ThermohalineOption
ThermohalineParams = _th.ThermohalineParams
TurbOutput = _types.TurbOutput

get_d_thermohaline = _th.get_d_thermohaline
set_mlt = _mlt.set_mlt
set_mlt_jax = _mlt.set_mlt_jax
set_semiconvection = _sc.set_semiconvection
set_semiconvection_jax = _sc.set_semiconvection_jax
set_tdc = _tdc.set_tdc
set_thermohaline = _th.set_thermohaline

__all__: list[str] = [
    "DrivingSpecies",
    "MixingType",
    "MltInputs",
    "MltOption",
    "MltParams",
    "SemiconvectionInputs",
    "SemiconvectionOption",
    "SemiconvectionParams",
    "TdcInputs",
    "TdcOutput",
    "TdcParams",
    "ThermohalineInputs",
    "ThermohalineOption",
    "ThermohalineParams",
    "TurbOutput",
    "get_d_thermohaline",
    "set_mlt",
    "set_mlt_jax",
    "set_semiconvection",
    "set_semiconvection_jax",
    "set_tdc",
    "set_thermohaline",
]
