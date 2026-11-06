# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Ionic charge tables, 4D partial pressures, and diffusion (`ionization`).

Provides the Three-Tier Hybrid implementation of MESA's `ionization` module
(`ionization_def.f90`, `ionization_lib.f90`, `mod_ionization.f90`,
`ion_tables_load.f90`, `ion_tables_eval.f90`) and `mod_typical_charge.f90`:
- **Fe56 in He4 (`fe56_he4`)**: 2D not-a-knot bicubic spline over the
  $105 \times 30$ Thomas-Fermi-Dirac-Hubbard $(\log_{10} n_e, \log_{10} T)$
  grid (`eval_charge_of_fe56_in_he4`, `eval_charge_of_fe56_in_he4_jax`,
  `eval_charge_of_fe56_in_he4_zones`).
- **4D Ionization Tables (`tables`)**: 3-stage $(Z, X, \log Q, \log T)$
  interpolator over 28 output quantities (`eval_ionization`,
  `eval_ionization_jax`, `eval_ionization_zones`, `IonizationResult`).
- **Diffusion Typical Charge (`typical_charge`)**: Paquette et al. (1986) +
  Dupuis et al. (1992) continuum depression (`chi_info`, `chi_info_jax`,
  `chi_effective`, `chi_effective_jax`) and Allen (1973) whole-number ionic
  charges (`eval_typical_charge`, `ionization_potential`).
"""

from __future__ import annotations

from flexmesa.ionization import fe56_he4, tables, typical_charge

ION_RESULT_NAMES = tables.ION_RESULT_NAMES
ION_XS = tables.ION_XS
ION_ZS = tables.ION_ZS
NUM_ION_VALS = tables.NUM_ION_VALS
IonizationResult = tables.IonizationResult
chi_effective = typical_charge.chi_effective
chi_effective_jax = typical_charge.chi_effective_jax
chi_info = typical_charge.chi_info
chi_info_jax = typical_charge.chi_info_jax
eval_charge_of_fe56_in_he4 = fe56_he4.eval_charge_of_fe56_in_he4
eval_charge_of_fe56_in_he4_jax = fe56_he4.eval_charge_of_fe56_in_he4_jax
eval_charge_of_fe56_in_he4_zones = fe56_he4.eval_charge_of_fe56_in_he4_zones
eval_ionization = tables.eval_ionization
eval_ionization_jax = tables.eval_ionization_jax
eval_ionization_zones = tables.eval_ionization_zones
eval_typical_charge = typical_charge.eval_typical_charge
ionization_potential = typical_charge.ionization_potential

__all__: list[str] = [
    "ION_RESULT_NAMES",
    "ION_XS",
    "ION_ZS",
    "NUM_ION_VALS",
    "IonizationResult",
    "chi_effective",
    "chi_effective_jax",
    "chi_info",
    "chi_info_jax",
    "eval_charge_of_fe56_in_he4",
    "eval_charge_of_fe56_in_he4_jax",
    "eval_charge_of_fe56_in_he4_zones",
    "eval_ionization",
    "eval_ionization_jax",
    "eval_ionization_zones",
    "eval_typical_charge",
    "fe56_he4",
    "ionization_potential",
    "tables",
    "typical_charge",
]
