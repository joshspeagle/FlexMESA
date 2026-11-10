# SPDX-License-Identifier: LGPL-3.0-or-later
"""Public interface for `flexmesa.neu` (`neu` thermal neutrino loss module).

Re-exports the top-level driver `neu_get`, JAX `float64` kernel
`neu_pair_plas_jax`, configuration/output dataclasses (`NeuChannel`, `NeuFlags`,
`NeuInputs`, `NeuRate`, `NeuResult`), individual channel routines (`pair_neu`,
`plas_neu`, `phot_neu`, `brem_neu`, `reco_neu`), and Antia (1993) Fermi-Dirac
rational approximations (`ifermi12`, `zfermim12`).

Index and Memory-Order Contract:
    - 0-based `NeuChannel` (`PAIR=0 .. RECO=4`) maps upstream 1-based channel
      indices `1..5`.
    - `sources` has row-major shape `(5, 5)` indexed by `[channel, deriv]`.
"""

from __future__ import annotations

from flexmesa.neu import brem, driver, inputs, pair_plas, phot, reco

LOG10_TMIN_NEU = inputs.LOG10_TMIN_NEU
TMIN_NEU = inputs.TMIN_NEU
NeuRate = inputs.NeuRate
NeuInputs = inputs.NeuInputs

NeuChannel = driver.NeuChannel
NeuFlags = driver.NeuFlags
NeuResult = driver.NeuResult
neu_get = driver.neu_get
neu_get_jax = driver.neu_get_jax
neu_channels_jax = driver.neu_channels_jax
neu_pair_plas_jax = driver.neu_pair_plas_jax
ifermi12_jax = driver.ifermi12_jax
zfermim12_jax = driver.zfermim12_jax

pair_neu = pair_plas.pair_neu
plas_neu = pair_plas.plas_neu
phot_neu = phot.phot_neu
brem_neu = brem.brem_neu
reco_neu = reco.reco_neu
ifermi12 = reco.ifermi12
zfermim12 = reco.zfermim12

__all__ = [
    "LOG10_TMIN_NEU",
    "TMIN_NEU",
    "NeuChannel",
    "NeuFlags",
    "NeuInputs",
    "NeuRate",
    "NeuResult",
    "brem_neu",
    "ifermi12",
    "ifermi12_jax",
    "neu_channels_jax",
    "neu_get",
    "neu_get_jax",
    "neu_pair_plas_jax",
    "pair_neu",
    "phot_neu",
    "plas_neu",
    "reco_neu",
    "zfermim12",
    "zfermim12_jax",
]
