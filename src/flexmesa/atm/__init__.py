# SPDX-License-Identifier: LGPL-3.0-or-later
"""Surface boundary conditions and model atmosphere evaluators (`atm`)."""

from __future__ import annotations

from flexmesa.atm import _core, _jax

ATM_TABLE_DB_WD_TAU_25 = _core.ATM_TABLE_DB_WD_TAU_25
ATM_TABLE_PHOTOSPHERE = _core.ATM_TABLE_PHOTOSPHERE
ATM_TABLE_TAU_1 = _core.ATM_TABLE_TAU_1
ATM_TABLE_TAU_10 = _core.ATM_TABLE_TAU_10
ATM_TABLE_TAU_100 = _core.ATM_TABLE_TAU_100
ATM_TABLE_TAU_1M1 = _core.ATM_TABLE_TAU_1M1
ATM_TABLE_WD_TAU_25 = _core.ATM_TABLE_WD_TAU_25
ATM_T_TAU_EDDINGTON = _core.ATM_T_TAU_EDDINGTON
ATM_T_TAU_KRISHNA_SWAMY = _core.ATM_T_TAU_KRISHNA_SWAMY
ATM_T_TAU_SOLAR_HOPF = _core.ATM_T_TAU_SOLAR_HOPF
ATM_T_TAU_TRAMPEDACH_SOLAR = _core.ATM_T_TAU_TRAMPEDACH_SOLAR
AtmEvalOutput = _core.AtmEvalOutput
AtmIrradiatedOutput = _core.AtmIrradiatedOutput
eval_e2 = _core.eval_e2
eval_irradiated = _core.eval_irradiated
eval_paczynski_gradr = _core.eval_paczynski_gradr
eval_paczynski_gradr_jax = _jax.eval_paczynski_gradr_jax
eval_t_tau = _core.eval_t_tau
eval_t_tau_dq_dtau = _core.eval_t_tau_dq_dtau
eval_t_tau_jax = _jax.eval_t_tau_jax
eval_t_tau_uniform = _core.eval_t_tau_uniform
eval_t_tau_varying = _core.eval_t_tau_varying
eval_table = _core.eval_table
eval_table_jax = _jax.eval_table_jax
get_t_tau_base = _core.get_t_tau_base
get_table_alfa_beta = _core.get_table_alfa_beta
get_table_base = _core.get_table_base

__all__: list[str] = [
    "ATM_TABLE_DB_WD_TAU_25",
    "ATM_TABLE_PHOTOSPHERE",
    "ATM_TABLE_TAU_1",
    "ATM_TABLE_TAU_1M1",
    "ATM_TABLE_TAU_10",
    "ATM_TABLE_TAU_100",
    "ATM_TABLE_WD_TAU_25",
    "ATM_T_TAU_EDDINGTON",
    "ATM_T_TAU_KRISHNA_SWAMY",
    "ATM_T_TAU_SOLAR_HOPF",
    "ATM_T_TAU_TRAMPEDACH_SOLAR",
    "AtmEvalOutput",
    "AtmIrradiatedOutput",
    "eval_e2",
    "eval_irradiated",
    "eval_paczynski_gradr",
    "eval_paczynski_gradr_jax",
    "eval_t_tau",
    "eval_t_tau_dq_dtau",
    "eval_t_tau_jax",
    "eval_t_tau_uniform",
    "eval_t_tau_varying",
    "eval_table",
    "eval_table_jax",
    "get_t_tau_base",
    "get_table_alfa_beta",
    "get_table_base",
]
