# SPDX-License-Identifier: LGPL-3.0-or-later
"""Nuclear reaction rates, plasma screening, and weak rates (`rates`).

Ports upstream MESA `rates` (Wave 3b.3) across Safe Rust (`flexmesa-rates`) and
Python/JAX (`flexmesa.rates`).
"""

from __future__ import annotations

from flexmesa.rates import core, jax_kernels, screening, weak

ReactionInfo = core.ReactionInfo
ScreenEval = screening.ScreenEval
WeakRateEval = weak.WeakRateEval
eval_coulomb = weak.eval_coulomb
eval_ecapture_pair = weak.eval_ecapture_pair
eval_fl_epsnuc_3alf = core.eval_fl_epsnuc_3alf
eval_fl_epsnuc_3alf_jax = jax_kernels.eval_fl_epsnuc_3alf_jax
eval_rate_table = core.eval_rate_table
eval_reaclib_handle = core.eval_reaclib_handle
eval_reaclib_jax = jax_kernels.eval_reaclib_jax
eval_weak_pair = weak.eval_weak_pair
eval_weak_table_index = weak.eval_weak_table_index
reaction_id_by_name = core.reaction_id_by_name
reaction_info = core.reaction_info
reaction_name_by_id = core.reaction_name_by_id
screen_chugunov_jax = jax_kernels.screen_chugunov_jax
screen_extended_jax = jax_kernels.screen_extended_jax
screen_pair = screening.screen_pair
screen_pair_zones = screening.screen_pair_zones
screen_salpeter_jax = jax_kernels.screen_salpeter_jax
set_raw_rate = core.set_raw_rate
set_raw_rates = core.set_raw_rates
set_raw_rates_zones = core.set_raw_rates_zones
weak_info_by_names = weak.weak_info_by_names

__all__: list[str] = [
    "ReactionInfo",
    "ScreenEval",
    "WeakRateEval",
    "core",
    "eval_coulomb",
    "eval_ecapture_pair",
    "eval_fl_epsnuc_3alf",
    "eval_fl_epsnuc_3alf_jax",
    "eval_rate_table",
    "eval_reaclib_handle",
    "eval_reaclib_jax",
    "eval_weak_pair",
    "eval_weak_table_index",
    "jax_kernels",
    "reaction_id_by_name",
    "reaction_info",
    "reaction_name_by_id",
    "screen_chugunov_jax",
    "screen_extended_jax",
    "screen_pair",
    "screen_pair_zones",
    "screen_salpeter_jax",
    "screening",
    "set_raw_rate",
    "set_raw_rates",
    "set_raw_rates_zones",
    "weak",
    "weak_info_by_names",
]
