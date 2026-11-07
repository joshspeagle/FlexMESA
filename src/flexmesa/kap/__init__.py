# SPDX-License-Identifier: LGPL-3.0-or-later
"""Radiative, conductive, and Compton opacity (`kap`) package for FlexMESA."""

from __future__ import annotations

from flexmesa.kap import core

I_FRAC_COMPTON = core.I_FRAC_COMPTON
I_FRAC_HIGH_T = core.I_FRAC_HIGH_T
I_FRAC_LOW_T = core.I_FRAC_LOW_T
I_FRAC_TYPE2 = core.I_FRAC_TYPE2
KapEvalOutput = core.KapEvalOutput
NUM_KAP_FRACS = core.NUM_KAP_FRACS
combine_rad_with_conduction_jax = core.combine_rad_with_conduction_jax
compton_opacity_jax = core.compton_opacity_jax
kap_get = core.kap_get
kap_get_aesopus = core.kap_get_aesopus
kap_get_co = core.kap_get_co
kap_get_compton_opacity = core.kap_get_compton_opacity
kap_get_elect_cond_opacity = core.kap_get_elect_cond_opacity
kap_get_fixed_metal = core.kap_get_fixed_metal
kap_get_kapcn = core.kap_get_kapcn
kap_get_results = core.kap_get_results
kap_get_zones = core.kap_get_zones
make_differentiable_kap = core.make_differentiable_kap

__all__: list[str] = [
    "I_FRAC_COMPTON",
    "I_FRAC_HIGH_T",
    "I_FRAC_LOW_T",
    "I_FRAC_TYPE2",
    "NUM_KAP_FRACS",
    "KapEvalOutput",
    "combine_rad_with_conduction_jax",
    "compton_opacity_jax",
    "core",
    "kap_get",
    "kap_get_aesopus",
    "kap_get_co",
    "kap_get_compton_opacity",
    "kap_get_elect_cond_opacity",
    "kap_get_fixed_metal",
    "kap_get_kapcn",
    "kap_get_results",
    "kap_get_zones",
    "make_differentiable_kap",
]
