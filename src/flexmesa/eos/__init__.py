# SPDX-License-Identifier: LGPL-3.0-or-later
"""Equation of State (`eos`) module for FlexMESA (Wave 3a).

Exposes the canonical 26-slot thermodynamic state vector (`eos_def`),
$C^2$-continuous polygonal quintic blending (`eos_blend`), Aparicio (1998)
20-point Gauss-Legendre/Gauss-Laguerre relativistic Fermi-Dirac integrals
(`gauss_fermi`), ground-state ionization energy offsets (`ion_offset`),
analytical neutral ideal gas plus radiation (`ideal`), Helmholtz biquintic /
bicubic electron-positron EOS (`helm`), Potekhin-Chabrier (`pc`) and Jermyn et
al. (2021) `Skye` (`skye`) plasma/crystallization EOS, and the unified
`eosDT_get` blend and `PT`/`DE` inverse solvers (`solver`).
"""

from __future__ import annotations

from flexmesa.eos import core, fermi, helm, ideal, pc, skye, solver

EOS_BASIC_RESULT_NAMES = core.EOS_BASIC_RESULT_NAMES
EosEvalOutput = core.EosEvalOutput
EosInverseOutput = solver.EosInverseOutput
EosPtOutput = solver.EosPtOutput
I_CHI_RHO = core.I_CHI_RHO
I_CHI_T = core.I_CHI_T
I_CP = core.I_CP
I_CV = core.I_CV
I_DE_DRHO = core.I_DE_DRHO
I_DS_DRHO = core.I_DS_DRHO
I_DS_DT = core.I_DS_DT
I_ETA = core.I_ETA
I_FRAC_CMS = core.I_FRAC_CMS
I_FRAC_FREE_EOS = core.I_FRAC_FREE_EOS
I_FRAC_HELM = core.I_FRAC_HELM
I_FRAC_IDEAL = core.I_FRAC_IDEAL
I_FRAC_OPAL_SCVH = core.I_FRAC_OPAL_SCVH
I_FRAC_PC = core.I_FRAC_PC
I_FRAC_SKYE = core.I_FRAC_SKYE
I_GAMMA1 = core.I_GAMMA1
I_GAMMA3 = core.I_GAMMA3
I_GRAD_AD = core.I_GRAD_AD
I_LATENT_DDLNRHO = core.I_LATENT_DDLNRHO
I_LATENT_DDLNT = core.I_LATENT_DDLNT
I_LN_E = core.I_LN_E
I_LN_FREE_E = core.I_LN_FREE_E
I_LN_PGAS = core.I_LN_PGAS
I_LN_S = core.I_LN_S
I_MU = core.I_MU
I_PHASE = core.I_PHASE
NIST_IONIZATION_TABLE_EV = core.NIST_IONIZATION_TABLE_EV
NUM_EOS_BASIC_RESULTS = core.NUM_EOS_BASIC_RESULTS
NUM_EOS_D_DXA_RESULTS = core.NUM_EOS_D_DXA_RESULTS
NUM_EOS_FRAC_RESULTS = core.NUM_EOS_FRAC_RESULTS
blend_eos_results = core.blend_eos_results
compute_ion_offset = core.compute_ion_offset
compute_ion_offset_from_za = core.compute_ion_offset_from_za
dfermi = fermi.dfermi
dfermi_jax = fermi.dfermi_jax
eos_dt_get = solver.eos_dt_get
eos_dt_get_rho = solver.eos_dt_get_rho
eos_dt_get_t = solver.eos_dt_get_t
eos_gamma_de_get = solver.eos_gamma_de_get
eos_pt_get = solver.eos_pt_get
eval_helm_cell_jax = helm.eval_helm_cell_jax
eval_helm_eos = helm.eval_helm_eos
eval_helm_eos_zones = helm.eval_helm_eos_zones
eval_ideal_eos = ideal.eval_ideal_eos
eval_ideal_eos_from_weights = ideal.eval_ideal_eos_from_weights
eval_ideal_eos_zones = ideal.eval_ideal_eos_zones
eval_pc_eos = pc.eval_pc_eos
eval_pc_eos_zones = pc.eval_pc_eos_zones
eval_skye_eos = skye.eval_skye_eos
get_pc_alfa = pc.get_pc_alfa
get_skye_alfa = skye.get_skye_alfa
get_skye_alfa_simple = skye.get_skye_alfa_simple
is_contained = core.is_contained
min_distance_to_polygon = core.min_distance_to_polygon
min_distance_to_polygon_jax = core.min_distance_to_polygon_jax
quintic_blend = core.quintic_blend
quintic_blend_jax = core.quintic_blend_jax

__all__: list[str] = [
    "EOS_BASIC_RESULT_NAMES",
    "I_CHI_RHO",
    "I_CHI_T",
    "I_CP",
    "I_CV",
    "I_DE_DRHO",
    "I_DS_DRHO",
    "I_DS_DT",
    "I_ETA",
    "I_FRAC_CMS",
    "I_FRAC_FREE_EOS",
    "I_FRAC_HELM",
    "I_FRAC_IDEAL",
    "I_FRAC_OPAL_SCVH",
    "I_FRAC_PC",
    "I_FRAC_SKYE",
    "I_GAMMA1",
    "I_GAMMA3",
    "I_GRAD_AD",
    "I_LATENT_DDLNRHO",
    "I_LATENT_DDLNT",
    "I_LN_E",
    "I_LN_FREE_E",
    "I_LN_PGAS",
    "I_LN_S",
    "I_MU",
    "I_PHASE",
    "NIST_IONIZATION_TABLE_EV",
    "NUM_EOS_BASIC_RESULTS",
    "NUM_EOS_D_DXA_RESULTS",
    "NUM_EOS_FRAC_RESULTS",
    "EosEvalOutput",
    "EosInverseOutput",
    "EosPtOutput",
    "blend_eos_results",
    "compute_ion_offset",
    "compute_ion_offset_from_za",
    "core",
    "dfermi",
    "dfermi_jax",
    "eos_dt_get",
    "eos_dt_get_rho",
    "eos_dt_get_t",
    "eos_gamma_de_get",
    "eos_pt_get",
    "eval_helm_cell_jax",
    "eval_helm_eos",
    "eval_helm_eos_zones",
    "eval_ideal_eos",
    "eval_ideal_eos_from_weights",
    "eval_ideal_eos_zones",
    "eval_pc_eos",
    "eval_pc_eos_zones",
    "eval_skye_eos",
    "fermi",
    "get_pc_alfa",
    "get_skye_alfa",
    "get_skye_alfa_simple",
    "helm",
    "ideal",
    "is_contained",
    "min_distance_to_polygon",
    "min_distance_to_polygon_jax",
    "pc",
    "quintic_blend",
    "quintic_blend_jax",
    "skye",
    "solver",
]
