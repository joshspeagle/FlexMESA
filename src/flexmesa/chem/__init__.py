# SPDX-License-Identifier: GPL-3.0-or-later
r"""Public interface for `flexmesa.chem` (`chem` chemical elements & nuclides).

Re-exports the canonical Safe Rust periodic table, Anders & Grevesse (1989) +
Lodders (2003) 286-isotope catalog, 10 solar elemental abundance scales (`AG89`
through `AAG21`), reaction $Q$-value evaluators (`reaction_qtotal`), and
composition moment kernels (`get_composition_info`, `basic_composition_info`,
`composition_info`, `get_composition_info_jax`, and
`get_composition_info_zones`) with Safe Rust as canonical (`strict` mode) and
derived `float64` JAX kernels for automatic differentiation and batching.

Architecture & Submodules:
    - `flexmesa.chem.catalog`: Static catalogs and reaction $Q$-values backed
      by Safe Rust (`Z in [0, 112]`, `isotopes.data`, `AG89`, `GN93`, `GS98`,
      `L03`, `AGS05`, `AGSS09`, `L09`, `A09_Prz`, `MB22`, `AAG21`).
    - `flexmesa.chem.composition`: Composition moments ($\bar{A}, \bar{Z},
      Y_e, z_2\bar{}, z_{5/3}\bar{}, \sum Y_i Q_i$) and analytical partial
      derivatives $\partial / \partial X_i$ in Safe Rust (canonical) and
      `@jax.jit` (derived, validated against Rust `strict` mode).
"""

from __future__ import annotations

from flexmesa.chem import catalog, composition

# ---------------------------------------------------------------------------
# Tier A: Static Element, Nuclide, Solar Scale, and Energetics Catalogs
# ---------------------------------------------------------------------------
MAX_EL_Z = catalog.MAX_EL_Z
NUM_CHEM_ELEMENTS = catalog.NUM_CHEM_ELEMENTS
NUM_CATEGORIES = catalog.NUM_CATEGORIES
NPART = catalog.NPART
SOLSIZ = catalog.SOLSIZ
NuclideInfo = catalog.NuclideInfo
chem_get_element_id = catalog.chem_get_element_id
element_name = catalog.element_name
element_long_name = catalog.element_long_name
element_main_iso_name = catalog.element_main_iso_name
category_id = catalog.category_id
category_name = catalog.category_name
get_partition_fcn_indx = catalog.get_partition_fcn_indx
generate_nuclide_names = catalog.generate_nuclide_names
generate_long_nuclide_names = catalog.generate_long_nuclide_names
get_nuclide_index = catalog.get_nuclide_index
get_nuclide_info = catalog.get_nuclide_info
ag89_zsol = catalog.ag89_zsol
ag89_yesol = catalog.ag89_yesol
chem_xsol = catalog.chem_xsol
chem_get_solar = catalog.chem_get_solar
lodders03_element_atom_percent = catalog.lodders03_element_atom_percent
ag89_namsol = catalog.ag89_namsol
get_stable_mass_frac = catalog.get_stable_mass_frac
solar_element_zfrac = catalog.solar_element_zfrac
solar_cno_fraction = catalog.solar_cno_fraction
chem_m_div_h = catalog.chem_m_div_h
reaction_qtotal = catalog.reaction_qtotal

# ---------------------------------------------------------------------------
# Tier B: Differentiable Composition Moments & Multi-Zone Kernels
# ---------------------------------------------------------------------------
CompositionInfo = composition.CompositionInfo
get_composition_info = composition.get_composition_info
basic_composition_info = composition.basic_composition_info
composition_info = composition.composition_info
get_composition_info_jax = composition.get_composition_info_jax
get_composition_info_zones = composition.get_composition_info_zones

__all__ = [
    "MAX_EL_Z",
    "NPART",
    "NUM_CATEGORIES",
    "NUM_CHEM_ELEMENTS",
    "SOLSIZ",
    "CompositionInfo",
    "NuclideInfo",
    "ag89_namsol",
    "ag89_yesol",
    "ag89_zsol",
    "basic_composition_info",
    "category_id",
    "category_name",
    "chem_get_element_id",
    "chem_get_solar",
    "chem_m_div_h",
    "chem_xsol",
    "composition_info",
    "element_long_name",
    "element_main_iso_name",
    "element_name",
    "generate_long_nuclide_names",
    "generate_nuclide_names",
    "get_composition_info",
    "get_composition_info_jax",
    "get_composition_info_zones",
    "get_nuclide_index",
    "get_nuclide_info",
    "get_partition_fcn_indx",
    "get_stable_mass_frac",
    "lodders03_element_atom_percent",
    "reaction_qtotal",
    "solar_cno_fraction",
    "solar_element_zfrac",
]
