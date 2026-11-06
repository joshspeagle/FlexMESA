# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Element tables, nuclide catalogs, solar abundance scales, and reaction Q.

Exposes the canonical Safe Rust (`flexmesa-chem`) catalogs and energetics via
`flexmesa._rust`, matching upstream MESA `chem_def.f90` and `chem_lib.f90`.

Index and Memory-Order Contract:
    - Element IDs (`e_h = 1 .. e_cn = 112`) and nuclide IDs (`chem_id >= 1`)
      preserve upstream 1-based MESA integer identifiers (`0` for `neut`).
    - Solar metal mass-fraction arrays (`solar_element_zfrac`) have shape
      `(112,)` in `float64`, where 0-based index `z - 1` corresponds to atomic
      number `z` ($Z \in [1, 112]$).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from flexmesa import _rust

MAX_EL_Z: int = 112
NUM_CHEM_ELEMENTS: int = 112
NUM_CATEGORIES: int = 24
NPART: int = 24
SOLSIZ: int = 286


@dataclasses.dataclass(frozen=True)
class NuclideInfo:
    r"""Properties of a single nuclear species in the built-in catalog.

    Attributes:
        chem_id: 1-based MESA nuclide identifier in `isotopes.data`.
        name: Trimmed isotope symbol (such as ``"o16"`` or ``"fe56"``).
        z: Proton number $Z_i$ (dimensionless integer).
        n: Neutron number $N_i$ (dimensionless integer).
        spin: Ground-state nuclear spin $J_i$ (dimensionless).
        mass_excess: Tabulated atomic mass excess $\Delta M_i$ ($\mathrm{MeV}$).
        binding_energy: Nuclear binding energy $B_i$ ($\mathrm{MeV}$).
        w: Recomputed double-precision atomic weight $W_i$ ($\mathrm{g/mol}$).
        z53: Pre-calculated Coulomb screening power $Z_i^{5/3}$.
        isomeric_state: Isomeric state index (`0` for ground state).
    """

    chem_id: int
    name: str
    z: int
    n: int
    spin: float
    mass_excess: float
    binding_energy: float
    w: float
    z53: float
    isomeric_state: int

    @property
    def a(self) -> int:
        r"""Total baryon number $A_i = Z_i + N_i$."""
        return self.z + self.n


def chem_get_element_id(name: str) -> int:
    r"""Return the 1-based atomic number $Z \in [1, 112]$ for an element symbol.

    Args:
        name: Case-insensitive element symbol (e.g. ``"h"``, ``"he"``).

    Returns:
        1-based element ID $Z \in [1, 112]$, or `-1` if `name` is not found.
    """
    res = _rust.py_chem_get_element_id(name)
    return -1 if res is None else int(res)


def _require_str(val: str | None, msg: str) -> str:
    r"""Return `val` if non-`None` or raise `ValueError(msg)`."""
    if val is None:
        raise ValueError(msg)
    return val


def element_name(z: int) -> str:
    r"""Return the short element symbol for atomic number $Z \in [0, 112]$.

    Args:
        z: Atomic number $Z \in [0, 112]$ (`0` returns ``"neut"``).

    Returns:
        Short element symbol (e.g. ``"he"`` for $Z = 2$).

    Raises:
        ValueError: If `z` is outside `[0, 112]`.
    """
    zi = int(z)
    res = None if zi < 0 else _rust.py_chem_element_name(zi)
    return _require_str(res, f"Atomic number z={z} is outside [0, 112].")


def element_long_name(z: int) -> str:
    r"""Return the full element name for atomic number $Z \in [0, 112]$.

    Args:
        z: Atomic number $Z \in [0, 112]$.

    Returns:
        Full element name (e.g. ``"helium"`` for $Z = 2$).

    Raises:
        ValueError: If `z` is outside `[0, 112]`.
    """
    zi = int(z)
    res = None if zi < 0 else _rust.py_chem_element_long_name(zi)
    return _require_str(res, f"Atomic number z={z} is outside [0, 112].")


def element_main_iso_name(z: int) -> str:
    r"""Return the dominant solar isotope name for $Z \in [1, 112]$.

    Args:
        z: Atomic number $Z \in [1, 112]$.

    Returns:
        Dominant isotope name (e.g. ``"fe56"`` for $Z = 26$).

    Raises:
        ValueError: If `z` is outside `[1, 112]`.
    """
    zi = int(z)
    res = None if zi < 1 else _rust.py_chem_element_main_iso_name(zi)
    return _require_str(res, f"Atomic number z={z} is outside [1, 112].")


def category_id(name: str) -> int:
    r"""Return the 1-based reaction category ID ($1..=24$) or `0` if unknown.

    Args:
        name: Reaction category name (e.g. ``"pp"``, ``"cno"``).

    Returns:
        1-based category index ($1..=24$), or `0` if not found.
    """
    return int(_rust.py_chem_category_id(name))


def category_name(cat_id: int) -> str:
    r"""Return the reaction category name for 1-based `cat_id` ($1..=24$).

    Args:
        cat_id: 1-based category index ($1..=24$).

    Returns:
        Category name string.

    Raises:
        ValueError: If `cat_id` is outside `[1, 24]`.
    """
    cid = int(cat_id)
    res = None if cid < 1 else _rust.py_chem_category_name(cid)
    return _require_str(res, f"Category id {cat_id} is outside [1, 24].")


def get_partition_fcn_indx(t9: float) -> int:
    r"""Return the 1-based partition function temperature bracket index.

    Args:
        t9: Temperature $T_9 = T / 10^9\,\mathrm{K}$ in $\mathrm{GK}$.

    Returns:
        1-based lower bracket index in `0..=24` on `T_PART`.
    """
    return int(_rust.py_chem_get_partition_fcn_indx(float(t9)))


def generate_nuclide_names(z: Sequence[int], a: Sequence[int]) -> list[str]:
    r"""Generate short nuclide names from parallel $(Z, A)$ sequences.

    Args:
        z: Sequence of proton numbers $Z_i$.
        a: Sequence of baryon numbers $A_i$.

    Returns:
        List of short nuclide identifiers (e.g. ``["c12", "fe56"]``).
    """
    z_i, a_i = [int(v) for v in z], [int(v) for v in a]
    return list(_rust.py_chem_generate_nuclide_names(z_i, a_i))


def generate_long_nuclide_names(
    z: Sequence[int], a: Sequence[int]
) -> list[str]:
    r"""Generate long nuclide names from parallel $(Z, A)$ sequences.

    Args:
        z: Sequence of proton numbers $Z_i$.
        a: Sequence of baryon numbers $A_i$.

    Returns:
        List of long nuclide names (e.g. ``["carbon-12", "iron-56"]``).
    """
    z_i, a_i = [int(v) for v in z], [int(v) for v in a]
    return list(_rust.py_chem_generate_long_nuclide_names(z_i, a_i))


def get_nuclide_index(name: str) -> int:
    r"""Return the 1-based `chem_id` for a nuclide name (`get_nuclide_index`).

    Args:
        name: Nuclide name (such as ``"h1"``, ``"c12"``, ``"o16"``, ``"fe56"``).

    Returns:
        1-based MESA `chem_id` in `isotopes.data`, or `-1` if not found.
    """
    res = _rust.py_chem_get_nuclide_index(name)
    return -1 if res is None else int(res)


def _resolve_chem_id(iso: int | str) -> int:
    r"""Resolve a 1-based `chem_id` int or isotope name to `chem_id >= 1`."""
    cid = get_nuclide_index(iso) if isinstance(iso, str) else int(iso)
    if cid < 1:
        raise ValueError(f"Unknown or invalid nuclide identifier '{iso}'.")
    return cid


def _resolve_chem_ids(chem_ids: Sequence[int | str]) -> tuple[int, ...]:
    r"""Resolve a sequence of nuclide names or 1-based IDs to a tuple of IDs."""
    if not chem_ids:
        raise ValueError("chem_ids must not be empty.")
    return tuple(_resolve_chem_id(item) for item in chem_ids)


def get_nuclide_info(iso: int | str) -> NuclideInfo:
    r"""Return the `NuclideInfo` record for a `chem_id` or isotope name.

    Args:
        iso: 1-based integer `chem_id` or string isotope name (e.g. ``"o16"``).

    Returns:
        `NuclideInfo` dataclass with atomic weight, binding energy, and spin.

    Raises:
        ValueError: If `iso` is not present in the catalog.
    """
    return NuclideInfo(*_rust.py_chem_nuclide_info(_resolve_chem_id(iso)))


def ag89_zsol() -> float:
    r"""Return the Anders & Grevesse (1989) bulk solar metallicity `zsol`.

    Returns:
        Solar metallicity $Z_\odot = 1.8987109631891819 \times 10^{-2}$.
    """
    return float(_rust.py_chem_ag89_zsol())


def ag89_yesol() -> float:
    r"""Return the Anders & Grevesse (1989) solar electron fraction `yesol`.

    Returns:
        Solar electron-to-baryon ratio $Y_{e,\odot} = 0.85280190902027331$.
    """
    return float(_rust.py_chem_ag89_yesol())


def chem_xsol(name: str) -> float:
    r"""Return the Anders & Grevesse (1989) solar mass fraction for `name`.

    Args:
        name: Isotope name (``"h1"``, ``"c12"``) or alias (``"p"``, ``"d"``).

    Returns:
        Solar mass fraction $X_{\odot}(\text{name})$ or `0.0` if absent.
    """
    return float(_rust.py_chem_xsol(name))


def chem_get_solar(name: str) -> tuple[float, float, float] | None:
    r"""Look up `(Z, A, X_sol)` for `name` in the Anders & Grevesse (1989) list.

    Args:
        name: Isotope name or alias (``"p"``, ``"d"``).

    Returns:
        Tuple `(z, a, x_sol)` in `float64`, or `None` if absent.
    """
    return _rust.py_chem_get_solar(name)


def lodders03_element_atom_percent(nuclei: str) -> float:
    r"""Return the Lodders (2003, Table 6) isotopic atom percent for `nuclei`.

    Args:
        nuclei: Isotope name (such as ``"h1"``, ``"fe56"``, ``"u238"``).

    Returns:
        Atom percentage within the parent element ($[0, 100]$), or `0.0`.
    """
    return float(_rust.py_chem_lodders03_atom_percent(nuclei))


def ag89_namsol() -> list[str]:
    r"""Return the ordered list of 286 solar isotope names (`namsol(1:286)`).

    Returns:
        List of 286 trimmed isotope names from ``"h1"`` to ``"u238"``.
    """
    return list(_rust.py_chem_ag89_namsol())


def get_stable_mass_frac(
    z: Sequence[int],
    a: Sequence[int],
    abun_in: Sequence[float] | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Map unstable network abundances onto the 286 stable solar isotopes.

    Args:
        z: Sequence of proton numbers $Z_i$ of length `n_species`.
        a: Sequence of baryon numbers $A_i$ of length `n_species`.
        abun_in: Input mass fractions $X_i$ of shape `(n_species,)`.

    Returns:
        1D `float64` NumPy array of shape `(286,)` on `ag89_namsol()`.
    """
    res = _rust.py_chem_get_stable_mass_frac(
        [int(v) for v in z], [int(v) for v in a], [float(v) for v in abun_in]
    )
    return np.asarray(res, dtype=np.float64)


def solar_element_zfrac(scale: str = "AGSS09") -> npt.NDArray[np.float64]:
    r"""Return the 112-element solar metal mass-fraction array.

    Args:
        scale: Solar scale name (``"AG89"`` through ``"AAG21"``).

    Returns:
        1D `float64` NumPy array of shape `(112,)` indexed by `z - 1`.

    Raises:
        ValueError: If `scale` is not a recognized solar scale name.
    """
    raw = _rust.py_chem_solar_element_zfrac(scale)
    return np.asarray(raw, dtype=np.float64)


def solar_cno_fraction(scale: str = "AGSS09") -> float:
    r"""Return the CNO mass fraction of metallicity ($Z_{\mathrm{CNO}} / Z$).

    Args:
        scale: Solar scale identifier (e.g. ``"GS98"``, ``"AGSS09"``).

    Returns:
        Dimensionless CNO fraction of $Z$.

    Raises:
        ValueError: If `scale` is not a recognized solar scale name.
    """
    return float(_rust.py_chem_solar_cno_fraction(scale))


def chem_m_div_h(x: float, z: float, scale: str = "AGSS09") -> float:
    r"""Compute logarithmic metallicity `[M/H]` (`chem_M_div_h`).

    Args:
        x: Hydrogen mass fraction $X > 0$ (dimensionless).
        z: Metal mass fraction $Z > 0$ (dimensionless).
        scale: Solar abundance scale name.

    Returns:
        Logarithmic metallicity $[\mathrm{M/H}]$ in dex.

    Raises:
        ValueError: If `x <= 0`, `z <= 0`, or `scale` is unknown.
    """
    return float(_rust.py_chem_m_div_h(float(x), float(z), scale))


def reaction_qtotal(
    num_in: int, num_out: int, reactants: Sequence[int | str]
) -> float:
    r"""Compute total reaction $Q$-value (`reaction_Qtotal`) in $\mathrm{MeV}$.

    Args:
        num_in: Number of incoming reactant species $N_{\mathrm{in}}$.
        num_out: Number of outgoing product species $N_{\mathrm{out}}$.
        reactants: Sequence of length `num_in + num_out` of `chem_id` or names.

    Returns:
        Total energy release $Q_{\mathrm{total}}$ in $\mathrm{MeV}$.

    Raises:
        ValueError: If any nuclide name or `chem_id` is unknown.
    """
    cids = list(_resolve_chem_ids(reactants))
    return float(_rust.py_chem_reaction_qtotal(int(num_in), int(num_out), cids))
