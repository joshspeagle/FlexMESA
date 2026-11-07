# SPDX-License-Identifier: LGPL-3.0-or-later
"""Surface boundary conditions and model atmosphere evaluators (`atm`)."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from flexmesa import _rust

ATM_T_TAU_EDDINGTON: int = 1
ATM_T_TAU_SOLAR_HOPF: int = 2
ATM_T_TAU_KRISHNA_SWAMY: int = 3
ATM_T_TAU_TRAMPEDACH_SOLAR: int = 4

ATM_TABLE_TAU_100: int = 100
ATM_TABLE_TAU_10: int = 101
ATM_TABLE_TAU_1: int = 102
ATM_TABLE_TAU_1M1: int = 103
ATM_TABLE_PHOTOSPHERE: int = 104
ATM_TABLE_WD_TAU_25: int = 105
ATM_TABLE_DB_WD_TAU_25: int = 106


@dataclasses.dataclass(frozen=True)
class AtmEvalOutput:
    """Surface boundary condition evaluation output (`AtmEvalOutput`).

    Attributes:
        ln_t: Natural logarithm of surface temperature $\\ln(T / \\mathrm{K})$.
        dln_t_dl: Partial derivative $\\partial \\ln T / \\partial L$
            ($\\mathrm{erg^{-1}\\,s}$).
        dln_t_dln_r: Partial derivative $\\partial \\ln T / \\partial \\ln R$.
        dln_t_dln_m: Partial derivative $\\partial \\ln T / \\partial \\ln M$.
        dln_t_dln_kap: Partial derivative
            $\\partial \\ln T / \\partial \\ln \\kappa$.
        ln_p: Natural logarithm of surface pressure
            $\\ln(P / \\mathrm{dyn\\,cm^{-2}})$.
        dln_p_dl: Partial derivative $\\partial \\ln P / \\partial L$
            ($\\mathrm{erg^{-1}\\,s}$).
        dln_p_dln_r: Partial derivative $\\partial \\ln P / \\partial \\ln R$.
        dln_p_dln_m: Partial derivative $\\partial \\ln P / \\partial \\ln M$.
        dln_p_dln_kap: Partial derivative
            $\\partial \\ln P / \\partial \\ln \\kappa$.
        kap: Converged surface opacity $\\kappa$ ($\\mathrm{cm^2\\,g^{-1}}$).
    """

    ln_t: float
    dln_t_dl: float
    dln_t_dln_r: float
    dln_t_dln_m: float
    dln_t_dln_kap: float
    ln_p: float
    dln_p_dl: float
    dln_p_dln_r: float
    dln_p_dln_m: float
    dln_p_dln_kap: float
    kap: float


@dataclasses.dataclass(frozen=True)
class AtmIrradiatedOutput:
    """Irradiated grey atmosphere evaluation output (`AtmIrradiatedOutput`).

    Attributes:
        kap: Converged thermal opacity $\\kappa$ ($\\mathrm{cm^2\\,g^{-1}}$).
        tau: Surface optical depth $\\tau = P_{\\mathrm{surf}}\\,\\kappa / g$.
        ln_t: Natural logarithm of surface temperature $\\ln(T / \\mathrm{K})$.
        dln_t_dl: Partial derivative $\\partial \\ln T / \\partial L$
            ($\\mathrm{erg^{-1}\\,s}$).
        dln_t_dln_r: Partial derivative $\\partial \\ln T / \\partial \\ln R$.
        dln_t_dln_m: Partial derivative $\\partial \\ln T / \\partial \\ln M$.
        dln_t_dln_kap: Partial derivative
            $\\partial \\ln T / \\partial \\ln \\kappa$.
    """

    kap: float
    tau: float
    ln_t: float
    dln_t_dl: float
    dln_t_dln_r: float
    dln_t_dln_m: float
    dln_t_dln_kap: float


def eval_t_tau(t_tau_id: int, tau: float, teff: float) -> float:
    """Evaluate $\\ln T(\\tau, T_{\\mathrm{eff}})$ for a $T(\\tau)$ relation.

    Args:
        t_tau_id: $T(\\tau)$ relation identifier (`1`..`4`).
        tau: Optical depth $\\tau$ (dimensionless).
        teff: Effective temperature $T_{\\mathrm{eff}}$ ($\\mathrm{K}$).

    Returns:
        Natural logarithm of temperature $\\ln(T / \\mathrm{K})$.
    """
    return float(_rust.atm_eval_t_tau(int(t_tau_id), float(tau), float(teff)))


def eval_t_tau_dq_dtau(t_tau_id: int, tau: float) -> float:
    """Evaluate Hopf-function derivative $dq/d\\tau$ at optical depth $\\tau$.

    Args:
        t_tau_id: $T(\\tau)$ relation identifier (`1`..`4`).
        tau: Optical depth $\\tau$ (dimensionless).

    Returns:
        Derivative $dq/d\\tau$ (dimensionless).
    """
    return float(_rust.atm_eval_t_tau_dq_dtau(int(t_tau_id), float(tau)))


def get_t_tau_base(t_tau_id: int) -> float:
    """Return base optical depth $\\tau_{\\mathrm{base}}$ of a $T(\\tau)$ law.

    Args:
        t_tau_id: $T(\\tau)$ relation identifier (`1`..`4`).

    Returns:
        Base optical depth $\\tau_{\\mathrm{base}}$ (dimensionless).
    """
    return float(_rust.atm_get_t_tau_base(int(t_tau_id)))


def eval_e2(x: float) -> tuple[float, float]:
    """Evaluate the second exponential integral $(E_2(x), dE_2/dx)$.

    Args:
        x: Non-negative optical-depth argument $x \\in [0, 10]$.

    Returns:
        Tuple `(e2, de2_dx)`.
    """
    e2, de2 = _rust.atm_eval_e2(float(x))
    return float(e2), float(de2)


def eval_paczynski_gradr(
    t: float,
    p: float,
    rho: float,
    tau: float,
    kap: float,
    luminosity: float,
    m: float,
    r: float,
    cgrav: float,
) -> float:
    """Evaluate the Paczynski (1969) diluted radiative temperature gradient.

    Args:
        t: Temperature $T$ ($\\mathrm{K}$).
        p: Total pressure $P$ ($\\mathrm{dyn\\,cm^{-2}}$).
        rho: Mass density $\\rho$ ($\\mathrm{g\\,cm^{-3}}$).
        tau: Optical depth $\\tau$ (dimensionless).
        kap: Rosseland mean opacity $\\kappa$ ($\\mathrm{cm^2\\,g^{-1}}$).
        luminosity: Luminosity $L$ ($\\mathrm{erg\\,s^{-1}}$).
        m: Enclosed mass $M$ ($\\mathrm{g}$).
        r: Radius $R$ ($\\mathrm{cm}$).
        cgrav: Gravitational constant $G$ ($\\mathrm{cm^3\\,g^{-1}\\,s^{-2}}$).

    Returns:
        Diluted radiative gradient $\\nabla_{\\mathrm{rad}}$ (dimensionless).
    """
    return float(
        _rust.atm_eval_paczynski_gradr(
            float(t),
            float(p),
            float(rho),
            float(tau),
            float(kap),
            float(luminosity),
            float(m),
            float(r),
            float(cgrav),
        )
    )


def eval_table(
    luminosity: float,
    r: float,
    m: float,
    cgrav: float,
    table_id: int,
    z: float,
    skip_partials: bool,
    teff: float,
) -> AtmEvalOutput:
    """Evaluate surface $(\\ln T, \\ln P)$ from a tabulated model atmosphere.

    Args:
        luminosity: Luminosity $L$ ($\\mathrm{erg\\,s^{-1}}$).
        r: Photospheric radius $R$ ($\\mathrm{cm}$).
        m: Stellar mass $M$ ($\\mathrm{g}$).
        cgrav: Gravitational constant $G$ ($\\mathrm{cm^3\\,g^{-1}\\,s^{-2}}$).
        table_id: Atmosphere table identifier (`100`..`106`).
        z: Metallicity mass fraction $Z$.
        skip_partials: If `True`, skip partial derivative calculation.
        teff: Effective temperature $T_{\\mathrm{eff}}$ ($\\mathrm{K}$).

    Returns:
        Populated `AtmEvalOutput` dataclass.
    """
    vals = _rust.atm_eval_table(
        float(luminosity),
        float(r),
        float(m),
        float(cgrav),
        int(table_id),
        float(z),
        bool(skip_partials),
        float(teff),
    )
    return AtmEvalOutput(*[float(v) for v in vals])


def get_table_alfa_beta(
    luminosity: float,
    teff: float,
    r: float,
    m: float,
    cgrav: float,
    table_id: int,
) -> tuple[float, float]:
    """Evaluate blending weights $(\\alpha, \\beta)$ near table boundaries.

    Args:
        luminosity: Luminosity $L$ ($\\mathrm{erg\\,s^{-1}}$).
        teff: Effective temperature $T_{\\mathrm{eff}}$ ($\\mathrm{K}$).
        r: Radius $R$ ($\\mathrm{cm}$).
        m: Mass $M$ ($\\mathrm{g}$).
        cgrav: Gravitational constant $G$ ($\\mathrm{cm^3\\,g^{-1}\\,s^{-2}}$).
        table_id: Atmosphere table identifier (`100`..`106`).

    Returns:
        Tuple `(alfa, beta)` where `alfa` weights grey $T(\\tau)$ and `beta`
        weights the atmosphere table.
    """
    a, b = _rust.atm_get_table_alfa_beta(
        float(luminosity),
        float(teff),
        float(r),
        float(m),
        float(cgrav),
        int(table_id),
    )
    return float(a), float(b)


def get_table_base(table_id: int) -> float:
    """Return base optical depth $\\tau_{\\mathrm{base}}$ of a table.

    Args:
        table_id: Atmosphere table identifier (`100`..`106`).

    Returns:
        Base optical depth $\\tau_{\\mathrm{base}}$ (dimensionless).
    """
    return float(_rust.atm_get_table_base(int(table_id)))


def eval_t_tau_uniform(
    tau_surf: float,
    luminosity: float,
    r: float,
    m: float,
    cgrav: float,
    kap_guess: float,
    pextra_factor: float,
    t_tau_id: int,
    errtol: float,
    max_iters: int,
    skip_partials: bool,
    teff: float,
    chem_id: Sequence[int] | None = None,
    xa: npt.ArrayLike | None = None,
    zbase: float = 0.02,
    low_t_option: str = "low_t_fa05_gs98",
) -> AtmEvalOutput:
    """Evaluate a grey $T(\\tau)$ atmosphere with uniform opacity.

    Args:
        tau_surf: Surface optical depth $\\tau_{\\mathrm{surf}}$.
        luminosity: Luminosity $L$ ($\\mathrm{erg\\,s^{-1}}$).
        r: Radius $R$ ($\\mathrm{cm}$).
        m: Mass $M$ ($\\mathrm{g}$).
        cgrav: Gravitational constant $G$ ($\\mathrm{cm^3\\,g^{-1}\\,s^{-2}}$).
        kap_guess: Initial opacity guess $\\kappa$ ($\\mathrm{cm^2\\,g^{-1}}$).
        pextra_factor: Radiation pressure enhancement factor.
        t_tau_id: $T(\\tau)$ relation identifier (`1`..`4`).
        errtol: Relative convergence tolerance on $\\kappa$.
        max_iters: Maximum opacity fixed-point iterations (`0` for fixed).
        skip_partials: If `True`, skip partial derivatives.
        teff: Effective temperature $T_{\\mathrm{eff}}$ ($\\mathrm{K}$).
        chem_id: Optional species catalog IDs (`[S]`).
        xa: Optional species mass fractions (`(S,)`).
        zbase: Base metallicity $Z_{\\mathrm{base}}$ for `kap`.
        low_t_option: Low-temperature opacity table selection.

    Returns:
        Populated `AtmEvalOutput` dataclass.
    """
    cid_list = [int(c) for c in chem_id] if chem_id is not None else None
    xa_arr = (
        np.ascontiguousarray(xa, dtype=np.float64) if xa is not None else None
    )
    vals = _rust.atm_eval_t_tau_uniform(
        float(tau_surf),
        float(luminosity),
        float(r),
        float(m),
        float(cgrav),
        float(kap_guess),
        float(pextra_factor),
        int(t_tau_id),
        float(errtol),
        int(max_iters),
        bool(skip_partials),
        float(teff),
        cid_list,
        xa_arr,
        float(zbase),
        str(low_t_option),
    )
    return AtmEvalOutput(*[float(v) for v in vals])


def eval_t_tau_varying(
    tau_surf: float,
    luminosity: float,
    r: float,
    m: float,
    cgrav: float,
    t_tau_id: int,
    errtol: float,
    max_steps: int,
    skip_partials: bool,
    teff: float,
    chem_id: Sequence[int],
    xa: npt.ArrayLike,
    zbase: float = 0.02,
    low_t_option: str = "low_t_fa05_gs98",
) -> AtmEvalOutput:
    """Evaluate a $T(\\tau)$ atmosphere integrating depth-varying opacity.

    Args:
        tau_surf: Surface optical depth $\\tau_{\\mathrm{surf}}$.
        luminosity: Luminosity $L$ ($\\mathrm{erg\\,s^{-1}}$).
        r: Radius $R$ ($\\mathrm{cm}$).
        m: Mass $M$ ($\\mathrm{g}$).
        cgrav: Gravitational constant $G$ ($\\mathrm{cm^3\\,g^{-1}\\,s^{-2}}$).
        t_tau_id: $T(\\tau)$ relation identifier (`1`..`4`).
        errtol: Relative tolerance for `dopri5` integration.
        max_steps: Maximum `dopri5` steps.
        skip_partials: If `True`, skip partial derivatives.
        teff: Effective temperature $T_{\\mathrm{eff}}$ ($\\mathrm{K}$).
        chem_id: Species catalog IDs (`[S]`).
        xa: Species mass fractions (`(S,)`).
        zbase: Base metallicity $Z_{\\mathrm{base}}$ for `kap`.
        low_t_option: Low-temperature opacity table selection.

    Returns:
        Populated `AtmEvalOutput` dataclass.
    """
    cid_list = [int(c) for c in chem_id]
    xa_arr = np.ascontiguousarray(xa, dtype=np.float64)
    vals = _rust.atm_eval_t_tau_varying(
        float(tau_surf),
        float(luminosity),
        float(r),
        float(m),
        float(cgrav),
        int(t_tau_id),
        float(errtol),
        int(max_steps),
        bool(skip_partials),
        float(teff),
        cid_list,
        xa_arr,
        float(zbase),
        str(low_t_option),
    )
    return AtmEvalOutput(*[float(v) for v in vals])


def eval_irradiated(
    luminosity: float,
    r: float,
    m: float,
    cgrav: float,
    t_eq: float,
    p_surf: float,
    kap_guess: float,
    kap_v: float,
    gamma: float,
    errtol: float,
    max_iters: int,
    skip_partials: bool,
    chem_id: Sequence[int] | None = None,
    xa: npt.ArrayLike | None = None,
    zbase: float = 0.02,
    low_t_option: str = "low_t_fa05_gs98",
) -> AtmIrradiatedOutput:
    """Evaluate a Guillot & Havel (2011) irradiated grey atmosphere.

    Args:
        luminosity: Intrinsic luminosity $L$ ($\\mathrm{erg\\,s^{-1}}$).
        r: Planetary/stellar radius $R$ ($\\mathrm{cm}$).
        m: Mass $M$ ($\\mathrm{g}$).
        cgrav: Gravitational constant $G$ ($\\mathrm{cm^3\\,g^{-1}\\,s^{-2}}$).
        t_eq: Equilibrium irradiation temperature $T_{\\mathrm{eq}}$
            ($\\mathrm{K}$).
        p_surf: Surface boundary pressure $P_{\\mathrm{surf}}$
            ($\\mathrm{dyn\\,cm^{-2}}$).
        kap_guess: Initial thermal opacity guess $\\kappa$
            ($\\mathrm{cm^2\\,g^{-1}}$).
        kap_v: Visible-band opacity $\\kappa_v$ ($\\mathrm{cm^2\\,g^{-1}}$).
        gamma: Optical-to-thermal opacity ratio $\\gamma = \\kappa_v / \\kappa$
            (if `<= 0`, computed dynamically as `kap_v / kap`).
        errtol: Relative convergence tolerance on $\\kappa$.
        max_iters: Maximum opacity fixed-point iterations (`0` for fixed).
        skip_partials: If `True`, skip partial derivatives.
        chem_id: Optional species catalog IDs (`[S]`).
        xa: Optional species mass fractions (`(S,)`).
        zbase: Base metallicity $Z_{\\mathrm{base}}$ for `kap`.
        low_t_option: Low-temperature opacity table selection.

    Returns:
        Populated `AtmIrradiatedOutput` dataclass.
    """
    cid_list = [int(c) for c in chem_id] if chem_id is not None else None
    xa_arr = (
        np.ascontiguousarray(xa, dtype=np.float64) if xa is not None else None
    )
    vals = _rust.atm_eval_irradiated(
        float(luminosity),
        float(r),
        float(m),
        float(cgrav),
        float(t_eq),
        float(p_surf),
        float(kap_guess),
        float(kap_v),
        float(gamma),
        float(errtol),
        int(max_iters),
        bool(skip_partials),
        cid_list,
        xa_arr,
        float(zbase),
        str(low_t_option),
    )
    return AtmIrradiatedOutput(*[float(v) for v in vals])
