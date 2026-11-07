# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tabulated weak nuclear rates (`weaklib`, `Suzuki2016`) and `ecapture`.

Provides Python interfaces to MESA's 463 weak reaction tables
(`weakreactions.tables` + `S13_r_be7_wk_li7.h5`), `sd`-shell `Suzuki2016.h5`
tables, `weak_info.list` half-lives, Coulomb chemical potential corrections
(`PCR2009`, `Itoh2002`), and on-the-fly relativistic Fermi-Dirac electron
capture / beta-decay phase-space integrals (`eval_ecapture_pair`).
"""

from __future__ import annotations

import dataclasses

from flexmesa import _rust


@dataclasses.dataclass(frozen=True)
class WeakRateEval:
    """Evaluated weak nuclear reaction rate and neutrino loss.

    Attributes:
        lambda_rate: Total weak transition rate
            $\\lambda = \\lambda_{\\mathrm{dec}} + \\lambda_{\\mathrm{cap}}$
            ($\\mathrm{s}^{-1}$).
        dlambda_dln_t: $\\partial \\lambda / \\partial \\ln T$
            ($\\mathrm{s}^{-1}$).
        dlambda_dln_rho: $\\partial \\lambda / \\partial \\ln(\\rho Y_e)$
            ($\\mathrm{s}^{-1}$).
        q: Total nuclear $Q$-value $\\Delta M_{\\mathrm{lhs}} -
            \\Delta M_{\\mathrm{rhs}}$ ($\\mathrm{MeV}$).
        dq_dln_t: $\\partial Q / \\partial \\ln T$ ($\\mathrm{MeV}$).
        dq_dln_rho: $\\partial Q / \\partial \\ln(\\rho Y_e)$ ($\\mathrm{MeV}$).
        qneu: Mean neutrino energy loss $Q_\\nu$ ($\\mathrm{MeV}$).
        dqneu_dln_t: $\\partial Q_\\nu / \\partial \\ln T$ ($\\mathrm{MeV}$).
        dqneu_dln_rho: $\\partial Q_\\nu / \\partial \\ln(\\rho Y_e)$
            ($\\mathrm{MeV}$).
    """

    lambda_rate: float
    dlambda_dln_t: float
    dlambda_dln_rho: float
    q: float
    dq_dln_t: float
    dq_dln_rho: float
    qneu: float
    dqneu_dln_t: float
    dqneu_dln_rho: float

    @classmethod
    def from_seq(cls, vals: list[float]) -> WeakRateEval:
        """Construct a [`WeakRateEval`] from a 9-element sequence."""
        return cls(
            lambda_rate=vals[0],
            dlambda_dln_t=vals[1],
            dlambda_dln_rho=vals[2],
            q=vals[3],
            dq_dln_t=vals[4],
            dq_dln_rho=vals[5],
            qneu=vals[6],
            dqneu_dln_t=vals[7],
            dqneu_dln_rho=vals[8],
        )


def eval_coulomb(
    temp: float,
    den: float,
    zbar: float,
    abar: float,
    z1: float,
    z2: float,
    *,
    which_mui: str = "pcr2009",
    which_vs: str = "itoh2002",
) -> tuple[float, float, float, float]:
    """Evaluate Coulomb chemical potentials and electron screening shift.

    Args:
        temp: Temperature $T$ in $\\mathrm{K}$.
        den: Mass density $\\rho$ in $\\mathrm{g\\,cm^{-3}}$.
        zbar: Mean ionic charge $\\bar{Z}$.
        abar: Mean atomic weight $\\bar{A}$.
        z1: Parent ion atomic number $Z_1$.
        z2: Daughter ion atomic number $Z_2$.
        which_mui: Ion chemical potential option (`"none"`, `"dcw1973"`,
            `"slr1982"`, `"pcr2009"`).
        which_vs: Electron screening shift option (`"none"`, `"throwe1983"`,
            `"itoh2002"`).

    Returns:
        Tuple `(mu1, mu2, dmu_eV, vs)` where `mu1` and `mu2` are dimensionless
        $\\mu_{\\mathrm{I}} / (k_{\\mathrm{B}} T)$, `dmu_eV` is
        $|\\Delta \\mu_{\\mathrm{I}}|$ in $\\mathrm{eV}$, and `vs` is the
        dimensionless screening shift $V_s$.
    """
    return _rust.rates_eval_coulomb(
        which_mui,
        which_vs,
        float(temp),
        float(den),
        float(zbar),
        float(abar),
        float(z1),
        float(z2),
    )


def eval_weak_pair(
    lhs_name: str,
    rhs_name: str,
    t9: float,
    ye_rho: float,
    *,
    use_suzuki: bool = False,
) -> WeakRateEval:
    """Evaluate a tabulated weak reaction rate (`weaklib` or `Suzuki2016`).

    Args:
        lhs_name: Parent nuclide name (e.g. `"mg24"`).
        rhs_name: Daughter nuclide name (e.g. `"na24"`).
        t9: Temperature $T_9 = T / 10^9\\,\\mathrm{K}$.
        ye_rho: Electron-weighted density $\\rho Y_e$ ($\\mathrm{g\\,cm^{-3}}$).
        use_suzuki: Whether to prefer `Suzuki2016.h5` over `weaklib`.

    Returns:
        Populated [`WeakRateEval`] instance.
    """
    vals = _rust.rates_eval_weak_pair(
        lhs_name, rhs_name, float(t9), float(ye_rho), bool(use_suzuki)
    )
    return WeakRateEval.from_seq(vals)


def eval_ecapture_pair(
    lhs_name: str,
    rhs_name: str,
    t9: float,
    ye_rho: float,
    eta_k: float,
    *,
    which_mui: str = "pcr2009",
    which_vs: str = "itoh2002",
    zbar: float = 10.909090909090908,
    abar: float = 21.818181818181817,
) -> WeakRateEval:
    """Evaluate an on-the-fly Fermi-Dirac electron-capture / beta-decay rate.

    Args:
        lhs_name: Parent nuclide name (e.g. `"mg24"`).
        rhs_name: Daughter nuclide name (e.g. `"na24"`).
        t9: Temperature $T_9 = T / 10^9\\,\\mathrm{K}$.
        ye_rho: Electron-weighted density $\\rho Y_e$ ($\\mathrm{g\\,cm^{-3}}$).
        eta_k: Electron degeneracy parameter
            $\\eta_e = \\mu_e / (k_{\\mathrm{B}} T)$ excluding rest mass.
        which_mui: Ion Coulomb chemical potential option (`"pcr2009"`).
        which_vs: Electron screening shift option (`"itoh2002"`).
        zbar: Mean ionic charge $\\bar{Z}$.
        abar: Mean atomic weight $\\bar{A}$.

    Returns:
        Populated [`WeakRateEval`] instance.
    """
    vals = _rust.rates_eval_ecapture_pair(
        lhs_name,
        rhs_name,
        float(t9),
        float(ye_rho),
        float(eta_k),
        which_mui,
        which_vs,
        float(zbar),
        float(abar),
    )
    return WeakRateEval.from_seq(vals)


def weak_info_by_names(lhs_name: str, rhs_name: str) -> tuple[float, float]:
    """Look up `(halflife_sec, qneu_mev)` from `weak_info.list`.

    Args:
        lhs_name: Parent nuclide name (e.g. `"o14"`).
        rhs_name: Daughter nuclide name (e.g. `"n14"`).

    Returns:
        Tuple `(halflife_sec, qneu_mev)` in $\\mathrm{s}$ and $\\mathrm{MeV}$.
    """
    return _rust.rates_weak_info_by_names(lhs_name, rhs_name)


def eval_weak_table_index(
    idx: int,
    t9: float,
    l_ye_rho: float,
) -> tuple[str, str, WeakRateEval]:
    """Evaluate the `idx`-th (`1..=463`) weak table at `(t9, l_ye_rho)`.

    Args:
        idx: 1-based weak table index (`1..=463`).
        t9: Temperature $T_9 = T / 10^9\\,\\mathrm{K}$.
        l_ye_rho: $\\log_{10}(\\rho Y_e / \\mathrm{g\\,cm^{-3}})$.

    Returns:
        Tuple `(lhs_name, rhs_name, WeakRateEval)`.
    """
    lhs, rhs, vals = _rust.rates_eval_weak_table_index(
        int(idx), float(t9), float(l_ye_rho)
    )
    return lhs, rhs, WeakRateEval.from_seq(vals)
