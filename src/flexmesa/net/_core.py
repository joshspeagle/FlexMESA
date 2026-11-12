# SPDX-License-Identifier: LGPL-3.0-or-later
"""Nuclear reaction networks, Jacobians, and one-zone burners (`net`)."""

from __future__ import annotations

import dataclasses

import numpy as np
import numpy.typing as npt

from flexmesa import _rust

type _ArrF64 = npt.NDArray[np.float64]


@dataclasses.dataclass(frozen=True)
class NetInfo:
    """Metadata for a compiled nuclear reaction network (`NetGeneralInfo`).

    Attributes:
        net_name: Network filename (e.g., `"basic.net"`, `"approx21.net"`).
        num_isos: Number of active nuclides (`S`).
        num_reactions: Number of active reactions (`R`).
        chem_id: 1-based catalog indices of the active nuclides (`[S]`).
        iso_names: Nuclide names (`[S]`).
        reaction_names: Reaction handle names (`[R]`).
    """

    net_name: str
    num_isos: int
    num_reactions: int
    chem_id: tuple[int, ...]
    iso_names: tuple[str, ...]
    reaction_names: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class NetEvalOutput:
    """Single-zone nuclear reaction network evaluation (`NetEvalResult`).

    Attributes:
        eps_nuc: Net nuclear energy generation rate
            ($\\mathrm{erg\\,g^{-1}\\,s^{-1}}$).
        d_eps_nuc_drho: Partial derivative
            $\\partial \\epsilon_{\\mathrm{nuc}} / \\partial \\rho$.
        d_eps_nuc_dt: Partial derivative
            $\\partial \\epsilon_{\\mathrm{nuc}} / \\partial T$.
        d_eps_nuc_dx: Partial derivatives
            $\\partial \\epsilon_{\\mathrm{nuc}} / \\partial X_i$ (`(S,)`).
        dxdt: Mass-fraction time derivatives $dX_i / dt$
            (`(S,)`, $\\mathrm{s^{-1}}$).
        d_dxdt_drho: Partial derivatives
            $\\partial(dX_i/dt)/\\partial \\rho$ (`(S,)`).
        d_dxdt_dt: Partial derivatives
            $\\partial(dX_i/dt)/\\partial T$ (`(S,)`).
        d_dxdt_dx: Analytical mass-fraction Jacobian
            $\\partial(dX_i/dt)/\\partial X_j$ (`(S, S)`).
        eps_nuc_categories: Category-resolved nuclear energy rates (`(24,)`).
        eps_neu_total: Neutrino energy loss rate
            ($\\mathrm{erg\\,g^{-1}\\,s^{-1}}$).
        rate_raw: Unscreened reaction rates (`(R,)`).
        rate_screened: Plasma-screened reaction rates (`(R,)`).
    """

    eps_nuc: float
    d_eps_nuc_drho: float
    d_eps_nuc_dt: float
    d_eps_nuc_dx: _ArrF64
    dxdt: _ArrF64
    d_dxdt_drho: _ArrF64
    d_dxdt_dt: _ArrF64
    d_dxdt_dx: _ArrF64
    eps_nuc_categories: _ArrF64
    eps_neu_total: float
    rate_raw: _ArrF64
    rate_screened: _ArrF64


@dataclasses.dataclass(frozen=True)
class Burn1ZoneOutput:
    """Output of a prescribed-$(T, \\rho)$ one-zone burn (`burn_1_zone`).

    Attributes:
        ending_x: Final mass fractions (`(S,)`).
        eps_nuc_categories: Final category-resolved energy rates (`(24,)`).
        avg_eps_nuc: Time-averaged net nuclear energy rate
            ($\\mathrm{erg\\,g^{-1}\\,s^{-1}}$).
        eps_neu_total: Final neutrino loss rate
            ($\\mathrm{erg\\,g^{-1}\\,s^{-1}}$).
        nfcn: Total right-hand-side evaluations.
        njac: Total Jacobian evaluations.
        nstep: Total integration substeps.
        naccpt: Accepted substeps.
        nrejct: Rejected substeps.
    """

    ending_x: _ArrF64
    eps_nuc_categories: _ArrF64
    avg_eps_nuc: float
    eps_neu_total: float
    nfcn: int
    njac: int
    nstep: int
    naccpt: int
    nrejct: int


@dataclasses.dataclass(frozen=True)
class BurnConstDensityOutput:
    """Output of `burn_1_zone_const_density` (constant-density variable-T burn).

    Attributes:
        ending_x: Final mass fractions (`(S,)`).
        ending_log10_t: Final $\\log_{10}(T_{\\mathrm{end}} / \\mathrm{K})$.
        eps_nuc_categories: Final category-resolved energy rates (`(24,)`).
        avg_eps_nuc: Time-averaged net nuclear energy rate
            ($\\mathrm{erg\\,g^{-1}\\,s^{-1}}$).
        eps_neu_total: Final neutrino loss rate
            ($\\mathrm{erg\\,g^{-1}\\,s^{-1}}$).
        nfcn: Total right-hand-side evaluations.
        njac: Total Jacobian evaluations.
        nstep: Total integration substeps.
        naccpt: Accepted substeps.
        nrejct: Rejected substeps.
    """

    ending_x: _ArrF64
    ending_log10_t: float
    eps_nuc_categories: _ArrF64
    avg_eps_nuc: float
    eps_neu_total: float
    nfcn: int
    njac: int
    nstep: int
    naccpt: int
    nrejct: int


@dataclasses.dataclass(frozen=True)
class BurnConstPOutput:
    """Output of a constant-pressure one-zone burn (`burn_1_zone_const_p`).

    Attributes:
        ending_x: Final mass fractions (`(S,)`).
        ending_temp: Final temperature $T$ ($\\mathrm{K}$).
        ending_rho: Final density $\\rho$ ($\\mathrm{g\\,cm^{-3}}$).
        ending_ln_s: Final natural log of specific entropy $\\ln S$.
        initial_rho: Initial density $\\rho_0$ ($\\mathrm{g\\,cm^{-3}}$).
        initial_ln_s: Initial natural log of specific entropy $\\ln S_0$.
        nfcn: Total right-hand-side evaluations.
        njac: Total Jacobian evaluations.
        nstep: Total Rosenbrock steps.
        naccpt: Accepted Rosenbrock steps.
        nrejct: Rejected Rosenbrock steps.
    """

    ending_x: _ArrF64
    ending_temp: float
    ending_rho: float
    ending_ln_s: float
    initial_rho: float
    initial_ln_s: float
    nfcn: int
    njac: int
    nstep: int
    naccpt: int
    nrejct: int


def list_networks() -> list[str]:
    """Return sorted list of built-in `.net` nuclear network filenames."""
    return list(_rust.net_list_networks())


def get_net_info(net_name: str = "basic.net") -> NetInfo:
    """Return nuclide and reaction metadata for `net_name`.

    Args:
        net_name: Built-in `.net` filename (e.g., `"basic.net"`).

    Returns:
        Populated `NetInfo` dataclass.
    """
    n_isos, n_rxns, cids, inames, rnames = _rust.net_info(str(net_name))
    return NetInfo(
        net_name=str(net_name),
        num_isos=int(n_isos),
        num_reactions=int(n_rxns),
        chem_id=tuple(int(c) for c in cids),
        iso_names=tuple(str(s) for s in inames),
        reaction_names=tuple(str(s) for s in rnames),
    )


def eval_net(
    net_name: str,
    x: npt.ArrayLike,
    temp: float,
    rho: float,
    abar: float,
    zbar: float,
    z2bar: float,
    ye: float,
    eta: float = 0.0,
    d_eta_dlnt: float = 0.0,
    d_eta_dlnrho: float = 0.0,
    log_t: float | None = None,
    log_rho: float | None = None,
    rate_factors: npt.ArrayLike | None = None,
    weak_rate_factor: float = 1.0,
    screening_mode: str = "extended",
    skip_jacobian: bool = False,
) -> NetEvalOutput:
    """Evaluate a nuclear reaction network at a single thermodynamic state.

    Args:
        net_name: Network filename (e.g., `"basic.net"`, `"approx21.net"`).
        x: Mass fractions (`(S,)`).
        temp: Temperature $T$ ($\\mathrm{K}$).
        rho: Mass density $\\rho$ ($\\mathrm{g\\,cm^{-3}}$).
        abar: Mean atomic mass number $\\bar{A}$.
        zbar: Mean atomic charge number $\\bar{Z}$.
        z2bar: Mean squared atomic charge $\\overline{Z^2}$.
        ye: Electron-to-baryon fraction $Y_e$.
        eta: Electron degeneracy parameter $\\eta$.
        d_eta_dlnt: $(\\partial \\eta / \\partial \\ln T)_\\rho$.
        d_eta_dlnrho: $(\\partial \\eta / \\partial \\ln \\rho)_T$.
        log_t: Optional $\\log_{10}(T / \\mathrm{K})$.
        log_rho: Optional $\\log_{10}(\\rho / \\mathrm{g\\,cm^{-3}})$.
        rate_factors: Optional per-reaction multipliers (`(R,)`).
        weak_rate_factor: Global weak-rate multiplier.
        screening_mode: `"extended"`, `"salpeter"`, `"chugunov"`,
            `"classic"`, or `"none"`.
        skip_jacobian: If `True`, skip Jacobian assembly.

    Returns:
        Populated `NetEvalOutput` dataclass.
    """
    x_arr = np.ascontiguousarray(x, dtype=np.float64)
    rf_arr = (
        np.ascontiguousarray(rate_factors, dtype=np.float64)
        if rate_factors is not None
        else None
    )
    t_val = float(temp)
    r_val = float(rho)
    lt_val = float(np.log10(t_val)) if log_t is None else float(log_t)
    lr_val = float(np.log10(r_val)) if log_rho is None else float(log_rho)
    out = _rust.net_eval(
        str(net_name),
        x_arr,
        t_val,
        lt_val,
        r_val,
        lr_val,
        float(abar),
        float(zbar),
        float(z2bar),
        float(ye),
        float(eta),
        float(d_eta_dlnt),
        float(d_eta_dlnrho),
        rf_arr,
        float(weak_rate_factor),
        str(screening_mode),
        bool(skip_jacobian),
    )
    return NetEvalOutput(
        eps_nuc=float(out[0]),
        d_eps_nuc_drho=float(out[1]),
        d_eps_nuc_dt=float(out[2]),
        d_eps_nuc_dx=np.asarray(out[3], dtype=np.float64),
        dxdt=np.asarray(out[4], dtype=np.float64),
        d_dxdt_drho=np.asarray(out[5], dtype=np.float64),
        d_dxdt_dt=np.asarray(out[6], dtype=np.float64),
        d_dxdt_dx=np.asarray(out[7], dtype=np.float64),
        eps_nuc_categories=np.asarray(out[8], dtype=np.float64),
        eps_neu_total=float(out[9]),
        rate_raw=np.asarray(out[10], dtype=np.float64),
        rate_screened=np.asarray(out[11], dtype=np.float64),
    )


def burn_1_zone(
    net_name: str,
    t_start: float,
    t_end: float,
    starting_x: npt.ArrayLike,
    times: npt.ArrayLike,
    log10_ts: npt.ArrayLike,
    log10_rhos: npt.ArrayLike,
    etas: npt.ArrayLike | None = None,
    rate_factors: npt.ArrayLike | None = None,
    weak_rate_factor: float = 1.0,
    screening_mode: str = "extended",
    stptry_in: float = 0.0,
    max_steps: int = 10000,
    eps: float = 1.0e-8,
    odescal: float = 1.0e-12,
) -> Burn1ZoneOutput:
    """Integrate a one-zone burn along $(T(t), \\rho(t))$ via Bader-Deuflhard.

    Args:
        net_name: Network filename (e.g., `"basic.net"`, `"approx21.net"`).
        t_start: Starting time $t_{\\mathrm{start}}$ ($\\mathrm{s}$).
        t_end: Ending time $t_{\\mathrm{end}}$ ($\\mathrm{s}$).
        starting_x: Initial mass fractions (`(S,)`).
        times: Time knot array (`(N_t,)`).
        log10_ts: $\\log_{10}(T / \\mathrm{K})$ at each time knot (`(N_t,)`).
        log10_rhos: $\\log_{10}(\\rho / \\mathrm{g\\,cm^{-3}})$ at each knot
            (`(N_t,)`).
        etas: Optional electron degeneracy $\\eta$ at each time knot (`(N_t,)`).
        rate_factors: Optional per-reaction multipliers (`(R,)`).
        weak_rate_factor: Global weak-rate multiplier.
        screening_mode: Plasma screening mode.
        stptry_in: Initial step size guess (`0.0` for automatic).
        max_steps: Maximum allowed substeps.
        eps: Relative error tolerance.
        odescal: Abundance scaling floor.

    Returns:
        Populated `Burn1ZoneOutput` dataclass.
    """
    x_arr = np.ascontiguousarray(starting_x, dtype=np.float64)
    t_arr = np.ascontiguousarray(times, dtype=np.float64)
    lgt_arr = np.ascontiguousarray(log10_ts, dtype=np.float64)
    lgr_arr = np.ascontiguousarray(log10_rhos, dtype=np.float64)
    eta_arr = (
        np.ascontiguousarray(etas, dtype=np.float64)
        if etas is not None
        else np.zeros_like(t_arr)
    )
    rf_arr = (
        np.ascontiguousarray(rate_factors, dtype=np.float64)
        if rate_factors is not None
        else None
    )
    out = _rust.net_burn_1_zone(
        str(net_name),
        float(t_start),
        float(t_end),
        x_arr,
        t_arr,
        lgt_arr,
        lgr_arr,
        eta_arr,
        rf_arr,
        float(weak_rate_factor),
        str(screening_mode),
        float(stptry_in),
        int(max_steps),
        float(eps),
        float(odescal),
    )
    return Burn1ZoneOutput(
        ending_x=np.asarray(out[0], dtype=np.float64),
        eps_nuc_categories=np.asarray(out[1], dtype=np.float64),
        avg_eps_nuc=float(out[2]),
        eps_neu_total=float(out[3]),
        nfcn=int(out[4]),
        njac=int(out[5]),
        nstep=int(out[6]),
        naccpt=int(out[7]),
        nrejct=int(out[8]),
    )


def burn_1_zone_const_density(
    net_name: str,
    t_start: float,
    t_end: float,
    starting_x: npt.ArrayLike,
    starting_log10_t: float,
    log10_rho: float,
    rate_factors: npt.ArrayLike | None = None,
    weak_rate_factor: float = 1.0,
    screening_mode: str = "extended",
    stptry_in: float = 0.0,
    max_steps: int = 10000,
    eps: float = 1.0e-5,
    odescal: float = 1.0e-5,
) -> BurnConstDensityOutput:
    """Integrate a constant-density variable-T burn via Bader-Deuflhard.

    Co-evolves molar abundances $Y_i = X_i / A_i$ and $\\ln T$ at fixed mass
    density $\\rho = 10^{\\log_{10}\\rho}$ using
    $d\\ln T / dt = \\epsilon_{\\mathrm{nuc}} / (c_V T)$.

    Args:
        net_name: Network filename (e.g., `"basic.net"`, `"approx21.net"`).
        t_start: Starting time $t_{\\mathrm{start}}$ ($\\mathrm{s}$).
        t_end: Ending time $t_{\\mathrm{end}}$ ($\\mathrm{s}$).
        starting_x: Initial mass fractions (`(S,)`).
        starting_log10_t: Initial $\\log_{10}(T_0 / \\mathrm{K})$.
        log10_rho: Fixed $\\log_{10}(\\rho / \\mathrm{g\\,cm^{-3}})$.
        rate_factors: Optional per-reaction multipliers (`(R,)`).
        weak_rate_factor: Global weak-rate multiplier.
        screening_mode: Plasma screening mode.
        stptry_in: Initial step size guess (`0.0` for automatic).
        max_steps: Maximum allowed substeps.
        eps: Relative integration error tolerance.
        odescal: Abundance scaling floor.

    Returns:
        Populated `BurnConstDensityOutput` dataclass.
    """
    x_arr = np.ascontiguousarray(starting_x, dtype=np.float64)
    rf_arr = (
        np.ascontiguousarray(rate_factors, dtype=np.float64)
        if rate_factors is not None
        else None
    )
    out = _rust.net_burn_1_zone_const_density(
        str(net_name),
        float(t_start),
        float(t_end),
        x_arr,
        float(starting_log10_t),
        float(log10_rho),
        rf_arr,
        float(weak_rate_factor),
        str(screening_mode),
        float(stptry_in),
        int(max_steps),
        float(eps),
        float(odescal),
    )
    return BurnConstDensityOutput(
        ending_x=np.asarray(out[0], dtype=np.float64),
        ending_log10_t=float(out[1]),
        eps_nuc_categories=np.asarray(out[2], dtype=np.float64),
        avg_eps_nuc=float(out[3]),
        eps_neu_total=float(out[4]),
        nfcn=int(out[5]),
        njac=int(out[6]),
        nstep=int(out[7]),
        naccpt=int(out[8]),
        nrejct=int(out[9]),
    )


def burn_1_zone_const_p(
    net_name: str,
    starting_temp: float,
    starting_x: npt.ArrayLike,
    t_end: float,
    pressure: float,
    solver: str = "rodas4",
    clip: bool = False,
    rate_factors: npt.ArrayLike | None = None,
    weak_rate_factor: float = 1.0,
    screening_mode: str = "extended",
    h0: float = 1.0e-6,
    max_step_size: float = 1.0,
    max_steps: int = 1000,
    rtol: float = 1.0e-7,
    atol: float = 1.0e-8,
    x_min: float = -1.0e-10,
) -> BurnConstPOutput:
    """Integrate a constant-pressure one-zone burn via Rosenbrock (`rodas4`).

    Args:
        net_name: Network filename (e.g., `"approx21.net"`).
        starting_temp: Initial temperature $T_0$ ($\\mathrm{K}$).
        starting_x: Initial mass fractions (`(S,)`).
        t_end: Integration duration $t_{\\mathrm{end}}$ ($\\mathrm{s}$).
        pressure: Constant total pressure $P$ ($\\mathrm{dyn\\,cm^{-2}}$).
        solver: Rosenbrock variant (`"rodas4"`, `"rodasp"`, `"ros3p"`,
            `"ros3pl"`).
        clip: Whether to clip negative abundances during integration.
        rate_factors: Optional per-reaction multipliers (`(R,)`).
        weak_rate_factor: Global weak-rate multiplier.
        screening_mode: Plasma screening mode.
        h0: Initial step size ($\\mathrm{s}$).
        max_step_size: Maximum step size ($\\mathrm{s}$).
        max_steps: Maximum Rosenbrock steps.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        x_min: Minimum allowed mass fraction before step rejection.

    Returns:
        Populated `BurnConstPOutput` dataclass.
    """
    x_arr = np.ascontiguousarray(starting_x, dtype=np.float64)
    rf_arr = (
        np.ascontiguousarray(rate_factors, dtype=np.float64)
        if rate_factors is not None
        else None
    )
    out = _rust.net_burn_1_zone_const_p(
        str(net_name),
        float(starting_temp),
        x_arr,
        float(t_end),
        float(pressure),
        str(solver),
        bool(clip),
        rf_arr,
        float(weak_rate_factor),
        str(screening_mode),
        float(h0),
        float(max_step_size),
        int(max_steps),
        float(rtol),
        float(atol),
        float(x_min),
    )
    return BurnConstPOutput(
        ending_x=np.asarray(out[0], dtype=np.float64),
        ending_temp=float(out[1]),
        ending_rho=float(out[2]),
        ending_ln_s=float(out[3]),
        initial_rho=float(out[4]),
        initial_ln_s=float(out[5]),
        nfcn=int(out[6]),
        njac=int(out[7]),
        nstep=int(out[8]),
        naccpt=int(out[9]),
        nrejct=int(out[10]),
    )
