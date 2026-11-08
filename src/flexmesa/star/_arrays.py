# SPDX-License-Identifier: LGPL-3.0-or-later
"""Zero-copy NumPy array property mixin (`StarModelArrays`) and result types."""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import numpy.typing as npt

from flexmesa import _rust

type _ArrF64 = npt.NDArray[np.float64]
type _ArrI32 = npt.NDArray[np.int32]


@dataclasses.dataclass(frozen=True)
class JaxStarState:
    """Immutable JAX PyTree snapshot of a 1D stellar structure model.

    Attributes:
        lnd: Cell natural log density of shape `(nz,)` (`ln(g cm^-3)`).
        ln_t: Cell natural log temperature of shape `(nz,)` (`ln(K)`).
        ln_r: Face natural log radius of shape `(nz,)` (`ln(cm)`).
        lum: Face luminosity of shape `(nz,)` (`erg s^-1`).
        dq: Cell relative mass fractions of shape `(nz,)`.
        xa: Cell species mass fractions of shape `(nz, species)`.
        mstar: Total stellar mass in grams (`g`).
    """

    lnd: jax.Array
    ln_t: jax.Array
    ln_r: jax.Array
    lum: jax.Array
    dq: jax.Array
    xa: jax.Array
    mstar: float


@dataclasses.dataclass(frozen=True)
class HenyeySystemResult:
    """Assembled 1D Henyey residual vector and block-tridiagonal Jacobian.

    Attributes:
        equ: Residual array of shape `(nz, nvar)`.
        lblk: Sub-diagonal Jacobian blocks of shape `(nz, nvar, nvar)`.
        dblk: Diagonal Jacobian blocks of shape `(nz, nvar, nvar)`.
        ublk: Super-diagonal Jacobian blocks of shape `(nz, nvar, nvar)`.
        x_scale: Variable scaling array of shape `(nz, nvar)`.
    """

    equ: _ArrF64
    lblk: _ArrF64
    dblk: _ArrF64
    ublk: _ArrF64
    x_scale: _ArrF64


@dataclasses.dataclass(frozen=True)
class HenyeySolverReport:
    """Convergence report returned by `StarModel.solve_henyey`.

    Attributes:
        converged: Whether convergence tolerances were satisfied.
        iterations: Number of Newton-Raphson iterations executed.
        residual_norm: Mean absolute residual `sum(|equ|) / (nz * nvar)`.
        max_residual: Maximum absolute residual `max(|equ|)`.
        correction_norm: Mean absolute scaled correction.
        max_correction: Maximum absolute scaled correction.
    """

    converged: bool
    iterations: int
    residual_norm: float
    max_residual: float
    correction_norm: float
    max_correction: float


@dataclasses.dataclass(frozen=True)
class EvolveStepReport:
    """Diagnostic summary returned by `StarModel.evolve_step`.

    Attributes:
        model_number: Evolutionary step counter after completion.
        star_age: Stellar age in years (`yr`).
        dt: Timestep taken in seconds (`s`).
        dt_next: Proposed next timestep in seconds (`s`).
        iterations: Number of Henyey Newton-Raphson iterations.
        retries: Number of step retries (`dt` halvings).
        num_splits: Number of zone splits during remeshing.
        num_merges: Number of zone merges during remeshing.
        varcontrol: Measured normalized structural variation.
        converged: Whether the step converged.
    """

    model_number: int
    star_age: float
    dt: float
    dt_next: float
    iterations: int
    retries: int
    num_splits: int
    num_merges: int
    varcontrol: float
    converged: bool


@dataclasses.dataclass(frozen=True)
class HistoryRow:
    """Scalar evolutionary summary corresponding to a `history.data` line."""

    model_number: int
    star_age: float
    star_mass: float
    log_dt: float
    log_lh: float
    log_l: float
    log_teff: float
    log_r: float
    log_g: float
    log_center_t: float
    log_center_rho: float
    center_h1: float
    center_he4: float
    he_core_mass: float
    num_zones: int
    num_retries: int
    total_num_solver_iterations: int


class StarModelArrays:
    """Mixin exposing scalar and 1D/2D NumPy array properties on `StarModel`."""

    _rust_model: _rust.RustStarModel

    @property
    def nz(self) -> int:
        """Number of Lagrangian zones (`nz`)."""
        return self._rust_model.nz

    @property
    def species(self) -> int:
        """Number of active nuclear species (`species`)."""
        return self._rust_model.species

    @property
    def model_number(self) -> int:
        """Current evolutionary step number."""
        return self._rust_model.model_number

    @property
    def star_age(self) -> float:
        """Stellar age in years (`yr`)."""
        return self._rust_model.star_age

    @property
    def dt(self) -> float:
        """Current timestep in seconds (`s`)."""
        return self._rust_model.dt

    @dt.setter
    def dt(self, value: float) -> None:
        self._rust_model.dt = float(value)

    @property
    def dt_next(self) -> float:
        """Proposed next timestep in seconds (`s`)."""
        return self._rust_model.dt_next

    @property
    def dt_old(self) -> float:
        """Previous step timestep in seconds (`s`)."""
        return self._rust_model.dt_old

    @property
    def mstar(self) -> float:
        """Total stellar mass in grams (`g`)."""
        return self._rust_model.mstar

    @property
    def teff(self) -> float:
        """Surface effective temperature in Kelvin (`K`)."""
        return self._rust_model.teff

    @property
    def ln_peos_surf(self) -> float:
        """Surface natural log total pressure from atmosphere BC."""
        return self._rust_model.ln_peos_surf

    @property
    def ln_t_surf(self) -> float:
        """Surface natural log temperature from atmosphere BC."""
        return self._rust_model.ln_t_surf

    @property
    def iso_names(self) -> list[str]:
        """Active isotope names in zone order (`len = species`)."""
        return self._rust_model.iso_names

    @property
    def chem_id(self) -> list[int]:
        """Active 1-based `chem_id` indices (`len = species`)."""
        return self._rust_model.chem_id

    @property
    def lnd(self) -> _ArrF64:
        """Cell natural log density of shape `(nz,)` (`ln(g cm^-3)`)."""
        return self._rust_model.lnd

    @property
    def lnd_start(self) -> _ArrF64:
        """Start-of-step cell natural log density of shape `(nz,)`."""
        return self._rust_model.lnd_start

    @property
    def ln_t(self) -> _ArrF64:
        """Cell natural log temperature of shape `(nz,)` (`ln(K)`)."""
        return self._rust_model.ln_t

    @property
    def ln_r(self) -> _ArrF64:
        """Face natural log radius of shape `(nz,)` (`ln(cm)`)."""
        return self._rust_model.ln_r

    @property
    def lum(self) -> _ArrF64:
        """Face luminosity of shape `(nz,)` (`erg s^-1`)."""
        return self._rust_model.lum

    @property
    def dq(self) -> _ArrF64:
        """Cell relative mass fraction of shape `(nz,)`."""
        return self._rust_model.dq

    @property
    def q(self) -> _ArrF64:
        """Face enclosed mass fraction of shape `(nz,)`."""
        return self._rust_model.q

    @property
    def xq(self) -> _ArrF64:
        """Face exterior mass fraction `1 - q` of shape `(nz,)`."""
        return self._rust_model.xq

    @property
    def m(self) -> _ArrF64:
        """Face enclosed mass of shape `(nz,)` (`g`)."""
        return self._rust_model.m

    @property
    def dm(self) -> _ArrF64:
        """Cell mass of shape `(nz,)` (`g`)."""
        return self._rust_model.dm

    @property
    def dm_bar(self) -> _ArrF64:
        """Face control-volume mass of shape `(nz,)` (`g`)."""
        return self._rust_model.dm_bar

    @property
    def r(self) -> _ArrF64:
        """Face radius of shape `(nz,)` (`cm`)."""
        return self._rust_model.r

    @property
    def rmid(self) -> _ArrF64:
        """Cell volumetric mid-radius of shape `(nz,)` (`cm`)."""
        return self._rust_model.rmid

    @property
    def grav(self) -> _ArrF64:
        """Face gravitational acceleration of shape `(nz,)` (`cm s^-2`)."""
        return self._rust_model.grav

    @property
    def xa(self) -> _ArrF64:
        """Cell species mass fractions of shape `(nz, species)`."""
        return self._rust_model.xa

    @property
    def xh_mass(self) -> _ArrF64:
        """Cell hydrogen mass fraction `X` of shape `(nz,)`."""
        return self._rust_model.xh_mass

    @property
    def yhe_mass(self) -> _ArrF64:
        """Cell helium mass fraction `Y` of shape `(nz,)`."""
        return self._rust_model.yhe_mass

    @property
    def rho(self) -> _ArrF64:
        """Cell mass density of shape `(nz,)` (`g cm^-3`)."""
        return self._rust_model.rho

    @property
    def t(self) -> _ArrF64:
        """Cell temperature of shape `(nz,)` (`K`)."""
        return self._rust_model.t

    @property
    def pgas(self) -> _ArrF64:
        """Cell gas pressure of shape `(nz,)` (`dyn cm^-2`)."""
        return self._rust_model.pgas

    @property
    def prad(self) -> _ArrF64:
        """Cell radiation pressure of shape `(nz,)` (`dyn cm^-2`)."""
        return self._rust_model.prad

    @property
    def peos(self) -> _ArrF64:
        """Cell total pressure of shape `(nz,)` (`dyn cm^-2`)."""
        return self._rust_model.peos

    @property
    def ln_peos(self) -> _ArrF64:
        """Cell natural log total pressure of shape `(nz,)`."""
        return self._rust_model.ln_peos

    @property
    def energy(self) -> _ArrF64:
        """Cell specific internal energy of shape `(nz,)` (`erg g^-1`)."""
        return self._rust_model.energy

    @property
    def energy_start(self) -> _ArrF64:
        """Start-of-step cell specific internal energy of shape `(nz,)`."""
        return self._rust_model.energy_start

    @property
    def entropy(self) -> _ArrF64:
        """Cell specific entropy of shape `(nz,)` (`erg g^-1 K^-1`)."""
        return self._rust_model.entropy

    @property
    def grada(self) -> _ArrF64:
        """Cell adiabatic temperature gradient of shape `(nz,)`."""
        return self._rust_model.grada

    @property
    def cp(self) -> _ArrF64:
        """Cell specific heat `C_P` of shape `(nz,)` (`erg g^-1 K^-1`)."""
        return self._rust_model.cp

    @property
    def cv(self) -> _ArrF64:
        """Cell specific heat `C_V` of shape `(nz,)` (`erg g^-1 K^-1`)."""
        return self._rust_model.cv

    @property
    def chi_rho(self) -> _ArrF64:
        """Cell `chi_rho = (d ln P / d ln rho)_T` of shape `(nz,)`."""
        return self._rust_model.chi_rho

    @property
    def chi_t(self) -> _ArrF64:
        """Cell `chi_T = (d ln P / d ln T)_rho` of shape `(nz,)`."""
        return self._rust_model.chi_t

    @property
    def gamma1(self) -> _ArrF64:
        """Cell adiabatic exponent `Gamma_1` of shape `(nz,)`."""
        return self._rust_model.gamma1

    @property
    def eta(self) -> _ArrF64:
        """Cell electron degeneracy parameter of shape `(nz,)`."""
        return self._rust_model.eta

    @property
    def mu(self) -> _ArrF64:
        """Cell mean molecular weight of shape `(nz,)`."""
        return self._rust_model.mu

    @property
    def csound(self) -> _ArrF64:
        """Cell adiabatic sound speed of shape `(nz,)` (`cm s^-1`)."""
        return self._rust_model.csound

    @property
    def opacity(self) -> _ArrF64:
        """Cell Rosseland mean opacity of shape `(nz,)` (`cm^2 g^-1`)."""
        return self._rust_model.opacity

    @property
    def tau(self) -> _ArrF64:
        """Face optical depth of shape `(nz,)`."""
        return self._rust_model.tau

    @property
    def non_nuc_neu(self) -> _ArrF64:
        """Cell thermal neutrino energy loss rate (`erg g^-1 s^-1`)."""
        return self._rust_model.non_nuc_neu

    @property
    def eps_nuc(self) -> _ArrF64:
        """Cell nuclear energy generation rate of shape `(nz,)`."""
        return self._rust_model.eps_nuc

    @property
    def eps_grav(self) -> _ArrF64:
        """Cell gravitational energy rate of shape `(nz,)` (`erg g^-1 s^-1`)."""
        return self._rust_model.eps_grav

    @property
    def eps_mdot(self) -> _ArrF64:
        """Cell mass-change heating rate of shape `(nz,)` (`erg g^-1 s^-1`)."""
        return self._rust_model.eps_mdot

    @property
    def dxdt_nuc(self) -> _ArrF64:
        """Cell nuclear abundance rates of shape `(nz, species)` (`s^-1`)."""
        return self._rust_model.dxdt_nuc

    @property
    def gradt(self) -> _ArrF64:
        """Face actual temperature gradient `grad_T` of shape `(nz,)`."""
        return self._rust_model.gradt

    @property
    def gradr(self) -> _ArrF64:
        """Face radiative temperature gradient `grad_rad` of shape `(nz,)`."""
        return self._rust_model.gradr

    @property
    def gradl(self) -> _ArrF64:
        """Face Ledoux temperature gradient `grad_L` of shape `(nz,)`."""
        return self._rust_model.gradl

    @property
    def scale_height(self) -> _ArrF64:
        """Face pressure scale height `H_P` of shape `(nz,)` (`cm`)."""
        return self._rust_model.scale_height

    @property
    def mlt_vc(self) -> _ArrF64:
        """Face convective velocity of shape `(nz,)` (`cm s^-1`)."""
        return self._rust_model.mlt_vc

    @property
    def mlt_d(self) -> _ArrF64:
        """Face MLT diffusion coefficient of shape `(nz,)` (`cm^2 s^-1`)."""
        return self._rust_model.mlt_d

    @property
    def mlt_cdc(self) -> _ArrF64:
        """Face convective conductivity of shape `(nz,)`."""
        return self._rust_model.mlt_cdc

    @property
    def d_mix(self) -> _ArrF64:
        """Face diffusion coefficient `D_mix` of shape `(nz,)` (`cm^2 s^-1`)."""
        return self._rust_model.d_mix

    @property
    def conv_vel(self) -> _ArrF64:
        """Face convective velocity after overshooting of shape `(nz,)`."""
        return self._rust_model.conv_vel

    @property
    def sig(self) -> _ArrF64:
        """Face Lagrangian coupling `sig` of shape `(nz,)` (`g s^-1`)."""
        return self._rust_model.sig

    @property
    def omega(self) -> _ArrF64:
        """Face rotational angular velocity of shape `(nz,)` (`rad s^-1`)."""
        return self._rust_model.omega

    @property
    def j_rot(self) -> _ArrF64:
        """Face specific angular momentum of shape `(nz,)` (`cm^2 s^-1`)."""
        return self._rust_model.j_rot

    @property
    def brunt_b(self) -> _ArrF64:
        """Face composition gradient term `B` of shape `(nz,)`."""
        return self._rust_model.brunt_b

    @property
    def brunt_n2(self) -> _ArrF64:
        """Face Brunt-Väisälä frequency `N^2` of shape `(nz,)` (`s^-2`)."""
        return self._rust_model.brunt_n2

    @property
    def mixing_type(self) -> _ArrI32:
        """Face integer mixing classification codes of shape `(nz,)`."""
        return self._rust_model.mixing_type
