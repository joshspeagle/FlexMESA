# SPDX-License-Identifier: LGPL-3.0-or-later
"""Python and JAX hybrid stellar structure model (`StarModel`).

Wraps `flexmesa._rust.RustStarModel` (Tier A/B Safe Rust engine) and exposes
both zero-copy NumPy arrays and pure `@jax.jit` staggered-grid and face-gradient
kernels (`flexmesa.star.jax_kernels`).
"""

from __future__ import annotations

import dataclasses
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from flexmesa import _rust
from flexmesa.star import jax_kernels

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


class StarModel:
    """1D Lagrangian stellar structure and evolution model (`Star_Info`).

    Provides a unified Python/JAX interface backed by `RustStarModel`
    (`flexmesa-star`) for ZAMS interpolation, $n=1.5$ pre-MS polytrope
    construction, ASCII `.mod` and binary `.photo` I/O, and parallelized
    zone/face microphysics evaluation.
    """

    def __init__(self, rust_model: _rust.RustStarModel) -> None:
        """Initialize a `StarModel` from a `RustStarModel` instance.

        Args:
            rust_model: Underlying compiled `RustStarModel`.
        """
        self._rust_model = rust_model

    @classmethod
    def from_zams(
        cls,
        initial_mass: float = 1.0,
        initial_z: float = 0.02,
        initial_y: float = -1.0,
        eval_hydro: bool = True,
    ) -> StarModel:
        """Load a Zero-Age Main Sequence (ZAMS) model (`get_zams_model`).

        Args:
            initial_mass: Initial stellar mass in solar masses (`M_sun`).
            initial_z: Initial metal mass fraction `Z`.
            initial_y: Initial helium mass fraction `Y` (`< 0` uses
                `0.24 + 2*Z`).
            eval_hydro: Whether to immediately evaluate zone/face microphysics.

        Returns:
            Initialized `StarModel`.
        """
        rm = _rust.RustStarModel.from_zams(
            initial_mass=float(initial_mass),
            initial_z=float(initial_z),
            initial_y=float(initial_y),
        )
        inst = cls(rm)
        if eval_hydro:
            inst.eval_hydro_vars()
        return inst

    @classmethod
    def from_pre_ms(
        cls,
        initial_mass: float = 1.0,
        initial_z: float = 0.02,
        initial_y: float = -1.0,
        t_c: float = 9.0e5,
        d_log10_p: float = 0.02,
        eval_hydro: bool = True,
    ) -> StarModel:
        """Construct an $n = 1.5$ pre-main-sequence polytrope model.

        Args:
            initial_mass: Initial stellar mass in solar masses (`M_sun`).
            initial_z: Initial metal mass fraction `Z`.
            initial_y: Initial helium mass fraction `Y`.
            t_c: Central temperature in Kelvin (`K`).
            d_log10_p: Outward integration step size in `log10(P_gas)`.
            eval_hydro: Whether to immediately evaluate zone/face microphysics.

        Returns:
            Initialized `StarModel`.
        """
        rm = _rust.RustStarModel.from_pre_ms(
            initial_mass=float(initial_mass),
            initial_z=float(initial_z),
            initial_y=float(initial_y),
            t_c=float(t_c),
            d_log10_p=float(d_log10_p),
        )
        inst = cls(rm)
        if eval_hydro:
            inst.eval_hydro_vars(skip_net=True, skip_net_jacobian=True)
        return inst

    @classmethod
    def from_mod_str(cls, text: str, eval_hydro: bool = False) -> StarModel:
        """Parse an upstream MESA ASCII `.mod` string into a `StarModel`.

        Args:
            text: ASCII `.mod` file contents.
            eval_hydro: Whether to evaluate microphysics after loading.

        Returns:
            Loaded `StarModel`.
        """
        inst = cls(_rust.RustStarModel.from_mod_str(text))
        if eval_hydro:
            inst.eval_hydro_vars()
        return inst

    @classmethod
    def from_mod_file(
        cls, path: str | pathlib.Path, eval_hydro: bool = False
    ) -> StarModel:
        """Read an ASCII `.mod` file from disk.

        Args:
            path: Path to the `.mod` file.
            eval_hydro: Whether to evaluate microphysics after loading.

        Returns:
            Loaded `StarModel`.
        """
        text = pathlib.Path(path).read_text(encoding="utf-8")
        return cls.from_mod_str(text, eval_hydro=eval_hydro)

    @classmethod
    def from_photo_bytes(
        cls, data: bytes, eval_hydro: bool = False
    ) -> StarModel:
        """Restore a `StarModel` from binary `.photo` restart bytes.

        Args:
            data: Binary `.photo` checkpoint payload.
            eval_hydro: Whether to evaluate microphysics after loading.

        Returns:
            Restored `StarModel`.
        """
        inst = cls(_rust.RustStarModel.from_photo_bytes(data))
        if eval_hydro:
            inst.eval_hydro_vars()
        return inst

    def to_mod_str(self) -> str:
        """Serialize the model to an ASCII `.mod` string."""
        return self._rust_model.to_mod_str()

    def write_mod_file(self, path: str | pathlib.Path) -> None:
        """Write the model to an ASCII `.mod` file on disk."""
        pathlib.Path(path).write_text(self.to_mod_str(), encoding="utf-8")

    def to_photo_bytes(self) -> bytes:
        """Serialize the model to binary `.photo` checkpoint bytes."""
        return self._rust_model.to_photo_bytes()

    def eval_hydro_vars(
        self,
        skip_net: bool = False,
        skip_net_jacobian: bool = False,
    ) -> None:
        """Evaluate zone and face microphysics (`set_hydro_vars`).

        Args:
            skip_net: Skip nuclear network evaluation (`do_net`).
            skip_net_jacobian: Skip dense nuclear composition Jacobian.
        """
        self._rust_model.eval_hydro_vars(
            skip_net=bool(skip_net),
            skip_net_jacobian=bool(skip_net_jacobian),
        )

    def to_jax_state(self) -> JaxStarState:
        """Export primary structure variables as an immutable `JaxStarState`."""
        return JaxStarState(
            lnd=jnp.asarray(self.lnd),
            ln_t=jnp.asarray(self.ln_t),
            ln_r=jnp.asarray(self.ln_r),
            lum=jnp.asarray(self.lum),
            dq=jnp.asarray(self.dq),
            xa=jnp.asarray(self.xa),
            mstar=float(self.mstar),
        )

    def eval_face_mlt_jax(self) -> dict[str, jax.Array]:
        """Evaluate face temperature gradients, MLT, and Brunt `N^2` via JAX."""
        return dict(
            jax_kernels.eval_face_mlt_and_brunt_jax(
                dq=jnp.asarray(self.dq),
                ln_r=jnp.asarray(self.ln_r),
                lum=jnp.asarray(self.lum),
                m_grav=jnp.asarray(self.m),
                rho=jnp.asarray(self.rho),
                t=jnp.asarray(self.t),
                peos=jnp.asarray(self.peos),
                ln_peos=jnp.asarray(self.ln_peos),
                grada=jnp.asarray(self.grada),
                cp=jnp.asarray(self.cp),
                chi_rho=jnp.asarray(self.chi_rho),
                chi_t=jnp.asarray(self.chi_t),
                mu=jnp.asarray(self.mu),
                opacity=jnp.asarray(self.opacity),
                tau_0=float(self.tau[0]),
            )
        )

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
        """Isotope names of active species (`len = species`)."""
        return self._rust_model.iso_names

    @property
    def chem_id(self) -> list[int]:
        """1-based `chem_id` indices of active species."""
        return self._rust_model.chem_id

    @property
    def lnd(self) -> _ArrF64:
        """Cell natural log density of shape `(nz,)` (`ln(g cm^-3)`)."""
        return self._rust_model.lnd

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
    def rho(self) -> _ArrF64:
        """Cell density of shape `(nz,)` (`g cm^-3`)."""
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
    def entropy(self) -> _ArrF64:
        """Cell specific entropy of shape `(nz,)` (`erg g^-1 K^-1`)."""
        return self._rust_model.entropy

    @property
    def grada(self) -> _ArrF64:
        """Cell adiabatic temperature gradient of shape `(nz,)`."""
        return self._rust_model.grada

    @property
    def cp(self) -> _ArrF64:
        """Cell isobaric specific heat of shape `(nz,)` (`erg g^-1 K^-1`)."""
        return self._rust_model.cp

    @property
    def cv(self) -> _ArrF64:
        """Cell isochoric specific heat of shape `(nz,)` (`erg g^-1 K^-1`)."""
        return self._rust_model.cv

    @property
    def chi_rho(self) -> _ArrF64:
        """Cell `(dlnP/dlnRho)_T` of shape `(nz,)`."""
        return self._rust_model.chi_rho

    @property
    def chi_t(self) -> _ArrF64:
        """Cell `(dlnP/dlnT)_Rho` of shape `(nz,)`."""
        return self._rust_model.chi_t

    @property
    def gamma1(self) -> _ArrF64:
        """Cell first adiabatic exponent `Gamma_1` of shape `(nz,)`."""
        return self._rust_model.gamma1

    @property
    def eta(self) -> _ArrF64:
        """Cell electron degeneracy parameter `eta` of shape `(nz,)`."""
        return self._rust_model.eta

    @property
    def mu(self) -> _ArrF64:
        """Cell mean molecular weight `mu` of shape `(nz,)`."""
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
        """Face optical depth `tau` of shape `(nz,)`."""
        return self._rust_model.tau

    @property
    def non_nuc_neu(self) -> _ArrF64:
        """Cell thermal neutrino rate of shape `(nz,)` (`erg g^-1 s^-1`)."""
        return self._rust_model.non_nuc_neu

    @property
    def eps_nuc(self) -> _ArrF64:
        """Cell nuclear energy rate of shape `(nz,)` (`erg g^-1 s^-1`)."""
        return self._rust_model.eps_nuc

    @property
    def dxdt_nuc(self) -> _ArrF64:
        """Cell nuclear abundance rate of shape `(nz, species)` (`s^-1`)."""
        return self._rust_model.dxdt_nuc

    @property
    def gradt(self) -> _ArrF64:
        """Face actual temperature gradient `gradT` of shape `(nz,)`."""
        return self._rust_model.gradt

    @property
    def gradr(self) -> _ArrF64:
        """Face radiative temperature gradient `gradr` of shape `(nz,)`."""
        return self._rust_model.gradr

    @property
    def gradl(self) -> _ArrF64:
        """Face Ledoux temperature gradient `gradL` of shape `(nz,)`."""
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


def yrs_for_init_timestep(m_msun: float) -> float:
    """Compute the initial ZAMS timestep in years (`yrs_for_init_timestep`)."""
    return float(_rust.star_yrs_for_init_timestep(float(m_msun)))


def interp_q(dq1: _ArrF64, v1: _ArrF64, dq2: _ArrF64) -> _ArrF64:
    """Interpolate profile `v1` from Lagrangian grid `dq1` onto `dq2`."""
    return _rust.star_interp_q(
        np.ascontiguousarray(dq1, dtype=np.float64),
        np.ascontiguousarray(v1, dtype=np.float64),
        np.ascontiguousarray(dq2, dtype=np.float64),
    )
