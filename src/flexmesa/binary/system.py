# SPDX-License-Identifier: LGPL-3.0-or-later
"""Coupled `BinarySystem` evolution wrapper (`flexmesa.binary`)."""

from __future__ import annotations

from flexmesa import _rust
from flexmesa.star import model as star_model


class BinarySystem:
    """Coupled binary system (`Donor + PointMass` or `Donor + Accretor`)."""

    def __init__(self, rust_sys: _rust.RustBinarySystem) -> None:
        """Initialize from a compiled `RustBinarySystem` handle."""
        self._inner = rust_sys

    @classmethod
    def from_point_mass(
        cls,
        donor: star_model.StarModel,
        companion_mass_msun: float,
        initial_period_days: float,
        *,
        eccentricity: float = 0.0,
        is_black_hole: bool = False,
        initial_bh_spin: float = 0.0,
    ) -> BinarySystem:
        """Construct a coupled `Donor + PointMass` binary system.

        Args:
            donor: Donor `StarModel`.
            companion_mass_msun: Companion mass ($M_\\odot$).
            initial_period_days: Initial orbital period (days).
            eccentricity: Initial orbital eccentricity.
            is_black_hole: Whether the companion is an accreting Kerr BH.
            initial_bh_spin: Initial dimensionless BH spin $a \\in [0, 1]$.

        Returns:
            Initialized `BinarySystem`.
        """
        return cls(
            _rust.RustBinarySystem.new_point_mass(
                donor._rust_model,
                float(companion_mass_msun),
                float(initial_period_days),
                float(eccentricity),
                bool(is_black_hole),
                float(initial_bh_spin),
            )
        )

    @classmethod
    def from_two_stars(
        cls,
        donor: star_model.StarModel,
        accretor: star_model.StarModel,
        initial_period_days: float,
        *,
        eccentricity: float = 0.0,
    ) -> BinarySystem:
        """Construct a coupled `Donor + Accretor` two-star binary system."""
        return cls(
            _rust.RustBinarySystem.new_two_stars(
                donor._rust_model,
                accretor._rust_model,
                float(initial_period_days),
                float(eccentricity),
            )
        )

    def step(self, dt: float) -> tuple[float, int]:
        """Advance the coupled binary system by `dt` (s)."""
        return self._inner.step(float(dt))

    @property
    def donor(self) -> star_model.StarModel:
        """Return a snapshot of the donor `StarModel`."""
        return star_model.StarModel(self._inner.donor)

    @property
    def accretor(self) -> star_model.StarModel | None:
        """Return a snapshot of the accretor `StarModel` if evolved."""
        acc = self._inner.accretor
        return None if acc is None else star_model.StarModel(acc)

    @property
    def donor_mass_g(self) -> float:
        """Current donor mass (g)."""
        return self._inner.donor_mass_g

    @property
    def accretor_mass_g(self) -> float:
        """Current accretor mass (g)."""
        return self._inner.accretor_mass_g

    @property
    def separation(self) -> float:
        """Orbital separation $a$ (cm)."""
        return self._inner.separation

    @property
    def period(self) -> float:
        """Orbital period $P$ (s)."""
        return self._inner.period

    @property
    def eccentricity(self) -> float:
        """Orbital eccentricity $e$."""
        return self._inner.eccentricity

    @property
    def angular_momentum_j(self) -> float:
        """Orbital angular momentum $J_{\\mathrm{orb}}$ (`g cm^2 s^-1`)."""
        return self._inner.angular_momentum_j

    @property
    def rl(self) -> tuple[float, float]:
        """Roche-lobe radii `(rl_donor, rl_accretor)` (cm)."""
        return self._inner.rl

    @property
    def rl_relative_gap(self) -> tuple[float, float]:
        """Relative Roche-lobe gaps `(gap_donor, gap_accretor)`."""
        return self._inner.rl_relative_gap

    @property
    def mtransfer_rate(self) -> float:
        """Current RLOF rate $\\dot{M}_{\\mathrm{tr}} \\le 0$ (`g s^-1`)."""
        return self._inner.mtransfer_rate

    @property
    def jdot(self) -> float:
        """Total orbital torque $\\dot{J}_{\\mathrm{orb}}$ (`g cm^2 s^-2`)."""
        return self._inner.jdot

    @property
    def binary_age_yr(self) -> float:
        """Binary evolutionary age (yr)."""
        return self._inner.binary_age_yr

    @property
    def model_number(self) -> int:
        """Completed binary step count."""
        return self._inner.model_number

    @property
    def max_timestep_sec(self) -> float:
        """Next maximum binary timestep (s)."""
        return self._inner.max_timestep_sec
