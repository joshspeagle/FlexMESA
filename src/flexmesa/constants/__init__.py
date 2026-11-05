# SPDX-License-Identifier: LGPL-3.0-or-later
"""Physical, mathematical, and astronomical CGS constants for FlexMESA.

This module ports upstream MESA's `const_def` module in IEEE-754 double
precision (`float64`). Fundamental physical constants follow CODATA 2018 exact
definitions, and solar/planetary reference values follow the IAU 2012 and
IAU 2015 Resolution B3 nominal conversion constants.

Examples:
    >>> from flexmesa import constants
    >>> constants.MixingType.from_code(0) is constants.MixingType.NO_MIXING
    True
"""

from __future__ import annotations

import enum

# Mathematical constants (`float`, dimensionless)

#: Ratio of a circle's circumference to its diameter ($\pi$).
PI: float = 3.1415926535897932384626433832795028841971693993751

#: Twice $\pi$ ($2\pi$).
TWO_PI: float = 2.0 * PI

#: Square of $\pi$ ($\pi^2$), matching upstream `const_def::pi2`.
PI2: float = PI * PI

#: Four $\pi$ ($4\pi$), matching upstream `const_def::pi4`.
PI4: float = 4.0 * PI

#: Alias for $4\pi$.
FOUR_PI: float = PI4

#: Euler-Mascheroni constant ($\gamma$), matching `const_def::eulercon`.
EULERCON: float = 0.577215664901532861

#: Base of the natural logarithm ($e$), matching `const_def::eulernum`.
EULERNUM: float = 2.71828182845904523536028747135266249

#: Natural logarithm of 2 ($\ln 2$).
LN2: float = 6.9314718055994529e-1

#: Natural logarithm of 3 ($\ln 3$).
LN3: float = 1.0986122886681096

#: Natural logarithm of $\pi$ ($\ln \pi$).
LNPI: float = 1.14472988584940017414343

#: Natural logarithm of 10 ($\ln 10$).
LN10: float = 2.3025850929940455

#: Reciprocal natural logarithm of 10 ($1 / \ln 10$).
ILN10: float = 0.43429448190325187

#: Degrees-to-radians conversion factor ($\pi / 180$).
A2RAD: float = PI / 180.0

#: Radians-to-degrees conversion factor ($180 / \pi$).
RAD2A: float = 180.0 / PI

#: Alias for degrees-to-radians conversion factor.
DEG2RAD: float = A2RAD

#: Alias for radians-to-degrees conversion factor.
RAD2DEG: float = RAD2A

#: One third ($1/3$).
ONE_THIRD: float = 1.0 / 3.0

#: Two thirds ($2/3$).
TWO_THIRDS: float = 2.0 / 3.0

#: Four thirds ($4/3$).
FOUR_THIRDS: float = 4.0 / 3.0

#: Five thirds ($5/3$).
FIVE_THIRDS: float = 5.0 / 3.0

#: One sixth ($1/6$).
ONE_SIXTH: float = 1.0 / 6.0

#: Four thirds $\pi$ ($\frac{4}{3}\pi$), used for spherical volume factors.
FOUR_THIRDS_PI: float = FOUR_THIRDS * PI

#: Natural logarithm of $\frac{4}{3}\pi$ ($\ln(4\pi/3)$).
LN4PI3: float = 1.4324119583011810

#: Cube root of 2 ($2^{1/3}$).
TWO_13: float = 1.2599210498948730

#: Cube root of 4 ($4^{1/3}$).
FOUR_13: float = 1.5874010519681994

#: Square root of 2 ($\sqrt{2}$).
SQRT2: float = 1.414213562373095

#: Square root of $2/3$ ($\sqrt{2/3}$).
SQRT_2_DIV_3: float = 0.816496580927726

# Exact SI / CODATA 2018 physical constants in CGS units

#: Avogadro constant $N_{\mathrm{A}}$ in $\mathrm{mol^{-1}}$.
AVO: float = 6.02214076e23

#: Atomic mass unit $m_{\mathrm{u}} = 1 / N_{\mathrm{A}}$ in $\mathrm{g}$.
AMU: float = 1.0 / AVO

#: Speed of light in vacuum $c$ in $\mathrm{cm\,s^{-1}}$.
CLIGHT: float = 2.99792458e10

#: Elementary electron charge $q_{\mathrm{e}}$ in electrostatic units
#: ($\mathrm{esu}$).
QE: float = (CLIGHT / 10.0) * 1.602176634e-19

#: Boltzmann constant $k_{\mathrm{B}}$ in $\mathrm{erg\,K^{-1}}$.
KERG: float = 1.380649e-16

#: Alias for Boltzmann constant $k_{\mathrm{B}}$ in $\mathrm{erg\,K^{-1}}$.
BOLTZM: float = KERG

#: Planck constant $h$ in $\mathrm{erg\,s}$.
PLANCK_H: float = 6.62607015e-27

#: Reduced Planck constant $\hbar = h / (2\pi)$ in $\mathrm{erg\,s}$.
HBAR: float = PLANCK_H / TWO_PI

#: Ideal gas constant $\mathcal{R} = k_{\mathrm{B}} N_{\mathrm{A}}$ in
#: $\mathrm{erg\,K^{-1}\,mol^{-1}}$.
CGAS: float = BOLTZM * AVO

#: Electron volt in $\mathrm{erg}$ ($1\;\mathrm{eV}$).
EV2ERG: float = 1.602176634e-12

#: Mega-electron-volt in $\mathrm{erg}$ ($1\;\mathrm{MeV}$).
MEV_TO_ERGS: float = 1.0e6 * EV2ERG

#: Mega-electron-volt per atomic mass unit ($\mathrm{MeV} / m_{\mathrm{u}}$).
MEV_AMU: float = MEV_TO_ERGS / AMU

#: Mega-electron-volt equivalent mass in $\mathrm{g}$ ($\mathrm{MeV} / c^2$).
MEV2GR: float = MEV_TO_ERGS / (CLIGHT * CLIGHT)

#: Alias for mega-electron-volt equivalent mass in $\mathrm{g}$.
MEV_TO_GRAMS: float = MEV2GR

#: Nuclear energy conversion factor ($\mathrm{MeV} \cdot N_{\mathrm{A}}$ in
#: $\mathrm{erg\,mol^{-1}}$).
QCONV: float = MEV_TO_ERGS * AVO

#: Temperature-to-eV conversion factor $k_{\mathrm{B}} / \mathrm{eV}$ in
#: $\mathrm{eV\,K^{-1}}$.
KEV: float = KERG / EV2ERG

#: Stefan-Boltzmann constant $\sigma_{\mathrm{SB}}$ in
#: $\mathrm{erg\,cm^{-2}\,s^{-1}\,K^{-4}}$.
BOLTZ_SIGMA: float = (PI * PI * BOLTZM * BOLTZM * BOLTZM * BOLTZM) / (
    60.0 * HBAR * HBAR * HBAR * CLIGHT * CLIGHT
)

#: Radiation density constant $a_{\mathrm{rad}} = 4\sigma_{\mathrm{SB}} / c$ in
#: $\mathrm{erg\,cm^{-3}\,K^{-4}}$.
CRAD: float = (BOLTZ_SIGMA * 4.0) / CLIGHT

# Astronomical constants in CGS units (IAU 2012 / 2015 Resolution B3)

#: Astronomical unit in $\mathrm{cm}$ (IAU 2012 exact).
AU: float = 1.49597870700e13

#: Parsec in $\mathrm{cm}$ ($3600 \cdot (180/\pi)\;\mathrm{au}$).
PC: float = (3.600e3 * RAD2A) * AU

#: Days per Julian year.
DAYYER: float = 365.25

#: Seconds per day ($86\,400\;\mathrm{s}$).
SECDAY: float = 86400.0

#: Seconds per Julian year ($365.25 \times 86\,400\;\mathrm{s}$).
SECYER: float = SECDAY * DAYYER

#: Light-year in $\mathrm{cm}$ ($c \cdot \mathrm{yr}$).
LY: float = CLIGHT * SECYER

# Measured particle and atomic constants (CODATA 2018)

#: Neutron mass $m_{\mathrm{n}}$ in $\mathrm{g}$ (CODATA 2018).
MN: float = 1.67492749804e-24

#: Proton mass $m_{\mathrm{p}}$ in $\mathrm{g}$ (CODATA 2018).
MP: float = 1.67262192369e-24

#: Electron mass $m_{\mathrm{e}}$ in $\mathrm{g}$ (CODATA 2018).
ME: float = 9.1093837015e-28

#: Bohr radius $a_0$ in $\mathrm{cm}$ (CODATA 2018).
RBOHR: float = 5.29177210903e-9

#: Fine-structure constant $\alpha$ (dimensionless, CODATA 2018).
FINE: float = 7.2973525693e-3

#: Hydrogen ionization potential (Rydberg energy) in $\mathrm{eV}$.
HION: float = 13.605693122994

#: Thomson scattering cross section $\sigma_{\mathrm{T}}$ in $\mathrm{cm^2}$.
SIGE: float = 6.6524587321e-25

#: Weak mixing angle $\sin^2 \theta_{\mathrm{W}}$.
WEINBERG_THETA: float = 0.22290

#: Effective number of light neutrino families $N_{\nu}$.
NUM_NEU_FAM: float = 3.0

# Solar and planetary reference constants (IAU 2015 Resolution B3)

#: Standard Newtonian gravitational constant $G$ in
#: $\mathrm{cm^3\,g^{-1}\,s^{-2}}$ (CODATA 2018).
STANDARD_CGRAV: float = 6.67430e-8

#: Nominal solar mass parameter $\mu_{\odot} = G M_{\odot}$ in
#: $\mathrm{cm^3\,s^{-2}}$ (IAU 2015 Resolution B3).
MU_SUN: float = 1.3271244e26

#: Nominal terrestrial mass parameter $\mu_{\oplus} = G M_{\oplus}$ in
#: $\mathrm{cm^3\,s^{-2}}$.
MU_EARTH: float = 3.986004e20

#: Nominal Jovian mass parameter $\mu_{\mathrm{J}} = G M_{\mathrm{J}}$ in
#: $\mathrm{cm^3\,s^{-2}}$.
MU_JUPITER: float = 1.2668653e23

#: Solar age in Julian years (Bahcall et al. 2005, ApJ, 618, 1049).
AGESUN: float = 4.57e9

#: Solar mass $M_{\odot} = \mu_{\odot} / G$ in $\mathrm{g}$.
MSUN: float = MU_SUN / STANDARD_CGRAV

#: Alias for solar mass $M_{\odot}$ in $\mathrm{g}$.
M_SUN: float = MSUN

#: Nominal solar radius $R_{\odot}$ in $\mathrm{cm}$ (IAU 2015 Resolution B3).
RSUN: float = 6.957e10

#: Alias for solar radius $R_{\odot}$ in $\mathrm{cm}$.
R_SUN: float = RSUN

#: Nominal solar luminosity $L_{\odot}$ in $\mathrm{erg\,s^{-1}}$
#: (IAU 2015 Resolution B3).
LSUN: float = 3.828e33

#: Alias for solar luminosity $L_{\odot}$ in $\mathrm{erg\,s^{-1}}$.
L_SUN: float = LSUN

#: Nominal solar effective temperature $T_{\mathrm{eff},\odot}$ in $\mathrm{K}$.
TEFFSUN: float = 5772.0

#: Base-10 logarithm of nominal solar surface gravity $\log_{10}(g_{\odot})$
#: in CGS ($\mathrm{cm\,s^{-2}}$).
LOGGSUN: float = 4.4380676273031332

#: Nominal solar absolute bolometric magnitude $M_{\mathrm{bol},\odot}$.
MBOLSUN: float = 4.74

#: Earth mass $M_{\oplus} = \mu_{\oplus} / G$ in $\mathrm{g}$.
M_EARTH: float = MU_EARTH / STANDARD_CGRAV

#: Nominal Earth equatorial radius $R_{\oplus}$ in $\mathrm{cm}$.
R_EARTH: float = 6.3781e8

#: Nominal Earth polar radius in $\mathrm{cm}$.
R_EARTH_POLAR: float = 6.3568e8

#: Jupiter mass $M_{\mathrm{J}} = \mu_{\mathrm{J}} / G$ in $\mathrm{g}$.
M_JUPITER: float = MU_JUPITER / STANDARD_CGRAV

#: Nominal Jupiter equatorial radius $R_{\mathrm{J}}$ in $\mathrm{cm}$.
R_JUPITER: float = 7.1492e9

#: Nominal Jupiter polar radius in $\mathrm{cm}$.
R_JUPITER_POLAR: float = 6.6854e9

#: Jupiter orbital semimajor axis in $\mathrm{cm}$.
SEMIMAJOR_AXIS_JUPITER: float = 7.7857e13

#: Upstream MESA sentinel for an omitted optional scalar argument (`-9e99`).
ARG_NOT_PROVIDED: float = -9.0e99

#: Upstream MESA sentinel for a missing scalar value (`-9e99`).
MISSING_VALUE: float = ARG_NOT_PROVIDED

#: Number of non-negative mixing-type categories (`phase_separation + 1`).
NUMBER_OF_MIXING_TYPES: int = 11


class MixingType(enum.IntEnum):
    """Mixing process classification matching upstream MESA `const_def`."""

    CRYSTALLIZED = -1
    NO_MIXING = 0
    CONVECTIVE_MIXING = 1
    OVERSHOOT_MIXING = 2
    SEMICONVECTIVE_MIXING = 3
    THERMOHALINE_MIXING = 4
    ROTATION_MIXING = 5
    RAYLEIGH_TAYLOR_MIXING = 6
    MINIMUM_MIXING = 7
    ANONYMOUS_MIXING = 8
    LEFTOVER_CONVECTIVE_MIXING = 9
    PHASE_SEPARATION_MIXING = 10

    @classmethod
    def from_code(cls, code: int) -> MixingType | None:
        """Convert an integer mixing code from MESA into a `MixingType`.

        Args:
            code: Integer mixing code (`-1` through `10`).

        Returns:
            Corresponding `MixingType` member, or `None` if out of range.
        """
        try:
            return cls(int(code))
        except ValueError:
            return None

    def code(self) -> int:
        """Return the integer mixing code matching MESA `const_def`."""
        return int(self.value)
