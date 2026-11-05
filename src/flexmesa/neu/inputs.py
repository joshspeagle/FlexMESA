# SPDX-License-Identifier: LGPL-3.0-or-later
"""Precomputed thermodynamic, composition, and weak-coupling invariants (`neu`).

Ports the electroweak coupling constants (`c_v`, `c_a`, `tfac0..tfac6`) and
shared thermodynamic/composition scaling quantities (`xl`, `rm`, `zeta` and
their analytical partial derivatives) from `neu/private/mod_neu.f90`.

Physical Units and Conventions:
    - Temperature `temp` in Kelvin ($\\mathrm{K}$), with scaled temperatures
      $T_9 = T / 10^9\\,\\mathrm{K}$ and $\\lambda = T / (5.9302 \\times
      10^9\\,\\mathrm{K})$.
    - Mass density `den` in $\\mathrm{g\\,cm^{-3}}$, with electron density
      factor $\\rho / \\mu_e = \\rho\\,\\bar{Z} / \\bar{A}$ (`rm`).
    - Dimensionless relativistic degeneracy parameter $\\xi = (10^{-9} \\rho
      Y_e)^{1/3} / \\lambda$ (`zeta`) and its analytical partial derivatives
      with respect to $(T, \\rho, \\bar{A}, \\bar{Z})$.
"""

from __future__ import annotations

import dataclasses
import math

from flexmesa import constants as _const

# Low-temperature hard cutoff in log10(T / K) (`log10Tmin_neu` in
# `neu_def.f90`).
LOG10_TMIN_NEU: float = 7.0

# Low-temperature hard cutoff in Kelvin (`Tmin_neu = 1e7` in `neu_def.f90`).
TMIN_NEU: float = 1.0e7

# Dimensionless relativistic electron temperature scale factor 1 / 5.9302
# (`con1` in `mod_neu.f90`).
CON1: float = 1.0 / 5.9302

# Number of additional neutrino families (`num_neu_fam - 1.0`).
SNEUT_Z: float = _const.NUM_NEU_FAM - 1.0

# Vector weak coupling C_V = 1/2 + 2 sin^2(theta_W).
C_V: float = 0.5 + 2.0 * _const.WEINBERG_THETA

# Axial-vector weak coupling C_A = 1/2.
C_A: float = 0.5

# Neutral-current vector weak coupling C_V' = 1 - C_V.
C_VP: float = 1.0 - C_V

# Neutral-current axial-vector weak coupling C_A' = 1 - C_A.
C_AP: float = 1.0 - C_A

# Squared vector weak coupling C_V^2 + (N_nu - 1) C_V'^2 (`tfac0` in
# `mod_neu.f90`).
TFAC0: float = C_V * C_V + SNEUT_Z * (C_VP * C_VP)

# Sum of squared electron weak couplings (`tfac1` in `mod_neu.f90`).
TFAC1: float = C_V * C_V + C_A * C_A + SNEUT_Z * (C_VP * C_VP + C_AP * C_AP)

# Difference of squared electron weak couplings (`tfac2` in `mod_neu.f90`).
TFAC2: float = C_V * C_V - C_A * C_A + SNEUT_Z * (C_VP * C_VP - C_AP * C_AP)

# Ratio TFAC2 / TFAC1 (`tfac3` in `mod_neu.f90`).
TFAC3: float = TFAC2 / TFAC1

# Half-sum of squared weak couplings 0.5 * TFAC1 (`tfac4` in `mod_neu.f90`).
TFAC4: float = 0.5 * TFAC1

# Half-difference of squared weak couplings 0.5 * TFAC2 (`tfac5` in
# `mod_neu.f90`).
TFAC5: float = 0.5 * TFAC2

# Recombination weak-coupling combination (`tfac6` in `mod_neu.f90`).
TFAC6: float = (
    C_V * C_V + 1.5 * C_A * C_A + SNEUT_Z * (C_VP * C_VP + 1.5 * C_AP * C_AP)
)


@dataclasses.dataclass(frozen=True)
class NeuRate:
    """Specific or volumetric neutrino energy loss rate and partial derivatives.

    Attributes:
        val: Energy loss rate $\\epsilon_\\nu$
            ($\\mathrm{erg\\,g^{-1}\\,s^{-1}}$) or volumetric rate $Q_\\nu$
            ($\\mathrm{erg\\,cm^{-3}\\,s^{-1}}$).
        d_t: Partial derivative $\\partial / \\partial T$
            ($\\mathrm{erg\\,g^{-1}\\,s^{-1}\\,K^{-1}}$).
        d_rho: Partial derivative $\\partial / \\partial \\rho$
            ($\\mathrm{erg\\,g^{-1}\\,s^{-1}\\,cm^3\\,g^{-1}}$).
        d_abar: Partial derivative $\\partial / \\partial \\bar{A}$
            ($\\mathrm{erg\\,g^{-1}\\,s^{-1}}$).
        d_zbar: Partial derivative $\\partial / \\partial \\bar{Z}$
            ($\\mathrm{erg\\,g^{-1}\\,s^{-1}}$).
    """

    val: float = 0.0
    d_t: float = 0.0
    d_rho: float = 0.0
    d_abar: float = 0.0
    d_zbar: float = 0.0

    def to_specific(self, deni: float) -> NeuRate:
        """Converts volumetric rate $Q$ to specific rate $Q / \\rho$.

        Args:
            deni: Reciprocal mass density $1 / \\rho$
                ($\\mathrm{cm^3\\,g^{-1}}$).

        Returns:
            Specific energy loss rate `NeuRate`
            ($\\mathrm{erg\\,g^{-1}\\,s^{-1}}$).
        """
        val = self.val * deni
        return NeuRate(
            val=val,
            d_t=self.d_t * deni,
            d_rho=self.d_rho * deni - val * deni,
            d_abar=self.d_abar * deni,
            d_zbar=self.d_zbar * deni,
        )

    def with_cutoff(self, f_cut: float, df_dt: float) -> NeuRate:
        """Applies smooth low-temperature cosine cutoff factor $f(T)$.

        Args:
            f_cut: Dimensionless cutoff multiplier $f(T) \\in [0, 1]$.
            df_dt: Temperature derivative $df / dT$ ($\\mathrm{K^{-1}}$).

        Returns:
            Cutoff-attenuated `NeuRate` with product-rule derivative.
        """
        return NeuRate(
            val=f_cut * self.val,
            d_t=f_cut * self.d_t + df_dt * self.val,
            d_rho=f_cut * self.d_rho,
            d_abar=f_cut * self.d_abar,
            d_zbar=f_cut * self.d_zbar,
        )

    def to_tuple(self) -> tuple[float, float, float, float, float]:
        """Returns `(val, d_t, d_rho, d_abar, d_zbar)` as a 5-tuple.

        Returns:
            Tuple `(val, d_t, d_rho, d_abar, d_zbar)` in `float64`.
        """
        return (self.val, self.d_t, self.d_rho, self.d_abar, self.d_zbar)


@dataclasses.dataclass(frozen=True)
class NeuInputs:
    """Precomputed thermodynamic and composition invariants for `neu` channels.

    Attributes:
        temp: Temperature $T$ ($\\mathrm{K}$).
        logtemp: Base-10 log temperature $\\log_{10}(T / \\mathrm{K})$.
        den: Mass density $\\rho$ ($\\mathrm{g\\,cm^{-3}}$).
        logden: Base-10 log density $\\log_{10}(\\rho / \\mathrm{g\\,cm^{-3}})$.
        abar: Mean atomic weight $\\bar{A}$.
        zbar: Mean ionic charge $\\bar{Z}$.
        tempi: Reciprocal temperature $1 / T$ ($\\mathrm{K^{-1}}$).
        deni: Reciprocal density $1 / \\rho$ ($\\mathrm{cm^3\\,g^{-1}}$).
        abari: Reciprocal mean atomic weight $1 / \\bar{A}$.
        zbari: Reciprocal mean ionic charge $1 / \\bar{Z}$.
        t9: Scaled temperature $T_9 = T / 10^9\\,\\mathrm{K}$.
        den6: Scaled density $\\rho_6 = \\rho / 10^6\\,\\mathrm{g\\,cm^{-3}}$.
        ye: Electron mole fraction $Y_e = \\bar{Z} / \\bar{A}$.
        xl: Relativistic temperature $\\lambda = T / (5.9302 \\times 10^9)$.
        xl2: Power $\\lambda^2$.
        xl3: Power $\\lambda^3$.
        xl4: Power $\\lambda^4$.
        xl5: Power $\\lambda^5$.
        xl6: Power $\\lambda^6$.
        xl7: Power $\\lambda^7$.
        xl8: Power $\\lambda^8$.
        xl9: Power $\\lambda^9$.
        xldt: Derivative $d\\lambda / dT = (5.9302 \\times 10^9)^{-1}$.
        xlm1: Inverse power $\\lambda^{-1}$.
        xlm2: Inverse power $\\lambda^{-2}$.
        xlm3: Inverse power $\\lambda^{-3}$.
        xlm4: Inverse power $\\lambda^{-4}$.
        xlp5: Fractional power $\\lambda^{1/2}$.
        xlmp5: Fractional power $\\lambda^{-1/2}$.
        rm: Electron mass density $\\rho Y_e$ ($\\mathrm{g\\,cm^{-3}}$).
        rmi: Reciprocal $1 / (\\rho Y_e)$.
        rmdd: Derivative $\\partial(\\rho Y_e)/\\partial \\rho = Y_e$.
        rmda: Derivative $\\partial(\\rho Y_e)/\\partial \\bar{A}$.
        rmdz: Derivative $\\partial(\\rho Y_e)/\\partial \\bar{Z}$.
        zeta: Degeneracy parameter $\\xi = (10^{-9}\\rho Y_e)^{1/3} / \\lambda$.
        zeta2: Power $\\xi^2$.
        zeta3: Power $\\xi^3$.
        zetadt: Derivative $\\partial \\xi / \\partial T$.
        zetadd: Derivative $\\partial \\xi / \\partial \\rho$.
        zetada: Derivative $\\partial \\xi / \\partial \\bar{A}$.
        zetadz: Derivative $\\partial \\xi / \\partial \\bar{Z}$.
    """

    temp: float
    logtemp: float
    den: float
    logden: float
    abar: float
    zbar: float
    tempi: float
    deni: float
    abari: float
    zbari: float
    t9: float
    den6: float
    ye: float
    xl: float
    xl2: float
    xl3: float
    xl4: float
    xl5: float
    xl6: float
    xl7: float
    xl8: float
    xl9: float
    xldt: float
    xlm1: float
    xlm2: float
    xlm3: float
    xlm4: float
    xlp5: float
    xlmp5: float
    rm: float
    rmi: float
    rmdd: float
    rmda: float
    rmdz: float
    zeta: float
    zeta2: float
    zeta3: float
    zetadt: float
    zetadd: float
    zetada: float
    zetadz: float

    @classmethod
    def create(
        cls,
        temp: float,
        logtemp: float,
        den: float,
        logden: float,
        abar: float,
        zbar: float,
    ) -> NeuInputs:
        """Constructs precomputed `NeuInputs` from thermodynamic state.

        Args:
            temp: Temperature $T$ ($\\mathrm{K}$).
            logtemp: Base-10 log of temperature $\\log_{10}(T / \\mathrm{K})$.
            den: Mass density $\\rho$ ($\\mathrm{g\\,cm^{-3}}$).
            logden: Base-10 logarithm of density $\\log_{10}(\\rho)$.
            abar: Mean atomic weight $\\bar{A}$.
            zbar: Mean ionic charge $\\bar{Z}$.

        Returns:
            Populated `NeuInputs` instance.
        """
        tempi = 1.0 / temp
        deni = 1.0 / den
        abari = 1.0 / abar
        zbari = 1.0 / zbar
        t9 = temp * 1.0e-9
        den6 = den * 1.0e-6
        ye = zbar * abari

        xldt = 1.0e-9 * CON1
        xl = t9 * CON1
        xl2 = xl * xl
        xl3 = xl2 * xl
        xl4 = xl3 * xl
        xl5 = xl4 * xl
        xl6 = xl5 * xl
        xl7 = xl6 * xl
        xl8 = xl7 * xl
        xl9 = xl8 * xl

        xlm1 = 1.0 / xl
        xlm2 = xlm1 * xlm1
        xlm3 = xlm1 * xlm2
        xlm4 = xlm1 * xlm3
        xlp5 = math.sqrt(xl)
        xlmp5 = 1.0 / xlp5

        rm = den * ye
        rmi = 1.0 / rm
        rmdd = ye
        rmda = -rm * abari
        rmdz = den * abari

        a0 = rm * 1.0e-9
        a1 = math.pow(a0, _const.ONE_THIRD)
        zeta = a1 * xlm1
        zeta2 = zeta * zeta
        zeta3 = zeta2 * zeta
        zetadt = -a1 * xlm2 * xldt
        a2 = _const.ONE_THIRD * a1 * rmi * xlm1
        zetadd = a2 * rmdd
        zetada = a2 * rmda
        zetadz = a2 * rmdz

        return cls(
            temp=temp,
            logtemp=logtemp,
            den=den,
            logden=logden,
            abar=abar,
            zbar=zbar,
            tempi=tempi,
            deni=deni,
            abari=abari,
            zbari=zbari,
            t9=t9,
            den6=den6,
            ye=ye,
            xl=xl,
            xl2=xl2,
            xl3=xl3,
            xl4=xl4,
            xl5=xl5,
            xl6=xl6,
            xl7=xl7,
            xl8=xl8,
            xl9=xl9,
            xldt=xldt,
            xlm1=xlm1,
            xlm2=xlm2,
            xlm3=xlm3,
            xlm4=xlm4,
            xlp5=xlp5,
            xlmp5=xlmp5,
            rm=rm,
            rmi=rmi,
            rmdd=rmdd,
            rmda=rmda,
            rmdz=rmdz,
            zeta=zeta,
            zeta2=zeta2,
            zeta3=zeta3,
            zetadt=zetadt,
            zetadd=zetadd,
            zetada=zetada,
            zetadz=zetadz,
        )
