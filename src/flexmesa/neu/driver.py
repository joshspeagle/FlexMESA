# SPDX-License-Identifier: LGPL-3.0-or-later
"""Top-level thermal neutrino loss driver (`neu_get`) and JAX `float64` kernel.

Ports `neu_get` (`neu/public/neu_lib.f90` and `neutrinos` in
`neu/private/mod_neu.f90`) and provides a pure-JAX `float64` differentiable
pair + plasmon neutrino loss function `neu_pair_plas_jax` compatible with
`jax.jit`, `jax.vmap`, and `jax.grad`.

Index and Memory-Order Contract:
    - Upstream 1-based channel indices (`pair_neu_type = 1 .. reco_neu_type =
      5`) map to 0-based `NeuChannel` (`PAIR = 0 .. RECO = 4`).
    - Upstream 1-based derivative indices (`ineu = 1 .. idneu_dzbar = 5`) map
      to `NeuRate` fields (`val`, `d_t`, `d_rho`, `d_abar`, `d_zbar`) and
      0-based row-major `sources` of shape `(5, 5)` (`sources[channel, deriv]`).
"""

from __future__ import annotations

import dataclasses
import enum
import math

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import constants as _const
from flexmesa import utils as _utils
from flexmesa.neu import brem as _brem
from flexmesa.neu import inputs as _inputs
from flexmesa.neu import pair_plas as _pair_plas
from flexmesa.neu import phot as _phot
from flexmesa.neu import reco as _reco


class NeuChannel(enum.IntEnum):
    """0-based thermal neutrino reaction channel index.

    Attributes:
        PAIR: Pair production ($e^+ + e^- \\to \\nu + \\bar{\\nu}$, index 0).
        PLAS: Plasmon decay ($\\tilde{\\gamma} \\to \\nu + \\bar{\\nu}$,
            index 1).
        PHOT: Photoneutrino ($e^\\pm + \\gamma \\to e^\\pm + \\nu +
            \\bar{\\nu}$, index 2).
        BREM: Bremsstrahlung ($e^- + (Z, A) \\to e^- + (Z, A) + \\nu +
            \\bar{\\nu}$, index 3).
        RECO: Recombination ($e^-_{\\mathrm{cont}} \\to e^-_{\\mathrm{bound}} +
            \\nu + \\bar{\\nu}$, index 4).
    """

    PAIR = 0
    PLAS = 1
    PHOT = 2
    BREM = 3
    RECO = 4


@dataclasses.dataclass(frozen=True)
class NeuFlags:
    """Per-channel enable flags (`flags(num_neu_types)` in `neu_def.f90`).

    Attributes:
        pair: Enable pair-production neutrino losses (`pair_neu_type`).
        plas: Enable plasmon-decay neutrino losses (`plas_neu_type`).
        phot: Enable photoneutrino losses (`phot_neu_type`).
        brem: Enable bremsstrahlung neutrino losses (`brem_neu_type`).
        reco: Enable recombination neutrino losses (`reco_neu_type`).
    """

    pair: bool = True
    plas: bool = True
    phot: bool = True
    brem: bool = True
    reco: bool = True

    @classmethod
    def all(cls) -> NeuFlags:
        """Enables all five neutrino reaction channels.

        Returns:
            `NeuFlags` with all five channels set to `True`.
        """
        return cls(pair=True, plas=True, phot=True, brem=True, reco=True)

    @classmethod
    def none(cls) -> NeuFlags:
        """Disables all five neutrino reaction channels.

        Returns:
            `NeuFlags` with all five channels set to `False`.
        """
        return cls(pair=False, plas=False, phot=False, brem=False, reco=False)

    @classmethod
    def default_stellar(cls) -> NeuFlags:
        """Standard stellar-evolution default (`reco = False`, others `True`).

        Returns:
            `NeuFlags` with `pair`, `plas`, `phot`, `brem` enabled.
        """
        return cls(pair=True, plas=True, phot=True, brem=True, reco=False)


@dataclasses.dataclass(frozen=True)
class NeuResult:
    """Complete thermal neutrino loss output (`loss` and `sources`).

    Attributes:
        total: Total specific neutrino energy loss rate (`NeuRate`).
        pair: Pair-production channel rate (`NeuRate`).
        plas: Plasmon-decay channel rate (`NeuRate`).
        phot: Photoneutrino channel rate (`NeuRate`).
        brem: Bremsstrahlung channel rate (`NeuRate`).
        reco: Recombination channel rate (`NeuRate`).
        loss: 1D `float64` NumPy array of shape `(5,)` containing
            `[val, d_t, d_rho, d_abar, d_zbar]`.
        sources: 2D row-major `float64` NumPy array of shape `(5, 5)` indexed
            as `sources[channel, deriv]`.
    """

    total: _inputs.NeuRate
    pair: _inputs.NeuRate
    plas: _inputs.NeuRate
    phot: _inputs.NeuRate
    brem: _inputs.NeuRate
    reco: _inputs.NeuRate
    loss: np.ndarray
    sources: np.ndarray

    def channel(self, ch: NeuChannel) -> _inputs.NeuRate:
        """Returns the `NeuRate` for the requested `NeuChannel`.

        Args:
            ch: Channel index (`NeuChannel`).

        Returns:
            Corresponding `NeuRate` structure.
        """
        return (self.pair, self.plas, self.phot, self.brem, self.reco)[int(ch)]


def _zero_result() -> NeuResult:
    """Constructs an all-zero `NeuResult` for below-cutoff temperatures."""
    z = _inputs.NeuRate()
    return NeuResult(
        total=z,
        pair=z,
        plas=z,
        phot=z,
        brem=z,
        reco=z,
        loss=np.zeros(5, dtype=np.float64),
        sources=np.zeros((5, 5), dtype=np.float64),
    )


def neu_get(
    t: float | None = None,
    log10_t: float | None = None,
    rho: float | None = None,
    log10_rho: float | None = None,
    abar: float = 1.0,
    zbar: float = 1.0,
    log10_tlim: float = 7.5,
    flags: NeuFlags | None = None,
) -> NeuResult:
    """Evaluates specific thermal neutrino energy loss rates and derivatives.

    Ports `neu_get` (`neu_lib.f90`) and `neutrinos` (`mod_neu.f90`). Accepts
    either linear or base-10 logarithmic temperature and density (computing
    whichever is omitted via IEEE-754 `log10` or `exp10`).

    Args:
        t: Temperature $T$ in Kelvin ($\\mathrm{K}$), or `None` if `log10_t` is
            provided.
        log10_t: Base-10 logarithm $\\log_{10}(T / \\mathrm{K})$, or `None` if
            `t` is provided.
        rho: Mass density $\\rho$ in $\\mathrm{g\\,cm^{-3}}$, or `None` if
            `log10_rho` is provided.
        log10_rho: Base-10 logarithm
            $\\log_{10}(\\rho / \\mathrm{g\\,cm^{-3}})$, or `None` if `rho` is
            provided.
        abar: Mean atomic weight $\\bar{A} > 0$ (dimensionless).
        zbar: Mean ionic charge $\\bar{Z} > 0$ (dimensionless).
        log10_tlim: Upper bound of smooth cosine cutoff window in
            $\\log_{10}(T / \\mathrm{K})$ (default `7.5`).
        flags: Per-channel enable flags (`NeuFlags`, defaults to
            `NeuFlags.all()`).

    Returns:
        `NeuResult` containing total and per-channel `NeuRate` values and
        row-major `float64` NumPy arrays `loss` `(5,)` and `sources` `(5, 5)`.

    Raises:
        ValueError: If neither `t` nor `log10_t` is provided, neither `rho` nor
            `log10_rho` is provided, or any input is non-positive, non-finite,
            or exceeds $\\log_{10} = 20$.
    """
    active_flags = NeuFlags.all() if flags is None else flags

    if t is None and log10_t is None:
        raise ValueError("Neither t nor log10_t was provided to neu_get.")
    if rho is None and log10_rho is None:
        raise ValueError("Neither rho nor log10_rho was provided to neu_get.")

    if t is not None:
        temp = t
    else:
        assert log10_t is not None
        temp = math.exp(log10_t * _const.LN10)
    if _utils.is_bad(temp) or temp <= 0.0:
        raise ValueError(f"Temperature t must be positive and finite: {temp}")
    logtemp = log10_t if log10_t is not None else math.log10(temp)
    if _utils.is_bad(logtemp) or logtemp > 20.0:
        raise ValueError(f"log10_t must be finite and <= 20: {logtemp}")

    if rho is not None:
        den = rho
    else:
        assert log10_rho is not None
        den = math.exp(log10_rho * _const.LN10)
    if _utils.is_bad(den) or den <= 0.0:
        raise ValueError(f"Density rho must be positive and finite: {den}")
    logden = log10_rho if log10_rho is not None else math.log10(den)
    if _utils.is_bad(logden) or logden > 20.0:
        raise ValueError(f"log10_rho must be finite and <= 20: {logden}")

    if (
        _utils.is_bad(abar)
        or _utils.is_bad(zbar)
        or _utils.is_bad(log10_tlim)
        or abar <= 0.0
        or zbar <= 0.0
    ):
        raise ValueError(
            f"abar, zbar, and log10_tlim must be valid: {abar}, {zbar}"
        )

    if temp <= _inputs.TMIN_NEU or logtemp <= _inputs.LOG10_TMIN_NEU:
        return _zero_result()

    inp = _inputs.NeuInputs.create(temp, logtemp, den, logden, abar, zbar)
    z_rate = _inputs.NeuRate()
    pair = (
        _pair_plas.pair_neu(inp).to_specific(inp.deni)
        if active_flags.pair
        else z_rate
    )
    plas = (
        _pair_plas.plas_neu(inp).to_specific(inp.deni)
        if active_flags.plas
        else z_rate
    )
    phot = (
        _phot.phot_neu(inp).to_specific(inp.deni)
        if active_flags.phot
        else z_rate
    )
    brem = (
        _brem.brem_neu(inp).to_specific(inp.deni)
        if active_flags.brem
        else z_rate
    )
    reco = (
        _reco.reco_neu(inp).to_specific(inp.deni)
        if active_flags.reco
        else z_rate
    )

    if inp.logtemp <= log10_tlim and log10_tlim > _inputs.LOG10_TMIN_NEU:
        dtlim = log10_tlim - _inputs.LOG10_TMIN_NEU
        phase = (inp.logtemp - _inputs.LOG10_TMIN_NEU) / dtlim
        f_cut = 0.5 * (1.0 - math.cos(_const.PI * phase))
        df_dt = (
            0.5
            * _const.PI
            * math.sin(_const.PI * phase)
            / (dtlim * temp * _const.LN10)
        )
        plas = plas.with_cutoff(f_cut, df_dt)
        pair = pair.with_cutoff(f_cut, df_dt)
        phot = phot.with_cutoff(f_cut, df_dt)
        brem = brem.with_cutoff(f_cut, df_dt)
        reco = reco.with_cutoff(f_cut, df_dt)

    sources_arr = np.array(
        [r.to_tuple() for r in (pair, plas, phot, brem, reco)],
        dtype=np.float64,
    )
    loss_arr = np.array(
        [
            plas.val + pair.val + phot.val + brem.val + reco.val,
            plas.d_t + pair.d_t + phot.d_t + brem.d_t + reco.d_t,
            plas.d_rho + pair.d_rho + phot.d_rho + brem.d_rho + reco.d_rho,
            plas.d_abar + pair.d_abar + phot.d_abar + brem.d_abar + reco.d_abar,
            plas.d_zbar + pair.d_zbar + phot.d_zbar + brem.d_zbar + reco.d_zbar,
        ],
        dtype=np.float64,
    )
    total = _inputs.NeuRate(*loss_arr.tolist())
    if _utils.is_bad(total.val):
        raise ValueError("Non-finite total neutrino loss encountered.")
    return NeuResult(total, pair, plas, phot, brem, reco, loss_arr, sources_arr)


def neu_pair_plas_jax(
    t: jax.Array,
    rho: jax.Array,
    abar: jax.Array,
    zbar: jax.Array,
) -> jax.Array:
    """Evaluates specific pair + plasmon neutrino loss in JAX `float64`.

    Pure-JAX `float64` implementation of `pair_neu + plas_neu` divided by
    $\\rho$, supporting `jax.jit`, `jax.vmap`, and `jax.grad` automatic
    differentiation with respect to `(t, rho, abar, zbar)`.

    Args:
        t: Temperature $T$ in Kelvin ($\\mathrm{K}$, scalar or `(*batch,)`).
        rho: Mass density $\\rho$ in $\\mathrm{g\\,cm^{-3}}$.
        abar: Mean atomic weight $\\bar{A}$.
        zbar: Mean ionic charge $\\bar{Z}$.

    Returns:
        Specific neutrino loss rate $\\epsilon_{\\mathrm{pair+plas}}$ in
        $\\mathrm{erg\\,g^{-1}\\,s^{-1}}$ (`jax.Array` in `float64`).
    """
    t = jnp.asarray(t, dtype=jnp.float64)
    rho = jnp.asarray(rho, dtype=jnp.float64)
    abar = jnp.asarray(abar, dtype=jnp.float64)
    zbar = jnp.asarray(zbar, dtype=jnp.float64)

    ye = zbar / abar
    rm = rho * ye
    xl = (t * 1.0e-9) * _inputs.CON1
    xl2 = xl * xl
    xl3 = xl2 * xl
    xl4 = xl3 * xl
    xl6 = xl4 * xl2
    xl8 = xl6 * xl2
    xl9 = xl8 * xl
    xlm1 = 1.0 / xl
    xlm2 = xlm1 * xlm1
    xlm3 = xlm1 * xlm2
    xlp5 = jnp.sqrt(xl)

    zeta = jnp.power(rm * 1.0e-9, _const.ONE_THIRD) * xlm1
    zeta2 = zeta * zeta
    zeta3 = zeta2 * zeta

    gl = 1.0 - 13.04 * xl2 + 133.5 * xl4 + 1534.0 * xl6 + 918.6 * xl8
    a1 = 6.002e19 + 2.084e20 * zeta + 1.872e21 * zeta2
    b1 = jnp.where(t < 1.0e10, jnp.exp(-5.5924 * zeta), jnp.exp(-4.9924 * zeta))
    den_a1 = jnp.where(
        t < 1.0e10,
        9.383e-1 * xlm1 - 4.141e-1 * xlm2 + 5.829e-2 * xlm3,
        1.2383 * xlm1 - 8.141e-1 * xlm2,
    )
    fpair = (a1 * b1) / (zeta3 + den_a1)
    q_xnum = 1.0 / (10.7480 * xl2 + 0.3967 * xlp5 + 1.005)
    d_a1 = 7.692e7 * xl3 + 9.715e6 * xlp5
    q_xden = jnp.power(1.0 + rm / d_a1, -0.3)
    qpair = q_xnum * q_xden
    q_pair_vol = (
        _inputs.TFAC4
        * (1.0 + _inputs.TFAC3 * qpair)
        * gl
        * jnp.exp(-2.0 * xlm1)
        * fpair
    )

    a2_p = jnp.power(1.019e-6 * rm, _const.TWO_THIRDS)
    b1_p = jnp.sqrt(1.0 + a2_p)
    gl2 = 1.1095e11 * rm / (t * t * b1_p)
    gl_p = jnp.sqrt(gl2)
    gl12 = jnp.sqrt(gl_p)
    gl32 = gl_p * gl12
    gl72 = gl2 * gl32
    gl6 = gl2 * gl2 * gl2
    ft = 2.4 + 0.6 * gl12 + 0.51 * gl_p + 1.25 * gl32
    fl = (8.6 * gl2 + 1.35 * gl72) / (225.0 - 17.0 * gl_p + gl2)
    cc = jnp.log10(2.0 * rm)
    xlnt = jnp.log10(t)
    xnum = _const.ONE_SIXTH * (17.5 + cc - 3.0 * xlnt)
    xden = _const.ONE_SIXTH * (-24.5 + cc + 3.0 * xlnt)
    p_a1 = 0.39 - 1.25 * xnum - 0.35 * jnp.sin(4.5 * xnum)
    p_b1 = 0.3 * jnp.exp(-jnp.square(4.5 * xnum + 0.9))
    c_val = jnp.minimum(0.0, xden - 1.6 + 1.25 * xnum)
    a3_val = c_val / (0.57 - 0.25 * xnum)
    fxy_inner = 1.05 + (p_a1 - p_b1) * jnp.exp(-a3_val * a3_val)
    fxy = jnp.where((jnp.abs(xnum) > 0.7) | (xden < 0.0), 1.0, fxy_inner)
    q_plas_vol = (
        _inputs.TFAC0 * 3.0e21 * xl9 * gl6 * jnp.exp(-gl_p) * (ft + fl) * fxy
    )
    return (q_pair_vol + q_plas_vol) / rho
