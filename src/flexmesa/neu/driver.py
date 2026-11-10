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
import numpy as np

from flexmesa import _dispatch, _rust
from flexmesa import constants as _const
from flexmesa.neu import _jax
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
    z_loss = np.zeros(5, dtype=np.float64)
    z_src = np.zeros((5, 5), dtype=np.float64)
    return NeuResult(z, z, z, z, z, z, z_loss, z_src)


def _build_neu_result(
    loss_arr: np.ndarray, sources_arr: np.ndarray
) -> NeuResult:
    """Packs `(loss_arr, sources_arr)` into a validated `NeuResult`."""
    total = _inputs.NeuRate(*(float(v) for v in loss_arr))
    if not math.isfinite(total.val):
        raise ValueError("Non-finite total neutrino loss encountered.")
    ch = [
        _inputs.NeuRate(*(float(v) for v in sources_arr[i])) for i in range(5)
    ]
    return NeuResult(
        total, ch[0], ch[1], ch[2], ch[3], ch[4], loss_arr, sources_arr
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
    *,
    backend: _dispatch.Backend = "auto",
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
        backend: Execution backend (``"auto"``, ``"rust"``, or ``"jax"``).

    Returns:
        `NeuResult` containing total and per-channel `NeuRate` values and
        row-major `float64` NumPy arrays `loss` `(5,)` and `sources` `(5, 5)`.

    Raises:
        ValueError: If neither `t` nor `log10_t` is provided, neither `rho` nor
            `log10_rho` is provided, or any input is non-positive, non-finite,
            or exceeds $\\log_{10} = 20$.
    """
    active = NeuFlags.all() if flags is None else flags
    if t is None and log10_t is None:
        raise ValueError("Neither t nor log10_t was provided to neu_get.")
    if rho is None and log10_rho is None:
        raise ValueError("Neither rho nor log10_rho was provided to neu_get.")

    temp = t if t is not None else math.exp((log10_t or 0.0) * _const.LN10)
    if not math.isfinite(temp) or temp <= 0.0:
        raise ValueError(f"Temperature t must be positive and finite: {temp}")
    logtemp = log10_t if log10_t is not None else math.log10(temp)
    if not math.isfinite(logtemp) or logtemp > 20.0:
        raise ValueError(f"log10_t must be finite and <= 20: {logtemp}")

    den = rho if rho is not None else math.exp((log10_rho or 0.0) * _const.LN10)
    if not math.isfinite(den) or den <= 0.0:
        raise ValueError(f"Density rho must be positive and finite: {den}")
    logden = log10_rho if log10_rho is not None else math.log10(den)
    if not math.isfinite(logden) or logden > 20.0:
        raise ValueError(f"log10_rho must be finite and <= 20: {logden}")

    if (
        not math.isfinite(abar)
        or not math.isfinite(zbar)
        or not math.isfinite(log10_tlim)
        or abar <= 0.0
        or zbar <= 0.0
    ):
        raise ValueError(
            f"abar, zbar, and log10_tlim must be valid: {abar}, {zbar}"
        )

    if temp <= _inputs.TMIN_NEU or logtemp <= _inputs.LOG10_TMIN_NEU:
        return _zero_result()

    if (
        _dispatch.resolve_backend(
            backend, temp, logtemp, den, logden, abar, zbar
        )
        == "rust"
    ):
        flag_tup = (
            active.pair,
            active.plas,
            active.phot,
            active.brem,
            active.reco,
        )
        t_f, lt_f = float(temp), float(logtemp)
        d_f, ld_f = float(den), float(logden)
        a_f, z_f, tl_f = float(abar), float(zbar), float(log10_tlim)
        loss_r, src_r = _rust.neu_get(
            t_f, lt_f, d_f, ld_f, a_f, z_f, tl_f, flag_tup
        )
        return _build_neu_result(
            np.asarray(loss_r, dtype=np.float64),
            np.asarray(src_r, dtype=np.float64),
        )

    inp = _inputs.NeuInputs.create(temp, logtemp, den, logden, abar, zbar)
    z_rate = _inputs.NeuRate()
    pair = (
        _pair_plas.pair_neu(inp).to_specific(inp.deni)
        if active.pair
        else z_rate
    )
    plas = (
        _pair_plas.plas_neu(inp).to_specific(inp.deni)
        if active.plas
        else z_rate
    )
    phot = _phot.phot_neu(inp).to_specific(inp.deni) if active.phot else z_rate
    brem = _brem.brem_neu(inp).to_specific(inp.deni) if active.brem else z_rate
    reco = _reco.reco_neu(inp).to_specific(inp.deni) if active.reco else z_rate

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
    loss_arr = np.sum(sources_arr, axis=0)
    return _build_neu_result(loss_arr, sources_arr)


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
    ch = _jax.neu_channels_jax(t, rho, abar, zbar, log10_tlim=7.0)
    return ch[0] + ch[1]


neu_channels_jax = _jax.neu_channels_jax
neu_get_jax = _jax.neu_get_jax
ifermi12_jax = _jax.ifermi12_jax
zfermim12_jax = _jax.zfermim12_jax
