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
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import _dispatch, _rust
from flexmesa import constants as _const
from flexmesa.neu import _jax
from flexmesa.neu import inputs as _inputs


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

    def as_tuple(self) -> tuple[bool, bool, bool, bool, bool]:
        """Return channels as ``(pair, plas, phot, brem, reco)``."""
        return (self.pair, self.plas, self.phot, self.brem, self.reco)


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
    loss_arr: Any, sources_arr: Any, *, is_jax: bool = False
) -> NeuResult:
    """Packs `(loss_arr, sources_arr)` into a validated `NeuResult`."""
    if is_jax or isinstance(loss_arr, jax.core.Tracer):
        total = _inputs.NeuRate(*(cast(float, loss_arr[k]) for k in range(5)))
        ch = [
            _inputs.NeuRate(*(cast(float, sources_arr[i, k]) for k in range(5)))
            for i in range(5)
        ]
        return NeuResult(
            total,
            ch[0],
            ch[1],
            ch[2],
            ch[3],
            ch[4],
            cast(np.ndarray, loss_arr),
            cast(np.ndarray, sources_arr),
        )
    loss_np = np.asarray(loss_arr, dtype=np.float64)
    src_np = np.asarray(sources_arr, dtype=np.float64)
    total = _inputs.NeuRate(*(float(v) for v in loss_np))
    if not math.isfinite(total.val):
        raise ValueError("Non-finite total neutrino loss encountered.")
    ch = [_inputs.NeuRate(*(float(v) for v in src_np[i])) for i in range(5)]
    return NeuResult(total, ch[0], ch[1], ch[2], ch[3], ch[4], loss_np, src_np)


def _validate_concrete_neu_inputs(
    t: float | None,
    log10_t: float | None,
    rho: float | None,
    log10_rho: float | None,
    abar: float,
    zbar: float,
    log10_tlim: float,
    *,
    use_rust_math: bool,
) -> tuple[float, float, float, float, float, float, float]:
    """Validate and resolve linear/log10 concrete inputs for `neu_get`."""
    if t is None:
        lt_in = float(log10_t)  # type: ignore[arg-type]
        if not math.isfinite(lt_in) or lt_in > 20.0:
            raise ValueError(f"log10_t must be finite and <= 20: {lt_in}")
        temp = (
            _rust.exp10(lt_in)
            if use_rust_math
            else math.exp(lt_in * _const.LN10)
        )
    else:
        temp = float(t)
    if not math.isfinite(temp) or temp <= 0.0:
        raise ValueError(f"Temperature t must be positive and finite: {temp}")
    if log10_t is None:
        logtemp = (
            _rust.safe_log10(temp)
            if (use_rust_math and temp > 1.0e-99)
            else math.log10(temp)
        )
    else:
        logtemp = float(log10_t)
    if not math.isfinite(logtemp) or logtemp > 20.0:
        raise ValueError(f"log10_t must be finite and <= 20: {logtemp}")

    if rho is None:
        ld_in = float(log10_rho)  # type: ignore[arg-type]
        if not math.isfinite(ld_in) or ld_in > 20.0:
            raise ValueError(f"log10_rho must be finite and <= 20: {ld_in}")
        den = (
            _rust.exp10(ld_in)
            if use_rust_math
            else math.exp(ld_in * _const.LN10)
        )
    else:
        den = float(rho)
    if not math.isfinite(den) or den <= 0.0:
        raise ValueError(f"Density rho must be positive and finite: {den}")
    if log10_rho is None:
        logden = (
            _rust.safe_log10(den)
            if (use_rust_math and den > 1.0e-99)
            else math.log10(den)
        )
    else:
        logden = float(log10_rho)
    if not math.isfinite(logden) or logden > 20.0:
        raise ValueError(f"log10_rho must be finite and <= 20: {logden}")

    a_f, z_f, tl_f = float(abar), float(zbar), float(log10_tlim)
    if (
        not math.isfinite(a_f)
        or not math.isfinite(z_f)
        or not math.isfinite(tl_f)
        or a_f <= 0.0
        or z_f <= 0.0
    ):
        raise ValueError(
            f"abar, zbar, and log10_tlim must be valid: {abar}, {zbar}"
        )
    return temp, logtemp, den, logden, a_f, z_f, tl_f


def _eval_neu_jax(
    t: float | jax.Array | None,
    log10_t: float | jax.Array | None,
    rho: float | jax.Array | None,
    log10_rho: float | jax.Array | None,
    abar: float | jax.Array,
    zbar: float | jax.Array,
    log10_tlim: float | jax.Array,
    active: NeuFlags,
    *,
    has_jax: bool,
) -> NeuResult:
    """Evaluate `neu_get` via `_jax.neu_get_jax`, validating concrete args."""
    if not has_jax:
        temp, logtemp, den, _ld, a_f, z_f, tl_f = _validate_concrete_neu_inputs(
            cast("float | None", t),
            cast("float | None", log10_t),
            cast("float | None", rho),
            cast("float | None", log10_rho),
            cast(float, abar),
            cast(float, zbar),
            cast(float, log10_tlim),
            use_rust_math=False,
        )
        if temp <= _inputs.TMIN_NEU or logtemp <= _inputs.LOG10_TMIN_NEU:
            return _zero_result()
        loss_j, src_j = _jax.neu_get_jax(
            jnp.asarray(temp, dtype=jnp.float64),
            jnp.asarray(den, dtype=jnp.float64),
            jnp.asarray(a_f, dtype=jnp.float64),
            jnp.asarray(z_f, dtype=jnp.float64),
            cast(float, jnp.asarray(tl_f, dtype=jnp.float64)),
            flags=active.as_tuple(),
        )
        return _build_neu_result(loss_j, src_j, is_jax=False)

    temp_j = (
        jnp.asarray(t, dtype=jnp.float64)
        if t is not None
        else jnp.power(10.0, jnp.asarray(log10_t, dtype=jnp.float64))
    )
    rho_j = (
        jnp.asarray(rho, dtype=jnp.float64)
        if rho is not None
        else jnp.power(10.0, jnp.asarray(log10_rho, dtype=jnp.float64))
    )
    loss_j, src_j = _jax.neu_get_jax(
        temp_j,
        rho_j,
        jnp.asarray(abar, dtype=jnp.float64),
        jnp.asarray(zbar, dtype=jnp.float64),
        cast(float, jnp.asarray(log10_tlim, dtype=jnp.float64)),
        flags=active.as_tuple(),
    )
    return _build_neu_result(loss_j, src_j, is_jax=True)


def neu_get(
    t: float | jax.Array | None = None,
    log10_t: float | jax.Array | None = None,
    rho: float | jax.Array | None = None,
    log10_rho: float | jax.Array | None = None,
    abar: float | jax.Array = 1.0,
    zbar: float | jax.Array = 1.0,
    log10_tlim: float | jax.Array = 7.5,
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
        row-major `float64` arrays `loss` `(5,)` and `sources` `(5, 5)`.

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

    resolved, has_jax = _dispatch.resolve_backend_with_jax(
        backend, t, log10_t, rho, log10_rho, abar, zbar, log10_tlim
    )
    if resolved == "jax":
        return _eval_neu_jax(
            t,
            log10_t,
            rho,
            log10_rho,
            abar,
            zbar,
            log10_tlim,
            active,
            has_jax=has_jax,
        )

    temp, logtemp, den, logden, a_f, z_f, tl_f = _validate_concrete_neu_inputs(
        cast("float | None", t),
        cast("float | None", log10_t),
        cast("float | None", rho),
        cast("float | None", log10_rho),
        cast(float, abar),
        cast(float, zbar),
        cast(float, log10_tlim),
        use_rust_math=True,
    )
    if temp <= _inputs.TMIN_NEU or logtemp <= _inputs.LOG10_TMIN_NEU:
        return _zero_result()

    loss_r, src_r = _rust.neu_get(
        temp, logtemp, den, logden, a_f, z_f, tl_f, active.as_tuple()
    )
    return _build_neu_result(loss_r, src_r, is_jax=False)


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
