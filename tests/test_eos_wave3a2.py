# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for Wave 3a.2 (`HELM` equation of state in Rust, Python, and JAX)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flexmesa import eos

GOLDEN_STATES: list[tuple[str, float, float, float, float]] = [
    ("pure_he_2e8_100", 2.0e8, 100.0, 4.0, 2.0),
    ("pure_h_2e8_100", 2.0e8, 100.0, 1.0, 1.0),
    ("mixed_z_2e8_100", 2.0e8, 100.0, 1.72, 1.15),
    ("solar_1e6_100", 1.0e6, 100.0, 1.29818593, 1.10539606),
    ("test_components", 10.0**6.1, 10.0**0.1, 1.29666341, 1.10668478),
    ("sample_eos_c12", 1.0e9, 1.0e6, 12.0, 6.0),
]


def _rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Return max relative difference `|a - b| / max(|a|, |b|, 1.0)`."""
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0)
    return float(np.max(np.abs(a - b) / denom))


def _res_of_ln_td_az(
    ln_d: jax.Array, ln_t: jax.Array, abar: jax.Array, zbar: jax.Array
) -> jax.Array:
    """Return `res` from `eval_helm_eos` at `(ln_d, ln_t, abar, zbar)`."""
    out, _, _ = eos.eval_helm_eos(
        jnp.exp(ln_t), jnp.exp(ln_d), abar, zbar, backend="jax"
    )
    return jnp.asarray(out.res)


@pytest.mark.parametrize(
    ("label", "temp", "rho", "abar", "zbar"), GOLDEN_STATES
)
def test_eval_helm_eos_rust_and_jax_match_across_golden_states(
    label: str, temp: float, rho: float, abar: float, zbar: float
) -> None:
    """Verify `eval_helm_eos` Rust and JAX backends agree on golden states."""
    del label
    out_r, da_r, dz_r = eos.eval_helm_eos(temp, rho, abar, zbar, backend="rust")
    out_j, da_j, dz_j = eos.eval_helm_eos(
        jnp.asarray(temp, dtype=jnp.float64),
        jnp.asarray(rho, dtype=jnp.float64),
        jnp.asarray(abar, dtype=jnp.float64),
        jnp.asarray(zbar, dtype=jnp.float64),
        backend="jax",
    )
    assert _rel_diff(np.asarray(out_r.res), np.asarray(out_j.res)) < 1.0e-10
    assert (
        _rel_diff(np.asarray(out_r.d_dlnd), np.asarray(out_j.d_dlnd)) < 1.0e-6
    )
    assert (
        _rel_diff(np.asarray(out_r.d_dln_t), np.asarray(out_j.d_dln_t)) < 1.0e-6
    )
    assert _rel_diff(np.asarray(da_r), np.asarray(da_j)) < 1.0e-6
    assert _rel_diff(np.asarray(dz_r), np.asarray(dz_j)) < 1.0e-6
    assert float(out_r.res[eos.I_FRAC_HELM]) == 1.0
    assert float(out_j.res[eos.I_FRAC_HELM]) == 1.0


def test_eval_helm_eos_zones_matches_scalar_across_backends() -> None:
    """Verify `eval_helm_eos_zones` matches scalar evaluations across zones."""
    temps = np.array([s[1] for s in GOLDEN_STATES], dtype=np.float64)
    rhos = np.array([s[2] for s in GOLDEN_STATES], dtype=np.float64)
    abars = np.array([s[3] for s in GOLDEN_STATES], dtype=np.float64)
    zbars = np.array([s[4] for s in GOLDEN_STATES], dtype=np.float64)

    batch_r = eos.eval_helm_eos_zones(temps, rhos, abars, zbars, backend="rust")
    batch_j = eos.eval_helm_eos_zones(
        jnp.asarray(temps),
        jnp.asarray(rhos),
        jnp.asarray(abars),
        jnp.asarray(zbars),
        backend="jax",
    )
    assert batch_r.res.shape == (len(GOLDEN_STATES), eos.NUM_EOS_BASIC_RESULTS)
    assert _rel_diff(np.asarray(batch_r.res), np.asarray(batch_j.res)) < 1.0e-10
    assert (
        _rel_diff(np.asarray(batch_r.d_dlnd), np.asarray(batch_j.d_dlnd))
        < 1.0e-6
    )
    assert (
        _rel_diff(np.asarray(batch_r.d_dln_t), np.asarray(batch_j.d_dln_t))
        < 1.0e-6
    )

    with pytest.raises(ValueError, match="eval_helm_eos failed"):
        eos.eval_helm_eos(
            1.0e6, 10.0**6.5, 13.714285714285715, 6.8571428571428577
        )
    out_invalid, _, _ = eos.eval_helm_eos(
        jnp.asarray(1.0e6),
        jnp.asarray(10.0**6.5),
        jnp.asarray(13.714285714285715),
        jnp.asarray(6.8571428571428577),
        backend="jax",
    )
    assert bool(jnp.all(jnp.isnan(out_invalid.res)))


def test_eval_helm_eos_jax_autodiff_matches_analytical_derivatives() -> None:
    """Verify `jax.jacfwd` of `eval_helm_eos` matches analytical partials."""
    temp0 = 2.0e8
    rho0 = 100.0
    abar0 = 1.72
    zbar0 = 1.15
    out_j, da_j, dz_j = eos.eval_helm_eos(
        temp0, rho0, abar0, zbar0, backend="jax"
    )
    ad_dlnd, ad_dlnt, ad_da, ad_dz = jax.jacfwd(
        _res_of_ln_td_az, argnums=(0, 1, 2, 3)
    )(
        jnp.log(jnp.asarray(rho0, dtype=jnp.float64)),
        jnp.log(jnp.asarray(temp0, dtype=jnp.float64)),
        jnp.asarray(abar0, dtype=jnp.float64),
        jnp.asarray(zbar0, dtype=jnp.float64),
    )

    assert (
        _rel_diff(np.asarray(out_j.d_dlnd[:16]), np.asarray(ad_dlnd[:16]))
        < 1.0e-8
    )
    assert (
        _rel_diff(np.asarray(out_j.d_dln_t[:16]), np.asarray(ad_dlnt[:16]))
        < 1.0e-8
    )
    primary_idx = [
        eos.I_LN_PGAS,
        eos.I_LN_E,
        eos.I_LN_S,
        eos.I_LN_FREE_E,
        eos.I_ETA,
        eos.I_CHI_T,
        eos.I_GAMMA3,
    ]
    assert (
        _rel_diff(np.asarray(da_j)[primary_idx], np.asarray(ad_da)[primary_idx])
        < 1.0e-10
    )
    assert (
        _rel_diff(np.asarray(dz_j)[primary_idx], np.asarray(ad_dz)[primary_idx])
        < 1.0e-10
    )
    active_non_mu = [i for i in range(16) if i != eos.I_MU]
    assert (
        _rel_diff(
            np.asarray(da_j)[active_non_mu], np.asarray(ad_da)[active_non_mu]
        )
        < 1.0e-6
    )
    assert (
        _rel_diff(
            np.asarray(dz_j)[active_non_mu], np.asarray(ad_dz)[active_non_mu]
        )
        < 1.0e-6
    )
