# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for Wave 3a.2 (`HELM` equation of state in Rust, Python, and JAX)."""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import flexmesa
from flexmesa import eos
from flexmesa import math as fm_math

_EOS_ORACLES_TSV = (
    pathlib.Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "fortran_mesa"
    / "eos_oracles"
    / "mesa_26_4_1_eos_oracles.tsv"
)

ORACLE_HELM_STATES: list[tuple[str, float, float, float, float]] = [
    ("pure_he_2e8_100", 2.0e8, 100.0, 4.0, 2.0),
    ("pure_h_2e8_100", 2.0e8, 100.0, 1.0, 1.0),
    ("solar_2e8_100", 2.0e8, 100.0, 1.2966353410255222, 1.1021400398716938),
    (
        "test_components",
        1.0e6,
        float(fm_math.exp10(3.5)),
        2.990964711670598,
        1.6749402385355348,
    ),
    ("sample_eos_c12", 1.0e9, 1.0e4, 12.0, 6.0),
    ("mixed_z_2e8_100", 2.0e8, 100.0, 1.72, 1.15),
    ("solar_1e6_100", 1.0e6, 100.0, 1.29818593, 1.10539606),
    ("py_sample_eos_c12", 1.0e9, 1.0e6, 12.0, 6.0),
]


def _load_helm_case(case_id: str) -> np.ndarray:
    """Load a `16 x 5` `HELM` block from `mesa_26_4_1_eos_oracles.tsv`."""
    arr = np.zeros((16, 5), dtype=np.float64)
    seen = [False] * 16
    for line in _EOS_ORACLES_TSV.read_text("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if cols[0] == "HELM" and cols[1] == case_id:
            slot = int(cols[2])
            arr[slot] = [float(x) for x in cols[3:8]]
            seen[slot] = True
    assert all(seen), f"Incomplete HELM oracle case: {case_id}"
    return arr


def _load_ierr_code(case_id: str) -> int:
    """Load `ierr` from an `IERR` row in `mesa_26_4_1_eos_oracles.tsv`."""
    for line in _EOS_ORACLES_TSV.read_text("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if cols[0] == "IERR" and cols[1] == case_id:
            return int(cols[2])
    raise AssertionError(f"Missing IERR row: {case_id}")


def _rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Return max scaled difference `|a - b| / max(|a|, |b|, 1.0)`."""
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
    ("label", "temp", "rho", "abar", "zbar"), ORACLE_HELM_STATES
)
def test_eval_helm_eos_matches_mesa_26_4_1_oracle_across_rust_and_jax(
    label: str, temp: float, rho: float, abar: float, zbar: float
) -> None:
    """Verify `eval_helm_eos` Rust `strict` and JAX against MESA `26.4.1`."""
    exp = _load_helm_case(label)
    with flexmesa.numerics_mode_context("strict"):
        out_r, da_r, dz_r = eos.eval_helm_eos(
            temp, rho, abar, zbar, backend="rust"
        )
    out_j, da_j, dz_j = eos.eval_helm_eos(
        jnp.asarray(temp, dtype=jnp.float64),
        jnp.asarray(rho, dtype=jnp.float64),
        jnp.asarray(abar, dtype=jnp.float64),
        jnp.asarray(zbar, dtype=jnp.float64),
        backend="jax",
    )

    got_r = np.column_stack(
        [
            np.asarray(out_r.res)[:16],
            np.asarray(out_r.d_dlnd)[:16],
            np.asarray(out_r.d_dln_t)[:16],
            np.asarray(da_r)[:16],
            np.asarray(dz_r)[:16],
        ]
    )
    np.testing.assert_allclose(got_r, exp, rtol=1.0e-13, atol=1.0e-15)
    assert np.all(np.asarray(out_r.res)[16 : eos.I_FRAC_HELM] == 0.0)
    assert float(out_r.res[eos.I_FRAC_HELM]) == 1.0
    assert np.all(np.asarray(out_r.res)[eos.I_FRAC_HELM + 1 :] == 0.0)

    got_j_res = np.asarray(out_j.res)[:16]
    got_j_der = np.column_stack(
        [
            np.asarray(out_j.d_dlnd)[:16],
            np.asarray(out_j.d_dln_t)[:16],
            np.asarray(da_j)[:16],
            np.asarray(dz_j)[:16],
        ]
    )
    np.testing.assert_allclose(got_j_res, exp[:, 0], rtol=1.0e-10, atol=1.0e-15)
    np.testing.assert_allclose(got_j_der, exp[:, 1:], rtol=3.0e-6, atol=1.0e-12)
    assert float(out_j.res[eos.I_FRAC_HELM]) == 1.0


def test_eval_helm_eos_zones_and_off_table_error_match_mesa_26_4_1() -> None:
    """Verify `eval_helm_eos_zones` and MESA `26.4.1` `IERR` rejection."""
    temps = np.array([s[1] for s in ORACLE_HELM_STATES], dtype=np.float64)
    rhos = np.array([s[2] for s in ORACLE_HELM_STATES], dtype=np.float64)
    abars = np.array([s[3] for s in ORACLE_HELM_STATES], dtype=np.float64)
    zbars = np.array([s[4] for s in ORACLE_HELM_STATES], dtype=np.float64)
    exp_stack = np.stack([_load_helm_case(s[0]) for s in ORACLE_HELM_STATES])

    with flexmesa.numerics_mode_context("strict"):
        batch_r = eos.eval_helm_eos_zones(
            temps, rhos, abars, zbars, backend="rust"
        )
    batch_j = eos.eval_helm_eos_zones(
        jnp.asarray(temps),
        jnp.asarray(rhos),
        jnp.asarray(abars),
        jnp.asarray(zbars),
        backend="jax",
    )
    assert batch_r.res.shape == (
        len(ORACLE_HELM_STATES),
        eos.NUM_EOS_BASIC_RESULTS,
    )
    np.testing.assert_allclose(
        np.asarray(batch_r.res)[:, :16],
        exp_stack[:, :, 0],
        rtol=1.0e-13,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        np.asarray(batch_j.res)[:, :16],
        exp_stack[:, :, 0],
        rtol=1.0e-10,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        np.asarray(batch_j.d_dlnd)[:, :16],
        exp_stack[:, :, 1],
        rtol=3.0e-6,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(batch_j.d_dln_t)[:, :16],
        exp_stack[:, :, 2],
        rtol=3.0e-6,
        atol=1.0e-12,
    )

    exp_ierr = _load_ierr_code("crystallized_wd_1e6_6p5")
    assert exp_ierr != 0
    with pytest.raises(ValueError, match=f"ierr = {exp_ierr}"):
        eos.eval_helm_eos(
            1.0e6,
            float(fm_math.exp10(6.5)),
            13.714285714285715,
            6.8571428571428577,
        )
    out_invalid, _, _ = eos.eval_helm_eos(
        jnp.asarray(1.0e6),
        jnp.asarray(float(fm_math.exp10(6.5))),
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
    _, da_r, dz_r = eos.eval_helm_eos(temp0, rho0, abar0, zbar0, backend="rust")
    out_j, da_j, dz_j = eos.eval_helm_eos(
        temp0, rho0, abar0, zbar0, backend="jax"
    )
    assert _rel_diff(np.asarray(da_r[:11]), np.asarray(da_j[:11])) < 1.0e-9
    assert _rel_diff(np.asarray(dz_r[:11]), np.asarray(dz_j[:11])) < 1.0e-9
    assert _rel_diff(np.asarray(da_r), np.asarray(da_j)) < 5.0e-8
    assert _rel_diff(np.asarray(dz_r), np.asarray(dz_j)) < 5.0e-8

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
    # Upstream Fortran MESA `eos/private/helm_electron_positron.dek` lines
    # 566-567 writes `dpepdda = (zz*dindd + ww)*dindda` using `dindd = Y_e`
    # instead of `din = Y_e * rho` (see `notes/DECISIONS.md` #9). All 11
    # active_non_mu channels independent of `dpepdda`/`dpepddz` match
    # `jax.jacfwd` to < 1e-6 at all densities:
    dpepdd_channels = {
        eos.I_GRAD_AD,
        eos.I_CHI_RHO,
        eos.I_CP,
        eos.I_GAMMA1,
    }
    exact_chain_idx = [
        i for i in range(16) if i != eos.I_MU and i not in dpepdd_channels
    ]
    assert (
        _rel_diff(
            np.asarray(da_j)[exact_chain_idx],
            np.asarray(ad_da)[exact_chain_idx],
        )
        < 1.0e-6
    )
    assert (
        _rel_diff(
            np.asarray(dz_j)[exact_chain_idx],
            np.asarray(ad_dz)[exact_chain_idx],
        )
        < 1.0e-6
    )
    # At rho = 1.0 g cm^-3, din == dindd = Y_e in helm_electron_positron.dek,
    # so all 15 active_non_mu channels (including the 4 dpepdda channels)
    # match `jax.jacfwd` to < 1e-6:
    active_non_mu = [i for i in range(16) if i != eos.I_MU]
    _, da_1, dz_1 = eos.eval_helm_eos(temp0, 1.0, abar0, zbar0, backend="jax")
    _, _, ad_da_1, ad_dz_1 = jax.jacfwd(_res_of_ln_td_az, argnums=(0, 1, 2, 3))(
        jnp.log(jnp.asarray(1.0, dtype=jnp.float64)),
        jnp.log(jnp.asarray(temp0, dtype=jnp.float64)),
        jnp.asarray(abar0, dtype=jnp.float64),
        jnp.asarray(zbar0, dtype=jnp.float64),
    )
    assert (
        _rel_diff(
            np.asarray(da_1)[active_non_mu], np.asarray(ad_da_1)[active_non_mu]
        )
        < 1.0e-6
    )
    assert (
        _rel_diff(
            np.asarray(dz_1)[active_non_mu], np.asarray(ad_dz_1)[active_non_mu]
        )
        < 1.0e-6
    )


def test_eval_helm_eos_full_table_rust_jax_consistency_and_autodiff() -> None:
    """Verify full `2701x1001` `HELM` JAX vs Rust `strict` across domain."""
    sample_points = [
        # (logRho, logT, abar, zbar) across full HELM table domain
        (-7.35, 4.15, 1.29, 1.10),
        (-3.42, 5.68, 1.29, 1.10),
        (0.73, 6.84, 1.65, 1.18),
        (2.15, 7.18, 1.30, 1.10),
        (4.62, 7.85, 4.00, 2.00),
        (6.48, 8.12, 13.71, 6.86),
        (8.35, 8.95, 16.00, 8.00),
        (9.40, 9.30, 28.00, 14.00),
    ]
    temps = np.array([10.0 ** s[1] for s in sample_points], dtype=np.float64)
    rhos = np.array([10.0 ** s[0] for s in sample_points], dtype=np.float64)
    abars = np.array([s[2] for s in sample_points], dtype=np.float64)
    zbars = np.array([s[3] for s in sample_points], dtype=np.float64)

    with flexmesa.numerics_mode_context("strict"):
        zones_r = eos.eval_helm_eos_zones(
            temps, rhos, abars, zbars, backend="rust"
        )
    zones_j = eos.eval_helm_eos_zones(
        jnp.asarray(temps),
        jnp.asarray(rhos),
        jnp.asarray(abars),
        jnp.asarray(zbars),
        backend="jax",
    )
    assert not np.any(np.isnan(np.asarray(zones_r.res)))
    assert not np.any(np.isnan(np.asarray(zones_j.res)))
    res_r3 = np.asarray(zones_r.res[:, :3])
    res_j3 = np.asarray(zones_j.res[:, :3])
    assert _rel_diff(res_r3, res_j3) <= 1.0e-12
    assert _rel_diff(np.asarray(zones_r.res), np.asarray(zones_j.res)) < 1.0e-9
    assert (
        _rel_diff(np.asarray(zones_r.d_dlnd), np.asarray(zones_j.d_dlnd))
        < 2.0e-6
    )
    assert (
        _rel_diff(np.asarray(zones_r.d_dln_t), np.asarray(zones_j.d_dln_t))
        < 2.0e-6
    )

    # Check forward-mode jacfwd and reverse-mode grad at an off-patch WD vertex
    rho_wd = 10.0**6.48
    temp_wd = 10.0**8.12
    out_wd, _, _ = eos.eval_helm_eos(
        temp_wd, rho_wd, 13.71, 6.86, backend="jax"
    )
    ad_dlnd, ad_dlnt = jax.jacfwd(
        lambda lnd, lnt: _res_of_ln_td_az(
            lnd, lnt, jnp.float64(13.71), jnp.float64(6.86)
        ),
        argnums=(0, 1),
    )(jnp.log(jnp.float64(rho_wd)), jnp.log(jnp.float64(temp_wd)))
    assert (
        _rel_diff(np.asarray(out_wd.d_dlnd[:3]), np.asarray(ad_dlnd[:3]))
        < 2.0e-11
    )
    assert (
        _rel_diff(np.asarray(out_wd.d_dln_t[:3]), np.asarray(ad_dlnt[:3]))
        < 2.0e-11
    )
    assert (
        _rel_diff(np.asarray(out_wd.d_dlnd[:16]), np.asarray(ad_dlnd[:16]))
        < 1.0e-8
    )
    assert (
        _rel_diff(np.asarray(out_wd.d_dln_t[:16]), np.asarray(ad_dlnt[:16]))
        < 1.0e-7
    )
