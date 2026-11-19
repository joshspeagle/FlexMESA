# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for Wave 3a.2 (`HELM` equation of state in Rust, Python, and JAX)."""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import flexmesa
from flexmesa import _rust, eos

_EOS_ORACLES_TSV = (
    pathlib.Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "fortran_mesa"
    / "eos_oracles"
    / "mesa_26_4_1_eos_oracles.tsv"
)

# `(label, temp_or_logT, rho_or_logRho, is_log10, abar, zbar)`
ORACLE_HELM_STATES: list[tuple[str, float, float, bool, float, float]] = [
    ("pure_he_2e8_100", 2.0e8, 100.0, False, 4.0, 2.0),
    ("pure_h_2e8_100", 2.0e8, 100.0, False, 1.0, 1.0),
    (
        "solar_2e8_100",
        2.0e8,
        100.0,
        False,
        1.2966353410255222,
        1.1021400398716938,
    ),
    (
        "test_components",
        6.0,
        3.5,
        True,
        2.990964711670598,
        1.6749402385355348,
    ),
    ("sample_eos_c12", 1.0e9, 1.0e4, False, 12.0, 6.0),
    ("mixed_z_2e8_100", 2.0e8, 100.0, False, 1.72, 1.15),
    ("solar_1e6_100", 1.0e6, 100.0, False, 1.29818593, 1.10539606),
    (
        "py_test_components",
        6.1,
        0.1,
        True,
        1.29666341,
        1.10668478,
    ),
    ("py_sample_eos_c12", 1.0e9, 1.0e6, False, 12.0, 6.0),
]

ORACLE_HELM_DOMAIN_POINTS: list[tuple[str, float, float, float, float]] = [
    ("helm_dom_0", -7.35, 4.15, 1.29, 1.10),
    ("helm_dom_1", -3.42, 5.68, 1.29, 1.10),
    ("helm_dom_2", 0.73, 6.84, 1.65, 1.18),
    ("helm_dom_3", 2.15, 7.18, 1.30, 1.10),
    ("helm_dom_4", 4.62, 7.85, 4.00, 2.00),
    ("helm_dom_5", 6.48, 8.12, 13.71, 6.86),
    ("helm_dom_6", 8.35, 8.95, 16.00, 8.00),
    ("helm_dom_7", 9.40, 9.30, 28.00, 14.00),
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


def _assert_unfloored(
    got: np.ndarray,
    ref: np.ndarray,
    *,
    rtol: float,
    zero_atol: float = 1.0e-14,
    near_zero_cutoff: float = 1.0e-12,
) -> None:
    """Assert unfloored `|got - ref| / |ref| <= rtol` (`zero_atol` near 0)."""
    got_a = np.asarray(got, dtype=np.float64)
    ref_a = np.asarray(ref, dtype=np.float64)
    assert got_a.shape == ref_a.shape
    near_zero = np.abs(ref_a) <= near_zero_cutoff
    if np.any(near_zero):
        max_abs = float(np.max(np.abs(got_a[near_zero] - ref_a[near_zero])))
        assert max_abs <= zero_atol, (
            f"Near-zero abs diff {max_abs:.3e} > {zero_atol:.1e}"
        )
    non_zero = ~near_zero
    if np.any(non_zero):
        rel = np.abs(got_a[non_zero] - ref_a[non_zero]) / np.abs(
            ref_a[non_zero]
        )
        max_rel = float(np.max(rel))
        assert max_rel <= rtol, f"Unfloored rel diff {max_rel:.3e} > {rtol:.1e}"


def _resolve_td(
    t_val: float, r_val: float, is_log10: bool
) -> tuple[float, float]:
    """Evaluate `(temp, rho)` using MESA `CRMATH` `exp10` when `is_log10`."""
    if is_log10:
        return float(_rust.exp10(t_val)), float(_rust.exp10(r_val))
    return t_val, r_val


def _res_of_ln_td_az(
    ln_d: jax.Array, ln_t: jax.Array, abar: jax.Array, zbar: jax.Array
) -> jax.Array:
    """Return `res` from `eval_helm_eos` at `(ln_d, ln_t, abar, zbar)`."""
    out, _, _ = eos.eval_helm_eos(
        jnp.exp(ln_t), jnp.exp(ln_d), abar, zbar, backend="jax"
    )
    return jnp.asarray(out.res)


@pytest.mark.parametrize(
    ("label", "t_in", "r_in", "is_log10", "abar", "zbar"), ORACLE_HELM_STATES
)
def test_eval_helm_eos_matches_mesa_26_4_1_oracle_across_rust_and_jax(
    label: str,
    t_in: float,
    r_in: float,
    is_log10: bool,
    abar: float,
    zbar: float,
) -> None:
    """Verify `eval_helm_eos` Rust `strict` and JAX against MESA `26.4.1`."""
    exp = _load_helm_case(label)
    with flexmesa.numerics_mode_context("strict"):
        temp, rho = _resolve_td(t_in, r_in, is_log10)
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
    _assert_unfloored(got_r, exp, rtol=3.0e-13, zero_atol=1.0e-14)
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
    _assert_unfloored(got_j_res, exp[:, 0], rtol=1.0e-10, zero_atol=1.0e-14)
    _assert_unfloored(got_j_der, exp[:, 1:], rtol=5.0e-6, zero_atol=1.0e-13)
    assert float(out_j.res[eos.I_FRAC_HELM]) == 1.0


def test_eval_helm_eos_zones_and_off_table_error_match_mesa_26_4_1() -> None:
    """Verify `eval_helm_eos_zones` and MESA `26.4.1` `IERR` rejection."""
    with flexmesa.numerics_mode_context("strict"):
        td_pairs = [_resolve_td(s[1], s[2], s[3]) for s in ORACLE_HELM_STATES]
        temps = np.array([p[0] for p in td_pairs], dtype=np.float64)
        rhos = np.array([p[1] for p in td_pairs], dtype=np.float64)
        abars = np.array([s[4] for s in ORACLE_HELM_STATES], dtype=np.float64)
        zbars = np.array([s[5] for s in ORACLE_HELM_STATES], dtype=np.float64)
        exp_stack = np.stack(
            [_load_helm_case(s[0]) for s in ORACLE_HELM_STATES]
        )
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
    _assert_unfloored(
        np.asarray(batch_r.res)[:, :16],
        exp_stack[:, :, 0],
        rtol=3.0e-13,
        zero_atol=1.0e-14,
    )
    _assert_unfloored(
        np.asarray(batch_r.d_dlnd)[:, :16],
        exp_stack[:, :, 1],
        rtol=3.0e-13,
        zero_atol=1.0e-14,
    )
    _assert_unfloored(
        np.asarray(batch_r.d_dln_t)[:, :16],
        exp_stack[:, :, 2],
        rtol=3.0e-13,
        zero_atol=1.0e-14,
    )
    _assert_unfloored(
        np.asarray(batch_j.res)[:, :16],
        exp_stack[:, :, 0],
        rtol=1.0e-10,
        zero_atol=1.0e-14,
    )
    _assert_unfloored(
        np.asarray(batch_j.d_dlnd)[:, :16],
        exp_stack[:, :, 1],
        rtol=5.0e-6,
        zero_atol=1.0e-13,
    )
    _assert_unfloored(
        np.asarray(batch_j.d_dln_t)[:, :16],
        exp_stack[:, :, 2],
        rtol=5.0e-6,
        zero_atol=1.0e-13,
    )

    exp_ierr = _load_ierr_code("crystallized_wd_1e6_6p5")
    assert exp_ierr != 0
    with flexmesa.numerics_mode_context("strict"):
        rho_cryst = float(_rust.exp10(6.5))
    with pytest.raises(ValueError, match=f"ierr = {exp_ierr}"):
        eos.eval_helm_eos(
            1.0e6,
            rho_cryst,
            13.714285714285715,
            6.8571428571428577,
        )
    out_invalid, _, _ = eos.eval_helm_eos(
        jnp.asarray(1.0e6),
        jnp.asarray(rho_cryst),
        jnp.asarray(13.714285714285715),
        jnp.asarray(6.8571428571428577),
        backend="jax",
    )
    assert bool(jnp.all(jnp.isnan(out_invalid.res)))


def test_eval_helm_eos_jax_autodiff_matches_analytical_derivatives() -> None:
    """Verify unfloored `jax.jacfwd` of `eval_helm_eos` matches partials."""
    temp0 = 2.0e8
    rho0 = 100.0
    abar0 = 1.72
    zbar0 = 1.15
    with flexmesa.numerics_mode_context("strict"):
        _, da_r, dz_r = eos.eval_helm_eos(
            temp0, rho0, abar0, zbar0, backend="rust"
        )
    out_j, da_j, dz_j = eos.eval_helm_eos(
        temp0, rho0, abar0, zbar0, backend="jax"
    )
    _assert_unfloored(np.asarray(da_j)[:16], np.asarray(da_r)[:16], rtol=5.0e-8)
    _assert_unfloored(np.asarray(dz_j)[:16], np.asarray(dz_r)[:16], rtol=5.0e-8)

    ad_dlnd, ad_dlnt, ad_da, ad_dz = jax.jacfwd(
        _res_of_ln_td_az, argnums=(0, 1, 2, 3)
    )(
        jnp.log(jnp.asarray(rho0, dtype=jnp.float64)),
        jnp.log(jnp.asarray(temp0, dtype=jnp.float64)),
        jnp.asarray(abar0, dtype=jnp.float64),
        jnp.asarray(zbar0, dtype=jnp.float64),
    )

    _assert_unfloored(
        np.asarray(out_j.d_dlnd)[:16], np.asarray(ad_dlnd)[:16], rtol=1.0e-7
    )
    _assert_unfloored(
        np.asarray(out_j.d_dln_t)[:16], np.asarray(ad_dlnt)[:16], rtol=1.0e-7
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
    _assert_unfloored(
        np.asarray(da_j)[primary_idx], np.asarray(ad_da)[primary_idx], rtol=1e-9
    )
    _assert_unfloored(
        np.asarray(dz_j)[primary_idx], np.asarray(ad_dz)[primary_idx], rtol=1e-9
    )

    # Upstream Fortran MESA `eos/private/helm_electron_positron.dek` lines
    # 566-567 writes `dpepdda = (zz*dindd + ww)*dindda` using `dindd = Y_e`
    # instead of `din = Y_e * rho` (see `notes/DECISIONS.md` #9). All 11
    # active_non_mu channels independent of `dpepdda`/`dpepddz` match
    # `jax.jacfwd` at all densities:
    dpepdd_channels = {
        eos.I_GRAD_AD,
        eos.I_CHI_RHO,
        eos.I_CP,
        eos.I_GAMMA1,
    }
    exact_chain_idx = [
        i for i in range(16) if i != eos.I_MU and i not in dpepdd_channels
    ]
    _assert_unfloored(
        np.asarray(da_j)[exact_chain_idx],
        np.asarray(ad_da)[exact_chain_idx],
        rtol=1.0e-6,
    )
    _assert_unfloored(
        np.asarray(dz_j)[exact_chain_idx],
        np.asarray(ad_dz)[exact_chain_idx],
        rtol=1.0e-6,
    )

    # At rho = 1.0 g cm^-3, din == dindd = Y_e in helm_electron_positron.dek,
    # so all 15 active_non_mu channels match `jax.jacfwd`:
    active_non_mu = [i for i in range(16) if i != eos.I_MU]
    _, da_1, dz_1 = eos.eval_helm_eos(temp0, 1.0, abar0, zbar0, backend="jax")
    _, _, ad_da_1, ad_dz_1 = jax.jacfwd(_res_of_ln_td_az, argnums=(0, 1, 2, 3))(
        jnp.log(jnp.asarray(1.0, dtype=jnp.float64)),
        jnp.log(jnp.asarray(temp0, dtype=jnp.float64)),
        jnp.asarray(abar0, dtype=jnp.float64),
        jnp.asarray(zbar0, dtype=jnp.float64),
    )
    _assert_unfloored(
        np.asarray(da_1)[active_non_mu],
        np.asarray(ad_da_1)[active_non_mu],
        rtol=5.0e-6,
    )
    _assert_unfloored(
        np.asarray(dz_1)[active_non_mu],
        np.asarray(ad_dz_1)[active_non_mu],
        rtol=5.0e-6,
    )


def test_eval_helm_eos_full_table_matches_mesa_26_4_1_and_autodiff() -> None:
    """Verify `helm_dom_0..7` across MESA `26.4.1`, Rust `strict`, and JAX."""
    with flexmesa.numerics_mode_context("strict"):
        temps = np.array(
            [float(_rust.exp10(s[2])) for s in ORACLE_HELM_DOMAIN_POINTS],
            dtype=np.float64,
        )
        rhos = np.array(
            [float(_rust.exp10(s[1])) for s in ORACLE_HELM_DOMAIN_POINTS],
            dtype=np.float64,
        )
        abars = np.array(
            [s[3] for s in ORACLE_HELM_DOMAIN_POINTS], dtype=np.float64
        )
        zbars = np.array(
            [s[4] for s in ORACLE_HELM_DOMAIN_POINTS], dtype=np.float64
        )
        exp_dom = np.stack(
            [_load_helm_case(s[0]) for s in ORACLE_HELM_DOMAIN_POINTS]
        )

        for idx, (label, _, _, ab, zb) in enumerate(ORACLE_HELM_DOMAIN_POINTS):
            out_r, da_r, dz_r = eos.eval_helm_eos(
                float(temps[idx]), float(rhos[idx]), ab, zb, backend="rust"
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
            _assert_unfloored(
                got_r, exp_dom[idx], rtol=3.0e-13, zero_atol=1.0e-14
            )
            assert float(out_r.res[eos.I_FRAC_HELM]) == 1.0, label

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
    _assert_unfloored(
        np.asarray(zones_r.res)[:, :16],
        exp_dom[:, :, 0],
        rtol=3.0e-13,
        zero_atol=1.0e-14,
    )
    _assert_unfloored(
        np.asarray(zones_j.res)[:, :3],
        exp_dom[:, :, 0][:, :3],
        rtol=1.0e-11,
        zero_atol=1.0e-14,
    )
    _assert_unfloored(
        np.asarray(zones_j.res)[:, :16],
        exp_dom[:, :, 0],
        rtol=5.0e-10,
        zero_atol=1.0e-14,
    )
    _assert_unfloored(
        np.asarray(zones_j.d_dlnd)[:, :16],
        exp_dom[:, :, 1],
        rtol=5.0e-5,
        zero_atol=1.0e-12,
    )
    _assert_unfloored(
        np.asarray(zones_j.d_dln_t)[:, :16],
        exp_dom[:, :, 2],
        rtol=5.0e-6,
        zero_atol=1.0e-12,
    )

    # Check forward-mode jacfwd at the off-patch WD vertex (`helm_dom_5`)
    rho_wd = float(rhos[5])
    temp_wd = float(temps[5])
    out_wd, _, _ = eos.eval_helm_eos(
        temp_wd, rho_wd, 13.71, 6.86, backend="jax"
    )
    ad_dlnd, ad_dlnt = jax.jacfwd(
        lambda lnd, lnt: _res_of_ln_td_az(
            lnd, lnt, jnp.float64(13.71), jnp.float64(6.86)
        ),
        argnums=(0, 1),
    )(jnp.log(jnp.float64(rho_wd)), jnp.log(jnp.float64(temp_wd)))
    _assert_unfloored(
        np.asarray(out_wd.d_dlnd)[:3], np.asarray(ad_dlnd)[:3], rtol=1.0e-9
    )
    _assert_unfloored(
        np.asarray(out_wd.d_dln_t)[:3], np.asarray(ad_dlnt)[:3], rtol=1.0e-9
    )
    _assert_unfloored(
        np.asarray(out_wd.d_dlnd)[:16], np.asarray(ad_dlnd)[:16], rtol=1.0e-7
    )
    _assert_unfloored(
        np.asarray(out_wd.d_dln_t)[:16], np.asarray(ad_dlnt)[:16], rtol=1.0e-6
    )
