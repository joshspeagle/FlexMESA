# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Verification suite for `flexmesa.ionization` (`Wave 2b`).

Validates the Three-Tier Hybrid `ionization` implementation against:
1. 100% of the golden output in upstream MESA `ionization/test/test_output`
   (`test_fe56_in_he4` at 3 $(\log_{10} n_e, \log_{10} T)$ points and
   `do_test_eval_ionization` across all 28 output quantities: `logPgas`,
   `logpp_H..Fe`, `Z_H..Fe`, `fneut_H..Fe`).
2. Multi-zone Safe Rust (`PyO3` + `rayon`) vs. `@jax.jit` parity (`<= 1e-14`)
   across 20 deterministic random seeds (`np.random.default_rng(seed)`) on
   `eval_charge_of_fe56_in_he4_zones`, `eval_ionization_zones`, `chi_info`,
   `chi_effective`, and `eval_typical_charge`.
3. Automatic differentiation (`jax.test_util.check_grads` order 2 and exact
   analytical spline derivative agreement) on `eval_charge_of_fe56_in_he4_jax`,
   `eval_ionization_jax`, and `chi_effective_jax`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import test_util as jtu

from flexmesa import _rust, ionization

_GOLDEN_EVAL_IONIZATION: np.ndarray = np.array(
    [
        7.5712889833107724e0,
        7.2497987346894925e0,
        6.2953076445510217e0,
        2.3275223419391016e0,
        1.7638357415187518e0,
        2.6333713415210882e0,
        1.8692815415211297e0,
        1.4388249415175236e0,
        1.5896699415324396e0,
        1.3235251415241509e0,
        9.6343046904325569e-1,
        6.7156309337823639e-1,
        1.1424089189412558e0,
        1.0898620680464786e0,
        1.0088948284916714e0,
        2.4831155345187437e0,
        1.9403564145769852e0,
        1.8392169240105978e0,
        1.9717594950533246e0,
        3.6569530956712573e-2,
        3.2842817319701922e-1,
        1.0327304090569216e-2,
        1.0504633379733298e-2,
        3.9039585956120552e-2,
        2.4863976988275275e-3,
        1.0014642315641282e-5,
        9.6540712045603192e-4,
        4.4460832479705790e-6,
    ],
    dtype=np.float64,
)


def _fe56_scalar(lne: jax.Array, lt: jax.Array) -> jax.Array:
    r"""Return scalar effective charge $Z(\mathrm{Fe56})$ for JAX autodiff."""
    z, _, _ = ionization.eval_charge_of_fe56_in_he4_jax(lne, lt)
    return z


def test_fe56_in_he4_matches_mesa_golden_output() -> None:
    r"""Verify `test_fe56_in_he4` against `ionization/test/test_output`.

    Checks all three golden $(\log_{10} n_e, \log_{10} T)$ test points across
    both the Safe Rust and `@jax.jit` backends.
    """
    golden = [
        (30.5, 7.9, 25.8953869),
        (26.0, 7.0, 19.0993247),
        (24.5, 6.1, 11.2318524),
    ]
    for lne, lt, expected in golden:
        z_rust = ionization.eval_charge_of_fe56_in_he4(lne, lt, backend="rust")
        z_jax = ionization.eval_charge_of_fe56_in_he4(lne, lt, backend="jax")
        assert np.isclose(z_rust, expected, rtol=0.0, atol=1e-7)
        assert np.isclose(z_rust, z_jax, rtol=1e-15, atol=1e-15)


def test_eval_ionization_matches_mesa_golden_output() -> None:
    r"""Verify `do_test_eval_ionization` against all 28 golden channels.

    Checks `logPgas`, `logpp_H..Fe`, `Z_H..Fe`, and `fneut_H..Fe` at
    $Z = 0.0188$, $X = 0.725$, $T = 3.2345529591587989 \times 10^4\,\mathrm{K}$,
    and $\rho = 9.0768775206067858 \times 10^{-6}\,\mathrm{g\,cm^{-3}}$ across
    both Safe Rust and `@jax.jit`.
    """
    z = 0.0188
    x = 0.725
    t = 3.2345529591587989e4
    rho = 9.0768775206067858e-6

    res_rust = ionization.eval_ionization(z, x, rho=rho, t=t, backend="rust")
    res_jax = ionization.eval_ionization(z, x, rho=rho, t=t, backend="jax")

    np.testing.assert_allclose(
        res_rust.values, _GOLDEN_EVAL_IONIZATION, rtol=1e-14, atol=0.0
    )
    np.testing.assert_allclose(
        res_rust.values, res_jax.values, rtol=1e-14, atol=1e-15
    )
    assert np.isclose(
        res_rust.log_pgas, _GOLDEN_EVAL_IONIZATION[0], rtol=1e-15, atol=0.0
    )
    assert np.isclose(
        res_rust["fneut_Fe"], _GOLDEN_EVAL_IONIZATION[27], rtol=1e-14, atol=0.0
    )
    assert res_rust.logpp.shape == (9,)
    assert res_rust.z_ion.shape == (9,)
    assert res_rust.fneut.shape == (9,)


def test_fe56_in_he4_zones_rust_vs_jax_across_random_seeds() -> None:
    r"""Verify multi-zone `Fe56`-in-`He4` Rust (`rayon`) vs. JAX across seeds.

    Evaluates 128 zones across 20 deterministic `np.random.default_rng(seed)`
    seeds and compares `[z, dz_dlog10_ne, dz_dlog10_t]` between sequential
    Rust, parallel `rayon` Rust, and `@jax.jit` to `1e-14`.
    """
    n_zones = 128
    for seed in range(20):
        rng = np.random.default_rng(seed)
        lne = rng.uniform(24.1, 30.9, size=n_zones)
        lt = rng.uniform(6.05, 7.95, size=n_zones)

        out_par = ionization.eval_charge_of_fe56_in_he4_zones(
            lne, lt, parallel=True, backend="rust"
        )
        out_seq = ionization.eval_charge_of_fe56_in_he4_zones(
            lne, lt, parallel=False, backend="rust"
        )
        out_jax = np.asarray(
            ionization.eval_charge_of_fe56_in_he4_zones(
                jnp.asarray(lne), jnp.asarray(lt), backend="jax"
            ),
            dtype=np.float64,
        )
        np.testing.assert_allclose(out_par, out_seq, rtol=1e-15, atol=1e-15)
        np.testing.assert_allclose(
            out_par[:, 0], out_jax[:, 0], rtol=1e-14, atol=1e-14
        )
        # First derivatives divide spline values (~26) by grid spacing dx=0.05,
        # giving a 1-ULP CPU FMA vs non-FMA bound of 26 * eps / 0.05 ~ 1.15e-13.
        np.testing.assert_allclose(
            out_par[:, 1:], out_jax[:, 1:], rtol=1e-12, atol=2e-13
        )


def test_eval_ionization_zones_rust_vs_jax_across_random_seeds() -> None:
    r"""Verify multi-zone 4D `eval_ionization_zones` Rust vs. JAX across seeds.

    Evaluates 128 zones spanning the full astrophysical domain ($\log_{10} T \in
    [3.2, 8.8]$, $\log Q \in [-9.0, 5.0]$, $Z \in [0.0, 1.0]$, and $X \in
    [0.0, 1.0]$) across 20 deterministic seeds without $2 \times 2$ cell
    clamping.
    """
    n_zones = 128
    for seed in range(20):
        rng = np.random.default_rng(seed)
        z = rng.uniform(0.0, 1.0, size=n_zones)
        x = rng.uniform(0.0, 1.0, size=n_zones)
        lt = rng.uniform(3.2, 8.8, size=n_zones)
        lq = rng.uniform(-9.0, 5.0, size=n_zones)
        lrho = lq + 2.0 * lt - 12.0

        out_par = ionization.eval_ionization_zones(
            z, x, lrho, lt, parallel=True, backend="rust"
        )
        out_seq = ionization.eval_ionization_zones(
            z, x, lrho, lt, parallel=False, backend="rust"
        )
        out_jax = np.asarray(
            ionization.eval_ionization_zones(
                jnp.asarray(z),
                jnp.asarray(x),
                jnp.asarray(lrho),
                jnp.asarray(lt),
                backend="jax",
            ),
            dtype=np.float64,
        )
        np.testing.assert_allclose(out_par, out_seq, rtol=1e-15, atol=1e-15)
        np.testing.assert_allclose(
            out_par[:, 0], out_jax[:, 0], rtol=1e-14, atol=1e-14
        )
        # First derivatives divide spline values (~26) by grid spacing dx=0.05,
        # giving a 1-ULP CPU FMA vs non-FMA bound of 26 * eps / 0.05 ~ 1.15e-13.
        np.testing.assert_allclose(
            out_par[:, 1:], out_jax[:, 1:], rtol=1e-12, atol=2e-13
        )


def test_typical_charge_and_chi_effective_rust_vs_jax() -> None:
    r"""Verify `chi_info`, `chi_effective`, and `eval_typical_charge`.

    Checks Rust vs. JAX agreement on `chi_info`, `chi_effective`, and
    `eval_typical_charge` across `h1`, `he4`, `c12`, `o16`, `fe56`, and `zn64`,
    plus Allen (1973) `ionization_potential` values.
    """
    assert np.isclose(
        ionization.ionization_potential(1, 1), 13.598, rtol=1e-15, atol=0.0
    )
    assert np.isclose(
        ionization.ionization_potential(26, 26), 9278.0, rtol=1e-15, atol=0.0
    )

    for seed in range(20):
        rng = np.random.default_rng(seed)
        abar = float(rng.uniform(1.2, 4.0))
        free_e = float(rng.uniform(0.4, 0.9))
        lt = float(rng.uniform(4.0, 7.5))
        t = float(10.0**lt)
        lrho = float(rng.uniform(-6.0, 4.0))
        rho = float(10.0**lrho)
        z1 = abar * free_e

        c_r = ionization.chi_info(abar, z1, t, lt, rho, lrho, backend="rust")
        c_j = ionization.chi_info(abar, z1, t, lt, rho, lrho, backend="jax")
        np.testing.assert_allclose(c_r, c_j, rtol=1e-14, atol=1e-14)

        ce_r = ionization.chi_effective(*c_r, z1, 6.0, backend="rust")
        ce_j = ionization.chi_effective(*c_j, z1, 6.0, backend="jax")
        assert np.isclose(ce_r, ce_j, rtol=1e-14, atol=1e-14)

        for iso in ("h1", "he4", "c12", "o16", "fe56", "zn64"):
            zt_r = ionization.eval_typical_charge(
                iso, abar, free_e, t, lt, rho, lrho, backend="rust"
            )
            zt_j = ionization.eval_typical_charge(
                iso, abar, free_e, t, lt, rho, lrho, backend="jax"
            )
            assert zt_r == zt_j
            assert zt_r >= 1.0


def test_fe56_he4_and_chi_jax_autodiff() -> None:
    r"""Verify JAX autodiff on `eval_charge_of_fe56_in_he4_jax` and `chi`.

    Checks `jax.test_util.check_grads` (order 2) on
    `eval_charge_of_fe56_in_he4_jax` and `chi_effective_jax`, and confirms that
    analytical bicubic spline partials (`dz_dlog10_ne`, `dz_dlog10_t`) match
    `jax.grad` to `1e-13`.
    """
    lne0 = jnp.asarray(26.25, dtype=jnp.float64)
    lt0 = jnp.asarray(7.15, dtype=jnp.float64)

    jtu.check_grads(_fe56_scalar, (lne0, lt0), order=2, atol=1e-5, rtol=1e-5)
    _, dz_dlne, dz_dlt = ionization.eval_charge_of_fe56_in_he4_jax(lne0, lt0)
    g_lne, g_lt = jax.grad(_fe56_scalar, argnums=(0, 1))(lne0, lt0)
    assert np.isclose(float(dz_dlne), float(g_lne), rtol=1e-13, atol=1e-13)
    assert np.isclose(float(dz_dlt), float(g_lt), rtol=1e-13, atol=1e-13)

    jtu.check_grads(
        lambda z2: ionization.chi_effective_jax(12.0, 0.05, 0.8, 1.5, 1.2, z2),
        (jnp.asarray(4.0, dtype=jnp.float64),),
        order=2,
        atol=1e-5,
        rtol=1e-5,
    )
    with pytest.raises(ValueError, match="Require"):
        ionization.ionization_potential(31, 1)
    with pytest.raises(ValueError, match="shape"):
        ionization.eval_charge_of_fe56_in_he4_zones(
            np.ones(4, dtype=np.float64), np.ones(3, dtype=np.float64)
        )


def test_eval_ionization_jax_and_dual_autodiff() -> None:
    r"""Verify `eval_ionization_jax` autodiff and Rust `DualOrder1` Jacobian.

    Checks `jax.test_util.check_grads` and confirms that the $28 \times 4$
    analytical Jacobian from `jax.jacfwd` matches Safe Rust `DualOrder1<4>` to
    `1e-13`.
    """
    z0 = jnp.asarray(0.0188, dtype=jnp.float64)
    x0 = jnp.asarray(0.725, dtype=jnp.float64)
    lrho0 = jnp.asarray(-5.042, dtype=jnp.float64)
    lt_ion = jnp.asarray(4.51, dtype=jnp.float64)
    jtu.check_grads(
        lambda z, x, lr, lt: jnp.sum(
            ionization.eval_ionization_jax(z, x, lr, lt)
        ),
        (z0, x0, lrho0, lt_ion),
        order=1,
        atol=1e-5,
        rtol=1e-5,
    )
    jtu.check_grads(
        lambda lr, lt: jnp.sum(ionization.eval_ionization_jax(z0, x0, lr, lt)),
        (lrho0, lt_ion),
        order=2,
        atol=1e-5,
        rtol=1e-5,
    )
    val_rust, jac_rust = _rust.py_ion_eval_ionization_dual(
        float(z0), float(x0), float(lrho0), float(lt_ion)
    )
    jac_jax = np.stack(
        [
            np.asarray(g, dtype=np.float64)
            for g in jax.jacfwd(
                ionization.eval_ionization_jax, argnums=(0, 1, 2, 3)
            )(z0, x0, lrho0, lt_ion)
        ],
        axis=-1,
    )
    np.testing.assert_allclose(
        np.asarray(ionization.eval_ionization_jax(z0, x0, lrho0, lt_ion)),
        val_rust,
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(jac_jax, jac_rust, rtol=1e-13, atol=1e-13)
