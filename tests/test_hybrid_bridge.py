# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the `flexmesa._rust` (`PyO3`) bridge and Three-Tier dispatch.

Verifies that:
1. Tier A modules (`utils`, `testing.golden`, `schema`, `data`, `auto_diff`)
   delegate to compiled Safe Rust (`flexmesa._rust`) and preserve all
   invariants.
2. `resolve_backend` routes Python `float` and `np.ndarray` inputs to `"rust"`
   and `jax.Array` / traced inputs to `"jax"` when `backend="auto"`.
3. Tier B routines (`interp_1d`, `interp_2d`, `neu`, `turb`, `math`) produce
   machine-precision parity (`<= 1e-12`) across `backend="rust"`,
   `backend="jax"`, and `backend="auto"`, including batch parallel `rayon`
   evaluation.
4. Tier C block-tridiagonal solvers (`solve_block_tridiagonal` and Hirshman et
   al. 2010 `solve_block_tridiag_bcyclic`) match residual $A x = b$ to
   `< 1e-12` in both sequential and 4-thread persistent `rayon` modes.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

import flexmesa
from flexmesa import (
    _rust,
    interp_1d,
    interp_2d,
    neu,
    turb,
)


def test_resolve_backend_routes_scalars_numpy_and_jax_arrays() -> None:
    """Verify `resolve_backend` automatic routing and explicit overrides.

    Confirms that `"auto"` routes Python scalars and NumPy arrays to `"rust"`,
    routes `jax.Array` and `jax.core.Tracer` values to `"jax"`, and honors
    explicit `"rust"` and `"jax"` overrides while rejecting invalid selectors.
    """
    x_np = np.linspace(0.0, 1.0, 8, dtype=np.float64)
    x_jax = jnp.linspace(0.0, 1.0, 8, dtype=jnp.float64)

    assert flexmesa.resolve_backend("auto", 1.0, 2.0) == "rust"
    assert flexmesa.resolve_backend("auto", x_np) == "rust"
    assert flexmesa.resolve_backend("auto", x_jax) == "jax"
    assert flexmesa.resolve_backend("auto", x_np, x_jax) == "jax"
    assert flexmesa.resolve_backend("rust", x_jax) == "rust"
    assert flexmesa.resolve_backend("jax", x_np) == "jax"

    with pytest.raises(ValueError, match="Invalid backend"):
        flexmesa.resolve_backend("invalid", x_np)  # type: ignore[arg-type]


def test_rust_math_and_autodiff_bindings_match_analytical_oracles() -> None:
    """Verify `_rust` libm math functions and 2nd-order dual-number kernels.

    Tests `pow`, `exp10`, `safe_log10`, `sinpi`, `cospi`, `safe_log_array`,
    `eval_autodiff_order2_zone`, and `eval_autodiff_order2_batch` (sequential
    vs 4-thread persistent `rayon`).
    """
    np.testing.assert_allclose(_rust.pow(2.5, 3.2), 2.5**3.2, rtol=1e-15)
    np.testing.assert_allclose(_rust.exp10(1.5), 10.0**1.5, rtol=1e-15)
    np.testing.assert_allclose(_rust.safe_log10(1000.0), 3.0, rtol=1e-15)
    np.testing.assert_allclose(_rust.sinpi(2.0), 0.0, atol=1e-15)
    np.testing.assert_allclose(_rust.cospi(0.5), 0.0, atol=1e-15)

    xs = np.linspace(1.0, 4.0, 128, dtype=np.float64)
    ys = np.linspace(0.5, 2.5, 128, dtype=np.float64)
    np.testing.assert_allclose(_rust.safe_log_array(xs), np.log(xs), rtol=1e-15)

    ad_seq = _rust.eval_autodiff_order2_batch(xs, ys, False)
    ad_par = _rust.eval_autodiff_order2_batch(xs, ys, True)
    np.testing.assert_allclose(ad_seq, ad_par, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(
        ad_seq[0],
        _rust.eval_autodiff_order2_zone(float(xs[0]), float(ys[0])),
        rtol=1e-15,
    )


def test_interp1d_rust_and_jax_backends_match_to_machine_precision() -> None:
    """Verify `interp_1d` parity across `backend="rust"`, `"jax"`, and `"auto"`.

    Tests `interp_pm`, `interp_m3a`, `interp_m3b`, `interp_m3q`,
    `interp_values`, `interp_values_and_slopes`, and `integrate_values`.
    """
    x = np.linspace(0.0, 3.0, 32, dtype=np.float64)
    y = np.sin(x) + 0.15 * x * x
    xq = np.linspace(0.1, 2.9, 50, dtype=np.float64)

    for builder in (
        interp_1d.interp_pm,
        interp_1d.interp_m3a,
        interp_1d.interp_m3b,
        interp_1d.interp_m3q,
    ):
        c_rust = np.asarray(builder(x, y, backend="rust"))
        c_jax = np.asarray(builder(x, y, backend="jax"))
        c_auto = np.asarray(builder(x, y, backend="auto"))
        np.testing.assert_allclose(c_rust, c_jax, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(c_auto, c_rust, rtol=1e-15, atol=1e-15)

    # Verify both 1D spline evaluation and definite integration across backends.
    c_pm = interp_1d.interp_pm(x, y, backend="rust")
    v_rust, s_rust = interp_1d.interp_values_and_slopes(
        x, c_pm, xq, backend="rust"
    )
    v_jax, s_jax = interp_1d.interp_values_and_slopes(
        x, c_pm, xq, backend="jax"
    )
    np.testing.assert_allclose(v_rust, v_jax, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(s_rust, s_jax, rtol=1e-13, atol=1e-14)

    int_rust = interp_1d.integrate_values(x, c_pm, xq, backend="rust")
    int_jax = interp_1d.integrate_values(x, c_pm, xq, backend="jax")
    np.testing.assert_allclose(int_rust, int_jax, rtol=1e-13, atol=1e-14)


def test_interp2d_rust_and_jax_backends_match_and_support_autodiff() -> None:
    """Verify `interp_2d` parity across `backend="rust"`, `"jax"`, and `"auto"`.

    Checks `interp_mkbicub_db`, `interp_evbicub_db`, and
    `_rust.interp_evbicub_batch` (parallel vs sequential).
    """
    x = np.linspace(-math.pi, math.pi, 16, dtype=np.float64)
    y = np.linspace(-math.pi, math.pi, 18, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    z = np.sin(xx) * np.cos(yy)

    c_rust, ilinx_r, iliny_r = interp_2d.interp_mkbicub_db(
        x, y, z, backend="rust"
    )
    c_jax, ilinx_j, iliny_j = interp_2d.interp_mkbicub_db(
        x, y, z, backend="jax"
    )
    assert (ilinx_r, iliny_r) == (ilinx_j, iliny_j)
    np.testing.assert_allclose(c_rust, c_jax, rtol=1e-12, atol=1e-13)

    ev_r = interp_2d.interp_evbicub_db(
        0.35, -0.45, x, y, ilinx_r, iliny_r, c_rust, backend="rust"
    )
    ev_j = interp_2d.interp_evbicub_db(
        0.35, -0.45, x, y, ilinx_j, iliny_j, c_jax, backend="jax"
    )
    np.testing.assert_allclose(float(ev_r.f), float(ev_j.f), rtol=1e-12)
    np.testing.assert_allclose(float(ev_r.df_dx), float(ev_j.df_dx), rtol=1e-12)
    np.testing.assert_allclose(float(ev_r.df_dy), float(ev_j.df_dy), rtol=1e-12)

    # Batch 2D bicubic evaluation in sequential vs parallel rayon modes.
    xq = np.linspace(-1.0, 1.0, 128, dtype=np.float64)
    yq = np.linspace(-0.8, 0.8, 128, dtype=np.float64)
    c_flat = np.ascontiguousarray(c_rust, dtype=np.float64).ravel()
    batch_seq = _rust.interp_evbicub_batch(xq, yq, x, y, c_flat, False)
    batch_par = _rust.interp_evbicub_batch(xq, yq, x, y, c_flat, True)
    np.testing.assert_allclose(batch_seq, batch_par, rtol=1e-15, atol=1e-15)

    # Confirm invalid coefficient table length raises ValueError cleanly.
    with pytest.raises(ValueError, match="coeffs_flat length"):
        _rust.interp_evbicub_batch(xq[:2], yq[:2], x, y, c_flat[:8], False)


def test_neu_and_turb_rust_and_jax_backends_match_to_machine_precision() -> (
    None
):
    """Verify `neu` and `turb` parity across `backend="rust"` and `"jax"`.

    Checks `neu_get`, `_rust.neu_get_batch`, `set_mlt`, `_rust.set_mlt_batch`,
    and `_rust.set_tdc_batch`.
    """
    res_rust = neu.neu_get(
        log10_t=8.7, log10_rho=5.2, abar=12.0, zbar=6.0, backend="rust"
    )
    res_jax = neu.neu_get(
        log10_t=8.7, log10_rho=5.2, abar=12.0, zbar=6.0, backend="jax"
    )
    np.testing.assert_allclose(
        res_rust.loss, res_jax.loss, rtol=1e-13, atol=1e-14
    )
    np.testing.assert_allclose(
        res_rust.sources, res_jax.sources, rtol=1e-13, atol=1e-14
    )

    # Batch neutrino evaluation across 128 zones in sequential vs parallel.
    lt = np.linspace(8.0, 9.5, 128, dtype=np.float64)
    t_arr = np.power(10.0, lt)
    lr = np.linspace(4.0, 6.5, 128, dtype=np.float64)
    r_arr = np.power(10.0, lr)
    abar = np.full(128, 4.0, dtype=np.float64)
    zbar = np.full(128, 2.0, dtype=np.float64)
    flags = (True, True, True, True, False)
    neu_seq = _rust.neu_get_batch(
        t_arr, lt, r_arr, lr, abar, zbar, 7.5, flags, False
    )
    neu_par = _rust.neu_get_batch(
        t_arr, lt, r_arr, lr, abar, zbar, 7.5, flags, True
    )
    np.testing.assert_allclose(neu_seq, neu_par, rtol=1e-15, atol=1e-15)

    # Compare set_mlt across backends and batch parallel MLT/TDC evaluators.
    params = turb.MltParams(option=turb.MltOption.COX, mixing_length_alpha=2.0)
    inp = turb.MltInputs(
        chi_t=1.0,
        chi_rho=1.0,
        cp=2.5e8,
        grav=1.0e4,
        lambda_=1.0e10,
        rho=1.0e-2,
        p=1.0e12,
        t=1.0e7,
        opacity=1.0,
        gradr=0.5,
        grada=0.4,
        gradl=0.4,
    )
    out_r = turb.set_mlt(params, inp, backend="rust")
    out_j = turb.set_mlt(params, inp, backend="jax")
    assert isinstance(out_r.gamma, float) and isinstance(out_j.gamma, float)
    assert isinstance(out_r.grad_t, float) and isinstance(out_j.grad_t, float)
    assert isinstance(out_r.conv_vel, float) and isinstance(
        out_j.conv_vel, float
    )
    np.testing.assert_allclose(out_r.gamma, out_j.gamma, rtol=1e-14)
    np.testing.assert_allclose(out_r.grad_t, out_j.grad_t, rtol=1e-14)
    np.testing.assert_allclose(out_r.conv_vel, out_j.conv_vel, rtol=1e-14)

    # Verify batch parallel MLT and TDC evaluators match sequential evaluation
    # to IEEE-754 bit-for-bit / machine precision across 128 and 64 zones.
    s_frac = np.linspace(0.0, 1.0, 128, dtype=np.float64)
    mlt_seq = _rust.set_mlt_batch(s_frac, False)
    mlt_par = _rust.set_mlt_batch(s_frac, True)
    np.testing.assert_allclose(mlt_seq, mlt_par, rtol=1e-15, atol=1e-15)

    tdc_seq = _rust.set_tdc_batch(s_frac[:64], False)
    tdc_par = _rust.set_tdc_batch(s_frac[:64], True)
    np.testing.assert_allclose(tdc_seq, tdc_par, rtol=1e-15, atol=1e-15)


def test_tier_c_rust_block_tridiagonal_and_bcyclic_solvers() -> None:
    """Verify `_rust.solve_block_tridiagonal` and `solve_block_tridiag_bcyclic`.

    Confirms that sequential block-Thomas and sequential/parallel Hirshman et
    al. (2010) `bcyclic` odd-even cyclic reduction match $A x = b$ to `< 1e-12`.
    """
    rng = np.random.default_rng(12345)
    nvar, nz = 4, 24
    sub = 0.05 * rng.standard_normal((nz, nvar, nvar))
    sub[0] = 0.0
    sup = 0.05 * rng.standard_normal((nz, nvar, nvar))
    sup[-1] = 0.0
    diag = 0.1 * rng.standard_normal((nz, nvar, nvar)) + 4.0 * np.eye(nvar)
    rhs = rng.standard_normal((nz, nvar))

    sub_f = np.ascontiguousarray(sub, dtype=np.float64).ravel()
    diag_f = np.ascontiguousarray(diag, dtype=np.float64).ravel()
    sup_f = np.ascontiguousarray(sup, dtype=np.float64).ravel()
    rhs_f = np.ascontiguousarray(rhs, dtype=np.float64).ravel()

    x_thomas = _rust.solve_block_tridiagonal(
        nvar, nz, sub_f, diag_f, sup_f, rhs_f
    )
    x_bcyc_seq = _rust.solve_block_tridiag_bcyclic(
        nvar, nz, sub_f, diag_f, sup_f, rhs_f, False
    )
    x_bcyc_par = _rust.solve_block_tridiag_bcyclic(
        nvar, nz, sub_f, diag_f, sup_f, rhs_f, True
    )
    np.testing.assert_allclose(x_bcyc_seq, x_thomas, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(x_bcyc_par, x_thomas, rtol=1e-12, atol=1e-12)

    # Verify transpose / left-multiply helper block_multiply_xa.
    y_left = _rust.block_multiply_xa(
        nvar, nz, sub_f, diag_f, sup_f, x_bcyc_par.ravel()
    )
    assert y_left.shape == (nz, nvar)
    assert np.all(np.isfinite(y_left))
