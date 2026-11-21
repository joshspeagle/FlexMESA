# SPDX-License-Identifier: LGPL-3.0-or-later
"""Autodiff, JAX, semiconvection, and thermohaline tests for `flexmesa.turb`.

Verifies:
1. Forward-mode `DualOrder1` derivatives against centered finite differences
   across all six `MltOption` prescriptions and `set_tdc` (`AutoDiffRealTdc`
   Newton-iteration partials).
2. Pure-JAX `float64` kernels (`set_mlt_jax` and `set_semiconvection_jax`)
   under `jax.jit`, `jax.vmap`, and `jax.grad`.
3. Both Langer (1985) semiconvection options and all three thermohaline
   options (`Kippenhahn`, `Traxler_Garaud_Stellmach_11`, and
   `Brown_Garaud_Stellmach_13`).
4. `AutoDiffRealTdc` second-order chain-rule identities, `set_mlt` early
   returns (`turb/private/mlt.f90:140,145,169`), and `set_tdc` unconditional
   Cox MLT call and Rust-versus-Python sensitivity agreement.
"""

from __future__ import annotations

import collections.abc
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flexmesa import _rust
from flexmesa import auto_diff as _ad
from flexmesa import constants as _const
from flexmesa import turb as _turb
from flexmesa.turb import types as _types


def _assert_rel(actual: float, expected: float, rtol: float = 1.0e-12) -> None:
    rel = abs(actual - expected) / abs(expected)
    assert rel <= rtol, (
        f"actual={actual:.16e}, expected={expected:.16e}, rel={rel:.3e}"
    )


def test_mlt_options_jax_and_autodiff_vs_finite_difference() -> None:
    """Verify all six MLT options, `DualOrder1`, and `set_mlt_jax` gradients."""
    t = 1.0e5
    rho = 1.0e-5
    grav = 1.0e4
    cp = 2.5 * _const.KERG / _const.MP
    p = rho * t * _const.KERG / _const.MP
    lambda_ = p / (rho * grav)
    opacity = 1.0
    grada = 0.4
    gradl = 0.4
    gradr = 0.8

    for opt in (
        _turb.MltOption.COX,
        _turb.MltOption.TDC,
        _turb.MltOption.HENYEY,
        _turb.MltOption.ML1,
        _turb.MltOption.ML2,
        _turb.MltOption.MIHALAS,
    ):
        params = _turb.MltParams(option=opt, mixing_length_alpha=1.8)
        dual_inp = _turb.MltInputs(
            chi_t=1.0,
            chi_rho=1.0,
            cp=cp,
            grav=grav,
            lambda_=lambda_,
            rho=rho,
            p=p,
            t=t,
            opacity=opacity,
            gradr=_ad.DualOrder1.independent(gradr, index=0, num_vars=1),
            grada=grada,
            gradl=gradl,
        )
        dual_out = _turb.set_mlt(params, dual_inp)
        assert isinstance(dual_out.grad_t, _ad.DualOrder1)
        d_gradr = gradr * 1.0e-6
        up = _turb.set_mlt(
            params,
            _turb.MltInputs(
                1.0,
                1.0,
                cp,
                grav,
                lambda_,
                rho,
                p,
                t,
                opacity,
                gradr + d_gradr,
                grada,
                gradl,
            ),
        )
        dn = _turb.set_mlt(
            params,
            _turb.MltInputs(
                1.0,
                1.0,
                cp,
                grav,
                lambda_,
                rho,
                p,
                t,
                opacity,
                gradr - d_gradr,
                grada,
                gradl,
            ),
        )
        fd = (_types.val_of(up.grad_t) - _types.val_of(dn.grad_t)) / (
            2.0 * d_gradr
        )
        _assert_rel(float(dual_out.grad_t.d1[0]), fd, 1.0e-6)

    jax_fn = jax.jit(
        lambda gr: _turb.set_mlt_jax(
            jnp.float64(1.0),
            jnp.float64(1.0),
            jnp.float64(cp),
            jnp.float64(grav),
            jnp.float64(lambda_),
            jnp.float64(rho),
            jnp.float64(p),
            jnp.float64(t),
            jnp.float64(opacity),
            gr,
            jnp.float64(grada),
            jnp.float64(gradl),
            1.8,
        )[1]
    )
    jax_grad_t = float(jax_fn(jnp.float64(gradr)))
    jax_dgrad_t = float(jax.grad(jax_fn)(jnp.float64(gradr)))
    vmapped = jax.vmap(jax_fn)(jnp.array([gradr, gradr + 0.01]))
    assert vmapped.shape == (2,)
    cox_out = _turb.set_mlt(
        _turb.MltParams.cox(1.8),
        _turb.MltInputs(
            1.0,
            1.0,
            cp,
            grav,
            lambda_,
            rho,
            p,
            t,
            opacity,
            _ad.DualOrder1.independent(gradr, index=0, num_vars=1),
            grada,
            gradl,
        ),
    )
    _assert_rel(jax_grad_t, _types.val_of(cox_out.grad_t), 1.0e-12)
    assert isinstance(cox_out.grad_t, _ad.DualOrder1)
    _assert_rel(jax_dgrad_t, float(cox_out.grad_t.d1[0]), 1.0e-11)


def test_tdc_autodiff_semiconvection_and_thermohaline() -> None:
    """Verify `set_tdc` Newton-iteration derivatives and diffusive models."""
    t = 1.0e5
    rho = 1.0e-5
    r = _const.RSUN
    m = _const.MSUN
    cgrav = _const.STANDARD_CGRAV
    grav = m * cgrav / (r * r)
    cp = 2.5 * _const.KERG / _const.MP
    p = rho * t * _const.KERG / _const.MP
    hp = p / (rho * grav)
    gradr = 0.45
    l_val = (
        64.0
        * _const.PI
        * _const.BOLTZ_SIGMA
        * math.pow(t, 4.0)
        * grav
        * r
        * r
        * gradr
        / (3.0 * p)
    )
    params = _turb.TdcParams(
        conv_vel_start=1.0e4,
        mixing_length_alpha=2.0,
        tdc_alpha_r=1.0,
        dt=1.0e4,
        cgrav=cgrav,
        m=m,
        scale=l_val * 1.0e-3,
        include_mlt_corr_to_tdc=True,
        use_tdc_enthalpy_flux_limiter=True,
    )
    dual_res = _turb.set_tdc(
        params,
        _turb.TdcInputs(
            1.0,
            1.0,
            _ad.DualOrder1.independent(gradr, 0, 1),
            r,
            p,
            t,
            rho,
            0.0,
            cp,
            1.0,
            hp,
            0.4,
            0.4,
            0.0,
            grav,
            1.5 * p / rho,
        ),
    )
    assert isinstance(dual_res.output.y_face, _ad.DualOrder1)
    d_gr = gradr * 1.0e-6
    up = _turb.set_tdc(
        params,
        _turb.TdcInputs(
            1.0,
            1.0,
            gradr + d_gr,
            r,
            p,
            t,
            rho,
            0.0,
            cp,
            1.0,
            hp,
            0.4,
            0.4,
            0.0,
            grav,
            1.5 * p / rho,
        ),
    )
    dn = _turb.set_tdc(
        params,
        _turb.TdcInputs(
            1.0,
            1.0,
            gradr - d_gr,
            r,
            p,
            t,
            rho,
            0.0,
            cp,
            1.0,
            hp,
            0.4,
            0.4,
            0.0,
            grav,
            1.5 * p / rho,
        ),
    )
    fd_y = (
        _types.val_of(up.output.y_face) - _types.val_of(dn.output.y_face)
    ) / (2.0 * d_gr)
    _assert_rel(float(dual_res.output.y_face.d1[0]), fd_y, 1.0e-5)

    for sc_opt in _turb.SemiconvectionOption:
        sc_params = _turb.SemiconvectionParams(
            option=sc_opt,
            alpha_semiconvection=0.1,
            gradl_composition_term=0.05,
        )
        sc_out = _turb.set_semiconvection(
            sc_params,
            _turb.SemiconvectionInputs(
                1.0e35,
                1.0e10,
                2.0e7,
                1.0e16,
                1.0e15,
                0.9,
                0.34,
                10.0,
                2.5e8,
                _ad.DualOrder1.independent(0.42, index=0, num_vars=1),
                0.40,
                0.45,
            ),
        )
        assert _types.val_of(sc_out.d) > 0.0
        assert sc_out.mixing_type == _const.MixingType.SEMICONVECTIVE_MIXING
        assert isinstance(sc_out.d, _ad.DualOrder1)
        assert abs(float(sc_out.d.d1[0])) > 0.0

    sc_jax_fn = jax.jit(
        lambda gr: _turb.set_semiconvection_jax(
            jnp.float64(1.0e10),
            jnp.float64(2.0e7),
            jnp.float64(0.34),
            jnp.float64(10.0),
            jnp.float64(2.5e8),
            gr,
            jnp.float64(0.40),
            jnp.float64(0.45),
            0.1,
        )[3]
    )
    sc_d_jax = float(sc_jax_fn(jnp.float64(0.42)))
    sc_dd_jax = float(jax.grad(sc_jax_fn)(jnp.float64(0.42)))
    assert sc_d_jax > 0.0
    assert abs(sc_dd_jax) > 0.0

    for th_opt in _turb.ThermohalineOption:
        th_out = _turb.set_thermohaline(
            _turb.ThermohalineParams(option=th_opt),
            _turb.ThermohalineInputs(
                _ad.DualOrder1.independent(1.0e9, index=0, num_vars=1),
                0.40,
                0.38,
                3.0e7,
                0.3,
                5.0,
                2.5e8,
            ),
        )
        assert _types.val_of(th_out.d) > 0.0
        assert th_out.mixing_type == _const.MixingType.THERMOHALINE_MIXING
        assert isinstance(th_out.conv_vel, _ad.DualOrder1)
        assert abs(float(th_out.conv_vel.d1[0])) > 0.0


def test_auto_diff_real_tdc_chain_rule_and_python_matches_rust() -> None:
    """Verify `AutoDiffRealTdc` chain rule and Python-vs-Rust TDC partials."""

    def _var(a: float, b: float, c: float, d: float) -> _turb.AutoDiffRealTdc:
        return _turb.AutoDiffRealTdc(
            a,
            b,
            np.array([c], dtype=np.float64),
            np.array([d], dtype=np.float64),
        )

    def _close(got: float, want: float) -> None:
        scale = max(abs(want), 1.0)
        assert abs(got - want) <= 2.0e-6 * scale, f"got {got:e}, want {want:e}"

    unary_cases: tuple[
        tuple[
            collections.abc.Callable[
                [_turb.AutoDiffRealTdc], _turb.AutoDiffRealTdc
            ],
            collections.abc.Callable[[float], float],
            float,
        ],
        ...,
    ] = (
        (lambda x: x.exp(), math.exp, 0.3),
        (lambda x: x.log(), math.log, 1.7),
        (lambda x: x.sqrt(), math.sqrt, 2.3),
        (lambda x: x.pow2(), lambda v: v * v, -1.4),
        (lambda x: x.pow3(), lambda v: v * v * v, 1.2),
        (lambda x: x.pow4(), lambda v: v * v * v * v, 0.9),
        (lambda x: x.tanh(), math.tanh, 0.6),
        (lambda x: x.tan(), math.tan, 0.4),
        (lambda x: x.atan(), math.atan, -0.8),
        (lambda x: x.powf(1.0 / 3.0), lambda v: math.pow(v, 1.0 / 3.0), 2.9),
        (lambda x: x.abs(), abs, -1.1),
        (lambda x: -x, lambda v: -v, 0.5),
    )
    b, c, d = 0.7, -1.3, 0.45
    for op, g, a in unary_cases:
        h = 1.0e-4 * max(abs(a), 1.0)
        g1 = (g(a + h) - g(a - h)) / (2.0 * h)
        g2 = (g(a + h) - 2.0 * g(a) + g(a - h)) / (h * h)
        out = op(_var(a, b, c, d))
        _close(out.val, g(a))
        _close(out.d1val1, g1 * b)
        _close(float(out.d1_array[0]), g1 * c)
        _close(float(out.d1val1_d1_array[0]), g2 * b * c + g1 * d)

    x = _var(2.0, 3.0, 5.0, 7.0)
    y = _var(11.0, 13.0, 17.0, 19.0)
    back = (x / y) * y
    _close(back.val, x.val)
    _close(back.d1val1, x.d1val1)
    _close(float(back.d1_array[0]), float(x.d1_array[0]))
    _close(float(back.d1val1_d1_array[0]), float(x.d1val1_d1_array[0]))

    # Compare Python `set_tdc` sensitivities directly against Rust
    # `_rust.set_tdc_batch` across all 8 benchmark shells.
    grav_solar = _const.MSUN * _const.STANDARD_CGRAV / (_const.RSUN**2)
    rgas_ideal = _const.KERG / _const.MP
    cp_ideal = 2.5 * rgas_ideal
    s_np = np.arange(8, dtype=np.float64) / 7.0
    rust_out = _rust.set_tdc_batch(s_np, False)
    for k, sf in enumerate(s_np):
        t_val = 1.0e5 * (1.0 + 0.2 * float(sf))
        rho_val = 1.0e-5 * (1.0 + 0.2 * float(sf))
        p_val = rho_val * t_val * rgas_ideal
        hp_val = p_val / (rho_val * grav_solar)
        gradr_val = 0.42 + 0.08 * float(sf)
        params = _turb.TdcParams(
            conv_vel_start=1.0e5,
            mixing_length_alpha=2.0,
            tdc_alpha_d=1.0,
            tdc_alpha_r=0.0,
            tdc_alpha_pt=0.0,
            tdc_alpha_c=1.0,
            tdc_alpha_s=1.0,
            dt=1.0e4,
            cgrav=_const.STANDARD_CGRAV,
            m=_const.MSUN,
            scale=70.0 * _const.LSUN * 1.0e-3,
            max_conv_vel=1.0e99,
            include_mlt_corr_to_tdc=True,
            use_tdc_enthalpy_flux_limiter=False,
        )
        inp = _turb.TdcInputs(
            chi_t=_ad.DualOrder1.constant(1.0, 6),
            chi_rho=_ad.DualOrder1.constant(1.0, 6),
            gradr=_ad.DualOrder1.independent(gradr_val, 0, 6),
            r=_ad.DualOrder1.constant(_const.RSUN, 6),
            p=_ad.DualOrder1.constant(p_val, 6),
            t=_ad.DualOrder1.independent(t_val, 4, 6),
            rho=_ad.DualOrder1.independent(rho_val, 5, 6),
            dv=_ad.DualOrder1.constant(0.0, 6),
            cp=_ad.DualOrder1.constant(cp_ideal, 6),
            opacity=_ad.DualOrder1.constant(1.0, 6),
            scale_height=_ad.DualOrder1.independent(hp_val, 3, 6),
            gradl=_ad.DualOrder1.independent(0.40, 2, 6),
            grada=_ad.DualOrder1.independent(0.40, 1, 6),
            eq_div_w=_ad.DualOrder1.constant(0.0, 6),
            grav=_ad.DualOrder1.constant(grav_solar, 6),
            energy=_ad.DualOrder1.constant(0.0, 6),
        )
        res = _turb.set_tdc(params, inp)
        assert isinstance(res.output.grad_t, _ad.DualOrder1)
        _assert_rel(float(res.output.grad_t.val), float(rust_out[k, 0]), 1e-12)
        _assert_rel(
            float(res.output.grad_t.d1[0]), float(rust_out[k, 3]), 1e-12
        )


def test_mlt_early_returns_and_tdc_unconditional_cox_python() -> None:
    """Verify Python `set_mlt` early returns and `set_tdc` Cox MLT call."""
    cp = 2.5 * _const.KERG / _const.MP
    # Optically thin Mihalas hits `f1 <= 0` (`turb/private/mlt.f90:140`).
    thin_inp = _turb.MltInputs(
        chi_t=1.0,
        chi_rho=1.0,
        cp=cp,
        grav=1.0e4,
        lambda_=1.0e-3,
        rho=1.0e-10,
        p=1.0,
        t=1.0e5,
        opacity=1.0e-3,
        gradr=0.4 + 1.0e-12,
        grada=0.4,
        gradl=0.4,
    )
    thin_out = _turb.set_mlt(
        _turb.MltParams(
            option=_turb.MltOption.MIHALAS,
            mixing_length_alpha=1.0,
            henyey_nu=0.0,
            henyey_y=0.0,
        ),
        thin_inp,
        backend="jax",
    )
    assert thin_out.gamma == 0.0
    assert thin_out.grad_t == thin_inp.gradr
    assert thin_out.y_face == thin_inp.gradr - thin_inp.gradl
    assert thin_out.mixing_type == _const.MixingType.NO_MIXING

    # Zero `lambda_` (`delta == 0` with `gradr > gradl`) hits `Gamma <= 0`
    # (`turb/private/mlt.f90:145`) and preserves the computed `Gamma`.
    zero_lam_inp = _turb.MltInputs(
        chi_t=1.0,
        chi_rho=1.0,
        cp=cp,
        grav=1.0e4,
        lambda_=0.0,
        rho=1.0e-5,
        p=1.0e-5 * 1.0e5 * _const.KERG / _const.MP,
        t=1.0e5,
        opacity=1.0,
        gradr=0.45,
        grada=0.4,
        gradl=0.4,
    )
    zero_lam_out = _turb.set_mlt(
        _turb.MltParams.cox(2.0), zero_lam_inp, backend="jax"
    )
    assert _types.val_of(zero_lam_out.gamma) <= 0.0
    assert zero_lam_out.grad_t == zero_lam_inp.gradr
    assert zero_lam_out.y_face == zero_lam_inp.gradr - zero_lam_inp.gradl
    assert zero_lam_out.mixing_type == _const.MixingType.NO_MIXING

    # `set_tdc` calls Cox MLT unconditionally even when
    # `include_mlt_corr_to_tdc=False` and rejects `alpha == 0` / `dt <= 0`.
    t = 1.0e5
    rho = 1.0e-5
    r = _const.RSUN
    m = _const.MSUN
    cgrav = _const.STANDARD_CGRAV
    grav = m * cgrav / (r * r)
    p = rho * t * _const.KERG / _const.MP
    hp = p / (rho * grav)
    gradr = 0.45
    lum = (
        64.0
        * _const.PI
        * _const.BOLTZ_SIGMA
        * math.pow(t, 4.0)
        * grav
        * r
        * r
        * gradr
        / (3.0 * p)
    )
    base_inp = _turb.TdcInputs(
        1.0,
        1.0,
        gradr,
        r,
        p,
        t,
        rho,
        0.0,
        cp,
        1.0,
        hp,
        0.4,
        0.4,
        0.0,
        grav,
        0.0,
    )
    tdc_no_corr = _turb.set_tdc(
        _turb.TdcParams(
            conv_vel_start=1.0e4,
            mixing_length_alpha=2.0,
            dt=1.0e4,
            cgrav=cgrav,
            m=m,
            scale=lum * 1.0e-3,
            include_mlt_corr_to_tdc=False,
        ),
        base_inp,
    )
    assert _types.val_of(tdc_no_corr.output.gamma) > 0.0

    with pytest.raises(ValueError, match="get_TDC_solution"):
        _turb.set_tdc(
            _turb.TdcParams(mixing_length_alpha=0.0, dt=1.0e4), base_inp
        )
    with pytest.raises(ValueError, match="get_TDC_solution"):
        _turb.set_tdc(
            _turb.TdcParams(mixing_length_alpha=2.0, dt=0.0), base_inp
        )
