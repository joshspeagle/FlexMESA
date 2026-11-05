# SPDX-License-Identifier: LGPL-3.0-or-later
"""Autodiff, JAX, semiconvection, and thermohaline tests for `flexmesa.turb`.

Verifies:
1. Forward-mode `DualOrder1` derivatives against centered finite differences
   across all six `MltOption` prescriptions and `set_tdc` (Implicit Function
   Theorem).
2. Pure-JAX `float64` kernels (`set_mlt_jax` and `set_semiconvection_jax`)
   under `jax.jit`, `jax.vmap`, and `jax.grad`.
3. Both Langer (1985) semiconvection options and all three thermohaline
   options (`Kippenhahn`, `Traxler_Garaud_Stellmach_11`, and
   `Brown_Garaud_Stellmach_13`).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

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
    """Verify `set_tdc` implicit derivatives and diffusive prescriptions."""
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
        y_face_guess=0.01,
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
