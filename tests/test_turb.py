# SPDX-License-Identifier: LGPL-3.0-or-later
"""Golden-baseline parity tests for `flexmesa.turb` (`turb/test/test_output`).

Verifies:
1. `check_efficient_MLT_scaling`: Convective speeds $v_{\\mathrm{conv}}$ at
   $L_1 = 10^5\\,L_\\odot$ and $L_2 = 10^8\\,L_\\odot$ and cubic scaling ratio
   $v_2 / v_1$ to $\\le 10^{-12}$ relative error.
2. `compare_TDC_and_Cox_MLT`: Steady-state convergence of `set_tdc` at
   $\\Delta t = 10^{40}\\,\\mathrm{s}$ and exact Cox MLT parity.
3. `check_TDC`: All 31 time-step rows $\\Delta t = 500 \\times 1.02^j$ s
   ($j = 0, \\dots, 30$) on the negative-$Y$ decaying-convection branch.
"""

from __future__ import annotations

import math

from flexmesa import constants as _const
from flexmesa import turb as _turb

_GOLDEN_TDC_ROWS: tuple[tuple[float, float, float], ...] = (
    (5.0000000000000000e02, 2.5204370043250246e-01, 6.0597238914198903e06),
    (5.1000000000000000e02, 2.5204285749162930e-01, 5.9050075093203261e06),
    (5.2020000000000005e02, 2.5204196393651124e-01, 5.7494001285570739e06),
    (5.3060399999999993e02, 2.5204101302204984e-01, 5.5925658656495046e06),
    (5.4121608000000003e02, 2.5203999647052466e-01, 5.4341006185611701e06),
    (5.5204040160000000e02, 2.5203890395733641e-01, 5.2735100646096319e06),
    (5.6308120963200008e02, 2.5203772235599331e-01, 5.1101775684554409e06),
    (5.7434283382464002e02, 2.5203643459023894e-01, 4.9433157147388468e06),
    (5.8582969050113286e02, 2.5203501781331550e-01, 4.7718899504500106e06),
    (5.9754628431115555e02, 2.5203344036462588e-01, 4.5944918361562928e06),
    (6.0949720999737872e02, 2.5203165633254127e-01, 4.4091141978058126e06),
    (6.2168715419732621e02, 2.5202959494944044e-01, 4.2127157422839301e06),
    (6.3412089728127273e02, 2.5202713722579184e-01, 4.0002689176325197e06),
    (6.4680331522689812e02, 2.5202405410296042e-01, 3.7622592559380517e06),
    (6.5973938153143615e02, 2.5201978092426058e-01, 3.4756430172060304e06),
    (6.7293416916206490e02, 2.5201140133059619e-01, 3.0239000429141554e06),
    (6.8639285254530625e02, -1.2020505435123705e05, 0.0),
    (7.0012070959621246e02, -1.2020505435123705e05, 0.0),
    (7.1412312378813670e02, -1.2020505435123705e05, 0.0),
    (7.2840558626389941e02, -1.2020505435123705e05, 0.0),
    (7.4297369798917748e02, -1.2020505435123705e05, 0.0),
    (7.5783317194896108e02, -1.2020505435123705e05, 0.0),
    (7.7298983538794027e02, -1.2020505435123705e05, 0.0),
    (7.8844963209569903e02, -1.2020505435123705e05, 0.0),
    (8.0421862473761303e02, -1.2020505435123705e05, 0.0),
    (8.2030299723236533e02, -1.2020505435123705e05, 0.0),
    (8.3670905717701260e02, -1.2020505435123705e05, 0.0),
    (8.5344323832055295e02, -1.2020505435123705e05, 0.0),
    (8.7051210308696398e02, -1.2020505435123705e05, 0.0),
    (8.8792234514870324e02, -1.2020505435123705e05, 0.0),
    (9.0568079205167737e02, -1.2020505435123705e05, 0.0),
)


def _assert_rel(actual: float, expected: float, rtol: float = 1.0e-12) -> None:
    rel = abs(actual - expected) / abs(expected)
    assert rel <= rtol, (
        f"actual={actual:.16e}, expected={expected:.16e}, rel={rel:.3e}"
    )


def test_efficient_mlt_scaling_matches_golden_baseline() -> None:
    """Verify `check_efficient_MLT_scaling` in `turb/test/test_output`."""
    params = _turb.MltParams(
        option=_turb.MltOption.COX,
        mixing_length_alpha=1.0,
        henyey_nu=0.0,
        henyey_y=0.0,
    )
    t = 1.0e5
    rho = 1.0e-5
    grav = 1.0e4
    r = _const.RSUN
    cp = 2.5 * _const.KERG / _const.MP
    p = rho * t * _const.KERG / _const.MP
    lambda_ = p / (rho * grav)
    opacity = 1.0
    grada = 0.4
    gradl = grada

    l1 = 1.0e5 * _const.LSUN
    gradr1 = (
        3.0
        * p
        * opacity
        * l1
        / (
            64.0
            * _const.PI
            * _const.BOLTZ_SIGMA
            * math.pow(t, 4.0)
            * grav
            * r
            * r
        )
    )
    inp1 = _turb.MltInputs(
        chi_t=1.0,
        chi_rho=1.0,
        cp=cp,
        grav=grav,
        lambda_=lambda_,
        rho=rho,
        p=p,
        t=t,
        opacity=opacity,
        gradr=gradr1,
        grada=grada,
        gradl=gradl,
    )
    out1 = _turb.set_mlt(params, inp1)
    assert isinstance(out1.conv_vel, float)
    _assert_rel(out1.conv_vel, 3.9654203785866620e06, 1.0e-12)
    assert out1.mixing_type == _const.MixingType.CONVECTIVE_MIXING

    l2 = 1.0e8 * _const.LSUN
    gradr2 = (
        3.0
        * p
        * opacity
        * l2
        / (
            64.0
            * _const.PI
            * _const.BOLTZ_SIGMA
            * math.pow(t, 4.0)
            * grav
            * r
            * r
        )
    )
    inp2 = _turb.MltInputs(
        chi_t=1.0,
        chi_rho=1.0,
        cp=cp,
        grav=grav,
        lambda_=lambda_,
        rho=rho,
        p=p,
        t=t,
        opacity=opacity,
        gradr=gradr2,
        grada=grada,
        gradl=gradl,
    )
    out2 = _turb.set_mlt(params, inp2)
    assert isinstance(out2.conv_vel, float)
    _assert_rel(out2.conv_vel, 3.9765811706780650e07, 1.0e-12)
    _assert_rel(out2.conv_vel / out1.conv_vel, 1.0028145293628064e01, 1.0e-12)


def test_compare_tdc_and_cox_mlt_matches_golden_baseline() -> None:
    """Verify `compare_TDC_and_Cox_MLT` in `turb/test/test_output`."""
    mixing_length_alpha = 2.0
    t = 1.0e5
    rho = 1.0e-5
    r = _const.RSUN
    m = _const.MSUN
    cgrav = _const.STANDARD_CGRAV
    grav = m * cgrav / (r * r)
    cp = 2.5 * _const.KERG / _const.MP
    p = rho * t * _const.KERG / _const.MP
    scale_height = p / (rho * grav)
    lambda_ = mixing_length_alpha * scale_height
    opacity = 1.0
    grada = 0.4
    gradl = grada

    l_val = 70.0 * _const.LSUN
    gradr = (
        3.0
        * p
        * opacity
        * l_val
        / (
            64.0
            * _const.PI
            * _const.BOLTZ_SIGMA
            * math.pow(t, 4.0)
            * grav
            * r
            * r
        )
    )
    l_val = l_val * (1.0 + 1.0e-5) * (grada / gradr)
    gradr = (
        3.0
        * p
        * opacity
        * l_val
        / (
            64.0
            * _const.PI
            * _const.BOLTZ_SIGMA
            * math.pow(t, 4.0)
            * grav
            * r
            * r
        )
    )
    _assert_rel(gradr - grada, 4.0000000000040004e-06, 1.0e-12)

    tdc_params = _turb.TdcParams(
        conv_vel_start=0.0,
        mixing_length_alpha=mixing_length_alpha,
        tdc_alpha_d=1.0,
        tdc_alpha_r=0.0,
        tdc_alpha_pt=0.0,
        tdc_alpha_c=1.0,
        tdc_alpha_s=1.0,
        dt=1.0e40,
        cgrav=cgrav,
        m=m,
        scale=l_val * 1.0e-3,
        max_conv_vel=1.0e99,
        include_mlt_corr_to_tdc=True,
        use_tdc_enthalpy_flux_limiter=False,
    )
    tdc_inp = _turb.TdcInputs(
        chi_t=1.0,
        chi_rho=1.0,
        gradr=gradr,
        r=r,
        p=p,
        t=t,
        rho=rho,
        dv=0.0,
        cp=cp,
        opacity=opacity,
        scale_height=scale_height,
        gradl=gradl,
        grada=grada,
        eq_div_w=0.0,
        grav=grav,
        energy=0.0,
    )
    tdc_res = _turb.set_tdc(tdc_params, tdc_inp)
    assert isinstance(tdc_res.output.grad_t, float)
    assert isinstance(tdc_res.output.y_face, float)
    assert isinstance(tdc_res.output.conv_vel, float)
    _assert_rel(tdc_res.output.grad_t, 0.4 + 3.9999852208023926e-06, 1.0e-15)
    _assert_rel(tdc_res.output.y_face, 3.9999852208023926e-06, 2.0e-11)
    _assert_rel(tdc_res.output.conv_vel, 1.5051085052069075e02, 2.0e-11)

    mlt_res = _turb.set_mlt(
        _turb.MltParams.cox(mixing_length_alpha),
        _turb.MltInputs(
            chi_t=1.0,
            chi_rho=1.0,
            cp=cp,
            grav=grav,
            lambda_=lambda_,
            rho=rho,
            p=p,
            t=t,
            opacity=opacity,
            gradr=gradr,
            grada=grada,
            gradl=gradl,
        ),
    )
    assert isinstance(mlt_res.y_face, float)
    assert isinstance(mlt_res.conv_vel, float)
    assert isinstance(mlt_res.gamma, float)
    _assert_rel(mlt_res.y_face, 3.9999830300230244e-06, 1.0e-12)
    _assert_rel(mlt_res.conv_vel, 1.5051080930415759e02, 1.0e-12)
    _assert_rel(mlt_res.gamma, 1.3740991947818479e-03, 1.0e-12)


def test_tdc_dt_sweep_31_golden_rows() -> None:
    """Verify all 31 time-step rows of `check_TDC` in `test_output`."""
    cgrav = 6.6743000000000004e-8
    m = 5.8707400456875664e34
    scale = 5.0386519362246294e45
    l_val = 1.09415288158835 * (-5.0386519362246294e45)
    r = 10314294541.567163
    p = 5.0581587249808894e20
    t = 613193666.5178368
    rho = 5204.573257474575
    dv = 3.8256494463482604e-7
    cp = 6628075118.460659
    opacity = 9.075017123146995e-2
    scale_height = 2638686602.006378
    gradl = 0.252075872673435
    grada = 0.2520469725687274
    gradr = (
        3.0
        * p
        * opacity
        * l_val
        / (64.0 * _const.PI * _const.BOLTZ_SIGMA * math.pow(t, 4.0) * cgrav * m)
    )
    grav = m * cgrav / (r * r)

    inp = _turb.TdcInputs(
        chi_t=1.0,
        chi_rho=1.0,
        gradr=gradr,
        r=r,
        p=p,
        t=t,
        rho=rho,
        dv=dv,
        cp=cp,
        opacity=opacity,
        scale_height=scale_height,
        gradl=gradl,
        grada=grada,
        eq_div_w=0.0,
        grav=grav,
        energy=0.0,
    )

    for j, (expected_dt, expected_grad_t, expected_vc) in enumerate(
        _GOLDEN_TDC_ROWS
    ):
        dt = 500.0 * math.pow(1.02, float(j))
        _assert_rel(dt, expected_dt, 1.0e-14)
        params = _turb.TdcParams(
            conv_vel_start=52320587.41515405,
            mixing_length_alpha=2.0,
            tdc_alpha_d=1.0,
            tdc_alpha_r=0.0,
            tdc_alpha_pt=0.0,
            tdc_alpha_c=1.0,
            tdc_alpha_s=1.0,
            dt=dt,
            cgrav=cgrav,
            m=m,
            scale=scale,
            max_conv_vel=1.0e99,
            include_mlt_corr_to_tdc=True,
            use_tdc_enthalpy_flux_limiter=False,
        )
        res = _turb.set_tdc(params, inp)
        assert isinstance(res.output.grad_t, float)
        assert isinstance(res.output.conv_vel, float)
        _assert_rel(res.output.grad_t, expected_grad_t, 1.0e-12)
        if expected_vc == 0.0:
            assert res.output.conv_vel == 0.0
            assert res.output.mixing_type == _const.MixingType.NO_MIXING
        else:
            _assert_rel(res.output.conv_vel, expected_vc, 1.0e-12)
            assert res.output.mixing_type == _const.MixingType.CONVECTIVE_MIXING
