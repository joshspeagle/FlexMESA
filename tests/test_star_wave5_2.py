# SPDX-License-Identifier: LGPL-3.0-or-later
"""Verification tests for Wave 5.2: HLLC Riemann hydrodynamics, `RSP2`
time-dependent turbulent convection, and `GYRE`/`FGONG`/`OSC` pulsation
structure exporters (Issues #80, #4, and #5).
"""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import jax.test_util
import numpy as np

from flexmesa import _rust, star

_GOLDEN_DIR = pathlib.Path(__file__).resolve().parent / "fixtures" / "golden"


def test_hllc_riemann_face_and_surface_match_jax_and_preserve_hydrostatic() -> (
    None
):
    """Verify HLLC Riemann face/surface kernels, Rust-vs-JAX parity, exact
    hydrostatic equilibrium preservation, and autodiff vs. finite differences.
    """
    r_face = 5.0e10
    m_grav = 1.0e33
    cgrav = 6.67430e-8
    dm = 1.0e30
    area = 4.0 * np.pi * r_face * r_face
    g_over_a = cgrav * m_grav / (r_face * r_face * area)
    p_star = 1.0e15
    p_l = p_star + 0.5 * dm * g_over_a
    p_r = p_star - 0.5 * dm * g_over_a

    # 1. Exact hydrostatic equilibrium: u_L = u_R = 0 and P_L^* == P_R^*
    f_rust = _rust.star_eval_hllc_face(
        p_l,
        10.0,
        5.0e7,
        0.0,
        dm,
        p_r,
        8.0,
        4.5e7,
        0.0,
        dm,
        r_face,
        m_grav,
        cgrav,
    )
    f_jax = star.eval_hllc_face_jax(
        jnp.asarray(p_l),
        jnp.asarray(10.0),
        jnp.asarray(5.0e7),
        jnp.asarray(0.0),
        jnp.asarray(dm),
        jnp.asarray(p_r),
        jnp.asarray(8.0),
        jnp.asarray(4.5e7),
        jnp.asarray(0.0),
        jnp.asarray(dm),
        jnp.asarray(r_face),
        jnp.asarray(m_grav),
        jnp.asarray(cgrav),
    )
    for r_val, j_val in zip(f_rust, f_jax, strict=True):
        np.testing.assert_allclose(
            r_val, float(j_val), rtol=1.0e-14, atol=1.0e-14
        )
    assert abs(f_rust[4]) <= 1.0e-14
    assert abs(f_rust[5] - p_star) / p_star <= 1.0e-14

    # 2. Outward pressure wave: P_L^* > P_R^* drives u_face > 0
    f_shock_r = _rust.star_eval_hllc_face(
        1.5 * p_l,
        12.0,
        5.5e7,
        1.0e5,
        dm,
        p_r,
        8.0,
        4.5e7,
        -2.0e4,
        dm,
        r_face,
        m_grav,
        cgrav,
    )
    f_shock_j = star.eval_hllc_face_jax(
        jnp.asarray(1.5 * p_l),
        jnp.asarray(12.0),
        jnp.asarray(5.5e7),
        jnp.asarray(1.0e5),
        jnp.asarray(dm),
        jnp.asarray(p_r),
        jnp.asarray(8.0),
        jnp.asarray(4.5e7),
        jnp.asarray(-2.0e4),
        jnp.asarray(dm),
        jnp.asarray(r_face),
        jnp.asarray(m_grav),
        jnp.asarray(cgrav),
    )
    for r_val, j_val in zip(f_shock_r, f_shock_j, strict=True):
        np.testing.assert_allclose(r_val, float(j_val), rtol=1.0e-14)
    assert f_shock_r[4] > 0.0

    # Central finite-difference verification of `eval_hllc_face_jax`
    jax.test_util.check_grads(
        lambda pl, pr: star.eval_hllc_face_jax(
            pl,
            jnp.asarray(12.0),
            jnp.asarray(5.5e7),
            jnp.asarray(1.0e5),
            jnp.asarray(dm),
            pr,
            jnp.asarray(8.0),
            jnp.asarray(4.5e7),
            jnp.asarray(-2.0e4),
            jnp.asarray(dm),
            jnp.asarray(r_face),
            jnp.asarray(m_grav),
            jnp.asarray(cgrav),
        )[4],
        (jnp.asarray(1.5 * p_l), jnp.asarray(p_r)),
        order=1,
        rtol=1.0e-6,
    )

    # 3. Surface HLLC boundary state and spherical geometric cancellation
    s_rust = _rust.star_eval_hllc_surface_face(
        1.0e5, 1.0e-7, 1.0e6, 0.0, 1.0e24, 7.0e10, 2.0e33, cgrav, 8.0e4
    )
    s_jax = star.eval_hllc_surface_face_jax(
        jnp.asarray(1.0e5),
        jnp.asarray(1.0e-7),
        jnp.asarray(1.0e6),
        jnp.asarray(0.0),
        jnp.asarray(1.0e24),
        jnp.asarray(7.0e10),
        jnp.asarray(2.0e33),
        jnp.asarray(cgrav),
        jnp.asarray(8.0e4),
    )
    for r_val, j_val in zip(s_rust, s_jax, strict=True):
        np.testing.assert_allclose(r_val, float(j_val), rtol=1.0e-14)

    res_r = _rust.star_eval_riemann_momentum_residual(
        0.0, 0.0, 10.0, 1.0e28, 5.0e10, 4.9e10, 1.0e14, 1.0e14, 1.0e14, 0.0
    )
    res_j = float(
        star.eval_riemann_momentum_residual_jax(
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray(10.0),
            jnp.asarray(1.0e28),
            jnp.asarray(5.0e10),
            jnp.asarray(4.9e10),
            jnp.asarray(1.0e14),
            jnp.asarray(1.0e14),
            jnp.asarray(1.0e14),
            jnp.asarray(0.0),
        )
    )
    assert abs(res_r) == 0.0
    assert abs(res_j) == 0.0


def test_rsp2_zone_face_and_stationary_solver_match_jax_and_autodiff() -> None:
    """Verify `RSP2` zone, face, and stationary quadratic pre-solver across
    Safe Rust and JAX, including subadiabatic gradient safety (`gammar=0.0`).
    """
    t = 1.0e5
    rho = 1.0e-5
    p = 1.0e8
    cp = 3.0e8
    grad_ad = 0.40
    delta = 1.2
    kappa = 2.0
    hp_cell = 1.0e9
    grad_t = 0.55

    w_eq_r = _rust.star_solve_stationary_rsp2_w(
        t, rho, p, cp, grad_ad, delta, kappa, hp_cell, grad_t
    )
    w_eq_j = float(
        star.solve_stationary_rsp2_w_jax(
            jnp.asarray(t),
            jnp.asarray(rho),
            jnp.asarray(p),
            jnp.asarray(cp),
            jnp.asarray(grad_ad),
            jnp.asarray(delta),
            jnp.asarray(kappa),
            jnp.asarray(hp_cell),
            jnp.asarray(grad_t),
        )
    )
    assert w_eq_r > 0.0
    np.testing.assert_allclose(w_eq_r, w_eq_j, rtol=1.0e-14)

    z_r = _rust.star_eval_rsp2_zone(
        w_eq_r,
        t,
        rho,
        p,
        cp,
        grad_ad,
        delta,
        kappa,
        hp_cell,
        grad_t,
        5.01e10,
        5.0e10,
        0.0,
        0.0,
        1.0e28,
        0.0,
    )
    z_j = star.eval_rsp2_zone_jax(
        jnp.asarray(w_eq_r),
        jnp.asarray(t),
        jnp.asarray(rho),
        jnp.asarray(p),
        jnp.asarray(cp),
        jnp.asarray(grad_ad),
        jnp.asarray(delta),
        jnp.asarray(kappa),
        jnp.asarray(hp_cell),
        jnp.asarray(grad_t),
        jnp.asarray(5.01e10),
        jnp.asarray(5.0e10),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(1.0e28),
        jnp.asarray(0.0),
    )
    for r_val, j_val in zip(z_r, z_j, strict=True):
        np.testing.assert_allclose(
            r_val, float(j_val), rtol=1.0e-13, atol=1.0e-14
        )
    assert abs(z_r[6]) / max(abs(z_r[3]), abs(z_r[4]), 1.0) <= 1.0e-13

    # Face evaluation parity
    face_args = (5.0e10, 1.0e33, 6.67430e-8, 1.0e8, 1.0e-5, 1.0e5, 3.0e8, 0.40)
    face_args2 = (2.0, w_eq_r, 0.8 * w_eq_r, (1.05e5) ** 4, (0.95e5) ** 4)
    face_args3 = (18.5, 18.3, z_r[1], -1.0e-5, 1.0e28)
    all_face = (*face_args, *face_args2, *face_args3)
    f_r = _rust.star_eval_rsp2_face(*all_face)
    f_j = star.eval_rsp2_face_jax(*(jnp.asarray(v) for v in all_face))
    for r_val, j_val in zip(f_r, f_j, strict=True):
        np.testing.assert_allclose(r_val, float(j_val), rtol=1.0e-13)

    # Central finite-difference verification of `solve_stationary_rsp2_w_jax`
    jax.test_util.check_grads(
        lambda gt_in: star.solve_stationary_rsp2_w_jax(
            jnp.asarray(t),
            jnp.asarray(rho),
            jnp.asarray(p),
            jnp.asarray(cp),
            jnp.asarray(grad_ad),
            jnp.asarray(delta),
            jnp.asarray(kappa),
            jnp.asarray(hp_cell),
            gt_in,
        ),
        (jnp.asarray(grad_t),),
        order=1,
        rtol=1.0e-6,
    )

    # Verify subadiabatic zone (grad_t < grad_ad) with gammar=0.0 produces
    # exact 0.0 and finite zero gradient (no 0 * inf = NaN propagation)
    grad_sub = jax.grad(
        lambda gt_in: star.solve_stationary_rsp2_w_jax(
            jnp.asarray(t),
            jnp.asarray(rho),
            jnp.asarray(p),
            jnp.asarray(cp),
            jnp.asarray(grad_ad),
            jnp.asarray(delta),
            jnp.asarray(kappa),
            jnp.asarray(hp_cell),
            gt_in,
            gammar=0.0,
        )
    )(jnp.asarray(0.25))
    assert jnp.isfinite(grad_sub)
    assert float(grad_sub) == 0.0

    # Verify `jax_rsp` flux limiter C^2 smoothness, Uq_face, turbulent residual,
    # and Stellingwerf / Richtmyer-Morton HYD fluxes + autodiff
    x_pts = jnp.array([0.80, 0.95, 0.975, 1.0, 1.2], dtype=jnp.float64)
    fl_j, dfl_j = star.rsp2_flux_limiter_jax(x_pts)
    np.testing.assert_allclose(
        np.asarray(fl_j[:2]), np.array([0.80, 0.95]), rtol=1.0e-15
    )
    np.testing.assert_allclose(
        np.asarray(fl_j[3:]), np.array([1.0, 1.0]), rtol=1.0e-15
    )
    np.testing.assert_allclose(
        np.asarray(dfl_j[:2]), np.array([1.0, 1.0]), rtol=1.0e-15
    )
    np.testing.assert_allclose(
        np.asarray(dfl_j[3:]), np.array([0.0, 0.0]), atol=1.0e-15
    )
    np.testing.assert_allclose(
        float(
            jax.grad(lambda x_in: star.rsp2_flux_limiter_jax(x_in)[0])(0.975)
        ),
        float(dfl_j[2]),
        rtol=1.0e-15,
    )
    jax.test_util.check_grads(
        lambda x_in: star.rsp2_flux_limiter_jax(x_in)[0],
        (jnp.asarray(0.975, dtype=jnp.float64),),
        order=1,
        eps=1.0e-7,
        rtol=1.0e-8,
    )

    uq_j = float(
        star.compute_rsp2_uq_face_jax(
            jnp.asarray(1.2e40),
            jnp.asarray(1.0e40),
            jnp.asarray(3.0e12),
            jnp.asarray(2.0e28),
        )
    )
    np.testing.assert_allclose(
        uq_j, 4.0 * np.pi * 0.2e40 / (3.0e12 * 2.0e28), rtol=1.0e-15
    )

    res_et = star.eval_rsp2_turbulent_eqn_residual_jax(
        jnp.asarray(w_eq_r),
        jnp.asarray(w_eq_r),
        jnp.asarray(rho),
        jnp.asarray(rho),
        jnp.asarray(z_r[0]),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(z_r[3] - z_r[4] - z_r[5]),
        jnp.asarray(z_r[2]),
        jnp.asarray(1.0e28),
        jnp.asarray(1.0),
        jnp.asarray(cp),
        jnp.asarray(t),
    )
    assert abs(float(res_et)) <= 1.0e-12
    res_et_pert = star.eval_rsp2_turbulent_eqn_residual_jax(
        jnp.asarray(1.1 * w_eq_r),
        jnp.asarray(w_eq_r),
        jnp.asarray(rho),
        jnp.asarray(rho),
        jnp.asarray(z_r[0]),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(z_r[3] - z_r[4] - z_r[5]),
        jnp.asarray(z_r[2]),
        jnp.asarray(1.0e28),
        jnp.asarray(1.0),
        jnp.asarray(cp),
        jnp.asarray(t),
    )
    np.testing.assert_allclose(
        float(res_et_pert),
        ((1.21 - 1.0) * (w_eq_r**2) - (z_r[3] - z_r[4] - z_r[5] + z_r[2]))
        / (cp * t),
        rtol=1.0e-14,
    )

    def _rsp2_grad_pytree(
        scale: jax.Array,
    ) -> tuple[
        tuple[jax.Array, ...],
        tuple[jax.Array, ...],
        jax.Array,
    ]:
        w_in = scale * (1.1 * w_eq_r)
        z_tail = (t, rho, p, cp, grad_ad, delta, kappa, hp_cell, grad_t)
        z_geom = (5.05e10, 5.00e10, -2.0e5, -1.5e5, 1.0e28, 0.0)
        z_out = star.eval_rsp2_zone_jax(w_in, *z_tail, *z_geom)
        f_head = (5.0e10, 1.0e33, 6.67430e-8, p, rho, t, cp, grad_ad, kappa)
        f_tail = ((1.05e5) ** 4, (0.95e5) ** 4, 18.5, 18.3, z_r[1], -1e-5, 1e28)
        f_out = star.eval_rsp2_face_jax(*f_head, w_in, 0.8 * w_in, *f_tail)
        et_args = (w_eq_r, rho, 0.97 * rho, z_r[0], 1.0e34, 0.8e34)
        et_args2 = (z_r[3] - z_r[4] - z_r[5], z_r[2], 1.0e28, 1.0, cp, t)
        et_out = star.eval_rsp2_turbulent_eqn_residual_jax(
            w_in, *et_args, *et_args2
        )
        return z_out, f_out, et_out

    jax.test_util.check_grads(
        _rsp2_grad_pytree,
        (jnp.asarray(1.0, dtype=jnp.float64),),
        order=1,
        eps=1.0e-6,
        rtol=1.0e-6,
    )
    jax.test_util.check_grads(
        lambda lt_in: sum(
            star.eval_rsp_hyd_fluxes_jax(
                jnp.asarray(3.0e12),
                0.98 * (10.0**lt_in),
                10.0**lt_in,
                jnp.asarray(1.5),
                jnp.asarray(1.8),
                jnp.asarray(rho),
                0.99 * (10.0**lt_in),
                jnp.asarray(f_r[2]),
                jnp.asarray(0.8 * w_eq_r),
                jnp.asarray(w_eq_r),
                jnp.asarray(hp_cell),
                jnp.asarray(1.0e28),
                jnp.asarray(5.0e5),
                jnp.asarray(2.0e6),
                jnp.asarray(rho),
            )
        ),
        (jnp.asarray(5.0, dtype=jnp.float64),),
        order=1,
        eps=1.0e-6,
        rtol=1.0e-6,
    )


def test_star_model_riemann_profile_and_pulse_exporters() -> None:
    """Verify full-profile HLLC evaluation, Dziembowski pulsation variables,
    double-point insertion, and `GYRE` / `FGONG` / `OSC` ASCII exporters.
    """
    model = star.StarModel.from_zams()
    model.eval_hydro_vars()

    faces, residuals = model.eval_riemann_profile(p_surf_bc=model.peos[0])
    assert faces.shape == (model.nz, 7)
    assert residuals.shape == (model.nz,)
    assert np.all(np.isfinite(faces))
    assert np.all(np.isfinite(residuals))

    pts = model.build_pulse_points(add_center_point=True)
    n_pts = model.nz + 1
    assert pts.shape == (n_pts, 26)
    assert pts[0, 0] == 0.0
    assert abs(pts[0, 24] - 3.0) <= 1.0e-14
    assert np.all(np.diff(pts[:, 0]) >= 0.0)
    np.testing.assert_allclose(
        pts[:, 18] + pts[:, 19] + pts[:, 20], 1.0, rtol=1.0e-3
    )

    # Compare non-center Dziembowski columns against face-interpolated pts[1:]
    v_j, vg_j, a_j, u_j, c1_j = star.eval_dziembowski_variables_jax(
        jnp.asarray(pts[1:, 0]),
        jnp.asarray(pts[1:, 1]),
        jnp.asarray(pts[1:, 3]),
        jnp.asarray(pts[1:, 5]),
        jnp.asarray(pts[1:, 8]),
        jnp.asarray(pts[1:, 7]),
        jnp.full(model.nz, 6.67430e-8, dtype=jnp.float64),
        float(model.r[0]),
        float(model.mstar),
    )
    np.testing.assert_allclose(pts[1:, 21], np.asarray(v_j), rtol=1.0e-12)
    np.testing.assert_allclose(pts[1:, 22], np.asarray(vg_j), rtol=1.0e-12)
    np.testing.assert_allclose(
        pts[1:, 23], np.asarray(a_j), rtol=1.0e-12, atol=1.0e-14
    )
    np.testing.assert_allclose(pts[1:, 24], np.asarray(u_j), rtol=1.0e-12)
    np.testing.assert_allclose(pts[1:, 25], np.asarray(c1_j), rtol=1.0e-12)

    # Verify double-point insertion preserves X + Y + Z = 1 on all points
    pts_dbl = model.build_pulse_points(
        add_center_point=True, add_double_points=True, grad_mu_threshold=0.0
    )
    assert pts_dbl.shape[0] > n_pts
    np.testing.assert_allclose(
        pts_dbl[:, 18] + pts_dbl[:, 19] + pts_dbl[:, 20], 1.0, rtol=1.0e-3
    )

    gyre_txt = model.to_gyre_str(add_center_point=True)
    assert len(gyre_txt.splitlines()) == 1 + n_pts
    fgong_txt = model.to_fgong_str(add_center_point=True)
    assert len(fgong_txt.splitlines()) == 5 + 3 + 8 * n_pts
    osc_txt = model.to_osc_str(add_center_point=True)
    assert len(osc_txt.splitlines()) == 6 + 3 + 8 * n_pts
    gsm_bytes = model.to_gsm_bytes(add_center_point=True)
    assert gsm_bytes.startswith(b"# GSM")
    saio_txt = model.to_saio_str(add_center_point=True)
    assert len(saio_txt.splitlines()) >= 1 + 4 * model.nz
    cafein_txt = model.to_cafein_str(add_center_point=True)
    assert len(cafein_txt.splitlines()) >= n_pts
    gr1d_txt = model.to_gr1d_str(add_center_point=True)
    assert len(gr1d_txt.splitlines()) >= model.nz
    for fmt in ("gyre", "fgong", "osc", "gsm", "saio", "cafein", "gr1d"):
        assert len(model.to_pulse_str(fmt)) > 100


def test_gyre_adipls_bvp_and_stella_golden_parity() -> None:
    """Verify 34-mode `GYRE` BVP, 46-mode `ADIPLS` BVP, and `STELLA` Sobolev
    golden parity from Python and JAX.
    """
    gyre_dir = _GOLDEN_DIR / "gyre"
    mesa_str = (gyre_dir / "model.dat").read_text(encoding="utf-8")
    modes_gyre = []
    for deg in (0, 1):
        modes_gyre.extend(
            star.solve_gyre_mesa_modes(mesa_str, deg, 1.0, 20.0, 200, False)
        )
    assert len(modes_gyre) == 34

    adipls_dir = _GOLDEN_DIR / "adipls"
    fgong_str = (adipls_dir / "Gemma.fgong").read_text(encoding="utf-8")
    modes_adipls = []
    for deg in (0, 1, 2):
        modes_adipls.extend(
            star.solve_adipls_fgong_modes(fgong_str, deg, 220.0, 1000.0, 500)
        )
    assert len(modes_adipls) == 46

    stella_dir = _GOLDEN_DIR / "stella"
    hyd_str = (stella_dir / "mesa.hyd").read_text(encoding="utf-8")
    abn_str = (stella_dir / "mesa.abn").read_text(encoding="utf-8")
    eta_str = (stella_dir / "FeII_5169_eta.dat").read_text(encoding="utf-8")
    lbol, vel = star.run_stella_from_hyd_abn(
        hyd_str, abn_str, eta_str, [float(d) for d in range(1, 31)]
    )
    assert len(lbol) == 30
    assert vel

    model = star.StarModel.from_zams()
    hyd_out, abn_out = model.to_stella_pair()
    assert len(hyd_out.splitlines()) == 401
    assert len(abn_out.splitlines()) == 400

    # Verify JAX Sobolev optical depth autodiff
    log_rho_j = jnp.array([-13.0, -12.3], dtype=jnp.float64)
    xfe_j = jnp.array([1.2e-3, 1.2e-3], dtype=jnp.float64)
    eta_j = jnp.array([1.0e-2, 2.0e-2], dtype=jnp.float64)
    tau_j = star.eval_sobolev_feii_5169_tau_jax(
        10.0**log_rho_j, xfe_j, eta_j, 10.0 * 86400.0
    )
    assert jnp.all(tau_j > 0.0)
    jax.test_util.check_grads(
        lambda lr_in: jnp.sum(
            star.eval_sobolev_feii_5169_tau_jax(
                10.0**lr_in, xfe_j, eta_j, 10.0 * 86400.0
            )
        ),
        (log_rho_j,),
        order=1,
        rtol=1.0e-6,
    )
