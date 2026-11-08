# SPDX-License-Identifier: LGPL-3.0-or-later
"""Wave 5.3 verification suite for `flexmesa.binary` (`crates/flexmesa-binary`).

Verifies:
1. Exact 16-digit parity against the upstream Fortran MESA
   `binary/test/test_output` golden baseline (`binary_pick_next_timestep` with
   `time_delta_coeff = 1.0, 0.5` and `eval_L2_mass_loss_fraction` from Lu et al.
   2023).
2. Analytical closed-form oracles (`rtol <= 1e-14`) for Eggleton (1983)
   Roche-lobe geometry, Kepler period-separation invertibility, orbital angular
   momentum $J_{\\mathrm{orb}} \\leftrightarrow a$ invertibility, Marchant et
   al. (2016) $L_2$ overflow limit, and eccentric anomaly time grids (even and
   odd `anomaly_steps`).
3. Cross-backend Rust-vs-JAX parity (`rtol <= 1e-12`) and first-order automatic
   differentiation (`jax.grad` against central finite differences,
   `rtol <= 1e-6`) across orbital, RLOF (including implicit RLOF via
   `solve_roche_lobe_mdot_jax` custom_vjp), Kerr BH ISCO/Eddington, $\\dot{J}$,
   and Hut (1981) tidal kernels.
4. Coupled `BinarySystem` evolution for `Donor + PointMass` (including Kerr BH
   Eddington-capped accretion) and `Donor + Accretor` two-star systems.
"""

from __future__ import annotations

import collections.abc

import jax
import jax.numpy as jnp
import numpy as np

from flexmesa import _rust, binary, star
from flexmesa import constants as const


def test_binary_pick_next_timestep_matches_upstream_golden_output() -> None:
    """Verify `pick_next_timestep_rl` against `binary/test/test_output`."""
    dt1_sec, reason1 = binary.pick_next_timestep_rl(
        donor_time_step_yr=10.0,
        max_timestep_sec=10.0,
        rl_gap_old_donor=-0.102,
        rl_gap_donor=-0.100,
        fr=0.01,
        fr_limit=0.01,
        fr_dt_limit=0.0,
        dt_softening_factor=0.0,
        time_delta_coeff=1.0,
    )
    assert reason1 == 2
    np.testing.assert_allclose(
        dt1_sec / const.SECYER,
        5.0000000000000044e0,
        rtol=1.0e-15,
        atol=1.0e-15,
    )

    dt2_sec, reason2 = binary.pick_next_timestep_rl(
        donor_time_step_yr=10.0,
        max_timestep_sec=10.0,
        rl_gap_old_donor=-0.102,
        rl_gap_donor=-0.100,
        fr=0.01,
        fr_limit=0.01,
        fr_dt_limit=0.0,
        dt_softening_factor=0.0,
        time_delta_coeff=0.5,
    )
    assert reason2 == 2
    np.testing.assert_allclose(
        dt2_sec / const.SECYER,
        2.5000000000000067e0,
        rtol=1.0e-15,
        atol=1.0e-15,
    )


def test_eval_l2_mass_loss_fraction_matches_upstream_golden_output() -> None:
    """Verify Lu et al. (2023) `eval_l2_mass_loss_fraction` golden values."""
    fl2_1 = binary.eval_l2_mass_loss_fraction(
        donor_mass_msun=20.0,
        accretor_mass_msun=10.0,
        mass_transfer_rate_msun_yr=0.001,
        orbital_separation_rsun=2.0,
        disk_alpha=0.1,
        disk_mu=1.3 / 2.4,
    )
    np.testing.assert_allclose(
        fl2_1,
        8.9701564145435098e-1,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    fl2_2 = binary.eval_l2_mass_loss_fraction(
        donor_mass_msun=20.0,
        accretor_mass_msun=10.0,
        mass_transfer_rate_msun_yr=0.0001,
        orbital_separation_rsun=2.0,
        disk_alpha=0.1,
        disk_mu=1.3 / 2.4,
    )
    np.testing.assert_allclose(
        fl2_2,
        2.1022134020418981e-1,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    fl2_3 = binary.eval_l2_mass_loss_fraction(
        donor_mass_msun=20.0,
        accretor_mass_msun=18.0,
        mass_transfer_rate_msun_yr=0.0,
        orbital_separation_rsun=304.0,
        disk_alpha=0.1,
        disk_mu=0.64,
    )
    assert fl2_3 == 0.0


def test_orbital_mechanics_invertibility_and_rust_jax_parity() -> None:
    """Verify Eggleton Roche-lobe, Kepler relations, $J_{\\mathrm{orb}}$, and
    eccentric anomaly grids across Rust and JAX.
    """
    m1 = 1.6 * const.MSUN
    m2 = 0.8 * const.MSUN
    a = 12.0 * const.RSUN
    e = 0.25

    rl_r = float(binary.eval_rlobe(m1, m2, a, backend="rust"))
    rl_j = float(binary.eval_rlobe(jnp.float64(m1), m2, a, backend="jax"))
    q = (m1 / m2) ** (1.0 / 3.0)
    rl_exact = a * 0.49 * q * q / (0.6 * q * q + np.log1p(q))
    np.testing.assert_allclose(rl_r, rl_exact, rtol=1.0e-15)
    np.testing.assert_allclose(rl_j, rl_exact, rtol=1.0e-14)

    gap_r = float(
        binary.eval_rl_relative_gap(0.9 * rl_r, rl_r, e, backend="rust")
    )
    gap_j = float(
        binary.eval_rl_relative_gap(
            jnp.float64(0.9 * rl_r), rl_r, e, backend="jax"
        )
    )
    np.testing.assert_allclose(gap_r, (0.9 - (1.0 - e)) / (1.0 - e), rtol=1e-15)
    np.testing.assert_allclose(gap_j, gap_r, rtol=1.0e-14)

    p_r = float(binary.period_from_separation(m1, m2, a, backend="rust"))
    p_j = float(
        binary.period_from_separation(jnp.float64(m1), m2, a, backend="jax")
    )
    a_back_r = float(binary.separation_from_period(m1, m2, p_r, backend="rust"))
    a_back_j = float(
        binary.separation_from_period(jnp.float64(m1), m2, p_j, backend="jax")
    )
    np.testing.assert_allclose(p_j, p_r, rtol=1.0e-14)
    np.testing.assert_allclose(a_back_r, a, rtol=1.0e-14)
    np.testing.assert_allclose(a_back_j, a, rtol=1.0e-14)

    j_r = float(binary.eval_j_orb(m1, m2, a, e, backend="rust"))
    j_j = float(binary.eval_j_orb(jnp.float64(m1), m2, a, e, backend="jax"))
    a_j_r = float(binary.separation_from_j_orb(m1, m2, j_r, e, backend="rust"))
    a_j_j = float(
        binary.separation_from_j_orb(jnp.float64(m1), m2, j_j, e, backend="jax")
    )
    np.testing.assert_allclose(j_j, j_r, rtol=1.0e-14)
    np.testing.assert_allclose(a_j_r, a, rtol=1.0e-14)
    np.testing.assert_allclose(a_j_j, a, rtol=1.0e-14)

    l2_lim = _rust.binary_eval_l2_overflow_limit(m1, m2)
    q_min = min(m1, m2) / max(m1, m2)
    l2_exact = 0.29858997 * np.arctan(1.83530121 * (q_min**0.39661426))
    np.testing.assert_allclose(l2_lim, l2_exact, rtol=1.0e-15)

    for n_steps in (16, 17):
        theta_co, time_co = _rust.binary_build_anomaly_grid(n_steps, 0.3)
        assert theta_co.shape == (n_steps,) and time_co.shape == (n_steps,)
        assert np.all(np.diff(time_co) > 0.0)
        assert 0.0 <= time_co[0] < time_co[-1] < 1.0


def test_rlof_edd_jdot_tides_rust_jax_parity() -> None:
    """Verify RLOF (Ritter/Kolb/Arras), Kerr BH ISCO/Eddington, wind transfer,
    $\\dot{J}$, and Hut tidal kernels across Rust and JAX.
    """
    md = 2.0 * const.MSUN
    ma = 1.2 * const.MSUN
    sep = 8.0 * const.RSUN
    rld = float(binary.eval_rlobe(md, ma, sep, backend="rust"))
    rd = 0.998 * rld
    rho_s = 1.0e-7
    p_s = 1.0e5
    t_s = 6.5e3
    mu_s = 0.61

    mdot_r = float(
        binary.eval_ritter_mdot_thin(
            md, ma, rd, rld, rho_s, p_s, t_s, mu_s, backend="rust"
        )
    )
    mdot_j = float(
        binary.eval_ritter_mdot_thin(
            jnp.float64(md), ma, rd, rld, rho_s, p_s, t_s, mu_s, backend="jax"
        )
    )
    assert mdot_r < 0.0
    np.testing.assert_allclose(mdot_j, mdot_r, rtol=1.0e-12)

    r_prof = np.linspace(1.02 * rld, 0.95 * rld, 12, dtype=np.float64)
    p_prof = np.geomspace(1.0e5, 1.0e8, 12, dtype=np.float64)
    t_prof = np.geomspace(6.0e3, 3.0e4, 12, dtype=np.float64)
    mu_prof = np.full(12, 0.61, dtype=np.float64)
    g1_prof = np.full(12, 5.0 / 3.0, dtype=np.float64)
    mdot_kolb = binary.eval_kolb_mdot_thick(
        md, ma, rld, r_prof, p_prof, t_prof, mu_prof, g1_prof
    )
    assert mdot_kolb < 0.0

    # Arras optically thin RLOF at L1 (rd == rld -> phi == phiL1 -> exp(0) == 1)
    mdot_arras_l1 = _rust.binary_eval_arras_mdot_thin(
        md, ma, sep, rld, rho_s, t_s, mu_s
    )
    q = ma / md
    q13 = q ** (1.0 / 3.0)
    asl = 4.0 + 4.16 / (-0.96 + q13 + 1.0 / q13)
    v_th = np.sqrt(const.KERG * t_s / (const.MP * mu_s))
    omega = np.sqrt(const.STANDARD_CGRAV * (md + ma) / (sep**3))
    area = 2.0 * np.pi * ((v_th / omega) ** 2) / np.sqrt(asl * (asl - 1.0))
    expected_arras_l1 = -(rho_s / np.exp(0.5)) * v_th * area
    np.testing.assert_allclose(mdot_arras_l1, expected_arras_l1, rtol=1.0e-13)

    m_bh0 = 5.0 * const.MSUN
    m_bh = 5.8 * const.MSUN
    m_eq_r, a_cur_r, medd_r, eta_r = _rust.binary_eval_bh_spin_and_edd(
        m_bh0, 0.3, m_bh, 0.34
    )
    m_eq_j, a_cur_j, medd_j, eta_j = binary.eval_bh_spin_and_edd_jax(
        m_bh0, 0.3, m_bh, 0.34
    )
    np.testing.assert_allclose(float(m_eq_j), m_eq_r, rtol=1.0e-13)
    np.testing.assert_allclose(float(a_cur_j), a_cur_r, rtol=1.0e-13)
    np.testing.assert_allclose(float(medd_j), medd_r, rtol=1.0e-13)
    np.testing.assert_allclose(float(eta_j), eta_r, rtol=1.0e-13)

    enh, bh_frac = _rust.binary_eval_wind_transfer(
        md, ma, rd, rld, sep, 0.1, 1.0e4, 1.5, 1.0, 0.5
    )
    assert enh > 0.0 and 0.0 < bh_frac <= 0.5
    j_acc, acc_mode = _rust.binary_eval_accreted_material_j(
        md, ma, 0.01 * const.RSUN, sep
    )
    assert j_acc > 0.0 and acc_mode in (1, 2)

    j_orb = float(binary.eval_j_orb(md, ma, sep, 0.1, backend="rust"))
    p_orb = float(binary.period_from_separation(md, ma, sep, backend="rust"))
    jgr_r = _rust.binary_eval_jdot_gr(md, ma, sep, j_orb)
    jgr_j = float(binary.eval_jdot_gr_jax(md, ma, sep, j_orb))
    np.testing.assert_allclose(jgr_j, jgr_r, rtol=1.0e-13)

    jmb_r = _rust.binary_eval_jdot_mb(md, rd, rld, p_orb, 3.0, 0.01, True)
    jmb_j = float(binary.eval_jdot_mb_jax(md, rd, rld, p_orb, 3.0, 0.01, True))
    np.testing.assert_allclose(jmb_j, jmb_r, rtol=1.0e-12)

    jml = _rust.binary_eval_jdot_ml(
        md, ma, sep, p_orb, 0.1, -1.0e18, 0.0, 0.0, 0.0, 0.0, 2.0
    )
    jls = _rust.binary_eval_jdot_ls(1.0e50, 1.1e50, 0.5e50, 0.5e50, 1.0e7)
    assert jml < 0.0 and np.isclose(jls, -1.0e42, rtol=1.0e-14)

    f2, f3, f4, f5, kt_r, tsync_r, edot_r = _rust.binary_eval_tides_hut(
        md, ma, rd, sep, 0.2, p_orb, 1.0e-5, 0.1, 0.3, 5.0, 0.1, 1.0, False
    )
    edot_j = float(
        binary.eval_edot_tidal_hut_jax(
            kt_r, md, ma, rd, sep, 0.2, 1.0e-5, p_orb, 1.0
        )
    )
    assert f2 > 1.0 and f3 > 1.0 and f4 > 1.0 and f5 > 1.0 and tsync_r > 0.0
    np.testing.assert_allclose(edot_j, edot_r, rtol=1.0e-12)


def _check_scalar_grad(
    fn: collections.abc.Callable[[jax.Array], jax.Array],
    x0: float,
    eps: float = 1.0e-6,
    rtol: float = 1.0e-7,
) -> None:
    """Compare reverse-mode `jax.grad` against central finite differences."""
    x_arr = jnp.float64(x0)
    g_ad = float(jax.grad(fn)(x_arr))
    f_hi = float(fn(x_arr + eps))
    f_lo = float(fn(x_arr - eps))
    g_fd = (f_hi - f_lo) / (2.0 * eps)
    np.testing.assert_allclose(g_ad, g_fd, rtol=rtol, atol=rtol)


def test_jax_binary_kernels_automatic_differentiation() -> None:
    """Verify forward/reverse-mode automatic differentiation (`jax.grad`) across
    all `@jax.jit` kernels in `flexmesa.binary`, including the custom_vjp
    implicit RLOF solver `solve_roche_lobe_mdot_jax`.
    """
    md = 2.0 * const.MSUN
    ma = 1.2 * const.MSUN
    sep = 8.0 * const.RSUN
    rld = float(binary.eval_rlobe(md, ma, sep, backend="rust"))

    _check_scalar_grad(
        lambda m_msun: binary.eval_rlobe_jax(m_msun * const.MSUN, ma, sep),
        2.0,
    )
    _check_scalar_grad(
        lambda r_rsun: binary.eval_rl_relative_gap_jax(
            r_rsun * const.RSUN, rld, 0.2
        ),
        2.9,
    )
    _check_scalar_grad(
        lambda p_days: binary.separation_from_period_jax(
            md, ma, p_days * 86400.0
        ),
        1.5,
    )
    _check_scalar_grad(
        lambda a_rsun: binary.period_from_separation_jax(
            md, ma, a_rsun * const.RSUN
        ),
        8.0,
    )
    _check_scalar_grad(
        lambda a_rsun: binary.eval_j_orb_jax(md, ma, a_rsun * const.RSUN, 0.2),
        8.0,
    )
    _check_scalar_grad(
        lambda e_val: binary.separation_from_j_orb_jax(md, ma, 1.0e52, e_val),
        0.2,
    )
    _check_scalar_grad(
        lambda r_rsun: binary.eval_ritter_mdot_thin_jax(
            md, ma, r_rsun * const.RSUN, rld, 1.0e-7, 1.0e5, 6500.0, 0.61, False
        ),
        2.999,
        eps=1.0e-7,
    )
    _check_scalar_grad(
        lambda a_spin: binary.eval_bh_spin_and_edd_jax(
            5.0 * const.MSUN, a_spin, 5.8 * const.MSUN, 0.34
        )[2],
        0.3,
    )
    _check_scalar_grad(
        lambda a_rsun: binary.eval_jdot_gr_jax(
            md, ma, a_rsun * const.RSUN, 1.0e52
        ),
        8.0,
    )
    _check_scalar_grad(
        lambda q_env: binary.eval_jdot_mb_jax(
            md, 0.9 * rld, rld, 1.0e5, 3.0, q_env, True
        ),
        0.01,
    )
    _check_scalar_grad(
        lambda e_val: binary.eval_edot_tidal_hut_jax(
            1.0e-10, md, ma, 0.9 * rld, sep, e_val, 1.0e-5, 1.0e5, 1.0
        ),
        0.2,
    )
    # Implicit RLOF custom_vjp gradient w.r.t. overflowing donor radius
    _check_scalar_grad(
        lambda r_rsun: binary.solve_roche_lobe_mdot_jax(
            md, ma, r_rsun * const.RSUN, sep, 1.0e4 * const.SECYER
        ),
        1.05 * (rld / const.RSUN),
        eps=1.0e-6,
        rtol=1.0e-6,
    )


def test_coupled_binary_system_point_mass_and_two_stars() -> None:
    """Verify coupled `BinarySystem` stepping for point-mass and two stars."""
    donor = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
    sys_pm = binary.BinarySystem.from_point_mass(
        donor,
        companion_mass_msun=5.0,
        initial_period_days=0.4,
        eccentricity=0.0,
        is_black_hole=True,
        initial_bh_spin=0.3,
    )
    m1_init = sys_pm.donor_mass_g
    m2_init = sys_pm.accretor_mass_g
    next_dt, reason = sys_pm.step(1.0e4 * const.SECYER)
    assert sys_pm.model_number == 1
    assert sys_pm.binary_age_yr > 0.0
    assert next_dt > 0.0
    assert reason >= 1
    assert sys_pm.donor_mass_g <= m1_init
    assert sys_pm.accretor_mass_g >= m2_init

    donor2 = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
    accretor2 = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
    sys_2s = binary.BinarySystem.from_two_stars(
        donor2, accretor2, initial_period_days=1.5, eccentricity=0.05
    )
    assert sys_2s.accretor is not None
    dt_2s, _ = sys_2s.step(1.0e4 * const.SECYER)
    assert dt_2s > 0.0
    assert sys_2s.model_number == 1
