# SPDX-License-Identifier: LGPL-3.0-or-later
"""Wave 5 four-stage advanced stellar physics, interacting binary, and
asteroseismology benchmarks (`flexmesa.bench.kernels_wave5`).

Evaluates the Three-Tier Hybrid Architecture (`flexmesa._rust` + JAX `float64`)
across Stages 0..3 for the four Wave 5 workloads:
1. `w5_diff_rot`: Roche rotational distortion (`f_P, f_T`), Caplan, Bauer &
   Freeman (2022) strong-coupling OCP self-diffusion $D^*(\\Gamma, \\kappa)$,
   five hydrodynamic rotational instabilities (`DSI`, `SH`, `SSI`, `ES`, `GSF`),
   and the Spruit-Tayler (2002) magnetic dynamo (`D_ST`, `nu_ST`, `B_phi`,
   `B_r`).
2. `w5_riemann_rsp2`: Well-balanced 1D spherical Lagrangian HLLC Riemann
   hydrodynamics, momentum residual assembly, Kuhfuss (1986) / Smolec &
   Moskalik (2008) `RSP2` turbulent convection equations, and Dziembowski
   dimensionless pulsation variables $(V, V_g, A^*, U, c_1)$.
3. `w5_binary_rlof`: Eggleton (1983) Roche-lobe radius, Keplerian orbital
   mechanics, Ritter (1988) optically thin RLOF mass transfer, Bardeen (1972) /
   King & Kolb (1999) / Podsiadlowski (2003) Kerr black hole ISCO spin and
   Eddington accretion, Peters (1964) + Rappaport, Verbunt & Joss (1983)
   orbital $\\dot{J}$, and Hut (1981) tidal eccentricity evolution.
4. `w5_astero_seism`: Cumulative acoustic radius/depth integrals, asymptotic and
   solar-scaled $\\Delta\\nu, \\nu_{\\max}, \\nu_{\\mathrm{ac}}, \\Delta\\Pi_1$,
   Ball & Gizon (2014) 1-term cubic and 2-term combined surface-effect
   frequency corrections, and Roxburgh & Vorontsov (2003) 5-point separation
   ratios $(r_{01}, r_{10})$.
"""

from __future__ import annotations

import collections.abc

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from flexmesa import _rust, astero, binary, constants, star
from flexmesa.bench import harness as _h

type _ArrF64 = npt.NDArray[np.float64]


@jax.jit
def _eval_diff_rot_jax(s: jax.Array) -> jax.Array:
    """Evaluate `w5_diff_rot` across dimensionless radial grid `s`."""
    r = constants.RSUN * (0.08 + 0.88 * s)
    m_grav = constants.MSUN * (0.05 + 0.95 * s * s)
    cgrav = jnp.full_like(s, constants.STANDARD_CGRAV)
    lum = constants.LSUN * (0.04 + 0.96 * s)
    opacity = 0.34 + 1.40 * (1.0 - s)
    rho = jnp.exp(constants.LN10 * (2.0 * (1.0 - s) - 0.8 * s))
    t = jnp.exp(constants.LN10 * (7.15 - 1.35 * s))
    cp = 2.4e8 * (1.0 + 0.25 * s)
    hp = 0.085 * r
    grada = jnp.full_like(s, 0.398)
    gradt = 0.26 + 0.06 * jnp.sin(5.0 * s)
    brunt_n2_t = 1.6e-6 * (0.4 + s)
    brunt_n2_mu = 4.5e-7 * (1.0 - 0.4 * s)
    w_div_wc = 0.12 + 0.58 * s
    dln_om = -0.40 - 0.30 * s
    abar = 1.28 + 0.42 * (1.0 - s)
    zbar = 1.08 + 0.22 * (1.0 - s)
    gam_ocp = 4.0 + 14.0 * (1.0 - s)
    kap_ocp = 0.20 + 0.55 * s

    om_crit = star.eval_omega_crit_jax(cgrav, m_grav, r, lum, opacity, 0.99)
    om = w_div_wc * om_crit
    fp, ft, i_rot_r2 = star.eval_rotation_fp_ft_jax(w_div_wc, 0.75, 0.95)
    d_star = star.eval_caplan_ocp_dstar_jax(gam_ocp, kap_ocp)
    grav = fp * (cgrav * m_grav / (r * r))
    ones = jnp.ones_like(s)
    (
        d_dsi,
        d_sh,
        d_ssi,
        d_es,
        d_gsf,
        d_st,
        nu_st,
        b_phi,
        b_r,
    ) = star.eval_rotational_mixing_zone_jax(
        r,
        hp,
        grav,
        rho,
        t,
        cp,
        opacity,
        ones,
        ones,
        grada,
        gradt,
        brunt_n2_t,
        brunt_n2_mu,
        om,
        dln_om,
        abar,
        zbar,
    )
    return jnp.asarray(
        fp
        + ft
        + i_rot_r2
        + d_star
        + 1.0e-12 * (d_dsi + d_sh + d_ssi + d_es + d_gsf + d_st + nu_st)
        + 1.0e-5 * (b_phi + b_r)
    )


_DIFF_ROT_BATCH_JIT = jax.jit(jax.vmap(_eval_diff_rot_jax, in_axes=(0,)))


@jax.jit
def _eval_riemann_rsp2_jax(s: jax.Array, idx: jax.Array) -> jax.Array:
    """Evaluate `w5_riemann_rsp2` across profile grid `(s, idx)`."""
    nz_f = jnp.float64(s.shape[0])
    r_out = constants.RSUN * (1.0 - 0.85 * (idx / nz_f))
    r_in = constants.RSUN * (1.0 - 0.85 * ((idx + 1.0) / nz_f))
    m_grav = constants.MSUN * ((1.0 - 0.92 * (idx / nz_f)) ** 2)
    rho = jnp.exp(constants.LN10 * (-4.5 + 5.5 * s))
    t = jnp.exp(constants.LN10 * (3.85 + 3.15 * s))
    p = 1.25e8 * rho * t
    cs = jnp.sqrt((5.0 / 3.0) * p / rho)
    u = 1.5e5 * (1.0 - s) * jnp.sin(4.0 * s)
    u_old = 0.98 * u
    dm = (constants.MSUN * 0.85 / nz_f) * (0.4 + 1.2 * s * (1.0 - s))
    cp = 2.1e8 * (1.0 + 0.15 * s)
    grad_t = 0.445 - 0.10 * s
    kappa = 0.40 + 2.5 * ((1.0 - s) ** 2)
    hp = 0.06 * r_out
    dlnd_dt = -1.0e-6 * jnp.cos(3.0 * s)
    cgrav = jnp.full_like(s, constants.STANDARD_CGRAV)

    _, _, _, _, u_f0, p_f0, _ = star.eval_hllc_surface_face_jax(
        p[0],
        rho[0],
        cs[0],
        u[0],
        dm[0],
        r_out[0],
        m_grav[0],
        cgrav[0],
        0.65 * p[0],
    )
    _, _, _, _, u_fi, p_fi, _ = star.eval_hllc_face_jax(
        p[1:],
        rho[1:],
        cs[1:],
        u[1:],
        dm[1:],
        p[:-1],
        rho[:-1],
        cs[:-1],
        u[:-1],
        dm[:-1],
        r_out[1:],
        m_grav[1:],
        cgrav[1:],
    )
    u_face = jnp.concatenate([jnp.expand_dims(u_f0, 0), u_fi])
    p_face = jnp.concatenate([jnp.expand_dims(p_f0, 0), p_fi])

    g_out = (
        constants.STANDARD_CGRAV * m_grav / jnp.maximum(r_out * r_out, 1e-30)
    )
    a_last = jnp.maximum(constants.PI4 * r_out[-1] * r_out[-1], 1.0e-30)
    p_f_last = p_face[-1] + 0.5 * dm[-1] * g_out[-1] / a_last
    p_f_in = jnp.concatenate([p_face[1:], jnp.expand_dims(p_f_last, 0)])
    g_in_int = (
        constants.STANDARD_CGRAV
        * m_grav[1:]
        / jnp.maximum(r_in[:-1] * r_in[:-1], 1.0e-30)
    )
    g_in = jnp.concatenate([g_in_int, jnp.zeros(1, dtype=jnp.float64)])
    g_cell = 0.5 * (g_out + g_in)

    res_mom = star.eval_riemann_momentum_residual_jax(
        u,
        u_old,
        jnp.float64(100.0),
        dm,
        r_out,
        r_in,
        p_face,
        p_f_in,
        p,
        g_cell,
    )
    grad_ad = jnp.full_like(s, 0.395)
    delta = jnp.ones_like(s)
    w_st = star.solve_stationary_rsp2_w_jax(
        t, rho, p, cp, grad_ad, delta, kappa, hp, grad_t
    )
    w = w_st + 2.0e4 * (1.0 - s)
    p_turb, _, _, _, _, _, de_t_dt = star.eval_rsp2_zone_jax(
        w,
        t,
        rho,
        p,
        cp,
        grad_ad,
        delta,
        kappa,
        hp,
        grad_t,
        r_out,
        r_in,
        u,
        0.95 * u,
        dm,
        dlnd_dt,
    )
    v_dz, v_g, a_st, u_dz, c_1 = star.eval_dziembowski_variables_jax(
        r_out,
        m_grav,
        p,
        rho,
        jnp.full_like(s, 5.0 / 3.0),
        1.0e-6 * s,
        cgrav,
        constants.RSUN,
        constants.MSUN,
    )
    return jnp.asarray(
        (u_face / 1.0e6)
        + (p_face / p)
        + (res_mom / 1.0e4)
        + (w / 1.0e5)
        + (p_turb / p)
        + (de_t_dt / 1.0e8)
        + 1.0e-2 * (v_dz + v_g + a_st + u_dz + c_1)
    )


_RIEMANN_RSP2_BATCH_JIT = jax.jit(
    jax.vmap(_eval_riemann_rsp2_jax, in_axes=(0, None))
)


@jax.jit
def _eval_binary_rlof_jax(s: jax.Array) -> jax.Array:
    """Evaluate `w5_binary_rlof` across dimensionless parameter grid `s`."""
    md = constants.MSUN * (1.20 + 6.80 * s)
    ma = constants.MSUN * (0.80 + 4.20 * (1.0 - 0.5 * s))
    per = constants.SECYER * (1.5e-3 + 2.5e-2 * s)
    ecc = 0.05 + 0.35 * s * (1.0 - s)
    rho_s = 1.0e-7 * (1.0 + 2.0 * s)
    p_s = 1.2e5 * (1.0 + 1.5 * s)
    t_s = 5500.0 + 6500.0 * s
    mu_s = 0.61 + 0.04 * (1.0 - s)
    m_bh_i = constants.MSUN * (5.0 + 5.0 * s)
    a_sp_i = 0.15 + 0.70 * s
    m_bh_c = m_bh_i * (1.05 + 0.25 * s)
    kap_s = jnp.full_like(s, 0.34)
    k_div_t = 1.2e-9 * (1.0 + s)
    om_orb = 2.0 * constants.PI / per
    om_star = 1.25 * om_orb

    sep = binary.separation_from_period_jax(md, ma, per)
    rl_d = binary.eval_rlobe_jax(md, ma, sep)
    r_d = rl_d * (0.985 + 0.030 * s)
    rl_gap = binary.eval_rl_relative_gap_jax(r_d, rl_d, ecc)
    j_orb = binary.eval_j_orb_jax(md, ma, sep, ecc)
    mdot_thin = binary.eval_ritter_mdot_thin_jax(
        md, ma, r_d, rl_d, rho_s, p_s, t_s, mu_s, False
    )
    _, a_cur, mdot_edd, eta = binary.eval_bh_spin_and_edd_jax(
        m_bh_i, a_sp_i, m_bh_c, kap_s
    )
    jdot_gr = binary.eval_jdot_gr_jax(md, ma, sep, j_orb)
    jdot_mb = binary.eval_jdot_mb_jax(md, r_d, rl_d, per, 3.0, 0.1, False)
    edot_tid = binary.eval_edot_tidal_hut_jax(
        k_div_t, md, ma, r_d, sep, ecc, om_star, per, 1.0
    )
    return jnp.asarray(
        (rl_d / sep)
        + rl_gap
        + (j_orb / 1.0e53)
        + (-mdot_thin / constants.MSUN * constants.SECYER)
        + a_cur
        + eta
        + (mdot_edd / constants.MSUN * constants.SECYER)
        + (-jdot_gr / 1.0e36)
        + (-jdot_mb / 1.0e36)
        + (-edot_tid * constants.SECYER * 1.0e4)
    )


_BINARY_RLOF_BATCH_JIT = jax.jit(jax.vmap(_eval_binary_rlof_jax, in_axes=(0,)))


@jax.jit
def _eval_astero_seism_jax(s: jax.Array, idx: jax.Array) -> jax.Array:
    """Evaluate `w5_astero_seism` across radial grid `(s, idx)`."""
    nz_f = jnp.float64(s.shape[0])
    r = constants.RSUN * (1.0 - 0.95 * (idx / nz_f))
    r_mid = constants.RSUN * (1.0 - 0.95 * s)
    csound = 1.0e6 * (0.8 + 4.5 * s)
    brunt_n2 = jnp.where(s > 0.35, 2.5e-6 * (s - 0.35), -1.0e-7)

    r_center = jnp.float64(constants.RSUN * 0.05)
    r_phot = jnp.float64(constants.RSUN * 0.995)
    m_star = jnp.float64(1.05 * constants.MSUN)
    teff = jnp.float64(5820.0)
    l_phot = jnp.float64(1.12 * constants.LSUN)
    g_surf = constants.STANDARD_CGRAV * m_star / (r_phot * r_phot)

    tau_r, tau_d, tau_0 = astero.jax_kernels.eval_acoustic_profile_jax(
        r, csound, r_center, r_phot
    )
    dnu_asymp = astero.jax_kernels.eval_delta_nu_asymptotic_jax(tau_0)
    dnu_scal = astero.jax_kernels.eval_delta_nu_scaling_jax(
        m_star, teff, l_phot
    )
    nu_max_v = astero.jax_kernels.eval_nu_max_jax(m_star, r_phot, teff)
    nu_ac_v = astero.jax_kernels.eval_nu_ac_jax(
        g_surf, jnp.float64(1.64), jnp.float64(2.2e-7), jnp.float64(1.1e5)
    )
    delta_pg = astero.jax_kernels.eval_delta_pg_traditional_jax(
        r, r_mid, brunt_n2
    )

    n_ord = jnp.arange(14, 26, dtype=jnp.float64)
    l0_freq = 135.0 * (n_ord + 1.45)
    l0_ine = 1.0e-7 * jnp.exp((3100.0 - l0_freq) / 1000.0)
    l0_sig = jnp.full_like(l0_freq, 0.15)
    target_b1 = -5.125_972_793_567_201_1e-5
    target_b3 = -1.446_307_489_879_336_1e-18
    l0_obs = l0_freq + (target_b1 / l0_freq + target_b3 * (l0_freq**3)) / l0_ine
    l1_mod = 135.0 * (n_ord + 1.93) - 1.8 * jnp.sin(0.3 * n_ord)
    l1_sig = jnp.full_like(l0_freq, 0.18)

    corr_cub, _ = astero.jax_kernels.eval_cubic_corr_jax(
        l0_obs, l0_sig, l0_freq, l0_ine, jnp.float64(1.0)
    )
    corr_comb, _, _ = astero.jax_kernels.eval_combined_corr_jax(
        l0_obs, l0_sig, l0_freq, l0_ine, jnp.float64(1.0)
    )
    r01, sig_r01, r10, sig_r10 = astero.jax_kernels.eval_ratios_r01_r10_jax(
        l0_freq, l0_sig, l1_mod, l1_sig
    )

    return jnp.asarray(
        (jnp.sum(tau_r + tau_d) / nz_f) / 1.0e3
        + (tau_0 / 1.0e3)
        + (dnu_asymp + dnu_scal + nu_max_v + nu_ac_v) / 1.0e3
        + (delta_pg / 1.0e3)
        + (jnp.sum(corr_cub + corr_comb) / 12.0) / 1.0e3
        + jnp.sum(r01 + sig_r01 + r10 + sig_r10) / 10.0
    )


_ASTERO_SEISM_BATCH_JIT = jax.jit(
    jax.vmap(_eval_astero_seism_jax, in_axes=(0, None))
)


def _make_s_and_idx(nz: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build 1D `(nz,)` and batched `(B, nz)` dimensionless grids."""
    idx = jnp.arange(nz, dtype=jnp.float64)
    s = (idx + 0.5) / float(nz)
    s_b = jnp.broadcast_to(s[None, :], (_h.STAGE3_BATCH_SIZE, nz))
    return s, idx, s_b


def _run_w5_four_stages(
    workload: str,
    nz: int,
    n_warmup: int,
    n_repeats: int,
    fn_rust: collections.abc.Callable[[bool], _ArrF64],
    fn_jax: collections.abc.Callable[[], jax.Array],
    fn_batch_jax: collections.abc.Callable[[], jax.Array],
) -> _h.BenchRecord:
    """Measure Stages 0..3 for a 1D Wave 5 workload and compute parity."""
    s0, v0 = _h.measure_and_capture(n_warmup, n_repeats, lambda: fn_rust(False))
    s1, v1 = _h.measure_and_capture(n_warmup, n_repeats, lambda: fn_rust(True))
    s2, v2_j = _h.measure_and_capture(
        n_warmup, n_repeats, fn_jax, record_first_as_compile=True
    )
    s3, v3_b = _h.measure_stage3_batch(n_warmup, n_repeats, fn_batch_jax)
    v2 = np.asarray(v2_j, dtype=np.float64)
    v3 = np.asarray(v3_b[0], dtype=np.float64)
    err = _h.max_err3(
        _h.max_rel_diff(v0, v1),
        _h.max_rel_diff(v0, v2),
        _h.max_rel_diff(v0, v3),
    )
    return _h.BenchRecord(
        workload=workload,
        n_zones=nz,
        stage0=s0,
        stage1=s1,
        stage2=s2,
        stage3=s3,
        max_rel_err=err,
        checksum=float(np.mean(v0)),
    )


def bench_w5_diff_rot(
    nz: int, n_warmup: int = 3, n_repeats: int = 9
) -> _h.BenchRecord:
    """Benchmark Wave 5.1 rotation, diffusion, and dynamo (`w5_diff_rot`)."""
    s_j, _, s_b = _make_s_and_idx(nz)
    return _run_w5_four_stages(
        "w5_diff_rot",
        nz,
        n_warmup,
        n_repeats,
        lambda par: _rust.w5_eval_diff_rot_batch(nz, par),
        lambda: _eval_diff_rot_jax(s_j),
        lambda: _DIFF_ROT_BATCH_JIT(s_b),
    )


def bench_w5_riemann_rsp2(
    nz: int, n_warmup: int = 3, n_repeats: int = 9
) -> _h.BenchRecord:
    """Benchmark Wave 5.2 HLLC Riemann and RSP2 (`w5_riemann_rsp2`)."""
    s_j, idx_j, s_b = _make_s_and_idx(nz)
    return _run_w5_four_stages(
        "w5_riemann_rsp2",
        nz,
        n_warmup,
        n_repeats,
        lambda par: _rust.w5_eval_riemann_rsp2_batch(nz, par),
        lambda: _eval_riemann_rsp2_jax(s_j, idx_j),
        lambda: _RIEMANN_RSP2_BATCH_JIT(s_b, idx_j),
    )


def bench_w5_binary_rlof(
    nz: int, n_warmup: int = 3, n_repeats: int = 9
) -> _h.BenchRecord:
    """Benchmark Wave 5.3 interacting binary physics (`w5_binary_rlof`)."""
    s_j, _, s_b = _make_s_and_idx(nz)
    return _run_w5_four_stages(
        "w5_binary_rlof",
        nz,
        n_warmup,
        n_repeats,
        lambda par: _rust.w5_eval_binary_rlof_batch(nz, par),
        lambda: _eval_binary_rlof_jax(s_j),
        lambda: _BINARY_RLOF_BATCH_JIT(s_b),
    )


def bench_w5_astero_seism(
    nz: int, n_warmup: int = 3, n_repeats: int = 9
) -> _h.BenchRecord:
    """Benchmark Wave 5.4 asteroseismic diagnostics (`w5_astero_seism`)."""
    s_j, idx_j, s_b = _make_s_and_idx(nz)
    s0, c0 = _h.measure_and_capture(
        n_warmup, n_repeats, lambda: _rust.w5_eval_astero_seism(nz)
    )
    s1, c1 = _h.measure_and_capture(
        n_warmup, n_repeats, lambda: _rust.w5_eval_astero_seism(nz)
    )
    s2, c2_j = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: _eval_astero_seism_jax(s_j, idx_j),
        record_first_as_compile=True,
    )
    s3, c3_b = _h.measure_stage3_batch(
        n_warmup, n_repeats, lambda: _ASTERO_SEISM_BATCH_JIT(s_b, idx_j)
    )
    v0 = np.array([float(c0)], dtype=np.float64)
    v1 = np.array([float(c1)], dtype=np.float64)
    v2 = np.array([float(c2_j)], dtype=np.float64)
    v3 = np.array([float(c3_b[0])], dtype=np.float64)
    err = _h.max_err3(
        _h.max_rel_diff(v0, v1),
        _h.max_rel_diff(v0, v2),
        _h.max_rel_diff(v0, v3),
    )
    return _h.BenchRecord(
        workload="w5_astero_seism",
        n_zones=nz,
        stage0=s0,
        stage1=s1,
        stage2=s2,
        stage3=s3,
        max_rel_err=err,
        checksum=float(v0[0]),
    )


def run_all_wave5_benchmarks(
    zone_counts: collections.abc.Sequence[int] = (128, 512, 1024),
    n_warmup: int = 3,
    n_repeats: int = 9,
) -> list[_h.BenchRecord]:
    """Run all four Wave 5 benchmarks across `zone_counts`."""
    records: list[_h.BenchRecord] = []
    for nz in zone_counts:
        records.append(bench_w5_diff_rot(nz, n_warmup, n_repeats))
        records.append(bench_w5_riemann_rsp2(nz, n_warmup, n_repeats))
        records.append(bench_w5_binary_rlof(nz, n_warmup, n_repeats))
        records.append(bench_w5_astero_seism(nz, n_warmup, n_repeats))
    return records
