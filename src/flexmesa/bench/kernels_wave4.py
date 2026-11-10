# SPDX-License-Identifier: LGPL-3.0-or-later
"""Wave 4 four-stage stellar structure and evolution benchmarks.

Evaluates the Three-Tier Hybrid Architecture (`flexmesa._rust` + JAX `float64`)
across Stages 0..3 for the five Wave 4 1D stellar engine workloads:
1. `star_micro_vars`: Zone EOS/opacity/neutrino/nuclear network + face MLT and
   Brunt-Väisälä evaluation (`eval_hydro_vars` vs. `@jax.jit` face kernels).
2. `star_hydro_eqns`: Coupled 1D Lagrangian hydrostatic/thermal residuals and
   block-tridiagonal Jacobian assembly (`eval_equations` vs. `@jax.jit`
   `eval_hydro_residuals_jax`).
3. `star_newton_step`: Single Henyey linearize-and-solve step comparing
   sequential `solve_block_tridiagonal`, 4-thread Hirshman `bcyclic`, `@jax.jit`
   Parallel Cyclic Reduction (`solver_mode="pcr"`), and `custom_vjp` Safe Rust
   `bcyclic` FFI (`solver_mode="rust_ffi"`).
4. `star_remesh`: Dimensionless mesh spacing function $g_k$ (`eval_mesh_gval`)
   and adaptive split/merge planning.
5. `star_evolve_step`: Full structure + nuclear burning + convective mixing
   evolutionary step (`step_struct_burn_mix` + Söderlind & Wang 2006 H211
   digital timestep controller).
"""

from __future__ import annotations

import collections.abc

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from flexmesa import _rust, star
from flexmesa.bench import harness as _h

type _ArrF64 = npt.NDArray[np.float64]


def _face_mlt_obs(*args: jax.Array | float) -> jax.Array:
    """Evaluate `ln_peos + gradt + ln(opacity)` from face MLT/Brunt."""
    out = star.eval_face_mlt_and_brunt_jax(*args)  # type: ignore[arg-type]
    return jnp.asarray(args[7] + out["gradt"] + jnp.log(args[13]))


_FACE_MLT_JIT = jax.jit(_face_mlt_obs)
_FACE_MLT_BATCH_JIT = jax.jit(
    jax.vmap(
        _face_mlt_obs,
        in_axes=(None, 0, 0, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None),
    )
)


def _hydro_res_obs(*args: jax.Array | float) -> jax.Array:
    """Evaluate `|equ| + |diag(dblk)|` via `eval_hydro_residuals_jax`."""
    equ = star.eval_hydro_residuals_jax(*args[:-1])
    return jnp.asarray(jnp.abs(equ) + jnp.abs(args[-1]))


_HYDRO_RES_JIT = jax.jit(_hydro_res_obs)
_HYDRO_RES_BATCH_JIT = jax.jit(
    jax.vmap(
        _hydro_res_obs,
        in_axes=(
            None,
            None,
            None,
            None,
            0,
            0,
            0,
            0,
            None,
            0,
            0,
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )
)

_SOLVE_PCR_JIT = jax.jit(
    lambda l_j, d_j, u_j, equ_j: star.solve_henyey_linear_jax(
        l_j, d_j, u_j, equ_j, solver_mode="pcr"
    )
)
_SOLVE_FFI_JIT = jax.jit(
    lambda l_j, d_j, u_j, equ_j: star.solve_henyey_linear_jax(
        l_j, d_j, u_j, equ_j, solver_mode="rust_ffi"
    )
)

_MESH_GVAL_JIT = jax.jit(star.eval_mesh_gval_jax)
_MESH_GVAL_BATCH_JIT = jax.jit(
    jax.vmap(lambda lnp, lnt: star.eval_mesh_gval_jax(lnp, lnt), in_axes=(0, 0))
)


def _bcast_stage3(x: jax.Array, nz: int) -> jax.Array:
    """Broadcast a 1D `(nz,)` profile across `STAGE3_BATCH_SIZE` stars."""
    return jnp.broadcast_to(x[None, :], (_h.STAGE3_BATCH_SIZE, nz))


def make_wave4_zams_model(nz: int) -> star.StarModel:
    """Construct a resampled $4.0\\,M_\\odot, Z = 0.02$ ZAMS `StarModel`.

    Matches `flexmesa_bench::make_wave4_zams_model(nz)` so cross-language
    benchmarks operate on identical initial stellar profiles.

    Args:
        nz: Target number of Lagrangian zones ($N_{\\mathrm{zones}} \\ge 4$).

    Returns:
        Initialized `StarModel` with `nz` zones.
    """
    base = star.StarModel.from_zams(4.0, 0.02)
    if nz == base.nz:
        return base

    nz_c = max(int(nz), 4)
    base_q = base.q
    base_nz = base.nz
    q_edges = np.empty(nz_c + 1, dtype=np.float64)
    q_edges[0] = 1.0
    q_edges[1] = base_q[1]
    for k in range(2, nz_c):
        frac = float(k - 1) / float(nz_c - 1)
        pos = 1.0 + frac * float(base_nz - 2)
        idx = min(int(np.floor(pos)), base_nz - 2)
        rem = pos - float(idx)
        q_edges[k] = (1.0 - rem) * base_q[idx] + rem * base_q[idx + 1]
    q_edges[nz_c] = 0.0

    dq = np.maximum(q_edges[:-1] - q_edges[1:], 1.0e-12)
    dq = np.ascontiguousarray(dq / np.sum(dq), dtype=np.float64)

    base_dq = base.dq
    lnd = star.interp_q(base_dq, base.lnd, dq)
    ln_t = star.interp_q(base_dq, base.ln_t, dq)
    lum = star.interp_q(base_dq, base.lum, dq)
    dm = base.mstar * dq
    rho = np.exp(lnd)
    dv = (3.0 * dm) / (4.0 * np.pi * np.maximum(rho, 1.0e-30))
    r3 = np.cumsum(dv[::-1])[::-1]
    ln_r = np.ascontiguousarray(np.log(r3) / 3.0, dtype=np.float64)

    base_xa = base.xa
    spec = base.species
    xa = np.empty((nz_c, spec), dtype=np.float64)
    for j in range(spec):
        col = np.ascontiguousarray(base_xa[:, j], dtype=np.float64)
        xa[:, j] = np.maximum(star.interp_q(base_dq, col, dq), 0.0)
    row_sums = np.sum(xa, axis=1, keepdims=True)
    xa = np.ascontiguousarray(xa / np.maximum(row_sums, 1.0e-30))

    model = star.StarModel.from_zams(4.0, 0.02, eval_hydro=False)
    model.set_primary_state(lnd, ln_t, ln_r, lum, dq, xa)
    model.eval_hydro_vars()
    model.save_start_values()
    model.save_old_values()
    return model


def bench_star_micro_vars(
    nz: int, n_warmup: int = 2, n_repeats: int = 7
) -> _h.BenchRecord:
    """Benchmark zone/face stellar microphysics (`star_micro_vars`)."""
    model = make_wave4_zams_model(nz)
    dq_j = jnp.asarray(model.dq)
    lnr_j = jnp.asarray(model.ln_r)
    lum_j = jnp.asarray(model.lum)
    m_j = jnp.asarray(model.m)
    rho_j = jnp.asarray(model.rho)
    t_j = jnp.asarray(model.t)
    peos_j = jnp.asarray(model.peos)
    lnp_j = jnp.asarray(model.ln_peos)
    ga_j = jnp.asarray(model.grada)
    cp_j = jnp.asarray(model.cp)
    cr_j = jnp.asarray(model.chi_rho)
    ct_j = jnp.asarray(model.chi_t)
    mu_j = jnp.asarray(model.mu)
    kap_j = jnp.asarray(model.opacity)
    tau0 = float(model.tau[0])

    nz_val = model.nz
    lnr_b, lum_b = _bcast_stage3(lnr_j, nz_val), _bcast_stage3(lum_j, nz_val)
    rho_b, t_b = _bcast_stage3(rho_j, nz_val), _bcast_stage3(t_j, nz_val)
    peos_b, lnp_b = _bcast_stage3(peos_j, nz_val), _bcast_stage3(lnp_j, nz_val)
    ga_b, cp_b = _bcast_stage3(ga_j, nz_val), _bcast_stage3(cp_j, nz_val)
    cr_b, ct_b = _bcast_stage3(cr_j, nz_val), _bcast_stage3(ct_j, nz_val)
    mu_b, kap_b = _bcast_stage3(mu_j, nz_val), _bcast_stage3(kap_j, nz_val)

    def _s0() -> _ArrF64:
        model.eval_hydro_vars(skip_net=False, skip_net_jacobian=False)
        return model.ln_peos + model.gradt + np.log(model.opacity)

    def _s1() -> _ArrF64:
        model.eval_hydro_vars(skip_net=False, skip_net_jacobian=True)
        return model.ln_peos + model.gradt + np.log(model.opacity)

    s2_args = (
        dq_j,
        lnr_j,
        lum_j,
        m_j,
        rho_j,
        t_j,
        peos_j,
        lnp_j,
        ga_j,
        cp_j,
        cr_j,
        ct_j,
        mu_j,
        kap_j,
        tau0,
    )
    s3_args = (
        dq_j,
        lnr_b,
        lum_b,
        m_j,
        rho_b,
        t_b,
        peos_b,
        lnp_b,
        ga_b,
        cp_b,
        cr_b,
        ct_b,
        mu_b,
        kap_b,
        tau0,
    )
    s0, v0 = _h.measure_and_capture(n_warmup, n_repeats, _s0)
    s1, v1 = _h.measure_and_capture(n_warmup, n_repeats, _s1)
    s2, v2_j = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: _FACE_MLT_JIT(*s2_args),
        record_first_as_compile=True,
    )
    s3, v3_b = _h.measure_stage3_batch(
        n_warmup, n_repeats, lambda: _FACE_MLT_BATCH_JIT(*s3_args)
    )
    v2 = np.asarray(v2_j, dtype=np.float64)
    v3 = np.asarray(v3_b[0], dtype=np.float64)
    err = _h.max_err3(
        _h.max_rel_diff(v0, v1),
        _h.max_rel_diff(v0, v2),
        _h.max_rel_diff(v0, v3),
    )
    return _h.BenchRecord(
        workload="star_micro_vars",
        n_zones=model.nz,
        stage0=s0,
        stage1=s1,
        stage2=s2,
        stage3=s3,
        max_rel_err=err,
        checksum=float(np.sum(v1)),
    )


def bench_star_hydro_eqns(
    nz: int, n_warmup: int = 2, n_repeats: int = 7
) -> _h.BenchRecord:
    """Benchmark coupled 1D Henyey residual & Jacobian (`star_hydro_eqns`)."""
    model = make_wave4_zams_model(nz)
    dq_j = jnp.asarray(model.dq)
    dm_j = jnp.asarray(model.dm)
    dmb_j = jnp.asarray(model.dm_bar)
    m_j = jnp.asarray(model.m)
    lnd_j = jnp.asarray(model.lnd)
    lnt_j = jnp.asarray(model.ln_t)
    lnr_j = jnp.asarray(model.ln_r)
    lum_j = jnp.asarray(model.lum)
    lnds_j = jnp.asarray(model.lnd_start)
    lnp_j = jnp.asarray(model.ln_peos)
    lne_j = jnp.log(jnp.asarray(model.energy))
    es_j = jnp.asarray(model.energy_start)
    gt_j = jnp.asarray(model.gradt)
    en_j = jnp.asarray(model.eps_nuc)
    eneu_j = jnp.asarray(model.non_nuc_neu)
    em_j = jnp.asarray(model.eps_mdot)
    lnps = float(model.ln_peos_surf)
    lnts = float(model.ln_t_surf)
    dt = float(model.dt)
    sys_init = model.eval_equations(include_chem=False)
    diag_j = jnp.asarray(np.diagonal(sys_init.dblk, axis1=1, axis2=2))

    nz_val = model.nz
    lnd_b, lnt_b = _bcast_stage3(lnd_j, nz_val), _bcast_stage3(lnt_j, nz_val)
    lnr_b, lum_b = _bcast_stage3(lnr_j, nz_val), _bcast_stage3(lum_j, nz_val)
    lnp_b = _bcast_stage3(lnp_j, nz_val)
    lne_b = _bcast_stage3(lne_j, nz_val)
    gt_b = _bcast_stage3(gt_j, nz_val)

    def _s0() -> _ArrF64:
        sys = model.eval_equations(include_chem=False)
        diag = np.diagonal(sys.dblk, axis1=1, axis2=2)
        return np.abs(sys.equ) + np.abs(diag)

    s2_args = (
        dq_j,
        dm_j,
        dmb_j,
        m_j,
        lnd_j,
        lnt_j,
        lnr_j,
        lum_j,
        lnds_j,
        lnp_j,
        lne_j,
        es_j,
        gt_j,
        en_j,
        eneu_j,
        em_j,
        lnps,
        lnts,
        dt,
        diag_j,
    )
    s3_args = (
        dq_j,
        dm_j,
        dmb_j,
        m_j,
        lnd_b,
        lnt_b,
        lnr_b,
        lum_b,
        lnds_j,
        lnp_b,
        lne_b,
        es_j,
        gt_b,
        en_j,
        eneu_j,
        em_j,
        lnps,
        lnts,
        dt,
        diag_j,
    )
    s0, v0 = _h.measure_and_capture(n_warmup, n_repeats, _s0)
    s1, v1 = _h.measure_and_capture(n_warmup, n_repeats, _s0)
    s2, v2_j = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: _HYDRO_RES_JIT(*s2_args),
        record_first_as_compile=True,
    )
    s3, v3_b = _h.measure_stage3_batch(
        n_warmup, n_repeats, lambda: _HYDRO_RES_BATCH_JIT(*s3_args)
    )
    v2 = np.asarray(v2_j, dtype=np.float64)
    v3 = np.asarray(v3_b[0], dtype=np.float64)
    err = _h.max_err3(
        _h.max_rel_diff(v0.ravel(), v1.ravel()),
        _h.max_rel_diff(v0.ravel(), v2.ravel()),
        _h.max_rel_diff(v0.ravel(), v3.ravel()),
    )
    return _h.BenchRecord(
        workload="star_hydro_eqns",
        n_zones=model.nz,
        stage0=s0,
        stage1=s1,
        stage2=s2,
        stage3=s3,
        max_rel_err=err,
        checksum=float(np.sum(v1)),
    )


def bench_star_newton_step(
    nz: int, n_warmup: int = 2, n_repeats: int = 7
) -> _h.BenchRecord:
    """Benchmark Henyey linearize-and-solve step (`star_newton_step`)."""
    model = make_wave4_zams_model(nz)
    model.dt = 1.0e2
    sys = model.eval_equations(include_chem=False)
    lblk_f = np.ascontiguousarray(sys.lblk.ravel(), dtype=np.float64)
    dblk_f = np.ascontiguousarray(sys.dblk.ravel(), dtype=np.float64)
    ublk_f = np.ascontiguousarray(sys.ublk.ravel(), dtype=np.float64)
    rhs_f = np.ascontiguousarray(-sys.equ.ravel(), dtype=np.float64)
    l_j = jnp.asarray(sys.lblk)
    d_j = jnp.asarray(sys.dblk)
    u_j = jnp.asarray(sys.ublk)
    equ_j = jnp.asarray(sys.equ)

    s0, v0 = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: _rust.solve_block_tridiagonal(
            4, model.nz, lblk_f, dblk_f, ublk_f, rhs_f
        ),
    )
    s1, v1 = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: _rust.solve_block_tridiag_bcyclic(
            4, model.nz, lblk_f, dblk_f, ublk_f, rhs_f, True
        ),
    )
    s2, v2_j = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: _SOLVE_PCR_JIT(l_j, d_j, u_j, equ_j),
        record_first_as_compile=True,
    )
    s3, v3_j = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: _SOLVE_FFI_JIT(l_j, d_j, u_j, equ_j),
        record_first_as_compile=True,
    )
    v2 = np.asarray(v2_j, dtype=np.float64)
    v3 = np.asarray(v3_j, dtype=np.float64)
    err = _h.max_err3(
        _h.max_rel_diff(v0.ravel(), v1.ravel()),
        _h.max_rel_diff(v0.ravel(), v2.ravel()),
        _h.max_rel_diff(v0.ravel(), v3.ravel()),
    )
    return _h.BenchRecord(
        workload="star_newton_step",
        n_zones=model.nz,
        stage0=s0,
        stage1=s1,
        stage2=s2,
        stage3=s3,
        max_rel_err=err,
        checksum=float(np.sum(np.abs(v1))),
    )


def bench_star_remesh(
    nz: int, n_warmup: int = 2, n_repeats: int = 7
) -> _h.BenchRecord:
    """Benchmark adaptive mesh spacing evaluation (`star_remesh`)."""
    model = make_wave4_zams_model(nz)
    lnp_j = jnp.asarray(model.ln_peos)
    lnt_j = jnp.asarray(model.ln_t)
    lnp_b = _bcast_stage3(lnp_j, model.nz)
    lnt_b = _bcast_stage3(lnt_j, model.nz)

    s0, v0_j = _h.measure_and_capture(
        n_warmup, n_repeats, lambda: model.eval_mesh_gval_jax()
    )
    s1, v1_j = _h.measure_and_capture(
        n_warmup, n_repeats, lambda: _MESH_GVAL_JIT(lnp_j, lnt_j)
    )
    s2, v2_j = _h.measure_and_capture(
        n_warmup,
        n_repeats,
        lambda: _MESH_GVAL_JIT(lnp_j, lnt_j),
        record_first_as_compile=True,
    )
    s3, v3_b = _h.measure_stage3_batch(
        n_warmup, n_repeats, lambda: _MESH_GVAL_BATCH_JIT(lnp_b, lnt_b)
    )
    v0 = np.asarray(v0_j, dtype=np.float64)
    v1 = np.asarray(v1_j, dtype=np.float64)
    v2 = np.asarray(v2_j, dtype=np.float64)
    v3 = np.asarray(v3_b[0], dtype=np.float64)
    err = _h.max_err3(
        _h.max_rel_diff(v0, v1),
        _h.max_rel_diff(v0, v2),
        _h.max_rel_diff(v0, v3),
    )
    return _h.BenchRecord(
        workload="star_remesh",
        n_zones=model.nz,
        stage0=s0,
        stage1=s1,
        stage2=s2,
        stage3=s3,
        max_rel_err=err,
        checksum=float(np.sum(v2)),
    )


def bench_star_evolve_step(
    nz: int, n_warmup: int = 2, n_repeats: int = 7
) -> _h.BenchRecord:
    """Benchmark full structure + burn + mix step (`star_evolve_step`)."""
    base = make_wave4_zams_model(nz)
    lnd0 = base.lnd.copy()
    lnt0 = base.ln_t.copy()
    lnr0 = base.ln_r.copy()
    lum0 = base.lum.copy()
    dq0 = base.dq.copy()
    xa0 = base.xa.copy()
    dt = float(base.dt)

    def _run(op_split: bool, use_bcyc: bool) -> _ArrF64:
        base.set_primary_state(lnd0, lnt0, lnr0, lum0, dq0, xa0)
        base.step_struct_burn_mix(
            dt, op_split_burn=op_split, use_bcyclic=use_bcyc
        )
        return base.ln_t + base.lnd + base.xa[:, 0]

    s0, v0 = _h.measure_and_capture(
        n_warmup, n_repeats, lambda: _run(False, False)
    )
    s1, v1 = _h.measure_and_capture(
        n_warmup, n_repeats, lambda: _run(False, True)
    )
    s2, v2 = _h.measure_and_capture(
        n_warmup, n_repeats, lambda: _run(False, True)
    )
    lnd_j, lnt_j = jnp.asarray(base.lnd), jnp.asarray(base.ln_t)
    lnr_j, lum_j = jnp.asarray(base.ln_r), jnp.asarray(base.lum)
    lnd0_j, lnt0_j = jnp.asarray(lnd0), jnp.asarray(lnt0)
    lnr0_j, lum0_j = jnp.asarray(lnr0), jnp.asarray(lum0)

    @jax.jit
    def _step_filter() -> jax.Array:
        vc = star.eval_varcontrol_jax(
            lnd_j, lnt_j, lnr_j, lum_j, lnd0_j, lnt0_j, lnr0_j, lum0_j
        )
        return jnp.asarray(
            star.eval_h211_timestep_jax(dt, dt, vc / 1.0e-4, 1.0)
        )

    def _run_hybrid() -> _ArrF64:
        out = _run(False, True)
        _step_filter().block_until_ready()
        return out

    s3, v3 = _h.measure_and_capture(
        n_warmup, n_repeats, _run_hybrid, record_first_as_compile=True
    )
    err = _h.max_err3(
        _h.max_rel_diff(v0, v1),
        _h.max_rel_diff(v0, v2),
        _h.max_rel_diff(v0, v3),
    )
    return _h.BenchRecord(
        workload="star_evolve_step",
        n_zones=base.nz,
        stage0=s0,
        stage1=s1,
        stage2=s2,
        stage3=s3,
        max_rel_err=err,
        checksum=float(np.sum(v1)),
    )


def run_all_wave4_benchmarks(
    zone_counts: collections.abc.Sequence[int] = (128, 512, 1024),
    n_warmup: int = 2,
    n_repeats: int = 7,
) -> list[_h.BenchRecord]:
    """Run all five Wave 4 1D stellar engine benchmarks across `zone_counts`."""
    records: list[_h.BenchRecord] = []
    for nz in zone_counts:
        records.append(bench_star_micro_vars(nz, n_warmup, n_repeats))
        records.append(bench_star_hydro_eqns(nz, n_warmup, n_repeats))
        records.append(bench_star_newton_step(nz, n_warmup, n_repeats))
        records.append(bench_star_remesh(nz, n_warmup, n_repeats))
        records.append(bench_star_evolve_step(nz, n_warmup, n_repeats))
    return records
