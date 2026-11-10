// SPDX-License-Identifier: LGPL-3.0-or-later
//! Wave 4 four-stage 1D stellar structure and evolution benchmarks
//! (`star_micro_vars`, `star_hydro_eqns`, `star_newton_step`, `star_remesh`,
//! and `star_evolve_step`).
//!
//! Evaluates the five Wave 4 `flexmesa-star` engine workloads on a resampled
//! $4.0\,M_\odot, Z = 0.02$ Zero-Age Main Sequence (`ZAMS`) model across four
//! progressive optimization stages in Safe Rust:
//! 1. `bench_star_micro_vars`: Zone EOS (`FreeEOS`/`HELM`), Rosseland opacity,
//!    thermal neutrino losses, 8-isotope `basic.net` nuclear burning, and
//!    face-centered `Cox` MLT + Brunt-Väisälä evaluation (`set_hydro_vars`).
//! 2. `bench_star_hydro_eqns`: Coupled 1D Lagrangian hydrostatic/thermal
//!    residual and $(4 \times 4)$ block-tridiagonal Jacobian assembly
//!    (`eval_henyey_equations`).
//! 3. `bench_star_newton_step`: Single linearize-and-solve Henyey iteration
//!    comparing sequential `solve_block_tridiagonal` against 4-thread Hirshman
//!    et al. (2010) `solve_block_tridiag_bcyclic` odd-even cyclic reduction.
//! 4. `bench_star_remesh`: Dimensionless mesh spacing function $g_k$
//!    evaluation (`eval_mesh_gval`), split/merge cell planning (`plan_mesh`),
//!    and conservative Lagrangian profile remapping (`do_adjust_mesh`).
//! 5. `bench_star_evolve_step`: Full operator-split/coupled structure +
//!    nuclear burning + convective mixing step (`do_struct_burn_mix`).

use flexmesa_mtx::{solve_block_tridiag_bcyclic, solve_block_tridiagonal};
use flexmesa_star::{
    do_adjust_mesh, do_struct_burn_mix, eval_henyey_equations, eval_mesh_gval,
    interp_q, load_zams_model, plan_mesh, set_hydro_vars, HenyeySystem,
    MeshFunctionConfig, StarControls, StarJobControls, StarModel,
};

use crate::{max_err3, max_rel_diff, measure_us, stage3_pool, BenchRecord};

/// Constructs a resampled $4.0\,M_\odot, Z = 0.02$ ZAMS [`StarModel`] with
/// `nz` Lagrangian zones by interpolating along the cumulative ZAMS mass grid.
#[must_use]
pub fn make_wave4_zams_model(nz: usize) -> StarModel {
    let controls = StarControls {
        initial_mass: 4.0,
        initial_z: 0.02,
        ..StarControls::default()
    };
    let job = StarJobControls::default();
    let base = load_zams_model(controls.clone(), job.clone())
        .expect("valid 4.0 Msun ZAMS model");
    if nz == base.nz {
        return base;
    }

    let nz_clamped = nz.max(4);
    let mut q_edges = Vec::with_capacity(nz_clamped + 1);
    q_edges.push(1.0_f64);
    q_edges.push(base.q[1]);
    for k in 2..nz_clamped {
        let frac = ((k - 1) as f64) / ((nz_clamped - 1) as f64);
        let pos = 1.0 + frac * ((base.nz - 2) as f64);
        let idx = (pos.floor() as usize).min(base.nz - 2);
        let rem = pos - (idx as f64);
        let q_interp = (1.0 - rem) * base.q[idx] + rem * base.q[idx + 1];
        q_edges.push(q_interp);
    }
    q_edges.push(0.0_f64);

    let mut dq = vec![0.0_f64; nz_clamped];
    for k in 0..nz_clamped {
        dq[k] = (q_edges[k] - q_edges[k + 1]).max(1.0e-12);
    }
    let sum_dq: f64 = dq.iter().sum();
    for d in &mut dq {
        *d /= sum_dq;
    }

    let mut model =
        StarModel::new(nz_clamped, base.chem_id.clone(), job, controls);
    model.mstar = base.mstar;
    model.mstar_start = base.mstar;
    model.mstar_old = base.mstar;
    model.dt = base.dt;
    model.dt_next = base.dt_next;
    model.dt_old = 0.0;
    model.dq = dq;
    model.lnd = interp_q(&base.dq, &base.lnd, &model.dq).expect("interp_q lnd");
    model.ln_t =
        interp_q(&base.dq, &base.ln_t, &model.dq).expect("interp_q ln_t");
    model.lum = interp_q(&base.dq, &base.lum, &model.dq).expect("interp_q lum");
    let mut r3_running = 0.0_f64;
    for k in (0..nz_clamped).rev() {
        let dm_k = model.mstar * model.dq[k];
        let rho_k = model.lnd[k].exp();
        r3_running +=
            (3.0 * dm_k) / (4.0 * std::f64::consts::PI * rho_k.max(1.0e-30));
        model.ln_r[k] = r3_running.ln() / 3.0;
    }

    let spec = base.species;
    for j in 0..spec {
        let col: Vec<f64> =
            (0..base.nz).map(|k| base.xa[k * spec + j]).collect();
        let col_i = interp_q(&base.dq, &col, &model.dq).expect("interp_q xa");
        for k in 0..nz_clamped {
            model.xa[k * spec + j] = col_i[k].max(0.0);
        }
    }
    for k in 0..nz_clamped {
        let row = model.cell_xa_mut(k);
        let s: f64 = row.iter().sum();
        if s > 0.0 {
            for x in row.iter_mut() {
                *x /= s;
            }
        }
    }
    model
        .finish_init_model()
        .expect("finish_init_model on resampled ZAMS");
    set_hydro_vars(&mut model, false, false)
        .expect("set_hydro_vars on resampled ZAMS");
    model.save_start_values();
    model.save_old_values();
    model.dt_old = 0.0;
    model
}

/// Benchmarks zone/face stellar microphysics (`star_micro_vars`) across
/// Stages 0..3 for `nz` Lagrangian zones.
#[must_use]
pub fn bench_star_micro_vars(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let base = make_wave4_zams_model(nz);
    let mut m0 = base.clone();
    let mut m1 = base.clone();
    let mut m2 = base.clone();
    let mut m3 = base;

    let stage0 = measure_us(n_warmup, n_repeats, || {
        let mut clone_m = m0.clone();
        set_hydro_vars(&mut clone_m, false, false).expect("stage0 micro");
        m0.ln_peos.copy_from_slice(&clone_m.ln_peos);
        m0.gradt.copy_from_slice(&clone_m.gradt);
    });
    let stage1 = measure_us(n_warmup, n_repeats, || {
        set_hydro_vars(&mut m1, false, false).expect("stage1 micro");
    });
    let stage2 = measure_us(n_warmup, n_repeats, || {
        set_hydro_vars(&mut m2, false, true).expect("stage2 micro");
    });
    let stage3 = measure_us(n_warmup, n_repeats, || {
        stage3_pool().install(|| {
            set_hydro_vars(&mut m3, false, true).expect("stage3 micro");
        });
    });

    let pack = |m: &StarModel| -> Vec<f64> {
        (0..m.nz)
            .map(|k| m.ln_peos[k] + m.gradt[k] + m.opacity[k].ln())
            .collect()
    };
    let v0 = pack(&m0);
    let v1 = pack(&m1);
    let v2 = pack(&m2);
    let v3 = pack(&m3);
    let max_rel_err = max_err3(
        max_rel_diff(&v0, &v1),
        max_rel_diff(&v0, &v2),
        max_rel_diff(&v0, &v3),
    );
    let checksum: f64 = v3.iter().sum();

    BenchRecord {
        workload: "star_micro_vars",
        n_zones: nz,
        stage0,
        stage1,
        stage2,
        stage3,
        max_rel_err,
        checksum,
    }
}

/// Benchmarks coupled 1D Henyey residual and block-tridiagonal Jacobian
/// assembly (`star_hydro_eqns`) across Stages 0..3 for `nz` zones.
#[must_use]
pub fn bench_star_hydro_eqns(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let base = make_wave4_zams_model(nz);
    let m0 = base.clone();
    let mut m1 = base.clone();
    let mut m2 = base.clone();
    let mut m3 = base;
    let mut out0 = vec![0.0_f64; m0.nz * 4];
    let mut out1 = vec![0.0_f64; m1.nz * 4];
    let mut out2 = vec![0.0_f64; m2.nz * 4];
    let mut out3 = vec![0.0_f64; m3.nz * 4];

    let pack_sys = |sys: &HenyeySystem, dst: &mut [f64]| {
        for k in 0..sys.nz {
            for i in 0..4 {
                dst[k * 4 + i] =
                    sys.equ[k * 4 + i].abs() + sys.dblk[k * 16 + i * 5].abs();
            }
        }
    };

    let stage0 = measure_us(n_warmup, n_repeats, || {
        let mut m_tmp = m0.clone();
        let sys =
            eval_henyey_equations(&mut m_tmp, false).expect("stage0 eval");
        pack_sys(&sys, &mut out0);
    });
    let stage1 = measure_us(n_warmup, n_repeats, || {
        let sys = eval_henyey_equations(&mut m1, false).expect("stage1 eval");
        pack_sys(&sys, &mut out1);
    });
    let stage2 = measure_us(n_warmup, n_repeats, || {
        let sys = eval_henyey_equations(&mut m2, false).expect("stage2 eval");
        pack_sys(&sys, &mut out2);
    });
    let stage3 = measure_us(n_warmup, n_repeats, || {
        stage3_pool().install(|| {
            let sys =
                eval_henyey_equations(&mut m3, false).expect("stage3 eval");
            pack_sys(&sys, &mut out3);
        });
    });

    let max_rel_err = max_err3(
        max_rel_diff(&out0, &out1),
        max_rel_diff(&out0, &out2),
        max_rel_diff(&out0, &out3),
    );
    let checksum: f64 = out3.iter().sum();

    BenchRecord {
        workload: "star_hydro_eqns",
        n_zones: nz,
        stage0,
        stage1,
        stage2,
        stage3,
        max_rel_err,
        checksum,
    }
}

/// Benchmarks a Henyey linearize-and-solve step (`star_newton_step`) comparing
/// sequential `solve_block_tridiagonal` against 4-thread Hirshman `bcyclic`
/// across `nz` zones.
#[must_use]
pub fn bench_star_newton_step(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let mut base = make_wave4_zams_model(nz);
    base.dt = 1.0e2_f64;
    let nvar = 4_usize;
    let n_z = base.nz;
    let mut m0 = base.clone();
    let mut m1 = base.clone();
    let mut m2 = base.clone();
    let mut m3 = base;

    let mut dx0 = vec![0.0_f64; n_z * nvar];
    let mut dx1 = vec![0.0_f64; n_z * nvar];
    let mut dx2 = vec![0.0_f64; n_z * nvar];
    let mut dx3 = vec![0.0_f64; n_z * nvar];
    let mut rhs = vec![0.0_f64; n_z * nvar];

    let stage0 = measure_us(n_warmup, n_repeats, || {
        let sys = eval_henyey_equations(&mut m0, false).expect("s0 eval");
        let rhs0: Vec<f64> = sys.equ.iter().map(|&r| -r).collect();
        dx0 = solve_block_tridiagonal(
            nvar, n_z, &sys.lblk, &sys.dblk, &sys.ublk, &rhs0,
        )
        .expect("s0 thomas");
    });
    let stage1 = measure_us(n_warmup, n_repeats, || {
        let sys = eval_henyey_equations(&mut m1, false).expect("s1 eval");
        for (dst, &r) in rhs.iter_mut().zip(&sys.equ) {
            *dst = -r;
        }
        dx1 = solve_block_tridiagonal(
            nvar, n_z, &sys.lblk, &sys.dblk, &sys.ublk, &rhs,
        )
        .expect("s1 thomas");
    });
    let stage2 = measure_us(n_warmup, n_repeats, || {
        let sys = eval_henyey_equations(&mut m2, false).expect("s2 eval");
        for (dst, &r) in rhs.iter_mut().zip(&sys.equ) {
            *dst = -r;
        }
        dx2 = solve_block_tridiag_bcyclic(
            nvar, n_z, &sys.lblk, &sys.dblk, &sys.ublk, &rhs, false,
        )
        .expect("s2 bcyclic");
    });
    let stage3 = measure_us(n_warmup, n_repeats, || {
        stage3_pool().install(|| {
            let sys = eval_henyey_equations(&mut m3, false).expect("s3 eval");
            for (dst, &r) in rhs.iter_mut().zip(&sys.equ) {
                *dst = -r;
            }
            dx3 = solve_block_tridiag_bcyclic(
                nvar, n_z, &sys.lblk, &sys.dblk, &sys.ublk, &rhs, true,
            )
            .expect("s3 bcyclic");
        });
    });

    let max_rel_err = max_err3(
        max_rel_diff(&dx0, &dx1),
        max_rel_diff(&dx0, &dx2),
        max_rel_diff(&dx0, &dx3),
    );
    let checksum: f64 = dx3.iter().map(|x| x.abs()).sum();

    BenchRecord {
        workload: "star_newton_step",
        n_zones: nz,
        stage0,
        stage1,
        stage2,
        stage3,
        max_rel_err,
        checksum,
    }
}

/// Benchmarks adaptive mesh spacing evaluation and remesh planning
/// (`star_remesh`) across Stages 0..3 for `nz` zones.
#[must_use]
pub fn bench_star_remesh(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let model = make_wave4_zams_model(nz);
    let cfg = MeshFunctionConfig::default();
    let mut g0 = Vec::new();
    let mut g1 = Vec::new();
    let mut g2 = Vec::new();
    let mut g3 = Vec::new();

    let stage0 = measure_us(n_warmup, n_repeats, || {
        let mut m_clone = model.clone();
        let _ = do_adjust_mesh(&mut m_clone, &cfg).expect("stage0 remesh");
        g0 = eval_mesh_gval(&model, &cfg);
    });
    let stage1 = measure_us(n_warmup, n_repeats, || {
        g1 = eval_mesh_gval(&model, &cfg);
        let _ = plan_mesh(&model, &cfg);
    });
    let stage2 = measure_us(n_warmup, n_repeats, || {
        g2 = eval_mesh_gval(&model, &cfg);
    });
    let stage3 = measure_us(n_warmup, n_repeats, || {
        stage3_pool().install(|| {
            g3 = eval_mesh_gval(&model, &cfg);
        });
    });

    let max_rel_err = max_err3(
        max_rel_diff(&g0, &g1),
        max_rel_diff(&g0, &g2),
        max_rel_diff(&g0, &g3),
    );
    let checksum: f64 = g3.iter().sum();

    BenchRecord {
        workload: "star_remesh",
        n_zones: nz,
        stage0,
        stage1,
        stage2,
        stage3,
        max_rel_err,
        checksum,
    }
}

/// Benchmarks a full structure + burn + mix evolutionary step
/// (`star_evolve_step`) across Stages 0..3 for `nz` zones.
#[must_use]
pub fn bench_star_evolve_step(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let base = make_wave4_zams_model(nz);
    let dt = base.dt;
    let mut out0 = vec![0.0_f64; base.nz];
    let mut out1 = vec![0.0_f64; base.nz];
    let mut out2 = vec![0.0_f64; base.nz];
    let mut out3 = vec![0.0_f64; base.nz];

    let pack_into = |m: &StarModel, dst: &mut [f64]| {
        for k in 0..m.nz {
            dst[k] = m.ln_t[k] + m.lnd[k] + m.xa[k * m.species];
        }
    };

    let stage0 = measure_us(n_warmup, n_repeats, || {
        let mut m = base.clone();
        do_struct_burn_mix(&mut m, dt, false, false, &[])
            .expect("stage0 struct_burn_mix");
        pack_into(&m, &mut out0);
    });
    let stage1 = measure_us(n_warmup, n_repeats, || {
        let mut m = base.clone();
        do_struct_burn_mix(&mut m, dt, false, true, &[])
            .expect("stage1 struct_burn_mix");
        pack_into(&m, &mut out1);
    });
    let stage2 = measure_us(n_warmup, n_repeats, || {
        let mut m = base.clone();
        do_struct_burn_mix(&mut m, dt, false, true, &[])
            .expect("stage2 struct_burn_mix");
        pack_into(&m, &mut out2);
    });
    let stage3 = measure_us(n_warmup, n_repeats, || {
        stage3_pool().install(|| {
            let mut m = base.clone();
            do_struct_burn_mix(&mut m, dt, false, true, &[])
                .expect("stage3 struct_burn_mix");
            pack_into(&m, &mut out3);
        });
    });

    let max_rel_err = max_err3(
        max_rel_diff(&out0, &out1),
        max_rel_diff(&out0, &out2),
        max_rel_diff(&out0, &out3),
    );
    let checksum: f64 = out3.iter().sum();

    BenchRecord {
        workload: "star_evolve_step",
        n_zones: nz,
        stage0,
        stage1,
        stage2,
        stage3,
        max_rel_err,
        checksum,
    }
}

/// Runs all five Wave 4 benchmarks across `zone_counts`.
#[must_use]
pub fn run_all_wave4_benchmarks(
    zone_counts: &[usize],
    n_warmup: usize,
    n_repeats: usize,
) -> Vec<BenchRecord> {
    let mut records = Vec::with_capacity(zone_counts.len() * 5);
    for &nz in zone_counts {
        records.push(bench_star_micro_vars(nz, n_warmup, n_repeats));
        records.push(bench_star_hydro_eqns(nz, n_warmup, n_repeats));
        records.push(bench_star_newton_step(nz, n_warmup, n_repeats));
        records.push(bench_star_remesh(nz, n_warmup, n_repeats));
        records.push(bench_star_evolve_step(nz, n_warmup, n_repeats));
    }
    records
}
