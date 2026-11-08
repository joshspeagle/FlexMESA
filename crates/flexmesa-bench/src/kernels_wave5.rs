// SPDX-License-Identifier: LGPL-3.0-or-later
//! Wave 5 four-stage advanced stellar physics, interacting binary, and
//! asteroseismology benchmarks (`w5_diff_rot`, `w5_riemann_rsp2`,
//! `w5_binary_rlof`, and `w5_astero_seism`).
//!
//! Evaluates the four Wave 5 workloads across four progressive optimization
//! stages in Safe Rust (`flexmesa-star`, `flexmesa-binary`, `flexmesa-astero`),
//! matching `benchmarks/fortran_mesa/bench_kernels_wave5.f90`:
//! 1. `bench_w5_diff_rot`: Roche shellular rotation distortion (`f_P, f_T`),
//!    Caplan, Bauer & Freeman (2022) strong-coupling OCP self-diffusion
//!    $D^*(\Gamma, \kappa)$, five hydrodynamic rotational mixing instabilities
//!    (`DSI`, `SH`, `SSI`, `ES`, `GSF`), and the Spruit-Tayler (2002) magnetic
//!    dynamo (`D_ST`, `nu_ST`, `B_phi`, `B_r`).
//! 2. `bench_w5_riemann_rsp2`: Well-balanced 1D spherical Lagrangian HLLC
//!    Riemann solver, momentum residual assembly, Kuhfuss (1986) / Smolec &
//!    Moskalik (2008) `RSP2` turbulent convection equations, and Dziembowski
//!    dimensionless pulsation variables $(V, V_g, A^*, U, c_1)$.
//! 3. `bench_w5_binary_rlof`: Eggleton (1983) Roche-lobe radius, Keplerian
//!    orbital mechanics, Ritter (1988) optically thin RLOF mass transfer,
//!    Bardeen (1972) / King & Kolb (1999) / Podsiadlowski (2003) Kerr black
//!    hole ISCO spin and Eddington accretion, Peters (1964) + Rappaport,
//!    Verbunt & Joss (1983) orbital $\dot{J}$, and Hut (1981) tidal
//!    eccentricity evolution.
//! 4. `bench_w5_astero_seism`: Cumulative acoustic radius/depth integrals,
//!    asymptotic and solar-scaled $\Delta\nu, \nu_{\max}, \nu_{\mathrm{ac}},
//!    \Delta\Pi_1$, Ball & Gizon (2014) 1-term cubic and 2-term combined
//!    surface-effect frequency corrections, and Roxburgh & Vorontsov (2003)
//!    5-point separation ratios $(r_{01}, r_{10})$.

use flexmesa_astero::{
    eval_acoustic_profile, eval_combined_corr, eval_cubic_corr,
    eval_delta_nu_asymptotic, eval_delta_nu_scaling, eval_delta_pg_traditional,
    eval_frequency_ratios_r01_r10, eval_nu_ac, eval_nu_max,
};
use flexmesa_binary::{
    bh_initial_equatorial_mass, eval_bh_spin, eval_edot_tidal_hut, eval_j_orb,
    eval_jdot_gr, eval_jdot_mb, eval_mdot_edd, eval_ritter_mdot_thin,
    eval_rl_relative_gap, eval_rlobe, separation_from_period,
};
use flexmesa_const::{LN10, LSUN, MSUN, PI, PI4, RSUN, SECYER, STANDARD_CGRAV};
use flexmesa_star::{
    eval_caplan_ocp_dstar, eval_hllc_face, eval_hllc_surface_face,
    eval_omega_crit, eval_riemann_momentum_residual, eval_rotation_fp_ft,
    eval_rotational_mixing_zone, eval_rsp2_zone, solve_stationary_rsp2_w,
    Rsp2Config,
};

use crate::kernels_micro::bench_zone_fn;
use crate::{max_err3, max_rel_diff, measure_us, BenchRecord};

/// Evaluates zone `k` (`0..nz`) of the `w5_diff_rot` benchmark workload,
/// matching `eval_diff_rot_zone` in `bench_kernels_wave5.f90`.
#[must_use]
pub fn eval_diff_rot_zone(k: usize, nz: usize) -> f64 {
    let s = ((k as f64) + 0.5) / (nz as f64);
    let r = RSUN * (0.08 + 0.88 * s);
    let m_grav = MSUN * (0.05 + 0.95 * s * s);
    let cgrav = STANDARD_CGRAV;
    let lum = LSUN * (0.04 + 0.96 * s);
    let opacity = 0.34 + 1.40 * (1.0 - s);
    let rho = (LN10 * (2.0 * (1.0 - s) - 0.8 * s)).exp();
    let t = (LN10 * (7.15 - 1.35 * s)).exp();
    let cp = 2.4e8 * (1.0 + 0.25 * s);
    let hp = 0.085 * r;
    let grada = 0.398;
    let gradt = 0.26 + 0.06 * (5.0 * s).sin();
    let brunt_n2_t = 1.6e-6 * (0.4 + s);
    let brunt_n2_mu = 4.5e-7 * (1.0 - 0.4 * s);
    let w_div_wc = 0.12 + 0.58 * s;
    let dln_om = -0.40 - 0.30 * s;
    let abar = 1.28 + 0.42 * (1.0 - s);
    let zbar = 1.08 + 0.22 * (1.0 - s);
    let gam_ocp = 4.0 + 14.0 * (1.0 - s);
    let kap_ocp = 0.20 + 0.55 * s;

    let om_crit = eval_omega_crit(cgrav, m_grav, r, lum, opacity, 0.99);
    let om = w_div_wc * om_crit;
    let (fp, ft, i_rot_r2) = eval_rotation_fp_ft(w_div_wc, 0.75, 0.95);
    let d_star = eval_caplan_ocp_dstar(gam_ocp, kap_ocp);
    let grav = fp * (cgrav * m_grav / (r * r));
    let rot = eval_rotational_mixing_zone(
        r,
        hp,
        grav,
        rho,
        t,
        cp,
        opacity,
        1.0,
        1.0,
        grada,
        gradt,
        brunt_n2_t,
        brunt_n2_mu,
        om,
        dln_om,
        abar,
        zbar,
        0.0,
    );

    fp + ft
        + i_rot_r2
        + d_star
        + 1.0e-12
            * (rot.d_dsi
                + rot.d_sh
                + rot.d_ssi
                + rot.d_es
                + rot.d_gsf
                + rot.d_st
                + rot.nu_st)
        + 1.0e-5 * (rot.b_phi + rot.b_r)
}

/// Per-zone hydrodynamic and thermodynamic state for `RiemannRsp2Grid`.
#[derive(Clone, Copy)]
struct RiemannZone {
    r_out: f64,
    r_in: f64,
    m_grav: f64,
    rho: f64,
    t: f64,
    p: f64,
    cs: f64,
    u: f64,
    u_old: f64,
    dm: f64,
    cp: f64,
    grad_t: f64,
    kappa: f64,
    hp: f64,
    dlnd_dt: f64,
}

/// Precomputed `nz`-zone hydrodynamic profile for `w5_riemann_rsp2`.
pub struct RiemannRsp2Grid {
    zones: Vec<RiemannZone>,
    p_bc: f64,
}

impl RiemannRsp2Grid {
    /// Constructs a deterministic `nz`-zone hydrodynamic profile for
    /// `w5_riemann_rsp2`.
    #[must_use]
    pub fn new(nz: usize) -> Self {
        let mut zones = Vec::with_capacity(nz);
        let nz_f = nz as f64;
        for k in 0..nz {
            let s = ((k as f64) + 0.5) / nz_f;
            let r_out = RSUN * (1.0 - 0.85 * ((k as f64) / nz_f));
            let r_in = RSUN * (1.0 - 0.85 * (((k + 1) as f64) / nz_f));
            let m_grav = MSUN * (1.0 - 0.92 * ((k as f64) / nz_f)).powi(2);
            let rho = (LN10 * (-4.5 + 5.5 * s)).exp();
            let t = (LN10 * (3.85 + 3.15 * s)).exp();
            let p = 1.25e8 * rho * t;
            let cs = ((5.0 / 3.0) * p / rho).sqrt();
            let u = 1.5e5 * (1.0 - s) * (4.0 * s).sin();
            zones.push(RiemannZone {
                r_out,
                r_in,
                m_grav,
                rho,
                t,
                p,
                cs,
                u,
                u_old: 0.98 * u,
                dm: (MSUN * 0.85 / nz_f) * (0.4 + 1.2 * s * (1.0 - s)),
                cp: 2.1e8 * (1.0 + 0.15 * s),
                grad_t: 0.445 - 0.10 * s,
                kappa: 0.40 + 2.5 * (1.0 - s).powi(2),
                hp: 0.06 * r_out,
                dlnd_dt: -1.0e-6 * (3.0 * s).cos(),
            });
        }
        let p_bc = 0.65 * zones[0].p;
        Self { zones, p_bc }
    }

    #[inline]
    fn eval_face(&self, k: usize) -> (f64, f64) {
        let zk = &self.zones[k];
        let st = if k == 0 {
            eval_hllc_surface_face(
                zk.p,
                zk.rho,
                zk.cs,
                zk.u,
                zk.dm,
                zk.r_out,
                zk.m_grav,
                STANDARD_CGRAV,
                self.p_bc,
            )
        } else {
            let z_prev = &self.zones[k - 1];
            eval_hllc_face(
                zk.p,
                zk.rho,
                zk.cs,
                zk.u,
                zk.dm,
                z_prev.p,
                z_prev.rho,
                z_prev.cs,
                z_prev.u,
                z_prev.dm,
                zk.r_out,
                zk.m_grav,
                STANDARD_CGRAV,
            )
        };
        (st.u_face, st.p_face)
    }

    /// Evaluates zone `k` (`0..nz`) of `w5_riemann_rsp2`.
    #[inline]
    #[must_use]
    pub fn eval_zone(&self, k: usize) -> f64 {
        let dt = 100.0_f64;
        let grad_ad = 0.395_f64;
        let delta = 1.0_f64;
        let cfg = Rsp2Config::default();
        let nz = self.zones.len();
        let s = ((k as f64) + 0.5) / (nz as f64);
        let zk = &self.zones[k];

        let (u_f, p_f_out) = self.eval_face(k);
        let g_out =
            STANDARD_CGRAV * zk.m_grav / (zk.r_out * zk.r_out).max(1.0e-30);
        let (p_f_in, g_in) = if k + 1 < nz {
            (
                self.eval_face(k + 1).1,
                STANDARD_CGRAV * self.zones[k + 1].m_grav
                    / (zk.r_in * zk.r_in).max(1.0e-30),
            )
        } else {
            let a_out = (PI4 * zk.r_out * zk.r_out).max(1.0e-30);
            (p_f_out + 0.5 * zk.dm * g_out / a_out, 0.0)
        };
        let g_cell = 0.5 * (g_out + g_in);
        let res_mom = eval_riemann_momentum_residual(
            zk.u, zk.u_old, dt, zk.dm, zk.r_out, zk.r_in, p_f_out, p_f_in,
            zk.p, g_cell,
        );

        let w_st = solve_stationary_rsp2_w(
            zk.t, zk.rho, zk.p, zk.cp, grad_ad, delta, zk.kappa, zk.hp,
            zk.grad_t, &cfg,
        );
        let w = w_st + 2.0e4 * (1.0 - s);
        let rsp = eval_rsp2_zone(
            w,
            zk.t,
            zk.rho,
            zk.p,
            zk.cp,
            grad_ad,
            delta,
            zk.kappa,
            zk.hp,
            zk.grad_t,
            zk.r_out,
            zk.r_in,
            zk.u,
            0.95 * zk.u,
            zk.dm,
            zk.dlnd_dt,
            &cfg,
        );

        let v_dz = g_out * zk.r_out * zk.rho / zk.p;
        let v_g = v_dz / (5.0 / 3.0);
        let a_st = zk.r_out * (1.0e-6 * s) / g_out;
        let u_dz = PI4 * zk.r_out.powi(3) * zk.rho / zk.m_grav;
        let c_1 = (zk.r_out / RSUN).powi(3) / (zk.m_grav / MSUN);

        (u_f / 1.0e6)
            + (p_f_out / zk.p)
            + (res_mom / 1.0e4)
            + (w / 1.0e5)
            + (rsp.p_turb / zk.p)
            + (rsp.de_t_dt / 1.0e8)
            + 1.0e-2 * (v_dz + v_g + a_st + u_dz + c_1)
    }
}

/// Evaluates configuration `k` (`0..nz`) of the `w5_binary_rlof` benchmark
/// workload, matching `eval_binary_rlof_zone` in `bench_kernels_wave5.f90`.
#[must_use]
pub fn eval_binary_rlof_zone(k: usize, nz: usize) -> f64 {
    let s = ((k as f64) + 0.5) / (nz as f64);
    let md = MSUN * (1.20 + 6.80 * s);
    let ma = MSUN * (0.80 + 4.20 * (1.0 - 0.5 * s));
    let per = SECYER * (1.5e-3 + 2.5e-2 * s);
    let ecc = 0.05 + 0.35 * s * (1.0 - s);
    let rho_s = 1.0e-7 * (1.0 + 2.0 * s);
    let p_s = 1.2e5 * (1.0 + 1.5 * s);
    let t_s = 5500.0 + 6500.0 * s;
    let mu_s = 0.61 + 0.04 * (1.0 - s);
    let m_bh_i = MSUN * (5.0 + 5.0 * s);
    let a_sp_i = 0.15 + 0.70 * s;
    let m_bh_c = m_bh_i * (1.05 + 0.25 * s);
    let kap_s = 0.34;
    let k_div_t = 1.2e-9 * (1.0 + s);
    let om_orb = 2.0 * PI / per;
    let om_star = 1.25 * om_orb;

    let sep = separation_from_period(md, ma, per);
    let rl_d = eval_rlobe(md, ma, sep);
    let r_d = rl_d * (0.985 + 0.030 * s);
    let rl_gap = eval_rl_relative_gap(r_d, rl_d, ecc);
    let j_orb = eval_j_orb(md, ma, sep, ecc);
    let mdot_thin =
        eval_ritter_mdot_thin(md, ma, r_d, rl_d, rho_s, p_s, t_s, mu_s, false);

    let m_eq = bh_initial_equatorial_mass(m_bh_i, a_sp_i);
    let a_cur = eval_bh_spin(m_bh_c, m_eq);
    let (mdot_edd, eta) = eval_mdot_edd(m_bh_c, m_eq, kap_s);
    let jdot_gr = eval_jdot_gr(md, ma, sep, j_orb);
    let jdot_mb = eval_jdot_mb(md, r_d, rl_d, per, 3.0, 0.1, false);
    let edot_tid =
        eval_edot_tidal_hut(k_div_t, md, ma, r_d, sep, ecc, om_star, per, 1.0);

    (rl_d / sep)
        + rl_gap
        + (j_orb / 1.0e53)
        + (-mdot_thin / MSUN * SECYER)
        + a_cur
        + eta
        + (mdot_edd / MSUN * SECYER)
        + (-jdot_gr / 1.0e36)
        + (-jdot_mb / 1.0e36)
        + (-edot_tid * SECYER * 1.0e4)
}

/// Evaluates the `w5_astero_seism` benchmark workload for `nz` zones, matching
/// `eval_astero_seism_workload` in `bench_kernels_wave5.f90`.
#[must_use]
pub fn eval_astero_seism_workload(nz: usize) -> f64 {
    const N_MODES: usize = 12;
    let r_center = RSUN * 0.05;
    let r_phot = RSUN * 0.995;
    let m_star = 1.05 * MSUN;
    let teff = 5820.0;
    let l_phot = 1.12 * LSUN;
    let g_surf = STANDARD_CGRAV * m_star / (r_phot * r_phot);
    let gam1_s = 1.64;
    let rho_s = 2.2e-7;
    let p_s = 1.1e5;

    let mut r = Vec::with_capacity(nz);
    let mut r_mid = Vec::with_capacity(nz);
    let mut csound = Vec::with_capacity(nz);
    let mut brunt_n2 = Vec::with_capacity(nz);
    let nz_f = nz as f64;
    for k in 0..nz {
        let s = ((k as f64) + 0.5) / nz_f;
        r.push(RSUN * (1.0 - 0.95 * ((k as f64) / nz_f)));
        r_mid.push(RSUN * (1.0 - 0.95 * s));
        csound.push(1.0e6 * (0.8 + 4.5 * s));
        brunt_n2.push(if s > 0.35 {
            2.5e-6 * (s - 0.35)
        } else {
            -1.0e-7
        });
    }

    let (tau_r, tau_d, tau_0) =
        eval_acoustic_profile(&r, &csound, r_center, r_phot);
    let dnu_asymp = eval_delta_nu_asymptotic(tau_0);
    let dnu_scal = eval_delta_nu_scaling(m_star, teff, l_phot);
    let nu_max_v = eval_nu_max(m_star, r_phot, teff);
    let nu_ac_v = eval_nu_ac(g_surf, gam1_s, rho_s, p_s);
    let delta_pg = eval_delta_pg_traditional(&r, &r_mid, &brunt_n2);

    let target_b1 = -5.125_972_793_567_201e-5_f64;
    let target_b3 = -1.446_307_489_879_336_1e-18_f64;
    let mut l0_freq = [0.0_f64; N_MODES];
    let mut l0_ine = [0.0_f64; N_MODES];
    let l0_sig = [0.15_f64; N_MODES];
    let mut l0_obs = [0.0_f64; N_MODES];
    let mut l1_mod = [0.0_f64; N_MODES];
    let l1_sig = [0.18_f64; N_MODES];
    for m in 0..N_MODES {
        let n_ord = (14 + m) as f64;
        let f0 = 135.0 * (n_ord + 1.45);
        let q0 = 1.0e-7 * ((3100.0 - f0) / 1000.0).exp();
        l0_freq[m] = f0;
        l0_ine[m] = q0;
        l0_obs[m] = f0 + (target_b1 / f0 + target_b3 * f0.powi(3)) / q0;
        l1_mod[m] = 135.0 * (n_ord + 1.93) - 1.8 * (0.3 * n_ord).sin();
    }

    let (corr_cub, _) =
        eval_cubic_corr(&l0_obs, &l0_sig, &l0_freq, &l0_ine, 1.0);
    let (corr_comb, _, _) =
        eval_combined_corr(&l0_obs, &l0_sig, &l0_freq, &l0_ine, 1.0);
    let (r01, sig_r01, r10, sig_r10) =
        eval_frequency_ratios_r01_r10(&l0_freq, &l0_sig, &l1_mod, &l1_sig);

    let sum_tau: f64 = tau_r.iter().zip(&tau_d).map(|(a, b)| a + b).sum();
    let sum_corr: f64 =
        corr_cub.iter().zip(&corr_comb).map(|(a, b)| a + b).sum();
    let sum_rat: f64 = (0..r01.len())
        .map(|i| r01[i] + sig_r01[i] + r10[i] + sig_r10[i])
        .sum();

    (sum_tau / nz_f) / 1.0e3
        + (tau_0 / 1.0e3)
        + (dnu_asymp + dnu_scal + nu_max_v + nu_ac_v) / 1.0e3
        + (delta_pg / 1.0e3)
        + (sum_corr / (N_MODES as f64)) / 1.0e3
        + sum_rat / (r01.len() as f64)
}

/// Shared 1D zone benchmark wrapper normalizing `rec.checksum` by `nz`.
fn bench_w5_zone_1d<F>(
    workload: &'static str,
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
    eval_zone: F,
) -> BenchRecord
where
    F: Fn(usize) -> f64 + Sync + Copy,
{
    let eval_k = move |k: usize| [eval_zone(k)];
    let mut rec = bench_zone_fn::<1, _, _>(
        workload, nz, n_warmup, n_repeats, eval_k, eval_k,
    );
    rec.checksum /= nz as f64;
    rec
}

/// Benchmarks Wave 5.1 diffusion, rotation, and Spruit-Tayler magnetic dynamo
/// kernels (`w5_diff_rot`) across Stages 0..3 for `nz` zones.
#[must_use]
pub fn bench_w5_diff_rot(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    bench_w5_zone_1d("w5_diff_rot", nz, n_warmup, n_repeats, |k| {
        eval_diff_rot_zone(k, nz)
    })
}

/// Benchmarks Wave 5.2 HLLC Riemann hydrodynamics, `RSP2` turbulent convection,
/// and Dziembowski pulsation variables (`w5_riemann_rsp2`) across Stages 0..3
/// for `nz` zones.
#[must_use]
pub fn bench_w5_riemann_rsp2(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let grid = RiemannRsp2Grid::new(nz);
    bench_w5_zone_1d("w5_riemann_rsp2", nz, n_warmup, n_repeats, |k| {
        grid.eval_zone(k)
    })
}

/// Benchmarks Wave 5.3 interacting binary orbital mechanics, RLOF mass
/// transfer, Kerr BH accretion, orbital $\dot{J}$, and Hut tides
/// (`w5_binary_rlof`) across Stages 0..3 for `nz` binary configurations.
#[must_use]
pub fn bench_w5_binary_rlof(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    bench_w5_zone_1d("w5_binary_rlof", nz, n_warmup, n_repeats, |k| {
        eval_binary_rlof_zone(k, nz)
    })
}

/// Benchmarks Wave 5.4 asteroseismology acoustic transit integrals, scaling
/// relations, Ball & Gizon surface corrections, and Roxburgh & Vorontsov
/// separation ratios (`w5_astero_seism`) across Stages 0..3 for `nz` zones.
#[must_use]
pub fn bench_w5_astero_seism(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let mut c0 = [0.0_f64; 1];
    let mut c1 = [0.0_f64; 1];
    let mut c2 = [0.0_f64; 1];
    let mut c3 = [0.0_f64; 1];

    let stage0 = measure_us(n_warmup, n_repeats, || {
        c0[0] = eval_astero_seism_workload(nz);
    });
    let stage1 = measure_us(n_warmup, n_repeats, || {
        c1[0] = eval_astero_seism_workload(nz);
    });
    let stage2 = measure_us(n_warmup, n_repeats, || {
        c2[0] = eval_astero_seism_workload(nz);
    });
    let stage3 = measure_us(n_warmup, n_repeats, || {
        c3[0] = eval_astero_seism_workload(nz);
    });

    let max_rel_err = max_err3(
        max_rel_diff(&c0, &c1),
        max_rel_diff(&c0, &c2),
        max_rel_diff(&c0, &c3),
    );
    BenchRecord {
        workload: "w5_astero_seism",
        n_zones: nz,
        stage0,
        stage1,
        stage2,
        stage3,
        max_rel_err,
        checksum: c3[0],
    }
}

/// Runs all four Wave 5 benchmarks across `zone_counts`.
#[must_use]
pub fn run_all_wave5_benchmarks(
    zone_counts: &[usize],
    n_warmup: usize,
    n_repeats: usize,
) -> Vec<BenchRecord> {
    let mut records = Vec::with_capacity(zone_counts.len() * 4);
    for &nz in zone_counts {
        records.push(bench_w5_diff_rot(nz, n_warmup, n_repeats));
        records.push(bench_w5_riemann_rsp2(nz, n_warmup, n_repeats));
        records.push(bench_w5_binary_rlof(nz, n_warmup, n_repeats));
        records.push(bench_w5_astero_seism(nz, n_warmup, n_repeats));
    }
    records
}
