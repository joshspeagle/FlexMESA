// SPDX-License-Identifier: LGPL-3.0-or-later
//! Wave 3b four-stage stellar microphysics benchmarks (`eos_4d`, `kap_fixed`,
//! `rates_screen`, `net_eval`, and `atm_table`).
//!
//! Evaluates the five Wave 3b stellar microphysics workloads across four
//! progressive optimization stages in Safe Rust:
//! 1. `bench_eos_4d`: 4D tabulated `OPAL/SCVH` equation of state
//!    (`eval_opal_scvh_eos` / `get1_eosdt_results`) for a 7-species solar
//!    composition at $\log_{10}\rho \in [-2.0, -1.99]$ and
//!    $\log_{10}T \in [5.0, 5.01]$.
//! 2. `bench_kap_fixed`: Type 1 fixed-metal (`gs98` + `lowT_fa05_gs98`)
//!    radiative, Compton, and electron-conduction opacity (`kap_get` /
//!    `get_kap_results`) at $\log_{10}\rho \in [-6.0, -5.96]$ and
//!    $\log_{10}T \in [6.0, 6.04]$.
//! 3. `bench_rates_screen`: REACLIB ${}^{12}\mathrm{C}(\alpha,\gamma)$
//!    (`r_c12_ag_o16`) forward rate plus `extended` `screen5` plasma
//!    electrostatic screening (`eval_reaclib_handle` + `screen_pair`) in a
//!    50/50 ${}^{12}\mathrm{C}/{}^{16}\mathrm{O}$ plasma at
//!    $\log_{10}\rho \in [2.0, 5.0]$ and $\log_{10}T \in [8.1, 8.5]$.
//! 4. `bench_net_eval`: 8-isotope `basic.net` nuclear reaction network RHS,
//!    analytical Jacobian, and neutrino/nuclear energy generation (`eval_net`
//!    / `eval_net_with_work`) at $\log_{10}\rho \in [2.0, 2.1]$ and
//!    $\log_{10}T \in [7.9, 7.95]$.
//! 5. `bench_atm_table`: `ATM_TABLE_PHOTOSPHERE` 2D bicubic/Steffen atmosphere
//!    boundary condition (`eval_table`) across $R \in [0.8, 1.4]\,R_\odot$ and
//!    $T_{\mathrm{eff}} \in [5200, 6400]\,\mathrm{K}$ at $M = M_\odot$ and
//!    $Z = 0.0179$.

use flexmesa_atm::{eval_table, ATM_TABLE_PHOTOSPHERE};
use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_const::{CLIGHT, CRAD, MSUN, PI, RSUN, STANDARD_CGRAV};
use flexmesa_eos::{
    eval_opal_scvh_eos, get1_eosdt_results, EosDtTableKind, EosDtTableSet,
    EosEvalOutput, OpalScvhConfig, I_FRAC_OPAL_SCVH, I_LN_PGAS,
    NUM_EOS_BASIC_RESULTS,
};
use flexmesa_kap::{get_kap_results, kap_get, KapConfig};
use flexmesa_math::{exp10, safe_log10};
use flexmesa_net::{
    eval_net_with_work, net_get, with_thread_local_work, F64x2, NetGeneralInfo,
    NUM_RVS,
};
use flexmesa_rates::{
    eval_reaclib_handle, eval_reaclib_terms, screen_extended, screen_pair,
    RatesCatalog, ScreeningMode, DEFAULT_REACLIB_MIN_T9,
};

use crate::kernels_micro::bench_zone_fn;
use crate::BenchRecord;

/// Canonical 7-species solar composition (`h1`, `he4`, `c12`, `n14`, `o16`,
/// `ne20`, `mg24`) used for `eos_4d` and `kap_fixed`.
const SOLAR7_NAMES: [&str; 7] =
    ["h1", "he4", "c12", "n14", "o16", "ne20", "mg24"];

/// Mass fractions for [`SOLAR7_NAMES`] ($X = 0.70$, $Y = 0.28$, $Z = 0.02$).
const SOLAR7_XA: [f64; 7] = [0.70, 0.28, 0.004, 0.001, 0.009, 0.003, 0.003];

/// Packs a 26-channel [`EosEvalOutput`] and its $(\ln\rho, \ln T)$ partials
/// (`d_dlnd`, `d_dln_t`) into a `[f64; 78]` array.
#[inline]
fn pack_eos_res(out: &EosEvalOutput) -> [f64; 3 * NUM_EOS_BASIC_RESULTS] {
    let mut row = [0.0_f64; 3 * NUM_EOS_BASIC_RESULTS];
    let n = NUM_EOS_BASIC_RESULTS;
    row[0..n].copy_from_slice(&out.res);
    row[n..2 * n].copy_from_slice(&out.d_dlnd);
    row[2 * n..3 * n].copy_from_slice(&out.d_dln_t);
    row
}

/// Benchmarks 4D `OPAL/SCVH` tabulated EOS (`eos_4d`) across Stages 0..3 for
/// `nz` radial zones.
///
/// - **Stage 0 & 1**: Per-zone [`eval_opal_scvh_eos`] with nuclide catalog
///   lookup and per-species `d_dxa` allocation.
/// - **Stage 2 & 3**: Hoisted `(X, Z)` mass fractions with zero-allocation
///   [`get1_eosdt_results`] and 4-thread `rayon` parallelism.
#[must_use]
pub fn bench_eos_4d(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let cat = NuclideCatalog::builtin_ref();
    let chem_ids: Vec<usize> = SOLAR7_NAMES
        .iter()
        .map(|&name| cat.get_nuclide_index(name).expect("valid nuclide"))
        .collect();
    let comp = get_composition_info(cat, &chem_ids, &SOLAR7_XA, true)
        .expect("valid solar7 composition");
    let tables = EosDtTableSet::builtin();
    let cfg = OpalScvhConfig::default();

    let denom = (nz.saturating_sub(1)).max(1) as f64;
    let mut rho = Vec::with_capacity(nz);
    let mut t = Vec::with_capacity(nz);
    let mut log_rho = Vec::with_capacity(nz);
    let mut log_t = Vec::with_capacity(nz);
    for k in 0..nz {
        let s = (k as f64) / denom;
        let r_val = exp10(-2.0 + 0.01 * s);
        let t_val = exp10(5.0 + 0.01 * s);
        rho.push(r_val);
        t.push(t_val);
        log_rho.push(r_val.log10());
        log_t.push(t_val.log10());
    }

    let eval_fast = |k: usize| -> EosEvalOutput {
        let (mut out, _, _) = get1_eosdt_results(
            tables,
            EosDtTableKind::OpalScvh,
            comp.xz,
            comp.xh,
            log_rho[k],
            log_t[k],
            cfg.use_linear_interp_for_x,
            cfg.tiny_fuzz,
        )
        .expect("valid get1_eosdt_results");
        out.set_single_component_fraction(I_FRAC_OPAL_SCVH);
        out
    };

    let mut rec = bench_zone_fn::<{ 3 * NUM_EOS_BASIC_RESULTS }, _, _>(
        "eos_4d",
        nz,
        n_warmup,
        n_repeats,
        |k| {
            let out = eval_opal_scvh_eos(
                tables, rho[k], t[k], &chem_ids, &SOLAR7_XA, &cfg,
            )
            .expect("valid eval_opal_scvh_eos");
            pack_eos_res(&out)
        },
        |k| pack_eos_res(&eval_fast(k)),
    );
    rec.checksum = (0..nz).map(|k| eval_fast(k).res[I_LN_PGAS]).sum();
    rec
}

/// Benchmarks Type 1 fixed-metal + conductive + Compton opacity (`kap_fixed`)
/// across Stages 0..3 for `nz` radial zones.
///
/// - **Stage 0 & 1**: Per-zone [`kap_get`] with nuclide catalog composition
///   reduction.
/// - **Stage 2 & 3**: Hoisted composition moments with [`get_kap_results`] and
///   4-thread `rayon` parallelism.
#[must_use]
pub fn bench_kap_fixed(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let cat = NuclideCatalog::builtin_ref();
    let chem_ids: Vec<usize> = SOLAR7_NAMES
        .iter()
        .map(|&name| cat.get_nuclide_index(name).expect("valid nuclide"))
        .collect();
    let comp = get_composition_info(cat, &chem_ids, &SOLAR7_XA, true)
        .expect("valid solar7 composition");
    let cfg = KapConfig {
        zbase: 0.02,
        ..KapConfig::default()
    };
    let lnfree_e = -0.160_442_8;
    let eta = -15.729_47;
    let (d_eta_dlnd, d_eta_dlnt) = (1.0, -1.5);

    let denom = (nz.saturating_sub(1)).max(1) as f64;
    let mut log_rho = Vec::with_capacity(nz);
    let mut log_t = Vec::with_capacity(nz);
    for k in 0..nz {
        let s = (k as f64) / denom;
        log_rho.push(-6.0 + 0.04 * s);
        log_t.push(6.0 + 0.04 * s);
    }

    let eval_fast = |k: usize| -> [f64; 3] {
        let out = get_kap_results(
            &cfg, comp.zbar, comp.xh, comp.xz, 0.004, 0.001, 0.009, 0.003,
            log_rho[k], log_t[k], lnfree_e, 0.0, 0.0, eta, d_eta_dlnd,
            d_eta_dlnt,
        )
        .expect("valid get_kap_results");
        [out.kap, out.dlnkap_dln_rho, out.dlnkap_dln_t]
    };
    let _ = eval_fast(0);

    let mut rec = bench_zone_fn::<3, _, _>(
        "kap_fixed",
        nz,
        n_warmup,
        n_repeats,
        |k| {
            let out = kap_get(
                &cfg, &chem_ids, &SOLAR7_XA, log_rho[k], log_t[k], lnfree_e,
                0.0, 0.0, eta, d_eta_dlnd, d_eta_dlnt,
            )
            .expect("valid kap_get");
            [out.kap, out.dlnkap_dln_rho, out.dlnkap_dln_t]
        },
        eval_fast,
    );
    rec.checksum = (0..nz).map(|k| eval_fast(k)[0]).sum();
    rec
}

/// Benchmarks REACLIB `r_c12_ag_o16` rate + `extended` `screen5` plasma
/// screening (`rates_screen`) across Stages 0..3 for `nz` radial zones.
#[must_use]
pub fn bench_rates_screen(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let (_, terms) = RatesCatalog::builtin()
        .reaclib_terms("r_c12_ag_o16")
        .expect("valid r_c12_ag_o16");
    let zbar = 6.857_142_857_142_857;
    let abar = 13.714_285_714_285_714;
    let z2bar = 47.428_571_428_571_43;

    let denom = (nz.saturating_sub(1)).max(1) as f64;
    let mut rho = Vec::with_capacity(nz);
    let mut t = Vec::with_capacity(nz);
    for k in 0..nz {
        let s = (k as f64) / denom;
        rho.push(exp10(2.0 + 3.0 * s));
        t.push(exp10(8.1 + 0.4 * s));
    }

    let eval_fast = |k: usize| -> [f64; 3] {
        let (fw, _, _, _) = eval_reaclib_terms(
            terms,
            t[k] * 1.0e-9,
            DEFAULT_REACLIB_MIN_T9,
            true,
        );
        let sc = screen_extended(
            t[k], rho[k], zbar, abar, z2bar, 12.0, 6.0, 4.0, 2.0,
        );
        [safe_log10(fw * sc.scor), fw, sc.scor]
    };

    let mut rec = bench_zone_fn::<3, _, _>(
        "rates_screen",
        nz,
        n_warmup,
        n_repeats,
        |k| {
            let (fw, _, _, _) =
                eval_reaclib_handle("r_c12_ag_o16", t[k] * 1.0e-9, true)
                    .expect("valid eval_reaclib_handle");
            let sc = screen_pair(
                ScreeningMode::Extended,
                t[k],
                rho[k],
                zbar,
                abar,
                z2bar,
                12.0,
                6.0,
                4.0,
                2.0,
            );
            [safe_log10(fw * sc.scor), fw, sc.scor]
        },
        eval_fast,
    );
    rec.checksum = (0..nz).map(|k| eval_fast(k)[0]).sum();
    rec
}

/// Benchmarks the 8-isotope `basic.net` nuclear reaction network RHS,
/// analytical Jacobian, and energy generation (`net_eval`) across Stages 0..3
/// for `nz` radial zones.
#[must_use]
pub fn bench_net_eval(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let g = NetGeneralInfo::from_name("basic.net").expect("valid basic.net");
    let cat = NuclideCatalog::builtin_ref();
    let xa = [
        0.70,
        2.98e-5,
        0.28,
        0.0034,
        0.001,
        0.0096,
        0.002,
        1.0 - (0.70 + 2.98e-5 + 0.28 + 0.0034 + 0.001 + 0.0096 + 0.002),
    ];
    let comp = get_composition_info(cat, &g.chem_id, &xa, true)
        .expect("valid basic.net composition");
    let rate_factors = vec![1.0_f64; g.num_reactions];

    let denom = (nz.saturating_sub(1)).max(1) as f64;
    let mut log_rho = Vec::with_capacity(nz);
    let mut rho = Vec::with_capacity(nz);
    let mut log_t = Vec::with_capacity(nz);
    let mut t = Vec::with_capacity(nz);
    for k in 0..nz {
        let s = (k as f64) / denom;
        let (lr, lt) = (2.0 + 0.1 * s, 7.9 + 0.05 * s);
        log_rho.push(lr);
        rho.push(exp10(lr));
        log_t.push(lt);
        t.push(exp10(lt));
    }

    let eval_fast = |k: usize| -> [f64; 4] {
        let (eps, deps_dt, deps_dr, _) = with_thread_local_work(&g, |w| {
            let mut dydt = [[F64x2::ZERO; NUM_RVS]; 32];
            eval_net_with_work(
                &g,
                w,
                &mut dydt[..g.num_isos],
                false,
                &xa,
                t[k],
                log_t[k],
                rho[k],
                log_rho[k],
                comp.abar,
                comp.zbar,
                comp.z2bar,
                comp.ye,
                0.0,
                0.0,
                0.0,
                &rate_factors,
                1.0,
                ScreeningMode::Extended,
                false,
                false,
                false,
            )
        })
        .expect("valid eval_net_with_work");
        [safe_log10(eps), eps, deps_dt, deps_dr]
    };

    let mut rec = bench_zone_fn::<4, _, _>(
        "net_eval",
        nz,
        n_warmup,
        n_repeats,
        |k| {
            let out = net_get(
                &g,
                false,
                &xa,
                t[k],
                log_t[k],
                rho[k],
                log_rho[k],
                comp.abar,
                comp.zbar,
                comp.z2bar,
                comp.ye,
                0.0,
                0.0,
                0.0,
                &rate_factors,
                1.0,
                ScreeningMode::Extended,
            )
            .expect("valid net_get");
            [
                safe_log10(out.eps_nuc),
                out.eps_nuc,
                out.d_eps_nuc_dt,
                out.d_eps_nuc_drho,
            ]
        },
        eval_fast,
    );
    rec.checksum = (0..nz).map(|k| eval_fast(k)[0]).sum();
    rec
}

/// Benchmarks `ATM_TABLE_PHOTOSPHERE` surface boundary conditions (`atm_table`)
/// across Stages 0..3 for `nz` radial zones.
#[must_use]
pub fn bench_atm_table(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let denom = (nz.saturating_sub(1)).max(1) as f64;
    let mut l_vec = Vec::with_capacity(nz);
    let mut r_vec = Vec::with_capacity(nz);
    let mut m_vec = Vec::with_capacity(nz);
    let mut teff_vec = Vec::with_capacity(nz);
    for k in 0..nz {
        let s = (k as f64) / denom;
        let r = RSUN * (0.8 + 0.6 * s);
        let teff = 5200.0 + 1200.0 * s;
        let l = PI * CRAD * CLIGHT * r * r * teff.powi(4);
        l_vec.push(l);
        r_vec.push(r);
        m_vec.push(MSUN);
        teff_vec.push(teff);
    }

    let eval_k = |k: usize| -> [f64; 8] {
        let out = eval_table(
            l_vec[k],
            r_vec[k],
            m_vec[k],
            STANDARD_CGRAV,
            ATM_TABLE_PHOTOSPHERE,
            0.0179,
            false,
            teff_vec[k],
        )
        .expect("valid eval_table");
        [
            out.ln_t,
            out.dln_t_dl,
            out.dln_t_dln_r,
            out.dln_t_dln_m,
            out.ln_p,
            out.dln_p_dl,
            out.dln_p_dln_r,
            out.dln_p_dln_m,
        ]
    };
    let _ = eval_k(0);

    let mut rec = bench_zone_fn::<8, _, _>(
        "atm_table",
        nz,
        n_warmup,
        n_repeats,
        eval_k,
        eval_k,
    );
    rec.checksum = (0..nz).map(|k| eval_k(k)[4]).sum();
    rec
}

/// Runs all five Wave 3b microphysics benchmark workloads across a list of zone
/// counts `zone_counts`.
#[must_use]
pub fn run_all_wave3b_benchmarks(
    zone_counts: &[usize],
    n_warmup: usize,
    n_repeats: usize,
) -> Vec<BenchRecord> {
    let mut records = Vec::with_capacity(zone_counts.len() * 5);
    for &nz in zone_counts {
        records.push(bench_eos_4d(nz, n_warmup, n_repeats));
        records.push(bench_kap_fixed(nz, n_warmup, n_repeats));
        records.push(bench_rates_screen(nz, n_warmup, n_repeats));
        records.push(bench_net_eval(nz, n_warmup, n_repeats));
        records.push(bench_atm_table(nz, n_warmup, n_repeats));
    }
    records
}
