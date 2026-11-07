// SPDX-License-Identifier: LGPL-3.0-or-later
//! Wave 3a four-stage equation-of-state (`eos`) benchmarks (`eos_ideal`,
//! `eos_helm`, `eos_pc`, and `eos_skye`).
//!
//! Evaluates the four Wave 3a stellar equation-of-state engines across four
//! progressive optimization stages in Safe Rust:
//! 1. `bench_eos_ideal`: Neutral ideal-gas plus Planckian radiation EOS
//!    (`eval_ideal_eos` / `eval_ideal_eos_from_weights`) over the 8-isotope
//!    `WAVE2_ISO_NAMES` composition profile at $\log_{10}\rho \in [-6, -2]$ and
//!    $\log_{10}T \in [3.5, 4.5]$.
//! 2. `bench_eos_helm`: Timmes & Swesty (2000) biquintic/bicubic Helmholtz
//!    electron-positron + Hull & Lewis (1956) Coulomb + ideal-ion + radiation
//!    EOS (`eval_helm_eos`) for a 50/50 ${}^{12}\mathrm{C}/{}^{16}\mathrm{O}$
//!    mixture at $\log_{10}\rho \in [1.0, 6.0]$ and
//!    $\log_{10}T \in [8.0, 8.008]$.
//! 3. `bench_eos_pc`: Potekhin & Chabrier (2010, 2013) non-ideal Coulomb liquid
//!    and bcc crystal EOS (`eval_pc_eos` / `eval_pc_eos_from_weights`) for a
//!    50/50 ${}^{12}\mathrm{C}/{}^{16}\mathrm{O}$ white dwarf interior crossing
//!    the liquid-to-crystal phase transition ($\log_{10}\rho \in [4.5, 7.5]$,
//!    $\log_{10}T \in [6.2, 7.4]$).
//! 4. `bench_eos_skye`: Jermyn et al. (2021) third-order Helmholtz free-energy
//!    minimization EOS (`eval_skye_eos` / `eval_skye_eos_from_weights`) for a
//!    50/50 ${}^{12}\mathrm{C}/{}^{16}\mathrm{O}$ plasma at
//!    $\log_{10}\rho \in [1.0, 6.0]$ and $\log_{10}T \in [8.0, 8.008]$.

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_eos::{
    do_convert_helm_results, eval_helm_eos, eval_ideal_eos,
    eval_ideal_eos_from_weights, eval_pc_eos, eval_pc_eos_from_weights,
    eval_skye_eos, eval_skye_eos_from_weights, helmeos2, EosEvalOutput,
    HelmConfig, HelmTable, PcConfig, SkyeConfig, I_LN_PGAS,
    NUM_EOS_BASIC_RESULTS,
};
use flexmesa_math::exp10;

use crate::kernels_micro::bench_zone_fn;
use crate::kernels_wave2::{make_wave2_composition_profile, WAVE2_ISO_NAMES};
use crate::BenchRecord;

/// Packs a 26-channel [`EosEvalOutput`] into a `[f64; 26]` array while forcing
/// evaluation of both first-derivative arrays (`d_dlnd`, `d_dln_t`).
#[inline]
fn pack_eos_res(out: &EosEvalOutput) -> [f64; NUM_EOS_BASIC_RESULTS] {
    std::hint::black_box(&out.d_dlnd);
    std::hint::black_box(&out.d_dln_t);
    out.res
}

type C12O16Props = (
    [usize; 2],
    [f64; 2],
    f64,
    f64,
    [usize; 2],
    [f64; 2],
    [f64; 2],
    [f64; 2],
);

/// Returns `(chem_ids, xa, abar, zbar, z_int, azion, acmi, a_baryon)` for the
/// canonical 50/50 ${}^{12}\mathrm{C}/{}^{16}\mathrm{O}$ benchmark mixture.
fn c12_o16_props() -> C12O16Props {
    let cat = NuclideCatalog::builtin_ref();
    let ic12 = cat.get_nuclide_index("c12").expect("valid c12");
    let io16 = cat.get_nuclide_index("o16").expect("valid o16");
    let chem_ids = [ic12, io16];
    let xa = [0.5_f64, 0.5_f64];
    let info = get_composition_info(cat, &chem_ids, &xa, true)
        .expect("valid c12/o16 composition_info");
    let nc12 = cat.get_by_id(ic12).expect("valid c12 entry");
    let no16 = cat.get_by_id(io16).expect("valid o16 entry");
    (
        chem_ids,
        xa,
        info.abar,
        info.zbar,
        [nc12.z, no16.z],
        [nc12.z as f64, no16.z as f64],
        [nc12.w, no16.w],
        [nc12.z_plus_n as f64, no16.z_plus_n as f64],
    )
}

/// Benchmarks neutral ideal-gas + radiation EOS (`eos_ideal`, 26 output
/// channels plus analytical density/temperature partials) across Stages 0..3
/// for `nz` radial zones.
///
/// - **Stage 0 & 1**: Per-zone [`eval_ideal_eos`] with nuclide catalog lookup.
/// - **Stage 2 & 3**: Hoisted species weights and baryon numbers with
///   zero-allocation [`eval_ideal_eos_from_weights`] and 4-thread `rayon`
///   parallelism.
#[must_use]
pub fn bench_eos_ideal(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let cat = NuclideCatalog::builtin_ref();
    let chem_ids: Vec<usize> = WAVE2_ISO_NAMES
        .iter()
        .map(|&name| cat.get_nuclide_index(name).expect("valid nuclide"))
        .collect();
    let xa = make_wave2_composition_profile(1)[0];
    let comp = get_composition_info(cat, &chem_ids, &xa, true)
        .expect("valid composition_info");
    let abar = comp.abar;
    let zbar = comp.zbar;
    let mut weights = [0.0_f64; 8];
    let mut a_baryon = [0.0_f64; 8];
    for (idx, &cid) in chem_ids.iter().enumerate() {
        let nuc = cat.get_by_id(cid).expect("valid nuclide");
        weights[idx] = nuc.w;
        a_baryon[idx] = nuc.z_plus_n as f64;
    }

    let denom = (nz.saturating_sub(1)).max(1) as f64;
    let mut rho = Vec::with_capacity(nz);
    let mut t = Vec::with_capacity(nz);
    for k in 0..nz {
        let s = (k as f64) / denom;
        rho.push(exp10(-6.0 + 4.0 * s));
        t.push(exp10(3.5 + 1.0 * s));
    }

    let mut rec = bench_zone_fn::<NUM_EOS_BASIC_RESULTS, _, _>(
        "eos_ideal",
        nz,
        n_warmup,
        n_repeats,
        |k| {
            let out = eval_ideal_eos(t[k], rho[k], abar, zbar, &xa, &chem_ids)
                .expect("valid eval_ideal_eos");
            pack_eos_res(&out)
        },
        |k| {
            let out = eval_ideal_eos_from_weights(
                t[k], rho[k], abar, zbar, &xa, &weights, &a_baryon,
            )
            .expect("valid eval_ideal_eos_from_weights");
            pack_eos_res(&out)
        },
    );
    rec.checksum = (0..nz)
        .map(|k| {
            eval_ideal_eos_from_weights(
                t[k], rho[k], abar, zbar, &xa, &weights, &a_baryon,
            )
            .expect("valid eval_ideal_eos_from_weights")
            .res[I_LN_PGAS]
        })
        .sum();
    rec
}

/// Benchmarks Timmes & Swesty (2000) `HELM` biquintic/bicubic EOS (`eos_helm`,
/// 26 output channels plus analytical partials) across Stages 0..3 for `nz`
/// radial zones.
#[must_use]
pub fn bench_eos_helm(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let (_, _, abar, zbar, _, _, _, _) = c12_o16_props();
    let cfg = HelmConfig::default();
    let ht = HelmTable::builtin();
    let denom = (nz.saturating_sub(1)).max(1) as f64;
    let mut log_rho = Vec::with_capacity(nz);
    let mut rho = Vec::with_capacity(nz);
    let mut log_t = Vec::with_capacity(nz);
    let mut t = Vec::with_capacity(nz);
    for k in 0..nz {
        let s = (k as f64) / denom;
        let lr = 1.0 + 5.0 * s;
        let lt = 8.0 + 0.008 * s;
        log_rho.push(lr);
        rho.push(exp10(lr));
        log_t.push(lt);
        t.push(exp10(lt));
    }

    let mut rec = bench_zone_fn::<NUM_EOS_BASIC_RESULTS, _, _>(
        "eos_helm",
        nz,
        n_warmup,
        n_repeats,
        |k| {
            let (out, _, _) = eval_helm_eos(t[k], rho[k], abar, zbar)
                .expect("valid eval_helm_eos");
            pack_eos_res(&out)
        },
        |k| {
            let (h_res, _) = helmeos2(
                ht, t[k], log_t[k], rho[k], log_rho[k], abar, zbar, &cfg,
            )
            .expect("valid helmeos2");
            let (out, _, _) =
                do_convert_helm_results(abar, zbar, rho[k], t[k], &h_res);
            pack_eos_res(&out)
        },
    );
    rec.checksum = (0..nz)
        .map(|k| {
            eval_helm_eos(t[k], rho[k], abar, zbar)
                .expect("valid eval_helm_eos")
                .0
                .res[I_LN_PGAS]
        })
        .sum();
    rec
}

/// Benchmarks Potekhin & Chabrier (2010, 2013) Coulomb liquid + bcc solid EOS
/// (`eos_pc`) across Stages 0..3 for `nz` radial zones.
#[must_use]
pub fn bench_eos_pc(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let (chem_ids, xa, abar, zbar, _, azion, acmi, _) = c12_o16_props();
    let cfg = PcConfig::default();
    let denom = (nz.saturating_sub(1)).max(1) as f64;
    let mut rho = Vec::with_capacity(nz);
    let mut t = Vec::with_capacity(nz);
    for k in 0..nz {
        let s = (k as f64) / denom;
        rho.push(exp10(4.5 + 3.0 * s));
        t.push(exp10(6.2 + 1.2 * s));
    }

    let mut rec = bench_zone_fn::<NUM_EOS_BASIC_RESULTS, _, _>(
        "eos_pc",
        nz,
        n_warmup,
        n_repeats,
        |k| {
            let out =
                eval_pc_eos(&cfg, rho[k], t[k], abar, zbar, &chem_ids, &xa)
                    .expect("valid eval_pc_eos");
            pack_eos_res(&out)
        },
        |k| {
            let out = eval_pc_eos_from_weights(
                &cfg, rho[k], t[k], abar, zbar, &xa, &azion, &acmi,
            )
            .expect("valid eval_pc_eos_from_weights");
            pack_eos_res(&out)
        },
    );
    rec.checksum = (0..nz)
        .map(|k| {
            eval_pc_eos_from_weights(
                &cfg, rho[k], t[k], abar, zbar, &xa, &azion, &acmi,
            )
            .expect("valid eval_pc_eos_from_weights")
            .res[I_LN_PGAS]
        })
        .sum();
    rec
}

/// Benchmarks Jermyn et al. (2021) third-order free-energy minimization `Skye`
/// EOS (`eos_skye`) across Stages 0..3 for `nz` radial zones.
#[must_use]
pub fn bench_eos_skye(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let (chem_ids, xa, abar, zbar, z_int, _, acmi, a_baryon) = c12_o16_props();
    let cfg = SkyeConfig::default();
    let ht = HelmTable::builtin();
    let denom = (nz.saturating_sub(1)).max(1) as f64;
    let mut rho = Vec::with_capacity(nz);
    let mut t = Vec::with_capacity(nz);
    for k in 0..nz {
        let s = (k as f64) / denom;
        rho.push(exp10(1.0 + 5.0 * s));
        t.push(exp10(8.0 + 0.008 * s));
    }

    let mut rec = bench_zone_fn::<NUM_EOS_BASIC_RESULTS, _, _>(
        "eos_skye",
        nz,
        n_warmup,
        n_repeats,
        |k| {
            let out = eval_skye_eos(
                &cfg, ht, rho[k], t[k], abar, zbar, &chem_ids, &xa,
            )
            .expect("valid eval_skye_eos");
            pack_eos_res(&out)
        },
        |k| {
            let out = eval_skye_eos_from_weights(
                &cfg, ht, t[k], rho[k], abar, zbar, &xa, &z_int, &acmi,
                &a_baryon,
            )
            .expect("valid eval_skye_eos_from_weights");
            pack_eos_res(&out)
        },
    );
    rec.checksum = (0..nz)
        .map(|k| {
            eval_skye_eos_from_weights(
                &cfg, ht, t[k], rho[k], abar, zbar, &xa, &z_int, &acmi,
                &a_baryon,
            )
            .expect("valid eval_skye_eos_from_weights")
            .res[I_LN_PGAS]
        })
        .sum();
    rec
}

/// Runs the complete Wave 3a four-stage benchmark suite (`eos_ideal`,
/// `eos_helm`, `eos_pc`, and `eos_skye`) across `zone_counts` and returns all
/// [`BenchRecord`] entries.
#[must_use]
pub fn run_all_wave3a_benchmarks(
    zone_counts: &[usize],
    n_warmup: usize,
    n_repeats: usize,
) -> Vec<BenchRecord> {
    let mut records = Vec::with_capacity(zone_counts.len() * 4);
    for &nz in zone_counts {
        records.push(bench_eos_ideal(nz, n_warmup, n_repeats));
        records.push(bench_eos_helm(nz, n_warmup, n_repeats));
        records.push(bench_eos_pc(nz, n_warmup, n_repeats));
        records.push(bench_eos_skye(nz, n_warmup, n_repeats));
    }
    records
}
