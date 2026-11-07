// SPDX-License-Identifier: LGPL-3.0-or-later
//! Wave 2 four-stage microphysics benchmarks (`chem_comp`, `ion_fe56`,
//! `ion_4d`, and `colors_sed`).
//!
//! Evaluates the four Wave 2 atomic and photometric microphysics workloads
//! across four progressive optimization stages in Safe Rust:
//! 1. `bench_chem_comp`: 8-isotope stellar composition moments (`X, Y, Z,
//!    abar, zbar, z2bar, z53bar, ye, mass_correction, sumx`) across a
//!    core-to-surface composition gradient (`get_composition_info`).
//! 2. `bench_ion_fe56`: Potekhin ${}^{56}\mathrm{Fe}$-in-${}^4\mathrm{He}$
//!    effective charge and partial derivatives (`[z, dz_dlog10_ne,
//!    dz_dlog10_T]`) on the $105 \times 30$ bicubic spline grid.
//! 3. `bench_ion_4d`: 4D `(Z, X, log10_rho, log10_T)` trilinear-over-bicubic
//!    mean ionic charge interpolation across all 28 `eval_ionization` channels.
//! 4. `bench_colors_sed`: 3D `(Teff, log_g, [M/H])` `Kurucz2003all` cubic
//!    Hermite SED synthesis (`1199` wavelengths) plus bolometric integration
//!    and 7-band `Generic/Johnson` (`B, I, J, M, R, U, V`) Vega photometry.

use flexmesa_chem::{
    eval_composition_moments_generic, get_composition_info, NuclideCatalog,
    SpeciesProps,
};
use flexmesa_colors::{
    builtin_sed_cube, calculate_bolometric_phot, compute_interp_radius,
    flux_to_magnitude, simpson_integration_fn, MagSystem, SedInterpMethod,
    D_10PC_CM, JOHNSON_FILTERS, NUM_COLORS_OUT, NUM_JOHNSON_FILTERS,
};
use flexmesa_const::RSUN;
use flexmesa_ionization::{
    eval_ionization, Fe56He4Table, IonTableGrid, NUM_ION_VALS,
};

use crate::bicub_fast::eval_bicub_prevalidated;
use crate::kernels_micro::bench_zone_fn;
use crate::BenchRecord;

/// Canonical 8-isotope stellar benchmark network (`h1`, `he3`, `he4`, `c12`,
/// `n14`, `o16`, `ne20`, `mg24`).
pub const WAVE2_ISO_NAMES: [&str; 8] =
    ["h1", "he3", "he4", "c12", "n14", "o16", "ne20", "mg24"];

/// Builds the deterministic 8-isotope mass fraction profile `xa` of length
/// `nz` spanning core-to-surface stellar compositions.
///
/// For zone $k \in [0, N_{\mathrm{zones}} - 1]$ with normalized radial
/// coordinate $s = k / \max(N_{\mathrm{zones}} - 1, 1) \in [0, 1]$:
/// - $Z(s) = 0.014 + 0.016 s$
/// - $X(s) = 0.70 (1 - s) + 0.05 s$
/// - $Y(s) = 1 - X(s) - Z(s)$
#[must_use]
pub fn make_wave2_composition_profile(nz: usize) -> Vec<[f64; 8]> {
    let denom = (nz.saturating_sub(1)).max(1) as f64;
    (0..nz)
        .map(|k| {
            let s = (k as f64) / denom;
            let z = 0.014 + 0.016 * s;
            let x = 0.70 * (1.0 - s) + 0.05 * s;
            let y = 1.0 - x - z;
            [
                x,
                1.0e-4 * y,
                0.9999 * y,
                0.18 * z,
                0.06 * z,
                0.48 * z,
                0.14 * z,
                0.14 * z,
            ]
        })
        .collect()
}

/// Benchmarks 8-isotope stellar composition moment evaluation (`chem_comp`)
/// across Stages 0..3 for `nz` radial zones.
///
/// - **Stage 0 & 1**: Per-zone `get_composition_info` with catalog lookup and
///   vector allocation.
/// - **Stage 2 & 3**: Hoisted [`SpeciesProps`] with stack-resident
///   [`eval_composition_moments_generic`] and 4-thread `rayon` parallelism.
#[must_use]
pub fn bench_chem_comp(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let cat = NuclideCatalog::builtin();
    let chem_ids: Vec<usize> = WAVE2_ISO_NAMES
        .iter()
        .map(|&name| cat.get_nuclide_index(name).expect("valid nuclide"))
        .collect();
    let props = SpeciesProps::from_catalog(&cat, &chem_ids)
        .expect("valid SpeciesProps");
    let xa_rows = make_wave2_composition_profile(nz);

    let mut rec = bench_zone_fn::<10, _, _>(
        "chem_comp",
        nz,
        n_warmup,
        n_repeats,
        |k| {
            let info = get_composition_info(&cat, &chem_ids, &xa_rows[k], true)
                .expect("valid composition_info");
            [
                info.xh,
                info.xhe,
                info.xz,
                info.abar,
                info.zbar,
                info.z2bar,
                info.z53bar,
                info.ye,
                info.mass_correction,
                info.sumx,
            ]
        },
        |k| {
            eval_composition_moments_generic::<f64>(&props, &xa_rows[k])
                .to_array()
        },
    );
    // Sum column 3 (mean atomic weight `abar`) for cross-language verification.
    rec.checksum = (0..nz)
        .map(|k| {
            eval_composition_moments_generic::<f64>(&props, &xa_rows[k]).abar
        })
        .sum();
    rec
}

/// Benchmarks Potekhin ${}^{56}\mathrm{Fe}$-in-${}^4\mathrm{He}$ effective
/// charge and partial derivative evaluation (`ion_fe56`) across Stages 0..3.
///
/// The cross-language `checksum` sums the effective charge $Z_{\mathrm{eff}}$
/// (column 0) across all `nz` zones so it matches upstream Fortran MESA's
/// scalar `eval_charge_of_Fe56_in_He4` directly.
#[must_use]
pub fn bench_ion_fe56(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let denom = (nz.saturating_sub(1)).max(1) as f64;
    let lne: Vec<f64> =
        (0..nz).map(|k| 24.5 + 5.5 * (k as f64) / denom).collect();
    let lt: Vec<f64> =
        (0..nz).map(|k| 6.2 + 1.6 * (k as f64) / denom).collect();
    let tbl = Fe56He4Table::builtin();

    bench_zone_fn::<3, _, _>(
        "ion_fe56",
        nz,
        n_warmup,
        n_repeats,
        |k| {
            let ev = tbl.eval_full(lne[k], lt[k]).expect("valid Fe56 query");
            [ev.f, ev.df_dx, ev.df_dy]
        },
        |k| {
            let [f, df_dx, df_dy, _, _, _] = eval_bicub_prevalidated(
                lne[k],
                lt[k],
                &tbl.log_ne,
                &tbl.log_t,
                &tbl.coeffs,
            );
            [f, df_dx, df_dy]
        },
    )
}

/// Benchmarks 4D `(Z, X, log10_rho, log10_T)` trilinear-over-bicubic
/// ionization table evaluation (`ion_4d`, 28 output channels) across
/// Stages 0..3.
#[must_use]
pub fn bench_ion_4d(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let denom = (nz.saturating_sub(1)).max(1) as f64;
    let mut z_vec = Vec::with_capacity(nz);
    let mut x_vec = Vec::with_capacity(nz);
    let mut lrho_vec = Vec::with_capacity(nz);
    let mut lt_vec = Vec::with_capacity(nz);
    for k in 0..nz {
        let s = (k as f64) / denom;
        let z = 0.005 + 0.030 * s;
        let x = 0.62 + 0.16 * s;
        let lt = 4.505 + 0.050 * s;
        let lq = -2.075 + 0.080 * s;
        let lrho = lq + 2.0 * lt - 12.0;
        z_vec.push(z);
        x_vec.push(x);
        lrho_vec.push(lrho);
        lt_vec.push(lt);
    }
    let grid = IonTableGrid::builtin();

    bench_zone_fn::<NUM_ION_VALS, _, _>(
        "ion_4d",
        nz,
        n_warmup,
        n_repeats,
        |k| {
            eval_ionization(
                z_vec[k],
                x_vec[k],
                None,
                Some(lrho_vec[k]),
                None,
                Some(lt_vec[k]),
            )
            .expect("valid eval_ionization")
        },
        |k| grid.eval_generic(z_vec[k], x_vec[k], lrho_vec[k], lt_vec[k]),
    )
}

/// Benchmarks 3D `Kurucz2003all` cubic Hermite SED synthesis (`1199`
/// wavelengths) plus bolometric and 7-band `Generic/Johnson` Vega photometry
/// (`colors_sed`, 10 output columns) across Stages 0..3.
///
/// - **Stage 0 & 1**: Allocates a `Vec<f64>` SED per model and evaluates
///   bolometric + 7 separate filter Simpson integrals.
/// - **Stage 2 & 3**: Evaluates the Hermite SED into a stack buffer `[f64;
///   1199]` and computes bolometric + 7 Johnson filter integrals in a single
///   fused pass over precomputed Simpson panel weights `[c0, c1, c2]`.
#[must_use]
pub fn bench_colors_sed(
    nz: usize,
    n_warmup: usize,
    n_repeats: usize,
) -> BenchRecord {
    let denom = (nz.saturating_sub(1)).max(1) as f64;
    let mut teff = Vec::with_capacity(nz);
    let mut logg = Vec::with_capacity(nz);
    let mut meta = Vec::with_capacity(nz);
    let mut rad = Vec::with_capacity(nz);
    for k in 0..nz {
        let s = (k as f64) / denom;
        teff.push(5760.0 + 230.0 * s);
        logg.push(4.05 + 0.40 * s);
        meta.push(-0.45 + 0.60 * s);
        rad.push(RSUN * (0.9 + 0.4 * s));
    }
    let cube = builtin_sed_cube();

    let mut rec = bench_zone_fn::<NUM_COLORS_OUT, _, _>(
        "colors_sed",
        nz,
        n_warmup,
        n_repeats,
        |k| {
            let irad = compute_interp_radius(
                teff[k],
                logg[k],
                meta[k],
                &cube.teff_grid,
                &cube.logg_grid,
                &cube.meta_grid,
            );
            let sed = cube.eval_sed_hermite_generic(
                teff[k], logg[k], meta[k], rad[k], D_10PC_CM,
            );
            let (mag_bol, flux_bol) =
                calculate_bolometric_phot(&cube.wavelengths, &sed)
                    .expect("valid bolometric");
            let mut out = [0.0_f64; NUM_COLORS_OUT];
            out[0] = mag_bol;
            out[1] = flux_bol;
            out[2] = irad;
            let nlam = cube.wavelengths.len();
            for f_idx in 0..NUM_JOHNSON_FILTERS {
                let tr = &cube.filters_on_sed_grid[f_idx];
                let num =
                    simpson_integration_fn(&cube.wavelengths, nlam, |i| {
                        let f_clean = if i + 1 < nlam
                            && (cube.wavelengths[i] <= 0.0 || sed[i] < 0.0)
                        {
                            0.0
                        } else {
                            sed[i]
                        };
                        f_clean * tr[i] * cube.wavelengths[i]
                    })
                    .expect("valid filter integral");
                let syn = num / cube.filter_norm_integrals[f_idx];
                let zp = JOHNSON_FILTERS[f_idx].zero_point(MagSystem::Vega);
                out[3 + f_idx] = flux_to_magnitude(syn / zp);
            }
            out
        },
        |k| {
            cube.eval_colors(
                teff[k],
                logg[k],
                meta[k],
                rad[k],
                D_10PC_CM,
                MagSystem::Vega,
                SedInterpMethod::Hermite,
            )
            .expect("valid eval_colors")
        },
    );
    // Sum column 9 (Johnson V magnitude) for cross-language parity.
    rec.checksum = (0..nz)
        .map(|k| {
            cube.eval_colors(
                teff[k],
                logg[k],
                meta[k],
                rad[k],
                D_10PC_CM,
                MagSystem::Vega,
                SedInterpMethod::Hermite,
            )
            .expect("valid eval_colors")[9]
        })
        .sum();
    rec
}

/// Runs the complete Wave 2 four-stage benchmark suite (`chem_comp`,
/// `ion_fe56`, `ion_4d`, and `colors_sed`) across `zone_counts` and returns all
/// [`BenchRecord`] entries.
#[must_use]
pub fn run_all_wave2_benchmarks(
    zone_counts: &[usize],
    n_warmup: usize,
    n_repeats: usize,
) -> Vec<BenchRecord> {
    let mut records = Vec::with_capacity(zone_counts.len() * 4);
    for &nz in zone_counts {
        records.push(bench_chem_comp(nz, n_warmup, n_repeats));
        records.push(bench_ion_fe56(nz, n_warmup, n_repeats));
        records.push(bench_ion_4d(nz, n_warmup, n_repeats));
        records.push(bench_colors_sed(nz, n_warmup, n_repeats));
    }
    records
}
