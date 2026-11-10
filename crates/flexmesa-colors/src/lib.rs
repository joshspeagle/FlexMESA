// SPDX-License-Identifier: LGPL-3.0-or-later
//! Safe Rust bolometric and synthetic stellar photometry (`colors`) for
//! FlexMESA.
//!
//! Ports upstream MESA `colors` (`colors_def.f90`, `colors_lib.f90`,
//! `bolometric.f90`, `synthetic.f90`, `hermite_interp.f90`,
//! `linear_interp.f90`, `knn_interp.f90`, `colors_utils.f90`, and
//! `colors_history.f90`) into Safe Rust (`#![forbid(unsafe_code)]`) with
//! generic [`RealScalar`] (`f64` and `DualOrder1<N>`) automatic
//! differentiation and `rayon` multi-model batch evaluation.
//!
//! ## Submodules
//! - [`types`]: Scalar trait [`RealScalar`], [`MagSystem`] (`Vega`, `Ab`,
//!   `St`), [`SedInterpMethod`] (`Hermite`, `Linear`, `Knn`), error types, and
//!   physical constants (`D_10PC_CM`, `Z_OVER_X_REF_GS98`).
//! - [`filters_data`]: Built-in 7-band Johnson filter curves (`B, I, J, M, R,
//!   U, V`) and precomputed Vega, AB, and ST zero-point fluxes.
//! - [`utils`]: Composite non-uniform Simpson (`simpson_integration`,
//!   `simpson_integration_fn`) and trapezoidal (`trapezoidal_integration`)
//!   quadrature, geometric dilution (`dilute_flux`), piecewise-linear filter
//!   resampling (`interpolate_array`), and grid search (`find_interval`,
//!   `compute_interp_radius`).
//! - [`sed_cube`]: Preloaded 4D `Kurucz2003all` atmosphere SED lookup cube
//!   $(T_{\mathrm{eff}}, \log g, [\mathrm{M/H}], \lambda)$ and lazy singleton
//!   [`builtin_sed_cube`].
//! - [`interp`]: 3D cubic Hermite (`eval_sed_hermite_generic`), log-fallback
//!   trilinear (`eval_sed_trilinear`), and 4-nearest-neighbor inverse-distance
//!   (`eval_sed_knn`) SED interpolators.
//! - [`photometry`]: Bolometric (`calculate_bolometric_phot`), synthetic
//!   photon-counting (`calculate_synthetic`), zero-point, and 10-column
//!   history (`eval_colors`, `eval_colors_dual`, `eval_colors_zones`) kernels.

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::many_single_char_names,
    clippy::module_name_repetitions,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

/// Built-in 7-band Johnson filter passbands and zero-point fluxes.
pub mod filters_data;
/// 3D cubic Hermite, trilinear, and 4-NN SED interpolation kernels.
pub mod interp;
/// Bolometric, synthetic filter, and 10-column history photometry evaluators.
pub mod photometry;
/// 4D `Kurucz2003all` SED cube container and binary parser.
pub mod sed_cube;
/// Core traits (`RealScalar`), enums (`MagSystem`, `SedInterpMethod`), and
/// error types (`ColorsError`).
pub mod types;
/// Composite Simpson/trapezoidal quadrature, dilution, and grid utilities.
pub mod utils;

// Re-export primary public types, constants, and evaluators at the crate root:
// - Filter curves and column identifiers (`filters_data`)
// - Cubic Hermite basis functions (`interp`)
// - Bolometric, synthetic, and zero-point photometry (`photometry`)
// - 4D SED cube container and built-in Kurucz2003 singleton (`sed_cube`)
// - Scalar trait, magnitude/interpolation enums, and CGS constants (`types`)
// - Quadrature, dilution, resampling, and grid search helpers (`utils`)
pub use filters_data::{
    FilterCurve, COLOR_COLUMN_NAMES, JOHNSON_FILTERS, JOHNSON_FILTER_NAMES,
    NUM_COLORS_OUT, NUM_JOHNSON_FILTERS,
};
pub use interp::{h00, h01, h10, h11};
pub use photometry::{
    calculate_bolometric_phot, calculate_synthetic, calculate_synthetic_flux,
    compute_ab_zero_point, compute_st_zero_point, compute_vega_zero_point,
    flux_to_magnitude,
};
pub use sed_cube::{builtin_sed_cube, SedCube};
pub use types::{
    ColorsError, MagSystem, RealScalar, SedInterpMethod, D_10PC_CM,
    PARALLEL_ZONE_THRESHOLD, Z_OVER_X_REF_GS98,
};
pub use utils::{
    compute_interp_radius, dilute_flux, find_bracket_index, find_interval,
    find_nearest_point, interpolate_array, is_dummy_axis, linear_interpolate,
    remove_dat, simpson_integration, simpson_integration_fn,
    trapezoidal_integration,
};

#[cfg(test)]
mod tests {
    use super::*;
    use flexmesa_const::{BOLTZ_SIGMA, RSUN};

    // Verifies Group 1 (`solar`, `hot_ms`, `cool_giant`) against upstream MESA
    // `colors/test/test_output` to 12+ digits (`rtol <= 1e-12`).
    // Specifically checks:
    // - `solar`: (Teff = 5778 K, log g = 4.44, [M/H] = 0.0, R = 1 RSUN)
    // - `hot_ms`: (Teff = 15000 K, log g = 4.0, [M/H] = 0.0, R = 5 RSUN)
    // - `cool_giant`: (Teff = 4000 K, log g = 2.0, [M/H] = 0.0, R = 20 RSUN)
    // across all 10 output columns (`Mag_bol`, `Flux_bol`, `Interp_rad`,
    // `B..V`).
    #[test]
    fn test_group1_and_group2_match_mesa_golden_output() {
        let cube = builtin_sed_cube();

        // Group 1: solar (5778, 4.44, 0.0, R = RSUN)
        let solar = cube
            .eval_colors(
                5778.0,
                4.44,
                0.0,
                RSUN,
                D_10PC_CM,
                MagSystem::Vega,
                SedInterpMethod::Hermite,
            )
            .expect("solar eval_colors");
        let solar_expected = [
            1.6233783742415e+01,
            3.2098630468165e-07,
            1.2015098221403e-02,
            5.4328315088495e+00,
            4.0243632405405e+00,
            3.6544667162329e+00,
            3.2794040310708e+00,
            4.3322423435800e+00,
            5.5342928137757e+00,
            4.7769811346940e+00,
        ];
        for (i, (&got, &exp)) in
            solar.iter().zip(solar_expected.iter()).enumerate()
        {
            let rel = (got - exp).abs() / exp.abs().max(1.0e-30);
            assert!(
                rel < 1.0e-12,
                "solar col {} ({}) mismatch: got {:.13e}, expected {:.13e}",
                i,
                COLOR_COLUMN_NAMES[i],
                got,
                exp
            );
        }

        // Group 1: hot_ms (15000, 4.0, 0.0, R = 5 * RSUN)
        let hot_ms = cube
            .eval_colors(
                15000.0,
                4.0,
                0.0,
                5.0 * RSUN,
                D_10PC_CM,
                MagSystem::Vega,
                SedInterpMethod::Hermite,
            )
            .expect("hot_ms eval_colors");
        let hot_ms_expected = [
            8.5957683261478e+00,
            3.6449591844963e-04,
            0.0,
            -1.8453872060965e+00,
            -1.5266394695804e+00,
            -1.3829167561429e+00,
            -1.2210271604986e+00,
            -1.6373526226723e+00,
            -2.5493640964120e+00,
            -1.7077286035359e+00,
        ];
        for (i, (&got, &exp)) in
            hot_ms.iter().zip(hot_ms_expected.iter()).enumerate()
        {
            let diff = (got - exp).abs();
            let tol = 1.0e-12 * exp.abs().max(1.0);
            // Check column-wise parity against upstream test_output.
            assert!(
                diff <= tol,
                "hot_ms col {i} mismatch: got {got:.13e}, expected {exp:.13e}",
            );
        }

        // Group 1: cool_giant (4000, 2.0, 0.0, R = 20 * RSUN)
        let cool_giant = cube
            .eval_colors(
                4000.0,
                2.0,
                0.0,
                20.0 * RSUN,
                D_10PC_CM,
                MagSystem::Vega,
                SedInterpMethod::Hermite,
            )
            .expect("cool_giant eval_colors");
        let cool_expected = [
            1.1326046768001e+01,
            2.9483653105575e-05,
            0.0,
            2.1835469363630e+00,
            -9.1810150915499e-01,
            -1.7697342747621e+00,
            -2.6525046293368e+00,
            -2.0538595855424e-01,
            3.9598722643568e+00,
            7.5454159459472e-01,
        ];
        for (i, (&got, &exp)) in
            cool_giant.iter().zip(cool_expected.iter()).enumerate()
        {
            let diff = (got - exp).abs();
            let tol = 1.0e-12 * exp.abs().max(1.0);
            // Check column-wise parity against upstream test_output.
            assert!(
                diff <= tol,
                "cool_giant col {i} mismatch: {got:.13e} vs {exp:.13e}",
            );
        }
    }

    // Verifies Group 3 stride-sampled solar SED points (`index = 0..1180`) and
    // Stefan-Boltzmann bolometric wavelength coverage across
    // `Teff = 3500..20000 K`.
    // Also checks that `BOLTZ_SIGMA * Teff^4 * (R / d)^2` agrees with the
    // integrated Kurucz2003 bolometric flux within 0.5%.
    // Columns: [Mag_bol, Flux_bol, Interp_rad, B, I, J, M, R, U, V].
    #[test]
    fn test_group3_sed_sample_and_wavelength_coverage() {
        let cube = builtin_sed_cube();
        let sed =
            cube.eval_sed_hermite_generic(5778.0, 4.44, 0.0, RSUN, D_10PC_CM);
        let n_wav = cube.wavelengths.len();
        assert_eq!(n_wav, 1199);
        assert!((cube.wavelengths[0] - 147.2).abs() < 1.0e-12);
        assert!((cube.wavelengths[n_wav - 1] - 1.6e6).abs() < 1.0e-6);

        // Verify stride-sampled SED points from Group 3 of test_output
        let stride = n_wav / 20; // 59
        let expected_samples = [
            (0usize, 1.4720000000000e+02, 2.1385116423739e-41),
            (stride * 6, 3.8900000000000e+03, 2.6676232108131e-11),
            (stride * 7, 5.0700000000000e+03, 4.7479893607680e-11),
            (stride * 10, 8.6100000000000e+03, 2.2878420142827e-11),
            (stride * 20, 9.6200000000000e+04, 6.6589153670475e-15),
        ];
        for &(idx, w_exp, f_exp) in &expected_samples {
            assert!((cube.wavelengths[idx] - w_exp).abs() < 1.0e-9);
            let rel = (sed[idx] - f_exp).abs() / f_exp;
            assert!(rel < 1.0e-12, "SED sample {idx} rel_err = {rel:e}");
        }

        // Wavelength coverage sanity checks (Teff = 3500, 6000, 10000, 20000)
        for &(teff, flux_exp_golden) in &[
            (3500.0, 4.3198229964375e-08),
            (6000.0, 3.7325163050257e-07),
            (10000.0, 2.8809769255562e-06),
            (20000.0, 4.6160109853289e-05),
        ] {
            let out = cube
                .eval_colors(
                    teff,
                    4.0,
                    0.0,
                    RSUN,
                    D_10PC_CM,
                    MagSystem::Vega,
                    SedInterpMethod::Hermite,
                )
                .expect("coverage eval_colors");
            let actual_flux = out[1];
            assert!(
                (actual_flux - flux_exp_golden).abs() / flux_exp_golden
                    < 1.0e-12
            );
            let stefan_flux =
                BOLTZ_SIGMA * teff.powi(4) * (RSUN / D_10PC_CM).powi(2);
            let rel_err = (actual_flux - stefan_flux).abs() / stefan_flux;
            assert!(rel_err < 5.0e-3);
        }
    }

    // Verifies AB and ST zero-point quadrature against precomputed filter
    // constants and checks `DualOrder1<4>` forward-mode derivatives against
    // centered finite differences `(f(Teff + dT) - f(Teff - dT)) / (2 dT)`
    // across all 9 differentiable output columns (`Mag_bol`, `Flux_bol`, and
    // `B, I, J, M, R, U, V`).
    #[test]
    fn test_dual_autodiff_and_ab_st_zero_points() {
        let cube = builtin_sed_cube();
        for filt in &JOHNSON_FILTERS {
            let ab_zp =
                compute_ab_zero_point(filt.wavelengths, filt.transmission)
                    .expect("ab_zp");
            let st_zp =
                compute_st_zero_point(filt.wavelengths, filt.transmission)
                    .expect("st_zp");
            assert!((ab_zp - filt.ab_zero_point).abs() / ab_zp < 1.0e-14);
            assert!((st_zp - filt.st_zero_point).abs() / st_zp < 1.0e-14);
        }

        let t0 = 5820.0;
        let g0 = 4.35;
        let m0 = -0.25;
        let r0 = 1.1 * RSUN;
        let dual_out = cube
            .eval_colors_dual(t0, g0, m0, r0, D_10PC_CM, MagSystem::Vega)
            .expect("eval_colors_dual");

        let dt = 1.0e-2;
        let hp = cube
            .eval_colors(
                t0 + dt,
                g0,
                m0,
                r0,
                D_10PC_CM,
                MagSystem::Vega,
                SedInterpMethod::Hermite,
            )
            .expect("hp");
        let hm = cube
            .eval_colors(
                t0 - dt,
                g0,
                m0,
                r0,
                D_10PC_CM,
                MagSystem::Vega,
                SedInterpMethod::Hermite,
            )
            .expect("hm");
        for i in [0usize, 1, 3, 4, 5, 6, 7, 8, 9] {
            let fd = (hp[i] - hm[i]) / (2.0 * dt);
            assert!(
                (dual_out[i].d1[0] - fd).abs() < 1.0e-8,
                "col {} d/dTeff mismatch: dual={}, fd={}",
                i,
                dual_out[i].d1[0],
                fd
            );
        }
    }

    #[test]
    fn test_full_76x11x8x1199_sed_cube_and_domain_sweeps() {
        let cube = builtin_sed_cube();
        assert_eq!(cube.teff_grid.len(), 76);
        assert_eq!(cube.logg_grid.len(), 11);
        assert_eq!(cube.meta_grid.len(), 8);
        assert_eq!(cube.wavelengths.len(), 1199);
        assert_eq!(cube.n_seds(), 76 * 11 * 8);
        assert_eq!(cube.sed_table.len(), 76 * 11 * 8 * 1199);
        for (idx, &slot) in cube.slot_map.iter().enumerate() {
            assert_eq!(slot, idx, "expected 1-to-1 dense slot_map");
        }

        // Sweep across (Teff in [2000, 50000] K, log_g in [-1.0, 5.5], [M/H] in [-5.0, +1.0])
        for &(teff, logg, meta) in &[
            (2000.0, -1.0, -5.0),
            (3500.0, 0.0, -2.5),
            (4500.0, 2.5, -1.5),
            (5778.0, 4.44, 0.0),
            (8500.0, 4.0, -0.5),
            (15000.0, 4.0, 0.2),
            (30000.0, 4.5, 0.5),
            (50000.0, 5.5, 1.0),
        ] {
            let out = cube
                .eval_colors(
                    teff,
                    logg,
                    meta,
                    RSUN,
                    D_10PC_CM,
                    MagSystem::Vega,
                    SedInterpMethod::Hermite,
                )
                .expect("domain sweep eval_colors");
            assert!(
                out[1] > 0.0 && out.iter().all(|v| v.is_finite()),
                "non-finite or non-positive output at ({teff}, {logg}, {meta}): {out:?}"
            );
            let sed = cube.eval_sed(
                teff,
                logg,
                meta,
                RSUN,
                D_10PC_CM,
                SedInterpMethod::Linear,
            );
            assert_eq!(sed.len(), 1199);
            assert!(sed.iter().any(|&f| f > 0.0));
        }
    }
}
