// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone zero-`$MESA_DIR` MESA `26.4.1` parity tests for
//! `flexmesa-colors` (`colors/test/src/test_colors.f90`,
//! `colors/private/*.f90`, and `star/test_suite/custom_colors`).

#![forbid(unsafe_code)]
#![allow(clippy::too_many_lines, clippy::unreadable_literal)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use flexmesa_colors::{
    builtin_sed_cube, calculate_bolometric_phot, calculate_synthetic,
    compute_ab_zero_point, compute_interp_radius, compute_interp_radius_lookup,
    compute_st_zero_point, compute_vega_zero_point, ColorsContext, ColorsError,
    ColorsLoadConfig, MagSystem, SedCube, SedInterpMethod, COLOR_COLUMN_NAMES,
    D_10PC_CM, NUM_COLORS_OUT, Z_OVER_X_REF_GS98,
};
use flexmesa_const::{BOLTZ_SIGMA, RSUN};
use flexmesa_math::{
    intrinsic_exp, intrinsic_log, intrinsic_log10, pow2, pow4,
    with_numerics_mode, NumericsMode,
};

static TEST_TMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

struct TempDirGuard(PathBuf);

impl TempDirGuard {
    fn new(tag: &str) -> Self {
        let id = TEST_TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "flexmesa_colors_parity_{tag}_{}_{id}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("create temp parity test dir");
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn in_strict<R>(f: impl FnOnce() -> R) -> R {
    with_numerics_mode(NumericsMode::Strict, f)
}

fn assert_rel_close(
    actual: f64,
    expected: f64,
    rel_tol: f64,
    abs_floor: f64,
    label: &str,
) {
    let diff = (actual - expected).abs();
    let tol = (expected.abs() * rel_tol).max(abs_floor);
    assert!(
        diff <= tol,
        "{label}: actual={actual:.17e}, expected={expected:.17e}, \
         diff={diff:.3e} > tol={tol:.3e}"
    );
}

fn default_kurucz_johnson_cfg<'a>() -> ColorsLoadConfig<'a> {
    ColorsLoadConfig {
        instrument: "data/colors_data/filters/Generic/Johnson",
        stellar_atm: "data/colors_data/stellar_models/Kurucz2003all",
        vega_sed: "data/colors_data/stellar_models/vega_flam.csv",
        mag_system: "Vega",
        distance: D_10PC_CM,
        z_over_x_ref: Z_OVER_X_REF_GS98,
        make_csv: false,
        sed_per_model: false,
        colors_results_directory: "SED",
    }
}

/// Verifies all 14 MESA `26.4.1` `colors/test/test_output` reference cases
/// (Group 1, Group 2a, Group 2b, Group 2c) offline without `$MESA_DIR` via
/// both `SedCube::eval_colors` and `ColorsContext::eval_and_maybe_write_csv`
/// (`colors/test/src/test_colors.f90:59-257`).
#[test]
fn test_mesa_26_4_1_reference_groups_1_and_2_zero_mesa_dir() {
    #[derive(Clone, Copy)]
    struct RefCase {
        label: &'static str,
        teff: f64,
        logg: f64,
        meta: f64,
        phot_x: f64,
        phot_z: f64,
        r_cm: f64,
        expected: [f64; NUM_COLORS_OUT],
    }

    let root10 = 10.0_f64.sqrt();
    let cases = [
        // Group 1 (`colors/test/test_output:5-37`)
        RefCase {
            label: "Group1_solar",
            teff: 5778.0,
            logg: 4.44,
            meta: 0.0,
            phot_x: 1.0,
            phot_z: Z_OVER_X_REF_GS98,
            r_cm: RSUN,
            expected: [
                1.6233783742415E+01,
                3.2098630468165E-07,
                1.2015098221403E-02,
                5.4328315088495E+00,
                4.0243632405405E+00,
                3.6544667162329E+00,
                3.2794040310708E+00,
                4.3322423435800E+00,
                5.5342928137757E+00,
                4.7769811346940E+00,
            ],
        },
        RefCase {
            label: "Group1_hot_ms",
            teff: 15000.0,
            logg: 4.0,
            meta: 0.0,
            phot_x: 1.0,
            phot_z: Z_OVER_X_REF_GS98,
            r_cm: 5.0 * RSUN,
            expected: [
                8.5957683261478E+00,
                3.6449591844963E-04,
                0.0,
                -1.8453872060965E+00,
                -1.5266394695804E+00,
                -1.3829167561429E+00,
                -1.2210271604986E+00,
                -1.6373526226723E+00,
                -2.5493640964120E+00,
                -1.7077286035359E+00,
            ],
        },
        RefCase {
            label: "Group1_cool_giant",
            teff: 4000.0,
            logg: 2.0,
            meta: 0.0,
            phot_x: 1.0,
            phot_z: Z_OVER_X_REF_GS98,
            r_cm: 20.0 * RSUN,
            expected: [
                1.1326046768001E+01,
                2.9483653105575E-05,
                0.0,
                2.1835469363630E+00,
                -9.1810150915499E-01,
                -1.7697342747621E+00,
                -2.6525046293368E+00,
                -2.0538595855424E-01,
                3.9598722643568E+00,
                7.5454159459472E-01,
            ],
        },
        // Group 2a (`colors/test/test_output:41-84`, MH=0.0 matches solar)
        RefCase {
            label: "Group2a_MH_-2.0",
            teff: 5778.0,
            logg: 4.44,
            meta: -2.0,
            phot_x: 100.0,
            phot_z: Z_OVER_X_REF_GS98,
            r_cm: RSUN,
            expected: [
                1.6233121880603E+01,
                3.2118203670543E-07,
                1.2015098221403E-02,
                5.4155699684922E+00,
                4.1196749571589E+00,
                3.7331196442677E+00,
                3.2764174462220E+00,
                4.4562859813888E+00,
                5.1227256937938E+00,
                4.8915514395591E+00,
            ],
        },
        RefCase {
            label: "Group2a_MH_-1.0",
            teff: 5778.0,
            logg: 4.44,
            meta: -1.0,
            phot_x: 10.0,
            phot_z: Z_OVER_X_REF_GS98,
            r_cm: RSUN,
            expected: [
                1.6234071206333E+01,
                3.2090133029621E-07,
                1.2015098221403E-02,
                5.4129001639500E+00,
                4.0797030492881E+00,
                3.7008262418547E+00,
                3.2696192851353E+00,
                4.4046484941712E+00,
                5.2577215874254E+00,
                4.8394911534358E+00,
            ],
        },
        RefCase {
            label: "Group2a_MH_+0.5",
            teff: 5778.0,
            logg: 4.44,
            meta: 0.5,
            phot_x: 1.0,
            phot_z: Z_OVER_X_REF_GS98 * root10,
            r_cm: RSUN,
            expected: [
                1.6234073995658E+01,
                3.2090050588168E-07,
                1.2015098221403E-02,
                5.4709583044458E+00,
                3.9901647950517E+00,
                3.6279241638132E+00,
                3.2887774292217E+00,
                4.2891305098437E+00,
                5.7511450514938E+00,
                4.7518221763185E+00,
            ],
        },
        // Group 2b (`colors/test/test_output:88-131`)
        RefCase {
            label: "Group2b_logg_1.0",
            teff: 5778.0,
            logg: 1.0,
            meta: 0.0,
            phot_x: 1.0,
            phot_z: Z_OVER_X_REF_GS98,
            r_cm: RSUN,
            expected: [
                1.6233777481775E+01,
                3.2098815557824E-07,
                6.0215053763441E-04,
                5.3645167572931E+00,
                3.9946594776964E+00,
                3.6420933559602E+00,
                3.2639524119839E+00,
                4.2976154690770E+00,
                5.8874283229464E+00,
                4.7180779330260E+00,
            ],
        },
        RefCase {
            label: "Group2b_logg_2.5",
            teff: 5778.0,
            logg: 2.5,
            meta: 0.0,
            phot_x: 1.0,
            phot_z: Z_OVER_X_REF_GS98,
            r_cm: RSUN,
            expected: [
                1.6233688922092E+01,
                3.2101433852074E-07,
                6.0215053763441E-04,
                5.3921064688310E+00,
                4.0149191259301E+00,
                3.6490375905121E+00,
                3.2745403559284E+00,
                4.3196935465011E+00,
                5.6695916672821E+00,
                4.7516224781321E+00,
            ],
        },
        RefCase {
            label: "Group2b_logg_4.0",
            teff: 5778.0,
            logg: 4.0,
            meta: 0.0,
            phot_x: 1.0,
            phot_z: Z_OVER_X_REF_GS98,
            r_cm: RSUN,
            expected: [
                1.6233750920162E+01,
                3.2099600837665E-07,
                6.0215053763441E-04,
                5.4245311240969E+00,
                4.0247043059247E+00,
                3.6545905374482E+00,
                3.2792933294422E+00,
                4.3323909428471E+00,
                5.5460781049756E+00,
                4.7740932524300E+00,
            ],
        },
        RefCase {
            label: "Group2b_logg_5.0",
            teff: 5778.0,
            logg: 5.0,
            meta: 0.0,
            phot_x: 1.0,
            phot_z: Z_OVER_X_REF_GS98,
            r_cm: RSUN,
            expected: [
                1.6233837927675E+01,
                3.2097028578831E-07,
                6.0215053763441E-04,
                5.4419095524136E+00,
                4.0211092600660E+00,
                3.6525712832213E+00,
                3.2786851353429E+00,
                4.3286664694765E+00,
                5.5395103794726E+00,
                4.7780378683469E+00,
            ],
        },
        // Group 2c (`colors/test/test_output:135-178`)
        RefCase {
            label: "Group2c_Teff_3500",
            teff: 3500.0,
            logg: 4.0,
            meta: 0.0,
            phot_x: 1.0,
            phot_z: Z_OVER_X_REF_GS98,
            r_cm: RSUN,
            expected: [
                1.8411335119795E+01,
                4.3198229964375E-08,
                0.0,
                1.0036329716496E+01,
                6.1438344514899E+00,
                5.1123434709086E+00,
                4.0338857238481E+00,
                7.4278203656696E+00,
                1.1095018314829E+01,
                8.8377137013872E+00,
            ],
        },
        RefCase {
            label: "Group2c_Teff_6000",
            teff: 6000.0,
            logg: 4.0,
            meta: 0.0,
            phot_x: 1.0,
            phot_z: Z_OVER_X_REF_GS98,
            r_cm: RSUN,
            expected: [
                1.6069995716096E+01,
                3.7325163050257E-07,
                0.0,
                5.1670419609711E+00,
                3.8998094567456E+00,
                3.5659889365361E+00,
                3.2256242225345E+00,
                4.1800559165600E+00,
                5.2180945995774E+00,
                4.5848107914209E+00,
            ],
        },
        RefCase {
            label: "Group2c_Teff_10000",
            teff: 10000.0,
            logg: 4.0,
            meta: 0.0,
            phot_x: 1.0,
            phot_z: Z_OVER_X_REF_GS98,
            r_cm: RSUN,
            expected: [
                1.3851150550189E+01,
                2.8809769255562E-06,
                0.0,
                2.5700839979881E+00,
                2.6333104842699E+00,
                2.6554635119236E+00,
                2.6823297126398E+00,
                2.6143644145113E+00,
                2.4829162763772E+00,
                2.6004126821216E+00,
            ],
        },
        RefCase {
            label: "Group2c_Teff_20000",
            teff: 20000.0,
            logg: 4.0,
            meta: 0.0,
            phot_x: 1.0,
            phot_z: Z_OVER_X_REF_GS98,
            r_cm: RSUN,
            expected: [
                1.0839332915758E+01,
                4.6160109853289E-05,
                0.0,
                1.0828563761601E+00,
                1.5457706838243E+00,
                1.7510186431433E+00,
                1.9941516825941E+00,
                1.3843911085916E+00,
                1.0937345701750E-01,
                1.2754965059690E+00,
            ],
        },
    ];

    let cube = builtin_sed_cube();
    let ctx = ColorsContext::load(&default_kurucz_johnson_cfg(), None)
        .expect("offline Kurucz2003all + Johnson + Vega load should succeed");
    assert_eq!(ctx.column_names(), COLOR_COLUMN_NAMES);

    in_strict(|| {
        for case in &cases {
            let direct = cube
                .eval_colors(
                    case.teff,
                    case.logg,
                    case.meta,
                    case.r_cm,
                    D_10PC_CM,
                    MagSystem::Vega,
                    SedInterpMethod::Hermite,
                )
                .expect("eval_colors should succeed");
            let hist = ctx
                .eval_and_maybe_write_csv(
                    case.teff,
                    case.logg,
                    case.phot_x,
                    case.phot_z,
                    case.r_cm,
                    1,
                    false,
                )
                .expect("ColorsContext eval should succeed");

            assert_eq!(hist.filter_mags.len(), 7);
            let hist_arr = [
                hist.mag_bol,
                hist.flux_bol,
                hist.interp_rad,
                hist.filter_mags[0],
                hist.filter_mags[1],
                hist.filter_mags[2],
                hist.filter_mags[3],
                hist.filter_mags[4],
                hist.filter_mags[5],
                hist.filter_mags[6],
            ];

            for (idx, ((&d_val, &h_val), &exp_val)) in direct
                .iter()
                .zip(hist_arr.iter())
                .zip(case.expected.iter())
                .enumerate()
            {
                let col = COLOR_COLUMN_NAMES[idx];
                assert_rel_close(
                    d_val,
                    exp_val,
                    1.0e-12,
                    1.0e-15,
                    &format!("{}::eval_colors::{col}", case.label),
                );
                assert_rel_close(
                    h_val,
                    exp_val,
                    1.0e-12,
                    1.0e-15,
                    &format!("{}::ColorsContext::{col}", case.label),
                );
            }
        }
    });
}

/// Verifies MESA `26.4.1` Group 3 sampled solar SED (`21` wavelengths across
/// `147.2..=96200.0` AA) and Stefan-Boltzmann `wavelength_coverage_sanity`
/// (`colors/test/src/test_colors.f90:265-339`).
#[test]
fn test_mesa_26_4_1_group3_sed_sample_and_bolometric_coverage() {
    let solar_samples: [(f64, f64); 21] = [
        (1.4720000000000E+02, 2.1385116423739E-41),
        (5.1200000000000E+02, 1.2669686045439E-39),
        (1.0950000000000E+03, 5.5890114496366E-21),
        (1.6650000000000E+03, 1.3984854179012E-15),
        (2.2350000000000E+03, 2.2027299592265E-12),
        (2.8150000000000E+03, 5.2619396119207E-12),
        (3.8900000000000E+03, 2.6676232108131E-11),
        (5.0700000000000E+03, 4.7479893607680E-11),
        (6.2500000000000E+03, 3.7949934810183E-11),
        (7.4300000000000E+03, 2.9943284765423E-11),
        (8.6100000000000E+03, 2.2878420142827E-11),
        (9.7900000000000E+03, 1.7769276100082E-11),
        (1.2425000000000E+04, 1.0638798086088E-11),
        (1.5375000000000E+04, 6.5440892744616E-12),
        (2.0650000000000E+04, 2.4263712729167E-12),
        (2.6550000000000E+04, 9.5609055742312E-13),
        (3.2900000000000E+04, 4.2443985723283E-13),
        (4.4700000000000E+04, 1.2606836963661E-13),
        (5.6500000000000E+04, 5.2911476520239E-14),
        (7.2600000000000E+04, 2.0114488666486E-14),
        (9.6200000000000E+04, 6.6589153670475E-15),
    ];

    let cube = builtin_sed_cube();
    assert_rel_close(cube.wavelengths[0], 147.2, 1.0e-12, 0.0, "Wav_min_AA");
    assert_rel_close(
        *cube.wavelengths.last().expect("non-empty wavelengths"),
        1.6e6,
        1.0e-12,
        0.0,
        "Wav_max_AA",
    );

    in_strict(|| {
        let sed = cube.eval_sed(
            5778.0,
            4.44,
            0.0,
            RSUN,
            D_10PC_CM,
            SedInterpMethod::Hermite,
        );
        for &(target_w, expected_f) in &solar_samples {
            let pos = cube
                .wavelengths
                .iter()
                .position(|&w| (w - target_w).abs() < 1.0e-6)
                .unwrap_or_else(|| {
                    panic!("missing sample wavelength {target_w}")
                });
            assert_rel_close(
                sed[pos],
                expected_f,
                1.0e-12,
                0.0,
                &format!("Group3_SED@{target_w}"),
            );
        }

        let sb_cases = [
            (3500.0, 4.3198229964375E-08),
            (6000.0, 3.7325163050257E-07),
            (10000.0, 2.8809769255562E-06),
            (20000.0, 4.6160109853289E-05),
        ];
        for &(teff, expected_actual) in &sb_cases {
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
                .expect("Group3 eval_colors should succeed");
            assert_rel_close(
                out[1],
                expected_actual,
                1.0e-12,
                0.0,
                &format!("Group3_Flux_actual@{teff}"),
            );
            let sb_flux = BOLTZ_SIGMA * pow4(teff) * pow2(RSUN / D_10PC_CM);
            let rel_err = (out[1] - sb_flux).abs() / sb_flux;
            assert!(
                rel_err <= 5.0e-3,
                "Group3 Stefan-Boltzmann rel_err={rel_err:.3e} > 5e-3 at Teff={teff}"
            );
        }
    });
}

/// Verifies lookup-table `compute_interp_radius_lookup` parity with
/// `colors/private/bolometric.f90:111-189` and `colors/public/colors_def.f90:61-64`
/// on `Kurucz2003all` missing Eddington-limit grid combinations (`3,808`
/// populated rows vs `6,688` Cartesian cells).
#[test]
fn test_lookup_interp_radius_on_incomplete_kurucz2003all_grid() {
    let ctx = ColorsContext::load(&default_kurucz_johnson_cfg(), None)
        .expect("offline Kurucz2003all load should succeed");

    assert_eq!(ctx.lu_teff.len(), 3808);
    assert_eq!(ctx.lu_logg.len(), 3808);
    assert_eq!(ctx.lu_meta.len(), 3808);

    // Closest populated Kurucz2003all row to `(20000, 2.0, 0.0)` is
    // `(19000, 2.5, 0.0)`, and to `(15000, 1.5, 0.0)` is `(14000, 2.0, 0.0)`.
    let expected_gap = (pow2(1000.0 / 46500.0) + pow2(0.5 / 5.0)).sqrt();

    in_strict(|| {
        for &(teff, logg) in &[(20000.0, 2.0), (15000.0, 1.5)] {
            // Cartesian `compute_interp_radius` reports `0.0` at unpopulated
            // corner combinations; `ColorsContext` must report the true lookup
            // radius matching `bolometric.f90:111-189`.
            let cart_rad = compute_interp_radius(
                teff,
                logg,
                0.0,
                &ctx.cube.teff_grid,
                &ctx.cube.logg_grid,
                &ctx.cube.meta_grid,
            );
            assert_eq!(cart_rad.to_bits(), 0.0_f64.to_bits());

            let out = ctx
                .eval_and_maybe_write_csv(
                    teff,
                    logg,
                    1.0,
                    Z_OVER_X_REF_GS98,
                    RSUN,
                    1,
                    false,
                )
                .expect("eval at unpopulated corner should succeed");
            assert_rel_close(
                out.interp_rad,
                expected_gap,
                1.0e-13,
                1.0e-15,
                &format!("lookup Interp_rad at ({teff}, {logg}, 0.0)"),
            );
        }

        // At `(25000, 2.5, 0.0)`, `logg = 2.5` only goes up to `19000 K`, so
        // the nearest populated row is `(25000, 3.0, 0.0)` -> `0.5 / 5.0 = 0.1`.
        let out_25k = ctx
            .eval_and_maybe_write_csv(
                25000.0,
                2.5,
                1.0,
                Z_OVER_X_REF_GS98,
                RSUN,
                1,
                false,
            )
            .expect("eval at (25000, 2.5, 0.0) should succeed");
        assert_rel_close(
            out_25k.interp_rad,
            0.1,
            1.0e-13,
            1.0e-15,
            "lookup Interp_rad at (25000, 2.5, 0.0)",
        );
    });

    // True upstream dummy axes (`0.0` or `+-999.0`, `bolometric.f90:128-138`)
    // must be ignored, while constant non-dummy axes clamp range at `eps = 1e-12`
    // and mismatched lookup vectors return `InvalidArrayLength`.
    let dummy_rad = compute_interp_radius_lookup(
        4500.0,
        9.0,
        -1.0,
        &[4000.0, 5000.0],
        &[-999.0, -999.0],
        &[0.0, 0.0],
    )
    .expect("dummy-axis lookup radius should succeed");
    assert_rel_close(dummy_rad, 0.5, 1.0e-14, 0.0, "dummy-axis lookup radius");

    let y_probe = 3.0 + 1.0e-12;
    let const_rad = compute_interp_radius_lookup(
        4500.0,
        y_probe,
        0.0,
        &[4000.0, 5000.0],
        &[3.0, 3.0],
        &[0.0, 0.0],
    )
    .expect("constant non-dummy axis should clamp range at 1e-12");
    let expected_const_rad =
        (pow2(0.5) + pow2((y_probe - 3.0) / 1.0e-12)).sqrt();
    assert_rel_close(
        const_rad,
        expected_const_rad,
        1.0e-14,
        0.0,
        "eps-clamped non-dummy constant axis",
    );

    let err = compute_interp_radius_lookup(
        4000.0,
        3.0,
        0.0,
        &[4000.0, 5000.0],
        &[3.0],
        &[0.0, 0.0],
    )
    .expect_err("mismatched lookup arrays must error");
    assert!(matches!(err, ColorsError::InvalidArrayLength { .. }));
}

/// Verifies on-disk incomplete atmosphere `build_grid_to_lu_map` nearest-row
/// fill and `1e-10` axis dedup (`colors/private/colors_utils.f90:714,748-791`),
/// headerless Vega loading, `.` instrument index resolution, `AB`/`ST`
/// magnitudes, zero-flux `f64::MAX` / `t_eff < 0` `-1.0` sentinels
/// (`synthetic.f90:110-120`, `colors_history.f90:119-129`), and `ES14.6` CSV
/// output without `$MESA_DIR`.
#[test]
fn test_incomplete_disk_atm_dot_instrument_vega_ab_st_csv_and_sentinels() {
    let temp = TempDirGuard::new("disk_incomplete");
    let atm_dir = temp.path().join("atm");
    let inst_dir = temp.path().join("my_inst");
    fs::create_dir_all(&atm_dir).expect("create atm dir");
    fs::create_dir_all(&inst_dir).expect("create inst dir");

    // 3 populated corners out of 2x2x1 (`(5000,1)`, `(5000,2)`, `(6000,1)`),
    // leaving `(6000, 2.0, 0.0)` unpopulated so L1 nearest-row maps it to
    // `(6000, 1.0, 0.0)`. Also exercise `< 1e-10` Teff dedup (`5000 + 5e-11`).
    fs::write(
        atm_dir.join("lookup_table.csv"),
        "#file_name,teff,logg,meta\n\
         sed_5k_g1.txt,5000.0,1.0,0.0\n\
         sed_5k_g2.txt,5000.00000000005,2.0,0.0\n\
         sed_6k_g1.txt,6000.0,1.0,0.0\n",
    )
    .expect("write lookup_table.csv");

    let write_flat_sed = |name: &str, flux: f64| {
        let body = format!(
            "3000.0 {flux:.6e}\n\
             4000.0 {flux:.6e}\n\
             4500.0 {flux:.6e}\n\
             5500.0 {flux:.6e}\n\
             6000.0 {flux:.6e}\n\
             7000.0 0.0\n\
             8000.0 0.0\n\
             8500.0 0.0\n\
             9000.0 0.0\n"
        );
        fs::write(atm_dir.join(name), body).expect("write sed file");
    };
    write_flat_sed("sed_5k_g1.txt", 1.0);
    write_flat_sed("sed_5k_g2.txt", 2.0);
    write_flat_sed("sed_6k_g1.txt", 4.0);

    // Vega spectrum with `#` comment + numeric rows covering both bands
    // (verifies `load_vega_file` filters `#` comments before header check).
    fs::write(
        temp.path().join("vega.dat"),
        "# wavelength(A) flux\n\
         3000.0 2.0e-9\n\
         5000.0 2.0e-9\n\
         7000.0 2.0e-9\n\
         8500.0 2.0e-9\n\
         9500.0 2.0e-9\n",
    )
    .expect("write vega.dat");

    // Instrument directory named `my_inst` with index file `my_inst`.
    // `top.dat` covers `4000..=6000` AA (inside SED), while `red.dat` covers
    // `8000..=9000` AA (outside SED -> `syn_flux = 0.0 -> f64::MAX`).
    fs::write(inst_dir.join("my_inst"), "# filters\ntop.dat\nred.dat\n")
        .expect("write instrument index");
    fs::write(
        inst_dir.join("top.dat"),
        "Wavelength Transmission\n\
         4000.0 0.0\n\
         4500.0 1.0\n\
         5500.0 1.0\n\
         6000.0 0.0\n",
    )
    .expect("write top.dat");
    fs::write(
        inst_dir.join("red.dat"),
        "8000.0 0.0\n8500.0 1.0\n9000.0 0.0\n",
    )
    .expect("write red.dat");

    let cfg_st = ColorsLoadConfig {
        instrument: ".",
        stellar_atm: "../atm",
        vega_sed: "../vega.dat",
        mag_system: "ST",
        distance: 1.0,
        z_over_x_ref: Z_OVER_X_REF_GS98,
        make_csv: true,
        sed_per_model: true,
        colors_results_directory: "../sed_out",
    };
    let ctx_st = ColorsContext::load(&cfg_st, Some(&inst_dir))
        .expect("incomplete on-disk grid + `.` instrument must load");

    assert_eq!(ctx_st.cube.teff_grid.len(), 2);
    assert_eq!(ctx_st.cube.logg_grid.len(), 2);
    assert_eq!(ctx_st.cube.meta_grid.len(), 1);
    assert_eq!(ctx_st.lu_teff.len(), 3);
    assert_eq!(
        ctx_st.column_names(),
        ["Mag_bol", "Flux_bol", "Interp_rad", "top", "red"]
    );

    // Verify L1 nearest-row mapping placed `(6000, 1.0)` SED (`flux = 4.0`) at
    // missing corner `(ix=1, iy=1, iz=0)`.
    let corner_flux = ctx_st.cube.sed_at(1, 1, 0);
    assert_rel_close(corner_flux[0], 4.0, 1.0e-15, 0.0, "L1 corner fill");

    in_strict(|| {
        // Querying `(5000.0, 1.0, r=1.0, d=1.0)` gives exact `F_lambda = 1.0`:
        // `syn_flux = 1.0`, `m_ST = -2.5 * log10(1.0 / 3.63e-9)`, and out-of-band
        // `red` returns `f64::MAX` (`synthetic.f90:110-120`).
        let out_st = ctx_st
            .eval_and_maybe_write_csv(
                5000.0,
                1.0,
                1.0,
                Z_OVER_X_REF_GS98,
                1.0,
                42,
                true,
            )
            .expect("eval ST with CSV should succeed");
        let expected_st_mag = -2.5 * intrinsic_log10(1.0 / 3.63e-9);
        assert_rel_close(
            out_st.filter_mags[0],
            expected_st_mag,
            1.0e-13,
            0.0,
            "ST magnitude on flat unit SED",
        );
        assert_eq!(out_st.filter_mags[1].to_bits(), f64::MAX.to_bits());

        // Lookup radius at missing corner `(6000, 1.75, 0.0)`: row `(6000, 1.0)`
        // is distance `0.75` away (closer than `(5000 + 5e-11, 2.0)` at `~1.03`),
        // verifying row-wise minimum on incomplete lookup tables.
        let out_miss = ctx_st
            .eval_and_maybe_write_csv(
                6000.0,
                1.75,
                1.0,
                Z_OVER_X_REF_GS98,
                1.0,
                43,
                false,
            )
            .expect("eval missing corner should succeed");
        assert_rel_close(
            out_miss.interp_rad,
            0.75,
            1.0e-14,
            0.0,
            "missing corner lookup Interp_rad",
        );

        // `t_eff < 0.0` sentinel returns `-1.0` for all filter bands
        // (`colors_history.f90:119-129`).
        let out_neg_t = ctx_st
            .eval_and_maybe_write_csv(
                -10.0,
                1.0,
                1.0,
                Z_OVER_X_REF_GS98,
                1.0,
                44,
                false,
            )
            .expect("eval negative Teff should succeed");
        assert_eq!(out_neg_t.filter_mags[0].to_bits(), (-1.0_f64).to_bits());
        assert_eq!(out_neg_t.filter_mags[1].to_bits(), (-1.0_f64).to_bits());

        // Check AB magnitude parity on the same disk setup.
        let cfg_ab = ColorsLoadConfig {
            mag_system: "AB",
            make_csv: false,
            ..cfg_st
        };
        let ctx_ab = ColorsContext::load(&cfg_ab, Some(&inst_dir))
            .expect("AB disk load should succeed");
        let out_ab = ctx_ab
            .eval_and_maybe_write_csv(
                5000.0,
                1.0,
                1.0,
                Z_OVER_X_REF_GS98,
                1.0,
                1,
                false,
            )
            .expect("eval AB should succeed");
        let ab_zp = compute_ab_zero_point(
            &ctx_ab.filters[0].wavelengths,
            &ctx_ab.filters[0].transmission,
        )
        .expect("compute_ab_zero_point");
        let expected_ab_mag = -2.5 * intrinsic_log10(1.0 / ab_zp);
        assert_rel_close(
            out_ab.filter_mags[0],
            expected_ab_mag,
            1.0e-13,
            0.0,
            "AB magnitude on flat unit SED",
        );
    });

    // Verify `sed_per_model = true` wrote `top_SED_00000042.csv` in `ES14.6`.
    let csv_path = temp.path().join("sed_out/top_SED_00000042.csv");
    let csv_text = fs::read_to_string(&csv_path).expect("read top SED csv");
    let mut lines = csv_text.lines();
    assert_eq!(
        lines.next(),
        Some(
            "wavelengths,fluxes,convolved_flux,filter_wavelengths,filter_trans"
        )
    );
    let first_data = lines.next().expect("csv must contain data rows");
    let cols: Vec<&str> = first_data.split(',').collect();
    assert_eq!(cols.len(), 5);
    for col in cols {
        assert_eq!(col.len(), 14, "ES14.6 field width mismatch: {col:?}");
        assert!(col.contains("E+03") || col.contains("E+00"));
    }
}

/// Verifies `SedCube::eval_sed` `Hermite` nearest-neighbor fallback whenever
/// `nx < 2 || ny < 2 || nz < 2` (`colors/private/hermite_interp.f90:203-213`).
#[test]
fn test_hermite_sub2_axis_falls_back_to_nearest_point() {
    let make_1x = |nt: usize, ng: usize, nm: usize| {
        let t_grid: Vec<f64> = (0..nt)
            .map(|i| 3000.0 + f64::from(u32::try_from(i).unwrap()) * 1000.0)
            .collect();
        let g_grid: Vec<f64> = (0..ng)
            .map(|i| 2.0 + f64::from(u32::try_from(i).unwrap()))
            .collect();
        let m_grid: Vec<f64> = (0..nm)
            .map(|i| -1.0 + f64::from(u32::try_from(i).unwrap()))
            .collect();
        let mut flux = Vec::new();
        for it in 0..nt {
            for ig in 0..ng {
                for im in 0..nm {
                    let val = 100.0 * f64::from(u32::try_from(it + 1).unwrap())
                        + 10.0 * f64::from(u32::try_from(ig).unwrap())
                        + f64::from(u32::try_from(im).unwrap());
                    flux.extend_from_slice(&[val, val, val]);
                }
            }
        }
        SedCube::from_dense(
            t_grid,
            g_grid,
            m_grid,
            vec![4000.0, 5000.0, 6000.0],
            flux,
        )
        .expect("construct sub-2 cube")
    };

    // `3 x 1 x 1` (`bbody`-like): `teff = 3800.0` is nearest `teff = 4000.0` (`200.0`).
    let cube_t = make_1x(3, 1, 1);
    let sed_t =
        cube_t.eval_sed(3800.0, 4.0, 0.0, 1.0, 1.0, SedInterpMethod::Hermite);
    for &v in &sed_t {
        assert_eq!(v.to_bits(), 200.0_f64.to_bits());
    }

    // `1 x 3 x 1`: `log_g = 3.6` is nearest `log_g = 4.0` (`120.0`).
    let cube_g = make_1x(1, 3, 1);
    let sed_g =
        cube_g.eval_sed(4000.0, 3.6, 0.0, 1.0, 1.0, SedInterpMethod::Hermite);
    for &v in &sed_g {
        assert_eq!(v.to_bits(), 120.0_f64.to_bits());
    }

    // `1 x 1 x 3`: `meta = -0.2` is nearest `meta = 0.0` (`101.0`).
    let cube_m = make_1x(1, 1, 3);
    let sed_m =
        cube_m.eval_sed(4000.0, 2.0, -0.2, 1.0, 1.0, SedInterpMethod::Hermite);
    for &v in &sed_m {
        assert_eq!(v.to_bits(), 101.0_f64.to_bits());
    }
}

/// Verifies `SedCube::eval_sed` `Knn` exact operation order, normalized weight
/// pre-blend before dilution, and `dist == 0.0 -> 1.0e-10`-only clamping
/// (`colors/private/knn_interp.f90:63,91-104,183-276`).
#[test]
fn test_knn_exact_operation_order_and_zero_distance_only() {
    let cube = SedCube::from_dense(
        vec![4000.0, 5000.0, 6000.0],
        vec![2.0, 3.0, 4.0],
        vec![-1.0, 0.0],
        vec![4000.0, 5000.0, 6000.0],
        (0..18)
            .flat_map(|idx| {
                let base = 1.0 + f64::from(u32::try_from(idx).unwrap());
                [base, 2.0 * base, 3.0 * base]
            })
            .collect(),
    )
    .expect("construct 3x3x2 KNN test cube");

    // Verbatim translation of `knn_interp.f90:63,91-104,183-276` for bit-exactness.
    let knn_ref = |teff: f64, log_g: f64, meta: f64, r: f64, d: f64| {
        let t_min = 4000.0_f64;
        let t_span = 2000.0_f64;
        let g_min = 2.0_f64;
        let g_span = 2.0_f64;
        let m_min = -1.0_f64;
        let m_span = 1.0_f64;
        let nt = (teff - t_min) / t_span;
        let ng = (log_g - g_min) / g_span;
        let nm = (meta - m_min) / m_span;
        let mut best_d2 = [f64::MAX; 4];
        let mut best_idx = [(0usize, 0usize, 0usize); 4];
        for (it, &tv) in cube.teff_grid.iter().enumerate() {
            let dt = pow2((tv - t_min) / t_span - nt);
            for (ig, &gv) in cube.logg_grid.iter().enumerate() {
                let dt_dg = dt + pow2((gv - g_min) / g_span - ng);
                for (im, &mv) in cube.meta_grid.iter().enumerate() {
                    let d2 = dt_dg + pow2((mv - m_min) / m_span - nm);
                    for k in 0..4 {
                        if d2 < best_d2[k] {
                            for s in ((k + 1)..4).rev() {
                                best_d2[s] = best_d2[s - 1];
                                best_idx[s] = best_idx[s - 1];
                            }
                            best_d2[k] = d2;
                            best_idx[k] = (it, ig, im);
                            break;
                        }
                    }
                }
            }
        }
        let inv_w = best_d2.map(|d2| {
            let dist = d2.sqrt();
            if dist == 0.0 {
                1.0 / 1.0e-10
            } else {
                1.0 / dist
            }
        });
        let w_sum: f64 = inv_w.iter().sum();
        let weights = inv_w.map(|w| w / w_sum);
        let mut out = vec![0.0_f64; 3];
        for (&(it, ig, im), &wk) in best_idx.iter().zip(weights.iter()) {
            let sed = cube.sed_at(it, ig, im);
            for (dst, &src) in out.iter_mut().zip(sed.iter()) {
                *dst += wk * src;
            }
        }
        let scale = (r / d) * (r / d);
        for v in &mut out {
            *v *= scale;
        }
        out
    };

    // Off-grid point + exact node (`dist == 0.0 -> 1.0e-10`) + ultra-close point
    // (`dist = 1.0e-12`, NOT clamped to `1.0e-10`).
    for &(teff, log_g, meta) in &[
        (4630.0, 3.27, -0.35),
        (5000.0, 3.0, 0.0),
        (5000.0 + 2.0e-9, 3.0, 0.0),
    ] {
        let got = cube.eval_sed(
            teff,
            log_g,
            meta,
            RSUN,
            D_10PC_CM,
            SedInterpMethod::Knn,
        );
        let exp_sed = knn_ref(teff, log_g, meta, RSUN, D_10PC_CM);
        for (lam, (&g_val, &e_val)) in
            got.iter().zip(exp_sed.iter()).enumerate()
        {
            assert_eq!(
                g_val.to_bits(),
                e_val.to_bits(),
                "KNN 0-ULP mismatch at ({teff}, {log_g}, {meta}) lam={lam}"
            );
        }
    }
}

/// Verifies `Linear` log-fallback (`linear_interp.f90:215-258`), `calculate_bolometric_phot`
/// negative-flux clamping (`bolometric.f90:81`), `colors_from_sed_generic` zero-flux
/// `f64::MAX` (`synthetic.f90:110-120`), and `SedCube::from_mesa_flux_cube`
/// rejecting negative/overflow dimensions (`colors_utils.f90:597`).
#[test]
fn test_linear_log_fallback_bolometric_cleaning_zero_flux_and_binary_validation(
) {
    // `2 x 2 x 2` cube: wavelength 0 is all `<= 0.0` (stays floored at `1e-10`),
    // wavelength 1 interpolates geometrically in log space (`[1, 100]` at mid `y` -> `10`).
    let mut flux = Vec::with_capacity(2 * 2 * 2 * 3);
    for _it in 0..2 {
        for ig in 0..2 {
            for _im in 0..2 {
                let f1 = if ig == 0 { 1.0 } else { 100.0 };
                flux.extend_from_slice(&[0.0, f1, 0.0]);
            }
        }
    }
    let cube = SedCube::from_dense(
        vec![4000.0, 5000.0],
        vec![2.0, 4.0],
        vec![-1.0, 0.0],
        vec![4000.0, 5000.0, 6000.0],
        flux,
    )
    .expect("construct linear test cube");

    in_strict(|| {
        let sed =
            cube.eval_sed(4500.0, 3.0, -0.5, 1.0, 1.0, SedInterpMethod::Linear);
        assert_eq!(sed[0].to_bits(), 1.0e-10_f64.to_bits());
        let expected_geo = intrinsic_exp(
            0.5 * intrinsic_log(1.0) + 0.5 * intrinsic_log(100.0),
        );
        assert_eq!(sed[1].to_bits(), expected_geo.to_bits());

        // All-zero SED must return `Mag_bol = 99.0` and `f64::MAX` for all
        // Johnson passbands (`bolometric.f90:flux_to_magnitude`, `synthetic.f90:110-120`).
        let zero_cube = SedCube::from_dense(
            vec![4000.0, 5000.0],
            vec![2.0, 4.0],
            vec![-1.0, 0.0],
            vec![4000.0, 5000.0, 6000.0],
            vec![0.0; 24],
        )
        .expect("construct zero cube");
        let z_colors = zero_cube
            .eval_colors(
                4500.0,
                3.0,
                -0.5,
                1.0,
                1.0,
                MagSystem::Vega,
                SedInterpMethod::Hermite,
            )
            .expect("eval_colors on zero SED should return sentinels");
        assert_eq!(z_colors[0].to_bits(), 99.0_f64.to_bits());
        assert_eq!(z_colors[1].to_bits(), 0.0_f64.to_bits());
        for &band_mag in &z_colors[3..] {
            assert_eq!(band_mag.to_bits(), f64::MAX.to_bits());
        }

        // `calculate_bolometric_phot` clamps negative fluxes over `0..(n - 1)`
        // to zero while keeping the final node `n - 1` (`bolometric.f90:81`).
        let (mag_c, flux_c) = calculate_bolometric_phot(
            &[4000.0, 5000.0, 6000.0],
            &[-5.0, -2.0, 3.0],
        )
        .expect("bolometric cleaning should succeed");
        let (_, flux_expected) = calculate_bolometric_phot(
            &[4000.0, 5000.0, 6000.0],
            &[0.0, 0.0, 3.0],
        )
        .expect("reference bolometric");
        assert_eq!(flux_c.to_bits(), flux_expected.to_bits());
        assert_rel_close(
            mag_c,
            -2.5 * intrinsic_log10(flux_expected),
            1.0e-14,
            0.0,
            "cleaned Mag_bol",
        );
    });

    // Malformed `flux_cube.bin` headers (negative i32, zero, overflow) must
    // return `BinaryParseError` instead of panicking (`colors_utils.f90:597`).
    let temp = TempDirGuard::new("bad_flux_cube");
    for (idx, dims) in
        [[-4_i32, 2, 2, 2], [2, 0, 2, 2], [i32::MAX, i32::MAX, 2, 2]]
            .iter()
            .enumerate()
    {
        let p = temp.path().join(format!("bad_{idx}.bin"));
        let mut bytes = Vec::with_capacity(16);
        for d in dims {
            bytes.extend_from_slice(&d.to_le_bytes());
        }
        fs::write(&p, bytes).expect("write malformed flux cube");
        let err = SedCube::from_mesa_flux_cube(&p)
            .expect_err("malformed flux_cube.bin header must error");
        assert!(matches!(err, ColorsError::BinaryParseError(_)));
    }
}

/// Verifies built-in Gaia DR3 (`G, Gbp, Grp, Grvs`) zero-`$MESA_DIR` loading,
/// zero-point cross-checks (`Vega`, `AB`, `ST`), and solar evaluation parity
/// against `calculate_synthetic`.
#[test]
fn test_builtin_gaia_filters_zero_mesa_dir() {
    let cfg = ColorsLoadConfig {
        instrument: "data/colors_data/filters/GAIA/GAIA",
        stellar_atm: "data/colors_data/stellar_models/Kurucz2003all",
        vega_sed: "data/colors_data/stellar_models/vega_flam.csv",
        mag_system: "Vega",
        distance: D_10PC_CM,
        z_over_x_ref: Z_OVER_X_REF_GS98,
        make_csv: false,
        sed_per_model: false,
        colors_results_directory: "SED",
    };
    let ctx = ColorsContext::load(&cfg, None)
        .expect("offline Gaia filter set must load cleanly");
    assert_eq!(
        ctx.column_names(),
        [
            "Mag_bol",
            "Flux_bol",
            "Interp_rad",
            "G",
            "Gbp",
            "Grp",
            "Grvs"
        ]
    );

    let expected_vega_zps = [
        4.696335300553758e-12,
        4.033333663969422e-9,
        1.5424820614450155e-11,
        8.867339794186975e-10,
    ];

    in_strict(|| {
        let sed = ctx.cube.eval_sed(
            5778.0,
            4.44,
            0.0,
            RSUN,
            D_10PC_CM,
            SedInterpMethod::Hermite,
        );
        let hist = ctx
            .eval_and_maybe_write_csv(
                5778.0,
                4.44,
                1.0,
                Z_OVER_X_REF_GS98,
                RSUN,
                1,
                false,
            )
            .expect("solar Gaia eval should succeed");

        assert_eq!(hist.filter_mags.len(), 4);
        for (idx, (filt, &exp_zp)) in
            ctx.filters.iter().zip(expected_vega_zps.iter()).enumerate()
        {
            assert_rel_close(
                filt.vega_zero_point,
                exp_zp,
                1.0e-12,
                0.0,
                &format!("Gaia Vega ZP {}", filt.name),
            );
            let ab_zp =
                compute_ab_zero_point(&filt.wavelengths, &filt.transmission)
                    .expect("Gaia AB ZP");
            let st_zp =
                compute_st_zero_point(&filt.wavelengths, &filt.transmission)
                    .expect("Gaia ST ZP");
            assert_rel_close(
                filt.ab_zero_point,
                ab_zp,
                1.0e-14,
                0.0,
                &format!("Gaia AB ZP {}", filt.name),
            );
            assert_rel_close(
                filt.st_zero_point,
                st_zp,
                1.0e-14,
                0.0,
                &format!("Gaia ST ZP {}", filt.name),
            );

            let direct_mag = calculate_synthetic(
                &ctx.cube.wavelengths,
                &sed,
                &filt.wavelengths,
                &filt.transmission,
                filt.vega_zero_point,
            )
            .expect("direct Gaia calculate_synthetic");
            assert_rel_close(
                hist.filter_mags[idx],
                direct_mag,
                1.0e-13,
                0.0,
                &format!("Gaia solar {}", filt.name),
            );
        }
    });

    // Also sanity-check `compute_vega_zero_point` directly on synthetic filter.
    let zp = compute_vega_zero_point(
        &[4000.0, 5000.0, 6000.0],
        &[2.0e-9, 2.0e-9, 2.0e-9],
        &[4000.0, 5000.0, 6000.0],
        &[0.0, 1.0, 0.0],
    )
    .expect("compute_vega_zero_point");
    assert_rel_close(zp, 2.0e-9, 1.0e-14, 0.0, "constant Vega ZP");
}
