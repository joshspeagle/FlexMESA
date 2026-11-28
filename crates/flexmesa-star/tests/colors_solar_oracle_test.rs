// SPDX-License-Identifier: GPL-3.0-or-later
//! MESA `26.4.1` Group 1 Solar Johnson photometry oracle test (`#454`, `#601`).
//!
//! Compares `evaluate_star_history_row` under `NumericsMode::Strict` against
//! upstream IEEE-754 hex captures from `colors/test/src/test_colors.f90:183-197`
//! (`mesa-module-hex-v1` `colors/exact_tested_values.tsv` records `6..15`).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use flexmesa_colors::Z_OVER_X_REF_GS98;
use flexmesa_const::{L_SUN, R_SUN, STANDARD_CGRAV};
use flexmesa_math::{with_numerics_mode, NumericsMode};
use flexmesa_star::controls::ColorsControls;
use flexmesa_star::mesa_history::{
    evaluate_star_history_row, resolve_star_history_columns,
};
use flexmesa_star::{StarControls, StarJobControls, StarModel};

static TMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

struct TempDirGuard(PathBuf);

impl TempDirGuard {
    fn new(tag: &str) -> Self {
        let id = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "flexmesa_colors_oracle_test_{tag}_{}_{id}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("create temp test dir");
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

fn solar_surface_model(colors: ColorsControls) -> StarModel {
    // Inputs follow `colors/test/src/test_colors.f90:59-62`:
    // line 59 (`Teff = 5778`), line 60 (`logg = 4.44`),
    // line 61 (`Fe_H = 0`), line 62 (`R = Rsun`).
    let ctrl = StarControls {
        initial_mass: 1.0,
        initial_z: 0.02,
        colors,
        ..StarControls::default()
    };
    let cat = flexmesa_chem::NuclideCatalog::builtin_ref();
    let h1 = cat.get_nuclide_index("h1").expect("h1 id");
    let he4 = cat.get_nuclide_index("he4").expect("he4 id");
    let mut model =
        StarModel::new(2, vec![h1, he4], StarJobControls::default(), ctrl);
    model.model_number = 7;
    let r0 = R_SUN;
    let g0 = flexmesa_math::pow(10.0, 4.44);
    let m0 = g0 * r0 * r0 / STANDARD_CGRAV;
    model.mstar = m0;
    model.t = vec![5778.0, 6200.0];
    model.r = vec![r0, 0.99 * r0];
    model.m = vec![m0, 0.99 * m0];
    model.cgrav = vec![STANDARD_CGRAV, STANDARD_CGRAV];
    model.grav = vec![g0, g0];
    model.dq = vec![0.01, 0.99];
    model.opacity = vec![1.0, 1.0];
    model.rho = vec![1.0e-7, 1.0e-6];
    model.lum = vec![L_SUN, L_SUN];
    // Choose X and Z so (Z / X) == Z_OVER_X_REF_GS98 -> [M/H] == 0.0 dex.
    let x = 0.70;
    let z = x * Z_OVER_X_REF_GS98;
    let y = 1.0 - x - z;
    model.xa = vec![x, y, x, y];
    model
}

fn signed_magnitude(bits: u64) -> i128 {
    if (bits >> 63) != 0 {
        -i128::from(bits ^ (1_u64 << 63))
    } else {
        i128::from(bits)
    }
}

fn ulp_distance(actual: f64, expected_bits: u64) -> u64 {
    let expected = f64::from_bits(expected_bits);
    if actual == 0.0 && expected == 0.0 {
        return 0;
    }
    if actual.is_nan() || expected.is_nan() {
        return if actual.is_nan() && expected.is_nan() {
            0
        } else {
            u64::MAX
        };
    }
    let a = signed_magnitude(actual.to_bits());
    let b = signed_magnitude(expected_bits);
    u64::try_from((a - b).unsigned_abs()).unwrap_or(u64::MAX)
}

#[test]
fn test_builtin_solar_johnson_history_columns() {
    with_numerics_mode(NumericsMode::Strict, || {
        let cc = ColorsControls {
            use_colors: true,
            ..ColorsControls::default()
        };
        let mut model = solar_surface_model(cc);
        // `0x40da_e592_5ec1_42ce` is a FlexMESA-chosen literal (not an upstream
        // value), equal to the correctly rounded `f64` for `10^(4.44 as f64)`
        // (7 ULP above decimal `10^4.44`). `solar_surface_model`'s strict-mode
        // `pow(10.0, 4.44)` returns `0x40da_e592_5ec1_42c8`, whose `log10`
        // evaluates to `4.44 - 1 ULP` rather than exact `4.44_f64`
        // (`colors/test/src/test_colors.f90:60`; `test_colors.f90:59-62` sets
        // `Teff = 5778`, `logg = 4.44`, `Fe_H = 0`, `R = Rsun`), so we keep
        // this `g0` override to feed exact `4.44_f64` into the table lookup.
        let g0 = f64::from_bits(0x40da_e592_5ec1_42ce);
        assert_eq!(model.grav[0].to_bits(), 0x40da_e592_5ec1_42c8);
        assert_ne!(
            flexmesa_math::log10(model.grav[0]).to_bits(),
            4.44_f64.to_bits()
        );
        assert_eq!(flexmesa_math::log10(g0).to_bits(), 4.44_f64.to_bits());
        // Override `solar_surface_model`'s `pow(10.0, 4.44)` gravity with `g0`.
        model.grav = vec![g0, g0];
        let dir = TempDirGuard::new("builtin_solar");
        fs::write(dir.path().join("history_columns.list"), "model_number\n")
            .expect("write list");
        let specs = resolve_star_history_columns(&model, Some(dir.path()))
            .expect("resolve default Johnson");
        let row =
            evaluate_star_history_row(&model, &specs).expect("history row");
        let names: Vec<&str> = row.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "model_number",
                "Mag_bol",
                "Flux_bol",
                "Interp_rad",
                "B",
                "I",
                "J",
                "M",
                "R",
                "U",
                "V",
            ]
        );
        // Upstream MESA `26.4.1` Group 1 Solar outputs
        // (`colors/test/src/test_colors.f90:183-197`, `mesa-module-hex-v1`
        // `colors/exact_tested_values.tsv` records `6..15`).
        // In `Strict` mode the 9 non-`J` columns match MESA at 0 ULP, whereas
        // `J` checks a pinned 1-ULP difference vs MESA (`0x400d_3c59_0bb4_5389`
        // vs MESA `0x400d_3c59_0bb4_538a`, tracked in `#198` around
        // `colors/private/synthetic.f90:111`), not agreement with MESA.
        let pinned_j_diff_bits_198: u64 = 0x400d_3c59_0bb4_5389;
        let expected = [
            ("Mag_bol", 0x4030_3bd9_4058_0275_u64),
            ("Flux_bol", 0x3e95_8a80_b289_f0bc_u64),
            ("Interp_rad", 0x3f88_9b5f_2f5a_6e1e_u64),
            ("B", 0x4015_bb38_2edc_bf58_u64),
            ("I", 0x4010_18f2_ad65_6265_u64),
            ("J", 0x400d_3c59_0bb4_538a_u64),
            ("M", 0x400a_3c38_2e3e_8e42_u64),
            ("R", 0x4011_5437_5640_16fe_u64),
            ("U", 0x4016_231d_a7c6_9e08_u64),
            ("V", 0x4013_1ba0_f14c_7a97_u64),
        ];
        let mut mismatches = Vec::new();
        for (col, mesa_bits) in expected {
            let val = row
                .iter()
                .find(|(k, _)| k == col)
                .expect("missing col")
                .1
                .as_f64();
            let got_bits = val.to_bits();
            if col == "J" {
                // Pinned 1-ULP difference vs MESA tracked in `#198` (not an
                // agreement check with MESA).
                let mesa_ulp = ulp_distance(val, mesa_bits);
                if got_bits != pinned_j_diff_bits_198 || mesa_ulp != 1 {
                    mismatches.push(format!(
                        "{col}: got 0x{got_bits:016x} ({val:.17e}), \
                         expected pinned #198 diff bits \
                         0x{pinned_j_diff_bits_198:016x} \
                         (mesa_ulp={mesa_ulp} vs MESA 0x{mesa_bits:016x})"
                    ));
                }
            } else if got_bits != mesa_bits {
                mismatches.push(format!(
                    "{col}: got 0x{got_bits:016x} ({val:.17e}), \
                     expected 0x{mesa_bits:016x} ({:.17e}), ulp={}",
                    f64::from_bits(mesa_bits),
                    ulp_distance(val, mesa_bits),
                ));
            }
        }
        assert!(
            mismatches.is_empty(),
            "solar johnson history hex mismatches:\n{}",
            mismatches.join("\n")
        );
    });
}
