// SPDX-License-Identifier: GPL-3.0-or-later
//! Bit-exact IEEE-754 `f64` regression tests for `eval_colors` and all `234`
//! rows of `colors/exact_tested_values.tsv` against MESA `26.4.1`
//! (`colors/test/src/test_colors.f90:1-340`).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use std::path::{Path, PathBuf};

use flexmesa_colors::{
    builtin_sed_cube, MagSystem, SedInterpMethod, COLOR_COLUMN_NAMES,
    D_10PC_CM, NUM_COLORS_OUT,
};
use flexmesa_const::{BOLTZ_SIGMA, RSUN};
use flexmesa_math::{pow2, pow4, with_numerics_mode, NumericsMode};

#[derive(Debug, Clone)]
struct HexRow {
    record_idx: usize,
    val_idx: usize,
    src_loc: String,
    label: String,
    raw_bits: u64,
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

fn parse_hex_bits(raw: &str) -> u64 {
    let trimmed = raw
        .trim()
        .strip_prefix("0x")
        .or_else(|| raw.trim().strip_prefix("0X"))
        .unwrap_or_else(|| {
            panic!("expected 0x-prefixed hex bits, got {raw:?}")
        });
    u64::from_str_radix(trimmed, 16)
        .unwrap_or_else(|e| panic!("invalid hex bits {raw:?}: {e}"))
}

fn require_zero_skips_enabled() -> bool {
    let Ok(val) = std::env::var("FLEXMESA_REQUIRE_ZERO_SKIPS") else {
        return true;
    };
    !matches!(
        val.trim().to_ascii_lowercase().as_str(),
        "0" | "false" | "no" | "off"
    )
}

fn resolve_module_hex_tsv() -> Option<PathBuf> {
    if let Some(explicit) =
        std::env::var_os("FLEXMESA_MODULE_HEX_DIR").filter(|v| !v.is_empty())
    {
        let path = PathBuf::from(explicit)
            .join("colors")
            .join("exact_tested_values.tsv");
        if path.is_file() {
            return Some(path);
        }
        assert!(
            !require_zero_skips_enabled(),
            "missing required MESA 26.4.1 module-hex file `{}` (set \
             FLEXMESA_MODULE_HEX_DIR, run `flexmesa data fetch \
             --include-test-inputs`, or set FLEXMESA_REQUIRE_ZERO_SKIPS=0 \
             locally)",
            path.display()
        );
        eprintln!(
            "NOT RUN (FLEXMESA_REQUIRE_ZERO_SKIPS opted out): missing {}; \
             the calling test checks nothing",
            path.display()
        );
        return None;
    }

    let mut candidates = Vec::new();
    if let Some(data_dir) =
        std::env::var_os("FLEXMESA_DATA_DIR").filter(|v| !v.is_empty())
    {
        let p = PathBuf::from(data_dir);
        if let Some(parent) = p.parent() {
            candidates.push(parent.join("mesa_module_hex"));
        }
    }
    if let Some(xdg) =
        std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty())
    {
        candidates
            .push(PathBuf::from(xdg).join("flexmesa").join("mesa_module_hex"));
    }
    if let Some(home) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
        candidates.push(
            PathBuf::from(home)
                .join(".cache")
                .join("flexmesa")
                .join("mesa_module_hex"),
        );
    }
    if let Some(local) =
        std::env::var_os("LOCALAPPDATA").filter(|v| !v.is_empty())
    {
        candidates.push(
            PathBuf::from(local)
                .join("flexmesa")
                .join("mesa_module_hex"),
        );
    }
    candidates.push(PathBuf::from(".mesa_module_hex"));

    for root in &candidates {
        let path = root.join("colors").join("exact_tested_values.tsv");
        if path.is_file() {
            return Some(path);
        }
    }

    let primary = candidates[0].join("colors").join("exact_tested_values.tsv");
    assert!(
        !require_zero_skips_enabled(),
        "missing required MESA 26.4.1 module-hex file `{}` (set \
         FLEXMESA_MODULE_HEX_DIR, run `flexmesa data fetch \
         --include-test-inputs`, or set FLEXMESA_REQUIRE_ZERO_SKIPS=0 locally)",
        primary.display()
    );
    eprintln!(
        "NOT RUN (FLEXMESA_REQUIRE_ZERO_SKIPS opted out): missing {}; \
         the calling test checks nothing",
        primary.display()
    );
    None
}

fn load_module_hex_rows(path: &Path) -> Vec<HexRow> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let mut rows = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.starts_with("record_idx\t")
        {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 8, "expected 8 columns in TSV line: {line:?}");
        let record_idx = cols[0].trim().parse::<usize>().unwrap_or_else(|e| {
            panic!("invalid record_idx {:?}: {e}", cols[0])
        });
        let val_idx = cols[1]
            .trim()
            .parse::<usize>()
            .unwrap_or_else(|e| panic!("invalid val_idx {:?}: {e}", cols[1]));
        rows.push(HexRow {
            record_idx,
            val_idx,
            src_loc: cols[2].trim().to_owned(),
            label: cols[3].trim().to_owned(),
            raw_bits: parse_hex_bits(cols[7]),
        });
    }
    rows
}

/// The 8 rows out of 234 in `colors/exact_tested_values.tsv` where `Strict`
/// `crlibm::log10_rn` differs by 1-2 ULP from MESA `26.4.1`'s glibc `libm`
/// `log10` (`colors/private/synthetic.f90:111` omits `use math_lib`; tracked
/// in `#198`).
///
/// Tuple fields: `(tsv_row_idx, label, crlibm_exact_bits, max_expected_ulp)`.
const LOG10_CRLIBM_GAPS: [(usize, &str, u64, u64); 8] = [
    (6, "J", 0x400d_3c59_0bb4_5389, 1), // Group 1 solar (#198)
    (47, "I", 0x4010_519d_ad18_55c5, 1), // Group 2a [M/H]=-1.0 (#198)
    (59, "J", 0x400d_3c59_0bb4_5389, 1), // Group 2a [M/H]=0.0 solar (#198)
    (71, "M", 0x400a_4f6a_8a72_a45e, 1), // Group 2a [M/H]=0.5 (#198)
    (83, "R", 0x4011_30c2_1c09_e0ac, 1), // Group 2b logg=1.0 (#198)
    (92, "J", 0x400d_313a_9ec8_fe1f, 1), // Group 2b logg=2.5 (#198)
    (104, "M", 0x400a_3bfe_241f_8fed, 1), // Group 2b logg=4.0 (#198)
    (114, "J", 0x400d_3877_4afd_f3d2, 2), // Group 2b logg=5.0 (#198)
];

#[derive(Default)]
struct CheckStats {
    checked: usize,
    zeros: usize,
    max_ulp: u64,
    failures: Vec<String>,
}

fn check_row(
    rows: &[HexRow],
    idx: usize,
    expected_label: &str,
    actual: f64,
    stats: &mut CheckStats,
) {
    let row = &rows[idx];
    assert_eq!(
        row.label, expected_label,
        "TSV row {idx} (record {}, {}): label mismatch",
        row.record_idx, row.src_loc
    );
    let ulp = ulp_distance(actual, row.raw_bits);
    stats.checked += 1;
    stats.max_ulp = stats.max_ulp.max(ulp);
    if let Some(&(_, gap_label, crlibm_bits, max_expected_ulp)) =
        LOG10_CRLIBM_GAPS.iter().find(|&&(r, ..)| r == idx)
    {
        if row.label != gap_label
            || actual.to_bits() != crlibm_bits
            || ulp == 0
            || ulp > max_expected_ulp
        {
            stats.failures.push(format!(
                "row {idx} (record {}, val {}, {}, `{}` #198 gap): \
                 got {:#018x} ({actual:.17e}), expected crlibm {:#018x}, \
                 mesa {:#018x}, ulp={ulp} (max {max_expected_ulp})",
                row.record_idx,
                row.val_idx,
                row.src_loc,
                row.label,
                actual.to_bits(),
                crlibm_bits,
                row.raw_bits,
            ));
        }
        return;
    }
    if ulp == 0 {
        stats.zeros += 1;
    } else {
        stats.failures.push(format!(
            "row {idx} (record {}, val {}, {}, `{}`): \
             got {:#018x} ({actual:.17e}), expected {:#018x} ({:.17e}), \
             ulp={ulp}",
            row.record_idx,
            row.val_idx,
            row.src_loc,
            row.label,
            actual.to_bits(),
            row.raw_bits,
            f64::from_bits(row.raw_bits),
        ));
    }
}

#[test]
fn test_solar_eval_colors_strict_bit_exact_hex() {
    // Upstream MESA `26.4.1` outputs (`colors/test/src/test_colors.f90:183-197`).
    // All 9 non-`J` columns match at 0 ULP; `J` uses exact `crlibm::log10_rn`
    // (`0x400d_3c59_0bb4_5389`, 1 ULP from MESA's
    // `colors/private/synthetic.f90:111` un-imported glibc `libm` `log10`, #198).
    let mesa_expected: [u64; NUM_COLORS_OUT] = [
        0x4030_3bd9_4058_0275, // Mag_bol
        0x3e95_8a80_b289_f0bc, // Flux_bol
        0x3f88_9b5f_2f5a_6e1e, // Interp_rad
        0x4015_bb38_2edc_bf58, // B
        0x4010_18f2_ad65_6265, // I
        // J: MESA glibc `libm` (`crlibm::log10_rn` = 0x400d_3c59_0bb4_5389)
        0x400d_3c59_0bb4_538a,
        0x400a_3c38_2e3e_8e42, // M
        0x4011_5437_5640_16fe, // R
        0x4016_231d_a7c6_9e08, // U
        0x4013_1ba0_f14c_7a97, // V
    ];
    let solar_j_crlibm_bits: u64 = 0x400d_3c59_0bb4_5389;

    let cube = builtin_sed_cube();
    with_numerics_mode(NumericsMode::Strict, || {
        let got = cube
            .eval_colors(
                5778.0,
                4.44,
                0.0,
                RSUN,
                D_10PC_CM,
                MagSystem::Vega,
                SedInterpMethod::Hermite,
            )
            .expect("eval_colors should succeed");

        let mut mismatches = Vec::new();
        for (idx, (&val, &mesa_bits)) in
            got.iter().zip(mesa_expected.iter()).enumerate()
        {
            let got_bits = val.to_bits();
            if idx == 5 {
                let mesa_ulp = ulp_distance(val, mesa_bits);
                if got_bits != solar_j_crlibm_bits || mesa_ulp != 1 {
                    mismatches.push(format!(
                        "{}: got 0x{got_bits:016x} ({val:.17e}), \
                         expected crlibm 0x{solar_j_crlibm_bits:016x} \
                         (mesa_ulp={mesa_ulp} vs 0x{mesa_bits:016x})",
                        COLOR_COLUMN_NAMES[idx],
                    ));
                }
            } else if got_bits != mesa_bits {
                mismatches.push(format!(
                    "{}: got 0x{got_bits:016x} ({val:.17e}), \
                     expected 0x{mesa_bits:016x} ({:.17e}), ulp={}",
                    COLOR_COLUMN_NAMES[idx],
                    f64::from_bits(mesa_bits),
                    ulp_distance(val, mesa_bits),
                ));
            }
        }
        assert!(
            mismatches.is_empty(),
            "solar eval_colors strict hex mismatches:\n{}",
            mismatches.join("\n")
        );
    });
}

#[test]
fn test_colors_all_234_module_hex_zero_ulp() {
    let Some(tsv_path) = resolve_module_hex_tsv() else {
        return;
    };
    let rows = load_module_hex_rows(&tsv_path);
    assert_eq!(
        rows.len(),
        234,
        "expected 234 colors module-hex rows in {}",
        tsv_path.display()
    );

    let cube = builtin_sed_cube();
    with_numerics_mode(NumericsMode::Strict, || {
        let mut stats = CheckStats::default();
        let mut cursor = 0_usize;

        // 0: Flux cube size in MiB (`colors/private/colors_utils.f90:664-668`).
        let nt = cube.teff_grid.len() as f64;
        let ng = cube.logg_grid.len() as f64;
        let nm = cube.meta_grid.len() as f64;
        let nlam = cube.wavelengths.len() as f64;
        let cube_mb = (((nt * ng) * nm) * nlam) * 8.0 / (1024.0 * 1024.0);
        check_row(
            &rows,
            cursor,
            "colors: flux cube loaded ( x x x ,",
            cube_mb,
            &mut stats,
        );
        cursor += 1;

        // Group 1: representative stellar types (`colors/test/src/test_colors.f90:59-65,183-197`).
        let group1_cases: [(&str, f64, f64, f64, f64); 3] = [
            ("solar", 5778.0, 4.44, 0.0, RSUN),
            ("hot_ms", 15000.0, 4.0, 0.0, 5.0 * RSUN),
            ("cool_giant", 4000.0, 2.0, 0.0, 20.0 * RSUN),
        ];
        for (_label, teff, log_g, feh, radius) in group1_cases {
            let vals = cube
                .eval_colors(
                    teff,
                    log_g,
                    feh,
                    radius,
                    D_10PC_CM,
                    MagSystem::Vega,
                    SedInterpMethod::Hermite,
                )
                .expect("group 1 eval_colors");
            for (col_idx, &col_name) in COLOR_COLUMN_NAMES.iter().enumerate() {
                check_row(&rows, cursor, col_name, vals[col_idx], &mut stats);
                cursor += 1;
            }
        }

        // Group 2a: vary [M/H] (`colors/test/src/test_colors.f90:73,203-217`).
        let sweep_meta: [f64; 4] = [-2.0, -1.0, 0.0, 0.5];
        for &meta in &sweep_meta {
            check_row(&rows, cursor, "# MH=", meta, &mut stats);
            cursor += 1;
            let vals = cube
                .eval_colors(
                    5778.0,
                    4.44,
                    meta,
                    RSUN,
                    D_10PC_CM,
                    MagSystem::Vega,
                    SedInterpMethod::Hermite,
                )
                .expect("group 2a eval_colors");
            for (col_idx, &col_name) in COLOR_COLUMN_NAMES.iter().enumerate() {
                check_row(&rows, cursor, col_name, vals[col_idx], &mut stats);
                cursor += 1;
            }
        }

        // Group 2b: vary log g (`colors/test/src/test_colors.f90:81,223-237`).
        let sweep_logg: [f64; 4] = [1.0, 2.5, 4.0, 5.0];
        for &log_g in &sweep_logg {
            check_row(&rows, cursor, "# logg=", log_g, &mut stats);
            cursor += 1;
            let vals = cube
                .eval_colors(
                    5778.0,
                    log_g,
                    0.0,
                    RSUN,
                    D_10PC_CM,
                    MagSystem::Vega,
                    SedInterpMethod::Hermite,
                )
                .expect("group 2b eval_colors");
            for (col_idx, &col_name) in COLOR_COLUMN_NAMES.iter().enumerate() {
                check_row(&rows, cursor, col_name, vals[col_idx], &mut stats);
                cursor += 1;
            }
        }

        // Group 2c: vary Teff (`colors/test/src/test_colors.f90:89,243-257`).
        let sweep_teff: [f64; 4] = [3500.0, 6000.0, 10000.0, 20000.0];
        for &teff in &sweep_teff {
            check_row(&rows, cursor, "# Teff=", teff, &mut stats);
            cursor += 1;
            let vals = cube
                .eval_colors(
                    teff,
                    4.0,
                    0.0,
                    RSUN,
                    D_10PC_CM,
                    MagSystem::Vega,
                    SedInterpMethod::Hermite,
                )
                .expect("group 2c eval_colors");
            for (col_idx, &col_name) in COLOR_COLUMN_NAMES.iter().enumerate() {
                check_row(&rows, cursor, col_name, vals[col_idx], &mut stats);
                cursor += 1;
            }
        }

        // Group 3a: solar SED sample (`colors/test/src/test_colors.f90:265-276`).
        let sed =
            cube.eval_sed_hermite_generic(5778.0, 4.44, 0.0, RSUN, D_10PC_CM);
        let n_wav = cube.wavelengths.len();
        let stride = (n_wav / 20).max(1);
        let mut idx = 0_usize;
        while idx < n_wav {
            check_row(&rows, cursor, "-", cube.wavelengths[idx], &mut stats);
            cursor += 1;
            check_row(&rows, cursor, "-", sed[idx], &mut stats);
            cursor += 1;
            idx += stride;
        }

        // Group 3b: wavelength-coverage sanity (`colors/test/src/test_colors.f90:279-283,308-339`).
        let bol_flux_rel_tol = 5.0e-3_f64;
        check_row(
            &rows,
            cursor,
            "# wavelength_coverage_sanity  logg=4.0  FeH=0.0  rel_tol=",
            bol_flux_rel_tol,
            &mut stats,
        );
        cursor += 1;

        let wav_min = cube
            .wavelengths
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let wav_max = cube
            .wavelengths
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        for &teff in &sweep_teff {
            let vals = cube
                .eval_colors(
                    teff,
                    4.0,
                    0.0,
                    RSUN,
                    D_10PC_CM,
                    MagSystem::Vega,
                    SedInterpMethod::Hermite,
                )
                .expect("wavelength coverage eval_colors");
            let local_flux = vals[1];
            let expected_flux =
                BOLTZ_SIGMA * pow4(teff) * pow2(RSUN / D_10PC_CM);
            let abs_err = (local_flux - expected_flux).abs();
            let rel_err = abs_err / expected_flux;

            check_row(&rows, cursor, "# case: Teff =", teff, &mut stats);
            cursor += 1;
            check_row(&rows, cursor, "Wav_min_AA", wav_min, &mut stats);
            cursor += 1;
            check_row(&rows, cursor, "Wav_max_AA", wav_max, &mut stats);
            cursor += 1;
            check_row(&rows, cursor, "Flux_actual", local_flux, &mut stats);
            cursor += 1;
            check_row(
                &rows,
                cursor,
                "Flux_expected",
                expected_flux,
                &mut stats,
            );
            cursor += 1;
            check_row(&rows, cursor, "Flux_abserr", abs_err, &mut stats);
            cursor += 1;
            check_row(&rows, cursor, "Flux_relerr", rel_err, &mut stats);
            cursor += 1;
        }

        assert_eq!(cursor, rows.len(), "expected to consume all 234 rows");
        assert_eq!(stats.checked, 234, "expected to check all 234 rows");
        assert!(
            stats.failures.is_empty(),
            "colors module-hex mismatches ({} / {}):\n{}",
            stats.failures.len(),
            stats.checked,
            stats.failures.join("\n")
        );
        assert_eq!(
            stats.zeros, 226,
            "expected 226/234 rows at 0 ULP, got {}/{}",
            stats.zeros, stats.checked
        );
        assert_eq!(
            stats.checked - stats.zeros,
            LOG10_CRLIBM_GAPS.len(),
            "expected exactly {} #198 crlibm::log10_rn gap rows",
            LOG10_CRLIBM_GAPS.len()
        );
        assert!(
            stats.max_ulp <= 2,
            "expected max_ulp <= 2 across all 234 rows, got {}",
            stats.max_ulp
        );
    });
}
