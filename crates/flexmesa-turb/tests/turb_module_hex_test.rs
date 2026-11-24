// SPDX-License-Identifier: GPL-3.0-or-later
//! MESA `26.4.1` Level-2 `turb` module-hex parity test (`#434`, `#157`).
//!
//! Verifies `set_tdc` and `set_mlt` in `f64` strict mode directly against
//! `turb/exact_tested_values.tsv` using the exact `turb/test/src/test_turb.f90`
//! inputs (`62/62` `check_TDC` `gradT` and `conv_vel` outputs at `0 ULP`, plus
//! all `9` `check_efficient_MLT_scaling` and `compare_TDC_and_Cox_MLT` outputs
//! for `71/71` `0 ULP` total).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use std::path::PathBuf;

use flexmesa_const::{
    BOLTZ_SIGMA, KERG, LSUN, MP, MSUN, PI, RSUN, STANDARD_CGRAV,
};
use flexmesa_math::{pow2, pow4, with_numerics_mode, NumericsMode};
use flexmesa_turb::{
    set_mlt, set_tdc, MltInputs, MltOption, MltParams, TdcInputs, TdcParams,
};
use flexmesa_utils::resolve_flexmesa_data_dir;

fn signed_magnitude(bits: u64) -> i128 {
    if (bits >> 63) != 0 {
        -((bits ^ (1_u64 << 63)) as i128)
    } else {
        bits as i128
    }
}

fn ulp_distance(actual: f64, expected: f64) -> u64 {
    if actual.is_nan() || expected.is_nan() {
        return if actual.is_nan() && expected.is_nan() {
            0
        } else {
            u64::MAX
        };
    }
    let a = signed_magnitude(actual.to_bits());
    let b = signed_magnitude(expected.to_bits());
    u64::try_from((a - b).unsigned_abs()).unwrap_or(u64::MAX)
}

fn parse_hex_f64(raw: &str) -> f64 {
    let trimmed = raw
        .trim()
        .strip_prefix("0x")
        .or_else(|| raw.trim().strip_prefix("0X"))
        .unwrap_or_else(|| {
            panic!("expected 0x-prefixed hex float, got {raw:?}")
        });
    let bits = u64::from_str_radix(trimmed, 16)
        .unwrap_or_else(|e| panic!("invalid hex bits {raw:?}: {e}"));
    f64::from_bits(bits)
}

fn resolve_turb_hex_tsv() -> PathBuf {
    let hex_root = std::env::var_os("FLEXMESA_MODULE_HEX_DIR")
        .filter(|v| !v.is_empty())
        .map_or_else(
            || {
                let data_dir = resolve_flexmesa_data_dir();
                data_dir
                    .parent()
                    .unwrap_or(&data_dir)
                    .join("mesa_module_hex")
            },
            PathBuf::from,
        );
    let tsv = hex_root.join("turb").join("exact_tested_values.tsv");
    assert!(
        tsv.is_file(),
        "Missing turb module-hex TSV at {}. Run \
         'flexmesa data fetch --include-test-inputs' or set FLEXMESA_MODULE_HEX_DIR.",
        tsv.display()
    );
    tsv
}

fn load_turb_hex_values() -> Vec<f64> {
    let tsv = resolve_turb_hex_tsv();
    let text = std::fs::read_to_string(&tsv)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", tsv.display()));
    let mut values = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || (idx == 0 && trimmed.starts_with("record_idx\t"))
        {
            continue;
        }
        let cols: Vec<&str> = trimmed.split('\t').collect();
        assert_eq!(cols.len(), 8, "unexpected TSV row {}: {trimmed}", idx + 1);
        values.push(parse_hex_f64(cols[7]));
    }
    assert_eq!(
        values.len(),
        136,
        "expected 136 rows in turb/exact_tested_values.tsv"
    );
    values
}

/// Upstream `auto_diff_real_star_order1` division (`x * powm1(y)`) used by
/// `turb/test/src/test_turb.f90` when constructing `Lambda`, `grav`, `gradr`,
/// and `L`.
#[inline]
fn ad_div(a: f64, b: f64) -> f64 {
    a * (1.0 / b)
}

/// Exact `check_TDC` inputs from `turb/test/src/test_turb.f90:182-214`, with
/// `L = 1.0941528815883500015d0 * (-scale)` (`test_turb.f90:196`) and
/// `pow4(T) = T * T * T * T` (`math/private/math_pown.f90:99`).
fn make_check_tdc_inputs() -> (TdcInputs<f64>, f64, f64, f64) {
    let cgrav = 6.674_300_000_000_000_4e-8;
    let m = 5.870_740_045_687_566_4e34;
    let scale = 5.038_651_936_224_629_4e45;
    let r = 10_314_294_541.567_163;
    let p = 5.058_158_724_980_889_4e20;
    let t = 613_193_666.517_836_8;
    let opacity = 9.075_017_123_146_995e-2;
    let l_val = 1.094_152_881_588_350_001_5 * (-scale);
    let gradr = ad_div(
        3.0 * p * opacity * l_val,
        64.0 * PI * BOLTZ_SIGMA * pow4(t) * cgrav * m,
    );
    assert_eq!(
        gradr.to_bits(),
        0xc0fd_58d0_de9f_6715,
        "upstream test_turb.f90:211 gradr bit pattern mismatch"
    );
    let inputs = TdcInputs {
        chi_t: 1.0,
        chi_rho: 1.0,
        gradr,
        r,
        p,
        t,
        rho: 5_204.573_257_474_575,
        dv: 3.825_649_446_348_260_4e-7,
        cp: 6_628_075_118.460_659,
        opacity,
        scale_height: 2_638_686_602.006_378,
        gradl: 0.252_075_872_673_435,
        grada: 0.252_046_972_568_727_4,
        eq_div_w: 0.0,
        grav: ad_div(m * cgrav, pow2(r)),
        energy: 0.0,
    };
    (inputs, cgrav, m, scale)
}

fn check_tdc_31_steps(values: &[f64]) -> (usize, usize, u64) {
    let (inputs, cgrav, m, scale) = make_check_tdc_inputs();
    let mut checked = 0_usize;
    let mut zeros = 0_usize;
    let mut max_ulp = 0_u64;
    let mut details = Vec::new();

    for j in 0..31_usize {
        let base = 3 + 4 * j;
        let dt = values[base];
        let want_grad_t = values[base + 1];
        let conv_vel_start = values[base + 2];
        let want_conv_vel = values[base + 3];

        let params = TdcParams {
            conv_vel_start,
            mixing_length_alpha: 2.0,
            tdc_alpha_d: 1.0,
            tdc_alpha_r: 0.0,
            tdc_alpha_pt: 0.0,
            tdc_alpha_c: 1.0,
            tdc_alpha_s: 1.0,
            dt,
            cgrav,
            m,
            scale,
            max_conv_vel: 1.0e99,
            include_mlt_corr_to_tdc: true,
            use_tdc_enthalpy_flux_limiter: false,
        };

        let res = set_tdc(&params, &inputs)
            .unwrap_or_else(|e| panic!("set_tdc failed at j={j}: {e}"));

        for (label, got, want) in [
            ("gradT", res.output.grad_t, want_grad_t),
            ("conv_vel", res.output.conv_vel, want_conv_vel),
        ] {
            let ulp = ulp_distance(got, want);
            checked += 1;
            if ulp == 0 {
                zeros += 1;
            } else {
                details.push(format!(
                    "j={j:02} {label:8}: ulp={ulp} got={:#018x} ({got:.17e}) want={:#018x} ({want:.17e})",
                    got.to_bits(),
                    want.to_bits(),
                ));
            }
            max_ulp = max_ulp.max(ulp);
        }
    }

    assert!(
        details.is_empty(),
        "check_TDC module-hex mismatch: {zeros}/{checked} at 0 ULP, max_ulp={max_ulp}\n{}",
        details.join("\n")
    );
    (checked, zeros, max_ulp)
}

#[test]
fn test_turb_module_hex_check_tdc_zero_ulp() {
    let values = load_turb_hex_values();
    with_numerics_mode(NumericsMode::Strict, || {
        let (checked, zeros, max_ulp) = check_tdc_31_steps(&values);
        assert_eq!(checked, 62);
        assert_eq!(zeros, 62);
        assert_eq!(max_ulp, 0);
    });
}

struct HexStats {
    checked: usize,
    zeros: usize,
    max_ulp: u64,
}

impl HexStats {
    fn new() -> Self {
        Self {
            checked: 0,
            zeros: 0,
            max_ulp: 0,
        }
    }

    fn record(&mut self, label: &str, got: f64, want: f64) {
        let ulp = ulp_distance(got, want);
        self.checked += 1;
        if ulp == 0 {
            self.zeros += 1;
        }
        self.max_ulp = self.max_ulp.max(ulp);
        assert_eq!(
            ulp,
            0,
            "{label}: got={:#018x} ({got:.17e}) want={:#018x} ({want:.17e})",
            got.to_bits(),
            want.to_bits()
        );
    }
}

#[test]
fn test_turb_module_hex_all_71_test_turb_outputs_zero_ulp() {
    let values = load_turb_hex_values();

    with_numerics_mode(NumericsMode::Strict, || {
        let mut stats = HexStats::new();

        // 1. `check_efficient_MLT_scaling` (`turb/test/src/test_turb.f90:27-80`)
        let t = 1.0e5_f64;
        let rho = 1.0e-5_f64;
        let grav = 1.0e4_f64;
        let r = RSUN;
        let cp = 2.5 * KERG / MP;
        let p = rho * t * KERG / MP;
        let lambda = ad_div(p, rho * grav);
        let opacity = 1.0_f64;
        let grada = 0.4_f64;
        let gradl = grada;
        let mlt_p = MltParams::cox(1.0);

        let l1 = 1.0e5 * LSUN;
        let gradr1 = ad_div(
            3.0 * p * opacity * l1,
            64.0 * PI * BOLTZ_SIGMA * pow4(t) * grav * pow2(r),
        );
        let out1 = set_mlt(
            &mlt_p,
            &MltInputs {
                chi_t: 1.0,
                chi_rho: 1.0,
                cp,
                grav,
                lambda,
                rho,
                p,
                t,
                opacity,
                gradr: gradr1,
                grada,
                gradl,
            },
        )
        .unwrap();
        stats.record("mlt_scaling vc1", out1.conv_vel, values[0]);

        let l2 = 1.0e8 * LSUN;
        let gradr2 = ad_div(
            3.0 * p * opacity * l2,
            64.0 * PI * BOLTZ_SIGMA * pow4(t) * grav * pow2(r),
        );
        let out2 = set_mlt(
            &mlt_p,
            &MltInputs {
                chi_t: 1.0,
                chi_rho: 1.0,
                cp,
                grav,
                lambda,
                rho,
                p,
                t,
                opacity,
                gradr: gradr2,
                grada,
                gradl,
            },
        )
        .unwrap();
        stats.record("mlt_scaling vc2", out2.conv_vel, values[1]);
        stats.record(
            "mlt_scaling ratio",
            out2.conv_vel / out1.conv_vel,
            values[2],
        );

        // 2. `check_TDC` (`turb/test/src/test_turb.f90:168-235`, 62 outputs)
        let (tdc_checked, tdc_zeros, tdc_max_ulp) = check_tdc_31_steps(&values);
        stats.checked += tdc_checked;
        stats.zeros += tdc_zeros;
        stats.max_ulp = stats.max_ulp.max(tdc_max_ulp);

        // 3. `compare_TDC_and_Cox_MLT` (`turb/test/src/test_turb.f90:82-166`)
        let alpha = 2.0_f64;
        let m_sun = MSUN;
        let cgrav_s = STANDARD_CGRAV;
        let grav_s = ad_div(m_sun * cgrav_s, pow2(r));
        let hp_s = ad_div(p, rho * grav_s);
        let lam_s = alpha * hp_s;
        let mut l_s = 70.0 * LSUN;
        let mut gr_s = ad_div(
            3.0 * p * opacity * l_s,
            64.0 * PI * BOLTZ_SIGMA * pow4(t) * grav_s * pow2(r),
        );
        l_s = l_s * (1.0 + 1.0e-5) * ad_div(grada, gr_s);
        gr_s = ad_div(
            3.0 * p * opacity * l_s,
            64.0 * PI * BOLTZ_SIGMA * pow4(t) * grav_s * pow2(r),
        );
        stats.record("compare gradR - gradA", gr_s - grada, values[127]);

        let tdc_out = set_tdc(
            &TdcParams {
                conv_vel_start: 0.0,
                mixing_length_alpha: alpha,
                tdc_alpha_d: 1.0,
                tdc_alpha_r: 0.0,
                tdc_alpha_pt: 0.0,
                tdc_alpha_c: 1.0,
                tdc_alpha_s: 1.0,
                dt: 1.0e40,
                cgrav: cgrav_s,
                m: m_sun,
                scale: l_s * 1.0e-3,
                max_conv_vel: 1.0e99,
                include_mlt_corr_to_tdc: true,
                use_tdc_enthalpy_flux_limiter: false,
            },
            &TdcInputs {
                chi_t: 1.0,
                chi_rho: 1.0,
                gradr: gr_s,
                r,
                p,
                t,
                rho,
                dv: 0.0,
                cp,
                opacity,
                scale_height: hp_s,
                gradl,
                grada,
                eq_div_w: 0.0,
                grav: grav_s,
                energy: 0.0,
            },
        )
        .unwrap();
        stats.record("compare TDC Y", tdc_out.output.y_face, values[128]);
        stats.record(
            "compare TDC conv_vel",
            tdc_out.output.conv_vel,
            values[130],
        );

        let mlt_out = set_mlt(
            &MltParams {
                option: MltOption::Cox,
                mixing_length_alpha: alpha,
                henyey_nu: 0.0,
                henyey_y: 0.0,
                max_conv_vel: 1.0e99,
            },
            &MltInputs {
                chi_t: 1.0,
                chi_rho: 1.0,
                cp,
                grav: grav_s,
                lambda: lam_s,
                rho,
                p,
                t,
                opacity,
                gradr: gr_s,
                grada,
                gradl,
            },
        )
        .unwrap();
        stats.record("compare MLT Y", mlt_out.y_face, values[132]);
        stats.record("compare MLT conv_vel", mlt_out.conv_vel, values[134]);
        stats.record("compare MLT Gamma", mlt_out.gamma, values[135]);

        assert_eq!(stats.checked, 71);
        assert_eq!(stats.zeros, 71);
        assert_eq!(stats.max_ulp, 0);
    });
}
