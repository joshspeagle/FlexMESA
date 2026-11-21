// SPDX-License-Identifier: LGPL-3.0-or-later
//! MESA `26.4.1` IEEE-754 hex oracle test for `turb::set_tdc` values and all 16
//! input `d1Array` partial derivatives (`#328`, `#321`, `DECISIONS #50`).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;

use flexmesa_autodiff::DualOrder1;
use flexmesa_const::MixingType;
use flexmesa_math::{NumericsMode, with_numerics_mode};
use flexmesa_turb::{TdcInputs, TdcParams, set_tdc};

type OracleRowMap = HashMap<(String, String, String, usize), f64>;

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

fn load_tdc_oracles() -> (Vec<String>, OracleRowMap) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../benchmarks/fortran_mesa/turb_tdc_oracles/mesa_26_4_1_turb_tdc_oracles.tsv",
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("failed to read {}: {e}", path.display());
    });

    let mut has_mesa_ver = false;
    let mut has_mesa_commit = false;
    let mut has_sdk_ver = false;
    let mut has_math_backend = false;
    let mut has_compiler = false;
    let mut has_fflags = false;
    let mut cases: Vec<String> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut map: OracleRowMap = HashMap::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("# mesa_version:") {
            has_mesa_ver = true;
            continue;
        }
        if trimmed.starts_with("# mesa_git_commit:") {
            has_mesa_commit = true;
            continue;
        }
        if trimmed.starts_with("# mesasdk_version:") {
            has_sdk_ver = true;
            continue;
        }
        if trimmed.starts_with("# math_backend:") {
            has_math_backend = true;
            continue;
        }
        if trimmed.starts_with("# compiler:") {
            assert!(
                trimmed.contains("GCC version"),
                "unexpected compiler: {trimmed}"
            );
            has_compiler = true;
            continue;
        }
        if trimmed.starts_with("# fflags:") {
            has_fflags = true;
            continue;
        }
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("section\t") {
            continue;
        }
        let cols: Vec<&str> = trimmed.split('\t').collect();
        assert_eq!(cols.len(), 8, "unexpected oracle row: {trimmed}");
        let cid = cols[1].to_string();
        let role = cols[2].to_string();
        let field = cols[3].to_string();
        let slot: usize = cols[4].parse().expect("parse slot");
        let hex = cols[6]
            .strip_prefix("0x")
            .unwrap_or_else(|| panic!("missing 0x prefix: {}", cols[6]));
        let val = f64::from_bits(
            u64::from_str_radix(hex, 16)
                .unwrap_or_else(|e| panic!("invalid hex {}: {e}", cols[6])),
        );
        if seen.insert(cid.clone()) {
            cases.push(cid.clone());
        }
        map.insert((cid, role, field, slot), val);
    }

    assert!(has_mesa_ver, "missing # mesa_version header");
    assert!(has_mesa_commit, "missing # mesa_git_commit header");
    assert!(has_sdk_ver, "missing # mesasdk_version header");
    assert!(has_math_backend, "missing # math_backend header");
    assert!(has_compiler, "missing # compiler header");
    assert!(has_fflags, "missing # fflags header");
    assert_eq!(cases.len(), 32, "expected 32 turb set_TDC oracle cases");
    (cases, map)
}

fn oracle_val(map: &OracleRowMap, cid: &str, role: &str, field: &str, slot: usize) -> f64 {
    *map.get(&(cid.to_string(), role.to_string(), field.to_string(), slot))
        .unwrap_or_else(|| panic!("missing ({cid}, {role}, {field}, {slot})"))
}

const SLOT_FIELDS: [&str; 16] = [
    "chiT",
    "chiRho",
    "gradr",
    "r",
    "P",
    "T",
    "rho",
    "dV",
    "Cp",
    "opacity",
    "scale_height",
    "gradL",
    "grada",
    "Eq_div_w",
    "grav",
    "energy",
];

#[test]
fn test_set_tdc_values_and_d1_partials_against_mesa_26_4_1_hex() {
    let (cases, map) = load_tdc_oracles();

    with_numerics_mode(NumericsMode::Strict, || {
        let mut max_val_ulp = 0_u64;
        let mut max_val_rel = 0.0_f64;
        let mut max_non_cancelled_rel = 0.0_f64;
        let mut max_check_tdc_scaled = 0.0_f64;
        let mut max_cox_scaled = 0.0_f64;
        let mut val_exact = 0_usize;
        let mut val_total = 0_usize;
        let mut y_d1_exact = 0_usize;
        let mut y_d1_total = 0_usize;
        let mut v_d1_exact = 0_usize;
        let mut v_d1_total = 0_usize;

        for cid in &cases {
            let p = TdcParams {
                conv_vel_start: oracle_val(&map, cid, "param", "conv_vel_start", 0),
                mixing_length_alpha: oracle_val(&map, cid, "param", "mixing_length_alpha", 0),
                tdc_alpha_d: oracle_val(&map, cid, "param", "TDC_alpha_D", 0),
                tdc_alpha_r: oracle_val(&map, cid, "param", "TDC_alpha_R", 0),
                tdc_alpha_pt: oracle_val(&map, cid, "param", "TDC_alpha_Pt", 0),
                tdc_alpha_c: oracle_val(&map, cid, "param", "TDC_alpha_C", 0),
                tdc_alpha_s: oracle_val(&map, cid, "param", "TDC_alpha_S", 0),
                dt: oracle_val(&map, cid, "param", "dt", 0),
                cgrav: oracle_val(&map, cid, "param", "cgrav", 0),
                m: oracle_val(&map, cid, "param", "m", 0),
                scale: oracle_val(&map, cid, "param", "scale", 0),
                max_conv_vel: oracle_val(&map, cid, "param", "max_conv_vel", 0),
                include_mlt_corr_to_tdc: true,
                use_tdc_enthalpy_flux_limiter: false,
            };

            let d = |idx: usize| -> DualOrder1<16> {
                let v = oracle_val(&map, cid, "input", SLOT_FIELDS[idx], idx + 1);
                DualOrder1::<16>::independent(v, idx)
            };

            let inp = TdcInputs {
                chi_t: d(0),
                chi_rho: d(1),
                gradr: d(2),
                r: d(3),
                p: d(4),
                t: d(5),
                rho: d(6),
                dv: d(7),
                cp: d(8),
                opacity: d(9),
                scale_height: d(10),
                gradl: d(11),
                grada: d(12),
                eq_div_w: d(13),
                grav: d(14),
                energy: d(15),
            };

            let res =
                set_tdc(&p, &inp).unwrap_or_else(|e| panic!("set_tdc failed for {cid}: {e}"));
            let want_iters = oracle_val(&map, cid, "output", "tdc_num_iters", 0) as usize;
            let want_mtype_code = oracle_val(&map, cid, "output", "mixing_type", 0) as i32;
            let got_mtype_code = match res.output.mixing_type {
                MixingType::NoMixing => 0,
                MixingType::ConvectiveMixing => 1,
                _ => -1,
            };
            assert_eq!(got_mtype_code, want_mtype_code, "{cid} mixing_type mismatch");
            if want_mtype_code == 1 {
                assert_eq!(res.num_iters, want_iters, "{cid} num_iters mismatch");
            } else {
                // Upstream `get_TDC_solution` leaves `intent(out) :: tdc_num_iters`
                // unassigned on `radY` early exits (`turb/private/tdc.f90:55, 177`).
                assert_eq!(res.num_iters, 0, "{cid} radiative branch num_iters must be 0");
            }

            let want_y_val = oracle_val(&map, cid, "output", "Y_face_val", 0);
            let want_v_val = oracle_val(&map, cid, "output", "conv_vel_val", 0);
            let want_gt_val = oracle_val(&map, cid, "output", "gradT_val", 0);
            let want_d_val = oracle_val(&map, cid, "output", "D_val", 0);

            for (name, got, want) in [
                ("Y_face_val", res.output.y_face.val, want_y_val),
                ("conv_vel_val", res.output.conv_vel.val, want_v_val),
                ("gradT_val", res.output.grad_t.val, want_gt_val),
                ("D_val", res.output.d.val, want_d_val),
            ] {
                let u = ulp_distance(got, want);
                let rel = if want == 0.0 {
                    got.abs()
                } else {
                    (got - want).abs() / want.abs()
                };
                val_total += 1;
                if u == 0 {
                    val_exact += 1;
                }
                assert_eq!(u, 0, "{cid} {name}: ulp={u}, rel={rel:.3e}");
                max_val_ulp = max_val_ulp.max(u);
                max_val_rel = max_val_rel.max(rel);
            }

            for idx in 0..16 {
                let sname = SLOT_FIELDS[idx];
                let x_i = oracle_val(&map, cid, "input", sname, idx + 1);

                for (fld_name, got, want, out_val, exact_cnt, total_cnt) in [
                    (
                        "Y_face_d1",
                        res.output.y_face.d1[idx],
                        oracle_val(&map, cid, "output", "Y_face_d1", idx + 1),
                        want_y_val,
                        &mut y_d1_exact,
                        &mut y_d1_total,
                    ),
                    (
                        "conv_vel_d1",
                        res.output.conv_vel.d1[idx],
                        oracle_val(&map, cid, "output", "conv_vel_d1", idx + 1),
                        want_v_val,
                        &mut v_d1_exact,
                        &mut v_d1_total,
                    ),
                ] {
                    *total_cnt += 1;
                    let u = ulp_distance(got, want);
                    if u == 0 {
                        *exact_cnt += 1;
                    }
                    let rel = if want == 0.0 {
                        got.abs()
                    } else {
                        (got - want).abs() / want.abs()
                    };
                    let scaled = if out_val == 0.0 {
                        (got - want).abs()
                    } else {
                        (got - want).abs() * x_i.abs() / out_val.abs()
                    };
                    let dim_sens = if out_val == 0.0 {
                        0.0
                    } else {
                        (x_i * want / out_val).abs()
                    };

                    if dim_sens >= 1.0e-3 && cid != "compare_tdc_cox_0" {
                        max_non_cancelled_rel = max_non_cancelled_rel.max(rel);
                        assert!(
                            rel <= 5.0e-15 && u <= 32,
                            "{cid} {fld_name}[{sname}] (non-cancelled): ulp={u}, rel={rel:.3e}"
                        );
                    }
                    if cid == "compare_tdc_cox_0" {
                        max_cox_scaled = max_cox_scaled.max(scaled);
                        assert!(
                            scaled <= 1.0e-10,
                            "{cid} {fld_name}[{sname}] scaled err={scaled:.3e}"
                        );
                    } else {
                        max_check_tdc_scaled = max_check_tdc_scaled.max(scaled);
                        assert!(
                            scaled <= 1.0e-13,
                            "{cid} {fld_name}[{sname}] scaled err={scaled:.3e}"
                        );
                    }
                }
            }
        }

        // Pin measured 0-ULP counts and tolerances (`DECISIONS #50`):
        // - Scalar values: 128/128 at 0 ULP.
        // - Y_face.d1: 389/512 at 0 ULP.
        // - conv_vel.d1: 413/512 at 0 ULP.
        // - Non-cancelled check_tdc partials: max_rel=1.644e-15.
        // - Near-cancelled check_tdc partials: scaled <= 7.344e-15.
        // - compare_tdc_cox_0: 1e5 sensitivity in L - L0*gradL gives max scaled=9.213e-11.
        // The partials that still differ come from the first-order dual
        // division, which does not yet use upstream's `div_self` formula
        // (`auto_diff/private/auto_diff_real_star_order1_module.f90`).
        assert_eq!(val_total, 128);
        assert_eq!(val_exact, 128, "val_exact regression: {val_exact}/128");
        assert_eq!(max_val_ulp, 0);
        assert_eq!(y_d1_total, 512);
        assert_eq!(v_d1_total, 512);
        assert!(y_d1_exact >= 389, "y_d1_exact regression: {y_d1_exact}/512");
        assert!(v_d1_exact >= 413, "v_d1_exact regression: {v_d1_exact}/512");
        assert!(max_non_cancelled_rel <= 5.0e-15);
        assert!(max_check_tdc_scaled <= 1.0e-13);
        assert!(max_cox_scaled <= 1.0e-10);
        println!(
            "set_TDC hex oracle summary: vals={val_exact}/{val_total} 0-ULP (max_ulp={max_val_ulp}, max_rel={max_val_rel:.3e}), \
             Y_face.d1={y_d1_exact}/{y_d1_total} 0-ULP, conv_vel.d1={v_d1_exact}/{v_d1_total} 0-ULP, \
             non_cancelled_rel={max_non_cancelled_rel:.3e}, check_tdc_scaled={max_check_tdc_scaled:.3e}, cox_scaled={max_cox_scaled:.3e}"
        );
    });
}
