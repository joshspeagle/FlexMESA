// SPDX-License-Identifier: LGPL-3.0-or-later
//! Issue `#404` regression tests: exact 0-ULP MESA `26.4.1` `AUX opacity` parity
//! in the low-T / high-T blend region (`kap/private/kap_eval.f90:345-365`)
//! across `star/16M_conv_premix`, `star/16M_predictive_mix`, and `star/low_z`.

#![forbid(unsafe_code)]
#![allow(clippy::float_cmp, clippy::similar_names, clippy::unreadable_literal)]

use flexmesa_const::LN10;
use flexmesa_kap::{
    get_kap_high_t_results, get_kap_results_blend_t, KapCoOption, KapConfig,
    KapOption,
};
use flexmesa_math::{exp10, safe_log10, with_numerics_mode, NumericsMode};
use flexmesa_star::micro::{do_eos, do_kap, global_basic_net, set_basic_vars};
use flexmesa_star::{StarControls, StarJobControls, StarModel};

struct AuxKapZoneOracle {
    zone_1based: usize,
    lnd_bits: u64,
    ln_t_bits: u64,
    xa_bits: [u64; 8],
    expected_kap_bits: u64,
}

fn signed_magnitude(bits: u64) -> i128 {
    if (bits >> 63) != 0 {
        -i128::from(bits ^ (1_u64 << 63))
    } else {
        i128::from(bits)
    }
}

fn ulp_distance(actual: f64, expected: f64) -> u64 {
    let a = signed_magnitude(actual.to_bits());
    let b = signed_magnitude(expected.to_bits());
    u64::try_from((a - b).unsigned_abs()).unwrap_or(u64::MAX)
}

fn verify_case_oracles(
    case_label: &str,
    kap_config: KapConfig,
    zbase: f64,
    oracles: &[AuxKapZoneOracle],
) {
    with_numerics_mode(NumericsMode::Strict, || {
        let net = global_basic_net();
        let nz = oracles.len();
        let controls = StarControls {
            zbase,
            ..StarControls::default()
        };
        let mut model = StarModel::new(
            nz,
            net.chem_id.clone(),
            StarJobControls::default(),
            controls,
        );
        assert_eq!(
            model.iso_names,
            ["h1", "he3", "he4", "c12", "n14", "o16", "ne20", "mg24"],
            "{case_label}: unexpected basic.net species order"
        );
        model.kap_config = kap_config;

        for (k, row) in oracles.iter().enumerate() {
            model.lnd[k] = f64::from_bits(row.lnd_bits);
            model.ln_t[k] = f64::from_bits(row.ln_t_bits);
            model.ln_r[k] = 25.0;
            for j in 0..8 {
                model.xa[k * 8 + j] = f64::from_bits(row.xa_bits[j]);
            }
        }

        set_basic_vars(&mut model).expect("set_basic_vars");
        do_eos(&mut model).expect("do_eos");
        do_kap(&mut model).expect("do_kap");

        for (k, row) in oracles.iter().enumerate() {
            let actual = model.opacity[k];
            let expected = f64::from_bits(row.expected_kap_bits);
            let ulp = ulp_distance(actual, expected);
            assert_eq!(
                actual.to_bits(),
                row.expected_kap_bits,
                "{case_label} zone {}: opacity mismatch: got 0x{:016X} \
                 ({actual:.17e}), want 0x{:016X} ({expected:.17e}), ULP={ulp}",
                row.zone_1based,
                actual.to_bits(),
                row.expected_kap_bits,
            );
        }
    });
}

#[test]
fn test_kap_aux_16m_conv_premix_zero_ulp() {
    // MESA 26.4.1 `star/16M_conv_premix` model 11 (`a09` + `a09_co`, `Zbase = 0.02`):
    // zone 111 previously diverged by 1 ULP (`0x3FE4E46784111DDB` vs `0x3FE4E46784111DDC`);
    // zones 107 and 117 also hit `u_div != u_mul` inside the [3.80, 3.88] blend.
    const XA: [u64; 8] = [
        0x3FE6666666666666,
        0x3EFF3EBDE7058CC2,
        0x3FD1EB0823C0B5D5,
        0x3F6D07996E870861,
        0x3F5104AAE4CAEF46,
        0x3F819A547042BDC8,
        0x3F5E96A1EB5D6DFD,
        0x3F744C3C52E53AEE,
    ];
    let oracles = [
        AuxKapZoneOracle {
            zone_1based: 107,
            lnd_bits: 0xC0328EC07F9242B9,
            ln_t_bits: 0x4021B64C8536DB0F,
            xa_bits: XA,
            expected_kap_bits: 0x3FDF9740636589AD,
        },
        AuxKapZoneOracle {
            zone_1based: 111,
            lnd_bits: 0xC03292AABB8965FF,
            ln_t_bits: 0x4021C2E448E725FB,
            xa_bits: XA,
            expected_kap_bits: 0x3FE4E46784111DDC,
        },
        AuxKapZoneOracle {
            zone_1based: 117,
            lnd_bits: 0xC03298D7CADE1A6D,
            ln_t_bits: 0x4021D89A6C91E265,
            xa_bits: XA,
            expected_kap_bits: 0x3FF1B545A49461FA,
        },
    ];
    let kap_cfg = KapConfig {
        zbase: 0.02,
        kap_option: KapOption::A09,
        kap_co_option: KapCoOption::A09Co,
        use_type2_opacities: true,
        ..KapConfig::default()
    };
    verify_case_oracles("star/16M_conv_premix", kap_cfg, 0.02, &oracles);
}

#[test]
fn test_kap_aux_16m_predictive_mix_zero_ulp() {
    // MESA 26.4.1 `star/16M_predictive_mix` model 11 (`gs98` + `gs98_co`, `Zbase = 0.02`):
    // zone 94 previously diverged by 2 ULP (`0x3FCE5E620D86BC12` vs `0x3FCE5E620D86BC10`);
    // zones 89, 90, 92, 93, and 95 also hit `u_div != u_mul` inside [3.80, 3.88].
    const XA: [u64; 8] = [
        0x3FE6666666666667,
        0x3EFF3EBDE7058CB6,
        0x3FD1EB0823C0B5D6,
        0x3F6C3CA369D8B423,
        0x3F508DAF41EC844F,
        0x3F8336191BA4210F,
        0x3F61149CF6C7EBFF,
        0x3F70B346E6A49EB9,
    ];
    let oracles = [
        AuxKapZoneOracle {
            zone_1based: 89,
            lnd_bits: 0xC03283A05B4BBEB7,
            ln_t_bits: 0x4021857188DAD1C6,
            xa_bits: XA,
            expected_kap_bits: 0x3FC6A6AC1582D5BF,
        },
        AuxKapZoneOracle {
            zone_1based: 90,
            lnd_bits: 0xC032844B13DDA5D4,
            ln_t_bits: 0x40218807F4D6A47A,
            xa_bits: XA,
            expected_kap_bits: 0x3FC7F8ADF378CE7D,
        },
        AuxKapZoneOracle {
            zone_1based: 92,
            lnd_bits: 0xC03285C29B6A2F6C,
            ln_t_bits: 0x40218D6FCC8A3B97,
            xa_bits: XA,
            expected_kap_bits: 0x3FCAEF948F0F8760,
        },
        AuxKapZoneOracle {
            zone_1based: 93,
            lnd_bits: 0xC032868ECC16FA1C,
            ln_t_bits: 0x4021903F9C2DDAD9,
            xa_bits: XA,
            expected_kap_bits: 0x3FCC97500FF8309D,
        },
        AuxKapZoneOracle {
            zone_1based: 94,
            lnd_bits: 0xC032876517398DD8,
            ln_t_bits: 0x4021932016A8032E,
            xa_bits: XA,
            expected_kap_bits: 0x3FCE5E620D86BC10,
        },
        AuxKapZoneOracle {
            zone_1based: 95,
            lnd_bits: 0xC03288448370556E,
            ln_t_bits: 0x4021960F23138DC1,
            xa_bits: XA,
            expected_kap_bits: 0x3FD0230C35774DA2,
        },
    ];
    let kap_cfg = KapConfig {
        zbase: 0.02,
        kap_option: KapOption::Gs98,
        kap_co_option: KapCoOption::Gs98Co,
        use_type2_opacities: true,
        ..KapConfig::default()
    };
    verify_case_oracles("star/16M_predictive_mix", kap_cfg, 0.02, &oracles);
}

#[test]
fn test_kap_aux_low_z_zero_ulp() {
    // MESA 26.4.1 `star/low_z` model 394 (`gs98` + `gs98_co`, `Zbase = 0.0001`):
    // zone 12 previously diverged by 2 ULP (`0x3FFF2FF5E83161F4` vs `0x3FFF2FF5E83161F2`).
    const XA: [u64; 8] = [
        0x3FE84F765FC5D094,
        0x3EFACDD08BF4F1AB,
        0x3FCEBE093606FA83,
        0x3EF212541AC81F56,
        0x3ED5303C7D57B126,
        0x3F08972A60D212CC,
        0x3EE5DCF1DFB835C9,
        0x3EF5605AC0D2B43D,
    ];
    let oracles = [
        AuxKapZoneOracle {
            zone_1based: 11,
            lnd_bits: 0xC02E91A0D9F20A6C,
            ln_t_bits: 0x4021B900A863F02E,
            xa_bits: XA,
            expected_kap_bits: 0x3FFABD75F05D43EF,
        },
        AuxKapZoneOracle {
            zone_1based: 12,
            lnd_bits: 0xC02E8B13E47965EB,
            ln_t_bits: 0x4021C0D06F7E0CF9,
            xa_bits: XA,
            expected_kap_bits: 0x3FFF2FF5E83161F2,
        },
        AuxKapZoneOracle {
            zone_1based: 13,
            lnd_bits: 0xC02E85C10CBB5F19,
            ln_t_bits: 0x4021C96838D4EACD,
            xa_bits: XA,
            expected_kap_bits: 0x40028CC7572E3468,
        },
    ];
    let kap_cfg = KapConfig {
        zbase: 0.0001,
        kap_option: KapOption::Gs98,
        kap_co_option: KapCoOption::Gs98Co,
        use_type2_opacities: true,
        ..KapConfig::default()
    };
    verify_case_oracles("star/low_z", kap_cfg, 0.0001, &oracles);
}

#[test]
fn test_kap_blend_t_derivatives_and_type2_dz_blend_consistency() {
    with_numerics_mode(NumericsMode::Strict, || {
        let cfg = KapConfig {
            zbase: 0.02,
            kap_option: KapOption::A09,
            kap_co_option: KapCoOption::A09Co,
            use_type2_opacities: true,
            ..KapConfig::default()
        };
        let log_rho = f64::from_bits(0xC03292AABB8965FF) / LN10;
        let log_t = f64::from_bits(0x4021C2E448E725FB) / LN10;
        let (f_lt, f_ht, _, kap, dln_dr, dln_dt) = get_kap_results_blend_t(
            &cfg, 0.7, 0.02, 0.0035, 0.001, 0.0086, 0.0019, log_rho, log_t,
        )
        .expect("blend_t eval");
        assert!(
            f_lt > 0.0 && f_ht > 0.0 && (f_lt + f_ht - 1.0).abs() <= 1.0e-15
        );

        let eps_ln = 1.0e-6;
        let dlog = eps_ln / LN10;
        let (_, _, _, kp_t, _, _) = get_kap_results_blend_t(
            &cfg,
            0.7,
            0.02,
            0.0035,
            0.001,
            0.0086,
            0.0019,
            log_rho,
            log_t + dlog,
        )
        .expect("blend_t +dT");
        let (_, _, _, km_t, _, _) = get_kap_results_blend_t(
            &cfg,
            0.7,
            0.02,
            0.0035,
            0.001,
            0.0086,
            0.0019,
            log_rho,
            log_t - dlog,
        )
        .expect("blend_t -dT");
        let fd_t =
            (safe_log10(kp_t) - safe_log10(km_t)) * LN10 / (2.0 * eps_ln);
        assert!(
            (dln_dt - fd_t).abs() <= 1.0e-6 * fd_t.abs().max(1.0),
            "blend_t dlnkap_dlnT={dln_dt} vs FD={fd_t}"
        );

        let (_, _, _, kp_r, _, _) = get_kap_results_blend_t(
            &cfg,
            0.7,
            0.02,
            0.0035,
            0.001,
            0.0086,
            0.0019,
            log_rho + dlog,
            log_t,
        )
        .expect("blend_t +drho");
        let (_, _, _, km_r, _, _) = get_kap_results_blend_t(
            &cfg,
            0.7,
            0.02,
            0.0035,
            0.001,
            0.0086,
            0.0019,
            log_rho - dlog,
            log_t,
        )
        .expect("blend_t -drho");
        let fd_r =
            (safe_log10(kp_r) - safe_log10(km_r)) * LN10 / (2.0 * eps_ln);
        assert!(
            (dln_dr - fd_r).abs() <= 1.0e-6 * fd_r.abs().max(1.0),
            "blend_t dlnkap_dlnRho={dln_dr} vs FD={fd_r}"
        );
        assert!(kap > 0.0);

        // Verify Type 2 `frac_dZ` exact order (`kap/private/kap_eval.f90:577-588`).
        let z = 0.025;
        let dz = z - cfg.zbase;
        let expected_frac_x = (cfg.kap_type2_full_off_x - 5.0e-4)
            / (cfg.kap_type2_full_off_x - cfg.kap_type2_full_on_x);
        let expected_frac_dz = (cfg.kap_type2_full_off_dz - dz)
            / (cfg.kap_type2_full_off_dz - cfg.kap_type2_full_on_dz);
        let expected_f_t2 = expected_frac_x * expected_frac_dz;
        let (f_t2, kap_ht, _, _) = get_kap_high_t_results(
            &cfg, 5.0e-4, z, 0.005, 0.001, 0.012, 0.002, -4.0, 6.5,
        )
        .expect("type2 highT eval");
        assert_eq!(f_t2.to_bits(), expected_f_t2.to_bits());
        assert!(kap_ht >= exp10(-30.0));
    });
}
