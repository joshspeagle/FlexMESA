// SPDX-License-Identifier: LGPL-3.0-or-later
//! Level-2 IEEE-754 module-hex parity tests for `flexmesa-kap` against MESA
//! `26.4.1` `kap/test/test_output` (`exact_tested_values.tsv`, 21 outputs) and
//! star model-1 `kap_get` / `d_opacity_dlnd` / `d_opacity_dlnT` hex oracles.

#![forbid(unsafe_code)]
#![allow(
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use flexmesa_chem::NuclideCatalog;
use flexmesa_kap::{
    get_kap_results, kap_get, KapCoOption, KapConfig, KapLowTOption, KapOption,
    I_FRAC_TYPE2,
};
use flexmesa_math::{with_numerics_mode, NumericsMode};
use flexmesa_utils::require_zero_skips_enabled;

const OUTPUT_LABELS: [&str; 7] = [
    "log10kap",
    "dlnkap_dlnRho",
    "dlnkap_dlnT",
    "kap",
    "dkap_dlnd",
    "dkap_dlnT",
    "frac_Type2",
];

fn resolve_kap_module_hex_tsv() -> Option<PathBuf> {
    let base = if let Some(val) =
        std::env::var_os("FLEXMESA_MODULE_HEX_DIR").filter(|v| !v.is_empty())
    {
        PathBuf::from(val)
    } else if let Some(xdg) =
        std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty())
    {
        PathBuf::from(xdg).join("flexmesa").join("mesa_module_hex")
    } else if let Some(home) =
        std::env::var_os("HOME").filter(|v| !v.is_empty())
    {
        PathBuf::from(home)
            .join(".cache")
            .join("flexmesa")
            .join("mesa_module_hex")
    } else {
        PathBuf::from(".flexmesa_module_hex")
    };
    let path = base.join("kap").join("exact_tested_values.tsv");
    if path.is_file() {
        return Some(path);
    }
    let has_env = std::env::var_os("FLEXMESA_MODULE_HEX_DIR")
        .is_some_and(|v| !v.is_empty());
    assert!(
        !has_env && !require_zero_skips_enabled(),
        "missing required MESA 26.4.1 kap module-hex TSV `{}` (set \
         FLEXMESA_MODULE_HEX_DIR, run `flexmesa data fetch --include-test-inputs`, \
         or set FLEXMESA_REQUIRE_ZERO_SKIPS=0 locally)",
        path.display()
    );
    None
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

fn parse_raw_hex_bits(raw: &str) -> u64 {
    let hex_str = raw.trim().trim_start_matches("0x").trim_start_matches("0X");
    u64::from_str_radix(hex_str, 16)
        .unwrap_or_else(|e| panic!("invalid IEEE-754 hex `{raw}`: {e}"))
}

#[derive(Clone, Copy)]
struct KapCaseCfg {
    name: &'static str,
    kap_option: KapOption,
    kap_co_option: KapCoOption,
    kap_low_t_option: KapLowTOption,
    use_type2_opacities: bool,
    use_zbase_for_type1: bool,
    eta: f64,
}

const CASE_CFGS: [KapCaseCfg; 3] = [
    KapCaseCfg {
        name: "test1_fixed_metals",
        kap_option: KapOption::Gs98,
        kap_co_option: KapCoOption::Gn93Co,
        kap_low_t_option: KapLowTOption::LowTFa05Gs98,
        use_type2_opacities: false,
        use_zbase_for_type1: false,
        eta: -10.0,
    },
    KapCaseCfg {
        name: "test2_co_enhanced",
        kap_option: KapOption::Gs98,
        kap_co_option: KapCoOption::Gn93Co,
        kap_low_t_option: KapLowTOption::LowTFa05Gs98,
        use_type2_opacities: true,
        use_zbase_for_type1: false,
        eta: -10.0,
    },
    KapCaseCfg {
        name: "test3_aesopus",
        kap_option: KapOption::A09,
        kap_co_option: KapCoOption::A09Co,
        kap_low_t_option: KapLowTOption::Aesopus,
        use_type2_opacities: true,
        use_zbase_for_type1: true,
        eta: -20.0,
    },
];

struct EmbeddedKapCase {
    cfg: KapCaseCfg,
    log_t_bits: u64,
    log_rho_bits: u64,
    z_bits: u64,
    zbase_bits: u64,
    zbar_bits: u64,
    x_bits: u64,
    xc_bits: u64,
    xn_bits: u64,
    xo_bits: u64,
    xne_bits: u64,
    lnfree_e_bits: u64,
    expected_bits: [u64; 7],
}

const EMBEDDED_CASES: [EmbeddedKapCase; 3] = [
    EmbeddedKapCase {
        cfg: CASE_CFGS[0],
        log_t_bits: 0x4018_0000_0000_0000,
        log_rho_bits: 0xc018_0000_0000_0000,
        z_bits: 0x3f92_6e97_8d4f_df40,
        zbase_bits: 0x0000_0000_0000_0000,
        zbar_bits: 0x3ff1_a152_f1fd_efa8,
        x_bits: 0x3fe6_6666_6666_6666,
        xc_bits: 0x0000_0000_0000_0000,
        xn_bits: 0x0000_0000_0000_0000,
        xo_bits: 0x0000_0000_0000_0000,
        xne_bits: 0x0000_0000_0000_0000,
        lnfree_e_bits: 0xbfc5_b71d_85e9_65cd,
        expected_bits: [
            0xbfdc_5048_16f0_5463,
            0x3f97_5d45_a608_4317,
            0x3fa2_1e40_a689_2363,
            0x3fd7_1be3_5062_3f4a,
            0x3f80_df66_ba71_02f6,
            0x3f8a_2b11_64b9_e93b,
            0x0000_0000_0000_0000,
        ],
    },
    EmbeddedKapCase {
        cfg: CASE_CFGS[1],
        log_t_bits: 0x4018_0000_0000_0000,
        log_rho_bits: 0xc018_0000_0000_0000,
        z_bits: 0x3fad_b22d_0e56_0420,
        zbase_bits: 0x3f92_6e97_8d4f_df3b,
        zbar_bits: 0x4000_97d2_5d38_53db,
        x_bits: 0x0000_0000_0000_0000,
        xc_bits: 0x3f98_b2d0_07a5_f7dd,
        xn_bits: 0x3f4f_59ae_ea5b_1f82,
        xo_bits: 0x3f9c_58f9_b01e_8514,
        xne_bits: 0x3f5d_192f_3fcd_643b,
        lnfree_e_bits: 0xbfe6_4d31_896f_08f8,
        expected_bits: [
            0xbfe5_40dc_c464_631e,
            0x3fa2_aea3_22f6_1f9f,
            0xbf93_8a1d_0f38_5437,
            0x3fcb_bc63_ad92_f592,
            0x3f80_3155_d58b_b2ed,
            0xbf70_ef90_ccd7_4245,
            0x3ff0_0000_0000_0000,
        ],
    },
    EmbeddedKapCase {
        cfg: CASE_CFGS[2],
        log_t_bits: 0x400c_750b_4c2c_8ffb,
        log_rho_bits: 0xc020_7fd1_3769_ccb8,
        z_bits: 0x3f94_8e4f_6fd7_6f00,
        zbase_bits: 0x3f8c_ac08_3126_e979,
        zbar_bits: 0x3ff1_9c8c_58b7_9d17,
        x_bits: 0x3fe5_b970_71a1_ff29,
        xc_bits: 0x3f68_8cd7_6039_7678,
        xn_bits: 0x3f59_0514_6aeb_0d98,
        xo_bits: 0x3f83_1f9a_fd78_f97f,
        xne_bits: 0x0000_0000_0000_0000,
        lnfree_e_bits: 0xc022_6bb2_01fe_476f,
        expected_bits: [
            0xc004_2fe2_d57e_3897,
            0x3fe2_54b6_a1b2_7d8f,
            0x4020_ecd4_642f_f87f,
            0x3f68_8c2b_f0e4_eb01,
            0x3f5c_1fa9_883a_6092,
            0x3f99_f785_5401_ff07,
            0x0000_0000_0000_0000,
        ],
    },
];

fn check_case_outputs(
    case: &EmbeddedKapCase,
    out_labels: [&str; 7],
    mismatches: &mut Vec<String>,
) {
    let cfg = KapConfig {
        zbase: f64::from_bits(case.zbase_bits),
        kap_option: case.cfg.kap_option,
        kap_co_option: case.cfg.kap_co_option,
        kap_low_t_option: case.cfg.kap_low_t_option,
        cubic_interpolation_in_x: false,
        cubic_interpolation_in_z: false,
        include_electron_conduction: true,
        use_zbase_for_type1: case.cfg.use_zbase_for_type1,
        use_type2_opacities: case.cfg.use_type2_opacities,
        ..KapConfig::default()
    };

    let out = get_kap_results(
        &cfg,
        f64::from_bits(case.zbar_bits),
        f64::from_bits(case.x_bits),
        f64::from_bits(case.z_bits),
        f64::from_bits(case.xc_bits),
        f64::from_bits(case.xn_bits),
        f64::from_bits(case.xo_bits),
        f64::from_bits(case.xne_bits),
        f64::from_bits(case.log_rho_bits),
        f64::from_bits(case.log_t_bits),
        f64::from_bits(case.lnfree_e_bits),
        0.0,
        0.0,
        case.cfg.eta,
        0.0,
        0.0,
    )
    .unwrap_or_else(|e| panic!("{} failed: {e}", case.cfg.name));

    let actual = [
        out.log10_kap,
        out.dlnkap_dln_rho,
        out.dlnkap_dln_t,
        out.kap,
        out.kap * out.dlnkap_dln_rho,
        out.kap * out.dlnkap_dln_t,
        out.kap_fracs[I_FRAC_TYPE2],
    ];

    for (j, (&act_val, &exp_bits)) in
        actual.iter().zip(case.expected_bits.iter()).enumerate()
    {
        let ulp = ulp_distance(act_val, exp_bits);
        if ulp != 0 {
            mismatches.push(format!(
                "{}.{}: got 0x{:016x} ({act_val:.17e}), \
                 want 0x{exp_bits:016x} ({:.17e}), ulp={ulp}",
                case.cfg.name,
                out_labels[j],
                act_val.to_bits(),
                f64::from_bits(exp_bits),
            ));
        }
    }
}

#[test]
fn test_kap_module_hex_tsv_21_outputs_zero_ulp() {
    let Some(tsv_path) = resolve_kap_module_hex_tsv() else {
        return;
    };
    let text = fs::read_to_string(&tsv_path).expect("read kap TSV");
    let mut rows: Vec<(String, String)> = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        if idx == 0 || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert!(
            cols.len() >= 8,
            "unexpected TSV column count at line {}: {}",
            idx + 1,
            cols.len()
        );
        rows.push((cols[3].to_string(), cols[7].to_string()));
    }
    assert_eq!(
        rows.len(),
        54,
        "expected 54 rows (3 cases x 18 rows) in kap TSV"
    );

    with_numerics_mode(NumericsMode::Strict, || {
        let mut mismatches = Vec::new();
        for (case_idx, &case_cfg) in CASE_CFGS.iter().enumerate() {
            let blk = &rows[case_idx * 18..(case_idx + 1) * 18];
            let inputs: HashMap<&str, u64> = blk[..11]
                .iter()
                .map(|(lbl, hex)| (lbl.as_str(), parse_raw_hex_bits(hex)))
                .collect();
            let mut expected_bits = [0_u64; 7];
            let mut out_labels = [""; 7];
            for (j, (lbl, hex)) in blk[11..18].iter().enumerate() {
                out_labels[j] = lbl.as_str();
                expected_bits[j] = parse_raw_hex_bits(hex);
            }
            let case = EmbeddedKapCase {
                cfg: case_cfg,
                log_t_bits: inputs["logT"],
                log_rho_bits: inputs["logRho"],
                z_bits: inputs["Z"],
                zbase_bits: inputs["Zbase"],
                zbar_bits: inputs["zbar"],
                x_bits: inputs["X"],
                xc_bits: inputs["xc"],
                xn_bits: inputs["xn"],
                xo_bits: inputs["xo"],
                xne_bits: inputs["xne"],
                lnfree_e_bits: inputs["lnfree_e"],
                expected_bits,
            };
            check_case_outputs(&case, out_labels, &mut mismatches);
        }
        assert!(
            mismatches.is_empty(),
            "kap module-hex TSV ULP mismatches ({}/21 non-zero):\n{}",
            mismatches.len(),
            mismatches.join("\n")
        );
    });
}

#[test]
fn test_kap_module_hex_embedded_21_outputs_zero_ulp() {
    with_numerics_mode(NumericsMode::Strict, || {
        let mut mismatches = Vec::new();
        for case in &EMBEDDED_CASES {
            check_case_outputs(case, OUTPUT_LABELS, &mut mismatches);
        }
        assert!(
            mismatches.is_empty(),
            "embedded kap module-hex ULP mismatches ({}/21 non-zero):\n{}",
            mismatches.len(),
            mismatches.join("\n")
        );
    });
}

struct StarZoneOracle {
    label: &'static str,
    isos: &'static [&'static str],
    xa_bits: &'static [u64],
    log_rho_bits: u64,
    log_t_bits: u64,
    lnfree_e_bits: u64,
    d_lnfree_e_dlnd_bits: u64,
    d_lnfree_e_dlnt_bits: u64,
    eta_bits: u64,
    d_eta_dlnd_bits: u64,
    d_eta_dlnt_bits: u64,
    kap_bits: u64,
    dlnkap_dlnd_bits: u64,
    dlnkap_dlnt_bits: u64,
    d_op_dlnd_bits: u64,
    d_op_dlnt_bits: u64,
}

const ISOS_1M: [&str; 8] =
    ["h1", "he3", "he4", "c12", "n14", "o16", "ne20", "mg24"];
const XA_1M: [u64; 8] = [
    0x3fe6_6666_6666_6667,
    0x3eff_3ebd_e705_8cc3,
    0x3fd1_eb08_23c0_b5d6,
    0x3f6c_3ca3_69d8_b430,
    0x3f50_8daf_41ec_844f,
    0x3f83_3619_1ba4_210f,
    0x3f61_149c_f6c7_ebff,
    0x3f70_b346_e6a4_9eb9,
];

const ISOS_COWD: [&str; 11] = [
    "h1", "he3", "he4", "c12", "n14", "o16", "o18", "ne20", "ne22", "mg24",
    "si28",
];
const XA_COWD: [u64; 11] = [
    0x3fe6_6666_6666_6667,
    0x3eff_3ebd_e705_8cc3,
    0x3fd1_eb08_23c0_b5d6,
    0x3f6c_3ca3_69d8_b430,
    0x3f50_8daf_41ec_844f,
    0x3f83_2b06_9afa_3b74,
    0x3ef6_2501_53cb_34fc,
    0x3f5f_9ad3_55cc_d0c6,
    0x3f24_7334_be18_39bc,
    0x3f49_eb88_14dc_ed84,
    0x3f6a_ebab_c812_0221,
];

const MODEL1_ZONES: [StarZoneOracle; 6] = [
    StarZoneOracle {
        label: "1M_pre_ms_to_wd_k1",
        isos: &ISOS_1M,
        xa_bits: &XA_1M,
        log_rho_bits: 0xc020_ac86_d26e_3314,
        log_t_bits: 0x400f_bb6e_27ff_0867,
        lnfree_e_bits: 0xc000_30c0_7398_7477,
        d_lnfree_e_dlnd_bits: 0xbfdc_8dae_758f_dfae,
        d_lnfree_e_dlnt_bits: 0x4020_991e_7d7c_5824,
        eta_bits: 0xc030_4762_6056_026e,
        d_eta_dlnd_bits: 0x3fe1_b928_ce29_1571,
        d_eta_dlnt_bits: 0x401b_323c_43f6_2f0d,
        kap_bits: 0x402c_9fd9_640c_1ecc,
        dlnkap_dlnd_bits: 0x3fcb_d0ab_15a6_49a6,
        dlnkap_dlnt_bits: 0x4027_1140_d7e9_e9a0,
        d_op_dlnd_bits: 0x4008_e187_7b03_dd28,
        d_op_dlnt_bits: 0x4064_a253_2c3b_de25,
    },
    StarZoneOracle {
        label: "1M_pre_ms_to_wd_k851",
        isos: &ISOS_1M,
        xa_bits: &XA_1M,
        log_rho_bits: 0xc016_0eed_3f27_dc3c,
        log_t_bits: 0x4011_70a5_cdf5_68db,
        lnfree_e_bits: 0xbfdd_d508_6348_0d69,
        d_lnfree_e_dlnd_bits: 0xbfb9_f340_7e96_3252,
        d_lnfree_e_dlnt_bits: 0x3fed_e619_f5fc_71ff,
        eta_bits: 0xc023_29af_b140_9157,
        d_eta_dlnd_bits: 0x3fec_c1c5_e919_d221,
        d_eta_dlnt_bits: 0xbfe2_1a12_30d6_9cd9,
        kap_bits: 0x40b3_bbbe_21b8_830b,
        dlnkap_dlnd_bits: 0x3feb_fb2a_98b5_f30f,
        dlnkap_dlnt_bits: 0xbfed_0a2f_0acc_835c,
        d_op_dlnd_bits: 0x40b1_414b_4c61_b506,
        d_op_dlnt_bits: 0xc0b1_e86b_fc61_cb3a,
    },
    StarZoneOracle {
        label: "1M_pre_ms_to_wd_k1702",
        isos: &ISOS_1M,
        xa_bits: &XA_1M,
        log_rho_bits: 0xc00a_5bdd_fa3c_6a57,
        log_t_bits: 0x4015_e86e_e278_2bff,
        lnfree_e_bits: 0xbfc6_3ab9_ce95_1964,
        d_lnfree_e_dlnd_bits: 0xbf42_96e5_2ba3_5b59,
        d_lnfree_e_dlnt_bits: 0x3f6d_09a0_47bd_4374,
        eta_bits: 0xc020_1250_1621_7635,
        d_eta_dlnd_bits: 0x3fef_fc4a_16cd_bde5,
        d_eta_dlnt_bits: 0xbff7_f292_521d_002d,
        kap_bits: 0x4069_5a95_b68e_d1b0,
        dlnkap_dlnd_bits: 0x3fe4_15e9_e945_16be,
        dlnkap_dlnt_bits: 0xc00b_9e60_919a_6d0b,
        d_op_dlnd_bits: 0x405f_d3f4_b17a_f2f1,
        d_op_dlnt_bits: 0xc085_e1ea_185b_d67d,
    },
    StarZoneOracle {
        label: "make_co_wd_k1",
        isos: &ISOS_COWD,
        xa_bits: &XA_COWD,
        log_rho_bits: 0xc020_b594_90d1_6c3a,
        log_t_bits: 0x400f_c85a_2961_e0f1,
        lnfree_e_bits: 0xbffe_31cd_1422_97e9,
        d_lnfree_e_dlnd_bits: 0xbfdc_0128_f482_36f4,
        d_lnfree_e_dlnt_bits: 0x4020_1073_c8f7_1f8c,
        eta_bits: 0xc030_3467_b8c3_9a7d,
        d_eta_dlnd_bits: 0x3fe1_ff6b_8f87_0cab,
        d_eta_dlnt_bits: 0x401a_20e6_d886_d9a9,
        kap_bits: 0x4030_b81d_1b76_f3f6,
        dlnkap_dlnd_bits: 0x3fcd_37ce_ca3e_b8d0,
        dlnkap_dlnt_bits: 0x4026_3498_57f7_0394,
        d_op_dlnd_bits: 0x400e_8805_bba2_beba,
        d_op_dlnt_bits: 0x4067_341d_955c_9339,
    },
    StarZoneOracle {
        label: "make_co_wd_k731",
        isos: &ISOS_COWD,
        xa_bits: &XA_COWD,
        log_rho_bits: 0xc018_7564_5fe5_48c4,
        log_t_bits: 0x4011_c192_22d1_403a,
        lnfree_e_bits: 0xbfd2_c73d_f161_b285,
        d_lnfree_e_dlnd_bits: 0xbf97_1157_d999_6155,
        d_lnfree_e_dlnt_bits: 0x3fcd_ce9e_47a3_d5c3,
        eta_bits: 0xc026_2071_c84c_6e16,
        d_eta_dlnd_bits: 0x3fef_4780_9dad_42a6,
        d_eta_dlnt_bits: 0xbff4_463c_b447_fc3e,
        kap_bits: 0x408e_a6c8_a11c_0326,
        dlnkap_dlnd_bits: 0x3ff0_f30f_1129_1420,
        dlnkap_dlnt_bits: 0xbffd_1e15_be14_2b2c,
        d_op_dlnd_bits: 0x4090_3c35_42b3_6d5e,
        d_op_dlnt_bits: 0xc09b_e3f7_019d_afac,
    },
    StarZoneOracle {
        label: "make_co_wd_k1462",
        isos: &ISOS_COWD,
        xa_bits: &XA_COWD,
        log_rho_bits: 0xc011_08f2_5364_beb2,
        log_t_bits: 0x4015_e86f_8561_5da8,
        lnfree_e_bits: 0xbfc6_2502_4df2_071d,
        d_lnfree_e_dlnd_bits: 0xbf25_1dac_48c0_41b6,
        d_lnfree_e_dlnt_bits: 0x3f59_328d_b817_73e4,
        eta_bits: 0xc024_825e_eaa1_b4cd,
        d_eta_dlnd_bits: 0x3fef_fec8_3bff_b70f,
        d_eta_dlnt_bits: 0xbff7_fa2a_7e63_92c8,
        kap_bits: 0x4045_5db7_3b76_f731,
        dlnkap_dlnd_bits: 0x3fe7_ed3a_dada_882b,
        dlnkap_dlnt_bits: 0xc00b_c412_b971_32f4,
        d_op_dlnd_bits: 0x403f_f382_2794_c054,
        d_op_dlnt_bits: 0xc062_89fd_1d01_0af2,
    },
];

#[test]
fn test_kap_get_model1_d_opacity_zero_ulp() {
    let catalog = NuclideCatalog::builtin_ref();
    let cfg = KapConfig {
        zbase: 0.02,
        kap_option: KapOption::Gs98,
        kap_co_option: KapCoOption::Gs98Co,
        kap_low_t_option: KapLowTOption::LowTFa05Gs98,
        cubic_interpolation_in_x: false,
        cubic_interpolation_in_z: false,
        include_electron_conduction: true,
        use_zbase_for_type1: true,
        use_type2_opacities: true,
        ..KapConfig::default()
    };

    with_numerics_mode(NumericsMode::Strict, || {
        let mut mismatches = Vec::new();
        for z in &MODEL1_ZONES {
            let chem_id: Vec<usize> = z
                .isos
                .iter()
                .map(|nm| {
                    catalog
                        .get_nuclide_index(nm)
                        .unwrap_or_else(|| panic!("unknown iso {nm}"))
                })
                .collect();
            let xa: Vec<f64> =
                z.xa_bits.iter().copied().map(f64::from_bits).collect();
            let out = kap_get(
                &cfg,
                &chem_id,
                &xa,
                f64::from_bits(z.log_rho_bits),
                f64::from_bits(z.log_t_bits),
                f64::from_bits(z.lnfree_e_bits),
                f64::from_bits(z.d_lnfree_e_dlnd_bits),
                f64::from_bits(z.d_lnfree_e_dlnt_bits),
                f64::from_bits(z.eta_bits),
                f64::from_bits(z.d_eta_dlnd_bits),
                f64::from_bits(z.d_eta_dlnt_bits),
            )
            .unwrap_or_else(|e| panic!("{} failed: {e}", z.label));

            let checks = [
                ("kap", out.kap, z.kap_bits),
                ("dlnkap_dlnd", out.dlnkap_dln_rho, z.dlnkap_dlnd_bits),
                ("dlnkap_dlnT", out.dlnkap_dln_t, z.dlnkap_dlnt_bits),
                (
                    "d_opacity_dlnd",
                    out.kap * out.dlnkap_dln_rho,
                    z.d_op_dlnd_bits,
                ),
                (
                    "d_opacity_dlnT",
                    out.kap * out.dlnkap_dln_t,
                    z.d_op_dlnt_bits,
                ),
            ];
            for (field, actual, exp_bits) in checks {
                let ulp = ulp_distance(actual, exp_bits);
                if ulp != 0 {
                    mismatches.push(format!(
                        "{}.{field}: got 0x{:016x}, want 0x{exp_bits:016x}, ulp={ulp}",
                        z.label,
                        actual.to_bits()
                    ));
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "model-1 kap_get ULP mismatches:\n{}",
            mismatches.join("\n")
        );
    });
}
