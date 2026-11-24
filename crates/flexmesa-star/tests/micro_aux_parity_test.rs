// SPDX-License-Identifier: LGPL-3.0-or-later
//! Exact Model-1 zone-level IEEE-hex parity tests for issue `#464` across
//! `do_eos`, `do_kap`, and `do_net` (`Peos`, `opacity`, `eps_nuc`, `grada`,
//! `chiRho`, `chiT`, `Cp`).

#![forbid(unsafe_code)]

use flexmesa_chem::NuclideCatalog;
use flexmesa_eos::EosConfig;
use flexmesa_kap::{KapCoOption, KapConfig, KapLowTOption, KapOption};
use flexmesa_math::{with_numerics_mode, NumericsMode};
use flexmesa_rates::ScreeningMode;
use flexmesa_star::micro::{
    do_eos, do_kap, do_net, resolve_model_net, set_basic_vars,
};
use flexmesa_star::{StarControls, StarJobControls, StarModel};

#[derive(Clone, Copy)]
struct ExpectedAux {
    peos: u64,
    opacity: u64,
    eps_nuc: Option<u64>,
    grada: Option<u64>,
    chi_rho: u64,
    chi_t: u64,
    cp: u64,
}

fn ulp_distance(a: f64, b: f64) -> u64 {
    let ba = a.to_bits();
    let bb = b.to_bits();
    if (ba >> 63) == (bb >> 63) {
        ba.abs_diff(bb)
    } else {
        (ba & 0x7fff_ffff_ffff_ffff).saturating_add(bb & 0x7fff_ffff_ffff_ffff)
    }
}

fn check_channel(
    case: &str,
    name: &str,
    actual: f64,
    expected_bits: u64,
    failures: &mut Vec<String>,
) {
    let actual_bits = actual.to_bits();
    if actual_bits != expected_bits {
        let expected = f64::from_bits(expected_bits);
        let ulps = ulp_distance(actual, expected);
        failures.push(format!(
            "{case} {name}: MESA {expected:.17e} ({expected_bits:016X}) vs Flex {actual:.17e} ({actual_bits:016X}) [{ulps} ULP]"
        ));
    }
}

#[allow(clippy::too_many_arguments)]
fn build_and_eval_zone(
    species_names: &[&str],
    net_name: &str,
    lnd_bits: u64,
    ln_t_bits: u64,
    ln_radius_bits: u64,
    xa_bits: &[u64],
    mut job: StarJobControls,
    controls: StarControls,
    eos_config: &EosConfig,
    kap_config: KapConfig,
    run_net: bool,
) -> StarModel {
    assert_eq!(species_names.len(), xa_bits.len());
    let catalog = NuclideCatalog::builtin_ref();
    let chem_id: Vec<usize> = species_names
        .iter()
        .map(|&name| {
            catalog
                .get_nuclide_index(name)
                .unwrap_or_else(|| panic!("unknown nuclide {name}"))
        })
        .collect();
    job.net_name = String::from(net_name);

    let mut model = StarModel::new(1, chem_id, job, controls);
    model.eos_config = *eos_config;
    model.kap_config = kap_config;
    model.lnd[0] = f64::from_bits(lnd_bits);
    model.ln_t[0] = f64::from_bits(ln_t_bits);
    model.ln_r[0] = f64::from_bits(ln_radius_bits);
    for (j, &bits) in xa_bits.iter().enumerate() {
        model.xa[j] = f64::from_bits(bits);
    }

    with_numerics_mode(NumericsMode::Strict, || {
        set_basic_vars(&mut model).expect("set_basic_vars");
        do_eos(&mut model).expect("do_eos");
        do_kap(&mut model).expect("do_kap");
        if run_net {
            let net_info =
                resolve_model_net(&model).expect("resolve_model_net");
            do_net(&mut model, &net_info, true).expect("do_net");
        }
    });

    model
}

fn assert_aux_parity(case: &str, model: &StarModel, exp: ExpectedAux) {
    let mut failures = Vec::new();
    check_channel(case, "Peos", model.peos[0], exp.peos, &mut failures);
    check_channel(
        case,
        "opacity",
        model.opacity[0],
        exp.opacity,
        &mut failures,
    );
    if let Some(eps) = exp.eps_nuc {
        check_channel(case, "eps_nuc", model.eps_nuc[0], eps, &mut failures);
    }
    if let Some(grada) = exp.grada {
        check_channel(case, "grada", model.grada[0], grada, &mut failures);
    }
    check_channel(case, "chiRho", model.chi_rho[0], exp.chi_rho, &mut failures);
    check_channel(case, "chiT", model.chi_t[0], exp.chi_t, &mut failures);
    check_channel(case, "Cp", model.cp[0], exp.cp, &mut failures);
    assert!(
        failures.is_empty(),
        "0-ULP AUX mismatches in {case}:\n  {}",
        failures.join("\n  ")
    );
}

const BASIC_NET_NAMES: &[&str] =
    &["h1", "he3", "he4", "c12", "n14", "o16", "ne20", "mg24"];

#[test]
fn test_make_pre_ccsn_13bvn_model3823_z1_aux_parity() {
    let species = &[
        "neut", "h1", "prot", "he3", "he4", "c12", "n14", "o16", "ne20",
        "mg24", "si28", "s32", "ar36", "ca40", "ti44", "cr48", "cr60", "fe52",
        "fe54", "fe56", "co56", "ni56",
    ];
    let xa_bits: &[u64] = &[
        0x2B61_AAF1_2333_F941,
        0x2E57_E6D0_4289_418B,
        0x2B61_AAF1_2333_F941,
        0x3E4F_529B_E809_85D0,
        0x3FEF_608A_DB48_FE64,
        0x3F22_F48C_3193_7578,
        0x3F89_84E5_D278_D45E,
        0x3F46_DC48_4533_C4AD,
        0x3F61_149C_F6C7_EC05,
        0x3F70_B346_E6A4_9EC2,
        0x2B61_7F7D_4ED8_C340,
        0x2B61_AAF1_2333_F941,
        0x2B61_AAF1_2333_F941,
        0x2B61_AAF1_2333_F941,
        0x2B61_AAF1_2333_F941,
        0x2B61_AAF1_2333_F941,
        0x2B61_AAF1_2333_F941,
        0x2B61_AAF1_2333_F941,
        0x2B61_AAF1_2333_F941,
        0x2C03_6587_6125_38D4,
        0x2B7B_4B20_AF3E_B522,
        0x2B62_240C_748A_EA81,
    ];
    let controls = StarControls {
        zbase: 0.02,
        screening_mode: ScreeningMode::Chugunov,
        op_split_burn: true,
        op_split_burn_min_t: 4.2e9,
        op_split_burn_eps: 1.0e-5,
        op_split_burn_odescal: 1.0e-5,
        ..StarControls::default()
    };
    let kap_config = KapConfig {
        zbase: 0.02,
        kap_option: KapOption::Gs98,
        kap_co_option: KapCoOption::Gs98Co,
        kap_low_t_option: KapLowTOption::LowTFa05Gs98,
        use_type2_opacities: true,
        ..KapConfig::default()
    };
    let model = build_and_eval_zone(
        species,
        "approx21_cr60_plus_co56.net",
        0xC034_8474_11BC_8F3D,
        0x4025_51B3_4757_F0F3,
        0x403A_78E4_C67C_4668,
        xa_bits,
        StarJobControls::default(),
        controls,
        &EosConfig::default(),
        kap_config,
        true,
    );
    assert_aux_parity(
        "star/make_pre_ccsn_13bvn_z1",
        &model,
        ExpectedAux {
            peos: 0x40C6_6974_C0B6_B6FC,
            opacity: 0x3FEE_FAAB_4A11_A4B8,
            eps_nuc: Some(0x2E66_D732_D892_A564),
            grada: None,
            chi_rho: 0x3FD1_6EE5_FC9D_330A,
            chi_t: 0x4009_E085_F74A_0E16,
            cp: 0x4203_D3C8_7EF6_939A,
        },
    );
}

#[test]
fn test_astero_adipls_and_gyre_z2824_aux_parity() {
    let xa_bits: &[u64] = &[
        0x3FE6_6469_04F5_E628,
        0x3F2F_E798_E1BB_F16C,
        0x3FD1_EB7F_75B4_EAD8,
        0x3F6C_3239_6738_C0B7,
        0x3F50_A5FB_2A31_1A4B,
        0x3F83_3619_3849_8D50,
        0x3F61_149C_F6C7_EC00,
        0x3F70_B346_E6A4_9EBA,
    ];
    let controls = StarControls {
        zbase: 0.02,
        ..StarControls::default()
    };
    let kap_config = KapConfig {
        zbase: 0.02,
        kap_option: KapOption::Gs98,
        ..KapConfig::default()
    };
    let model = build_and_eval_zone(
        BASIC_NET_NAMES,
        "basic.net",
        0x400B_70AE_00EB_2310,
        0x4030_1531_05C8_93FA,
        0x4037_859D_E41C_9B6F,
        xa_bits,
        StarJobControls::default(),
        controls,
        &EosConfig::default(),
        kap_config,
        true,
    );
    assert_aux_parity(
        "astero/astero_adipls_gyre_z2824",
        &model,
        ExpectedAux {
            peos: 0x4361_B616_2302_0984,
            opacity: 0x4000_5A41_86C5_EEDC,
            eps_nuc: Some(0x3FE8_8D73_0862_6EC8),
            grada: None,
            chi_rho: 0x3FF0_0FA0_39F4_E538,
            chi_t: 0x3FF0_0AD2_9F56_9A08,
            cp: 0x41B4_183B_ADA3_46EB,
        },
    );
}

#[test]
fn test_jdot_gr_check_z1223_aux_parity() {
    let xa_bits: &[u64] = &[
        0x3FE6_665E_7FC2_A0F9,
        0x3F01_9907_B6BA_1649,
        0x3FD1_EB08_23C2_2360,
        0x3F6C_3CA3_69CC_E918,
        0x3F50_8DAF_4208_07E1,
        0x3F83_3619_1BA4_2155,
        0x3F61_149C_F6C7_EC2D,
        0x3F70_B346_E6A4_9EF1,
    ];
    let controls = StarControls {
        zbase: 0.02,
        ..StarControls::default()
    };
    let kap_config = KapConfig {
        zbase: 0.02,
        ..KapConfig::default()
    };
    let model = build_and_eval_zone(
        BASIC_NET_NAMES,
        "basic.net",
        0x3FF6_8815_3CA7_706D,
        0x402D_0238_A14E_CB87,
        0x4037_E20B_5809_0760,
        xa_bits,
        StarJobControls::default(),
        controls,
        &EosConfig::default(),
        kap_config,
        true,
    );
    assert_aux_parity(
        "binary/jdot_gr_check_z1223",
        &model,
        ExpectedAux {
            peos: 0x430E_4F84_7526_D543,
            opacity: 0x4057_A0C8_4FAE_9BD0,
            eps_nuc: Some(0x3ED0_B3C7_C993_13A7),
            grada: None,
            chi_rho: 0x3FEF_F35A_034D_F322,
            chi_t: 0x3FF0_7043_0939_25A3,
            cp: 0x41B4_7C87_EB01_1E42,
        },
    );
}

#[test]
fn test_make_brown_dwarf_z388_aux_parity() {
    let xa_bits: &[u64] = &[
        0x3FE8_4F76_5FD8_ADAD,
        0x3EFA_CDC7_1D68_0D6B,
        0x3FCE_BE09_3606_FA8A,
        0x3EF2_1254_1AC8_1F32,
        0x3ED5_303C_7D57_B102,
        0x3F08_972A_60D2_12D5,
        0x3EE5_DCF1_DFB8_35A5,
        0x3EF5_605A_C0D2_B419,
    ];
    let controls = StarControls {
        zbase: 0.0001,
        ..StarControls::default()
    };
    let kap_config = KapConfig {
        zbase: 0.0001,
        kap_low_t_option: KapLowTOption::LowTFreedman11,
        ..KapConfig::default()
    };
    let model = build_and_eval_zone(
        BASIC_NET_NAMES,
        "basic.net",
        0xBFE2_ECF1_A992_F067,
        0x4027_FCEB_3B92_AF45,
        0x4037_0F35_6024_A1CA,
        xa_bits,
        StarJobControls::default(),
        controls,
        &EosConfig::default(),
        kap_config,
        true,
    );
    assert_aux_parity(
        "star/make_brown_dwarf_z388",
        &model,
        ExpectedAux {
            peos: 0x42A2_2AC9_C40F_D59D,
            opacity: 0x40B0_FC5A_894F_1126,
            eps_nuc: Some(0x3C10_F6AB_5BB6_59C6),
            grada: None,
            chi_rho: 0x3FF0_3A0B_8F26_38E8,
            chi_t: 0x3FF2_F78D_305D_4CC7,
            cp: 0x41B4_F1A3_C10D_9EE6,
        },
    );
}

#[test]
fn test_r_crb_star_z1089_aux_parity() {
    let species = &[
        "h1", "he3", "he4", "c12", "n14", "o16", "ne20", "mg24", "si28",
    ];
    let xa_bits: &[u64] = &[
        0x0000_0000_0000_0000,
        0x2B63_BB92_2FC3_A5F0,
        0x3FCA_DDA1_4076_777F,
        0x3FE5_A7EB_CE1E_728A,
        0x0000_0000_0000_0000,
        0x3FBB_062C_970A_24D4,
        0x3F7B_08D8_655D_3BCB,
        0x3F53_A93C_2FE0_BEAF,
        0x3CF2_454F_00B9_3903,
    ];
    let controls = StarControls {
        zbase: 0.006,
        ..StarControls::default()
    };
    let kap_config = KapConfig {
        zbase: 0.006,
        use_type2_opacities: true,
        kap_blend_log_t_lower_bdy: 4.1,
        kap_blend_log_t_upper_bdy: 4.2,
        kap_low_t_option: KapLowTOption::Aesopus,
        ..KapConfig::default()
    };
    let model = build_and_eval_zone(
        species,
        "co_burn.net",
        0x401F_51F2_8876_9E11,
        0x4033_12A2_6B80_603B,
        0x4035_42FE_7F49_8F5A,
        xa_bits,
        StarJobControls::default(),
        controls,
        &EosConfig::default(),
        kap_config,
        false,
    );
    assert_aux_parity(
        "star/R_CrB_star_z1089",
        &model,
        ExpectedAux {
            peos: 0x43F8_7760_FDD3_FB00,
            opacity: 0x3FC2_5BD5_69A4_81CD,
            eps_nuc: None,
            grada: None,
            chi_rho: 0x3FEC_42DB_5603_0314,
            chi_t: 0x3FF5_C1FC_40FB_BED2,
            cp: 0x41B1_3AA8_BF4C_FE17,
        },
    );
}

#[test]
fn test_make_he_wd_z954_aux_parity() {
    let species = &[
        "h1", "he3", "he4", "c12", "n14", "o16", "o18", "ne20", "ne22", "mg24",
    ];
    let xa_bits: &[u64] = &[
        0x3C27_C84D_F1ED_DDDA,
        0x0000_0000_0000_0000,
        0x3FEF_B01E_E94E_46FB,
        0x3F08_3502_F99E_5B5D,
        0x3F79_3DF5_9638_8669,
        0x3F3D_84E8_B028_C250,
        0x3EE6_2501_53CB_3B90,
        0x3F4F_9AD3_55CC_D9F4,
        0x3F14_7334_BE18_3FCE,
        0x3F60_B346_E6A4_A3BC,
    ];
    let controls = StarControls {
        zbase: 0.01,
        ..StarControls::default()
    };
    let kap_config = KapConfig {
        zbase: 0.01,
        kap_option: KapOption::Gs98,
        ..KapConfig::default()
    };
    let model = build_and_eval_zone(
        species,
        "o18_and_ne22.net",
        0x401F_82E0_B1AF_6D32,
        0x4031_0853_1354_0E37,
        0x4035_6615_A91D_8AF5,
        xa_bits,
        StarJobControls::default(),
        controls,
        &EosConfig::default(),
        kap_config,
        true,
    );
    assert_aux_parity(
        "star/make_he_wd_z954",
        &model,
        ExpectedAux {
            peos: 0x43CF_DF08_7CBA_376B,
            opacity: 0x3FEF_D216_D6D3_AF87,
            eps_nuc: Some(0x3D56_593C_DC4F_8589),
            grada: None,
            chi_rho: 0x3FF1_ED2B_A0B5_08D1,
            chi_t: 0x3FEA_A9E0_F735_934A,
            cp: 0x419F_75A9_217B_6B47,
        },
    );
}

#[test]
fn test_wd_stable_h_burn_z396_aux_parity() {
    let species = &[
        "h1", "he3", "he4", "c12", "c13", "n13", "n14", "n15", "o14", "o15",
        "o16", "o17", "o18", "f17", "f18", "f19", "ne18", "ne19", "ne20",
        "mg22", "mg24",
    ];
    let xa_bits: &[u64] = &[
        0x2B61_7F7D_4ED8_C33F,
        0x2B61_7F7D_4ED8_C33F,
        0x3FEF_61DD_2381_E44A,
        0x3F37_1D8A_45E6_9223,
        0x3F02_56D5_3856_3876,
        0x2B61_7F7D_4ED8_C33F,
        0x3F88_F26B_A8FD_07D6,
        0x3ED3_453D_7C64_65A6,
        0x2B61_7F7D_4ED8_C33F,
        0x2B61_7F7D_4ED8_C33F,
        0x3F3F_559D_3C58_9D1B,
        0x3ED1_0B54_D9BC_2627,
        0x3F0C_932D_FA19_36CF,
        0x2B61_7F7D_4ED8_C33F,
        0x2B61_7F7D_4ED8_C33F,
        0x2B61_7F7D_4ED8_C33F,
        0x2B61_7F7D_4ED8_C33F,
        0x2B61_7F7D_4ED8_C33F,
        0x3F61_0439_E5C6_C235,
        0x2B61_7F7D_4ED8_C33F,
        0x3F70_DC5F_5712_1090,
    ];
    let controls = StarControls {
        zbase: 0.02,
        net_logtcut_lo: 0.0,
        net_logtcut_lim: 0.0,
        ..StarControls::default()
    };
    let kap_config = KapConfig {
        zbase: 0.02,
        use_type2_opacities: true,
        use_zbase_for_type1: false,
        ..KapConfig::default()
    };
    let model = build_and_eval_zone(
        species,
        "cno_extras.net",
        0x4022_7F2F_E2A2_66EE,
        0x4031_4B15_1C7C_8D86,
        0x4034_1C83_4701_3C33,
        xa_bits,
        StarJobControls::default(),
        controls,
        &EosConfig::default(),
        kap_config,
        true,
    );
    assert_aux_parity(
        "star/wd_stable_h_burn_z396",
        &model,
        ExpectedAux {
            peos: 0x43F8_659D_EE35_1084,
            opacity: 0x3FEE_61DB_8E99_0501,
            eps_nuc: Some(0x3CC5_407E_7241_3046),
            grada: None,
            chi_rho: 0x3FF4_1CF7_058B_0455,
            chi_t: 0x3FE3_F74A_8B43_CE6C,
            cp: 0x4199_3F59_68FE_2C68,
        },
    );
}

#[test]
fn test_carbon_kh_z1_aux_parity() {
    let species = &["h1", "he4", "c12", "o16"];
    let xa_bits: &[u64] = &[
        0x2B61_7F7D_4ED8_C33E,
        0x2B71_7F7D_4ED8_C33E,
        0x3FF0_0000_0000_0000,
        0x2B81_7F7D_4ED8_C33E,
    ];
    let controls = StarControls {
        use_simple_es_for_kap: true,
        eps_nuc_factor: 0.0,
        dxdt_nuc_factor: 0.0,
        ..StarControls::default()
    };
    let eos_config = EosConfig {
        use_cms: false,
        use_opal_scvh: false,
        use_skye: false,
        use_pc: false,
        use_free_eos: false,
        ..EosConfig::default()
    };
    let model = build_and_eval_zone(
        species,
        "minimal.net",
        0xC031_9A1F_121B_FB4B,
        0x4024_9DBC_E52B_9ED0,
        0x4039_9875_82B2_CB85,
        xa_bits,
        StarJobControls::default(),
        controls,
        &eos_config,
        KapConfig::default(),
        false,
    );
    assert_aux_parity(
        "star/carbon_kh_z1",
        &model,
        ExpectedAux {
            peos: 0x40E0_FC2E_0584_B2DF,
            opacity: 0x3FC9_9999_9999_999A,
            eps_nuc: None,
            grada: Some(0x3FD6_467E_F110_2643),
            chi_rho: 0x3FEE_098D_1F09_26F9,
            chi_t: 0x3FF2_F1AC_6BDD_D0DD,
            cp: 0x41A6_1CAD_2EEF_B6A5,
        },
    );
}
