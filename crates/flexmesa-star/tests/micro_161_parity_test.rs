// SPDX-License-Identifier: GPL-3.0-or-later
//! Targeted `#161` microphysics parity tests for Train-7 strict near-misses
//! (`star/high_z`, `star/wd_c_core_ignition`, `star/ns_h`,
//! `star/check_pulse_atm`) and open `BOARD.md:131` `eosPT_get_T` probe.

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use flexmesa_atm::{
    eval_t_tau_varying, get_t_tau_base, AtmEosCallbackResult, AtmError,
    AtmKapCallbackResult, ATM_T_TAU_TRAMPEDACH_SOLAR,
};
use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_const::{AVO, CRAD, KERG, LN10, M_SUN, STANDARD_CGRAV};
use flexmesa_eos::{
    eos_dt_get, eos_dt_get_rho, eos_pt_get, eos_pt_get_t,
    eos_pt_get_with_linear, EosConfig, HelmTable, SkyeConfig, I_ETA,
    I_LN_FREE_E, I_LN_PGAS, I_LN_S, NUM_EOS_BASIC_RESULTS,
};
use flexmesa_kap::{kap_get, KapCoOption, KapConfig, KapLowTOption, KapOption};
use flexmesa_math::{exp, exp10, log, log10, with_numerics_mode, NumericsMode};
use flexmesa_net::derivs_support::fl_epsnuc_3alf_with_derivs;
use flexmesa_rates::{
    eval_fl_epsnuc_3alf, eval_fl_epsnuc_3alf_with_derivs, eval_reaction_raw,
    screen_pair, RatesCatalog, ScreeningMode, TFactors,
};
use flexmesa_star::controls::{AtmOption, AtmTTauOpacity};
use flexmesa_star::hydro_brunt_surf::eval_surf_bc;
use flexmesa_star::job_controls::AtmTTauRelation;
use flexmesa_star::micro::{
    do_eos, do_kap, do_net, do_neu, resolve_model_net, set_basic_vars,
};
use flexmesa_star::{StarControls, StarJobControls, StarModel};

#[derive(Clone, Copy)]
struct ExpectedAux {
    peos: u64,
    opacity: u64,
    eps_nuc: Option<u64>,
    grada: u64,
    chi_rho: u64,
    chi_t: u64,
    cp: u64,
}

fn signed_magnitude(bits: u64) -> i128 {
    if (bits >> 63) != 0 {
        -((bits ^ (1_u64 << 63)) as i128)
    } else {
        bits as i128
    }
}

fn ulp_distance(a: f64, b: f64) -> u64 {
    if a.is_nan() || b.is_nan() {
        return if a.is_nan() && b.is_nan() {
            0
        } else {
            u64::MAX
        };
    }
    let da = signed_magnitude(a.to_bits());
    let db = signed_magnitude(b.to_bits());
    u64::try_from((da - db).unsigned_abs()).unwrap_or(u64::MAX)
}

fn assert_ulp0(label: &str, actual: f64, expected_bits: u64) {
    let actual_bits = actual.to_bits();
    if actual_bits != expected_bits {
        let expected = f64::from_bits(expected_bits);
        let ulps = ulp_distance(actual, expected);
        panic!(
            "{label}: MESA {expected:.17e} ({expected_bits:016X}) vs \
             Flex {actual:.17e} ({actual_bits:016X}) [{ulps} ULP]"
        );
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
            "{case} {name}: MESA {expected:.17e} ({expected_bits:016X}) vs \
             Flex {actual:.17e} ({actual_bits:016X}) [{ulps} ULP]"
        ));
    }
}

fn build_and_eval_zone(
    species_names: &[&str],
    net_name: &str,
    lnd_bits: u64,
    ln_t_bits: u64,
    ln_r_bits: u64,
    xa_bits: &[u64],
    mut job: StarJobControls,
    controls: StarControls,
    eos_config: &EosConfig,
    kap_config: &KapConfig,
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
    model.kap_config = kap_config.clone();
    model.lnd[0] = f64::from_bits(lnd_bits);
    model.ln_t[0] = f64::from_bits(ln_t_bits);
    model.ln_r[0] = f64::from_bits(ln_r_bits);
    for (j, &bits) in xa_bits.iter().enumerate() {
        model.xa[j] = f64::from_bits(bits);
    }

    with_numerics_mode(NumericsMode::Strict, || {
        set_basic_vars(&mut model).expect("set_basic_vars");
        model.r[0] = exp(model.ln_r[0]);
        do_eos(&mut model).expect("do_eos");
        do_kap(&mut model).expect("do_kap");
        do_neu(&mut model).expect("do_neu");
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
    check_channel(case, "grada", model.grada[0], exp.grada, &mut failures);
    check_channel(case, "chiRho", model.chi_rho[0], exp.chi_rho, &mut failures);
    check_channel(case, "chiT", model.chi_t[0], exp.chi_t, &mut failures);
    check_channel(case, "Cp", model.cp[0], exp.cp, &mut failures);
    assert!(
        failures.is_empty(),
        "0-ULP AUX mismatches in {case}:\n  {}",
        failures.join("\n  ")
    );
}

#[test]
fn test_ns_h_model1_zones_1_527_1054_aux_parity() {
    let species = &[
        "h1", "he3", "he4", "c12", "c13", "n13", "n14", "n15", "o14", "o15",
        "o16", "o17", "o18", "f17", "f18", "f19", "ne18", "ne19", "ne20",
        "mg22", "mg24", "fe56",
    ];
    let eos_config = EosConfig {
        skye: SkyeConfig {
            use_simple_skye_blends: true,
            log_t_min_for_any_skye: 7.2,
            log_t_min_for_all_skye: 7.4,
            log_rho_min_for_any_skye: 4.9,
            log_rho_min_for_all_skye: 5.0,
            ..SkyeConfig::default()
        },
        ..EosConfig::default()
    };
    let kap_config = KapConfig {
        zbase: 0.02,
        kap_option: KapOption::Gs98,
        use_type2_opacities: true,
        ..KapConfig::default()
    };
    let controls = StarControls {
        zbase: 0.02,
        max_abar_for_burning: 55.0,
        ..StarControls::default()
    };

    // z=1 (surface H/He/Fe mixture)
    let mut xa_z1 = [0x2B5F_227D_248B_A076_u64; 22];
    xa_z1[2] = 0x3FBC_3EEC_58D8_8FDA;
    xa_z1[21] = 0x3FEC_7822_74E4_EE05;
    let m1 = build_and_eval_zone(
        species,
        "cno_extras_plus_fe56.net",
        0x4002_0035_3AA5_06E1,
        0x402E_E9CC_1486_3D66,
        0x402B_A147_29BD_40E5,
        &xa_z1,
        StarJobControls::default(),
        controls.clone(),
        &eos_config,
        &kap_config,
        true,
    );
    assert_aux_parity(
        "star/ns_h_m1_z1",
        &m1,
        ExpectedAux {
            peos: 0x431F_1E6F_3F4C_4EC4,
            opacity: 0x4017_C63A_FE21_BB52,
            eps_nuc: Some(0x0000_0000_0000_0000),
            grada: 0x3FD8_3766_57C3_1EFC,
            chi_rho: 0x3FEE_92C1_DC1E_F905,
            chi_t: 0x3FF2_54F9_2123_25D5,
            cp: 0x41A0_E4FF_4A20_004A,
        },
    );

    // z=527 (Skye/HELM envelope Fe56)
    let mut xa_fe = [0x2B61_7F7D_4ED8_C33E_u64; 22];
    xa_fe[21] = 0x3FF0_0000_0000_0000;
    let m527 = build_and_eval_zone(
        species,
        "cno_extras_plus_fe56.net",
        0x4024_DF24_F008_D876,
        0x4031_B189_7A7F_96CE,
        0x402B_A13F_2BA2_7EFF,
        &xa_fe,
        StarJobControls::default(),
        controls.clone(),
        &eos_config,
        &kap_config,
        true,
    );
    assert_aux_parity(
        "star/ns_h_m1_z527",
        &m527,
        ExpectedAux {
            peos: 0x4416_5906_8CA7_8151,
            opacity: 0x3FE8_9B66_1516_C7D3,
            eps_nuc: Some(0x0000_0000_0000_0000),
            grada: 0x3FD7_7E09_F24F_1B34,
            chi_rho: 0x3FF7_13CC_AC18_7560,
            chi_t: 0x3FDA_EE59_A703_130F,
            cp: 0x4187_BB11_1730_AD1E,
        },
    );

    // z=1054 (core Skye Fe56)
    let m1054 = build_and_eval_zone(
        species,
        "cno_extras_plus_fe56.net",
        0x402F_7746_550E_57B7,
        0x4033_0E9F_47B9_89E1,
        0x402B_A0CA_8E62_4D25,
        &xa_fe,
        StarJobControls::default(),
        controls,
        &eos_config,
        &kap_config,
        true,
    );
    assert_aux_parity(
        "star/ns_h_m1_z1054",
        &m1054,
        ExpectedAux {
            peos: 0x44D5_E55A_F009_5A52,
            opacity: 0x3F95_3D12_CD3C_58CD,
            eps_nuc: Some(0x0000_0000_0000_0000),
            grada: 0x3FD4_0C3E_1B00_1B58,
            chi_rho: 0x3FF7_8716_0442_4E5F,
            chi_t: 0x3F94_0A5F_A78E_6B57,
            cp: 0x416A_11DD_73C8_D9F3,
        },
    );
}

#[test]
fn test_high_z_model1_zones_1_5_875_aux_parity() {
    let species = &[
        "h1", "he3", "he4", "c12", "n14", "o16", "o18", "ne20", "ne22", "mg24",
    ];
    let controls = StarControls {
        zbase: 0.07,
        ..StarControls::default()
    };
    let kap_config = KapConfig {
        zbase: 0.07,
        use_type2_opacities: true,
        ..KapConfig::default()
    };

    let xa_z5: &[u64] = &[
        0x3FE1_9999_9999_999A,
        0x3F05_33B7_B831_7AE9,
        0x3FD8_5141_E760_F6C6,
        0x3F88_AB60_8177_E0CC,
        0x3F6C_E7EA_71FC_3B10,
        0x3FA0_BBAD_68C2_072C,
        0x3F14_E517_F069_B89C,
        0x3F7C_4085_9C67_3917,
        0x3F3D_8881_8D7E_89A8,
        0x3F8D_51C7_2A43_1EAD,
    ];
    let m5 = build_and_eval_zone(
        species,
        "o18_and_ne22.net",
        0xC035_09AB_43DD_0423,
        0x4023_ADC6_5A9E_CB0E,
        0x403A_A4E6_55C8_B244,
        xa_z5,
        StarJobControls::default(),
        controls.clone(),
        &EosConfig::default(),
        &kap_config,
        true,
    );
    assert_aux_parity(
        "star/high_z_m1_z5",
        &m5,
        ExpectedAux {
            peos: 0x409B_C302_D695_AEC2,
            opacity: 0x400A_2278_E18B_C38F,
            eps_nuc: Some(0x3A14_F28B_1EE2_AA68),
            grada: 0x3FD1_244C_51D1_9DA8,
            chi_rho: 0x3FEA_47E5_4AF6_696D,
            chi_t: 0x3FF9_327D_5A0C_F142,
            cp: 0x41CB_AD0C_7B2E_A53D,
        },
    );

    let xa_z875: &[u64] = &[
        0x3FE1_985C_FEC4_EDDB,
        0x3EF9_459A_57A6_352F,
        0x3FD8_5179_8A3E_D4A1,
        0x3F86_C6AC_B494_1B33,
        0x3F72_DF39_3D62_FF33,
        0x3FA0_BBA2_AC71_6B5C,
        0x3F14_E517_E815_643A,
        0x3F7C_4085_9C67_3909,
        0x3F3D_8881_8D7E_89A8,
        0x3F8D_51C7_2A43_1EA0,
    ];
    let m875 = build_and_eval_zone(
        species,
        "o18_and_ne22.net",
        0x3FFD_0D61_E5FF_5C06,
        0x4030_FF44_91C8_8E0D,
        0x4033_AB27_8325_43B6,
        xa_z875,
        StarJobControls::default(),
        controls,
        &EosConfig::default(),
        &kap_config,
        true,
    );
    assert_aux_parity(
        "star/high_z_m1_z875",
        &m875,
        ExpectedAux {
            peos: 0x4350_4D91_918B_1FD6,
            opacity: 0x3FDC_F809_2A58_AE1C,
            eps_nuc: Some(0x40D9_B6FE_A3AF_7624),
            grada: 0x3FD6_C73F_F335_29CC,
            chi_rho: 0x3FEE_83C6_03B7_49A4,
            chi_t: 0x3FF2_3CF5_9822_EC4A,
            cp: 0x41B8_D2D6_A0B5_E937,
        },
    );
}

#[test]
fn test_wd_c_core_ignition_model1_zones_1_894_aux_parity() {
    let species = &[
        "neut", "h1", "prot", "he3", "he4", "c12", "n14", "o16", "ne20",
        "mg24", "si28", "s32", "ar36", "ca40", "ti44", "cr48", "cr56", "fe52",
        "fe54", "fe56", "ni56",
    ];
    let controls = StarControls {
        zbase: 0.02,
        ..StarControls::default()
    };
    let kap_config = KapConfig {
        zbase: 0.02,
        use_type2_opacities: true,
        ..KapConfig::default()
    };
    let job = StarJobControls {
        use_3a_fl87: true,
        ..StarJobControls::default()
    };

    let xa_z1: &[u64] = &[
        0x2B61_7F7D_4ED8_C33E,
        0x2B61_7F7D_4ED8_C33E,
        0x2B61_7F7D_4ED8_C33E,
        0x3B36_4CC1_CEA0_4C92,
        0x3D69_45C2_6693_FA2A,
        0x3FE1_B748_BAD2_09FD,
        0x3BF6_DEA8_BAFD_C706,
        0x3FDA_ED13_3C2B_1AB0,
        0x3F96_13DC_54AA_42CD,
        0x3F4A_661F_3FF8_C369,
        0x3F6A_F53C_A2FF_1E73,
        0x2B61_7F7D_4ED8_C33E,
        0x2B61_7F7D_4ED8_C33E,
        0x2B61_7F7D_4ED8_C33E,
        0x2B61_7F7D_4ED8_C33E,
        0x2B61_7F7D_4ED8_C33E,
        0x2B61_7F7D_4ED8_C33E,
        0x2B61_7F7D_4ED8_C33E,
        0x2B61_7F7D_4ED8_C33E,
        0x2B61_7F7D_4ED8_C33E,
        0x2B61_7F7D_4ED8_C33E,
    ];
    let m1 = build_and_eval_zone(
        species,
        "approx21.net",
        0xC01E_7BF6_BCA7_4790,
        0x4026_5228_CA59_29DF,
        0x4034_313F_F467_E627,
        xa_z1,
        job.clone(),
        controls.clone(),
        &EosConfig::default(),
        &kap_config,
        true,
    );
    assert_aux_parity(
        "star/wd_c_core_ignition_m1_z1",
        &m1,
        ExpectedAux {
            peos: 0x41CD_31FF_B6A5_E404,
            opacity: 0x40E0_3AB3_C705_C48C,
            eps_nuc: Some(0x2DAF_752C_196A_696C),
            grada: 0x3FD1_42F9_3EEF_8E74,
            chi_rho: 0x3FEE_31E2_5B9B_5D4D,
            chi_t: 0x3FF7_8E34_D557_A70F,
            cp: 0x41A5_C53F_8307_A91C,
        },
    );

    let xa_z894: &[u64] = &[
        0x2B61_7F7D_4ED8_C33E,
        0x34B8_5353_1936_0ABE,
        0x2B61_7F7D_4ED8_C33E,
        0x39D6_C20C_07B1_E72C,
        0x38AD_BAD1_8995_0332,
        0x3FD6_966D_701A_3BE2,
        0x39F8_47E2_7816_FBB5,
        0x3FE3_E0F1_2B06_4246,
        0x3F96_4AB2_9317_50A0,
        0x3F4A_35C5_3C98_F3A8,
        0x3F6A_F517_04BF_06EB,
        0x2B61_7F7D_4ED8_C33E,
        0x2B61_7F7D_4ED8_C33E,
        0x2B61_7F7D_4ED8_C33E,
        0x2B61_7F7D_4ED8_C33E,
        0x2B61_7F7D_4ED8_C33E,
        0x2B61_7F7D_4ED8_C33E,
        0x2B61_7F7D_4ED8_C33E,
        0x2B61_7F7D_4ED8_C33E,
        0x2B61_7F7D_4ED8_C33E,
        0x2B61_7F7D_4ED8_C33E,
    ];
    let m894 = build_and_eval_zone(
        species,
        "approx21.net",
        0x4031_0C9C_E2F1_05FF,
        0x4031_C589_E888_D51B,
        0x402B_DAEE_DFE2_A121,
        xa_z894,
        job,
        controls,
        &EosConfig::default(),
        &kap_config,
        true,
    );
    assert_aux_parity(
        "star/wd_c_core_ignition_m1_z894",
        &m894,
        ExpectedAux {
            peos: 0x4504_C976_F66C_B39F,
            opacity: 0x3F06_B3A6_9DE4_7046,
            eps_nuc: Some(0x38EC_3CD2_C86D_3D41),
            grada: 0x3FD7_D4F4_E680_DEE6,
            chi_rho: 0x3FF6_9243_5722_5CD6,
            chi_t: 0x3F6C_18CA_5872_E92B,
            cp: 0x416D_8099_42A2_BDF4,
        },
    );
}

#[test]
fn test_check_pulse_atm_model1_z1_aux_and_surf_bc_probe() {
    let species = &["h1", "he3", "he4", "c12", "n14", "o16", "ne20", "mg24"];
    let xa_z1: &[u64] = &[
        0x3FE6_6666_6666_6667,
        0x3EFF_3EBD_E705_8C15,
        0x3FD1_EB08_23C0_B5D7,
        0x3F6C_3CA3_69D8_B381,
        0x3F50_8DAF_41EC_844F,
        0x3F83_3619_1BA4_210F,
        0x3F61_149C_F6C7_EBFF,
        0x3F70_B346_E6A4_9EB9,
    ];
    let mut controls = StarControls {
        zbase: 0.02,
        atm_option: AtmOption::TTau,
        atm_t_tau_relation: AtmTTauRelation::TrampedachSolar,
        atm_t_tau_opacity: AtmTTauOpacity::Varying,
        tau_factor: 0.1,
        ..StarControls::default()
    };
    controls.mix_wind.use_t_tau_gradr_factor = true;
    let kap_config = KapConfig {
        zbase: 0.02,
        ..KapConfig::default()
    };

    let catalog = NuclideCatalog::builtin_ref();
    let chem_id: Vec<usize> = species
        .iter()
        .map(|&name| catalog.get_nuclide_index(name).unwrap())
        .collect();
    let job = StarJobControls {
        net_name: String::from("basic.net"),
        ..StarJobControls::default()
    };

    let mut model = StarModel::new(1, chem_id, job, controls);
    model.kap_config = kap_config;
    model.m_center = 0.0;
    model.mstar = M_SUN;
    model.xmstar = M_SUN;
    model.q[0] = 1.0;
    model.dq[0] = 1.0;
    model.cgrav[0] = STANDARD_CGRAV;
    model.tau_base = get_t_tau_base(ATM_T_TAU_TRAMPEDACH_SOLAR).unwrap();
    model.lnd[0] = f64::from_bits(0xC02E_E0D3_BB2C_CBBA);
    model.ln_t[0] = f64::from_bits(0x4021_44A6_7676_DD89);
    model.ln_r[0] = f64::from_bits(0x4038_D92E_46DE_E199);
    model.lum[0] = f64::from_bits(0x46E0_BA80_3301_2DBB);
    for (j, &bits) in xa_z1.iter().enumerate() {
        model.xa[j] = f64::from_bits(bits);
    }

    with_numerics_mode(NumericsMode::Strict, || {
        set_basic_vars(&mut model).expect("set_basic_vars");
        model.r[0] = exp(model.ln_r[0]);
        do_eos(&mut model).expect("do_eos");
        do_kap(&mut model).expect("do_kap");
        do_neu(&mut model).expect("do_neu");
        let net_info = resolve_model_net(&model).expect("resolve_model_net");
        do_net(&mut model, &net_info, true).expect("do_net");
        eval_surf_bc(&mut model).expect("eval_surf_bc");
    });

    assert_aux_parity(
        "star/check_pulse_atm_m1_z1",
        &model,
        ExpectedAux {
            peos: 0x40F1_3871_88A5_506B,
            opacity: 0x3FD5_11B2_A874_475F,
            eps_nuc: Some(0x3AA1_E97D_E6D3_DA6C),
            grada: 0x3FD8_FD85_3E60_B8EB,
            chi_rho: 0x3FEF_FC0D_5CF2_C4AA,
            chi_t: 0x3FF0_13C6_C22A_9FAC,
            cp: 0x41A3_88F3_BA64_F844,
        },
    );
    // `eval_surf_bc` matches MESA on lnT_surf and Teff at 0 ULP; lnP_surf shows
    // the 239-ULP shift from `hydro_brunt_surf.rs:512` passing `log(pgas)` vs
    // MESA `atm_support.f90:1221` passing `log10(pgas) * ln10`.
    assert_ulp0(
        "check_pulse_atm eval_surf_bc.lnT_surf",
        model.ln_t_surf_ad.val,
        0x4020_EFCA_C731_8375,
    );
    assert_ulp0(
        "check_pulse_atm eval_surf_bc.Teff",
        model.teff,
        0x40B5_F066_5F78_3041,
    );
    assert_ulp0(
        "check_pulse_atm eval_surf_bc.lnP_surf (log(pgas) path)",
        model.ln_peos_surf_ad.val,
        0x4024_7529_4814_B3E7,
    );

    // Pin 0-ULP `eval_t_tau_varying` 579-callback parity against MESA 26.4.1
    // `atm_eval_T_tau_varying` using `atm_support.f90:1221` (`log10(pgas)*ln10`).
    let ht = HelmTable::builtin().expect("builtin HELM table");
    let comp = get_composition_info(catalog, &model.chem_id, &model.xa, true)
        .expect("comp");
    let abar = comp.abar;
    assert_ulp0("check_pulse_atm comp.abar", abar, 0x3FF4_BF0C_B7F5_65DD);
    let tau_surf = model.controls.tau_factor * model.tau_base;
    let l_surf = model.lum[0];
    let r_surf = model.r[0];
    let m_surf = model.m[0];
    let cgrav = model.cgrav[0];
    let errtol = model.controls.atm_t_tau_errtol;
    let max_steps = model.controls.atm_t_tau_max_steps;
    let teff = model.teff;

    let run_atm_variant =
        |use_log10_times_ln10: bool| -> (flexmesa_atm::AtmEvalOutput, usize) {
            let chem_id = &model.chem_id;
            let xa = &model.xa;
            let eos_cfg = &model.eos_config;
            let kap_cfg = &model.kap_config;
            let ncall = std::cell::Cell::new(0_usize);
            let eos_cb =
                |ln_p: f64,
                 ln_t: f64|
                 -> Result<AtmEosCallbackResult, AtmError> {
                    let p = exp(ln_p);
                    let t = exp(ln_t);
                    let log_t = ln_t / LN10;
                    let prad = (CRAD / 3.0) * t * t * t * t;
                    let pgas = (p - prad).max(1.0e-99);
                    let rho_g = (p / t) * abar / (AVO * KERG);
                    let log_rho_guess = log10(rho_g);
                    let target = if use_log10_times_ln10 {
                        log10(pgas) * LN10
                    } else {
                        log(pgas)
                    };
                    let inv = eos_dt_get_rho(
                        ht,
                        chem_id,
                        xa,
                        log_t,
                        I_LN_PGAS as isize,
                        target,
                        log_rho_guess,
                        1.0e-11,
                        1.0e-11 * LN10,
                        100,
                        eos_cfg,
                    )
                    .map_err(|e| {
                        AtmError::CallbackError {
                            context: "probe_eos_dt",
                            detail: format!("{e}"),
                        }
                    })?;
                    let mut res = [0.0_f64; NUM_EOS_BASIC_RESULTS];
                    let mut dres_dln_rho = [0.0_f64; NUM_EOS_BASIC_RESULTS];
                    let mut dres_dln_t = [0.0_f64; NUM_EOS_BASIC_RESULTS];
                    res.copy_from_slice(&inv.eos.res);
                    dres_dln_rho.copy_from_slice(&inv.eos.d_dlnd);
                    dres_dln_t.copy_from_slice(&inv.eos.d_dln_t);
                    ncall.set(ncall.get() + 1);
                    Ok(AtmEosCallbackResult {
                        ln_rho: inv.log10_result * LN10,
                        res,
                        dres_dln_rho,
                        dres_dln_t,
                    })
                };
            let kap_cb =
                |ln_rho: f64,
                 ln_t: f64,
                 eos: &AtmEosCallbackResult|
                 -> Result<AtmKapCallbackResult, AtmError> {
                    let ko = kap_get(
                        kap_cfg,
                        chem_id,
                        xa,
                        ln_rho / LN10,
                        ln_t / LN10,
                        eos.res[I_LN_FREE_E],
                        eos.dres_dln_rho[I_LN_FREE_E],
                        eos.dres_dln_t[I_LN_FREE_E],
                        eos.res[I_ETA],
                        eos.dres_dln_rho[I_ETA],
                        eos.dres_dln_t[I_ETA],
                    )
                    .map_err(|e| {
                        AtmError::CallbackError {
                            context: "probe_kap",
                            detail: format!("{e}"),
                        }
                    })?;
                    Ok(AtmKapCallbackResult {
                        kap: ko.kap,
                        dlnkap_dln_rho: ko.dlnkap_dln_rho,
                        dlnkap_dln_t: ko.dlnkap_dln_t,
                    })
                };
            let res_atm = eval_t_tau_varying(
                tau_surf,
                l_surf,
                r_surf,
                m_surf,
                cgrav,
                ATM_T_TAU_TRAMPEDACH_SOLAR,
                eos_cb,
                kap_cb,
                errtol,
                max_steps,
                false,
                teff,
            )
            .expect("eval_t_tau_varying probe");
            (res_atm, ncall.get())
        };

    with_numerics_mode(NumericsMode::Strict, || {
        let (out_mesa_arg, ncall_exact) = run_atm_variant(true);
        assert_eq!(ncall_exact, 579, "expected exact 579 MESA callbacks");
        assert_ulp0(
            "atm_t_tau_varying.ln_t",
            out_mesa_arg.ln_t,
            0x4020_EFCA_C731_8375,
        );
        assert_ulp0(
            "atm_t_tau_varying.ln_p",
            out_mesa_arg.ln_p,
            0x4024_7529_4814_B4D6,
        );
        assert_ulp0(
            "atm_t_tau_varying.dln_t_dl",
            out_mesa_arg.dln_t_dl,
            0x38DE_9B3E_12E2_C5A6,
        );
        assert_ulp0(
            "atm_t_tau_varying.dln_t_dln_r",
            out_mesa_arg.dln_t_dln_r,
            0xBFE0_0000_0000_0000,
        );
        assert_eq!(out_mesa_arg.dln_t_dln_m, 0.0);
        assert_eq!(out_mesa_arg.dln_t_dln_kap, 0.0);
        assert_eq!(out_mesa_arg.dln_p_dln_m, 0.0);
        assert_eq!(out_mesa_arg.dln_p_dln_kap, 0.0);
        let mesa_dlnp_dl = f64::from_bits(0xB8D9_B37F_0B44_2F72);
        let mesa_dlnp_dlnr = f64::from_bits(0x3FDA_DF13_9E0D_8BDC);
        assert!(
            ((out_mesa_arg.dln_p_dl - mesa_dlnp_dl) / mesa_dlnp_dl).abs()
                < 2.0e-11
        );
        assert!(
            ((out_mesa_arg.dln_p_dln_r - mesa_dlnp_dlnr) / mesa_dlnp_dlnr)
                .abs()
                < 2.0e-11
        );

        let (out_log_arg, ncall_log) = run_atm_variant(false);
        assert_eq!(ncall_log, 579);
        assert_eq!(
            out_log_arg.ln_p.to_bits(),
            model.ln_peos_surf_ad.val.to_bits(),
            "eval_surf_bc lnP_surf mismatch explained solely by log(pgas) vs log10(pgas)*ln10"
        );

        // Call 1 exact (lnRho, lnT) kap_get parity = 0 ULP across (kap, dkr, dkt).
        let lnr_mesa1 = f64::from_bits(0xC036_C24A_06C7_A652);
        let lnt1 = f64::from_bits(0x4020_7360_154A_9EDB);
        let eos1 = eos_dt_get(
            ht,
            exp(lnr_mesa1),
            exp(lnt1),
            &model.chem_id,
            &model.xa,
            &model.eos_config,
        )
        .expect("eos_dt_get call1");
        let ko1 = kap_get(
            &model.kap_config,
            &model.chem_id,
            &model.xa,
            lnr_mesa1 / LN10,
            lnt1 / LN10,
            eos1.res[I_LN_FREE_E],
            eos1.d_dlnd[I_LN_FREE_E],
            eos1.d_dln_t[I_LN_FREE_E],
            eos1.res[I_ETA],
            eos1.d_dlnd[I_ETA],
            eos1.d_dln_t[I_ETA],
        )
        .expect("kap_get call1");
        assert_ulp0(
            "call1.lnfree_e",
            eos1.res[I_LN_FREE_E],
            0xC022_6BEA_8BD9_1ABA,
        );
        assert_ulp0("call1.kap", ko1.kap, 0x3F41_4EB2_7CA3_AFC7);
        assert_ulp0("call1.dkr", ko1.dlnkap_dln_rho, 0x3FE2_5EBE_254A_496F);
        assert_ulp0("call1.dkt", ko1.dlnkap_dln_t, 0x4001_DE52_F83C_BFF1);
    });
}

fn assert_probe_point(label: &str, model: &StarModel, exp: [u64; 7]) {
    let got = [
        model.peos[0],
        model.opacity[0],
        model.grada[0],
        model.chi_rho[0],
        model.chi_t[0],
        model.cp[0],
        model.non_nuc_neu[0],
    ];
    let names = ["Peos", "kap", "grada", "chiRho", "chiT", "Cp", "neu"];
    let mut failures = Vec::new();
    for ((&name, &g), &eb) in names.iter().zip(&got).zip(&exp) {
        check_channel(label, name, g, eb, &mut failures);
    }
    assert!(
        failures.is_empty(),
        "0-ULP mid-run probe mismatches in {label}:\n  {}",
        failures.join("\n  ")
    );
}

#[test]
fn test_midrun_161_regimes_and_pycno_probe() {
    // 1. HIGHZ_M340
    let mut xa_hz = [
        0.0, 0.0, 0.803521, 0.108974, 0.0, 0.010230, 0.0, 0.006897, 0.0, 0.0,
    ];
    xa_hz[9] = 1.0 - xa_hz[..9].iter().sum::<f64>();
    let xa_hz_bits: Vec<u64> = xa_hz.iter().map(|v| v.to_bits()).collect();
    let hz_kap = KapConfig {
        zbase: 0.07,
        kap_option: KapOption::Gs98,
        kap_co_option: KapCoOption::Gs98Co,
        use_type2_opacities: true,
        ..KapConfig::default()
    };
    let m_hz = build_and_eval_zone(
        &[
            "h1", "he3", "he4", "c12", "n14", "o16", "o18", "ne20", "ne22",
            "mg24",
        ],
        "o18_and_ne22.net",
        (3.4921578275053786 * LN10).to_bits(),
        (8.169_043_301_529_7 * LN10).to_bits(),
        0x4030_0000_0000_0000,
        &xa_hz_bits,
        StarJobControls::default(),
        StarControls {
            zbase: 0.07,
            ..StarControls::default()
        },
        &EosConfig::default(),
        &hz_kap,
        false,
    );
    assert_probe_point(
        "HIGHZ_M340",
        &m_hz,
        [
            0x43F8_D3D5_A3C5_A3A7,
            0x3FC4_58DB_DC87_294D,
            0x3FD6_8E94_1D91_0702,
            0x3FEE_F93F_D5BD_2C01,
            0x3FF1_CDD3_A0C0_7DBF,
            0x41A8_485E_02B5_28B2,
            0x4016_B145_ACB1_DDAC,
        ],
    );

    // 2. WD_M370 and WD_M668
    let wd_kap = KapConfig {
        zbase: 0.02,
        kap_option: KapOption::Gs98,
        kap_co_option: KapCoOption::Gs98Co,
        use_type2_opacities: true,
        ..KapConfig::default()
    };
    let mut xa_370 = [0.0, 0.350611, 0.622309, 0.022977, 0.0];
    xa_370[4] = 1.0 - xa_370[..4].iter().sum::<f64>();
    let xa_370_bits: Vec<u64> = xa_370.iter().map(|v| v.to_bits()).collect();
    let m_370 = build_and_eval_zone(
        &["he4", "c12", "o16", "ne20", "si28"],
        "approx21.net",
        (9.893157 * LN10).to_bits(),
        (8.191366 * LN10).to_bits(),
        0x402B_0000_0000_0000,
        &xa_370_bits,
        StarJobControls::default(),
        StarControls {
            zbase: 0.02,
            ..StarControls::default()
        },
        &EosConfig::default(),
        &wd_kap,
        false,
    );
    assert_probe_point(
        "WD_M370",
        &m_370,
        [
            0x45B8_34E8_75E8_6ACE,
            0x3E7E_040A_DA7A_85E4,
            0x3FD8_5AF9_FD6E_9DD8,
            0x3FF5_5EA4_8F96_3D56,
            0x3F53_9FE0_A18C_8FBD,
            0x416B_B7E6_D649_C648,
            0x4014_F76C_6E4C_665A,
        ],
    );

    let mut xa_668 = [0.0, 0.419117, 0.549867, 0.025894, 0.0];
    xa_668[4] = 1.0 - xa_668[..4].iter().sum::<f64>();
    let xa_668_bits: Vec<u64> = xa_668.iter().map(|v| v.to_bits()).collect();
    let m_668 = build_and_eval_zone(
        &["he4", "c12", "o16", "ne20", "si28"],
        "approx21.net",
        (9.806680 * LN10).to_bits(),
        (8.595580 * LN10).to_bits(),
        0x402B_0000_0000_0000,
        &xa_668_bits,
        StarJobControls::default(),
        StarControls {
            zbase: 0.02,
            ..StarControls::default()
        },
        &EosConfig::default(),
        &wd_kap,
        false,
    );
    assert_probe_point(
        "WD_M668",
        &m_668,
        [
            0x45B2_98B8_7B6B_38BA,
            0x3EB6_1F4F_1D9D_846C,
            0x3FD9_12BE_B0B8_6E54,
            0x3FF5_5CCF_F6C7_F7CB,
            0x3F6C_9C5F_697D_A7A9,
            0x416D_0736_5FA0_87BB,
            0x40D1_E6CE_2BD5_047D,
        ],
    );

    // 3. NS_SKYE_HE4 and NS_SKYE_FE56
    let ns_eos = EosConfig {
        skye: SkyeConfig {
            use_simple_skye_blends: true,
            log_t_min_for_any_skye: 7.2,
            log_t_min_for_all_skye: 7.4,
            log_rho_min_for_any_skye: 4.9,
            log_rho_min_for_all_skye: 5.0,
            ..SkyeConfig::default()
        },
        ..EosConfig::default()
    };
    let m_ns_he4 = build_and_eval_zone(
        &["he4", "fe56"],
        "cno_extras_plus_fe56.net",
        (4.95 * LN10).to_bits(),
        (7.30 * LN10).to_bits(),
        0x402B_A147_29BD_40E5,
        &[1.0_f64.to_bits(), 0.0_f64.to_bits()],
        StarJobControls::default(),
        StarControls {
            zbase: 0.02,
            ..StarControls::default()
        },
        &ns_eos,
        &wd_kap,
        false,
    );
    assert_probe_point(
        "NS_SKYE_HE4",
        &m_ns_he4,
        [
            0x443E_C111_5130_1C29,
            0x3F9C_EDCA_9A30_7F9B,
            0x3FD8_C423_19EA_6319,
            0x3FF9_6F43_6825_F7C2,
            0x3FB7_259C_E817_4002,
            0x4186_5E58_9E54_030B,
            0x3F4D_A297_70DE_1CF1,
        ],
    );

    let m_ns_fe56 = build_and_eval_zone(
        &["he4", "fe56"],
        "cno_extras_plus_fe56.net",
        (6.99 * LN10).to_bits(),
        (7.30 * LN10).to_bits(),
        0x402B_A147_29BD_40E5,
        &[0.0_f64.to_bits(), 1.0_f64.to_bits()],
        StarJobControls::default(),
        StarControls {
            zbase: 0.02,
            ..StarControls::default()
        },
        &ns_eos,
        &wd_kap,
        false,
    );
    assert_probe_point(
        "NS_SKYE_FE56",
        &m_ns_fe56,
        [
            0x44E2_6767_3767_8DD7,
            0x3F07_D801_CCB6_D791,
            0x3FD5_A4A2_D17A_076A,
            0x3FF7_5F59_CEE4_9DE6,
            0x3F48_6771_0C33_D4B6,
            0x4154_81DD_6B39_286C,
            0x3F09_E933_C4E7_5B15,
        ],
    );

    // 4. PYCNO_1..3 value and derivative parity (`0 ULP` vs rates/private/pycno.f90)
    let pycno_pts = [
        (
            "PYCNO_1",
            7.943_282_347_242_822e7_f64,
            3.162_277_691_791_156e9_f64,
            0x4428_3AEE_9D5C_6665_u64,
            0x42CA_AACB_9AA9_45FA_u64,
            0x4269_3EE0_FF7E_73DA_u64,
        ),
        (
            "PYCNO_2",
            1.0e8_f64,
            5.0e8_f64,
            0x4333_B486_40DD_E653_u64,
            0x41D6_EC6D_E5F2_81CB_u64,
            0x4175_2884_50EB_FEF7_u64,
        ),
        (
            "PYCNO_3",
            exp10(8.191366),
            exp10(9.893157),
            0x44DE_BCD8_F480_6EAF_u64,
            0x4344_21E3_9F5C_E493_u64,
            0x42E0_E24A_7605_ABC1_u64,
        ),
    ];
    with_numerics_mode(NumericsMode::Strict, || {
        for &(lbl, t, rho, er, edt, edrho) in &pycno_pts {
            let r_scalar = eval_fl_epsnuc_3alf(t, rho, 1.0, 2.0);
            let (r, dr_dt, dr_drho) =
                eval_fl_epsnuc_3alf_with_derivs(t, rho, 1.0, 2.0);
            let (rn, dtn, drhon) = fl_epsnuc_3alf_with_derivs(t, rho, 1.0, 2.0);
            assert_eq!(r.to_bits(), r_scalar.to_bits());
            assert_eq!(rn.to_bits(), r.to_bits());
            assert_eq!(dtn.to_bits(), dr_dt.to_bits());
            assert_eq!(drhon.to_bits(), dr_drho.to_bits());
            assert_ulp0(&format!("{lbl}.r"), r, er);
            assert_ulp0(&format!("{lbl}.dr_dt"), dr_dt, edt);
            assert_ulp0(&format!("{lbl}.dr_drho"), dr_drho, edrho);
        }
    });
}

#[test]
fn test_board_131_eos_pt_get_t_brown_dwarf_rho_probe() {
    let ht = HelmTable::builtin().expect("builtin HELM table");
    let catalog = NuclideCatalog::builtin_ref();
    let names = ["h1", "he3", "he4", "c12", "n14", "o16", "ne20", "mg24"];
    let chem_id: Vec<usize> = names
        .iter()
        .map(|n| catalog.get_nuclide_index(n).unwrap())
        .collect();
    let xa: Vec<f64> = [
        0x3FE8_4F76_5FD8_ADAD_u64,
        0x3EFA_CDC7_1D68_0D6B,
        0x3FCE_BE09_3606_FA8A,
        0x3EF2_1254_1AC8_1F32,
        0x3ED5_303C_7D57_B102,
        0x3F08_972A_60D2_12D5,
        0x3EE5_DCF1_DFB8_35A5,
        0x3EF5_605A_C0D2_B419,
    ]
    .into_iter()
    .map(f64::from_bits)
    .collect();
    let p = 23_480_446.85_f64;
    let s = 949_255_972.02_f64;
    let cfg = EosConfig::default();

    let inv = with_numerics_mode(NumericsMode::Strict, || {
        eos_pt_get_t(
            ht,
            &chem_id,
            &xa,
            log10(p),
            I_LN_S,
            log(s),
            4.0,
            1.0e-6,
            1.0e-6,
            100,
            &cfg,
        )
        .expect("eos_pt_get_t converges")
    });
    assert_ulp0("board_131.log_t", inv.log10_result, 0x400C_DAF5_4DD7_0EE4);
    assert_ulp0(
        "board_131.log10_rho",
        inv.pt.log10_rho,
        0xC00E_DAA0_E5F0_F0A9,
    );
    assert_ulp0("board_131.rho", inv.pt.rho, 0x3F22_3A91_A8DF_8D44);
    assert_eq!(inv.eos_calls, 10, "expected exact 10 MESA eosPT calls");
    let s_rel = (exp(inv.pt.eos.res[I_LN_S]) - s).abs() / s;
    assert!(
        s_rel < 1.0e-8,
        "board_131 relative S error too large: {s_rel}"
    );

    // Sweep a 15x15 grid around `(P, S)` verifying `brent_safe_zero` stays
    // strictly within `ytol = 1e-6` (`eospt_eval.f90:530`, `mod_brent.f90:574`).
    let mut max_s_err = 0.0_f64;
    with_numerics_mode(NumericsMode::Strict, || {
        for ip in 0..15 {
            for is_idx in 0..15 {
                let p_i = p * (0.95 + 0.007 * (ip as f64));
                let s_i = s * (0.98 + 0.003 * (is_idx as f64));
                let inv_i = eos_pt_get_t(
                    ht,
                    &chem_id,
                    &xa,
                    log10(p_i),
                    I_LN_S,
                    log(s_i),
                    4.0,
                    1.0e-6,
                    1.0e-6,
                    100,
                    &cfg,
                )
                .expect("grid eos_pt_get_t");
                assert!(inv_i.pt.rho.is_finite());
                let err_i = (inv_i.pt.eos.res[I_LN_S] - log(s_i)).abs();
                if err_i > max_s_err {
                    max_s_err = err_i;
                }
            }
        }
    });
    assert!(max_s_err < 1.0e-6, "grid max |lnS - target| = {max_s_err}");
}

#[test]
fn test_deep_eospt_kap_highz_and_rates_probes() {
    let ht = HelmTable::builtin().expect("builtin HELM table");
    let catalog = NuclideCatalog::builtin_ref();
    let cfg = EosConfig::default();

    with_numerics_mode(NumericsMode::Strict, || {
        // 1. EOSPT_CALL1 (check_pulse_atm Call 1)
        let names8 = ["h1", "he3", "he4", "c12", "n14", "o16", "ne20", "mg24"];
        let cid8: Vec<usize> = names8
            .iter()
            .map(|n| catalog.get_nuclide_index(n).unwrap())
            .collect();
        let xa_cpa: Vec<f64> = [
            0x3FE6_6666_6666_6667_u64,
            0x3EFF_3EBD_E705_8C15,
            0x3FD1_EB08_23C0_B5D7,
            0x3F6C_3CA3_69D8_B381,
            0x3F50_8DAF_41EC_844F,
            0x3F83_3619_1BA4_210F,
            0x3F61_149C_F6C7_EBFF,
            0x3F70_B346_E6A4_9EB9,
        ]
        .into_iter()
        .map(f64::from_bits)
        .collect();
        let p_cpa = exp(f64::from_bits(0x400B_9C2C_6763_1F62));
        let t_cpa = exp(f64::from_bits(0x4020_7360_154A_9EDB));
        let pgas_cpa = p_cpa - (CRAD / 3.0) * t_cpa * t_cpa * t_cpa * t_cpa;
        let pt1 = eos_pt_get_with_linear(
            ht,
            &cid8,
            &xa_cpa,
            log10(pgas_cpa),
            log(t_cpa) / LN10,
            Some(pgas_cpa),
            Some(t_cpa),
            &cfg,
        )
        .expect("eos_pt_get_with_linear cpa");
        let pt1_log = eos_pt_get(
            ht,
            &cid8,
            &xa_cpa,
            log10(pgas_cpa),
            log(t_cpa) / LN10,
            &cfg,
        )
        .expect("eos_pt_get cpa");
        assert_ulp0("eospt_call1_lin.rho", pt1.rho, 0x3DE1_F2CD_7628_DE03);
        assert_ulp0(
            "eospt_call1_lin.log10_rho",
            pt1.log10_rho,
            0xC023_C4A6_852B_2B21,
        );
        assert_ulp0(
            "eospt_call1_lin.lnfree_e",
            pt1.eos.res[I_LN_FREE_E],
            0xC022_6BEA_8BD9_1ABA,
        );
        // Log-only `eos_pt_get` reconstructs `Pgas`/`T` via `exp10` (18/1 ULP).
        assert_ulp0("eospt_call1_log.rho", pt1_log.rho, 0x3DE1_F2CD_7628_DE15);
        assert_ulp0(
            "eospt_call1_log.log10_rho",
            pt1_log.log10_rho,
            0xC023_C4A6_852B_2B20,
        );

        // 2. EOSPT_BD131
        let xa_bd: Vec<f64> = [
            0x3FE8_4F76_5FD8_ADAD_u64,
            0x3EFA_CDC7_1D68_0D6B,
            0x3FCE_BE09_3606_FA8A,
            0x3EF2_1254_1AC8_1F32,
            0x3ED5_303C_7D57_B102,
            0x3F08_972A_60D2_12D5,
            0x3EE5_DCF1_DFB8_35A5,
            0x3EF5_605A_C0D2_B419,
        ]
        .into_iter()
        .map(f64::from_bits)
        .collect();
        let p_bd = 23_480_446.85_f64;
        let logt_bd = f64::from_bits(0x400C_DAF5_4DD7_0EE4);
        let t_bd = exp10(logt_bd);
        let pgas_bd = p_bd - (CRAD / 3.0) * t_bd * t_bd * t_bd * t_bd;
        let pt_bd = eos_pt_get_with_linear(
            ht,
            &cid8,
            &xa_bd,
            log10(pgas_bd),
            logt_bd,
            Some(pgas_bd),
            Some(t_bd),
            &cfg,
        )
        .expect("eos_pt_get_with_linear bd");
        assert_ulp0("eospt_bd131_lin.rho", pt_bd.rho, 0x3F22_3A91_9F87_70BB);
        assert_ulp0(
            "eospt_bd131_lin.log10_rho",
            pt_bd.log10_rho,
            0xC00E_DAA0_E7B8_DEB1,
        );
        assert_ulp0(
            "eospt_bd131_lin.ln_s",
            pt_bd.eos.res[I_LN_S],
            0x4034_ABD3_0B8D_339E,
        );

        // 3. KAP_HIGHZ_LOWT_1..3
        let mut xa_hz_low = [
            0.54985745_f64,
            1.0e-5,
            0.37997195,
            0.0120456,
            0.005,
            0.0326819,
            0.01,
            0.0,
        ];
        xa_hz_low[7] = 1.0 - xa_hz_low[..7].iter().sum::<f64>();
        let hz_kap_cfg = KapConfig {
            zbase: 0.07,
            kap_option: KapOption::Gs98,
            kap_co_option: KapCoOption::Gs98Co,
            kap_low_t_option: KapLowTOption::LowTFa05Gs98,
            use_type2_opacities: true,
            ..KapConfig::default()
        };
        let kap_exp = [
            (
                3.55_f64,
                -8.5_f64,
                0x3F73_425A_09AE_1C1D_u64,
                0x3FE1_8E12_CE52_0C93_u64,
                0x4022_4A41_FA0F_3FEA_u64,
            ),
            (
                3.85_f64,
                -8.0_f64,
                0x3FDF_5DE4_D0E2_5D1E_u64,
                0x3FD7_9153_0727_2D3A_u64,
                0x4023_DD4D_16CD_599C_u64,
            ),
            (
                4.10_f64,
                -7.5_f64,
                0x4069_F2EF_7833_7E73_u64,
                0x3FE1_7672_D4E8_FF3B_u64,
                0x4003_934F_8072_A035_u64,
            ),
        ];
        for (i, &(lt, lr, ek, edr, edt)) in kap_exp.iter().enumerate() {
            let eo =
                eos_dt_get(ht, exp10(lr), exp10(lt), &cid8, &xa_hz_low, &cfg)
                    .expect("eos highz lowt");
            let ko = kap_get(
                &hz_kap_cfg,
                &cid8,
                &xa_hz_low,
                lr,
                lt,
                eo.res[I_LN_FREE_E],
                eo.d_dlnd[I_LN_FREE_E],
                eo.d_dln_t[I_LN_FREE_E],
                eo.res[I_ETA],
                eo.d_dlnd[I_ETA],
                eo.d_dln_t[I_ETA],
            )
            .expect("kap highz lowt");
            assert_ulp0(&format!("KAP_HIGHZ_LOWT_{}.kap", i + 1), ko.kap, ek);
            assert_ulp0(
                &format!("KAP_HIGHZ_LOWT_{}.dkr", i + 1),
                ko.dlnkap_dln_rho,
                edr,
            );
            assert_ulp0(
                &format!("KAP_HIGHZ_LOWT_{}.dkt", i + 1),
                ko.dlnkap_dln_t,
                edt,
            );
        }

        // 4. WD_M370 and HIGHZ_M340 screening + raw rates
        let rcat = RatesCatalog::builtin();
        let wd_names = ["he4", "c12", "o16", "ne20", "si28"];
        let wd_cid: Vec<usize> = wd_names
            .iter()
            .map(|n| catalog.get_nuclide_index(n).unwrap())
            .collect();
        let mut wd_xa = [0.0_f64, 0.350611, 0.622309, 0.022977, 0.0];
        wd_xa[4] = 1.0 - wd_xa[..4].iter().sum::<f64>();
        let wd_comp = get_composition_info(catalog, &wd_cid, &wd_xa, true)
            .expect("wd comp");
        let wd_lt = 8.191366_f64;
        let wd_lr = 9.893157_f64;
        let wd_t = exp10(wd_lt);
        let wd_rho = exp10(wd_lr);
        let sc_chug = screen_pair(
            ScreeningMode::Chugunov,
            wd_t,
            wd_rho,
            wd_comp.zbar,
            wd_comp.abar,
            wd_comp.z2bar,
            12.0,
            6.0,
            12.0,
            6.0,
        );
        let sc_ext = screen_pair(
            ScreeningMode::Extended,
            wd_t,
            wd_rho,
            wd_comp.zbar,
            wd_comp.abar,
            wd_comp.z2bar,
            12.0,
            6.0,
            12.0,
            6.0,
        );
        assert_ulp0(
            "SCR_WD_C12C12_CHUG.scor",
            sc_chug.scor,
            0x4407_F2D6_C8E3_2655,
        );
        assert_ulp0(
            "SCR_WD_C12C12_CHUG.dscor_dt",
            sc_chug.dscor_dt,
            0xC2AA_0714_64CB_F739,
        );
        assert_ulp0(
            "SCR_WD_C12C12_CHUG.dscor_dd",
            sc_chug.dscor_dd,
            0x4234_B647_F9B6_366D,
        );
        assert_ulp0(
            "SCR_WD_C12C12_EXT.scor",
            sc_ext.scor,
            0x43FA_3E6D_E6A1_DA68,
        );
        assert_ulp0(
            "SCR_WD_C12C12_EXT.dscor_dt",
            sc_ext.dscor_dt,
            0xC29B_6EEF_8FA8_5978,
        );
        assert_ulp0(
            "SCR_WD_C12C12_EXT.dscor_dd",
            sc_ext.dscor_dd,
            0x4225_91D3_FC19_74FD,
        );
        let tf_wd = TFactors::from_temp(wd_t);
        for (rname, exp_b) in [
            ("r_he4_he4_he4_to_c12", 0x3C50_6925_50CE_75D4_u64),
            ("r_c12_ag_o16", 0x3C97_381D_4729_6422_u64),
            ("r_c12_c12_to_he4_ne20", 0x3761_394B_5DBB_F9EA_u64),
            ("r_c12_c12_to_h1_na23", 0x375A_DDF1_9D9C_D484_u64),
        ] {
            let rid = rcat.reaction_id_by_name(rname).unwrap();
            let rv = eval_reaction_raw(
                rcat,
                rcat.reaction_by_id(rid).unwrap(),
                wd_t,
                &tf_wd,
            )
            .unwrap();
            assert_ulp0(&format!("RAW_WD_{rname}"), rv, exp_b);
        }

        let hz_names = [
            "h1", "he3", "he4", "c12", "n14", "o16", "o18", "ne20", "ne22",
            "mg24",
        ];
        let hz_cid: Vec<usize> = hz_names
            .iter()
            .map(|n| catalog.get_nuclide_index(n).unwrap())
            .collect();
        let mut hz_xa = [
            0.0_f64, 0.0, 0.803521, 0.108974, 0.0, 0.010230, 0.0, 0.006897,
            0.0, 0.0,
        ];
        hz_xa[9] = 1.0 - hz_xa[..9].iter().sum::<f64>();
        let hz_comp = get_composition_info(catalog, &hz_cid, &hz_xa, true)
            .expect("hz comp");
        let hz_t = exp10(8.169_043_301_529_7_f64);
        let hz_rho = exp10(3.4921578275053786_f64);
        let sc_hz = screen_pair(
            ScreeningMode::Chugunov,
            hz_t,
            hz_rho,
            hz_comp.zbar,
            hz_comp.abar,
            hz_comp.z2bar,
            4.0,
            2.0,
            12.0,
            6.0,
        );
        assert_ulp0(
            "SCR_HZ_HE4C12_CHUG.scor",
            sc_hz.scor,
            0x3FF1_57A7_D3B8_5F5D,
        );
        assert_ulp0(
            "SCR_HZ_HE4C12_CHUG.dscor_dt",
            sc_hz.dscor_dt,
            0xBE0C_6DEC_D1FF_AB13,
        );
        assert_ulp0(
            "SCR_HZ_HE4C12_CHUG.dscor_dd",
            sc_hz.dscor_dd,
            0x3EEB_7CB4_A751_9984,
        );
        let tf_hz = TFactors::from_temp(hz_t);
        for (rname, exp_b) in [
            ("r_he4_he4_he4_to_c12", 0x3C31_21CE_F374_D4E9_u64),
            ("r_c12_ag_o16", 0x3C81_BF6E_5674_E7EA_u64),
            ("r_o18_ag_ne22", 0x3CF1_B167_3CC4_BB85_u64),
            ("r_ne22_ag_mg26", 0x3BB0_2FB4_F53E_D3C5_u64),
        ] {
            let rid = rcat.reaction_id_by_name(rname).unwrap();
            let rv = eval_reaction_raw(
                rcat,
                rcat.reaction_by_id(rid).unwrap(),
                hz_t,
                &tf_hz,
            )
            .unwrap();
            assert_ulp0(&format!("RAW_HZ_{rname}"), rv, exp_b);
        }
    });
}
