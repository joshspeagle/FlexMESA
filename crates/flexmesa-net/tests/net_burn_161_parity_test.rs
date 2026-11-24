// SPDX-License-Identifier: GPL-3.0-or-later
//! Exact MESA 26.4.1 IEEE-754 hex oracle parity tests for `flexmesa-net`.
//!
//! Covers `net_get`, `net_get_symbolic_d_dxdt_dx`, screening modes, pycnonuclear
//! screening, FL87 triple-alpha and Suzuki flag forwarding, `eval_single_weak_blended`
//! temperature derivatives, `GeneralTwoTwo` reaction ordering, `op_code == 2`
//! weaklib-only `exp(-99)` raw-rate tables, and 15 one-zone burn trajectories
//! across `burn_1_zone`, `burn_1_zone_const_p` (`rodas4`, `ros3p`, `ros3pl`,
//! `rodasp`), and `burn_1_zone_const_density`.

use std::collections::HashMap;

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_eos::{EosConfig, HelmTable};
use flexmesa_math::{exp10, with_numerics_mode, NumericsMode};
use flexmesa_net::{
    burn_1_zone, burn_1_zone_const_density, burn_1_zone_const_p, net_get,
    net_get_symbolic_d_dxdt_dx, Burn1ZoneInput, BurnConstDensityInput,
    BurnConstPInput, NetGeneralInfo,
};
use flexmesa_num::RosenbrockVariant;
use flexmesa_rates::ScreeningMode;

fn signed_mag(bits: u64) -> i128 {
    if (bits >> 63) != 0 {
        -(i128::from(bits ^ (1_u64 << 63)))
    } else {
        i128::from(bits)
    }
}

fn ulp_diff(actual: f64, expected_bits: u64) -> u64 {
    let expected = f64::from_bits(expected_bits);
    if actual.is_nan() || expected.is_nan() {
        return if actual.is_nan() && expected.is_nan() {
            0
        } else {
            u64::MAX
        };
    }
    u64::try_from(
        (signed_mag(actual.to_bits()) - signed_mag(expected_bits))
            .unsigned_abs(),
    )
    .unwrap_or(u64::MAX)
}

fn assert_ulp_le(label: &str, actual: f64, expected_bits: u64, max_ulp: u64) {
    let got_ulp = ulp_diff(actual, expected_bits);
    let expected = f64::from_bits(expected_bits);
    assert!(
        got_ulp <= max_ulp,
        "{label}: actual={actual:.16e} (0x{:016X}) vs \
         expected={expected:.16e} (0x{expected_bits:016X}), \
         ulp={got_ulp} > max_ulp={max_ulp}",
        actual.to_bits(),
    );
}

fn assert_rel_le(label: &str, actual: f64, expected_bits: u64, max_rel: f64) {
    let expected = f64::from_bits(expected_bits);
    let denom = expected.abs().max(actual.abs()).max(1.0e-300);
    let rel = (actual - expected).abs() / denom;
    assert!(
        rel <= max_rel,
        "{label}: actual={actual:.16e} (0x{:016X}) vs \
         expected={expected:.16e} (0x{expected_bits:016X}), \
         rel={rel:.3e} > max_rel={max_rel:.3e}",
        actual.to_bits(),
    );
}

const PRESET_H_SOLAR: &[(&str, f64)] = &[
    ("h1", 0.35),
    ("h2", 1.0e-15),
    ("he3", 2.0e-5),
    ("he4", 0.63),
    ("c12", 3.0e-3),
    ("n14", 1.0e-3),
    ("o16", 1.0e-2),
    ("ne20", 2.0e-3),
    ("mg24", 1.0e-3),
];

const PRESET_H_CNO: &[(&str, f64)] = &[
    ("h1", 0.45),
    ("he4", 0.30),
    ("c12", 0.05),
    ("c13", 0.01),
    ("n13", 1.0e-4),
    ("n14", 0.08),
    ("n15", 0.01),
    ("o14", 1.0e-4),
    ("o15", 1.0e-4),
    ("o16", 0.08),
    ("f17", 1.0e-5),
    ("ne20", 0.02),
];

const PRESET_HE_RICH: &[(&str, f64)] = &[
    ("h1", 1.0e-4),
    ("he4", 0.90),
    ("c12", 0.05),
    ("c13", 2.0e-3),
    ("n14", 5.0e-3),
    ("o16", 0.02),
    ("o18", 5.0e-3),
    ("ne20", 5.0e-3),
    ("ne22", 1.0e-2),
    ("mg24", 2.0e-3),
    ("fe56", 1.0e-3),
];

const PRESET_CO_RICH: &[(&str, f64)] = &[
    ("he4", 0.01),
    ("c12", 0.45),
    ("o16", 0.48),
    ("ne20", 0.03),
    ("ne22", 0.02),
    ("na23", 2.0e-3),
    ("mg24", 5.0e-3),
    ("si28", 2.0e-3),
];

const PRESET_SI_RICH: &[(&str, f64)] = &[
    ("neut", 1.0e-6),
    ("prot", 5.0e-5),
    ("he4", 0.05),
    ("o16", 0.15),
    ("ne20", 0.03),
    ("mg24", 0.05),
    ("si28", 0.40),
    ("s32", 0.15),
    ("ar36", 0.05),
    ("ca40", 0.04),
    ("ti44", 0.01),
    ("cr48", 0.01),
    ("fe52", 0.02),
    ("fe54", 0.02),
    ("fe56", 0.01),
    ("ni56", 0.01),
];

const PRESET_NSE_APPROX21: &[(&str, f64)] = &[
    ("neut", 2.0e-5),
    ("prot", 8.0e-4),
    ("he4", 0.03),
    ("si28", 0.10),
    ("s32", 0.06),
    ("ar36", 0.03),
    ("ca40", 0.03),
    ("ti44", 0.01),
    ("cr48", 0.02),
    ("fe52", 0.05),
    ("fe54", 0.25),
    ("fe56", 0.30),
    ("co56", 0.04),
    ("ni56", 0.07),
    ("cr60", 0.01),
];

fn fill_preset_comp(preset: &str, g: &NetGeneralInfo) -> Vec<f64> {
    let cat = NuclideCatalog::builtin_ref();
    let entries = match preset {
        "h_solar" => PRESET_H_SOLAR,
        "h_cno" => PRESET_H_CNO,
        "he_rich" => PRESET_HE_RICH,
        "co_rich" => PRESET_CO_RICH,
        "si_rich" => PRESET_SI_RICH,
        "nse_approx21" => PRESET_NSE_APPROX21,
        other => panic!("unknown preset: {other}"),
    };
    let mut x = vec![0.0_f64; g.num_isos];
    for &(nm, val) in entries {
        if let Some(cid) = cat.get_nuclide_index(nm) {
            if cid < g.net_iso.len() && g.net_iso[cid] > 0 {
                x[g.net_iso[cid] - 1] = val;
            }
        }
    }
    for v in &mut x {
        if *v < 1.0e-12 {
            *v = 1.0e-12;
        }
    }
    let s: f64 = x.iter().sum();
    for v in &mut x {
        *v /= s;
    }
    x
}

fn iso_map(g: &NetGeneralInfo) -> HashMap<String, usize> {
    let cat = NuclideCatalog::builtin_ref();
    g.chem_id
        .iter()
        .enumerate()
        .map(|(i, &cid)| (cat.get_by_id(cid).unwrap().name.clone(), i))
        .collect()
}

struct NetCaseSpec {
    id: &'static str,
    net_name: &'static str,
    log_t: f64,
    log_rho: f64,
    eta: f64,
    scr: ScreeningMode,
    use_fl87: bool,
    fe56ec_ff: f64,
    wk_fac: f64,
    neuq_fac: f64,
    rfac_scale: f64,
    preset: &'static str,
    max_ulp: u64,
    eps_nuc_bits: u64,
    d_eps_nuc_dt_bits: u64,
    d_eps_nuc_drho_bits: u64,
    eps_neu_bits: u64,
    first_rxn: (usize, &'static str, u64),
    key_dxdt: &'static [(&'static str, u64)],
    key_ddxdt_dt: &'static [(&'static str, u64)],
}

fn run_net_case_spec(c: &NetCaseSpec) {
    let cat = NuclideCatalog::builtin_ref();
    let mut g = NetGeneralInfo::from_name(c.net_name).unwrap();
    g.fe56ec_fake_factor = c.fe56ec_ff;
    g.use_3a_fl87 = c.use_fl87;
    g.reaction_neuqs_factor = c.neuq_fac;

    let x = fill_preset_comp(c.preset, &g);
    let comp = get_composition_info(cat, &g.chem_id, &x, false).unwrap();
    let temp = exp10(c.log_t);
    let rho = exp10(c.log_rho);
    let mut rate_factors = vec![c.rfac_scale; g.num_reactions];
    if g.num_reactions > 1 && (c.rfac_scale - 1.0).abs() > 1.0e-12 {
        rate_factors[0] = 1.1;
    }

    let res = net_get(
        &g,
        false,
        &x,
        temp,
        c.log_t,
        rho,
        c.log_rho,
        comp.abar,
        comp.zbar,
        comp.z2bar,
        comp.ye,
        c.eta,
        0.0,
        0.0,
        &rate_factors,
        c.wk_fac,
        c.scr,
    )
    .unwrap();

    let id = c.id;
    assert_ulp_le(
        &format!("{id}.eps_nuc"),
        res.eps_nuc,
        c.eps_nuc_bits,
        c.max_ulp,
    );
    assert_ulp_le(
        &format!("{id}.d_eps_nuc_dt"),
        res.d_eps_nuc_dt,
        c.d_eps_nuc_dt_bits,
        c.max_ulp,
    );
    assert_ulp_le(
        &format!("{id}.d_eps_nuc_drho"),
        res.d_eps_nuc_drho,
        c.d_eps_nuc_drho_bits,
        c.max_ulp,
    );
    assert_ulp_le(
        &format!("{id}.eps_neu_total"),
        res.eps_neu_total,
        c.eps_neu_bits,
        c.max_ulp,
    );

    let (rxn_pos0, rxn_name, rxn_raw_bits) = c.first_rxn;
    assert_eq!(
        g.reactions[rxn_pos0].name, rxn_name,
        "{id}.rxn_order[{rxn_pos0}]"
    );
    assert_ulp_le(
        &format!("{id}.rate_raw[{rxn_pos0}:{rxn_name}]"),
        res.rate_raw[rxn_pos0],
        rxn_raw_bits,
        c.max_ulp,
    );

    let im = iso_map(&g);
    for &(iso, bits) in c.key_dxdt {
        assert_ulp_le(
            &format!("{id}.dxdt[{iso}]"),
            res.dxdt[im[iso]],
            bits,
            c.max_ulp,
        );
    }
    for &(iso, bits) in c.key_ddxdt_dt {
        assert_ulp_le(
            &format!("{id}.d_dxdt_dt[{iso}]"),
            res.d_dxdt_dt[im[iso]],
            bits,
            c.max_ulp,
        );
    }
}
const NET_CASES: &[NetCaseSpec] = &[
    NetCaseSpec {
        id: "pp_cno_lowT_blend",
        net_name: "pp_and_cno_extras.net",
        log_t: 5.34_f64,
        log_rho: -2.0_f64,
        eta: -5.0_f64,
        scr: ScreeningMode::Chugunov,
        use_fl87: false,
        fe56ec_ff: 1.0e-4_f64,
        wk_fac: 1.0_f64,
        neuq_fac: 1.0_f64,
        rfac_scale: 1.0_f64,
        preset: "h_solar",
        max_ulp: 0,
        eps_nuc_bits: 0x4133_D807_5444_3ED5,
        d_eps_nuc_dt_bits: 0xBE7F_2F30_684E_99D9,
        d_eps_nuc_drho_bits: 0x4033_3B33_F0C9_7728,
        eps_neu_bits: 0x4127_CA92_A9D4_95C6,
        first_rxn: (0, "r_b8_gp_be7", 0x0000_0000_0000_0000),
        key_dxdt: &[
            ("h1", 0xB83D_1D98_2C75_BFB2),
            ("c12", 0x0000_0000_0000_0000),
            ("n14", 0x3D06_2C98_80CB_BA0F),
        ],
        key_ddxdt_dt: &[
            ("h1", 0xB768_18F4_794B_A389),
            ("c12", 0x0000_0000_0000_0000),
            ("n14", 0x0000_0000_0000_0000),
        ],
    },
    NetCaseSpec {
        id: "pp_cno_solar",
        net_name: "pp_and_cno_extras.net",
        log_t: 7.18_f64,
        log_rho: 2.1_f64,
        eta: -1.3_f64,
        scr: ScreeningMode::Chugunov,
        use_fl87: false,
        fe56ec_ff: 1.0e-4_f64,
        wk_fac: 1.0_f64,
        neuq_fac: 1.0_f64,
        rfac_scale: 1.0_f64,
        preset: "h_solar",
        max_ulp: 0,
        eps_nuc_bits: 0x4143_81C1_B49F_9575,
        d_eps_nuc_dt_bits: 0x3FD7_C6CD_2C52_F630,
        d_eps_nuc_drho_bits: 0x40C3_C5F7_34AF_A4D2,
        eps_neu_bits: 0x4127_C400_4D99_D2D9,
        first_rxn: (0, "r_b8_gp_be7", 0x359B_1431_6E8F_6B26),
        key_dxdt: &[
            ("h1", 0xBD50_A533_479E_0B4D),
            ("he4", 0x3D6F_D055_19B9_0A61),
            ("c12", 0xBC90_0C1E_C024_D73C),
            ("n14", 0x3D06_2C8C_DE9B_DF77),
        ],
        key_ddxdt_dt: &[
            ("h1", 0xBBF4_9007_ED34_E3F3),
            ("he4", 0x3B90_A387_B475_F9DD),
            ("c12", 0xBB53_9507_B6D5_A5AA),
            ("n14", 0xBABF_7128_9492_55B1),
        ],
    },
    NetCaseSpec {
        id: "o18_ne22_heflash",
        net_name: "o18_and_ne22.net",
        log_t: 8.20_f64,
        log_rho: 5.5_f64,
        eta: 10.0_f64,
        scr: ScreeningMode::Chugunov,
        use_fl87: false,
        fe56ec_ff: 1.0e-4_f64,
        wk_fac: 1.0_f64,
        neuq_fac: 1.0_f64,
        rfac_scale: 1.0_f64,
        preset: "he_rich",
        max_ulp: 0,
        eps_nuc_bits: 0x4321_9C53_57AA_389D,
        d_eps_nuc_dt_bits: 0x419E_7AB8_0A69_B5B6,
        d_eps_nuc_drho_bits: 0x4200_3EFC_71F8_1219,
        eps_neu_bits: 0x42E2_2E76_A398_96EE,
        first_rxn: (0, "r34_pp2", 0x4071_DC9E_F1A9_1731),
        key_dxdt: &[
            ("he4", 0x3EF5_6496_5EFC_666C),
            ("c12", 0xBF65_F0C6_5939_6AA4),
            ("o18", 0x3DDB_118C_1033_128E),
            ("ne22", 0x3DBF_C055_3373_ABF6),
        ],
        key_ddxdt_dt: &[
            ("he4", 0x3D7F_8F23_BC6E_5CA5),
            ("c12", 0xBDE2_4D2F_FEEA_390A),
            ("o18", 0x3C76_2331_A5C8_D733),
            ("ne22", 0x3C57_568A_CFA3_7887),
        ],
    },
    NetCaseSpec {
        id: "co_burn_wd_pycno_chug",
        net_name: "co_burn.net",
        log_t: 8.15_f64,
        log_rho: 9.3_f64,
        eta: 80.0_f64,
        scr: ScreeningMode::Chugunov,
        use_fl87: false,
        fe56ec_ff: 1.0e-4_f64,
        wk_fac: 1.0_f64,
        neuq_fac: 1.0_f64,
        rfac_scale: 1.0_f64,
        preset: "co_rich",
        max_ulp: 0,
        eps_nuc_bits: 0x4375_34C3_7CE2_20AF,
        d_eps_nuc_dt_bits: 0x4200_6087_E5F8_1B33,
        d_eps_nuc_drho_bits: 0x41B3_DBEA_606A_BB9F,
        eps_neu_bits: 0x42C5_9069_2633_3E13,
        first_rxn: (0, "r_c12_ag_o16", 0x3E5C_A055_1AC1_7479),
        key_dxdt: &[
            ("c12", 0x3FC4_A394_97B2_C637),
            ("o16", 0xBEF1_8F7C_0ED4_F4DD),
            ("ne20", 0x3E31_7E8B_EBD8_5B6E),
            ("mg24", 0x3E02_4220_712C_D829),
        ],
        key_ddxdt_dt: &[
            ("c12", 0x3E50_1EBD_E2C7_31F5),
            ("o16", 0x3D72_BBAC_DCB5_4F1C),
            ("ne20", 0x3CB4_8CB7_4007_D89A),
            ("mg24", 0xBC67_01CB_B461_82A2),
        ],
    },
    NetCaseSpec {
        id: "co_burn_wd_pycno_ext",
        net_name: "co_burn.net",
        log_t: 8.15_f64,
        log_rho: 9.3_f64,
        eta: 80.0_f64,
        scr: ScreeningMode::Extended,
        use_fl87: false,
        fe56ec_ff: 1.0e-4_f64,
        wk_fac: 1.0_f64,
        neuq_fac: 1.0_f64,
        rfac_scale: 1.0_f64,
        preset: "co_rich",
        max_ulp: 0,
        eps_nuc_bits: 0x43CF_975C_D8EC_8965,
        d_eps_nuc_dt_bits: 0x4252_10EA_27DF_341C,
        d_eps_nuc_drho_bits: 0x4211_2DCA_2E31_13ED,
        eps_neu_bits: 0x42B3_FBF8_F570_6277,
        first_rxn: (0, "r_c12_ag_o16", 0x3E5C_A055_1AC1_7479),
        key_dxdt: &[
            ("c12", 0x401F_2164_A4E2_36F4),
            ("o16", 0xBE6B_39FF_8A40_54C7),
            ("ne20", 0x3E1F_409D_0527_4B69),
            ("mg24", 0x3DE9_34BB_2656_5F96),
        ],
        key_ddxdt_dt: &[
            ("c12", 0x3EA1_CDDD_2E55_3D80),
            ("o16", 0x3D63_91A9_ED15_967B),
            ("ne20", 0x3CA3_DDB9_5A5E_8EFC),
            ("mg24", 0xBC40_4344_D419_F537),
        ],
    },
    NetCaseSpec {
        id: "cno_extras_chug",
        net_name: "cno_extras.net",
        log_t: 8.45_f64,
        log_rho: 3.5_f64,
        eta: -1.0_f64,
        scr: ScreeningMode::Chugunov,
        use_fl87: false,
        fe56ec_ff: 1.0e-4_f64,
        wk_fac: 1.0_f64,
        neuq_fac: 1.0_f64,
        rfac_scale: 1.0_f64,
        preset: "h_cno",
        max_ulp: 0,
        eps_nuc_bits: 0x4485_196D_5000_C4F6,
        d_eps_nuc_dt_bits: 0x42F6_7ABE_8032_0876,
        d_eps_nuc_drho_bits: 0x43CB_9599_30C6_68F9,
        eps_neu_bits: 0x4247_1DBC_26CA_1D7B,
        first_rxn: (0, "r34_pp2", 0x405D_7448_55E4_E036),
        key_dxdt: &[
            ("h1", 0xC0A4_4C1B_8593_DCD5),
            ("he4", 0x40C4_429D_489C_9155),
            ("c12", 0x40DE_6124_09AC_2CE6),
            ("n14", 0xC042_ABE8_1265_5A08),
            ("o15", 0x4046_DECA_E840_F438),
        ],
        key_ddxdt_dt: &[
            ("h1", 0xBF15_A01E_D9CC_6F07),
            ("he4", 0x3F35_9637_056C_FDA4),
            ("c12", 0x3F50_2F38_590C_4D52),
            ("n14", 0xBEB4_F6FA_DA65_B8CF),
            ("o15", 0x3EB8_D0F2_9D55_EC7A),
        ],
    },
    NetCaseSpec {
        id: "cno_extras_salp",
        net_name: "cno_extras.net",
        log_t: 8.45_f64,
        log_rho: 3.5_f64,
        eta: -1.0_f64,
        scr: ScreeningMode::Salpeter,
        use_fl87: false,
        fe56ec_ff: 1.0e-4_f64,
        wk_fac: 1.0_f64,
        neuq_fac: 1.0_f64,
        rfac_scale: 1.0_f64,
        preset: "h_cno",
        max_ulp: 0,
        eps_nuc_bits: 0x4485_3451_4A96_454F,
        d_eps_nuc_dt_bits: 0x42F6_9211_DFE8_E19C,
        d_eps_nuc_drho_bits: 0x43CB_CC0C_F09C_E83F,
        eps_neu_bits: 0x4247_1DBC_A5F7_F1A6,
        first_rxn: (0, "r34_pp2", 0x405D_7448_55E4_E036),
        key_dxdt: &[
            ("h1", 0xC0A4_65F9_C338_6530),
            ("he4", 0x40C4_5C6F_A85C_8BE2),
            ("c12", 0x40DE_87DC_4DC8_7FA7),
            ("n14", 0xC042_C3F1_6FEB_A09D),
            ("o15", 0x4046_FBF0_E0FD_CCD9),
        ],
        key_ddxdt_dt: &[
            ("h1", 0xBF15_B68F_57CC_EFEE),
            ("he4", 0x3F35_AC9D_73F4_0D98),
            ("c12", 0x3F50_4003_C8BC_FD2B),
            ("n14", 0xBEB5_0D20_2F7C_FD55),
            ("o15", 0x3EB8_EACE_1E27_8C59),
        ],
    },
    NetCaseSpec {
        id: "cno_extras_noscr",
        net_name: "cno_extras.net",
        log_t: 8.45_f64,
        log_rho: 3.5_f64,
        eta: -1.0_f64,
        scr: ScreeningMode::NoScreening,
        use_fl87: false,
        fe56ec_ff: 1.0e-4_f64,
        wk_fac: 1.0_f64,
        neuq_fac: 1.0_f64,
        rfac_scale: 1.0_f64,
        preset: "h_cno",
        max_ulp: 0,
        eps_nuc_bits: 0x4484_B2C4_168D_55FA,
        d_eps_nuc_dt_bits: 0x42F6_1F07_09C8_359B,
        d_eps_nuc_drho_bits: 0x43CA_CF53_B159_6FDF,
        eps_neu_bits: 0x4247_1DBA_9CC1_9A8F,
        first_rxn: (0, "r34_pp2", 0x405D_7448_55E4_E036),
        key_dxdt: &[
            ("h1", 0xC0A3_E959_DA8B_5D4E),
            ("he4", 0x40C3_E007_C9C7_6E9B),
            ("c12", 0x40DD_CD4F_27DA_64B8),
            ("n14", 0xC042_4EF7_5A77_E410),
            ("o15", 0x4046_6F82_2133_A923),
        ],
        key_ddxdt_dt: &[
            ("h1", 0xBF15_47E3_D372_A9A3),
            ("he4", 0x3F35_3E22_9EF8_1FB1),
            ("c12", 0x3F4F_DA5B_E838_EDD9),
            ("n14", 0xBEB4_9F3B_9B18_7A97),
            ("o15", 0x3EB8_6B56_BC1E_9F1D),
        ],
    },
    NetCaseSpec {
        id: "sagb_nena_mgal",
        net_name: "sagb_NeNa_MgAl.net",
        log_t: 8.85_f64,
        log_rho: 4.8_f64,
        eta: 5.0_f64,
        scr: ScreeningMode::Chugunov,
        use_fl87: false,
        fe56ec_ff: 1.0e-4_f64,
        wk_fac: 1.0_f64,
        neuq_fac: 1.0_f64,
        rfac_scale: 1.0_f64,
        preset: "co_rich",
        max_ulp: 0,
        eps_nuc_bits: 0x4318_7499_D150_128D,
        d_eps_nuc_dt_bits: 0x4180_9550_BCB4_5E61,
        d_eps_nuc_drho_bits: 0x421A_B2C0_1D29_5A06,
        eps_neu_bits: 0x41D4_488C_0A97_AE3A,
        first_rxn: (0, "r_ne20_ag_mg24", 0x3FFD_9089_3B36_9CEE),
        key_dxdt: &[
            ("he4", 0xBF58_D60F_BB1E_A34E),
            ("c12", 0xBEF5_82F9_F4CA_E4A2),
            ("ne22", 0xBE86_E124_9199_CA2E),
            ("na23", 0xBE22_F777_11F3_E6CF),
            ("mg24", 0x3F27_63B8_1CC7_78AF),
        ],
        key_ddxdt_dt: &[
            ("he4", 0xBDC0_AE8A_5E73_9B7D),
            ("c12", 0xBD52_7568_7329_4213),
            ("ne22", 0xBCDE_034B_92C4_594A),
            ("na23", 0x3CA9_48FB_C85B_2ED9),
            ("mg24", 0x3D8A_DCF6_E08D_D214),
        ],
    },
    NetCaseSpec {
        id: "mesa_49_siburn",
        net_name: "mesa_49.net",
        log_t: 9.35_f64,
        log_rho: 6.8_f64,
        eta: 1.5_f64,
        scr: ScreeningMode::Chugunov,
        use_fl87: false,
        fe56ec_ff: 1.0e-4_f64,
        wk_fac: 1.0_f64,
        neuq_fac: 1.0_f64,
        rfac_scale: 1.0_f64,
        preset: "si_rich",
        max_ulp: 0,
        eps_nuc_bits: 0x44DA_103C_9BF7_17E9,
        d_eps_nuc_dt_bits: 0xC2D1_A02F_54C5_535A,
        d_eps_nuc_drho_bits: 0x4371_58D0_4B56_31D4,
        eps_neu_bits: 0x41B0_AB59_3B22_0DEC,
        first_rxn: (0, "r_al25_ng_al26", 0x422B_769B_2AF0_928E),
        key_dxdt: &[
            ("neut", 0xC0F0_45C4_1DB6_A066),
            ("he4", 0xC0C6_3074_385E_77EC),
            ("o16", 0xC0F1_5534_B7CB_5EBE),
            ("si28", 0xC11D_74A5_1188_D1A8),
            ("s32", 0xC133_6B2A_26E7_D78F),
        ],
        key_ddxdt_dt: &[
            ("neut", 0xBEC8_3D2D_140D_8B7A),
            ("he4", 0xBF23_97E6_C5F3_6925),
            ("o16", 0xBF0D_689E_7ACF_6F1C),
            ("si28", 0xBF22_A03C_AB5C_5FBF),
            ("s32", 0xBF37_E834_8439_1C2C),
        ],
    },
    NetCaseSpec {
        id: "mesa_80_nse",
        net_name: "mesa_80.net",
        log_t: 9.55_f64,
        log_rho: 7.5_f64,
        eta: 2.5_f64,
        scr: ScreeningMode::Chugunov,
        use_fl87: false,
        fe56ec_ff: 1.0e-4_f64,
        wk_fac: 1.0_f64,
        neuq_fac: 1.0_f64,
        rfac_scale: 1.0_f64,
        preset: "si_rich",
        max_ulp: 0,
        eps_nuc_bits: 0xC510_0F42_7B8D_9403,
        d_eps_nuc_dt_bits: 0xC360_6F84_DB7E_C739,
        d_eps_nuc_drho_bits: 0xC384_20A7_26B4_F5FF,
        eps_neu_bits: 0x4276_8A70_BFD3_C65F,
        first_rxn: (0, "r_al25_ng_al26", 0x4252_F6AA_D722_AE8D),
        key_dxdt: &[
            ("neut", 0xC139_4AFF_A639_BCA3),
            ("he4", 0xC17C_4698_34BF_1C00),
            ("si28", 0xC198_FC81_7D6C_DFE9),
            ("fe54", 0xC084_BE41_7C11_E1C1),
            ("ni56", 0xC187_5915_1BBC_EA58),
        ],
        key_ddxdt_dt: &[
            ("neut", 0xBF48_57E8_29F9_DBE4),
            ("he4", 0xBFB6_AFCE_CD91_DC54),
            ("si28", 0xBFD4_8FBC_6BCE_EC2A),
            ("fe54", 0xBECA_EB93_B798_D89C),
            ("ni56", 0xBF92_AF25_5313_DB9A),
        ],
    },
    NetCaseSpec {
        id: "rp_153_xrburst",
        net_name: "rp_153.net",
        log_t: 9.10_f64,
        log_rho: 5.8_f64,
        eta: 1.0_f64,
        scr: ScreeningMode::Chugunov,
        use_fl87: false,
        fe56ec_ff: 1.0e-4_f64,
        wk_fac: 1.0_f64,
        neuq_fac: 1.0_f64,
        rfac_scale: 1.0_f64,
        preset: "h_cno",
        max_ulp: 0,
        eps_nuc_bits: 0x45B3_663F_1DC1_3F1A,
        d_eps_nuc_dt_bits: 0x43E5_7B97_23A2_5723,
        d_eps_nuc_drho_bits: 0x4480_5747_75BD_070F,
        eps_neu_bits: 0x4248_E640_47EB_E2C4,
        first_rxn: (0, "r_al22_ng_al23", 0x4202_90E5_864B_F1A2),
        key_dxdt: &[
            ("h1", 0xC1D2_AF6C_7B2A_D9AD),
            ("he4", 0x41F2_A706_5F33_3045),
            ("c12", 0x420B_F057_9BCC_E993),
            ("o15", 0x414E_8778_48DE_B277),
            ("ne20", 0xC0E8_E4E5_D70D_9F91),
        ],
        key_ddxdt_dt: &[
            ("h1", 0xC004_B05B_C189_2BAC),
            ("he4", 0x4024_A83B_C360_C77C),
            ("c12", 0x403E_F281_6CCD_F3C3),
            ("o15", 0x3F6A_901D_F885_5BAD),
            ("ne20", 0xBF32_AA59_4600_A777),
        ],
    },
    NetCaseSpec {
        id: "approx21_fl87_on",
        net_name: "approx21.net",
        log_t: 8.15_f64,
        log_rho: 5.8_f64,
        eta: 12.0_f64,
        scr: ScreeningMode::Chugunov,
        use_fl87: true,
        fe56ec_ff: 1.0e-7_f64,
        wk_fac: 1.0_f64,
        neuq_fac: 1.0_f64,
        rfac_scale: 1.0_f64,
        preset: "he_rich",
        max_ulp: 0,
        eps_nuc_bits: 0x431F_0E8E_FEF6_A0F8,
        d_eps_nuc_dt_bits: 0x419D_7564_436E_1790,
        d_eps_nuc_drho_bits: 0x41ED_E34B_F5A8_F5A8,
        eps_neu_bits: 0x42E0_0015_CCD5_53FD,
        first_rxn: (0, "r_ar36_ag_ca40", 0x39CF_ED5F_36C1_58DC),
        key_dxdt: &[
            ("he4", 0x3EE0_1BC2_AAA5_1AEB),
            ("c12", 0xBF63_E9CF_5DCC_B3ED),
            ("o16", 0xBEE4_65CA_329A_9EE1),
            ("fe56", 0x3C6C_3108_2FB1_7979),
        ],
        key_ddxdt_dt: &[
            ("he4", 0x3D46_262F_E140_9681),
            ("c12", 0xBDE3_29FF_EE07_086D),
            ("o16", 0xBD66_5B46_BDDA_62F0),
            ("fe56", 0x0000_0000_0000_0000),
        ],
    },
    NetCaseSpec {
        id: "approx21_fl87_off",
        net_name: "approx21.net",
        log_t: 8.15_f64,
        log_rho: 5.8_f64,
        eta: 12.0_f64,
        scr: ScreeningMode::Chugunov,
        use_fl87: false,
        fe56ec_ff: 1.0e-7_f64,
        wk_fac: 1.0_f64,
        neuq_fac: 1.0_f64,
        rfac_scale: 1.0_f64,
        preset: "he_rich",
        max_ulp: 0,
        eps_nuc_bits: 0x431F_0E8E_2D01_01FA,
        d_eps_nuc_dt_bits: 0x419D_7561_A19A_3512,
        d_eps_nuc_drho_bits: 0x41ED_E34B_900A_891A,
        eps_neu_bits: 0x42E0_0015_CCD5_53FD,
        first_rxn: (0, "r_ar36_ag_ca40", 0x39CF_ED5F_36C1_58DC),
        key_dxdt: &[
            ("he4", 0x3EE0_1C91_96BF_507A),
            ("c12", 0xBF63_E9D0_2CB8_CE23),
            ("o16", 0xBEE4_65CA_329A_9EE1),
            ("fe56", 0x3C6C_3108_2FB1_7979),
        ],
        key_ddxdt_dt: &[
            ("he4", 0x3D46_3090_338E_BA2B),
            ("c12", 0xBDE3_2A02_861B_9BF6),
            ("o16", 0xBD66_5B46_BDDA_62F0),
            ("fe56", 0x0000_0000_0000_0000),
        ],
    },
    NetCaseSpec {
        id: "approx21_co56_nse",
        net_name: "approx21_plus_co56.net",
        log_t: 9.55_f64,
        log_rho: 7.2_f64,
        eta: 2.0_f64,
        scr: ScreeningMode::Chugunov,
        use_fl87: false,
        fe56ec_ff: 1.0e-7_f64,
        wk_fac: 0.5_f64,
        neuq_fac: 0.7_f64,
        rfac_scale: 0.8_f64,
        preset: "nse_approx21",
        max_ulp: 1,
        eps_nuc_bits: 0x4560_FD63_4F18_D2CC,
        d_eps_nuc_dt_bits: 0x42C7_C26E_8FE3_1310,
        d_eps_nuc_drho_bits: 0x43E2_204C_5520_3A11,
        eps_neu_bits: 0x425C_79C9_A8CC_7368,
        first_rxn: (0, "r_ar36_ag_ca40", 0x4144_24AE_D4B6_C5D5),
        key_dxdt: &[
            ("neut", 0xC16E_2EEC_950B_D1B9),
            ("prot", 0xC11D_B775_0DDF_C5D6),
            ("he4", 0x4126_B335_0D87_C952),
            ("si28", 0xC106_4283_CBE9_E50C),
            ("fe54", 0xC1AC_7CC1_EC71_1F80),
            ("co56", 0x3F11_3DA6_04AD_14EF),
            ("ni56", 0x4143_B894_D1DD_9F63),
        ],
        key_ddxdt_dt: &[
            ("neut", 0xBF31_8B4A_6093_CE3E),
            ("prot", 0x3F43_A0FD_8C94_CDEB),
            ("he4", 0xBF4B_4792_0B6E_6D43),
            ("si28", 0xBF33_E095_F1CA_692C),
            ("fe54", 0xBF87_01CC_03FA_C05A),
            ("co56", 0x0000_0000_0000_0000),
            ("ni56", 0xBF7A_14F5_2846_A8E3),
        ],
    },
    NetCaseSpec {
        id: "approx21_cr60_nse",
        net_name: "approx21_cr60_plus_co56.net",
        log_t: 9.75_f64,
        log_rho: 8.5_f64,
        eta: 5.0_f64,
        scr: ScreeningMode::Chugunov,
        use_fl87: false,
        fe56ec_ff: 1.0e-7_f64,
        wk_fac: 1.0_f64,
        neuq_fac: 1.0_f64,
        rfac_scale: 1.0_f64,
        preset: "nse_approx21",
        max_ulp: 1,
        eps_nuc_bits: 0x45A8_22BC_FA7D_52F9,
        d_eps_nuc_dt_bits: 0xC3A1_1FA9_CFEC_B91C,
        d_eps_nuc_drho_bits: 0x43E5_5102_2E1A_C988,
        eps_neu_bits: 0x42F1_4B74_B6E2_75F5,
        first_rxn: (0, "r_ar36_ag_ca40", 0x41CA_D1D2_2EE9_783F),
        key_dxdt: &[
            ("neut", 0xC1B5_4CE5_15FE_5DAC),
            ("prot", 0xC156_2BC2_D6F3_4DEC),
            ("he4", 0xC167_38F6_DE66_EE81),
            ("si28", 0xC186_A10B_05A0_F1F9),
            ("fe54", 0xC1F3_3484_DB77_B8F7),
            ("co56", 0x3FA0_7BCB_F3D3_8AB2),
            ("ni56", 0xC163_0B05_CBBA_EAE9),
            ("cr60", 0x3E51_A1D2_4CA9_4889),
        ],
        key_ddxdt_dt: &[
            ("neut", 0x3FAB_F10C_BC83_0311),
            ("prot", 0x3F9A_C1BE_1736_C9FD),
            ("he4", 0xBFAF_D264_3163_0C83),
            ("si28", 0xBF94_C6F9_9555_2C9E),
            ("fe54", 0x3FEC_3655_AEFE_0F50),
            ("co56", 0x0000_0000_0000_0000),
            ("ni56", 0xBFB0_18D2_6AF6_C141),
            ("cr60", 0x0000_0000_0000_0000),
        ],
    },
];

#[test]
fn net_get_all_16_isolated_mesa_26_4_1_hex_oracles() {
    with_numerics_mode(NumericsMode::Strict, || {
        for c in NET_CASES {
            run_net_case_spec(c);
        }
    });
}

#[test]
fn net_get_symbolic_forwards_use_3a_fl87_exact_0ulp() {
    with_numerics_mode(NumericsMode::Strict, || {
        let mut g_a21 = NetGeneralInfo::from_name("approx21.net").unwrap();
        g_a21.fe56ec_fake_factor = 1.0e-7;
        g_a21.use_3a_fl87 = true;
        let x_a21 = fill_preset_comp("he_rich", &g_a21);
        let cat = NuclideCatalog::builtin_ref();
        let comp =
            get_composition_info(cat, &g_a21.chem_id, &x_a21, false).unwrap();
        let temp = exp10(8.15);
        let rho = exp10(5.8);
        let rf_a21 = vec![1.0; g_a21.num_reactions];
        let full = net_get(
            &g_a21,
            false,
            &x_a21,
            temp,
            8.15,
            rho,
            5.8,
            comp.abar,
            comp.zbar,
            comp.z2bar,
            comp.ye,
            12.0,
            0.0,
            0.0,
            &rf_a21,
            1.0,
            ScreeningMode::Chugunov,
        )
        .unwrap();
        let sym = net_get_symbolic_d_dxdt_dx(
            &g_a21,
            &x_a21,
            temp,
            8.15,
            rho,
            5.8,
            comp.abar,
            comp.zbar,
            comp.z2bar,
            comp.ye,
            12.0,
            0.0,
            0.0,
            &rf_a21,
            1.0,
            ScreeningMode::Chugunov,
        )
        .unwrap();
        for idx in 0..(g_a21.num_isos * g_a21.num_isos) {
            assert_ulp_le(
                &format!("approx21_sym_jac[{idx}]"),
                sym[idx],
                full.d_dxdt_dx[idx].to_bits(),
                0,
            );
        }
    });
}

#[test]
fn weaklib_only_op_code_2_exp_m99_and_reaclib_order_parity() {
    with_numerics_mode(NumericsMode::Strict, || {
        let cat = NuclideCatalog::builtin_ref();
        let checks = [
            (
                "mesa_49.net",
                202,
                "r_f18_ap_ne21",
                0x4205_66E5_06A1_F112,
                372,
                "r_na23_wk-minus_mg23",
                0x3702_0A71_7E64_A9BC,
                9.35_f64,
                6.8_f64,
                1.5_f64,
                "si_rich",
            ),
            (
                "mesa_80.net",
                303,
                "r_p30_pa_si27",
                0x4176_A933_D667_8226,
                573,
                "r_na23_wk-minus_mg23",
                0x3702_0A71_7E64_A9BC,
                9.55_f64,
                7.5_f64,
                2.5_f64,
                "si_rich",
            ),
            (
                "rp_153.net",
                804,
                "r_p26_ap_s29",
                0x4062_4EB6_E02B_F943,
                1608,
                "r_v49_wk-minus_cr49",
                0x3702_0A71_7E64_A9BC,
                9.10_f64,
                5.8_f64,
                1.0_f64,
                "h_cno",
            ),
        ];
        for (
            net_nm,
            mid_idx,
            mid_nm,
            mid_bits,
            wk_idx,
            wk_nm,
            wk_bits,
            lt,
            lr,
            eta,
            preset,
        ) in checks
        {
            let g = NetGeneralInfo::from_name(net_nm).unwrap();
            assert_eq!(g.reactions[mid_idx].name, mid_nm, "{net_nm} mid order");
            assert_eq!(g.reactions[wk_idx].name, wk_nm, "{net_nm} wk order");
            let x = fill_preset_comp(preset, &g);
            let cmp = get_composition_info(cat, &g.chem_id, &x, false).unwrap();
            let rf = vec![1.0; g.num_reactions];
            let r = net_get(
                &g,
                true,
                &x,
                exp10(lt),
                lt,
                exp10(lr),
                lr,
                cmp.abar,
                cmp.zbar,
                cmp.z2bar,
                cmp.ye,
                eta,
                0.0,
                0.0,
                &rf,
                1.0,
                ScreeningMode::Chugunov,
            )
            .unwrap();
            assert_ulp_le(
                &format!("{net_nm}.{mid_nm}"),
                r.rate_raw[mid_idx],
                mid_bits,
                0,
            );
            assert_ulp_le(
                &format!("{net_nm}.{wk_nm}"),
                r.rate_raw[wk_idx],
                wk_bits,
                0,
            );
        }
    });
}

struct CpSpec {
    id: &'static str,
    net_name: &'static str,
    solver: RosenbrockVariant,
    t0: f64,
    log_p: f64,
    t_end: f64,
    h0: f64,
    rtol: f64,
    atol: f64,
    scr: ScreeningMode,
    cnt: (usize, usize, usize, usize, usize),
    init_rho: u64,
    init_lns: u64,
    end_temp: u64,
    end_rho: u64,
    end_lns: u64,
    majors: &'static [(&'static str, u64)],
}

const CP_SPECS: &[CpSpec] = &[
    CpSpec {
        id: "burn8b_rodas4",
        net_name: "approx21.net",
        solver: RosenbrockVariant::Rodas4,
        t0: 3.0e9,
        log_p: 24.0,
        t_end: 1.0e-4,
        h0: 1.0e-6,
        rtol: 1.0e-7,
        atol: 1.0e-8,
        scr: ScreeningMode::Extended,
        cnt: (2110, 419, 422, 419, 0),
        init_rho: 0x4153_11A7_9A6F_F239,
        init_lns: 0x4033_92C6_8083_8920,
        end_temp: 0x41E8_6D4F_3E60_6807,
        end_rho: 0x414E_7228_4DA9_05CC,
        end_lns: 0x4033_B661_32E2_FDC2,
        majors: &[
            ("he4", 0x3F0C_2B61_CCFA_5BA5),
            ("c12", 0x3F78_4EAF_5B07_3BD6),
            ("o16", 0x3FE6_4256_F307_A4DE),
            ("ne20", 0x3F79_F8DF_3BBC_DFC4),
            ("mg24", 0x3FA3_6239_3E0B_5C07),
            ("si28", 0x3FCA_3339_7E75_B12B),
            ("s32", 0x3FA6_B775_B572_295A),
        ],
    },
    CpSpec {
        id: "burn8b_ros3p",
        net_name: "approx21.net",
        solver: RosenbrockVariant::Ros3p,
        t0: 3.0e9,
        log_p: 24.0,
        t_end: 1.0e-4,
        h0: 1.0e-6,
        rtol: 1.0e-7,
        atol: 1.0e-8,
        scr: ScreeningMode::Extended,
        cnt: (878, 436, 439, 436, 0),
        init_rho: 0x4153_11A7_9A6F_F239,
        init_lns: 0x4033_92C6_8083_8920,
        end_temp: 0x41E8_6D4D_EB41_9982,
        end_rho: 0x414E_7233_16BF_3565,
        end_lns: 0x4033_B660_F89D_C13C,
        majors: &[
            ("he4", 0x3F0C_2B94_6B60_CE44),
            ("c12", 0x3F78_4ED2_0092_FCAE),
            ("o16", 0x3FE6_4256_B36A_5AC0),
            ("ne20", 0x3F79_F921_59E6_DF3D),
            ("mg24", 0x3FA3_6242_21AC_4C20),
            ("si28", 0x3FCA_3337_00C4_521B),
            ("s32", 0x3FA6_B76F_9B78_9E2A),
        ],
    },
    CpSpec {
        id: "burn8b_ros3pl",
        net_name: "approx21.net",
        solver: RosenbrockVariant::Ros3pl,
        t0: 3.0e9,
        log_p: 24.0,
        t_end: 1.0e-4,
        h0: 1.0e-6,
        rtol: 1.0e-7,
        atol: 1.0e-8,
        scr: ScreeningMode::Extended,
        cnt: (918, 303, 306, 303, 0),
        init_rho: 0x4153_11A7_9A6F_F239,
        init_lns: 0x4033_92C6_8083_8920,
        end_temp: 0x41E8_6D4F_8AEC_AD8F,
        end_rho: 0x414E_7226_800C_9FB1,
        end_lns: 0x4033_B661_3E00_06AF,
        majors: &[
            ("he4", 0x3F0C_2B4C_D3AC_8E6C),
            ("c12", 0x3F78_4EA0_3DDE_D33E),
            ("o16", 0x3FE6_4257_1C61_2033),
            ("ne20", 0x3F79_F8BD_DBCA_42C3),
            ("mg24", 0x3FA3_6236_AB99_FA5D),
            ("si28", 0x3FCA_333A_31C1_EE5F),
            ("s32", 0x3FA6_B778_4BFD_7D8B),
        ],
    },
    CpSpec {
        id: "burn8b_rodasp",
        net_name: "approx21.net",
        solver: RosenbrockVariant::Rodasp,
        t0: 3.0e9,
        log_p: 24.0,
        t_end: 1.0e-4,
        h0: 1.0e-6,
        rtol: 1.0e-7,
        atol: 1.0e-8,
        scr: ScreeningMode::Extended,
        cnt: (1695, 336, 339, 336, 0),
        init_rho: 0x4153_11A7_9A6F_F239,
        init_lns: 0x4033_92C6_8083_8920,
        end_temp: 0x41E8_6D4F_CC7A_01E1,
        end_rho: 0x414E_7225_9788_0C26,
        end_lns: 0x4033_B661_461B_6354,
        majors: &[
            ("he4", 0x3F0C_2B4C_6BAC_356F),
            ("c12", 0x3F78_4E99_EC17_B73E),
            ("o16", 0x3FE6_4257_1AE6_3FDA),
            ("ne20", 0x3F79_F8B6_4100_6FEB),
            ("mg24", 0x3FA3_6235_4463_953E),
            ("si28", 0x3FCA_333A_CBF9_F62F),
            ("s32", 0x3FA6_B778_FF05_1206),
        ],
    },
    CpSpec {
        id: "burn_cp_coburn",
        net_name: "co_burn.net",
        solver: RosenbrockVariant::Rodas4,
        t0: 1.0e9,
        log_p: 23.5,
        t_end: 1.0,
        h0: 1.0e-3,
        rtol: 1.0e-7,
        atol: 1.0e-8,
        scr: ScreeningMode::Chugunov,
        cnt: (35, 7, 7, 7, 0),
        init_rho: 0x4150_0204_0DD5_434B,
        init_lns: 0x4032_F5E6_4FF6_EB98,
        end_temp: 0x41CD_CD80_9073_D6A4,
        end_rho: 0x4150_01FF_1364_E700,
        end_lns: 0x4032_F5E6_B00C_C829,
        majors: &[
            ("he4", 0x3E02_3499_0D64_DB9D),
            ("c12", 0x3FD7_AE07_63ED_5B27),
            ("o16", 0x3FE4_28F1_697B_CF3E),
            ("ne20", 0x3ED5_C87E_E2E8_AC17),
            ("mg24", 0x3DB2_1A0B_11FC_CB10),
            ("si28", 0x3DB7_E09C_12D6_E444),
        ],
    },
    CpSpec {
        id: "burn_cp_mesa49",
        net_name: "mesa_49.net",
        solver: RosenbrockVariant::Rodas4,
        t0: 3.5e9,
        log_p: 24.2,
        t_end: 1.0e-4,
        h0: 1.0e-7,
        rtol: 1.0e-6,
        atol: 1.0e-8,
        scr: ScreeningMode::Chugunov,
        cnt: (1020, 203, 204, 203, 1),
        init_rho: 0x4158_1FCA_6559_3D0B,
        init_lns: 0x4033_A835_BAFD_E3B7,
        end_temp: 0x41EB_ACA9_8E43_5C9E,
        end_rho: 0x4154_0901_760F_7DB7,
        end_lns: 0x4033_C368_6E8E_DCFB,
        majors: &[
            ("he4", 0x3F0C_21CE_C292_A681),
            ("c12", 0x3F3C_6B34_6BB1_85BC),
            ("o16", 0x3FE6_2308_D13B_FBC5),
            ("ne20", 0x3F3A_A596_6E1E_0EB6),
            ("mg24", 0x3F7C_5CD8_45EE_67D3),
            ("si28", 0x3FCC_39D3_9D1C_5E0B),
            ("s32", 0x3FB4_2614_55F0_9F98),
            ("si27", 0x3EE5_ABA4_831E_C5CD),
        ],
    },
];

#[test]
fn burn_1_zone_const_p_all_solvers_mesa_26_4_1_parity() {
    with_numerics_mode(NumericsMode::Strict, || {
        let ht = HelmTable::builtin().unwrap();
        let eos_cfg = EosConfig::default();
        let cat = NuclideCatalog::builtin_ref();
        for s in CP_SPECS {
            let g = NetGeneralInfo::from_name(s.net_name).unwrap();
            let mut x0 = vec![0.0_f64; g.num_isos];
            let c12_slot = g.net_iso[cat.get_nuclide_index("c12").unwrap()];
            let o16_slot = g.net_iso[cat.get_nuclide_index("o16").unwrap()];
            if c12_slot > 0 {
                x0[c12_slot - 1] = 0.37;
            }
            if o16_slot > 0 {
                x0[o16_slot - 1] = 0.63;
            }
            let rf = vec![1.0_f64; g.num_reactions];
            let out = burn_1_zone_const_p(
                &g,
                ht,
                &eos_cfg,
                &BurnConstPInput {
                    solver: s.solver,
                    starting_temp: s.t0,
                    starting_x: &x0,
                    clip: false,
                    t_end: s.t_end,
                    pressure: exp10(s.log_p),
                    rate_factors: &rf,
                    weak_rate_factor: 1.0,
                    reaction_qs: &[],
                    reaction_neu_qs: &[],
                    screening_mode: s.scr,
                    h0: s.h0,
                    max_step_size: 1.0,
                    max_steps: 10000,
                    rtol: s.rtol,
                    atol: s.atol,
                    x_min: -1.0e-10,
                },
            )
            .unwrap();
            assert_eq!(
                (out.nfcn, out.njac, out.nstep, out.naccpt, out.nrejct),
                s.cnt,
                "{}: Rosenbrock counts mismatch",
                s.id,
            );
            assert_rel_le(
                &format!("{}.initial_rho", s.id),
                out.initial_rho,
                s.init_rho,
                1.0e-12,
            );
            assert_rel_le(
                &format!("{}.initial_lns", s.id),
                out.initial_lns,
                s.init_lns,
                1.0e-14,
            );
            assert_rel_le(
                &format!("{}.ending_temp", s.id),
                out.ending_temp,
                s.end_temp,
                1.0e-13,
            );
            assert_rel_le(
                &format!("{}.ending_rho", s.id),
                out.ending_rho,
                s.end_rho,
                1.0e-12,
            );
            assert_rel_le(
                &format!("{}.ending_lns", s.id),
                out.ending_lns,
                s.end_lns,
                1.0e-13,
            );
            let im = iso_map(&g);
            for &(iso, bits) in s.majors {
                assert_rel_le(
                    &format!("{}.x[{iso}]", s.id),
                    out.ending_x[im[iso]],
                    bits,
                    1.0e-11,
                );
            }
        }
    });
}

struct CdSpec {
    id: &'static str,
    net_name: &'static str,
    lt0: f64,
    lr0: f64,
    t_end: f64,
    preset: &'static str,
    max_rel: f64,
    cnt: (usize, usize, usize, usize, usize),
    end_logt: u64,
    avg_eps: u64,
    eps_neu: u64,
    majors: &'static [(&'static str, u64)],
}

const CD_SPECS: &[CdSpec] = &[
    CdSpec {
        id: "burn_cd_approx21",
        net_name: "approx21.net",
        lt0: 9.0,
        lr0: 7.0,
        t_end: 1.0e-2,
        preset: "co_rich",
        max_rel: 1.0e-15,
        cnt: (914, 24, 24, 24, 0),
        end_logt: 0x4022_295E_AE0E_A9F4,
        avg_eps: 0x43B0_EB57_415C_89AB,
        eps_neu: 0x4112_1298_621C_8220,
        majors: &[
            ("he4", 0x3E5A_08EB_92A9_6864),
            ("c12", 0x3FDD_795B_C0AD_49E2),
            ("o16", 0x3FDC_F1F7_1F40_06A0),
            ("ne20", 0x3FB3_D4DA_CB72_7236),
            ("mg24", 0x3F7E_CB16_5DC2_477D),
            ("si28", 0x3F62_2247_6490_2B2D),
            ("s32", 0x3EB5_AAA1_CD77_E898),
            ("fe54", 0x3D72_019E_F3D6_D0A6),
            ("ni56", 0x3D72_019A_C1DB_2968),
        ],
    },
    CdSpec {
        id: "burn_cd_coburn",
        net_name: "co_burn.net",
        lt0: 8.9,
        lr0: 7.2,
        t_end: 1.0,
        preset: "co_rich",
        max_rel: 1.0e-12,
        cnt: (3691, 101, 101, 93, 8),
        end_logt: 0x4021_CCCC_D180_8DDE,
        avg_eps: 0x4345_71E4_4C3F_1FA4,
        eps_neu: 0x3F25_E8D8_15A5_B10F,
        majors: &[
            ("c12", 0x3FDD_795B_0EAA_8E48),
            ("o16", 0x3FDC_EC5E_196A_B419),
            ("ne20", 0x3FB4_0DE0_6C27_2290),
            ("mg24", 0x3F7D_1CE9_6C0E_1859),
            ("si28", 0x3F61_2D8C_BFF2_F992),
        ],
    },
    CdSpec {
        id: "burn_cd_ppcno",
        net_name: "pp_and_cno_extras.net",
        lt0: 7.4,
        lr0: 2.5,
        t_end: 1.0e9,
        preset: "h_solar",
        max_rel: 1.0e-9,
        cnt: (11540, 154, 154, 125, 29),
        end_logt: 0x401E_46AE_0FD1_C51E,
        avg_eps: 0xC11B_80DE_C797_2B14,
        eps_neu: 0x4142_5130_23B6_59B3,
        majors: &[
            ("he4", 0x3FE4_38A0_1E87_0D76),
            ("c12", 0x3F4D_79C5_B98C_7D54),
            ("o16", 0x3F84_87FF_FD9F_486A),
            ("ne20", 0x3F60_6ED5_BC8A_D03F),
            ("mg24", 0x3F50_6ED7_236C_7B46),
            ("c13", 0x3F35_6897_9ECE_5865),
        ],
    },
    CdSpec {
        id: "burn_cd_mesa49",
        net_name: "mesa_49.net",
        lt0: 9.2,
        lr0: 6.8,
        t_end: 1.0e-3,
        preset: "si_rich",
        max_rel: 1.0e-10,
        cnt: (3082, 40, 40, 38, 2),
        end_logt: 0x4022_6666_6701_30F4,
        avg_eps: 0x4416_DCF5_3548_AEC8,
        eps_neu: 0x428F_7227_4243_F6F4,
        majors: &[
            ("he4", 0x3EC3_179E_FD2B_D33F),
            ("c12", 0x3E8F_6A2F_F883_17F5),
            ("o16", 0x3FB3_3A2A_4A52_EAC2),
            ("ne20", 0x3FB3_02FF_4FE6_4F8F),
            ("mg24", 0x3FBB_64EE_242A_EF62),
            ("si28", 0x3FE0_EEF8_6B07_5068),
            ("s32", 0x3FCB_58AF_897B_7FBB),
            ("na21", 0x3F2A_8EBA_9883_5CC1),
            ("si27", 0x3F03_B887_8F0B_6D2C),
        ],
    },
];

#[test]
fn burn_1_zone_const_density_all_nets_mesa_26_4_1_parity() {
    with_numerics_mode(NumericsMode::Strict, || {
        let ht = HelmTable::builtin().unwrap();
        let eos_cfg = EosConfig::default();
        for (idx, s) in CD_SPECS.iter().enumerate() {
            let g = NetGeneralInfo::from_name(s.net_name).unwrap();
            let x0 = fill_preset_comp(s.preset, &g);
            let rf = vec![1.0_f64; g.num_reactions];
            let out = burn_1_zone_const_density(
                &g,
                ht,
                &eos_cfg,
                &BurnConstDensityInput {
                    t_start: 0.0,
                    t_end: s.t_end,
                    starting_x: &x0,
                    starting_log10_t: s.lt0,
                    log10_rho: s.lr0,
                    rate_factors: &rf,
                    weak_rate_factor: 1.0,
                    screening_mode: ScreeningMode::Chugunov,
                    stptry_in: 0.0,
                    max_steps: 50000,
                    eps: 1.0e-7,
                    odescal: 1.0e-10,
                },
            )
            .unwrap();
            assert_eq!(
                (out.nfcn, out.njac, out.nstep, out.naccpt, out.nrejct),
                s.cnt,
                "{}: const-density counts mismatch",
                s.id,
            );
            let im = iso_map(&g);
            if idx == 0 {
                // `burn_cd_approx21` is 0-ULP bit-identical across all outputs.
                assert_ulp_le(
                    "burn_cd_approx21.end_logt",
                    out.ending_log10_t,
                    s.end_logt,
                    0,
                );
                assert_ulp_le(
                    "burn_cd_approx21.avg_eps",
                    out.avg_eps_nuc,
                    s.avg_eps,
                    0,
                );
                assert_ulp_le(
                    "burn_cd_approx21.eps_neu",
                    out.eps_neu_total,
                    s.eps_neu,
                    0,
                );
                for &(iso, bits) in s.majors {
                    assert_ulp_le(
                        &format!("burn_cd_approx21.x[{iso}]"),
                        out.ending_x[im[iso]],
                        bits,
                        0,
                    );
                }
            } else {
                assert_rel_le(
                    &format!("{}.end_logt", s.id),
                    out.ending_log10_t,
                    s.end_logt,
                    s.max_rel,
                );
                assert_rel_le(
                    &format!("{}.avg_eps", s.id),
                    out.avg_eps_nuc,
                    s.avg_eps,
                    s.max_rel,
                );
                assert_rel_le(
                    &format!("{}.eps_neu", s.id),
                    out.eps_neu_total,
                    s.eps_neu,
                    s.max_rel,
                );
                for &(iso, bits) in s.majors {
                    assert_rel_le(
                        &format!("{}.x[{iso}]", s.id),
                        out.ending_x[im[iso]],
                        bits,
                        s.max_rel,
                    );
                }
            }
        }
    });
}

#[test]
fn burn_1_zone_case8a_mesa_26_4_1_parity() {
    with_numerics_mode(NumericsMode::Strict, || {
        let ht = HelmTable::builtin().unwrap();
        let eos_cfg = EosConfig::default();
        let cat = NuclideCatalog::builtin_ref();
        let g = NetGeneralInfo::from_name("approx21.net").unwrap();
        let ihe4 = g.net_iso[cat.get_nuclide_index("he4").unwrap()] - 1;
        let mut x0 = vec![0.0_f64; g.num_isos];
        x0[ihe4] = 1.0;
        let times = [0.0, 100.0, 150.0];
        let lts = [8.3, 8.4, 8.3];
        let lrs = [7.0, 7.5, 8.1];
        let etas = [0.0, 0.0, 0.0];
        let rf = vec![1.0_f64; g.num_reactions];
        let out = burn_1_zone(
            &g,
            ht,
            &eos_cfg,
            &Burn1ZoneInput {
                t_start: 0.0,
                t_end: 150.0,
                starting_x: &x0,
                times: &times,
                log10_ts: &lts,
                log10_rhos: &lrs,
                etas: &etas,
                dxdt_source_term: None,
                rate_factors: &rf,
                weak_rate_factor: 1.0,
                reaction_qs: &[],
                reaction_neu_qs: &[],
                screening_mode: ScreeningMode::Extended,
                stptry_in: 0.0,
                max_steps: 50000,
                eps: 1.0e-8,
                odescal: 1.0e-12,
            },
        )
        .unwrap();
        assert_eq!(
            (out.nfcn, out.njac, out.nstep, out.naccpt, out.nrejct),
            (1531, 27, 27, 23, 4),
            "burn8a_approx21 counts mismatch",
        );
        assert_rel_le(
            "burn8a.avg_eps",
            out.avg_eps_nuc,
            0x432B_5A7E_5D5D_097E,
            1.0e-10,
        );
        assert_rel_le(
            "burn8a.eps_neu",
            out.eps_neu_total,
            0x3E9D_6305_92AC_0998,
            1.0e-8,
        );
        let im = iso_map(&g);
        for &(iso, bits) in &[
            ("he4", 0x3F8A_3808_6E3C_7B72),
            ("c12", 0x3FEF_96AD_CD88_C288),
            ("o16", 0x3F0C_842F_7EF4_2FC1),
        ] {
            assert_rel_le(
                &format!("burn8a.x[{iso}]"),
                out.ending_x[im[iso]],
                bits,
                1.0e-9,
            );
        }
    });
}

struct B1Spec {
    id: &'static str,
    net_name: &'static str,
    lt0: f64,
    lr0: f64,
    t_end: f64,
    preset: &'static str,
    max_rel: f64,
    cnt: (usize, usize, usize, usize, usize),
    avg_eps: u64,
    eps_neu: u64,
    majors: &'static [(&'static str, u64)],
}

const B1_SPECS: &[B1Spec] = &[
    B1Spec {
        id: "burn1_ppcno",
        net_name: "pp_and_cno_extras.net",
        lt0: 7.3,
        lr0: 2.2,
        t_end: 1.0e11,
        preset: "h_solar",
        max_rel: 1.0e-11,
        cnt: (1075, 39, 39, 39, 0),
        avg_eps: 0x40AF_25BF_EC94_8026,
        eps_neu: 0x4075_8119_10D7_A304,
        majors: &[
            ("he4", 0x3FE4_3882_0D51_C717),
            ("c12", 0x3F63_3D17_4CA4_1E23),
            ("o16", 0x3F84_8A6D_B176_46BE),
            ("ne20", 0x3F60_6ED7_1DE5_308C),
            ("mg24", 0x3F50_6ED7_236C_7B46),
            ("c13", 0x3F3D_64A8_66B4_5133),
            ("n13", 0x3D97_FB0A_DAC0_A3B3),
        ],
    },
    B1Spec {
        id: "burn1_coburn",
        net_name: "co_burn.net",
        lt0: 8.95,
        lr0: 6.8,
        t_end: 10.0,
        preset: "co_rich",
        max_rel: 1.0e-12,
        cnt: (3622, 100, 100, 92, 8),
        avg_eps: 0x4311_1C7E_FD71_F9E0,
        eps_neu: 0x3F19_6406_7718_C4D5,
        majors: &[
            ("he4", 0x3DE8_BA61_C57D_1994),
            ("c12", 0x3FDD_796B_BE27_A926),
            ("o16", 0x3FDC_E9C1_94D3_C7D4),
            ("ne20", 0x3FB4_24C4_6977_9D1B),
            ("mg24", 0x3F7C_39E1_5AF1_ADD6),
            ("si28", 0x3F61_5D09_41FB_AC4E),
        ],
    },
    B1Spec {
        id: "burn1_cnoextras",
        net_name: "cno_extras.net",
        lt0: 8.3,
        lr0: 3.8,
        t_end: 100.0,
        preset: "h_cno",
        max_rel: 1.0e-12,
        cnt: (1887, 35, 35, 35, 0),
        avg_eps: 0x4322_2B90_B9B3_8318,
        eps_neu: 0x42E5_19C8_B7CA_666A,
        majors: &[
            ("he4", 0x3FD5_E2C7_1213_D895),
            ("c12", 0x3F00_941D_C5CE_6AB0),
            ("o16", 0x3EB0_84D1_58B6_4B25),
            ("ne20", 0x3F73_AC88_DDD7_81A7),
            ("c13", 0x3E2B_9FA6_D321_9BE1),
            ("n13", 0x3F2B_957B_144E_5EB6),
        ],
    },
    B1Spec {
        id: "burn1_mesa49",
        net_name: "mesa_49.net",
        lt0: 9.3,
        lr0: 6.5,
        t_end: 1.0e-2,
        preset: "si_rich",
        max_rel: 1.0e-12,
        cnt: (3542, 48, 48, 48, 0),
        avg_eps: 0x43E3_68F4_1F8A_C539,
        eps_neu: 0x42A2_652E_30E2_FD36,
        majors: &[
            ("he4", 0x3E51_F15B_BEBF_FE9C),
            ("c12", 0x3E5D_41F6_A755_66D2),
            ("o16", 0x3FBB_AFDB_6B8C_030F),
            ("ne20", 0x3FA3_73AD_F285_C674),
            ("mg24", 0x3FB4_7DE0_2A32_63FA),
            ("si28", 0x3FE0_72E5_D3AD_1F4A),
            ("s32", 0x3FD0_9160_51E0_01F8),
            ("na21", 0x3F05_7458_7D19_AEA1),
            ("si27", 0x3F0F_A72F_CF01_F2FE),
        ],
    },
];

#[test]
fn burn_1_zone_other_nets_mesa_26_4_1_parity() {
    with_numerics_mode(NumericsMode::Strict, || {
        let ht = HelmTable::builtin().unwrap();
        let eos_cfg = EosConfig::default();

        for s in B1_SPECS {
            let g = NetGeneralInfo::from_name(s.net_name).unwrap();
            let x0 = fill_preset_comp(s.preset, &g);
            let rf = vec![1.0_f64; g.num_reactions];
            let times = [s.t_end];
            let lts = [s.lt0];
            let lrs = [s.lr0];
            let etas = [0.0];
            let out = burn_1_zone(
                &g,
                ht,
                &eos_cfg,
                &Burn1ZoneInput {
                    t_start: 0.0,
                    t_end: s.t_end,
                    starting_x: &x0,
                    times: &times,
                    log10_ts: &lts,
                    log10_rhos: &lrs,
                    etas: &etas,
                    dxdt_source_term: None,
                    rate_factors: &rf,
                    weak_rate_factor: 1.0,
                    reaction_qs: &[],
                    reaction_neu_qs: &[],
                    screening_mode: ScreeningMode::Chugunov,
                    stptry_in: 0.0,
                    max_steps: 50000,
                    eps: 1.0e-7,
                    odescal: 1.0e-10,
                },
            )
            .unwrap();
            assert_eq!(
                (out.nfcn, out.njac, out.nstep, out.naccpt, out.nrejct),
                s.cnt,
                "{}: burn1 counts mismatch",
                s.id,
            );
            assert_rel_le(
                &format!("{}.avg_eps", s.id),
                out.avg_eps_nuc,
                s.avg_eps,
                s.max_rel,
            );
            assert_rel_le(
                &format!("{}.eps_neu", s.id),
                out.eps_neu_total,
                s.eps_neu,
                s.max_rel,
            );
            let im = iso_map(&g);
            for &(iso, bits) in s.majors {
                assert_rel_le(
                    &format!("{}.x[{iso}]", s.id),
                    out.ending_x[im[iso]],
                    bits,
                    s.max_rel,
                );
            }
        }
    });
}
