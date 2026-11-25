// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone MESA `26.4.1` IEEE-754 hex parity unit tests for `#161` `flexmesa-kap` fixes.

#![forbid(unsafe_code)]
#![allow(clippy::too_many_lines, clippy::unreadable_literal)]

use flexmesa_const::LN10;
use flexmesa_kap::{
    get_kap_results, get_kap_results_blend_t, KapCoOption, KapConfig,
    KapLowTOption, KapOption, I_FRAC_COMPTON, I_FRAC_HIGH_T, I_FRAC_LOW_T,
};
use flexmesa_math::{with_numerics_mode, NumericsMode};

struct KapOracleRow {
    tag: &'static str,
    cfg_kind: &'static str,
    lnd: u64,
    lnt: u64,
    x: u64,
    z: u64,
    xc: u64,
    xn: u64,
    xo: u64,
    xne: u64,
    frac_low_t: u64,
    frac_high_t: u64,
    frac_type2: u64,
    kap_rad: u64,
    rad_dlnd: u64,
    rad_dlnt: u64,
}

const ROWS: &[KapOracleRow] = &[
    KapOracleRow {
        tag: "kapcn_t0",
        cfg_kind: "inl_gn93_cn.txt",
        lnd: 0xC032E1962D337736,
        lnt: 0x401E64E54284CC64,
        x: 0x3FE5C28F5C28F5C3,
        z: 0x3F948E4F6FD76F00,
        xc: 0x3F747AE147AE147B,
        xn: 0x3F60624DD2F1A9FC,
        xo: 0x3F826E978D4FDF3B,
        xne: 0x0000000000000000,
        frac_low_t: 0x3FF0000000000000,
        frac_high_t: 0x0000000000000000,
        frac_type2: 0x0000000000000000,
        kap_rad: 0x3F808FABE9CEA8D2,
        rad_dlnd: 0x3F95316FAE2D0AD7,
        rad_dlnt: 0x400B71E2FD2A2BF9,
    },
    KapOracleRow {
        tag: "kapcn_t1",
        cfg_kind: "inl_gn93_cn.txt",
        lnd: 0xC032E1962D337736,
        lnt: 0x4020592DBCFDBB83,
        x: 0x3FE5C28F5C28F5C3,
        z: 0x3F948E4F6FD76F00,
        xc: 0x3F747AE147AE147B,
        xn: 0x3F60624DD2F1A9FC,
        xo: 0x3F826E978D4FDF3B,
        xne: 0x0000000000000000,
        frac_low_t: 0x3FF0000000000000,
        frac_high_t: 0x0000000000000000,
        frac_type2: 0x0000000000000000,
        kap_rad: 0x3F731541731E124B,
        rad_dlnd: 0x3FE130303ED2D2FF,
        rad_dlnt: 0x4023C20AA75FA0E0,
    },
    KapOracleRow {
        tag: "kapcn_t2",
        cfg_kind: "inl_gn93_cn.txt",
        lnd: 0xC032E1962D337736,
        lnt: 0x4021BADB117821E5,
        x: 0x3FE5C28F5C28F5C3,
        z: 0x3F948E4F6FD76F00,
        xc: 0x3F747AE147AE147B,
        xn: 0x3F60624DD2F1A9FC,
        xo: 0x3F826E978D4FDF3B,
        xne: 0x0000000000000000,
        frac_low_t: 0x3FF0000000000000,
        frac_high_t: 0x0000000000000000,
        frac_type2: 0x0000000000000000,
        kap_rad: 0x3FE0ABDD3CA4CFD7,
        rad_dlnd: 0x3FD66BA5D7086ACD,
        rad_dlnt: 0x40287498B37128DA,
    },
    KapOracleRow {
        tag: "cn005_t1",
        cfg_kind: "inl_cn_z005.txt",
        lnd: 0xC033926CD770AA67,
        lnt: 0x4020592DBCFDBB83,
        x: 0x3FE199999999999A,
        z: 0x3FA1EB851EB851E0,
        xc: 0x3F889374BC6A7EFA,
        xn: 0x3F80624DD2F1A9FC,
        xo: 0x3F8EB851EB851EB8,
        xne: 0x0000000000000000,
        frac_low_t: 0x3FF0000000000000,
        frac_high_t: 0x0000000000000000,
        frac_type2: 0x0000000000000000,
        kap_rad: 0x3F6FE621D49FCA11,
        rad_dlnd: 0x3FE57E39CF7B79AD,
        rad_dlnt: 0x400328CC7DA370A6,
    },
    KapOracleRow {
        tag: "cn000_t0",
        cfg_kind: "inl_cn_z000.txt",
        lnd: 0xC033926CD770AA67,
        lnt: 0x401D791C5F888824,
        x: 0x3FE199999999999A,
        z: 0x3FA1EB851EB851E0,
        xc: 0x3F889374BC6A7EFA,
        xn: 0x3F80624DD2F1A9FC,
        xo: 0x3F8EB851EB851EB8,
        xne: 0x0000000000000000,
        frac_low_t: 0x3FF0000000000000,
        frac_high_t: 0x0000000000000000,
        frac_type2: 0x0000000000000000,
        kap_rad: 0x3F7E7004E4E49716,
        rad_dlnd: 0xBF9CBA49580B4D55,
        rad_dlnt: 0xC01285A075A41ACA,
    },
    KapOracleRow {
        tag: "cn000_zero_xc",
        cfg_kind: "inl_cn_z000.txt",
        lnd: 0xC033926CD770AA67,
        lnt: 0x401D791C5F888824,
        x: 0x3FE6666666666666,
        z: 0x0000000000000000,
        xc: 0x0000000000000000,
        xn: 0x0000000000000000,
        xo: 0x0000000000000000,
        xne: 0x0000000000000000,
        frac_low_t: 0x3FF0000000000000,
        frac_high_t: 0x0000000000000000,
        frac_type2: 0x0000000000000000,
        kap_rad: 0x3F8272DB4248D74B,
        rad_dlnd: 0xBF9B0EE4AA376FC7,
        rad_dlnt: 0xC01266BDCFB9B253,
    },
    KapOracleRow {
        tag: "opa09_t4_r0_z01",
        cfg_kind: "inl_op_a09_z01.txt",
        lnd: 0xC00D791C5F888818,
        lnt: 0x403230BF82F64406,
        x: 0x3FE6666666666666,
        z: 0x3F847AE147AE1480,
        xc: 0x0000000000000000,
        xn: 0x0000000000000000,
        xo: 0x0000000000000000,
        xne: 0x0000000000000000,
        frac_low_t: 0x0000000000000000,
        frac_high_t: 0x3FA660CEFEA98830,
        frac_type2: 0x0000000000000000,
        kap_rad: 0x3FD33D6BC5785A10,
        rad_dlnd: 0xBF31E778CA182C1A,
        rad_dlnt: 0xBFB9C7178E6F7D2F,
    },
    KapOracleRow {
        tag: "opa09_t4_r0",
        cfg_kind: "inl_op_a09.txt",
        lnd: 0xC00D791C5F888818,
        lnt: 0x403230BF82F64406,
        x: 0x3FE6666666666666,
        z: 0x3F8CAC083126E980,
        xc: 0x0000000000000000,
        xn: 0x0000000000000000,
        xo: 0x0000000000000000,
        xne: 0x0000000000000000,
        frac_low_t: 0x0000000000000000,
        frac_high_t: 0x3FA660CEFEA98830,
        frac_type2: 0x0000000000000000,
        kap_rad: 0x3FD33D6BC5A79795,
        rad_dlnd: 0xBF31E7853AE1BCA0,
        rad_dlnt: 0xBFB88E13A25148F7,
    },
    KapOracleRow {
        tag: "opa09_t2_r0",
        cfg_kind: "inl_op_a09.txt",
        lnd: 0xC015A4F0D62843F4,
        lnt: 0x40319D61F518995D,
        x: 0x3FE6666666666666,
        z: 0x3F8CAC083126E980,
        xc: 0x0000000000000000,
        xn: 0x0000000000000000,
        xo: 0x0000000000000000,
        xne: 0x0000000000000000,
        frac_low_t: 0x0000000000000000,
        frac_high_t: 0x3FE9E73579769DD8,
        frac_type2: 0x0000000000000000,
        kap_rad: 0x3FD43CD1817B490A,
        rad_dlnd: 0x3F3F2C56A46DE403,
        rad_dlnt: 0xBFB08D4C2A8A934A,
    },
    KapOracleRow {
        tag: "op_gs98_t1",
        cfg_kind: "inl_op_gs98.txt",
        lnd: 0xC03569FE9D6932EA,
        lnt: 0x4021F5CD4A3732F5,
        x: 0x3FD6666666666666,
        z: 0x3FBEB851EB851EB8,
        xc: 0x3FAEB851EB851EB8,
        xn: 0x0000000000000000,
        xo: 0x3FA47AE147AE147B,
        xne: 0x0000000000000000,
        frac_low_t: 0x0000000000000000,
        frac_high_t: 0x3FF0000000000000,
        frac_type2: 0x0000000000000000,
        kap_rad: 0x3FE375D788C17105,
        rad_dlnd: 0x3FC43546565C5BB2,
        rad_dlnt: 0x402B911791902B04,
    },
    KapOracleRow {
        tag: "op_gs98_t4",
        cfg_kind: "inl_op_gs98.txt",
        lnd: 0x3FF61AD547A66629,
        lnt: 0x4030941FF5BCCC94,
        x: 0x3FD6666666666666,
        z: 0x3FBEB851EB851EB8,
        xc: 0x3FAEB851EB851EB8,
        xn: 0x0000000000000000,
        xo: 0x3FA47AE147AE147B,
        xne: 0x0000000000000000,
        frac_low_t: 0x0000000000000000,
        frac_high_t: 0x3FF0000000000000,
        frac_type2: 0x0000000000000000,
        kap_rad: 0x3FDB3D012068E40C,
        rad_dlnd: 0x3FCD103779F9066F,
        rad_dlnt: 0xBFE9C2C85A75B320,
    },
    KapOracleRow {
        tag: "t2b_z02_dz",
        cfg_kind: "inl_t2blend_z02.txt",
        lnd: 0x40026BB1BBB55516,
        lnt: 0x40317FE8D8B910D5,
        x: 0x0000000000000000,
        z: 0x3F999999999999A0,
        xc: 0x3F8EB851EB851EB8,
        xn: 0x0000000000000000,
        xo: 0x3F847AE147AE147B,
        xne: 0x0000000000000000,
        frac_low_t: 0x0000000000000000,
        frac_high_t: 0x3FF0000000000000,
        frac_type2: 0x3FDC71C71C71C747,
        kap_rad: 0x3FC88F9A1BB8950B,
        rad_dlnd: 0x3FA44360E85516F4,
        rad_dlnt: 0xBFD0BAD20D36B733,
    },
    KapOracleRow {
        tag: "t2b_z02_xdz",
        cfg_kind: "inl_t2blend_z02.txt",
        lnd: 0x40026BB1BBB55516,
        lnt: 0x40317FE8D8B910D5,
        x: 0x3F40624DD2F1A9FC,
        z: 0x3F999999999999A0,
        xc: 0x3F8EB851EB851EB8,
        xn: 0x0000000000000000,
        xo: 0x3F847AE147AE147B,
        xne: 0x0000000000000000,
        frac_low_t: 0x0000000000000000,
        frac_high_t: 0x3FF0000000000000,
        frac_type2: 0x3FCC79111CBAAB49,
        kap_rad: 0x3FC891E192AFE94C,
        rad_dlnd: 0x3FA44A29B9173B06,
        rad_dlnt: 0xBFD092EB4BA9D14C,
    },
    KapOracleRow {
        tag: "t2b_z07_x",
        cfg_kind: "inl_t2blend_z07.txt",
        lnd: 0x40126BB1BBB55516,
        lnt: 0x4031F5CD4A3732F5,
        x: 0x3F33A92A30553261,
        z: 0x3FD3333333333334,
        xc: 0x3FC3333333333333,
        xn: 0x0000000000000000,
        xo: 0x3FC3333333333333,
        xne: 0x0000000000000000,
        frac_low_t: 0x0000000000000000,
        frac_high_t: 0x3FF0000000000000,
        frac_type2: 0x3FE66C23E03966C2,
        kap_rad: 0x3FC8B03E1B9A7C0F,
        rad_dlnd: 0x3FB2C84E26707F8D,
        rad_dlnt: 0xBFE006FB94C7CCE9,
    },
    KapOracleRow {
        tag: "t2b_z07_dz",
        cfg_kind: "inl_t2blend_z07.txt",
        lnd: 0x40126BB1BBB55516,
        lnt: 0x4031F5CD4A3732F5,
        x: 0x0000000000000000,
        z: 0x3FB374BC6A7EF9D8,
        xc: 0x3FA26E978D4FDF3B,
        xn: 0x0000000000000000,
        xo: 0x3FA47AE147AE147B,
        xne: 0x0000000000000000,
        frac_low_t: 0x0000000000000000,
        frac_high_t: 0x3FF0000000000000,
        frac_type2: 0x3FE1C71C71C71C3F,
        kap_rad: 0x3FC835B5143607B4,
        rad_dlnd: 0x3FAFF1FA3865A540,
        rad_dlnt: 0xBFDD9D33D7011B6B,
    },
    KapOracleRow {
        tag: "zbt1_c0_a",
        cfg_kind: "inl_gs98_zbase_t1.txt",
        lnd: 0xC0126BB1BBB55516,
        lnt: 0x40295414621954FE,
        x: 0x3FE0000000000000,
        z: 0x3FB70A3D70A3D708,
        xc: 0x3FA47AE147AE147B,
        xn: 0x0000000000000000,
        xo: 0x3F9EB851EB851EB8,
        xne: 0x0000000000000000,
        frac_low_t: 0x0000000000000000,
        frac_high_t: 0x3FF0000000000000,
        frac_type2: 0x0000000000000000,
        kap_rad: 0x408A89719CE5556E,
        rad_dlnd: 0x3FDE0AD814DDA688,
        rad_dlnt: 0xC00D8FD20E78B818,
    },
    KapOracleRow {
        tag: "zbt1_c2_a",
        cfg_kind: "inl_gs98_zbase_t1.txt",
        lnd: 0xC0126BB1BBB55516,
        lnt: 0x40295414621954FE,
        x: 0x0000000000000000,
        z: 0x3FF0000000000000,
        xc: 0x3FE6666666666666,
        xn: 0x0000000000000000,
        xo: 0x3FD999999999999A,
        xne: 0x0000000000000000,
        frac_low_t: 0x0000000000000000,
        frac_high_t: 0x3FF0000000000000,
        frac_type2: 0x3FF0000000000000,
        kap_rad: 0x40A6FFAD0FDC7A3E,
        rad_dlnd: 0x3FE05EF2E0B2BB07,
        rad_dlnt: 0xC010F905B48977CB,
    },
    KapOracleRow {
        tag: "aes_t0",
        cfg_kind: "inl_a09_aes.txt",
        lnd: 0xC032E1962D337736,
        lnt: 0x401E64E54284CC64,
        x: 0x3FE5C28F5C28F5C3,
        z: 0x3F948E4F6FD76F00,
        xc: 0x3F747AE147AE147B,
        xn: 0x3F60624DD2F1A9FC,
        xo: 0x3F826E978D4FDF3B,
        xne: 0x0000000000000000,
        frac_low_t: 0x3FF0000000000000,
        frac_high_t: 0x0000000000000000,
        frac_type2: 0x0000000000000000,
        kap_rad: 0x3F69B7FC2DAC94EA,
        rad_dlnd: 0x3FA09D0A67E14055,
        rad_dlnt: 0x400A7B457657DC50,
    },
    KapOracleRow {
        tag: "aes_int_t1",
        cfg_kind: "inl_aes_z009.txt",
        lnd: 0xC033926CD770AA67,
        lnt: 0x40210A04673AEEB5,
        x: 0x3FD0000000000000,
        z: 0x3FB9999999999998,
        xc: 0x3FA999999999999A,
        xn: 0x3F847AE147AE147B,
        xo: 0x3FA47AE147AE147B,
        xne: 0x0000000000000000,
        frac_low_t: 0x3FF0000000000000,
        frac_high_t: 0x0000000000000000,
        frac_type2: 0x0000000000000000,
        kap_rad: 0x3F7727BA6A929FA0,
        rad_dlnd: 0x3FE50828C1C00B72,
        rad_dlnt: 0x400F2A2EFA160F48,
    },
    KapOracleRow {
        tag: "aes_int_t3",
        cfg_kind: "inl_aes_z009.txt",
        lnd: 0xC033926CD770AA67,
        lnt: 0x4022F92A43E64AD7,
        x: 0x3FD0000000000000,
        z: 0x3FB9999999999998,
        xc: 0x3FA999999999999A,
        xn: 0x3F847AE147AE147B,
        xo: 0x3FA47AE147AE147B,
        xne: 0x0000000000000000,
        frac_low_t: 0x3FC4DFCE3150D970,
        frac_high_t: 0x3FEAC80C73ABC9A4,
        frac_type2: 0x0000000000000000,
        kap_rad: 0x401C60647D5B7C9A,
        rad_dlnd: 0x3FED0706B8315318,
        rad_dlnt: 0xC0200A7B83818332,
    },
    KapOracleRow {
        tag: "fa05_gs98_t1",
        cfg_kind: "inl_fa05_gs98.txt",
        lnd: 0xC03A6F05727B4080,
        lnt: 0x4021AF11061EB815,
        x: 0x3FE4CCCCCCCCCCCD,
        z: 0x3F947AE147AE1480,
        xc: 0x3F747AE147AE147B,
        xn: 0x0000000000000000,
        xo: 0x3F8EB851EB851EB8,
        xne: 0x0000000000000000,
        frac_low_t: 0x3FDFFFFFFFFFFE88,
        frac_high_t: 0x3FE00000000000BC,
        frac_type2: 0x0000000000000000,
        kap_rad: 0x3FC4EAB53828FFF6,
        rad_dlnd: 0xBFC5D7873B2A49F1,
        rad_dlnt: 0x402787E01F7850B4,
    },
    KapOracleRow {
        tag: "fa05_gn93_t1",
        cfg_kind: "inl_fa05_gn93.txt",
        lnd: 0xC03A6F05727B4080,
        lnt: 0x4021AF11061EB815,
        x: 0x3FE4CCCCCCCCCCCD,
        z: 0x3F947AE147AE1480,
        xc: 0x3F747AE147AE147B,
        xn: 0x0000000000000000,
        xo: 0x3F8EB851EB851EB8,
        xne: 0x0000000000000000,
        frac_low_t: 0x3FDFFFFFFFFFFE88,
        frac_high_t: 0x3FE00000000000BC,
        frac_type2: 0x0000000000000000,
        kap_rad: 0x3FC4E1104804D919,
        rad_dlnd: 0xBFC5CF81F36B507E,
        rad_dlnt: 0x40279936AE0FF47C,
    },
    KapOracleRow {
        tag: "fa05_a09_t1",
        cfg_kind: "inl_fa05_a09.txt",
        lnd: 0xC03A6F05727B4080,
        lnt: 0x4021AF11061EB815,
        x: 0x3FE4CCCCCCCCCCCD,
        z: 0x3F947AE147AE1480,
        xc: 0x3F747AE147AE147B,
        xn: 0x0000000000000000,
        xo: 0x3F8EB851EB851EB8,
        xne: 0x0000000000000000,
        frac_low_t: 0x3FDFFFFFFFFFFE88,
        frac_high_t: 0x3FE00000000000BC,
        frac_type2: 0x0000000000000000,
        kap_rad: 0x3FC4F037E571B6CF,
        rad_dlnd: 0xBFC5A04D3BBB92AC,
        rad_dlnt: 0x40275C453253EDC8,
    },
    KapOracleRow {
        tag: "fr_hz_t0",
        cfg_kind: "inl_freedman_highz.txt",
        lnd: 0xC0226BB1BBB55516,
        lnt: 0x4017069E2AA2AA5C,
        x: 0x3FDCCCCCCCCCCCCD,
        z: 0x3FC3333333333330,
        xc: 0x0000000000000000,
        xn: 0x0000000000000000,
        xo: 0x3FC3333333333333,
        xne: 0x0000000000000000,
        frac_low_t: 0x3FF0000000000000,
        frac_high_t: 0x0000000000000000,
        frac_type2: 0x0000000000000000,
        kap_rad: 0x3FBA3E5A86A5F13B,
        rad_dlnd: 0x3FE40610D33FC0AF,
        rad_dlnt: 0x3FE7C90A33246BD3,
    },
    KapOracleRow {
        tag: "fr_z0_t0",
        cfg_kind: "inl_freedman_highz.txt",
        lnd: 0xC0226BB1BBB55516,
        lnt: 0x4017069E2AA2AA5C,
        x: 0x3FE8000000000000,
        z: 0x0000000000000000,
        xc: 0x0000000000000000,
        xn: 0x0000000000000000,
        xo: 0x0000000000000000,
        xne: 0x0000000000000000,
        frac_low_t: 0x3FF0000000000000,
        frac_high_t: 0x0000000000000000,
        frac_type2: 0x0000000000000000,
        kap_rad: 0x3F8A2E84DDF1CF39,
        rad_dlnd: 0x3FE8413020823C4A,
        rad_dlnt: 0xBFD02177B1AE5E08,
    },
];

fn cfg_for(kind: &str) -> KapConfig {
    match kind {
        "inl_gn93_cn.txt" => KapConfig {
            zbase: 0.02,
            kap_option: KapOption::Gn93,
            kap_co_option: KapCoOption::Gn93Co,
            kap_low_t_option: KapLowTOption::KapCn,
            use_type2_opacities: true,
            kap_blend_log_t_lower_bdy: 4.00,
            kap_blend_log_t_upper_bdy: 4.05,
            ..KapConfig::default()
        },
        "inl_cn_z005.txt" => KapConfig {
            zbase: 0.005,
            kap_option: KapOption::Gn93,
            kap_co_option: KapCoOption::Gn93Co,
            kap_low_t_option: KapLowTOption::KapCn,
            use_type2_opacities: true,
            ..KapConfig::default()
        },
        "inl_cn_z000.txt" => KapConfig {
            zbase: 0.0,
            kap_option: KapOption::Gn93,
            kap_co_option: KapCoOption::Gn93Co,
            kap_low_t_option: KapLowTOption::KapCn,
            use_type2_opacities: true,
            ..KapConfig::default()
        },
        "inl_op_a09_z01.txt" => KapConfig {
            zbase: 0.01,
            kap_option: KapOption::OpA09,
            kap_co_option: KapCoOption::A09Co,
            kap_low_t_option: KapLowTOption::LowTFa05A09p,
            use_type2_opacities: false,
            ..KapConfig::default()
        },
        "inl_op_a09.txt" => KapConfig {
            zbase: 0.014,
            kap_option: KapOption::OpA09,
            kap_co_option: KapCoOption::A09Co,
            kap_low_t_option: KapLowTOption::LowTFa05A09p,
            use_type2_opacities: true,
            ..KapConfig::default()
        },
        "inl_op_gs98.txt" => KapConfig {
            zbase: 0.02,
            kap_option: KapOption::OpGs98,
            kap_co_option: KapCoOption::Gs98Co,
            kap_low_t_option: KapLowTOption::LowTFa05Gs98,
            use_type2_opacities: true,
            use_zbase_for_type1: true,
            cubic_interpolation_in_x: true,
            cubic_interpolation_in_z: true,
            ..KapConfig::default()
        },
        "inl_t2blend_z02.txt" => KapConfig {
            zbase: 0.02,
            kap_option: KapOption::Gs98,
            kap_co_option: KapCoOption::Gs98Co,
            use_type2_opacities: true,
            ..KapConfig::default()
        },
        "inl_t2blend_z07.txt" => KapConfig {
            zbase: 0.07,
            kap_option: KapOption::Gs98,
            kap_co_option: KapCoOption::Gs98Co,
            use_type2_opacities: true,
            ..KapConfig::default()
        },
        "inl_gs98_zbase_t1.txt" => KapConfig {
            zbase: 0.02,
            kap_option: KapOption::Gs98,
            kap_co_option: KapCoOption::Gs98Co,
            use_type2_opacities: true,
            use_zbase_for_type1: true,
            cubic_interpolation_in_x: true,
            cubic_interpolation_in_z: true,
            ..KapConfig::default()
        },
        "inl_a09_aes.txt" => KapConfig {
            zbase: 0.014,
            kap_option: KapOption::A09,
            kap_co_option: KapCoOption::A09Co,
            kap_low_t_option: KapLowTOption::Aesopus,
            aesopus_filename: String::from("AESOPUS_AGSS09.h5"),
            use_type2_opacities: true,
            ..KapConfig::default()
        },
        "inl_aes_z009.txt" => KapConfig {
            zbase: 0.009,
            kap_option: KapOption::A09,
            kap_co_option: KapCoOption::A09Co,
            kap_low_t_option: KapLowTOption::Aesopus,
            aesopus_filename: String::from("AESOPUS_AGSS09.h5"),
            kap_blend_log_t_lower_bdy: 4.05,
            kap_blend_log_t_upper_bdy: 4.15,
            use_type2_opacities: true,
            ..KapConfig::default()
        },
        "inl_fa05_gs98.txt" => KapConfig {
            zbase: 0.02,
            kap_option: KapOption::Gs98,
            kap_co_option: KapCoOption::Gs98Co,
            kap_low_t_option: KapLowTOption::LowTFa05Gs98,
            use_type2_opacities: true,
            ..KapConfig::default()
        },
        "inl_fa05_gn93.txt" => KapConfig {
            zbase: 0.02,
            kap_option: KapOption::Gn93,
            kap_co_option: KapCoOption::Gn93Co,
            kap_low_t_option: KapLowTOption::LowTFa05Gn93,
            use_type2_opacities: true,
            ..KapConfig::default()
        },
        "inl_fa05_a09.txt" => KapConfig {
            zbase: 0.014,
            kap_option: KapOption::A09,
            kap_co_option: KapCoOption::A09Co,
            kap_low_t_option: KapLowTOption::LowTFa05A09p,
            use_type2_opacities: true,
            ..KapConfig::default()
        },
        "inl_freedman_highz.txt" => KapConfig {
            zbase: 0.15,
            kap_option: KapOption::Gs98,
            kap_low_t_option: KapLowTOption::LowTFreedman11,
            use_type2_opacities: true,
            cubic_interpolation_in_x: true,
            cubic_interpolation_in_z: true,
            ..KapConfig::default()
        },
        other => panic!("unexpected cfg_kind `{other}`"),
    }
}

fn assert_row_zero_ulp(row: &KapOracleRow) {
    let cfg = cfg_for(row.cfg_kind);
    let log_rho = f64::from_bits(row.lnd) / LN10;
    let log_t = f64::from_bits(row.lnt) / LN10;
    let x = f64::from_bits(row.x);
    let z = f64::from_bits(row.z);
    let xc = f64::from_bits(row.xc);
    let xn = f64::from_bits(row.xn);
    let xo = f64::from_bits(row.xo);
    let xne = f64::from_bits(row.xne);
    let (fl, fh, ft2, kap, dr, dt) =
        get_kap_results_blend_t(&cfg, x, z, xc, xn, xo, xne, log_rho, log_t)
            .unwrap_or_else(|e| {
                panic!("{}: get_kap_results_blend_t failed: {e}", row.tag)
            });

    assert_eq!(ft2.to_bits(), row.frac_type2, "{}: frac_type2", row.tag);
    assert_eq!(kap.to_bits(), row.kap_rad, "{}: kap_rad", row.tag);
    assert_eq!(dr.to_bits(), row.rad_dlnd, "{}: rad_dlnd", row.tag);
    assert_eq!(dt.to_bits(), row.rad_dlnt, "{}: rad_dlnt", row.tag);

    let out = get_kap_results(
        &cfg, 1.0, x, z, xc, xn, xo, xne, log_rho, log_t, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0,
    )
    .unwrap_or_else(|e| panic!("{}: get_kap_results failed: {e}", row.tag));
    assert_eq!(
        out.kap_fracs[I_FRAC_LOW_T].to_bits(),
        row.frac_low_t,
        "{}: frac_low_t",
        row.tag
    );
    assert_eq!(
        out.kap_fracs[I_FRAC_HIGH_T].to_bits(),
        row.frac_high_t,
        "{}: frac_high_t",
        row.tag
    );
    if out.kap_fracs[I_FRAC_COMPTON] == 0.0 {
        assert_eq!(fl.to_bits(), row.frac_low_t, "{}: blend fl", row.tag);
        assert_eq!(fh.to_bits(), row.frac_high_t, "{}: blend fh", row.tag);
    }
}

#[test]
fn test_kapcn_lagrange_and_zbase_zero_mesa_hex_0ulp() {
    with_numerics_mode(NumericsMode::Strict, || {
        for row in ROWS.iter().filter(|r| {
            r.tag.starts_with("kapcn_") || r.tag.starts_with("cn00")
        }) {
            assert_row_zero_ulp(row);
        }
    });
}

#[test]
fn test_op_a09_upper_log_t_spline_boundary_and_op_gs98_0ulp() {
    with_numerics_mode(NumericsMode::Strict, || {
        for row in ROWS.iter().filter(|r| {
            r.tag.starts_with("opa09_") || r.tag.starts_with("op_gs98_")
        }) {
            assert_row_zero_ulp(row);
        }
    });
}

#[test]
fn test_type2_blend_aesopus_and_lowt_options_mesa_hex_0ulp() {
    with_numerics_mode(NumericsMode::Strict, || {
        for row in ROWS.iter().filter(|r| {
            r.tag.starts_with("t2b_")
                || r.tag.starts_with("zbt1_")
                || r.tag.starts_with("aes_")
                || r.tag.starts_with("fa05_")
                || r.tag.starts_with("fr_")
        }) {
            assert_row_zero_ulp(row);
        }
    });
}
