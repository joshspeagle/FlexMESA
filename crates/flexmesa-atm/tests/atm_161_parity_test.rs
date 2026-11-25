// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone zero-`$MESA_DIR` MESA `26.4.1` parity tests for `flexmesa-atm` (#161).

use flexmesa_atm::{
    atm_teff, build_t_tau_uniform, build_t_tau_varying, eval_e2,
    eval_irradiated, eval_t_tau, eval_t_tau_atm, eval_t_tau_dq_dtau,
    eval_t_tau_uniform, eval_t_tau_varying, eval_table, eval_table_atm,
    get_t_tau_base, get_t_tau_id, get_table_alfa_beta, get_table_base,
    get_table_id, get_table_values, AtmEosCallbackResult, AtmError,
    AtmKapCallbackResult, ATM_TABLE_DB_WD_TAU_25, ATM_TABLE_PHOTOSPHERE,
    ATM_TABLE_TAU_100, ATM_TABLE_WD_TAU_25, ATM_T_TAU_EDDINGTON,
    ATM_T_TAU_KRISHNA_SWAMY, ATM_T_TAU_SOLAR_HOPF, ATM_T_TAU_TRAMPEDACH_SOLAR,
    NUM_RESULTS_FOR_BUILD_ATM,
};
use flexmesa_eos::{
    I_CHI_RHO, I_CHI_T, I_CP, I_CV, I_ETA, I_GAMMA1, I_GAMMA3, I_GRAD_AD,
    I_LN_E, I_LN_FREE_E, I_LN_PGAS, I_LN_S, I_MU, NUM_EOS_BASIC_RESULTS,
};
use flexmesa_math::{exp, log, log10, pow};

const L_SUN_TEST: f64 = 3.828e33;
const R_SUN_TEST: f64 = 6.957e10;
const M_SUN_TEST: f64 = 1.98841e33;
const CGRAV_TEST: f64 = 6.67430e-8;

struct GtvOracleCase {
    name: &'static str,
    tid: i32,
    z: f64,
    logg: f64,
    teff: f64,
    expected: [u64; 6],
    dpgas_teff_max_ulp: u64,
}

struct TblOracleCase {
    name: &'static str,
    tid: i32,
    hex: [u64; 12],
}

const A1_K1: [u64; 19] = [
    0x0000_0000_0000_0000,
    0x41AA_607C_37D0_3FC6,
    0x4001_C875_E33B_89A4,
    0xC039_2D0B_B01B_8650,
    0x4020_F99C_E7B5_AD5F,
    0x3F30_828D_725E_115E,
    0x3FD6_6666_6666_6666,
    0x3FF8_86E8_758F_4D4C,
    0x3FD3_A519_A443_7B66,
    0x3FF7_46CE_218E_970E,
    0x3FEB_2621_3EF6_45F7,
    0x41BE_1991_D0EA_D97E,
    0x41CB_3163_C77F_973C,
    0xBFD6_6666_6666_6666,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x4000_77C6_225C_FAE0,
    0x3F1A_36E2_EB1C_4333,
    0x3F30_828D_725E_115E,
];
const A1_K20: [u64; 19] = [
    0x0000_0000_0000_0000,
    0x419B_80B8_BF00_2E6F,
    0x4019_80BF_5E3F_A11E,
    0xC034_DD27_0E37_2E31,
    0x4020_FB05_9B20_BBFA,
    0x3F68_35AA_5F3B_086D,
    0x3FD6_6666_6666_6666,
    0x3FFA_9A77_CD3F_F07E,
    0x3FD9_6AED_43AF_9DBB,
    0x3FF0_1D9B_4AB8_84B8,
    0x3FEF_EC43_2384_FCDA,
    0x41A9_41FF_542B_F8E8,
    0x41B5_0C87_4B8E_571C,
    0xBFD6_6666_6666_6666,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x4019_7E46_FF95_FB20,
    0x3F7E_A37A_3C73_A7A3,
    0x3F68_35AA_5F3B_086D,
];
const A1_K40: [u64; 19] = [
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x4025_BA11_2B43_CEAC,
    0xC030_8B59_72D8_9CA1,
    0x4021_5251_097F_3253,
    0x3FC0_001C_1974_DE9A,
    0x3FD6_6666_6666_6666,
    0x3FFA_AA4D_0957_9192,
    0x3FD9_988B_F134_D2FC,
    0x3FF0_00A8_9795_2104,
    0x3FEF_FF8F_9AF1_E9FE,
    0x41A8_CA26_C68E_5591,
    0x41B4_A875_ABD5_1D72,
    0xBFD6_6666_6666_6666,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x4025_BA0A_24E6_9709,
    0x3FE5_5555_5555_5555,
    0x3FC0_001C_1974_DE9A,
];

const A2_K1: [u64; 19] = [
    0x0000_0000_0000_0000,
    0x41A3_F970_00A3_928C,
    0x3FF8_F16C_E72C_5F96,
    0xC039_C160_3658_CEC5,
    0x4020_BBD3_CC1F_7E82,
    0x3F35_B61E_DA6B_92F0,
    0x3FE8_0599_6C0C_796A,
    0x3FF8_4D22_9940_D094,
    0x3FD3_1AC5_FFA3_1F33,
    0x3FF8_B8F2_E64E_2D9A,
    0x3FEA_2F5E_1121_36EF,
    0x41C1_3342_669A_225F,
    0x41CF_ECFC_FDEA_CDD0,
    0xBFD6_6666_6666_6666,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x3FF5_BBFB_0433_F789,
    0x3F1A_36E2_EB1C_4333,
    0x3F35_B61E_DA6B_92F0,
];
const A2_K28: [u64; 19] = [
    0x0000_0000_0000_0000,
    0x4195_8F51_7848_3BA1,
    0x4010_D5B0_64EB_C786,
    0xC036_F945_EC02_3151,
    0x4020_D6C9_8B2D_132E,
    0x3F55_0817_BC39_6E93,
    0x3FE8_0599_6C0C_796A,
    0x3FFA_4854_1E17_B1BA,
    0x3FD8_7DDB_5FAB_5B17,
    0x3FF0_C2CC_546D_34E2,
    0x3FEF_7E22_7261_DCBF,
    0x41AB_F8E0_0B2D_4563,
    0x41B7_582C_9CE4_8A5A,
    0xBFD6_6666_6666_6666,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x4010_C553_6882_9D09,
    0x3F61_C8A1_90D0_C47F,
    0x3F55_0817_BC39_6E93,
];
const A2_K56: [u64; 19] = [
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x401D_661C_D767_19D6,
    0xC033_E447_A64A_A364,
    0x4020_FCB8_D38A_5E65,
    0x3F93_55F2_6CA8_87A9,
    0x3FE8_0599_6C0C_796A,
    0x3FFA_A468_FF72_5DAA,
    0x3FD9_8793_F72F_7D10,
    0x3FF0_0B54_0121_E02A,
    0x3FEF_F872_A9E9_6A8F,
    0x41A8_F642_149D_843D,
    0x41B4_CD3E_70FE_1A2C,
    0xBFD6_6666_6666_6666,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x401D_652B_101B_6B2A,
    0x3FAA_5B7E_5AED_FEE8,
    0x3F93_55F2_6CA8_87A9,
];

const B1_K1: [u64; 19] = [
    0x0000_0000_0000_0000,
    0x41A0_DBAF_E090_4467,
    0x400F_1037_8A10_3CFA,
    0xC037_617A_38FA_AD5E,
    0x4020_F99C_E7B5_AD5F,
    0x3F3B_769B_DCE3_5A59,
    0x3FD6_E520_6067_6B70,
    0x3FFA_041A_9A83_C6BA,
    0x3FD7_B90C_299E_72E6,
    0x3FF1_622D_BC99_08E8,
    0x3FEF_13E1_8244_A4BA,
    0x41AE_A98B_0151_14E1,
    0x41B9_AB15_D68E_98B7,
    0xBFD6_6666_6666_6666,
    0xBFB9_0D8C_B178_0163,
    0x3FA3_F5BB_9D7B_756D,
    0x400E_D451_DB63_C271,
    0x3F1A_36E2_EB1C_4333,
    0x3F3B_769B_DCE3_5A59,
];
const B1_K20: [u64; 19] = [
    0x0000_0000_0000_0000,
    0x4193_734C_41A8_64C7,
    0x4019_2C7C_E4E3_2F15,
    0xC034_F24B_8429_D801,
    0x4020_FB12_01D2_9AC1,
    0x3F6E_D171_CBBB_643D,
    0x3FDF_63B2_B6B5_0FCA,
    0x3FFA_9918_4F8B_F168,
    0x3FD9_66F8_0D91_BD5A,
    0x3FF0_2028_71EC_F338,
    0x3FEF_EA8F_B40C_B330,
    0x41A9_4C96_5AF4_3A76,
    0x41B5_1563_9760_2180,
    0xBFD6_6666_6666_6666,
    0xBFEE_AD8A_3686_2836,
    0x3FD2_9A57_1C61_E7C1,
    0x4019_29CD_F52F_116D,
    0x3F7F_AF0D_ED60_28DB,
    0x3F6E_D171_CBBB_643D,
];
const B1_K40: [u64; 19] = [
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x4022_F027_49FA_1D5E,
    0xC031_F059_0A77_997C,
    0x4021_5251_097F_3253,
    0x3FC5_D9C1_BD01_0296,
    0x3FFE_D67C_B3F8_BDD1,
    0x3FFA_A931_8110_CDC0,
    0x3FD9_955B_5348_BE16,
    0x3FF0_02A7_D4D5_082C,
    0x3FEF_FE3A_C771_FA8E,
    0x41A8_D266_AE3D_09CA,
    0x41B4_AF55_F8CB_E9E4,
    0xBFD6_6666_6666_6666,
    0xC006_50B3_4941_7A90,
    0x3FE9_F004_F2E5_D762,
    0x4022_F00A_F5A8_9D53,
    0x3FE5_5555_5555_5555,
    0x3FC5_D9C1_BD01_0296,
];

const B2_K1: [u64; 19] = [
    0x0000_0000_0000_0000,
    0x419A_53BC_B5B6_F230,
    0x400D_2EEF_50BE_A3F2,
    0xC037_7D15_5594_BD59,
    0x4020_BBD3_CC1F_7E82,
    0x3F42_921B_4124_B931,
    0x3FD7_1EB5_2E87_F68E,
    0x3FFA_242D_FE21_FD9E,
    0x3FD8_1582_BAA6_EF72,
    0x3FF1_1478_58A8_DF56,
    0x3FEF_47AF_C4E4_C071,
    0x41AD_577D_1C79_331D,
    0x41B8_857F_572E_B87E,
    0xBFD6_6666_6666_6666,
    0xBFC2_9AFF_5019_45F3,
    0x3FA9_C74E_E57C_FDA3,
    0x400D_0054_883A_8721,
    0x3F1A_36E2_EB1C_4333,
    0x3F42_921B_4124_B931,
];
const B2_K28: [u64; 19] = [
    0x0000_0000_0000_0000,
    0x4190_E9AC_B220_71AA,
    0x4014_5344_9AE3_0F8A,
    0xC036_177D_2E08_2969,
    0x4020_D6C9_8B2D_132E,
    0x3F5A_214A_92EB_CD2C,
    0x3FD9_DA14_8C14_3B2D,
    0x3FFA_7F47_570E_9CDE,
    0x3FD9_1C81_D043_3985,
    0x3FF0_5165_6BAA_1276,
    0x3FEF_C9BC_62E3_F3B1,
    0x41AA_19D3_7D95_01DC,
    0x41B5_C1B1_5956_4E6B,
    0xBFD6_6666_6666_6666,
    0xBFDD_C14E_9872_AC88,
    0x3FC2_A9FF_4ACC_2333,
    0x4014_4C76_606A_BCA8,
    0x3F61_C8A1_90D0_C47F,
    0x3F5A_214A_92EB_CD2C,
];
const B2_K56: [u64; 19] = [
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x401E_3646_84F1_6202,
    0xC033_B032_1B5E_D4F4,
    0x4020_FCB8_D38A_5E65,
    0x3F9A_7896_991B_B051,
    0x3FEA_E7F3_3C52_C5E4,
    0x3FFA_A58E_3859_E8E2,
    0x3FD9_8AE0_B817_597C,
    0x3FF0_093E_89D1_01F2,
    0x3FEF_F9D6_4EC9_FEB4,
    0x41A8_EDA3_4241_1E74,
    0x41B4_C60C_F546_04AE,
    0xBFD6_6666_6666_6666,
    0xBFFF_C829_EA2A_44A9,
    0x3FE2_9E9E_DE02_674F,
    0x401E_3581_3BCA_A4EB,
    0x3FAA_5B7E_5AED_FEE8,
    0x3F9A_7896_991B_B051,
];

const GTV_CASES: [GtvOracleCase; 5] = [
    GtvOracleCase {
        name: "PHOTO_MID",
        tid: ATM_TABLE_PHOTOSPHERE,
        z: 0.004,
        logg: 4.0,
        teff: 5500.0,
        expected: [
            0x40F7_6635_3767_7292,
            0xC043_DC5B_3B9D_E5CC,
            0x40FE_59B3_A8DD_F3E0,
            0x40B5_7C00_0000_0000,
            0x3FF0_0000_0000_0000,
            0x0000_0000_0000_0000,
        ],
        dpgas_teff_max_ulp: 0,
    },
    GtvOracleCase {
        name: "PHOTO_LOW_CLAMP",
        tid: ATM_TABLE_PHOTOSPHERE,
        z: 0.0,
        logg: -2.0,
        teff: 1000.0,
        expected: [
            0x40C8_8B80_0000_0000,
            0x0000_0000_0000_0000,
            0x0000_0000_0000_0000,
            0x409F_4000_0000_0000,
            0x0000_0000_0000_0000,
            0x0000_0000_0000_0000,
        ],
        dpgas_teff_max_ulp: 0,
    },
    GtvOracleCase {
        name: "PHOTO_TEFF_CLAMP",
        tid: ATM_TABLE_PHOTOSPHERE,
        z: 0.02,
        logg: 4.0,
        teff: 1000.0,
        expected: [
            0x410B_0E9A_2CCC_7AFC,
            0x0000_0000_0000_0000,
            0x411E_BC68_9E7D_9865,
            0x409F_4000_0000_0000,
            0x0000_0000_0000_0000,
            0x0000_0000_0000_0000,
        ],
        dpgas_teff_max_ulp: 0,
    },
    GtvOracleCase {
        name: "PHOTO_LOGG_CLAMP",
        tid: ATM_TABLE_PHOTOSPHERE,
        z: 0.02,
        logg: 7.0,
        teff: 5500.0,
        expected: [
            0x4114_1D72_DBA4_DDD2,
            0xC051_61C5_B3A4_E383,
            0x0000_0000_0000_0000,
            0x40B5_7C00_0000_0000,
            0x3FF0_0000_0000_0000,
            0x0000_0000_0000_0000,
        ],
        dpgas_teff_max_ulp: 2,
    },
    GtvOracleCase {
        name: "DB_WD",
        tid: ATM_TABLE_DB_WD_TAU_25,
        z: 0.02,
        logg: 8.0,
        teff: 20000.0,
        expected: [
            0x4179_4983_7000_0000,
            0xC0B9_EDC9_2AFF_FEC1,
            0x4188_2BBF_D9EA_005F,
            0x40DA_8E2C_CCCC_CCCD,
            0x4004_6867_804E_7CC2,
            0xC092_28B2_C813_B1EB,
        ],
        dpgas_teff_max_ulp: 0,
    },
];

const TBL_CASES: [TblOracleCase; 7] = [
    TblOracleCase {
        name: "PURE_TABLE",
        tid: ATM_TABLE_TAU_100,
        hex: [
            0x4702_E205_DC02_9AED,
            0x423A_D282_1146_E51C,
            0x46D8_8250_CE73_9EC6,
            0x40B7_6FFF_FFFF_FFFB,
            0x0000_0000_0000_0000,
            0x3FF0_0000_0000_0000,
            0x4022_7680_1977_886B,
            0x38B0_C6FE_3073_4A0F,
            0xBFD3_C10C_0F2C_F2C9,
            0x4026_D43A_F58D_6FDD,
            0xB8DF_7B57_2750_D75F,
            0x3FE2_C5FD_2D24_3CA4,
        ],
    },
    TblOracleCase {
        name: "BLEND_X",
        tid: ATM_TABLE_TAU_100,
        hex: [
            0x47A9_F04E_3935_A025,
            0x422E_2AA4_7B01_7065,
            0x46D8_8250_CE73_9EC6,
            0x40E7_ECFF_FFFF_FFF4,
            0x3FBF_6335_AC2A_F8BC,
            0x3FEC_1399_4A7A_A0E8,
            0x4027_A53C_0934_58B7,
            0xB82A_E82D_50FE_60C1,
            0x3FF5_40FF_6551_5A57,
            0x402D_8C95_AD34_04FC,
            0x385A_B9E0_0D91_7C1B,
            0x3FCF_4E61_99A3_E109,
        ],
    },
    TblOracleCase {
        name: "BLEND_Y_LO",
        tid: ATM_TABLE_TAU_100,
        hex: [
            0x47C9_ECA5_C5EC_B4CF,
            0x429F_6D8D_DB9B_F22B,
            0x46D8_8250_CE73_9EC6,
            0x40B7_6FFF_FFFF_FFFB,
            0x3FE0_0000_0000_0000,
            0x3FE0_0000_0000_0000,
            0x4023_8494_EC0B_0078,
            0x37F2_9F82_A187_984D,
            0xBFDE_3619_7AE6_5B73,
            0x4017_DA33_D619_F05D,
            0xB7D0_195A_D95A_805D,
            0x3FC6_D7D8_5A60_E8F9,
        ],
    },
    TblOracleCase {
        name: "BLEND_Y_HI",
        tid: ATM_TABLE_TAU_100,
        hex: [
            0x46C0_FD64_B1E1_7F43,
            0x4219_7139_0198_193B,
            0x46D8_8250_CE73_9EC6,
            0x40B7_6FFF_FFFF_FFFB,
            0x0000_0000_0000_0000,
            0x3FF0_0000_0000_0000,
            0x4022_7626_573B_226D,
            0x38F2_415D_EF67_F3B3,
            0xBFD3_5FD6_3EFC_74F5,
            0x402A_5799_1964_9DEA,
            0xB91F_328B_6EE3_20AB,
            0x3FC8_F186_FE8A_1D1D,
        ],
    },
    TblOracleCase {
        name: "BLEND_CORNER",
        tid: ATM_TABLE_TAU_100,
        hex: [
            0x4782_7353_D489_6B98,
            0x4219_7139_0198_193B,
            0x46D8_8250_CE73_9EC6,
            0x40E7_ECFF_FFFF_FFF4,
            0x3FBF_6335_AC2A_F8BC,
            0x3FEC_1399_4A7A_A0E8,
            0x4027_A63F_5DB8_E393,
            0x3846_8D5F_FCA2_E38D,
            0xBFEA_0B99_A099_0F44,
            0x402E_10F3_C73D_0D8E,
            0x383C_4BE1_27AB_36DF,
            0xBF83_0E09_E9AB_C3FC,
        ],
    },
    TblOracleCase {
        name: "BLEND_WD_CORNER",
        tid: ATM_TABLE_WD_TAU_25,
        hex: [
            0x4595_B6DB_B063_60B7,
            0x41A5_C3EE_E1FC_9EF3,
            0x46D8_8250_CE73_9EC6,
            0x40A6_A7FF_FFFF_FFFB,
            0x3FF0_0000_0000_0000,
            0x0000_0000_0000_0000,
            0x4020_B480_FC85_CAA6,
            0x3A14_70E2_B06B_0692,
            0xBFCB_BDC6_449A_BC62,
            0x4038_E0C4_D5C7_0006,
            0xBA42_4E20_3F16_09C4,
            0x0000_0000_0000_0000,
        ],
    },
    TblOracleCase {
        name: "PURE_GREY",
        tid: ATM_TABLE_TAU_100,
        hex: [
            0x47E5_5A14_47BE_6EC4,
            0x423A_D282_1146_E51C,
            0x46D8_8250_CE73_9EC6,
            0x40F1_16FF_FFFF_FFFC,
            0x3FF0_0000_0000_0000,
            0x0000_0000_0000_0000,
            0x4027_8D14_5880_BCF1,
            0x0000_0000_0000_0000,
            0xBFB5_62BF_4F11_6C87,
            0x402D_2BAF_5772_01AA,
            0x0000_0000_0000_0000,
            0x3FD1_9060_728D_8354,
        ],
    },
];

fn hex_to_f64(hex: u64) -> f64 {
    f64::from_bits(hex)
}

fn ulp_distance(a: f64, b: f64) -> u64 {
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    let to_ordered = |x: f64| -> i64 {
        let bits = x.to_bits().cast_signed();
        if bits < 0 {
            i64::MIN - bits
        } else {
            bits
        }
    };
    to_ordered(a).abs_diff(to_ordered(b))
}

fn assert_hex_ulp(label: &str, actual: f64, expected_hex: u64, max_ulp: u64) {
    let expected = hex_to_f64(expected_hex);
    let ulp = ulp_distance(actual, expected);
    assert!(
        ulp <= max_ulp,
        "{label}: actual={actual:.16e} (0x{:016X}), expected={expected:.16e} (0x{expected_hex:016X}), ulp={ulp} > {max_ulp}",
        actual.to_bits(),
    );
}

fn make_analytic_eos(
    ln_p_floor: f64,
) -> impl FnMut(f64, f64) -> Result<AtmEosCallbackResult, AtmError> {
    move |ln_p: f64, ln_t: f64| {
        const R_SPEC: f64 = 8.314_462_618_153_24e7 / 0.6;
        const ARAD_THIRD: f64 = 7.565_733_250_280_005e-15 / 3.0;

        if ln_p < ln_p_floor {
            return Err(AtmError::CallbackError {
                context: "analytic_eos",
                detail: "lnP below mock floor".to_string(),
            });
        }

        let p = exp(ln_p);
        let t = exp(ln_t);
        let p_rad = ARAD_THIRD * (t * t * t * t);
        let p_gas = p - p_rad;
        if p_gas <= 1.0e-30 * p {
            return Err(AtmError::CallbackError {
                context: "analytic_eos",
                detail: "P_gas <= 1e-30 * P".to_string(),
            });
        }

        let rho = p_gas / (R_SPEC * t);
        let ln_rho = log(rho);

        let beta_gas = p_gas / p;
        let chi_rho = beta_gas;
        let chi_t = 4.0 - 3.0 * beta_gas;
        let gamma3m1 = (2.0 * (4.0 - 3.0 * beta_gas))
            / (3.0 * beta_gas + 24.0 * (1.0 - beta_gas));
        let gamma1 = beta_gas + gamma3m1 * chi_t;
        let grad_ad = gamma3m1 / gamma1;
        let cv = R_SPEC * (1.5 + 12.0 * (1.0 - beta_gas) / beta_gas);
        let cp = cv + (p * (chi_t * chi_t)) / (rho * t * chi_rho);

        let mut res = [0.0_f64; NUM_EOS_BASIC_RESULTS];
        let mut dres_dln_rho = [0.0_f64; NUM_EOS_BASIC_RESULTS];
        let mut dres_dln_t = [0.0_f64; NUM_EOS_BASIC_RESULTS];

        res[I_LN_PGAS] = log(p_gas);
        res[I_LN_E] = log(cv * t);
        res[I_LN_S] = 1.0;
        res[I_MU] = 0.6;
        res[I_LN_FREE_E] = -0.35;
        res[I_ETA] = -5.0;
        res[I_GRAD_AD] = grad_ad;
        res[I_CHI_RHO] = chi_rho;
        res[I_CHI_T] = chi_t;
        res[I_CP] = cp;
        res[I_CV] = cv;
        res[I_GAMMA1] = gamma1;
        res[I_GAMMA3] = 1.0 + gamma3m1;

        dres_dln_rho[I_LN_PGAS] = 1.0;
        dres_dln_t[I_LN_PGAS] = 1.0;

        Ok(AtmEosCallbackResult {
            ln_rho,
            res,
            dres_dln_rho,
            dres_dln_t,
        })
    }
}

fn analytic_kap_val(ln_rho: f64, ln_t: f64) -> AtmKapCallbackResult {
    let rho = exp(ln_rho);
    let t = exp(ln_t);
    let kap_es = 0.34;
    let kap_kr = 1.0e23 * 0.02 * 0.7 * rho * pow(t, -3.5);
    let kap_hminus = 0.05 * pow(rho / 1.0e-7, 0.3) * pow(t / 5000.0, 1.5);
    let k_tot = kap_es + kap_kr + kap_hminus;
    AtmKapCallbackResult {
        kap: k_tot,
        dlnkap_dln_rho: (kap_kr + 0.3 * kap_hminus) / k_tot,
        dlnkap_dln_t: (-3.5 * kap_kr + 1.5 * kap_hminus) / k_tot,
    }
}

fn analytic_kap_cb(
    ln_rho: f64,
    ln_t: f64,
    _eos: &AtmEosCallbackResult,
) -> Result<AtmKapCallbackResult, AtmError> {
    if !ln_rho.is_finite() || !ln_t.is_finite() {
        return Err(AtmError::CallbackError {
            context: "analytic_kap",
            detail: "non-finite lnRho or lnT".to_string(),
        });
    }
    Ok(analytic_kap_val(ln_rho, ln_t))
}

fn assert_structure_row(
    tag: &str,
    row: &[f64; NUM_RESULTS_FOR_BUILD_ATM],
    expected_hex: &[u64; NUM_RESULTS_FOR_BUILD_ATM],
    max_ulp: u64,
) {
    for col in 0..NUM_RESULTS_FOR_BUILD_ATM {
        assert_hex_ulp(
            &format!("{tag} col {col}"),
            row[col],
            expected_hex[col],
            max_ulp,
        );
    }
}

#[test]
fn test_build_t_tau_uniform_19_col_eddington_parity() {
    let mut eos = make_analytic_eos(-1.0e99);
    let teff_a1 = atm_teff(L_SUN_TEST, R_SUN_TEST);
    assert_hex_ulp("A1 Teff", teff_a1, 0x40B6_8C00_E0BA_BBAB, 0);
    let tau_surf_a1 = 2.0 / 3.0;
    let eval_a1 = eval_t_tau_uniform(
        tau_surf_a1,
        L_SUN_TEST,
        R_SUN_TEST,
        M_SUN_TEST,
        CGRAV_TEST,
        0.35,
        1.0,
        ATM_T_TAU_EDDINGTON,
        &mut eos,
        analytic_kap_cb,
        1.0e-4,
        0,
        false,
        teff_a1,
    )
    .expect("eval_t_tau_uniform A1");
    assert_hex_ulp("A1 lnT", eval_a1.ln_t, 0x4021_5251_097F_3253, 0);
    assert_hex_ulp("A1 lnP", eval_a1.ln_p, 0x4025_BA11_2B43_CEAC, 0);
    assert_hex_ulp("A1 dlnP_dl", eval_a1.dln_p_dl, 0x3803_0E78_22AF_5AAF, 0);
    assert_hex_ulp(
        "A1 dlnP_dlnR",
        eval_a1.dln_p_dln_r,
        0xC000_0000_0000_0000,
        0,
    );
    assert_hex_ulp(
        "A1 dlnP_dlnM",
        eval_a1.dln_p_dln_m,
        0x3FEF_FFC7_CD78_F4FF,
        0,
    );

    let st_a1 = build_t_tau_uniform(
        tau_surf_a1,
        L_SUN_TEST,
        R_SUN_TEST,
        teff_a1,
        M_SUN_TEST,
        CGRAV_TEST,
        0.35,
        1.0,
        1.0e-4,
        ATM_T_TAU_EDDINGTON,
        &mut eos,
        1.0e-4,
        0.1,
    )
    .expect("build_t_tau_uniform A1");
    assert_eq!(st_a1.points.len(), 40);
    assert_structure_row("UNIF_A1 pt1", &st_a1.points[0], &A1_K1, 0);
    assert_structure_row("UNIF_A1 pt20", &st_a1.points[19], &A1_K20, 0);
    assert_structure_row("UNIF_A1 pt40", &st_a1.points[39], &A1_K40, 0);

    let empty = build_t_tau_uniform(
        1.0e-4,
        L_SUN_TEST,
        R_SUN_TEST,
        teff_a1,
        M_SUN_TEST,
        CGRAV_TEST,
        0.35,
        1.0,
        1.0e-3,
        ATM_T_TAU_EDDINGTON,
        &mut eos,
        1.0e-4,
        0.1,
    )
    .expect("tau_outer > tau_surf returns empty");
    assert_eq!(empty.points.len(), 0);
}

#[test]
fn test_build_t_tau_uniform_19_col_trampedach_iterated_parity() {
    let mut eos = make_analytic_eos(-1.0e99);
    let lum = 1.2 * L_SUN_TEST;
    let rad = 1.1 * R_SUN_TEST;
    let teff = atm_teff(lum, rad);
    assert_hex_ulp("A2 Teff", teff, 0x40B6_800A_68EA_87EA, 0);
    let tau_surf = get_t_tau_base(ATM_T_TAU_TRAMPEDACH_SOLAR).unwrap() * 0.1;
    let eval_a2 = eval_t_tau_uniform(
        tau_surf,
        lum,
        rad,
        M_SUN_TEST,
        CGRAV_TEST,
        0.4,
        1.25,
        ATM_T_TAU_TRAMPEDACH_SOLAR,
        &mut eos,
        analytic_kap_cb,
        1.0e-7,
        100,
        false,
        teff,
    )
    .expect("eval_t_tau_uniform A2");
    assert_hex_ulp("A2 lnT", eval_a2.ln_t, 0x4020_FCB8_D38A_5E65, 0);
    assert_hex_ulp("A2 lnP", eval_a2.ln_p, 0x401D_661C_E3BF_3F38, 0);
    assert_hex_ulp("A2 dlnT_dl", eval_a2.dln_t_dl, 0x38D2_15D7_DEF0_00EA, 0);
    assert_hex_ulp("A2 dlnP_dl", eval_a2.dln_p_dl, 0x38DB_B44D_98BA_25DE, 0);
    assert_hex_ulp(
        "A2 dlnP_dlnM",
        eval_a2.dln_p_dln_m,
        0x3FE4_DBA8_D2C5_2C61,
        0,
    );

    let kap_a2 = hex_to_f64(0x3FE8_0599_6C0C_796A);
    let st_a2 = build_t_tau_uniform(
        tau_surf,
        lum,
        rad,
        teff,
        M_SUN_TEST,
        CGRAV_TEST,
        kap_a2,
        1.25,
        1.0e-4,
        ATM_T_TAU_TRAMPEDACH_SOLAR,
        &mut eos,
        1.0e-5,
        0.05,
    )
    .expect("build_t_tau_uniform A2");
    assert_eq!(st_a2.points.len(), 56);
    assert_structure_row("UNIF_A2 pt1", &st_a2.points[0], &A2_K1, 0);
    assert_structure_row("UNIF_A2 pt28", &st_a2.points[27], &A2_K28, 0);
    assert_structure_row("UNIF_A2 pt56", &st_a2.points[55], &A2_K56, 0);
}

#[test]
fn test_build_t_tau_varying_19_col_eddington_parity() {
    let mut eos = make_analytic_eos(-1.0e99);
    let teff = atm_teff(L_SUN_TEST, R_SUN_TEST);
    let tau_surf = get_t_tau_base(ATM_T_TAU_EDDINGTON).unwrap();
    let eval_b1 = eval_t_tau_varying(
        tau_surf,
        L_SUN_TEST,
        R_SUN_TEST,
        M_SUN_TEST,
        CGRAV_TEST,
        ATM_T_TAU_EDDINGTON,
        &mut eos,
        analytic_kap_cb,
        1.0e-5,
        500,
        false,
        teff,
    )
    .expect("eval_t_tau_varying B1");
    assert_hex_ulp("B1 lnT", eval_b1.ln_t, 0x4021_5251_097F_3252, 0);
    assert_hex_ulp("B1 lnP", eval_b1.ln_p, 0x4022_F027_49FA_1D5E, 0);
    assert_hex_ulp("B1 dlnT_dl", eval_b1.dln_t_dl, 0x38D5_B3CF_D853_344B, 0);
    assert_hex_ulp(
        "B1 dlnT_dlnR",
        eval_b1.dln_t_dln_r,
        0xBFE0_0000_0000_0000,
        0,
    );
    assert_hex_ulp(
        "B1 dlnT_dlnM",
        eval_b1.dln_t_dln_m,
        0x0000_0000_0000_0000,
        0,
    );
    assert_hex_ulp("B1 dlnP_dl", eval_b1.dln_p_dl, 0x38E4_CADE_610B_41BD, 0);
    assert_hex_ulp(
        "B1 dlnP_dlnM",
        eval_b1.dln_p_dln_m,
        0x0000_0000_0000_0000,
        0,
    );

    let st_b1 = build_t_tau_varying(
        tau_surf,
        L_SUN_TEST,
        R_SUN_TEST,
        teff,
        M_SUN_TEST,
        CGRAV_TEST,
        eval_b1.ln_p,
        1.0e-4,
        ATM_T_TAU_EDDINGTON,
        &mut eos,
        analytic_kap_cb,
        1.0e-5,
        0.1,
    )
    .expect("build_t_tau_varying B1");
    assert_eq!(st_b1.points.len(), 40);
    assert_structure_row("VAR_B1 pt1", &st_b1.points[0], &B1_K1, 0);
    assert_structure_row("VAR_B1 pt20", &st_b1.points[19], &B1_K20, 0);
    assert_structure_row("VAR_B1 pt40", &st_b1.points[39], &B1_K40, 0);
}

#[test]
fn test_build_t_tau_varying_19_col_trampedach_parity() {
    let mut eos = make_analytic_eos(-1.0e99);
    let lum = 1.2 * L_SUN_TEST;
    let rad = 1.1 * R_SUN_TEST;
    let teff = atm_teff(lum, rad);
    let tau_surf = get_t_tau_base(ATM_T_TAU_TRAMPEDACH_SOLAR).unwrap() * 0.1;
    let eval_b2 = eval_t_tau_varying(
        tau_surf,
        lum,
        rad,
        M_SUN_TEST,
        CGRAV_TEST,
        ATM_T_TAU_TRAMPEDACH_SOLAR,
        &mut eos,
        analytic_kap_cb,
        1.0e-5,
        500,
        false,
        teff,
    )
    .expect("eval_t_tau_varying B2");
    assert_hex_ulp("B2 lnT", eval_b2.ln_t, 0x4020_FCB8_D38A_5E64, 0);
    assert_hex_ulp("B2 lnP", eval_b2.ln_p, 0x401E_3646_84F1_6202, 0);
    assert_hex_ulp("B2 dlnP_dl", eval_b2.dln_p_dl, 0x38D8_6344_699D_4DC4, 0);

    let st_b2 = build_t_tau_varying(
        tau_surf,
        lum,
        rad,
        teff,
        M_SUN_TEST,
        CGRAV_TEST,
        eval_b2.ln_p,
        1.0e-4,
        ATM_T_TAU_TRAMPEDACH_SOLAR,
        &mut eos,
        analytic_kap_cb,
        1.0e-5,
        0.05,
    )
    .expect("build_t_tau_varying B2");
    assert_eq!(st_b2.points.len(), 56);
    assert_structure_row("VAR_B2 pt1", &st_b2.points[0], &B2_K1, 0);
    assert_structure_row("VAR_B2 pt28", &st_b2.points[27], &B2_K28, 0);
    assert_structure_row("VAR_B2 pt56", &st_b2.points[55], &B2_K56, 0);

    let empty = build_t_tau_varying(
        1.0e-4,
        lum,
        rad,
        teff,
        M_SUN_TEST,
        CGRAV_TEST,
        eval_b2.ln_p,
        1.0e-3,
        ATM_T_TAU_TRAMPEDACH_SOLAR,
        &mut eos,
        analytic_kap_cb,
        1.0e-5,
        0.05,
    )
    .expect("tau_outer > tau_surf returns empty");
    assert_eq!(empty.points.len(), 0);
}

#[test]
fn test_eval_t_tau_varying_retry_and_dispatch() {
    let teff = atm_teff(L_SUN_TEST, R_SUN_TEST);
    let tau_surf = 2.0 / 3.0;

    let mut eos_try2 = make_analytic_eos(3.0);
    let out_try2 = eval_t_tau_varying(
        tau_surf,
        L_SUN_TEST,
        R_SUN_TEST,
        M_SUN_TEST,
        CGRAV_TEST,
        ATM_T_TAU_EDDINGTON,
        &mut eos_try2,
        analytic_kap_cb,
        1.0e-5,
        500,
        true,
        teff,
    )
    .expect("retry 2 should succeed");
    assert_hex_ulp("TRY2 lnT", out_try2.ln_t, 0x4021_5251_097F_3253, 0);
    assert_hex_ulp("TRY2 lnP", out_try2.ln_p, 0x4022_F027_49FA_1D60, 0);

    let mut eos_try3 = make_analytic_eos(3.6);
    let out_try3 = eval_t_tau_varying(
        tau_surf,
        L_SUN_TEST,
        R_SUN_TEST,
        M_SUN_TEST,
        CGRAV_TEST,
        ATM_T_TAU_EDDINGTON,
        &mut eos_try3,
        analytic_kap_cb,
        1.0e-5,
        500,
        true,
        teff,
    )
    .expect("retry 3 should succeed");
    assert_hex_ulp("TRY3 lnT", out_try3.ln_t, 0x4021_5251_097F_3253, 0);
    assert_hex_ulp("TRY3 lnP", out_try3.ln_p, 0x4022_F027_49FA_1D60, 0);

    let mut eos_fail = make_analytic_eos(6.5);
    assert!(eval_t_tau_varying(
        tau_surf,
        L_SUN_TEST,
        R_SUN_TEST,
        M_SUN_TEST,
        CGRAV_TEST,
        ATM_T_TAU_EDDINGTON,
        &mut eos_fail,
        analytic_kap_cb,
        1.0e-5,
        500,
        true,
        teff,
    )
    .is_err());

    let mut eos_ok = make_analytic_eos(-1.0e99);
    for op in ["fixed", "iterated", "varying"] {
        let out = eval_t_tau_atm(
            tau_surf,
            L_SUN_TEST,
            R_SUN_TEST,
            M_SUN_TEST,
            CGRAV_TEST,
            0.35,
            1.0,
            "Krishna_Swamy",
            op,
            &mut eos_ok,
            analytic_kap_cb,
            1.0e-6,
            50,
            500,
            false,
            teff,
        )
        .expect("eval_t_tau_atm dispatch");
        assert!(out.ln_t.is_finite() && out.ln_p.is_finite());
    }
    assert!(matches!(
        eval_t_tau_atm(
            tau_surf,
            L_SUN_TEST,
            R_SUN_TEST,
            M_SUN_TEST,
            CGRAV_TEST,
            0.35,
            1.0,
            "Eddington",
            "bad_opacity",
            &mut eos_ok,
            analytic_kap_cb,
            1.0e-6,
            50,
            500,
            true,
            teff,
        ),
        Err(AtmError::InvalidOption { .. })
    ));
}

#[test]
fn test_get_table_values_clamping_and_z_slices_parity() {
    let z_slices: [(u64, u64, u64, u64); 4] = [
        (
            0x3F5D_53CD_DD6E_04C0,
            0x412A_F387_FFFF_FFFF,
            0xC07F_9382_D5C7_A0B7,
            1,
        ),
        (
            0x3F77_2F70_BA51_FDAE,
            0x4121_92D4_0000_0000,
            0xC065_17A9_23B7_3C65,
            0,
        ),
        (
            0x3F92_5460_AA64_C2F8,
            0x4115_3308_0000_0000,
            0xC052_0464_B396_7BB6,
            0,
        ),
        (
            0x3FAC_FB4C_E8E6_7D17,
            0x4108_5560_0000_0001,
            0xC04D_1B9C_FC21_AFC8,
            2,
        ),
    ];
    for (idx, (zh, pgh, dpgh, max_ulp)) in z_slices.into_iter().enumerate() {
        let (pgas, dpgas_teff, dpgas_logg, _, _, _) = get_table_values(
            ATM_TABLE_PHOTOSPHERE,
            hex_to_f64(zh),
            7.0,
            5500.0,
        )
        .expect("z_slice");
        assert_hex_ulp(&format!("ZSLICE {} Pgas", idx + 1), pgas, pgh, 0);
        assert_hex_ulp(
            &format!("ZSLICE {} dPgas_dTeff", idx + 1),
            dpgas_teff,
            dpgh,
            max_ulp,
        );
        assert_hex_ulp(
            &format!("ZSLICE {} dPgas_dlogg", idx + 1),
            dpgas_logg,
            0,
            0,
        );
    }

    for case in &GTV_CASES {
        let (pgas, dpgas_teff, dpgas_logg, temp, dtemp_teff, dtemp_logg) =
            get_table_values(case.tid, case.z, case.logg, case.teff)
                .expect(case.name);
        assert_hex_ulp(
            &format!("{} Pgas", case.name),
            pgas,
            case.expected[0],
            0,
        );
        assert_hex_ulp(
            &format!("{} dPgas_dTeff", case.name),
            dpgas_teff,
            case.expected[1],
            case.dpgas_teff_max_ulp,
        );
        assert_hex_ulp(
            &format!("{} dPgas_dlogg", case.name),
            dpgas_logg,
            case.expected[2],
            0,
        );
        assert_hex_ulp(&format!("{} T", case.name), temp, case.expected[3], 0);
        assert_hex_ulp(
            &format!("{} dT_dTeff", case.name),
            dtemp_teff,
            case.expected[4],
            0,
        );
        assert_hex_ulp(
            &format!("{} dT_dlogg", case.name),
            dtemp_logg,
            case.expected[5],
            0,
        );
    }
}

#[test]
fn test_eval_table_alfa_beta_regions() {
    for case in &TBL_CASES {
        let h = case.hex;
        let lum = hex_to_f64(h[0]);
        let rad = hex_to_f64(h[1]);
        let mass = hex_to_f64(h[2]);
        let teff = hex_to_f64(h[3]);
        let (alfa, beta) =
            get_table_alfa_beta(lum, teff, rad, mass, CGRAV_TEST, case.tid)
                .expect(case.name);
        assert_hex_ulp(&format!("{} alfa", case.name), alfa, h[4], 0);
        assert_hex_ulp(&format!("{} beta", case.name), beta, h[5], 0);

        let out =
            eval_table(lum, rad, mass, CGRAV_TEST, case.tid, 0.02, false, teff)
                .expect(case.name);
        assert_hex_ulp(&format!("{} lnT", case.name), out.ln_t, h[6], 0);
        assert_hex_ulp(
            &format!("{} dlnT_dl", case.name),
            out.dln_t_dl,
            h[7],
            0,
        );
        assert_hex_ulp(
            &format!("{} dlnT_dlnR", case.name),
            out.dln_t_dln_r,
            h[8],
            0,
        );
        assert_hex_ulp(&format!("{} lnP", case.name), out.ln_p, h[9], 0);
        assert_hex_ulp(
            &format!("{} dlnP_dl", case.name),
            out.dln_p_dl,
            h[10],
            0,
        );
        assert_hex_ulp(
            &format!("{} dlnP_dlnM", case.name),
            out.dln_p_dln_m,
            h[11],
            0,
        );
    }
}

#[test]
fn test_eval_table_atm_off_table_options() {
    let mut eos = make_analytic_eos(-1.0e99);
    let pure_h = TBL_CASES[0].hex;
    assert_hex_ulp(
        "tau_100 base",
        get_table_base(ATM_TABLE_TAU_100).unwrap(),
        100.0_f64.to_bits(),
        0,
    );
    let tau_err = eval_table_atm(
        hex_to_f64(pure_h[0]),
        hex_to_f64(pure_h[1]),
        hex_to_f64(pure_h[2]),
        CGRAV_TEST,
        "tau_100",
        0.02,
        0.5,
        "T_tau",
        "Eddington",
        "fixed",
        0.35,
        1.0,
        &mut eos,
        analytic_kap_cb,
        1.0e-5,
        50,
        500,
        true,
        hex_to_f64(pure_h[3]),
    );
    assert!(matches!(tau_err, Err(AtmError::InvalidArgument { .. })));

    let grey_h = TBL_CASES[6].hex;
    let lum_grey = hex_to_f64(grey_h[0]);
    let rad_grey = hex_to_f64(grey_h[1]);
    let mass_grey = hex_to_f64(grey_h[2]);
    let teff_grey = hex_to_f64(grey_h[3]);

    assert!(eval_table_atm(
        lum_grey,
        rad_grey,
        mass_grey,
        CGRAV_TEST,
        "tau_100",
        0.02,
        1.0,
        "",
        "Eddington",
        "fixed",
        0.35,
        1.0,
        &mut eos,
        analytic_kap_cb,
        1.0e-5,
        50,
        500,
        true,
        teff_grey,
    )
    .is_err());

    let off_bad = eval_table_atm(
        lum_grey,
        rad_grey,
        mass_grey,
        CGRAV_TEST,
        "tau_100",
        0.02,
        1.0,
        "fixed_Tsurf",
        "Eddington",
        "fixed",
        0.35,
        1.0,
        &mut eos,
        analytic_kap_cb,
        1.0e-5,
        50,
        500,
        true,
        teff_grey,
    );
    assert!(matches!(off_bad, Err(AtmError::InvalidOption { .. })));
}

#[test]
fn test_irradiated_mesa_26_4_1_parity() {
    let mut eos = make_analytic_eos(-1.0e99);
    let l_irr = 1.0e-4 * L_SUN_TEST;
    let r_irr = 0.1 * R_SUN_TEST;
    let m_irr = 1.0e-3 * M_SUN_TEST;

    let irr_gp = eval_irradiated(
        l_irr,
        r_irr,
        m_irr,
        CGRAV_TEST,
        1500.0,
        2.0e5,
        0.02,
        1.0e-3,
        0.5,
        &mut eos,
        analytic_kap_cb,
        1.0e-7,
        100,
        false,
    )
    .expect("IRR GP");
    assert_hex_ulp("IRR GP kap", irr_gp.kap, 0x4019_610D_2FAB_4DC8, 0);
    assert_hex_ulp("IRR GP tau", irr_gp.tau, 0x407C_EC8C_CEF6_EF72, 0);
    assert_hex_ulp("IRR GP lnT", irr_gp.ln_t, 0x4021_F209_6EFF_779D, 0);
    assert_hex_ulp("IRR GP dlnT_dl", irr_gp.dln_t_dl, 0x3999_FDAB_5208_E429, 0);
    assert_hex_ulp(
        "IRR GP dlnT_dlnR",
        irr_gp.dln_t_dln_r,
        0xBF37_1F45_F57D_3094,
        0,
    );
    assert_hex_ulp(
        "IRR GP dlnT_dlnM",
        irr_gp.dln_t_dln_m,
        0xBFBF_596E_6E98_03D1,
        0,
    );
    assert_hex_ulp("IRR GP dlnT_dlnkap", irr_gp.dln_t_dln_kap, 0, 0);

    let irr_teq0 = eval_irradiated(
        l_irr,
        r_irr,
        m_irr,
        CGRAV_TEST,
        0.0,
        2.0e5,
        0.02,
        1.0e-3,
        0.5,
        &mut eos,
        analytic_kap_cb,
        1.0e-7,
        0,
        false,
    )
    .expect("IRR TEQ0");
    assert_hex_ulp("IRR TEQ0 kap", irr_teq0.kap, 0x3F94_7AE1_47AE_147B, 0);
    assert_hex_ulp("IRR TEQ0 tau", irr_teq0.tau, 0x3FF7_5732_A084_4FE3, 0);
    assert_hex_ulp("IRR TEQ0 lnT", irr_teq0.ln_t, 0x401E_8115_557F_2F76, 0);
    assert_hex_ulp(
        "IRR TEQ0 dlnT_dl",
        irr_teq0.dln_t_dl,
        0x39AA_7DFF_3791_9155,
        0,
    );
    assert_hex_ulp(
        "IRR TEQ0 dlnT_dlnR",
        irr_teq0.dln_t_dln_r,
        0xBFC4_12FA_46DB_01FA,
        0,
    );
    assert_hex_ulp(
        "IRR TEQ0 dlnT_dlnM",
        irr_teq0.dln_t_dln_m,
        0xBFC5_F682_DC92_7F03,
        0,
    );
    assert_hex_ulp(
        "IRR TEQ0 dlnT_dlnkap",
        irr_teq0.dln_t_dln_kap,
        0x3FC5_F682_DC92_7F03,
        0,
    );
}

fn check_irr_dl_fd(kap_init: f64, errtol: f64, max_iters: usize) {
    let mut eos = make_analytic_eos(-1.0e99);
    let l_irr = 1.0e-4 * L_SUN_TEST;
    let r_irr = 0.1 * R_SUN_TEST;
    let m_irr = 1.0e-3 * M_SUN_TEST;
    let eps = 1.0e-6;
    let mut run = |lum: f64, skip: bool| {
        eval_irradiated(
            lum,
            r_irr,
            m_irr,
            CGRAV_TEST,
            1500.0,
            2.0e5,
            kap_init,
            1.0e-3,
            0.5,
            &mut eos,
            analytic_kap_cb,
            errtol,
            max_iters,
            skip,
        )
        .unwrap()
    };
    let center = run(l_irr, false);
    let hi = run(l_irr * (1.0 + eps), true);
    let lo = run(l_irr * (1.0 - eps), true);
    let fd_dl = (hi.ln_t - lo.ln_t) / (2.0 * eps * l_irr);
    assert!((fd_dl - center.dln_t_dl).abs() <= 1.0e-6 * center.dln_t_dl.abs());
}

#[test]
fn test_irradiated_fd_partials() {
    let kap_gp = hex_to_f64(0x4019_610D_2FAB_4DC8);
    check_irr_dl_fd(kap_gp, 1.0e-7, 0);
    check_irr_dl_fd(kap_gp, 1.0e-13, 200);
}

#[test]
fn test_e2_clamped_slopes_and_t_tau_dq_dtau_parity() {
    let (e2_neg1, de2_neg1) = eval_e2(-1.0).expect("eval_e2(-1.0)");
    assert_hex_ulp("E2(-1.0)", e2_neg1, 0x3FF0_0000_0000_0000, 0);
    assert_hex_ulp("dE2(-1.0)/dx", de2_neg1, 0xC017_218E_65D9_3703, 0);
    let (e2_12, de2_12) = eval_e2(12.0).expect("eval_e2(12.0)");
    assert_hex_ulp("E2(12.0)", e2_12, 0x3E9D_B937_070B_9D1B, 0);
    assert_hex_ulp("dE2(12.0)/dx", de2_12, 0xBE9F_E2F8_911F_FFA5, 0);

    let teff = 5772.0;
    let tau_probe = 0.35;
    let h = 1.0e-6;
    for id in [
        ATM_T_TAU_EDDINGTON,
        ATM_T_TAU_SOLAR_HOPF,
        ATM_T_TAU_KRISHNA_SWAMY,
    ] {
        let ln_t_hi = eval_t_tau(id, tau_probe + h, teff).unwrap();
        let ln_t_lo = eval_t_tau(id, tau_probe - h, teff).unwrap();
        let dt4_dtau = (exp(4.0 * ln_t_hi) - exp(4.0 * ln_t_lo)) / (2.0 * h);
        let q_prime_fd = dt4_dtau / (0.75 * pow(teff, 4.0)) - 1.0;
        let q_prime = eval_t_tau_dq_dtau(id, tau_probe).unwrap();
        assert!(
            (q_prime_fd - q_prime).abs() < 1.0e-8,
            "id={id}: q_prime={q_prime}, fd={q_prime_fd}"
        );
    }

    let ln_t_hi =
        eval_t_tau(ATM_T_TAU_TRAMPEDACH_SOLAR, tau_probe + h, teff).unwrap();
    let ln_t_lo =
        eval_t_tau(ATM_T_TAU_TRAMPEDACH_SOLAR, tau_probe - h, teff).unwrap();
    let dt4_dtau = (exp(4.0 * ln_t_hi) - exp(4.0 * ln_t_lo)) / (2.0 * h);
    let q_prime_fd = dt4_dtau / (0.75 * pow(teff, 4.0)) - 1.0;
    let b_ball = 0.114_874_290_276_943_3;
    let w_ball = 0.051_499_904_716_986_9;
    let ball_factor = 1.0 + exp((log10(tau_probe) - b_ball) / w_ball);
    let q_prime_tramp =
        eval_t_tau_dq_dtau(ATM_T_TAU_TRAMPEDACH_SOLAR, tau_probe).unwrap();
    assert!((q_prime_tramp - q_prime_fd / ball_factor).abs() < 1.0e-8);
}

#[test]
fn test_invalid_table_and_e2_inputs_fail_loudly() {
    let teff = atm_teff(L_SUN_TEST, R_SUN_TEST);
    assert!(matches!(
        get_t_tau_id("invalid_rel"),
        Err(AtmError::InvalidOption { .. })
    ));
    assert!(matches!(
        get_table_id("invalid_table"),
        Err(AtmError::InvalidOption { .. })
    ));
    assert!(matches!(
        get_t_tau_base(999),
        Err(AtmError::UnknownId { .. })
    ));
    assert!(matches!(
        get_table_base(999),
        Err(AtmError::UnknownId { .. })
    ));
    assert!(matches!(
        eval_e2(f64::NAN),
        Err(AtmError::InvalidArgument { .. })
    ));
    assert!(matches!(
        eval_e2(f64::INFINITY),
        Err(AtmError::InvalidArgument { .. })
    ));
    assert!(matches!(
        get_table_values(ATM_TABLE_PHOTOSPHERE, -0.01, 4.0, 5500.0),
        Err(AtmError::InvalidArgument { .. })
    ));
    assert!(matches!(
        get_table_values(ATM_TABLE_PHOTOSPHERE, 0.02, f64::NAN, 5500.0),
        Err(AtmError::InvalidArgument { .. })
    ));
    assert!(matches!(
        eval_table(
            f64::NAN,
            R_SUN_TEST,
            M_SUN_TEST,
            CGRAV_TEST,
            ATM_TABLE_TAU_100,
            0.02,
            true,
            teff,
        ),
        Err(AtmError::InvalidArgument { .. })
    ));
    assert!(matches!(
        eval_table(
            L_SUN_TEST,
            R_SUN_TEST,
            M_SUN_TEST,
            CGRAV_TEST,
            ATM_TABLE_TAU_100,
            0.02,
            true,
            0.0,
        ),
        Err(AtmError::InvalidArgument { .. })
    ));
    assert!(matches!(
        get_table_alfa_beta(
            L_SUN_TEST,
            f64::NAN,
            R_SUN_TEST,
            M_SUN_TEST,
            CGRAV_TEST,
            ATM_TABLE_TAU_100,
        ),
        Err(AtmError::InvalidArgument { .. })
    ));
}

#[test]
fn test_invalid_t_tau_inputs_fail_loudly() {
    let mut eos = make_analytic_eos(-1.0e99);
    let teff = atm_teff(L_SUN_TEST, R_SUN_TEST);

    assert!(matches!(
        eval_t_tau_uniform(
            f64::NAN,
            L_SUN_TEST,
            R_SUN_TEST,
            M_SUN_TEST,
            CGRAV_TEST,
            0.35,
            1.0,
            ATM_T_TAU_EDDINGTON,
            &mut eos,
            analytic_kap_cb,
            1.0e-4,
            0,
            true,
            teff,
        ),
        Err(AtmError::InvalidArgument { .. })
    ));
    assert!(matches!(
        eval_t_tau_uniform(
            2.0 / 3.0,
            L_SUN_TEST,
            R_SUN_TEST,
            M_SUN_TEST,
            CGRAV_TEST,
            0.35,
            1.0,
            ATM_T_TAU_EDDINGTON,
            &mut eos,
            analytic_kap_cb,
            0.0,
            10,
            true,
            teff,
        ),
        Err(AtmError::InvalidArgument { .. })
    ));
    assert!(matches!(
        build_t_tau_uniform(
            2.0 / 3.0,
            L_SUN_TEST,
            R_SUN_TEST,
            teff,
            M_SUN_TEST,
            CGRAV_TEST,
            0.35,
            1.0,
            1.0e-4,
            ATM_T_TAU_EDDINGTON,
            &mut eos,
            1.0e-4,
            f64::NAN,
        ),
        Err(AtmError::InvalidArgument { .. })
    ));
}

#[test]
fn test_invalid_varying_and_irradiated_inputs_fail_loudly() {
    let mut eos = make_analytic_eos(-1.0e99);
    let teff = atm_teff(L_SUN_TEST, R_SUN_TEST);

    assert!(matches!(
        eval_t_tau_varying(
            2.0 / 3.0,
            L_SUN_TEST,
            R_SUN_TEST,
            M_SUN_TEST,
            CGRAV_TEST,
            ATM_T_TAU_EDDINGTON,
            &mut eos,
            analytic_kap_cb,
            0.0,
            100,
            true,
            teff,
        ),
        Err(AtmError::InvalidArgument { .. })
    ));
    assert!(matches!(
        build_t_tau_varying(
            2.0 / 3.0,
            L_SUN_TEST,
            R_SUN_TEST,
            teff,
            M_SUN_TEST,
            CGRAV_TEST,
            f64::NAN,
            1.0e-4,
            ATM_T_TAU_EDDINGTON,
            &mut eos,
            analytic_kap_cb,
            1.0e-5,
            0.1,
        ),
        Err(AtmError::InvalidArgument { .. })
    ));
    assert!(matches!(
        eval_irradiated(
            L_SUN_TEST,
            R_SUN_TEST,
            M_SUN_TEST,
            CGRAV_TEST,
            1000.0,
            1.0e5,
            0.35,
            0.0,
            0.0,
            &mut eos,
            analytic_kap_cb,
            1.0e-6,
            10,
            true,
        ),
        Err(AtmError::InvalidArgument { .. })
    ));
}
