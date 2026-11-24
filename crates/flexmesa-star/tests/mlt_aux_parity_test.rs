// SPDX-License-Identifier: GPL-3.0-or-later
//! Bit-exact MESA 26.4.1 `AUX` parity regression tests for MLT/TDC (`#393`).
//!
//! Verifies upstream-exact IEEE-754 parity for `gradT` and `mlt_vc` across:
//! 1. Uniform-omega spin-up via `use_xh_to_update_i_rot_and_j_rot` +
//!    `set_rotation_info(false)` (`star/private/hydro_rotation.f90:501-543`,
//!    fixing `star/conserve_angular_momentum` zone 1).
//! 2. Preserving `dt` separately from `dt_next` and keeping
//!    `okay_to_set_mlt_vc = !have_mlt_vc` on `.mod` load
//!    (`star/private/init.f90:541`, `star/private/read_model.f90:157-162`,
//!    `490,498`, fixing `star/gyre_in_mesa_rsg` zone 46 and TDC `vc0`
//!    cases).
//! 3. Direct cell-pressure, radius, energy, and `rho_start` wrapping in
//!    `do1_mlt_2` (`wrap_Peos_00`/`wrap_Peos_m1`, `wrap_r_00`, `wrap_e_00`,
//!    `get_rho_start_face`, and `set_rotation_info(true)` in
//!    `star/private/hydro_vars.f90:541`, fixing `star/R_CrB_star` zone 69 and
//!    `star/high_mass` zone 558 plus `1-2 ULP` face cases).

use flexmesa_const::{M_SUN, STANDARD_CGRAV};
use flexmesa_star::controls::{StarControls, StarJobControls};
use flexmesa_star::hydro_vars::{set_hydro_vars, set_mlt_vars};
use flexmesa_star::model::StarModel;
use flexmesa_star::model_io::read_mod_str;
use flexmesa_star::relax::apply_uniform_rotation;
use flexmesa_star::star_ad::{StarAdExt, StarAdOrder1};

#[derive(Clone, Copy)]
struct CellHex {
    dq: u64,
    dm: u64,
    m: u64,
    m_grav: u64,
    cgrav: u64,
    lnd: u64,
    rho: u64,
    ln_t: u64,
    t: u64,
    ln_r: u64,
    r: u64,
    lum: u64,
    peos: u64,
    ln_pgas: u64,
    opacity: u64,
    grada: u64,
    cp: u64,
    chi_rho: u64,
    chi_t: u64,
    energy: u64,
    d_lnpeos_dlnd: u64,
    d_lnpeos_dlnt: u64,
}

struct MltCaseHex {
    name: &'static str,
    zone: usize,
    dt: u64,
    alpha: u64,
    vc0: u64,
    km1: CellHex,
    k0: CellHex,
    expected_gradt: u64,
    expected_mlt_vc: u64,
}

fn populate_cell(model: &mut StarModel, idx: usize, c: CellHex) {
    model.dq[idx] = f64::from_bits(c.dq);
    model.dm[idx] = f64::from_bits(c.dm);
    model.m[idx] = f64::from_bits(c.m);
    model.m_grav[idx] = f64::from_bits(c.m_grav);
    model.cgrav[idx] = f64::from_bits(c.cgrav);
    model.lnd[idx] = f64::from_bits(c.lnd);
    model.rho[idx] = f64::from_bits(c.rho);
    model.ln_t[idx] = f64::from_bits(c.ln_t);
    model.t[idx] = f64::from_bits(c.t);
    model.ln_r[idx] = f64::from_bits(c.ln_r);
    model.r[idx] = f64::from_bits(c.r);
    model.lum[idx] = f64::from_bits(c.lum);
    model.peos[idx] = f64::from_bits(c.peos);
    model.ln_peos[idx] = flexmesa_math::log(model.peos[idx]);
    model.ln_pgas[idx] = f64::from_bits(c.ln_pgas);
    model.pgas[idx] = flexmesa_math::exp(model.ln_pgas[idx]);
    model.opacity[idx] = f64::from_bits(c.opacity);
    model.grada[idx] = f64::from_bits(c.grada);
    model.cp[idx] = f64::from_bits(c.cp);
    model.chi_rho[idx] = f64::from_bits(c.chi_rho);
    model.chi_t[idx] = f64::from_bits(c.chi_t);
    model.energy[idx] = f64::from_bits(c.energy);
    model.ln_e[idx] = flexmesa_math::log(model.energy[idx]);
    model.ln_peos_ad[idx] = StarAdOrder1::from_dlnd_dlnt_00(
        model.ln_peos[idx],
        f64::from_bits(c.d_lnpeos_dlnd),
        f64::from_bits(c.d_lnpeos_dlnt),
    );
}

fn build_two_zone_mlt_model(case: &MltCaseHex) -> StarModel {
    let controls = StarControls {
        mixing_length_alpha: f64::from_bits(case.alpha),
        ..StarControls::default()
    };
    let mut model =
        StarModel::new(2, vec![1], StarJobControls::default(), controls);
    model.dt = f64::from_bits(case.dt);
    model.dt_next = model.dt;
    model.aux.have_mlt_vc = true;
    model.aux.okay_to_set_mlt_vc = true;
    model.aux.mlt_vc_old[1] = f64::from_bits(case.vc0);
    model.aux.mixing.gradr_factor.fill(1.0);
    model.aux.mixing.alpha_mlt.fill(f64::from_bits(case.alpha));
    populate_cell(&mut model, 0, case.km1);
    populate_cell(&mut model, 1, case.k0);
    model.lnd_start.clone_from(&model.lnd);
    model.lnd_old.clone_from(&model.lnd);
    model.ln_t_start.clone_from(&model.ln_t);
    model
}

fn assert_hex_eq(label: &str, got: f64, expected_bits: u64) {
    assert_eq!(
        got.to_bits(),
        expected_bits,
        "{label}: got 0x{:016X} ({got:.16e}), expected 0x{:016X} ({:.16e})",
        got.to_bits(),
        expected_bits,
        f64::from_bits(expected_bits),
    );
}

const CASES: &[MltCaseHex] = &[
    MltCaseHex {
        name: "gyre_in_mesa_rsg",
        zone: 46,
        dt: 0x41E4_F0FB_E1B9_AAC5,
        alpha: 0x4000_0000_0000_0000,
        vc0: 0x406C_89CA_11BD_774C,
        km1: CellHex {
            dq: 0x3F02_788B_6145_9B90,
            dm: 0x4632_9163_169D_BD62,
            m: 0x4720_1222_DA34_18F4,
            m_grav: 0x4720_1222_DA34_18F4,
            cgrav: 0x3E71_EA8B_45ED_F98C,
            lnd: 0xC033_AD25_DDC9_2952,
            rho: 0x3E28_78A2_29E3_156D,
            ln_t: 0x4020_5ED1_BD4A_E519,
            t: 0x40AC_06E0_E976_1AB6,
            ln_r: 0x403F_F278_80B5_F2D0,
            r: 0x42D1_07A6_6ED5_407D,
            lum: 0x47F4_158A_7A2C_33AA,
            peos: 0x4084_1713_A153_DCE9,
            ln_pgas: 0x4019_DC7B_9B21_3C7D,
            opacity: 0x3F63_6B7E_CF7F_D7EE,
            grada: 0x3FD7_93A8_FA8F_B7C5,
            cp: 0x41A4_BBD5_5038_C135,
            chi_rho: 0x3FEF_F251_66E4_7B30,
            chi_t: 0x3FF0_4520_4AFA_EC4B,
            energy: 0x427A_59EA_5627_A7C8,
            d_lnpeos_dlnd: 0x3FEF_F251_75BF_63A4,
            d_lnpeos_dlnt: 0x3FF0_4507_C13E_9EA0,
        },
        k0: CellHex {
            dq: 0x3F00_68D7_3EC7_8AD0,
            dm: 0x4630_7EE9_3674_F1F1,
            m: 0x4720_11FD_B76D_EBB9,
            m_grav: 0x4720_11FD_B76D_EBB9,
            cgrav: 0x3E71_EA8B_45ED_F98C,
            lnd: 0xC033_AB05_204F_BEAB,
            rho: 0x3E28_ACEC_2D69_99F7,
            ln_t: 0x4020_623F_AAA4_A25E,
            t: 0x40AC_3718_FB30_8C67,
            ln_r: 0x403F_F272_179B_31FD,
            r: 0x42D1_0739_4461_B315,
            lum: 0x47F4_158A_8877_7004,
            peos: 0x4084_655A_54C7_7277,
            ln_pgas: 0x4019_EBF1_4D9D_D0BD,
            opacity: 0x3F64_A9EA_7734_7ABF,
            grada: 0x3FD7_BDF7_3A22_7D66,
            cp: 0x41A4_9073_BA3D_CE2C,
            chi_rho: 0x3FEF_F2F1_5BDB_5716,
            chi_t: 0x3FF0_3FE2_6E2C_4387,
            energy: 0x427A_6382_C10F_FCFD,
            d_lnpeos_dlnd: 0x3FEF_F2F1_6B5D_F7A1,
            d_lnpeos_dlnt: 0x3FF0_3FC4_96DF_4379,
        },
        expected_gradt: 0x3FD8_6CAE_6102_16E7,
        expected_mlt_vc: 0x4060_2204_20DB_DE39,
    },
    MltCaseHex {
        name: "R_CrB_star",
        zone: 69,
        dt: 0x421D_291B_DD29_4AF3,
        alpha: 0x4000_0000_0000_0000,
        vc0: 0x404D_3026_3887_7D31,
        km1: CellHex {
            dq: 0x3E02_8904_B0FD_7F00,
            dm: 0x44E8_D7EE_1345_BB00,
            m: 0x46D5_7206_968F_5D38,
            m_grav: 0x46D5_7206_968F_5D38,
            cgrav: 0x3E71_EA8B_45ED_F98C,
            lnd: 0xC034_1A5D_5A01_F28D,
            rho: 0x3E1F_F1EE_BA95_8FD7,
            ln_t: 0x4024_782A_A1A4_32A0,
            t: 0x40DB_3347_A6CD_85A5,
            ln_r: 0x403B_1174_7F09_3358,
            r: 0x4260_93C8_0324_2A52,
            lum: 0x47BB_057E_7CD3_D540,
            peos: 0x40AB_ABDC_E5E9_796E,
            ln_pgas: 0x401E_7396_349A_2FBE,
            opacity: 0x3FE1_72F1_A927_6D7C,
            grada: 0x3FCF_5372_2A1E_F112,
            cp: 0x41D1_1775_8D51_8C32,
            chi_rho: 0x3FE2_36E7_B894_ED25,
            chi_t: 0x4002_B11A_549E_862B,
            energy: 0x42A1_C61F_CB15_EF3C,
            d_lnpeos_dlnd: 0x3FE2_36E8_A697_64C5,
            d_lnpeos_dlnt: 0x4002_B0D3_B5B6_2A3F,
        },
        k0: CellHex {
            dq: 0x3E0B_CD87_097C_3E80,
            dm: 0x44F2_A1F2_8E74_4C40,
            m: 0x46D5_7206_965D_AD5C,
            m_grav: 0x46D5_7206_965D_AD5C,
            cgrav: 0x3E71_EA8B_45ED_F98C,
            lnd: 0xC034_1566_120E_CD33,
            rho: 0x3E20_490F_285D_2674,
            ln_t: 0x4024_7C07_6C2B_92EC,
            t: 0x40DB_6802_3789_CF0F,
            ln_r: 0x403B_1166_3F26_229D,
            r: 0x4260_92DB_D1B4_885E,
            lum: 0x47BB_057E_7CD3_DDCB,
            peos: 0x40AC_7A09_F320_65FB,
            ln_pgas: 0x401E_8FD2_7EC2_38F0,
            opacity: 0x3FE1_AD88_CFAF_8C13,
            grada: 0x3FCF_0AD4_554A_DCE0,
            cp: 0x41D1_5E4A_FD1E_46F0,
            chi_rho: 0x3FE2_2F58_E11D_56E4,
            chi_t: 0x4002_C26F_B0B5_B124,
            energy: 0x42A1_DFBB_39C4_0E76,
            d_lnpeos_dlnd: 0x3FE2_2F59_EA4B_3EFB,
            d_lnpeos_dlnt: 0x4002_C269_D352_22B4,
        },
        expected_gradt: 0x3FD0_CB50_589C_8A3B,
        expected_mlt_vc: 0x404D_3023_7EE6_49D4,
    },
    MltCaseHex {
        name: "high_mass",
        zone: 558,
        dt: 0x41F2_EE3F_6F55_C191,
        alpha: 0x4000_0000_0000_0000,
        vc0: 0x40DF_7C32_FE45_D0A8,
        km1: CellHex {
            dq: 0x3F4C_DD1B_9E5F_9EF8,
            dm: 0x46B9_E814_372A_EAB0,
            m: 0x4759_F07A_134F_C6AF,
            m_grav: 0x4759_F07A_134F_C6AF,
            cgrav: 0x3E71_EA8B_45ED_F98C,
            lnd: 0xC004_006A_2DCB_4A7A,
            rho: 0x3FB5_026E_E0D9_A773,
            ln_t: 0x4030_BEF5_0C90_D796,
            t: 0x4171_DE14_0943_D81F,
            ln_r: 0x403B_3DD0_C19D_3A2F,
            r: 0x4263_B6B6_F258_6348,
            lum: 0x4851_F9AB_6C00_BAB5,
            peos: 0x42FD_EC8E_5A4C_C5B2,
            ln_pgas: 0x4040_80A0_B572_DECA,
            opacity: 0x3FD5_8639_68CC_8BBE,
            grada: 0x3FD0_5ED5_9F6B_9665,
            cp: 0x4200_DC8C_F95B_9EDB,
            chi_rho: 0x3FDA_389F_CA20_FA38,
            chi_t: 0x4006_2AC6_FC50_0211,
            energy: 0x434B_38D6_FF90_6BBB,
            d_lnpeos_dlnd: 0x3FDA_389F_912B_ED5E,
            d_lnpeos_dlnt: 0x4006_2AC7_11AB_D84A,
        },
        k0: CellHex {
            dq: 0x3F5C_DD1B_9E5F_9E80,
            dm: 0x46C9_E814_372A_EA45,
            m: 0x4759_EA00_0E41_FBF4,
            m_grav: 0x4759_EA00_0E41_FBF4,
            cgrav: 0x3E71_EA8B_45ED_F98C,
            lnd: 0xC003_E319_C1AA_12B2,
            rho: 0x3FB5_4FF8_9FBA_652C,
            ln_t: 0x4030_C049_7CC3_206E,
            t: 0x4171_F5E6_A41B_D84F,
            ln_r: 0x403B_3D65_0D03_3FB6,
            r: 0x4263_AE6D_69F6_9C94,
            lum: 0x4851_F98B_2F48_4636,
            peos: 0x42FE_8954_4F58_C4E8,
            ln_pgas: 0x4040_831F_F5C3_6D8B,
            opacity: 0x3FD5_8514_0DC8_27B6,
            grada: 0x3FD0_5EA7_9C2B_950E,
            cp: 0x4200_E4A7_4D68_D174,
            chi_rho: 0x3FDA_3397_9FC2_4122,
            chi_t: 0x4006_2CAA_109B_213C,
            energy: 0x434B_63B7_93E6_226D,
            d_lnpeos_dlnd: 0x3FDA_3397_65C0_FC73,
            d_lnpeos_dlnt: 0x4006_2CAA_265B_8473,
        },
        expected_gradt: 0x3FD0_5EC6_85D4_1854,
        expected_mlt_vc: 0x40E1_22A9_8708_244B,
    },
    MltCaseHex {
        name: "1.3M_ms_high_Z",
        zone: 13,
        dt: 0x42C5_EF3E_D89C_D52F,
        alpha: 0x4000_0000_0000_0000,
        vc0: 0x40BC_C2BA_8995_23BC,
        km1: CellHex {
            dq: 0x3D9B_7CDF_D9D7_BDC0,
            dm: 0x448B_5E79_7788_EFC8,
            m: 0x46DF_DC9C_1CCD_C7A8,
            m_grav: 0x46DF_DC9C_1CCD_C7A8,
            cgrav: 0x3E71_EA8B_45ED_F98C,
            lnd: 0xC02F_799C_AC42_044A,
            rho: 0x3E83_A33F_5528_AD2B,
            ln_t: 0x4021_A2A0_C446_543D,
            t: 0x40BA_6043_0261_718C,
            ln_r: 0x4039_3A8D_C09F_E284,
            r: 0x4235_12D0_051B_F448,
            lum: 0x46F7_6BB9_11AC_FA6B,
            peos: 0x40EC_CF69_D425_FED6,
            ln_pgas: 0x4025_F873_FD0F_33CF,
            opacity: 0x3FF3_0CEE_DB3F_C5B8,
            grada: 0x3FD6_74D7_4DE0_AE12,
            cp: 0x41A4_9D58_1115_2925,
            chi_rho: 0x3FEF_F9EC_A253_B06C,
            chi_t: 0x3FF0_3E7E_352C_81A7,
            energy: 0x427C_EEDE_275B_D2F2,
            d_lnpeos_dlnd: 0x3FEF_F9EC_9E7B_8D7E,
            d_lnpeos_dlnt: 0x3FF0_3E7E_7E10_7A77,
        },
        k0: CellHex {
            dq: 0x3D9B_7CDF_D9D7_BDB8,
            dm: 0x448B_5E79_7788_EFC0,
            m: 0x46DF_DC9C_1CCC_ECB4,
            m_grav: 0x46DF_DC9C_1CCC_ECB4,
            cgrav: 0x3E71_EA8B_45ED_F98C,
            lnd: 0xC02F_69B8_664B_2008,
            rho: 0x3E84_41BB_756E_246E,
            ln_t: 0x4021_AE74_058F_4039,
            t: 0x40BA_FE07_175A_DD21,
            ln_r: 0x4039_3A8C_F9CB_E180,
            r: 0x4235_12BF_A721_C1C9,
            lum: 0x46F7_6BB9_11AD_0066,
            peos: 0x40EE_6C93_343C_E0EC,
            ln_pgas: 0x4026_145B_BEA1_BBB7,
            opacity: 0x3FF6_F394_182B_EED8,
            grada: 0x3FD5_CDD6_1DDC_E2E2,
            cp: 0x41A5_53B4_9401_6CFE,
            chi_rho: 0x3FEF_F87B_DA9B_D69E,
            chi_t: 0x3FF0_4EE3_4790_2F73,
            energy: 0x427D_310B_BE72_FFD2,
            d_lnpeos_dlnd: 0x3FEF_F87B_D6A3_25A8,
            d_lnpeos_dlnt: 0x3FF0_4EE3_EDA8_9539,
        },
        expected_gradt: 0x3FDB_1FE5_4E6C_78A6,
        expected_mlt_vc: 0x40BC_C2BA_89DE_F4E2,
    },
    MltCaseHex {
        name: "gyre_in_mesa_bcep_and_envelope",
        zone: 21,
        dt: 0x4225_AA87_1B30_E68A,
        alpha: 0x4000_0000_0000_0000,
        vc0: 0x403B_6D23_2908_7409,
        km1: CellHex {
            dq: 0x3D7B_212E_EA95_94DE,
            dm: 0x449F_2B0D_9EC8_2411,
            m: 0x4712_61BC_86C6_88F1,
            m_grav: 0x4712_61BC_86C6_88F1,
            cgrav: 0x3E71_EA8B_45ED_F98C,
            lnd: 0xC033_B057_AF91_A0D6,
            rho: 0x3E28_2AF1_790F_09F7,
            ln_t: 0x4024_DCC2_8592_F767,
            t: 0x40E0_8D81_663D_CC71,
            ln_r: 0x403A_7766_2BB0_C609,
            r: 0x4252_29C8_B754_99D6,
            lum: 0x47C0_0006_A2F7_31DB,
            peos: 0x40CE_5541_2E3B_25ED,
            ln_pgas: 0x4022_D17F_DA07_86EA,
            opacity: 0x4009_500F_42C0_41EF,
            grada: 0x3FD0_2C06_2D30_563A,
            cp: 0x41D4_DF77_D9C7_466A,
            chi_rho: 0x3FE9_0AB6_F9D3_D3CE,
            chi_t: 0x3FFB_377A_AE6B_1ADC,
            energy: 0x42B4_A775_2B1C_68FB,
            d_lnpeos_dlnd: 0x3FE9_0AB7_2116_DDCF,
            d_lnpeos_dlnt: 0x3FFB_372A_DF4B_47B1,
        },
        k0: CellHex {
            dq: 0x3D88_CC54_E26C_BC7E,
            dm: 0x44AC_7D5A_7322_F3BF,
            m: 0x4712_61BC_86C6_69C6,
            m_grav: 0x4712_61BC_86C6_69C6,
            cgrav: 0x3E71_EA8B_45ED_F98C,
            lnd: 0xC033_A877_DE7F_F362,
            rho: 0x3E28_EC34_36F4_A542,
            ln_t: 0x4024_E2B0_99B8_95AE,
            t: 0x40E0_BEDE_A227_3F72,
            ln_r: 0x403A_7763_ECF7_80A5,
            r: 0x4252_299F_F0AD_35E6,
            lum: 0x47C0_0006_A2F7_31E7,
            peos: 0x40CF_B110_6F35_14E8,
            ln_pgas: 0x4022_E794_EE1A_CA9E,
            opacity: 0x4009_CF0C_0398_212D,
            grada: 0x3FCF_C317_9CAA_F508,
            cp: 0x41D5_7238_4D04_D94E,
            chi_rho: 0x3FE9_025F_8C2B_B648,
            chi_t: 0x3FFB_62C9_3E31_AF6C,
            energy: 0x42B4_D619_D195_864B,
            d_lnpeos_dlnd: 0x3FE9_025F_B587_8CC9,
            d_lnpeos_dlnt: 0x3FFB_62A2_4639_7F73,
        },
        expected_gradt: 0x3FD0_D6F8_13AF_2AC0,
        expected_mlt_vc: 0x4040_C33B_95FA_9162,
    },
    MltCaseHex {
        name: "gyre_in_mesa_ms",
        zone: 15,
        dt: 0x42A4_DF59_28A4_7511,
        alpha: 0x4000_0000_0000_0000,
        vc0: 0x40F6_B212_DEE8_59D8,
        km1: CellHex {
            dq: 0x3D8B_7CDF_D9D7_BDB0,
            dm: 0x4475_0D98_8355_A4B5,
            m: 0x46D8_8250_B3B1_539F,
            m_grav: 0x46D8_8250_B3B1_539F,
            cgrav: 0x3E71_EA8B_45ED_F98C,
            lnd: 0xC02E_1AA1_C77B_2FF2,
            rho: 0x3E93_7CFD_3EF6_C733,
            ln_t: 0x4021_A8E0_D8B9_2FFA,
            t: 0x40BA_B332_355E_3835,
            ln_r: 0x4038_E368_38FB_582F,
            r: 0x422D_FC89_A869_0C56,
            lum: 0x46E2_E15F_B459_0BE1,
            peos: 0x40FE_E131_3B12_C447,
            ln_pgas: 0x4027_7EE2_4C9F_C73E,
            opacity: 0x3FFC_739B_53D2_A400,
            grada: 0x3FD7_052E_34C9_E108,
            cp: 0x41A5_65E1_8AFC_6418,
            chi_rho: 0x3FEF_FAC1_CD94_0C2E,
            chi_t: 0x3FF0_3381_3E57_66C8,
            energy: 0x427F_7A63_9F36_0D49,
            d_lnpeos_dlnd: 0x3FEF_FAC1_CD7C_70CC,
            d_lnpeos_dlnt: 0x3FF0_3387_22B4_C5E1,
        },
        k0: CellHex {
            dq: 0x3D94_9DA7_E361_CE48,
            dm: 0x447F_9464_C500_7715,
            m: 0x46D8_8250_B3B0_FF69,
            m_grav: 0x46D8_8250_B3B0_FF69,
            cgrav: 0x3E71_EA8B_45ED_F98C,
            lnd: 0xC02E_1415_8966_98A1,
            rho: 0x3E93_BD33_A46F_FFBF,
            ln_t: 0x4021_B51E_0033_173F,
            t: 0x40BB_588D_B9B8_5BF8,
            ln_r: 0x4038_E367_CDFF_E688,
            r: 0x422D_FC7D_2066_BFEA,
            lum: 0x46E2_E15F_B459_0C3B,
            peos: 0x4100_05AC_1089_7E39,
            ln_pgas: 0x4027_91D6_8459_0073,
            opacity: 0x4001_3FFC_D380_9CBF,
            grada: 0x3FD6_7140_7CFD_3FCC,
            cp: 0x41A6_07DE_B444_C0A4,
            chi_rho: 0x3FEF_F99F_49FA_0BF8,
            chi_t: 0x3FF0_4106_816C_F0C3,
            energy: 0x427F_C1D6_4821_607C,
            d_lnpeos_dlnd: 0x3FEF_F99F_4A03_B5F4,
            d_lnpeos_dlnt: 0x3FF0_40FF_C1D6_6B41,
        },
        expected_gradt: 0x3FE4_A2B8_94B3_EAB2,
        expected_mlt_vc: 0x40F8_64BA_D280_BFC5,
    },
    MltCaseHex {
        name: "gyre_in_mesa_spb",
        zone: 76,
        dt: 0x4234_97FB_EF09_52EC,
        alpha: 0x4000_0000_0000_0000,
        vc0: 0x4081_FBCD_4A6A_447B,
        km1: CellHex {
            dq: 0x3DAD_8724_7702_BE3C,
            dm: 0x44BC_4501_8A5A_3D7D,
            m: 0x46FE_A2E4_E07A_3159,
            m_grav: 0x46FE_A2E4_E07A_3159,
            cgrav: 0x3E71_EA8B_45ED_F98C,
            lnd: 0xC032_530D_7DA5_543D,
            rho: 0x3E47_A509_84B7_A9CE,
            ln_t: 0x4024_F169_9DE7_6921,
            t: 0x40E1_3BED_EC5A_A3F2,
            ln_r: 0x403A_3320_3557_F019,
            r: 0x424B_D29E_52CE_2CF6,
            lum: 0x477E_7F91_A9CF_6A69,
            peos: 0x40EA_22AE_C8F3_9AFC,
            ln_pgas: 0x4025_9FBB_DFDF_0536,
            opacity: 0x4021_756E_2BF9_D517,
            grada: 0x3FD2_44A6_9969_1900,
            cp: 0x41C3_9D89_2381_09EC,
            chi_rho: 0x3FED_9544_6CBB_6808,
            chi_t: 0x3FF4_2D24_FDDB_D835,
            energy: 0x42B2_80B5_469A_2008,
            d_lnpeos_dlnd: 0x3FED_9544_7DC7_E477,
            d_lnpeos_dlnt: 0x3FF4_2CFC_92B1_5C88,
        },
        k0: CellHex {
            dq: 0x3DB6_255B_5942_0EAD,
            dm: 0x44C5_33C1_27C3_AE1E,
            m: 0x46FE_A2E4_E078_6D09,
            m_grav: 0x46FE_A2E4_E078_6D09,
            cgrav: 0x3E71_EA8B_45ED_F98C,
            lnd: 0xC032_4AAE_17F0_D4C6,
            rho: 0x3E48_6E47_52C2_2524,
            ln_t: 0x4024_F91A_7F3C_C9BB,
            t: 0x40E1_7EB3_ED87_CF0B,
            ln_r: 0x403A_331B_9367_717D,
            r: 0x424B_D21D_6FE6_A337,
            lum: 0x477E_7F91_A9CF_6CC5,
            peos: 0x40EB_7462_C324_3357,
            ln_pgas: 0x4025_B881_37A1_1FFA,
            opacity: 0x4021_DE2F_6BD9_B68F,
            grada: 0x3FD1_B191_73A8_E5DD,
            cp: 0x41C4_7F17_C817_E793,
            chi_rho: 0x3FED_8A9F_0504_32B0,
            chi_t: 0x3FF4_5C52_15A4_92B4,
            energy: 0x42B2_ACA2_F67A_A3D5,
            d_lnpeos_dlnd: 0x3FED_8A9F_13F3_7BD1,
            d_lnpeos_dlnt: 0x3FF4_5C5B_55EE_555D,
        },
        expected_gradt: 0x3FD3_75D8_8CE4_B374,
        expected_mlt_vc: 0x4087_A6A9_C08F_4592,
    },
    MltCaseHex {
        name: "hse_riemann",
        zone: 37,
        dt: 0x4009_3EFF_1950_331E,
        alpha: 0x3FFE_6666_6666_6666,
        vc0: 0x404A_CE51_4E17_D7CD,
        km1: CellHex {
            dq: 0x3E27_7A71_4039_8BC2,
            dm: 0x44C4_C75E_69DA_362F,
            m: 0x4712_61BC_86B7_3683,
            m_grav: 0x4712_61BC_86B7_3683,
            cgrav: 0x3E71_EA8B_45ED_F98C,
            lnd: 0xC034_42A9_EC4C_93BF,
            rho: 0x3E1B_4AC9_D02B_40E1,
            ln_t: 0x4024_BDAB_D6CF_78BD,
            t: 0x40DF_27B8_7F20_26B5,
            ln_r: 0x403B_1232_B24A_4AE3,
            r: 0x4260_A01D_97AB_59D4,
            lum: 0x47CF_AC6F_F91C_DB7C,
            peos: 0x40C1_BDA6_46E5_4645,
            ln_pgas: 0x4021_8CD4_D2E1_CB36,
            opacity: 0x4001_CA2A_FEB5_CF6D,
            grada: 0x3FD0_AE50_FC6D_99A3,
            cp: 0x41DB_4D32_DB6E_67C4,
            chi_rho: 0x3FE6_BF24_1510_5ED6,
            chi_t: 0x3FFE_4DDC_564C_4157,
            energy: 0x42B5_6A01_C31E_862D,
            d_lnpeos_dlnd: 0x3FE6_BF24_2B2C_93CE,
            d_lnpeos_dlnt: 0x3FFE_4DAC_7238_0A6E,
        },
        k0: CellHex {
            dq: 0x3E37_7A71_4039_8BC2,
            dm: 0x44D4_C75E_69DA_362F,
            m: 0x4712_61BC_86B6_9048,
            m_grav: 0x4712_61BC_86B6_9048,
            cgrav: 0x3E71_EA8B_45ED_F98C,
            lnd: 0xC034_3C1A_70C0_103A,
            rho: 0x3E1C_0026_DD9F_6AF7,
            ln_t: 0x4024_C2F8_F469_D78D,
            t: 0x40DF_7ABA_E57C_8E07,
            ln_r: 0x403B_122F_3D72_5044,
            r: 0x4260_9FE4_2121_B9AE,
            lum: 0x47CF_AC6F_F91C_DB7B,
            peos: 0x40C2_6CDB_6145_7C81,
            ln_pgas: 0x4021_9F79_2BBD_CCFF,
            opacity: 0x4002_0CD5_FC35_1547,
            grada: 0x3FD0_830C_DA52_56B0,
            cp: 0x41DB_CDBA_6301_FAE5,
            chi_rho: 0x3FE6_B498_B8FB_57BB,
            chi_t: 0x3FFE_6FDC_EF4B_1855,
            energy: 0x42B5_93BD_7063_8746,
            d_lnpeos_dlnd: 0x3FE6_B498_ED18_3A12,
            d_lnpeos_dlnt: 0x3FFE_6FAA_36BA_289D,
        },
        expected_gradt: 0x3FD1_6D86_4832_C0B0,
        expected_mlt_vc: 0x404A_CE51_4211_4E12,
    },
];

#[test]
fn test_set_mlt_vars_matches_mesa_aux_hex_zero_ulp_across_tdc_suite() {
    for case in CASES {
        let mut model = build_two_zone_mlt_model(case);
        set_mlt_vars(&mut model).expect("set_mlt_vars succeeds");
        assert_hex_eq(
            &format!("{} zone {} gradT", case.name, case.zone),
            model.gradt[1],
            case.expected_gradt,
        );
        assert_hex_eq(
            &format!("{} zone {} mlt_vc", case.name, case.zone),
            model.mlt_vc[1],
            case.expected_mlt_vc,
        );
    }
}

#[test]
fn test_do1_mlt_2_uses_model_peos_directly_for_rcrb_and_high_mass_zero_ulp() {
    for case_name in ["R_CrB_star", "high_mass"] {
        let case = CASES
            .iter()
            .find(|c| c.name == case_name)
            .expect("case present");
        let mut model = build_two_zone_mlt_model(case);
        set_mlt_vars(&mut model).expect("set_mlt_vars with model.peos");
        assert_hex_eq(
            &format!("{case_name} gradT"),
            model.gradt[1],
            case.expected_gradt,
        );
        assert_hex_eq(
            &format!("{case_name} mlt_vc"),
            model.mlt_vc[1],
            case.expected_mlt_vc,
        );

        // Clearing `model.peos` exercises the fallback `(CRAD/3)*T^4 + exp(ln_pgas)`
        // ordering, confirming the pre-fix 1-ULP pressure difference shifts these faces.
        let mut fallback = build_two_zone_mlt_model(case);
        fallback.peos.fill(0.0);
        set_mlt_vars(&mut fallback).expect("fallback pressure order runs");
        assert_ne!(
            fallback.mlt_vc[1].to_bits(),
            case.expected_mlt_vc,
            "{case_name} fallback pressure must reproduce the pre-fix divergence"
        );
    }
}

#[test]
fn test_apply_uniform_rotation_uses_roche_j_rot_and_matches_conserve_j_surface_zero_ulp(
) {
    let mut controls = StarControls {
        rotation_flag: true,
        ..StarControls::default()
    };
    controls.mix_wind.mlt_use_rotation_correction = true;
    let mut model =
        StarModel::new(2, vec![1], StarJobControls::default(), controls);
    let alpha = model.controls.mixing_length_alpha;
    model.aux.mixing.gradr_factor.fill(1.0);
    model.aux.mixing.alpha_mlt.fill(alpha);

    // Surface cell 1 (idx 0) from `standard_zams.mod` / `conserve_angular_momentum`
    // step 1 (`new_surface_rotation_v = 10 km/s`).
    let lnr0 = f64::from_bits(0x4038_E370_2474_2048);
    let r0 = flexmesa_math::exp(lnr0);
    let omega0 = (10.0 * 1.0e5) / r0;
    let dm0 = f64::from_bits(0x445A_F2AE_BC96_9563);
    let dm1 = 2.5e-12 * M_SUN;

    model.xmstar = M_SUN;
    model.mstar = M_SUN;
    model.dq = vec![dm0 / M_SUN, dm1 / M_SUN];
    model.dm = vec![dm0, dm1];
    model.m = vec![M_SUN, M_SUN - dm0];
    model.m_grav = model.m.clone();
    model.cgrav.fill(STANDARD_CGRAV);
    model.lnd = vec![
        f64::from_bits(0xC02F_08A0_A56D_AB4C),
        -15.489_084_669_591_536_f64,
    ];
    model.rho = vec![
        flexmesa_math::exp(model.lnd[0]),
        flexmesa_math::exp(model.lnd[1]),
    ];
    model.ln_t = vec![
        f64::from_bits(0x4021_4A26_BBFD_D625),
        8.648_920_816_857_114_f64,
    ];
    model.t = vec![
        flexmesa_math::exp(model.ln_t[0]),
        flexmesa_math::exp(model.ln_t[1]),
    ];
    model.ln_r = vec![lnr0, 24.888_426_660_365_61_f64];
    model.r = vec![r0, flexmesa_math::exp(model.ln_r[1])];
    model.lum = vec![
        f64::from_bits(0x46E2_E15F_B459_0F3E),
        3.063_514_628_275_467_5e33_f64,
    ];
    model.peos[0] = f64::from_bits(0x40F0_1B24_BF9F_05EA);
    model.peos[1] = model.peos[0];
    model.opacity[0] = f64::from_bits(0x3FD4_E323_8D78_26F5);
    model.opacity[1] = model.opacity[0];
    model.grada[0] = f64::from_bits(0x3FD8_F9B8_B0EC_BF6C);
    model.grada[1] = model.grada[0];
    model.chi_rho[0] = f64::from_bits(0x3FEF_FC7A_0F2E_AA31);
    model.chi_rho[1] = model.chi_rho[0];
    model.chi_t[0] = f64::from_bits(0x3FF0_1255_55E4_EF21);
    model.chi_t[1] = model.chi_t[0];
    model.cp[0] = f64::from_bits(0x41A3_8A56_62F7_4C68);
    model.cp[1] = model.cp[0];

    apply_uniform_rotation(&mut model, omega0);

    // Must use Roche shell moment of inertia (`use_xh_to_update_i_rot_and_j_rot`),
    // not spherical `(2/3) r^2 omega`.
    let naive_j_rot = (2.0 / 3.0) * r0 * r0 * omega0;
    assert_ne!(
        model.j_rot[0].to_bits(),
        naive_j_rot.to_bits(),
        "apply_uniform_rotation must not use naive spherical (2/3) r^2 omega"
    );
    assert!(model.aux.w_div_wc[0] > 0.0);
    assert!(model.aux.fp_rot[0] < 1.0 && model.aux.ft_rot[0] < 1.0);

    // `set_mlt_vars` calls `set_rotation_info(model, true)` (`hydro_vars.f90:541`)
    // and reproduces MESA zone 1 `gradr` / `gradT` bit-for-bit (`0 ULP` vs pre-fix 4.52e8 ULP).
    let expected_w_div_wc = model.aux.w_div_wc[0];
    set_mlt_vars(&mut model).expect("set_mlt_vars rotating surface");
    assert_eq!(
        model.aux.w_div_wc[0].to_bits(),
        expected_w_div_wc.to_bits(),
        "set_mlt_vars must pass skip_w_div_w_crit_roche = true"
    );
    assert_hex_eq(
        "conserve_angular_momentum zone 1 gradr",
        model.gradr[0],
        0x3FC0_147B_6223_FABC,
    );
    assert_hex_eq(
        "conserve_angular_momentum zone 1 gradT",
        model.gradt[0],
        0x3FC0_147B_6223_FABC,
    );
    assert_hex_eq(
        "conserve_angular_momentum zone 1 mlt_vc",
        model.mlt_vc[0],
        0x0000_0000_0000_0000,
    );
}

#[test]
fn test_read_mod_preserves_timestep_separately_and_guards_mlt_vc_for_rsg_step1()
{
    let mod_with_vc = concat!(
        "            36 -- test model with mlt_vc\n",
        "                  version_number   'r24.08.1'\n",
        "                          M/Msun      2.0999999999999996D+01\n",
        "                    model_number                           1\n",
        "                        star_age      1.0000000000000000D+06\n",
        "                       initial_z      2.0000000000000000D-02\n",
        "                        n_shells                           2\n",
        "                        net_name   'basic.net'\n",
        "                         species                           8\n",
        "               lnd                        lnT                        lnR                          L                         dq                     mlt_vc                     h1                        he3                        he4                        c12                        n14                        o16                        ne20                       mg24\n",
        "    1    -1.9676359044644823D+01     8.1851939348148779D+00     3.1947151226445214D+01     4.2714250882829565D+38     9.9996870129703217D-01     1.5477056941954044D+02     6.8809754269901235D-01     1.8850380503921046D-05     2.9180405246109814D-01     2.3189765162550894D-03     3.3394288965240123D-03     7.5986643013020482D-03     1.8669682735524757D-03     4.9555164717520594D-03\n",
        "    2    -1.9668046969873949D+01     8.1918919874162235D+00     3.1947053408994169D+01     4.2714252694795343D+38     3.1298702967826769D-05     2.2830591666226462D+02     6.8809754212691010D-01     1.8850380449969011D-05     2.9180405303398393D-01     2.3189765097329307D-03     3.3394289168501299D-03     7.5986642867684360D-03     1.8669682735524757D-03     4.9555164717520594D-03\n",
        "\n",
        "        previous model\n",
        "               previous n_shells                           2\n",
        "           previous mass (grams)      4.1756607284659057D+34\n",
        "              timestep (seconds)      2.3422496115017467D+09\n",
        "               dt_next (seconds)      2.8106995338020959D+09\n",
    );
    let controls = StarControls::default();
    let job = StarJobControls::default();
    let mut loaded = read_mod_str(mod_with_vc, controls.clone(), job.clone())
        .expect("read synthetic RSG .mod");

    assert_hex_eq("loaded timestep dt", loaded.dt, 0x41E1_737C_9170_0E4F);
    assert_hex_eq("loaded dt_next", loaded.dt_next, 0x41E4_F0FB_E1B9_AAC5);
    assert!(loaded.aux.have_mlt_vc);
    assert!(
        !loaded.aux.okay_to_set_mlt_vc,
        "okay_to_set_mlt_vc must start false when .mod supplies mlt_vc"
    );
    let saved_vc1_bits = 0x406C_89CA_11BD_774C_u64;
    assert_hex_eq("saved .mod mlt_vc", loaded.mlt_vc[1], saved_vc1_bits);

    // Running startup `set_hydro_vars` + `save_old_values` must NOT step TDC
    // onto `mlt_vc` or `mlt_vc_old` (`turb_info.f90:266`).
    set_hydro_vars(&mut loaded, false, false).expect("startup set_hydro_vars");
    loaded.save_old_values();
    assert_hex_eq("startup mlt_vc preserved", loaded.mlt_vc[1], saved_vc1_bits);
    assert_hex_eq(
        "startup mlt_vc_old preserved",
        loaded.aux.mlt_vc_old[1],
        saved_vc1_bits,
    );

    // Without saved `mlt_vc`, `read_mod_str` sets `okay_to_set_mlt_vc = true` (`read_model.f90:157`).
    let mod_no_vc = mod_with_vc
        .replace("mlt_vc", "")
        .replace("     1.5477056941954044D+02", "")
        .replace("     2.2830591666226462D+02", "");
    let no_vc = read_mod_str(&mod_no_vc, controls, job)
        .expect("read .mod without mlt_vc");
    assert!(!no_vc.aux.have_mlt_vc);
    assert!(no_vc.aux.okay_to_set_mlt_vc);
}

#[test]
fn test_set_mlt_vars_uses_face_energy_and_lnd_start_per_upstream() {
    let base_case = CASES
        .iter()
        .find(|c| c.name == "gyre_in_mesa_rsg")
        .expect("rsg fixture");

    // 1. Face specific energy `get_e_face = alfa * energy[k] + beta * energy[k-1]`
    // (`star_utils.f90:3872-3883`) vs incorrect `(alfa * ln_e[k] + beta * ln_e[k-1]).exp()`.
    let ms_case = CASES
        .iter()
        .find(|c| c.name == "gyre_in_mesa_ms")
        .expect("ms fixture");
    let mut e_model = build_two_zone_mlt_model(ms_case);
    e_model.controls.rsp2.use_tdc_enthalpy_flux_limiter = true;
    e_model.lum[1] *= 20.0;
    e_model.energy[0] = 1.0e11;
    e_model.energy[1] = 2.0e12;
    e_model.ln_e[0] = flexmesa_math::log(e_model.energy[0]);
    e_model.ln_e[1] = flexmesa_math::log(e_model.energy[1]);
    set_mlt_vars(&mut e_model)
        .expect("set_mlt_vars with enthalpy flux limiter");

    let mut fallback_ln_e = e_model.clone();
    fallback_ln_e.energy = vec![0.0, 0.0];
    set_mlt_vars(&mut fallback_ln_e).expect("fallback ln_e interpolation");
    assert_ne!(
        e_model.mlt_vc[1].to_bits(),
        fallback_ln_e.mlt_vc[1].to_bits(),
        "get_e_face linear energy interpolation must differ from exp(interp(ln_e))"
    );

    // 2. TDC compression work `dV = 1/rho - 1/rho_start` uses `lnd_start`, not `lnd_old`
    // (`turb_support.f90:167-168`).
    let mut dv_start_differs = build_two_zone_mlt_model(base_case);
    dv_start_differs.controls.tdc_alpha_pt = 1.0;
    dv_start_differs.lnd_start[0] -= 0.05;
    dv_start_differs.lnd_start[1] -= 0.05;
    // Keep `lnd_old == lnd` so only `lnd_start` drives `dv_f`.
    set_mlt_vars(&mut dv_start_differs).expect("set_mlt_vars with lnd_start");

    let mut dv_ref = dv_start_differs.clone();
    dv_ref.lnd_old.clone_from(&dv_ref.lnd_start);
    set_mlt_vars(&mut dv_ref)
        .expect("set_mlt_vars matching lnd_old = lnd_start");

    assert_hex_eq(
        "dv_f ignores lnd_old when lnd_start populated (gradT)",
        dv_start_differs.gradt[1],
        dv_ref.gradt[1].to_bits(),
    );
    assert_hex_eq(
        "dv_f ignores lnd_old when lnd_start populated (mlt_vc)",
        dv_start_differs.mlt_vc[1],
        dv_ref.mlt_vc[1].to_bits(),
    );

    let mut dv_zero = build_two_zone_mlt_model(base_case);
    dv_zero.controls.tdc_alpha_pt = 1.0;
    set_mlt_vars(&mut dv_zero).expect("set_mlt_vars with zero dV");
    assert_ne!(
        dv_start_differs.mlt_vc[1].to_bits(),
        dv_zero.mlt_vc[1].to_bits(),
        "nonzero lnd_start delta must enter TDC through dv_f"
    );
}
