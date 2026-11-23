// SPDX-License-Identifier: LGPL-3.0-or-later
//! Issue `#395` deterministic `0-ULP` IEEE-754 hex parity tests for
//! [`change_model_network`] against MESA `26.4.1` `adjust_xyz.f90:51-629` and
//! `net_eval.f90:822-850`.

#![forbid(unsafe_code)]

use flexmesa_star::micro::{change_model_network, get_or_load_net};
use flexmesa_star::model::StarModel;
use flexmesa_star::{StarControls, StarJobControls};

type ZoneInput<'a> = (f64, f64, &'a [(&'a str, f64)]);

fn build_test_model(
    net_name: &str,
    initial_z: f64,
    zones: &[ZoneInput<'_>],
) -> StarModel {
    let net_info = get_or_load_net(net_name, None).expect("load test net");
    let job = StarJobControls {
        net_name: net_name.to_string(),
        ..StarJobControls::default()
    };
    let controls = StarControls {
        initial_z,
        zbase: initial_z,
        ..StarControls::default()
    };
    let mut model =
        StarModel::new(zones.len(), net_info.chem_id.clone(), job, controls);
    let sp = model.species;
    for (k, &(ln_t, mlt_d, abuns)) in zones.iter().enumerate() {
        model.ln_t[k] = ln_t;
        model.ln_t_start[k] = ln_t;
        model.ln_t_old[k] = ln_t;
        model.mlt_d[k] = mlt_d;
        for &(iso, val) in abuns {
            let j = model
                .iso_names
                .iter()
                .position(|name| name == iso)
                .unwrap_or_else(|| panic!("missing iso {iso} in {net_name}"));
            model.xa[k * sp + j] = val;
        }
    }
    model.xa_start.clone_from(&model.xa);
    model
}

fn assert_zone_bits(model: &StarModel, zone: usize, expected: &[(&str, u64)]) {
    let sp = model.species;
    assert_eq!(sp, expected.len(), "species count mismatch");
    for (j, &(iso, exp_bits)) in expected.iter().enumerate() {
        assert_eq!(model.iso_names[j], iso, "iso slot {j}");
        let got = model.xa[zone * sp + j];
        let got_bits = got.to_bits();
        assert_eq!(
            got_bits,
            exp_bits,
            "zone {zone} iso {iso}: got {got:.16e} (0x{got_bits:016X}), \
             expected {exp_val:.16e} (0x{exp_bits:016X})",
            exp_val = f64::from_bits(exp_bits),
        );
    }
}

#[test]
fn test_change_model_network_same_net_is_exact_zero_ulp_noop() {
    // `star/private/adjust_xyz.f90:93`: `if (new_net_name == s% net_name) return`.
    // Note: row sums slightly below `1.0` must NOT dump deficit into the
    // heaviest isotope or renormalize existing fractions.
    let pp_cno_zone: &[(&str, f64)] = &[
        ("h1", f64::from_bits(0x3FE6_6666_665F_5E42)),
        ("h2", f64::from_bits(0x3C84_4EBB_160A_D2EF)),
        ("he3", f64::from_bits(0x3EFF_3EC0_F690_AEF2)),
        ("he4", f64::from_bits(0x3FD1_EB08_23CF_46E9)),
        ("li7", f64::from_bits(0x3E4D_FB38_1A87_795D)),
        ("be7", f64::from_bits(0x36CB_4D21_439F_783F)),
        ("b8", f64::from_bits(0x2B61_7FC4_7495_EFD6)),
        ("c12", f64::from_bits(0x3F6C_AE78_94CD_E081)),
        ("c13", f64::from_bits(0x3F06_4836_6E49_B061)),
        ("n13", f64::from_bits(0x2B65_A820_7CD5_72D0)),
        ("n14", f64::from_bits(0x3F50_F393_C85D_E7C1)),
        ("n15", f64::from_bits(0x3ED1_171C_6D07_8994)),
        ("o14", f64::from_bits(0x2B61_8655_E2D2_0087)),
        ("o15", f64::from_bits(0x2B61_871A_DD0D_C912)),
        ("o16", f64::from_bits(0x3F81_8E68_5101_C5F1)),
        ("o17", f64::from_bits(0x3ECC_7F70_C02C_C436)),
        ("o18", f64::from_bits(0x3EF4_4850_69EA_13B1)),
        ("f17", f64::from_bits(0x2B61_8635_C13D_F93A)),
        ("f18", f64::from_bits(0x2B74_5174_3A6E_3C43)),
        ("f19", f64::from_bits(0x3EA9_64F8_0BEA_D3AB)),
        ("ne18", f64::from_bits(0x2B61_7F9D_CD13_AAF2)),
        ("ne19", f64::from_bits(0x2B61_8392_718D_511E)),
        ("ne20", f64::from_bits(0x3F5E_96A1_EB5D_6CD9)),
        ("mg22", f64::from_bits(0x386A_DAF3_13BE_340B)),
        ("mg24", f64::from_bits(0x3F74_4B6D_688E_1B11)),
    ];
    let ln_t = f64::from_bits(0x4020_E86D_3111_8A79);
    let mut model = build_test_model(
        "pp_and_cno_extras.net",
        0.02,
        &[(ln_t, 0.0, pp_cno_zone)],
    );
    let before: Vec<u64> = model.xa.iter().map(|v| v.to_bits()).collect();
    change_model_network(&mut model, "pp_and_cno_extras.net", None)
        .expect("same-net noop");
    let after: Vec<u64> = model.xa.iter().map(|v| v.to_bits()).collect();
    assert_eq!(after, before);

    // Zero `ni56` with sub-unity row sum in `approx21_cr60_plus_co56.net`
    // must remain exact 0.0 rather than absorbing `1.0 - sum`.
    let approx21_zone: &[(&str, f64)] = &[
        ("neut", f64::from_bits(0x2B61_7F7D_4ED8_D638)),
        ("h1", f64::from_bits(0x3FE6_E53C_95D4_80E2)),
        ("prot", f64::from_bits(0x2B61_7F7D_4ED8_D638)),
        ("he3", f64::from_bits(0x3F01_1962_A16B_6157)),
        ("he4", f64::from_bits(0x3FD1_4C32_F7DC_F5A9)),
        ("c12", f64::from_bits(0x3F64_3078_412F_64A7)),
        ("n14", f64::from_bits(0x3F4A_2253_3F46_51DC)),
        ("o16", f64::from_bits(0x3F78_FEFA_D3DF_783F)),
        ("ne20", f64::from_bits(0x3F55_B7BE_F31E_AF36)),
        ("mg24", f64::from_bits(0x3F6C_D2A6_802F_5908)),
        ("si28", f64::from_bits(0x2B61_7F7D_4ED8_D638)),
        ("s32", f64::from_bits(0x2B61_7F7D_4ED8_D638)),
        ("ar36", f64::from_bits(0x2B61_7F7D_4ED8_D638)),
        ("ca40", f64::from_bits(0x2B61_7F7D_4ED8_D638)),
        ("ti44", f64::from_bits(0x2B61_7F7D_4ED8_D638)),
        ("cr48", f64::from_bits(0x2B61_7F7D_4ED8_D638)),
        ("cr60", f64::from_bits(0x2B61_7F7D_4ED8_D638)),
        ("fe52", f64::from_bits(0x2B61_7F7D_4ED8_D638)),
        ("fe54", f64::from_bits(0x2B61_7F7D_4ED8_D638)),
        ("fe56", f64::from_bits(0x2C18_C485_7EF1_46A8)),
        ("co56", f64::from_bits(0x2B61_87FD_C271_21CE)),
        ("ni56", 0.0),
    ];
    let mut m2 = build_test_model(
        "approx21_cr60_plus_co56.net",
        0.0142,
        &[(f64::from_bits(0x4020_1612_F550_60B3), 0.0, approx21_zone)],
    );
    let before2: Vec<u64> = m2.xa.iter().map(|v| v.to_bits()).collect();
    change_model_network(&mut m2, "approx21_cr60_plus_co56.net", None)
        .expect("approx21 noop");
    let after2: Vec<u64> = m2.xa.iter().map(|v| v.to_bits()).collect();
    assert_eq!(after2, before2);
    assert_eq!(m2.xa[21].to_bits(), 0);
}

#[test]
fn test_change_model_network_basic_to_o18_ne22_high_z_exact_hex_parity() {
    // High-Z (`initial_z = 0.07`) transition `basic.net -> o18_and_ne22.net`
    // matches MESA `26.4.1` `set_new_abundances` + `do_clean1` at 0 ULP on the
    // un-remeshed core zone 875 (`standard_zams.mod` -> `high_z.stepdump`).
    let zone0: &[(&str, f64)] = &[
        ("h1", f64::from_bits(0x3FE1_985C_FEC4_EDD9)),
        ("he3", f64::from_bits(0x3EF9_459A_57A6_352C)),
        ("he4", f64::from_bits(0x3FD8_5179_8A3E_D49E)),
        ("c12", f64::from_bits(0x3F86_C6AC_B494_1B32)),
        ("n14", f64::from_bits(0x3F72_DF39_3D62_FF31)),
        ("o16", f64::from_bits(0x3FA0_C615_3865_760C)),
        ("ne20", f64::from_bits(0x3F7E_190D_B53F_21A1)),
        ("mg24", f64::from_bits(0x3F8D_51C7_2A43_1E9D)),
    ];
    let mut model = build_test_model(
        "basic.net",
        0.07,
        &[(f64::from_bits(0x4030_FF44_91C8_8E0D), 0.0, zone0)],
    );
    change_model_network(&mut model, "o18_and_ne22.net", None)
        .expect("basic -> o18_and_ne22");

    assert_zone_bits(
        &model,
        0,
        &[
            ("h1", 0x3FE1_985C_FEC4_EDDB),
            ("he3", 0x3EF9_459A_57A6_352F),
            ("he4", 0x3FD8_5179_8A3E_D4A1),
            ("c12", 0x3F86_C6AC_B494_1B33),
            ("n14", 0x3F72_DF39_3D62_FF33),
            ("o16", 0x3FA0_BBA2_AC71_6B5C),
            ("o18", 0x3F14_E517_E815_643A),
            ("ne20", 0x3F7C_4085_9C67_3909),
            ("ne22", 0x3F3D_8881_8D7E_89A8),
            ("mg24", 0x3F8D_51C7_2A43_1EA0),
        ],
    );
}

#[test]
fn test_change_model_network_co_burn_extras_to_approx21_exact_hex_parity() {
    // Tests `ne20 + ne22 -> ne20` element conservation and `1e-99` `clean1`
    // flooring for missing trace species (`co_burn_extras.net -> approx21.net`).
    let zone0: &[(&str, f64)] = &[
        ("h1", 0.0),
        ("he3", f64::from_bits(0x3B36_4CC1_CEA0_4C93)),
        ("he4", f64::from_bits(0x3D69_45C2_6693_FA2B)),
        ("c12", f64::from_bits(0x3FE1_B748_BAD2_09FD)),
        ("n14", f64::from_bits(0x3BF6_DEA8_BAFD_C705)),
        ("o16", f64::from_bits(0x3FDA_ED13_3C2B_1AB0)),
        ("o18", f64::from_bits(0x3BC4_CFD9_AEDA_13E8)),
        ("ne20", f64::from_bits(0x3F60_BBE0_6C98_2552)),
        ("ne22", f64::from_bits(0x3F93_FC60_4717_3E23)),
        ("mg24", f64::from_bits(0x3F4A_661F_3FF8_C368)),
        ("si28", f64::from_bits(0x3F6A_F53C_A2FF_1E73)),
    ];
    let mut model = build_test_model(
        "co_burn_extras.net",
        0.02,
        &[(f64::from_bits(0x4026_5228_CA59_29DF), 0.0, zone0)],
    );
    change_model_network(&mut model, "approx21.net", None)
        .expect("co_burn_extras -> approx21");

    assert_zone_bits(
        &model,
        0,
        &[
            ("neut", 0x2B61_7F7D_4ED8_C33E),
            ("h1", 0x2B61_7F7D_4ED8_C33E),
            ("prot", 0x2B61_7F7D_4ED8_C33E),
            ("he3", 0x3B36_4CC1_CEA0_4C92),
            ("he4", 0x3D69_45C2_6693_FA2A),
            ("c12", 0x3FE1_B748_BAD2_09FD),
            ("n14", 0x3BF6_DEA8_BAFD_C706),
            ("o16", 0x3FDA_ED13_3C2B_1AB0),
            ("ne20", 0x3F96_13DC_54AA_42CD),
            ("mg24", 0x3F4A_661F_3FF8_C369),
            ("si28", 0x3F6A_F53C_A2FF_1E73),
            ("s32", 0x2B61_7F7D_4ED8_C33E),
            ("ar36", 0x2B61_7F7D_4ED8_C33E),
            ("ca40", 0x2B61_7F7D_4ED8_C33E),
            ("ti44", 0x2B61_7F7D_4ED8_C33E),
            ("cr48", 0x2B61_7F7D_4ED8_C33E),
            ("cr56", 0x2B61_7F7D_4ED8_C33E),
            ("fe52", 0x2B61_7F7D_4ED8_C33E),
            ("fe54", 0x2B61_7F7D_4ED8_C33E),
            ("fe56", 0x2B61_7F7D_4ED8_C33E),
            ("ni56", 0x2B61_7F7D_4ED8_C33E),
        ],
    );
}

#[test]
fn test_change_model_network_leftover_si_dumped_into_last_species_exact_hex() {
    // `co_burn_plus.net -> cno_extras_o18_to_mg26_plus_fe56.net`:
    // `mg24 + mg26` share `old_total_mg`, removed `si28` dumps into `fe56`
    // (final isotope), and absent unstable species receive `1e-99` (`clean1`).
    let zone0: &[(&str, f64)] = &[
        ("h1", f64::from_bits(0x36A0_EA17_6E12_6F55)),
        ("he3", f64::from_bits(0x3775_573B_C140_A929)),
        ("he4", f64::from_bits(0x3FEF_FFFF_FFFF_FFAD)),
        ("c12", f64::from_bits(0x3CF8_9504_3971_9D4A)),
        ("c13", f64::from_bits(0x3C24_768F_50EA_08C1)),
        ("n13", f64::from_bits(0x2D4D_62F2_6961_92DE)),
        ("n14", f64::from_bits(0x371A_7547_1556_3F1D)),
        ("n15", f64::from_bits(0x3BB0_5C01_EFE2_2811)),
        ("o16", f64::from_bits(0x3CEC_20BB_E518_5511)),
        ("o17", f64::from_bits(0x3BD1_2952_68BA_65FC)),
        ("o18", f64::from_bits(0x37FB_3FAC_A631_3237)),
        ("f19", f64::from_bits(0x3B26_6591_B1C1_AD3D)),
        ("ne20", f64::from_bits(0x3C8E_4213_2F47_7B9F)),
        ("ne22", f64::from_bits(0x3CC4_1E32_2961_8024)),
        ("mg24", f64::from_bits(0x3C9D_64FB_BEE8_97F7)),
        ("si28", f64::from_bits(0x3C73_C576_1D41_3FAF)),
    ];
    let mut model = build_test_model(
        "co_burn_plus.net",
        0.02,
        &[(f64::from_bits(0x4025_9FE5_66B9_BDB6), 0.0, zone0)],
    );
    change_model_network(
        &mut model,
        "cno_extras_o18_to_mg26_plus_fe56.net",
        None,
    )
    .expect("co_burn_plus -> cno_extras_o18_to_mg26_plus_fe56");

    assert_zone_bits(
        &model,
        0,
        &[
            ("h1", 0x36A0_EA17_6E12_6F55),
            ("he3", 0x3775_573B_C140_A929),
            ("he4", 0x3FEF_FFFF_FFFF_FFAD),
            ("c12", 0x3CF8_9504_3971_9D4A),
            ("c13", 0x3C24_768F_50EA_08C1),
            ("n13", 0x2D4D_62F2_6961_92DE),
            ("n14", 0x371A_7547_1556_3F1D),
            ("n15", 0x3BB0_5C01_EFE2_2811),
            ("o14", 0x2B61_7F7D_4ED8_C33E),
            ("o15", 0x2B61_7F7D_4ED8_C33E),
            ("o16", 0x3CEC_20BB_E518_5511),
            ("o17", 0x3BD1_2952_68BA_65FC),
            ("o18", 0x37FB_3FAC_A631_3237),
            ("f17", 0x2B61_7F7D_4ED8_C33E),
            ("f18", 0x2B61_7F7D_4ED8_C33E),
            ("f19", 0x3B26_6591_B1C1_AD3D),
            ("ne18", 0x2B61_7F7D_4ED8_C33E),
            ("ne19", 0x2B61_7F7D_4ED8_C33E),
            ("ne20", 0x3C8E_4213_2F47_7B9F),
            ("ne22", 0x3CC4_1E32_2961_8024),
            ("mg22", 0x2B61_7F7D_4ED8_C33E),
            ("mg24", 0x3A24_2913_CC15_2108),
            ("mg26", 0x3C9D_64FB_BEE8_6FA4),
            ("fe56", 0x3C73_C576_1D41_3FAF),
        ],
    );
}

#[test]
fn test_change_model_network_light_element_cospi_taper_and_mlt_d_walk() {
    // `basic.net -> pp_and_cno_extras.net`: verifies cold solar `h2` (`lgT < 5.2`),
    // `mlt_d >= 100` downward walk into `(5.2, 5.5)` with `cospi` taper, and
    // hot cutoff `lgT >= 5.5` clipped to `1e-99`.
    let cold_xa: &[(&str, f64)] = &[
        ("h1", f64::from_bits(0x3FE6_6666_6666_6667)),
        ("he3", f64::from_bits(0x3EFF_3EBD_E705_8CC3)),
        ("he4", f64::from_bits(0x3FD1_EB08_23C0_B5D6)),
        ("c12", f64::from_bits(0x3F6C_3192_DD1B_4A1E)),
        ("n14", f64::from_bits(0x3F50_8485_F7FD_D8A4)),
        ("o16", f64::from_bits(0x3F83_2B92_3C42_8DCA)),
        ("ne20", f64::from_bits(0x3F61_32E3_42FF_8100)),
        ("mg24", f64::from_bits(0x3F70_C104_1826_5AB9)),
    ];
    let subsurf_xa: &[(&str, f64)] = &[
        ("h1", f64::from_bits(0x3FE6_6666_6616_44A0)),
        ("he3", f64::from_bits(0x3EFF_3EE5_F7E8_2F37)),
        ("he4", f64::from_bits(0x3FD1_EB08_23C0_B5D8)),
        ("c12", f64::from_bits(0x3F6C_3192_DD1B_49A1)),
        ("n14", f64::from_bits(0x3F50_8485_F7FD_D8A6)),
        ("o16", f64::from_bits(0x3F83_2B92_3C42_8DCC)),
        ("ne20", f64::from_bits(0x3F61_32E3_42FF_8102)),
        ("mg24", f64::from_bits(0x3F70_C104_1826_5ABB)),
    ];

    let mut model = build_test_model(
        "basic.net",
        0.02,
        &[
            (f64::from_bits(0x4021_973A_76EB_BDFC), 0.0, cold_xa),
            (f64::from_bits(0x4021_973A_76EB_BDFC), 0.0, cold_xa),
            (f64::from_bits(0x4027_8F32_196C_B939), 500.0, subsurf_xa),
            (f64::from_bits(0x4029_4240_92A9_BF30), 200.0, subsurf_xa),
            (f64::from_bits(0x4029_4A95_9A6D_D8FB), 10.0, subsurf_xa),
            (f64::from_bits(0x4029_579B_D5FF_F081), 10.0, subsurf_xa),
        ],
    );
    change_model_network(&mut model, "pp_and_cno_extras.net", None)
        .expect("basic -> pp_and_cno_extras");

    let sp = model.species;
    assert_eq!(model.iso_names[0], "h1");
    assert_eq!(model.iso_names[1], "h2");
    // Zone 0: cold surface (`lgT ~ 3.82 < 5.2`) bounded below by radiative
    // zone 1 (`mlt_d = 0 < 100`) receives full `chem_xsol("h2")` scaled into
    // `total_h`.
    assert_eq!(model.xa[0].to_bits(), 0x3FE6_6601_B8FE_C593);
    assert_eq!(model.xa[1].to_bits(), 0x3F09_2B59_E834_CA3D);
    // Zone 2: `mlt_d = 500` walks down to Zone 3 (`mlt_d = 200`, `lgT ~ 5.485`)
    // and stops before Zone 4 (`mlt_d = 10 < 100`), sharing Zone 3's taper.
    assert_eq!(model.xa[2 * sp].to_bits(), 0x3FE6_6665_C4D6_BB37);
    assert_eq!(model.xa[2 * sp + 1].to_bits(), 0x3E94_27F1_2D25_986C);
    assert_eq!(model.xa[3 * sp].to_bits(), 0x3FE6_6665_C4D6_BB37);
    assert_eq!(model.xa[3 * sp + 1].to_bits(), 0x3E94_27F1_2D25_986C);
    // Zone 4: standalone zone inside `(5.2, 5.5)` (`cospi` taper).
    assert_eq!(model.xa[4 * sp].to_bits(), 0x3FE6_6666_3846_56C6);
    assert_eq!(model.xa[4 * sp + 1].to_bits(), 0x3E76_E7F6_ED13_3F47);
    // Zone 5: hot zone (`lgT >= 5.5`) clips `h2` to `1e-99` via `clean1`.
    assert_eq!(model.xa[5 * sp].to_bits(), 0x3FE6_6666_6616_44A0);
    assert_eq!(model.xa[5 * sp + 1].to_bits(), 0x2B61_7F7D_4ED8_C33E);
}

#[test]
fn test_change_model_network_remaps_xa_old_when_generations_gt_1() {
    let zone_now: &[(&str, f64)] = &[
        ("h1", 0.65),
        ("he3", 3.0e-5),
        ("he4", 0.32997),
        ("c12", 0.0034),
        ("n14", 0.0010),
        ("o16", 0.0094),
        ("ne20", 0.0021),
        ("mg24", 0.0041),
    ];
    let mut model =
        build_test_model("basic.net", 0.02, &[(15.0, 0.0, zone_now)]);
    model.init_model_number = 0;
    model.model_number = 2;
    model.xa_old = vec![
        0.70, 2.0e-5, 0.27998, 0.0034, 0.0010, 0.0094, 0.0021, 0.0041,
    ];
    model.ln_t_old = vec![15.0];

    change_model_network(&mut model, "o18_and_ne22.net", None)
        .expect("remap xa and xa_old");
    assert_ne!(model.xa_old, model.xa);
    assert!((model.xa[0] - 0.65).abs() < 1.0e-15);
    assert!((model.xa_old[0] - 0.70).abs() < 1.0e-15);
}
