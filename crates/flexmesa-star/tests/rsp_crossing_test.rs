// SPDX-License-Identifier: LGPL-3.0-or-later
//! Upstream Fortran MESA (`r24.08.1`, commit `0b2e6bb`) Cepheid 2nd-crossing
//! instability-strip boundary interpolation tests (`rsp_check_2nd_crossing`,
//! Issue `#5`).

use std::fmt::Write as _;
use std::fs;

use flexmesa_eos::EosConfig;
use flexmesa_kap::{KapCoOption, KapConfig, KapLowTOption, KapOption};
use flexmesa_star::{
    eval_rsp_2nd_crossing_with_controls, StarControls, StarJobControls,
};
use flexmesa_utils::resolve_mesa_test_inputs_dir;

fn a09_kap_config(zbase: f64) -> KapConfig {
    KapConfig {
        kap_option: KapOption::A09,
        kap_co_option: KapCoOption::A09Co,
        kap_low_t_option: KapLowTOption::LowTFa05A09p,
        zbase,
        ..Default::default()
    }
}

fn base_rsp_job() -> StarJobControls {
    StarJobControls {
        create_rsp_model: true,
        initial_zfracs: 6,
        ..Default::default()
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_rsp_lina_and_steffen_edges_match_golden_5m_list() {
    // 13 sampled 2nd-crossing points up to `num_beyond_blue_edge == 2`
    // (`run_star_extras.f90:223` in MESA `r24.08.1`), storing both exact
    // 17-digit MESA `r24.08.1` (`FreeEOS`) `LINA` reference values and
    // historical MESA `v11461` (`OPAL` EOS) `list` values.
    let golden_pts: [(i64, f64, f64, f64, f64, f64, f64); 13] = [
        (
            2182,
            5_117.814_030_014_12,
            1_515.874_556_268_378_7,
            6.015_024_817_638_153e5 / 86_400.0,
            -1.464_218_682_915_794_7e-2,
            6.9798,
            -1.4649e-2,
        ),
        (
            2194,
            5_219.704_046_631_389,
            1_539.199_742_418_412_8,
            5.607_115_052_868_42e5 / 86_400.0,
            -9.901_928_079_745_98e-3,
            6.5072,
            -9.8892e-3,
        ),
        (
            2205,
            5_329.114_836_734_605,
            1_559.631_654_892_086_3,
            5.215_282_297_889_703e5 / 86_400.0,
            -4.863_571_008_226_757e-3,
            6.0527,
            -4.869_439_213_642_279e-3,
        ),
        (
            2215,
            5_436.915_620_674_93,
            1_578.414_513_983_107_6,
            4.874_781_823_508_197_6e5 / 86_400.0,
            -1.044_680_598_213_734_6e-4,
            5.6574,
            -2.440_425_682_565_504e-4,
        ),
        (
            2224,
            5_538.172_717_431_118,
            1_595.663_842_458_941,
            4.589_592_835_194_748e5 / 86_400.0,
            4.073_300_166_790_113e-3,
            5.3266,
            4.045_355_766_436_185e-3,
        ),
        (
            2233,
            5_641.070_412_718_255,
            1_612.966_443_952_483_4,
            4.327_848_595_051_984e5 / 86_400.0,
            8.085_525_277_106_233e-3,
            5.0226,
            8.141_442_149_700_16e-3,
        ),
        (
            2242,
            5_747.635_013_234_617,
            1_630.692_294_723_043_2,
            4.081_325_912_454_219e5 / 86_400.0,
            1.133_212_228_545_723_4e-2,
            4.7364,
            1.1213e-2,
        ),
        (
            2251,
            5_857.062_559_238_12,
            1_648.779_536_817_859_3,
            3.850_341_405_518_177e5 / 86_400.0,
            1.262_171_509_132_802_4e-2,
            4.4683,
            1.2694e-2,
        ),
        (
            2260,
            5_969.348_676_651_82,
            1_667.201_609_538_533_6,
            3.633_405_009_525_161_6e5 / 86_400.0,
            1.142_481_483_142_316e-2,
            4.2169,
            1.1268e-2,
        ),
        (
            2268,
            6_071.304_465_353_705,
            1_683.971_577_622_985_7,
            3.452_702_916_275_847e5 / 86_400.0,
            8.381_239_237_986_78e-3,
            4.0071,
            8.689_902_602_617_62e-3,
        ),
        (
            2276,
            6_176.179_601_321_493,
            1_701.172_948_889_821_8,
            3.281_073_459_697_96e5 / 86_400.0,
            4.248_266_288_919_205e-3,
            3.8078,
            4.316_245_731_944_837e-3,
        ),
        (
            2284,
            6_284.581_093_120_282,
            1_718.874_374_974_722_5,
            3.117_201_220_586_360_4e5 / 86_400.0,
            -4.136_798_069_435_589e-4,
            3.6174,
            -2.786_712_154_970_167_4e-4,
        ),
        (
            2292,
            6_396.226_491_520_547,
            1_737.050_610_191_606,
            2.960_904_036_216_642e5 / 86_400.0,
            -5.133_231_648_719_714_5e-3,
            3.4361,
            -5.184_950_937_238_88e-3,
        ),
    ];

    let tmp_dir = std::env::temp_dir()
        .join(format!("flexmesa_rsp_5m_crossing_{}", std::process::id()));
    fs::create_dir_all(&tmp_dir).expect("create tmp dir");
    let mut hist = String::from(
        "h1\nh2\nh3\nh4\nh5\nmodel_number star_age star_mass log_Teff log_L\n",
    );
    hist.push_str("2100 5.0e7 5.0 3.6800000000000000 3.1600000000000000\n");
    for (idx, &(modnum, teff, lum, _, _, _, _)) in golden_pts.iter().enumerate()
    {
        let idx_f64 = f64::from(u32::try_from(idx).expect("idx fits in u32"));
        let _ = writeln!(
            hist,
            "{modnum} {:.6e} 5.0 {:.16e} {:.16e}",
            5.1e7 + idx_f64 * 1.0e4,
            teff.log10(),
            lum.log10()
        );
    }
    fs::write(tmp_dir.join("history_5M.data"), &hist).expect("write history");

    let mut ctrl = StarControls {
        rsp_mass: 5.0,
        rsp_x: 0.736,
        rsp_z: 0.008,
        zbase: 0.008,
        ..Default::default()
    };
    ctrl.rsp.t_anchor_tolerance = 1.0e-6;
    ctrl.rsp.t_inner_tolerance = 1.0e-5;
    ctrl.x_ctrl[1] = 100.0;
    ctrl.x_ctrl[2] = 3.76;
    ctrl.x_ctrl[3] = 4.5;
    ctrl.x_ctrl[4] = 3.83;
    ctrl.x_ctrl[5] = 2.4;
    ctrl.x_ctrl[6] = 3.65;
    ctrl.x_ctrl[7] = 4.5;
    ctrl.x_ctrl[8] = 3.77;
    ctrl.x_ctrl[9] = 2.4;
    ctrl.x_integer_ctrl[1] = 6;
    ctrl.x_integer_ctrl[2] = 1;
    ctrl.x_integer_ctrl[3] = 2;
    ctrl.x_integer_ctrl[4] = 4;
    ctrl.x_integer_ctrl[5] = 5;
    ctrl.x_character_ctrl[2] = String::from("blue_edge_5M_A.data");

    let rep = eval_rsp_2nd_crossing_with_controls(
        &tmp_dir,
        &ctrl,
        &base_rsp_job(),
        &EosConfig::default(),
        &a09_kap_config(0.008),
        "history_5M.data",
    )
    .expect("eval_rsp_2nd_crossing_with_controls");

    assert_eq!(rep.points.len(), 13);
    for (p, &(_m, _t, _l, p_r24, g_r24, p_v11461, g_v11461)) in
        rep.points.iter().zip(golden_pts.iter())
    {
        let rel_p_r24 = (p.period_days - p_r24).abs() / p_r24;
        let abs_g_r24 = (p.growth_rate - g_r24).abs();
        let rel_p_legacy = (p.period_days - p_v11461).abs() / p_v11461;
        let abs_g_legacy = (p.growth_rate - g_v11461).abs();
        assert!(
            rel_p_r24 < 1.0e-6,
            "mod {}: P_days={} vs r24 {p_r24} (rel={rel_p_r24:.3e})",
            p.model_number,
            p.period_days
        );
        assert!(
            abs_g_r24 < 5.0e-6,
            "mod {}: growth={} vs r24 {g_r24} (diff={abs_g_r24:.3e})",
            p.model_number,
            p.growth_rate
        );
        assert!(rel_p_legacy < 3.0e-3);
        assert!(abs_g_legacy < 5.0e-4);
    }

    assert!((rep.teff_red_k - 5_442.635_091_792_743).abs() < 5.0);
    assert!((rep.teff_blue_k - 6_278.109_355_229_067).abs() < 5.0);
    assert_eq!(rep.offsets.len(), 10);
    assert!(tmp_dir.join("blue_edge_5M_A.data").is_file());

    let _ = fs::remove_file(tmp_dir.join("history_5M.data"));
    let _ = fs::remove_file(tmp_dir.join("blue_edge_5M_A.data"));
    let _ = fs::remove_dir(&tmp_dir);
}

#[test]
fn test_rsp_check_2nd_crossing_on_7m_mesa_test_suite() {
    let inputs_dir = resolve_mesa_test_inputs_dir();
    let case_dir = inputs_dir
        .join("star")
        .join("test_suite")
        .join("rsp_check_2nd_crossing");
    let hist_rel = "history_logs/history_7M.data";
    if !case_dir.join(hist_rel).is_file() {
        assert!(
            std::env::var("FLEXMESA_REQUIRE_ZERO_SKIPS").as_deref() != Ok("1"),
            "missing {} with FLEXMESA_REQUIRE_ZERO_SKIPS=1",
            case_dir.join(hist_rel).display()
        );
        return;
    }

    let mut ctrl = StarControls {
        rsp_mass: 7.0,
        rsp_x: 0.736,
        rsp_z: 0.008,
        zbase: 0.008,
        ..Default::default()
    };
    ctrl.rsp.alfam = 0.5;
    ctrl.rsp.gammar = 1.0;
    ctrl.rsp.t_anchor_tolerance = 1.0e-6;
    ctrl.rsp.t_inner_tolerance = 1.0e-5;
    ctrl.x_ctrl[1] = 100.0;
    ctrl.x_ctrl[2] = 3.76;
    ctrl.x_ctrl[3] = 4.5;
    ctrl.x_ctrl[4] = 3.83;
    ctrl.x_ctrl[5] = 2.4;
    ctrl.x_ctrl[6] = 3.65;
    ctrl.x_ctrl[7] = 4.5;
    ctrl.x_ctrl[8] = 3.77;
    ctrl.x_ctrl[9] = 2.4;
    ctrl.x_integer_ctrl[1] = 6;
    ctrl.x_integer_ctrl[2] = 1;
    ctrl.x_integer_ctrl[3] = 2;
    ctrl.x_integer_ctrl[4] = 4;
    ctrl.x_integer_ctrl[5] = 5;

    let rep = eval_rsp_2nd_crossing_with_controls(
        &case_dir,
        &ctrl,
        &base_rsp_job(),
        &EosConfig::default(),
        &a09_kap_config(0.008),
        hist_rel,
    )
    .expect("7M rsp_check_2nd_crossing");

    assert!(rep.teff_red_k > 4300.0 && rep.teff_red_k < 5600.0);
    assert!(
        rep.teff_blue_k > rep.teff_red_k + 300.0 && rep.teff_blue_k < 6500.0
    );
    assert!(rep.offsets.len() >= 4);
}
