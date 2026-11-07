// SPDX-License-Identifier: LGPL-3.0-or-later
//! Golden parity and analytical Jacobian tests for `flexmesa-net` against
//! upstream Fortran MESA `net/test/test_output`.

#![forbid(unsafe_code)]
#![allow(
    clippy::approx_constant,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::unreadable_literal
)]

use flexmesa_chem::{
    category_id, get_composition_info, NuclideCatalog, CATEGORY_NAME,
};
use flexmesa_eos::{EosConfig, HelmTable};
use flexmesa_math::exp10;
use flexmesa_net::{
    burn_1_zone, burn_1_zone_const_p, Burn1ZoneInput, BurnConstPInput,
    NetEvalResult, NetGeneralInfo,
};
use flexmesa_num::RosenbrockVariant;
use flexmesa_rates::{RatesCatalog, ScreeningMode};

fn set_test_composition(net_file: &str, g: &NetGeneralInfo) -> Vec<f64> {
    let cat = NuclideCatalog::builtin_ref();
    let species = g.num_isos;
    let mut xin = vec![0.0; species];
    let idx = |name: &str| -> usize {
        let cid = cat
            .get_nuclide_index(name)
            .unwrap_or_else(|| panic!("unknown nuclide {name}"));
        let iso = g.net_iso[cid];
        assert!(iso > 0, "nuclide {name} not in {net_file}");
        iso - 1
    };

    match net_file {
        "basic.net" => {
            xin[idx("h1")] = f64::from(0.655_186_f32);
            xin[idx("he4")] = 0.310_021_64;
            xin[idx("c12")] = 0.002_725e-1;
            xin[idx("n14")] =
                0.203_101e-1 + 0.612_124e-6 + 0.109_305e-2 + 0.356_004e-4;
            xin[idx("o16")] = 0.094_000e-1;
            xin[idx("ne20")] = 0.162_163e-2;
            xin[idx("mg24")] = 0.658_226e-25;
            xin[idx("he3")] = 0.201_852e-2;
        }
        "o18_and_ne22.net" => {
            xin[idx("h1")] = f64::from(0.655_186_f32);
            xin[idx("he4")] = 0.310_021_64;
            xin[idx("c12")] = 0.002_725e-1;
            xin[idx("n14")] = 0.203_101e-1 + 0.612_124e-6 + 0.109_305e-2;
            xin[idx("o16")] = 0.094_000e-1;
            xin[idx("o18")] = 1.0e-20;
            xin[idx("ne20")] = 0.162_163e-2;
            xin[idx("mg24")] = 0.658_226e-25;
            xin[idx("ne22")] = 0.201_852e-2;
        }
        "pp_extras.net" => {
            xin[idx("h1")] = f64::from(0.655_186_f32);
            xin[idx("he4")] = 0.310_021_64;
            xin[idx("c12")] = 0.002_725e-1;
            xin[idx("n14")] =
                0.203_101e-1 + 0.612_124e-6 + 0.109_305e-2 + 0.356_004e-4;
            xin[idx("o16")] = 0.094_000e-1;
            xin[idx("ne20")] = 0.162_163e-2;
            xin[idx("mg24")] = 0.658_226e-25;
            xin[idx("h2")] = 0.632_956e-17;
            xin[idx("he3")] = 0.201_852e-2;
            xin[idx("li7")] = 0.664_160e-15;
            xin[idx("be7")] = 0.103_866e-15;
        }
        "cno_extras.net" => {
            xin[idx("h1")] = 0.173_891_680_788e-1;
            xin[idx("he4")] = 0.963_245_225_401;
            xin[idx("c12")] = 0.238_935_745_993e-3;
            xin[idx("n14")] = 0.134_050_688_300e-1;
            xin[idx("o16")] = 0.268_791_618_452e-3;
            xin[idx("ne20")] = 0.180_001_692_845e-2;
            xin[idx("mg24")] = 0.353_667_702_698e-2;
            xin[idx("c13")] = 0.717_642_727_071e-4;
            xin[idx("n13")] = 0.370_732_258_156e-9;
            xin[idx("n15")] = 0.450_484_708_137e-6;
            xin[idx("o14")] = 0.100_000_000_000e-49;
            xin[idx("o15")] = 0.874_815_374_966e-10;
            xin[idx("f17")] = 0.100_000_000_000e-49;
            xin[idx("f18")] = 0.100_000_000_000e-49;
            xin[idx("ne18")] = 0.100_000_000_000e-49;
            xin[idx("ne19")] = 0.100_000_000_000e-49;
            xin[idx("mg22")] = 0.439_011_547_696e-4;
        }
        "approx21.net" | "approx21_plus_co56.net" => {
            xin[idx("fe56")] = 8.038_702_148_431_816_6e-1;
            xin[idx("fe54")] = 1.609_664_873_676_083_2e-1;
            xin[idx("cr56")] = 2.948_094_553_592_052_5e-2;
            xin[idx("he4")] = 4.862_463_716_132_056_5e-3;
            xin[idx("ni56")] = 2.837_627_073_189_036_0e-4;
            xin[idx("si28")] = 1.501_862_890_613_573_9e-4;
            xin[idx("s32")] = 1.161_327_163_557_345_7e-4;
            xin[idx("prot")] = 1.113_943_163_367_365_3e-4;
            xin[idx("ca40")] = 5.368_837_747_349_418_5e-5;
            xin[idx("ar36")] = 5.270_283_182_256_706_2e-5;
            xin[idx("fe52")] = 3.786_650_413_193_518_5e-5;
            xin[idx("cr48")] = 5.840_112_303_766_797_4e-6;
            xin[idx("neut")] = 4.914_122_770_311_839_7e-6;
            xin[idx("ti44")] = 1.508_503_856_115_474_6e-6;
            xin[idx("o16")] = 9.538_401_960_904_925_5e-7;
            xin[idx("mg24")] = 6.380_820_771_772_558_0e-7;
            xin[idx("c12")] = 2.904_865_667_399_186_8e-7;
            xin[idx("ne20")] = 9.646_886_502_360_942_7e-9;
            xin[idx("he3")] = 4.620_360_386_226_309_6e-80;
            xin[idx("n14")] = 7.586_747_222_584_123_5e-99;
        }
        "approx21_cr60_plus_co56.net" => {
            xin[idx("he4")] = 3.455_539_253_481_393_9e-1;
            xin[idx("o16")] = 1.936_777_842_043_093_7e-1;
            xin[idx("h1")] = 1.905_293_136_717_250_1e-1;
            xin[idx("si28")] = 9.102_393_603_206_460_1e-2;
            xin[idx("ni56")] = 5.849_245_965_329_500_5e-2;
            xin[idx("s32")] = 4.141_664_247_699_990_8e-2;
            xin[idx("ne20")] = 2.092_778_233_247_755_8e-2;
            xin[idx("c12")] = 2.058_961_710_444_831_2e-2;
            xin[idx("mg24")] = 1.897_591_410_840_610_4e-2;
            xin[idx("ar36")] = 7.060_033_878_593_995_6e-3;
            xin[idx("ca40")] = 4.918_217_198_135_372_6e-3;
            xin[idx("fe52")] = 3.161_882_080_600_528_0e-3;
            xin[idx("n14")] = 1.987_846_008_276_095_2e-3;
            xin[idx("fe56")] = 7.166_877_662_113_019_0e-4;
            xin[idx("co56")] = 4.497_497_109_992_118_1e-4;
            xin[idx("fe54")] = 3.343_242_389_098_842_2e-4;
            xin[idx("cr48")] = 1.760_262_690_776_976_2e-4;
            xin[idx("he3")] = 5.165_009_719_163_154_5e-6;
            xin[idx("ti44")] = 2.645_242_220_338_990_6e-6;
            xin[idx("cr60")] = 4.765_335_309_593_798_2e-8;
            xin[idx("prot")] = 1.273_902_240_724_625_0e-11;
            xin[idx("neut")] = 0.0;
        }
        _ => panic!("unrecognized net_file {net_file}"),
    }

    let mut sum = 0.0;
    for xi in &mut xin {
        if *xi < 1.0e-99 {
            *xi = 1.0e-99;
        }
        sum += *xi;
    }
    let adj = idx("mg24");
    xin[adj] += 1.0 - sum;
    assert!(xin[adj] >= 0.0);
    xin
}

fn eval_test_net(
    net_file: &str,
    log_t: f64,
    log_rho: f64,
    eta: f64,
) -> (NetGeneralInfo, NetEvalResult, Vec<f64>) {
    let g = NetGeneralInfo::from_name(net_file)
        .unwrap_or_else(|e| panic!("failed to load {net_file}: {e}"));
    let xin = set_test_composition(net_file, &g);
    let cat = NuclideCatalog::builtin_ref();
    let comp = get_composition_info(cat, &g.chem_id, &xin, false).unwrap();
    let temp = exp10(log_t);
    let rho = exp10(log_rho);
    let rate_factors = vec![1.0; g.num_reactions];
    let res = flexmesa_net::eval_net(
        &g,
        false,
        false,
        &xin,
        temp,
        log_t,
        rho,
        log_rho,
        comp.abar,
        comp.zbar,
        comp.z2bar,
        comp.ye,
        eta,
        0.0,
        0.0,
        &rate_factors,
        1.0,
        ScreeningMode::Extended,
        false,
        false,
        false,
        false,
    )
    .unwrap_or_else(|e| panic!("eval_net failed for {net_file}: {e}"));
    (g, res, xin)
}

fn assert_rel_close(label: &str, actual: f64, expected: f64, rtol: f64) {
    let diff = (actual - expected).abs();
    let scale = expected.abs().max(1.0e-30);
    assert!(
        diff / scale <= rtol,
        "{label}: actual={actual:.12e}, expected={expected:.12e}, rel_err={:.3e} > {rtol:.3e}",
        diff / scale
    );
}

#[test]
fn test_net_golden_sections_1_to_4_standard_networks_match_mesa_test_output() {
    let cases = [
        (
            "basic.net",
            12.924_100_440,
            0.839_654_151e13,
            9.328_757_835,
            1.009_292_766,
            vec![("pp", 0.836_629e13), ("cno", 0.302_480e11)],
        ),
        (
            "o18_and_ne22.net",
            10.480_185_432,
            0.302_124_143e11,
            11.356_884_345,
            1.015_667_220,
            vec![("cno", 0.302_124e11), ("pp", 0.109_224e4)],
        ),
        (
            "pp_extras.net",
            12.921_874_596,
            0.835_361_769e13,
            9.326_560_302,
            1.009_292_884,
            vec![("pp", 0.832_337e13), ("cno", 0.302_480e11)],
        ),
        (
            "cno_extras.net",
            8.729_178_865,
            0.536_017_372e9,
            11.559_205_087,
            1.013_842_798,
            vec![("cno", 0.536_003e9), ("ne_alpha", 0.146_543e5)],
        ),
    ];

    let log_t = 7.833;
    let log_rho = 2.0;
    let temp = exp10(log_t);
    let rho = exp10(log_rho);

    for (
        net_file,
        exp_log_eps,
        exp_eps,
        exp_dlneps_dlnt,
        exp_dlneps_dlnrho,
        exp_cats,
    ) in cases
    {
        let (_g, work, _xin) = eval_test_net(net_file, log_t, log_rho, 0.0);
        let log_eps = work.eps_nuc.log10();
        let d_lneps_dlnt = work.d_eps_nuc_dt * temp / work.eps_nuc;
        let d_lneps_dlnrho = work.d_eps_nuc_drho * rho / work.eps_nuc;

        assert_rel_close(
            &format!("{net_file} log(eps_nuc)"),
            log_eps,
            exp_log_eps,
            1.0e-8,
        );
        assert_rel_close(
            &format!("{net_file} eps_nuc"),
            work.eps_nuc,
            exp_eps,
            1.0e-8,
        );
        assert_rel_close(
            &format!("{net_file} d_lneps_dlnT"),
            d_lneps_dlnt,
            exp_dlneps_dlnt,
            1.0e-8,
        );
        assert_rel_close(
            &format!("{net_file} d_lneps_dlnRho"),
            d_lneps_dlnrho,
            exp_dlneps_dlnrho,
            1.0e-8,
        );

        for (cat_name, exp_cat_val) in exp_cats {
            let icat = category_id(cat_name) - 1;
            assert_rel_close(
                &format!("{net_file} category {cat_name}"),
                work.eps_nuc_categories[icat],
                exp_cat_val,
                1.0e-5,
            );
        }
    }
}

#[test]
fn test_net_golden_sections_5_to_6_approx21_and_co56_match_mesa_test_output() {
    let log_t = 9.6;
    let log_rho = 6.0;
    let temp = exp10(log_t);
    let rho = exp10(log_rho);

    for net_file in ["approx21.net", "approx21_plus_co56.net"] {
        let (_g, work, _xin) = eval_test_net(net_file, log_t, log_rho, 0.0);
        let log_eps = work.eps_nuc.log10();
        let d_lneps_dlnt = work.d_eps_nuc_dt * temp / work.eps_nuc;
        let d_lneps_dlnrho = work.d_eps_nuc_drho * rho / work.eps_nuc;

        assert_rel_close(
            &format!("{net_file} log(eps_nuc)"),
            log_eps,
            24.178_055_699,
            1.0e-8,
        );
        assert_rel_close(
            &format!("{net_file} eps_nuc"),
            work.eps_nuc,
            0.150_680_030e25,
            1.0e-8,
        );
        assert_rel_close(
            &format!("{net_file} d_lneps_dlnT"),
            d_lneps_dlnt,
            -0.251_306_088,
            1.0e-7,
        );
        assert_rel_close(
            &format!("{net_file} d_lneps_dlnRho"),
            d_lneps_dlnrho,
            1.008_425_906,
            1.0e-8,
        );

        let common_cats = [
            ("fe_co_ni", 0.150_665e25),
            ("photo", 0.145_046e21),
            ("si_alpha", 0.151_575e19),
            ("s_alpha", 0.113_141e19),
            ("ar_alpha", 0.355_805e18),
            ("mg_alpha", 0.108_517e18),
            ("ti_alpha", 0.278_504e17),
            ("ca_alpha", 0.105_102e17),
            ("cr_alpha", 0.585_792e16),
            ("o_alpha", 0.184_396e16),
            ("ne_alpha", 0.545_696e15),
            ("c_alpha", 0.118_422e15),
            ("tri_alpha", 0.327_188e12),
            ("pnhe4", 0.237_411e12),
            ("c12_c12", 0.114_836e12),
            ("c12_o16", 0.152_505e11),
            ("ni56_co56", 0.381_664e10),
            ("o16_o16", 0.697_742e8),
        ];
        for (cat_name, exp_val) in common_cats {
            let icat = category_id(cat_name) - 1;
            assert_rel_close(
                &format!(
                    "{net_file} category {cat_name} ({})",
                    CATEGORY_NAME[icat]
                ),
                work.eps_nuc_categories[icat],
                exp_val,
                1.0e-5,
            );
        }
        if net_file == "approx21.net" {
            let icat = category_id("co56_fe56") - 1;
            assert_rel_close(
                "approx21.net category co56_fe56",
                work.eps_nuc_categories[icat],
                0.824_071e10,
                1.0e-5,
            );
        }
    }
}

#[test]
fn test_net_golden_section_7_approx21_cr60_plus_co56_weak_rates_match_mesa() {
    let log_t = 4.623_300_792_265_933_3;
    let log_rho = -10.746_410_107_891_649;
    let eta = -22.590_260_158_215_202;
    let (g, work, _xin) =
        eval_test_net("approx21_cr60_plus_co56.net", log_t, log_rho, eta);

    let j_co56 = g.net_reaction_by_name["rco56ec_to_fe56"];
    let j_ni56 = g.net_reaction_by_name["rni56ec_to_co56"];
    assert!(j_co56 > 0 && j_ni56 > 0);

    assert_rel_close(
        "rate_raw rco56ec_to_fe56",
        work.rate_raw[j_co56 - 1],
        1.040_292_931_952_491_8e-7,
        1.0e-12,
    );
    assert_rel_close(
        "rate_raw rni56ec_to_co56",
        work.rate_raw[j_ni56 - 1],
        1.315_269_792_333_862_1e-6,
        1.0e-12,
    );
}

#[test]
fn test_net_golden_section_8a_one_zone_burn_small_net_matches_mesa() {
    let _ = RatesCatalog::builtin();
    let g = NetGeneralInfo::from_name("approx21.net").unwrap();
    let ht = HelmTable::builtin();
    let eos_cfg = EosConfig::default();

    let cat = NuclideCatalog::builtin_ref();
    let ihe4 = g.net_iso[cat.get_nuclide_index("he4").unwrap()] - 1;
    let ic12 = g.net_iso[cat.get_nuclide_index("c12").unwrap()] - 1;

    let mut starting_x = vec![0.0; g.num_isos];
    starting_x[ihe4] = 1.0;

    let times = [0.0, 100.0, 150.0];
    let log10_ts = [8.3, 8.4, 8.3];
    let log10_rhos = [7.0, 7.5, 8.1];
    let etas = [0.0, 0.0, 0.0];
    let rate_factors = vec![1.0; g.num_reactions];

    let input = Burn1ZoneInput {
        t_start: 0.0,
        t_end: 150.0,
        starting_x: &starting_x,
        times: &times,
        log10_ts: &log10_ts,
        log10_rhos: &log10_rhos,
        etas: &etas,
        dxdt_source_term: None,
        rate_factors: &rate_factors,
        weak_rate_factor: 1.0,
        reaction_qs: &[],
        reaction_neu_qs: &[],
        screening_mode: ScreeningMode::Extended,
        stptry_in: 0.0,
        max_steps: 50_000,
        eps: 1.0e-8,
        odescal: 1.0e-12,
    };

    let out = burn_1_zone(&g, ht, &eos_cfg, &input).unwrap();
    let xsum: f64 = out.ending_x.iter().sum();

    assert_rel_close(
        "burn_1_zone c12",
        out.ending_x[ic12],
        9.871_434_225_406_838_8e-1,
        1.0e-6,
    );
    assert_rel_close(
        "burn_1_zone he4",
        out.ending_x[ihe4],
        1.280_218_683_637_965_7e-2,
        1.0e-5,
    );
    assert_rel_close("burn_1_zone xsum", xsum, 1.000_000_000_000_633, 1.0e-8);
}

#[test]
fn test_net_golden_section_8b_one_zone_burn_const_p_matches_mesa() {
    let g = NetGeneralInfo::from_name("approx21.net").unwrap();
    let ht = HelmTable::builtin();
    let eos_cfg = EosConfig::default();

    let cat = NuclideCatalog::builtin_ref();
    let idx = |name: &str| g.net_iso[cat.get_nuclide_index(name).unwrap()] - 1;
    let ic12 = idx("c12");
    let io16 = idx("o16");
    let img24 = idx("mg24");
    let isi28 = idx("si28");
    let is32 = idx("s32");

    let mut starting_x = vec![0.0; g.num_isos];
    starting_x[ic12] = 0.37;
    starting_x[io16] = 0.63;

    let rate_factors = vec![1.0; g.num_reactions];

    let input = BurnConstPInput {
        solver: RosenbrockVariant::Rodas4,
        starting_temp: 3.0e9,
        starting_x: &starting_x,
        clip: false,
        t_end: 1.0e-4,
        pressure: exp10(24.0),
        rate_factors: &rate_factors,
        weak_rate_factor: 1.0,
        reaction_qs: &[],
        reaction_neu_qs: &[],
        screening_mode: ScreeningMode::Extended,
        h0: 1.0e-6,
        max_step_size: 1.0,
        max_steps: 1000,
        rtol: 1.0e-7,
        atol: 1.0e-8,
        x_min: -1.0e-10,
    };

    let out = burn_1_zone_const_p(&g, ht, &eos_cfg, &input).unwrap();
    let xsum: f64 = out.ending_x.iter().sum();

    assert_rel_close(
        "burn_const_P o16",
        out.ending_x[io16],
        6.955_981_012_989_815_9e-1,
        1.0e-9,
    );
    assert_rel_close(
        "burn_const_P si28",
        out.ending_x[isi28],
        2.046_882_503_012_717_5e-1,
        1.0e-8,
    );
    assert_rel_close(
        "burn_const_P s32",
        out.ending_x[is32],
        4.436_843_719_401_110_9e-2,
        1.0e-8,
    );
    assert_rel_close(
        "burn_const_P mg24",
        out.ending_x[img24],
        3.785_876_161_937_664_5e-2,
        1.0e-8,
    );
    assert_rel_close("burn_const_P xsum", xsum, 1.0, 1.0e-12);
}
