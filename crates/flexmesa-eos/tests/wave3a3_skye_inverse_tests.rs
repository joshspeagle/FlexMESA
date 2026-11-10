// SPDX-License-Identifier: LGPL-3.0-or-later
//! Golden parity integration tests for Wave 3a.3 (`Skye` equation of state,
//! `get_skye_alfa`, `eosDT_get` `Skye`/`HELM` blending, and `eosPT_get` /
//! `eosDT_get_T` / `eosDT_get_Rho` / `eos_gamma_DE_get` inverse solvers)
//! against upstream Fortran MESA (`skye.f90`, `eosdt_eval.f90`,
//! `eospt_eval.f90`, and `eosde_eval.f90`).

#![allow(
    clippy::approx_constant,
    clippy::cast_possible_wrap,
    clippy::excessive_precision,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_const::LN10;
use flexmesa_eos::{
    eos_dt_get, eos_dt_get_rho, eos_dt_get_t, eos_gamma_de_get, eos_pt_get,
    eval_skye_eos, get_skye_alfa, get_skye_alfa_simple, EosConfig,
    EosEvalOutput, HelmTable, SkyeConfig, SkyeSolidMixingRule,
    EOS_BASIC_RESULT_NAMES, I_LN_E, I_LN_PGAS, NUM_EOS_BASIC_RESULTS,
};

fn make_comp(
    y_in: f64,
    xc_in: f64,
    xo_in: f64,
) -> (Vec<usize>, Vec<f64>, f64, f64) {
    let cat = NuclideCatalog::builtin();
    let chem_id = vec![
        cat.get_by_name("h1").unwrap().chem_id,
        cat.get_by_name("he4").unwrap().chem_id,
        cat.get_by_name("c12").unwrap().chem_id,
        cat.get_by_name("n14").unwrap().chem_id,
        cat.get_by_name("o16").unwrap().chem_id,
        cat.get_by_name("ne20").unwrap().chem_id,
        cat.get_by_name("mg24").unwrap().chem_id,
    ];
    let xa = vec![0.0, y_in, xc_in, 0.0, xo_in, 0.0, 0.0];
    let comp = get_composition_info(&cat, &chem_id, &xa, true).unwrap();
    (chem_id, xa, comp.abar, comp.zbar)
}

fn assert_eos_case(
    label: &str,
    out: &EosEvalOutput,
    expected: &[[f64; 3]; NUM_EOS_BASIC_RESULTS],
    tol: f64,
) {
    for (idx, exp) in expected.iter().enumerate() {
        let name = EOS_BASIC_RESULT_NAMES[idx];
        let got = [out.res[idx], out.d_dlnd[idx], out.d_dln_t[idx]];
        for (col, (&g, &e)) in got.iter().zip(exp.iter()).enumerate() {
            let scale = e.abs().max(1.0);
            let rel_err = (g - e).abs() / scale;
            assert!(
                rel_err < tol,
                "{label} [{idx}: {name}] col {col}: got {g:.16e}, exp {e:.16e} \
                 (rel_err = {rel_err:.3e} >= {tol:.1e})"
            );
        }
    }
}

#[test]
#[rustfmt::skip]
fn test_skye_liquid_and_ion_offset_regimes_match_fortran_oracle() {
    let ht = HelmTable::builtin();
    let (chem_id, xa, abar, zbar) = make_comp(0.0, 1.0, 0.0);

    // 1. skye_c12_liq (use_ion_offsets = false)
    let mut cfg = SkyeConfig {
        skye_use_ion_offsets: false,
        skye_solid_mixing_rule: SkyeSolidMixingRule::Ogata,
        ..SkyeConfig::default()
    };
    let out1 = eval_skye_eos(&cfg, ht, 1.0e6, 1.0e8, abar, zbar, &chem_id, &xa)
        .unwrap();
    let exp_c12_liq: [[f64; 3]; 26] = [
        [5.1635980952299732e1, 1.5394240603497165e0, 5.6266242057777235e-2],
        [3.8290006586052314e1, 5.9009160249847770e-1, 6.1711025464067704e-2],
        [1.8526649059675723e1, -1.3481568971217803e-1, 2.3630879238943694e-1],
        [1.7142857142857142e0, 0.0, 0.0],
        [-6.9314718055994540e-1, 0.0, 0.0],
        [1.6590126339635113e1, 9.9255820137419235e0, -1.6745100827018351e1],
        [3.6302588454390750e-1, -2.9257929435979020e-3, -4.6194082270026131e-3],
        [1.5394094768307469e0, -1.8045430640561442e-2, -4.6719193039714155e-2],
        [5.6303602469083738e-2, -4.6719193039714141e-2, 7.8112002135655029e-2],
        [2.6820702231799945e7, -7.2571850176630868e6, 1.3053981532013737e7],
        [2.6272496060720835e7, -6.6495458176626824e6, 1.2033592264883243e7],
        [2.5122219547514553e10, -1.0327340757571991e10, -6.6495458176626825e8],
        [2.6272496060720835e-1, -6.6495458176626845e-2, -1.4238903795837596e-1],
        [-1.4988628400459321e1, 1.9340775066462808e1, -6.6495458176626840e0],
        [1.5715310450694795e0, -4.5896013165406702e-2, -2.6180425371709467e-3],
        [1.5705064477245592e0, -2.1259415218719914e-2, -8.2099606464140270e-3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [1.0, 0.0, 0.0],
        [0.0; 3],
        [0.0; 3],
    ];
    assert_eos_case("skye_c12_liq", &out1, &exp_c12_liq, 1.0e-12);

    // 2. skye_c12_liq_ion (use_ion_offsets = true)
    cfg.skye_use_ion_offsets = true;
    let out2 = eval_skye_eos(&cfg, ht, 1.0e6, 1.0e8, abar, zbar, &chem_id, &xa)
        .unwrap();
    let mut exp_c12_liq_ion = exp_c12_liq;
    exp_c12_liq_ion[1] = [3.8291950165675473e1, 5.8894582629837955e-1, 6.1591201653728764e-2];
    assert_eos_case("skye_c12_liq_ion", &out2, &exp_c12_liq_ion, 1.0e-12);

    // 3. skye_co_liq
    cfg.skye_use_ion_offsets = false;
    let (chem_id_co, xa_co, abar_co, zbar_co) = make_comp(0.0, 0.5, 0.5);
    let out3 = eval_skye_eos(
        &cfg,
        ht,
        1.0e6,
        1.0e7,
        abar_co,
        zbar_co,
        &chem_id_co,
        &xa_co,
    )
    .unwrap();
    let exp_co_liq: [[f64; 3]; 26] = [
        [5.1590441654670123e1, 1.5717247130541014e0, 3.5662933553839127e-3],
        [3.8237206857396821e1, 6.2760240164645187e-1, 4.2178908347575966e-3],
        [1.8003907633427506e1, -1.3761441841292232e-1, 2.5840775263763865e-1],
        [1.7454545454545454e0, 0.0, 0.0],
        [-6.9314718055994540e-1, 0.0, 0.0],
        [1.6666635909359002e2, 9.8929566934109957e1, -1.6668177581555085e2],
        [3.3842111152193294e-1, -3.4810890751585714e-4, 1.9061355371752933e-2],
        [1.5717247114957618e0, -4.7475456404621141e-2, -2.1293843054855017e-3],
        [3.5662973177833521e-3, -2.1293843054855021e-3, 3.5094037415172477e-3],
        [1.7054058483251877e7, 1.0017265627987537e5, -1.1486997411877925e6],
        [1.7033475764372379e7, 1.1236257114724358e5, -1.1687270325312091e6],
        [2.5345014171570206e10, -1.0800480821728235e10, 1.1236257114724356e6],
        [1.7033475764372372e0, 1.1236257114724334e-2, -1.8202202796903582e0],
        [-9.0711359738178636e0, 9.3011868507485165e0, 1.1236257114724334e-1],
        [1.5736239344341794e0, -4.8670148918055815e-2, -1.5367283915352441e-4],
        [1.5325475610087322e0, -1.7018798403443135e-2, 2.9943398902708866e-2],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [1.0, 0.0, 0.0],
        [0.0; 3],
        [0.0; 3],
    ];
    assert_eos_case("skye_co_liq", &out3, &exp_co_liq, 1.0e-11);
}

#[test]
#[rustfmt::skip]
fn test_skye_solid_and_phase_transition_regimes_match_fortran_oracle() {
    let ht = HelmTable::builtin();
    let (chem_id_co, xa_co, abar_co, zbar_co) = make_comp(0.0, 0.5, 0.5);
    let mut cfg = SkyeConfig {
        skye_use_ion_offsets: false,
        skye_solid_mixing_rule: SkyeSolidMixingRule::Ogata,
        ..SkyeConfig::default()
    };

    // 4. skye_co_sol_ogata (T = 2e6 K, rho = 1e6 g/cm^3)
    let out_og = eval_skye_eos(
        &cfg,
        ht,
        1.0e6,
        2.0e6,
        abar_co,
        zbar_co,
        &chem_id_co,
        &xa_co,
    )
    .unwrap();
    let exp_og: [[f64; 3]; 26] = [
        [5.1587427168273294e1, 1.5734680542819544e0, 7.1213235028186794e-4],
        [3.8233423396161776e1, 6.2988427188296325e-1, 8.9541590645723728e-4],
        [1.7298014789469370e1, -2.7748103586764178e-1, 5.5351225868004905e-1],
        [1.7454545454545454e0, 0.0, 0.0],
        [-6.9314718055994540e-1, 0.0, 0.0],
        [8.3336879273744080e2, 4.9463220628165021e2, -8.3337334553495714e2],
        [3.1852946270197890e-1, 1.0438089722961971e-2, -1.3172401848639884e-2],
        [1.5734680542794510e0, -4.8644045284437387e-2, -5.0040713441014189e-4],
        [7.1213235664538866e-4, -5.0040713441014178e-4, 8.9760677340828187e-4],
        [1.8015990287705507e7, -2.3614418140730597e6, 5.4558420626837090e6],
        [1.8011903628081579e7, -2.3581684261378562e6, 5.4496224573745020e6],
        [2.5341106228253265e10, -1.0796101439003977e10, -4.7163368522757124e6],
        [9.0059518140407882e0, -1.1790842130689279e0, -6.2811405853535360e0],
        [-9.0295410775305349e0, 1.0196341346769977e1, -2.3581684261378557e0],
        [1.5738250531008828e0, -4.8894296437525804e-2, -6.5206245155995248e-5],
        [1.5013096485511377e0, 8.5345313907602143e-4, -2.0751826149136184e-2],
        [9.9998982470934628e-1, 3.2032509951368754e-4, -1.0342874613539180e-3],
        [-1.2036666262855533e11, 3.2719359195671099e12, -1.0697889915426936e13],
        [3.7257028474115829e10, -1.0106764408817386e12, 3.3091929480412256e12],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [1.0, 0.0, 0.0],
        [0.0; 3],
        [0.0; 3],
    ];
    assert_eos_case("skye_co_sol_ogata", &out_og, &exp_og, 1.0e-9);

    // 5. skye_co_sol_pc (PC solid mixing rule)
    cfg.skye_solid_mixing_rule = SkyeSolidMixingRule::Pc;
    let out_pc = eval_skye_eos(
        &cfg,
        ht,
        1.0e6,
        2.0e6,
        abar_co,
        zbar_co,
        &chem_id_co,
        &xa_co,
    )
    .unwrap();
    let exp_pc: [[f64; 3]; 26] = [
        [5.1587443284692888e1, 1.5734641842012194e0, 7.1212087335059487e-4],
        [3.8233453872083665e1, 6.2987523435743720e-1, 8.9538861824782988e-4],
        [1.7298014789469370e1, -2.7748103586764350e-1, 5.5351225868004905e-1],
        [1.7454545454545454e0, 0.0, 0.0],
        [-6.9314718055994540e-1, 0.0, 0.0],
        [8.3336879273744080e2, 4.9463220628165021e2, -8.3337334553495714e2],
        [3.1853024714141603e-1, 1.0437791946021124e-2, -1.3172435026995048e-2],
        [1.5734641841987158e0, -4.8642331997130549e-2, -5.0039631373848524e-4],
        [7.1212087971401312e-4, -5.0039631373848524e-4, 8.9759231549043196e-4],
        [1.8015990231895074e7, -2.3614417576927869e6, 5.4558419777661897e6],
        [1.8011903628081579e7, -2.3581684261378562e6, 5.4496224573745020e6],
        [2.5341514930495277e10, -1.0796373907165318e10, -4.7163368522757124e6],
        [9.0059518140407882e0, -1.1790842130689279e0, -6.2811405853535360e0],
        [-9.0295410775305918e0, 1.0196341346769977e1, -2.3581684261378557e0],
        [1.5738211772666513e0, -4.8892577735506877e-2, -6.5202434210767931e-5],
        [1.5013096485511408e0, 8.5345313907943016e-4, -2.0751826149137193e-2],
        [7.9955778725151072e-1, 4.5051065373017245e0, -1.4670028937649430e1],
        [1.9081668383850775e14, -7.7585326377946470e15, 2.5479081605183016e16],
        [-5.8603621269252812e13, 2.3778658190834595e15, -7.8171362590639000e15],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [1.0, 0.0, 0.0],
        [0.0; 3],
        [0.0; 3],
    ];
    assert_eos_case("skye_co_sol_pc", &out_pc, &exp_pc, 1.0e-9);

    // 6. skye_c12_phase (T = 2.65e6 K, rho = 1e6 g/cm^3)
    cfg.skye_solid_mixing_rule = SkyeSolidMixingRule::Ogata;
    let (chem_id_c, xa_c, abar_c, zbar_c) = make_comp(0.0, 1.0, 0.0);
    let out_ph = eval_skye_eos(
        &cfg, ht, 1.0e6, 2.65e6, abar_c, zbar_c, &chem_id_c, &xa_c,
    )
    .unwrap();
    let exp_ph: [[f64; 3]; 26] = [
        [5.1591011822094835e1, 1.5724818941092962e0, 1.1011074015228320e-3],
        [3.8240091322893214e1, 6.2770072864441739e-1, 1.3938698541871332e-3],
        [1.7468272942194005e1, -2.7409494028886966e-1, 5.5215683610561939e-1],
        [1.7142857142857142e0, 0.0, 0.0],
        [-6.9314718055994540e-1, 0.0, 0.0],
        [6.2895669332972716e2, 3.7330769266265213e2, -6.2896196705352679e2],
        [3.1557455330660084e-1, -3.0739249976749171e-5, 2.1451350072571103e-2],
        [1.5724818941016114e0, -4.8178413768271929e-2, -6.5313356943676485e-4],
        [1.1011074210645136e-3, -6.5313356943676485e-4, 1.0771293339194793e-3],
        [2.1310220139973432e7, 2.1431959110778780e5, -1.8803161482264430e6],
        [2.1302815233085830e7, 2.1863813953031204e5, -1.8874097808477948e6],
        [2.5422208823955162e10, -1.0851832163129303e10, 5.7939106975532684e5],
        [8.0387982011644645e0, 8.2504958313325008e-2, -8.7510283071447645e0],
        [-1.0574882872917236e1, 1.0793558062454863e1, 2.1863813953031128e-1],
        [1.5730284923742273e0, -4.8519547229790919e-2, -8.1311147852431532e-5],
        [1.4964077638195525e0, -1.5359888159727431e-2, 3.3717925134885911e-2],
        [6.3288356637203787e-8, 1.7674254713432971e-6, -5.5940130063712902e-6],
        [-1.2957525739867537e9, -3.3015629358273663e10, 1.0345582587693739e11],
        [4.0890025538347083e8, 1.0417529345529982e10, -3.2606729102890186e10],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [1.0, 0.0, 0.0],
        [0.0; 3],
        [0.0; 3],
    ];
    assert_eos_case("skye_c12_phase", &out_ph, &exp_ph, 1.0e-9);
}

#[test]
#[rustfmt::skip]
fn test_skye_alfa_blend_and_inverse_solvers_match_fortran_oracle() {
    let ht = HelmTable::builtin();
    let skye_cfg = SkyeConfig::default();

    let (a_in, dt_in, dr_in) = get_skye_alfa(ht, 4.0, 7.5, 12.0, 6.0);
    assert_eq!((a_in, dt_in, dr_in), (0.0, 0.0, 0.0));

    let (a_ed, dt_ed, dr_ed) = get_skye_alfa(ht, 3.45, 7.0, 12.0, 6.0);
    assert!((a_ed - 6.3634997198555698e-1).abs() < 1.0e-13);
    assert!((dt_ed - 0.0).abs() < 1.0e-13);
    assert!((dr_ed - (-10.0)).abs() < 1.0e-13);

    let (a_sm, dt_sm, dr_sm) = get_skye_alfa_simple(&skye_cfg, 2.35, 7.3);
    assert!((a_sm - 0.25).abs() < 1.0e-13);
    assert!((dt_sm - (-2.5)).abs() < 1.0e-13);
    assert!((dr_sm - (-5.0)).abs() < 1.0e-13);

    // eosdt_skye_helm_blend
    let (chem_id, xa, abar, zbar) = make_comp(0.0, 1.0, 0.0);
    let mut cfg = EosConfig {
        use_skye: true,
        use_pc: false,
        ..EosConfig::default()
    };
    cfg.skye.skye_use_ion_offsets = false;
    let rho_bl = flexmesa_math::exp10(3.45);
    let out_blend = eos_dt_get(ht, rho_bl, 1.0e7, &chem_id, &xa, &cfg).unwrap();
    let exp_blend: [[f64; 3]; 26] = [
        [4.2210418820363131e1, 1.2100334780180333e0, 4.6150838080474643e-1],
        [3.4540234690251864e1, -1.6100758671538529e-1, 5.4453093989554646e-1],
        [1.8908461259753668e1, -2.3108985544072880e-1, 3.3467235044345145e-1],
        [1.7142857142857142e0, 0.0, 0.0],
        [-6.9314705306508606e-1, -1.1967798112522132e-6, -3.7442553280671653e-36],
        [3.5323627321843363e0, 2.7048672208454505e0, -4.0734200453777287e0],
        [3.7957328168382026e-1, -2.6744105939924268e-2, 1.3121433169775871e-2],
        [1.4014486492686040e0, 2.4949553986568773e-1, -3.0879652057223367e-1],
        [4.6154995449889474e-1, -2.7015093609700713e-1, 4.3121760037306284e-1],
        [6.6085615973382726e7, -3.1895450552850068e7, 3.8428848869633988e7],
        [5.4508258739223808e7, -1.8710596668921947e7, 2.0480327365950957e7],
        [1.4553142497694409e11, -4.1650380972746948e10, -4.9361311760115700e10],
        [5.4508258739223816e0, -1.8710596668921866e0, -3.4027931373272846e0],
        [-1.2473253136645690e4, 1.7151653528279334e4, -4.9361311760115686e3],
        [1.6991154517385296e0, 6.5735202854833474e-2, -2.4742344687876134e-2],
        [1.6449328324463304e0, -2.0591720674408620e-2, 1.2904166910856227e-2],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [7.4326433673290881e-1, -6.9769381884226718e0, 0.0],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [2.5673566326709119e-1, 6.9769381884226718e0, 0.0],
        [0.0; 3],
        [0.0; 3],
    ];
    assert_eos_case("eosdt_skye_helm_blend", &out_blend, &exp_blend, 1.0e-8);

    // Inverse solvers (pure HELM: use_skye = false, use_pc = false)
    cfg.use_skye = false;
    cfg.use_pc = false;
    let fwd = eos_dt_get(ht, 1.0e4, 1.0e8, &chem_id, &xa, &cfg).unwrap();
    let log_pgas = fwd.res[I_LN_PGAS] / LN10;
    let log_e = fwd.res[I_LN_E] / LN10;
    let e_val = fwd.res[I_LN_E].exp();

    let pt = eos_pt_get(ht, &chem_id, &xa, log_pgas, 8.0, &cfg).unwrap();
    assert!((pt.rho - 1.0000000011641523e4).abs() / 1.0e4 < 1.0e-8);
    assert!((pt.log10_rho - 4.0000000005055849e0).abs() < 1.0e-8);
    assert!(
        (pt.dln_rho_dln_pgas_const_t - 9.3116984895460597e-1).abs() < 1.0e-8
    );
    assert!(
        (pt.dln_rho_dln_t_const_pgas - (-8.3501056279150876e-1)).abs() < 1.0e-8
    );

    let inv_t = eos_dt_get_t(
        ht,
        &chem_id,
        &xa,
        4.0,
        I_LN_E as isize,
        log_e * LN10,
        7.5,
        1.0e-12,
        1.0e-12,
        100,
        &cfg,
    )
    .unwrap();
    assert!((inv_t.log10_result - 7.9999999999693525e0).abs() < 1.0e-9);
    assert!((inv_t.eos.res[I_LN_PGAS] - fwd.res[I_LN_PGAS]).abs() < 1.0e-11);
    assert!((inv_t.eos.res[I_LN_PGAS] - 4.5395078603101041e1).abs() < 1.0e-8);

    let inv_rho = eos_dt_get_rho(
        ht,
        &chem_id,
        &xa,
        8.0,
        I_LN_PGAS as isize,
        log_pgas * LN10,
        3.5,
        1.0e-12,
        1.0e-12,
        100,
        &cfg,
    )
    .unwrap();
    assert!((inv_rho.log10_result - 3.9999999999999081e0).abs() < 1.0e-10);
    assert!((inv_rho.eos.res[I_LN_E] - fwd.res[I_LN_E]).abs() < 1.0e-11);
    assert!((inv_rho.eos.res[I_LN_E] - 3.6600933525797927e1).abs() < 1.0e-8);

    let gam = eos_gamma_de_get(abar, zbar, 1.0e4, e_val, 5.0 / 3.0);
    assert!(
        (gam.t - 7.5655283790336466e8).abs() / 7.5655283790336466e8 < 1.0e-13
    );
    assert!((gam.log10_t - 8.8788392646903720e0).abs() < 1.0e-13);
    let d_gam = (
        gam.dln_t_dln_e_c_rho,
        gam.dln_t_dlnd_c_e,
        gam.dln_pgas_dln_e_c_rho,
        gam.dln_pgas_dlnd_c_e,
    );
    assert_eq!(d_gam, (1.0, 0.0, 1.0, 1.0));
    assert!((gam.res[flexmesa_eos::I_MU] - 1.0 / abar).abs() < 1.0e-15);
    assert_eq!(gam.res[flexmesa_eos::I_LN_FREE_E], -1.0e99);
    assert_eq!(gam.res[flexmesa_eos::I_ETA], 0.0);
    assert_eq!(gam.res[flexmesa_eos::I_DS_DT], 0.0);
    assert_eq!(gam.res[flexmesa_eos::I_DS_DRHO], 0.0);
    assert_eq!(gam.d_dlnd[flexmesa_eos::I_LN_S], 0.0);
    assert_eq!(gam.d_dln_t[flexmesa_eos::I_LN_S], 0.0);
}
