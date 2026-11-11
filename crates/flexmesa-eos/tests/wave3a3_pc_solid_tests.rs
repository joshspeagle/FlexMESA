// SPDX-License-Identifier: LGPL-3.0-or-later
//! Golden parity integration tests for Wave 3a.3 solid/quantum-melting `PC`
//! regimes, `get_pc_alfa`, and `eosDT_get` `PC`/`HELM` blending against
//! upstream Fortran MESA (`pc_eos.f90`, `eospc_eval.f90`, and `eosdt_eval.f90`).

#![allow(
    clippy::approx_constant,
    clippy::excessive_precision,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_eos::{
    eos_dt_get, eval_pc_eos, get_pc_alfa, EosConfig, EosEvalOutput, HelmTable,
    PcConfig, EOS_BASIC_RESULT_NAMES, I_FRAC_FREE_EOS, I_FRAC_HELM, I_FRAC_PC,
    NUM_EOS_BASIC_RESULTS,
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
            let col_tol = if col == 0 { tol } else { tol.max(5.0e-9) };
            let scale = e.abs().max(1.0);
            let rel_err = (g - e).abs() / scale;
            assert!(
                rel_err < col_tol,
                "{label} [{idx}: {name}] col {col}: got {g:.16e}, exp {e:.16e} \
                 (rel_err = {rel_err:.3e} >= {col_tol:.1e})"
            );
        }
    }
}

#[test]
fn test_pc_solid_and_quantum_melting_regimes_match_fortran_oracle() {
    let cfg = PcConfig::default();

    // 5. pc_c12_sol: T = 1e6 K, rho = 1e6 g/cm^3
    let (chem_id, xa, abar, zbar) = make_comp(0.0, 1.0, 0.0);
    let out_sol =
        eval_pc_eos(&cfg, 1.0e6, 1.0e6, abar, zbar, &chem_id, &xa).unwrap();
    let exp_sol: [[f64; 3]; 26] = [
        [
            5.1590246240472943e1,
            1.5729539071989946e0,
            3.1572377139226068e-4,
        ],
        [
            3.8239024169244395e1,
            6.2837937755835604e-1,
            3.9641892159663912e-4,
        ],
        [
            1.6516000641655559e1,
            -5.4554028553473366e-1,
            1.0770591605987894e0,
        ],
        [1.7142857142857142e0, 0.0, 0.0],
        [-6.9314718055994529e-1, 0.0, 0.0],
        [
            1.6667398391180764e3,
            9.8926346762011315e2,
            -1.6667412312435770e3,
        ],
        [
            3.2197873383256836e-1,
            7.0277000644259607e-3,
            -3.0080608062463695e-3,
        ],
        [
            1.5729531474352512e0,
            -4.8466096620122405e-2,
            -2.6374870396389391e-4,
        ],
        [
            3.1935660743058540e-4,
            -2.6311629031980881e-4,
            4.7175793310095339e-4,
        ],
        [
            1.6035857686652673e7,
            -3.8799632215831312e6,
            7.8100953276634384e6,
        ],
        [
            1.6034208782971267e7,
            -3.8782417269147285e6,
            7.8068718682797328e6,
        ],
        [
            2.5422634010036266e10,
            -1.0849945276655396e10,
            -3.9706307486259779e6,
        ],
        [
            1.6034208782971266e1,
            -3.8782417269147280e0,
            -8.2273369146915343e0,
        ],
        [
            -8.1214797976300890e0,
            1.0159492259856700e1,
            -3.8782449936838890e0,
        ],
        [
            1.5731149046052240e0,
            -4.8600834355243809e-2,
            -2.6312717019010234e-5,
        ],
        [
            1.5065095451579316e0,
            -4.5930553924642178e-3,
            -4.7404974235744470e-3,
        ],
        [1.0, 0.0, 0.0],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [1.0, 0.0, 0.0],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
    ];
    assert_eos_case("pc_c12_sol", &out_sol, &exp_sol, 1.0e-11);

    // 6. pc_co_sol: 50/50 C12/O16 at T = 1e6 K, rho = 1e6 g/cm^3
    let (chem_id_co, xa_co, abar_co, zbar_co) = make_comp(0.0, 0.5, 0.5);
    let out_co_sol =
        eval_pc_eos(&cfg, 1.0e6, 1.0e6, abar_co, zbar_co, &chem_id_co, &xa_co)
            .unwrap();
    let exp_co_sol: [[f64; 3]; 26] = [
        [
            5.1587369810520904e1,
            1.5736993529215988e0,
            2.7730750911646895e-4,
        ],
        [
            3.8233169185235894e1,
            6.3019230343475641e-1,
            3.4795875673775020e-4,
        ],
        [
            1.6660055990713047e1,
            -4.1410566244721209e-1,
            8.1375355055563803e-1,
        ],
        [1.7454545454545454e0, 0.0, 0.0],
        [-6.9314718055994529e-1, 0.0, 0.0],
        [
            1.6668970952006880e3,
            9.8934782577686201e2,
            -1.6668984872175788e3,
        ],
        [
            3.2338277882873762e-1,
            6.4529211869610243e-3,
            -1.9237946561200751e-3,
        ],
        [
            1.5736983194534988e0,
            -4.8812400223315400e-2,
            -2.3114797966810149e-4,
        ],
        [
            2.8082197006249100e-4,
            -2.3048027736964521e-4,
            4.1092260804663104e-4,
        ],
        [
            1.3992739747278353e7,
            -3.3018939748301194e6,
            6.5718059790847525e6,
        ],
        [
            1.3991469024757113e7,
            -3.3005765507160109e6,
            6.5693573038839959e6,
        ],
        [
            2.5350589805054558e10,
            -1.0801128385497511e10,
            -3.3901477354910127e6,
        ],
        [
            1.3991469024757112e1,
            -3.3005765507160110e0,
            -7.4221117208731169e0,
        ],
        [
            -7.1210023051976865e0,
            8.8801401108931994e0,
            -3.3010290612693267e0,
        ],
        [
            1.5738412446811949e0,
            -4.8931295492473770e-2,
            -2.2859955748212749e-5,
        ],
        [
            1.5089531551402839e0,
            -5.6676647953298603e-3,
            -3.0351398921128090e-3,
        ],
        [1.0, 0.0, 0.0],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [1.0, 0.0, 0.0],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
    ];
    assert_eos_case("pc_co_sol", &out_co_sol, &exp_co_sol, 1.0e-11);

    // 7. pc_he4_qmelt: pure He4 at T = 1e5 K, rho = 1e8 g/cm^3 (quantum melting RS < RS2)
    let (chem_id_he, xa_he, abar_he, zbar_he) = make_comp(1.0, 0.0, 0.0);
    let out_he =
        eval_pc_eos(&cfg, 1.0e8, 1.0e5, abar_he, zbar_he, &chem_id_he, &xa_he)
            .unwrap();
    let exp_he: [[f64; 3]; 26] = [
        [
            5.8379839553945864e1,
            1.4084156086233177e0,
            -5.8634139029530449e-2,
        ],
        [
            4.0767321618456592e1,
            4.7211458010994189e-1,
            -5.2290459158946444e-2,
        ],
        [
            2.5608215152707853e1,
            1.0022322179717038e0,
            -2.0041893268860962e0,
        ],
        [1.3333333333333333e0, 0.0, 0.0],
        [-6.9314718055994529e-1, 0.0, 0.0],
        [
            1.6888403534656495e5,
            7.0924220765083432e4,
            -1.6888403535742022e5,
        ],
        [
            3.6236598162569417e-1,
            4.0416534764203824e-3,
            1.9345752477727847e-3,
        ],
        [
            1.4084156098090921e0,
            1.8156769368626278e-3,
            -3.4713394564854172e-2,
        ],
        [
            -5.8634139139081269e-2,
            -3.4713394267823161e-2,
            5.5245281921083231e-2,
        ],
        [
            -2.5960585084880963e11,
            -2.5649233782000034e11,
            5.1441654564973999e11,
        ],
        [
            -2.6512170087527844e11,
            -2.6526912592831494e11,
            5.3051396762323761e11,
        ],
        [
            2.3921481226236043e9,
            -1.3367171898228533e9,
            -2.6509683339830744e8,
        ],
        [
            -2.6512170087527847e6,
            -2.6526912592831496e6,
            7.9563566849851608e6,
        ],
        [
            1.3249293658455988e3,
            5.9548131218391542e-1,
            -2.6509683364585449e3,
        ],
        [
            1.3791135600221560e0,
            -1.5529003741284460e-2,
            -7.1102691044876353e-3,
        ],
        [
            1.4997438389507343e0,
            -5.3283570137584778e-5,
            9.1479313416162959e-5,
        ],
        [1.0, 0.0, 0.0],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [1.0, 0.0, 0.0],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
    ];
    assert_eos_case("pc_he4_qmelt", &out_he, &exp_he, 1.0e-11);
}

#[test]
fn test_pc_alfa_and_eosdt_pc_helm_blend_match_fortran_oracle() {
    let mut pc_cfg = PcConfig::default();
    let (a1, dt1, dr1) = get_pc_alfa(&pc_cfg, 2.9, 6.5, 12.0, 6.0);
    assert!((a1 - 4.9748743718592991e-1).abs() < 1.0e-14);
    assert!((dt1 - 0.0).abs() < 1.0e-14);
    assert!((dr1 - (-5.0251256281406960e0)).abs() < 1.0e-13);

    let (a2, dt2, dr2) = get_pc_alfa(&pc_cfg, 3.5, 8.35, 12.0, 6.0);
    assert!((a2 - 1.0).abs() < 1.0e-14);
    assert!((dt2 - 0.0).abs() < 1.0e-14);
    assert!((dr2 - 0.0).abs() < 1.0e-14);

    pc_cfg.pc_use_gamma_limit_instead_of_t = false;
    let (a3, dt3, dr3) = get_pc_alfa(&pc_cfg, 3.3, 7.65, 12.0, 6.0);
    assert!((a3 - 0.5).abs() < 1.0e-14);
    assert!((dt3 - 10.0).abs() < 1.0e-13);
    assert!((dr3 - 0.0).abs() < 1.0e-14);

    // eosdt_pc_helm_blend (explicitly testing PC + HELM with 4D tables disabled)
    let ht = HelmTable::builtin();
    let (chem_id, xa, _, _) = make_comp(0.0, 1.0, 0.0);
    let mut cfg = EosConfig {
        use_skye: false,
        use_pc: true,
        use_free_eos: false,
        use_opal_scvh: false,
        ..EosConfig::default()
    };
    cfg.pc.pc_use_gamma_limit_instead_of_t = true;
    let rho = 10.0_f64.powf(2.9);
    let t = 10.0_f64.powf(6.5);
    let out = eos_dt_get(ht, rho, t, &chem_id, &xa, &cfg).unwrap();
    let exp_blend: [[f64; 3]; 26] = [
        [
            3.9883224524523321e1,
            1.3148358857235110e0,
            3.6632019057710008e-1,
        ],
        [
            3.3276600391254107e1,
            -2.4470291565680213e-1,
            5.3833020739962811e-1,
        ],
        [
            1.8794998833785847e1,
            -2.1237573626515882e-1,
            3.3017982807471979e-1,
        ],
        [1.7142857142857140e0, 0.0, 0.0],
        [
            -6.9314709560120114e-1,
            -7.0187528455889492e-7,
            9.0091655421012968e-35,
        ],
        [
            4.9766347414178558e0,
            3.5582811524670035e0,
            -5.3448967484699761e0,
        ],
        [
            3.6432967700139346e-1,
            -2.4174437817462424e-2,
            1.9782228069943369e-2,
        ],
        [
            1.5132846380390128e0,
            2.6362186634374574e-1,
            -3.0882185385852584e-1,
        ],
        [
            3.6629728896189995e-1,
            -2.9047134306279054e-1,
            4.1339969850349334e-1,
        ],
        [
            5.5406252210586250e7,
            -3.2459269612740099e7,
            3.5693454505720459e7,
        ],
        [
            4.8012311153379574e7,
            -2.1775349561970860e7,
            2.2186161781576522e7,
        ],
        [
            2.1043373135298892e11,
            -4.7841396388961914e10,
            -6.0131880554576202e10,
        ],
        [
            1.5182825897338509e1,
            -6.8859701462178080e0,
            -8.1669455206624271e0,
        ],
        [
            -3.8455791952760657e4,
            5.6838800738894788e4,
            -1.9018165660815881e4,
        ],
        [
            1.7463250684322569e0,
            7.3150587788213706e-2,
            -3.8289113610860598e-2,
        ],
        [
            1.6362102161784289e0,
            -1.5519537919078452e-2,
            2.0583768400153912e-2,
        ],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [4.9528902403170577e-1, -4.0917639635928058e0, 0.0],
        [0.0; 3],
        [0.0; 3],
        [5.0471097596829417e-1, 4.0917639635928058e0, 0.0],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
    ];
    assert_eos_case("eosdt_pc_helm_blend", &out, &exp_blend, 1.0e-11);

    // With full data-v1 FreeEOS active, Z = 1.0 blends PC + FreeEOS directly
    let mut cfg_free = cfg;
    cfg_free.use_free_eos = true;
    cfg_free.okay_to_convert_ierr_to_skip = false;
    let out_free = eos_dt_get(ht, rho, t, &chem_id, &xa, &cfg_free).unwrap();
    assert!((out_free.res[I_FRAC_FREE_EOS] - 4.9528902403170577e-1).abs() < 1.0e-14);
    assert!((out_free.res[I_FRAC_PC] - 5.0471097596829417e-1).abs() < 1.0e-14);
    assert_eq!(out_free.res[I_FRAC_HELM], 0.0);
}
