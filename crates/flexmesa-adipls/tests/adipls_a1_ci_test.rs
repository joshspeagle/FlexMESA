// SPDX-License-Identifier: GPL-3.0-or-later
//! Self-contained always-on CI tests for `flexmesa-adipls` (Package A1, #171).
//!
//! Covers:
//! - Exact parsing of MESA 26.4.1 `adipls.c.pruned.in` and both
//!   `redistrb.c.pruned.in` variants, plus loud failure on malformed inputs.
//! - `fgong_amdl` and `store_amdl`, including `#479` explicit errors on unset
//!   `aa` rows `6..=9` (`DECISIONS #43`) and out-of-bounds `iriche = 1` shift.
//! - `set_adipls_controls` (`cadr_param`, `cadi_param`, `coutpt`), `sig_fac`,
//!   and `#480` caller argument shadowing documentation/verification.
//! - `spcout_adi`, `ModeRecord`, `ObsSt`, `Csummm`, and `AdiModeStore`.

use flexmesa_adipls::{
    dfsig_from_delta_nu, fgong_amdl, parse_adipls_pruned_in,
    parse_redistrb_pruned_in, resolve_store_model_flags, set_adipls_controls,
    sig_fac_adipls_support, sig_fac_astero_support, sig_from_nu_uhz,
    spcout_adi, spcout_adi_store, store_amdl, AdiFileUnit, AdiModeStore,
    AdiplsCallerFlags, AdiplsControlState, AdiplsError, AmdlAa,
    AsteroDefPulseFlags, Csummm, FgongModel, ObsSt, ADIPLS_DEFAULT_EPSPRC,
    IAA_ARG,
};
use flexmesa_const::{FOUR_THIRDS_PI, PI4, STANDARD_CGRAV};
use flexmesa_math::{exp, pow2, pow3};

const SAMPLE_ADIPLS_PRUNED_IN: &str = "\
2  'amdl.mesa'
9  'ttt.adipls.prt'
11 'agsm.mesa'
15 'ttt.adipls.ssm'
16 'ttt.adipls.fsm'
-1 ''
mod.osc.cst.int.out
,,,,,,,
,,1,,,,,,,,,,,,,,,,,,,,
   0,3,0,1,,,,,,,
    1,  220,   ,    1,10,,,,,,,,
,2,100,1000,,,,,,,,,,,,
,,,,,,,,,,,,,,,,
6.67430e-8
,,0,,
1,,,,,,,,,,,,,,,
5,0,0.99,,,,,,,,,,,,,,
,,,,,-1,,,,,,,,,,,,,,
9,10,,20,1,,,,,,,,
1,,,,,,,,,,,,,,,
0,,,,6,,,,,,
,,,,,-5000,100,,,,,,,,,,,,,,,,,,,,,,,,
";

const SAMPLE_REDISTRB_2001: &str = "\
2001,,,
24,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
30,,,,,,,,,
,,,,,,,,,,,
";

const SAMPLE_REDISTRB_2201: &str = "\
2201,,,
24,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
30,,,,10,,,,,
,,,,,,,,,,,
";

#[test]
fn test_parse_adipls_pruned_in_matches_mesa_26_4_1_controls() {
    let parsed = parse_adipls_pruned_in(SAMPLE_ADIPLS_PRUNED_IN)
        .expect("parse adipls.c.pruned.in");

    assert_eq!(
        parsed.files,
        vec![
            AdiFileUnit {
                unit: 2,
                path: String::from("amdl.mesa")
            },
            AdiFileUnit {
                unit: 9,
                path: String::from("ttt.adipls.prt")
            },
            AdiFileUnit {
                unit: 11,
                path: String::from("agsm.mesa")
            },
            AdiFileUnit {
                unit: 15,
                path: String::from("ttt.adipls.ssm")
            },
            AdiFileUnit {
                unit: 16,
                path: String::from("ttt.adipls.fsm")
            },
        ]
    );
    assert_eq!(parsed.cntrd_raw, "mod.osc.cst.int.out");
    assert!(parsed.cntrd.model);
    assert!(parsed.cntrd.osc);
    assert!(parsed.cntrd.cst);
    assert!(parsed.cntrd.int);
    assert!(parsed.cntrd.out);
    assert!(!parsed.cntrd.dsn);
    assert!(!parsed.cntrd.rot);
    assert!(!parsed.cntrd.dgn);

    assert_eq!(parsed.mod_ctrl.ifind, -1);
    assert_eq!(parsed.mod_ctrl.in_step, 1);
    assert_eq!(parsed.mod_ctrl.imdmod, 1);

    assert_eq!(parsed.osc.el.to_bits(), 0.0_f64.to_bits());
    assert_eq!(parsed.osc.nsel, 3);
    assert_eq!(parsed.osc.els1.to_bits(), 0.0_f64.to_bits());
    assert_eq!(parsed.osc.dels.to_bits(), 1.0_f64.to_bits());
    assert_eq!(parsed.osc.itrsig, 1);
    assert_eq!(parsed.osc.sig1.to_bits(), 220.0_f64.to_bits());
    assert_eq!(parsed.osc.istsig, 1);
    assert_eq!(parsed.osc.inomd1, 1);
    assert_eq!(parsed.osc.itrds, 10);
    assert_eq!(parsed.osc.nsig, 2);
    assert_eq!(parsed.osc.iscan, 100);
    assert_eq!(parsed.osc.sig2.to_bits(), 1000.0_f64.to_bits());
    assert_eq!(parsed.osc.eltrw2.to_bits(), (-1.0_f64).to_bits());
    assert_eq!(parsed.osc.sgtrw2.to_bits(), (-1.0_f64).to_bits());

    assert_eq!(parsed.cst.cgrav.to_bits(), STANDARD_CGRAV.to_bits());

    assert_eq!(parsed.int_ctrl.icow, 0);
    assert_eq!(parsed.int_ctrl.istsbc, 1);
    assert_eq!(parsed.int_ctrl.mdintg, 5);
    assert_eq!(parsed.int_ctrl.iriche, 0);
    assert_eq!(parsed.int_ctrl.xfit.to_bits(), 0.99_f64.to_bits());
    assert_eq!(
        parsed.int_ctrl.eps.to_bits(),
        (2.0_f64 * ADIPLS_DEFAULT_EPSPRC).to_bits()
    );
    assert_eq!(parsed.int_ctrl.itsord, -1);

    assert_eq!(parsed.out.istdpr, 9);
    assert_eq!(parsed.out.nout, 10);
    assert_eq!(parsed.out.irsord, 20);
    assert_eq!(parsed.out.iekinr, 1);
    assert_eq!(parsed.out.iper, 1);
    assert_eq!(parsed.out.irotkr, 0);
    assert_eq!(parsed.out.npgmkr, 50);
    assert_eq!(parsed.out.ispcpr, 6);
    assert_eq!(parsed.out.iorwn1, -5000);
    assert_eq!(parsed.out.iorwn2, 100);
}

#[test]
fn test_parse_redistrb_pruned_in_both_variants_and_srdist_presets() {
    let r2001 = parse_redistrb_pruned_in(SAMPLE_REDISTRB_2001)
        .expect("parse redistrb 2001");
    assert_eq!(r2001.nn, 2001);
    assert_eq!(r2001.icase, 24);
    assert_eq!(r2001.nsmth, 9);
    assert_eq!(r2001.ndisc, -60);
    assert_eq!(r2001.ndisca, 60);
    assert_eq!(r2001.dlxdsc.to_bits(), f64::from(0.0002_f32).to_bits());
    assert_eq!(r2001.cg.to_bits(), f64::from(0.0001_f32).to_bits());
    assert_eq!(r2001.cx.to_bits(), 5.0_f64.to_bits());
    assert_eq!(r2001.ca.to_bits(), f64::from(0.005_f32).to_bits());
    assert_eq!(r2001.cdg.to_bits(), f64::from(0.01_f32).to_bits());
    assert_eq!(r2001.adda4.to_bits(), 0.5_f64.to_bits());
    assert_eq!(r2001.nout, 30);
    assert_eq!(r2001.iresa4, 0);
    assert_eq!(r2001.itstcg, 0);

    let r2201 = parse_redistrb_pruned_in(SAMPLE_REDISTRB_2201)
        .expect("parse redistrb 2201");
    assert_eq!(r2201.nn, 2201);
    assert_eq!(r2201.icase, 24);
    assert_eq!(r2201.nout, 30);
    assert_eq!(r2201.iresa4, 10);
}

#[test]
fn test_parsers_handle_all_cntrd_sections_repeats_and_reject_bad_inputs() {
    let custom = "\
2 '0'
-1 ''
dsn.mod.osc.rot.cst.int.out.dgn
11,15,4,12,13,20
-1,2.0d0,2,1,0
0.0,0,0
2.0,0,2.0,1.0,2*0.0,1,1
0,100.0,1,1,10
0.0,1,50,500.0
4*0.0
1,1.0,1,0.0,0.5
6.67430d-8
0,0,2,1.0
1,0.0,0,0.0
5,0,0.95,0.5,1.0d-12,1.0d-5,8,0.0
0.001,0.1,2,0.0,3,0
6,50,0,20,1
1,1,2,0,0
0,0,0,50,6
-1,0.0,-1.0,0.0,-1.0,0,-1,0.0,-1.0
0,0,0,0,0
1,2,2,0
";
    let parsed = parse_adipls_pruned_in(custom).expect("parse full cntrd");
    assert_eq!(parsed.files[0].path, "/dev/null");
    assert_eq!(parsed.rot.irotsl, 1);
    assert_eq!(parsed.rot.em, 1.0);
    assert_eq!(parsed.dgn.imissl, 2);

    // No executable section in cntrd must error loudly.
    let bad_cntrd = "2 'a'\n-1 ''\ndsn.rot\n";
    assert!(matches!(
        parse_adipls_pruned_in(bad_cntrd),
        Err(AdiplsError::Parse { .. })
    ));

    // Truncated redistrb input must error loudly.
    assert!(matches!(
        parse_redistrb_pruned_in("2001,,,\n"),
        Err(AdiplsError::Parse { .. })
    ));

    // Non-positive nn in redistrb must error loudly.
    assert!(matches!(
        parse_redistrb_pruned_in("0,,,\n24,,,\n,,,\n30,,,\n,,,\n"),
        Err(AdiplsError::Parse { .. })
    ));

    // Extra non-empty value past expected field count must error loudly.
    assert!(matches!(
        parse_redistrb_pruned_in("2001,0,999\n24,,,\n,,,\n30,,,\n,,,\n"),
        Err(AdiplsError::Parse { .. })
    ));
}

fn build_sample_3zone_fgong() -> FgongModel {
    // Minimal 3-point surface-to-center FGONG model (reversed by fgong_amdl)
    // with ivar = 15 and glob.len() = 15, glob(11) < 0.
    let ivar = 15_usize;
    let nn_in = 3_usize;
    let m_tot = 1.9884e33_f64;
    let r_tot = 6.957e10_f64;
    let mut glob = vec![0.0_f64; 15];
    glob[0] = m_tot;
    glob[1] = r_tot;
    glob[10] = -80.0_f64;
    glob[11] = -90.0_f64;

    // Zone order in input: surface (r = r_tot), middle (r = 0.5*r_tot), center (r = 0)
    let mut var = vec![0.0_f64; ivar * nn_in];
    let set_zone = |buf: &mut [f64],
                    z: usize,
                    r: f64,
                    lnq: f64,
                    p: f64,
                    rho: f64,
                    g1: f64,
                    a_brunt: f64| {
        let base = z * ivar;
        buf[base] = r;
        buf[base + 1] = lnq;
        buf[base + 3] = p;
        buf[base + 4] = rho;
        buf[base + 9] = g1;
        buf[base + 14] = a_brunt;
    };
    set_zone(
        &mut var,
        0,
        r_tot,
        0.0_f64,
        1.0e5_f64,
        1.0e-7_f64,
        5.0 / 3.0,
        20.0,
    );
    set_zone(
        &mut var,
        1,
        0.5 * r_tot,
        -0.5_f64,
        1.0e14_f64,
        1.0_f64,
        5.0 / 3.0,
        1.5,
    );
    set_zone(
        &mut var,
        2,
        0.0_f64,
        -100.0_f64,
        2.0e17_f64,
        100.0_f64,
        5.0 / 3.0,
        0.0,
    );

    FgongModel::new(glob, ivar, nn_in, var).expect("fgong")
}

#[test]
fn test_fgong_amdl_exact_math_and_unset_rows() {
    let fg = build_sample_3zone_fgong();
    let m_tot = fg.glob[0];
    let r_tot = fg.glob[1];
    let fg_out = fgong_amdl(STANDARD_CGRAV, &fg).expect("fgong_amdl");
    assert_eq!(fg_out.nn, 3);

    let x_mid = (0.5_f64 * r_tot) / r_tot;
    let q_mid = exp(-0.5_f64);
    assert_eq!(fg_out.aa.get(1, 0).unwrap(), 0.0_f64);
    assert_eq!(
        fg_out.aa.get(2, 0).unwrap().to_bits(),
        (((FOUR_THIRDS_PI * 100.0_f64) * pow3(r_tot)) / m_tot).to_bits()
    );
    assert_eq!(fg_out.aa.get(6, 0).unwrap(), 3.0_f64);
    assert_eq!(fg_out.aa.get(1, 1).unwrap().to_bits(), x_mid.to_bits());
    assert_eq!(
        fg_out.aa.get(2, 1).unwrap().to_bits(),
        (q_mid / pow3(x_mid)).to_bits()
    );
    assert_eq!(
        fg_out.aa.get(3, 1).unwrap().to_bits(),
        ((((STANDARD_CGRAV * m_tot) * q_mid) * 1.0_f64)
            / (((5.0_f64 / 3.0_f64) * 1.0e14_f64) * (0.5_f64 * r_tot)))
            .to_bits()
    );
    assert_eq!(
        fg_out.aa.get(6, 1).unwrap().to_bits(),
        (((PI4 * 1.0_f64) * pow3(0.5_f64 * r_tot)) / (m_tot * q_mid)).to_bits()
    );
    assert_eq!(
        fg_out.data[4].to_bits(),
        (-(-80.0_f64) / (5.0_f64 / 3.0_f64)).to_bits()
    );
    assert_eq!(fg_out.data[5].to_bits(), 90.0_f64.to_bits());
    assert_eq!(fg_out.data[6], -1.0_f64);
    assert_eq!(fg_out.data[7], 0.0_f64);

    for row in 7..=IAA_ARG {
        assert!(!fg_out.aa.is_row_set(row).unwrap());
        assert_eq!(fg_out.aa.row(row), Err(AdiplsError::UnsetAmdlRow { row }));
    }
}

#[test]
fn test_amdl_aa_bounds_and_row_validation() {
    let mut custom_aa = AmdlAa::new_unset(2);
    assert_eq!(
        custom_aa.row(0),
        Err(AdiplsError::InvalidAmdlIndex {
            row: 0,
            zone: 0,
            nn: 2
        })
    );
    assert_eq!(
        custom_aa.get(1, 2),
        Err(AdiplsError::InvalidAmdlIndex {
            row: 1,
            zone: 2,
            nn: 2
        })
    );
    assert!(matches!(
        custom_aa.set_row(1, vec![1.0]),
        Err(AdiplsError::InvalidModel { .. })
    ));
}

#[test]
fn test_store_amdl_shift_479_unset_iggt_iriche_and_iturpr() {
    let fg = build_sample_3zone_fgong();
    let fg_out = fgong_amdl(STANDARD_CGRAV, &fg).expect("fgong_amdl");

    // store_amdl shifts rows 2..=6 -> 1..=5, sets row 10 = 1.0, leaves 6..=9 unset (#479).
    let st_out = store_amdl(fg_out.nn, 0, 0, fg_out.data, &fg_out.aa)
        .expect("store_amdl");
    assert_eq!(st_out.nn, 3);
    assert_eq!(st_out.iturpr, 0);
    assert_eq!(st_out.x, fg_out.aa.row(1).unwrap());
    for i in 1..=5 {
        assert_eq!(
            st_out.aa.row(i).unwrap(),
            fg_out.aa.row(i + 1).unwrap(),
            "shifted row {i}"
        );
    }
    assert_eq!(st_out.aa.row(10).unwrap(), &[1.0_f64, 1.0_f64, 1.0_f64]);
    for row in 6..=9 {
        assert!(!st_out.aa.is_row_set(row).unwrap());
        assert_eq!(st_out.aa.row(row), Err(AdiplsError::UnsetAmdlRow { row }));
        assert_eq!(
            st_out.aa.get(row, 0),
            Err(AdiplsError::UnsetAmdlRow { row })
        );
    }

    // `iggt == 1` triggers explicit UnsetAmdlRow { row: 6 } (#479).
    let mut d_iggt = fg_out.data;
    d_iggt[7] = 20.0_f64;
    assert_eq!(
        store_amdl(fg_out.nn, 0, 0, d_iggt, &fg_out.aa),
        Err(AdiplsError::UnsetAmdlRow { row: 6 })
    );

    // `iriche == 1` with even (nn - nsin) -> nn=3, nsin=1 (sincen=true, sinsur=false) => nn-nsin=2 (even) -> explicit error (DECISIONS #43).
    assert!(matches!(
        store_amdl(fg_out.nn, 1, 0, fg_out.data, &fg_out.aa),
        Err(AdiplsError::InvalidModel { .. })
    ));

    // `iturpr == 1` sets row 10 near surface (x >= 0.999) to 1/(x^3 * aa(1, n)).
    let st_turb =
        store_amdl(fg_out.nn, 0, 1, fg_out.data, &fg_out.aa).expect("iturpr=1");
    let expected_surf_r10 = 1.0_f64
        / (st_turb.x[2]
            * st_turb.x[2]
            * st_turb.x[2]
            * st_turb.aa.get(1, 2).unwrap());
    assert_eq!(st_turb.aa.get(10, 0).unwrap(), 1.0_f64);
    assert_eq!(st_turb.aa.get(10, 1).unwrap(), 1.0_f64);
    assert_eq!(
        st_turb.aa.get(10, 2).unwrap().to_bits(),
        expected_surf_r10.to_bits()
    );
}

#[test]
fn test_fgong_amdl_center_prepend_outer_chop_and_else_data56() {
    // Center point missing (r_inner > 1e6), outer aa(5,nn) <= 10.0 chopped,
    // and glob(11) == 0 triggers fallback data(5), data(6).
    let ivar = 20_usize;
    let nn_in = 3_usize;
    let m_tot = 2.0e33_f64;
    let r_tot = 7.0e10_f64;
    let mut glob = vec![0.0_f64; 15];
    glob[0] = m_tot;
    glob[1] = r_tot;
    glob[10] = 0.0_f64; // triggers else branch

    let mut var = vec![0.0_f64; ivar * nn_in];
    for (z, (r, lnq, p, rho, g1, a_brunt)) in [
        (
            2.0e6_f64,
            -20.0_f64,
            1.0e17_f64,
            80.0_f64,
            5.0 / 3.0,
            0.1_f64,
        ),
        (
            0.5 * r_tot,
            -0.7_f64,
            1.0e14_f64,
            1.0_f64,
            5.0 / 3.0,
            2.0_f64,
        ),
        (r_tot, 0.0_f64, 1.0e5_f64, 1.0e-7_f64, 5.0 / 3.0, 5.0_f64), // <= 10 -> chopped
    ]
    .into_iter()
    .enumerate()
    {
        let b = z * ivar;
        var[b] = r;
        var[b + 1] = lnq;
        var[b + 3] = p;
        var[b + 4] = rho;
        var[b + 9] = g1;
        var[b + 14] = a_brunt;
    }

    let fg = FgongModel::new(glob, ivar, nn_in, var).expect("fgong");
    let out = fgong_amdl(STANDARD_CGRAV, &fg).expect("fgong_amdl");
    // 3 -> +1 center prepend -> 4 -> -1 outer chop -> 3
    assert_eq!(out.nn, 3);
    assert_eq!(out.aa.get(1, 0).unwrap(), 0.0_f64);
    let expected_d5 = ((FOUR_THIRDS_PI * STANDARD_CGRAV)
        * pow2(80.0_f64 * r_tot))
        / (1.0e17_f64 * (5.0_f64 / 3.0_f64));
    assert_eq!(out.data[4].to_bits(), expected_d5.to_bits());
    assert!(out.data[5] > out.data[4]);

    // Prepend with ivar < 20 must fail loudly instead of out-of-bounds read.
    let small_fg = FgongModel::new(
        vec![
            m_tot, r_tot, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        15,
        2,
        {
            let mut v = vec![0.0; 30];
            v[0] = 2.0e6;
            v[15] = r_tot;
            v
        },
    )
    .unwrap();
    assert!(matches!(
        fgong_amdl(STANDARD_CGRAV, &small_fg),
        Err(AdiplsError::InvalidModel { .. })
    ));
}

#[test]
fn test_set_adipls_controls_and_480_shadowing() {
    let adipls_in = parse_adipls_pruned_in(SAMPLE_ADIPLS_PRUNED_IN).unwrap();
    let mut state = AdiplsControlState::from_input(&adipls_in);

    // Before set_adipls_controls: parsed values from adipls.c.pruned.in.
    assert_eq!(state.cadr_param.sig1, 220.0_f64);
    assert_eq!(state.cadr_param.sig2, 1000.0_f64);
    assert_eq!(state.cadi_param.iscan, 100);

    let m = 2.386_091_844_837_660_4e33_f64;
    let r = 8.171_130_695_061_145e10_f64;
    let sf_adi = sig_fac_adipls_support(r, STANDARD_CGRAV, m);
    let sf_ast = sig_fac_astero_support(r, STANDARD_CGRAV, m);
    let sig1 = sig_from_nu_uhz(sf_adi, 50.0_f64);
    let sig2 = sig_from_nu_uhz(sf_adi, 1000.0_f64);
    let dfsig = dfsig_from_delta_nu(sf_ast, 0.0_f64);

    set_adipls_controls(
        &mut state, 0, 0, 0.0_f64, 1.0_f64, 1, 200, sig1, sig2, dfsig, 2, 0, 0,
        0, 0,
    );

    assert_eq!(state.cadr_param.el.to_bits(), 0.0_f64.to_bits());
    assert_eq!(state.cadr_param.els1.to_bits(), 0.0_f64.to_bits());
    assert_eq!(state.cadr_param.dels.to_bits(), 1.0_f64.to_bits());
    assert_eq!(state.cadr_param.sig1.to_bits(), sig1.to_bits());
    assert_eq!(state.cadr_param.sig2.to_bits(), sig2.to_bits());
    assert_eq!(state.cadr_param.dfsig.to_bits(), 0.0_f64.to_bits());
    // Untouched fields from adipls.c.pruned.in remain intact.
    assert_eq!(state.cadr_param.eltrw2, -1.0_f64);
    assert_eq!(state.cadr_param.sgtrw2, -1.0_f64);
    assert_eq!(state.cadi_param.nsig1, 1);
    assert_eq!(state.cadi_param.nsig2, 1);
    assert_eq!(state.cadi_param.istsig, 1);
    assert_eq!(state.cadi_param.inomd1, 1);
    // Updated int and coutpt fields.
    assert_eq!(state.cadi_param.nsel, 0);
    assert_eq!(state.cadi_param.itrsig, 1);
    assert_eq!(state.cadi_param.nsig, 2);
    assert_eq!(state.cadi_param.iscan, 200);
    assert_eq!(state.coutpt.to_array(), [0, 0, 0, 0]);

    // #480 shadowing check: caller's add_center_point / keep_surface_point are
    // shadowed by astero_def while add_atmosphere / do_redistribute_mesh pass through.
    let caller = AdiplsCallerFlags {
        caller_add_center_point: false,
        caller_keep_surface_point: true,
        add_atmosphere: true,
        do_redistribute_mesh: false,
    };
    let astero_def = AsteroDefPulseFlags::default();
    let resolved = resolve_store_model_flags(caller, astero_def);
    assert!(resolved.add_center_point);
    assert!(!resolved.keep_surface_point);
    assert!(resolved.add_atmosphere);
    assert!(!resolved.do_redistribute_mesh);
}

#[test]
fn test_spcout_adi_and_mode_output_structs() {
    let obs = ObsSt::from_array([
        2.0_f64,
        -3.0_f64,
        1250.5_f64,
        1.25e-5_f64,
        0.0_f64,
        0.0_f64,
        0.0_f64,
        0.0_f64,
        0.0_f64,
        0.0_f64,
    ]);
    let mut csum = Csummm::zeroed();
    csum.set_f64(38, -2.0_f64).unwrap();
    csum.set_icsumm([1, 1738, 5, 1, 10010, 3, 1, 0]);
    assert_eq!(csum.icsumm(), [1, 1738, 5, 1, 10010, 3, 1, 0]);
    assert_eq!(csum.as_bits()[38], 0x0000_06CA_0000_0001_u64);
    assert_eq!(csum.as_bits()[40], 0x0000_0003_0000_271A_u64);
    assert_eq!(
        csum.get_f64(0),
        Err(AdiplsError::InvalidCsummmIndex { index: 0 })
    );
    assert_eq!(
        csum.get_f64(51),
        Err(AdiplsError::InvalidCsummmIndex { index: 51 })
    );

    // Negative order: obs_st(2) = -3.0 -> int(-3.0 + 0.5) = -2 (Fortran truncation toward zero).
    let rec = spcout_adi(&obs, &csum).expect("spcout_adi");
    assert_eq!(rec.el, 2);
    assert_eq!(rec.order, -2);
    assert_eq!(rec.em, -2);
    assert_eq!(rec.cyclic_freq_uhz.to_bits(), 1250.5_f64.to_bits());
    assert_eq!(rec.inertia.to_bits(), (1.25e-5_f64 * PI4).to_bits());
    assert_eq!(rec.growth_rate, 0.0_f64);

    let mut store = AdiModeStore::new();
    let mut obs_hi = obs;
    obs_hi.0[2] = 1500.0_f64;
    spcout_adi_store(&obs_hi, &csum, &mut store).unwrap();
    spcout_adi_store(&obs, &csum, &mut store).unwrap();
    assert_eq!(store.len(), 2);
    store.sort_by_frequency();
    assert_eq!(store.records[0].cyclic_freq_uhz, 1250.5_f64);
    assert_eq!(store.records[1].cyclic_freq_uhz, 1500.0_f64);

    let mut bad_obs = obs;
    bad_obs.0[0] = f64::NAN;
    assert!(matches!(
        spcout_adi(&bad_obs, &csum),
        Err(AdiplsError::ConversionOverflow { .. })
    ));
}
