// SPDX-License-Identifier: LGPL-3.0-or-later
//! Golden parity verification for `flexmesa-rates` against all 7 sections of
//! upstream MESA `rates/test/test_output` (`tests/fixtures/golden/rates/test_output.txt`).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::excessive_precision,
    clippy::similar_names,
    clippy::unreadable_literal
)]

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{KEV, LN2};
use flexmesa_eos::{eval_helm_eos, I_ETA};
use flexmesa_rates::{
    eval_ecapture_pair, eval_fl_epsnuc_3alf, eval_mui_coulomb, eval_vs_coulomb,
    eval_weak_reaction_pair, eval_weaklib_one, screen_pair, set_raw_rates,
    CoulombMuiOption, CoulombVsOption, RatesCatalog, ScreeningMode, TFactors,
    NUM_BUNDLED_REACTIONS,
};

const GOLDEN_TXT: &str =
    include_str!("../../../tests/fixtures/golden/rates/test_output.txt");

fn parse_f64_fortran(s: &str) -> f64 {
    let mut t = s.replace(['D', 'd'], "E");
    if t.len() > 4 && !t.contains('E') {
        let bytes = t.as_bytes();
        let sign_pos = t.len() - 4;
        if (bytes[sign_pos] == b'+' || bytes[sign_pos] == b'-')
            && bytes[sign_pos - 1].is_ascii_digit()
        {
            t.insert(sign_pos, 'E');
        }
    }
    t.parse::<f64>().unwrap()
}

fn rel_err(actual: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        actual.abs()
    } else {
        ((actual - expected) / expected).abs()
    }
}

#[test]
fn test_section1_screening_modes_match_golden_oracle() {
    let cat = NuclideCatalog::builtin_ref();
    let log_t = 7.7110722845770692_f64;
    let log_rho = 4.5306372623742392_f64;
    let temp = 10.0_f64.powf(log_t);
    let den = 10.0_f64.powf(log_rho);
    let zbar = 2.0156203413143605_f64;
    let abar = 4.0312406826287210_f64;
    let z2bar = 4.1624054993206103_f64;
    let he4 = cat.get_by_name("he4").unwrap();
    let c12 = cat.get_by_name("c12").unwrap();
    let z1 = he4.z as f64;
    let a1 = he4.w;
    let z2 = c12.z as f64;
    let a2 = c12.w;

    let sal = screen_pair(
        ScreeningMode::Salpeter,
        temp,
        den,
        zbar,
        abar,
        z2bar,
        a1,
        z1,
        a2,
        z2,
    );
    assert!(rel_err(sal.scor, 4.0373162996606471) < 1e-13);
    assert!(rel_err(sal.dscor_dt, -1.6438665008313957e-7) < 1e-13);
    assert!(rel_err(sal.dscor_dd, 8.3019538122528578e-5) < 1e-13);

    let ext = screen_pair(
        ScreeningMode::Extended,
        temp,
        den,
        zbar,
        abar,
        z2bar,
        a1,
        z1,
        a2,
        z2,
    );
    assert!(rel_err(ext.scor, 2.4319226633761173) < 1e-13);
    assert!(rel_err(ext.dscor_dt, -4.2425820200522379e-8) < 1e-13);
    assert!(rel_err(ext.dscor_dd, 2.1424210372719004e-5) < 1e-13);

    let chug = screen_pair(
        ScreeningMode::Chugunov,
        temp,
        den,
        zbar,
        abar,
        z2bar,
        a1,
        z1,
        a2,
        z2,
    );
    assert!(rel_err(chug.scor, 2.3827727933330571) < 1e-13);
    assert!(rel_err(chug.dscor_dt, -4.6135154522099626e-8) < 1e-13);
    assert!(rel_err(chug.dscor_dd, 2.3297997100016832e-5) < 1e-13);

    let none = screen_pair(
        ScreeningMode::NoScreening,
        temp,
        den,
        zbar,
        abar,
        z2bar,
        a1,
        z1,
        a2,
        z2,
    );
    assert!((none.scor - 1.0).abs() < 1e-15);
    assert!(none.dscor_dt.abs() < 1e-15);
    assert!(none.dscor_dd.abs() < 1e-15);
}

#[test]
fn test_section2_weak_tables_282_rows_match_golden_oracle() {
    let lines: Vec<&str> = GOLDEN_TXT.lines().collect();
    let rcat = RatesCatalog::builtin();
    let ncat = NuclideCatalog::builtin_ref();
    let o14_id = ncat.get_nuclide_index("o14").unwrap() as u16;
    let n14_id = ncat.get_nuclide_index("n14").unwrap() as u16;
    let (hl_o14, qneu_o14) = rcat.weak_info_by_ids(o14_id, n14_id).unwrap();
    assert!(rel_err(hl_o14, 70.64) < 1e-14);
    assert!(rel_err(qneu_o14, 2.22) < 1e-14);

    let sec2_start = lines
        .iter()
        .position(|l| {
            l.contains("halflife") && l.contains("Qneu") && l.contains("Qtotal")
        })
        .unwrap()
        + 1;
    let t9 = (10.0_f64.powf(9.0e8_f64.log10())) * 1.0e-9;
    let ye_rho = 0.5 * (10.0_f64.powf(4.5e5_f64.log10()));
    let l_ye_rho = ye_rho.log10();
    let eta = 10.0;

    let mut rows = Vec::new();
    for tab in &rcat.weak_tables {
        let pair = (tab.lhs_name.as_str(), tab.rhs_name.as_str());
        let w = if pair == ("mg24", "na24") || pair == ("na24", "mg24") {
            eval_ecapture_pair(
                tab.lhs_name.as_str(),
                tab.rhs_name.as_str(),
                t9,
                ye_rho,
                eta,
                CoulombMuiOption::None,
                CoulombVsOption::None,
                10.909090909090908,
                21.818181818181817,
            )
            .unwrap()
        } else {
            eval_weaklib_one(tab, t9, l_ye_rho)
        };
        if w.qneu >= 1.0e-20 && w.lambda >= 1.0e-20 {
            rows.push((
                tab.lhs_name.as_str(),
                tab.rhs_name.as_str(),
                LN2 / w.lambda,
                w.qneu,
                w.q,
            ));
        }
    }
    assert_eq!(rows.len(), 282);
    for (idx, (lhs, rhs, hl, qneu, q)) in rows.iter().enumerate() {
        let toks: Vec<&str> =
            lines[sec2_start + idx].split_whitespace().collect();
        assert_eq!(toks[0], *lhs);
        assert_eq!(toks[1], *rhs);
        let hl_g = parse_f64_fortran(toks[2]);
        let qneu_g = parse_f64_fortran(toks[3]);
        let q_g = parse_f64_fortran(toks[4]);
        assert!(rel_err(*hl, hl_g) < 5e-6);
        assert!(rel_err(*qneu, qneu_g) < 5e-6);
        assert!(rel_err(*q, q_g) < 5e-6);
    }
}

#[test]
fn test_section3_ecapture_coulomb_special_and_suzuki_match_golden_oracle() {
    let zbar = 10.909090909090908_f64;
    let abar = 21.818181818181817_f64;
    let ye = zbar / abar;

    // Part A: do_test_coulomb at logT = 8.5, logRho = 9.5
    let t_coul = 10.0_f64.powf(8.5);
    let rho_coul = 10.0_f64.powf(9.5);
    let mu1 = eval_mui_coulomb(
        CoulombMuiOption::Pcr2009,
        t_coul,
        rho_coul,
        zbar,
        abar,
        12.0,
    );
    let mu2 = eval_mui_coulomb(
        CoulombMuiOption::Pcr2009,
        t_coul,
        rho_coul,
        zbar,
        abar,
        11.0,
    );
    let dmu_ev = (mu1 - mu2).abs() * KEV * t_coul;
    let vs = eval_vs_coulomb(
        CoulombVsOption::Itoh2002,
        t_coul,
        rho_coul,
        zbar,
        abar,
        12.0,
    );
    assert!(rel_err(mu1, -43.7453051893886027) < 1e-13);
    assert!(rel_err(mu2, -37.5095737212022584) < 1e-13);
    assert!(rel_err(dmu_ev, 169926.1797009507135954) < 1e-13);
    assert!(rel_err(vs, 0.0079074257877945) < 1e-13);

    // Part B: do_test_special_weak and do_test_suzuki at logT = 8.3, logRho = 9.8
    let log_t = 8.3_f64;
    let log_rho = 9.8_f64;
    let temp = 10.0_f64.powf(log_t);
    let rho = 10.0_f64.powf(log_rho);
    let t9 = temp * 1.0e-9;
    let ye_rho = ye * rho;

    let (helm_out, _, _) = eval_helm_eos(temp, rho, abar, zbar).unwrap();
    let eta = helm_out.res[I_ETA];

    let w_mg =
        eval_weak_reaction_pair("mg24", "na24", t9, ye_rho, false).unwrap();
    assert!(rel_err(w_mg.lambda, 1.4421177069786498e-20) < 1e-12);

    let sp_mg = eval_ecapture_pair(
        "mg24",
        "na24",
        t9,
        ye_rho,
        eta,
        CoulombMuiOption::Pcr2009,
        CoulombVsOption::Itoh2002,
        zbar,
        abar,
    )
    .unwrap();
    assert!(rel_err(sp_mg.lambda, 3.8507552351991037e-3) < 1e-12);

    let suz_mg =
        eval_weak_reaction_pair("mg24", "na24", t9, ye_rho, true).unwrap();
    assert!(rel_err(suz_mg.lambda, 4.4797491275210752e-3) < 1e-12);

    let w_na =
        eval_weak_reaction_pair("na24", "mg24", t9, ye_rho, false).unwrap();
    assert!(rel_err(w_na.lambda, 2.3600389877416182e-58) < 1e-12);

    let sp_na = eval_ecapture_pair(
        "na24",
        "mg24",
        t9,
        ye_rho,
        eta,
        CoulombMuiOption::Pcr2009,
        CoulombVsOption::Itoh2002,
        zbar,
        abar,
    )
    .unwrap();
    assert!(rel_err(sp_na.lambda, 5.3938055004878743e-40) < 1e-11);

    let suz_na =
        eval_weak_reaction_pair("na24", "mg24", t9, ye_rho, true).unwrap();
    assert!(rel_err(suz_na.lambda, 3.7018793325371620e-40) < 1e-12);
}

#[test]
fn test_section4_all_307_raw_rates_across_9_temperatures_match_golden_oracle() {
    let lines: Vec<&str> = GOLDEN_TXT.lines().collect();
    let start = lines
        .iter()
        .position(|l| l.contains("do_test_suzuki"))
        .unwrap();
    let end = lines
        .iter()
        .position(|l| l.contains("FL_epsnuc_3alf"))
        .unwrap();

    let rcat = RatesCatalog::builtin();
    let irs: Vec<usize> = (1..=307).collect();
    let mut i = start;
    let mut temp_blocks = 0_usize;
    let mut total_checked = 0_usize;
    while i < end {
        let toks: Vec<&str> = lines[i].split_whitespace().collect();
        if toks.len() == 2 && toks[0] == "logT" {
            let log_t = parse_f64_fortran(toks[1]);
            let temp = 10.0_f64.powf(log_t);
            let tf = TFactors::from_log_t(log_t, temp);
            let vals = set_raw_rates(&irs, temp, &tf).unwrap();
            temp_blocks += 1;
            i += 3;
            while i < end {
                let row = lines[i].trim();
                if row.is_empty() || row == "done" {
                    break;
                }
                let rtoks: Vec<&str> = row.split_whitespace().collect();
                if rtoks.len() == 2 {
                    let rname = rtoks[0];
                    let exp_val = parse_f64_fortran(rtoks[1]);
                    let ir = rcat.reaction_id_by_name(rname).unwrap();
                    let got = vals[ir - 1];
                    if exp_val > 1e-290 {
                        assert!(
                            rel_err(got, exp_val) < 1e-11,
                            "temp={temp} rxn={rname}: got={got:e} exp={exp_val:e}"
                        );
                    }
                    total_checked += 1;
                }
                i += 1;
            }
        }
        i += 1;
    }
    assert_eq!(temp_blocks, 9);
    assert_eq!(total_checked, 9 * 307);
    assert_eq!(rcat.reactions.len(), NUM_BUNDLED_REACTIONS);
    assert_eq!(rcat.reaction_by_id(1).unwrap().name, "r1212");
}

#[test]
fn test_sections_5_6_7_pycno_rate_table_and_reaclib_reverse_match_golden() {
    // Section 5: Fushiki & Lamb (1987) pycnonuclear 3-alpha eps_nuc
    let eps = eval_fl_epsnuc_3alf(1.0e7, 1.0e10, 1.0, 2.0);
    assert!(rel_err(eps, 1.1675037697640312e21) < 1e-13);

    // Section 6: Tabulated rate `r3` (ir = 310) at T = 9e8 K
    let rcat = RatesCatalog::builtin();
    let temp_r3 = 10.0_f64.powf(9.0e8_f64.log10());
    let (r3_val, _) = rcat.eval_rate_table(310, temp_r3).unwrap();
    assert!(rel_err(r3_val, 1.0e-8) < 1e-14);

    // Section 7: REACLIB inverse_coefficients(1)
    let checks = [
        ("r_neut_neut_he4_to_he6", 4.6825459795335419e1),
        ("r_neut_pg_h2", 2.2274664971242341e1),
        ("r_h2_h2_to_neut_he3", 1.2425970576203393e0),
        ("r_o25_gn_o24", -2.2270683296520463e1),
    ];
    for (handle, exp_inv_c1) in checks {
        let (_, terms) = rcat.reaclib_terms(handle).unwrap();
        assert!(rel_err(terms[0].inv_c1, exp_inv_c1) < 1e-15);
    }
}
