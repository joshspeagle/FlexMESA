// SPDX-License-Identifier: LGPL-3.0-or-later
//! Bit-exact IEEE-754 `f64` regression tests for `eval_ecapture_pair`,
//! `eval_mui_coulomb`, and `eval_vs_coulomb` against MESA 26.4.1
//! (`rates/test/src/{test_weak.f90, test_ecapture.f90}`).

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{KEV, LN2};
use flexmesa_eos::{eos_dt_get, EosConfig, HelmTable, I_ETA};
use flexmesa_math::{exp10, with_numerics_mode, NumericsMode};
use flexmesa_rates::{
    eval_ecapture_pair, eval_mui_coulomb, eval_vs_coulomb, CoulombMuiOption,
    CoulombVsOption,
};

fn assert_bits(label: &str, got: f64, expected_bits: u64) {
    let got_bits = got.to_bits();
    assert_eq!(
        got_bits,
        expected_bits,
        "{label}: got 0x{got_bits:016x} ({got:.17e}), expected \
         0x{expected_bits:016x} ({:.17e}), ulp={}",
        f64::from_bits(expected_bits),
        got_bits.abs_diff(expected_bits),
    );
}

#[test]
fn test_weak_ecapture_strict_bit_exact_hex() {
    with_numerics_mode(NumericsMode::Strict, || {
        let t9_wk = f64::from_bits(0x3fec_cccc_cccc_cc98);
        let log_rho_wk = f64::from_bits(0x4016_9ce3_bdc0_031f);
        let ye_rho_wk = 0.5 * exp10(log_rho_wk);
        let wk = eval_ecapture_pair(
            "na24",
            "mg24",
            t9_wk,
            ye_rho_wk,
            10.0,
            CoulombMuiOption::None,
            CoulombVsOption::None,
            0.0,
            1.0,
        )
        .expect("na24->mg24 weak ecapture should succeed");
        assert_bits(
            "weak na24->mg24 t1/2",
            LN2 / wk.lambda,
            0x40d2_b2c2_aa12_f458,
        );
        assert_bits("weak na24->mg24 qneu", wk.qneu, 0x4008_e0bf_ba17_edd9);
        assert_bits("weak na24->mg24 q", wk.q, 0x4016_0fd2_e946_8018);
    });
}

#[test]
fn test_coulomb_strict_bit_exact_hex() {
    with_numerics_mode(NumericsMode::Strict, || {
        let ya_ec = [0.5_f64 / 20.0_f64, 0.5_f64 / 24.0_f64];
        let abar_ec = 1.0_f64 / (ya_ec[0] + ya_ec[1]);
        let zbar_ec = abar_ec * (ya_ec[0] * 10.0_f64 + ya_ec[1] * 12.0_f64);
        let t_coul = exp10(8.5_f64);
        let rho_coul = exp10(9.5_f64);
        let mu1 = eval_mui_coulomb(
            CoulombMuiOption::Pcr2009,
            t_coul,
            rho_coul,
            zbar_ec,
            abar_ec,
            12.0,
        );
        let mu2 = eval_mui_coulomb(
            CoulombMuiOption::Pcr2009,
            t_coul,
            rho_coul,
            zbar_ec,
            abar_ec,
            11.0,
        );
        let dmu = (mu1 - mu2).abs() * KEV * t_coul;
        let vs = eval_vs_coulomb(
            CoulombVsOption::Itoh2002,
            t_coul,
            rho_coul,
            zbar_ec,
            abar_ec,
            12.0,
        );
        assert_bits("coulomb mu(mg24)", mu1, 0xc045_df66_2912_fb48);
        assert_bits("coulomb mu(na24)", mu2, 0xc042_c139_b631_bb7c);
        assert_bits("coulomb dmu_eV", dmu, 0x4104_be31_7007_0d53);
        assert_bits("coulomb vs(mg24)", vs, 0x3f80_31c4_b93b_a57a);
    });
}

#[test]
fn test_special_ecapture_strict_bit_exact_hex() {
    with_numerics_mode(NumericsMode::Strict, || {
        let ya_ec = [0.5_f64 / 20.0_f64, 0.5_f64 / 24.0_f64];
        let abar_ec = 1.0_f64 / (ya_ec[0] + ya_ec[1]);
        let zbar_ec = abar_ec * (ya_ec[0] * 10.0_f64 + ya_ec[1] * 12.0_f64);
        let cat = NuclideCatalog::builtin();
        let chem_id = [
            cat.get_nuclide_index("ne20").expect("ne20"),
            cat.get_nuclide_index("mg24").expect("mg24"),
        ];
        let xa = [0.5_f64, 0.5_f64];
        let temp_ec = exp10(8.3_f64);
        let rho_ec = exp10(9.8_f64);
        let ht = HelmTable::builtin().expect("HelmTable");
        let eos_out = eos_dt_get(
            ht,
            rho_ec,
            temp_ec,
            &chem_id,
            &xa,
            &EosConfig::default(),
        )
        .expect("eos_dt_get should succeed");
        let eta_ec = eos_out.res[I_ETA];
        let t9_ec = temp_ec * 1.0e-9;
        let ye_rho_ec = 0.5 * rho_ec;

        let sp_mg = eval_ecapture_pair(
            "mg24",
            "na24",
            t9_ec,
            ye_rho_ec,
            eta_ec,
            CoulombMuiOption::Pcr2009,
            CoulombVsOption::Itoh2002,
            zbar_ec,
            abar_ec,
        )
        .expect("mg24->na24 ecapture should succeed");
        let sp_na = eval_ecapture_pair(
            "na24",
            "mg24",
            t9_ec,
            ye_rho_ec,
            eta_ec,
            CoulombMuiOption::Pcr2009,
            CoulombVsOption::Itoh2002,
            zbar_ec,
            abar_ec,
        )
        .expect("na24->mg24 ecapture should succeed");

        assert_bits(
            "ecap mg24->na24 lambda",
            sp_mg.lambda,
            0x3f6f_8b9e_799a_4357,
        );
        assert_bits(
            "ecap na24->mg24 lambda",
            sp_na.lambda,
            0x37c7_7e4b_4a85_3729,
        );
    });
}
