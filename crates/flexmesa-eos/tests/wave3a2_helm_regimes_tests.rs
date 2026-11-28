// SPDX-License-Identifier: GPL-3.0-or-later
//! Upstream Fortran MESA `26.4.1` oracle tests for Wave 3a.2 (`HELM`
//! degenerate, relativistic pair-production, crystallized WD, and Hermite basis
//! regimes; `#161`, `T2-15`).
//!
//! Expected `16 x 5` reference rows and `IERR` codes are loaded from
//! `benchmarks/fortran_mesa/eos_oracles/mesa_26_4_1_eos_oracles.tsv`.

#![allow(clippy::similar_names)]

mod common;

use flexmesa_eos::helm::{
    H_CHID, H_CHIT, H_CP, H_CS, H_CV, H_DCHIDDA, H_DCHIDDD, H_DCHIDDT,
    H_DCHIDDZ, H_DCHITDA, H_DCHITDD, H_DCHITDT, H_DCHITDZ, H_DCPDA, H_DCPDD,
    H_DCPDT, H_DCPDZ, H_DCVDA, H_DCVDD, H_DCVDT, H_DCVDZ, H_DEA, H_DED, H_DEDA,
    H_DEDD, H_DEDT, H_DEDZ, H_DET, H_DETAA, H_DETAD, H_DETAT, H_DETAZ, H_DEZ,
    H_DGAM1DA, H_DGAM1DD, H_DGAM1DT, H_DGAM1DZ, H_DGAM2DA, H_DGAM2DD,
    H_DGAM2DT, H_DGAM2DZ, H_DGAM3DA, H_DGAM3DD, H_DGAM3DT, H_DGAM3DZ, H_DNABDA,
    H_DNABDD, H_DNABDT, H_DNABDZ, H_DPA, H_DPD, H_DPGASA, H_DPGASD, H_DPGASDD,
    H_DPGASDT, H_DPGAST, H_DPGASZ, H_DPRADD, H_DPRADT, H_DPT, H_DPZ, H_DSA,
    H_DSD, H_DSDA, H_DSDD, H_DSDT, H_DSDZ, H_DST, H_DSTA, H_DSTT, H_DSTZ,
    H_DSZ, H_DXNEA, H_DXNED, H_DXNET, H_DXNEZ, H_EGAS, H_ERAD, H_ETAELE,
    H_ETOT, H_GAM1, H_GAM2, H_GAM3, H_NABAD, H_PGAS, H_PLASG, H_PRAD, H_PTOT,
    H_SGAS, H_SRAD, H_STOT, H_XNE, H_XNI,
};
use flexmesa_eos::{
    eval_helm_eos, eval_helm_eos_zones, helmeos2, CubicBasis1D, HelmConfig,
    HelmTable, QuinticBasis1D, EOS_BASIC_RESULT_NAMES, I_ETA, I_LN_PGAS,
};
use flexmesa_math::{with_numerics_mode, NumericsMode};

const EOS_PGAS_ION_ORACLES: &str = include_str!(
    "../../../benchmarks/fortran_mesa/eos_pgas_ion_oracles/mesa_26_4_1_eos_pgas_ion_oracles.tsv"
);

fn mesa_helm_bits(case: &str, role: &str, field: &str, part: &str) -> u64 {
    for line in EOS_PGAS_ION_ORACLES.lines() {
        if line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() >= 5
            && cols[0] == case
            && cols[1] == role
            && cols[2] == field
            && cols[3] == part
        {
            let hex = cols[4].trim_start_matches("0x");
            return u64::from_str_radix(hex, 16).expect("hex column");
        }
    }
    panic!("no MESA helm oracle row for {case} {role} {field} {part}");
}

fn assert_helm_bits(label: &str, actual: f64, exp: u64) {
    let got_bits = actual.to_bits();
    assert_eq!(
        got_bits,
        exp,
        "{label}: MESA {e:.17e} ({exp:016X}) vs Flex {actual:.17e} ({got_bits:016X})",
        e = f64::from_bits(exp),
    );
}

fn check_helm_case_strict(
    case_id: &str,
    temp: f64,
    rho: f64,
    abar: f64,
    zbar: f64,
) {
    for (fld, val) in
        [("T", temp), ("Rho", rho), ("abar", abar), ("zbar", zbar)]
    {
        assert_helm_bits(
            &format!("{case_id}.in.{fld}"),
            val,
            mesa_helm_bits(case_id, "in", fld, "val"),
        );
    }
    let (out, d_dabar, d_dzbar) =
        eval_helm_eos(temp, rho, abar, zbar).expect(case_id);
    for idx in 0..16 {
        let fld = format!("res_{idx}");
        let name = EOS_BASIC_RESULT_NAMES[idx];
        for (part, got) in [
            ("val", out.res[idx]),
            ("d_dlnd", out.d_dlnd[idx]),
            ("d_dlnT", out.d_dln_t[idx]),
            ("d_dabar", d_dabar[idx]),
            ("d_dzbar", d_dzbar[idx]),
        ] {
            assert_helm_bits(
                &format!("{case_id}.out[{idx}:{name}].{part}"),
                got,
                mesa_helm_bits(case_id, "out", &fld, part),
            );
        }
    }
}

#[test]
fn test_helm_eos_degenerate_and_pair_regimes_match_mesa_26_4_1_oracle() {
    // 1. test_components (degenerate, plasg > 1): logT = 6.0, logRho = 3.5
    common::assert_helm_against_mesa(
        "test_components",
        1.0e6,
        flexmesa_math::exp10(3.5),
        2.990_964_711_670_598,
        1.674_940_238_535_534_8,
        1.0e-13,
    );

    // 2. sample_eos (relativistic e-e+ pairs): T = 1e9, Rho = 1e4, C12
    common::assert_helm_against_mesa(
        "sample_eos_c12",
        1.0e9,
        1.0e4,
        12.0,
        6.0,
        1.0e-13,
    );

    // 3. Cold crystallized WD core where sgas < 1e-20 returns Err(1) in MESA
    let (exp_ierr, _) = common::load_ierr_row("crystallized_wd_1e6_6p5");
    let res_err = eval_helm_eos(
        1.0e6,
        flexmesa_math::exp10(6.5),
        13.714_285_714_285_715,
        6.857_142_857_142_858,
    );
    assert_eq!(res_err, Err(exp_ierr));

    with_numerics_mode(NumericsMode::Strict, || {
        check_helm_case_strict(
            "helm_test_components",
            1.0e6,
            flexmesa_math::exp10(3.5),
            2.990_964_711_670_598,
            1.674_940_238_535_534_8,
        );
        check_helm_case_strict("helm_sample_eos_c12", 1.0e9, 1.0e4, 12.0, 6.0);
    });
}

fn check_helm_zones_and_helmeos2_strict(table: &HelmTable, cfg: &HelmConfig) {
    with_numerics_mode(NumericsMode::Strict, || {
        for (cid, t, r, a, z) in [
            ("helm_pure_he_2e8_100", 2.0e8, 100.0, 4.0, 2.0),
            ("helm_pure_h_2e8_100", 2.0e8, 100.0, 1.0, 1.0),
        ] {
            check_helm_case_strict(cid, t, r, a, z);
        }
        let batch_strict = eval_helm_eos_zones(
            &[2.0e8, 2.0e8, 1.0e9],
            &[100.0, 100.0, 1.0e4],
            &[4.0, 1.0, 12.0],
            &[2.0, 1.0, 6.0],
        );
        for (i, cid) in [
            "helm_pure_he_2e8_100",
            "helm_pure_h_2e8_100",
            "helm_sample_eos_c12",
        ]
        .iter()
        .enumerate()
        {
            assert_helm_bits(
                &format!("{cid}.zones.lnPgas"),
                batch_strict[i].res[I_LN_PGAS],
                mesa_helm_bits(cid, "out", "res_0", "val"),
            );
        }

        let (hres, off_table) = helmeos2(
            table,
            2.0e8,
            flexmesa_math::log10(2.0e8_f64),
            100.0,
            2.0,
            4.0,
            2.0,
            cfg,
        )
        .expect("helmeos2 pure_he_2e8_100");
        assert!(!off_table);
        let h_slots = [
            H_PTOT, H_DPT, H_DPD, H_DPA, H_DPZ, H_ETOT, H_DET, H_DED, H_DEA,
            H_DEZ, H_DEDD, H_DEDT, H_DEDA, H_DEDZ, H_STOT, H_DST, H_DSD, H_DSA,
            H_DSZ, H_DSDD, H_DSDT, H_DSDA, H_DSDZ, H_DSTT, H_DSTA, H_DSTZ,
            H_PRAD, H_DPRADT, H_DPRADD, H_ERAD, H_SRAD, H_PGAS, H_DPGAST,
            H_DPGASD, H_DPGASA, H_DPGASZ, H_DPGASDD, H_DPGASDT, H_EGAS, H_SGAS,
            H_XNI, H_ETAELE, H_DETAT, H_DETAD, H_DETAA, H_DETAZ, H_XNE,
            H_DXNET, H_DXNED, H_DXNEA, H_DXNEZ, H_PLASG, H_CP, H_DCPDD,
            H_DCPDT, H_DCPDA, H_DCPDZ, H_CV, H_DCVDD, H_DCVDT, H_DCVDA,
            H_DCVDZ, H_GAM1, H_DGAM1DD, H_DGAM1DT, H_DGAM1DA, H_DGAM1DZ,
            H_GAM2, H_DGAM2DD, H_DGAM2DT, H_DGAM2DA, H_DGAM2DZ, H_GAM3,
            H_DGAM3DD, H_DGAM3DT, H_DGAM3DA, H_DGAM3DZ, H_NABAD, H_DNABDD,
            H_DNABDT, H_DNABDA, H_DNABDZ, H_CS, H_CHIT, H_DCHITDD, H_DCHITDT,
            H_DCHITDA, H_DCHITDZ, H_CHID, H_DCHIDDD, H_DCHIDDT, H_DCHIDDA,
            H_DCHIDDZ,
        ];
        for &slot in &h_slots {
            assert_helm_bits(
                &format!("helmeos2.helm_{slot}"),
                hres[slot - 1],
                mesa_helm_bits(
                    "helm_pure_he_2e8_100",
                    "out",
                    &format!("helm_{slot}"),
                    "val",
                ),
            );
        }
    });
}

#[test]
fn test_helm_basis_hermite_endpoint_conditions_and_zones() {
    let q0 = QuinticBasis1D::eval(0.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    let q1 = QuinticBasis1D::eval(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    assert!((q0.w0[0] - 1.0).abs() < 1.0e-15);
    assert!(q0.w0[3].abs() < 1.0e-15);
    assert!(q1.w0[0].abs() < 1.0e-15);
    assert!((q1.w0[3] - 1.0).abs() < 1.0e-15);

    let c0 = CubicBasis1D::eval(0.0, 1.0, 1.0, 1.0, 1.0);
    let c1 = CubicBasis1D::eval(1.0, 1.0, 1.0, 1.0, 1.0);
    assert!((c0.w0[0] - 1.0).abs() < 1.0e-15);
    assert!((c1.w0[2] - 1.0).abs() < 1.0e-15);

    let pure_he_oracle =
        common::load_indexed_rows::<16, 5>("HELM", "pure_he_2e8_100");
    let pure_h_oracle =
        common::load_indexed_rows::<16, 5>("HELM", "pure_h_2e8_100");
    let pair_oracle =
        common::load_indexed_rows::<16, 5>("HELM", "sample_eos_c12");

    let table = HelmTable::builtin().expect("builtin HELM table");
    let (f_ele, eta_ele, xne_fer) = table.eval_electron_positron_order3(
        2.0e8,
        flexmesa_math::log10(2.0e8_f64),
        100.0,
        4.0,
        2.0,
    );
    assert!(f_ele.val.is_finite() && f_ele.d3val1.is_finite());
    assert!((eta_ele.val - pure_he_oracle[I_ETA][0]).abs() < 1.0e-14);
    assert!(xne_fer.val > 0.0);

    let batch = eval_helm_eos_zones(
        &[2.0e8, 2.0e8, 1.0e9],
        &[100.0, 100.0, 1.0e4],
        &[4.0, 1.0, 12.0],
        &[2.0, 1.0, 6.0],
    );
    assert_eq!(batch.len(), 3);
    assert!(
        (batch[0].res[I_LN_PGAS] - pure_he_oracle[I_LN_PGAS][0]).abs()
            < 1.0e-14
    );
    assert!(
        (batch[1].res[I_LN_PGAS] - pure_h_oracle[I_LN_PGAS][0]).abs() < 1.0e-14
    );
    assert!(
        (batch[2].res[I_LN_PGAS] - pair_oracle[I_LN_PGAS][0]).abs() < 1.0e-14
    );

    // Off-table temperature returns Err(1) matching `eos/private/helm.f90:194`
    let (exp_low_t_ierr, _) = common::load_ierr_row("helmeos2_low_t_100_100");
    let cfg = HelmConfig::default();
    let res_low_t = helmeos2(table, 100.0, 2.0, 100.0, 2.0, 4.0, 2.0, &cfg);
    assert_eq!(res_low_t, Err(exp_low_t_ierr));

    check_helm_zones_and_helmeos2_strict(table, &cfg);
}
