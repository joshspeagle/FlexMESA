// SPDX-License-Identifier: LGPL-3.0-or-later
//! Analytical composition partial derivative (`d_dxa`) verification against
//! central finite differences across `HELM`, `FreeEOS`, `OPAL/SCVH`, `CMS`,
//! and the blended `eos_dt_get` cascade.

#![allow(
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use flexmesa_chem::NuclideCatalog;
use flexmesa_eos::{
    eos_dt_get, eval_cms_eos, eval_free_eos, eval_helm_eos, eval_opal_scvh_eos,
    CmsConfig, CmsTableSet, EosConfig, EosDtTableSet, FreeEosConfig, HelmTable,
    OpalScvhConfig, I_FRAC_CMS, I_FRAC_FREE_EOS, I_FRAC_HELM, I_FRAC_OPAL_SCVH,
    I_LN_E, I_LN_PGAS,
};
use flexmesa_math::exp10;

fn solar_chem() -> (Vec<usize>, Vec<f64>) {
    let cat = NuclideCatalog::builtin_ref();
    let names = ["h1", "he4", "c12", "n14", "o16", "ne20", "mg24"];
    let chem_id: Vec<usize> = names
        .iter()
        .map(|n| cat.get_by_name(n).expect("valid isotope").chem_id)
        .collect();
    let xa = vec![0.70, 0.285, 0.003, 0.001, 0.007, 0.002, 0.002];
    (chem_id, xa)
}

fn assert_dxa_close(label: &str, analytic: f64, fd: f64, rtol: f64) {
    let diff = (analytic - fd).abs();
    let scale = analytic.abs().max(fd.abs());
    assert!(
        diff <= 1.0e-11 || diff <= rtol * scale,
        "{label}: analytic={analytic:.10e}, fd={fd:.10e}, diff={diff:.3e}"
    );
}

#[test]
fn test_helm_single_species_and_compensated_d_dxa_match_central_fd() {
    let ht = HelmTable::builtin();
    let cat = NuclideCatalog::builtin_ref();
    let (chem_id, xa) = solar_chem();
    let a_vec: Vec<f64> = chem_id
        .iter()
        .map(|&cid| {
            f64::from(
                u32::try_from(cat.get_by_id(cid).unwrap().z_plus_n).unwrap(),
            )
        })
        .collect();
    let z_vec: Vec<f64> = chem_id
        .iter()
        .map(|&cid| {
            f64::from(u32::try_from(cat.get_by_id(cid).unwrap().z).unwrap())
        })
        .collect();
    let exact_abar_zbar = |x: &[f64]| -> (f64, f64) {
        let sumx: f64 = x.iter().sum();
        let sumy: f64 =
            x.iter().zip(a_vec.iter()).map(|(&xi, &ai)| xi / ai).sum();
        let sumyz: f64 = x
            .iter()
            .zip(a_vec.iter())
            .zip(z_vec.iter())
            .map(|((&xi, &ai), &zi)| (xi / ai) * zi)
            .sum();
        (sumx / sumy, sumyz / sumy)
    };

    let cfg = EosConfig {
        use_free_eos: false,
        use_opal_scvh: false,
        use_cms: false,
        use_skye: false,
        use_pc: false,
        ..EosConfig::default()
    };
    let rho = exp10(3.5);
    let temp = exp10(8.6);
    let base = eos_dt_get(ht, rho, temp, &chem_id, &xa, &cfg).unwrap();
    assert!((base.res[I_FRAC_HELM] - 1.0).abs() < 1.0e-12);

    // 1. Unconstrained single-species FD against `HELM(T, rho, abar(X), zbar_true(X))`
    //    verifies every individual `base.d_dxa[j]` without sink subtraction.
    let h = 1.0e-6;
    for j in 0..chem_id.len() {
        let mut xp = xa.clone();
        let mut xm = xa.clone();
        xp[j] += h;
        xm[j] -= h;
        let (ap, zp) = exact_abar_zbar(&xp);
        let (am, zm) = exact_abar_zbar(&xm);
        let (op, _, _) = eval_helm_eos(temp, rho, ap, zp).unwrap();
        let (om, _, _) = eval_helm_eos(temp, rho, am, zm).unwrap();
        let fd_p = (op.res[I_LN_PGAS] - om.res[I_LN_PGAS]) / (2.0 * h);
        let fd_e = (op.res[I_LN_E] - om.res[I_LN_E]) / (2.0 * h);
        assert_dxa_close(
            "HELM single dlnPgas/dX_j",
            base.d_dxa[j][0],
            fd_p,
            5.0e-6,
        );
        assert_dxa_close(
            "HELM single dlnE/dX_j",
            base.d_dxa[j][1],
            fd_e,
            5.0e-6,
        );
    }

    // 2. Baryon-conserving simplex FD (`sum(X) == 1`) through `eos_dt_get`.
    for (j, sink) in [(0_usize, 1_usize), (2, 1), (4, 1), (0, 4), (6, 2)] {
        let mut xp = xa.clone();
        let mut xm = xa.clone();
        xp[j] += h;
        xp[sink] -= h;
        xm[j] -= h;
        xm[sink] += h;
        let op = eos_dt_get(ht, rho, temp, &chem_id, &xp, &cfg).unwrap();
        let om = eos_dt_get(ht, rho, temp, &chem_id, &xm, &cfg).unwrap();
        assert_dxa_close(
            "HELM simplex dlnPgas",
            base.d_dxa[j][0] - base.d_dxa[sink][0],
            (op.res[I_LN_PGAS] - om.res[I_LN_PGAS]) / (2.0 * h),
            5.0e-6,
        );
        assert_dxa_close(
            "HELM simplex dlnE",
            base.d_dxa[j][1] - base.d_dxa[sink][1],
            (op.res[I_LN_E] - om.res[I_LN_E]) / (2.0 * h),
            5.0e-6,
        );
    }
}

#[test]
fn test_freeeos_opal_cms_and_blended_d_dxa_match_central_fd() {
    let ht = HelmTable::builtin();
    let dt_tables = EosDtTableSet::builtin();
    let cms_tables = CmsTableSet::builtin();
    let (chem_id, xa) = solar_chem();
    let h = 1.0e-5;

    // 1. FreeEOS interior point (X vs He4 and Z_metal vs He4)
    let rho_f = exp10(-2.0);
    let t_f = exp10(5.5);
    let fcfg = FreeEosConfig::default();
    let base_f =
        eval_free_eos(dt_tables, rho_f, t_f, &chem_id, &xa, &fcfg).unwrap();
    assert!((base_f.res[I_FRAC_FREE_EOS] - 1.0).abs() < 1.0e-12);
    assert_eq!(base_f.d_dxa[1], [0.0, 0.0]);
    for (label, idx) in [("FreeEOS H1-He4", 0_usize), ("FreeEOS C12-He4", 2)] {
        let mut xp = xa.clone();
        let mut xm = xa.clone();
        xp[idx] += h;
        xp[1] -= h;
        xm[idx] -= h;
        xm[1] += h;
        let op =
            eval_free_eos(dt_tables, rho_f, t_f, &chem_id, &xp, &fcfg).unwrap();
        let om =
            eval_free_eos(dt_tables, rho_f, t_f, &chem_id, &xm, &fcfg).unwrap();
        let fd_p = (op.res[I_LN_PGAS] - om.res[I_LN_PGAS]) / (2.0 * h);
        let fd_e = (op.res[I_LN_E] - om.res[I_LN_E]) / (2.0 * h);
        assert_dxa_close(
            label,
            base_f.d_dxa[idx][0] - base_f.d_dxa[1][0],
            fd_p,
            2.0e-5,
        );
        assert_dxa_close(
            label,
            base_f.d_dxa[idx][1] - base_f.d_dxa[1][1],
            fd_e,
            2.0e-5,
        );
    }

    // 2. OPAL/SCVH interior point
    let ocfg = OpalScvhConfig::default();
    let rho_o = exp10(-3.0);
    let t_o = exp10(5.0);
    let base_o =
        eval_opal_scvh_eos(dt_tables, rho_o, t_o, &chem_id, &xa, &ocfg)
            .unwrap();
    assert!((base_o.res[I_FRAC_OPAL_SCVH] - 1.0).abs() < 1.0e-12);
    let mut xp = xa.clone();
    let mut xm = xa.clone();
    xp[0] += h;
    xp[1] -= h;
    xm[0] -= h;
    xm[1] += h;
    let op_o = eval_opal_scvh_eos(dt_tables, rho_o, t_o, &chem_id, &xp, &ocfg)
        .unwrap();
    let om_o = eval_opal_scvh_eos(dt_tables, rho_o, t_o, &chem_id, &xm, &ocfg)
        .unwrap();
    assert_dxa_close(
        "OPAL H1-He4 Pgas",
        base_o.d_dxa[0][0],
        (op_o.res[I_LN_PGAS] - om_o.res[I_LN_PGAS]) / (2.0 * h),
        2.0e-5,
    );
    assert_dxa_close(
        "OPAL H1-He4 E",
        base_o.d_dxa[0][1],
        (op_o.res[I_LN_E] - om_o.res[I_LN_E]) / (2.0 * h),
        2.0e-5,
    );

    // 3. CMS H/He table (linear X interpolation -> exact to machine precision)
    let ccfg = CmsConfig::default();
    let chem_hhe = vec![chem_id[0], chem_id[1]];
    let xa_hhe = vec![0.72, 0.28];
    let rho_c = exp10(0.0);
    let t_c = exp10(4.5);
    let base_c =
        eval_cms_eos(cms_tables, &ccfg, rho_c, t_c, &chem_hhe, &xa_hhe)
            .unwrap();
    assert!((base_c.res[I_FRAC_CMS] - 1.0).abs() < 1.0e-12);
    let op_c = eval_cms_eos(
        cms_tables,
        &ccfg,
        rho_c,
        t_c,
        &chem_hhe,
        &[0.72 + h, 0.28 - h],
    )
    .unwrap();
    let om_c = eval_cms_eos(
        cms_tables,
        &ccfg,
        rho_c,
        t_c,
        &chem_hhe,
        &[0.72 - h, 0.28 + h],
    )
    .unwrap();
    assert_dxa_close(
        "CMS H1-He4 Pgas",
        base_c.d_dxa[0][0],
        (op_c.res[I_LN_PGAS] - om_c.res[I_LN_PGAS]) / (2.0 * h),
        1.0e-9,
    );
    assert_dxa_close(
        "CMS H1-He4 E",
        base_c.d_dxa[0][1],
        (op_c.res[I_LN_E] - om_c.res[I_LN_E]) / (2.0 * h),
        1.0e-9,
    );

    // 4. Blended FreeEOS + HELM transition zone via eos_dt_get
    let cfg = EosConfig::default();
    let rho_b = exp10(1.5);
    let t_b = exp10(6.95);
    let base_b = eos_dt_get(ht, rho_b, t_b, &chem_id, &xa, &cfg).unwrap();
    let op_b = eos_dt_get(ht, rho_b, t_b, &chem_id, &xp, &cfg).unwrap();
    let om_b = eos_dt_get(ht, rho_b, t_b, &chem_id, &xm, &cfg).unwrap();
    assert_dxa_close(
        "Blend H1-He4 Pgas",
        base_b.d_dxa[0][0] - base_b.d_dxa[1][0],
        (op_b.res[I_LN_PGAS] - om_b.res[I_LN_PGAS]) / (2.0 * h),
        2.0e-5,
    );
    assert_dxa_close(
        "Blend H1-He4 E",
        base_b.d_dxa[0][1] - base_b.d_dxa[1][1],
        (op_b.res[I_LN_E] - om_b.res[I_LN_E]) / (2.0 * h),
        2.0e-5,
    );
}
