// SPDX-License-Identifier: LGPL-3.0-or-later
//! Full-table verification for `HELM`, `OPAL/SCVH`, `FreeEOS`, `CMS`, and `eos_dt_get`
//! across multi-zone stellar profiles (`1 M_sun` ZAMS, `0.1 M_sun` brown dwarf,
//! `0.6 M_sun` white dwarf, and `25 M_sun` massive star).

#![allow(
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::similar_names
)]

use flexmesa_chem::NuclideCatalog;
use flexmesa_eos::helm::{HELM_IMAX, HELM_JMAX};
use flexmesa_eos::{
    eos_dt_get, eval_cms_eos, eval_free_eos, eval_helm_eos, eval_opal_scvh_eos,
    CmsConfig, CmsTableSet, EosConfig, EosDtTableSet, FreeEosConfig, HelmTable,
    OpalScvhConfig, I_CHI_RHO, I_CHI_T, I_GAMMA1, I_LN_E, I_LN_PGAS, I_LN_S,
};

fn standard_chem_ids() -> [usize; 7] {
    let cat = NuclideCatalog::builtin_ref();
    ["h1", "he4", "c12", "n14", "o16", "ne20", "mg24"]
        .map(|name| cat.get_by_name(name).expect("standard nuclide").chem_id)
}

#[test]
fn test_helm_full_table_loaded_and_dense_across_full_grid() {
    let tbl = HelmTable::builtin();
    if tbl.is_full_table() {
        let dense = tbl.dense_vertices().expect("dense vertices");
        assert_eq!(dense.len(), HELM_IMAX * HELM_JMAX);
        for &(i, j) in &[
            (1, 1),
            (137, 42),
            (999, 333),
            (1845, 612),
            (2540, 888),
            (HELM_IMAX, HELM_JMAX),
        ] {
            let v = tbl.vertex(i, j).expect("dense vertex must exist");
            assert!(v.f[0].is_finite());
            assert!(v.dpdf[0].is_finite());
            assert!(v.ef[0].is_finite());
            assert!(v.xf[0].is_finite());
        }
    }

    for it in 0..5 {
        let t = 10.0_f64.powf(5.5 + 0.9 * (it as f64));
        for ir in 0..5 {
            let rho = 10.0_f64.powf(-4.5 + 1.8 * (ir as f64));
            let (out, d_da, d_dz) =
                eval_helm_eos(t, rho, 1.35, 1.12).expect("helm eval");
            assert!(out.res[I_LN_PGAS].is_finite());
            assert!(out.res[I_LN_E].is_finite());
            assert!(out.res[I_LN_S].is_finite());
            assert!(out.d_dlnd[I_LN_PGAS] > 0.0);
            assert!(d_da[I_LN_PGAS].is_finite());
            assert!(d_dz[I_LN_PGAS].is_finite());
        }
    }
}

#[test]
fn test_4d_tables_and_eos_dt_get_across_four_stellar_profiles() {
    let helm_tbl = HelmTable::builtin();
    let dt_tables = EosDtTableSet::builtin();
    let cms_tables = CmsTableSet::builtin();
    if dt_tables.data_dir.is_none() {
        return;
    }

    let chem_id = standard_chem_ids();
    let xa_solar = [0.70, 0.28, 0.004, 0.001, 0.009, 0.003, 0.003];
    let xa_bd = [0.73, 0.25, 0.004, 0.001, 0.009, 0.003, 0.003];
    let xa_wd_core = [0.00, 0.02, 0.48, 0.01, 0.47, 0.01, 0.01];
    let xa_massive = [0.68, 0.30, 0.004, 0.001, 0.009, 0.003, 0.003];

    let opal_cfg = OpalScvhConfig::default();
    let free_cfg = FreeEosConfig::default();
    let cms_cfg = CmsConfig::default();
    let dt_cfg = EosConfig::default();

    let n_zones = 32usize;

    // 1. 1 M_sun ZAMS profile: logT in [3.76, 7.18], logRho in [-6.8, 2.18]
    for k in 0..n_zones {
        let s = (k as f64) / ((n_zones - 1) as f64);
        let log_t = 3.76 + s * (7.18 - 3.76);
        let log_rho = -6.8 + s * (2.18 - (-6.8));
        let (rho, t) = (10.0_f64.powf(log_rho), 10.0_f64.powf(log_t));

        let out_opal = eval_opal_scvh_eos(
            dt_tables, rho, t, &chem_id, &xa_solar, &opal_cfg,
        )
        .expect("1 M_sun OPAL/SCVH zone must succeed");
        let out_free =
            eval_free_eos(dt_tables, rho, t, &chem_id, &xa_solar, &free_cfg)
                .expect("1 M_sun FreeEOS zone must succeed");
        let out_dt = eos_dt_get(helm_tbl, rho, t, &chem_id, &xa_solar, &dt_cfg)
            .expect("1 M_sun eos_dt_get zone must succeed");

        for out in [&out_opal, &out_free, &out_dt] {
            assert!(out.res[I_LN_PGAS].is_finite());
            assert!(out.res[I_LN_E].is_finite());
            assert!(out.res[I_LN_S].is_finite());
            assert!(out.res[I_GAMMA1] > 1.0);
        }
    }

    // 2. 0.1 M_sun brown dwarf / VLMS profile: logT in [3.15, 6.45], logRho in [-4.5, 2.75]
    for k in 0..n_zones {
        let s = (k as f64) / ((n_zones - 1) as f64);
        let log_t = 3.15 + s * (6.45 - 3.15);
        let log_rho = -4.5 + s * (2.75 - (-4.5));
        let (rho, t) = (10.0_f64.powf(log_rho), 10.0_f64.powf(log_t));

        let out_cms =
            eval_cms_eos(cms_tables, &cms_cfg, rho, t, &chem_id, &xa_bd)
                .expect("0.1 M_sun CMS zone must succeed");
        let out_opal =
            eval_opal_scvh_eos(dt_tables, rho, t, &chem_id, &xa_bd, &opal_cfg)
                .expect("0.1 M_sun OPAL/SCVH zone must succeed");
        let out_dt = eos_dt_get(helm_tbl, rho, t, &chem_id, &xa_bd, &dt_cfg)
            .expect("0.1 M_sun eos_dt_get zone must succeed");

        for out in [&out_cms, &out_opal, &out_dt] {
            assert!(out.res[I_LN_PGAS].is_finite());
            assert!(out.res[I_CHI_RHO] > 0.0);
            assert!(out.res[I_CHI_T].is_finite());
        }
    }

    // 3. 0.6 M_sun C/O white dwarf profile: logT in [5.5, 7.6], logRho in [-1.0, 6.5]
    for k in 0..n_zones {
        let s = (k as f64) / ((n_zones - 1) as f64);
        let log_t = 5.5 + s * (7.6 - 5.5);
        let log_rho = -1.0 + s * (6.5 - (-1.0));
        let (rho, t) = (10.0_f64.powf(log_rho), 10.0_f64.powf(log_t));

        let out_dt =
            eos_dt_get(helm_tbl, rho, t, &chem_id, &xa_wd_core, &dt_cfg)
                .expect("0.6 M_sun WD eos_dt_get zone must succeed");
        assert!(out_dt.res[I_LN_PGAS].is_finite());
        assert!(out_dt.res[I_LN_E].is_finite());
        assert!(out_dt.res[I_LN_S].is_finite());
    }

    // 4. 25 M_sun massive star profile: logT in [4.3, 8.3], logRho in [-8.0, 2.5]
    for k in 0..n_zones {
        let s = (k as f64) / ((n_zones - 1) as f64);
        let log_t = 4.3 + s * (8.3 - 4.3);
        let log_rho = -8.0 + s * (2.5 - (-8.0));
        let (rho, t) = (10.0_f64.powf(log_rho), 10.0_f64.powf(log_t));

        let out_dt =
            eos_dt_get(helm_tbl, rho, t, &chem_id, &xa_massive, &dt_cfg)
                .expect("25 M_sun massive star eos_dt_get zone must succeed");
        assert!(out_dt.res[I_LN_PGAS].is_finite());
        assert!(out_dt.res[I_LN_E].is_finite());
        assert!(out_dt.res[I_LN_S].is_finite());
    }
}
