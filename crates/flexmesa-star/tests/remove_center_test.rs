// SPDX-License-Identifier: GPL-3.0-or-later
//! Package D4 (`#574`) tests for `remove_initial_center_by_entropy`,
//! `remove_center_by_entropy`, `remove_center_set_zero_v_center`,
//! `remove_center_adjust_L_center`, and `do_remove_center`
//! (`star/private/remove_shells.f90:452-470` and
//! `star/private/remove_shells.f90:609-698`).

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;

use flexmesa_const::{AVO, KERG};
use flexmesa_math::exp;
use flexmesa_star::hydro_state::{StarModelHook, StarModelHookHandle};
use flexmesa_star::inlist::InlistBundle;
use flexmesa_star::inlist_inventory::{STAR_JOB_SUPPORTED, STAR_JOB_UNUSED};
use flexmesa_star::inlist_runner::{DefaultStarExtras, StarExtras};
use flexmesa_star::{
    do_remove_center, do_remove_center_by_entropy, do_remove_inner_fraction_q,
    init_or_update_model_from_bundle, load_zams_model, star_evolve_step,
    StarControls, StarError, StarJobControls, StarModel,
};

struct CcsnIipAlphaMltHook;

impl StarModelHook for CcsnIipAlphaMltHook {
    fn apply(&self, model: &mut StarModel) -> Result<(), StarError> {
        // Mirrors `star/test_suite/ccsn_IIp/src/run_star_extras.f90:138-161`:
        // `X_MLT_ALPHA = 21`, `X_MLT_OTHER = 22`, `X_MLT_H_LIM = 23` (1-based).
        let alpha_h = model.controls.x_ctrl[20];
        let alpha_other = model.controls.x_ctrl[21];
        let h_limit = model.controls.x_ctrl[22];
        let Some(h1_idx) = model.iso_names.iter().position(|n| n == "h1")
        else {
            return Ok(());
        };
        if alpha_h <= 0.0 || alpha_other <= 0.0 {
            return Ok(());
        }
        let species = model.species;
        for (k, slot) in model
            .aux
            .mixing
            .alpha_mlt
            .iter_mut()
            .take(model.nz)
            .enumerate()
        {
            let x_h1 = model.xa[k * species + h1_idx];
            *slot = if x_h1 >= h_limit {
                alpha_h
            } else {
                alpha_other
            };
        }
        Ok(())
    }
}

struct CcsnIipAlphaMltExtras;

impl StarExtras for CcsnIipAlphaMltExtras {
    fn other_alpha_mlt_handle(&self) -> Option<StarModelHookHandle> {
        Some(StarModelHookHandle::new(CcsnIipAlphaMltHook))
    }
}

struct TempCaseGuard {
    root: PathBuf,
}

impl TempCaseGuard {
    fn new(tag: &str) -> Self {
        static SEQ: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let id = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "flexmesa_star_574_{tag}_{}_{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create temp dir");
        Self { root }
    }
}

impl Drop for TempCaseGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn star_suite_dir() -> PathBuf {
    let base = std::env::var("MESA_DIR").map_or_else(
        |_| {
            let home =
                std::env::var("HOME").unwrap_or_else(|_| String::from("."));
            PathBuf::from(home)
                .join(".cache")
                .join("flexmesa")
                .join("mesa_test_inputs")
        },
        PathBuf::from,
    );
    base.join("star").join("test_suite")
}

fn ccsn_iip_inlist_path() -> PathBuf {
    let path = star_suite_dir()
        .join("ccsn_IIp")
        .join("inlist_infall_header");
    assert!(
        path.is_file(),
        "missing MESA test input {}; run `flexmesa data fetch --include-test-inputs` or set MESA_DIR",
        path.display()
    );
    path
}

fn default_zams() -> flexmesa_star::StarModel {
    load_zams_model(
        StarControls {
            initial_mass: 1.0,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 1 Msun ZAMS model")
}

#[test]
fn test_remove_center_controls_inventory_defaults_and_strict_parse() {
    let expected_keys = [
        "remove_center_adjust_l_center",
        "remove_center_by_entropy",
        "remove_center_set_zero_v_center",
        "remove_initial_center_by_entropy",
    ];
    for key in expected_keys {
        assert!(
            STAR_JOB_SUPPORTED.contains(&key),
            "missing `{key}` from STAR_JOB_SUPPORTED"
        );
        assert!(
            !STAR_JOB_UNUSED.contains(&key),
            "`{key}` must not appear in STAR_JOB_UNUSED"
        );
    }
    assert!(
        STAR_JOB_SUPPORTED.windows(2).all(|w| w[0] < w[1]),
        "STAR_JOB_SUPPORTED must stay strictly sorted"
    );

    let defaults = StarJobControls::default();
    assert_eq!(
        defaults.remove_initial_center_by_entropy.to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(
        defaults.remove_center_by_entropy.to_bits(),
        0.0_f64.to_bits()
    );
    assert!(!defaults.remove_center_set_zero_v_center);
    assert!(defaults.remove_center_adjust_l_center);

    let tmp = TempCaseGuard::new("parse_4_controls");
    let inlist = tmp.root.join("inlist");
    fs::write(
        &inlist,
        "&star_job\n\
           remove_initial_center_by_entropy = 4.25d0\n\
           remove_center_by_entropy = 3.5d0\n\
           remove_center_set_zero_v_center = .true.\n\
           remove_center_adjust_L_center = .false.\n\
         /\n\
         &controls\n\
           initial_mass = 1.0d0\n\
         /\n",
    )
    .expect("write test inlist");
    let bundle = InlistBundle::from_file(&inlist)
        .expect("strict parse of center-removal &star_job controls");
    assert_eq!(
        bundle.validation_report.unsupported_controls.len(),
        0,
        "unexpected unsupported controls: {:?}",
        bundle.validation_report.unsupported_controls
    );
    assert_eq!(
        bundle.job.remove_initial_center_by_entropy.to_bits(),
        4.25_f64.to_bits()
    );
    assert_eq!(
        bundle.job.remove_center_by_entropy.to_bits(),
        3.5_f64.to_bits()
    );
    assert!(bundle.job.remove_center_set_zero_v_center);
    assert!(!bundle.job.remove_center_adjust_l_center);
}

#[test]
fn test_ccsn_iip_infall_init_excises_to_850_zones_and_evolves_strictly() {
    let inlist_path = ccsn_iip_inlist_path();
    let bundle = InlistBundle::from_file(&inlist_path).unwrap_or_else(|e| {
        panic!("strict parse of {} failed: {e}", inlist_path.display())
    });
    assert_eq!(
        bundle.validation_report.unsupported_controls.len(),
        0,
        "ccsn_IIp/inlist_infall_header had unsupported controls: {:?}",
        bundle.validation_report.unsupported_controls
    );
    assert_eq!(
        bundle.job.remove_initial_center_by_entropy.to_bits(),
        4.0_f64.to_bits()
    );
    assert!(!bundle.job.remove_center_set_zero_v_center);
    assert!(bundle.job.remove_center_adjust_l_center);

    let mut extras = CcsnIipAlphaMltExtras;
    let mut model =
        init_or_update_model_from_bundle(&bundle, None, &mut extras)
            .expect("initialize ccsn_IIp/inlist_infall_header");

    assert_eq!(
        model.nz, 850,
        "ccsn_IIp/inlist_infall_header after remove_initial_center_by_entropy = 4 must match MESA 26.4.1 nz = 850"
    );
    assert_eq!(model.star_age.to_bits(), 0.0_f64.to_bits());
    assert_eq!(model.model_number, 0);
    assert!(
        (model.dt_next - 1.0e-4).abs() <= 1.0e-18,
        "expected dt_next = 1e-4, got {}",
        model.dt_next
    );
    assert_eq!(model.generations, 1);

    assert!(model.m_center.is_finite() && model.m_center > 0.0);
    assert!(model.r_center.is_finite() && model.r_center > 0.0);
    assert!(model.l_center.is_finite());
    assert!(model.v_center.is_finite());
    assert!(model.xmstar.is_finite() && model.xmstar > 0.0);
    assert!(
        (model.xmstar - (model.m[0] - model.m_center)).abs() / model.m[0]
            <= 1.0e-14
    );

    assert_eq!(model.m_center_old.to_bits(), model.m_center.to_bits());
    assert_eq!(model.xmstar_old.to_bits(), model.xmstar.to_bits());
    assert_eq!(model.r_center_old.to_bits(), model.r_center.to_bits());
    assert_eq!(model.l_center_old.to_bits(), model.l_center.to_bits());
    assert_eq!(model.v_center_old.to_bits(), model.v_center.to_bits());

    assert_eq!(model.dq.len(), 850);
    assert_eq!(model.dm.len(), 850);
    assert_eq!(model.q.len(), 850);
    let dq_sum: f64 = model.dq.iter().sum();
    assert!((dq_sum - 1.0).abs() <= 1.0e-12, "dq sum = {dq_sum}");
    assert!((model.q[0] - 1.0).abs() <= 1.0e-14, "q[0] = {}", model.q[0]);
    assert!(model.q[849] > 0.0);
    assert!(model.dm.iter().all(|&dm_k| dm_k.is_finite() && dm_k > 0.0));
    let mass_sum: f64 = model.dm.iter().sum();
    assert!(
        (mass_sum - model.xmstar).abs() / model.xmstar <= 1.0e-12,
        "sum(dm) = {mass_sum}, xmstar = {}",
        model.xmstar
    );

    // Also verify `remove_center_by_entropy` path produces the identical state
    // when substituted for `remove_initial_center_by_entropy`.
    let mut bundle_always = bundle;
    bundle_always.job.remove_initial_center_by_entropy = 0.0;
    bundle_always.job.remove_center_by_entropy = 4.0;
    let model_always =
        init_or_update_model_from_bundle(&bundle_always, None, &mut extras)
            .expect("init via remove_center_by_entropy = 4");
    assert_eq!(model_always.nz, 850);
    assert_eq!(model_always.m_center.to_bits(), model.m_center.to_bits());
    assert_eq!(model_always.r_center.to_bits(), model.r_center.to_bits());
    assert_eq!(model_always.l_center.to_bits(), model.l_center.to_bits());
    assert_eq!(model_always.v_center.to_bits(), model.v_center.to_bits());

    // Evolve >= 3 steps strictly without overriding unsupported controls.
    let do_remesh =
        model.controls.okay_to_remesh && model.controls.mesh_delta_coeff > 0.0;
    let op_split = model.controls.op_split_burn;
    for step in 1..=3 {
        let rep = star_evolve_step(&mut model, do_remesh, op_split, true)
            .unwrap_or_else(|e| {
                panic!("strict ccsn_IIp infall step {step} failed: {e}")
            });
        assert_eq!(model.model_number, step);
        assert!(model.nz > 0);
        assert!(model.star_age.is_finite() && model.star_age > 0.0);
        assert!(rep.dt.is_finite() && rep.dt > 0.0);
        assert!(model.m_center.is_finite() && model.r_center.is_finite());
        assert!(model.l_center.is_finite() && model.v_center.is_finite());
    }
}

#[test]
fn test_do_remove_center_boundary_toggles_noops_and_errors() {
    let mut extras = DefaultStarExtras;
    let base = default_zams();
    let nz0 = base.nz;
    assert!(nz0 > 20, "expected ZAMS model with > 20 zones");

    // 1. k_cut == 0 and k_cut >= nz are exact no-ops.
    for k_cut in [0, nz0, nz0 + 5] {
        let mut m = base.clone();
        do_remove_center(&mut m, k_cut, &mut extras)
            .expect("no-op k_cut succeeds");
        assert_eq!(m.nz, nz0);
        assert_eq!(m.m_center.to_bits(), base.m_center.to_bits());
        assert_eq!(m.r_center.to_bits(), base.r_center.to_bits());
        assert_eq!(m.l_center.to_bits(), base.l_center.to_bits());
        assert_eq!(m.v_center.to_bits(), base.v_center.to_bits());
    }

    // 2. Preserve star_age, model_number, dt_next, cumulative_energy_error;
    //    check remove_center_adjust_l_center = false vs true.
    let k_cut = nz0 / 2;
    let mut m_keep_l = base.clone();
    m_keep_l.star_age = 1.25e6;
    m_keep_l.model_number = 42;
    m_keep_l.dt_next = 3.5e4;
    m_keep_l.aux.cumulative_energy_error = -9.5e38;
    m_keep_l.l_center = 1.23e31;
    m_keep_l.ye_start = m_keep_l.ye.clone();
    m_keep_l.eps_nuc_start = m_keep_l.eps_nuc.clone();
    m_keep_l.lum_by_category_start =
        vec![[0.0; flexmesa_chem::NUM_CATEGORIES]; nz0];
    m_keep_l.job.remove_center_adjust_l_center = false;
    let want_mass_center = m_keep_l.m[k_cut];
    let want_radius_center = m_keep_l.r[k_cut];

    do_remove_center(&mut m_keep_l, k_cut, &mut extras)
        .expect("cut with adjust_l_center = false");
    assert_eq!(m_keep_l.nz, k_cut);
    assert_eq!(m_keep_l.generations, 1);
    assert_eq!(m_keep_l.star_age.to_bits(), 1.25e6_f64.to_bits());
    assert_eq!(m_keep_l.model_number, 42);
    assert_eq!(m_keep_l.dt_next.to_bits(), 3.5e4_f64.to_bits());
    assert_eq!(
        m_keep_l.aux.cumulative_energy_error.to_bits(),
        (-9.5e38_f64).to_bits()
    );
    assert_eq!(m_keep_l.m_center.to_bits(), want_mass_center.to_bits());
    assert_eq!(m_keep_l.r_center.to_bits(), want_radius_center.to_bits());
    assert_eq!(m_keep_l.l_center.to_bits(), 1.23e31_f64.to_bits());
    assert_eq!(m_keep_l.v_center.to_bits(), 0.0_f64.to_bits());
    assert_eq!(m_keep_l.ye_start.len(), k_cut);
    assert_eq!(m_keep_l.eps_nuc_start.len(), k_cut);
    assert_eq!(m_keep_l.lum_by_category_start.len(), k_cut);
    assert_eq!(m_keep_l.lnd_start.len(), k_cut);
    assert_eq!(
        m_keep_l.lnd_start[k_cut - 1].to_bits(),
        m_keep_l.lnd[k_cut - 1].to_bits()
    );

    let mut m_adj_l = base;
    m_adj_l.l_center = 1.23e31;
    m_adj_l.job.remove_center_adjust_l_center = true;
    let want_lum_center = m_adj_l.lum[k_cut];
    do_remove_center(&mut m_adj_l, k_cut, &mut extras)
        .expect("cut with adjust_l_center = true");
    assert_eq!(m_adj_l.l_center.to_bits(), want_lum_center.to_bits());
}

#[test]
fn test_do_remove_center_velocity_bcs_entropy_units_and_errors() {
    let mut extras = DefaultStarExtras;
    let base = default_zams();
    let nz0 = base.nz;
    let k_cut = nz0 / 2;

    // Velocity BC toggles for u_flag and v_flag + non-finite guard.
    let mut m_u = base.clone();
    m_u.controls.u_flag = true;
    m_u.u = vec![0.0; nz0];
    m_u.u[k_cut] = -4.2e7;
    m_u.job.remove_center_set_zero_v_center = false;
    do_remove_center(&mut m_u, k_cut, &mut extras)
        .expect("u_flag boundary copy");
    assert_eq!(m_u.v_center.to_bits(), (-4.2e7_f64).to_bits());

    let mut m_u_zero = base.clone();
    m_u_zero.controls.u_flag = true;
    m_u_zero.u = vec![0.0; nz0];
    m_u_zero.u[k_cut] = -4.2e7;
    m_u_zero.job.remove_center_set_zero_v_center = true;
    do_remove_center(&mut m_u_zero, k_cut, &mut extras)
        .expect("u_flag set_zero_v_center override");
    assert_eq!(m_u_zero.v_center.to_bits(), 0.0_f64.to_bits());

    let mut m_v = base.clone();
    m_v.controls.v_flag = true;
    m_v.v = vec![0.0; nz0];
    m_v.v[k_cut] = 1.5e6;
    m_v.job.remove_center_set_zero_v_center = false;
    do_remove_center(&mut m_v, k_cut, &mut extras)
        .expect("v_flag boundary copy");
    assert_eq!(m_v.v_center.to_bits(), 1.5e6_f64.to_bits());

    let mut m_bad_u = base.clone();
    m_bad_u.controls.u_flag = true;
    m_bad_u.u = vec![0.0; nz0];
    m_bad_u.u[k_cut] = f64::NAN;
    m_bad_u.job.remove_center_set_zero_v_center = true;
    let err = do_remove_center(&mut m_bad_u, k_cut, &mut extras)
        .expect_err("non-finite u(k) must fail before set_zero_v_center");
    assert!(matches!(err, StarError::InvalidInput(_)));

    // 4. Entropy cut uses exp(ln_s) / (AVO * KERG) in k_B/baryon, ignoring
    //    raw CGS model.entropy, and errors when no zone matches.
    let mut m_ent = base.clone();
    m_ent.entropy.fill(9.99e99);
    let target_k = k_cut;
    let s_kb_target = exp(m_ent.ln_s[target_k]) / (AVO * KERG);
    let s_kb_prev = exp(m_ent.ln_s[target_k - 1]) / (AVO * KERG);
    assert!(
        s_kb_prev > s_kb_target,
        "expected outward entropy to exceed cut point: {s_kb_prev} vs {s_kb_target}"
    );
    do_remove_center_by_entropy(&mut m_ent, s_kb_target, &mut extras)
        .expect("entropy cut finds first zone <= s_kb_target");
    assert_eq!(m_ent.nz, target_k);

    let mut m_nomatch = base.clone();
    let err = do_remove_center_by_entropy(&mut m_nomatch, -1.0, &mut extras)
        .expect_err("no zone matches negative entropy cut");
    assert!(matches!(err, StarError::InvalidInput(_)));

    let mut m_bad_q = base;
    for bad_q in [-0.1, 1.1, f64::NAN] {
        let err = do_remove_inner_fraction_q(&mut m_bad_q, bad_q, &mut extras)
            .expect_err("out-of-range q must fail");
        assert!(matches!(err, StarError::InvalidInput(_)));
    }
}
