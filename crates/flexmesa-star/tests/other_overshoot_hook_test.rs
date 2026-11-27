// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration tests for `other_overshooting_scheme`,
//! `other_adjust_mlt_gradT_fraction`, and `OverBdyAnchorCache` (`#449`).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names
)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use flexmesa_const::{MixingType, SECYER};
use flexmesa_star::hydro_state::{
    OtherAdjustMltGradtFractionHandle, OtherAdjustMltGradtFractionHook,
    OtherOvershootingSchemeHandle, OtherOvershootingSchemeHook,
};
use flexmesa_star::inlist::InlistLoadOptions;
use flexmesa_star::inlist_runner::{
    run_test_suite_case_with_extras, StarExtras, StarLogsOptions,
};
use flexmesa_star::overshoot::{
    add_overshooting, eval_conv_bdy_k, eval_over_bdy_params, OverBdyAnchorCache,
};
use flexmesa_star::primary_state::PrimaryStateBuffer;
use flexmesa_star::{
    load_zams_model, set_hydro_vars, StarControls, StarError, StarJobControls,
    StarModel,
};

struct TempTestDir(PathBuf);

impl TempTestDir {
    fn new(label: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "flexmesa_star_449_{label}_{}_{n}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("create temp test dir");
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempTestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn load_convective_core_zams(overshoot_scheme: &str) -> StarModel {
    let mut controls = StarControls {
        initial_mass: 2.5,
        initial_z: 0.02,
        zbase: 0.02,
        ..StarControls::default()
    };
    controls.mix_wind.overshoot_d_min = 1.0e-2;
    controls.mix_wind.overshoot_sets[0].scheme = String::from(overshoot_scheme);
    controls.mix_wind.overshoot_sets[0].zone_type = String::from("any");
    controls.mix_wind.overshoot_sets[0].zone_loc = String::from("core");
    controls.mix_wind.overshoot_sets[0].bdy_loc = String::from("top");
    controls.mix_wind.overshoot_sets[0].f = 0.05;
    controls.mix_wind.overshoot_sets[0].f0 = 0.005;
    controls.mix_wind.overshoot_sets[0].delta0 = 1.0;

    let mut model = load_zams_model(controls, StarJobControls::default())
        .expect("load zams");
    model.dt = 1.0e3 * SECYER;
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model.save_start_values().expect("save start values");
    model
}

struct NullOvershootHook;

impl OtherOvershootingSchemeHook for NullOvershootHook {
    fn eval_overshooting(
        &self,
        _model: &StarModel,
        _i: usize,
        _j: usize,
        _d_out: &mut [f64],
        _vc_out: &mut [f64],
    ) -> Result<Option<(usize, usize)>, StarError> {
        Ok(None)
    }
}

struct OutOfRangeOvershootHook;

impl OtherOvershootingSchemeHook for OutOfRangeOvershootHook {
    fn eval_overshooting(
        &self,
        model: &StarModel,
        _i: usize,
        _j: usize,
        _d_out: &mut [f64],
        _vc_out: &mut [f64],
    ) -> Result<Option<(usize, usize)>, StarError> {
        Ok(Some((model.nz, 0)))
    }
}

struct NanOvershootHook;

impl OtherOvershootingSchemeHook for NanOvershootHook {
    fn eval_overshooting(
        &self,
        model: &StarModel,
        i: usize,
        _j: usize,
        d_out: &mut [f64],
        _vc_out: &mut [f64],
    ) -> Result<Option<(usize, usize)>, StarError> {
        let k_cb = eval_conv_bdy_k(model, i)?;
        d_out[k_cb] = f64::NAN;
        Ok(Some((k_cb, 0)))
    }
}

struct StepOtherOvershootHook {
    calls: Arc<AtomicUsize>,
}

impl OtherOvershootingSchemeHook for StepOtherOvershootHook {
    fn eval_overshooting(
        &self,
        model: &StarModel,
        i: usize,
        j: usize,
        d_out: &mut [f64],
        vc_out: &mut [f64],
    ) -> Result<Option<(usize, usize)>, StarError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let set = &model.controls.mix_wind.overshoot_sets[j];
        let cache = OverBdyAnchorCache::from_model(model, i)?;
        let (k_ob, r_ob, d_ob, vc_ob) = cache.eval_at(set.f0)?;
        let hp = cache.hp_cb();
        let d_min = model.controls.mix_wind.overshoot_d_min;
        let dr_limit = set.f * hp;
        let mut k_b = 0_usize;
        for k in (0..=k_ob).rev() {
            let dr = model.r[k] - r_ob;
            if dr <= dr_limit && d_ob >= d_min {
                d_out[k] = set.d0 + set.delta0 * d_ob;
                vc_out[k] = vc_ob;
            } else {
                d_out[k] = 0.0;
                vc_out[k] = 0.0;
                k_b = k;
                break;
            }
        }
        Ok(Some((k_ob, k_b)))
    }
}

struct FixedGradtFractionHook {
    calls: Arc<AtomicUsize>,
    blend: f64,
}

impl OtherAdjustMltGradtFractionHook for FixedGradtFractionHook {
    fn apply(&self, model: &mut StarModel) -> Result<(), StarError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let nz = model.nz;
        for k in 0..nz {
            if model.d_mix[k] > 0.0 {
                model.aux.mixing.adjust_mlt_gradt_fraction[k] = self.blend;
            } else {
                model.aux.mixing.adjust_mlt_gradt_fraction[k] = -1.0;
            }
        }
        Ok(())
    }
}

struct RunnerOvershootExtras {
    overshoot_calls: Arc<AtomicUsize>,
    gradt_calls: Arc<AtomicUsize>,
}

impl StarExtras for RunnerOvershootExtras {
    fn other_overshooting_scheme_handle(
        &self,
    ) -> Option<OtherOvershootingSchemeHandle> {
        Some(OtherOvershootingSchemeHandle::new(StepOtherOvershootHook {
            calls: Arc::clone(&self.overshoot_calls),
        }))
    }

    fn other_adjust_mlt_gradt_fraction_handle(
        &self,
    ) -> Option<OtherAdjustMltGradtFractionHandle> {
        Some(OtherAdjustMltGradtFractionHandle::new(
            FixedGradtFractionHook {
                calls: Arc::clone(&self.gradt_calls),
                blend: 0.35,
            },
        ))
    }
}

#[test]
fn test_other_overshoot_missing_null_oob_and_nan_fail_loudly() {
    let mut model = load_convective_core_zams("none");
    model.controls.mix_wind.overshoot_sets[0].scheme = String::from("other");

    let err_missing = add_overshooting(&mut model)
        .expect_err("overshoot_scheme='other' without hook must fail");
    assert!(
        err_missing
            .to_string()
            .contains("null_other_overshooting_scheme"),
        "unexpected error: {err_missing}"
    );

    model.aux.other_overshooting_scheme =
        Some(OtherOvershootingSchemeHandle::new(NullOvershootHook));
    let err_null = add_overshooting(&mut model).expect_err(
        "hook returning None must match null_other_overshooting_scheme",
    );
    assert!(
        err_null
            .to_string()
            .contains("null_other_overshooting_scheme"),
        "unexpected error: {err_null}"
    );

    model.aux.other_overshooting_scheme =
        Some(OtherOvershootingSchemeHandle::new(OutOfRangeOvershootHook));
    let err_oob = add_overshooting(&mut model)
        .expect_err("out-of-range bounds must fail cleanly");
    assert!(
        err_oob.to_string().contains("out-of-range"),
        "unexpected error: {err_oob}"
    );

    model.aux.other_overshooting_scheme =
        Some(OtherOvershootingSchemeHandle::new(NanOvershootHook));
    let err_nan =
        add_overshooting(&mut model).expect_err("NaN diffusivity must fail");
    assert!(
        err_nan.to_string().contains("non-finite"),
        "unexpected error: {err_nan}"
    );
}

#[test]
fn test_other_overshoot_custom_hook_and_anchor_cache_consistency() {
    let mut model = load_convective_core_zams("none");
    assert!(
        !model.aux.mixing.conv_bdy_loc.is_empty(),
        "2.5 Msun ZAMS should have a convective core"
    );

    let cache = OverBdyAnchorCache::from_model(&model, 0)
        .expect("build OverBdyAnchorCache");
    for f0 in [0.002, 0.005, 0.015] {
        let direct = eval_over_bdy_params(&model, 0, f0).expect("direct eval");
        let cached = cache.eval_at(f0).expect("cached eval");
        assert_eq!(direct.0, cached.0);
        assert!(
            (direct.1 - cached.1).abs() <= 1.0e-15 * direct.1.abs().max(1.0)
        );
        assert!(
            (direct.2 - cached.2).abs() <= 1.0e-15 * direct.2.abs().max(1.0)
        );
        assert!(
            (direct.3 - cached.3).abs() <= 1.0e-15 * direct.3.abs().max(1.0)
        );
    }

    let k_cb = eval_conv_bdy_k(&model, 0).expect("conv_bdy_k");
    let calls = Arc::new(AtomicUsize::new(0));
    model.controls.mix_wind.overshoot_sets[0].scheme = String::from("other");
    model.aux.other_overshooting_scheme =
        Some(OtherOvershootingSchemeHandle::new(StepOtherOvershootHook {
            calls: Arc::clone(&calls),
        }));

    add_overshooting(&mut model)
        .expect("apply custom other_overshooting_scheme");
    assert_eq!(calls.load(Ordering::Relaxed), 1);

    let has_overshoot_face =
        (0..k_cb).any(|k| model.mixing_type[k] == MixingType::OvershootMixing);
    assert!(
        has_overshoot_face,
        "custom other_overshooting_scheme should populate OvershootMixing faces"
    );

    let buf = PrimaryStateBuffer::from_model(&model);
    model.aux.other_overshooting_scheme = None;
    buf.restore_into(&mut model)
        .expect("restore overshoot hook");
    assert!(model.aux.other_overshooting_scheme.is_some());
    add_overshooting(&mut model).expect("invoke restored overshoot hook");
    assert_eq!(calls.load(Ordering::Relaxed), 2);
}

#[test]
fn test_other_adjust_mlt_gradt_fraction_blends_gradt_and_guards() {
    let mut model = load_convective_core_zams("none");
    let calls = Arc::new(AtomicUsize::new(0));
    let blend = 0.25_f64;
    model.aux.other_adjust_mlt_gradt_fraction = Some(
        OtherAdjustMltGradtFractionHandle::new(FixedGradtFractionHook {
            calls: Arc::clone(&calls),
            blend,
        }),
    );

    // Locked mixing info or doing_struct_burn_mix or rsp2_flag skips the hook.
    model.aux.mixing.okay_to_set_mixing_info = false;
    set_hydro_vars(&mut model, false, false).expect("locked mixing info");
    assert_eq!(calls.load(Ordering::Relaxed), 0);

    model.aux.mixing.okay_to_set_mixing_info = true;
    model.aux.doing_struct_burn_mix = true;
    set_hydro_vars(&mut model, false, false).expect("doing_struct_burn_mix");
    assert_eq!(calls.load(Ordering::Relaxed), 0);

    model.aux.doing_struct_burn_mix = false;
    set_hydro_vars(&mut model, false, false).expect("active hook");
    assert_eq!(calls.load(Ordering::Relaxed), 1);

    let k_test = (0..model.nz)
        .find(|&k| {
            model.d_mix[k] > 0.0 && model.aux.mixing.gradt_excess_alpha == 0.0
        })
        .expect("at least one mixed face");
    let expected_gradt = blend * model.aux.mixing.grada_face[k_test]
        + (1.0 - blend) * model.gradr[k_test];
    assert!(
        (model.gradt[k_test] - expected_gradt).abs() <= 1.0e-12,
        "gradt[{}]={:.12e} != expected {:.12e}",
        k_test,
        model.gradt[k_test],
        expected_gradt
    );

    // PrimaryStateBuffer capture + restore preserves both handles cleanly
    // (skipping mixing_info during restore_pre_hydro, then active afterward).
    let buf = PrimaryStateBuffer::from_model(&model);
    model.aux.other_overshooting_scheme = None;
    model.aux.other_adjust_mlt_gradt_fraction = None;
    buf.restore_into(&mut model).expect("restore_into");
    assert!(model.aux.other_adjust_mlt_gradt_fraction.is_some());
    set_hydro_vars(&mut model, false, false).expect("post-restore hook");
    assert!(calls.load(Ordering::Relaxed) >= 2);
}

#[test]
fn test_inlist_runner_installs_handles_and_evolves_with_other_overshoot() {
    let dir = TempTestDir::new("runner");
    let rn_path = dir.path().join("rn");
    let inlist_path = dir.path().join("inlist_other_overshoot");

    fs::write(
        &inlist_path,
        "\
&star_job
  load_saved_model = .false.
  create_pre_main_sequence_model = .false.
  save_model_when_terminate = .false.
/
&kap
  Zbase = 0.02
/
&controls
  initial_mass = 2.5
  initial_z = 0.02
  overshoot_scheme(1) = 'other'
  overshoot_zone_type(1) = 'any'
  overshoot_zone_loc(1) = 'core'
  overshoot_bdy_loc(1) = 'top'
  overshoot_f(1) = 0.055
  overshoot_f0(1) = 0.005
  overshoot_Delta0(1) = 1.0
  overshoot_D_min = 1d-2
  max_model_number = 5
/
",
    )
    .expect("write inlist");
    fs::write(
        &rn_path,
        "#!/bin/bash\ndo_one inlist_other_overshoot final.mod\n",
    )
    .expect("write rn");

    let mut extras = RunnerOvershootExtras {
        overshoot_calls: Arc::new(AtomicUsize::new(0)),
        gradt_calls: Arc::new(AtomicUsize::new(0)),
    };
    let report = run_test_suite_case_with_extras(
        dir.path(),
        true,
        Some(2),
        false,
        StarLogsOptions::default(),
        InlistLoadOptions::default(),
        &mut extras,
    )
    .expect("run case with custom other_overshooting_scheme");

    assert_eq!(report.total_steps, 2);
    assert!(
        extras.overshoot_calls.load(Ordering::Relaxed) > 0,
        "expected other_overshooting_scheme calls during 2 evolution steps"
    );
    assert!(
        extras.gradt_calls.load(Ordering::Relaxed) > 0,
        "expected other_adjust_mlt_gradt_fraction calls during 2 evolution steps"
    );
}
