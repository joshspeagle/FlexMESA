// SPDX-License-Identifier: GPL-3.0-or-later
//! Unit tests for `StarModel::set_v_flag` (`#579`,
//! `star/private/set_flags.f90:31-119`).

#![forbid(unsafe_code)]
#![allow(
    clippy::float_cmp,
    clippy::manual_midpoint,
    clippy::similar_names,
    clippy::too_many_lines
)]

use flexmesa_star::evolve::star_evolve_step_with_extras;
use flexmesa_star::inlist_runner::{ExtrasDirective, StarExtras};
use flexmesa_star::{
    load_zams_model, PrimaryStateBuffer, StarControls, StarError,
    StarJobControls, StarModel, VarLayout,
};

fn populate_test_hydro_fields(model: &mut StarModel) {
    let nz = model.nz;
    for k in 0..nz {
        let kf = f64::from(u32::try_from(k + 1).expect("nz fits u32"));
        model.lnd[k] = 1.0 + 0.1 * kf;
        model.ln_t[k] = 15.0 + 0.1 * kf;
        model.ln_r[k] = 24.0 - 0.1 * kf;
        model.lum[k] = 1.0e33 * kf;
        model.lnd_start[k] = 0.9 + 0.1 * kf;
        model.ln_t_start[k] = 14.9 + 0.1 * kf;
        model.ln_r_start[k] = 23.9 - 0.1 * kf;
        model.lum_start[k] = 0.9e33 * kf;
    }
}

fn fill_xh_start_from_model(model: &mut StarModel, hp_vals: &[f64]) {
    let nvar = model.layout.nvar_hydro;
    model.xh_start.resize(model.nz * nvar, 0.0);
    for k in 0..model.nz {
        let base = k * nvar;
        model.xh_start[base + model.layout.i_lnd] = model.lnd_start[k];
        model.xh_start[base + model.layout.i_lnt] = model.ln_t_start[k];
        model.xh_start[base + model.layout.i_lnr] = model.ln_r_start[k];
        model.xh_start[base + model.layout.i_lum] = model.lum_start[k];
        if let Some(iv) = model.layout.i_v {
            model.xh_start[base + iv] = model.v_start[k];
        }
        if let Some(iu) = model.layout.i_u {
            model.xh_start[base + iu] = model.u_start[k];
        }
        if let Some(iw) = model.layout.i_w {
            model.xh_start[base + iw] = model.aux.rsp.w_start[k];
        }
        if let Some(ihp) = model.layout.i_hp {
            model.xh_start[base + ihp] = hp_vals[k];
        }
        if let Some(ia) = model.layout.i_alpha_rti {
            model.xh_start[base + ia] = model.aux.alpha_rti_start[k];
        }
    }
}

#[test]
fn test_set_v_flag_noop_when_unchanged() {
    let mut model = StarModel::new(
        3,
        vec![1],
        StarJobControls::default(),
        StarControls::default(),
    );
    model.v_center = 12.5;
    model.v = vec![1.0, 2.0, 3.0];
    model.v_start = vec![4.0, 5.0, 6.0];
    model.v_old = vec![7.0, 8.0, 9.0];
    model.aux.need_to_setvars = false;
    let xh_before = model.xh.clone();
    let xh_start_before = model.xh_start.clone();

    model.set_v_flag(false).expect("noop false");

    assert!(!model.controls.v_flag);
    assert!(!model.aux.need_to_setvars);
    assert_eq!(model.layout.i_v, None);
    assert_eq!(model.layout.nvar_hydro, 4);
    assert_eq!(model.v_center, 12.5);
    assert_eq!(model.v, vec![1.0, 2.0, 3.0]);
    assert_eq!(model.v_start, vec![4.0, 5.0, 6.0]);
    assert_eq!(model.v_old, vec![7.0, 8.0, 9.0]);
    assert_eq!(model.xh, xh_before);
    assert_eq!(model.xh_start, xh_start_before);

    let ctrl_on = StarControls {
        v_flag: true,
        ..StarControls::default()
    };
    let mut model_on =
        StarModel::new(3, vec![1], StarJobControls::default(), ctrl_on);
    model_on.v_center = -5.0;
    model_on.v = vec![10.0, 20.0, 30.0];
    model_on.v_start = vec![11.0, 21.0, 31.0];
    model_on.v_old = vec![12.0, 22.0, 32.0];
    model_on.aux.need_to_setvars = false;
    model_on.set_v_flag(true).expect("noop true");
    assert!(model_on.controls.v_flag);
    assert!(!model_on.aux.need_to_setvars);
    assert_eq!(model_on.layout.i_v, Some(4));
    assert_eq!(model_on.v_center, -5.0);
    assert_eq!(model_on.v, vec![10.0, 20.0, 30.0]);
    assert_eq!(model_on.v_start, vec![11.0, 21.0, 31.0]);
    assert_eq!(model_on.v_old, vec![12.0, 22.0, 32.0]);
}

#[test]
fn test_set_v_flag_false_to_true_hydrostatic() {
    let nz = 4;
    let mut model = StarModel::new(
        nz,
        vec![1, 2],
        StarJobControls::default(),
        StarControls::default(),
    );
    populate_test_hydro_fields(&mut model);
    model.generations = 2;
    model.v_center = 99.0;
    model.v = vec![1.0; nz];
    model.v_start = vec![2.0; nz];
    model.v_old = vec![3.0; nz];
    model.aux.need_to_setvars = false;
    model.pack_xh();
    fill_xh_start_from_model(&mut model, &[]);

    model
        .set_v_flag(true)
        .expect("enable v_flag from hydrostatic");

    assert!(model.controls.v_flag);
    assert!(!model.controls.u_flag);
    assert!(model.aux.need_to_setvars);
    assert_eq!(model.v_center, 0.0);
    assert_eq!(model.v, vec![0.0; nz]);
    assert_eq!(model.v_start, vec![0.0; nz]);
    assert_eq!(model.v_old, vec![0.0; nz]);
    assert_eq!(model.layout.i_v, Some(4));
    assert_eq!(model.layout.i_u, None);
    assert_eq!(model.layout.nvar_hydro, 5);
    assert_eq!(model.xh.len(), nz * 5);
    assert_eq!(model.xh_start.len(), nz * 5);
    for k in 0..nz {
        let base = k * 5;
        assert_eq!(model.xh[base + model.layout.i_lnd], model.lnd[k]);
        assert_eq!(model.xh[base + model.layout.i_lnt], model.ln_t[k]);
        assert_eq!(model.xh[base + model.layout.i_lnr], model.ln_r[k]);
        assert_eq!(model.xh[base + model.layout.i_lum], model.lum[k]);
        assert_eq!(model.xh[base + 4], 0.0);
        assert_eq!(
            model.xh_start[base + model.layout.i_lnd],
            model.lnd_start[k]
        );
        assert_eq!(
            model.xh_start[base + model.layout.i_lnt],
            model.ln_t_start[k]
        );
        assert_eq!(
            model.xh_start[base + model.layout.i_lnr],
            model.ln_r_start[k]
        );
        assert_eq!(
            model.xh_start[base + model.layout.i_lum],
            model.lum_start[k]
        );
        assert_eq!(model.xh_start[base + 4], 0.0);
    }
}

#[test]
fn test_set_v_flag_false_to_true_from_u_flag_face_averages() {
    let nz = 4;
    let ctrl = StarControls {
        u_flag: true,
        ..StarControls::default()
    };
    let mut model =
        StarModel::new(nz, vec![1], StarJobControls::default(), ctrl);
    populate_test_hydro_fields(&mut model);
    model.generations = 2;
    model.u = vec![10.0, 20.0, 40.0, 100.0];
    model.u_start = vec![1.0, 2.0, 3.0, 4.0];
    model.u_old = vec![5.0, 6.0, 7.0, 8.0];
    model.v_center = 42.0;
    model.aux.need_to_setvars = false;
    model.pack_xh();
    fill_xh_start_from_model(&mut model, &[]);

    model
        .set_v_flag(true)
        .expect("enable v_flag from hydrostatic");

    assert!(model.controls.v_flag);
    assert!(!model.controls.u_flag);
    assert!(model.aux.need_to_setvars);
    assert_eq!(model.layout.i_v, Some(4));
    assert_eq!(model.layout.i_u, None);
    assert_eq!(model.layout.nvar_hydro, 5);
    assert_eq!(model.v_center, 0.0);
    assert_eq!(model.v, vec![10.0, 15.0, 30.0, 70.0]);
    assert_eq!(model.v_start, vec![0.0; nz]);
    assert_eq!(model.v_old, vec![0.0; nz]);
    assert_eq!(model.u, vec![10.0, 20.0, 40.0, 100.0]);
    assert_eq!(model.u_start, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(model.u_old, vec![0.0; nz]);
    assert_eq!(model.xh.len(), nz * 5);
    assert_eq!(model.xh_start.len(), nz * 5);
    for k in 0..nz {
        let base = k * 5;
        assert_eq!(model.xh[base + 4], model.v[k]);
        assert_eq!(
            model.xh_start[base + model.layout.i_lnd],
            model.lnd_start[k]
        );
        assert_eq!(
            model.xh_start[base + model.layout.i_lnt],
            model.ln_t_start[k]
        );
        assert_eq!(
            model.xh_start[base + model.layout.i_lnr],
            model.ln_r_start[k]
        );
        assert_eq!(
            model.xh_start[base + model.layout.i_lum],
            model.lum_start[k]
        );
        assert_eq!(model.xh_start[base + 4], 0.0);
    }
}

#[test]
fn test_set_v_flag_true_to_false_preserves_other_hydro_vars() {
    let nz = 3;
    let ctrl = StarControls {
        v_flag: true,
        ..StarControls::default()
    };
    let mut model =
        StarModel::new(nz, vec![1], StarJobControls::default(), ctrl);
    populate_test_hydro_fields(&mut model);
    model.generations = 2;
    model.v = vec![1.0e5, 2.0e5, 3.0e5];
    model.v_start = vec![0.5e5, 1.0e5, 1.5e5];
    model.v_old = vec![0.25e5, 0.5e5, 0.75e5];
    model.aux.need_to_setvars = false;
    model.pack_xh();
    fill_xh_start_from_model(&mut model, &[]);

    model.set_v_flag(false).expect("disable v_flag");

    assert!(!model.controls.v_flag);
    assert!(model.aux.need_to_setvars);
    assert_eq!(model.layout.i_v, None);
    assert_eq!(model.layout.nvar_hydro, 4);
    assert_eq!(model.v, vec![1.0e5, 2.0e5, 3.0e5]);
    assert_eq!(model.v_start, vec![0.5e5, 1.0e5, 1.5e5]);
    assert_eq!(model.v_old, vec![0.0; nz]);
    assert_eq!(model.xh.len(), nz * 4);
    assert_eq!(model.xh_start.len(), nz * 4);
    for k in 0..nz {
        let base = k * 4;
        assert_eq!(model.xh[base + model.layout.i_lnd], model.lnd[k]);
        assert_eq!(model.xh[base + model.layout.i_lnt], model.ln_t[k]);
        assert_eq!(model.xh[base + model.layout.i_lnr], model.ln_r[k]);
        assert_eq!(model.xh[base + model.layout.i_lum], model.lum[k]);
        assert_eq!(
            model.xh_start[base + model.layout.i_lnd],
            model.lnd_start[k]
        );
        assert_eq!(
            model.xh_start[base + model.layout.i_lnt],
            model.ln_t_start[k]
        );
        assert_eq!(
            model.xh_start[base + model.layout.i_lnr],
            model.ln_r_start[k]
        );
        assert_eq!(
            model.xh_start[base + model.layout.i_lum],
            model.lum_start[k]
        );
    }
}

#[test]
fn test_set_v_flag_preserves_rsp2_hp_start_and_handles_zero_zones() {
    let nz = 3;
    let ctrl = StarControls {
        rsp2_flag: true,
        rti_flag: true,
        ..StarControls::default()
    };
    let mut model =
        StarModel::new(nz, vec![1], StarJobControls::default(), ctrl);
    populate_test_hydro_fields(&mut model);
    model.aux.rsp.w_start = vec![10.0, 20.0, 30.0];
    model.aux.rsp.hp_face = vec![1.0e9, 2.0e9, 3.0e9];
    model.aux.alpha_rti_start = vec![0.1, 0.2, 0.3];
    let custom_hp_start = [1.5e9, 2.5e9, 3.5e9];
    model.layout = VarLayout::with_flags(1, false, false, true, true);
    model.pack_xh();
    fill_xh_start_from_model(&mut model, &custom_hp_start);

    model
        .set_v_flag(true)
        .expect("enable v_flag from hydrostatic");
    let nvar_on = model.layout.nvar_hydro;
    assert_eq!(nvar_on, 8);
    let iw_on = model.layout.i_w.expect("rsp2 w slot");
    let ihp_on = model.layout.i_hp.expect("rsp2 hp slot");
    let ia_on = model.layout.i_alpha_rti.expect("rti slot");
    for (k, &expected_hp) in custom_hp_start.iter().enumerate() {
        let base = k * nvar_on;
        assert_eq!(model.xh_start[base + iw_on], model.aux.rsp.w_start[k]);
        assert_eq!(model.xh_start[base + ihp_on], expected_hp);
        assert_eq!(model.xh_start[base + ia_on], model.aux.alpha_rti_start[k]);
    }

    model.set_v_flag(false).expect("disable v_flag");
    let nvar_off = model.layout.nvar_hydro;
    assert_eq!(nvar_off, 7);
    let ihp_off = model.layout.i_hp.expect("rsp2 hp slot after toggle off");
    for (k, &expected_hp) in custom_hp_start.iter().enumerate() {
        let base = k * nvar_off;
        assert_eq!(model.xh_start[base + ihp_off], expected_hp);
    }

    let mut empty = StarModel::new(
        0,
        vec![1],
        StarJobControls::default(),
        StarControls::default(),
    );
    empty.set_v_flag(true).expect("enable empty");
    assert!(empty.controls.v_flag);
    assert_eq!(empty.xh, [] as [f64; 0]);
    assert_eq!(empty.xh_start, [] as [f64; 0]);
    empty.set_v_flag(false).expect("disable empty");
    assert!(!empty.controls.v_flag);
}

#[test]
fn test_set_v_flag_shifts_unclamped_rsp2_rti_columns_and_rejects_mismatches() {
    let nz = 3;
    let ctrl = StarControls {
        u_flag: true,
        rsp2_flag: true,
        rti_flag: true,
        ..StarControls::default()
    };
    let mut model =
        StarModel::new(nz, vec![1], StarJobControls::default(), ctrl);
    populate_test_hydro_fields(&mut model);
    model.aux.rsp.hp_face = vec![1.0e9, 2.0e9, 3.0e9];
    let custom_hp_start = [0.25, 2.5e9, -3.5];
    let custom_w_start = [-1.5, 0.0, 4.25];
    let custom_alpha_start = [-0.25, 0.2, 1.5];
    model.aux.rsp.w_start = custom_w_start.to_vec();
    model.aux.alpha_rti_start = custom_alpha_start.to_vec();
    model.layout = VarLayout::with_flags(1, false, true, true, true);
    let old_nvar = model.layout.nvar_hydro;
    let iu_old = model.layout.i_u.expect("u_flag slot");
    let iw_old = model.layout.i_w.expect("rsp2 w slot");
    let ihp_old = model.layout.i_hp.expect("rsp2 hp slot");
    let ia_old = model.layout.i_alpha_rti.expect("rti slot");
    // Write unclamped values into `xh` and distinct values in `u` vs
    // `xh[:, i_u]` to prove `set_v_flag` shifts raw columns without `pack_xh`
    // clamps and face-averages `xh[:, i_u]` directly.
    let raw_xh_u = [10.0, 20.0, 40.0];
    let raw_xh_w = [-2.0, 0.0, 3.0];
    let raw_xh_hp = [0.25, 0.5, -1.0];
    let raw_xh_alpha = [-0.5, 0.0, 0.75];
    model.u = vec![999.0; nz];
    for k in 0..nz {
        let base = k * old_nvar;
        model.xh[base + model.layout.i_lnd] = model.lnd[k];
        model.xh[base + model.layout.i_lnt] = model.ln_t[k];
        model.xh[base + model.layout.i_lnr] = model.ln_r[k];
        model.xh[base + model.layout.i_lum] = model.lum[k];
        model.xh[base + iu_old] = raw_xh_u[k];
        model.xh[base + iw_old] = raw_xh_w[k];
        model.xh[base + ihp_old] = raw_xh_hp[k];
        model.xh[base + ia_old] = raw_xh_alpha[k];
    }
    fill_xh_start_from_model(&mut model, &custom_hp_start);

    model
        .set_v_flag(true)
        .expect("u_flag -> v_flag with rsp2 + rti");
    assert_eq!(model.v, vec![10.0, 15.0, 30.0]);
    let nvar_on = model.layout.nvar_hydro;
    assert_eq!(nvar_on, 8);
    let iw_on = model.layout.i_w.expect("rsp2 w slot");
    let ihp_on = model.layout.i_hp.expect("rsp2 hp slot");
    let ia_on = model.layout.i_alpha_rti.expect("rti slot");
    for k in 0..nz {
        let base = k * nvar_on;
        assert_eq!(model.xh[base + iw_on].to_bits(), raw_xh_w[k].to_bits());
        assert_eq!(model.xh[base + ihp_on].to_bits(), raw_xh_hp[k].to_bits());
        assert_eq!(model.xh[base + ia_on].to_bits(), raw_xh_alpha[k].to_bits());
        assert_eq!(
            model.xh_start[base + iw_on].to_bits(),
            custom_w_start[k].to_bits()
        );
        assert_eq!(
            model.xh_start[base + ihp_on].to_bits(),
            custom_hp_start[k].to_bits()
        );
        assert_eq!(
            model.xh_start[base + ia_on].to_bits(),
            custom_alpha_start[k].to_bits()
        );
    }

    model
        .set_v_flag(false)
        .expect("disable v_flag preserving unclamped rsp2 + rti");
    let nvar_off = model.layout.nvar_hydro;
    assert_eq!(nvar_off, 7);
    let iw_off = model.layout.i_w.expect("rsp2 w slot after toggle off");
    let ihp_off = model.layout.i_hp.expect("rsp2 hp slot after toggle off");
    let ia_off = model.layout.i_alpha_rti.expect("rti slot after toggle off");
    for k in 0..nz {
        let base = k * nvar_off;
        assert_eq!(model.xh[base + iw_off].to_bits(), raw_xh_w[k].to_bits());
        assert_eq!(model.xh[base + ihp_off].to_bits(), raw_xh_hp[k].to_bits());
        assert_eq!(
            model.xh[base + ia_off].to_bits(),
            raw_xh_alpha[k].to_bits()
        );
        assert_eq!(
            model.xh_start[base + ihp_off].to_bits(),
            custom_hp_start[k].to_bits()
        );
    }

    let mut bad = model.clone();
    bad.xh_start.pop();
    let xh_before = bad.xh.clone();
    let xh_start_before = bad.xh_start.clone();
    let layout_before = bad.layout.clone();
    let err = bad
        .set_v_flag(true)
        .expect_err("truncated xh_start must fail loudly");
    assert!(matches!(err, StarError::InvalidInput(_)));
    assert!(!bad.controls.v_flag);
    assert_eq!(bad.xh, xh_before);
    assert_eq!(bad.xh_start, xh_start_before);
    assert_eq!(bad.layout, layout_before);

    let mut bad_v = model;
    bad_v.v.pop();
    let err_v = bad_v
        .set_v_flag(true)
        .expect_err("truncated v vector must fail loudly");
    assert!(matches!(err_v, StarError::InvalidInput(_)));
}

#[test]
fn test_set_v_flag_preserves_old_velocities_before_new_generation_and_when_dq_old_differs(
) {
    let nz = 3;
    let ctrl = StarControls {
        u_flag: true,
        ..StarControls::default()
    };
    let mut pre_gen =
        StarModel::new(nz, vec![1], StarJobControls::default(), ctrl.clone());
    populate_test_hydro_fields(&mut pre_gen);
    pre_gen.generations = 1;
    pre_gen.u = vec![10.0, 20.0, 30.0];
    pre_gen.v_old = vec![11.0, 22.0, 33.0];
    pre_gen.u_old = vec![44.0, 55.0, 66.0];
    pre_gen.pack_xh();
    fill_xh_start_from_model(&mut pre_gen, &[]);

    pre_gen
        .set_v_flag(true)
        .expect("generations <= 1 must preserve v_old and u_old");
    assert_eq!(pre_gen.v_old, vec![11.0, 22.0, 33.0]);
    assert_eq!(pre_gen.u_old, vec![44.0, 55.0, 66.0]);

    let mut remeshed =
        StarModel::new(nz, vec![1], StarJobControls::default(), ctrl);
    populate_test_hydro_fields(&mut remeshed);
    remeshed.generations = 2;
    remeshed.dq_old = vec![0.5, 0.5];
    remeshed.v_old = vec![11.0, 22.0];
    remeshed.u_old = vec![44.0, 55.0];
    remeshed.pack_xh();
    fill_xh_start_from_model(&mut remeshed, &[]);

    remeshed
        .set_v_flag(true)
        .expect("dq_old.len() != nz must allow pre-remesh v_old/u_old");
    assert_eq!(remeshed.v_old, vec![11.0, 22.0]);
    assert_eq!(remeshed.u_old, vec![44.0, 55.0]);
}

struct RetryAfterVFlagExtras {
    calls: usize,
}

impl StarExtras for RetryAfterVFlagExtras {
    fn extras_check_model(&mut self, model: &mut StarModel) -> ExtrasDirective {
        self.calls += 1;
        if self.calls == 1 {
            model
                .set_v_flag(true)
                .expect("toggle v_flag inside extras_check_model");
            return ExtrasDirective::Retry;
        }
        ExtrasDirective::KeepGoing
    }
}

#[test]
fn test_primary_state_buffer_and_retry_with_restore_mesh_preserve_v_flag_layout_sync(
) {
    let ctrl = StarControls {
        initial_mass: 1.0,
        zbase: 0.02,
        u_flag: true,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(ctrl, StarJobControls::default()).expect("zams u_flag");
    model.save_start_values().expect("save_start_values");
    model.save_old_values();
    let snap = PrimaryStateBuffer::from_model(&model);
    let expected_layout = model.layout.clone();

    model
        .set_v_flag(true)
        .expect("enable v_flag before restore_into");
    assert!(model.controls.v_flag);
    assert!(!model.controls.u_flag);
    snap.restore_into(&mut model).expect("restore_into");
    assert!(!model.controls.v_flag);
    assert!(model.controls.u_flag);
    assert_eq!(model.layout, expected_layout);
    assert_eq!(model.xh.len(), model.nz * model.layout.nvar_hydro);

    let mut step_ctrl = StarControls {
        initial_mass: 1.0,
        zbase: 0.02,
        mesh_delta_coeff: 0.25,
        okay_to_remesh: true,
        ..StarControls::default()
    };
    step_ctrl.mix_wind.restore_mesh_on_retry = true;
    let mut step_model =
        load_zams_model(step_ctrl, StarJobControls::default()).expect("zams");
    step_model.doing_first_model_of_run = false;
    let mut extras = RetryAfterVFlagExtras { calls: 0 };

    let rep = star_evolve_step_with_extras(
        &mut step_model,
        true,
        false,
        true,
        &mut extras,
    )
    .expect("retry after extras_check_model set_v_flag must succeed");
    assert_eq!(extras.calls, 2);
    assert!(rep.retries >= 1);
    assert!(step_model.controls.v_flag);
    assert!(!step_model.controls.u_flag);
    assert!(step_model.layout.i_v.is_some());
    assert_eq!(
        step_model.xh.len(),
        step_model.nz * step_model.layout.nvar_hydro
    );
    assert_eq!(
        step_model.xh_start.len(),
        step_model.nz * step_model.layout.nvar_hydro
    );
}
