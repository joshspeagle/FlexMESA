// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone invariant and driver tests for combined mesh fixes
//! (`#624`: `adjust_mesh_split_merge.f90`, `adjust_mesh_support.f90`,
//! `mesh_adjust.f90`, and `adjust_mesh.f90`).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines
)]

use flexmesa_chem::NUM_CATEGORIES;
use flexmesa_const::{FOUR_THIRDS_PI, LN10, STANDARD_CGRAV};
use flexmesa_star::adjust_mesh_support::set_delta_gval_max_with_controls;
use flexmesa_star::{
    do_adjust_mesh, do_prune_mesh_surface, load_zams_model, remap_onto_dq,
    remesh_split_merge, set_hydro_vars, try_plan_standard_mesh,
    MeshFunctionConfig, StarControls, StarError, StarJobControls, StarModel,
};

fn load_clean_zams_1m() -> StarModel {
    let controls = StarControls {
        initial_mass: 1.0,
        initial_z: 0.02,
        zbase: 0.02,
        okay_to_remesh: true,
        ..StarControls::default()
    };
    let mut model = load_zams_model(controls, StarJobControls::default())
        .expect("load 1 Msun ZAMS");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model
}

fn synthetic_split_merge_model(nz: usize, xmstar: f64) -> StarModel {
    let mut model = load_clean_zams_1m();
    let sp = model.species;
    let xa_zone0 = model.xa[..sp].to_vec();
    model.resize_zones(nz);
    model.controls.rotation_flag = true;
    model.controls.okay_to_remesh = true;
    model.m_center = 0.0;
    model.xmstar = xmstar;
    model.mstar = xmstar;
    model.r_center = 1.0e8;
    model.dq = vec![1.0 / (nz as f64); nz];
    model.set_qs();
    model.set_xqs();
    model.set_m_and_dm();
    model.cgrav.fill(STANDARD_CGRAV);
    model.lnd.fill(0.0);
    model.rho.fill(1.0);
    model.ln_t.fill(16.0);
    model.t.fill(flexmesa_math::exp(16.0));
    model.lum.fill(3.828e33);
    model.l_center = 0.0;
    model.tau.fill(1.0e5);
    model.opacity.fill(1.0);
    let mut v3 = model.r_center * model.r_center * model.r_center;
    for k in (0..nz).rev() {
        let dv = model.dm[k] / model.rho[k];
        v3 += dv / FOUR_THIRDS_PI;
        model.r[k] = flexmesa_math::pow(v3, 1.0 / 3.0);
        model.ln_r[k] = flexmesa_math::log(model.r[k]);
    }
    model.set_m_grav_and_grav();
    for k in 0..nz {
        model.xa[k * sp..(k + 1) * sp].copy_from_slice(&xa_zone0);
    }
    set_hydro_vars(&mut model, false, false).expect("synthetic set_hydro_vars");
    model
}

#[test]
fn test_remesh_split_merge_noop_preserves_r_rmid_xq_mgrav_bitwise() {
    let mut model = load_clean_zams_1m();
    model.controls.mix_wind.use_split_merge_amr = true;
    model.controls.mix_wind.split_merge_amr_max_iters = 0;
    model.controls.mix_wind.split_merge_amr_dq_min = 0.0;
    model.controls.mix_wind.split_merge_amr_dq_max = 1.0;

    // Deliberately mark derived grid arrays to prove a 0-split/0-merge call
    // does not invoke `set_xqs()` or `set_m_grav_and_grav()`.
    model.xq[1] *= 1.0 + 1.0e-9;
    model.r[1] *= 1.0 + 1.0e-9;
    model.rmid[1] *= 1.0 + 1.0e-9;
    model.m_grav[1] *= 1.0 + 1.0e-9;
    model.grav[1] *= 1.0 + 1.0e-9;

    let xq_bits: Vec<u64> = model.xq.iter().map(|v| v.to_bits()).collect();
    let r_bits: Vec<u64> = model.r.iter().map(|v| v.to_bits()).collect();
    let rmid_bits: Vec<u64> = model.rmid.iter().map(|v| v.to_bits()).collect();
    let mg_bits: Vec<u64> = model.m_grav.iter().map(|v| v.to_bits()).collect();
    let g_bits: Vec<u64> = model.grav.iter().map(|v| v.to_bits()).collect();

    let (ns, nm) =
        remesh_split_merge(&mut model).expect("noop remesh_split_merge");
    assert_eq!((ns, nm), (0, 0));

    for k in 0..model.nz {
        assert_eq!(model.xq[k].to_bits(), xq_bits[k], "xq[{k}] changed");
        assert_eq!(model.r[k].to_bits(), r_bits[k], "r[{k}] changed");
        assert_eq!(model.rmid[k].to_bits(), rmid_bits[k], "rmid[{k}] changed");
        assert_eq!(
            model.m_grav[k].to_bits(),
            mg_bits[k],
            "m_grav[{k}] changed"
        );
        assert_eq!(model.grav[k].to_bits(), g_bits[k], "grav[{k}] changed");
    }
}

#[test]
fn test_remesh_split_merge_avoid_repeated_remesh_sets_need_to_setvars() {
    let mut model = load_clean_zams_1m();
    let nz_before = model.nz;
    model.aux.need_to_setvars = false;
    {
        let mw = &mut model.controls.mix_wind;
        mw.use_split_merge_amr = true;
        mw.split_merge_amr_nz_baseline = 5;
        mw.split_merge_amr_maxshort = 1.05;
        mw.split_merge_amr_maxlong = 1.0e10;
        mw.split_merge_amr_max_iters = 200;
        mw.merge_amr_ignore_surface_cells = false;
        mw.merge_amr_ignore_core_cells = false;
        mw.split_merge_amr_avoid_repeated_remesh = true;
    }

    let (ns, nm) =
        remesh_split_merge(&mut model).expect("merge with repeated guard");
    assert_eq!(ns, 0);
    assert!(nm > 0);
    assert!(
        model.nz > nz_before - nm,
        "expected avoid_repeated_remesh early return (nz={}, before={}, nm={})",
        model.nz,
        nz_before,
        nm
    );
    assert!(model.aux.need_to_setvars);
}

#[test]
fn test_remesh_split_merge_eos_failure_returns_non_retriable_terminated() {
    let mut model = load_clean_zams_1m();
    model.controls.mix_wind.use_split_merge_amr = true;
    model.controls.mix_wind.split_merge_amr_nz_baseline = 2000;
    model.controls.mix_wind.split_merge_amr_maxlong = 1.2;
    model.controls.mix_wind.split_merge_amr_maxshort = 1.0e10;
    model.controls.mix_wind.split_merge_amr_max_iters = 1;
    model.controls.mix_wind.split_merge_amr_okay_to_split_nz = true;
    // Unreachably tiny positive energy passes `get_cell_energies` (`ie > 0`)
    // but forces `solve_cell_lnt_for_energy` inside `update_xh_eos` to fail.
    model.energy.fill(1.0);

    let err = remesh_split_merge(&mut model)
        .expect_err("EOS inversion failure must terminate");
    assert!(
        matches!(err, StarError::Terminated { .. }),
        "expected StarError::Terminated, got {err:?}"
    );
    assert!(!err.is_retriable());
}

#[test]
fn test_remesh_split_merge_detects_nonzero_j_from_zero_old_j() {
    // 8 uniform power-of-two zones give exact `dm_bar` cancellation
    // (`old_j == 0`) when splitting center zone 7 (`i1 == nz_old`),
    // leaving `new_j != 0`.
    let xmstar = 8.0 * flexmesa_math::powi(2.0, 104);
    let mut base = synthetic_split_merge_model(8, xmstar);
    base.controls.mix_wind.use_split_merge_amr = true;
    base.controls.mix_wind.split_merge_amr_max_iters = 0;
    base.controls.mix_wind.split_merge_amr_dq_min = 0.0;
    base.controls.mix_wind.split_merge_amr_dq_max = 0.10;
    base.controls.mix_wind.split_merge_amr_avoid_repeated_remesh = true;
    base.dq[0] = 0.0625;
    base.dq[7] = 0.1875;
    base.set_qs();
    base.set_xqs();
    base.set_m_and_dm();

    // Case 1: `old_j == 0.0 && new_j == 0.0` succeeds cleanly.
    let mut zero_j = base.clone();
    zero_j.j_rot.fill(0.0);
    zero_j.omega.fill(0.0);
    let (ns_ok, nm_ok) =
        remesh_split_merge(&mut zero_j).expect("zero J remains zero");
    assert!(ns_ok > 0 && nm_ok == 0);

    // Case 2: exact cancelling `j_rot` on center pair (`dm_bar[6] = 2^104`,
    // `dm_bar[7] = 2^104 * (0.5 + 1.5) = 2 * 2^104`, ratio `1 : 2`) so
    // `old_j == 0.0` bitwise before center split, while center split yields
    // `new_j != 0.0` and must trigger `Err(InvalidInput)`.
    let mut cancel_j = base;
    cancel_j.j_rot.fill(0.0);
    cancel_j.omega.fill(0.0);
    let j_unit = flexmesa_math::powi(2.0, 50);
    cancel_j.j_rot[6] = 2.0 * j_unit;
    cancel_j.j_rot[7] = -j_unit;
    assert_eq!(
        flexmesa_star::total_angular_momentum(&cancel_j).to_bits(),
        0.0_f64.to_bits()
    );
    let err = remesh_split_merge(&mut cancel_j)
        .expect_err("nonzero new_j from zero old_j must error");
    assert!(matches!(err, StarError::InvalidInput(_)), "got {err:?}");
    assert!(!err.is_retriable());
}

#[test]
fn test_do1_dlog_eps_dlogp_non_finite_returns_non_retriable_error() {
    let mut model = load_clean_zams_1m();
    let mut mw = model.controls.mix_wind.clone();
    mw.mesh_dlog_pp_dlogp_extra = 0.5;
    mw.mesh_dlog_eps_min_for_extra = -30.0;
    model.aux.eps_nuc_categories = vec![[0.0; NUM_CATEGORIES]; model.nz];
    model.aux.eps_nuc_categories[0][0] = 1.0e-20;
    model.aux.eps_nuc_categories[1][0] = 1.0e300;
    model.ln_peos[0] = 10.0 * LN10;
    model.ln_peos[1] = 11.0 * LN10;

    let err = set_delta_gval_max_with_controls(&model, &mw, &model.xa)
        .expect_err("non-finite dlog_eps must fail non-retriably");
    assert!(matches!(err, StarError::InvalidInput(_)), "got {err:?}");
    assert!(!err.is_retriable());
}

#[test]
fn test_adjust_mesh_clean_up_fractions_failure_is_retriable() {
    let mut model = load_clean_zams_1m();
    model.xa[0] = 25.0;
    let cfg = MeshFunctionConfig::from_controls(&model.controls);

    let plan_err = try_plan_standard_mesh(&model, &cfg)
        .expect_err("corrupt xa in try_plan_standard_mesh");
    assert!(
        plan_err.is_retriable(),
        "expected retriable, got {plan_err:?}"
    );

    let adjust_err = do_adjust_mesh(&mut model, &cfg)
        .expect_err("corrupt xa in do_adjust_mesh");
    assert!(
        adjust_err.is_retriable(),
        "expected retriable, got {adjust_err:?}"
    );
}

#[test]
fn test_do_split_non_finite_tau_returns_non_retriable_error() {
    let mut model = load_clean_zams_1m();
    model.controls.mix_wind.use_split_merge_amr = true;
    model.controls.mix_wind.split_merge_amr_nz_baseline = 2000;
    model.controls.mix_wind.split_merge_amr_maxlong = 1.2;
    model.controls.mix_wind.split_merge_amr_maxshort = 1.0e10;
    model.controls.mix_wind.split_merge_amr_max_iters = 1;
    model.controls.mix_wind.split_merge_amr_okay_to_split_nz = false;
    // Alternating +-1e308 leaves `tau_l + tau_r == 0.0` finite at entry while
    // `tau_l - tau_r` overflows in `tau[ip0]`, firing the non-finite guard.
    for k in 0..model.nz {
        model.tau[k] = if k % 2 == 0 { 1.0e308 } else { -1.0e308 };
    }

    let err = remesh_split_merge(&mut model)
        .expect_err("non-finite split tau must fail non-retriably");
    assert!(matches!(err, StarError::InvalidInput(_)), "got {err:?}");
    assert!(!err.is_retriable());
}

#[test]
fn test_do_prune_mesh_surface_shifts_all_rotmix_arrays_bitwise() {
    let mut model = load_clean_zams_1m();
    let nz_old = model.nz;
    let skip = 7_usize;
    let nz_new = nz_old - skip;
    for k in 0..nz_old {
        let kf = k as f64;
        model.aux.rotmix.nu_st.push(1.0e3 + kf);
        model.aux.rotmix.d_st.push(2.0e3 + kf);
        model.aux.rotmix.d_dsi.push(3.0e3 + kf);
        model.aux.rotmix.d_sh.push(4.0e3 + kf);
        model.aux.rotmix.d_ssi.push(5.0e3 + kf);
        model.aux.rotmix.d_es.push(6.0e3 + kf);
        model.aux.rotmix.d_gsf.push(7.0e3 + kf);
        model.aux.rotmix.d_omega.push(8.0e3 + kf);
    }
    let old = [
        model.aux.rotmix.nu_st.clone(),
        model.aux.rotmix.d_st.clone(),
        model.aux.rotmix.d_dsi.clone(),
        model.aux.rotmix.d_sh.clone(),
        model.aux.rotmix.d_ssi.clone(),
        model.aux.rotmix.d_es.clone(),
        model.aux.rotmix.d_gsf.clone(),
        model.aux.rotmix.d_omega.clone(),
    ];

    do_prune_mesh_surface(&mut model, nz_new).expect("prune rotmix surface");
    let shifted = [
        &model.aux.rotmix.nu_st,
        &model.aux.rotmix.d_st,
        &model.aux.rotmix.d_dsi,
        &model.aux.rotmix.d_sh,
        &model.aux.rotmix.d_ssi,
        &model.aux.rotmix.d_es,
        &model.aux.rotmix.d_gsf,
        &model.aux.rotmix.d_omega,
    ];
    for (arr_idx, arr) in shifted.iter().enumerate() {
        assert_eq!(arr.len(), nz_new, "rotmix field {arr_idx} len mismatch");
        for k in 0..nz_new {
            assert_eq!(
                arr[k].to_bits(),
                old[arr_idx][k + skip].to_bits(),
                "rotmix field {arr_idx}[{k}] mismatch"
            );
        }
    }
}

#[test]
fn test_do_etrb_reads_xh_w_directly_without_roundtrip_drift() {
    let mut model = load_zams_model(
        StarControls {
            initial_mass: 3.0,
            initial_z: 0.02,
            zbase: 0.02,
            u_flag: true,
            rsp2_flag: true,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 3 Msun rsp2 ZAMS");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");

    let nz_old = model.nz;
    let nvar = model.layout.nvar_hydro;
    let iw = model.layout.i_w.expect("i_w slot under rsp2_flag");
    let w0 = 1.0e4 + 0.125;
    let w_last = 2.0e4 + 0.375;
    model.aux.rsp.et_rsp.fill(999.0);
    model.aux.etrb.fill(999.0);
    for k in 0..nz_old {
        let w = if k == 0 {
            w0
        } else if k + 1 == nz_old {
            w_last
        } else {
            1.5e4 + (k as f64)
        };
        model.xh[k * nvar + iw] = w;
    }

    // Split middle zone only so outer zone `0` and center zone `nz - 1` stay
    // unchanged faces in `do_etrb` / `adjust1_etrb`.
    let mid = nz_old / 2;
    let mut dq_new = Vec::with_capacity(nz_old + 1);
    dq_new.extend_from_slice(&model.dq[..mid]);
    dq_new.push(0.5 * model.dq[mid]);
    dq_new.push(0.5 * model.dq[mid]);
    dq_new.extend_from_slice(&model.dq[(mid + 1)..]);

    remap_onto_dq(&mut model, &dq_new).expect("remap_onto_dq rsp2");
    assert_eq!(model.nz, nz_old + 1);
    assert_eq!(model.aux.rsp.et_rsp[0].to_bits(), (w0 * w0).to_bits());
    assert_eq!(model.aux.etrb[0].to_bits(), (w0 * w0).to_bits());
    assert_eq!(
        model.aux.rsp.et_rsp[model.nz - 1].to_bits(),
        (w_last * w_last).to_bits()
    );
    assert_eq!(
        model.aux.etrb[model.nz - 1].to_bits(),
        (w_last * w_last).to_bits()
    );
}
