// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration tests for `#153` `adjust_omega` control-volume remapping parity
//! on `CellType::Unchanged` faces (`star/private/mesh_adjust.f90:232-236,
//! 1943-2112`).

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]

use flexmesa_star::{
    do_adjust_mesh, load_zams_model, set_hydro_vars, total_angular_momentum,
    try_plan_standard_mesh, update_rotation_distortion_factors, CellType,
    MeshFunctionConfig, StarControls, StarJobControls, StarModel,
};

#[allow(clippy::manual_midpoint)]
fn dqbar_profile(dq: &[f64]) -> Vec<f64> {
    let n = dq.len();
    let mut dqbar = vec![0.0; n];
    if n == 0 {
        return dqbar;
    }
    if n == 1 {
        dqbar[0] = dq[0];
        return dqbar;
    }
    dqbar[0] = 0.5 * dq[0];
    for k in 1..(n - 1) {
        dqbar[k] = 0.5 * (dq[k - 1] + dq[k]);
    }
    dqbar[n - 1] = 0.5 * dq[n - 2] + dq[n - 1];
    dqbar
}

fn xout_profile(xq: &[f64], dqbar: &[f64]) -> Vec<f64> {
    let n = xq.len();
    let mut xout = vec![0.0; n];
    if n > 0 {
        xout[0] = xq[0];
        for k in 1..n {
            xout[k] = xout[k - 1] + dqbar[k - 1];
        }
    }
    xout
}

#[allow(clippy::too_many_arguments)]
fn expected_adjust1_omega_jrot(
    k1: usize,
    nz: usize,
    nz_old: usize,
    comes_from: &[usize],
    xout_old: &[f64],
    xout_new: &[f64],
    dqbar_old: &[f64],
    new_point_dqbar: f64,
    j_rot_old: &[f64],
) -> f64 {
    let xq_outer = xout_new[k1 - 1];
    let xq_inner = if k1 < nz {
        xq_outer + new_point_dqbar
    } else {
        1.0
    };
    let mut dq_sum = 0.0_f64;
    let mut j_tot = 0.0_f64;
    let k_outer1 = if xq_outer >= xout_old[nz_old - 1] {
        nz_old
    } else if k1 == 1 {
        1
    } else {
        comes_from[k1 - 2] + 1
    };

    for kk1 in k_outer1..=nz_old {
        let xq1 = if kk1 == nz_old { 1.0 } else { xout_old[kk1] };
        if xq1 <= xq_outer {
            continue;
        }
        let xq0 = xout_old[kk1 - 1];
        if xq0 >= xq_inner {
            if dq_sum < new_point_dqbar && kk1 > 1 {
                let dq = new_point_dqbar - dq_sum;
                dq_sum = new_point_dqbar;
                j_tot += j_rot_old[kk1 - 2] * dq;
            }
            break;
        }
        if xq0 >= xq_outer && xq1 <= xq_inner {
            let mut dq = dqbar_old[kk1 - 1];
            dq_sum += dq;
            if dq_sum > new_point_dqbar {
                dq -= new_point_dqbar - dq_sum;
                dq_sum = new_point_dqbar;
            }
            j_tot += j_rot_old[kk1 - 1] * dq;
        } else if xq0 <= xq_outer && xq1 >= xq_inner {
            let dq = new_point_dqbar;
            dq_sum += dq;
            j_tot += j_rot_old[kk1 - 1] * dq;
        } else {
            let dq = if xq_inner <= xq1 {
                let rem = new_point_dqbar - dq_sum;
                dq_sum = new_point_dqbar;
                rem
            } else {
                let mut ov = (xq1 - xq_outer).max(0.0);
                if dq_sum + ov > new_point_dqbar {
                    ov = new_point_dqbar - dq_sum;
                }
                dq_sum += ov;
                ov
            };
            j_tot += j_rot_old[kk1 - 1] * dq;
        }
        if dq_sum >= new_point_dqbar {
            break;
        }
    }
    if dq_sum > 0.0 {
        j_tot / dq_sum
    } else {
        0.0
    }
}

fn build_rotating_4msun() -> StarModel {
    let ctrl = StarControls {
        initial_mass: 4.0,
        rotation_flag: true,
        zbase: 0.02,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(ctrl, StarJobControls::default()).expect("4 Msun ZAMS");
    let nz = model.nz;
    let denom = f64::from(u32::try_from(nz.saturating_sub(1).max(1)).unwrap());
    for (k, om) in model.omega.iter_mut().enumerate().take(nz) {
        let frac = f64::from(u32::try_from(k).unwrap()) / denom;
        *om = 1.2e-4 * (1.0 + 1.5 * frac);
    }
    update_rotation_distortion_factors(&mut model);
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model
}

#[test]
fn test_do_adjust_mesh_runs_adjust1_omega_on_unchanged_faces_bit_exact() {
    let mut model = build_rotating_4msun();
    model.controls.mesh_delta_coeff = 1.5;
    let cfg = MeshFunctionConfig::from_controls(&model.controls);
    let plan = try_plan_standard_mesh(&model, &cfg)
        .expect("try_plan_standard_mesh")
        .expect("non-empty mesh plan");

    let nz_old = plan.nz_old;
    let dqbar_old = dqbar_profile(&plan.dq_old);
    let xout_old = xout_profile(&plan.xq_old, &dqbar_old);
    let j_rot_old = model.j_rot.clone();
    let lnr_old = model.ln_r.clone();
    let j_tot_before = total_angular_momentum(&model);

    let (splits, merges) =
        do_adjust_mesh(&mut model, &cfg).expect("do_adjust_mesh");
    assert!(splits + merges > 0, "expected standard remesh changes");
    assert_eq!(model.nz, plan.dq_new.len());

    let nz = model.nz;
    let dqbar_new = dqbar_profile(&model.dq);
    let xout_new = xout_profile(&model.xq, &dqbar_new);

    let mut unchanged_faces = 0_usize;
    let mut changed_bits_on_unchanged = 0_usize;
    for k in 0..nz {
        let expected_jr = expected_adjust1_omega_jrot(
            k + 1,
            nz,
            nz_old,
            &plan.comes_from,
            &xout_old,
            &xout_new,
            &dqbar_old,
            dqbar_new[k],
            &j_rot_old,
        );
        assert_eq!(
            model.j_rot[k].to_bits(),
            expected_jr.to_bits(),
            "j_rot bit mismatch vs adjust1_omega at zone k={k} ({:?})",
            plan.cell_type[k],
        );
        assert_eq!(
            model.omega[k].to_bits(),
            (model.j_rot[k] / model.aux.i_rot[k]).to_bits(),
            "omega must equal j_rot / i_rot at zone k={k}",
        );

        let ko = plan.comes_from[k];
        if plan.cell_type[k] == CellType::Unchanged
            && (k == 0 || plan.cell_type[k - 1] == CellType::Unchanged)
            && model.ln_r[k].to_bits() == lnr_old[ko].to_bits()
        {
            unchanged_faces += 1;
            if model.j_rot[k].to_bits() != j_rot_old[ko].to_bits() {
                changed_bits_on_unchanged += 1;
            }
        }
    }

    assert!(
        unchanged_faces > 0,
        "test must exercise CellType::Unchanged faces alongside splits/merges"
    );
    assert!(
        changed_bits_on_unchanged > 0,
        "expected adjust1_omega control-volume integration to shift j_rot on at least one CellType::Unchanged face"
    );

    let j_tot_after = total_angular_momentum(&model);
    let rel_j = (j_tot_after - j_tot_before).abs() / j_tot_before.abs();
    assert!(
        rel_j < 1.0e-12,
        "total angular momentum drift too large: rel={rel_j:.3e}"
    );
}
