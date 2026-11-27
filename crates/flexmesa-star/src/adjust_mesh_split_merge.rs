// SPDX-License-Identifier: GPL-3.0-or-later
//! Hydrodynamic Adaptive Mesh Refinement (`remesh_split_merge`, `amr`,
//! `biggest_smallest`, `do_merge`, `do_split`, `revise_star_radius`, `minmod`,
//! and `get1_grad` in `star/private/adjust_mesh_split_merge.f90:1-1461`).

use flexmesa_const::{FOUR_PI, FOUR_THIRDS_PI, LN10, ONE_THIRD};

use crate::adjust_mesh_support::{
    fortran_maxloc_0based, fortran_minloc_0based,
};
use crate::controls::StarError;
use crate::hydro_vars::set_hydro_vars;
use crate::mesh_remap::{solve_cell_lnt_for_energy, total_angular_momentum};
use crate::model::StarModel;

/// Three-argument slope limiter (`minmod` in
/// `star/private/adjust_mesh_split_merge.f90:1363-1376`).
#[must_use]
pub fn minmod3(a: f64, b: f64, c: f64) -> f64 {
    let mut m = a;
    if a * b < 0.0 {
        return 0.0;
    }
    if b.abs() < m.abs() {
        m = b;
    }
    if b * c < 0.0 {
        return 0.0;
    }
    if c.abs() < m.abs() {
        m = c;
    }
    m
}

/// Slope limiter over `(vL, vC, vR)` (`get1_grad` in
/// `star/private/adjust_mesh_split_merge.f90:1379-1392`).
#[must_use]
pub fn get1_grad(
    v_l: f64,
    v_c: f64,
    v_r: f64,
    d_left: f64,
    d_cntr: f64,
    d_rght: f64,
) -> f64 {
    if d_left <= 0.0 || d_cntr <= 0.0 || d_rght <= 0.0 {
        return 0.0;
    }
    let s_l = (v_c - v_l) / d_left;
    let s_r = (v_r - v_c) / d_rght;
    let s_c = (v_r - v_l) / d_cntr;
    minmod3(s_l, s_c, s_r)
}

#[inline]
fn get_dv(model: &StarModel, k0: usize) -> f64 {
    let rk = model.r[k0];
    let rl = if k0 + 1 == model.nz {
        model.r_center
    } else {
        model.r[k0 + 1]
    };
    FOUR_THIRDS_PI * (rk * rk * rk - rl * rl * rl)
}

fn get_star_pe(model: &StarModel) -> f64 {
    let nz = model.nz;
    let mut tot_pe = 0.0_f64;
    for k0 in 0..nz {
        let r_l = if k0 + 1 == nz {
            model.r_center
        } else {
            model.r[k0 + 1]
        };
        let r_c = 0.5 * (r_l + model.r[k0]);
        let dm = model.dm[k0];
        let m_c = model.m[k0] - 0.5 * dm;
        let pe = -model.cgrav[k0] * m_c * dm / r_c;
        tot_pe += pe;
    }
    tot_pe
}

fn revise_star_radius(model: &mut StarModel, star_pe0: f64, star_pe1: f64) {
    if star_pe1 == 0.0 || star_pe0 == star_pe1 {
        return;
    }
    let frac = star_pe1 / star_pe0;
    for k0 in 0..model.nz {
        model.r[k0] *= frac;
        model.ln_r[k0] = flexmesa_math::log(model.r[k0]);
    }
    model.r_center *= frac;
}

fn set_rmid_cell(model: &mut StarModel, k0: usize) {
    let r_inner = if k0 + 1 < model.nz {
        model.r[k0 + 1]
    } else {
        model.r_center
    };
    model.rmid[k0] = if model.controls.rsp_flag {
        0.5 * (model.r[k0] + r_inner)
    } else {
        flexmesa_math::pow(
            0.5 * (flexmesa_math::pow3(model.r[k0])
                + flexmesa_math::pow3(r_inner)),
            ONE_THIRD,
        )
    };
}

fn get_cell_energies(
    model: &StarModel,
    k0: usize,
) -> Result<(f64, f64, f64, f64, f64), StarError> {
    let dm = model.dm[k0];
    let ke = if model.controls.u_flag {
        0.5 * dm * model.u[k0] * model.u[k0]
    } else if model.controls.v_flag {
        let v0 = model.v[k0];
        let v1 = if k0 + 1 < model.nz {
            model.v[k0 + 1]
        } else {
            model.v_center
        };
        0.25 * dm * (v0 * v0 + v1 * v1)
    } else {
        0.0
    };
    let ie = model.energy[k0] * dm;
    let etrb = if model.controls.rsp2_flag {
        model.aux.rsp.et_rsp.get(k0).copied().unwrap_or(0.0) * dm
    } else {
        0.0
    };
    let pe = if model.cgrav[k0] == 0.0 {
        0.0
    } else {
        let r_l = if k0 + 1 == model.nz {
            model.r_center
        } else {
            model.r[k0 + 1]
        };
        let r_c = 0.5 * (r_l + model.r[k0]);
        let m_c = model.m[k0] - 0.5 * dm;
        -model.cgrav[k0] * m_c * dm / r_c
    };
    let etot = ie + ke + pe;
    if !(etot + ie + ke + pe).is_finite() || ie <= 0.0 || ke < 0.0 {
        return Err(StarError::InvalidInput(format!(
            "remesh_split_merge::get_cell_energies invalid energies at zone {}: IE={ie}, KE={ke}, PE={pe}",
            k0 + 1
        )));
    }
    Ok((etot, ke, pe, ie, etrb))
}

fn split1_non_negative(
    val: f64,
    grad: f64,
    dr: f64,
    dv: f64,
    dvr: f64,
    dvl: f64,
) -> (f64, f64) {
    if val < 1.0e-99 || grad == 0.0 {
        return (val, val);
    }
    let x_l = (val - (grad * dr) / 4.0).clamp(0.0, 1.0);
    let x_r = (val + (grad * dr) / 4.0).clamp(0.0, 1.0);
    let x_ml = x_l * dvl;
    let x_mr = x_r * dvr;
    let x_m = val * dv;
    if x_m < 1.0e-99 || x_ml + x_mr < 1.0e-99 {
        (val, val)
    } else {
        let f = x_m / (x_ml + x_mr);
        (f * x_l, f * x_r)
    }
}

fn update_xh_eos(model: &mut StarModel, k0: usize) -> Result<(), StarError> {
    let dv = get_dv(model, k0);
    if dv <= 0.0 {
        return Err(StarError::InvalidInput(format!(
            "remesh_split_merge::update_xh_eos non-positive dV={dv} at zone {}",
            k0 + 1
        )));
    }
    let rho = model.dm[k0] / dv;
    model.lnd[k0] = flexmesa_math::log(rho);
    model.rho[k0] = flexmesa_math::exp(model.lnd[k0]);
    let log_rho = model.lnd[k0] / LN10;
    let species = model.species;
    let xa_k = model.xa[k0 * species..(k0 + 1) * species].to_vec();
    let (new_lnt, _rev_e) = solve_cell_lnt_for_energy(
        model,
        &xa_k,
        log_rho,
        model.energy[k0],
        model.ln_t[k0],
        1.0e-11,
    )?;
    model.ln_t[k0] = new_lnt;
    model.t[k0] = flexmesa_math::exp(new_lnt);
    Ok(())
}

/// Evaluates `(too_big, too_small, i_too_big_1based, i_too_small_1based)`
/// (`biggest_smallest` in `star/private/adjust_mesh_split_merge.f90:243-480`).
pub(crate) fn biggest_smallest(
    model: &StarModel,
    tau_center: f64,
    has_undergone: &[bool],
) -> Result<(f64, f64, isize, isize), StarError> {
    let mw = &model.controls.mix_wind;
    if mw.split_merge_amr_flipped_hybrid_zoning {
        // Upstream `star/private/adjust_mesh_split_merge.f90:298, 362` subtracts
        // `R_center` in cm from `log(max(1d0, r_core_cm))` and reads uninitialized
        // stack variable `target_dr_core` (deferred bug #242).
        return Err(StarError::InvalidInput(String::from(
            "split_merge_amr_flipped_hybrid_zoning = .true. reads uninitialized target_dr_core in MESA 26.4.1 (deferred upstream bug #242)",
        )));
    }

    let nz = model.nz;
    let hybrid_zoning = mw.split_merge_amr_hybrid_zoning;
    let log_zoning = mw.split_merge_amr_log_zoning;
    let logtau_zoning = mw.split_merge_amr_logtau_zoning;

    let mut nz_baseline = mw.split_merge_amr_nz_baseline;
    let mut nz_r_core = mw.split_merge_amr_nz_r_core;
    if mw.split_merge_amr_mesh_delta_coeff != 1.0 {
        nz_baseline = ((nz_baseline as f64)
            / mw.split_merge_amr_mesh_delta_coeff)
            .trunc() as isize;
        nz_r_core = ((nz_r_core as f64) / mw.split_merge_amr_mesh_delta_coeff)
            .trunc() as isize;
    }

    let r_core_cm = if mw.split_merge_amr_r_core_cm > 0.0 {
        mw.split_merge_amr_r_core_cm
    } else if mw.split_merge_amr_nz_r_core_fraction > 0.0 {
        model.r_center
            + mw.split_merge_amr_nz_r_core_fraction
                * (model.r[0] - model.r_center)
    } else {
        model.r_center
    };

    let dq_min = mw.split_merge_amr_dq_min;
    let dq_max = mw.split_merge_amr_dq_max;

    let mut target_dr_core = 0.0;
    let mut target_dlnr_envelope = 0.0;
    let mut xmin = 0.0;
    let mut inner_dx_baseline = 0.0;

    if hybrid_zoning {
        if nz_r_core == 0 || nz_baseline == nz_r_core {
            return Err(StarError::InvalidInput(format!(
                "split_merge_amr_hybrid_zoning zero denominator: nz_r_core={nz_r_core}, nz_baseline={nz_baseline}"
            )));
        }
        target_dr_core = (r_core_cm - model.r_center) / (nz_r_core as f64);
        target_dlnr_envelope = (model.ln_r[0]
            - flexmesa_math::log((1.0_f64).max(r_core_cm)))
            / ((nz_baseline - nz_r_core) as f64);
    } else if logtau_zoning {
        if nz_baseline == 0 {
            return Err(StarError::InvalidInput(String::from(
                "split_merge_amr_nz_baseline must be nonzero",
            )));
        }
        xmin = flexmesa_math::log(tau_center);
        let xmax = flexmesa_math::log(model.tau[0]);
        inner_dx_baseline = (xmin - xmax) / (nz_baseline as f64);
    } else if log_zoning {
        if nz_baseline == 0 {
            return Err(StarError::InvalidInput(String::from(
                "split_merge_amr_nz_baseline must be nonzero",
            )));
        }
        xmin = flexmesa_math::log((1.0_f64).max(model.r_center));
        let xmax = model.ln_r[0];
        inner_dx_baseline = (xmax - xmin) / (nz_baseline as f64);
    } else {
        if nz_baseline == 0 {
            return Err(StarError::InvalidInput(String::from(
                "split_merge_amr_nz_baseline must be nonzero",
            )));
        }
        xmin = model.r_center;
        let xmax = model.r[0];
        inner_dx_baseline = (xmax - xmin) / (nz_baseline as f64);
    }

    let has_vel = model.controls.u_flag || model.controls.v_flag;
    let vel_at = |idx0: usize| -> f64 {
        if model.controls.u_flag {
            model.u[idx0]
        } else {
            model.v[idx0]
        }
    };
    let r_vel_at = |idx0: usize| -> f64 {
        if model.controls.u_flag {
            model.rmid[idx0]
        } else {
            model.r[idx0]
        }
    };

    let mut too_big = 0.0_f64;
    let mut too_small = 0.0_f64;
    let mut i_too_big: isize = -1;
    let mut i_too_small: isize = -1;
    let mut x_r = xmin;

    for k1 in (1..=nz).rev() {
        let k0 = k1 - 1;
        let mut x_l = x_r;
        let dx_baseline;
        if hybrid_zoning {
            if model.r[k0] < r_core_cm {
                x_r = model.r[k0];
                x_l = if k1 == nz {
                    model.r_center
                } else {
                    model.r[k0 + 1]
                };
                dx_baseline = target_dr_core;
            } else {
                x_r = flexmesa_math::log(model.r[k0]);
                x_l = if k1 == nz {
                    flexmesa_math::log((1.0_f64).max(model.r_center))
                } else {
                    flexmesa_math::log(model.r[k0 + 1])
                };
                dx_baseline = target_dlnr_envelope;
            }
        } else if logtau_zoning {
            x_r = flexmesa_math::log(model.tau[k0]);
            dx_baseline = inner_dx_baseline;
        } else if log_zoning {
            x_r = flexmesa_math::log(model.r[k0]);
            dx_baseline = inner_dx_baseline;
        } else {
            x_r = model.r[k0];
            dx_baseline = inner_dx_baseline;
        }

        if mw.split_merge_amr_avoid_repeated_remesh && has_undergone[k0] {
            continue;
        }

        let mut dx_actual = x_r - x_l;
        if logtau_zoning {
            dx_actual = -dx_actual;
        }

        if mw.split_amr_ignore_core_cells
            && model.ln_t[k0] / LN10 >= mw.split_amr_logt_for_ignore_core_cells
        {
            continue;
        }

        let oversize_ratio = dx_actual / dx_baseline;
        if too_big < oversize_ratio
            && model.dq[k0] > 5.0 * dq_min
            && (k1 < nz || mw.split_merge_amr_okay_to_split_nz)
            && (k1 > 1 || mw.split_merge_amr_okay_to_split_1)
        {
            too_big = oversize_ratio;
            i_too_big = k1 as isize;
        }

        if mw.merge_amr_ignore_surface_cells
            && (k1 as isize) <= mw.merge_amr_k_for_ignore_surface_cells
        {
            continue;
        }
        if mw.merge_amr_ignore_core_cells
            && model.ln_t[k0] / LN10 >= mw.merge_amr_logt_for_ignore_core_cells
        {
            continue;
        }

        let mut undersize_ratio = if dx_actual.abs() > 0.0 {
            (dx_baseline / dx_actual).max(dq_min / model.dq[k0])
        } else {
            dq_min / model.dq[k0]
        };

        if mw.merge_amr_max_abs_du_div_cs >= 0.0 {
            let mut du_div_cs_limit_flag = false;
            if !mw.merge_amr_du_div_cs_limit_only_for_compression {
                du_div_cs_limit_flag = true;
            } else if has_vel {
                if k1 < nz
                    && vel_at(k0 + 1) * flexmesa_math::pow2(r_vel_at(k0 + 1))
                        > vel_at(k0) * flexmesa_math::pow2(r_vel_at(k0))
                {
                    du_div_cs_limit_flag = true;
                }
                if !du_div_cs_limit_flag
                    && k1 > 1
                    && vel_at(k0) * flexmesa_math::pow2(r_vel_at(k0))
                        > vel_at(k0 - 1) * flexmesa_math::pow2(r_vel_at(k0 - 1))
                {
                    du_div_cs_limit_flag = true;
                }
            }

            let abs_du_div_cs = if du_div_cs_limit_flag && has_vel {
                if k1 == 1 {
                    (vel_at(0) - vel_at(1)).abs() / model.csound[0]
                } else if k1 == nz {
                    (vel_at(nz - 2) - vel_at(nz - 1)).abs()
                        / model.csound[nz - 1]
                } else {
                    (vel_at(k0) - vel_at(k0 + 1))
                        .abs()
                        .max((vel_at(k0) - vel_at(k0 - 1)).abs())
                        / model.csound[k0]
                }
            } else {
                0.0
            };

            if du_div_cs_limit_flag {
                if mw.merge_amr_inhibit_at_jumps {
                    if abs_du_div_cs > mw.merge_amr_max_abs_du_div_cs {
                        undersize_ratio *=
                            mw.merge_amr_max_abs_du_div_cs / abs_du_div_cs;
                    }
                    if too_small < undersize_ratio
                        && model.dq[k0] < dq_max / 5.0
                    {
                        too_small = undersize_ratio;
                        i_too_small = k1 as isize;
                    }
                } else if too_small < undersize_ratio
                    && abs_du_div_cs <= mw.merge_amr_max_abs_du_div_cs
                    && model.dq[k0] < dq_max / 5.0
                {
                    too_small = undersize_ratio;
                    i_too_small = k1 as isize;
                }
            } else if too_small < undersize_ratio && model.dq[k0] < dq_max / 5.0
            {
                too_small = undersize_ratio;
                i_too_small = k1 as isize;
            }
        } else if too_small < undersize_ratio && model.dq[k0] < dq_max / 5.0 {
            too_small = undersize_ratio;
            i_too_small = k1 as isize;
        }
    }

    Ok((too_big, too_small, i_too_big, i_too_small))
}

pub(crate) fn emergency_merge_1based(model: &StarModel) -> usize {
    if model.nz < 2 {
        return 0;
    }
    let k0 = fortran_minloc_0based(&model.dq[..model.nz]);
    if model.dq[k0] < model.controls.mix_wind.split_merge_amr_dq_min {
        k0 + 1
    } else {
        0
    }
}

pub(crate) fn emergency_split_1based(model: &StarModel) -> usize {
    if model.nz == 0 {
        return 0;
    }
    let k0 = fortran_maxloc_0based(&model.dq[..model.nz]);
    if model.dq[k0] > model.controls.mix_wind.split_merge_amr_dq_max {
        k0 + 1
    } else {
        0
    }
}

fn do_merge(
    model: &mut StarModel,
    i_merge_1based: usize,
    has_undergone: &mut Vec<bool>,
) -> Result<(), StarError> {
    let nz = model.nz;
    if nz < 2 {
        return Ok(());
    }
    let species = model.species;
    let star_pe0 = get_star_pe(model);
    let mut i1 = i_merge_1based;
    model.aux.num_hydro_merges += 1;

    if i1 > 1 && i1 < nz {
        let qi_max =
            fortran_maxloc_0based(&model.xa[(i1 - 1) * species..i1 * species]);
        let qim_max = fortran_maxloc_0based(
            &model.xa[(i1 - 2) * species..(i1 - 1) * species],
        );
        if qi_max == qim_max {
            let dr_l = if i1 + 1 == nz {
                model.r[nz - 1] - model.r_center
            } else {
                model.r[i1 - 1] - model.r[i1]
            };
            let dr_r = model.r[i1 - 2] - model.r[i1 - 1];
            if dr_r < dr_l {
                i1 -= 1;
            }
        }
    }

    if i1 == nz {
        i1 -= 1;
    }
    let ip1 = i1 + 1;
    let i0 = i1 - 1;
    let ip0 = ip1 - 1;

    if model
        .controls
        .mix_wind
        .split_merge_amr_avoid_repeated_remesh
        && (has_undergone[i0] || has_undergone[ip0])
    {
        has_undergone[i0] = true;
        has_undergone[ip0] = true;
        return Ok(());
    }

    let (_esum_i, ke_i, _pe_i, ie_i, etrb_i) = get_cell_energies(model, i0)?;
    let (_esum_ip, ke_ip, _pe_ip, ie_ip, etrb_ip) =
        get_cell_energies(model, ip0)?;

    let mut j_rot_new = 0.0_f64;
    let mut j_rot_p1_new = 0.0_f64;
    if model.controls.rotation_flag {
        let dmbar_old = if i1 == 1 {
            0.5 * model.dm[i0]
        } else {
            0.5 * (model.dm[i0] + model.dm[i0 - 1])
        };
        if ip1 == nz {
            let dmbar_p1_old = model.dm[nz - 1] + 0.5 * model.dm[nz - 2];
            let dmbar_new = dmbar_old + dmbar_p1_old;
            j_rot_new = (dmbar_old * model.j_rot[i0]
                + dmbar_p1_old * model.j_rot[ip0])
                / dmbar_new;
        } else {
            let dmbar_p1_old = 0.5 * (model.dm[ip0] + model.dm[ip0 - 1]);
            let dmbar_p2_old = if ip1 + 1 == nz {
                model.dm[nz - 1] + 0.5 * model.dm[nz - 2]
            } else {
                0.5 * (model.dm[ip0 + 1] + model.dm[ip0])
            };
            let dmbar_new = dmbar_old + dmbar_p1_old - 0.5 * model.dm[i0];
            let dmbar_p1_new = dmbar_p2_old + 0.5 * model.dm[i0];
            let j_old_pair =
                dmbar_old * model.j_rot[i0] + dmbar_p1_old * model.j_rot[ip0];
            j_rot_new = j_old_pair / (dmbar_old + dmbar_p1_old);
            let j_old_triple = j_old_pair + dmbar_p2_old * model.j_rot[ip0 + 1];
            j_rot_p1_new =
                (j_old_triple - j_rot_new * dmbar_new) / dmbar_p1_new;
        }
    }

    let dm_i = model.dm[i0];
    let dm_ip = model.dm[ip0];
    let dm = dm_i + dm_ip;
    model.dm[i0] = dm;
    model.dq[i0] = dm / model.xmstar;

    if model.controls.rti_flag {
        model.aux.alpha_rti[i0] = (dm_i * model.aux.alpha_rti[i0]
            + dm_ip * model.aux.alpha_rti[ip0])
            / dm;
    }
    for q in 0..species {
        model.xa[i0 * species + q] = (dm_i * model.xa[i0 * species + q]
            + dm_ip * model.xa[ip0 * species + q])
            / dm;
    }

    let ke = ke_i + ke_ip;
    if model.controls.u_flag {
        let mut v = (ke / (0.5 * dm)).max(0.0).sqrt();
        if model.u[i0] < 0.0 {
            v = -v;
        }
        model.u[i0] = v;
    }

    model.energy[i0] = (ie_i + ie_ip) / dm;
    if model.controls.rsp2_flag {
        let et = (etrb_i + etrb_ip) / dm;
        model.aux.rsp.et_rsp[i0] = et;
        model.aux.etrb[i0] = et;
    }

    if model.controls.rotation_flag {
        model.j_rot[i0] = j_rot_new;
        if ip1 != nz {
            model.j_rot[ip0 + 1] = j_rot_p1_new;
        }
    }

    // Shift `ip0 + 1 .. nz` down to `ip0 .. nz - 1` (`adjust_mesh_split_merge.f90:629-667`).
    model
        .xa
        .copy_within(((ip0 + 1) * species)..(nz * species), ip0 * species);
    for vec in [
        &mut model.r,
        &mut model.ln_r,
        &mut model.rmid,
        &mut model.dm,
        &mut model.m,
        &mut model.m_grav,
        &mut model.dq,
        &mut model.q,
        &mut model.lum,
        &mut model.u,
        &mut model.v,
        &mut model.energy,
        &mut model.lnd,
        &mut model.rho,
        &mut model.ln_t,
        &mut model.t,
        &mut model.cgrav,
        &mut model.d_mix,
        &mut model.mlt_vc,
        &mut model.csound,
        &mut model.tau,
        &mut model.opacity,
        &mut model.j_rot,
        &mut model.omega,
        &mut model.aux.dpdr_drhodr_info,
        &mut model.aux.alpha_rti,
        &mut model.aux.i_rot,
        &mut model.aux.fp_rot,
        &mut model.aux.ft_rot,
        // `star/private/adjust_mesh_split_merge.f90:656` shifts `alpha_mlt`.
        &mut model.aux.mixing.alpha_mlt,
    ] {
        if vec.len() >= nz {
            vec.copy_within((ip0 + 1)..nz, ip0);
        }
    }
    has_undergone.copy_within((ip0 + 1)..nz, ip0);
    has_undergone.truncate(nz - 1);
    has_undergone[i0] = true;

    model.resize_zones(nz - 1);
    update_xh_eos(model, i0)?;
    set_rmid_cell(model, i0);

    let star_pe1 = get_star_pe(model);
    revise_star_radius(model, star_pe0, star_pe1);
    Ok(())
}

fn do_split(
    model: &mut StarModel,
    i_split_1based: usize,
    tau_center: f64,
    has_undergone: &mut Vec<bool>,
) -> Result<(), StarError> {
    let nz_old = model.nz;
    if nz_old < 3 {
        return Err(StarError::InvalidInput(String::from(
            "remesh_split_merge::do_split requires at least 3 zones",
        )));
    }

    let species = model.species;
    let star_pe0 = get_star_pe(model);
    model.aux.num_hydro_splits += 1;

    let i1 = i_split_1based;
    let ip1 = i1 + 1;
    let i0 = i1 - 1;
    let ip0 = ip1 - 1;

    let _ = get_cell_energies(model, i0)?;

    let mut j_old = 0.0_f64;
    if model.controls.rotation_flag && i1 < nz_old {
        let dmbar_old = if i1 == 1 {
            0.5 * model.dm[i0]
        } else {
            0.5 * (model.dm[i0] + model.dm[i0 - 1])
        };
        let dmbar_p1_old = if ip1 == nz_old {
            model.dm[ip0] + 0.5 * model.dm[ip0 - 1]
        } else {
            0.5 * (model.dm[ip0] + model.dm[ip0 - 1])
        };
        j_old = dmbar_old * model.j_rot[i0] + dmbar_p1_old * model.j_rot[ip0];
    }

    let r_r = model.r[i0];
    let mlt_vc_r = model.mlt_vc[i0];
    let (r_l, mlt_vc_l, tau_l) = if i1 == nz_old {
        (model.r_center, 0.0, tau_center)
    } else {
        (model.r[ip0], model.mlt_vc[ip0], model.tau[ip0])
    };
    let tau_r = model.tau[i0];
    if !(tau_l + tau_r).is_finite() {
        return Err(StarError::InvalidInput(String::from(
            "remesh_split_merge::do_split non-finite optical depth",
        )));
    }

    let dr = r_r - r_l;
    let r_c = 0.5 * (r_r + r_l);
    let dv = FOUR_THIRDS_PI * (r_r * r_r * r_r - r_l * r_l * r_l);
    let dvr = FOUR_THIRDS_PI * (r_r * r_r * r_r - r_c * r_c * r_c);
    let dvl = dv - dvr;

    let dm = model.dm[i0];
    let rho = dm / dv;
    let (v, v2) = if model.controls.u_flag {
        (model.u[i0], model.u[i0] * model.u[i0])
    } else {
        (0.0, 0.0)
    };
    let energy = model.energy[i0];

    let (ir0, ic0, il0) = if i1 > 1 && i1 < nz_old {
        (i0 - 1, i0, i0 + 1)
    } else if i1 == 1 {
        (0, 1, 2)
    } else {
        (nz_old - 3, nz_old - 2, nz_old - 1)
    };

    let energy_r_st = model.energy[ir0];
    let rho_r_st = model.dm[ir0] / get_dv(model, ir0);
    let energy_c_st = model.energy[ic0];
    let rho_c_st = model.dm[ic0] / get_dv(model, ic0);
    let energy_l_st = model.energy[il0];
    let rho_l_st = model.dm[il0] / get_dv(model, il0);

    let d_left = if il0 + 1 == nz_old {
        0.5 * (model.r[ic0] - model.r_center)
    } else {
        0.5 * (model.r[ic0] - model.r[il0 + 1])
    };
    let d_rght = 0.5 * (model.r[ir0] - model.r[il0]);
    let d_cntr = d_left + d_rght;

    let grad_rho = if model.controls.mix_wind.equal_split_density_amr {
        0.0
    } else {
        get1_grad(rho_l_st, rho_c_st, rho_r_st, d_left, d_cntr, d_rght)
    };
    let grad_energy = get1_grad(
        energy_l_st,
        energy_c_st,
        energy_r_st,
        d_left,
        d_cntr,
        d_rght,
    );
    let grad_alpha = if model.controls.rti_flag {
        get1_grad(
            model.aux.alpha_rti[il0],
            model.aux.alpha_rti[ic0],
            model.aux.alpha_rti[ir0],
            d_left,
            d_cntr,
            d_rght,
        )
    } else {
        0.0
    };

    let mut grad_v2 = 0.0;
    let mut v2_l_face = 0.0;
    let mut v2_r_face = 0.0;
    if model.controls.u_flag {
        let vr = model.u[ir0];
        let vc = model.u[ic0];
        let vl = model.u[il0];
        if (vl - vc) * (vc - vr) > 0.0 {
            grad_v2 =
                get1_grad(vl * vl, vc * vc, vr * vr, d_left, d_cntr, d_rght);
        }
    } else if model.controls.v_flag {
        let vl = if il0 + 1 == nz_old {
            model.v_center
        } else {
            model.v[ip0]
        };
        v2_l_face = vl * vl;
        v2_r_face = model.v[i0] * model.v[i0];
    }

    let mut grad_xa = vec![0.0; species];
    for q in 0..species {
        grad_xa[q] = get1_grad(
            model.xa[il0 * species + q],
            model.xa[ic0 * species + q],
            model.xa[ir0 * species + q],
            d_left,
            d_cntr,
            d_rght,
        );
    }

    let nz = nz_old + 1;
    model.resize_zones(nz);
    has_undergone.resize(nz, false);

    if i1 < nz_old {
        model
            .xa
            .copy_within((i0 * species)..(nz_old * species), ip0 * species);
        for vec in [
            &mut model.r,
            &mut model.ln_r,
            &mut model.rmid,
            &mut model.dm,
            &mut model.m,
            &mut model.m_grav,
            &mut model.dq,
            &mut model.q,
            &mut model.lum,
            &mut model.u,
            &mut model.v,
            &mut model.energy,
            &mut model.lnd,
            &mut model.rho,
            &mut model.ln_t,
            &mut model.t,
            &mut model.cgrav,
            &mut model.d_mix,
            &mut model.mlt_vc,
            &mut model.csound,
            &mut model.tau,
            &mut model.opacity,
            &mut model.j_rot,
            &mut model.omega,
            &mut model.aux.dpdr_drhodr_info,
            &mut model.aux.alpha_rti,
            &mut model.aux.i_rot,
            &mut model.aux.fp_rot,
            &mut model.aux.ft_rot,
            // `star/private/adjust_mesh_split_merge.f90:1037` shifts `alpha_mlt`.
            &mut model.aux.mixing.alpha_mlt,
        ] {
            vec.copy_within(i0..nz_old, ip0);
        }
        has_undergone.copy_within(i0..nz_old, ip0);
    }
    has_undergone[i0] = true;
    has_undergone[ip0] = true;

    let dm_parent = rho * dv;
    let mut rho_l = rho - (grad_rho * dr) / 4.0;
    let mut rho_r = rho + (grad_rho * dr) / 4.0;
    let (mut dm_l, mut dm_r) =
        if grad_rho == 0.0 || rho_l <= 0.0 || rho_r <= 0.0 {
            rho_l = rho;
            let dml = rho_l * dvl;
            (dml, dm_parent - dml)
        } else {
            let dml0 = rho_l * dvl;
            let dmr0 = rho_r * dvr;
            let f = dm_parent / (dml0 + dmr0);
            rho_l *= f;
            rho_r *= f;
            let dml = rho_l * dvl;
            let dmr = rho_r * dvr;
            if (dml + dmr - dm_parent).abs() > 1.0e-14 * dm_parent {
                return Err(StarError::InvalidInput(format!(
                "do_split mass mismatch at cell {i1}: (dML + dMR - dM)/dM = {}",
                (dml + dmr - dm_parent) / dm_parent
            )));
            }
            (dml, dm_parent - dml)
        };

    let min_dm = model.controls.mix_wind.split_merge_amr_dq_min * model.xmstar;
    if dm_l < min_dm || dm_r < min_dm {
        let dml = rho * dvl;
        dm_l = dml;
        dm_r = dm_parent - dml;
    }

    let mut energy_r = energy + (grad_energy * dr) / 4.0;
    let mut energy_l = (dm_parent * energy - dm_r * energy_r) / dm_l;
    if energy_r < 0.0 || energy_l < 0.0 {
        energy_r = energy;
        energy_l = energy;
    }
    model.energy[i0] = energy_r;
    model.energy[ip0] = energy_l;

    if model.controls.u_flag {
        let mut v2_r = v2 + (grad_v2 * dr) / 4.0;
        let mut v2_l = (dm_parent * v2 - dm_r * v2_r) / dm_l;
        if v2_r < 0.0 || v2_l < 0.0 {
            v2_r = v2;
            v2_l = v2;
        }
        model.u[i0] = v2_r.sqrt();
        model.u[ip0] = v2_l.sqrt();
        if v < 0.0 {
            model.u[i0] = -model.u[i0];
            model.u[ip0] = -model.u[ip0];
        }
    } else if model.controls.v_flag {
        model.v[ip0] = (0.5 * (v2_l_face + v2_r_face)).sqrt();
    }

    if model.controls.rti_flag {
        if i1 == 1 {
            model.aux.alpha_rti[ip0] = model.aux.alpha_rti[i0];
        } else {
            let (new_al, new_ar) = split1_non_negative(
                model.aux.alpha_rti[i0],
                grad_alpha,
                dr,
                dv,
                dvr,
                dvl,
            );
            model.aux.alpha_rti[i0] = new_ar;
            model.aux.alpha_rti[ip0] = new_al;
        }
        model.aux.dpdr_drhodr_info[ip0] = model.aux.dpdr_drhodr_info[i0];
    }

    model.mlt_vc[ip0] = mlt_vc_r + ((mlt_vc_l - mlt_vc_r) * dm_r) / dm_parent;
    model.tau[ip0] = tau_r + ((tau_l - tau_r) * dm_r) / dm_parent;

    if i1 == 1 {
        let src = model.xa[i0 * species..(i0 + 1) * species].to_vec();
        model.xa[ip0 * species..(ip0 + 1) * species].copy_from_slice(&src);
    } else {
        let sum_g: f64 = grad_xa.iter().sum();
        let j_piv = if sum_g > 0.0 {
            fortran_maxloc_0based(&grad_xa)
        } else {
            fortran_minloc_0based(&grad_xa)
        };
        grad_xa[j_piv] = 0.0;
        let rem_g: f64 = grad_xa.iter().sum();
        grad_xa[j_piv] = -rem_g;

        for q in 0..species {
            let (x_l, x_r) = split1_non_negative(
                model.xa[i0 * species + q],
                grad_xa[q],
                dr,
                dv,
                dvr,
                dvl,
            );
            model.xa[i0 * species + q] = x_r.clamp(0.0, 1.0);
            model.xa[ip0 * species + q] = x_l.clamp(0.0, 1.0);
        }
        let sum_i: f64 =
            model.xa[i0 * species..(i0 + 1) * species].iter().sum();
        let sum_ip: f64 =
            model.xa[ip0 * species..(ip0 + 1) * species].iter().sum();
        for q in 0..species {
            model.xa[i0 * species + q] /= sum_i;
            model.xa[ip0 * species + q] /= sum_ip;
        }
    }

    if model.controls.u_flag {
        model.aux.u_face_ad[ip0].val = 0.5 * (model.u[i0] + model.u[ip0]);
    }

    let dm_r_final = model.dm[i0] - dm_l;
    model.dm[i0] = dm_r_final;
    model.dq[i0] = model.dm[i0] / model.xmstar;

    model.r[ip0] = r_c;
    model.ln_r[ip0] = flexmesa_math::log(r_c);
    model.m[ip0] = model.m[i0] - model.dm[i0];
    model.q[ip0] = model.q[i0] - model.dq[i0];
    model.dm[ip0] = dm_l;
    model.dq[ip0] = model.dm[ip0] / model.xmstar;

    model.cgrav[ip0] = model.cgrav[i0];
    // Upstream `star/private/adjust_mesh_split_merge.f90:1239`.
    model.aux.mixing.alpha_mlt[ip0] = model.aux.mixing.alpha_mlt[i0];
    model.ln_t[ip0] = model.ln_t[i0];
    model.t[ip0] = model.t[i0];
    model.d_mix[ip0] = model.d_mix[i0];
    model.csound[ip0] = model.csound[i0];
    model.opacity[ip0] = model.opacity[i0];
    if model.controls.rti_flag {
        // Upstream `star/private/adjust_mesh_split_merge.f90:1245` overwrites
        // `alpha_RTI(ip) = alpha_RTI(i)` after `split1_non_negative` (`:1172-1173`).
        model.aux.alpha_rti[ip0] = model.aux.alpha_rti[i0];
    }

    if model.controls.rotation_flag {
        let j_rot_new = if i1 == nz_old {
            model.j_rot[i0]
        } else {
            let dmbar_new = if i1 == 1 {
                0.5 * model.dm[i0]
            } else {
                0.5 * (model.dm[i0] + model.dm[i0 - 1])
            };
            let dmbar_p1_new = if ip1 == nz {
                model.dm[ip0] + 0.5 * model.dm[ip0 - 1]
            } else {
                0.5 * (model.dm[ip0] + model.dm[ip0 - 1])
            };
            let dmbar_p2_new = if ip1 + 1 == nz {
                model.dm[ip0 + 1] + 0.5 * model.dm[ip0]
            } else {
                0.5 * (model.dm[ip0 + 1] + model.dm[ip0])
            };
            (j_old
                - dmbar_new * model.j_rot[i0]
                - dmbar_p2_new * model.j_rot[ip0 + 1])
                / dmbar_p1_new
        };
        model.j_rot[ip0] = j_rot_new;
    }

    model.lum[ip0] = if ip1 < nz_old {
        0.5 * (model.lum[i0] + model.lum[ip0 + 1])
    } else {
        0.5 * (model.lum[i0] + model.l_center)
    };

    update_xh_eos(model, i0)?;
    update_xh_eos(model, ip0)?;
    set_rmid_cell(model, i0);
    set_rmid_cell(model, ip0);

    let star_pe1 = get_star_pe(model);
    revise_star_radius(model, star_pe0, star_pe1);
    Ok(())
}

/// Checks whether `remesh_split_merge` would perform at least one split or merge
/// on `model` without mutating it.
///
/// # Errors
/// Returns [`StarError`] if `RSP2_flag` or `split_merge_amr_flipped_hybrid_zoning`
/// (`#242`) is enabled.
pub fn preview_split_merge_amr(
    model: &StarModel,
) -> Result<(usize, usize), StarError> {
    if model.controls.rsp2_flag {
        return Err(StarError::InvalidInput(String::from(
            "remesh_split_merge does not support RSP2_flag",
        )));
    }
    if !model.controls.okay_to_remesh || model.nz < 3 {
        return Ok((0, 0));
    }
    let mw = &model.controls.mix_wind;
    let k_c = model.nz - 1;
    let tau_center = model.tau[k_c]
        + (model.dm[k_c] * model.opacity[k_c])
            / (FOUR_PI * model.rmid[k_c] * model.rmid[k_c]);
    let has_undergone = vec![false; model.nz];
    let (too_big, too_small, i_too_big, i_too_small) =
        biggest_smallest(model, tau_center, &has_undergone)?;
    let will_split = (mw.split_merge_amr_max_iters > 0
        && i_too_big > 0
        && too_big > mw.split_merge_amr_maxlong)
        || emergency_split_1based(model) > 0;
    let will_merge = (mw.split_merge_amr_max_iters > 0
        && i_too_small > 0
        && too_small > mw.split_merge_amr_maxshort)
        || emergency_merge_1based(model) > 0;
    Ok((usize::from(will_split), usize::from(will_merge)))
}

/// Executes hydrodynamic split-merge AMR (`remesh_split_merge` in
/// `star/private/adjust_mesh_split_merge.f90:36-212`), returning
/// `(num_splits, num_merges)`.
///
/// # Errors
/// Returns [`StarError`] if `RSP2_flag` is set, `split_merge_amr_flipped_hybrid_zoning`
/// is enabled (`#242`), EOS inversion fails, or angular momentum conservation
/// deviates by more than `1e-14`.
pub fn remesh_split_merge(
    model: &mut StarModel,
) -> Result<(usize, usize), StarError> {
    remesh_split_merge_with(model, true)
}

/// [`remesh_split_merge`] with the ending of
/// [`crate::mesh_remap::execute_adjust_mesh_from_plan_with`]: with
/// `finish = false` (the step loop) a changed mesh only sets
/// `need_to_setvars = .true.` (`star/private/adjust_mesh_split_merge.f90:505,
/// 840`) and `prepare_for_new_step` sets the variables
/// (`evolve.f90:1891`).
///
/// # Errors
/// As [`remesh_split_merge`].
pub(crate) fn remesh_split_merge_with(
    model: &mut StarModel,
    finish: bool,
) -> Result<(usize, usize), StarError> {
    if model.controls.rsp2_flag {
        return Err(StarError::InvalidInput(String::from(
            "remesh_split_merge does not support RSP2_flag",
        )));
    }
    if !model.controls.okay_to_remesh {
        return Ok((0, 0));
    }
    if model
        .controls
        .mix_wind
        .split_merge_amr_flipped_hybrid_zoning
    {
        return Err(StarError::InvalidInput(String::from(
            "split_merge_amr_flipped_hybrid_zoning = .true. reads uninitialized target_dr_core in MESA 26.4.1 (deferred upstream bug #242)",
        )));
    }
    if model.nz < 3 {
        return Ok((0, 0));
    }

    if model.controls.rotation_flag
        && model.j_rot.iter().all(|&j| j == 0.0)
        && model.omega.iter().any(|&w| w != 0.0)
    {
        crate::rotation::update_rotation_distortion_factors(model);
    }
    let old_j = total_angular_momentum(model);
    let (age_old, dt_old, mstar_old) =
        (model.star_age_old, model.dt_old, model.mstar_old);
    let prev_okay_mlt_vc = model.aux.okay_to_set_mlt_vc;

    let mut has_undergone = vec![false; model.nz];
    let mut num_split = 0_usize;
    let mut num_merge = 0_usize;
    let max_too_small = model.controls.mix_wind.split_merge_amr_maxshort;
    let max_too_big = model.controls.mix_wind.split_merge_amr_maxlong;
    let max_iters = model.controls.mix_wind.split_merge_amr_max_iters;

    let k_c = model.nz - 1;
    let tau_center = model.tau[k_c]
        + (model.dm[k_c] * model.opacity[k_c])
            / (FOUR_PI * model.rmid[k_c] * model.rmid[k_c]);

    for iter in 1..=max_iters {
        let (too_big, too_small, i_too_big, i_too_small) =
            biggest_smallest(model, tau_center, &has_undergone)?;
        if iter % 2 == 0 {
            if i_too_small > 0 && too_small > max_too_small {
                do_merge(model, i_too_small as usize, &mut has_undergone)?;
                num_merge += 1;
            } else if i_too_big > 0 && too_big > max_too_big {
                do_split(
                    model,
                    i_too_big as usize,
                    tau_center,
                    &mut has_undergone,
                )?;
                num_split += 1;
            } else {
                break;
            }
        } else if i_too_big > 0 && too_big > max_too_big {
            do_split(
                model,
                i_too_big as usize,
                tau_center,
                &mut has_undergone,
            )?;
            num_split += 1;
        } else if i_too_small > 0 && too_small > max_too_small {
            do_merge(model, i_too_small as usize, &mut has_undergone)?;
            num_merge += 1;
        } else {
            break;
        }
    }

    for _ in 1..=100 {
        let i_small = emergency_merge_1based(model);
        if i_small == 0 {
            break;
        }
        if model
            .controls
            .mix_wind
            .split_merge_amr_avoid_repeated_remesh
            && has_undergone[i_small - 1]
        {
            break;
        }
        do_merge(model, i_small, &mut has_undergone)?;
        num_merge += 1;
    }

    for _ in 1..=100 {
        let i_big = emergency_split_1based(model);
        if i_big == 0 {
            break;
        }
        if model
            .controls
            .mix_wind
            .split_merge_amr_avoid_repeated_remesh
            && has_undergone[i_big - 1]
        {
            break;
        }
        do_split(model, i_big, tau_center, &mut has_undergone)?;
        num_split += 1;
    }

    model.aux.mesh_call_number += 1;
    let nz = model.nz;
    model.q[nz - 1] = model.dq[nz - 1];
    model.m[nz - 1] = model.dm[nz - 1] + model.m_center;
    model.set_xqs();
    model.set_dm_bar();
    model.set_m_grav_and_grav();

    if model.controls.rotation_flag {
        // Verbatim `amr` post-split/merge rotation update
        // (`star/private/adjust_mesh_split_merge.f90:165-174`).
        let max_w = model.controls.mix_wind.w_div_wcrit_max;
        let max_w2 = model.controls.mix_wind.w_div_wcrit_max2;
        for k in 0..nz {
            let r00 = flexmesa_math::exp(model.ln_r[k]);
            model.aux.w_div_wc[k] = crate::hydro_rotation::w_div_w_roche_jrot(
                r00,
                model.m[k],
                model.j_rot[k],
                model.cgrav[k],
                max_w,
                max_w2,
                false,
            );
            crate::hydro_rotation::update1_i_rot_from_xh(model, k);
            model.omega[k] = model.j_rot[k] / model.aux.i_rot[k];
        }
        let new_j = total_angular_momentum(model);
        if old_j.abs() > 0.0 && ((old_j - new_j) / old_j).abs() > 1.0e-14 {
            return Err(StarError::InvalidInput(format!(
                "angular momentum conservation error in remesh_split_merge: old_J={old_j}, new_J={new_j}"
            )));
        }
    }

    if num_split + num_merge > 0 && !finish {
        model.pack_xh();
        model.aux.need_to_setvars = true;
    } else if num_split + num_merge > 0 {
        model.pack_xh();
        model.aux.okay_to_set_mlt_vc = false;
        let hydro_res = set_hydro_vars(model, false, false);
        model.aux.okay_to_set_mlt_vc = prev_okay_mlt_vc;
        hydro_res?;
        model.save_start_values()?;
        model.save_old_values();
        model.star_age_old = age_old;
        model.dt_old = dt_old;
        model.mstar_old = mstar_old;
    }

    Ok((num_split, num_merge))
}
