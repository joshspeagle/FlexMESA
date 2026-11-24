// SPDX-License-Identifier: GPL-3.0-only
//! GYRE grid refinement (`grid/grid_refine_m.fypp:61-699`) for a single
//! `grid_spec_t(cx, omega)`.

use crate::context::Context;
use crate::grid::{Grid, Point};
use crate::model::idx;
use crate::par::{GridPar, OscPar};
use crate::GyreError;

fn sq(x: f64) -> f64 {
    x * x
}

/// `refine_grid(gr, cx, omega, gr_p, os_p)` (`grid_refine_m.fypp:61-161`).
///
/// # Errors
///
/// Model and context errors; [`GyreError::Unported`] for thermal and
/// structural refinement weights.
pub fn refine_grid(
    gr: &mut Grid,
    cx: &Context<'_>,
    omega: &[f64],
    gr_p: &GridPar,
    os_p: &OscPar,
) -> Result<(), GyreError> {
    // add_points_ (grid_refine_m.fypp:165-289)
    let mut n = gr.n();
    let mut split: Vec<bool> =
        (0..n - 1).map(|j| gr.pt[j + 1].s == gr.pt[j].s).collect();
    for _i_iter in 1..=gr_p.n_iter_max {
        for j in 0..n - 1 {
            if !split[j] {
                continue;
            }
            let pt_a = gr.pt[j];
            let pt_b = gr.pt[j + 1];
            let dx = pt_b.x - pt_a.x;
            if dx > gr_p.dx_min {
                split[j] = if pt_a.x != 0.0 {
                    split_mech(cx, pt_a, pt_b, omega, gr_p, os_p)?
                        || split_therm(gr_p)?
                        || split_struct(gr_p)?
                        || dx > gr_p.dx_max
                } else {
                    split_center(cx, pt_a, pt_b, omega, gr_p)?
                        || dx > gr_p.dx_max
                };
            } else {
                split[j] = false;
            }
        }
        let n_split = split.iter().filter(|&&b| b).count();
        if n_split == 0 {
            break;
        }
        *gr = Grid::split(gr, &split)?;
        let mut split_new = Vec::with_capacity(n + n_split - 1);
        for &sj in &split[..n - 1] {
            split_new.push(sj);
            if sj {
                split_new.push(sj);
            }
        }
        n += n_split;
        split = split_new;
    }
    Ok(())
}

/// `split_mech_` (`grid_refine_m.fypp:293-423`).
fn split_mech(
    cx: &Context<'_>,
    pt_a: Point,
    pt_b: Point,
    omega: &[f64],
    gr_p: &GridPar,
    os_p: &OscPar,
) -> Result<bool, GyreError> {
    if gr_p.w_osc == 0.0 && gr_p.w_exp == 0.0 {
        return Ok(false);
    }
    let pt = Point {
        s: pt_a.s,
        x: 0.5 * (pt_a.x + pt_b.x),
    };
    // math_m's LOG is CRMATH's correctly rounded log.
    let dlnx = flexmesa_math::log(pt_b.x) - flexmesa_math::log(pt_a.x);
    let ml = cx.ml;
    let v = ml.coeff(idx::V_2, pt)? * sq(pt.x);
    let a_s = ml.coeff(idx::AS, pt)?;
    let u = ml.coeff(idx::U, pt)?;
    let c_1 = ml.coeff(idx::C_1, pt)?;
    let gamma_1 = ml.coeff(idx::GAMMA_1, pt)?;
    let omega_rot = cx.omega_rot(pt)?;
    let omega_rot_i = cx.omega_rot(cx.pt_i)?;
    let twopi = cx.k.math.twopi;
    let mut split = false;
    for &w in omega {
        let omega_c = cx.omega_c_r(omega_rot, w);
        let lambda = cx.lambda_r(omega_rot, w);
        let l_i = cx.l_e_r(omega_rot_i, w);
        let c_4 = -4.0 * v / gamma_1 * c_1 * os_p.alpha_gam;
        let c_2 = sq(a_s - v / gamma_1 - u + 4.0)
            + 4.0 * v / gamma_1 * a_s * os_p.alpha_gam * os_p.alpha_pi
            + 4.0 * lambda;
        let c_0 = -4.0 * lambda * a_s / c_1 * os_p.alpha_pi;
        let psi2 = if c_0 != 0.0 {
            let oc2 = sq(omega_c);
            (c_4 * sq(oc2) + c_2 * oc2 + c_0) / oc2
        } else {
            c_4 * sq(omega_c) + c_2
        };
        let (chi_r, chi_i) = if psi2 < 0.0 {
            (
                0.5 * (a_s + v / gamma_1 - u + 2.0 - 2.0 * l_i),
                0.5 * (-psi2).sqrt(),
            )
        } else {
            let chi_r_p =
                0.5 * (a_s + v / gamma_1 - u + 2.0 - 2.0 * l_i + psi2.sqrt());
            let chi_r_m =
                0.5 * (a_s + v / gamma_1 - u + 2.0 - 2.0 * l_i - psi2.sqrt());
            (
                if chi_r_p.abs() > chi_r_m.abs() {
                    chi_r_p
                } else {
                    chi_r_m
                },
                0.0,
            )
        };
        split = split
            || dlnx * gr_p.w_exp * chi_r.abs() > twopi
            || dlnx * gr_p.w_osc * chi_i.abs() > twopi;
        if split {
            break;
        }
    }
    Ok(split)
}

/// `split_therm_` (`grid_refine_m.fypp:427-508`).
fn split_therm(gr_p: &GridPar) -> Result<bool, GyreError> {
    if gr_p.w_thm == 0.0 {
        Ok(false)
    } else {
        Err(GyreError::Unported {
            what: "thermal grid refinement (w_thm /= 0)",
            at: "grid/grid_refine_m.fypp:458-502",
        })
    }
}

/// `split_struct_` (`grid_refine_m.fypp:512-574`).
fn split_struct(gr_p: &GridPar) -> Result<bool, GyreError> {
    if gr_p.w_str == 0.0 {
        Ok(false)
    } else {
        Err(GyreError::Unported {
            what: "structural grid refinement (w_str /= 0)",
            at: "grid/grid_refine_m.fypp:537-568",
        })
    }
}

/// `split_center_` (`grid_refine_m.fypp:578-699`).
fn split_center(
    cx: &Context<'_>,
    pt_a: Point,
    pt_b: Point,
    omega: &[f64],
    gr_p: &GridPar,
) -> Result<bool, GyreError> {
    if !gr_p.resolve_ctr || gr_p.w_ctr == 0.0 {
        return Ok(false);
    }
    let pt = Point {
        s: pt_a.s,
        x: 0.5 * (pt_a.x + pt_b.x),
    };
    let ml = cx.ml;
    let v = ml.coeff(idx::V_2, pt)? * sq(pt.x);
    let a_s = ml.coeff(idx::AS, pt)?;
    let u = ml.coeff(idx::U, pt)?;
    let c_1 = ml.coeff(idx::C_1, pt)?;
    let gamma_1 = ml.coeff(idx::GAMMA_1, pt)?;
    let omega_rot = cx.omega_rot(pt)?;
    let omega_rot_i = cx.omega_rot(cx.pt_i)?;
    let mut split = false;
    for &w in omega {
        let omega_c = cx.omega_c_r(omega_rot, w);
        let lambda = cx.lambda_r(omega_rot, w);
        let l_i = cx.l_e_r(omega_rot_i, w);
        let c_4 = -4.0 * v / gamma_1 * c_1;
        let c_2 = sq(a_s - v / gamma_1 - u + 4.0)
            + 4.0 * v / gamma_1 * a_s
            + 4.0 * lambda;
        let c_0 = -4.0 * lambda * a_s / c_1;
        let psi2 = if c_0 != 0.0 {
            let oc2 = sq(omega_c);
            (c_4 * sq(oc2) + c_2 * oc2 + c_0) / oc2
        } else {
            c_4 * sq(omega_c) + c_2
        };
        let (chi_r, chi_i) = if psi2 < 0.0 {
            (
                0.5 * (a_s + v / gamma_1 - u + 2.0 - 2.0 * l_i),
                0.5 * (-psi2).sqrt(),
            )
        } else {
            (
                0.5 * (a_s + v / gamma_1 - u + 2.0 - 2.0 * l_i + psi2.sqrt()),
                0.0,
            )
        };
        split = split || chi_i != 0.0 || gr_p.w_ctr * chi_r.abs() > 1.0;
        if split {
            break;
        }
    }
    Ok(split)
}
