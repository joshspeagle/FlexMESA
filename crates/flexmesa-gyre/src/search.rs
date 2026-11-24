// SPDX-License-Identifier: GPL-3.0-only
//! GYRE frequency scans (`search/scan_m.fypp`), discriminant evaluation
//! (`search/discrim_m.fypp`), Brent root solving in extended arithmetic
//! (`math/root_m.fypp`, `xr` instances) and the bracket search
//! (`search/bracket_search_m.fypp`).

use std::sync::atomic::{AtomicI32, Ordering};

use crate::context::Context;
use crate::ext_real::ExtR;
use crate::forum::unique_indices;
use crate::grid::Grid;
use crate::mode::{rad_wave_hom, Mode};
use crate::par::{ModePar, NumPar, ScanPar};
use crate::rad::{GyreTap, RadBvp};
use crate::GyreError;

/// `bracket_search_m`'s `integer, save :: id_m = 0`
/// (`search/bracket_search_m.fypp:47`, incremented at `:135`): one mode
/// counter for the whole process, as upstream.
static ID_M: AtomicI32 = AtomicI32::new(0);

/// `build_scan(cx, md_p, os_p, sc_p, axis, omega)` (`scan_m.fypp:52-103`).
///
/// # Errors
///
/// [`GyreError::Unported`] for file grids, `INVERSE` grids and single- or
/// zero-point scans; frequency-unit errors.
pub fn build_scan(
    cx: &Context<'_>,
    sc_p: &[ScanPar],
    axis: &str,
) -> Result<Vec<f64>, GyreError> {
    if sc_p.is_empty() {
        return Err(GyreError::Assert("empty scan_par_t"));
    }
    let mut omega: Vec<f64> = Vec::new();
    for s in sc_p {
        if axis != s.axis {
            continue;
        }
        match s.grid_type.as_str() {
            "FILE" => {
                return Err(GyreError::Unported {
                    what: "scan grid_type = 'FILE'",
                    at: "search/scan_m.fypp:81-82",
                })
            }
            _ => build_scan_grid(cx, s, &mut omega)?,
        }
    }
    if !omega.is_empty() {
        let idx = unique_indices(&omega, false);
        omega = idx
            .iter()
            .filter_map(|&i| usize::try_from(i - 1).ok().map(|k| omega[k]))
            .collect();
    }
    Ok(omega)
}

/// `build_scan_grid_` (`scan_m.fypp:107-216`).
fn build_scan_grid(
    cx: &Context<'_>,
    sc_p: &ScanPar,
    omega: &mut Vec<f64>,
) -> Result<(), GyreError> {
    let omega_i_min = sc_p.freq_min / cx.freq_scale(&sc_p.freq_min_units)?
        - cx.freq_shift(&sc_p.freq_frame)?;
    let omega_i_max = sc_p.freq_max / cx.freq_scale(&sc_p.freq_max_units)?
        - cx.freq_shift(&sc_p.freq_frame)?;
    if sc_p.n_freq < 1 {
        return Err(GyreError::Unported {
            what: "scans with n_freq < 1",
            at: "search/scan_m.fypp:134-139",
        });
    } else if sc_p.n_freq == 1 {
        return Err(GyreError::Unported {
            what: "scans with n_freq = 1",
            at: "search/scan_m.fypp:141-158",
        });
    }
    if omega_i_max > omega_i_min {
        let omega_g_min = omega_i_min + cx.freq_shift(&sc_p.grid_frame)?;
        let omega_g_max = omega_i_max + cx.freq_shift(&sc_p.grid_frame)?;
        let nf = sc_p.n_freq;
        for i in 1..=nf {
            let omega_g = match sc_p.grid_type.as_str() {
                "LINEAR" => {
                    (f64::from(nf - i) * omega_g_min
                        + f64::from(i - 1) * omega_g_max)
                        / f64::from(nf - 1)
                }
                "INVERSE" => {
                    return Err(GyreError::Unported {
                        what: "scan grid_type = 'INVERSE'",
                        at: "search/scan_m.fypp:180-181",
                    })
                }
                _ => return Err(GyreError::Abort("invalid grid_type")),
            };
            omega.push(omega_g - cx.freq_shift(&sc_p.grid_frame)?);
        }
    }
    // else: upstream only logs that the scan is empty.
    Ok(())
}

/// `check_scan(cx, gr, omega, md_p, os_p)` (`scan_m.fypp:285-334`).
///
/// # Errors
///
/// [`GyreError::Assert`] for a critical layer or a prograde/retrograde
/// transition.
pub fn check_scan(
    cx: &Context<'_>,
    gr: &Grid,
    omega: &[f64],
    md_p: &ModePar,
) -> Result<(), GyreError> {
    if omega.is_empty() {
        return Ok(());
    }
    let m = f64::from(md_p.m);
    let eval = |w: f64| -> Result<Vec<f64>, GyreError> {
        gr.pt
            .iter()
            .map(|&p| Ok(w - m * cx.omega_rot(p)?))
            .collect()
    };
    let mut omega_c = eval(omega[0])?;
    if !(omega_c.iter().all(|&o| o > 0.0) || omega_c.iter().all(|&o| o < 0.0)) {
        return Err(GyreError::Assert("critical layer encountered"));
    }
    for &w in &omega[1..] {
        let prev = omega_c;
        omega_c = eval(w)?;
        if !omega_c
            .iter()
            .zip(&prev)
            .all(|(a, b)| 1.0_f64.copysign(*a) == 1.0_f64.copysign(*b))
        {
            return Err(GyreError::Assert(
                "transition between prograde and retrograde",
            ));
        }
    }
    Ok(())
}

/// `eval_discrim(bp, st, discrim)` (`discrim_m.fypp:56-73`).
///
/// # Errors
///
/// BVP errors.
pub fn eval_discrim(
    bp: &mut RadBvp<'_>,
    omega: f64,
    tap: &mut dyn GyreTap,
) -> Result<ExtR, GyreError> {
    bp.build(omega, false, tap)?;
    bp.factor()?;
    let d = bp.det()?;
    tap.discrim(omega, d);
    Ok(d)
}

/// Status of a root solve (`common/status_m.fypp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stat {
    /// `STAT_OK`.
    Ok,
    /// `STAT_OMEGA_DOMAIN`.
    OmegaDomain,
    /// `STAT_ITER_MAX`.
    IterMax,
}

/// `narrow_brent_xr_` (`root_m.fypp:244-459`) as `solve_root_xr_` calls it
/// (`relative_tol` absent, both end values present). Returns
/// `(x_root, stat, n_iter)`.
fn solve_brent<F>(
    mut eval: F,
    x_a: ExtR,
    x_b: ExtR,
    x_tol: ExtR,
    n_iter_max: i32,
    f_x_a: ExtR,
    f_x_b: ExtR,
) -> Result<(ExtR, Stat, i32), GyreError>
where
    F: FnMut(ExtR) -> Result<(ExtR, Stat), GyreError>,
{
    let zero = ExtR::from_real(0.0);
    let r = ExtR::from_real;
    let mut a = x_a;
    let mut b = x_b;
    let mut f_a = f_x_a;
    let mut f_b = f_x_b;
    let mut c = b;
    let mut f_c = f_b;
    let mut d = zero;
    let mut e = zero;
    if !((f_a.ge(zero) && f_b.le(zero)) || (f_a.le(zero) && f_b.ge(zero))) {
        return Err(GyreError::Assert("root is not bracketed"));
    }
    let mut i_iter = 0;
    let mut stat = Stat::Ok;
    loop {
        i_iter += 1;
        if i_iter > n_iter_max {
            stat = Stat::IterMax;
            break;
        }
        if f_b.gt(zero) == f_c.gt(zero) {
            c = a;
            f_c = f_a;
            d = b - a;
            e = d;
        }
        if f_c.abs().lt(f_b.abs()) {
            a = b;
            b = c;
            c = a;
            f_a = f_b;
            f_b = f_c;
            f_c = f_a;
        }
        let tol = r(2.0 * f64::EPSILON) * b.abs() + x_tol;
        let m = r(0.5) * (c - b);
        if m.abs().le(tol) || f_b.ext_eq(zero) {
            break;
        }
        if e.abs().lt(tol) || f_a.abs().lt(f_b.abs()) {
            d = m;
            e = d;
        } else {
            let mut s = f_b / f_a;
            let (mut p, mut q);
            if a.ext_eq(c) {
                p = r(2.0) * m * s;
                q = r(1.0) - s;
            } else {
                q = f_a / f_c;
                let rr = f_b / f_c;
                p = s * (r(2.0) * m * q * (q - rr) - (b - a) * (rr - r(1.0)));
                q = (q - r(1.0)) * (rr - r(1.0)) * (s - r(1.0));
            }
            if p.gt(zero) {
                q = -q;
            } else {
                p = -p;
            }
            s = e;
            e = d;
            if (r(2.0) * p).lt(r(3.0) * m * q - (tol * q).abs())
                && p.lt((r(0.5) * s * q).abs())
            {
                d = p / q;
            } else {
                d = m;
                e = d;
            }
        }
        a = b;
        f_a = f_b;
        if d.abs().gt(tol) {
            b = b + d;
        } else if m.gt(zero) {
            b = b + tol;
        } else {
            b = b - tol;
        }
        let (fb, st) = eval(b)?;
        f_b = fb;
        stat = st;
        if stat != Stat::Ok {
            break;
        }
    }
    Ok((b, stat, i_iter))
}

/// `bracket_search(bp, omega, omega_min, omega_max, nm_p, process_mode)`
/// (`bracket_search_m.fypp:59-318`) for the radial BVP. Mode identifiers
/// come from the process-wide counter `id_m`.
///
/// # Errors
///
/// BVP and mode-construction errors; a failed root solve is reported and
/// skipped as upstream does.
#[allow(clippy::too_many_arguments)]
pub fn bracket_search<'a>(
    bp: &mut RadBvp<'a>,
    omega: &[f64],
    omega_min: f64,
    omega_max: f64,
    nm_p: &NumPar,
    tap: &mut dyn GyreTap,
    process_mode: &mut dyn FnMut(&Mode<'a>, i32, ExtR) -> Result<(), GyreError>,
) -> Result<(), GyreError> {
    // find_brackets_
    let mut discrim = Vec::with_capacity(omega.len());
    for &w in omega {
        let d = eval_discrim(bp, w, tap)?;
        // discrim = scale(ext_rt(FRACTION(d)), EXPONENT(d))
        discrim.push(ExtR::from_real(d.f).scale(d.e));
    }
    let zero = ExtR::from_real(0.0);
    let mut i_brack = Vec::new();
    for i in 0..omega.len().saturating_sub(1) {
        if (discrim[i] * discrim[i + 1]).le(zero) {
            i_brack.push(i);
        }
    }
    for &i in &i_brack {
        let x_a = ExtR::from_real(omega[i]);
        let x_b = ExtR::from_real(omega[i + 1]);
        let (x_root, stat, n_iter) = {
            let mut eval = |x: ExtR| -> Result<(ExtR, Stat), GyreError> {
                let w = x.to_real();
                if w >= omega_min && w <= omega_max {
                    Ok((eval_discrim(bp, w, tap)?, Stat::Ok))
                } else {
                    Ok((zero, Stat::OmegaDomain))
                }
            };
            match nm_p.r_root_solver.as_str() {
                "BRENT" => solve_brent(
                    &mut eval,
                    x_a,
                    x_b,
                    zero,
                    nm_p.n_iter_max,
                    discrim[i],
                    discrim[i + 1],
                )?,
                _ => return Err(GyreError::Abort("invalid r_root_solver")),
            }
        };
        if stat != Stat::Ok {
            // report_stat_(stat, 'solve'); cycle mode_loop
            continue;
        }
        // id_m = id_m + 1
        let id = ID_M.fetch_add(1, Ordering::Relaxed) + 1;
        let wv = rad_wave_hom(bp, x_root.to_real(), id, tap)?;
        let md = Mode::new(wv)?;
        let chi = md.wave.discrim.abs()
            / ExtR::max(discrim[i].abs(), discrim[i + 1].abs());
        process_mode(&md, n_iter, chi)?;
    }
    Ok(())
}
