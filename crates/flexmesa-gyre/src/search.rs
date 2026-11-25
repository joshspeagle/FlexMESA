// SPDX-License-Identifier: GPL-3.0-only
//! GYRE frequency scans (`search/scan_m.fypp`), discriminant evaluation
//! (`search/discrim_m.fypp`), Brent root solving in extended arithmetic
//! (`math/root_m.fypp`, `xr` instances) and the bracket search
//! (`search/bracket_search_m.fypp`).

use std::sync::atomic::{AtomicI32, Ordering};

use crate::ad::{ad_wave_hom, AdBvp};
use crate::context::Context;
use crate::ext_real::ExtR;
use crate::forum::unique_indices;
use crate::grid::Grid;
use crate::mode::{rad_wave_hom, Mode, Wave};
use crate::model::idx;
use crate::par::{ModePar, NumPar, OscPar, ScanPar};
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
/// [`GyreError::Unported`] for file grids; frequency-unit errors.
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
        return Ok(());
    } else if sc_p.n_freq == 1 {
        if omega_i_max == omega_i_min {
            omega.push(omega_i_min);
        }
        return Ok(());
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
                    f64::from(nf - 1)
                        / (f64::from(nf - i) / omega_g_min
                            + f64::from(i - 1) / omega_g_max)
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

/// Real boundary-value problem supporting discriminant evaluation and
/// homogeneous eigensolution recovery (`bvp_rt` implementations: [`RadBvp`]
/// and [`AdBvp`]).
pub trait RealBvp<'a> {
    /// Assemble boundary and difference blocks for real `omega`.
    ///
    /// # Errors
    ///
    /// Component build errors.
    fn build(
        &mut self,
        omega: f64,
        mode_build: bool,
        tap: &mut dyn GyreTap,
    ) -> Result<(), GyreError>;

    /// Factorize the assembled system matrix.
    ///
    /// # Errors
    ///
    /// LAPACK factorization errors.
    fn factor(&mut self) -> Result<(), GyreError>;

    /// Determinant of the factorized system matrix.
    ///
    /// # Errors
    ///
    /// Unfactored matrix or LAPACK errors.
    fn det(&self) -> Result<ExtR, GyreError>;

    /// Solve for the homogeneous wave at eigenfrequency `omega`.
    ///
    /// # Errors
    ///
    /// BVP and wave construction errors.
    fn wave_hom(
        &mut self,
        omega: f64,
        id: i32,
        tap: &mut dyn GyreTap,
    ) -> Result<Wave<'a>, GyreError>;
}

impl<'a> RealBvp<'a> for RadBvp<'a> {
    fn build(
        &mut self,
        omega: f64,
        mode_build: bool,
        tap: &mut dyn GyreTap,
    ) -> Result<(), GyreError> {
        RadBvp::build(self, omega, mode_build, tap)
    }

    fn factor(&mut self) -> Result<(), GyreError> {
        RadBvp::factor(self)
    }

    fn det(&self) -> Result<ExtR, GyreError> {
        RadBvp::det(self)
    }

    fn wave_hom(
        &mut self,
        omega: f64,
        id: i32,
        tap: &mut dyn GyreTap,
    ) -> Result<Wave<'a>, GyreError> {
        rad_wave_hom(self, omega, id, tap)
    }
}

impl<'a> RealBvp<'a> for AdBvp<'a> {
    fn build(
        &mut self,
        omega: f64,
        mode_build: bool,
        tap: &mut dyn GyreTap,
    ) -> Result<(), GyreError> {
        AdBvp::build(self, omega, mode_build, tap)
    }

    fn factor(&mut self) -> Result<(), GyreError> {
        AdBvp::factor(self)
    }

    fn det(&self) -> Result<ExtR, GyreError> {
        AdBvp::det(self)
    }

    fn wave_hom(
        &mut self,
        omega: f64,
        id: i32,
        tap: &mut dyn GyreTap,
    ) -> Result<Wave<'a>, GyreError> {
        ad_wave_hom(self, omega, id, tap)
    }
}

/// `eval_discrim(bp, st, discrim)` (`discrim_m.fypp:56-73`).
///
/// # Errors
///
/// BVP errors.
pub fn eval_discrim<'a, B: RealBvp<'a>>(
    bp: &mut B,
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

trait BrentScalar: Copy {
    fn zero() -> Self;
    fn from_r(r: f64) -> Self;
    fn abs(self) -> Self;
    fn neg(self) -> Self;
    fn add(self, rhs: Self) -> Self;
    fn sub(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;
    fn div(self, rhs: Self) -> Self;
    fn eq(self, rhs: Self) -> bool;
    fn lt(self, rhs: Self) -> bool;
    fn gt(self, rhs: Self) -> bool;
    fn le(self, rhs: Self) -> bool;
    fn ge(self, rhs: Self) -> bool;
}

impl BrentScalar for ExtR {
    fn zero() -> Self {
        Self::from_real(0.0)
    }
    fn from_r(r: f64) -> Self {
        Self::from_real(r)
    }
    fn abs(self) -> Self {
        Self::abs(self)
    }
    fn neg(self) -> Self {
        -self
    }
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
    fn div(self, rhs: Self) -> Self {
        self / rhs
    }
    fn eq(self, rhs: Self) -> bool {
        self.ext_eq(rhs)
    }
    fn lt(self, rhs: Self) -> bool {
        Self::lt(self, rhs)
    }
    fn gt(self, rhs: Self) -> bool {
        Self::gt(self, rhs)
    }
    fn le(self, rhs: Self) -> bool {
        Self::le(self, rhs)
    }
    fn ge(self, rhs: Self) -> bool {
        Self::ge(self, rhs)
    }
}

impl BrentScalar for f64 {
    fn zero() -> Self {
        0.0
    }
    fn from_r(r: f64) -> Self {
        r
    }
    fn abs(self) -> Self {
        Self::abs(self)
    }
    fn neg(self) -> Self {
        -self
    }
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
    fn div(self, rhs: Self) -> Self {
        self / rhs
    }
    fn eq(self, rhs: Self) -> bool {
        self == rhs
    }
    fn lt(self, rhs: Self) -> bool {
        self < rhs
    }
    fn gt(self, rhs: Self) -> bool {
        self > rhs
    }
    fn le(self, rhs: Self) -> bool {
        self <= rhs
    }
    fn ge(self, rhs: Self) -> bool {
        self >= rhs
    }
}

/// `narrow_brent_` (`root_m.fypp:244-459`) with `relative_tol` absent.
/// Returns `(x_root, stat, n_iter)`.
fn solve_brent_core<T, F>(
    mut eval: F,
    x_a: T,
    x_b: T,
    x_tol: T,
    n_iter_max: Option<i32>,
    f_x_a: Option<T>,
    f_x_b: Option<T>,
) -> Result<(T, Stat, i32), GyreError>
where
    T: BrentScalar,
    F: FnMut(T) -> Result<(T, Stat), GyreError>,
{
    let zero = T::zero();
    let r = T::from_r;
    let mut a = x_a;
    let mut b = x_b;
    let mut f_a = if let Some(fa) = f_x_a {
        fa
    } else {
        let (fa, st) = eval(a)?;
        if st != Stat::Ok {
            return Ok((b, st, 0));
        }
        fa
    };
    let mut f_b = if let Some(fb) = f_x_b {
        fb
    } else {
        let (fb, st) = eval(b)?;
        if st != Stat::Ok {
            return Ok((b, st, 0));
        }
        fb
    };
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
        if let Some(lim) = n_iter_max {
            if i_iter > lim {
                stat = Stat::IterMax;
                break;
            }
        }
        if f_b.gt(zero) == f_c.gt(zero) {
            c = a;
            f_c = f_a;
            d = b.sub(a);
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
        let tol = r(2.0 * f64::EPSILON).mul(b.abs()).add(x_tol);
        let m = r(0.5).mul(c.sub(b));
        if m.abs().le(tol) || f_b.eq(zero) {
            break;
        }
        if e.abs().lt(tol) || f_a.abs().lt(f_b.abs()) {
            d = m;
            e = d;
        } else {
            let mut s = f_b.div(f_a);
            let (mut p, mut q);
            if a.eq(c) {
                p = r(2.0).mul(m).mul(s);
                q = r(1.0).sub(s);
            } else {
                q = f_a.div(f_c);
                let rr = f_b.div(f_c);
                p = s.mul(
                    r(2.0)
                        .mul(m)
                        .mul(q)
                        .mul(q.sub(rr))
                        .sub(b.sub(a).mul(rr.sub(r(1.0)))),
                );
                q = q.sub(r(1.0)).mul(rr.sub(r(1.0))).mul(s.sub(r(1.0)));
            }
            if p.gt(zero) {
                q = q.neg();
            } else {
                p = p.neg();
            }
            s = e;
            e = d;
            if r(2.0).mul(p).lt(r(3.0).mul(m).mul(q).sub(tol.mul(q).abs()))
                && p.lt(r(0.5).mul(s).mul(q).abs())
            {
                d = p.div(q);
            } else {
                d = m;
                e = d;
            }
        }
        a = b;
        f_a = f_b;
        if d.abs().gt(tol) {
            b = b.add(d);
        } else if m.gt(zero) {
            b = b.add(tol);
        } else {
            b = b.sub(tol);
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

/// `solve_root_r_` (`math/root_m.fypp:69-128`, `relative_tol` absent).
///
/// # Errors
///
/// Unbracketed interval, invalid `r_root_solver`, or callback errors.
pub fn solve_root_r<F>(
    eval: F,
    x_a: f64,
    x_b: f64,
    x_tol: f64,
    nm_p: &NumPar,
    n_iter_max: Option<i32>,
) -> Result<(f64, Stat, i32), GyreError>
where
    F: FnMut(f64) -> Result<(f64, Stat), GyreError>,
{
    match nm_p.r_root_solver.as_str() {
        "BRENT" => {
            solve_brent_core(eval, x_a, x_b, x_tol, n_iter_max, None, None)
        }
        _ => Err(GyreError::Abort("invalid r_root_solver")),
    }
}

/// Propagation discriminant `gamma_` (`grid/grid_util_m.fypp:163-225`).
fn gamma_turn(
    cx: &Context<'_>,
    pt: crate::grid::Point,
    omega: f64,
    alpha_gam: f64,
    alpha_pi: f64,
) -> Result<f64, GyreError> {
    let ml = cx.ml;
    if ml.is_vacuum(pt)? {
        return Ok(f64::MAX);
    }
    let v = ml.coeff(idx::V_2, pt)? * (pt.x * pt.x);
    let a_s = ml.coeff(idx::AS, pt)?;
    let u = ml.coeff(idx::U, pt)?;
    let c_1 = ml.coeff(idx::C_1, pt)?;
    let gamma_1 = ml.coeff(idx::GAMMA_1, pt)?;
    let omega_rot = cx.omega_rot(pt)?;
    let omega_c = cx.omega_c_r(omega_rot, omega);
    let lambda = cx.lambda_r(omega_rot, omega);
    let g_4 = -4.0 * v / gamma_1 * c_1 * alpha_gam;
    let d = a_s - v / gamma_1 - u + 4.0;
    let g_2 =
        d * d + 4.0 * v / gamma_1 * a_s * alpha_gam * alpha_pi + 4.0 * lambda;
    let g_0 = -4.0 * lambda * a_s / c_1 * alpha_pi;
    let oc2 = omega_c * omega_c;
    if g_0 != 0.0 {
        Ok((g_4 * (oc2 * oc2) + g_2 * oc2 + g_0) / oc2)
    } else {
        Ok(g_4 * oc2 + g_2)
    }
}

/// `find_turn(cx, gr, st, nm_p, os_p, j_turn, x_turn)`
/// (`grid/grid_util_m.fypp:53-159`), returning 1-based `(j_turn, x_turn)`.
///
/// # Errors
///
/// Model evaluation or root-solver errors.
pub fn find_turn(
    cx: &Context<'_>,
    gr: &Grid,
    omega: f64,
    nm_p: &NumPar,
    os_p: &OscPar,
) -> Result<(usize, f64), GyreError> {
    let mut j_turn = gr.n();
    let mut x_turn = f64::MAX;
    let mut gamma_b =
        gamma_turn(cx, cx.pt_i, omega, os_p.alpha_gam, os_p.alpha_pi)?;
    if gamma_b <= 0.0 {
        return Ok((1, cx.pt_i.x));
    }
    for j in 1..gr.n() {
        let gamma_a = gamma_b;
        gamma_b =
            gamma_turn(cx, gr.pt[j], omega, os_p.alpha_gam, os_p.alpha_pi)?;
        if gamma_a > 0.0 && gamma_b <= 0.0 {
            j_turn = j;
            let pt_a = gr.pt[j - 1];
            let pt_b = gr.pt[j];
            if pt_a.s == pt_b.s {
                if gamma_a.abs() < f64::EPSILON * gamma_b.abs() {
                    x_turn = pt_a.x;
                } else if gamma_b.abs() < f64::EPSILON * gamma_a.abs() {
                    x_turn = pt_b.x;
                } else {
                    let s = pt_a.s;
                    let (xr, _, _) = solve_root_r(
                        |x| {
                            let pt = crate::grid::Point { s, x };
                            let g = gamma_turn(
                                cx,
                                pt,
                                omega,
                                os_p.alpha_gam,
                                os_p.alpha_pi,
                            )?;
                            Ok((g, Stat::Ok))
                        },
                        pt_a.x,
                        pt_b.x,
                        0.0,
                        nm_p,
                        None,
                    )?;
                    x_turn = xr;
                }
            } else {
                x_turn = pt_a.x;
            }
            break;
        }
    }
    Ok((j_turn, x_turn))
}

/// `bracket_search(bp, omega, omega_min, omega_max, nm_p, process_mode)`
/// (`bracket_search_m.fypp:59-318`). Mode identifiers come from the
/// process-wide counter `id_m`.
///
/// # Errors
///
/// BVP and mode-construction errors; a failed root solve is reported and
/// skipped as upstream does.
#[allow(clippy::too_many_arguments)]
pub fn bracket_search<'a, B: RealBvp<'a>>(
    bp: &mut B,
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
                "BRENT" => solve_brent_core(
                    &mut eval,
                    x_a,
                    x_b,
                    zero,
                    Some(nm_p.n_iter_max),
                    Some(discrim[i]),
                    Some(discrim[i + 1]),
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
        let wv = bp.wave_hom(x_root.to_real(), id, tap)?;
        let md = Mode::new(wv)?;
        let chi = md.wave.discrim.abs()
            / ExtR::max(discrim[i].abs(), discrim[i + 1].abs());
        process_mode(&md, n_iter, chi)?;
    }
    Ok(())
}
