// SPDX-License-Identifier: GPL-3.0-only
//! GYRE's complex root solving in extended arithmetic (`math/root_m.fypp`,
//! `xc` instances: Ridders' method and bracket expansion) and the
//! proximity search from adiabatic frequencies
//! (`search/prox_search_m.fypp`).
//!
//! Mixed `ext_ct`/real operands go through `ext_ct(r)` as upstream's
//! `op_*_cx_r_` and `op_*_r_cx_` do (`ext/ext_complex_m.fypp:473-571`).

use crate::crmath::abs_c;
use crate::ext_complex::ExtC;
use crate::ext_real::ExtR;
use crate::fcomplex::Cplx;
use crate::mode::Mode;
use crate::nad::{nad_wave_hom, NadBvp};
use crate::par::NumPar;
use crate::rad::GyreTap;
use crate::search::Stat;
use crate::GyreError;

/// `eval_discrim(bp, st, discrim)` for the nonadiabatic BVP
/// (`discrim_m.fypp:56-73`).
///
/// # Errors
///
/// BVP errors.
pub fn eval_discrim_c(
    bp: &mut NadBvp<'_>,
    omega: Cplx,
    tap: &mut dyn GyreTap,
) -> Result<ExtC, GyreError> {
    bp.build(omega, false, tap)?;
    bp.factor()?;
    let d = bp.det()?;
    tap.discrim_c(omega, d);
    Ok(d)
}

fn xr(r: f64) -> ExtR {
    ExtR::from_real(r)
}

fn xc(r: f64) -> ExtC {
    ExtC::from_real(r)
}

/// The result of a narrowing or expansion: the bracket, its function
/// values and the iteration count.
struct Bracket {
    z_a: ExtC,
    z_b: ExtC,
    f_a: ExtC,
    f_b: ExtC,
    n_iter: i32,
}

/// `narrow_bracket_xc_` (`root_m.fypp:564-605`) with
/// `c_root_solver = 'RIDDERS'` (`narrow_ridders_xc_`, `:753-919`) and
/// `relative_tol` absent. `f_z` holds the optional `f_z_a`, `f_z_b`.
/// Returns the status and, when the initial evaluations succeed, the
/// narrowed bracket.
fn narrow_bracket(
    eval: &mut dyn FnMut(ExtC) -> Result<(ExtC, Stat), GyreError>,
    z_a: ExtC,
    z_b: ExtC,
    z_tol: ExtR,
    nm_p: &NumPar,
    n_iter_max: i32,
    f_z: Option<(ExtC, ExtC)>,
) -> Result<(Stat, Option<Bracket>), GyreError> {
    match nm_p.c_root_solver.as_str() {
        "RIDDERS" => {}
        "SECANT" => {
            return Err(GyreError::Unported {
                what: "c_root_solver = 'SECANT'",
                at: "math/root_m.fypp:609-749",
            })
        }
        _ => return Err(GyreError::Abort("invalid c_root_solver")),
    }
    if z_a.ext_eq(z_b) {
        return Err(GyreError::Assert("invalid initial bracket"));
    }
    let mut a = z_a;
    let mut b = z_b;
    let (mut f_a, mut f_b) = if let Some(f) = f_z {
        f
    } else {
        let (fa, st) = eval(a)?;
        if st != Stat::Ok {
            return Ok((st, None));
        }
        let (fb, st) = eval(b)?;
        if st != Stat::Ok {
            return Ok((st, None));
        }
        (fa, fb)
    };
    if f_a.abs().lt(f_b.abs()) {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut f_a, &mut f_b);
    }
    let eps = f64::EPSILON;
    let mut i_iter = 0;
    let mut stat = Stat::Ok;
    loop {
        if f_b.ext_eq(xc(0.0)) {
            break;
        }
        i_iter += 1;
        if i_iter > n_iter_max {
            stat = Stat::IterMax;
            break;
        }
        let c = xc(0.5) * (a + b);
        let (f_c, st) = eval(c)?;
        stat = st;
        if stat != Stat::Ok {
            break;
        }
        // sqrt(f_c*f_c - f_a*f_b), written twice upstream
        let disc = (f_c * f_c - f_a * f_b).sqrt();
        let exp_q_p = (f_c + disc) / f_b;
        let exp_q_m = (f_c - disc) / f_b;
        let exp_q = if (exp_q_p - xc(1.0)).abs().lt((exp_q_m - xc(1.0)).abs()) {
            exp_q_p
        } else {
            exp_q_m
        };
        let f_dz = f_b * (exp_q * exp_q) * (b - a);
        let rho = f_b * (exp_q * exp_q) - f_a;
        if (b * rho).abs().lt(xr(8.0 * eps) * f_dz.abs()) {
            return Err(GyreError::Abort("singular correction in secant"));
        }
        a = b;
        f_a = f_b;
        b = b - f_dz / rho;
        let (fb, st) = eval(b)?;
        f_b = fb;
        stat = st;
        if stat != Stat::Ok {
            break;
        }
        let tol = xr(4.0 * eps) * b.abs() + z_tol;
        if (b - a).abs().le(tol) {
            break;
        }
    }
    Ok((
        stat,
        Some(Bracket {
            z_a: a,
            z_b: b,
            f_a,
            f_b,
            n_iter: i_iter,
        }),
    ))
}

/// `expand_bracket_xc_` (`root_m.fypp:923-1026`) without clamping and
/// with `relative_tol` absent. Returns the status and the expanded
/// bracket when the initial evaluations succeed.
fn expand_bracket(
    eval: &mut dyn FnMut(ExtC) -> Result<(ExtC, Stat), GyreError>,
    z_a: ExtC,
    z_b: ExtC,
    f_z_tol: ExtR,
    golden_r: f64,
) -> Result<(Stat, Option<Bracket>), GyreError> {
    if z_a.ext_eq(z_b) {
        return Err(GyreError::Assert("invalid initial bracket"));
    }
    let (mut z_a, mut z_b) = (z_a, z_b);
    let (mut f_a, st) = eval(z_a)?;
    if st != Stat::Ok {
        return Ok((st, None));
    }
    let (mut f_b, st) = eval(z_b)?;
    if st != Stat::Ok {
        return Ok((st, None));
    }
    let mut stat = Stat::Ok;
    loop {
        let tol =
            xr(4.0 * f64::EPSILON) * ExtR::max(f_a.abs(), f_b.abs()) + f_z_tol;
        if (f_a - f_b).abs().gt(tol) {
            break;
        }
        let move_a = f_b.abs().gt(f_a.abs());
        if move_a {
            z_a = z_a + xc(golden_r) * (z_a - z_b);
            let (f, st) = eval(z_a)?;
            f_a = f;
            stat = st;
        } else {
            z_b = z_b + xc(golden_r) * (z_b - z_a);
            let (f, st) = eval(z_b)?;
            f_b = f;
            stat = st;
        }
        if stat != Stat::Ok {
            break;
        }
    }
    Ok((
        stat,
        Some(Bracket {
            z_a,
            z_b,
            f_a,
            f_b,
            n_iter: 0,
        }),
    ))
}

/// `prox_search_1_` and `prox_search_2_` (`prox_search_m.fypp:62-347`)
/// for the nonadiabatic BVP. A failed stage is reported and skipped, as
/// upstream does.
///
/// # Errors
///
/// BVP and mode-construction errors; upstream `ASSERT`/`ABORT`s.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn prox_search<'a>(
    bp: &mut NadBvp<'a>,
    omega_in: &[Cplx],
    id_in: &[i32],
    omega_min: f64,
    omega_max: f64,
    nm_p: &NumPar,
    tap: &mut dyn GyreTap,
    process_mode: &mut dyn FnMut(&Mode<'a>, i32, ExtR) -> Result<(), GyreError>,
) -> Result<(), GyreError> {
    if id_in.len() != omega_in.len() {
        return Err(GyreError::Assert("CHECK_BOUNDS: SIZE(id_in)"));
    }
    // prox_search_1_: convert initial frequencies into pairs
    // (sqrt(EPSILON(0._RD)) folds to 2**-26).
    let sqrt_eps = f64::from_bits(0x3E50_0000_0000_0000); // 2**-26
    let mut omega_in_a = Vec::with_capacity(omega_in.len());
    let mut omega_in_b = Vec::with_capacity(omega_in.len());
    for &w in omega_in {
        let domega = Cplx::real_mul(sqrt_eps * abs_c(w), Cplx::new(0.0, 1.0));
        omega_in_a.push(w + domega);
        omega_in_b.push(w - domega);
    }
    let golden_r = bp.cx.k.math.golden_r;

    // prox_search_2_
    let mut omega_def: Vec<Cplx> = Vec::new();
    for i in 0..omega_in_a.len() {
        let mut n_iter_def = 0;
        // omega_r = real(0.5_RD*(omega_in_a(i) + omega_in_b(i)))
        let omega_r = Cplx::real_mul(0.5, omega_in_a[i] + omega_in_b[i]).re;
        let _ = omega_r; // state_ct%omega_r: read only by FILE eps sources
        let discrim_a = eval_discrim_c(bp, omega_in_a[i], tap)?;
        let discrim_b = eval_discrim_c(bp, omega_in_b[i], tap)?;
        let mut z_a = ExtC::from_cplx(omega_in_a[i]);
        let mut z_b = ExtC::from_cplx(omega_in_b[i]);
        let mut f_z_a = discrim_a;
        let mut f_z_b = discrim_b;

        // eval_discrim_z_
        let eval_z = |bp: &mut NadBvp<'a>,
                      tap: &mut dyn GyreTap,
                      z: ExtC,
                      deflate: Option<&[Cplx]>|
         -> Result<(ExtC, Stat), GyreError> {
            let omega = z.to_cplx();
            if omega.re >= omega_min && omega.re <= omega_max {
                let mut d = eval_discrim_c(bp, omega, tap)?;
                if let Some(def) = deflate {
                    // PRODUCT(omega/(omega - omega_def))
                    let mut p = Cplx::new(1.0, 0.0);
                    for &w in def {
                        p = p * (omega / (omega - w));
                    }
                    d = d * ExtC::from_cplx(p);
                }
                Ok((d, Stat::Ok))
            } else {
                Ok((ExtC::from_real(0.0), Stat::OmegaDomain))
            }
        };

        if nm_p.deflate_roots {
            let def = omega_def.clone();
            let (stat, br) = narrow_bracket(
                &mut |z| eval_z(bp, tap, z, Some(&def)),
                z_a,
                z_b,
                xr(0.0),
                nm_p,
                nm_p.n_iter_max,
                None,
            )?;
            if stat != Stat::Ok {
                // report_stat_(stat, 'deflate narrow')
                continue;
            }
            let br = br.ok_or(GyreError::Assert("narrow_bracket state"))?;
            z_a = br.z_a;
            z_b = br.z_b;
            n_iter_def = br.n_iter;
            if z_b.ext_eq(z_a) {
                z_b = z_a
                    * (xc(1.0)
                        + xc(f64::EPSILON)
                            * (z_a / ExtC::from_extr(z_a.abs())));
            }
            let (stat, br) = expand_bracket(
                &mut |z| eval_z(bp, tap, z, Some(&def)),
                z_a,
                z_b,
                xr(0.0),
                golden_r,
            )?;
            if stat != Stat::Ok {
                // report_stat_(stat, 'deflate re-expand')
                continue;
            }
            let br = br.ok_or(GyreError::Assert("expand_bracket state"))?;
            z_a = br.z_a;
            z_b = br.z_b;
            f_z_a = br.f_a;
            f_z_b = br.f_b;
        }

        // solve_root_xc_ (root_m.fypp:136-195) without deflation
        let (stat, br) = narrow_bracket(
            &mut |z| eval_z(bp, tap, z, None),
            z_a,
            z_b,
            xr(0.0),
            nm_p,
            nm_p.n_iter_max - n_iter_def,
            Some((f_z_a, f_z_b)),
        )?;
        if stat != Stat::Ok {
            // report_stat_(stat, 'solve')
            continue;
        }
        let br = br.ok_or(GyreError::Assert("narrow_bracket state"))?;
        let z_root = br.z_b;
        let n_iter = br.n_iter;

        let wv = nad_wave_hom(bp, z_root.to_cplx(), id_in[i], tap)?;
        let md = Mode::new(wv)?;
        let chi =
            md.wave.discrim.abs() / ExtR::max(discrim_a.abs(), discrim_b.abs());
        process_mode(&md, n_iter_def + n_iter, chi)?;
        omega_def.push(z_root.to_cplx());
    }
    Ok(())
}
