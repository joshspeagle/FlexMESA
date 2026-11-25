// SPDX-License-Identifier: GPL-3.0-only
//! GYRE's reduced-order radial adiabatic problem: variable transforms
//! (`rad/rad_trans_m.fypp`), oscillation equations (`rad/rad_eqns_m.fypp`),
//! boundary conditions (`rad/rad_bound_m.fypp`, with
//! `common/atmos_m.fypp`), difference equations (`rad/rad_diff_m.fypp`
//! over `diff/colloc_diff_m.fypp`), and the boundary-value problem
//! (`rad/rad_bvp_m.fypp` over `bvp/bvp_m.fypp`).
//!
//! All `2 x 2` matrices are column-major `[a11, a21, a12, a22]`.

use crate::context::Context;
use crate::ext_complex::ExtC;
use crate::ext_real::ExtR;
use crate::fcomplex::Cplx;
use crate::grid::{Grid, Point};
use crate::lapack::lu::dgesvx;
use crate::magnus::{magnus_build_r, magnus_stencil};
use crate::matmul::matmul_rr;
use crate::model::idx;
use crate::par::{ModePar, NumPar, OscPar};
use crate::GyreError;

fn sq(x: f64) -> f64 {
    x * x
}

/// `rad_trans_t` (`rad_trans_m.fypp`) for the `GYRE` (`SET`, no
/// transformation) and `JCD` variable sets.
#[derive(Debug, Clone)]
pub struct RadTrans {
    /// Per stencil point: `[V_2, dV_2, c_1, dc_1]` (`J_V_2 .. J_DC_1`).
    coeff: Vec<[f64; 4]>,
    /// `set == JCD_SET` (otherwise `SET`).
    jcd: bool,
}

impl RadTrans {
    /// `rad_trans_t(cx, md_p, os_p)` (`rad_trans_m.fypp:96-128`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for variable sets other than `GYRE` and
    /// `JCD`.
    pub fn new(os_p: &OscPar) -> Result<Self, GyreError> {
        match os_p.variables_set.as_str() {
            "JCD" => Ok(Self {
                coeff: Vec::new(),
                jcd: true,
            }),
            "GYRE" => Ok(Self {
                coeff: Vec::new(),
                jcd: false,
            }),
            "DZIEM" | "MIX" | "LAGP" => Err(GyreError::Unported {
                what: "radial variables_set other than 'GYRE' and 'JCD'",
                at: "rad/rad_trans_m.fypp:110-117",
            }),
            _ => Err(GyreError::Abort("invalid variables_set")),
        }
    }

    /// `stencil(pt)` (`rad_trans_m.fypp:132-168`).
    ///
    /// # Errors
    ///
    /// Model evaluation errors.
    pub fn stencil(
        &mut self,
        cx: &Context<'_>,
        pt: &[Point],
    ) -> Result<(), GyreError> {
        let ml = cx.ml;
        ml.check(&[idx::V_2, idx::U, idx::C_1])?;
        self.coeff = Vec::with_capacity(pt.len());
        for &p in pt {
            let (v_2, dv_2) = if ml.is_vacuum(p)? {
                (f64::MAX, f64::MAX)
            } else {
                (ml.coeff(idx::V_2, p)?, ml.dcoeff(idx::V_2, p)?)
            };
            let c_1 = ml.coeff(idx::C_1, p)?;
            let dc_1 = ml.dcoeff(idx::C_1, p)?;
            self.coeff.push([v_2, dv_2, c_1, dc_1]);
        }
        Ok(())
    }

    /// `G_jcd_(i, st)` (`rad_trans_m.fypp:360-392`); `i` is 0-based.
    fn g(&self, i: usize, omega: f64) -> [f64; 4] {
        let c_1 = self.coeff[i][2];
        [1.0, 0.0, 0.0, 1.0 / (c_1 * sq(omega))]
    }

    /// `H_jcd_(i, st)` (`rad_trans_m.fypp:454-486`).
    fn h(&self, i: usize, omega: f64) -> [f64; 4] {
        let c_1 = self.coeff[i][2];
        [1.0, 0.0, 0.0, c_1 * sq(omega)]
    }

    /// `dH_jcd_(i, st)` (`rad_trans_m.fypp:547-580`).
    fn dh(&self, i: usize, omega: f64) -> [f64; 4] {
        let c_1 = self.coeff[i][2];
        let dc_1 = self.coeff[i][3];
        [0.0, 0.0, 0.0, c_1 * dc_1 * sq(omega)]
    }

    /// `trans_eqns(xA, i, st)` with `from = .TRUE.`
    /// (`rad_trans_m.fypp:172-221`): `xA = MATMUL(G, MATMUL(xA, H) - dH)`.
    ///
    /// # Errors
    ///
    /// [`GyreError::Num`] if `MATMUL` were not inlined.
    pub fn trans_eqns(
        &self,
        xa: &[f64; 4],
        i: usize,
        omega: f64,
    ) -> Result<[f64; 4], GyreError> {
        // `set /= SET .AND. set /= DZIEM_SET .AND. set /= MIX_SET`
        if !self.jcd {
            return Ok(*xa);
        }
        let g = self.g(i, omega);
        let h = self.h(i, omega);
        let dh = self.dh(i, omega);
        let xah = matmul_rr(xa, &h, 2, 2, 2)?;
        let t: Vec<f64> =
            xah.iter().zip(dh.iter()).map(|(a, b)| a - b).collect();
        let r = matmul_rr(&g, &t, 2, 2, 2)?;
        Ok([r[0], r[1], r[2], r[3]])
    }

    /// `trans_cond(C, i, st)` with `from = .TRUE.`
    /// (`rad_trans_m.fypp:225-275`): `C = MATMUL(C, H)` for a `1 x 2` `C`.
    ///
    /// # Errors
    ///
    /// [`GyreError::Num`] if `MATMUL` were not inlined.
    pub fn trans_cond(
        &self,
        c: &[f64; 2],
        i: usize,
        omega: f64,
    ) -> Result<[f64; 2], GyreError> {
        if !self.jcd {
            return Ok(*c);
        }
        let h = self.h(i, omega);
        let r = matmul_rr(c, &h, 1, 2, 2)?;
        Ok([r[0], r[1]])
    }

    /// `trans_cond(C, i, st)` with `from = .TRUE.` for a `2 x 2` `C`
    /// (`rad_match`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Num`] if `MATMUL` were not inlined.
    pub fn trans_cond_2x2(
        &self,
        c: &[f64; 4],
        i: usize,
        omega: f64,
    ) -> Result<[f64; 4], GyreError> {
        if !self.jcd {
            return Ok(*c);
        }
        let h = self.h(i, omega);
        let r = matmul_rr(c, &h, 2, 2, 2)?;
        Ok([r[0], r[1], r[2], r[3]])
    }

    /// `trans_vars(y, i, st, from=.FALSE.)` (`rad_trans_m.fypp:279-329`):
    /// `y = MATMUL(H, y)`.
    #[must_use]
    pub fn trans_vars_to(&self, y: [f64; 2], i: usize, omega: f64) -> [f64; 2] {
        if !self.jcd {
            return y;
        }
        let h = self.h(i, omega);
        let r = crate::matmul::matvec_r(&h, &y, 2, 2);
        [r[0], r[1]]
    }
}

/// `rad_eqns_t` (`rad_eqns_m.fypp`).
#[derive(Debug, Clone)]
pub struct RadEqns {
    tr: RadTrans,
    alpha_omg: f64,
    /// Per stencil point `[V, As, U, c_1, Gamma_1]`.
    coeff: Vec<[f64; 5]>,
    x: Vec<f64>,
}

fn alpha_omg(os_p: &OscPar) -> Result<f64, GyreError> {
    match os_p.time_factor.as_str() {
        "OSC" => Ok(1.0),
        "EXP" => Err(GyreError::Unported {
            what: "time_factor = 'EXP'",
            at: "rad/rad_eqns_m.fypp:100-101",
        }),
        _ => Err(GyreError::Abort("invalid time_factor")),
    }
}

impl RadEqns {
    /// `rad_eqns_t(cx, md_p, os_p)` (`rad_eqns_m.fypp:84-112`).
    ///
    /// # Errors
    ///
    /// As [`RadTrans::new`]; invalid `time_factor`.
    pub fn new(os_p: &OscPar) -> Result<Self, GyreError> {
        Ok(Self {
            tr: RadTrans::new(os_p)?,
            alpha_omg: alpha_omg(os_p)?,
            coeff: Vec::new(),
            x: Vec::new(),
        })
    }

    /// `stencil(pt)` (`rad_eqns_m.fypp:116-160`).
    ///
    /// # Errors
    ///
    /// Model evaluation errors; a vacuum stencil point (`ASSERT`).
    pub fn stencil(
        &mut self,
        cx: &Context<'_>,
        pt: &[Point],
    ) -> Result<(), GyreError> {
        let ml = cx.ml;
        ml.check(&[idx::V_2, idx::AS, idx::U, idx::C_1, idx::GAMMA_1])?;
        self.coeff = Vec::with_capacity(pt.len());
        for &p in pt {
            if ml.is_vacuum(p)? {
                return Err(GyreError::Assert(
                    "Attempt to stencil at vacuum point",
                ));
            }
            self.coeff.push([
                ml.coeff(idx::V_2, p)? * sq(p.x),
                ml.coeff(idx::AS, p)?,
                ml.coeff(idx::U, p)?,
                ml.coeff(idx::C_1, p)?,
                ml.coeff(idx::GAMMA_1, p)?,
            ]);
        }
        self.tr.stencil(cx, pt)?;
        self.x = pt.iter().map(|p| p.x).collect();
        Ok(())
    }

    /// `xA(i, st)` (`rad_eqns_m.fypp:183-...`).
    fn xa(&self, i: usize, omega: f64) -> Result<[f64; 4], GyreError> {
        let [v, a_s, u, c_1, gamma_1] = self.coeff[i];
        let omega_c = omega;
        let xa = [
            v / gamma_1 - 1.0,
            c_1 * self.alpha_omg * sq(omega_c) + u - a_s,
            -(v / gamma_1),
            a_s - u + 3.0,
        ];
        self.tr.trans_eqns(&xa, i, omega)
    }

    /// `A(i, st) = xA(i, st)/pt(i)%x` (`rad_eqns_m.fypp:164-...`).
    ///
    /// # Errors
    ///
    /// As [`RadTrans::trans_eqns`].
    pub fn a(&self, i: usize, omega: f64) -> Result<[f64; 4], GyreError> {
        let xa = self.xa(i, omega)?;
        let x = self.x[i];
        Ok([xa[0] / x, xa[1] / x, xa[2] / x, xa[3] / x])
    }
}

/// `atmos_chi(V, As, c_1, Gamma_1, alpha_omg, omega, lambda)` for real
/// arguments (`atmos_chi_r_`, `common/atmos_m.fypp:59-114`).
#[must_use]
pub fn atmos_chi_r(
    v: f64,
    a_s: f64,
    c_1: f64,
    gamma_1: f64,
    alpha_omg: f64,
    omega: f64,
    lambda: f64,
) -> f64 {
    let a_11 = v / gamma_1 - 3.0;
    let a_12 = lambda / (c_1 * alpha_omg * sq(omega)) - v / gamma_1;
    let a_21 = c_1 * alpha_omg * sq(omega) - a_s;
    let a_22 = a_s + 1.0;
    let b = -(a_11 + a_22);
    let c = a_11 * a_22 - a_12 * a_21;
    let psi2 = sq(b) - 4.0 * c;
    if psi2 < 0.0 {
        // Upstream also warns once ('Discarding imaginary part ...').
        -b / 2.0
    } else if b >= 0.0 {
        (-b - psi2.sqrt()) / 2.0
    } else {
        2.0 * c / (-b + psi2.sqrt())
    }
}

/// `rad_bound_t` (`rad_bound_m.fypp`) with `REGULAR`/`ZERO_R` inner and
/// `JCD`/`VACUUM` outer boundaries.
#[derive(Debug, Clone)]
pub struct RadBound {
    tr: RadTrans,
    alpha_omg: f64,
    zero_r: bool,
    /// `coeff(1, J_C_1)`.
    c_1_i: f64,
    /// `coeff(2, [J_V, J_AS, J_C_1, J_GAMMA_1])` (`JCD_TYPE`).
    atm_o: [f64; 4],
    /// `type_o == VACUUM_TYPE` (otherwise `JCD_TYPE`).
    vacuum_o: bool,
}

impl RadBound {
    /// `n_i`, `n_o`, `n_e`.
    pub const N_I: usize = 1;
    /// See [`Self::N_I`].
    pub const N_O: usize = 1;
    /// See [`Self::N_I`].
    pub const N_E: usize = 2;

    /// `rad_bound_t(cx, md_p, os_p)` (`rad_bound_m.fypp:100-183`) and its
    /// `stencil_` (`:187-242`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for outer boundary types other than `VACUUM`
    /// and `JCD`; model evaluation errors.
    pub fn new(cx: &Context<'_>, os_p: &OscPar) -> Result<Self, GyreError> {
        let tr = RadTrans::new(os_p)?;
        let ml = cx.ml;
        let pt_i = cx.pt_i;
        let pt_o = cx.pt_o;
        let zero_r = match os_p.inner_bound.as_str() {
            "REGULAR" => {
                if pt_i.x != 0.0 {
                    return Err(GyreError::Assert(
                        "boundary condition invalid for x /= 0",
                    ));
                }
                false
            }
            "ZERO_R" => {
                if pt_i.x == 0.0 {
                    return Err(GyreError::Assert(
                        "boundary condition invalid for x == 0",
                    ));
                }
                true
            }
            _ => return Err(GyreError::Abort("invalid inner_bound")),
        };
        // rad_bound_m.fypp:132-162: every type falls back to VACUUM_TYPE at
        // a vacuum outer point.
        let vacuum_o = match os_p.outer_bound.as_str() {
            "VACUUM" => true,
            "JCD" => ml.is_vacuum(pt_o)?,
            "DZIEM" | "ISOTHERMAL" | "UNNO" => {
                if !ml.is_vacuum(pt_o)? {
                    return Err(GyreError::Unported {
                        what: "radial outer_bound 'DZIEM', 'ISOTHERMAL' or 'UNNO' at a non-vacuum outer point",
                        at: "rad/rad_bound_m.fypp:135-152",
                    });
                }
                true
            }
            _ => return Err(GyreError::Abort("invalid outer_bound")),
        };
        let alpha_omg = match os_p.time_factor.as_str() {
            "OSC" => 1.0,
            "EXP" => {
                return Err(GyreError::Unported {
                    what: "time_factor = 'EXP'",
                    at: "rad/rad_bound_m.fypp:166-167",
                })
            }
            _ => return Err(GyreError::Abort("invalid time_factor")),
        };
        // stencil_
        ml.check(&[idx::V_2, idx::U, idx::C_1, idx::GAMMA_1])?;
        let c_1_i = if zero_r {
            0.0
        } else {
            ml.coeff(idx::C_1, pt_i)?
        };
        let atm_o = if vacuum_o {
            // coeff(2,J_U) (rad_bound_m.fypp:217), not used by build_vacuum_o_
            let _ = ml.coeff(idx::U, pt_o)?;
            [0.0; 4]
        } else {
            // eval_atmos_coeffs_isothrm (atmos_m.fypp:275-296)
            let v = ml.coeff(idx::V_2, pt_o)? * sq(pt_o.x);
            let a_s = ml.coeff(idx::V_2, pt_o)?
                * sq(pt_o.x)
                * (1.0 - 1.0 / ml.coeff(idx::GAMMA_1, pt_o)?);
            let c_1 = ml.coeff(idx::C_1, pt_o)?;
            let gamma_1 = ml.coeff(idx::GAMMA_1, pt_o)?;
            [v, a_s, c_1, gamma_1]
        };
        let mut bd = Self {
            tr,
            alpha_omg,
            zero_r,
            c_1_i,
            atm_o,
            vacuum_o,
        };
        bd.tr.stencil(cx, &[pt_i, pt_o])?;
        Ok(bd)
    }

    /// `build_i(st, B, scl)` (`rad_bound_m.fypp:246-347`).
    ///
    /// # Errors
    ///
    /// As [`RadTrans::trans_cond`].
    pub fn build_i(
        &self,
        omega: f64,
    ) -> Result<([f64; 2], [f64; 1]), GyreError> {
        let omega_c = omega;
        let b = if self.zero_r {
            [1.0, 0.0]
        } else {
            [self.c_1_i * self.alpha_omg * sq(omega_c), 0.0]
        };
        Ok((self.tr.trans_cond(&b, 0, omega)?, [1.0]))
    }

    /// `build_o(st, B, scl)` with `build_vacuum_o_` or `build_jcd_o_`
    /// (`rad_bound_m.fypp:351-567`).
    ///
    /// # Errors
    ///
    /// As [`RadTrans::trans_cond`].
    pub fn build_o(
        &self,
        omega: f64,
    ) -> Result<([f64; 2], [f64; 1]), GyreError> {
        if self.vacuum_o {
            // build_vacuum_o_ (rad_bound_m.fypp:392-417)
            return Ok((self.tr.trans_cond(&[1.0, -1.0], 1, omega)?, [1.0]));
        }
        let [v, a_s, c_1, gamma_1] = self.atm_o;
        let omega_c = omega;
        let chi =
            atmos_chi_r(v, a_s, c_1, gamma_1, self.alpha_omg, omega_c, 0.0);
        let b_11 = v / gamma_1 - 3.0;
        let b_12 = -(v / gamma_1);
        let b = [chi - b_11, -b_12];
        Ok((self.tr.trans_cond(&b, 1, omega)?, [1.0]))
    }
}

/// Difference scheme selected by `diff_factory_m.fypp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffKind {
    /// `COLLOC_GL2`.
    CollocGl2,
    /// `COLLOC_GL4`.
    CollocGl4,
    /// `MAGNUS_GL2`.
    MagnusGl2,
}

/// Dispatch `diff_scheme` (`diff/diff_factory_m.fypp:63-98`).
///
/// # Errors
///
/// [`GyreError::Unported`] for the remaining schemes.
pub fn diff_kind(diff_scheme: &str) -> Result<DiffKind, GyreError> {
    match diff_scheme {
        "COLLOC_GL2" => Ok(DiffKind::CollocGl2),
        "COLLOC_GL4" => Ok(DiffKind::CollocGl4),
        "MAGNUS_GL2" => Ok(DiffKind::MagnusGl2),
        "MAGNUS_GL4" | "MAGNUS_GL6" | "COLLOC_GL6" | "TRAPZ" | "MIRK" => {
            Err(GyreError::Unported {
                what: "diff_scheme other than 'COLLOC_GL2', 'COLLOC_GL4', and 'MAGNUS_GL2'",
                at: "diff/diff_factory_m.fypp:74-89",
            })
        }
        _ => Err(GyreError::Abort("invalid diff_scheme")),
    }
}

/// `diff_scheme` of the collocation difference equations
/// (`diff/diff_factory_m.fypp:63-98`): `true` for `COLLOC_GL2`, `false`
/// for `COLLOC_GL4`.
///
/// # Errors
///
/// [`GyreError::Unported`] for the other schemes.
pub fn colloc_scheme(diff_scheme: &str) -> Result<bool, GyreError> {
    match diff_kind(diff_scheme)? {
        DiffKind::CollocGl2 => Ok(true),
        DiffKind::CollocGl4 => Ok(false),
        DiffKind::MagnusGl2 => Err(GyreError::Unported {
            what: "diff_scheme other than 'COLLOC_GL2' and 'COLLOC_GL4'",
            at: "diff/diff_factory_m.fypp:74-89",
        }),
    }
}

/// The collocation points of `colloc_diff_t(eq, pt_a, pt_b, scheme)`
/// (`colloc_diff_m.fypp:121-177`): `(reverse, dx, pt)`.
#[must_use]
pub fn colloc_stencil(
    pt_a: Point,
    pt_b: Point,
    gl2: bool,
) -> (bool, f64, Vec<Point>) {
    let c_i: Vec<f64> = if gl2 { vec![0.5] } else { gl4().2.to_vec() };
    let reverse = pt_a.x >= 0.5;
    let (dx, x0) = if reverse {
        (pt_a.x - pt_b.x, pt_b.x)
    } else {
        (pt_b.x - pt_a.x, pt_a.x)
    };
    let pt = c_i
        .iter()
        .map(|&c| Point {
            s: pt_a.s,
            x: x0 + c * dx,
        })
        .collect();
    (reverse, dx, pt)
}

/// Shared `build_GL2_` helper for `N_E x N_E` real collocation blocks
/// (`colloc_diff_m.fypp:216-244`).
pub(crate) fn colloc_gl2_build<const N_E: usize>(
    dx: f64,
    reverse: bool,
    a: &[f64],
) -> (Vec<f64>, Vec<f64>, ExtR) {
    let mut plus = vec![0.0_f64; N_E * N_E];
    let mut minus = vec![0.0_f64; N_E * N_E];
    for c in 0..N_E {
        for r in 0..N_E {
            let k = r + c * N_E;
            let h = 0.5 * dx * a[k];
            let id = if r == c { 1.0 } else { 0.0 };
            plus[k] = h + id;
            minus[k] = h - id;
        }
    }
    let scl = ExtR::from_real(1.0);
    if reverse {
        (minus, plus, scl)
    } else {
        (plus, minus, scl)
    }
}

/// Shared `build_GL4_` (`build_irk_`) helper for `N_E x N_E` real
/// collocation blocks (`colloc_diff_m.fypp:181-387`).
pub(crate) fn colloc_gl4_build<const N_E: usize>(
    dx: f64,
    reverse: bool,
    a: [&[f64]; 2],
) -> Result<(Vec<f64>, Vec<f64>, ExtR), GyreError> {
    const N_S: usize = 2;
    let nm = N_S * N_E;
    let (a_ij, b_i, _) = gl4();
    let mut m = vec![0.0_f64; nm * nm];
    let mut b = vec![0.0_f64; nm * N_E];
    for i in 0..N_S {
        let i_a = i * N_E;
        for j in 0..N_S {
            let j_a = j * N_E;
            let aij = a_ij[i + j * N_S];
            for c in 0..N_E {
                for r in 0..N_E {
                    m[(i_a + r) + (j_a + c) * nm] =
                        -(dx * aij * a[i][r + c * N_E]);
                }
            }
            if i == j {
                for c in 0..N_E {
                    for r in 0..N_E {
                        let id = if r == c { 1.0 } else { 0.0 };
                        m[(i_a + r) + (j_a + c) * nm] += id;
                    }
                }
            }
        }
        for c in 0..N_E {
            for r in 0..N_E {
                b[(i_a + r) + c * nm] = a[i][r + c * N_E];
            }
        }
    }
    let mut af = vec![0.0_f64; nm * nm];
    let mut ipiv = vec![0usize; nm];
    let mut x = vec![0.0_f64; nm * N_E];
    let o = dgesvx(
        b'E', b'N', nm, N_E, &mut m, &mut af, &mut ipiv, &mut b, &mut x,
    )?;
    if o.info != 0 {
        return Err(GyreError::LapackInfo {
            routine: "XGESVX",
            info: i64::try_from(o.info).unwrap_or(i64::MAX),
        });
    }
    let mut e_a: Vec<f64> = (0..N_E * N_E)
        .map(|k| if k.is_multiple_of(N_E + 1) { 1.0 } else { 0.0 })
        .collect();
    for i in 0..N_S {
        let i_a = i * N_E;
        for c in 0..N_E {
            for r in 0..N_E {
                let k = x[(i_a + r) + c * nm];
                e_a[r + c * N_E] += dx * b_i[i] * k;
            }
        }
    }
    let e_b: Vec<f64> = (0..N_E * N_E)
        .map(|k| {
            if k.is_multiple_of(N_E + 1) {
                -1.0
            } else {
                -0.0
            }
        })
        .collect();
    let scl = ExtR::from_real(1.0);
    if reverse {
        Ok((e_b, e_a, scl))
    } else {
        Ok((e_a, e_b, scl))
    }
}

/// Gauss-Legendre collocation constants (`colloc_diff_m.fypp:48-59`),
/// folded at compile time with one rounding per operation.
fn gl4() -> ([f64; 4], [f64; 2], [f64; 2]) {
    let r3_6 = 3.0_f64.sqrt() / 6.0;
    // GL4_A_IJ = RESHAPE([0.25, 0.25+r, 0.25-r, 0.25], [2,2]) (column-major)
    let a_ij = [0.25, 0.25 + r3_6, 0.25 - r3_6, 0.25];
    let b_i = [0.5, 0.5];
    let c_i = [0.5 - r3_6, 0.5 + r3_6];
    (a_ij, b_i, c_i)
}

/// `rad_match_t` (`rad/rad_match_m.fypp`).
#[derive(Debug, Clone)]
pub struct RadMatch {
    u_l: f64,
    u_r: f64,
    tr: RadTrans,
}

impl RadMatch {
    /// `rad_match_t(cx, pt_a, pt_b, md_p, os_p)` (`rad_match_m.fypp:80-132`).
    ///
    /// # Errors
    ///
    /// Model evaluation or transform errors.
    pub fn new(
        cx: &Context<'_>,
        pt_a: Point,
        pt_b: Point,
        os_p: &OscPar,
    ) -> Result<Self, GyreError> {
        let ml = cx.ml;
        ml.check(&[idx::U])?;
        let u_l = ml.coeff(idx::U, pt_a)?;
        let u_r = ml.coeff(idx::U, pt_b)?;
        let mut tr = RadTrans::new(os_p)?;
        tr.stencil(cx, &[pt_a, pt_b])?;
        Ok(Self { u_l, u_r, tr })
    }

    /// `build(st, E_l, E_r, scl)` (`rad_match_m.fypp:136-187`).
    ///
    /// # Errors
    ///
    /// Transform errors.
    pub fn build(
        &self,
        omega: f64,
    ) -> Result<([f64; 4], [f64; 4], ExtR), GyreError> {
        let e_l = self.tr.trans_cond_2x2(
            &[-1.0, self.u_l, 0.0, -self.u_l],
            0,
            omega,
        )?;
        let e_r = self.tr.trans_cond_2x2(
            &[1.0, -self.u_r, 0.0, self.u_r],
            1,
            omega,
        )?;
        Ok((e_l, e_r, ExtR::from_real(1.0)))
    }
}

#[derive(Debug, Clone)]
enum RadDiffKind {
    Colloc {
        eq: RadEqns,
        dx: f64,
        reverse: bool,
        gl2: bool,
    },
    MagnusGl2 {
        eq: RadEqns,
        dx: f64,
    },
    Match(RadMatch),
}

/// `rad_diff_t` (`rad_diff_m.fypp:69-131`): `COLLOC_GL2`, `COLLOC_GL4`,
/// `MAGNUS_GL2`, and segment `rad_match_t`.
#[derive(Debug, Clone)]
pub struct RadDiff {
    kind: RadDiffKind,
}

impl RadDiff {
    /// `rad_diff_t(cx, pt_a, pt_b, md_p, nm_p, os_p)`.
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for difference schemes other than `COLLOC_GL2`,
    /// `COLLOC_GL4`, and `MAGNUS_GL2`; stencil errors.
    pub fn new(
        cx: &Context<'_>,
        pt_a: Point,
        pt_b: Point,
        nm_p: &NumPar,
        os_p: &OscPar,
    ) -> Result<Self, GyreError> {
        if pt_a.s != pt_b.s {
            return Ok(Self {
                kind: RadDiffKind::Match(RadMatch::new(cx, pt_a, pt_b, os_p)?),
            });
        }
        let mut eq = RadEqns::new(os_p)?;
        match diff_kind(&nm_p.diff_scheme)? {
            DiffKind::CollocGl2 => {
                let (reverse, dx, pt) = colloc_stencil(pt_a, pt_b, true);
                eq.stencil(cx, &pt)?;
                Ok(Self {
                    kind: RadDiffKind::Colloc {
                        eq,
                        dx,
                        reverse,
                        gl2: true,
                    },
                })
            }
            DiffKind::CollocGl4 => {
                let (reverse, dx, pt) = colloc_stencil(pt_a, pt_b, false);
                eq.stencil(cx, &pt)?;
                Ok(Self {
                    kind: RadDiffKind::Colloc {
                        eq,
                        dx,
                        reverse,
                        gl2: false,
                    },
                })
            }
            DiffKind::MagnusGl2 => {
                let (dx, pt) = magnus_stencil(pt_a, pt_b, "GL2")?;
                eq.stencil(cx, &pt)?;
                Ok(Self {
                    kind: RadDiffKind::MagnusGl2 { eq, dx },
                })
            }
        }
    }

    /// `build(st, E_l, E_r, scl)` (`rad_diff_m.fypp:109-131`).
    ///
    /// # Errors
    ///
    /// LAPACK errors, eigendecomposition errors, and nonzero `XGESVX` info.
    pub fn build(
        &self,
        omega: f64,
    ) -> Result<([f64; 4], [f64; 4], ExtR), GyreError> {
        let (e_l, e_r, scl) = match &self.kind {
            RadDiffKind::Match(mt) => return mt.build(omega),
            RadDiffKind::MagnusGl2 { eq, dx } => {
                let mut domega = eq.a(0, omega)?;
                magnus_build_r(2, *dx, &mut domega)?
            }
            RadDiffKind::Colloc {
                eq,
                dx,
                reverse,
                gl2: true,
            } => {
                let a = eq.a(0, omega)?;
                colloc_gl2_build::<2>(*dx, *reverse, &a)
            }
            RadDiffKind::Colloc {
                eq,
                dx,
                reverse,
                gl2: false,
            } => {
                let a0 = eq.a(0, omega)?;
                let a1 = eq.a(1, omega)?;
                colloc_gl4_build::<2>(*dx, *reverse, [&a0, &a1])?
            }
        };
        Ok((
            [e_l[0], e_l[1], e_l[2], e_l[3]],
            [e_r[0], e_r[1], e_r[2], e_r[3]],
            scl,
        ))
    }
}

/// `rad_bvp_t` (`rad_bvp_m.fypp:84-148`) over `bvp_rt` (`bvp_m.fypp`).
#[derive(Debug, Clone)]
pub struct RadBvp<'a> {
    /// The context.
    pub cx: &'a Context<'a>,
    /// The (refined) grid.
    pub gr: Grid,
    bd: RadBound,
    df: Vec<RadDiff>,
    /// Variable transform on the full grid.
    pub tr: RadTrans,
    /// The system matrix.
    pub sm: crate::sysmtx::BlockSysmtx<f64>,
    factored: bool,
    /// `md_p`, `nm_p`, `os_p`.
    pub md_p: ModePar,
    /// See [`Self::md_p`].
    pub nm_p: NumPar,
    /// See [`Self::md_p`].
    pub os_p: OscPar,
}

/// Observer of intermediate values at the points where the G0 oracle
/// taps sit (`benchmarks/fortran_mesa/gyre_oracles/apply_taps.py`). The
/// default methods do nothing.
pub trait GyreTap {
    /// After `refine_grid` (`lib/gyre_mesa_m.fypp:319`): the refined grid
    /// and the real scan frequencies.
    fn grid_scan(&mut self, _gr: &Grid, _omega_re: &[f64]) {}
    /// Start of `bvp%build` after the boundary rows are set
    /// (`bvp_m.fypp:155`); `mode_build` marks builds made to construct a
    /// mode.
    fn build(
        &mut self,
        _n: usize,
        _mode_build: bool,
        _omega: f64,
        _b_i: &[f64; 2],
        _b_o: &[f64; 2],
    ) {
    }
    /// After `set_E(j, ...)` (`bvp_m.fypp:162`), `j` 1-based.
    fn block(
        &mut self,
        _j: usize,
        _e_l: &[f64; 4],
        _e_r: &[f64; 4],
        _scl: ExtR,
    ) {
    }
    /// After `discrim = bp%det()` (`search/discrim_m.fypp:67`).
    fn discrim(&mut self, _omega: f64, _det: ExtR) {}
    /// [`Self::build`] for a complex (`bvp_ct`) build: `B_i(3, 6)` and
    /// `B_o(3, 6)` column-major.
    fn build_c(
        &mut self,
        _n: usize,
        _mode_build: bool,
        _omega: Cplx,
        _b_i: &[Cplx; 18],
        _b_o: &[Cplx; 18],
    ) {
    }
    /// [`Self::block`] for a complex build: `6 x 6` blocks, column-major.
    fn block_c(
        &mut self,
        _j: usize,
        _e_l: &[Cplx; 36],
        _e_r: &[Cplx; 36],
        _scl: ExtC,
    ) {
    }
    /// [`Self::discrim`] for a complex discriminant.
    fn discrim_c(&mut self, _omega: Cplx, _det: ExtC) {}
    /// [`Self::build`] for a 4x4 adiabatic (`ad_bvp_t`) build:
    /// `B_i(2, 4)` and `B_o(2, 4)` column-major.
    fn build_ad(
        &mut self,
        _n: usize,
        _mode_build: bool,
        _omega: f64,
        _b_i: &[f64; 8],
        _b_o: &[f64; 8],
    ) {
    }
    /// [`Self::block`] for a 4x4 adiabatic build: `4 x 4` blocks, column-major.
    fn block_ad(
        &mut self,
        _j: usize,
        _e_l: &[f64; 16],
        _e_r: &[f64; 16],
        _scl: ExtR,
    ) {
    }
    /// An adiabatic mode that passed the `n_pg` filter of
    /// `process_mode_ad` (`lib/gyre_mesa_m.fypp:417`): `omega` and
    /// `[l, n_p, n_g, n_pg, n]`.
    fn mode_ad(&mut self, _omega: Cplx, _ints: [i64; 5]) {}
    /// A nonadiabatic mode passed to `process_mode_nad`
    /// (`lib/gyre_mesa_m.fypp:462`).
    fn mode_nad(&mut self, _omega: Cplx, _ints: [i64; 5]) {}
    /// Start of a mode-parameter group in `get_modes`.
    fn mode_par(&mut self, _l: i32, _i: usize) {}
}

/// A tap that records nothing.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoTap;

impl GyreTap for NoTap {}

impl<'a> RadBvp<'a> {
    /// `rad_bvp_t(cx, gr, md_p, nm_p, os_p)`.
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for `alpha_grv /= 1`, quasi-adiabatic
    /// eigenfunctions and the paths of the components.
    pub fn new(
        cx: &'a Context<'a>,
        gr: &Grid,
        md_p: &ModePar,
        nm_p: &NumPar,
        os_p: &OscPar,
    ) -> Result<Self, GyreError> {
        if os_p.alpha_grv != 1.0 {
            return Err(GyreError::Unported {
                what: "radial problem with alpha_grv /= 1",
                at: "rad/rad_bvp_m.fypp:102-104",
            });
        }
        let bd = RadBound::new(cx, os_p)?;
        let mut df = Vec::with_capacity(gr.n() - 1);
        for j in 0..gr.n() - 1 {
            df.push(RadDiff::new(cx, gr.pt[j], gr.pt[j + 1], nm_p, os_p)?);
        }
        let n = df.len() + 1;
        match nm_p.matrix_type.as_str() {
            "BLOCK" => {}
            "BAND" => {
                return Err(GyreError::Unported {
                    what: "matrix_type = 'BAND'",
                    at: "matrix/sysmtx_factory_m.fypp:71-72",
                })
            }
            _ => return Err(GyreError::Abort("invalid matrix_type")),
        }
        let sm = crate::sysmtx::BlockSysmtx::new(
            n - 1,
            RadBound::N_E,
            RadBound::N_I,
            RadBound::N_O,
        );
        let mut tr = RadTrans::new(os_p)?;
        tr.stencil(cx, &gr.pt)?;
        if os_p.quasiad_eigfuncs {
            return Err(GyreError::Unported {
                what: "quasiad_eigfuncs",
                at: "rad/rad_bvp_m.fypp:134-138",
            });
        }
        Ok(Self {
            cx,
            gr: gr.clone(),
            bd,
            df,
            tr,
            sm,
            factored: false,
            md_p: md_p.clone(),
            nm_p: nm_p.clone(),
            os_p: os_p.clone(),
        })
    }

    /// Number of grid points `n`.
    #[must_use]
    pub fn n(&self) -> usize {
        self.df.len() + 1
    }

    /// `build(st)` (`bvp_m.fypp:133-173`).
    ///
    /// # Errors
    ///
    /// Component errors.
    pub fn build(
        &mut self,
        omega: f64,
        mode_build: bool,
        tap: &mut dyn GyreTap,
    ) -> Result<(), GyreError> {
        let (b_i, scl_i) = self.bd.build_i(omega)?;
        self.sm.set_b_i(&b_i, &scl_i);
        let (b_o, scl_o) = self.bd.build_o(omega)?;
        self.sm.set_b_o(&b_o, &scl_o);
        tap.build(self.n(), mode_build, omega, &b_i, &b_o);
        for j in 0..self.n() - 1 {
            let (e_l, e_r, scl) = self.df[j].build(omega)?;
            self.sm.set_e(j, &e_l, &e_r, scl);
            tap.block(j + 1, &e_l, &e_r, scl);
        }
        self.factored = false;
        Ok(())
    }

    /// `factor()` (`bvp_m.fypp:177-191`).
    ///
    /// # Errors
    ///
    /// LAPACK errors.
    pub fn factor(&mut self) -> Result<(), GyreError> {
        self.sm.factor()?;
        self.factored = true;
        Ok(())
    }

    /// `det()` (`bvp_m.fypp:195-210`).
    ///
    /// # Errors
    ///
    /// Unfactored matrix (`ASSERT`), LAPACK errors.
    pub fn det(&self) -> Result<ExtR, GyreError> {
        if !self.factored {
            return Err(GyreError::Assert("matrix has not been factorized"));
        }
        self.sm.det()
    }

    /// `soln_vec_hom()` (`bvp_m.fypp:214-233`): `y(n_e, n)` column-major.
    ///
    /// # Errors
    ///
    /// LAPACK errors.
    pub fn soln_vec_hom(&self) -> Result<Vec<f64>, GyreError> {
        self.sm.soln_vec_hom()
    }
}
