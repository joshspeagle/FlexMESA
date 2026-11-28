// SPDX-License-Identifier: GPL-3.0-only
//! GYRE's non-radial adiabatic problem (`ad/*_m.fypp`): variable transforms
//! (`ad_trans_m`), oscillation equations (`ad_eqns_m`), boundary conditions
//! (`ad_bound_m`), match conditions (`ad_match_m`), difference equations
//! (`ad_diff_m`), and the BVP (`ad_bvp_m`).
//!
//! All `4 x 4` matrices are column-major (`a[(i-1) + 4*(j-1)]`).

use crate::context::Context;
use crate::ext_complex::ExtC;
use crate::ext_real::ExtR;
use crate::fcomplex::Cplx;
use crate::grid::{Grid, Point};
use crate::magnus::{magnus_build_r, magnus_stencil};
use crate::matmul::{matmul_rr, matvec_r};
use crate::mode::Wave;
use crate::model::idx;
use crate::par::{ModePar, NumPar, OscPar};
use crate::rad::{
    atmos_chi_r, colloc_gl2_build, colloc_gl4_build, colloc_stencil, diff_kind,
    DiffKind, GyreTap,
};
use crate::sysmtx::BlockSysmtx;
use crate::GyreError;

const N_E: usize = 4;

fn sq(x: f64) -> f64 {
    x * x
}

fn ix(i: usize, j: usize) -> usize {
    (i - 1) + N_E * (j - 1)
}

/// `ad_trans_t` (`ad/ad_trans_m.fypp`) for the `GYRE` (`SET`) and `JCD`
/// (`JCD_SET`) variable sets.
#[derive(Debug, Clone)]
pub struct AdTrans {
    pt: Vec<Point>,
    /// `[V_2, dV_2, U, dU, c_1, dc_1]` per stencil point.
    coeff: Vec<[f64; 6]>,
    l: i32,
    jcd: bool,
}

impl AdTrans {
    /// `ad_trans_t(cx, md_p, os_p)` (`ad_trans_m.fypp:105-139`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for `DZIEM`, `MIX`, and `LAGP`;
    /// [`GyreError::Abort`] for unrecognized `variables_set`.
    pub fn new(md_p: &ModePar, os_p: &OscPar) -> Result<Self, GyreError> {
        let jcd = match os_p.variables_set.as_str() {
            "GYRE" => false,
            "JCD" => true,
            "DZIEM" | "MIX" | "LAGP" => {
                return Err(GyreError::Unported {
                    what: "adiabatic variables_set other than 'GYRE' and 'JCD'",
                    at: "ad/ad_trans_m.fypp:116-129",
                });
            }
            _ => return Err(GyreError::Abort("invalid variables_set")),
        };
        Ok(Self {
            pt: Vec::new(),
            coeff: Vec::new(),
            l: md_p.l,
            jcd,
        })
    }

    /// `stencil(pt)` (`ad_trans_m.fypp:143-186`).
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
            let (v_2, dv_2, du) = if ml.is_vacuum(p)? {
                (f64::MAX, f64::MAX, -f64::MAX)
            } else {
                (
                    ml.coeff(idx::V_2, p)?,
                    ml.dcoeff(idx::V_2, p)?,
                    ml.dcoeff(idx::U, p)?,
                )
            };
            let u = ml.coeff(idx::U, p)?;
            let c_1 = ml.coeff(idx::C_1, p)?;
            let dc_1 = ml.dcoeff(idx::C_1, p)?;
            self.coeff.push([v_2, dv_2, u, du, c_1, dc_1]);
        }
        self.pt = pt.to_vec();
        Ok(())
    }

    fn jcd_scales(
        &self,
        cx: &Context<'_>,
        i: usize,
        omega: f64,
    ) -> Result<(f64, Option<f64>), GyreError> {
        let omega_rot = cx.omega_rot(self.pt[i])?;
        let omega_c = cx.omega_c_r(omega_rot, omega);
        let lambda = (self.l != 0).then(|| cx.lambda_r(omega_rot, omega));
        Ok((sq(omega_c), lambda))
    }

    /// `G_jcd_(i, st)` (`ad_trans_m.fypp:412-491`); `i` is 0-based.
    ///
    /// # Errors
    ///
    /// Context rotation errors.
    pub fn g_jcd(
        &self,
        cx: &Context<'_>,
        i: usize,
        omega: f64,
    ) -> Result<[f64; 16], GyreError> {
        let [_, _, u, _, c_1, _] = self.coeff[i];
        let (w2, lambda) = self.jcd_scales(cx, i, omega)?;
        let scale = lambda.unwrap_or(1.0) / (c_1 * w2);
        let mut g = [0.0_f64; 16];
        g[ix(1, 1)] = 1.0;
        g[ix(2, 2)] = scale;
        g[ix(2, 3)] = scale;
        g[ix(3, 3)] = -1.0;
        g[ix(4, 3)] = -(1.0 - u);
        g[ix(4, 4)] = -1.0;
        Ok(g)
    }

    /// `H_jcd_(i, st)` (`ad_trans_m.fypp:656-735`); `i` is 0-based.
    ///
    /// # Errors
    ///
    /// Context rotation errors.
    pub fn h_jcd(
        &self,
        cx: &Context<'_>,
        i: usize,
        omega: f64,
    ) -> Result<[f64; 16], GyreError> {
        let [_, _, u, _, c_1, _] = self.coeff[i];
        let (w2, lambda) = self.jcd_scales(cx, i, omega)?;
        let mut h = [0.0_f64; 16];
        h[ix(1, 1)] = 1.0;
        h[ix(2, 2)] = match lambda {
            Some(lam) => c_1 * w2 / lam,
            None => c_1 * w2,
        };
        h[ix(2, 3)] = 1.0;
        h[ix(3, 3)] = -1.0;
        h[ix(4, 3)] = 1.0 - u;
        h[ix(4, 4)] = -1.0;
        Ok(h)
    }

    /// `dH_jcd_(i, st)` (`ad_trans_m.fypp:861-943`); `i` is 0-based.
    ///
    /// # Errors
    ///
    /// Context rotation errors.
    pub fn dh_jcd(
        &self,
        cx: &Context<'_>,
        i: usize,
        omega: f64,
    ) -> Result<[f64; 16], GyreError> {
        let [_, _, u, du, c_1, dc_1] = self.coeff[i];
        let (w2, lambda) = self.jcd_scales(cx, i, omega)?;
        let mut dh = [0.0_f64; 16];
        dh[ix(2, 2)] = match lambda {
            Some(lam) => c_1 * dc_1 * w2 / lam,
            None => c_1 * dc_1 * w2,
        };
        dh[ix(4, 3)] = -u * du;
        Ok(dh)
    }

    /// `trans_eqns(xA, i, st)` with `from = .TRUE.`
    /// (`ad_trans_m.fypp:190-237`): `xA = MATMUL(G, MATMUL(xA, H) - dH)`.
    ///
    /// # Errors
    ///
    /// Context or matrix multiplication errors.
    pub fn trans_eqns(
        &self,
        cx: &Context<'_>,
        xa: &[f64; 16],
        i: usize,
        omega: f64,
    ) -> Result<[f64; 16], GyreError> {
        if !self.jcd {
            return Ok(*xa);
        }
        let g = self.g_jcd(cx, i, omega)?;
        let h = self.h_jcd(cx, i, omega)?;
        let dh = self.dh_jcd(cx, i, omega)?;
        let xah = matmul_rr(xa, &h, 4, 4, 4)?;
        let mut t = [0.0_f64; 16];
        for k in 0..16 {
            t[k] = xah[k] - dh[k];
        }
        let r = matmul_rr(&g, &t, 4, 4, 4)?;
        Ok(std::array::from_fn(|k| r[k]))
    }

    fn trans_cond_rows(
        &self,
        cx: &Context<'_>,
        c: &[f64],
        rows: usize,
        i: usize,
        omega: f64,
    ) -> Result<Vec<f64>, GyreError> {
        let h = self.h_jcd(cx, i, omega)?;
        Ok(matmul_rr(c, &h, rows, 4, 4)?)
    }

    /// `trans_cond(C, i, st)` with `from = .TRUE.`
    /// (`ad_trans_m.fypp:241-287`): `C = MATMUL(C, H)` for a `2 x 4` `C`
    /// (`ad_bound`).
    ///
    /// # Errors
    ///
    /// Context or matrix multiplication errors.
    pub fn trans_cond_2x4(
        &self,
        cx: &Context<'_>,
        c: &[f64; 8],
        i: usize,
        omega: f64,
    ) -> Result<[f64; 8], GyreError> {
        if !self.jcd {
            return Ok(*c);
        }
        let r = self.trans_cond_rows(cx, c, 2, i, omega)?;
        Ok(std::array::from_fn(|k| r[k]))
    }

    /// `trans_cond(C, i, st)` with `from = .TRUE.` for a `4 x 4` `C`
    /// (`ad_match`).
    ///
    /// # Errors
    ///
    /// Context or matrix multiplication errors.
    pub fn trans_cond_4x4(
        &self,
        cx: &Context<'_>,
        c: &[f64; 16],
        i: usize,
        omega: f64,
    ) -> Result<[f64; 16], GyreError> {
        if !self.jcd {
            return Ok(*c);
        }
        let r = self.trans_cond_rows(cx, c, 4, i, omega)?;
        Ok(std::array::from_fn(|k| r[k]))
    }

    /// `trans_vars(y, i, st, from=.FALSE.)` (`ad_trans_m.fypp:291-337`):
    /// `y = MATMUL(H, y)`.
    ///
    /// # Errors
    ///
    /// Context rotation errors.
    pub fn trans_vars_to(
        &self,
        cx: &Context<'_>,
        y: [f64; 4],
        i: usize,
        omega: f64,
    ) -> Result<[f64; 4], GyreError> {
        if !self.jcd {
            return Ok(y);
        }
        let h = self.h_jcd(cx, i, omega)?;
        let r = matvec_r(&h, &y, 4, 4);
        Ok([r[0], r[1], r[2], r[3]])
    }
}

#[derive(Debug, Clone, Copy)]
struct AdCoeff {
    v: f64,
    a_s: f64,
    u: f64,
    c_1: f64,
    gamma_1: f64,
}

/// `ad_eqns_t` (`ad/ad_eqns_m.fypp`).
#[derive(Debug, Clone)]
pub struct AdEqns {
    tr: AdTrans,
    alpha_grv: f64,
    alpha_gam: f64,
    alpha_pi: f64,
    alpha_omg: f64,
    x_atm: f64,
    coeff: Vec<AdCoeff>,
    pt: Vec<Point>,
}

impl AdEqns {
    /// `ad_eqns_t(cx, md_p, os_p)` (`ad_eqns_m.fypp:88-128`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] or invalid options.
    pub fn new(
        cx: &Context<'_>,
        md_p: &ModePar,
        os_p: &OscPar,
    ) -> Result<Self, GyreError> {
        let tr = AdTrans::new(md_p, os_p)?;
        let alpha_omg = match os_p.time_factor.as_str() {
            "OSC" => 1.0,
            "EXP" => -1.0,
            _ => return Err(GyreError::Abort("invalid time_factor")),
        };
        let x_atm = if os_p.x_atm < 0.0 {
            cx.pt_o.x
        } else {
            os_p.x_atm
        };
        Ok(Self {
            tr,
            alpha_grv: os_p.alpha_grv,
            alpha_gam: os_p.alpha_gam,
            alpha_pi: os_p.alpha_pi,
            alpha_omg,
            x_atm,
            coeff: Vec::new(),
            pt: Vec::new(),
        })
    }

    /// `stencil(pt)` (`ad_eqns_m.fypp:132-176`).
    ///
    /// # Errors
    ///
    /// Model evaluation errors; vacuum stencil point (`ASSERT`).
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
            self.coeff.push(AdCoeff {
                v: ml.coeff(idx::V_2, p)? * sq(p.x),
                a_s: ml.coeff(idx::AS, p)?,
                u: ml.coeff(idx::U, p)?,
                c_1: ml.coeff(idx::C_1, p)?,
                gamma_1: ml.coeff(idx::GAMMA_1, p)?,
            });
        }
        self.tr.stencil(cx, pt)?;
        self.pt = pt.to_vec();
        Ok(())
    }

    /// `xA(i, st)` (`ad_eqns_m.fypp:199-269`).
    fn xa(
        &self,
        cx: &Context<'_>,
        i: usize,
        omega: f64,
    ) -> Result<[f64; 16], GyreError> {
        let co = self.coeff[i];
        let pt = self.pt[i];
        let omega_rot = cx.omega_rot(pt)?;
        let omega_rot_i = cx.omega_rot(cx.pt_i)?;
        let omega_c = cx.omega_c_r(omega_rot, omega);
        let lambda = cx.lambda_r(omega_rot, omega);
        let l_i = cx.l_e_r(omega_rot_i, omega);
        let ca = co.c_1 * self.alpha_omg * sq(omega_c);
        let mut a = [0.0_f64; 16];
        a[ix(1, 1)] = co.v / co.gamma_1 - 1.0 - l_i;
        a[ix(1, 2)] = lambda / ca - co.v / co.gamma_1 * self.alpha_gam;
        a[ix(1, 3)] = self.alpha_grv * (lambda / ca);
        a[ix(1, 4)] = self.alpha_grv * 0.0;

        let as_fac = if co.a_s > 0.0 {
            if pt.x < self.x_atm {
                self.alpha_pi
            } else {
                self.alpha_gam
            }
        } else {
            1.0
        };
        a[ix(2, 1)] = ca - co.a_s * as_fac;
        a[ix(2, 2)] = co.a_s - co.u + 3.0 - l_i;
        a[ix(2, 3)] = self.alpha_grv * 0.0;
        a[ix(2, 4)] = -self.alpha_grv;

        a[ix(3, 1)] = self.alpha_grv * 0.0;
        a[ix(3, 2)] = self.alpha_grv * 0.0;
        a[ix(3, 3)] = self.alpha_grv * (3.0 - co.u - l_i);
        a[ix(3, 4)] = self.alpha_grv * 1.0;

        a[ix(4, 1)] = self.alpha_grv * (co.u * co.a_s);
        a[ix(4, 2)] = self.alpha_grv * (co.u * co.v / co.gamma_1);
        a[ix(4, 3)] = self.alpha_grv * lambda;
        a[ix(4, 4)] = self.alpha_grv * (-co.u - l_i + 2.0);
        self.tr.trans_eqns(cx, &a, i, omega)
    }

    /// `A(i, st) = xA(i, st)/pt(i)%x` (`ad_eqns_m.fypp:180-195`).
    ///
    /// # Errors
    ///
    /// Context and model errors.
    pub fn a(
        &self,
        cx: &Context<'_>,
        i: usize,
        omega: f64,
    ) -> Result<[f64; 16], GyreError> {
        let xa = self.xa(cx, i, omega)?;
        let x = self.pt[i].x;
        Ok(xa.map(|v| v / x))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdInnerBound {
    Regular,
    ZeroR,
    ZeroH,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdOuterBound {
    Vacuum,
    Jcd,
}

/// `ad_bound_t` (`ad/ad_bound_m.fypp`) with `REGULAR`/`ZERO_R`/`ZERO_H` inner
/// and `VACUUM`/`JCD` outer boundaries.
#[derive(Debug, Clone)]
pub struct AdBound {
    tr: AdTrans,
    alpha_grv: f64,
    alpha_gbc: f64,
    alpha_omg: f64,
    inner: AdInnerBound,
    outer: AdOuterBound,
    c_1_i: f64,
    atm_o: [f64; 5], // [V, As, U, c_1, Gamma_1]
    pt: [Point; 2],
}

impl AdBound {
    /// `n_i`, `n_o`, `n_e`.
    pub const N_I: usize = 2;
    /// See [`Self::N_I`].
    pub const N_O: usize = 2;
    /// See [`Self::N_I`].
    pub const N_E: usize = 4;

    /// `ad_bound_t(cx, md_p, os_p)` (`ad_bound_m.fypp:110-203`) and
    /// `stencil_` (`:207-277`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for unported boundary conditions or transforms;
    /// model evaluation errors.
    pub fn new(
        cx: &Context<'_>,
        md_p: &ModePar,
        os_p: &OscPar,
    ) -> Result<Self, GyreError> {
        let mut tr = AdTrans::new(md_p, os_p)?;
        let ml = cx.ml;
        let (pt_i, pt_o) = (cx.pt_i, cx.pt_o);
        let inner = match os_p.inner_bound.as_str() {
            "REGULAR" => {
                if pt_i.x != 0.0 {
                    return Err(GyreError::Assert(
                        "boundary condition invalid for x /= 0",
                    ));
                }
                AdInnerBound::Regular
            }
            "ZERO_R" => {
                if pt_i.x == 0.0 {
                    return Err(GyreError::Assert(
                        "boundary condition invalid for x == 0",
                    ));
                }
                AdInnerBound::ZeroR
            }
            "ZERO_H" => {
                if pt_i.x == 0.0 {
                    return Err(GyreError::Assert(
                        "boundary condition invalid for x == 0",
                    ));
                }
                AdInnerBound::ZeroH
            }
            _ => return Err(GyreError::Abort("invalid inner_bound")),
        };
        let outer = match os_p.outer_bound.as_str() {
            "VACUUM" => AdOuterBound::Vacuum,
            "JCD" => {
                if ml.is_vacuum(pt_o)? {
                    AdOuterBound::Vacuum
                } else {
                    AdOuterBound::Jcd
                }
            }
            "DZIEM" | "ISOTHERMAL" | "UNNO" | "GAMMA1" | "GAMMA2" => {
                if !ml.is_vacuum(pt_o)? {
                    return Err(GyreError::Unported {
                        what: "adiabatic outer_bound other than 'VACUUM' and 'JCD' at a non-vacuum outer point",
                        at: "ad/ad_bound_m.fypp:148-176",
                    });
                }
                AdOuterBound::Vacuum
            }
            _ => return Err(GyreError::Abort("invalid outer_bound")),
        };
        let alpha_omg = match os_p.time_factor.as_str() {
            "OSC" => 1.0,
            "EXP" => -1.0,
            _ => return Err(GyreError::Abort("invalid time_factor")),
        };
        ml.check(&[idx::V_2, idx::U, idx::C_1, idx::GAMMA_1])?;
        let c_1_i = if inner == AdInnerBound::Regular {
            ml.coeff(idx::C_1, pt_i)?
        } else {
            0.0
        };
        let atm_o = match outer {
            AdOuterBound::Vacuum => {
                [0.0, 0.0, ml.coeff(idx::U, pt_o)?, 0.0, 0.0]
            }
            AdOuterBound::Jcd => {
                let v = ml.coeff(idx::V_2, pt_o)? * sq(pt_o.x);
                let gamma_1 = ml.coeff(idx::GAMMA_1, pt_o)?;
                let a_s = ml.coeff(idx::V_2, pt_o)?
                    * sq(pt_o.x)
                    * (1.0 - 1.0 / gamma_1);
                let u = ml.coeff(idx::U, pt_o)?;
                let c_1 = ml.coeff(idx::C_1, pt_o)?;
                [v, a_s, u, c_1, gamma_1]
            }
        };
        tr.stencil(cx, &[pt_i, pt_o])?;
        Ok(Self {
            tr,
            alpha_grv: os_p.alpha_grv,
            alpha_gbc: os_p.alpha_gbc,
            alpha_omg,
            inner,
            outer,
            c_1_i,
            atm_o,
            pt: [pt_i, pt_o],
        })
    }

    /// `build_i(st, B, scl)` (`ad_bound_m.fypp:281-447`): `B(2, 4)` column-major.
    ///
    /// # Errors
    ///
    /// Context errors.
    pub fn build_i(
        &self,
        cx: &Context<'_>,
        omega: f64,
    ) -> Result<([f64; 8], [f64; 2]), GyreError> {
        let mut b = [0.0_f64; 8];
        let bx = |i: usize, j: usize| (i - 1) + 2 * (j - 1);
        let alpha_grv = self.alpha_grv;
        match self.inner {
            AdInnerBound::Regular => {
                let omega_rot = cx.omega_rot(self.pt[0])?;
                let omega_c = cx.omega_c_r(omega_rot, omega);
                let l_i = cx.l_e_r(omega_rot, omega);
                b[bx(1, 1)] = self.c_1_i * self.alpha_omg * sq(omega_c);
                b[bx(1, 2)] = -l_i;
                b[bx(1, 3)] = alpha_grv * (-l_i);
                b[bx(1, 4)] = alpha_grv * 0.0;
                b[bx(2, 1)] = alpha_grv * 0.0;
                b[bx(2, 2)] = alpha_grv * 0.0;
                b[bx(2, 3)] = alpha_grv * l_i;
                b[bx(2, 4)] = -alpha_grv + (1.0 - alpha_grv);
            }
            AdInnerBound::ZeroR => {
                b[bx(1, 1)] = 1.0;
                b[bx(1, 3)] = alpha_grv * 0.0;
                b[bx(1, 4)] = alpha_grv * 0.0;
                b[bx(2, 1)] = alpha_grv * 0.0;
                b[bx(2, 2)] = alpha_grv * 0.0;
                b[bx(2, 3)] = alpha_grv * 0.0;
                b[bx(2, 4)] = alpha_grv * 1.0 + (1.0 - alpha_grv);
            }
            AdInnerBound::ZeroH => {
                b[bx(1, 2)] = 1.0;
                b[bx(1, 3)] = alpha_grv * 1.0;
                b[bx(1, 4)] = alpha_grv * 0.0;
                b[bx(2, 1)] = alpha_grv * 0.0;
                b[bx(2, 2)] = alpha_grv * 0.0;
                b[bx(2, 3)] = alpha_grv * 0.0;
                b[bx(2, 4)] = alpha_grv * 1.0 + (1.0 - alpha_grv);
            }
        }
        let b = self.tr.trans_cond_2x4(cx, &b, 0, omega)?;
        Ok((b, [1.0, 1.0]))
    }

    /// `build_o(st, B, scl)` (`ad_bound_m.fypp:451-538, 715-780`): `B(2, 4)` column-major.
    ///
    /// # Errors
    ///
    /// Context errors.
    pub fn build_o(
        &self,
        cx: &Context<'_>,
        omega: f64,
    ) -> Result<([f64; 8], [f64; 2]), GyreError> {
        let mut b = [0.0_f64; 8];
        let bx = |i: usize, j: usize| (i - 1) + 2 * (j - 1);
        let [v, a_s, u, c_1, gamma_1] = self.atm_o;
        let omega_rot = cx.omega_rot(self.pt[1])?;
        let l_e = cx.l_e_r(omega_rot, omega);
        match self.outer {
            AdOuterBound::Vacuum => {
                b[bx(1, 1)] = 1.0;
                b[bx(1, 2)] = -1.0;
                b[bx(1, 3)] = self.alpha_grv * 0.0;
                b[bx(1, 4)] = self.alpha_grv * 0.0;
                b[bx(2, 1)] = self.alpha_grv * self.alpha_gbc * u;
                b[bx(2, 2)] = self.alpha_grv * 0.0;
                b[bx(2, 3)] =
                    self.alpha_grv * (l_e + 1.0) + (1.0 - self.alpha_grv);
                b[bx(2, 4)] = self.alpha_grv * 1.0;
            }
            AdOuterBound::Jcd => {
                let omega_c = cx.omega_c_r(omega_rot, omega);
                let lambda = cx.lambda_r(omega_rot, omega);
                let chi = atmos_chi_r(
                    v,
                    a_s,
                    c_1,
                    gamma_1,
                    self.alpha_omg,
                    omega_c,
                    lambda,
                );
                let ca = c_1 * self.alpha_omg * sq(omega_c);
                let b_11 = v / gamma_1 - 3.0;
                let b_12 = lambda / ca - v / gamma_1;
                b[bx(1, 1)] = chi - b_11;
                b[bx(1, 2)] = -b_12;
                b[bx(1, 3)] = self.alpha_grv
                    * ((lambda / ca - l_e - 1.0) * b_12 / (v / gamma_1 + a_s));
                b[bx(1, 4)] = self.alpha_grv * 0.0;
                b[bx(2, 1)] = self.alpha_grv * self.alpha_gbc * u;
                b[bx(2, 2)] = self.alpha_grv * 0.0;
                b[bx(2, 3)] =
                    self.alpha_grv * (l_e + 1.0) + (1.0 - self.alpha_grv);
                b[bx(2, 4)] = self.alpha_grv * 1.0;
            }
        }
        let b = self.tr.trans_cond_2x4(cx, &b, 1, omega)?;
        Ok((b, [1.0, 1.0]))
    }
}

/// `ad_match_t` (`ad/ad_match_m.fypp`).
#[derive(Debug, Clone)]
pub struct AdMatch {
    tr: AdTrans,
    u_l: f64,
    u_r: f64,
}

impl AdMatch {
    /// `ad_match_t(cx, pt_a, pt_b, md_p, os_p)` (`ad_match_m.fypp:80-136`).
    ///
    /// # Errors
    ///
    /// Model or transform errors.
    pub fn new(
        cx: &Context<'_>,
        pt_a: Point,
        pt_b: Point,
        md_p: &ModePar,
        os_p: &OscPar,
    ) -> Result<Self, GyreError> {
        let ml = cx.ml;
        ml.check(&[idx::U])?;
        let u_l = ml.coeff(idx::U, pt_a)?;
        let u_r = ml.coeff(idx::U, pt_b)?;
        let mut tr = AdTrans::new(md_p, os_p)?;
        tr.stencil(cx, &[pt_a, pt_b])?;
        Ok(Self { tr, u_l, u_r })
    }

    /// `build(st, E_l, E_r, scl)` (`ad_match_m.fypp:140-220`).
    ///
    /// # Errors
    ///
    /// Context rotation or transform errors.
    pub fn build(
        &self,
        cx: &Context<'_>,
        omega: f64,
    ) -> Result<([f64; 16], [f64; 16], ExtR), GyreError> {
        let mut e_l = [0.0_f64; 16];
        let mut e_r = [0.0_f64; 16];
        e_l[ix(1, 1)] = -1.0;
        e_l[ix(2, 1)] = self.u_l;
        e_l[ix(2, 2)] = -self.u_l;
        e_l[ix(3, 3)] = -1.0;
        e_l[ix(4, 1)] = -self.u_l;
        e_l[ix(4, 4)] = -1.0;

        e_r[ix(1, 1)] = 1.0;
        e_r[ix(2, 1)] = -self.u_r;
        e_r[ix(2, 2)] = self.u_r;
        e_r[ix(3, 3)] = 1.0;
        e_r[ix(4, 1)] = self.u_r;
        e_r[ix(4, 4)] = 1.0;
        let e_l = self.tr.trans_cond_4x4(cx, &e_l, 0, omega)?;
        let e_r = self.tr.trans_cond_4x4(cx, &e_r, 1, omega)?;
        Ok((e_l, e_r, ExtR::from_real(1.0)))
    }
}

#[derive(Debug, Clone)]
enum AdDiffKind {
    Colloc {
        eq: AdEqns,
        dx: f64,
        reverse: bool,
        gl2: bool,
    },
    MagnusGl2 {
        eq: AdEqns,
        dx: f64,
    },
    Match(AdMatch),
}

/// `ad_diff_t` (`ad/ad_diff_m.fypp`): `COLLOC_GL2`, `COLLOC_GL4`,
/// `MAGNUS_GL2`, and segment `ad_match_t`.
#[derive(Debug, Clone)]
pub struct AdDiff {
    kind: AdDiffKind,
}

impl AdDiff {
    /// `ad_diff_t(cx, pt_a, pt_b, md_p, nm_p, os_p)`.
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for remaining schemes; stencil errors.
    pub fn new(
        cx: &Context<'_>,
        pt_a: Point,
        pt_b: Point,
        md_p: &ModePar,
        nm_p: &NumPar,
        os_p: &OscPar,
    ) -> Result<Self, GyreError> {
        if pt_a.s != pt_b.s {
            return Ok(Self {
                kind: AdDiffKind::Match(AdMatch::new(
                    cx, pt_a, pt_b, md_p, os_p,
                )?),
            });
        }
        let mut eq = AdEqns::new(cx, md_p, os_p)?;
        match diff_kind(&nm_p.diff_scheme)? {
            DiffKind::CollocGl2 => {
                let (reverse, dx, pt) = colloc_stencil(pt_a, pt_b, true);
                eq.stencil(cx, &pt)?;
                Ok(Self {
                    kind: AdDiffKind::Colloc {
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
                    kind: AdDiffKind::Colloc {
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
                    kind: AdDiffKind::MagnusGl2 { eq, dx },
                })
            }
        }
    }

    /// `build(st, E_l, E_r, scl)` (`ad_diff_m.fypp:110-132`).
    ///
    /// # Errors
    ///
    /// Context, LAPACK, and eigendecomposition errors.
    pub fn build(
        &self,
        cx: &Context<'_>,
        omega: f64,
    ) -> Result<([f64; 16], [f64; 16], ExtR), GyreError> {
        let (e_l, e_r, scl) = match &self.kind {
            AdDiffKind::Match(mt) => return mt.build(cx, omega),
            AdDiffKind::MagnusGl2 { eq, dx } => {
                let mut domega = eq.a(cx, 0, omega)?;
                magnus_build_r(N_E, *dx, &mut domega)?
            }
            AdDiffKind::Colloc {
                eq,
                dx,
                reverse,
                gl2: true,
            } => {
                let a = eq.a(cx, 0, omega)?;
                colloc_gl2_build::<N_E>(*dx, *reverse, &a)
            }
            AdDiffKind::Colloc {
                eq,
                dx,
                reverse,
                gl2: false,
            } => {
                let a0 = eq.a(cx, 0, omega)?;
                let a1 = eq.a(cx, 1, omega)?;
                colloc_gl4_build::<N_E>(*dx, *reverse, [&a0, &a1])?
            }
        };
        Ok((
            std::array::from_fn(|k| e_l[k]),
            std::array::from_fn(|k| e_r[k]),
            scl,
        ))
    }
}

/// `ad_bvp_t` (`ad/ad_bvp_m.fypp:84-139`) over `bvp_rt` (`bvp_m.fypp`).
#[derive(Debug, Clone)]
pub struct AdBvp<'a> {
    /// The context.
    pub cx: &'a Context<'a>,
    /// The grid.
    pub gr: Grid,
    bd: AdBound,
    df: Vec<AdDiff>,
    /// Variable transform on the full grid.
    pub tr: AdTrans,
    /// The system matrix.
    pub sm: BlockSysmtx<f64>,
    factored: bool,
    /// `md_p`.
    pub md_p: ModePar,
    /// `nm_p`.
    pub nm_p: NumPar,
    /// `os_p`.
    pub os_p: OscPar,
}

impl<'a> AdBvp<'a> {
    /// `ad_bvp_t(cx, gr, md_p, nm_p, os_p)`.
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for `quasiad_eigfuncs` or unported options;
    /// component construction errors.
    pub fn new(
        cx: &'a Context<'a>,
        gr: &Grid,
        md_p: &ModePar,
        nm_p: &NumPar,
        os_p: &OscPar,
    ) -> Result<Self, GyreError> {
        let bd = AdBound::new(cx, md_p, os_p)?;
        let mut df = Vec::with_capacity(gr.n() - 1);
        for j in 0..gr.n() - 1 {
            df.push(AdDiff::new(cx, gr.pt[j], gr.pt[j + 1], md_p, nm_p, os_p)?);
        }
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
        let sm = BlockSysmtx::new(gr.n() - 1, N_E, AdBound::N_I, AdBound::N_O);
        let mut tr = AdTrans::new(md_p, os_p)?;
        tr.stencil(cx, &gr.pt)?;
        if os_p.quasiad_eigfuncs {
            return Err(GyreError::Unported {
                what: "quasiad_eigfuncs",
                at: "ad/ad_bvp_m.fypp:125-129",
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
    /// Component build errors.
    pub fn build(
        &mut self,
        omega: f64,
        mode_build: bool,
        tap: &mut dyn GyreTap,
    ) -> Result<(), GyreError> {
        let (b_i, scl_i) = self.bd.build_i(self.cx, omega)?;
        self.sm.set_b_i(&b_i, &scl_i);
        let (b_o, scl_o) = self.bd.build_o(self.cx, omega)?;
        self.sm.set_b_o(&b_o, &scl_o);
        tap.build_ad(self.n(), mode_build, omega, &b_i, &b_o);
        for j in 0..self.n() - 1 {
            let (e_l, e_r, scl) = self.df[j].build(self.cx, omega)?;
            self.sm.set_e(j, &e_l, &e_r, scl);
            tap.block_ad(j + 1, &e_l, &e_r, scl);
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

    /// `soln_vec_hom()` (`bvp_m.fypp:214-233`): `y(4, n)` column-major.
    ///
    /// # Errors
    ///
    /// LAPACK errors.
    pub fn soln_vec_hom(&self) -> Result<Vec<f64>, GyreError> {
        self.sm.soln_vec_hom()
    }
}

/// `wave_t(bp, st, id)` for the adiabatic BVP (`wave_t_hom_` and
/// `wave_t_y_`, `ad/ad_bvp_m.fypp:143-260`).
///
/// # Errors
///
/// BVP and wave construction errors.
pub fn ad_wave_hom<'a>(
    bp: &mut AdBvp<'a>,
    omega: f64,
    id: i32,
    tap: &mut dyn GyreTap,
) -> Result<Wave<'a>, GyreError> {
    bp.build(omega, true, tap)?;
    bp.factor()?;
    let y = bp.soln_vec_hom()?;
    let n = bp.n();
    let mut y_c = vec![Cplx::from_real(0.0); 6 * n];
    for j in 0..n {
        let yj = bp.tr.trans_vars_to(
            bp.cx,
            [y[4 * j], y[4 * j + 1], y[4 * j + 2], y[4 * j + 3]],
            j,
            omega,
        )?;
        for r in 0..4 {
            y_c[6 * j + r] = Cplx::from_real(yj[r]);
        }
    }
    let discrim = ExtC::from_extr(bp.det()?);
    Wave::new(
        Cplx::from_real(omega),
        y_c,
        discrim,
        bp.cx,
        &bp.gr,
        &bp.md_p,
        &bp.nm_p,
        &bp.os_p,
        id,
    )
}
