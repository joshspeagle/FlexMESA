// SPDX-License-Identifier: GPL-3.0-only
//! GYRE's nonadiabatic problem (`nad/*_m.fypp`): variable transforms
//! (`nad_trans_m`), oscillation equations (`nad_eqns_m`), boundary
//! conditions (`nad_bound_m`), difference equations (`nad_diff_m` over the
//! complex collocation of `diff/colloc_diff_m.fypp`), and the BVP
//! (`nad_bvp_m` over `bvp_ct`).
//!
//! Complex expressions follow the Fortran parse with gfortran's semantics:
//! a real operand of a complex operation is widened to `(r, +0)` and the
//! full complex operation is kept (`crate::fcomplex`, checked against the
//! `-fdump-tree-optimized` output of GYRE's own build of these modules).
//! All `6 x 6` matrices are column-major (`a[(i-1) + 6*(j-1)]`).

use crate::context::Context;
use crate::crmath::sqrt_c;
use crate::ext_complex::ExtC;
use crate::fcomplex::Cplx;
use crate::grid::{Grid, Point};
use crate::magnus::{magnus_build_c, magnus_stencil};
use crate::mode::Wave;
use crate::model::idx;
use crate::par::{ModePar, NumPar, OscPar};
use crate::rad::{colloc_stencil, diff_kind, DiffKind, GyreTap};
use crate::sysmtx::BlockSysmtx;
use crate::GyreError;

const N_E: usize = 6;

/// `(r, +0)`: a real operand widened to complex.
fn c(r: f64) -> Cplx {
    Cplx::from_real(r)
}

fn ix(i: usize, j: usize) -> usize {
    (i - 1) + N_E * (j - 1)
}

/// `nad_trans_t` (`nad_trans_m.fypp`) for the `GYRE` variable set
/// (`SET`: no transformation).
#[derive(Debug, Clone)]
pub struct NadTrans;

impl NadTrans {
    /// `nad_trans_t(cx, md_p, os_p)` (`nad_trans_m.fypp:102-135`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for variable sets other than `GYRE`.
    pub fn new(os_p: &OscPar) -> Result<Self, GyreError> {
        match os_p.variables_set.as_str() {
            "GYRE" => Ok(Self),
            "DZIEM" | "JCD" | "LAGP" => Err(GyreError::Unported {
                what: "nonadiabatic variables_set other than 'GYRE'",
                at: "nad/nad_trans_m.fypp:116-121",
            }),
            _ => Err(GyreError::Abort("invalid variables_set")),
        }
    }

    /// `stencil(pt)` (`nad_trans_m.fypp:139-182`). With `SET` the
    /// coefficients are never used; they are still evaluated, as upstream
    /// does, so that evaluation errors surface.
    ///
    /// # Errors
    ///
    /// Model evaluation errors.
    pub fn stencil(
        &self,
        cx: &Context<'_>,
        pt: &[Point],
    ) -> Result<(), GyreError> {
        let ml = cx.ml;
        ml.check(&[idx::V_2, idx::U, idx::C_1])?;
        for &p in pt {
            if !ml.is_vacuum(p)? {
                let _ = ml.coeff(idx::V_2, p)?;
                let _ = ml.dcoeff(idx::V_2, p)?;
                let _ = ml.dcoeff(idx::U, p)?;
            }
            let _ = ml.coeff(idx::U, p)?;
            let _ = ml.coeff(idx::C_1, p)?;
            let _ = ml.dcoeff(idx::C_1, p)?;
        }
        Ok(())
    }
}

/// Stencilled coefficients of `nad_eqns_t` (`J_V .. J_KAP_T`).
#[derive(Debug, Clone, Copy)]
struct NadCoeff {
    v: f64,
    a_s: f64,
    u: f64,
    c_1: f64,
    gamma_1: f64,
    ups_t: f64,
    nabla_ad: f64,
    dnabla_ad: f64,
    nabla: f64,
    c_rad: f64,
    dc_rad: f64,
    c_thn: f64,
    dc_thn: f64,
    c_thk: f64,
    c_eps: f64,
    c_egv: f64,
    kap_rho: f64,
    kap_t: f64,
}

/// `nad_eqns_t` (`nad_eqns_m.fypp`).
#[derive(Debug, Clone)]
pub struct NadEqns {
    tr: NadTrans,
    alpha_grv: f64,
    alpha_thm: f64,
    alpha_hfl: f64,
    alpha_gam: f64,
    alpha_pi: f64,
    alpha_kar: f64,
    alpha_kat: f64,
    alpha_rht: f64,
    alpha_egv: f64,
    alpha_omg: f64,
    x_atm: f64,
    coeff: Vec<NadCoeff>,
    pt: Vec<Point>,
}

impl NadEqns {
    /// `nad_eqns_t(cx, md_p, os_p)` (`nad_eqns_m.fypp:115-170`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for `time_factor = 'EXP'` and
    /// `conv_scheme = 'FROZEN_PESNELL_4'`; invalid options.
    pub fn new(cx: &Context<'_>, os_p: &OscPar) -> Result<Self, GyreError> {
        let tr = NadTrans::new(os_p)?;
        let alpha_omg = match os_p.time_factor.as_str() {
            "OSC" => 1.0,
            "EXP" => {
                return Err(GyreError::Unported {
                    what: "time_factor = 'EXP'",
                    at: "nad/nad_eqns_m.fypp:143-144",
                })
            }
            _ => return Err(GyreError::Abort("invalid time_factor")),
        };
        match os_p.conv_scheme.as_str() {
            "FROZEN_PESNELL_1" => {}
            "FROZEN_PESNELL_4" => {
                return Err(GyreError::Unported {
                    what: "conv_scheme = 'FROZEN_PESNELL_4'",
                    at: "nad/nad_eqns_m.fypp:152-153",
                })
            }
            _ => return Err(GyreError::Abort("invalid conv_scheme")),
        }
        let x_atm = if os_p.x_atm < 0.0 {
            cx.pt_o.x
        } else {
            os_p.x_atm
        };
        Ok(Self {
            tr,
            alpha_grv: os_p.alpha_grv,
            alpha_thm: os_p.alpha_thm,
            alpha_hfl: os_p.alpha_hfl,
            alpha_gam: os_p.alpha_gam,
            alpha_pi: os_p.alpha_pi,
            alpha_kar: os_p.alpha_kar,
            alpha_kat: os_p.alpha_kat,
            alpha_rht: os_p.alpha_rht,
            alpha_egv: os_p.alpha_egv,
            alpha_omg,
            x_atm,
            coeff: Vec::new(),
            pt: Vec::new(),
        })
    }

    /// `stencil(pt)` (`nad_eqns_m.fypp:174-236`).
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
        ml.check(&[
            idx::V_2,
            idx::AS,
            idx::U,
            idx::C_1,
            idx::GAMMA_1,
            idx::NABLA,
            idx::NABLA_AD,
            idx::UPS_T,
            idx::C_LUM,
            idx::C_RAD,
            idx::C_THN,
            idx::C_THK,
            idx::C_EPS,
            idx::C_EGV,
            idx::KAP_RHO,
            idx::KAP_T,
        ])?;
        self.coeff = Vec::with_capacity(pt.len());
        for &p in pt {
            if ml.is_vacuum(p)? {
                return Err(GyreError::Assert(
                    "attempt to stencil at vacuum point",
                ));
            }
            let v = ml.coeff(idx::V_2, p)? * (p.x * p.x);
            let co = NadCoeff {
                v,
                a_s: ml.coeff(idx::AS, p)?,
                u: ml.coeff(idx::U, p)?,
                c_1: ml.coeff(idx::C_1, p)?,
                gamma_1: ml.coeff(idx::GAMMA_1, p)?,
                nabla_ad: ml.coeff(idx::NABLA_AD, p)?,
                dnabla_ad: ml.dcoeff(idx::NABLA_AD, p)?,
                nabla: ml.coeff(idx::NABLA, p)?,
                ups_t: ml.coeff(idx::UPS_T, p)?,
                c_rad: 0.0,
                dc_rad: 0.0,
                c_thn: 0.0,
                dc_thn: 0.0,
                c_thk: 0.0,
                c_eps: 0.0,
                c_egv: 0.0,
                kap_rho: 0.0,
                kap_t: 0.0,
            };
            // J_C_LUM, J_DC_LUM are stencilled but unused by xA.
            let _ = ml.coeff(idx::C_LUM, p)?;
            let _ = ml.dcoeff(idx::C_LUM, p)?;
            self.coeff.push(NadCoeff {
                c_rad: ml.coeff(idx::C_RAD, p)?,
                dc_rad: ml.dcoeff(idx::C_RAD, p)?,
                c_thn: ml.coeff(idx::C_THN, p)?,
                dc_thn: ml.dcoeff(idx::C_THN, p)?,
                c_thk: ml.coeff(idx::C_THK, p)?,
                c_eps: ml.coeff(idx::C_EPS, p)?,
                c_egv: ml.coeff(idx::C_EGV, p)?,
                kap_rho: ml.coeff(idx::KAP_RHO, p)?,
                kap_t: ml.coeff(idx::KAP_T, p)?,
                ..co
            });
        }
        self.tr.stencil(cx, pt)?;
        self.pt = pt.to_vec();
        Ok(())
    }

    /// `A(i, st) = xA(i, st)/pt(i)%x` (`nad_eqns_m.fypp:240-255`), `i`
    /// 0-based.
    ///
    /// # Errors
    ///
    /// Context and model errors.
    pub fn a(
        &self,
        cx: &Context<'_>,
        i: usize,
        omega: Cplx,
    ) -> Result<[Cplx; 36], GyreError> {
        let xa = self.xa(cx, i, omega)?;
        let x = self.pt[i].x;
        Ok(xa.map(|z| z.div_real(x)))
    }

    /// `xA(i, st)` (`nad_eqns_m.fypp:259-411`).
    #[allow(clippy::too_many_lines)]
    fn xa(
        &self,
        cx: &Context<'_>,
        i: usize,
        omega: Cplx,
    ) -> Result<[Cplx; 36], GyreError> {
        let co = self.coeff[i];
        let pt = self.pt[i];
        let x = pt.x;
        let NadCoeff {
            v,
            a_s,
            u,
            c_1,
            gamma_1,
            ups_t,
            nabla_ad,
            dnabla_ad,
            nabla,
            c_rad,
            dc_rad,
            c_thn,
            dc_thn,
            c_thk,
            c_eps,
            c_egv,
            kap_rho,
            kap_t,
        } = co;
        let (alpha_grv, alpha_thm, alpha_hfl, alpha_rht, alpha_omg) = (
            self.alpha_grv,
            self.alpha_thm,
            self.alpha_hfl,
            self.alpha_rht,
            self.alpha_omg,
        );
        let (alpha_gam, alpha_pi, alpha_kar, alpha_kat, alpha_egv) = (
            self.alpha_gam,
            self.alpha_pi,
            self.alpha_kar,
            self.alpha_kat,
            self.alpha_egv,
        );

        let omega_rot = cx.omega_rot(pt)?;
        let omega_rot_i = cx.omega_rot(cx.pt_i)?;
        let omega_c = cx.omega_c_c(omega_rot, omega);
        let i_omega_c = Cplx::new(0.0, 1.0) * sqrt_c(c(alpha_omg)) * omega_c;
        let lambda = cx.lambda_c(omega_rot)?;
        let l_i = cx.l_e_c(omega_rot_i)?;
        let oc2 = omega_c.powi2();
        // c_1*alpha_omg*omega_c**2, written out each time upstream
        let c1ao2 = c(c_1 * alpha_omg) * oc2;

        let f_rh = c(1.0) - c(0.25 * alpha_rht) * i_omega_c * c(c_thn);
        let df_rh =
            -(c(0.25 * alpha_rht) * i_omega_c * c(c_thn) * c(dc_thn) / f_rh);
        // P1_CONV_SCHEME
        let conv_term = lambda * c(c_rad) * c(3.0 + dc_rad) / c1ao2;

        let eps_rho = cx.eps_rho_c(pt)?;
        let eps_t = cx.eps_t_c(pt)?;
        let c_eps_ad = c(c_eps) * (c(nabla_ad) * eps_t + eps_rho / c(gamma_1));
        let c_eps_s = c(c_eps) * (eps_t - c(ups_t) * eps_rho);
        let c_kap_ad =
            nabla_ad * alpha_kat * kap_t + alpha_kar * kap_rho / gamma_1;
        let c_kap_s = alpha_kat * kap_t - ups_t * alpha_kar * kap_rho;
        let c_dif = -4.0 * nabla_ad * v * nabla + nabla_ad * (dnabla_ad + v);

        let mut a = [c(0.0); 36];
        a[ix(1, 1)] = c(v / gamma_1 - 1.0) - l_i;
        a[ix(1, 2)] = lambda / c1ao2 - c(v / gamma_1 * alpha_gam);
        a[ix(1, 3)] = c(alpha_grv) * (lambda / c1ao2);
        a[ix(1, 4)] = c(alpha_grv * 0.0);
        a[ix(1, 5)] = c(ups_t);
        a[ix(1, 6)] = c(0.0);

        let as_fac = if a_s > 0.0 {
            if x < self.x_atm {
                alpha_pi
            } else {
                alpha_gam
            }
        } else {
            1.0
        };
        a[ix(2, 1)] = c1ao2 - c(a_s * as_fac);
        a[ix(2, 2)] = c(a_s - u + 3.0) - l_i;
        a[ix(2, 3)] = c(alpha_grv * 0.0);
        // alpha_grv*(-1._RD), exactly -alpha_grv
        a[ix(2, 4)] = c(-alpha_grv);
        a[ix(2, 5)] = c(ups_t);
        a[ix(2, 6)] = c(0.0);

        a[ix(3, 1)] = c(alpha_grv * 0.0);
        a[ix(3, 2)] = c(alpha_grv * 0.0);
        a[ix(3, 3)] = c(alpha_grv) * (c(3.0 - u) - l_i);
        a[ix(3, 4)] = c(alpha_grv * 1.0);
        a[ix(3, 5)] = c(alpha_grv * 0.0);
        a[ix(3, 6)] = c(alpha_grv * 0.0);

        a[ix(4, 1)] = c(alpha_grv * (u * a_s));
        a[ix(4, 2)] = c(alpha_grv * (u * v / gamma_1));
        a[ix(4, 3)] = c(alpha_grv) * lambda;
        a[ix(4, 4)] = c(alpha_grv) * (c(-u) - l_i + c(2.0));
        a[ix(4, 5)] = c(alpha_grv * (-u * ups_t));
        a[ix(4, 6)] = c(alpha_grv * 0.0);

        a[ix(5, 1)] = c(v)
            * (c(nabla_ad) * (c(u) - c1ao2) - c(4.0 * (nabla_ad - nabla))
                + c(c_kap_ad * v * nabla)
                + c(c_dif))
            / f_rh;
        a[ix(5, 2)] = c(v)
            * (lambda / c1ao2 * c(nabla_ad - nabla)
                - c(c_kap_ad * v * nabla)
                - c(c_dif))
            / f_rh;
        a[ix(5, 3)] =
            c(alpha_grv) * (c(v) * lambda / c1ao2 * c(nabla_ad - nabla)) / f_rh;
        a[ix(5, 4)] = c(alpha_grv * (v * nabla_ad)) / f_rh;
        a[ix(5, 5)] = c(v * nabla) * (c(4.0) * f_rh - c(c_kap_s)) / f_rh
            - df_rh
            - (l_i - c(2.0));
        a[ix(5, 6)] = -(c(v * nabla) / (c(c_rad) * f_rh));

        a[ix(6, 1)] =
            c(alpha_hfl) * lambda * c(nabla_ad / nabla - 1.0) * c(c_rad)
                - c(v) * c_eps_ad
                - c(alpha_egv * c_egv * nabla_ad * v);
        a[ix(6, 2)] = c(v) * c_eps_ad
            - lambda * c(c_rad) * c(alpha_hfl) * c(nabla_ad) / c(nabla)
            + conv_term
            + c(alpha_egv * c_egv * nabla_ad * v);
        a[ix(6, 3)] = c(alpha_grv) * conv_term;
        a[ix(6, 4)] = c(alpha_grv * 0.0);
        a[ix(6, 5)] = if x > 0.0 {
            c_eps_s - c(alpha_hfl) * lambda * c(c_rad) / c(nabla * v)
                + c(alpha_thm) * i_omega_c * c(c_thk)
                + c(alpha_egv * c_egv)
        } else {
            c(-(alpha_hfl * f64::MAX))
        };
        a[ix(6, 6)] = c(-1.0) - l_i;
        // trans_eqns: no transformation for SET
        Ok(a)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NadInnerBound {
    Regular,
    ZeroR,
    ZeroH,
}

/// `nad_bound_t` (`nad_bound_m.fypp`) with `REGULAR`/`ZERO_R`/`ZERO_H` inner
/// and `VACUUM` outer boundaries.
#[derive(Debug, Clone)]
pub struct NadBound {
    alpha_grv: f64,
    alpha_gbc: f64,
    alpha_rht: f64,
    alpha_omg: f64,
    inner: NadInnerBound,
    /// `coeff(1, J_C_1)`.
    c_1_i: f64,
    /// `coeff(2, [J_V, J_U, J_NABLA_AD, J_C_THN])`.
    coeff_o: [f64; 4],
    pt: [Point; 2],
}

impl NadBound {
    /// `n_i`.
    pub const N_I: usize = 3;
    /// `n_o`.
    pub const N_O: usize = 3;

    /// `nad_bound_t(cx, md_p, os_p)` (`nad_bound_m.fypp:112-204`) and
    /// `stencil_` (`:208-276`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for non-vacuum outer boundaries and
    /// `time_factor = 'EXP'`; model evaluation errors.
    pub fn new(cx: &Context<'_>, os_p: &OscPar) -> Result<Self, GyreError> {
        let tr = NadTrans::new(os_p)?;
        let ml = cx.ml;
        let (pt_i, pt_o) = (cx.pt_i, cx.pt_o);
        let inner = match os_p.inner_bound.as_str() {
            "REGULAR" => {
                if pt_i.x != 0.0 {
                    return Err(GyreError::Assert(
                        "boundary condition invalid for x /= 0",
                    ));
                }
                NadInnerBound::Regular
            }
            "ZERO_R" => {
                if pt_i.x == 0.0 {
                    return Err(GyreError::Assert(
                        "boundary condition invalid for x == 0",
                    ));
                }
                NadInnerBound::ZeroR
            }
            "ZERO_H" => {
                if pt_i.x == 0.0 {
                    return Err(GyreError::Assert(
                        "boundary condition invalid for x == 0",
                    ));
                }
                NadInnerBound::ZeroH
            }
            _ => return Err(GyreError::Abort("invalid inner_bound")),
        };
        // nad_bound_m.fypp:148-180: every type falls back to VACUUM_TYPE
        // at a vacuum outer point.
        match os_p.outer_bound.as_str() {
            "VACUUM" => {}
            "DZIEM" | "ISOTHERMAL" | "UNNO" | "JCD" => {
                if !ml.is_vacuum(pt_o)? {
                    return Err(GyreError::Unported {
                        what: "nonadiabatic outer_bound other than 'VACUUM' at a non-vacuum outer point",
                        at: "nad/nad_bound_m.fypp:150-180",
                    });
                }
            }
            _ => return Err(GyreError::Abort("invalid outer_bound")),
        }
        let alpha_omg = match os_p.time_factor.as_str() {
            "OSC" => 1.0,
            "EXP" => {
                return Err(GyreError::Unported {
                    what: "time_factor = 'EXP'",
                    at: "nad/nad_bound_m.fypp:185-186",
                })
            }
            _ => return Err(GyreError::Abort("invalid time_factor")),
        };
        // stencil_
        ml.check(&[
            idx::V_2,
            idx::U,
            idx::C_1,
            idx::GAMMA_1,
            idx::NABLA_AD,
            idx::C_THN,
        ])?;
        let c_1_i = if inner == NadInnerBound::Regular {
            ml.coeff(idx::C_1, pt_i)?
        } else {
            0.0
        };
        let v = ml.coeff(idx::V_2, pt_o)? * (pt_o.x * pt_o.x);
        let u = ml.coeff(idx::U, pt_o)?;
        let nabla_ad = ml.coeff(idx::NABLA_AD, pt_o)?;
        let c_thn = ml.coeff(idx::C_THN, pt_o)?;
        tr.stencil(cx, &[pt_i, pt_o])?;
        Ok(Self {
            alpha_grv: os_p.alpha_grv,
            alpha_gbc: os_p.alpha_gbc,
            alpha_rht: os_p.alpha_rht,
            alpha_omg,
            inner,
            c_1_i,
            coeff_o: [v, u, nabla_ad, c_thn],
            pt: [pt_i, pt_o],
        })
    }

    /// `build_i(st, B, scl)` (`nad_bound_m.fypp:280-479`):
    /// `B(3, 6)` column-major and `scl(3)`.
    ///
    /// # Errors
    ///
    /// Context errors.
    pub fn build_i(
        &self,
        cx: &Context<'_>,
        omega: Cplx,
    ) -> Result<([Cplx; 18], [Cplx; 3]), GyreError> {
        let (c_1, alpha_grv, alpha_omg) =
            (self.c_1_i, self.alpha_grv, self.alpha_omg);
        let mut b = [c(0.0); 18];
        let bx = |i: usize, j: usize| (i - 1) + 3 * (j - 1);
        match self.inner {
            NadInnerBound::Regular => {
                let omega_rot = cx.omega_rot(self.pt[0])?;
                let omega_c = cx.omega_c_c(omega_rot, omega);
                let l_i = cx.l_e_c(omega_rot)?;
                b[bx(1, 1)] = c(c_1 * alpha_omg) * omega_c.powi2();
                b[bx(1, 2)] = -l_i;
                b[bx(1, 3)] = c(alpha_grv) * (-l_i);
                b[bx(1, 4)] = c(alpha_grv * 0.0);
                b[bx(1, 5)] = c(0.0);
                b[bx(1, 6)] = c(0.0);
                b[bx(2, 1)] = c(alpha_grv * 0.0);
                b[bx(2, 2)] = c(alpha_grv * 0.0);
                b[bx(2, 3)] = c(alpha_grv) * l_i;
                b[bx(2, 4)] = c(-alpha_grv + (1.0 - alpha_grv));
                b[bx(2, 5)] = c(alpha_grv * 0.0);
                b[bx(2, 6)] = c(alpha_grv * 0.0);
                b[bx(3, 1)] = c(0.0);
                b[bx(3, 2)] = c(0.0);
                b[bx(3, 3)] = c(alpha_grv * 0.0);
                b[bx(3, 4)] = c(alpha_grv * 0.0);
                b[bx(3, 5)] = c(1.0);
                b[bx(3, 6)] = c(0.0);
            }
            NadInnerBound::ZeroR => {
                b[bx(1, 1)] = c(1.0);
                b[bx(1, 2)] = c(0.0);
                b[bx(1, 3)] = c(alpha_grv * 0.0);
                b[bx(1, 4)] = c(alpha_grv * 0.0);
                b[bx(1, 5)] = c(0.0);
                b[bx(1, 6)] = c(0.0);
                b[bx(2, 1)] = c(alpha_grv * 0.0);
                b[bx(2, 2)] = c(alpha_grv * 0.0);
                b[bx(2, 3)] = c(alpha_grv * 0.0);
                b[bx(2, 4)] = c(alpha_grv * 1.0 + (1.0 - alpha_grv));
                b[bx(2, 5)] = c(alpha_grv * 0.0);
                b[bx(2, 6)] = c(alpha_grv * 0.0);
                b[bx(3, 1)] = c(0.0);
                b[bx(3, 2)] = c(0.0);
                b[bx(3, 3)] = c(alpha_grv * 0.0);
                b[bx(3, 4)] = c(alpha_grv * 0.0);
                b[bx(3, 5)] = c(1.0);
                b[bx(3, 6)] = c(0.0);
            }
            NadInnerBound::ZeroH => {
                b[bx(1, 1)] = c(0.0);
                b[bx(1, 2)] = c(1.0);
                b[bx(1, 3)] = c(alpha_grv * 1.0);
                b[bx(1, 4)] = c(alpha_grv * 0.0);
                b[bx(1, 5)] = c(0.0);
                b[bx(1, 6)] = c(0.0);
                b[bx(2, 1)] = c(alpha_grv * 0.0);
                b[bx(2, 2)] = c(alpha_grv * 0.0);
                b[bx(2, 3)] = c(alpha_grv * 0.0);
                b[bx(2, 4)] = c(alpha_grv * 1.0 + (1.0 - alpha_grv));
                b[bx(2, 5)] = c(alpha_grv * 0.0);
                b[bx(2, 6)] = c(alpha_grv * 0.0);
                b[bx(3, 1)] = c(0.0);
                b[bx(3, 2)] = c(0.0);
                b[bx(3, 3)] = c(alpha_grv * 0.0);
                b[bx(3, 4)] = c(alpha_grv * 0.0);
                b[bx(3, 5)] = c(1.0);
                b[bx(3, 6)] = c(0.0);
            }
        }
        // trans_cond: no transformation for SET
        Ok((b, [c(1.0); 3]))
    }

    /// `build_o(st, B, scl)` with `build_vacuum_o_`
    /// (`nad_bound_m.fypp:483-590`).
    ///
    /// # Errors
    ///
    /// Context errors.
    pub fn build_o(
        &self,
        cx: &Context<'_>,
        omega: Cplx,
    ) -> Result<([Cplx; 18], [Cplx; 3]), GyreError> {
        let [v, u, nabla_ad, c_thn] = self.coeff_o;
        let (alpha_grv, alpha_gbc, alpha_rht, alpha_omg) = (
            self.alpha_grv,
            self.alpha_gbc,
            self.alpha_rht,
            self.alpha_omg,
        );
        let omega_rot = cx.omega_rot(self.pt[1])?;
        let omega_c = cx.omega_c_c(omega_rot, omega);
        let i_omega_c = Cplx::new(0.0, 1.0) * sqrt_c(c(alpha_omg)) * omega_c;
        let l_e = cx.l_e_c(omega_rot)?;
        let f_rht = c(1.0) - c(0.25 * alpha_rht) * i_omega_c * c(c_thn);
        let mut b = [c(0.0); 18];
        let bx = |i: usize, j: usize| (i - 1) + 3 * (j - 1);
        b[bx(1, 1)] = c(1.0);
        b[bx(1, 2)] = c(-1.0);
        b[bx(1, 3)] = c(alpha_grv * 0.0);
        b[bx(1, 4)] = c(alpha_grv * 0.0);
        b[bx(1, 5)] = c(0.0);
        b[bx(1, 6)] = c(0.0);
        b[bx(2, 1)] = c(alpha_grv * alpha_gbc * u);
        b[bx(2, 2)] = c(alpha_grv * 0.0);
        b[bx(2, 3)] = c(alpha_grv) * (l_e + c(1.0)) + c(1.0 - alpha_grv);
        b[bx(2, 4)] = c(alpha_grv * 1.0);
        b[bx(2, 5)] = c(alpha_grv * 0.0);
        b[bx(2, 6)] = c(alpha_grv * 0.0);
        b[bx(3, 1)] = c(2.0 - 4.0 * nabla_ad * v);
        b[bx(3, 2)] = c(4.0 * nabla_ad * v);
        b[bx(3, 3)] = c(alpha_grv * 0.0);
        b[bx(3, 4)] = c(alpha_grv * 0.0);
        b[bx(3, 5)] = c(4.0) * f_rht;
        b[bx(3, 6)] = c(-1.0);
        Ok((b, [c(1.0); 3]))
    }
}

/// `nad_match_t` (`nad/nad_match_m.fypp`).
#[derive(Debug, Clone)]
pub struct NadMatch {
    v_l: f64,
    v_r: f64,
    u_l: f64,
    u_r: f64,
    nabla_ad_l: f64,
    nabla_ad_r: f64,
}

impl NadMatch {
    /// `nad_match_t(cx, pt_a, pt_b, md_p, os_p)` (`nad_match_m.fypp:80-143`).
    ///
    /// # Errors
    ///
    /// Model evaluation errors.
    pub fn new(
        cx: &Context<'_>,
        pt_a: Point,
        pt_b: Point,
        os_p: &OscPar,
    ) -> Result<Self, GyreError> {
        let ml = cx.ml;
        ml.check(&[idx::V_2, idx::U, idx::NABLA_AD])?;
        let v_l = ml.coeff(idx::V_2, pt_a)? * (pt_a.x * pt_a.x);
        let v_r = ml.coeff(idx::V_2, pt_b)? * (pt_b.x * pt_b.x);
        let u_l = ml.coeff(idx::U, pt_a)?;
        let u_r = ml.coeff(idx::U, pt_b)?;
        let nabla_ad_l = ml.coeff(idx::NABLA_AD, pt_a)?;
        let nabla_ad_r = ml.coeff(idx::NABLA_AD, pt_b)?;
        let tr = NadTrans::new(os_p)?;
        tr.stencil(cx, &[pt_a, pt_b])?;
        Ok(Self {
            v_l,
            v_r,
            u_l,
            u_r,
            nabla_ad_l,
            nabla_ad_r,
        })
    }

    /// `build(st, E_l, E_r, scl)` (`nad_match_m.fypp:147-273`).
    #[must_use]
    pub fn build(&self) -> ([Cplx; 36], [Cplx; 36], ExtC) {
        let mut e_l = [c(0.0); 36];
        let mut e_r = [c(0.0); 36];
        e_l[ix(1, 1)] = c(-1.0);
        e_l[ix(2, 1)] = c(self.u_l);
        e_l[ix(2, 2)] = c(-self.u_l);
        e_l[ix(3, 3)] = c(-1.0);
        e_l[ix(4, 1)] = c(-self.u_l);
        e_l[ix(4, 4)] = c(-1.0);
        e_l[ix(5, 1)] = c(self.v_l * self.nabla_ad_l);
        e_l[ix(5, 2)] = c(-self.v_l * self.nabla_ad_l);
        e_l[ix(5, 5)] = c(-1.0);
        e_l[ix(6, 6)] = c(-1.0);

        e_r[ix(1, 1)] = c(1.0);
        e_r[ix(2, 1)] = c(-self.u_r);
        e_r[ix(2, 2)] = c(self.u_r);
        e_r[ix(3, 3)] = c(1.0);
        e_r[ix(4, 1)] = c(self.u_r);
        e_r[ix(4, 4)] = c(1.0);
        e_r[ix(5, 1)] = c(-self.v_r * self.nabla_ad_r);
        e_r[ix(5, 2)] = c(self.v_r * self.nabla_ad_r);
        e_r[ix(5, 5)] = c(1.0);
        e_r[ix(6, 6)] = c(1.0);
        (e_l, e_r, ExtC::from_real(1.0))
    }
}

#[derive(Debug, Clone)]
enum NadDiffKind {
    CollocGl2 {
        colloc_eq: NadEqns,
        dx: f64,
        reverse: bool,
        omega_rot: f64,
    },
    MagnusGl2 {
        df_eq: NadEqns,
        mid_eq: NadEqns,
        dx: f64,
    },
    Match(NadMatch),
}

/// `nad_diff_t` (`nad_diff_m.fypp:77-198`): `COLLOC_GL2`, `MAGNUS_GL2`, and
/// segment `nad_match_t`.
#[derive(Debug, Clone)]
pub struct NadDiff {
    kind: NadDiffKind,
}

impl NadDiff {
    /// `nad_diff_t(cx, pt_a, pt_b, md_p, nm_p, os_p)`.
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for schemes other than `COLLOC_GL2` and
    /// `MAGNUS_GL2`; stencil errors.
    pub fn new(
        cx: &Context<'_>,
        pt_a: Point,
        pt_b: Point,
        nm_p: &NumPar,
        os_p: &OscPar,
    ) -> Result<Self, GyreError> {
        if pt_a.s != pt_b.s {
            return Ok(Self {
                kind: NadDiffKind::Match(NadMatch::new(cx, pt_a, pt_b, os_p)?),
            });
        }
        let eq = NadEqns::new(cx, os_p)?;
        if nm_p.diff_scheme == "TRAPZ" {
            return Err(GyreError::Unported {
                what: "nonadiabatic diff_scheme 'TRAPZ'",
                at: "nad/nad_diff_m.fypp:101-102",
            });
        }
        let kind = diff_kind(&nm_p.diff_scheme)?;
        let dx_m = pt_b.x - pt_a.x;
        let pt_m = Point {
            s: pt_a.s,
            x: pt_a.x + 0.5 * dx_m,
        };
        let mut mid_eq = eq.clone();
        mid_eq.stencil(cx, &[pt_m])?;
        let omega_rot = cx.omega_rot(pt_m)?;
        match kind {
            DiffKind::CollocGl2 => {
                let (reverse, dx, pt) = colloc_stencil(pt_a, pt_b, true);
                let mut colloc_eq = eq;
                colloc_eq.stencil(cx, &pt)?;
                Ok(Self {
                    kind: NadDiffKind::CollocGl2 {
                        colloc_eq,
                        dx,
                        reverse,
                        omega_rot,
                    },
                })
            }
            DiffKind::MagnusGl2 => {
                let (dx, pt) = magnus_stencil(pt_a, pt_b, "GL2")?;
                let mut df_eq = eq;
                df_eq.stencil(cx, &pt)?;
                Ok(Self {
                    kind: NadDiffKind::MagnusGl2 { df_eq, mid_eq, dx },
                })
            }
            DiffKind::CollocGl4 => Err(GyreError::Unported {
                what:
                    "complex COLLOC_GL4 (linear_solve with XGESVX for complex)",
                at: "diff/colloc_diff_m.fypp:248-268",
            }),
        }
    }

    /// `build(st, E_l, E_r, scl)` (`nad_diff_m.fypp:142-198`).
    ///
    /// # Errors
    ///
    /// Context, eigendecomposition, and model errors.
    pub fn build(
        &self,
        cx: &Context<'_>,
        omega: Cplx,
    ) -> Result<([Cplx; 36], [Cplx; 36], ExtC), GyreError> {
        match &self.kind {
            NadDiffKind::Match(mt) => Ok(mt.build()),
            NadDiffKind::CollocGl2 {
                colloc_eq,
                dx,
                reverse,
                omega_rot,
            } => {
                let a = colloc_eq.a(cx, 0, omega)?;
                let h: [Cplx; 36] =
                    std::array::from_fn(|k| Cplx::real_mul(0.5 * *dx, a[k]));
                let id = |k: usize| {
                    if k.is_multiple_of(N_E + 1) {
                        c(1.0)
                    } else {
                        c(0.0)
                    }
                };
                let plus: [Cplx; 36] = std::array::from_fn(|k| h[k] + id(k));
                let minus: [Cplx; 36] = std::array::from_fn(|k| h[k] - id(k));
                let (e_l, e_r) = if *reverse {
                    (minus, plus)
                } else {
                    (plus, minus)
                };
                // scl = ext_ct(1._RD); then scl = scl/sqrt(omega_c) (nad_diff_m.fypp:188-190)
                let omega_c = cx.omega_c_c(*omega_rot, omega);
                let scl = ExtC::from_real(1.0)
                    .divide(ExtC::from_cplx(sqrt_c(omega_c)));
                Ok((e_l, e_r, scl))
            }
            NadDiffKind::MagnusGl2 { df_eq, mid_eq, dx } => {
                let mut domega = df_eq.a(cx, 0, omega)?;
                let (el, er, mut scl) = magnus_build_c(N_E, *dx, &mut domega)?;
                let a_mid = mid_eq.a(cx, 0, omega)?;
                let lambda3 = sqrt_c(a_mid[ix(6, 5)] * a_mid[ix(5, 6)]);
                let arg = (-lambda3).mul_real(*dx);
                scl = scl.times(ExtC::from_cplx(arg).exp());
                Ok((
                    std::array::from_fn(|k| el[k]),
                    std::array::from_fn(|k| er[k]),
                    scl,
                ))
            }
        }
    }
}

/// `nad_bvp_t` (`nad_bvp_m.fypp:81-129`) over `bvp_ct` (`bvp_m.fypp`).
#[derive(Debug, Clone)]
pub struct NadBvp<'a> {
    /// The context.
    pub cx: &'a Context<'a>,
    /// The grid.
    pub gr: Grid,
    bd: NadBound,
    df: Vec<NadDiff>,
    /// The system matrix.
    pub sm: BlockSysmtx<Cplx>,
    factored: bool,
    /// `md_p`.
    pub md_p: ModePar,
    /// `nm_p`.
    pub nm_p: NumPar,
    /// `os_p`.
    pub os_p: OscPar,
}

impl<'a> NadBvp<'a> {
    /// `nad_bvp_t(cx, gr, md_p, nm_p, os_p)`.
    ///
    /// # Errors
    ///
    /// Component errors.
    pub fn new(
        cx: &'a Context<'a>,
        gr: &Grid,
        md_p: &ModePar,
        nm_p: &NumPar,
        os_p: &OscPar,
    ) -> Result<Self, GyreError> {
        let bd = NadBound::new(cx, os_p)?;
        let mut df = Vec::with_capacity(gr.n() - 1);
        for j in 0..gr.n() - 1 {
            df.push(NadDiff::new(cx, gr.pt[j], gr.pt[j + 1], nm_p, os_p)?);
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
        let sm =
            BlockSysmtx::new(gr.n() - 1, N_E, NadBound::N_I, NadBound::N_O);
        // bp%tr = nad_trans_t(...); call bp%tr%stencil(gr%pt)
        NadTrans::new(os_p)?.stencil(cx, &gr.pt)?;
        Ok(Self {
            cx,
            gr: gr.clone(),
            bd,
            df,
            sm,
            factored: false,
            md_p: md_p.clone(),
            nm_p: nm_p.clone(),
            os_p: os_p.clone(),
        })
    }

    /// Number of grid points.
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
        omega: Cplx,
        mode_build: bool,
        tap: &mut dyn GyreTap,
    ) -> Result<(), GyreError> {
        let (b_i, scl_i) = self.bd.build_i(self.cx, omega)?;
        self.sm.set_b_i(&b_i, &scl_i);
        let (b_o, scl_o) = self.bd.build_o(self.cx, omega)?;
        self.sm.set_b_o(&b_o, &scl_o);
        tap.build_c(self.n(), mode_build, omega, &b_i, &b_o);
        for j in 0..self.n() - 1 {
            let (e_l, e_r, scl) = self.df[j].build(self.cx, omega)?;
            self.sm.set_e(j, &e_l, &e_r, scl);
            tap.block_c(j + 1, &e_l, &e_r, scl);
        }
        self.factored = false;
        Ok(())
    }

    /// `factor()`.
    ///
    /// # Errors
    ///
    /// LAPACK errors.
    pub fn factor(&mut self) -> Result<(), GyreError> {
        self.sm.factor()?;
        self.factored = true;
        Ok(())
    }

    /// `det()`.
    ///
    /// # Errors
    ///
    /// Unfactored matrix (`ASSERT`); LAPACK errors.
    pub fn det(&self) -> Result<ExtC, GyreError> {
        if !self.factored {
            return Err(GyreError::Assert("matrix has not been factorized"));
        }
        self.sm.det()
    }

    /// `soln_vec_hom()`: `y(6, n)` column-major.
    ///
    /// # Errors
    ///
    /// LAPACK errors.
    pub fn soln_vec_hom(&self) -> Result<Vec<Cplx>, GyreError> {
        self.sm.soln_vec_hom()
    }
}

/// `wave_t(bp, st, id)` for the nonadiabatic BVP (`wave_t_hom_` and
/// `wave_t_y_`, `nad_bvp_m.fypp:133-229`). The `SET` variables need no
/// transformation back (`nad_trans_m.fypp:287-333`).
///
/// # Errors
///
/// Component errors.
pub fn nad_wave_hom<'a>(
    bp: &mut NadBvp<'a>,
    omega: Cplx,
    id: i32,
    tap: &mut dyn GyreTap,
) -> Result<Wave<'a>, GyreError> {
    bp.build(omega, true, tap)?;
    bp.factor()?;
    let y = bp.soln_vec_hom()?;
    let discrim = bp.det()?;
    Wave::new(
        omega, y, discrim, bp.cx, &bp.gr, &bp.md_p, &bp.nm_p, &bp.os_p, id,
    )
}
