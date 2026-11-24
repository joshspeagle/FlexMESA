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
use crate::ext_real::ExtR;
use crate::grid::{Grid, Point};
use crate::lapack::lu::dgesvx;
use crate::matmul::matmul_rr;
use crate::model::idx;
use crate::par::{ModePar, NumPar, OscPar};
use crate::GyreError;

fn sq(x: f64) -> f64 {
    x * x
}

/// `rad_trans_t` (`rad_trans_m.fypp`) for the `JCD` variable set.
#[derive(Debug, Clone)]
pub struct RadTrans {
    /// Per stencil point: `[V_2, dV_2, c_1, dc_1]` (`J_V_2 .. J_DC_1`).
    coeff: Vec<[f64; 4]>,
}

impl RadTrans {
    /// `rad_trans_t(cx, md_p, os_p)` (`rad_trans_m.fypp:96-128`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for variable sets other than `JCD`.
    pub fn new(os_p: &OscPar) -> Result<Self, GyreError> {
        match os_p.variables_set.as_str() {
            "JCD" => Ok(Self { coeff: Vec::new() }),
            "GYRE" | "DZIEM" | "MIX" | "LAGP" => Err(GyreError::Unported {
                what: "radial variables_set other than 'JCD'",
                at: "rad/rad_trans_m.fypp:107-117",
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
        let h = self.h(i, omega);
        let r = matmul_rr(c, &h, 1, 2, 2)?;
        Ok([r[0], r[1]])
    }

    /// `trans_vars(y, i, st, from=.FALSE.)` (`rad_trans_m.fypp:279-329`):
    /// `y = MATMUL(H, y)`.
    #[must_use]
    pub fn trans_vars_to(&self, y: [f64; 2], i: usize, omega: f64) -> [f64; 2] {
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

/// `rad_bound_t` (`rad_bound_m.fypp`) with a `REGULAR` inner and `JCD`
/// outer boundary.
#[derive(Debug, Clone)]
pub struct RadBound {
    tr: RadTrans,
    alpha_omg: f64,
    /// `coeff(1, J_C_1)`.
    c_1_i: f64,
    /// `coeff(2, [J_V, J_AS, J_C_1, J_GAMMA_1])`.
    atm_o: [f64; 4],
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
    /// [`GyreError::Unported`] for boundary types other than `REGULAR`
    /// and `JCD`; model evaluation errors.
    pub fn new(cx: &Context<'_>, os_p: &OscPar) -> Result<Self, GyreError> {
        let tr = RadTrans::new(os_p)?;
        let ml = cx.ml;
        let pt_i = cx.pt_i;
        let pt_o = cx.pt_o;
        match os_p.inner_bound.as_str() {
            "REGULAR" => {
                if pt_i.x != 0.0 {
                    return Err(GyreError::Assert(
                        "boundary condition invalid for x /= 0",
                    ));
                }
            }
            "ZERO_R" => {
                return Err(GyreError::Unported {
                    what: "inner_bound = 'ZERO_R'",
                    at: "rad/rad_bound_m.fypp:125-127",
                })
            }
            _ => return Err(GyreError::Abort("invalid inner_bound")),
        }
        match os_p.outer_bound.as_str() {
            "JCD" => {
                if ml.is_vacuum(pt_o)? {
                    return Err(GyreError::Unported {
                        what: "outer_bound = 'JCD' at a vacuum outer point",
                        at: "rad/rad_bound_m.fypp:154-155",
                    });
                }
            }
            "VACUUM" | "DZIEM" | "ISOTHERMAL" | "UNNO" => {
                return Err(GyreError::Unported {
                    what: "radial outer_bound other than 'JCD'",
                    at: "rad/rad_bound_m.fypp:133-152",
                })
            }
            _ => return Err(GyreError::Abort("invalid outer_bound")),
        }
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
        let c_1_i = ml.coeff(idx::C_1, pt_i)?;
        // eval_atmos_coeffs_isothrm (atmos_m.fypp:275-296)
        let v = ml.coeff(idx::V_2, pt_o)? * sq(pt_o.x);
        let a_s = ml.coeff(idx::V_2, pt_o)?
            * sq(pt_o.x)
            * (1.0 - 1.0 / ml.coeff(idx::GAMMA_1, pt_o)?);
        let c_1 = ml.coeff(idx::C_1, pt_o)?;
        let gamma_1 = ml.coeff(idx::GAMMA_1, pt_o)?;
        let mut bd = Self {
            tr,
            alpha_omg,
            c_1_i,
            atm_o: [v, a_s, c_1, gamma_1],
        };
        bd.tr.stencil(cx, &[pt_i, pt_o])?;
        Ok(bd)
    }

    /// `build_i(st, B, scl)` (`rad_bound_m.fypp:246-317`).
    ///
    /// # Errors
    ///
    /// As [`RadTrans::trans_cond`].
    pub fn build_i(
        &self,
        omega: f64,
    ) -> Result<([f64; 2], [f64; 1]), GyreError> {
        let omega_c = omega;
        let b = [self.c_1_i * self.alpha_omg * sq(omega_c), 0.0];
        Ok((self.tr.trans_cond(&b, 0, omega)?, [1.0]))
    }

    /// `build_o(st, B, scl)` with `build_jcd_o_` (`rad_bound_m.fypp:351-567`).
    ///
    /// # Errors
    ///
    /// As [`RadTrans::trans_cond`].
    pub fn build_o(
        &self,
        omega: f64,
    ) -> Result<([f64; 2], [f64; 1]), GyreError> {
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

/// `rad_diff_t` over `colloc_diff_rt` with the `GL4` scheme
/// (`rad_diff_m.fypp:69-131`, `colloc_diff_m.fypp:121-387`).
#[derive(Debug, Clone)]
pub struct RadDiff {
    eq: RadEqns,
    dx: f64,
    reverse: bool,
}

impl RadDiff {
    /// `rad_diff_t(cx, pt_a, pt_b, md_p, nm_p, os_p)`.
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for segment boundaries and difference
    /// schemes other than `COLLOC_GL4`; stencil errors.
    pub fn new(
        cx: &Context<'_>,
        pt_a: Point,
        pt_b: Point,
        nm_p: &NumPar,
        os_p: &OscPar,
    ) -> Result<Self, GyreError> {
        if pt_a.s != pt_b.s {
            return Err(GyreError::Unported {
                what: "radial difference equations across a segment boundary",
                at: "rad/rad_diff_m.fypp:91-97",
            });
        }
        let eq = RadEqns::new(os_p)?;
        match nm_p.diff_scheme.as_str() {
            "COLLOC_GL4" => {}
            "MAGNUS_GL2" | "MAGNUS_GL4" | "MAGNUS_GL6" | "COLLOC_GL2"
            | "COLLOC_GL6" | "TRAPZ" | "MIRK" => {
                return Err(GyreError::Unported {
                    what: "diff_scheme other than 'COLLOC_GL4'",
                    at: "diff/diff_factory_m.fypp:74-89",
                })
            }
            _ => return Err(GyreError::Abort("invalid diff_scheme")),
        }
        let (_, _, c_i) = gl4();
        let reverse = pt_a.x >= 0.5;
        let (dx, xs) = if reverse {
            let dx = pt_a.x - pt_b.x;
            (dx, [pt_b.x + c_i[0] * dx, pt_b.x + c_i[1] * dx])
        } else {
            let dx = pt_b.x - pt_a.x;
            (dx, [pt_a.x + c_i[0] * dx, pt_a.x + c_i[1] * dx])
        };
        let pt = [
            Point {
                s: pt_a.s,
                x: xs[0],
            },
            Point {
                s: pt_a.s,
                x: xs[1],
            },
        ];
        let mut df = Self { eq, dx, reverse };
        df.eq.stencil(cx, &pt)?;
        Ok(df)
    }

    /// `build(st, E_l, E_r, scl)` (`build_GL4_` and `build_irk_`,
    /// `colloc_diff_m.fypp:181-387`).
    ///
    /// # Errors
    ///
    /// LAPACK errors and a non-zero `XGESVX` return (`ASSERT`).
    pub fn build(
        &self,
        omega: f64,
    ) -> Result<([f64; 4], [f64; 4], ExtR), GyreError> {
        const N_E: usize = 2;
        const N_S: usize = 2;
        const NM: usize = N_S * N_E;
        let (a_ij, b_i, _) = gl4();
        let a: [[f64; 4]; N_S] = [self.eq.a(0, omega)?, self.eq.a(1, omega)?];
        // M (4x4) and B (4x2), column-major with leading dimension 4.
        let mut m = [0.0_f64; NM * NM];
        let mut b = [0.0_f64; NM * N_E];
        for i in 0..N_S {
            let i_a = i * N_E;
            for j in 0..N_S {
                let j_a = j * N_E;
                let aij = a_ij[i + j * N_S];
                for c in 0..N_E {
                    for r in 0..N_E {
                        m[(i_a + r) + (j_a + c) * NM] =
                            -(self.dx * aij * a[i][r + c * N_E]);
                    }
                }
                if i == j {
                    // M + identity_matrix(n_e): every element gets the
                    // identity entry added, including the zeros.
                    for c in 0..N_E {
                        for r in 0..N_E {
                            let id = if r == c { 1.0 } else { 0.0 };
                            m[(i_a + r) + (j_a + c) * NM] += id;
                        }
                    }
                }
            }
            for c in 0..N_E {
                for r in 0..N_E {
                    b[(i_a + r) + c * NM] = a[i][r + c * N_E];
                }
            }
        }
        // X = linear_solve(M, B) (linalg_m.fypp:448-493)
        let mut af = [0.0_f64; NM * NM];
        let mut ipiv = [0usize; NM];
        let mut x = [0.0_f64; NM * N_E];
        let o = dgesvx(
            b'E', b'N', NM, N_E, &mut m, &mut af, &mut ipiv, &mut b, &mut x,
        )?;
        if o.info != 0 {
            return Err(GyreError::LapackInfo {
                routine: "XGESVX",
                info: i64::try_from(o.info).unwrap_or(i64::MAX),
            });
        }
        // E_a = I + sum_i dx*b_i*K_i ; E_b = -I
        let mut e_a = [1.0, 0.0, 0.0, 1.0];
        for i in 0..N_S {
            let i_a = i * N_E;
            for c in 0..N_E {
                for r in 0..N_E {
                    let k = x[(i_a + r) + c * NM];
                    e_a[r + c * N_E] += self.dx * b_i[i] * k;
                }
            }
        }
        let e_b = [-1.0, -0.0, -0.0, -1.0];
        let scl = ExtR::from_real(1.0);
        if self.reverse {
            Ok((e_b, e_a, scl))
        } else {
            Ok((e_a, e_b, scl))
        }
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
    pub sm: crate::sysmtx::BlockSysmtx,
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
