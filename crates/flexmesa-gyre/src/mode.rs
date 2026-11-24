// SPDX-License-Identifier: GPL-3.0-only
//! GYRE waves and modes: `wave_t` (`mode/wave_m.fypp`), its construction
//! from the radial BVP (`rad/rad_bvp_m.fypp:152-339`), and `mode_t` with
//! radial classification (`mode/mode_m.fypp:79-317`).

use crate::context::Context;
use crate::crmath::abs_c;
use crate::ext_complex::ExtC;
use crate::fcomplex::Cplx;
use crate::grid::{fmax, fmin, Grid};
use crate::integ::{integral, integrate};
use crate::math_m::pow_rc;
use crate::model::idx;
use crate::par::{ModePar, NumPar, OscPar};
use crate::rad::{GyreTap, RadBvp};
use crate::GyreError;

fn sq(x: f64) -> f64 {
    x * x
}

/// `wave_t` for the states GYRE's real radial path builds.
#[derive(Debug, Clone)]
pub struct Wave<'a> {
    /// Complex state frequency `st%omega`.
    pub st_omega: Cplx,
    /// The context.
    pub cx: &'a Context<'a>,
    /// The grid.
    pub gr: Grid,
    /// `md_p`.
    pub md_p: ModePar,
    /// `nm_p`.
    pub nm_p: NumPar,
    /// `os_p`.
    pub os_p: OscPar,
    /// `y_c(6, n)`, column-major (`y_c[i + 6*j]`).
    pub y_c: Vec<Cplx>,
    e_scl2: f64,
    /// Discriminant.
    pub discrim: ExtC,
    /// Eigenfunction scale `scl`.
    pub scl: Cplx,
    /// `omega`.
    pub omega: Cplx,
    /// Effective harmonic degree `l_i`.
    pub l_i: Cplx,
    /// Number of points.
    pub n: usize,
    /// Reference point index (1-based).
    pub j_ref: usize,
    /// Harmonic degree.
    pub l: i32,
    /// Azimuthal order.
    pub m: i32,
    /// Mode identifier.
    pub id: i32,
    /// Vacuum outer boundary.
    pub is_vacuum: bool,
}

impl<'a> Wave<'a> {
    /// `wave_t(st, y_c, discrim, cx, gr, md_p, nm_p, os_p, id)`
    /// (`wave_m.fypp:166-246`), with `st = state_ct(omega, omega_r)`.
    ///
    /// # Errors
    ///
    /// Model and unported-path errors from the eigenfunction functions.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        omega: Cplx,
        y_c: Vec<Cplx>,
        discrim: ExtC,
        cx: &'a Context<'a>,
        gr: &Grid,
        md_p: &ModePar,
        nm_p: &NumPar,
        os_p: &OscPar,
        id: i32,
    ) -> Result<Self, GyreError> {
        let l_i = cx.l_e_c(cx.omega_rot(cx.pt_i)?)?;
        let pt_i = gr.pt_i();
        let pt_o = gr.pt_o();
        let x_ref = fmin(fmax(os_p.x_ref, pt_i.x), pt_o.x);
        // MINLOC(abs(gr%pt%x - x_ref), DIM=1): first minimum.
        let mut j_ref = 1;
        let mut best = f64::INFINITY;
        let mut first = true;
        for (j, p) in gr.pt.iter().enumerate() {
            let d = (p.x - x_ref).abs();
            if first || d < best {
                best = d;
                j_ref = j + 1;
                first = false;
            }
        }
        let mut wv = Self {
            st_omega: omega,
            cx,
            gr: gr.clone(),
            md_p: md_p.clone(),
            nm_p: nm_p.clone(),
            os_p: os_p.clone(),
            y_c,
            e_scl2: 0.0,
            discrim,
            scl: Cplx::new(1.0, 0.0),
            omega,
            l_i,
            n: gr.n(),
            j_ref,
            l: md_p.l,
            m: md_p.m,
            id,
            is_vacuum: false,
        };
        wv.e_scl2 = wv.e_uncached()? / sq(abs_c(wv.scl));
        wv.is_vacuum = cx.ml.is_vacuum(pt_o)?;
        Ok(wv)
    }

    /// `x(j)` (1-based).
    #[must_use]
    pub fn x(&self, j: usize) -> f64 {
        self.gr.pt[j - 1].x
    }

    /// `y_i(i, j)` (1-based): `scl*y_c(i, j)`.
    #[must_use]
    pub fn y_i(&self, i: usize, j: usize) -> Cplx {
        self.scl * self.y_c[(i - 1) + 6 * (j - 1)]
    }

    /// `freq(units)` without a frame (`wave_m.fypp:322-347`).
    ///
    /// # Errors
    ///
    /// As [`Context::freq_scale`].
    pub fn freq(&self, units: &str) -> Result<Cplx, GyreError> {
        Ok(self.st_omega.mul_real(self.cx.freq_scale(units)?))
    }

    /// `xi_r(j)` (`wave_m.fypp:534-573`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for `l_i == 1`.
    pub fn xi_r(&self, j: usize) -> Result<Cplx, GyreError> {
        let x = self.x(j);
        let y_1 = self.y_i(1, j);
        if !self.l_i.feq(Cplx::from_real(1.0)) {
            if x != 0.0 {
                Ok(y_1 * pow_rc(x, self.l_i - Cplx::from_real(1.0)))
            } else {
                Ok(Cplx::from_real(0.0))
            }
        } else {
            Err(GyreError::Unported {
                what: "xi_r for l_i = 1",
                at: "mode/wave_m.fypp:561-565",
            })
        }
    }

    /// `xi_h(j)` (`wave_m.fypp:577-671`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for `l /= 0`.
    pub fn xi_h(&self, _j: usize) -> Result<Cplx, GyreError> {
        if self.l != 0 {
            return Err(GyreError::Unported {
                what: "xi_h for l /= 0",
                at: "mode/wave_m.fypp:597-659",
            });
        }
        Ok(Cplx::from_real(0.0))
    }

    /// `lambda(j)` (`wave_m.fypp:1077-1100`).
    ///
    /// # Errors
    ///
    /// Model and context errors.
    pub fn lambda(&self, j: usize) -> Result<Cplx, GyreError> {
        let omega_rot = self.cx.omega_rot(self.gr.pt[j - 1])?;
        self.cx.lambda_c(omega_rot)
    }

    /// `lag_S(j)` (`wave_m.fypp:765-794`).
    ///
    /// # Errors
    ///
    /// None in practice; kept fallible for symmetry.
    pub fn lag_s(&self, j: usize) -> Result<Cplx, GyreError> {
        let x = self.x(j);
        let y_5 = self.y_i(5, j);
        if x != 0.0 {
            Ok(y_5 * pow_rc(x, self.l_i - Cplx::from_real(2.0)))
        } else {
            Ok(Cplx::from_real(0.0))
        }
    }

    /// `lag_P(j)` (`wave_m.fypp:875-917`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for `l_i /= 0`; model errors.
    pub fn lag_p(&self, j: usize) -> Result<Cplx, GyreError> {
        let y_1 = self.y_i(1, j);
        let y_2 = self.y_i(2, j);
        let v_2 = self.cx.ml.coeff(idx::V_2, self.gr.pt[j - 1])?;
        if !self.l_i.feq(Cplx::from_real(0.0)) {
            return Err(GyreError::Unported {
                what: "lag_P for l_i /= 0",
                at: "mode/wave_m.fypp:897-903",
            });
        }
        Ok(Cplx::real_mul(v_2, y_2 - y_1))
    }

    /// `lag_T(j)` (`wave_m.fypp:1043-1073`).
    ///
    /// # Errors
    ///
    /// As [`Self::lag_p`].
    pub fn lag_t(&self, j: usize) -> Result<Cplx, GyreError> {
        let lag_p = self.lag_p(j)?;
        let lag_s = self.lag_s(j)?;
        let nabla_ad = self.cx.ml.coeff(idx::NABLA_AD, self.gr.pt[j - 1])?;
        Ok(Cplx::real_mul(nabla_ad, lag_p) + lag_s)
    }

    /// `dE_dx(j)` (`wave_m.fypp:1104-1139`).
    ///
    /// # Errors
    ///
    /// As [`Self::xi_r`], [`Self::xi_h`], [`Self::lambda`].
    pub fn de_dx(&self, j: usize) -> Result<f64, GyreError> {
        let pt = self.gr.pt[j - 1];
        let xi_r = self.xi_r(j)?;
        let xi_h = self.xi_h(j)?;
        let lambda = self.lambda(j)?;
        let u = self.cx.ml.coeff(idx::U, pt)?;
        let c_1 = self.cx.ml.coeff(idx::C_1, pt)?;
        Ok(
            (sq(abs_c(xi_r)) + abs_c(lambda) * sq(abs_c(xi_h))) * u * sq(pt.x)
                / c_1,
        )
    }

    /// `dW_dx(j)` (`wave_m.fypp:1143-1186`).
    ///
    /// # Errors
    ///
    /// As [`Self::lag_t`].
    pub fn dw_dx(&self, j: usize) -> Result<f64, GyreError> {
        let ml = self.cx.ml;
        let g = self.cx.k.g_gravity;
        let t_dyn =
            (crate::powi::powi_const(ml.r_star, 3)? / (g * ml.m_star)).sqrt();
        let t_kh = (g * sq(ml.m_star) / ml.r_star) / ml.l_star;
        let pt = self.gr.pt[j - 1];
        let lag_t = self.lag_t(j)?;
        let lag_s = self.lag_s(j)?;
        let c_thk = ml.coeff(idx::C_THK, pt)?;
        Ok(self.cx.k.math.pi
            * (lag_t.conj() * lag_s).im
            * c_thk
            * sq(pt.x)
            * t_dyn
            / t_kh)
    }

    fn e_uncached(&self) -> Result<f64, GyreError> {
        let x = self.gr.xs();
        let mut de = Vec::with_capacity(self.n);
        for j in 1..=self.n {
            de.push(self.de_dx(j)?);
        }
        Ok(integrate(&x, &de, None))
    }

    /// `E()` with the cache (`wave_m.fypp:2021-2063`).
    #[must_use]
    pub fn e(&self) -> f64 {
        self.e_scl2 * sq(abs_c(self.scl))
    }

    /// `E_norm()` (`wave_m.fypp:2127-2174`).
    ///
    /// # Errors
    ///
    /// As [`Self::xi_r`]; invalid `inertia_norm`.
    pub fn e_norm(&self) -> Result<f64, GyreError> {
        let j = self.j_ref;
        let e = self.e();
        let xi_r = self.xi_r(j)?;
        let xi_h = self.xi_h(j)?;
        let lambda = self.lambda(j)?;
        let a2 = match self.os_p.inertia_norm.as_str() {
            "RADIAL" => sq(abs_c(xi_r)),
            "HORIZ" => abs_c(lambda) * sq(abs_c(xi_h)),
            "BOTH" => sq(abs_c(xi_r)) + abs_c(lambda) * sq(abs_c(xi_h)),
            _ => return Err(GyreError::Abort("Invalid inertia_norm")),
        };
        // Upstream also warns ('Amplitude at x_ref is zero ...') when A2 == 0.
        Ok(if a2 == 0.0 { e } else { e / a2 })
    }
}

/// `mode_t`: a wave with its classification.
#[derive(Debug, Clone)]
pub struct Mode<'a> {
    /// The normalised wave.
    pub wave: Wave<'a>,
    /// `n_pg`.
    pub n_pg: i32,
    /// `n_p`.
    pub n_p: i32,
    /// `n_g`.
    pub n_g: i32,
}

/// `count_windings_(y_1, y_2, n_c, n_a)` (`mode_m.fypp:255-317`).
fn count_windings(y_1: &[f64], y_2: &[f64]) -> (i32, i32) {
    let (mut n_c, mut n_a) = (0, 0);
    for j in 0..y_1.len().saturating_sub(1) {
        if y_1[j] >= 0.0 && y_1[j + 1] < 0.0 {
            let y_2_cross =
                y_2[j] - y_1[j] * (y_2[j + 1] - y_2[j]) / (y_1[j + 1] - y_1[j]);
            if y_2_cross >= 0.0 {
                n_a += 1;
            } else {
                n_c += 1;
            }
        } else if y_1[j] <= 0.0 && y_1[j + 1] > 0.0 {
            let y_2_cross =
                y_2[j] - y_1[j] * (y_2[j + 1] - y_2[j]) / (y_1[j + 1] - y_1[j]);
            if y_2_cross <= 0.0 {
                n_a += 1;
            } else {
                n_c += 1;
            }
        }
    }
    (n_c, n_a)
}

impl<'a> Mode<'a> {
    /// `mode_t(wv)` (`mode_m.fypp:79-112`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for `l /= 0`; eigenfunction errors.
    pub fn new(wv: Wave<'a>) -> Result<Self, GyreError> {
        let mut wave = wv;
        let y_1_ref = wave.y_i(1, wave.j_ref);
        let f_phase = if abs_c(y_1_ref) > f64::MIN_POSITIVE {
            y_1_ref.conj().div_real(abs_c(y_1_ref))
        } else {
            Cplx::from_real(1.0)
        };
        wave.scl = wave.scl.div_real(wave.e().sqrt()) * f_phase;
        let mut md = Self {
            wave,
            n_pg: 0,
            n_p: 0,
            n_g: 0,
        };
        md.classify()?;
        Ok(md)
    }

    /// `classify_()` (`mode_m.fypp:116-251`), radial branch.
    fn classify(&mut self) -> Result<(), GyreError> {
        if self.wave.l != 0 {
            return Err(GyreError::Unported {
                what: "classification of non-radial modes",
                at: "mode/mode_m.fypp:166-247",
            });
        }
        let n = self.wave.n;
        let y_1: Vec<f64> = (1..=n).map(|j| self.wave.y_i(1, j).re).collect();
        let y_2: Vec<f64> = (1..=n).map(|j| self.wave.y_i(2, j).re).collect();
        let mut j_i = n;
        for j in 2..n {
            let (a, b, c) = (y_1[j - 2], y_1[j - 1], y_1[j]);
            if (b >= a && c >= b) || (b <= a && c <= b) {
                j_i = j;
                break;
            }
        }
        let (n_c, n_a) = count_windings(&y_1[j_i - 1..], &y_2[j_i - 1..]);
        self.n_p = n_a + n_c + 1;
        self.n_g = 0;
        self.n_pg = self.n_p - self.n_g;
        Ok(())
    }

    /// `E_norm()`.
    ///
    /// # Errors
    ///
    /// As [`Wave::e_norm`].
    pub fn e_norm(&self) -> Result<f64, GyreError> {
        self.wave.e_norm()
    }

    /// `freq(units)`.
    ///
    /// # Errors
    ///
    /// As [`Wave::freq`].
    pub fn freq(&self, units: &str) -> Result<Cplx, GyreError> {
        self.wave.freq(units)
    }
}

/// `wave_t(bp, st, id)` for the radial BVP (`wave_t_hom_` and
/// `wave_t_y_`, `rad_bvp_m.fypp:152-339`).
///
/// # Errors
///
/// Component errors; segmented grids are unported.
pub fn rad_wave_hom<'a>(
    bp: &mut RadBvp<'a>,
    omega: f64,
    id: i32,
    tap: &mut dyn GyreTap,
) -> Result<Wave<'a>, GyreError> {
    bp.build(omega, true, tap)?;
    bp.factor()?;
    let mut y = bp.soln_vec_hom()?;
    let n = bp.n();
    for j in 0..n {
        let yj = bp.tr.trans_vars_to([y[2 * j], y[2 * j + 1]], j, omega);
        y[2 * j] = yj[0];
        y[2 * j + 1] = yj[1];
    }
    // wave_t_y_
    let cx = bp.cx;
    let ml = cx.ml;
    let mut c_1 = vec![0.0; n];
    let mut y_4 = vec![0.0; n];
    let mut deul_phi = vec![0.0; n];
    for j in 0..n {
        let pt = bp.gr.pt[j];
        let u = ml.coeff(idx::U, pt)?;
        c_1[j] = ml.coeff(idx::C_1, pt)?;
        y_4[j] = -u * y[2 * j];
        deul_phi[j] = if pt.x != 0.0 {
            y_4[j] / (c_1[j] * pt.x)
        } else {
            0.0
        };
    }
    if bp.gr.s_i() != bp.gr.s_o() {
        return Err(GyreError::Unported {
            what: "radial eigenfunctions on a segmented grid",
            at: "rad/rad_bvp_m.fypp:292-294",
        });
    }
    let x = bp.gr.xs();
    let mut eul_phi = integral(&x, &deul_phi);
    let (e_n, y4_n) = (eul_phi[n - 1], y_4[n - 1]);
    for v in &mut eul_phi {
        *v = *v - e_n - y4_n;
    }
    let mut y_c = vec![Cplx::from_real(0.0); 6 * n];
    for j in 0..n {
        y_c[6 * j] = Cplx::from_real(y[2 * j]);
        y_c[6 * j + 1] = Cplx::from_real(y[2 * j + 1]);
        y_c[6 * j + 2] = Cplx::from_real(c_1[j] * eul_phi[j]);
        y_c[6 * j + 3] = Cplx::from_real(y_4[j]);
    }
    let discrim = ExtC::from_extr(bp.det()?);
    Wave::new(
        Cplx::from_real(omega),
        y_c,
        discrim,
        cx,
        &bp.gr,
        &bp.md_p,
        &bp.nm_p,
        &bp.os_p,
        id,
    )
}
