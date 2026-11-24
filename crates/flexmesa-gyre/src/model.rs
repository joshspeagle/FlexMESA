// SPDX-License-Identifier: GPL-3.0-only
//! GYRE evolutionary models built from MESA's in-memory pulsation data:
//! `mesa_file_m::init_mesa_model` (`model/mesa_file_m.fypp:143-458`),
//! `evol_model_t` (`model/evol_model_m.fypp`), `snap_points` and
//! `check_model` (`model/model_util_m.fypp`), cubic interpolation
//! (`interp/cubic_interp_m.fypp`) and monotone derivatives
//! (`interp/deriv_m.fypp:212-270`).

use crate::constants::GyreConstants;
use crate::forum::locate;
use crate::grid::{fmin, Grid, Point};
use crate::par::ModelPar;
use crate::GyreError;

/// Structure coefficient indices (`include/model.inc`, `COEFF_NAMES`),
/// 1-based as upstream.
pub mod idx {
    /// `I_V_2` (`V_2`).
    pub const V_2: usize = 1;
    /// `I_AS` (`As`).
    pub const AS: usize = 2;
    /// `I_U` (`U`).
    pub const U: usize = 3;
    /// `I_C_1` (`c_1`).
    pub const C_1: usize = 4;
    /// `I_GAMMA_1` (`Gamma_1`).
    pub const GAMMA_1: usize = 5;
    /// `I_UPS_T` (`ups_T`).
    pub const UPS_T: usize = 6;
    /// `I_NABLA_AD` (`nabla_ad`).
    pub const NABLA_AD: usize = 7;
    /// `I_NABLA` (`nabla`).
    pub const NABLA: usize = 8;
    /// `I_BETA_RAD` (`beta_rad`).
    pub const BETA_RAD: usize = 9;
    /// `I_C_LUM` (`c_lum`).
    pub const C_LUM: usize = 10;
    /// `I_C_RAD` (`c_rad`).
    pub const C_RAD: usize = 11;
    /// `I_C_THN` (`c_thn`).
    pub const C_THN: usize = 12;
    /// `I_C_THK` (`c_thk`).
    pub const C_THK: usize = 13;
    /// `I_C_EPS` (`c_eps`).
    pub const C_EPS: usize = 14;
    /// `I_C_EGV` (`c_egv`).
    pub const C_EGV: usize = 15;
    /// `I_EPS_RHO` (`eps_rho`).
    pub const EPS_RHO: usize = 16;
    /// `I_EPS_T` (`eps_T`).
    pub const EPS_T: usize = 17;
    /// `I_KAP_RHO` (`kap_rho`).
    pub const KAP_RHO: usize = 18;
    /// `I_KAP_T` (`kap_T`).
    pub const KAP_T: usize = 19;
    /// `I_OMEGA_ROT` (`Omega_rot`).
    pub const OMEGA_ROT: usize = 20;
    /// `I_LAST`.
    pub const LAST: usize = 20;
}

fn p2(x: f64) -> f64 {
    x * x
}

fn p3(x: f64) -> f64 {
    // powi_as_mults chain for 3 (see `powi.rs`).
    let x2 = x * x;
    x * x2
}

fn p4(x: f64) -> f64 {
    let x2 = x * x;
    x2 * x2
}

/// `dphi_dt_(t, 0)` (`cubic_interp_m.fypp:305-334`).
fn dphi_dt_0(t: f64) -> f64 {
    3.0 * p2(t) - 2.0 * p3(t)
}

/// `dpsi_dt_(t, 0)` (`cubic_interp_m.fypp:338-367`).
fn dpsi_dt_0(t: f64) -> f64 {
    p3(t) - p2(t)
}

/// `cubic_interp_rt` (`cubic_interp_m.fypp:85-111`).
#[derive(Debug, Clone, PartialEq)]
pub struct CubicInterp {
    x: Vec<f64>,
    f: Vec<f64>,
    df_dx: Vec<f64>,
}

impl CubicInterp {
    /// Constructor; the abscissa must increase strictly.
    ///
    /// # Errors
    ///
    /// [`GyreError::Assert`] for a non-monotonic abscissa.
    pub fn new(x: &[f64], f: &[f64], df_dx: &[f64]) -> Result<Self, GyreError> {
        if !x.windows(2).all(|w| w[1] > w[0]) {
            return Err(GyreError::Assert("non-monotonic abscissa"));
        }
        Ok(Self {
            x: x.to_vec(),
            f: f.to_vec(),
            df_dx: df_dx.to_vec(),
        })
    }

    /// `eval_f` (`cubic_interp_m.fypp:219-258`). Upstream keeps the
    /// `locate` guess in a `SAVE` variable; the bracket `locate` returns
    /// does not depend on the guess for a strictly increasing abscissa.
    ///
    /// # Errors
    ///
    /// [`GyreError::Assert`] for out-of-bounds interpolation.
    pub fn eval_f(&self, x: f64) -> Result<f64, GyreError> {
        let n = self.x.len();
        let mut i: i32 = 1;
        locate(&self.x, x, &mut i, false);
        let n_i =
            i32::try_from(n).map_err(|_| GyreError::Assert("grid too long"))?;
        if !(i >= 1 && i <= n_i) {
            return Err(GyreError::Assert("out-of-bounds interpolation"));
        }
        let j = usize::try_from(i.min(n_i - 1) - 1)
            .map_err(|_| GyreError::Assert("out-of-bounds interpolation"))?;
        let h = self.x[j + 1] - self.x[j];
        let w = (x - self.x[j]) / h;
        let hm1 = 1.0 / h;
        Ok(
            self.f[j] * dphi_dt_0(1.0 - w) + self.f[j + 1] * dphi_dt_0(w)
                - self.df_dx[j] * dpsi_dt_0(1.0 - w) / hm1
                + self.df_dx[j + 1] * dpsi_dt_0(w) / hm1,
        )
    }
}

/// `deriv(x, f, 'MONO', df_dx_a, df_dx_b)` (`deriv_mono_r_`,
/// `deriv_m.fypp:212-270`).
#[must_use]
pub fn deriv_mono(
    x: &[f64],
    f: &[f64],
    df_dx_a: Option<f64>,
    df_dx_b: Option<f64>,
) -> Vec<f64> {
    let n = x.len();
    let h: Vec<f64> = (0..n - 1).map(|i| x[i + 1] - x[i]).collect();
    let s: Vec<f64> = (0..n - 1).map(|i| (f[i + 1] - f[i]) / h[i]).collect();
    let mut p = vec![0.0; n];
    p[0] = df_dx_a.unwrap_or(s[0]);
    for i in 1..n - 1 {
        p[i] = (s[i - 1] * h[i] + s[i] * h[i - 1]) / (h[i - 1] + h[i]);
    }
    p[n - 1] = df_dx_b.unwrap_or(s[n - 2]);
    let mut df_dx = vec![0.0; n];
    df_dx[0] = p[0];
    for i in 1..n - 1 {
        df_dx[i] = (1.0_f64.copysign(s[i - 1]) + 1.0_f64.copysign(s[i]))
            * fmin(fmin(s[i - 1].abs(), s[i].abs()), 0.5 * p[i].abs());
    }
    df_dx[n - 1] = p[n - 1];
    df_dx
}

/// `snap_points(dx_snap, x, m)` (`model_util_m.fypp:52-112`).
pub fn snap_points(dx_snap: f64, x: &mut [f64], m: &mut [f64]) {
    if x[0] > 0.0 && x[0] < dx_snap {
        x[0] = 0.0;
        m[0] = 0.0;
    }
    for i in 1..x.len() - 1 {
        if x[i + 1] - x[i] > 0.0 && x[i + 1] - x[i] < dx_snap {
            let x_snap = 0.5 * (x[i + 1] + x[i]);
            x[i] = x_snap;
            x[i + 1] = x_snap;
            let m_snap = 0.5 * (m[i + 1] + m[i]);
            m[i] = m_snap;
            m[i + 1] = m_snap;
        }
    }
}

/// A committed `evol_model_t`.
#[derive(Debug, Clone)]
pub struct EvolModel {
    /// Model grid `gr`.
    pub gr: Grid,
    /// `M_star` (g).
    pub m_star: f64,
    /// `R_star` (cm).
    pub r_star: f64,
    /// `L_star` (erg/s).
    pub l_star: f64,
    constrain_derivs: bool,
    s_i: i32,
    defined: [bool; idx::LAST + 1],
    /// Interpolants `in(s, i)`, indexed `[s - s_i][i]`.
    interp: Vec<Vec<Option<CubicInterp>>>,
}

/// Builder state of `evol_model_t` before `commit`.
struct EvolBuilder {
    gr: Grid,
    add_center: bool,
    coeffs: Vec<Vec<f64>>,
    defined: [bool; idx::LAST + 1],
    m_star: f64,
    r_star: f64,
    l_star: f64,
    repair_as: bool,
    constrain_derivs: bool,
    interp_type: String,
    deriv_type: String,
}

impl EvolBuilder {
    /// `evol_model_t(x, M_star, R_star, L_star, ml_p)`
    /// (`evol_model_m.fypp:96-167`).
    fn new(
        x: &[f64],
        m_star: f64,
        r_star: f64,
        l_star: f64,
        ml_p: &ModelPar,
    ) -> Result<Self, GyreError> {
        let (gr, add_center) = if ml_p.add_center && x[0] != 0.0 {
            return Err(GyreError::Unported {
                what: "evol_model_t: adding a central point",
                at: "model/evol_model_m.fypp:111-119",
            });
        } else {
            (Grid::from_x(x)?, false)
        };
        Ok(Self {
            coeffs: vec![Vec::new(); idx::LAST + 1],
            gr,
            add_center,
            defined: [false; idx::LAST + 1],
            m_star,
            r_star,
            l_star,
            repair_as: ml_p.repair_as,
            constrain_derivs: ml_p.constrain_derivs,
            interp_type: ml_p.interp_type.clone(),
            deriv_type: ml_p.deriv_type.clone(),
        })
    }

    /// `define(i, coeff)` (`evol_model_m.fypp:171-234`).
    fn define(&mut self, i: usize, coeff: Vec<f64>) -> Result<(), GyreError> {
        if self.add_center {
            return Err(GyreError::Unported {
                what: "evol_model_t%define with an added central point",
                at: "model/evol_model_m.fypp:185-211",
            });
        }
        self.coeffs[i] = coeff;
        self.defined[i] = true;
        if i == idx::AS && self.repair_as {
            return Err(GyreError::Unported {
                what: "repair_As",
                at: "model/evol_model_m.fypp:227",
            });
        }
        Ok(())
    }

    /// `commit()` with `cubic_interp_()` (`evol_model_m.fypp:280-413`).
    fn commit(self) -> Result<EvolModel, GyreError> {
        match self.interp_type.as_str() {
            "CUBIC" => {}
            "LINEAR" => {
                return Err(GyreError::Unported {
                    what: "interp_type = 'LINEAR'",
                    at: "model/evol_model_m.fypp:289-290",
                })
            }
            _ => return Err(GyreError::Abort("invalid deriv_type")),
        }
        let s_i = self.gr.s_i();
        let s_o = self.gr.s_o();
        let mut interp = Vec::new();
        for s in s_i..=s_o {
            let j_i = self.gr.j_s_i(s);
            let j_o = self.gr.j_s_o(s);
            let x: Vec<f64> =
                self.gr.pt[j_i - 1..j_o].iter().map(|p| p.x).collect();
            let mut row: Vec<Option<CubicInterp>> = vec![None; idx::LAST + 1];
            for i in 1..=idx::LAST {
                if !self.defined[i] {
                    continue;
                }
                let coeff = &self.coeffs[i][j_i - 1..j_o];
                let dcoeff_dx = match self.deriv_type.as_str() {
                    "MONO" => deriv_mono(
                        &x,
                        coeff,
                        (x[0] == 0.0).then_some(0.0),
                        None,
                    ),
                    "SPLINE" | "FINDIFF" => {
                        return Err(GyreError::Unported {
                            what: "deriv_type other than 'MONO'",
                            at: "interp/deriv_m.fypp:69-72",
                        })
                    }
                    _ => return Err(GyreError::Abort("invalid deriv_type")),
                };
                row[i] = Some(CubicInterp::new(&x, coeff, &dcoeff_dx)?);
            }
            interp.push(row);
        }
        // Rows are stored per segment s in s_i..=s_o; the loop above runs
        // coefficient-major upstream, which only orders independent work.
        Ok(EvolModel {
            gr: self.gr,
            m_star: self.m_star,
            r_star: self.r_star,
            l_star: self.l_star,
            constrain_derivs: self.constrain_derivs,
            s_i,
            defined: self.defined,
            interp,
        })
    }
}

impl EvolModel {
    /// `init_mesa_model(ml_p, global_data, point_data, version, em)`
    /// (`mesa_file_m.fypp:143-296`). `point_data` is column-major
    /// `point_data(1:n_cols, 1:n)`, given here as one row of columns per
    /// point (`point_data[k][c]` is `point_data(c+1, k+1)`).
    ///
    /// # Errors
    ///
    /// Unported schema versions and the paths listed in [`GyreError`].
    #[allow(clippy::too_many_lines)]
    pub fn from_mesa(
        ml_p: &ModelPar,
        global_data: &[f64],
        point_data: &[Vec<f64>],
        version: i32,
        k: &GyreConstants,
    ) -> Result<Self, GyreError> {
        let m_star = global_data[0];
        let r_star = global_data[1];
        let l_star = global_data[2];
        let n = point_data.len();
        if !(version == 100 || version == 101) {
            return Err(match version {
                1 | 19 | 120 => GyreError::Unported {
                    what: "MESA pulsation data versions other than 1.00/1.01",
                    at: "model/mesa_file_m.fypp:197-204",
                },
                _ => GyreError::Abort("unrecognized MESA memory version"),
            });
        }
        // extract_data_v1_0X_ (mesa_file_m.fypp:408-458)
        if point_data.iter().any(|row| row.len() != 18) {
            return Err(GyreError::Assert(
                "point_data must have 18 columns (N_COLS_V1_0X)",
            ));
        }
        let col = |c: usize| -> Vec<f64> {
            point_data.iter().map(|row| row[c - 1]).collect()
        };
        let r = col(1);
        let mut m_r = col(2);
        let l_r = col(3);
        let p = col(4);
        let t = col(5);
        let rho = col(6);
        let nabla = col(7);
        let n2 = col(8);
        let gamma_1 = col(9);
        let nabla_ad = col(10);
        let ups_t = col(11);
        let kap = col(12);
        let kap_kap_t = col(13);
        let kap_kap_rho = col(14);
        let eps = col(15);
        let eps_eps_t = col(16);
        let eps_eps_rho = col(17);
        let mut omega_rot = col(18);
        let mut eps_rho = vec![0.0; n];
        let mut eps_t = vec![0.0; n];
        for j in 0..n {
            if eps[j] != 0.0 {
                eps_rho[j] = eps_eps_rho[j] / eps[j];
                eps_t[j] = eps_eps_t[j] / eps[j];
            } else {
                eps_rho[j] = 0.0;
                eps_t[j] = 0.0;
            }
        }
        let kap_t: Vec<f64> = (0..n).map(|j| kap_kap_t[j] / kap[j]).collect();
        let kap_rho: Vec<f64> =
            (0..n).map(|j| kap_kap_rho[j] / kap[j]).collect();

        let mut x: Vec<f64> = r.iter().map(|&rr| rr / r_star).collect();
        snap_points(
            crate::grid::fmax(ml_p.dx_snap, f64::EPSILON),
            &mut x,
            &mut m_r,
        );

        let pi = k.math.pi;
        let (g, a_rad, c_light) = (k.g_gravity, k.a_radiation, k.c_light);
        let mut v_2 = vec![0.0; n];
        let mut a_s = vec![0.0; n];
        let mut u = vec![0.0; n];
        let mut c_1 = vec![0.0; n];
        let mut c_lum = vec![0.0; n];
        for j in 0..n {
            if x[j] != 0.0 {
                v_2[j] = g * m_r[j] * rho[j] / (p[j] * r[j] * p2(x[j]));
                a_s[j] = p3(r[j]) * n2[j] / (g * m_r[j]);
                u[j] = 4.0 * pi * rho[j] * p3(r[j]) / m_r[j];
                c_1[j] = p3(r[j] / r_star) / (m_r[j] / m_star);
                c_lum[j] = (l_r[j] / l_star) / p3(x[j]);
            } else {
                v_2[j] = 4.0 * pi * g * p2(rho[0]) * p2(r_star) / (3.0 * p[0]);
                a_s[j] = 0.0;
                u[j] = 3.0;
                c_1[j] = 3.0 * (m_star / p3(r_star)) / (4.0 * pi * rho[j]);
                c_lum[j] = 4.0 * pi * rho[0] * eps[0] * p3(r_star) / l_star;
            }
        }
        let beta_rad: Vec<f64> =
            (0..n).map(|j| a_rad * p4(t[j]) / (3.0 * p[j])).collect();
        let c_p: Vec<f64> = (0..n)
            .map(|j| p[j] * ups_t[j] / (rho[j] * t[j] * nabla_ad[j]))
            .collect();
        let c_rad: Vec<f64> = (0..n)
            .map(|j| {
                16.0 * pi
                    * a_rad
                    * c_light
                    * p4(t[j])
                    * r_star
                    * nabla[j]
                    * v_2[j]
                    / (3.0 * kap[j] * rho[j] * l_star)
            })
            .collect();
        let sq = (g * m_star / p3(r_star)).sqrt();
        let c_thn: Vec<f64> = (0..n)
            .map(|j| c_p[j] * sq / (a_rad * c_light * kap[j] * p3(t[j])))
            .collect();
        let c_thk: Vec<f64> = (0..n)
            .map(|j| {
                4.0 * pi * rho[j] * t[j] * c_p[j] * sq * p3(r_star) / l_star
            })
            .collect();
        let c_eps: Vec<f64> = (0..n)
            .map(|j| 4.0 * pi * rho[j] * eps[j] * p3(r_star) / l_star)
            .collect();
        let c_egv = vec![0.0; n];
        let sq_o = (p3(r_star) / (g * m_star)).sqrt();
        for o in &mut omega_rot {
            *o *= sq_o;
        }

        let mut em = EvolBuilder::new(&x, m_star, r_star, l_star, ml_p)?;
        em.define(idx::V_2, v_2)?;
        em.define(idx::AS, a_s)?;
        em.define(idx::U, u)?;
        em.define(idx::C_1, c_1)?;
        em.define(idx::GAMMA_1, gamma_1)?;
        em.define(idx::UPS_T, ups_t)?;
        em.define(idx::NABLA_AD, nabla_ad)?;
        em.define(idx::NABLA, nabla)?;
        em.define(idx::BETA_RAD, beta_rad)?;
        em.define(idx::C_LUM, c_lum)?;
        em.define(idx::C_RAD, c_rad)?;
        em.define(idx::C_THN, c_thn)?;
        em.define(idx::C_THK, c_thk)?;
        em.define(idx::C_EPS, c_eps)?;
        em.define(idx::C_EGV, c_egv)?;
        em.define(idx::EPS_RHO, eps_rho)?;
        em.define(idx::EPS_T, eps_t)?;
        em.define(idx::KAP_RHO, kap_rho)?;
        em.define(idx::KAP_T, kap_t)?;
        em.define(idx::OMEGA_ROT, omega_rot)?;
        em.commit()
    }

    /// `is_defined(i)`.
    #[must_use]
    pub fn is_defined(&self, i: usize) -> bool {
        self.defined[i]
    }

    /// `coeff(i, pt)` (`evol_model_m.fypp:419-446`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Assert`] for out-of-bounds interpolation or an
    /// undefined coefficient.
    pub fn coeff(&self, i: usize, pt: Point) -> Result<f64, GyreError> {
        let s = usize::try_from(pt.s - self.s_i)
            .map_err(|_| GyreError::Assert("invalid segment"))?;
        let ci = self
            .interp
            .get(s)
            .and_then(|row| row[i].as_ref())
            .ok_or(GyreError::Assert("undefined coefficient"))?;
        ci.eval_f(pt.x)
    }

    /// `dcoeff(i, pt)` (`evol_model_m.fypp:450-537`), the constrained
    /// branches.
    ///
    /// # Errors
    ///
    /// As [`Self::coeff`]; [`GyreError::Unported`] for the unconstrained
    /// derivative path.
    pub fn dcoeff(&self, i: usize, pt: Point) -> Result<f64, GyreError> {
        let x = pt.x;
        if self.constrain_derivs
            && (i == idx::V_2 || i == idx::U || i == idx::C_1)
        {
            match i {
                idx::V_2 => {
                    let v_2 = self.coeff(idx::V_2, pt)?;
                    let a_s = self.coeff(idx::AS, pt)?;
                    let u = self.coeff(idx::U, pt)?;
                    let gamma_1 = self.coeff(idx::GAMMA_1, pt)?;
                    Ok(v_2 * p2(x) - v_2 * p2(x) / gamma_1 - a_s + u - 3.0)
                }
                idx::U => {
                    let v_2 = self.coeff(idx::V_2, pt)?;
                    let a_s = self.coeff(idx::AS, pt)?;
                    let u = self.coeff(idx::U, pt)?;
                    let gamma_1 = self.coeff(idx::GAMMA_1, pt)?;
                    Ok(-(v_2 * p2(x) / gamma_1) - a_s - u + 3.0)
                }
                _ => {
                    let u = self.coeff(idx::U, pt)?;
                    Ok(3.0 - u)
                }
            }
        } else {
            Err(GyreError::Unported {
                what: "evol_model_t%dcoeff without constrained derivatives",
                at: "model/evol_model_m.fypp:519-529",
            })
        }
    }

    /// `is_vacuum(pt)` (`evol_model_m.fypp:634-650`).
    ///
    /// # Errors
    ///
    /// As [`Self::coeff`].
    pub fn is_vacuum(&self, pt: Point) -> Result<bool, GyreError> {
        Ok(self.coeff(idx::U, pt)? == 0.0)
    }

    /// `check_model(ml, i_req)` (`model_util_m.fypp:116-154`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Abort`] when a required coefficient is missing.
    pub fn check(&self, i_req: &[usize]) -> Result<(), GyreError> {
        if i_req.iter().any(|&i| !self.is_defined(i)) {
            return Err(GyreError::Abort("model check failed"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::powi::powi_const;

    #[test]
    fn local_power_helpers_equal_the_verified_gcc_chains() {
        for &x in &[0.3_f64, 1.7, -2.9, 1e-100, 3.3e77] {
            assert_eq!(p2(x).to_bits(), powi_const(x, 2).unwrap().to_bits());
            assert_eq!(p3(x).to_bits(), powi_const(x, 3).unwrap().to_bits());
            assert_eq!(p4(x).to_bits(), powi_const(x, 4).unwrap().to_bits());
        }
    }
}
