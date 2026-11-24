// SPDX-License-Identifier: GPL-3.0-only
//! GYRE `context_t` (`context/context_m.fypp`), the null rotation model
//! (`rot/null_rot_m.fypp`, `rot/rot_factory_m.fypp`) and frequency scales
//! (`common/freq_m.fypp`, `common/freq_context_m.fypp`).

use crate::constants::GyreConstants;
use crate::fcomplex::Cplx;
use crate::grid::{Grid, Point};
use crate::model::{idx, EvolModel};
use crate::par::{ModePar, OscPar, RotPar};
use crate::powi::powi_const;
use crate::GyreError;

/// `context_t`: model, end points, azimuthal order and rotation model.
#[derive(Debug, Clone)]
pub struct Context<'a> {
    /// The model `ml`.
    pub ml: &'a EvolModel,
    /// Constants (`constants_m`, `math_m`).
    pub k: &'a GyreConstants,
    /// Inner point `pt_i`.
    pub pt_i: Point,
    /// Outer point `pt_o`.
    pub pt_o: Point,
    /// Azimuthal order `m`.
    pub m: i32,
    /// `l` of the null rotation model `rt`.
    l: i32,
    complex_lambda: bool,
}

impl<'a> Context<'a> {
    /// `context_t(ml, gr, md_p, os_p, rt_p)` (`context_m.fypp:100-146`),
    /// with `rot_ct(md_p, rt_p)` (`rot_factory_m.fypp:60-81`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for rotation models and sources other than
    /// `NULL` and `MODEL`; [`GyreError::Abort`] for invalid options.
    pub fn new(
        ml: &'a EvolModel,
        k: &'a GyreConstants,
        gr: &Grid,
        md_p: &ModePar,
        os_p: &OscPar,
        rt_p: &RotPar,
    ) -> Result<Self, GyreError> {
        match rt_p.coriolis_method.as_str() {
            "NULL" => {}
            "TAR" => {
                return Err(GyreError::Unported {
                    what: "coriolis_method = 'TAR'",
                    at: "rot/rot_factory_m.fypp:71-72",
                })
            }
            _ => return Err(GyreError::Abort("invalid coriolis_method")),
        }
        match rt_p.omega_rot_source.as_str() {
            "MODEL" => {}
            "UNIFORM" => {
                return Err(GyreError::Unported {
                    what: "Omega_rot_source = 'UNIFORM'",
                    at: "context/context_m.fypp:125-127",
                })
            }
            _ => return Err(GyreError::Abort("invalid Omega_rot_source")),
        }
        match os_p.deps_source.as_str() {
            "MODEL" => {}
            "FILE" => {
                return Err(GyreError::Unported {
                    what: "deps_source = 'FILE'",
                    at: "context/context_m.fypp:135-137",
                })
            }
            _ => return Err(GyreError::Abort("invalid deps_source")),
        }
        Ok(Self {
            ml,
            k,
            pt_i: gr.pt_i(),
            pt_o: gr.pt_o(),
            m: md_p.m,
            l: md_p.l,
            complex_lambda: rt_p.complex_lambda,
        })
    }

    /// `omega_c(Omega_rot, st)` for a real state (`omega_c_r_`).
    #[must_use]
    pub fn omega_c_r(&self, omega_rot: f64, omega: f64) -> f64 {
        omega - f64::from(self.m) * omega_rot
    }

    /// `omega_c(Omega_rot, st)` for a complex state (`omega_c_c_`,
    /// `context_m.fypp:220-248`, `use_omega_r` absent):
    /// `st%omega - m*Omega_rot`.
    #[must_use]
    pub fn omega_c_c(&self, omega_rot: f64, omega: Cplx) -> Cplx {
        omega - Cplx::from_real(f64::from(self.m) * omega_rot)
    }

    /// `eps_rho(st, pt)` for a complex state (`eps_rho_c_`,
    /// `context_m.fypp:398-444`), model source.
    ///
    /// # Errors
    ///
    /// Model interpolation errors.
    pub fn eps_rho_c(&self, pt: Point) -> Result<Cplx, GyreError> {
        Ok(Cplx::from_real(self.ml.coeff(idx::EPS_RHO, pt)?))
    }

    /// `eps_T(st, pt)` for a complex state (`eps_t_c_`), model source.
    ///
    /// # Errors
    ///
    /// Model interpolation errors.
    pub fn eps_t_c(&self, pt: Point) -> Result<Cplx, GyreError> {
        Ok(Cplx::from_real(self.ml.coeff(idx::EPS_T, pt)?))
    }

    /// `null_rot_ct%lambda`: `l*(l+1)` (integer product, then complex).
    fn rt_lambda(&self) -> Cplx {
        Cplx::new(f64::from(self.l * (self.l + 1)), 0.0)
    }

    /// `null_rot_ct%l_e`: `l`.
    fn rt_l_e(&self) -> Cplx {
        Cplx::new(f64::from(self.l), 0.0)
    }

    /// `lambda(Omega_rot, st)` for a real state (`lambda_r_`).
    #[must_use]
    pub fn lambda_r(&self, _omega_rot: f64, _omega: f64) -> f64 {
        self.rt_lambda().re
    }

    /// `lambda(Omega_rot, st)` for a complex state (`lambda_c_`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for `complex_lambda`.
    pub fn lambda_c(&self, _omega_rot: f64) -> Result<Cplx, GyreError> {
        if self.complex_lambda {
            return Err(GyreError::Unported {
                what: "complex_lambda",
                at: "context/context_m.fypp:280-284",
            });
        }
        Ok(self.rt_lambda())
    }

    /// `l_e(Omega_rot, st)` for a real state (`l_e_r_`).
    #[must_use]
    pub fn l_e_r(&self, _omega_rot: f64, _omega: f64) -> f64 {
        self.rt_l_e().re
    }

    /// `l_e(Omega_rot, st)` for a complex state (`l_e_c_`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for `complex_lambda`.
    pub fn l_e_c(&self, _omega_rot: f64) -> Result<Cplx, GyreError> {
        if self.complex_lambda {
            return Err(GyreError::Unported {
                what: "complex_lambda",
                at: "context/context_m.fypp:326-330",
            });
        }
        Ok(self.rt_l_e())
    }

    /// `Omega_rot(pt)` (`context_m.fypp:344-372`), model source.
    ///
    /// # Errors
    ///
    /// Model interpolation errors.
    pub fn omega_rot(&self, pt: Point) -> Result<f64, GyreError> {
        self.ml.coeff(idx::OMEGA_ROT, pt)
    }

    /// `freq_scale(units, cx, md_p, os_p)` (`freq_context_m.fypp:64-142`,
    /// `freq_m.fypp:81-131` and the model-free `freq_scale_`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for the context-dependent unit systems;
    /// [`GyreError::Abort`] for invalid units.
    pub fn freq_scale(&self, units: &str) -> Result<f64, GyreError> {
        match units {
            // freq_context_m.fypp:86-93
            "ACOUSTIC_DELTA" => Ok(1.0
                / (self.k.math.twopi
                    * self.ml.delta_p(self.pt_i.x, self.pt_o.x)?)),
            "GRAVITY_DELTA" => {
                let l = f64::from(self.l);
                let dg =
                    self.ml.delta_g(self.pt_i.x, self.pt_o.x, l * (l + 1.0))?;
                Ok(1.0 / (self.k.math.twopi * dg))
            }
            "UPPER_DELTA" => {
                let l = f64::from(self.l);
                let dp = self.ml.delta_p(self.pt_i.x, self.pt_o.x)?;
                let dg =
                    self.ml.delta_g(self.pt_i.x, self.pt_o.x, l * (l + 1.0))?;
                Ok(1.0 / (self.k.math.twopi * dp.max(dg)))
            }
            "LOWER_DELTA" => {
                let l = f64::from(self.l);
                let dp = self.ml.delta_p(self.pt_i.x, self.pt_o.x)?;
                let dg =
                    self.ml.delta_g(self.pt_i.x, self.pt_o.x, l * (l + 1.0))?;
                Ok(1.0 / (self.k.math.twopi * dp.min(dg)))
            }
            "ROSSBY_I" | "ROSSBY_O" | "ACOUSTIC_CUTOFF" | "GRAVITY_CUTOFF" => {
                Err(GyreError::Unported {
                    what: "frequency units depending on the context",
                    at: "common/freq_context_m.fypp:94-133",
                })
            }
            _ => {
                let ml = self.ml;
                let twopi = self.k.math.twopi;
                let base = || {
                    Ok::<f64, GyreError>(
                        (self.k.g_gravity * ml.m_star
                            / powi_const(ml.r_star, 3)?)
                        .sqrt(),
                    )
                };
                Ok(match units {
                    "HZ" => base()? / twopi,
                    "UHZ" => base()? * 1e6 / twopi,
                    "RAD_PER_SEC" => base()?,
                    "CYC_PER_DAY" => base()? * 86400.0 / twopi,
                    "NONE" => 1.0,
                    "CRITICAL" => (27.0_f64 / 8.0).sqrt(),
                    _ => {
                        return Err(GyreError::Abort("Invalid frequency units"))
                    }
                })
            }
        }
    }

    /// `freq_shift(frame, cx, md_p)` (`freq_context_m.fypp:146-178`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Unported`] for co-rotating frames.
    pub fn freq_shift(&self, frame: &str) -> Result<f64, GyreError> {
        match frame {
            "INERTIAL" => Ok(0.0),
            "COROT_I" | "COROT_O" => Err(GyreError::Unported {
                what: "co-rotating frequency frames",
                at: "common/freq_context_m.fypp:166-169",
            }),
            _ => Err(GyreError::Abort("Invalid frequency frame")),
        }
    }
}
