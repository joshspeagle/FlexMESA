// SPDX-License-Identifier: LGPL-3.0-or-later
//! Local Mixing-Length Theory (`set_mlt`, `turb/private/mlt.f90`).
//!
//! Solves the cubic convective efficiency equation (Cox & Giuli 1968, §14.82):
//! $$a_0 \Gamma^3 + \Gamma^2 + \Gamma - a_0 B^3 = 0$$
//! across six geometric MLT prescriptions:
//! - [`MltOption::Cox`] (and [`MltOption::Tdc`] fallback alias): optically
//!   thick Cox & Giuli (1968) prescription with $a_0 = 9/4$.
//! - [`MltOption::Henyey`]: Henyey, Vardya, & Bodenheimer (1965) with tunable
//!   parameters $(\nu, y)$.
//! - [`MltOption::Ml1`]: Bohm-Vitense (1958) standard parametrization
//!   $(f_1, f_2, f_3, f_4) = (1/8, 1/2, 24, 0)$.
//! - [`MltOption::Ml2`]: Bohm & Cassinelli (1971) parametrization
//!   $(f_1, f_2, f_3, f_4) = (1, 2, 16, 0)$.
//! - [`MltOption::Mihalas`]: Mihalas (1978) optically thin/thick bridged
//!   parametrization $(f_1, f_2, f_3, f_4) = (1/8, 1/2, 16, 2)$.

use std::str::FromStr;

use flexmesa_const::{MixingType, CLIGHT, CRAD, FOUR_13, ONE_THIRD, TWO_13};
use flexmesa_utils::is_bad;

use crate::{TurbError, TurbOutput, TurbResult, TurbScalar};

/// Geometric Mixing-Length Theory prescription (`MLT_option` in `mlt.f90`).
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum MltOption {
    /// Cox & Giuli (1968) optically thick prescription ($a_0 = 9/4$).
    Cox,
    /// Alias for [`MltOption::Cox`] used when falling back from TDC to MLT.
    Tdc,
    /// Henyey, Vardya, & Bodenheimer (1965) prescription with $(\nu, y)$.
    Henyey,
    /// Bohm-Vitense (1958) standard ML1 prescription.
    Ml1,
    /// Bohm & Cassinelli (1971) ML2 prescription.
    Ml2,
    /// Mihalas (1978) prescription including optical depth factor $f_4 = 2$.
    Mihalas,
}

impl FromStr for MltOption {
    type Err = TurbError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "Cox" => Ok(Self::Cox),
            "TDC" => Ok(Self::Tdc),
            "Henyey" => Ok(Self::Henyey),
            "ML1" => Ok(Self::Ml1),
            "ML2" => Ok(Self::Ml2),
            "Mihalas" => Ok(Self::Mihalas),
            other => Err(TurbError::UnknownMltOption(other.to_string())),
        }
    }
}

/// Non-differentiable configuration parameters for [`set_mlt`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MltParams {
    /// Geometric MLT prescription ([`MltOption`]).
    pub option: MltOption,
    /// Mixing-length parameter $\alpha_{\mathrm{MLT}}$ (dimensionless).
    pub mixing_length_alpha: f64,
    /// Henyey velocity parameter $\nu$ (when `option == MltOption::Henyey`).
    pub henyey_nu: f64,
    /// Henyey optical-depth parameter $y$ (when `option == MltOption::Henyey`).
    pub henyey_y: f64,
    /// Upper limit on convective velocity $v_{\mathrm{conv},\max}$ in
    /// $\mathrm{cm\,s^{-1}}$.
    pub max_conv_vel: f64,
}

impl MltParams {
    /// Creates standard Cox MLT parameters with the given
    /// $\alpha_{\mathrm{MLT}}$.
    #[must_use]
    pub const fn cox(mixing_length_alpha: f64) -> Self {
        Self {
            option: MltOption::Cox,
            mixing_length_alpha,
            henyey_nu: 8.0,
            henyey_y: ONE_THIRD,
            max_conv_vel: 1.0e99,
        }
    }
}

/// Differentiable thermodynamic and structure inputs for [`set_mlt`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MltInputs<T> {
    /// Logarithmic pressure derivative
    /// $\chi_T = (\partial \ln P / \partial \ln T)_\rho$.
    pub chi_t: T,
    /// Logarithmic pressure derivative
    /// $\chi_\rho = (\partial \ln P / \partial \ln \rho)_T$.
    pub chi_rho: T,
    /// Specific heat at constant pressure $C_P$ in
    /// $\mathrm{erg\,g^{-1}\,K^{-1}}$.
    pub cp: T,
    /// Local gravitational acceleration $g$ in $\mathrm{cm\,s^{-2}}$.
    pub grav: T,
    /// Mixing length $\Lambda = \alpha_{\mathrm{MLT}} H_P$ in $\mathrm{cm}$.
    pub lambda: T,
    /// Mass density $\rho$ in $\mathrm{g\,cm^{-3}}$.
    pub rho: T,
    /// Total pressure $P$ in $\mathrm{dyn\,cm^{-2}}$.
    pub p: T,
    /// Temperature $T$ in $\mathrm{K}$.
    pub t: T,
    /// Rosseland mean opacity $\kappa$ in $\mathrm{cm^2\,g^{-1}}$.
    pub opacity: T,
    /// Radiative temperature gradient $\nabla_{\mathrm{rad}}$ (dimensionless).
    pub gradr: T,
    /// Adiabatic temperature gradient $\nabla_{\mathrm{ad}}$ (dimensionless).
    pub grada: T,
    /// Ledoux composition-adjusted gradient $\nabla_L$ (dimensionless).
    pub gradl: T,
}

/// Evaluates local Mixing-Length Theory (`calc_MLT` in `turb/private/mlt.f90`).
///
/// Supports both scalar `f64` and `DualOrder1<N>` inputs via [`TurbScalar`].
///
/// Upstream `turb/private/mlt.f90:53-181` never assigns `ierr`, and
/// `star/private/turb_support.f90:282-298` initializes the outputs to the
/// radiative no-mixing state (`Gamma = 0`, `gradT = gradr`,
/// `Y_face = gradr - gradL`, `conv_vel = 0`, `D = 0`, `mixing_type = no_mixing`,
/// `ierr = 0`) before calling `set_MLT`. Each bare `return` in `calc_MLT`
/// (`f1 <= 0` at `mlt.f90:140`, `Gamma <= 0` at `mlt.f90:145`, and
/// `is_bad(Zeta)` at `mlt.f90:169`) therefore returns `Ok` with whichever
/// outputs have already been written and the caller's no-mixing baseline for
/// the rest. Adding explicit input validation on degenerate MLT inputs is
/// deferred to `#330`.
///
/// # Errors
/// Currently always returns `Ok` (`ierr = 0`), matching upstream `calc_MLT`'s
/// effective runtime behavior (`#330`).
pub fn set_mlt<T: TurbScalar>(
    params: &MltParams,
    inp: &MltInputs<T>,
) -> TurbResult<TurbOutput<T>> {
    let zero = T::from_f64(0.0);
    let one = T::from_f64(1.0);

    let (gamma, mut zeta, conv_vel, d) = if inp.gradr.val() > inp.gradl.val() {
        let q = inp.chi_t / inp.chi_rho;
        let (a0, a_val) = match params.option {
            MltOption::Cox | MltOption::Tdc => {
                let a0_c = T::from_f64(9.0 / 4.0);
                // `.../9d0` divides by a real (`turb/private/mlt.f90:80`).
                let conv_cond = (inp.cp
                    * inp.grav
                    * inp.lambda.pow2()
                    * inp.rho
                    * (q * inp.rho / (T::from_f64(2.0) * inp.p)).sqrt())
                .div_real(9.0);
                let rad_cond = T::from_f64(4.0 / 3.0 * CRAD * CLIGHT)
                    * inp.t.pow3()
                    / (inp.opacity * inp.rho);
                (a0_c, conv_cond / rad_cond)
            }
            MltOption::Henyey
            | MltOption::Ml1
            | MltOption::Ml2
            | MltOption::Mihalas => {
                let (ff1, ff2, ff3, ff4) = match params.option {
                    MltOption::Henyey => (
                        1.0 / params.henyey_nu,
                        0.5,
                        8.0 / params.henyey_y,
                        1.0 / params.henyey_y,
                    ),
                    MltOption::Ml1 => (0.125, 0.5, 24.0, 0.0),
                    MltOption::Ml2 => (1.0, 2.0, 16.0, 0.0),
                    MltOption::Mihalas => (0.125, 0.5, 16.0, 2.0),
                    MltOption::Cox | MltOption::Tdc => unreachable!(),
                };
                let omega = inp.lambda * inp.rho * inp.opacity;
                let ff4_omega2_plus_1 = T::from_f64(ff4) / omega.pow2() + one;
                let a0_h =
                    T::from_f64((3.0 / 16.0) * ff2 * ff3) / ff4_omega2_plus_1;
                let a_1 = T::from_f64(4.0)
                    * inp.cp
                    * (T::from_f64(ff1) * inp.p * q * inp.rho).sqrt();
                let a_2 = T::from_f64(params.mixing_length_alpha)
                    * omega
                    * ff4_omega2_plus_1;
                let a_num = a_1 * a_2;
                let a_den = T::from_f64(ff3 * CRAD * CLIGHT) * inp.t.pow3();
                (a0_h, a_num / a_den)
            }
        };

        // The closed form is always evaluated, with no small-`delta` series
        // (`#322`) and no `delta <= 0` shortcut (`turb/private/mlt.f90:122-145`).
        // Solve cubic efficiency equation:
        // a0 * Gamma^3 + Gamma^2 + Gamma - a0 * B^3 = 0, with delta = a0 * B^3.
        let b_cubed = (a_val.pow2() / a0) * (inp.gradr - inp.gradl);
        let delta = a0 * b_cubed;
        let f = T::from_f64(-2.0)
            + T::from_f64(9.0) * a0
            + T::from_f64(27.0) * a0 * a0 * delta;
        let f0 = if f.val() > 1.0e100 {
            f
        } else {
            let term = T::from_f64(-1.0) + T::from_f64(3.0) * a0;
            let f0_sq = f.pow2() + T::from_f64(4.0) * term * term * term;
            if f0_sq.val() <= 0.0 {
                f
            } else {
                f0_sq.sqrt()
            }
        };
        let f1_arg = f + f0;
        // Bare `return` in `turb/private/mlt.f90:140` before any output is
        // written; caller baseline is `turb_support.f90:283-288`.
        if f1_arg.val() <= 0.0 {
            return Ok(TurbOutput {
                gamma: zero,
                grad_t: inp.gradr,
                y_face: inp.gradr - inp.gradl,
                conv_vel: zero,
                d: zero,
                mixing_type: MixingType::NoMixing,
            });
        }
        let f1 = f1_arg.powf(ONE_THIRD);
        let f2 = T::from_f64(2.0 * TWO_13) * (one - T::from_f64(3.0) * a0) / f1;
        let gamma_val = (T::from_f64(FOUR_13) * f1 + f2 - T::from_f64(2.0))
            / (T::from_f64(6.0) * a0);
        // Bare `return` in `turb/private/mlt.f90:145` after `Gamma` was
        // assigned at line 143; remaining outputs keep caller baseline.
        if gamma_val.val() <= 0.0 {
            return Ok(TurbOutput {
                gamma: gamma_val,
                grad_t: inp.gradr,
                y_face: inp.gradr - inp.gradl,
                conv_vel: zero,
                d: zero,
                mixing_type: MixingType::NoMixing,
            });
        }

        let mut vc = T::from_f64(params.mixing_length_alpha)
            * (q * inp.p / (T::from_f64(8.0) * inp.rho)).sqrt()
            * gamma_val
            / a_val;
        if vc.val() > params.max_conv_vel {
            vc = vc.with_val(params.max_conv_vel);
        }
        // `D = conv_vel*Lambda/3d0` (`turb/private/mlt.f90:153`).
        let d_val = (vc * inp.lambda).div_real(3.0);
        let zeta_val =
            (T::from_f64(3.0) * gamma_val.log() - b_cubed.log()).exp();
        (gamma_val, zeta_val, vc, d_val)
    } else {
        (T::from_f64(-1.0e99), zero, zero, zero)
    };

    // Bare `return` in `turb/private/mlt.f90:169` after `Gamma`, `conv_vel`,
    // and `D` were assigned; `gradT`, `Y_face`, and `mixing_type` keep the
    // caller's no-mixing baseline (`turb_support.f90:283-288`, `#330`).
    if is_bad(zeta.val()) {
        return Ok(TurbOutput {
            gamma,
            grad_t: inp.gradr,
            y_face: inp.gradr - inp.gradl,
            conv_vel,
            d,
            mixing_type: MixingType::NoMixing,
        });
    }
    if zeta.val() > 1.0 {
        zeta = one;
    }

    let grad_t = (one - zeta) * inp.gradr + zeta * inp.gradl;
    let y_face = grad_t - inp.gradl;
    let mixing_type = if y_face.val() > 0.0 {
        MixingType::ConvectiveMixing
    } else {
        MixingType::NoMixing
    };

    Ok(TurbOutput {
        gamma,
        grad_t,
        y_face,
        conv_vel,
        d,
        mixing_type,
    })
}
