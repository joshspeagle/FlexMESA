// SPDX-License-Identifier: LGPL-3.0-or-later
//! Time-Dependent Convection solver (`set_tdc`, `turb/private/tdc.f90`).
//!
//! Solves the Kuhfuss (1986) / Smolec time-dependent convection energy balance
//! equation
//! $$Q(Y) = (L - L_0 \nabla_L) - L_0 Y
//!   - c_0 A_f(Y_{\mathrm{env}}) Y_{\mathrm{env}} = 0$$
//!
//! over $Z = \ln|Y| \in [-100, 100]$ using bisection bracketing followed by
//! damped Newton-Raphson refinement (`bracket_plus_newton_search`).
//!
//! # Derivative Propagation (Implicit Function Theorem)
//! At a converged root $Y^*$ where $Q(Y^*, \mathbf{p}) = 0$, the first-order
//! sensitivity with respect to stellar structure parameters $\mathbf{p}$ is
//! given by the Implicit Function Theorem:
//! $$\frac{\partial Y^*}{\partial \mathbf{p}} =
//!   -\left(\frac{\partial Q}{\partial Y}\right)^{-1}
//!   \left.\frac{\partial Q}{\partial \mathbf{p}}\right|_{Y^*},$$
//! matching the derivative computed by upstream MESA's nested hyper-dual type
//! `auto_diff_real_tdc` while keeping the inner 1D root-finder strictly 1D.

use flexmesa_autodiff::AutoDiffReal1VarOrder1;
use flexmesa_const::{MixingType, BOLTZ_SIGMA, CLIGHT, CRAD, PI, SQRT_2_DIV_3};

use crate::mlt::{set_mlt, MltInputs, MltParams};
use crate::tdc_search::{
    af_bisection_search, bracket_plus_newton_search, dq_dz_bisection_search,
};
use crate::tdc_support::{compute_q, set_y, TdcInfo, X_ALFAS};
use crate::{TurbError, TurbOutput, TurbResult, TurbScalar};

/// Non-differentiable configuration parameters for [`set_tdc`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TdcParams {
    /// Convection speed $v_{\mathrm{conv,start}}$ at the start of the step in
    /// $\mathrm{cm\,s^{-1}}$.
    pub conv_vel_start: f64,
    /// Mixing-length parameter $\alpha_{\mathrm{MLT}}$.
    pub mixing_length_alpha: f64,
    /// Turbulent dissipation multiplier $\alpha_D$.
    pub tdc_alpha_d: f64,
    /// Radiative damping multiplier $\alpha_R$.
    pub tdc_alpha_r: f64,
    /// Turbulent pressure work multiplier $\alpha_{P_t}$.
    pub tdc_alpha_pt: f64,
    /// Convective flux multiplier $\alpha_C$.
    pub tdc_alpha_c: f64,
    /// Buoyant source multiplier $\alpha_S$.
    pub tdc_alpha_s: f64,
    /// Time step $\Delta t$ in $\mathrm{s}$.
    pub dt: f64,
    /// Gravitational constant $G$ in $\mathrm{cm^3\,g^{-1}\,s^{-2}}$.
    pub cgrav: f64,
    /// Enclosed mass $m$ at the cell face in $\mathrm{g}$.
    pub m: f64,
    /// Luminosity scale for relative residual convergence in
    /// $\mathrm{erg\,s^{-1}}$.
    pub scale: f64,
    /// Upper cap on convective velocity $v_{\mathrm{conv},\max}$ in
    /// $\mathrm{cm\,s^{-1}}$.
    pub max_conv_vel: f64,
    /// Include steady-state Cox MLT geometric correction
    /// $\Gamma / (1 + \Gamma)$.
    pub include_mlt_corr_to_tdc: bool,
    /// Enable Wuchterl & Feuchtinger (1998) quintic enthalpy flux limiter.
    pub use_tdc_enthalpy_flux_limiter: bool,
    /// Optional positive superadiabaticity seed $Y_{\mathrm{face,guess}}$ for
    /// local bracketing ($\le 0$ disables seeding).
    pub y_face_guess: f64,
}

/// Differentiable thermodynamic and structure inputs for [`set_tdc`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TdcInputs<T> {
    /// Logarithmic pressure derivative
    /// $\chi_T = (\partial \ln P / \partial \ln T)_\rho$.
    pub chi_t: T,
    /// Logarithmic pressure derivative
    /// $\chi_\rho = (\partial \ln P / \partial \ln \rho)_T$.
    pub chi_rho: T,
    /// Radiative temperature gradient $\nabla_{\mathrm{rad}}$.
    pub gradr: T,
    /// Radial coordinate $r$ at the cell face in $\mathrm{cm}$.
    pub r: T,
    /// Total pressure $P$ in $\mathrm{dyn\,cm^{-2}}$.
    pub p: T,
    /// Temperature $T$ in $\mathrm{K}$.
    pub t: T,
    /// Mass density $\rho$ in $\mathrm{g\,cm^{-3}}$.
    pub rho: T,
    /// Specific volume change $\Delta(1/\rho)$ in $\mathrm{cm^3\,g^{-1}}$.
    pub dv: T,
    /// Specific heat at constant pressure $C_P$ in
    /// $\mathrm{erg\,g^{-1}\,K^{-1}}$.
    pub cp: T,
    /// Rosseland mean opacity $\kappa$ in $\mathrm{cm^2\,g^{-1}}$.
    pub opacity: T,
    /// Pressure scale height $H_P$ in $\mathrm{cm}$.
    pub scale_height: T,
    /// Ledoux temperature gradient $\nabla_L$.
    pub gradl: T,
    /// Adiabatic temperature gradient $\nabla_{\mathrm{ad}}$.
    pub grada: T,
    /// Coupled source term $E_q / w$ in $\mathrm{cm\,s^{-2}}$.
    pub eq_div_w: T,
    /// Local gravitational acceleration $g$ in $\mathrm{cm\,s^{-2}}$.
    pub grav: T,
    /// Specific internal energy $e$ in $\mathrm{erg\,g^{-1}}$.
    pub energy: T,
}

/// Output structure returned by [`set_tdc`], pairing [`TurbOutput<T>`] with the
/// TDC Newton iteration count `num_iters`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TdcOutput<T> {
    /// Standard turbulent transport outputs (`gamma`, `grad_t`, `y_face`,
    /// `conv_vel`, `d`, `mixing_type`).
    pub output: TurbOutput<T>,
    /// Number of Newton iterations taken in the TDC root solver.
    pub num_iters: usize,
}

/// Evaluates Time-Dependent Convection (`set_TDC` in `turb/public/turb.f90` and
/// `get_TDC_solution` in `turb/private/tdc.f90`).
///
/// # Errors
/// Returns [`TurbError`] if `mixing_length_alpha == 0`, `dt <= 0`, or the TDC
/// root-finder fails to converge.
pub fn set_tdc<T: TurbScalar>(
    params: &TdcParams,
    inp: &TdcInputs<T>,
) -> TurbResult<TdcOutput<T>> {
    if params.mixing_length_alpha <= 0.0
        || params.dt <= 0.0
        || params.scale <= 0.0
        || params.mixing_length_alpha.is_nan()
        || params.dt.is_nan()
        || params.scale.is_nan()
        || inp.t.val() <= 0.0
        || inp.rho.val() <= 0.0
        || inp.p.val() <= 0.0
        || inp.cp.val() <= 0.0
        || inp.opacity.val() <= 0.0
        || inp.scale_height.val() <= 0.0
        || inp.gradr.val().is_nan()
        || inp.gradl.val().is_nan()
        || inp.grada.val().is_nan()
    {
        return Err(TurbError::InvalidParameter(
            "invalid or non-positive parameter in set_tdc",
        ));
    }

    let l = T::from_f64(64.0 * PI * BOLTZ_SIGMA)
        * inp.t.pow4()
        * inp.grav
        * inp.r.pow2()
        * inp.gradr
        / (T::from_f64(3.0) * inp.p * inp.opacity);

    let gamma = if params.include_mlt_corr_to_tdc {
        let lambda = T::from_f64(params.mixing_length_alpha) * inp.scale_height;
        let mlt_params = MltParams::cox(params.mixing_length_alpha);
        let mlt_inp = MltInputs {
            chi_t: inp.chi_t,
            chi_rho: inp.chi_rho,
            cp: inp.cp,
            grav: inp.grav,
            lambda,
            rho: inp.rho,
            p: inp.p,
            t: inp.t,
            opacity: inp.opacity,
            gradr: inp.gradr,
            grada: inp.grada,
            gradl: inp.gradl,
        };
        set_mlt(&mlt_params, &mlt_inp)?.gamma
    } else {
        T::from_f64(0.0)
    };

    let c0 = T::from_f64(
        params.tdc_alpha_c * params.mixing_length_alpha * X_ALFAS * 4.0 * PI,
    ) * inp.rho
        * inp.t
        * inp.cp
        * inp.r.pow2();
    let l0 = T::from_f64(
        (16.0 * PI * CRAD * CLIGHT / 3.0) * params.cgrav * params.m,
    ) * inp.t.pow4()
        / (inp.p * inp.opacity);
    let a0 = T::from_f64(params.conv_vel_start / SQRT_2_DIV_3);
    let h = inp.energy + inp.p / inp.rho;

    let info = TdcInfo {
        include_mlt_corr_to_tdc: params.include_mlt_corr_to_tdc,
        use_tdc_enthalpy_flux_limiter: params.use_tdc_enthalpy_flux_limiter,
        mixing_length_alpha: params.mixing_length_alpha,
        tdc_alpha_c: params.tdc_alpha_c,
        tdc_alpha_s: params.tdc_alpha_s,
        tdc_alpha_d: params.tdc_alpha_d,
        tdc_alpha_r: params.tdc_alpha_r,
        tdc_alpha_pt: params.tdc_alpha_pt,
        dt: params.dt,
        a0,
        c0,
        l,
        l0,
        gradl: inp.gradl,
        grada: inp.grada,
        t: inp.t,
        rho: inp.rho,
        dv: inp.dv,
        cp: inp.cp,
        kap: inp.opacity,
        hp: inp.scale_height,
        gamma,
        eq_div_w: inp.eq_div_w,
        h,
    };

    let (mut conv_vel, mut y_face, num_iters) = get_tdc_solution(
        &info,
        params.scale,
        -100.0,
        100.0,
        params.y_face_guess,
    )?;

    if conv_vel.val() > params.max_conv_vel {
        conv_vel = T::from_f64(params.max_conv_vel);
        let one = T::from_f64(1.0);
        let num = info.l - info.l0 * info.gradl;
        y_face = if params.include_mlt_corr_to_tdc {
            num / (info.l0
                + info.c0
                    * T::from_f64(SQRT_2_DIV_3 * params.max_conv_vel)
                    * (info.gamma / (one + info.gamma)))
        } else {
            num / (info.l0
                + info.c0 * T::from_f64(SQRT_2_DIV_3 * params.max_conv_vel))
        };
    }

    let grad_t = y_face + inp.gradl;
    let d = conv_vel
        * inp.scale_height
        * T::from_f64(params.mixing_length_alpha / 3.0);
    let mixing_type = if conv_vel.val() > 0.0 {
        MixingType::ConvectiveMixing
    } else {
        MixingType::NoMixing
    };

    Ok(TdcOutput {
        output: TurbOutput {
            gamma,
            grad_t,
            y_face,
            conv_vel,
            d,
            mixing_type,
        },
        num_iters,
    })
}

/// Solves for `(conv_vel, Y_face, num_iters)` (`get_TDC_solution` in
/// `tdc.f90`).
fn get_tdc_solution<T: TurbScalar>(
    info: &TdcInfo<T>,
    scale: f64,
    zlb: f64,
    zub: f64,
    y_face_guess: f64,
) -> TurbResult<(T, T, usize)> {
    let info_p = info.to_primal();
    let info_z = info.to_z_dual();
    let (q0, _) = compute_q(&info_p, 0.0);
    let y_is_positive = q0 > 0.0;

    let mut num_iters = 0_usize;
    let y_sol: T = if y_is_positive {
        let (y_star, iters) = bracket_plus_newton_search(
            &info_p,
            &info_z,
            scale,
            true,
            zlb,
            zub,
            y_face_guess,
        )?;
        num_iters = iters;
        lift_implicit_root(info, &info_z, y_star)
    } else {
        let rad_y = (info.l - info.l0 * info.gradl) / info.l0;
        if info.a0.val() < 1.0e-10 {
            rad_y
        } else {
            let (_, af_lb) = compute_q(&info_p, set_y(false, zlb));
            if af_lb == 0.0 {
                rad_y
            } else {
                let (_, af_ub) = compute_q(&info_p, set_y(false, zub));
                let z0 = if af_ub > 0.0 {
                    zub
                } else {
                    af_bisection_search(&info_p, zlb, zub)?
                };

                if let Some(z1) = dq_dz_bisection_search(&info_z, zlb, z0) {
                    let (q1, _) = compute_q(&info_p, set_y(false, z1));
                    if q1 < 0.0 {
                        rad_y
                    } else if let Ok((y_star, iters)) =
                        bracket_plus_newton_search(
                            &info_p, &info_z, scale, false, zlb, z1, 0.0,
                        )
                    {
                        num_iters = iters;
                        lift_implicit_root(info, &info_z, y_star)
                    } else {
                        let (y_star, iters) = bracket_plus_newton_search(
                            &info_p, &info_z, scale, false, z1, z0, 0.0,
                        )?;
                        num_iters = iters;
                        lift_implicit_root(info, &info_z, y_star)
                    }
                } else {
                    let (q_y0, _) = compute_q(&info_p, set_y(false, z0));
                    if q_y0 > 0.0 {
                        let (y_star, iters) = bracket_plus_newton_search(
                            &info_p, &info_z, scale, false, zlb, z0, 0.0,
                        )?;
                        num_iters = iters;
                        lift_implicit_root(info, &info_z, y_star)
                    } else {
                        rad_y
                    }
                }
            }
        }
    };

    let (_, af) = compute_q(info, y_sol);
    let conv_vel = T::from_f64(SQRT_2_DIV_3) * af;
    Ok((conv_vel, y_sol, num_iters))
}

/// Propagates structure derivatives into a converged scalar root $Y^*$ via the
/// Implicit Function Theorem
/// $\nabla_{\mathbf{p}} Y^* = -(\partial Q / \partial Y)^{-1}
/// \nabla_{\mathbf{p}} Q|_{Y^*}$.
fn lift_implicit_root<T: TurbScalar>(
    info: &TdcInfo<T>,
    info_z: &TdcInfo<AutoDiffReal1VarOrder1>,
    y_star: f64,
) -> T {
    let (q_dual_y, _) =
        compute_q(info_z, AutoDiffReal1VarOrder1::independent(y_star, 0));
    let dq_dy = q_dual_y.d1[0];
    let y_const = T::from_f64(y_star);
    if dq_dy == 0.0 || !dq_dy.is_finite() {
        return y_const;
    }
    let (q_p, _) = compute_q(info, y_const);
    let zero_val_q_p = q_p - T::from_f64(q_p.val());
    y_const - zero_val_q_p / T::from_f64(dq_dy)
}
