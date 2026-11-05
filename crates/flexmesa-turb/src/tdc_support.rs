// SPDX-License-Identifier: LGPL-3.0-or-later
//! Time-dependent convection physics kernels (`turb/private/tdc_support.f90`).
//!
//! Provides the Kuhfuss (1986) / Smolec time-dependent convection velocity
//! evolution (`eval_xis`, `eval_af`), enthalpy flux limiter
//! (`flux_limiter_function`, Wuchterl & Feuchtinger 1998), and luminosity
//! residual evaluation (`compute_q`) generic over [`TurbScalar`]:
//! - `eval_xis`: Evaluates coefficients $(\xi_0, \xi_1, \xi_2)$ of the
//!   convective velocity equation $2\,dw/dt = \xi_0 + \xi_1 w + \xi_2 w^2$.
//! - `eval_af`: Integrates the convective velocity equation analytically over
//!   a time step $\Delta t$ across hyperbolic ($J^2 > 0$) and trigonometric
//!   ($J^2 < 0$) branches.
//! - `compute_q`: Evaluates the total luminosity balance residual
//!   $Q(Y) = (L - L_0 \nabla_L) - L_0 Y - c_0 A_f(Y_{\mathrm{env}})
//!   Y_{\mathrm{env}}$.

use flexmesa_autodiff::AutoDiffReal1VarOrder1;
use flexmesa_const::{BOLTZ_SIGMA, PI, SQRT_2_DIV_3};

use crate::TurbScalar;

/// Source-term coefficient $x_{\alpha_s} = \frac{1}{2}\sqrt{2/3}$.
pub const X_ALFAS: f64 = 0.5 * SQRT_2_DIV_3;

/// Turbulent dissipation coefficient $x_{C_D} = \frac{8}{3}\sqrt{2/3}$.
pub const X_CEDE: f64 = (8.0 / 3.0) * SQRT_2_DIV_3;

/// Turbulent pressure work coefficient $x_{\alpha_p} = 2/3$.
pub const X_ALFAP: f64 = 2.0 / 3.0;

/// Internal state container for TDC evaluations independent of $Y$ (`tdc_info`
/// in `tdc_support.f90`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TdcInfo<T> {
    /// Include steady-state Cox MLT geometric correction
    /// $\Gamma / (1 + \Gamma)$.
    pub include_mlt_corr_to_tdc: bool,
    /// Enable Wuchterl & Feuchtinger (1998) $C^2$ quintic enthalpy flux
    /// limiter.
    pub use_tdc_enthalpy_flux_limiter: bool,
    /// Mixing-length parameter $\alpha_{\mathrm{MLT}}$.
    pub mixing_length_alpha: f64,
    /// Convective flux multiplier $\alpha_C$.
    pub tdc_alpha_c: f64,
    /// Buoyant source multiplier $\alpha_S$.
    pub tdc_alpha_s: f64,
    /// Turbulent drag multiplier $\alpha_D$.
    pub tdc_alpha_d: f64,
    /// Radiative damping multiplier $\alpha_R$.
    pub tdc_alpha_r: f64,
    /// Turbulent pressure work multiplier $\alpha_{P_t}$.
    pub tdc_alpha_pt: f64,
    /// Time step $\Delta t$ in $\mathrm{s}$.
    pub dt: f64,
    /// Initial scaled convective speed
    /// $A_0 = v_{\mathrm{conv,start}} / \sqrt{2/3}$.
    pub a0: T,
    /// Convective luminosity prefactor $c_0$ in $\mathrm{erg\,cm^{-1}}$.
    pub c0: T,
    /// Total luminosity $L$ in $\mathrm{erg\,s^{-1}}$.
    pub l: T,
    /// Radiative luminosity scale $L_0 = L / \nabla_{\mathrm{rad}}$ in
    /// $\mathrm{erg\,s^{-1}}$.
    pub l0: T,
    /// Ledoux temperature gradient $\nabla_L$.
    pub gradl: T,
    /// Adiabatic temperature gradient $\nabla_{\mathrm{ad}}$.
    pub grada: T,
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
    pub kap: T,
    /// Pressure scale height $H_P$ in $\mathrm{cm}$.
    pub hp: T,
    /// Steady-state Cox MLT efficiency $\Gamma$.
    pub gamma: T,
    /// Coupled source term $E_q / w$ in $\mathrm{cm\,s^{-2}}$.
    pub eq_div_w: T,
    /// Specific enthalpy $h = e + P/\rho$ in $\mathrm{erg\,g^{-1}}$.
    pub h: T,
}

impl<T: TurbScalar> TdcInfo<T> {
    /// Projects `self` to a pure `f64` primal structure for fast bracket and
    /// line-search evaluations.
    #[must_use]
    pub fn to_primal(&self) -> TdcInfo<f64> {
        TdcInfo {
            include_mlt_corr_to_tdc: self.include_mlt_corr_to_tdc,
            use_tdc_enthalpy_flux_limiter: self.use_tdc_enthalpy_flux_limiter,
            mixing_length_alpha: self.mixing_length_alpha,
            tdc_alpha_c: self.tdc_alpha_c,
            tdc_alpha_s: self.tdc_alpha_s,
            tdc_alpha_d: self.tdc_alpha_d,
            tdc_alpha_r: self.tdc_alpha_r,
            tdc_alpha_pt: self.tdc_alpha_pt,
            dt: self.dt,
            a0: self.a0.val(),
            c0: self.c0.val(),
            l: self.l.val(),
            l0: self.l0.val(),
            gradl: self.gradl.val(),
            grada: self.grada.val(),
            t: self.t.val(),
            rho: self.rho.val(),
            dv: self.dv.val(),
            cp: self.cp.val(),
            kap: self.kap.val(),
            hp: self.hp.val(),
            gamma: self.gamma.val(),
            eq_div_w: self.eq_div_w.val(),
            h: self.h.val(),
        }
    }

    /// Projects `self` to a constant [`AutoDiffReal1VarOrder1`]
    /// (`DualOrder1<1>`) structure for 1D differentiation with respect to
    /// $Z = \ln|Y|$.
    #[must_use]
    pub fn to_z_dual(&self) -> TdcInfo<AutoDiffReal1VarOrder1> {
        let c = |x: T| AutoDiffReal1VarOrder1::constant(x.val());
        TdcInfo {
            include_mlt_corr_to_tdc: self.include_mlt_corr_to_tdc,
            use_tdc_enthalpy_flux_limiter: self.use_tdc_enthalpy_flux_limiter,
            mixing_length_alpha: self.mixing_length_alpha,
            tdc_alpha_c: self.tdc_alpha_c,
            tdc_alpha_s: self.tdc_alpha_s,
            tdc_alpha_d: self.tdc_alpha_d,
            tdc_alpha_r: self.tdc_alpha_r,
            tdc_alpha_pt: self.tdc_alpha_pt,
            dt: self.dt,
            a0: c(self.a0),
            c0: c(self.c0),
            l: c(self.l),
            l0: c(self.l0),
            gradl: c(self.gradl),
            grada: c(self.grada),
            t: c(self.t),
            rho: c(self.rho),
            dv: c(self.dv),
            cp: c(self.cp),
            kap: c(self.kap),
            hp: c(self.hp),
            gamma: c(self.gamma),
            eq_div_w: c(self.eq_div_w),
            h: c(self.h),
        }
    }
}

/// Maps logarithmic superadiabaticity $Z = \ln|Y|$ to $Y = \pm \exp(Z)$
/// (`set_Y` in `tdc_support.f90`).
#[must_use]
#[inline]
pub fn set_y<T: TurbScalar>(y_is_positive: bool, z: T) -> T {
    if y_is_positive {
        z.exp()
    } else {
        -z.exp()
    }
}

/// Numerically safe hyperbolic tangent clamping $|x| > 50$ to $\pm 1$
/// (`safe_tanh` in `tdc_support.f90`).
#[must_use]
#[inline]
pub fn safe_tanh<T: TurbScalar>(x: T) -> T {
    if x.val() > 50.0 {
        T::from_f64(1.0)
    } else if x.val() < -50.0 {
        T::from_f64(-1.0)
    } else {
        x.tanh()
    }
}

/// Wuchterl & Feuchtinger (1998) $C^2$ quintic smootherstep enthalpy flux
/// limiter (`flux_limiter_function` in `tdc_support.f90`).
#[must_use]
pub fn flux_limiter_function<T: TurbScalar>(x: T) -> T {
    const X0: f64 = 0.95;
    const X1: f64 = 1.0;
    if x.val() < X0 {
        x
    } else if x.val() >= X1 {
        T::from_f64(1.0)
    } else {
        let s = (x - T::from_f64(X0)) / T::from_f64(X1 - X0);
        let p = s.pow3()
            * (T::from_f64(10.0)
                + s * (T::from_f64(-15.0) + T::from_f64(6.0) * s));
        x + (T::from_f64(1.0) - x) * p
    }
}

/// Evaluates coefficients $(\xi_0, \xi_1, \xi_2)$ of the TDC velocity equation
/// $2\,dw/dt = \xi_0 + \xi_1 w + \xi_2 w^2$ (`eval_xis` in `tdc_support.f90`).
#[must_use]
pub fn eval_xis<T: TurbScalar>(info: &TdcInfo<T>, y: T) -> (T, T, T) {
    let one = T::from_f64(1.0);
    let x_gammar = 2.0 * 3.0_f64.sqrt();

    let mut scale = one;
    if y.val() > 0.0 && info.use_tdc_enthalpy_flux_limiter {
        let factor = info.mixing_length_alpha * info.tdc_alpha_s * X_ALFAS
            / SQRT_2_DIV_3;
        let x = (info.cp * info.t / info.h) * T::from_f64(factor) * y;
        let fl = flux_limiter_function(x);
        if x.val().abs() >= 0.95 {
            scale = fl / x;
        }
    }

    let s0_pref =
        T::from_f64(info.tdc_alpha_s * X_ALFAS * info.mixing_length_alpha)
            * info.cp
            * info.t
            / info.hp
            * info.grada;
    let s0 = if info.use_tdc_enthalpy_flux_limiter {
        s0_pref * (scale * y) + info.eq_div_w
    } else {
        s0_pref * y + info.eq_div_w
    };

    let d0 = T::from_f64(info.tdc_alpha_d * X_CEDE / info.mixing_length_alpha)
        / info.hp;
    let gammar_div_alfa =
        T::from_f64(info.tdc_alpha_r * x_gammar / info.mixing_length_alpha)
            / info.hp;
    let dr0 =
        T::from_f64(4.0 * BOLTZ_SIGMA) * gammar_div_alfa.pow2() * info.t.pow3()
            / (info.rho.pow2() * info.cp * info.kap);
    let pt0 = T::from_f64(info.tdc_alpha_pt * X_ALFAP) * info.rho;
    let dv_dt = info.dv / T::from_f64(info.dt);

    (s0, -(dr0 + pt0 * dv_dt), -d0)
}

/// Evaluates final scaled convective speed $A_f(\Delta t)$ across hyperbolic
/// ($J^2 > 0$) and trigonometric ($J^2 < 0$) branches (`eval_Af` in
/// `tdc_support.f90`).
#[must_use]
pub fn eval_af<T: TurbScalar>(dt: f64, a0: T, xi0: T, xi1: T, xi2: T) -> T {
    let j2 = xi1.pow2() - T::from_f64(4.0) * xi0 * xi2;
    if j2.val() > 0.0 {
        let j = j2.abs().sqrt();
        let jt4 = T::from_f64(0.25 * dt) * j;
        let th = safe_tanh(jt4);
        let num = th * (T::from_f64(2.0) * xi0 + a0 * xi1) + a0 * j;
        let den = th * (xi1 + T::from_f64(2.0) * a0 * xi2) - j;
        let af = num / den;
        if af.val() < 0.0 {
            -af
        } else {
            af
        }
    } else if j2.val() < 0.0 {
        let j = j2.abs().sqrt();
        let jt4 = T::from_f64(0.25 * dt) * j;
        let y_for_atan = xi1 + T::from_f64(2.0) * a0 * xi2;
        let atan_y = (y_for_atan / j).atan();
        let mut root = (xi1 / j).atan() - atan_y;
        if root.val() > PI {
            root = root - T::from_f64(PI);
        } else if root.val() < -PI {
            root = root + T::from_f64(2.0 * PI);
        } else if root.val() < 0.0 {
            root = root + T::from_f64(PI);
        }

        if jt4.val() < root.val() {
            let num = -xi1 + j * (jt4 + atan_y).tan();
            let den = T::from_f64(2.0) * xi2;
            num / den
        } else {
            T::from_f64(0.0)
        }
    } else {
        a0
    }
}

/// Evaluates luminosity residual
/// $Q(Y) = (L - L_0\nabla_L) - L_0 Y - c_0 A_f Y_{\mathrm{env}}$
/// and final scaled speed $A_f$ (`compute_Q` in `tdc_support.f90`).
#[must_use]
pub fn compute_q<T: TurbScalar>(info: &TdcInfo<T>, y: T) -> (T, T) {
    let y_env = if y.val() > 0.0 && info.include_mlt_corr_to_tdc {
        y * (info.gamma / (T::from_f64(1.0) + info.gamma))
    } else {
        y
    };
    let (xi0, xi1, xi2) = eval_xis(info, y_env);
    let af = eval_af(info.dt, info.a0, xi0, xi1, xi2);
    let q =
        (info.l - info.l0 * info.gradl) - info.l0 * y - info.c0 * af * y_env;
    (q, af)
}
