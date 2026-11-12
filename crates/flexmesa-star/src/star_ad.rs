// SPDX-License-Identifier: LGPL-3.0-or-later
//! Three-cell stencil forward-mode automatic differentiation type
//! [`StarAdOrder1`] (`auto_diff_real_star_order1` in
//! `star/private/auto_diff_support.f90`).
//!
//! Tracks first-order directional derivatives of any cell-centered or
//! face-centered stellar structure quantity with respect to the primary
//! hydrodynamic variables across the 3-cell stencil $(k-1, k, k+1)$:
//! - `_m1`: outer neighbor cell $k-1$
//! - `_00`: current cell $k$
//! - `_p1`: inner neighbor cell $k+1$

use flexmesa_autodiff::DualOrder1;

/// Number of directional derivative slots in [`StarAdOrder1`] (8 hydro variable
/// families $\times$ 3 stencil cells `m1`, `00`, `p1`).
pub const NUM_STAR_AD_VARS: usize = 24;

/// Directional derivative slot for $\ln\rho_{k-1}$ (`i_lnd_m1`).
pub const I_LND_M1: usize = 0;
/// Directional derivative slot for $\ln\rho_k$ (`i_lnd_00`).
pub const I_LND_00: usize = 1;
/// Directional derivative slot for $\ln\rho_{k+1}$ (`i_lnd_p1`).
pub const I_LND_P1: usize = 2;

/// Directional derivative slot for $\ln T_{k-1}$ (`i_lnT_m1`).
pub const I_LNT_M1: usize = 3;
/// Directional derivative slot for $\ln T_k$ (`i_lnT_00`).
pub const I_LNT_00: usize = 4;
/// Directional derivative slot for $\ln T_{k+1}$ (`i_lnT_p1`).
pub const I_LNT_P1: usize = 5;

/// Directional derivative slot for $\ln R_{k-1}$ (`i_lnR_m1`).
pub const I_LNR_M1: usize = 6;
/// Directional derivative slot for $\ln R_k$ (`i_lnR_00`).
pub const I_LNR_00: usize = 7;
/// Directional derivative slot for $\ln R_{k+1}$ (`i_lnR_p1`).
pub const I_LNR_P1: usize = 8;

/// Directional derivative slot for $L_{k-1}$ (`i_L_m1`).
pub const I_L_M1: usize = 9;
/// Directional derivative slot for $L_k$ (`i_L_00`).
pub const I_L_00: usize = 10;
/// Directional derivative slot for $L_{k+1}$ (`i_L_p1`).
pub const I_L_P1: usize = 11;

/// Directional derivative slot for $v_{k-1}$ or $u_{k-1}$ (`i_v_m1`).
pub const I_V_M1: usize = 12;
/// Directional derivative slot for $v_k$ or $u_k$ (`i_v_00`).
pub const I_V_00: usize = 13;
/// Directional derivative slot for $v_{k+1}$ or $u_{k+1}$ (`i_v_p1`).
pub const I_V_P1: usize = 14;

/// Directional derivative slot for $\Omega_{k-1}$ (`i_omega_m1`).
pub const I_OMEGA_M1: usize = 15;
/// Directional derivative slot for $\Omega_k$ (`i_omega_00`).
pub const I_OMEGA_00: usize = 16;
/// Directional derivative slot for $\Omega_{k+1}$ (`i_omega_p1`).
pub const I_OMEGA_P1: usize = 17;

/// Directional derivative slot for turbulent velocity $w_{k-1}$ (`i_w_m1`).
pub const I_W_M1: usize = 18;
/// Directional derivative slot for turbulent velocity $w_k$ (`i_w_00`).
pub const I_W_00: usize = 19;
/// Directional derivative slot for turbulent velocity $w_{k+1}$ (`i_w_p1`).
pub const I_W_P1: usize = 20;

/// Directional derivative slot for pressure scale height $H_{P,k-1}$ (`i_Hp_m1`).
pub const I_HP_M1: usize = 21;
/// Directional derivative slot for pressure scale height $H_{P,k}$ (`i_Hp_00`).
pub const I_HP_00: usize = 22;
/// Directional derivative slot for pressure scale height $H_{P,k+1}$ (`i_Hp_p1`).
pub const I_HP_P1: usize = 23;

/// First-order 3-cell stencil automatic differentiation scalar
/// (`auto_diff_real_star_order1` in MESA `star`), backed by [`DualOrder1<24>`].
pub type StarAdOrder1 = DualOrder1<NUM_STAR_AD_VARS>;

/// Three-cell stencil helper constructors and shift operations for
/// [`StarAdOrder1`].
pub trait StarAdExt: Sized {
    /// Zero constant (`val = 0.0`, `d1 = [0.0; 24]`).
    const ZERO: Self;
    /// Unit constant (`val = 1.0`, `d1 = [0.0; 24]`).
    const ONE: Self;

    /// Creates a seeded variable with `d1[slot] = 1.0` and all other partials
    /// equal to zero.
    #[must_use]
    fn variable(val: f64, slot: usize) -> Self;

    /// Creates a cell-$k$ thermodynamic quantity with known partials wrt
    /// $(\ln\rho_k, \ln T_k)$.
    #[must_use]
    fn from_dlnd_dlnt_00(val: f64, d_dlnd: f64, d_dlnt: f64) -> Self;

    /// Creates a cell-$(k-1)$ thermodynamic quantity with known partials wrt
    /// $(\ln\rho_{k-1}, \ln T_{k-1})$.
    #[must_use]
    fn from_dlnd_dlnt_m1(val: f64, d_dlnd: f64, d_dlnt: f64) -> Self;

    /// Creates a cell-$(k+1)$ thermodynamic quantity with known partials wrt
    /// $(\ln\rho_{k+1}, \ln T_{k+1})$.
    #[must_use]
    fn from_dlnd_dlnt_p1(val: f64, d_dlnd: f64, d_dlnt: f64) -> Self;

    /// Shifts a quantity evaluated in cell $(k-1)$'s frame into cell $k$'s
    /// stencil (`shift_m1` in `auto_diff_support.f90:390`).
    #[must_use]
    fn shift_m1(&self) -> Self;

    /// Shifts a quantity evaluated in cell $(k+1)$'s frame into cell $k$'s
    /// stencil (`shift_p1` in `auto_diff_support.f90:35`).
    #[must_use]
    fn shift_p1(&self) -> Self;

    /// Cube root $x^{1/3}$.
    #[must_use]
    fn cbrt(self) -> Self;
}

impl StarAdExt for StarAdOrder1 {
    const ZERO: Self = Self::constant(0.0);
    const ONE: Self = Self::constant(1.0);

    #[inline]
    fn variable(val: f64, slot: usize) -> Self {
        Self::independent(val, slot)
    }

    #[inline]
    fn from_dlnd_dlnt_00(val: f64, d_dlnd: f64, d_dlnt: f64) -> Self {
        let mut d1 = [0.0; NUM_STAR_AD_VARS];
        d1[I_LND_00] = d_dlnd;
        d1[I_LNT_00] = d_dlnt;
        Self { val, d1 }
    }

    #[inline]
    fn from_dlnd_dlnt_m1(val: f64, d_dlnd: f64, d_dlnt: f64) -> Self {
        let mut d1 = [0.0; NUM_STAR_AD_VARS];
        d1[I_LND_M1] = d_dlnd;
        d1[I_LNT_M1] = d_dlnt;
        Self { val, d1 }
    }

    #[inline]
    fn from_dlnd_dlnt_p1(val: f64, d_dlnd: f64, d_dlnt: f64) -> Self {
        let mut d1 = [0.0; NUM_STAR_AD_VARS];
        d1[I_LND_P1] = d_dlnd;
        d1[I_LNT_P1] = d_dlnt;
        Self { val, d1 }
    }

    #[inline]
    fn shift_m1(&self) -> Self {
        let mut d1 = [0.0; NUM_STAR_AD_VARS];
        for grp in 0..8 {
            let b = 3 * grp;
            d1[b] = self.d1[b + 1];
            d1[b + 1] = self.d1[b + 2];
            d1[b + 2] = 0.0;
        }
        Self { val: self.val, d1 }
    }

    #[inline]
    fn shift_p1(&self) -> Self {
        let mut d1 = [0.0; NUM_STAR_AD_VARS];
        for grp in 0..8 {
            let b = 3 * grp;
            d1[b] = 0.0;
            d1[b + 1] = self.d1[b];
            d1[b + 2] = self.d1[b + 1];
        }
        Self { val: self.val, d1 }
    }

    #[inline]
    fn cbrt(self) -> Self {
        let c = self.val.cbrt();
        let dc = if c == 0.0 { 0.0 } else { 1.0 / (3.0 * c * c) };
        self.make_unary_operator(c, dc)
    }
}
