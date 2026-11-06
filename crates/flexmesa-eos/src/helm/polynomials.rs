//! Biquintic (`h5`) and bicubic (`h3`) Hermite basis polynomials for the
//! Helmholtz (`HELM`) electron-positron equation of state.
//!
//! This module ports `eos/private/helm_polynomials.f90` and the basis-weight
//! assembly from `eos/private/helm_electron_positron.dek` (Timmes & Swesty
//! 2000, ApJS, 126, 501).
//!
//! Given normalized cell coordinates $z \in [0, 1]$, the three 1D quintic
//! Hermite basis functions $\psi_0(z), \psi_1(z), \psi_2(z)$ and their first
//! three derivatives interpolate a scalar field together with its first and
//! second partial derivatives at the four corners of a $(\rho Y_e, T)$ cell.
//! Similarly, the two 1D cubic Hermite basis functions $\chi_0(z), \chi_1(z)$
//! and their first three derivatives interpolate supplementary
//! electron-positron tables (`dpdf`, `ef`, `xf`) together with their first
//! partial derivatives.

/// Evaluates the zero-derivative quintic Hermite basis function $\psi_0(z)$.
///
/// # Arguments
/// * `z` - Normalized cell coordinate in $[0, 1]$ (dimensionless).
#[inline]
#[must_use]
pub fn psi0(z: f64) -> f64 {
    z * z * z * (z * (-6.0 * z + 15.0) - 10.0) + 1.0
}

/// First derivative $d\psi_0 / dz$.
#[inline]
#[must_use]
pub fn dpsi0(z: f64) -> f64 {
    z * z * (z * (-30.0 * z + 60.0) - 30.0)
}

/// Second derivative $d^2\psi_0 / dz^2$.
#[inline]
#[must_use]
pub fn ddpsi0(z: f64) -> f64 {
    z * (z * (-120.0 * z + 180.0) - 60.0)
}

/// Third derivative $d^3\psi_0 / dz^3$.
#[inline]
#[must_use]
pub fn dddpsi0(z: f64) -> f64 {
    z * (-360.0 * z + 360.0) - 60.0
}

/// Evaluates the first-derivative quintic Hermite basis function $\psi_1(z)$.
///
/// # Arguments
/// * `z` - Normalized cell coordinate in $[0, 1]$ (dimensionless).
#[inline]
#[must_use]
pub fn psi1(z: f64) -> f64 {
    z * (z * z * (z * (-3.0 * z + 8.0) - 6.0) + 1.0)
}

/// First derivative $d\psi_1 / dz$.
#[inline]
#[must_use]
pub fn dpsi1(z: f64) -> f64 {
    z * z * (z * (-15.0 * z + 32.0) - 18.0) + 1.0
}

/// Second derivative $d^2\psi_1 / dz^2$.
#[inline]
#[must_use]
pub fn ddpsi1(z: f64) -> f64 {
    z * (z * (-60.0 * z + 96.0) - 36.0)
}

/// Third derivative $d^3\psi_1 / dz^3$.
#[inline]
#[must_use]
pub fn dddpsi1(z: f64) -> f64 {
    z * (-180.0 * z + 192.0) - 36.0
}

/// Evaluates the second-derivative quintic Hermite basis function $\psi_2(z)$.
///
/// # Arguments
/// * `z` - Normalized cell coordinate in $[0, 1]$ (dimensionless).
#[inline]
#[must_use]
pub fn psi2(z: f64) -> f64 {
    0.5 * z * z * (z * (z * (-z + 3.0) - 3.0) + 1.0)
}

/// First derivative $d\psi_2 / dz$.
#[inline]
#[must_use]
pub fn dpsi2(z: f64) -> f64 {
    0.5 * z * (z * (z * (-5.0 * z + 12.0) - 9.0) + 2.0)
}

/// Second derivative $d^2\psi_2 / dz^2$.
#[inline]
#[must_use]
#[allow(clippy::manual_midpoint)]
pub fn ddpsi2(z: f64) -> f64 {
    0.5 * (z * (z * (-20.0 * z + 36.0) - 18.0) + 2.0)
}

/// Third derivative $d^3\psi_2 / dz^3$.
#[inline]
#[must_use]
pub fn dddpsi2(z: f64) -> f64 {
    0.5 * (z * (-60.0 * z + 72.0) - 18.0)
}

/// Evaluates the 36-term biquintic Hermite tensor product `h5`.
///
/// # Arguments
/// * `fi` - 36 corner values ordered as `[f, ft, ftt, fd, fdd, fdt, fddt, fdtt,
///   fddtt]` across the 4 cell corners `[(0,0), (1,0), (0,1), (1,1)]`.
/// * `wt` - 6 temperature basis weights `[w0t, w1t, w2t, w0mt, w1mt, w2mt]`.
/// * `wd` - 6 density basis weights `[w0d, w1d, w2d, w0md, w1md, w2md]`.
#[inline]
#[must_use]
pub fn h5(fi: &[f64; 36], wt: &[f64; 6], wd: &[f64; 6]) -> f64 {
    let [w0t, w1t, w2t, w0mt, w1mt, w2mt] = *wt;
    let [w0d, w1d, w2d, w0md, w1md, w2md] = *wd;
    fi[0] * w0d * w0t
        + fi[1] * w0md * w0t
        + fi[2] * w0d * w0mt
        + fi[3] * w0md * w0mt
        + fi[4] * w0d * w1t
        + fi[5] * w0md * w1t
        + fi[6] * w0d * w1mt
        + fi[7] * w0md * w1mt
        + fi[8] * w0d * w2t
        + fi[9] * w0md * w2t
        + fi[10] * w0d * w2mt
        + fi[11] * w0md * w2mt
        + fi[12] * w1d * w0t
        + fi[13] * w1md * w0t
        + fi[14] * w1d * w0mt
        + fi[15] * w1md * w0mt
        + fi[16] * w2d * w0t
        + fi[17] * w2md * w0t
        + fi[18] * w2d * w0mt
        + fi[19] * w2md * w0mt
        + fi[20] * w1d * w1t
        + fi[21] * w1md * w1t
        + fi[22] * w1d * w1mt
        + fi[23] * w1md * w1mt
        + fi[24] * w2d * w1t
        + fi[25] * w2md * w1t
        + fi[26] * w2d * w1mt
        + fi[27] * w2md * w1mt
        + fi[28] * w1d * w2t
        + fi[29] * w1md * w2t
        + fi[30] * w1d * w2mt
        + fi[31] * w1md * w2mt
        + fi[32] * w2d * w2t
        + fi[33] * w2md * w2t
        + fi[34] * w2d * w2mt
        + fi[35] * w2md * w2mt
}

/// Evaluates the zero-derivative cubic Hermite basis function $\chi_0(z)$.
#[inline]
#[must_use]
pub fn xpsi0(z: f64) -> f64 {
    z * z * (2.0 * z - 3.0) + 1.0
}

/// First derivative $d\chi_0 / dz$.
#[inline]
#[must_use]
pub fn xdpsi0(z: f64) -> f64 {
    z * (6.0 * z - 6.0)
}

/// Second derivative $d^2\chi_0 / dz^2$.
#[inline]
#[must_use]
pub fn xddpsi0(z: f64) -> f64 {
    12.0 * z - 6.0
}

/// Third derivative $d^3\chi_0 / dz^3$.
#[inline]
#[must_use]
pub fn xdddpsi0(_z: f64) -> f64 {
    12.0
}

/// Evaluates the first-derivative cubic Hermite basis function $\chi_1(z)$.
#[inline]
#[must_use]
pub fn xpsi1(z: f64) -> f64 {
    z * (z * (z - 2.0) + 1.0)
}

/// First derivative $d\chi_1 / dz$.
#[inline]
#[must_use]
pub fn xdpsi1(z: f64) -> f64 {
    z * (3.0 * z - 4.0) + 1.0
}

/// Second derivative $d^2\chi_1 / dz^2$.
#[inline]
#[must_use]
pub fn xddpsi1(z: f64) -> f64 {
    6.0 * z - 4.0
}

/// Third derivative $d^3\chi_1 / dz^3$.
#[inline]
#[must_use]
pub fn xdddpsi1(_z: f64) -> f64 {
    6.0
}

/// Evaluates the 16-term bicubic Hermite tensor product `h3`.
///
/// # Arguments
/// * `fi` - 16 corner values ordered as `[f, ft, fd, fdt]` across the 4 cell
///   corners `[(0,0), (1,0), (0,1), (1,1)]`.
/// * `wt` - 4 temperature basis weights `[w0t, w1t, w0mt, w1mt]`.
/// * `wd` - 4 density basis weights `[w0d, w1d, w0md, w1md]`.
#[inline]
#[must_use]
pub fn h3(fi: &[f64; 16], wt: &[f64; 4], wd: &[f64; 4]) -> f64 {
    let [w0t, w1t, w0mt, w1mt] = *wt;
    let [w0d, w1d, w0md, w1md] = *wd;
    fi[0] * w0d * w0t
        + fi[1] * w0md * w0t
        + fi[2] * w0d * w0mt
        + fi[3] * w0md * w0mt
        + fi[4] * w0d * w1t
        + fi[5] * w0md * w1t
        + fi[6] * w0d * w1mt
        + fi[7] * w0md * w1mt
        + fi[8] * w1d * w0t
        + fi[9] * w1md * w0t
        + fi[10] * w1d * w0mt
        + fi[11] * w1md * w0mt
        + fi[12] * w1d * w1t
        + fi[13] * w1md * w1t
        + fi[14] * w1d * w1mt
        + fi[15] * w1md * w1mt
}

/// Precomputed 1D quintic Hermite basis weights up to third order.
#[derive(Debug, Clone, Copy)]
pub struct QuinticBasis1D {
    /// Zeroth-derivative weights `[si0, si1, si2, si0m, si1m, si2m]`.
    pub w0: [f64; 6],
    /// First-derivative weights `[dsi0, dsi1, dsi2, dsi0m, dsi1m, dsi2m]`.
    pub w1: [f64; 6],
    /// Second-derivative weights
    /// `[ddsi0, ddsi1, ddsi2, ddsi0m, ddsi1m, ddsi2m]`.
    pub w2: [f64; 6],
    /// Third-derivative weights
    /// `[dddsi0, dddsi1, dddsi2, dddsi0m, dddsi1m, dddsi2m]`.
    pub w3: [f64; 6],
}

impl QuinticBasis1D {
    /// Evaluates 1D quintic Hermite basis weights for cell coordinate `x` and
    /// step sizes `(dh, dh2, dhi, dh2i, dh3i)`.
    #[inline]
    #[must_use]
    pub fn eval(
        x: f64,
        dh: f64,
        dh2: f64,
        dhi: f64,
        dh2i: f64,
        dh3i: f64,
    ) -> Self {
        let mx = 1.0 - x;
        Self {
            w0: [
                psi0(x),
                psi1(x) * dh,
                psi2(x) * dh2,
                psi0(mx),
                -psi1(mx) * dh,
                psi2(mx) * dh2,
            ],
            w1: [
                dpsi0(x) * dhi,
                dpsi1(x),
                dpsi2(x) * dh,
                -dpsi0(mx) * dhi,
                dpsi1(mx),
                -dpsi2(mx) * dh,
            ],
            w2: [
                ddpsi0(x) * dh2i,
                ddpsi1(x) * dhi,
                ddpsi2(x),
                ddpsi0(mx) * dh2i,
                -ddpsi1(mx) * dhi,
                ddpsi2(mx),
            ],
            w3: [
                dddpsi0(x) * dh3i,
                dddpsi1(x) * dh2i,
                dddpsi2(x) * dhi,
                -dddpsi0(mx) * dh3i,
                dddpsi1(mx) * dh2i,
                -dddpsi2(mx) * dhi,
            ],
        }
    }
}

/// Precomputed 1D cubic Hermite basis weights up to third order.
#[derive(Debug, Clone, Copy)]
pub struct CubicBasis1D {
    /// Zeroth-derivative weights `[si0, si1, si0m, si1m]`.
    pub w0: [f64; 4],
    /// First-derivative weights `[dsi0, dsi1, dsi0m, dsi1m]`.
    pub w1: [f64; 4],
    /// Second-derivative weights `[ddsi0, ddsi1, ddsi0m, ddsi1m]`.
    pub w2: [f64; 4],
    /// Third-derivative weights `[dddsi0, dddsi1, dddsi0m, dddsi1m]`.
    pub w3: [f64; 4],
}

impl CubicBasis1D {
    /// Evaluates 1D cubic Hermite basis weights for cell coordinate `x` and
    /// step sizes `(dh, dhi, dh2i, dh3i)`.
    #[inline]
    #[must_use]
    pub fn eval(x: f64, dh: f64, dhi: f64, dh2i: f64, dh3i: f64) -> Self {
        let mx = 1.0 - x;
        Self {
            w0: [xpsi0(x), xpsi1(x) * dh, xpsi0(mx), -xpsi1(mx) * dh],
            w1: [
                xdpsi0(x) * dhi,
                xdpsi1(x),
                -xdpsi0(mx) * dhi,
                xdpsi1(mx),
            ],
            w2: [
                xddpsi0(x) * dh2i,
                xddpsi1(x) * dhi,
                xddpsi0(mx) * dh2i,
                -xddpsi1(mx) * dhi,
            ],
            w3: [
                xdddpsi0(x) * dh3i,
                xdddpsi1(x) * dh2i,
                -xdddpsi0(mx) * dh3i,
                xdddpsi1(mx) * dh2i,
            ],
        }
    }
}
