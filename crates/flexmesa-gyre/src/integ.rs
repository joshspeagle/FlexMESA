// SPDX-License-Identifier: GPL-3.0-only
//! Trapezoidal quadrature: GYRE `integ_m` (`gyre/gyre/src/math/integ_m.fypp`),
//! real variants (the ones MESA's GYRE calls reach).

/// `integrate(x, y [, mask])` (`integrate_r_`, `integ_m.fypp:61-91`):
/// `SUM(0.5*(y(2:)+y(:n-1))*(x(2:)-x(:n-1)) [, MASK=mask(2:) .AND. mask(:n-1)])`.
///
/// gfortran inlines `SUM` as a sequential accumulation from `+0` over
/// increasing index; each term is `(0.5*(y(i+1)+y(i)))*(x(i+1)-x(i))`.
///
/// # Panics
///
/// Panics if `y` (or `mask`) does not have the length of `x`.
#[must_use]
pub fn integrate(x: &[f64], y: &[f64], mask: Option<&[bool]>) -> f64 {
    assert_eq!(x.len(), y.len(), "x and y lengths differ");
    if let Some(m) = mask {
        assert_eq!(m.len(), x.len(), "mask length differs");
    }
    let mut s = 0.0_f64;
    for i in 0..x.len().saturating_sub(1) {
        if let Some(m) = mask {
            if !(m[i + 1] && m[i]) {
                continue;
            }
        }
        s += 0.5 * (y[i + 1] + y[i]) * (x[i + 1] - x[i]);
    }
    s
}

/// `integral(x, y)` (`integral_r_`, `integ_m.fypp:95-120`): running
/// trapezoidal integral with `int_y(1) = 0`.
///
/// # Panics
///
/// Panics if `y` does not have the length of `x`.
#[must_use]
pub fn integral(x: &[f64], y: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), y.len(), "x and y lengths differ");
    let n = x.len();
    let mut out = vec![0.0_f64; n];
    for i in 1..n {
        out[i] = out[i - 1] + 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]);
    }
    out
}
