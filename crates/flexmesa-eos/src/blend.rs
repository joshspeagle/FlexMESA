// SPDX-License-Identifier: LGPL-3.0-or-later
//! Polygonal winding-number containment, minimum-distance metrics, and
//! $C^2$-continuous quintic EOS blending.
//!
//! Ports upstream MESA `eos/private/eos_blend.f90` (Moscato et al. 2019
//! winding-number algorithm and differentiable line-segment distance) and
//! `Do_Blend` from `eos/private/eosdt_support.f90`.

use flexmesa_autodiff::AutoDiffReal2VarOrder1;
use flexmesa_const::LN10;

use crate::types::{
    EosEvalOutput, NUM_EOS_BASIC_RESULTS, NUM_EOS_D_DXA_RESULTS,
};

/// Regularization floor (`1e-10_f32 as f64`, matching the single-precision
/// literal `real(dp), parameter :: eps = 1e-10` in `eos_blend.f90` line 136)
/// added in quadrature on interior line-segment projections to prevent
/// derivative singularities near polygon boundaries.
const LINE_SEGMENT_EPS: f64 = 1.0e-10_f32 as f64;

/// Determines which quadrant (`1..=4`) a 2D point `(x, y)` lies in.
///
/// Matches `quadrant` in `eos_blend.f90`:
/// - Quadrant 1: $x \ge 0$ and $y \ge 0$
/// - Quadrant 2: $x < 0$ and $y \ge 0$
/// - Quadrant 3: $x < 0$ and $y < 0$
/// - Quadrant 4: $x \ge 0$ and $y < 0$
#[must_use]
#[inline]
pub fn quadrant(p: [f64; 2]) -> i32 {
    if p[0] >= 0.0 && p[1] >= 0.0 {
        1
    } else if p[0] < 0.0 && p[1] >= 0.0 {
        2
    } else if p[0] < 0.0 && p[1] < 0.0 {
        3
    } else {
        4
    }
}

/// Zero-allocation winding number around `origin`.
#[must_use]
fn winding_number_around(coords: &[[f64; 2]], origin: [f64; 2]) -> i32 {
    let n = coords.len();
    if n == 0 {
        return 0;
    }
    let mut w = 0_i32;
    for i in 0..n {
        let c_start = coords[i];
        let c_end = if i + 1 == n { coords[0] } else { coords[i + 1] };
        let x_start = [c_start[0] - origin[0], c_start[1] - origin[1]];
        let x_end = [c_end[0] - origin[0], c_end[1] - origin[1]];
        let dist = [x_end[0] - x_start[0], x_end[1] - x_start[1]];
        let determinant = x_start[1] * dist[0] - x_start[0] * dist[1];
        let q_start = quadrant(x_start);
        let q_end = quadrant(x_end);
        let increment = if q_start == q_end {
            0
        } else if q_end == (q_start % 4) + 1 {
            1
        } else if q_start == (q_end % 4) + 1 {
            -1
        } else if determinant < 0.0 {
            2
        } else {
            -2
        };
        w += increment;
    }
    w
}

/// Computes the topological winding number of a closed polygon around the
/// origin `(0, 0)` using the quadrant-transition algorithm of Moscato, Titolo,
/// Feliu, and Munoz (2019), matching `winding_number` in `eos_blend.f90`.
#[must_use]
pub fn winding_number(coords: &[[f64; 2]]) -> i32 {
    winding_number_around(coords, [0.0, 0.0])
}

/// Returns `true` if the point `p` (evaluated at its primal values) is
/// contained inside the closed polygon `coords`, matching `is_contained` in
/// `eos_blend.f90`.
#[must_use]
#[inline]
pub fn is_contained(
    coords: &[[f64; 2]],
    p: [AutoDiffReal2VarOrder1; 2],
) -> bool {
    is_contained_scalar(coords, [p[0].val, p[1].val])
}

/// Convenience scalar wrapper for [`is_contained`] taking a plain `[f64; 2]`.
#[must_use]
#[inline]
pub fn is_contained_scalar(coords: &[[f64; 2]], p: [f64; 2]) -> bool {
    winding_number_around(coords, p) != 0
}

/// Computes the differentiable minimum distance from point `p` to the line
/// segment connecting `line_start` and `line_end`, propagating first-order
/// partial derivatives via [`AutoDiffReal2VarOrder1`].
///
/// Matches `min_distance_from_point_to_line_segment` in `eos_blend.f90`. Note
/// that the regularization `eps = 1e-10` is added in quadrature only when the
/// projection parameter $\lambda \in [0, 1]$ lies in the segment interior.
#[must_use]
pub fn min_distance_from_point_to_line_segment(
    line_start: [f64; 2],
    line_end: [f64; 2],
    p: [AutoDiffReal2VarOrder1; 2],
) -> AutoDiffReal2VarOrder1 {
    let diff_start = [p[0] - line_start[0], p[1] - line_start[1]];
    let diff_end = [p[0] - line_end[0], p[1] - line_end[1]];
    let diff_line = [
        AutoDiffReal2VarOrder1::constant(line_end[0] - line_start[0]),
        AutoDiffReal2VarOrder1::constant(line_end[1] - line_start[1]),
    ];
    let length_squared = diff_line[0].pow2() + diff_line[1].pow2();
    let lambda = (diff_start[0] * diff_line[0] + diff_start[1] * diff_line[1])
        / length_squared;

    if lambda.val < 0.0 {
        (diff_start[0].pow2() + diff_start[1].pow2()).sqrt()
    } else if lambda.val > 1.0 {
        (diff_end[0].pow2() + diff_end[1].pow2()).sqrt()
    } else {
        let near0 = line_start[0] + lambda * diff_line[0] - p[0];
        let near1 = line_start[1] + lambda * diff_line[1] - p[1];
        let eps_sq = LINE_SEGMENT_EPS * LINE_SEGMENT_EPS;
        (near0.pow2() + near1.pow2() + eps_sq).sqrt()
    }
}

/// Computes the minimum distance from point `p` to any edge of the closed
/// polygon `coords`, propagating first-order partial derivatives.
///
/// Matches `min_distance_to_polygon` in `eos_blend.f90`.
#[must_use]
pub fn min_distance_to_polygon(
    coords: &[[f64; 2]],
    p: [AutoDiffReal2VarOrder1; 2],
) -> AutoDiffReal2VarOrder1 {
    let n = coords.len();
    let mut d = AutoDiffReal2VarOrder1::constant(1.0e99);
    for i in 0..n {
        let line_start = coords[i];
        let line_end = if i + 1 == n { coords[0] } else { coords[i + 1] };
        let line_dist =
            min_distance_from_point_to_line_segment(line_start, line_end, p);
        d = d.min(line_dist);
    }
    d
}

/// Evaluates the $C^2$-continuous quintic smoothing polynomial and its
/// derivative factor on $\alpha_0 \in [0, 1]$:
/// $$S(\alpha_0) = -\alpha_0^3 (-10 + \alpha_0 (15 - 6 \alpha_0))
///   = 10 \alpha_0^3 - 15 \alpha_0^4 + 6 \alpha_0^5,$$
/// $$S'(\alpha_0) = 30 (\alpha_0 - 1)^2 \alpha_0^2.$$
///
/// Returns `(s, ds_dalpha0)` with clamping outside `[0, 1]`.
#[must_use]
#[inline]
pub fn quintic_blend(alpha0: f64) -> (f64, f64) {
    if alpha0 <= 0.0 {
        (0.0, 0.0)
    } else if alpha0 >= 1.0 {
        (1.0, 0.0)
    } else {
        let a2 = alpha0 * alpha0;
        let a3 = a2 * alpha0;
        let s = -a3 * (-10.0 + alpha0 * (15.0 - 6.0 * alpha0));
        let ds = 30.0 * (alpha0 - 1.0) * (alpha0 - 1.0) * a2;
        (s, ds)
    }
}

/// Blends two [`EosEvalOutput`] evaluations (`res_1` with weight $\alpha$ and
/// `res_2` with weight $\beta = 1 - \alpha$) including the chain-rule terms
/// from $\partial\alpha/\partial\log_{10}T$ and
/// $\partial\alpha/\partial\log_{10}\rho$.
///
/// Matches `Do_Blend` in `eos/private/eosdt_support.f90`.
#[must_use]
pub fn blend_eos_results(
    alpha_in: f64,
    d_alpha_dlog_t_in: f64,
    d_alpha_dlog_rho_in: f64,
    linear_blend: bool,
    res_1: &EosEvalOutput,
    res_2: &EosEvalOutput,
) -> EosEvalOutput {
    let d_alpha0_dlnt = d_alpha_dlog_t_in / LN10;
    let d_alpha0_dlnd = d_alpha_dlog_rho_in / LN10;
    let (alpha, d_alpha_dlnd, d_alpha_dlnt) = if linear_blend {
        (alpha_in, d_alpha0_dlnd, d_alpha0_dlnt)
    } else {
        let alpha0 = alpha_in;
        let a2 = alpha0 * alpha0;
        let a3 = a2 * alpha0;
        let a = -a3 * (-10.0 + alpha0 * (15.0 - 6.0 * alpha0));
        let big_a = 30.0 * (alpha0 - 1.0) * (alpha0 - 1.0) * a2;
        (a, big_a * d_alpha0_dlnd, big_a * d_alpha0_dlnt)
    };

    let beta = 1.0 - alpha;
    let d_beta_dlnd = -d_alpha_dlnd;
    let d_beta_dlnt = -d_alpha_dlnt;

    let mut res = [0.0_f64; NUM_EOS_BASIC_RESULTS];
    let mut d_dlnd = [0.0_f64; NUM_EOS_BASIC_RESULTS];
    let mut d_dln_t = [0.0_f64; NUM_EOS_BASIC_RESULTS];

    for j in 0..NUM_EOS_BASIC_RESULTS {
        res[j] = alpha * res_1.res[j] + beta * res_2.res[j];
        d_dlnd[j] = alpha * res_1.d_dlnd[j]
            + beta * res_2.d_dlnd[j]
            + d_alpha_dlnd * res_1.res[j]
            + d_beta_dlnd * res_2.res[j];
        d_dln_t[j] = alpha * res_1.d_dln_t[j]
            + beta * res_2.d_dln_t[j]
            + d_alpha_dlnt * res_1.res[j]
            + d_beta_dlnt * res_2.res[j];
    }

    let species = res_1.d_dxa.len().max(res_2.d_dxa.len());
    let mut d_dxa = vec![[0.0_f64; NUM_EOS_D_DXA_RESULTS]; species];
    for (i, row) in d_dxa.iter_mut().enumerate() {
        let r1 = res_1
            .d_dxa
            .get(i)
            .copied()
            .unwrap_or([0.0; NUM_EOS_D_DXA_RESULTS]);
        let r2 = res_2
            .d_dxa
            .get(i)
            .copied()
            .unwrap_or([0.0; NUM_EOS_D_DXA_RESULTS]);
        for (val, (&v1, &v2)) in row.iter_mut().zip(r1.iter().zip(r2.iter())) {
            *val = alpha * v1 + beta * v2;
        }
    }

    EosEvalOutput {
        res,
        d_dlnd,
        d_dln_t,
        d_dxa,
    }
}
