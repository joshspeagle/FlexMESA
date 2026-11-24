// SPDX-License-Identifier: GPL-3.0-only
//! GYRE grids: `point_t` (`grid/point_m.fypp`) and `grid_t`
//! (`grid/grid_m.fypp`), plus the scaffold constructor
//! (`grid/grid_scaffold_m.fypp:54-82`).

use crate::GyreError;

/// `point_t`: a segment index `s` and abscissa `x`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// Segment index (1-based).
    pub s: i32,
    /// Fractional radius `x = r/R`.
    pub x: f64,
}

/// `grid_t`: points ordered in `x`, a new segment starting at each
/// repeated abscissa.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Grid {
    /// The points, length `n`.
    pub pt: Vec<Point>,
}

impl Grid {
    /// `grid_t(x)` (`grid_t_x_`, `grid_m.fypp:71-121`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Assert`] for a segment with fewer than two points.
    pub fn from_x(x: &[f64]) -> Result<Self, GyreError> {
        let n = x.len();
        let mut pt = Vec::with_capacity(n);
        if n > 0 {
            let mut s = 1;
            pt.push(Point { s, x: x[0] });
            for j in 1..n {
                if x[j] == x[j - 1] {
                    s += 1;
                }
                pt.push(Point { s, x: x[j] });
            }
        }
        let gr = Self { pt };
        // check_loop (grid_m.fypp:113-115)
        if n > 0 {
            for s in 1..=gr.pt[n - 1].s {
                if gr.pt.iter().filter(|p| p.s == s).count() < 2 {
                    return Err(GyreError::Assert("degenerate segment"));
                }
            }
        }
        Ok(gr)
    }

    /// `grid_t(gr_base, x_i, x_o)` (`grid_t_nest_`, `grid_m.fypp:125-165`).
    #[must_use]
    pub fn nest(gr_base: &Self, x_i: f64, x_o: f64) -> Self {
        let x_i_ = fmax(x_i, gr_base.x_i());
        let x_o_ = fmin(x_o, gr_base.x_o());
        let pt_i = gr_base.pt_x(x_i_, false);
        let pt_o = gr_base.pt_x(x_o_, true);
        let mut pt = vec![pt_i];
        pt.extend(
            gr_base
                .pt
                .iter()
                .filter(|p| p.x > x_i_ && p.x < x_o_)
                .copied(),
        );
        pt.push(pt_o);
        Self { pt }
    }

    /// `grid_t(gr_base, split)` (`grid_t_split_`, `grid_m.fypp:169-225`).
    ///
    /// # Errors
    ///
    /// [`GyreError::Assert`] when a split straddles a segment boundary.
    pub fn split(gr_base: &Self, split: &[bool]) -> Result<Self, GyreError> {
        let n_base = gr_base.n();
        let mut pt =
            Vec::with_capacity(n_base + split.iter().filter(|&&b| b).count());
        for j_base in 0..n_base - 1 {
            let pt_a = gr_base.pt[j_base];
            let pt_b = gr_base.pt[j_base + 1];
            pt.push(pt_a);
            if split[j_base] {
                if pt_a.s != pt_b.s {
                    return Err(GyreError::Assert(
                        "attempt to add points at segment boundary",
                    ));
                }
                pt.push(Point {
                    s: pt_a.s,
                    x: 0.5 * (pt_a.x + pt_b.x),
                });
            }
        }
        pt.push(gr_base.pt[n_base - 1]);
        Ok(Self { pt })
    }

    /// Number of points `n`.
    #[must_use]
    pub fn n(&self) -> usize {
        self.pt.len()
    }

    /// `pt_i()`.
    #[must_use]
    pub fn pt_i(&self) -> Point {
        self.pt[0]
    }

    /// `pt_o()`.
    #[must_use]
    pub fn pt_o(&self) -> Point {
        self.pt[self.n() - 1]
    }

    /// `x_i()`.
    #[must_use]
    pub fn x_i(&self) -> f64 {
        self.pt_i().x
    }

    /// `x_o()`.
    #[must_use]
    pub fn x_o(&self) -> f64 {
        self.pt_o().x
    }

    /// `s_i()`.
    #[must_use]
    pub fn s_i(&self) -> i32 {
        self.pt_i().s
    }

    /// `s_o()`.
    #[must_use]
    pub fn s_o(&self) -> i32 {
        self.pt_o().s
    }

    /// `pt_x(x, back)` (`grid_m.fypp:263-281`).
    #[must_use]
    pub fn pt_x(&self, x: f64, back: bool) -> Point {
        Point {
            s: self.s_x(x, back),
            x,
        }
    }

    /// `s_x(x, back)` (`grid_m.fypp:327-376`). When no segment contains
    /// `x`, the loop runs to completion and the DO variable holds
    /// `s_b + ds`, which is what is returned.
    #[must_use]
    pub fn s_x(&self, x: f64, back: bool) -> i32 {
        let (s_a, s_b, ds) = if back {
            (self.pt[self.n() - 1].s, self.pt[0].s, -1)
        } else {
            (self.pt[0].s, self.pt[self.n() - 1].s, 1)
        };
        let mut s = s_a;
        while (ds > 0 && s <= s_b) || (ds < 0 && s >= s_b) {
            let j_i = self.j_s_i(s);
            let j_o = self.j_s_o(s);
            if x >= self.pt[j_i - 1].x && x <= self.pt[j_o - 1].x {
                return s;
            }
            s += ds;
        }
        s
    }

    /// `j_s_i(s)` (1-based, `grid_m.fypp:422-441`); `n + 1` if absent.
    #[must_use]
    pub fn j_s_i(&self, s: i32) -> usize {
        for (j, p) in self.pt.iter().enumerate() {
            if p.s == s {
                return j + 1;
            }
        }
        self.n() + 1
    }

    /// `j_s_o(s)` (1-based, `grid_m.fypp:445-464`); `0` if absent.
    #[must_use]
    pub fn j_s_o(&self, s: i32) -> usize {
        for j in (0..self.n()).rev() {
            if self.pt[j].s == s {
                return j + 1;
            }
        }
        0
    }

    /// The abscissae `pt(:)%x`.
    #[must_use]
    pub fn xs(&self) -> Vec<f64> {
        self.pt.iter().map(|p| p.x).collect()
    }
}

/// Fortran `MAX(a, b)` for two reals as gfortran 15 compiles it at `-O2`
/// (`a > b ? a : b` with the first argument kept on ties and NaN `b`).
#[must_use]
#[inline]
pub fn fmax(a: f64, b: f64) -> f64 {
    if b > a || a.is_nan() {
        b
    } else {
        a
    }
}

/// Fortran `MIN(a, b)` (counterpart of [`fmax`]).
#[must_use]
#[inline]
pub fn fmin(a: f64, b: f64) -> f64 {
    if b < a || a.is_nan() {
        b
    } else {
        a
    }
}
