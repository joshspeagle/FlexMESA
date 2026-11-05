//! Dormand-Prince 5(4) Butcher tableau and Shampine dense-output polynomial.
//!
//! Stores the rational Dormand-Prince 5(4) stage abscissas $c_i$, stage weights
//! $a_{ij}$, embedded 5(4) error coefficients $e_i$, and Shampine (`1986`)
//! 4th-order continuous dense-output polynomial coefficients $d_i$ (`contd5` in
//! `num/private/mod_dopri5.f`), together with [`DopriResult`].

/// Dormand-Prince 5(4) stage abscissas $c_2, \dots, c_6$.
pub(crate) const C: [f64; 5] = [0.2, 0.3, 0.8, 8.0 / 9.0, 1.0];

/// Stage 2 coupling coefficient $a_{21}$.
pub(crate) const A21: f64 = 0.2;

/// Stage 3 coupling coefficients $[a_{31}, a_{32}]$.
pub(crate) const A3: [f64; 2] = [3.0 / 40.0, 9.0 / 40.0];

/// Stage 4 coupling coefficients $[a_{41}, a_{42}, a_{43}]$.
pub(crate) const A4: [f64; 3] = [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0];

/// Stage 5 coupling coefficients $[a_{51}, a_{52}, a_{53}, a_{54}]$.
pub(crate) const A5: [f64; 4] = [
    19_372.0 / 6_561.0,
    -25_360.0 / 2_187.0,
    64_448.0 / 6_561.0,
    -212.0 / 729.0,
];

/// Stage 6 coupling coefficients $[a_{61}, \dots, a_{65}]$.
pub(crate) const A6: [f64; 5] = [
    9_017.0 / 3_168.0,
    -355.0 / 33.0,
    46_732.0 / 5_247.0,
    49.0 / 176.0,
    -5_103.0 / 18_656.0,
];

/// Stage 7 (FSAL 5th-order solution) weights $[a_{71}, \dots, a_{76}]$.
pub(crate) const A7: [f64; 6] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1_113.0,
    125.0 / 192.0,
    -2_187.0 / 6_784.0,
    11.0 / 84.0,
];

/// Embedded 5(4) local error weights $[e_1, \dots, e_7]$.
pub(crate) const E: [f64; 7] = [
    71.0 / 57_600.0,
    0.0,
    -71.0 / 16_695.0,
    71.0 / 1_920.0,
    -17_253.0 / 339_200.0,
    22.0 / 525.0,
    -1.0 / 40.0,
];

/// Shampine 4th-order dense-output coefficients $[d_1, \dots, d_7]$.
pub(crate) const D: [f64; 7] = [
    -12_715_105_075.0 / 11_282_082_432.0,
    0.0,
    87_487_479_700.0 / 32_700_410_799.0,
    -10_690_763_975.0 / 1_880_347_072.0,
    701_980_252_875.0 / 199_316_789_632.0,
    -1_453_857_185.0 / 822_651_844.0,
    69_997_945.0 / 29_380_423.0,
];

/// Result of an adaptive Dormand-Prince (`dopri5` / `dop853`) integration.
#[derive(Debug, Clone)]
pub struct DopriResult {
    /// Final state vector $y(x_{\mathrm{end}})$ of length $n$.
    pub y: Vec<f64>,
    /// Total number of right-hand side evaluations (`nfcn`).
    pub nfcn: usize,
    /// Total number of steps attempted (`nstep`).
    pub nstep: usize,
    /// Number of accepted steps (`naccpt`).
    pub naccpt: usize,
    /// Number of rejected steps (`nrejct`).
    pub nrejct: usize,
    /// Dense-output polynomial history `(x_old, h, rcont[5][n])` per step.
    pub history: Vec<(f64, f64, [Vec<f64>; 5])>,
}

impl DopriResult {
    /// Evaluates Shampine's 4th-order dense output polynomial (`contd5`) at any
    /// abscissa `x_query` within the integration interval using normalized
    /// step fraction $\theta = (x - x_{\mathrm{old}}) / h$ and nested Horner
    /// evaluation over $r_1, \dots, r_5$.
    #[must_use]
    pub fn dense_eval(&self, x_query: f64) -> Vec<f64> {
        if self.history.is_empty() {
            return self.y.clone();
        }
        let mut chosen = &self.history[self.history.len() - 1];
        for seg in &self.history {
            let (x0, h, _) = *seg;
            let xmin = x0.min(x0 + h);
            let xmax = x0.max(x0 + h);
            if x_query >= xmin - 1e-14 && x_query <= xmax + 1e-14 {
                chosen = seg;
                break;
            }
        }
        let (x0, h, ref rc) = *chosen;
        let theta = if h == 0.0 { 0.0 } else { (x_query - x0) / h };
        let theta1 = 1.0 - theta;
        let n = rc[0].len();
        (0..n)
            .map(|i| {
                rc[0][i]
                    + theta
                        * (rc[1][i]
                            + theta1
                                * (rc[2][i]
                                    + theta * (rc[3][i] + theta1 * rc[4][i])))
            })
            .collect()
    }
}
