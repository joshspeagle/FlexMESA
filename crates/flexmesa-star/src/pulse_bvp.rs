// SPDX-License-Identifier: LGPL-3.0-or-later
//! 2nd-order ($\ell = 0$) and 4th-order ($\ell \ge 1$) Gauss-Legendre Magnus
//! (`MAGNUS_GL2`) oscillation boundary-value problem (BVP) structures and
//! 2nd-compound-matrix (Plücker) shooting determinant (`gyre/src/diff/magnus_gl2_diff_m.fypp`,
//! `adipls/adipack.c/adipls/adirhs.c.d.f`, and `setbcs.n.d.f`, resolving Issue #4).

#![allow(missing_docs)]

/// Outer mechanical boundary condition at $x = x_{\mathrm{surf}}$ (`gyre` & `adipls`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OuterBoundKind {
    /// Zero Lagrangian pressure perturbation $\delta P = 0 \implies y_1 - y_2 = 0$.
    Vacuum,
    /// Isothermal atmosphere wave-matching condition (`istsbc = 1` in `setbcs.n.d.f`).
    AdiplsIsothermal,
}

/// Surface normalization convention for normalized mode inertia $E_{\mathrm{norm}}$.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InertiaNormKind {
    /// Normalize by $|\tilde{\xi}_r|^2 + \ell(\ell+1)|\tilde{\xi}_h|^2$ (`'BOTH'` in GYRE).
    Both,
    /// Normalize by $|\tilde{\xi}_r|^2$ (`'RADIAL'` in GYRE, `iekinr = 0` in ADIPLS).
    Radial,
}

/// Dimensionless pulsation structure point at fractional radius $x = r / R_\star \in [0, 1]$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BvpPoint {
    pub x: f64,
    pub v_2: f64,
    pub a_star: f64,
    pub u: f64,
    pub c_1: f64,
    pub gamma_1: f64,
    pub nabla_ad: f64,
    pub ups_t: f64,
    pub nabla: f64,
    pub c_rad: f64,
    pub c_thn: f64,
    pub c_thk: f64,
    pub c_eps: f64,
    pub eps_rho: f64,
    pub eps_t: f64,
    pub kap_rho: f64,
    pub kap_t: f64,
}

/// Stellar oscillation BVP model with precomputed midpoint structure coefficients.
#[derive(Debug, Clone)]
pub struct BvpModel {
    pub m_star: f64,
    pub r_star: f64,
    pub l_star: f64,
    pub cgrav: f64,
    pub pts: Vec<BvpPoint>,
    pub mids: Vec<BvpPoint>,
}

/// Solved adiabatic and non-adiabatic oscillation eigenmode on a [`BvpModel`].
#[derive(Debug, Clone, PartialEq)]
pub struct BvpEigenmode {
    pub l: usize,
    pub n_p: i32,
    pub n_g: i32,
    pub n_pg: i32,
    pub omega: f64,
    pub omega_int: f64,
    pub freq_uhz: f64,
    pub freq_var_uhz: f64,
    pub e_norm: f64,
    pub e_adipls: f64,
    pub omega_imag: f64,
    pub growth_rate_s: f64,
    pub stellingwerf_eta: f64,
    pub y: Vec<[f64; 4]>,
}

/// Computes Steffen (1990) monotonic cubic spline derivatives $\mathrm{d}f/\mathrm{d}x$
/// with clamped central derivative $f'(x_0) = 0$.
#[must_use]
pub fn steffen_slopes(x: &[f64], f: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let mut h = vec![0.0; n - 1];
    let mut s = vec![0.0; n - 1];
    for i in 0..(n - 1) {
        h[i] = (x[i + 1] - x[i]).max(1.0e-30);
        s[i] = (f[i + 1] - f[i]) / h[i];
    }
    let mut df = vec![0.0; n];
    for i in 1..(n - 1) {
        let p = (s[i - 1] * h[i] + s[i] * h[i - 1]) / (h[i - 1] + h[i]);
        df[i] = if s[i - 1] * s[i] <= 0.0 {
            0.0
        } else {
            let lim = 2.0 * s[i - 1].abs().min(s[i].abs());
            s[i - 1].signum() * p.abs().min(lim)
        };
    }
    df[n - 1] = s[n - 2];
    df
}

impl BvpModel {
    /// Builds a [`BvpModel`] from center-to-surface [`BvpPoint`]s, prepending $x = 0$
    /// by parabolic extrapolation if $x_0 > 0$ and precomputing Steffen cubic midpoints.
    #[must_use]
    pub fn from_points(
        m_star: f64,
        r_star: f64,
        l_star: f64,
        cgrav: f64,
        raw_pts: Vec<BvpPoint>,
    ) -> Self {
        let mut pts: Vec<BvpPoint> = Vec::with_capacity(raw_pts.len() + 1);
        for p in raw_pts {
            if let Some(last) = pts.last_mut() {
                if (p.x - last.x).abs() < 1.0e-14 {
                    *last = p;
                    continue;
                }
            }
            pts.push(p);
        }
        if pts.len() >= 2 && pts[0].x > 0.0 && pts[0].x < 0.01 {
            let (x1_2, x2_2) = (pts[0].x * pts[0].x, pts[1].x * pts[1].x);
            let denom = (x2_2 - x1_2).max(1.0e-30);
            let para = |v1: f64, v2: f64| (v1 * x2_2 - v2 * x1_2) / denom;
            let (p1, p2) = (pts[0], pts[1]);
            pts.insert(
                0,
                BvpPoint {
                    x: 0.0,
                    v_2: para(p1.v_2, p2.v_2),
                    a_star: 0.0,
                    u: 3.0,
                    c_1: para(p1.c_1, p2.c_1),
                    gamma_1: para(p1.gamma_1, p2.gamma_1),
                    nabla_ad: para(p1.nabla_ad, p2.nabla_ad),
                    ups_t: para(p1.ups_t, p2.ups_t),
                    nabla: para(p1.nabla, p2.nabla),
                    c_rad: para(p1.c_rad, p2.c_rad),
                    c_thn: para(p1.c_thn, p2.c_thn),
                    c_thk: para(p1.c_thk, p2.c_thk),
                    c_eps: para(p1.c_eps, p2.c_eps),
                    eps_rho: para(p1.eps_rho, p2.eps_rho),
                    eps_t: para(p1.eps_t, p2.eps_t),
                    kap_rho: para(p1.kap_rho, p2.kap_rho),
                    kap_t: para(p1.kap_t, p2.kap_t),
                },
            );
        }
        let n = pts.len();
        let xs: Vec<f64> = pts.iter().map(|p| p.x).collect();
        let mid_col = |ext: fn(&BvpPoint) -> f64| -> Vec<f64> {
            let fs: Vec<f64> = pts.iter().map(ext).collect();
            let dfs = steffen_slopes(&xs, &fs);
            (0..n.saturating_sub(1))
                .map(|i| {
                    0.5 * (fs[i] + fs[i + 1])
                        + 0.125 * (xs[i + 1] - xs[i]) * (dfs[i] - dfs[i + 1])
                })
                .collect()
        };
        let (v2_m, as_m, u_m, c1_m, g1_m) = (
            mid_col(|p| p.v_2),
            mid_col(|p| p.a_star),
            mid_col(|p| p.u),
            mid_col(|p| p.c_1),
            mid_col(|p| p.gamma_1),
        );
        let (nad_m, ut_m, nab_m, crad_m, cthn_m, cthk_m, ceps_m) = (
            mid_col(|p| p.nabla_ad),
            mid_col(|p| p.ups_t),
            mid_col(|p| p.nabla),
            mid_col(|p| p.c_rad),
            mid_col(|p| p.c_thn),
            mid_col(|p| p.c_thk),
            mid_col(|p| p.c_eps),
        );
        let (erho_m, et_m, krho_m, kt_m) = (
            mid_col(|p| p.eps_rho),
            mid_col(|p| p.eps_t),
            mid_col(|p| p.kap_rho),
            mid_col(|p| p.kap_t),
        );
        let mids = (0..n.saturating_sub(1))
            .map(|i| BvpPoint {
                x: 0.5 * (xs[i] + xs[i + 1]),
                v_2: v2_m[i],
                a_star: as_m[i],
                u: u_m[i],
                c_1: c1_m[i],
                gamma_1: g1_m[i],
                nabla_ad: nad_m[i],
                ups_t: ut_m[i],
                nabla: nab_m[i],
                c_rad: crad_m[i],
                c_thn: cthn_m[i],
                c_thk: cthk_m[i],
                c_eps: ceps_m[i],
                eps_rho: erho_m[i],
                eps_t: et_m[i],
                kap_rho: krho_m[i],
                kap_t: kt_m[i],
            })
            .collect();
        Self {
            m_star,
            r_star,
            l_star,
            cgrav,
            pts,
            mids,
        }
    }
}

#[must_use]
pub(crate) fn expm2(m: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let tr = 0.5 * (m[0][0] + m[1][1]);
    let (a00, a01, a10, a11) = (m[0][0] - tr, m[0][1], m[1][0], m[1][1] - tr);
    let d2 = a00 * a00 + a01 * a10;
    let et = tr.exp();
    let (c, s) = if d2 > 1.0e-16 {
        let d = d2.sqrt();
        (d.cosh(), d.sinh() / d)
    } else if d2 < -1.0e-16 {
        let d = (-d2).sqrt();
        (d.cos(), d.sin() / d)
    } else {
        (1.0 + 0.5 * d2, 1.0 + d2 / 6.0)
    };
    [
        [et * (c + s * a00), et * s * a01],
        [et * s * a10, et * (c + s * a11)],
    ]
}

#[must_use]
fn expm_taylor<const N: usize>(
    a: &[[f64; N]; N],
    max_scale: u32,
) -> [[f64; N]; N] {
    let norm = a
        .iter()
        .map(|r| r.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    let k_scale = if norm > 0.25 {
        ((norm * 4.0).log2().ceil() as u32).min(max_scale)
    } else {
        0
    };
    let scale = 0.5_f64.powi(k_scale as i32);
    let mut m = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            m[i][j] = a[i][j] * scale;
        }
    }
    let mul = |x: &[[f64; N]; N], y: &[[f64; N]; N]| -> [[f64; N]; N] {
        let mut z = [[0.0; N]; N];
        for i in 0..N {
            for k in 0..N {
                let xik = x[i][k];
                for j in 0..N {
                    z[i][j] += xik * y[k][j];
                }
            }
        }
        z
    };
    let mut sum = [[0.0; N]; N];
    let mut term = [[0.0; N]; N];
    for i in 0..N {
        sum[i][i] = 1.0;
        term[i][i] = 1.0;
    }
    for deg in 1..=10 {
        term = mul(&term, &m);
        let inv_d = 1.0 / (deg as f64);
        for i in 0..N {
            for j in 0..N {
                term[i][j] *= inv_d;
                sum[i][j] += term[i][j];
            }
        }
    }
    for _ in 0..k_scale {
        sum = mul(&sum, &sum);
    }
    sum
}

#[must_use]
fn step_compound6(a2: &[[f64; 6]; 6], w: [f64; 6]) -> [f64; 6] {
    let exp_a2 = expm_taylor(a2, 20);
    let mut out = [0.0; 6];
    for i in 0..6 {
        for j in 0..6 {
            out[i] += exp_a2[i][j] * w[j];
        }
    }
    out
}

#[must_use]
pub(crate) fn expm4(a: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    expm_taylor(&a, 16)
}

#[must_use]
pub(crate) fn ad_matrix4(
    pt: &BvpPoint,
    l: usize,
    om2: f64,
    dlnx: f64,
) -> [[f64; 4]; 4] {
    let lf = l as f64;
    let lam = lf * (lf + 1.0);
    let vg = pt.v_2 * pt.x * pt.x / pt.gamma_1;
    let (a, u, c1_om2) = (pt.a_star, pt.u, (pt.c_1 * om2).max(1.0e-30));
    [
        [
            (vg - 1.0 - lf) * dlnx,
            (lam / c1_om2 - vg) * dlnx,
            (lam / c1_om2) * dlnx,
            0.0,
        ],
        [(c1_om2 - a) * dlnx, (a - u + 3.0 - lf) * dlnx, 0.0, -dlnx],
        [0.0, 0.0, (3.0 - u - lf) * dlnx, dlnx],
        [
            (u * a) * dlnx,
            (u * vg) * dlnx,
            lam * dlnx,
            (2.0 - u - lf) * dlnx,
        ],
    ]
}

#[must_use]
fn compound_matrix6(a: [[f64; 4]; 4]) -> [[f64; 6]; 6] {
    const PAIRS: [(usize, usize); 6] =
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    const PAIR_LUT: [[(usize, f64); 4]; 4] = [
        [(0, 0.0), (0, 1.0), (1, 1.0), (2, 1.0)],
        [(0, -1.0), (0, 0.0), (3, 1.0), (4, 1.0)],
        [(1, -1.0), (3, -1.0), (0, 0.0), (5, 1.0)],
        [(2, -1.0), (4, -1.0), (5, -1.0), (0, 0.0)],
    ];
    let mut a2 = [[0.0; 6]; 6];
    for (row, &(i, j)) in PAIRS.iter().enumerate() {
        a2[row][row] += a[i][i] + a[j][j];
        for k in 0..4 {
            if k != i && k != j {
                let (col1, sgn1) = PAIR_LUT[k][j];
                a2[row][col1] += sgn1 * a[i][k];
                let (col2, sgn2) = PAIR_LUT[i][k];
                a2[row][col2] += sgn2 * a[j][k];
            }
        }
    }
    a2
}

#[must_use]
pub(crate) fn outer_bound_rows_l(
    surf: &BvpPoint,
    l: usize,
    om2: f64,
    outer_bc: OuterBoundKind,
) -> [[f64; 4]; 2] {
    let lf = l as f64;
    let lam = lf * (lf + 1.0);
    let r1 = match outer_bc {
        OuterBoundKind::Vacuum => [1.0, -1.0, 0.0, 0.0],
        OuterBoundKind::AdiplsIsothermal => {
            let vg = surf.v_2 * surf.x * surf.x / surf.gamma_1;
            let a_iso = vg * (surf.gamma_1 - 1.0);
            let sig_x = surf.c_1 * om2;
            let eta = lam / sig_x.max(1.0e-30);
            let gam =
                (a_iso - vg + 4.0).powi(2) + 4.0 * (eta - vg) * (sig_x - a_iso);
            let c21 =
                (0.5 * (vg - a_iso + gam.max(0.0).sqrt()) - 2.0) / (vg - eta);
            let b3 = (lf + 1.0 - eta) / (vg + a_iso).max(1.0e-30);
            [c21, -1.0, -b3, 0.0]
        }
    };
    [r1, [0.0, 0.0, lf + 1.0, 1.0]]
}

/// Evaluates the shooting determinant $D(\omega)$ for degree `l` and frequency `omega`.
#[must_use]
pub fn shoot_determinant(
    model: &BvpModel,
    l: usize,
    omega: f64,
    outer_bc: OuterBoundKind,
) -> f64 {
    let om2 = omega * omega;
    if l == 0 {
        let mut y = if model.pts[0].x > 0.0 {
            [0.0, 1.0]
        } else {
            [1.0, (model.pts[0].c_1 * om2) / 3.0]
        };
        for (i, mid) in model.mids.iter().enumerate() {
            let dlnx = (model.pts[i + 1].x - model.pts[i].x) / mid.x;
            let vg = mid.v_2 * mid.x * mid.x / mid.gamma_1;
            let (a, u) = (mid.a_star, mid.u);
            let m = expm2([
                [(vg - 1.0) * dlnx, -vg * dlnx],
                [(mid.c_1 * om2 - a + u) * dlnx, (a - u + 3.0) * dlnx],
            ]);
            y = [
                m[0][0] * y[0] + m[0][1] * y[1],
                m[1][0] * y[0] + m[1][1] * y[1],
            ];
            let sc = y[0].abs().max(y[1].abs()).max(1.0e-300);
            y = [y[0] / sc, y[1] / sc];
        }
        let surf = model.pts.last().unwrap();
        match outer_bc {
            OuterBoundKind::Vacuum => y[0] - y[1],
            OuterBoundKind::AdiplsIsothermal => {
                let vg = surf.v_2 * surf.x * surf.x / surf.gamma_1;
                let a_iso = vg * (surf.gamma_1 - 1.0);
                let sig_x = surf.c_1 * om2;
                let gam =
                    (a_iso - vg + 4.0).powi(2) - 4.0 * vg * (sig_x - a_iso);
                let c21 = (0.5 * (vg - a_iso + gam.max(0.0).sqrt()) - 2.0)
                    / vg.max(1.0e-30);
                c21 * y[0] - y[1]
            }
        }
    } else {
        let lf = l as f64;
        let c1_0 = model.pts[0].c_1;
        let mut w = [
            -lf * lf,
            lf * lf,
            lf.powi(3),
            c1_0 * om2 * lf,
            c1_0 * om2 * lf * lf,
            0.0,
        ];
        let sc0 = w
            .iter()
            .fold(0.0_f64, |acc, x| acc.max(x.abs()))
            .max(1.0e-300);
        for v in &mut w {
            *v /= sc0;
        }
        for (i, mid) in model.mids.iter().enumerate() {
            let dlnx = (model.pts[i + 1].x - model.pts[i].x) / mid.x;
            let a2 = compound_matrix6(ad_matrix4(mid, l, om2, dlnx));
            w = step_compound6(&a2, w);
            let sc = w
                .iter()
                .fold(0.0_f64, |acc, x| acc.max(x.abs()))
                .max(1.0e-300);
            for v in &mut w {
                *v /= sc;
            }
        }
        let [r1, r2] =
            outer_bound_rows_l(model.pts.last().unwrap(), l, om2, outer_bc);
        const PAIRS: [(usize, usize); 6] =
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        PAIRS
            .iter()
            .enumerate()
            .map(|(idx, &(i, j))| (r1[i] * r2[j] - r1[j] * r2[i]) * w[idx])
            .sum()
    }
}
