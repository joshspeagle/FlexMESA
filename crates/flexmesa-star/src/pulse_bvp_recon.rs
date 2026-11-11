// SPDX-License-Identifier: LGPL-3.0-or-later
//! Forward-QR / backward-substitution eigenfunction reconstructor, variational
//! integral $\omega_{\mathrm{int}}$, normalized mode inertia $E_{\mathrm{norm}}$,
//! quasi-non-adiabatic work integral $\mathrm{d}W/\mathrm{d}x$, Eckart-Scuflaire-Osaki-Takata
//! phase-plane classifier, and `GYRE`/`FGONG` BVP model parsers (`gyre/src/mode/wave_m.fypp`,
//! `gyre/src/mode/mode_m.fypp`, `adipls/adipack.c/adipls/takata.d.f`, resolving Issue #4).

#![allow(missing_docs)]

use std::f64::consts::PI;

use flexmesa_const::{CLIGHT, CRAD, L_SUN, M_SUN, R_SUN, STANDARD_CGRAV};

use crate::pulse_bvp::{
    ad_matrix4, expm2, expm4, outer_bound_rows_l, BvpEigenmode, BvpModel,
    BvpPoint, InertiaNormKind, OuterBoundKind,
};

/// Reconstructs the 4-component GYRE eigenfunction $\mathbf{y}(x_k)$ at all grid points
/// and classifies $(n_p, n_g, n_{pg})$, $E_{\mathrm{norm}}$, $\omega_{\mathrm{int}}$, and non-adiabatic growth.
#[must_use]
pub fn reconstruct_and_analyze_mode(
    model: &BvpModel,
    l: usize,
    omega: f64,
    outer_bc: OuterBoundKind,
    inertia_norm: InertiaNormKind,
) -> BvpEigenmode {
    let n = model.pts.len();
    let om2 = omega * omega;
    let mut y_grid = vec![[0.0; 4]; n];

    if l == 0 {
        let mut y = if model.pts[0].x > 0.0 {
            [0.0, 1.0]
        } else {
            [1.0, (model.pts[0].c_1 * om2) / 3.0]
        };
        y_grid[0] = [y[0], y[1], 0.0, -model.pts[0].u * y[0]];
        let mut scales = vec![1.0; n - 1];
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
            scales[i] = sc;
            y_grid[i + 1] = [y[0], y[1], 0.0, -model.pts[i + 1].u * y[0]];
        }
        let mut cum = 1.0;
        for i in (0..(n - 1)).rev() {
            cum /= scales[i];
            for c in &mut y_grid[i] {
                *c *= cum;
            }
        }
    } else {
        let lf = l as f64;
        let c1_0 = model.pts[0].c_1;
        let mut q0 = [lf, c1_0 * om2, 0.0, 0.0];
        let mut q1 = [0.0, -lf, lf, lf * lf];
        let n0 = q0.iter().map(|v| v * v).sum::<f64>().sqrt();
        for v in &mut q0 {
            *v /= n0;
        }
        let dot01: f64 = (0..4).map(|r| q0[r] * q1[r]).sum();
        for r in 0..4 {
            q1[r] -= dot01 * q0[r];
        }
        let n1 = q1.iter().map(|v| v * v).sum::<f64>().sqrt();
        for v in &mut q1 {
            *v /= n1;
        }
        let mut q_hist = Vec::with_capacity(n);
        let mut r_hist = Vec::with_capacity(n - 1);
        q_hist.push([q0, q1]);
        for (i, mid) in model.mids.iter().enumerate() {
            let dlnx = (model.pts[i + 1].x - model.pts[i].x) / mid.x;
            let m = expm4(ad_matrix4(mid, l, om2, dlnx));
            let [p0, p1] = q_hist[i];
            let mut z0 = [0.0; 4];
            let mut z1 = [0.0; 4];
            for r in 0..4 {
                for k in 0..4 {
                    z0[r] += m[r][k] * p0[k];
                    z1[r] += m[r][k] * p1[k];
                }
            }
            let r00 =
                z0.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0e-300);
            let u0 = [z0[0] / r00, z0[1] / r00, z0[2] / r00, z0[3] / r00];
            let r01: f64 = (0..4).map(|r| u0[r] * z1[r]).sum();
            let mut u1 = [
                z1[0] - r01 * u0[0],
                z1[1] - r01 * u0[1],
                z1[2] - r01 * u0[2],
                z1[3] - r01 * u0[3],
            ];
            let r11 =
                u1.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0e-300);
            for v in &mut u1 {
                *v /= r11;
            }
            q_hist.push([u0, u1]);
            r_hist.push([[r00, r01], [0.0, r11]]);
        }
        let [b1, b2] =
            outer_bound_rows_l(model.pts.last().unwrap(), l, om2, outer_bc);
        let [qs0, qs1] = q_hist[n - 1];
        let m00: f64 = (0..4).map(|r| b1[r] * qs0[r]).sum();
        let m01: f64 = (0..4).map(|r| b1[r] * qs1[r]).sum();
        let m10: f64 = (0..4).map(|r| b2[r] * qs0[r]).sum();
        let m11: f64 = (0..4).map(|r| b2[r] * qs1[r]).sum();
        let mut c = if m00.abs() + m01.abs() >= m10.abs() + m11.abs() {
            [-m01, m00]
        } else {
            [-m11, m10]
        };
        let cn = (c[0] * c[0] + c[1] * c[1]).sqrt().max(1.0e-300);
        c = [c[0] / cn, c[1] / cn];
        for r in 0..4 {
            y_grid[n - 1][r] = qs0[r] * c[0] + qs1[r] * c[1];
        }
        for i in (0..(n - 1)).rev() {
            let rmat = r_hist[i];
            let c1_new = c[1] / rmat[1][1];
            let c0_new = (c[0] - rmat[0][1] * c1_new) / rmat[0][0];
            c = [c0_new, c1_new];
            let [qi0, qi1] = q_hist[i];
            for r in 0..4 {
                y_grid[i][r] = qi0[r] * c[0] + qi1[r] * c[1];
            }
        }
    }
    let sgn = if y_grid[n - 1][0] >= 0.0 { 1.0 } else { -1.0 };
    for y in &mut y_grid {
        for v in y {
            *v *= sgn;
        }
    }

    let (n_p, n_g, n_pg) = classify_mode_orders(&model.pts, &y_grid, l);
    let lf = l as f64;
    let lam = lf * (lf + 1.0);
    let mut de_dx = vec![0.0; n];
    let mut dzeta_dx = vec![0.0; n];
    let mut dw_dx = vec![0.0; n];
    let tdyn_tkh = (model.l_star * model.r_star.powf(2.5))
        / (model.cgrav * model.m_star).powf(2.5).max(1.0e-30);

    for (i, pt) in model.pts.iter().enumerate() {
        let [y1, y2, y3, y4] = y_grid[i];
        let x_li1 = if l == 0 {
            if pt.x == 0.0 {
                0.0
            } else {
                1.0 / pt.x
            }
        } else if l == 1 {
            1.0
        } else {
            pt.x.powi((l as i32) - 1)
        };
        let xi_r = x_li1 * y1;
        let xi_h = if l == 0 {
            0.0
        } else {
            x_li1 * (y2 + y3) / (pt.c_1 * om2)
        };
        de_dx[i] =
            (xi_r * xi_r + lam * xi_h * xi_h) * pt.u * pt.x * pt.x / pt.c_1;
        let x_2li = if l == 0 {
            1.0
        } else {
            pt.x.powi(2 * (l as i32))
        };
        let term_g = y4 + (lf + 1.0) * y3;
        dzeta_dx[i] = x_2li
            * (y2 * y2 * pt.u * pt.v_2 * pt.x * pt.x / pt.gamma_1
                + y1 * y1 * pt.u * pt.a_star
                - term_g * term_g)
            / (pt.c_1 * pt.c_1);

        let v = pt.v_2 * pt.x * pt.x;
        let lag_p = pt.v_2
            * (y2 - y1)
            * (if l == 0 { 1.0 } else { pt.x.powi(l as i32) });
        let lag_t_ad = pt.nabla_ad * lag_p;
        let c_kap_ad = pt.nabla_ad * pt.kap_t + pt.kap_rho / pt.gamma_1;
        let dl_r_rel = -c_kap_ad * lag_p + 4.0 * lag_t_ad;
        let phase_lag =
            1.0 / (1.0 + (omega * pt.c_thk / pt.c_rad.max(1.0e-30)).powi(2));
        let im_lag_s =
            -phase_lag * v * pt.nabla * dl_r_rel / omega.max(1.0e-12);
        dw_dx[i] = PI * lag_t_ad * im_lag_s * pt.c_thk * pt.x * pt.x * tdyn_tkh;
    }

    let trapz = |f: &[f64]| -> f64 {
        (0..(n - 1))
            .map(|i| {
                0.5 * (model.pts[i + 1].x - model.pts[i].x) * (f[i] + f[i + 1])
            })
            .sum()
    };
    let e_tot = trapz(&de_dx).max(1.0e-99);
    let zeta = trapz(&dzeta_dx);
    let surf = &model.pts[n - 1];
    let [ys1, ys2, ys3, ys4] = y_grid[n - 1];
    let s_surf = match outer_bc {
        OuterBoundKind::Vacuum => 0.0,
        OuterBoundKind::AdiplsIsothermal => {
            surf.x.powi(2 * (l as i32) + 1)
                * (surf.u * (ys2 + ys3) * ys1 + ys3 * (ys4 + (lf + 1.0) * ys3))
                / (surf.c_1 * surf.c_1)
        }
    };
    let omega_int = ((zeta + s_surf) / e_tot).max(0.0).sqrt();
    let x_li1_ref = if l == 0 {
        1.0 / surf.x
    } else if l == 1 {
        1.0
    } else {
        surf.x.powi((l as i32) - 1)
    };
    let xi_r_ref = x_li1_ref * ys1;
    let xi_h_ref = if l == 0 {
        0.0
    } else {
        x_li1_ref * (ys2 + ys3) / (surf.c_1 * om2)
    };
    let a2_ref = match inertia_norm {
        InertiaNormKind::Both => {
            xi_r_ref * xi_r_ref + lam * xi_h_ref * xi_h_ref
        }
        InertiaNormKind::Radial => xi_r_ref * xi_r_ref,
    }
    .max(1.0e-99);
    let e_norm = e_tot / a2_ref;
    let e_adipls = e_norm / (4.0 * PI);
    let w_tot = trapz(&dw_dx);
    let w_abs: Vec<f64> = dw_dx.iter().map(|v| v.abs()).collect();
    let w_denom = trapz(&w_abs);
    let stellingwerf_eta = if w_denom > 1.0e-99 {
        (w_tot / w_denom).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let h_energy = 0.5 * om2 * e_tot;
    let omega_imag = if h_energy > 1.0e-99 {
        0.25 * omega * (w_tot / (2.0 * PI * h_energy))
    } else {
        0.0
    };
    let omega_dyn = (model.cgrav * model.m_star / model.r_star.powi(3)).sqrt();
    let freq_scale_uhz = 1.0e6 * omega_dyn / (2.0 * PI);

    BvpEigenmode {
        l,
        n_p,
        n_g,
        n_pg,
        omega,
        omega_int,
        freq_uhz: omega * freq_scale_uhz,
        freq_var_uhz: omega_int * freq_scale_uhz,
        e_norm,
        e_adipls,
        omega_imag,
        growth_rate_s: omega_imag * omega_dyn,
        stellingwerf_eta,
        y: y_grid,
    }
}

/// Classifies radial order $(n_p, n_g, n_{pg})$ via the Eckart-Scuflaire-Osaki-Takata
/// phase-plane winding scheme (`gyre/src/mode/mode_m.fypp`).
#[must_use]
pub fn classify_mode_orders(
    pts: &[BvpPoint],
    y: &[[f64; 4]],
    l: usize,
) -> (i32, i32, i32) {
    let n = pts.len();
    let mut y1 = vec![0.0; n];
    let mut y2 = vec![0.0; n];
    for i in 0..n {
        if l == 0 {
            y1[i] = y[i][0];
            y2[i] = y[i][1];
        } else if l == 1 {
            y1[i] =
                (1.0 - pts[i].u / 3.0) * y[i][0] + (y[i][2] - y[i][3]) / 3.0;
            y2[i] = y[i][1] - y[i][0];
        } else {
            let scale = pts[i].x.powi(l as i32 - 1);
            y1[i] = scale * y[i][0];
            y2[i] = scale * (y[i][1] + y[i][2]);
        }
    }
    let mut j_i = 0;
    for j in 1..(n.saturating_sub(2)) {
        let (dy_a, dy_b) = (y1[j] - y1[j - 1], y1[j + 1] - y1[j]);
        if (y1[j] >= 0.0 && dy_a >= 0.0 && dy_b >= 0.0)
            || (y1[j] <= 0.0 && dy_a <= 0.0 && dy_b <= 0.0)
        {
            j_i = j;
            break;
        }
    }
    let j_o = if y1[n - 1] == 0.0 { n - 2 } else { n - 1 };
    let (mut n_a, mut n_c) = (0, 0);
    for j in j_i..j_o {
        let (a, b) = (y1[j], y1[j + 1]);
        if (a >= 0.0 && b < 0.0) || (a <= 0.0 && b > 0.0) {
            let y2_cross = y2[j] - a * (y2[j + 1] - y2[j]) / (b - a);
            if a >= 0.0 && b < 0.0 {
                if y2_cross >= 0.0 {
                    n_a += 1;
                } else {
                    n_c += 1;
                }
            } else if y2_cross <= 0.0 {
                n_a += 1;
            } else {
                n_c += 1;
            }
        }
    }
    let (n_p, n_g) = if l == 0 {
        (n_a + n_c + 1, 0)
    } else {
        (n_a, n_c)
    };
    let n_pg = if l == 1 {
        if n_p >= n_g {
            n_p - n_g + 1
        } else {
            n_p - n_g
        }
    } else {
        n_p - n_g
    };
    (n_p, n_g, n_pg)
}

/// Parses a Fortran floating-point token (handling `D`/`d` and 3-digit exponents
/// without `E` such as `0.100000000000+100`).
#[must_use]
pub fn parse_fortran_float(tok: &str) -> f64 {
    let s = tok.trim().replace(['D', 'd'], "E");
    if let Ok(v) = s.parse::<f64>() {
        return v;
    }
    let bytes = s.as_bytes();
    for i in (1..bytes.len()).rev() {
        if (bytes[i] == b'+' || bytes[i] == b'-')
            && bytes[i - 1] != b'E'
            && bytes[i - 1] != b'e'
        {
            let mut fixed = String::with_capacity(s.len() + 1);
            fixed.push_str(&s[..i]);
            fixed.push('E');
            fixed.push_str(&s[i..]);
            return fixed.parse::<f64>().unwrap_or(0.0);
        }
    }
    0.0
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn make_bvp_point_from_cgs(
    r: f64,
    m_r: f64,
    p: f64,
    t: f64,
    rho: f64,
    nabla: f64,
    a_star: f64,
    gamma_1: f64,
    nabla_ad: f64,
    ups_t: f64,
    cp: f64,
    kap: f64,
    kap_t: f64,
    kap_rho: f64,
    eps: f64,
    eps_t: f64,
    eps_rho: f64,
    m_star: f64,
    r_star: f64,
    l_star: f64,
    cgrav: f64,
    omega_dyn: f64,
) -> BvpPoint {
    let x = r / r_star;
    let (v_2, u, c_1) = if x > 0.0 && m_r > 0.0 {
        (
            cgrav * m_r * rho / (p * r * x * x),
            4.0 * PI * rho * r.powi(3) / m_r,
            x.powi(3) / (m_r / m_star),
        )
    } else {
        (
            4.0 * PI * cgrav * rho * rho * r_star * r_star / (3.0 * p),
            3.0,
            3.0 * (m_star / r_star.powi(3)) / (4.0 * PI * rho),
        )
    };
    let c_rad = 16.0 * PI * CRAD * CLIGHT * t.powi(4) * r_star * nabla * v_2
        / (3.0 * kap * rho * l_star.max(1.0e-30)).max(1.0e-30);
    let c_thn = cp * omega_dyn / (CRAD * CLIGHT * kap * t.powi(3)).max(1.0e-30);
    let c_thk = 4.0 * PI * rho * t * cp * omega_dyn * r_star.powi(3)
        / l_star.max(1.0e-30);
    let c_eps = 4.0 * PI * rho * eps * r_star.powi(3) / l_star.max(1.0e-30);
    BvpPoint {
        x,
        v_2,
        a_star: if x > 0.0 && m_r > 0.0 { a_star } else { 0.0 },
        u,
        c_1,
        gamma_1,
        nabla_ad,
        ups_t,
        nabla,
        c_rad,
        c_thn,
        c_thk,
        c_eps,
        eps_rho,
        eps_t,
        kap_rho,
        kap_t,
    }
}

/// Builds a [`BvpModel`] from a `GYRE` MESA-format ASCII string (`v0.01`..`v1.20`).
#[must_use]
pub fn parse_gyre_mesa_model(content: &str) -> BvpModel {
    let mut lines = content.lines().filter(|l| !l.trim().is_empty());
    let h: Vec<&str> = lines.next().unwrap_or("").split_whitespace().collect();
    let m_star = h.get(1).map_or(M_SUN, |s| parse_fortran_float(s));
    let r_star = h.get(2).map_or(R_SUN, |s| parse_fortran_float(s));
    let l_star = h.get(3).map_or(L_SUN, |s| parse_fortran_float(s));
    let version = h.get(4).and_then(|s| s.parse::<usize>().ok()).unwrap_or(1);
    let cgrav = STANDARD_CGRAV;
    let omega_dyn = (cgrav * m_star / r_star.powi(3)).sqrt();

    let mut raw_pts = Vec::new();
    for line in lines {
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < 18 {
            continue;
        }
        let c = |idx: usize| parse_fortran_float(toks[idx]);
        let r = c(1);
        let (mr, pr, tr, rhor, nab, n2r) = (c(2), c(4), c(5), c(6), c(7), c(8));
        let (m_r, gam1, nad, ut, cp, kr, kt, krho, ep, et, erho) = if version
            == 1
        {
            let (cv, cp, ct, cr) =
                (c(9).max(1.0e-30), c(10), c(11), c(12).max(1.0e-30));
            let gam1 = cr * cp / cv;
            let ut = ct / cr;
            let nad = pr * ut / (rhor * tr * cp).max(1.0e-30);
            let (kr, kt, krho, ep) = (c(13).max(1.0e-30), c(14), c(15), c(16));
            let et = if ep.abs() > 1.0e-30 { c(17) / ep } else { 0.0 };
            let er = if ep.abs() > 1.0e-30 { c(18) / ep } else { 0.0 };
            (
                mr / (1.0 + mr) * m_star,
                gam1,
                nad,
                ut,
                cp,
                kr,
                kt,
                krho,
                ep,
                et,
                er,
            )
        } else {
            let (gam1, nad, ut, kr) = (c(9), c(10), c(11), c(12).max(1.0e-30));
            let cp = pr * ut / (rhor * tr * nad.max(1.0e-30)).max(1.0e-30);
            let (kt, krho, ep) = (c(13) / kr, c(14) / kr, c(15));
            let et = if ep.abs() > 1.0e-30 { c(16) / ep } else { 0.0 };
            let er = if ep.abs() > 1.0e-30 { c(17) / ep } else { 0.0 };
            (mr, gam1, nad, ut, cp, kr, kt, krho, ep, et, er)
        };
        let a_star = if r > 0.0 && m_r > 0.0 {
            r.powi(3) * n2r / (cgrav * m_r)
        } else {
            0.0
        };
        raw_pts.push(make_bvp_point_from_cgs(
            r, m_r, pr, tr, rhor, nab, a_star, gam1, nad, ut, cp, kr, kt, krho,
            ep, et, erho, m_star, r_star, l_star, cgrav, omega_dyn,
        ));
    }
    BvpModel::from_points(m_star, r_star, l_star, cgrav, raw_pts)
}

/// Builds a [`BvpModel`] from an `FGONG` ASCII string (`v300` or `v1300`).
#[must_use]
pub fn parse_fgong_model(content: &str) -> BvpModel {
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() < 6 {
        return BvpModel::from_points(
            M_SUN,
            R_SUN,
            L_SUN,
            STANDARD_CGRAV,
            Vec::new(),
        );
    }
    let counts: Vec<usize> = lines[4]
        .split_whitespace()
        .filter_map(|s| s.parse::<usize>().ok())
        .collect();
    let (nn, iconst, ivar) = (
        counts.first().copied().unwrap_or(0),
        counts.get(1).copied().unwrap_or(15),
        counts.get(2).copied().unwrap_or(40),
    );
    let mut nums = Vec::with_capacity(iconst + nn * ivar);
    for line in &lines[5..] {
        for tok in line.split_whitespace() {
            let b = tok.as_bytes();
            let mut start = 0;
            for i in 1..b.len() {
                if (b[i] == b'+' || b[i] == b'-')
                    && !matches!(b[i - 1], b'E' | b'e' | b'D' | b'd')
                    && tok[i..].contains('.')
                {
                    nums.push(parse_fortran_float(&tok[start..i]));
                    start = i;
                }
            }
            nums.push(parse_fortran_float(&tok[start..]));
        }
    }
    let m_star = nums.first().copied().unwrap_or(M_SUN);
    let r_star = nums.get(1).copied().unwrap_or(R_SUN);
    let l_star = nums.get(2).copied().unwrap_or(L_SUN);
    let cg = nums.get(14).copied().unwrap_or(0.0);
    let cgrav = if cg > 1.0e-10 { cg } else { 6.67232e-8 };
    let omega_dyn = (cgrav * m_star / r_star.powi(3)).sqrt();

    let mut pts_rev = Vec::with_capacity(nn);
    for i in (0..nn).rev() {
        let b = iconst + i * ivar;
        if b + 15 > nums.len() {
            continue;
        }
        let (r, m_r, t, p, rho) = (
            nums[b],
            nums[b + 1].exp() * m_star,
            nums[b + 2].max(100.0),
            nums[b + 3].max(1.0e-30),
            nums[b + 4].max(1.0e-30),
        );
        let (kap, eps, gam1, nad, ut, cp, a_star) = (
            nums[b + 7].max(1.0e-30),
            nums[b + 8],
            nums[b + 9].max(1.0),
            nums[b + 10].max(0.01),
            nums[b + 11].max(0.01),
            nums[b + 12].max(1.0),
            nums[b + 14],
        );
        pts_rev.push(make_bvp_point_from_cgs(
            r, m_r, p, t, rho, nad, a_star, gam1, nad, ut, cp, kap, 0.0, 0.0,
            eps, 0.0, 0.0, m_star, r_star, l_star, cgrav, omega_dyn,
        ));
    }
    if pts_rev.len() > 2 && pts_rev.last().unwrap().u <= 10.0 {
        pts_rev.pop();
    }
    BvpModel::from_points(m_star, r_star, l_star, cgrav, pts_rev)
}
