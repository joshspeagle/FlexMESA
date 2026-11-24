// SPDX-License-Identifier: GPL-3.0-or-later
//! Safe Rust ports of MESA 26.4.1 `adipls/adipack.c/gensr/` numerical helpers
//! (`#171`, Package A2a).
//!
//! Covers the reached general-purpose array, linear-system, differentiation,
//! quadrature, interpolation, windowing, and 4th-order Cash-Moore ODE
//! integration routines:
//!
//! - [`zero`] (`adipls/adipack.c/gensr/zero.d.f`)
//! - [`store`] (`adipls/adipack.c/gensr/store.d.f`)
//! - [`notwin`] (`adipls/adipack.c/gensr/notwin.d.f`) and [`notwni`]
//!   (`adipls/adipack.c/gensr/notwni.f`)
//! - [`leq`], [`leqdet`], and [`leqsd`] (`adipls/adipack.c/gensr/leq.d.f`,
//!   `adipls/adipack.c/adipls/setups_adi.c.d.f:78`)
//! - [`lininh4`] (`adipls/adipack.c/gensr/lininh4.d.f`)
//! - [`aramax`] (`adipls/adipack.c/gensr/aramax.d.f`)
//! - [`derive()`] and [`deriv2`] (`adipls/adipack.c/gensr/derive.d.f`)
//! - [`derivk`] (`adipls/adipack.c/gensr/derivk.new.d.f`)
//! - [`vinta`] (`adipls/adipack.c/gensr/vinta.d.f`)
//! - [`vintk`] (`adipls/adipack.c/gensr/vintk.new.d.f`)
//! - [`lir`] and [`lir1`] (`adipls/adipack.c/gensr/lir.d.f`)

use crate::error::AdiplsError;

/// Zero the first `nn` elements of `a`, matching `zero` (`adipls/adipack.c/gensr/zero.d.f:1-23`).
pub fn zero(a: &mut [f64], nn: usize) {
    for x in a.iter_mut().take(nn) {
        *x = 0.0_f64;
    }
}

/// Copy the first `n` elements from `a` into `b`, matching `store`
/// (`adipls/adipack.c/gensr/store.d.f:1-21`).
pub fn store(a: &[f64], b: &mut [f64], n: usize) {
    b[..n].copy_from_slice(&a[..n]);
}

/// Floating-point window rejection predicate matching `notwin`
/// (`adipls/adipack.c/gensr/notwin.d.f:1-21`).
///
/// Returns `true` when `xw1 <= xw2` and `x` lies outside `[xw1, xw2]`.
#[must_use]
#[inline]
pub fn notwin(xw1: f64, xw2: f64, x: f64) -> bool {
    if xw1 > xw2 {
        false
    } else {
        x < xw1 || x > xw2
    }
}

/// Integer window rejection predicate matching `notwni`
/// (`adipls/adipack.c/gensr/notwni.f:1-12`).
///
/// Returns `true` when `kw1 <= kw2` and `k` lies outside `[kw1, kw2]`.
#[must_use]
#[inline]
pub const fn notwni(kw1: i32, kw2: i32, k: i32) -> bool {
    if kw1 > kw2 {
        false
    } else {
        k < kw1 || k > kw2
    }
}

/// Solve `A x = B` by Gaussian elimination with row equilibration and partial
/// column pivoting, returning `det(A)`, matching `leq` (`adipls/adipack.c/gensr/leq.d.f:1-223`).
///
/// Arrays `a` and `b` use Fortran column-major storage with leading dimensions
/// `ia >= nn` and `ib >= nn`. When `mm == 0`, only `det(A)` is computed and `b`
/// is untouched.
#[allow(clippy::too_many_lines, clippy::many_single_char_names)]
pub fn leq(
    a: &mut [f64],
    b: &mut [f64],
    nn: usize,
    mm: usize,
    ia: usize,
    ib: usize,
) -> f64 {
    let n = nn;
    let m = mm;
    let mut detsc = 1.0_f64;

    if n == 0 {
        return 0.0_f64;
    }
    if n == 1 {
        let err = a[0];
        if m == 0 {
            return err;
        }
        let ai = 1.0_f64 / err;
        for col in 0..m {
            b[col * ib] *= ai;
        }
        return err;
    }

    // Row equilibration (`leq.d.f:62-80`).
    for i in 0..n {
        let mut r = a[i].abs();
        for col in 0..n {
            r = r.max(a[col * ia + i].abs());
        }
        if r == 0.0_f64 {
            return 0.0_f64;
        }
        for col in 0..n {
            a[col * ia + i] /= r;
        }
        if m > 0 {
            for col in 0..m {
                b[col * ib + i] /= r;
            }
        }
        detsc *= r;
    }

    // Partial pivoting and forward elimination (`leq.d.f:84-171`).
    let n2 = n - 1;
    for i in 0..n2 {
        let ialow = i * ia + i;
        let iaup = i * ia + (n - 1);
        let mut r = a[ialow].abs();
        let mut imax = ialow;
        for (j, &val) in a.iter().enumerate().take(iaup + 1).skip(ialow + 1) {
            let aj = val.abs();
            if r - aj < 0.0_f64 {
                imax = j;
                r = aj;
            }
        }
        if imax > ialow {
            let im = imax - i * ia;
            for col in 0..n {
                let jj = col * ia + i;
                let ji = col * ia + im;
                a.swap(jj, ji);
            }
            detsc = -detsc;
            if m > 0 {
                for col in 0..m {
                    let jj = col * ib + i;
                    let ji = col * ib + im;
                    b.swap(jj, ji);
                }
            }
        }

        // Elimination below the diagonal (`leq.d.f:125-168`).
        for j in ialow..iaup {
            if a[ialow] == 0.0_f64 {
                // In `leq.d.f:126-151`, `joy + 1 - n < 0` exits immediately
                // for `i < n - 1`, and for `i == n - 1` `a[ialow+1] == 0`
                // also exits immediately.
                return 0.0_f64;
            }
            let j1 = j + 1;
            let row_target = j1 - i * ia;
            let factor = -a[j1] / a[ialow];
            for col in i..n {
                let k_idx = col * ia + row_target;
                let i1_idx = col * ia + i;
                a[k_idx] += a[i1_idx] * factor;
            }
            if m > 0 {
                for col in 0..m {
                    let i2_idx = col * ib + row_target;
                    let i1_idx = col * ib + i;
                    b[i2_idx] += b[i1_idx] * factor;
                }
            }
        }
    }

    // Determinant product (`leq.d.f:175-186`).
    let mut err = 1.0_f64;
    for i in 0..n {
        err *= a[i * ia + i];
        if err == 0.0_f64 {
            return 0.0_f64;
        }
    }
    err *= detsc;

    if m == 0 {
        return err;
    }

    // Back-substitution (`leq.d.f:190-206`), preserving reverse inner loop order.
    let ka = (n - 1) * ia + (n - 1);
    for k in 0..m {
        let kb = k * ib + (n - 1);
        b[kb] /= a[ka];
        for l in 1..=n2 {
            let i = n - 1 - l;
            let mut sum = 0.0_f64;
            for jj in ((i + 1)..n).rev() {
                let ja = jj * ia + i;
                let jb = k * ib + jj;
                sum += a[ja] * b[jb];
            }
            let la = i * ia + i;
            let lb = k * ib + i;
            b[lb] = (b[lb] - sum) / a[la];
        }
    }

    err
}

/// Compatibility wrapper matching `leqdet` (`adipls/adipack.c/adipls/setups_adi.c.d.f:78-90`).
pub fn leqdet(
    a: &mut [f64],
    b: &mut [f64],
    nn: usize,
    mm: usize,
    ia: usize,
    ib: usize,
) -> f64 {
    leq(a, b, nn, mm, ia, ib)
}

/// Compatibility wrapper matching `leqsd` (`adipls/adipack.c/gensr/leq.d.f:224-230`).
pub fn leqsd(
    a: &mut [f64],
    b: &mut [f64],
    n: usize,
    m: usize,
    ia: usize,
    ib: usize,
) -> f64 {
    leq(a, b, n, m, ia, ib)
}

/// Integrate a linear ODE system `dy/dx = A(x) y + f(x)` using the 4th-order
/// Cash & Moore (1980) implicit scheme, matching `lininh4`
/// (`adipls/adipack.c/gensr/lininh4.d.f:1-259`).
///
/// The solver steps `nn` points (`1..=nn`) starting from `x[x_start]` and
/// `y[:, y_start]` in column-major layout (leading dimension `iy`), stepping
/// both `x`, `y` columns, and the 1-based RHS point index `nd` by `isn`.
///
/// # Errors
/// Propagates any error returned by the `rhs` callback.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::manual_midpoint
)]
pub fn lininh4<F>(
    x: &[f64],
    y: &mut [f64],
    iy: usize,
    ii: usize,
    ig: usize,
    isn: isize,
    nd1: usize,
    nn: usize,
    x_start: usize,
    y_start: usize,
    mut rhs: F,
) -> Result<(), AdiplsError>
where
    F: FnMut(
        f64,
        &mut [[f64; 4]; 4],
        &mut [f64; 8],
        usize,
    ) -> Result<(), AdiplsError>,
{
    if nn <= 1 {
        return Ok(());
    }

    let mut fd = [[0.0_f64; 4]; 4];
    let mut fdp = [[0.0_f64; 4]; 4];
    let mut fdh = [[0.0_f64; 4]; 4];
    let mut finh = [0.0_f64; 8];
    let mut fp = [0.0_f64; 4];
    let mut w = [[0.0_f64; 4]; 4];

    let mut n_off = 0_isize;
    let mut nc = 1_usize;
    let mut nd = nd1;
    let mut x2 = x[x_start];
    rhs(x2, &mut fd, &mut finh, nd)?;

    loop {
        for i in 0..ii {
            for j in 0..ii {
                fdp[i][j] = fd[i][j];
            }
        }

        let n1_off = n_off;
        n_off += isn;
        nc += 1;
        let x1 = x2;
        let x_idx = usize::try_from(
            isize::try_from(x_start).map_err(|_| AdiplsError::Solver {
                routine: "lininh4",
                message: String::from("x_start index overflow"),
            })? + n_off,
        )
        .map_err(|_| AdiplsError::Solver {
            routine: "lininh4",
            message: String::from("negative x index in lininh4"),
        })?;
        x2 = x[x_idx];
        nd = usize::try_from(
            isize::try_from(nd).map_err(|_| AdiplsError::Solver {
                routine: "lininh4",
                message: String::from("nd index overflow"),
            })? + isn,
        )
        .map_err(|_| AdiplsError::Solver {
            routine: "lininh4",
            message: String::from("negative nd index in lininh4"),
        })?;

        rhs(x2, &mut fd, &mut finh, nd)?;

        for i in 0..ii {
            for j in 0..ii {
                fdh[i][j] = 0.5_f64 * (fd[i][j] + fdp[i][j]);
            }
        }
        let dx = x2 - x1;

        let y_n1_col = usize::try_from(
            isize::try_from(y_start).map_err(|_| AdiplsError::Solver {
                routine: "lininh4",
                message: String::from("y_start index overflow"),
            })? + n1_off,
        )
        .map_err(|_| AdiplsError::Solver {
            routine: "lininh4",
            message: String::from("negative y_n1 column index"),
        })?;
        let y_n_col = usize::try_from(
            isize::try_from(y_start).map_err(|_| AdiplsError::Solver {
                routine: "lininh4",
                message: String::from("y_start index overflow"),
            })? + n_off,
        )
        .map_err(|_| AdiplsError::Solver {
            routine: "lininh4",
            message: String::from("negative y_n column index"),
        })?;

        let y_n1_base = y_n1_col * iy;
        let y_n_base = y_n_col * iy;

        let mut kg = 0_usize;
        for _l in 0..ig {
            for i in 0..ii {
                let mut sum = 0.0_f64;
                for j in 0..ii {
                    sum += fdp[i][j] * y[y_n1_base + kg + j];
                }
                fp[i] = sum;
            }
            for i in 0..ii {
                let mut sum1 = 0.0_f64;
                let mut sum2 = 0.0_f64;
                for j in 0..ii {
                    sum1 += fdh[i][j] * y[y_n1_base + kg + j];
                    sum2 += fdh[i][j] * fp[j];
                }
                y[y_n_base + kg + i] = ((y[y_n1_base + kg + i]
                    + (dx * fp[i]) / 6.0_f64)
                    + (dx * sum1) / 3.0_f64)
                    + ((dx * dx) * sum2) / 12.0_f64;
            }
            kg += ii;
        }

        // Coefficient matrix (`lininh4.d.f:161-170`).
        for i in 0..ii {
            for j in 0..ii {
                let mut sum = 0.0_f64;
                for k in 0..ii {
                    sum += fdh[i][k] * fd[k][j];
                }
                w[i][j] = ((-(fd[i][j] + 2.0_f64 * fdh[i][j])) * dx) / 6.0_f64
                    + ((sum * dx) * dx) / 12.0_f64;
            }
            w[i][i] += 1.0_f64;
        }

        // Gaussian elimination without pivoting (`lininh4.d.f:182-252`).
        if ii > 1 {
            let ii1 = ii - 1;
            for i in 0..ii1 {
                let r = w[i][i];
                if r == 0.0_f64 {
                    return Ok(());
                }
                let r = -1.0_f64 / r;
                let i1 = i + 1;
                for j in i1..ii {
                    let rj = r * w[j][i];
                    for k in i1..ii {
                        w[j][k] += rj * w[i][k];
                    }
                    let mut js = j;
                    let mut is_idx = i;
                    for _l in 0..ig {
                        y[y_n_base + js] += rj * y[y_n_base + is_idx];
                        js += ii;
                        is_idx += ii;
                    }
                }
            }
        }

        for i in (0..ii).rev() {
            let r = w[i][i];
            if r == 0.0_f64 {
                return Ok(());
            }
            let r = 1.0_f64 / r;
            let mut is_idx = i;
            let i1 = i + 1;
            for _l in 0..ig {
                let mut sum = y[y_n_base + is_idx];
                if i1 < ii {
                    let mut j1 = is_idx;
                    for &wij in w[i].iter().take(ii).skip(i1) {
                        j1 += 1;
                        sum -= wij * y[y_n_base + j1];
                    }
                }
                y[y_n_base + is_idx] = r * sum;
                is_idx += ii;
            }
        }

        if nc == nn {
            return Ok(());
        }
    }
}

/// Find the maximum of `|a(1, n)|` and its parabolic-fit location `xext`,
/// matching `aramax` (`adipls/adipack.c/gensr/aramax.d.f:1-96`).
///
/// Returns `(xext, aext, d2aext)`.
#[must_use]
pub fn aramax(x: &[f64], a: &[f64], ia: usize, nn: usize) -> (f64, f64, f64) {
    if nn == 0 {
        return (0.0_f64, 0.0_f64, 0.0_f64);
    }
    let mut axt = a[0].abs();
    let mut nxt = 0_usize;
    for n in 1..nn {
        let aa = a[n * ia].abs();
        if axt - aa <= 0.0_f64 {
            nxt = n;
            axt = aa;
        }
    }
    if nxt == 0 || nxt == nn - 1 {
        return (x[nxt], axt, 0.0_f64);
    }

    let xtr = x[nxt];
    let scl = x[nxt + 1] - x[nxt - 1];
    if scl == 0.0_f64 {
        return (xtr, axt, 0.0_f64);
    }

    let mut wrk = [0.0_f64; 9];
    let mut b = [0.0_f64; 3];
    for i in 0..3 {
        let mut t = 1.0_f64;
        wrk[i] = t;
        let i1 = nxt - 1 + i;
        let xx = (x[i1] - xtr) / scl;
        for k in 1..3 {
            t *= xx;
            wrk[k * 3 + i] = t;
        }
        b[i] = a[i1 * ia];
    }

    let det = leq(&mut wrk, &mut b, 3, 1, 3, 1);
    if det == 0.0_f64 || b[2] == 0.0_f64 {
        return (xtr, axt, 0.0_f64);
    }
    let xext_rel = -b[1] / (2.0_f64 * b[2]);
    let ampl_ext = (b[0] + xext_rel * (b[1] + xext_rel * b[2])).abs();
    let xext = xtr + scl * xext_rel;
    let d2aext = (2.0_f64 * b[2]) / (scl * scl);
    (xext, ampl_ext, d2aext)
}

/// First derivative `dydx(1, n) = dy(1, n)/dx(n)` with 2nd-order interior and
/// 3rd-order endpoint stencils, matching `derive` (`adipls/adipack.c/gensr/derive.d.f:1-234`).
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn derive(
    x: &[f64],
    y: &[f64],
    dydx: &mut [f64],
    nn: usize,
    iy: usize,
    idy: usize,
    ipy: usize,
    ipdy: usize,
) {
    derive_impl(x, y, dydx, nn, iy, idy, ipy, ipdy, false);
}

/// Second derivative `dydx(1, n) = d^2 y(1, n)/dx(n)^2`, matching `deriv2`
/// (`adipls/adipack.c/gensr/derive.d.f:51-234`).
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn deriv2(
    x: &[f64],
    y: &[f64],
    dydx: &mut [f64],
    nn: usize,
    iy: usize,
    idy: usize,
    ipy: usize,
    ipdy: usize,
) {
    derive_impl(x, y, dydx, nn, iy, idy, ipy, ipdy, true);
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::similar_names
)]
fn derive_impl(
    x: &[f64],
    y: &[f64],
    dydx: &mut [f64],
    nn: usize,
    iy: usize,
    idy: usize,
    ipy: usize,
    ipdy: usize,
    second: bool,
) {
    const EPSIL: f64 = 1.0e-8_f64;

    let ky = iy * ipy;
    let kdy = idy * ipdy;
    if ipdy != 1 {
        let mut j = 0_isize - isize::try_from(kdy).unwrap_or(0);
        for _ in 0..nn {
            j += isize::try_from(kdy).unwrap_or(0);
            if let Ok(idx) = usize::try_from(j) {
                dydx[idx] = 0.0_f64;
            }
        }
    }

    if nn < 4 {
        return;
    }

    let n1 = nn - 1;
    let mut i = 0_usize;
    let mut k = ky;
    let mut hn = x[1] - x[0];
    let mut xn = x[0].abs();
    let mut hna = hn.abs();
    let mut e = y[k] - y[0];

    let mut hn1 = 0.0_f64;
    let mut h3: f64;
    let mut xn1: f64;
    let mut hna1 = 0.0_f64;
    let mut ha3: f64;
    let mut hnin = 0.0_f64;

    if !second {
        hn1 = x[2] - x[1];
        h3 = x[3] - x[2];
        xn1 = x[3].abs();
        hna1 = hn1.abs();
        ha3 = h3.abs();
        if hn == 0.0_f64 {
            dydx[i] = 0.0_f64;
        } else {
            hnin = 1.0_f64 / hn;
            let hn1_s = hnin * hn1;
            let h3_s = hnin * h3;
            dydx[i] = endpoint_deriv1(
                hnin,
                hn1_s,
                h3_s,
                hna,
                hna1,
                ha3,
                xn,
                xn1,
                EPSIL,
                e,
                y[k + ky],
                y[0],
                y[k + 2 * ky],
                false,
            );
        }
    }

    // Interior points (`n = 2..=nn-1` in 1-based, `1..n1` in 0-based).
    for n in 1..n1 {
        i += kdy;
        let j = k;
        k += ky;
        xn1 = xn;
        xn = x[n].abs();
        let xxa = (xn1 + xn) * EPSIL;
        let d = e;
        e = y[k] - y[j];
        hn1 = hn;
        hn = x[n + 1] - x[n];
        hna1 = hna;
        hna = hn.abs();
        if hna < xxa || hna1 < xxa {
            dydx[i] = dydx[i - kdy];
            continue;
        }
        hnin = 1.0_f64 / hn;
        hn1 *= hnin;
        let mut c = hn1 * (1.0_f64 + hn1);
        let (a, b, dysc) = if second {
            c *= 0.5_f64;
            (hn1, -1.0_f64, hnin * hnin)
        } else {
            (hn1 * hn1, 1.0_f64, hnin)
        };
        dydx[i] = dysc * (a * e + b * d) / c;
    }

    h3 = x[nn - 3] - x[nn - 4];
    ha3 = h3.abs();
    h3 *= hnin;
    xn = x[nn - 1].abs();
    xn1 = x[nn - 4].abs();

    if !second {
        i += kdy;
        let j = k;
        let k_end = k - ky * 3;
        let l = k_end;
        e = -e;
        dydx[i] = endpoint_deriv1(
            hnin,
            hn1,
            h3,
            hna,
            hna1,
            ha3,
            xn,
            xn1,
            EPSIL,
            e,
            y[k_end + ky],
            y[j],
            y[l],
            true,
        );
        return;
    }

    // Second derivative at endpoints (`derive.d.f:190-230`).
    let mut j = i;
    let mut k2 = i - kdy;
    let mut l = k2 - kdy;
    i += kdy;
    loop {
        let a = 1.0_f64 + hn1;
        let b = a + h3;
        let c = hn1 + h3;
        let xxa = (xn1 + xn) * EPSIL;
        if hna >= xxa && hna1 >= xxa && ha3 >= xxa {
            dydx[i] = (a * b / (hn1 * c)) * dydx[j]
                - (b / (hn1 * h3)) * dydx[k2]
                + (a / (c * h3)) * dydx[l];
        } else {
            dydx[i] = 0.0_f64;
        }
        if i == 0 {
            return;
        }
        i = 0;
        j = i + kdy;
        k2 = j + kdy;
        l = k2 + kdy;
        hn = x[1] - x[0];
        hn1 = x[2] - x[1];
        h3 = x[3] - x[2];
        xn = x[0].abs();
        xn1 = x[3].abs();
        hna = hn.abs();
        hna1 = hn1.abs();
        ha3 = h3.abs();
        if hn == 0.0_f64 {
            dydx[i] = 0.0_f64;
            return;
        }
        hnin = 1.0_f64 / hn;
        hn1 *= hnin;
        h3 *= hnin;
    }
}

#[allow(
    clippy::too_many_arguments,
    clippy::many_single_char_names,
    clippy::similar_names
)]
fn endpoint_deriv1(
    hnin: f64,
    hn1: f64,
    h3: f64,
    hna: f64,
    hna1: f64,
    ha3: f64,
    xn: f64,
    xn1: f64,
    epsil: f64,
    e: f64,
    yk_plus_ky: f64,
    yj: f64,
    yl: f64,
    negate_c: bool,
) -> f64 {
    let hn2 = 1.0_f64 + hn1;
    let hn3 = hn2 + h3;
    let xxa = (xn + xn1) * epsil;
    if hna < xxa || hna1 < xxa || ha3 < xxa {
        return 0.0_f64;
    }
    let mut c = hn2
        * hn3
        * (hn2 * ((hn1 + h3) * (hn3 + 1.0_f64) - 2.0_f64 * h3 - hn1 * hn3)
            - (h3 * h3 + hn1 * hn3));
    let d = hn2 * hn2;
    let f = 1.0_f64;
    let b = f * h3;
    let f = f * hn1;
    let hn1_sq = hn3 * hn3;
    let a = d * hn1_sq * h3;
    let b = -hn1_sq * (b + f);
    let d = d * f;
    if negate_c {
        c = -c;
    }
    hnin * (a * e + b * (yk_plus_ky - yj) + d * (yl - yj)) / c
}

/// Persistent differentiation coefficient table for [`derivk`]
/// (`common/cderst/` in `adipls/adipack.c/gensr/derivk.new.d.f`).
#[derive(Debug, Clone, Default)]
pub struct DerivkCache {
    k: usize,
    k2: usize,
    dc: Vec<f64>,
}

/// Compute `2k`-th order derivative `dydx(1..=ii, 1..=nn)`, matching `derivk`
/// (`adipls/adipack.c/gensr/derivk.new.d.f:1-193`).
///
/// When `karg <= 0`, reuses previously cached coefficients in `cache`. Returns
/// `ii` (reset to `0` on invalid non-monotonic grid, matching upstream).
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
pub fn derivk(
    x: &[f64],
    y: &[f64],
    dydx: &mut [f64],
    ii: usize,
    nn: usize,
    iy: usize,
    idy: usize,
    karg: isize,
    cache: &mut DerivkCache,
) -> usize {
    if karg > 0 {
        let mut k = usize::try_from(karg).unwrap_or(1);
        let mut k2 = 2 * k + 1;
        if nn < k2 {
            k = (nn - 1) / 2;
            k2 = 2 * k + 1;
        }
        cache.k = k;
        cache.k2 = k2;
        let iw = k2;

        let diff = x[nn - 1] - x[0];
        if diff == 0.0_f64 {
            return 0;
        }
        cache.dc.resize(k2 * nn, 0.0_f64);
        let mut wrk = vec![0.0_f64; iw * iw];

        let mut jdc = 0_usize;
        for n in 0..nn {
            let xtr = x[n];
            let i = n.saturating_sub(k).min(nn - k2);
            let dlx = x[i + k2 - 1] - x[i];
            if diff * dlx <= 0.0_f64 {
                return 0;
            }
            let dlxi = 1.0_f64 / dlx;
            let mut m = 0_usize;
            for l in 0..k2 {
                let l1 = i + l;
                let xl = (x[l1] - xtr) * dlxi;
                let mut aa = 1.0_f64;
                wrk[m] = aa;
                m += 1;
                for _ir in 1..k2 {
                    aa *= xl;
                    wrk[m] = aa;
                    m += 1;
                }
                cache.dc[jdc + l] = 0.0_f64;
            }
            cache.dc[jdc + 1] = dlxi;

            let det = leq(&mut wrk, &mut cache.dc[jdc..jdc + k2], k2, 1, iw, 1);
            if det == 0.0_f64 {
                for l in 0..k2 {
                    cache.dc[jdc + l] = 0.0_f64;
                }
                if n > 0 && n + 1 < nn {
                    let dxp = x[n + 1] - x[n];
                    let dxm = x[n - 1] - x[n];
                    cache.dc[jdc + k] = -(dxp + dxm) / (dxp * dxm);
                    cache.dc[jdc + k - 1] = dxp / (dxm * (dxp - dxm));
                    cache.dc[jdc + k + 1] = -dxm / (dxp * (dxp - dxm));
                }
            }
            jdc += k2;
        }
    }

    let k = cache.k;
    let k2 = cache.k2;
    let mut jdc = 0_usize;
    for n in 0..nn {
        let i = n.saturating_sub(k).min(nn - k2);
        for is_idx in 0..ii {
            let mut sum = 0.0_f64;
            for l in 0..k2 {
                sum += cache.dc[jdc + l] * y[(i + l) * iy + is_idx];
            }
            dydx[n * idy + is_idx] = sum;
        }
        jdc += k2;
    }
    ii
}

/// Simpson / trapezoidal cumulative quadrature on arbitrary monotonic grids,
/// matching `vinta` (`adipls/adipack.c/gensr/vinta.d.f:1-134`).
#[allow(clippy::many_single_char_names, clippy::similar_names)]
pub fn vinta(
    x: &[f64],
    a: &[f64],
    y: &mut [f64],
    nn: usize,
    ia: usize,
    id: usize,
) {
    if nn == 0 {
        return;
    }
    if nn == 1 {
        y[0] = 0.0_f64;
        return;
    }
    if nn == 2 {
        y[0] = 0.0_f64;
        y[id] = 0.5_f64 * (x[1] - x[0]) * (a[0] + a[ia]);
        return;
    }

    let mut wc = a[0];
    let is = ia;
    let ird = id;
    let n = nn;
    let mut m = n / 2;
    let ie = n - m * 2;
    if ie == 0 {
        m -= 1;
    }
    let r = x[n - 1] - x[0];
    if r == 0.0_f64 {
        return;
    }

    let mut j3 = 0_usize;
    let mut k3 = 0_usize;
    y[0] = 0.0_f64;
    let mut vsimf = 0.0_f64;
    let mut wb = 0.0_f64;

    for i in 1..=m {
        let i2 = i * 2;
        let i1 = i2 - 1;
        let i3 = i2 + 1;
        let hn = (x[i2 - 1] - x[i1 - 1]) / 6.0_f64;
        let hn1 = (x[i3 - 1] - x[i2 - 1]) / 6.0_f64;
        if r * hn <= 0.0_f64 || r * hn1 <= 0.0_f64 {
            return;
        }
        let wa_h = hn + hn1;
        let wb_h = hn - hn1;
        let wd = wa_h / hn;
        let ah = wd * (hn + wb_h);
        let ah2_tmp = wa_h / hn1;
        let ah1 = wa_h * wd * ah2_tmp;
        let ah2 = ah2_tmp * (hn1 - wb_h);
        let bh1_tmp = hn + 3.0_f64 * hn1;
        let bh2_tmp = hn / hn1;
        let bh = (bh1_tmp + hn) * hn / wa_h;
        let bh1 = bh1_tmp * bh2_tmp;
        let bh2 = -hn * hn * bh2_tmp / wa_h;

        let wa = wc;
        let k2 = k3 + is;
        k3 = k2 + is;
        wb = a[k2];
        wc = a[k3];
        let j2 = j3 + ird;
        j3 = j2 + ird;
        y[j2] = vsimf + bh * wa + bh1 * wb + bh2 * wc;
        vsimf = vsimf + ah * wa + ah1 * wb + ah2 * wc;
        y[j3] = vsimf;
    }

    if ie != 0 {
        return;
    }

    let wa_x = x[n - 2];
    let hn = (wa_x - x[n - 3]) / 6.0_f64;
    let hn1 = (x[n - 1] - wa_x) / 6.0_f64;
    if r * hn <= 0.0_f64 || r * hn1 <= 0.0_f64 {
        return;
    }
    let wa_h = hn + hn1;
    let wd = hn1 / hn;
    let ah = -hn1 * hn1 * wd / wa_h;
    vsimf += ah * wb;
    let wb_h = 3.0_f64 * hn + hn1;
    let ah1 = wd * wb_h;
    let ah2 = hn1 * (wb_h + hn1) / wa_h;
    let wa = wc;
    let wb = a[k3 + is];
    vsimf = vsimf + ah1 * wa + ah2 * wb;
    y[j3 + ird] = vsimf;
}

/// Persistent integration coefficient table for [`vintk`]
/// (`common/cintst/` in `adipls/adipack.c/gensr/vintk.new.d.f`).
#[derive(Debug, Clone, Default)]
pub struct VintkCache {
    k: usize,
    k2: usize,
    dc: Vec<f64>,
}

/// `(2k - 1)`-th order cumulative integration, matching `vintk`
/// (`adipls/adipack.c/gensr/vintk.new.d.f:1-182`).
///
/// Caller must provide initial values `y[0..ii]` at `n = 0`. Returns `ii`
/// (reset to `0` on non-monotonic or singular grids, matching upstream).
#[allow(
    clippy::too_many_arguments,
    clippy::many_single_char_names,
    clippy::similar_names
)]
pub fn vintk(
    x: &[f64],
    a: &[f64],
    y: &mut [f64],
    ii: usize,
    nn: usize,
    idima: usize,
    idimy: usize,
    karg: isize,
    cache: &mut VintkCache,
) -> usize {
    if karg > 0 {
        let mut k = usize::try_from(karg).unwrap_or(1);
        let mut k2 = 2 * k;
        if nn < k2 {
            k = nn / 2;
            k2 = 2 * k;
        }
        cache.k = k;
        cache.k2 = k2;
        let iw = k2;
        let diff = x[nn - 1] - x[0];
        if diff == 0.0_f64 {
            return 0;
        }
        cache.dc.resize(k2 * (nn - 1), 0.0_f64);
        let mut wrk = vec![0.0_f64; iw * iw];
        let mut jdc = 0_usize;
        for n in 1..nn {
            let n1 = n - 1;
            let xtr = x[n1];
            // Upstream 1-based `n1 = n_1based - 1` equals 0-based `n`.
            let i = n.saturating_sub(k).min(nn - k2);
            let dlx = x[i + k2 - 1] - x[i];
            if diff * dlx <= 0.0_f64 {
                return 0;
            }
            let dlxi = 1.0_f64 / dlx;
            let mut m = 0_usize;
            for l in 0..k2 {
                let l1 = i + l;
                let xl = (x[l1] - xtr) * dlxi;
                let mut aa = 1.0_f64;
                wrk[m] = aa;
                m += 1;
                for _ir in 1..k2 {
                    aa *= xl;
                    wrk[m] = aa;
                    m += 1;
                }
            }
            let xl = (x[n] - xtr) * dlxi;
            let mut aa = 1.0_f64;
            for j in 0..k2 {
                aa *= xl;
                cache.dc[jdc + j] =
                    dlx * aa / f64::from(u32::try_from(j + 1).unwrap_or(1));
            }
            let det = leq(&mut wrk, &mut cache.dc[jdc..jdc + k2], k2, 1, iw, 1);
            if det == 0.0_f64 {
                return 0;
            }
            jdc += k2;
        }
    }

    let k = cache.k;
    let k2 = cache.k2;
    let mut jdc = 0_usize;
    for n in 1..nn {
        let n1 = n - 1;
        let i = n.saturating_sub(k).min(nn - k2);
        for is_idx in 0..ii {
            let mut sum = 0.0_f64;
            for l in 0..k2 {
                sum += cache.dc[jdc + l] * a[(i + l) * idima + is_idx];
            }
            y[n * idimy + is_idx] = y[n1 * idimy + is_idx] + sum;
        }
        jdc += k2;
    }
    ii
}

/// Persistent scan cursor for [`lir`] and [`lir1`] (`data n/-1/` in
/// `adipls/adipack.c/gensr/lir.d.f`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LirState {
    last_n: isize,
}

impl Default for LirState {
    fn default() -> Self {
        Self { last_n: -1 }
    }
}

/// Cubic Lagrange interpolation/extrapolation (linear when `nt < 4`), matching
/// `lir` (`adipls/adipack.c/gensr/lir.d.f:1-167`).
///
/// Returns `1` for interpolation, `0` for extrapolation, or `-1` on error.
#[allow(clippy::too_many_arguments)]
pub fn lir(
    z: f64,
    zi: &[f64],
    y: &mut [f64],
    yi: &[f64],
    ii: usize,
    id: usize,
    nt: usize,
    l: i32,
    state: &mut LirState,
) -> i32 {
    lir_impl(z, zi, y, yi, ii, id, nt, l, false, state)
}

/// Linear interpolation/extrapolation matching `lir1` (`adipls/adipack.c/gensr/lir.d.f:45-167`).
#[allow(clippy::too_many_arguments)]
pub fn lir1(
    z: f64,
    zi: &[f64],
    y: &mut [f64],
    yi: &[f64],
    ii: usize,
    id: usize,
    nt: usize,
    l: i32,
    state: &mut LirState,
) -> i32 {
    lir_impl(z, zi, y, yi, ii, id, nt, l, true, state)
}

#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
fn lir_impl(
    z: f64,
    zi: &[f64],
    y: &mut [f64],
    yi: &[f64],
    ii: usize,
    id: usize,
    nt: usize,
    l: i32,
    linear_only: bool,
    state: &mut LirState,
) -> i32 {
    if nt < 2 {
        return -1;
    }
    let il = linear_only || nt < 4;
    let mut inter = 1_i32;
    let diff = zi[nt - 1] - zi[0];
    if diff == 0.0_f64 {
        return -1;
    }

    // 1-based position `n` as in `lir.d.f:63-72`.
    let mut n = state.last_n - 1;
    if l <= 1 || n < 1 {
        n = 1;
    }

    let mut on_mesh = false;
    while n <= isize::try_from(nt).unwrap_or(isize::MAX) {
        let nu = usize::try_from(n - 1).unwrap_or(0);
        let cmp = zi[nu] - z;
        if cmp == 0.0_f64 {
            on_mesh = true;
            break;
        }
        if (diff > 0.0_f64 && cmp > 0.0_f64)
            || (diff < 0.0_f64 && cmp < 0.0_f64)
        {
            break;
        }
        n += 1;
    }

    if on_mesh {
        let nu = usize::try_from(n - 1).unwrap_or(0);
        let j = nu * id;
        y[..ii].copy_from_slice(&yi[j..(ii + j)]);
        state.last_n = n;
        return inter;
    }

    if n > isize::try_from(nt).unwrap_or(isize::MAX) || n <= 1 {
        inter = 0;
    }

    if il {
        let mut nu = usize::try_from(n).unwrap_or(2);
        if nu == 1 {
            nu = 2;
        }
        if nu > nt {
            nu = nt;
        }
        let z1 = zi[nu - 1];
        let y1 = (z1 - z) / (z1 - zi[nu - 2]);
        let y2 = 1.0_f64 - y1;
        let j = (nu - 1) * id;
        let m = (nu - 2) * id;
        for i in 0..ii {
            y[i] = y1 * yi[m + i] + y2 * yi[j + i];
        }
        n = isize::try_from(nu).unwrap_or(n);
    } else {
        let mut m = n;
        let mut k = 3_usize;
        if n <= 2 {
            m = 3;
            k = usize::try_from(n).unwrap_or(1);
        }
        if n >= isize::try_from(nt).unwrap_or(isize::MAX) {
            m = isize::try_from(nt - 1).unwrap_or(3);
            k = 4;
        }
        let mu = usize::try_from(m).unwrap_or(3);
        let y1 = zi[mu - 3];
        let y2 = zi[mu - 2];
        let y3 = zi[mu - 1];
        let y4 = zi[mu];

        let z1 = z - y1;
        let z2 = z - y2;
        let z3 = z - y3;
        let z4 = z - y4;
        let z12 = z1 * z2;
        let z34 = z3 * z4;

        let mut a = [
            z2 * z34 / ((y1 - y2) * (y1 - y3) * (y1 - y4)),
            z1 * z34 / ((y2 - y1) * (y2 - y3) * (y2 - y4)),
            z12 * z4 / ((y3 - y1) * (y3 - y2) * (y3 - y4)),
            z12 * z3 / ((y4 - y1) * (y4 - y2) * (y4 - y3)),
        ];
        let sum_a = a[0] + a[1] + a[2] + a[3];
        a[k - 1] = (1.0_f64 + a[k - 1]) - sum_a;

        let base = (mu - 3) * id;
        for i in 0..ii {
            let mut yy = 0.0_f64;
            for (j, &aj) in a.iter().enumerate() {
                yy += aj * yi[base + j * id + i];
            }
            y[i] = yy;
        }
    }

    state.last_n = n;
    inter
}
