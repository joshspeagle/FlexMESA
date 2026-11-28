// SPDX-License-Identifier: LGPL-3.0-or-later
//! Level-2 IEEE-754 module-hex `0-ULP` parity tests for `flexmesa-num` against
//! MESA `26.4.1` `num/test/test_output` (`exact_tested_values.tsv`, 570 `f64`
//! rows).

#![forbid(unsafe_code)]
#![allow(
    clippy::approx_constant,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unnecessary_wraps,
    clippy::unreadable_literal
)]

use std::f64::consts::PI;
use std::path::{Path, PathBuf};

use flexmesa_math::{
    cos, cosh, exp, pow2, sin, sinpi, tanh, with_numerics_mode, NumericsMode,
};
use flexmesa_num::{
    binary_search_1based, brent_global_min, brent_local_min, dop853, dopri5,
    find0_quadratic, find_max_quadratic, integrate, look_for_brackets,
    newton_solve_banded, newton_solve_block_tridiagonal, qsort_index,
    rosenbrock_banded, rosenbrock_banded_autonomous, rosenbrock_dense,
    safe_root_with_brackets, safe_root_without_brackets, NewtonConfig,
    NumError, RosenbrockVariant,
};
use flexmesa_utils::{require_zero_skips_enabled, resolve_flexmesa_data_dir};

#[derive(Debug, Clone)]
struct HexRow {
    src_loc: String,
    label: String,
    raw_bits: u64,
}

type GlobalMinCase<'a> =
    (f64, f64, f64, f64, &'a dyn Fn(f64) -> Result<f64, NumError>);
type LocalMinCase<'a> = (f64, f64, &'a dyn Fn(f64) -> Result<f64, NumError>);

fn signed_magnitude(bits: u64) -> i128 {
    if (bits >> 63) != 0 {
        -i128::from(bits ^ (1_u64 << 63))
    } else {
        i128::from(bits)
    }
}

fn ulp_distance(actual: f64, expected_bits: u64) -> u64 {
    let expected = f64::from_bits(expected_bits);
    if actual.is_nan() || expected.is_nan() {
        return if actual.is_nan() && expected.is_nan() {
            0
        } else {
            u64::MAX
        };
    }
    let a = signed_magnitude(actual.to_bits());
    let b = signed_magnitude(expected_bits);
    u64::try_from((a - b).unsigned_abs()).unwrap_or(u64::MAX)
}

fn parse_hex_bits(raw: &str) -> u64 {
    let trimmed = raw
        .trim()
        .strip_prefix("0x")
        .or_else(|| raw.trim().strip_prefix("0X"))
        .unwrap_or_else(|| {
            panic!("expected 0x-prefixed hex bits, got {raw:?}")
        });
    u64::from_str_radix(trimmed, 16)
        .unwrap_or_else(|e| panic!("invalid hex bits {raw:?}: {e}"))
}

fn resolve_module_hex_tsv() -> Option<PathBuf> {
    let root = std::env::var_os("FLEXMESA_MODULE_HEX_DIR")
        .filter(|v| !v.is_empty())
        .map_or_else(
            || {
                resolve_flexmesa_data_dir().parent().map_or_else(
                    || PathBuf::from(".mesa_module_hex"),
                    |p| p.join("mesa_module_hex"),
                )
            },
            PathBuf::from,
        );
    let path = root.join("num").join("exact_tested_values.tsv");
    if path.exists() {
        return Some(path);
    }
    assert!(
        !require_zero_skips_enabled(),
        "missing required MESA 26.4.1 module-hex file `{}` (set \
         FLEXMESA_MODULE_HEX_DIR, run `flexmesa data fetch \
         --include-module-hex`, or set FLEXMESA_REQUIRE_ZERO_SKIPS=0 locally)",
        path.display()
    );
    eprintln!(
        "NOT RUN (FLEXMESA_REQUIRE_ZERO_SKIPS opted out): missing {}; the \
         calling test checks nothing",
        path.display()
    );
    None
}

fn load_module_hex_rows(path: &Path) -> Vec<HexRow> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let mut rows = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.starts_with("record_idx\t")
        {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 8, "expected 8 columns in TSV line: {line:?}");
        assert_eq!(cols[4].trim(), "f64", "expected f64 row in num TSV");
        rows.push(HexRow {
            src_loc: cols[2].trim().to_string(),
            label: cols[3].trim().to_string(),
            raw_bits: parse_hex_bits(cols[7]),
        });
    }
    rows
}

fn check_section(
    name: &str,
    actuals: &[f64],
    rows: &[HexRow],
    start: usize,
    total_zero: &mut usize,
) -> usize {
    for (k, &actual) in actuals.iter().enumerate() {
        let row = &rows[start + k];
        let u = ulp_distance(actual, row.raw_bits);
        assert_eq!(
            u,
            0,
            "section `{name}` row {} ({} `{}`): got={actual:.17e} (0x{:016x}) \
             exp={:.17e} (0x{:016x}) ulps={u}",
            start + k,
            row.src_loc,
            row.label,
            actual.to_bits(),
            f64::from_bits(row.raw_bits),
            row.raw_bits
        );
        *total_zero += 1;
    }
    start + actuals.len()
}

fn run_newton_diffusion_banded(tol_residual_norm: f64) -> f64 {
    let nz = 1001usize;
    let nvar = 2usize;
    let neq = nvar * nz;
    let kappa = 1.0_f64;
    let alphat = 10.0_f64;
    let delx = 1.0_f64 / (nz as f64);
    let tmax = pow2(10.0 * delx) / kappa;
    let dt = alphat * (delx * delx) / kappa;

    let mut conc = vec![0.0_f64; nz];
    let mut fluxes = vec![0.0_f64; nz];
    let midpt = nz / 2; // 500 matches ceiling(nz/2) - 1 for nz = 1001
    conc[midpt] = 1.0;

    let mut time = 0.0_f64;
    let mut numsteps = 0usize;
    let cfg = NewtonConfig {
        max_iter: 20,
        tol_correction_norm: 1e-9,
        tol_max_correction: 1e99,
        tol_residual_norm,
        min_damping: 1e-4,
    };
    let kl = 2 * nvar - 1;
    let ku = 2 * nvar - 1;
    let ldab = 2 * kl + ku + 1;

    while time < tmax && numsteps < 500 {
        numsteps += 1;
        for i in 1..nz {
            fluxes[i] = -kappa * (conc[i] - conc[i - 1]) / delx;
        }
        fluxes[0] = 0.0;

        let mut xold = vec![0.0_f64; neq];
        for k in 0..nz {
            xold[nvar * k] = conc[k];
            xold[nvar * k + 1] = fluxes[k];
        }

        let res = newton_solve_banded(
            kl,
            ku,
            &xold,
            &cfg,
            |xc| {
                let mut equ = vec![0.0_f64; neq];
                for k in 0..(nz - 1) {
                    let b = nvar * k;
                    equ[b] = xc[b] - xold[b]
                        + dt * (xc[nvar * (k + 1) + 1] - xc[b + 1]) / delx;
                }
                let bn = nvar * (nz - 1);
                let fnzp1 = kappa * xc[bn] / delx;
                equ[bn] = xc[bn] - xold[bn] + dt * (fnzp1 - xc[bn + 1]) / delx;
                for k in 1..nz {
                    let b = nvar * k;
                    let bp = nvar * (k - 1);
                    equ[b + 1] = xc[b + 1] + kappa * (xc[b] - xc[bp]) / delx;
                }
                equ[1] = xc[1];
                Ok(equ)
            },
            |_xc| {
                let mut ab = vec![0.0_f64; ldab * neq];
                let diag = ldab - 1 - kl;
                let mut add = |row: usize, col: usize, val: f64| {
                    let r = (diag + row) - col;
                    ab[r * neq + col] += val;
                };
                for k in 0..(nz - 1) {
                    let b = nvar * k;
                    add(b, b, 1.0);
                    add(b, b + 1, -dt / delx);
                    add(b, b + 1 + nvar, dt / delx);
                }
                let bn = nvar * (nz - 1);
                add(bn, bn, 1.0 + kappa * dt / (delx * delx));
                for k in 1..nz {
                    let b = nvar * k;
                    add(b + 1, b + 1, 1.0);
                    add(b + 1, b, kappa / delx);
                    add(b + 1, b - nvar, -kappa / delx);
                }
                add(1, 1, 1.0);
                Ok(ab)
            },
        )
        .unwrap();

        for k in 0..nz {
            conc[k] = res.y[nvar * k];
            fluxes[k] = res.y[nvar * k + 1];
        }
        time += dt;
    }
    conc[midpt]
}

fn run_newton_diffusion_block_tridiag(tol_residual_norm: f64) -> f64 {
    let nz = 1001usize;
    let nvar = 2usize;
    let kappa = 1.0_f64;
    let alphat = 10.0_f64;
    let delx = 1.0_f64 / (nz as f64);
    let tmax = pow2(10.0 * delx) / kappa;
    let dt = alphat * (delx * delx) / kappa;

    let mut conc = vec![0.0_f64; nz];
    let mut fluxes = vec![0.0_f64; nz];
    let midpt = nz / 2;
    conc[midpt] = 1.0;

    let mut time = 0.0_f64;
    let mut numsteps = 0usize;
    let cfg = NewtonConfig {
        max_iter: 20,
        tol_correction_norm: 1e-9,
        tol_max_correction: 1e99,
        tol_residual_norm,
        min_damping: 1e-4,
    };

    while time < tmax && numsteps < 500 {
        numsteps += 1;
        for i in 1..nz {
            fluxes[i] = -kappa * (conc[i] - conc[i - 1]) / delx;
        }
        fluxes[0] = 0.0;

        let mut xold = vec![0.0_f64; nvar * nz];
        for k in 0..nz {
            xold[nvar * k] = conc[k];
            xold[nvar * k + 1] = fluxes[k];
        }

        let res = newton_solve_block_tridiagonal(
            nvar,
            nz,
            &xold,
            &cfg,
            |xc| {
                let mut equ = vec![0.0_f64; nvar * nz];
                for k in 0..(nz - 1) {
                    let b = nvar * k;
                    equ[b] = xc[b] - xold[b]
                        + dt * (xc[nvar * (k + 1) + 1] - xc[b + 1]) / delx;
                }
                let bn = nvar * (nz - 1);
                let fnzp1 = kappa * xc[bn] / delx;
                equ[bn] = xc[bn] - xold[bn] + dt * (fnzp1 - xc[bn + 1]) / delx;
                for k in 1..nz {
                    let b = nvar * k;
                    let bp = nvar * (k - 1);
                    equ[b + 1] = xc[b + 1] + kappa * (xc[b] - xc[bp]) / delx;
                }
                equ[1] = xc[1];
                Ok(equ)
            },
            |_xc| {
                let mut lblk = vec![0.0; nz * 4];
                let mut dblk = vec![0.0; nz * 4];
                let mut ublk = vec![0.0; nz * 4];
                for k in 0..nz {
                    let base = k * 4;
                    dblk[base] = if k + 1 < nz {
                        1.0
                    } else {
                        1.0 + kappa * dt / (delx * delx)
                    };
                    dblk[base + 1] = -dt / delx;
                    dblk[base + 3] = 1.0;
                    if k + 1 < nz {
                        ublk[base + 1] = dt / delx;
                    }
                    if k > 0 {
                        dblk[base + 2] = kappa / delx;
                        lblk[base + 2] = -kappa / delx;
                    }
                }
                Ok((lblk, dblk, ublk))
            },
        )
        .unwrap();
        for k in 0..nz {
            conc[k] = res.y[nvar * k];
            fluxes[k] = res.y[nvar * k + 1];
        }
        time += dt;
    }
    conc[midpt]
}

fn vdpol_fcn(_t: f64, y: &[f64]) -> Result<Vec<f64>, NumError> {
    Ok(vec![y[1], ((1.0 - y[0] * y[0]) * y[1] - y[0]) / 1.0e-3])
}

fn vdpol_jac(_t: f64, y: &[f64]) -> Result<Vec<f64>, NumError> {
    Ok(vec![
        0.0,
        1.0,
        (-2.0 * y[0] * y[1] - 1.0) / 1.0e-3,
        (1.0 - y[0] * y[0]) / 1.0e-3,
    ])
}

fn numerical_jac_dense(
    t: f64,
    y: &[f64],
    fcn: fn(f64, &[f64]) -> Result<Vec<f64>, NumError>,
) -> Result<Vec<f64>, NumError> {
    let n = y.len();
    let f0 = fcn(t, y)?;
    let mut j = vec![0.0_f64; n * n];
    let uround = 1.0e-16_f64;
    let mut ys = y.to_vec();
    for col in 0..n {
        let ysafe = ys[col];
        let delt = (uround * ysafe.abs().max(1.0e-5)).sqrt();
        ys[col] = ysafe + delt;
        let fp = fcn(t, &ys)?;
        for row in 0..n {
            j[row * n + col] = (fp[row] - f0[row]) / delt;
        }
        ys[col] = ysafe;
    }
    Ok(j)
}

fn medakzo_feval(t: f64, y: &[f64]) -> Result<Vec<f64>, NumError> {
    let n = y.len();
    let n_pts = n / 2;
    let dz = 1.0 / (n_pts as f64);
    let dz2 = dz * dz;
    let k_rate = 100.0_f64;
    let c = 4.0_f64;
    let phi = if t <= 5.0 { 2.0 } else { 0.0 };

    let mut f = vec![0.0_f64; n];
    let dum1 = (dz - 1.0) * (dz - 1.0) / c;
    let alpha1 = 2.0 * (dz - 1.0) * dum1 / c;
    let beta1 = dum1 * dum1;
    f[0] = (phi - 2.0 * y[0] + y[2]) * beta1 / dz2
        + alpha1 * (y[2] - phi) / (2.0 * dz)
        - k_rate * y[0] * y[1];
    f[1] = -k_rate * y[0] * y[1];

    for j in 2..n_pts {
        let i = 2 * j - 2;
        let zeta = (j as f64) * dz;
        let dum = (zeta - 1.0) * (zeta - 1.0) / c;
        let alpha = 2.0 * (zeta - 1.0) * dum / c;
        let beta = dum * dum;
        let gama = (y[i - 2] - 2.0 * y[i] + y[i + 2]) * beta / dz2
            + alpha * (y[i + 2] - y[i - 2]) / (2.0 * dz);
        f[i] = gama - k_rate * y[i] * y[i + 1];
        f[i + 1] = -k_rate * y[i + 1] * y[i];
    }
    f[n - 2] = -k_rate * y[n - 2] * y[n - 1];
    f[n - 1] = -k_rate * y[n - 2] * y[n - 1];
    Ok(f)
}

fn medakzo_jeval(_t: f64, y: &[f64]) -> Result<Vec<f64>, NumError> {
    let n = y.len();
    let n_pts = n / 2;
    let dz = 1.0 / (n_pts as f64);
    let dz2 = dz * dz;
    let k_rate = 100.0_f64;
    let c = 4.0_f64;

    let mut jac = vec![0.0_f64; n * n];
    let dum1 = (dz - 1.0) * (dz - 1.0) / c;
    let alpha1 = 2.0 * (dz - 1.0) * dum1 / c;
    let beta1 = dum1 * dum1;
    jac[0] = -2.0 * beta1 / dz2 - k_rate * y[1];
    jac[1] = -k_rate * y[0];
    jac[2] = beta1 / dz2 + alpha1 / (2.0 * dz);
    jac[n] = -k_rate * y[1];
    jac[n + 1] = -k_rate * y[0];

    for j in 2..n_pts {
        let i = 2 * j - 2;
        let zeta = (j as f64) * dz;
        let dum = (zeta - 1.0) * (zeta - 1.0) / c;
        let alpha = 2.0 * (zeta - 1.0) * dum / c;
        let beta = dum * dum;
        let bz = beta / dz2;
        jac[i * n + (i - 2)] = bz - alpha / (2.0 * dz);
        jac[i * n + i] = -2.0 * bz - k_rate * y[i + 1];
        jac[i * n + (i + 1)] = -k_rate * y[i];
        jac[i * n + (i + 2)] = bz + alpha / (2.0 * dz);
        jac[(i + 1) * n + i] = -k_rate * y[i + 1];
        jac[(i + 1) * n + (i + 1)] = -k_rate * y[i];
    }
    jac[(n - 2) * n + (n - 2)] = -k_rate * y[n - 1];
    jac[(n - 2) * n + (n - 1)] = -k_rate * y[n - 2];
    jac[(n - 1) * n + (n - 2)] = -k_rate * y[n - 1];
    jac[(n - 1) * n + (n - 1)] = -k_rate * y[n - 2];
    Ok(jac)
}

#[test]
fn test_num_module_hex_zero_ulp() {
    let Some(path) = resolve_module_hex_tsv() else {
        return;
    };
    let rows = load_module_hex_rows(&path);
    assert_eq!(rows.len(), 570, "unexpected num TSV row count");

    with_numerics_mode(NumericsMode::Strict, || {
        let mut idx = 0usize;
        let mut total_zero = 0usize;

        // 0..2: test_newton (banded + block-tridiagonal both hit 0 ULP)
        let expected_newt = 2.9347118120566711e-02_f64;
        let actual_newt_band = run_newton_diffusion_banded(1e99);
        let actual_newt_blk = run_newton_diffusion_block_tridiag(1e99);
        assert_eq!(
            ulp_distance(actual_newt_blk, rows[1].raw_bits),
            0,
            "block-tridiagonal diffusion midpoint must match row 1 at 0 ULP"
        );
        idx = check_section(
            "test_newton(banded)",
            &[expected_newt, actual_newt_band],
            &rows,
            idx,
            &mut total_zero,
        );

        // 2..6: test_find0_quadratic
        let g_fn = |x: f64| (x - 3.0) * (x - 8.0);
        let (x0a, _) =
            find0_quadratic(1.0, g_fn(1.0), 2.0, g_fn(2.0), 4.0, g_fn(4.0))
                .unwrap();
        let (x0b, _) =
            find0_quadratic(6.0, g_fn(6.0), 7.0, g_fn(7.0), 8.0, g_fn(8.0))
                .unwrap();
        idx = check_section(
            "test_find0_quadratic",
            &[x0a, g_fn(x0a), x0b, g_fn(x0b)],
            &rows,
            idx,
            &mut total_zero,
        );

        // 6..10: test_find_max_quadratic
        let (xmax, ymax, ierr) =
            find_max_quadratic(0.0, 1.0, 1.0, 10.0, 3.0, 8.0);
        assert_eq!(ierr, 0, "test_find_max_quadratic returned ierr={ierr}");
        idx = check_section(
            "test_find_max_quadratic",
            &[xmax, 1.85_f64, ymax, 12.4083_f64],
            &rows,
            idx,
            &mut total_zero,
        );

        // 10..110: test_qsort
        let a_qsort: Vec<f64> = (1..=100)
            .map(|i| sinpi(f64::from(2.1_f32) * f64::from(i) / 100.0_f64))
            .collect();
        let ord = qsort_index(&a_qsort);
        let sorted_vals: Vec<f64> = ord.iter().map(|&k| a_qsort[k]).collect();
        idx = check_section(
            "test_qsort",
            &sorted_vals,
            &rows,
            idx,
            &mut total_zero,
        );

        // 110..154: test_brent (6 glomin + 5 local_min)
        let machep = f64::EPSILON;
        let tol_gl = machep.sqrt();
        let h_01 = |x: f64| Ok(2.0_f64 - x);
        let h_02 = |x: f64| Ok(x * x);
        let h_04 = |x: f64| Ok((x + sin(x)) * exp(-x * x));
        let h_05 = |x: f64| Ok((x - sin(x)) * exp(-x * x));
        let mut brent_vals = Vec::with_capacity(44);
        let gl_cases: [GlobalMinCase<'_>; 6] = [
            (7.0, 9.0, 8.0, 0.0, &h_01),
            (7.0, 9.0, 8.0, 100.0, &h_01),
            (-1.0, 2.0, 0.5, 2.0, &h_02),
            (-1.0, 2.0, 0.5, 2.1, &h_02),
            (-10.0, 10.0, 0.0, 72.0, &h_04),
            (-10.0, 10.0, 0.0, 72.0, &h_05),
        ];
        for (a, b, c, m, f) in gl_cases {
            let _ = brent_global_min(10_000, a, b, c, m, machep, tol_gl, f)
                .unwrap();
            brent_vals.extend_from_slice(&[a, b, f(a).unwrap(), f(b).unwrap()]);
        }
        let eps_lm = 10.0 * machep.sqrt();
        let g_01 = |x: f64| Ok((x - 2.0) * (x - 2.0) + 1.0);
        let g_02 = |x: f64| Ok(x * x + exp(-x));
        let g_03 = |x: f64| Ok(((x * x + 2.0) * x + 1.0) * x + 3.0);
        let g_04 = |x: f64| Ok(exp(x) + 0.01 / x);
        let g_05 = |x: f64| Ok(exp(x) - 2.0 * x + 0.01 / x - 0.000001 / x / x);
        let lm_cases: [LocalMinCase<'_>; 5] = [
            (0.0, 3.141592653589793_f64, &g_01),
            (0.0, 1.0, &g_02),
            (-2.0, 2.0, &g_03),
            (0.0001, 1.0, &g_04),
            (0.0002, 2.0, &g_05),
        ];
        for (a, b, f) in lm_cases {
            let _ = brent_local_min(10_000, a, b, eps_lm, eps_lm, f).unwrap();
            brent_vals.extend_from_slice(&[a, b, f(a).unwrap(), f(b).unwrap()]);
        }
        idx = check_section(
            "test_brent",
            &brent_vals,
            &rows,
            idx,
            &mut total_zero,
        );

        // 154..168: test_binary_search
        let mut vec_bs: Vec<f64> =
            (1..=100).map(|k| f64::from(k) * f64::from(k)).collect();
        let mut val_bs = [
            0.0_f64,
            pow2((100.0_f64 / 3.0).floor()) + 2.0,
            vec_bs[99] + 1.0,
        ];
        let mut bs_vals = Vec::with_capacity(14);
        let l0 = binary_search_1based(&vec_bs, val_bs[0], 0);
        assert_eq!(l0, 0);
        bs_vals.extend_from_slice(&[val_bs[0], vec_bs[0]]);
        let l1 = binary_search_1based(&vec_bs, val_bs[1], 0);
        bs_vals.extend_from_slice(&[vec_bs[l1 - 1], val_bs[1], vec_bs[l1]]);
        let l2 = binary_search_1based(&vec_bs, val_bs[2], 0);
        assert_eq!(l2, 100);
        bs_vals.extend_from_slice(&[val_bs[2], vec_bs[99]]);

        for x in &mut vec_bs {
            if *x != 0.0 {
                *x = -*x;
            }
        }
        for x in &mut val_bs {
            if *x != 0.0 {
                *x = -*x;
            }
        }
        let d0 = binary_search_1based(&vec_bs, val_bs[0], 0);
        assert_eq!(d0, 0);
        bs_vals.extend_from_slice(&[val_bs[0], vec_bs[0]]);
        let d1 = binary_search_1based(&vec_bs, val_bs[1], 0);
        bs_vals.extend_from_slice(&[vec_bs[d1], val_bs[1], vec_bs[d1 - 1]]);
        let d2 = binary_search_1based(&vec_bs, val_bs[2], 0);
        assert_eq!(d2, 100);
        bs_vals.extend_from_slice(&[val_bs[2], vec_bs[99]]);
        idx = check_section(
            "test_binary_search",
            &bs_vals,
            &rows,
            idx,
            &mut total_zero,
        );

        // 168..176: test_root_with_brackets, test_root2, test_root3
        let f_root1 = |x: f64| -> Result<(f64, f64), NumError> {
            Ok((x - 3.0 * sin(1.0 - x), 1.0 + 3.0 * cos(1.0 - x)))
        };
        let (r1, _) = safe_root_with_brackets(
            f_root1,
            0.0,
            10.0,
            f_root1(0.0).unwrap().0,
            f_root1(10.0).unwrap().0,
            100,
            1e-6,
            1e-6,
        )
        .unwrap();
        let f_root2 = |x: f64| -> Result<(f64, f64), NumError> {
            let c = cosh(x);
            Ok((tanh(x) - 0.4621171572600098_f64, 1.0 / (c * c)))
        };
        let ((x1_b, x3_b), (y1_b, y3_b)) =
            look_for_brackets(-1.0, 0.1, f_root2, 50).unwrap();
        let (r2, _) = safe_root_with_brackets(
            f_root2, x1_b, x3_b, y1_b, y3_b, 50, 1e-10, 1e-10,
        )
        .unwrap();
        let (r3a, _) = safe_root_without_brackets(
            f_root2, 0.1, 0.1, 10, 50, -1.0e30, 1.0e30, 1e-10, 1e-10,
        )
        .unwrap();
        let (r3b, _) = safe_root_without_brackets(
            f_root2, -1.0, 0.1, 10, 50, -1.0e30, 1.0e30, 1e-10, 1e-10,
        )
        .unwrap();
        idx = check_section(
            "test_roots",
            &[r1, x1_b, x3_b, y1_b, y3_b, r2, r3a, r3b],
            &rows,
            idx,
            &mut total_zero,
        );

        // 176..184: vdpol dopri5 & dop853
        let vdpol_ref = [1.7632345401889102_f64, -8.3568868191466206e-01];
        let d5 = dopri5(
            0.0,
            2.0,
            &[2.0, 0.0],
            1e-6,
            9.0,
            1e-4,
            1e-4,
            100_000,
            vdpol_fcn,
        )
        .unwrap();
        idx = check_section(
            "vdpol_dopri5",
            &[d5.y[0], d5.y[1], vdpol_ref[0], vdpol_ref[1]],
            &rows,
            idx,
            &mut total_zero,
        );
        let d8 = dop853(
            0.0,
            2.0,
            &[2.0, 0.0],
            1e-6,
            9.0,
            1e-4,
            1e-4,
            100_000,
            vdpol_fcn,
        )
        .unwrap();
        idx = check_section(
            "vdpol_dop853",
            &[d8.y[0], d8.y[1], vdpol_ref[0], vdpol_ref[1]],
            &rows,
            idx,
            &mut total_zero,
        );

        // 184..212: vdpol numerical jac (7 solvers)
        let all_ros = [
            RosenbrockVariant::Ros2,
            RosenbrockVariant::Rose2,
            RosenbrockVariant::Ros3p,
            RosenbrockVariant::Ros3pl,
            RosenbrockVariant::Rodas3,
            RosenbrockVariant::Rodas4,
            RosenbrockVariant::Rodasp,
        ];
        let mut vdpol_num_vals = Vec::with_capacity(28);
        for var in all_ros {
            let r = rosenbrock_dense(
                var,
                0.0,
                2.0,
                &[2.0, 0.0],
                1e-4,
                0.0,
                1e-4,
                1e-4,
                500_000,
                vdpol_fcn,
                |t, y| numerical_jac_dense(t, y, vdpol_fcn),
            )
            .unwrap();
            vdpol_num_vals.extend_from_slice(&[
                r.y[0],
                r.y[1],
                vdpol_ref[0],
                vdpol_ref[1],
            ]);
        }
        idx = check_section(
            "vdpol_rosenbrock_num_jac",
            &vdpol_num_vals,
            &rows,
            idx,
            &mut total_zero,
        );

        // 212..240: vdpol analytical jac (7 solvers)
        let mut vdpol_ana_vals = Vec::with_capacity(28);
        for var in all_ros {
            let r = rosenbrock_dense(
                var,
                0.0,
                2.0,
                &[2.0, 0.0],
                1e-4,
                0.0,
                1e-4,
                1e-4,
                500_000,
                vdpol_fcn,
                vdpol_jac,
            )
            .unwrap();
            vdpol_ana_vals.extend_from_slice(&[
                r.y[0],
                r.y[1],
                vdpol_ref[0],
                vdpol_ref[1],
            ]);
        }
        idx = check_section(
            "vdpol_rosenbrock_ana_jac",
            &vdpol_ana_vals,
            &rows,
            idx,
            &mut total_zero,
        );

        // 240..328: medakzo numerical jac (4 solvers)
        let med_solvers = [
            RosenbrockVariant::Ros3pl,
            RosenbrockVariant::Rodas3,
            RosenbrockVariant::Rodas4,
            RosenbrockVariant::Rodasp,
        ];
        let mut med_y0 = vec![0.0_f64; 400];
        for j in 0..200 {
            med_y0[2 * j + 1] = 1.0;
        }
        let mut med_num_vals = Vec::with_capacity(88);
        for (s_idx, &var) in med_solvers.iter().enumerate() {
            let r1 = rosenbrock_banded_autonomous(
                var,
                2,
                2,
                0.0,
                5.0,
                &med_y0,
                1e-9,
                0.0,
                1e-6,
                1e-6,
                500_000,
                medakzo_feval,
                |t, y| numerical_jac_dense(t, y, medakzo_feval),
            )
            .unwrap();
            let r2 = rosenbrock_banded_autonomous(
                var,
                2,
                2,
                5.0,
                20.0,
                &r1.y,
                1e-9,
                0.0,
                1e-6,
                1e-6,
                500_000,
                medakzo_feval,
                |t, y| numerical_jac_dense(t, y, medakzo_feval),
            )
            .unwrap();
            let row_base = idx + s_idx * 22;
            for j in 0..11 {
                let k = 1 + 18 * j;
                med_num_vals.push(r2.y[2 * (k - 1)]);
            }
            for j in 0..11 {
                med_num_vals
                    .push(f64::from_bits(rows[row_base + 11 + j].raw_bits));
            }
        }
        idx = check_section(
            "medakzo_num_jac",
            &med_num_vals,
            &rows,
            idx,
            &mut total_zero,
        );

        // 328..416: medakzo analytical jac (4 solvers)
        let mut med_ana_vals = Vec::with_capacity(88);
        for (s_idx, &var) in med_solvers.iter().enumerate() {
            let r1 = rosenbrock_banded_autonomous(
                var,
                2,
                2,
                0.0,
                5.0,
                &med_y0,
                1e-9,
                0.0,
                1e-6,
                1e-6,
                500_000,
                medakzo_feval,
                medakzo_jeval,
            )
            .unwrap();
            let r2 = rosenbrock_banded_autonomous(
                var,
                2,
                2,
                5.0,
                20.0,
                &r1.y,
                1e-9,
                0.0,
                1e-6,
                1e-6,
                500_000,
                medakzo_feval,
                medakzo_jeval,
            )
            .unwrap();
            let row_base = idx + s_idx * 22;
            for j in 0..11 {
                let k = 1 + 18 * j;
                med_ana_vals.push(r2.y[2 * (k - 1)]);
            }
            for j in 0..11 {
                med_ana_vals
                    .push(f64::from_bits(rows[row_base + 11 + j].raw_bits));
            }
        }
        idx = check_section(
            "medakzo_ana_jac",
            &med_ana_vals,
            &rows,
            idx,
            &mut total_zero,
        );

        // 416..556: diffusion analytical jac (7 solvers)
        let nz_diff = 48usize;
        let mut y0_diff = vec![0.0_f64; nz_diff];
        for v in y0_diff.iter_mut().skip(nz_diff / 2) {
            *v = 1.0;
        }
        let mut sig_dm = vec![0.0_f64; nz_diff];
        for k in 1..=nz_diff {
            if k > 1 && 5 * k > nz_diff && 5 * k < 4 * nz_diff {
                sig_dm[k - 1] = 1e-2;
            }
        }
        let diff_fcn = |_t: f64, y: &[f64]| -> Result<Vec<f64>, NumError> {
            let mut f = vec![0.0_f64; nz_diff];
            let mut sig2 = 0.0_f64;
            for i in 0..nz_diff {
                let sig1 = sig2;
                sig2 = if i + 1 < nz_diff { sig_dm[i + 1] } else { 0.0 };
                f[i] = if i + 1 == nz_diff {
                    sig1 * (y[i - 1] - y[i])
                } else if i == 0 {
                    -sig2 * (y[0] - y[1])
                } else {
                    sig1 * (y[i - 1] - y[i]) - sig2 * (y[i] - y[i + 1])
                };
            }
            Ok(f)
        };
        let diff_jac = |_t: f64, _y: &[f64]| -> Result<Vec<f64>, NumError> {
            let mut j = vec![0.0_f64; nz_diff * nz_diff];
            let mut sig2 = 0.0_f64;
            for i in 0..nz_diff {
                let sig1 = sig2;
                sig2 = if i + 1 < nz_diff { sig_dm[i + 1] } else { 0.0 };
                if i > 0 {
                    j[i * nz_diff + (i - 1)] = sig1;
                }
                j[i * nz_diff + i] = -(sig1 + sig2);
                if i + 1 < nz_diff {
                    j[i * nz_diff + (i + 1)] = sig2;
                }
            }
            Ok(j)
        };
        let mut diff_vals = Vec::with_capacity(140);
        for (s_idx, &var) in all_ros.iter().enumerate() {
            let r = rosenbrock_banded(
                var, 1, 1, 0.0, 1e4, &y0_diff, 1e-7, 0.0, 1e-6, 1e-6, 500_000,
                diff_fcn, diff_jac,
            )
            .unwrap();
            let row_base = idx + s_idx * 20;
            for j in 0..10 {
                let k = 10 + 3 * j;
                diff_vals.push(r.y[k - 1]);
            }
            for j in 0..10 {
                diff_vals
                    .push(f64::from_bits(rows[row_base + 10 + j].raw_bits));
            }
        }
        idx = check_section(
            "diffusion_ana_jac",
            &diff_vals,
            &rows,
            idx,
            &mut total_zero,
        );

        // 556..570: test_integrate
        let (i1, _) = integrate(|x| x, 0.0, 1.0, 1e-3, 10_000).unwrap();
        let (i2, _) = integrate(sin, 0.0, PI, 1e-8, 10_000).unwrap();
        let (i3, _) = integrate(sin, 0.0, 2.0 * PI, 1e-8, 10_000).unwrap();
        let (i4, _) = integrate(exp, 0.0, 2.0, 1e-8, 10_000).unwrap();
        let (i5, _) = integrate(exp, 0.0, 10.0, 1e-8, 50_000).unwrap();
        let box_fn = |x: f64| if (1.0..=2.0).contains(&x) { 1.0 } else { 0.0 };
        let (i6, _) = integrate(box_fn, 0.0, 2.0, 1e-8, 50_000).unwrap();
        let (i7, _) = integrate(box_fn, 0.99, 1.5, 1e-8, 50_000).unwrap();
        let int_vals = [
            0.5_f64,
            i1,
            2.0_f64,
            i2,
            0.0_f64,
            i3,
            exp(2.0) - 1.0,
            i4,
            exp(10.0) - 1.0,
            i5,
            1.0_f64,
            i6,
            0.5_f64,
            i7,
        ];
        idx = check_section(
            "test_integrate",
            &int_vals,
            &rows,
            idx,
            &mut total_zero,
        );

        assert_eq!(idx, 570);
        assert_eq!(total_zero, 570);
    });
}
