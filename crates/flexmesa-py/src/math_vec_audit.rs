// SPDX-License-Identifier: LGPL-3.0-or-later
//! Reference-vector `Strict` `SHA-256` hashing, `Fast`-vs-`Strict` comparison,
//! and dispatch-overhead microbenchmark bindings (`#165` Items 2, 15, 17).

use std::collections::BTreeMap;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

use flexmesa_data::sha256::compute_sha256_bytes;
use flexmesa_math::{
    acos, acosh, acospi, asin, asinh, asinpi, atan, atan2, atanh, atanpi, cos,
    cosh, cospi, exp, exp10, expm1, fma, hypot, intrinsic_exp, intrinsic_log,
    intrinsic_log10, intrinsic_pow, log, log10, log1p, log2, math_init, pow,
    pow2, pow3, pow4, pow5, pow6, pow7, pow8, powi, powi_f90, powm1, safe_log,
    safe_log10, safe_sqrt, sin, sinh, sinpi, tan, tanh, tanpi,
    with_numerics_mode, NumericsMode,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type FastVsStrictRow = (String, usize, usize, usize, u64, f64, f64);

#[inline]
fn canonical_strict_bits(val: f64) -> u64 {
    if val.is_nan() {
        0x7ff8_0000_0000_0000_u64
    } else {
        val.to_bits()
    }
}

fn eval_unary_sec(sec: &str, x: f64) -> Result<f64, String> {
    let val = match sec {
        "safe_sqrt" => safe_sqrt(x),
        "log" => log(x),
        "safe_log" => safe_log(x),
        "log10" => log10(x),
        "safe_log10" => safe_log10(x),
        "log1p" => log1p(x),
        "log2" => log2(x),
        "exp" => exp(x),
        "exp10" => exp10(x),
        "expm1" => expm1(x),
        "powm1" => powm1(x),
        "pow2" => pow2(x),
        "pow3" => pow3(x),
        "pow4" => pow4(x),
        "pow5" => pow5(x),
        "pow6" => pow6(x),
        "pow7" => pow7(x),
        "pow8" => pow8(x),
        "cos" => cos(x),
        "sin" => sin(x),
        "tan" => tan(x),
        "cospi" => cospi(x),
        "sinpi" => sinpi(x),
        "tanpi" => tanpi(x),
        "acos" => acos(x),
        "asin" => asin(x),
        "atan" => atan(x),
        "acospi" => acospi(x),
        "asinpi" => asinpi(x),
        "atanpi" => atanpi(x),
        "cosh" => cosh(x),
        "sinh" => sinh(x),
        "tanh" => tanh(x),
        "acosh" => acosh(x),
        "asinh" => asinh(x),
        "atanh" => atanh(x),
        "dlog" => intrinsic_log(x),
        "dlog10" | "intrinsic_log10" => intrinsic_log10(x),
        "intrinsic_exp" => intrinsic_exp(x),
        other => return Err(format!("unknown unary section: {other}")),
    };
    Ok(val)
}

fn eval_binary_f64_sec(sec: &str, a: f64, b: f64) -> Result<f64, String> {
    let val = match sec {
        "pow" => pow(a, b),
        "intrinsic_pow" => intrinsic_pow(a, b),
        "atan2" | "datan2" => atan2(a, b),
        "hypot" => hypot(a, b),
        other => return Err(format!("unknown binary_f64 section: {other}")),
    };
    Ok(val)
}

fn eval_binary_i32_sec(sec: &str, a: f64, exp_i: i32) -> Result<f64, String> {
    let val = match sec {
        "powi" => powi(a, exp_i),
        "powi_f90" => powi_f90(a, exp_i),
        other => return Err(format!("unknown binary_i32 section: {other}")),
    };
    Ok(val)
}

fn parse_u64_hex(tok: &str) -> Result<u64, PyErr> {
    u64::from_str_radix(tok, 16).map_err(|err| {
        PyValueError::new_err(format!("invalid u64 hex `{tok}`: {err}"))
    })
}

#[inline]
fn signed_ulp_diff(strict_val: f64, fast_val: f64) -> u64 {
    let sb = strict_val.to_bits();
    let fb = fast_val.to_bits();
    if sb == fb || (strict_val == 0.0 && fast_val == 0.0) {
        return 0;
    }
    let to_ord = |bits: u64| -> i64 {
        let signed = bits.cast_signed();
        if signed < 0 {
            i64::MIN - signed
        } else {
            signed
        }
    };
    to_ord(sb).abs_diff(to_ord(fb))
}

/// Evaluates all inputs in `vec_path` under `NumericsMode::Strict` and returns
/// `(vector_count, sha256_hex)` over canonicalized `(section, output_u64_be)`
/// records (`#165` Item 15).
#[pyfunction]
pub fn strict_math_vectors_sha256(vec_path: &str) -> PyResult<(usize, String)> {
    let raw = fs::read_to_string(vec_path).map_err(|err| {
        PyValueError::new_err(format!("failed to read `{vec_path}`: {err}"))
    })?;
    with_numerics_mode(NumericsMode::Strict, || {
        math_init();
        let mut stream = Vec::with_capacity(raw.lines().count() * 9);
        let mut count = 0_usize;
        let mut kind = "";
        let mut sec_name = String::new();

        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            if let Some(inner) =
                trimmed.strip_prefix('[').and_then(|s| s.strip_suffix(']'))
            {
                let (k, name) = inner.split_once(':').ok_or_else(|| {
                    PyValueError::new_err(format!("bad header `{trimmed}`"))
                })?;
                kind = k;
                sec_name = name.to_string();
                stream.extend_from_slice(sec_name.as_bytes());
                stream.push(0);
                continue;
            }
            let out_val = match kind {
                "unary" => {
                    let in_tok =
                        trimmed.split_whitespace().next().ok_or_else(|| {
                            PyValueError::new_err("missing unary arg")
                        })?;
                    let x = f64::from_bits(parse_u64_hex(in_tok)?);
                    eval_unary_sec(&sec_name, x)
                        .map_err(PyValueError::new_err)?
                }
                "binary_f64" => {
                    let mut parts = trimmed.split_whitespace();
                    let a_tok = parts
                        .next()
                        .ok_or_else(|| PyValueError::new_err("missing arg1"))?;
                    let b_tok = parts
                        .next()
                        .ok_or_else(|| PyValueError::new_err("missing arg2"))?;
                    let a = f64::from_bits(parse_u64_hex(a_tok)?);
                    let b = f64::from_bits(parse_u64_hex(b_tok)?);
                    eval_binary_f64_sec(&sec_name, a, b)
                        .map_err(PyValueError::new_err)?
                }
                "binary_i32" => {
                    let mut parts = trimmed.split_whitespace();
                    let a_tok = parts
                        .next()
                        .ok_or_else(|| PyValueError::new_err("missing arg1"))?;
                    let n_tok = parts
                        .next()
                        .ok_or_else(|| PyValueError::new_err("missing exp"))?;
                    let a = f64::from_bits(parse_u64_hex(a_tok)?);
                    let n_u32 =
                        u32::from_str_radix(n_tok, 16).map_err(|e| {
                            PyValueError::new_err(format!(
                                "bad u32 `{n_tok}`: {e}"
                            ))
                        })?;
                    eval_binary_i32_sec(&sec_name, a, n_u32.cast_signed())
                        .map_err(PyValueError::new_err)?
                }
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown section kind `{other}`"
                    )));
                }
            };
            stream.extend_from_slice(
                &canonical_strict_bits(out_val).to_be_bytes(),
            );
            count += 1;
        }
        Ok((count, compute_sha256_bytes(&stream)))
    })
}

/// Evaluates all inputs in `vec_path` under `NumericsMode::Fast` and compares
/// them exclusively against `NumericsMode::Strict` (`#165` Item 17; never
/// against MESA), returning `(section, total, finite_pairs,
/// nonfinite_mismatch_count, max_ulp, max_abs, max_rel)` in section order.
#[pyfunction]
#[allow(clippy::too_many_lines)]
pub fn compare_fast_vs_strict_vectors(
    vec_path: &str,
) -> PyResult<Vec<FastVsStrictRow>> {
    let raw = fs::read_to_string(vec_path).map_err(|err| {
        PyValueError::new_err(format!("failed to read `{vec_path}`: {err}"))
    })?;
    math_init();

    let mut order: Vec<String> = Vec::new();
    let mut stats: BTreeMap<String, (usize, usize, usize, u64, f64, f64)> =
        BTreeMap::new();
    let mut kind = "";
    let mut sec_name = String::new();

    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(inner) =
            trimmed.strip_prefix('[').and_then(|s| s.strip_suffix(']'))
        {
            let (k, name) = inner.split_once(':').ok_or_else(|| {
                PyValueError::new_err(format!("bad header `{trimmed}`"))
            })?;
            kind = k;
            sec_name = name.to_string();
            if !stats.contains_key(&sec_name) {
                order.push(sec_name.clone());
                stats.insert(sec_name.clone(), (0, 0, 0, 0, 0.0, 0.0));
            }
            continue;
        }
        let (strict_val, fast_val) = match kind {
            "unary" => {
                let in_tok =
                    trimmed.split_whitespace().next().ok_or_else(|| {
                        PyValueError::new_err("missing unary arg")
                    })?;
                let x = f64::from_bits(parse_u64_hex(in_tok)?);
                let sv = with_numerics_mode(NumericsMode::Strict, || {
                    eval_unary_sec(&sec_name, x)
                })
                .map_err(PyValueError::new_err)?;
                let fv = with_numerics_mode(NumericsMode::Fast, || {
                    eval_unary_sec(&sec_name, x)
                })
                .map_err(PyValueError::new_err)?;
                (sv, fv)
            }
            "binary_f64" => {
                let mut parts = trimmed.split_whitespace();
                let a_tok = parts
                    .next()
                    .ok_or_else(|| PyValueError::new_err("missing arg1"))?;
                let b_tok = parts
                    .next()
                    .ok_or_else(|| PyValueError::new_err("missing arg2"))?;
                let a = f64::from_bits(parse_u64_hex(a_tok)?);
                let b = f64::from_bits(parse_u64_hex(b_tok)?);
                let sv = with_numerics_mode(NumericsMode::Strict, || {
                    eval_binary_f64_sec(&sec_name, a, b)
                })
                .map_err(PyValueError::new_err)?;
                let fv = with_numerics_mode(NumericsMode::Fast, || {
                    eval_binary_f64_sec(&sec_name, a, b)
                })
                .map_err(PyValueError::new_err)?;
                (sv, fv)
            }
            "binary_i32" => {
                let mut parts = trimmed.split_whitespace();
                let a_tok = parts
                    .next()
                    .ok_or_else(|| PyValueError::new_err("missing arg1"))?;
                let n_tok = parts
                    .next()
                    .ok_or_else(|| PyValueError::new_err("missing exp"))?;
                let a = f64::from_bits(parse_u64_hex(a_tok)?);
                let n_u32 = u32::from_str_radix(n_tok, 16).map_err(|e| {
                    PyValueError::new_err(format!("bad u32 `{n_tok}`: {e}"))
                })?;
                let exp_i = n_u32.cast_signed();
                let sv = with_numerics_mode(NumericsMode::Strict, || {
                    eval_binary_i32_sec(&sec_name, a, exp_i)
                })
                .map_err(PyValueError::new_err)?;
                let fv = with_numerics_mode(NumericsMode::Fast, || {
                    eval_binary_i32_sec(&sec_name, a, exp_i)
                })
                .map_err(PyValueError::new_err)?;
                (sv, fv)
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown section kind `{other}`"
                )));
            }
        };
        let entry = stats.get_mut(&sec_name).expect("initialized");
        entry.0 += 1;
        if strict_val.is_finite() && fast_val.is_finite() {
            entry.1 += 1;
            let ulp = signed_ulp_diff(strict_val, fast_val);
            let abs_d = (fast_val - strict_val).abs();
            let scale = strict_val.abs().max(fast_val.abs());
            let rel_d = if scale > 0.0 { abs_d / scale } else { 0.0 };
            if ulp > entry.3 {
                entry.3 = ulp;
            }
            if abs_d > entry.4 {
                entry.4 = abs_d;
            }
            if rel_d > entry.5 {
                entry.5 = rel_d;
            }
        } else {
            let same_nonfinite = (strict_val.is_nan() && fast_val.is_nan())
                || (strict_val.is_infinite()
                    && fast_val.is_infinite()
                    && strict_val.is_sign_negative()
                        == fast_val.is_sign_negative());
            if !same_nonfinite {
                entry.2 += 1;
            }
        }
    }

    Ok(order
        .into_iter()
        .map(|name| {
            let (tot, fin, nf_mis, mulp, mabs, mrel) = stats[&name];
            (name, tot, fin, nf_mis, mulp, mabs, mrel)
        })
        .collect())
}

/// Measures `(strict_exp_ns, fast_exp_ns, strict_fma_ns, fast_fma_ns)` over
/// `n_iters` black-boxed calls (`#165` Item 2).
#[pyfunction]
#[allow(clippy::cast_precision_loss)]
pub fn measure_dispatch_overhead_ns(n_iters: usize) -> (f64, f64, f64, f64) {
    let iters = n_iters.max(1);
    let denom = iters as f64;

    let t_s_exp = with_numerics_mode(NumericsMode::Strict, || {
        let mut acc = 0.0_f64;
        let t0 = Instant::now();
        for idx in 0..iters {
            let arg = black_box(0.5 + (idx & 255) as f64 * 0.001_953_125);
            acc += black_box(exp(arg));
        }
        let dt = t0.elapsed().as_nanos() as f64 / denom;
        black_box(acc);
        dt
    });

    let t_f_exp = with_numerics_mode(NumericsMode::Fast, || {
        let mut acc = 0.0_f64;
        let t0 = Instant::now();
        for idx in 0..iters {
            let arg = black_box(0.5 + (idx & 255) as f64 * 0.001_953_125);
            acc += black_box(exp(arg));
        }
        let dt = t0.elapsed().as_nanos() as f64 / denom;
        black_box(acc);
        dt
    });

    let t_s_fma = with_numerics_mode(NumericsMode::Strict, || {
        let mut acc = 1.0_f64;
        let t0 = Instant::now();
        for idx in 0..iters {
            let arg = black_box(1.0 + (idx & 255) as f64 * 1.0e-7);
            acc = black_box(fma(acc, 0.999_999_9, arg * 1.0e-7));
        }
        let dt = t0.elapsed().as_nanos() as f64 / denom;
        black_box(acc);
        dt
    });

    let t_f_fma = with_numerics_mode(NumericsMode::Fast, || {
        let mut acc = 1.0_f64;
        let t0 = Instant::now();
        for idx in 0..iters {
            let arg = black_box(1.0 + (idx & 255) as f64 * 1.0e-7);
            acc = black_box(fma(acc, 0.999_999_9, arg * 1.0e-7));
        }
        let dt = t0.elapsed().as_nanos() as f64 / denom;
        black_box(acc);
        dt
    });

    (t_s_exp, t_f_exp, t_s_fma, t_f_fma)
}

/// Registers vector-audit and dispatch-timing helpers on `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(strict_math_vectors_sha256, m)?)?;
    m.add_function(wrap_pyfunction!(compare_fast_vs_strict_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(measure_dispatch_overhead_ns, m)?)?;
    Ok(())
}
