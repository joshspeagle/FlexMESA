// SPDX-License-Identifier: LGPL-3.0-or-later
//! Reference-vector `Strict` `SHA-256` hashing, `Fast`-vs-`Strict` comparison,
//! and dispatch-overhead microbenchmark bindings (`#165` Items 2, 15, 17).

use std::collections::BTreeMap;
use std::fs;

use flexmesa_math::vector_eval::{
    canonical_strict_bits, eval_binary_f64_section, eval_binary_i32_section,
    eval_unary_section,
};
use flexmesa_math::{math_init, with_numerics_mode, NumericsMode};
use flexmesa_utils::sha256::compute_sha256_bytes;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type FastVsStrictRow = (String, usize, usize, usize, u64, f64, f64);

#[derive(Clone, Copy)]
enum ParsedVecInput {
    Unary(f64),
    BinaryF64(f64, f64),
    BinaryI32(f64, i32),
}

fn parse_u64_hex(tok: &str) -> Result<u64, String> {
    u64::from_str_radix(tok, 16)
        .map_err(|err| format!("invalid u64 hex `{tok}`: {err}"))
}

fn parse_vector_sections(
    raw: &str,
) -> Result<Vec<(String, Vec<ParsedVecInput>)>, String> {
    let mut sections: Vec<(String, Vec<ParsedVecInput>)> = Vec::new();
    let mut kind = "";

    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(inner) =
            trimmed.strip_prefix('[').and_then(|s| s.strip_suffix(']'))
        {
            let (k, name) = inner
                .split_once(':')
                .ok_or_else(|| format!("bad header `{trimmed}`"))?;
            kind = k;
            sections.push((name.to_string(), Vec::with_capacity(10_300)));
            continue;
        }
        let item = match kind {
            "unary" => {
                let in_tok = trimmed
                    .split_whitespace()
                    .next()
                    .ok_or_else(|| String::from("missing unary arg"))?;
                ParsedVecInput::Unary(f64::from_bits(parse_u64_hex(in_tok)?))
            }
            "binary_f64" => {
                let mut parts = trimmed.split_whitespace();
                let a_tok =
                    parts.next().ok_or_else(|| String::from("missing arg1"))?;
                let b_tok =
                    parts.next().ok_or_else(|| String::from("missing arg2"))?;
                ParsedVecInput::BinaryF64(
                    f64::from_bits(parse_u64_hex(a_tok)?),
                    f64::from_bits(parse_u64_hex(b_tok)?),
                )
            }
            "binary_i32" => {
                let mut parts = trimmed.split_whitespace();
                let a_tok =
                    parts.next().ok_or_else(|| String::from("missing arg1"))?;
                let n_tok =
                    parts.next().ok_or_else(|| String::from("missing exp"))?;
                let n_u32 = u32::from_str_radix(n_tok, 16)
                    .map_err(|e| format!("bad u32 `{n_tok}`: {e}"))?;
                ParsedVecInput::BinaryI32(
                    f64::from_bits(parse_u64_hex(a_tok)?),
                    n_u32.cast_signed(),
                )
            }
            other => return Err(format!("unknown section kind `{other}`")),
        };
        sections
            .last_mut()
            .ok_or_else(|| String::from("vector line before section header"))?
            .1
            .push(item);
    }
    Ok(sections)
}

#[inline]
fn eval_input(sec_name: &str, inp: ParsedVecInput) -> Result<f64, String> {
    match inp {
        ParsedVecInput::Unary(x) => eval_unary_section(sec_name, x),
        ParsedVecInput::BinaryF64(a, b) => {
            eval_binary_f64_section(sec_name, a, b)
        }
        ParsedVecInput::BinaryI32(a, n) => {
            eval_binary_i32_section(sec_name, a, n)
        }
    }
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
pub fn strict_math_vectors_sha256(
    py: Python<'_>,
    vec_path: &str,
) -> PyResult<(usize, String)> {
    let path_owned = vec_path.to_owned();
    py.detach(move || {
        let raw = fs::read_to_string(&path_owned)
            .map_err(|err| format!("failed to read `{path_owned}`: {err}"))?;
        let sections = parse_vector_sections(&raw)?;
        with_numerics_mode(NumericsMode::Strict, || {
            math_init();
            let total_est: usize = sections.iter().map(|(_, v)| v.len()).sum();
            let mut stream = Vec::with_capacity(total_est * 9);
            let mut count = 0_usize;
            for (sec_name, inputs) in &sections {
                stream.extend_from_slice(sec_name.as_bytes());
                stream.push(0);
                for &inp in inputs {
                    let out_val = eval_input(sec_name, inp)?;
                    stream.extend_from_slice(
                        &canonical_strict_bits(out_val).to_be_bytes(),
                    );
                    count += 1;
                }
            }
            Ok::<_, String>((count, compute_sha256_bytes(&stream)))
        })
    })
    .map_err(PyValueError::new_err)
}

/// Evaluates all inputs in `vec_path` under `NumericsMode::Fast` and compares
/// them exclusively against `NumericsMode::Strict` (`#165` Item 17; never
/// against MESA), returning `(section, total, finite_pairs,
/// nonfinite_mismatch_count, max_ulp, max_abs, max_rel)` in section order.
#[pyfunction]
pub fn compare_fast_vs_strict_vectors(
    py: Python<'_>,
    vec_path: &str,
) -> PyResult<Vec<FastVsStrictRow>> {
    let path_owned = vec_path.to_owned();
    py.detach(move || {
        let raw = fs::read_to_string(&path_owned)
            .map_err(|err| format!("failed to read `{path_owned}`: {err}"))?;
        let sections = parse_vector_sections(&raw)?;
        math_init();

        let mut order: Vec<String> = Vec::with_capacity(sections.len());
        let mut stats: BTreeMap<String, (usize, usize, usize, u64, f64, f64)> =
            BTreeMap::new();

        for (sec_name, inputs) in &sections {
            let strict_vals = with_numerics_mode(NumericsMode::Strict, || {
                inputs
                    .iter()
                    .map(|&inp| eval_input(sec_name, inp))
                    .collect::<Result<Vec<f64>, String>>()
            })?;
            let fast_vals = with_numerics_mode(NumericsMode::Fast, || {
                inputs
                    .iter()
                    .map(|&inp| eval_input(sec_name, inp))
                    .collect::<Result<Vec<f64>, String>>()
            })?;

            if !stats.contains_key(sec_name) {
                order.push(sec_name.clone());
                stats.insert(sec_name.clone(), (0, 0, 0, 0, 0.0, 0.0));
            }
            let entry = stats.get_mut(sec_name).expect("initialized");
            for (&strict_val, &fast_val) in strict_vals.iter().zip(&fast_vals) {
                entry.0 += 1;
                if strict_val.is_finite() && fast_val.is_finite() {
                    entry.1 += 1;
                    let ulp = signed_ulp_diff(strict_val, fast_val);
                    let abs_d = (fast_val - strict_val).abs();
                    let scale = strict_val.abs().max(fast_val.abs());
                    let rel_d = if scale > 0.0 { abs_d / scale } else { 0.0 };
                    entry.3 = entry.3.max(ulp);
                    entry.4 = entry.4.max(abs_d);
                    entry.5 = entry.5.max(rel_d);
                } else {
                    let same_nonfinite = (strict_val.is_nan()
                        && fast_val.is_nan())
                        || (strict_val.is_infinite()
                            && fast_val.is_infinite()
                            && strict_val.is_sign_negative()
                                == fast_val.is_sign_negative());
                    if !same_nonfinite {
                        entry.2 += 1;
                    }
                }
            }
        }

        Ok::<_, String>(
            order
                .into_iter()
                .map(|name| {
                    let (tot, fin, nf_mis, mulp, mabs, mrel) = stats[&name];
                    (name, tot, fin, nf_mis, mulp, mabs, mrel)
                })
                .collect(),
        )
    })
    .map_err(PyValueError::new_err)
}

/// Measures `(strict_exp_ns, fast_exp_ns, strict_fma_ns, fast_fma_ns)` over
/// `n_iters` black-boxed calls (`#165` Item 2).
#[pyfunction]
pub fn measure_dispatch_overhead_ns(
    py: Python<'_>,
    n_iters: usize,
) -> (f64, f64, f64, f64) {
    py.detach(move || {
        flexmesa_math::vector_eval::measure_dispatch_overhead_ns(n_iters)
    })
}

/// Registers vector-audit and dispatch-timing helpers on `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(strict_math_vectors_sha256, m)?)?;
    m.add_function(wrap_pyfunction!(compare_fast_vs_strict_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(measure_dispatch_overhead_ns, m)?)?;
    Ok(())
}
