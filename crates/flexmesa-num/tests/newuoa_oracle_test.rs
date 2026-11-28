//! Bit-for-bit replay of MESA `26.4.1` `NEWUOA` oracle runs
//! (`benchmarks/fortran_mesa/newuoa/`, driver `newuoa_oracle.f90` linked
//! against MESA's built `libnum`).
//!
//! Each full run asserts that the whole `CALFUN` sequence (every `X(1:N)` and
//! `F`) and the returned `X` equal MESA's IEEE-754 words. `BIGDEN` is reached
//! by none of the full runs, so `bigden_oracle.f90` calls MESA's compiled
//! `BIGDEN` directly and the last test replays those inputs and outputs.

// The objectives mirror the Fortran loops and integer-to-real conversions.
#![allow(
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap
)]

use flexmesa_num::newuoa::{
    bigden, newuoa, newuoa_with_workspace, NewuoaResult,
};

const CHEBYQUAD_N2: &str = include_str!(
    "../../../benchmarks/fortran_mesa/newuoa/mesa_26_4_1_newuoa_chebyquad_n2.hex"
);
const CHEBYQUAD_N4: &str = include_str!(
    "../../../benchmarks/fortran_mesa/newuoa/mesa_26_4_1_newuoa_chebyquad_n4.hex"
);
const CHEBYQUAD_N6: &str = include_str!(
    "../../../benchmarks/fortran_mesa/newuoa/mesa_26_4_1_newuoa_chebyquad_n6.hex"
);
const CHEBYQUAD_N4_NPT6: &str = include_str!(
    "../../../benchmarks/fortran_mesa/newuoa/mesa_26_4_1_newuoa_chebyquad_n4_npt6.hex"
);
const QUAD_RHOEND1: &str = include_str!(
    "../../../benchmarks/fortran_mesa/newuoa/mesa_26_4_1_newuoa_quad_rhoend1.hex"
);
const QUAD_RHOEND1EM6: &str = include_str!(
    "../../../benchmarks/fortran_mesa/newuoa/mesa_26_4_1_newuoa_quad_rhoend1em6.hex"
);
const BIGDEN_CASES: &str = include_str!(
    "../../../benchmarks/fortran_mesa/newuoa/mesa_26_4_1_newuoa_bigden.hex"
);

fn hex(word: &str) -> u64 {
    u64::from_str_radix(word, 16).expect("hex word")
}

fn hex_f64(word: &str) -> f64 {
    f64::from_bits(hex(word))
}

/// One oracle run: arguments, `CALFUN` records `(X bits, F bits)` and the
/// returned `X` bits.
struct OracleRun {
    n: usize,
    npt: usize,
    maxfun: usize,
    rhobeg: f64,
    rhoend: f64,
    max_valid_value: f64,
    x0: Vec<f64>,
    calls: Vec<(Vec<u64>, u64)>,
    xout: Vec<u64>,
}

fn parse_run(text: &str) -> OracleRun {
    let mut n = 0;
    let mut npt = 0;
    let mut maxfun = 0;
    let mut rho = Vec::new();
    let mut x0 = Vec::new();
    let mut calls = Vec::new();
    let mut xout = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("# N = ") {
            // "# N = 3, NPT = 7, MAXFUN = 1000"
            let parts: Vec<&str> = rest.split(',').collect();
            n = parts[0].trim().parse().unwrap();
            npt = parts[1]
                .trim()
                .trim_start_matches("NPT = ")
                .parse()
                .unwrap();
            maxfun = parts[2]
                .trim()
                .trim_start_matches("MAXFUN = ")
                .parse()
                .unwrap();
        } else if let Some(rest) =
            line.strip_prefix("# RHOBEG RHOEND max_valid_value:")
        {
            rho = rest.split_whitespace().map(hex_f64).collect();
        } else if let Some(rest) = line.strip_prefix("# X0:") {
            x0 = rest.split_whitespace().map(hex_f64).collect();
        } else if let Some(rest) = line.strip_prefix("CALL ") {
            let words: Vec<&str> = rest.split_whitespace().collect();
            let k: usize = words[0].parse().unwrap();
            assert_eq!(k, calls.len() + 1, "CALL records out of order");
            let xs: Vec<u64> =
                words[1..words.len() - 1].iter().map(|w| hex(w)).collect();
            calls.push((xs, hex(words[words.len() - 1])));
        } else if let Some(rest) = line.strip_prefix("XOUT") {
            xout = rest.split_whitespace().map(hex).collect();
        }
    }
    assert!(n > 0 && rho.len() == 3 && x0.len() == n && xout.len() == n);
    OracleRun {
        n,
        npt,
        maxfun,
        rhobeg: rho[0],
        rhoend: rho[1],
        max_valid_value: rho[2],
        x0,
        calls,
        xout,
    }
}

/// Chebyquad (Fletcher 1965) in the arithmetic order of
/// `num/test/src/test_newuoa.f90:44-77`.
fn chebyquad(x: &[f64]) -> f64 {
    let n = x.len();
    // Y(I, J), I = 1..N+1, J = 1..N (1-based, row 0 and column 0 unused).
    let mut y = vec![vec![0.0_f64; n + 1]; n + 2];
    for j in 1..=n {
        y[1][j] = 1.0;
        y[2][j] = 2.0 * x[j - 1] - 1.0;
    }
    for i in 2..=n {
        for j in 1..=n {
            y[i + 1][j] = 2.0 * y[2][j] * y[i][j] - y[i - 1][j];
        }
    }
    let mut f = 0.0_f64;
    let np = n + 1;
    let mut iw = 1_i32;
    for i in 1..=np {
        let mut sum = 0.0_f64;
        for j in 1..=n {
            sum += y[i][j];
        }
        sum /= n as f64;
        if iw > 0 {
            let ii = i as i64;
            sum += 1.0 / ((ii * ii - 2 * ii) as f64);
        }
        iw = -iw;
        f += sum * sum;
    }
    f
}

/// Quadratic of `newuoa_oracle.f90` `quad_f`, in its arithmetic order.
fn quad_f(x: &[f64]) -> f64 {
    let a = x[0] - 0.3;
    let b = x[1] + 0.2;
    let c = x[2] - 0.1;
    a * a + 2.0 * (b * b) + 0.5 * (c * c) + 0.25 * (x[0] * x[1])
        - 0.125 * (x[1] * x[2])
        + 1.5
}

/// Runs [`newuoa_with_workspace`] on an oracle run and asserts the whole `CALFUN` sequence
/// and the returned `X` match MESA bit for bit. Returns the call count.
fn replay(text: &str, objective: fn(&[f64]) -> f64) -> usize {
    let run = parse_run(text);
    let mut x = run.x0.clone();
    let mut k = 0usize;
    // The driver's workspace is `W(10000)` (`newuoa_oracle.f90`).
    let mut w = vec![0.0_f64; 10_000];
    let res: NewuoaResult = newuoa_with_workspace(
        run.n,
        run.npt,
        &mut x,
        run.rhobeg,
        run.rhoend,
        run.maxfun,
        &mut w,
        run.max_valid_value,
        |xs| {
            let f = objective(xs);
            assert!(
                k < run.calls.len(),
                "more CALFUN calls than MESA's {}",
                run.calls.len()
            );
            let (want_x, want_f) = &run.calls[k];
            let got_x: Vec<u64> = xs.iter().map(|v| v.to_bits()).collect();
            assert_eq!(&got_x, want_x, "X of CALFUN call {}", k + 1);
            assert_eq!(f.to_bits(), *want_f, "F of CALFUN call {}", k + 1);
            k += 1;
            Ok(f)
        },
    )
    .expect("newuoa");
    assert_eq!(k, run.calls.len(), "number of CALFUN calls");
    assert_eq!(res.num_evals, run.calls.len(), "NF at the return");
    let got: Vec<u64> = x.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, run.xout, "returned X");
    k
}

#[test]
fn newuoa_chebyquad_n2_npt5_matches_mesa_calfun_sequence_bit_for_bit() {
    assert_eq!(replay(CHEBYQUAD_N2, chebyquad), 31);
}

#[test]
fn newuoa_chebyquad_n4_npt9_matches_mesa_calfun_sequence_bit_for_bit() {
    assert_eq!(replay(CHEBYQUAD_N4, chebyquad), 91);
}

#[test]
fn newuoa_chebyquad_n6_npt13_matches_mesa_calfun_sequence_bit_for_bit() {
    assert_eq!(replay(CHEBYQUAD_N6, chebyquad), 188);
}

#[test]
fn newuoa_chebyquad_n4_npt6_matches_mesa_calfun_sequence_bit_for_bit() {
    assert_eq!(replay(CHEBYQUAD_N4_NPT6, chebyquad), 77);
}

#[test]
fn newuoa_astero_shape_rhoend_1_matches_mesa_calfun_sequence_bit_for_bit() {
    assert_eq!(replay(QUAD_RHOEND1, quad_f), 9);
}

#[test]
fn newuoa_astero_shape_rhoend_1em6_matches_mesa_calfun_sequence_bit_for_bit() {
    assert_eq!(replay(QUAD_RHOEND1EM6, quad_f), 36);
}

/// One `BIGDEN` case: scalar arguments and named input/output arrays.
struct BigdenCase {
    name: String,
    n: usize,
    npt: usize,
    idz: usize,
    kopt: usize,
    knew: usize,
    inputs: Vec<(String, Vec<u64>)>,
    outputs: Vec<(String, Vec<u64>)>,
}

fn parse_bigden(text: &str) -> Vec<BigdenCase> {
    let mut cases = Vec::new();
    let mut cur: Option<BigdenCase> = None;
    for line in text.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.first().copied() {
            Some("CASE") => {
                let int = |i: usize| words[i].parse::<usize>().unwrap();
                cur = Some(BigdenCase {
                    name: words[1].to_string(),
                    n: int(2),
                    npt: int(3),
                    idz: int(4),
                    kopt: int(5),
                    knew: int(6),
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                });
            }
            Some(tag @ ("IN" | "OUT")) => {
                let c = cur.as_mut().expect("record inside a CASE");
                let arr = (
                    words[1].to_string(),
                    words[2..].iter().map(|w| hex(w)).collect(),
                );
                if tag == "IN" {
                    c.inputs.push(arr);
                } else {
                    c.outputs.push(arr);
                }
            }
            Some("END") => cases.push(cur.take().expect("END after CASE")),
            _ => {}
        }
    }
    cases
}

#[test]
fn bigden_matches_mesa_outputs_bit_for_bit_on_unit_cases() {
    let cases = parse_bigden(BIGDEN_CASES);
    assert_eq!(cases.len(), 2);
    for c in &cases {
        let (n, npt) = (c.n, c.npt);
        let ndim = npt + n;
        let nptm = npt - n - 1;
        // One workspace holding every array, column-major as upstream.
        let lens = [
            ("XOPT", n),
            ("XPT", npt * n),
            ("BMAT", ndim * n),
            ("ZMAT", npt * nptm),
            ("D", n),
            ("W", ndim),
            ("VLAG", ndim),
            ("S", n),
            ("WVEC", ndim * 5),
            ("PROD", ndim * 5),
        ];
        let mut off = std::collections::HashMap::new();
        let mut total = 0;
        for (name, len) in lens {
            off.insert(name, (total, len));
            total += len;
        }
        let mut w = vec![0.0_f64; total];
        for (name, vals) in &c.inputs {
            let (o, len) = off[name.as_str()];
            assert_eq!(vals.len(), len, "case {} input {name}", c.name);
            for (i, v) in vals.iter().enumerate() {
                w[o + i] = f64::from_bits(*v);
            }
        }
        let o = |name: &str| off[name].0;
        let beta = bigden(
            n,
            npt,
            &mut w,
            o("XOPT"),
            o("XPT"),
            o("BMAT"),
            o("ZMAT"),
            c.idz,
            ndim,
            c.kopt,
            c.knew,
            o("D"),
            o("W"),
            o("VLAG"),
            o("S"),
            o("WVEC"),
            o("PROD"),
        );
        assert_eq!(c.outputs.len(), 7, "case {}", c.name);
        for (name, want) in &c.outputs {
            let got: Vec<u64> = if name == "BETA" {
                vec![beta.to_bits()]
            } else {
                let (o, len) = off[name.as_str()];
                w[o..o + len].iter().map(|v| v.to_bits()).collect()
            };
            assert_eq!(&got, want, "case {} output {name}", c.name);
        }
    }
}

#[test]
fn newuoa_with_maxfun_below_npt_errors_because_upstream_reads_unset_xopt() {
    // mod_newuoa.f:432-436 returns from the initialization loop and
    // :689-692 then reads XOPT, which :250 has not set (DECISIONS #43).
    let mut x = vec![0.0_f64; 3];
    let mut calls = 0;
    let res = newuoa(3, 7, &mut x, 0.45, 1.0, 4, 1.0e6, |xs| {
        calls += 1;
        Ok(quad_f(xs))
    });
    assert!(res.is_err(), "expected the DECISIONS #43 error");
    assert_eq!(calls, 4);
}

#[test]
fn newuoa_with_workspace_shorter_than_partition_errors_before_any_call() {
    let mut x = vec![0.0_f64; 3];
    let mut w = vec![0.0_f64; 20];
    let res = newuoa_with_workspace(
        3,
        7,
        &mut x,
        0.45,
        1.0,
        1000,
        &mut w,
        1.0e6,
        |_| -> Result<f64, flexmesa_num::NumError> {
            panic!("CALFUN must not be reached")
        },
    );
    assert!(res.is_err(), "expected the DECISIONS #43 error");
}
