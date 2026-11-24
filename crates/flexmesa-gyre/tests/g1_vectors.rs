// SPDX-License-Identifier: GPL-3.0-only
//! Bit-for-bit check of the G1 numerics substrate against MESA 26.4.1
//! reference vectors (`tests/data/mesa_26_4_1_gyre_g1_vectors.tsv`, produced
//! by `benchmarks/fortran_mesa/gyre_g1_vectors/gen_gyre_g1_vectors.f90` with
//! GYRE's flags and the pristine MESA GYRE, ForUM and CRMATH builds).
//!
//! Every real output must match all 64 bits (signed zeros included) and
//! every integer output exactly.

#![allow(
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::cast_sign_loss,
    clippy::doc_markdown
)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use flexmesa_gyre::crmath::{abs_c, atan2, exp_c, hypot, log_c, sqrt_c};
use flexmesa_gyre::ext_complex::{self, ExtC};
use flexmesa_gyre::ext_real::{self, exponent, ExtR};
use flexmesa_gyre::fcomplex::Cplx;
use flexmesa_gyre::forum::{locate, sort_indices};
use flexmesa_gyre::integ::{integral, integrate};
use flexmesa_gyre::math_m::{pow_rc, pow_rr, MathConstants};
use flexmesa_gyre::matmul::{matmul_cc, matmul_rr, matvec_c, matvec_r};
use flexmesa_gyre::powi::{powi_const, powi_var};

struct Rec {
    sec: String,
    id: i64,
    rin: Vec<f64>,
    iin: Vec<i64>,
    rout: Vec<u64>,
    iout: Vec<i64>,
}

fn parse_list<T>(s: &str, f: impl Fn(&str) -> T) -> Vec<T> {
    if s == "-" {
        Vec::new()
    } else {
        s.split(' ').map(f).collect()
    }
}

fn load() -> Vec<Rec> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/mesa_26_4_1_gyre_g1_vectors.tsv");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let hexbits = |h: &str| u64::from_str_radix(h, 16).expect("bad hex");
    let mut recs = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let c: Vec<&str> = line.split('\t').collect();
        assert_eq!(c.len(), 6, "bad record: {line}");
        recs.push(Rec {
            sec: c[0].to_string(),
            id: c[1].parse().expect("id"),
            rin: parse_list(c[2], |h| f64::from_bits(hexbits(h))),
            iin: parse_list(c[3], |h| h.parse().expect("int")),
            rout: parse_list(c[4], hexbits),
            iout: parse_list(c[5], |h| h.parse().expect("int")),
        });
    }
    recs
}

fn cp(v: &[f64]) -> Vec<Cplx> {
    v.chunks(2).map(|p| Cplx::new(p[0], p[1])).collect()
}

fn cbits(z: &[Cplx]) -> Vec<f64> {
    z.iter().flat_map(|z| [z.re, z.im]).collect()
}

fn xr(a: f64, de: i64) -> ExtR {
    ExtR::from_real(a).scale(de as i32)
}

fn xc(z: Cplx, de: i64) -> ExtC {
    ExtC::from_cplx(z).scale(de as i32)
}

fn b(v: bool) -> i64 {
    i64::from(v)
}

/// Evaluate one record; returns (real outputs, int outputs).
fn eval(r: &Rec) -> (Vec<f64>, Vec<i64>) {
    let x = &r.rin;
    let i = &r.iin;
    let none: Vec<i64> = Vec::new();
    match r.sec.as_str() {
        "cmul" => (cbits(&[cp(&x[0..2])[0] * cp(&x[2..4])[0]]), none),
        "cdiv" => (cbits(&[cp(&x[0..2])[0] / cp(&x[2..4])[0]]), none),
        "cmulr" => (cbits(&[Cplx::new(x[0], x[1]).mul_real(x[2])]), none),
        "rmulc" => {
            (cbits(&[Cplx::real_mul(x[0], Cplx::new(x[1], x[2]))]), none)
        }
        "cdivr" => (cbits(&[Cplx::new(x[0], x[1]).div_real(x[2])]), none),
        "rdivc" => {
            (cbits(&[Cplx::real_div(x[0], Cplx::new(x[1], x[2]))]), none)
        }
        "cpow2" => (cbits(&[Cplx::new(x[0], x[1]).powi2()]), none),
        "cpowm1" => (cbits(&[Cplx::new(x[0], x[1]).powi_m1()]), none),
        "rpow_const" => (
            [2, 3, 4, 5, 6, 7, 8, -1, -2, -3]
                .iter()
                .map(|&n| powi_const(x[0], n).expect("powi_const"))
                .collect(),
            none,
        ),
        "rpow_var" => (vec![powi_var(x[0], i[0] as i32)], none),
        "abs_c" => (vec![abs_c(Cplx::new(x[0], x[1]))], none),
        "sqrt_c" => (cbits(&[sqrt_c(Cplx::new(x[0], x[1]))]), none),
        "exp_c" => (cbits(&[exp_c(Cplx::new(x[0], x[1]))]), none),
        "log_c" => {
            (cbits(&[log_c(Cplx::new(x[0], x[1])).expect("log_c")]), none)
        }
        "atan2" => (vec![atan2(x[0], x[1]).expect("atan2")], none),
        "hypot" => (vec![hypot(x[0], x[1])], none),
        "exp_r" => (vec![flexmesa_math::exp(x[0])], none),
        "log_r" => (vec![flexmesa_math::log(x[0])], none),
        "acos_r" => (vec![flexmesa_math::acos(x[0])], none),
        "pow_rr" => (vec![pow_rr(x[0], x[1])], none),
        "pow_rc" => (cbits(&[pow_rc(x[0], Cplx::new(x[1], x[2]))]), none),
        "math_m_constants" => {
            let c = MathConstants::init();
            (
                vec![
                    c.pi,
                    c.twopi,
                    c.halfpi,
                    c.golden_r,
                    c.deg_to_rad,
                    c.rad_to_deg,
                ],
                none,
            )
        }
        "matmul_rr" => {
            let (m, k, n) = (i[0] as usize, i[1] as usize, i[2] as usize);
            (
                matmul_rr(&x[..m * k], &x[m * k..], m, k, n).expect("matmul"),
                none,
            )
        }
        "matvec_r" => {
            let (m, k) = (i[0] as usize, i[1] as usize);
            (matvec_r(&x[..m * k], &x[m * k..], m, k), none)
        }
        "matmul_cc" => {
            let (m, k, n) = (i[0] as usize, i[1] as usize, i[2] as usize);
            let z = cp(x);
            (
                cbits(
                    &matmul_cc(&z[..m * k], &z[m * k..], m, k, n)
                        .expect("matmul"),
                ),
                none,
            )
        }
        "matvec_c" => {
            let (m, k) = (i[0] as usize, i[1] as usize);
            let z = cp(x);
            (cbits(&matvec_c(&z[..m * k], &z[m * k..], m, k)), none)
        }
        "matmul_nested_r" => {
            let n = i[0] as usize;
            let nn = n * n;
            let (g, xa, h, dh) =
                (&x[..nn], &x[nn..2 * nn], &x[2 * nn..3 * nn], &x[3 * nn..]);
            let t = matmul_rr(xa, h, n, n, n).expect("matmul");
            let t2: Vec<f64> = t.iter().zip(dh).map(|(a, b)| a - b).collect();
            (matmul_rr(g, &t2, n, n, n).expect("matmul"), none)
        }
        "commutator_c" => {
            let n = i[0] as usize;
            let z = cp(x);
            let (a, bm) = (&z[..n * n], &z[n * n..]);
            let ab = matmul_cc(a, bm, n, n, n).expect("matmul");
            let ba = matmul_cc(bm, a, n, n, n).expect("matmul");
            (
                cbits(
                    &ab.iter()
                        .zip(&ba)
                        .map(|(p, q)| *p - *q)
                        .collect::<Vec<_>>(),
                ),
                none,
            )
        }
        // ext_rt
        "xr_ctor" => {
            let v = xr(x[0], i[0]);
            (vec![v.f], vec![i64::from(v.e)])
        }
        "xr_add" | "xr_sub" | "xr_mul" | "xr_div" => {
            let (p, q) = (xr(x[0], i[0]), xr(x[1], i[1]));
            let v = match r.sec.as_str() {
                "xr_add" => p.plus(q),
                "xr_sub" => p.minus(q),
                "xr_mul" => p.times(q),
                _ => p.divide(q),
            };
            (vec![v.f], vec![i64::from(v.e)])
        }
        "xr_neg" => {
            let v = xr(x[0], i[0]).negate();
            (vec![v.f], vec![i64::from(v.e)])
        }
        "xr_cmp" => {
            let (p, q) = (xr(x[0], i[0]), xr(x[1], i[1]));
            (
                vec![],
                vec![
                    b(p.ext_eq(q)),
                    b(!p.ext_eq(q)),
                    b(p.lt(q)),
                    b(p.gt(q)),
                    b(p.le(q)),
                    b(p.ge(q)),
                ],
            )
        }
        "xr_abs" => {
            let v = xr(x[0], i[0]).abs();
            (vec![v.f], vec![i64::from(v.e)])
        }
        "xr_max" => {
            let (p, q) = (xr(x[0], i[0]), xr(x[1], i[1]));
            let (mx, mn) = (ExtR::max(p, q), ExtR::min(p, q));
            (vec![mx.f, mn.f], vec![i64::from(mx.e), i64::from(mn.e)])
        }
        "xr_valid" => (vec![], vec![b(xr(x[0], i[0]).valid())]),
        "xr_sqrt" => {
            let v = xr(x[0], i[0]).sqrt();
            (vec![v.f], vec![i64::from(v.e)])
        }
        "xr_mixed" => {
            let p = xr(x[0], i[0]);
            let s = x[1];
            let es = ExtR::from_real(s);
            let ops = [
                p.add_r(s),
                es.plus(p),
                p.sub_r(s),
                es.minus(p),
                p.mul_r(s),
                es.times(p),
                p.div_r(s),
            ];
            let mut ints: Vec<i64> =
                ops.iter().map(|v| i64::from(v.e)).collect();
            ints.extend([
                b(p.ext_eq(es)),
                b(p.lt(es)),
                b(p.gt(es)),
                b(p.le(es)),
                b(p.ge(es)),
                b(es.lt(p)),
            ]);
            (ops.iter().map(|v| v.f).collect(), ints)
        }
        "xr_real" => {
            let a = x[0];
            (
                vec![ExtR::from_real(a)
                    .scale(i[0] as i32 - exponent(a))
                    .to_real()],
                none,
            )
        }
        "xr_exp" => {
            let v = ExtR::from_real(x[0]).exp();
            (vec![v.f], vec![i64::from(v.e)])
        }
        "xr_from_c" => {
            let v = ExtR::from_real(x[0]);
            (vec![v.f], vec![i64::from(v.e)])
        }
        "xr_scale" => {
            let v = xr(x[0], i[0]).scale(i[1] as i32);
            (vec![v.f], vec![i64::from(v.e)])
        }
        "xr_product" => {
            let a: Vec<ExtR> =
                x.iter().zip(i).map(|(&a, &d)| xr(a, d)).collect();
            let v = ext_real::product(&a);
            (vec![v.f], vec![i64::from(v.e)])
        }
        // ext_ct
        "xc_ctor" => {
            let v = xc(Cplx::new(x[0], x[1]), i[0]);
            (vec![v.f.re, v.f.im], vec![i64::from(v.e)])
        }
        "xc_add" | "xc_sub" | "xc_mul" | "xc_div" => {
            let p = xc(Cplx::new(x[0], x[1]), i[0]);
            let q = xc(Cplx::new(x[2], x[3]), i[1]);
            let v = match r.sec.as_str() {
                "xc_add" => p.plus(q),
                "xc_sub" => p.minus(q),
                "xc_mul" => p.times(q),
                _ => p.divide(q),
            };
            (vec![v.f.re, v.f.im], vec![i64::from(v.e)])
        }
        "xc_neg" => {
            let p = xc(Cplx::new(x[0], x[1]), i[0]);
            let (n, c) = (p.negate(), p.conj());
            (
                vec![n.f.re, n.f.im, c.f.re, c.f.im],
                vec![i64::from(n.e), i64::from(c.e)],
            )
        }
        "xc_cmp" => {
            let p = xc(Cplx::new(x[0], x[1]), i[0]);
            let q = xc(Cplx::new(x[2], x[3]), i[1]);
            (vec![], vec![b(p.ext_eq(q)), b(!p.ext_eq(q))])
        }
        "xc_abs" => {
            let v = xc(Cplx::new(x[0], x[1]), i[0]).abs();
            (vec![v.f], vec![i64::from(v.e)])
        }
        "xc_valid" => {
            (vec![], vec![b(xc(Cplx::new(x[0], x[1]), i[0]).valid())])
        }
        "xc_sqrt" => {
            let v = xc(Cplx::new(x[0], x[1]), i[0]).sqrt();
            (vec![v.f.re, v.f.im], vec![i64::from(v.e)])
        }
        "xc_parts" => {
            let p = xc(Cplx::new(x[0], x[1]), i[0]);
            let (re, im, rr) = (p.real_part(), p.imag_part(), p.to_extr());
            (
                vec![re.f, im.f, rr.f],
                vec![i64::from(re.e), i64::from(im.e), i64::from(rr.e)],
            )
        }
        "xc_mixed" => {
            let p = xc(Cplx::new(x[0], x[1]), i[0]);
            let w = Cplx::new(x[2], x[3]);
            let a = x[4];
            let rx = ExtC::from_extr(xr(a, i[1]));
            let wx = ExtC::from_cplx(w);
            let ax = ExtC::from_real(a);
            let ops = [
                p.plus(rx),
                rx.plus(p),
                p.minus(rx),
                rx.minus(p),
                p.times(rx),
                rx.times(p),
                p.divide(rx),
                p.plus(wx),
                wx.plus(p),
                p.minus(wx),
                wx.minus(p),
                p.times(wx),
                wx.times(p),
                p.divide(wx),
                p.plus(ax),
                ax.plus(p),
                p.minus(ax),
                ax.minus(p),
                p.times(ax),
                ax.times(p),
                p.divide(ax),
            ];
            let mut ints: Vec<i64> =
                ops.iter().map(|v| i64::from(v.e)).collect();
            ints.extend([b(p.ext_eq(rx)), b(p.ext_eq(wx)), b(p.ext_eq(ax))]);
            (ops.iter().flat_map(|v| [v.f.re, v.f.im]).collect(), ints)
        }
        "xc_ctor2" => {
            let (a, bb) = (x[0], x[1]);
            let rx = xr(a, i[0]);
            let vs = [
                ExtC::from_real(a),
                ExtC::from_parts(a, bb),
                ExtC::from_extr(rx),
                ExtC::from_extr_parts(rx, xr(bb, i[1])),
            ];
            (
                vs.iter().flat_map(|v| [v.f.re, v.f.im]).collect(),
                vs.iter().map(|v| i64::from(v.e)).collect(),
            )
        }
        "xc_cmplx" => {
            let z = Cplx::new(x[0], x[1]);
            let e0 = ExtC::from_cplx(z).e;
            let v = xc(z, i[0] - i64::from(e0));
            let c = v.to_cplx();
            (vec![c.re, c.im, v.to_real()], none)
        }
        "xc_exp" => {
            let v = ExtC::from_cplx(Cplx::new(x[0], x[1])).exp();
            (vec![v.f.re, v.f.im], vec![i64::from(v.e)])
        }
        "xc_scale" => {
            let v = xc(Cplx::new(x[0], x[1]), i[0]).scale(i[1] as i32);
            (vec![v.f.re, v.f.im], vec![i64::from(v.e)])
        }
        "xc_product" => {
            let z = cp(x);
            let a: Vec<ExtC> =
                z.iter().zip(i).map(|(&z, &d)| xc(z, d)).collect();
            let v = ext_complex::product(&a);
            (vec![v.f.re, v.f.im], vec![i64::from(v.e)])
        }
        // integ_m
        "integrate" => {
            let n = i[0] as usize;
            (vec![integrate(&x[..n], &x[n..], None)], none)
        }
        "integrate_mask" => {
            let n = i[0] as usize;
            let mask: Vec<bool> = i[1..].iter().map(|&m| m == 1).collect();
            (vec![integrate(&x[..n], &x[n..], Some(&mask))], none)
        }
        "integral" => {
            let n = i[0] as usize;
            (integral(&x[..n], &x[n..]), none)
        }
        // ForUM
        "sort_indices" => (
            vec![],
            sort_indices(x, i[1] == 1)
                .iter()
                .map(|&v| i64::from(v))
                .collect(),
        ),
        "locate" => {
            let n = i[0] as usize;
            let (xs, q) = (&x[..n], x[n]);
            let mut il = i[1] as i32;
            let mut ir = i[1] as i32;
            locate(xs, q, &mut il, false);
            locate(xs, q, &mut ir, true);
            (vec![], vec![i64::from(il), i64::from(ir)])
        }
        other => panic!("unknown section {other}"),
    }
}

#[test]
fn g1_substrate_matches_mesa_26_4_1_bit_for_bit() {
    flexmesa_math::math_init();
    let recs = load();
    let mut per: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut first: Vec<String> = Vec::new();
    for r in &recs {
        let (ro, io) = eval(r);
        let ok_r = ro.len() == r.rout.len()
            && ro.iter().zip(&r.rout).all(|(a, e)| a.to_bits() == *e);
        let ok_i = io == r.iout;
        let ent = per.entry(r.sec.clone()).or_insert((0, 0));
        ent.0 += 1;
        if !(ok_r && ok_i) {
            ent.1 += 1;
            if first.len() < 25 {
                first.push(format!(
                    "{} {}: got r={:?} i={:?}; want r={:?} i={:?}",
                    r.sec,
                    r.id,
                    ro.iter()
                        .map(|v| format!("{:016X}", v.to_bits()))
                        .collect::<Vec<_>>(),
                    io,
                    r.rout
                        .iter()
                        .map(|v| format!("{v:016X}"))
                        .collect::<Vec<_>>(),
                    r.iout
                ));
            }
        }
    }
    let bad: usize = per.values().map(|v| v.1).sum();
    let summary: Vec<String> = per
        .iter()
        .map(|(k, (n, m))| format!("{k}: {n} cases, {m} mismatched"))
        .collect();
    assert!(
        bad == 0,
        "{bad} mismatched records\n{}\nfirst mismatches:\n{}",
        summary.join("\n"),
        first.join("\n")
    );
    assert!(recs.len() > 10_000, "too few records: {}", recs.len());
}

/// Count records of `sec` whose real outputs an alternative implementation
/// fails to reproduce bit for bit.
fn count_mismatches(
    recs: &[Rec],
    sec: &str,
    alt: impl Fn(&Rec) -> Vec<f64>,
) -> (usize, usize) {
    let mut n = 0;
    let mut bad = 0;
    for r in recs.iter().filter(|r| r.sec == sec) {
        n += 1;
        let got = alt(r);
        if got.len() != r.rout.len()
            || got.iter().zip(&r.rout).any(|(a, e)| a.to_bits() != *e)
        {
            bad += 1;
        }
    }
    (n, bad)
}

/// The vectors discriminate between the expansions under test and the
/// obvious alternatives, so passing them is evidence for the specific
/// compiler behaviour, not just for numerical closeness.
#[test]
fn negative_controls_show_the_vectors_discriminate() {
    flexmesa_math::math_init();
    let recs = load();

    // Division: textbook (ac+bd)/(c^2+d^2) instead of Smith's method
    let (n, bad) = count_mismatches(&recs, "cdiv", |r| {
        let (a, bb, c, d) = (r.rin[0], r.rin[1], r.rin[2], r.rin[3]);
        let den = c * c + d * d;
        vec![(a * c + bb * d) / den, (bb * c - a * d) / den]
    });
    eprintln!("textbook division differs on {bad}/{n} cdiv cases");
    assert!(
        bad > n / 4,
        "textbook division matched too often: {bad}/{n}"
    );

    // Constant powers: binary squaring (__powidf2) instead of GCC's chains
    let (n, bad) = count_mismatches(&recs, "rpow_const", |r| {
        [2, 3, 4, 5, 6, 7, 8, -1, -2, -3]
            .iter()
            .map(|&k| powi_var(r.rin[0], k))
            .collect()
    });
    eprintln!("__powidf2 differs on {bad}/{n} rpow_const cases");
    assert!(
        bad > n / 10,
        "__powidf2 matched the constant chains too often: {bad}/{n}"
    );

    // MATMUL: reversed summation order over the inner index
    let (n, bad) = count_mismatches(&recs, "matmul_rr", |r| {
        let (m, k, nn) =
            (r.iin[0] as usize, r.iin[1] as usize, r.iin[2] as usize);
        let (a, bm) = (&r.rin[..m * k], &r.rin[m * k..]);
        let mut c = vec![0.0_f64; m * nn];
        for j in 0..nn {
            for i in 0..m {
                let mut s = 0.0;
                for l in (0..k).rev() {
                    s += a[i + l * m] * bm[l + j * k];
                }
                c[i + j * m] = s;
            }
        }
        c
    });
    eprintln!("reversed MATMUL order differs on {bad}/{n} matmul_rr cases");
    assert!(
        bad > n / 4,
        "reversed MATMUL order matched too often: {bad}/{n}"
    );

    // atan2: the intrinsic-style flexmesa_math::atan2 instead of CRMATH's
    let (n, bad) = count_mismatches(&recs, "atan2", |r| {
        vec![flexmesa_math::atan2(r.rin[0], r.rin[1])]
    });
    eprintln!("intrinsic-style atan2 differs on {bad}/{n} atan2 cases");
    assert!(
        bad > 0,
        "intrinsic atan2 matched CRMATH atan2 on all {n} cases"
    );

    // Complex times real: componentwise (r*zr, r*zi) instead of the widened
    // product; this only changes signed zeros, so it is reported, not asserted.
    let (n, bad) = count_mismatches(&recs, "cmulr", |r| {
        vec![r.rin[0] * r.rin[2], r.rin[1] * r.rin[2]]
    });
    eprintln!("componentwise complex*real differs on {bad}/{n} cmulr cases");
}
