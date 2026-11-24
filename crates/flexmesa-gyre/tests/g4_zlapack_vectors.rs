// SPDX-License-Identifier: GPL-3.0-only
//! Bit-for-bit check of the complex LAPACK/BLAS port against the MESA SDK
//! `x86_64-linux-26.6.1` crmath-slot libraries
//! (`tests/data/mesa_26_4_1_gyre_g4_vectors.tsv`, produced by
//! `benchmarks/fortran_mesa/gyre_g4_vectors/gen_gyre_g4_vectors.f90`).
//!
//! Inputs are GYRE's own complex LAPACK calls and blocks from the G0 oracle
//! runs (`*_tap`: the `ZGESVD` boundary matrices, also factored with
//! `ZGETRF` as `det_c_` does, and GYRE's cyclic-reduction and
//! back-substitution kernels on tapped blocks) and synthetic edge cases
//! (`*_syn` and the leaf routines). The tapped inputs, and GYRE's own
//! `ZGESVD` outputs for `gesvd_tap`, are read from the committed G0 taps
//! (`benchmarks/fortran_mesa/gyre_oracles/data/*.taps.tsv`) rather than
//! stored twice; [`tapped`] numbers them as `extract_tapped_inputs.py`
//! does. Every real output and both parts of
//! every complex output must match all 64 bits, and every integer output
//! exactly.

#![allow(
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::similar_names
)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use flexmesa_gyre::fcomplex::Cplx;
use flexmesa_gyre::lapack::zaux::{
    dladiv, dlapy3, zladiv, zlange, zlarfg, zlaswp,
};
use flexmesa_gyre::lapack::zblas::{dznrm2, izamax, zgemm, zgemv, ztrsm};
use flexmesa_gyre::lapack::zlu::zgetrf;
use flexmesa_gyre::lapack::zsvd::{zbdsqr, zgesvd};
use flexmesa_gyre::lapack::LapackError;

const ONE: Cplx = Cplx::new(1.0, 0.0);
const ZERO: Cplx = Cplx::new(0.0, 0.0);

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

/// The tapped inputs of `extract_tapped_inputs.py`, by record id:
/// `(kind, n_e or rows, values as reals)`, and GYRE's tapped `ZGESVD`
/// outputs `(info, S, U, VH)` as bit patterns.
struct Tapped {
    inputs: BTreeMap<i64, (&'static str, usize, usize, Vec<f64>)>,
    svd_out: BTreeMap<i64, (i64, Vec<u64>)>,
}

fn tapped() -> Tapped {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/fortran_mesa/gyre_oracles/data");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".taps.tsv"))
        })
        .collect();
    paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    let hexf =
        |h: &str| f64::from_bits(u64::from_str_radix(h, 16).expect("bad hex"));
    // `[rows x cols,colmajor,reim]` of a complex tap name.
    let shape = |name: &str| -> Option<(usize, usize)> {
        let t = name.strip_suffix(",colmajor,reim]")?;
        let (_, dims) = t.rsplit_once('[')?;
        let (r, c) = dims.split_once('x')?;
        Some((r.parse().ok()?, c.parse().ok()?))
    };
    let mut out = Tapped {
        inputs: BTreeMap::new(),
        svd_out: BTreeMap::new(),
    };
    let mut ident = 0i64;
    for path in paths {
        let text = std::fs::read_to_string(&path).expect("read taps");
        let mut blocks: Vec<Vec<f64>> = Vec::new();
        let mut svd_id = 0i64;
        let mut pending: BTreeMap<String, String> = BTreeMap::new();
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let f: Vec<&str> = line.split('\t').collect();
            let (kind, name) = (f[0], f[1]);
            let vals = || f[3].split(' ').map(hexf).collect::<Vec<f64>>();
            if kind == "BUILD" && name == "omega" {
                blocks.clear();
            } else if kind == "BLOCK"
                && (name.contains(":E_l[") || name.contains(":E_r["))
            {
                let Some((ne, _)) = shape(name) else { continue };
                if blocks.len() < 4 {
                    blocks.push(vals());
                }
                if blocks.len() == 4 {
                    ident += 1;
                    let all: Vec<f64> = blocks.concat();
                    out.inputs.insert(ident, ("FACTOR", ne, ne, all));
                    ident += 1;
                    let pair: Vec<f64> =
                        [&blocks[0][..], &blocks[1][..]].concat();
                    out.inputs.insert(ident, ("GEMV", ne, ne, pair));
                    blocks.push(Vec::new()); // one record pair per BUILD
                }
            } else if kind == "LAPACK_IN" && name.starts_with("ZGESVD:A") {
                let (r, c) = shape(name).expect("ZGESVD:A shape");
                ident += 1;
                svd_id = ident;
                out.inputs.insert(ident, ("GESVD", r, c, vals()));
                pending.clear();
            } else if kind == "LAPACK_OUT" && name.starts_with("ZGESVD:") {
                let key = name[7..].split('[').next().expect("key").to_string();
                pending.insert(key, f[3].to_string());
                if pending.len() == 4 {
                    let bits = |k: &str| -> Vec<u64> {
                        pending[k]
                            .split(' ')
                            .map(|h| u64::from_str_radix(h, 16).expect("hex"))
                            .collect()
                    };
                    let mut v = bits("S");
                    v.extend(bits("U"));
                    v.extend(bits("VH"));
                    let info: i64 = pending["info"].parse().expect("info");
                    out.svd_out.insert(svd_id, (info, v));
                    pending.clear();
                }
            }
        }
    }
    out
}

/// The `S`, `U`, `VT` slice of a [`gesvd_out`] output for an `n x n` call.
fn svd_slice(v: &[f64], n: usize) -> Vec<f64> {
    let q = 2 * n * n;
    v[q..3 * q + n].to_vec()
}

fn load() -> Vec<Rec> {
    let mut recs = load_file();
    let tp = tapped();
    for r in &mut recs {
        let src = |kind: &str| -> &Vec<f64> {
            let (k, _, _, v) = tp
                .inputs
                .get(&r.id)
                .unwrap_or_else(|| panic!("no tapped input {}", r.id));
            assert_eq!(*k, kind, "{} {} tapped kind", r.sec, r.id);
            v
        };
        match r.sec.as_str() {
            "getrf_det_tap" => r.rin = src("GESVD").clone(),
            "factor_tap" => r.rin = src("FACTOR").clone(),
            "gemv_tap" => {
                let mut v = src("GEMV").clone();
                v.extend(&r.rin);
                r.rin = v;
            }
            _ => {}
        }
    }
    // gesvd_tap: GYRE's own tapped calls, compared on S, U, VH and INFO.
    for (id, (info, bits)) in &tp.svd_out {
        let (_, m, n, a) = &tp.inputs[id];
        recs.push(Rec {
            sec: "gesvd_tap".into(),
            id: *id,
            rin: a.clone(),
            iin: vec![*m as i64, *n as i64],
            rout: bits.clone(),
            iout: vec![*info],
        });
    }
    recs
}

fn load_file() -> Vec<Rec> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/mesa_26_4_1_gyre_g4_vectors.tsv");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let hexbits = |h: &str| u64::from_str_radix(h, 16).expect("bad hex");
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|line| {
            let c: Vec<&str> = line.split('\t').collect();
            assert_eq!(c.len(), 6, "bad record: {line}");
            Rec {
                sec: c[0].to_string(),
                id: c[1].parse().expect("id"),
                rin: parse_list(c[2], |h| f64::from_bits(hexbits(h))),
                iin: parse_list(c[3], |h| h.parse().expect("int")),
                rout: parse_list(c[4], hexbits),
                iout: parse_list(c[5], |h| h.parse().expect("int")),
            }
        })
        .collect()
}

/// (re, im) pairs to complex values.
fn cx(v: &[f64]) -> Vec<Cplx> {
    v.as_chunks::<2>()
        .0
        .iter()
        .map(|p| Cplx::new(p[0], p[1]))
        .collect()
}

/// Complex values to (re, im) pairs.
fn flat(v: &[Cplx]) -> Vec<f64> {
    v.iter().flat_map(|z| [z.re, z.im]).collect()
}

/// Sequential reader of a record's complex inputs.
struct Take<'a> {
    v: &'a [Cplx],
    p: usize,
}

impl Take<'_> {
    fn next(&mut self, n: usize) -> Vec<Cplx> {
        let out = self.v[self.p..self.p + n].to_vec();
        self.p += n;
        out
    }
}

fn ipiv_ints(info: usize, ipiv: &[usize]) -> Vec<i64> {
    let mut ints = vec![info as i64];
    ints.extend(ipiv.iter().map(|&p| p as i64));
    ints
}

/// `(M_U, M_E, M_G, info, ipiv)` of [`factor_step`].
type Factored = (Vec<Cplx>, Vec<Cplx>, Vec<Cplx>, usize, Vec<usize>);

/// GYRE's cyclic-reduction step (`block_sysmtx_m.fypp` `factor_c_`):
/// returns `(M_U, M_E, M_G, info, ipiv)`.
fn factor_step(
    ne: usize,
    el1: &[Cplx],
    er1: &[Cplx],
    el2: &[Cplx],
    er2: &[Cplx],
) -> Result<Factored, LapackError> {
    let ld = 2 * ne;
    let mut m_g = vec![ZERO; ld * ne];
    let mut m_u = vec![ZERO; ld * ne];
    let mut m_e = vec![ZERO; ld * ne];
    for c in 0..ne {
        for r in 0..ne {
            m_g[r + c * ld] = el1[r + c * ne];
            m_u[r + c * ld] = er1[r + c * ne];
            m_u[ne + r + c * ld] = el2[r + c * ne];
            m_e[ne + r + c * ld] = er2[r + c * ne];
        }
    }
    let mut ipiv = vec![0usize; ne];
    let info = zgetrf(ld, ne, &mut m_u, ld, &mut ipiv)?;
    for mm in [&mut m_e, &mut m_g] {
        zlaswp(ne, mm, ld, 1, ne, &ipiv, 1);
        ztrsm(b'L', b'L', b'N', b'U', ne, ne, ONE, &m_u, ld, mm, ld)?;
        let top = mm.clone();
        zgemm(
            b'N',
            b'N',
            ne,
            ne,
            ne,
            -ONE,
            &m_u[ne..],
            ld,
            &top,
            ld,
            ONE,
            &mut mm[ne..],
            ld,
        )?;
        ztrsm(b'L', b'U', b'N', b'N', ne, ne, ONE, &m_u, ld, mm, ld)?;
    }
    Ok((m_u, m_e, m_g, info, ipiv))
}

fn gesvd_out(
    m: usize,
    n: usize,
    a0: &[Cplx],
    tolmul: Option<f64>,
) -> (Vec<f64>, Vec<i64>) {
    let mut a = a0.to_vec();
    let mut u = vec![ZERO; m * m];
    let mut vt = vec![ZERO; n * n];
    let lwork = 2 * n + m;
    let o = match tolmul {
        None => zgesvd(m, n, &mut a, &mut u, &mut vt, lwork),
        Some(t) => flexmesa_gyre::lapack::zsvd::zgesvd_with_tolmul(
            m, n, &mut a, &mut u, &mut vt, lwork, t,
        ),
    }
    .expect("zgesvd");
    let mut v = flat(&a);
    v.extend(&o.s);
    v.extend(flat(&u));
    v.extend(flat(&vt));
    v.extend([o.work1, 0.0]);
    (v, vec![o.info as i64])
}

fn eval(r: &Rec) -> (Vec<f64>, Vec<i64>) {
    let x = &r.rin;
    let iv: Vec<usize> = r.iin.iter().map(|&i| i as usize).collect();
    let z = cx(x);
    let mut t = Take { v: &z, p: 0 };
    match r.sec.as_str() {
        "dznrm2" => (vec![dznrm2(iv[0], &z, 1)], vec![]),
        "izamax" => (vec![], vec![izamax(iv[0], &z, 1) as i64]),
        "dlapy3" => (vec![dlapy3(x[0], x[1], x[2])], vec![]),
        "dladiv" => {
            let (p, q) = dladiv(x[0], x[1], x[2], x[3]);
            (vec![p, q], vec![])
        }
        "zladiv" => (
            flat(&[zladiv(Cplx::new(x[0], x[1]), Cplx::new(x[2], x[3]))]),
            vec![],
        ),
        "zlarfg" => {
            let n = iv[0];
            let mut alpha = z[0];
            let mut v = z[1..].to_vec();
            let tau = zlarfg(n, &mut alpha, &mut v, 1);
            let mut out = vec![alpha, tau];
            out.extend(v);
            (flat(&out), vec![])
        }
        "zlange" => {
            let (m, n, nm) = (iv[0], iv[1], iv[2] as u8);
            (vec![zlange(nm, m, n, &z, m).expect("zlange")], vec![])
        }
        "zgemv" => {
            let (tr, m, n, incx, incy) =
                (iv[0] as u8, iv[1], iv[2], iv[3], iv[4]);
            let (lenx, leny) = if tr == b'N' { (n, m) } else { (m, n) };
            let ab = t.next(2);
            let a = t.next(m * n);
            let xv = t.next(lenx * incx);
            let mut y = t.next(leny * incy);
            zgemv(tr, m, n, ab[0], &a, m, &xv, incx, ab[1], &mut y, incy)
                .expect("zgemv");
            (flat(&y), vec![])
        }
        "zgemm" => {
            let (m, n, k) = (iv[0], iv[1], iv[2]);
            let ab = t.next(2);
            let a = t.next(m * k.max(1));
            let b = t.next(k.max(1) * n);
            let mut c = t.next(m * n);
            zgemm(
                b'N',
                b'N',
                m,
                n,
                k,
                ab[0],
                &a,
                m,
                &b,
                k.max(1),
                ab[1],
                &mut c,
                m,
            )
            .expect("zgemm");
            (flat(&c), vec![])
        }
        "ztrsm" => {
            let (ul, tr, dg, m, n) =
                (iv[0] as u8, iv[1] as u8, iv[2] as u8, iv[3], iv[4]);
            let alpha = t.next(1)[0];
            let a = t.next(m * m);
            let mut b = t.next(m * n);
            ztrsm(b'L', ul, tr, dg, m, n, alpha, &a, m, &mut b, m)
                .expect("ztrsm");
            (flat(&b), vec![])
        }
        "zbdsqr" => {
            let (ul, n, ncv, nru) = (iv[0] as u8, iv[1], iv[2], iv[3]);
            let mut d = x[..n].to_vec();
            let mut e = x[n..2 * n - 1].to_vec();
            let cz = cx(&x[2 * n - 1..]);
            let mut vt = cz[..n * ncv].to_vec();
            let mut u = cz[n * ncv..].to_vec();
            let info = zbdsqr(
                ul, n, ncv, nru, &mut d, &mut e, &mut vt, n, &mut u, nru,
            )
            .expect("zbdsqr");
            let mut v = d;
            v.extend(e);
            v.extend(flat(&vt));
            v.extend(flat(&u));
            (v, vec![info as i64])
        }
        "getrf_syn" | "getrf_det_tap" => {
            let (m, n) = (iv[0], iv[1]);
            let mut a = z.clone();
            let mut ipiv = vec![0usize; m.min(n)];
            let info = zgetrf(m, n, &mut a, m, &mut ipiv).expect("zgetrf");
            (flat(&a), ipiv_ints(info, &ipiv))
        }
        "gesvd_syn" => gesvd_out(iv[0], iv[1], &z, None),
        "gesvd_tap" => {
            let (v, i) = gesvd_out(iv[0], iv[1], &z, None);
            (svd_slice(&v, iv[1]), i)
        }
        "factor_tap" => {
            let ne = iv[0];
            let q = ne * ne;
            let (m_u, m_e, m_g, info, ipiv) = factor_step(
                ne,
                &z[..q],
                &z[q..2 * q],
                &z[2 * q..3 * q],
                &z[3 * q..],
            )
            .expect("factor step");
            let mut v = flat(&m_u);
            v.extend(flat(&m_e));
            v.extend(flat(&m_g));
            (v, ipiv_ints(info, &ipiv))
        }
        "gemv_tap" => {
            let ne = iv[0];
            let el = t.next(ne * ne);
            let er = t.next(ne * ne);
            let x1 = t.next(ne);
            let x2 = t.next(ne);
            let mut y = t.next(ne);
            zgemv(b'N', ne, ne, -ONE, &el, ne, &x1, 1, ZERO, &mut y, 1)
                .expect("zgemv");
            zgemv(b'N', ne, ne, -ONE, &er, ne, &x2, 1, ONE, &mut y, 1)
                .expect("zgemv");
            (flat(&y), vec![])
        }
        other => panic!("unknown section {other}"),
    }
}

#[test]
fn g4_complex_lapack_matches_mesa_sdk_bit_for_bit() {
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
            if first.len() < 12 {
                let bad: Vec<String> = ro
                    .iter()
                    .zip(&r.rout)
                    .enumerate()
                    .filter(|(_, (a, e))| a.to_bits() != **e)
                    .take(4)
                    .map(|(j, (a, e))| {
                        format!("[{j}] got {:016X} want {e:016X}", a.to_bits())
                    })
                    .collect();
                first.push(format!(
                    "{} {}: len {} vs {}; ints got {:?} want {:?}; {}",
                    r.sec,
                    r.id,
                    ro.len(),
                    r.rout.len(),
                    &io[..io.len().min(4)],
                    &r.iout[..r.iout.len().min(4)],
                    bad.join(", ")
                ));
            }
        }
    }
    // Every tapped call is checked: 23 ZGESVD calls (and their ZGETRF), 30
    // factor steps and 30 back-substitution pairs.
    for (sec, n) in [
        ("gesvd_tap", 23),
        ("getrf_det_tap", 23),
        ("factor_tap", 30),
        ("gemv_tap", 30),
    ] {
        assert_eq!(per.get(sec).map(|v| v.0), Some(n), "{sec} records");
    }
    let bad: usize = per.values().map(|v| v.1).sum();
    let summary: Vec<String> = per
        .iter()
        .map(|(k, (n, m))| format!("{k}: {n} cases, {m} mismatched"))
        .collect();
    eprintln!("{}", summary.join("\n"));
    assert!(
        bad == 0,
        "{bad} mismatched records\n{}\nfirst:\n{}",
        summary.join("\n"),
        first.join("\n")
    );
}

/// The vectors distinguish the SDK behaviour from plausible alternatives:
/// netlib's `TOLMUL = EPS**(-1/8)` in `ZBDSQR`, plain Smith division in
/// place of `ZLADIV`/`DLADIV`, and the classical right-looking unblocked LU
/// (`ZGETF2` order) in place of the recursive `ZGETRF2`.
#[test]
fn negative_controls_show_the_vectors_discriminate() {
    use flexmesa_gyre::lapack::zsvd::zbdsqr_with_tolmul;
    flexmesa_math::math_init();
    let recs = load();

    // ZBDSQR with netlib's TOLMUL = max(10, min(100, eps**(-1/8)))
    let eps = flexmesa_gyre::lapack::dlamch::EPS;
    let netlib = 10.0_f64.max(
        100.0_f64.min(flexmesa_math::exp(flexmesa_math::log(eps) * -0.125)),
    );
    let (mut n_svd, mut bad_svd) = (0, 0);
    for r in recs.iter().filter(|r| r.sec.starts_with("gesvd")) {
        let (mut v, i) = gesvd_out(
            r.iin[0] as usize,
            r.iin[1] as usize,
            &cx(&r.rin),
            Some(netlib),
        );
        if r.sec == "gesvd_tap" {
            v = svd_slice(&v, r.iin[1] as usize);
        }
        n_svd += 1;
        if i != r.iout || v.iter().zip(&r.rout).any(|(a, e)| a.to_bits() != *e)
        {
            bad_svd += 1;
        }
    }
    let (mut n_bd, mut bad_bd) = (0, 0);
    for r in recs.iter().filter(|r| r.sec == "zbdsqr") {
        let iv: Vec<usize> = r.iin.iter().map(|&i| i as usize).collect();
        let (ul, n, ncv, nru) = (iv[0] as u8, iv[1], iv[2], iv[3]);
        let x = &r.rin;
        let mut d = x[..n].to_vec();
        let mut e = x[n..2 * n - 1].to_vec();
        let cz = cx(&x[2 * n - 1..]);
        let mut vt = cz[..n * ncv].to_vec();
        let mut u = cz[n * ncv..].to_vec();
        let info = zbdsqr_with_tolmul(
            ul, n, ncv, nru, &mut d, &mut e, &mut vt, n, &mut u, nru, netlib,
        )
        .expect("zbdsqr");
        let mut v = d;
        v.extend(e);
        v.extend(flat(&vt));
        v.extend(flat(&u));
        n_bd += 1;
        if vec![info as i64] != r.iout
            || v.iter().zip(&r.rout).any(|(a, e)| a.to_bits() != *e)
        {
            bad_bd += 1;
        }
    }
    eprintln!(
        "netlib TOLMUL ({netlib}) differs on {bad_svd}/{n_svd} ZGESVD and {bad_bd}/{n_bd} ZBDSQR records"
    );
    assert!(bad_svd > 0, "netlib TOLMUL reproduced every ZGESVD record");

    // Smith's complex division in place of ZLADIV
    let (mut n_div, mut bad_div) = (0, 0);
    for r in recs.iter().filter(|r| r.sec == "zladiv") {
        let x = &r.rin;
        let q = Cplx::new(x[0], x[1]) / Cplx::new(x[2], x[3]);
        n_div += 1;
        if q.re.to_bits() != r.rout[0] || q.im.to_bits() != r.rout[1] {
            bad_div += 1;
        }
    }
    eprintln!("Smith division differs on {bad_div}/{n_div} ZLADIV records");
    assert!(bad_div > 0, "Smith division reproduced every ZLADIV record");

    // Classical right-looking unblocked LU (ZGETF2 order)
    let (mut n_lu, mut bad_lu) = (0, 0);
    for r in recs.iter().filter(|r| r.sec.starts_with("getrf")) {
        let (m, n) = (r.iin[0] as usize, r.iin[1] as usize);
        let mut a = cx(&r.rin);
        let d1 = |z: Cplx| z.re.abs() + z.im.abs();
        for j in 0..m.min(n) {
            let mut p = j;
            for i in j + 1..m {
                if d1(a[i + j * m]) > d1(a[p + j * m]) {
                    p = i;
                }
            }
            if !a[p + j * m].is_zero() {
                if p != j {
                    for k in 0..n {
                        a.swap(j + k * m, p + k * m);
                    }
                }
                let rcp = Cplx::real_div(1.0, a[j + j * m]);
                for i in j + 1..m {
                    a[i + j * m] = rcp * a[i + j * m];
                }
            }
            for k in j + 1..n {
                let tk = a[j + k * m];
                for i in j + 1..m {
                    a[i + k * m] = a[i + k * m] - a[i + j * m] * tk;
                }
            }
        }
        n_lu += 1;
        let v = flat(&a);
        if v.iter().zip(&r.rout).any(|(x, e)| x.to_bits() != *e) {
            bad_lu += 1;
        }
    }
    eprintln!(
        "classical ZGETF2 order differs on {bad_lu}/{n_lu} ZGETRF records"
    );
    assert!(bad_lu > 0, "classical LU reproduced every ZGETRF record");
}
