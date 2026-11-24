// SPDX-License-Identifier: GPL-3.0-only
//! Bit-for-bit check of the real LAPACK/BLAS port against the MESA SDK
//! `x86_64-linux-26.6.1` crmath-slot libraries
//! (`tests/data/mesa_26_4_1_gyre_g2_vectors.tsv`, produced by
//! `benchmarks/fortran_mesa/gyre_g2_vectors/gen_gyre_g2_vectors.f90`).
//!
//! Inputs are GYRE's own LAPACK calls from the G0 oracle runs (`*_tap`) and
//! synthetic edge cases (`*_syn`, leaf routines). Every defined real output
//! must match all 64 bits and every integer output exactly. Outputs that
//! upstream leaves undefined (`X`, `FERR`, `BERR` of `DGESVX` when the
//! factor is singular) are not compared.

#![allow(
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names
)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use flexmesa_gyre::lapack::aux::{dlapy2, dlarfg, dlartg, dlas2, dlasv2};
use flexmesa_gyre::lapack::blas::dnrm2;
use flexmesa_gyre::lapack::lu::{dgesvx, dgetrf};
use flexmesa_gyre::lapack::svd::dgesvd;
use flexmesa_gyre::lapack::{dlamch, ilaenv_mnthr, ilaenv_nb};

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
        .join("tests/data/mesa_26_4_1_gyre_g2_vectors.tsv");
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

/// (real outputs, int outputs, mask of real outputs to compare)
type Eval = (Vec<f64>, Vec<i64>, Vec<bool>);

fn all(v: Vec<f64>, i: Vec<i64>) -> Eval {
    let m = vec![true; v.len()];
    (v, i, m)
}

fn eval(r: &Rec) -> Eval {
    let x = &r.rin;
    let iv = &r.iin;
    match r.sec.as_str() {
        "dlamch" => all(
            vec![
                dlamch::EPS,
                dlamch::SFMIN,
                dlamch::BASE,
                dlamch::PREC,
                dlamch::RMAX,
                dlamch::RMIN,
            ],
            vec![],
        ),
        "ilaenv" => {
            let mut v: Vec<i64> =
                ["DGETRF", "DGEQRF", "DGEBRD", "DORGQR", "DORGLQ"]
                    .iter()
                    .map(|n| i64::from(ilaenv_nb(n)))
                    .collect();
            v.extend((1..=12).map(|k| ilaenv_mnthr(k, k) as i64));
            all(vec![], v)
        }
        "dnrm2" => all(vec![dnrm2(iv[0] as usize, x, 1)], vec![]),
        "dlartg" => {
            let (c, s, rr) = dlartg(x[0], x[1]);
            all(vec![c, s, rr], vec![])
        }
        "dlas2" => {
            let (a, b) = dlas2(x[0], x[1], x[2]);
            all(vec![a, b], vec![])
        }
        "dlasv2" => {
            let o = dlasv2(x[0], x[1], x[2]);
            all(vec![o.ssmin, o.ssmax, o.snr, o.csr, o.snl, o.csl], vec![])
        }
        "dlapy2" => all(vec![dlapy2(x[0], x[1])], vec![]),
        "dlarfg" => {
            let n = iv[0] as usize;
            let mut alpha = x[0];
            let mut v = x[1..].to_vec();
            let tau = dlarfg(n, &mut alpha, &mut v, 1);
            let mut out = vec![alpha, tau];
            out.extend(v);
            all(out, vec![])
        }
        "getrf_tap" | "getrf_syn" => {
            let (m, n) = (iv[0] as usize, iv[1] as usize);
            let mut a = x.clone();
            let mut ipiv = vec![0usize; m.min(n)];
            let info = dgetrf(m, n, &mut a, m, &mut ipiv).expect("dgetrf");
            let mut ints = vec![info as i64];
            ints.extend(ipiv.iter().map(|&p| p as i64));
            all(a, ints)
        }
        "gesvx_tap" | "gesvx_syn" => {
            let (n, nrhs) = (iv[0] as usize, iv[1] as usize);
            let mut a = x[..n * n].to_vec();
            let mut b = x[n * n..].to_vec();
            let mut af = vec![0.0; n * n];
            let mut xs = vec![0.0; n * nrhs];
            let mut ipiv = vec![0usize; n];
            let o = dgesvx(
                b'E', b'N', n, nrhs, &mut a, &mut af, &mut ipiv, &mut b,
                &mut xs,
            )
            .expect("dgesvx");
            let singular = o.info > 0 && o.info <= n;
            let mut v = Vec::new();
            let mut mask = Vec::new();
            let mut push = |vals: &[f64], keep: bool| {
                v.extend_from_slice(vals);
                mask.extend(std::iter::repeat_n(keep, vals.len()));
            };
            push(&a, true);
            push(&af, true);
            push(&o.r, true);
            push(&o.c, true);
            push(&b, true);
            push(&xs, !singular);
            push(&[o.rcond], true);
            push(&o.ferr, !singular);
            push(&o.berr, !singular);
            push(&[o.rpvgrw], true);
            let mut ints = vec![o.info as i64, i64::from(o.equed)];
            ints.extend(ipiv.iter().map(|&p| p as i64));
            (v, ints, mask)
        }
        "gesvd_tap" | "gesvd_syn" => {
            let n = iv[0] as usize;
            let mut a = x.clone();
            let mut u = vec![0.0; n * n];
            let mut vt = vec![0.0; n * n];
            let o =
                dgesvd(n, n, &mut a, &mut u, &mut vt, 5 * n).expect("dgesvd");
            let mut v = a;
            v.extend(o.s);
            v.extend(u);
            v.extend(vt);
            v.push(o.work1);
            all(v, vec![o.info as i64])
        }
        other => panic!("unknown section {other}"),
    }
}

#[test]
fn g2_lapack_matches_mesa_sdk_bit_for_bit() {
    flexmesa_math::math_init();
    let recs = load();
    let mut per: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut first: Vec<String> = Vec::new();
    for r in &recs {
        let (ro, io, mask) = eval(r);
        let ok_r = ro.len() == r.rout.len()
            && ro
                .iter()
                .zip(&r.rout)
                .zip(&mask)
                .all(|((a, e), &k)| !k || a.to_bits() == *e);
        let ok_i = io == r.iout;
        let ent = per.entry(r.sec.clone()).or_insert((0, 0));
        ent.0 += 1;
        if !(ok_r && ok_i) {
            ent.1 += 1;
            if first.len() < 12 {
                let bad: Vec<String> = ro
                    .iter()
                    .zip(&r.rout)
                    .zip(&mask)
                    .enumerate()
                    .filter(|(_, ((a, e), &k))| k && a.to_bits() != **e)
                    .take(4)
                    .map(|(j, ((a, e), _))| {
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
    let bad: usize = per.values().map(|v| v.1).sum();
    let summary: Vec<String> = per
        .iter()
        .map(|(k, (n, m))| format!("{k}: {n} cases, {m} mismatched"))
        .collect();
    assert!(
        bad == 0,
        "{bad} mismatched records\n{}\nfirst:\n{}",
        summary.join("\n"),
        first.join("\n")
    );
}

/// The vectors distinguish the SDK behaviour from the plausible
/// alternatives: netlib's `TOLMUL = EPS**(-1/8)` in `DBDSQR`, and the
/// classical right-looking `DGETF2` instead of the recursive `DGETRF2`.
#[test]
fn negative_controls_show_the_vectors_discriminate() {
    use flexmesa_gyre::lapack::svd::dgesvd_with_tolmul;
    flexmesa_math::math_init();
    let recs = load();

    // DBDSQR with netlib's TOLMUL = max(10, min(100, eps**(-1/8)))
    let eps = dlamch::EPS;
    let netlib = 10.0_f64.max(
        100.0_f64.min(flexmesa_math::exp(flexmesa_math::log(eps) * -0.125)),
    );
    let (mut n_svd, mut bad_svd) = (0, 0);
    for r in recs.iter().filter(|r| r.sec.starts_with("gesvd")) {
        let n = r.iin[0] as usize;
        let mut a = r.rin.clone();
        let mut u = vec![0.0; n * n];
        let mut vt = vec![0.0; n * n];
        let o =
            dgesvd_with_tolmul(n, n, &mut a, &mut u, &mut vt, 5 * n, netlib)
                .expect("dgesvd");
        let mut v = a;
        v.extend(o.s);
        v.extend(u);
        v.extend(vt);
        v.push(o.work1);
        n_svd += 1;
        if v.iter().zip(&r.rout).any(|(a, e)| a.to_bits() != *e) {
            bad_svd += 1;
        }
    }
    eprintln!(
        "netlib TOLMUL ({netlib}) differs on {bad_svd}/{n_svd} DGESVD records"
    );
    assert!(bad_svd > 0, "netlib TOLMUL reproduced every DGESVD record");

    // Classical right-looking unblocked LU (DGETF2 order)
    let (mut n_lu, mut bad_lu) = (0, 0);
    for r in recs.iter().filter(|r| r.sec.starts_with("getrf")) {
        let (m, n) = (r.iin[0] as usize, r.iin[1] as usize);
        let mut a = r.rin.clone();
        for j in 0..m.min(n) {
            let mut p = j;
            for i in j + 1..m {
                if a[i + j * m].abs() > a[p + j * m].abs() {
                    p = i;
                }
            }
            if a[p + j * m] != 0.0 {
                if p != j {
                    for k in 0..n {
                        a.swap(j + k * m, p + k * m);
                    }
                }
                let rcp = 1.0 / a[j + j * m];
                for i in j + 1..m {
                    a[i + j * m] *= rcp;
                }
            }
            for k in j + 1..n {
                let t = a[j + k * m];
                for i in j + 1..m {
                    a[i + k * m] -= a[i + j * m] * t;
                }
            }
        }
        n_lu += 1;
        if a.iter().zip(&r.rout).any(|(x, e)| x.to_bits() != *e) {
            bad_lu += 1;
        }
    }
    eprintln!(
        "classical DGETF2 order differs on {bad_lu}/{n_lu} DGETRF records"
    );
    assert!(bad_lu > 0, "classical LU reproduced every DGETRF record");
}
