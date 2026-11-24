// SPDX-License-Identifier: GPL-3.0-or-later
//! Bit-for-bit verification of G5 eigensolver and BLAS additions (`dgeev`,
//! `zgeev`, `dlapy3`, `dzasum`) against committed MESA 26.4.1 G0 oracle taps
//! (`benchmarks/fortran_mesa/gyre_oracles/data/*.taps.tsv`) and MESA SDK
//! 26.6.1 crmath-slot vectors (`tests/data/mesa_26_4_1_gyre_g5_eig_vectors.tsv`).

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
use flexmesa_gyre::lapack::eig::dgeev;
use flexmesa_gyre::lapack::zaux::dlapy3;
use flexmesa_gyre::lapack::zblas::dzasum;
use flexmesa_gyre::lapack::zeig::zgeev;

fn parse_hex(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).expect("hex"))
}

fn parse_list<T>(s: &str, f: impl Fn(&str) -> T) -> Vec<T> {
    if s == "-" {
        Vec::new()
    } else {
        s.split(' ').map(f).collect()
    }
}

fn parse_shape(name: &str) -> Option<(usize, usize)> {
    let b = name.find('[')?;
    let e = name.find(']')?;
    let dims = name[b + 1..e].split(',').next()?;
    let (r, c) = dims.split_once('x')?;
    Some((r.parse().ok()?, c.parse().ok()?))
}

#[test]
fn dgeev_and_zgeev_match_all_g0_oracle_taps_bit_for_bit() {
    flexmesa_math::math_init();
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
    paths.sort();

    let mut dgeev_count = 0usize;
    let mut zgeev_count = 0usize;

    for p in paths {
        let text = std::fs::read_to_string(&p).expect("read taps.tsv");
        let mut active: Option<(&'static str, usize, Vec<u64>)> = None;
        let mut pending: BTreeMap<String, String> = BTreeMap::new();
        for line in text.lines() {
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            let (kind, name) = (f[0], f[1]);
            let bits = |s: &str| -> Vec<u64> {
                s.split(' ')
                    .map(|h| u64::from_str_radix(h, 16).expect("hex"))
                    .collect()
            };
            if kind == "LAPACK_IN" && name.starts_with("DGEEV:A[") {
                let (r, c) = parse_shape(name).expect("DGEEV shape");
                assert_eq!(r, c);
                active = Some(("DGEEV", r, bits(f[3])));
                pending.clear();
            } else if kind == "LAPACK_OUT" && name.starts_with("DGEEV:") {
                let key = name[6..].split('[').next().expect("key").to_string();
                pending.insert(key, f[3].to_string());
                if pending.len() == 5 {
                    let (_, n, raw) = active.take().expect("DGEEV active");
                    let mut a: Vec<f64> =
                        raw.into_iter().map(f64::from_bits).collect();
                    let mut vl = vec![0.0_f64; n * n];
                    let mut vr = vec![0.0_f64; n * n];
                    let out = dgeev(
                        b'V',
                        b'V',
                        n,
                        &mut a,
                        n,
                        &mut vl,
                        n,
                        &mut vr,
                        n,
                        4 * n,
                    )
                    .expect("dgeev");
                    assert_eq!(
                        out.info,
                        pending["info"].parse::<usize>().expect("info")
                    );
                    let got_wr: Vec<u64> =
                        out.wr.iter().map(|v| v.to_bits()).collect();
                    let got_wi: Vec<u64> =
                        out.wi.iter().map(|v| v.to_bits()).collect();
                    let got_vl: Vec<u64> =
                        vl.iter().map(|v| v.to_bits()).collect();
                    let got_vr: Vec<u64> =
                        vr.iter().map(|v| v.to_bits()).collect();
                    assert_eq!(got_wr, bits(&pending["wr"]), "DGEEV wr");
                    assert_eq!(got_wi, bits(&pending["wi"]), "DGEEV wi");
                    assert_eq!(got_vl, bits(&pending["VL"]), "DGEEV VL");
                    assert_eq!(got_vr, bits(&pending["VR"]), "DGEEV VR");
                    dgeev_count += 1;
                    pending.clear();
                }
            } else if kind == "LAPACK_IN" && name.starts_with("ZGEEV:A[") {
                let (r, c) = parse_shape(name).expect("ZGEEV shape");
                assert_eq!(r, c);
                active = Some(("ZGEEV", r, bits(f[3])));
                pending.clear();
            } else if kind == "LAPACK_OUT" && name.starts_with("ZGEEV:") {
                let key = name[6..].split('[').next().expect("key").to_string();
                pending.insert(key, f[3].to_string());
                if pending.len() == 4 {
                    let (_, n, raw) = active.take().expect("ZGEEV active");
                    let reals: Vec<f64> =
                        raw.into_iter().map(f64::from_bits).collect();
                    let mut a: Vec<Cplx> = reals
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|&[re, im]| Cplx::new(re, im))
                        .collect();
                    let mut vl = vec![Cplx::new(0.0, 0.0); n * n];
                    let mut vr = vec![Cplx::new(0.0, 0.0); n * n];
                    let out = zgeev(
                        b'V',
                        b'V',
                        n,
                        &mut a,
                        n,
                        &mut vl,
                        n,
                        &mut vr,
                        n,
                        2 * n,
                    )
                    .expect("zgeev");
                    assert_eq!(
                        out.info,
                        pending["info"].parse::<usize>().expect("info")
                    );
                    let flat = |v: &[Cplx]| -> Vec<u64> {
                        v.iter()
                            .flat_map(|z| [z.re.to_bits(), z.im.to_bits()])
                            .collect()
                    };
                    assert_eq!(flat(&out.w), bits(&pending["w"]), "ZGEEV w");
                    assert_eq!(flat(&vl), bits(&pending["VL"]), "ZGEEV VL");
                    assert_eq!(flat(&vr), bits(&pending["VR"]), "ZGEEV VR");
                    zgeev_count += 1;
                    pending.clear();
                }
            }
        }
    }
    assert_eq!(dgeev_count, 4, "expected 4 G0 DGEEV taps");
    assert_eq!(zgeev_count, 20, "expected 20 G0 ZGEEV taps");
}

#[test]
fn sdk_eig_and_zblas_vectors_match_bit_for_bit() {
    flexmesa_math::math_init();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/mesa_26_4_1_gyre_g5_eig_vectors.tsv");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for line in text.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        let sec = f[0];
        let id: usize = f[1].parse().expect("id");
        let rin = parse_list(f[2], parse_hex);
        let iin: Vec<i64> = parse_list(f[3], |v| v.parse().expect("int"));
        let rout: Vec<u64> =
            parse_list(f[4], |h| u64::from_str_radix(h, 16).expect("hex"));
        let iout: Vec<i64> = parse_list(f[5], |v| v.parse().expect("int"));
        *counts.entry(sec.to_string()).or_insert(0) += 1;
        match sec {
            "dlapy3" => {
                assert_eq!(
                    dlapy3(rin[0], rin[1], rin[2]).to_bits(),
                    rout[0],
                    "dlapy3 {id}"
                );
            }
            "dzasum_vec" => {
                let n = iin[0] as usize;
                let zx: Vec<Cplx> = rin
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&[re, im]| Cplx::new(re, im))
                    .collect();
                assert_eq!(dzasum(n, &zx, 1).to_bits(), rout[0], "dzasum {id}");
            }
            "dgeev_syn" => {
                let n = iin[0] as usize;
                let lwork = iin[1] as usize;
                let mut a = rin;
                let mut vl = vec![0.0; n * n];
                let mut vr = vec![0.0; n * n];
                let out = dgeev(
                    b'V', b'V', n, &mut a, n, &mut vl, n, &mut vr, n, lwork,
                )
                .expect("dgeev");
                assert_eq!(out.info, iout[0] as usize, "dgeev_syn {id} info");
                let mut got = Vec::new();
                got.extend(out.wr.iter().map(|v| v.to_bits()));
                got.extend(out.wi.iter().map(|v| v.to_bits()));
                got.extend(vl.iter().map(|v| v.to_bits()));
                got.extend(vr.iter().map(|v| v.to_bits()));
                got.push(out.work1.to_bits());
                assert_eq!(got, rout, "dgeev_syn {id}");
            }
            "zgeev_syn" => {
                let n = iin[0] as usize;
                let lwork = iin[1] as usize;
                let mut a: Vec<Cplx> = rin
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&[re, im]| Cplx::new(re, im))
                    .collect();
                let mut vl = vec![Cplx::new(0.0, 0.0); n * n];
                let mut vr = vec![Cplx::new(0.0, 0.0); n * n];
                let out = zgeev(
                    b'V', b'V', n, &mut a, n, &mut vl, n, &mut vr, n, lwork,
                )
                .expect("zgeev");
                assert_eq!(out.info, iout[0] as usize, "zgeev_syn {id} info");
                let mut got = Vec::new();
                for z in out.w.iter().chain(vl.iter()).chain(vr.iter()) {
                    got.push(z.re.to_bits());
                    got.push(z.im.to_bits());
                }
                got.push(out.work1.re.to_bits());
                got.push(out.work1.im.to_bits());
                assert_eq!(got, rout, "zgeev_syn {id}");
            }
            other => panic!("unexpected vector section {other}"),
        }
    }
    assert_eq!(counts["dgeev_syn"], 15);
    assert_eq!(counts["zgeev_syn"], 12);
    assert_eq!(counts["dlapy3"], 50);
    assert_eq!(counts["dzasum_vec"], 50);
}
