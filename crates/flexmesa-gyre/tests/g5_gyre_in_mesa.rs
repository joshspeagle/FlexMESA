// SPDX-License-Identifier: GPL-3.0-or-later
//! Bit-for-bit verification of the remaining six MESA 26.4.1 `#170` GYRE test
//! suite cases (`gyre_in_mesa_bcep`, `gyre_in_mesa_envelope`,
//! `gyre_in_mesa_rsg`, `gyre_in_mesa_wd`, `gyre_in_mesa_spb`, and `rsp_gyre`)
//! plus standalone deterministic coverage of `delta_g`, `GRAVITY_DELTA`,
//! `INVERSE` scans, `MAGNUS_GL2`, `ZERO_R`/`ZERO_H`, `AdBvp` 4x4 non-radial
//! adiabatic solves, `find_turn`, `prop_type`, dipole Takata classification,
//! and segmented match conditions.

#![allow(
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::type_complexity
)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use flexmesa_gyre::ad::{ad_wave_hom, AdBound, AdBvp, AdDiff, AdEqns, AdMatch};
use flexmesa_gyre::constants::GyreConstants;
use flexmesa_gyre::context::Context;
use flexmesa_gyre::ext_complex::ExtC;
use flexmesa_gyre::ext_real::ExtR;
use flexmesa_gyre::fcomplex::Cplx;
use flexmesa_gyre::grid::Grid;
use flexmesa_gyre::magnus::{
    eigen_decompose_2_c, eigen_decompose_2_r, eigen_decompose_c,
    eigen_decompose_r,
};
use flexmesa_gyre::mesa::GyreMesa;
use flexmesa_gyre::mode::{Mode, Wave};
use flexmesa_gyre::model::{idx, EvolModel};
use flexmesa_gyre::nad::{NadBound, NadDiff, NadMatch};
use flexmesa_gyre::par::{ModePar, ModelPar, NumPar, OscPar, RotPar, ScanPar};
use flexmesa_gyre::rad::{GyreTap, NoTap, RadBound, RadDiff, RadMatch};
use flexmesa_gyre::search::{build_scan, find_turn};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn histories_dir() -> PathBuf {
    if let Some(v) = std::env::var_os("FLEXMESA_HISTORIES_DIR") {
        if !v.is_empty() {
            return PathBuf::from(v);
        }
    }
    let data = flexmesa_utils::resolve_flexmesa_data_dir();
    if let Some(parent) = data.parent() {
        let c = parent.join("mesa_histories");
        if c.is_dir() {
            return c;
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".cache/flexmesa/mesa_histories");
    }
    PathBuf::from(".flexmesa_histories")
}

fn require(path: PathBuf, what: &str) -> Option<PathBuf> {
    if path.exists() {
        return Some(path);
    }
    assert!(
        !flexmesa_utils::require_zero_skips_enabled(),
        "missing {what} `{}` (run `flexmesa data fetch --include-test-inputs` \
         or set FLEXMESA_REQUIRE_ZERO_SKIPS=0)",
        path.display()
    );
    None
}

fn parse_f(s: &str) -> f64 {
    s.replace(['D', 'd'], "e").parse().expect("real")
}

fn load_model(path: &Path) -> (Vec<f64>, Vec<Vec<f64>>, i32) {
    let text = std::fs::read_to_string(path).expect("read model");
    let mut lines = text.lines();
    let head: Vec<&str> =
        lines.next().expect("header").split_whitespace().collect();
    let n: usize = head[0].parse().expect("n");
    let global: Vec<f64> = head[1..4].iter().map(|v| parse_f(v)).collect();
    let version: i32 = head[4].parse().expect("version");
    let mut rows = Vec::with_capacity(n);
    for (k, l) in lines.take(n).enumerate() {
        let f: Vec<&str> = l.split_whitespace().collect();
        assert_eq!(f[0].parse::<usize>().expect("k"), k + 1);
        rows.push(f[1..].iter().map(|v| parse_f(v)).collect());
    }
    (global, rows, version)
}

fn standard_constants() -> GyreConstants {
    use flexmesa_const as c;
    let mut k = GyreConstants::new();
    k.set_real("G_GRAVITY", c::STANDARD_CGRAV).unwrap();
    k.set_real("C_LIGHT", c::CLIGHT).unwrap();
    k.set_real("A_RADIATION", c::CRAD).unwrap();
    k.set_real("M_SUN", c::MSUN).unwrap();
    k.set_real("R_SUN", c::RSUN).unwrap();
    k.set_real("L_SUN", c::LSUN).unwrap();
    k.set_char("GYRE_DIR", "mesa/gyre/gyre").unwrap();
    k
}

fn new_gyre(gyre_in: &str) -> GyreMesa {
    use flexmesa_const as c;
    let mut g = GyreMesa::init(gyre_in).expect("init");
    g.set_constant("G_GRAVITY", c::STANDARD_CGRAV).unwrap();
    g.set_constant("C_LIGHT", c::CLIGHT).unwrap();
    g.set_constant("A_RADIATION", c::CRAD).unwrap();
    g.set_constant("M_SUN", c::MSUN).unwrap();
    g.set_constant("R_SUN", c::RSUN).unwrap();
    g.set_constant("L_SUN", c::LSUN).unwrap();
    g.set_constant_str("GYRE_DIR", "mesa/gyre/gyre").unwrap();
    g
}

fn hexbits(h: &str) -> u64 {
    u64::from_str_radix(h, 16).expect("hex")
}

fn cbits(z: Cplx) -> [u64; 2] {
    [z.re.to_bits(), z.im.to_bits()]
}

fn scl_parts(z: ExtC) -> ([u64; 2], i32) {
    (cbits(z.f), z.e)
}

fn hexs(v: &[String]) -> Vec<u64> {
    v.iter().map(|s| hexbits(s)).collect()
}

fn ext_e(h: &str) -> i32 {
    f64::from_bits(hexbits(h)) as i32
}

fn read_tsv(path: &Path) -> Vec<(String, String, Vec<String>)> {
    std::fs::read_to_string(path)
        .expect("read oracle")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            (
                f[0].to_string(),
                f[1].to_string(),
                f.last()
                    .expect("values")
                    .split(' ')
                    .map(str::to_string)
                    .collect(),
            )
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BuildKind {
    Rad,
    Ad,
    Complex,
}

#[derive(Default)]
struct Rec {
    grid_x: Vec<u64>,
    grid_s: Vec<i64>,
    scan: Vec<u64>,
    builds: Vec<(usize, bool, u64, Vec<u64>, Vec<u64>)>,
    blocks: Vec<(usize, usize, Vec<u64>, Vec<u64>, u64, i32)>,
    want_j: Vec<usize>,
    discrim_all: Vec<(u64, u64, u64, u64, i32, bool)>,
    kinds: Vec<BuildKind>,
    builds_c: BTreeMap<usize, (usize, bool, [u64; 2], Vec<u64>, Vec<u64>)>,
    blocks_c: Vec<(usize, usize, Vec<u64>, Vec<u64>, [u64; 2], i32)>,
    mode_ad: Vec<([u64; 2], [i64; 5])>,
    mode_nad: Vec<([u64; 2], [i64; 5])>,
}

impl GyreTap for Rec {
    fn grid_scan(&mut self, gr: &Grid, omega_re: &[f64]) {
        self.grid_x = gr.pt.iter().map(|p| p.x.to_bits()).collect();
        self.grid_s = gr.pt.iter().map(|p| i64::from(p.s)).collect();
        self.scan = omega_re.iter().map(|w| w.to_bits()).collect();
    }
    fn build(
        &mut self,
        n: usize,
        mode_build: bool,
        omega: f64,
        b_i: &[f64; 2],
        b_o: &[f64; 2],
    ) {
        self.kinds.push(BuildKind::Rad);
        self.builds.push((
            n,
            mode_build,
            omega.to_bits(),
            b_i.iter().map(|v| v.to_bits()).collect(),
            b_o.iter().map(|v| v.to_bits()).collect(),
        ));
    }
    fn build_ad(
        &mut self,
        n: usize,
        mode_build: bool,
        omega: f64,
        b_i: &[f64; 8],
        b_o: &[f64; 8],
    ) {
        self.kinds.push(BuildKind::Ad);
        self.builds.push((
            n,
            mode_build,
            omega.to_bits(),
            b_i.iter().map(|v| v.to_bits()).collect(),
            b_o.iter().map(|v| v.to_bits()).collect(),
        ));
    }
    fn build_c(
        &mut self,
        n: usize,
        mode_build: bool,
        omega: Cplx,
        b_i: &[Cplx; 18],
        b_o: &[Cplx; 18],
    ) {
        self.kinds.push(BuildKind::Complex);
        self.builds.push((n, mode_build, 0, Vec::new(), Vec::new()));
        let flat = |b: &[Cplx; 18]| -> Vec<u64> {
            b.iter().flat_map(|z| cbits(*z)).collect()
        };
        self.builds_c.insert(
            self.builds.len(),
            (n, mode_build, cbits(omega), flat(b_i), flat(b_o)),
        );
    }
    fn block_c(
        &mut self,
        j: usize,
        e_l: &[Cplx; 36],
        e_r: &[Cplx; 36],
        scl: ExtC,
    ) {
        let b = self.builds.len();
        let dumped = b == 1 || self.builds[b - 1].1;
        if dumped && self.want_j.contains(&j) {
            let (f, e) = scl_parts(scl);
            self.blocks_c.push((
                b,
                j,
                e_l.iter().flat_map(|z| cbits(*z)).collect(),
                e_r.iter().flat_map(|z| cbits(*z)).collect(),
                f,
                e,
            ));
        }
    }
    fn discrim_c(&mut self, omega: Cplx, det: ExtC) {
        let (f, e) = scl_parts(det);
        let w = cbits(omega);
        self.discrim_all.push((w[0], w[1], f[0], f[1], e, true));
    }
    fn mode_ad(&mut self, omega: Cplx, ints: [i64; 5]) {
        self.mode_ad.push((cbits(omega), ints));
    }
    fn mode_nad(&mut self, omega: Cplx, ints: [i64; 5]) {
        self.mode_nad.push((cbits(omega), ints));
    }
    fn block(&mut self, j: usize, e_l: &[f64; 4], e_r: &[f64; 4], scl: ExtR) {
        let b = self.builds.len();
        let dumped = b == 1 || self.builds[b - 1].1;
        if dumped && self.want_j.contains(&j) {
            self.blocks.push((
                b,
                j,
                e_l.iter().map(|v| v.to_bits()).collect(),
                e_r.iter().map(|v| v.to_bits()).collect(),
                scl.f.to_bits(),
                scl.e,
            ));
        }
    }
    fn block_ad(
        &mut self,
        j: usize,
        e_l: &[f64; 16],
        e_r: &[f64; 16],
        scl: ExtR,
    ) {
        let b = self.builds.len();
        let dumped = b == 1 || self.builds[b - 1].1;
        if dumped && self.want_j.contains(&j) {
            self.blocks.push((
                b,
                j,
                e_l.iter().map(|v| v.to_bits()).collect(),
                e_r.iter().map(|v| v.to_bits()).collect(),
                scl.f.to_bits(),
                scl.e,
            ));
        }
    }
    fn discrim(&mut self, omega: f64, det: ExtR) {
        self.discrim_all.push((
            omega.to_bits(),
            0,
            det.f.to_bits(),
            0,
            det.e,
            false,
        ));
    }
}

fn assert_case_g0_and_pulse_taps(
    case: &str,
    expected_evals: usize,
    expected_modes: usize,
) {
    flexmesa_math::math_init();
    let Some(gin_path) = require(
        flexmesa_utils::resolve_mesa_test_inputs_dir()
            .join(format!("star/test_suite/{case}/gyre.in")),
        &format!("MESA 26.4.1 {case} gyre.in"),
    ) else {
        return;
    };
    let gin = std::fs::read_to_string(gin_path).expect("read gyre.in");
    let Some(pulse_dir) = require(
        histories_dir().join(format!("runs/star/{case}/pulse_tap")),
        &format!("MESA 26.4.1 {case} pulse_tap captures"),
    ) else {
        return;
    };

    let data = repo_root().join("benchmarks/fortran_mesa/gyre_oracles/data");
    let taps = read_tsv(&data.join(format!("star__{case}__e000001.taps.tsv")));
    let modes_text = std::fs::read_to_string(
        data.join(format!("star__{case}__e000001.modes.tsv")),
    )
    .expect("read modes oracle");
    let model_rel = modes_text
        .lines()
        .find_map(|l| {
            l.strip_prefix("# model_file: ")
                .map(|r| r.split(" (").next().expect("model path").to_string())
        })
        .expect("model_file header");
    let degree: i32 = modes_text
        .lines()
        .find_map(|l| {
            l.strip_prefix("# degrees_called: ")
                .and_then(|s| s.trim().parse().ok())
        })
        .expect("degrees_called header");
    let model_path =
        histories_dir().join("runs/star").join(case).join(model_rel);
    let (global, rows, version) = load_model(&model_path);

    let mut rec = Rec::default();
    for (tag, name, _) in &taps {
        if tag == "BLOCK" && name.contains(":scl[") {
            let j: usize = name[2..name.find(':').unwrap()].parse().unwrap();
            if !rec.want_j.contains(&j) {
                rec.want_j.push(j);
            }
        }
    }

    let modes: Vec<Vec<String>> = modes_text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| l.split('\t').map(str::to_string).collect())
        .collect();
    let stride: usize = modes_text
        .lines()
        .find_map(|l| {
            l.strip_prefix("# eigenfunction_stride: ")
                .and_then(|s| s.parse().ok())
        })
        .expect("stride header");
    let mut out: Vec<(String, Vec<u64>)> = Vec::new();
    let mut mode_ints: Vec<Vec<i64>> = Vec::new();
    let mut gyre = new_gyre(&gin);
    gyre.set_model(&global, &rows, version).expect("set_model");
    let mut count = 0usize;
    gyre.get_modes_tapped(
        degree,
        &mut |md: &Mode<'_>| {
            count += 1;
            let w = &md.wave;
            mode_ints.push(vec![
                i64::from(w.l),
                i64::from(w.m),
                i64::from(md.n_p),
                i64::from(md.n_g),
                i64::from(md.n_pg),
                w.n as i64,
                i64::from(w.id),
            ]);
            out.push((format!("OMEGA {count}"), cbits(w.omega).to_vec()));
            out.push((
                format!("FREQ_UHZ {count}"),
                cbits(md.freq("UHZ")?).to_vec(),
            ));
            out.push((
                format!("FREQ_HZ {count}"),
                cbits(md.freq("HZ")?).to_vec(),
            ));
            out.push((
                format!("FREQ_RAD_PER_SEC {count}"),
                cbits(md.freq("RAD_PER_SEC")?).to_vec(),
            ));
            out.push((format!("E_NORM {count}"), vec![md.e_norm()?.to_bits()]));
            let ml = w.cx.ml;
            for k in 1..=w.n {
                if (k - 1) % stride != 0 && k != w.n {
                    continue;
                }
                let mut eig = vec![w.x(k).to_bits()];
                for i in 1..=6 {
                    eig.extend(cbits(w.y_i(i, k)));
                }
                eig.extend(cbits(w.xi_r(k)?));
                eig.extend(cbits(w.xi_h(k)?));
                eig.push(w.dw_dx(k)?.to_bits());
                out.push((format!("EIG {count} {k}"), eig));
                let pt = w.gr.pt[k - 1];
                let mut co = Vec::new();
                for ic in 1..=idx::LAST {
                    co.push(ml.coeff(ic, pt).map_or(0, f64::to_bits));
                }
                out.push((format!("COEFF {count} {k}"), co));
            }
            Ok(0)
        },
        &mut rec,
    )
    .expect("get_modes_tapped");

    for (tag, name, v) in &taps {
        match (tag.as_str(), name.as_str()) {
            ("GRID", "x") => assert_eq!(rec.grid_x, hexs(v), "{case} GRID x"),
            ("GRID", "s") => assert_eq!(
                rec.grid_s,
                v.iter()
                    .map(|s| s.parse::<i64>().unwrap())
                    .collect::<Vec<_>>(),
                "{case} GRID s"
            ),
            ("SCAN", _) => assert_eq!(rec.scan, hexs(v), "{case} SCAN"),
            _ => {}
        }
    }
    let mut b_idx = 0usize;
    let mut bound: Vec<Vec<u64>> = Vec::new();
    for (tag, name, v) in &taps {
        match tag.as_str() {
            "BUILD" if name.starts_with("index") => {
                b_idx = v[0].parse().unwrap();
                let (n, mb, ..) = rec.builds[b_idx - 1];
                assert_eq!(
                    n,
                    v[1].parse::<usize>().unwrap(),
                    "{case} build {b_idx} n"
                );
                assert_eq!(mb, v[2] == "1", "{case} build {b_idx} mb");
            }
            "BUILD" => {
                if v.len() == 1 {
                    assert_ne!(rec.kinds[b_idx - 1], BuildKind::Complex);
                    assert_eq!(
                        rec.builds[b_idx - 1].2,
                        hexbits(&v[0]),
                        "{case} build {b_idx} omega"
                    );
                } else {
                    assert_eq!(rec.kinds[b_idx - 1], BuildKind::Complex);
                    assert_eq!(
                        rec.builds_c[&b_idx].2.to_vec(),
                        hexs(v),
                        "{case} build {b_idx} omega_c"
                    );
                }
            }
            "BOUND" => {
                bound.push(hexs(v));
                if name == "scl_o" {
                    match rec.kinds[b_idx - 1] {
                        BuildKind::Complex => {
                            let (_, _, _, bi, bo) = &rec.builds_c[&b_idx];
                            let one = cbits(Cplx::new(1.0, 0.0));
                            let ones: Vec<u64> =
                                (0..3).flat_map(|_| one).collect();
                            assert_eq!(
                                &bound[0], bi,
                                "{case} build {b_idx} B_i"
                            );
                            assert_eq!(bound[1], ones, "{case} scl_i");
                            assert_eq!(
                                &bound[2], bo,
                                "{case} build {b_idx} B_o"
                            );
                            assert_eq!(bound[3], ones, "{case} scl_o");
                        }
                        BuildKind::Rad | BuildKind::Ad => {
                            let (_, _, _, bi, bo) = &rec.builds[b_idx - 1];
                            let n_bc = if rec.kinds[b_idx - 1] == BuildKind::Ad
                            {
                                2
                            } else {
                                1
                            };
                            let ones = vec![1.0_f64.to_bits(); n_bc];
                            assert_eq!(
                                &bound[0], bi,
                                "{case} build {b_idx} B_i"
                            );
                            assert_eq!(bound[1], ones, "{case} scl_i");
                            assert_eq!(
                                &bound[2], bo,
                                "{case} build {b_idx} B_o"
                            );
                            assert_eq!(bound[3], ones, "{case} scl_o");
                        }
                    }
                    bound.clear();
                }
            }
            "BLOCK" => {
                let j: usize =
                    name[2..name.find(':').unwrap()].parse().unwrap();
                let bits = hexs(v);
                if rec.kinds[b_idx - 1] == BuildKind::Complex {
                    let (_, _, el, er, f, e) = rec
                        .blocks_c
                        .iter()
                        .find(|b| b.0 == b_idx && b.1 == j)
                        .unwrap();
                    if name.contains(":scl") {
                        assert_eq!(
                            &bits[0..2],
                            f,
                            "{case} build {b_idx} block {j} scl f"
                        );
                        assert_eq!(
                            ext_e(&v[2]),
                            *e,
                            "{case} build {b_idx} block {j} scl e"
                        );
                    } else {
                        let got = if name.contains(":E_l") { el } else { er };
                        assert_eq!(
                            got, &bits,
                            "{case} build {b_idx} block {name}"
                        );
                    }
                } else {
                    let (_, _, el, er, f, e) = rec
                        .blocks
                        .iter()
                        .find(|b| b.0 == b_idx && b.1 == j)
                        .unwrap();
                    if name.contains(":scl") {
                        assert_eq!(
                            *f, bits[0],
                            "{case} build {b_idx} block {j} scl f"
                        );
                        assert_eq!(
                            ext_e(&v[1]),
                            *e,
                            "{case} build {b_idx} block {j} scl e"
                        );
                    } else {
                        let got = if name.contains(":E_l") { el } else { er };
                        assert_eq!(
                            got.as_slice(),
                            bits.as_slice(),
                            "{case} build {b_idx} block {name}"
                        );
                    }
                }
            }
            _ => {}
        }
    }
    assert_eq!(rec.builds.len(), b_idx, "{case} build count");
    let mut d_t = Vec::new();
    let mut om: Vec<u64> = Vec::new();
    for (tag, name, v) in &taps {
        if tag == "DISCRIM" {
            if name == "omega" {
                om = hexs(v);
            } else if om.len() == 1 {
                d_t.push((om[0], 0, hexbits(&v[0]), 0, ext_e(&v[1]), false));
            } else {
                d_t.push((
                    om[0],
                    om[1],
                    hexbits(&v[0]),
                    hexbits(&v[1]),
                    ext_e(&v[2]),
                    true,
                ));
            }
        }
    }
    for (k, (a, b)) in rec.discrim_all.iter().zip(&d_t).enumerate() {
        assert_eq!(a, b, "{case} discrim {}", k + 1);
    }
    assert_eq!(rec.discrim_all.len(), d_t.len(), "{case} discrim count");

    for (kind, got) in [("MODE_AD", &rec.mode_ad), ("MODE_NAD", &rec.mode_nad)]
    {
        let mut want: Vec<([u64; 2], [i64; 5])> = Vec::new();
        let mut w_om = [0u64; 2];
        for (tag, name, v) in &taps {
            if tag == kind {
                if name == "omega" {
                    w_om = [hexbits(&v[0]), hexbits(&v[1])];
                } else {
                    let ints: Vec<i64> =
                        v.iter().map(|s| s.parse().unwrap()).collect();
                    want.push((w_om, ints.try_into().unwrap()));
                }
            }
        }
        assert_eq!(got, &want, "{case} {kind}");
    }

    let mut i_out = 0;
    let mut i_mode = 0;
    for f in &modes {
        match f[0].as_str() {
            "GET_MODES" => {}
            "MODE" => {
                let want: Vec<i64> =
                    f[2..].iter().map(|s| s.parse().unwrap()).collect();
                assert_eq!(
                    mode_ints[i_mode][..6],
                    want[..6],
                    "{case} MODE {}",
                    i_mode + 1
                );
                i_mode += 1;
            }
            tag => {
                let (key, vals) = if tag == "EIG" || tag == "COEFF" {
                    (format!("{tag} {} {}", f[1], f[2]), &f[4])
                } else {
                    (format!("{tag} {}", f[1]), &f[2])
                };
                let (gk, gv) = &out[i_out];
                assert_eq!(gk, &key, "{case} record order");
                if tag == "COEFF" {
                    let want: Vec<Option<u64>> = vals
                        .split(' ')
                        .map(|h| (h != "undefined").then(|| hexbits(h)))
                        .collect();
                    for (i, w) in want.iter().enumerate() {
                        if let Some(w) = w {
                            assert_eq!(
                                gv[i],
                                *w,
                                "{case} {key} coeff {}",
                                i + 1
                            );
                        }
                    }
                } else {
                    let want: Vec<u64> = vals.split(' ').map(hexbits).collect();
                    assert_eq!(gv, &want, "{case} {key}");
                }
                i_out += 1;
            }
        }
    }
    assert_eq!(i_mode, mode_ints.len());
    assert_eq!(i_out, out.len());

    // All captured calls in pulse_tap/modes.tsv.
    let text = std::fs::read_to_string(pulse_dir.join("modes.tsv"))
        .expect("modes.tsv");
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let col = |n: &str| head.iter().position(|h| *h == n).unwrap();
    let (c_eval, c_l, c_npg, c_np, c_ng) = (
        col("eval_id"),
        col("l"),
        col("n_pg"),
        col("n_p"),
        col("n_g"),
    );
    let (c_fu, c_fh, c_i, c_g, c_gh, c_file) = (
        col("freq_uHz_bits"),
        col("freq_Hz_bits"),
        col("inertia_bits"),
        col("growth_rate_rad_s_bits"),
        col("growth_rate_Hz_bits"),
        col("saved_pulse_file"),
    );
    let mut evals: BTreeMap<u32, (String, i32, Vec<[u64; 8]>)> =
        BTreeMap::new();
    let hb =
        |s: &str| u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap();
    for l in lines {
        if l.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = l.split('\t').collect();
        let e: u32 = f[c_eval].parse().unwrap();
        let row = [
            f[c_npg].parse::<i64>().unwrap() as u64,
            f[c_np].parse::<i64>().unwrap() as u64,
            f[c_ng].parse::<i64>().unwrap() as u64,
            hb(f[c_fu]),
            hb(f[c_fh]),
            hb(f[c_i]),
            hb(f[c_g]),
            hb(f[c_gh]),
        ];
        evals
            .entry(e)
            .or_insert_with(|| {
                (f[c_file].to_string(), f[c_l].parse().unwrap(), Vec::new())
            })
            .2
            .push(row);
    }
    let mut n_modes = 0usize;
    for (e, (file, l, want)) in &evals {
        let path = pulse_dir.join(file.trim_start_matches("pulse_tap/"));
        let (global, rows, version) = load_model(&path);
        gyre.set_model(&global, &rows, version).expect("set_model");
        let mut got: Vec<[u64; 8]> = Vec::new();
        gyre.get_modes(*l, &mut |md: &Mode<'_>| {
            got.push([
                md.n_pg as u64,
                md.n_p as u64,
                md.n_g as u64,
                md.freq("UHZ")?.re.to_bits(),
                md.freq("HZ")?.re.to_bits(),
                md.e_norm()?.to_bits(),
                md.freq("RAD_PER_SEC")?.im.to_bits(),
                md.freq("HZ")?.im.to_bits(),
            ]);
            Ok(0)
        })
        .expect("get_modes");
        assert_eq!(&got, want, "{case} eval {e}");
        n_modes += got.len();
    }
    assert_eq!(evals.len(), expected_evals, "{case} pulse_tap eval count");
    assert_eq!(n_modes, expected_modes, "{case} pulse_tap mode count");
}

#[test]
fn gyre_in_mesa_bcep_matches_g0_and_pulse_taps() {
    assert_case_g0_and_pulse_taps("gyre_in_mesa_bcep", 54, 192);
}

#[test]
fn gyre_in_mesa_envelope_matches_g0_and_pulse_taps() {
    assert_case_g0_and_pulse_taps("gyre_in_mesa_envelope", 5, 18);
}

#[test]
fn gyre_in_mesa_rsg_matches_g0_and_pulse_taps() {
    assert_case_g0_and_pulse_taps("gyre_in_mesa_rsg", 10, 50);
}

#[test]
fn gyre_in_mesa_wd_matches_g0_and_pulse_taps() {
    assert_case_g0_and_pulse_taps("gyre_in_mesa_wd", 52, 413);
}

#[test]
fn gyre_in_mesa_spb_matches_g0_and_pulse_taps() {
    assert_case_g0_and_pulse_taps("gyre_in_mesa_spb", 51, 511);
}

#[test]
fn rsp_gyre_matches_g0_and_pulse_taps() {
    assert_case_g0_and_pulse_taps("rsp_gyre", 1, 4);
}

/// Build an analytical stellar structure model (`version = 101`) with a smooth
/// radiative envelope (`As > 0`) and convective core (`As <= 0`) to exercise
/// `delta_g`, `INVERSE` scans, `MAGNUS_GL2`, `ZERO_R`/`ZERO_H`, `AdBvp`,
/// `find_turn`, dipole `Yt_1`/`Yt_2`, and segment-boundary matching completely
/// offline without external MESA caches.
fn synthetic_mesa_model(
    n_pts: usize,
    segmented: bool,
    x_min: f64,
    add_center: bool,
) -> (Vec<f64>, Vec<Vec<f64>>, i32, EvolModel, GyreConstants) {
    use flexmesa_const as c;
    let constants = standard_constants();
    let m_star = c::MSUN;
    let r_star = c::RSUN;
    let l_star = c::LSUN;
    let mut rows = Vec::with_capacity(n_pts);
    let to_f = |v: usize| f64::from(u32::try_from(v).expect("n_pts fits u32"));
    for idx_pt in 0..n_pts {
        let frac = if segmented && idx_pt >= n_pts / 2 {
            to_f(idx_pt - 1) / to_f(n_pts - 2)
        } else if segmented {
            to_f(idx_pt) / to_f(n_pts - 2)
        } else {
            to_f(idx_pt) / to_f(n_pts - 1)
        };
        let x_val = x_min + (1.0 - x_min) * frac;
        let rad = x_val * r_star;
        let x_m = x_val.max(1e-4);
        // Smooth mass-density-pressure structure with radiative exterior for As > 0.
        let m_r = m_star * (x_m * x_m * x_m) / (1.0 + 2.0 * x_m * x_m) * 3.0;
        let m_r = if x_val == 0.0 { 0.0 } else { m_r.min(m_star) };
        let m_r = if x_val >= 0.999_999_9 { m_star } else { m_r };
        let l_r = l_star * (0.2 + 0.8 * x_val * x_val);
        let pres = 1.0e17 * (1.0 - 0.9999 * x_val * x_val);
        let temp = 1.5e7 * (1.0 - 0.99 * x_val);
        let rho_base = if segmented && idx_pt >= n_pts / 2 {
            1.2
        } else {
            1.0
        };
        let rho = 100.0 * rho_base * (1.0 - 0.999 * x_val * x_val);
        let gamma_1 = 5.0 / 3.0;
        let nabla_ad = 0.4;
        let nabla = if x_val < 0.25 { 0.42 } else { 0.22 };
        let n2 = if x_val < 0.25 {
            0.0
        } else {
            5.0e-5 * (x_val - 0.25)
        };
        let ups_t = 1.0;
        let kap = 0.5;
        let eps = if x_val < 0.3 { 10.0 } else { 0.0 };
        rows.push(vec![
            rad,
            m_r,
            l_r,
            pres,
            temp,
            rho,
            nabla,
            n2,
            gamma_1,
            nabla_ad,
            ups_t,
            kap,
            -3.5 * kap,
            1.0 * kap,
            eps,
            4.0 * eps,
            1.0 * eps,
            0.0,
        ]);
    }
    let ml_p = ModelPar {
        add_center,
        ..ModelPar::default()
    };
    let ml = EvolModel::from_mesa(
        &ml_p,
        &[m_star, r_star, l_star],
        &rows,
        101,
        &constants,
    )
    .expect("from_mesa");
    (vec![m_star, r_star, l_star], rows, 101, ml, constants)
}

#[test]
fn standalone_offline_nonradial_magnus_turn_and_match_invariants() {
    flexmesa_math::math_init();

    // 1. Delta_g, GRAVITY_DELTA / UPPER_DELTA / LOWER_DELTA, and INVERSE scans.
    let (_gd, _rows, _ver, ml, k) = synthetic_mesa_model(65, false, 0.0, true);
    let md_p = ModePar {
        l: 1,
        ..ModePar::default()
    };
    let os_p = OscPar::default();
    let rt_p = RotPar::default();
    let cx = Context::new(&ml, &k, &ml.gr, &md_p, &os_p, &rt_p).unwrap();

    let dg = ml.delta_g(cx.pt_i.x, cx.pt_o.x, 2.0).unwrap();
    let dp = ml.delta_p(cx.pt_i.x, cx.pt_o.x).unwrap();
    assert!(dg > 0.0 && dp > 0.0);
    assert_eq!(
        cx.freq_scale("GRAVITY_DELTA").unwrap().to_bits(),
        (1.0 / (k.math.twopi * dg)).to_bits()
    );
    assert_eq!(
        cx.freq_scale("UPPER_DELTA").unwrap().to_bits(),
        (1.0 / (k.math.twopi * dp.max(dg))).to_bits()
    );
    assert_eq!(
        cx.freq_scale("LOWER_DELTA").unwrap().to_bits(),
        (1.0 / (k.math.twopi * dp.min(dg))).to_bits()
    );

    let sc_inv = ScanPar {
        freq_min: 0.2,
        freq_max: 1.0,
        n_freq: 17,
        freq_min_units: "GRAVITY_DELTA".into(),
        freq_max_units: "GRAVITY_DELTA".into(),
        grid_type: "INVERSE".into(),
        ..ScanPar::default()
    };
    let omega = build_scan(&cx, &[sc_inv], "REAL").unwrap();
    assert_eq!(omega.len(), 17);
    assert!(omega.windows(2).all(|w| w[0] < w[1]));
    // Spacing in period 1/omega must be uniform.
    let diffs: Vec<f64> =
        omega.windows(2).map(|w| 1.0 / w[0] - 1.0 / w[1]).collect();
    let max_dev = diffs
        .iter()
        .map(|d| (d - diffs[0]).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_dev < 1e-13 * diffs[0]);

    // Edge counts: n_freq < 1 returns empty, n_freq == 1 when equal pushes 1 point.
    let sc_zero = ScanPar {
        n_freq: 0,
        freq_min_units: "NONE".into(),
        freq_max_units: "NONE".into(),
        ..ScanPar::default()
    };
    assert_eq!(build_scan(&cx, &[sc_zero], "REAL").unwrap(), [] as [f64; 0]);
    let sc_one = ScanPar {
        freq_min: 2.0,
        freq_max: 2.0,
        n_freq: 1,
        freq_min_units: "NONE".into(),
        freq_max_units: "NONE".into(),
        ..ScanPar::default()
    };
    assert_eq!(build_scan(&cx, &[sc_one], "REAL").unwrap(), vec![2.0]);

    // 2. Eigendecomposition parity invariants (2x2 real/complex, 4x4 real, 6x6 complex).
    let (l2, vl2, vr2) = eigen_decompose_2_r(&[1.2, 0.4, -0.9, 0.8]);
    let d_check = vl2[0] * vr2[0] + vl2[2] * vr2[1];
    assert!((d_check.re - 1.0).abs() < 1e-14 && d_check.im.abs() < 1e-14);
    assert!((l2[0] + l2[1] - Cplx::new(2.0, 0.0)).re.abs() < 1e-14);

    let (l2c, vl2c, vr2c) = eigen_decompose_2_c(&[
        Cplx::new(1.0, 0.5),
        Cplx::new(-0.3, 0.2),
        Cplx::new(0.4, -0.1),
        Cplx::new(-0.2, 0.7),
    ]);
    assert!(
        ((vl2c[0] * vr2c[0] + vl2c[2] * vr2c[1]) - Cplx::new(1.0, 0.0))
            .re
            .abs()
            < 1e-14
    );
    let _ = l2c;

    let mut a4 = [
        -2.0, 0.3, 0.0, 0.1, -0.5, 1.5, 0.0, -0.2, 0.0, 0.0, -1.0, 0.4, 0.0,
        0.1, 0.5, 3.0,
    ];
    let decomp4 = eigen_decompose_r(4, &mut a4, true).unwrap();
    assert!(decomp4.lambda.windows(2).all(|w| w[0].re <= w[1].re));
    let mut a6 = [Cplx::new(0.0, 0.0); 36];
    for r in 0..6 {
        let rf = f64::from(u8::try_from(r).unwrap());
        a6[r + 6 * r] = Cplx::new(rf - 2.5, 0.1 * rf);
        if r + 1 < 6 {
            a6[r + 6 * (r + 1)] = Cplx::new(0.05, -0.02);
        }
    }
    let decomp6 = eigen_decompose_c(6, &mut a6, false).unwrap();
    for i in 0..6 {
        let mut norm = Cplx::new(0.0, 0.0);
        for k in 0..6 {
            norm = norm + decomp6.vl[i + 6 * k] * decomp6.vr[k + 6 * i];
        }
        assert!((norm - Cplx::new(1.0, 0.0)).re.abs() < 1e-12);
        assert!(norm.im.abs() < 1e-12);
    }

    // 3. AdEqns / AdBound / AdDiff (COLLOC_GL2, COLLOC_GL4, MAGNUS_GL2) + find_turn + AdBvp.
    let eq4 = AdEqns::new(&cx, &os_p).unwrap();
    let _ = &eq4;
    let nm_gl2 = NumPar {
        diff_scheme: "COLLOC_GL2".into(),
        ..NumPar::default()
    };
    let nm_gl4 = NumPar {
        diff_scheme: "COLLOC_GL4".into(),
        ..NumPar::default()
    };
    let nm_m2 = NumPar {
        diff_scheme: "MAGNUS_GL2".into(),
        ..NumPar::default()
    };
    let df_gl2 =
        AdDiff::new(&cx, ml.gr.pt[5], ml.gr.pt[6], &nm_gl2, &os_p).unwrap();
    let df_gl4 =
        AdDiff::new(&cx, ml.gr.pt[5], ml.gr.pt[6], &nm_gl4, &os_p).unwrap();
    let df_m2 =
        AdDiff::new(&cx, ml.gr.pt[5], ml.gr.pt[6], &nm_m2, &os_p).unwrap();
    let _ = (
        df_gl2.build(&cx, 1.5).unwrap(),
        df_gl4.build(&cx, 1.5).unwrap(),
        df_m2.build(&cx, 1.5).unwrap(),
    );

    let (j_turn, x_turn) = find_turn(&cx, &ml.gr, 1.5, &nm_gl2, &os_p).unwrap();
    assert!(j_turn >= 1 && j_turn <= ml.gr.n());
    assert!(x_turn >= ml.gr.pt_i().x && x_turn <= ml.gr.pt_o().x);

    // 4. Envelope model (x_min > 0, ZERO_R / ZERO_H / MAGNUS_GL2 / JCD).
    let (_gd_e, _rows_e, _v_e, ml_env, k_e) =
        synthetic_mesa_model(33, false, 0.1, false);
    let os_zero_r = OscPar {
        inner_bound: "ZERO_R".into(),
        outer_bound: "VACUUM".into(),
        ..OscPar::default()
    };
    let os_zero_r_jcd = OscPar {
        inner_bound: "ZERO_R".into(),
        outer_bound: "JCD".into(),
        ..OscPar::default()
    };
    let os_zero_h = OscPar {
        inner_bound: "ZERO_H".into(),
        ..OscPar::default()
    };
    let cx_zr =
        Context::new(&ml_env, &k_e, &ml_env.gr, &md_p, &os_zero_r, &rt_p)
            .unwrap();
    let cx_zr_jcd =
        Context::new(&ml_env, &k_e, &ml_env.gr, &md_p, &os_zero_r_jcd, &rt_p)
            .unwrap();
    let cx_zh =
        Context::new(&ml_env, &k_e, &ml_env.gr, &md_p, &os_zero_h, &rt_p)
            .unwrap();
    let _ = RadBound::new(&cx_zr_jcd, &os_zero_r_jcd)
        .unwrap()
        .build_i(2.0)
        .unwrap();
    let _ = RadDiff::new(
        &cx_zr_jcd,
        ml_env.gr.pt[0],
        ml_env.gr.pt[1],
        &nm_m2,
        &os_zero_r_jcd,
    )
    .unwrap()
    .build(2.0)
    .unwrap();
    let _ = NadBound::new(&cx_zr, &os_zero_r)
        .unwrap()
        .build_i(&cx_zr, Cplx::new(2.0, 0.01))
        .unwrap();
    let _ = NadBound::new(&cx_zh, &os_zero_h)
        .unwrap()
        .build_i(&cx_zh, Cplx::new(2.0, 0.01))
        .unwrap();
    let _ = NadDiff::new(
        &cx_zr,
        ml_env.gr.pt[0],
        ml_env.gr.pt[1],
        &nm_m2,
        &os_zero_r,
    )
    .unwrap()
    .build(&cx_zr, Cplx::new(2.0, 0.01))
    .unwrap();
    let bd_ad = AdBound::new(&cx_zr_jcd, &os_zero_r_jcd).unwrap();
    let _ = bd_ad.build_i(&cx_zr_jcd, 2.0).unwrap();
    let _ = bd_ad.build_o(&cx_zr_jcd, 2.0).unwrap();
    let _ = AdBound::new(&cx_zh, &os_zero_h)
        .unwrap()
        .build_i(&cx_zh, 2.0)
        .unwrap();

    // 5. Segmented grid match conditions (RadMatch, AdMatch, NadMatch).
    let (_gd_s, _rows_s, _v_s, ml_seg, k_s) =
        synthetic_mesa_model(34, true, 0.0, true);
    let idx_s = ml_seg
        .gr
        .pt
        .windows(2)
        .position(|w| w[0].s != w[1].s)
        .expect("segment boundary");
    let (pa, pb) = (ml_seg.gr.pt[idx_s], ml_seg.gr.pt[idx_s + 1]);
    let cx_s =
        Context::new(&ml_seg, &k_s, &ml_seg.gr, &md_p, &os_p, &rt_p).unwrap();
    let _ = RadMatch::new(&cx_s, pa, pb, &os_p)
        .unwrap()
        .build(2.0)
        .unwrap();
    let _ = AdMatch::new(&cx_s, pa, pb, &os_p).unwrap().build();
    let _ = NadMatch::new(&cx_s, pa, pb, &os_p).unwrap().build();

    // 6. Full AdBvp -> Wave -> Mode::classify on l=1 dipole & l=2 non-radial.
    let mut bp = AdBvp::new(&cx, &ml.gr, &md_p, &nm_m2, &os_p).unwrap();
    let wv: Wave<'_> = ad_wave_hom(&mut bp, 2.5, 1, &mut NoTap).unwrap();
    let _ = (
        wv.xi_r(10).unwrap(),
        wv.xi_h(10).unwrap(),
        wv.lag_p(10).unwrap(),
        wv.yt_1(10).unwrap(),
        wv.yt_2(10),
        wv.prop_type(10).unwrap(),
    );
    let _md1 = Mode::new(wv).unwrap();

    let md_p2 = ModePar {
        l: 2,
        ..ModePar::default()
    };
    let cx2 = Context::new(&ml, &k, &ml.gr, &md_p2, &os_p, &rt_p).unwrap();
    let mut bp2 = AdBvp::new(&cx2, &ml.gr, &md_p2, &nm_gl2, &os_p).unwrap();
    let wv2 = ad_wave_hom(&mut bp2, 2.5, 2, &mut NoTap).unwrap();
    let _ = (
        wv2.xi_r(10).unwrap(),
        wv2.xi_h(10).unwrap(),
        wv2.lag_p(10).unwrap(),
    );
    let _md2 = Mode::new(wv2).unwrap();

    // 7. End-to-end GyreMesa on synthetic model with l=1, GRAVITY_DELTA, INVERSE, COLLOC_GL2.
    let (gd, rows, ver, _, _) = synthetic_mesa_model(65, false, 0.0, true);
    let gin = r"
&model
/
&mode
  l = 1
/
&osc
/
&rot
/
&num
  diff_scheme = 'COLLOC_GL2'
/
&scan
  grid_type = 'INVERSE'
  freq_min = 0.05
  freq_max = 0.5
  freq_min_units = 'GRAVITY_DELTA'
  freq_max_units = 'GRAVITY_DELTA'
  n_freq = 40
/
&grid
  w_osc = 10
  w_exp = 2
  w_ctr = 10
/
";
    let mut g = new_gyre(gin);
    g.set_model(&gd, &rows, ver).unwrap();
    let mut found = 0usize;
    g.get_modes(1, &mut |md| {
        assert_eq!(md.wave.l, 1);
        assert!(md.freq("HZ").unwrap().re.is_finite());
        found += 1;
        Ok(0)
    })
    .unwrap();
    assert!(found > 0, "expected l=1 dipole modes on synthetic model");
}
