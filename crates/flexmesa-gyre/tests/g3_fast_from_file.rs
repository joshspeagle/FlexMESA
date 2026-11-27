// SPDX-License-Identifier: GPL-3.0-only
//! Bit-for-bit check of GYRE's adiabatic `JCD` path (`variables_set = 'JCD'`,
//! `outer_bound = 'JCD'`, `COLLOC_GL4`, `l = 0, 1, 2`) against MESA 26.4.1's
//! own GYRE 8.1 build for `astero/fast_from_file` (`#170`):
//!
//! - call 216 (`call_216_matches_g0_oracles`): intermediate taps and driver
//!   outputs committed by package G0
//!   (`benchmarks/fortran_mesa/gyre_oracles/data/astero__fast_from_file__e000216.*`),
//!   covering `l = 0` (`RadBvp`, `2 x 2`) and nonradial `l = 1, 2` (`AdBvp`,
//!   `4 x 4` with `JCD` variable and boundary transforms).
//! - all calls (`all_calls_match_mesa_pulse_taps`): every `(eval_id, l)` call
//!   in `pulse_tap/modes.tsv` of `astero/fast_from_file`.

#![forbid(unsafe_code)]
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

use flexmesa_gyre::ext_real::ExtR;
use flexmesa_gyre::fcomplex::Cplx;
use flexmesa_gyre::grid::Grid;
use flexmesa_gyre::mesa::GyreMesa;
use flexmesa_gyre::mode::Mode;
use flexmesa_gyre::model::idx;
use flexmesa_gyre::rad::GyreTap;

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

fn gyre_in() -> Option<String> {
    let p = require(
        flexmesa_utils::resolve_mesa_test_inputs_dir()
            .join("astero/test_suite/fast_from_file/gyre.in"),
        "MESA 26.4.1 fast_from_file gyre.in",
    )?;
    Some(std::fs::read_to_string(p).expect("read gyre.in"))
}

fn parse_f(s: &str) -> f64 {
    s.replace(['D', 'd'], "e").parse().expect("real")
}

/// A `.gyre` pulsation file (`n M R L version`, then `k` and 18 columns).
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

fn hexs(v: &[String]) -> Vec<u64> {
    v.iter().map(|h| hexbits(h)).collect()
}

fn cbits(z: Cplx) -> [u64; 2] {
    [z.re.to_bits(), z.im.to_bits()]
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

#[derive(Default)]
struct Rec {
    mode_pars: Vec<[i64; 2]>,
    grid_scans: Vec<(Vec<u64>, Vec<i64>, Vec<u64>)>,
    /// Every build `(n, mode_build, omega, B_i, B_o, is_ad)`.
    builds: Vec<(usize, bool, u64, Vec<u64>, Vec<u64>, bool)>,
    /// Blocks of dumped builds `(build_idx, j, E_l, E_r, scl_f, scl_e, is_ad)`.
    blocks: Vec<(usize, usize, Vec<u64>, Vec<u64>, u64, i32, bool)>,
    want_j: Vec<usize>,
    discrim: Vec<(u64, u64, i32)>,
    mode_ad: Vec<([u64; 2], [i64; 5])>,
}

impl Rec {
    fn push_block(
        &mut self,
        j: usize,
        e_l: &[f64],
        e_r: &[f64],
        scl: ExtR,
        is_ad: bool,
    ) {
        let b = self.builds.len();
        let dumped =
            b == 1 || self.builds[b - 1].1 || self.is_first_in_group(b);
        if dumped && self.want_j.contains(&j) {
            self.blocks.push((
                b,
                j,
                e_l.iter().copied().map(f64::to_bits).collect(),
                e_r.iter().copied().map(f64::to_bits).collect(),
                scl.f.to_bits(),
                scl.e,
                is_ad,
            ));
        }
    }

    fn is_first_in_group(&self, b: usize) -> bool {
        b >= 2 && self.builds[b - 2].5 != self.builds[b - 1].5
    }
}

impl GyreTap for Rec {
    fn mode_par(&mut self, l: i32, i: usize) {
        self.mode_pars.push([i64::from(l), i as i64]);
    }
    fn grid_scan(&mut self, gr: &Grid, omega_re: &[f64]) {
        self.grid_scans.push((
            gr.pt.iter().map(|p| p.x.to_bits()).collect(),
            gr.pt.iter().map(|p| i64::from(p.s)).collect(),
            omega_re.iter().map(|w| w.to_bits()).collect(),
        ));
    }
    fn build(
        &mut self,
        n: usize,
        mode_build: bool,
        omega: f64,
        b_i: &[f64; 2],
        b_o: &[f64; 2],
    ) {
        self.builds.push((
            n,
            mode_build,
            omega.to_bits(),
            b_i.iter().copied().map(f64::to_bits).collect(),
            b_o.iter().copied().map(f64::to_bits).collect(),
            false,
        ));
    }
    fn block(&mut self, j: usize, e_l: &[f64; 4], e_r: &[f64; 4], scl: ExtR) {
        self.push_block(j, e_l, e_r, scl, false);
    }
    fn build_ad(
        &mut self,
        n: usize,
        mode_build: bool,
        omega: f64,
        b_i: &[f64; 8],
        b_o: &[f64; 8],
    ) {
        self.builds.push((
            n,
            mode_build,
            omega.to_bits(),
            b_i.iter().copied().map(f64::to_bits).collect(),
            b_o.iter().copied().map(f64::to_bits).collect(),
            true,
        ));
    }
    fn block_ad(
        &mut self,
        j: usize,
        e_l: &[f64; 16],
        e_r: &[f64; 16],
        scl: ExtR,
    ) {
        self.push_block(j, e_l, e_r, scl, true);
    }
    fn discrim(&mut self, omega: f64, det: ExtR) {
        self.discrim.push((omega.to_bits(), det.f.to_bits(), det.e));
    }
    fn mode_ad(&mut self, omega: Cplx, ints: [i64; 5]) {
        self.mode_ad.push((cbits(omega), ints));
    }
}

fn call_216_matches_g0_oracles() {
    flexmesa_math::math_init();
    let Some(gin) = gyre_in() else { return };
    let Some(model) = require(
        histories_dir().join(
            "runs/astero/fast_from_file/pulse_tap/models/\
             gyre_eval_000216_step_0000431.gyre",
        ),
        "MESA 26.4.1 fast_from_file pulse_tap model",
    ) else {
        return;
    };
    let data = repo_root().join("benchmarks/fortran_mesa/gyre_oracles/data");
    let taps = read_tsv(&data.join("astero__fast_from_file__e000216.taps.tsv"));
    let modes: Vec<Vec<String>> = std::fs::read_to_string(
        data.join("astero__fast_from_file__e000216.modes.tsv"),
    )
    .expect("read modes oracle")
    .lines()
    .filter(|l| !l.starts_with('#') && !l.is_empty())
    .map(|l| l.split('\t').map(str::to_string).collect())
    .collect();

    let mut rec = Rec::default();
    for (tag, name, _) in &taps {
        if tag == "BLOCK" && name.contains(":scl[") {
            let j: usize = name[2..name.find(':').unwrap()].parse().unwrap();
            if !rec.want_j.contains(&j) {
                rec.want_j.push(j);
            }
        }
    }

    let mut g = new_gyre(&gin);
    let (global, rows, version) = load_model(&model);
    g.set_model(&global, &rows, version).expect("set_model");

    let mut out: Vec<(String, Vec<u64>)> = Vec::new();
    let mut mode_ints: Vec<Vec<i64>> = Vec::new();
    let stride = 64;
    let mut count = 0usize;
    for l in 0..=2 {
        let mut user =
            |md: &Mode<'_>| -> Result<i32, flexmesa_gyre::GyreError> {
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
                out.push((
                    format!("E_NORM {count}"),
                    vec![md.e_norm()?.to_bits()],
                ));
                let ml = w.cx.ml;
                for k in 1..=w.n {
                    if (k - 1) % stride != 0 && k != w.n {
                        continue;
                    }
                    let mut v = vec![w.x(k).to_bits()];
                    for i in 1..=6 {
                        v.extend(cbits(w.y_i(i, k)));
                    }
                    v.extend(cbits(w.xi_r(k)?));
                    v.extend(cbits(w.xi_h(k)?));
                    v.push(w.dw_dx(k)?.to_bits());
                    out.push((format!("EIG {count} {k}"), v));
                    let mut c = Vec::new();
                    for i in 1..=idx::LAST {
                        c.push(ml.coeff(i, w.gr.pt[k - 1])?.to_bits());
                    }
                    out.push((format!("COEFF {count} {k}"), c));
                }
                Ok(0)
            };
        g.get_modes_tapped(l, &mut user, &mut rec)
            .expect("get_modes");
    }

    // MODEPAR, GRID, SCAN per degree group.
    let mut i_mp = 0usize;
    let mut i_gs = 0usize;
    for (tag, name, v) in &taps {
        match (tag.as_str(), name.as_str()) {
            ("MODEPAR", _) => {
                let want = [
                    v[0].parse::<i64>().unwrap(),
                    v[1].parse::<i64>().unwrap(),
                ];
                assert_eq!(rec.mode_pars[i_mp], want, "MODEPAR {i_mp}");
                i_mp += 1;
            }
            ("GRID", "x") => {
                assert_eq!(rec.grid_scans[i_gs].0, hexs(v), "GRID {i_gs} x");
            }
            ("GRID", n) if n.starts_with('s') => {
                let want: Vec<i64> =
                    v.iter().map(|s| s.parse::<i64>().unwrap()).collect();
                assert_eq!(rec.grid_scans[i_gs].1, want, "GRID {i_gs} s");
            }
            ("SCAN", _) => {
                assert_eq!(rec.grid_scans[i_gs].2, hexs(v), "SCAN {i_gs}");
                i_gs += 1;
            }
            _ => {}
        }
    }
    assert_eq!(i_mp, 3);
    assert_eq!(i_gs, 3);

    // BUILD, BOUND, and dumped BLOCKs in sequential order across all three calls.
    let mut b_idx = 0usize;
    let mut bound: Vec<Vec<u64>> = Vec::new();
    let one_scl = 1.0_f64.to_bits();
    for (tag, name, v) in &taps {
        match tag.as_str() {
            "BUILD" if name.starts_with("index") => {
                b_idx += 1;
                assert!(b_idx <= rec.builds.len(), "build {b_idx} missing");
                let (n, mb, ..) = &rec.builds[b_idx - 1];
                assert_eq!(
                    *n,
                    v[1].parse::<usize>().unwrap(),
                    "build {b_idx} n"
                );
                assert_eq!(*mb, v[2] == "1", "build {b_idx} mode_build");
            }
            "BUILD" => {
                assert_eq!(
                    rec.builds[b_idx - 1].2,
                    hexbits(&v[0]),
                    "build {b_idx} omega"
                );
            }
            "BOUND" => {
                bound.push(hexs(v));
                if name == "scl_o" {
                    let (_, _, _, bi, bo, is_ad) = &rec.builds[b_idx - 1];
                    let expected_scl = if *is_ad {
                        vec![one_scl, one_scl]
                    } else {
                        vec![one_scl]
                    };
                    assert_eq!(&bound[0], bi, "build {b_idx} B_i");
                    assert_eq!(bound[1], expected_scl, "build {b_idx} scl_i");
                    assert_eq!(&bound[2], bo, "build {b_idx} B_o");
                    assert_eq!(bound[3], expected_scl, "build {b_idx} scl_o");
                    bound.clear();
                }
            }
            "BLOCK" => {
                let j: usize =
                    name[2..name.find(':').unwrap()].parse().unwrap();
                let bits = hexs(v);
                if let Some((_, _, el, er, f, e, _is_ad)) =
                    rec.blocks.iter().find(|b| b.0 == b_idx && b.1 == j)
                {
                    if name.contains(":scl") {
                        assert_eq!(
                            *f, bits[0],
                            "build {b_idx} block {j} scl f"
                        );
                        assert_eq!(
                            ext_e(&v[1]),
                            *e,
                            "build {b_idx} block {j} scl e"
                        );
                    } else {
                        let got = if name.contains(":E_l") { el } else { er };
                        assert_eq!(
                            got.as_slice(),
                            bits.as_slice(),
                            "build {b_idx} block {name}"
                        );
                    }
                }
            }
            _ => {}
        }
    }
    assert_eq!(rec.builds.len(), b_idx, "number of builds");

    // DISCRIM
    let mut d_t = Vec::new();
    let mut om = 0u64;
    for (tag, name, v) in &taps {
        if tag == "DISCRIM" {
            if name == "omega" {
                om = hexbits(&v[0]);
            } else {
                d_t.push((om, hexbits(&v[0]), ext_e(&v[1])));
            }
        }
    }
    assert_eq!(rec.discrim.len(), d_t.len(), "discrim count");
    for (k, (a, b)) in rec.discrim.iter().zip(&d_t).enumerate() {
        assert_eq!(a, b, "discriminant evaluation {}", k + 1);
    }

    // MODE_AD
    let mut want_mad: Vec<([u64; 2], [i64; 5])> = Vec::new();
    let mut w_om = [0u64; 2];
    for (tag, name, v) in &taps {
        if tag == "MODE_AD" {
            if name == "omega" {
                w_om = [hexbits(&v[0]), hexbits(&v[1])];
            } else {
                let ints: Vec<i64> =
                    v.iter().map(|s| s.parse().unwrap()).collect();
                want_mad.push((w_om, ints.try_into().unwrap()));
            }
        }
    }
    assert_eq!(rec.mode_ad, want_mad, "MODE_AD");

    // modes.tsv
    let mut i_out = 0usize;
    let mut i_mode = 0usize;
    for f in &modes {
        match f[0].as_str() {
            "GET_MODES" => {}
            "MODE" => {
                let want: Vec<i64> =
                    f[2..].iter().map(|s| s.parse().unwrap()).collect();
                assert_eq!(mode_ints[i_mode], want, "MODE {}", i_mode + 1);
                i_mode += 1;
            }
            tag => {
                let (key, vals) = if tag == "EIG" || tag == "COEFF" {
                    (format!("{tag} {} {}", f[1], f[2]), &f[4])
                } else {
                    (format!("{tag} {}", f[1]), &f[2])
                };
                let (gk, gv) = &out[i_out];
                assert_eq!(gk, &key, "record order");
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
                                "{key} coefficient {}",
                                i + 1
                            );
                        }
                    }
                } else {
                    let want: Vec<u64> = vals.split(' ').map(hexbits).collect();
                    assert_eq!(gv, &want, "{key}");
                }
                i_out += 1;
            }
        }
    }
    assert_eq!(i_mode, 23);
    assert_eq!(i_mode, mode_ints.len());
    assert_eq!(i_out, out.len());
}

fn all_calls_match_mesa_pulse_taps() {
    flexmesa_math::math_init();
    let Some(gin) = gyre_in() else { return };
    let Some(dir) = require(
        histories_dir().join("runs/astero/fast_from_file/pulse_tap"),
        "MESA 26.4.1 fast_from_file pulse_tap captures",
    ) else {
        return;
    };
    let text =
        std::fs::read_to_string(dir.join("modes.tsv")).expect("modes.tsv");
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let col = |n: &str| {
        head.iter()
            .position(|h| *h == n)
            .unwrap_or_else(|| panic!("column {n}"))
    };
    let (c_eval, c_l, c_npg, c_np, c_ng) = (
        col("eval_id"),
        col("l"),
        col("n_pg"),
        col("n_p"),
        col("n_g"),
    );
    let (c_fu, c_fh, c_i, c_file) = (
        col("freq_uHz_bits"),
        col("freq_Hz_bits"),
        col("inertia_bits"),
        col("saved_pulse_file"),
    );
    // (eval_id, l) -> (model file, expected rows)
    let mut evals: BTreeMap<(u32, i32), (String, Vec<[u64; 6]>)> =
        BTreeMap::new();
    for line in lines {
        let f: Vec<&str> = line.split('\t').collect();
        let e: u32 = f[c_eval].parse().unwrap();
        let l: i32 = f[c_l].parse().unwrap();
        let hb = |s: &str| {
            u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap()
        };
        let row = [
            f[c_npg].parse::<i64>().unwrap() as u64,
            f[c_np].parse::<i64>().unwrap() as u64,
            f[c_ng].parse::<i64>().unwrap() as u64,
            hb(f[c_fu]),
            hb(f[c_fh]),
            hb(f[c_i]),
        ];
        evals
            .entry((e, l))
            .or_insert_with(|| (f[c_file].to_string(), Vec::new()))
            .1
            .push(row);
    }
    let mut g = new_gyre(&gin);
    let mut loaded_file = String::new();
    let mut n_modes = 0usize;
    for ((e, l), (file, want)) in &evals {
        if *file != loaded_file {
            let path = dir.join(file.trim_start_matches("pulse_tap/"));
            let (global, rows, version) = load_model(&path);
            g.set_model(&global, &rows, version).expect("set_model");
            loaded_file.clone_from(file);
        }
        let mut got: Vec<[u64; 6]> = Vec::new();
        let mut user =
            |md: &Mode<'_>| -> Result<i32, flexmesa_gyre::GyreError> {
                got.push([
                    md.n_pg as u64,
                    md.n_p as u64,
                    md.n_g as u64,
                    md.freq("UHZ")?.re.to_bits(),
                    md.freq("HZ")?.re.to_bits(),
                    md.e_norm()?.to_bits(),
                ]);
                Ok(0)
            };
        g.get_modes(*l, &mut user).expect("get_modes");
        assert_eq!(&got, want, "eval {e} l={l}");
        n_modes += got.len();
    }
    assert_eq!(evals.len(), 702);
    assert_eq!(n_modes, 5508);
}

#[test]
fn fast_from_file_matches_mesa_g0_oracles_then_all_pulse_taps() {
    call_216_matches_g0_oracles();
    all_calls_match_mesa_pulse_taps();
}
