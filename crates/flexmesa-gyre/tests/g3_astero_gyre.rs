// SPDX-License-Identifier: GPL-3.0-only
//! Bit-for-bit check of the GYRE radial adiabatic path (`#170`, package G3)
//! against MESA 26.4.1's own GYRE 8.1 build, for the calls the
//! `astero_gyre` test case makes.
//!
//! - call 1 (`call_1_matches_g0_oracles`): the intermediate taps and
//!   driver outputs of call 1 committed by package G0
//!   (`benchmarks/fortran_mesa/gyre_oracles/data/astero__astero_gyre__e000001.*`):
//!   refined grid and scan, every boundary row and the sampled difference
//!   blocks of the dumped builds, every discriminant (scan points and
//!   Brent iterates), the mode frequencies, classifications and inertias,
//!   and eigenfunction and structure-coefficient samples.
//! - all calls (`all_calls_match_mesa_pulse_taps`): every mode of the ten
//!   GYRE calls captured in the reference run (`pulse_tap/modes.tsv` of the
//!   `mesa_histories` bundle): `n_pg`, `n_p`, `n_g`, frequency, inertia and
//!   growth rate as MESA's `gyre_call_back` stores them.

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
            .join("astero/test_suite/astero_gyre/gyre.in"),
        "MESA 26.4.1 astero_gyre gyre.in",
    )?;
    Some(std::fs::read_to_string(p).expect("read gyre.in"))
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

fn parse_f(s: &str) -> f64 {
    s.replace(['D', 'd'], "e").parse().expect("real")
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

/// Records in file order: `(tag, name, values)`.
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
    grid_x: Vec<u64>,
    grid_s: Vec<i64>,
    scan: Vec<u64>,
    /// Every build: (n, mode_build, omega, B_i, B_o).
    builds: Vec<(usize, bool, u64, [u64; 2], [u64; 2])>,
    /// Blocks of dumped builds: (build index, j, E_l, E_r, scl f, scl e).
    blocks: Vec<(usize, usize, [u64; 4], [u64; 4], u64, i32)>,
    want_j: Vec<usize>,
    discrim: Vec<(u64, u64, i32)>,
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
        self.builds.push((
            n,
            mode_build,
            omega.to_bits(),
            [b_i[0].to_bits(), b_i[1].to_bits()],
            [b_o[0].to_bits(), b_o[1].to_bits()],
        ));
    }
    fn block(&mut self, j: usize, e_l: &[f64; 4], e_r: &[f64; 4], scl: ExtR) {
        let b = self.builds.len();
        let dumped = b == 1 || self.builds[b - 1].1;
        if dumped && self.want_j.contains(&j) {
            self.blocks.push((
                b,
                j,
                e_l.map(f64::to_bits),
                e_r.map(f64::to_bits),
                scl.f.to_bits(),
                scl.e,
            ));
        }
    }
    fn discrim(&mut self, omega: f64, det: ExtR) {
        self.discrim.push((omega.to_bits(), det.f.to_bits(), det.e));
    }
}

fn cbits(z: Cplx) -> [u64; 2] {
    [z.re.to_bits(), z.im.to_bits()]
}

fn call_1_matches_g0_oracles() {
    flexmesa_math::math_init();
    let Some(gin) = gyre_in() else { return };
    let Some(model) = require(
        histories_dir().join("runs/astero/astero_gyre/pulse_tap/models/gyre_eval_000001_step_0000050.gyre"),
        "MESA 26.4.1 astero_gyre pulse_tap model",
    ) else {
        return;
    };
    let data = repo_root().join("benchmarks/fortran_mesa/gyre_oracles/data");
    let taps = read_tsv(&data.join("astero__astero_gyre__e000001.taps.tsv"));
    let modes: Vec<Vec<String>> = std::fs::read_to_string(
        data.join("astero__astero_gyre__e000001.modes.tsv"),
    )
    .expect("read modes oracle")
    .lines()
    .filter(|l| !l.starts_with('#') && !l.is_empty())
    .map(|l| l.split('\t').map(str::to_string).collect())
    .collect();

    // Which blocks were dumped.
    let mut rec = Rec::default();
    for (tag, name, _) in &taps {
        if tag == "BLOCK" && name.ends_with(":scl[ext:f,e]") {
            let j: usize = name[2..name.find(':').unwrap()].parse().unwrap();
            if !rec.want_j.contains(&j) {
                rec.want_j.push(j);
            }
        }
    }

    let mut g = new_gyre(&gin);
    let (global, rows, version) = load_model(&model);
    g.set_model(&global, &rows, version).expect("set_model");

    // Driver outputs (gyre_oracle_driver.f90 user_sub), eig_stride 64.
    let mut out: Vec<(String, Vec<u64>)> = Vec::new();
    let mut mode_ints: Vec<Vec<i64>> = Vec::new();
    let stride = 64;
    let mut count = 0;
    {
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
        g.get_modes_tapped(0, &mut user, &mut rec)
            .expect("get_modes");
    }

    let mut checked: BTreeMap<&str, usize> = BTreeMap::new();
    let mut bump = |k: &'static str| *checked.entry(k).or_insert(0) += 1;

    // GRID / SCAN
    for (tag, name, v) in &taps {
        match (tag.as_str(), name.as_str()) {
            ("GRID", "x") => {
                assert_eq!(
                    rec.grid_x,
                    v.iter().map(|h| hexbits(h)).collect::<Vec<_>>(),
                    "GRID x"
                );
                bump("GRID");
            }
            ("GRID", n) if n.starts_with('s') => {
                assert_eq!(
                    rec.grid_s,
                    v.iter()
                        .map(|s| s.parse::<i64>().unwrap())
                        .collect::<Vec<_>>()
                );
            }
            ("SCAN", _) => {
                assert_eq!(
                    rec.scan,
                    v.iter().map(|h| hexbits(h)).collect::<Vec<_>>(),
                    "SCAN"
                );
                bump("SCAN");
            }
            _ => {}
        }
    }

    // BUILD / BOUND / BLOCK in build order
    let mut b_idx = 0usize;
    let mut bound: Vec<u64> = Vec::new();
    for (tag, name, v) in &taps {
        match tag.as_str() {
            "BUILD" if name.starts_with("index") => {
                b_idx = v[0].parse().unwrap();
                let (n, mb, ..) = rec.builds[b_idx - 1];
                assert_eq!(
                    n,
                    v[1].parse::<usize>().unwrap(),
                    "build {b_idx} n"
                );
                assert_eq!(mb, v[2] == "1", "build {b_idx} mode_build");
                bump("BUILD");
            }
            "BUILD" => {
                assert_eq!(
                    rec.builds[b_idx - 1].2,
                    hexbits(&v[0]),
                    "build {b_idx} omega"
                );
            }
            "BOUND" => {
                let bits: Vec<u64> = v.iter().map(|h| hexbits(h)).collect();
                bound.extend(bits);
                if name == "scl_o" {
                    let (_, _, _, bi, bo) = rec.builds[b_idx - 1];
                    assert_eq!(&bound[0..2], &bi, "build {b_idx} B_i");
                    assert_eq!(bound[2], 1.0_f64.to_bits(), "scl_i");
                    assert_eq!(&bound[3..5], &bo, "build {b_idx} B_o");
                    assert_eq!(bound[5], 1.0_f64.to_bits(), "scl_o");
                    bound.clear();
                    bump("BOUND");
                }
            }
            "BLOCK" if name.ends_with(":scl[ext:f,e]") => {
                let j: usize =
                    name[2..name.find(':').unwrap()].parse().unwrap();
                let (_, _, _, _, f, e) = rec
                    .blocks
                    .iter()
                    .find(|b| b.0 == b_idx && b.1 == j)
                    .unwrap_or_else(|| {
                        panic!("no block {j} for build {b_idx}")
                    });
                assert_eq!(*f, hexbits(&v[0]), "block {j} scl f");
                assert_eq!(
                    f64::from(*e).to_bits(),
                    hexbits(&v[1]),
                    "block {j} scl e"
                );
                bump("BLOCK scl");
            }
            "BLOCK" => {
                let j: usize =
                    name[2..name.find(':').unwrap()].parse().unwrap();
                let (_, _, el, er, _, _) = rec
                    .blocks
                    .iter()
                    .find(|b| b.0 == b_idx && b.1 == j)
                    .unwrap_or_else(|| {
                        panic!("no block {j} for build {b_idx}")
                    });
                let bits: Vec<u64> = v.iter().map(|h| hexbits(h)).collect();
                let got = if name.contains(":E_l") { el } else { er };
                assert_eq!(
                    got.as_slice(),
                    bits.as_slice(),
                    "build {b_idx} block {name}"
                );
                bump("BLOCK E");
            }
            _ => {}
        }
    }
    assert_eq!(rec.builds.len(), b_idx, "number of builds");

    // DISCRIM: every evaluation in order
    let mut d_t = Vec::new();
    let mut om = 0u64;
    for (tag, name, v) in &taps {
        if tag == "DISCRIM" {
            if name == "omega" {
                om = hexbits(&v[0]);
            } else {
                let e = f64::from_bits(hexbits(&v[1]));
                d_t.push((om, hexbits(&v[0]), e as i32));
            }
        }
    }
    assert_eq!(
        rec.discrim.len(),
        d_t.len(),
        "number of discriminant evaluations"
    );
    for (k, (a, b)) in rec.discrim.iter().zip(&d_t).enumerate() {
        assert_eq!(a, b, "discriminant evaluation {}", k + 1);
    }
    for _ in 0..d_t.len() {
        bump("DISCRIM");
    }

    // modes.tsv
    let mut i_out = 0;
    let mut i_mode = 0;
    for f in &modes {
        match f[0].as_str() {
            "GET_MODES" => {}
            "MODE" => {
                // MODE idx l m n_p n_g n_pg n id
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
                let want: Vec<u64> = vals.split(' ').map(hexbits).collect();
                assert_eq!(gv, &want, "{key}");
                i_out += 1;
                bump("modes.tsv records");
            }
        }
    }
    assert_eq!(i_mode, mode_ints.len());
    assert_eq!(i_out, out.len());
    eprintln!("checked: {checked:?}");
}

fn all_calls_match_mesa_pulse_taps() {
    flexmesa_math::math_init();
    let Some(gin) = gyre_in() else { return };
    let Some(dir) = require(
        histories_dir().join("runs/astero/astero_gyre/pulse_tap"),
        "MESA 26.4.1 astero_gyre pulse_tap captures",
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
    let (c_f, c_i, c_g, c_file) = (
        col("freq_uHz_bits"),
        col("inertia_bits"),
        col("growth_rate_rad_s_bits"),
        col("saved_pulse_file"),
    );
    // eval -> (model file, l, expected rows)
    let mut evals: BTreeMap<u32, (String, i32, Vec<[u64; 6]>)> =
        BTreeMap::new();
    for l in lines {
        let f: Vec<&str> = l.split('\t').collect();
        let e: u32 = f[c_eval].parse().unwrap();
        let hb = |s: &str| {
            u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap()
        };
        let row = [
            f[c_npg].parse::<i64>().unwrap() as u64,
            f[c_np].parse::<i64>().unwrap() as u64,
            f[c_ng].parse::<i64>().unwrap() as u64,
            hb(f[c_f]),
            hb(f[c_i]),
            hb(f[c_g]),
        ];
        evals
            .entry(e)
            .or_insert_with(|| {
                (f[c_file].to_string(), f[c_l].parse().unwrap(), Vec::new())
            })
            .2
            .push(row);
    }
    let mut g = new_gyre(&gin);
    let mut n_modes = 0;
    for (e, (file, l, want)) in &evals {
        let path = dir.join(file.trim_start_matches("pulse_tap/"));
        let (global, rows, version) = load_model(&path);
        g.set_model(&global, &rows, version).expect("set_model");
        let mut got: Vec<[u64; 6]> = Vec::new();
        let mut user =
            |md: &Mode<'_>| -> Result<i32, flexmesa_gyre::GyreError> {
                // gyre_support.f90 gyre_call_back
                got.push([
                    md.n_pg as u64,
                    md.n_p as u64,
                    md.n_g as u64,
                    md.freq("UHZ")?.re.to_bits(),
                    md.e_norm()?.to_bits(),
                    md.freq("RAD_PER_SEC")?.im.to_bits(),
                ]);
                Ok(0)
            };
        g.get_modes(*l, &mut user).expect("get_modes");
        assert_eq!(&got, want, "eval {e}");
        n_modes += got.len();
    }
    assert_eq!(n_modes, 92);
}

/// Both checks run in one test, in this order, because mode identifiers come
/// from upstream's process-wide counter `id_m`: the G0 oracle ran call 1 in a
/// fresh process, so it must be the first GYRE search in this one.
#[test]
fn astero_gyre_matches_mesa_g0_oracles_then_all_pulse_taps() {
    call_1_matches_g0_oracles();
    all_calls_match_mesa_pulse_taps();
}
