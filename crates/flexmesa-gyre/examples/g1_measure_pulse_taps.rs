// SPDX-License-Identifier: GPL-3.0-only
//! Package G1 measurement driver: replay every GYRE call that MESA 26.4.1
//! made in `astero/fast_from_file` and `star/gyre_in_mesa_wd` (the
//! `pulse_tap` captures of the `mesa-histories` bundle) through
//! [`GyreMesa::get_modes`] and count, per call and per quantity, the records
//! that match MESA at 0 ULP.
//!
//! The replay follows the existing tests:
//!
//! - `fast_from_file` as `tests/g3_fast_from_file.rs`
//!   (`all_calls_match_mesa_pulse_taps`): one [`GyreMesa`] for the whole run,
//!   `set_model` whenever the captured pulsation file changes, then
//!   `get_modes(l)` for every `(eval_id, l)` in `pulse_tap/modes.tsv`;
//! - `gyre_in_mesa_wd` as `tests/g5_gyre_in_mesa.rs`: `set_model` then
//!   `get_modes(l)` for every `eval_id`.
//!
//! Unlike those tests it does not stop at the first difference. It prints one
//! JSON object per case on stdout; `scripts/audit/gyre_adipls_measurement.py`
//! aggregates them into `notes/audit/gyre_adipls_measurement.{json,md}`.
//!
//! Usage: `cargo run --profile ci -p flexmesa-gyre --example
//! g1_measure_pulse_taps [-- <case> ...]` with `<case>` one of
//! `astero/fast_from_file` and `star/gyre_in_mesa_wd` (default: both).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use flexmesa_gyre::mesa::GyreMesa;
use flexmesa_gyre::mode::Mode;

/// Quantities compared per mode: `(name, is_integer)`.
///
/// Integer quantities count as matching only when equal; their "ULP" is the
/// absolute difference of the integers.
const QUANTITIES: [(&str, bool); 8] = [
    ("n_pg", true),
    ("n_p", true),
    ("n_g", true),
    ("freq_uHz", false),
    ("freq_Hz", false),
    ("inertia", false),
    ("growth_rate_rad_s", false),
    ("growth_rate_Hz", false),
];

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

fn parse_f(s: &str) -> f64 {
    s.replace(['D', 'd'], "e").parse().expect("real")
}

/// A `.gyre` pulsation file (`n M R L version`, then `k` and 18 columns).
fn load_model(path: &Path) -> (Vec<f64>, Vec<Vec<f64>>, i32) {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
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

/// Distance in units in the last place between two binary64 bit patterns
/// (ordered so that `-0` and `+0` are 0 apart).
fn ulp_distance(a: u64, b: u64) -> u64 {
    fn ordered(bits: u64) -> i64 {
        let s = bits.cast_signed();
        if s < 0 {
            i64::MIN - s
        } else {
            s
        }
    }
    ordered(a).abs_diff(ordered(b))
}

/// One captured call: `(eval_id, l)`, its pulsation file, MESA's modes.
struct Call {
    eval_id: u32,
    l: i32,
    file: String,
    want: Vec<[u64; 8]>,
}

fn read_calls(modes_tsv: &Path) -> Vec<Call> {
    let text = std::fs::read_to_string(modes_tsv)
        .unwrap_or_else(|e| panic!("read {}: {e}", modes_tsv.display()));
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let col = |n: &str| {
        head.iter()
            .position(|h| *h == n)
            .unwrap_or_else(|| panic!("column {n}"))
    };
    let c = [
        col("eval_id"),
        col("l"),
        col("n_pg"),
        col("n_p"),
        col("n_g"),
        col("freq_uHz_bits"),
        col("freq_Hz_bits"),
        col("inertia_bits"),
        col("growth_rate_rad_s_bits"),
        col("growth_rate_Hz_bits"),
        col("saved_pulse_file"),
    ];
    let hb =
        |s: &str| u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap();
    let mut by_key: BTreeMap<(u32, i32), Call> = BTreeMap::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        let e: u32 = f[c[0]].parse().unwrap();
        let l: i32 = f[c[1]].parse().unwrap();
        let row = [
            f[c[2]].parse::<i64>().unwrap() as u64,
            f[c[3]].parse::<i64>().unwrap() as u64,
            f[c[4]].parse::<i64>().unwrap() as u64,
            hb(f[c[5]]),
            hb(f[c[6]]),
            hb(f[c[7]]),
            hb(f[c[8]]),
            hb(f[c[9]]),
        ];
        by_key
            .entry((e, l))
            .or_insert_with(|| Call {
                eval_id: e,
                l,
                file: f[c[10]].to_string(),
                want: Vec::new(),
            })
            .want
            .push(row);
    }
    by_key.into_values().collect()
}

#[derive(Default)]
struct QStat {
    compared: usize,
    zero_ulp: usize,
    ulps: Vec<u64>,
    first: Option<String>,
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn measure_case(case: &str) -> String {
    let (gyre_in_rel, per_call_set_model) = match case {
        "astero/fast_from_file" => {
            ("astero/test_suite/fast_from_file/gyre.in", false)
        }
        "star/gyre_in_mesa_wd" => {
            ("star/test_suite/gyre_in_mesa_wd/gyre.in", true)
        }
        other => panic!("unknown case {other}"),
    };
    let gin_path =
        flexmesa_utils::resolve_mesa_test_inputs_dir().join(gyre_in_rel);
    let pulse_dir = histories_dir().join("runs").join(case).join("pulse_tap");
    let modes_tsv = pulse_dir.join("modes.tsv");
    if !gin_path.is_file() || !modes_tsv.is_file() {
        assert!(
            !flexmesa_utils::require_zero_skips_enabled(),
            "{case}: missing captures ({} or {}); run `flexmesa data fetch \
             --include-test-inputs` or set FLEXMESA_REQUIRE_ZERO_SKIPS=0",
            gin_path.display(),
            modes_tsv.display()
        );
        return format!(
            "{{\"case\":{},\"code\":\"GYRE\",\"status\":\"missing captures\"}}",
            json_str(case)
        );
    }
    let gin = std::fs::read_to_string(&gin_path).expect("read gyre.in");
    let calls = read_calls(&modes_tsv);

    let mut g = new_gyre(&gin);
    let mut loaded = String::new();
    let mut stats: Vec<QStat> =
        QUANTITIES.iter().map(|_| QStat::default()).collect();
    let mut n_modes_want = 0usize;
    let mut n_modes_got = 0usize;
    let mut calls_identical = 0usize;
    let mut calls_count_mismatch = 0usize;
    let mut calls_error = 0usize;
    let mut first_call: Option<String> = None;
    let mut evals = std::collections::BTreeSet::new();

    for call in &calls {
        evals.insert(call.eval_id);
        if per_call_set_model || call.file != loaded {
            let path =
                pulse_dir.join(call.file.trim_start_matches("pulse_tap/"));
            let (global, rows, version) = load_model(&path);
            g.set_model(&global, &rows, version).expect("set_model");
            loaded.clone_from(&call.file);
        }
        let mut got: Vec<[u64; 8]> = Vec::new();
        let res = g.get_modes(call.l, &mut |md: &Mode<'_>| {
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
        });
        n_modes_want += call.want.len();
        n_modes_got += got.len();
        let tag = format!("eval {} l={}", call.eval_id, call.l);
        if let Err(e) = res {
            calls_error += 1;
            first_call
                .get_or_insert_with(|| format!("{tag}: get_modes error {e}"));
            continue;
        }
        let mut identical = got.len() == call.want.len();
        if got.len() != call.want.len() {
            calls_count_mismatch += 1;
            first_call.get_or_insert_with(|| {
                format!("{tag}: {} modes, MESA {}", got.len(), call.want.len())
            });
        }
        for (k, (gr, wr)) in got.iter().zip(&call.want).enumerate() {
            for (q, (name, is_int)) in QUANTITIES.iter().enumerate() {
                let st = &mut stats[q];
                st.compared += 1;
                let (a, b) = (gr[q], wr[q]);
                if a == b {
                    st.zero_ulp += 1;
                    continue;
                }
                identical = false;
                let d = if *is_int {
                    (a as i64).abs_diff(b as i64)
                } else {
                    ulp_distance(a, b)
                };
                st.ulps.push(d);
                let msg = if *is_int {
                    format!(
                        "{tag} mode {}: {name} {} vs MESA {}",
                        k + 1,
                        a as i64,
                        b as i64
                    )
                } else {
                    format!(
                        "{tag} mode {}: {name} {:e} (0x{a:016x}) vs MESA {:e} \
                         (0x{b:016x}), {d} ULP",
                        k + 1,
                        f64::from_bits(a),
                        f64::from_bits(b)
                    )
                };
                st.first.get_or_insert_with(|| msg.clone());
                first_call.get_or_insert(msg);
            }
        }
        if identical {
            calls_identical += 1;
        }
    }

    let mut q_json = Vec::new();
    for ((name, is_int), st) in QUANTITIES.iter().zip(&mut stats) {
        st.ulps.sort_unstable();
        let max = st.ulps.last().copied().unwrap_or(0);
        let median = if st.ulps.is_empty() {
            0
        } else {
            st.ulps[st.ulps.len() / 2]
        };
        q_json.push(format!(
            "{{\"name\":{},\"integer\":{is_int},\"compared\":{},\
             \"zero_ulp\":{},\"differ\":{},\"max_ulp\":{max},\
             \"median_ulp_of_differing\":{median},\"first_difference\":{}}}",
            json_str(name),
            st.compared,
            st.zero_ulp,
            st.ulps.len(),
            st.first.as_deref().map_or_else(|| "null".into(), json_str)
        ));
    }
    format!(
        "{{\"case\":{},\"code\":\"GYRE\",\"status\":\"measured\",\
         \"evals\":{},\"calls\":{},\"calls_identical\":{calls_identical},\
         \"calls_mode_count_mismatch\":{calls_count_mismatch},\
         \"calls_error\":{calls_error},\"modes_mesa\":{n_modes_want},\
         \"modes_flexmesa\":{n_modes_got},\"first_differing_call\":{},\
         \"quantities\":[{}]}}",
        json_str(case),
        evals.len(),
        calls.len(),
        first_call
            .as_deref()
            .map_or_else(|| "null".into(), json_str),
        q_json.join(",")
    )
}

fn main() {
    flexmesa_math::math_init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cases: Vec<String> = if args.is_empty() {
        vec![
            "astero/fast_from_file".into(),
            "star/gyre_in_mesa_wd".into(),
        ]
    } else {
        args
    };
    for case in &cases {
        println!("{}", measure_case(case));
    }
}
