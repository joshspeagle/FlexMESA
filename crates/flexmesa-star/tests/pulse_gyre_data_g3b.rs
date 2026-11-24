// SPDX-License-Identifier: GPL-3.0-or-later
//! `get_gyre_data` against MESA 26.4.1 (#170, package G3b).
//!
//! The inputs are MESA's own star state at each GYRE `set_model` call of
//! the `astero_gyre` reference run, captured by
//! `benchmarks/fortran_mesa/gyre_pulse_data/run_capture.sh`. The expected
//! outputs are the `global_data`/`point_data` arrays MESA passed to GYRE in
//! those calls: in the capture files themselves when present, and in every
//! case the `pulse_tap/models/*.gyre` files of the `mesa_histories` bundle,
//! which hold the same arrays in `1PE26.16` (17 significant digits, so a
//! correctly rounded read recovers every bit).

#![allow(clippy::float_cmp, clippy::too_many_lines)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use flexmesa_star::pulse_gyre_data::{get_gyre_data, GyreDataInput};

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
    std::env::var_os("HOME").map_or_else(
        || PathBuf::from(".flexmesa_histories"),
        |h| PathBuf::from(h).join(".cache/flexmesa/mesa_histories"),
    )
}

struct Capture {
    recs: HashMap<String, Vec<String>>,
}

impl Capture {
    fn load(path: &Path) -> Self {
        let text = std::fs::read_to_string(path).expect("read capture");
        let mut recs = HashMap::new();
        for l in text
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
        {
            let f: Vec<&str> = l.split('\t').collect();
            let n: usize = f[1].parse().expect("count");
            let vals: Vec<String> =
                f[2].split_whitespace().map(str::to_string).collect();
            assert_eq!(vals.len(), n, "{}", f[0]);
            recs.insert(f[0].to_string(), vals);
        }
        Self { recs }
    }
    fn reals(&self, name: &str) -> Vec<f64> {
        self.recs[name]
            .iter()
            .map(|h| f64::from_bits(u64::from_str_radix(h, 16).expect("hex")))
            .collect()
    }
    fn real(&self, name: &str) -> f64 {
        self.reals(name)[0]
    }
    fn int(&self, name: &str) -> i32 {
        self.recs[name][0].parse().expect("int")
    }
    fn flag(&self, name: &str) -> bool {
        self.recs[name][0] == "T"
    }
}

fn parse_f(s: &str) -> f64 {
    s.replace(['D', 'd'], "e").parse().expect("real")
}

/// `.gyre` file: `n M R L version`, then `k` and the columns.
fn load_gyre(path: &Path) -> (Vec<f64>, Vec<Vec<f64>>) {
    let text = std::fs::read_to_string(path).expect("read .gyre");
    let mut lines = text.lines();
    let head: Vec<&str> =
        lines.next().expect("header").split_whitespace().collect();
    let n: usize = head[0].parse().expect("n");
    let global = head[1..4].iter().map(|v| parse_f(v)).collect();
    let rows = lines
        .take(n)
        .map(|l| l.split_whitespace().skip(1).map(parse_f).collect())
        .collect();
    (global, rows)
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn get_gyre_data_matches_mesa_on_every_astero_gyre_call() {
    flexmesa_math::math_init();
    let data = repo_root().join("benchmarks/fortran_mesa/gyre_pulse_data/data");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&data)
        .expect("capture directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "tsv"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no captures in {}", data.display());
    let models =
        histories_dir().join("runs/astero/astero_gyre/pulse_tap/models");
    let require = flexmesa_utils::require_zero_skips_enabled();
    let mut n_points = 0;
    for path in &files {
        let c = Capture::load(path);
        let arr: HashMap<&str, Vec<f64>> = [
            "r",
            "m",
            "m_grav",
            "dm",
            "dq",
            "rmid",
            "L",
            "Peos",
            "rho",
            "T",
            "mu",
            "brunt_N2",
            "grav",
            "cgrav",
            "gamma1",
            "grada",
            "chiT",
            "chiRho",
            "gradT",
            "opacity",
            "d_opacity_dlnT",
            "d_opacity_dlnd",
            "eps_nuc",
            "d_epsnuc_dlnT",
            "d_epsnuc_dlnd",
            "omega",
        ]
        .iter()
        .map(|&n| (n, c.reals(n)))
        .collect();
        let input = GyreDataInput {
            r: &arr["r"],
            m: &arr["m"],
            m_grav: &arr["m_grav"],
            dm: &arr["dm"],
            dq: &arr["dq"],
            rmid: &arr["rmid"],
            l: &arr["L"],
            peos: &arr["Peos"],
            rho: &arr["rho"],
            t: &arr["T"],
            mu: &arr["mu"],
            brunt_n2: &arr["brunt_N2"],
            grav: &arr["grav"],
            cgrav: &arr["cgrav"],
            gamma1: &arr["gamma1"],
            grada: &arr["grada"],
            chi_t: &arr["chiT"],
            chi_rho: &arr["chiRho"],
            grad_t: &arr["gradT"],
            opacity: &arr["opacity"],
            d_opacity_dlnt: &arr["d_opacity_dlnT"],
            d_opacity_dlnd: &arr["d_opacity_dlnd"],
            eps_nuc: &arr["eps_nuc"],
            d_epsnuc_dlnt: &arr["d_epsnuc_dlnT"],
            d_epsnuc_dlnd: &arr["d_epsnuc_dlnd"],
            omega: &arr["omega"],
            photosphere_r: c.real("photosphere_r"),
            gyre_data_schema: c.int("gyre_data_schema"),
            rotation_flag: c.flag("rotation_flag"),
            interpolate_rho_for_pulse_data: c
                .flag("interpolate_rho_for_pulse_data"),
            add_double_points_to_pulse_data: c
                .flag("add_double_points_to_pulse_data"),
            threshold_grad_mu_for_double_point: c
                .real("threshold_grad_mu_for_double_point"),
            max_number_of_double_points: c.int("max_number_of_double_points"),
            calculate_brunt_n2: c.flag("calculate_Brunt_N2"),
        };
        let (global, point) = get_gyre_data(
            &input,
            c.flag("add_center_point"),
            c.flag("keep_surface_point"),
            c.flag("add_atmosphere"),
        )
        .expect("get_gyre_data");
        let name = path.file_name().unwrap().to_string_lossy().to_string();

        // Against the arrays captured in the same call, when present.
        if c.recs.contains_key("point_data") {
            assert_eq!(
                bits(&global),
                bits(&c.reals("global_data")),
                "{name} global_data"
            );
            let flat: Vec<f64> = point.iter().flatten().copied().collect();
            assert_eq!(
                bits(&flat),
                bits(&c.reals("point_data")),
                "{name} point_data"
            );
        }

        // Against the .gyre file of the reference run.
        let eval: u32 = name["gyre_eval_".len().."gyre_eval_".len() + 6]
            .parse()
            .unwrap();
        let mn = c.int("model_number");
        let g = models.join(format!("gyre_eval_{eval:06}_step_{mn:07}.gyre"));
        if !g.exists() {
            assert!(
                !require,
                "missing MESA 26.4.1 reference pulse file {}",
                g.display()
            );
            continue;
        }
        let (g_global, g_rows) = load_gyre(&g);
        assert_eq!(
            bits(&global),
            bits(&g_global),
            "{name} global_data vs .gyre"
        );
        assert_eq!(point.len(), g_rows.len(), "{name} point count");
        for (k, (a, b)) in point.iter().zip(&g_rows).enumerate() {
            assert_eq!(bits(a), bits(b), "{name} point {}", k + 1);
        }
        n_points += point.len();
    }
    eprintln!("{} calls, {n_points} points compared", files.len());
}
