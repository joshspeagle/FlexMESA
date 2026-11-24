// SPDX-License-Identifier: GPL-3.0-or-later
//! `FlexMESA`'s GYRE, FGONG and OSC export against MESA 26.4.1 on MESA's own
//! star state (#491).
//!
//! `benchmarks/fortran_mesa/pulse_export_tap/` dumps every `star_info`
//! field the producers read before each `export_pulse_data` call of a MESA
//! run and keeps the file MESA wrote. This test rebuilds the producer input
//! from each dump, runs `get_*_data` and `write_*_data`, and compares the
//! text byte for byte with MESA's. The tap also has MESA write OSC on every
//! tapped state. `FLEXMESA_PULSE_TAP_DIR` names the directory holding
//! `call_NNNN.{dump,flags,out,osc}`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::MixingType;
use flexmesa_eos::EosConfig;
use flexmesa_star::pulse_fgong::{
    get_fgong_data, PulseDataInput, PulseIsoIndices,
};
use flexmesa_star::pulse_gyre_data::{get_gyre_data, GyreDataInput};
use flexmesa_star::pulse_osc::get_osc_data;
use flexmesa_star::pulse_write::{
    write_fgong_data, write_gyre_data, write_osc_data,
    DEFAULT_FORMAT_FOR_OSC_DATA,
};
use flexmesa_star::{default_fgong_header, MESA_VERSION_NUMBER};

/// One tapped `export_pulse_data` call.
struct Dump {
    format: String,
    ints: HashMap<String, i64>,
    bools: HashMap<String, bool>,
    arrays: HashMap<String, Vec<f64>>,
    isos: Vec<String>,
    mixing_type: Vec<MixingType>,
}

fn parse_dump(path: &Path) -> Dump {
    let text = fs::read_to_string(path).expect("read dump");
    let mut lines = text.lines();
    let mut d = Dump {
        format: String::new(),
        ints: HashMap::new(),
        bools: HashMap::new(),
        arrays: HashMap::new(),
        isos: Vec::new(),
        mixing_type: Vec::new(),
    };
    while let Some(line) = lines.next() {
        let mut parts = line.splitn(2, ' ');
        let key = parts.next().unwrap_or_default();
        let rest = parts.next().unwrap_or_default().trim();
        match key {
            "format" => rest.clone_into(&mut d.format),
            "iso" => d.isos.push(rest.to_owned()),
            "mixing_type" => {
                let row = lines.next().expect("mixing_type row");
                d.mixing_type = row
                    .split_whitespace()
                    .map(|t| {
                        MixingType::from_code(t.parse().expect("int"))
                            .expect("mixing code")
                    })
                    .collect();
            }
            "rotation_flag"
            | "interpolate_rho_for_pulse_data"
            | "add_double_points_to_pulse_data"
            | "calculate_Brunt_N2" => {
                d.bools.insert(key.to_owned(), rest == "T");
            }
            _ => {
                let n: i64 = rest.parse().expect("count or int");
                let is_array = !matches!(
                    key,
                    "nz" | "species"
                        | "model_number"
                        | "gyre_data_schema"
                        | "fgong_ivers"
                        | "max_number_of_double_points"
                        | "largest_conv_mixing_region_bottom"
                );
                if is_array {
                    let v: Vec<f64> = (0..n)
                        .map(|_| {
                            let h = lines.next().expect("value");
                            f64::from_bits(
                                u64::from_str_radix(h.trim(), 16).expect("hex"),
                            )
                        })
                        .collect();
                    d.arrays.insert(key.to_owned(), v);
                } else {
                    d.ints.insert(key.to_owned(), n);
                }
            }
        }
    }
    d
}

/// `(format, matches)` for the call's own format and for OSC.
#[allow(clippy::too_many_lines)]
fn run_call(
    dir: &Path,
    n: usize,
    eos: &EosConfig,
) -> Option<Vec<(String, bool)>> {
    let dump_path = dir.join(format!("call_{n:04}.dump"));
    if !dump_path.is_file() {
        return None;
    }
    let d = parse_dump(&dump_path);
    let flags = fs::read_to_string(dir.join(format!("call_{n:04}.flags")))
        .expect("flags");
    let flag = |name: &str| {
        flags
            .lines()
            .find_map(|l| l.strip_prefix(name))
            .is_some_and(|v| v.trim() == "1")
    };
    let expected = fs::read_to_string(dir.join(format!("call_{n:04}.out")))
        .expect("MESA output");
    let expected_osc =
        fs::read_to_string(dir.join(format!("call_{n:04}.osc"))).ok();
    let a = |k: &str| d.arrays[k].as_slice();
    let scalar = |k: &str| d.arrays[k][0];
    let species = d.isos.len();
    let nz = usize::try_from(d.ints["nz"]).expect("usize");
    let base = GyreDataInput {
        r: a("r"),
        m: a("m"),
        m_grav: a("m_grav"),
        dm: a("dm"),
        dq: a("dq"),
        rmid: a("rmid"),
        l: a("L"),
        peos: a("Peos"),
        rho: a("rho"),
        t: a("T"),
        mu: a("mu"),
        brunt_n2: a("brunt_N2"),
        grav: a("grav"),
        cgrav: a("cgrav"),
        gamma1: a("gamma1"),
        grada: a("grada"),
        chi_t: a("chiT"),
        chi_rho: a("chiRho"),
        grad_t: a("gradT"),
        opacity: a("opacity"),
        d_opacity_dlnt: a("d_opacity_dlnT"),
        d_opacity_dlnd: a("d_opacity_dlnd"),
        eps_nuc: a("eps_nuc"),
        d_epsnuc_dlnt: a("d_epsnuc_dlnT"),
        d_epsnuc_dlnd: a("d_epsnuc_dlnd"),
        omega: a("omega"),
        photosphere_r: scalar("photosphere_r"),
        gyre_data_schema: i32::try_from(d.ints["gyre_data_schema"])
            .expect("i32"),
        rotation_flag: d.bools["rotation_flag"],
        interpolate_rho_for_pulse_data: d.bools
            ["interpolate_rho_for_pulse_data"],
        add_double_points_to_pulse_data: d.bools
            ["add_double_points_to_pulse_data"],
        threshold_grad_mu_for_double_point: scalar(
            "threshold_grad_mu_for_double_point",
        ),
        max_number_of_double_points: i32::try_from(
            d.ints["max_number_of_double_points"],
        )
        .expect("i32"),
        calculate_brunt_n2: d.bools["calculate_Brunt_N2"],
    };
    let (acp, ksp, aa) = (
        flag("add_center_point"),
        flag("keep_surface_point"),
        flag("add_atmosphere"),
    );
    let produce = |fmt: &str| -> String {
        match fmt {
            "GYRE" => {
                let (g, pd) = get_gyre_data(&base, acp, ksp, aa).expect("gyre");
                write_gyre_data(&g, &pd, base.gyre_data_schema).expect("write")
            }
            fmt @ ("FGONG" | "OSC") => {
                let mut xa = vec![0.0_f64; nz * species];
                for (j, name) in d.isos.iter().enumerate() {
                    for (k, v) in a(&format!("xa:{name}")).iter().enumerate() {
                        xa[k * species + j] = *v;
                    }
                }
                let idx = |nm: &str| d.isos.iter().position(|n| n == nm);
                let catalog = NuclideCatalog::builtin_ref();
                let chem_id: Vec<usize> = d
                    .isos
                    .iter()
                    .map(|n| catalog.get_nuclide_index(n).expect("iso"))
                    .collect();
                let iso = PulseIsoIndices {
                    h1: idx("h1"),
                    h2: idx("h2"),
                    he3: idx("he3"),
                    he4: idx("he4"),
                    li7: idx("li7"),
                    be7: idx("be7"),
                    be9: idx("be9"),
                    c12: idx("c12"),
                    c13: idx("c13"),
                    n14: idx("n14"),
                    n15: idx("n15"),
                    o16: idx("o16"),
                    o17: idx("o17"),
                    o18: idx("o18"),
                    ne20: idx("ne20"),
                    si28: idx("si28"),
                };
                let bottom = d.ints["largest_conv_mixing_region_bottom"];
                let input = PulseDataInput {
                    base,
                    species,
                    xa: &xa,
                    iso,
                    x: a("X").to_vec(),
                    y: a("Y").to_vec(),
                    cp: a("cp"),
                    lnfree_e: a("lnfree_e"),
                    eps_grav: a("eps_grav").to_vec(),
                    dxdt_nuc_h1: d.arrays.get("dxdt_nuc:h1").cloned(),
                    mixing_type: &d.mixing_type,
                    pgas: a("Pgas"),
                    gradr: a("gradr"),
                    chem_id: &chem_id,
                    eos_config: eos,
                    teff: scalar("Teff"),
                    star_age: scalar("star_age"),
                    initial_z: scalar("initial_z"),
                    initial_y: scalar("initial_y"),
                    mixing_length_alpha: scalar("mixing_length_alpha"),
                    fgong_zero_a_inside_r: scalar("fgong_zero_A_inside_r"),
                    largest_conv_mixing_region_bottom: usize::try_from(bottom)
                        .ok()
                        .filter(|&b| b > 0),
                };
                if fmt == "FGONG" {
                    let (g, pd) =
                        get_fgong_data(&input, acp, ksp, aa).expect("fgong");
                    write_fgong_data(
                        &g,
                        &pd,
                        &default_fgong_header(),
                        i32::try_from(d.ints["fgong_ivers"]).expect("i32"),
                    )
                    .expect("write")
                } else {
                    let (g, pd) =
                        get_osc_data(&input, acp, ksp, aa).expect("osc");
                    write_osc_data(
                        &g,
                        &pd,
                        MESA_VERSION_NUMBER,
                        DEFAULT_FORMAT_FOR_OSC_DATA,
                    )
                    .expect("write")
                }
            }
            other => panic!("unexpected format {other}"),
        }
    };
    let mut out = Vec::new();
    let own = d.format.trim().to_owned();
    let got = produce(&own);
    let same = got == expected;
    if !same {
        report_diff(&own, n, &got, &expected);
    }
    out.push((own, same));
    if let Some(exp) = expected_osc {
        let got = produce("OSC");
        let same = got == exp;
        if !same {
            report_diff("OSC", n, &got, &exp);
        }
        out.push((String::from("OSC"), same));
    }
    Some(out)
}

/// Prints the first differing lines and, per FGONG/GYRE column, how many
/// values differ.
fn report_diff(format: &str, n: usize, got: &str, expected: &str) {
    let (gl, el): (Vec<&str>, Vec<&str>) =
        (got.lines().collect(), expected.lines().collect());
    eprintln!("call {n} ({format}): {} vs {} lines", gl.len(), el.len());
    let mut shown = 0;
    for (i, (g, e)) in gl.iter().zip(el.iter()).enumerate() {
        if g != e && shown < 6 {
            eprintln!("  line {}:\n    got {g}\n    mesa {e}", i + 1);
            shown += 1;
        }
    }
    let mut diffs = 0usize;
    for (g, e) in gl.iter().zip(el.iter()) {
        let gt: Vec<&str> = g.split_whitespace().collect();
        let et: Vec<&str> = e.split_whitespace().collect();
        diffs += gt.iter().zip(et.iter()).filter(|(a, b)| a != b).count();
    }
    eprintln!("  differing tokens: {diffs}");
}

fn require_zero_skips_enabled() -> bool {
    !matches!(
        std::env::var("FLEXMESA_REQUIRE_ZERO_SKIPS")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "no" | "off"
    )
}

/// `FLEXMESA_PULSE_TAP_DIR`, else the `mesa-pulse-export-v1` bundle.
fn tap_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("FLEXMESA_PULSE_TAP_DIR")
        .filter(|v| !v.is_empty())
        .map_or_else(
            || {
                flexmesa_utils::resolve_flexmesa_data_dir()
                    .join("mesa_pulse_export")
                    .join("astero")
                    .join("fast_from_file")
            },
            PathBuf::from,
        );
    if dir.join("call_0001.dump").is_file() {
        return Some(dir);
    }
    assert!(
        !require_zero_skips_enabled(),
        "missing MESA pulse-export tap dumps in `{}` (run `flexmesa data \
         fetch --include-test-inputs` or set FLEXMESA_REQUIRE_ZERO_SKIPS=0)",
        dir.display()
    );
    None
}

#[test]
fn pulse_exports_match_mesa_on_tapped_state() {
    let Some(dir) = tap_dir() else {
        return;
    };
    let eos = EosConfig::default();
    let mut results = Vec::new();
    for n in 1..=12 {
        if let Some(r) = run_call(&dir, n, &eos) {
            for (fmt, ok) in r {
                results.push((n, fmt, ok));
            }
        }
    }
    eprintln!("{results:?}");
    assert!(!results.is_empty(), "no tapped calls in {}", dir.display());
    assert!(results.iter().all(|(_, _, ok)| *ok), "{results:?}");
}
