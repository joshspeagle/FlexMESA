// SPDX-License-Identifier: LGPL-3.0-or-later
//! MESA `26.4.1` IEEE-754 hex oracle test for `flexmesa_net::net_get` across
//! six upstream reaction networks (`#328`, `#161`).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use std::collections::HashMap;
use std::path::PathBuf;

use flexmesa_chem::NuclideCatalog;
use flexmesa_math::{NumericsMode, with_numerics_mode};
use flexmesa_net::{NetGeneralInfo, net_get};
use flexmesa_rates::ScreeningMode;

#[derive(Default)]
struct NetOracleCase {
    scalars_in: HashMap<String, f64>,
    x_in: Vec<(String, f64)>,
    scalars_out: Vec<(String, f64)>,
    iso_out: Vec<(String, String, f64)>,
    jac_out: Vec<(String, String, f64)>,
    cat_out: Vec<(usize, String, f64)>,
    rate_out: Vec<(String, String, usize, f64)>,
}

#[derive(Default)]
struct GroupStats {
    total: usize,
    zero_ulp: usize,
    max_ulp: u64,
    max_abs: f64,
    max_rel: f64,
    worst_label: String,
    worst_got: f64,
    worst_want: f64,
}

impl GroupStats {
    fn record(&mut self, label: impl Into<String>, got: f64, want: f64) {
        self.total += 1;
        let ulp = ulp_distance(got, want);
        if ulp == 0 {
            self.zero_ulp += 1;
        }
        if ulp > self.max_ulp {
            self.max_ulp = ulp;
        }
        let abs = (got - want).abs();
        if abs > self.max_abs {
            self.max_abs = abs;
        }
        let rel = rel_err(got, want);
        if rel > self.max_rel || (self.worst_label.is_empty() && ulp > 0) {
            self.max_rel = rel;
            self.worst_label = label.into();
            self.worst_got = got;
            self.worst_want = want;
        }
    }
}

fn signed_magnitude(bits: u64) -> i128 {
    if (bits >> 63) != 0 {
        -((bits ^ (1_u64 << 63)) as i128)
    } else {
        bits as i128
    }
}

fn ulp_distance(actual: f64, expected: f64) -> u64 {
    if actual.is_nan() || expected.is_nan() {
        return if actual.is_nan() && expected.is_nan() {
            0
        } else {
            u64::MAX
        };
    }
    let a = signed_magnitude(actual.to_bits());
    let b = signed_magnitude(expected.to_bits());
    u64::try_from((a - b).unsigned_abs()).unwrap_or(u64::MAX)
}

fn parse_hex_f64(raw: &str) -> f64 {
    let trimmed = raw
        .trim()
        .strip_prefix("0x")
        .or_else(|| raw.trim().strip_prefix("0X"))
        .unwrap_or_else(|| panic!("expected 0x-prefixed hex float, got {raw:?}"));
    let bits = u64::from_str_radix(trimmed, 16)
        .unwrap_or_else(|e| panic!("invalid hex bits {raw:?}: {e}"));
    f64::from_bits(bits)
}

fn rel_err(actual: f64, expected: f64) -> f64 {
    if actual == expected {
        return 0.0;
    }
    let denom = expected.abs();
    if denom == 0.0 {
        actual.abs()
    } else {
        (actual - expected).abs() / denom
    }
}

fn load_net_oracles() -> Vec<(String, NetOracleCase)> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/fortran_mesa/net_oracles/mesa_26_4_1_net_oracles.tsv");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("failed to read {}: {e}", path.display());
    });

    let mut has_mesa_ver = false;
    let mut has_mesa_commit = false;
    let mut has_sdk_ver = false;
    let mut has_math_backend = false;
    let mut has_compiler = false;
    let mut has_fflags = false;

    let mut order: Vec<String> = Vec::new();
    let mut map: HashMap<String, NetOracleCase> = HashMap::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("# mesa_version:") {
            has_mesa_ver = true;
            continue;
        }
        if trimmed.starts_with("# mesa_git_commit:") {
            has_mesa_commit = true;
            continue;
        }
        if trimmed.starts_with("# mesasdk_version:") {
            has_sdk_ver = true;
            continue;
        }
        if trimmed.starts_with("# math_backend:") {
            has_math_backend = true;
            continue;
        }
        if trimmed.starts_with("# compiler:") {
            assert!(
                trimmed.contains("GCC version"),
                "unexpected compiler header: {trimmed}"
            );
            has_compiler = true;
            continue;
        }
        if trimmed.starts_with("# fflags:") {
            has_fflags = true;
            continue;
        }
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("row_kind\t") {
            continue;
        }

        let cols: Vec<&str> = trimmed.split('\t').collect();
        assert_eq!(cols.len(), 7, "unexpected net oracle row: {trimmed}");
        let kind = cols[0];
        let net_name = cols[1].to_string();
        if !map.contains_key(&net_name) {
            order.push(net_name.clone());
            map.insert(net_name.clone(), NetOracleCase::default());
        }
        let case = map.get_mut(&net_name).expect("inserted network");
        let val = parse_hex_f64(cols[5]);

        match kind {
            "INPUT" => match cols[2] {
                "scalar" => {
                    case.scalars_in.insert(cols[3].to_string(), val);
                }
                "x" => case.x_in.push((cols[3].to_string(), val)),
                other => panic!("unexpected INPUT key1 {other:?}"),
            },
            "SCALAR" => case.scalars_out.push((cols[2].to_string(), val)),
            "ISO" => case
                .iso_out
                .push((cols[2].to_string(), cols[3].to_string(), val)),
            "JAC" => case
                .jac_out
                .push((cols[3].to_string(), cols[4].to_string(), val)),
            "CAT" => {
                let c_idx: usize = cols[3]
                    .parse()
                    .unwrap_or_else(|e| panic!("invalid category index {}: {e}", cols[3]));
                case.cat_out.push((c_idx, cols[4].to_string(), val));
            }
            "RATE" => {
                let r_idx: usize = cols[4]
                    .parse()
                    .unwrap_or_else(|e| panic!("invalid reaction index {}: {e}", cols[4]));
                case.rate_out
                    .push((cols[2].to_string(), cols[3].to_string(), r_idx, val));
            }
            other => panic!("unexpected row_kind {other:?}"),
        }
    }

    assert!(has_mesa_ver, "missing # mesa_version header");
    assert!(has_mesa_commit, "missing # mesa_git_commit header");
    assert!(has_sdk_ver, "missing # mesasdk_version header");
    assert!(has_math_backend, "missing # math_backend header");
    assert!(has_compiler, "missing # compiler header");
    assert!(has_fflags, "missing # fflags header");

    order
        .into_iter()
        .map(|name| {
            let case = map.remove(&name).expect("network present");
            (name, case)
        })
        .collect()
}

fn scalar_in(case: &NetOracleCase, key: &str) -> f64 {
    *case
        .scalars_in
        .get(key)
        .unwrap_or_else(|| panic!("missing scalar input {key}"))
}

fn iso_slot(g: &NetGeneralInfo, cat: &NuclideCatalog, name: &str) -> usize {
    let cid = cat
        .get_nuclide_index(name)
        .unwrap_or_else(|| panic!("unknown nuclide {name:?}"));
    let slot = g.net_iso[cid];
    assert!(slot > 0, "isotope {name:?} missing from {}", g.net_filename);
    slot - 1
}

#[test]
fn mesa_26_4_1_net_get_hex_oracles() {
    let cases = load_net_oracles();
    assert_eq!(cases.len(), 6, "expected six network oracle cases");
    let cat = NuclideCatalog::builtin_ref();

    with_numerics_mode(NumericsMode::Strict, || {
        for (net_name, case) in &cases {
            let g = NetGeneralInfo::from_name(net_name)
                .unwrap_or_else(|e| panic!("failed to load {net_name}: {e}"));
            assert_eq!(
                case.x_in.len(),
                g.num_isos,
                "{net_name}: isotope count mismatch"
            );

            let mut x = vec![0.0_f64; g.num_isos];
            for (iso_name, val) in &case.x_in {
                x[iso_slot(&g, cat, iso_name)] = *val;
            }
            let rate_factors = vec![1.0_f64; g.num_reactions];

            let res = net_get(
                &g,
                false,
                &x,
                scalar_in(case, "temp"),
                scalar_in(case, "logT"),
                scalar_in(case, "rho"),
                scalar_in(case, "logRho"),
                scalar_in(case, "abar"),
                scalar_in(case, "zbar"),
                scalar_in(case, "z2bar"),
                scalar_in(case, "ye"),
                scalar_in(case, "eta"),
                scalar_in(case, "d_eta_dlnT"),
                scalar_in(case, "d_eta_dlnRho"),
                &rate_factors,
                1.0,
                ScreeningMode::Extended,
            )
            .unwrap_or_else(|e| panic!("net_get failed for {net_name}: {e}"));

            let mut s_stats = GroupStats::default();
            for (field, want) in &case.scalars_out {
                let got = match field.as_str() {
                    "eps_nuc" => res.eps_nuc,
                    "d_eps_nuc_dRho" => res.d_eps_nuc_drho,
                    "d_eps_nuc_dT" => res.d_eps_nuc_dt,
                    "eps_neu_total" => res.eps_neu_total,
                    other => panic!("unknown scalar output {other:?}"),
                };
                s_stats.record(field.as_str(), got, *want);
            }

            let mut i_stats = GroupStats::default();
            for (field, iso_name, want) in &case.iso_out {
                let idx = iso_slot(&g, cat, iso_name);
                let got = match field.as_str() {
                    "dxdt" => res.dxdt[idx],
                    "d_dxdt_dRho" => res.d_dxdt_drho[idx],
                    "d_dxdt_dT" => res.d_dxdt_dt[idx],
                    "d_eps_nuc_dx" => res.d_eps_nuc_dx[idx],
                    other => panic!("unknown ISO output {other:?}"),
                };
                i_stats.record(format!("{field}[{iso_name}]"), got, *want);
            }

            let mut j_stats = GroupStats::default();
            for (iso_i, iso_j, want) in &case.jac_out {
                let ii = iso_slot(&g, cat, iso_i);
                let jj = iso_slot(&g, cat, iso_j);
                let got = res.d_dxdt_dx[ii * g.num_isos + jj];
                j_stats.record(format!("jac[{iso_i},{iso_j}]"), got, *want);
            }

            let mut c_stats = GroupStats::default();
            for (c_idx, cname, want) in &case.cat_out {
                let got = res.eps_nuc_categories[c_idx - 1];
                c_stats.record(format!("cat[{c_idx}:{cname}]"), got, *want);
            }

            let mut r_stats = GroupStats::default();
            for (field, rname, _r_idx, want) in &case.rate_out {
                let r_1based = g
                    .reaction_index_by_name(rname)
                    .unwrap_or_else(|| panic!("{net_name}: missing reaction {rname:?}"));
                let got = match field.as_str() {
                    "rate_raw" => res.rate_raw[r_1based - 1],
                    "rate_screened" => res.rate_screened[r_1based - 1],
                    other => panic!("unknown RATE field {other:?}"),
                };
                r_stats.record(format!("{field}[{rname}]"), got, *want);
            }

            for (gname, st) in [
                ("SCALAR", &s_stats),
                ("ISO", &i_stats),
                ("JAC", &j_stats),
                ("CAT", &c_stats),
                ("RATE", &r_stats),
            ] {
                eprintln!(
                    "net_oracle[{net_name}][{gname}]: zero_ulp={}/{} max_ulp={} max_rel={:.3e} worst={} (got={:.17e}, want={:.17e})",
                    st.zero_ulp,
                    st.total,
                    st.max_ulp,
                    st.max_rel,
                    st.worst_label,
                    st.worst_got,
                    st.worst_want
                );
            }

            assert_eq!(
                j_stats.max_ulp, 0,
                "{net_name} JAC must be 0 ULP (worst={}, got={:.17e}, want={:.17e})",
                j_stats.worst_label, j_stats.worst_got, j_stats.worst_want
            );
            assert_eq!(
                r_stats.max_ulp, 0,
                "{net_name} RATE must be 0 ULP (worst={}, got={:.17e}, want={:.17e})",
                r_stats.worst_label, r_stats.worst_got, r_stats.worst_want
            );
            assert_eq!(
                s_stats.max_ulp, 0,
                "{net_name} SCALAR must be 0 ULP (worst={}, got={:.17e}, want={:.17e})",
                s_stats.worst_label, s_stats.worst_got, s_stats.worst_want
            );
            assert_eq!(
                i_stats.max_ulp, 0,
                "{net_name} ISO must be 0 ULP (worst={}, got={:.17e}, want={:.17e})",
                i_stats.worst_label, i_stats.worst_got, i_stats.worst_want
            );
            assert_eq!(
                c_stats.max_ulp, 0,
                "{net_name} CAT must be 0 ULP (worst={}, got={:.17e}, want={:.17e})",
                c_stats.worst_label, c_stats.worst_got, c_stats.worst_want
            );
        }
    });
}
