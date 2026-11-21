// SPDX-License-Identifier: LGPL-3.0-or-later
//! MESA `26.4.1` IEEE-754 hex oracle test for `eps_mdot` (`eval_total_energy_profile`,
//! `calculate_eps_mdot`, `mdot_acoustic_surface`, and `mdot_adiabatic_surface`)
//! across accreting and mass-losing `1.0 Msun` ZAMS models (`#328`, `#318`).

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

use flexmesa_math::{NumericsMode, with_numerics_mode};
use flexmesa_star::{
    AdjustMassRecord, StarControls, StarJobControls, StarModel, calculate_eps_mdot,
    eval_total_energy_profile,
};

#[derive(Default)]
struct OracleCase {
    scalars: HashMap<String, f64>,
    zones: Vec<ZoneOracleRow>,
}

struct ZoneOracleRow {
    k_1based: usize,
    dm_before: f64,
    te_profile_before: f64,
    dm: f64,
    rho: f64,
    peos: f64,
    t: f64,
    cp: f64,
    lum: f64,
    gradt: f64,
    grada: f64,
    energy: f64,
    m: f64,
    m_grav: f64,
    r: f64,
    cgrav: f64,
    te_profile_after: f64,
    eps_mdot: f64,
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

fn scalar_f64(case: &OracleCase, key: &str) -> f64 {
    *case
        .scalars
        .get(key)
        .unwrap_or_else(|| panic!("missing scalar {key}"))
}

fn scalar_exact_usize(case: &OracleCase, key: &str) -> usize {
    let val = scalar_f64(case, key);
    assert!(
        val.is_finite() && val >= 0.0 && val.fract() == 0.0,
        "scalar {key} is not a non-negative integer: {val}"
    );
    val as usize
}

fn load_eps_mdot_oracles() -> Vec<(String, OracleCase)> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../benchmarks/fortran_mesa/eps_mdot_oracles/mesa_26_4_1_eps_mdot_oracles.tsv",
    );
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
    let mut map: HashMap<String, OracleCase> = HashMap::new();

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
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let cols: Vec<&str> = trimmed.split('\t').collect();
        match cols[0] {
            "SCALAR" => {
                assert_eq!(cols.len(), 6, "unexpected SCALAR row: {trimmed}");
                let case_id = cols[1].to_string();
                if !map.contains_key(&case_id) {
                    order.push(case_id.clone());
                    map.insert(case_id.clone(), OracleCase::default());
                }
                let entry = map.get_mut(&case_id).expect("inserted case");
                let prev = entry
                    .scalars
                    .insert(cols[3].to_string(), parse_hex_f64(cols[4]));
                assert!(prev.is_none(), "duplicate scalar {} in {case_id}", cols[3]);
            }
            "ZONE" => {
                assert_eq!(cols.len(), 21, "unexpected ZONE row: {trimmed}");
                let case_id = cols[1].to_string();
                let entry = map
                    .get_mut(&case_id)
                    .unwrap_or_else(|| panic!("ZONE row before SCALAR for {case_id}"));
                let k_1based: usize = cols[2]
                    .parse()
                    .unwrap_or_else(|e| panic!("invalid zone index {}: {e}", cols[2]));
                entry.zones.push(ZoneOracleRow {
                    k_1based,
                    dm_before: parse_hex_f64(cols[3]),
                    te_profile_before: parse_hex_f64(cols[4]),
                    dm: parse_hex_f64(cols[5]),
                    rho: parse_hex_f64(cols[6]),
                    peos: parse_hex_f64(cols[7]),
                    t: parse_hex_f64(cols[8]),
                    cp: parse_hex_f64(cols[9]),
                    lum: parse_hex_f64(cols[10]),
                    gradt: parse_hex_f64(cols[11]),
                    grada: parse_hex_f64(cols[12]),
                    energy: parse_hex_f64(cols[13]),
                    m: parse_hex_f64(cols[14]),
                    m_grav: parse_hex_f64(cols[15]),
                    r: parse_hex_f64(cols[16]),
                    cgrav: parse_hex_f64(cols[17]),
                    te_profile_after: parse_hex_f64(cols[18]),
                    eps_mdot: parse_hex_f64(cols[19]),
                });
            }
            other => panic!("unexpected row tag {other:?} in {trimmed}"),
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
        .map(|id| {
            let case = map.remove(&id).expect("case present");
            (id, case)
        })
        .collect()
}

#[derive(Default)]
struct SeriesStats {
    total: usize,
    zero_ulp: usize,
    max_ulp: u64,
    max_abs: f64,
    max_rel: f64,
    max_rel_nonzero: f64,
    worst_k: usize,
    worst_got: f64,
    worst_want: f64,
}

impl SeriesStats {
    fn record(&mut self, k_1based: usize, got: f64, want: f64) {
        self.total += 1;
        let ulp = ulp_distance(got, want);
        if ulp == 0 {
            self.zero_ulp += 1;
        }
        let abs = (got - want).abs();
        let rel = rel_err(got, want);
        if ulp > self.max_ulp {
            self.max_ulp = ulp;
        }
        if abs > self.max_abs {
            self.max_abs = abs;
        }
        if rel > self.max_rel {
            self.max_rel = rel;
            self.worst_k = k_1based;
            self.worst_got = got;
            self.worst_want = want;
        }
        if want != 0.0 && rel > self.max_rel_nonzero {
            self.max_rel_nonzero = rel;
        }
    }
}

#[test]
fn mesa_26_4_1_eps_mdot_hex_oracles() {
    let cases = load_eps_mdot_oracles();
    assert_eq!(cases.len(), 2, "expected accreting + mass-losing oracle cases");

    with_numerics_mode(NumericsMode::Strict, || {
        for (case_id, case) in &cases {
            let nz = scalar_exact_usize(case, "nz");
            assert_eq!(case.zones.len(), nz, "{case_id}: zone row count mismatch");

            let dt = scalar_f64(case, "dt");
            let mut ctrl = StarControls::default();
            ctrl.mix_wind.eps_mdot_leak_frac_factor = scalar_f64(case, "eps_mdot_leak_frac_factor");
            ctrl.mix_wind.eps_mdot_factor = scalar_f64(case, "eps_mdot_factor");

            let mut model = StarModel::new(nz, vec![1], StarJobControls::default(), ctrl);
            model.mstar_dot = scalar_f64(case, "mstar_dot");
            model.m_center = scalar_f64(case, "m_center");
            model.r_center = scalar_f64(case, "r_center");

            let mut dm_before = Vec::with_capacity(nz);
            let mut total_energy_profile_before = Vec::with_capacity(nz);

            for (idx, row) in case.zones.iter().enumerate() {
                assert_eq!(row.k_1based, idx + 1, "{case_id}: non-contiguous zone index");
                dm_before.push(row.dm_before);
                total_energy_profile_before.push(row.te_profile_before);
                model.dm[idx] = row.dm;
                model.rho[idx] = row.rho;
                model.peos[idx] = row.peos;
                model.t[idx] = row.t;
                model.cp[idx] = row.cp;
                model.lum[idx] = row.lum;
                model.gradt[idx] = row.gradt;
                model.grada[idx] = row.grada;
                model.energy[idx] = row.energy;
                model.m[idx] = row.m;
                model.m_grav[idx] = row.m_grav;
                model.r[idx] = row.r;
                model.cgrav[idx] = row.cgrav;
            }

            let record = AdjustMassRecord {
                dm_before,
                total_energy_profile_before,
                outer_frac_sub1: scalar_f64(case, "outer_frac_sub1"),
                mid_frac_sub1: scalar_f64(case, "mid_frac_sub1"),
                inner_frac_sub1: scalar_f64(case, "inner_frac_sub1"),
                k_below_const_q: scalar_exact_usize(case, "k_below_const_q"),
                k_const_mass: scalar_exact_usize(case, "k_const_mass"),
            };

            let te_after = eval_total_energy_profile(&model)
                .unwrap_or_else(|e| panic!("{case_id}: eval_total_energy_profile failed: {e}"));
            let surface = calculate_eps_mdot(&mut model, &record, dt)
                .unwrap_or_else(|e| panic!("{case_id}: calculate_eps_mdot failed: {e}"));

            let want_acoustic = scalar_f64(case, "mdot_acoustic_surface");
            let want_adiabatic = scalar_f64(case, "mdot_adiabatic_surface");

            let mut te_stats = SeriesStats::default();
            let mut eps_stats = SeriesStats::default();
            for (idx, row) in case.zones.iter().enumerate() {
                te_stats.record(row.k_1based, te_after[idx], row.te_profile_after);
                eps_stats.record(row.k_1based, model.eps_mdot[idx], row.eps_mdot);
            }

            let ac_ulp = ulp_distance(surface.mdot_acoustic_surface, want_acoustic);
            let ac_rel = rel_err(surface.mdot_acoustic_surface, want_acoustic);
            let ad_ulp = ulp_distance(surface.mdot_adiabatic_surface, want_adiabatic);
            let ad_rel = rel_err(surface.mdot_adiabatic_surface, want_adiabatic);

            eprintln!(
                "eps_mdot_oracle[{case_id}]: te_after zero_ulp={}/{} max_ulp={} max_rel={:.3e} (k={})",
                te_stats.zero_ulp, te_stats.total, te_stats.max_ulp, te_stats.max_rel, te_stats.worst_k
            );
            eprintln!(
                "eps_mdot_oracle[{case_id}]: eps_mdot zero_ulp={}/{} max_ulp={} max_rel={:.3e} max_rel_nz={:.3e} (k={}, got={:.17e}, want={:.17e})",
                eps_stats.zero_ulp,
                eps_stats.total,
                eps_stats.max_ulp,
                eps_stats.max_rel,
                eps_stats.max_rel_nonzero,
                eps_stats.worst_k,
                eps_stats.worst_got,
                eps_stats.worst_want
            );
            eprintln!(
                "eps_mdot_oracle[{case_id}]: mdot_acoustic got={:.17e} want={:.17e} ulp={} rel={:.3e}",
                surface.mdot_acoustic_surface, want_acoustic, ac_ulp, ac_rel
            );
            eprintln!(
                "eps_mdot_oracle[{case_id}]: mdot_adiabatic got={:.17e} want={:.17e} ulp={} rel={:.3e}",
                surface.mdot_adiabatic_surface, want_adiabatic, ad_ulp, ad_rel
            );

            assert_eq!(
                te_stats.max_ulp, 0,
                "{case_id} te_after max_ulp = {} > 0 (k={})",
                te_stats.max_ulp, te_stats.worst_k
            );
            assert_eq!(
                eps_stats.max_ulp, 0,
                "{case_id} eps_mdot max_ulp = {} > 0 (k={})",
                eps_stats.max_ulp, eps_stats.worst_k
            );
            assert_eq!(
                ac_ulp, 0,
                "{case_id} mdot_acoustic_surface ulp = {ac_ulp} > 0"
            );
            assert_eq!(
                ad_ulp, 0,
                "{case_id} mdot_adiabatic_surface ulp = {ad_ulp} > 0"
            );
        }
    });
}
