// SPDX-License-Identifier: GPL-3.0-or-later
//! Env-gated parity suite for ADIPLS Package A2b (`#171`, `DECISIONS #60`):
//!
//! 1. Verifies `ENTER`, `DET`, `MATCH`, `ITC`, `KINER`, and `VARFRQ`
//!    intermediate hex tap records (and confirms zero `GRAVPO` taps) across all
//!    5 committed `benchmarks/fortran_mesa/adipls_oracles/data/*.taps.tsv`
//!    files at `0 ULP`.
//! 2. Replays all `2,165` per-model ADIPLS calls (`27,361` modes) across all 5
//!    MESA `26.4.1` cases in `FLEXMESA_ADIPLS_CAPTURES` through
//!    [`AdiplsSolver::run_call`], checking `degree`, `order`, `em`,
//!    `cyclic_freq_uHz`, `inertia`, `obs_st(1:10)`, and `csummm(1:50)`.
//!
//! When `FLEXMESA_ADIPLS_CAPTURES` is unset or empty, this test logs an
//! `INFO:` note and returns early even under `FLEXMESA_REQUIRE_ZERO_SKIPS=1`
//! so CI stays self-contained without the external A0 captures archive. When
//! `FLEXMESA_ADIPLS_CAPTURES` is set, missing files or mismatches fail loudly.

use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use flexmesa_adipls::{
    parse_adipls_pruned_in, set_adipls_controls, AdiplsControlState,
    AdiplsSolver, AmdlAa, TapRecorder,
};

const ADIPLS_PRUNED_IN: &str = "\
2  'amdl.mesa'
9  'ttt.adipls.prt'
11 'agsm.mesa'
15 'ttt.adipls.ssm'
16 'ttt.adipls.fsm'
-1 ''
mod.osc.cst.int.out
,,,,,,,
,,1,,,,,,,,,,,,,,,,,,,,
   0,3,0,1,,,,,,,
    1,  220,   ,    1,10,,,,,,,,
,2,100,1000,,,,,,,,,,,,
,,,,,,,,,,,,,,,,
6.67430e-8
,,0,,
1,,,,,,,,,,,,,,,
5,0,0.99,,,,,,,,,,,,,,
,,,,,-1,,,,,,,,,,,,,,
9,10,,20,1,,,,,,,,
1,,,,,,,,,,,,,,,
0,,,,6,,,,,,
,,,,,-5000,100,,,,,,,,,,,,,,,,,,,,,,,,
";

struct OracleCase {
    case_name: &'static str,
    oracle_file: &'static str,
    call_id: usize,
    expected_amdl: &'static str,
}

const ORACLE_CASES: [OracleCase; 5] = [
    OracleCase {
        case_name: "astero_adipls",
        oracle_file: "astero__astero_adipls__c000001.taps.tsv",
        call_id: 1,
        expected_amdl: "call_000001.amdl.bin",
    },
    OracleCase {
        case_name: "example_astero",
        oracle_file: "astero__example_astero__c000029.taps.tsv",
        call_id: 29,
        expected_amdl: "call_000027.amdl.bin",
    },
    OracleCase {
        case_name: "fast_scan_grid",
        oracle_file: "astero__fast_scan_grid__c000211.taps.tsv",
        call_id: 211,
        expected_amdl: "call_000210.amdl.bin",
    },
    OracleCase {
        case_name: "surface_effects",
        oracle_file: "astero__surface_effects__c000002.taps.tsv",
        call_id: 2,
        expected_amdl: "call_000002.amdl.bin",
    },
    OracleCase {
        case_name: "surface_effects",
        oracle_file: "astero__surface_effects__c000004.taps.tsv",
        call_id: 4,
        expected_amdl: "call_000004.amdl.bin",
    },
];

struct CaseParitySpec {
    name: &'static str,
    expected_calls: usize,
    expected_modes: usize,
}

const CAPTURE_CASES: [CaseParitySpec; 5] = [
    CaseParitySpec {
        name: "astero_adipls",
        expected_calls: 10,
        expected_modes: 92,
    },
    CaseParitySpec {
        name: "example_astero",
        expected_calls: 65,
        expected_modes: 1079,
    },
    CaseParitySpec {
        name: "fast_scan_grid",
        expected_calls: 1107,
        expected_modes: 16301,
    },
    CaseParitySpec {
        name: "fast_simplex",
        expected_calls: 979,
        expected_modes: 9779,
    },
    CaseParitySpec {
        name: "surface_effects",
        expected_calls: 4,
        expected_modes: 110,
    },
];

#[allow(dead_code)]
struct CallRow {
    call_id: usize,
    nn: usize,
    amdl_file: String,
    el: f64,
    els1: f64,
    dels: f64,
    dfsig1: f64,
    dfsig2: f64,
    sig1: f64,
    sig2: f64,
    dfsig: f64,
    eltrw1: f64,
    eltrw2: f64,
    sgtrw1: f64,
    sgtrw2: f64,
    nsel: i32,
    nsig1: i32,
    nsig2: i32,
    itrsig: i32,
    nsig: i32,
    istsig: i32,
    inomd1: i32,
    iscan: i32,
    irotkr: i32,
    nprtkr: i32,
    igm1kr: i32,
    npgmkr: i32,
}

struct ExpectedTap {
    tag: String,
    values: Vec<u64>,
}

struct ExpectedMode {
    call_id: usize,
    k: usize,
    el: i32,
    order: i32,
    em: i32,
    freq_bits: u64,
    inertia_bits: u64,
    obs_bits: [u64; 10],
    csum_bits: [u64; 50],
}

#[derive(Default)]
#[allow(clippy::struct_field_names)]
struct UlpHistogram {
    zero_ulp: usize,
    one_ulp: usize,
    gt_one_ulp: usize,
    max_ulp: u64,
}

impl UlpHistogram {
    fn record_f64(&mut self, got: u64, want: u64) {
        if got == want {
            self.zero_ulp += 1;
            return;
        }
        let d = ulp_distance(got, want);
        self.max_ulp = self.max_ulp.max(d);
        if d <= 1 {
            self.one_ulp += 1;
        } else {
            self.gt_one_ulp += 1;
        }
    }

    fn record_exact_bits(&mut self, got: u64, want: u64) {
        if got == want {
            self.zero_ulp += 1;
        } else {
            self.gt_one_ulp += 1;
        }
    }
}

fn captures_root_or_skip() -> Option<PathBuf> {
    let raw = std::env::var("FLEXMESA_ADIPLS_CAPTURES").ok();
    let path_str = raw.as_deref().map_or("", str::trim);
    if path_str.is_empty() {
        eprintln!(
            "INFO: FLEXMESA_ADIPLS_CAPTURES unset; skipping external A2b captures parity test"
        );
        return None;
    }
    let root = PathBuf::from(path_str);
    assert!(
        root.is_dir(),
        "FLEXMESA_ADIPLS_CAPTURES path `{}` is not a directory",
        root.display()
    );
    Some(root)
}

fn parse_hex_u64(hex: &str) -> u64 {
    u64::from_str_radix(hex.trim(), 16)
        .unwrap_or_else(|err| panic!("bad hex binary64 `{hex}`: {err}"))
}

fn parse_hex_f64(hex: &str) -> f64 {
    f64::from_bits(parse_hex_u64(hex))
}

fn parse_call_row(line: &str, calls_path: &Path, line_idx: usize) -> CallRow {
    let cols: Vec<&str> = line.split('\t').collect();
    assert!(
        cols.len() >= 34,
        "calls.tsv row {} in `{}` expected >= 34 columns, got {}",
        line_idx + 1,
        calls_path.display(),
        cols.len()
    );
    let parse_i32 = |col: usize| -> i32 {
        cols[col].trim().parse().unwrap_or_else(|err| {
            panic!("bad i32 col[{col}] `{}`: {err}", cols[col]);
        })
    };
    let call_id: usize = cols[0].trim().parse().unwrap_or_else(|err| {
        panic!("bad call_id `{}`: {err}", cols[0]);
    });
    let nn: usize = cols[4].trim().parse().unwrap_or_else(|err| {
        panic!("bad nn `{}`: {err}", cols[4]);
    });
    CallRow {
        call_id,
        nn,
        amdl_file: cols[5].trim().to_owned(),
        el: parse_hex_f64(cols[10]),
        els1: parse_hex_f64(cols[11]),
        dels: parse_hex_f64(cols[12]),
        dfsig1: parse_hex_f64(cols[13]),
        dfsig2: parse_hex_f64(cols[14]),
        sig1: parse_hex_f64(cols[15]),
        sig2: parse_hex_f64(cols[16]),
        dfsig: parse_hex_f64(cols[17]),
        eltrw1: parse_hex_f64(cols[18]),
        eltrw2: parse_hex_f64(cols[19]),
        sgtrw1: parse_hex_f64(cols[20]),
        sgtrw2: parse_hex_f64(cols[21]),
        nsel: parse_i32(22),
        nsig1: parse_i32(23),
        nsig2: parse_i32(24),
        itrsig: parse_i32(25),
        nsig: parse_i32(26),
        istsig: parse_i32(27),
        inomd1: parse_i32(28),
        iscan: parse_i32(29),
        irotkr: parse_i32(30),
        nprtkr: parse_i32(31),
        igm1kr: parse_i32(32),
        npgmkr: parse_i32(33),
    }
}

fn read_all_call_rows(calls_path: &Path) -> Vec<CallRow> {
    let text = fs::read_to_string(calls_path).unwrap_or_else(|err| {
        panic!("failed to read `{}`: {err}", calls_path.display())
    });
    let mut out = Vec::new();
    for (line_idx, line) in text.lines().enumerate() {
        if line_idx == 0 || line.trim().is_empty() {
            continue;
        }
        out.push(parse_call_row(line, calls_path, line_idx));
    }
    out
}

fn read_amdl_bin_strict(
    amdl_path: &Path,
    expected_nn: usize,
) -> ([f64; 8], Vec<f64>, AmdlAa) {
    let file = File::open(amdl_path).unwrap_or_else(|err| {
        panic!("failed to open `{}`: {err}", amdl_path.display())
    });
    let mut reader = BufReader::new(file);
    let mut header = [0_u8; 8];
    reader
        .read_exact(&mut header)
        .expect("read amdl header (nn, iaa)");
    let nn_i32 = i32::from_le_bytes(header[0..4].try_into().unwrap());
    let iaa_i32 = i32::from_le_bytes(header[4..8].try_into().unwrap());
    let nn = usize::try_from(nn_i32).expect("positive nn");
    let iaa = usize::try_from(iaa_i32).expect("positive iaa");
    assert_eq!(nn, expected_nn, "amdl nn mismatch");
    assert_eq!(iaa, 10, "expected iaa=10 in a0tap amdl binary");

    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).expect("read amdl payload");
    let total_f64 = 8 + nn + iaa * nn;
    assert_eq!(
        buf.len(),
        total_f64 * 8,
        "unexpected byte size in `{}`",
        amdl_path.display()
    );
    let read_f64 = |idx: usize| -> f64 {
        let start = idx * 8;
        let bytes: [u8; 8] = buf[start..start + 8].try_into().unwrap();
        f64::from_le_bytes(bytes)
    };

    let mut data = [0.0_f64; 8];
    for (idx, slot) in data.iter_mut().enumerate() {
        *slot = read_f64(idx);
    }
    let mut x = Vec::with_capacity(nn);
    for idx in 0..nn {
        x.push(read_f64(8 + idx));
    }

    // Load ONLY 1-based rows 1..=5 and 10 so rows 6..=9 remain unset (#479).
    let mut rows: Vec<Vec<f64>> =
        (0..5).map(|_| Vec::with_capacity(nn)).collect();
    let aa_offset = 8 + nn;
    for zone in 0..nn {
        let col_base = aa_offset + zone * iaa;
        for (row_idx, row_vec) in rows.iter_mut().enumerate() {
            row_vec.push(read_f64(col_base + row_idx));
        }
    }
    let mut aa = AmdlAa::new_unset(nn);
    for (row_idx, values) in rows.into_iter().enumerate() {
        aa.set_row(row_idx + 1, values).expect("set row 1..=5");
    }
    let row10: Vec<f64> = (0..nn)
        .map(|zone| read_f64(aa_offset + zone * iaa + 9))
        .collect();
    aa.set_row(10, row10).expect("set row 10");

    (data, x, aa)
}

fn read_expected_all_taps(oracle_path: &Path) -> Vec<ExpectedTap> {
    let text = fs::read_to_string(oracle_path).unwrap_or_else(|err| {
        panic!("failed to read `{}`: {err}", oracle_path.display())
    });
    let mut out = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        assert!(
            parts.len() >= 2,
            "malformed tap line {} in `{}`: `{line}`",
            line_no + 1,
            oracle_path.display()
        );
        let tag = parts[0].trim();
        assert_ne!(
            tag,
            "GRAVPO",
            "unexpected GRAVPO tap in {}",
            oracle_path.display()
        );
        let count: usize = parts[1].trim().parse().unwrap_or_else(|err| {
            panic!("bad field count on line {}: {err}", line_no + 1)
        });
        let values: Vec<u64> = if count == 0 {
            Vec::new()
        } else {
            let hex_col = parts.get(2).copied().unwrap_or("");
            hex_col.split_whitespace().map(parse_hex_u64).collect()
        };
        assert_eq!(
            values.len(),
            count,
            "field count mismatch on line {} in `{}`",
            line_no + 1,
            oracle_path.display()
        );
        out.push(ExpectedTap {
            tag: tag.to_owned(),
            values,
        });
    }
    out
}

fn read_expected_modes(modes_path: &Path) -> Vec<ExpectedMode> {
    let text = fs::read_to_string(modes_path).unwrap_or_else(|err| {
        panic!("failed to read `{}`: {err}", modes_path.display())
    });
    let mut out = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        if idx == 0 || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        assert_eq!(
            parts.len(),
            10,
            "modes.tsv row {} in `{}` expected 10 cols",
            idx + 1,
            modes_path.display()
        );
        let call_id: usize = parts[0].trim().parse().unwrap();
        let k: usize = parts[1].trim().parse().unwrap();
        let el: i32 = parts[3].trim().parse().unwrap();
        let order: i32 = parts[4].trim().parse().unwrap();
        let em: i32 = parts[5].trim().parse().unwrap();
        let freq_bits = parse_hex_u64(parts[6]);
        let inertia_bits = parse_hex_u64(parts[7]);
        let obs_bits: [u64; 10] = parts[8]
            .split_whitespace()
            .map(parse_hex_u64)
            .collect::<Vec<_>>()
            .try_into()
            .expect("10 obs_st hex words");
        let csum_bits: [u64; 50] = parts[9]
            .split_whitespace()
            .map(parse_hex_u64)
            .collect::<Vec<_>>()
            .try_into()
            .expect("50 csummm hex words");
        out.push(ExpectedMode {
            call_id,
            k,
            el,
            order,
            em,
            freq_bits,
            inertia_bits,
            obs_bits,
            csum_bits,
        });
    }
    out
}

fn ulp_distance(a: u64, b: u64) -> u64 {
    fn to_ordered(bits: u64) -> i64 {
        let signed = bits.cast_signed();
        if signed < 0 {
            i64::MIN - signed
        } else {
            signed
        }
    }
    to_ordered(a).abs_diff(to_ordered(b))
}

fn build_controls(
    base: &AdiplsControlState,
    row: &CallRow,
) -> AdiplsControlState {
    #[allow(clippy::cast_possible_truncation)]
    let el_i32 = row.el as i32;
    assert_eq!(f64::from(el_i32).to_bits(), row.el.to_bits());
    let mut controls = *base;
    set_adipls_controls(
        &mut controls,
        el_i32,
        row.nsel,
        row.els1,
        row.dels,
        row.itrsig,
        row.iscan,
        row.sig1,
        row.sig2,
        row.dfsig,
        row.nsig,
        row.irotkr,
        row.nprtkr,
        row.igm1kr,
        row.npgmkr,
    );
    controls
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_adipls_a2b_full_taps_and_2165_calls_27361_modes_parity() {
    let Some(captures_root) = captures_root_or_skip() else {
        return;
    };

    let oracles_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/fortran_mesa/adipls_oracles/data");
    let input = parse_adipls_pruned_in(ADIPLS_PRUNED_IN)
        .expect("parse ADIPLS_PRUNED_IN");
    let base_controls = AdiplsControlState::from_input(&input);
    let solver = AdiplsSolver::new(input);

    // 1. Check all 5 representative `*.taps.tsv` files including `KINER` & `VARFRQ`.
    let mut tap_enter = 0_usize;
    let mut tap_det = 0_usize;
    let mut tap_match = 0_usize;
    let mut tap_itc = 0_usize;
    let mut tap_kiner = 0_usize;
    let mut tap_varfrq = 0_usize;

    for spec in &ORACLE_CASES {
        let calls_path =
            captures_root.join(spec.case_name).join("a0tap/calls.tsv");
        let rows = read_all_call_rows(&calls_path);
        let row = rows
            .iter()
            .find(|r| r.call_id == spec.call_id)
            .unwrap_or_else(|| {
                panic!("call_id {} missing in {}", spec.call_id, spec.case_name)
            });
        assert_eq!(row.amdl_file, spec.expected_amdl);

        let amdl_path = captures_root
            .join(spec.case_name)
            .join("a0tap")
            .join(&row.amdl_file);
        let (data, x, aa) = read_amdl_bin_strict(&amdl_path, row.nn);
        let controls = build_controls(&base_controls, row);

        let mut recorder = TapRecorder::new();
        let res = solver
            .run_call_with_observer(&controls, &x, &aa, data, &mut recorder)
            .unwrap_or_else(|err| {
                panic!(
                    "A2b solver failed on {} call {}: {err}",
                    spec.case_name, spec.call_id
                )
            });
        assert_eq!(res.ierr_param, 0);

        let oracle_path = oracles_dir.join(spec.oracle_file);
        let expected = read_expected_all_taps(&oracle_path);
        assert_eq!(
            recorder.events.len(),
            expected.len(),
            "tap count mismatch for {}",
            spec.oracle_file
        );

        for (idx, (got_ev, exp_ev)) in
            recorder.events.iter().zip(expected.iter()).enumerate()
        {
            assert_eq!(
                got_ev.tag(),
                exp_ev.tag,
                "tag mismatch at event {idx} in {}",
                spec.oracle_file
            );
            match got_ev.tag() {
                "ENTER" => tap_enter += 1,
                "DET" => tap_det += 1,
                "MATCH" => tap_match += 1,
                "ITC" => tap_itc += 1,
                "KINER" => tap_kiner += 1,
                "VARFRQ" => tap_varfrq += 1,
                other => panic!("unexpected tap tag {other}"),
            }
            let got_vals = got_ev.to_values();
            assert_eq!(got_vals.len(), exp_ev.values.len());
            for (field_idx, (&got_f64, &exp_bits)) in
                got_vals.iter().zip(exp_ev.values.iter()).enumerate()
            {
                let got_bits = got_f64.to_bits();
                assert_eq!(
                    got_bits,
                    exp_bits,
                    "0-ULP tap mismatch in {} event #{idx} {} field[{field_idx}]: \
                     got {got_bits:016X} vs expected {exp_bits:016X} (ulp={})",
                    spec.oracle_file,
                    got_ev.tag(),
                    ulp_distance(got_bits, exp_bits)
                );
            }
        }
    }
    assert_eq!(tap_enter, 5);
    assert_eq!(tap_det, 1602);
    assert_eq!(tap_match, 97);
    assert_eq!(tap_itc, 97);
    assert_eq!(tap_kiner, 93);
    assert_eq!(tap_varfrq, 93);

    // 2. Full end-to-end replay across all 2,165 calls / 27,361 modes.
    let mut total_calls = 0_usize;
    let mut total_modes = 0_usize;
    let mut freq_hist = UlpHistogram::default();
    let mut inertia_hist = UlpHistogram::default();
    let mut obs_hist = UlpHistogram::default();
    let mut csum_f64_hist = UlpHistogram::default();
    let mut csum_bits_hist = UlpHistogram::default();

    for spec in &CAPTURE_CASES {
        let tap_dir = captures_root.join(spec.name).join("a0tap");
        let calls = read_all_call_rows(&tap_dir.join("calls.tsv"));
        assert_eq!(calls.len(), spec.expected_calls);
        let expected_modes = read_expected_modes(&tap_dir.join("modes.tsv"));
        assert_eq!(expected_modes.len(), spec.expected_modes);

        let mut cached_amdl_name = String::new();
        let mut cached_model: Option<([f64; 8], Vec<f64>, AmdlAa)> = None;
        let mut mode_cursor = 0_usize;

        for row in &calls {
            total_calls += 1;
            if row.amdl_file != cached_amdl_name {
                let model =
                    read_amdl_bin_strict(&tap_dir.join(&row.amdl_file), row.nn);
                cached_amdl_name = row.amdl_file.clone();
                cached_model = Some(model);
            }
            let (data, x, aa) = cached_model.as_ref().unwrap();
            let controls = build_controls(&base_controls, row);
            let res = solver.run_call(&controls, x, aa, *data).unwrap_or_else(
                |err| {
                    panic!(
                        "run_call failed on {} call {}: {err}",
                        spec.name, row.call_id
                    )
                },
            );
            assert_eq!(res.ierr_param, 0);
            assert_eq!(res.icobs_st, 6);
            assert_eq!(res.obs_st.len(), res.modes.len());

            let n_call_modes = res.modes.len();
            let call_expected =
                &expected_modes[mode_cursor..mode_cursor + n_call_modes];
            mode_cursor += n_call_modes;

            for (idx, (got_rec, exp_rec)) in res
                .modes
                .records
                .iter()
                .zip(call_expected.iter())
                .enumerate()
            {
                total_modes += 1;
                assert_eq!(exp_rec.call_id, row.call_id);
                assert_eq!(exp_rec.k, idx + 1);
                assert_eq!(
                    got_rec.el, exp_rec.el,
                    "{} call {} mode {}: el mismatch",
                    spec.name, row.call_id, exp_rec.k
                );
                assert_eq!(
                    got_rec.order, exp_rec.order,
                    "{} call {} mode {}: order mismatch",
                    spec.name, row.call_id, exp_rec.k
                );
                assert_eq!(
                    got_rec.em, exp_rec.em,
                    "{} call {} mode {}: em mismatch",
                    spec.name, row.call_id, exp_rec.k
                );

                freq_hist.record_f64(
                    got_rec.cyclic_freq_uhz.to_bits(),
                    exp_rec.freq_bits,
                );
                inertia_hist.record_f64(
                    got_rec.inertia.to_bits(),
                    exp_rec.inertia_bits,
                );
                for (oi, (&got_obs, &want_obs)) in got_rec
                    .obs_st
                    .as_array()
                    .iter()
                    .zip(exp_rec.obs_bits.iter())
                    .enumerate()
                {
                    let got_b = got_obs.to_bits();
                    assert_eq!(
                        got_b,
                        want_obs,
                        "{} call {} mode {} obs_st[{}]: got {got_b:016X} vs {want_obs:016X} (ulp={})",
                        spec.name,
                        row.call_id,
                        exp_rec.k,
                        oi + 1,
                        ulp_distance(got_b, want_obs)
                    );
                    obs_hist.record_f64(got_b, want_obs);
                }

                let got_csum = got_rec.csummm.as_bits();
                for (ci, (&got_w, &want_w)) in
                    got_csum.iter().zip(exp_rec.csum_bits.iter()).enumerate()
                {
                    if ci < 38 {
                        assert_eq!(
                            got_w,
                            want_w,
                            "{} call {} mode {} csummm[{}]: got {got_w:016X} vs {want_w:016X} (ulp={})",
                            spec.name,
                            row.call_id,
                            exp_rec.k,
                            ci + 1,
                            ulp_distance(got_w, want_w)
                        );
                        csum_f64_hist.record_f64(got_w, want_w);
                    } else {
                        assert_eq!(
                            got_w,
                            want_w,
                            "{} call {} mode {} csummm[{}] (int/name): got {got_w:016X} vs {want_w:016X}",
                            spec.name,
                            row.call_id,
                            exp_rec.k,
                            ci + 1
                        );
                        csum_bits_hist.record_exact_bits(got_w, want_w);
                    }
                }
            }
        }
        assert_eq!(mode_cursor, expected_modes.len());
    }

    eprintln!(
        "[adipls_a2b_modes_parity_test] calls={total_calls}, modes={total_modes}; \
         taps(ENTER={tap_enter}, DET={tap_det}, MATCH={tap_match}, ITC={tap_itc}, \
         KINER={tap_kiner}, VARFRQ={tap_varfrq}); \
         freq=(0ulp:{},1ulp:{},max:{}); inertia=(0ulp:{},1ulp:{},max:{}); \
         obs_st=(0ulp:{},1ulp:{},max:{}); csum_f64=(0ulp:{},1ulp:{},max:{}); \
         csum_bits=(0ulp:{},mismatch:{})",
        freq_hist.zero_ulp,
        freq_hist.one_ulp,
        freq_hist.max_ulp,
        inertia_hist.zero_ulp,
        inertia_hist.one_ulp,
        inertia_hist.max_ulp,
        obs_hist.zero_ulp,
        obs_hist.one_ulp,
        obs_hist.max_ulp,
        csum_f64_hist.zero_ulp,
        csum_f64_hist.one_ulp,
        csum_f64_hist.max_ulp,
        csum_bits_hist.zero_ulp,
        csum_bits_hist.gt_one_ulp,
    );

    assert_eq!(total_calls, 2165);
    assert_eq!(total_modes, 27361);
    assert_eq!(freq_hist.max_ulp, 0);
    assert_eq!(inertia_hist.max_ulp, 0);
    assert_eq!(obs_hist.max_ulp, 0);
    assert_eq!(csum_f64_hist.max_ulp, 0);
    assert_eq!(csum_bits_hist.gt_one_ulp, 0);
}
