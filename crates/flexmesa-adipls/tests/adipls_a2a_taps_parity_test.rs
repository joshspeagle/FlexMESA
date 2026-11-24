// SPDX-License-Identifier: GPL-3.0-or-later
//! Env-gated 0-ULP bit-exact parity check for Package A2a (`mdintg = 5`
//! scan and solve core) against the 5 representative MESA 26.4.1 Package A0
//! `ENTER`/`DET`/`MATCH`/`ITC` tap oracle files (`#171`).
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
    AdiplsSolver, AmdlAa, NoopSigoutHandler, TapRecorder,
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
    expected_enter: usize,
    expected_det: usize,
    expected_match: usize,
    expected_itc: usize,
}

const ORACLE_CASES: [OracleCase; 5] = [
    OracleCase {
        case_name: "astero_adipls",
        oracle_file: "astero__astero_adipls__c000001.taps.tsv",
        call_id: 1,
        expected_amdl: "call_000001.amdl.bin",
        expected_enter: 1,
        expected_det: 236,
        expected_match: 7,
        expected_itc: 7,
    },
    OracleCase {
        case_name: "example_astero",
        oracle_file: "astero__example_astero__c000029.taps.tsv",
        call_id: 29,
        expected_amdl: "call_000027.amdl.bin",
        expected_enter: 1,
        expected_det: 358,
        expected_match: 23,
        expected_itc: 23,
    },
    OracleCase {
        case_name: "fast_scan_grid",
        oracle_file: "astero__fast_scan_grid__c000211.taps.tsv",
        call_id: 211,
        expected_amdl: "call_000210.amdl.bin",
        expected_enter: 1,
        expected_det: 185,
        expected_match: 16,
        expected_itc: 16,
    },
    OracleCase {
        case_name: "surface_effects",
        oracle_file: "astero__surface_effects__c000002.taps.tsv",
        call_id: 2,
        expected_amdl: "call_000002.amdl.bin",
        expected_enter: 1,
        expected_det: 448,
        expected_match: 27,
        expected_itc: 27,
    },
    OracleCase {
        case_name: "surface_effects",
        oracle_file: "astero__surface_effects__c000004.taps.tsv",
        call_id: 4,
        expected_amdl: "call_000004.amdl.bin",
        expected_enter: 1,
        expected_det: 375,
        expected_match: 24,
        expected_itc: 24,
    },
];

struct CallRow {
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

#[derive(Debug)]
struct ExpectedTap {
    tag: String,
    values: Vec<u64>,
}

fn captures_root_or_skip() -> Option<PathBuf> {
    let raw = std::env::var("FLEXMESA_ADIPLS_CAPTURES").ok();
    let path_str = raw.as_deref().map_or("", str::trim);
    if path_str.is_empty() {
        eprintln!(
            "INFO: FLEXMESA_ADIPLS_CAPTURES unset; skipping external A2a captures parity test"
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

fn parse_hex_f64(hex: &str) -> f64 {
    let bits = u64::from_str_radix(hex.trim(), 16)
        .unwrap_or_else(|err| panic!("bad hex binary64 `{hex}`: {err}"));
    f64::from_bits(bits)
}

fn read_call_row(calls_path: &Path, target_call_id: usize) -> CallRow {
    let text = fs::read_to_string(calls_path).unwrap_or_else(|err| {
        panic!("failed to read `{}`: {err}", calls_path.display())
    });
    for (line_idx, line) in text.lines().enumerate() {
        if line_idx == 0 || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert!(
            cols.len() >= 34,
            "calls.tsv row {} in `{}` expected >= 34 columns, got {}",
            line_idx + 1,
            calls_path.display(),
            cols.len()
        );
        let call_id: usize = cols[0].parse().unwrap_or_else(|err| {
            panic!("bad call_id `{}`: {err}", cols[0]);
        });
        if call_id != target_call_id {
            continue;
        }
        let parse_i32 = |col: usize| -> i32 {
            cols[col].trim().parse().unwrap_or_else(|err| {
                panic!("bad i32 col[{col}] `{}`: {err}", cols[col]);
            })
        };
        let nn: usize = cols[4].trim().parse().unwrap_or_else(|err| {
            panic!("bad nn `{}`: {err}", cols[4]);
        });
        return CallRow {
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
        };
    }
    panic!(
        "call_id={target_call_id} not found in `{}`",
        calls_path.display()
    );
}

fn read_amdl_bin_strict_a2a(
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

    // Load ONLY 1-based rows 1..=5 and 10 so rows 6..=9 stay unset (#479).
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

    // Sanity check central value: at x=0, aa(5, 1) == 3.0.
    if x[0].to_bits() == 0 {
        assert!(
            (aa.get(5, 0).unwrap() - 3.0_f64).abs() < 1.0e-12_f64,
            "unexpected central aa(5, 1)={}",
            aa.get(5, 0).unwrap()
        );
    }

    (data, x, aa)
}

fn read_expected_a2a_taps(oracle_path: &Path) -> Vec<ExpectedTap> {
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
        if !matches!(tag, "ENTER" | "DET" | "MATCH" | "ITC") {
            // `KINER`, `VARFRQ`, etc. belong to Package A2b post-processing.
            continue;
        }
        let count: usize = parts[1].trim().parse().unwrap_or_else(|err| {
            panic!("bad field count on line {}: {err}", line_no + 1)
        });
        let values: Vec<u64> = if count == 0 {
            Vec::new()
        } else {
            let hex_col = parts.get(2).copied().unwrap_or("");
            hex_col
                .split_whitespace()
                .map(|h| {
                    u64::from_str_radix(h, 16).unwrap_or_else(|err| {
                        panic!("bad hex `{h}` on line {}: {err}", line_no + 1)
                    })
                })
                .collect()
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

#[test]
#[allow(clippy::too_many_lines)]
fn test_adipls_a2a_bit_exact_taps_against_representative_oracles() {
    let Some(captures_root) = captures_root_or_skip() else {
        return;
    };

    let oracles_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/fortran_mesa/adipls_oracles/data");
    let input = parse_adipls_pruned_in(ADIPLS_PRUNED_IN)
        .expect("parse ADIPLS_PRUNED_IN");

    let mut total_enter = 0_usize;
    let mut total_det = 0_usize;
    let mut total_match = 0_usize;
    let mut total_itc = 0_usize;

    for spec in &ORACLE_CASES {
        let calls_path =
            captures_root.join(spec.case_name).join("a0tap/calls.tsv");
        let row = read_call_row(&calls_path, spec.call_id);
        assert_eq!(
            row.amdl_file, spec.expected_amdl,
            "amdl filename mismatch for {} call {}",
            spec.case_name, spec.call_id
        );

        let amdl_path = captures_root
            .join(spec.case_name)
            .join("a0tap")
            .join(&row.amdl_file);
        let (data, x, aa) = read_amdl_bin_strict_a2a(&amdl_path, row.nn);

        #[allow(clippy::cast_possible_truncation)]
        let el_i32 = row.el as i32;
        assert_eq!(f64::from(el_i32).to_bits(), row.el.to_bits());

        let mut controls = AdiplsControlState::from_input(&input);
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

        assert_eq!(controls.cadr_param.el.to_bits(), row.el.to_bits());
        assert_eq!(controls.cadr_param.els1.to_bits(), row.els1.to_bits());
        assert_eq!(controls.cadr_param.dels.to_bits(), row.dels.to_bits());
        assert_eq!(controls.cadr_param.dfsig1.to_bits(), row.dfsig1.to_bits());
        assert_eq!(controls.cadr_param.dfsig2.to_bits(), row.dfsig2.to_bits());
        assert_eq!(controls.cadr_param.sig1.to_bits(), row.sig1.to_bits());
        assert_eq!(controls.cadr_param.sig2.to_bits(), row.sig2.to_bits());
        assert_eq!(controls.cadr_param.dfsig.to_bits(), row.dfsig.to_bits());
        assert_eq!(controls.cadr_param.eltrw1.to_bits(), row.eltrw1.to_bits());
        assert_eq!(controls.cadr_param.eltrw2.to_bits(), row.eltrw2.to_bits());
        assert_eq!(controls.cadr_param.sgtrw1.to_bits(), row.sgtrw1.to_bits());
        assert_eq!(controls.cadr_param.sgtrw2.to_bits(), row.sgtrw2.to_bits());
        assert_eq!(controls.cadi_param.nsel, row.nsel);
        assert_eq!(controls.cadi_param.nsig1, row.nsig1);
        assert_eq!(controls.cadi_param.nsig2, row.nsig2);
        assert_eq!(controls.cadi_param.itrsig, row.itrsig);
        assert_eq!(controls.cadi_param.nsig, row.nsig);
        assert_eq!(controls.cadi_param.istsig, row.istsig);
        assert_eq!(controls.cadi_param.inomd1, row.inomd1);
        assert_eq!(controls.cadi_param.iscan, row.iscan);
        assert_eq!(controls.coutpt.irotkr, row.irotkr);
        assert_eq!(controls.coutpt.nprtkr, row.nprtkr);
        assert_eq!(controls.coutpt.igm1kr, row.igm1kr);
        assert_eq!(controls.coutpt.npgmkr, row.npgmkr);

        let solver = AdiplsSolver::new(input.clone());
        let mut recorder = TapRecorder::new();
        let mut sigout = NoopSigoutHandler;
        solver
            .run_call_with_hooks(
                &controls,
                &x,
                &aa,
                data,
                &mut recorder,
                &mut sigout,
            )
            .unwrap_or_else(|err| {
                panic!(
                    "solver failed for {} call {}: {err}",
                    spec.case_name, spec.call_id
                )
            });

        let oracle_path = oracles_dir.join(spec.oracle_file);
        let expected = read_expected_a2a_taps(&oracle_path);

        assert_eq!(
            recorder.events.len(),
            expected.len(),
            "tap count mismatch for {}: got {}, expected {}",
            spec.oracle_file,
            recorder.events.len(),
            expected.len()
        );

        let mut enter_count = 0_usize;
        let mut det_count = 0_usize;
        let mut match_count = 0_usize;
        let mut itc_count = 0_usize;

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
                "ENTER" => enter_count += 1,
                "DET" => det_count += 1,
                "MATCH" => match_count += 1,
                "ITC" => itc_count += 1,
                other => panic!("unexpected tag {other}"),
            }
            let got_vals = got_ev.to_values();
            assert_eq!(
                got_vals.len(),
                exp_ev.values.len(),
                "value len mismatch at event {idx} ({}) in {}",
                got_ev.tag(),
                spec.oracle_file
            );
            for (field_idx, (&got_f64, &exp_bits)) in
                got_vals.iter().zip(exp_ev.values.iter()).enumerate()
            {
                let got_bits = got_f64.to_bits();
                if got_bits != exp_bits {
                    let exp_f64 = f64::from_bits(exp_bits);
                    let ulp = ulp_distance(got_bits, exp_bits);
                    panic!(
                        "0-ULP mismatch in {} at event #{idx} {} field[{field_idx}]: \
                         got {got_bits:016X} ({got_f64:.17e}), expected {exp_bits:016X} \
                         ({exp_f64:.17e}), ulp={ulp}",
                        spec.oracle_file,
                        got_ev.tag()
                    );
                }
            }
        }

        assert_eq!(enter_count, spec.expected_enter);
        assert_eq!(det_count, spec.expected_det);
        assert_eq!(match_count, spec.expected_match);
        assert_eq!(itc_count, spec.expected_itc);

        total_enter += enter_count;
        total_det += det_count;
        total_match += match_count;
        total_itc += itc_count;
    }

    assert_eq!(total_enter, 5);
    assert_eq!(total_det, 1602);
    assert_eq!(total_match, 97);
    assert_eq!(total_itc, 97);
}
