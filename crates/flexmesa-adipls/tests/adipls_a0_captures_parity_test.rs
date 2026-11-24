// SPDX-License-Identifier: GPL-3.0-or-later
//! Env-gated 0-ULP bit-exact parity check against the MESA 26.4.1 Package A0
//! ADIPLS captures archive (`FLEXMESA_ADIPLS_CAPTURES`).
//!
//! When `FLEXMESA_ADIPLS_CAPTURES` is unset or empty, this test logs a
//! diagnostic note and returns without invoking the external archive so CI runs
//! remain self-contained. When `FLEXMESA_ADIPLS_CAPTURES` is set, missing case
//! files or any bit mismatch (`> 0 ULP`) fail loudly.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use flexmesa_adipls::{
    fgong_amdl, parse_adipls_pruned_in, parse_redistrb_pruned_in,
    set_adipls_controls, sig_fac_adipls_support, sig_fac_astero_support,
    sig_from_nu_uhz, spcout_adi, store_amdl, AdiplsControlState, AdiplsError,
    AdiplsInput, Csummm, FgongModel, ObsSt,
};
use flexmesa_const::STANDARD_CGRAV;

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

const REDISTRB_2001: &str = "\
2001,,,
24,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
30,,,,,,,,,
,,,,,,,,,,,
";

const REDISTRB_2201: &str = "\
2201,,,
24,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
30,,,,10,,,,,
,,,,,,,,,,,
";

#[derive(Clone, Copy)]
struct CaseSpec {
    name: &'static str,
    redistrb_txt: &'static str,
    expected_nn_redist: i32,
    expected_iresa4: i32,
    expected_calls: usize,
    expected_modes: usize,
    uses_adipls_support_sig_fac: bool,
    nu_lower_factor: f64,
    nu_upper_factor: f64,
    /// Per-degree `(nu_first, nu_last)` targets (`0..=3`), `None` if `nl(l) == 0`.
    target_bounds: [Option<(f64, f64)>; 4],
}

const CASES: [CaseSpec; 5] = [
    CaseSpec {
        name: "astero_adipls",
        redistrb_txt: REDISTRB_2001,
        expected_nn_redist: 2001,
        expected_iresa4: 0,
        expected_calls: 10,
        expected_modes: 92,
        uses_adipls_support_sig_fac: true,
        nu_lower_factor: 1.0,
        nu_upper_factor: 1.0,
        target_bounds: [Some((50.0, 1000.0)), None, None, None],
    },
    CaseSpec {
        name: "surface_effects",
        redistrb_txt: REDISTRB_2001,
        expected_nn_redist: 2001,
        expected_iresa4: 0,
        expected_calls: 4,
        expected_modes: 110,
        uses_adipls_support_sig_fac: false,
        nu_lower_factor: 0.8,
        nu_upper_factor: 1.2,
        target_bounds: [
            Some((972.615, 3984.214)),
            Some((1185.604, 3913.570)),
            Some((1394.689, 3976.930)),
            Some((1591.536, 3762.530)),
        ],
    },
    CaseSpec {
        name: "example_astero",
        redistrb_txt: REDISTRB_2201,
        expected_nn_redist: 2201,
        expected_iresa4: 10,
        expected_calls: 65,
        expected_modes: 1079,
        uses_adipls_support_sig_fac: false,
        nu_lower_factor: 0.8,
        nu_upper_factor: 1.2,
        target_bounds: [
            Some((799.70, 1250.12)),
            Some((748.60, 1216.95)),
            Some((794.55, 1246.78)),
            None,
        ],
    },
    CaseSpec {
        name: "fast_scan_grid",
        redistrb_txt: REDISTRB_2201,
        expected_nn_redist: 2201,
        expected_iresa4: 10,
        expected_calls: 1107,
        expected_modes: 16301,
        uses_adipls_support_sig_fac: false,
        nu_lower_factor: 0.8,
        nu_upper_factor: 1.2,
        target_bounds: [
            Some((3223.0092, 3977.6098)),
            Some((3142.8886, 4049.0662)),
            Some((3208.5016, 3964.6278)),
            None,
        ],
    },
    CaseSpec {
        name: "fast_simplex",
        redistrb_txt: REDISTRB_2201,
        expected_nn_redist: 2201,
        expected_iresa4: 10,
        expected_calls: 979,
        expected_modes: 9779,
        uses_adipls_support_sig_fac: false,
        nu_lower_factor: 0.8,
        nu_upper_factor: 1.2,
        target_bounds: [Some((3223.1, 3373.8)), None, None, None],
    },
];

fn read_i32_le<R: Read>(r: &mut R) -> i32 {
    let mut buf = [0_u8; 4];
    r.read_exact(&mut buf).expect("read i32");
    i32::from_le_bytes(buf)
}

fn read_f64_vec_le<R: Read>(r: &mut R, count: usize) -> Vec<f64> {
    let mut bytes = vec![0_u8; count * 8];
    r.read_exact(&mut bytes).expect("read f64 vec");
    let (chunks, rem) = bytes.as_chunks::<8>();
    assert_eq!(rem, []);
    chunks.iter().map(|c| f64::from_le_bytes(*c)).collect()
}

fn read_fgong_bin(path: &Path) -> FgongModel {
    let mut f = BufReader::new(File::open(path).expect("open fgong.bin"));
    let iconst = usize::try_from(read_i32_le(&mut f)).expect("iconst >= 0");
    let ivar = usize::try_from(read_i32_le(&mut f)).expect("ivar >= 0");
    let nn_in = usize::try_from(read_i32_le(&mut f)).expect("nn_in >= 0");
    let glob = read_f64_vec_le(&mut f, iconst);
    let var = read_f64_vec_le(&mut f, ivar * nn_in);
    FgongModel::new(glob, ivar, nn_in, var).expect("valid captured FgongModel")
}

fn read_amdl_fgong_bin(path: &Path) -> (usize, [f64; 8], Vec<f64>) {
    let mut f = BufReader::new(File::open(path).expect("open amdl_fgong.bin"));
    let nn = usize::try_from(read_i32_le(&mut f)).expect("nn >= 0");
    let rows = usize::try_from(read_i32_le(&mut f)).expect("rows >= 0");
    assert_eq!(rows, 6, "expected 6 rows in {}", path.display());
    let data: [f64; 8] = read_f64_vec_le(&mut f, 8).try_into().unwrap();
    let aa = read_f64_vec_le(&mut f, 6 * nn);
    (nn, data, aa)
}

fn read_amdl_store_bin(path: &Path) -> (usize, [f64; 8], Vec<f64>, Vec<f64>) {
    let mut f = BufReader::new(File::open(path).expect("open amdl_store.bin"));
    let nn = usize::try_from(read_i32_le(&mut f)).expect("nn >= 0");
    let rows = usize::try_from(read_i32_le(&mut f)).expect("rows >= 0");
    assert_eq!(rows, 10, "expected 10 rows in {}", path.display());
    let data: [f64; 8] = read_f64_vec_le(&mut f, 8).try_into().unwrap();
    let x = read_f64_vec_le(&mut f, nn);
    let aa = read_f64_vec_le(&mut f, 10 * nn);
    (nn, data, x, aa)
}

fn parse_be_hex_bits(hex_str: &str) -> u64 {
    u64::from_str_radix(hex_str.trim(), 16)
        .unwrap_or_else(|e| panic!("invalid hex '{hex_str}': {e}"))
}

fn parse_be_hex_f64(hex_str: &str) -> f64 {
    f64::from_bits(parse_be_hex_bits(hex_str))
}

fn col_index(header: &[&str], name: &str) -> usize {
    header
        .iter()
        .position(|&h| h == name)
        .unwrap_or_else(|| panic!("missing column {name}"))
}

fn verify_stored_model(
    tap_dir: &Path,
    case_name: &str,
    store_id: usize,
    adipls_in: &AdiplsInput,
    fgong: &FgongModel,
) {
    // 1. Verify fgong_amdl bit-exact (0 ULP) against store_*.amdl_fgong.bin
    let fg_out = fgong_amdl(adipls_in.cst.cgrav, fgong).expect("fgong_amdl");
    let (nn_af, data_af, aa_af) = read_amdl_fgong_bin(
        &tap_dir.join(format!("store_{store_id:06}.amdl_fgong.bin")),
    );
    assert_eq!(fg_out.nn, nn_af);
    for k in 0..8 {
        assert_eq!(
            fg_out.data[k].to_bits(),
            data_af[k].to_bits(),
            "{case_name} store {store_id} amdl_fgong data[{k}]",
        );
    }
    for row in 1..=6 {
        let got_row = fg_out.aa.row(row).unwrap();
        for n in 0..nn_af {
            assert_eq!(
                got_row[n].to_bits(),
                aa_af[n * 6 + (row - 1)].to_bits(),
                "{case_name} store {store_id} amdl_fgong aa({row},{n})",
            );
        }
    }
    for row in 7..=10 {
        assert_eq!(fg_out.aa.row(row), Err(AdiplsError::UnsetAmdlRow { row }));
    }

    // 2. Verify store_amdl bit-exact (0 ULP) on x, data, aa rows 1..=5 & 10,
    //    and #479 explicit error on rows 6..=9.
    let st_out = store_amdl(
        fg_out.nn,
        adipls_in.int_ctrl.iriche,
        adipls_in.int_ctrl.iturpr,
        fg_out.data,
        &fg_out.aa,
    )
    .expect("store_amdl");
    let (nn_st, data_st, x_st, aa_st) = read_amdl_store_bin(
        &tap_dir.join(format!("store_{store_id:06}.amdl_store.bin")),
    );
    assert_eq!(st_out.nn, nn_st);
    for k in 0..8 {
        assert_eq!(
            st_out.data[k].to_bits(),
            data_st[k].to_bits(),
            "{case_name} store {store_id} amdl_store data[{k}]",
        );
    }
    for n in 0..nn_st {
        assert_eq!(
            st_out.x[n].to_bits(),
            x_st[n].to_bits(),
            "{case_name} store {store_id} x[{n}]",
        );
        for row in 1..=5 {
            assert_eq!(
                st_out.aa.get(row, n).unwrap().to_bits(),
                aa_st[n * 10 + (row - 1)].to_bits(),
                "{case_name} store {store_id} amdl_store aa({row},{n})",
            );
        }
        assert_eq!(
            st_out.aa.get(10, n).unwrap().to_bits(),
            aa_st[n * 10 + 9].to_bits(),
            "{case_name} store {store_id} amdl_store aa(10,{n})",
        );
    }
    for row in 6..=9 {
        assert_eq!(st_out.aa.row(row), Err(AdiplsError::UnsetAmdlRow { row }));
    }
}

fn verify_call_controls(
    spec: &CaseSpec,
    adipls_in: &AdiplsInput,
    header: &[&str],
    fields: &[&str],
    call_id: usize,
    fgong: &FgongModel,
) {
    let col = |name: &str| col_index(header, name);
    let mut ctrl = AdiplsControlState::from_input(adipls_in);
    let degree_f64 = parse_be_hex_f64(fields[col("para_el")]);
    let el_i32 = (0_i32..=10)
        .find(|&l| f64::from(l).to_bits() == degree_f64.to_bits())
        .unwrap_or_else(|| {
            panic!("unexpected para_el {degree_f64} in {}", spec.name)
        });
    let el_usize = usize::try_from(el_i32).unwrap();
    let (nu_lo, nu_hi) = spec.target_bounds[el_usize]
        .unwrap_or_else(|| panic!("unexpected el {el_i32} in {}", spec.name));
    let nu1 = spec.nu_lower_factor * nu_lo;
    let nu2 = spec.nu_upper_factor * nu_hi;
    let m_g = fgong.glob[0];
    let r_cm = fgong.glob[1];
    let sig_fac = if spec.uses_adipls_support_sig_fac {
        sig_fac_adipls_support(r_cm, STANDARD_CGRAV, m_g)
    } else {
        sig_fac_astero_support(r_cm, STANDARD_CGRAV, m_g)
    };
    let sig1 = sig_from_nu_uhz(sig_fac, nu1);
    let sig2 = sig_from_nu_uhz(sig_fac, nu2);
    let dfsig = parse_be_hex_f64(fields[col("para_dfsig")]);

    let nsel: i32 = fields[col("ipara_nsel")].parse().unwrap();
    let itrsig: i32 = fields[col("ipara_itrsig")].parse().unwrap();
    let nsig: i32 = fields[col("ipara_nsig")].parse().unwrap();
    let iscan: i32 = fields[col("ipara_iscan")].parse().unwrap();
    let irotkr: i32 = fields[col("irotkr")].parse().unwrap();
    let nprtkr: i32 = fields[col("nprtkr")].parse().unwrap();
    let igm1kr: i32 = fields[col("igm1kr")].parse().unwrap();
    let npgmkr: i32 = fields[col("npgmkr")].parse().unwrap();

    set_adipls_controls(
        &mut ctrl,
        el_i32,
        nsel,
        f64::from(el_i32),
        1.0_f64,
        itrsig,
        iscan,
        sig1,
        sig2,
        dfsig,
        nsig,
        irotkr,
        nprtkr,
        igm1kr,
        npgmkr,
    );

    let got_r = ctrl.cadr_param.to_array();
    let expected_real_cols = [
        "para_el",
        "para_els1",
        "para_dels",
        "para_dfsig1",
        "para_dfsig2",
        "para_sig1",
        "para_sig2",
        "para_dfsig",
        "para_eltrw1",
        "para_eltrw2",
        "para_sgtrw1",
        "para_sgtrw2",
    ];
    for (idx, col_name) in expected_real_cols.iter().enumerate() {
        let want_bits = parse_be_hex_bits(fields[col(col_name)]);
        assert_eq!(
            got_r[idx].to_bits(),
            want_bits,
            "{} call {call_id} {col_name}",
            spec.name
        );
    }

    let got_i = ctrl.cadi_param.to_array();
    let expected_int_cols = [
        "ipara_nsel",
        "ipara_nsig1",
        "ipara_nsig2",
        "ipara_itrsig",
        "ipara_nsig",
        "ipara_istsig",
        "ipara_inomd1",
        "ipara_iscan",
    ];
    for (idx, col_name) in expected_int_cols.iter().enumerate() {
        let want_val: i32 = fields[col(col_name)].parse().unwrap();
        assert_eq!(
            got_i[idx], want_val,
            "{} call {call_id} {col_name}",
            spec.name
        );
    }

    assert_eq!(
        ctrl.coutpt.to_array(),
        [irotkr, nprtkr, igm1kr, npgmkr],
        "{} call {call_id} coutpt",
        spec.name
    );
}

fn verify_case_modes(tap_dir: &Path, spec: &CaseSpec) -> usize {
    let modes_txt =
        fs::read_to_string(tap_dir.join("modes.tsv")).expect("read modes.tsv");
    let mut mlines = modes_txt.lines().filter(|l| !l.trim().is_empty());
    let _mheader = mlines.next().expect("modes.tsv header");
    let mut case_modes = 0_usize;
    for line in mlines {
        case_modes += 1;
        let parts: Vec<&str> = line.split('\t').collect();
        assert_eq!(parts.len(), 10);
        let expected_degree: i32 = parts[3].parse().unwrap();
        let expected_order: i32 = parts[4].parse().unwrap();
        let expected_az_order: i32 = parts[5].parse().unwrap();
        let want_freq_bits = parse_be_hex_bits(parts[6]);
        let want_inertia_bits = parse_be_hex_bits(parts[7]);

        let obs_vals: [f64; 10] = parts[8]
            .split_whitespace()
            .map(parse_be_hex_f64)
            .collect::<Vec<_>>()
            .try_into()
            .expect("10 obs_st values");
        let csum_words: [u64; 50] = parts[9]
            .split_whitespace()
            .map(parse_be_hex_bits)
            .collect::<Vec<_>>()
            .try_into()
            .expect("50 csummm values");

        let rec = spcout_adi(
            &ObsSt::from_array(obs_vals),
            &Csummm::from_bits(csum_words),
        )
        .expect("spcout_adi");
        assert_eq!(rec.el, expected_degree);
        assert_eq!(rec.order, expected_order);
        assert_eq!(rec.em, expected_az_order);
        assert_eq!(rec.cyclic_freq_uhz.to_bits(), want_freq_bits);
        assert_eq!(rec.inertia.to_bits(), want_inertia_bits);
    }
    assert_eq!(case_modes, spec.expected_modes);
    case_modes
}

fn verify_case_a0_captures(
    root: &Path,
    adipls_in: &AdiplsInput,
    spec: &CaseSpec,
) -> (usize, usize, usize) {
    let redist = parse_redistrb_pruned_in(spec.redistrb_txt)
        .expect("parse redistrb.c.pruned.in");
    assert_eq!(redist.nn, spec.expected_nn_redist);
    assert_eq!(redist.iresa4, spec.expected_iresa4);

    let tap_dir = root.join(spec.name).join("a0tap");
    assert!(
        tap_dir.is_dir(),
        "missing tap dir for {}: {}",
        spec.name,
        tap_dir.display()
    );

    let calls_txt =
        fs::read_to_string(tap_dir.join("calls.tsv")).expect("read calls.tsv");
    let mut lines = calls_txt.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines
        .next()
        .expect("calls.tsv header")
        .split('\t')
        .collect();

    let mut fgong_by_store: HashMap<usize, FgongModel> = HashMap::new();
    let mut case_stored = 0_usize;
    let mut case_calls = 0_usize;

    for line in lines {
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields.len(), header.len());
        case_calls += 1;

        let call_id: usize =
            fields[col_index(&header, "call_id")].parse().unwrap();
        let store_id: usize =
            fields[col_index(&header, "store_id")].parse().unwrap();
        let stored_before =
            fields[col_index(&header, "stored_before_call")] == "T";

        let fgong = fgong_by_store.entry(store_id).or_insert_with(|| {
            read_fgong_bin(
                &tap_dir.join(format!("store_{store_id:06}.fgong.bin")),
            )
        });

        if stored_before {
            case_stored += 1;
            verify_stored_model(
                &tap_dir, spec.name, store_id, adipls_in, fgong,
            );
        }

        verify_call_controls(spec, adipls_in, &header, &fields, call_id, fgong);
    }
    assert_eq!(case_calls, spec.expected_calls);
    let case_modes = verify_case_modes(&tap_dir, spec);
    (case_stored, case_calls, case_modes)
}

#[test]
fn test_a0_captures_bit_exact_models_controls_and_modes() {
    let Some(raw_dir) =
        std::env::var_os("FLEXMESA_ADIPLS_CAPTURES").filter(|v| !v.is_empty())
    else {
        eprintln!(
            "INFO: FLEXMESA_ADIPLS_CAPTURES unset; skipping external A0 captures parity test"
        );
        return;
    };
    let root = PathBuf::from(raw_dir);
    assert!(
        root.is_dir(),
        "FLEXMESA_ADIPLS_CAPTURES path does not exist: {}",
        root.display()
    );

    let adipls_in = parse_adipls_pruned_in(ADIPLS_PRUNED_IN)
        .expect("parse adipls.c.pruned.in");

    let mut total_stored = 0_usize;
    let mut total_calls = 0_usize;
    let mut total_modes = 0_usize;
    for spec in &CASES {
        let (stored, calls, modes) =
            verify_case_a0_captures(&root, &adipls_in, spec);
        total_stored += stored;
        total_calls += calls;
        total_modes += modes;
    }

    assert_eq!(total_stored, 1963);
    assert_eq!(total_calls, 2165);
    assert_eq!(total_modes, 27361);
}
