// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone unit and parity tests for custom `rate_tables_dir` loading,
//! safe-Rust HDF5 weak reaction table parsing, `eval_weak_reaction_pair_with_override`,
//! and `eval_ecapture_pair_with_tables_and_context` (`#161`).

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use flexmesa_chem::NuclideCatalog;
use flexmesa_rates::{
    eval_ecapture_pair_with_tables, eval_ecapture_pair_with_tables_and_context,
    eval_reaction_raw_with_overrides, eval_weak_reaction_pair,
    eval_weak_reaction_pair_with_override, load_rate_tables_dir,
    load_weak_table_hdf5_bytes, parse_weak_rate_name, CoulombMuiOption,
    CoulombVsOption, EcaptureTables, RatesCatalog, RatesError, TFactors,
};

static TEMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let n = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("{prefix}_{pid}_{n}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn pad8(buf: &mut Vec<u8>) {
    while !buf.len().is_multiple_of(8) {
        buf.push(0);
    }
}

fn push_v1_msg(buf: &mut Vec<u8>, mtype: u16, body: &[u8]) {
    buf.extend_from_slice(&mtype.to_le_bytes());
    buf.extend_from_slice(&(body.len() as u16).to_le_bytes());
    buf.extend_from_slice(&[0_u8; 4]);
    buf.extend_from_slice(body);
    pad8(buf);
}

fn build_v1_dataset_ohdr(
    dims: &[usize],
    is_f32_be: bool,
    data_addr: u64,
    byte_size: u64,
) -> Vec<u8> {
    let mut msgs = Vec::new();

    let mut dspace = vec![1_u8, dims.len() as u8, 0, 0, 0, 0, 0, 0];
    for &d in dims {
        dspace.extend_from_slice(&(d as u64).to_le_bytes());
    }
    for &d in dims {
        dspace.extend_from_slice(&(d as u64).to_le_bytes());
    }
    push_v1_msg(&mut msgs, 0x0001, &dspace);

    let mut dtype = Vec::with_capacity(20);
    if is_f32_be {
        dtype.extend_from_slice(&[0x11, 0x21, 0x00, 0x00]);
        dtype.extend_from_slice(&4_u32.to_le_bytes());
        dtype.extend_from_slice(&0_u16.to_le_bytes());
        dtype.extend_from_slice(&32_u16.to_le_bytes());
        dtype.extend_from_slice(&[23, 8, 0, 23]);
        dtype.extend_from_slice(&127_u32.to_le_bytes());
    } else {
        dtype.extend_from_slice(&[0x11, 0x20, 0x00, 0x00]);
        dtype.extend_from_slice(&8_u32.to_le_bytes());
        dtype.extend_from_slice(&0_u16.to_le_bytes());
        dtype.extend_from_slice(&64_u16.to_le_bytes());
        dtype.extend_from_slice(&[52, 11, 0, 52]);
        dtype.extend_from_slice(&1023_u32.to_le_bytes());
    }
    push_v1_msg(&mut msgs, 0x0003, &dtype);

    let mut layout = Vec::with_capacity(18);
    layout.extend_from_slice(&[3_u8, 1_u8]);
    layout.extend_from_slice(&data_addr.to_le_bytes());
    layout.extend_from_slice(&byte_size.to_le_bytes());
    push_v1_msg(&mut msgs, 0x0008, &layout);

    let mut oh = Vec::with_capacity(16 + msgs.len());
    oh.extend_from_slice(&[1_u8, 0_u8]);
    oh.extend_from_slice(&3_u16.to_le_bytes());
    oh.extend_from_slice(&1_u32.to_le_bytes());
    oh.extend_from_slice(&(msgs.len() as u32).to_le_bytes());
    oh.extend_from_slice(&0_u32.to_le_bytes());
    oh.extend_from_slice(&msgs);
    oh
}

/// Synthesizes a minimal valid HDF5 v0 image (superblock v0, v1 symbol table
/// `TREE`/`SNOD`/`HEAP`, v1 object headers, contiguous datasets).
fn build_hdf5_v0_weak(datasets: &[(&str, &[usize], &[f64])]) -> Vec<u8> {
    let mut buf = vec![0_u8; 96];
    buf[0..8].copy_from_slice(b"\x89HDF\r\n\x1a\n");
    buf[8] = 0; // superblock version 0
    buf[13] = 8; // offset size
    buf[14] = 8; // length size
    buf[16..18].copy_from_slice(&4_u16.to_le_bytes());
    buf[18..20].copy_from_slice(&16_u16.to_le_bytes());
    buf[32..40].copy_from_slice(&u64::MAX.to_le_bytes());
    buf[48..56].copy_from_slice(&u64::MAX.to_le_bytes());

    let root_oh_addr = buf.len() as u64;
    buf.extend_from_slice(&[1_u8, 0_u8]);
    buf.extend_from_slice(&1_u16.to_le_bytes());
    buf.extend_from_slice(&1_u32.to_le_bytes());
    buf.extend_from_slice(&24_u32.to_le_bytes());
    buf.extend_from_slice(&0_u32.to_le_bytes());
    let symtab_msg_pos = buf.len();
    buf.extend_from_slice(&0x0011_u16.to_le_bytes());
    buf.extend_from_slice(&16_u16.to_le_bytes());
    buf.extend_from_slice(&[0_u8; 4]);
    buf.extend_from_slice(&[0_u8; 16]); // patched with (bt_addr, hp_addr)

    let mut heap_seg = vec![0_u8];
    let mut name_offs = Vec::with_capacity(datasets.len());
    for &(name, _, _) in datasets {
        let off = heap_seg.len() as u64;
        heap_seg.extend_from_slice(name.as_bytes());
        heap_seg.push(0);
        name_offs.push(off);
    }
    while heap_seg.len() % 8 != 0 {
        heap_seg.push(0);
    }

    let hp_addr = buf.len() as u64;
    let data_seg_addr = hp_addr + 32;
    buf.extend_from_slice(b"HEAP");
    buf.extend_from_slice(&[0_u8; 4]);
    buf.extend_from_slice(&(heap_seg.len() as u64).to_le_bytes());
    buf.extend_from_slice(&u64::MAX.to_le_bytes());
    buf.extend_from_slice(&data_seg_addr.to_le_bytes());
    buf.extend_from_slice(&heap_seg);

    let mut data_addrs = Vec::with_capacity(datasets.len());
    let mut byte_sizes = Vec::with_capacity(datasets.len());
    for &(_, _, vals) in datasets {
        pad8(&mut buf);
        data_addrs.push(buf.len() as u64);
        byte_sizes.push((vals.len() * 8) as u64);
        for &v in vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    let mut ds_oh_addrs = Vec::with_capacity(datasets.len());
    for (idx, &(_, dims, _)) in datasets.iter().enumerate() {
        pad8(&mut buf);
        ds_oh_addrs.push(buf.len() as u64);
        let oh = build_v1_dataset_ohdr(
            dims,
            false,
            data_addrs[idx],
            byte_sizes[idx],
        );
        buf.extend_from_slice(&oh);
    }

    pad8(&mut buf);
    let snod_addr = buf.len() as u64;
    buf.extend_from_slice(b"SNOD");
    buf.push(1);
    buf.push(0);
    buf.extend_from_slice(&(datasets.len() as u16).to_le_bytes());
    for idx in 0..datasets.len() {
        buf.extend_from_slice(&name_offs[idx].to_le_bytes());
        buf.extend_from_slice(&ds_oh_addrs[idx].to_le_bytes());
        buf.extend_from_slice(&0_u32.to_le_bytes());
        buf.extend_from_slice(&0_u32.to_le_bytes());
        buf.extend_from_slice(&[0_u8; 16]);
    }

    pad8(&mut buf);
    let bt_addr = buf.len() as u64;
    buf.extend_from_slice(b"TREE");
    buf.push(0);
    buf.push(0);
    buf.extend_from_slice(&1_u16.to_le_bytes());
    buf.extend_from_slice(&u64::MAX.to_le_bytes());
    buf.extend_from_slice(&u64::MAX.to_le_bytes());
    buf.extend_from_slice(&0_u64.to_le_bytes());
    buf.extend_from_slice(&snod_addr.to_le_bytes());

    let eof_addr = buf.len() as u64;
    buf[40..48].copy_from_slice(&eof_addr.to_le_bytes());
    buf[64..72].copy_from_slice(&root_oh_addr.to_le_bytes());
    buf[72..76].copy_from_slice(&1_u32.to_le_bytes());
    buf[80..88].copy_from_slice(&bt_addr.to_le_bytes());
    buf[88..96].copy_from_slice(&hp_addr.to_le_bytes());
    buf[symtab_msg_pos + 8..symtab_msg_pos + 16]
        .copy_from_slice(&bt_addr.to_le_bytes());
    buf[symtab_msg_pos + 16..symtab_msg_pos + 24]
        .copy_from_slice(&hp_addr.to_le_bytes());

    buf
}

fn build_v2_link_msg(name: &str, target_oh: u64) -> Vec<u8> {
    let mut msg = vec![1_u8, 0x00];
    msg.push(name.len() as u8);
    msg.extend_from_slice(name.as_bytes());
    msg.extend_from_slice(&target_oh.to_le_bytes());
    msg
}

/// Synthesizes a minimal HDF5 v2 superblock + compact dataset (`f32` BE) image.
fn build_hdf5_v2_compact_weak_f32_be(
    t9s: &[f64],
    lyerhos: &[f64],
    ldec_c: &[f64],
    lcap_c: &[f64],
    lneu_c: &[f64],
) -> Vec<u8> {
    let mut buf = vec![0_u8; 48];
    buf[0..8].copy_from_slice(b"\x89HDF\r\n\x1a\n");
    buf[8] = 2; // superblock version 2
    buf[9] = 8;
    buf[10] = 8;
    buf[12..20].copy_from_slice(&0_u64.to_le_bytes());
    buf[20..28].copy_from_slice(&u64::MAX.to_le_bytes());

    let items: [(&str, &[usize], &[f64]); 5] = [
        ("T9s", &[t9s.len()], t9s),
        ("lYeRhos", &[lyerhos.len()], lyerhos),
        ("ldecay", &[lyerhos.len(), t9s.len()], ldec_c),
        ("lcapture", &[lyerhos.len(), t9s.len()], lcap_c),
        ("lneutrino", &[lyerhos.len(), t9s.len()], lneu_c),
    ];

    let mut link_msgs = Vec::new();
    for (name, dims, vals) in items {
        let mut msgs = Vec::new();
        let mut dspace = vec![2_u8, dims.len() as u8, 0_u8, 0_u8];
        for &d in dims {
            dspace.extend_from_slice(&(d as u64).to_le_bytes());
        }
        msgs.push(0x01_u8);
        msgs.extend_from_slice(&(dspace.len() as u16).to_le_bytes());
        msgs.push(0_u8);
        msgs.extend_from_slice(&dspace);

        let mut dtype = vec![0x11_u8, 0x21, 0x00, 0x00];
        dtype.extend_from_slice(&4_u32.to_le_bytes());
        dtype.extend_from_slice(&0_u16.to_le_bytes());
        dtype.extend_from_slice(&32_u16.to_le_bytes());
        dtype.extend_from_slice(&[23, 8, 0, 23]);
        dtype.extend_from_slice(&127_u32.to_le_bytes());
        msgs.push(0x03_u8);
        msgs.extend_from_slice(&(dtype.len() as u16).to_le_bytes());
        msgs.push(0_u8);
        msgs.extend_from_slice(&dtype);

        let raw_len = vals.len() * 4;
        let mut layout = Vec::with_capacity(4 + raw_len);
        layout.push(4_u8);
        layout.push(0_u8); // compact
        layout.extend_from_slice(&(raw_len as u16).to_le_bytes());
        for &v in vals {
            layout.extend_from_slice(&(v as f32).to_be_bytes());
        }
        msgs.push(0x08_u8);
        msgs.extend_from_slice(&(layout.len() as u16).to_le_bytes());
        msgs.push(0_u8);
        msgs.extend_from_slice(&layout);

        let oh_addr = buf.len() as u64;
        buf.extend_from_slice(b"OHDR");
        buf.push(2); // ver 2
        buf.push(0x01); // 2-byte chunk0 size
        let chunk_sz = (msgs.len() + 4) as u16;
        buf.extend_from_slice(&chunk_sz.to_le_bytes());
        buf.extend_from_slice(&msgs);
        buf.extend_from_slice(&0_u32.to_le_bytes()); // checksum placeholder

        let lm = build_v2_link_msg(name, oh_addr);
        link_msgs.push(0x06_u8);
        link_msgs.extend_from_slice(&(lm.len() as u16).to_le_bytes());
        link_msgs.push(0_u8);
        link_msgs.extend_from_slice(&lm);
    }

    let root_oh_addr = buf.len() as u64;
    buf.extend_from_slice(b"OHDR");
    buf.push(2);
    buf.push(0x01);
    let root_chunk_sz = (link_msgs.len() + 4) as u16;
    buf.extend_from_slice(&root_chunk_sz.to_le_bytes());
    buf.extend_from_slice(&link_msgs);
    buf.extend_from_slice(&0_u32.to_le_bytes());

    let eof_addr = buf.len() as u64;
    buf[28..36].copy_from_slice(&eof_addr.to_le_bytes());
    buf[36..44].copy_from_slice(&root_oh_addr.to_le_bytes());
    buf[44..48].copy_from_slice(&0_u32.to_le_bytes());
    buf
}

#[test]
fn test_parse_weak_rate_name_accepts_ecapture_and_beta_minus_and_rejects_invalid(
) {
    let (lhs, rhs, is_beta_minus) =
        parse_weak_rate_name("  r_n14_wk_c14 ").expect("valid _wk_ rate");
    assert_eq!(lhs, "n14");
    assert_eq!(rhs, "c14");
    assert!(!is_beta_minus);

    let (lhs, rhs, is_beta_minus) =
        parse_weak_rate_name("r_c14_wk-minus_n14").expect("valid _wk-minus_");
    assert_eq!(lhs, "c14");
    assert_eq!(rhs, "n14");
    assert!(is_beta_minus);

    assert!(matches!(
        parse_weak_rate_name("n14_wk_c14"),
        Err(RatesError::InvalidOption(_))
    ));
    assert!(matches!(
        parse_weak_rate_name("r_c14_ag_o18"),
        Err(RatesError::InvalidOption(_))
    ));
    assert!(matches!(
        parse_weak_rate_name("r_bogus123_wk_c14"),
        Err(RatesError::UnknownWeakPair(_, _))
    ));
}

#[test]
fn test_load_weak_table_hdf5_bytes_transposes_asymmetric_grid_and_evaluates_exact_nodes(
) {
    let t9s = [0.01_f64, 0.1, 0.5];
    let lyerhos = [5.0_f64, 7.0];
    // C-order on disk: shape (n_lye=2, n_t9=3) -> row 0 is lye=5.0, row 1 is lye=7.0.
    let ldecay_c = [-60.0, -50.0, -40.0, -58.0, -48.0, -38.0];
    let lcapture_c = [-12.0, -10.0, -8.0, -11.0, -9.0, -7.0];
    let lneutrino_c = [-13.0, -11.0, -9.0, -12.0, -10.0, -8.0];

    let bytes = build_hdf5_v0_weak(&[
        ("T9s", &[3], &t9s),
        ("lYeRhos", &[2], &lyerhos),
        ("ldecay", &[2, 3], &ldecay_c),
        ("lcapture", &[2, 3], &lcapture_c),
        ("lneutrino", &[2, 3], &lneutrino_c),
    ]);

    let tab =
        load_weak_table_hdf5_bytes(&bytes, "n14", "c14").expect("parse v0 H5");
    assert_eq!(tab.lhs_name, "n14");
    assert_eq!(tab.rhs_name, "c14");
    assert_eq!(tab.t9s, t9s);
    assert_eq!(tab.lyerhos, lyerhos);

    let nuclides = NuclideCatalog::builtin_ref();
    let q_expected = nuclides
        .get_by_id(usize::from(tab.lhs_id))
        .unwrap()
        .mass_excess
        - nuclides
            .get_by_id(usize::from(tab.rhs_id))
            .unwrap()
            .mass_excess;
    assert!((tab.q_tot - q_expected).abs() <= 1e-14);

    // Transposed row-major (nt9=3, nlye=2): index (it=2, ily=0) -> it*2 + ily = 4.
    assert_eq!(tab.lcapture[4], -8.0);
    assert_eq!(tab.lcapture[1], -11.0);

    // Evaluate at exact grid corner (t9=0.1, lye=7.0 => ye_rho=1e7).
    let ev_node = eval_weak_reaction_pair_with_override(
        "n14",
        "c14",
        0.1,
        1.0e7,
        true,
        Some(&tab),
    )
    .expect("override eval at node");
    let expected_lam = flexmesa_math::exp10(-48.0) + flexmesa_math::exp10(-9.0);
    assert!((ev_node.lambda - expected_lam).abs() <= 1e-22);
    assert!((ev_node.q - q_expected).abs() <= 1e-14);

    // Out-of-bounds below t9[0] and above lyerhos[1] clamps cleanly to corners.
    let ev_clamp = eval_weak_reaction_pair_with_override(
        "n14",
        "c14",
        0.001,
        1.0e9,
        false,
        Some(&tab),
    )
    .expect("clamped override eval");
    let expected_clamp =
        flexmesa_math::exp10(-58.0) + flexmesa_math::exp10(-11.0);
    assert!((ev_clamp.lambda - expected_clamp).abs() <= 1e-24);

    // Verify v2 superblock + compact f32 BE layout works identically.
    let v2_bytes = build_hdf5_v2_compact_weak_f32_be(
        &t9s,
        &lyerhos,
        &ldecay_c,
        &lcapture_c,
        &lneutrino_c,
    );
    let tab_v2 = load_weak_table_hdf5_bytes(&v2_bytes, "n14", "c14")
        .expect("parse v2 compact f32 H5");
    assert_eq!(tab_v2.lcapture[4], -8.0);

    // Corrupt payload, missing dataset, and shape mismatch must return errors.
    assert!(matches!(
        load_weak_table_hdf5_bytes(b"not-hdf5", "n14", "c14"),
        Err(RatesError::InvalidTableData(_))
    ));
    let missing_ds = build_hdf5_v0_weak(&[
        ("T9s", &[3], &t9s),
        ("lYeRhos", &[2], &lyerhos),
        ("ldecay", &[2, 3], &ldecay_c),
        ("lcapture", &[2, 3], &lcapture_c),
    ]);
    assert!(matches!(
        load_weak_table_hdf5_bytes(&missing_ds, "n14", "c14"),
        Err(RatesError::InvalidTableData(_))
    ));
    let bad_shape = build_hdf5_v0_weak(&[
        ("T9s", &[3], &t9s),
        ("lYeRhos", &[2], &lyerhos),
        ("ldecay", &[3, 2], &ldecay_c),
        ("lcapture", &[2, 3], &lcapture_c),
        ("lneutrino", &[2, 3], &lneutrino_c),
    ]);
    assert!(matches!(
        load_weak_table_hdf5_bytes(&bad_shape, "n14", "c14"),
        Err(RatesError::InvalidTableData(_))
    ));
}

#[test]
fn test_load_rate_tables_dir_loads_raw_and_weak_manifests() {
    let dir = unique_temp_dir("flexmesa_custom_rates_161");

    let t9s = [0.01_f64, 0.2];
    let lyerhos = [4.0_f64, 6.0];
    let ldecay_c = [-40.0, -30.0, -35.0, -25.0];
    let lcapture_c = [-20.0, -15.0, -18.0, -12.0];
    let lneutrino_c = [-21.0, -16.0, -19.0, -13.0];
    let h5 = build_hdf5_v0_weak(&[
        ("T9s", &[2], &t9s),
        ("lYeRhos", &[2], &lyerhos),
        ("ldecay", &[2, 2], &ldecay_c),
        ("lcapture", &[2, 2], &lcapture_c),
        ("lneutrino", &[2, 2], &lneutrino_c),
    ]);
    fs::write(dir.join("custom_n14_c14.h5"), &h5).expect("write h5");
    fs::write(
        dir.join("weak_rate_list.txt"),
        "! comment line\nr_n14_wk_c14   'custom_n14_c14.h5' ! inline note\n",
    )
    .expect("write weak_rate_list.txt");

    fs::write(
        dir.join("c12_ag_o16.txt"),
        "! 1D rate table (T8, rate)\n3\n0.1 1.0d-12\n1.0 2.5d-6\n5.0 4.0d-3\n",
    )
    .expect("write raw rate table");
    fs::write(dir.join("rate_list.txt"), "r_c12_ag_o16 'c12_ag_o16.txt'\n")
        .expect("write rate_list.txt");

    let loaded = load_rate_tables_dir(&dir).expect("load_rate_tables_dir");
    assert_eq!(loaded.raw_rates.len(), 1);
    assert_eq!(loaded.weak_tables.len(), 1);

    let weak_tab = loaded
        .weak_tables
        .get(&(String::from("n14"), String::from("c14")))
        .expect("n14->c14 weak table present");
    assert_eq!(weak_tab.id, 1);

    let cat = RatesCatalog::builtin();
    let ir = cat
        .reaction_id_by_name("r_c12_ag_o16")
        .expect("r_c12_ag_o16");
    let info = cat.reaction_by_id(ir).expect("reaction info");
    let temp = 1.0e8_f64;
    let tf = TFactors::from_temp(temp);
    let raw_rate = eval_reaction_raw_with_overrides(
        cat,
        info,
        temp,
        &tf,
        &loaded.raw_rates,
        None,
    )
    .expect("raw override eval");
    assert!((raw_rate - 2.5e-6).abs() <= 1e-18);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_eval_weak_reaction_pair_with_override_rejects_malformed_table_and_delegates_none(
) {
    let base = eval_weak_reaction_pair("be7", "li7", 0.02, 80.0, false)
        .expect("builtin be7->li7");
    let via_none = eval_weak_reaction_pair_with_override(
        "be7", "li7", 0.02, 80.0, false, None,
    )
    .expect("None override delegates identically");
    assert_eq!(base.lambda.to_bits(), via_none.lambda.to_bits());
    assert_eq!(base.q.to_bits(), via_none.q.to_bits());
    assert_eq!(base.qneu.to_bits(), via_none.qneu.to_bits());

    let mut bad_tab = RatesCatalog::builtin()
        .weak_by_pair("be7", "li7")
        .expect("be7->li7")
        .clone();
    bad_tab.ldecay.pop();
    assert!(matches!(
        eval_weak_reaction_pair_with_override(
            "be7",
            "li7",
            0.02,
            80.0,
            false,
            Some(&bad_tab)
        ),
        Err(RatesError::InvalidTableData(_))
    ));
}

fn assert_serde<T: serde::Serialize + serde::de::DeserializeOwned>() {}

#[test]
fn test_coulomb_options_support_serde_and_upstream_aliases() {
    assert_serde::<CoulombMuiOption>();
    assert_serde::<CoulombVsOption>();
    assert_eq!(
        CoulombMuiOption::from_name("DGC1973").unwrap(),
        CoulombMuiOption::Dcw1973
    );
    assert_eq!(
        CoulombMuiOption::from_name("I1993").unwrap(),
        CoulombMuiOption::Slr1982
    );
    assert_eq!(
        CoulombMuiOption::from_name("PCR2009").unwrap(),
        CoulombMuiOption::Pcr2009
    );
    assert_eq!(
        CoulombVsOption::from_name("ThomasFermi").unwrap(),
        CoulombVsOption::Throwe1983
    );
    assert_eq!(
        CoulombVsOption::from_name("Itoh2002").unwrap(),
        CoulombVsOption::Itoh2002
    );
}

#[test]
fn test_eval_ecapture_pair_with_tables_and_context_matches_exact_temp_rho() {
    let tables = EcaptureTables::builtin();
    let t9 = 0.2_f64;
    let ye_rho = 3.15e9_f64;
    let zbar = 11.0_f64;
    let abar = 22.0_f64;
    let temp = t9 * 1.0e9;
    let rho = ye_rho / (zbar / abar);

    let legacy = eval_ecapture_pair_with_tables(
        tables,
        "mg24",
        "na24",
        t9,
        ye_rho,
        320.0,
        -300.0,
        100.0,
        CoulombMuiOption::Pcr2009,
        CoulombVsOption::Itoh2002,
        zbar,
        abar,
    )
    .expect("legacy eval_ecapture_pair_with_tables");

    let ctx_same = eval_ecapture_pair_with_tables_and_context(
        tables,
        "mg24",
        "na24",
        temp,
        rho,
        t9,
        320.0,
        -300.0,
        100.0,
        CoulombMuiOption::Pcr2009,
        CoulombVsOption::Itoh2002,
        zbar,
        abar,
    )
    .expect("context eval matching legacy (temp, rho)");
    assert!(ctx_same.lambda > 0.0);
    assert_eq!(legacy.lambda.to_bits(), ctx_same.lambda.to_bits());
    assert_eq!(
        legacy.dlambda_dln_t.to_bits(),
        ctx_same.dlambda_dln_t.to_bits()
    );
    assert_eq!(
        legacy.dlambda_dln_rho.to_bits(),
        ctx_same.dlambda_dln_rho.to_bits()
    );

    // Supplying a distinct explicit plasma density (rho != ye_rho / (zbar / abar))
    // must change Coulomb screening/mui shifts when enabled.
    let ctx_shifted_rho = eval_ecapture_pair_with_tables_and_context(
        tables,
        "mg24",
        "na24",
        temp,
        rho * 1.5,
        t9,
        320.0,
        -300.0,
        100.0,
        CoulombMuiOption::Pcr2009,
        CoulombVsOption::Itoh2002,
        zbar,
        abar,
    )
    .expect("context eval with distinct rho");
    assert_ne!(ctx_same.lambda.to_bits(), ctx_shifted_rho.lambda.to_bits());

    let ctx_unscreened = eval_ecapture_pair_with_tables_and_context(
        tables,
        "mg24",
        "na24",
        temp,
        rho * 1.5,
        t9,
        320.0,
        -300.0,
        100.0,
        CoulombMuiOption::None,
        CoulombVsOption::None,
        zbar,
        abar,
    )
    .expect("unscreened context eval");
    assert_ne!(
        ctx_shifted_rho.lambda.to_bits(),
        ctx_unscreened.lambda.to_bits()
    );
}
