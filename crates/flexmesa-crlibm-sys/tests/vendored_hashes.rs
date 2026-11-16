// SPDX-License-Identifier: LGPL-3.0-or-later
//! Provenance, SHA-256 integrity, safety-comment, and multi-thread reentrancy
//! tests for `flexmesa-crlibm-sys` (`#165`).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;

const K256: [u32; 64] = [
    0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5,
    0x3956_c25b, 0x59f1_11f1, 0x923f_82a4, 0xab1c_5ed5,
    0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3,
    0x72be_5d74, 0x80de_b1fe, 0x9bdc_06a7, 0xc19b_f174,
    0xe49b_69c1, 0xefbe_4786, 0x0fc1_9dc6, 0x240c_a1cc,
    0x2de9_2c6f, 0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da,
    0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7,
    0xc6e0_0bf3, 0xd5a7_9147, 0x06ca_6351, 0x1429_2967,
    0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc, 0x5338_0d13,
    0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85,
    0xa2bf_e8a1, 0xa81a_664b, 0xc24b_8b70, 0xc76c_51a3,
    0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070,
    0x19a4_c116, 0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5,
    0x391c_0cb3, 0x4ed8_aa4a, 0x5b9c_ca4f, 0x682e_6ff3,
    0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208,
    0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7, 0xc671_78f2,
];

fn sha256_hex(data: &[u8]) -> String {
    let mut state: [u32; 8] = [
        0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372, 0xa54f_f53a,
        0x510e_527f, 0x9b05_688c, 0x1f83_d9ab, 0x5be0_cd19,
    ];
    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0x00);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for block in msg.as_chunks::<64>().0 {
        let mut sched = [0u32; 64];
        for (idx, word) in block.as_chunks::<4>().0.iter().enumerate() {
            sched[idx] = u32::from_be_bytes(*word);
        }
        for idx in 16..64 {
            let s0 = sched[idx - 15].rotate_right(7)
                ^ sched[idx - 15].rotate_right(18)
                ^ (sched[idx - 15] >> 3);
            let s1 = sched[idx - 2].rotate_right(17)
                ^ sched[idx - 2].rotate_right(19)
                ^ (sched[idx - 2] >> 10);
            sched[idx] = sched[idx - 16]
                .wrapping_add(s0)
                .wrapping_add(sched[idx - 7])
                .wrapping_add(s1);
        }
        let mut reg = state;
        for idx in 0..64 {
            let s1 =
                reg[4].rotate_right(6) ^ reg[4].rotate_right(11) ^ reg[4].rotate_right(25);
            let ch = (reg[4] & reg[5]) ^ ((!reg[4]) & reg[6]);
            let temp1 = reg[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K256[idx])
                .wrapping_add(sched[idx]);
            let s0 =
                reg[0].rotate_right(2) ^ reg[0].rotate_right(13) ^ reg[0].rotate_right(22);
            let maj = (reg[0] & reg[1]) ^ (reg[0] & reg[2]) ^ (reg[1] & reg[2]);
            let temp2 = s0.wrapping_add(maj);

            reg[7] = reg[6];
            reg[6] = reg[5];
            reg[5] = reg[4];
            reg[4] = reg[3].wrapping_add(temp1);
            reg[3] = reg[2];
            reg[2] = reg[1];
            reg[1] = reg[0];
            reg[0] = temp1.wrapping_add(temp2);
        }
        for (acc, val) in state.iter_mut().zip(reg) {
            *acc = acc.wrapping_add(val);
        }
    }

    let mut out = String::with_capacity(64);
    for word in state {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{word:08x}");
    }
    out
}

fn collect_files(base: &Path, dir: &Path, out: &mut Vec<String>) {
    for entry in fs::read_dir(dir).expect("read_dir") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.is_dir() {
            collect_files(base, &path, out);
        } else if path.file_name().and_then(|s| s.to_str()) != Some(".gitkeep") {
            let rel = path
                .strip_prefix(base)
                .expect("strip_prefix")
                .to_string_lossy()
                .replace('\\', "/");
            out.push(rel);
        }
    }
}

#[test]
fn test_vendored_manifest_hashes_match_all_csrc_and_config_files() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let vendored_toml =
        fs::read_to_string(manifest_dir.join("VENDORED.toml")).expect("VENDORED.toml");

    assert!(vendored_toml.contains("crmath_version = \"1.2\""));
    assert!(vendored_toml.contains("crlibm_version = \"1.0beta4\""));

    let mut recorded: BTreeMap<String, String> = BTreeMap::new();
    let mut current_path: Option<String> = None;
    for line in vendored_toml.lines() {
        let trimmed = line.trim();
        if trimmed == "[[files]]" {
            current_path = None;
        } else if let Some(rest) = trimmed.strip_prefix("path = \"") {
            current_path = Some(rest.trim_end_matches('"').to_string());
        } else if let Some(rest) = trimmed.strip_prefix("sha256 = \"") {
            let sha = rest.trim_end_matches('"').to_string();
            let path = current_path.take().expect("sha256 preceded by path");
            recorded.insert(path, sha);
        }
    }
    assert_eq!(recorded.len(), 52, "expected 52 tracked vendored/config files");

    for required_license in [
        "csrc/COPYING.LIB",
        "csrc/COPYING",
        "csrc/scs_lib/COPYING",
    ] {
        assert!(
            recorded.contains_key(required_license),
            "missing license file {required_license}"
        );
    }

    let mut disk_files = Vec::new();
    collect_files(&manifest_dir, &manifest_dir.join("csrc"), &mut disk_files);
    collect_files(&manifest_dir, &manifest_dir.join("config"), &mut disk_files);
    disk_files.sort();

    let recorded_keys: Vec<String> = recorded.keys().cloned().collect();
    assert_eq!(
        disk_files, recorded_keys,
        "on-disk files under csrc/ and config/ must match VENDORED.toml 1-to-1"
    );

    for (rel, expected_sha) in &recorded {
        let bytes = fs::read(manifest_dir.join(rel)).expect("read tracked file");
        let actual_sha = sha256_hex(&bytes);
        assert_eq!(
            &actual_sha, expected_sha,
            "SHA-256 mismatch on vendored file {rel}"
        );
    }
}

#[test]
fn test_lib_rs_enforces_unsafe_invariants_and_safety_comments() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src = fs::read_to_string(manifest_dir.join("src/lib.rs")).expect("src/lib.rs");
    assert!(src.contains("#![deny(unsafe_op_in_unsafe_fn)]"));
    assert!(
        !src.contains("pub unsafe fn"),
        "flexmesa-crlibm-sys must expose only safe public functions"
    );

    let lines: Vec<&str> = src.lines().collect();
    let mut unsafe_blocks = 0usize;
    for (idx, line) in lines.iter().enumerate() {
        if line.trim_start().starts_with("unsafe {") {
            unsafe_blocks += 1;
            let prev_comment: String = lines[idx.saturating_sub(5)..idx].join("\n");
            assert!(
                prev_comment.contains("// SAFETY:"),
                "unsafe block at line {} lacks a preceding // SAFETY: comment",
                idx + 1
            );
        }
    }
    assert_eq!(unsafe_blocks, 21, "expected 21 documented extern call sites");
}

#[test]
fn test_crlibm_init_is_noop_and_entries_are_thread_reentrant() {
    assert_eq!(flexmesa_crlibm_sys::crlibm_init(), 0);

    let eval_digest = || -> u64 {
        let mut acc = 0xcbf2_9ce4_8422_2325_u64;
        for step in 1..=256_u32 {
            let arg = f64::from(step) * 0.003_906_25;
            let half = arg * 0.5;
            for val in [
                flexmesa_crlibm_sys::log_rz(arg),
                flexmesa_crlibm_sys::log1p_rz(arg),
                flexmesa_crlibm_sys::log2_rz(arg),
                flexmesa_crlibm_sys::log10_rz(arg),
                flexmesa_crlibm_sys::exp_rd(arg),
                flexmesa_crlibm_sys::expm1_rz(arg),
                flexmesa_crlibm_sys::cos_rz(arg),
                flexmesa_crlibm_sys::sin_rz(arg),
                flexmesa_crlibm_sys::tan_rz(arg),
                flexmesa_crlibm_sys::cospi_rz(arg),
                flexmesa_crlibm_sys::sinpi_rz(arg),
                flexmesa_crlibm_sys::tanpi_rz(arg * 0.49),
                flexmesa_crlibm_sys::acos_rd(half),
                flexmesa_crlibm_sys::asin_rz(half),
                flexmesa_crlibm_sys::atan_rz(arg),
                flexmesa_crlibm_sys::acospi_rd(half),
                flexmesa_crlibm_sys::asinpi_rz(half),
                flexmesa_crlibm_sys::atanpi_rz(arg),
                flexmesa_crlibm_sys::cosh_rz(arg),
                flexmesa_crlibm_sys::sinh_rz(arg),
            ] {
                acc ^= val.to_bits();
                acc = acc.wrapping_mul(0x0100_0000_01b3);
            }
        }
        acc
    };

    let expected = eval_digest();
    let handles: Vec<_> = (0..4).map(|_| thread::spawn(eval_digest)).collect();
    for handle in handles {
        assert_eq!(handle.join().expect("worker thread"), expected);
    }
}
