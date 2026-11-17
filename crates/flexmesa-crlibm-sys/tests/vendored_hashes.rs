// SPDX-License-Identifier: LGPL-3.0-or-later
//! Provenance, SHA-256 integrity, safety-comment, and multi-thread reentrancy
//! tests for `flexmesa-crlibm-sys` (`#165`).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;

use flexmesa_utils::sha256::compute_sha256_bytes;

fn collect_files(base: &Path, dir: &Path, out: &mut Vec<String>) {
    for entry in fs::read_dir(dir).expect("read_dir") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.is_dir() {
            collect_files(base, &path, out);
        } else if path.file_name().and_then(|s| s.to_str()) != Some(".gitkeep")
        {
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
    let vendored_toml = fs::read_to_string(manifest_dir.join("VENDORED.toml"))
        .expect("VENDORED.toml");

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
    assert_eq!(
        recorded.len(),
        54,
        "expected 54 tracked vendored/config files"
    );

    for required_license in
        ["csrc/COPYING.LIB", "csrc/COPYING", "csrc/scs_lib/COPYING"]
    {
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
        let bytes =
            fs::read(manifest_dir.join(rel)).expect("read tracked file");
        let actual_sha = compute_sha256_bytes(&bytes);
        assert_eq!(
            &actual_sha, expected_sha,
            "SHA-256 mismatch on vendored file {rel}"
        );
    }
}

#[test]
fn test_lib_rs_enforces_unsafe_invariants_and_safety_comments() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut src_files = Vec::new();
    collect_files(&manifest_dir, &manifest_dir.join("src"), &mut src_files);
    assert_eq!(
        src_files,
        vec![String::from("src/lib.rs")],
        "flexmesa-crlibm-sys/src/ must contain only src/lib.rs"
    );

    let src = fs::read_to_string(manifest_dir.join("src/lib.rs"))
        .expect("src/lib.rs");
    assert!(src.contains("#![deny(unsafe_op_in_unsafe_fn)]"));
    assert!(
        !src.contains("unsafe fn")
            && !src.contains("unsafe extern")
            && !src.contains("unsafe impl"),
        "flexmesa-crlibm-sys must expose only safe Rust functions/traits"
    );

    // Cross-check `build.rs` C_SOURCES against `VENDORED.toml` `.c` files.
    let build_rs =
        fs::read_to_string(manifest_dir.join("build.rs")).expect("build.rs");
    let vendored_toml = fs::read_to_string(manifest_dir.join("VENDORED.toml"))
        .expect("VENDORED.toml");
    for line in vendored_toml.lines() {
        if let Some(rest) = line.trim().strip_prefix("path = \"") {
            let rel = rest.trim_end_matches('"');
            let is_standalone = rel.ends_with("-standalone.c");
            if std::path::Path::new(rel)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("c"))
                && !is_standalone
            {
                assert!(
                    build_rs.contains(&format!("\"{rel}\"")),
                    "build.rs C_SOURCES is missing tracked C file {rel}"
                );
            }
        }
    }

    let lines: Vec<&str> = src.lines().collect();
    let mut unsafe_blocks = 0usize;
    for (idx, line) in lines.iter().enumerate() {
        if line.trim_start().starts_with("unsafe {") {
            unsafe_blocks += 1;
            let prev_comment: String =
                lines[idx.saturating_sub(5)..idx].join("\n");
            assert!(
                prev_comment.contains("// SAFETY:"),
                "unsafe block at line {} lacks a preceding // SAFETY: comment",
                idx + 1
            );
        }
    }
    assert_eq!(
        unsafe_blocks, 26,
        "expected 26 documented extern call sites"
    );
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
                flexmesa_crlibm_sys::log_rn(arg),
                flexmesa_crlibm_sys::log10_rn(arg),
                flexmesa_crlibm_sys::exp_rn(arg),
                flexmesa_crlibm_sys::atan_rn(arg),
                flexmesa_crlibm_sys::pow_rn(arg, half),
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
