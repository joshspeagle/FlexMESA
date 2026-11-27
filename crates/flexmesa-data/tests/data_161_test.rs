// SPDX-License-Identifier: GPL-3.0-or-later
//! Boundary, JSON-escaping, offset-ordering, and concurrent atomic-save
//! tests for `flexmesa-data` (`#161`).

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::thread;

use flexmesa_data::converters::parse_helm_table_text;
use flexmesa_data::safetensors::{
    decode_safetensors_bytes, encode_safetensors_bytes, load_safetensors_file,
    save_safetensors_file, SafeTensorBundle, TensorData,
};

fn build_min_helm_text(imax: u32, jmax: u32) -> String {
    let mut out = String::new();
    for (sec, n_cols) in [(0u32, 9i32), (1, 4), (2, 4), (3, 4)] {
        for j in 0..jmax {
            for i in 0..imax {
                let val = f64::from(100 * sec + 10 * j + i + 1);
                for _ in 0..n_cols {
                    let _ = write!(out, "  {val:.16E}");
                }
                out.push('\n');
            }
        }
    }
    out
}

fn wrap_raw_safetensors(header_json: &str, payload: &[u8]) -> Vec<u8> {
    let mut padded = header_json.as_bytes().to_vec();
    while !padded.len().is_multiple_of(8) {
        padded.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + padded.len() + payload.len());
    out.extend_from_slice(&u64::try_from(padded.len()).unwrap().to_le_bytes());
    out.extend_from_slice(&padded);
    out.extend_from_slice(payload);
    out
}

#[test]
fn test_parse_helm_table_rejects_non_finite_bounds() {
    let text = build_min_helm_text(2, 2);
    for (r0, r1, t0, t1) in [
        (f64::NEG_INFINITY, 15.0, 3.0, 13.0),
        (-12.0, f64::INFINITY, 3.0, 13.0),
        (-12.0, 15.0, f64::NAN, 13.0),
        (-12.0, 15.0, 3.0, f64::INFINITY),
        (5.0, 5.0, 3.0, 13.0),
    ] {
        assert!(parse_helm_table_text(&text, 2, 2, r0, r1, t0, t1).is_err());
    }
}

#[test]
fn test_safetensors_json_escapes_and_surrogate_roundtrip() {
    let special_key = "k\x08\x0c\x01/\"\\\n\r\t";
    let meta = BTreeMap::from([
        ("note".to_owned(), "rocket:\x08\x0c\x1f\"\\".to_owned()),
        ("slash".to_owned(), "a/b".to_owned()),
    ]);
    let tensors = BTreeMap::from([(
        special_key.to_owned(),
        TensorData::f64(vec![2], vec![1.5, -2.25]),
    )]);
    let bundle = SafeTensorBundle {
        tensors,
        metadata: meta,
    };
    let encoded = encode_safetensors_bytes(&bundle, true).unwrap();
    let decoded = decode_safetensors_bytes(&encoded, true).unwrap();
    assert_eq!(decoded, bundle);

    // Explicit \b, \f, \/, and UTF-16 surrogate pair (\uD83D\uDE80) decoding.
    let mut payload = Vec::new();
    payload.extend_from_slice(&42.0_f64.to_le_bytes());
    let header = r#"{"__metadata__":{"emoji":"\uD83D\uDE80","bf":"a\b\f\/b"},"x":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}}"#;
    let raw = wrap_raw_safetensors(header, &payload);
    let parsed = decode_safetensors_bytes(&raw, true).unwrap();
    assert_eq!(parsed.metadata["emoji"], "🚀");
    assert_eq!(parsed.metadata["bf"], "a\x08\x0c/b");

    // Invalid escape or unpaired surrogate must return Err.
    let bad_escape =
        r#"{"x\q":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}}"#;
    assert!(decode_safetensors_bytes(
        &wrap_raw_safetensors(bad_escape, &payload),
        true,
    )
    .is_err());

    let bad_surr =
        r#"{"\uD83D":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}}"#;
    assert!(decode_safetensors_bytes(
        &wrap_raw_safetensors(bad_surr, &payload),
        true,
    )
    .is_err());
}

#[test]
fn test_safetensors_decodes_out_of_order_keys_and_rejects_gaps() {
    let mut payload = Vec::new();
    payload.extend_from_slice(&1.0_f64.to_le_bytes());
    payload.extend_from_slice(&2.0_f64.to_le_bytes());

    // Tensor keys in non-offset JSON order ("zeta" -> [0,8], "alpha" -> [8,16]).
    let header = r#"{"zeta":{"dtype":"F64","shape":[1],"data_offsets":[0,8]},"alpha":{"dtype":"F64","shape":[1],"data_offsets":[8,16]}}"#;
    let decoded =
        decode_safetensors_bytes(&wrap_raw_safetensors(header, &payload), true)
            .unwrap();
    assert_eq!(decoded.tensors["zeta"].as_f64_slice().unwrap(), &[1.0]);
    assert_eq!(decoded.tensors["alpha"].as_f64_slice().unwrap(), &[2.0]);

    // Gap between tensors ([0,8] and [16,24]) must still fail loudly.
    let mut payload_gap = vec![0u8; 24];
    payload_gap[..8].copy_from_slice(&1.0_f64.to_le_bytes());
    payload_gap[16..24].copy_from_slice(&2.0_f64.to_le_bytes());
    let gap_header = r#"{"zeta":{"dtype":"F64","shape":[1],"data_offsets":[0,8]},"alpha":{"dtype":"F64","shape":[1],"data_offsets":[16,24]}}"#;
    assert!(decode_safetensors_bytes(
        &wrap_raw_safetensors(gap_header, &payload_gap),
        true,
    )
    .is_err());
}

#[test]
fn test_save_safetensors_file_concurrent_atomic_and_clean() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("concurrent.safetensors");
    let bundle = SafeTensorBundle {
        tensors: BTreeMap::from([(
            "vals".to_owned(),
            TensorData::f64(vec![2], vec![3.25, -4.5]),
        )]),
        metadata: BTreeMap::from([("tag".to_owned(), "thread".to_owned())]),
    };

    thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| {
                let sha =
                    save_safetensors_file(&target, &bundle, true).unwrap();
                let loaded =
                    load_safetensors_file(&target, Some(&sha), true).unwrap();
                assert_eq!(loaded, bundle);
            });
        }
    });

    // Ensure no temp files remain after concurrent writes.
    let leftovers: Vec<_> = fs::read_dir(dir.path())
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
        .collect();
    assert!(leftovers.is_empty(), "leaked temp files: {leftovers:?}");
}
