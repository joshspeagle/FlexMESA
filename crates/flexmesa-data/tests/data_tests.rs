// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration and property-based tests for `flexmesa-data`.
//!
//! Verifies:
//! 1. FIPS 180-4 `SHA-256` test vectors and `manifest.toml` parsing.
//! 2. `.safetensors` `F64` and `I64` round-trip encoding, decoding, file I/O,
//!    checksum verification, and non-finite float rejection.
//! 3. Bit-exact conversion of `lodders03.data`, `isotopes.data`, and
//!    `helm_table.dat` fixtures.
//! 4. `proptest` round-trip invariants for arbitrary `f64` and `i64` tensors.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use flexmesa_data::converters::{
    parse_helm_table_text, parse_isotopes_text, parse_lodders03_text,
};
use flexmesa_data::parse_manifest_text;
use flexmesa_data::safetensors::{
    decode_safetensors_bytes, encode_safetensors_bytes, load_safetensors_file,
    save_safetensors_file, SafeTensorBundle, TensorData,
};
use flexmesa_data::sha256::compute_sha256_bytes;
use proptest::prelude::*;

const BUNDLED_MANIFEST: &str =
    include_str!("../../../src/flexmesa/data/manifest.toml");

const SAMPLE_LODDERS03: &str = "\
! Table 6 of Lodders (2003), ApJ, 591: 1220
! Z El     A     %            N(Si= 1e6)
!=======================================
  1 H      1     99.99806     2.431e10
  1 H      2      0.00194     4.716e5
  2 He     4     99.983403    2.343e9
 26 Fe    56     91.754       768,900
";

const SAMPLE_ISOTOPES: &str = "\n\
neut        1.0086649    0    1   0.5   8.071317100
 1.00000E+00  1.00000E+00  1.00000E+00  1.00000E+00  1.00000E+00  1.00000E+00  1.00000E+00  1.00000E+00
 1.00000E+00  1.00000E+00  1.00000E+00  1.00000E+00  1.00000E+00  1.00000E+00  1.00000E+00  1.00000E+00
 1.00000E+00  1.00000E+00  1.00000E+00  1.00000E+00  1.00000E+00  1.00000E+00  1.00000E+00  1.00000E+00
al26-1     25.9868919   13   13   5.0 -12.210151000
 1.00000E+00  1.00000E+00  1.00000E+00  1.00000E+00  1.00000E+00  1.00000E+00  1.00000E+00  1.00000E+00
 1.00000E+00  1.00000E+00  1.00000E+00  1.01000E+00  1.02000E+00  1.04000E+00  1.06000E+00  1.08000E+00
 1.11000E+00  1.14000E+00  1.17000E+00  1.21000E+00  1.25000E+00  1.30000E+00  1.36000E+00  1.43000E+00
";

/// Build a synthetic `helm_table.dat` text fixture of size `(imax, jmax)`.
#[allow(clippy::cast_precision_loss)]
fn build_synthetic_helm_text(imax: usize, jmax: usize) -> String {
    let mut out = String::new();
    for j in 0..jmax {
        for i in 0..imax {
            let base = (100 * j + 10 * i) as f64;
            for k in 0..9 {
                let _ = write!(out, "  {:.16E}", base + f64::from(k) + 1.0);
            }
            out.push('\n');
        }
    }
    for sec in 1..4 {
        for j in 0..jmax {
            for i in 0..imax {
                let base = (1000 * sec + 100 * j + 10 * i) as f64;
                for k in 0..4 {
                    let _ = write!(out, "  {:.16E}", base + f64::from(k) + 1.0);
                }
                out.push('\n');
            }
        }
    }
    out
}

/// Verify FIPS 180-4 `SHA-256` vectors and bundled `manifest.toml` parsing.
#[test]
fn test_sha256_vectors_and_manifest_parsing() {
    assert_eq!(
        compute_sha256_bytes(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        compute_sha256_bytes(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        compute_sha256_bytes(
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
        ),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );

    let tables = parse_manifest_text(BUNDLED_MANIFEST).unwrap();
    assert!(tables.contains_key("lodders03"));
    assert!(tables.contains_key("isotopes"));
    assert!(tables.contains_key("helm_table"));
    assert_eq!(tables["lodders03"].filename, "lodders03.safetensors");
    assert_eq!(tables["isotopes"].tensors.len(), 7);
}

/// Verify `.safetensors` file round-trip, `SHA-256` checking, and `NaN` rejection.
#[test]
fn test_safetensors_file_roundtrip_and_integrity() {
    let dir = tempfile::tempdir().unwrap();
    let st_path = dir.path().join("sample.safetensors");

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "grid".to_owned(),
        TensorData::Float64 {
            shape: vec![2, 2],
            values: vec![1.25, -2.5, 3.75, 4.0],
        },
    );
    tensors.insert(
        "indices".to_owned(),
        TensorData::Int64 {
            shape: vec![3],
            values: vec![10, 20, 30],
        },
    );
    let mut metadata = BTreeMap::new();
    metadata.insert("source".to_owned(), "rust_test".to_owned());
    let bundle = SafeTensorBundle { tensors, metadata };

    let digest = save_safetensors_file(&st_path, &bundle, true).unwrap();
    let loaded = load_safetensors_file(&st_path, Some(&digest), true).unwrap();
    assert_eq!(loaded, bundle);

    assert!(
        load_safetensors_file(&st_path, Some(&"0".repeat(64)), true).is_err()
    );

    let mut bad_nan = BTreeMap::new();
    bad_nan.insert(
        "bad".to_owned(),
        TensorData::Float64 {
            shape: vec![2],
            values: vec![1.0, f64::NAN],
        },
    );
    let nan_bundle = SafeTensorBundle {
        tensors: bad_nan,
        metadata: BTreeMap::new(),
    };
    assert!(encode_safetensors_bytes(&nan_bundle, true).is_err());
}

/// Verify MESA `lodders03.data`, `isotopes.data`, and `helm_table.dat` conversion.
#[test]
fn test_converters_parse_lodders03_isotopes_and_helm_table() {
    let lod = parse_lodders03_text(SAMPLE_LODDERS03).unwrap();
    assert_eq!(
        lod.tensors["atomic_number_z"].as_i64_slice().unwrap(),
        &[1, 1, 2, 26]
    );
    let nsi = lod.tensors["number_abundance_si"].as_f64_slice().unwrap();
    assert!((nsi[3] - 768_900.0).abs() < 1e-9);
    assert_eq!(lod.metadata["isotopes"], "h1,h2,he4,fe56");

    let iso = parse_isotopes_text(SAMPLE_ISOTOPES).unwrap();
    assert_eq!(
        iso.tensors["isomeric_state"].as_i64_slice().unwrap(),
        &[0, 1]
    );
    assert_eq!(iso.tensors["partition_function"].shape(), &[2, 24]);
    let pf = iso.tensors["partition_function"].as_f64_slice().unwrap();
    assert!((pf[47] - 1.43).abs() < 1e-12);

    let helm_text = build_synthetic_helm_text(3, 2);
    let helm = parse_helm_table_text(&helm_text, 3, 2, -12.0, 15.0, 3.0, 13.0)
        .unwrap();
    assert_eq!(helm.tensors["f"].shape(), &[3, 2]);
    assert_eq!(helm.tensors["f_jet"].shape(), &[3, 2, 9]);
    let f_vals = helm.tensors["f"].as_f64_slice().unwrap();
    // Row-major index for [i_rho = 2, j_temp = 1] with shape (3, 2) is 2 * 2 + 1 = 5.
    assert!((f_vals[5] - 121.0).abs() < 1e-12);
}

proptest! {
    /// Property test: finite `f64` and `i64` tensors round-trip bit-exactly
    /// through `encode_safetensors_bytes` and `decode_safetensors_bytes`.
    #[test]
    fn prop_safetensors_f64_and_i64_roundtrip(
        f_vals in prop::collection::vec(-1.0e50_f64..1.0e50_f64, 1..32),
        i_vals in prop::collection::vec(-1_000_000_i64..1_000_000_i64, 1..32),
    ) {
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "floats".to_owned(),
            TensorData::Float64 {
                shape: vec![f_vals.len()],
                values: f_vals,
            },
        );
        tensors.insert(
            "ints".to_owned(),
            TensorData::Int64 {
                shape: vec![i_vals.len()],
                values: i_vals,
            },
        );
        let bundle = SafeTensorBundle {
            tensors,
            metadata: BTreeMap::new(),
        };
        let encoded = encode_safetensors_bytes(&bundle, true).unwrap();
        let decoded = decode_safetensors_bytes(&encoded, true).unwrap();
        prop_assert_eq!(decoded, bundle);
    }
}
