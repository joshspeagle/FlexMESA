// SPDX-License-Identifier: LGPL-3.0-or-later
//! Every pinned raw 4-D EOS slice fits without a non-finite value (`#164`).
//!
//! Upstream `Check_XEoS_Interpolation_Data`
//! (`eos/private/eosdt_load_tables.f90:408-424`) replaces non-finite spline
//! entries with zero after the fit. `FlexMESA` raises an error on them instead
//! (`DECISIONS #33`), which reproduces upstream only while no pinned slice
//! produces one. This test is that check: it fits all slices in the
//! `eos_4d_raw` bundle from the raw files, with no runtime cache involved, and
//! fails if any fit reports an error.

use std::path::PathBuf;

use flexmesa_eos::{CmsXTable, EosDtXzTable};
use flexmesa_utils::resolve_flexmesa_data_dir;

/// Number of raw slices in the `eos_4d_raw` bundle built from MESA `26.4.1`
/// (16 `eosDT`, 89 `FreeEOS`, 11 `CMS`).
const NUM_PINNED_RAW_SLICES: usize = 116;

#[test]
fn test_all_pinned_raw_eos_slices_fit_without_non_finite_entries() {
    let raw_dir = resolve_flexmesa_data_dir().join("eos_4d_raw");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&raw_dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", raw_dir.display()))
        .map(|entry| entry.expect("directory entry").path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "raw"))
        .collect();
    paths.sort();
    assert_eq!(
        paths.len(),
        NUM_PINNED_RAW_SLICES,
        "unexpected number of raw slices in {}",
        raw_dir.display()
    );

    let mut failures = Vec::new();
    for path in &paths {
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .expect("utf-8 file name");
        let outcome = if name.starts_with("mesa-CMS_") {
            CmsXTable::from_raw_bin(path).map(|_| ())
        } else {
            EosDtXzTable::from_raw_bin(path).map(|_| ())
        };
        if let Err(e) = outcome {
            failures.push(format!("{name}: {e}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} raw slices failed to fit:\n{}",
        failures.len(),
        paths.len(),
        failures.join("\n")
    );
}
