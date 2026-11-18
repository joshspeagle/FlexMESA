// SPDX-License-Identifier: LGPL-3.0-or-later
//! Workspace governance tests for `#165` Items 4 and 12.
//!
//! Verifies that:
//! 1. `clippy.toml` configures `disallowed-methods` for every `f64` method
//!    mandated by `#165` Item 12 (`exp`, `exp2`, `exp_m1`, `ln`, `ln_1p`,
//!    `log`, `log2`, `log10`, `powf`, `powi`, `sin`, `cos`, `sin_cos`, `tan`,
//!    `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`,
//!    `acosh`, `atanh`, `cbrt`, `hypot`, and `mul_add`).
//! 2. No `allow(...disallowed_methods...)` suppression exists anywhere under
//!    `crates/` outside `flexmesa-math` and `flexmesa-crlibm-sys`.
//! 3. `flexmesa-crlibm-sys` is the sole crate in `crates/` permitted to use
//!    `unsafe`, enforces `unsafe_op_in_unsafe_fn = "deny"`, and no other crate
//!    relaxes `unsafe_code = "forbid"`.

use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root two levels above flexmesa-math")
        .to_path_buf()
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

#[test]
fn test_clippy_toml_disallows_all_required_f64_methods() {
    let clippy_toml = workspace_root().join("clippy.toml");
    let text = fs::read_to_string(&clippy_toml)
        .expect("workspace clippy.toml must exist");

    let required = [
        "f64::exp",
        "f64::exp2",
        "f64::exp_m1",
        "f64::ln",
        "f64::ln_1p",
        "f64::log",
        "f64::log2",
        "f64::log10",
        "f64::powf",
        "f64::powi",
        "f64::sin",
        "f64::cos",
        "f64::sin_cos",
        "f64::tan",
        "f64::asin",
        "f64::acos",
        "f64::atan",
        "f64::atan2",
        "f64::sinh",
        "f64::cosh",
        "f64::tanh",
        "f64::asinh",
        "f64::acosh",
        "f64::atanh",
        "f64::cbrt",
        "f64::hypot",
        "f64::mul_add",
    ];

    for method in required {
        let needle = format!("path = \"{method}\"");
        assert!(
            text.contains(&needle),
            "clippy.toml is missing disallowed-methods entry: {needle}"
        );
    }
}

#[test]
fn test_no_disallowed_methods_suppression_outside_math_crates() {
    let crates_root = workspace_root().join("crates");
    let entries = fs::read_dir(&crates_root).expect("crates directory");

    for entry in entries {
        let pkg_dir = entry.expect("crate entry").path();
        if !pkg_dir.is_dir() {
            continue;
        }
        let crate_name = pkg_dir
            .file_name()
            .and_then(|s| s.to_str())
            .expect("crate dir name");

        if crate_name == "flexmesa-crlibm-sys" {
            continue;
        }
        if crate_name == "flexmesa-math" {
            let mut math_files = Vec::new();
            collect_rs_files(&pkg_dir.join("src"), &mut math_files);
            let allowed_fast = pkg_dir.join("src/fast_ops.rs");
            for file in math_files {
                if file == allowed_fast {
                    continue;
                }
                let src = fs::read_to_string(&file).expect("read math .rs");
                assert!(
                    !src.contains("disallowed_methods"),
                    "{} must not suppress clippy::disallowed_methods (confined to src/fast_ops.rs)",
                    file.display()
                );
            }
            continue;
        }

        let cargo_toml = pkg_dir.join("Cargo.toml");
        if cargo_toml.exists() {
            let manifest =
                fs::read_to_string(&cargo_toml).expect("read crate Cargo.toml");
            assert!(
                !manifest.contains("disallowed_methods")
                    && !manifest.contains("disallowed-methods"),
                "{} must not override clippy::disallowed_methods",
                cargo_toml.display()
            );
        }

        let mut rs_files = Vec::new();
        collect_rs_files(&pkg_dir, &mut rs_files);
        for file in rs_files {
            let src = fs::read_to_string(&file).expect("read .rs source");
            assert!(
                !src.contains("disallowed_methods")
                    && !src.contains("allow(clippy::all")
                    && !src.contains("allow(clippy::style")
                    && !src.contains("allow(warnings"),
                "{} must not suppress clippy::disallowed_methods or group lints (#165 Item 12)",
                file.display()
            );
            // Physics library `src/` files must source mathematical constants from
            // `flexmesa_const` rather than `std::f64::consts` / `core::f64::consts`.
            let is_src_file = file
                .strip_prefix(&pkg_dir)
                .is_ok_and(|rel| rel.starts_with("src"));
            let exempt_consts = matches!(
                crate_name,
                "flexmesa-testing" | "flexmesa-const" | "flexmesa-utils"
            );
            if is_src_file && !exempt_consts {
                for (lineno, line) in src.lines().enumerate() {
                    let trimmed = line.trim_start();
                    if trimmed.starts_with("//") {
                        continue;
                    }
                    assert!(
                        !line.contains("f64::consts::"),
                        "{}:{} must use `flexmesa_const` constants instead of `f64::consts::`",
                        file.display(),
                        lineno + 1
                    );
                }
            }
        }
    }
}

#[test]
fn test_flexmesa_crlibm_sys_is_sole_unsafe_crate_in_workspace() {
    let root = workspace_root();
    let root_manifest = fs::read_to_string(root.join("Cargo.toml"))
        .expect("read workspace Cargo.toml");
    assert!(
        root_manifest.contains("unsafe_code = \"forbid\""),
        "workspace Cargo.toml must set unsafe_code = \"forbid\""
    );

    let crates_root = root.join("crates");
    let mut seen_crlibm = false;
    let mut on_disk_members = Vec::new();

    for entry in fs::read_dir(&crates_root).expect("crates directory") {
        let pkg_dir = entry.expect("crate entry").path();
        if !pkg_dir.is_dir() {
            continue;
        }
        let crate_name = pkg_dir
            .file_name()
            .and_then(|s| s.to_str())
            .expect("crate dir name");
        let manifest_path = pkg_dir.join("Cargo.toml");
        if !manifest_path.exists() {
            continue;
        }
        on_disk_members.push(format!("crates/{crate_name}"));
        let manifest =
            fs::read_to_string(&manifest_path).expect("read crate Cargo.toml");

        if crate_name == "flexmesa-crlibm-sys" {
            seen_crlibm = true;
            assert!(
                manifest.contains("unsafe_op_in_unsafe_fn = \"deny\""),
                "flexmesa-crlibm-sys Cargo.toml must set unsafe_op_in_unsafe_fn = \"deny\""
            );
            continue;
        }

        assert!(
            manifest.contains("[lints]\nworkspace = true")
                || manifest.contains("[lints]\r\nworkspace = true"),
            "{} must inherit workspace lints (`unsafe_code = \"forbid\"`)",
            manifest_path.display()
        );
        assert!(
            !manifest.contains("unsafe_code"),
            "{} must not override unsafe_code",
            manifest_path.display()
        );

        let mut rs_files = Vec::new();
        collect_rs_files(&pkg_dir, &mut rs_files);
        let self_path =
            root.join("crates/flexmesa-math/tests/routing_governance.rs");
        for file in rs_files {
            if file == self_path {
                continue;
            }
            let src = fs::read_to_string(&file).expect("read .rs source");
            assert!(
                !src.contains("allow(unsafe_code)")
                    && !src.contains("warn(unsafe_code)")
                    && !src.contains("deny(unsafe_code)"),
                "{} must not relax `#![forbid(unsafe_code)]` (#165 Item 4)",
                file.display()
            );
        }
    }

    for member in &on_disk_members {
        assert!(
            root_manifest.contains(&format!("\"{member}\"")),
            "workspace Cargo.toml `[workspace] members` is missing `{member}`"
        );
    }
    assert!(seen_crlibm, "flexmesa-crlibm-sys crate must exist");
}

/// A test that switches the process-wide numerics mode must not share a test
/// binary with tests that read it (issue #210): `with_numerics_mode`
/// serializes switchers against each other, but ordinary code reads the mode
/// without the lock, so a concurrent test can see the mode change halfway
/// through a calculation.
///
/// Outside `flexmesa-math` and the `flexmesa-py` bindings, the mode may
/// therefore be switched only in an integration-test file named
/// `*numerics_mode_test.rs`, which Cargo builds as its own process. A call
/// that names `NumericsMode::Strict` literally is allowed anywhere, since it
/// selects the default.
#[test]
fn test_numerics_mode_is_switched_only_in_dedicated_test_binaries() {
    let crates_root = workspace_root().join("crates");
    let mut files = Vec::new();
    collect_rs_files(&crates_root, &mut files);
    let mut offenders: Vec<String> = Vec::new();
    for path in files {
        let rel = path
            .strip_prefix(&crates_root)
            .expect("under crates/")
            .to_string_lossy()
            .replace('\\', "/");
        if rel.starts_with("flexmesa-math/") || rel.starts_with("flexmesa-py/")
        {
            continue;
        }
        if rel.contains("/tests/") && rel.ends_with("numerics_mode_test.rs") {
            continue;
        }
        let text = fs::read_to_string(&path).expect("read source file");
        for (idx, line) in text.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            let switches = code.contains("with_numerics_mode(")
                || code.contains("set_numerics_mode(");
            if switches && !code.contains("NumericsMode::Strict") {
                offenders.push(format!("{rel}:{}", idx + 1));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "numerics mode switched outside a dedicated `*numerics_mode_test.rs` \
         test binary (issue #210): {offenders:?}"
    );
}
