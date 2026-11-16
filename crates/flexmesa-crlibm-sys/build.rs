// SPDX-License-Identifier: LGPL-3.0-or-later
//! Build script compiling vendored `crlibm 1.0beta4` C sources with fixed
//! per-target configuration headers (no autoconf at build time, `#165`).

use std::env;
use std::path::PathBuf;

const C_SOURCES: &[&str] = &[
    "csrc/crlibm_private.c",
    "csrc/triple-double.c",
    "csrc/exp-td.c",
    "csrc/exp-td-standalone.c",
    "csrc/expm1-standalone.c",
    "csrc/expm1.c",
    "csrc/log.c",
    "csrc/log1p.c",
    "csrc/rem_pio2_accurate.c",
    "csrc/trigo_fast.c",
    "csrc/trigo_accurate.c",
    "csrc/trigpi.c",
    "csrc/asincos.c",
    "csrc/atan_fast.c",
    "csrc/atan_accurate.c",
    "csrc/csh_fast.c",
    "csrc/pow.c",
    "csrc/scs_lib/scs_private.c",
    "csrc/scs_lib/double2scs.c",
    "csrc/scs_lib/division_scs.c",
    "csrc/scs_lib/addition_scs.c",
    "csrc/scs_lib/multiplication_scs.c",
    "csrc/scs_lib/scs2double.c",
    "csrc/scs_lib/zero_scs.c",
];

fn main() {
    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"),
    );
    let target_arch =
        env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH");
    let target_endian =
        env::var("CARGO_CFG_TARGET_ENDIAN").expect("CARGO_CFG_TARGET_ENDIAN");
    let target_ptr_width = env::var("CARGO_CFG_TARGET_POINTER_WIDTH")
        .expect("CARGO_CFG_TARGET_POINTER_WIDTH");

    // 32-bit x86 is rejected at compile time in `src/lib.rs` via `compile_error!`;
    // return early here so `rustc` surfaces that explicit `compile_error!`.
    if target_arch == "x86" {
        return;
    }
    assert_eq!(
        target_ptr_width, "64",
        "flexmesa-crlibm-sys requires a 64-bit target architecture (got {target_ptr_width}-bit)"
    );

    let config_subdir = match (target_arch.as_str(), target_endian.as_str()) {
        ("x86_64", "little") => "config/x86_64",
        ("aarch64", "little") => "config/aarch64",
        (_, "little") => "config/generic_le64",
        (_, "big") => "config/generic_be64",
        _ => panic!("unsupported target endianness: {target_endian}"),
    };

    let cfg_dir = manifest_dir.join(config_subdir);
    let scs_inc_dir = cfg_dir.join("scs_inc");
    let csrc_dir = manifest_dir.join("csrc");
    let scs_src_dir = csrc_dir.join("scs_lib");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=VENDORED.toml");
    println!(
        "cargo:rerun-if-changed={}",
        cfg_dir.join("crlibm_config.h").display()
    );

    let mut build = cc::Build::new();
    build
        .opt_level(2)
        .warnings(false)
        .define("HAVE_CONFIG_H", "1")
        .include(&cfg_dir)
        .include(&scs_inc_dir)
        .include(&csrc_dir)
        .include(&scs_src_dir)
        .flag("-std=c99")
        .flag("-ffp-contract=off")
        .flag("-fno-fast-math");

    if target_arch == "x86_64" {
        build.flag("-msse2").flag("-mfpmath=sse");
    }

    for rel in C_SOURCES {
        let p = manifest_dir.join(rel);
        println!("cargo:rerun-if-changed={}", p.display());
        build.file(p);
    }

    // Track headers as well so header edits trigger rebuilds.
    for hdr in [
        "csrc/crlibm.h",
        "csrc/crlibm_private.h",
        "csrc/triple-double.h",
        "csrc/exp-td.h",
        "csrc/expm1.h",
        "csrc/log.h",
        "csrc/log-td.h",
        "csrc/rem_pio2_accurate.h",
        "csrc/trigo_fast.h",
        "csrc/trigo_accurate.h",
        "csrc/trigpi.h",
        "csrc/asincos.h",
        "csrc/atan_fast.h",
        "csrc/atan_accurate.h",
        "csrc/csh_fast.h",
        "csrc/pow.h",
        "csrc/scs_lib/scs.h",
        "csrc/scs_lib/scs_private.h",
    ] {
        println!(
            "cargo:rerun-if-changed={}",
            manifest_dir.join(hdr).display()
        );
    }

    // Statically archive into `libflexmesa_crlibm.a` so a system `libcrlibm` is
    // never linked.
    build.compile("flexmesa_crlibm");
}
