// SPDX-License-Identifier: LGPL-3.0-or-later
//! Issue `#375` tests for Wave 1 `.net` search precedence, `include` chains,
//! loud failures, and exact species/reaction graphs for `minimal.net`,
//! `nco.net`, and `wd_aic.net`.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_net::{net_get, NetError, NetGeneralInfo, ParsedNetDef};
use flexmesa_rates::ScreeningMode;

struct TempNetDir {
    root: PathBuf,
}

impl TempNetDir {
    fn new(tag: &str) -> Self {
        static SEQ: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let id = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "flexmesa_net_375_{tag}_{}_{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create temp net dir");
        Self { root }
    }
}

impl Drop for TempNetDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn optional_mesa_star_suite_dir() -> Option<PathBuf> {
    let base = std::env::var("MESA_DIR").map_or_else(
        |_| {
            let home = std::env::var("HOME").unwrap_or_default();
            PathBuf::from(home)
                .join(".cache")
                .join("flexmesa")
                .join("mesa_test_inputs")
        },
        PathBuf::from,
    );
    let suite = base.join("star").join("test_suite");
    suite.is_dir().then_some(suite)
}

fn iso_names(g: &NetGeneralInfo) -> Vec<String> {
    let nuclides = NuclideCatalog::builtin_ref();
    g.chem_id
        .iter()
        .map(|&cid| {
            nuclides
                .get_by_id(cid)
                .map_or_else(|| panic!("unknown chem_id {cid}"), |n| n.name.clone())
        })
        .collect()
}

fn assert_net_evaluates_cleanly(g: &NetGeneralInfo, label: &str) {
    let nuclides = NuclideCatalog::builtin_ref();
    let mut x = vec![0.0_f64; g.num_isos];
    let n_isos = u32::try_from(g.num_isos).expect("num_isos fits u32");
    let fill = 1.0 / f64::from(n_isos);
    x.fill(fill);
    let comp = get_composition_info(nuclides, &g.chem_id, &x, false)
        .unwrap_or_else(|e| panic!("{label} composition info: {e}"));
    let log_t = 8.3_f64;
    let temp = flexmesa_math::exp10(log_t);
    let log_rho = 4.0_f64;
    let rho = flexmesa_math::exp10(log_rho);
    let rf = vec![1.0_f64; g.num_reactions];
    let out = net_get(
        g,
        false,
        &x,
        temp,
        log_t,
        rho,
        log_rho,
        comp.abar,
        comp.zbar,
        comp.z2bar,
        comp.ye,
        0.0,
        0.0,
        0.0,
        &rf,
        1.0,
        ScreeningMode::Extended,
    )
    .unwrap_or_else(|e| panic!("{label} net_get: {e}"));
    assert!(out.eps_nuc.is_finite(), "{label} eps_nuc must be finite");
    assert_eq!(out.dxdt.len(), g.num_isos);
    assert!(
        out.dxdt.iter().all(|v| v.is_finite()),
        "{label} dxdt must be finite"
    );
}

#[test]
fn test_upstream_net_search_precedence_and_include_resolution() {
    let dir = TempNetDir::new("precedence");
    let nets_dir = dir.root.join("nets");
    fs::create_dir_all(&nets_dir).expect("create nets subdir");

    // 1. `<dir>/nets/<file>` overrides built-in `basic.net` when `<dir>/<file>` is absent.
    fs::write(
        nets_dir.join("basic.net"),
        "add_isos(\n   h1\n   he4\n   c12\n)\n",
    )
    .expect("write nets/basic.net");
    let from_subdir =
        ParsedNetDef::parse_from_name_or_dir("basic.net", Some(&dir.root))
            .expect("resolve <dir>/nets/basic.net");
    assert_eq!(from_subdir.active_isos.len(), 3);

    // 2. `<dir>/<file>` takes precedence over `<dir>/nets/<file>` and built-in.
    fs::write(dir.root.join("basic.net"), "add_isos(\n   h1\n   he4\n)\n")
        .expect("write dir/basic.net");
    let from_root =
        ParsedNetDef::parse_from_name_or_dir(" basic.net ", Some(&dir.root))
            .expect("resolve <dir>/basic.net");
    assert_eq!(from_root.active_isos.len(), 2);

    // 3. `include` follows the same 3-stage search order across local root,
    //    `<dir>/nets/`, and the embedded `data/net_data/nets` bundle.
    fs::write(nets_dir.join("local_sub.inc"), "add_isos(n14)\n")
        .expect("write local_sub.inc");
    fs::write(
        dir.root.join("combo.net"),
        "include 'basic.net'\ninclude 'local_sub.inc'\ninclude 'add_co_burn'\n",
    )
    .expect("write combo.net");
    let combo =
        NetGeneralInfo::from_name_or_dir("combo.net", Some(&dir.root))
            .expect("resolve mixed local + subdir + builtin includes");
    assert!(!combo.doing_approx21);
    assert!(iso_names(&combo).contains(&String::from("h1")));
    assert!(iso_names(&combo).contains(&String::from("he4")));
    assert!(iso_names(&combo).contains(&String::from("n14")));
    assert!(
        combo.reaction_index_by_name("r_c12_c12_to_ne20_he4").is_some()
            || combo.num_reactions > 0,
        "builtin include 'add_co_burn' should add co-burn reactions"
    );
}

#[test]
fn test_missing_cyclic_and_corrupt_local_nets_fail_loudly_without_fallback() {
    let dir = TempNetDir::new("loud_fail");

    // Missing network file without and with search_dir -> UnknownNetFile.
    let err_none = ParsedNetDef::parse_from_name("nonexistent_w1.net")
        .expect_err("missing net without search_dir must fail");
    assert!(matches!(err_none, NetError::UnknownNetFile(_)));

    let err_dir = ParsedNetDef::parse_from_name_or_dir(
        "nonexistent_w1.net",
        Some(&dir.root),
    )
    .expect_err("missing net with search_dir must fail");
    assert!(matches!(err_dir, NetError::UnknownNetFile(_)));

    let err_blank = ParsedNetDef::parse_from_name_or_dir("   ", Some(&dir.root))
        .expect_err("blank net name must fail");
    assert!(matches!(err_blank, NetError::UnknownNetFile(_)));

    // Missing include target must fail loudly.
    fs::write(
        dir.root.join("bad_inc.net"),
        "include 'basic.net'\ninclude 'does_not_exist.inc'\n",
    )
    .expect("write bad_inc.net");
    let err_inc =
        ParsedNetDef::parse_from_name_or_dir("bad_inc.net", Some(&dir.root))
            .expect_err("missing include target must fail");
    assert!(matches!(err_inc, NetError::UnknownNetFile(_)));

    // Cyclic include must fail cleanly without blowing the stack.
    fs::write(dir.root.join("cycle_a.net"), "include 'cycle_b.net'\n")
        .expect("write cycle_a.net");
    fs::write(dir.root.join("cycle_b.net"), "include 'cycle_a.net'\n")
        .expect("write cycle_b.net");
    let err_cycle =
        ParsedNetDef::parse_from_name_or_dir("cycle_a.net", Some(&dir.root))
            .expect_err("cyclic include must fail");
    assert!(matches!(err_cycle, NetError::ParseError(_)));

    // Invalid UTF-8 in `<dir>/basic.net` must fail loudly instead of falling
    // back to built-in `basic.net`.
    fs::write(dir.root.join("basic.net"), [0xff_u8, 0xfe_u8, 0x00_u8])
        .expect("write invalid UTF-8 basic.net");
    let err_corrupt =
        ParsedNetDef::parse_from_name_or_dir("basic.net", Some(&dir.root))
            .expect_err("corrupt local net must not fall back to builtin");
    assert!(matches!(err_corrupt, NetError::ParseError(_)));

    // Bogus weak transition absent from both REACLIB and weaklib must fail loudly.
    fs::write(
        dir.root.join("bogus_wk.net"),
        "add_isos(c12 o16)\nadd_reactions(r_c12_wk_o16)\n",
    )
    .expect("write bogus_wk.net");
    let err_bogus =
        ParsedNetDef::parse_from_name_or_dir("bogus_wk.net", Some(&dir.root))
            .expect_err("invalid weak reaction must fail");
    assert!(matches!(err_bogus, NetError::InvalidReactionOrIso(_)));
}

fn verify_fixture_net(
    tag: &str,
    net_name: &str,
    contents: &str,
    expected_isos: &[&str],
    expected_reactions: usize,
    required_reactions: &[&str],
) -> NetGeneralInfo {
    assert!(matches!(
        ParsedNetDef::parse_from_name(net_name),
        Err(NetError::UnknownNetFile(_))
    ));
    let dir = TempNetDir::new(tag);
    fs::write(dir.root.join(net_name), contents).expect("write fixture net");
    let net = NetGeneralInfo::from_name_or_dir(net_name, Some(&dir.root))
        .unwrap_or_else(|e| panic!("load {net_name}: {e}"));
    assert!(!net.doing_approx21);
    assert_eq!(net.num_isos, expected_isos.len());
    assert_eq!(iso_names(&net), expected_isos);
    assert_eq!(net.num_reactions, expected_reactions);
    for &rxn in required_reactions {
        assert!(
            net.reaction_index_by_name(rxn).is_some(),
            "{net_name} missing expected reaction {rxn}"
        );
    }
    assert_net_evaluates_cleanly(&net, net_name);
    if let Some(suite_dir) = optional_mesa_star_suite_dir() {
        let case_path: &Path = &suite_dir.join(tag);
        if case_path.join(net_name).is_file() {
            let on_disk = NetGeneralInfo::from_name_or_dir(
                net_name,
                Some(case_path),
            )
            .unwrap_or_else(|e| panic!("on-disk {tag}/{net_name}: {e}"));
            assert_eq!(on_disk.chem_id, net.chem_id);
            assert_eq!(on_disk.num_reactions, net.num_reactions);
            assert_eq!(on_disk.net_reaction_by_name, net.net_reaction_by_name);
        }
    }
    net
}

#[test]
fn test_wave1_case_nets_minimal_nco_and_wd_aic_match_mesa_26_4_1() {
    let co_burn = NetGeneralInfo::from_name("co_burn.net")
        .expect("builtin co_burn.net");
    assert_eq!(co_burn.num_isos, 9);

    verify_fixture_net(
        "carbon_kh",
        "minimal.net",
        "add_isos(\n   h1\n   he4\n   c12\n   o16\n)\n",
        &["h1", "he4", "c12", "o16"],
        0,
        &[],
    );

    verify_fixture_net(
        "custom_rates",
        "nco.net",
        "      \n         include 'basic.net'\n         include 'add_co_burn'\n      \n      add_isos_and_reactions(\n\tc14\n\to18\n\t)",
        &[
            "h1", "he3", "he4", "c12", "c14", "n14", "o16", "o18", "ne20",
            "mg24", "si28",
        ],
        co_burn.num_reactions + 4,
        &[
            "r_n14_wk_c14",
            "r_c14_wk-minus_n14",
            "r_c14_ag_o18",
            "r_o18_ga_c14",
        ],
    );

    verify_fixture_net(
        "wd_aic",
        "wd_aic.net",
        "      include 'co_burn.net'\n\n      add_isos(\n         o20 \n         f20\n         ne24\n         na24 \n         )\n         \n      add_reactions(\n          r_ne20_wk_f20\n          r_f20_wk-minus_ne20,\n          r_f20_wk_o20,\n          r_o20_wk-minus_f20,\n          r_mg24_wk_na24\n          r_na24_wk-minus_mg24\n          r_na24_wk_ne24,\n          r_ne24_wk-minus_na24\n         ) ",
        &[
            "h1", "he3", "he4", "c12", "n14", "o16", "o20", "f20", "ne20",
            "ne24", "na24", "mg24", "si28",
        ],
        co_burn.num_reactions + 8,
        &[
            "r_ne20_wk_f20",
            "r_f20_wk-minus_ne20",
            "r_f20_wk_o20",
            "r_o20_wk-minus_f20",
            "r_mg24_wk_na24",
            "r_na24_wk-minus_mg24",
            "r_na24_wk_ne24",
            "r_ne24_wk-minus_na24",
        ],
    );
}
