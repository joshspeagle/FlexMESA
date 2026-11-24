// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration tests verifying MESA 26.4.1 parity for special-case reactions,
//! `wk-h1`/`wk-he4` weak pairs, and `al26-1`/`al26-2` isomer transitions.

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_math::exp10;
use flexmesa_net::{net_get, NetError, NetGeneralInfo, ReaclibKind};
use flexmesa_rates::ScreeningMode;

static TMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

struct TempNetDir {
    path: PathBuf,
}

impl TempNetDir {
    fn new(prefix: &str) -> Self {
        let id = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "flexmesa_{prefix}_{}_{}",
            std::process::id(),
            id
        ));
        fs::create_dir_all(&path).expect("create temp net dir");
        Self { path }
    }

    fn write(&self, name: &str, contents: &str) {
        fs::write(self.path.join(name), contents).expect("write temp net file");
    }

    fn dir(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempNetDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn reaction_names(g: &NetGeneralInfo) -> HashSet<&str> {
    g.reactions.iter().map(|r| r.name.as_str()).collect()
}

#[test]
fn test_add_isos_and_reactions_standard_nets_match_mesa_26_4_1_counts() {
    let h_burn = NetGeneralInfo::from_name("h_burn.net").expect("h_burn.net");
    assert_eq!(h_burn.num_isos, 44);
    assert_eq!(h_burn.num_reactions, 374);
    let hb_names = reaction_names(&h_burn);
    for expected in [
        "r_h1_h1_wk_h2",
        "r_h1_h1_ec_h2",
        "r_h1_he3_wk_he4",
        "r_he3_he3_to_h1_h1_he4",
        "r_h1_h1_he4_to_he3_he3",
        "r_be7_wk_li7",
        "r_he4_he4_he4_to_c12",
        "r_c12_to_he4_he4_he4",
        "r_c12_c12_to_he4_ne20",
        "r_c12_c12_to_h1_na23",
        "r_o16_o16_to_he4_si28",
        "r_c12_o16_to_he4_mg24",
    ] {
        assert!(
            hb_names.contains(expected),
            "h_burn.net missing special reaction {expected}"
        );
    }
    // `r_h1_h1_wk_h2` and `r_h1_h1_ec_h2` have identical stoichiometry (`2 h1 -> h2`)
    // and must remain unpaired `Other` reactions rather than forward/reverse pairs.
    for (idx, rxn) in h_burn.reactions.iter().enumerate() {
        if rxn.name == "r_h1_h1_wk_h2" || rxn.name == "r_h1_h1_ec_h2" {
            assert_eq!(h_burn.reaction_reaclib_kind[idx], ReaclibKind::Other);
            assert_eq!(h_burn.reverse_id_for_kind_ne_other[idx], 0);
        }
    }

    let mesa_75 =
        NetGeneralInfo::from_name("mesa_75.net").expect("mesa_75.net");
    assert_eq!(mesa_75.num_isos, 75);
    assert_eq!(mesa_75.num_reactions, 549);

    let rp_153 = NetGeneralInfo::from_name("rp_153.net").expect("rp_153.net");
    assert_eq!(rp_153.num_isos, 153);
    assert_eq!(rp_153.num_reactions, 1609);
    let rp_names = reaction_names(&rp_153);
    for expected in [
        "r_c9_wk_h1_he4_he4",
        "r_s27_wk_h1_h1_al25",
        "r_ar31_wk_h1_h1_p29",
        "r_he4_ca36_to_h1_h1_ca38",
        "r_h1_h1_ca38_to_he4_ca36",
    ] {
        assert!(
            rp_names.contains(expected),
            "rp_153.net missing special reaction {expected}"
        );
    }
    // MESA 26.4.1 `net_initialize.f90:933-951` uses hyphenated `'wk-h1'`/`'wk-he4'`,
    // which do not match `_wk_h1_` / `_wk_he4_` reaclib handles in MESA either.
    let unexpected_wk_h: Vec<&str> = rp_names
        .iter()
        .copied()
        .filter(|name| {
            (name.contains("_wk_h1_") || name.contains("_wk_he4_"))
                && !matches!(
                    *name,
                    "r_c9_wk_h1_he4_he4"
                        | "r_s27_wk_h1_h1_al25"
                        | "r_ar31_wk_h1_h1_p29"
                        | "r_b8_wk_he4_he4"
                )
        })
        .collect();
    assert!(
        unexpected_wk_h.is_empty(),
        "unexpected non-MESA wk_h1/wk_he4 reactions in rp_153.net: {unexpected_wk_h:?}"
    );
}

#[test]
fn test_all_51_special_case_reaction_entries_activate() {
    let dir = TempNetDir::new("all_special");
    dir.write(
        "all_special.net",
        "add_isos_and_reactions(\n\
         neut h1 h2 h3 he3 he4 li6 li7 be7 be9 b8 b11\n\
         c9 c11 c12 c13 c14 n14 n15 o13 o15 o16\n\
         ne20 na23 mg23 mg24 al25 al27 si27 si28\n\
         p29 p31 s27 s31 ar31 ca36 ca38\n\
         )\n",
    );
    let g =
        NetGeneralInfo::from_name_or_dir("all_special.net", Some(dir.dir()))
            .expect("load all_special.net");
    let names = reaction_names(&g);

    let expected_reactions = [
        "r_h1_h1_wk_h2",
        "r_h1_h1_ec_h2",
        "r_neut_h1_h1_to_h1_h2",
        "r_h1_h2_to_neut_h1_h1",
        "r_h2_h2_to_h1_h3",
        "r_h1_h3_to_h2_h2",
        "r_he3_ec_h3",
        "r_h2_h2_to_neut_he3",
        "r_neut_he3_to_h2_h2",
        "r_h1_he3_wk_he4",
        "r_h2_h2_to_he4",
        "r_he4_to_h2_h2",
        "r_h2_h3_to_neut_he4",
        "r_neut_he4_to_h2_h3",
        "r_h2_he3_to_h1_he4",
        "r_h1_he4_to_h2_he3",
        "r_h3_he3_to_h2_he4",
        "r_h2_he4_to_h3_he3",
        "r_h3_h3_to_neut_neut_he4",
        "r_neut_neut_he4_to_h3_h3",
        "r_he3_he3_to_h1_h1_he4",
        "r_h1_h1_he4_to_he3_he3",
        "r_li6_to_neut_h1_he4",
        "r_neut_h1_he4_to_li6",
        "r_h2_li7_to_neut_he4_he4",
        "r_neut_he4_he4_to_h2_li7",
        "r_h3_li7_to_neut_neut_he4_he4",
        "r_neut_neut_he4_he4_to_h3_li7",
        "r_h1_li7_to_h2_li6",
        "r_h2_li6_to_h1_li7",
        "r_be7_wk_li7",
        "r_h2_be7_to_h1_he4_he4",
        "r_h1_he4_he4_to_h2_be7",
        "r_neut_be7_to_he4_he4",
        "r_he4_he4_to_neut_be7",
        "r_he3_be7_to_h1_h1_he4_he4",
        "r_h1_h1_he4_he4_to_he3_be7",
        "r_neut_be7_to_h2_li6",
        "r_h2_li6_to_neut_be7",
        "r_neut_be9_to_h3_li7",
        "r_h3_li7_to_neut_be9",
        "r_be9_to_neut_he4_he4",
        "r_neut_he4_he4_to_be9",
        "r_h1_be9_to_h2_he4_he4",
        "r_h2_he4_he4_to_h1_be9",
        "r_neut_b8_to_h1_he4_he4",
        "r_h1_he4_he4_to_neut_b8",
        "r_h1_b11_to_he4_he4_he4",
        "r_he4_he4_he4_to_h1_b11",
        "r_neut_b11_to_h3_be9",
        "r_h3_be9_to_neut_b11",
        "r_c9_wk_h1_he4_he4",
        "r_neut_c11_to_he4_he4_he4",
        "r_he4_he4_he4_to_neut_c11",
        "r_he4_he4_he4_to_c12",
        "r_c12_to_he4_he4_he4",
        "r_neut_n14_to_h2_c13",
        "r_h2_c13_to_neut_n14",
        "r_neut_n15_to_h2_c14",
        "r_h2_c14_to_neut_n15",
        "r_he4_o13_to_h1_h1_o15",
        "r_h1_h1_o15_to_he4_o13",
        "r_c12_c12_to_he4_ne20",
        "r_he4_ne20_to_c12_c12",
        "r_c12_c12_to_h1_na23",
        "r_h1_na23_to_c12_c12",
        "r_neut_mg23_to_c12_c12",
        "r_c12_c12_to_neut_mg23",
        "r_c12_o16_to_he4_mg24",
        "r_he4_mg24_to_c12_o16",
        "r_c12_o16_to_h1_al27",
        "r_h1_al27_to_c12_o16",
        "r_neut_si27_to_c12_o16",
        "r_c12_o16_to_neut_si27",
        "r_o16_o16_to_he4_si28",
        "r_he4_si28_to_o16_o16",
        "r_c12_ne20_to_he4_si28",
        "r_he4_si28_to_c12_ne20",
        "r_o16_o16_to_h1_p31",
        "r_h1_p31_to_o16_o16",
        "r_c12_ne20_to_h1_p31",
        "r_h1_p31_to_c12_ne20",
        "r_s27_wk_h1_h1_al25",
        "r_o16_o16_to_neut_s31",
        "r_neut_s31_to_o16_o16",
        "r_c12_ne20_to_neut_s31",
        "r_neut_s31_to_c12_ne20",
        "r_ar31_wk_h1_h1_p29",
        "r_he4_ca36_to_h1_h1_ca38",
        "r_h1_h1_ca38_to_he4_ca36",
        "r_h3_he3_to_neut_h1_he4",
        "r_he3_li7_to_neut_h1_he4_he4",
        "r_h3_be7_to_neut_h1_he4_he4",
    ];
    assert_eq!(expected_reactions.len(), 93);
    for rxn in expected_reactions {
        assert!(names.contains(rxn), "missing special-case reaction {rxn}");
    }
}

#[test]
fn test_al26_isomer_links_and_validation() {
    let dir = TempNetDir::new("al26");
    dir.write(
        "mesa_al26.net",
        "add_isos_and_reactions(neut h1 he4 mg25 al26-1 al26-2)\n",
    );
    let g = NetGeneralInfo::from_name_or_dir("mesa_al26.net", Some(dir.dir()))
        .expect("load mesa_al26.net");
    assert_eq!(g.num_isos, 6);
    assert_eq!(g.num_reactions, 8);
    let ordered: Vec<&str> =
        g.reactions.iter().map(|r| r.name.as_str()).collect();
    assert_eq!(
        ordered,
        vec![
            "r_mg25_pg_al26-1",
            "r_al26-1_gp_mg25",
            "r_mg25_pg_al26-2",
            "r_al26-2_gp_mg25",
            "r_al26-2_to_al26-1",
            "r_h1_wk_neut",
            "r_al26-1_to_al26-2",
            "r_neut_wk-minus_h1",
        ]
    );

    // Split across an include file must pass top-level validation.
    dir.write("part1.net", "add_isos_and_reactions(al26-1)\n");
    dir.write(
        "split.net",
        "include 'part1.net'\nadd_isos_and_reactions(al26-2)\n",
    );
    let g_split =
        NetGeneralInfo::from_name_or_dir("split.net", Some(dir.dir()))
            .expect("split include al26-1/2");
    assert_eq!(g_split.num_isos, 2);
    assert_eq!(g_split.num_reactions, 2);

    // Invalid combinations must fail loudly with ParseError.
    for (fname, body, msg) in [
        (
            "both_1.net",
            "add_isos(al26 al26-1)\n",
            "cannot specify al26 and al26-1 or al26-2",
        ),
        (
            "both_2.net",
            "add_isos(al26 al26-2)\n",
            "cannot specify al26 and al26-1 or al26-2",
        ),
        (
            "only_1.net",
            "add_isos_and_reactions(al26-1)\n",
            "must specify al26-2 if al26-1 is set",
        ),
        (
            "only_2.net",
            "add_isos_and_reactions(al26-2)\n",
            "must specify al26-1 if al26-2 is set",
        ),
    ] {
        dir.write(fname, body);
        let err = NetGeneralInfo::from_name_or_dir(fname, Some(dir.dir()))
            .expect_err("expected al26 validation failure");
        assert!(
            matches!(err, NetError::ParseError(ref s) if s.contains(msg)),
            "{fname}: unexpected error {err:?}"
        );
    }
}

#[test]
fn test_h_burn_pp_chain_produces_solar_energy_generation() {
    let g = NetGeneralInfo::from_name("h_burn.net").expect("h_burn.net");
    let cat = NuclideCatalog::builtin_ref();
    let mut x = vec![0.0; g.num_isos];
    let set_iso = |x: &mut [f64], name: &str, val: f64| {
        let cid = cat.get_nuclide_index(name).unwrap();
        let ni = g.net_iso[cid];
        assert!(ni > 0, "missing {name}");
        x[ni - 1] = val;
    };
    set_iso(&mut x, "h1", 0.70);
    set_iso(&mut x, "he3", 1.0e-5);
    set_iso(&mut x, "he4", 0.28);
    set_iso(&mut x, "c12", 0.004);
    set_iso(&mut x, "n14", 0.006);
    set_iso(&mut x, "o16", 0.01);

    let comp = get_composition_info(cat, &g.chem_id, &x, false).unwrap();
    let log_t = 7.19;
    let log_rho = 2.18;
    let rf = vec![1.0; g.num_reactions];
    let res = net_get(
        &g,
        true,
        &x,
        exp10(log_t),
        log_t,
        exp10(log_rho),
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
    .expect("net_get h_burn");
    assert!(
        res.eps_nuc > 1.0,
        "expected solar-core h_burn eps_nuc > 1 erg/g/s via pp-chain, got {:.3e}",
        res.eps_nuc
    );

    // Pre-MS low-temperature pp start (`T ~ 2.0e6 K`, `rho ~ 0.1 g/cm^3`):
    // without `r_h1_h1_wk_h2`, no CNO or triple-alpha barrier can fire.
    let log_t_low = 6.30;
    let log_rho_low = -1.0;
    let res_low = net_get(
        &g,
        true,
        &x,
        exp10(log_t_low),
        log_t_low,
        exp10(log_rho_low),
        log_rho_low,
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
    .expect("net_get pre-ms h_burn");
    assert!(
        res_low.eps_nuc > 1.0e-8,
        "expected pre-MS pp eps_nuc > 1e-8 erg/g/s, got {:.3e}",
        res_low.eps_nuc
    );
}
