// SPDX-License-Identifier: GPL-3.0-or-later
//! REACLIB handle coverage of the embedded rates bundle (`#632`).
//!
//! `tests/data/reaclib_handles_light_nuclei.txt` lists every REACLIB handle
//! (forward and reverse) that MESA `26.4.1` builds from
//! `data/rates_data/jina_reaclib_results_20171020_default` for reactions among
//! nuclides with `Z <= 10` and `A <= 26`. This is the range the adaptive
//! network of `star/adjust_net` reaches from a pre-main-sequence composition,
//! and it includes every handle that touches `b12` and `f16`. The file is
//! produced by `scripts/data/build_rates_tables.py --list-handles`, never by
//! hand. Every handle must resolve to at least one REACLIB term, so that
//! `add_reaction_by_name` never silently drops a reaction.

#![forbid(unsafe_code)]

use flexmesa_rates::RatesCatalog;

const HANDLES: &str = include_str!("data/reaclib_handles_light_nuclei.txt");

#[test]
fn light_nuclei_reaclib_handles_all_resolve_with_terms() {
    let catalog = RatesCatalog::builtin();
    let missing: Vec<&str> = HANDLES
        .lines()
        .filter(|h| !h.is_empty())
        .filter(|h| {
            catalog
                .reaclib_terms(h)
                .map_or(true, |(_, terms)| terms.is_empty())
        })
        .collect();
    assert!(
        missing.is_empty(),
        "{} REACLIB handles absent from rates_tables.bin, first few: {:?}",
        missing.len(),
        &missing[..missing.len().min(10)]
    );
}

#[test]
fn b12_and_f16_reaction_handles_are_present_with_expected_direction() {
    let catalog = RatesCatalog::builtin();
    // (handle, is_reverse): forward handles are the exothermic direction.
    let cases = [
        ("r_b11_ng_b12", false),
        ("r_b12_gn_b11", true),
        ("r_li8_ag_b12", false),
        ("r_b12_wk-minus_c12", false),
        ("r_o15_pg_f16", true),
        ("r_f16_gp_o15", false),
        ("r_f15_ng_f16", false),
        ("r_f16_ag_na20", false),
    ];
    for (handle, rev) in cases {
        let (is_rev, terms) = catalog
            .reaclib_terms(handle)
            .unwrap_or_else(|e| panic!("{handle}: {e}"));
        assert!(!terms.is_empty(), "{handle} has no terms");
        assert_eq!(is_rev, rev, "{handle} direction");
    }
}
