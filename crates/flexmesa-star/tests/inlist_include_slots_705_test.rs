// SPDX-License-Identifier: GPL-3.0-or-later
//! Deferred bug `#705`: the extra-inlist include loop resets the include
//! slots in place, as MESA's `read_star_job_file`
//! (`star/private/star_job_ctrls_io.f90:597-607`) and `read_controls_file`
//! (`star/private/ctrls_io.f90:700-709`) do. A child inlist that sets slot
//! `j` overrides the parent's slot `j`, and the parent's later slots are
//! already reset when the child returns, so they are never read. The files
//! below are synthetic; the structure mirrors `star/twin_studies`
//! (`inlist_star1_header`: slot 1 names a file that itself reads an extra
//! inlist, slot 2 names `inlist_star1`, which MESA never reads).

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;

use flexmesa_star::inlist::InlistBundle;

struct TempCase {
    root: PathBuf,
}

impl TempCase {
    fn new(tag: &str, files: &[(&str, &str)]) -> Self {
        let root = std::env::temp_dir()
            .join(format!("flexmesa_star_705_{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create temp dir");
        for (name, body) in files {
            fs::write(root.join(name), body).expect("write inlist");
        }
        Self { root }
    }
}

impl Drop for TempCase {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn controls_value(case: &TempCase, entry: &str, key: &str) -> Option<f64> {
    let sections = InlistBundle::load_sections_in_case_dir(&case.root, entry)
        .expect("load sections");
    sections["controls"]
        .try_get_f64("controls", key)
        .expect("typed read")
}

#[test]
fn child_setting_a_slot_overrides_the_parents_slot_and_resets_it() {
    // parent: slot 1 -> child1, slot 2 -> child2. child1 sets slot 2 ->
    // child3. MESA reads parent, child1, child3; the parent's slot 2 was
    // reset by child1's loop, so child2 is never read.
    let case = TempCase::new(
        "override",
        &[
            (
                "parent",
                "&controls\n initial_mass = 1d0\n\
                 read_extra_controls_inlist(1) = .true.\n\
                 extra_controls_inlist_name(1) = 'child1'\n\
                 read_extra_controls_inlist(2) = .true.\n\
                 extra_controls_inlist_name(2) = 'child2'\n/\n",
            ),
            (
                "child1",
                "&controls\n initial_mass = 2d0\n\
                 read_extra_controls_inlist(2) = .true.\n\
                 extra_controls_inlist_name(2) = 'child3'\n/\n",
            ),
            (
                "child2",
                "&controls\n initial_mass = 3d0\n marker_child2 = 1d0\n/\n",
            ),
            ("child3", "&controls\n initial_mass = 4d0\n/\n"),
        ],
    );
    assert_eq!(controls_value(&case, "parent", "initial_mass"), Some(4.0));
    assert_eq!(controls_value(&case, "parent", "marker_child2"), None);
}

#[test]
fn twin_studies_shape_never_reads_the_sibling_inlist() {
    // inlist_star1_header: slot 1 -> inlist_to_end_header, slot 2 ->
    // inlist_star1. inlist_to_end_header sets slot 1 -> inlist_to_end and
    // slot 2 -> inlist_mass, overriding the parent's slot 2: MESA reads
    // inlist_to_end_header, inlist_to_end and inlist_mass, never
    // inlist_star1.
    let case = TempCase::new(
        "twin",
        &[
            (
                "inlist_star1_header",
                "&controls\n read_extra_controls_inlist(1) = .true.\n\
                 extra_controls_inlist_name(1) = 'inlist_to_end_header'\n\
                 read_extra_controls_inlist(2) = .true.\n\
                 extra_controls_inlist_name(2) = 'inlist_star1'\n/\n",
            ),
            (
                "inlist_to_end_header",
                "&controls\n read_extra_controls_inlist(1) = .true.\n\
                 extra_controls_inlist_name(1) = 'inlist_to_end'\n\
                 read_extra_controls_inlist(2) = .true.\n\
                 extra_controls_inlist_name(2) = 'inlist_mass'\n/\n",
            ),
            ("inlist_to_end", "&controls\n initial_mass = 20d0\n/\n"),
            ("inlist_mass", "&controls\n initial_y = 0.28d0\n/\n"),
            ("inlist_star1", "&controls\n initial_mass = 99d0\n/\n"),
        ],
    );
    assert_eq!(
        controls_value(&case, "inlist_star1_header", "initial_mass"),
        Some(20.0)
    );
    assert_eq!(
        controls_value(&case, "inlist_star1_header", "initial_y"),
        Some(0.28)
    );
}

#[test]
fn a_file_named_twice_is_read_twice_as_in_mesa() {
    // parent: slot 1 -> common (x = 1), slot 2 -> mid (x = 2). mid sets slot
    // 1 -> common again, which MESA reads again (x = 1 last).
    let case = TempCase::new(
        "twice",
        &[
            (
                "parent",
                "&controls\n read_extra_controls_inlist(1) = .true.\n\
                 extra_controls_inlist_name(1) = 'common'\n\
                 read_extra_controls_inlist(2) = .true.\n\
                 extra_controls_inlist_name(2) = 'mid'\n/\n",
            ),
            ("common", "&controls\n marker_x = 1d0\n/\n"),
            (
                "mid",
                "&controls\n marker_x = 2d0\n\
                 read_extra_controls_inlist(1) = .true.\n\
                 extra_controls_inlist_name(1) = 'common'\n/\n",
            ),
        ],
    );
    assert_eq!(controls_value(&case, "parent", "marker_x"), Some(1.0));
}

#[test]
fn a_chain_of_independent_slots_is_read_in_slot_order() {
    // Plain case: slots 1..3 name three files that include nothing.
    let case = TempCase::new(
        "plain",
        &[
            (
                "parent",
                "&controls\n read_extra_controls_inlist(1) = .true.\n\
                 extra_controls_inlist_name(1) = 'a'\n\
                 read_extra_controls_inlist(2) = .true.\n\
                 extra_controls_inlist_name(2) = 'b'\n\
                 read_extra_controls_inlist(3) = .true.\n\
                 extra_controls_inlist_name(3) = 'c'\n/\n",
            ),
            ("a", "&controls\n marker_x = 1d0\n marker_a = 1d0\n/\n"),
            ("b", "&controls\n marker_x = 2d0\n marker_b = 1d0\n/\n"),
            ("c", "&controls\n marker_x = 3d0\n marker_c = 1d0\n/\n"),
        ],
    );
    assert_eq!(controls_value(&case, "parent", "marker_x"), Some(3.0));
    for k in ["marker_a", "marker_b", "marker_c"] {
        assert_eq!(controls_value(&case, "parent", k), Some(1.0), "{k}");
    }
}

#[test]
fn a_file_that_includes_itself_fails_at_the_nesting_limit() {
    let case = TempCase::new(
        "cycle",
        &[(
            "loop",
            "&controls\n read_extra_controls_inlist(1) = .true.\n\
             extra_controls_inlist_name(1) = 'loop'\n/\n",
        )],
    );
    let err = InlistBundle::load_sections_in_case_dir(&case.root, "loop")
        .expect_err("cycle must fail");
    assert!(err.to_string().contains("too many levels"), "{err}");
}
