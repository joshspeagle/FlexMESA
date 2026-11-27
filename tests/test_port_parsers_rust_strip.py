# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for ``_port_parsers.strip_rust_comments_and_strings``."""

from __future__ import annotations

import pathlib
import textwrap

from flexmesa.testing import _port_parsers, routine_gate

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _extracted_fns(tmp_path: pathlib.Path, src: str) -> set[str]:
    """Write ``src`` to a scratch Rust file and extract ``fn`` items."""
    rel = "sample.rs"
    (tmp_path / rel).write_text(textwrap.dedent(src), encoding="utf-8")
    cache: dict[str, set[str]] = {}
    names = routine_gate._fn_names(tmp_path, rel, cache)
    assert names is not None
    return names


def test_char_and_byte_char_literals_preserve_trailing_fns(
    tmp_path: pathlib.Path,
) -> None:
    """Char/byte-char quote literals and escapes do not open string matches."""
    src = """\
        pub fn before_literals(b: u8, c: char) -> bool {
            let q1 = b'"';
            let q2 = '"';
            let a1 = '\\'';
            let a2 = b'\\'';
            let x1 = '\\x41';
            let x2 = b'\\x22';
            let u1 = '\\u{1F600}';
            let u2 = '\\"';
            let bs = '\\\\';
            b == q1 || c == q2 || a1 == u2 || a2 == x2 || x1 == u1 || bs == 'x'
        }

        pub fn after_literals() {}
    """
    cleaned = _port_parsers.strip_rust_comments_and_strings(
        textwrap.dedent(src)
    )
    assert cleaned.count("\n") == textwrap.dedent(src).count("\n")
    assert _extracted_fns(tmp_path, src) == {
        "before_literals",
        "after_literals",
    }


def test_lifetimes_and_loop_labels_are_not_stripped(
    tmp_path: pathlib.Path,
) -> None:
    """Lifetimes and loop labels must not be misread as char literals."""
    src = """\
        pub struct Holder<'a> {
            text: &'a str,
        }

        impl<'a> Holder<'a> {
            pub fn borrowed<'b>(&self, other: &'b str) -> &'static str {
                let mut i = 0;
                'outer: loop {
                    if i > 0 {
                        break 'outer;
                    }
                    i += 1;
                }
                'a: loop {
                    break 'a;
                }
                let _ = (self.text, other);
                "ok"
            }
        }

        pub fn after_lifetimes() {}
    """
    cleaned = _port_parsers.strip_rust_comments_and_strings(
        textwrap.dedent(src)
    )
    assert "&'a str" in cleaned
    assert "'static" in cleaned
    assert "'outer: loop" in cleaned
    assert "impl<'a>" in cleaned
    assert _extracted_fns(tmp_path, src) == {"borrowed", "after_lifetimes"}


def test_apostrophes_in_strings_raw_strings_and_comments(
    tmp_path: pathlib.Path,
) -> None:
    """Comments and normal/raw strings containing ``'`` strip cleanly."""
    src = """\
        // don't match this single quote or fn fake_comment()
        /* block comment shouldn't match ' either: fn fake_block() */
        pub fn with_quotes() {
            let s1 = "don't open a char literal here; fn fake_str() {}";
            let s2 = b"byte string don't; fn fake_bstr() {}";
            let r1 = r#"raw's 'quoted' "inner"; fn fake_raw() {}"#;
            let r2 = br##"byte raw don't "# end; fn fake_braw() {}"##;
            let s3 = r#"check('"') and "unsafe fn" don't leak"#;
            let _ = (s1, s2, r1, r2, s3);
        }

        pub fn trim_end_matches_regression(rest: &str, src: &str) -> bool {
            let _ = rest.trim_end_matches('"');
            src.contains("unsafe fn") && !src.contains("fn bogus()")
        }

        pub fn after_all_quotes() {}
    """
    assert _extracted_fns(tmp_path, src) == {
        "with_quotes",
        "trim_end_matches_regression",
        "after_all_quotes",
    }


def test_previously_truncated_repo_rust_symbols_visible() -> None:
    """Functions following byte-char/char quotes in crates/ remain visible."""
    cache: dict[str, set[str]] = {}
    expected = {
        "crates/flexmesa-star/src/report_utils.rs": (
            "eval_surf_avg_rotation_info"
        ),
        "crates/flexmesa-star/src/mesa_history.rs": "evaluate_star_history_row",
        "crates/flexmesa-star/src/mesa_profile.rs": "save_profile",
        "crates/flexmesa-star/src/inlist_populate.rs": (
            "populate_controls_output"
        ),
        "crates/flexmesa-astero/src/sample_eval.rs": "scan_degree",
        "crates/flexmesa-binary/src/binary_history.rs": (
            "evaluate_binary_history_row"
        ),
    }
    for rel, sym in expected.items():
        names = routine_gate._fn_names(REPO_ROOT, rel, cache)
        assert names is not None and sym in names, f"{rel} missing {sym}"
