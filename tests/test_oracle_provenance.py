# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixture tests for the per-routine oracle provenance checks (`DECISIONS #80`).

Each test builds a small repository under ``tmp_path`` with one Rust test
file, one reference and (where needed) a bundle manifest or golden manifest,
then checks that ``_oracle_provenance.evaluate`` passes or fails the claim
on exactly the expected check.
"""

from __future__ import annotations

import hashlib
import pathlib

import pytest

from flexmesa.testing import _oracle_provenance as op

pytestmark = pytest.mark.invariant

RID = "turb/private/mlt.f90::calc_MLT"
TARGETS = {RID: {"calc_mlt"}}
TEST_REL = "crates/flexmesa-turb/tests/mlt_oracle_test.rs"
REF_REL = "benchmarks/fortran_mesa/mlt_oracles/mesa_26_4_1_mlt_oracles.tsv"
RECIPE_REL = "benchmarks/fortran_mesa/mlt_oracles/gen_mlt_oracles.f90"
HEADER = "# mesa_version: r26.4.1\n# mesa_git_commit: c8c2026\n"

# A Rust test whose #[test] reaches the target and the reference through a
# helper, with a 0-ULP comparison on that path.
EXACT_RS = """
const DATA: &str = include_str!(
    "../../../benchmarks/fortran_mesa/mlt_oracles/mesa_26_4_1_mlt_oracles.tsv"
);

fn load() -> Vec<u64> {
    DATA.lines().map(|l| u64::from_str_radix(l, 16).unwrap()).collect()
}

#[test]
fn test_calc_mlt_matches_mesa() {
    for want in load() {
        let got = calc_mlt(1.0);
        assert_eq!(got.to_bits(), want);
    }
}
"""


def _write(root: pathlib.Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _repo(
    root: pathlib.Path,
    *,
    code: str = EXACT_RS,
    ref_text: str = HEADER + "3FF0000000000000\n",
    recipe: bool = True,
) -> pathlib.Path:
    _write(root, TEST_REL, code)
    _write(root, REF_REL, ref_text)
    if recipe:
        _write(root, RECIPE_REL, "program gen\nend program gen\n")
    return root


def _entry(**kw: object) -> op.OracleEntry:
    base: dict[str, object] = {
        "reference": REF_REL,
        "precision": "exact",
        "routines": (RID,),
        "recipe": RECIPE_REL,
    }
    base.update(kw)
    return op.OracleEntry(**base)  # type: ignore[arg-type]


def _run(
    root: pathlib.Path, entry: op.OracleEntry, test_rel: str = TEST_REL
) -> op.ClaimResult:
    (res,) = op.evaluate(root, [(test_rel, (entry,))], TARGETS)
    return res


def _checks(res: op.ClaimResult) -> set[str]:
    return {k for k, _ in res.failures}


def test_exact_claim_with_header_recipe_link_scope_and_zero_ulp_counts(
    tmp_path: pathlib.Path,
) -> None:
    """The reference claim passes every check and counts as exact."""
    res = _run(_repo(tmp_path), _entry())
    assert res.failures == ()
    assert op.verified_ids([res]) == ({RID}, set())


def test_missing_or_misplaced_reference_fails_reference(
    tmp_path: pathlib.Path,
) -> None:
    """A missing file and a file outside the reference areas both fail."""
    root = _repo(tmp_path)
    res = _run(root, _entry(reference=REF_REL.replace("mlt_or", "nope_or")))
    assert "reference" in _checks(res)
    _write(root, "notes/mesa_26_4_1_mlt_oracles.tsv", HEADER)
    res = _run(root, _entry(reference="notes/mesa_26_4_1_mlt_oracles.tsv"))
    assert "reference" in _checks(res)


def test_reference_without_version_header_fails_origin(
    tmp_path: pathlib.Path,
) -> None:
    """A headerless table or a self-generated snapshot is not MESA output."""
    res = _run(_repo(tmp_path, ref_text="3FF0000000000000\n"), _entry())
    assert _checks(res) == {"origin"}
    root = _repo(
        tmp_path, ref_text=HEADER + "# self-generated\n3FF0000000000000\n"
    )
    assert _checks(_run(root, _entry())) == {"origin"}


def test_repository_reference_without_existing_recipe_fails_recipe(
    tmp_path: pathlib.Path,
) -> None:
    """An empty or nonexistent recipe fails the recipe check."""
    root = _repo(tmp_path, recipe=False)
    assert _checks(_run(root, _entry(recipe=""))) == {"recipe"}
    assert _checks(_run(root, _entry())) == {"recipe"}


def test_test_that_never_names_the_reference_fails_linkage_and_scope(
    tmp_path: pathlib.Path,
) -> None:
    """Without the path in code the test cannot read the reference."""
    code = (
        "#[test]\nfn test_calc_mlt() {\n"
        "    assert_eq!(calc_mlt(1.0).to_bits(), 0x3FF0000000000000);\n}\n"
    )
    res = _run(_repo(tmp_path, code=code), _entry())
    assert _checks(res) == {"linkage", "scope"}


def test_embedded_hex_and_decimal_values_must_appear_in_the_reference(
    tmp_path: pathlib.Path,
) -> None:
    """Consts link embedded values: hex as text, decimals as equal floats."""
    code = (
        "const WANT: [u64; 1] = [0x3FF0_0000_0000_0000];\n"
        "#[test]\nfn test_calc_mlt() {\n"
        "    let expected = [1.000_000_000_000_000_0e+00];\n"
        "    assert_eq!(calc_mlt(expected[0]).to_bits(), WANT[0]);\n}\n"
    )
    ref = HEADER + "3FF0000000000000 1.0000000000000000D+00\n"
    root = _repo(tmp_path, code=code, ref_text=ref)
    res = _run(root, _entry(consts=("WANT", "expected")))
    assert res.failures == ()
    _write(root, REF_REL, HEADER + "4000000000000000 2.0D+00\n")
    res = _run(root, _entry(consts=("WANT", "expected")))
    assert _checks(res) == {"consts"}
    assert len(res.failures) == 2
    res = _run(root, _entry(consts=("MISSING",)))
    assert "not found" in res.failures[0][1]


def test_consts_against_a_bundle_member_fail_closed(
    tmp_path: pathlib.Path,
) -> None:
    """Embedded values need a committed repository reference."""
    root = _bundle_repo(tmp_path)
    code = (
        "const WANT: [u64; 1] = [0x3FF0000000000000];\n"
        "#[test]\nfn test_calc_mlt() {\n"
        '    let p = stepdump_path("star/a.stepdump");\n'
        "    assert_eq!(calc_mlt(1.0).to_bits(), WANT[0]);\n}\n"
    )
    _write(root, TEST_REL, code)
    entry = _entry(
        reference="mesa_stepdumps:mesa_stepdumps/star/a.stepdump",
        recipe="",
        consts=("WANT",),
    )
    assert _checks(_run(root, entry)) == {"consts"}


def test_target_call_and_reference_read_in_different_tests_fail_scope(
    tmp_path: pathlib.Path,
) -> None:
    """The assertion that uses the data must be on a path that calls it."""
    code = (
        'const DATA: &str = include_str!("mesa_26_4_1_mlt_oracles.tsv");\n'
        "#[test]\nfn test_reads() {\n    assert!(!DATA.is_empty());\n}\n"
        "#[test]\nfn test_calls() {\n"
        "    assert_eq!(calc_mlt(1.0).to_bits(), 0x3FF0000000000000);\n}\n"
    )
    res = _run(_repo(tmp_path, code=code), _entry())
    assert _checks(res) == {"scope"}


def test_unparseable_layout_and_python_tests_fail_scope_closed(
    tmp_path: pathlib.Path,
) -> None:
    """When the layout cannot be decided the routine does not count."""
    broken = EXACT_RS.replace("    }\n}\n", "    }\n", 1)
    res = _run(_repo(tmp_path, code=broken), _entry())
    assert _checks(res) == {"scope"}
    assert "could not be parsed" in res.failures[0][1]
    py_rel = "tests/test_mlt_oracle.py"
    _write(tmp_path, py_rel, "# mesa_26_4_1_mlt_oracles.tsv\n")
    res = _run(tmp_path, _entry(), test_rel=py_rel)
    assert _checks(res) == {"scope"}


def test_upper_case_targets_are_constants_named_not_called(
    tmp_path: pathlib.Path,
) -> None:
    """A routine mapped to a Rust constant is reached by naming it."""
    code = EXACT_RS.replace("calc_mlt(1.0)", "c::PI * 1.0")
    root = _repo(tmp_path, code=code)
    (res,) = op.evaluate(root, [(TEST_REL, (_entry(),))], {RID: {"PI"}})
    assert res.failures == ()


def test_exact_entry_without_zero_ulp_comparison_fails_precision(
    tmp_path: pathlib.Path,
) -> None:
    """A tolerance comparison cannot evidence an exact claim."""
    code = EXACT_RS.replace(
        "assert_eq!(got.to_bits(), want);",
        "assert!((got - f64::from_bits(want)).abs() < 1e-12);",
    )
    root = _repo(tmp_path, code=code)
    assert _checks(_run(root, _entry())) == {"precision"}
    res = _run(root, _entry(precision="print", tolerance="1e-12"))
    assert res.failures == ()
    assert op.verified_ids([res]) == (set(), {RID})


def test_print_entry_needs_a_positive_tolerance(
    tmp_path: pathlib.Path,
) -> None:
    """``print`` without a stated tolerance fails precision."""
    root = _repo(tmp_path)
    for tol in ("", "0", "abc"):
        res = _run(root, _entry(precision="print", tolerance=tol))
        assert _checks(res) == {"precision"}, tol


def _bundle_repo(root: pathlib.Path) -> pathlib.Path:
    _write(
        root,
        op.BUNDLES_REL,
        '[[bundles]]\nname = "mesa_stepdumps"\n'
        'manifest_file = "bundle_hashes/mesa_stepdumps.sha256.tsv"\n'
        '[[bundles]]\nname = "kap_full"\n'
        'manifest_file = "bundle_hashes/kap_full.sha256.tsv"\n',
    )
    _write(
        root,
        "src/flexmesa/data/bundle_hashes/mesa_stepdumps.sha256.tsv",
        "# sha256\tsize_bytes\trelpath\n"
        "00\t1\tmesa_stepdumps/star/a.stepdump\n",
    )
    _write(
        root,
        "src/flexmesa/data/bundle_hashes/kap_full.sha256.tsv",
        "# sha256\tsize_bytes\trelpath\n00\t1\tkap/a.data\n",
    )
    return root


def test_bundle_member_must_be_listed_in_a_mesa_origin_manifest(
    tmp_path: pathlib.Path,
) -> None:
    """Listed members pass; unlisted members and other bundles fail."""
    root = _bundle_repo(tmp_path)
    code = EXACT_RS.replace(
        "DATA.lines()", 'read_stepdump("star/a.stepdump").lines()'
    ).replace(
        "../../../benchmarks/fortran_mesa/mlt_oracles/"
        "mesa_26_4_1_mlt_oracles.tsv",
        "unused",
    )
    _write(root, TEST_REL, code)
    ok = _entry(
        reference="mesa_stepdumps:mesa_stepdumps/star/a.stepdump", recipe=""
    )
    assert _run(root, ok).failures == ()
    gone = _entry(
        reference="mesa_stepdumps:mesa_stepdumps/star/b.stepdump", recipe=""
    )
    assert "reference" in _checks(_run(root, gone))
    other = _entry(reference="kap_full:kap/a.data", recipe="")
    assert "reference" in _checks(_run(root, other))


def test_golden_upstream_copy_needs_a_matching_manifest_entry(
    tmp_path: pathlib.Path,
) -> None:
    """A listed verbatim copy needs no recipe; a changed copy fails."""
    rel = "tests/fixtures/golden/turb/test_output.txt"
    code = EXACT_RS.replace(
        "../../../benchmarks/fortran_mesa/mlt_oracles/"
        "mesa_26_4_1_mlt_oracles.tsv",
        "../../../tests/fixtures/golden/turb/test_output.txt",
    )
    root = _repo(tmp_path, code=code)
    _write(root, rel, "vc at 1d5 Lsun    3.9654203785866620D+06\n")
    digest = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    _write(
        root,
        op.GOLDEN_MANIFEST_REL,
        f'[[files]]\npath = "{rel}"\nupstream = "turb/test/test_output"\n'
        f'sha256 = "{digest}"\n',
    )
    entry = _entry(reference=rel, recipe="")
    assert _run(root, entry).failures == ()
    _write(root, rel, "edited\n")
    assert _checks(_run(root, entry)) == {"origin"}


def test_unlisted_golden_file_is_judged_by_header_and_recipe(
    tmp_path: pathlib.Path,
) -> None:
    """A generated golden file is a repository reference like any other."""
    rel = "tests/fixtures/golden/rsp/lina_26_4_1.tsv"
    code = EXACT_RS.replace(
        "../../../benchmarks/fortran_mesa/mlt_oracles/"
        "mesa_26_4_1_mlt_oracles.tsv",
        "../../../tests/fixtures/golden/rsp/lina_26_4_1.tsv",
    )
    root = _repo(tmp_path, code=code)
    _write(root, rel, "3FF0000000000000\n")
    assert _checks(_run(root, _entry(reference=rel))) == {"origin"}
    _write(root, rel, HEADER + "3FF0000000000000\n")
    assert _run(root, _entry(reference=rel)).failures == ()


def test_summary_lists_routines_failing_only_on_scope_or_precision() -> None:
    """Exact beats print; scope-only failures are reported separately."""
    a, b, c = "m/a.f90::a", "m/b.f90::b", "m/c.f90::c"
    res = [
        op.ClaimResult("t1.rs", "r", a, "exact"),
        op.ClaimResult("t2.rs", "r", a, "print"),
        op.ClaimResult("t1.rs", "r", b, "exact", (("scope", "x"),)),
        op.ClaimResult("t1.rs", "r", c, "exact", (("origin", "x"),)),
    ]
    assert op.verified_ids(res) == ({a}, set())
    s = op.summarize_claims(res)
    assert sorted(s["routines_not_counted"]) == [b, c]
    assert s["not_counted_only_on_scope_or_precision"] == [b]
    assert "| `m/b.f90::b` |" in op.render_report_markdown(res)


def test_structural_errors_name_missing_fields() -> None:
    """Empty reference, unknown precision and empty routines are errors."""
    errs = op.structural_errors(
        "x.rs", op.OracleEntry(reference="", precision="ulp", routines=())
    )
    assert len(errs) == 3
