# SPDX-License-Identifier: GPL-3.0-or-later
"""Progress dashboard generator, freshness gate, and CLI (`#174`).

Orchestrates collection across ``benchmarks/dashboard/port_manifest.toml``,
repository source counts, ``benchmarks/history_harness/``, control defaults,
``cargo llvm-cov`` and ``pytest --cov`` summaries, ``#[ignore]``/``xfail``
annotations, GitHub issue/PR snapshots, and merge history, and renders the
marked blocks in ``README.md``, ``docs/benchmarks/progress-dashboard.md``,
``docs/index.md``, and ``benchmarks/dashboard/dashboard.json``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any

from flexmesa.data import fetch
from flexmesa.testing import (
    _dashboard_collect,
    _dashboard_coverage,
    _dashboard_github,
    _dashboard_render,
    case_manifest,
    port_manifest,
    test_taxonomy,
)

HISTORY_HEADER = (
    "date_utc\tcommit\tevent\tengine_ported_pct\tall_verified_pct\t"
    "cases_passing\trust_cov_pct\tpython_cov_pct\tclosed_140\ttotal_140\t"
    "deferred_169"
)


def load_history_tsv(history_path: pathlib.Path) -> list[dict[str, str]]:
    """Read merge-by-merge headline rows from ``history.tsv``.

    Args:
        history_path: Path to ``benchmarks/dashboard/history.tsv``.

    Returns:
        Ordered list of row dictionaries keyed by ``HISTORY_HEADER`` columns.
    """
    if not history_path.is_file():
        return []
    lines = [
        ln
        for ln in history_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    if not lines:
        return []
    cols = lines[0].split("\t")
    return [
        dict(zip(cols, vals, strict=True))
        for raw in lines[1:]
        if len(vals := raw.split("\t")) == len(cols)
    ]


def build_dashboard_data(
    repo_root: pathlib.Path,
    mesa_dir: pathlib.Path | None = None,
    manifest: port_manifest.PortManifest | None = None,
) -> dict[str, Any]:
    """Assemble the complete dashboard data dictionary from repository files.

    Args:
        repo_root: Root directory of the ``FlexMESA`` repository checkout.
        mesa_dir: Optional MESA 26.4.1 ``mesa_test_inputs`` directory.
        manifest: Optional pre-loaded ``PortManifest`` instance.

    Returns:
        JSON-serializable dictionary backing all Markdown and JSON outputs.
    """
    dash_dir = repo_root / "benchmarks" / "dashboard"
    loaded_manifest = (
        manifest
        if manifest is not None
        else port_manifest.load_port_manifest(dash_dir / "port_manifest.toml")
    )
    cov_data: dict[str, Any] = json.loads(
        (dash_dir / "coverage_summary.json").read_text(encoding="utf-8")
    )
    gh_data: dict[str, Any] = json.loads(
        (dash_dir / "github_snapshot.json").read_text(encoding="utf-8")
    )
    return {
        "schema_version": 1,
        "upstream_reference": {
            "version": loaded_manifest.upstream_version,
            "commit": loaded_manifest.upstream_commit,
            "archive_md5": loaded_manifest.archive_md5,
        },
        "module_coverage": _dashboard_collect.collect_module_coverage(
            loaded_manifest, repo_root
        ),
        "numerics": _dashboard_collect.collect_numerics_and_module_verification(
            repo_root
        ),
        "harness": _dashboard_collect.collect_harness_summary(repo_root),
        "controls": _dashboard_collect.collect_controls_summary(
            repo_root, mesa_dir
        ),
        "coverage": cov_data,
        "skips": _dashboard_collect.collect_ignored_and_skipped_tests(
            repo_root
        ),
        "test_taxonomy": _dashboard_collect.collect_test_taxonomy(
            repo_root, loaded_manifest
        ),
        "github": gh_data,
        "history": load_history_tsv(dash_dir / "history.tsv"),
    }


def render_expected_files(
    repo_root: pathlib.Path,
    mesa_dir: pathlib.Path | None = None,
    data: dict[str, Any] | None = None,
) -> dict[pathlib.Path, str]:
    """Compute the expected UTF-8 contents of all four dashboard target files.

    Args:
        repo_root: Repository checkout root directory.
        mesa_dir: Optional MESA 26.4.1 ``mesa_test_inputs`` directory.
        data: Optional pre-computed ``build_dashboard_data`` dictionary.

    Returns:
        Mapping from target ``pathlib.Path`` to expected file contents.
    """
    dash_data = (
        data if data is not None else build_dashboard_data(repo_root, mesa_dir)
    )
    dash_json_path = repo_root / "benchmarks" / "dashboard" / "dashboard.json"
    readme_path = repo_root / "README.md"
    docs_page_path = repo_root / "docs" / "benchmarks" / "progress-dashboard.md"
    docs_index_path = repo_root / "docs" / "index.md"

    readme_updated = _dashboard_render.replace_marked_block(
        readme_path.read_text(encoding="utf-8"),
        _dashboard_render.render_readme_block(dash_data),
        _dashboard_render.BEGIN_MARKER,
        _dashboard_render.END_MARKER,
    )
    docs_index_updated = _dashboard_render.replace_marked_block(
        docs_index_path.read_text(encoding="utf-8"),
        _dashboard_render.render_docs_index_summary(dash_data),
        _dashboard_render.BEGIN_SUMMARY_MARKER,
        _dashboard_render.END_SUMMARY_MARKER,
    )
    return {
        dash_json_path: json.dumps(dash_data, indent=2) + "\n",
        readme_path: readme_updated,
        docs_page_path: _dashboard_render.render_docs_page(dash_data),
        docs_index_path: docs_index_updated,
    }


def _load_git_base_manifest(
    repo_root: pathlib.Path, base_ref: str
) -> tuple[port_manifest.PortManifest | None, frozenset[str]]:
    """Load ``port_manifest.toml`` and changed file list vs ``base_ref``."""
    mb_proc = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", base_ref, "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    merge_base = mb_proc.stdout.strip()
    diff_proc = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--name-only", merge_base],
        check=True,
        capture_output=True,
        text=True,
    )
    changed = frozenset(
        ln.strip() for ln in diff_proc.stdout.splitlines() if ln.strip()
    )
    show_proc = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "show",
            f"{merge_base}:benchmarks/dashboard/port_manifest.toml",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if show_proc.returncode != 0:
        return None, changed
    base_m = port_manifest.parse_port_manifest_text(
        show_proc.stdout, f"{base_ref}:benchmarks/dashboard/port_manifest.toml"
    )
    return base_m, changed


def verify_all_dashboard_invariants(
    repo_root: pathlib.Path,
    mesa_dir: pathlib.Path | None = None,
    base_ref: str | None = None,
    check_outputs: bool = True,
) -> list[str]:
    """Run manifest, skip-link, and byte-exact freshness checks.

    Args:
        repo_root: Repository checkout root directory.
        mesa_dir: Optional MESA 26.4.1 ``mesa_test_inputs`` directory.
        base_ref: Optional git base ref (e.g. ``origin/main``) for PR status
            promotion checks.
        check_outputs: When ``False``, the dashboard is still rendered, so
            a rendering failure is reported, but the rendered text is not
            compared with the committed files. Pull requests use this:
            ``main`` regenerates the files after each merge
            (``DECISIONS #47``).

    Returns:
        List of validation error messages (empty when all checks pass).
    """
    dash_dir = repo_root / "benchmarks" / "dashboard"
    manifest = port_manifest.load_port_manifest(dash_dir / "port_manifest.toml")
    base_m: port_manifest.PortManifest | None = None
    changed: frozenset[str] | None = None
    if base_ref:
        base_m, changed = _load_git_base_manifest(repo_root, base_ref)

    errors = port_manifest.verify_port_manifest(
        manifest=manifest,
        repo_root=repo_root,
        mesa_dir=mesa_dir,
        baseline_harness_json=(
            repo_root
            / "benchmarks"
            / "history_harness"
            / "baseline_run_26_4_1.json"
        ),
        base_manifest=base_m,
        changed_paths=changed,
    )

    data = build_dashboard_data(repo_root, mesa_dir, manifest=manifest)
    expected = render_expected_files(repo_root, mesa_dir, data=data)
    for path, exp_text in expected.items() if check_outputs else ():
        rel = path.relative_to(repo_root).as_posix()
        if not path.is_file():
            errors.append(f"{rel}: missing generated dashboard file")
        elif path.read_text(encoding="utf-8") != exp_text:
            errors.append(
                f"{rel}: committed content differs from "
                "`flexmesa dashboard render`"
            )

    for loc in data["skips"]["unlinked_locations"]:
        errors.append(f"{loc}: #[ignore]/xfail/skip lacks an #<issue> link")
    errors.extend(
        test_taxonomy.verify_test_taxonomy(
            repo_root,
            taxonomy_path=dash_dir / "test_taxonomy.toml",
            manifest_path=dash_dir / "port_manifest.toml",
        )
    )
    return errors


def record_history_entry(
    repo_root: pathlib.Path,
    mesa_dir: pathlib.Path | None,
    date_utc: str,
    commit_short: str,
    event_label: str,
) -> None:
    """Append or update one merge headline line in ``history.tsv``.

    Args:
        repo_root: Repository checkout root directory.
        mesa_dir: Optional MESA 26.4.1 ``mesa_test_inputs`` directory.
        date_utc: ISO date ``YYYY-MM-DD`` in UTC.
        commit_short: Short git commit SHA.
        event_label: Short merge event label (e.g. ``PR #200 (#174)``).
    """
    data = build_dashboard_data(repo_root, mesa_dir)
    head = _dashboard_render.summarize_headline(data)
    cov = data["coverage"]
    new_row = {
        "date_utc": date_utc,
        "commit": commit_short,
        "event": event_label,
        "engine_ported_pct": f"{head['engine_tot']['pct_ported']:.1f}",
        "all_verified_pct": f"{head['all_tot']['pct_verified']:.1f}",
        "cases_passing": str(head["hh_tot"]["passing"]),
        "rust_cov_pct": f"{cov['rust']['overall']['percent']:.1f}",
        "python_cov_pct": f"{cov['python']['overall']['percent']:.1f}",
        "closed_140": str(head["closed_140"]),
        "total_140": str(head["total_140"]),
        "deferred_169": str(head["deferred_169_count"]),
    }

    hist_path = repo_root / "benchmarks" / "dashboard" / "history.tsv"
    existing = [
        r
        for r in load_history_tsv(hist_path)
        if r.get("commit") != commit_short
    ]
    existing.append(new_row)
    cols = HISTORY_HEADER.split("\t")
    out_lines = [
        "# SPDX-License-Identifier: GPL-3.0-or-later",
        "# Headline progress history across merges to main (#174).",
        HISTORY_HEADER,
        *("\t".join(r[c] for c in cols) for r in existing),
    ]
    hist_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _resolve_mesa_dir(
    override: str | None, allow_missing: bool = False
) -> pathlib.Path | None:
    if override:
        resolved = pathlib.Path(override).resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"--mesa-dir does not exist or is not a directory: {resolved}"
            )
        return resolved
    try:
        return fetch.resolve_mesa_test_inputs_dir()
    except FileNotFoundError:
        if allow_missing:
            return None
        raise


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flexmesa dashboard",
        description="Generate and validate the progress dashboard (#174).",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)
    for cmd_name, help_txt in (
        ("render", "Write updated dashboard blocks to README, docs, and JSON."),
        ("check", "Fail if port_manifest or committed dashboard is stale."),
        ("verify-manifest", "Validate port_manifest.toml schema and evidence."),
    ):
        p = sub.add_parser(cmd_name, help=help_txt)
        p.add_argument("--repo-root", type=pathlib.Path, default=None)
        p.add_argument("--mesa-dir", type=str, default=None)
        p.add_argument("--allow-missing-mesa-dir", action="store_true")
        if cmd_name in {"check", "verify-manifest"}:
            p.add_argument("--base-ref", type=str, default=None)
        if cmd_name == "check":
            p.add_argument(
                "--no-outputs",
                action="store_true",
                help=(
                    "Render the dashboard but do not compare it with the "
                    "committed README, docs page and JSON (pull requests; "
                    "main regenerates them after each merge)."
                ),
            )

    p_gh = sub.add_parser(
        "sync-github", help="Refresh github_snapshot.json from the GitHub API."
    )
    p_gh.add_argument("--repo-root", type=pathlib.Path, default=None)
    p_gh.add_argument("--repo", type=str, default="joshspeagle/FlexMESA")

    p_cov = sub.add_parser(
        "collect-coverage",
        help="Combine llvm-cov and pytest --cov JSON into summary.",
    )
    p_cov.add_argument("--repo-root", type=pathlib.Path, default=None)
    p_cov.add_argument("--rust-json", type=pathlib.Path, required=True)
    p_cov.add_argument("--python-json", type=pathlib.Path, required=True)

    p_hist = sub.add_parser(
        "record-history",
        help="Append or update a merge headline entry in history.tsv.",
    )
    p_hist.add_argument("--repo-root", type=pathlib.Path, default=None)
    p_hist.add_argument("--mesa-dir", type=str, default=None)
    p_hist.add_argument("--allow-missing-mesa-dir", action="store_true")
    p_hist.add_argument("--date-utc", type=str, required=True)
    p_hist.add_argument("--commit", type=str, required=True)
    p_hist.add_argument("--event", type=str, required=True)
    return parser


def dashboard_main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``flexmesa dashboard`` subcommands.

    Args:
        argv: Optional list of CLI arguments following ``dashboard``.

    Returns:
        Process exit code (``0`` on success, ``1`` on check failure).
    """
    ns = _build_cli_parser().parse_args(argv)
    repo_root = (
        ns.repo_root.resolve()
        if getattr(ns, "repo_root", None) is not None
        else case_manifest.repo_root()
    )

    if ns.subcommand == "sync-github":
        snap = repo_root / "benchmarks" / "dashboard" / "github_snapshot.json"
        _dashboard_github.sync_github_snapshot(snap, repo=str(ns.repo))
        print(f"Updated {snap.relative_to(repo_root)}")
        return 0

    if ns.subcommand == "collect-coverage":
        out = repo_root / "benchmarks" / "dashboard" / "coverage_summary.json"
        _dashboard_coverage.write_coverage_summary(
            ns.rust_json.resolve(), ns.python_json.resolve(), out
        )
        print(f"Updated {out.relative_to(repo_root)}")
        return 0

    try:
        mesa_dir = _resolve_mesa_dir(
            getattr(ns, "mesa_dir", None),
            allow_missing=bool(getattr(ns, "allow_missing_mesa_dir", False)),
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if ns.subcommand == "record-history":
        record_history_entry(
            repo_root, mesa_dir, ns.date_utc, ns.commit, ns.event
        )
        print("Updated benchmarks/dashboard/history.tsv")
        return 0

    if ns.subcommand == "render":
        for path, text in render_expected_files(repo_root, mesa_dir).items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"Wrote {path.relative_to(repo_root)}")
        return 0

    if ns.subcommand == "verify-manifest":
        manifest = port_manifest.load_port_manifest(
            repo_root / "benchmarks" / "dashboard" / "port_manifest.toml"
        )
        base_m, changed = (
            _load_git_base_manifest(repo_root, ns.base_ref)
            if ns.base_ref
            else (None, None)
        )
        errs = port_manifest.verify_port_manifest(
            manifest=manifest,
            repo_root=repo_root,
            mesa_dir=mesa_dir,
            baseline_harness_json=(
                repo_root
                / "benchmarks"
                / "history_harness"
                / "baseline_run_26_4_1.json"
            ),
            base_manifest=base_m,
            changed_paths=changed,
        )
        if errs:
            for e in errs:
                print(f"ERROR: {e}", file=sys.stderr)
            return 1
        print(
            f"Verified {len(manifest.entries)} port manifest entries cleanly."
        )
        return 0

    if ns.subcommand == "check":
        errs = verify_all_dashboard_invariants(
            repo_root=repo_root,
            mesa_dir=mesa_dir,
            base_ref=ns.base_ref,
            check_outputs=not ns.no_outputs,
        )
        if errs:
            for e in errs:
                print(f"ERROR: {e}", file=sys.stderr)
            return 1
        if ns.no_outputs:
            print("Port manifest is valid and the dashboard renders.")
            return 0
        print("Dashboard and port manifest are up to date.")
        return 0

    return 2
