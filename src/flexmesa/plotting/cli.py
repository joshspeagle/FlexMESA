# SPDX-License-Identifier: LGPL-3.0-or-later
"""Command-line interface for ``flexmesa plot`` diagnostic figures."""

from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile

import matplotlib.pyplot as plt

from flexmesa import _dispatch
from flexmesa.plotting import compare
from flexmesa.testing import case_manifest, harness_runner


def _build_plot_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flexmesa plot",
        description=(
            "Render publication-styled comparison and radial profile figures "
            "from stored FlexMESA and MESA 26.4.1 outputs."
        ),
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_cmp = sub.add_parser(
        "compare",
        help="Compare a FlexMESA run against a stored MESA 26.4.1 history.",
    )
    p_cmp.add_argument(
        "case",
        help="Manifest case name or '<suite>/<case>' identifier.",
    )
    p_cmp.add_argument(
        "--suite",
        default=None,
        choices=("star", "binary", "astero"),
        help="Optional suite disambiguation.",
    )
    p_cmp.add_argument(
        "--flexmesa-history",
        default=None,
        help="Direct path to FlexMESA history.data or binary_history.data.",
    )
    p_cmp.add_argument(
        "--flexmesa-dir",
        default=None,
        help=(
            "Output root directory containing FlexMESA LOGS/ or "
            "binary_history.data."
        ),
    )
    p_cmp.add_argument(
        "--mesa-history",
        default=None,
        help="Optional explicit path overriding the stored MESA reference.",
    )
    p_cmp.add_argument(
        "--mesa-dir",
        default=None,
        help="Optional MESA test-suite root when auto-running trace.",
    )
    p_cmp.add_argument(
        "--manifest",
        default=None,
        help="Optional override path to case_manifest.toml.",
    )
    p_cmp.add_argument(
        "--out",
        default=None,
        help="Output figure path (defaults to '<case>_compare.png').",
    )
    p_cmp.add_argument(
        "--profile",
        default=None,
        help="Optional FlexMESA profileN.data to plot alongside history.",
    )
    p_cmp.add_argument(
        "--mesa-profile",
        default=None,
        help="Optional reference MESA profileN.data to overlay on --profile.",
    )
    p_cmp.add_argument(
        "--profile-out",
        default=None,
        help=(
            "Output path when --profile is set "
            "(defaults to '<case>_profile.png')."
        ),
    )
    p_cmp.add_argument(
        "--numerics-mode",
        choices=_dispatch.VALID_NUMERICS_MODES,
        default=None,
        help="Arithmetic mode when running tracer automatically.",
    )
    p_cmp.add_argument(
        "--allow-unsupported-controls",
        action="store_true",
        help="Bypass strict #147 rejection when running tracer automatically.",
    )
    p_cmp.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional per-stage step cap when running tracer automatically.",
    )
    p_cmp.add_argument(
        "--timeout-sec",
        type=float,
        default=300.0,
        help="Wall-clock timeout in seconds when auto-running (default 300).",
    )
    p_cmp.add_argument(
        "--max-rss-gb",
        type=float,
        default=5.5,
        help="Resident-memory ceiling in GB when auto-running (default 5.5).",
    )

    p_prof = sub.add_parser(
        "profile",
        help="Plot four-panel radial structure from stored profileN.data.",
    )
    p_prof.add_argument("profile_path", help="Path to FlexMESA profileN.data.")
    p_prof.add_argument(
        "--mesa-profile",
        default=None,
        help="Optional reference MESA profileN.data to overlay.",
    )
    p_prof.add_argument(
        "--out",
        default="profile_panels.png",
        help="Output figure path (default 'profile_panels.png').",
    )
    return parser


def _render_profile(
    profile_path: pathlib.Path | str,
    ref_profile_path: pathlib.Path | str | None,
    out_path: pathlib.Path | str,
) -> None:
    fig, _ = compare.plot_profile_panels(
        profile_path,
        ref_profile_path=ref_profile_path,
        save_path=out_path,
    )
    plt.close(fig)
    print(f"Wrote profile figure: {out_path}")


def _render_comparison(
    entry: case_manifest.CaseManifestEntry,
    ns: argparse.Namespace,
    out_path: pathlib.Path,
    *,
    flex_hist: pathlib.Path | str | None,
    flex_dir: pathlib.Path | str | None,
    term_code: str,
    eff_mode: str,
    ctrl_ignored: bool,
) -> None:
    fig, _ = compare.plot_case_comparison(
        entry.case,
        suite=entry.suite,
        flexmesa_history=flex_hist,
        flexmesa_dir=flex_dir,
        mesa_history=ns.mesa_history,
        manifest_path=ns.manifest,
        save_path=out_path,
        termination_code=term_code,
        numerics_mode=eff_mode,
        controls_ignored=ctrl_ignored,
        max_steps=ns.max_steps,
        case_entry=entry,
    )
    plt.close(fig)
    print(f"Wrote comparison figure: {out_path}")


def _handle_compare(ns: argparse.Namespace) -> int:
    entry = compare.resolve_case_entry(
        ns.case, suite=ns.suite, manifest_path=ns.manifest
    )
    if ns.mesa_history is None and not entry.reference_path:
        print(
            f"error: {entry.suite}/{entry.case} has reference_status="
            f"{entry.reference_status!r} and no stored reference_path "
            "(pass --mesa-history to override)",
            file=sys.stderr,
        )
        return 2

    out_path = pathlib.Path(ns.out or f"{entry.case}_compare.png")
    if ns.flexmesa_history is None and ns.flexmesa_dir is None:
        eff_mode = harness_runner.resolve_effective_numerics_mode(
            ns.numerics_mode
        )
        case_dir = harness_runner.resolve_case_dir(
            entry.suite, entry.case, mesa_dir=ns.mesa_dir
        )
        with tempfile.TemporaryDirectory(prefix="flexmesa_plot_") as tmp_dir:
            outcome = harness_runner.run_case_with_diagnostic(
                entry.suite,
                case_dir,
                pathlib.Path(tmp_dir),
                max_steps=ns.max_steps,
                numerics_mode=eff_mode,
                allow_unsupported_controls=bool(ns.allow_unsupported_controls),
                timeout_sec=ns.timeout_sec,
                max_rss_gb=ns.max_rss_gb,
            )
            if outcome.history_path is None:
                print(
                    f"error: no history written for {entry.suite}/{entry.case} "
                    f"({outcome.error_message or outcome.status})",
                    file=sys.stderr,
                )
                return 1
            _render_comparison(
                entry,
                ns,
                out_path,
                flex_hist=outcome.history_path,
                flex_dir=None,
                term_code=outcome.termination_code,
                eff_mode=eff_mode,
                ctrl_ignored=outcome.controls_ignored,
            )
    else:
        _render_comparison(
            entry,
            ns,
            out_path,
            flex_hist=ns.flexmesa_history,
            flex_dir=ns.flexmesa_dir,
            term_code="",
            eff_mode="",
            ctrl_ignored=False,
        )

    if ns.profile is not None:
        _render_profile(
            ns.profile,
            ns.mesa_profile,
            pathlib.Path(ns.profile_out or f"{entry.case}_profile.png"),
        )
    return 0


def _handle_profile(ns: argparse.Namespace) -> int:
    _render_profile(ns.profile_path, ns.mesa_profile, ns.out)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Execute ``flexmesa plot compare|profile`` CLI commands.

    Args:
        argv: Arguments following ``flexmesa plot`` (default ``sys.argv[1:]``).

    Returns:
        Exit status code (``0`` on success, non-zero on missing inputs/error).
    """
    cli_args = list(sys.argv[1:] if argv is None else argv)
    ns = _build_plot_parser().parse_args(cli_args)
    if ns.subcommand == "compare":
        return _handle_compare(ns)
    return _handle_profile(ns)


if __name__ == "__main__":
    raise SystemExit(main())
