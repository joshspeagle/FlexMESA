# SPDX-License-Identifier: LGPL-3.0-or-later
"""Top-level ``flexmesa`` command-line interface."""

from __future__ import annotations

import argparse
import sys

from flexmesa.data import fetch
from flexmesa.testing import dashboard, trace_cli


def main(argv: list[str] | None = None) -> int:
    """Dispatch top-level ``flexmesa`` CLI subcommands.

    Args:
        argv: Optional CLI argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit status (``0`` on success).
    """
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] == "data":
        return fetch.main(args_list[1:])
    if args_list and args_list[0] == "trace":
        return trace_cli.trace_main(args_list[1:])
    if args_list and args_list[0] == "harness":
        return trace_cli.harness_main(args_list[1:])
    if args_list and args_list[0] == "dashboard":
        return dashboard.dashboard_main(args_list[1:])

    parser = argparse.ArgumentParser(
        prog="flexmesa",
        description="FlexMESA command-line utilities.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the installed FlexMESA version and exit.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser(
        "data",
        help="Download, verify, or inspect FlexMESA physics & test bundles.",
    )
    sub.add_parser(
        "trace",
        help="Run a star, binary, or astero case and stream MESA history logs.",
    )
    sub.add_parser(
        "harness",
        help="Generate the 106-case reference table or run trajectory gates.",
    )
    sub.add_parser(
        "dashboard",
        help="Generate or validate the MESA 26.4.1 progress dashboard.",
    )
    parsed = parser.parse_args(args_list)
    if parsed.version:
        import flexmesa

        ref = fetch.load_upstream_reference()
        print(f"{flexmesa.__version__} (upstream: {ref.summary_line()})")
        return 0
    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
