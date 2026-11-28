#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate the GYRE and ADIPLS per-call measurement table (Stream G, G1).

Replays every GYRE and ADIPLS call MESA 26.4.1 made in the five class-I
cases that reach them, through the FlexMESA ports, and counts per case and
per quantity the records that match MESA at 0 ULP:

* GYRE (``astero/fast_from_file``, ``star/gyre_in_mesa_wd``): the
  ``pulse_tap`` captures of the ``mesa-histories`` bundle, replayed by
  ``crates/flexmesa-gyre/examples/g1_measure_pulse_taps.rs``.
* ADIPLS (``astero/example_astero``, ``astero/fast_scan_grid``,
  ``astero/fast_simplex``): the A0 glue-tap captures under
  ``FLEXMESA_ADIPLS_CAPTURES`` (``<case>/a0tap``), replayed by
  ``crates/flexmesa-adipls/examples/g1_measure_captures.rs``. Without those
  captures the ADIPLS rows read "captures pending"; the table regenerates
  with no code change once they exist.

Outputs ``notes/audit/gyre_adipls_measurement.json`` and ``.md``.

Usage:
    python scripts/audit/gyre_adipls_measurement.py         # run and write
    python scripts/audit/gyre_adipls_measurement.py --check # md == render(json)

The default mode builds and runs two cargo examples (``--profile ci``); run it
through the heavy slot. It sets ``FLEXMESA_REQUIRE_ZERO_SKIPS=1``, so a
missing GYRE capture fails instead of being skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_JSON = REPO_ROOT / "notes" / "audit" / "gyre_adipls_measurement.json"
OUT_MD = REPO_ROOT / "notes" / "audit" / "gyre_adipls_measurement.md"
GENERATOR = "scripts/audit/gyre_adipls_measurement.py"

GYRE_CASES = ("astero/fast_from_file", "star/gyre_in_mesa_wd")
ADIPLS_CASES = ("example_astero", "fast_scan_grid", "fast_simplex")
CASE_ORDER = (
    "astero/fast_from_file",
    "star/gyre_in_mesa_wd",
    "astero/example_astero",
    "astero/fast_scan_grid",
    "astero/fast_simplex",
)


def _histories_dir() -> pathlib.Path:
    """Resolve the bundle like ``flexmesa data`` (without importing the
    package, which needs the compiled extension)."""
    path = REPO_ROOT / "src" / "flexmesa" / "data" / "fetch.py"
    spec = importlib.util.spec_from_file_location("_g1_fetch", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return pathlib.Path(mod.resolve_mesa_histories_dir())


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_example(
    package: str, example: str, env: dict[str, str]
) -> list[dict[str, Any]]:
    cmd = [
        "cargo",
        "run",
        "--profile",
        "ci",
        "-j",
        "4",
        "-q",
        "-p",
        package,
        "--example",
        example,
    ]
    out = subprocess.run(
        cmd, cwd=REPO_ROOT, env=env, check=True, capture_output=True, text=True
    ).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def _provenance(histories: pathlib.Path) -> dict[str, Any]:
    prov = json.loads((histories / "PROVENANCE.json").read_text())
    captures: dict[str, str] = {}
    for case in GYRE_CASES:
        rel = f"runs/{case}/pulse_tap/modes.tsv"
        captures[rel] = _sha256(histories / rel)
    adipls_root = os.environ.get("FLEXMESA_ADIPLS_CAPTURES", "").strip()
    adipls: dict[str, str] = {}
    if adipls_root:
        for case in ADIPLS_CASES:
            for name in ("calls.tsv", "modes.tsv"):
                p = pathlib.Path(adipls_root) / case / "a0tap" / name
                if p.is_file():
                    adipls[f"{case}/a0tap/{name}"] = _sha256(p)
    status = _git("status", "--porcelain", "--", "crates")
    return {
        "generator": GENERATOR,
        "mesa_release": prov["mesa"]["release"],
        "mesa_git_commit": prov["mesa"]["git_commit"],
        "mesasdk_version": prov["sdk_and_compiler"]["mesasdk_version"],
        "gyre_capture_bundle": prov["bundle_name"],
        "gyre_capture_bundle_date": prov["date"],
        "gyre_capture_sha256": captures,
        "adipls_capture_bundle": (
            "A0 glue-tap captures (adipls_oracles/run_mesa_captures.sh)"
            if adipls
            else None
        ),
        "adipls_capture_sha256": adipls,
        "flexmesa_commit": _git("rev-parse", "HEAD"),
        "flexmesa_crates_dirty": bool(status),
        "require_zero_skips": True,
    }


def _fmt_int(v: int) -> str:
    return f"{v:,}"


def render_md(doc: dict[str, Any]) -> str:
    p = doc["provenance"]
    lines = [
        "<!-- Generated by scripts/audit/gyre_adipls_measurement.py; "
        "do not edit. -->",
        "",
        "# GYRE and ADIPLS: every captured call, bit for bit (G1)",
        "",
        f"- **Generator:** `{p['generator']}`, with "
        "`FLEXMESA_REQUIRE_ZERO_SKIPS=1`.",
        f"- **MESA:** {p['mesa_release']} (`{p['mesa_git_commit'][:12]}`), "
        f"SDK `{p['mesasdk_version']}`.",
        "- **GYRE captures:** `pulse_tap` of bundle "
        f"`{p['gyre_capture_bundle']}`"
        f" ({p['gyre_capture_bundle_date'][:10]}); `modes.tsv` sha256 "
        + ", ".join(
            f"`{k.split('/')[2]}` `{v[:12]}`"
            for k, v in sorted(p["gyre_capture_sha256"].items())
        )
        + ".",
        "- **ADIPLS captures:** "
        + (
            f"{p['adipls_capture_bundle']}; sha256 "
            + ", ".join(
                f"`{k}` `{v[:12]}`"
                for k, v in sorted(p["adipls_capture_sha256"].items())
            )
            if p["adipls_capture_bundle"]
            else "not present (FLEXMESA_ADIPLS_CAPTURES unset); "
            "ADIPLS rows are not measured"
        )
        + ".",
        f"- **FlexMESA:** `{p['flexmesa_commit'][:12]}`"
        + (
            " with uncommitted crate changes"
            if p["flexmesa_crates_dirty"]
            else ""
        )
        + ".",
        "",
        "A call is one `get_modes(l)` (GYRE) or one per-model `adipls` call "
        "(ADIPLS). A call is fully bit-identical when it returns the same "
        "number of modes as MESA and every compared quantity of every mode "
        "has MESA's 64 bits. ULP distances are over binary64 bit patterns; "
        "for integer quantities the column gives the absolute difference.",
        "",
        "## Per case",
        "",
        "| Case | Code | Calls | Bit-identical calls | Mode-count mismatches | "
        "Errors | Modes (MESA / FlexMESA) | First differing record |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |",
    ]
    for r in doc["cases"]:
        if r["status"] != "measured":
            lines.append(
                f"| `{r['case']}` | {r['code']} | not measured | | | | | "
                f"{r['status']}: {r.get('reason', '')} |"
            )
            continue
        first = r["first_differing_call"] or "none"
        lines.append(
            f"| `{r['case']}` | {r['code']} | {_fmt_int(r['calls'])} | "
            f"{_fmt_int(r['calls_identical'])} | "
            f"{_fmt_int(r['calls_mode_count_mismatch'])} | "
            f"{_fmt_int(r['calls_error'])} | "
            f"{_fmt_int(r['modes_mesa'])} / {_fmt_int(r['modes_flexmesa'])} | "
            f"{first} |"
        )
    lines += [
        "",
        "## Per quantity",
        "",
        "Records compared are the modes present in both MESA and FlexMESA "
        "for a call (times the number of words for `obs_st` and `csummm`).",
        "",
        "| Case | Quantity | Records compared | Match at 0 ULP | Differ | "
        "Max ULP | Median ULP (differing) | First difference |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for r in doc["cases"]:
        if r["status"] != "measured":
            continue
        for q in r["quantities"]:
            lines.append(
                f"| `{r['case']}` | `{q['name']}` | "
                f"{_fmt_int(q['compared'])} | "
                f"{_fmt_int(q['zero_ulp'])} | {_fmt_int(q['differ'])} | "
                f"{_fmt_int(q['max_ulp'])} | "
                f"{_fmt_int(q['median_ulp_of_differing'])} | "
                f"{q['first_difference'] or ''} |"
            )
    lines.append("")
    return "\n".join(lines)


def _order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: CASE_ORDER.index(r["case"]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--check",
        action="store_true",
        help="confirm the committed .md is the rendering of the committed "
        ".json (no replay)",
    )
    args = ap.parse_args(argv)

    if args.check:
        doc = json.loads(OUT_JSON.read_text())
        if OUT_MD.read_text() != render_md(doc):
            print(
                f"{OUT_MD.relative_to(REPO_ROOT)} is stale; rerun {GENERATOR}"
            )
            return 1
        print("gyre_adipls_measurement: up to date")
        return 0

    histories = _histories_dir()
    env = dict(os.environ)
    env["FLEXMESA_REQUIRE_ZERO_SKIPS"] = "1"
    env["FLEXMESA_HISTORIES_DIR"] = str(histories)
    rows = _run_example("flexmesa-gyre", "g1_measure_pulse_taps", env)
    rows += _run_example("flexmesa-adipls", "g1_measure_captures", env)
    doc = {"provenance": _provenance(histories), "cases": _order(rows)}
    OUT_JSON.write_text(json.dumps(doc, indent=2) + "\n")
    OUT_MD.write_text(render_md(doc))
    print(f"wrote {OUT_JSON.relative_to(REPO_ROOT)} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
