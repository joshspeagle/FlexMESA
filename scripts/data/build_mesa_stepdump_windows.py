#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build a ``mesa-stepdump-windows-<version>.tar.zst`` bundle (#608).

Packs the mid-run MESA 26.4.1 step-dump windows listed in
``benchmarks/fortran_mesa/stepdumps/windows.toml`` into a deterministic tar
archive (sorted members, fixed mtime, root ownership) compressed with zstd,
with a ``PROVENANCE.json`` and a copy of ``windows.toml`` at its root, and
writes the per-file manifest (``sha256``, size, relative path).

Each window is found by its header, not by its location: every ``*.stepdump``
under the ``--dumps`` directories whose ``case``, ``first_model``,
``steps_per_star``, ``occurrence`` and ``stage`` lines match a manifest entry.
A dump written before the ``occurrence`` and ``stage`` lines existed counts as
occurrence 1 and, for a ``stage: S of N`` line, stage S (stage 1 without one).
A window found in several places is taken from the earliest ``--dumps``
directory. Files named ``*.withxtra.stepdump`` and ``*.allstages.*`` are
never packed.

Each packed file is the output ``run_stepdumps.py`` writes: record types the
committed ``stepdump_patch.py`` does not write (diagnostic records from a
locally patched solver) are dropped, and the ``occurrence`` and ``stage``
header lines are set from the manifest. Nothing else changes.

Usage::

    python scripts/data/build_mesa_stepdump_windows.py \\
        --dumps <dir> [<dir> ...] --out <archive> \\
        [--manifest-out <tsv>] [--version v1] [--allow-missing]

Prints the archive SHA-256 and the sizes for ``bundles.toml``. Without
``--allow-missing`` the build fails when a manifest window has no dump.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import pathlib
import re
import sys
import tarfile
from typing import Any, BinaryIO

REPO = pathlib.Path(__file__).resolve().parents[2]
STEPDUMPS = REPO / "benchmarks" / "fortran_mesa" / "stepdumps"
ROOT = "mesa_stepdump_windows"
CANONICAL_MTIME = 0
FORBIDDEN = (b"/home/", b"/tmp/", b"/mnt/", b"/Users/")
# Record types stepdump_patch.py writes, plus the file-structure lines.
KNOWN_TAGS = frozenset(
    "RUN DONE NAMES EQNAMES CALL STATE DM AUX EQU CORR NEWTON FOLD TRY "
    "RESID".split()
)


def _load_regenerate() -> Any:
    spec = importlib.util.spec_from_file_location(
        "regenerate_windows", STEPDUMPS / "regenerate_windows.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _zstd_writer(path: pathlib.Path) -> BinaryIO:
    from compression import zstd  # Python >= 3.14

    return zstd.open(path, "wb", level=19)  # type: ignore[return-value]


def _add(tf: tarfile.TarFile, rel: str, payload: bytes) -> tuple[str, int]:
    for needle in FORBIDDEN:
        if needle in payload:
            raise ValueError(f"machine path {needle!r} in {rel}")
    info = tarfile.TarInfo(name=rel)
    info.size = len(payload)
    info.mtime = CANONICAL_MTIME
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    tf.addfile(info, io.BytesIO(payload))
    return hashlib.sha256(payload).hexdigest(), len(payload)


def header_of(path: pathlib.Path) -> dict[str, str]:
    """Returns the ``# key: value`` header lines of a dump (first wins)."""
    head: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            key, _, val = line[1:].partition(":")
            head.setdefault(key.strip(), val.strip())
    return head


def window_key(
    head: dict[str, str],
) -> tuple[str, int, int, int, int] | None:
    """Returns ``(case, stage, occurrence, first_model, steps)`` of a dump.

    Args:
        head: Header from :func:`header_of`.

    Returns:
        The key, or ``None`` when the dump has no numeric window (a dump of
        the start of a run).
    """
    try:
        first = int(head["first_model"])
        steps = int(head["steps_per_star"])
        case = head["case"]
    except (KeyError, ValueError):
        return None
    occ = int(head.get("occurrence", "1"))
    m = re.match(r"\d+", head.get("stage", ""))
    return case, int(m.group()) if m else 1, occ, first, steps


def normalized(text: str, stage: int, occurrence: int) -> bytes:
    """Returns a dump as ``run_stepdumps.py`` writes it.

    Args:
        text: Dump contents.
        stage: Manifest stage.
        occurrence: Manifest occurrence.

    Returns:
        The header with ``occurrence`` and ``stage`` set from the manifest
        and only the record types of ``stepdump_patch.py`` in the body.
    """
    head: list[str] = []
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            if re.match(r"#\s*(stage|occurrence)\s*:", line):
                continue
            head.append(line)
        elif line.split(maxsplit=1)[:1] and line.split()[0] in KNOWN_TAGS:
            body.append(line)
    # The two window lines follow ``first_model``, as run_stepdumps.py writes.
    at = next(
        (i + 1 for i, h in enumerate(head) if h.startswith("# first_model")),
        len(head),
    )
    head[at:at] = [f"# occurrence: {occurrence}", f"# stage: {stage}"]
    return ("\n".join(head + body) + "\n").encode("utf-8")


def find_dumps(
    roots: list[pathlib.Path],
) -> dict[tuple[str, int, int, int, int], list[pathlib.Path]]:
    """Returns the dump files of every window found under ``roots``.

    Args:
        roots: Directories searched recursively.

    Returns:
        Window key to the files carrying it (in the order of ``roots``, so the
        first entry is from the earliest root).
    """
    found: dict[tuple[str, int, int, int, int], list[pathlib.Path]] = {}
    for root in roots:
        for path in sorted(root.rglob("*.stepdump")):
            if "withxtra" in path.name or "allstages" in path.name:
                continue
            key = window_key(header_of(path))
            if key is not None:
                found.setdefault(key, []).append(path)
    return found


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", type=pathlib.Path, nargs="+", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--manifest-out", type=pathlib.Path, default=None)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    regen = _load_regenerate()
    windows = regen.load_windows()
    found = find_dumps(args.dumps)
    packed: list[tuple[dict[str, Any], bytes]] = []
    first_path: pathlib.Path | None = None
    missing: list[str] = []
    for w in windows:
        key = (
            w["case"],
            w["stage"],
            w["occurrence"],
            w["first_model"],
            w["steps"],
        )
        paths = found.get(key, [])
        if not paths:
            missing.append(regen.window_id(w))
            continue
        if len(paths) > 1:
            print(
                f"note: {regen.window_id(w)} is carried by {len(paths)} files; "
                f"using the one under the earliest --dumps directory "
                f"({paths[0]})"
            )
        path = paths[0]
        payload = normalized(
            path.read_text(encoding="utf-8"), w["stage"], w["occurrence"]
        )
        packed.append((w, payload))
        first_path = first_path or path
    if missing and not args.allow_missing:
        raise SystemExit(f"no dump for windows: {', '.join(missing)}")
    if not packed:
        raise SystemExit("no window found")
    assert first_path is not None
    head = header_of(first_path)
    provenance = {
        "bundle": f"mesa-stepdump-windows-{args.version}",
        "mesa_version": head.get("mesa_version"),
        "compiler": head.get("compiler"),
        "mesa_sdk_version": head.get("mesa_sdk_version"),
        "omp_num_threads": head.get("omp_num_threads"),
        "window": "mid-run windows: the S-th process of a case, opening at "
        "the K-th rise of the model number to the first model; see "
        "windows.toml",
        "recipe": "benchmarks/fortran_mesa/stepdumps/run_stepdumps.py with "
        "stepdump_patch.py (write statements only), driven by "
        "regenerate_windows.py; MESA_SKIP_OPTIONAL=t",
        "layout": "<window id>/<suite>/<case>.stepdump; window id = "
        "<case>_s<stage>_k<occurrence>_m<first model>_n<steps>",
        "normalization": "record types the committed patch does not write "
        "were dropped; occurrence and stage header lines set from "
        "windows.toml",
        "capture_note": "windows captured before #608 came from a scratch "
        "driver and an overlay whose patch also wrote diagnostic records "
        "(dropped here); regeneration with this recipe was compared record "
        "for record for wd_stable_h_burn_s1_k2_m70_n12 and "
        "ns_h_s2_k1_m90_n12 only",
        "windows": [regen.window_id(w) for w, _ in packed],
        "not_packed": missing,
    }
    rows: list[tuple[str, int, str]] = []
    with _zstd_writer(args.out) as zfh:
        with tarfile.open(
            fileobj=zfh, mode="w|", format=tarfile.PAX_FORMAT
        ) as tf:
            rel = f"{ROOT}/PROVENANCE.json"
            doc = (json.dumps(provenance, indent=2) + "\n").encode("utf-8")
            rows.append((*_add(tf, rel, doc), rel))
            rel = f"{ROOT}/windows.toml"
            rows.append((*_add(tf, rel, regen.WINDOWS.read_bytes()), rel))
            members = []
            for w, payload in packed:
                suite, case = w["case"].split("/", 1)
                members.append(
                    (
                        f"{ROOT}/{regen.window_id(w)}/{suite}/{case}.stepdump",
                        payload,
                    )
                )
            for rel, payload in sorted(members):
                rows.append((*_add(tf, rel, payload), rel))
    if args.manifest_out is not None:
        args.manifest_out.write_text(
            "# sha256\tsize_bytes\trelpath\n"
            + "\n".join(f"{s}\t{n}\t{r}" for s, n, r in rows)
            + "\n",
            encoding="utf-8",
        )
    arc = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"archive:             {args.out.name}")
    print(f"archive_sha256:      {arc}")
    print(f"archive_size_bytes:  {args.out.stat().st_size}")
    print(f"unpacked_size_bytes: {sum(n for _, n, _ in rows)}")
    print(f"members:             {len(rows)}")
    print(f"windows packed:      {len(packed)} of {len(windows)}")
    if missing:
        print(f"windows missing:     {', '.join(missing)}")


if __name__ == "__main__":
    main()
