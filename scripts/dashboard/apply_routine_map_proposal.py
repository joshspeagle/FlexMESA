#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Apply a reviewed routine-map proposal to ``port_manifest.toml``.

Each ``[[files]]`` entry of the proposal must name an ``absent`` entry of the
manifest. That entry is raised to ``partial`` and takes the proposal's
``flexmesa_files``, ``routine_map`` (in proposal order) and
``routine_map_partial``. The proposal's ``notes`` replace the entry's notes,
prefixed with the review record that evidences the raise (`DECISIONS #79`).
Only the changed ``[[files]]`` blocks are rewritten. Every other byte of the
manifest is kept. The record's ``files`` list is set to the raised paths.

Usage::

    python scripts/dashboard/apply_routine_map_proposal.py \\
        --proposal benchmarks/dashboard/routine_map_gyre_adipls_proposed.toml \\
        --record notes/audit/routine_map_reviews/610b.toml
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import tomllib

from flexmesa.testing import port_manifest

MANIFEST = pathlib.Path("benchmarks/dashboard/port_manifest.toml")
_BLOCK_SEP = "\n[[files]]\n"


def _render_entry(entry: port_manifest.PortManifestEntry) -> str:
    """Return the ``[[files]]`` block body of ``entry`` in canonical form."""
    one = port_manifest.PortManifest(
        schema_version=1,
        upstream_version="",
        upstream_commit="",
        archive_md5="",
        entries=(entry,),
    )
    text = port_manifest.dump_port_manifest(one)
    return text.split("[[files]]\n", 1)[1]


def apply_proposal(
    manifest_text: str, proposal: dict[str, object], record_rel: str
) -> tuple[str, list[str]]:
    """Return the new manifest text and the raised ``upstream_path`` list.

    Args:
        manifest_text: Current ``port_manifest.toml`` text.
        proposal: Parsed proposal TOML.
        record_rel: Repository path of the review record, cited in notes.

    Returns:
        ``(text, raised)``.

    Raises:
        ValueError: If a proposed file is missing from the manifest, is not
            ``absent``, or names a partial key that is not mapped.
    """
    manifest = port_manifest.parse_port_manifest_text(manifest_text)
    by_path = manifest.by_upstream_path()
    head, *blocks = manifest_text.split(_BLOCK_SEP)
    index = {}
    for i, block in enumerate(blocks):
        m = re.match(r'upstream_path = "([^"]+)"', block)
        if m is None:
            raise ValueError(f"[[files]] block {i} has no upstream_path")
        index[m.group(1)] = i
    raised: list[str] = []
    files = proposal.get("files")
    if not isinstance(files, list):
        raise ValueError("proposal has no [[files]] list")
    for item in files:
        up = str(item["upstream_path"])
        prev = by_path.get(up)
        if prev is None or up not in index:
            raise ValueError(f"{up}: not in port_manifest.toml")
        if prev.status != port_manifest.PortStatus.ABSENT:
            raise ValueError(f"{up}: status is {prev.status.value}, not absent")
        rmap = tuple((str(k), str(v)) for k, v in item["routine_map"].items())
        keys = {k for k, _ in rmap}
        partial = tuple(
            sorted(str(k) for k in item.get("routine_map_partial", []))
        )
        missing = [k for k in partial if k not in keys]
        if missing:
            raise ValueError(f"{up}: partial keys not mapped: {missing}")
        note = str(item.get("notes", "")).strip()
        prefix = f"Routine map reviewed in {record_rel} (DECISIONS #79)."
        entry = port_manifest.PortManifestEntry(
            upstream_path=up,
            module=prev.module,
            upstream_lines=prev.upstream_lines,
            upstream_sha256=prev.upstream_sha256,
            status=port_manifest.PortStatus.PARTIAL,
            flexmesa_files=tuple(
                sorted(str(f) for f in item["flexmesa_files"])
            ),
            routine_map=rmap,
            routine_map_partial=partial,
            notes=f"{prefix} {note}",
        )
        old = blocks[index[up]]
        tail = "\n" if old.endswith("\n") else ""
        blocks[index[up]] = _render_entry(entry).rstrip("\n") + tail
        raised.append(up)
    return _BLOCK_SEP.join([head, *blocks]), sorted(raised)


def set_record_files(record_text: str, files: list[str]) -> str:
    """Return ``record_text`` with its ``files`` list replaced by ``files``."""
    listed = "".join(f'  "{f}",\n' for f in files)
    new, n = re.subn(
        r"^files = \[[^\]]*\]\n?",
        f"files = [\n{listed}]\n",
        record_text,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise ValueError("review record must hold exactly one files list")
    return new


def main(argv: list[str] | None = None) -> int:
    """Apply the proposal and update the review record."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--proposal", type=pathlib.Path, required=True)
    ap.add_argument("--record", type=pathlib.Path, required=True)
    ap.add_argument("--manifest", type=pathlib.Path, default=MANIFEST)
    args = ap.parse_args(argv)
    proposal = tomllib.loads(args.proposal.read_text(encoding="utf-8"))
    if args.record.is_absolute():
        ap.error("--record must be a path relative to the repository root")
    record_rel = args.record.as_posix()
    text, raised = apply_proposal(
        args.manifest.read_text(encoding="utf-8"), proposal, record_rel
    )
    port_manifest.parse_port_manifest_text(text)
    args.manifest.write_text(text, encoding="utf-8")
    args.record.write_text(
        set_record_files(args.record.read_text(encoding="utf-8"), raised),
        encoding="utf-8",
    )
    print(f"raised {len(raised)} entries from absent to partial")
    return 0


if __name__ == "__main__":
    sys.exit(main())
