#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Window-test planner and manifest generator for ``#684`` W1.

Implements ``.coordination/triage/window-rules.md`` sections 2, 3 and 4.3:

* **Pass 1 (no clock yet):** reads ``mesa_stage_steps.json`` plus MESA stage
  inlists / ``rn`` scripts (when ``--mesa-dir`` is provided) and reference
  histories to compute deterministic stage start points (CRC32 seed per
  ``star/<case>/<stage>``, point 1 + 25/50/75% ±5% + 95%, or point 1 only for
  short stages of ``<= 60`` models) and write per-case overlay snapshot specs
  ``<case>.spec`` (``<stage> <model>`` lines).
* **Pass 2 (after MESA overlay runs):** reads per-step wall clocks
  ``<case>.clock.tsv`` and ``<case>_<stage>_<model>.mod`` snapshots, computes
  ``W = clamp(steps_60s, 20, 300)`` from exact per-step wall times, removes
  duplicates within ``W`` (keeping the earlier point), checks ``.mod`` headers
  against Rule 4.2 (`prev_model` / `no-prev-model`) and Rule 4.3 (`net_name`,
  `initial_z`, `M/Msun`), computes snapshot SHA-256 hashes, and writes
  ``benchmarks/history_harness/windows_e2e.toml``.

``--check`` verifies determinism and zero machine paths, regenerating and
comparing ``windows_e2e.toml`` byte-for-byte when ``--clock-dir`` and
``--snapshot-dir`` are provided.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import re
import sys
import tomllib
import zlib
from typing import Any

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_STAGE_STEPS = (
    REPO / "benchmarks" / "history_harness" / "mesa_stage_steps.json"
)
DEFAULT_REF_HIST_DIR = (
    REPO / "benchmarks" / "history_harness" / "reference_histories" / "star"
)
DEFAULT_TIMINGS = REPO / "benchmarks" / "history_harness" / "case_timings.toml"
DEFAULT_OUT_TOML = REPO / "benchmarks" / "history_harness" / "windows_e2e.toml"

SHORT_STAGE_MAX_MODELS = 60
TARGET_WALL_SEC = 60.0
MIN_WINDOW_STEPS = 20
MAX_WINDOW_STEPS = 300
EMPTY_STAGE_REASON = "window not built (no MESA stage steps)"
FORBIDDEN_PATH_TOKENS = ("/home/", "/tmp/", "/mnt/", "/Users/")

# MESA write_model.f90 / read_model.f90 file_type bit indices.
BIT_FOR_2MODELS = 2
BIT_FOR_VELOCITY = 3
BIT_FOR_ROTATION = 4
BIT_FOR_MLT_VC = 5
BIT_FOR_RSP2 = 6
BIT_FOR_RTI = 7
BIT_FOR_U = 9
BIT_FOR_RSP = 15

_BOOL_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s*=\s*\.(true|false)\.", re.IGNORECASE
)
_INT_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*([-+]?\d+)\s*$")
_FLOAT_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[dDeE][-+]?\d+)?)\s*$"
)
_STR_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*'([^']*)'\s*$")
_EXTRA_INLIST_RE = re.compile(
    r"^\s*extra_(?:star_job|controls|eos|kap)_inlist\d*_name(?:\(\d+\))?\s*="
    r"\s*'([^']+)'",
    re.IGNORECASE,
)


@dataclasses.dataclass(frozen=True)
class NotBuiltWindow:
    """A planned window whose snapshot fails a Rule 4.3 check.

    Rule 4.3 leaves such a window unbuilt until claude-a rules on it; the
    manifest records it as ``window not built (<reason>)``.
    """

    window_id: str
    case: str
    stage: int
    start_model: int
    snapshot_file: str
    snapshot_sha256: str
    reason: str


@dataclasses.dataclass(frozen=True)
class StageStepsOverride:
    """Planner-local stage list that replaces a ``mesa_stage_steps.json`` row.

    Attributes:
        stage_steps: End model number of each stage, one per stage.
        out_txt_sha256: SHA-256 of the stored ``out.txt`` that the row in
            ``mesa_stage_steps.json`` was extracted from; the override applies
            only while the row still records this hash.
        source: Where the stage list comes from.
        source_lines: Quoted ``out.txt`` lines that justify the override.
    """

    stage_steps: tuple[int, ...]
    out_txt_sha256: str
    source: str
    source_lines: tuple[str, ...]


# twin_studies evolves two stars in one binary run: stages 1 and 2 are the
# two stars (80 steps each). mesa_stage_steps.json lists a third stage of 80
# steps because the final aggregate runtime line also matches the stage rule;
# that row is fixed in #693, so the planner overrides it here.
STAGE_STEPS_OVERRIDES: dict[str, StageStepsOverride] = {
    "twin_studies": StageStepsOverride(
        stage_steps=(80, 80),
        out_txt_sha256=(
            "2685d48338f5f3e84997203f3e849cd8d48134323d37cbcf1dbbef05da9eebfc"
        ),
        source=(
            "mesa-histories bundle runs/star/twin_studies/out.txt lines "
            "170-177: two forced_stop terminations (one per star), each "
            "followed by its runtime line, then a trailing aggregate runtime "
            "line that is not a stage"
        ),
        source_lines=(
            "terminated evolution: forced_stop",
            "runtime (minutes), retries, steps        1.64         0        80",
            "terminated evolution: forced_stop",
            "runtime (minutes), retries, steps        1.64         0        80",
            "runtime (minutes), retries, steps        1.64         0        80",
        ),
    ),
}


def case_stage_steps(case: str, row: dict[str, Any]) -> list[int]:
    """Return the stage end models of one case, applying any override.

    Args:
        case: Test-suite case name without the ``star/`` prefix.
        row: The case's row in ``mesa_stage_steps.json``.

    Returns:
        End model number of each stage; empty when MESA recorded no stages.

    Raises:
        SystemExit: An override exists but the row's ``out_txt_sha256`` no
            longer matches the ``out.txt`` the override was written against.
    """
    override = STAGE_STEPS_OVERRIDES.get(case)
    if override is None:
        return [int(n) for n in row.get("stage_steps", ())]
    row_sha = str(row.get("out_txt_sha256", ""))
    if row_sha != override.out_txt_sha256:
        raise SystemExit(
            f"{case}: stage-step override was written for out.txt "
            f"{override.out_txt_sha256}, but mesa_stage_steps.json records "
            f"{row_sha!r}; review the override"
        )
    return list(override.stage_steps)


@dataclasses.dataclass(frozen=True)
class StagePlanInput:
    """Metadata describing one stage of a ``star/<case>``."""

    case: str
    stage: int
    m_first: int
    m_last: int
    stage_len: int
    loads_prev_stage_cleanly: bool
    has_relax: bool
    inlist_name: str
    expected_net_name: str | None = None
    expected_initial_z: float | None = None
    expected_mass_msun: float | None = None
    expected_rotation: bool | None = None


@dataclasses.dataclass(frozen=True)
class CandidatePoint:
    """One Pass-1 candidate start point before deduplication."""

    stage: int
    kind: str
    start_model: int
    seed: int
    needs_overlay_snapshot: bool


@dataclasses.dataclass(frozen=True)
class StepRecord:
    """Parsed ``STEP`` line from ``<case>.clock.tsv``."""

    stage: int
    star_id: int
    model_number: int
    dt_wall: float
    nz: int
    nz_old: int
    generations: int
    prev_flag: bool


@dataclasses.dataclass(frozen=True)
class SnapRecord:
    """Parsed ``SNAP`` line from ``<case>.clock.tsv``."""

    stage: int
    model_number: int
    prev_flag: bool
    filename: str


@dataclasses.dataclass(frozen=True)
class StageEndRecord:
    """Parsed ``STAGE_END`` line from ``<case>.clock.tsv``."""

    stage: int
    star_id: int
    end_model: int
    termination_code: str
    result_reason: str


@dataclasses.dataclass(frozen=True)
class ClockData:
    """All records parsed from ``<case>.clock.tsv``."""

    header: dict[str, str]
    steps: dict[int, list[StepRecord]]
    snaps: dict[tuple[int, int], SnapRecord]
    stage_ends: dict[int, StageEndRecord]


@dataclasses.dataclass(frozen=True)
class ModHeader:
    """Parsed header fields of a MESA ``.mod`` snapshot."""

    file_type: int
    prev_model: bool
    v_flag: bool
    u_flag: bool
    rotation_flag: bool
    rti_flag: bool
    version_number: str
    mass_msun: float
    model_number: int
    star_age: float
    initial_z: float
    n_shells: int
    net_name: str
    species: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class PlannedWindow:
    """One resolved window entry emitted to ``windows_e2e.toml``."""

    window_id: str
    suite: str
    case: str
    stage: int
    point_kind: str
    seed: int
    stage_m_first: int
    stage_m_last: int
    start_model: int
    window_w: int
    mesa_window_steps: int
    mesa_stop_reason: str
    mean_step_sec: float
    snapshot_file: str
    snapshot_sha256: str
    prev_model: bool
    no_prev_model: bool
    net_name: str
    mass_msun: float
    initial_z: float
    rotation_flag: bool
    rule43_ok: bool


def stage_seed(case: str, stage: int, suite: str = "star") -> int:
    """Return the CRC32 seed defined in window-rules.md section 2."""
    return zlib.crc32(f"{suite}/{case}/{stage}".encode()) & 0xFFFFFFFF


def _strip_fortran_comment(line: str) -> str:
    """Strip ``!`` comments outside single/double quotes."""
    in_sq = False
    in_dq = False
    for idx, ch in enumerate(line):
        if ch == "'" and not in_dq:
            in_sq = not in_sq
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
        elif ch == "!" and not in_sq and not in_dq:
            return line[:idx]
    return line


def _parse_f64(raw: str) -> float:
    return float(raw.replace("d", "e").replace("D", "E"))


def _read_inlist_chain(
    case_dir: pathlib.Path, entry_inlist: str, max_depth: int = 5
) -> dict[str, Any]:
    """Read key-value settings across ``extra_*_inlist*_name`` includes."""
    seen: set[pathlib.Path] = set()
    ordered: list[pathlib.Path] = []

    def walk(path: pathlib.Path, depth: int) -> None:
        if depth > max_depth or not path.is_file() or path in seen:
            return
        seen.add(path)
        ordered.append(path)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for raw in lines:
            clean = _strip_fortran_comment(raw)
            m = _EXTRA_INLIST_RE.match(clean)
            if m:
                inc = case_dir / m.group(1)
                walk(inc, depth + 1)

    walk(case_dir / entry_inlist, 0)
    kv: dict[str, Any] = {}
    for path in ordered:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for raw in lines:
            clean = _strip_fortran_comment(raw).strip()
            if not clean:
                continue
            if (mb := _BOOL_RE.match(clean)) is not None:
                kv[mb.group(1).lower()] = mb.group(2).lower() == "true"
            elif (mi := _INT_RE.match(clean)) is not None:
                kv[mi.group(1).lower()] = int(mi.group(2))
            elif (mf := _FLOAT_RE.match(clean)) is not None:
                kv[mf.group(1).lower()] = _parse_f64(mf.group(2))
            elif (ms := _STR_RE.match(clean)) is not None:
                kv[ms.group(1).lower()] = ms.group(2)
    return kv


def parse_mod_header(path: pathlib.Path) -> ModHeader:
    """Parse header metadata and SHA-256 from a MESA ``.mod`` file."""
    raw_bytes = path.read_bytes()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    lines = raw_bytes.decode("utf-8", errors="replace").splitlines()
    idx = 0
    file_type: int | None = None
    while idx < len(lines):
        ln = lines[idx].strip()
        idx += 1
        if not ln or ln.startswith("!"):
            continue
        file_type = int(ln.split(None, 1)[0])
        break
    if file_type is None:
        raise ValueError(f"{path}: missing file_type header line")
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    props: dict[str, str] = {}
    while idx < len(lines):
        ln = lines[idx].strip()
        idx += 1
        if not ln:
            break
        ln = ln.split("!", 1)[0].strip()
        if not ln:
            continue
        parts = ln.split(None, 1)
        if len(parts) == 2:
            props[parts[0]] = parts[1].strip()
    return ModHeader(
        file_type=file_type,
        prev_model=bool(file_type & (1 << BIT_FOR_2MODELS)),
        v_flag=bool(file_type & (1 << BIT_FOR_VELOCITY)),
        u_flag=bool(file_type & (1 << BIT_FOR_U)),
        rotation_flag=bool(file_type & (1 << BIT_FOR_ROTATION)),
        rti_flag=bool(file_type & (1 << BIT_FOR_RTI)),
        version_number=props.get("version_number", "").strip("'\" "),
        mass_msun=_parse_f64(props["M/Msun"]),
        model_number=int(props["model_number"]),
        star_age=_parse_f64(props["star_age"]),
        initial_z=_parse_f64(props["initial_z"]),
        n_shells=int(props["n_shells"]),
        net_name=props.get("net_name", "").strip("'\" "),
        species=int(props["species"]),
        sha256=sha256,
    )


def _parse_rn_stages(
    case_dir: pathlib.Path,
) -> tuple[list[tuple[str, str | None]], dict[str, str]]:
    """Parse active ``do_one`` calls and ``MESA_SKIP_OPTIONAL`` copies."""
    rn = case_dir / "rn"
    if not rn.is_file():
        return [], {}
    stages: list[tuple[str, str | None]] = []
    skipped_copies: dict[str, str] = {}
    branch_stack: list[str] = []
    for raw in rn.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if re.match(r"^if\s+\[\s*-n\s+[\"']?\$MESA_SKIP_OPTIONAL", line):
            branch_stack.append("skip_then")
            continue
        if re.match(r"^if\s+\[\s*-z\s+[\"']?\$MESA_SKIP_OPTIONAL", line):
            branch_stack.append("skip_else")
            continue
        if line == "else" and branch_stack:
            top = branch_stack.pop()
            branch_stack.append(
                "skip_else" if top == "skip_then" else "skip_then"
            )
            continue
        if line == "fi" and branch_stack:
            branch_stack.pop()
            continue
        if "skip_else" in branch_stack:
            continue
        toks = line.split()
        if toks[0] == "cp" and len(toks) >= 3:
            skipped_copies[toks[2]] = toks[1]
        elif toks[0] == "do_one":
            inlist = toks[1] if len(toks) >= 2 else "inlist"
            out_mod = toks[2] if len(toks) >= 3 else None
            stages.append((inlist, out_mod))
    return stages, skipped_copies


def _history_first_last_model(hist: pathlib.Path) -> tuple[int, int] | None:
    if not hist.is_file():
        return None
    lines = hist.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 7:
        return None
    cols = lines[5].split()
    if "model_number" not in cols:
        return None
    mi = cols.index("model_number")
    first_m: int | None = None
    last_m: int | None = None
    for ln in lines[6:]:
        parts = ln.split()
        if len(parts) <= mi:
            continue
        m = int(float(parts[mi]))
        if first_m is None:
            first_m = m
        last_m = m
    if first_m is None or last_m is None:
        return None
    return first_m, last_m


def _inlist_has_relax(kv: dict[str, Any]) -> bool:
    return any(
        k.startswith(("relax_", "relax_initial_")) and v is True
        for k, v in kv.items()
    )


def plan_case_stages(
    case: str,
    stage_steps: list[int],
    mesa_dir: pathlib.Path | None,
    ref_hist_dir: pathlib.Path = DEFAULT_REF_HIST_DIR,
) -> list[StagePlanInput]:
    """Infer ``(m_first, m_last, loads_prev_stage_cleanly)`` per stage."""
    case_dir = (
        mesa_dir / "star" / "test_suite" / case
        if mesa_dir is not None
        else None
    )
    rn_stages: list[tuple[str, str | None]] = []
    skipped_copies: dict[str, str] = {}
    if case_dir is not None and case_dir.is_dir():
        rn_stages, skipped_copies = _parse_rn_stages(case_dir)

    hist_bounds = _history_first_last_model(ref_hist_dir / f"{case}.data")
    out: list[StagePlanInput] = []
    saved_models: dict[str, int] = {}

    for s_idx, m_last in enumerate(stage_steps, start=1):
        inlist_name = (
            rn_stages[s_idx - 1][0] if s_idx - 1 < len(rn_stages) else "inlist"
        )
        out_mod = (
            rn_stages[s_idx - 1][1] if s_idx - 1 < len(rn_stages) else None
        )
        kv = (
            _read_inlist_chain(case_dir, inlist_name)
            if case_dir is not None and case_dir.is_dir()
            else {}
        )
        has_relax = _inlist_has_relax(kv)
        loads_prev_cleanly = False
        m_0 = 0
        if kv.get("set_initial_model_number") is True:
            m_0 = int(kv.get("initial_model_number", 0))
        elif kv.get("load_saved_model") is True:
            lfile = str(kv.get("load_model_filename", ""))
            if lfile in saved_models:
                m_0 = saved_models[lfile]
                if not has_relax:
                    loads_prev_cleanly = True
            elif case_dir is not None:
                src_name = skipped_copies.get(lfile, lfile)
                mod_path = case_dir / src_name
                if mod_path.is_file():
                    m_0 = parse_mod_header(mod_path).model_number
                elif s_idx == len(stage_steps) and hist_bounds is not None:
                    # Fallback if external model is unavailable.
                    m_0 = max(0, hist_bounds[0] - 1)
            elif s_idx == len(stage_steps) and hist_bounds is not None:
                m_0 = max(0, hist_bounds[0] - 1)
        elif (
            case_dir is None
            and s_idx == len(stage_steps)
            and hist_bounds is not None
        ):
            m_0 = max(0, hist_bounds[0] - 1)
        elif case_dir is None and s_idx > 1 and m_last > stage_steps[s_idx - 2]:
            m_0 = stage_steps[s_idx - 2]

        m_first = m_0 + 1
        stage_len = max(1, m_last - m_0)
        if m_first > m_last:
            m_first = m_last
        if out_mod:
            saved_models[out_mod] = m_last

        exp_net = (
            str(kv["new_net_name"])
            if (
                (kv.get("change_net") or kv.get("change_initial_net"))
                and "new_net_name" in kv
            )
            else None
        )
        exp_z = (
            float(kv["new_z"])
            if kv.get("change_initial_z") and "new_z" in kv
            else None
        )
        rot_on = kv.get("change_rotation_flag") or kv.get(
            "change_initial_rotation_flag"
        )
        exp_rot = (
            bool(kv["new_rotation_flag"])
            if rot_on and "new_rotation_flag" in kv
            else None
        )
        out.append(
            StagePlanInput(
                case=case,
                stage=s_idx,
                m_first=m_first,
                m_last=m_last,
                stage_len=stage_len,
                loads_prev_stage_cleanly=loads_prev_cleanly,
                has_relax=has_relax,
                inlist_name=inlist_name,
                expected_net_name=exp_net,
                expected_initial_z=exp_z,
                expected_rotation=exp_rot,
            )
        )
    return out


def candidate_points_for_stage(
    case: str,
    stage: int,
    m_first: int,
    m_last: int,
    loads_prev_stage_cleanly: bool = False,
) -> list[CandidatePoint]:
    """Return Rule-2 candidate start points for one stage."""
    stage_len = max(1, m_last - m_first + 1)
    seed = stage_seed(case, stage)
    max_start = m_last if m_first >= m_last else m_last - 1

    if stage_len <= SHORT_STAGE_MAX_MODELS:
        return [
            CandidatePoint(
                stage=stage,
                kind="short_stage",
                start_model=m_first,
                seed=seed,
                needs_overlay_snapshot=not loads_prev_stage_cleanly,
            )
        ]

    rng = np.random.default_rng(seed)
    points = [
        CandidatePoint(
            stage=stage,
            kind="start",
            start_model=m_first,
            seed=seed,
            needs_overlay_snapshot=not loads_prev_stage_cleanly,
        )
    ]
    base = m_first - 1
    for kind, frac in (("frac_25", 0.25), ("frac_50", 0.50), ("frac_75", 0.75)):
        offset = float(rng.uniform(-0.05 * stage_len, 0.05 * stage_len))
        raw = base + round(frac * stage_len + offset)
        clamped = max(m_first, min(max_start, raw))
        points.append(
            CandidatePoint(
                stage=stage,
                kind=kind,
                start_model=clamped,
                seed=seed,
                needs_overlay_snapshot=True,
            )
        )
    raw_95 = base + round(0.95 * stage_len)
    clamped_95 = max(m_first, min(max_start, raw_95))
    points.append(
        CandidatePoint(
            stage=stage,
            kind="tail_95",
            start_model=clamped_95,
            seed=seed,
            needs_overlay_snapshot=True,
        )
    )
    return points


def deduplicate_stage_candidates(
    candidates_with_w: list[tuple[CandidatePoint, int]],
) -> list[tuple[CandidatePoint, int]]:
    """Apply Rule 2 duplicate removal: if within W, keep earlier point."""
    ordered = sorted(candidates_with_w, key=lambda item: item[0].start_model)
    kept: list[tuple[CandidatePoint, int]] = []
    for cand, w in ordered:
        if not kept:
            kept.append((cand, w))
            continue
        prev_cand, prev_w = kept[-1]
        if cand.start_model - prev_cand.start_model >= prev_w:
            kept.append((cand, w))
    return kept


def prune_case_snapshots(
    case: str,
    clock_path: pathlib.Path,
    snapshot_dir: pathlib.Path,
) -> tuple[list[str], list[str]]:
    """Keep only post-deduplication ``.mod`` files for ``case``.

    Raises before deleting any file if a kept snapshot is missing or the clock
    contains no ``STEP`` records, and uses an exact ``<case>_<stage>_<model>``
    pattern so prefix-sharing cases are never touched.
    """
    cdata = parse_clock_tsv(clock_path)
    if not cdata.steps:
        raise ValueError(f"{case}: no STEP records in {clock_path}")
    kept_names: set[str] = set()
    for stage, st_records in sorted(cdata.steps.items()):
        emp_first = st_records[0].model_number
        emp_last = st_records[-1].model_number
        cands = candidate_points_for_stage(
            case,
            stage,
            emp_first,
            emp_last,
            loads_prev_stage_cleanly=False,
        )
        with_w: list[tuple[CandidatePoint, int]] = []
        for cand in cands:
            w, _, _ = compute_window_w(st_records, cand.start_model)
            with_w.append((cand, w))
        kept = deduplicate_stage_candidates(with_w)
        for cand, _ in kept:
            kept_names.add(f"{case}_{stage}_{cand.start_model}.mod")

    if not kept_names:
        raise ValueError(f"{case}: zero kept snapshots computed")
    missing = [
        name
        for name in sorted(kept_names)
        if not (snapshot_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{case}: missing kept snapshot(s): {', '.join(missing)}"
        )

    pat = re.compile(rf"^{re.escape(case)}_(\d+)_(\d+)\.mod$")
    deleted: list[str] = []
    for entry in sorted(snapshot_dir.iterdir()):
        if (
            entry.is_file()
            and pat.match(entry.name) is not None
            and entry.name not in kept_names
        ):
            entry.unlink()
            deleted.append(entry.name)
    return sorted(kept_names), deleted


def parse_clock_tsv(path: pathlib.Path) -> ClockData:
    """Parse a ``<case>.clock.tsv`` file emitted by ``run_stepdumps.py``."""
    header: dict[str, str] = {}
    steps: dict[int, list[StepRecord]] = {}
    snaps: dict[tuple[int, int], SnapRecord] = {}
    stage_ends: dict[int, StageEndRecord] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        if line.startswith("# "):
            key, _, val = line[2:].partition(":")
            header[key.strip()] = val.strip()
            continue
        parts = line.split("\t")
        tag = parts[0]
        if tag == "STEP":
            rec = StepRecord(
                stage=int(parts[1]),
                star_id=int(parts[2]),
                model_number=int(parts[3]),
                dt_wall=float(parts[4]),
                nz=int(parts[5]),
                nz_old=int(parts[6]),
                generations=int(parts[7]),
                prev_flag=parts[8].strip().upper() == "T",
            )
            steps.setdefault(rec.stage, []).append(rec)
        elif tag == "SNAP":
            srec = SnapRecord(
                stage=int(parts[1]),
                model_number=int(parts[2]),
                prev_flag=parts[3].strip().upper() == "T",
                filename=parts[4],
            )
            snaps[(srec.stage, srec.model_number)] = srec
        elif tag == "STAGE_END":
            erec = StageEndRecord(
                stage=int(parts[1]),
                star_id=int(parts[2]),
                end_model=int(parts[3]),
                termination_code=parts[4],
                result_reason=parts[5],
            )
            stage_ends[erec.stage] = erec
    return ClockData(
        header=header, steps=steps, snaps=snaps, stage_ends=stage_ends
    )


def compute_window_w(
    stage_steps: list[StepRecord], start_model: int
) -> tuple[int, int, float]:
    """Compute Rule 3 ``W``, ``mesa_window_steps`` and local mean step time.

    Returns:
        ``(W, mesa_window_steps, mean_step_sec)``.
    """
    after = [s for s in stage_steps if s.model_number > start_model]
    if not after:
        # Single-step stage fallback.
        mean_sec = (
            sum(s.dt_wall for s in stage_steps) / len(stage_steps)
            if stage_steps
            else 1.0
        )
        raw = round(TARGET_WALL_SEC / max(mean_sec, 1e-6))
        w = max(MIN_WINDOW_STEPS, min(MAX_WINDOW_STEPS, raw))
        return w, 0, mean_sec

    elapsed = 0.0
    steps_in_60s = 0
    reached_60 = False
    for s in after:
        elapsed += max(0.0, s.dt_wall)
        steps_in_60s += 1
        if elapsed >= TARGET_WALL_SEC:
            reached_60 = True
            break

    if reached_60:
        mean_sec = elapsed / steps_in_60s
        raw_steps = steps_in_60s
    else:
        total_rem = sum(max(0.0, s.dt_wall) for s in after)
        mean_sec = total_rem / len(after)
        raw_steps = (
            round(TARGET_WALL_SEC / mean_sec)
            if mean_sec > 1e-9
            else MAX_WINDOW_STEPS
        )
    w = max(MIN_WINDOW_STEPS, min(MAX_WINDOW_STEPS, raw_steps))
    m_end = stage_steps[-1].model_number
    mesa_window_steps = min(w, max(0, m_end - start_model))
    return w, mesa_window_steps, mean_sec


def check_rule43(
    hdr: ModHeader, stage_info: StagePlanInput
) -> tuple[bool, list[str]]:
    """Verify snapshot header consistency with stage controls (Rule 4.3)."""
    issues: list[str] = []
    if (
        stage_info.expected_net_name is not None
        and hdr.net_name != stage_info.expected_net_name
    ):
        issues.append(
            f"net_name {hdr.net_name!r} != {stage_info.expected_net_name!r}"
        )
    if stage_info.expected_initial_z is not None:
        tol = 1e-12 * max(1.0, abs(stage_info.expected_initial_z))
        if abs(hdr.initial_z - stage_info.expected_initial_z) > tol:
            issues.append(
                f"initial_z {hdr.initial_z} != {stage_info.expected_initial_z}"
            )
    if stage_info.expected_rotation is True and not hdr.rotation_flag:
        issues.append("rotation_flag is false when stage enables rotation")
    return not issues, issues


def write_case_specs(
    stage_steps_path: pathlib.Path,
    mesa_dir: pathlib.Path | None,
    spec_dir: pathlib.Path,
    cases: list[str] | None = None,
) -> dict[str, list[StagePlanInput]]:
    """Write ``<case>.spec`` files for Pass 1 and return planned stages."""
    data = json.loads(stage_steps_path.read_text(encoding="utf-8"))["cases"]
    spec_dir.mkdir(parents=True, exist_ok=True)
    planned: dict[str, list[StagePlanInput]] = {}
    target_set = set(cases) if cases is not None else None
    for ident, row in sorted(data.items()):
        if not ident.startswith("star/"):
            continue
        case = ident.split("/", 1)[1]
        if target_set is not None and case not in target_set:
            continue
        st_steps = case_stage_steps(case, row)
        if not st_steps:
            planned[case] = []
            continue
        stages = plan_case_stages(case, st_steps, mesa_dir)
        planned[case] = stages
        spec_lines: list[str] = []
        for st in stages:
            cands = candidate_points_for_stage(
                case,
                st.stage,
                st.m_first,
                st.m_last,
                loads_prev_stage_cleanly=st.loads_prev_stage_cleanly,
            )
            seen_models: set[int] = set()
            for c in cands:
                if c.start_model not in seen_models:
                    seen_models.add(c.start_model)
                    spec_lines.append(f"{st.stage} {c.start_model}")
        (spec_dir / f"{case}.spec").write_text(
            "\n".join(spec_lines) + "\n", encoding="utf-8"
        )
    return planned


def build_windows_from_clocks(
    stage_steps_path: pathlib.Path,
    mesa_dir: pathlib.Path | None,
    clock_dir: pathlib.Path,
    snapshot_dir: pathlib.Path,
    cases: list[str] | None = None,
) -> tuple[
    list[PlannedWindow],
    list[tuple[str, str]],
    dict[str, str],
    list[NotBuiltWindow],
]:
    """Build Pass-2 windows from overlay clock TSVs and ``.mod`` snapshots.

    Returns:
        ``(windows, skipped, meta_hdr, not_built)``: the built windows,
        ``(case, reason)`` for cases without MESA stage steps, MESA version
        metadata from the clock headers, and the windows whose snapshot
        fails a Rule 4.3 check.
    """
    raw_json = json.loads(stage_steps_path.read_text(encoding="utf-8"))
    ref_cases = raw_json["cases"]
    windows: list[PlannedWindow] = []
    skipped: list[tuple[str, str]] = []
    not_built: list[NotBuiltWindow] = []
    meta_hdr: dict[str, str] = {}
    target_set = set(cases) if cases is not None else None

    for ident, row in sorted(ref_cases.items()):
        if not ident.startswith("star/"):
            continue
        case = ident.split("/", 1)[1]
        if target_set is not None and case not in target_set:
            continue
        st_steps = case_stage_steps(case, row)
        if not st_steps:
            skipped.append((case, EMPTY_STAGE_REASON))
            continue
        clock_path = clock_dir / f"{case}.clock.tsv"
        if not clock_path.is_file():
            raise SystemExit(f"{case}: missing clock log {clock_path}")
        cdata = parse_clock_tsv(clock_path)
        for k in ("mesa_version", "compiler", "mesa_sdk_version"):
            if k in cdata.header and k not in meta_hdr:
                meta_hdr[k] = cdata.header[k]
        stages = plan_case_stages(case, st_steps, mesa_dir)
        for st in stages:
            st_records = cdata.steps.get(st.stage, [])
            if not st_records:
                raise SystemExit(
                    f"{case} stage {st.stage}: no STEP records in {clock_path}"
                )
            emp_first = st_records[0].model_number
            emp_last = st_records[-1].model_number
            if emp_last != st.m_last:
                raise SystemExit(
                    f"{case} stage {st.stage}: empirical end model {emp_last} "
                    f"!= mesa_stage_steps.json {st.m_last}"
                )
            cands = candidate_points_for_stage(
                case,
                st.stage,
                emp_first,
                emp_last,
                loads_prev_stage_cleanly=False,
            )
            with_w: list[tuple[CandidatePoint, int]] = []
            w_info: dict[int, tuple[int, int, float]] = {}
            for c in cands:
                w, m_steps, mean_s = compute_window_w(st_records, c.start_model)
                with_w.append((c, w))
                w_info[c.start_model] = (w, m_steps, mean_s)
            kept = deduplicate_stage_candidates(with_w)
            end_rec = cdata.stage_ends.get(st.stage)
            stop_reason = (
                end_rec.termination_code if end_rec is not None else "unknown"
            )
            for cand, _ in kept:
                w, m_steps, mean_s = w_info[cand.start_model]
                snap_name = f"{case}_{st.stage}_{cand.start_model}.mod"
                snap_path = snapshot_dir / snap_name
                if not snap_path.is_file():
                    raise SystemExit(
                        f"{case} stage {st.stage} model {cand.start_model}: "
                        f"missing snapshot {snap_path}"
                    )
                hdr = parse_mod_header(snap_path)
                if hdr.model_number != cand.start_model:
                    raise SystemExit(
                        f"{snap_name}: header model {hdr.model_number} != "
                        f"{cand.start_model}"
                    )
                snap_rec = cdata.snaps.get((st.stage, cand.start_model))
                if (
                    snap_rec is not None
                    and snap_rec.prev_flag != hdr.prev_model
                ):
                    raise SystemExit(
                        f"{snap_name}: overlay prev_flag {snap_rec.prev_flag} "
                        f"!= .mod file_type prev_model {hdr.prev_model}"
                    )
                r43_ok, r43_issues = check_rule43(hdr, st)
                wid = f"{case}_s{st.stage}_m{cand.start_model}"
                if not r43_ok:
                    not_built.append(
                        NotBuiltWindow(
                            window_id=wid,
                            case=case,
                            stage=st.stage,
                            start_model=cand.start_model,
                            snapshot_file=snap_name,
                            snapshot_sha256=hdr.sha256,
                            reason=(
                                "window not built (Rule 4.3: "
                                + "; ".join(r43_issues)
                                + ")"
                            ),
                        )
                    )
                    continue
                windows.append(
                    PlannedWindow(
                        window_id=wid,
                        suite="star",
                        case=case,
                        stage=st.stage,
                        point_kind=cand.kind,
                        seed=cand.seed,
                        stage_m_first=emp_first,
                        stage_m_last=emp_last,
                        start_model=cand.start_model,
                        window_w=w,
                        mesa_window_steps=m_steps,
                        mesa_stop_reason=stop_reason,
                        mean_step_sec=round(mean_s, 6),
                        snapshot_file=snap_name,
                        snapshot_sha256=hdr.sha256,
                        prev_model=hdr.prev_model,
                        no_prev_model=not hdr.prev_model,
                        net_name=hdr.net_name,
                        mass_msun=hdr.mass_msun,
                        initial_z=hdr.initial_z,
                        rotation_flag=hdr.rotation_flag,
                        rule43_ok=r43_ok,
                    )
                )
    return windows, skipped, meta_hdr, not_built


def render_windows_toml(
    windows: list[PlannedWindow],
    skipped: list[tuple[str, str]],
    meta_hdr: dict[str, str],
    overrides: dict[str, StageStepsOverride] | None = None,
    not_built: list[NotBuiltWindow] | None = None,
) -> str:
    """Render deterministic TOML for ``windows_e2e.toml``.

    Args:
        windows: Resolved windows, in emission order.
        skipped: ``(case, reason)`` for each case without windows.
        meta_hdr: MESA version metadata read from the clock headers.
        overrides: Stage-step overrides applied while planning, keyed by
            case; each is recorded with its source lines.
        not_built: Windows left unbuilt by a failed Rule 4.3 check.

    Returns:
        The TOML text.
    """
    lines = [
        "# FlexMESA MESA 26.4.1 windowed integration manifest (#684 W1)",
        f"# mesa_version: {meta_hdr.get('mesa_version', 'r26.4.1')}",
        "# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd",
        f"# mesasdk_version: {meta_hdr.get('mesa_sdk_version', 'unknown')}",
        f"# compiler: {meta_hdr.get('compiler', 'unknown')}",
        "# recipe: scripts/windows/plan_windows.py",
        "",
        "[metadata]",
        'schema = "windows_e2e_v1"',
        f"window_count = {len(windows)}",
        f"skipped_case_count = {len(skipped)}",
        f"not_built_window_count = {len(not_built or [])}",
        f"short_stage_max_models = {SHORT_STAGE_MAX_MODELS}",
        f"target_wall_sec = {int(TARGET_WALL_SEC)}",
        f"min_w = {MIN_WINDOW_STEPS}",
        f"max_w = {MAX_WINDOW_STEPS}",
        "",
    ]
    for case, ovr in sorted((overrides or {}).items()):
        steps = ", ".join(str(n) for n in ovr.stage_steps)
        quoted = ", ".join(json.dumps(ln) for ln in ovr.source_lines)
        lines.extend(
            [
                "[[stage_steps_overrides]]",
                'suite = "star"',
                f'case = "{case}"',
                f"stage_steps = [{steps}]",
                f'out_txt_sha256 = "{ovr.out_txt_sha256}"',
                f"source = {json.dumps(ovr.source)}",
                f"source_lines = [{quoted}]",
                "",
            ]
        )
    for case, reason in skipped:
        lines.extend(
            [
                "[[cases_skipped]]",
                'suite = "star"',
                f'case = "{case}"',
                f'reason = "{reason}"',
                "",
            ]
        )
    for nb in not_built or []:
        lines.extend(
            [
                "[[windows_not_built]]",
                f'id = "{nb.window_id}"',
                'suite = "star"',
                f'case = "{nb.case}"',
                f"stage = {nb.stage}",
                f"start_model = {nb.start_model}",
                f'snapshot_file = "{nb.snapshot_file}"',
                f'snapshot_sha256 = "{nb.snapshot_sha256}"',
                f"reason = {json.dumps(nb.reason)}",
                "",
            ]
        )
    for w in windows:
        lines.extend(
            [
                "[[windows]]",
                f'id = "{w.window_id}"',
                f'suite = "{w.suite}"',
                f'case = "{w.case}"',
                f"stage = {w.stage}",
                f'point_kind = "{w.point_kind}"',
                f"seed = {w.seed}",
                f"stage_m_first = {w.stage_m_first}",
                f"stage_m_last = {w.stage_m_last}",
                f"start_model = {w.start_model}",
                f"window_w = {w.window_w}",
                f"mesa_window_steps = {w.mesa_window_steps}",
                f'mesa_stop_reason = "{w.mesa_stop_reason}"',
                f"mean_step_sec = {w.mean_step_sec:.6f}",
                f'snapshot_file = "{w.snapshot_file}"',
                f'snapshot_sha256 = "{w.snapshot_sha256}"',
                f"prev_model = {str(w.prev_model).lower()}",
                f"no_prev_model = {str(w.no_prev_model).lower()}",
                f'net_name = "{w.net_name}"',
                f"mass_msun = {w.mass_msun:.17e}",
                f"initial_z = {w.initial_z:.17e}",
                f"rotation_flag = {str(w.rotation_flag).lower()}",
                f"rule43_ok = {str(w.rule43_ok).lower()}",
                "",
            ]
        )
    text = "\n".join(lines)
    for needle in FORBIDDEN_PATH_TOKENS:
        if needle in text:
            raise ValueError(f"machine path {needle!r} leaked into TOML")
    return text


def check_committed_toml(toml_path: pathlib.Path) -> tuple[int, int]:
    """Validate committed ``windows_e2e.toml`` schema and path hygiene."""
    text = toml_path.read_text(encoding="utf-8")
    for needle in FORBIDDEN_PATH_TOKENS:
        if needle in text:
            raise SystemExit(f"{toml_path}: contains machine path {needle!r}")
    parsed = tomllib.loads(text)
    wins = parsed.get("windows", [])
    if not isinstance(wins, list) or not wins:
        raise SystemExit(f"{toml_path}: missing [[windows]] entries")
    seen_ids: set[str] = set()
    for w in wins:
        wid = str(w["id"])
        if wid in seen_ids:
            raise SystemExit(f"{toml_path}: duplicate window id {wid}")
        seen_ids.add(wid)
        sha = str(w["snapshot_sha256"])
        if len(sha) != 64:
            raise SystemExit(f"{toml_path}: bad sha256 on {wid}")
        if not (MIN_WINDOW_STEPS <= int(w["window_w"]) <= MAX_WINDOW_STEPS):
            raise SystemExit(f"{toml_path}: window_w out of bounds on {wid}")
    for nb in parsed.get("windows_not_built", []):
        wid = str(nb.get("id", ""))
        if not wid or wid in seen_ids:
            raise SystemExit(f"{toml_path}: bad or duplicate window id {wid!r}")
        seen_ids.add(wid)
        if len(str(nb.get("snapshot_sha256", ""))) != 64:
            raise SystemExit(f"{toml_path}: bad sha256 on {wid}")
        if not str(nb.get("reason", "")).startswith("window not built ("):
            raise SystemExit(f"{toml_path}: missing not-built reason on {wid}")
    skipped = parsed.get("cases_skipped", [])
    seen_skipped: set[str] = set()
    for entry in skipped:
        suite = str(entry.get("suite", ""))
        case = str(entry.get("case", ""))
        reason = str(entry.get("reason", ""))
        if not suite or not case or not reason:
            raise SystemExit(f"{toml_path}: incomplete [[cases_skipped]] entry")
        if case in seen_skipped:
            raise SystemExit(f"{toml_path}: duplicate skipped case {case}")
        seen_skipped.add(case)
    return len(wins), len(skipped)


def estimate_full_star_workload(
    stage_steps_path: pathlib.Path,
    mesa_dir: pathlib.Path | None,
    timings_path: pathlib.Path = DEFAULT_TIMINGS,
) -> tuple[int, int, int, float]:
    """Summarize Pass-1 stage/candidate counts and estimated MESA hours."""
    data = json.loads(stage_steps_path.read_text(encoding="utf-8"))["cases"]
    timings = tomllib.loads(timings_path.read_text(encoding="utf-8"))
    wall_by_case: dict[str, float] = {}
    for row in timings.get("cases", []):
        if row.get("suite") == "star" and "mesa_wall_sec" in row:
            wall_by_case[str(row["case"])] = float(row["mesa_wall_sec"])
    n_active = 0
    n_stages = 0
    n_candidates = 0
    total_sec = 0.0
    for ident, row in sorted(data.items()):
        if not ident.startswith("star/"):
            continue
        case = ident.split("/", 1)[1]
        st_steps = case_stage_steps(case, row)
        if not st_steps:
            continue
        n_active += 1
        stages = plan_case_stages(case, st_steps, mesa_dir)
        n_stages += len(stages)
        for st in stages:
            n_candidates += len(
                candidate_points_for_stage(
                    case, st.stage, st.m_first, st.m_last
                )
            )
        total_sec += wall_by_case.get(case, 0.0)
    return n_active, n_stages, n_candidates, total_sec / 3600.0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-steps", type=pathlib.Path, default=DEFAULT_STAGE_STEPS
    )
    parser.add_argument("--mesa-dir", type=pathlib.Path, default=None)
    parser.add_argument("--spec-dir", type=pathlib.Path, default=None)
    parser.add_argument("--clock-dir", type=pathlib.Path, default=None)
    parser.add_argument("--snapshot-dir", type=pathlib.Path, default=None)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--prune-case",
        type=str,
        default=None,
        help="Prune unkept Pass-1 .mod files for one case after its clock log.",
    )
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT_TOML)
    parser.add_argument("--write-toml", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    if args.prune_case is not None:
        if args.clock_dir is None or args.snapshot_dir is None:
            parser.error(
                "--prune-case requires both --clock-dir and --snapshot-dir"
            )
        kept, deleted = prune_case_snapshots(
            args.prune_case,
            args.clock_dir / f"{args.prune_case}.clock.tsv",
            args.snapshot_dir,
        )
        print(
            f"{args.prune_case}: kept {len(kept)} snapshot(s), "
            f"deleted {len(deleted)} unkept"
        )
        return 0

    if args.plan_only:
        if args.spec_dir is not None:
            planned = write_case_specs(
                args.stage_steps, args.mesa_dir, args.spec_dir, args.cases
            )
            total_s = sum(len(v) for v in planned.values())
            print(
                f"Pass 1 planned {len(planned)} cases, {total_s} stages -> "
                f"{args.spec_dir}"
            )
        n_act, n_st, n_cand, hrs = estimate_full_star_workload(
            args.stage_steps, args.mesa_dir
        )
        print(
            f"Full star suite estimate: active_cases={n_act}, "
            f"stages={n_st}, pass1_candidates={n_cand}, "
            f"estimated_mesa_hours={hrs:.2f}h (~{hrs / 4.0:.2f}h @ 4 jobs)"
        )
        return 0

    if args.clock_dir is not None and args.snapshot_dir is not None:
        wins, skipped, hdr, not_built = build_windows_from_clocks(
            args.stage_steps,
            args.mesa_dir,
            args.clock_dir,
            args.snapshot_dir,
            args.cases,
        )
        used_cases = {w.case for w in wins}
        applied = {
            case: ovr
            for case, ovr in STAGE_STEPS_OVERRIDES.items()
            if case in used_cases
        }
        rendered = render_windows_toml(wins, skipped, hdr, applied, not_built)
        if args.check:
            existing = args.out.read_text(encoding="utf-8")
            if existing != rendered:
                raise SystemExit(
                    f"{args.out} does not match regenerated windows TOML"
                )
            print(
                f"{args.out}: matches regenerated output ({len(wins)} windows)"
            )
            return 0
        if args.write_toml:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered, encoding="utf-8")
            print(
                f"Wrote {len(wins)} windows ({len(not_built)} not built, "
                f"{len(skipped)} cases skipped) to {args.out}"
            )
        else:
            sys.stdout.write(rendered)
        return 0

    if args.check:
        n_wins, n_skip = check_committed_toml(args.out)
        print(f"{args.out}: valid ({n_wins} windows, {n_skip} skipped)")
        return 0

    parser.error(
        "pass --plan-only, --check, or both --clock-dir and --snapshot-dir"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
