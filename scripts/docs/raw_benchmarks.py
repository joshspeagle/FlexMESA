# SPDX-License-Identifier: GPL-3.0-or-later
"""Load committed raw benchmark captures (`benchmarks/raw/<release>/`)."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
from collections.abc import Mapping

DEFAULT_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_MESA_RELEASE = "26.4.1"

WAVE_ORDER: tuple[str, ...] = (
    "wave1",
    "wave2",
    "wave3a",
    "wave3b",
    "wave4",
    "wave5",
)

_WAVE_WORKLOADS: dict[str, tuple[str, ...]] = {
    "wave1": (
        "auto_diff",
        "interp_2d",
        "neu",
        "mtx_tridiag",
        "turb_mlt",
        "turb_tdc",
    ),
    "wave2": ("chem_comp", "ion_fe56", "ion_4d", "colors_sed"),
    "wave3a": ("eos_ideal", "eos_helm", "eos_pc", "eos_skye"),
    "wave3b": (
        "eos_4d",
        "kap_fixed",
        "rates_screen",
        "net_eval",
        "atm_table",
    ),
    "wave4": (
        "star_micro_vars",
        "star_hydro_eqns",
        "star_newton_step",
        "star_remesh",
        "star_evolve_step",
    ),
    "wave5": (
        "w5_diff_rot",
        "w5_riemann_rsp2",
        "w5_binary_rlof",
        "w5_astero_seism",
    ),
}

WORKLOAD_TO_WAVE: dict[str, str] = {
    workload: wave
    for wave, workloads in _WAVE_WORKLOADS.items()
    for workload in workloads
}

_TABLE_ROW_RE = re.compile(
    r"^\|\s*`([^`]+)`\s*\|\s*(\d+)\s*\|"
    r"\s*([0-9eE+.\-]+)\s*\+/-\s*([0-9eE+.\-]+)\s*\|"
    r"\s*([0-9eE+.\-]+)\s*\+/-\s*([0-9eE+.\-]+)\s*\|"
    r"\s*([0-9eE+.\-]+)\s*\+/-\s*([0-9eE+.\-]+)\s*\|"
    r"\s*([0-9eE+.\-]+)\s*\+/-\s*([0-9eE+.\-]+)\s*\|"
    r"\s*([0-9eE+.\-]+)\s*\|"
    r"\s*([0-9eE+.\-]+)\s*\|$"
)

_BCYCLIC_RE = re.compile(
    r"^#\s*N_zones\s*=\s*(\d+):\s*"
    r"1-thread\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)\s*us,\s*"
    r"4-thread OpenMP\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)\s*us$"
)


@dataclasses.dataclass(frozen=True)
class StageTiming:
    """Median and IQR timing (plus optional compile time) in microseconds."""

    median_us: float
    iqr_us: float
    compile_us: float = 0.0


@dataclasses.dataclass(frozen=True)
class RawBenchRecord:
    """Four-stage latency and checksum record for one workload and size."""

    wave: str
    workload: str
    n_zones: int
    stage0: StageTiming
    stage1: StageTiming
    stage2: StageTiming
    stage3: StageTiming
    max_rel_err: float
    checksum: float


@dataclasses.dataclass(frozen=True)
class BcyclicTiming:
    """Single-matrix MESA `bcyclic_dble_decsolblk` 1t and 4t OpenMP timings."""

    n_zones: int
    t1_median_us: float
    t1_iqr_us: float
    t4_median_us: float
    t4_iqr_us: float


@dataclasses.dataclass(frozen=True)
class RawBenchmarkSuite:
    """Complete parsed raw benchmark dataset for one MESA release."""

    provenance: dict[str, object]
    python_metadata: dict[str, object]
    fortran: dict[tuple[str, int], RawBenchRecord]
    rust: dict[tuple[str, int], RawBenchRecord]
    python: dict[tuple[str, int], RawBenchRecord]
    fortran_bcyclic: dict[int, BcyclicTiming]
    comparator_kind: dict[str, str]

    def get(self, backend: str, workload: str, n_zones: int) -> RawBenchRecord:
        """Return record for `(backend, workload, n_zones)`."""
        table_map = {
            "fortran": self.fortran,
            "rust": self.rust,
            "python": self.python,
        }
        if backend not in table_map:
            raise KeyError(f"unknown backend {backend!r}")
        return table_map[backend][(workload, n_zones)]


def _parse_markdown_records(
    text: str, *, expected_wave: str | None = None
) -> dict[tuple[str, int], RawBenchRecord]:
    """Parse Markdown four-stage benchmark rows from raw capture text."""
    out: dict[tuple[str, int], RawBenchRecord] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("| `"):
            continue
        match = _TABLE_ROW_RE.match(line)
        if match is None:
            raise ValueError(f"Unparseable benchmark row: {line!r}")
        workload = match.group(1).strip()
        wave = WORKLOAD_TO_WAVE.get(workload)
        if wave is None:
            raise ValueError(f"Unknown workload name {workload!r}")
        if expected_wave is not None and wave != expected_wave:
            raise ValueError(
                f"Workload {workload!r} belongs to {wave}, not {expected_wave}"
            )
        nz = int(match.group(2))
        rec = RawBenchRecord(
            wave=wave,
            workload=workload,
            n_zones=nz,
            stage0=StageTiming(float(match.group(3)), float(match.group(4))),
            stage1=StageTiming(float(match.group(5)), float(match.group(6))),
            stage2=StageTiming(float(match.group(7)), float(match.group(8))),
            stage3=StageTiming(float(match.group(9)), float(match.group(10))),
            max_rel_err=float(match.group(11)),
            checksum=float(match.group(12)),
        )
        key = (workload, nz)
        if key in out:
            raise ValueError(f"Duplicate benchmark record {key}")
        out[key] = rec
    return out


def _parse_bcyclic_comments(text: str) -> dict[int, BcyclicTiming]:
    """Parse `Single-Matrix BCYCLIC` OpenMP timing lines in Wave 1 output."""
    out: dict[int, BcyclicTiming] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = _BCYCLIC_RE.match(line)
        if match is None:
            continue
        nz = int(match.group(1))
        out[nz] = BcyclicTiming(
            n_zones=nz,
            t1_median_us=float(match.group(2)),
            t1_iqr_us=float(match.group(3)),
            t4_median_us=float(match.group(4)),
            t4_iqr_us=float(match.group(5)),
        )
    return out


def _parse_stage_dict(obj: object) -> StageTiming:
    """Convert JSON stage timing dict into a `StageTiming`."""
    if not isinstance(obj, Mapping):
        raise ValueError("stage timing must be a JSON object")
    return StageTiming(
        median_us=float(str(obj["median_us"])),
        iqr_us=float(str(obj["iqr_us"])),
        compile_us=float(str(obj.get("compile_us", 0.0))),
    )


def load_raw_benchmarks(
    repo_root: pathlib.Path = DEFAULT_REPO_ROOT,
    mesa_release: str = DEFAULT_MESA_RELEASE,
) -> RawBenchmarkSuite:
    """Load Fortran, Rust, and Python/JAX captures from `benchmarks/raw/`."""
    raw_dir = repo_root / "benchmarks" / "raw" / mesa_release
    prov = json.loads((raw_dir / "provenance.json").read_text("utf-8"))
    if not isinstance(prov, dict):
        raise ValueError("provenance.json must decode to a JSON object")

    comparators_raw = prov.get("comparators")
    if not isinstance(comparators_raw, dict):
        raise ValueError("provenance.json missing 'comparators' object")
    comp_map: dict[str, str] = {}
    for kind in (
        "mesa_26_4_1_all_stages",
        "mesa_26_4_1_stages_0_1_reference_fortran_stages_2_3",
        "hybrid_mesa_helpers_and_reference_fortran",
        "standalone_reference_fortran_non_mesa",
    ):
        items = comparators_raw.get(kind)
        if not isinstance(items, list):
            raise ValueError(f"provenance.json missing comparators.{kind}")
        for name in items:
            comp_map[str(name)] = kind

    fortran: dict[tuple[str, int], RawBenchRecord] = {}
    bcyclic: dict[int, BcyclicTiming] = {}
    for wave in WAVE_ORDER:
        f_txt = (raw_dir / f"fortran_{wave}.txt").read_text("utf-8")
        fortran.update(_parse_markdown_records(f_txt, expected_wave=wave))
        if wave == "wave1":
            bcyclic.update(_parse_bcyclic_comments(f_txt))

    rust_txt = (raw_dir / "rust_waves_1_to_5.txt").read_text("utf-8")
    rust = _parse_markdown_records(rust_txt)

    py_doc = json.loads(
        (raw_dir / "python_waves_1_to_5.json").read_text("utf-8")
    )
    if not isinstance(py_doc, dict):
        raise ValueError("python_waves_1_to_5.json must decode to an object")
    py_meta_raw = py_doc.get("metadata", {})
    py_waves_raw = py_doc.get("waves", {})
    if not isinstance(py_meta_raw, dict) or not isinstance(py_waves_raw, dict):
        raise ValueError("python_waves_1_to_5.json malformed")

    python: dict[tuple[str, int], RawBenchRecord] = {}
    for wave in WAVE_ORDER:
        entries = py_waves_raw.get(wave, [])
        if not isinstance(entries, list):
            raise ValueError(f"python_waves_1_to_5.json[{wave!r}] not a list")
        for row in entries:
            if not isinstance(row, dict):
                raise ValueError("python record must be an object")
            workload = str(row["workload"])
            nz = int(row["n_zones"])
            python[(workload, nz)] = RawBenchRecord(
                wave=wave,
                workload=workload,
                n_zones=nz,
                stage0=_parse_stage_dict(row["stage0"]),
                stage1=_parse_stage_dict(row["stage1"]),
                stage2=_parse_stage_dict(row["stage2"]),
                stage3=_parse_stage_dict(row["stage3"]),
                max_rel_err=float(row["max_rel_err"]),
                checksum=float(row["checksum"]),
            )

    return RawBenchmarkSuite(
        provenance=prov,
        python_metadata=py_meta_raw,
        fortran=fortran,
        rust=rust,
        python=python,
        fortran_bcyclic=bcyclic,
        comparator_kind=comp_map,
    )
