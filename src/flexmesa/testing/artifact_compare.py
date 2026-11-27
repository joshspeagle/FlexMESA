# SPDX-License-Identifier: GPL-3.0-or-later
"""Artifact comparison for cases that take zero evolution steps (`#606`).

Some upstream test cases never start the evolution loop. Their `MESA`
`26.4.1` run ends inside a hook (for example `other_rsp_build_model` of
`star/rsp_check_2nd_crossing`) after writing a result file, so there is no
`history.data` to compare. For such a case the manifest declares an
``[cases.artifact]`` table: the file `MESA` wrote, a committed reference copy
of it, a parser kind, and one tolerance per field group. This module parses
both files, compares every declared field, and reports the first field out of
tolerance and the field furthest out (in tolerance units).

Tolerances are the print precision of the upstream format statement, not
fitted values. A tolerance is either ``{abs = x}`` or ``{sig_digits = n}``:
the latter allows one unit in the last printed significant digit of the
reference value (for `1pd20.10`, ``n = 11``; for `E-format` with five
digits, ``n = 5``). Integers (model numbers, offsets) must be equal.

Only one parser kind exists, ``rsp_instability_strip``, for the output of
`star/test_suite/rsp_check_2nd_crossing/src/run_star_extras.f90:135-348`.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import re
from typing import Any

PARSER_RSP_INSTABILITY_STRIP = "rsp_instability_strip"
VALID_PARSERS: tuple[str, ...] = (PARSER_RSP_INSTABILITY_STRIP,)

# Field groups of the ``rsp_instability_strip`` parser, in comparison order.
STRIP_GROUPS: tuple[str, ...] = (
    "header",
    "model_period",
    "model_growth",
    "model_teff",
    "model_lum",
    "model_age",
    "edge",
    "table_teff",
    "table_lum",
)
# Groups compared only when per-model lines of standard output are supplied.
STRIP_STDOUT_GROUPS: tuple[str, ...] = (
    "stdout_teff",
    "stdout_lum",
    "stdout_period",
    "stdout_growth",
)

_FLOAT_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([DdEe][+-]?\d+)?$")
_STDOUT_MODEL_RE = re.compile(
    r"call star_do1_rsp_build model Teff L\s+(\d+)\s+(\S+)\s+(\S+)\s*$"
)


@dataclasses.dataclass(frozen=True)
class ArtifactSpec:
    """Manifest declaration of an artifact comparison.

    Attributes:
        file: Name of the file `MESA` writes (and `FlexMESA`'s hook writes
            next to the case inputs).
        reference: Repository-relative path of the committed `MESA` copy.
        parser: Parser kind (see :data:`VALID_PARSERS`).
        tolerances: Mapping from field group to ``{"abs": x}`` or
            ``{"sig_digits": n}``.
        required_steps: Evolution steps both runs take (``0`` for the cases
            this exists for); a different count fails the case.
        stdout_reference: Optional repository-relative path of the per-model
            lines extracted from `MESA`'s terminal output.
    """

    file: str
    reference: str
    parser: str
    tolerances: dict[str, dict[str, float]]
    required_steps: int = 0
    stdout_reference: str = ""


@dataclasses.dataclass(frozen=True)
class StripScan:
    """Parsed instability-strip scan file.

    Attributes:
        header: Model parameters (``RSP_mass``, ...), by label.
        models: One ``(model_number, period_d, growth, Teff, L, age)`` row
            per scanned model; ``period_d`` in days, ``Teff`` in K, ``L`` in
            ``Lsun``, ``age`` in yr.
        edges: ``Teff_red_edge`` and ``Teff_blue_edge`` in K, by label.
        table: ``(offset, Teff, L)`` rows (K, K, ``Lsun``).
    """

    header: dict[str, float]
    models: list[tuple[int, float, float, float, float, float]]
    edges: dict[str, float]
    table: list[tuple[int, float, float]]


@dataclasses.dataclass(frozen=True)
class StdoutModel:
    """One per-model block of `MESA`'s terminal output.

    Attributes:
        model_number: Model number of the track point.
        teff: Effective temperature (K) printed with list-directed output.
        lum: Luminosity (``Lsun``) printed with list-directed output.
        period_d: Period (days), five significant digits.
        growth: Growth rate (per period), five significant digits.
    """

    model_number: int
    teff: float
    lum: float
    period_d: float
    growth: float


@dataclasses.dataclass(frozen=True)
class FieldDiff:
    """One compared field.

    Attributes:
        label: Field name, such as ``model_growth[3]`` or ``edge[Teff_red]``.
        reference: Value in the `MESA` reference.
        run: Value in the `FlexMESA` file (``nan`` when absent).
        tolerance: Allowed absolute difference.
        excess: ``|run - reference| / tolerance`` (``inf`` for a nonzero
            difference against a zero tolerance, ``0`` for equal values).
    """

    label: str
    reference: float
    run: float
    tolerance: float
    excess: float

    @property
    def out_of_tolerance(self) -> bool:
        """Return whether the field is outside its tolerance."""
        return not self.excess <= 1.0


@dataclasses.dataclass(frozen=True)
class ArtifactComparison:
    """Result of comparing one run artifact with its `MESA` reference.

    Attributes:
        status: ``"ok"`` (every field within tolerance), ``"differs"``,
            ``"missing"`` (no run file), or ``"unparsable"``.
        n_fields: Number of fields compared.
        n_out: Number of fields outside tolerance (or absent).
        first_failure: First field out of tolerance, in comparison order.
        largest: Field with the largest ``excess`` (``None`` when empty).
        stdout_checked: Whether per-model standard-output lines were compared.
        message: One-line description for the case table.
    """

    status: str
    n_fields: int = 0
    n_out: int = 0
    first_failure: FieldDiff | None = None
    largest: FieldDiff | None = None
    stdout_checked: bool = False
    message: str = ""

    @property
    def ok(self) -> bool:
        """Return whether the artifact agrees within every tolerance."""
        return self.status == "ok"


def parse_spec(raw: dict[str, Any], ctx: str) -> ArtifactSpec:
    """Validate a ``[cases.artifact]`` TOML table.

    Args:
        raw: Parsed TOML table.
        ctx: ``suite/case`` label for error messages.

    Returns:
        The validated :class:`ArtifactSpec`.

    Raises:
        ValueError: On a missing key, unknown parser, absent tolerance group,
            or malformed tolerance.
    """
    for key in ("file", "reference", "parser"):
        if not str(raw.get(key, "")).strip():
            raise ValueError(f"{ctx}: artifact.{key} is required")
    parser = str(raw["parser"]).strip()
    if parser not in VALID_PARSERS:
        raise ValueError(f"{ctx}: unknown artifact parser {parser!r}")
    stdout_ref = str(raw.get("stdout_reference", "")).strip()
    tols_raw = raw.get("tolerances", {})
    tols: dict[str, dict[str, float]] = {}
    for group, spec in tols_raw.items():
        if not isinstance(spec, dict) or set(spec) not in (
            {"abs"},
            {"sig_digits"},
        ):
            raise ValueError(
                f"{ctx}: artifact.tolerances.{group} must be {{abs = x}} or "
                "{sig_digits = n}"
            )
        value = float(next(iter(spec.values())))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{ctx}: artifact.tolerances.{group} is invalid")
        tols[str(group)] = {k: float(v) for k, v in spec.items()}
    needed = STRIP_GROUPS + (STRIP_STDOUT_GROUPS if stdout_ref else ())
    unknown = set(tols) - set(STRIP_GROUPS) - set(STRIP_STDOUT_GROUPS)
    absent = [g for g in needed if g not in tols]
    if unknown or absent:
        raise ValueError(
            f"{ctx}: artifact.tolerances unknown {sorted(unknown)} "
            f"absent {absent}"
        )
    steps = int(raw.get("required_steps", 0))
    if steps < 0:
        raise ValueError(f"{ctx}: artifact.required_steps must be >= 0")
    return ArtifactSpec(
        file=str(raw["file"]).strip(),
        reference=str(raw["reference"]).strip(),
        parser=parser,
        tolerances=tols,
        required_steps=steps,
        stdout_reference=stdout_ref,
    )


def _real(tok: str, line: str) -> float:
    if not _FLOAT_RE.match(tok):
        raise ValueError(f"not a real number {tok!r} in line {line!r}")
    return float(tok.replace("D", "E").replace("d", "e"))


def parse_instability_strip(text: str) -> StripScan:
    """Parse the file written by `rsp_check_2nd_crossing`.

    Args:
        text: File contents.

    Returns:
        The parsed :class:`StripScan`.

    Raises:
        ValueError: On a line the format does not allow.
    """
    header: dict[str, float] = {}
    models: list[tuple[int, float, float, float, float, float]] = []
    edges: dict[str, float] = {}
    table: list[tuple[int, float, float]] = []
    state = "header"
    for line in text.splitlines():
        toks = line.split()
        if not toks:
            continue
        if toks[0] == "model_number":
            state = "models"
        elif toks[0] == "offset":
            state = "table"
        elif toks[0] in ("Teff_red_edge", "Teff_blue_edge"):
            edges[toks[0]] = _real(toks[1], line)
        elif state == "header" and len(toks) == 2:
            header[toks[0]] = _real(toks[1], line)
        elif state == "models" and len(toks) == 6:
            vals = [_real(t, line) for t in toks[1:]]
            models.append(
                (int(toks[0]), vals[0], vals[1], vals[2], vals[3], vals[4])
            )
        elif state == "table" and len(toks) == 3:
            table.append(
                (int(toks[0]), _real(toks[1], line), _real(toks[2], line))
            )
        else:
            raise ValueError(f"unexpected line in {state} section: {line!r}")
    return StripScan(header, models, edges, table)


def parse_stdout_models(text: str) -> list[StdoutModel]:
    """Parse the per-model blocks of `MESA`'s terminal output.

    Each block is the line ``call star_do1_rsp_build model Teff L <n> <Teff>
    <L>``, a ``P(days) growth`` header, and one line ``<mode> <P> <growth>``.

    Args:
        text: Terminal output (or the extract kept as the reference).

    Returns:
        One :class:`StdoutModel` per block, in order.

    Raises:
        ValueError: If a block is incomplete or malformed.
    """
    lines = text.splitlines()
    out: list[StdoutModel] = []
    i = 0
    while i < len(lines):
        m = _STDOUT_MODEL_RE.search(lines[i])
        if m is None:
            i += 1
            continue
        if i + 2 >= len(lines) or "P(days)" not in lines[i + 1]:
            raise ValueError(f"incomplete per-model block at line {i + 1}")
        vals = lines[i + 2].split()
        if len(vals) != 3:
            raise ValueError(f"malformed period line {lines[i + 2]!r}")
        out.append(
            StdoutModel(
                int(m.group(1)),
                _real(m.group(2), lines[i]),
                _real(m.group(3), lines[i]),
                _real(vals[1], lines[i + 2]),
                _real(vals[2], lines[i + 2]),
            )
        )
        i += 3
    return out


def _tolerance(tol: dict[str, float], reference: float) -> float:
    if "abs" in tol:
        return tol["abs"]
    if reference == 0.0:
        return 0.0
    digits = int(tol["sig_digits"])
    return 10.0 ** (math.floor(math.log10(abs(reference))) - (digits - 1))


def _diff(
    label: str, ref: float, run: float, tol: dict[str, float]
) -> FieldDiff:
    allowed = _tolerance(tol, ref)
    delta = abs(run - ref)
    if math.isnan(run) or math.isnan(delta):
        excess = math.inf
    elif delta == 0.0:
        excess = 0.0
    elif allowed == 0.0:
        excess = math.inf
    else:
        excess = delta / allowed
    return FieldDiff(label, ref, run, allowed, excess)


def _missing(label: str, ref: float) -> FieldDiff:
    return FieldDiff(label, ref, math.nan, 0.0, math.inf)


def _strip_fields(
    ref: StripScan, run: StripScan, tols: dict[str, dict[str, float]]
) -> tuple[list[FieldDiff], list[str]]:
    """Compare two scans; return field diffs and structural mismatches."""
    diffs: list[FieldDiff] = []
    notes: list[str] = []
    for key, rval in ref.header.items():
        if key in run.header:
            diffs.append(
                _diff(f"header[{key}]", rval, run.header[key], tols["header"])
            )
        else:
            diffs.append(_missing(f"header[{key}]", rval))
    if len(ref.models) != len(run.models):
        notes.append(f"models {len(run.models)} vs {len(ref.models)}")
    cols = (
        "model_period",
        "model_growth",
        "model_teff",
        "model_lum",
        "model_age",
    )
    for idx, rrow in enumerate(ref.models):
        if idx >= len(run.models):
            diffs.append(_missing(f"model[{idx}]", float(rrow[0])))
            continue
        crow = run.models[idx]
        diffs.append(
            _diff(
                f"model_number[{idx}]",
                float(rrow[0]),
                float(crow[0]),
                {"abs": 0.0},
            )
        )
        for k, group in enumerate(cols, start=1):
            diffs.append(
                _diff(f"{group}[{idx}]", rrow[k], crow[k], tols[group])
            )
    for key, rval in ref.edges.items():
        if key in run.edges:
            diffs.append(
                _diff(f"edge[{key}]", rval, run.edges[key], tols["edge"])
            )
        else:
            diffs.append(_missing(f"edge[{key}]", rval))
    if len(ref.table) != len(run.table):
        notes.append(f"table rows {len(run.table)} vs {len(ref.table)}")
    for idx, trow in enumerate(ref.table):
        if idx >= len(run.table):
            diffs.append(_missing(f"table[{idx}]", float(trow[0])))
            continue
        crow2 = run.table[idx]
        diffs.append(
            _diff(
                f"table_offset[{idx}]",
                float(trow[0]),
                float(crow2[0]),
                {"abs": 0.0},
            )
        )
        diffs.append(
            _diff(f"table_teff[{idx}]", trow[1], crow2[1], tols["table_teff"])
        )
        diffs.append(
            _diff(f"table_lum[{idx}]", trow[2], crow2[2], tols["table_lum"])
        )
    return diffs, notes


def _stdout_fields(
    ref: list[StdoutModel],
    run: list[StdoutModel],
    tols: dict[str, dict[str, float]],
) -> tuple[list[FieldDiff], list[str]]:
    diffs: list[FieldDiff] = []
    notes: list[str] = []
    if len(ref) != len(run):
        notes.append(f"stdout models {len(run)} vs {len(ref)}")
    for idx, r in enumerate(ref):
        if idx >= len(run):
            diffs.append(
                _missing(f"stdout_model[{idx}]", float(r.model_number))
            )
            continue
        c = run[idx]
        diffs.append(
            _diff(
                f"stdout_number[{idx}]",
                float(r.model_number),
                float(c.model_number),
                {"abs": 0.0},
            )
        )
        for name, rv, cv in (
            ("stdout_teff", r.teff, c.teff),
            ("stdout_lum", r.lum, c.lum),
            ("stdout_period", r.period_d, c.period_d),
            ("stdout_growth", r.growth, c.growth),
        ):
            diffs.append(_diff(f"{name}[{idx}]", rv, cv, tols[name]))
    return diffs, notes


def compare_artifact(
    spec: ArtifactSpec,
    reference_text: str,
    run_text: str | None,
    *,
    reference_stdout: str | None = None,
    run_stdout: str | None = None,
) -> ArtifactComparison:
    """Compare a run's artifact with the committed `MESA` reference.

    Args:
        spec: Manifest declaration.
        reference_text: Contents of the committed `MESA` file.
        run_text: Contents of the file `FlexMESA` wrote, or ``None`` when it
            wrote none.
        reference_stdout: Committed per-model terminal lines, when declared.
        run_stdout: Captured per-model terminal lines of the run. Without
            them the stdout groups are not compared and the result says so.

    Returns:
        The :class:`ArtifactComparison`.
    """
    if run_text is None:
        return ArtifactComparison(
            "missing", message=f"artifact {spec.file} not written by the run"
        )
    try:
        ref = parse_instability_strip(reference_text)
        run = parse_instability_strip(run_text)
    except ValueError as err:
        return ArtifactComparison(
            "unparsable", message=f"artifact {spec.file}: {err}"
        )
    diffs, notes = _strip_fields(ref, run, spec.tolerances)
    stdout_checked = False
    if reference_stdout is not None and run_stdout is not None:
        try:
            s_diffs, s_notes = _stdout_fields(
                parse_stdout_models(reference_stdout),
                parse_stdout_models(run_stdout),
                spec.tolerances,
            )
        except ValueError as err:
            return ArtifactComparison("unparsable", message=f"stdout: {err}")
        diffs += s_diffs
        notes += s_notes
        stdout_checked = True
    bad = [d for d in diffs if d.out_of_tolerance]
    first = bad[0] if bad else None
    largest = max(diffs, key=lambda d: d.excess, default=None)
    if not bad and not notes:
        return ArtifactComparison(
            "ok",
            n_fields=len(diffs),
            largest=largest,
            stdout_checked=stdout_checked,
            message=(
                f"artifact {spec.file}: {len(diffs)} fields within tolerance"
            ),
        )
    parts = list(notes)
    if first is not None:
        parts.append(
            f"first out of tolerance {first.label} run {first.run!r} vs "
            f"MESA {first.reference!r} (tol {first.tolerance:.3g}); "
            f"{len(bad)} of {len(diffs)} fields out"
        )
    return ArtifactComparison(
        "differs",
        n_fields=len(diffs),
        n_out=len(bad),
        first_failure=first,
        largest=largest,
        stdout_checked=stdout_checked,
        message=f"artifact {spec.file}: " + "; ".join(parts),
    )


def compare_files(
    spec: ArtifactSpec,
    root_dir: pathlib.Path,
    run_dirs: list[pathlib.Path],
    *,
    run_stdout: str | None = None,
) -> ArtifactComparison:
    """Locate the run's artifact in ``run_dirs`` and compare it.

    Args:
        spec: Manifest declaration.
        root_dir: Directory against which the reference paths resolve.
        run_dirs: Directories to search, in order, for ``spec.file``.
        run_stdout: Captured per-model terminal lines, if any.

    Returns:
        The :class:`ArtifactComparison`.

    Raises:
        FileNotFoundError: If a committed reference file is absent.
    """
    reference_text = (root_dir / spec.reference).read_text(encoding="utf-8")
    ref_stdout = (
        (root_dir / spec.stdout_reference).read_text(encoding="utf-8")
        if spec.stdout_reference
        else None
    )
    run_text: str | None = None
    for d in run_dirs:
        cand = d / spec.file
        if cand.is_file():
            run_text = cand.read_text(encoding="utf-8")
            break
    return compare_artifact(
        spec,
        reference_text,
        run_text,
        reference_stdout=ref_stdout,
        run_stdout=run_stdout,
    )


def _field_to_dict(f: FieldDiff | None) -> dict[str, Any] | None:
    if f is None:
        return None
    return {
        "label": f.label,
        "reference": f.reference,
        "run": None if math.isnan(f.run) else f.run,
        "tolerance": f.tolerance,
        "excess": None if math.isinf(f.excess) else f.excess,
    }


def _field_from_dict(d: dict[str, Any] | None) -> FieldDiff | None:
    if d is None:
        return None
    return FieldDiff(
        str(d["label"]),
        float(d["reference"]),
        math.nan if d["run"] is None else float(d["run"]),
        float(d["tolerance"]),
        math.inf if d["excess"] is None else float(d["excess"]),
    )


def comparison_to_dict(c: ArtifactComparison) -> dict[str, Any]:
    """Convert a comparison to a JSON-serializable dictionary."""
    return {
        "status": c.status,
        "n_fields": c.n_fields,
        "n_out": c.n_out,
        "first_failure": _field_to_dict(c.first_failure),
        "largest": _field_to_dict(c.largest),
        "stdout_checked": c.stdout_checked,
        "message": c.message,
    }


def comparison_from_dict(d: dict[str, Any]) -> ArtifactComparison:
    """Rebuild a comparison from :func:`comparison_to_dict` output."""
    return ArtifactComparison(
        status=str(d["status"]),
        n_fields=int(d.get("n_fields", 0)),
        n_out=int(d.get("n_out", 0)),
        first_failure=_field_from_dict(d.get("first_failure")),
        largest=_field_from_dict(d.get("largest")),
        stdout_checked=bool(d.get("stdout_checked", False)),
        message=str(d.get("message", "")),
    )
