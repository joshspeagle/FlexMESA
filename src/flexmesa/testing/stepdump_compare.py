# SPDX-License-Identifier: GPL-3.0-or-later
"""First-divergence comparison of MESA and FlexMESA step dumps (#372).

Both sides write the record layout of
``benchmarks/fortran_mesa/stepdumps/stepdump_patch.py``: per star, the
solver records of the first few models (``CALL``, ``CORR``, ``NEWTON``,
``FOLD``, ``TRY``, ``RESID``) and, at the first solver call of each model,
the per-zone state (``STATE``), cell mass (``DM``) and residuals (``EQU``).
Values are IEEE-754 binary64 bit patterns in hexadecimal, so equality here
is bitwise.

Models are aligned per star by order of appearance, not by model number, so
that an offset in model numbering (#359) does not hide every later
difference; both model numbers are reported. Within a model the solver
records are compared in order. The first record whose type differs marks a
change of control flow (for example a retry on one side only); the first
record whose values differ marks a numerical divergence.
"""

from __future__ import annotations

import dataclasses
import pathlib
import struct

SOLVER_TAGS = ("CALL", "CORR", "NEWTON", "FOLD", "TRY", "RESID")
ZONE_TAGS = ("STATE", "DM", "AUX", "EQU")
# Number of leading integer fields after ``tag id model`` per record type,
# and number of trailing integer fields after the hex values.
_LAYOUT: dict[str, tuple[int, int]] = {
    "CALL": (0, 2),
    "CORR": (1, 2),
    "NEWTON": (1, 0),
    "FOLD": (1, 0),
    "TRY": (2, 0),
    "RESID": (1, 2),
    "STATE": (1, 0),
    "DM": (1, 0),
    "AUX": (1, 0),
    "EQU": (1, 0),
}
# Every record type the parser reads; any other type is skipped and counted.
_KNOWN_TAGS = frozenset(_LAYOUT) | {"NAMES", "EQNAMES"}
AUX_COLUMNS = (
    "Peos",
    "opacity",
    "gradT",
    "gradr",
    "grada",
    "eps_nuc",
    "mlt_vc",
    "chiRho",
    "chiT",
    "Cp",
)
_FIELDS: dict[str, tuple[str, ...]] = {
    "CALL": ("dt", "residual_norm", "max_residual"),
    "CORR": ("correction_norm", "max_correction"),
    "NEWTON": ("min_corr_coeff", "correction_factor", "slope"),
    "FOLD": ("fold",),
    "TRY": ("alam", "coeff", "f", "f_target"),
    "RESID": ("coeff", "residual_norm", "max_residual"),
}


@dataclasses.dataclass(frozen=True)
class Record:
    """One parsed record.

    Attributes:
        tag: Record type.
        star: Star id (MESA ``s% id``).
        model: Model number as written.
        ints: Leading integer fields (iteration counters or zone).
        bits: IEEE-754 bit patterns of the values.
        tail: Trailing integer fields (argmax zone and index, 1-based).
    """

    tag: str
    star: int
    model: int
    ints: tuple[int, ...]
    bits: tuple[int, ...]
    tail: tuple[int, ...]


@dataclasses.dataclass
class ModelBlock:
    """Records of one model of one star, in file order.

    Attributes:
        model: Model number as written.
        solver: Solver records in order.
        zones: ``(tag, zone) -> bits`` for ``STATE``, ``DM`` and ``EQU``.
    """

    model: int
    solver: list[Record] = dataclasses.field(default_factory=list)
    zones: dict[tuple[str, int], tuple[int, ...]] = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass
class StepDump:
    """A parsed step dump.

    Attributes:
        header: ``#`` lines without the leading ``# ``.
        names: Star id -> solver variable names (hydro then species).
        eqnames: Star id -> equation names.
        nvar_hydro: Star id -> hydro variables per zone.
        blocks: Star id -> model blocks in order of appearance.
        skipped: Record type -> count of lines of a type this parser does not
            know (diagnostic records added by a patched solver, for
            example). They are left out of every comparison.
    """

    header: list[str] = dataclasses.field(default_factory=list)
    names: dict[int, list[str]] = dataclasses.field(default_factory=dict)
    eqnames: dict[int, list[str]] = dataclasses.field(default_factory=dict)
    nvar_hydro: dict[int, int] = dataclasses.field(default_factory=dict)
    blocks: dict[int, list[ModelBlock]] = dataclasses.field(
        default_factory=dict
    )
    skipped: dict[str, int] = dataclasses.field(default_factory=dict)


def bits_to_float(bits: int) -> float:
    """Returns the binary64 value of a 64-bit pattern."""
    return float(struct.unpack("<d", struct.pack("<Q", bits))[0])


def ulp_distance(a: int, b: int) -> int:
    """Returns the number of representable doubles between two patterns.

    Args:
        a: First bit pattern.
        b: Second bit pattern.

    Returns:
        The distance on the ordered integer line of IEEE-754 doubles.
    """

    def ordered(x: int) -> int:
        return x if x < (1 << 63) else (1 << 63) - x

    return abs(ordered(a) - ordered(b))


def parse_record(line: str) -> Record:
    """Parses one record line.

    Args:
        line: A solver or zone record.

    Returns:
        The record.

    Raises:
        ValueError: If the line does not match its record layout.
    """
    parts = line.split()
    tag = parts[0]
    n_lead, n_tail = _LAYOUT[tag]
    star, model = int(parts[1]), int(parts[2])
    rest = parts[3:]
    if len(rest) < n_lead + n_tail:
        raise ValueError(f"short {tag} record: {line!r}")
    ints = tuple(int(x) for x in rest[:n_lead])
    vals = rest[n_lead : len(rest) - n_tail]
    tail = tuple(int(x) for x in rest[len(rest) - n_tail :])
    return Record(tag, star, model, ints, tuple(int(v, 16) for v in vals), tail)


def parse_dump(text: str) -> StepDump:
    """Parses a step dump.

    Args:
        text: Contents of a ``.stepdump`` file.

    Returns:
        The parsed dump.
    """
    dump = StepDump()
    seen_records = False
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            dump.header.append(line.lstrip("# "))
            continue
        tag = line.split(maxsplit=1)[0]
        if tag == "RUN":
            # Each ./rn process appends its own RUN segment; only the first
            # segment with records is compared, because the FlexMESA dump
            # counts models over the whole trace process (#372).
            if seen_records:
                break
            continue
        if tag == "DONE":
            continue
        if tag not in _KNOWN_TAGS:
            dump.skipped[tag] = dump.skipped.get(tag, 0) + 1
            continue
        seen_records = True
        if tag == "NAMES":
            p = line.split()
            star = int(p[1])
            dump.nvar_hydro[star] = int(p[3])
            dump.names[star] = p[6:]
            continue
        if tag == "EQNAMES":
            p = line.split()
            dump.eqnames[int(p[1])] = p[3:]
            continue
        rec = parse_record(line)
        blocks = dump.blocks.setdefault(rec.star, [])
        if not blocks or blocks[-1].model != rec.model:
            blocks.append(ModelBlock(rec.model))
        block = blocks[-1]
        if tag in ZONE_TAGS:
            block.zones[(tag, rec.ints[0])] = rec.bits
        else:
            block.solver.append(rec)
    return dump


@dataclasses.dataclass(frozen=True)
class Divergence:
    """The first difference between two dumps.

    Attributes:
        star: Star id.
        model_index: 0-based position of the model in the star's sequence.
        mesa_model: MESA's model number.
        flex_model: FlexMESA's model number.
        kind: ``"control"`` (record types differ), ``"value"``,
            ``"location"`` (argmax zone or index differs), ``"zone"``
            (``STATE``/``EQU`` differ) or ``"missing"`` (one side ends).
        record: Record type, or ``STATE``/``EQU``.
        position: 0-based position among the model's solver records.
        zone: 1-based zone for zone records, else ``None``.
        field: Name of the differing field or column.
        mesa: MESA's value (or record type for ``"control"``).
        flex: FlexMESA's value.
        ulps: ULP distance for a value difference, else ``None``.
    """

    star: int
    model_index: int
    mesa_model: int
    flex_model: int
    kind: str
    record: str
    position: int | None
    zone: int | None
    field: str
    mesa: str
    flex: str
    ulps: int | None


@dataclasses.dataclass(frozen=True)
class ColumnSummary:
    """Per-column difference count of one zone record type in one model.

    Attributes:
        record: ``STATE`` or ``EQU``.
        column: Column name.
        n_diff: Zones whose bits differ.
        first_zone: First differing zone (1-based).
        max_rel: Largest relative difference.
    """

    record: str
    column: str
    n_diff: int
    first_zone: int
    max_rel: float


def _fmt(bits: int) -> str:
    return f"{bits_to_float(bits):.17e} ({bits:016X})"


def _column_names(dump: StepDump, star: int, tag: str, n: int) -> list[str]:
    if tag == "DM":
        return ["dm"]
    if tag == "AUX":
        return list(AUX_COLUMNS)
    names = dump.names if tag == "STATE" else dump.eqnames
    # STATE holds xh(1:nvar_hydro) then xa(1:species), the order of NAMES.
    cols = names.get(star, [])
    return [cols[i] if i < len(cols) else f"col{i + 1}" for i in range(n)]


def compare_zones(
    mesa: StepDump, flex: StepDump, star: int, index: int
) -> list[ColumnSummary]:
    """Summarises ``STATE`` and ``EQU`` differences of one model.

    Args:
        mesa: MESA dump.
        flex: FlexMESA dump.
        star: Star id.
        index: 0-based model position.

    Returns:
        One summary per column with at least one differing zone.
    """
    a, b = mesa.blocks[star][index], flex.blocks[star][index]
    out = []
    for tag in ZONE_TAGS:
        zones = sorted(k for (t, k) in a.zones if t == tag)
        if not zones:
            continue
        ncol = len(a.zones[(tag, zones[0])])
        names = _column_names(mesa, star, tag, ncol)
        for j in range(ncol):
            n_diff, first, max_rel = 0, 0, 0.0
            for k in zones:
                vb = b.zones.get((tag, k))
                if vb is None or j >= len(vb):
                    continue
                xa_, xb_ = a.zones[(tag, k)][j], vb[j]
                if xa_ == xb_:
                    continue
                n_diff += 1
                first = first or k
                fa, fb = bits_to_float(xa_), bits_to_float(xb_)
                rel = abs(fa - fb) / max(abs(fa), 1e-300)
                max_rel = max(max_rel, rel)
            if n_diff:
                out.append(ColumnSummary(tag, names[j], n_diff, first, max_rel))
    return out


def first_divergence(
    mesa: StepDump, flex: StepDump, ignore: frozenset[str] = frozenset()
) -> Divergence | None:
    """Returns the first difference between two dumps, or ``None``.

    Within each model, the zone records (the state at the first solver call)
    are checked before the solver records, since they come first in the
    step.

    Args:
        mesa: MESA dump.
        flex: FlexMESA dump.
        ignore: Record types (``EQU``, ``DM``, ``TRY``, ...), column names
            (``h1``, ``lnd``, ...) or ``xa`` (every species column) left out
            of the comparison, so that a known difference does not hide the
            next one.

    Returns:
        The first divergence in star, model, then record order.
    """
    for star in sorted(set(flex.blocks) - set(mesa.blocks)):
        b = flex.blocks[star][0]
        return Divergence(
            star,
            0,
            -1,
            b.model,
            "missing",
            "star",
            None,
            None,
            "",
            "none",
            f"star {star}",
            None,
        )
    for star in sorted(mesa.blocks):
        a_blocks = mesa.blocks[star]
        b_blocks = flex.blocks.get(star, [])
        for i, a in enumerate(a_blocks):
            if i >= len(b_blocks):
                return Divergence(
                    star,
                    i,
                    a.model,
                    -1,
                    "missing",
                    "model",
                    None,
                    None,
                    "",
                    str(a.model),
                    "none",
                    None,
                )
            b = b_blocks[i]
            div = _first_zone_divergence(mesa, star, i, a, b, ignore)
            if div is not None:
                return div
            div = _first_solver_divergence(star, i, a, b, ignore)
            if div is not None:
                return div
        if len(b_blocks) > len(a_blocks):
            b = b_blocks[len(a_blocks)]
            return Divergence(
                star,
                len(a_blocks),
                -1,
                b.model,
                "missing",
                "model",
                None,
                None,
                "",
                "none",
                str(b.model),
                None,
            )
    return None


def _skip_column(
    mesa: StepDump,
    star: int,
    tag: str,
    j: int,
    name: str,
    ignore: frozenset[str],
) -> bool:
    """Returns whether column ``j`` of a zone record is ignored."""
    if name in ignore:
        return True
    # 'xa' covers every species column: the STATE mass fractions and the
    # EQU species residuals.
    return (
        tag in ("STATE", "EQU")
        and "xa" in ignore
        and j >= mesa.nvar_hydro.get(star, 0)
    )


def _first_zone_divergence(
    mesa: StepDump,
    star: int,
    i: int,
    a: ModelBlock,
    b: ModelBlock,
    ignore: frozenset[str],
) -> Divergence | None:
    for tag in ZONE_TAGS:
        if tag in ignore:
            continue
        for k in sorted(z for (t, z) in a.zones if t == tag):
            va = a.zones[(tag, k)]
            vb = b.zones.get((tag, k))
            if vb is None:
                return Divergence(
                    star,
                    i,
                    a.model,
                    b.model,
                    "missing",
                    tag,
                    None,
                    k,
                    "",
                    "present",
                    "absent",
                    None,
                )
            names = _column_names(mesa, star, tag, len(va))
            for j, (xa_, xb_) in enumerate(zip(va, vb, strict=False)):
                if _skip_column(mesa, star, tag, j, names[j], ignore):
                    continue
                if xa_ != xb_:
                    return Divergence(
                        star,
                        i,
                        a.model,
                        b.model,
                        "zone",
                        tag,
                        None,
                        k,
                        names[j],
                        _fmt(xa_),
                        _fmt(xb_),
                        ulp_distance(xa_, xb_),
                    )
            if len(va) != len(vb):
                return Divergence(
                    star,
                    i,
                    a.model,
                    b.model,
                    "zone",
                    tag,
                    None,
                    k,
                    "column count",
                    str(len(va)),
                    str(len(vb)),
                    None,
                )
    return None


def _first_solver_divergence(
    star: int,
    i: int,
    a_block: ModelBlock,
    b_block: ModelBlock,
    ignore: frozenset[str],
) -> Divergence | None:
    a = ModelBlock(
        a_block.model, [r for r in a_block.solver if r.tag not in ignore]
    )
    b = ModelBlock(
        b_block.model, [r for r in b_block.solver if r.tag not in ignore]
    )
    for pos, ra in enumerate(a.solver):
        if pos >= len(b.solver):
            return Divergence(
                star,
                i,
                a.model,
                b.model,
                "missing",
                ra.tag,
                pos,
                None,
                "",
                ra.tag,
                "none",
                None,
            )
        rb = b.solver[pos]
        if ra.tag != rb.tag or ra.ints != rb.ints:
            return Divergence(
                star,
                i,
                a.model,
                b.model,
                "control",
                ra.tag,
                pos,
                None,
                "record",
                f"{ra.tag} {ra.ints}",
                f"{rb.tag} {rb.ints}",
                None,
            )
        names = _FIELDS[ra.tag]
        for j, (xa_, xb_) in enumerate(zip(ra.bits, rb.bits, strict=True)):
            if xa_ != xb_:
                return Divergence(
                    star,
                    i,
                    a.model,
                    b.model,
                    "value",
                    ra.tag,
                    pos,
                    None,
                    names[j],
                    _fmt(xa_),
                    _fmt(xb_),
                    ulp_distance(xa_, xb_),
                )
        if ra.tail != rb.tail:
            return Divergence(
                star,
                i,
                a.model,
                b.model,
                "location",
                ra.tag,
                pos,
                None,
                "argmax (zone, index)",
                str(ra.tail),
                str(rb.tail),
                None,
            )
    if len(b.solver) > len(a.solver):
        rb = b.solver[len(a.solver)]
        return Divergence(
            star,
            i,
            a.model,
            b.model,
            "missing",
            rb.tag,
            len(a.solver),
            None,
            "",
            "none",
            rb.tag,
            None,
        )
    return None


def format_report(
    mesa: StepDump, flex: StepDump, div: Divergence | None
) -> str:
    """Returns a human-readable report of a comparison.

    Args:
        mesa: MESA dump.
        flex: FlexMESA dump.
        div: Result of :func:`first_divergence`.

    Returns:
        The report text.
    """
    lines = [f"MESA dump: {h}" for h in mesa.header if h.startswith("case")]
    for who, dump in (("MESA", mesa), ("FlexMESA", flex)):
        if dump.skipped:
            kinds = ", ".join(
                f"{k} x{n}" for k, n in sorted(dump.skipped.items())
            )
            lines.append(f"{who} dump: skipped unknown record types: {kinds}")
    if div is None:
        n = sum(len(v) for v in mesa.blocks.values())
        lines.append(f"no difference in {n} model(s): bit-identical records")
        return "\n".join(lines) + "\n"
    lines.append(
        f"first divergence: star {div.star}, model #{div.model_index + 1} "
        f"(MESA model {div.mesa_model}, FlexMESA model {div.flex_model}), "
        f"{div.kind} in {div.record}"
        + (f" record {div.position}" if div.position is not None else "")
        + (f" zone {div.zone}" if div.zone is not None else "")
    )
    lines.append(f"  field: {div.field}")
    lines.append(f"  MESA:     {div.mesa}")
    lines.append(f"  FlexMESA: {div.flex}")
    if div.ulps is not None:
        lines.append(f"  ULP distance: {div.ulps}")
    if div.model_index < len(flex.blocks.get(div.star, [])):
        summary = compare_zones(mesa, flex, div.star, div.model_index)
        if summary:
            lines.append("  per-column differences at this model's first call:")
            for s in summary:
                lines.append(
                    f"    {s.record:5s} {s.column:12s} {s.n_diff:6d} zones "
                    f"(first {s.first_zone}), max rel {s.max_rel:.3e}"
                )
    return "\n".join(lines) + "\n"


def compare_files(
    mesa_path: pathlib.Path,
    flex_path: pathlib.Path,
    ignore: frozenset[str] = frozenset(),
) -> tuple[str, Divergence | None]:
    """Compares two dump files.

    Args:
        mesa_path: MESA ``.stepdump`` file.
        flex_path: FlexMESA dump file.
        ignore: See :func:`first_divergence`.

    Returns:
        The report text and the first divergence (``None`` when the records
        are bit-identical).
    """
    mesa = parse_dump(mesa_path.read_text(encoding="utf-8"))
    flex = parse_dump(flex_path.read_text(encoding="utf-8"))
    div = first_divergence(mesa, flex, ignore)
    report = format_report(mesa, flex, div)
    if ignore:
        report = f"ignored: {', '.join(sorted(ignore))}\n" + report
    return report, div
