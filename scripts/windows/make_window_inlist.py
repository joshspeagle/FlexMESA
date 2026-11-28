#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build window inlists for the window tests (#684 W2).

A window runs a stage's own inlists from a MESA-written ``.mod`` snapshot for
a bounded number of steps. The window inlist is a mechanical function of
three inputs: the stage's inlist chain, the snapshot header, and the rules in
``.coordination/triage/window-rules.md`` (sections 4 and 5, referred to below
by their rule numbers 4.1 to 4.5). The same file is used for MESA and for
FlexMESA.

Reading the chain
    The stage inlist is read as MESA reads it: ``read_section`` transcribes
    the include loop of ``read_star_job_file``
    (``star/private/star_job_ctrls_io.f90:597-607``), where the include flags
    and names are module-level arrays reset in place, so the walk is not a
    tree. The Fortran namelist text is parsed by ``flexmesa.schema.namelist``
    (the parser the harness uses). Rust ``InlistBundle`` still walks a tree
    (``crates/flexmesa-star/src/inlist.rs:655-707``) and differs from MESA,
    and from this reader, on ``twin_studies``; its flattened result is not
    exposed to Python.

Loud failures
    A ``&star_job`` key that no rule covers raises ``UncoveredKeyError``. A
    change action whose target the snapshot does not satisfy raises
    ``SnapshotMismatchError``. Both carry every offending key so that one
    pass lists them all. Nothing is dropped silently: every change, with its
    rule number, goes to ``window_diff.txt``.

Subcommands
    ``build``     write one window inlist and its diff.
    ``dry-run``   build a window for every stage of every star case with a
                  synthetic snapshot header that satisfies the stage's own
                  change actions, and report per case the diff size, the
                  nominal windows per stage and any uncovered key.
    ``extras-state``  list the star cases whose ``run_star_extras.f90`` uses
                  ``s% xtra*`` or ``s% ixtra*`` (rule 5).
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import math
import pathlib
import re
import sys
import tomllib
from collections.abc import Callable, Iterable, Sequence
from typing import Any

# Snapshot ``file_type`` bits (``write_model.f90:70-84``; the same values the
# Rust writer uses in ``crates/flexmesa-star/src/model_io.rs``).
BIT_PREV_MODEL = 4
BIT_V = 8
BIT_ROTATION = 16
BIT_RSP2 = 64
BIT_RTI = 128
BIT_U = 512
BIT_RSP = 32768

#: Relative tolerance of the "equals" checks of rule 4.3.
CHECK_RTOL = 1.0e-12

#: Namelist sections this constructor rewrites or passes through.
RULED_SECTIONS = ("star_job", "eos", "kap", "controls")
PASSTHROUGH_SECTIONS = ("pgstar", "colors")
#: Passed-through sections whose missing include files are skipped and
#: logged (headless plotting, as the harness does); others must resolve.
TOLERANT_INCLUDE_SECTIONS = ("pgstar",)
INCLUDE_SLOTS = range(1, 6)


class WindowBuildError(Exception):
    """A window cannot be built (the window is reported as not built)."""


class UncoveredKeyError(WindowBuildError):
    """``&star_job`` keys that no rule in window-rules.md covers.

    Attributes:
        keys: Sorted lowercase key names.
    """

    def __init__(self, keys: Iterable[str]) -> None:
        self.keys = sorted(set(keys))
        super().__init__(
            "star_job keys not covered by window-rules.md: "
            + ", ".join(self.keys)
        )


class SnapshotMismatchError(WindowBuildError):
    """A change action the snapshot header does not already satisfy.

    Attributes:
        problems: One message per failed check (key, target, header value).
    """

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


@dataclasses.dataclass(frozen=True)
class Assign:
    """One namelist assignment ``name(sub) = value``.

    Attributes:
        name: Lowercase key.
        sub: Subscript text (``"1"``), or ``None``.
        value: ``bool``, ``int``, ``float`` or ``str``.
    """

    name: str
    sub: str | None
    value: bool | int | float | str


@dataclasses.dataclass(frozen=True)
class SnapshotHeader:
    """The fields of a ``.mod`` header that the rules read.

    Attributes:
        file_type: The ``file_type`` bit mask (first line of the model).
        model_number: ``model_number``.
        star_age: ``star_age`` in years.
        initial_z: ``initial_z``.
        mass_msun: ``M/Msun``.
        net_name: ``net_name`` without quotes.
        l_center: ``L_center`` (erg/s); the writer omits it when it is 0.
        v_center: ``v_center`` (cm/s); the writer omits it when it is 0.
        initial_y: ``initial_y`` when the header carries it (the r26.4.1
            writer does not), else ``None``.
    """

    file_type: int
    model_number: int
    star_age: float
    initial_z: float
    mass_msun: float
    net_name: str
    l_center: float = 0.0
    v_center: float = 0.0
    initial_y: float | None = None

    @property
    def has_rotation(self) -> bool:
        """Whether rotation is on in the snapshot."""
        return bool(self.file_type & BIT_ROTATION)

    @property
    def has_prev_model(self) -> bool:
        """Whether the previous-model block is present (``no-prev-model``)."""
        return bool(self.file_type & BIT_PREV_MODEL)


def _fortran_float(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def parse_mod_header(text: str) -> SnapshotHeader:
    """Read the header of a MESA ``.mod`` file.

    Args:
        text: Full text of the ``.mod`` (only the header is read).

    Returns:
        The header fields used by the window rules.

    Raises:
        WindowBuildError: If a needed field is missing.
    """
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("!")]
    it = iter(lines)
    first = next((ln for ln in it if ln.strip()), "")
    try:
        file_type = int(first.split()[0])
    except (IndexError, ValueError) as exc:
        raise WindowBuildError(f"bad .mod file_type line: {first!r}") from exc
    fields: dict[str, str] = {}
    for ln in it:
        if not ln.strip():
            if fields:
                break
            continue
        parts = ln.split(None, 1)
        if len(parts) == 2:
            fields.setdefault(parts[0], parts[1].split("  ")[0].strip())
    need = ("model_number", "star_age", "initial_z", "M/Msun", "net_name")
    missing = [k for k in need if k not in fields]
    if missing:
        raise WindowBuildError(f".mod header lacks {', '.join(missing)}")
    return SnapshotHeader(
        file_type=file_type,
        model_number=int(fields["model_number"]),
        star_age=_fortran_float(fields["star_age"]),
        initial_z=_fortran_float(fields["initial_z"]),
        mass_msun=_fortran_float(fields["M/Msun"]),
        net_name=fields["net_name"].strip().strip("'\""),
        l_center=_fortran_float(fields.get("L_center", "0")),
        v_center=_fortran_float(fields.get("v_center", "0")),
        initial_y=(
            _fortran_float(fields["initial_y"])
            if "initial_y" in fields
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Reading a stage's inlist chain
# ---------------------------------------------------------------------------


def _resolve(case_dir: pathlib.Path, name: str) -> pathlib.Path:
    """Find an inlist in the case directory or ``../shared`` (as the harness).

    Raises:
        WindowBuildError: If the file does not exist.
    """
    direct = case_dir / name.strip()
    if direct.is_file():
        return direct
    shared = case_dir.parent / "shared" / name.strip()
    if shared.is_file():
        return shared
    raise WindowBuildError(f"missing inlist {name!r} for {case_dir}")


#: ``max_extra_inlists`` (``const/public/const_def.f90:39``).
MAX_EXTRA_INLISTS = 5
#: Nesting limit of the ctrls readers (``level >= 10`` is an error).
MAX_INCLUDE_LEVEL = 10
#: Sections whose namelist group must exist in every file read (a missing
#: group is a read error in ``read_star_job_file`` and
#: ``read_controls_file``); ``&eos`` and ``&kap`` accept its absence.
STRICT_SECTIONS = ("star_job", "controls")


def _include_slot(section: str, name: str, sub: str | None) -> tuple[str, int]:
    """Classify an include key as ``("read"|"name", slot)`` or ``("", 0)``.

    The namelist variables are the arrays ``read_extra_<s>_inlist(1:5)`` and
    ``extra_<s>_inlist_name(1:5)``; an unsubscripted array name assigns
    element 1.

    Raises:
        WindowBuildError: For the numbered spelling ``..._inlist<i>``, which
            is not a MESA namelist variable.
    """
    for kind, pattern in (
        ("read", rf"read_extra_{section}_inlist"),
        ("name", rf"extra_{section}_inlist_name"),
    ):
        if name == pattern:
            slot = 1 if sub is None or not sub.strip() else int(sub)
            return kind, slot
    if re.fullmatch(
        rf"(read_extra_{section}_inlist\d|extra_{section}_inlist\d_name)",
        name,
    ):
        raise WindowBuildError(
            f"{name} is not a MESA namelist variable (use the subscripted "
            f"array form)"
        )
    return "", 0


def read_section(
    case_dir: pathlib.Path,
    entry: pathlib.Path,
    section: str,
    parse: Callable[[pathlib.Path], list[Any]],
    notes: list[str],
    files: list[str] | None = None,
) -> list[Assign]:
    """Flatten one namelist section of an inlist chain as MESA reads it.

    Transcribes the include loop of ``read_star_job_file``
    (``star/private/star_job_ctrls_io.f90:597-607``; the same pattern is in
    ``read_controls_file``, ``star/private/ctrls_io.f90:700-709``, and the
    ``eos`` and ``kap`` readers). The include flags and names are
    module-level arrays: a file's namelist read sets some slots, then for
    ``i = 1..5`` the slot is copied, reset (``.false.`` and ``'undefined'``)
    and, if it was set, the named file is read at once, recursively. A child
    that sets slot ``j`` therefore overrides the parent's slot ``j``, and the
    parent's later slots are already reset when the child returns. This is
    not a tree walk. Only the first ``&<section>`` group of a file is read
    (a Fortran namelist read stops at the first match).

    Args:
        case_dir: Case directory (extra inlists are found here or in
            ``../shared``).
        entry: File to read first.
        section: Lowercase section name.
        parse: ``flexmesa.schema.namelist.parse_namelist_file``.
        notes: Receives one diff line per skipped include of a headless
            section.
        files: When given, receives the names of the files read, in order.

    Returns:
        Assignments in the order MESA applies them (later ones win).

    Raises:
        WindowBuildError: A missing include, a missing group in a strict
            section, or more than 9 levels of nesting.
    """
    flags = dict.fromkeys(range(1, MAX_EXTRA_INLISTS + 1), False)
    names = dict.fromkeys(range(1, MAX_EXTRA_INLISTS + 1), "undefined")
    out: list[Assign] = []

    def read_file(path: pathlib.Path, level: int) -> None:
        if level >= MAX_INCLUDE_LEVEL:
            raise WindowBuildError(
                f"too many levels of nested extra {section} inlist files "
                f"at {path.name}"
            )
        if files is not None:
            files.append(path.name)
        groups = [b for b in parse(path) if b.name.lower() == section]
        if groups:
            for a in groups[0].assignments:
                low = a.name.lower()
                kind, slot = _include_slot(section, low, a.subscript)
                if kind == "read":
                    flags[slot] = bool(a.value)
                elif kind == "name":
                    names[slot] = str(a.value)
                else:
                    out.append(Assign(low, a.subscript, a.value))
        elif section in STRICT_SECTIONS:
            raise WindowBuildError(f"no &{section} namelist in {path.name}")
        for i in range(1, MAX_EXTRA_INLISTS + 1):
            read_extra, extra = flags[i], names[i]
            flags[i], names[i] = False, "undefined"
            if not read_extra:
                continue
            try:
                nxt = _resolve(case_dir, extra)
            except WindowBuildError:
                if section not in TOLERANT_INCLUDE_SECTIONS:
                    raise
                # Headless plotting section: the harness skips a missing
                # include too. Recorded, never silent.
                notes.append(
                    f"4.5 | {section} | extra_{section}_inlist_name({i}) | "
                    f"{extra!r} not found, skipped (pgstar_flag is .false.)"
                )
                continue
            read_file(nxt, level + 1)

    read_file(entry, 1)
    return out


@dataclasses.dataclass
class Stage:
    """A stage's flattened inlist chain.

    Attributes:
        sections: ``{section: assignments}`` in entry-file order.
        notes: Diff lines for includes skipped while reading.
        files: ``{section: names of the files read, in order}``.
    """

    sections: dict[str, list[Assign]]
    notes: list[str]
    files: dict[str, list[str]] = dataclasses.field(default_factory=dict)


def read_stage(
    case_dir: pathlib.Path,
    entry_name: str,
    parse: Callable[[pathlib.Path], list[Any]] | None = None,
) -> Stage:
    """Read every section of a stage's inlist chain.

    Args:
        case_dir: Case directory.
        entry_name: The stage's entry inlist (the ``do_one`` inlist).
        parse: Namelist file parser (default: the harness parser).

    Returns:
        The sections that appear in the entry file, in file order.

    Raises:
        WindowBuildError: If an inlist file is missing.
    """
    if parse is None:
        from flexmesa.schema import namelist

        parse = namelist.parse_namelist_file
    entry = _resolve(case_dir, entry_name)
    sections: dict[str, list[Assign]] = {}
    notes: list[str] = []
    files: dict[str, list[str]] = {}
    for block in parse(entry):
        low = block.name.lower()
        if low in sections:
            continue
        files[low] = []
        sections[low] = read_section(
            case_dir, entry, low, parse, notes, files[low]
        )
    return Stage(sections, notes, files)


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------

# Rules 4.2, 4.3, 4.5 and 4.6: a boolean switch becomes .false.; any other
# key is removed.
_OFF_PATTERNS: tuple[tuple[str, str], ...] = (
    ("4.2", r"set_initial_age|initial_age"),
    ("4.2", r"set_initial_model_number|initial_model_number"),
    ("4.2", r"set_initial_cumulative_energy_error"),
    ("4.2", r"new_cumulative_energy_error"),
    ("4.2", r"set_initial_dt|years_for_initial_dt|limit_initial_dt"),
    ("4.2", r"seconds_for_initial_dt"),
    ("4.2", r"set_initial_number_retries"),
    ("4.2", r"pre_ms_.*"),
    ("4.2", r"initial_zfracs|set_uniform_initial_composition"),
    ("4.2", r"initial_(h1|h2|he3|he4)"),
    # Rule 4.6: any key containing "relax" (substring, not only prefix).
    ("4.2", r".*relax.*"),
    ("4.2", r"remove_surface_.*|remove_initial_center_.*|remove_center_.*"),
    ("4.2", r"set_initial_tau_factor|set_tau_factor|set_to_this_tau_factor"),
    ("4.2", r"profile_starting_model|steps_to_take_before_terminate"),
    # Rule 4.6: parameters of relax, create and removal actions.
    (
        "4.2",
        r"core_avg_eps|core_avg_rho|dlg_core_mass_per_step|dlgm_per_step"
        r"|lg_max_abs_mdot|dlogtau_factor|irrad_col_depth"
        r"|mass_in_gm_for_create_initial_model"
        r"|radius_in_cm_for_create_initial_model"
        r"|extra_mass_retained_by_remove_h_env"
        r"|dump_missing_metals_into_heaviest",
    ),
    (
        "4.3",
        r"replace_initial_element|replace_element.*|set_nzlo"
        r"|set_initial_to_xa_for_accretion",
    ),
    ("4.5", r"pause_before_terminate"),
)

# Rule 4.3: action flags and targets, checked against the snapshot only
# when the action flag is on in the stage (rule 4.6 (c)).
_ROTATION_KEYS = re.compile(
    r"new_omega_div_omega_crit|set_initial_omega.*|new_omega"
    r"|set_initial_surface_rotation_v|new_surface_rotation_v"
)
_ROTATION_ACTIONS = re.compile(
    r"(set|relax)_(initial_)?(omega|omega_div_omega_crit"
    r"|surface_rotation_v)"
)
_ROTATION_TARGETS = (
    "new_omega",
    "new_omega_div_omega_crit",
    "new_surface_rotation_v",
)
_MASS_KEYS = re.compile(r"new_mass|new_core_mass|change_mass_years_for_dt")
_NET_KEYS = re.compile(r"change_net|change_initial_net|new_net_name")
_Z_KEYS = re.compile(r"change_initial_z|new_z")
_Y_KEYS = re.compile(r"new_y")
_CENTER_KEYS = re.compile(r"(set_initial|new)_(l|v)_center")

# Rule 4.3: kept on, not checked (the .mod records only some of these).
_FLAG_KEEP = re.compile(
    r"(change|change_initial|new)_(v|u|rotation|d_omega|w_div_wc|rti)_flag"
)

# Rule 4.4 and 4.6: kept. The &colors section is kept verbatim as well.
_KEEP = re.compile(
    r"history_columns_file|profile_columns_file|filename_of_special_rate"
    r"|use_special_weak_rates|use_3a_fl87|remove_fallback_at_each_step"
    r"|remove_fallback_speed_limit|set_to_xa_for_accretion"
    r"|set_cumulative_energy_error_at_step"
    r"|set_cumulative_energy_error_each_step_if_age_less_than|show_.*"
    r"|adv_net|auto_extend_net|enable_adaptive_network|min_x_for_add"
    r"|min_x_for_keep|min_x_for_n|electron_coulomb_corrections"
    r"|ion_coulomb_corrections|num_special_rate_factors"
    r"|reaction_for_special_factor|special_rate_factor"
    r"|special_weak_states_file|special_weak_transitions_file"
    r"|rate_tables_dir|rate_cache_suffix|fallback_check_total_energy"
    r"|first_model_for_timing|warn_run_star_extras|mesa_dir"
    r"|required_termination_code_string|save_model_when_terminate"
    r"|write_profile_when_terminate|save_pulse_data_when_terminate"
)

# Rules 4.1 and 4.5: set by the constructor, any stage value replaced.
_SET_KEYS = frozenset(
    {
        "load_saved_model",
        "load_model_filename",
        "create_pre_main_sequence_model",
        "create_initial_model",
        "create_rsp_model",
        "save_model_filename",
        "filename_for_profile_when_terminate",
        "save_photo_filename",
        "pgstar_flag",
        "save_pgstar_files_when_terminate",
        "save_photo_when_terminate",
    }
)
#: Filenames rewritten into the window directory when the stage sets them.
_REWRITE_IF_PRESENT = (
    "filename_for_profile_when_terminate",
    "save_photo_filename",
)


def _classify(key: str) -> str | None:
    """Return the rule family of a ``&star_job`` key, or ``None``."""
    if key in _SET_KEYS:
        return "set"
    if _FLAG_KEEP.fullmatch(key) or _KEEP.fullmatch(key):
        return "keep"
    if (
        _ROTATION_KEYS.fullmatch(key)
        or _MASS_KEYS.fullmatch(key)
        or _NET_KEYS.fullmatch(key)
        or _Z_KEYS.fullmatch(key)
        or _Y_KEYS.fullmatch(key)
        or _CENTER_KEYS.fullmatch(key)
    ):
        return "check"
    for _rule, pat in _OFF_PATTERNS:
        if re.fullmatch(pat, key):
            return "off"
    return None


def _off_rule(key: str) -> str:
    for rule, pat in _OFF_PATTERNS:
        if re.fullmatch(pat, key):
            return rule
    raise AssertionError(key)


def uncovered_star_job_keys(assigns: Iterable[Assign]) -> list[str]:
    """Return the sorted ``&star_job`` keys that no rule covers."""
    return sorted({a.name for a in assigns if _classify(a.name) is None})


def _fmt(value: bool | int | float | str) -> str:
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, float):
        return repr(value).replace("e", "d")
    return str(value)


def _label(a: Assign) -> str:
    return f"{a.name}({a.sub})" if a.sub is not None else a.name


def _last(assigns: Sequence[Assign], key: str) -> Assign | None:
    for a in reversed(assigns):
        if a.name == key:
            return a
    return None


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=CHECK_RTOL, abs_tol=0.0)


@dataclasses.dataclass
class Window:
    """Inputs of one window.

    Attributes:
        prefix: Window name; output paths are rewritten under it.
        snapshot: ``load_model_filename`` value (the ``.mod`` path).
        header: The snapshot header.
        length: W, the number of steps of the window.
    """

    prefix: str
    snapshot: str
    header: SnapshotHeader
    length: int


@dataclasses.dataclass
class WindowInlist:
    """A constructed window.

    Attributes:
        sections: Flattened assignments per namelist section.
        diff: One line per change: ``rule | section | key | change``.
        flags: Manifest flags: ``relax-assumed`` (a core relax was turned
            off without a check, rule 4.6 (c)) and ``no-prev-model`` (the
            snapshot has no previous-model block, rule 4.2).
    """

    sections: dict[str, list[Assign]]
    diff: list[str]
    flags: list[str] = dataclasses.field(default_factory=list)


class _Editor:
    """Applies edits to one section's assignments and logs each."""

    def __init__(self, section: str, assigns: list[Assign], diff: list[str]):
        self.section = section
        self.assigns = assigns
        self.diff = diff

    def _log(self, rule: str, key: str, text: str) -> None:
        self.diff.append(f"{rule} | {self.section} | {key} | {text}")

    def off(self, rule: str, a: Assign) -> None:
        """Turn a boolean switch off, or remove a value key."""
        if isinstance(a.value, bool):
            if a.value:
                self.replace(rule, a, Assign(a.name, a.sub, False))
            return
        self.remove(rule, a, "removed")

    def replace(self, rule: str, old: Assign, new: Assign) -> None:
        """Replace one assignment in place."""
        self.assigns[self.assigns.index(old)] = new
        self._log(rule, _label(old), f"{_fmt(old.value)} -> {_fmt(new.value)}")

    def remove(self, rule: str, a: Assign, how: str) -> None:
        """Remove one assignment."""
        self.assigns.remove(a)
        self._log(rule, _label(a), f"{how} (was {_fmt(a.value)})")

    def set(self, rule: str, key: str, value: bool | int | float | str) -> None:
        """Set a scalar key to ``value``, dropping earlier assignments."""
        old = [a for a in self.assigns if a.name == key and a.sub is None]
        new = Assign(key, None, value)
        if old and all(a.value == value for a in old):
            return
        self.assigns[:] = [a for a in self.assigns if a not in old]
        self.assigns.append(new)
        was = _fmt(old[-1].value) if old else "absent"
        self._log(rule, key, f"{was} -> {_fmt(value)}")


@dataclasses.dataclass
class _Plan:
    """Checks and edits that depend on the snapshot header (rule 4.3)."""

    problems: list[str] = dataclasses.field(default_factory=list)
    off: list[str] = dataclasses.field(default_factory=list)
    remove: list[str] = dataclasses.field(default_factory=list)
    flags: list[str] = dataclasses.field(default_factory=list)


def _plan_checks(star: Sequence[Assign], hdr: SnapshotHeader) -> _Plan:
    """Check each action that is on in the stage against the header.

    A target key is checked only when its action flag is on (rule 4.6 (c));
    with the flag off the key is inert and passes through unchanged.
    """
    plan = _Plan()

    def on(*keys: str) -> bool:
        return any(
            (a := _last(star, k)) is not None and a.value is True for k in keys
        )

    def target(key: str, why: str) -> float | None:
        t = _last(star, key)
        if t is None:
            plan.problems.append(f"{why} is on but {key} is absent")
            return None
        return float(t.value)

    if on("change_net", "change_initial_net"):
        t = _last(star, "new_net_name")
        if t is None:
            plan.problems.append("change_net is on but new_net_name is absent")
        elif str(t.value) != hdr.net_name:
            plan.problems.append(
                f"new_net_name {t.value!r} != header net_name {hdr.net_name!r}"
            )
        else:
            plan.off += ["change_net", "change_initial_net"]
    if on("change_initial_z", "relax_z", "relax_initial_z"):
        z = target("new_z", "a z action")
        if z is not None and not _close(z, hdr.initial_z):
            plan.problems.append(
                f"new_z {z} != header initial_z {hdr.initial_z!r}"
            )
        elif z is not None:
            plan.off.append("change_initial_z")
    if on("relax_mass", "relax_initial_mass"):
        m = target("new_mass", "a mass relax")
        if m is not None and not _close(m, hdr.mass_msun):
            plan.problems.append(
                f"new_mass {m} != header M/Msun {hdr.mass_msun!r}"
            )
        elif m is not None:
            plan.remove += ["new_mass", "change_mass_years_for_dt"]
    if on("relax_core", "relax_initial_core"):
        # The core mass is not in the header: the window must start after
        # the relax (rule 2.1) and is flagged (rule 4.6 (c)).
        plan.flags.append("relax-assumed")
        plan.remove.append("new_core_mass")
    if on("relax_y", "relax_initial_y"):
        y = target("new_y", "a y relax")
        if y is not None and hdr.initial_y is None:
            plan.problems.append(
                f"new_y {y} cannot be checked: the .mod header carries no "
                "initial_y"
            )
        elif y is not None and hdr.initial_y is not None:
            if _close(y, hdr.initial_y):
                plan.remove.append("new_y")
            else:
                plan.problems.append(
                    f"new_y {y} != header initial_y {hdr.initial_y!r}"
                )
    rot_on = sorted(
        {
            a.name
            for a in star
            if _ROTATION_ACTIONS.fullmatch(a.name) and on(a.name)
        }
    )
    if rot_on:
        if not hdr.has_rotation:
            plan.problems.append(
                "rotation actions ("
                + ", ".join(rot_on)
                + ") but rotation is off in the snapshot"
            )
        else:
            plan.off += [k for k in rot_on if k.startswith("set_")]
            plan.remove += list(_ROTATION_TARGETS)
    for axis, value in (("l", hdr.l_center), ("v", hdr.v_center)):
        flag = f"set_initial_{axis}_center"
        if on(flag):
            tv = target(f"new_{axis}_center", flag)
            if tv is not None and not _close(tv, value):
                plan.problems.append(
                    f"new_{axis}_center {tv} != header {axis.upper()}_center "
                    f"{value!r}"
                )
            elif tv is not None:
                plan.off.append(flag)
                plan.remove.append(f"new_{axis}_center")
    return plan


def _apply_star_job(
    assigns: list[Assign], win: Window, diff: list[str], flags: list[str]
) -> list[Assign]:
    """Apply rules 4.1 to 4.6 to a flattened ``&star_job``."""
    uncovered = uncovered_star_job_keys(assigns)
    if uncovered:
        raise UncoveredKeyError(uncovered)
    out = list(assigns)
    ed = _Editor("star_job", out, diff)
    plan = _plan_checks(assigns, win.header)
    if plan.problems:
        raise SnapshotMismatchError(plan.problems)
    for flag in plan.flags:
        if flag not in flags:
            flags.append(flag)
        diff.append(
            f"4.3 | star_job | {flag} | core mass not in the header; relax "
            "turned off unchecked"
        )
    for a in list(out):
        if _classify(a.name) == "off":
            ed.off(_off_rule(a.name), a)
    for key in plan.off:
        for a in [x for x in out if x.name == key]:
            ed.off("4.3", a)
    for key in dict.fromkeys(plan.remove):
        for a in [x for x in out if x.name == key]:
            ed.remove("4.3", a, "removed")
    # Rule 4.1: load the snapshot.
    ed.set("4.1", "load_saved_model", True)
    ed.set("4.1", "load_model_filename", win.snapshot)
    for key in (
        "create_pre_main_sequence_model",
        "create_initial_model",
        "create_rsp_model",
    ):
        ed.set("4.1", key, False)
    # Rule 4.5: output and headless.
    ed.set("4.5", "save_model_filename", f"{win.prefix}/final.mod")
    for key in _REWRITE_IF_PRESENT:
        old = _last(out, key)
        if old is not None:
            ed.set("4.5", key, f"{win.prefix}/{old.value}")
    ed.set("4.5", "pgstar_flag", False)
    ed.set("4.5", "save_pgstar_files_when_terminate", False)
    ed.set("4.5", "save_photo_when_terminate", False)
    return out


def _apply_controls(
    assigns: list[Assign], win: Window, diff: list[str]
) -> list[Assign]:
    """Apply rule 4.5 to a flattened ``&controls`` (all else is kept)."""
    out = list(assigns)
    ed = _Editor("controls", out, diff)
    stop = win.header.model_number + win.length
    old = _last(out, "max_model_number")
    if old is not None and int(old.value) < stop:
        stop = int(old.value)
    ed.set("4.5", "max_model_number", stop)
    ed.set("4.5", "log_directory", f"{win.prefix}/LOGS")
    ed.set("4.5", "photo_directory", f"{win.prefix}/photos")
    return out


def build_window_inlist(stage: Stage, win: Window) -> WindowInlist:
    """Apply the window rules to a stage's flattened sections.

    Args:
        stage: Output of ``read_stage``.
        win: The window's inputs.

    Returns:
        The window's sections and its diff.

    Raises:
        UncoveredKeyError: A ``&star_job`` key no rule covers.
        SnapshotMismatchError: A change action the snapshot does not satisfy.
    """
    sections = stage.sections
    diff: list[str] = list(stage.notes)
    flags: list[str] = []
    if not win.header.has_prev_model:
        flags.append("no-prev-model")
    out: dict[str, list[Assign]] = {}
    unknown = [
        f"&{n}"
        for n in sections
        if n not in RULED_SECTIONS + PASSTHROUGH_SECTIONS
    ]
    if unknown:
        star_keys = uncovered_star_job_keys(sections.get("star_job", []))
        raise UncoveredKeyError([*unknown, *star_keys])
    for name, assigns in sections.items():
        if name == "star_job":
            out[name] = _apply_star_job(assigns, win, diff, flags)
        elif name == "controls":
            out[name] = _apply_controls(assigns, win, diff)
        else:
            out[name] = list(assigns)
    # The rules write these even when the stage's inlist lacks the section.
    if "star_job" not in out:
        out["star_job"] = _apply_star_job([], win, diff, flags)
    if "controls" not in out:
        out["controls"] = _apply_controls([], win, diff)
    return WindowInlist(out, diff, flags)


def format_inlist(sections: dict[str, list[Assign]]) -> str:
    """Write flattened sections as a MESA inlist."""
    parts: list[str] = []
    for name, assigns in sections.items():
        parts.append(f"&{name}\n")
        parts.extend(f"   {_label(a)} = {_fmt(a.value)}\n" for a in assigns)
        parts.append(f"/ ! end of {name} namelist\n\n")
    return "".join(parts)


def format_diff(
    window: Window, diff: Sequence[str], flags: Sequence[str] = ()
) -> str:
    """Write ``window_diff.txt``: every change with its rule number."""
    head = [
        f"# window {window.prefix}: snapshot {window.snapshot}, "
        f"start model {window.header.model_number}, W = {window.length}",
    ]
    if flags:
        head.append(f"# flags: {', '.join(flags)}")
    head.append("# rule | section | key | change")
    return "\n".join([*head, *diff, ""])


def write_window(
    out_dir: pathlib.Path, window: Window, built: WindowInlist
) -> None:
    """Write ``inlist_window`` and ``window_diff.txt`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inlist_window").write_text(
        format_inlist(built.sections), encoding="utf-8"
    )
    (out_dir / "window_diff.txt").write_text(
        format_diff(window, built.diff, built.flags), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# run_star_extras state (rule 5)
# ---------------------------------------------------------------------------

_XTRA = re.compile(r"\bs\s*%\s*([il]?)xtra", re.IGNORECASE)


def extras_state_use(extras_path: pathlib.Path) -> dict[str, int]:
    """Count the uses of the saved-state arrays in a ``run_star_extras``.

    Rule 5 names ``s% xtra*`` and ``s% ixtra*``. ``s% lxtra*`` (the logical
    array of the same kind, saved the same way) is counted separately so that
    claude-a can rule on it. Comment text after ``!`` is ignored.

    Returns:
        ``{"xtra": n, "ixtra": n, "lxtra": n}`` with only non-zero entries.
    """
    if not extras_path.is_file():
        return {}
    counts: collections.Counter[str] = collections.Counter()
    text = extras_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        for m in _XTRA.finditer(line.split("!", 1)[0]):
            counts[m.group(1).lower() + "xtra"] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Dry run over the star cases
# ---------------------------------------------------------------------------


def synthetic_header(stage: Stage) -> SnapshotHeader:
    """A header that satisfies the stage's own change actions.

    Used only by the dry run, to exercise key coverage and the diff without
    a snapshot. Real snapshots (W1) are checked by ``build_window_inlist``.
    """
    star = stage.sections.get("star_job", [])
    net = _last(star, "new_net_name")
    z = _last(star, "new_z")
    mass = _last(star, "new_mass")
    y = _last(star, "new_y")
    l_c = _last(star, "new_l_center")
    v_c = _last(star, "new_v_center")
    rot = any(_ROTATION_ACTIONS.fullmatch(a.name) for a in star)
    return SnapshotHeader(
        file_type=(BIT_ROTATION if rot else 0) | BIT_PREV_MODEL,
        model_number=1,
        star_age=0.0,
        initial_z=float(z.value) if z is not None else 0.02,
        mass_msun=float(mass.value) if mass is not None else 1.0,
        net_name=str(net.value) if net is not None else "basic.net",
        l_center=float(l_c.value) if l_c is not None else 0.0,
        v_center=float(v_c.value) if v_c is not None else 0.0,
        initial_y=float(y.value) if y is not None else None,
    )


#: Stages of 60 models or fewer get one window (rule 2); others up to five
#: start points (first model, 25, 50, 75 and 95 percent).
SHORT_STAGE_MODELS = 60
NOMINAL_LONG_WINDOWS = 5


def nominal_windows(stage_steps: int | None) -> int:
    """Windows per stage before the duplicate rule (which needs W)."""
    if stage_steps is None or stage_steps <= SHORT_STAGE_MODELS:
        return 1
    return NOMINAL_LONG_WINDOWS


@dataclasses.dataclass
class StageReport:
    """Dry-run result of one stage."""

    inlist: str
    windows: int
    diff_lines: int
    uncovered: list[str]
    error: str
    built: bool = False
    flags: tuple[str, ...] = ()


def dry_run_case(
    case_dir: pathlib.Path,
    stage_steps: Sequence[int] | None,
    parse: Callable[[pathlib.Path], list[Any]] | None = None,
    stages: Sequence[str] | None = None,
) -> list[StageReport]:
    """Build a synthetic window for each ``do_one`` stage of one case."""
    if stages is None:
        from flexmesa.star import e2e

        stages = [
            a1
            for kind, a1, _a2 in e2e.parse_rn_stages(case_dir)
            if kind == "do_one"
        ]
    reports = []
    for i, inlist in enumerate(stages):
        steps = stage_steps[i] if stage_steps and i < len(stage_steps) else None
        try:
            stage = read_stage(case_dir, inlist, parse)
            win = Window(
                "w0", "snapshot.mod", synthetic_header(stage), length=100
            )
            built = build_window_inlist(stage, win)
            reports.append(
                StageReport(
                    inlist,
                    nominal_windows(steps),
                    len(built.diff),
                    [],
                    "",
                    True,
                    tuple(built.flags),
                )
            )
        except UncoveredKeyError as exc:
            reports.append(
                StageReport(inlist, nominal_windows(steps), 0, exc.keys, "")
            )
        except WindowBuildError as exc:
            reports.append(
                StageReport(inlist, nominal_windows(steps), 0, [], str(exc))
            )
    return reports


def _star_cases(manifest: pathlib.Path) -> list[str]:
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    return [c["case"] for c in data["cases"] if c["suite"] == "star"]


def _cmd_dry_run(args: argparse.Namespace) -> int:
    import json

    from flexmesa.testing import harness_runner

    steps = json.loads(args.stage_steps.read_text(encoding="utf-8"))["cases"]
    cases = args.cases or _star_cases(args.manifest)
    key_cases: dict[str, set[str]] = collections.defaultdict(set)
    other_errors: list[str] = []
    n_stages = built_stages = built_cases = 0
    nominal = built_windows = 0
    flag_stages: collections.Counter[str] = collections.Counter()
    print("case | stages | built | windows | diff lines | flags | uncovered")
    for case in cases:
        case_dir = harness_runner.resolve_case_dir("star", case, args.mesa_dir)
        ss = steps.get(f"star/{case}", {}).get("stage_steps")
        reps = dry_run_case(case_dir, ss)
        n_stages += len(reps)
        nominal += sum(r.windows for r in reps)
        ok = [r for r in reps if r.built]
        built_stages += len(ok)
        built_windows += sum(r.windows for r in ok)
        built_cases += len(ok) == len(reps)
        flags = sorted({f for r in ok for f in r.flags})
        for f in flags:
            flag_stages[f] += sum(f in r.flags for r in ok)
        unc = sorted({k for r in reps for k in r.uncovered})
        for k in unc:
            key_cases[k].add(case)
        other_errors.extend(
            f"{case}/{r.inlist}: {r.error}" for r in reps if r.error
        )
        print(
            f"{case} | {len(reps)} | {len(ok)} | "
            f"{sum(r.windows for r in ok)} | "
            f"{sum(r.diff_lines for r in ok)} | {','.join(flags) or '-'} | "
            f"{', '.join(unc) or '-'}"
        )
    print(
        f"\ncases: {len(cases)}; cases fully built: {built_cases}; "
        f"stages: {n_stages}; stages built: {built_stages}; "
        f"nominal windows: {nominal}; windows built: {built_windows}; "
        f"distinct uncovered keys: {len(key_cases)}"
    )
    for f, n in sorted(flag_stages.items()):
        print(f"FLAG {f}: {n} stages")
    for k in sorted(key_cases):
        print(f"UNCOVERED {k}: {len(key_cases[k])} cases")
    for e in other_errors:
        print(f"ERROR {e}")
    return 1 if key_cases or other_errors else 0


def _cmd_extras(args: argparse.Namespace) -> int:
    from flexmesa.testing import harness_runner

    print("# star cases whose run_star_extras.f90 uses s% xtra/ixtra/lxtra")
    print("# (rule 5; generated by make_window_inlist.py extras-state)")
    print("# case | xtra | ixtra | lxtra (use counts)")
    for case in args.cases or _star_cases(args.manifest):
        case_dir = harness_runner.resolve_case_dir("star", case, args.mesa_dir)
        use = extras_state_use(case_dir / "src" / "run_star_extras.f90")
        if use:
            print(
                f"{case} | {use.get('xtra', 0)} | {use.get('ixtra', 0)} | "
                f"{use.get('lxtra', 0)}"
            )
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    header = parse_mod_header(args.snapshot.read_text(encoding="utf-8"))
    stage = read_stage(args.case_dir, args.stage_inlist)
    win = Window(args.prefix, args.snapshot.name, header, args.steps)
    try:
        built = build_window_inlist(stage, win)
    except WindowBuildError as exc:
        print(f"window not built ({exc})", file=sys.stderr)
        return 2
    write_window(args.out, win, built)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    root = pathlib.Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write one window inlist and its diff")
    b.add_argument("--case-dir", type=pathlib.Path, required=True)
    b.add_argument("--stage-inlist", required=True)
    b.add_argument("--snapshot", type=pathlib.Path, required=True)
    b.add_argument("--prefix", required=True)
    b.add_argument("--steps", type=int, required=True)
    b.add_argument("--out", type=pathlib.Path, required=True)
    b.set_defaults(fn=_cmd_build)
    for name, fn in (("dry-run", _cmd_dry_run), ("extras-state", _cmd_extras)):
        d = sub.add_parser(name)
        d.add_argument("--cases", nargs="*")
        d.add_argument("--mesa-dir", type=pathlib.Path, default=None)
        d.add_argument(
            "--manifest",
            type=pathlib.Path,
            default=root / "benchmarks/history_harness/case_manifest.toml",
        )
        d.add_argument(
            "--stage-steps",
            type=pathlib.Path,
            default=root / "benchmarks/history_harness/mesa_stage_steps.json",
        )
        d.set_defaults(fn=fn)
    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
