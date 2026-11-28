# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-routine oracle provenance checks (`DECISIONS #80`).

A routine counts as oracle-tested only through an ``[[tests.oracles]]``
entry of a ``kind = "oracle"`` test whose expected values trace to a
committed MESA ``26.4.1`` artifact (`DECISIONS #80`). Each entry names:

* ``reference``: a repository file under one of ``REPO_REF_PREFIXES`` (or a
  crate's ``tests/data/``), or ``<bundle>:<member>`` for a member of a
  MESA-origin data bundle listed in its ``bundle_hashes`` manifest;
* ``recipe``: the generator, script or Makefile that produced a repository
  reference (not needed for verbatim copies of upstream files listed in
  ``GOLDEN_MANIFEST_REL``, or for bundle members);
* ``precision``: ``exact`` (0 ULP) or ``print`` (a tolerance comparison,
  with ``tolerance`` given);
* ``consts``: optional names of constants in the test that embed expected
  values; every 64-bit hex literal in them must appear in the reference;
* ``routines``: the ``<upstream path>::<routine>`` ids the entry evidences.

Checks, applied per entry and per routine (``CHECKS``):

``reference``
    the reference exists in an allowed location or bundle manifest;
``origin``
    a repository file carries a MESA ``26.4.1`` header and is not a
    self-generated snapshot, or it is a verbatim upstream copy listed in
    ``GOLDEN_MANIFEST_REL`` whose SHA-256 still matches;
``recipe``
    a repository reference names an existing recipe;
``linkage``
    the test code names the reference (path or bundle member), unless the
    entry lists ``consts``, whose check links the values instead;
``consts``
    every literal embedded in the named ``const``, ``static`` or ``let``
    items appears in a committed repository reference (hex literals as hex
    text, decimal literals as an equal ``float64``);
``scope``
    some ``#[test]`` function reaches both a call to the routine's Rust
    target and the reference data (through helper functions of the same
    file), so the assertion that uses the data exercises the routine;
``precision``
    ``exact`` entries show a 0-ULP comparison on that path; ``print``
    entries state a positive tolerance.

When a check cannot be decided (for example an unparsable function layout or
a Python test), it fails closed: the routine does not count.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import tomllib
from dataclasses import dataclass, field
from typing import Any, Literal, cast

Precision = Literal["exact", "print"]
VALID_PRECISIONS: tuple[Precision, ...] = ("exact", "print")

CHECKS: tuple[str, ...] = (
    "reference",
    "origin",
    "recipe",
    "linkage",
    "consts",
    "scope",
    "precision",
)

REPO_REF_PREFIXES: tuple[str, ...] = (
    "benchmarks/fortran_mesa/",
    "benchmarks/module_hex/",
    "benchmarks/history_harness/reference_histories/",
    "benchmarks/history_harness/reference_artifacts/",
    "tests/fixtures/golden/",
)
_CRATE_DATA_RE = re.compile(r"^crates/flexmesa-[a-z0-9_-]+/tests/data/")
GOLDEN_PREFIX = "tests/fixtures/golden/"
GOLDEN_MANIFEST_REL = "tests/fixtures/golden/PROVENANCE.toml"
BUNDLES_REL = "src/flexmesa/data/bundles.toml"
REPORT_MD_RELPATH = "notes/audit/oracle_provenance.md"
REPORT_JSON_RELPATH = "notes/audit/oracle_provenance.json"
BUNDLE_HASHES_DIR = "src/flexmesa/data"

# Bundles whose contents are MESA 26.4.1 output, with code markers that show
# a test loads them.
MESA_ORIGIN_BUNDLES: dict[str, tuple[str, ...]] = {
    "mesa_test_inputs": (
        "mesa_test_input",
        "require_mesa_test_input_path",
        "resolve_mesa_test_inputs_dir",
    ),
    "mesa_module_hex": ("module_hex",),
    "mesa_histories": ("mesa_histories", "histories"),
    "mesa_stepdumps": ("stepdump",),
    "mesa_stepdump_windows": ("stepdump_window",),
    "mesa_replay": ("replay",),
    "mesa_astero_tracks": ("astero_tracks", "astero-tracks"),
    "mesa_pulse_export": ("pulse_export",),
}

_VERSION_HDR_RE = re.compile(r"\br?26\.4\.1\b")
_HEX_LIT_RE = re.compile(r"0[xX]([0-9a-fA-F_]{4,})")
# Rust decimal float literals (a fraction or an exponent; integers excluded).
_DEC_LIT_RE = re.compile(
    r"(?<![\w.])\d[\d_]*(?:\.\d[\d_]*(?:[eE][-+]?\d+)?|[eE][-+]?\d+)"
)
# Numbers printed in a reference file (Fortran `D` exponents included).
_REF_NUM_RE = re.compile(
    r"(?<![\w.])\d+(?:\.\d*(?:[eEdD][-+]?\d+)?|[eEdD][-+]?\d+)"
)
_TEST_ATTR_RE = re.compile(r"#\[\s*(?:tokio::)?test(?:\([^\)]*\))?\s*\]")
_FN_RE = re.compile(r"\bfn\s+([A-Za-z_]\w*)")
_ITEM_RE = re.compile(r"\b(?:const|static)\s+([A-Z_][A-Z0-9_]*)\b")
_CALL_RE = re.compile(r"(?<![\w])([A-Za-z_]\w*)\s*(?:::\s*<[^>()]*>)?\s*\(")
_EXACT_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\.to_bits\(\)"),
    re.compile(r"\bassert_bits_eq\w*\s*\("),
    re.compile(r"\bulps?\w*\s*(?:==|<=)\s*0(?:_u64)?\b"),
    re.compile(r"\bassert_eq!\(\s*(?:u|ulps?|max_ulp)\s*,\s*0\b"),
    re.compile(r"\bnonzero_ulp"),
)
# A statement that mentions ULPs, hex or bits and passes a final `0` argument
# (`check_field(.., 0)`, `assert_hex_ulp(.., 0)`) is a 0-ULP comparison.
_EXACT_STMT_RE = re.compile(r",\s*0(?:_u64)?\s*,?\s*\)\s*$")
_EXACT_STMT_HINT_RE = re.compile(r"ulp|hex|bits", re.IGNORECASE)


def _has_exact_comparison(text: str) -> bool:
    """Return ``True`` when ``text`` contains a recognised 0-ULP comparison."""
    if any(rx.search(text) for rx in _EXACT_RES):
        return True
    for stmt in re.split(r"[;{}]", text):
        s = stmt.strip()
        if _EXACT_STMT_RE.search(s) and _EXACT_STMT_HINT_RE.search(s):
            return True
    return False


@dataclass(frozen=True)
class OracleEntry:
    """One ``[[tests.oracles]]`` entry of a taxonomy test record."""

    reference: str
    precision: str
    routines: tuple[str, ...]
    recipe: str = ""
    tolerance: str = ""
    consts: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClaimResult:
    """Outcome for one ``(test, entry, routine)`` oracle claim."""

    test: str
    reference: str
    routine: str
    precision: str
    failures: tuple[tuple[str, str], ...] = ()

    @property
    def counts(self) -> bool:
        """Whether the claim is verified (no failed check)."""
        return not self.failures


@dataclass
class _Ctx:
    """Caches shared across entries of one verification run."""

    repo_root: pathlib.Path
    bundle_members: dict[str, set[str]] = field(default_factory=dict)
    golden: dict[str, dict[str, str]] | None = None
    text: dict[str, str] = field(default_factory=dict)


def parse_oracles(item: dict[str, object]) -> tuple[OracleEntry, ...]:
    """Parse the ``oracles`` array of one raw ``[[tests]]`` table."""
    out: list[OracleEntry] = []
    raw = item.get("oracles", ())
    if not isinstance(raw, list):
        return ()
    for o in raw:
        if not isinstance(o, dict):
            continue
        out.append(
            OracleEntry(
                reference=str(o.get("reference", "")),
                precision=str(o.get("precision", "")),
                routines=tuple(str(r) for r in o.get("routines", ())),
                recipe=str(o.get("recipe", "")),
                tolerance=str(o.get("tolerance", "")),
                consts=tuple(str(c) for c in o.get("consts", ())),
            )
        )
    return tuple(out)


def structural_errors(path: str, entry: OracleEntry) -> list[str]:
    """Return schema errors for one entry (these fail the taxonomy check)."""
    errs: list[str] = []
    if not entry.reference:
        errs.append(f"{path}: oracle entry without a reference")
    if entry.precision not in VALID_PRECISIONS:
        errs.append(
            f"{path}: oracle entry {entry.reference!r} has precision "
            f"{entry.precision!r}; expected one of {VALID_PRECISIONS}"
        )
    if not entry.routines:
        errs.append(
            f"{path}: oracle entry {entry.reference!r} lists no routines"
        )
    return errs


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_golden(ctx: _Ctx) -> dict[str, dict[str, str]]:
    if ctx.golden is None:
        man = ctx.repo_root / GOLDEN_MANIFEST_REL
        ctx.golden = {}
        if man.is_file():
            data = tomllib.loads(man.read_text(encoding="utf-8"))
            for f in data.get("files", ()):
                ctx.golden[str(f["path"])] = {
                    "upstream": str(f.get("upstream", "")),
                    "sha256": str(f.get("sha256", "")),
                }
    return ctx.golden


def _bundle_members(ctx: _Ctx, bundle: str) -> set[str] | None:
    if bundle in ctx.bundle_members:
        return ctx.bundle_members[bundle]
    reg = ctx.repo_root / BUNDLES_REL
    if not reg.is_file():
        return None
    data = tomllib.loads(reg.read_text(encoding="utf-8"))
    for b in data.get("bundles", ()):
        if b.get("name") == bundle and b.get("manifest_file"):
            man = ctx.repo_root / BUNDLE_HASHES_DIR / str(b["manifest_file"])
            if not man.is_file():
                return None
            members: set[str] = set()
            for line in man.read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    members.add(parts[2].strip())
            ctx.bundle_members[bundle] = members
            return members
    return None


def _is_bundle_ref(ref: str) -> bool:
    """Return ``True`` for a ``<bundle>:<member>`` reference."""
    return ":" in ref and not ref.startswith(("/", "."))


def _is_repo_ref_location(rel: str) -> bool:
    return rel.startswith(REPO_REF_PREFIXES) or bool(_CRATE_DATA_RE.match(rel))


def _ref_text(ctx: _Ctx, rel: str) -> str:
    if rel not in ctx.text:
        p = ctx.repo_root / rel
        ctx.text[rel] = (
            p.read_text(encoding="utf-8", errors="replace")
            if p.is_file()
            else ""
        )
    return ctx.text[rel]


def check_reference_entry(
    ctx: _Ctx, entry: OracleEntry
) -> list[tuple[str, str]]:
    """Run the entry-level checks ``reference``, ``origin`` and ``recipe``."""
    ref = entry.reference
    fails: list[tuple[str, str]] = []
    if _is_bundle_ref(ref):
        bundle, member = ref.split(":", 1)
        if bundle not in MESA_ORIGIN_BUNDLES:
            fails.append(
                ("reference", f"bundle {bundle!r} is not a MESA-origin bundle")
            )
            return fails
        members = _bundle_members(ctx, bundle)
        if members is None:
            fails.append(("reference", f"no manifest for bundle {bundle!r}"))
        elif member not in members:
            fails.append(
                ("reference", f"{member!r} is not in the {bundle} manifest")
            )
        return fails
    path = ctx.repo_root / ref
    if not _is_repo_ref_location(ref):
        fails.append(("reference", f"{ref!r} is outside the reference areas"))
        return fails
    if not path.is_file():
        fails.append(("reference", f"{ref!r} does not exist"))
        return fails
    golden = (
        _load_golden(ctx).get(ref) if ref.startswith(GOLDEN_PREFIX) else None
    )
    if golden is not None:
        # A verbatim copy of an upstream file: its origin is the recorded
        # upstream path and SHA-256, so no header or recipe is needed.
        if not golden["upstream"]:
            fails.append(
                (
                    "origin",
                    f"{ref!r} has no upstream path in {GOLDEN_MANIFEST_REL}",
                )
            )
        elif golden["sha256"] != _sha256(path):
            fails.append(
                ("origin", f"{ref!r} no longer matches its recorded SHA-256")
            )
        return fails
    head = "\n".join(_ref_text(ctx, ref).splitlines()[:60])
    low = head.lower()
    if "self-generated" in low or "regression snapshot" in low:
        fails.append(("origin", f"{ref!r} is marked self-generated"))
    elif not _VERSION_HDR_RE.search(head):
        fails.append(("origin", f"{ref!r} has no MESA 26.4.1 version header"))
    if not entry.recipe:
        fails.append(("recipe", f"{ref!r} names no recipe"))
    elif not (ctx.repo_root / entry.recipe).exists():
        fails.append(("recipe", f"recipe {entry.recipe!r} does not exist"))
    return fails


def reference_tokens(ref: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(any_of, markers)`` code tokens that show a test loads ``ref``.

    A repository reference is linked when any token of ``any_of`` appears.
    A bundle member is linked when its relative path appears, or when its
    file name and one of the bundle's markers both appear.
    """
    if _is_bundle_ref(ref):
        bundle, member = ref.split(":", 1)
        return (member, pathlib.PurePosixPath(member).name), (
            MESA_ORIGIN_BUNDLES.get(bundle, ())
        )
    from flexmesa.testing import test_taxonomy

    return tuple(test_taxonomy._path_tokens_for_code_check(ref)), ()


def _linked(code: str, ref: str) -> bool:
    any_of, markers = reference_tokens(ref)
    if not markers:
        return any(t in code for t in any_of)
    member, name = any_of
    if member in code:
        return True
    return name in code and any(m in code for m in markers)


def _skip_literal(code: str, i: int) -> int:
    """Return the index after a string/char literal starting at ``i``."""
    n = len(code)
    ch = code[i]
    if ch == "r" and i + 1 < n and code[i + 1] in '#"':
        k = i + 1
        hashes = 0
        while k < n and code[k] == "#":
            hashes += 1
            k += 1
        if k < n and code[k] == '"':
            end = code.find('"' + "#" * hashes, k + 1)
            return n if end == -1 else end + 1 + hashes
        return i + 1
    if ch == '"':
        j = i + 1
        while j < n:
            if code[j] == "\\":
                j += 2
                continue
            if code[j] == '"':
                return j + 1
            j += 1
        return n
    if ch == "'":
        if i + 2 < n and code[i + 1] == "\\":
            end = code.find("'", i + 2)
            if end != -1 and end - i <= 10:
                return end + 1
        if i + 2 < n and code[i + 2] == "'":
            return i + 3
    return i + 1


def split_rust_functions(code: str) -> tuple[dict[str, str], set[str]] | None:
    """Return ``({fn name: body}, {#[test] fn names})`` or ``None``.

    ``code`` must have comments removed and string literals kept. Bodies of
    functions with the same name are concatenated. ``None`` means the layout
    could not be parsed (unbalanced braces), so callers fail closed.
    """
    bodies: dict[str, str] = {}
    tests: set[str] = set()
    for m in _FN_RE.finditer(code):
        name = m.group(1)
        j = m.end()
        n = len(code)
        depth_paren = 0
        while j < n:
            c = code[j]
            if c in "\"'" or (c == "r" and j + 1 < n and code[j + 1] in '#"'):
                j = _skip_literal(code, j)
                continue
            # Parentheses and brackets (`-> [f64; C]`) may hold `;`.
            if c in "([":
                depth_paren += 1
            elif c in ")]":
                depth_paren -= 1
            elif c == ";" and depth_paren == 0:
                break
            elif c == "{" and depth_paren == 0:
                break
            j += 1
        if j >= n or code[j] != "{":
            continue
        start = j
        depth = 0
        while j < n:
            c = code[j]
            if c in "\"'" or (c == "r" and j + 1 < n and code[j + 1] in '#"'):
                j = _skip_literal(code, j)
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            return None
        bodies[name] = bodies.get(name, "") + code[start : j + 1]
        prefix = code[max(0, m.start() - 300) : m.start()]
        attrs = prefix.rsplit("}", 1)[-1]
        if _TEST_ATTR_RE.search(attrs):
            tests.add(name)
    return bodies, tests


def _closure(
    bodies: dict[str, str], calls: dict[str, set[str]], seed: set[str]
) -> set[str]:
    """Return functions in ``seed`` plus those calling them, transitively."""
    out = set(seed)
    changed = True
    while changed:
        changed = False
        for f, cs in calls.items():
            if f not in out and cs & out:
                out.add(f)
                changed = True
    return out


def _reachable(calls: dict[str, set[str]], start: str) -> set[str]:
    seen = {start}
    stack = [start]
    while stack:
        f = stack.pop()
        for g in calls.get(f, ()):
            if g in calls and g not in seen:
                seen.add(g)
                stack.append(g)
    return seen


def _const_block(code: str, name: str) -> str | None:
    """Return the initialiser of ``const``/``static``/``let`` ``name``."""
    m = re.search(
        rf"\b(?:const|static|let)\s+(?:mut\s+)?{re.escape(name)}\b", code
    )
    if m is None:
        return None
    n = len(code)
    j = m.end()
    depth = 0
    # Skip the type annotation (which may contain `;` inside brackets).
    while j < n and not (code[j] == "=" and depth == 0):
        if code[j] in "([{<":
            depth += 1
        elif code[j] in ")]}>":
            depth -= 1
        elif code[j] == ";" and depth == 0:
            return None
        j += 1
    if j >= n:
        return None
    start = j + 1
    j = start
    depth = 0
    while j < n:
        c = code[j]
        if c in "\"'":
            j = _skip_literal(code, j)
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == ";" and depth == 0:
            return code[start:j]
        j += 1
    return None


def _norm_hex(s: str) -> str:
    return s.replace("_", "").lower().rjust(16, "0")


def _ref_numbers(text: str) -> set[float]:
    """Return every decimal floating-point number printed in ``text``."""
    out: set[float] = set()
    for m in _REF_NUM_RE.finditer(text):
        try:
            out.add(float(m.group(0).replace("D", "e").replace("d", "e")))
        except ValueError:
            continue
    return out


def check_consts(
    ctx: _Ctx, entry: OracleEntry, code: str
) -> list[tuple[str, str]]:
    """Check that values embedded in ``entry.consts`` appear in the reference.

    Hex literals must appear as hex text; decimal floating-point literals
    must equal, as ``float64``, a number printed in the reference.
    """
    if not entry.consts:
        return []
    if _is_bundle_ref(entry.reference):
        return [
            (
                "consts",
                "embedded values need a committed repository reference, "
                "not a bundle member",
            )
        ]
    text = _ref_text(ctx, entry.reference)
    hay = re.sub(r"[_\s]", "", text.lower())
    numbers = _ref_numbers(text)
    fails: list[tuple[str, str]] = []
    for name in entry.consts:
        block = _const_block(code, name)
        if block is None:
            fails.append(("consts", f"constant {name!r} not found"))
            continue
        hexes = [_norm_hex(h) for h in _HEX_LIT_RE.findall(block)]
        decs = _DEC_LIT_RE.findall(_HEX_LIT_RE.sub(" ", block))
        if not hexes and not decs:
            fails.append(("consts", f"constant {name!r} has no literals"))
            continue
        missing = [f"0x{h}" for h in hexes if h not in hay]
        missing += [d for d in decs if float(d.replace("_", "")) not in numbers]
        if missing:
            fails.append(
                (
                    "consts",
                    f"{len(missing)} of {len(hexes) + len(decs)} literals of "
                    f"{name!r} are not in {entry.reference!r} "
                    f"(first: {missing[0]})",
                )
            )
    return fails


def evaluate_test(
    ctx: _Ctx,
    test_path: str,
    entries: tuple[OracleEntry, ...],
    targets: dict[str, set[str]],
) -> list[ClaimResult]:
    """Return one ``ClaimResult`` per listed routine of every entry.

    Args:
        ctx: Verification caches.
        test_path: Repository path of the test file.
        entries: The file's oracle entries.
        targets: ``routine id -> leaf Rust symbols`` from the manifest.
    """
    from flexmesa.testing import test_taxonomy

    full = ctx.repo_root / test_path
    results: list[ClaimResult] = []
    code = ""
    parsed: tuple[dict[str, str], set[str]] | None = None
    if full.is_file() and full.suffix == ".rs":
        code = test_taxonomy._read_non_comment_code(
            full, strip_strings=False, include_rust_common=True
        )
        parsed = split_rust_functions(code)
    for entry in entries:
        entry_fails = check_reference_entry(ctx, entry)
        # Embedded values are linked by the consts check itself.
        if code and not entry.consts and not _linked(code, entry.reference):
            entry_fails.append(
                ("linkage", f"test code does not name {entry.reference!r}")
            )
        entry_fails.extend(check_consts(ctx, entry, code))
        if entry.precision == "print":
            try:
                tol_ok = float(entry.tolerance) > 0.0
            except ValueError:
                tol_ok = False
            if not tol_ok:
                entry_fails.append(
                    ("precision", "print entry needs a positive tolerance")
                )
        for rid in entry.routines:
            fails = list(entry_fails)
            fails.extend(
                _scope_and_exact(
                    code, parsed, entry, targets.get(rid, set()), full
                )
            )
            results.append(
                ClaimResult(
                    test=test_path,
                    reference=entry.reference,
                    routine=rid,
                    precision=entry.precision,
                    failures=tuple(fails),
                )
            )
    return results


def _scope_and_exact(
    code: str,
    parsed: tuple[dict[str, str], set[str]] | None,
    entry: OracleEntry,
    syms: set[str],
    full: pathlib.Path,
) -> list[tuple[str, str]]:
    """Run the ``scope`` and (for exact entries) ``precision`` checks."""
    if full.suffix != ".rs" or not code:
        return [("scope", "only Rust test files can be scoped")]
    if parsed is None:
        return [("scope", "function layout could not be parsed")]
    if not syms:
        return [("scope", "routine has no Rust target in the manifest")]
    bodies, tests = parsed
    calls = {f: set(_CALL_RE.findall(b)) for f, b in bodies.items()}
    any_of, markers = reference_tokens(entry.reference)

    def names_ref(text: str) -> bool:
        if markers:
            member, name = any_of
            return member in text or (
                name in text and any(m in text for m in markers)
            )
        return any(t in text for t in any_of)

    # Module-level items (`const DATA: &str = include_str!(..)`) that load
    # the reference make every function that names them a reader.
    holders = [
        name
        for name in _ITEM_RE.findall(code)
        if (block := _const_block(code, name)) is not None and names_ref(block)
    ]

    def touches(body: str) -> bool:
        if any(c in body for c in entry.consts):
            return True
        if any(re.search(rf"\b{h}\b", body) for h in holders):
            return True
        return names_ref(body)

    # Upper-case targets are Rust constants: naming one is the "call".
    const_syms = [s for s in syms if s == s.upper()]
    callers = _closure(
        bodies,
        calls,
        {
            f
            for f, cs in calls.items()
            if cs & syms
            or any(re.search(rf"\b{c}\b", bodies[f]) for c in const_syms)
        },
    )
    readers = _closure(
        bodies, calls, {f for f, b in bodies.items() if touches(b)}
    )
    qualifying = sorted(tests & callers & readers)
    if not qualifying:
        return [
            (
                "scope",
                "no #[test] function reaches both a call to "
                f"{sorted(syms)} and the reference data",
            )
        ]
    if entry.precision != "exact":
        return []
    for t in qualifying:
        text = "\n".join(bodies[g] for g in _reachable(calls, t))
        if _has_exact_comparison(text):
            return []
    return [
        (
            "precision",
            "exact entry shows no 0-ULP comparison on a qualifying test path",
        )
    ]


def evaluate(
    repo_root: pathlib.Path,
    records: list[tuple[str, tuple[OracleEntry, ...]]],
    targets: dict[str, set[str]],
) -> list[ClaimResult]:
    """Evaluate the oracle entries of ``records`` (``(test path, entries)``)."""
    ctx = _Ctx(repo_root=repo_root)
    out: list[ClaimResult] = []
    for path, entries in records:
        if entries:
            out.extend(evaluate_test(ctx, path, entries, targets))
    return out


def verified_ids(results: list[ClaimResult]) -> tuple[set[str], set[str]]:
    """Return ``(exact ids, print-only ids)`` of verified routines."""
    exact = {r.routine for r in results if r.counts and r.precision == "exact"}
    printed = {
        r.routine for r in results if r.counts and r.precision == "print"
    }
    return exact, printed - exact


def cast_precision(value: str) -> Precision:
    """Return ``value`` as a ``Precision`` (validated by callers)."""
    return cast(Precision, value)


def summarize_claims(results: list[ClaimResult]) -> dict[str, Any]:
    """Return per-routine outcomes and the counts the report publishes.

    A routine counts when any of its claims is verified. A routine that does
    not count is reported with every failing claim. ``strict_only`` lists the
    routines whose claims fail only on ``scope`` or ``precision``, the two
    checks that depend on parsing the test source.
    """
    by_routine: dict[str, list[ClaimResult]] = {}
    for r in results:
        by_routine.setdefault(r.routine, []).append(r)
    exact, printed = verified_ids(results)
    dropped: dict[str, list[dict[str, Any]]] = {}
    strict_only: list[str] = []
    for rid, claims in sorted(by_routine.items()):
        if rid in exact or rid in printed:
            continue
        dropped[rid] = [
            {
                "test": c.test,
                "reference": c.reference,
                "precision": c.precision,
                "failures": [list(f) for f in c.failures],
            }
            for c in claims
        ]
        if all(
            {k for k, _ in c.failures} <= {"scope", "precision"} for c in claims
        ):
            strict_only.append(rid)
    return {
        "claims": len(results),
        "claims_verified": sum(r.counts for r in results),
        "routines_claimed": len(by_routine),
        "routines_exact": sorted(exact),
        "routines_print": sorted(printed),
        "routines_not_counted": dropped,
        "not_counted_only_on_scope_or_precision": strict_only,
    }


def render_report_json(results: list[ClaimResult]) -> str:
    """Render the claim summary as deterministic JSON."""
    return (
        json.dumps(summarize_claims(results), indent=1, sort_keys=True) + "\n"
    )


def render_report_markdown(results: list[ClaimResult]) -> str:
    """Render the oracle provenance report for ``notes/audit/``."""
    s = summarize_claims(results)
    lines = [
        "# Oracle provenance report",
        "",
        "<!-- GENERATED VIA scripts/dashboard/build_routine_table.py "
        "(DECISIONS #80). Do not edit by hand. -->",
        "",
        "Each `[[tests.oracles]]` entry of `benchmarks/dashboard/"
        "test_taxonomy.toml` is checked by `flexmesa.testing."
        "_oracle_provenance`: the reference is a committed MESA 26.4.1 "
        "artifact with a recipe, the test reads it, embedded values "
        "appear in it, a `#[test]` function reaches both the routine and the "
        "data, and exact entries compare at 0 ULP. A routine counts at the "
        "routine table's oracle levels only through a verified claim.",
        "",
        "| Measure | Value |",
        "| :--- | ---: |",
        f"| Claims (test, reference, routine) | {s['claims']:,} |",
        f"| Claims verified | {s['claims_verified']:,} |",
        f"| Routines claimed | {s['routines_claimed']:,} |",
        f"| Routines verified exact (0 ULP) | {len(s['routines_exact']):,} |",
        f"| Routines verified at print precision only | "
        f"{len(s['routines_print']):,} |",
        f"| Routines not counted | {len(s['routines_not_counted']):,} |",
        f"| Not counted only because of `scope` or `precision` | "
        f"{len(s['not_counted_only_on_scope_or_precision']):,} |",
        "",
        "## Routines claimed but not counted",
        "",
        "| Routine | Test | Reference | Failing checks |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for rid, claims in s["routines_not_counted"].items():
        for c in claims:
            fails = "; ".join(f"`{k}`: {m}" for k, m in c["failures"])
            lines.append(
                f"| `{rid}` | `{c['test']}` | `{c['reference']}` | "
                f"{fails.replace('|', '/')} |"
            )
    lines += [
        "",
        "## Verified routines",
        "",
        "| Routine | Level |",
        "| :--- | :--- |",
    ]
    lines.extend(f"| `{r}` | exact |" for r in s["routines_exact"])
    lines.extend(f"| `{r}` | print |" for r in s["routines_print"])
    lines.append("")
    return "\n".join(lines)
