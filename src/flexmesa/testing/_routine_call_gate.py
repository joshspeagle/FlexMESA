# SPDX-License-Identifier: GPL-3.0-or-later
"""Crate-tied Rust and PyO3 `_rust` call-site verification (`#605`).

Enforces `DECISIONS #65-#68` / `#563` Wave 3b-1 routine-claim rules:
- Rust bare calls count only when imported from or path-qualified by the
  target `flexmesa_<crate>` (including `use ...::sym as alias` and glob
  imports scoped to the mapped Rust module).
- Locally defined `fn sym` functions and `let`-bound locals (`let sym = ...`)
  in test files never satisfy bare-call claims.
- Struct field reads (`z.d1val1`, `o16.binding_energy`) never satisfy claims.
- Method calls (`x.sym(...)`) count only when the mapped Rust target symbol is
  a method on a type whose type (or constructor) is imported/referenced from the
  target crate.
- Calls inside `#[ignore]` / `#[cfg(any())]` Rust items or `@pytest.mark.skip`
  / `@unittest.skip` Python definitions do not satisfy claims.
- Python claims count only when the callee resolves in the AST to a name
  imported from `flexmesa._rust` (or via `_rust` / `flexmesa._rust`).

Known static-analysis limitations:
- `Type::meth` method calls verify that `Type` (or one of its constructors) is
  referenced from the target crate and `.meth(...)` is called somewhere in the
  file, without full Rust type inference tying the specific receiver
  expression to `Type`.
- Python AST matching does not perform control-flow reachability, so calls
  inside never-invoked unskipped helper functions or aliases bound to a locally
  reassigned `_rust` name are not distinguished at runtime.
"""

from __future__ import annotations

import ast
import pathlib
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

_RUST_USE_RE = re.compile(r"\b(?:pub(?:\([^)]*\))?\s+)?use\s+([^;]+);")
_RUST_FN_DEF_RE = re.compile(
    r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^;{}]*?>)?\s*\("
)
_RUST_LET_VAR_RE = re.compile(r"\blet\s+(?:mut\s+)?([A-Za-z_][A-Za-z0-9_]*)\b")
_RUST_DISABLED_ATTR_RE = re.compile(
    r"(?<!\w)#\[\s*(?:ignore(?:\s*=[^\]\n]*)?|cfg\s*\(\s*any\s*\(\s*\)\s*\))\s*\]"
)
_RUST_DOT_MEMBER_RE = re.compile(r"(?<!\.)\.\s*([A-Za-z_][A-Za-z0-9_]*)")
_RUST_METHOD_CALL_RE = re.compile(
    r"(?<!\.)\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:::<[^;{}]*?>)?\s*\("
)
_RUST_BARE_CALL_RE = re.compile(
    r"(?<![.:\w])([A-Za-z_][A-Za-z0-9_]*)\s*(?:::<[^;{}]*?>)?\s*\("
)
_RUST_BARE_REF_RE = re.compile(r"(?<![.:\w])([A-Za-z_][A-Za-z0-9_]*)\b")
_RUST_QUAL_REF_RE = re.compile(
    r"(?<![.\w])((?:[A-Za-z_][A-Za-z0-9_]*::)+)([A-Za-z_][A-Za-z0-9_]*)\b"
)
_RUST_USE_AS_RE = re.compile(r"^(.*?)\s+as\s+([A-Za-z_][A-Za-z0-9_]*)$", re.S)
_RUST_TARGET_FILE_RE = re.compile(
    r"^crates/(flexmesa-[a-z0-9_-]+)/src/(.+)\.rs$"
)
_RUST_TARGET_CRATE_RE = re.compile(r"^crates/(flexmesa-[a-z0-9_-]+)/")
_RUST_IMPL_RE = re.compile(
    r"^(?:pub\s+)?impl(?:<[^>]*>)?\s+"
    r"(?:[A-Za-z0-9_:<>]+\s+for\s+)?([A-Za-z_][A-Za-z0-9_]*)"
)
_RUST_SRC_FN_RE = re.compile(
    r"\b(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:unsafe\s+)?fn\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\b"
)
_RUST_SRC_CONST_RE = re.compile(
    r"\bpub(?:\([^)]*\))?\s+(?:const|static)\s+([A-Za-z_][A-Za-z0-9_]*)\b"
)
_RUST_TYPE_ALIAS_RE = re.compile(
    r"\bpub(?:\([^)]*\))?\s+type\s+([A-Za-z_][A-Za-z0-9_]*)"
    r"\s*(?:<[^>]*>)?\s*=\s*(?:[A-Za-z0-9_]+::)*([A-Za-z_][A-Za-z0-9_]*)"
)
_RUST_RET_TYPE_RE = re.compile(
    r"->\s*(?:Result<|Option<)?(?:[A-Za-z0-9_]+::)*([A-Z][A-Za-z0-9_]*)"
)
_RUST_MACRO_CALL_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*!\s*\(\s*([a-z_][A-Za-z0-9_]*)\b"
)


@dataclass
class RustSymbolInfo:
    """Definitions found for one symbol inside a Rust source file."""

    is_free_or_const: bool = False
    method_types: set[str] = field(default_factory=set)
    assoc_types: set[str] = field(default_factory=set)
    ctor_fns: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class RustCallEvidence:
    """Extracted crate imports and non-comment call/reference evidence."""

    referenced_crates: frozenset[str]
    sym_aliases: dict[str, dict[str, set[str]]]
    path_prefixes: dict[str, set[str]]
    globs: dict[str, set[tuple[str, ...]]]
    bare_calls: frozenset[str]
    bare_refs: frozenset[str]
    method_calls: frozenset[str]
    qual_refs: tuple[tuple[tuple[str, ...], str], ...]


@dataclass(frozen=True)
class PythonCallEvidence:
    """Extracted AST `flexmesa._rust` imports and `_rust` call targets."""

    imports_rust: bool
    called_syms: frozenset[str]


def _split_top_level_commas(expr: str) -> list[str]:
    """Split a `use` brace body by top-level commas only."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in expr:
        if ch == "{":
            depth += 1
            buf.append(ch)
        elif ch == "}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            item = "".join(buf).strip()
            if item:
                parts.append(item)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _expand_rust_use_tree(
    prefix: tuple[str, ...], expr: str
) -> list[tuple[tuple[str, ...], str]]:
    """Expand one Rust `use` expression into ``(full_path, local_name)``."""
    clean = expr.strip()
    if not clean:
        return []
    if clean.startswith("::"):
        clean = clean[2:].strip()
    brace_idx = clean.find("{")
    if brace_idx != -1:
        end_idx = clean.rfind("}")
        if end_idx <= brace_idx:
            return []
        head = clean[:brace_idx].strip().rstrip(":").strip()
        head_segs = tuple(p.strip() for p in head.split("::") if p.strip())
        inner = clean[brace_idx + 1 : end_idx]
        expanded: list[tuple[tuple[str, ...], str]] = []
        for sub in _split_top_level_commas(inner):
            expanded.extend(_expand_rust_use_tree(prefix + head_segs, sub))
        return expanded

    alias_m = _RUST_USE_AS_RE.match(clean)
    if alias_m:
        path_str = alias_m.group(1).strip()
        alias = alias_m.group(2).strip()
    else:
        path_str = clean
        alias = ""
    segs = tuple(p.strip() for p in path_str.split("::") if p.strip())
    full = prefix + segs
    if not full:
        return []
    if full[-1] == "self":
        full = full[:-1]
        if not full:
            return []
    if full[-1] == "*":
        return [(full, "*")]
    local = alias or full[-1]
    return [(full, local)]


def _strip_disabled_rust_items(code: str) -> str:
    """Strip Rust items annotated with `#[ignore]` or `#[cfg(any())]`."""
    out: list[str] = []
    pos = 0
    n = len(code)
    for match in _RUST_DISABLED_ATTR_RE.finditer(code):
        start = match.start()
        if start < pos:
            continue
        out.append(code[pos:start])
        out.append(" ")
        i = match.end()
        paren_bracket_depth = 0
        body_start = -1
        semi_end = -1
        while i < n:
            ch = code[i]
            if ch in "([":
                paren_bracket_depth += 1
            elif ch in ")]":
                if paren_bracket_depth > 0:
                    paren_bracket_depth -= 1
            elif paren_bracket_depth == 0:
                if ch == "{":
                    body_start = i
                    break
                if ch == ";":
                    semi_end = i + 1
                    break
                if ch == "}":
                    break
            i += 1
        if body_start != -1:
            brace_depth = 1
            j = body_start + 1
            while j < n and brace_depth > 0:
                if code[j] == "{":
                    brace_depth += 1
                elif code[j] == "}":
                    brace_depth -= 1
                j += 1
            pos = j
        elif semi_end != -1:
            pos = semi_end
        else:
            pos = match.end()
    out.append(code[pos:])
    return "".join(out)


def extract_rust_evidence(code: str) -> RustCallEvidence:
    """Parse stripped Rust test code into crate-scoped call evidence.

    Args:
        code: Rust test code already stripped of comments, string literals,
            and `fn test_*` declaration names.

    Returns:
        Structured `RustCallEvidence` preserving crate ownership, aliases,
        globs, method calls, qualified paths, and bare calls/references.
    """
    if not code:
        return RustCallEvidence(
            referenced_crates=frozenset(),
            sym_aliases={},
            path_prefixes={},
            globs={},
            bare_calls=frozenset(),
            bare_refs=frozenset(),
            method_calls=frozenset(),
            qual_refs=(),
        )
    referenced_crates: set[str] = set()
    sym_aliases: dict[str, dict[str, set[str]]] = {}
    path_prefixes: dict[str, set[str]] = {}
    globs: dict[str, set[tuple[str, ...]]] = {}

    for use_match in _RUST_USE_RE.finditer(code):
        for segs, local in _expand_rust_use_tree((), use_match.group(1)):
            if not segs or not segs[0].startswith("flexmesa_"):
                continue
            crate = segs[0]
            referenced_crates.add(crate)
            if local == "_":
                continue
            if local == "*":
                globs.setdefault(crate, set()).add(segs[:-1])
                continue
            orig = segs[-1]
            sym_aliases.setdefault(crate, {}).setdefault(orig, set()).add(local)
            if len(segs) == 1 or orig == orig.lower():
                path_prefixes.setdefault(crate, set()).add(local)

    body_raw = _strip_disabled_rust_items(_RUST_USE_RE.sub(" ", code))
    local_fn_defs = set(_RUST_FN_DEF_RE.findall(body_raw)) - {"_test_fn"}
    body = _RUST_FN_DEF_RE.sub("fn _def(", body_raw)
    local_vars = set(_RUST_LET_VAR_RE.findall(body))
    method_calls = frozenset(_RUST_METHOD_CALL_RE.findall(body))
    body_no_dots = _RUST_DOT_MEMBER_RE.sub(" ", body)

    bare_calls = (
        frozenset(_RUST_BARE_CALL_RE.findall(body_no_dots))
        - local_fn_defs
        - local_vars
    )
    bare_refs = (
        frozenset(_RUST_BARE_REF_RE.findall(body_no_dots))
        - local_vars
        - local_fn_defs
    )
    qual_refs_list: list[tuple[tuple[str, ...], str]] = []
    for qm in _RUST_QUAL_REF_RE.finditer(body_no_dots):
        pref = tuple(p for p in qm.group(1).split("::") if p)
        if not pref:
            continue
        if pref[0].startswith("flexmesa_"):
            referenced_crates.add(pref[0])
        qual_refs_list.append((pref, qm.group(2)))

    return RustCallEvidence(
        referenced_crates=frozenset(referenced_crates),
        sym_aliases=sym_aliases,
        path_prefixes=path_prefixes,
        globs=globs,
        bare_calls=bare_calls,
        bare_refs=bare_refs,
        method_calls=method_calls,
        qual_refs=tuple(qual_refs_list),
    )


def _is_skip_decorator(dec: ast.expr) -> bool:
    """Return True if decorator unconditionally marks a test/class skipped."""
    target = dec.func if isinstance(dec, ast.Call) else dec
    if not isinstance(target, ast.Attribute) or target.attr != "skip":
        return False
    val = target.value
    if isinstance(val, ast.Attribute) and val.attr == "mark":
        return isinstance(val.value, ast.Name) and val.value.id == "pytest"
    return isinstance(val, ast.Name) and val.id in {"mark", "unittest"}


def _iter_active_ast_nodes(root: ast.AST) -> Iterator[ast.AST]:
    """Yield AST nodes outside `@pytest.mark.skip` / `@unittest.skip` items."""
    stack: list[ast.AST] = [root]
    while stack:
        node = stack.pop()
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ) and any(_is_skip_decorator(d) for d in node.decorator_list):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def extract_python_evidence(path: pathlib.Path) -> PythonCallEvidence:
    """Resolve `flexmesa._rust` imports and `_rust`-rooted calls via AST."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return PythonCallEvidence(imports_rust=False, called_syms=frozenset())

    active_nodes = list(_iter_active_ast_nodes(tree))
    rust_mods: set[str] = set()
    fm_mods: set[str] = set()
    direct_syms: dict[str, str] = {}
    local_defs: set[str] = set()
    rust_glob = False
    imports_rust = False

    for node in active_nodes:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            local_defs.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "_rust":
                    rust_mods.add(alias.asname or "_rust")
                    imports_rust = True
                elif alias.name == "flexmesa._rust":
                    imports_rust = True
                    if alias.asname:
                        rust_mods.add(alias.asname)
                    else:
                        fm_mods.add("flexmesa")
                elif alias.name == "flexmesa":
                    fm_mods.add(alias.asname or "flexmesa")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "flexmesa":
                for alias in node.names:
                    if alias.name == "_rust":
                        rust_mods.add(alias.asname or "_rust")
                        imports_rust = True
            elif node.module == "flexmesa._rust":
                imports_rust = True
                for alias in node.names:
                    if alias.name == "*":
                        rust_glob = True
                    else:
                        direct_syms[alias.asname or alias.name] = alias.name

    called_syms: set[str] = set()
    for node in active_nodes:
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in local_defs:
                continue
            if func.id in direct_syms:
                called_syms.add(direct_syms[func.id])
            elif rust_glob:
                called_syms.add(func.id)
        elif isinstance(func, ast.Attribute):
            recv = func.value
            if isinstance(recv, ast.Name) and recv.id in rust_mods:
                called_syms.add(func.attr)
            elif (
                isinstance(recv, ast.Attribute)
                and recv.attr == "_rust"
                and isinstance(recv.value, ast.Name)
                and recv.value.id in fm_mods
            ):
                imports_rust = True
                called_syms.add(func.attr)

    return PythonCallEvidence(
        imports_rust=imports_rust,
        called_syms=frozenset(called_syms),
    )


def inspect_rust_source_symbols(
    repo_root: pathlib.Path,
    rel_path: str,
    scan_rust_fn: Callable[[str], str],
) -> dict[str, RustSymbolInfo]:
    """Classify symbols in a Rust source file as free/const/method/assoc."""
    src_path = repo_root / rel_path
    if not src_path.is_file():
        return {}

    alias_to_base: dict[str, str] = {}
    base_to_aliases: dict[str, set[str]] = {}
    alias_files = [src_path]
    cm = _RUST_TARGET_CRATE_RE.match(rel_path)
    if cm is not None:
        lib_rs = repo_root / "crates" / cm.group(1) / "src" / "lib.rs"
        if lib_rs.is_file() and lib_rs != src_path:
            alias_files.append(lib_rs)
    for af in alias_files:
        acode = scan_rust_fn(af.read_text(encoding="utf-8"))
        for tm in _RUST_TYPE_ALIAS_RE.finditer(acode):
            alias_name, base_name = tm.group(1), tm.group(2)
            alias_to_base[alias_name] = base_name
            base_to_aliases.setdefault(base_name, set()).add(alias_name)

    code = scan_rust_fn(src_path.read_text(encoding="utf-8"))
    lines = code.splitlines()
    info: dict[str, RustSymbolInfo] = {}
    ctors_by_type: dict[str, set[str]] = {}
    cur_impl: str | None = None
    brace_depth = 0
    impl_depth: int | None = None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if cur_impl is None:
            impl_m = _RUST_IMPL_RE.match(stripped)
            if impl_m is not None and not stripped.endswith(";"):
                cur_impl = impl_m.group(1)
                impl_depth = brace_depth

        fn_m = _RUST_SRC_FN_RE.search(stripped)
        if fn_m is not None:
            sym = fn_m.group(1)
            sig = " ".join(lines[idx : min(len(lines), idx + 4)])
            is_method = bool(
                re.search(
                    rf"fn\s+{re.escape(sym)}\s*(?:<[^>]*>)?\s*\("
                    r"\s*(?:&(?:mut\s+)?self|mut\s+self|self)\b",
                    sig,
                )
            )
            rec = info.setdefault(sym, RustSymbolInfo())
            if cur_impl is not None:
                types = {cur_impl} | base_to_aliases.get(cur_impl, set())
                if is_method:
                    rec.method_types.update(types)
                else:
                    rec.assoc_types.update(types)
            else:
                rec.is_free_or_const = True
                ret_m = _RUST_RET_TYPE_RE.search(sig)
                if ret_m is not None:
                    rt = ret_m.group(1)
                    base = alias_to_base.get(rt, rt)
                    for target_t in {rt, base} | base_to_aliases.get(
                        base, set()
                    ):
                        ctors_by_type.setdefault(target_t, set()).add(sym)
        elif cur_impl is None:
            cm_const = _RUST_SRC_CONST_RE.search(stripped)
            if cm_const is not None:
                sym = cm_const.group(1)
                info.setdefault(sym, RustSymbolInfo()).is_free_or_const = True
        else:
            mac_m = _RUST_MACRO_CALL_RE.match(stripped)
            if mac_m is not None:
                sym = mac_m.group(1)
                types = {cur_impl} | base_to_aliases.get(cur_impl, set())
                info.setdefault(sym, RustSymbolInfo()).method_types.update(
                    types
                )

        brace_depth += line.count("{") - line.count("}")
        if (
            cur_impl is not None
            and impl_depth is not None
            and brace_depth <= impl_depth
            and "}" in line
        ):
            cur_impl = None
            impl_depth = None

    for rec in info.values():
        for tname in rec.method_types:
            rec.ctor_fns.update(ctors_by_type.get(tname, set()))
    return info


def _split_target(rust_target: str) -> tuple[str, str]:
    """Split a manifest `routine_map` value into `(file_path, symbols)`."""
    if "::" in rust_target:
        rel_fm, syms_raw = rust_target.split("::", 1)
        return (rel_fm.strip(), syms_raw.strip())
    return ("", rust_target.strip())


def parse_target_symbols(rust_target: str) -> set[str]:
    """Return concrete leaf symbols from a manifest `routine_map` target."""
    _, sym_part = _split_target(rust_target)
    out: set[str] = set()
    for item in sym_part.split("+"):
        clean = item.strip()
        if not clean:
            continue
        leaf = clean.split("::")[-1].strip()
        if leaf:
            out.add(leaf)
    return out


def _type_present_for_crate(
    ev: RustCallEvidence,
    crate: str,
    mod_path: tuple[str, ...],
    tname: str,
) -> bool:
    """Return True if type `tname` from `crate` is referenced in test code."""
    for local in ev.sym_aliases.get(crate, {}).get(tname, set()):
        if local in ev.bare_refs or any(
            local in pref for pref, _ in ev.qual_refs
        ):
            return True
    prefixes = ev.path_prefixes.get(crate, set())
    for pref, leaf in ev.qual_refs:
        if (pref[0] == crate or pref[0] in prefixes) and (
            leaf == tname or tname in pref
        ):
            return True
    for glob_pref in ev.globs.get(crate, set()):
        if len(glob_pref) <= len(mod_path) and mod_path[: len(glob_pref)] == (
            glob_pref
        ):
            if tname in ev.bare_refs or any(
                tname in pref for pref, _ in ev.qual_refs
            ):
                return True
    return False


def _free_symbol_matches(
    ev: RustCallEvidence,
    crate: str,
    mod_path: tuple[str, ...],
    sym: str,
) -> bool:
    """Return True if free fn / const `sym` from `crate` is called/used."""
    for local in ev.sym_aliases.get(crate, {}).get(sym, set()):
        if local in ev.bare_calls or local in ev.bare_refs:
            return True
    for glob_pref in ev.globs.get(crate, set()):
        if len(glob_pref) <= len(mod_path) and mod_path[: len(glob_pref)] == (
            glob_pref
        ):
            if sym in ev.bare_calls or sym in ev.bare_refs:
                return True
    prefixes = ev.path_prefixes.get(crate, set())
    for pref, leaf in ev.qual_refs:
        if leaf == sym and (pref[0] == crate or pref[0] in prefixes):
            return True
    return False


def _assoc_symbol_matches(
    ev: RustCallEvidence,
    crate: str,
    mod_path: tuple[str, ...],
    cand_types: set[str],
    sym: str,
) -> bool:
    """Return True if `Type::sym` from `crate` is called/referenced."""
    prefixes = ev.path_prefixes.get(crate, set())
    globs = ev.globs.get(crate, set())
    for pref, leaf in ev.qual_refs:
        if leaf != sym:
            continue
        for tname in cand_types:
            t_locals = ev.sym_aliases.get(crate, {}).get(tname, set())
            if pref[-1] in t_locals:
                return True
            if (pref[0] == crate or pref[0] in prefixes) and pref[-1] == tname:
                return True
            if pref[-1] == tname and any(
                len(g) <= len(mod_path) and mod_path[: len(g)] == g
                for g in globs
            ):
                return True
    return False


def _method_symbol_matches(
    ev: RustCallEvidence,
    crate: str,
    mod_path: tuple[str, ...],
    sinfo: RustSymbolInfo,
    sym: str,
) -> bool:
    """Return True if `x.sym(...)` matches a method on a `crate` type."""
    if sym not in ev.method_calls or not sinfo.method_types:
        return False
    for tname in sinfo.method_types:
        if _type_present_for_crate(ev, crate, mod_path, tname):
            return True
    for ctor in sinfo.ctor_fns:
        if _free_symbol_matches(ev, crate, mod_path, ctor):
            return True
    return False


def check_rust_routine_claim(
    ev: RustCallEvidence,
    repo_root: pathlib.Path,
    module: str,
    rust_target: str,
    scan_rust_fn: Callable[[str], str],
    source_cache: dict[str, dict[str, RustSymbolInfo]],
) -> tuple[str | None, bool]:
    """Check one claimed `routine_map` target against `RustCallEvidence`.

    Returns ``(missing_crate, matched)`` where ``missing_crate`` is the
    crate identifier when unreferenced in the test file.
    """
    rel_fm, syms_raw = _split_target(rust_target)
    file_m = _RUST_TARGET_FILE_RE.match(rel_fm) if rel_fm else None
    if file_m is not None:
        crate = file_m.group(1).replace("-", "_")
        parts = tuple(
            s
            for s in file_m.group(2).split("/")
            if s and s not in {"lib", "mod"}
        )
        mod_path: tuple[str, ...] = (crate, *parts)
    else:
        crate = f"flexmesa_{module}"
        mod_path = (crate,)

    missing_crate = None if crate in ev.referenced_crates else crate

    if not rel_fm:
        file_syms: dict[str, RustSymbolInfo] = {}
    else:
        if rel_fm not in source_cache:
            source_cache[rel_fm] = inspect_rust_source_symbols(
                repo_root, rel_fm, scan_rust_fn
            )
        file_syms = source_cache[rel_fm]

    matched = False
    for raw_item in syms_raw.split("+"):
        item = raw_item.strip()
        if not item:
            continue
        if "::" in item:
            owner_type, sym = (s.strip() for s in item.split("::", 1))
            sinfo = RustSymbolInfo(
                is_free_or_const=False,
                method_types={owner_type},
                assoc_types={owner_type},
            )
        else:
            sym = item
            sinfo = file_syms.get(sym, RustSymbolInfo(is_free_or_const=True))

        is_free = sinfo.is_free_or_const or (
            not sinfo.method_types and not sinfo.assoc_types
        )
        if is_free and _free_symbol_matches(ev, crate, mod_path, sym):
            matched = True
            break
        cand_types = sinfo.assoc_types | sinfo.method_types
        if cand_types and _assoc_symbol_matches(
            ev, crate, mod_path, cand_types, sym
        ):
            matched = True
            break
        if sinfo.method_types and _method_symbol_matches(
            ev, crate, mod_path, sinfo, sym
        ):
            matched = True
            break

    return (missing_crate, matched)
