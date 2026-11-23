#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare FlexMESA Rust control defaults against MESA 26.4.1 ``*.defaults``.

Generates ``notes/audit/defaults_diff_26_4_1.md`` for ``#141`` (feeding
``#149``) from the unpacked ``mesa-test-inputs-v2`` bundle and the Rust
``impl Default`` blocks across ``crates/flexmesa-{star,binary,astero,eos,kap}``.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import math
import os
import pathlib
import re
import struct
import sys
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_fetch_module() -> Any:
    try:
        from flexmesa.data import fetch

        return fetch
    except ImportError:
        fetch_path = REPO_ROOT / "src" / "flexmesa" / "data" / "fetch.py"
        spec = importlib.util.spec_from_file_location(
            "flexmesa_data_fetch_standalone", fetch_path
        )
        if spec is None or spec.loader is None:
            raise
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod


fetch = _load_fetch_module()

ENUM_VALUE_MAP: dict[str, str] = {
    "MltOption::Cox": "Cox",
    "MltOption::Henyey": "Henyey",
    "MltOption::Mihalas": "Mihalas",
    "MltOption::BohmVitense": "Bohm-Vitense",
    "MltOption::Tdc": "TDC",
    "MltOption::None": "none",
    "SemiconvectionOption::Langer85GradTIsGradr": (
        "Langer_85 mixing; gradT = gradr"
    ),
    "SemiconvectionOption::Langer85InterpolateGradT": (
        "Langer_85 mixing; interpolate gradT"
    ),
    "ThermohalineOption::Kippenhahn": "Kippenhahn",
    "ThermohalineOption::TraxlerGaraudStellmach11": (
        "Traxler_Garaud_Stellmach_11"
    ),
    "ThermohalineOption::BrownGaraudStellmach13": "Brown_Garaud_Stellmach_13",
    "ScreeningMode::None": "",
    "ScreeningMode::Weak": "weak",
    "ScreeningMode::Extended": "extended",
    "ScreeningMode::Salpeter": "salpeter",
    "ScreeningMode::Chugunov": "chugunov",
    "AtmOption::TTau": "T_tau",
    "AtmOption::Table": "table",
    "AtmOption::IrradiatedGrey": "irradiated_grey",
    "AtmTTauRelation::Eddington": "Eddington",
    "AtmTTauRelation::SolarHopf": "solar_Hopf",
    "AtmTTauRelation::KrishnaSwamy": "Krishna_Swamy",
    "AtmTTauRelation::TrampedachSolar": "Trampedach_solar",
    "AtmTTauOpacity::Fixed": "fixed",
    "AtmTTauOpacity::Iterated": "iterated",
    "AtmTTauOpacity::Varying": "varying",
    "AtmIrradiatedOpacity::Fixed": "fixed",
    "AtmIrradiatedOpacity::Iterated": "iterated",
    "EnergyEqnOption::Dedt": "dedt",
    "EnergyEqnOption::EpsGrav": "eps_grav",
    "KapOption::Gn93": "gn93",
    "KapOption::Gs98": "gs98",
    "KapOption::A09": "a09",
    "KapOption::OpGs98": "OP_gs98",
    "KapOption::OpA09": "OP_a09_nans_removed_by_hand",
    "KapCoOption::Gn93Co": "gn93_co",
    "KapCoOption::Gs98Co": "gs98_co",
    "KapCoOption::A09Co": "a09_co",
    "KapLowTOption::LowTFa05Gs98": "lowT_fa05_gs98",
    "KapLowTOption::LowTFa05Gn93": "lowT_fa05_gn93",
    "KapLowTOption::LowTFa05A09p": "lowT_fa05_a09p",
    "KapLowTOption::LowTFreedman11": "lowT_Freedman11",
    "KapLowTOption::KapCn": "kapCN",
    "KapLowTOption::Aesopus": "AESOPUS",
    "SkyeSolidMixingRule::Pc": "PC",
    "SkyeSolidMixingRule::Ogata": "Ogata",
}

FIELD_ALIASES: dict[str, str] = {
    "m1_msun": "m1",
    "m2_msun": "m2",
    "feh_param": "first_feh",
    "use_type2": "use_type2_opacities",
    "use_zbase_for_type1": "use_zbase_for_type1",
    "kap_prefix": "kap_file_prefix",
    "co_prefix": "kap_co_prefix",
    "low_t_prefix": "kap_lowt_prefix",
    "kap_option": "kap_file_prefix",
    "kap_co_option": "kap_co_prefix",
    "kap_low_t_option": "kap_lowt_prefix",
    "kap_blend_log_t_upper_bdy": "kap_blend_logt_upper_bdy",
    "kap_blend_log_t_lower_bdy": "kap_blend_logt_lower_bdy",
    "blend_logt_upper": "kap_blend_logt_upper_bdy",
    "blend_logt_lower": "kap_blend_logt_lower_bdy",
    "include_electron_conduction": "use_simple_es_for_kap",
    "column_depth_for_irradiation": "irrad_col_depth",
}

ALLOW_LIST: frozenset[tuple[str, str]] = frozenset()

NAMELIST_SPECS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "star &controls",
        (
            "star/defaults/controls.defaults",
            "star/defaults/controls_dev.defaults",
        ),
        (
            "crates/flexmesa-star/src/controls_defaults.rs",
            "crates/flexmesa-star/src/controls_diffusion.rs",
            "crates/flexmesa-star/src/controls_output.rs",
            "crates/flexmesa-star/src/controls_timestep_stop.rs",
            "crates/flexmesa-star/src/controls_mix_wind.rs",
            "crates/flexmesa-star/src/mesh_adjust.rs",
            "crates/flexmesa-star/src/overshoot.rs",
            "crates/flexmesa-star/src/rsp_config.rs",
        ),
    ),
    (
        "star &star_job",
        (
            "star/defaults/star_job.defaults",
            "star/defaults/star_job_dev.defaults",
        ),
        ("crates/flexmesa-star/src/job_controls.rs",),
    ),
    (
        "binary &binary_controls + &binary_job",
        (
            "binary/defaults/binary_controls.defaults",
            "binary/defaults/binary_job.defaults",
        ),
        (
            "crates/flexmesa-binary/src/inlist.rs",
            "crates/flexmesa-binary/src/timestep.rs",
            "crates/flexmesa-binary/src/mdot_implicit.rs",
        ),
    ),
    (
        "astero &astero_search_controls",
        ("astero/defaults/astero_search.defaults",),
        (
            "crates/flexmesa-astero/src/inlist_astero.rs",
            "crates/flexmesa-astero/src/ratios_chi2.rs",
        ),
    ),
    (
        "eos &eos",
        ("eos/defaults/eos.defaults",),
        (
            "crates/flexmesa-eos/src/blend_dt.rs",
            "crates/flexmesa-eos/src/helm/eval.rs",
            "crates/flexmesa-eos/src/pc/mod.rs",
            "crates/flexmesa-eos/src/skye/mod.rs",
            "crates/flexmesa-eos/src/cms/config.rs",
            "crates/flexmesa-eos/src/tables_4d/blend_alfa.rs",
        ),
    ),
    (
        "kap &kap",
        ("kap/defaults/kap.defaults",),
        ("crates/flexmesa-kap/src/types.rs",),
    ),
)


@dataclasses.dataclass(frozen=True)
class UpstreamDefault:
    """Parsed scalar or indexed entry from a MESA ``*.defaults`` file."""

    name: str
    rel_file: str
    line: int
    raw_value: str
    parsed: bool | float | str


@dataclasses.dataclass(frozen=True)
class RustDefault:
    """Parsed scalar field initializer inside an ``impl Default`` block."""

    field: str
    rel_file: str
    line: int
    raw_expr: str
    parsed: bool | float | str | None
    is_option_none: bool


@dataclasses.dataclass(frozen=True)
class UnparsedRustField:
    """Field inside ``impl Default`` whose initializer is non-scalar."""

    field: str
    rel_file: str
    line: int
    raw_expr: str
    reason: str


@dataclasses.dataclass(frozen=True)
class NamelistCoverage:
    """Per-namelist upstream denominator and Rust comparison counts."""

    label: str
    upstream_scalars: int
    upstream_array_only: int
    matched: int
    agree: int
    val_mismatch: int
    none_disabled: int
    skipped_unparseable_rust: int
    skipped_no_rust_field: int


@dataclasses.dataclass(frozen=True)
class DefaultsCoverageSummary:
    """Aggregate denominator counts across all audited MESA namelists."""

    by_namelist: tuple[NamelistCoverage, ...]
    upstream_scalars: int
    upstream_array_only: int
    matched: int
    agree: int
    val_mismatch: int
    none_disabled: int
    skipped_unparseable_rust: int
    skipped_no_rust_field: int


def _strip_fortran_comment(line: str) -> str:
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


def _f32_to_f64(val: float) -> float:
    return float(struct.unpack("f", struct.pack("f", val))[0])


def _parse_fortran_scalar(raw: str) -> bool | float | str:
    tok = raw.strip()
    low = tok.lower()
    if low in (".true.", "true"):
        return True
    if low in (".false.", "false"):
        return False
    if (tok.startswith("'") and tok.endswith("'")) or (
        tok.startswith('"') and tok.endswith('"')
    ):
        return tok[1:-1]
    is_dp = bool(re.search(r"[dD]", tok)) or tok.endswith("_dp")
    num_tok = re.sub(r"[dD]", "e", re.sub(r"_dp$", "", tok))
    try:
        parsed = float(num_tok)
    except ValueError:
        return tok
    if not is_dp and ("." in num_tok or "e" in num_tok.lower()):
        try:
            return _f32_to_f64(parsed)
        except OverflowError:
            return parsed
    return parsed


def scan_mesa_defaults_file(
    mesa_root: pathlib.Path,
    rel_file: str,
) -> tuple[dict[str, UpstreamDefault], frozenset[str]]:
    """Parse scalar/1-slot defaults and list multi-subscript array-only keys."""
    fpath = mesa_root / rel_file
    assign_re = re.compile(
        r"^\s*([A-Za-z_][A-Za-z0-9_]*)(?:\(([^)]*)\))?\s*=\s*(.+?)\s*$"
    )
    entries: dict[str, UpstreamDefault] = {}
    is_slice: dict[str, bool] = {}
    all_keys: set[str] = set()
    for lno, raw_line in enumerate(
        fpath.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        code = _strip_fortran_comment(raw_line).strip()
        if not code:
            continue
        match = assign_re.match(code)
        if match is None:
            continue
        name, sub, rhs = match.group(1), match.group(2), match.group(3)
        key = name.lower()
        all_keys.add(key)
        sub_clean = sub.strip() if sub is not None else None
        if sub_clean is not None and sub_clean not in ("1", ":", "1:"):
            continue
        cur_is_slice = sub_clean in (":", "1:")
        should_set = (
            key not in entries
            or sub_clean is None
            or (is_slice.get(key, False) and not cur_is_slice)
        )
        if should_set:
            entries[key] = UpstreamDefault(
                name=name,
                rel_file=rel_file,
                line=lno,
                raw_value=rhs,
                parsed=_parse_fortran_scalar(rhs),
            )
            is_slice[key] = cur_is_slice
    array_only = frozenset(all_keys - set(entries))
    return entries, array_only


def parse_mesa_defaults_file(
    mesa_root: pathlib.Path,
    rel_file: str,
) -> dict[str, UpstreamDefault]:
    """Parse scalar defaults from ``mesa_root / rel_file``."""
    entries, _ = scan_mesa_defaults_file(mesa_root, rel_file)
    return entries


def _parse_rust_expr(expr: str) -> tuple[bool | float | str | None, bool]:
    tok = expr.strip().rstrip(",")
    if tok == "None":
        return None, True
    if tok.startswith("Some(") and tok.endswith(")"):
        inner, _ = _parse_rust_expr(tok[5:-1])
        return inner, False
    vec_rep_m = re.match(r"^vec!\[([^;\[\]]+);\s*[^;\[\]]+\]$", tok)
    if vec_rep_m is not None:
        return _parse_rust_expr(vec_rep_m.group(1))
    vec_m = re.match(r"^vec!\[([^\[\]]+)\]$", tok)
    if vec_m is not None:
        inner_tok = vec_m.group(1).strip()
        if inner_tok:
            first_elem = inner_tok.split(",", 1)[0].strip()
            return _parse_rust_expr(first_elem)
    arr_rep_m = re.match(r"^\[([^;\[\]]+);\s*[^;\[\]]+\]$", tok)
    if arr_rep_m is not None:
        return _parse_rust_expr(arr_rep_m.group(1))
    from_fn_m = re.match(r"^std::array::from_fn\(\|_\|(.+)\)$", tok)
    if from_fn_m is not None:
        return _parse_rust_expr(from_fn_m.group(1))
    if tok in ("true", "false"):
        return (tok == "true"), False
    if tok == "String::new()":
        return "", False
    str_m = re.match(r'^(?:String::from\("([^"]*)"\)|"([^"]*)")$', tok)
    if str_m is not None:
        val = str_m.group(1) if str_m.group(1) is not None else str_m.group(2)
        return val or "", False
    if tok in ENUM_VALUE_MAP:
        return ENUM_VALUE_MAP[tok], False
    fixed_m = re.match(r"^AsteroSearchParam::fixed\(([^)]+)\)$", tok)
    if fixed_m is not None:
        return _parse_rust_expr(fixed_m.group(1))
    f32_m = re.match(r"^f64::from\(([^)]+)_f32\)$", tok)
    if f32_m is not None:
        raw_f32 = f32_m.group(1).strip().replace("_", "")
        try:
            return _f32_to_f64(float(raw_f32)), False
        except ValueError:
            return None, False
    num_clean = tok.replace("_", "")
    try:
        return float(num_clean), False
    except ValueError:
        return None, False


def _classify_unparsed_rust_expr(expr: str) -> str:
    tok = expr.strip().rstrip(",")
    if tok == "Vec::new()":
        return "empty Vec initializer"
    if tok.startswith("<local:"):
        return "local variable initializer"
    if tok.startswith("["):
        return "non-uniform array initializer"
    return "computed/non-scalar Rust initializer"


STRUCT_PREFIX_MAP: dict[str, str] = {
    "RspConfig": "rsp_",
}
UNPREFIXED_STRUCT_FIELDS: frozenset[tuple[str, str]] = frozenset()
SKIP_DEFAULT_STRUCTS: frozenset[str] = frozenset({"Rsp2Config"})


def scan_rust_defaults(
    rel_files: tuple[str, ...],
) -> tuple[dict[str, RustDefault], dict[str, UnparsedRustField]]:
    """Extract parsed scalar defaults and unparsed/non-scalar Rust fields."""
    parsed_out: dict[str, RustDefault] = {}
    unparsed_out: dict[str, UnparsedRustField] = {}
    impl_re = re.compile(r"impl\s+Default\s+for\s+([A-Za-z_][A-Za-z0-9_]*)")
    field_kv_re = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$")
    field_shorthand_re = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*$")

    for rel in rel_files:
        fpath = REPO_ROOT / rel
        lines = fpath.read_text(encoding="utf-8").splitlines()
        in_default = False
        in_self_lit = False
        cur_struct = ""
        brace_depth = 0
        self_depth = 0
        for lno, raw_line in enumerate(lines, 1):
            code = raw_line.split("//", 1)[0]
            if not in_default:
                im = impl_re.search(code)
                if im is not None:
                    cur_struct = im.group(1)
                    in_default = True
                    in_self_lit = False
                    brace_depth = code.count("{") - code.count("}")
                continue
            prev_depth = brace_depth
            brace_depth += code.count("{") - code.count("}")
            if cur_struct not in SKIP_DEFAULT_STRUCTS:
                if not in_self_lit and re.match(
                    rf"^\s*(?:Self|{re.escape(cur_struct)})\s*\{{", code
                ):
                    in_self_lit = True
                    self_depth = brace_depth
                elif in_self_lit:
                    if prev_depth == self_depth:
                        m_kv = field_kv_re.match(code)
                        m_sh = field_shorthand_re.match(code)
                        fname: str | None = None
                        expr = ""
                        if m_kv is not None and m_kv.group(1) not in (
                            "let",
                            "mut",
                            "fn",
                            "Self",
                        ):
                            fname = m_kv.group(1)
                            expr = m_kv.group(2).strip().rstrip(",")
                            if not expr and lno < len(lines):
                                expr = (
                                    lines[lno]
                                    .split("//", 1)[0]
                                    .strip()
                                    .rstrip(",")
                                )
                        elif m_sh is not None:
                            fname = m_sh.group(1)
                            expr = f"<local:{fname}>"
                        if fname is not None:
                            prefix = (
                                ""
                                if (cur_struct, fname.lower())
                                in UNPREFIXED_STRUCT_FIELDS
                                else STRUCT_PREFIX_MAP.get(cur_struct, "")
                            )
                            low = f"{prefix}{fname.lower()}"
                            val, is_none = _parse_rust_expr(expr)
                            if val is None and not is_none:
                                urec = UnparsedRustField(
                                    field=fname,
                                    rel_file=rel,
                                    line=lno,
                                    raw_expr=expr,
                                    reason=_classify_unparsed_rust_expr(expr),
                                )
                                unparsed_out.setdefault(low, urec)
                                if low in FIELD_ALIASES:
                                    unparsed_out.setdefault(
                                        FIELD_ALIASES[low], urec
                                    )
                            else:
                                rec = RustDefault(
                                    field=fname,
                                    rel_file=rel,
                                    line=lno,
                                    raw_expr=expr,
                                    parsed=val,
                                    is_option_none=is_none,
                                )
                                parsed_out.setdefault(low, rec)
                                if low in FIELD_ALIASES:
                                    parsed_out.setdefault(
                                        FIELD_ALIASES[low], rec
                                    )
                    if brace_depth < self_depth:
                        in_self_lit = False
            if brace_depth <= 0:
                in_default = False
                in_self_lit = False
                cur_struct = ""
    return parsed_out, unparsed_out


def parse_rust_defaults(rel_files: tuple[str, ...]) -> dict[str, RustDefault]:
    """Extract scalar field initializers inside ``impl Default`` blocks."""
    parsed_out, _ = scan_rust_defaults(rel_files)
    return parsed_out


def _is_inactive_upstream_sentinel(
    name: str,
    uval: bool | float | str,
) -> bool:
    if not isinstance(uval, float):
        return False
    if abs(uval) >= 0.99e99 or uval <= -0.99e99:
        return True
    if uval < 0.0 and (
        name.startswith(("max_", "min_", "stop_", "num_"))
        or name.endswith(("_limit", "_target", "_max", "_min"))
    ):
        return True
    return False


def _values_equal(
    name: str,
    uval: bool | float | str,
    rrec: RustDefault,
) -> tuple[bool, str]:
    if rrec.is_option_none:
        if _is_inactive_upstream_sentinel(name, uval):
            return True, ""
        return False, "active upstream limit disabled as None"
    rval = rrec.parsed
    if isinstance(uval, bool) and isinstance(rval, bool):
        return (uval == rval), ("boolean mismatch" if uval != rval else "")
    if isinstance(uval, float) and isinstance(rval, float):
        if uval == 0.0 and rval == 0.0:
            return True, ""
        scale = max(1.0, abs(uval), abs(rval))
        ok = math.isclose(uval, rval, rel_tol=1.0e-12, abs_tol=1.0e-15 * scale)
        return ok, ("" if ok else "numeric mismatch")
    if isinstance(uval, str) and isinstance(rval, str):
        ok = uval.strip().lower() == rval.strip().lower()
        return ok, ("" if ok else "string/option mismatch")
    return False, "type mismatch"


def evaluate_defaults_parity(
    mesa_root: pathlib.Path,
) -> tuple[str, int, int, list[tuple[str, str, str]], DefaultsCoverageSummary]:
    """Build report and return parity counts plus full denominator summary."""
    ref = fetch.load_upstream_reference()
    lines: list[str] = [
        (
            "<!-- GENERATED BY `python scripts/audit/diff_mesa_defaults.py` "
            "(#141). DO NOT EDIT BY HAND. -->"
        ),
        "# FlexMESA vs MESA `26.4.1` Control Defaults Diff",
        "",
        (
            f"Generated mechanically against **{ref.name}** "
            f"(git tag `{ref.git_tag}`, commit `{ref.git_commit[:7]}`, "
            f"Zenodo `{ref.zenodo_record}`) for `#141` "
            "(feeds `#149` / Audit B3)."
        ),
        "",
        "## Summary by Namelist",
        "",
        (
            "| Namelist | Upstream `26.4.1` Scalars | Matched & Compared "
            "| Agree | Value Mismatch | Active Limit Disabled (`None`) "
            "| Skipped (Non-Scalar/Computed Rust) | Skipped (No Rust Field) |"
        ),
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    sections_md: list[str] = []
    skipped_rows: list[tuple[str, UpstreamDefault, UnparsedRustField]] = []
    per_namelist: list[NamelistCoverage] = []
    total_up = 0
    total_array_only = 0
    total_matched = 0
    total_agree = 0
    total_val_diff = 0
    total_none_diff = 0
    total_unparsed = 0
    total_no_field = 0
    unallowed: list[tuple[str, str, str]] = []

    for label, up_files, rust_files in NAMELIST_SPECS:
        up_map: dict[str, UpstreamDefault] = {}
        array_only_keys: set[str] = set()
        for uf in up_files:
            entries, arr_only = scan_mesa_defaults_file(mesa_root, uf)
            up_map.update(entries)
            array_only_keys.update(arr_only)
        array_only_keys -= set(up_map)

        rust_map, unparsed_map = scan_rust_defaults(rust_files)
        rust_collapsed: dict[str, RustDefault] = {}
        for rk, rv in rust_map.items():
            rust_collapsed.setdefault(rk.replace("_", ""), rv)
        unparsed_collapsed: dict[str, UnparsedRustField] = {}
        for uk, uv in unparsed_map.items():
            unparsed_collapsed.setdefault(uk.replace("_", ""), uv)

        n_up = len(up_map)
        n_arr_only = len(array_only_keys)
        n_matched = 0
        n_agree = 0
        n_val_diff = 0
        n_none_diff = 0
        n_unparsed = 0
        n_no_field = 0
        mismatches: list[tuple[UpstreamDefault, RustDefault, str]] = []

        for key in sorted(up_map):
            u_ent = up_map[key]
            r_ent = rust_map.get(key)
            if r_ent is None:
                r_ent = rust_collapsed.get(key.replace("_", ""))
            if r_ent is None:
                u_rs = unparsed_map.get(key)
                if u_rs is None:
                    u_rs = unparsed_collapsed.get(key.replace("_", ""))
                if u_rs is not None:
                    n_unparsed += 1
                    skipped_rows.append((label, u_ent, u_rs))
                else:
                    n_no_field += 1
                continue
            n_matched += 1
            ok, reason = _values_equal(key, u_ent.parsed, r_ent)
            if ok:
                n_agree += 1
            else:
                if reason.startswith("active upstream"):
                    n_none_diff += 1
                else:
                    n_val_diff += 1
                mismatches.append((u_ent, r_ent, reason))
                if (label, key) not in ALLOW_LIST:
                    unallowed.append((label, u_ent.name, reason))

        per_namelist.append(
            NamelistCoverage(
                label=label,
                upstream_scalars=n_up,
                upstream_array_only=n_arr_only,
                matched=n_matched,
                agree=n_agree,
                val_mismatch=n_val_diff,
                none_disabled=n_none_diff,
                skipped_unparseable_rust=n_unparsed,
                skipped_no_rust_field=n_no_field,
            )
        )
        total_up += n_up
        total_array_only += n_arr_only
        total_matched += n_matched
        total_agree += n_agree
        total_val_diff += n_val_diff
        total_none_diff += n_none_diff
        total_unparsed += n_unparsed
        total_no_field += n_no_field

        lines.append(
            f"| {label} | {n_up} | {n_matched} | {n_agree} | "
            f"{n_val_diff} | {n_none_diff} | {n_unparsed} | {n_no_field} |"
        )

        sections_md.extend(
            [
                "",
                f"## {label}",
                "",
            ]
        )
        if not mismatches:
            sections_md.append(
                f"All `{n_matched}` matched scalar defaults agree with "
                f"MESA `26.4.1` (`{n_unparsed}` skipped as non-scalar/computed "
                f"Rust initializers; `{n_no_field}` have no Rust field)."
            )
            continue

        sections_md.extend(
            [
                (
                    "| Control | Upstream `26.4.1` Source "
                    "| Upstream Default | `FlexMESA` Source "
                    "| `FlexMESA` Default | Classification |"
                ),
                "| :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for u_ent, r_ent, reason in mismatches:
            u_loc = f"`{u_ent.rel_file}:{u_ent.line}`"
            r_loc = f"`{r_ent.rel_file}:{r_ent.line}`"
            u_raw = u_ent.raw_value.replace("|", "\\|")
            r_raw = r_ent.raw_expr.replace("|", "\\|")
            sections_md.append(
                f"| `{u_ent.name}` | {u_loc} | `{u_raw}` | "
                f"{r_loc} | `{r_raw}` | {reason} |"
            )

    lines.append(
        f"| **Total** | **{total_up}** | **{total_matched}** | "
        f"**{total_agree}** | **{total_val_diff}** | **{total_none_diff}** | "
        f"**{total_unparsed}** | **{total_no_field}** |"
    )
    lines.extend(
        [
            "",
            (
                f"Denominator: `{total_matched} / {total_up}` upstream MESA "
                f"`26.4.1` scalar/1-slot defaults matched and compared "
                f"(`{total_agree}` agree, `{total_val_diff}` value mismatches, "
                f"`{total_none_diff}` `None`-disabled active limits); "
                f"`{total_unparsed}` skipped due to non-scalar/computed Rust "
                f"initializers; `{total_no_field}` skipped with no Rust field "
                f"in `FlexMESA` (plus `{total_array_only}` multi-subscript "
                f"array-only upstream controls outside the scalar denominator)."
            ),
        ]
    )

    if skipped_rows:
        lines.extend(
            [
                "",
                (
                    "## Skipped Controls with Rust Fields "
                    "(Non-Scalar / Computed Initializers)"
                ),
                "",
                (
                    "| Namelist | Control | Upstream `26.4.1` Source "
                    "| Upstream Default | `FlexMESA` Source "
                    "| Rust Initializer | Skip Reason |"
                ),
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for label, u_ent, u_rs in skipped_rows:
            u_loc = f"`{u_ent.rel_file}:{u_ent.line}`"
            r_loc = f"`{u_rs.rel_file}:{u_rs.line}`"
            u_raw = u_ent.raw_value.replace("|", "\\|")
            r_raw = u_rs.raw_expr.replace("|", "\\|")
            lines.append(
                f"| {label} | `{u_ent.name}` | {u_loc} | `{u_raw}` | "
                f"{r_loc} | `{r_raw}` | {u_rs.reason} |"
            )

    lines.extend(sections_md)
    lines.append("")
    summary = DefaultsCoverageSummary(
        by_namelist=tuple(per_namelist),
        upstream_scalars=total_up,
        upstream_array_only=total_array_only,
        matched=total_matched,
        agree=total_agree,
        val_mismatch=total_val_diff,
        none_disabled=total_none_diff,
        skipped_unparseable_rust=total_unparsed,
        skipped_no_rust_field=total_no_field,
    )
    return "\n".join(lines), total_val_diff, total_none_diff, unallowed, summary


def check_defaults_parity(
    mesa_root: pathlib.Path,
) -> tuple[str, int, int, list[tuple[str, str, str]]]:
    """Build report and return ``(md, val_diffs, none_diffs, unallowed)``."""
    md_text, total_val_diff, total_none_diff, unallowed, _ = (
        evaluate_defaults_parity(mesa_root)
    )
    return md_text, total_val_diff, total_none_diff, unallowed


def generate_markdown(mesa_root: pathlib.Path) -> str:
    """Build the complete ``defaults_diff_26_4_1.md`` report."""
    md_text, _, _, _, _ = evaluate_defaults_parity(mesa_root)
    return md_text


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesa-dir",
        default=None,
        help="Optional override for unpacked mesa-test-inputs-v2 root.",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "notes" / "audit" / "defaults_diff_26_4_1.md"),
        help="Output Markdown file path.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that --out matches generated content.",
    )
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Exit non-zero if any non-allowlisted default mismatch remains.",
    )
    ns = parser.parse_args(argv)
    mesa_root = fetch.resolve_mesa_test_inputs_dir(ns.mesa_dir)
    md_text, total_val_diff, total_none_diff, unallowed, summary = (
        evaluate_defaults_parity(mesa_root)
    )
    out_path = pathlib.Path(ns.out)
    rc = 0
    if ns.check:
        if (
            not out_path.is_file()
            or out_path.read_text(encoding="utf-8") != md_text
        ):
            print(f"{out_path} is out of date", file=sys.stderr)
            rc = 1
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md_text, encoding="utf-8")
        try:
            disp = out_path.relative_to(REPO_ROOT)
        except ValueError:
            disp = out_path
        print(f"Wrote {disp}")
    if ns.enforce and (total_val_diff > 0 or total_none_diff > 0 or unallowed):
        print(
            f"Defaults parity enforcement failed: {total_val_diff} value "
            f"mismatches, {total_none_diff} None-disabled active limits, "
            f"{len(unallowed)} unallowed items",
            file=sys.stderr,
        )
        for label, ctrl, reason in unallowed:
            print(f"  - [{label}] {ctrl}: {reason}", file=sys.stderr)
        return 1
    if rc == 0 and (ns.check or ns.enforce):
        print(
            f"Defaults parity OK: {summary.matched}/{summary.upstream_scalars} "
            f"upstream scalars compared ({summary.val_mismatch} mismatches, "
            f"{summary.none_disabled} None-disabled active limits; skipped: "
            f"{summary.skipped_unparseable_rust} non-scalar/computed Rust "
            f"initializers, {summary.skipped_no_rust_field} no Rust field; "
            f"plus {summary.upstream_array_only} multi-subscript array-only "
            "upstream defaults)"
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
