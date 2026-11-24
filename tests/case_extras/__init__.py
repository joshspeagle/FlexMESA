# SPDX-License-Identifier: GPL-3.0-or-later
"""Upstream MESA `run_star_extras.f90` and `run_binary_extras.f90` dispatch
for test suites and the history harness (`#148`).

Dispatch inspects normalized Fortran source text inside `<case_dir>/src/` so
relocating or renaming a case directory never changes which extras handler is
attached.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
from typing import Any

from tests.case_extras import (
    astero_extras,
    binary_extras,
    rsp_extras,
    star_extras,
)

_INCLUDE_RE = re.compile(r'include\s+[\'"]([^\'"]+)[\'"]', re.IGNORECASE)


def load_fortran_source(case_dir: pathlib.Path, rel_name: str) -> str:
    """Read `<case_dir>/src/<rel_name>` and inline local `src/*.inc` files.

    Args:
        case_dir: Upstream MESA case directory root.
        rel_name: Relative filename inside `<case_dir>/src/`.

    Returns:
        Concatenated Fortran source text, or `""` when the file is absent.
    """
    src_dir = case_dir / "src"
    target = src_dir / rel_name
    if not target.is_file():
        return ""
    raw = target.read_text(encoding="utf-8", errors="replace")
    parts = [raw]
    for inc_name in _INCLUDE_RE.findall(raw):
        inc_path = src_dir / inc_name
        if inc_path.is_file():
            parts.append(inc_path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def normalize_fortran(text: str) -> str:
    """Strip Fortran comments and normalize whitespace and casing.

    Args:
        text: Raw Fortran source string.

    Returns:
        Lowercased, whitespace-collapsed string without `!` comments.
    """
    lines: list[str] = []
    for line in text.splitlines():
        code = line.split("!", 1)[0].strip()
        if code:
            lines.append(code.lower())
    return re.sub(r"\s+", " ", " ".join(lines))


def source_sha256(normalized_text: str) -> str:
    """Return the SHA-256 hex digest of normalized Fortran text."""
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def extras_source_digest(case_dir: pathlib.Path | str) -> str:
    """Return the SHA-256 of a case's normalized upstream extras sources.

    Args:
        case_dir: Path to the case directory on disk.

    Returns:
        Hex digest over normalized `run_star_extras.f90` and
        `run_binary_extras.f90` (an absent file contributes an empty string).
    """
    root = pathlib.Path(case_dir)
    texts = []
    for rel_name in ("run_star_extras.f90", "run_binary_extras.f90"):
        raw = load_fortran_source(root, rel_name)
        texts.append(normalize_fortran(raw) if raw else "")
    return source_sha256("\x00".join(texts))


# Digests of extras sources that were read and contain no case logic: every
# routine returns `keep_going`, reports zero extra columns, or calls only the
# shared `test_suite_startup` / `test_suite_after_evolve` timing helpers.
STOCK_EXTRAS_DIGESTS: frozenset[str] = frozenset(
    {
        # star/test_suite/1.3M_ms_high_Z/src/run_star_extras.f90:1-155 and 12
        # cases with identical normalized text.
        "4b7b570f3a144678fa67d368b2c5b485ad89e03f9b43abe8575c1635939c1a5a",
        # star/test_suite/make_zams/src/run_star_extras.f90:1-166,
        # make_zams_low_mass, and make_zams_ultra_high_mass (adds no-op
        # extras_start_step = 0).
        "506d0aac408f309287ac27f7bc3ca449cb309f7ce73379c95e348ac4228c7902",
        # star/test_suite/high_z/src/run_star_extras.f90:1-157 (unused local
        # real(dp) :: Mass in extras_after_evolve).
        "cee123d06d0473df8468de41c0b53c508ca76ecbb347e8ae08e00caa640ec426",
        # star/test_suite/7M_prems_to_AGB/src/run_star_extras.f90:1-158 (unused
        # local character(len=strlen) :: test in extras_after_evolve).
        "809c0bcd009192cf71888bd476fb2dc94f77e1a190c914ba2b37dba790032faf",
        # star/test_suite/wd_c_core_ignition/src/run_star_extras.f90:1-154
        # (duplicate ierr = 0 before test_suite_after_evolve).
        "ac62d4397d7617eb25e31531a0e81b981c27965a32c5a09430e9084db5639ddc",
        # star/test_suite/16M_conv_premix/src/run_star_extras.f90:1-157 and
        # 16M_predictive_mix (unused use chem_def in extras_finish_step).
        "dc0c5abaf8ce92977981d709bc64664c21ec2f5fe5ba74ab443bd14b681c61fe",
        # star/test_suite/custom_colors/src/run_star_extras.f90:1-256 (standard
        # template with dead `if (.false. .and. ...)` example block and zero
        # extra columns or header items).
        "5b83f1aa177d576018f1e32c7c068439fd4ff4066a1a70c5f83b1dc4865bb175",
        # binary/test_suite/evolve_both_stars/src/run_binary_extras.f90 and 2
        # cases with identical normalized text.
        "f4e8cb4d5f84304947b6a817879063dc65c04cacca90bba857f07fd50020e440",
    }
)


# Star handlers that have been run through `PyStarExtrasAdapter` on their own
# case without failing on state the adapter does not provide. A handler not
# listed here is written but cannot run yet because its hooks require an
# unported engine or `StarExtras` capability:
# - `#152` (solver/timestep/state hooks): `TzoStarExtras`
#   (`star_set_v_flag`, center mass and luminosity mutation).
# - `#153` (mesh/surface-cut hooks): `MakeCoWdStarExtras`
#   (`star_remove_surface_at_cell_k`).
# - `#154` (startup/relax/build-model hooks):
#   `RelaxCompositionJEntropyStarExtras` (`initial_mass` mutation before
#   model build).
# - `#155` (per-cell MLT/overshoot/rotation diffusivity hooks):
#   `PreMs12MCoreCollapseStarExtras`, `PreMs20MCoreCollapseStarExtras`,
#   `SplitBurnBigNetStarExtras`, `MakePreCcsn13bvnStarExtras` (per-cell
#   `alpha_mlt(1:nz)` is overwritten by `hydro_vars.rs` during `set_vars`
#   after `inlist_runner.rs` calls `other_alpha_mlt`, plus per-zone
#   `burn_num_iters` and `dvdt_drag`/`fdotv_drag_energy` are absent).
# - `#159` (rotation-distortion hooks): `PpisnStarExtras` (`other_kap_get`,
#   `other_eval_fp_ft`, `other_before_struct_burn_mix`, `#152`/`#159`/`#161`).
# - `#161` (opacity/surface-PT/other-physics hooks): `ConductiveFlameStarExtras`
#   (`other_surface_PT`, `warn_rates_for_high_temp`, custom `t_xtra1`, `#150`;
#   `other_build_initial_model` now adapter-served),
#   `MsOpMonoStarExtras` (`op_mono` tables), `RadiativeLevitationStarExtras`
#   (`set_op_mono_factors`), `OtherPhysicsHooksStarExtras` (`other_kap_get`,
#   `other_eos_results`, `other_screening`, `other_rate_get`,
#   `other_close_gaps`, `other_pressure`, `use_other_mesh_functions` /
#   `use_other_mesh_delta_coeff_factor` namelist inventory;
#   `other_overshooting_scheme`, `other_timestep_limit`,
#   `other_mesh_delta_coeff_factor`, `other_mesh_functions`, `other_wind` are
#   now adapter-served).
# - `#162` (pulse export & simplex calibration): `CheckPulseAtmStarExtras`
#   (`star_export_pulse_data('FGONG')` with `add_atmosphere_to_pulse_data`,
#   `atm_build_tau_outer`, `atm_T_tau_relation` atmosphere integration, `#161`),
#   `SimplexSolarCalibrationStarExtras` (`star_simplex_procs`, `#153`).
# - `#231` (custom termination codes): `C13PocketStarExtras`,
#   `PisnStarExtras` and `WdAicStarExtras` set `termination_code_str(t_xtra*)`
#   (`star/test_suite/c13_pocket/src/run_star_extras.f90:416`,
#   `star/test_suite/pisn/src/run_star_extras.f90:227`,
#   `star/test_suite/wd_aic/src/run_star_extras.f90:149`) and the case
#   manifest requires those strings; the adapter reports only the hook name.
# - `#152` (extras retry): `MakeSdbStarExtras` returns `retry`
#   (`star/test_suite/make_sdb/src/run_star_extras.f90:108`), which the step
#   driver does not implement.
# - `#172` (CCSN/STELLA + per-cell `alpha_mlt`): `CcsnIIpStarExtras`
#   (`#154`/`#155`/`#158`/`#172`).
#
# Attaching an unserved handler stops a run that otherwise takes steps, so
# `resolve_case_extras` leaves it detached and the case stays `unported` (it
# runs with default no-op hooks and never counts as a pass, `DECISIONS #32`).
ADAPTER_SERVED_STAR_HANDLERS: frozenset[str] = frozenset(
    {
        "AccretedMaterialJStarExtras",
        "AdjustNetStarExtras",
        "CarbonKhStarExtras",
        "CburnInwardStarExtras",
        "CepheidBlueLoopStarExtras",
        "CheckRedoStarExtras",
        "ConserveAngularMomentumStarExtras",
        "ConvCoreCpmStarExtras",
        "DiffusionSmoothnessStarExtras",
        "Dynamo15MStarExtras",
        "ExtendedConvectivePenetrationStarExtras",
        "GyreInMesaStarExtras",
        "Hb2MStarExtras",
        "HighRotDarkeningStarExtras",
        "HighRotation20MZ2m2StarExtras",
        "HotCoolWindStarExtras",
        "HseRiemannStarExtras",
        "LowZStarExtras",
        "MagneticBrakingStarExtras",
        "MakeEnvStarExtras",
        "MakeONeWdStarExtras",
        "NeutronStarEnvStarExtras",
        "PreMs1MToWdStarExtras",
        "RCrBStarExtras",
        "RspGyreExtras",
        "SemiconvectionStarExtras",
        "TTauGradrStarExtras",
        "Thermohaline1MStarExtras",
        "WdAccSmallDmStarExtras",
        "WdCool06MStarExtras",
        "WdDiffusionStarExtras",
        "WdNovaBurstStarExtras",
    }
)

# Binary handlers whose snapshot keys are supplied by `build_binary_snapshot`
# (`crates/flexmesa-py/src/extras_py.rs`). All 5 custom binary handlers in the
# MESA 26.4.1 manifest are now adapter-served (`DECISIONS #32`).
ADAPTER_SERVED_BINARY_HANDLERS: frozenset[str] = frozenset(
    {
        "DoubleBhBinaryExtras",
        "JdotGrCheckBinaryExtras",
        "JdotLsCheckBinaryExtras",
        "JdotMlCheckBinaryExtras",
        "WindFedBhHmxbBinaryExtras",
    }
)


def classify_case_extras(suite: str, case_dir: pathlib.Path | str) -> str:
    """Classify a case's upstream extras sources.

    Args:
        suite: Suite identifier (`"star"`, `"binary"`, or `"astero"`).
        case_dir: Path to the case directory on disk.

    Returns:
        `"ported"` when `resolve_case_extras` attaches a handler, `"stock"`
        when the normalized sources equal a reviewed no-logic template, and
        `"unported"` otherwise. An unported case runs with default no-op
        hooks, so its upstream extras logic is not executed. A case whose
        handler is written but not in `ADAPTER_SERVED_STAR_HANDLERS` or
        `ADAPTER_SERVED_BINARY_HANDLERS` is `"unported"`, as is one whose
        attached handler lists `unported_routines`.
    """
    handler = resolve_case_extras(suite, case_dir)
    if handler is not None:
        # An attached handler that still skips routines with case logic
        # (for example an astero case's `extras_after_evolve`) is unported.
        if getattr(handler, "unported_routines", ()):
            return "unported"
        return "ported"
    if extras_source_digest(case_dir) in STOCK_EXTRAS_DIGESTS:
        return "stock"
    return "unported"


def resolve_case_extras(suite: str, case_dir: pathlib.Path | str) -> Any:
    """Resolve the test-side extras handler to attach to a run.

    Args:
        suite: Suite identifier (`"star"`, `"binary"`, or `"astero"`).
        case_dir: Path to the case directory on disk.

    Returns:
        Python extras object implementing `StarExtras` or `BinaryExtras`
        callbacks or an astero case's `set_param`, or `None` when the
        upstream source uses default no-op hooks or its handler cannot run
        through the adapter yet (see `ADAPTER_SERVED_STAR_HANDLERS` and
        `ADAPTER_SERVED_BINARY_HANDLERS`).
    """
    handler = match_case_extras(suite, case_dir)
    if handler is None:
        return None
    if (
        suite == "star"
        and not isinstance(handler, rsp_extras.RspPeriodLimitExtras)
        and type(handler).__name__ not in ADAPTER_SERVED_STAR_HANDLERS
    ):
        return None
    if (
        suite == "binary"
        and type(handler).__name__ not in ADAPTER_SERVED_BINARY_HANDLERS
    ):
        return None
    return handler


def match_case_extras(suite: str, case_dir: pathlib.Path | str) -> Any:
    """Match the handler written for a case, whether or not it can run.

    Args:
        suite: Suite identifier (`"star"`, `"binary"`, or `"astero"`).
        case_dir: Path to the case directory on disk.

    Returns:
        The handler object matched from the upstream Fortran source, or
        `None` when no handler is written for it. Use `resolve_case_extras`
        for the handler to attach to a run.
    """
    root = pathlib.Path(case_dir)
    if suite == "star":
        raw_star = load_fortran_source(root, "run_star_extras.f90")
        if not raw_star:
            return None
        norm_star = normalize_fortran(raw_star)
        rsp_handler = rsp_extras.match_rsp_extras(norm_star)
        if rsp_handler is not None:
            return rsp_handler
        return star_extras.match_star_extras(norm_star)
    if suite == "binary":
        raw_bin = load_fortran_source(root, "run_binary_extras.f90")
        if not raw_bin:
            return None
        raw_star = load_fortran_source(root, "run_star_extras.f90")
        norm_bin = normalize_fortran(raw_bin)
        norm_star = normalize_fortran(raw_star) if raw_star else ""
        return binary_extras.match_binary_extras(norm_bin, norm_star)
    if suite == "astero":
        raw_star = load_fortran_source(root, "run_star_extras.f90")
        if not raw_star:
            return None
        return astero_extras.match_astero_extras(
            normalize_fortran(raw_star), source_sha256
        )
    return None
