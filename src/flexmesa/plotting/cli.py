# SPDX-License-Identifier: LGPL-3.0-or-later
"""Command-line interface for ``flexmesa plot`` diagnostic figures (`#173`)."""

from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile

import matplotlib.pyplot as plt

from flexmesa import _dispatch
from flexmesa.plotting import (
    compare,
    kippenhahn,
    live,
    pgbinary_panels,
    pgstar_extra,
    pgstar_grid,
    pgstar_nucleosyn,
    pgstar_panels,
    pgstar_summary,
    pgstar_text,
)
from flexmesa.testing import case_manifest, harness_runner

_PROFILE_MODES: tuple[str, ...] = (
    "summary4",
    "trho",
    "power",
    "abundance",
    "mixing",
    "dynamo",
    "network",
    "production",
    "summary",
    "summary_burn",
)


def _build_plot_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flexmesa plot",
        description=(
            "Render publication-styled comparison, Kippenhahn, radial profile, "
            "binary, track, and multi-panel grid figures from stored FlexMESA "
            "and MESA 26.4.1 outputs."
        ),
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_cmp = sub.add_parser(
        "compare",
        help="Compare a FlexMESA run against a stored MESA 26.4.1 history.",
    )
    p_cmp.add_argument(
        "case",
        help="Manifest case name or '<suite>/<case>' identifier.",
    )
    p_cmp.add_argument(
        "--suite",
        default=None,
        choices=("star", "binary", "astero"),
        help="Optional suite disambiguation.",
    )
    p_cmp.add_argument(
        "--flexmesa-history",
        default=None,
        help="Direct path to FlexMESA history.data or binary_history.data.",
    )
    p_cmp.add_argument(
        "--flexmesa-dir",
        default=None,
        help=(
            "Output root directory containing FlexMESA LOGS/ or "
            "binary_history.data."
        ),
    )
    p_cmp.add_argument(
        "--mesa-history",
        default=None,
        help="Optional explicit path overriding the stored MESA reference.",
    )
    p_cmp.add_argument(
        "--mesa-dir",
        default=None,
        help="Optional MESA test-suite root when auto-running trace.",
    )
    p_cmp.add_argument(
        "--manifest",
        default=None,
        help="Optional override path to case_manifest.toml.",
    )
    p_cmp.add_argument(
        "--out",
        default=None,
        help="Output figure path (defaults to '<case>_compare.png').",
    )
    p_cmp.add_argument(
        "--profile",
        default=None,
        help="Optional FlexMESA profileN.data to plot alongside history.",
    )
    p_cmp.add_argument(
        "--mesa-profile",
        default=None,
        help="Optional reference MESA profileN.data to overlay on --profile.",
    )
    p_cmp.add_argument(
        "--profile-out",
        default=None,
        help=(
            "Output path when --profile is set "
            "(defaults to '<case>_profile.png')."
        ),
    )
    p_cmp.add_argument(
        "--numerics-mode",
        choices=_dispatch.VALID_NUMERICS_MODES,
        default=None,
        help="Arithmetic mode when running tracer automatically.",
    )
    p_cmp.add_argument(
        "--allow-unsupported-controls",
        action="store_true",
        help="Bypass strict #147 rejection when running tracer automatically.",
    )
    p_cmp.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional per-stage step cap when running tracer automatically.",
    )
    p_cmp.add_argument(
        "--timeout-sec",
        type=float,
        default=300.0,
        help="Wall-clock timeout in seconds when auto-running (default 300).",
    )
    p_cmp.add_argument(
        "--max-rss-gb",
        type=float,
        default=5.5,
        help="Resident-memory ceiling in GB when auto-running (default 5.5).",
    )

    p_prof = sub.add_parser(
        "profile",
        help="Plot radial structure or individual pgstar profile panels.",
    )
    p_prof.add_argument("profile_path", help="Path to FlexMESA profileN.data.")
    p_prof.add_argument(
        "--mesa-profile",
        default=None,
        help="Optional reference MESA profileN.data to overlay.",
    )
    p_prof.add_argument(
        "--panel",
        choices=_PROFILE_MODES,
        default="summary4",
        help="Profile panel builder (default 'summary4').",
    )
    p_prof.add_argument(
        "--xaxis",
        default=None,
        help="Optional radial coordinate override (mass, q, radius, logRho).",
    )
    p_prof.add_argument(
        "--out",
        default="profile_panels.png",
        help="Output figure path (default 'profile_panels.png').",
    )

    p_kipp = sub.add_parser(
        "kipp",
        help="Plot a Kippenhahn convective/burning diagram from outputs.",
    )
    p_kipp.add_argument(
        "--history",
        default=None,
        help="Path to history.data providing mix/burn regions or core masses.",
    )
    p_kipp.add_argument(
        "--profiles",
        nargs="*",
        default=None,
        help="Optional sequence of profileN.data files.",
    )
    p_kipp.add_argument(
        "--xaxis",
        choices=("model_number", "star_age"),
        default="model_number",
        help="Horizontal axis variable (default 'model_number').",
    )
    p_kipp.add_argument(
        "--yaxis",
        choices=("mass", "q", "radius"),
        default="mass",
        help="Vertical structure coordinate (default 'mass').",
    )
    p_kipp.add_argument(
        "--show-luminosities",
        action="store_true",
        help="Overlay log_L, log_LH, log_LHe, and log_Lneu on right axis.",
    )
    p_kipp.add_argument(
        "--out",
        default="kippenhahn.png",
        help="Output figure path (default 'kippenhahn.png').",
    )

    p_bin = sub.add_parser(
        "binary",
        help="Plot pgbinary orbital, RLOF, mdot, and component HR panels.",
    )
    p_bin.add_argument(
        "--binary-history",
        default=None,
        help="Path to FlexMESA binary_history.data.",
    )
    p_bin.add_argument(
        "--flexmesa-dir",
        default=None,
        help="Run output directory containing binary_history.data and LOGS1/2.",
    )
    p_bin.add_argument(
        "--case",
        default=None,
        help="Optional manifest binary case name for reference lookup.",
    )
    p_bin.add_argument(
        "--mesa-history",
        default=None,
        help="Optional MESA 26.4.1 binary_history.data reference.",
    )
    p_bin.add_argument(
        "--star1-history",
        default=None,
        help="Optional Star 1 history.data path.",
    )
    p_bin.add_argument(
        "--star2-history",
        default=None,
        help="Optional Star 2 history.data path.",
    )
    p_bin.add_argument(
        "--mesa-star1-history",
        default=None,
        help="Optional MESA reference Star 1 history.data.",
    )
    p_bin.add_argument(
        "--mesa-star2-history",
        default=None,
        help="Optional MESA reference Star 2 history.data.",
    )
    p_bin.add_argument(
        "--orbit-only",
        action="store_true",
        help="Render standalone 2D Orbit and Roche equipotential figure.",
    )
    p_bin.add_argument(
        "--model-number",
        type=int,
        default=None,
        help="Optional model_number snapshot for 2D Orbit geometry.",
    )
    p_bin.add_argument(
        "--out",
        default="binary_panels.png",
        help="Output figure path (default 'binary_panels.png').",
    )

    p_trk = sub.add_parser(
        "track",
        help="Plot a named pgstar track preset or custom History_Track.",
    )
    p_trk.add_argument("history_path", help="Path to history.data.")
    p_trk.add_argument(
        "--preset",
        default=None,
        help="Named pgstar track preset (HR, TRho, TmaxRho, dPg_dnu, ...).",
    )
    p_trk.add_argument("--x", default=None, help="Horizontal column name.")
    p_trk.add_argument("--y", default=None, help="Vertical column name.")
    p_trk.add_argument(
        "--mesa-history",
        default=None,
        help="Optional MESA 26.4.1 reference history overlay.",
    )
    p_trk.add_argument(
        "--out",
        default="history_track.png",
        help="Output figure path (default 'history_track.png').",
    )

    p_grd = sub.add_parser(
        "grid",
        help="Compose a multi-panel pgstar/pgbinary Grid figure.",
    )
    p_grd.add_argument(
        "--panels",
        default=None,
        help="Comma-separated panel list (e.g. 'HR,TRho,Kipp,TRho_Profile').",
    )
    p_grd.add_argument(
        "--inlist",
        default=None,
        help="Optional &pgstar or &pgbinary namelist defining Grid<N>_*.",
    )
    p_grd.add_argument(
        "--grid-index",
        type=int,
        default=1,
        help="1-based Grid<N> index when reading --inlist (default 1).",
    )
    p_grd.add_argument(
        "--namelist",
        default=None,
        choices=("pgstar", "pgbinary"),
        help="Optional namelist group override ('pgstar' or 'pgbinary').",
    )
    p_grd.add_argument("--rows", type=int, default=None, help="Grid row count.")
    p_grd.add_argument(
        "--cols", type=int, default=None, help="Grid column count."
    )
    p_grd.add_argument(
        "--history", default=None, help="Path to star history.data."
    )
    p_grd.add_argument(
        "--mesa-history", default=None, help="Optional reference history.data."
    )
    p_grd.add_argument(
        "--profile", default=None, help="Path to star profileN.data."
    )
    p_grd.add_argument(
        "--profiles",
        nargs="*",
        default=None,
        help="Optional sequence of profileN.data files for Kipp/rti cells.",
    )
    p_grd.add_argument(
        "--mesa-profile", default=None, help="Optional reference profileN.data."
    )
    p_grd.add_argument(
        "--binary-history", default=None, help="Path to binary_history.data."
    )
    p_grd.add_argument(
        "--mesa-binary-history",
        default=None,
        help="Optional reference binary_history.data.",
    )
    p_grd.add_argument(
        "--star1-history", default=None, help="Optional Star 1 history.data."
    )
    p_grd.add_argument(
        "--star2-history", default=None, help="Optional Star 2 history.data."
    )
    p_grd.add_argument(
        "--star1-profile", default=None, help="Optional Star 1 profileN.data."
    )
    p_grd.add_argument(
        "--star2-profile", default=None, help="Optional Star 2 profileN.data."
    )
    p_grd.add_argument(
        "--title", default="", help="Optional grid figure title."
    )
    p_grd.add_argument(
        "--out",
        default="pgstar_grid.png",
        help="Output figure path (default 'pgstar_grid.png').",
    )

    p_rti = sub.add_parser(
        "rti",
        help="Plot Rayleigh-Taylor Instability (rti) evolution panel.",
    )
    p_rti.add_argument(
        "--history",
        default=None,
        help="Path to history.data with rti_type_*/mix_type_* columns.",
    )
    p_rti.add_argument(
        "--profiles",
        nargs="*",
        default=None,
        help="Optional sequence of profileN.data files.",
    )
    p_rti.add_argument(
        "--mesa-history",
        default=None,
        help="Optional MESA 26.4.1 reference history.data.",
    )
    p_rti.add_argument(
        "--xaxis",
        choices=("model_number", "star_age"),
        default="model_number",
        help="Horizontal axis variable (default 'model_number').",
    )
    p_rti.add_argument(
        "--age-units",
        choices=("yr", "Myr", "s"),
        default="yr",
        help="Units when --xaxis star_age (default 'yr').",
    )
    p_rti.add_argument(
        "--log-xaxis",
        action="store_true",
        help="Use log10(star_age) on horizontal axis.",
    )
    p_rti.add_argument(
        "--time-from-present",
        action="store_true",
        help="Measure age relative to final model.",
    )
    p_rti.add_argument(
        "--out",
        default="rti.png",
        help="Output figure path (default 'rti.png').",
    )

    p_cmd = sub.add_parser(
        "cmd",
        help="Plot photometric Color_magnitude<N> diagram from history.data.",
    )
    p_cmd.add_argument("history_path", help="Path to history.data.")
    p_cmd.add_argument(
        "--x1", default=None, help="Primary horizontal magnitude/BC column."
    )
    p_cmd.add_argument(
        "--x2", default=None, help="Subtracted horizontal column for color."
    )
    p_cmd.add_argument(
        "--y",
        default=None,
        help="Comma-separated vertical panels ('y1' or 'y1-y2').",
    )
    p_cmd.add_argument(
        "--inlist",
        default=None,
        help="Optional &pgstar namelist defining Color_magnitude<N>_*.",
    )
    p_cmd.add_argument(
        "--panel-index",
        type=int,
        default=1,
        help="1-based Color_magnitude<N> index (default 1).",
    )
    p_cmd.add_argument(
        "--mesa-history",
        default=None,
        help="Optional reference history.data overlay.",
    )
    p_cmd.add_argument(
        "--out",
        default="color_magnitude.png",
        help="Output figure path (default 'color_magnitude.png').",
    )

    p_txt = sub.add_parser(
        "text-summary",
        help="Render a Text_Summary<N> diagnostic table from history output.",
    )
    p_txt.add_argument(
        "history_path", help="Path to history.data or binary_history.data."
    )
    p_txt.add_argument(
        "--binary",
        action="store_true",
        help="Use pgbinary Text_Summary default layout.",
    )
    p_txt.add_argument(
        "--inlist",
        default=None,
        help="Optional &pgstar or &pgbinary namelist defining Text_Summary<N>.",
    )
    p_txt.add_argument(
        "--summary-index",
        type=int,
        default=1,
        help="1-based Text_Summary<N> index (default 1).",
    )
    p_txt.add_argument(
        "--mesa-history",
        default=None,
        help="Optional reference history.data for provenance label.",
    )
    p_txt.add_argument(
        "--title", default="", help="Optional summary panel title."
    )
    p_txt.add_argument(
        "--out",
        default="text_summary.png",
        help="Output figure path (default 'text_summary.png').",
    )

    p_w = sub.add_parser(
        "watch",
        help="Watch a live LOGS/ run directory and emit updated frames.",
    )
    p_w.add_argument(
        "logs_dir",
        help="Directory holding history.data (or LOGS/ / binary_history.data).",
    )
    p_w.add_argument(
        "--out-dir",
        default="live_frames",
        help="Output directory for live frames (default 'live_frames').",
    )
    p_w.add_argument(
        "--panels",
        default="HR,TRho",
        help="Comma-separated panel list when --inlist is omitted.",
    )
    p_w.add_argument(
        "--inlist",
        default=None,
        help="Optional &pgstar or &pgbinary namelist defining Grid<N>_*.",
    )
    p_w.add_argument(
        "--grid-index",
        type=int,
        default=1,
        help="1-based Grid<N> index when reading --inlist (default 1).",
    )
    p_w.add_argument(
        "--binary",
        action="store_true",
        help="Watch binary_history.data and LOGS1/LOGS2 outputs.",
    )
    p_w.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Polling interval in seconds (default 0.5).",
    )
    p_w.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on emitted frames before exiting.",
    )
    p_w.add_argument(
        "--max-wait",
        type=float,
        default=None,
        help="Optional wall-clock timeout in seconds.",
    )
    p_w.add_argument(
        "--prefix",
        default="frame_",
        help="Filename prefix for emitted frames (default 'frame_').",
    )
    p_w.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Raster resolution for live frames (default 150).",
    )
    p_w.add_argument("--title", default="", help="Optional figure title.")
    return parser


def _render_profile(
    profile_path: pathlib.Path | str,
    ref_profile_path: pathlib.Path | str | None,
    out_path: pathlib.Path | str,
    *,
    panel: str = "summary4",
    xaxis: str | None = None,
) -> None:
    if panel == "summary4":
        fig, _ = compare.plot_profile_panels(
            profile_path,
            ref_profile_path=ref_profile_path,
            save_path=out_path,
        )
    elif panel == "trho":
        trho_x = (
            xaxis if xaxis in ("logRho", "column_depth", "logxm") else "logRho"
        )
        fig, _ = pgstar_panels.plot_trho_profile(
            profile_path,
            ref_profile_path=ref_profile_path,
            xaxis=trho_x,  # type: ignore[arg-type]
            save_path=out_path,
        )
    elif panel == "power":
        fig, _ = pgstar_panels.plot_power_profile(
            profile_path,
            ref_profile_path=ref_profile_path,
            xaxis=xaxis or "mass",
            save_path=out_path,
        )
    elif panel == "abundance":
        fig, _ = pgstar_panels.plot_abundance_profile(
            profile_path,
            ref_profile_path=ref_profile_path,
            xaxis=xaxis or "mass",
            save_path=out_path,
        )
    elif panel == "mixing":
        fig, _ = pgstar_panels.plot_mixing_profile(
            profile_path,
            ref_profile_path=ref_profile_path,
            xaxis=xaxis or "mass",
            save_path=out_path,
        )
    elif panel == "dynamo":
        fig, _ = pgstar_extra.plot_dynamo_profile(
            profile_path,
            ref_profile_path=ref_profile_path,
            xaxis=xaxis or "mass",
            save_path=out_path,
        )
    elif panel == "network":
        fig, _ = pgstar_nucleosyn.plot_network_chart(
            profile_path,
            save_path=out_path,
        )
    elif panel == "production":
        fig, _ = pgstar_nucleosyn.plot_production_profile(
            profile_path,
            ref_profile_path=ref_profile_path,
            save_path=out_path,
        )
    elif panel == "summary":
        fig, _ = pgstar_summary.plot_summary_profile(
            profile_path,
            ref_profile_path=ref_profile_path,
            xaxis=xaxis or "mass",
            save_path=out_path,
        )
    else:
        fig, _ = pgstar_summary.plot_summary_burn_profile(
            profile_path,
            ref_profile_path=ref_profile_path,
            xaxis=xaxis or "mass",
            save_path=out_path,
        )
    plt.close(fig)
    print(f"Wrote profile figure: {out_path}")


def _render_comparison(
    entry: case_manifest.CaseManifestEntry,
    ns: argparse.Namespace,
    out_path: pathlib.Path,
    *,
    flex_hist: pathlib.Path | str | None,
    flex_dir: pathlib.Path | str | None,
    term_code: str,
    eff_mode: str,
    ctrl_ignored: bool,
    n_ctrl_ignored: int = 0,
) -> None:
    fig, _ = compare.plot_case_comparison(
        entry.case,
        suite=entry.suite,
        flexmesa_history=flex_hist,
        flexmesa_dir=flex_dir,
        mesa_history=ns.mesa_history,
        manifest_path=ns.manifest,
        save_path=out_path,
        termination_code=term_code,
        numerics_mode=eff_mode,
        controls_ignored=ctrl_ignored,
        n_controls_ignored=n_ctrl_ignored,
        max_steps=ns.max_steps,
        case_entry=entry,
    )
    plt.close(fig)
    print(f"Wrote comparison figure: {out_path}")


def _handle_compare(ns: argparse.Namespace) -> int:
    entry = compare.resolve_case_entry(
        ns.case, suite=ns.suite, manifest_path=ns.manifest
    )
    if ns.mesa_history is None and not entry.reference_path:
        print(
            f"error: {entry.suite}/{entry.case} has reference_status="
            f"{entry.reference_status!r} and no stored reference_path "
            "(pass --mesa-history to override)",
            file=sys.stderr,
        )
        return 2

    out_path = pathlib.Path(ns.out or f"{entry.case}_compare.png")
    if ns.flexmesa_history is None and ns.flexmesa_dir is None:
        eff_mode = harness_runner.resolve_effective_numerics_mode(
            ns.numerics_mode
        )
        case_dir = harness_runner.resolve_case_dir(
            entry.suite, entry.case, mesa_dir=ns.mesa_dir
        )
        with tempfile.TemporaryDirectory(prefix="flexmesa_plot_") as tmp_dir:
            outcome = harness_runner.run_case_with_diagnostic(
                entry.suite,
                case_dir,
                pathlib.Path(tmp_dir),
                max_steps=ns.max_steps,
                numerics_mode=eff_mode,
                allow_unsupported_controls=bool(ns.allow_unsupported_controls),
                timeout_sec=ns.timeout_sec,
                max_rss_gb=ns.max_rss_gb,
            )
            if outcome.history_path is None:
                print(
                    f"error: no history written for {entry.suite}/{entry.case} "
                    f"({outcome.error_message or outcome.status})",
                    file=sys.stderr,
                )
                return 1
            _render_comparison(
                entry,
                ns,
                out_path,
                flex_hist=outcome.history_path,
                flex_dir=None,
                term_code=outcome.termination_code,
                eff_mode=eff_mode,
                ctrl_ignored=outcome.controls_ignored,
                n_ctrl_ignored=outcome.n_controls_ignored,
            )
    else:
        _render_comparison(
            entry,
            ns,
            out_path,
            flex_hist=ns.flexmesa_history,
            flex_dir=ns.flexmesa_dir,
            term_code="",
            eff_mode="",
            ctrl_ignored=False,
        )

    if ns.profile is not None:
        _render_profile(
            ns.profile,
            ns.mesa_profile,
            pathlib.Path(ns.profile_out or f"{entry.case}_profile.png"),
        )
    return 0


def _handle_profile(ns: argparse.Namespace) -> int:
    _render_profile(
        ns.profile_path,
        ns.mesa_profile,
        ns.out,
        panel=ns.panel,
        xaxis=ns.xaxis,
    )
    return 0


def _handle_kipp(ns: argparse.Namespace) -> int:
    fig, _ = kippenhahn.plot_history_kippenhahn(
        history_path=ns.history,
        profile_paths=ns.profiles,
        xaxis=ns.xaxis,
        yaxis=ns.yaxis,
        show_luminosities=bool(ns.show_luminosities),
        save_path=ns.out,
    )
    plt.close(fig)
    print(f"Wrote Kippenhahn figure: {ns.out}")
    return 0


def _handle_binary(ns: argparse.Namespace) -> int:
    if ns.orbit_only:
        if ns.binary_history is None:
            print(
                "error: --orbit-only requires --binary-history", file=sys.stderr
            )
            return 2
        fig, _ = pgbinary_panels.plot_pgbinary_orbit(
            ns.binary_history,
            model_number=ns.model_number,
            mesa_binary_history=ns.mesa_history,
            save_path=ns.out,
        )
    else:
        fig, _ = pgbinary_panels.plot_pgbinary_panels(
            binary_history=ns.binary_history,
            case=ns.case,
            flexmesa_dir=ns.flexmesa_dir,
            mesa_binary_history=ns.mesa_history,
            star1_history=ns.star1_history,
            star2_history=ns.star2_history,
            mesa_star1_history=ns.mesa_star1_history,
            mesa_star2_history=ns.mesa_star2_history,
            orbit_model_number=ns.model_number,
            save_path=ns.out,
        )
    plt.close(fig)
    print(f"Wrote binary figure: {ns.out}")
    return 0


def _handle_track(ns: argparse.Namespace) -> int:
    fig, _ = pgstar_grid.plot_history_track(
        ns.history_path,
        ns.x,
        ns.y,
        preset=ns.preset,
        ref_history_path=ns.mesa_history,
        save_path=ns.out,
    )
    plt.close(fig)
    print(f"Wrote track figure: {ns.out}")
    return 0


def _handle_grid(ns: argparse.Namespace) -> int:
    panel_list = (
        [p.strip() for p in ns.panels.split(",") if p.strip()]
        if ns.panels is not None
        else None
    )
    fig, _ = pgstar_grid.plot_pgstar_grid(
        panels=panel_list,
        num_rows=ns.rows,
        num_cols=ns.cols,
        inlist_path=ns.inlist,
        grid_index=ns.grid_index,
        namelist_name=ns.namelist,
        history_path=ns.history,
        ref_history_path=ns.mesa_history,
        profile_path=ns.profile,
        ref_profile_path=ns.mesa_profile,
        profile_paths=ns.profiles,
        binary_history_path=ns.binary_history,
        ref_binary_history_path=ns.mesa_binary_history,
        star1_history_path=ns.star1_history,
        star2_history_path=ns.star2_history,
        star1_profile_path=ns.star1_profile,
        star2_profile_path=ns.star2_profile,
        title=ns.title,
        save_path=ns.out,
    )
    plt.close(fig)
    print(f"Wrote grid figure: {ns.out}")
    return 0


def _handle_rti(ns: argparse.Namespace) -> int:
    fig, _ = pgstar_extra.plot_rti_history(
        history_path=ns.history,
        profile_paths=ns.profiles,
        ref_history_path=ns.mesa_history,
        xaxis=ns.xaxis,
        age_units=ns.age_units,
        log_xaxis=bool(ns.log_xaxis),
        time_from_present=bool(ns.time_from_present),
        save_path=ns.out,
    )
    plt.close(fig)
    print(f"Wrote RTI figure: {ns.out}")
    return 0


def _handle_cmd(ns: argparse.Namespace) -> int:
    panels_arg: list[tuple[str, str | None]] | None = None
    if ns.y is not None:
        panels_arg = []
        for tok in str(ns.y).split(","):
            clean = tok.strip()
            if not clean:
                continue
            parts = [p.strip() for p in clean.split("-", 1)]
            panels_arg.append(
                (parts[0], parts[1] if len(parts) == 2 and parts[1] else None)
            )
    fig, _ = pgstar_extra.plot_color_magnitude(
        ns.history_path,
        xaxis1=ns.x1,
        xaxis2=ns.x2,
        panels=panels_arg,
        ref_history_path=ns.mesa_history,
        inlist_path=ns.inlist,
        panel_index=ns.panel_index,
        save_path=ns.out,
    )
    plt.close(fig)
    print(f"Wrote Color-Magnitude figure: {ns.out}")
    return 0


def _handle_text_summary(ns: argparse.Namespace) -> int:
    layout = None
    eff_title = ns.title
    if ns.inlist is not None:
        spec = pgstar_text.parse_inlist_text_summary_spec(
            ns.inlist,
            summary_index=ns.summary_index,
            namelist_name="pgbinary" if ns.binary else None,
        )
        layout = spec.layout
        eff_title = eff_title or spec.title
    fig, _ = pgstar_text.plot_text_summary(
        ns.history_path,
        layout=layout,
        is_binary=bool(ns.binary),
        ref_history_path=ns.mesa_history,
        title=eff_title,
        save_path=ns.out,
    )
    plt.close(fig)
    print(f"Wrote Text_Summary figure: {ns.out}")
    return 0


def _handle_watch(ns: argparse.Namespace) -> int:
    panel_list = (
        None
        if ns.inlist is not None
        else [p.strip() for p in str(ns.panels).split(",") if p.strip()]
    )
    frames = live.watch_logs(
        ns.logs_dir,
        ns.out_dir,
        panels=panel_list,
        inlist_path=ns.inlist,
        grid_index=ns.grid_index,
        is_binary=bool(ns.binary),
        poll_interval_sec=float(ns.poll_interval),
        max_frames=ns.max_frames,
        max_wait_sec=ns.max_wait,
        file_prefix=ns.prefix,
        dpi=int(ns.dpi),
        title=ns.title,
    )
    print(f"Wrote {len(frames)} live frame(s) to: {ns.out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Execute ``flexmesa plot`` CLI subcommands.

    Args:
        argv: Arguments following ``flexmesa plot`` (default ``sys.argv[1:]``).

    Returns:
        Exit status code (``0`` on success, non-zero on missing inputs/error).
    """
    cli_args = list(sys.argv[1:] if argv is None else argv)
    ns = _build_plot_parser().parse_args(cli_args)
    handlers = {
        "compare": _handle_compare,
        "profile": _handle_profile,
        "kipp": _handle_kipp,
        "binary": _handle_binary,
        "track": _handle_track,
        "grid": _handle_grid,
        "rti": _handle_rti,
        "cmd": _handle_cmd,
        "text-summary": _handle_text_summary,
        "watch": _handle_watch,
    }
    return handlers[str(ns.subcommand)](ns)


if __name__ == "__main__":
    raise SystemExit(main())
