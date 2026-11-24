# SPDX-License-Identifier: GPL-3.0-or-later
r"""Live-evolution plotting hooks and log watchers (`#173`).

Provides headless, non-leaking live frame rendering during FlexMESA single-star
and binary evolutions via :class:`LiveEvolutionPlotter` (compatible with
``StarExtras`` and ``BinaryExtras`` step hooks) and :func:`watch_logs` for
watching incrementally written ``LOGS/`` directories.

Mirrors the upstream ``pgstar``/``pgbinary`` ``*_file_cnt`` frame-emission
cadence (`star/defaults/pgstar.defaults:17-58`,
`binary/defaults/pgbinary.defaults:17-48`) while writing each frame atomically
and always closing every ``matplotlib`` figure.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt

from flexmesa.plotting import pgstar_grid
from flexmesa.testing import history_compare

_PROFILE_FILE_RE = re.compile(r"^profile(\d+)\.data$")


@dataclasses.dataclass(frozen=True)
class LiveFrameRecord:
    """Metadata for a frame written by :class:`LiveEvolutionPlotter`.

    Attributes:
        model_number: Model index associated with the rendered frame.
        path: Numbered output frame path on disk.
        latest_path: Optional atomically updated ``latest`` frame path.
        profile_path: Latest ``profileN.data`` used when rendering (if any).
        num_rows: Number of history rows present when the frame was rendered.
    """

    model_number: int
    path: pathlib.Path
    latest_path: pathlib.Path | None
    profile_path: pathlib.Path | None
    num_rows: int


@dataclasses.dataclass(frozen=True)
class _DiscoveredLogs:
    """Resolved history and profile files discovered under a run directory."""

    history_path: pathlib.Path | None
    binary_history_path: pathlib.Path | None
    star1_history_path: pathlib.Path | None
    star2_history_path: pathlib.Path | None
    profile_paths: tuple[pathlib.Path, ...]
    star1_profile_paths: tuple[pathlib.Path, ...]
    star2_profile_paths: tuple[pathlib.Path, ...]


def _sorted_profiles(directory: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """Return ``profileN.data`` files in ``directory`` ordered by ``N``."""
    if not directory.is_dir():
        return ()
    found: list[tuple[int, pathlib.Path]] = []
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        m = _PROFILE_FILE_RE.match(entry.name)
        if m:
            found.append((int(m.group(1)), entry))
    found.sort(key=lambda item: item[0])
    return tuple(path for _, path in found)


def _discover_log_paths(
    logs_dir: pathlib.Path, *, is_binary: bool
) -> _DiscoveredLogs:
    """Locate ``history.data``, ``binary_history.data``, and profiles."""
    bin_hist: pathlib.Path | None = None
    for cand in (
        logs_dir / "binary_history.data",
        logs_dir.parent / "binary_history.data",
    ):
        if cand.is_file():
            bin_hist = cand
            break

    star_hist: pathlib.Path | None = None
    star_log_dir = logs_dir
    if not is_binary:
        if (logs_dir / "history.data").is_file():
            star_hist = logs_dir / "history.data"
            star_log_dir = logs_dir
        elif (logs_dir / "LOGS" / "history.data").is_file():
            star_hist = logs_dir / "LOGS" / "history.data"
            star_log_dir = logs_dir / "LOGS"

    s1_dir = logs_dir / "LOGS1"
    s2_dir = logs_dir / "LOGS2"
    s1_hist = (
        s1_dir / "history.data" if (s1_dir / "history.data").is_file() else None
    )
    s2_hist = (
        s2_dir / "history.data" if (s2_dir / "history.data").is_file() else None
    )

    return _DiscoveredLogs(
        history_path=star_hist,
        binary_history_path=bin_hist if is_binary else None,
        star1_history_path=s1_hist,
        star2_history_path=s2_hist,
        profile_paths=_sorted_profiles(star_log_dir) if not is_binary else (),
        star1_profile_paths=_sorted_profiles(s1_dir),
        star2_profile_paths=_sorted_profiles(s2_dir),
    )


def _inspect_history_progress(path: pathlib.Path) -> tuple[int, int]:
    """Return ``(num_rows, latest_model_number)`` for a MESA history file."""
    if not path.is_file() or path.stat().st_size == 0:
        return 0, 0
    try:
        tbl = history_compare.load_history_file(path)
    except (OSError, ValueError):
        return 0, 0
    if tbl.num_rows <= 0:
        return 0, 0
    if "model_number" in tbl.columns:
        latest_m = round(float(tbl.columns["model_number"][-1]))
    else:
        latest_m = tbl.num_rows
    return tbl.num_rows, latest_m


def _atomic_save_figure(
    fig: mpl_figure.Figure,
    target_path: pathlib.Path,
    latest_path: pathlib.Path | None,
    *,
    dpi: int,
) -> None:
    """Save ``fig`` atomically via temporary file + ``os.replace``."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = target_path.suffix or ".png"
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target_path.parent),
        prefix=f".{target_path.stem}_",
        suffix=suffix,
    )
    os.close(fd)
    tmp_path = pathlib.Path(tmp_name)
    try:
        fig.savefig(tmp_path, dpi=dpi, bbox_inches="tight")
        os.replace(tmp_path, target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    if latest_path is not None:
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        fd2, tmp_latest_name = tempfile.mkstemp(
            dir=str(latest_path.parent),
            prefix=f".{latest_path.stem}_",
            suffix=latest_path.suffix or suffix,
        )
        os.close(fd2)
        tmp_latest = pathlib.Path(tmp_latest_name)
        try:
            shutil.copyfile(target_path, tmp_latest)
            os.replace(tmp_latest, latest_path)
        except Exception:
            if tmp_latest.exists():
                tmp_latest.unlink()
            raise


class LiveEvolutionPlotter:
    """Headless live-evolution frame renderer and ``Extras`` step hook.

    Compatible with ``flexmesa.star.run_test_suite_case(..., extras=...)``,
    ``flexmesa.binary.run_binary_test_suite_case(..., extras=...)``, and
    custom Python evolution loops. Every ``file_cnt`` steps it renders the
    configured ``pgstar``/``pgbinary`` panels from incrementally written
    ``history.data``/``profileN.data`` files, writes frames atomically, and
    closes the ``matplotlib`` figure immediately to prevent memory leaks.
    """

    def __init__(
        self,
        out_dir: pathlib.Path | str,
        *,
        logs_dir: pathlib.Path | str | None = None,
        panels: Sequence[pgstar_grid.GridPanelSpec | str] | None = (
            "HR",
            "TRho",
        ),
        inlist_path: pathlib.Path | str | None = None,
        grid_index: int = 1,
        namelist_name: str | None = None,
        file_cnt: int = 5,
        file_prefix: str = "frame_",
        file_format: str = "png",
        dpi: int = 150,
        write_latest: bool = True,
        render_on_finish: bool = True,
        is_binary: bool = False,
        ref_history_path: pathlib.Path | str | None = None,
        ref_profile_path: pathlib.Path | str | None = None,
        ref_binary_history_path: pathlib.Path | str | None = None,
        mesa_dir: pathlib.Path | str | None = None,
        title: str = "",
        on_frame: Callable[[LiveFrameRecord], None] | None = None,
    ) -> None:
        """Configure live-evolution plotting parameters.

        Args:
            out_dir: Directory receiving rendered frame images.
            logs_dir: Directory containing ``history.data`` (or ``LOGS/`` and
                ``binary_history.data``). When omitted, inferred from
                ``state["case_dir"]`` during ``extras_startup``.
            panels: Panel sequence passed to :func:`plot_pgstar_grid` when
                ``inlist_path`` is ``None``.
            inlist_path: Optional ``&pgstar``/``&pgbinary`` namelist defining
                ``Grid<N>_*``.
            grid_index: 1-based ``Grid<N>`` index when ``inlist_path`` is set.
            namelist_name: Optional ``"pgstar"`` or ``"pgbinary"`` override.
            file_cnt: Emit one frame every ``file_cnt`` model steps
                (matching upstream ``*_file_cnt``, must be ``>= 1``).
            file_prefix: Filename prefix for emitted frames.
            file_format: Output image extension without leading dot
                (default ``"png"``).
            dpi: Raster resolution for saved frames (default ``150`` for fast
                live iteration).
            write_latest: Also atomically update ``<file_prefix>latest.<ext>``.
            render_on_finish: Emit a terminal frame in ``extras_after_evolve``
                if the final model was not already rendered.
            is_binary: Set ``True`` when plotting ``pgbinary`` evolution logs.
            ref_history_path: Optional MESA ``26.4.1`` star history overlay.
            ref_profile_path: Optional MESA ``26.4.1`` star profile overlay.
            ref_binary_history_path: Optional MESA binary history overlay.
            mesa_dir: Optional MESA root for ``psi4.data`` / reference tracks.
            title: Optional figure super-title.
            on_frame: Optional callback invoked with each
                :class:`LiveFrameRecord` after atomic save.
        """
        if file_cnt < 1:
            raise ValueError(f"file_cnt must be >= 1, got {file_cnt!r}")
        if inlist_path is None and not panels:
            raise ValueError("provide non-empty panels or inlist_path")
        self._out_dir = pathlib.Path(out_dir)
        self._logs_dir = (
            pathlib.Path(logs_dir) if logs_dir is not None else None
        )
        self._panels = None if inlist_path is not None else tuple(panels or ())
        self._inlist_path = inlist_path
        self._grid_index = int(grid_index)
        self._namelist_name = namelist_name
        self._file_cnt = int(file_cnt)
        self._file_prefix = file_prefix
        self._file_format = file_format.lstrip(".")
        self._dpi = int(dpi)
        self._write_latest = bool(write_latest)
        self._render_on_finish = bool(render_on_finish)
        self._is_binary = bool(is_binary)
        self._ref_history_path = ref_history_path
        self._ref_profile_path = ref_profile_path
        self._ref_binary_history_path = ref_binary_history_path
        self._mesa_dir = mesa_dir
        self._title = title
        self._on_frame = on_frame
        self._frames: list[LiveFrameRecord] = []
        self._last_rendered_model: int | None = None
        self._last_rendered_rows: int = 0
        self._step_calls = 0

    @property
    def frames(self) -> tuple[LiveFrameRecord, ...]:
        """Chronological tuple of frames emitted by this plotter."""
        return tuple(self._frames)

    @property
    def logs_dir(self) -> pathlib.Path | None:
        """Resolved log directory currently watched by this plotter."""
        return self._logs_dir

    def _resolve_logs_from_state(
        self, state: dict[str, Any], *, binary_hint: bool
    ) -> None:
        if self._logs_dir is not None:
            return
        raw_dir = state.get("log_directory") or state.get("case_dir")
        if not raw_dir:
            return
        base = pathlib.Path(str(raw_dir))
        if not binary_hint and (base / "LOGS").is_dir():
            self._logs_dir = base / "LOGS"
        else:
            self._logs_dir = base

    def extras_startup(self, state: dict[str, Any], restart: bool) -> None:
        """``StarExtras`` startup hook capturing ``case_dir`` if needed."""
        _ = restart
        self._resolve_logs_from_state(state, binary_hint=self._is_binary)

    def extras_binary_startup(
        self, state: dict[str, Any], restart: bool
    ) -> None:
        """``BinaryExtras`` startup hook capturing ``case_dir`` if needed."""
        _ = restart
        self._is_binary = True
        self._resolve_logs_from_state(state, binary_hint=True)

    def _step_hook(self, state: dict[str, Any], *, binary_hint: bool) -> str:
        self._step_calls += 1
        self._resolve_logs_from_state(state, binary_hint=binary_hint)
        raw_model = state.get("model_number")
        model_no = int(raw_model) if raw_model is not None else self._step_calls
        if model_no > 0 and model_no % self._file_cnt == 0:
            self.render_now(model_number=model_no)
        return "keep_going"

    def extras_finish_step(self, state: dict[str, Any]) -> str:
        """``StarExtras`` step hook emitting frames every ``file_cnt`` steps."""
        return self._step_hook(state, binary_hint=self._is_binary)

    def finish_step(self, state: dict[str, Any]) -> str:
        """``BinaryExtras`` hook emitting frames every ``file_cnt`` steps."""
        self._is_binary = True
        return self._step_hook(state, binary_hint=True)

    def extras_after_evolve(self, state: dict[str, Any]) -> None:
        """``StarExtras`` end-of-evolution hook ensuring a final frame."""
        if not self._render_on_finish:
            return
        self._resolve_logs_from_state(state, binary_hint=self._is_binary)
        raw_model = state.get("model_number")
        model_no = int(raw_model) if raw_model is not None else None
        cur_rows = 0
        if self._logs_dir is not None:
            logs = _discover_log_paths(
                self._logs_dir, is_binary=self._is_binary
            )
            primary_hist = (
                logs.binary_history_path
                if self._is_binary
                else logs.history_path
            )
            if primary_hist is None and logs.binary_history_path is not None:
                primary_hist = logs.binary_history_path
            if primary_hist is not None:
                cur_rows, _ = _inspect_history_progress(primary_hist)
        if (
            model_no is None
            or model_no != self._last_rendered_model
            or cur_rows > self._last_rendered_rows
        ):
            self.render_now(model_number=model_no)

    def extras_binary_after_evolve(self, state: dict[str, Any]) -> None:
        """``BinaryExtras`` end-of-evolution hook ensuring a final frame."""
        self._is_binary = True
        self.extras_after_evolve(state)

    def render_now(
        self, *, model_number: int | None = None
    ) -> LiveFrameRecord | None:
        """Render one frame immediately from currently saved log files.

        Returns ``None`` if no non-empty history or profile log is present yet.
        Always closes the allocated ``matplotlib`` figure before returning.
        """
        if self._logs_dir is None:
            return None
        logs = _discover_log_paths(self._logs_dir, is_binary=self._is_binary)
        primary_hist = (
            logs.binary_history_path if self._is_binary else logs.history_path
        )
        if primary_hist is None and logs.binary_history_path is not None:
            primary_hist = logs.binary_history_path

        num_rows = 0
        file_model = 0
        if primary_hist is not None:
            num_rows, file_model = _inspect_history_progress(primary_hist)
        has_profiles = bool(
            logs.profile_paths
            or logs.star1_profile_paths
            or logs.star2_profile_paths
        )
        if num_rows == 0 and not has_profiles:
            return None

        eff_model = (
            int(model_number)
            if model_number is not None
            else (file_model if file_model > 0 else len(self._frames) + 1)
        )
        latest_prof = logs.profile_paths[-1] if logs.profile_paths else None
        s1_prof = (
            logs.star1_profile_paths[-1] if logs.star1_profile_paths else None
        )
        s2_prof = (
            logs.star2_profile_paths[-1] if logs.star2_profile_paths else None
        )

        eff_title = self._title or f"Model {eff_model}"
        fig, _axes = pgstar_grid.plot_pgstar_grid(
            panels=self._panels,
            inlist_path=self._inlist_path,
            grid_index=self._grid_index,
            namelist_name=self._namelist_name,
            history_path=logs.history_path if num_rows > 0 else None,
            ref_history_path=self._ref_history_path,
            profile_path=latest_prof,
            ref_profile_path=self._ref_profile_path,
            profile_paths=logs.profile_paths or None,
            binary_history_path=(
                logs.binary_history_path
                if (logs.binary_history_path is not None and num_rows > 0)
                else None
            ),
            ref_binary_history_path=self._ref_binary_history_path,
            star1_history_path=logs.star1_history_path,
            star2_history_path=logs.star2_history_path,
            star1_profile_path=s1_prof,
            star2_profile_path=s2_prof,
            star1_profile_paths=logs.star1_profile_paths or None,
            star2_profile_paths=logs.star2_profile_paths or None,
            mesa_dir=self._mesa_dir,
            title=eff_title,
        )
        target = (
            self._out_dir
            / f"{self._file_prefix}{eff_model:06d}.{self._file_format}"
        )
        latest = (
            self._out_dir / f"{self._file_prefix}latest.{self._file_format}"
            if self._write_latest
            else None
        )
        try:
            _atomic_save_figure(fig, target, latest, dpi=self._dpi)
        finally:
            plt.close(fig)

        rec = LiveFrameRecord(
            model_number=eff_model,
            path=target,
            latest_path=latest,
            profile_path=latest_prof or s1_prof,
            num_rows=num_rows,
        )
        self._frames.append(rec)
        self._last_rendered_model = eff_model
        self._last_rendered_rows = num_rows
        if self._on_frame is not None:
            self._on_frame(rec)
        return rec


def _logs_signature(
    logs_dir: pathlib.Path, *, is_binary: bool
) -> tuple[int, int, str, int]:
    """Compute a lightweight progress signature for ``watch_logs`` polling."""
    logs = _discover_log_paths(logs_dir, is_binary=is_binary)
    primary = logs.binary_history_path if is_binary else logs.history_path
    num_rows, latest_m = (
        _inspect_history_progress(primary) if primary is not None else (0, 0)
    )
    prof_name = ""
    prof_mtime = 0
    cand_profs = logs.profile_paths or logs.star1_profile_paths
    if cand_profs:
        newest = cand_profs[-1]
        prof_name = newest.name
        try:
            prof_mtime = newest.stat().st_mtime_ns
        except OSError:
            prof_mtime = 0
    return num_rows, latest_m, prof_name, prof_mtime


def watch_logs(
    logs_dir: pathlib.Path | str,
    out_dir: pathlib.Path | str,
    *,
    panels: Sequence[pgstar_grid.GridPanelSpec | str] | None = ("HR", "TRho"),
    inlist_path: pathlib.Path | str | None = None,
    grid_index: int = 1,
    namelist_name: str | None = None,
    is_binary: bool = False,
    poll_interval_sec: float = 0.5,
    max_frames: int | None = None,
    max_wait_sec: float | None = None,
    stop_event: threading.Event | None = None,
    file_prefix: str = "frame_",
    file_format: str = "png",
    dpi: int = 150,
    write_latest: bool = True,
    ref_history_path: pathlib.Path | str | None = None,
    ref_profile_path: pathlib.Path | str | None = None,
    ref_binary_history_path: pathlib.Path | str | None = None,
    mesa_dir: pathlib.Path | str | None = None,
    title: str = "",
    on_frame: Callable[[LiveFrameRecord], None] | None = None,
) -> list[LiveFrameRecord]:
    """Watch ``logs_dir`` and render frames whenever new evolution data lands.

    Polls ``logs_dir`` at ``poll_interval_sec`` until ``stop_event`` is set,
    ``max_frames`` frames have been written, or ``max_wait_sec`` elapses.

    Args:
        logs_dir: Run or ``LOGS/`` directory written by a running evolution.
        out_dir: Output directory for rendered frames.
        panels: Panel names or ``GridPanelSpec`` cells when ``inlist_path`` is
            ``None``.
        inlist_path: Optional ``&pgstar``/``&pgbinary`` inlist defining
            ``Grid<N>_*``.
        grid_index: 1-based ``Grid<N>`` index when ``inlist_path`` is set.
        namelist_name: Optional ``"pgstar"`` or ``"pgbinary"`` override.
        is_binary: Set ``True`` to watch ``binary_history.data`` and
            ``LOGS1``/``LOGS2``.
        poll_interval_sec: Polling sleep interval in seconds (must be ``> 0``).
        max_frames: Optional cap on emitted frames before returning.
        max_wait_sec: Optional wall-clock timeout in seconds before returning.
        stop_event: Optional ``threading.Event`` signalling clean shutdown.
        file_prefix: Filename prefix for emitted frames.
        file_format: Image extension without leading dot (default ``"png"``).
        dpi: Raster resolution for saved frames.
        write_latest: Atomically maintain ``<file_prefix>latest.<ext>``.
        ref_history_path: Optional MESA ``26.4.1`` reference history overlay.
        ref_profile_path: Optional MESA ``26.4.1`` reference profile overlay.
        ref_binary_history_path: Optional MESA reference binary history overlay.
        mesa_dir: Optional MESA root directory.
        title: Optional figure super-title.
        on_frame: Optional callback invoked after each emitted frame.

    Returns:
        List of emitted :class:`LiveFrameRecord` instances.
    """
    if poll_interval_sec <= 0.0:
        raise ValueError(
            f"poll_interval_sec must be > 0, got {poll_interval_sec!r}"
        )
    logs_path = pathlib.Path(logs_dir)
    plotter = LiveEvolutionPlotter(
        out_dir,
        logs_dir=logs_path,
        panels=panels,
        inlist_path=inlist_path,
        grid_index=grid_index,
        namelist_name=namelist_name,
        file_cnt=1,
        file_prefix=file_prefix,
        file_format=file_format,
        dpi=dpi,
        write_latest=write_latest,
        is_binary=is_binary,
        ref_history_path=ref_history_path,
        ref_profile_path=ref_profile_path,
        ref_binary_history_path=ref_binary_history_path,
        mesa_dir=mesa_dir,
        title=title,
        on_frame=on_frame,
    )

    emitted: list[LiveFrameRecord] = []
    last_sig: tuple[int, int, str, int] | None = None
    t_start = time.monotonic()

    while True:
        if stop_event is not None and stop_event.is_set():
            break
        if max_frames is not None and len(emitted) >= max_frames:
            break
        if (
            max_wait_sec is not None
            and (time.monotonic() - t_start) >= max_wait_sec
        ):
            break

        sig = _logs_signature(logs_path, is_binary=is_binary)
        if sig != last_sig and (sig[0] > 0 or bool(sig[2])):
            rec = plotter.render_now(model_number=sig[1] or None)
            if rec is not None:
                emitted.append(rec)
                last_sig = sig
                if max_frames is not None and len(emitted) >= max_frames:
                    break

        if stop_event is not None:
            if stop_event.wait(timeout=poll_interval_sec):
                break
        else:
            time.sleep(poll_interval_sec)

    return emitted
