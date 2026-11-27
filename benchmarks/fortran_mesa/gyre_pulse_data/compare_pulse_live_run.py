#!/usr/bin/env python3
"""Run a pulsation case with FlexMESA and compare every oscillation call with
MESA 26.4.1's `pulse_tap/modes.tsv` (#612, work item 1; #170, #171).

The case runs through the in-process trace entry point with a recording
subclass of the case's own extras handler. The subclass only records; it does
not change what the handler computes. Each oscillation call (a model number
and the modes returned) is matched with the reference run's call at the same
model number (the k-th call at a model number against the k-th reference
call there) and then mode by mode on `(l, m, n_pg)`. Per quantity (frequency,
inertia, growth rate) the result gives the number of values compared, how
many are bit-identical, and the maximum relative difference. Calls and modes
present on one side only are counted. A quantity the extras do not expose is
reported as unavailable, not compared.

Handlers covered: `GyreInMesaStarExtras` (frequency and growth rate as the
real and imaginary parts of the mode frequency in Hz; no inertia),
`RspGyreExtras` (every mode of `GyreMesa.get_modes_full`; frequency and
growth rate only), `AsteroAdiplsExtras` and `AsteroGyreExtras` (their
`calls` lists: frequency in microHz, inertia, growth rate).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import math
import os
import pathlib
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from typing import Any, ClassVar

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
QUANTITIES = ("freq", "inertia", "growth")


@dataclasses.dataclass(frozen=True)
class ModeRec:
    """One mode of an oscillation call.

    Attributes:
        ell: Harmonic degree.
        m: Azimuthal order.
        n_pg: Radial order `n_pg`.
        freq: Frequency in the reference column's unit.
        inertia: Mode inertia, or `None` when the extras do not expose it.
        growth: Growth rate in the reference column's unit, or `None`.
    """

    ell: int
    m: int
    n_pg: int
    freq: float
    inertia: float | None
    growth: float | None


Call = tuple[int, list[ModeRec]]


@dataclasses.dataclass
class QuantityStat:
    """Comparison of one quantity over all matched modes."""

    available: bool = False
    compared: int = 0
    bit_identical: int = 0
    max_rel: float = 0.0


@dataclasses.dataclass
class CallResult:
    """Comparison of one matched oscillation call."""

    model_number: int
    occurrence: int
    n_modes: int
    n_modes_mesa: int
    n_common: int
    order_same: bool
    only_flexmesa: list[int]
    only_mesa: list[int]
    stats: dict[str, QuantityStat]


@dataclasses.dataclass
class Comparison:
    """Whole-run comparison."""

    calls: list[CallResult]
    calls_only_flexmesa: list[tuple[int, int]]
    calls_only_mesa: list[tuple[int, int]]
    modes_only_flexmesa: int
    modes_only_mesa: int
    totals: dict[str, QuantityStat]


def bits(x: float) -> int:
    """Return the IEEE-754 binary64 bit pattern of `x`."""
    return int(struct.unpack("<Q", struct.pack("<d", x))[0])


def rel_diff(a: float, b: float) -> float:
    """Relative difference `|a - b| / |b|`; 0 when equal, inf if `b == 0`."""
    if a == b:
        return 0.0
    if b == 0.0:
        return math.inf
    return abs(a - b) / abs(b)


def _group_by_occurrence(calls: Sequence[Call]) -> dict[tuple[int, int], Call]:
    seen: dict[int, int] = {}
    out: dict[tuple[int, int], Call] = {}
    for mn, modes in calls:
        k = seen.get(mn, 0)
        seen[mn] = k + 1
        out[(mn, k)] = (mn, modes)
    return out


def compare_calls(mine: Sequence[Call], ref: Sequence[Call]) -> Comparison:
    """Compare FlexMESA's oscillation calls with MESA's.

    Args:
        mine: FlexMESA's calls, `(model_number, modes)` in call order.
        ref: MESA's calls in the same form.

    Returns:
        Per-call and total comparison. A mode is matched by `(l, m, n_pg)`.
        `order_same` is true when the matched modes appear in the same
        relative order on both sides.
    """
    a = _group_by_occurrence(mine)
    b = _group_by_occurrence(ref)
    totals = {q: QuantityStat() for q in QUANTITIES}
    results: list[CallResult] = []
    modes_only_a = modes_only_b = 0
    for key in sorted(set(a) & set(b)):
        ma = a[key][1]
        mb = b[key][1]
        ia = {(m.ell, m.m, m.n_pg): m for m in ma}
        ib = {(m.ell, m.m, m.n_pg): m for m in mb}
        common = [k for k in ib if k in ia]
        modes_only_a += len(ia) - len(common)
        modes_only_b += len(ib) - len(common)
        order_same = common == [
            (m.ell, m.m, m.n_pg) for m in ma if (m.ell, m.m, m.n_pg) in ib
        ]
        stats = {q: QuantityStat() for q in QUANTITIES}
        for k in common:
            for q in QUANTITIES:
                x = getattr(ia[k], q)
                y = getattr(ib[k], q)
                if x is None or y is None:
                    continue
                for s in (stats[q], totals[q]):
                    s.available = True
                    s.compared += 1
                    s.bit_identical += int(bits(x) == bits(y))
                    s.max_rel = max(s.max_rel, rel_diff(x, y))
        results.append(
            CallResult(
                key[0],
                key[1],
                len(ma),
                len(mb),
                len(common),
                order_same,
                [k[2] for k in ia if k not in ib],
                [k[2] for k in ib if k not in ia],
                stats,
            )
        )
    return Comparison(
        results,
        sorted(set(a) - set(b)),
        sorted(set(b) - set(a)),
        modes_only_a,
        modes_only_b,
        totals,
    )


def load_reference(
    modes_tsv: pathlib.Path, freq_col: str, growth_col: str
) -> list[Call]:
    """Read MESA's `modes.tsv` into calls ordered by `eval_id`.

    Args:
        modes_tsv: `pulse_tap/modes.tsv` of a reference run.
        freq_col: Hex frequency column (`freq_Hz_hex` or `freq_uHz_hex`).
        growth_col: Hex growth-rate column.

    Returns:
        `(model_number, modes)` per `eval_id`, in `eval_id` order.
    """
    lines = modes_tsv.read_text().splitlines()
    col = {h: i for i, h in enumerate(lines[0].split("\t"))}
    by_eval: dict[int, tuple[int, list[ModeRec]]] = {}
    for ln in lines[1:]:
        f = ln.split("\t")
        ev = int(f[col["eval_id"]])
        mn = int(f[col["model_number"]])
        mode = ModeRec(
            ell=int(f[col["l"]]),
            m=int(f[col["m"]]),
            n_pg=int(f[col["n_pg"]]),
            freq=float.fromhex(f[col[freq_col]]),
            inertia=float.fromhex(f[col["inertia_hex"]]),
            growth=float.fromhex(f[col[growth_col]]),
        )
        by_eval.setdefault(ev, (mn, []))[1].append(mode)
    return [by_eval[e] for e in sorted(by_eval)]


def from_astero_calls(
    calls: Sequence[tuple[int, Sequence[tuple[Any, ...]]]],
) -> list[Call]:
    """Convert `(el, order, em, inertia, freq, growth)` result calls."""
    return [
        (
            mn,
            [
                ModeRec(int(el), int(em), int(order), freq, inertia, growth)
                for el, order, em, inertia, freq, growth in res
            ],
        )
        for mn, res in calls
    ]


def from_star_gyre_calls(
    calls: Sequence[tuple[int, Sequence[tuple[int, tuple[float, float]]]]],
    ls: Sequence[int],
) -> list[Call]:
    """Convert `GyreInMesaStarExtras.calls` (frequency `(re, im)` in Hz).

    Args:
        calls: `(model_number, [(n_pg, (freq_re, freq_im))])` per call.
        ls: Harmonic degree of each call (`x_integer_ctrl(3)`).

    Returns:
        Calls with growth rate set to `freq_im` and inertia unavailable.
    """
    return [
        (
            mn,
            [ModeRec(ls[i], 0, n_pg, fr, None, fi) for n_pg, (fr, fi) in modes],
        )
        for i, (mn, modes) in enumerate(calls)
    ]


def _fmt(x: float) -> str:
    return f"{x:.3e}"


def format_rows(cmp: Comparison) -> tuple[list[str], list[str]]:
    """Return `(column header, per-call rows)` as TSV text."""
    head = (
        "model_number\toccurrence\tn_modes\tn_modes_mesa\tn_common\torder_same"
        "\tfreq_n\tfreq_bitexact\tfreq_max_rel"
        "\tinertia_n\tinertia_bitexact\tinertia_max_rel"
        "\tgrowth_n\tgrowth_bitexact\tgrowth_max_rel"
        "\tn_pg_only_flexmesa\tn_pg_only_mesa"
    )
    rows = []
    for c in cmp.calls:
        cells = [
            str(c.model_number),
            str(c.occurrence),
            str(c.n_modes),
            str(c.n_modes_mesa),
            str(c.n_common),
            str(int(c.order_same)),
        ]
        for q in QUANTITIES:
            s = c.stats[q]
            if s.available:
                cells += [
                    str(s.compared),
                    str(s.bit_identical),
                    _fmt(s.max_rel),
                ]
            else:
                cells += ["n/a", "n/a", "n/a"]
        cells.append(",".join(map(str, c.only_flexmesa)) or "-")
        cells.append(",".join(map(str, c.only_mesa)) or "-")
        rows.append("\t".join(cells))
    return [head], rows


def format_summary(cmp: Comparison) -> list[str]:
    """Return `# summary` lines for the result header."""
    out = [
        f"# summary: calls compared {len(cmp.calls)}, calls only in FlexMESA "
        f"{cmp.calls_only_flexmesa}, calls only in MESA {cmp.calls_only_mesa}",
        f"# summary: modes only in FlexMESA {cmp.modes_only_flexmesa}, modes "
        f"only in MESA {cmp.modes_only_mesa}, calls with different mode "
        f"order {sum(not c.order_same for c in cmp.calls)}",
    ]
    for q in QUANTITIES:
        s = cmp.totals[q]
        if s.available:
            out.append(
                f"# summary: {q}: {s.compared} compared, {s.bit_identical} "
                f"bit-identical, max relative difference {_fmt(s.max_rel)}"
            )
        else:
            out.append(
                f"# summary: {q}: not exposed by the extras, not compared"
            )
    return out


class _RecordingGyre:
    """Proxy for `_rust.GyreMesa` that keeps `get_modes_full` results."""

    sink: ClassVar[list[tuple[int, int, int, tuple[float, float]]]] = []

    def __init__(self, real: Any) -> None:
        self._real = real

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def get_modes_full(self, ell: int) -> Any:
        modes = self._real.get_modes_full(ell)
        for md in modes:
            self.sink.append((ell, 0, md.n_pg, tuple(md.freq)))
        return modes


@contextlib.contextmanager
def _patched_gyre(module: Any) -> Iterator[None]:
    real_rust = module._rust

    class _Rust:
        def __getattr__(self, name: str) -> Any:
            return getattr(real_rust, name)

        @staticmethod
        def GyreMesa(*args: Any, **kwargs: Any) -> _RecordingGyre:
            return _RecordingGyre(real_rust.GyreMesa(*args, **kwargs))

    module._rust = _Rust()
    try:
        yield
    finally:
        module._rust = real_rust


def make_recorder(handler: Any) -> tuple[Any, str, Any]:
    """Wrap a resolved handler in a recording subclass.

    Args:
        handler: Handler returned by the case's extras factory.

    Returns:
        `(recorder, kind, extract)`: the recorder instance to attach to the
        run, a label (`star_gyre`, `rsp_gyre`, `adipls` or `astero_gyre`),
        and a function returning the recorded `list[Call]` after the run.
        The recorder shares the handler's state and adds recording only.
    """
    from tests.case_extras import astero_extras, rsp_extras, star_atm_pulse

    cls = type(handler)
    if isinstance(handler, star_atm_pulse.GyreInMesaStarExtras):

        class StarGyreRecorder(cls):  # type: ignore[valid-type, misc]
            def extras_finish_step(self, state: dict[str, Any]) -> str:
                n = len(self.calls)
                out = super().extras_finish_step(state)
                if len(self.calls) > n:
                    self.call_ls.append(int(state["x_integer_ctrl"][2]))
                return str(out)

        rec = StarGyreRecorder.__new__(StarGyreRecorder)
        rec.__dict__.update(handler.__dict__)
        rec.call_ls = []
        return (
            rec,
            "star_gyre",
            lambda: from_star_gyre_calls(rec.calls, rec.call_ls),
        )
    if isinstance(handler, rsp_extras.RspGyreExtras):

        class RspRecorder(cls):  # type: ignore[valid-type, misc]
            def other_rsp_linear_analysis(
                self, state: dict[str, Any], *args: Any, **kwargs: Any
            ) -> dict[str, Any]:
                start = len(_RecordingGyre.sink)
                with _patched_gyre(rsp_extras):
                    out = super().other_rsp_linear_analysis(
                        state, *args, **kwargs
                    )
                self.recorded.append(
                    (int(state["model_number"]), _RecordingGyre.sink[start:])
                )
                return dict(out)

        rec = RspRecorder.__new__(RspRecorder)
        rec.__dict__.update(handler.__dict__)
        rec.recorded = []
        _RecordingGyre.sink = []
        return (
            rec,
            "rsp_gyre",
            lambda: [
                (
                    mn,
                    [
                        ModeRec(ell, m, n, fr, None, fi)
                        for ell, m, n, (fr, fi) in modes
                    ],
                )
                for mn, modes in rec.recorded
            ],
        )
    if isinstance(handler, astero_extras.AsteroAdiplsExtras):
        return handler, "adipls", lambda: from_astero_calls(handler.calls)
    if isinstance(handler, astero_extras.AsteroGyreExtras):
        return handler, "astero_gyre", lambda: from_astero_calls(handler.calls)
    raise SystemExit(f"no recorder for extras class {cls.__name__}")


def _manifest_factory(suite: str, case: str) -> str | None:
    from flexmesa.testing import case_manifest

    entry = case_manifest.load_case_manifest().by_key().get((suite, case))
    if entry is None:
        raise SystemExit(f"{suite}/{case} is not in the case manifest")
    return str(entry.extras_factory or "") or None


def _watchdog(timeout_s: float, max_rss_gb: float) -> None:
    t0 = time.time()

    def loop() -> None:
        while True:
            time.sleep(2.0)
            rss_gb = 0.0
            with contextlib.suppress(OSError, ValueError):
                for ln in (
                    pathlib.Path("/proc/self/status").read_text().splitlines()
                ):
                    if ln.startswith("VmRSS:"):
                        rss_gb = int(ln.split()[1]) / 1048576.0
            if rss_gb > max_rss_gb:
                print(
                    f"WATCHDOG: RSS {rss_gb:.2f} GB > {max_rss_gb} GB",
                    flush=True,
                )
                os._exit(3)
            if time.time() - t0 > timeout_s:
                print(f"WATCHDOG: timeout after {timeout_s:.0f} s", flush=True)
                os._exit(4)

    threading.Thread(target=loop, daemon=True).start()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True, choices=("star", "astero"))
    ap.add_argument("--case", required=True)
    ap.add_argument("--histories", type=pathlib.Path, required=True)
    ap.add_argument("--work", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--timeout-sec", type=float, default=3000.0)
    ap.add_argument("--max-rss-gb", type=float, default=6.0)
    ap.add_argument("--command", default="", help="command line for the header")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(REPO_ROOT))
    from flexmesa.testing import harness_runner

    _watchdog(args.timeout_sec, args.max_rss_gb)
    case_dir = harness_runner.resolve_case_dir(args.suite, args.case)
    spec = _manifest_factory(args.suite, args.case)
    factory = harness_runner.load_extras_factory(spec)
    handler = factory(args.suite, case_dir) if factory else None
    if handler is None:
        raise SystemExit("the case resolves to no extras handler")
    rec, kind, extract = make_recorder(handler)
    hz = kind in ("star_gyre", "rsp_gyre")
    freq_col = "freq_Hz_hex" if hz else "freq_uHz_hex"
    growth_col = "growth_rate_Hz_hex" if hz else "growth_rate_rad_s_hex"
    tap = args.histories / "runs" / args.suite / args.case / "pulse_tap"
    ref = load_reference(tap / "modes.tsv", freq_col, growth_col)

    t0 = time.time()
    rep = harness_runner.run_trace_inprocess(
        args.suite,
        case_dir,
        args.work / "LOGS",
        extras=rec,
        numerics_mode="strict",
    )
    wall = time.time() - t0
    cmp = compare_calls(extract(), ref)

    head, rows = format_rows(cmp)
    commit = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "log",
            "-1",
            "--format=%h",
            "--",
            "crates",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    units = (
        "frequency Hz, growth = imaginary part in Hz"
        if hz
        else "frequency microHz, growth rad/s"
    )
    hdr = [
        f"# FlexMESA {args.suite}/{args.case} run vs MESA 26.4.1 reference "
        "run, per oscillation call (#612)",
        "# reference: mesa_histories bundle "
        f"runs/{args.suite}/{args.case}/pulse_tap/modes.tsv "
        "(MESA r26.4.1, git tag 26.4.1, gfortran 15.2.0, SDK "
        "x86_64-linux-26.6.1)",
        "# generator: benchmarks/fortran_mesa/gyre_pulse_data/"
        "compare_pulse_live_run.py",
        f"# command: {args.command}",
        f"# extras: {type(handler).__name__} ({kind}); units: {units}",
        f"# flexmesa engine commit (last change under crates/): {commit}",
        f"# flexmesa run: {rep.get('total_steps')} steps, termination "
        f"{rep.get('termination_code')}, wall {wall:.0f} s, numerics strict",
        "# columns *_n: values compared; *_bitexact: values with equal "
        "binary64 bits; *_max_rel: max |a-b|/|b| over compared values "
        "(n/a: not exposed by the extras)",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(hdr + format_summary(cmp) + head + rows) + "\n"
    args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
