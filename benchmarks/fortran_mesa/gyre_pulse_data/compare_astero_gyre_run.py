#!/usr/bin/env python3
"""Run FlexMESA's `astero_gyre` with its ported extras and compare every
GYRE call with MESA 26.4.1's reference run (#170, package G3c).

For each GYRE call (every 50 models and after the run) this records the
modes `astero_gyre`'s callback stores and the GYRE input arrays
(`star_get_pulse_data('GYRE')`) of FlexMESA's own star model, and compares
them with the reference run's `pulse_tap/modes.tsv` and
`pulse_tap/models/*.gyre` in the `mesa_histories` bundle. The result is
written as a TSV with a provenance header.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import struct
import time
from typing import Any

from tests.case_extras import resolve_case_extras
from tests.case_extras.astero_extras import AsteroGyreExtras

from flexmesa import _rust


def bits(x: float) -> int:
    return int(struct.unpack("<Q", struct.pack("<d", x))[0])


def load_gyre(path: pathlib.Path) -> tuple[list[float], list[list[float]]]:
    lines = path.read_text().splitlines()
    head = lines[0].split()
    n = int(head[0])
    glob = [float(v.replace("D", "e")) for v in head[1:4]]
    rows = [
        [float(v.replace("D", "e")) for v in ln.split()[1:]]
        for ln in lines[1 : 1 + n]
    ]
    return glob, rows


class Recorder(AsteroGyreExtras):
    """`AsteroGyreExtras` that also keeps the GYRE input of each call."""

    def __init__(self) -> None:
        super().__init__()
        self.pulse: dict[int, tuple[list[float], list[list[float]]]] = {}

    def _get_gyre_frequency_info(
        self, state: dict[str, Any], check_match: bool
    ) -> bool:
        mn = int(state["model_number"])
        # astero_gyre: add_center_point, no surface point, no atmosphere
        # (astero/defaults/astero_search.defaults:1277-1293).
        self.pulse[mn] = state["star_model"].gyre_pulse_data(True, False, False)
        return super()._get_gyre_frequency_info(state, check_match)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case-dir", type=pathlib.Path, required=True)
    ap.add_argument("--histories", type=pathlib.Path, required=True)
    ap.add_argument("--work", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    assert type(resolve_case_extras("astero", args.case_dir)).__name__ == (
        "AsteroGyreExtras"
    )
    rec = Recorder()
    t0 = time.time()
    rep = _rust.trace_astero_case(
        str(args.case_dir), str(args.work / "LOGS"), extras=rec
    )
    wall = time.time() - t0
    tap = args.histories / "runs/astero/astero_gyre/pulse_tap"
    ref: dict[int, list[list[str]]] = {}
    files: dict[int, str] = {}
    lines = (tap / "modes.tsv").read_text().splitlines()
    head = lines[0].split("\t")
    col = {h: i for i, h in enumerate(head)}
    for ln in lines[1:]:
        f = ln.split("\t")
        mn = int(f[col["model_number"]])
        ref.setdefault(mn, []).append(f)
        files[mn] = f[col["saved_pulse_file"]]
    rows = []
    for mn, results in rec.calls:
        want = ref.get(mn, [])
        same_modes = 0
        dfreq = 0.0
        for k, (_el, order, _em, inertia, freq, growth) in enumerate(results):
            if k >= len(want):
                break
            w = want[k]
            wf = float.fromhex(w[col["freq_uHz_hex"]])
            wi = float.fromhex(w[col["inertia_hex"]])
            wg = float.fromhex(w[col["growth_rate_rad_s_hex"]])
            if (
                order == int(w[col["n_pg"]])
                and bits(freq) == bits(wf)
                and bits(inertia) == bits(wi)
                and bits(growth) == bits(wg)
            ):
                same_modes += 1
            dfreq = max(dfreq, abs(freq - wf) / abs(wf))
        g_same = g_total = n_mine = 0
        dmax = "n/a"
        glob_same = False
        if mn in files and mn in rec.pulse:
            gref, pref = load_gyre(tap / files[mn].removeprefix("pulse_tap/"))
            gmine, pmine = rec.pulse[mn]
            g_total = len(pref)
            n_mine = len(pmine)
            glob_same = [bits(x) for x in gmine] == [bits(x) for x in gref]
            if n_mine == g_total:
                d = 0.0
                for a, b in zip(pmine, pref, strict=True):
                    if [bits(x) for x in a] == [bits(x) for x in b]:
                        g_same += 1
                    # r, P, T, rho (columns 1, 4, 5, 6)
                    for c in (0, 3, 4, 5):
                        if b[c] != 0.0:
                            d = max(d, abs(a[c] - b[c]) / abs(b[c]))
                dmax = f"{d:.3e}"
        rows.append(
            (
                mn,
                len(results),
                len(want),
                same_modes,
                f"{dfreq:.3e}",
                int(glob_same),
                n_mine,
                g_total,
                g_same,
                dmax,
            )
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    hdr = [
        "# FlexMESA astero_gyre run vs MESA 26.4.1 reference run, per GYRE call (#170, G3c)",  # noqa: E501
        "# reference: mesa_histories bundle runs/astero/astero_gyre/pulse_tap (MESA r26.4.1, SDK x86_64-linux-26.6.1)",  # noqa: E501
        "# generator: benchmarks/fortran_mesa/gyre_pulse_data/compare_astero_gyre_run.py",  # noqa: E501
        f"# flexmesa run: {rep.get('total_steps')} steps, termination "
        f"{rep.get('termination_code')}, wall {wall:.0f} s, numerics "
        f"{os.environ.get('FLEXMESA_NUMERICS_MODE', 'strict')}",
        "# modes_bitexact: modes whose n_pg, freq_uHz, inertia and growth rate equal MESA's bits",  # noqa: E501
        "# n_points: GYRE input points from FlexMESA's own model and from MESA's .gyre",  # noqa: E501
        "# pulse_points_bitexact: points with all 18 columns equal to MESA's bits",  # noqa: E501
        "# max_rel_d_r_P_T_rho: max relative difference of r, P, T, rho over points (equal point counts only)",  # noqa: E501
        "model_number\tn_modes\tn_modes_mesa\tmodes_bitexact\tmax_rel_dfreq"
        "\tglobal_bitexact\tn_points\tn_points_mesa\tpulse_points_bitexact"
        "\tmax_rel_d_r_P_T_rho",
    ]
    body = ["\t".join(str(v) for v in r) for r in rows]
    args.out.write_text("\n".join(hdr + body) + "\n")
    print("\n".join(hdr[-1:] + body))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
