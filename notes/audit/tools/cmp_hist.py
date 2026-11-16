"""Compare a FlexMESA per-step history against an upstream MESA history.

The reference may be a raw MESA ``LOGS/history.data`` file or a thinned TSV
written by ``thin_mesa_history.py``. The FlexMESA side is the TSV written by
the ``star_case_trace`` example. Both tracks are interpolated onto a common
grid of an independent variable (default ``star_age``, in years) and the final
values and the largest deviations along the track are printed.

Usage:
    python cmp_hist.py <mesa history.data | thinned.tsv> <flexmesa.tsv> [xcol]
"""

import sys

import numpy as np

SKIP_ALONG_TRACK = ("num_zones", "num_retries", "num_iters", "log_dt")
ALIASES = {"log_cntr_T": "log_center_T", "log_cntr_Rho": "log_center_Rho"}


def load(path: str) -> dict[str, np.ndarray]:
    """Load a MESA ``history.data`` file or a tab-separated history table.

    Returns:
        Mapping from column name to a 1D ``float64`` array with one entry per
        stored model.
    """
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    if lines[0].startswith("#") or "\t" in lines[0]:
        body = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
        names, rows = body[0].split("\t"), [ln.split("\t") for ln in body[1:]]
    else:
        names = lines[5].split()
        rows = [ln.split() for ln in lines[6:] if ln.strip()]
    data = np.array([[float(x) for x in row] for row in rows])
    cols = {ALIASES.get(n, n): data[:, i] for i, n in enumerate(names)}
    if "stage" in cols:
        last = cols["stage"] == cols["stage"].max()
        cols = {k: v[last] for k, v in cols.items()}
    return cols


def main() -> None:
    ref, fm = load(sys.argv[1]), load(sys.argv[2])
    x = sys.argv[3] if len(sys.argv) > 3 else "star_age"
    print(f"stored models: MESA {len(ref[x])}  FlexMESA {len(fm[x])}")
    shared = [
        c for c in fm if c in ref and c not in ("model_number", x, "stage")
    ]
    print("final values:")
    for c in [x, *shared]:
        a, b = ref[c][-1], fm[c][-1]
        print(f"  {c:16s} MESA {a: .8e}  FlexMESA {b: .8e}  diff {b - a: .3e}")
    lo = max(ref[x].min(), fm[x].min())
    hi = min(ref[x].max(), fm[x].max())
    grid = np.linspace(lo, hi, 400)[1:-1]
    order_r, order_f = np.argsort(ref[x]), np.argsort(fm[x])
    print(f"largest |diff| on the common {x} range [{lo:.4e}, {hi:.4e}]:")
    for c in shared:
        if c in SKIP_ALONG_TRACK:
            continue
        a = np.interp(grid, ref[x][order_r], ref[c][order_r])
        b = np.interp(grid, fm[x][order_f], fm[c][order_f])
        d = np.abs(a - b)
        i = int(np.argmax(d))
        print(
            f"  {c:16s} max {d.max():.3e} at {x}={grid[i]:.4e}"
            f"  median {np.median(d):.3e}"
        )
    for c in ("num_zones", "log_dt"):
        if c in ref and c in fm:
            print(
                f"  {c}: MESA min/median/max "
                f"{ref[c].min():.3g}/{np.median(ref[c]):.3g}/{ref[c].max():.3g}"
                f"  FlexMESA {fm[c].min():.3g}/{np.median(fm[c]):.3g}/"
                f"{fm[c].max():.3g}"
            )


if __name__ == "__main__":
    main()
