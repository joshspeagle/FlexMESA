# Window-test snapshot capture and planning (`#684` W1)

* **MESA version:** `r26.4.1` (`c8c202662922a22b3efefa824473ad615c623afd`)
* **SDK / compiler:** `x86_64-linux-26.6.1` (`gfortran 15.2.0`, `CRMATH`)
* **Rules:** `.coordination/triage/window-rules.md` (W0 v1, sections 2, 3 and 4.3)

## 1. Pass 1: Plan candidate start points

```bash
python scripts/windows/plan_windows.py --plan-only \
    --mesa-dir "$MESA_DIR" --spec-dir <spec-dir>
```

Uses CRC32 of `"star/<case>/<stage>"` to draw start points (`<= 60` models:
point 1 only; otherwise point 1, `25%`/`50%`/`75%` shifted `±5%`, and `95%`).

## 2. Build overlay and capture snapshots & clocks

```bash
source "$MESASDK_ROOT/bin/mesasdk_init.sh"
LIBRARY_PATH="$HOME/mesasdk-sysdeps/usr/lib/x86_64-linux-gnu" \
python benchmarks/fortran_mesa/stepdumps/run_stepdumps.py \
    --mesa-dir "$MESA_DIR" --mesasdk-root "$MESASDK_ROOT" \
    --scratch <overlay-scratch> --out <clock-out> \
    --snapshots --snapshot-spec-dir <spec-dir> \
    --snapshot-dir <snapshot-dir> \
    --jobs 4 --timeout-sec 14400
```

Snapshots `<case>_<stage>_<model>.mod` remain in `<snapshot-dir>` on local
scratch outside git (packed later in W3).

## 3. Pass 2: Generate and verify `windows_e2e.toml`

```bash
python scripts/windows/plan_windows.py \
    --mesa-dir "$MESA_DIR" \
    --clock-dir <clock-out>/star \
    --snapshot-dir <snapshot-dir> \
    --write-toml --out benchmarks/history_harness/windows_e2e.toml

python scripts/windows/plan_windows.py --check \
    --mesa-dir "$MESA_DIR" \
    --clock-dir <clock-out>/star \
    --snapshot-dir <snapshot-dir> \
    --out benchmarks/history_harness/windows_e2e.toml
```
