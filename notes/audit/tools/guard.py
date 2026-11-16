"""Run a command with a wall-clock timeout and a resident-memory ceiling.

Prints the exit status, wall time and peak resident set size of the direct
child. Used to keep long FlexMESA campaigns from exhausting a small machine.

Usage:
    python guard.py <timeout_s> <max_rss_gb> <command> [args...]
"""

import subprocess
import sys
import time


def rss_kb(pid: int) -> int:
    """Return the resident set size of ``pid`` in kB (0 if it has exited)."""
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def main() -> None:
    timeout_s = float(sys.argv[1])
    cap_kb = float(sys.argv[2]) * 1.0e6
    proc = subprocess.Popen(sys.argv[3:])
    t0 = time.time()
    peak = 0
    reason = ""
    while proc.poll() is None:
        peak = max(peak, rss_kb(proc.pid))
        if time.time() - t0 > timeout_s:
            reason = "TIMEOUT"
        elif peak > cap_kb:
            reason = "RSS_CAP"
        if reason:
            proc.kill()
            proc.wait()
            break
        time.sleep(2)
    print(
        f"[guard] exit={proc.returncode} {reason} "
        f"wall_s={time.time() - t0:.0f} peak_rss_gb={peak / 1e6:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
