"""Reduce an upstream MESA ``LOGS/history.data`` file to a compact TSV.

Keeps one row per model and the columns that FlexMESA's per-step tracer also
writes, so a real MESA trajectory can be stored in the repository as a
reference for end-to-end comparison.

Usage:
    python thin_mesa_history.py <history.data> <out.tsv> "<provenance note>"
"""

import sys

KEEP = [
    "model_number",
    "num_zones",
    "star_age",
    "log_dt",
    "star_mass",
    "log_LH",
    "log_LHe",
    "log_L",
    "log_Teff",
    "log_R",
    "log_g",
    "log_cntr_T",
    "log_cntr_Rho",
    "log_center_T",
    "log_center_Rho",
    "center_h1",
    "center_he4",
    "center_c12",
    "he_core_mass",
    "co_core_mass",
    "mass_conv_core",
    "num_retries",
    "num_iters",
]
RENAME = {"log_cntr_T": "log_center_T", "log_cntr_Rho": "log_center_Rho"}


def main() -> None:
    src, dst, note = sys.argv[1], sys.argv[2], sys.argv[3]
    lines = open(src, encoding="utf-8").read().split("\n")
    names = lines[5].split()
    idx = [(RENAME.get(n, n), i) for i, n in enumerate(names) if n in KEEP]
    with open(dst, "w", encoding="utf-8") as out:
        out.write(f"# {note}\n")
        out.write("\t".join(n for n, _ in idx) + "\n")
        for line in lines[6:]:
            cols = line.split()
            if cols:
                out.write("\t".join(cols[i] for _, i in idx) + "\n")


if __name__ == "__main__":
    main()
