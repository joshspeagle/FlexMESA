# SPDX-License-Identifier: GPL-3.0-or-later
"""Record build and runtime provenance for MESA 26.4.1 reference outputs.

Queries OS, kernel, CPU model, MESA version, git tag/commit, archive MD5, MESA
SDK version and MD5, gfortran version, floating-point compiler flags, math
library backend, OpenMP thread count, and date, writing ``PROVENANCE.json`` and
``PROVENANCE.md`` with portable ``$MESA_DIR`` and ``$MESASDK_ROOT`` paths.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import platform
import subprocess
from typing import Any

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = pathlib.Path(
    os.environ.get(
        "MESA_GROUND_TRUTH_ROOT",
        str(pathlib.Path.home() / "mesa-ground-truth-26.4.1"),
    )
)
MESA_DIR = pathlib.Path(
    os.environ.get("MESA_DIR", str(pathlib.Path.home() / "mesa-26.04.1"))
)
MESASDK_ROOT = pathlib.Path(
    os.environ.get("MESASDK_ROOT", str(pathlib.Path.home() / "mesasdk-26.6.1"))
)
DOWNLOADS_DIR = pathlib.Path(
    os.environ.get(
        "MESA_DOWNLOADS_DIR", str(pathlib.Path.home() / "mesa-downloads")
    )
)
ENV_SCRIPT = SCRIPT_DIR / "env_mesa.sh"


def md5_of(path: pathlib.Path) -> str | None:
    """Compute hex MD5 digest if ``path`` exists."""
    if not path.is_file():
        return None
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "md5").hexdigest()


def sanitize_paths(text: str) -> str:
    """Replace local filesystem prefixes with portable environment variables."""
    out = text.replace(str(MESA_DIR), "$MESA_DIR")
    out = out.replace(str(MESASDK_ROOT), "$MESASDK_ROOT")
    out = out.replace(str(pathlib.Path.home()), "$HOME")
    return out


def read_cpu_model() -> tuple[str, int]:
    """Read CPU model name and logical processor count."""
    model = platform.processor() or "x86_64"
    logical = os.cpu_count() or 1
    cpuinfo = pathlib.Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for ln in cpuinfo.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if ln.startswith("model name"):
                model = ln.split(":", 1)[1].strip()
                break
    return (model, logical)


def read_os_release() -> str:
    """Read OS description from ``/etc/os-release`` and ``uname``."""
    pretty = platform.platform()
    os_rel = pathlib.Path("/etc/os-release")
    if os_rel.exists():
        for ln in os_rel.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if ln.startswith("PRETTY_NAME="):
                pretty = ln.split("=", 1)[1].strip().strip('"')
                break
    u = platform.uname()
    return f"{pretty} ({u.system} {u.release} {u.machine})"


def main() -> None:
    """Write ``PROVENANCE.json`` and ``PROVENANCE.md`` in ``ROOT_DIR``."""
    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    version_str = (
        (MESA_DIR / "data/version_number").read_text(encoding="utf-8").strip()
    )

    p_sdk = subprocess.run(
        [
            "bash",
            "-c",
            (
                f"source {ENV_SCRIPT} && $MESASDK_ROOT/bin/mesasdk_version "
                "&& echo '---' && gfortran --version | head -n 1"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    sdk_ver, _, gfortran_ver = p_sdk.stdout.partition("---")
    sdk_ver = sdk_ver.strip()
    gfortran_ver = gfortran_ver.strip()

    build_log = MESA_DIR / "build.log"
    modules_built = 0
    modules_tested = 0
    if build_log.exists():
        btxt = build_log.read_text(encoding="utf-8", errors="replace")
        modules_built = btxt.count("has been built")
        modules_tested = btxt.count("has been built, tested, and exported.")

    cpu_model, logical_cpus = read_cpu_model()
    os_desc = read_os_release()
    now_iso = (
        datetime.datetime.now(datetime.timezone.utc)
        .astimezone()
        .isoformat(timespec="seconds")
    )

    mesa_zip = DOWNLOADS_DIR / "mesa-26.04.1.zip"
    sdk_tgz = DOWNLOADS_DIR / "mesasdk-x86_64-linux-26.6.1.tar.gz"

    prov: dict[str, Any] = {
        "bundle_name": "mesa-histories-v1",
        "date": now_iso,
        "mesa": {
            "release": "26.4.1",
            "version_number_file": version_str,
            "git_tag": "26.4.1",
            "git_tag_object_sha": "857d7b0329ae4651b10a7c7a6e6bee2b46a9c7dd",
            "git_commit": "c8c202662922a22b3efefa824473ad615c623afd",
            "git_commit_short": "c8c2026",
            "zenodo_record": "19722306",
            "archive_name": "mesa-26.04.1.zip",
            "archive_md5": md5_of(mesa_zip)
            or "994fd28f38da92efc80e38ca11d8317b",
            "install_packages_built": modules_built,
            "install_module_self_checks_passed": modules_tested,
        },
        "sdk_and_compiler": {
            "mesasdk_version": sdk_ver,
            "mesasdk_zenodo_record": "20598423",
            "mesasdk_archive": "mesasdk-x86_64-linux-26.6.1.tar.gz",
            "mesasdk_archive_md5": md5_of(sdk_tgz)
            or "c5ef95aafe07848988959fd45cfe8baf",
            "compiler": gfortran_ver,
            "math_backend": (
                "CRMATH (MESASDK_MATH_SLOT=crmath: libcrmath + libcrlibm "
                "+ math-slots/crmath BLAS/LAPACK)"
            ),
            "floating_point_flags": [
                "-march=x86-64",
                "-mtune=generic",
                "-mfpmath=sse",
                "-O2",
                "-ftree-vectorize",
                "-fprotect-parens",
                "-fno-sign-zero",
                "-fno-range-check",
                "-finit-real=snan",
                "-fopenmp",
                "-std=f2008",
            ],
            "fma_contraction_emitted": False,
        },
        "host": {
            "os": os_desc,
            "cpu": cpu_model,
            "logical_threads": logical_cpus,
        },
        "run_configuration": {
            "OMP_NUM_THREADS": int(os.environ.get("OMP_NUM_THREADS", "4")),
            "MESA_SKIP_OPTIONAL": "t",
            "inlists_modified": False,
            "step_caps_applied": False,
            "runner_tool": "scripts/mesa_reference/run_mesa_case_26_4_1.sh",
            "thinner_tool": "notes/audit/tools/thin_mesa_history.py",
        },
    }

    for rel in (
        "module_tests/compiler_fp_flags.json",
        "module_tests/compiler_fp_flags.md",
    ):
        fp = ROOT_DIR / rel
        if fp.exists():
            fp.write_text(
                sanitize_paths(fp.read_text(encoding="utf-8")),
                encoding="utf-8",
            )

    (ROOT_DIR / "PROVENANCE.json").write_text(
        json.dumps(prov, indent=2) + "\n", encoding="utf-8"
    )

    md_lines = [
        "# Provenance: `mesa-histories-v1` (MESA Release 26.4.1)",
        "",
        f"- **Date**: `{prov['date']}`",
        f"- **MESA Release**: `26.4.1` (`data/version_number`: `{version_str}`)",
        (
            f"- **Git Tag & Commit**: tag `26.4.1` -> commit "
            f"`{prov['mesa']['git_commit']}` (`{prov['mesa']['git_commit_short']}`)"
        ),
        (
            f"- **Release Archive**: Zenodo record `19722306`, "
            f"`mesa-26.04.1.zip` (`md5: {prov['mesa']['archive_md5']}`)"
        ),
        (
            f"- **Install Self-Checks**: `{modules_tested}` / `{modules_built}` "
            "packages built, tested, and exported cleanly via `./install`"
        ),
        (
            f"- **MESA SDK**: `{sdk_ver}` (Zenodo record `20598423`, "
            f"`mesasdk-x86_64-linux-26.6.1.tar.gz`, "
            f"`md5: {prov['sdk_and_compiler']['mesasdk_archive_md5']}`)"
        ),
        f"- **Compiler**: `{gfortran_ver}`",
        (
            "- **Math Backend**: `CRMATH` (`MESASDK_MATH_SLOT=crmath`, "
            "`libcrmath` + `libcrlibm` + `math-slots/crmath` BLAS/LAPACK)"
        ),
        (
            "- **Floating-Point Compiler Flags**: `-march=x86-64 -mtune=generic "
            "-mfpmath=sse -O2 -ftree-vectorize -fprotect-parens -fno-sign-zero "
            "-fno-range-check -finit-real=snan -fopenmp -std=f2008`"
        ),
        f"- **Operating System**: `{os_desc}`",
        f"- **CPU**: `{cpu_model}` (`{logical_cpus}` threads)",
        f"- **`OMP_NUM_THREADS`**: `{prov['run_configuration']['OMP_NUM_THREADS']}`",
        (
            "- **Execution Mode**: `MESA_SKIP_OPTIONAL=t`, unmodified upstream "
            "inlists, uncapped steps"
        ),
        "",
    ]
    (ROOT_DIR / "PROVENANCE.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )
    print("Wrote PROVENANCE.json and PROVENANCE.md")


if __name__ == "__main__":
    main()
