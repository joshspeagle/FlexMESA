#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regenerate upstream MESA 26.4.1 `RSP` and `star/test` golden fixtures.

Runs compiled upstream Fortran MESA `26.4.1` (`$MESA_DIR`, with MESA SDK active
in `PATH`) inside temporary working directories (never modifying `$MESA_DIR`)
to produce:

1. `tests/fixtures/golden/rsp/rsp_<slug>_{init,step1}_26_4_1.mod` (14 files)
2. `tests/fixtures/golden/rsp/lina_26_4_1.tsv` (7 x 3 `LINA` modes at 17 digits)
3. `tests/fixtures/golden/rsp/crossing_5m_26_4_1.tsv` & `7B_2nd_crossing_26_4_1.data`
4. `tests/fixtures/golden/star_test/history.data` (`star/test` 11-step history)
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import tempfile

RSP_CASES: tuple[tuple[str, str, str], ...] = (
    ("rsp_Cepheid", "rsp_cepheid", "inlist_rsp_Cepheid_header"),
    ("rsp_Cepheid_6M", "rsp_cepheid_6M", "inlist_rsp_Cepheid_header"),
    ("rsp_RR_Lyrae", "rsp_RR_Lyrae", "inlist_rsp_RR_Lyrae_header"),
    ("rsp_BEP", "rsp_BEP", "inlist_rsp_BEP_header"),
    ("rsp_BLAP", "rsp_BLAP", "inlist_rsp_BLAP_header"),
    ("rsp_Delta_Scuti", "rsp_Delta_Scuti", "inlist_rsp_Delta_Scuti_header"),
    (
        "rsp_Type_II_Cepheid",
        "rsp_Type_II_Cepheid",
        "inlist_rsp_Type_II_Cepheid_header",
    ),
)


def _write_wrapper_inlists(
    work_dir: pathlib.Path,
    header_inlist: str,
    save_fname: str,
    max_model: int,
    lina_out: str = "",
) -> None:
    """Write top-level `inlist` + `inlist_override` for a bounded RSP run."""
    inlist_txt = f"""&star_job
   read_extra_star_job_inlist(1) = .true.
   extra_star_job_inlist_name(1) = '{header_inlist}'
   read_extra_star_job_inlist(2) = .true.
   extra_star_job_inlist_name(2) = 'inlist_override'
/
&eos
   read_extra_eos_inlist(1) = .true.
   extra_eos_inlist_name(1) = '{header_inlist}'
/
&kap
   read_extra_kap_inlist(1) = .true.
   extra_kap_inlist_name(1) = '{header_inlist}'
/
&colors
   read_extra_colors_inlist(1) = .true.
   extra_colors_inlist_name(1) = '{header_inlist}'
/
&controls
   read_extra_controls_inlist(1) = .true.
   extra_controls_inlist_name(1) = '{header_inlist}'
   read_extra_controls_inlist(2) = .true.
   extra_controls_inlist_name(2) = 'inlist_override'
/
&pgstar
/
"""
    extra_ctrl = f"   x_character_ctrl(10) = '{lina_out}'\n" if lina_out else ""
    override_txt = f"""&star_job
   save_model_when_terminate = .true.
   save_model_filename = '{save_fname}'
   pgstar_flag = .false.
/
&controls
   max_model_number = {max_model}
   x_integer_ctrl(1) = 0
{extra_ctrl}/
"""
    (work_dir / "inlist").write_text(inlist_txt, encoding="utf-8")
    (work_dir / "inlist_override").write_text(override_txt, encoding="utf-8")


def generate_rsp_models_and_lina(
    mesa_dir: pathlib.Path,
    rsp_out_dir: pathlib.Path,
    env: dict[str, str],
) -> None:
    """Build and run the 7 RSP cases at `max_model_number = 0` and `1`."""
    rsp_out_dir.mkdir(parents=True, exist_ok=True)
    ver = (
        (mesa_dir / "data" / "version_number")
        .read_text(encoding="utf-8")
        .strip()
    )
    lina_lines = [
        "# FlexMESA upstream Fortran MESA RSP LINA eigenmode reference table",
        f"# mesa_version_number={ver}",
        "# mesa_git_tag=26.4.1",
        "# mesa_git_commit=c8c202662922a22b3efefa824473ad615c623afd",
        "# columns: case_slug\tp0_sec\tg0\tp1_sec\tg1\tp2_sec\tg2",
    ]

    for suite_name, slug, header_inlist in RSP_CASES:
        src_case = mesa_dir / "star" / "test_suite" / suite_name
        with tempfile.TemporaryDirectory(prefix=f"mesa_rsp_{slug}_") as tmp:
            work = pathlib.Path(tmp) / suite_name
            shutil.copytree(src_case, work)
            # Emit `<slug>_init_26_4_1.mod` (model_number = 0) and 17-digit LINA
            # in `extras_startup` before step 1, then terminate after step 1 to
            # write `<slug>_step1_26_4_1.mod` (model_number = 1) in a single run.
            rse = work / "src" / "run_star_extras.f90"
            rse_txt = rse.read_text(encoding="utf-8")
            init_name = f"{slug}_init_26_4_1.mod"
            step1_name = f"{slug}_step1_26_4_1.mod"
            startup_hook = (
                "      subroutine extras_startup(id, restart, ierr)\n"
                "         integer, intent(in) :: id\n"
                "         logical, intent(in) :: restart\n"
                "         integer, intent(out) :: ierr\n"
                "         type (star_info), pointer :: s\n"
                "         integer :: io_l, i_l\n"
                "         ierr = 0\n"
                "         call star_ptr(id, s, ierr)\n"
                "         if (ierr /= 0) return\n"
                "         call test_suite_startup(s, restart, ierr)\n"
                "         need_to_write_LINA_data = .false.\n"
                f"         call star_write_model(id, '{init_name}', ierr)\n"
                "         if (ierr /= 0) return\n"
                "         open(newunit=io_l, file='lina_26_4_1.dat', status='replace')\n"
                "         write(io_l, '(99(1x,es26.17e3))') (s% rsp_LINA_periods(i_l), s% rsp_LINA_growth_rates(i_l), i_l=1, 3)\n"
                "         close(io_l)\n"
                "      end subroutine extras_startup\n"
            )
            orig_st = rse_txt.index("      subroutine extras_startup")
            end_st = rse_txt.index(
                "      end subroutine extras_startup", orig_st
            ) + len("      end subroutine extras_startup")
            rse.write_text(
                rse_txt[:orig_st] + startup_hook + rse_txt[end_st:],
                encoding="utf-8",
            )

            subprocess.run(["./mk"], cwd=work, env=env, check=True)

            _write_wrapper_inlists(work, header_inlist, step1_name, 1)
            subprocess.run(["./star"], cwd=work, env=env, check=True)
            shutil.copyfile(work / init_name, rsp_out_dir / init_name)
            shutil.copyfile(work / step1_name, rsp_out_dir / step1_name)
            vals = [
                float(x)
                for x in (work / "lina_26_4_1.dat")
                .read_text(encoding="utf-8")
                .split()
            ]
            assert len(vals) == 6, f"{slug}: expected 6 LINA values, got {vals}"
            lina_lines.append(
                f"{slug}\t" + "\t".join(f"{v:.17e}" for v in vals)
            )
            print(f"Generated {init_name} (model 0) and {step1_name} (model 1)")

    (rsp_out_dir / "lina_26_4_1.tsv").write_text(
        "\n".join(lina_lines) + "\n", encoding="utf-8"
    )


def generate_rsp_2nd_crossing(
    mesa_dir: pathlib.Path,
    rsp_out_dir: pathlib.Path,
    env: dict[str, str],
) -> None:
    """Run `rsp_check_2nd_crossing` on both default `7M` and 13-point `5M`."""
    src_case = mesa_dir / "star" / "test_suite" / "rsp_check_2nd_crossing"
    ver = (
        (mesa_dir / "data" / "version_number")
        .read_text(encoding="utf-8")
        .strip()
    )
    with tempfile.TemporaryDirectory(prefix="mesa_rsp_2nd_") as tmp:
        work = pathlib.Path(tmp) / "rsp_check_2nd_crossing"
        shutil.copytree(src_case, work)
        subprocess.run(["./mk"], cwd=work, env=env, check=True)

        # 1. Unmodified default 7M run -> 7B_2nd_crossing_26_4_1.data
        subprocess.run(["./rn"], cwd=work, env=env, check=True)
        shutil.copyfile(
            work / "7B_2nd_crossing.data",
            rsp_out_dir / "7B_2nd_crossing_26_4_1.data",
        )

        # 2. 13-point 5M run at 17-digit precision -> crossing_5m_26_4_1.tsv
        rse = work / "src" / "run_star_extras.f90"
        txt = rse.read_text(encoding="utf-8")
        txt = txt.replace("(1pd20.10)", "(1x,1pes26.17e3)")
        txt = txt.replace("(a20,f20.10)", "(a20,1x,1pes26.17e3)")
        rse.write_text(txt, encoding="utf-8")
        subprocess.run(["./mk"], cwd=work, env=env, check=True)

        pts_5m = [
            (2182, 5117.81403001412, 1515.8745562683787),
            (2194, 5219.704046631389, 1539.1997424184128),
            (2205, 5329.114836734605, 1559.6316548920863),
            (2215, 5436.91562067493, 1578.4145139831076),
            (2224, 5538.172717431118, 1595.663842458941),
            (2233, 5641.070412718255, 1612.9664439524834),
            (2242, 5747.635013234617, 1630.6922947230432),
            (2251, 5857.06255923812, 1648.7795368178593),
            (2260, 5969.34867665182, 1667.2016095385336),
            (2268, 6071.304465353705, 1683.9715776229857),
            (2276, 6176.179601321493, 1701.1729488898218),
            (2284, 6284.581093120282, 1718.8743749747225),
            (2292, 6396.226491520547, 1737.050610191606),
        ]
        import math

        hist_lines = [
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "model_number star_age star_mass log_Teff log_L",
            "2100 5.0e7 5.0 3.6800000000000000 3.1600000000000000",
        ]
        for idx, (m, teff, lum) in enumerate(pts_5m):
            hist_lines.append(
                f"{m} {5.1e7 + idx * 1.0e4:.6e} 5.0 "
                f"{math.log10(teff):.16e} {math.log10(lum):.16e}"
            )
        (work / "history_5M_test.data").write_text(
            "\n".join(hist_lines) + "\n", encoding="utf-8"
        )
        inlist_5m = """&star_job
   read_extra_star_job_inlist(1) = .true.
   extra_star_job_inlist_name(1) = 'inlist_rsp_check_2nd_crossing'
/
&eos
   read_extra_eos_inlist(1) = .true.
   extra_eos_inlist_name(1) = 'inlist_rsp_check_2nd_crossing'
/
&kap
   read_extra_kap_inlist(1) = .true.
   extra_kap_inlist_name(1) = 'inlist_rsp_check_2nd_crossing'
/
&colors
/
&controls
   read_extra_controls_inlist(1) = .true.
   extra_controls_inlist_name(1) = 'inlist_rsp_check_2nd_crossing'
   read_extra_controls_inlist(2) = .true.
   extra_controls_inlist_name(2) = 'inlist_5m_override'
/
"""
        override_5m = """&controls
   RSP_mass = 5d0
   RSP_alfam = 0.25d0
   RSP_alfap = 0.0d0
   RSP_alfat = 0.0d0
   RSP_gammar = 0.0d0
   x_character_ctrl(1) = 'history_5M_test.data'
   x_character_ctrl(2) = 'out_5M.data'
   x_integer_ctrl(3) = 2
/
"""
        (work / "inlist").write_text(inlist_5m, encoding="utf-8")
        (work / "inlist_5m_override").write_text(override_5m, encoding="utf-8")
        subprocess.run(["./star"], cwd=work, env=env, check=True)

        raw_out = (
            (work / "out_5M.data").read_text(encoding="utf-8").splitlines()
        )
        tsv_lines = [
            "# FlexMESA upstream Fortran MESA rsp_check_2nd_crossing 5M reference",
            f"# mesa_version_number={ver}",
            "# mesa_git_tag=26.4.1",
            "# mesa_git_commit=c8c202662922a22b3efefa824473ad615c623afd",
        ]
        for ln in raw_out:
            s = ln.strip()
            if not s:
                continue
            parts = s.split()
            if parts[0] in ("Teff_red_edge", "Teff_blue_edge"):
                tsv_lines.append(
                    f"EDGE\t{parts[0]}\t{float(parts[1].replace('D', 'E')):.17e}"
                )
            elif len(parts) == 6 and parts[0].isdigit():
                mod_n = int(parts[0])
                p_days = float(parts[1].replace("D", "E"))
                growth = float(parts[2].replace("D", "E"))
                teff = float(parts[3].replace("D", "E"))
                lum = float(parts[4].replace("D", "E"))
                tsv_lines.append(
                    f"PT\t{mod_n}\t{teff:.17e}\t{lum:.17e}\t"
                    f"{p_days:.17e}\t{growth:.17e}"
                )
        (rsp_out_dir / "crossing_5m_26_4_1.tsv").write_text(
            "\n".join(tsv_lines) + "\n", encoding="utf-8"
        )
        print(
            "Generated 7B_2nd_crossing_26_4_1.data and crossing_5m_26_4_1.tsv"
        )


def generate_star_test_history(
    mesa_dir: pathlib.Path,
    star_test_out_dir: pathlib.Path,
    env: dict[str, str],
) -> None:
    """Copy or run `$MESA_DIR/star/test` to produce `LOGS/history.data`."""
    star_test_out_dir.mkdir(parents=True, exist_ok=True)
    existing = mesa_dir / "star" / "test" / "LOGS" / "history.data"
    if existing.is_file():
        shutil.copyfile(existing, star_test_out_dir / "history.data")
        print(f"Copied {existing.name} -> {star_test_out_dir / 'history.data'}")
        return
    with tempfile.TemporaryDirectory(prefix="mesa_star_test_") as tmp:
        work = pathlib.Path(tmp) / "test"
        shutil.copytree(mesa_dir / "star" / "test", work)
        subprocess.run(["./mk"], cwd=work, env=env, check=True)
        subprocess.run(["./ck"], cwd=work, env=env, check=True)
        shutil.copyfile(
            work / "LOGS" / "history.data", star_test_out_dir / "history.data"
        )


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesa-dir",
        type=pathlib.Path,
        default=pathlib.Path(os.environ.get("MESA_DIR", "")),
        help="Path to compiled MESA 26.4.1 tree.",
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[2],
        help="Path to FlexMESA repository root.",
    )
    parser.add_argument(
        "--skip-rsp-models",
        action="store_true",
        help="Skip regenerating the 14 rsp_*_{init,step1}_26_4_1.mod files.",
    )
    args = parser.parse_args()

    if not args.mesa_dir or not (args.mesa_dir / "star").is_dir():
        raise SystemExit("Provide --mesa-dir or set $MESA_DIR.")

    env = dict(os.environ)
    env["MESA_DIR"] = str(args.mesa_dir)
    env.setdefault("OMP_NUM_THREADS", "4")

    golden = args.repo_root / "tests" / "fixtures" / "golden"
    if not args.skip_rsp_models:
        generate_rsp_models_and_lina(args.mesa_dir, golden / "rsp", env)
    generate_rsp_2nd_crossing(args.mesa_dir, golden / "rsp", env)
    generate_star_test_history(args.mesa_dir, golden / "star_test", env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
