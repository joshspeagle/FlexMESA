#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# ruff: noqa: E501
"""Trims raw MESA rotation-mixing dumps to the committed oracle fixture.

Reads the `HDR`/`IN` rows written by `rotmix_oracle_inputs` and the `RMX`
rows written by `rotmix_oracle_dump` (see `instrumentation.patch`) and keeps
one `update_rotation_mixing_info` call. `RMX` rows keep their first 18 values
(the outputs of `update_rotation_mixing_info`); the remaining diagnostic
columns are dropped.

Usage:
    python make_fixture.py RAW_IN RAW_OUT NCALL OUT_PATH
"""

import sys

HEADER = """\
# FlexMESA MESA 26.4.1 set_rotation_mixing_info oracle (#158)
# mesa_version: r26.4.1
# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd
# mesasdk_version: x86_64-linux-26.6.1
# math_backend: CRMATH
# compiler: GNU Fortran (GCC) 15.2.0
# fflags: MESA 26.4.1 default star build (utils/makefile_header)
# case: star/test_suite/20M_z2m2_high_rotation, inlist_to_end_core_he_burn_header
# call: update_rotation_mixing_info call {ncall} with omega /= 0, not doing_relax,
#   not doing_first_model_of_run, prev_mesh_have_ST_start_info (model {model})
# HDR: ncall model_number nz doing_relax doing_first_model_of_run
#   prev_mesh_have_ST_start_info R_center mstar_old dt
# IN ncall k: r m L j_rot omega gradT grada_face grav cgrav eps_nuc
#   eps_nuc_neu_total non_nuc_neu csound rho T Peos cp Cv chiRho chiT abar
#   zbar opacity gamma1 dm dq rmid grad_density grad_temperature
#   smoothed_brunt_B mixing_type tau lnT ye D_mix_non_rotation kap_cond q
#   prev_mesh_dq prev_mesh_D_ST_start prev_mesh_nu_ST_start prev_mesh_nz
#   (inputs at entry to update_rotation_mixing_info)
# RMX tag model k: D_visc D_DSI D_SH D_SSI D_ES D_GSF D_ST nu_ST D_mix
#   D_mix_non_rotation D_mix_rotation am_nu_rot am_nu_non_rot am_nu_omega
#   am_nu_j richardson_number omega_shear domega_dlnR
#   (values at exit from update_rotation_mixing_info)
# Values are big-endian IEEE-754 binary64 raw bits (Z16.16).
"""


def main() -> None:
    raw_in, raw_out, ncall_s, out_path = sys.argv[1:5]
    ncall = int(ncall_s)
    tag = f"c{ncall}"
    keep_in: list[str] = []
    model = "?"
    with open(raw_in) as fh:
        for line in fh:
            t = line.split()
            if t[0] in ("HDR", "IN") and int(t[1]) == ncall:
                keep_in.append(line.rstrip("\n"))
                if t[0] == "HDR":
                    model = t[2]
    keep_out: list[str] = []
    with open(raw_out) as fh:
        for line in fh:
            t = line.split()
            if t[0] == "RMX" and t[1] == tag:
                keep_out.append(" ".join(t[: 4 + 18]))
    with open(out_path, "w") as fh:
        fh.write(HEADER.format(ncall=ncall, model=model))
        fh.write("\n".join(keep_in + keep_out) + "\n")


if __name__ == "__main__":
    main()
