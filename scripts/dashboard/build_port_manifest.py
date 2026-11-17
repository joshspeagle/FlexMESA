#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Seed or refresh ``benchmarks/dashboard/port_manifest.toml`` from MESA 26.4.1.

Counts exact upstream line counts and SHA-256 digests directly from
``mesa-26.04.1.zip`` (including bundled ``gyre-8.1.tar.gz``, ``ADIPLS``,
``STELLA``, and ``pgstar``/``pgbinary``), seeds statuses and file mappings from
``notes/audit/coverage-map.md`` and repository crate inspection, maps public
Fortran routines to verified ``FlexMESA`` symbols, downgrades any ``PORTED``
candidate missing a public-routine counterpart to ``partial``, and ensures zero
entries start in a ``verified`` tier (`#174`).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import pathlib
import re
import stat
import sys
import tarfile
import zipfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from flexmesa.testing import _port_parsers, port_manifest  # noqa: E402

# Explicit routine-counterpart hints where Rust uses snake_case / struct methods
# rather than identical Fortran procedure spelling.
ROUTINE_COUNTERPART_HINTS: dict[str, dict[str, str]] = {
    "star/private/eps_grav.f90": {
        "eval_eps_grav_and_partials": (
            "crates/flexmesa-star/src/eps_grav.rs::eval_eps_grav"
        ),
        "zero_eps_grav_and_partials": (
            "crates/flexmesa-star/src/eps_grav.rs::eval_eps_grav"
        ),
    },
    "star/private/hydro_chem_eqns.f90": {
        "do_chem_eqns": (
            "crates/flexmesa-star/src/hydro_eqns.rs::assemble_henyey_system"
        ),
        "do1_chem_eqns": (
            "crates/flexmesa-star/src/hydro_eqns.rs::do1_chem_eqns"
        ),
    },
    "star/private/hydro_momentum.f90": {
        "do_surf_momentum_eqn": (
            "crates/flexmesa-star/src/hydro_momentum.rs::"
            "eval_surf_momentum_bc_pressure_ad"
        ),
    },
    "star/private/init_model.f90": {
        "get_zams_model": ("crates/flexmesa-star/src/zams.rs::load_zams_model"),
    },
    "star/private/ionization_potentials.f90": {
        "set_ionization_potentials": (
            "crates/flexmesa-ionization/src/potentials_data.rs::"
            "IONIZATION_POTENTIALS"
        ),
    },
    "star/private/neu.f90": {
        "do_neu_for_cell": "crates/flexmesa-star/src/micro.rs::do_neu",
        "do_clear_neu_for_cell": "crates/flexmesa-star/src/micro.rs::do_neu",
    },
    "star/private/pulse_utils.f90": {
        "eval_face_X": (
            "crates/flexmesa-star/src/pulse_utils.rs::zone_iso_fractions"
        ),
        "eval_face_rho": ("crates/flexmesa-star/src/pulse_utils.rs::eval_face"),
        "eval_center": (
            "crates/flexmesa-star/src/pulse_utils.rs::eval_center_parabolic"
        ),
        "eval_center_X": (
            "crates/flexmesa-star/src/pulse_utils.rs::zone_iso_fractions"
        ),
    },
    "star/private/rsp.f90": {
        "rsp_setup_part1": (
            "crates/flexmesa-star/src/rsp_driver.rs::rsp_setup_model"
        ),
        "rsp_setup_part2": (
            "crates/flexmesa-star/src/rsp_driver.rs::rsp_setup_model"
        ),
        "build_rsp_model": (
            "crates/flexmesa-star/src/rsp_build.rs::do_rsp_build_model"
        ),
        "do1_rsp_build": (
            "crates/flexmesa-star/src/rsp_build.rs::build_rsp_envelope_model"
        ),
    },
    "star/private/rsp_build.f90": {
        "do_rsp_build": (
            "crates/flexmesa-star/src/rsp_build.rs::do_rsp_build_model"
        ),
    },
    "star/private/rsp_def.f90": {
        "init_def": "crates/flexmesa-star/src/rsp_def.rs::RspDerivedConsts",
        "init_allocate": "crates/flexmesa-star/src/rsp_def.rs::RspState",
        "init_free": "crates/flexmesa-star/src/rsp_def.rs::RspState",
        "rsp_phase_time0": "crates/flexmesa-star/src/rsp_def.rs::RspState",
        "rsp_WORK": (
            "crates/flexmesa-star/src/rsp_stats.rs::calculate_work_integrals"
        ),
        "rsp_WORKQ": (
            "crates/flexmesa-star/src/rsp_stats.rs::calculate_work_integrals"
        ),
        "rsp_WORKT": (
            "crates/flexmesa-star/src/rsp_stats.rs::calculate_work_integrals"
        ),
        "rsp_WORKC": (
            "crates/flexmesa-star/src/rsp_stats.rs::calculate_work_integrals"
        ),
        "rsp_photo_out": (
            "crates/flexmesa-star/src/rsp_state_io.rs::pack_scalars"
        ),
        "rsp_photo_in": (
            "crates/flexmesa-star/src/rsp_state_io.rs::unpack_scalars"
        ),
        "finish_rsp_photo_in": (
            "crates/flexmesa-star/src/rsp_state_io.rs::unpack_scalars"
        ),
        "set_build_vars": (
            "crates/flexmesa-star/src/rsp_grid.rs::finish_build_rsp_model"
        ),
        "set_star_vars": (
            "crates/flexmesa-star/src/rsp_grid.rs::set_star_vars_rsp"
        ),
        "check_for_T_or_P_inversions": (
            "crates/flexmesa-star/src/rsp_grid.rs::finish_after_build_model"
        ),
        "check_R": (
            "crates/flexmesa-star/src/rsp_grid.rs::finish_after_build_model"
        ),
        "rsp_dump_for_debug": ("crates/flexmesa-star/src/rsp_def.rs::RspState"),
    },
    "star/private/rsp_eval_eos_and_kap.f90": {
        "restart_rsp_eos_and_kap": (
            "crates/flexmesa-star/src/rsp_eos_kap.rs::RspChemCache"
        ),
        "init_for_rsp_eos_and_kap": (
            "crates/flexmesa-star/src/rsp_eos_kap.rs::RspChemCache"
        ),
        "eval1_mesa_eosDEgas_and_kap": (
            "crates/flexmesa-star/src/rsp_eos_inv.rs::"
            "eval1_mesa_eos_degas_and_kap"
        ),
        "eval1_mesa_eosDE_and_kap": (
            "crates/flexmesa-star/src/rsp_eos_inv.rs::eval1_mesa_eos_de_and_kap"
        ),
        "eval1_kap": (
            "crates/flexmesa-star/src/rsp_eos_kap.rs::eval1_kap_with_eos"
        ),
        "eval1_gamma_PT_getRho": (
            "crates/flexmesa-star/src/rsp_eos_inv.rs::eval1_gamma_pt_get_rho"
        ),
        "update_eos_and_kap": (
            "crates/flexmesa-star/src/rsp_eos_kap.rs::eval_mesa_eos_and_kap"
        ),
    },
    "star/private/rsp_lina.f90": {
        "mesa_eos_kap": ("crates/flexmesa-star/src/rsp_grid.rs::mesa_eos_kap"),
        "SORT": "crates/flexmesa-star/src/rsp_lina.rs::do_lina",
    },
    "star/private/rsp_relax_env.f90": {
        "EOP": "crates/flexmesa-star/src/rsp_grid.rs::eop",
    },
    "star/private/rsp_step.f90": {
        "calc_equations": "crates/flexmesa-star/src/rsp_eqns.rs::eval_eqns",
        "calc_Fr": "crates/flexmesa-star/src/rsp_eqns.rs::fr_eqn",
        "calc_cell_equations": (
            "crates/flexmesa-star/src/rsp_eqns_cell.rs::"
            "total_energy_eqn+turbulent_energy_eqn+t_form_of_erad_eqn"
        ),
    },
    "star/private/star_bcyclic.f90": {
        "bcyclic_factor": (
            "crates/flexmesa-mtx/src/bcyclic.rs::solve_block_tridiag_bcyclic"
        ),
        "bcyclic_solve": (
            "crates/flexmesa-mtx/src/bcyclic.rs::solve_block_tridiag_bcyclic"
        ),
        "clear_storage": (
            "crates/flexmesa-mtx/src/bcyclic.rs::solve_block_tridiag_bcyclic"
        ),
    },
    "star/private/write_model.f90": {
        "do_write_model": (
            "crates/flexmesa-star/src/model_io.rs::write_mod_string"
        ),
    },
    "star/job/run_star.f90": {
        "do_run_star": (
            "crates/flexmesa-star/src/inlist_runner.rs::run_test_suite_case"
        ),
    },
    "binary/job/run_binary.f90": {
        "do_run_binary": (
            "crates/flexmesa-binary/src/e2e.rs::run_binary_test_suite_case"
        ),
    },
    "binary/private/binary_disk.f90": {
        "kap": "crates/flexmesa-binary/src/disk.rs::kap_kramers",
        "T_the": "crates/flexmesa-binary/src/disk.rs::t_the_nonzero",
        "fL2_the": "crates/flexmesa-binary/src/disk.rs::fl2_from_the_and_t",
    },
    "binary/private/binary_edot.f90": {
        "get_edot": (
            "crates/flexmesa-binary/src/system_jdot.rs::eval_coupled_edot"
        ),
        "edot_tidal": (
            "crates/flexmesa-binary/src/system_jdot.rs::eval_coupled_edot"
        ),
        "edot_tidal_Hut": (
            "crates/flexmesa-binary/src/jdot_tides.rs::hut_f2+hut_f3"
        ),
        "edot_enhancement_Isotropic": (
            "crates/flexmesa-binary/src/mdot.rs::"
            "eval_edot_enhancement_isotropic"
        ),
    },
    "binary/private/binary_jdot.f90": {
        "get_jdot": (
            "crates/flexmesa-binary/src/system_jdot.rs::eval_gated_jdot"
        ),
        "default_jdot_gr": (
            "crates/flexmesa-binary/src/jdot_tides.rs::eval_jdot_gr"
        ),
        "default_jdot_ml": (
            "crates/flexmesa-binary/src/jdot_tides.rs::eval_jdot_ml"
        ),
        "default_jdot_ls": (
            "crates/flexmesa-binary/src/jdot_tides.rs::eval_jdot_ls"
        ),
        "default_jdot_missing_wind": (
            "crates/flexmesa-binary/src/jdot_tides.rs::eval_jdot_missing_wind"
        ),
        "default_jdot_mb": (
            "crates/flexmesa-binary/src/jdot_tides.rs::eval_jdot_mb"
        ),
    },
    "binary/private/binary_mdot.f90": {
        "check_implicit_rlo": (
            "crates/flexmesa-binary/src/mdot_implicit.rs::ImplicitMdotSolver"
        ),
        "pick_mdot_for_implicit_rlo": (
            "crates/flexmesa-binary/src/mdot_implicit.rs::record_and_propose"
        ),
        "adjust_mdots": (
            "crates/flexmesa-binary/src/system_rlof.rs::prepare_coupled_mdots"
        ),
        "rlo_mdot": (
            "crates/flexmesa-binary/src/system_rlof.rs::eval_scheme_explicit_mdot"
        ),
        "get_info_for_ritter": (
            "crates/flexmesa-binary/src/mdot.rs::eval_ritter_mdot_thin"
        ),
        "calculate_kolb_mdot_thick": (
            "crates/flexmesa-binary/src/mdot.rs::eval_kolb_mdot_thick"
        ),
        "get_info_for_kolb": (
            "crates/flexmesa-binary/src/mdot.rs::eval_kolb_mdot_thick"
        ),
        "get_info_for_arras": (
            "crates/flexmesa-binary/src/mdot.rs::eval_arras_mdot_thin"
        ),
        "get_info_for_ritter_eccentric": (
            "crates/flexmesa-binary/src/mdot.rs::eval_ritter_mdot_eccentric"
        ),
        "get_info_for_kolb_eccentric": (
            "crates/flexmesa-binary/src/mdot.rs::eval_kolb_mdot_eccentric"
        ),
        "set_accretion_composition": (
            "crates/flexmesa-binary/src/system_rlof.rs::surface_h1_and_he4"
        ),
    },
    "binary/private/binary_wind.f90": {
        "eval_wind_xfer_fractions": (
            "crates/flexmesa-binary/src/edd_wind.rs::"
            "eval_bondi_hoyle_wind_fraction"
        ),
        "Bondi_Hoyle_wind_transfer": (
            "crates/flexmesa-binary/src/edd_wind.rs::"
            "eval_bondi_hoyle_wind_fraction"
        ),
        "Tout_enhance_wind": (
            "crates/flexmesa-binary/src/edd_wind.rs::eval_tout_wind_enhancement"
        ),
    },
    "const/public/const_def.f90": {
        "do_const_init": "crates/flexmesa-const/src/lib.rs::PI",
    },
    "const/public/const_lib.f90": {
        "const_init": "crates/flexmesa-const/src/lib.rs::PI",
    },
    "turb/private/mlt.f90": {
        "calc_MLT": "crates/flexmesa-turb/src/mlt.rs::set_mlt",
    },
    "turb/private/semiconvection.f90": {
        "calc_semiconvection": (
            "crates/flexmesa-turb/src/semiconvection.rs::set_semiconvection"
        ),
    },
    "turb/private/tdc.f90": {
        "get_TDC_solution": (
            "crates/flexmesa-turb/src/tdc.rs::get_tdc_solution"
        ),
    },
    "turb/private/thermohaline.f90": {
        "get_D_thermohaline": (
            "crates/flexmesa-turb/src/thermohaline.rs::get_d_thermohaline"
        ),
    },
    "turb/public/turb.f90": {
        "set_thermohaline": (
            "crates/flexmesa-turb/src/thermohaline.rs::get_d_thermohaline"
        ),
        "set_MLT": "crates/flexmesa-turb/src/mlt.rs::set_mlt",
        "set_TDC": "crates/flexmesa-turb/src/tdc.rs::get_tdc_solution",
        "set_semiconvection": (
            "crates/flexmesa-turb/src/semiconvection.rs::set_semiconvection"
        ),
    },
    "neu/private/mod_neu.f90": {
        "neutrinos": "crates/flexmesa-neu/src/lib.rs::neu_get",
        "set_inputs": "crates/flexmesa-neu/src/inputs.rs::NeuInputs",
    },
    "neu/public/neu_lib.f90": {
        "neu_get": "crates/flexmesa-neu/src/lib.rs::neu_get",
    },
    "interp_2d/private/bipm_db.f90": {
        "do_mkbipm_db": (
            "crates/flexmesa-interp2d/src/bipm.rs::interp_mkbipm_db"
        ),
        "do_evbipm_db": (
            "crates/flexmesa-interp2d/src/bipm.rs::interp_evbipm_db"
        ),
    },
}


def _md5_file(path: pathlib.Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as fh:
        while chunk := fh.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_mesa_zip_sources(
    mesa_zip: pathlib.Path,
) -> dict[str, tuple[str, int, str, str]]:
    """Return ``{relpath: (module, lines, sha256, decoded_text)}`` from zip."""
    out: dict[str, tuple[str, int, str, str]] = {}
    with zipfile.ZipFile(mesa_zip, "r") as zf:
        top = zf.namelist()[0].split("/")[0] + "/"
        for zi in zf.infolist():
            if zi.is_dir() or not zi.filename.startswith(top):
                continue
            raw_mode = (zi.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(raw_mode):
                continue
            rel = zi.filename[len(top) :]
            mod = _port_parsers.classify_upstream_relpath(rel)
            if mod is None:
                continue
            data = zf.read(zi)
            out[rel] = (
                mod,
                len(data.splitlines()),
                hashlib.sha256(data).hexdigest(),
                data.decode("utf-8", errors="replace"),
            )

        gyre_tar = zf.read(top + "gyre/gyre-8.1.tar.gz")
        with tarfile.open(fileobj=io.BytesIO(gyre_tar), mode="r:gz") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                mpath = pathlib.PurePosixPath(member.name)
                sub_rel = "/".join(mpath.parts[1:])
                rel = f"gyre/gyre/{sub_rel}"
                mod = _port_parsers.classify_upstream_relpath(rel)
                if mod is None:
                    continue
                fobj = tf.extractfile(member)
                assert fobj is not None
                data = fobj.read()
                out[rel] = (
                    mod,
                    len(data.splitlines()),
                    hashlib.sha256(data).hexdigest(),
                    data.decode("utf-8", errors="replace"),
                )
    return out


def _resolve_coverage_map_files(upstream_path: str, raw_fm: str) -> list[str]:
    """Resolve shorthand file tokens from ``notes/audit/coverage-map.md``."""
    low = raw_fm.strip()
    if low in {"none", "n/a", ""} or "internal BVP solver" in raw_fm:
        return []
    if low.startswith("none "):
        return []
    top = upstream_path.split("/")[0]
    default_crate = {
        "star": "crates/flexmesa-star/src",
        "star_data": "crates/flexmesa-star/src",
        "binary": "crates/flexmesa-binary/src",
        "astero": "crates/flexmesa-astero/src",
    }.get(top, "crates/flexmesa-star/src")

    resolved: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9_/*.-]+\.rs", raw_fm):
        if token.startswith("crates/"):
            pat = token
        elif "/" in token:
            pat = f"crates/{token}"
        else:
            pat = f"{default_crate}/{token}"
        for match in sorted(REPO_ROOT.glob(pat)):
            if match.is_file():
                resolved.add(match.relative_to(REPO_ROOT).as_posix())
    if "flexmesa-schema" in raw_fm:
        resolved.add("crates/flexmesa-schema/src/namelist.rs")
    return sorted(resolved)


def _parse_coverage_map() -> dict[
    str, tuple[port_manifest.PortStatus, list[str], str]
]:
    """Parse per-file classifications from ``notes/audit/coverage-map.md``."""
    cm_path = REPO_ROOT / "notes" / "audit" / "coverage-map.md"
    status_lookup = {
        "ABSENT": port_manifest.PortStatus.ABSENT,
        "STUB": port_manifest.PortStatus.STUB,
        "PARTIAL": port_manifest.PortStatus.PARTIAL,
        "PORTED": port_manifest.PortStatus.PORTED_UNVERIFIED,
    }
    parsed: dict[str, tuple[port_manifest.PortStatus, list[str], str]] = {}
    for line in cm_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 5:
            continue
        up = cols[0].strip("`")
        st_raw = cols[2].strip()
        if "/" not in up or st_raw not in status_lookup:
            continue
        st = status_lookup[st_raw]
        fm_files = _resolve_coverage_map_files(up, cols[3])
        note = f"seeded from coverage-map.md ({st_raw.lower()}): {cols[4]}"
        if st != port_manifest.PortStatus.ABSENT and not fm_files:
            # Upstream compile-without-library stubs with no FlexMESA file.
            st = port_manifest.PortStatus.ABSENT
        if st == port_manifest.PortStatus.ABSENT:
            fm_files = []
        parsed[up] = (st, fm_files, note)
    return parsed


def _classify_non_coverage_map_file(
    relpath: str, module: str
) -> tuple[port_manifest.PortStatus, list[str], str]:
    """Classify files outside the 199 ``coverage-map.md`` engine `.f90` rows."""
    p = port_manifest.PortStatus
    if module in {"gyre", "adipls", "stella", "plotting"}:
        pkg = {
            "gyre": "#170",
            "adipls": "#171",
            "stella": "#172",
            "plotting": "#173",
        }[module]
        return p.ABSENT, [], f"not ported ({pkg})"

    # Engine .inc / other / work files described in aggregate in coverage-map.md
    if relpath.startswith("star_data/public/") and relpath.endswith(".inc"):
        return (
            p.PARTIAL,
            [
                "crates/flexmesa-star/src/model.rs",
                "crates/flexmesa-star/src/hydro_state.rs",
            ],
            "seeded from coverage-map.md aggregate: StarModel subset",
        )
    if relpath.startswith("star_data/private/star_controls"):
        return (
            p.PARTIAL,
            ["crates/flexmesa-star/src/controls.rs"],
            "seeded from coverage-map.md aggregate: &controls subset",
        )
    if relpath.startswith("star_data/private/star_job_controls"):
        return (
            p.PARTIAL,
            ["crates/flexmesa-star/src/job_controls.rs"],
            "seeded from coverage-map.md aggregate: &star_job subset",
        )
    if relpath in {
        "binary/public/binary_controls.inc",
        "binary/private/binary_job_controls.inc",
    }:
        return (
            p.PARTIAL,
            ["crates/flexmesa-binary/src/inlist.rs"],
            "seeded from coverage-map.md aggregate: binary controls subset",
        )
    if relpath == "binary/public/binary_data.inc":
        return (
            p.PARTIAL,
            ["crates/flexmesa-binary/src/system.rs"],
            "seeded from coverage-map.md aggregate: BinarySystem subset",
        )
    if relpath.startswith(
        (
            "star/other/",
            "binary/other/",
            "star/job/",
            "binary/job/",
            "star/work/",
            "binary/work/",
            "astero/work/",
            "astero/private/adipls_support",
            "astero/public/",
        )
    ):
        return p.ABSENT, [], "seeded from coverage-map.md aggregate: absent"

    # Offline builders, preprocessors, plotters, single/quad precision variants
    parts = relpath.split("/")
    if len(parts) >= 2 and (
        parts[1] in {"preprocessor", "plotter", "other"}
        or parts[1].endswith("_builder")
    ):
        return p.ABSENT, [], "offline table builder, plotter, or hook stub"
    if relpath.endswith(("_sg.f90", "_sg.f", "_qp.f90", "_sp.f90", ".c", ".h")):
        return (
            p.ABSENT,
            [],
            "single/quad-precision or C helper not used in f64 Rust",
        )

    # Module-specific file mappings from coverage-map.md section 7
    exact_ported: dict[str, list[str]] = {
        "const/public/const_def.f90": ["crates/flexmesa-const/src/lib.rs"],
        "const/public/const_lib.f90": ["crates/flexmesa-const/src/lib.rs"],
        "turb/private/mlt.f90": ["crates/flexmesa-turb/src/mlt.rs"],
        "turb/private/semiconvection.f90": [
            "crates/flexmesa-turb/src/semiconvection.rs"
        ],
        "turb/private/tdc.f90": ["crates/flexmesa-turb/src/tdc.rs"],
        "turb/private/thermohaline.f90": [
            "crates/flexmesa-turb/src/thermohaline.rs"
        ],
        "turb/public/turb.f90": [
            "crates/flexmesa-turb/src/lib.rs",
            "crates/flexmesa-turb/src/mlt.rs",
            "crates/flexmesa-turb/src/tdc.rs",
            "crates/flexmesa-turb/src/semiconvection.rs",
            "crates/flexmesa-turb/src/thermohaline.rs",
        ],
        "neu/private/mod_neu.f90": [
            "crates/flexmesa-neu/src/lib.rs",
            "crates/flexmesa-neu/src/inputs.rs",
            "crates/flexmesa-neu/src/pair_plas.rs",
            "crates/flexmesa-neu/src/phot.rs",
            "crates/flexmesa-neu/src/brem.rs",
            "crates/flexmesa-neu/src/reco.rs",
        ],
        "neu/public/neu_def.f90": ["crates/flexmesa-neu/src/lib.rs"],
        "neu/public/neu_lib.f90": ["crates/flexmesa-neu/src/lib.rs"],
        "interp_2d/private/bipm_db.f90": [
            "crates/flexmesa-interp2d/src/bipm.rs"
        ],
        "auto_diff/private/auto_diff_real_1var_order1_module.f90": [
            "crates/flexmesa-autodiff/src/order1.rs"
        ],
        "auto_diff/private/auto_diff_real_2var_order1_module.f90": [
            "crates/flexmesa-autodiff/src/order1.rs"
        ],
        "auto_diff/private/auto_diff_real_4var_order1_module.f90": [
            "crates/flexmesa-autodiff/src/order1.rs"
        ],
        "auto_diff/private/auto_diff_real_15var_order1_module.f90": [
            "crates/flexmesa-autodiff/src/order1.rs"
        ],
        "auto_diff/private/auto_diff_real_star_order1_module.f90": [
            "crates/flexmesa-autodiff/src/order1.rs"
        ],
        "auto_diff/private/auto_diff_real_2var_order2_module.f90": [
            "crates/flexmesa-autodiff/src/order2.rs"
        ],
        "auto_diff/private/auto_diff_real_2var_order3_module.f90": [
            "crates/flexmesa-autodiff/src/order3.rs"
        ],
        "auto_diff/public/auto_diff.f90": [
            "crates/flexmesa-autodiff/src/lib.rs"
        ],
    }
    if relpath in exact_ported:
        return (
            p.PORTED_UNVERIFIED,
            exact_ported[relpath],
            "seeded from coverage-map.md section 7 (unverified)",
        )

    absent_micro: tuple[str, ...] = (
        "eos/private/create_EXCOR7_table.f90",
        "eos/private/create_FSCRliq8_table.f90",
        "kap/private/op_",
        "rates/private/reaclib_print.f90",
        "ionization/private/ion_table_plot.f90",
        "colors/private/colors_ctrls_io.f90",
        "colors/private/colors_history.f90",
        "num/private/estrad.dek",
        "num/private/slvrad.dek",
        "num/private/slvseu.dek",
        "num/public/num_radau5_debug_routine.dek",
        "mtx/private/pre_conditioners.f90",
        "mtx/private/DGBSVX_wrapper.f90",
        "auto_diff/private/auto_diff_real_tdc_module.f90",
        "math/private/math_io.f90",
        "math/public/math_lib_crmath.f90",
        "utils/private/utils_dict.f90",
        "utils/private/utils_openmp",
        "utils/private/utils_system.f90",
    )
    if any(relpath.startswith(pref) for pref in absent_micro):
        return p.ABSENT, [], "seeded from coverage-map.md section 7: absent"

    crate_default: dict[str, list[str]] = {
        "eos": ["crates/flexmesa-eos/src/lib.rs"],
        "kap": ["crates/flexmesa-kap/src/lib.rs"],
        "net": ["crates/flexmesa-net/src/lib.rs"],
        "rates": ["crates/flexmesa-rates/src/lib.rs"],
        "neu": [
            "crates/flexmesa-neu/src/pair_plas.rs",
            "crates/flexmesa-neu/src/phot.rs",
            "crates/flexmesa-neu/src/brem.rs",
            "crates/flexmesa-neu/src/reco.rs",
        ],
        "atm": ["crates/flexmesa-atm/src/lib.rs"],
        "turb": ["crates/flexmesa-turb/src/tdc_support.rs"],
        "chem": ["crates/flexmesa-chem/src/lib.rs"],
        "ionization": ["crates/flexmesa-ionization/src/lib.rs"],
        "colors": ["crates/flexmesa-colors/src/lib.rs"],
        "num": ["crates/flexmesa-num/src/lib.rs"],
        "mtx": ["crates/flexmesa-mtx/src/lib.rs"],
        "interp_1d": ["crates/flexmesa-interp1d/src/lib.rs"],
        "interp_2d": ["crates/flexmesa-interp2d/src/lib.rs"],
        "auto_diff": ["crates/flexmesa-autodiff/src/lib.rs"],
        "math": ["crates/flexmesa-math/src/lib.rs"],
        "const": ["crates/flexmesa-const/src/lib.rs"],
        "utils": ["crates/flexmesa-utils/src/lib.rs"],
    }
    return (
        p.PARTIAL,
        crate_default[module],
        "seeded from coverage-map.md section 7: partial / uncompared",
    )


def _build_routine_map(
    upstream_path: str,
    source_text: str,
    flexmesa_files: list[str],
    sym_cache: dict[str, set[str]],
) -> tuple[dict[str, str], list[str]]:
    """Match public Fortran procedures to symbols in ``flexmesa_files``."""
    suffix = pathlib.PurePosixPath(upstream_path).suffix
    pubs = _port_parsers.extract_fortran_public_routines(source_text, suffix)
    hints = ROUTINE_COUNTERPART_HINTS.get(upstream_path, {})

    for rel_fm in flexmesa_files:
        if rel_fm not in sym_cache:
            sym_cache[rel_fm] = _port_parsers.extract_flexmesa_symbols(
                REPO_ROOT / rel_fm
            )

    rmap: dict[str, str] = {}
    missing: list[str] = []
    for proc in pubs:
        if proc in hints:
            target = hints[proc]
            rel_fm, syms_raw = target.split("::", 1)
            if rel_fm in flexmesa_files and all(
                s in sym_cache[rel_fm] for s in syms_raw.split("+")
            ):
                rmap[proc] = target
                continue
        found: str | None = None
        proc_low = proc.lower()
        for rel_fm in flexmesa_files:
            by_low = {s.lower(): s for s in sym_cache[rel_fm]}
            if proc_low in by_low:
                found = f"{rel_fm}::{by_low[proc_low]}"
                break
        if found is not None:
            rmap[proc] = found
        else:
            missing.append(proc)
    return rmap, missing


def build_manifest(mesa_zip: pathlib.Path, out_path: pathlib.Path) -> None:
    """Build and validate ``port_manifest.toml`` from ``mesa_zip``."""
    actual_md5 = _md5_file(mesa_zip)
    if actual_md5 != port_manifest.EXPECTED_ARCHIVE_MD5:
        raise ValueError(
            f"{mesa_zip.name} MD5 {actual_md5} != "
            f"{port_manifest.EXPECTED_ARCHIVE_MD5}"
        )

    sources = _load_mesa_zip_sources(mesa_zip)
    cm_rows = _parse_coverage_map()

    # Update #163 history/profile files from pre-#163 coverage-map entries
    post_audit_partials: dict[str, tuple[list[str], str]] = {
        "star/private/history.f90": (
            ["crates/flexmesa-star/src/mesa_history.rs"],
            (
                "updated for #163: selectable history_columns.list writer "
                "(partial; microphysics columns under #161)"
            ),
        ),
        "star/private/history_specs.f90": (
            ["crates/flexmesa-star/src/mesa_history.rs"],
            (
                "updated for #163: recursive history_columns.list parser "
                "(partial; #161)"
            ),
        ),
        "star/private/profile.f90": (
            ["crates/flexmesa-star/src/mesa_profile.rs"],
            (
                "updated for #163: profileN.data and profiles.index priority "
                "writer (partial; #161)"
            ),
        ),
        "star/private/profile_getval.f90": (
            ["crates/flexmesa-star/src/mesa_profile.rs"],
            "updated for #163: profile column evaluator (partial; #161)",
        ),
        "binary/private/binary_history.f90": (
            ["crates/flexmesa-binary/src/binary_history.rs"],
            (
                "updated for #163: binary_history.data writer "
                "(partial; CE columns under #160)"
            ),
        ),
        "binary/private/binary_history_specs.f90": (
            ["crates/flexmesa-binary/src/binary_history.rs"],
            (
                "updated for #163: binary_history_columns.list parser "
                "(partial; #160)"
            ),
        ),
    }

    sym_cache: dict[str, set[str]] = {}
    entries: list[port_manifest.PortManifestEntry] = []

    for relpath in sorted(sources):
        mod, nlines, sha, text = sources[relpath]
        if relpath in post_audit_partials:
            fm_files, note = post_audit_partials[relpath]
            status = port_manifest.PortStatus.PARTIAL
        elif relpath in cm_rows:
            status, fm_files, note = cm_rows[relpath]
        else:
            status, fm_files, note = _classify_non_coverage_map_file(
                relpath, mod
            )

        rmap_tuple: tuple[tuple[str, str], ...] = ()
        if status == port_manifest.PortStatus.PORTED_UNVERIFIED:
            # Expand all .rs files needed by hints if hinted file is in crate
            hinted_files = {
                t.split("::", 1)[0]
                for t in ROUTINE_COUNTERPART_HINTS.get(relpath, {}).values()
            }
            merged_fm = sorted(set(fm_files) | hinted_files)
            rmap, missing = _build_routine_map(
                relpath, text, merged_fm, sym_cache
            )
            if missing:
                status = port_manifest.PortStatus.PARTIAL
                note = (
                    f"{note}; downgraded to partial (unmapped public "
                    f"routines: {', '.join(missing[:5])})"
                )
            else:
                fm_files = merged_fm
                rmap_tuple = tuple(sorted(rmap.items()))

        entries.append(
            port_manifest.PortManifestEntry(
                upstream_path=relpath,
                module=mod,
                upstream_lines=nlines,
                upstream_sha256=sha,
                status=status,
                flexmesa_files=tuple(fm_files),
                routine_map=rmap_tuple,
                notes=note,
            )
        )

    manifest = port_manifest.PortManifest(
        schema_version=1,
        upstream_version=port_manifest.EXPECTED_UPSTREAM_VERSION,
        upstream_commit=port_manifest.EXPECTED_UPSTREAM_COMMIT,
        archive_md5=port_manifest.EXPECTED_ARCHIVE_MD5,
        entries=tuple(entries),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        port_manifest.dump_port_manifest(manifest), encoding="utf-8"
    )
    print(f"Wrote {len(entries)} entries to {out_path.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesa-zip",
        type=pathlib.Path,
        required=True,
        help="Path to mesa-26.04.1.zip (MD5 994fd28f38da92efc80e38ca11d8317b).",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=REPO_ROOT / "benchmarks" / "dashboard" / "port_manifest.toml",
        help="Output path for port_manifest.toml.",
    )
    args = parser.parse_args()
    build_manifest(args.mesa_zip.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
