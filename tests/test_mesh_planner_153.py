# SPDX-License-Identifier: LGPL-3.0-or-later
"""Python integration tests for MESA 26.4.1 mesh planning and remap (#153)."""

from __future__ import annotations

import pathlib
import struct
import tempfile

import numpy as np

from flexmesa import star

_MESH_CONTROLS_INLIST: str = "\n".join(
    [
        "&star_job",
        "/",
        "&eos",
        "/",
        "&kap",
        "  Zbase = 0.02d0",
        "/",
        "&controls",
        "  mesh_delta_coeff = 0.8d0",
        "  mesh_max_allowed_ratio = 2.5d0",
        "  max_dq = 1.0d-2",
        "  min_dq = 1.0d-14",
        "  log_tau_function_weight = 15.0d0",
        "  log_kap_function_weight = 12.0d0",
        "  xa_function_species(1) = 'he4'",
        "  xa_function_weight(1) = 30.0d0",
        "  xa_function_param(1) = 1.0d-2",
        "  mesh_dlog_pp_dlogP_extra = 0.25d0",
        "  mesh_dlog_cno_dlogP_extra = 0.25d0",
        "  mesh_logX_species(1) = 'h1'",
        "  mesh_logX_min_for_extra(1) = -6.0d0",
        "  mesh_dlogX_dlogP_extra(1) = 0.25d0",
        "  mesh_delta_coeff_for_highT = 1.5d0",
        "  mesh_Pgas_div_P_exponent = 0.25d0",
        "  use_split_merge_amr = .false.",
        "  split_merge_amr_nz_baseline = 600",
        "  split_merge_amr_max_iters = 40",
        "/",
        "",
    ]
)


def test_zams_adjust_mesh_conserves_species_mass_and_total_dq() -> None:
    """Verify StarModel.adjust_mesh conserves species mass and sum(dq) == 1."""
    model = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
    model.eval_hydro_vars()
    dq_before = np.asarray(model.dq, dtype=np.float64)
    xa_before = np.asarray(model.xa, dtype=np.float64)
    species_mass_before = xa_before.T @ dq_before

    splits, merges = model.adjust_mesh()
    assert splits >= 0
    assert merges >= 0
    assert splits + merges > 0

    dq_after = np.asarray(model.dq, dtype=np.float64)
    xa_after = np.asarray(model.xa, dtype=np.float64)
    species_mass_after = xa_after.T @ dq_after

    assert abs(float(np.sum(dq_after)) - 1.0) < 1.0e-13
    max_species_err = float(
        np.max(np.abs(species_mass_after - species_mass_before))
    )
    assert max_species_err < 1.0e-13
    max_row_sum_err = float(np.max(np.abs(np.sum(xa_after, axis=1) - 1.0)))
    assert max_row_sum_err < 1.0e-13


def test_photo_checkpoint_v22_header_and_restart_bit_identity() -> None:
    """Verify FMPHOTO2 header version (v24 since #159) and restart parity."""
    model = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
    photo_bytes = model.to_photo_bytes()
    assert photo_bytes[:8] == b"FMPHOTO2"
    (version,) = struct.unpack("<I", photo_bytes[8:12])
    assert version == 24

    reloaded = star.StarModel.from_photo_bytes(photo_bytes)
    assert reloaded.nz == model.nz
    assert np.array_equal(
        np.asarray(reloaded.dq, dtype=np.float64),
        np.asarray(model.dq, dtype=np.float64),
    )

    c_direct, c_restarted = model.verify_photo_restart(n_steps=1)
    assert c_direct == c_restarted


def test_parse_inlist_bundle_accepts_mesa_26_4_1_mesh_and_amr_controls() -> (
    None
):
    """Verify strict inlist parsing accepts #153 mesh and AMR controls."""
    with tempfile.TemporaryDirectory() as tmpdir:
        inlist_path = pathlib.Path(tmpdir) / "inlist"
        inlist_path.write_text(_MESH_CONTROLS_INLIST, encoding="utf-8")
        parsed = star.parse_inlist_bundle(
            inlist_path, allow_unsupported_controls=False
        )
    assert parsed["okay_to_remesh"] is True
    assert sorted(parsed["sections"]) == [
        "controls",
        "eos",
        "kap",
        "star_job",
    ]
