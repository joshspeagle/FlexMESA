# SPDX-License-Identifier: GPL-3.0-or-later
"""Python and JAX tests for Wave 3b.1 4D tabulated EOS and `(P, T)` solvers."""

from __future__ import annotations

import pathlib
import re
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import flexmesa
from flexmesa import chem, constants, eos
from flexmesa import math as fm_math

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_EOS_TEST_OUTPUT = (
    _REPO_ROOT / "tests" / "fixtures" / "golden" / "eos" / ("test_output.txt")
)
_EOS_ORACLES_TSV = (
    _REPO_ROOT
    / "benchmarks"
    / "fortran_mesa"
    / "eos_oracles"
    / "mesa_26_4_1_eos_oracles.tsv"
)
_FLOAT_RE = re.compile(r"[-+]?\d+\.\d+(?:[ED][-+]?\d+)")


def _make_xz_composition(x: float, z: float) -> tuple[list[int], np.ndarray]:
    """Build 7-species composition matching `Init_Composition` in MESA."""
    names = ("h1", "he4", "c12", "n14", "o16", "ne20", "mg24")
    cids = [int(chem.get_nuclide_index(n) or 0) for n in names]
    assert all(c > 0 for c in cids)
    y = max(0.0, 1.0 - (x + z))
    xa = np.array(
        [
            x,
            y,
            z * 0.173312,
            z * 0.053177,
            z * 0.482398,
            z * 0.098675,
            0.0,
        ],
        dtype=np.float64,
    )
    xa[6] = max(0.0, 1.0 - float(np.sum(xa[:6])))
    return cids, xa


def _solar_composition() -> tuple[list[int], np.ndarray]:
    """Build 7-species `(X=0.70, Z=0.02)` composition from `eos_support.f90`."""
    return _make_xz_composition(0.70, 0.02)


def _load_eos_case(case_id: str) -> np.ndarray:
    """Load a `26 x 3` `EOS` block from `mesa_26_4_1_eos_oracles.tsv`."""
    arr = np.zeros((eos.NUM_EOS_BASIC_RESULTS, 3), dtype=np.float64)
    seen = [False] * eos.NUM_EOS_BASIC_RESULTS
    for line in _EOS_ORACLES_TSV.read_text("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if cols[0] == "EOS" and cols[1] == case_id:
            slot = int(cols[2])
            arr[slot] = [float(cols[3]), float(cols[4]), float(cols[5])]
            seen[slot] = True
    assert all(seen), f"Incomplete EOS oracle case: {case_id}"
    return arr


def _load_inv_row(case_id: str) -> tuple[float, ...]:
    """Load an `INV` scalar row from `mesa_26_4_1_eos_oracles.tsv`."""
    for line in _EOS_ORACLES_TSV.read_text("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if cols[0] == "INV" and cols[1] == case_id:
            return tuple(float(x) for x in cols[2:])
    raise AssertionError(f"Missing INV row: {case_id}")


def _parse_upstream_eos_test_output() -> tuple[
    list[tuple[dict[str, float], list[tuple[str, float]]]],
    list[tuple[str, float]],
    list[tuple[str, float]],
    int,
]:
    """Parse all sections of upstream `eos/test/test_output` (`391` floats)."""
    text = _EOS_TEST_OUTPUT.read_text("utf-8")
    total_floats = len(_FLOAT_RE.findall(text))
    lines = text.splitlines()

    doit_blocks: list[tuple[dict[str, float], list[tuple[str, float]]]] = []
    idx = 0
    while idx < len(lines) and not lines[idx].strip().startswith("tolerance"):
        if lines[idx].strip() == "eosDT_get":
            hdr: dict[str, float] = {}
            for k in range(idx - 9, idx):
                parts = lines[k].split()
                if len(parts) == 2:
                    hdr[parts[0]] = float(parts[1].replace("D", "E"))
            vals: list[tuple[str, float]] = []
            for k in range(idx + 1, idx + 13):
                parts = lines[k].split()
                vals.append((parts[0], float(parts[1].replace("D", "E"))))
            doit_blocks.append((hdr, vals))
            idx += 13
        else:
            idx += 1

    inv_entries: list[tuple[str, float]] = []
    while idx < len(lines) and lines[idx].strip() != "test1_eosPT":
        s = lines[idx].strip()
        if s:
            m = _FLOAT_RE.search(s)
            assert m is not None
            key = s[: m.start()].strip()
            inv_entries.append((key, float(m.group(0).replace("D", "E"))))
        idx += 1

    assert lines[idx].strip() == "test1_eosPT"
    idx += 1
    pt_entries: list[tuple[str, float]] = []
    while idx < len(lines) and lines[idx].strip() not in {"T", "F"}:
        s = lines[idx].strip()
        if s:
            parts = s.split()
            assert len(parts) == 2
            pt_entries.append((parts[0], float(parts[1].replace("D", "E"))))
        idx += 1

    blend_floats = sum(len(_FLOAT_RE.findall(ln)) for ln in lines[idx:])
    parsed_count = (
        sum(len(h) + len(v) for h, v in doit_blocks)
        + len(inv_entries)
        + len(pt_entries)
        + blend_floats
    )
    assert parsed_count == total_floats == 391
    return doit_blocks, inv_entries, pt_entries, total_floats


def test_all_320_do_one_trho_scalars_in_upstream_test_output() -> None:
    """Verify all 16 `Do_One_TRho` blocks (`320` floats) in Rust and JAX."""
    doit_blocks, _, _, _ = _parse_upstream_eos_test_output()
    assert len(doit_blocks) == 16

    compositions = (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.75 * (1.0 - 0.03), 0.03),
        (0.70, 0.02),
    )
    with flexmesa.numerics_mode_context("strict"):
        for c_idx, (x_in, z_in) in enumerate(compositions):
            cids, xa = _make_xz_composition(x_in, z_in)
            infos = [chem.get_nuclide_info(c) for c in cids]
            a_int = np.array([float(inf.a) for inf in infos], dtype=np.float64)
            z_int = np.array([float(inf.z) for inf in infos], dtype=np.float64)
            abar = float(1.0 / np.sum(xa / a_int))
            zbar = float(abar * np.sum(xa * z_int / a_int))

            for p_idx in range(4):
                hdr, vals = doit_blocks[c_idx * 4 + p_idx]
                assert len(hdr) == 8 and len(vals) == 12
                np.testing.assert_allclose(
                    float(xa[0]), hdr["X"], rtol=1e-15, atol=0.0
                )
                np.testing.assert_allclose(
                    float(xa[1]), hdr["Y"], rtol=1e-15, atol=0.0
                )
                np.testing.assert_allclose(
                    float(np.sum(xa[2:])), hdr["Z"], rtol=1e-14, atol=1e-15
                )
                np.testing.assert_allclose(abar, hdr["abar"], rtol=1e-15)
                np.testing.assert_allclose(zbar, hdr["zbar"], rtol=1e-15)

                rho = float(fm_math.exp10(hdr["logRho"]))
                temp = hdr["T6"] * 1.0e6
                np.testing.assert_allclose(
                    float(np.log10(temp)), hdr["logT"], rtol=1e-14
                )

                out_r = eos.eos_dt_get(rho, temp, cids, xa, backend="rust")
                out_j = eos.eos_dt_get(
                    jnp.float64(rho),
                    jnp.float64(temp),
                    cids,
                    jnp.asarray(xa),
                    backend="jax",
                )
                jax_rtol = 1.0e-12 if p_idx < 3 else 2.0e-10
                for b_name, out, rtol in (
                    ("rust", out_r, 1.0e-13),
                    ("jax", out_j, jax_rtol),
                ):
                    res = np.asarray(out.res)
                    prad = constants.CRAD * (temp**4) / 3.0
                    got_map = {
                        "P": float(np.exp(res[eos.I_LN_PGAS])) + prad,
                        "E": float(np.exp(res[eos.I_LN_E])),
                        "S": float(np.exp(res[eos.I_LN_S])),
                        "mu": float(res[eos.I_MU]),
                        "lnfree_e": float(res[eos.I_LN_FREE_E]),
                        "eta": float(res[eos.I_ETA]),
                        "grad_ad": float(res[eos.I_GRAD_AD]),
                        "chiRho": float(res[eos.I_CHI_RHO]),
                        "chiT": float(res[eos.I_CHI_T]),
                        "gamma1": float(res[eos.I_GAMMA1]),
                        "gamma3": float(res[eos.I_GAMMA3]),
                    }
                    for v_name, exp_v in vals:
                        got_v = got_map[v_name]
                        if exp_v == 0.0:
                            assert abs(got_v) <= (
                                0.0 if b_name == "rust" else 1.0e-15
                            )
                        else:
                            np.testing.assert_allclose(
                                got_v, exp_v, rtol=rtol, atol=0.0
                            )


def test_all_56_inversion_and_eospt_scalars_in_upstream_test_output() -> None:
    """Verify `test_get_Rho_T` (`16`) and `test1_eosPT` (`40`) scalars."""
    _, inv_entries, pt_entries, total_floats = _parse_upstream_eos_test_output()
    assert total_floats == 391
    assert len(inv_entries) == 16
    assert len(pt_entries) == 40

    cids, xa = _solar_composition()

    # 1. `test_get_Rho_T` (16 scalars)
    tol_name, tol_val = inv_entries[0]
    assert tol_name == "tolerance"
    np.testing.assert_allclose(tol_val, 1.0e-5, rtol=1e-15, atol=0.0)

    r_blk = dict(inv_entries[1:6])
    ts_blk = dict(inv_entries[6:11])
    tp_blk = dict(inv_entries[11:16])

    ln_s_wanted = float(np.log(2.968064512e9))
    np.testing.assert_allclose(
        ln_s_wanted / constants.LN10, r_blk["wanted logS"], rtol=1e-15
    )
    np.testing.assert_allclose(
        ln_s_wanted / constants.LN10, ts_blk["wanted logS"], rtol=1e-15
    )

    with flexmesa.numerics_mode_context("strict"):
        inv_rho = eos.eos_dt_get_rho(
            float(fm_math.exp10(4.8)),
            eos.I_LN_S,
            ln_s_wanted,
            cids,
            xa,
            log_rho_guess=r_blk["guess logRho"],
        )
        np.testing.assert_allclose(
            r_blk["actual logRho"], -7.0, rtol=1e-15, atol=0.0
        )
        np.testing.assert_allclose(
            float(inv_rho.log10_result), r_blk["found logRho"], rtol=1e-9
        )
        np.testing.assert_allclose(
            float(inv_rho.eos.res[eos.I_LN_S]) / constants.LN10,
            r_blk["got logS"],
            rtol=1e-11,
        )

        inv_ts = eos.eos_dt_get_t(
            1.0e-7,
            eos.I_LN_S,
            ln_s_wanted,
            cids,
            xa,
            log_t_guess=ts_blk["guess logT"],
        )
        np.testing.assert_allclose(
            ts_blk["actual logT"], 4.8, rtol=1e-15, atol=0.0
        )
        np.testing.assert_allclose(
            float(inv_ts.log10_result), ts_blk["found logT"], rtol=1e-9
        )
        np.testing.assert_allclose(
            float(inv_ts.eos.res[eos.I_LN_S]) / constants.LN10,
            ts_blk["got logS"],
            rtol=1e-11,
        )

        t_48 = float(fm_math.exp10(tp_blk["actual logT"]))
        prad_48 = constants.CRAD * (t_48**4) / 3.0
        wanted_log_p = float(
            np.log10(prad_48 + np.exp(inv_ts.eos.res[eos.I_LN_PGAS]))
        )
        np.testing.assert_allclose(
            wanted_log_p, tp_blk["wanted logP"], rtol=1e-10
        )
        inv_tp = eos.eos_dt_get_t(
            1.0e-7,
            -2,
            tp_blk["wanted logP"],
            cids,
            xa,
            log_t_guess=tp_blk["guess logT"],
        )
        assert abs(float(inv_tp.log10_result) - tp_blk["found logT"]) < tol_val
        t_tp = float(fm_math.exp10(float(inv_tp.log10_result)))
        got_log_p = float(
            np.log10(
                constants.CRAD * (t_tp**4) / 3.0
                + np.exp(inv_tp.eos.res[eos.I_LN_PGAS])
            )
        )
        np.testing.assert_allclose(got_log_p, tp_blk["wanted logP"], rtol=1e-12)

        # 2. `test1_eosPT` (40 scalars across both Rust `strict` and JAX)
        pt_hdr = dict(pt_entries[:8])
        pt_state = dict(pt_entries[8:12])
        pt_part = dict(pt_entries[12:14])
        pt_eos_rows = pt_entries[14:]
        assert len(pt_eos_rows) == 26

        pgas_in = float(fm_math.exp10(pt_hdr["logPgas"]))
        temp_in = float(fm_math.exp10(pt_hdr["logT"]))
        np.testing.assert_allclose(temp_in, pt_hdr["T"], rtol=1e-15)
        np.testing.assert_allclose(pgas_in, pt_hdr["Pgas"], rtol=1e-15)
        np.testing.assert_allclose(float(xa[0]), pt_hdr["X"], rtol=1e-15)
        np.testing.assert_allclose(
            float(np.sum(xa[2:])), pt_hdr["Z"], rtol=1e-14, atol=1e-15
        )
        np.testing.assert_allclose(pt_state["logT"], pt_hdr["logT"], rtol=1e-15)
        np.testing.assert_allclose(
            pt_state["logPgas"], pt_hdr["logPgas"], rtol=1e-15
        )

        pt_r = eos.eos_pt_get(pgas_in, temp_in, cids, xa, backend="rust")
    pt_j = eos.eos_pt_get(
        jnp.float64(pgas_in),
        jnp.float64(temp_in),
        cids,
        jnp.asarray(xa),
        backend="jax",
    )

    for b_name, pt_out, rtol in (
        ("rust", pt_r, 1.0e-12),
        ("jax", pt_j, 1.0e-11),
    ):
        np.testing.assert_allclose(
            float(pt_out.rho), pt_state["rho"], rtol=rtol, atol=0.0
        )
        np.testing.assert_allclose(
            float(pt_out.log10_rho), pt_state["logRho"], rtol=rtol, atol=0.0
        )
        np.testing.assert_allclose(
            float(pt_out.dln_rho_dln_pgas_const_t),
            pt_part["dlnRho_dlnPgas_c_T"],
            rtol=rtol,
            atol=0.0,
        )
        np.testing.assert_allclose(
            float(pt_out.dln_rho_dln_t_const_pgas),
            pt_part["dlnRho_dlnT_c_Pgas"],
            rtol=rtol,
            atol=0.0,
        )

        res = np.asarray(pt_out.eos.res)
        prad = constants.CRAD * (temp_in**4) / 3.0
        got_slots = [
            pgas_in + prad,
            float(np.exp(res[eos.I_LN_E])),
            float(np.exp(res[eos.I_LN_S])),
            *[float(res[k]) for k in range(3, eos.NUM_EOS_BASIC_RESULTS)],
        ]
        for (slot_name, exp_val), got_val in zip(
            pt_eos_rows, got_slots, strict=True
        ):
            if exp_val == 0.0:
                assert abs(got_val) <= (0.0 if b_name == "rust" else 1.0e-15), (
                    f"{b_name} {slot_name}: expected 0.0, got {got_val}"
                )
            else:
                np.testing.assert_allclose(
                    got_val, exp_val, rtol=rtol, atol=0.0
                )


def test_get_t_logp_matches_early_stopped_upstream_iterate() -> None:
    """Check `eos_dt_get_t` on `logP` (`othertol=1e-12`, `tol=1e-5`)."""
    _, inv_entries, _, _ = _parse_upstream_eos_test_output()
    tp_blk = dict(inv_entries[11:16])
    cids, xa = _solar_composition()
    with flexmesa.numerics_mode_context("strict"):
        inv_tp = eos.eos_dt_get_t(
            1.0e-7,
            -2,
            tp_blk["wanted logP"],
            cids,
            xa,
            log_t_guess=tp_blk["guess logT"],
            logt_tol=1.0e-5,
            other_tol=1.0e-12,
        )
    np.testing.assert_allclose(
        float(inv_tp.log10_result), tp_blk["found logT"], rtol=1e-12, atol=0.0
    )


def test_standalone_4d_tables_and_pt_get_t_match_mesa_26_4_1_oracle() -> None:
    """Verify standalone `OPAL/SCVH`, `FreeEOS`, `CMS`, and `eos_pt_get_t`."""
    cids, xa = _solar_composition()
    with flexmesa.numerics_mode_context("strict"):
        for case_id, eval_fn, rho, temp, der_rtol in (
            (
                "opal_scvh_solar_1em2_1e5",
                eos.eval_opal_scvh_eos,
                1.0e-2,
                1.0e5,
                1.0e-13,
            ),
            (
                "free_eos_solar_1em2_1e6",
                eos.eval_free_eos,
                1.0e-2,
                1.0e6,
                1.0e-13,
            ),
            (
                "cms_solar_1em2_1e5",
                eos.eval_cms_eos,
                1.0e-2,
                1.0e5,
                2.0e-12,
            ),
        ):
            exp = _load_eos_case(case_id)
            out_r = eval_fn(rho, temp, cids, xa, backend="rust")
            out_j = eval_fn(
                jnp.float64(rho),
                jnp.float64(temp),
                cids,
                jnp.asarray(xa),
                backend="jax",
            )
            for out in (out_r, out_j):
                np.testing.assert_allclose(
                    np.asarray(out.res), exp[:, 0], rtol=1.0e-13, atol=1.0e-15
                )
                np.testing.assert_allclose(
                    np.asarray(out.d_dlnd),
                    exp[:, 1],
                    rtol=der_rtol,
                    atol=1.0e-14,
                )
                np.testing.assert_allclose(
                    np.asarray(out.d_dln_t),
                    exp[:, 2],
                    rtol=der_rtol,
                    atol=1.0e-14,
                )

            def _ln_p(
                r: jax.Array, t: jax.Array, fn: Any = eval_fn
            ) -> jax.Array:
                res: jax.Array = fn(
                    r, t, cids, jnp.asarray(xa), backend="jax"
                ).res
                return res[eos.I_LN_PGAS]

            r0 = jnp.asarray(rho, dtype=jnp.float64)
            t0 = jnp.asarray(temp, dtype=jnp.float64)
            dr = jnp.asarray(rho * 1.0e-2, dtype=jnp.float64)
            dt = jnp.asarray(temp * 1.0e-2, dtype=jnp.float64)
            _, dln_p = jax.jvp(_ln_p, (r0, t0), (dr, dt))
            expected_jvp = 1.0e-2 * (
                float(out_r.d_dlnd[eos.I_LN_PGAS])
                + float(out_r.d_dln_t[eos.I_LN_PGAS])
            )
            np.testing.assert_allclose(float(dln_p), expected_jvp, rtol=1.0e-12)

        exp_pt_t = _load_inv_row("solar_pt_get_t_15_7p2")
        pt_helm = eos.eos_pt_get(
            1.0e15, float(fm_math.exp10(7.2)), cids, xa, backend="rust"
        )
        target_lne = float(pt_helm.eos.res[eos.I_LN_E])
        inv_t = eos.eos_pt_get_t(
            1.0e15, eos.I_LN_E, target_lne, cids, xa, log_t_guess=7.0
        )
        np.testing.assert_allclose(
            float(inv_t.log10_result), exp_pt_t[0], rtol=1.0e-12, atol=0.0
        )
        np.testing.assert_allclose(
            float(inv_t.eos.res[eos.I_LN_E]),
            exp_pt_t[2],
            rtol=1.0e-12,
            atol=0.0,
        )

        inv_pgas = eos.eos_pt_get_pgas(
            float(fm_math.exp10(7.2)),
            eos.I_LN_E,
            target_lne,
            cids,
            xa,
            log_pgas_guess=14.8,
        )
        np.testing.assert_allclose(
            float(inv_pgas.log10_result), 15.0, rtol=1.0e-5
        )


def test_eval_cms_eos_default_matches_mesa_26_4_1_default_radiation() -> None:
    """Check `eval_cms_eos` default and `norad` against MESA `26.4.1`."""
    cids, xa = _solar_composition()
    exp_cms = _load_eos_case("cms_solar_1em2_1e5")
    exp_norad = _load_eos_case("cms_solar_1em2_1e5_norad")
    with flexmesa.numerics_mode_context("strict"):
        out_r = eos.eval_cms_eos(1.0e-2, 1.0e5, cids, xa, backend="rust")
        out_nr = eos.eval_cms_eos(
            1.0e-2, 1.0e5, cids, xa, include_radiation=False, backend="rust"
        )
    out_nj = eos.eval_cms_eos(
        jnp.float64(1.0e-2),
        jnp.float64(1.0e5),
        cids,
        jnp.asarray(xa),
        include_radiation=False,
        backend="jax",
    )
    np.testing.assert_allclose(
        np.asarray(out_r.res), exp_cms[:, 0], rtol=1.0e-12, atol=1.0e-14
    )
    for out_n in (out_nr, out_nj):
        np.testing.assert_allclose(
            np.asarray(out_n.res), exp_norad[:, 0], rtol=1.0e-12, atol=1.0e-14
        )


def test_eos_pt_get_jax_ift_jvp_matches_finite_differences() -> None:
    """Verify JAX IFT `custom_jvp` on `eos_pt_get` against finite diff."""
    cids, xa = _solar_composition()

    def _lrho_fn(pg: jax.Array, t: jax.Array) -> jax.Array:
        return jnp.asarray(
            eos.eos_pt_get(
                pg, t, cids, jnp.asarray(xa), backend="jax"
            ).log10_rho
        )

    p0_f = 10.0**4.2
    t0_f = 10.0**3.0
    p0 = jnp.asarray(p0_f, dtype=jnp.float64)
    t0 = jnp.asarray(t0_f, dtype=jnp.float64)
    dp = p0 * 1.0e-3
    dt = t0 * 2.0e-3
    _, dlrho = jax.jvp(_lrho_fn, (p0, t0), (dp, dt))
    eps = 1.0e-3
    lr_plus = eos.eos_pt_get(
        p0_f * (1.0 + eps * 1.0e-3),
        t0_f * (1.0 + eps * 2.0e-3),
        cids,
        xa,
        backend="rust",
    ).log10_rho
    lr_minus = eos.eos_pt_get(
        p0_f * (1.0 - eps * 1.0e-3),
        t0_f * (1.0 - eps * 2.0e-3),
        cids,
        xa,
        backend="rust",
    ).log10_rho
    fd_dlrho = float(lr_plus - lr_minus) / (2.0 * eps)
    np.testing.assert_allclose(float(dlrho), fd_dlrho, rtol=1.0e-5)


def test_4d_tables_and_eos_dt_get_across_stellar_profiles_and_autodiff() -> (
    None
):
    """Verify full 4D tables & `eos_dt_get` across 4 stellar profiles and AD."""
    with flexmesa.numerics_mode_context("strict"):
        for x_val, z_val, log_rho, log_t in [
            (0.72, 0.015, -3.25, 4.35),
            (0.55, 0.025, -1.15, 5.45),
            (0.35, 0.030, 0.45, 6.15),
        ]:
            cids, xa = _make_xz_composition(x_val, z_val)
            rho = 10.0**log_rho
            temp = 10.0**log_t
            for fn in (
                eos.eval_opal_scvh_eos,
                eos.eval_free_eos,
                eos.eval_cms_eos,
            ):
                out_r = fn(rho, temp, cids, xa, backend="rust")
                out_j = fn(
                    jnp.float64(rho),
                    jnp.float64(temp),
                    cids,
                    jnp.asarray(xa),
                    backend="jax",
                )
                np.testing.assert_allclose(
                    out_r.res, np.asarray(out_j.res), rtol=1.0e-12, atol=1.0e-12
                )
                np.testing.assert_allclose(
                    out_r.d_dlnd,
                    np.asarray(out_j.d_dlnd),
                    rtol=1.0e-10,
                    atol=1.0e-12,
                )
                np.testing.assert_allclose(
                    out_r.d_dln_t,
                    np.asarray(out_j.d_dln_t),
                    rtol=1.0e-10,
                    atol=1.0e-12,
                )

        cids_sol, xa_sol = _solar_composition()
        cids_wd, xa_wd = _make_xz_composition(0.0, 1.0)
        profiles = [
            (
                cids_sol,
                xa_sol,
                [
                    (-6.8, 3.76),
                    (-4.2, 4.45),
                    (-1.5, 5.75),
                    (0.5, 6.65),
                    (2.18, 7.19),
                ],
            ),
            (
                cids_sol,
                xa_sol,
                [
                    (-5.0, 3.35),
                    (-2.4, 4.10),
                    (-0.2, 4.95),
                    (1.6, 5.70),
                    (2.85, 6.45),
                ],
            ),
            (
                cids_wd,
                xa_wd,
                [
                    (-3.0, 4.50),
                    (0.0, 5.60),
                    (2.5, 6.40),
                    (4.8, 6.95),
                    (6.5, 7.15),
                ],
            ),
            (
                cids_sol,
                xa_sol,
                [
                    (-7.5, 4.45),
                    (-5.2, 5.30),
                    (-3.0, 6.20),
                    (-0.8, 7.05),
                    (0.65, 7.58),
                ],
            ),
        ]
        for cids_p, xa_p, pts in profiles:
            for lr, lt in pts:
                r_v, t_v = 10.0**lr, 10.0**lt
                dr = eos.eos_dt_get(r_v, t_v, cids_p, xa_p, backend="rust")
                dj = eos.eos_dt_get(
                    jnp.float64(r_v),
                    jnp.float64(t_v),
                    cids_p,
                    jnp.asarray(xa_p),
                    backend="jax",
                )
                assert not np.any(np.isnan(np.asarray(dr.res)))
                np.testing.assert_allclose(
                    dr.res[:3],
                    np.asarray(dj.res[:3]),
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
                np.testing.assert_allclose(
                    dr.res, np.asarray(dj.res), rtol=1.0e-8, atol=1.0e-10
                )

    r_ad = jnp.float64(10.0 ** (-1.15))
    t_ad = jnp.float64(10.0**5.45)
    out_ad = eos.eos_dt_get(
        r_ad, t_ad, cids_sol, jnp.asarray(xa_sol), backend="jax"
    )
    jac_r, jac_t = jax.jacfwd(
        lambda r, t: jnp.asarray(
            eos.eos_dt_get(
                r, t, cids_sol, jnp.asarray(xa_sol), backend="jax"
            ).res
        ),
        argnums=(0, 1),
    )(r_ad, t_ad)
    np.testing.assert_allclose(
        np.asarray(out_ad.d_dlnd[:16]),
        np.asarray(jac_r * r_ad)[:16],
        rtol=2.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(out_ad.d_dln_t[:16]),
        np.asarray(jac_t * t_ad)[:16],
        rtol=2.0e-12,
        atol=1.0e-12,
    )

    cids_unvis, xa_unvis = _make_xz_composition(0.15, 0.08)
    cms_j = eos.eval_cms_eos(
        jnp.float64(1.0),
        jnp.float64(1.0e5),
        cids_unvis,
        jnp.asarray(xa_unvis),
        backend="jax",
    )
    cms_r = eos.eval_cms_eos(1.0, 1.0e5, cids_unvis, xa_unvis, backend="rust")
    np.testing.assert_allclose(
        np.asarray(cms_j.res[:4]), cms_r.res[:4], rtol=1.0e-12, atol=1.0e-12
    )
    jit_dt = jax.jit(
        lambda r, t: eos.eos_dt_get(r, t, cids_sol, xa_sol, backend="jax").res
    )
    np.testing.assert_allclose(
        np.asarray(jit_dt(r_ad, t_ad)),
        np.asarray(out_ad.res),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_helm_ideal_edge_blend_matches_mesa_26_4_1_across_rust_and_jax() -> (
    None
):
    """Verify `#176` `HELM`/`ideal` edge blend and `F2-6` off-table skip."""
    names = ("h1", "he4", "c12", "n14", "o16", "ne20", "mg24")
    cids = [int(chem.get_nuclide_index(n) or 0) for n in names]
    xa_c12 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    exp = _load_eos_case("eosdt_helm_ideal_hi_rho_blend")

    with flexmesa.numerics_mode_context("strict"):
        rho = float(flexmesa._rust.exp10(15.05))
        temp = float(flexmesa._rust.exp10(12.5))
        out_r = eos.eos_dt_get(
            rho,
            temp,
            cids,
            xa_c12,
            use_skye=False,
            use_pc=False,
            use_cms=False,
            use_free_eos=False,
            use_opal_scvh=False,
            backend="rust",
        )
    out_j = eos.eos_dt_get(
        jnp.float64(rho),
        jnp.float64(temp),
        cids,
        jnp.asarray(xa_c12),
        use_skye=False,
        use_pc=False,
        use_cms=False,
        use_free_eos=False,
        use_opal_scvh=False,
        backend="jax",
    )
    for b_name, out, der_rtol in (
        ("rust", out_r, 1.0e-12),
        ("jax", out_j, 5.0e-6),
    ):
        np.testing.assert_allclose(
            np.asarray(out.res),
            exp[:, 0],
            rtol=1.0e-10 if b_name == "jax" else 1.0e-12,
            atol=1.0e-14,
        )
        np.testing.assert_allclose(
            np.asarray(out.d_dlnd), exp[:, 1], rtol=der_rtol, atol=1.0e-13
        )
        np.testing.assert_allclose(
            np.asarray(out.d_dln_t), exp[:, 2], rtol=der_rtol, atol=1.0e-13
        )

    out_off, _, _ = eos.eval_helm_eos(
        jnp.float64(100.0),
        jnp.float64(100.0),
        jnp.float64(4.0),
        jnp.float64(2.0),
        backend="jax",
    )
    assert bool(jnp.all(jnp.isnan(out_off.res)))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#295: AutoDiffReal2VarOrder3 evaluates MESA 26.4.1's auto_diff "
        "expressions (#349), which moves the cancellation-dominated de_dd "
        "of the Rust ideal-gas EOS by a few ULPs of its 5.7e10 terms; the "
        "JAX twin keeps its own rounding."
    ),
)
def test_helm_ideal_edge_skip_blend_rust_and_jax_agree() -> None:
    """Rust and JAX agree on the `F2-6` off-table ideal-gas skip blend."""
    cids_sol, xa_sol = _solar_composition()
    with flexmesa.numerics_mode_context("strict"):
        t_lo = float(flexmesa._rust.exp10(2.95))
        out_skip_r = eos.eos_dt_get(
            1.0,
            t_lo,
            cids_sol,
            xa_sol,
            use_skye=False,
            use_pc=False,
            use_cms=False,
            use_free_eos=False,
            use_opal_scvh=False,
            backend="rust",
        )
    out_skip_j = eos.eos_dt_get(
        jnp.float64(1.0),
        jnp.float64(t_lo),
        cids_sol,
        jnp.asarray(xa_sol),
        use_skye=False,
        use_pc=False,
        use_cms=False,
        use_free_eos=False,
        use_opal_scvh=False,
        backend="jax",
    )
    assert float(out_skip_r.res[eos.I_FRAC_IDEAL]) == 1.0
    assert float(out_skip_j.res[eos.I_FRAC_IDEAL]) == 1.0
    # `de_dd` (`res[11]`) subtracts `df_dd - T * d2f_dtdd` (`~5.7e10`), leaving
    # `~8.4e-6` (`~1.5 ULP * 5.7e10`) roundoff between `crlibm` and XLA.
    np.testing.assert_allclose(
        np.asarray(out_skip_r.res),
        np.asarray(out_skip_j.res),
        rtol=1.0e-12,
        atol=2.0e-5,
    )
