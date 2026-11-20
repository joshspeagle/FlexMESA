# SPDX-License-Identifier: LGPL-3.0-or-later
"""Provenance, cache isolation, and corruption tests for `flexmesa.data`."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import pathlib
import struct
import subprocess
import sys
import tarfile
import types

import h5py
import numpy as np
import pytest

from flexmesa import cli
from flexmesa.data import fetch
from flexmesa.kap import tables_jax

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_build_physics_module() -> types.ModuleType:
    """Import `scripts/data/build_physics_bundles.py` without editing path."""
    script_path = _REPO_ROOT / "scripts" / "data" / "build_physics_bundles.py"
    spec = importlib.util.spec_from_file_location(
        "build_physics_bundles", script_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_physics_bundles"] = mod
    spec.loader.exec_module(mod)
    return mod


def _compress_zst(payload: bytes) -> bytes:
    """Compress raw bytes with stdlib zstd or zstandard."""
    try:
        from compression import zstd as std_zstd

        return bytes(std_zstd.compress(payload))
    except ImportError:
        import zstandard

        return bytes(zstandard.ZstdCompressor().compress(payload))


def _write_synthetic_dt_raw(path: pathlib.Path) -> None:
    """Write a minimal valid 4x4 `FMDTRW01` EOS raw slice."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_t, n_q = 4, 4
    header = struct.pack(
        "<8siddIdddIddd",
        b"FMDTRW01",
        51,
        0.70,
        0.02,
        n_t,
        4.0,
        4.3,
        0.1,
        n_q,
        -5.0,
        -4.4,
        0.2,
    )
    body = bytearray()
    for iq in range(n_q):
        for jt in range(n_t):
            for k in range(16):
                val = 10.0 + 0.05 * iq + 0.1 * jt + 0.01 * k
                body.extend(struct.pack("<d", val))
    path.write_bytes(header + bytes(body))


def _write_mini_bundle_fixture(
    tmp_path: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, bytes, bytes]:
    """Build deterministic staging bundles + custom `bundles.toml`."""
    src_dir = tmp_path / "staging"
    src_dir.mkdir(parents=True, exist_ok=True)
    data_dest = tmp_path / "dest_data"
    test_dest = tmp_path / "dest_tests"

    payload = b"FMHELM01_TEST_PAYLOAD"
    payload_sha = hashlib.sha256(payload).hexdigest()
    single_zst = _compress_zst(payload)
    single_sha = hashlib.sha256(single_zst).hexdigest()
    (src_dir / "mini_helm.bin.zst").write_bytes(single_zst)

    tar_buf = io.BytesIO()
    data_bytes = b"golden_row 1.0\n"
    with tarfile.open(fileobj=tar_buf, mode="w") as tf:
        info = tarfile.TarInfo("neu/test/test_output")
        info.size = len(data_bytes)
        tf.addfile(info, io.BytesIO(data_bytes))
    tar_zst = _compress_zst(tar_buf.getvalue())
    tar_sha = hashlib.sha256(tar_zst).hexdigest()
    (src_dir / "mini_tests.tar.zst").write_bytes(tar_zst)

    hashes_dir = tmp_path / "bundle_hashes"
    hashes_dir.mkdir(parents=True, exist_ok=True)
    data_sha = hashlib.sha256(data_bytes).hexdigest()
    (hashes_dir / "mini_tests.sha256.tsv").write_text(
        f"# sha256\tsize_bytes\trelpath\n"
        f"{data_sha}\t{len(data_bytes)}\tneu/test/test_output\n",
        encoding="utf-8",
    )
    (hashes_dir / "mini_helm.sha256.tsv").write_text(
        f"# sha256\tsize_bytes\trelpath\n"
        f"{payload_sha}\t{len(payload)}\thelm_table_full.bin\n",
        encoding="utf-8",
    )

    custom_toml = tmp_path / "bundles.toml"
    custom_toml.write_text(
        f'[release]\nrepo = "joshspeagle/FlexMESA"\ntag = "data-v1"\n\n'
        f'[[bundles]]\nname = "mini_helm"\narchive = "mini_helm.bin.zst"\n'
        f'kind = "single_zst"\ntarget_relpath = "helm_table_full.bin"\n'
        f'manifest_file = "bundle_hashes/mini_helm.sha256.tsv"\n'
        f'archive_sha256 = "{single_sha}"\n'
        f"archive_size_bytes = {len(single_zst)}\n"
        f'unpacked_sha256 = "{payload_sha}"\n'
        f"unpacked_size_bytes = {len(payload)}\ntest_only = false\n\n"
        f'[[bundles]]\nname = "mini_tests"\narchive = "mini_tests.tar.zst"\n'
        f'kind = "tar_zst"\ntarget_relpath = "neu"\n'
        f'sentinel_relpaths = ["neu/test/test_output"]\n'
        f'manifest_file = "bundle_hashes/mini_tests.sha256.tsv"\n'
        f'archive_sha256 = "{tar_sha}"\n'
        f"archive_size_bytes = {len(tar_zst)}\n"
        f'unpacked_sha256 = ""\nunpacked_size_bytes = {len(data_bytes)}\n'
        f"test_only = true\n",
        encoding="utf-8",
    )
    return data_dest, test_dest, custom_toml, payload, data_bytes


def _run_py_subprocess(
    code: str, env_overrides: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run an isolated Python subprocess with clean FlexMESA env overrides."""
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("RAYON_NUM_THREADS", "1")
    pp = [str(_REPO_ROOT / "src"), str(_REPO_ROOT)]
    if old_pp := env.get("PYTHONPATH", "").strip():
        pp.append(old_pp)
    env["PYTHONPATH"] = os.pathsep.join(pp)
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_build_physics_converters_and_nonfinite_rejection(
    tmp_path: pathlib.Path,
) -> None:
    """Converters build deterministic binaries and reject non-finite inputs."""
    bp = _load_build_physics_module()

    # 1. nets.bin deterministic roundtrip
    nets_dir = tmp_path / "mesa" / "data" / "net_data" / "nets"
    nets_dir.mkdir(parents=True)
    (nets_dir / "README").write_text("ignore me", encoding="utf-8")
    (nets_dir / "basic.net").write_text("add_isos(h1, he4)\n", encoding="utf-8")
    out_nets = tmp_path / "nets.bin"
    sha_a = bp.build_nets_bin(tmp_path / "mesa", out_nets)
    sha_b = bp.build_nets_bin(tmp_path / "mesa", out_nets)
    assert sha_a == sha_b
    assert out_nets.read_bytes().startswith(b"FNET0001")

    # 2. DT ASCII -> FMDTRW01 raw (multi-word MESA headers, 22-col FreeEOS,
    #    trailing rows past num_logQs, and NaN rejection)
    dt_ascii = tmp_path / "mesa-eosDT_02z60x.data"
    hdr = (
        "version X Z num logTs logT min logT max del logT "
        "num logQs logQ min logQ max del logQ\n"
        "51 0.60 0.02 2 4.0 4.1 0.1 2 -5.0 -4.9 0.1\n"
        "\nlogQ = logRho - 2*logT + 12\n-5.0\n\nlogT logPgas\n"
    )
    row_good = "4.0 " + " ".join(["1.25D+00"] * 16) + "\n"
    row_freeeos = (
        "4.0 "
        + " ".join(["1.25E+00"] * 16)
        + " 1.0E+00 -1.78E+01 0.0 0.0 0.0\n"
    )
    row_bad = "4.0 " + " ".join(["NaN"] + ["1.25D+00"] * 15) + "\n"
    dt_ascii.write_text(hdr + row_good * 4, encoding="utf-8")
    dt_raw = tmp_path / "dt.raw"
    bp.convert_dt_ascii_to_raw(dt_ascii, dt_raw)
    assert dt_raw.read_bytes()[:8] == b"FMDTRW01"
    assert len(dt_raw.read_bytes()) == 84 + 4 * 16 * 8

    # 22-column FreeEOS rows + unreferenced rows past num_logQs * num_logTs
    fe_ascii = tmp_path / "mesa-FreeEOS_04z50x.data"
    fe_ascii.write_text(hdr + row_freeeos * 6, encoding="utf-8")
    fe_raw = tmp_path / "fe.raw"
    bp.convert_dt_ascii_to_raw(fe_ascii, fe_raw)
    assert fe_raw.read_bytes() == dt_raw.read_bytes()

    dt_ascii_bad = tmp_path / "mesa-eosDT_bad.data"
    dt_ascii_bad.write_text(hdr + row_bad + row_good * 3, encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        bp.convert_dt_ascii_to_raw(dt_ascii_bad, tmp_path / "bad.raw")

    # 3. CMS ASCII -> FMCMSR01 raw (20-col MESA rows + Inf rejection)
    cms_ascii = tmp_path / "mesa-CMS_070x.data"
    cms_hdr = (
        "version X Z num logTs logT min logT max del logT "
        "num logRhos logRho min logRho max del logRho\n"
        "1 0.70 0.00 2 3.0 3.1 0.1 2 -4.0 -3.9 0.1\n\n"
        "logT logRho logPgas\n"
    )
    cms_row = "3.0 -4.0 " + " ".join(["2.5D+00"] * 16) + " 1.5E+00 -1.6E+00\n"
    cms_row_inf = "3.0 -4.0 " + " ".join(["Inf"] + ["2.5D+00"] * 15) + "\n"
    cms_ascii.write_text(cms_hdr + cms_row * 4, encoding="utf-8")
    cms_raw = tmp_path / "cms.raw"
    bp.convert_cms_ascii_to_raw(cms_ascii, cms_raw)
    assert cms_raw.read_bytes()[:8] == b"FMCMSR01"
    assert len(cms_raw.read_bytes()) == 76 + 4 * 16 * 8

    cms_bad = tmp_path / "mesa-CMS_bad.data"
    cms_bad.write_text(cms_hdr + cms_row_inf + cms_row * 3, encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        bp.convert_cms_ascii_to_raw(cms_bad, tmp_path / "bad_cms.raw")

    # Deterministic tar.zst omits embedded SHA256SUMS.tsv so verify succeeds
    tar_stage = tmp_path / "tar_stage"
    (tar_stage / "eos_4d_raw").mkdir(parents=True)
    shutil_target = tar_stage / "eos_4d_raw" / "mesa-CMS_070x.raw"
    shutil_target.write_bytes(cms_raw.read_bytes())
    tar_out = tmp_path / "cms_pack.tar.zst"
    tsv_out = tmp_path / "cms_pack.sha256.tsv"
    bp.pack_deterministic_tar_zst(
        tar_stage, ["eos_4d_raw/mesa-CMS_070x.raw"], tar_out, tsv_out
    )
    assert "SHA256SUMS.tsv" not in tsv_out.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="0 member files"):
        bp.pack_deterministic_tar_zst(
            tar_stage, [], tmp_path / "empty.tar.zst", tsv_out
        )

    # 4. AESOPUS HDF5 native float32 conversion & non-f32 / NaN rejection
    # Fixture Zsun updated from placeholder 0.0134 -> 0.01337
    # (0x1.b61bb05faebc4p-7) to match upstream MESA 26.4.1
    # kap/preprocessor/AESOPUS/AGSS09.yml:29 (`Zsun: 0.01337`), written into
    # data/kap_data/AESOPUS_AGSS09.h5 by kap/preprocessor/AESOPUS/aesopus.py:
    # 172,192 and read at kap/private/kap_aesopus.f90:122 (with fCO_ref,
    # fC_ref, fN_ref from AGSS09.yml:30-32 and aesopus.py:188-195).
    h5_path = tmp_path / "AESOPUS_AGSS09.h5"
    f32_grid_z = np.array([0.02], dtype=np.float32)
    f32_kap_vals = np.array(
        [[[[-2.5]], [[-1.1234567]]], [[[-0.33333334]], [[0.10000001]]]],
        dtype=np.float32,
    ).reshape(2, 2, 1, 1, 1, 1)
    with h5py.File(h5_path, "w") as f:
        for k, v in (
            ("Zsun", 0.01337),
            ("fCO_ref", -0.384518988806599),
            ("fC_ref", -0.7522932808456995),
            ("fN_ref", -1.2855395523722974),
        ):
            f.create_dataset(k, data=np.float64(v))
        f.create_dataset("logRs", data=np.array([-8.0, -7.5], dtype=np.float32))
        f.create_dataset("logTs", data=np.array([3.2, 3.22], dtype=np.float32))
        f.create_dataset("Zs", data=f32_grid_z)
        g = f.create_group(f"{float(f32_grid_z[0]):.6f}")
        g.create_dataset("Xs", data=np.array([0.7], dtype=np.float32))
        g.create_dataset("fCOs", data=np.array([0.0], dtype=np.float32))
        g.create_dataset("fCs", data=np.array([0.0], dtype=np.float32))
        g.create_dataset("fNs", data=np.array([0.0], dtype=np.float32))
        g.create_dataset("kap", data=f32_kap_vals)

    sec_out = tmp_path / "aesopus_section.bin"
    bp.convert_aesopus_hdf5_to_section(h5_path, sec_out)
    sec_bytes = sec_out.read_bytes()
    assert len(sec_bytes) > 44
    # Last 16 bytes store the 4 IEEE-754 float32 kap entries in (logT, logR)
    decoded_f32 = np.frombuffer(sec_bytes[-16:], dtype="<f4")
    expected_f32 = f32_kap_vals[:, :, 0, 0, 0, 0].reshape(-1)
    assert np.array_equal(decoded_f32, expected_f32)

    # Non-float32 kap dataset must fail loudly
    h5_f64 = tmp_path / "AESOPUS_f64.h5"
    with h5py.File(h5_f64, "w") as f:
        for k, v in (
            ("Zsun", 0.01337),
            ("fCO_ref", -0.35),
            ("fC_ref", -0.5),
            ("fN_ref", -1.1),
        ):
            f.create_dataset(k, data=np.float64(v))
        f.create_dataset("logRs", data=np.array([-8.0, -7.5], dtype=np.float32))
        f.create_dataset("logTs", data=np.array([3.2, 3.22], dtype=np.float32))
        f.create_dataset("Zs", data=f32_grid_z)
        g = f.create_group(f"{float(f32_grid_z[0]):.6f}")
        for axis_k in ("Xs", "fCOs", "fCs", "fNs"):
            g.create_dataset(axis_k, data=np.array([0.0], dtype=np.float32))
        g.create_dataset(
            "kap", data=np.full((2, 2, 1, 1, 1, 1), -2.5, dtype=np.float64)
        )
    with pytest.raises(ValueError, match="expected float32 kap dtype"):
        bp.convert_aesopus_hdf5_to_section(
            h5_f64, tmp_path / "bad_f64_aesopus.bin"
        )

    # NaN in float32 kap dataset must fail loudly
    h5_nan = tmp_path / "AESOPUS_nan.h5"
    with h5py.File(h5_nan, "w") as f:
        for k, v in (
            ("Zsun", 0.01337),
            ("fCO_ref", -0.35),
            ("fC_ref", -0.5),
            ("fN_ref", -1.1),
        ):
            f.create_dataset(k, data=np.float64(v))
        f.create_dataset("logRs", data=np.array([-8.0, -7.5], dtype=np.float32))
        f.create_dataset("logTs", data=np.array([3.2, 3.22], dtype=np.float32))
        f.create_dataset("Zs", data=f32_grid_z)
        g = f.create_group(f"{float(f32_grid_z[0]):.6f}")
        for axis_k in ("Xs", "fCOs", "fCs", "fNs"):
            g.create_dataset(axis_k, data=np.array([0.0], dtype=np.float32))
        nan_kap = f32_kap_vals.copy()
        nan_kap[0, 1, 0, 0, 0, 0] = np.float32("nan")
        g.create_dataset("kap", data=nan_kap)
    with pytest.raises(ValueError, match="non-finite kap values"):
        bp.convert_aesopus_hdf5_to_section(
            h5_nan, tmp_path / "bad_nan_aesopus.bin"
        )

    # 5. HELM ASCII -> FMHELM01 mini-grid roundtrip + NaN/EOF/trailing checks
    helm_lines: list[str] = []
    for sec, ncols in ((0, 9), (1, 4), (2, 4), (3, 4)):
        for row_i in range(6):
            base = float(100 * sec + 10 * row_i)
            helm_lines.append(
                " ".join(f"{base + c + 1:.8E}" for c in range(ncols))
            )
    helm_txt = "\n".join(helm_lines) + "\n"
    helm_src = tmp_path / "helm_mini.dat"
    helm_dst = tmp_path / "helm_mini.bin"
    helm_src.write_text(helm_txt, encoding="utf-8")
    bp.convert_helm_ascii_to_full(helm_src, helm_dst, imax=3, jmax=2)
    helm_raw = helm_dst.read_bytes()
    assert len(helm_raw) == 16 + 6 * 21 * 8
    magic, h_imax, h_jmax = struct.unpack("<8sII", helm_raw[:16])
    assert (magic, h_imax, h_jmax) == (b"FMHELM01", 3, 2)
    helm_arr = np.frombuffer(helm_raw, dtype="<f8", offset=16).reshape(6, 21)
    assert helm_arr[0, 0] == pytest.approx(1.0)
    assert helm_arr[5, 20] == pytest.approx(354.0)

    helm_nan = tmp_path / "helm_nan.dat"
    helm_nan.write_text(
        "\n".join(["NaN " + " ".join(["1.0"] * 8), *helm_lines[1:]]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite"):
        bp.convert_helm_ascii_to_full(
            helm_nan, tmp_path / "bad_helm.bin", imax=3, jmax=2
        )

    helm_eof = tmp_path / "helm_eof.dat"
    helm_eof.write_text("\n".join(helm_lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected EOF"):
        bp.convert_helm_ascii_to_full(
            helm_eof, tmp_path / "eof_helm.bin", imax=3, jmax=2
        )

    helm_trail = tmp_path / "helm_trail.dat"
    helm_trail.write_text(helm_txt + "1.0 2.0 3.0 4.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="trailing"):
        bp.convert_helm_ascii_to_full(
            helm_trail, tmp_path / "trail_helm.bin", imax=3, jmax=2
        )

    # Live upstream AESOPUS dtype check across all Z groups when available
    if mesa_env := os.environ.get("MESA_DIR", "").strip():
        live_h5 = (
            pathlib.Path(mesa_env) / "data" / "kap_data" / "AESOPUS_AGSS09.h5"
        )
        if live_h5.is_file():
            with h5py.File(live_h5, "r") as live_f:
                assert live_f["logRs"].dtype == np.dtype("float32")
                assert live_f["logTs"].dtype == np.dtype("float32")
                assert live_f["Zs"].dtype == np.dtype("float32")
                zs_live = live_f["Zs"][:]
                assert len(zs_live) == 13
                for z_val in zs_live:
                    grp = live_f[f"{float(z_val):.6f}"]
                    for axis_name in ("Xs", "fCOs", "fCs", "fNs"):
                        assert grp[axis_name].dtype == np.dtype("float32")
                    assert grp["kap"].dtype == np.dtype("float32")


def test_shipped_fmkap001_aesopus_section_preserves_native_float32_grids() -> (
    None
):
    """Portable check on shipped `kap_tables_full.bin` AESOPUS float32 payload.

    Upstream MESA `26.4.1` `kap/preprocessor/AESOPUS/aesopus.py` (`HDF5_OPTS =
    {"dtype": "float32", "compression": "gzip"}`) writes `logRs`, `logTs`,
    `Zs`, `Xs`, `fCOs`, `fCs`, `fNs`, and all `14,625` `(logT, logR)` `kap`
    slices in `AESOPUS_AGSS09.h5` as IEEE-754 `float32`. Verifies directly on
    the fetched `FMKAP001` bundle that every grid coordinate is the exact
    `float64` widening of an IEEE-754 `float32` value and that all `14,625`
    `float32` opacity slices are finite.
    """
    raw_buf = tables_jax.ensure_indexed()
    aes = tables_jax.AESOPUS_META
    # Upstream MESA 26.4.1 kap/preprocessor/AESOPUS/AGSS09.yml:29 (Zsun =
    # 0.01337), written to data/kap_data/AESOPUS_AGSS09.h5 by
    # kap/preprocessor/AESOPUS/aesopus.py:172,192.
    assert float(aes["zsun"]) == pytest.approx(0.01337, rel=0.0, abs=1.0e-15)
    log_rs = np.asarray(aes["log_rs_np"], dtype=np.float64)
    log_ts = np.asarray(aes["log_ts_np"], dtype=np.float64)
    zs = np.asarray(aes["zs"], dtype=np.float64)
    assert log_rs.shape == (19,)
    assert log_ts.shape == (42,)
    assert zs.shape == (13,)

    for arr in (log_rs, log_ts, zs):
        widened = arr.astype(np.float32).astype(np.float64)
        assert np.array_equal(arr, widened)

    present_slices = 0
    slice_n = len(log_rs) * len(log_ts)
    slice_bytes = slice_n * 4
    for z_rec in aes["ts"]:
        assert bool(z_rec["z_present"])
        for key in ("xs", "fcos", "fcs", "fns"):
            arr = np.asarray(z_rec[key], dtype=np.float64)
            assert np.array_equal(
                arr, arr.astype(np.float32).astype(np.float64)
            )
        cursor = int(z_rec["offset"])
        for _ in range(int(z_rec["total_slices"])):
            present = int(raw_buf[cursor]) != 0
            cursor += 1
            if present:
                vals = np.frombuffer(
                    raw_buf, dtype="<f4", count=slice_n, offset=cursor
                )
                assert bool(np.all(np.isfinite(vals)))
                cursor += slice_bytes
                present_slices += 1
    assert present_slices == 14625


def test_incomplete_ionization_data_dir_errors_loudly_without_fallback(
    tmp_path: pathlib.Path,
) -> None:
    """Incomplete `ionization_data/` raises `ValueError` naming file and dir."""
    cache_dir = tmp_path / "cache"

    # 1. Empty $FLEXMESA_DATA_DIR/ionization_data -> loud ValueError
    data_empty = tmp_path / "data_empty"
    (data_empty / "ionization_data").mkdir(parents=True)
    for backend_name in ("rust", "jax"):
        call_expr = (
            "from flexmesa.ionization import eval_ionization; "
            "eval_ionization(0.02, 0.70, log10_rho=-5.0, log10_t=4.5, "
            f"backend={backend_name!r})"
        )
        proc = _run_py_subprocess(
            call_expr,
            {
                "FLEXMESA_DATA_DIR": str(data_empty),
                "FLEXMESA_CACHE_DIR": str(cache_dir),
                "MESA_DIR": "",
            },
        )
        assert proc.returncode != 0
        assert "PanicException" not in proc.stderr
        assert "ValueError" in proc.stderr
        assert "ionization_data" in proc.stderr
        assert "ion_00z00x.data" in proc.stderr

    # 2. Incomplete $MESA_DIR/data/ionization_data (has ion_00z00x.data,
    #    missing ion_00z20x.data) -> loud ValueError naming ion_00z20x.data
    data_clean = tmp_path / "data_clean"
    data_clean.mkdir()
    mesa_partial = tmp_path / "mesa_partial"
    mesa_ion = mesa_partial / "data" / "ionization_data"
    mesa_ion.mkdir(parents=True)
    (mesa_ion / "ion_00z00x.data").write_text("placeholder\n", encoding="utf-8")
    proc_mesa = _run_py_subprocess(
        "from flexmesa.ionization import eval_ionization; "
        "eval_ionization(0.02, 0.70, log10_rho=-5.0, log10_t=4.5, "
        "backend='rust')",
        {
            "FLEXMESA_DATA_DIR": str(data_clean),
            "FLEXMESA_CACHE_DIR": str(cache_dir),
            "MESA_DIR": str(mesa_partial),
        },
    )
    assert proc_mesa.returncode != 0
    assert "PanicException" not in proc_mesa.stderr
    assert "ValueError" in proc_mesa.stderr
    assert "ionization_data" in proc_mesa.stderr
    assert "ion_00z20x.data" in proc_mesa.stderr

    # 3. Standard data-v1 layout without ionization_data -> embedded table works
    mesa_test_only = tmp_path / "mesa_test_only"
    (mesa_test_only / "data" / "star_data").mkdir(parents=True)
    proc_ok = _run_py_subprocess(
        "from flexmesa.ionization import eval_ionization; "
        "res = eval_ionization("
        "0.02, 0.70, log10_rho=-5.0, log10_t=4.5, backend='rust'); "
        "assert res.values.shape == (28,)",
        {
            "FLEXMESA_DATA_DIR": str(data_clean),
            "FLEXMESA_CACHE_DIR": str(cache_dir),
            "MESA_DIR": str(mesa_test_only),
        },
    )
    assert proc_ok.returncode == 0, proc_ok.stderr


def test_verify_bundles_rejects_unexpected_files_and_clean_cache_removes_them(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """`verify` catches unexpected/legacy files and `clean-cache` drops them."""
    data_dest, test_dest, custom_toml, _, _ = _write_mini_bundle_fixture(
        tmp_path
    )
    orig_loader = fetch.load_bundles_manifest
    monkeypatch.setattr(
        fetch,
        "load_bundles_manifest",
        lambda manifest_path=None: orig_loader(custom_toml),
    )

    fetch.fetch_bundles(
        dest=data_dest,
        mesa_test_dest=test_dest,
        include_test_inputs=True,
        source_dir=tmp_path / "staging",
        quiet=True,
    )
    assert (
        fetch.verify_bundles(
            dest=data_dest,
            mesa_test_dest=test_dest,
            include_test_inputs=True,
            manifest_path=custom_toml,
            quiet=True,
        )
        is True
    )

    # Plant unexpected legacy cache artifacts inside data_dest
    legacy_dir = data_dest / "eos_4d_slices"
    legacy_dir.mkdir()
    (legacy_dir / "stale.bin").write_bytes(b"stale")
    assert (
        fetch.verify_bundles(
            dest=data_dest,
            mesa_test_dest=test_dest,
            include_test_inputs=True,
            manifest_path=custom_toml,
            quiet=True,
        )
        is False
    )

    # Runtime cache cleanup removes only cache_root/v1 and preserves siblings
    cache_root = tmp_path / "runtime_cache"
    v1_cache = cache_root / "v1" / "eos_4d_slices"
    v1_cache.mkdir(parents=True)
    (v1_cache / "cached.bin").write_bytes(b"ok")
    (cache_root / "keep.txt").write_text("do not delete\n", encoding="utf-8")

    assert (
        cli.main(
            [
                "data",
                "clean-cache",
                "--data-dir",
                str(data_dest),
                "--cache-dir",
                str(cache_root),
                "--purge-legacy-in-data",
                "--quiet",
            ]
        )
        == 0
    )
    assert not (cache_root / "v1").exists()
    assert (cache_root / "keep.txt").is_file()
    assert not legacy_dir.exists()

    # Cleanup still removes v1 when FLEXMESA_DISABLE_RUNTIME_CACHE=1 is active
    v1_cache.mkdir(parents=True)
    (v1_cache / "cached2.bin").write_bytes(b"ok")
    monkeypatch.setenv("FLEXMESA_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("FLEXMESA_DATA_DIR", str(data_dest))
    monkeypatch.setenv("FLEXMESA_DISABLE_RUNTIME_CACHE", "1")
    assert fetch.resolve_cache_dir() is None
    assert cli.main(["data", "clean-cache", "--quiet"]) == 0
    assert not (cache_root / "v1").exists()
    assert (cache_root / "keep.txt").is_file()
    monkeypatch.delenv("FLEXMESA_DISABLE_RUNTIME_CACHE")

    # Overlapping cache_dir and data_dir raises ValueError in resolve & clean
    for bad_cache, bad_data in (
        (data_dest, data_dest),
        (data_dest / "sub", data_dest),
        (cache_root, cache_root / "v1" / "data"),
    ):
        monkeypatch.setenv("FLEXMESA_DATA_DIR", str(bad_data))
        monkeypatch.setenv("FLEXMESA_CACHE_DIR", str(bad_cache))
        with pytest.raises(ValueError, match="overlaps physics data directory"):
            fetch.resolve_cache_dir()
        with pytest.raises(ValueError, match="overlaps physics data directory"):
            fetch.clean_runtime_cache(
                cache_dir=bad_cache, data_dir=bad_data, quiet=True
            )
    monkeypatch.setenv("FLEXMESA_DATA_DIR", str(data_dest))
    monkeypatch.setenv("FLEXMESA_CACHE_DIR", str(cache_root))

    assert (
        fetch.verify_bundles(
            dest=data_dest,
            mesa_test_dest=test_dest,
            include_test_inputs=True,
            manifest_path=custom_toml,
            quiet=True,
        )
        is True
    )


def test_runtime_cache_isolation_disable_and_pyo3_error_handling(
    tmp_path: pathlib.Path,
) -> None:
    """Missing/corrupt bundles raise Python exceptions and never mutate data."""
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    data_dir.mkdir()

    # 1. Missing kap bundle raises ValueError/FileNotFoundError, never panic
    proc_missing = _run_py_subprocess(
        "from flexmesa import _rust; "
        "_rust.kap_get_fixed_metal_rust(0.02, 0.70, -5.0, 5.0)",
        {
            "FLEXMESA_DATA_DIR": str(data_dir),
            "FLEXMESA_CACHE_DIR": str(cache_dir),
            "MESA_DIR": "",
        },
    )
    assert proc_missing.returncode != 0
    assert "PanicException" not in proc_missing.stderr
    assert "ValueError" in proc_missing.stderr

    # 2. Corrupt FMKAP001 bundle raises ValueError, never PanicException
    kap_file = data_dir / "kap_full" / "kap_tables_full.bin"
    kap_file.parent.mkdir(parents=True)
    kap_file.write_bytes(b"FMKAP001_TRUNCATED")
    proc_corrupt_kap = _run_py_subprocess(
        "from flexmesa import _rust; "
        "_rust.kap_get_fixed_metal_rust(0.02, 0.70, -5.0, 5.0)",
        {
            "FLEXMESA_DATA_DIR": str(data_dir),
            "FLEXMESA_CACHE_DIR": str(cache_dir),
            "MESA_DIR": "",
        },
    )
    assert proc_corrupt_kap.returncode != 0
    assert "PanicException" not in proc_corrupt_kap.stderr
    assert "ValueError" in proc_corrupt_kap.stderr

    # 3. Synthetic 4D EOS raw slice loading isolates runtime caches cleanly
    raw_dir = data_dir / "eos_4d_raw"
    raw_dir.mkdir(parents=True)
    _write_synthetic_dt_raw(raw_dir / "mesa-eosDT_00z00x.raw")

    # Disabled mode writes zero files to cache_dir and data_dir/eos_4d_slices
    proc_dis = _run_py_subprocess(
        "from flexmesa.eos._cache_io import load_dt_slice_jax; "
        "res = load_dt_slice_jax(0, 1, 1, 'mesa-eosDT_00z00x'); "
        "assert res[-1].shape == (4, 4, 16, 4)",
        {
            "FLEXMESA_DATA_DIR": str(data_dir),
            "FLEXMESA_CACHE_DIR": str(cache_dir),
            "FLEXMESA_DISABLE_RUNTIME_CACHE": "1",
        },
    )
    assert proc_dis.returncode == 0, proc_dis.stderr
    assert not (cache_dir / "v1").exists()
    assert not (data_dir / "eos_4d_slices").exists()

    # Enabled mode populates cache_dir/v1/eos_4d_slices only
    proc_en = _run_py_subprocess(
        "from flexmesa.eos._cache_io import load_dt_slice_jax; "
        "res = load_dt_slice_jax(0, 1, 1, 'mesa-eosDT_00z00x'); "
        "assert res[-1].shape == (4, 4, 16, 4)",
        {
            "FLEXMESA_DATA_DIR": str(data_dir),
            "FLEXMESA_CACHE_DIR": str(cache_dir),
        },
    )
    assert proc_en.returncode == 0, proc_en.stderr
    assert not (data_dir / "eos_4d_slices").exists()
    cached_bins = list((cache_dir / "v1" / "eos_4d_slices").glob("*.bin"))
    assert len(cached_bins) == 1

    # Flip 1 byte in runtime cache -> PyO3 raises ValueError cleanly
    blob = bytearray(cached_bins[0].read_bytes())
    blob[-1] ^= 0x01
    cached_bins[0].write_bytes(bytes(blob))
    proc_bad = _run_py_subprocess(
        "from flexmesa.eos._cache_io import load_dt_slice_jax; "
        "load_dt_slice_jax(0, 1, 1, 'mesa-eosDT_00z00x')",
        {
            "FLEXMESA_DATA_DIR": str(data_dir),
            "FLEXMESA_CACHE_DIR": str(cache_dir),
        },
    )
    assert proc_bad.returncode != 0
    assert "PanicException" not in proc_bad.stderr
    assert "ValueError" in proc_bad.stderr


def _build_minimal_fmkap001(aes_f32: np.ndarray) -> bytes:
    """Build a valid 258-byte `FMKAP001` payload with one 4x4 AESOPUS slice."""
    buf = bytearray(b"FMKAP001")
    buf.extend(struct.pack("<2I", 0, 0))
    buf.extend(struct.pack("<4d", 0.0134, -0.35, -0.5, -1.1))
    buf.extend(struct.pack("<3I", 4, 4, 1))
    buf.extend(np.array([-8.0, -7.0, -6.0, -5.0], dtype="<f8").tobytes())
    buf.extend(np.array([3.2, 3.3, 3.4, 3.5], dtype="<f8").tobytes())
    buf.extend(np.array([0.02], dtype="<f8").tobytes())
    buf.extend(struct.pack("<4I", 1, 1, 1, 1))
    buf.extend(np.array([0.7, 0.0, 0.0, 0.0], dtype="<f8").tobytes())
    buf.extend(bytes([1, 1]))
    buf.extend(aes_f32.astype("<f4").tobytes())
    buf.extend(struct.pack("<3I", 0, 0, 0))
    return bytes(buf)


def test_jax_kap_loader_rejects_nan_opacities_and_corrupt_fmkap001(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """JAX kap loader rejects NaN opacities and corrupt/truncated FMKAP001."""
    from flexmesa.kap import tables_co_lowt_jax, tables_jax

    lrs = np.array([-8.0, -7.0, -6.0, -5.0], dtype=np.float64)
    lts = np.array([3.2, 3.3, 3.4, 3.5], dtype=np.float64)
    raw_nan = np.full((1, 4, 4), -2.0, dtype=np.float64)
    raw_nan[0, 2, 1] = np.nan
    with pytest.raises(ValueError, match="Non-finite raw log10\\(kappa\\)"):
        tables_jax.fit_bicubic_batch(lrs, lts, raw_nan)

    lrs_nan = lrs.copy()
    lrs_nan[0] = np.inf
    with pytest.raises(ValueError, match="Non-finite grid coordinate"):
        tables_jax.fit_bicubic_batch(
            lrs_nan, lts, np.full((1, 4, 4), -2.0, dtype=np.float64)
        )

    data_dir = tmp_path / "kap_jax_data"
    kap_path = data_dir / "kap_full" / "kap_tables_full.bin"
    kap_path.parent.mkdir(parents=True)
    monkeypatch.setenv("FLEXMESA_DATA_DIR", str(data_dir))
    tables_jax.reset_index_cache()
    tables_co_lowt_jax._AESOPUS_SPLINE_CACHE.clear()

    valid = _build_minimal_fmkap001(np.full(16, -2.0, dtype=np.float32))
    bad_scalar = bytearray(valid)
    bad_scalar[16:24] = struct.pack("<d", float("nan"))

    try:
        for bad_bytes, pat in (
            (b"", "Truncated"),
            (valid[:-5], "Unexpected EOF"),
            (valid + b"\x00", "trailing"),
            (bytes(bad_scalar), "Non-finite f64"),
        ):
            kap_path.write_bytes(bad_bytes)
            with pytest.raises(ValueError, match=pat):
                tables_jax.ensure_indexed()
            assert tables_jax._RAW_FILE is None
            assert tables_jax._RAW_MMAP is None
            assert tables_jax._RAW_BYTES is None
            assert tables_jax.FIXED_INDEX == {}

        # Float32 NaN inside AESOPUS slice must fail fit_bicubic_batch loudly
        aes_nan = np.full(16, -2.0, dtype=np.float32)
        aes_nan[5] = np.float32("nan")
        kap_path.write_bytes(_build_minimal_fmkap001(aes_nan))
        _ = tables_jax.ensure_indexed()
        with pytest.raises(ValueError, match="Non-finite raw log10\\(kappa\\)"):
            tables_co_lowt_jax.get_aesopus_z_coeffs_jax(0)
        tables_jax.reset_index_cache()
        tables_co_lowt_jax._AESOPUS_SPLINE_CACHE.clear()

        # Valid minimal FMKAP001 indexes and spline-fits cleanly
        kap_path.write_bytes(valid)
        coeffs = tables_co_lowt_jax.get_aesopus_z_coeffs_jax(0)
        assert coeffs.shape == (1, 4, 4, 4)
        assert bool(np.all(np.isfinite(np.asarray(coeffs))))
    finally:
        tables_jax.reset_index_cache()
        tables_co_lowt_jax._AESOPUS_SPLINE_CACHE.clear()
