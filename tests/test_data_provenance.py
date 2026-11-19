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
from types import ModuleType

import h5py
import numpy as np
import pytest

from flexmesa import cli
from flexmesa.data import fetch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_build_physics_module() -> ModuleType:
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

    # 2. DT ASCII -> FMDTRW01 raw + NaN rejection
    dt_ascii = tmp_path / "mesa-EOS5_02z60x.data"
    hdr = (
        "version X Z num_logTs logT_min logT_max del_logT "
        "num_logQs logQ_min logQ_max del_logQ\n"
        "51 0.60 0.02 2 4.0 4.1 0.1 2 -5.0 -4.9 0.1\n"
        "\n\n"
    )
    row_good = "4.0 " + " ".join(["1.25D+00"] * 16) + "\n"
    row_bad = "4.0 " + " ".join(["NaN"] + ["1.25D+00"] * 15) + "\n"
    dt_ascii.write_text(hdr + row_good * 4, encoding="utf-8")
    dt_raw = tmp_path / "dt.raw"
    bp.convert_dt_ascii_to_raw(dt_ascii, dt_raw)
    assert dt_raw.read_bytes()[:8] == b"FMDTRW01"
    assert len(dt_raw.read_bytes()) == 84 + 4 * 16 * 8

    dt_ascii_bad = tmp_path / "mesa-EOS5_bad.data"
    dt_ascii_bad.write_text(hdr + row_bad + row_good * 3, encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        bp.convert_dt_ascii_to_raw(dt_ascii_bad, tmp_path / "bad.raw")

    # 3. CMS ASCII -> FMCMSR01 raw + Inf rejection
    cms_ascii = tmp_path / "mesa-CMS_070x.data"
    cms_hdr = (
        "version X Z num_logTs logT_min logT_max del_logT "
        "num_logRhos logRho_min logRho_max del_logRho\n"
        "1 0.70 0.00 2 3.0 3.1 0.1 2 -4.0 -3.9 0.1\n\n\n"
    )
    cms_row = "-4.0 0.0 " + " ".join(["2.5D+00"] * 16) + "\n"
    cms_row_inf = "-4.0 0.0 " + " ".join(["Inf"] + ["2.5D+00"] * 15) + "\n"
    cms_ascii.write_text(cms_hdr + cms_row * 4, encoding="utf-8")
    cms_raw = tmp_path / "cms.raw"
    bp.convert_cms_ascii_to_raw(cms_ascii, cms_raw)
    assert cms_raw.read_bytes()[:8] == b"FMCMSR01"
    assert len(cms_raw.read_bytes()) == 76 + 4 * 16 * 8

    cms_bad = tmp_path / "mesa-CMS_bad.data"
    cms_bad.write_text(cms_hdr + cms_row_inf + cms_row * 3, encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        bp.convert_cms_ascii_to_raw(cms_bad, tmp_path / "bad_cms.raw")

    # 4. AESOPUS HDF5 native float32 conversion & non-f32 / NaN rejection
    h5_path = tmp_path / "AESOPUS_AGSS09.h5"
    with h5py.File(h5_path, "w") as f:
        for k, v in (
            ("Zsun", 0.0134),
            ("fCO_ref", -0.35),
            ("fC_ref", -0.5),
            ("fN_ref", -1.1),
        ):
            f.create_dataset(k, data=np.float64(v))
        f.create_dataset("logRs", data=np.array([-8.0, -7.0], dtype=np.float64))
        f.create_dataset("logTs", data=np.array([3.2, 3.3], dtype=np.float64))
        f.create_dataset("Zs", data=np.array([0.02], dtype=np.float64))
        g = f.create_group("0.020000")
        g.create_dataset("Xs", data=np.array([0.7], dtype=np.float64))
        g.create_dataset("fCOs", data=np.array([0.0], dtype=np.float64))
        g.create_dataset("fCs", data=np.array([0.0], dtype=np.float64))
        g.create_dataset("fNs", data=np.array([0.0], dtype=np.float64))
        kap_arr = np.full((2, 2, 1, 1, 1, 1), -2.5, dtype=np.float32)
        g.create_dataset("kap", data=kap_arr)

    sec_out = tmp_path / "aesopus_section.bin"
    bp.convert_aesopus_hdf5_to_section(h5_path, sec_out)
    assert sec_out.stat().st_size > 44

    # Live upstream AESOPUS dtype check when $MESA_DIR is available
    if mesa_env := os.environ.get("MESA_DIR", "").strip():
        live_h5 = (
            pathlib.Path(mesa_env) / "data" / "kap_data" / "AESOPUS_AGSS09.h5"
        )
        if live_h5.is_file():
            with h5py.File(live_h5, "r") as live_f:
                z0 = float(live_f["Zs"][0])
                assert live_f[f"{z0:.6f}"]["kap"].dtype == np.dtype("float32")


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

    # Also test runtime cache directory cleanup + legacy purge
    cache_root = tmp_path / "runtime_cache"
    v1_cache = cache_root / "v1" / "eos_4d_slices"
    v1_cache.mkdir(parents=True)
    (v1_cache / "cached.bin").write_bytes(b"ok")

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
    assert not legacy_dir.exists()
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
