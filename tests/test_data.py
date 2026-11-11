# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit and property-based tests for `flexmesa.data` (`Wave 0C`).

Verifies:
1. `manifest.toml` parsing, `SHA-256` digest computation, and integrity
   rejection on corrupted files or invalid digests.
2. Memory-mapped `.safetensors` saving and loading with `float64`/`int64`
   enforcement and non-finite (`NaN`/`Inf`) rejection.
3. Bit-exact conversion of MESA `lodders03.data`, `isotopes.data`, and
   `helm_table.dat` fixtures into row-major `.safetensors` bundles.
4. Property-based round-trip invariants via `hypothesis`.
"""

from __future__ import annotations

import pathlib
from typing import Any

import hypothesis
import numpy as np
import pytest
import safetensors.numpy
from hypothesis import strategies as st

from flexmesa import cli
from flexmesa.data import converters, fetch, loader, manifest

_SAMPLE_LODDERS03 = """\
! Table 6 of Lodders (2003), ApJ, 591: 1220
! Z El     A     %            N(Si= 1e6)
!=======================================
  1 H      1     99.99806     2.431e10
  1 H      2      0.00194     4.716e5
  2 He     4     99.983403    2.343e9
 26 Fe    56     91.754       768,900
"""

_ONES_ROW = "  ".join(["1.00000E+00"] * 8)
_AL26_ROW2 = (
    "1.00000E+00 1.00000E+00 1.00000E+00 1.01000E+00 "
    "1.02000E+00 1.04000E+00 1.06000E+00 1.08000E+00"
)
_AL26_ROW3 = (
    "1.11000E+00 1.14000E+00 1.17000E+00 1.21000E+00 "
    "1.25000E+00 1.30000E+00 1.36000E+00 1.43000E+00"
)
_SAMPLE_ISOTOPES = "\n".join(
    [
        "",
        "neut        1.0086649    0    1   0.5   8.071317100",
        _ONES_ROW,
        _ONES_ROW,
        _ONES_ROW,
        "al26-1     25.9868919   13   13   5.0 -12.210151000",
        _ONES_ROW,
        _AL26_ROW2,
        _AL26_ROW3,
        "",
    ]
)


def _build_synthetic_helm_text(imax: int = 3, jmax: int = 2) -> str:
    """Build a synthetic `helm_table.dat` fixture of size `(imax, jmax)`.

    Populates all four sections (`f`, `dpdf`, `ef`, `xf`) in MESA's outer `j`
    (temperature), inner `i` (density) row ordering so transpose parity can be
    tested against known cell formulas.
    """
    lines: list[str] = []
    for sec, n_cols in ((0, 9), (1, 4), (2, 4), (3, 4)):
        for j in range(jmax):
            for i in range(imax):
                base = float(1000 * sec + 100 * j + 10 * i)
                vals = [f"{base + k + 1:.16E}" for k in range(n_cols)]
                lines.append("  ".join(vals))
    return "\n".join(lines) + "\n"


def test_default_manifest_loads_and_validates_entries() -> None:
    """Verify `load_default_manifest` parses `manifest.toml` cleanly.

    Checks that `lodders03`, `isotopes`, and `helm_table` entries exist with
    valid 64-character `SHA-256` hex digests and that duplicate table names are
    rejected.
    """
    tables = manifest.load_default_manifest()
    assert {"lodders03", "isotopes", "helm_table"} <= set(tables.keys())
    lod = tables["lodders03"]
    assert lod.filename == "lodders03.safetensors"
    assert manifest.is_valid_sha256_hex(lod.sha256)
    assert "isotopic_percent" in lod.tensors

    sha_a = "a" * 64
    sha_b = "b" * 64
    with pytest.raises(ValueError, match="Duplicate table name"):
        manifest.parse_manifest_text(
            f'[[tables]]\nname="a"\nfilename="a.st"\n'
            f'sha256="{sha_a}"\nsize_bytes=1\ntensors=["x"]\n'
            f'[[tables]]\nname="a"\nfilename="b.st"\n'
            f'sha256="{sha_b}"\nsize_bytes=1\ntensors=["x"]\n'
        )


def test_safetensors_bundle_roundtrips_and_enforces_integrity(
    tmp_path: pathlib.Path,
) -> None:
    """Verify `.safetensors` saving, SHA-256 checks, and NaN/dtype rejection.

    Confirms memory-mapped loading via `load_safetensors_bundle` and verifies
    that corrupted `SHA-256` digests, missing required tensors, `NaN` values,
    and `float32` tensors raise appropriate exceptions.
    """
    st_path = tmp_path / "sample.safetensors"
    f_arr = np.array([[1.25, -2.5], [3.75, 4.0]], dtype=np.float64)
    i_arr = np.array([10, 20, 30], dtype=np.int64)
    digest = loader.save_safetensors_bundle(
        st_path,
        {"grid": f_arr, "indices": i_arr},
        metadata={"source": "unit_test"},
    )
    loaded = loader.load_safetensors_bundle(
        st_path,
        expected_sha256=digest,
        required_tensors=("grid", "indices"),
    )
    assert np.array_equal(loaded.tensors["grid"], f_arr)
    assert np.array_equal(loaded.tensors["indices"], i_arr)
    assert loaded.metadata["source"] == "unit_test"

    with pytest.raises(ValueError, match="SHA-256 checksum mismatch"):
        loader.load_safetensors_bundle(st_path, expected_sha256="0" * 64)
    with pytest.raises(KeyError, match="Missing required tensor"):
        loader.load_safetensors_bundle(
            st_path, required_tensors=("grid", "missing_tensor")
        )
    with pytest.raises(ValueError, match="non-finite"):
        loader.save_safetensors_bundle(
            tmp_path / "nan.safetensors",
            {"bad": np.array([1.0, np.nan], dtype=np.float64)},
        )
    with pytest.raises(TypeError, match="Unsupported tensor dtype"):
        bad_f32: Any = {"bad": np.array([1.0, 2.0], dtype=np.float32)}
        loader.save_safetensors_bundle(
            tmp_path / "f32.safetensors",
            bad_f32,
        )


def test_converters_parse_lodders03_isotopes_and_helm_table(
    tmp_path: pathlib.Path,
) -> None:
    """Verify conversion of `lodders03`, `isotopes`, and `helm_table`.

    Checks thousands-separator handling (`768,900`), isomeric state parsing
    (`al26-1`), and `(imax, jmax)` row-major indexing for `helm_table`.
    """
    lod = converters.parse_lodders03_text(_SAMPLE_LODDERS03)
    assert np.array_equal(
        lod.tensors["atomic_number_z"], np.array([1, 1, 2, 26], dtype=np.int64)
    )
    assert lod.tensors["number_abundance_si"][3] == pytest.approx(768900.0)
    assert lod.metadata["isotopes"] == "h1,h2,he4,fe56"
    lod_in = tmp_path / "lodders03.data"
    lod_out = tmp_path / "lodders03.safetensors"
    lod_in.write_text(_SAMPLE_LODDERS03, encoding="utf-8")
    assert (
        converters.convert_lodders03_file(lod_in, lod_out)
        == "877856c26bf3eec3a187b6977e37b9c6bef811116d6f3655a9a35d6261c126d3"
    )

    iso = converters.parse_isotopes_text(_SAMPLE_ISOTOPES)
    assert np.array_equal(
        iso.tensors["isomeric_state"], np.array([0, 1], dtype=np.int64)
    )
    assert iso.tensors["partition_function"].shape == (2, 24)
    assert iso.tensors["partition_function"][1, 23] == pytest.approx(1.43)

    helm_text = _build_synthetic_helm_text(imax=3, jmax=2)
    helm = converters.parse_helm_table_text(helm_text, imax=3, jmax=2)
    assert helm.tensors["f"].shape == (3, 2)
    assert helm.tensors["f_jet"].shape == (3, 2, 9)
    assert helm.tensors["f"][2, 1] == pytest.approx(121.0)
    assert helm.tensors["fddtt"][2, 1] == pytest.approx(129.0)
    assert helm.tensors["xfdt"][2, 1] == pytest.approx(3124.0)

    out_path = tmp_path / "helm.safetensors"
    sha = loader.save_safetensors_bundle(out_path, helm.tensors, helm.metadata)
    reloaded = loader.load_safetensors_bundle(out_path, expected_sha256=sha)
    assert np.array_equal(reloaded.tensors["f_jet"], helm.tensors["f_jet"])


@hypothesis.given(
    rows=st.integers(min_value=1, max_value=8),
    cols=st.integers(min_value=1, max_value=8),
    seed=st.integers(min_value=0, max_value=10_000),
)
@hypothesis.settings(max_examples=40)
def test_safetensors_float64_and_int64_roundtrip_bit_exact(
    rows: int,
    cols: int,
    seed: int,
) -> None:
    """Property test: finite `float64` and `int64` arrays round-trip.

    Generates random 2D `float64` and 1D `int64` arrays using `default_rng` and
    verifies bit-exact serialization via `loader.encode_safetensors_bytes` and
    deserialization via `safetensors.numpy.load`.
    """
    rng = np.random.default_rng(seed)
    f_arr = rng.standard_normal((rows, cols), dtype=np.float64)
    i_arr = rng.integers(-1_000_000, 1_000_000, size=(rows,), dtype=np.int64)
    raw_bytes = loader.encode_safetensors_bytes(
        {"f": f_arr, "i": i_arr}, metadata={"seed": str(seed)}
    )
    decoded = safetensors.numpy.load(raw_bytes)
    assert np.array_equal(decoded["f"], f_arr)
    assert np.array_equal(decoded["i"], i_arr)


def _compress_zst_bytes(raw: bytes) -> bytes:
    """Compress `raw` into Zstandard format across Python 3.12+."""
    try:
        from compression import zstd as std_zstd

        return bytes(std_zstd.compress(raw))
    except ImportError:
        import zstandard

        return bytes(zstandard.ZstdCompressor().compress(raw))


def test_bundles_manifest_and_directory_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Verify `bundles.toml` integrity and directory resolution precedence."""
    repo, tag, bundles = fetch.load_bundles_manifest()
    assert repo == "joshspeagle/FlexMESA"
    assert tag == "data-v1"
    names = {b.name for b in bundles}
    assert names == {
        "helm_full",
        "kap_full",
        "eos_opal_scvh",
        "eos_cms",
        "eos_free_eos",
        "mesa_test_inputs",
    }
    for b in bundles:
        assert manifest.is_valid_sha256_hex(b.archive_sha256)
        assert b.archive_size_bytes > 0
        assert b.unpacked_size_bytes > 0
        if b.kind == "single_zst":
            assert manifest.is_valid_sha256_hex(b.unpacked_sha256)

    exp_dir = tmp_path / "explicit_data"
    env_dir = tmp_path / "env_data"
    xdg_dir = tmp_path / "xdg"
    monkeypatch.setenv("FLEXMESA_DATA_DIR", str(env_dir))
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_dir))
    assert fetch.resolve_data_dir(exp_dir) == exp_dir.resolve()
    assert fetch.resolve_data_dir() == env_dir.resolve()
    monkeypatch.delenv("FLEXMESA_DATA_DIR")
    assert fetch.resolve_data_dir() == (xdg_dir / "flexmesa" / "data").resolve()

    mesa_env = tmp_path / "mesa_env"
    monkeypatch.setenv("MESA_DIR", str(mesa_env))
    assert fetch.resolve_mesa_test_inputs_dir() == mesa_env.resolve()
    monkeypatch.delenv("MESA_DIR")
    assert (
        fetch.resolve_mesa_test_inputs_dir(data_dir=exp_dir)
        == (tmp_path / "mesa_test_inputs").resolve()
    )


def test_fetch_and_cli_verify_unpack_and_reject_corrupted_archives(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Verify `fetch_bundles`, `cli.main`, and SHA-256 corruption rejection."""
    import hashlib
    import io
    import tarfile

    src_dir = tmp_path / "staging"
    src_dir.mkdir()
    data_dest = tmp_path / "dest_data"
    test_dest = tmp_path / "dest_tests"

    # 1. Single .zst payload
    payload = b"FMHELM01_TEST_PAYLOAD"
    payload_sha = hashlib.sha256(payload).hexdigest()
    single_zst = _compress_zst_bytes(payload)
    single_sha = hashlib.sha256(single_zst).hexdigest()
    (src_dir / "mini_helm.bin.zst").write_bytes(single_zst)

    # 2. Tar .zst payload
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w") as tf:
        info = tarfile.TarInfo("neu/test/test_output")
        data_bytes = b"golden_row 1.0\n"
        info.size = len(data_bytes)
        tf.addfile(info, io.BytesIO(data_bytes))
    tar_raw = tar_buf.getvalue()
    tar_zst = _compress_zst_bytes(tar_raw)
    tar_sha = hashlib.sha256(tar_zst).hexdigest()
    (src_dir / "mini_tests.tar.zst").write_bytes(tar_zst)

    custom_toml = tmp_path / "bundles.toml"
    custom_toml.write_text(
        f'[release]\nrepo = "joshspeagle/FlexMESA"\ntag = "data-v1"\n\n'
        f'[[bundles]]\nname = "mini_helm"\narchive = "mini_helm.bin.zst"\n'
        f'kind = "single_zst"\ntarget_relpath = "helm_table_full.bin"\n'
        f'archive_sha256 = "{single_sha}"\n'
        f"archive_size_bytes = {len(single_zst)}\n"
        f'unpacked_sha256 = "{payload_sha}"\n'
        f"unpacked_size_bytes = {len(payload)}\ntest_only = false\n\n"
        f'[[bundles]]\nname = "mini_tests"\narchive = "mini_tests.tar.zst"\n'
        f'kind = "tar_zst"\ntarget_relpath = "neu"\n'
        f'sentinel_relpaths = ["neu/test/test_output"]\n'
        f'archive_sha256 = "{tar_sha}"\n'
        f"archive_size_bytes = {len(tar_zst)}\n"
        f'unpacked_sha256 = ""\nunpacked_size_bytes = {len(data_bytes)}\n'
        f"test_only = true\n",
        encoding="utf-8",
    )
    orig_loader = fetch.load_bundles_manifest
    monkeypatch.setattr(
        fetch,
        "load_bundles_manifest",
        lambda manifest_path=None: orig_loader(custom_toml),
    )

    rc = cli.main(
        [
            "data",
            "fetch",
            "--data-dir",
            str(data_dest),
            "--mesa-test-dest",
            str(test_dest),
            "--include-test-inputs",
            "--source-dir",
            str(src_dir),
            "--quiet",
        ]
    )
    assert rc == 0
    assert (data_dest / "helm_table_full.bin").read_bytes() == payload
    assert (test_dest / "neu" / "test" / "test_output").is_file()
    assert (
        cli.main(
            [
                "data",
                "status",
                "--dest",
                str(data_dest),
                "--mesa-test-dest",
                str(test_dest),
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "data",
                "verify",
                "--data-dir",
                str(data_dest),
                "--mesa-test-dest",
                str(test_dest),
                "--include-test-inputs",
                "--quiet",
            ]
        )
        == 0
    )
    assert cli.main(["data", "path", "--data-dir", str(data_dest)]) == 0
    assert (
        cli.main(
            [
                "data",
                "path",
                "--data-dir",
                str(data_dest),
                "--mesa-test-dest",
                str(test_dest),
                "--test-inputs",
            ]
        )
        == 0
    )
    assert cli.main(["--version"]) == 0

    # Corrupt archive SHA-256 must raise ValueError
    bad_src = tmp_path / "bad_staging"
    bad_src.mkdir()
    (bad_src / "mini_helm.bin.zst").write_bytes(single_zst + b"x")
    with pytest.raises(ValueError, match="Archive SHA-256 mismatch"):
        fetch.fetch_bundles(
            dest=tmp_path / "bad_dest",
            source_dir=bad_src,
            quiet=True,
        )
