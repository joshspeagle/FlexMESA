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

import hypothesis
import numpy as np
import pytest
import safetensors.numpy
from hypothesis import strategies as st

from flexmesa.data import converters, loader, manifest

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
        loader.save_safetensors_bundle(
            tmp_path / "f32.safetensors",
            {"bad": np.array([1.0, 2.0], dtype=np.float32)},
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
