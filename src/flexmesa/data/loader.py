# SPDX-License-Identifier: LGPL-3.0-or-later
"""Memory-mapped `.safetensors` bundle reader and writer for physics tables.

Enforces row-major (`C`-contiguous) `float64` (`F64`) and `int64` (`I64`)
tensor layouts, finite IEEE-754 floating-point checks, and optional `SHA-256`
digest verification prior to memory-mapping `.safetensors` files.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import struct
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import safetensors

from flexmesa.data import manifest

type TableTensor = npt.NDArray[np.float64] | npt.NDArray[np.int64]


@dataclasses.dataclass(frozen=True, slots=True)
class LoadedTableBundle:
    """Loaded `.safetensors` physics table bundle.

    Attributes:
        tensors: Dictionary mapping tensor name to its row-major `float64` or
            `int64` NumPy array.
        metadata: Dictionary of string key-value metadata recorded in the
            `.safetensors` header (`__metadata__`).
    """

    tensors: dict[str, TableTensor]
    metadata: dict[str, str]


def _validate_and_normalize_tensor(
    name: str,
    arr: npt.NDArray[np.generic],
    *,
    require_finite: bool,
) -> TableTensor:
    """Validate `arr` has dtype `float64` or `int64` and is C-contiguous."""
    if not name or not name.strip() or name == "__metadata__":
        raise ValueError(f"Invalid tensor name: {name!r}")
    if arr.dtype == np.dtype(np.float64):
        contig_f64: npt.NDArray[np.float64] = np.ascontiguousarray(
            arr, dtype=np.float64
        )
        if require_finite and not bool(np.all(np.isfinite(contig_f64))):
            raise ValueError(
                f"Tensor {name!r} contains non-finite (NaN or Inf) float64 "
                f"values"
            )
        return contig_f64
    if arr.dtype == np.dtype(np.int64):
        contig_i64: npt.NDArray[np.int64] = np.ascontiguousarray(
            arr, dtype=np.int64
        )
        return contig_i64
    raise TypeError(
        f"Unsupported tensor dtype {arr.dtype} for {name!r}; only float64 "
        f"and int64 are permitted in FlexMESA physics tables"
    )


def encode_safetensors_bytes(
    tensors: Mapping[str, TableTensor],
    metadata: Mapping[str, str] | None = None,
    *,
    require_finite: bool = True,
) -> bytes:
    """Serialize `tensors` and `metadata` into deterministic `.safetensors`.

    Orders both `"__metadata__"` keys and tensor keys lexicographically so the
    resulting byte buffer and `SHA-256` digest are 100% deterministic across
    runs and byte-for-byte identical to Safe Rust `encode_safetensors_bytes`.

    Args:
        tensors: Mapping from tensor name to `float64` or `int64` NumPy array.
        metadata: Optional string-to-string header metadata dictionary.
        require_finite: If `True` (default), reject `NaN` or `Inf` values in any
            `float64` tensor.

    Returns:
        Raw `.safetensors` binary byte buffer.

    Raises:
        ValueError: If `tensors` is empty, a tensor name is invalid, or any
            `float64` tensor contains non-finite values.
        TypeError: If any tensor has a dtype other than `float64` or `int64`.
    """
    if not tensors:
        raise ValueError("Cannot encode an empty .safetensors bundle")
    normalized: dict[str, TableTensor] = {
        name: _validate_and_normalize_tensor(
            name, tensors[name], require_finite=require_finite
        )
        for name in sorted(tensors)
    }
    header_obj: dict[str, object] = {}
    if metadata:
        header_obj["__metadata__"] = {
            str(k): str(metadata[k]) for k in sorted(metadata.keys())
        }
    offset = 0
    payload_chunks: list[bytes] = []
    for name, arr in normalized.items():
        dtype_str = "F64" if arr.dtype == np.dtype(np.float64) else "I64"
        le_dtype = "<f8" if dtype_str == "F64" else "<i8"
        raw_tensor = arr.astype(le_dtype, copy=False).tobytes(order="C")
        end = offset + len(raw_tensor)
        header_obj[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [offset, end],
        }
        payload_chunks.append(raw_tensor)
        offset = end

    header_bytes = json.dumps(
        header_obj, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    pad_len = (-len(header_bytes)) % 8
    if pad_len:
        header_bytes += b" " * pad_len
    prefix = struct.pack("<Q", len(header_bytes))
    return b"".join([prefix, header_bytes, *payload_chunks])


def save_safetensors_bundle(
    path: str | pathlib.Path,
    tensors: Mapping[str, TableTensor],
    metadata: Mapping[str, str] | None = None,
    *,
    require_finite: bool = True,
) -> str:
    """Write a `.safetensors` physics table bundle and return its SHA-256.

    Args:
        path: Destination `.safetensors` file path.
        tensors: Mapping from tensor name to `float64` or `int64` NumPy array.
        metadata: Optional string-to-string header metadata dictionary.
        require_finite: If `True` (default), reject `NaN` or `Inf` values in any
            `float64` tensor.

    Returns:
        Lowercase 64-character hexadecimal `SHA-256` digest of the written file.

    Raises:
        ValueError: If `tensors` is empty or any `float64` tensor contains
            non-finite values when `require_finite=True`.
        TypeError: If any tensor has a dtype other than `float64` or `int64`.
    """
    raw_bytes = encode_safetensors_bytes(
        tensors, metadata, require_finite=require_finite
    )
    out_path = pathlib.Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}")
    tmp_path.write_bytes(raw_bytes)
    os.replace(tmp_path, out_path)
    return manifest.compute_sha256_bytes(raw_bytes)


def load_safetensors_bundle(
    path: str | pathlib.Path,
    *,
    expected_sha256: str | None = None,
    required_tensors: Sequence[str] | None = None,
    require_finite: bool = True,
) -> LoadedTableBundle:
    """Memory-map and validate a `.safetensors` physics table bundle from disk.

    Args:
        path: Path to the `.safetensors` file on disk.
        expected_sha256: Optional 64-character `SHA-256` hex digest to verify
            prior to opening the file.
        required_tensors: Optional sequence of tensor names that must exist in
            the loaded bundle.
        require_finite: If `True` (default), verify that all `float64` tensors
            contain only finite IEEE-754 values.

    Returns:
        `LoadedTableBundle` containing the memory-mapped `float64`/`int64`
        arrays and string metadata dictionary.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If the `SHA-256` checksum mismatches, the bundle is empty,
            or non-finite values are encountered when `require_finite=True`.
        KeyError: If any tensor listed in `required_tensors` is missing.
        TypeError: If any stored tensor has a dtype other than `float64` or
            `int64`.
    """
    file_path = pathlib.Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"SafeTensors file not found: {file_path}")
    if expected_sha256 is not None:
        manifest.verify_file_sha256(file_path, expected_sha256)

    tensors: dict[str, TableTensor] = {}
    with safetensors.safe_open(str(file_path), framework="np") as handle:
        available_keys = [str(k) for k in handle.keys()]
        if not available_keys:
            raise ValueError(f"SafeTensors file {file_path.name} has 0 tensors")
        if required_tensors is not None:
            present_keys = set(available_keys)
            missing = [t for t in required_tensors if t not in present_keys]
            if missing:
                raise KeyError(
                    f"Missing required tensor(s) {missing} in {file_path.name}"
                )
        raw_meta = handle.metadata()
        metadata: dict[str, str] = (
            {str(k): str(v) for k, v in raw_meta.items()}
            if isinstance(raw_meta, dict)
            else {}
        )
        for key in available_keys:
            tensors[key] = _validate_and_normalize_tensor(
                key, handle.get_tensor(key), require_finite=require_finite
            )
    return LoadedTableBundle(tensors=tensors, metadata=metadata)
