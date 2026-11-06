# SPDX-License-Identifier: LGPL-3.0-or-later
"""SHA-256 manifest parser and file integrity verifier for physics tables.

Parses ``manifest.toml`` declarations (``[[tables]]``) into typed
``TableArtifact`` records and verifies ``.safetensors`` files against their
recorded ``SHA-256`` digests via ``flexmesa._rust`` prior to memory-mapping.
"""

from __future__ import annotations

import dataclasses
import pathlib
import tomllib

from flexmesa import _rust

_ALLOWED_TABLE_KEYS = frozenset(
    {"name", "filename", "sha256", "size_bytes", "tensors", "description"}
)


@dataclasses.dataclass(frozen=True, slots=True)
class TableArtifact:
    """Metadata and integrity record for a ``.safetensors`` physics table.

    Attributes:
        name: Logical table identifier (e.g., ``'lodders03'``, ``'isotopes'``,
            ``'helm_table'``).
        filename: Canonical ``.safetensors`` filename (e.g.,
            ``'lodders03.safetensors'``).
        sha256: Lowercase 64-character hexadecimal ``SHA-256`` digest of the
            uncompressed ``.safetensors`` file.
        size_bytes: Uncompressed ``.safetensors`` file size in bytes (``>= 0``).
        tensors: Ordered tuple of tensor names required in the bundle.
        description: Human-readable summary of the physical data and source.
    """

    name: str
    filename: str
    sha256: str
    size_bytes: int
    tensors: tuple[str, ...]
    description: str


def is_valid_sha256_hex(digest: str) -> bool:
    """Return ``True`` if ``digest`` is a 64-character hexadecimal string.

    Delegates to ``flexmesa_data::sha256::is_valid_sha256_hex`` via
    ``flexmesa._rust``.

    Args:
        digest: Candidate ``SHA-256`` hexadecimal digest string.

    Returns:
        ``True`` if ``digest`` has length 64 and contains only ASCII hex digits.

    Examples:
        >>> is_valid_sha256_hex("0" * 64)
        True
        >>> is_valid_sha256_hex("not-a-digest")
        False
    """
    return _rust.is_valid_sha256_hex(digest.strip())


def compute_sha256_bytes(data: bytes) -> str:
    """Compute the lowercase 64-character ``SHA-256`` hex digest of ``data``.

    Delegates to ``flexmesa_data::sha256::compute_sha256_bytes`` via
    ``flexmesa._rust``.

    Args:
        data: Raw byte buffer to hash.

    Returns:
        Lowercase 64-character hexadecimal ``SHA-256`` digest.

    Examples:
        >>> compute_sha256_bytes(b"")
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    """
    return _rust.compute_sha256_bytes(data)


def compute_sha256_file(path: str | pathlib.Path) -> str:
    """Compute the lowercase 64-character ``SHA-256`` hex digest of a file.

    Delegates to ``flexmesa_data::sha256::compute_sha256_file`` via
    ``flexmesa._rust``.

    Args:
        path: Path to the file on disk.

    Returns:
        Lowercase 64-character hexadecimal ``SHA-256`` digest.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    return _rust.compute_sha256_file(str(path))


def verify_file_sha256(
    path: str | pathlib.Path,
    expected_sha256: str,
) -> str:
    """Verify that ``path`` matches ``expected_sha256``.

    Args:
        path: Path to the file on disk.
        expected_sha256: Expected 64-character hexadecimal ``SHA-256`` digest.

    Returns:
        Verified lowercase 64-character hexadecimal ``SHA-256`` digest.

    Raises:
        ValueError: If ``expected_sha256`` is not a valid 64-character hex
            string or if the computed file digest does not match
            ``expected_sha256``.
    """
    if not is_valid_sha256_hex(expected_sha256):
        raise ValueError(
            f"Invalid expected SHA-256 hex digest: {expected_sha256!r}"
        )
    expected_norm = expected_sha256.strip().lower()
    actual = compute_sha256_file(path)
    if actual != expected_norm:
        raise ValueError(
            f"SHA-256 checksum mismatch for {pathlib.Path(path).name}: "
            f"expected {expected_norm}, got {actual}"
        )
    return actual


def _validate_table_entry(raw: dict[str, object]) -> TableArtifact:
    """Validate and convert one ``[[tables]]`` dict to ``TableArtifact``."""
    if unknown := set(raw.keys()) - _ALLOWED_TABLE_KEYS:
        raise ValueError(f"Unknown manifest table field(s): {sorted(unknown)}")
    name, filename = raw.get("name"), raw.get("filename")
    sha256, size_bytes = raw.get("sha256"), raw.get("size_bytes")
    tensors, description = raw.get("tensors"), raw.get("description", "")

    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Manifest table entry missing valid 'name': {raw!r}")
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError(f"Manifest table {name!r} missing valid 'filename'")
    if not isinstance(sha256, str) or not is_valid_sha256_hex(sha256):
        raise ValueError(f"Manifest table {name!r} has invalid 'sha256'")
    if (
        not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes < 0
    ):
        raise ValueError(f"Manifest table {name!r} has invalid 'size_bytes'")
    if (
        not isinstance(tensors, list)
        or not tensors
        or not all(isinstance(t, str) and t.strip() for t in tensors)
    ):
        raise ValueError(
            f"Manifest table {name!r} must declare non-empty tensor names"
        )
    cleaned_tensors = tuple(str(t).strip() for t in tensors)
    if len(set(cleaned_tensors)) != len(cleaned_tensors) or not isinstance(
        description, str
    ):
        raise ValueError(f"Invalid tensors or description in table {name!r}")
    return TableArtifact(
        name=name.strip(),
        filename=filename.strip(),
        sha256=sha256.strip().lower(),
        size_bytes=size_bytes,
        tensors=cleaned_tensors,
        description=description.strip(),
    )


def parse_manifest_text(text: str) -> dict[str, TableArtifact]:
    """Parse ``manifest.toml`` text into a dict of ``TableArtifact`` entries.

    Args:
        text: Raw TOML content containing a ``[[tables]]`` array of tables.

    Returns:
        Dictionary mapping logical table ``name`` to its ``TableArtifact``.

    Raises:
        ValueError: If the TOML syntax is invalid, ``[[tables]]`` is missing,
            duplicate table names appear, or any field fails validation.
    """
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid manifest TOML syntax: {exc}") from exc

    raw_tables = parsed.get("tables")
    if not isinstance(raw_tables, list) or not raw_tables:
        raise ValueError(
            "Manifest TOML must define a non-empty [[tables]] list"
        )

    artifacts: dict[str, TableArtifact] = {}
    for entry in raw_tables:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid table entry in manifest: {entry!r}")
        artifact = _validate_table_entry(entry)
        if artifact.name in artifacts:
            raise ValueError(
                f"Duplicate table name {artifact.name!r} in manifest"
            )
        artifacts[artifact.name] = artifact
    return artifacts


def parse_manifest_file(path: str | pathlib.Path) -> dict[str, TableArtifact]:
    """Read and parse a ``manifest.toml`` file from disk.

    Args:
        path: Path to the ``manifest.toml`` file.

    Returns:
        Dictionary mapping logical table ``name`` to its ``TableArtifact``.
    """
    return parse_manifest_text(pathlib.Path(path).read_text(encoding="utf-8"))


def load_default_manifest() -> dict[str, TableArtifact]:
    """Load the bundled ``src/flexmesa/data/manifest.toml`` table manifest.

    Returns:
        Dictionary mapping logical table ``name`` to its ``TableArtifact``.
    """
    manifest_path = pathlib.Path(__file__).with_name("manifest.toml")
    return parse_manifest_file(manifest_path)


__all__ = [
    "TableArtifact",
    "compute_sha256_bytes",
    "compute_sha256_file",
    "is_valid_sha256_hex",
    "load_default_manifest",
    "parse_manifest_file",
    "parse_manifest_text",
    "verify_file_sha256",
]
