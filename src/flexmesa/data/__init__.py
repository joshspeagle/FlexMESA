# SPDX-License-Identifier: LGPL-3.0-or-later
"""Physics table manifest, `.safetensors` memory-mapped loader, and converters.

Provides `SHA-256` checksum verification against `manifest.toml`, zero-copy
memory-mapped `.safetensors` loading, and ASCII-to-`.safetensors` converters for
MESA nuclear, abundance, and equation-of-state data tables (`isotopes.data`,
`lodders03.data`, and `helm_table.dat`).
"""

from __future__ import annotations

from flexmesa.data import converters, fetch, loader, manifest

__all__ = (
    "converters",
    "fetch",
    "loader",
    "manifest",
)
