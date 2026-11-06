# SPDX-License-Identifier: LGPL-3.0-or-later
"""ASCII-to-``.safetensors`` converters for MESA nuclear and EOS tables.

Converts MESA's ASCII tables (``chem/data/lodders03.data``,
``chem/data/isotopes.data``, and ``eos/helm_table.dat``) into row-major
``float64`` and ``int64`` ``.safetensors`` bundles obeying the Mandatory Index &
Memory-Order Contract (``docs/design/wave-0-spec.md`` §2.2) by delegating text
parsing to the canonical Safe Rust ``flexmesa_data::converters`` implementation
in ``flexmesa._rust``.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable

from flexmesa import _rust
from flexmesa.data import loader


def parse_lodders03_text(text: str) -> loader.LoadedTableBundle:
    """Parse ``chem/data/lodders03.data`` text into a ``LoadedTableBundle``.

    Delegates directly to ``flexmesa_data::converters::parse_lodders03_text``
    via ``flexmesa._rust``.

    Args:
        text: Raw ASCII content of ``lodders03.data`` (or ``lodders09.data``).

    Returns:
        ``LoadedTableBundle`` with 1D arrays ``'atomic_number_z'`` (``int64``,
        shape ``(N,)``, dimensionless proton number $Z$), ``'mass_number_a'``
        (``int64``, shape ``(N,)``, dimensionless nucleon number $A$),
        ``'isotopic_percent'`` (``float64``, shape ``(N,)``, atom ``%``), and
        ``'number_abundance_si'`` (``float64``, shape ``(N,)``, solar number
        abundance normalized to $N(\\text{Si}) = 10^6$), plus metadata
        ``'isotopes'`` (comma-separated lowercase names such as
        ``'h1,h2,he3'``).

    Raises:
        ValueError: If no entries are found or any row fails to parse.
    """
    tensors, metadata = _rust.parse_lodders03_text(text)
    return loader.LoadedTableBundle(tensors=tensors, metadata=metadata)


def parse_isotopes_text(text: str) -> loader.LoadedTableBundle:
    """Parse MESA ``chem/data/isotopes.data`` text into a ``LoadedTableBundle``.

    Delegates directly to ``flexmesa_data::converters::parse_isotopes_text``
    via ``flexmesa._rust``.

    Args:
        text: Raw ASCII content of ``isotopes.data`` (4 non-empty lines per
            nuclide consisting of a ``name W Z N spin mass_excess`` header
            followed by 24 partition-function values across 3 lines of 8
            values).

    Returns:
        ``LoadedTableBundle`` containing ``'atomic_number_z'`` (``int64``,
        ``(N,)``), ``'neutron_number_n'`` (``int64``, ``(N,)``),
        ``'isomeric_state'`` (``int64``, ``(N,)``), ``'atomic_mass_w'``
        (``float64``, ``(N,)``, atomic mass units $\\text{u}$), ``'spin'``
        (``float64``, ``(N,)``, ground-state spin in $\\hbar$),
        ``'mass_excess_mev'`` (``float64``, ``(N,)``, $\\text{MeV}$), and
        ``'partition_function'`` (``float64``, ``(N, 24)``, dimensionless),
        plus metadata ``'nuclide_names'``.

    Raises:
        ValueError: If the non-empty line count is zero or not a multiple of 4,
            or if any record is malformed.
    """
    tensors, metadata = _rust.parse_isotopes_text(text)
    return loader.LoadedTableBundle(tensors=tensors, metadata=metadata)


def parse_helm_table_text(
    text: str,
    *,
    imax: int = 2701,
    jmax: int = 1001,
    logdlo: float = -12.0,
    logdhi: float = 15.0,
    logtlo: float = 3.0,
    logthi: float = 13.0,
) -> loader.LoadedTableBundle:
    """Parse MESA ``eos/helm_table.dat`` text into a ``LoadedTableBundle``.

    Delegates directly to ``flexmesa_data::converters::parse_helm_table_text``
    via ``flexmesa._rust``. Follows MESA's ``read_helm_table`` loop order
    (``j = 0..jmax`` outer temperature loop, ``i = 0..imax`` inner density loop)
    and stores ``FlexMESA``'s canonical row-major ``(imax, jmax)`` layout where
    ``[i_rho, j_temp]`` matches Fortran ``h% f(i_rho + 1, j_temp + 1)``.

    Args:
        text: Raw ASCII content of ``helm_table.dat``.
        imax: Number of density grid points (``>= 2``, default ``2701``).
        jmax: Number of temperature grid points (``>= 2``, default ``1001``).
        logdlo: Minimum $\\log_{10}(\\rho / Y_e)$ (default ``-12.0``).
        logdhi: Maximum $\\log_{10}(\\rho / Y_e)$ (default ``15.0``).
        logtlo: Minimum $\\log_{10} T$ in $\\text{K}$ (default ``3.0``).
        logthi: Maximum $\\log_{10} T$ in $\\text{K}$ (default ``13.0``).

    Returns:
        ``LoadedTableBundle`` with 1D grids (``'log_d_grid'``, ``'d_grid'`` of
        shape ``(imax,)`` in $\\text{g cm}^{-3}$; ``'log_t_grid'``, ``'t_grid'``
        of shape ``(jmax,)`` in $\\text{K}$), the 21 ``(imax, jmax)`` CGS
        tables, and the 4 packed jet tensors (``'f_jet'`` ``(imax, jmax, 9)``,
        ``'dpdf_jet'``, ``'ef_jet'``, ``'xf_jet'`` ``(imax, jmax, 4)``).

    Raises:
        ValueError: If ``imax < 2``, ``jmax < 2``, grid bounds are inverted, or
            the line count does not equal ``4 * imax * jmax``.
    """
    tensors, metadata = _rust.parse_helm_table_text(
        text, imax, jmax, logdlo, logdhi, logtlo, logthi
    )
    return loader.LoadedTableBundle(tensors=tensors, metadata=metadata)


def _convert_file_with(
    input_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    parser: Callable[[str], loader.LoadedTableBundle],
) -> str:
    """Read ``input_path``, parse via ``parser``, and save ``.safetensors``."""
    text = pathlib.Path(input_path).read_text(encoding="utf-8")
    bundle = parser(text)
    return loader.save_safetensors_bundle(
        output_path,
        bundle.tensors,
        metadata=bundle.metadata,
    )


def convert_lodders03_file(
    input_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
) -> str:
    """Convert ``lodders03.data`` on disk to ``.safetensors`` (SHA-256).

    Args:
        input_path: Path to the source ``lodders03.data`` ASCII file.
        output_path: Destination ``.safetensors`` path.

    Returns:
        Lowercase 64-character hexadecimal ``SHA-256`` digest of output file.
    """
    return _convert_file_with(
        input_path,
        output_path,
        parse_lodders03_text,
    )


def convert_isotopes_file(
    input_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
) -> str:
    """Convert ``isotopes.data`` on disk to ``.safetensors`` and return SHA-256.

    Args:
        input_path: Path to the source ``isotopes.data`` ASCII file.
        output_path: Destination ``.safetensors`` path.

    Returns:
        Lowercase 64-character hexadecimal ``SHA-256`` digest of output file.
    """
    return _convert_file_with(
        input_path,
        output_path,
        parse_isotopes_text,
    )


def convert_helm_table_file(
    input_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    *,
    imax: int = 2701,
    jmax: int = 1001,
    logdlo: float = -12.0,
    logdhi: float = 15.0,
    logtlo: float = 3.0,
    logthi: float = 13.0,
) -> str:
    """Convert ``helm_table.dat`` on disk to ``.safetensors`` (SHA-256).

    Args:
        input_path: Path to the source ``helm_table.dat`` ASCII file.
        output_path: Destination ``.safetensors`` path.
        imax: Number of density grid points (default ``2701``).
        jmax: Number of temperature grid points (default ``1001``).
        logdlo: Minimum $\\log_{10}(\\rho / Y_e)$ (default ``-12.0``).
        logdhi: Maximum $\\log_{10}(\\rho / Y_e)$ (default ``15.0``).
        logtlo: Minimum $\\log_{10} T$ in $\\text{K}$ (default ``3.0``).
        logthi: Maximum $\\log_{10} T$ in $\\text{K}$ (default ``13.0``).

    Returns:
        Lowercase 64-character hexadecimal ``SHA-256`` digest of output file.
    """
    return _convert_file_with(
        input_path,
        output_path,
        lambda txt: parse_helm_table_text(
            txt,
            imax=imax,
            jmax=jmax,
            logdlo=logdlo,
            logdhi=logdhi,
            logtlo=logtlo,
            logthi=logthi,
        ),
    )


__all__ = [
    "convert_helm_table_file",
    "convert_isotopes_file",
    "convert_lodders03_file",
    "parse_helm_table_text",
    "parse_isotopes_text",
    "parse_lodders03_text",
]
