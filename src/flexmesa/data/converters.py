# SPDX-License-Identifier: LGPL-3.0-or-later
"""ASCII-to-`.safetensors` converters for MESA nuclear and EOS tables.

Converts MESA's ASCII tables (`chem/data/lodders03.data`,
`chem/data/isotopes.data`, and `eos/helm_table.dat`) into row-major `float64`
and `int64` `.safetensors` bundles obeying the Mandatory Index & Memory-Order
Contract (`docs/design/wave-0-spec.md` §2.2).
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from flexmesa.data import loader
from flexmesa.testing import golden

_PARTITION_FCN_LEN = 24
_HELM_F_COLS = (
    "f",
    "fd",
    "ft",
    "fdd",
    "ftt",
    "fdt",
    "fddt",
    "fdtt",
    "fddtt",
)
_HELM_DPDF_COLS = ("dpdf", "dpdfd", "dpdft", "dpdfdt")
_HELM_EF_COLS = ("ef", "efd", "eft", "efdt")
_HELM_XF_COLS = ("xf", "xfd", "xft", "xfdt")


def _non_empty_lines(text: str) -> list[str]:
    """Return stripped non-empty lines from `text`."""
    return [
        stripped for line in text.splitlines() if (stripped := line.strip())
    ]


def _parse_isomeric_state(name: str) -> int:
    """Extract the integer isomeric state suffix (`al26-1` -> `1`, else `0`)."""
    if "-" not in name:
        return 0
    _, suffix = name.split("-", 1)
    try:
        state = int(suffix)
    except ValueError as exc:
        raise ValueError(
            f"Invalid isomeric state in nuclide name {name!r}"
        ) from exc
    if not 0 <= state <= 99:
        raise ValueError(
            f"Isomeric state {state} out of range [0, 99] in {name!r}"
        )
    return state


def parse_lodders03_text(text: str) -> loader.LoadedTableBundle:
    """Parse MESA `chem/data/lodders03.data` text into a `LoadedTableBundle`.

    Args:
        text: Raw ASCII content of `lodders03.data` (or `lodders09.data`).

    Returns:
        `LoadedTableBundle` with 1D arrays `'atomic_number_z'` (`int64`,
        shape `(N,)`), `'mass_number_a'` (`int64`, shape `(N,)`),
        `'isotopic_percent'` (`float64`, shape `(N,)`), and
        `'number_abundance_si'` (`float64`, shape `(N,)`), plus metadata
        `'isotopes'` (comma-separated lowercase names such as `'h1,h2,he3'`).

    Raises:
        ValueError: If no entries are found or any row fails to parse.
    """
    z_vals: list[int] = []
    a_vals: list[int] = []
    pct_vals: list[float] = []
    nsi_vals: list[float] = []
    iso_names: list[str] = []

    for line_num, raw_line in enumerate(text.splitlines(), start=1):
        code = raw_line.split("!", 1)[0].strip()
        if not code:
            continue
        tokens = code.split()
        if len(tokens) != 5:
            raise ValueError(
                f"Malformed lodders03 row at line {line_num}: {raw_line!r}"
            )
        z_val, el_sym, a_val = int(tokens[0]), tokens[1].lower(), int(tokens[2])
        z_vals.append(z_val)
        a_vals.append(a_val)
        pct_vals.append(golden.parse_fortran_float(tokens[3]))
        nsi_vals.append(golden.parse_fortran_float(tokens[4].replace(",", "")))
        iso_names.append(f"{el_sym}{a_val}")

    if not z_vals:
        raise ValueError("No isotopic abundance rows found in lodders03 input")

    return loader.LoadedTableBundle(
        tensors={
            "atomic_number_z": np.asarray(z_vals, dtype=np.int64),
            "mass_number_a": np.asarray(a_vals, dtype=np.int64),
            "isotopic_percent": np.asarray(pct_vals, dtype=np.float64),
            "number_abundance_si": np.asarray(nsi_vals, dtype=np.float64),
        },
        metadata={
            "table": "lodders03",
            "isotopes": ",".join(iso_names),
        },
    )


def _parse_single_nuclide_record(
    lines: list[str],
    idx: int,
) -> tuple[str, float, int, int, int, float, float, list[float]]:
    """Parse one 4-line nuclide record from `isotopes.data`."""
    hdr = lines[idx].split()
    if len(hdr) != 6:
        raise ValueError(
            f"Malformed isotopes.data header at record {idx // 4 + 1}: "
            f"{lines[idx]!r}"
        )
    name = hdr[0].strip()
    w_val = golden.parse_fortran_float(hdr[1])
    z_val = int(hdr[2])
    n_val = int(hdr[3])
    iso_state = _parse_isomeric_state(name)
    spin_val = golden.parse_fortran_float(hdr[4])
    mex_val = golden.parse_fortran_float(hdr[5])

    pf_vals: list[float] = []
    for offset in (1, 2, 3):
        row_tokens = lines[idx + offset].split()
        if len(row_tokens) != 8:
            raise ValueError(
                f"Expected 8 partition-function values on line "
                f"{idx + offset + 1} for {name!r}, got {len(row_tokens)}"
            )
        pf_vals.extend(golden.parse_fortran_float(tok) for tok in row_tokens)
    return name, w_val, z_val, n_val, iso_state, spin_val, mex_val, pf_vals


def parse_isotopes_text(text: str) -> loader.LoadedTableBundle:
    """Parse MESA `chem/data/isotopes.data` text into a `LoadedTableBundle`.

    Args:
        text: Raw ASCII content of `isotopes.data` (4 non-empty lines per
            nuclide: header `name W Z N spin mass_excess` followed by 24
            partition-function values across 3 lines of 8 values).

    Returns:
        `LoadedTableBundle` containing `'atomic_number_z'` (`int64`, `(N,)`),
        `'neutron_number_n'` (`int64`, `(N,)`), `'isomeric_state'` (`int64`,
        `(N,)`), `'atomic_mass_w'` (`float64`, `(N,)`), `'spin'` (`float64`,
        `(N,)`), `'mass_excess_mev'` (`float64`, `(N,)`), and
        `'partition_function'` (`float64`, `(N, 24)`), plus metadata
        `'nuclide_names'`.

    Raises:
        ValueError: If the non-empty line count is zero or not a multiple of 4,
            or if any record is malformed.
    """
    non_empty = _non_empty_lines(text)
    if not non_empty or len(non_empty) % 4 != 0:
        raise ValueError(
            f"isotopes.data must have a positive multiple of 4 non-empty "
            f"lines, got {len(non_empty)}"
        )
    n_nuclides = len(non_empty) // 4
    z_arr = np.empty(n_nuclides, dtype=np.int64)
    n_arr = np.empty(n_nuclides, dtype=np.int64)
    iso_arr = np.empty(n_nuclides, dtype=np.int64)
    w_arr = np.empty(n_nuclides, dtype=np.float64)
    spin_arr = np.empty(n_nuclides, dtype=np.float64)
    mex_arr = np.empty(n_nuclides, dtype=np.float64)
    pf_arr = np.empty((n_nuclides, _PARTITION_FCN_LEN), dtype=np.float64)
    names: list[str] = []

    for i in range(n_nuclides):
        rec = _parse_single_nuclide_record(non_empty, i * 4)
        names.append(rec[0])
        w_arr[i], z_arr[i], n_arr[i], iso_arr[i] = rec[1:5]
        spin_arr[i], mex_arr[i], pf_arr[i, :] = rec[5], rec[6], rec[7]

    return loader.LoadedTableBundle(
        tensors={
            "atomic_number_z": z_arr,
            "neutron_number_n": n_arr,
            "isomeric_state": iso_arr,
            "atomic_mass_w": w_arr,
            "spin": spin_arr,
            "mass_excess_mev": mex_arr,
            "partition_function": pf_arr,
        },
        metadata={"table": "isotopes", "nuclide_names": ",".join(names)},
    )


def _parse_helm_section(
    lines: list[str],
    start_idx: int,
    imax: int,
    jmax: int,
    n_cols: int,
) -> npt.NDArray[np.float64]:
    """Parse one `helm_table.dat` section into `(imax, jmax, n_cols)`."""
    out = np.empty((imax, jmax, n_cols), dtype=np.float64)
    for j in range(jmax):
        for i in range(imax):
            line = lines[start_idx + j * imax + i]
            tokens = line.split()
            if len(tokens) != n_cols:
                raise ValueError(
                    f"Expected {n_cols} columns in helm_table row "
                    f"{start_idx + j * imax + i + 1}, got {len(tokens)}"
                )
            out[i, j, :] = [golden.parse_fortran_float(tok) for tok in tokens]
    return out


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
    """Parse MESA `eos/helm_table.dat` text into a `LoadedTableBundle`.

    Follows MESA's `read_helm_table` loop order (`j = 0..jmax` outer temperature
    loop, `i = 0..imax` inner density loop) and stores `FlexMESA`'s canonical
    row-major `(imax, jmax)` layout where `[i_rho, j_temp]` matches Fortran
    `h% f(i_rho + 1, j_temp + 1)`.

    Args:
        text: Raw ASCII content of `helm_table.dat`.
        imax: Number of density grid points (`>= 2`, default `2701`).
        jmax: Number of temperature grid points (`>= 2`, default `1001`).
        logdlo: Minimum $\\log_{10}(\\rho / Y_e)$ (`default -12.0`).
        logdhi: Maximum $\\log_{10}(\\rho / Y_e)$ (`default 15.0`).
        logtlo: Minimum $\\log_{10} T$ (`default 3.0`).
        logthi: Maximum $\\log_{10} T$ (`default 13.0`).

    Returns:
        `LoadedTableBundle` with 1D grids (`'log_d_grid'`, `'log_t_grid'`,
        `'d_grid'`, `'t_grid'`), the 21 individual `(imax, jmax)` `float64`
        tables (`'f'` .. `'xfdt'`), and the 4 packed jet tensors (`'f_jet'`
        `(imax, jmax, 9)`, `'dpdf_jet'`, `'ef_jet'`, `'xf_jet'`
        `(imax, jmax, 4)`).

    Raises:
        ValueError: If `imax < 2`, `jmax < 2`, or the line count does not equal
            `4 * imax * jmax`.
    """
    if imax < 2 or jmax < 2:
        raise ValueError(f"imax and jmax must be >= 2, got ({imax}, {jmax})")
    non_empty = _non_empty_lines(text)
    n_block = imax * jmax
    if len(non_empty) != 4 * n_block:
        raise ValueError(
            f"Expected {4 * n_block} non-empty rows for ({imax}, {jmax}) "
            f"helm_table, got {len(non_empty)}"
        )
    f_jet = _parse_helm_section(non_empty, 0, imax, jmax, 9)
    dpdf_jet = _parse_helm_section(non_empty, n_block, imax, jmax, 4)
    ef_jet = _parse_helm_section(non_empty, 2 * n_block, imax, jmax, 4)
    xf_jet = _parse_helm_section(non_empty, 3 * n_block, imax, jmax, 4)

    log_d = np.linspace(logdlo, logdhi, imax, dtype=np.float64)
    log_t = np.linspace(logtlo, logthi, jmax, dtype=np.float64)
    tensors: dict[str, loader.TableTensor] = {
        "log_d_grid": log_d,
        "log_t_grid": log_t,
        "d_grid": np.power(10.0, log_d, dtype=np.float64),
        "t_grid": np.power(10.0, log_t, dtype=np.float64),
        "f_jet": f_jet,
        "dpdf_jet": dpdf_jet,
        "ef_jet": ef_jet,
        "xf_jet": xf_jet,
    }
    for col_names, jet in (
        (_HELM_F_COLS, f_jet),
        (_HELM_DPDF_COLS, dpdf_jet),
        (_HELM_EF_COLS, ef_jet),
        (_HELM_XF_COLS, xf_jet),
    ):
        for col_idx, name in enumerate(col_names):
            tensors[name] = np.ascontiguousarray(jet[:, :, col_idx])

    return loader.LoadedTableBundle(
        tensors=tensors,
        metadata={"table": "helm_table", "imax": str(imax), "jmax": str(jmax)},
    )


def _convert_text_file(
    input_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    parser: Callable[[str], loader.LoadedTableBundle],
) -> str:
    """Read `input_path`, parse it with `parser`, and save `.safetensors`."""
    bundle = parser(pathlib.Path(input_path).read_text(encoding="utf-8"))
    return loader.save_safetensors_bundle(
        output_path, bundle.tensors, bundle.metadata
    )


def convert_lodders03_file(
    input_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
) -> str:
    """Convert `lodders03.data` into `.safetensors` and return SHA-256."""
    return _convert_text_file(input_path, output_path, parse_lodders03_text)


def convert_isotopes_file(
    input_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
) -> str:
    """Convert `isotopes.data` into `.safetensors` and return SHA-256."""
    return _convert_text_file(input_path, output_path, parse_isotopes_text)


def convert_helm_table_file(
    input_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    *,
    imax: int = 2701,
    jmax: int = 1001,
) -> str:
    """Convert `helm_table.dat` into `.safetensors` and return SHA-256."""
    return _convert_text_file(
        input_path,
        output_path,
        lambda text: parse_helm_table_text(text, imax=imax, jmax=jmax),
    )
