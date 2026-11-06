# SPDX-License-Identifier: LGPL-3.0-or-later
"""Type annotations for the compiled ``flexmesa._rust`` PyO3 extension.

Declares the public Python-to-Rust boundary across the Three-Tier Hybrid
Architecture:

1. **Tier A (`utils`, `testing`, `schema`, `data`, `auto_diff`)**:
   Canonical Safe Rust implementations of IEEE-754 finiteness predicates,
   Fortran lexical scanners, golden-baseline parsers, control schema and
   namelist parsers, SHA-256 digests, `.safetensors` table converters, and
   2nd-order dual-number benchmark kernels.
2. **Tier B (`math`, `interp_1d`, `interp_2d`, `neu`, `turb`)**:
   Domain-safe libm intrinsics, 1D Steffen/Huynh monotonic splines, 2D
   PSPLINE/Steffen bicubic surface interpolants, Itoh et al. (1996) 5-channel
   thermal neutrino losses, and Cox (1968) / Kuhfuss (1986) convective
   transport (`MLT` and `TDC`), with both single-zone scalar entry points and
   persistent 4-thread `rayon` multi-zone batch evaluators.
3. **Tier C (`mtx`)**:
   Coupled block-tridiagonal linear solvers (`solve_block_tridiagonal`
   sequential block-Thomas, `solve_block_tridiag_bcyclic` Hirshman et al. 2010
   odd-even cyclic reduction, and `block_multiply_xa` transpose/left-product
   kernel for reverse-mode Implicit Function Theorem adjoints).

All array arguments (`npt.NDArray[np.float64]`) expect 1D C-contiguous
``float64`` buffers borrowed zero-copy via ``PyReadonlyArray1::as_slice``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Tier A: Infrastructure (`utils`, `testing`, `schema`, `data`, `auto_diff`)
# ---------------------------------------------------------------------------
# IEEE-754 finiteness checks, Fortran lexical scanners, golden record parsers,
# control schema and namelist parsers, SHA-256 hashing, and `.safetensors`
# ASCII table converters (`lodders03.data`, `isotopes.data`, `helm_table.dat`).
def is_bad(x: float) -> bool: ...
def is_bad_array(xs: npt.NDArray[np.float64]) -> bool: ...
def replace_bad_nums(
    xs: npt.NDArray[np.float64], replacement: float
) -> npt.NDArray[np.float64]: ...
def remove_underbars(text: str) -> str: ...
def split_line(line: str, num: int) -> list[str]: ...
def tokenize(text: str) -> list[tuple[str, str]]: ...
def is_fortran_float_token(token: str) -> bool: ...
def parse_fortran_float(token: str) -> float: ...
# Golden baseline parser (`parse_golden_text`), control schema and namelist
# parsers (`parse_control_value`, `parse_defaults_text`, `parse_columns_text`,
# `parse_namelist_text`), SHA-256 hex validators and digests, `.safetensors`
# ASCII table converters, and 2nd-order dual-number benchmark kernels.
def parse_golden_text(
    text: str,
) -> list[tuple[int, str, str, list[int], npt.NDArray[np.float64]]]: ...
def parse_control_value(raw: str) -> tuple[str, bool | int | float | str]: ...
def parse_defaults_text(
    text: str,
) -> list[
    tuple[
        str,
        str,
        bool | int | float | str,
        bool,
        list[tuple[str, bool | int | float | str]],
        str,
        str,
        int,
    ]
]: ...
def parse_columns_text(text: str) -> list[tuple[str, str, str, int]]: ...
def parse_namelist_text(
    text: str,
) -> list[
    tuple[
        str,
        list[tuple[str, str | None, str, bool | int | float | str, int]],
    ]
]: ...
def is_valid_sha256_hex(digest: str) -> bool: ...
def compute_sha256_bytes(data: bytes) -> str: ...
def compute_sha256_file(path: str) -> str: ...
def parse_lodders03_text(
    text: str,
) -> tuple[
    dict[str, npt.NDArray[np.float64] | npt.NDArray[np.int64]], dict[str, str]
]: ...
def parse_isotopes_text(
    text: str,
) -> tuple[
    dict[str, npt.NDArray[np.float64] | npt.NDArray[np.int64]], dict[str, str]
]: ...
def parse_helm_table_text(
    text: str,
    imax: int,
    jmax: int,
    logdlo: float,
    logdhi: float,
    logtlo: float,
    logthi: float,
) -> tuple[
    dict[str, npt.NDArray[np.float64] | npt.NDArray[np.int64]], dict[str, str]
]: ...
def eval_autodiff_order2_zone(x: float, y: float) -> list[float]: ...
def eval_autodiff_order2_batch(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    parallel: bool = False,
) -> npt.NDArray[np.float64]: ...

# ---------------------------------------------------------------------------
# Tier B: Microphysics & Numerics (`math`, `interp_1d`, `interp_2d`, `neu`,
# `turb`)
# ---------------------------------------------------------------------------
# Domain-safe scalar and 1D array elementary functions (`safe_sqrt`, `safe_log`,
# `safe_log10`, `exp10`, `pow`, `sinpi`, `cospi`, `tanpi`, `sgn`, `heaviside`),
# 1D Steffen/Huynh spline builders and evaluators, 2D PSPLINE/Steffen bicubic
# setup and evaluation, Itoh et al. (1996) 5-channel thermal neutrino losses,
# and Cox (1968) / Kuhfuss (1986) convective transport (`MLT` and `TDC`).
def safe_sqrt(x: float) -> float: ...
def safe_log(x: float) -> float: ...
def safe_log10(x: float) -> float: ...
def exp10(x: float) -> float: ...
def pow(x: float, y: float) -> float: ...
def sinpi(x: float) -> float: ...
def cospi(x: float) -> float: ...
def tanpi(x: float) -> float: ...
def sgn(x: float) -> float: ...
def heaviside(x: float) -> float: ...
# 1D C-contiguous NumPy array wrappers for elementary functions (`safe_log`,
# `safe_log10`, `safe_sqrt`, `exp10`) and whitespace-delimited vector parsing.
def str_to_vector(text: str) -> npt.NDArray[np.float64]: ...
def safe_log_array(xs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
def safe_log10_array(
    xs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def safe_sqrt_array(
    xs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def exp10_array(xs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
# 1D Steffen (1990) and Huynh (1993) spline builders (`(nx, 4)` output) and
# evaluators (`(nq,)` output) over strictly monotonic 1D knot grids.
def interp_pm(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]: ...
def interp_m3a(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]: ...
def interp_m3b(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]: ...
def interp_m3q(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]: ...
def interp_values(
    init_x: npt.NDArray[np.float64],
    coeffs_flat: npt.NDArray[np.float64],
    x_query: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def interp_values_and_slopes(
    init_x: npt.NDArray[np.float64],
    coeffs_flat: npt.NDArray[np.float64],
    x_query: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def integrate_values(
    init_x: npt.NDArray[np.float64],
    coeffs_flat: npt.NDArray[np.float64],
    x_query: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...

# 2D PSPLINE C^2 bicubic (`interp_mkbicub_db`, `interp_evbicub_db`,
# `interp_evbicub_batch`) and Steffen (1990) 2D piecewise-monotonic
# (`interp_mkbipm_db`, `interp_evbipm_db`) surface interpolation.
def interp_mkbicub_db(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z_flat: npt.NDArray[np.float64],
    ibcxmin: int,
    bcxmin: npt.NDArray[np.float64],
    ibcxmax: int,
    bcxmax: npt.NDArray[np.float64],
    ibcymin: int,
    bcymin: npt.NDArray[np.float64],
    ibcymax: int,
    bcymax: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], bool, bool]: ...
def interp_evbicub_db(
    xget: float,
    yget: float,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    ilinx: bool,
    iliny: bool,
    coeffs_flat: npt.NDArray[np.float64],
) -> list[float]: ...
def interp_evbicub_batch(
    x_q: npt.NDArray[np.float64],
    y_q: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    coeffs_flat: npt.NDArray[np.float64],
    parallel: bool = False,
) -> npt.NDArray[np.float64]: ...
def interp_mkbipm_db(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z_flat: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def interp_evbipm_db(
    xget: float,
    yget: float,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    coeffs_flat: npt.NDArray[np.float64],
) -> float: ...
# Itoh et al. (1996) 5-channel thermal neutrino loss rates (`neu_get`,
# `neu_get_batch`) and Cox (1968) / Kuhfuss (1986) convective transport
# (`set_mlt_f64`, `set_mlt_batch`, `set_tdc_batch`).
def neu_get(
    t: float | None = None,
    log10_t: float | None = None,
    rho: float | None = None,
    log10_rho: float | None = None,
    abar: float = 4.0,
    zbar: float = 2.0,
    log10_tlim: float = 7.5,
    flags: Sequence[bool] = (True, True, True, True, True),
) -> tuple[list[float], list[list[float]]]: ...
def neu_get_batch(
    t: npt.NDArray[np.float64],
    log10_t: npt.NDArray[np.float64],
    rho: npt.NDArray[np.float64],
    log10_rho: npt.NDArray[np.float64],
    abar: npt.NDArray[np.float64],
    zbar: npt.NDArray[np.float64],
    log10_tlim: float = 7.5,
    flags: Sequence[bool] = (True, True, True, True, True),
    parallel: bool = False,
) -> npt.NDArray[np.float64]: ...
def set_mlt_f64(
    option: str,
    mixing_length_alpha: float,
    henyey_y_param: float,
    henyey_nu_param: float,
    chi_t: float,
    chi_rho: float,
    cp: float,
    grav: float,
    lambda_mlt: float,
    rho: float,
    p: float,
    t: float,
    opacity: float,
    gradr: float,
    grada: float,
    gradl: float,
) -> tuple[list[float], int]: ...
def set_mlt_batch(
    s_frac: npt.NDArray[np.float64],
    parallel: bool = False,
) -> npt.NDArray[np.float64]: ...
def set_tdc_batch(
    s_frac: npt.NDArray[np.float64],
    parallel: bool = False,
) -> npt.NDArray[np.float64]: ...

# ---------------------------------------------------------------------------
# Tier C: Coupled Linear Solvers (`mtx`)
# ---------------------------------------------------------------------------
# Sequential block-Thomas (`solve_block_tridiagonal`), Hirshman et al. (2010)
# odd-even block cyclic reduction (`solve_block_tridiag_bcyclic`) with optional
# persistent 4-thread `rayon` execution, and left block-tridiagonal product
# (`block_multiply_xa`) for reverse-mode Implicit Function Theorem adjoints.
def solve_block_tridiagonal(
    nvar: int,
    nz: int,
    lblk: npt.NDArray[np.float64],
    dblk: npt.NDArray[np.float64],
    ublk: npt.NDArray[np.float64],
    rhs: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def solve_block_tridiag_bcyclic(
    nvar: int,
    nz: int,
    lblk: npt.NDArray[np.float64],
    dblk: npt.NDArray[np.float64],
    ublk: npt.NDArray[np.float64],
    rhs: npt.NDArray[np.float64],
    parallel: bool = False,
) -> npt.NDArray[np.float64]: ...
def block_multiply_xa(
    nvar: int,
    nz: int,
    lblk: npt.NDArray[np.float64],
    dblk: npt.NDArray[np.float64],
    ublk: npt.NDArray[np.float64],
    vec_x: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
