# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generates publication-styled Wave 2 diagnostic figures for FlexMESA docs.

Produces two 300 dpi PNG figures in `docs/assets/figures/` following the
AASTeX publication plot style (Computer Modern typography, inward ticks on all
four sides, frameless legends, and journal-derived widths):
1. `colors_kurucz_sed_johnson.png`: Kurucz (2003) 3D cubic Hermite emergent
   surface flux spectra overlaid with Johnson $UBVRI$ filter curves (left) and
   synthetic color indices across $T_{\\mathrm{eff}}$ and $[\\mathrm{M/H}]$
   (right).
2. `wave2_benchmark_staircase.png`: Four-way Wave 2.5 runtime comparison at
   $N_{\\mathrm{zones}} = 1024$ across Upstream Fortran MESA, Safe Rust,
   Python/PyO3 (`flexmesa._rust`), and Python/JAX (`@jax.jit(vmap)`) (left),
   paired with $\\mathrm{Fe}^{56}$ in $\\mathrm{He}^4$ effective charge
   $Z_{\\mathrm{eff}}(\\log_{10} n_e, \\log_{10} T)$ profiles (right).
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure

from flexmesa import colors as _colors
from flexmesa import constants as _const
from flexmesa import ionization as _ion

PT_PER_INCH = 72.27
TEXT_WIDTH = 513.11743 / PT_PER_INCH


def _two_panel_fig() -> tuple[figure.Figure, np.ndarray]:
    """Creates a two-panel text-width publication figure."""
    fig, ax = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.38 * TEXT_WIDTH))
    fig.subplots_adjust(wspace=0.26)
    return fig, ax


def _save_fig(fig: figure.Figure, path: pathlib.Path) -> None:
    """Saves `fig` as a tight 300 dpi PNG and closes it."""
    fig.savefig(str(path), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_colors_kurucz_sed_johnson(out_dir: pathlib.Path) -> None:
    """Plots Kurucz emergent SEDs with Johnson filters and color indices.

    Args:
        out_dir: Target directory for `colors_kurucz_sed_johnson.png`.
    """
    grid = _colors.get_builtin_colors_grid()
    wl, sed_solar = _colors.eval_sed(
        5920.0, 4.30, 0.0, 1.0, 1.0, backend="rust"
    )
    _, sed_cool = _colors.eval_sed(
        5770.0, 4.10, -0.40, 1.0, 1.0, backend="rust"
    )
    wl_np = np.asarray(wl, dtype=np.float64)
    s_sol = np.maximum(np.asarray(sed_solar, dtype=np.float64), 1.0)
    s_cool = np.maximum(np.asarray(sed_cool, dtype=np.float64), 1.0)

    fig, ax = _two_panel_fig()
    mask = (wl_np >= 3000.0) & (wl_np <= 9500.0)
    ax[0].plot(
        wl_np[mask] * 1.0e-3,
        np.log10(s_sol[mask]),
        color="C0",
        lw=1.7,
        label=r"$T_{\mathrm{eff}}=5920\,\mathrm{K},\,[\mathrm{M/H}]=0.0$",
    )
    ax[0].plot(
        wl_np[mask] * 1.0e-3,
        np.log10(s_cool[mask]),
        color="C1",
        lw=1.4,
        ls="--",
        label=r"$T_{\mathrm{eff}}=5770\,\mathrm{K},\,[\mathrm{M/H}]=-0.4$",
    )
    # Overlay scaled Johnson B (0), V (6), R (4) passbands near bottom of panel
    f_grid_np = np.asarray(grid.filters_on_sed_grid, dtype=np.float64)
    for f_idx, f_col, f_lbl in ((0, "C2", "B"), (6, "C3", "V"), (4, "C4", "R")):
        f_tr = f_grid_np[f_idx, mask]
        f_scaled = 5.65 + 0.55 * (f_tr / max(float(np.max(f_tr)), 1.0e-12))
        ax[0].plot(
            wl_np[mask] * 1.0e-3,
            f_scaled,
            color=f_col,
            lw=1.1,
            alpha=0.85,
            label=f_lbl,
        )

    ax[0].set_xlim(3.0, 9.5)
    ax[0].set_ylim(5.6, 7.65)
    ax[0].set_xlabel(r"Wavelength [$\lambda / 10^3\,\mathrm{\AA}$]")
    ax[0].set_ylabel(
        r"Surface Flux "
        r"[$\log_{10}(F_\lambda/\mathrm{erg\,cm^{-2}\,s^{-1}\,\AA^{-1}})$]"
    )
    ax[0].legend(frameon=False, loc="upper right", ncol=2, fontsize=8)

    teff_arr = np.linspace(5760.0, 5990.0, 32, dtype=np.float64)
    log_g_arr = np.full_like(teff_arr, 4.25)
    r_arr = np.full_like(teff_arr, _const.RSUN)

    res_sol = np.asarray(
        _colors.eval_colors_zones(
            teff_arr,
            log_g_arr,
            np.zeros_like(teff_arr),
            r_arr,
            backend="rust",
        ),
        dtype=np.float64,
    )
    res_mp = np.asarray(
        _colors.eval_colors_zones(
            teff_arr,
            log_g_arr,
            np.full_like(teff_arr, -0.40),
            r_arr,
            backend="rust",
        ),
        dtype=np.float64,
    )
    res_jax = np.asarray(
        _colors.eval_colors_zones(
            teff_arr[::4],
            log_g_arr[::4],
            np.zeros_like(teff_arr[::4]),
            r_arr[::4],
            backend="jax",
        ),
        dtype=np.float64,
    )
    # Column 3 is B, Column 9 is V in COLOR_COLUMN_NAMES
    bv_sol = res_sol[:, 3] - res_sol[:, 9]
    bv_mp = res_mp[:, 3] - res_mp[:, 9]
    bv_jax = res_jax[:, 3] - res_jax[:, 9]

    ax[1].plot(
        teff_arr, bv_sol, color="C0", lw=1.8, label=r"$[\mathrm{M/H}] = 0.0$"
    )
    ax[1].plot(
        teff_arr,
        bv_mp,
        color="C1",
        lw=1.6,
        ls="-.",
        label=r"$[\mathrm{M/H}] = -0.4$",
    )
    ax[1].scatter(
        teff_arr[::4],
        bv_jax,
        color="C3",
        s=18,
        zorder=5,
        label=r"JAX $\mathtt{@jit}$ ($[\mathrm{M/H}]=0.0$)",
    )
    ax[1].set_xlim(5760.0, 5990.0)
    ax[1].set_ylim(0.545, 0.688)
    ax[1].set_xlabel(r"Effective Temperature [$T_{\mathrm{eff}}/\mathrm{K}$]")
    ax[1].set_ylabel(r"Synthetic Color Index [$B - V$ (mag)]")
    ax[1].legend(frameon=False, loc="upper right", fontsize=8.5)
    _save_fig(fig, out_dir / "colors_kurucz_sed_johnson.png")


def plot_wave2_benchmark_staircase(out_dir: pathlib.Path) -> None:
    """Plots the 4-way Wave 2.5 runtime comparison and Fe56 ionization curves.

    Args:
        out_dir: Target directory for `wave2_benchmark_staircase.png`.
    """
    workloads = ["chem_comp", "ion_fe56", "ion_4d", "colors_sed"]
    x_pos = np.arange(len(workloads), dtype=np.float64)
    w_bar = 0.19
    fortran_4t = np.array([3.78, 8.65, 290.53, 28570.68])
    rust_4t = np.array([3.82, 17.80, 230.60, 1638.59])
    pyo3_4t = np.array([10.51, 12.76, 293.93, 1570.04])
    jax_vmap = np.array([45.96, 359.66, 318.62, 744.64])

    fig, ax = _two_panel_fig()
    bars = (
        (-1.5, fortran_4t, "C2", 0.85, "Fortran MESA (4t)"),
        (-0.5, rust_4t, "C0", 0.95, "Safe Rust (4t)"),
        (0.5, pyo3_4t, "C1", 0.90, "PyO3 (4t)"),
        (1.5, jax_vmap, "C3", 0.85, "JAX (vmap)"),
    )
    for mult, arr, col, alp, lbl in bars:
        ax[0].bar(
            x_pos + mult * w_bar,
            np.log10(arr),
            width=w_bar,
            color=col,
            alpha=alp,
            label=lbl,
        )
    ax[0].set_xticks(x_pos)
    ax[0].set_xticklabels(workloads, fontsize=8.5)
    ax[0].set_ylim(0.0, 5.0)
    ax[0].set_xlabel(r"Wave 2 Workload ($N_{\mathrm{zones}} = 1024$)")
    ax[0].set_ylabel(r"Latency [$\log_{10}(t / \mu\mathrm{s})$]")
    ax[0].legend(frameon=False, loc="upper left", fontsize=8)

    log10_t = np.linspace(6.0, 8.0, 96, dtype=np.float64)
    for idx, (lne_val, col, ls) in enumerate(
        ((24.5, "C0", "-"), (26.5, "C1", "--"), (28.5, "C2", "-."))
    ):
        lne_arr = np.full_like(log10_t, lne_val)
        z_out = np.asarray(
            _ion.eval_charge_of_fe56_in_he4_zones(
                lne_arr, log10_t, backend="rust"
            ),
            dtype=np.float64,
        )
        lbl = rf"$\log_{{10}} n_e = {lne_val:.1f}$"
        ax[1].plot(
            log10_t,
            z_out[:, 0],
            color=col,
            ls=ls,
            lw=1.7 - 0.1 * idx,
            label=lbl,
        )
    ax[1].set_xlim(6.0, 8.0)
    ax[1].set_ylim(14.0, 26.5)
    ax[1].set_xlabel(r"Temperature [$\log_{10}(T/\mathrm{K})$]")
    ax[1].set_ylabel(
        r"Effective Charge "
        r"[$Z_{\mathrm{eff}}(\mathrm{Fe}^{56}\text{ in }\mathrm{He}^4)$]"
    )
    ax[1].legend(frameon=False, loc="lower right", fontsize=8.5)
    _save_fig(fig, out_dir / "wave2_benchmark_staircase.png")
