# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generates publication-styled diagnostic figures for the FlexMESA docs.

Produces four 300 dpi PNG figures in `docs/assets/figures/` following the
AASTeX publication plot style (Computer Modern typography, inward ticks on all
four sides, frameless legends, and journal-derived widths):
1. `neu_regimes.png`: Itoh et al. (1996) thermal neutrino energy-loss channels
   and logarithmic temperature sensitivity across $\\log_{10}(T/\\mathrm{K})$.
2. `interp_stencil_accuracy.png`: 1D monotonic vs. quartic-slope piecewise
   cubic interpolation (`interp_pm` vs. `interp_m3b`) across a sharp composition
   front and measured 2D bicubic spline (`interp_2d`) convergence.
3. `turb_mlt_tdc_comparison.png`: Steady-state Cox (1968) MLT vs. transient
   Kuhfuss (1986) 1D Time-Dependent Convection superadiabaticity and Implicit
   Function Theorem sensitivity $\\partial \\nabla_T / \\partial \\nabla_r$.
4. `wave1_benchmark_staircase.png`: Four-way runtime comparison at
   $N_{\\mathrm{zones}} = 4096$ across Upstream Fortran MESA, Safe Rust,
   Python/PyO3 (`flexmesa._rust`), and Python/JAX (`@jax.jit(vmap)`).
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure, font_manager

from flexmesa import auto_diff as _ad
from flexmesa import constants as _const
from flexmesa import interp_1d as _i1d
from flexmesa import interp_2d as _i2d
from flexmesa import neu as _neu
from flexmesa import turb as _turb

PT_PER_INCH = 72.27
COLUMN_WIDTH = 242.26653 / PT_PER_INCH
TEXT_WIDTH = 513.11743 / PT_PER_INCH
SMALL_SIZE = 9
NORMAL_SIZE = 10
BIG_SIZE = 12

RC_PARAMS = {
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Latin Modern Roman", "cmr10"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": 10,
    "legend.title_fontsize": 10,
    "legend.frameon": False,
    "figure.titlesize": 12,
    "figure.facecolor": "w",
    "figure.dpi": 300,
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.format": "png",
}


def main() -> None:
    """Generates all four publication-styled documentation figures."""
    _register_latin_modern()
    mpl.rcParams.update(RC_PARAMS)  # type: ignore[arg-type,unused-ignore]
    root = pathlib.Path(__file__).resolve().parent.parent
    out_dir = root / "docs" / "assets" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_neu_regimes(out_dir)
    plot_interp_stencil_accuracy(out_dir)
    plot_turb_mlt_tdc(out_dir)
    plot_wave1_benchmark_staircase(out_dir)


def _register_latin_modern() -> None:
    """Registers TeX Live Latin Modern Roman fonts with matplotlib."""
    known = {entry.name for entry in font_manager.fontManager.ttflist}
    if known & {"CMU Serif", "Latin Modern Roman"}:
        return
    kpsewhich = shutil.which("kpsewhich")
    if kpsewhich is None:
        return
    for name in (
        "lmroman10-regular.otf",
        "lmroman10-italic.otf",
        "lmroman10-bold.otf",
        "lmroman10-bolditalic.otf",
    ):
        found = subprocess.run(
            [kpsewhich, name], capture_output=True, text=True, check=False
        ).stdout.strip()
        if found:
            font_manager.fontManager.addfont(found)


def _two_panel_fig() -> tuple[figure.Figure, np.ndarray]:
    """Creates a two-panel text-width publication figure."""
    fig, ax = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.38 * TEXT_WIDTH))
    fig.subplots_adjust(wspace=0.25)
    return fig, ax


def _save_fig(fig: figure.Figure, path: pathlib.Path) -> None:
    """Saves `fig` as a tight 300 dpi PNG and closes it."""
    fig.savefig(str(path), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_neu_regimes(out_dir: pathlib.Path) -> None:
    """Plots Itoh et al. (1996) neutrino channels and temperature sensitivity.

    Args:
        out_dir: Target directory for `neu_regimes.png`.
    """
    log10_t = np.linspace(7.55, 9.60, 160, dtype=np.float64)
    t_arr = np.power(10.0, log10_t)
    rho, log10_rho, abar, zbar = 1.0e6, 6.0, 12.0, 6.0

    total = np.zeros_like(log10_t)
    pair = np.zeros_like(log10_t)
    plas = np.zeros_like(log10_t)
    phot = np.zeros_like(log10_t)
    brem = np.zeros_like(log10_t)
    dln_eps_dln_t = np.zeros_like(log10_t)

    for i, (t_val, lt_val) in enumerate(zip(t_arr, log10_t, strict=True)):
        res = _neu.neu_get(t_val, lt_val, rho, log10_rho, abar, zbar, 7.5)
        total[i] = max(res.loss[0], 1.0e-30)
        pair[i] = max(res.sources[0, 0], 1.0e-30)
        plas[i] = max(res.sources[1, 0], 1.0e-30)
        phot[i] = max(res.sources[2, 0], 1.0e-30)
        brem[i] = max(res.sources[3, 0], 1.0e-30)
        dln_eps_dln_t[i] = t_val * res.loss[1] / total[i]

    log10_total = np.log10(total)
    fig, ax = _two_panel_fig()
    ax[0].plot(log10_t, log10_total, color="C0", lw=2.0, label="Total")
    ax[0].plot(
        log10_t, np.log10(pair), color="C1", lw=1.3, ls="--", label="Pair"
    )
    ax[0].plot(
        log10_t, np.log10(plas), color="C2", lw=1.3, ls="-.", label="Plasmon"
    )
    ax[0].plot(
        log10_t, np.log10(phot), color="C3", lw=1.3, ls=":", label="Photo"
    )
    ax[0].plot(
        log10_t, np.log10(brem), color="C4", lw=1.2, alpha=0.85, label="Bremss."
    )
    ax[0].set_xlim(7.55, 9.60)
    ax[0].set_ylim(-6.0, 14.0)
    ax[0].set_xlabel(r"Temperature [$\log_{10}(T/\mathrm{K})$]")
    ax[0].set_ylabel(
        r"Neutrino Loss "
        r"[$\log_{10}(\epsilon_\nu/\mathrm{erg\,g^{-1}\,s^{-1}})$]"
    )
    ax[0].legend(frameon=False, loc="upper left")

    ax[1].plot(
        log10_t,
        dln_eps_dln_t,
        color="C0",
        lw=1.8,
        label=r"Analytic AD ($\mathrm{DualOrder1}$)",
    )
    fd_slope = np.gradient(log10_total, log10_t[1] - log10_t[0])
    ax[1].scatter(
        log10_t[::6],
        fd_slope[::6],
        color="C1",
        s=14,
        alpha=0.8,
        label="Centered Finite Diff.",
    )
    ax[1].set_xlim(7.55, 9.60)
    ax[1].set_xlabel(r"Temperature [$\log_{10}(T/\mathrm{K})$]")
    ax[1].set_ylabel(
        r"Sensitivity [$\partial \ln \epsilon_\nu / \partial \ln T$]"
    )
    ax[1].legend(frameon=False, loc="upper right")
    _save_fig(fig, out_dir / "neu_regimes.png")


def plot_interp_stencil_accuracy(out_dir: pathlib.Path) -> None:
    """Plots 1D monotonic vs. quartic-slope interpolation and 2D convergence.

    Args:
        out_dir: Target directory for `interp_stencil_accuracy.png`.
    """
    x_knots = np.linspace(0.0, 1.0, 11, dtype=np.float64)
    y_knots = 0.05 * x_knots + np.where(x_knots >= 0.45, 0.90, 0.0)
    x_fine = np.linspace(0.0, 1.0, 240, dtype=np.float64)
    y_exact = 0.05 * x_fine + np.where(x_fine >= 0.45, 0.90, 0.0)

    f_pm = _i1d.interp_pm(x_knots, y_knots)
    f_m3 = _i1d.interp_m3b(x_knots, y_knots)
    y_pm, _ = _i1d.interp_values_and_slopes(x_knots, f_pm, x_fine)
    y_m3, _ = _i1d.interp_values_and_slopes(x_knots, f_m3, x_fine)

    fig, ax = _two_panel_fig()
    ax[0].plot(x_fine, y_exact, color="C3", ls="--", lw=1.0, label="Exact Step")
    ax[0].plot(x_fine, y_pm, color="C0", lw=1.9, label="Steffen Monotonic (pm)")
    ax[0].plot(
        x_fine, y_m3, color="C1", lw=1.5, ls="-.", label="Quartic Slope (m3b)"
    )
    ax[0].scatter(x_knots, y_knots, color="C2", s=18, zorder=5, label="Knots")
    ax[0].set_xlim(0.2, 0.8)
    ax[0].set_ylim(-0.15, 1.15)
    ax[0].set_xlabel(r"Normalized Mass Coordinate [$m/M_\star$]")
    ax[0].set_ylabel(r"Helium Mass Fraction [$Y$]")
    ax[0].legend(frameon=False, loc="upper left")

    # Measure empirical 2D bicubic spline error on f(x, y) = sin(x) cos(y).
    n_grids = np.array([16, 32, 64, 128], dtype=np.float64)
    err_val = np.zeros_like(n_grids)
    err_grad = np.zeros_like(n_grids)
    xq = np.linspace(0.15, 1.35, 12, dtype=np.float64)
    yq = np.linspace(0.20, 1.30, 12, dtype=np.float64)
    for k, ng in enumerate((16, 32, 64, 128)):
        xg = np.linspace(0.0, 1.5, ng, dtype=np.float64)
        yg = np.linspace(0.0, 1.5, ng, dtype=np.float64)
        z_grid = np.sin(xg)[:, None] * np.cos(yg)[None, :]
        coeffs, ilinx, iliny = _i2d.interp_mkbicub_db(xg, yg, z_grid)
        v_err, g_err = 0.0, 0.0
        for xv in xq:
            for yv in yq:
                ev = _i2d.interp_evbicub_db(
                    float(xv), float(yv), xg, yg, ilinx, iliny, coeffs
                )
                f_ex = float(np.sin(xv) * np.cos(yv))
                fx_ex = float(np.cos(xv) * np.cos(yv))
                v_err = max(v_err, abs(float(ev.f) - f_ex))
                g_err = max(g_err, abs(float(ev.df_dx) - fx_ex))
        err_val[k], err_grad[k] = v_err, g_err

    lg_n = np.log10(n_grids)
    lbl_v = r"Value Error ($\mathcal{O}(h^4)$)"
    lbl_g = r"Gradient Error ($\mathcal{O}(h^3)$)"
    ax[1].plot(lg_n, np.log10(err_val), c="C0", marker="o", lw=1.8, label=lbl_v)
    ax[1].plot(
        lg_n,
        np.log10(err_grad),
        c="C1",
        marker="s",
        lw=1.5,
        ls="--",
        label=lbl_g,
    )
    ax[1].set_xlabel(r"Grid Resolution [$\log_{10}(N_{\mathrm{grid}})$]")
    ax[1].set_ylabel(r"Max Absolute Error [$\log_{10}(\varepsilon_{\max})$]")
    ax[1].legend(frameon=False, loc="lower left")
    _save_fig(fig, out_dir / "interp_stencil_accuracy.png")


def plot_turb_mlt_tdc(out_dir: pathlib.Path) -> None:
    """Plots Cox (1968) MLT vs. transient Kuhfuss (1986) TDC gradients.

    Args:
        out_dir: Target directory for `turb_mlt_tdc_comparison.png`.
    """
    gradr = np.linspace(0.39, 0.54, 120, dtype=np.float64)
    grad_l, t_val, rho_val = 0.40, 1.1e5, 1.1e-5
    grav = _const.MSUN * _const.STANDARD_CGRAV / (_const.RSUN * _const.RSUN)
    rgas = _const.KERG / _const.MP
    cp_val = 2.5 * rgas
    p_val = rho_val * t_val * rgas
    hp_val = p_val / (rho_val * grav)

    gt_mlt = np.zeros_like(gradr)
    dgt_mlt = np.zeros_like(gradr)
    gt_tdc = np.zeros_like(gradr)
    dgt_tdc = np.zeros_like(gradr)

    mlt_p = _turb.MltParams.cox(2.0)
    tdc_p = _turb.TdcParams(
        conv_vel_start=2.0e4,
        mixing_length_alpha=2.0,
        dt=15.0,
        cgrav=_const.STANDARD_CGRAV,
        m=_const.MSUN,
        scale=70.0 * _const.LSUN * 1.0e-3,
    )
    d1_unit = np.ones(1, dtype=np.float64)
    m_pre = (1.0, 1.0, cp_val, grav, 2.0 * hp_val, rho_val, p_val, t_val, 1.0)
    t_post = (
        _const.RSUN,
        p_val,
        t_val,
        rho_val,
        0.0,
        cp_val,
        1.0,
        hp_val,
        grad_l,
        grad_l,
        0.0,
        grav,
        1.5 * rgas * t_val,
    )
    for i, gr in enumerate(gradr):
        gr_d = _ad.DualOrder1(float(gr), d1_unit)
        m_out = _turb.set_mlt(
            mlt_p, _turb.MltInputs(*m_pre, gr_d, grad_l, grad_l)
        )
        assert isinstance(m_out.grad_t, _ad.DualOrder1)
        gt_mlt[i], dgt_mlt[i] = m_out.grad_t.val, float(m_out.grad_t.d1[0])

        t_out = _turb.set_tdc(tdc_p, _turb.TdcInputs(1.0, 1.0, gr_d, *t_post))
        assert isinstance(t_out.output.grad_t, _ad.DualOrder1)
        gt_tdc[i] = t_out.output.grad_t.val
        dgt_tdc[i] = float(t_out.output.grad_t.d1[0])

    fig, ax = _two_panel_fig()
    ax[0].axvline(grad_l, c="k", ls="--", lw=0.9, alpha=0.7)
    ax[0].plot(gradr, gt_mlt, color="C0", lw=1.9, label="Steady Cox MLT")
    ax[0].plot(
        gradr,
        gt_tdc,
        color="C1",
        lw=1.7,
        ls="-.",
        label=r"Transient TDC ($\Delta t = 15\,\mathrm{s}$)",
    )
    ax[0].set_xlim(0.39, 0.54)
    ax[0].set_xlabel(r"Radiative Gradient [$\nabla_r$]")
    ax[0].set_ylabel(r"Actual Temperature Gradient [$\nabla_T$]")
    ax[0].legend(frameon=False, loc="upper left")

    ax[1].axvline(grad_l, c="k", ls="--", lw=0.9, alpha=0.7)
    ax[1].plot(gradr, dgt_mlt, color="C0", lw=1.9, label="Steady Cox MLT")
    ax[1].plot(
        gradr, dgt_tdc, color="C1", lw=1.7, ls="-.", label="Transient TDC (IFT)"
    )
    ax[1].set_xlim(0.39, 0.54)
    ax[1].set_ylim(-0.02, 1.05)
    ax[1].set_xlabel(r"Radiative Gradient [$\nabla_r$]")
    ax[1].set_ylabel(
        r"Implicit Sensitivity [$\partial \nabla_T / \partial \nabla_r$]"
    )
    ax[1].legend(frameon=False, loc="upper right")
    _save_fig(fig, out_dir / "turb_mlt_tdc_comparison.png")


def plot_wave1_benchmark_staircase(out_dir: pathlib.Path) -> None:
    """Plots the 4-way Wave 1.5 runtime comparison at $N_z = 4096$.

    Args:
        out_dir: Target directory for `wave1_benchmark_staircase.png`.
    """
    workloads = [
        "auto_diff",
        "interp_2d",
        "neu",
        "mtx_tridiag",
        "turb_mlt",
        "turb_tdc",
    ]
    x_pos = np.arange(len(workloads), dtype=np.float64)
    w_bar = 0.20
    fortran_4t = np.array([5.65, 41.15, 152.32, 791.43, 1784.93, 58176.83])
    rust_4t = np.array([5.89, 36.95, 161.32, 1018.85, 142.14, 1075.24])
    pyo3_best = np.array([100.22, 142.37, 312.58, 1652.80, 290.02, 1205.71])
    jax_batch = np.array([321.12, 322.45, 319.25, 5356.85, 317.73, 398.61])

    fig, ax = plt.subplots(1, 1, figsize=(TEXT_WIDTH, 0.44 * TEXT_WIDTH))
    bars = (
        (-1.5, fortran_4t, "C2", 0.85, "Fortran MESA (4t OpenMP)"),
        (-0.5, rust_4t, "C0", 0.95, "Safe Rust (4t rayon)"),
        (0.5, pyo3_best, "C1", 0.90, "Python PyO3 (flexmesa._rust)"),
        (1.5, jax_batch, "C3", 0.85, "Python JAX (jit + vmap / 4)"),
    )
    for mult, arr, col, alp, lbl in bars:
        ax.bar(
            x_pos + mult * w_bar,
            np.log10(arr),
            width=w_bar,
            color=col,
            alpha=alp,
            label=lbl,
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(workloads)
    ax.set_ylim(0.0, 5.7)
    ax.set_xlabel(r"Wave 1 Workload ($N_{\mathrm{zones}} = 4096$)")
    ax.set_ylabel(r"Latency per Star [$\log_{10}(t / \mu\mathrm{s})$]")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    _save_fig(fig, out_dir / "wave1_benchmark_staircase.png")


if __name__ == "__main__":
    main()
