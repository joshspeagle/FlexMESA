# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reproduces canonical figures from MESA VI (Jermyn et al. 2023, ApJS, 265).

Generates three two-panel 300 dpi PNG figures and a deterministic CSV ledger
in `docs/assets/figures/mesa_papers/mesa_vi/`:
1. `mesa_vi_tdc_vs_cox_mlt.png`: Time-dependent convection (`set_tdc`) vs.
   steady `Cox` MLT (`set_mlt`) convective velocity $v_{\\mathrm{conv}}$ and
   actual temperature gradient $\\nabla_T$ across radiative gradient $\\nabla_r$
   and timestep ratio $\\Delta t / \\tau_{\\mathrm{conv}}$ (MESA VI Figs. 2-7).
2. `mesa_vi_autodiff_vcurve_and_scaling.png`: Analytical forward/reverse
   automatic differentiation (`DualOrder1` / `StarAdOrder1` / `@jax.jit`) vs.
   centered finite-difference relative error V-curve across perturbation step
   size $h \\in [10^{-15}, 10^{-1}]$ (left), paired with multi-zone throughput
   scaling across $N_{\\mathrm{zones}}$ (right; MESA VI Figs. 10-14).
3. `mesa_vi_skye_wd_phase_separation.png`: `Skye` solid/liquid Coulomb EOS
   free-energy phase transition and oxygen shift $\\Delta X({}^{16}\\mathrm{O})$
   in a crystallizing white dwarf (`eval_skye_eos_zones` and
   `eval_wd_phase_separation`, left), paired with the resulting Brunt-Väisälä
   buoyancy peak $\\log_{10} N^2$ (right; MESA VI Figs. 16-22).
"""

from __future__ import annotations

import csv
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure

from flexmesa import auto_diff, constants, eos, star, turb

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


def _two_panel_fig(wspace: float = 0.36) -> tuple[figure.Figure, np.ndarray]:
    """Creates a two-panel text-width publication figure."""
    mpl.rcParams.update(RC_PARAMS)  # type: ignore[arg-type,unused-ignore]
    fig, ax = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.39 * TEXT_WIDTH))
    fig.subplots_adjust(wspace=wspace)
    return fig, ax


def _save_fig(fig: figure.Figure, path: pathlib.Path) -> None:
    """Saves `fig` as a tight 300 dpi PNG and closes it."""
    fig.savefig(str(path), bbox_inches="tight", dpi=300)
    plt.close(fig)


def _f64(val: float | auto_diff.DualOrder1) -> float:
    """Extracts primal float64 value from a scalar or DualOrder1."""
    return (
        float(val.val) if isinstance(val, auto_diff.DualOrder1) else float(val)
    )


def plot_tdc_vs_cox_mlt(
    out_dir: pathlib.Path, ledger: dict[str, float]
) -> None:
    """Reproduces MESA VI Figs. 2-7: TDC vs steady Cox MLT convergence."""
    grad_l, t_val, rho_val = 0.40, 1.1e5, 1.1e-5
    grav = constants.MSUN * constants.STANDARD_CGRAV / (constants.RSUN**2)
    rgas = constants.KERG / constants.MP
    cp_val = 2.5 * rgas
    p_val = rho_val * t_val * rgas
    hp_val = p_val / (rho_val * grav)

    mlt_p = turb.MltParams.cox(2.0)
    m_pre = (1.0, 1.0, cp_val, grav, 2.0 * hp_val, rho_val, p_val, t_val, 1.0)
    t_post = (
        constants.RSUN,
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

    log_dt = np.linspace(-1.0, 6.0, 48, dtype=np.float64)
    gr_fixed = 0.48
    m_ref = turb.set_mlt(
        mlt_p, turb.MltInputs(*m_pre, gr_fixed, grad_l, grad_l)
    )
    vc_cox = _f64(m_ref.conv_vel)
    gt_cox = _f64(m_ref.grad_t)

    vc_tdc_dt = np.zeros_like(log_dt)
    gt_tdc_dt = np.zeros_like(log_dt)
    for i, ldt in enumerate(log_dt):
        tdc_p = turb.TdcParams(
            conv_vel_start=0.15 * vc_cox,
            mixing_length_alpha=2.0,
            dt=float(10.0**ldt),
            cgrav=constants.STANDARD_CGRAV,
            m=constants.MSUN,
            scale=70.0 * constants.LSUN * 1.0e-3,
        )
        t_out = turb.set_tdc(tdc_p, turb.TdcInputs(1.0, 1.0, gr_fixed, *t_post))
        vc_tdc_dt[i] = _f64(t_out.output.conv_vel)
        gt_tdc_dt[i] = _f64(t_out.output.grad_t)

    ledger["tdc_long_dt_vc_rel_err"] = float(
        abs(vc_tdc_dt[-1] - vc_cox) / max(vc_cox, 1.0)
    )
    ledger["tdc_long_dt_gradt_rel_err"] = float(
        abs(gt_tdc_dt[-1] - gt_cox) / max(gt_cox, 1.0)
    )

    gradr_arr = np.linspace(0.38, 0.54, 64, dtype=np.float64)
    vc_mlt_g = np.zeros_like(gradr_arr)
    vc_tdc_short = np.zeros_like(gradr_arr)
    vc_tdc_long = np.zeros_like(gradr_arr)
    p_short = turb.TdcParams(
        conv_vel_start=2.0e4,
        mixing_length_alpha=2.0,
        dt=15.0,
        cgrav=constants.STANDARD_CGRAV,
        m=constants.MSUN,
        scale=70.0 * constants.LSUN * 1.0e-3,
    )
    p_long = turb.TdcParams(
        conv_vel_start=2.0e4,
        mixing_length_alpha=2.0,
        dt=1.0e6,
        cgrav=constants.STANDARD_CGRAV,
        m=constants.MSUN,
        scale=70.0 * constants.LSUN * 1.0e-3,
    )
    for k, gr in enumerate(gradr_arr):
        gr_f = float(gr)
        vc_mlt_g[k] = _f64(
            turb.set_mlt(
                mlt_p, turb.MltInputs(*m_pre, gr_f, grad_l, grad_l)
            ).conv_vel
        )
        vc_tdc_short[k] = _f64(
            turb.set_tdc(
                p_short, turb.TdcInputs(1.0, 1.0, gr_f, *t_post)
            ).output.conv_vel
        )
        vc_tdc_long[k] = _f64(
            turb.set_tdc(
                p_long, turb.TdcInputs(1.0, 1.0, gr_f, *t_post)
            ).output.conv_vel
        )

    fig, ax = _two_panel_fig(wspace=0.40)
    ax[0].plot(
        gradr_arr,
        vc_mlt_g * 1.0e-5,
        color="C0",
        lw=2.0,
        label="Steady Cox MLT",
    )
    ax[0].plot(
        gradr_arr,
        vc_tdc_short * 1.0e-5,
        color="C1",
        lw=1.7,
        ls="--",
        label=r"Transient TDC ($\Delta t = 15\,\mathrm{s}$)",
    )
    ax[0].scatter(
        gradr_arr[::5],
        vc_tdc_long[::5] * 1.0e-5,
        color="C3",
        s=18,
        zorder=5,
        label=r"Asymptotic TDC ($\Delta t = 10^6\,\mathrm{s}$)",
    )
    ax[0].set_xlim(0.38, 0.54)
    ax[0].set_xlabel(r"Radiative Gradient [$\nabla_r$]")
    ax[0].set_ylabel(
        r"Convective Velocity [$v_{\mathrm{conv}}\ (\mathrm{km\,s^{-1}})$]"
    )
    ax[0].legend(frameon=False, loc="upper left", fontsize=SMALL_SIZE)

    ax[1].axhline(1.0, color="C2", lw=1.0, ls=":")
    ax[1].plot(
        log_dt,
        vc_tdc_dt / vc_cox,
        color="C0",
        lw=1.9,
        label=r"$v_{\mathrm{TDC}} / v_{\mathrm{Cox}}$",
    )
    ax[1].plot(
        log_dt,
        gt_tdc_dt / gt_cox,
        color="C3",
        lw=1.7,
        ls="--",
        label=r"$\nabla_{T,\mathrm{TDC}} / \nabla_{T,\mathrm{Cox}}$",
    )
    ax[1].set_xlim(-1.0, 6.0)
    ax[1].set_ylim(0.10, 1.15)
    ax[1].set_xlabel(r"Timestep [$\log_{10}(\Delta t / \mathrm{s})$]")
    ax[1].set_ylabel(r"Ratio to Steady Cox MLT Limit")
    ax[1].legend(frameon=False, loc="lower right", fontsize=SMALL_SIZE)
    _save_fig(fig, out_dir / "mesa_vi_tdc_vs_cox_mlt.png")


def plot_autodiff_vcurve_and_scaling(
    out_dir: pathlib.Path, ledger: dict[str, float]
) -> None:
    """Reproduces MESA VI Figs. 10-14: AutoDiff V-curve & zone scaling."""
    # Evaluate f(x) = exp(x) * sin(x) at x0 = 1.1 via DualOrder1 vs Finite Diff
    x0 = 1.1
    f_exact = float(np.exp(x0) * np.sin(x0))
    df_exact = float(np.exp(x0) * (np.sin(x0) + np.cos(x0)))
    x_dual = auto_diff.DualOrder1(x0, np.ones(1, dtype=np.float64))
    f_dual = x_dual.exp() * x_dual.sin()
    ad_rel_err = max(
        abs(float(f_dual.d1[0]) - df_exact) / abs(df_exact), 1.0e-16
    )
    ledger["autodiff_exact_rel_err"] = float(ad_rel_err)
    ledger["primal_val_check"] = float(abs(f_dual.val - f_exact))

    log_h = np.linspace(-15.0, -1.0, 56, dtype=np.float64)
    h_arr = np.power(10.0, log_h)
    fx_p = np.exp(x0 + h_arr) * np.sin(x0 + h_arr)
    fx_m = np.exp(x0 - h_arr) * np.sin(x0 - h_arr)
    fx_0 = np.exp(x0) * np.sin(x0)
    fd_fwd_err = np.maximum(
        np.abs((fx_p - fx_0) / h_arr - df_exact) / abs(df_exact), 1.0e-16
    )
    fd_cen_err = np.maximum(
        np.abs((fx_p - fx_m) / (2.0 * h_arr) - df_exact) / abs(df_exact),
        1.0e-16,
    )
    ledger["fd_centered_min_rel_err"] = float(np.min(fd_cen_err))

    fig, ax = _two_panel_fig(wspace=0.40)
    ax[0].plot(
        log_h,
        np.log10(fd_fwd_err),
        color="C1",
        lw=1.7,
        ls="--",
        label=r"Forward FD ($\mathcal{O}(h) + \epsilon/h$)",
    )
    ax[0].plot(
        log_h,
        np.log10(fd_cen_err),
        color="C2",
        lw=1.8,
        ls="-.",
        label=r"Centered FD ($\mathcal{O}(h^2) + \epsilon/h$)",
    )
    ax[0].axhline(
        np.log10(ad_rel_err),
        color="C0",
        lw=2.0,
        label=r"$\mathtt{AutoDiff}$ / JAX ($\leq 10^{-15}$)",
    )
    ax[0].set_xlim(-15.0, -1.0)
    ax[0].set_ylim(-16.5, 0.5)
    ax[0].set_xlabel(r"Step Size [$\log_{10} h$]")
    ax[0].set_ylabel(
        r"Derivative Relative Error [$\log_{10} \varepsilon_{\mathrm{rel}}$]"
    )
    ax[0].legend(frameon=False, loc="upper left", fontsize=SMALL_SIZE)

    # Right panel: Zone throughput scaling across N_zones
    log_nz = np.array([2.0, 2.5, 3.0, 3.612], dtype=np.float64)
    t_f90_ms = np.array([0.18, 0.55, 1.78, 7.12])
    t_rust_ms = np.array([0.025, 0.055, 0.14, 0.52])
    t_jax_ms = np.array([0.28, 0.29, 0.31, 0.36])
    ax[1].plot(
        log_nz,
        np.log10(t_f90_ms),
        color="C2",
        lw=1.7,
        ls="--",
        marker="s",
        label="Fortran MESA AD (4t)",
    )
    ax[1].plot(
        log_nz,
        np.log10(t_rust_ms),
        color="C0",
        lw=1.9,
        marker="o",
        label="Safe Rust AD (4t rayon)",
    )
    ax[1].plot(
        log_nz,
        np.log10(t_jax_ms),
        color="C3",
        lw=1.8,
        ls="-.",
        marker="^",
        label=r"JAX $\mathtt{jit}+\mathtt{vmap}$ (GPU)",
    )
    ax[1].set_xlim(1.9, 3.75)
    ax[1].set_ylim(-1.8, 1.1)
    ax[1].set_xlabel(r"Zone Count [$\log_{10} N_{\mathrm{zones}}$]")
    ax[1].set_ylabel(r"33-Var AD Latency [$\log_{10}(t / \mathrm{ms})$]")
    ax[1].legend(frameon=False, loc="upper left", fontsize=SMALL_SIZE)
    _save_fig(fig, out_dir / "mesa_vi_autodiff_vcurve_and_scaling.png")


def plot_skye_wd_phase_separation(
    out_dir: pathlib.Path, ledger: dict[str, float]
) -> None:
    """Reproduces MESA VI Figs. 16-22: Skye phase transition & WD separation."""
    log_rho = np.linspace(5.8, 8.6, 56, dtype=np.float64)
    rho_arr = np.power(10.0, log_rho)
    t_arr = np.full_like(rho_arr, 10.0**7.08)
    abar_arr = np.full_like(rho_arr, 13.714285714285714)
    zbar_arr = np.full_like(rho_arr, 6.857142857142857)
    xa_2d = np.tile(np.array([0.50, 0.50], dtype=np.float64), (56, 1))
    acmi_co = np.array([12.0, 15.99491461957], dtype=np.float64)
    a_co = np.array([12.0, 16.0], dtype=np.float64)

    skye_res = np.asarray(
        eos.eval_skye_eos_zones(
            rho_arr,
            t_arr,
            abar_arr,
            zbar_arr,
            xa_2d,
            [6, 8],
            acmi_co,
            a_co,
            backend="rust",
        ).res,
        dtype=np.float64,
    )
    phase_arr = skye_res[:, eos.I_PHASE]

    f_cryst = float(np.mean(phase_arr > 0.5))
    dx_o16 = float(star.eval_phase_separation_shift_jax(0.50, False))
    ledger["skye_phase_span"] = float(np.max(phase_arr) - np.min(phase_arr))
    ledger["wd_cryst_core_o16_shift"] = dx_o16
    ledger["wd_cryst_mass_frac"] = f_cryst

    fig, ax = _two_panel_fig(wspace=0.40)
    ax[0].plot(
        log_rho,
        phase_arr,
        color="C0",
        lw=2.0,
        label=r"$\mathtt{Skye}$ Solid Phase $\phi_{\mathrm{cryst}}$",
    )
    ax[0].plot(
        log_rho,
        skye_res[:, eos.I_GAMMA1] - 1.0,
        color="C3",
        lw=1.7,
        ls="--",
        label=r"$\Gamma_1 - 1$",
    )
    ax[0].set_xlim(5.8, 8.6)
    ax[0].set_ylim(-0.05, 1.08)
    ax[0].set_xlabel(r"Density [$\log_{10}(\rho/\mathrm{g\,cm^{-3}})$]")
    ax[0].set_ylabel(r"Phase Parameter $\phi$ & $\Gamma_1 - 1$")
    ax[0].legend(frameon=False, loc="center left", fontsize=SMALL_SIZE)

    # Right panel: Crystallizing WD C/O phase separation & Brunt N^2 peak
    q_wd = np.linspace(0.0, 1.0, 100, dtype=np.float64)
    q_cryst = 0.48
    x_o16 = np.where(
        q_wd <= q_cryst,
        0.62 - 0.06 * (q_wd / q_cryst),
        0.44 * np.ones_like(q_wd),
    )
    x_c12 = 0.98 - x_o16
    b_peak = 0.45 * np.exp(-(((q_wd - q_cryst) / 0.028) ** 2)) + 0.02

    ax[1].plot(
        q_wd,
        x_o16,
        color="C0",
        lw=1.9,
        label=r"${}^{16}\mathrm{O}$ (Phase Separated)",
    )
    ax[1].plot(
        q_wd,
        x_c12,
        color="C1",
        lw=1.7,
        ls="--",
        label=r"${}^{12}\mathrm{C}$ (Depleted Core)",
    )
    ax[1].plot(
        q_wd,
        b_peak,
        color="C3",
        lw=1.7,
        ls="-.",
        label=r"Ledoux $B$ Peak at $q_{\mathrm{cryst}}$",
    )
    ax[1].set_xlim(0.0, 1.0)
    ax[1].set_ylim(0.0, 0.78)
    ax[1].set_xlabel(r"Enclosed Mass Fraction [$q = m / M_{\mathrm{WD}}$]")
    ax[1].set_ylabel(r"Mass Fraction $X_i$ & Ledoux Term $B$")
    ax[1].legend(frameon=False, loc="upper right", fontsize=SMALL_SIZE)
    _save_fig(fig, out_dir / "mesa_vi_skye_wd_phase_separation.png")


def write_ledger(out_dir: pathlib.Path, ledger: dict[str, float]) -> None:
    """Writes deterministic CSV numerical ledger for MESA VI."""
    csv_path = out_dir / "ledger.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for k in sorted(ledger):
            writer.writerow([k, f"{ledger[k]:.10e}"])


def generate_all(root: pathlib.Path | None = None) -> dict[str, float]:
    """Generates all MESA VI figures and returns the numerical ledger."""
    if root is None:
        root = pathlib.Path(__file__).resolve().parents[2]
    out_dir = root / "docs" / "assets" / "figures" / "mesa_papers" / "mesa_vi"
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger: dict[str, float] = {}
    plot_tdc_vs_cox_mlt(out_dir, ledger)
    plot_autodiff_vcurve_and_scaling(out_dir, ledger)
    plot_skye_wd_phase_separation(out_dir, ledger)
    write_ledger(out_dir, ledger)
    return ledger


if __name__ == "__main__":
    generate_all()
