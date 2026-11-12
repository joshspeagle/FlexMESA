// SPDX-License-Identifier: LGPL-3.0-or-later
//! Coupled 1D Henyey stellar structure and composition residual equations and
//! analytical block-tridiagonal Jacobian assembly (`hydro_eqns.f90`,
//! `hydro_momentum.f90`, `hydro_temperature.f90`, `hydro_energy.f90`,
//! `hydro_chem_eqns.f90`, and `solver_support.f90`).
//!
//! Assembles the dimensionless residual vector `equ` of shape `(nz, nvar)` and
//! the scaled block-tridiagonal Jacobian `(lblk, dblk, ublk)` of shape
//! `(nz, nvar, nvar)` using 3-cell stencil forward-mode automatic
//! differentiation ([`StarAdOrder1`]).

use flexmesa_const::{FOUR_PI, L_SUN};

use crate::controls::{StarError, VarLayout};
use crate::eps_grav::eval_eps_grav;
pub use crate::hydro_energy::do1_energy_eqn;
pub use crate::hydro_momentum::{do1_momentum_eqn, do1_radius_eqn};
pub use crate::hydro_temperature::do1_temperature_eqn;
use crate::hydro_vars::set_hydro_vars;
use crate::model::StarModel;
use crate::star_ad::{
    StarAdExt, StarAdOrder1, I_HP_00, I_HP_M1, I_HP_P1, I_LND_00, I_LND_M1,
    I_LND_P1, I_LNR_00, I_LNR_M1, I_LNR_P1, I_LNT_00, I_LNT_M1, I_LNT_P1,
    I_L_00, I_L_M1, I_L_P1, I_V_00, I_V_M1, I_V_P1, I_W_00, I_W_M1, I_W_P1,
};

/// Assembled 1D Henyey residual vector and scaled block-tridiagonal Jacobian
/// $L_k \delta\tilde{x}_{k-1} + D_k \delta\tilde{x}_k
/// + U_k \delta\tilde{x}_{k+1} = -\mathrm{equ}_k$.
#[derive(Debug, Clone)]
pub struct HenyeySystem {
    /// Number of coupled variables per zone (`nvar_hydro` or `nvar_total`).
    pub nvar: usize,
    /// Number of Lagrangian zones (`nz`).
    pub nz: usize,
    /// Variable scaling factors `x_scale[k * nvar + j]` (`solver_support.f90:312`).
    pub x_scale: Vec<f64>,
    /// Residual vector `equ[k * nvar + i]` (`len = nz * nvar`).
    pub equ: Vec<f64>,
    /// Sub-diagonal Jacobian blocks `lblk[k * nvar * nvar + i * nvar + j]`.
    pub lblk: Vec<f64>,
    /// Diagonal Jacobian blocks `dblk[k * nvar * nvar + i * nvar + j]`.
    pub dblk: Vec<f64>,
    /// Super-diagonal Jacobian blocks `ublk[k * nvar * nvar + i * nvar + j]`.
    pub ublk: Vec<f64>,
}

impl HenyeySystem {
    /// Allocates a zero-initialized [`HenyeySystem`] for `(nz, nvar)` with
    /// `x_scale`.
    #[must_use]
    pub fn new(nz: usize, nvar: usize, x_scale: Vec<f64>) -> Self {
        let blk_len = nz * nvar * nvar;
        Self {
            nvar,
            nz,
            x_scale,
            equ: vec![0.0; nz * nvar],
            lblk: vec![0.0; blk_len],
            dblk: vec![0.0; blk_len],
            ublk: vec![0.0; blk_len],
        }
    }

    /// Accumulates a scalar partial derivative
    /// $\partial \mathrm{equ}_{k,\mathrm{equ}} / \partial x_{k_{\mathrm{var}}, j_{\mathrm{var}}}$
    /// scaled by `x_scale[k_var, j_var]` into `lblk`, `dblk`, or `ublk`.
    #[inline]
    pub fn add_partial(
        &mut self,
        k_equ: usize,
        i_equ: usize,
        k_var: usize,
        j_var: usize,
        deriv: f64,
    ) {
        if deriv == 0.0 {
            return;
        }
        let nvar = self.nvar;
        let blk_sz = nvar * nvar;
        let scaled = deriv * self.x_scale[k_var * nvar + j_var];
        let idx = k_equ * blk_sz + i_equ * nvar + j_var;
        if k_var == k_equ {
            self.dblk[idx] += scaled;
        } else if k_var + 1 == k_equ {
            self.lblk[idx] += scaled;
        } else if k_equ + 1 == k_var {
            self.ublk[idx] += scaled;
        }
    }

    /// Accumulates 3-cell stencil partials `(d_m1, d_00, d_p1)` for variable
    /// slot `j_var` at equation `(k, i_equ)`.
    #[inline]
    fn add_stencil_partials(
        &mut self,
        k: usize,
        i_equ: usize,
        j_var: usize,
        d_m1: f64,
        d_00: f64,
        d_p1: f64,
    ) {
        self.add_partial(k, i_equ, k, j_var, d_00);
        if k > 0 {
            self.add_partial(k, i_equ, k - 1, j_var, d_m1);
        }
        if k + 1 < self.nz {
            self.add_partial(k, i_equ, k + 1, j_var, d_p1);
        }
    }

    /// Stores `equ[k, i_equ] = resid.val` and unpacks the 3-cell stencil
    /// [`StarAdOrder1`] partials into `lblk`, `dblk`, and `ublk`
    /// (`save_eqn_residual_info` in `star/private/hydro_eqns.f90:1032-1190`).
    pub fn save_eqn_residual(
        &mut self,
        layout: &VarLayout,
        k: usize,
        i_equ: usize,
        resid: &StarAdOrder1,
    ) {
        self.equ[k * self.nvar + i_equ] = resid.val;
        let slots = [
            (layout.i_lnd, I_LND_M1, I_LND_00, I_LND_P1),
            (layout.i_lnt, I_LNT_M1, I_LNT_00, I_LNT_P1),
            (layout.i_lnr, I_LNR_M1, I_LNR_00, I_LNR_P1),
            (layout.i_lum, I_L_M1, I_L_00, I_L_P1),
        ];
        for (j_var, m1_slot, d0_slot, p1_slot) in slots {
            self.add_stencil_partials(
                k,
                i_equ,
                j_var,
                resid.d1[m1_slot],
                resid.d1[d0_slot],
                resid.d1[p1_slot],
            );
        }
        if let Some(iv) = layout.i_v.or(layout.i_u) {
            self.add_stencil_partials(
                k,
                i_equ,
                iv,
                resid.d1[I_V_M1],
                resid.d1[I_V_00],
                resid.d1[I_V_P1],
            );
        }
        if let Some(iw) = layout.i_w {
            self.add_stencil_partials(
                k,
                i_equ,
                iw,
                resid.d1[I_W_M1],
                resid.d1[I_W_00],
                resid.d1[I_W_P1],
            );
        }
        if let Some(ihp) = layout.i_hp {
            self.add_stencil_partials(
                k,
                i_equ,
                ihp,
                resid.d1[I_HP_M1],
                resid.d1[I_HP_00],
                resid.d1[I_HP_P1],
            );
        }
    }
}

/// Computes the per-zone variable scaling vector `x_scale` (`set_xscale_info`
/// in `star/private/solver_support.f90:312-363`).
#[must_use]
pub fn compute_x_scale(model: &StarModel, include_chem: bool) -> Vec<f64> {
    let nz = model.nz;
    let layout = &model.layout;
    let nvar = layout.nvar(include_chem);
    let mut x_scale = vec![1.0; nz * nvar];

    let max_abs_l = model
        .lum_start
        .iter()
        .fold(0.0_f64, |acc, &x| acc.max(x.abs()));
    let l_scale_max = L_SUN.max(1.0e-3 * max_abs_l);

    for k in 0..nz {
        let base = k * nvar;
        x_scale[base + layout.i_lnd] = model.lnd_start[k].abs().max(1.0);
        x_scale[base + layout.i_lnt] = model.ln_t_start[k].abs().max(1.0);
        x_scale[base + layout.i_lnr] = model.ln_r_start[k].abs().max(1.0);
        x_scale[base + layout.i_lum] = l_scale_max
            .max(max_abs_l * model.q[k].min(1.0))
            .max(model.lum_start[k].abs());
        if let Some(iv) = layout.i_v.or(layout.i_u) {
            let r_over_dt =
                model.ln_r_start[k].exp().max(1.0) / model.dt.max(1.0e-30);
            x_scale[base + iv] =
                r_over_dt.min(model.csound[k].max(1.0)).max(1.0e-30);
        }
        if let Some(iw) = layout.i_w {
            let w_s =
                model.aux.rsp.w_start.get(k).copied().unwrap_or(0.0).abs();
            x_scale[base + iw] = w_s.max(1.0e3);
        }
        if let Some(ihp) = layout.i_hp {
            let hp = model.aux.rsp.hp_face.get(k).copied().unwrap_or(1.0).abs();
            x_scale[base + ihp] = hp.max(1.0);
        }
        if let Some(ia) = layout.i_alpha_rti {
            let a_s = model
                .aux
                .alpha_rti_start
                .get(k)
                .copied()
                .unwrap_or(model.aux.alpha_rti[k])
                .abs();
            x_scale[base + ia] = a_s.max(1.0);
        }
        if include_chem {
            let xa_min = model.controls.xa_scale.max(1.0e-30);
            for j in 0..model.species {
                let ivar = layout.chem_var_index(j);
                x_scale[base + ivar] =
                    model.xa_start[k * model.species + j].max(xa_min);
            }
        }
    }
    x_scale
}

/// Continuity / volume equation for cell `k` (`do1_density_eqn` in
/// `star/private/hydro_eqns.f90:550-641`):
/// $$\mathrm{equ}(\mathrm{i\_dlnd\_dt}, k) =
/// \ln R_k - \frac{1}{3}\ln\!\left(r_{k+1}^3 + \frac{3\,\Delta m_k}{4\pi \rho_k}\right).$$
///
/// # Errors
/// Returns [`StarError::ConvergenceFailure`] if the trial volume is
/// non-positive.
pub fn do1_density_eqn(
    model: &StarModel,
    k: usize,
    sys: &mut HenyeySystem,
) -> Result<(), StarError> {
    let lnr_00 = StarAdOrder1::variable(model.ln_r[k], I_LNR_00);
    let r_p1_cubed = if k + 1 < model.nz {
        StarAdOrder1::variable(model.ln_r[k + 1], I_LNR_P1)
            .exp()
            .powi(3)
    } else {
        StarAdOrder1::constant(model.r_center.powi(3))
    };
    let rho_00 = StarAdOrder1::variable(model.lnd[k], I_LND_00).exp();
    let mc = if model.controls.use_mass_corrections {
        model.aux.mass_correction.get(k).copied().unwrap_or(1.0)
    } else {
        1.0
    };
    let dr3 = (3.0 * model.dm[k] * mc / FOUR_PI) / rho_00;
    let rp13_p_dr3 = r_p1_cubed + dr3;
    if rp13_p_dr3.val <= 0.0 || !rp13_p_dr3.val.is_finite() {
        return Err(StarError::ConvergenceFailure(format!(
            "non-positive inner volume at zone {k}: {}",
            rp13_p_dr3.val
        )));
    }
    let resid = lnr_00 - rp13_p_dr3.ln() / 3.0;
    sys.save_eqn_residual(&model.layout, k, model.layout.i_dlnd_dt, &resid);
    Ok(())
}

/// Surface pressure and temperature boundary condition equations at `k = 0`
/// (`set_Psurf_BC` and `set_Tsurf_BC` in `star/private/hydro_eqns.f90:857-1029`).
pub fn do_surf_bc_eqns(model: &StarModel, sys: &mut HenyeySystem) {
    if model.nz == 0 {
        return;
    }
    let r_0 = StarAdOrder1::variable(model.ln_r[0], I_LNR_00).exp();
    let fp_0 = if model.controls.rotation_flag
        && model.controls.use_gravity_rotation_correction
    {
        model.aux.fp_rot.first().copied().unwrap_or(1.0)
    } else {
        1.0
    };
    let mc_0 = if model.controls.use_mass_corrections {
        model.aux.mass_correction.first().copied().unwrap_or(1.0)
    } else {
        1.0
    };
    let dp_0 = if model.controls.use_momentum_outer_bc {
        StarAdOrder1::ZERO
    } else {
        (model.cgrav[0] * model.m_grav[0] * fp_0 * model.dm[0] * mc_0
            / (2.0 * FOUR_PI))
            / r_0.pow4()
    };

    let use_surf_momentum_bc = model.controls.use_momentum_outer_bc
        || model.controls.use_zero_pgas_outer_bc
        || model.controls.use_fixed_psurf_outer_bc;
    if model.controls.use_compression_outer_bc && model.nz >= 2 {
        let lnd_0 = StarAdOrder1::variable(model.lnd[0], I_LND_00);
        let lnd_1 = StarAdOrder1::variable(model.lnd[1], I_LND_P1);
        let rho_0 = lnd_0.exp();
        let rho_1 = lnd_1.exp();
        let dlnd_0 =
            lnd_0 - model.lnd_start.first().copied().unwrap_or(model.lnd[0]);
        let dlnd_1 =
            lnd_1 - model.lnd_start.get(1).copied().unwrap_or(model.lnd[1]);
        let resid_c = (rho_1 * dlnd_0 - rho_0 * dlnd_1) / model.dt.max(1.0e-30);
        sys.save_eqn_residual(&model.layout, 0, model.layout.i_dv_dt, &resid_c);
    } else if model.controls.use_fixed_vsurf_outer_bc
        && (model.controls.v_flag || model.controls.u_flag)
    {
        let vsurf_val = if model.controls.v_flag {
            model.v[0]
        } else {
            model.u[0]
        };
        let vsurf = StarAdOrder1::variable(vsurf_val, I_V_00);
        let cs_0 = model
            .aux
            .csound_start
            .first()
            .copied()
            .filter(|&c| c > 0.0)
            .unwrap_or(model.csound[0])
            .max(1.0);
        let resid_v = (vsurf - model.controls.fixed_vsurf) / cs_0;
        sys.save_eqn_residual(&model.layout, 0, model.layout.i_dv_dt, &resid_v);
    } else if !use_surf_momentum_bc {
        let p_surf = model.ln_peos_surf_ad.exp();
        let ln_p_bc = (p_surf + dp_0).max(StarAdOrder1::constant(1.0e-30)).ln();
        let ln_peos_0 = model.ln_peos_ad[0];
        let denom_p = if ln_peos_0.val > 1.0 {
            ln_peos_0
        } else {
            StarAdOrder1::ONE
        };
        let resid_p = (ln_p_bc - ln_peos_0) / denom_p;
        sys.save_eqn_residual(&model.layout, 0, model.layout.i_dv_dt, &resid_p);
    }

    let skip_surf_t_bc = model.controls.use_rsp_l_eqn_outer_bc
        || (model.controls.rsp2_flag
            && model.controls.rsp2.use_l_eqn_at_surface);
    if !skip_surf_t_bc {
        let lnt_0 = StarAdOrder1::variable(model.ln_t[0], I_LNT_00);
        let t_0 = lnt_0.exp();
        let peos_0 = model.ln_peos_ad[0].exp();
        let dt_0 = if model.controls.rsp2_flag {
            StarAdOrder1::ZERO
        } else {
            dp_0 * model.gradt_ad[0] * (t_0 / peos_0)
        };
        let t_surf = model.ln_t_surf_ad.exp();
        let ln_t_bc = (t_surf + dt_0).max(StarAdOrder1::constant(1.0)).ln();
        let denom_t = if lnt_0.val > 1.0 {
            lnt_0
        } else {
            StarAdOrder1::ONE
        };
        let resid_t = (ln_t_bc - lnt_0) / denom_t;
        sys.save_eqn_residual(&model.layout, 0, model.layout.i_equ_l, &resid_t);
    }
}

/// Coupled nuclear burning and diffusive mixing equations for cell `k`
/// (`do1_chem_eqns` in `star/private/hydro_chem_eqns.f90:50-222`).
///
/// Combines start-of-step Lagrangian diffusive mixing $\sigma_k$ across faces
/// `k` and `k + 1` (gated by `do_mix` and `mix_factor`) with the
/// reaction-network abundance rates $\dot{X}_{\mathrm{nuc},j,k}$ and their
/// partial derivatives with respect to $(\ln\rho_k, \ln T_k, X_{m,k})$ (gated
/// by `do_burn`, `dxdt_nuc_factor`, and `!doing_op_split_burn`).
pub fn do1_chem_eqns(model: &StarModel, k: usize, sys: &mut HenyeySystem) {
    let nz = model.nz;
    let species = model.species;
    let dt = model.dt.max(1.0e-30);
    let dm_k = model.dm[k].max(1.0e-30);

    let t_start_k = model
        .ln_t_start
        .get(k)
        .copied()
        .unwrap_or(model.ln_t[k])
        .exp();
    let doing_op_split_burn = model.controls.op_split_burn
        && t_start_k >= model.controls.op_split_burn_min_t;
    let active_burn = model.controls.do_burn
        && model.controls.dxdt_nuc_factor != 0.0
        && !doing_op_split_burn;

    let mix_fac = if model.controls.do_mix {
        model.controls.mix_factor
    } else {
        0.0
    };
    let sig_00 = if k > 0 { model.sig[k] * mix_fac } else { 0.0 };
    let sig_p1 = if k + 1 < nz {
        model.sig[k + 1] * mix_fac
    } else {
        0.0
    };
    let sig_00_dm = sig_00 / dm_k;
    let sig_p1_dm = sig_p1 / dm_k;

    for j in 0..species {
        let kj = k * species + j;
        let i_var = model.layout.chem_var_index(j);
        let i_equ = i_var;
        let nuc_base = k * species * species + j * species;
        let x_scale_kj = sys.x_scale[k * sys.nvar + i_var];
        let eqn_scale = model
            .controls
            .min_chem_eqn_scale
            .max(x_scale_kj / dt)
            .max(1.0e-30);

        let x_00 = model.xa[kj];
        let x_start = model.xa_start[kj];
        let f_00 = if k > 0 {
            sig_00 * (model.xa[(k - 1) * species + j] - x_00)
        } else {
            0.0
        };
        let f_p1 = if k + 1 < nz {
            sig_p1 * (x_00 - model.xa[(k + 1) * species + j])
        } else {
            0.0
        };
        let dxdt_mix = (f_00 - f_p1) / dm_k;
        let dxdt_nuc = if active_burn { model.dxdt_nuc[kj] } else { 0.0 };
        let dxdt_actual = (x_00 - x_start) / dt;
        sys.equ[k * sys.nvar + i_equ] =
            (dxdt_mix + dxdt_nuc - dxdt_actual) / eqn_scale;

        if active_burn {
            sys.add_partial(
                k,
                i_equ,
                k,
                model.layout.i_lnd,
                model.d_dxdt_nuc_dlnd[kj] / eqn_scale,
            );
            sys.add_partial(
                k,
                i_equ,
                k,
                model.layout.i_lnt,
                model.d_dxdt_nuc_dlnt[kj] / eqn_scale,
            );
        }

        let d_diag = (-(sig_00_dm + sig_p1_dm) - 1.0 / dt) / eqn_scale;
        sys.add_partial(k, i_equ, k, i_var, d_diag);
        if k > 0 && sig_00_dm != 0.0 {
            sys.add_partial(k, i_equ, k - 1, i_var, sig_00_dm / eqn_scale);
        }
        if k + 1 < nz && sig_p1_dm != 0.0 {
            sys.add_partial(k, i_equ, k + 1, i_var, sig_p1_dm / eqn_scale);
        }

        if active_burn {
            for m in 0..species {
                let d_nuc = model.d_dxdt_nuc_dx[nuc_base + m];
                if d_nuc != 0.0 {
                    sys.add_partial(
                        k,
                        i_equ,
                        k,
                        model.layout.chem_var_index(m),
                        d_nuc / eqn_scale,
                    );
                }
            }
        }
    }
}

/// Assembles the coupled 1D Henyey residual vector and scaled block-tridiagonal
/// Jacobian blocks into `sys` from the current microphysical and [`StarAdOrder1`]
/// fields on `model` without re-running `set_hydro_vars` or `eval_eps_grav`.
///
/// # Errors
/// Returns [`StarError`] if the continuity volume equation encounters a
/// non-positive trial shell volume.
pub fn assemble_henyey_system(
    model: &mut StarModel,
    include_chem: bool,
    sys: &mut HenyeySystem,
) -> Result<(), StarError> {
    let nz = model.nz;
    let nvar = model.layout.nvar(include_chem);
    model.aux.total_internal_energy_start = (0..nz)
        .map(|k| model.energy_start[k].max(0.0) * model.dm[k])
        .sum();
    sys.nvar = nvar;
    sys.nz = nz;
    sys.x_scale = compute_x_scale(model, include_chem);
    sys.equ.clear();
    sys.equ.resize(nz * nvar, 0.0);
    let blk_len = nz * nvar * nvar;
    sys.lblk.clear();
    sys.lblk.resize(blk_len, 0.0);
    sys.dblk.clear();
    sys.dblk.resize(blk_len, 0.0);
    sys.ublk.clear();
    sys.ublk.resize(blk_len, 0.0);

    do_surf_bc_eqns(model, sys);
    let riemann_p_surf = if model.controls.u_flag
        && !model.controls.v_flag
        && !model.controls.use_compression_outer_bc
        && !model.controls.use_fixed_vsurf_outer_bc
    {
        if model.controls.use_zero_pgas_outer_bc {
            let t_s0 = model
                .ln_t_start
                .first()
                .copied()
                .unwrap_or_else(|| model.ln_t.first().copied().unwrap_or(0.0))
                .exp();
            Some(StarAdOrder1::constant(
                (flexmesa_const::CRAD / 3.0) * t_s0.powi(4),
            ))
        } else if model.controls.use_fixed_psurf_outer_bc {
            Some(StarAdOrder1::constant(model.controls.fixed_psurf))
        } else if model.controls.use_momentum_outer_bc {
            Some(model.ln_peos_surf_ad.exp())
        } else {
            None
        }
    } else {
        None
    };

    for k in 0..nz {
        do1_density_eqn(model, k, sys)?;
        if model.controls.u_flag && !model.controls.v_flag {
            crate::riemann_ad::do1_riemann_momentum_eqn(
                model,
                k,
                if k == 0 { riemann_p_surf } else { None },
                sys,
            );
        } else {
            do1_momentum_eqn(model, k, sys);
        }
        if !model.controls.rsp2_flag {
            do1_temperature_eqn(model, k, sys);
        }
        do1_energy_eqn(model, k, include_chem, sys);
        do1_radius_eqn(model, k, sys);
        if model.controls.rti_flag {
            crate::hydro_rti::do1_dalpha_rti_dt_eqn(model, k, sys);
        }
        if include_chem {
            do1_chem_eqns(model, k, sys);
        }
    }
    if model.controls.rsp2_flag {
        crate::rsp2_ad::eval_and_save_rsp2_eqns(model, sys);
    }
    Ok(())
}

/// Evaluates microphysics (`set_hydro_vars`), gravitational energy generation
/// (`eval_eps_grav`), and all coupled 1D Henyey residual equations and scaled
/// block-tridiagonal Jacobian blocks (`eval_equations` in
/// `star/private/hydro_eqns.f90:91-265`).
///
/// # Errors
/// Returns [`StarError`] if microphysics or volume equation evaluation fails.
pub fn eval_henyey_equations(
    model: &mut StarModel,
    include_chem: bool,
) -> Result<HenyeySystem, StarError> {
    set_hydro_vars(model, false, !include_chem)?;
    eval_eps_grav(model);
    let mut sys = HenyeySystem::new(0, 0, Vec::new());
    assemble_henyey_system(model, include_chem, &mut sys)?;
    Ok(sys)
}
