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

use crate::controls::{EnergyEqnOption, StarError, VarLayout};
use crate::eps_grav::eval_eps_grav;
use crate::hydro_vars::set_hydro_vars;
use crate::model::StarModel;
use crate::star_ad::{
    StarAdExt, StarAdOrder1, I_LND_00, I_LND_M1, I_LND_P1, I_LNR_00, I_LNR_M1,
    I_LNR_P1, I_LNT_00, I_LNT_M1, I_LNT_P1, I_L_00, I_L_M1, I_L_P1, I_V_00,
    I_V_M1, I_V_P1,
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
        if include_chem {
            for j in 0..model.species {
                let ivar = layout.chem_var_index(j);
                x_scale[base + ivar] =
                    model.xa_start[k * model.species + j].max(1.0e-4);
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
    let dr3 = (3.0 * model.dm[k] / FOUR_PI) / rho_00;
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
    let dp_0 = (model.cgrav[0] * model.m_grav[0] * model.dm[0]
        / (2.0 * FOUR_PI))
        / r_0.pow4();

    if !model.controls.use_momentum_outer_bc {
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

    let lnt_0 = StarAdOrder1::variable(model.ln_t[0], I_LNT_00);
    let t_0 = lnt_0.exp();
    let peos_0 = model.ln_peos_ad[0].exp();
    let dt_0 = dp_0 * model.gradt_ad[0] * (t_0 / peos_0);
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

/// Interpolates total EOS pressure $P_{\mathrm{eos}}$ to interior face `k >= 1`
/// with 3-cell stencil [`StarAdOrder1`] derivatives.
#[inline]
fn interp_peos_ad_to_face(model: &StarModel, k: usize) -> StarAdOrder1 {
    let (alfa, beta) = model.get_face_weights(k);
    let p_00 = model.ln_peos_ad[k].exp();
    let p_m1 = model.ln_peos_ad[k - 1].shift_m1().exp();
    p_00 * alfa + p_m1 * beta
}

/// Hydrostatic / hydrodynamic momentum equation at face `k` (`do1_momentum_eqn`
/// in `star/private/hydro_momentum.f90:79-436`).
///
/// Evaluates the pressure-gradient balance across face `k` normalized by the
/// mass-weighted face pressure $P_{\mathrm{face},k}$.
pub fn do1_momentum_eqn(model: &StarModel, k: usize, sys: &mut HenyeySystem) {
    if k == 0 && !model.controls.use_momentum_outer_bc {
        return;
    }
    let r_k = StarAdOrder1::variable(model.ln_r[k], I_LNR_00).exp();
    let r2 = r_k * r_k;
    let area = FOUR_PI * r2;
    let g_hse = -model.cgrav[k] * model.m_grav[k] / r2;
    let accel =
        if (model.controls.v_flag || model.controls.u_flag) && model.dt > 0.0 {
            let (v_curr, v_st) = if model.controls.v_flag {
                (model.v[k], model.v_start[k])
            } else {
                (model.u[k], model.u_start[k])
            };
            (StarAdOrder1::variable(v_curr, I_V_00) - v_st) / model.dt
        } else {
            StarAdOrder1::ZERO
        };
    let dp_hse = (g_hse - accel) * (model.dm_bar[k] / area);

    let resid = if k == 0 {
        let p_0 = model.ln_peos_ad[0].exp();
        let p_surf = model.ln_peos_surf_ad.exp();
        (dp_hse - (p_surf - p_0)) / p_0
    } else {
        let p_00 = model.ln_peos_ad[k].exp();
        let p_m1 = model.ln_peos_ad[k - 1].shift_m1().exp();
        let p_avg = interp_peos_ad_to_face(model, k);
        (dp_hse - (p_m1 - p_00)) / p_avg
    };
    sys.save_eqn_residual(&model.layout, k, model.layout.i_dv_dt, &resid);
}

/// Interior temperature transport equation at face `k >= 1`
/// (`do1_dlnT_dm_eqn` in `star/private/hydro_temperature.f90:240-310`).
///
/// Compares the theoretical temperature difference predicted by $\nabla_{T,k}$
/// against the finite-difference cell temperature contrast across face `k`.
pub fn do1_temperature_eqn(
    model: &StarModel,
    k: usize,
    sys: &mut HenyeySystem,
) {
    if k == 0 {
        return;
    }
    let (alfa, beta) = model.get_face_weights(k);
    let p_point = interp_peos_ad_to_face(model, k);

    let r_k = StarAdOrder1::variable(model.ln_r[k], I_LNR_00).exp();
    let dlnp_dm_qhse =
        -(model.cgrav[k] * model.m_grav[k]) / (FOUR_PI * r_k.pow4() * p_point);
    let dm_k = 0.5 * (model.dm[k - 1] + model.dm[k]);
    let dln_t_theory = dlnp_dm_qhse * model.gradt_ad[k] * dm_k;

    let t_00 = StarAdOrder1::variable(model.ln_t[k], I_LNT_00).exp();
    let t_m1 = StarAdOrder1::variable(model.ln_t[k - 1], I_LNT_M1).exp();
    let t_point = t_00 * alfa + t_m1 * beta;
    let dln_t_actual = (t_m1 - t_00) / t_point;

    let resid = dln_t_theory - dln_t_actual;
    sys.save_eqn_residual(&model.layout, k, model.layout.i_equ_l, &resid);
}

/// Energy conservation equation for cell `k` (`do1_energy_eqn` in
/// `star/private/hydro_energy.f90:36-905`).
pub fn do1_energy_eqn(
    model: &StarModel,
    k: usize,
    include_chem: bool,
    sys: &mut HenyeySystem,
) {
    let lum_00 = StarAdOrder1::variable(model.lum[k], I_L_00);
    let lum_p1 = if k + 1 < model.nz {
        StarAdOrder1::variable(model.lum[k + 1], I_L_P1)
    } else {
        StarAdOrder1::constant(model.l_center)
    };
    let dl_dm = (lum_00 - lum_p1) / model.dm[k];

    let eps_nuc_ad = StarAdOrder1::from_dlnd_dlnt_00(
        model.eps_nuc[k],
        model.d_epsnuc_dlnd[k],
        model.d_epsnuc_dlnt[k],
    );
    let non_nuc_neu_ad = StarAdOrder1::from_dlnd_dlnt_00(
        model.non_nuc_neu[k],
        model.d_non_nuc_neu_dlnd[k],
        model.d_non_nuc_neu_dlnt[k],
    );
    let sources = eps_nuc_ad - non_nuc_neu_ad + model.eps_mdot[k];
    let dt = model.dt.max(1.0e-30);

    let esum = match model.controls.energy_eqn_option {
        EnergyEqnOption::EpsGrav => -dl_dm + sources + model.eps_grav_ad[k],
        EnergyEqnOption::Dedt => {
            let e_ad = model.ln_e_ad[k].exp();
            let de_dt = (e_ad - model.energy_start[k]) / dt;
            let p_ad = model.ln_peos_ad[k].exp();
            let inv_rho_ad =
                (-StarAdOrder1::variable(model.lnd[k], I_LND_00)).exp();
            let inv_rho_start = (-model.lnd_start[k]).exp();
            let pdv_dt = p_ad * (inv_rho_ad - inv_rho_start) / dt;
            -dl_dm + sources - de_dt - pdv_dt
        }
    };

    let mut scal = dt / model.energy_start[k].max(1.0);
    if k == 0 {
        scal *= 1.0e-6;
    }
    let resid = esum * scal;
    let i_equ = model.layout.i_dlne_dt;
    sys.save_eqn_residual(&model.layout, k, i_equ, &resid);

    if include_chem {
        for j in 0..model.species {
            let d_eps_dx = model.d_epsnuc_dx[k * model.species + j];
            sys.add_partial(
                k,
                i_equ,
                k,
                model.layout.chem_var_index(j),
                scal * d_eps_dx,
            );
        }
    }
}

/// Optional kinematic radius-velocity equation for face `k` (`do1_radius_eqn`
/// in `star/private/hydro_momentum.f90:552-608`).
pub fn do1_radius_eqn(model: &StarModel, k: usize, sys: &mut HenyeySystem) {
    let Some(i_dlnr_dt) = model.layout.i_dlnr_dt else {
        return;
    };
    let vel_val = if model.controls.v_flag {
        model.v[k]
    } else {
        model.u[k]
    };
    let v_ad = StarAdOrder1::variable(vel_val, I_V_00);
    let lnr_ad = StarAdOrder1::variable(model.ln_r[k], I_LNR_00);
    let r_start = model.ln_r_start[k].exp().max(1.0);
    let resid = v_ad * (model.dt / r_start)
        - ((lnr_ad - model.ln_r_start[k]).exp() - 1.0);
    sys.save_eqn_residual(&model.layout, k, i_dlnr_dt, &resid);
}

/// Coupled nuclear burning and diffusive mixing equations for cell `k`
/// (`do1_chem_eqns` in `star/private/hydro_chem_eqns.f90:50-222`).
///
/// Combines start-of-step Lagrangian diffusive mixing $\sigma_k$ across faces
/// `k` and `k + 1` with the reaction-network abundance rates $\dot{X}_{\mathrm{nuc},j,k}$
/// and their partial derivatives with respect to $(\ln\rho_k, \ln T_k, X_{m,k})$.
pub fn do1_chem_eqns(model: &StarModel, k: usize, sys: &mut HenyeySystem) {
    let nz = model.nz;
    let species = model.species;
    let dt = model.dt.max(1.0e-30);
    let dm_k = model.dm[k].max(1.0e-30);
    let sig_00 = if k > 0 { model.sig[k] } else { 0.0 };
    let sig_p1 = if k + 1 < nz { model.sig[k + 1] } else { 0.0 };
    let sig_00_dm = sig_00 / dm_k;
    let sig_p1_dm = sig_p1 / dm_k;

    for j in 0..species {
        let kj = k * species + j;
        let i_var = model.layout.chem_var_index(j);
        let i_equ = i_var;
        let nuc_base = k * species * species + j * species;
        let d_self = model.d_dxdt_nuc_dx[nuc_base + j].abs();
        let x_scale_kj = sys.x_scale[k * sys.nvar + i_var];
        let rate_scale =
            (1.0 / dt + sig_00_dm + sig_p1_dm + d_self).max(1.0e-30);
        let eqn_scale = (x_scale_kj * rate_scale).max(1.0e-12);

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
        let dxdt_nuc = model.dxdt_nuc[kj];
        let dxdt_actual = (x_00 - x_start) / dt;
        sys.equ[k * sys.nvar + i_equ] =
            (dxdt_mix + dxdt_nuc - dxdt_actual) / eqn_scale;

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

        let d_diag = (-(sig_00_dm + sig_p1_dm) - 1.0 / dt) / eqn_scale;
        sys.add_partial(k, i_equ, k, i_var, d_diag);
        if k > 0 {
            sys.add_partial(k, i_equ, k - 1, i_var, sig_00_dm / eqn_scale);
        }
        if k + 1 < nz {
            sys.add_partial(k, i_equ, k + 1, i_var, sig_p1_dm / eqn_scale);
        }

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
    let skip_net = model.controls.op_split_burn && !include_chem;
    set_hydro_vars(model, skip_net, !include_chem)?;
    eval_eps_grav(model);

    let nz = model.nz;
    let nvar = model.layout.nvar(include_chem);
    let x_scale = compute_x_scale(model, include_chem);
    let mut sys = HenyeySystem::new(nz, nvar, x_scale);

    do_surf_bc_eqns(model, &mut sys);
    for k in 0..nz {
        do1_density_eqn(model, k, &mut sys)?;
        do1_momentum_eqn(model, k, &mut sys);
        do1_temperature_eqn(model, k, &mut sys);
        do1_energy_eqn(model, k, include_chem, &mut sys);
        do1_radius_eqn(model, k, &mut sys);
        if include_chem {
            do1_chem_eqns(model, k, &mut sys);
        }
    }
    Ok(sys)
}
