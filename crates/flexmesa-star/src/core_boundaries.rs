// SPDX-License-Identifier: LGPL-3.0-or-later
//! Sub-cell core-boundary mass and boundary-property interpolation
//! (`get_core_info`, `set_core_info`, `get_info_at_q`) and nuclear burning
//! category luminosity summation (`star/private/report.f90:76-113` and
//! `1029-1386`, `star/private/star_utils.f90:309-325`).

use flexmesa_chem::{NuclideCatalog, NUM_CATEGORIES};
use flexmesa_const::{LN10, L_SUN, M_SUN, R_SUN};

use crate::model::StarModel;

/// Category index for proton-proton burning (`"pp"`, `CATEGORY_NAME[0]`).
const CAT_PP: usize = 0;
/// Category index for CNO cycle burning (`"cno"`, `CATEGORY_NAME[1]`).
const CAT_CNO: usize = 1;
/// Category index for triple-alpha He burning (`"tri_alpha"`, `CATEGORY_NAME[2]`).
const CAT_3ALF: usize = 2;
/// Category index for photodisintegration (`"photo"`, `CATEGORY_NAME[19]`).
const CAT_PHOTO: usize = 19;

/// Sub-cell interpolated thermodynamic and kinematic properties at a single
/// composition core boundary (`set_core_info` and `get_info_at_q` in
/// `star/private/report.f90:1212-1386`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CoreBoundaryInfo {
    /// 1-based boundary zone index (`kbdy` in `report.f90:1286-1322`; `0` when absent).
    pub kbdy_1based: usize,
    /// Enclosed core mass in $M_\odot$ (`bdy_m`).
    pub mass_msun: f64,
    /// Interpolated core radius in $R_\odot$ (`bdy_r`).
    pub radius_rsun: f64,
    /// Interpolated $\log_{10}(T/\mathrm{K})$ (`bdy_lgT`).
    pub lg_t: f64,
    /// Interpolated $\log_{10}(\rho/\mathrm{g\,cm^{-3}})$ (`bdy_lgRho`).
    pub lg_rho: f64,
    /// Interpolated luminosity in $L_\odot$ (`bdy_L`).
    pub lum_lsun: f64,
    /// Interpolated radial velocity in $\mathrm{cm\,s^{-1}}$ (`bdy_v`).
    pub vel_cms: f64,
    /// Interpolated angular velocity in $\mathrm{rad\,s^{-1}}$ (`bdy_omega`).
    pub omega: f64,
    /// Interpolated $\Omega / \Omega_{\mathrm{crit}}$ (`bdy_omega_div_omega_crit`).
    pub omega_div_omega_crit: f64,
}

/// Sub-cell interpolated core boundary masses ($M_\odot$), outermost core zone
/// indices (`0`-based `kbdy - 1`, `None` when absent), and full boundary
/// interpolants matching `star/private/report.f90:1029-1386`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CoreMasses {
    /// Helium core mass in $M_\odot$ (`he_core_mass`).
    pub he_core_mass: f64,
    /// Outermost 0-based zone index (`kbdy - 1`) inside the He core (`he_core_k`).
    pub he_core_k: Option<usize>,
    /// Full sub-cell boundary properties for the He core.
    pub he_bdy: CoreBoundaryInfo,
    /// Carbon-oxygen core mass in $M_\odot$ (`co_core_mass`).
    pub co_core_mass: f64,
    /// Outermost 0-based zone index (`kbdy - 1`) inside the CO core (`co_core_k`).
    pub co_core_k: Option<usize>,
    /// Full sub-cell boundary properties for the CO core.
    pub co_bdy: CoreBoundaryInfo,
    /// Oxygen-neon core mass in $M_\odot$ (`one_core_mass`).
    pub one_core_mass: f64,
    /// Outermost 0-based zone index (`kbdy - 1`) inside the ONe core (`one_core_k`).
    pub one_core_k: Option<usize>,
    /// Full sub-cell boundary properties for the ONe core.
    pub one_bdy: CoreBoundaryInfo,
    /// Iron-group core mass in $M_\odot$ (`fe_core_mass`).
    pub fe_core_mass: f64,
    /// Outermost 0-based zone index (`kbdy - 1`) inside the Fe core (`fe_core_k`).
    pub fe_core_k: Option<usize>,
    /// Full sub-cell boundary properties for the Fe core.
    pub fe_bdy: CoreBoundaryInfo,
    /// Neutron-rich core mass in $M_\odot$ (`neutron_rich_core_mass`).
    pub neutron_rich_core_mass: f64,
    /// Outermost 0-based zone index (`kbdy - 1`) inside the neutron-rich core (`neutron_rich_core_k`).
    pub neutron_rich_core_k: Option<usize>,
    /// Full sub-cell boundary properties for the neutron-rich core.
    pub neutron_rich_bdy: CoreBoundaryInfo,
}

/// Integrated nuclear and neutrino luminosities in solar luminosities ($L_\odot$)
/// matching `star/private/report.f90:76-113`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerCategories {
    /// Total nuclear energy generation rate in $L_\odot$ (`power_nuc_burn`).
    pub power_nuc_burn: f64,
    /// Hydrogen-burning luminosity (`pp + cno`) in $L_\odot$ (`power_h_burn`).
    pub power_h_burn: f64,
    /// Helium-burning luminosity (`tri_alpha`) in $L_\odot$ (`power_he_burn`).
    pub power_he_burn: f64,
    /// Metal-burning luminosity (`power_nuc_burn - (power_h_burn + power_he_burn)`) in $L_\odot$ (`power_z_burn`).
    pub power_z_burn: f64,
    /// Photodisintegration luminosity in $L_\odot$ (`power_photo`).
    pub power_photo: f64,
    /// Nuclear reaction neutrino luminosity in $L_\odot$ (`power_nuc_neutrinos`).
    pub power_nuc_neutrinos: f64,
    /// Thermal (non-nuclear) neutrino luminosity in $L_\odot$ (`power_non_nuc_neu`).
    pub power_non_nuc_neu: f64,
    /// Total neutrino luminosity in $L_\odot$ (`power_neutrinos`).
    pub power_neutrinos: f64,
    /// Per-category integrated nuclear luminosity in $L_\odot$ (`L_by_category`, `report.f90:96`).
    pub l_by_category_lsun: [f64; NUM_CATEGORIES],
}

impl Default for PowerCategories {
    fn default() -> Self {
        Self {
            power_nuc_burn: 0.0,
            power_h_burn: 0.0,
            power_he_burn: 0.0,
            power_z_burn: 0.0,
            power_photo: 0.0,
            power_nuc_neutrinos: 0.0,
            power_non_nuc_neu: 0.0,
            power_neutrinos: 0.0,
            l_by_category_lsun: [0.0; NUM_CATEGORIES],
        }
    }
}

#[inline]
fn find0_or_mid(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    flexmesa_num::find0(x1, y1, x2, y2).unwrap_or(0.5 * (x1 + x2))
}

pub(crate) fn dominant_species_index(row: &[f64]) -> usize {
    let mut best_j = 0usize;
    let mut best_x = f64::NEG_INFINITY;
    for (j, &x) in row.iter().enumerate() {
        if x > best_x {
            best_x = x;
            best_j = j;
        }
    }
    best_j
}

/// 1-based zone index `k` satisfying `q(k) >= bdy_q > q(k) - dq(k)`
/// (`k_for_q` in `star/private/star_utils.f90:309-325`).
#[must_use]
pub fn k_for_q_1based(model: &StarModel, q: f64) -> usize {
    let nz = model.nz;
    if nz == 0 || q >= 1.0 {
        return 1;
    }
    if q <= model.q[nz - 1] {
        return nz;
    }
    for i in 0..(nz - 1) {
        if q > model.q[i + 1] {
            return i + 1;
        }
    }
    nz
}

/// Sub-cell core mass fraction $q_{\mathrm{core}}$ and interpolated boundary
/// properties from dominant-species crossover between cell centers `k - 1` and
/// `k` (`set_core_info` + `get_info_at_q` in `star/private/report.f90:1212-1386`).
fn eval_set_core_info(model: &StarModel, k: usize) -> CoreBoundaryInfo {
    let core_q = if k == 0 || model.species == 0 {
        1.0
    } else {
        let row_m1 = model.cell_xa(k - 1);
        let row_00 = model.cell_xa(k);
        let jm1 = dominant_species_index(row_m1);
        let j00 = dominant_species_index(row_00);
        let qm1 = model.q[k - 1] - 0.5 * model.dq[k - 1];
        let q00 = model.q[k] - 0.5 * model.dq[k];
        let dm1 = row_m1[j00] - row_m1[jm1];
        let d00 = row_00[j00] - row_00[jm1];
        if dm1 * d00 > 0.0 || (dm1 == 0.0 && d00 == 0.0) {
            0.5 * (qm1 + q00)
        } else if dm1 == 0.0 {
            qm1
        } else if d00 == 0.0 {
            q00
        } else {
            find0_or_mid(qm1, dm1, q00, d00)
        }
    };
    get_info_at_q(model, core_q)
}

/// Interpolates stellar structure properties at fractional mass coordinate
/// `bdy_q` (`get_info_at_q` in `star/private/report.f90:1262-1386`).
///
/// Reproduces the exact `interp3` formula and `[fmin, fmax]` clamp at
/// `report.f90:1375-1378` (logged as `#187` under `#169`).
#[must_use]
pub fn get_info_at_q(model: &StarModel, bdy_q: f64) -> CoreBoundaryInfo {
    let nz = model.nz;
    if bdy_q <= 0.0 || nz == 0 {
        return CoreBoundaryInfo::default();
    }
    let k_1 = k_for_q_1based(model, bdy_q);
    if k_1 >= nz {
        return CoreBoundaryInfo {
            kbdy_1based: nz,
            ..CoreBoundaryInfo::default()
        };
    }
    if k_1 <= 1 {
        let vel_cms = if model.controls.u_flag {
            model.u[0]
        } else if model.controls.v_flag {
            model.v[0]
        } else {
            0.0
        };
        let (omega, omega_div_omega_crit) = if model.controls.rotation_flag {
            let om = model.omega[0];
            let wc = crate::report_utils::eval_upstream_omega_crit(model, 0);
            let ratio = if wc > 0.0 { (om / wc).min(1.0) } else { 0.0 };
            (om, ratio)
        } else {
            (0.0, 0.0)
        };
        return CoreBoundaryInfo {
            kbdy_1based: 1,
            mass_msun: model.mstar / M_SUN,
            radius_rsun: model.r[0] / R_SUN,
            lg_t: model.ln_t[0] / LN10,
            lg_rho: model.lnd[0] / LN10,
            lum_lsun: model.lum[0] / L_SUN,
            vel_cms,
            omega,
            omega_div_omega_crit,
        };
    }

    // `2 <= k_1 < nz` => `1 <= k0 <= nz - 2`, so `k0 - 1`, `k0`, `k0 + 1` are in `0..nz`.
    let k0 = k_1 - 1;
    let kbdy_1based = k_1 + 1;
    let mass_msun = (model.m_center + model.xmstar * bdy_q) / M_SUN;

    let x = model.q[k0 - 1] - bdy_q;
    let x0 = 0.5 * model.dq[k0 - 1];
    let x1 = 0.5 * model.dq[k0] + model.dq[k0 - 1];
    let x2 = 0.5 * model.dq[k0 + 1] + model.dq[k0] + model.dq[k0 - 1];
    let alfa = ((bdy_q - model.q[k0 + 1]) / model.dq[k0]).clamp(0.0, 1.0);

    let interp3 = |f0: f64, f1: f64, f2: f64| -> f64 {
        let fmin = f0.min(f1).min(f2);
        let fmax = f0.max(f1).max(f2);
        let denom = (x0 - x1) * (x0 - x2) * (-x1 + x2);
        if denom == 0.0 {
            return f1;
        }
        // Verbatim reproduction of `report.f90:1375-1378` (`#187` under `#169`).
        let raw = (f1 * (x - x0) * (x - x2) * (x0 - x2)
            - (x - x1) * (f0 * (x - x2) * (x1 - x2) + (x - x0) * (x0 - x1) * x2))
            / denom;
        raw.max(fmin).min(fmax)
    };
    let interp2 = |f0: f64, f1: f64| -> f64 { alfa * f0 + (1.0 - alfa) * f1 };

    let lg_t = interp3(model.ln_t[k0 - 1], model.ln_t[k0], model.ln_t[k0 + 1]) / LN10;
    let lg_rho = interp3(model.lnd[k0 - 1], model.lnd[k0], model.lnd[k0 + 1]) / LN10;
    let r0 = model.r[k0];
    let r1 = model.r[k0 + 1];
    let radius_rsun = interp2(r0 * r0 * r0, r1 * r1 * r1).powf(1.0 / 3.0) / R_SUN;
    let lum_lsun = interp2(model.lum[k0], model.lum[k0 + 1]) / L_SUN;
    let vel_cms = if model.controls.u_flag {
        interp2(model.u[k0], model.u[k0 + 1])
    } else if model.controls.v_flag {
        interp2(model.v[k0], model.v[k0 + 1])
    } else {
        0.0
    };
    let (omega, omega_div_omega_crit) = if model.controls.rotation_flag {
        let om = interp2(model.omega[k0], model.omega[k0 + 1]);
        let wc = interp2(
            crate::report_utils::eval_upstream_omega_crit(model, k0),
            crate::report_utils::eval_upstream_omega_crit(model, k0 + 1),
        );
        let ratio = if wc > 0.0 { (om / wc).min(1.0) } else { 0.0 };
        (om, ratio)
    } else {
        (0.0, 0.0)
    };

    CoreBoundaryInfo {
        kbdy_1based,
        mass_msun,
        radius_rsun,
        lg_t,
        lg_rho,
        lum_lsun,
        vel_cms,
        omega,
        omega_div_omega_crit,
    }
}

/// Evaluates sub-cell interpolated core boundary masses ($M_\odot$), zone
/// indices, and boundary properties (`get_core_info` in
/// `star/private/report.f90:1029-1386`).
#[must_use]
pub fn compute_core_masses(model: &StarModel) -> CoreMasses {
    const A_MAX_FE: usize = 47;
    const A_MAX_ONE: usize = 20;
    const A_MAX_CO: usize = 16;
    const A_MAX_HE: usize = 4;

    let nz = model.nz;
    let sp = model.species;
    if nz == 0 || sp == 0 {
        return CoreMasses::default();
    }

    let catalog = NuclideCatalog::builtin_ref();
    let ts = &model.controls.ts_stop;
    let min_x = ts.min_boundary_fraction;

    let col_of = |name: &str| -> Option<usize> {
        let cid = catalog.get_nuclide_index(name)?;
        model.chem_id.iter().position(|&id| id == cid)
    };
    let h1 = col_of("h1");
    let he4 = col_of("he4");
    let c12 = col_of("c12");
    let o16 = col_of("o16");
    let ne20 = col_of("ne20");
    let si28 = col_of("si28");

    let a_of: Vec<usize> = model
        .chem_id
        .iter()
        .map(|&cid| catalog.get_by_id(cid).map_or(1, |n| n.z + n.n))
        .collect();

    let mut out = CoreMasses::default();

    for k in 0..nz {
        if model.ye[k] <= ts.neutron_rich_core_boundary_ye_max {
            let bdy = eval_set_core_info(model, k);
            out.neutron_rich_bdy = bdy;
            out.neutron_rich_core_mass = bdy.mass_msun;
            out.neutron_rich_core_k = (bdy.kbdy_1based > 0).then(|| bdy.kbdy_1based - 1);
            break;
        }
    }

    let mut have_he = false;
    let mut have_co = false;
    let mut have_one = false;

    for k in 0..nz {
        let row = model.cell_xa(k);
        let j_dom = dominant_species_index(row);
        let a_max = a_of[j_dom];

        let hit_fe = if ts.fe_core_boundary_si28_fraction < 0.0 {
            a_max >= A_MAX_FE
        } else if let Some(j_si) = si28 {
            if row[j_si] <= ts.fe_core_boundary_si28_fraction {
                let sum_fe: f64 = (0..sp)
                    .filter(|&j| a_of[j] >= A_MAX_FE)
                    .map(|j| row[j])
                    .sum();
                sum_fe >= min_x
            } else {
                false
            }
        } else {
            false
        };
        if hit_fe {
            let bdy = eval_set_core_info(model, k);
            out.fe_bdy = bdy;
            out.fe_core_mass = bdy.mass_msun;
            out.fe_core_k = (bdy.kbdy_1based > 0).then(|| bdy.kbdy_1based - 1);
            break;
        }

        if !have_one {
            let hit_one = if ts.one_core_boundary_he4_c12_fraction < 0.0 {
                a_max >= A_MAX_ONE
            } else if let (Some(j_he), Some(j_c), Some(j_o), Some(j_ne)) =
                (he4, c12, o16, ne20)
            {
                row[j_he] + row[j_c] <= ts.one_core_boundary_he4_c12_fraction
                    && row[j_o] + row[j_ne] >= min_x
            } else {
                false
            };
            if hit_one {
                have_one = true;
                let bdy = eval_set_core_info(model, k);
                out.one_bdy = bdy;
                out.one_core_mass = bdy.mass_msun;
                out.one_core_k = (bdy.kbdy_1based > 0).then(|| bdy.kbdy_1based - 1);
                if have_co && have_he {
                    break;
                }
            }
        }

        if !have_co {
            let hit_co = if ts.co_core_boundary_he4_fraction < 0.0 {
                a_max >= A_MAX_CO
            } else if let (Some(j_he), Some(j_c), Some(j_o)) = (he4, c12, o16) {
                row[j_he] <= ts.co_core_boundary_he4_fraction
                    && row[j_c] + row[j_o] >= min_x
            } else {
                false
            };
            if hit_co {
                have_co = true;
                let bdy = eval_set_core_info(model, k);
                out.co_bdy = bdy;
                out.co_core_mass = bdy.mass_msun;
                out.co_core_k = (bdy.kbdy_1based > 0).then(|| bdy.kbdy_1based - 1);
                if have_he {
                    break;
                }
            }
        }

        if !have_he {
            let hit_he = if ts.he_core_boundary_h1_fraction < 0.0 {
                a_max >= A_MAX_HE
            } else if let (Some(j_h), Some(j_he)) = (h1, he4) {
                row[j_h] <= ts.he_core_boundary_h1_fraction
                    && row[j_he] >= min_x
            } else {
                false
            };
            if hit_he {
                have_he = true;
                let bdy = eval_set_core_info(model, k);
                out.he_bdy = bdy;
                out.he_core_mass = bdy.mass_msun;
                out.he_core_k = (bdy.kbdy_1based > 0).then(|| bdy.kbdy_1based - 1);
            }
        }
    }

    out
}

/// Computes integrated nuclear burning category luminosities and neutrino
/// losses in $L_\odot$ (`star/private/report.f90:76-113`).
#[must_use]
pub fn compute_power_categories(model: &StarModel) -> PowerCategories {
    let nz = model.nz;
    if nz == 0 {
        return PowerCategories::default();
    }
    let op_split = model.controls.op_split_burn;
    let op_min_t = model.controls.op_split_burn_min_t;

    let mut l_nuc_erg = 0.0_f64;
    let mut l_nuc_neu_erg = 0.0_f64;
    let mut l_non_nuc_neu_erg = 0.0_f64;
    let mut cat_erg = [0.0_f64; NUM_CATEGORIES];

    for k in 0..nz {
        let dm_k = model.dm[k];
        let t_start = model
            .ln_t_start
            .get(k)
            .copied()
            .unwrap_or(model.ln_t[k])
            .exp();
        let eps_k = if op_split
            && t_start >= op_min_t
            && k < model.aux.burn_avg_epsnuc.len()
        {
            model.aux.burn_avg_epsnuc[k]
        } else {
            model.eps_nuc[k]
        };
        l_nuc_erg += dm_k * eps_k;
        l_nuc_neu_erg +=
            dm_k * model.eps_nuc_neu_total.get(k).copied().unwrap_or(0.0);
        l_non_nuc_neu_erg +=
            dm_k * model.non_nuc_neu.get(k).copied().unwrap_or(0.0);

        if let Some(cats) = model.aux.eps_nuc_categories.get(k) {
            for (c, &val) in cats.iter().enumerate() {
                cat_erg[c] += dm_k * val;
            }
        }
    }

    let mut l_by_category_lsun = [0.0_f64; NUM_CATEGORIES];
    // Matches `set_power_info` in `star/private/report.f90:76-113`.
    let (
        power_nuc_burn,
        power_nuc_neutrinos,
        power_h_burn,
        power_he_burn,
        power_z_burn,
        power_photo,
    ) = if model.controls.eps_nuc_factor == 0.0 {
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    } else {
        for (dst, &erg) in l_by_category_lsun.iter_mut().zip(&cat_erg) {
            *dst = erg / L_SUN;
        }
        let p_nuc = l_nuc_erg / L_SUN;
        let p_nuc_neu = l_nuc_neu_erg / L_SUN;
        let lh = (cat_erg[CAT_PP] + cat_erg[CAT_CNO]) / L_SUN;
        let lhe = cat_erg[CAT_3ALF] / L_SUN;
        let lz = p_nuc - (lh + lhe);
        let lphoto = cat_erg[CAT_PHOTO] / L_SUN;
        (p_nuc, p_nuc_neu, lh, lhe, lz, lphoto)
    };
    let power_non_nuc_neu = if model.controls.non_nuc_neu_factor == 0.0 {
        0.0
    } else {
        l_non_nuc_neu_erg / L_SUN
    };
    PowerCategories {
        power_nuc_burn,
        power_h_burn,
        power_he_burn,
        power_z_burn,
        power_photo,
        power_nuc_neutrinos,
        power_non_nuc_neu,
        power_neutrinos: power_nuc_neutrinos + power_non_nuc_neu,
        l_by_category_lsun,
    }
}

/// Snapshots end-of-step scalar luminosities, $T_{\mathrm{eff}}$, $\dot{M}$,
/// and rotational `*_old` profiles into [`crate::hydro_state::HydroAuxState`].
pub fn snapshot_step_history_old(model: &mut StarModel) {
    let powers = compute_power_categories(model);
    let l_surf = model.surface_l_lsun();
    let teff = model.teff;
    let mdot = model.mstar_dot;
    model.aux.omega_old.clone_from(&model.omega);
    model.aux.j_rot_old.clone_from(&model.j_rot);
    model.aux.eps_mdot_old.clone_from(&model.eps_mdot);
    let sh = &mut model.aux.step_hist;
    sh.teff_old = teff;
    sh.l_surf_old = l_surf;
    sh.mstar_dot_old = mdot;
    sh.power_h_burn_old = powers.power_h_burn;
    sh.power_he_burn_old = powers.power_he_burn;
    sh.power_z_burn_old = powers.power_z_burn;
    sh.power_photo_old = powers.power_photo;
    sh.power_nuc_burn_old = powers.power_nuc_burn;
    sh.total_energy_old = model.aux.total_energy;
    sh.cumulative_energy_error_old = model.aux.cumulative_energy_error;
}
