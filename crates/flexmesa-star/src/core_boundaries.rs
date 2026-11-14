// SPDX-License-Identifier: LGPL-3.0-or-later
//! Sub-cell core-boundary mass integration (`get_core_info`, `set_core_info`)
//! and nuclear burning category luminosity summation (`report.f90:76-113` and
//! `1029-1259`).

use flexmesa_chem::{NuclideCatalog, NUM_CATEGORIES};
use flexmesa_const::{L_SUN, M_SUN};

use crate::model::StarModel;

/// Category index for proton-proton burning (`"pp"`, `CATEGORY_NAME[0]`).
const CAT_PP: usize = 0;
/// Category index for CNO cycle burning (`"cno"`, `CATEGORY_NAME[1]`).
const CAT_CNO: usize = 1;
/// Category index for triple-alpha He burning (`"tri_alpha"`, `CATEGORY_NAME[2]`).
const CAT_3ALF: usize = 2;
/// Category index for photodisintegration (`"photo"`, `CATEGORY_NAME[19]`).
const CAT_PHOTO: usize = 19;

/// Sub-cell interpolated core boundary masses ($M_\odot$) and outermost core
/// zone indices (`0`-based `0..nz`, `None` when absent) matching
/// `star/private/report.f90:1029-1259`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CoreMasses {
    /// Helium core mass in $M_\odot$ (`he_core_mass`).
    pub he_core_mass: f64,
    /// Outermost zone index inside the He core (`he_core_k`).
    pub he_core_k: Option<usize>,
    /// Carbon-oxygen core mass in $M_\odot$ (`co_core_mass`).
    pub co_core_mass: f64,
    /// Outermost zone index inside the CO core (`co_core_k`).
    pub co_core_k: Option<usize>,
    /// Oxygen-neon core mass in $M_\odot$ (`one_core_mass`).
    pub one_core_mass: f64,
    /// Outermost zone index inside the ONe core (`one_core_k`).
    pub one_core_k: Option<usize>,
    /// Iron-group core mass in $M_\odot$ (`fe_core_mass`).
    pub fe_core_mass: f64,
    /// Outermost zone index inside the Fe core (`fe_core_k`).
    pub fe_core_k: Option<usize>,
    /// Neutron-rich core mass in $M_\odot$ (`neutron_rich_core_mass`).
    pub neutron_rich_core_mass: f64,
    /// Outermost zone index inside the neutron-rich core (`neutron_rich_core_k`).
    pub neutron_rich_core_k: Option<usize>,
}

/// Integrated nuclear and neutrino luminosities in solar luminosities ($L_\odot$)
/// matching `star/private/report.f90:76-113`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
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

/// Sub-cell core mass fraction $q_{\mathrm{core}}$ from crossing `target` between
/// zones `k - 1` and `k`.
fn interp_core_q_threshold(
    model: &StarModel,
    k: usize,
    y_m1: f64,
    y_00: f64,
    target: f64,
) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let qm1 = model.q[k - 1] - 0.5 * model.dq[k - 1];
    let q00 = model.q[k] - 0.5 * model.dq[k];
    let dm1 = y_m1 - target;
    let d00 = y_00 - target;
    let core_q = if dm1 * d00 > 0.0 || (dm1 == 0.0 && d00 == 0.0) {
        0.5 * (qm1 + q00)
    } else if dm1 == 0.0 {
        qm1
    } else if d00 == 0.0 {
        q00
    } else {
        find0_or_mid(qm1, dm1, q00, d00)
    };
    core_q.clamp(0.0, 1.0)
}

/// Sub-cell core mass fraction $q_{\mathrm{core}}$ from dominant-species
/// crossover between cell centers `k - 1` and `k` (`set_core_info` in
/// `report.f90:1212-1259`).
fn interp_core_q_species(model: &StarModel, k: usize) -> f64 {
    if k == 0 || model.species == 0 {
        return 1.0;
    }
    let row_m1 = model.cell_xa(k - 1);
    let row_00 = model.cell_xa(k);
    let jm1 = dominant_species_index(row_m1);
    let j00 = dominant_species_index(row_00);
    let qm1 = model.q[k - 1] - 0.5 * model.dq[k - 1];
    let q00 = model.q[k] - 0.5 * model.dq[k];
    let dm1 = row_m1[j00] - row_m1[jm1];
    let d00 = row_00[j00] - row_00[jm1];
    let core_q = if dm1 * d00 > 0.0 || (dm1 == 0.0 && d00 == 0.0) {
        0.5 * (qm1 + q00)
    } else if dm1 == 0.0 {
        qm1
    } else if d00 == 0.0 {
        q00
    } else {
        find0_or_mid(qm1, dm1, q00, d00)
    };
    core_q.clamp(0.0, 1.0)
}

#[inline]
fn q_to_core_mass_msun(model: &StarModel, core_q: f64) -> f64 {
    (model.m_center + model.xmstar * core_q) / M_SUN
}

/// Evaluates sub-cell interpolated core boundary masses ($M_\odot$) and zone
/// indices (`get_core_info` in `star/private/report.f90:1029-1259`).
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
            let core_q = if k == 0 {
                1.0
            } else {
                interp_core_q_threshold(
                    model,
                    k,
                    model.ye[k - 1],
                    model.ye[k],
                    ts.neutron_rich_core_boundary_ye_max,
                )
            };
            out.neutron_rich_core_k = Some(k);
            out.neutron_rich_core_mass = q_to_core_mass_msun(model, core_q);
            break;
        }
    }

    for k in 0..nz {
        let row = model.cell_xa(k);
        let j_dom = dominant_species_index(row);
        let a_max = a_of[j_dom];

        if out.fe_core_k.is_none() {
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
                let core_q =
                    if ts.fe_core_boundary_si28_fraction < 0.0 || k == 0 {
                        interp_core_q_species(model, k)
                    } else if let Some(j_si) = si28 {
                        interp_core_q_threshold(
                            model,
                            k,
                            model.cell_xa(k - 1)[j_si],
                            row[j_si],
                            ts.fe_core_boundary_si28_fraction,
                        )
                    } else {
                        interp_core_q_species(model, k)
                    };
                out.fe_core_k = Some(k);
                out.fe_core_mass = q_to_core_mass_msun(model, core_q);
                break;
            }
        }

        if out.one_core_k.is_none() {
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
                let core_q =
                    if ts.one_core_boundary_he4_c12_fraction < 0.0 || k == 0 {
                        interp_core_q_species(model, k)
                    } else if let (Some(j_he), Some(j_c)) = (he4, c12) {
                        let row_m1 = model.cell_xa(k - 1);
                        interp_core_q_threshold(
                            model,
                            k,
                            row_m1[j_he] + row_m1[j_c],
                            row[j_he] + row[j_c],
                            ts.one_core_boundary_he4_c12_fraction,
                        )
                    } else {
                        interp_core_q_species(model, k)
                    };
                out.one_core_k = Some(k);
                out.one_core_mass = q_to_core_mass_msun(model, core_q);
            }
        }

        if out.co_core_k.is_none() {
            let hit_co = if ts.co_core_boundary_he4_fraction < 0.0 {
                a_max >= A_MAX_CO
            } else if let (Some(j_he), Some(j_c), Some(j_o)) = (he4, c12, o16) {
                row[j_he] <= ts.co_core_boundary_he4_fraction
                    && row[j_c] + row[j_o] >= min_x
            } else {
                false
            };
            if hit_co {
                let core_q = if ts.co_core_boundary_he4_fraction < 0.0 || k == 0
                {
                    interp_core_q_species(model, k)
                } else if let Some(j_he) = he4 {
                    interp_core_q_threshold(
                        model,
                        k,
                        model.cell_xa(k - 1)[j_he],
                        row[j_he],
                        ts.co_core_boundary_he4_fraction,
                    )
                } else {
                    interp_core_q_species(model, k)
                };
                out.co_core_k = Some(k);
                out.co_core_mass = q_to_core_mass_msun(model, core_q);
            }
        }

        if out.he_core_k.is_none() {
            let hit_he = if ts.he_core_boundary_h1_fraction < 0.0 {
                a_max >= A_MAX_HE
            } else if let (Some(j_h), Some(j_he)) = (h1, he4) {
                row[j_h] <= ts.he_core_boundary_h1_fraction
                    && row[j_he] >= min_x
            } else {
                false
            };
            if hit_he {
                let core_q = if ts.he_core_boundary_h1_fraction < 0.0 || k == 0
                {
                    interp_core_q_species(model, k)
                } else if let Some(j_h) = h1 {
                    interp_core_q_threshold(
                        model,
                        k,
                        model.cell_xa(k - 1)[j_h],
                        row[j_h],
                        ts.he_core_boundary_h1_fraction,
                    )
                } else {
                    interp_core_q_species(model, k)
                };
                out.he_core_k = Some(k);
                out.he_core_mass = q_to_core_mass_msun(model, core_q);
            }
        }
    }

    out
}

/// Computes integrated nuclear burning category luminosities and neutrino
/// losses in $L_\odot$ (`report.f90:76-113`).
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
    let mut any_cat = false;

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
                if val != 0.0 {
                    any_cat = true;
                    cat_erg[c] += dm_k * val;
                }
            }
        }
    }

    let power_nuc_burn = l_nuc_erg / L_SUN;
    let (power_h_burn, power_he_burn, power_z_burn, power_photo) = if any_cat {
        let lh = (cat_erg[CAT_PP] + cat_erg[CAT_CNO]) / L_SUN;
        let lhe = cat_erg[CAT_3ALF] / L_SUN;
        let lphoto = cat_erg[CAT_PHOTO] / L_SUN;
        let lz = power_nuc_burn - (lh + lhe);
        (lh, lhe, lz, lphoto)
    } else {
        let c_idx = nz - 1;
        let xh_c = model.xh_mass.get(c_idx).copied().unwrap_or(0.0);
        let yhe_c = model.yhe_mass.get(c_idx).copied().unwrap_or(0.0);
        if xh_c > 1.0e-4 {
            (power_nuc_burn.max(0.0), 0.0, 0.0, 0.0)
        } else if yhe_c > 1.0e-4 {
            (0.0, power_nuc_burn.max(0.0), 0.0, 0.0)
        } else {
            (0.0, 0.0, power_nuc_burn.max(0.0), 0.0)
        }
    };

    let power_nuc_neutrinos = l_nuc_neu_erg / L_SUN;
    let power_non_nuc_neu = l_non_nuc_neu_erg / L_SUN;
    PowerCategories {
        power_nuc_burn,
        power_h_burn,
        power_he_burn,
        power_z_burn,
        power_photo,
        power_nuc_neutrinos,
        power_non_nuc_neu,
        power_neutrinos: power_nuc_neutrinos + power_non_nuc_neu,
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
