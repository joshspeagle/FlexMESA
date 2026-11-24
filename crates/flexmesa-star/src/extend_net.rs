// SPDX-License-Identifier: GPL-3.0-or-later
//! Automatic network extension `h_he_net` -> `co_net` -> `adv_net`
//! (`extend_net` in `star/job/run_star_support.f90:1804-1886`, called each
//! step from `before_step_loop` when `auto_extend_net` is set,
//! `star/job/run_star_support.f90:579-582`).

use std::path::Path;

use flexmesa_chem::nuclides::NuclideCatalog;
use flexmesa_const::LN10;

use crate::controls::StarError;
use crate::micro::change_model_network_with_adjust;
use crate::model::StarModel;
use crate::report_utils::center_value;

/// `current_abundance_at_point(id, iso, k, ierr)`
/// (`star/public/star_lib.f90:1917-1943`) through `eval_current_abundance`
/// (`star/private/star_utils.f90:264-287`): the `dm`-weighted mass fraction
/// of `iso` over the single cell `k` (0-based here), or `0` when `iso` is
/// not in the net.
fn current_abundance_at_point(model: &StarModel, iso: &str, k: usize) -> f64 {
    let Some(cid) = NuclideCatalog::builtin_ref().get_nuclide_index(iso)
    else {
        return 0.0;
    };
    let Some(j) = model.net_iso.get(cid).copied().flatten() else {
        return 0.0;
    };
    let mut total_mass = 0.0_f64;
    let mut jmass = 0.0_f64;
    let cell_mass = model.dm[k];
    total_mass += cell_mass;
    jmass += cell_mass * model.xa[k * model.species + j];
    jmass / total_mass
}

/// `change_net` inside `extend_net` (`run_star_support.f90:1853-1883`):
/// `star_change_to_new_net` with `s% job% adjust_abundances_for_new_isos`,
/// then `s% dt_next = s% dt_next/5`.
fn change_net(
    model: &mut StarModel,
    net_name: &str,
    case_dir: &Path,
) -> Result<(), StarError> {
    let adjust = model.job.adjust_abundances_for_new_isos;
    change_model_network_with_adjust(model, net_name, Some(case_dir), adjust)?;
    if model.job.net_name.trim() != net_name.trim() {
        return Err(StarError::InvalidInput(format!(
            "extend_net: failed to change net to {net_name} (star/job/run_star_support.f90:1866-1871)"
        )));
    }
    // `write(*,'(a)') ' new net = ' // trim(s% net_name)` (`:1873`).
    eprintln!(" new net = {}", model.job.net_name.trim());
    model.dt_next /= 5.0;
    Ok(())
}

/// Extends the network when the evolutionary stage needs it (`extend_net`,
/// `star/job/run_star_support.f90:1804-1886`). Returns whether the net
/// changed.
///
/// `s% log_max_temperature` and `s% log_center_density` are the `do_report`
/// values (`star/private/report.f90:173-174`), evaluated here from the
/// current model. After a change upstream re-reads `profile_columns_file`
/// (`star_set_profile_columns`, `run_star_support.f90:1823-1827,1842-1846`);
/// FlexMESA resolves profile column names against the current species when a
/// profile is written, so the re-read leaves the columns unchanged.
///
/// # Errors
/// Propagates [`change_model_network_with_adjust`] errors.
pub fn extend_net(
    model: &mut StarModel,
    case_dir: &Path,
) -> Result<bool, StarError> {
    // `tiny = 1d-10, small = 1d-2` (`run_star_support.f90:1809`).
    const TINY: f64 = 1.0e-10;
    const SMALL: f64 = 1.0e-2;

    let net_name = model.job.net_name.trim().to_string();
    if net_name == model.job.adv_net.trim() {
        return Ok(false);
    }

    let nz = model.nz;
    let log_max_temperature = model
        .ln_t
        .iter()
        .take(nz)
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        / LN10;
    let log_center_density = center_value(model, &model.lnd) / LN10;

    if net_name == model.job.co_net.trim() {
        if log_max_temperature > 9.0 || log_center_density > 9.0 {
            let adv = model.job.adv_net.clone();
            change_net(model, &adv, case_dir)?;
            // No `star_set_profile_columns` re-read (`run_star_support.f90:
            // 1823-1827,1842-1846`): profile columns resolve by name at write.
            return Ok(true);
        }
        return Ok(false);
    }

    if net_name == model.job.h_he_net.trim() {
        let cntr_h = current_abundance_at_point(model, "h1", nz - 1);
        if cntr_h > TINY {
            return Ok(false);
        }
        let cntr_he = current_abundance_at_point(model, "he4", nz - 1);
        if cntr_he > SMALL {
            return Ok(false);
        }
        if log_max_temperature > 8.3 || log_center_density > 8.5 {
            let co = model.job.co_net.clone();
            change_net(model, &co, case_dir)?;
            // No `star_set_profile_columns` re-read (`run_star_support.f90:
            // 1823-1827,1842-1846`): profile columns resolve by name at write.
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controls::{StarControls, StarJobControls};
    use crate::zams::load_zams_model;

    /// A ZAMS model on `h_he_net` with the centre set to `(h1, he4)` and
    /// `log_max_temperature` raised to `lg_tmax`.
    fn basic_net_model(h1: f64, he4: f64, lg_tmax: f64) -> StarModel {
        let mut m =
            load_zams_model(StarControls::default(), StarJobControls::default())
                .expect("1 Msun ZAMS should load");
        assert_eq!(m.job.net_name.trim(), m.job.h_he_net.trim());
        let cat = NuclideCatalog::builtin_ref();
        let jh = m.net_iso[cat.get_nuclide_index("h1").unwrap()].unwrap();
        let jhe = m.net_iso[cat.get_nuclide_index("he4").unwrap()].unwrap();
        let k = m.nz - 1;
        let sp = m.species;
        let jc = m.net_iso[cat.get_nuclide_index("c12").unwrap()].unwrap();
        // Keep the centre cell normalised: the shortfall goes to c12.
        let others: f64 = (0..sp)
            .filter(|&j| j != jh && j != jhe && j != jc)
            .map(|j| m.xa[k * sp + j])
            .sum();
        m.xa[k * sp + jh] = h1;
        m.xa[k * sp + jhe] = he4;
        m.xa[k * sp + jc] = 1.0 - h1 - he4 - others;
        m.ln_t[k] = lg_tmax * LN10;
        m
    }

    #[test]
    fn h_he_net_extends_to_co_net_when_center_h_and_he_exhausted_and_hot() {
        let mut m = basic_net_model(0.0, 1.0e-3, 8.31);
        let dt_next = m.dt_next;
        let dir = std::env::temp_dir();
        assert!(extend_net(&mut m, &dir).expect("extend_net"));
        assert_eq!(m.job.net_name.trim(), "co_burn.net");
        assert_eq!(m.dt_next, dt_next / 5.0);
    }

    #[test]
    fn h_he_net_is_kept_while_center_h1_exceeds_1e_10() {
        let mut m = basic_net_model(2.0e-10, 1.0e-3, 8.31);
        let dir = std::env::temp_dir();
        assert!(!extend_net(&mut m, &dir).expect("extend_net"));
        assert_eq!(m.job.net_name.trim(), "basic.net");
    }

    #[test]
    fn h_he_net_is_kept_below_both_temperature_and_density_thresholds() {
        let mut m = basic_net_model(0.0, 1.0e-3, 8.29);
        let dir = std::env::temp_dir();
        assert!(!extend_net(&mut m, &dir).expect("extend_net"));
        assert_eq!(m.job.net_name.trim(), "basic.net");
    }
}
