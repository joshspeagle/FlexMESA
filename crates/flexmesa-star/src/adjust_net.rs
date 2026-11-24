// SPDX-License-Identifier: LGPL-3.0-or-later
//! Heger-style adaptive nuclear network (Woosley, Heger, et al. 2004, ApJS
//! 151, 75): before each step the network is rebuilt from the isotopes whose
//! abundance is significant, together with their reaction neighbours
//! (`check_adjust_net` in `star/private/adjust_net.f90:31-247`, called from
//! `before_step_loop` in `star/job/run_star_support.f90:567-577` through
//! `star_adjust_net` in `star/public/star_lib.f90:944-958`).

use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use flexmesa_chem::elements::{EL_NAME, MAX_EL_Z};
use flexmesa_chem::nuclides::NuclideCatalog;

use crate::controls::StarError;
use crate::micro::change_model_network_with_adjust;
use crate::model::StarModel;

/// Smallest and largest neutron number per element over every isotope in the
/// isotope table (`element_min_N`, `element_max_N`;
/// `chem/private/chem_isos_io.f90:141-148`). Elements without isotopes keep
/// `(99999, -1)`.
fn element_n_range() -> &'static [(i64, i64); MAX_EL_Z + 1] {
    static RANGE: OnceLock<[(i64, i64); MAX_EL_Z + 1]> = OnceLock::new();
    RANGE.get_or_init(|| {
        let mut range = [(99_999_i64, -1_i64); MAX_EL_Z + 1];
        for e in NuclideCatalog::builtin_ref().entries() {
            if e.z > MAX_EL_Z {
                continue;
            }
            let n = e.n as i64;
            let r = &mut range[e.z];
            if n < r.0 {
                r.0 = n;
            }
            if n > r.1 {
                r.1 = n;
            }
        }
        range
    })
}

/// Fortran `write(num, '(a,i05.05)')` on `mod(n, 10**num_digits)`, or the
/// left-adjusted list-directed `n` when that remainder is zero
/// (`get_string_for_model_number` in `star/private/star_utils.f90:336-351`).
fn string_for_model_number(n: usize, num_digits: u32) -> String {
    let val = n % 10_usize.pow(num_digits);
    if val == 0 {
        return n.to_string();
    }
    format!("{val:0width$}", width = num_digits as usize)
}

/// Fortran `f10.5` for the `log10 max x` comment column.
fn fortran_f10_5(x: f64) -> String {
    format!("{x:10.5}")
}

/// Fortran `a8`: right-justified in eight columns.
fn fortran_a8(s: &str) -> String {
    format!("{s:>8}")
}

/// Rebuilds the network from the current abundances and, when its isotope
/// list changes, switches the model to it (`check_adjust_net`,
/// `star/private/adjust_net.f90:31-247`).
///
/// The generated net is written as `<case_dir>/nets/<model>_<species>.net`
/// (with the scratch `<case_dir>/.temp_net`), as upstream writes `nets/` and
/// `.temp_net` in the run directory (`adjust_net.f90:122-123,194-196`).
///
/// # Errors
/// Returns [`StarError::IoError`] if the net files cannot be written and
/// propagates [`change_model_network_with_adjust`] errors.
#[allow(clippy::too_many_lines)]
pub fn check_adjust_net(
    model: &mut StarModel,
    case_dir: &Path,
) -> Result<(), StarError> {
    let job = &model.job;
    let (min_x_for_keep, min_x_for_n, min_x_for_add) =
        (job.min_x_for_keep, job.min_x_for_n, job.min_x_for_add);
    let (max_z_for_add, max_n_for_add, max_a_for_add) =
        (job.max_z_for_add, job.max_n_for_add, job.max_a_for_add);
    let catalog = NuclideCatalog::builtin_ref();
    let element_n = element_n_range();
    let species = model.species;
    let nz = model.nz;

    // `adjust_net.f90:61-64`.
    let mut min_n = [-1_i64; MAX_EL_Z + 1];
    let mut max_n = [-1_i64; MAX_EL_Z + 1];
    let next_model_number = model.model_number + 1;

    // `include_iso`, `adjust_net.f90:239-245`.
    let mut include_iso = |z: i64, n: i64| {
        if z < 0 || z > MAX_EL_Z as i64 || n < 0 {
            return;
        }
        let zu = z as usize;
        let (el_min, el_max) = element_n[zu];
        if n < el_min || n > el_max {
            return;
        }
        if n < min_n[zu] || min_n[zu] < 0 {
            min_n[zu] = n;
        }
        if n > max_n[zu] {
            max_n[zu] = n;
        }
    };

    // `adjust_net.f90:66-79`.
    include_iso(0, 1); // neut
    include_iso(1, 0); // h1
    include_iso(2, 2); // he4
    include_iso(6, 6); // c12
    include_iso(6, 7); // c13
    include_iso(7, 7); // n14
    include_iso(8, 8); // o16
    include_iso(10, 10); // ne20
    include_iso(10, 12); // ne22
    include_iso(12, 12); // mg24
    include_iso(12, 14); // mg26
    include_iso(14, 14); // si28
    include_iso(16, 16); // s32
    include_iso(26, 30); // fe56

    // `adjust_net.f90:81-120`.
    let mut max_x_for_species = vec![0.0_f64; species];
    for j in 0..species {
        // `maxval(s% xa(j,1:nz))`
        let mut max_x = f64::NEG_INFINITY;
        for k in 0..nz {
            let x = model.xa[k * species + j];
            if x > max_x {
                max_x = x;
            }
        }
        max_x_for_species[j] = max_x;
        if max_x < min_x_for_keep {
            continue;
        }
        let cid = model.chem_id[j];
        let entry = catalog.get_by_id(cid).ok_or_else(|| {
            StarError::InvalidInput(format!(
                "check_adjust_net: unknown chem_id {cid} (star/private/adjust_net.f90:84-86)"
            ))
        })?;
        let z = entry.z as i64;
        let n = entry.n as i64;
        let a = z + n;
        let (zf, nf, af) = (z as f64, n as f64, a as f64);
        include_iso(z, n);
        if max_x >= min_x_for_n
            && nf + 1.0 <= max_n_for_add
            && af + 1.0 <= max_a_for_add
        {
            include_iso(z, n + 1); // (n,g)
            include_iso(z, n - 1); // (g,n)
        }
        if max_x >= min_x_for_add {
            if zf + 1.0 <= max_z_for_add && af + 1.0 <= max_a_for_add {
                include_iso(z + 1, n); // (p,g)
                include_iso(z - 1, n); // (g,p)
            }
            if zf + 2.0 <= max_z_for_add
                && nf + 2.0 <= max_n_for_add
                && af + 4.0 <= max_a_for_add
            {
                include_iso(z + 2, n + 2); // (a,g)
                include_iso(z - 2, n - 2); // (g,a)
            }
            if zf + 2.0 <= max_z_for_add
                && nf + 1.0 <= max_n_for_add
                && af + 3.0 <= max_a_for_add
            {
                include_iso(z + 2, n + 1); // (a,n)
                include_iso(z - 2, n - 1); // (n,a)
            }
            if zf + 1.0 <= max_z_for_add
                && nf + 2.0 <= max_n_for_add
                && af + 3.0 <= max_a_for_add
            {
                include_iso(z + 1, n + 2); // (a,p)
                include_iso(z - 1, n - 2); // (p,a)
            }
            if zf + 1.0 <= max_z_for_add
                && nf + 1.0 <= max_n_for_add
                && af + 2.0 <= max_a_for_add
            {
                include_iso(z + 1, n - 1); // (p,n)
                include_iso(z - 1, n + 1); // (n,p)
            }
            if zf + 4.0 <= max_z_for_add
                && nf + 4.0 <= max_n_for_add
                && af + 8.0 <= max_a_for_add
            {
                include_iso(z + 4, n + 4); // (2a,g)
                include_iso(z + 3, n + 4); // (2a,p)
            }
        }
    }

    // `adjust_net.f90:129-177`: the `add_isos_and_reactions` list.
    let mut temp = String::new();
    temp.push_str("add_isos_and_reactions(\n");
    temp.push_str("!       iso    log10 max x\n");
    let mut new_species = 0_usize;
    let mut have_new_isos = false;
    let log10_max_x = |i: usize| -> f64 {
        flexmesa_math::log10(f64::max(1.0e-199, max_x_for_species[i]))
    };
    for z in 0..=MAX_EL_Z {
        if min_n[z] < 0 || max_n[z] < min_n[z] {
            continue;
        }
        if z == 0 {
            // `ineut`: `el_name(0) = 'neut'` is the neutron's iso name.
            let cid = catalog.get_nuclide_index("neut").ok_or_else(|| {
                StarError::InvalidInput(String::from(
                    "check_adjust_net: isotope table lacks neut (star/private/adjust_net.f90:138)",
                ))
            })?;
            match model.net_iso.get(cid).copied().flatten() {
                None => {
                    have_new_isos = true;
                    let _ = writeln!(
                        temp,
                        "   {}   ! newly added",
                        fortran_a8(EL_NAME[0])
                    );
                }
                Some(i) => {
                    let _ = writeln!(
                        temp,
                        "   {}   ! {}",
                        fortran_a8(EL_NAME[0]),
                        fortran_f10_5(log10_max_x(i))
                    );
                }
            }
            new_species += 1;
            continue;
        }
        temp.push('\n');
        for n in min_n[z]..=max_n[z] {
            let cname = format!("{}{}", EL_NAME[z], z as i64 + n);
            // `chem_get_iso_id(cname)`; `cycle` when not an isotope.
            let Some(cid) = catalog.get_nuclide_index(&cname) else {
                continue;
            };
            match model.net_iso.get(cid).copied().flatten() {
                None => {
                    have_new_isos = true;
                    let _ = writeln!(
                        temp,
                        "   {}   ! newly added",
                        fortran_a8(&cname)
                    );
                }
                Some(i) => {
                    let _ = writeln!(
                        temp,
                        "   {}   ! {}",
                        fortran_a8(&cname),
                        fortran_f10_5(log10_max_x(i))
                    );
                }
            }
            new_species += 1;
        }
    }
    temp.push_str(")\n\n");

    let temp_path = case_dir.join(".temp_net");
    fs::write(&temp_path, &temp).map_err(|e| {
        StarError::IoError(format!(
            "failed to open {} (star/private/adjust_net.f90:123-127): {e}",
            temp_path.display()
        ))
    })?;

    // `adjust_net.f90:181`.
    if new_species == species && !have_new_isos {
        return Ok(());
    }

    // `adjust_net.f90:189-209`.
    let net_name = format!(
        "{}_{}.net",
        string_for_model_number(next_model_number, 5),
        new_species
    );
    let nets_dir = case_dir.join("nets");
    fs::create_dir_all(&nets_dir).map_err(|e| {
        StarError::IoError(format!(
            "failed to create {} (star/private/adjust_net.f90:195): {e}",
            nets_dir.display()
        ))
    })?;
    let mut text = format!("! {net_name}{new_species:8} species\n\n");
    for line in temp.lines() {
        text.push_str(line.trim_end());
        text.push('\n');
    }
    let fname = nets_dir.join(&net_name);
    fs::write(&fname, text).map_err(|e| {
        StarError::IoError(format!(
            "failed to open {} (star/private/adjust_net.f90:196-200): {e}",
            fname.display()
        ))
    })?;

    // `adjust_net.f90:218-235`: `adjust_abundances_for_new_isos = .false.`
    // (`:55`).
    change_model_network_with_adjust(model, &net_name, Some(case_dir), false)?;
    if model.job.net_name.trim() != net_name {
        return Err(StarError::InvalidInput(format!(
            "check_adjust_net: failed to change net to {net_name} (star/private/adjust_net.f90:226-231)"
        )));
    }
    // `s% using_revised_net_name`, `s% revised_net_name` (`:233-234`) feed
    // only photo restarts, which reject this control
    // (`photo_io::read_photo_bytes`).
    model.aux.need_to_setvars = true;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_number_string_zero_pads_to_five_digits_and_wraps() {
        assert_eq!(string_for_model_number(1, 5), "00001");
        assert_eq!(string_for_model_number(37, 5), "00037");
        assert_eq!(string_for_model_number(123_456, 5), "23456");
        assert_eq!(string_for_model_number(100_000, 5), "100000");
    }

    #[test]
    fn element_n_range_spans_neut_and_the_xtra_placeholders_for_z0() {
        // `chem/data/isotopes.data`: Z = 0 holds neut (N = 1) and the
        // placeholders xtra1 (N = 100) and xtra2 (N = 200).
        let r = element_n_range();
        assert_eq!(r[0], (1, 200));
        assert_eq!(r[1].0, 0);
        assert!(r[1].1 >= 2);
    }
}
