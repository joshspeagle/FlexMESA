//! Anders & Grevesse (1989) solar isotope queries, Lodders (2003) isotopic
//! percentages, and radioactive beta-decay mapping (`get_stable_mass_frac`).
//!
//! Matches the exact Fortran double-precision accumulation in `init_ag_data`
//! (`chem_def.f90`) and `chem_lib.f90`.

use crate::solar_ag89_data::{AG89_TABLE, SOLSIZ};

/// Return the Anders & Grevesse (1989) bulk solar metal mass fraction `zsol`.
///
/// Computed by summing `solx(j)` over all isotopes with $Z_j \ge 3$ in order,
/// yielding `1.8987109631891819e-2`.
#[must_use]
pub fn ag89_zsol() -> f64 {
    let mut sum = 0.0_f64;
    for &(_, solx, iz, _, _, _) in &AG89_TABLE {
        if iz >= 3 {
            sum += solx;
        }
    }
    sum
}

/// Return the Anders & Grevesse (1989) solar electron-to-baryon ratio `yesol`.
///
/// Computed by summing $(Z_j / A_j) X_j$ over all 286 stable isotopes in order,
/// yielding `8.5280190902027331e-1`.
#[must_use]
pub fn ag89_yesol() -> f64 {
    let mut sum = 0.0_f64;
    for &(_, solx, iz, ia, _, _) in &AG89_TABLE {
        sum += (f64::from(iz) / f64::from(ia)) * solx;
    }
    sum
}

/// Look up `(Z, A, X_sol)` for a solar isotope in Anders & Grevesse (1989).
///
/// Supports the aliases `"p"` -> `"h1"` and `"d"` -> `"h2"` defined in
/// `init_ag_data`.
#[must_use]
pub fn chem_get_solar(name: &str) -> Option<(f64, f64, f64)> {
    let target = match name.trim() {
        "p" => "h1",
        "d" => "h2",
        other => other,
    };
    AG89_TABLE.iter().find_map(|&(nm, solx, iz, ia, _, _)| {
        (nm == target).then(|| (f64::from(iz), f64::from(ia), solx))
    })
}

/// Return the Anders & Grevesse (1989) solar mass fraction $X_{\odot}(i)$.
///
/// Returns `0.0` if `name` is not in the 286-isotope table (`chem_Xsol`).
#[must_use]
pub fn chem_xsol(name: &str) -> f64 {
    chem_get_solar(name).map_or(0.0, |(_, _, x)| x)
}

/// Return the Lodders (2003, Table 6) isotopic atom percentage within its
/// element for `nuclei` (e.g. `"fe56"` -> `91.754`), or `0.0` if absent.
#[must_use]
pub fn lodders03_element_atom_percent(nuclei: &str) -> f64 {
    let target = nuclei.trim();
    AG89_TABLE
        .iter()
        .find_map(|&(nm, _, _, _, _, pct)| (nm == target).then_some(pct))
        .unwrap_or(0.0)
}

/// Map an arbitrary nuclear network abundance vector onto the 286 stable
/// solar isotopes assuming complete beta decay (`get_stable_mass_frac`).
///
/// Special-cases ${}^8\mathrm{Be} \to 2\,{}^4\mathrm{He}$ and the 12 odd-odd
/// branching exceptions from Frank Timmes's `decay.zip` in `chem_lib.f90`.
#[must_use]
pub fn get_stable_mass_frac(
    z_in: &[usize],
    a_in: &[usize],
    abun_in: &[f64],
) -> Vec<f64> {
    let n = z_in.len().min(a_in.len()).min(abun_in.len());
    let mut abun_out = vec![0.0_f64; SOLSIZ];
    for i in 0..n {
        let z = z_in[i];
        let a = a_in[i];
        if z == 4 && a == 8 {
            // Special case in upstream chem_lib.f90: j = 2 (second table row)
            abun_out[1] += 2.0 * abun_in[i];
        }
        for (j, &(_, _, iz_u16, ia_u16, jc, _)) in AG89_TABLE.iter().enumerate()
        {
            let iz = usize::from(iz_u16);
            let ia = usize::from(ia_u16);
            if a != ia {
                continue;
            }
            let special = matches!(
                (z, a, iz),
                (17, 36, 18)
                    | (21, 46 | 48, 22)
                    | (25, 54, 24)
                    | (27, 58, 26)
                    | (29, 64, 28)
                    | (31, 70, 32)
                    | (33, 74, 32)
                    | (33, 76, 34)
                    | (35, 78, 34)
                    | (35, 80 | 82, 36)
                    | (37, 84, 36)
            );
            if jc == 0
                || (z >= iz && jc == 1)
                || (z <= iz && jc == 2)
                || (z == iz && jc == 3)
                || special
            {
                abun_out[j] += abun_in[i];
            }
        }
    }
    let total: f64 = abun_out.iter().sum();
    if total > 0.0 {
        for val in &mut abun_out {
            *val /= total;
        }
    }
    abun_out
}
