// SPDX-License-Identifier: LGPL-3.0-or-later
//! `relax_initial_Z` and `relax_initial_Y` through `do_internal_evolve`
//! (`do_relax_Z`, `star/private/relax.f90:3536-3701`; `do_relax_Y`,
//! `star/private/relax.f90:3704-3873`), from the built-in 4 Msun ZAMS model
//! to `new_Z = 0.01`, `new_Y = 0.27`.
//!
//! Expected values: MESA `26.4.1` (release tag `26.4.1`, MESA SDK
//! `x86_64-linux-26.6.1`, `gfortran 15.2.0`) run on the inlist below, with
//! one diagnostic `write` statement added at the end of `do_internal_evolve`
//! (`star/private/relax.f90:4146`) printing the values as IEEE-754 `Z16.16`
//! hexadecimal. The Z relax took 5 steps and the Y relax 3; upstream then
//! restores `model_number`. These are upstream reference values, not
//! `FlexMESA` output.

#![forbid(unsafe_code)]

use std::fs;

use flexmesa_star::{
    init_or_update_model_from_bundle, DefaultStarExtras, InlistBundle,
};

const INLIST: &str = "\
&star_job
   relax_initial_Z = .true.
   new_Z = 0.01d0
   relax_initial_Y = .true.
   new_Y = 0.27d0
/
&eos
/
&kap
   Zbase = 0.02d0
   kap_file_prefix = 'gs98'
/
&controls
   initial_mass = 4
   initial_z = 0.02d0
   max_model_number = 2
/
";

/// `s% xa(1,1)` (`h1` at the surface) after `do_relax_Y`.
const MESA_XA_H1_SURF: u64 = 0x3FE7_0AB7_231E_FF63;
/// `s% xa(1,nz)` (`h1` at the centre) after `do_relax_Y`.
const MESA_XA_H1_CENTER: u64 = 0x3FE7_094D_13DF_9F80;
/// `s% initial_z` after `do_relax_Z` (its last `next_z`).
const MESA_INITIAL_Z: u64 = 0x3F84_7AE1_47AE_147D;
/// `s% initial_y` after `do_relax_Y` (its last `next_y`).
const MESA_INITIAL_Y: u64 = 0x3FD1_8198_F618_DA45;
/// Zone count after the two relaxes.
const MESA_NZ: usize = 972;

#[test]
fn relax_initial_z_then_y_from_zams_matches_mesa_26_4_1_composition() {
    let dir = std::env::temp_dir()
        .join(format!("flexmesa_relax_z_y_{}", std::process::id()));
    fs::create_dir_all(&dir).expect("create case dir");
    fs::write(dir.join("inlist"), INLIST).expect("write inlist");
    let bundle =
        InlistBundle::from_file_in_case_dir(&dir, "inlist").expect("inlist");
    let model =
        init_or_update_model_from_bundle(&bundle, None, &mut DefaultStarExtras)
            .expect("relax Z then Y");
    let _ = fs::remove_dir_all(&dir);

    assert_eq!(model.nz, MESA_NZ);
    // `restore_stuff` puts back `model_number` (`relax.f90:4234-4283`).
    assert_eq!(model.model_number, 0);
    let sp = model.species;
    let h1 = model
        .iso_names
        .iter()
        .position(|n| n.eq_ignore_ascii_case("h1"))
        .expect("h1");
    assert_eq!(model.xa[h1].to_bits(), MESA_XA_H1_SURF);
    assert_eq!(
        model.xa[(model.nz - 1) * sp + h1].to_bits(),
        MESA_XA_H1_CENTER
    );
    assert_eq!(model.controls.initial_z.to_bits(), MESA_INITIAL_Z);
    assert_eq!(model.controls.initial_y.to_bits(), MESA_INITIAL_Y);
}
