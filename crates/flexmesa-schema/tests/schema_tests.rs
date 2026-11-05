// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration and property-based tests for `flexmesa-schema`.
//!
//! Verifies:
//! 1. Parsing of MESA `<pkg>/defaults/*.defaults` files into [`ControlDefault`]
//!    entries across logical, integer, float, and string types, underlined
//!    section headers, docstrings, and array overrides.
//! 2. Active column extraction and deduplication from `*_columns.list` files.
//! 3. Fortran `&namelist` (`inlist`) parsing and schema validation, including
//!    rejection of unknown controls, scalar indexing, and type mismatches.
//! 4. Property-based round-trip invariants via `proptest`.

use flexmesa_schema::{
    parse_columns_text, parse_control_value, parse_defaults_text,
    parse_namelist_text, validate_namelist_block, ControlType, ControlValue,
};
use proptest::prelude::*;

/// Representative MESA `eos.defaults` fixture with logical, float, integer,
/// and array controls under underlined section headings.
const SAMPLE_EOS_DEFAULTS: &str = "\
! eos module controls
! ===================

! PC EOS controls
! ---------------
   ! Select whether to use the PC eos.
   ! ::
      use_PC = .false.
      mass_fraction_limit_for_PC = 1d-4
      logRho1_PC_limit = 2.999d0 ! keep < logRho2

! FreeEOS controls
! ----------------
      Zbase = -1
      suffix_for_FreeEOS_Z(:) = ''
      suffix_for_FreeEOS_Z(13) = '_CO'
";

/// Representative `history_columns.list` fixture with active, commented,
/// and duplicate column declarations.
const SAMPLE_COLUMNS: &str = "\
!# general info
   model_number ! step count
   num_zones
   !## age
   star_age ! simulated years
   !star_age_sec
   model_number ! duplicate
";

/// Representative MESA `&eos` namelist fixture.
const SAMPLE_INLIST: &str = "\
&eos
   use_PC = .true.
   mass_fraction_limit_for_PC = 2.5d-4, logRho1_PC_limit = 3
   Zbase = 0.02d0
   suffix_for_FreeEOS_Z(13) = '_O_NE'
/
";

/// Verify `.defaults` and `*_columns.list` extraction and deduplication.
#[test]
fn test_parse_defaults_and_columns() {
    let schema = parse_defaults_text(SAMPLE_EOS_DEFAULTS).unwrap();
    assert_eq!(schema.len(), 5);
    assert_eq!(schema[0].name, "use_PC");
    assert_eq!(schema[0].control_type, ControlType::Logical);
    assert_eq!(schema[0].section, "PC EOS controls");
    assert!(!schema[0].is_array);

    assert_eq!(schema[4].name, "suffix_for_FreeEOS_Z");
    assert!(schema[4].is_array);
    assert_eq!(
        schema[4].element_overrides,
        vec![("13".to_owned(), ControlValue::String("_CO".to_owned()))]
    );

    let cols = parse_columns_text(SAMPLE_COLUMNS);
    assert_eq!(cols.len(), 3);
    assert_eq!(cols[0].name, "model_number");
    assert_eq!(cols[0].section, "general info");
    assert_eq!(cols[2].name, "star_age");
    assert_eq!(cols[2].section, "age");
}

/// Verify Fortran `&namelist` parsing and schema validation rules.
#[test]
fn test_parse_and_validate_namelist() {
    let schema = parse_defaults_text(SAMPLE_EOS_DEFAULTS).unwrap();
    let blocks = parse_namelist_text(SAMPLE_INLIST).unwrap();
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].name, "eos");
    assert!(validate_namelist_block(&blocks[0], &schema, false).is_ok());
    assert!(validate_namelist_block(&blocks[0], &schema, true).is_err());

    let bad_unknown =
        parse_namelist_text("&eos\n  no_such_ctrl = 1\n/\n").unwrap();
    assert!(validate_namelist_block(&bad_unknown[0], &schema, false).is_err());

    let bad_sub =
        parse_namelist_text("&eos\n  use_PC(1) = .true.\n/\n").unwrap();
    assert!(validate_namelist_block(&bad_sub[0], &schema, false).is_err());

    let bad_type = parse_namelist_text("&eos\n  use_PC = 1.0d0\n/\n").unwrap();
    assert!(validate_namelist_block(&bad_type[0], &schema, false).is_err());
}

proptest! {
    /// Property test: integer and boolean control literals round-trip cleanly.
    #[test]
    fn prop_control_value_integer_and_logical_roundtrip(
        int_val in -10_000_000_i64..10_000_000_i64,
        bool_val in any::<bool>(),
    ) {
        let parsed_int = parse_control_value(&int_val.to_string(), 1).unwrap();
        prop_assert_eq!(parsed_int, ControlValue::Integer(int_val));

        let bool_str = if bool_val { ".true." } else { ".false." };
        let parsed_bool = parse_control_value(bool_str, 1).unwrap();
        prop_assert_eq!(parsed_bool, ControlValue::Logical(bool_val));
    }
}
