// SPDX-License-Identifier: GPL-3.0-or-later
#![forbid(unsafe_code)]
//! Parity tests for Fortran namelist parsing in `flexmesa-schema`.
//!
//! Verifies Fortran 90/2003 namelist rules including array-slice value lists
//! (`key(1:6) = v1, v2, ...`), broadcast slice preservation, signed/strided
//! subscripts, column-major multi-dimensional slices, repeat counts (`n*val`),
//! null entries, quoted string escapes and continuations, `&` line
//! continuations, and `/` + `&end` terminators.

use flexmesa_schema::{
    parse_defaults_text, parse_namelist_text, validate_namelist_block,
    ControlValue,
};

#[test]
fn test_namelist_array_slice_comma_value_lists() {
    let text = "\
&astero_search_controls
  l0_obs_freq(1:6) = 100.5, 200.5, 300.5, 400.5, 500.5, 600.5
  partial_slice(2:6) = 1.0d0, 2.0d0, 3.0d0
  space_list(1:4) = 10.0 20.0 30.0 40.0
  spaced_parens ( 1 : 3 ) = 1.5 , 2.5 , 3.5
  tight(1:3)=4.0,5.0,6.0
/
";
    let blocks = parse_namelist_text(text).unwrap();
    assert_eq!(blocks.len(), 1);
    let assigns = &blocks[0].assignments;
    assert_eq!(assigns.len(), 19);

    for (idx, expected) in [100.5, 200.5, 300.5, 400.5, 500.5, 600.5]
        .iter()
        .enumerate()
    {
        let want_sub = (idx + 1).to_string();
        assert_eq!(assigns[idx].name, "l0_obs_freq");
        assert_eq!(assigns[idx].subscript.as_deref(), Some(want_sub.as_str()));
        assert_eq!(assigns[idx].value, ControlValue::Float(*expected));
    }

    assert_eq!(assigns[6].name, "partial_slice");
    assert_eq!(assigns[6].subscript.as_deref(), Some("2"));
    assert_eq!(assigns[7].subscript.as_deref(), Some("3"));
    assert_eq!(assigns[8].subscript.as_deref(), Some("4"));
    assert_eq!(assigns[8].value, ControlValue::Float(3.0));

    assert_eq!(assigns[9].name, "space_list");
    assert_eq!(assigns[9].subscript.as_deref(), Some("1"));
    assert_eq!(assigns[12].subscript.as_deref(), Some("4"));
    assert_eq!(assigns[12].value, ControlValue::Float(40.0));

    assert_eq!(assigns[13].name, "spaced_parens");
    assert_eq!(assigns[13].subscript.as_deref(), Some("1"));
    assert_eq!(assigns[15].subscript.as_deref(), Some("3"));
    assert_eq!(assigns[15].value, ControlValue::Float(3.5));

    assert_eq!(assigns[16].name, "tight");
    assert_eq!(assigns[16].subscript.as_deref(), Some("1"));
    assert_eq!(assigns[18].subscript.as_deref(), Some("3"));
    assert_eq!(assigns[18].value, ControlValue::Float(6.0));
}

#[test]
fn test_namelist_broadcast_slices_preserved() {
    let text = "\
&controls
  a(1:6) = 0.0d0
  b(:) = -1.0d0
  c(3:) = 2.0d0
  d(:5) = 3.0d0
  e( 2 : 5 ) = 4.0d0
  f(3:3) = 9.0d0
  freq_target(0, :) = -1.0d0
  freq_target(0, 1:4) = 100.0d0
/
";
    let blocks = parse_namelist_text(text).unwrap();
    let assigns = &blocks[0].assignments;
    assert_eq!(assigns.len(), 8);
    assert_eq!(assigns[0].subscript.as_deref(), Some("1:6"));
    assert_eq!(assigns[1].subscript.as_deref(), Some(":"));
    assert_eq!(assigns[2].subscript.as_deref(), Some("3:"));
    assert_eq!(assigns[3].subscript.as_deref(), Some(":5"));
    assert_eq!(assigns[4].subscript.as_deref(), Some("2:5"));
    assert_eq!(assigns[5].subscript.as_deref(), Some("3"));
    assert_eq!(assigns[6].subscript.as_deref(), Some("0,:"));
    assert_eq!(assigns[7].subscript.as_deref(), Some("0,1:4"));
}

#[test]
fn test_namelist_negative_subscripts_and_strided_slices() {
    let text = "\
&controls
  neg(-3) = 10, 20, 30
  neg_slice(-2:2) = 1, 2, 3, 4, 5
  stride(1:6:2) = 11, 22, 33
  rev(6:1:-2) = 60, 40, 20
  open_all(:) = 'a', 'b', 'c'
  open_lo(4:) = 'd', 'e'
  open_hi(:3) = 'x', 'y', 'z'
  strided_open(2::3) = 101, 102, 103
/
";
    let blocks = parse_namelist_text(text).unwrap();
    let a = &blocks[0].assignments;
    assert_eq!(a.len(), 25);

    assert_eq!(a[0].subscript.as_deref(), Some("-3"));
    assert_eq!(a[1].subscript.as_deref(), Some("-2"));
    assert_eq!(a[2].subscript.as_deref(), Some("-1"));

    assert_eq!(a[3].subscript.as_deref(), Some("-2"));
    assert_eq!(a[5].subscript.as_deref(), Some("0"));
    assert_eq!(a[7].subscript.as_deref(), Some("2"));
    assert_eq!(a[7].value, ControlValue::Integer(5));

    assert_eq!(a[8].subscript.as_deref(), Some("1"));
    assert_eq!(a[9].subscript.as_deref(), Some("3"));
    assert_eq!(a[10].subscript.as_deref(), Some("5"));
    assert_eq!(a[10].value, ControlValue::Integer(33));

    assert_eq!(a[11].subscript.as_deref(), Some("6"));
    assert_eq!(a[12].subscript.as_deref(), Some("4"));
    assert_eq!(a[13].subscript.as_deref(), Some("2"));
    assert_eq!(a[13].value, ControlValue::Integer(20));

    assert_eq!(a[14].subscript.as_deref(), Some("1"));
    assert_eq!(a[16].subscript.as_deref(), Some("3"));
    assert_eq!(a[17].subscript.as_deref(), Some("4"));
    assert_eq!(a[18].subscript.as_deref(), Some("5"));
    assert_eq!(a[19].subscript.as_deref(), Some("1"));
    assert_eq!(a[21].subscript.as_deref(), Some("3"));

    assert_eq!(a[22].subscript.as_deref(), Some("2"));
    assert_eq!(a[23].subscript.as_deref(), Some("5"));
    assert_eq!(a[24].subscript.as_deref(), Some("8"));
}

#[test]
fn test_namelist_multidim_slices_column_major() {
    let text = "\
&astero_search_controls
  freq_target(0, 1:4) = 1000.0, 1100.0, 1200.0, 1300.0
  mat(1:2, 1:2) = 11, 21, 12, 22
  tail(1:2, 3:) = 1, 2, 3, 4, 5
  single_2d(0, 1) = 77.0
/
";
    let blocks = parse_namelist_text(text).unwrap();
    let a = &blocks[0].assignments;
    assert_eq!(a.len(), 14);

    assert_eq!(a[0].subscript.as_deref(), Some("0,1"));
    assert_eq!(a[3].subscript.as_deref(), Some("0,4"));
    assert_eq!(a[3].value, ControlValue::Float(1300.0));

    assert_eq!(a[4].subscript.as_deref(), Some("1,1"));
    assert_eq!(a[4].value, ControlValue::Integer(11));
    assert_eq!(a[5].subscript.as_deref(), Some("2,1"));
    assert_eq!(a[5].value, ControlValue::Integer(21));
    assert_eq!(a[6].subscript.as_deref(), Some("1,2"));
    assert_eq!(a[6].value, ControlValue::Integer(12));
    assert_eq!(a[7].subscript.as_deref(), Some("2,2"));
    assert_eq!(a[7].value, ControlValue::Integer(22));

    assert_eq!(a[8].subscript.as_deref(), Some("1,3"));
    assert_eq!(a[9].subscript.as_deref(), Some("2,3"));
    assert_eq!(a[10].subscript.as_deref(), Some("1,4"));
    assert_eq!(a[11].subscript.as_deref(), Some("2,4"));
    assert_eq!(a[12].subscript.as_deref(), Some("1,5"));
    assert_eq!(a[12].value, ControlValue::Integer(5));

    assert_eq!(a[13].subscript.as_deref(), Some("0,1"));
    assert_eq!(a[13].value, ControlValue::Float(77.0));
}

#[test]
fn test_namelist_repeat_counts() {
    let text = "\
&controls
  rep(1:6) = 2*1.0d0, 4*2.5d0
  strs(1:3) = 3*'a*b c'
  empties(1:2) = 2*''
  negs(1:3) = 3*-7
/
";
    let blocks = parse_namelist_text(text).unwrap();
    let a = &blocks[0].assignments;

    let rep: Vec<(&str, f64)> = a
        .iter()
        .filter(|x| x.name == "rep")
        .map(|x| {
            let ControlValue::Float(v) = x.value else {
                panic!("expected float");
            };
            (x.subscript.as_deref().unwrap(), v)
        })
        .collect();
    assert_eq!(
        rep,
        vec![
            ("1", 1.0),
            ("2", 1.0),
            ("3", 2.5),
            ("4", 2.5),
            ("5", 2.5),
            ("6", 2.5),
        ]
    );

    let strs: Vec<&str> = a
        .iter()
        .filter(|x| x.name == "strs")
        .map(|x| match &x.value {
            ControlValue::String(s) => s.as_str(),
            _ => panic!("expected string"),
        })
        .collect();
    assert_eq!(strs, vec!["a*b c", "a*b c", "a*b c"]);

    let empties: Vec<&str> = a
        .iter()
        .filter(|x| x.name == "empties")
        .map(|x| match &x.value {
            ControlValue::String(s) => s.as_str(),
            _ => panic!("expected string"),
        })
        .collect();
    assert_eq!(empties, vec!["", ""]);

    let negs: Vec<(&str, i64)> = a
        .iter()
        .filter(|x| x.name == "negs")
        .map(|x| {
            let ControlValue::Integer(v) = x.value else {
                panic!("expected integer");
            };
            (x.subscript.as_deref().unwrap(), v)
        })
        .collect();
    assert_eq!(negs, vec![("1", -7), ("2", -7), ("3", -7)]);
}

#[test]
fn test_namelist_null_items_and_boolean_shorthands() {
    let text = "\
&controls
  skips(1:6) = 10, , 30, 2*, 60
  lead(1:3) = , 20, 30
  trail = 5,
  flags = t, f, t
  t = 42
  f(1:2) = 1, 2
/
";
    let blocks = parse_namelist_text(text).unwrap();
    let a = &blocks[0].assignments;

    let skips: Vec<(&str, i64)> = a
        .iter()
        .filter(|x| x.name == "skips")
        .map(|x| {
            let ControlValue::Integer(v) = x.value else {
                panic!("expected integer");
            };
            (x.subscript.as_deref().unwrap(), v)
        })
        .collect();
    assert_eq!(skips, vec![("1", 10), ("3", 30), ("6", 60)]);

    let lead: Vec<(&str, i64)> = a
        .iter()
        .filter(|x| x.name == "lead")
        .map(|x| {
            let ControlValue::Integer(v) = x.value else {
                panic!("expected integer");
            };
            (x.subscript.as_deref().unwrap(), v)
        })
        .collect();
    assert_eq!(lead, vec![("2", 20), ("3", 30)]);

    let trail = a.iter().find(|x| x.name == "trail").unwrap();
    assert_eq!(trail.subscript, None);
    assert_eq!(trail.value, ControlValue::Integer(5));

    let flags: Vec<(&str, bool)> = a
        .iter()
        .filter(|x| x.name == "flags")
        .map(|x| {
            let ControlValue::Logical(b) = x.value else {
                panic!("expected bool");
            };
            (x.subscript.as_deref().unwrap(), b)
        })
        .collect();
    assert_eq!(flags, vec![("1", true), ("2", false), ("3", true)]);

    let t_var = a.iter().find(|x| x.name == "t").unwrap();
    assert_eq!(t_var.subscript, None);
    assert_eq!(t_var.value, ControlValue::Integer(42));
}

#[test]
fn test_namelist_multiline_continuations_comments_and_quotes() {
    let text = "\
! Leading file comment
&star_job
  split_& ! inline comment after continuation
  ! comment line inside continuation

  &key = 7
  x_ctrl(1:3) =
    1.0d0, & ! value 1
    & 2.0d0,
    3.0d0
  escaped_s = 'don''t'
  double_q = \"say \"\"hi\"\"\"
  pgplot_lbl = 'log \\(0650) ! / = , &end'
  cont_str = 'hello &
             &world'
  nested_q = '''quoted'''
/
";
    let blocks = parse_namelist_text(text).unwrap();
    let a = &blocks[0].assignments;
    assert_eq!(a[0].name, "split_key");
    assert_eq!(a[0].value, ControlValue::Integer(7));
    assert_eq!(a[1].name, "x_ctrl");
    assert_eq!(a[1].subscript.as_deref(), Some("1"));
    assert_eq!(a[1].value, ControlValue::Float(1.0));
    assert_eq!(a[2].subscript.as_deref(), Some("2"));
    assert_eq!(a[2].value, ControlValue::Float(2.0));
    assert_eq!(a[3].subscript.as_deref(), Some("3"));
    assert_eq!(a[3].value, ControlValue::Float(3.0));
    assert_eq!(a[4].value, ControlValue::String("don't".to_owned()));
    assert_eq!(a[5].value, ControlValue::String("say \"hi\"".to_owned()));
    assert_eq!(
        a[6].value,
        ControlValue::String("log \\(0650) ! / = , &end".to_owned())
    );
    assert_eq!(a[7].value, ControlValue::String("hello world".to_owned()));
    assert_eq!(a[8].value, ControlValue::String("'quoted'".to_owned()));
}

#[test]
fn test_namelist_terminators_and_multiple_blocks() {
    let text = "\
&eos use_PC = .true., end_control = .false. &End &kap Zbase = 0.02 &END
&star_job &end
&controls/
";
    let blocks = parse_namelist_text(text).unwrap();
    assert_eq!(blocks.len(), 4);
    assert_eq!(blocks[0].name, "eos");
    assert_eq!(blocks[0].assignments.len(), 2);
    assert_eq!(blocks[0].assignments[1].name, "end_control");
    assert_eq!(blocks[0].assignments[1].value, ControlValue::Logical(false));
    assert_eq!(blocks[1].name, "kap");
    assert_eq!(blocks[1].assignments.len(), 1);
    assert_eq!(blocks[1].assignments[0].value, ControlValue::Float(0.02));
    assert_eq!(blocks[2].name, "star_job");
    assert_eq!(blocks[2].assignments, Vec::new());
    assert_eq!(blocks[3].name, "controls");
    assert_eq!(blocks[3].assignments, Vec::new());
}

#[test]
fn test_namelist_error_diagnostics() {
    let invalid_cases = [
        "&eos\n  a(1:3) = 1, 2, 3, 4\n/\n",
        "&eos\n  a(1:3) = 4*1\n/\n",
        "&eos\n  a(1:3) = 1, 3*\n/\n",
        "&eos\n  a(1:3) = 1, 2*, 4\n/\n",
        "&eos\n  a(5:1) = 10\n/\n",
        "&eos\n  a(1:3:0) = 1\n/\n",
        "&eos\n  a(:1:-1) = 1, 2\n/\n",
        "&eos\n  a = 0*1\n/\n",
        "&eos\n  a = -2*1\n/\n",
        "&eos\n  a = 2*3*4\n/\n",
        "&eos\n  m(:, 1:2) = 1, 2\n/\n",
        "&eos\n  m(1, 2) = 1, 2\n/\n",
        "&eos\n  a = \n/\n",
        "&eos\n  a = b = 1\n/\n",
        "&eos\n  a = 'unterminated\n/\n",
        "&eos\n  a = 1\n",
        "&end\n",
        "&eos\n  a = 1 / extra\n",
        "&eos\n  a = 1 &end extra\n",
        "&eos\n  &bad_token = 1\n/\n",
    ];
    for case in invalid_cases {
        assert!(
            parse_namelist_text(case).is_err(),
            "expected ParseError for:\n{case}"
        );
    }

    let trailing_slash = parse_namelist_text("&eos\n  a = 1 / extra\n")
        .expect_err("expected trailing slash error");
    assert!(
        trailing_slash
            .to_string()
            .contains("unexpected trailing tokens after '/'"),
        "got: {trailing_slash}"
    );
    let trailing_end = parse_namelist_text("&eos\n  a = 1 &end extra\n")
        .expect_err("expected trailing &end error");
    assert!(
        trailing_end
            .to_string()
            .contains("unexpected trailing tokens after '&end'"),
        "got: {trailing_end}"
    );
}

#[test]
fn test_validate_namelist_block_with_expanded_slices() {
    let defaults_src = "\
  use_PC = .false.
  x_ctrl(:) = 0.0d0
";
    let schema = parse_defaults_text(defaults_src).unwrap();

    let good = parse_namelist_text(
        "&controls\n  x_ctrl(1:4) = 1.0d0, 2.0d0, 3.0d0, 4.0d0\n/\n",
    )
    .unwrap();
    assert!(validate_namelist_block(&good[0], &schema, false).is_ok());

    let bad_type = parse_namelist_text(
        "&controls\n  x_ctrl(1:3) = 1.0d0,\n  'bad',\n  3.0d0\n/\n",
    )
    .unwrap();
    let err = validate_namelist_block(&bad_type[0], &schema, false)
        .expect_err("expected validation type mismatch on element 2");
    assert!(err.to_string().contains("line 3"), "got: {err}");

    let bad_scalar =
        parse_namelist_text("&controls\n  use_PC(1:2) = .true., .false.\n/\n")
            .unwrap();
    assert!(validate_namelist_block(&bad_scalar[0], &schema, false).is_err());
}
