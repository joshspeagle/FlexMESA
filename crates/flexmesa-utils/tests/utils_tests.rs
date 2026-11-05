// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration tests verifying `flexmesa-utils` against upstream MESA golden
//! output (`utils/test/test_output`).

use flexmesa_utils::{
    is_bad, is_bad_slice, remove_underbars, replace_bad_nums, split_line, tokenize, IntegerDict,
    IntegerIdict, Token,
};

#[test]
fn test_integer_dict_matches_upstream_mesa() {
    let mut dict = IntegerDict::new();
    let keys = [
        "he4", "c12", "n14", "o16", "ne20", "mg24", "si28", "s32", "ar36", "ca40", "ti44", "cr48",
        "fe52", "fe54", "fe56", "ni56", "neut", "h1", "prot",
    ];
    for (idx, &k) in keys.iter().enumerate() {
        let value = i64::try_from(idx + 1).expect("index fits in i64");
        let duplicate = dict.define_and_report_duplicates(k, value);
        assert!(!duplicate);
    }

    assert!(dict.define_and_report_duplicates("o16", 6));
    assert!(dict.define_and_report_duplicates("ne20", 7));
    dict.create_hash();
    assert!(dict.define_and_report_duplicates("o16", 4));
    assert!(dict.define_and_report_duplicates("ne20", 5));

    for (idx, &k) in keys.iter().enumerate() {
        let expected = i64::try_from(idx + 1).expect("index fits in i64");
        assert_eq!(dict.lookup(k), Some(expected));
    }
    assert_eq!(dict.lookup("missing"), None);
    assert_eq!(dict.len(), keys.len());
}

#[test]
fn test_integer_idict_matches_upstream_mesa() {
    let mut idict = IntegerIdict::new();
    let k1s = [
        4, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 54, 56, 56, 1, 1, 1,
    ];
    let k2s = [
        2, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 26, 26, 28, 0, 1, 2,
    ];
    for (idx, (&k1, &k2)) in k1s.iter().zip(k2s.iter()).enumerate() {
        let value = i64::try_from(idx + 1).expect("index fits in i64");
        let duplicate = idict.define_and_report_duplicates(k1, k2, value);
        assert!(!duplicate);
    }

    assert!(idict.define_and_report_duplicates(16, 8, 6));
    assert!(idict.define_and_report_duplicates(20, 10, 7));
    idict.create_hash();
    assert!(idict.define_and_report_duplicates(16, 8, 4));
    assert!(idict.define_and_report_duplicates(20, 10, 5));

    for (idx, (&k1, &k2)) in k1s.iter().zip(k2s.iter()).enumerate() {
        let expected = i64::try_from(idx + 1).expect("index fits in i64");
        assert_eq!(idict.lookup(k1, k2), Some(expected));
    }
    assert_eq!(idict.lookup(99, 99), None);
    assert_eq!(idict.len(), k1s.len());
}

#[test]
fn test_token_stream_matches_upstream_mesa_token_txt() {
    let input = "\
!test input for token_read

add_isos(
   ih1, ihe4, & ! ignore & and comment
   ic12
   )

'string1'  \"string2\"

add_reactions(rc12pg, rc12ap)
";
    let tokens = tokenize(input);
    let expected = vec![
        Token::Name("add_isos".to_owned()),
        Token::LeftParen,
        Token::Name("ih1".to_owned()),
        Token::Comma,
        Token::Name("ihe4".to_owned()),
        Token::Comma,
        Token::Name("ic12".to_owned()),
        Token::RightParen,
        Token::String("string1".to_owned()),
        Token::String("string2".to_owned()),
        Token::Name("add_reactions".to_owned()),
        Token::LeftParen,
        Token::Name("rc12pg".to_owned()),
        Token::Comma,
        Token::Name("rc12ap".to_owned()),
        Token::RightParen,
        Token::Eof,
    ];
    assert_eq!(tokens, expected);
}

#[test]
fn test_finiteness_guards_and_string_helpers() {
    assert!(!is_bad(0.0));
    assert!(!is_bad(-1.0e99));
    assert!(is_bad(f64::NAN));
    assert!(is_bad(f64::INFINITY));
    assert!(is_bad(f64::NEG_INFINITY));

    let xs = [1.0, f64::NAN, -2.5, f64::INFINITY];
    assert!(is_bad_slice(&xs));
    let filtered = replace_bad_nums(&xs, -1.0e99);
    assert!(!is_bad_slice(&filtered));
    assert_eq!(filtered, vec![1.0, -1.0e99, -2.5, -1.0e99]);

    assert_eq!(remove_underbars("star_mass_msun   "), "star mass msun");
    assert_eq!(
        split_line("alpha   beta  gamma delta", 3),
        vec!["alpha", "beta", "gamma"]
    );
}
