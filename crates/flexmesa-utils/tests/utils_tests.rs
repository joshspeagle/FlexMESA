// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration and property-based tests verifying `flexmesa-utils` against
//! upstream MESA golden output (`utils/test/test_output`).
//!
//! Covers:
//! - String-keyed (`IntegerDict`) and integer-pair-keyed (`IntegerIdict`)
//!   dictionaries with duplicate-key reporting and trailing-space trimming.
//! - The `.net` reaction-network token stream lexer (`Tokenizer`, `tokenize`)
//!   against `utils/test/token.txt` and inline comment/continuation edge cases.
//! - IEEE-754 finiteness guards (`is_bad`, `is_bad_slice`, `replace_bad_nums`)
//!   and property-based fuzzing with `proptest`.

use flexmesa_utils::{
    is_bad, is_bad_slice, remove_underbars, replace_bad_nums, split_line,
    tokenize, IntegerDict, IntegerIdict, Token, Tokenizer,
};
use proptest::prelude::*;

fn verify_upstream_utils_files(expected: &[Token]) {
    let mesa_env = std::env::var_os("MESA_DIR");
    let mesa_root = mesa_env
        .as_ref()
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("XDG_CACHE_HOME")
                .filter(|v| !v.is_empty())
                .map(|p| std::path::PathBuf::from(p).join("flexmesa/mesa_test_inputs"))
        })
        .or_else(|| {
            std::env::var_os("HOME")
                .filter(|v| !v.is_empty())
                .map(|p| std::path::PathBuf::from(p).join(".cache/flexmesa/mesa_test_inputs"))
        });
    let Some(root) = mesa_root else {
        return;
    };
    let upstream_token_path = root.join("utils/test/token.txt");
    let upstream_output_path = root.join("utils/test/test_output");
    if mesa_env.as_ref().is_some_and(|v| !v.is_empty()) {
        assert!(upstream_token_path.is_file(), "missing {}", upstream_token_path.display());
        assert!(upstream_output_path.is_file(), "missing {}", upstream_output_path.display());
    }
    if upstream_token_path.is_file() {
        let upstream_txt = std::fs::read_to_string(&upstream_token_path)
            .expect("read utils/test/token.txt");
        assert_eq!(tokenize(&upstream_txt), expected);
    }
    if upstream_output_path.is_file() {
        let out_txt = std::fs::read_to_string(&upstream_output_path)
            .expect("read utils/test/test_output");
        let token_lines: Vec<String> = out_txt
            .lines()
            .map(str::trim)
            .filter(|l| l.ends_with("_token") || l.contains("_token "))
            .map(str::to_owned)
            .collect();
        let formatted: Vec<String> = expected
            .iter()
            .map(|t| match t {
                Token::Name(s) => format!("name_token {:11} {}", s.len(), s),
                Token::String(s) => format!("string_token {:11} {}", s.len(), s),
                Token::LeftParen => "left_paren_token".to_owned(),
                Token::RightParen => "right_paren_token".to_owned(),
                Token::Comma => "comma_token".to_owned(),
                Token::Eof => "eof_token".to_owned(),
            })
            .collect();
        assert_eq!(formatted, token_lines);
    }
}

/// Verify `IntegerDict` insertion, duplicate reporting, hash creation hook,
/// and key lookup against `test_dict` in `utils/test/src/test_utils.f90`.
#[test]
fn test_integer_dict_matches_upstream_mesa() {
    let mut dict = IntegerDict::new();
    assert!(dict.is_empty());
    let keys = [
        "he4", "c12", "n14", "o16", "ne20", "mg24", "si28", "s32", "ar36",
        "ca40", "ti44", "cr48", "fe52", "fe54", "fe56", "ni56", "neut", "h1",
        "prot",
    ];
    for (&k, value) in keys.iter().zip(1_i64..) {
        let duplicate = dict.define_and_report_duplicates(k, value);
        assert!(!duplicate);
    }

    // Redefining existing keys reports duplicate = true and updates values.
    assert!(dict.define_and_report_duplicates("o16", 6));
    assert!(dict.define_and_report_duplicates("ne20", 7));
    dict.create_hash();
    assert!(dict.define_and_report_duplicates("o16", 4));
    assert!(dict.define_and_report_duplicates("ne20", 5));

    for (&k, expected) in keys.iter().zip(1_i64..) {
        assert_eq!(dict.lookup(k), Some(expected));
    }
    assert_eq!(dict.lookup("missing"), None);
    assert_eq!(dict.len(), keys.len());
}

/// Verify `IntegerIdict` insertion, duplicate reporting, and pair lookup
/// against `test_idict` in `utils/test/src/test_utils.f90`.
#[test]
fn test_integer_idict_matches_upstream_mesa() {
    let mut idict = IntegerIdict::new();
    assert!(idict.is_empty());
    let k1s = [
        4, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 54, 56, 56, 1, 1, 1,
    ];
    let k2s = [
        2, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 26, 26, 28, 0, 1, 2,
    ];
    for ((&k1, &k2), value) in k1s.iter().zip(k2s.iter()).zip(1_i64..) {
        let duplicate = idict.define_and_report_duplicates(k1, k2, value);
        assert!(!duplicate);
    }

    // Redefining existing integer pairs reports duplicate = true.
    assert!(idict.define_and_report_duplicates(16, 8, 6));
    assert!(idict.define_and_report_duplicates(20, 10, 7));
    idict.create_hash();
    assert!(idict.define_and_report_duplicates(16, 8, 4));
    assert!(idict.define_and_report_duplicates(20, 10, 5));

    for ((&k1, &k2), expected) in k1s.iter().zip(k2s.iter()).zip(1_i64..) {
        assert_eq!(idict.lookup(k1, k2), Some(expected));
    }
    assert_eq!(idict.lookup(99, 99), None);
    assert_eq!(idict.len(), k1s.len());
}

/// Verify that `tokenize` reproduces the exact token sequence from upstream
/// `utils/test/token.txt` and `utils/test/test_output`, and stops identifiers
/// cleanly at adjacent `!` comments or `&` markers.
#[test]
fn test_token_stream_matches_upstream_mesa_token_txt() {
    let token_txt = "\
\n      ! test token reader
      
      file = \"token.txt\"
      file2 = 'token2.txt'
      
      add_isos(ih1, ihe4, ic12)
      
      add_reactions(
          rc12pg,           ! c12(p,g)n13
          rc12ap            ! c12(a,p)n15
          )
         ";
    let expected = vec![
        Token::Name("file".to_owned()),
        Token::Name("=".to_owned()),
        Token::String("token.txt".to_owned()),
        Token::Name("file2".to_owned()),
        Token::Name("=".to_owned()),
        Token::String("token2.txt".to_owned()),
        Token::Name("add_isos".to_owned()),
        Token::LeftParen,
        Token::Name("ih1".to_owned()),
        Token::Comma,
        Token::Name("ihe4".to_owned()),
        Token::Comma,
        Token::Name("ic12".to_owned()),
        Token::RightParen,
        Token::Name("add_reactions".to_owned()),
        Token::LeftParen,
        Token::Name("rc12pg".to_owned()),
        Token::Comma,
        Token::Name("rc12ap".to_owned()),
        Token::RightParen,
        Token::Eof,
    ];
    let tokens = tokenize(token_txt);
    assert_eq!(tokens, expected);

    let mut tz = Tokenizer::new(token_txt);
    for exp in &expected {
        assert_eq!(&tz.next_token(), exp);
    }
    assert_eq!(tz.next_token(), Token::Eof);

    // Verify directly against upstream MESA utils/test/token.txt and test_output
    verify_upstream_utils_files(&expected);

    // Also test inline comment and & continuation stripping
    let inline_input = "add_isos(ih1, ihe4, & ! ignore &\n ic12!inline\n)";
    assert_eq!(
        tokenize(inline_input),
        vec![
            Token::Name("add_isos".to_owned()),
            Token::LeftParen,
            Token::Name("ih1".to_owned()),
            Token::Comma,
            Token::Name("ihe4".to_owned()),
            Token::Comma,
            Token::Name("ic12".to_owned()),
            Token::RightParen,
            Token::Eof,
        ]
    );
}

/// Verify IEEE-754 finiteness checks (`is_bad`, `is_bad_slice`,
/// `replace_bad_nums`) and string manipulation helpers (`remove_underbars`,
/// `split_line`).
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

proptest! {
    /// Property test verifying that `replace_bad_nums` produces a finite slice
    /// for any combination of finite values, `NaN`, and infinities, and that
    /// `IntegerDict` normalizes trailing spaces consistently.
    #[test]
    fn prop_replace_bad_nums_and_dict_normalization(
        finite_vals in prop::collection::vec(-1.0e50_f64..1.0e50_f64, 1..32),
        key in "[a-z][a-z0-9_]{0,15}",
        val1 in any::<i64>(),
        val2 in any::<i64>(),
    ) {
        let mut mixed = finite_vals.clone();
        mixed.push(f64::NAN);
        mixed.push(f64::INFINITY);
        prop_assert!(is_bad_slice(&mixed));

        let cleaned = replace_bad_nums(&mixed, 0.0);
        prop_assert!(!is_bad_slice(&cleaned));
        prop_assert_eq!(cleaned.len(), mixed.len());

        let mut dict = IntegerDict::new();
        let padded = format!("{key}   ");
        prop_assert!(!dict.define_and_report_duplicates(&padded, val1));
        prop_assert_eq!(dict.lookup(&key), Some(val1));
        prop_assert!(dict.define_and_report_duplicates(&key, val2));
        prop_assert_eq!(dict.lookup(&padded), Some(val2));
    }
}
