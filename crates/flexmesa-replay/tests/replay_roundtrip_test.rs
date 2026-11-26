// SPDX-License-Identifier: GPL-3.0-or-later
//! Self-consistency tests of the record/replay pipeline (`#552`).
//!
//! These tests check the capture format and the comparison machinery, not
//! agreement with MESA: the "captured" outputs are produced here by the
//! `FlexMESA` kernels themselves (a regression snapshot of the pipeline, not
//! a reference). Agreement with MESA is measured by replaying real captures
//! from the overlay in `benchmarks/fortran_mesa/replay/`.

#![forbid(unsafe_code)]
#![allow(clippy::float_cmp, clippy::unreadable_literal)]

use std::fmt::Write as _;
use std::io::Cursor;

use flexmesa_replay::record::{MesaAd, STAR_AD_LANES};
use flexmesa_replay::turb::{MLT_SCHEMA, TDC_SCHEMA};
use flexmesa_turb::{
    set_mlt, set_tdc, MltInputs, MltOption, MltParams, TdcInputs, TdcParams,
};

/// Encodes a real as the overlay does (`Z16.16`).
fn r(x: f64) -> String {
    format!("{:016X}", x.to_bits())
}

/// Encodes an auto-diff value as the overlay does: the value, then every
/// lane whose bit pattern is not `+0.0`.
fn a(x: &MesaAd) -> String {
    let mut s = r(x.val);
    for (i, d) in x.d1.iter().enumerate() {
        if d.to_bits() != 0 {
            write!(s, "/{}:{}", i + 1, r(*d)).unwrap();
        }
    }
    s
}

/// A value with partials in the given 1-based MESA lanes.
fn ad(val: f64, lanes: &[(usize, f64)]) -> MesaAd {
    let mut d1 = [0.0; STAR_AD_LANES];
    for &(lane, d) in lanes {
        d1[lane - 1] = d;
    }
    MesaAd::new(val, d1)
}

/// Thermodynamic face values of a convective envelope cell, synthetic
/// (rounded from the scale of a 1 Msun model), with partials in the
/// `lnd`/`lnT` lanes of cells `k-1` and `k` and in `lnR_00` and `L_00`.
fn tdc_case() -> (TdcParams, TdcInputs<MesaAd>) {
    let th = |v: f64| {
        ad(
            v,
            &[(1, 0.1 * v), (2, 0.2 * v), (4, -0.3 * v), (5, 0.4 * v)],
        )
    };
    let params = TdcParams {
        conv_vel_start: 9.5e3,
        mixing_length_alpha: 1.6,
        tdc_alpha_d: 1.0,
        tdc_alpha_r: 0.0,
        tdc_alpha_pt: 0.0,
        tdc_alpha_c: 1.0,
        tdc_alpha_s: 1.0,
        dt: 1.4e13,
        cgrav: 6.6743e-8,
        m: 1.99e33,
        scale: 2.8e33,
        max_conv_vel: 1.0e99,
        include_mlt_corr_to_tdc: true,
        use_tdc_enthalpy_flux_limiter: false,
    };
    let rr = ad(6.2e10, &[(11, 6.2e10)]);
    let inp = TdcInputs {
        chi_t: th(1.05),
        chi_rho: th(0.98),
        gradr: ad(223.5, &[(1, 1.0), (5, -2.0), (17, 1.0e-31)]),
        r: rr,
        p: th(4.0e11),
        t: ad(3.05e5, &[(4, 1.5e5), (5, 1.5e5)]),
        rho: ad(1.0e-2, &[(1, 5.0e-3), (2, 5.0e-3)]),
        dv: ad(1.68, &[(1, -50.0), (2, -50.0)]),
        cp: th(3.6e8),
        opacity: th(860.0),
        scale_height: th(1.15e9),
        gradl: th(0.388),
        grada: th(0.388),
        eq_div_w: MesaAd::constant(0.0),
        grav: MesaAd::constant(6.6743e-8 * 1.99e33) / (rr * rr),
        energy: th(7.3e13),
    };
    (params, inp)
}

fn tdc_line(params: &TdcParams, inp: &TdcInputs<MesaAd>) -> String {
    let out = set_tdc(params, inp).expect("synthetic TDC call converges");
    let o = out.output;
    let b = |x: bool| if x { "T" } else { "F" };
    [
        "TDC 1 1 2 1 400 0".to_string(),
        r(params.conv_vel_start),
        r(params.mixing_length_alpha),
        r(params.tdc_alpha_d),
        r(params.tdc_alpha_r),
        r(params.tdc_alpha_pt),
        r(params.dt),
        r(params.cgrav),
        r(params.m),
        r(params.scale),
        a(&inp.chi_t),
        a(&inp.chi_rho),
        a(&inp.gradr),
        a(&inp.r),
        a(&inp.p),
        a(&inp.t),
        a(&inp.rho),
        a(&inp.dv),
        a(&inp.cp),
        a(&inp.opacity),
        a(&inp.scale_height),
        a(&inp.gradl),
        a(&inp.grada),
        r(params.max_conv_vel),
        a(&inp.eq_div_w),
        a(&inp.grav),
        b(params.include_mlt_corr_to_tdc).to_string(),
        r(params.tdc_alpha_c),
        r(params.tdc_alpha_s),
        b(params.use_tdc_enthalpy_flux_limiter).to_string(),
        a(&inp.energy),
        (o.mixing_type as i64).to_string(),
        a(&o.conv_vel),
        a(&o.d),
        a(&o.y_face),
        a(&o.grad_t),
        out.num_iters.to_string(),
        "0".to_string(),
    ]
    .join(" ")
}

fn mlt_line() -> String {
    let (_, t) = tdc_case();
    let params = MltParams {
        option: MltOption::Cox,
        mixing_length_alpha: 1.6,
        henyey_nu: 8.0,
        henyey_y: 1.0 / 3.0,
        max_conv_vel: 1.0e99,
    };
    let inp = MltInputs {
        chi_t: t.chi_t,
        chi_rho: t.chi_rho,
        cp: t.cp,
        grav: t.grav,
        lambda: MesaAd::constant(1.6) * t.scale_height,
        rho: t.rho,
        p: t.p,
        t: t.t,
        opacity: t.opacity,
        gradr: t.gradr,
        grada: t.grada,
        gradl: t.gradl,
    };
    let o = set_mlt(&params, &inp).expect("synthetic MLT call");
    [
        "MLT 3 1 2 1 400 Cox".to_string(),
        r(params.mixing_length_alpha),
        r(params.henyey_nu),
        r(params.henyey_y),
        a(&inp.chi_t),
        a(&inp.chi_rho),
        a(&inp.cp),
        a(&inp.grav),
        a(&inp.lambda),
        a(&inp.rho),
        a(&inp.p),
        a(&inp.t),
        a(&inp.opacity),
        a(&inp.gradr),
        a(&inp.grada),
        a(&inp.gradl),
        r(params.max_conv_vel),
        a(&o.gamma),
        a(&o.grad_t),
        a(&o.y_face),
        a(&o.conv_vel),
        a(&o.d),
        (o.mixing_type as i64).to_string(),
        "0".to_string(),
    ]
    .join(" ")
}

fn capture(records: &[String]) -> String {
    let mut s = String::from("# case: synthetic\nRUN\n");
    writeln!(s, "SCHEMA TDC {TDC_SCHEMA}").unwrap();
    writeln!(s, "SCHEMA MLT {MLT_SCHEMA}").unwrap();
    for rec in records {
        s.push_str(rec);
        s.push('\n');
    }
    s
}

/// Flips the lowest bit of the value of field number `field` (0-based,
/// counting the family tag) of a record line.
fn flip_value_bit(line: &str, field: usize) -> String {
    let mut toks: Vec<String> =
        line.split(' ').map(ToString::to_string).collect();
    let (val, rest) = toks[field]
        .split_once('/')
        .map_or((toks[field].clone(), String::new()), |(v, r)| {
            (v.to_string(), format!("/{r}"))
        });
    let bits = u64::from_str_radix(&val, 16).unwrap() ^ 1;
    toks[field] = format!("{bits:016X}{rest}");
    toks.join(" ")
}

#[test]
fn self_generated_tdc_and_mlt_records_replay_identically() {
    let (params, inp) = tdc_case();
    let text = capture(&[tdc_line(&params, &inp), mlt_line()]);
    let summary = flexmesa_replay::replay(Cursor::new(text)).unwrap();
    assert_eq!(summary.families["TDC"].calls, 1);
    assert_eq!(summary.families["MLT"].calls, 1);
    assert!(summary.all_identical());
    assert_eq!(summary.header, vec!["case: synthetic".to_string()]);
}

#[test]
fn one_ulp_change_in_a_captured_output_is_reported_with_field_and_bits() {
    let (params, inp) = tdc_case();
    let line = tdc_line(&params, &inp);
    // Token 0 of a record line is the family tag.
    let schema: Vec<&str> = TDC_SCHEMA.split_whitespace().collect();
    let field = 1 + schema.iter().position(|f| *f == "out_gradT:a").unwrap();
    let text = capture(&[flip_value_bit(&line, field)]);
    let summary = flexmesa_replay::replay(Cursor::new(text)).unwrap();
    let tdc = &summary.families["TDC"];
    assert_eq!((tdc.calls, tdc.identical), (1, 0));
    let (ctx, diff) = tdc.first.as_ref().unwrap();
    assert!(ctx.contains("model=2 solver_iter=1 k=400"), "{ctx}");
    assert_eq!(diff.mismatches.len(), 1);
    let m = &diff.mismatches[0];
    assert_eq!((m.field.as_str(), m.lane), ("gradT", None));
    assert_eq!(m.mesa.to_bits() ^ m.flexmesa.to_bits(), 1);
}

#[test]
fn capture_whose_schema_differs_from_the_tool_is_rejected() {
    let (params, inp) = tdc_case();
    let text = capture(&[tdc_line(&params, &inp)])
        .replace("out_ierr:i\nSCHEMA MLT", "out_ierr:r\nSCHEMA MLT");
    let err = flexmesa_replay::replay(Cursor::new(text)).unwrap_err();
    assert!(err.0.contains("TDC schema"), "{err}");
}

#[test]
fn record_of_a_family_without_replay_is_rejected() {
    let text = "RUN\nSCHEMA KAP k:i\nKAP 1\n";
    let err = flexmesa_replay::replay(Cursor::new(text)).unwrap_err();
    assert!(err.0.contains("no replay for record family `KAP`"), "{err}");
}

#[test]
fn negative_zero_partial_is_kept_and_distinguished_from_positive_zero() {
    let neg = ad(1.0, &[(7, -0.0)]);
    assert_eq!(a(&neg), "3FF0000000000000/7:8000000000000000");
    let mut diff = flexmesa_replay::compare::CallDiff::default();
    diff.ad("x", &neg, &ad(1.0, &[]));
    assert_eq!(diff.mismatches.len(), 1);
    assert_eq!(diff.mismatches[0].lane, Some(7));
}
