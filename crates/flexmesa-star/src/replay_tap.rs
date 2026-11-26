// SPDX-License-Identifier: GPL-3.0-or-later
//! Opt-in argument tap for record/replay comparison with MESA 26.4.1
//! (`#552`).
//!
//! Disabled unless the environment variable `FLEXMESA_REPLAY_TAP` names an
//! output file. When enabled, the convection code writes the records the
//! MESA overlay of `benchmarks/fortran_mesa/replay/` writes, in the same
//! format and the same lane layout: one `GETR` record per `Get_results`
//! evaluation (the arguments `FlexMESA` assembled and the results), and one
//! `TDC` or `MLT` record per kernel call. `flexmesa-replay compare` then
//! matches them with a MESA capture call by call.
//!
//! The tap reads values and writes text; it assigns nothing the engine
//! reads, so a run with the tap on is the same run as with it off. When off,
//! each call site costs one initialized `OnceLock` read.
//!
//! Only the first `FLEXMESA_REPLAY_TAP_MODELS` (default 1) distinct model
//! numbers are written, starting at `FLEXMESA_REPLAY_TAP_FIRST` when set,
//! as in the MESA overlay.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write as _};
use std::sync::{Mutex, OnceLock};

use flexmesa_replay::record::{encode_ad, encode_real, MesaAd, NOT_AVAILABLE};
use flexmesa_replay::turb::{GETR_SCHEMA, MLT_SCHEMA, TDC_SCHEMA};
use flexmesa_turb::{
    MltInputs, MltOption, MltParams, TdcInputs, TdcOutput, TdcParams,
    TurbOutput,
};

use crate::model::StarModel;
use crate::star_ad::{StarAdOrder1, NUM_STAR_AD_VARS};

/// Environment variable naming the tap's output file.
pub const TAP_ENV: &str = "FLEXMESA_REPLAY_TAP";

/// MESA `d1Array` lane (1-based, `star_data/public/star_data_def.inc:255-288`)
/// of each [`StarAdOrder1`] slot: `lnd`, `lnT`, `lnR`, `L`, `v`, the unused
/// `omega` slots (written to MESA's `w_div_wc` lanes), `w`, `Hp`.
const MESA_LANE: [usize; NUM_STAR_AD_VARS] = [
    1, 2, 3, 4, 5, 6, 10, 11, 12, 16, 17, 18, 13, 14, 15, 22, 23, 24, 7, 8, 9,
    19, 20, 21,
];

struct Tap {
    out: BufWriter<File>,
    nmax: usize,
    first_model: usize,
    nseen: usize,
    last_model: Option<usize>,
}

static TAP: OnceLock<Option<Mutex<Tap>>> = OnceLock::new();

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

fn tap() -> Option<&'static Mutex<Tap>> {
    TAP.get_or_init(|| {
        let path = std::env::var(TAP_ENV).ok()?;
        if path.trim().is_empty() {
            return None;
        }
        // A tap that was asked for but cannot be opened is an error the
        // user must see, not a silently empty capture.
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.trim())
            .unwrap_or_else(|e| panic!("{TAP_ENV}={path}: cannot open: {e}"));
        let mut out = BufWriter::new(file);
        let _ = writeln!(out, "RUN");
        let _ = writeln!(out, "SCHEMA TDC {}", squash(TDC_SCHEMA));
        let _ = writeln!(out, "SCHEMA MLT {}", squash(MLT_SCHEMA));
        let _ = writeln!(out, "SCHEMA GETR {}", squash(GETR_SCHEMA));
        Some(Mutex::new(Tap {
            out,
            nmax: env_usize("FLEXMESA_REPLAY_TAP_MODELS").unwrap_or(1),
            first_model: env_usize("FLEXMESA_REPLAY_TAP_FIRST").unwrap_or(0),
            nseen: 0,
            last_model: None,
        }))
    })
    .as_ref()
}

fn squash(schema: &str) -> String {
    schema.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// True when the tap is enabled and `model`'s current model number is
/// inside the window. Advances the window on a new model number.
#[must_use]
pub(crate) fn active(model: &StarModel) -> bool {
    let Some(m) = tap() else { return false };
    let Ok(mut t) = m.lock() else { return false };
    let n = model.model_number;
    if n < t.first_model {
        return false;
    }
    if t.last_model != Some(n) {
        if t.nseen >= t.nmax {
            return false;
        }
        t.nseen += 1;
        t.last_model = Some(n);
    }
    true
}

fn emit(line: &str) {
    if let Some(m) = tap() {
        if let Ok(mut t) = m.lock() {
            let _ = writeln!(t.out, "{line}");
            let _ = t.out.flush();
        }
    }
}

/// `x` in MESA's 33-lane layout, encoded.
fn ad(x: &StarAdOrder1) -> String {
    let mut d1 = [0.0_f64; flexmesa_replay::record::STAR_AD_LANES];
    for (slot, lane) in MESA_LANE.iter().enumerate() {
        d1[lane - 1] = x.d1[slot];
    }
    encode_ad(&MesaAd::new(x.val, d1))
}

fn logical(x: bool) -> &'static str {
    if x {
        "T"
    } else {
        "F"
    }
}

/// MESA's `MLT_option` string for `opt`.
fn option_name(opt: MltOption) -> &'static str {
    match opt {
        MltOption::Cox => "Cox",
        MltOption::Tdc => "TDC",
        MltOption::Henyey => "Henyey",
        MltOption::Ml1 => "ML1",
        MltOption::Ml2 => "ML2",
        MltOption::Mihalas => "Mihalas",
    }
}

/// Context fields `site id model solver_iter k` (MESA's 1-based `k`).
fn context(family: &str, site: u8, model: &StarModel, k: usize) -> String {
    format!(
        "{family} {site} 1 {} {} {}",
        model.model_number,
        model.aux.solver_iter,
        k + 1
    )
}

/// Writes a `TDC` record for a `set_tdc` call at face `k` (0-based).
pub(crate) fn tdc(
    model: &StarModel,
    k: usize,
    site: u8,
    p: &TdcParams,
    i: &TdcInputs<StarAdOrder1>,
    o: &TdcOutput<StarAdOrder1>,
) {
    let r = encode_real;
    let f = [
        context("TDC", site, model, k),
        // MESA passes `s% tdc_num_iters(k)` in; `set_tdc` has no such input.
        NOT_AVAILABLE.to_string(),
        r(p.conv_vel_start),
        r(p.mixing_length_alpha),
        r(p.tdc_alpha_d),
        r(p.tdc_alpha_r),
        r(p.tdc_alpha_pt),
        r(p.dt),
        r(p.cgrav),
        r(p.m),
        r(p.scale),
        ad(&i.chi_t),
        ad(&i.chi_rho),
        ad(&i.gradr),
        ad(&i.r),
        ad(&i.p),
        ad(&i.t),
        ad(&i.rho),
        ad(&i.dv),
        ad(&i.cp),
        ad(&i.opacity),
        ad(&i.scale_height),
        ad(&i.gradl),
        ad(&i.grada),
        r(p.max_conv_vel),
        ad(&i.eq_div_w),
        ad(&i.grav),
        logical(p.include_mlt_corr_to_tdc).to_string(),
        r(p.tdc_alpha_c),
        r(p.tdc_alpha_s),
        logical(p.use_tdc_enthalpy_flux_limiter).to_string(),
        ad(&i.energy),
        (o.output.mixing_type as i64).to_string(),
        ad(&o.output.conv_vel),
        ad(&o.output.d),
        ad(&o.output.y_face),
        ad(&o.output.grad_t),
        o.num_iters.to_string(),
        "0".to_string(),
    ];
    emit(&f.join(" "));
}

/// Writes an `MLT` record for a `set_mlt` call at face `k` (0-based).
pub(crate) fn mlt(
    model: &StarModel,
    k: usize,
    site: u8,
    p: &MltParams,
    i: &MltInputs<StarAdOrder1>,
    o: &TurbOutput<StarAdOrder1>,
) {
    let r = encode_real;
    let f = [
        context("MLT", site, model, k),
        option_name(p.option).to_string(),
        r(p.mixing_length_alpha),
        r(p.henyey_nu),
        r(p.henyey_y),
        ad(&i.chi_t),
        ad(&i.chi_rho),
        ad(&i.cp),
        ad(&i.grav),
        ad(&i.lambda),
        ad(&i.rho),
        ad(&i.p),
        ad(&i.t),
        ad(&i.opacity),
        ad(&i.gradr),
        ad(&i.grada),
        ad(&i.gradl),
        r(p.max_conv_vel),
        ad(&o.gamma),
        ad(&o.grad_t),
        ad(&o.y_face),
        ad(&o.conv_vel),
        ad(&o.d),
        (o.mixing_type as i64).to_string(),
        "0".to_string(),
    ];
    emit(&f.join(" "));
}

/// Arguments and results of one `Get_results` evaluation, as assembled by
/// `do1_mlt_2` (`star/private/turb_support.f90:188-192`).
pub(crate) struct GetResultsArgs<'a> {
    pub mlt_option: MltOption,
    pub r: &'a StarAdOrder1,
    pub l: &'a StarAdOrder1,
    pub t: &'a StarAdOrder1,
    pub p: &'a StarAdOrder1,
    pub opacity: &'a StarAdOrder1,
    pub rho: &'a StarAdOrder1,
    /// `None` when the start-of-step density is not available.
    pub dv: Option<StarAdOrder1>,
    pub chi_rho: &'a StarAdOrder1,
    pub chi_t: &'a StarAdOrder1,
    pub cp: &'a StarAdOrder1,
    pub gradr: &'a StarAdOrder1,
    pub grada: &'a StarAdOrder1,
    pub scale_height: &'a StarAdOrder1,
    pub iso: usize,
    pub xh1: f64,
    pub cgrav: f64,
    pub m: f64,
    pub gradl_composition_term: f64,
    pub mixing_length_alpha: f64,
    pub alpha_semiconvection: f64,
    pub thermohaline_coeff: f64,
    /// `None` when not available.
    pub energy: Option<StarAdOrder1>,
    pub out: &'a TurbOutput<StarAdOrder1>,
}

/// Writes a `GETR` record for face `k` (0-based).
pub(crate) fn get_results(model: &StarModel, k: usize, a: &GetResultsArgs) {
    let r = encode_real;
    let opt = |x: &Option<StarAdOrder1>| {
        x.as_ref().map_or_else(|| NOT_AVAILABLE.to_string(), ad)
    };
    let f = [
        context("GETR", 5, model, k),
        option_name(a.mlt_option).to_string(),
        ad(a.r),
        ad(a.l),
        ad(a.t),
        ad(a.p),
        ad(a.opacity),
        ad(a.rho),
        opt(&a.dv),
        ad(a.chi_rho),
        ad(a.chi_t),
        ad(a.cp),
        ad(a.gradr),
        ad(a.grada),
        ad(a.scale_height),
        a.iso.to_string(),
        r(a.xh1),
        r(a.cgrav),
        r(a.m),
        r(a.gradl_composition_term),
        r(a.mixing_length_alpha),
        r(a.alpha_semiconvection),
        r(a.thermohaline_coeff),
        opt(&a.energy),
        (a.out.mixing_type as i64).to_string(),
        ad(&a.out.grad_t),
        ad(&a.out.y_face),
        ad(&a.out.conv_vel),
        ad(&a.out.d),
        ad(&a.out.gamma),
        "0".to_string(),
    ];
    emit(&f.join(" "));
}
