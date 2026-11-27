// SPDX-License-Identifier: GPL-3.0-or-later
#![forbid(unsafe_code)]
//! Unit invariant tests for `flexmesa-adipls` issue #171 fixes.

use flexmesa_adipls::{
    order, parse_adipls_pruned_in, rsnfit, sigout, stevft, takata, AdiFileUnit,
    AmdlAa, BcsDat, Cincnt, Cprcns, Csummm, RhsDat, RotDat, SigoutArgs,
    SigoutState, TapRecorder,
};
use flexmesa_const::STANDARD_CGRAV;
use flexmesa_math::sin;

const BASE_PRUNED_IN: &str = "\
2  'amdl.mesa'
4  '0'
9  'ttt.adipls.prt'
11 'agsm.mesa'
15 '0'
-1 ''
mod.osc.cst.int.out
,,,,,,,
,,1,,,,,,,,,,,,,,,,,,,,
   0,3,0,1,,,,,,,
    1,  220,   ,    1,10,,,,,,,,
,2,100,1000,,,,,,,,,,,,
,,,,,,,,,,,,,,,,
6.67430e-8
,,0,,
1,,,,,,,,,,,,,,,
5,0,0.99,,,,,,,,,,,,,,
,,,,,-1,,,,,,,,,,,,,,
9,10,,20,1,,,,,,,,
1,,,,,,,,,,,,,,,
0,,,,6,,,,,,
,,,,,-5000,100,,,,,,,,,,,,,,,,,,,,,,,,
";

fn uniform_grid(nn: usize) -> Vec<f64> {
    let denom =
        u16::try_from(nn.saturating_sub(1).max(1)).expect("grid size fits u16");
    (0..nn)
        .map(|k| {
            f64::from(u16::try_from(k).expect("idx fits u16"))
                / f64::from(denom)
        })
        .collect()
}

fn build_uniform_sphere_test_amdl(nn: usize) -> ([f64; 8], Vec<f64>, AmdlAa) {
    let x = uniform_grid(nn);

    let gamma1 = 5.0_f64 / 3.0_f64;
    let a1 = vec![1.0_f64; nn];
    let a2: Vec<f64> = x
        .iter()
        .map(|&xk| {
            let x2 = xk * xk;
            (2.0_f64 * 0.99_f64 * x2) / (gamma1 * (1.0_f64 - 0.99_f64 * x2))
        })
        .collect();
    let a3 = vec![gamma1; nn];
    let a4 = vec![0.0_f64; nn];
    let a5 = vec![3.0_f64; nn];
    let a6 = vec![0.0_f64; nn];
    let a10 = vec![1.0_f64; nn];

    let mut aa = AmdlAa::new_unset(nn);
    aa.set_row(1, a1).expect("set a1");
    aa.set_row(2, a2).expect("set a2");
    aa.set_row(3, a3).expect("set a3");
    aa.set_row(4, a4).expect("set a4");
    aa.set_row(5, a5).expect("set a5");
    aa.set_row(6, a6).expect("set a6");
    aa.set_row(10, a10).expect("set a10");

    let data = [
        1.989e33_f64,
        6.9599e10_f64,
        1.0_f64,
        1.0_f64,
        0.0_f64,
        0.0_f64,
        -1.0_f64,
        0.0_f64,
    ];
    (data, x, aa)
}

#[test]
fn test_rsnfit_one_sided_clamps_on_odd_and_even_steps() {
    // MESA `rsnfit.n.d.f:28-32`:
    //   odd  nftcnt: max0(10, nint(nfit - 0.05*nftcnt*nn))  (no nn-10 upper clamp)
    //   even nftcnt: min0(nint(nfit + 0.05*nftcnt*nn), nn-10) (no 10 lower clamp)
    let x101 = uniform_grid(101);

    // Odd step near the surface (`nn=101, nfit=98, nftcnt: 0 -> 1`):
    // `nint(98 - 5.05) = 93`; `max0(10, 93) = 93`; parity with `nibc=2` -> `94`.
    let mut nfit = 98_usize;
    let mut xfit1 = x101[nfit - 1];
    let mut nftcnt = 0_i32;
    rsnfit(&x101, 2, 101, &mut nfit, &mut xfit1, &mut nftcnt, 0);
    assert_eq!(nftcnt, 1);
    assert_eq!(nfit, 94);
    assert_eq!(xfit1.to_bits(), x101[93].to_bits());

    // Even step near the center (`nn=41, nfit=2, nftcnt: 1 -> 2`):
    // `nint(2 + 4.1) = 6`; `min0(6, 31) = 6`; parity with `nibc=2` -> `6`.
    let x41 = uniform_grid(41);
    nfit = 2;
    xfit1 = x41[nfit - 1];
    nftcnt = 1;
    rsnfit(&x41, 2, 41, &mut nfit, &mut xfit1, &mut nftcnt, 0);
    assert_eq!(nftcnt, 2);
    assert_eq!(nfit, 6);
    assert_eq!(xfit1.to_bits(), x41[5].to_bits());

    // Odd step lower clamp (`max0(10, nint(12 - 5.05)) = 10`, even vs `nibc=2`).
    nfit = 12;
    xfit1 = x101[nfit - 1];
    nftcnt = 0;
    rsnfit(&x101, 2, 101, &mut nfit, &mut xfit1, &mut nftcnt, 0);
    assert_eq!(nfit, 10);
    assert_eq!(xfit1.to_bits(), x101[9].to_bits());

    // Even step upper clamp (`min0(nint(88 + 10.1), 91) = 91`, parity -> `92`).
    nfit = 88;
    xfit1 = x101[nfit - 1];
    nftcnt = 1;
    rsnfit(&x101, 2, 101, &mut nfit, &mut xfit1, &mut nftcnt, 0);
    assert_eq!(nfit, 92);
    assert_eq!(xfit1.to_bits(), x101[91].to_bits());

    // `nftrcs != 0` preserves `[10, nn - 10]` bounds cleanly.
    nfit = 97;
    xfit1 = 0.96_f64;
    nftcnt = 1;
    rsnfit(&x101, 2, 101, &mut nfit, &mut xfit1, &mut nftcnt, 1);
    assert_eq!(nfit, 92);
    assert_eq!(xfit1.to_bits(), x101[91].to_bits());

    // Tiny grid (`nn=5, nibc=2`) never panics from `lo > hi` clamp or underflow.
    let x5 = uniform_grid(5);
    for &nftrcs in &[0_i32, 1_i32] {
        for &start_cnt in &[0_i32, 1_i32] {
            nfit = 3;
            xfit1 = x5[nfit - 1];
            nftcnt = start_cnt;
            rsnfit(&x5, 2, 5, &mut nfit, &mut xfit1, &mut nftcnt, nftrcs);
            assert!((2..=5).contains(&nfit));
        }
    }
}

#[test]
fn test_stevft_small_grid_alignment_and_zero_iasn_guard() {
    let input = parse_adipls_pruned_in(BASE_PRUNED_IN).expect("parse input");
    let cincnt = Cincnt::from_input(&input);

    // `stevft` (`stevft.n.d.f:95-108`) on `nn = 31, nibc = 2`:
    // `nn > 30` clamps `nev1 = nn - 30 = 1 < nibc`, which previously
    // panicked on `(*nev1 - nibc)` underflow.
    let (data31, x31, aa31) = build_uniform_sphere_test_amdl(31);
    let rhs31 = RhsDat {
        el: 1.0_f64,
        ell: 2.0_f64,
        alb: 1.0_f64,
        els: 2.0_f64 / 15.0_f64,
        el1: 0.0_f64,
        sig: 15.0_f64,
        anres: 0.0_f64,
        perfac: 1.0_f64,
        data: data31,
    };
    let mut nev1 = 15_usize;
    let mut nfit_st = 20_usize;
    let mut xfit_st = x31[nfit_st - 1];
    stevft(
        &x31,
        31,
        2,
        &mut nfit_st,
        2,
        false,
        true,
        &mut nev1,
        &mut xfit_st,
        &aa31,
        &rhs31,
        &cincnt,
    )
    .expect("stevft nn=31, nibc=2, iasn=2 must not underflow");
    assert!((1..=31).contains(&nev1));
    assert_eq!(nev1 % 2, 0);

    // `iasn = 0` must also return cleanly without division by zero.
    let mut nev1_zero = 2_usize;
    stevft(
        &x31,
        31,
        2,
        &mut nfit_st,
        0,
        false,
        true,
        &mut nev1_zero,
        &mut xfit_st,
        &aa31,
        &rhs31,
        &cincnt,
    )
    .expect("stevft iasn=0 guard");
}

#[test]
fn test_takata_and_order_coarse_and_single_point_guards() {
    // 1. `takata` (`takata.d.f:67-72`) with `iasn = 2` on odd `nn = 101`:
    //    `k = 100` yields `k0 = 99, k1 = 101 == nn`, which previously indexed OOB.
    let (data101, x101, aa101) = build_uniform_sphere_test_amdl(101);
    let mut y101 = vec![0.0_f64; 8 * 101];
    for (n, &xn) in x101.iter().enumerate() {
        let phase = std::f64::consts::PI * 2.5_f64 * xn;
        y101[n * 8] = xn * sin(phase);
        y101[n * 8 + 1] = xn;
    }
    let iord_iasn2 = takata(
        &x101,
        &y101,
        101,
        2,
        &data101,
        &aa101,
        8,
        15.0_f64,
        STANDARD_CGRAV,
    )
    .expect("takata iasn=2 on odd nn=101 must not index OOB");
    assert!(iord_iasn2 >= 0);
    assert_eq!(
        takata(
            &x101,
            &y101,
            101,
            0,
            &data101,
            &aa101,
            8,
            15.0_f64,
            STANDARD_CGRAV
        )
        .expect("takata iasn=0"),
        0
    );
    assert_eq!(
        takata(
            &[],
            &[],
            0,
            1,
            &data101,
            &aa101,
            8,
            15.0_f64,
            STANDARD_CGRAV
        )
        .expect("takata nn=0"),
        0
    );

    // 3. `order` (`order.d.f:117-123`) on `nn = 1, x[0] = 0.0` (`ns = 2, nsr = 1`):
    //    previously panicked on `nsr - ns` `usize` underflow.
    let mut iord_single = 7_i32;
    let x_single = [0.0_f64];
    let y_single = [0.0_f64; 8];
    let aa_single = AmdlAa::new_unset(1);
    order(
        &x_single,
        &y_single,
        &data101,
        &aa_single,
        0.0_f64,
        10.0_f64,
        0,
        0,
        8,
        &mut iord_single,
        1,
        1,
        5,
        1.0e-12_f64,
        STANDARD_CGRAV,
    )
    .expect("order nn=1 full model must not underflow nsr - ns");
    assert_eq!(iord_single, 7);

    order(
        &[],
        &[],
        &data101,
        &aa_single,
        0.0_f64,
        10.0_f64,
        0,
        0,
        8,
        &mut iord_single,
        0,
        1,
        5,
        1.0e-12_f64,
        STANDARD_CGRAV,
    )
    .expect("order nn=0 guard");
}

#[test]
fn test_nscfil_respects_adipls_open_order_and_unused_dev_null() {
    let input = parse_adipls_pruned_in(BASE_PRUNED_IN).expect("parse input");

    // 1. With `nfmode = 0`, unit `4 '0'` (`idsefn`) is NOT opened by ADIPLS
    //    (`adipls.c.d.f:1519`), so `15 '0'` (`idsssm`) is the 1st `/dev/null`
    //    and `nscfil(15)` must return `true` while `nscfil(4)` returns `false`.
    let state = SigoutState::new(&input, 0, 0, 0, 0, 0, 20_000);
    assert!(!state.nscfil(4));
    assert!(state.nscfil(9));
    assert!(state.nscfil(11));
    assert!(state.nscfil(15));
    assert!(!state.nscfil(0));
    assert!(!state.nscfil(-1));
    assert!(!state.nscfil(99));

    // 2. When `nfmode = 1`, ADIPLS opens `11 -> 15 -> 4` (`adipls.c.d.f:1517-1519`)
    //    regardless of input listing order (`4 '0'` listed before `15 '0'`).
    let mut state_nf1 = SigoutState::new(&input, 0, 0, 0, 0, 0, 20_000);
    state_nf1.out.nfmode = 1;
    assert!(state_nf1.nscfil(15));
    assert!(!state_nf1.nscfil(4));

    // 3. Reversed `/dev/null` order in `files`: `15 '0'` listed before `11 '0'`
    //    still opens `idsgsm = 11` before `idsssm = 15`.
    let mut state_rev = SigoutState::new(&input, 0, 0, 0, 0, 0, 20_000);
    state_rev.files = vec![
        AdiFileUnit {
            unit: 9,
            path: String::from("ttt.adipls.prt"),
        },
        AdiFileUnit {
            unit: 15,
            path: String::from("/dev/null"),
        },
        AdiFileUnit {
            unit: 11,
            path: String::from("/dev/null"),
        },
    ];
    assert!(state_rev.nscfil(11));
    assert!(!state_rev.nscfil(15));

    // 4. `istdpr = 9` mapped to `9 '0'` opens before `idslog`/`11`/`15`,
    //    consuming the single open `/dev/null` slot (`adipls.c.d.f:1256-1259`).
    let mut state_prt_null = SigoutState::new(&input, 0, 0, 0, 0, 0, 20_000);
    state_prt_null.files = vec![
        AdiFileUnit {
            unit: 9,
            path: String::from("/dev/null"),
        },
        AdiFileUnit {
            unit: 11,
            path: String::from("/dev/null"),
        },
        AdiFileUnit {
            unit: 15,
            path: String::from("/dev/null"),
        },
    ];
    assert!(state_prt_null.nscfil(9));
    assert!(!state_prt_null.nscfil(11));
    assert!(!state_prt_null.nscfil(15));
}

#[test]
fn test_sigout_emits_ssm_records_when_unopened_dev_null_precedes_idsssm() {
    let input = parse_adipls_pruned_in(BASE_PRUNED_IN).expect("parse input");
    let mut state = SigoutState::new(&input, 0, 0, 0, 0, 0, 20_000);

    // End-to-end `sigout` must emit `ssm_records` when `4 '0'` (`nfmode=0`)
    // precedes `15 '0'` (`idsssm`).
    let (data, x, aa) = build_uniform_sphere_test_amdl(31);
    let mut y = vec![0.0_f64; 8 * 32];
    for (k, &xk) in x.iter().enumerate() {
        y[k * 8] = xk;
        y[k * 8 + 1] = xk * (5.0_f64 / 3.0_f64);
    }
    let rhs = RhsDat {
        el: 0.0_f64,
        ell: 0.0_f64,
        alb: 1.0_f64,
        els: 0.0_f64,
        el1: 0.0_f64,
        sig: 10.0_f64,
        anres: 0.0_f64,
        perfac: 1.0_f64,
        data,
    };
    let rot = RotDat::default();
    let bcs = BcsDat::default();
    let cincnt = Cincnt::from_input(&input);
    let cprcns = Cprcns::default();
    let mut csummm = Csummm::zeroed();
    let mut obs = TapRecorder::new();
    {
        let mut args = SigoutArgs {
            sig: 10.0_f64,
            x: &x,
            y: &mut y,
            iy: 8,
            ii: 2,
            nw1: 1,
            nibc: 2,
            nn: 31,
            nnw: 31,
            mdintg: 5,
            nev1: 2,
            nfit: 15,
            istsb1: 1,
            iord: 1,
            isolcv: 1,
            isigcv: 1,
            aa: &aa,
            rhs: &rhs,
            rot: &rot,
            bcs: &bcs,
            cincnt: &cincnt,
            cprcns: &cprcns,
            csummm: &mut csummm,
            observer: &mut obs,
        };
        sigout(&mut args, &mut state)
            .expect("sigout emits ssm with unopened 4 '0'");
    }
    assert_eq!(state.ssm_records.len(), 1);
    assert_eq!(state.mode_store.records.len(), 1);
}
