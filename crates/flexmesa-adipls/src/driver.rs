// SPDX-License-Identifier: GPL-3.0-or-later
//! Safe Rust ports of MESA 26.4.1 ADIPLS top-level call driver (`adipls`,
//! `adipls/adipack.c/adipls/adipls.c.d.f:1-2094`) and parameter sync
//! (`res_adipar`, `adipls/adipack.c/adipls/res_adipar.c.d.f:1-91`)
//! (`#171`, Packages A2a and A2b).

use flexmesa_const::PI;
use flexmesa_math::powi_f90;

use crate::controls::AdiplsControlState;
use crate::error::AdiplsError;
use crate::model::{AmdlAa, IVARMD};
use crate::output::{AdiModeStore, Csummm, ObsSt};
use crate::parser::AdiplsInput;
use crate::postproc::{DefaultSigoutHandler, SetssmOutput};
use crate::rhs::{
    res_adimod, setups_adi, sp_radial_eps, AdiplsSetupState, BcsDat, Cincnt,
    Cnrkbc, Cprcns, Csexpc, RhsDat, RotDat, XlmultCache,
};
use crate::solve::{
    signew, sigscn, sigsol, AdiStatusLogEntry, AdiplsTapEvent,
    AdiplsTapObserver, NoopTapObserver, SignwPgState, SigoutArgs,
    SigoutHandler, SolverContext,
};

/// Result returned by a single per-model [`AdiplsSolver::run_call`] invocation.
#[derive(Debug, Clone, PartialEq)]
pub struct AdiplsCallResult {
    /// Status code `ierr_param` (`0` on normal completion).
    pub ierr_param: i32,
    /// Number of mesh points `nn` in the solved model.
    pub nn: usize,
    /// 1-based inner solution index `nw1`.
    pub nw1: usize,
    /// 1-based inner boundary condition index `nibc`.
    pub nibc: usize,
    /// 1-based fitting point index `nfit` used.
    pub nfit: usize,
    /// Final status log entries emitted via [`setlog`](crate::solve::setlog).
    pub status_log: Vec<AdiStatusLogEntry>,
    /// Final grand summary buffer `csummm(1:50)`.
    pub csummm: Csummm,
    /// Observation mode flag `icobs_st` (`common/cobs_param/`).
    pub icobs_st: i32,
    /// Accumulated `obs_st(1:10, 1:nobs_st)` records populated by `setobs_st`.
    pub obs_st: Vec<ObsSt>,
    /// Short-summary records emitted by `setssm`.
    pub ssm_records: Vec<SetssmOutput>,
    /// Complete mode records collected via [`spcout_adi`](crate::output::spcout_adi).
    pub modes: AdiModeStore,
}

/// Pure-Rust reentrant ADIPLS solver driver initialized from parsed
/// `adipls.c.pruned.in` configuration.
#[derive(Debug, Clone)]
pub struct AdiplsSolver {
    input: AdiplsInput,
    setup: AdiplsSetupState,
}

impl AdiplsSolver {
    /// Construct a new [`AdiplsSolver`] from a parsed [`AdiplsInput`],
    /// matching `setups_adi` and the initial parameter-reading call to
    /// `adipls(1, ...)` (`adipls.c.d.f:493-1233`).
    #[must_use]
    pub const fn new(input: AdiplsInput) -> Self {
        Self {
            input,
            setup: setups_adi(),
        }
    }

    /// Return the workspace setup constants initialized by `setups_adi`.
    #[must_use]
    pub const fn setup(&self) -> AdiplsSetupState {
        self.setup
    }

    /// Borrow the base parsed `adipls.c.pruned.in` configuration.
    #[must_use]
    pub const fn input(&self) -> &AdiplsInput {
        &self.input
    }

    /// Run one per-model `adipls` scan/solve and post-processing call,
    /// returning complete `ModeRecord`, `ObsSt`, `Ssummm`, and `Csummm`
    /// outputs.
    ///
    /// # Errors
    /// Propagates any [`AdiplsError`] from model validation, `#479` unset row
    /// reads, unsupported controls, post-processing, or solver failures.
    pub fn run_call(
        &self,
        controls: &AdiplsControlState,
        x_arg: &[f64],
        aa_arg: &AmdlAa,
        data_arg: [f64; 8],
    ) -> Result<AdiplsCallResult, AdiplsError> {
        let mut obs = NoopTapObserver;
        self.run_call_with_observer(controls, x_arg, aa_arg, data_arg, &mut obs)
    }

    /// Run one per-model `adipls` scan/solve and post-processing call with a
    /// tap observer (`ENTER`, `DET`, `MATCH`, `ITC`, `KINER`, `VARFRQ`).
    ///
    /// # Errors
    /// Propagates any [`AdiplsError`] from model validation, `#479` unset row
    /// reads, unsupported controls, post-processing, or solver failures.
    pub fn run_call_with_observer(
        &self,
        controls: &AdiplsControlState,
        x_arg: &[f64],
        aa_arg: &AmdlAa,
        data_arg: [f64; 8],
        observer: &mut impl AdiplsTapObserver,
    ) -> Result<AdiplsCallResult, AdiplsError> {
        let mut itrsig = controls.cadi_param.itrsig;
        let iscan = controls.cadi_param.iscan;
        if iscan > 1 && itrsig > 1 {
            itrsig = 1;
        }
        let nfmod1 = self.input.out.nfmode / 10;
        let nfmode_lo = self.input.out.nfmode % 10;
        let nfmscn = i32::from(
            (iscan > 1 && itrsig != 1 && nfmode_lo > 0)
                || (self.input.int_ctrl.itmax == 0 && nfmod1 == 1),
        );
        let mut sigout = DefaultSigoutHandler::new(
            &self.input,
            controls.coutpt.irotkr,
            controls.coutpt.nprtkr,
            controls.coutpt.igm1kr,
            controls.coutpt.npgmkr,
            nfmscn,
            self.setup.nobs_stmx,
        );
        let mut res = self.run_call_with_hooks(
            controls,
            x_arg,
            aa_arg,
            data_arg,
            observer,
            &mut sigout,
        )?;
        res.icobs_st = sigout.state.obs_state.icobs_st;
        res.obs_st = sigout.state.obs_state.records;
        res.ssm_records = sigout.state.ssm_records;
        res.modes = sigout.state.mode_store;
        Ok(res)
    }

    /// Run one per-model `adipls` scan/solve call with custom
    /// [`AdiplsTapObserver`] and [`SigoutHandler`] hooks.
    ///
    /// Mirrors `adipls(1, ierr_param, 0, x_arg, aa_arg, data_arg, nn_arg,
    /// ivarmd, iaa_arg)` starting at label `10000` (`adipls.c.d.f:1240-2042`).
    ///
    /// # Errors
    /// Propagates any [`AdiplsError`] from model setup, `#479` unset row reads,
    /// unsupported controls (`#147`), or solver execution.
    #[allow(clippy::too_many_lines, clippy::similar_names, clippy::float_cmp)]
    pub fn run_call_with_hooks(
        &self,
        controls: &AdiplsControlState,
        x_arg: &[f64],
        aa_arg: &AmdlAa,
        data_arg: [f64; 8],
        observer: &mut impl AdiplsTapObserver,
        sigout: &mut impl SigoutHandler,
    ) -> Result<AdiplsCallResult, AdiplsError> {
        observer.on_tap(AdiplsTapEvent::Enter);

        let cprcns = Cprcns::default();
        let mut cincnt = Cincnt::from_input(&self.input);
        let mut bcs = BcsDat {
            fctsbc: self.input.int_ctrl.fctsbc,
            fcttbc: self.input.int_ctrl.fcttbc,
            istsbc: self.input.int_ctrl.istsbc,
            ibotbc: self.input.int_ctrl.ibotbc,
        };
        let mut rot = RotDat {
            em: self.input.rot.em,
            irotsl: self.input.rot.irotsl,
            isprtp: 0,
            omgrtp: Vec::new(),
        };
        let mut rhs = RhsDat {
            alb: self.input.int_ctrl.alb,
            ..RhsDat::default()
        };

        let mut mdintg = self.input.int_ctrl.mdintg;
        let mut nfmode = self.input.out.nfmode;
        let cgrav = self.input.cst.cgrav;
        let nsem = self.input.rot.nsem;
        let ems1 = self.input.rot.ems1;
        let dems = self.input.rot.dems;

        // `res_adipar(2, ...)` (`adipls.c.d.f:1268-1273`, `res_adipar.c.d.f:66-89`).
        let mut el = controls.cadr_param.el;
        let els1 = controls.cadr_param.els1;
        let dels = controls.cadr_param.dels;
        let dfsig1 = controls.cadr_param.dfsig1;
        let dfsig2 = controls.cadr_param.dfsig2;
        let sig1 = controls.cadr_param.sig1;
        let sig2 = controls.cadr_param.sig2;
        let dfsig = controls.cadr_param.dfsig;
        let sgtrw1 = controls.cadr_param.sgtrw1;
        let sgtrw2 = controls.cadr_param.sgtrw2;

        let mut nsel = controls.cadi_param.nsel;
        let nsig1 = controls.cadi_param.nsig1;
        let nsig2 = controls.cadi_param.nsig2;
        let mut itrsig = controls.cadi_param.itrsig;
        let mut nsig = controls.cadi_param.nsig;
        let istsig = controls.cadi_param.istsig;
        cincnt.inomd1 = controls.cadi_param.inomd1;
        let mut iscan = controls.cadi_param.iscan;

        // Parameter consistency adjustments (`adipls.c.d.f:1278-1328`).
        if iscan > 1 && itrsig > 1 {
            itrsig = 1;
        }
        if istsig > 1 && itrsig % 2 == 0 && nsel > 1 && iscan <= 0 {
            nsel = 0;
        }
        if istsig > 1 && itrsig == 0 && iscan <= 0 {
            return Err(AdiplsError::UnsupportedControl {
                control: "istsig",
                message: format!(
                    "istsig={istsig} not allowed with itrsig=0 and iscan={iscan}"
                ),
            });
        }
        if cincnt.itsord == 1 && itrsig.abs() < 2 {
            cincnt.itsord = 0;
        }
        if cincnt.eps <= cprcns.epsprc {
            cincnt.eps = 1.1_f64 * cprcns.epsprc;
        }
        if (mdintg == 2 || mdintg == 4 || mdintg == 5) && cincnt.iriche == 1 {
            cincnt.iriche = 0;
        }

        let nfmod1 = nfmode / 10;
        nfmode %= 10;
        let nfmscn = i32::from(
            (iscan > 1 && itrsig != 1 && nfmode > 0)
                || (cincnt.itmax == 0 && nfmod1 == 1),
        );

        let itrsga = itrsig.abs();
        cincnt.fsig0 = cincnt.fsig;

        // `i_inout == 0`: reset input model via `res_adimod` (`adipls.c.d.f:1538-1546`).
        let nn_in = x_arg.len();
        let mod_out = res_adimod(
            x_arg,
            aa_arg,
            data_arg,
            nn_in,
            IVARMD,
            self.input.mod_ctrl.in_step,
            cincnt.iriche,
            &mut cincnt.iturpr,
            rot.isprtp,
            &mut rot.omgrtp,
        )?;
        cincnt.iturpr = mod_out.iturpr;
        let x = mod_out.x;
        let aa = mod_out.aa;
        rhs.data = mod_out.data;
        let nn0 = mod_out.nn;

        rhs.perfac = (PI / 30.0_f64)
            * (powi_f90(rhs.data[1], 3) / (rhs.data[0] * cgrav)).sqrt();

        if self.input.mod_ctrl.xtrnct > 0.0_f64
            || self.input.mod_ctrl.ntrnsf >= 1
        {
            return Err(AdiplsError::UnsupportedControl {
                control: "xtrnct/ntrnsf",
                message: String::from(
                    "model truncation (trnmod) is unported in Package A2a",
                ),
            });
        }

        let nw1 = 1_usize;
        let (nibc, fulmod, xtrnc1) = if x[0] == 0.0_f64 {
            (2_usize, true, 0.0_f64)
        } else {
            (1_usize, false, x[0])
        };
        let nn = nn0;
        let nnw = nn - nw1 + 1;

        let mut csummm = Csummm::zeroed();
        csummm.set_f64(10, aa.get(2, nn - 1)?)?;
        csummm.set_f64(11, aa.get(5, nn - 1)?)?;
        csummm.set_f64(12, xtrnc1)?;
        let mut ics = csummm.icsumm();
        ics[0] = self.input.mod_ctrl.in_step;
        ics[1] = i32::try_from(nnw).unwrap_or(i32::MAX);
        csummm.set_icsumm(ics);

        // Fitting point (`adipls.c.d.f:1669-1696`).
        let mut nfit_isize = if cincnt.xfit == -1.0_f64 {
            isize::try_from(nn).unwrap_or(1)
        } else {
            let mut nf = -1_isize;
            for (idx, &xn) in x.iter().take(nn).enumerate() {
                if xn < cincnt.xfit {
                    nf = isize::try_from(idx + 1).unwrap_or(-1);
                }
            }
            nf
        };

        if nfit_isize <= isize::try_from(nibc).unwrap_or(1) && mdintg != 3 {
            #[allow(clippy::cast_precision_loss)]
            let frfit = f64::from((nfit_isize as f32) / (nn as f32));
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_precision_loss
            )]
            let new_nf = (isize::try_from(nw1).unwrap_or(1) - 1)
                + (frfit * f64::from(i32::try_from(nn + 1 - nw1).unwrap_or(0)))
                    as isize;
            nfit_isize = new_nf;
        }
        if nfit_isize < 1 || usize::try_from(nfit_isize).is_err() {
            return Err(AdiplsError::InvalidModel {
                message: format!(
                    "adipls could not place fitting point nfit={nfit_isize} for xfit={}",
                    cincnt.xfit
                ),
            });
        }
        let mut nfit = usize::try_from(nfit_isize).unwrap_or(1);

        let (mut sigs1, mut sigs2) = if iscan > 1 {
            if istsig <= 1 {
                (sig1, sig2)
            } else {
                let sigfct = 4.0e-6_f64 * (PI * PI) * powi_f90(rhs.data[1], 3)
                    / (rhs.data[0] * cgrav);
                (sigfct * (sig1 * sig1), sigfct * (sig2 * sig2))
            }
        } else {
            (0.0_f64, 0.0_f64)
        };

        let iy = 8_usize;
        let mut y = vec![0.0_f64; iy * (nn + 2)];
        let mut csexpc = Csexpc::default();
        let mut cnrkbc = Cnrkbc::default();
        let mut kdgbcs = 0_i32;
        let mut xlmult_cache = XlmultCache::default();
        let mut signw_pg_state = SignwPgState::default();
        let mut status_log = Vec::new();
        let mut nev1 = nibc;
        let mut icry = 0_i32;

        let mut nel = nsel;
        if nsel >= 1 {
            el = els1;
            nel = 1;
        }

        // Outer degree loop (`20` in `adipls.c.d.f:1723-2008`).
        loop {
            let radial = el <= sp_radial_eps();
            if radial && cincnt.xfit == -1.0_f64 {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_precision_loss,
                    clippy::cast_sign_loss
                )]
                {
                    nfit = (0.5_f32 * (nn as f32)) as usize;
                }
            }
            if (nfit.cast_signed() - nibc.cast_signed()).rem_euclid(2) != 0
                && cincnt.iriche == 1
            {
                nfit += 1;
            }
            let xfit1 = x[nfit - 1];
            csummm.set_f64(14, xfit1)?;

            let nonrad = !radial || nsel > 1;
            if (cincnt.iplneq == 1 || (nonrad && !fulmod))
                && cincnt.icow != 2
                && cincnt.icow != 3
            {
                cincnt.icow = 2;
            }
            if mdintg == 2 && !radial && cincnt.icow != 2 {
                mdintg = 4;
            } else if mdintg == 4 && (radial || cincnt.icow == 2) {
                mdintg = 2;
            }

            if !(radial && cincnt.iplneq == 1) {
                let mut nem = 0_i32;
                if rot.irotsl == 1 && nsem != 0 {
                    rot.em = if nsem == -1 { -el } else { ems1 };
                    nem = 1;
                }

                // Azimuthal order loop (`25` in `adipls.c.d.f:1797-1985`).
                loop {
                    if rot.irotsl == 1 && rot.em > el {
                        break;
                    }

                    let mut sig = if iscan > 1 { sigs1 } else { sig1 };
                    let mut jstsig =
                        if istsig > 1 && iscan <= 1 { 1 } else { istsig };

                    // Frequency step loop (`30` in `adipls.c.d.f:1831-1962`).
                    loop {
                        if iscan <= 1
                            && istsig >= 1
                            && (itrsig >= 2 || (itrsig == 1 && jstsig >= 2))
                        {
                            let sigp = sig;
                            sig = signew(
                                itrsig, nsig, sigp, dfsig, sgtrw1, sgtrw2,
                                &mut icry,
                            )?;
                            if icry == -1 {
                                break;
                            } else if icry == -2 {
                                return Err(AdiplsError::Solver {
                                    routine: "adipls",
                                    message: String::from(
                                        "signew exceeded mode limit (icry=-2)",
                                    ),
                                });
                            }
                        }

                        for idx in 18..=38 {
                            csummm.set_f64(idx, 0.0_f64)?;
                        }

                        rhs.el = el;
                        rhs.ell = if cincnt.iplneq == 1 {
                            el * el
                        } else {
                            el * (el + 1.0_f64)
                        };
                        let mut istsb1 = bcs.istsbc;

                        if cincnt.iriche == 1 {
                            sig -= cincnt.dsigre;
                        }

                        let mut ctx = SolverContext {
                            rhs: &mut rhs,
                            rot: &mut rot,
                            bcs: &mut bcs,
                            cincnt: &mut cincnt,
                            csexpc: &mut csexpc,
                            cnrkbc: &mut cnrkbc,
                            cprcns: &cprcns,
                            kdgbcs: &mut kdgbcs,
                            cgrav,
                            xlmult_cache: &mut xlmult_cache,
                            signw_pg_state: &mut signw_pg_state,
                            csummm: &mut csummm,
                            iordtr: 0,
                            status_log: &mut status_log,
                        };

                        if iscan <= 1 {
                            let mut iord = 0_i32;
                            let mut isolcv = 0_i32;
                            let mut isigcv = 0_i32;
                            let mut sigtst = 0.0_f64;
                            sigsol(
                                &mut sig,
                                &x,
                                &mut y,
                                iy,
                                nw1,
                                nibc,
                                nn,
                                nnw,
                                mdintg,
                                &mut nev1,
                                &mut nfit,
                                &mut istsb1,
                                iscan,
                                &mut iord,
                                &mut icry,
                                &mut isolcv,
                                &mut isigcv,
                                &mut sigtst,
                                &aa,
                                &mut ctx,
                                observer,
                            )?;
                            if icry < 0 {
                                return Err(AdiplsError::Solver {
                                    routine: "adipls",
                                    message: format!(
                                        "sigsol failed with icry={icry}"
                                    ),
                                });
                            }
                            let ii_out = if radial || ctx.cincnt.icow >= 2 {
                                2
                            } else {
                                4
                            };
                            let mut args = SigoutArgs {
                                sig,
                                x: &x,
                                y: &mut y,
                                iy,
                                ii: ii_out,
                                nw1,
                                nibc,
                                nn,
                                nnw,
                                mdintg,
                                nev1,
                                nfit,
                                istsb1,
                                iord,
                                isolcv,
                                isigcv,
                                aa: &aa,
                                rhs: ctx.rhs,
                                rot: ctx.rot,
                                bcs: ctx.bcs,
                                cincnt: ctx.cincnt,
                                cprcns: ctx.cprcns,
                                csummm: ctx.csummm,
                                observer,
                            };
                            sigout.on_sigout(&mut args)?;
                        } else {
                            sigscn(
                                sigs1,
                                sigs2,
                                &mut iscan,
                                &mut nsig,
                                itrsig,
                                &x,
                                &mut y,
                                iy,
                                nw1,
                                nibc,
                                nn,
                                nnw,
                                mdintg,
                                &mut nev1,
                                &mut nfit,
                                &mut istsb1,
                                &mut icry,
                                nfmscn,
                                sgtrw1,
                                sgtrw2,
                                &aa,
                                &mut ctx,
                                observer,
                                sigout,
                            )?;
                            if icry < 0 {
                                return Err(AdiplsError::Solver {
                                    routine: "adipls",
                                    message: format!(
                                        "sigscn failed with icry={icry}"
                                    ),
                                });
                            }
                        }

                        if jstsig < istsig && itrsga >= 1 {
                            jstsig += 1;
                            continue;
                        }
                        break;
                    }

                    if rot.irotsl == 1
                        && nsem != 0
                        && (nsem == -1 || nem < nsem)
                        && rot.em <= el - dems + 1.0e-10_f64
                    {
                        nem += 1;
                        rot.em += dems;
                        if iscan > 1 {
                            if dfsig1 != 0.0_f64 {
                                sigs1 = signew(
                                    1, nsig1, sigs1, dfsig1, sgtrw1, sgtrw2,
                                    &mut icry,
                                )?;
                            }
                            if dfsig2 != 0.0_f64 {
                                sigs2 = signew(
                                    1, nsig2, sigs2, dfsig2, sgtrw1, sgtrw2,
                                    &mut icry,
                                )?;
                            }
                        }
                        continue;
                    }
                    break;
                }
            }

            if nel < nsel {
                nel += 1;
                el += dels;
                if iscan > 1 {
                    if dfsig1 != 0.0_f64 {
                        sigs1 = signew(
                            1, nsig1, sigs1, dfsig1, sgtrw1, sgtrw2, &mut icry,
                        )?;
                    }
                    if dfsig2 != 0.0_f64 {
                        sigs2 = signew(
                            1, nsig2, sigs2, dfsig2, sgtrw1, sgtrw2, &mut icry,
                        )?;
                    }
                }
                continue;
            }
            break;
        }

        Ok(AdiplsCallResult {
            ierr_param: 0,
            nn,
            nw1,
            nibc,
            nfit,
            status_log,
            csummm,
            icobs_st: 0,
            obs_st: Vec::new(),
            ssm_records: Vec::new(),
            modes: AdiModeStore::new(),
        })
    }
}
