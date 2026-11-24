// SPDX-License-Identifier: GPL-3.0-or-later
//! Safe Rust port of MESA 26.4.1 ADIPLS parameter parsing, FGONG-to-AMDL
//! conversion, control setup, `mdintg=5` scan/solve core, mode
//! post-processing, and mode output contracts (`#171`, Packages A1, A2a, and
//! A2b).
//!
//! # Scope and Status
//!
//! This crate covers the MESA-side ADIPLS glue, input/output contracts,
//! `mdintg=5` scan/solve core, and mode post-processing from MESA release
//! `26.4.1` (`astero/private/adipls_support.f90`,
//! `astero/private/adipls_support_procs.f90`,
//! `adipls/adipack.c/adipls/*.d.f`, `adipls/adipack.c/gensr/*.d.f`,
//! `adipls/adipack.c/adiajobs/sr/srdist.c.d.f`, and
//! `adipls/adipack.c/gensr/ofiles.c.f`):
//!
//! - Parsers for `adipls.c.pruned.in` and `redistrb.c.pruned.in`
//!   ([`parser`]).
//! - [`fgong_amdl`] and [`store_amdl`] with `#479` (`DECISIONS #43`) explicit
//!   unset tracking on AMDL `aa` rows `6..=9` ([`model`]).
//! - [`set_adipls_controls`] (`cadr_param`, `cadi_param`, `coutpt`), `sig_fac`
//!   helpers, and `#480` shadowing documentation ([`controls`]).
//! - Reached `gensr/` helpers ([`gensr`]): `lininh4`, `leq`, `derive`,
//!   `derivk`, `lir`, `notwin`, `notwni`, `vinta`, `vintk`, `aramax`, `store`,
//!   `zero`.
//! - ADIPLS model setup, boundary conditions, ODE RHS, and scaling
//!   ([`rhs`]): `setups_adi`, `res_adimod`, `setbcs`, `adirhs`, `geninh`,
//!   `danorm`, `xlmult`, `sclasl`.
//! - ADIPLS `mdintg=5` shooting scan and eigenvalue solve core ([`solve`],
//!   [`driver`]): `shtint`, `sigint`, `mchsol`, `mchcff`, `stevft`, `rsnfit`,
//!   `sigscn`, `sigitr`, `signew`, `sigitc`, `sigsol`, `order`, `takata`,
//!   `setlog`, and [`AdiplsSolver`].
//! - Mode post-processing and output generation ([`postproc`], [`output`]):
//!   `kiner`, `varfrq`, `sigout`, `setobs_st`, `setssm`, [`spcout_adi`],
//!   [`DefaultSigoutHandler`], and mode output buffers [`ModeRecord`],
//!   [`ObsSt`], [`Ssummm`], and [`Csummm`].

#![forbid(unsafe_code)]

pub mod controls;
pub mod driver;
pub mod error;
pub mod gensr;
pub mod model;
pub mod output;
pub mod parser;
pub mod postproc;
pub mod rhs;
pub mod solve;

pub use controls::{
    dfsig_from_delta_nu, resolve_store_model_flags, set_adipls_controls,
    sig_fac_adipls_support, sig_fac_astero_support, sig_from_nu_uhz,
    AdiplsCallerFlags, AdiplsControlState, AsteroDefPulseFlags, CadiParam,
    CadrParam, CoutptKernelControls, ResolvedPulseExportFlags,
    SetAdiplsControlsArgs,
};
pub use driver::{AdiplsCallResult, AdiplsSolver};
pub use error::AdiplsError;
pub use gensr::{
    aramax, deriv2, derive, derivk, leq, leqdet, leqsd, lininh4, lir, lir1,
    notwin, notwni, store, vinta, vintk, zero, DerivkCache, LirState,
    VintkCache,
};
pub use model::{
    fgong_amdl, store_amdl, AmdlAa, FgongAmdlOutput, FgongModel,
    StoreAmdlOutput, IAA_ARG, IVARMD,
};
pub use output::{
    spcout_adi, spcout_adi_store, AdiModeStore, Csummm, ModeRecord, ObsSt,
};
pub use parser::{
    parse_adipls_pruned_file, parse_adipls_pruned_in,
    parse_redistrb_pruned_file, parse_redistrb_pruned_in, AdiFileUnit,
    AdiplsCstControls, AdiplsDgnControls, AdiplsDsnControls, AdiplsInput,
    AdiplsIntControls, AdiplsModControls, AdiplsOscControls, AdiplsOutControls,
    AdiplsRotControls, CntrdSections, RedistrbInput, ADIPLS_DEFAULT_EPSPRC,
    ADIPLS_DEFAULT_EPSUFL,
};
pub use postproc::{
    kiner, kiner_with_observer, setobs_st, setssm, sigout, varfrq,
    varfrq_with_observer, CrotSplit, DefaultSigoutHandler, KinerOutput,
    ObsStState, SetssmOutput, SetssmState, SigoutState, SsmModelRecord, Ssummm,
    Varcon, VarfrqCache,
};
pub use rhs::{
    adirhs, danorm, geninh, res_adimod, sclasl, setbcs, setups_adi, xlmult,
    AdiplsSetupState, BcsDat, Cincnt, Cnrkbc, Cprcns, Csexpc, ResAdimodOutput,
    RhsDat, RotDat, XlmultCache, ADIPLS_AMSUN, ADIPLS_NNMAX, ADIPLS_NOBS_STMAX,
};
pub use solve::{
    mchcf1, mchcff, mchsol, order, rsnfit, setlog, shtint, sigint, sigitc,
    sigitr, signew, signw_pg, sigscn, sigsol, stevft, takata,
    AdiStatusLogEntry, AdiplsTapEvent, AdiplsTapObserver, NoopSigoutHandler,
    NoopTapObserver, SignwPgState, SigoutArgs, SigoutHandler, SolverContext,
    TapRecorder,
};
