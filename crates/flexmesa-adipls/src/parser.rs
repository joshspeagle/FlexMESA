// SPDX-License-Identifier: GPL-3.0-or-later
//! Exact Fortran list-directed parsers for `adipls.c.pruned.in` and
//! `redistrb.c.pruned.in` (`adipls/adipack.c/adipls/adipls.c.d.f:499-1208`,
//! `adipls/adipack.c/gensr/ofiles.c.f:1-87`, and
//! `adipls/adipack.c/adiajobs/sr/srdist.c.d.f:93-348`).

use std::fs;
use std::path::Path;

use crate::error::AdiplsError;

/// Default `epsprc` precision parameter (`adipls.c.d.f:137`).
pub const ADIPLS_DEFAULT_EPSPRC: f64 = 1.0e-13_f64;

/// Default `epsufl` underflow threshold (`adipls.c.d.f:138`).
pub const ADIPLS_DEFAULT_EPSUFL: f64 = 1.0e-37_f64;

/// Single `(unit, file)` pair read by ADIPLS `ofiles` (`ofiles.c.f:61-73`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdiFileUnit {
    /// Fortran I/O unit number.
    pub unit: i32,
    /// File path (with `"0"` normalized to `"/dev/null"`, `ofiles.c.f:66-68`).
    pub path: String,
}

/// Decoded `cntrd` section flags (`adipls.c.d.f:1083-1097`).
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CntrdSections {
    /// Read output dataset unit numbers (`dsn`).
    pub dsn: bool,
    /// Read equilibrium model controls (`mod`).
    pub model: bool,
    /// Read oscillation frequency/degree scan controls (`osc`).
    pub osc: bool,
    /// Read rotational solution controls (`rot`).
    pub rot: bool,
    /// Read fundamental physical constants (`cst`).
    pub cst: bool,
    /// Read integration and boundary-condition controls (`int`).
    pub int: bool,
    /// Read output, variational frequency, and kernel controls (`out`).
    pub out: bool,
    /// Read diagnostic controls (`dgn`).
    pub dgn: bool,
}

impl Default for CntrdSections {
    fn default() -> Self {
        Self::from_cntrd("mod.osc.int.out.dgn")
    }
}

impl CntrdSections {
    /// Decode a `cntrd` control string using exact case-sensitive substring
    /// matching as in `adipls.c.d.f:1083-1090`.
    #[must_use]
    pub fn from_cntrd(cntrd: &str) -> Self {
        Self {
            dsn: cntrd.contains("dsn"),
            model: cntrd.contains("mod"),
            osc: cntrd.contains("osc"),
            rot: cntrd.contains("rot"),
            cst: cntrd.contains("cst"),
            int: cntrd.contains("int"),
            out: cntrd.contains("out"),
            dgn: cntrd.contains("dgn"),
        }
    }

    /// Return whether at least one input section beyond `dsn`/`rot` is active
    /// (`adipls.c.d.f:1094`).
    #[must_use]
    pub const fn has_executable_section(self) -> bool {
        self.model || self.osc || self.cst || self.int || self.out || self.dgn
    }
}

/// `dsn:` section unit numbers (`adipls.c.d.f:520-532, 1099-1105`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdiplsDsnControls {
    /// Grand summary output unit (`idsgsm`, default `11`).
    pub idsgsm: i32,
    /// Short summary output unit (`idsssm`, default `15`).
    pub idsssm: i32,
    /// Eigenfunction output unit (`idsefn`, default `4`).
    pub idsefn: i32,
    /// Rotational kernel output unit (`idsrkr`, default `12`).
    pub idsrkr: i32,
    /// Gamma1 kernel output unit (`idsgkr`, default `13`).
    pub idsgkr: i32,
    /// Status log output unit (`idslog`, default `20`).
    pub idslog: i32,
}

impl Default for AdiplsDsnControls {
    fn default() -> Self {
        Self {
            idsgsm: 11,
            idsssm: 15,
            idsefn: 4,
            idsrkr: 12,
            idsgkr: 13,
            idslog: 20,
        }
    }
}

/// `mod:` section equilibrium-model controls (`adipls.c.d.f:548-570, 1107-1116`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdiplsModControls {
    /// Model selection flag (`ifind`, default `-1`).
    pub ifind: i32,
    /// Model index (`xmod`, default `0.0`).
    pub xmod: f64,
    /// Model input unit (`imlds`, default `2`).
    pub imlds: i32,
    /// Subsampling stride (`in` in Fortran, default `1`).
    pub in_step: i32,
    /// Diagnostic print stride (`nprmod`, default `0`).
    pub nprmod: i32,
    /// Fractional radius truncation (`xtrnct`, default `0.0`).
    pub xtrnct: f64,
    /// Surface point truncation count (`ntrnsf`, default `0`).
    pub ntrnsf: i32,
    /// Callback flag for `modmod` (`imdmod`, default `0`).
    pub imdmod: i32,
}

impl Default for AdiplsModControls {
    fn default() -> Self {
        Self {
            ifind: -1,
            xmod: 0.0_f64,
            imlds: 2,
            in_step: 1,
            nprmod: 0,
            xtrnct: 0.0_f64,
            ntrnsf: 0,
            imdmod: 0,
        }
    }
}

/// `osc:` section mode-selection and frequency-scan controls
/// (`adipls.c.d.f:579-678, 1118-1134`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdiplsOscControls {
    /// Degree $l$ (`el`, default `1.0`).
    pub el: f64,
    /// Number of degree steps (`nsel`, default `0`).
    pub nsel: i32,
    /// Starting degree when `nsel >= 1` (`els1`, default `10.0`).
    pub els1: f64,
    /// Degree step (`dels`, default `1.0`).
    pub dels: f64,
    /// Lower scan bound increment (`dfsig1`, default `0.0`).
    pub dfsig1: f64,
    /// Upper scan bound increment (`dfsig2`, default `0.0`).
    pub dfsig2: f64,
    /// Step mode for `dfsig1` (`nsig1`, default `1`).
    pub nsig1: i32,
    /// Step mode for `dfsig2` (`nsig2`, default `1`).
    pub nsig2: i32,
    /// Trial frequency mode (`itrsig`, default `0`).
    pub itrsig: i32,
    /// Lower scan bound or trial squared frequency (`sig1`, default `10.0`).
    pub sig1: f64,
    /// Frequency limit unit/step control (`istsig`, default `1`).
    pub istsig: i32,
    /// Initial mode index / scan stop control (`inomd1`, default `1`).
    pub inomd1: i32,
    /// Trial dataset unit (`itrds`, default `10`).
    pub itrds: i32,
    /// Trial frequency step (`dfsig`, default `0.0`).
    pub dfsig: f64,
    /// Frequency spacing type (`nsig`, default `1`).
    pub nsig: i32,
    /// Number of frequency scan points (`iscan`, default `1`).
    pub iscan: i32,
    /// Upper scan bound (`sig2`, default `0.0`).
    pub sig2: f64,
    /// Lower trial degree window (`eltrw1`, default `0.0`).
    pub eltrw1: f64,
    /// Upper trial degree window (`eltrw2`, default `-1.0`).
    pub eltrw2: f64,
    /// Lower trial $\sigma^2$ window (`sgtrw1`, default `0.0`).
    pub sgtrw1: f64,
    /// Upper trial $\sigma^2$ window (`sgtrw2`, default `-1.0`).
    pub sgtrw2: f64,
}

impl Default for AdiplsOscControls {
    fn default() -> Self {
        Self {
            el: 1.0_f64,
            nsel: 0,
            els1: 10.0_f64,
            dels: 1.0_f64,
            dfsig1: 0.0_f64,
            dfsig2: 0.0_f64,
            nsig1: 1,
            nsig2: 1,
            itrsig: 0,
            sig1: 10.0_f64,
            istsig: 1,
            inomd1: 1,
            itrds: 10,
            dfsig: 0.0_f64,
            nsig: 1,
            iscan: 1,
            sig2: 0.0_f64,
            eltrw1: 0.0_f64,
            eltrw2: -1.0_f64,
            sgtrw1: 0.0_f64,
            sgtrw2: -1.0_f64,
        }
    }
}

/// `rot:` section rotational solution controls (`adipls.c.d.f:685-693, 1136-1142`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdiplsRotControls {
    /// First-order rotational flag (`irotsl`, default `0`).
    pub irotsl: i32,
    /// Azimuthal order $m$ (`em`, default `0.0`).
    pub em: f64,
    /// Azimuthal step count (`nsem`, default `0`).
    pub nsem: i32,
    /// Starting $m$ (`ems1`, default `0.0`).
    pub ems1: f64,
    /// Step in $m$ (`dems`, default `0.0`).
    pub dems: f64,
}

impl Default for AdiplsRotControls {
    fn default() -> Self {
        Self {
            irotsl: 0,
            em: 0.0_f64,
            nsem: 0,
            ems1: 0.0_f64,
            dems: 0.0_f64,
        }
    }
}

/// `cst:` section fundamental constant controls (`adipls.c.d.f:700, 1144-1149`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdiplsCstControls {
    /// Gravitational constant $G$ in CGS ($\mathrm{cm^3\,g^{-1}\,s^{-2}}$;
    /// ADIPLS internal default `6.6732e-8`, overridden to `6.67430e-8` by MESA
    /// 26.4.1 `adipls.c.pruned.in`).
    pub cgrav: f64,
}

impl Default for AdiplsCstControls {
    fn default() -> Self {
        Self {
            cgrav: 6.6732e-8_f64,
        }
    }
}

/// `int:` section equation and integration controls (`adipls.c.d.f:708-852, 1152-1171`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdiplsIntControls {
    /// Plane-parallel equations flag (`iplneq`, default `0`).
    pub iplneq: i32,
    /// Turbulent pressure treatment (`iturpr`, default `0`).
    pub iturpr: i32,
    /// Cowling approximation mode (`icow`, default `0`).
    pub icow: i32,
    /// Poisson fudge factor $\lambda$ (`alb`, default `1.0`).
    pub alb: f64,
    /// Surface mechanical boundary condition (`istsbc`, default `1`).
    pub istsbc: i32,
    /// Surface BC interpolation factor (`fctsbc`, default `0.0`).
    pub fctsbc: f64,
    /// Bottom boundary condition (`ibotbc`, default `0`).
    pub ibotbc: i32,
    /// Bottom BC interpolation factor (`fcttbc`, default `0.0`).
    pub fcttbc: f64,
    /// Integration method (`mdintg`, default `1`).
    pub mdintg: i32,
    /// Richardson extrapolation flag (`iriche`, default `0`).
    pub iriche: i32,
    /// Fractional fitting point radius (`xfit`, default `0.5`).
    pub xfit: f64,
    /// Normalization radius fraction for NRKM (`fcnorm`, default `0.5`).
    pub fcnorm: f64,
    /// Eigenfrequency convergence tolerance (`eps`, default `2.0 * 1e-13`).
    pub eps: f64,
    /// Eigenfunction matching tolerance (`epssol`, default `1e-5`).
    pub epssol: f64,
    /// Maximum frequency iterations (`itmax`, default `8`).
    pub itmax: i32,
    /// Richardson trial decrement (`dsigre`, default `0.0`).
    pub dsigre: f64,
    /// Secant relative step (`fsig`, default `0.001`).
    pub fsig: f64,
    /// Maximum relative frequency step per iteration (`dsigmx`, default `0.1`).
    pub dsigmx: f64,
    /// Evanescent transition reset mode (`irsevn`, default `2`).
    pub irsevn: i32,
    /// Inner minimum radius for evanescent transition (`xmnevn`, default `0.0`).
    pub xmnevn: f64,
    /// Fitting-point retry limit (`nftmax`, default `3`).
    pub nftmax: i32,
    /// Trial-order check flag (`itsord`, default `0`).
    pub itsord: i32,
}

impl Default for AdiplsIntControls {
    fn default() -> Self {
        Self {
            iplneq: 0,
            iturpr: 0,
            icow: 0,
            alb: 1.0_f64,
            istsbc: 1,
            fctsbc: 0.0_f64,
            ibotbc: 0,
            fcttbc: 0.0_f64,
            mdintg: 1,
            iriche: 0,
            xfit: 0.5_f64,
            fcnorm: 0.5_f64,
            eps: 2.0_f64 * ADIPLS_DEFAULT_EPSPRC,
            epssol: 1.0e-5_f64,
            itmax: 8,
            dsigre: 0.0_f64,
            fsig: 0.001_f64,
            dsigmx: 0.1_f64,
            irsevn: 2,
            xmnevn: 0.0_f64,
            nftmax: 3,
            itsord: 0,
        }
    }
}

/// `out:` section output, variational, kernel, and windowing controls
/// (`adipls.c.d.f:861-960, 1173-1192`; `blstio.f:19-20`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdiplsOutControls {
    /// Printed output unit (`istdpr`, default `6` from `blstio`).
    pub istdpr: i32,
    /// Mesh points printed (`nout`, default `50`).
    pub nout: i32,
    /// Near-center print count (`nprcen`, default `0`).
    pub nprcen: i32,
    /// Order classification scheme (`irsord`, default `0`; `20` selects Takata).
    pub irsord: i32,
    /// Kinetic energy normalization (`iekinr`, default `0`).
    pub iekinr: i32,
    /// Variational frequency flag (`iper`, default `0`).
    pub iper: i32,
    /// Variational integral formulation (`ivarf`, default `1`).
    pub ivarf: i32,
    /// Variational differentiation degree parameter (`kvarf`, default `2`).
    pub kvarf: i32,
    /// Variational integrand print count (`npvarf`, default `0`).
    pub npvarf: i32,
    /// Eigenfunction output control (`nfmode`, default `0`).
    pub nfmode: i32,
    /// Rotational kernel control (`irotkr`, default `0`).
    pub irotkr: i32,
    /// Rotational kernel print count (`nprtkr`, default `0`).
    pub nprtkr: i32,
    /// Gamma1 kernel control (`igm1kr`, default `0`).
    pub igm1kr: i32,
    /// Gamma1 kernel print count (`npgmkr`, default `50`).
    pub npgmkr: i32,
    /// Special output selector (`ispcpr`, default `0`; `6` selects `cs(21)`).
    pub ispcpr: i32,
    /// Case filter (`icaswn`, default `-1`).
    pub icaswn: i32,
    /// Lower $\sigma^2$ output window (`sigwn1`, default `0.0`).
    pub sigwn1: f64,
    /// Upper $\sigma^2$ output window (`sigwn2`, default `-1.0`).
    pub sigwn2: f64,
    /// Lower frequency output window ($\mu\mathrm{Hz}$, `frqwn1`, default `0.0`).
    pub frqwn1: f64,
    /// Upper frequency output window ($\mu\mathrm{Hz}$, `frqwn2`, default `-1.0`).
    pub frqwn2: f64,
    /// Lower radial order output window (`iorwn1`, default `0`).
    pub iorwn1: i32,
    /// Upper radial order output window (`iorwn2`, default `-1`).
    pub iorwn2: i32,
    /// Lower $\nu / (l + 1/2)$ output window (`frlwn1`, default `0.0`).
    pub frlwn1: f64,
    /// Upper $\nu / (l + 1/2)$ output window (`frlwn2`, default `-1.0`).
    pub frlwn2: f64,
}

impl Default for AdiplsOutControls {
    fn default() -> Self {
        Self {
            istdpr: 6,
            nout: 50,
            nprcen: 0,
            irsord: 0,
            iekinr: 0,
            iper: 0,
            ivarf: 1,
            kvarf: 2,
            npvarf: 0,
            nfmode: 0,
            irotkr: 0,
            nprtkr: 0,
            igm1kr: 0,
            npgmkr: 50,
            ispcpr: 0,
            icaswn: -1,
            sigwn1: 0.0_f64,
            sigwn2: -1.0_f64,
            frqwn1: 0.0_f64,
            frqwn2: -1.0_f64,
            iorwn1: 0,
            iorwn2: -1,
            frlwn1: 0.0_f64,
            frlwn2: -1.0_f64,
        }
    }
}

/// `dgn:` section diagnostic controls (`adipls.c.d.f:968-990, 1194-1204`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdiplsDgnControls {
    /// NRKM test flag (`itssol`, default `0`).
    pub itssol: i32,
    /// NRKM test detail flag (`idgtss`, default `0`).
    pub idgtss: i32,
    /// Determinant modification flag (`moddet`, default `0`).
    pub moddet: i32,
    /// Determinant print flag (`iprdet`, default `0`).
    pub iprdet: i32,
    /// Separate solution print count (`npout`, default `0`).
    pub npout: i32,
    /// Matching coefficient selection mode (`imstsl`, default `1`).
    pub imstsl: i32,
    /// Ignored matching equation (`imissl`, default `2`).
    pub imissl: i32,
    /// Fixed matching coefficient (`imjssl`, default `2`).
    pub imjssl: i32,
    /// NRKM integration diagnostic level (`idgnrk`, default `0`).
    pub idgnrk: i32,
}

impl Default for AdiplsDgnControls {
    fn default() -> Self {
        Self {
            itssol: 0,
            idgtss: 0,
            moddet: 0,
            iprdet: 0,
            npout: 0,
            imstsl: 1,
            imissl: 2,
            imjssl: 2,
            idgnrk: 0,
        }
    }
}

/// Complete parsed configuration from `adipls.c.pruned.in`.
#[derive(Debug, Clone, PartialEq)]
pub struct AdiplsInput {
    /// Unit-to-path mappings parsed before `-1 ''`.
    pub files: Vec<AdiFileUnit>,
    /// Raw trimmed `cntrd` control string.
    pub cntrd_raw: String,
    /// Decoded section selection flags.
    pub cntrd: CntrdSections,
    /// `dsn:` controls.
    pub dsn: AdiplsDsnControls,
    /// `mod:` controls.
    pub mod_ctrl: AdiplsModControls,
    /// `osc:` controls.
    pub osc: AdiplsOscControls,
    /// `rot:` controls.
    pub rot: AdiplsRotControls,
    /// `cst:` controls.
    pub cst: AdiplsCstControls,
    /// `int:` controls (`int` section).
    pub int_ctrl: AdiplsIntControls,
    /// `out:` controls.
    pub out: AdiplsOutControls,
    /// `dgn:` controls.
    pub dgn: AdiplsDgnControls,
}

/// Complete parsed configuration from `redistrb.c.pruned.in` matching
/// `srdist` (`adipls/adipack.c/adiajobs/sr/srdist.c.d.f:93-348`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RedistrbInput {
    /// Target mesh point count (`nn`, default `601`; e.g., `2001` or `2201`).
    pub nn: i32,
    /// Congruent mesh flag (`icnmsh`, default `0`).
    pub icnmsh: i32,
    /// Redistribution preset selector (`icase`, default `0`).
    pub icase: i32,
    /// Convective envelope boundary mesh flag (`icvzbn`, default `0`).
    pub icvzbn: i32,
    /// Gaussian running-mean smoothing width (`nsmth`, set to `9` when
    /// `icase >= 20`).
    pub nsmth: i32,
    /// Discontinuity point count / fraction (`xdisc`, default `0.0`).
    pub xdisc: f64,
    /// Discontinuity distribution width in $\ln x$ (`dlxdsc`).
    pub dlxdsc: f64,
    /// Minimum $d\log\rho / d\log P$ for discontinuity (`dlgrmx`, default `10.0`).
    pub dlgrmx: f64,
    /// Amplitude at base of convection zone (`cacvzb`, default `0.0`).
    pub cacvzb: f64,
    /// Resolved non-negative g-mode mesh weight (`cg`).
    pub cg: f64,
    /// Resolved p-mode minimum resolution test flag (`itstcg`, `1` if input `cg < 0`).
    pub itstcg: i32,
    /// Constant mesh term weight (`cx`).
    pub cx: f64,
    /// Near-surface mesh weight (`ca`).
    pub ca: f64,
    /// `A_4` term weight (`cdgr`).
    pub cdgr: f64,
    /// `Delta A_4` term weight (`cddgr`).
    pub cddgr: f64,
    /// Interior `A` gradient weight (`cdg`).
    pub cdg: f64,
    /// Surface polytrope cutoff (`alphsf`).
    pub alphsf: f64,
    /// Core `A_4 / x^2` offset weight (`adda4`).
    pub adda4: f64,
    /// Minimum `A_4` in convective core (`accrmn`).
    pub accrmn: f64,
    /// Printed mesh output stride (`nout`).
    pub nout: i32,
    /// Innermost non-zero point control (`cn`).
    pub cn: f64,
    /// Reset `U` at innermost non-zero point (`irsu`).
    pub irsu: i32,
    /// Replacement `U` value when `irsu > 0` (`unew`).
    pub unew: f64,
    /// Reset `A_4` near center flag (`iresa4`, e.g., `0` or `10`).
    pub iresa4: i32,
    /// Model count (`nmodel`).
    pub nmodel: i32,
    /// Starting model index (`kmodel`).
    pub kmodel: i32,
    /// Conversion check flag (`itsaml`).
    pub itsaml: i32,
    /// Old expansion flag (`ioldex`).
    pub ioldex: i32,
    /// Input model unit (`idsin`).
    pub idsin: i32,
    /// Output model unit (`idsout`).
    pub idsout: i32,
    /// Signed discontinuity point count (`ndisc`, `srdist.c.d.f:276-281`).
    pub ndisc: i32,
    /// Unsigned discontinuity point count (`ndisca`, `srdist.c.d.f:335-347`).
    pub ndisca: i32,
}

impl Default for RedistrbInput {
    fn default() -> Self {
        Self {
            nn: 601,
            icnmsh: 0,
            icase: 0,
            icvzbn: 0,
            nsmth: 0,
            xdisc: 0.0_f64,
            dlxdsc: f64::from(1.0e-3_f32),
            dlgrmx: 10.0_f64,
            cacvzb: 0.0_f64,
            cg: 1.0_f64,
            itstcg: 0,
            cx: f64::from(40.0_f32),
            ca: f64::from(0.01_f32),
            cdgr: 0.0_f64,
            cddgr: 0.0_f64,
            cdg: 0.0_f64,
            alphsf: 0.0_f64,
            adda4: 0.0_f64,
            accrmn: 0.0_f64,
            nout: 100,
            cn: 2.0_f64,
            irsu: 0,
            unew: 3.0_f64,
            iresa4: 0,
            nmodel: 0,
            kmodel: 0,
            itsaml: 0,
            ioldex: 0,
            idsin: 2,
            idsout: 3,
            ndisc: 0,
            ndisca: 0,
        }
    }
}

/// Parse an `adipls.c.pruned.in` file from disk.
///
/// # Errors
/// Returns [`AdiplsError::Io`] on read failure or [`AdiplsError::Parse`] on
/// malformed contents.
pub fn parse_adipls_pruned_file(
    path: &Path,
) -> Result<AdiplsInput, AdiplsError> {
    let text = fs::read_to_string(path).map_err(|e| AdiplsError::Io {
        path: Some(path.display().to_string()),
        message: e.to_string(),
    })?;
    parse_adipls_pruned_in(&text)
}

/// Parse a `redistrb.c.pruned.in` file from disk.
///
/// # Errors
/// Returns [`AdiplsError::Io`] on read failure or [`AdiplsError::Parse`] on
/// malformed contents.
pub fn parse_redistrb_pruned_file(
    path: &Path,
) -> Result<RedistrbInput, AdiplsError> {
    let text = fs::read_to_string(path).map_err(|e| AdiplsError::Io {
        path: Some(path.display().to_string()),
        message: e.to_string(),
    })?;
    parse_redistrb_pruned_in(&text)
}

/// Parse the contents of `adipls.c.pruned.in` using exact Fortran
/// `ofiles` + `adipls` record order (`adipls.c.d.f:1065-1204`).
///
/// # Errors
/// Returns [`AdiplsError::Parse`] when required records are missing, `cntrd`
/// selects zero executable sections, or any field fails integer/float parsing.
pub fn parse_adipls_pruned_in(text: &str) -> Result<AdiplsInput, AdiplsError> {
    const KIND: &str = "adipls.c.pruned.in";
    let lines: Vec<(usize, &str)> =
        text.lines().enumerate().map(|(i, s)| (i + 1, s)).collect();
    let mut cursor = 0_usize;

    let files = parse_ofiles_block(&lines, &mut cursor, KIND)?;

    let (cntrd_line, cntrd_str) =
        next_line(&lines, &mut cursor, KIND, "cntrd line")?;
    let cntrd_raw = cntrd_str.trim().to_owned();
    let cntrd = CntrdSections::from_cntrd(&cntrd_raw);
    if !cntrd.has_executable_section() {
        return Err(AdiplsError::Parse {
            kind: KIND,
            line: cntrd_line,
            message: format!(
                "cntrd '{cntrd_raw}' selects no executable section"
            ),
        });
    }

    let dsn = if cntrd.dsn {
        parse_dsn_section(&lines, &mut cursor, KIND)?
    } else {
        AdiplsDsnControls::default()
    };
    let mod_ctrl = if cntrd.model {
        parse_mod_section(&lines, &mut cursor, KIND)?
    } else {
        AdiplsModControls::default()
    };
    let osc = if cntrd.osc {
        parse_osc_section(&lines, &mut cursor, KIND)?
    } else {
        AdiplsOscControls::default()
    };
    let rot = if cntrd.rot {
        parse_rot_section(&lines, &mut cursor, KIND)?
    } else {
        AdiplsRotControls::default()
    };
    let cst = if cntrd.cst {
        parse_cst_section(&lines, &mut cursor, KIND)?
    } else {
        AdiplsCstControls::default()
    };
    let int_ctrl = if cntrd.int {
        parse_int_section(&lines, &mut cursor, KIND)?
    } else {
        AdiplsIntControls::default()
    };
    let out = if cntrd.out {
        parse_out_section(&lines, &mut cursor, KIND)?
    } else {
        AdiplsOutControls::default()
    };
    let dgn = if cntrd.dgn {
        parse_dgn_section(&lines, &mut cursor, KIND)?
    } else {
        AdiplsDgnControls::default()
    };

    for &(lineno, trailing) in &lines[cursor..] {
        if !trailing.trim().is_empty() {
            return Err(AdiplsError::Parse {
                kind: KIND,
                line: lineno,
                message: format!(
                    "unexpected trailing data after cntrd sections: '{trailing}'"
                ),
            });
        }
    }

    Ok(AdiplsInput {
        files,
        cntrd_raw,
        cntrd,
        dsn,
        mod_ctrl,
        osc,
        rot,
        cst,
        int_ctrl,
        out,
        dgn,
    })
}

fn parse_ofiles_block(
    lines: &[(usize, &str)],
    cursor: &mut usize,
    kind: &'static str,
) -> Result<Vec<AdiFileUnit>, AdiplsError> {
    let mut files = Vec::new();
    loop {
        let (lineno, line) = next_line(lines, cursor, kind, "ofiles entry")?;
        let slots = parse_fortran_record(line, kind, lineno)?;
        if slots.len() < 2 {
            return Err(AdiplsError::Parse {
                kind,
                line: lineno,
                message: String::from(
                    "expected '<unit> <filename>' in ofiles block",
                ),
            });
        }
        check_no_extra_values(&slots, 2, kind, lineno)?;
        let unit = match &slots[0] {
            Some(tok) => parse_i32_token(tok, kind, lineno)?,
            None => {
                return Err(AdiplsError::Parse {
                    kind,
                    line: lineno,
                    message: String::from(
                        "missing unit number in ofiles block",
                    ),
                });
            }
        };
        if unit < 0 {
            break;
        }
        let raw_path = slots[1].clone().ok_or_else(|| AdiplsError::Parse {
            kind,
            line: lineno,
            message: String::from("missing filename in ofiles block"),
        })?;
        let path = if raw_path == "0" {
            String::from("/dev/null")
        } else {
            raw_path
        };
        files.push(AdiFileUnit { unit, path });
    }
    Ok(files)
}

fn parse_dsn_section(
    lines: &[(usize, &str)],
    cursor: &mut usize,
    kind: &'static str,
) -> Result<AdiplsDsnControls, AdiplsError> {
    let mut dsn = AdiplsDsnControls::default();
    let (ln, raw) = next_line(lines, cursor, kind, "dsn unit numbers")?;
    let slots = parse_fortran_record(raw, kind, ln)?;
    check_no_extra_values(&slots, 6, kind, ln)?;
    apply_i32(&slots, 0, &mut dsn.idsgsm, kind, ln)?;
    apply_i32(&slots, 1, &mut dsn.idsssm, kind, ln)?;
    apply_i32(&slots, 2, &mut dsn.idsefn, kind, ln)?;
    apply_i32(&slots, 3, &mut dsn.idsrkr, kind, ln)?;
    apply_i32(&slots, 4, &mut dsn.idsgkr, kind, ln)?;
    apply_i32(&slots, 5, &mut dsn.idslog, kind, ln)?;
    Ok(dsn)
}

fn parse_mod_section(
    lines: &[(usize, &str)],
    cursor: &mut usize,
    kind: &'static str,
) -> Result<AdiplsModControls, AdiplsError> {
    let mut mod_ctrl = AdiplsModControls::default();
    let (ln1, raw1) = next_line(lines, cursor, kind, "mod record 1")?;
    let s1 = parse_fortran_record(raw1, kind, ln1)?;
    check_no_extra_values(&s1, 5, kind, ln1)?;
    apply_i32(&s1, 0, &mut mod_ctrl.ifind, kind, ln1)?;
    apply_f64(&s1, 1, &mut mod_ctrl.xmod, kind, ln1)?;
    apply_i32(&s1, 2, &mut mod_ctrl.imlds, kind, ln1)?;
    apply_i32(&s1, 3, &mut mod_ctrl.in_step, kind, ln1)?;
    apply_i32(&s1, 4, &mut mod_ctrl.nprmod, kind, ln1)?;

    let (ln2, raw2) = next_line(lines, cursor, kind, "mod record 2")?;
    let s2 = parse_fortran_record(raw2, kind, ln2)?;
    check_no_extra_values(&s2, 3, kind, ln2)?;
    apply_f64(&s2, 0, &mut mod_ctrl.xtrnct, kind, ln2)?;
    apply_i32(&s2, 1, &mut mod_ctrl.ntrnsf, kind, ln2)?;
    apply_i32(&s2, 2, &mut mod_ctrl.imdmod, kind, ln2)?;
    Ok(mod_ctrl)
}

fn parse_osc_section(
    lines: &[(usize, &str)],
    cursor: &mut usize,
    kind: &'static str,
) -> Result<AdiplsOscControls, AdiplsError> {
    let mut osc = AdiplsOscControls::default();
    let (ln1, raw1) = next_line(lines, cursor, kind, "osc record 1")?;
    let s1 = parse_fortran_record(raw1, kind, ln1)?;
    check_no_extra_values(&s1, 8, kind, ln1)?;
    apply_f64(&s1, 0, &mut osc.el, kind, ln1)?;
    apply_i32(&s1, 1, &mut osc.nsel, kind, ln1)?;
    apply_f64(&s1, 2, &mut osc.els1, kind, ln1)?;
    apply_f64(&s1, 3, &mut osc.dels, kind, ln1)?;
    apply_f64(&s1, 4, &mut osc.dfsig1, kind, ln1)?;
    apply_f64(&s1, 5, &mut osc.dfsig2, kind, ln1)?;
    apply_i32(&s1, 6, &mut osc.nsig1, kind, ln1)?;
    apply_i32(&s1, 7, &mut osc.nsig2, kind, ln1)?;

    let (ln2, raw2) = next_line(lines, cursor, kind, "osc record 2")?;
    let s2 = parse_fortran_record(raw2, kind, ln2)?;
    check_no_extra_values(&s2, 5, kind, ln2)?;
    apply_i32(&s2, 0, &mut osc.itrsig, kind, ln2)?;
    apply_f64(&s2, 1, &mut osc.sig1, kind, ln2)?;
    apply_i32(&s2, 2, &mut osc.istsig, kind, ln2)?;
    apply_i32(&s2, 3, &mut osc.inomd1, kind, ln2)?;
    apply_i32(&s2, 4, &mut osc.itrds, kind, ln2)?;

    let (ln3, raw3) = next_line(lines, cursor, kind, "osc record 3")?;
    let s3 = parse_fortran_record(raw3, kind, ln3)?;
    check_no_extra_values(&s3, 4, kind, ln3)?;
    apply_f64(&s3, 0, &mut osc.dfsig, kind, ln3)?;
    apply_i32(&s3, 1, &mut osc.nsig, kind, ln3)?;
    apply_i32(&s3, 2, &mut osc.iscan, kind, ln3)?;
    apply_f64(&s3, 3, &mut osc.sig2, kind, ln3)?;

    let (ln4, raw4) = next_line(lines, cursor, kind, "osc record 4")?;
    let s4 = parse_fortran_record(raw4, kind, ln4)?;
    check_no_extra_values(&s4, 4, kind, ln4)?;
    apply_f64(&s4, 0, &mut osc.eltrw1, kind, ln4)?;
    apply_f64(&s4, 1, &mut osc.eltrw2, kind, ln4)?;
    apply_f64(&s4, 2, &mut osc.sgtrw1, kind, ln4)?;
    apply_f64(&s4, 3, &mut osc.sgtrw2, kind, ln4)?;
    Ok(osc)
}

fn parse_rot_section(
    lines: &[(usize, &str)],
    cursor: &mut usize,
    kind: &'static str,
) -> Result<AdiplsRotControls, AdiplsError> {
    let mut rot = AdiplsRotControls::default();
    let (ln, raw) = next_line(lines, cursor, kind, "rot record")?;
    let s = parse_fortran_record(raw, kind, ln)?;
    check_no_extra_values(&s, 5, kind, ln)?;
    apply_i32(&s, 0, &mut rot.irotsl, kind, ln)?;
    apply_f64(&s, 1, &mut rot.em, kind, ln)?;
    apply_i32(&s, 2, &mut rot.nsem, kind, ln)?;
    apply_f64(&s, 3, &mut rot.ems1, kind, ln)?;
    apply_f64(&s, 4, &mut rot.dems, kind, ln)?;
    Ok(rot)
}

fn parse_cst_section(
    lines: &[(usize, &str)],
    cursor: &mut usize,
    kind: &'static str,
) -> Result<AdiplsCstControls, AdiplsError> {
    let mut cst = AdiplsCstControls::default();
    let (ln, raw) = next_line(lines, cursor, kind, "cst record")?;
    let s = parse_fortran_record(raw, kind, ln)?;
    check_no_extra_values(&s, 1, kind, ln)?;
    apply_f64(&s, 0, &mut cst.cgrav, kind, ln)?;
    Ok(cst)
}

fn parse_int_section(
    lines: &[(usize, &str)],
    cursor: &mut usize,
    kind: &'static str,
) -> Result<AdiplsIntControls, AdiplsError> {
    let mut int_ctrl = AdiplsIntControls::default();
    let (ln1, raw1) = next_line(lines, cursor, kind, "int record 1")?;
    let s1 = parse_fortran_record(raw1, kind, ln1)?;
    check_no_extra_values(&s1, 4, kind, ln1)?;
    apply_i32(&s1, 0, &mut int_ctrl.iplneq, kind, ln1)?;
    apply_i32(&s1, 1, &mut int_ctrl.iturpr, kind, ln1)?;
    apply_i32(&s1, 2, &mut int_ctrl.icow, kind, ln1)?;
    apply_f64(&s1, 3, &mut int_ctrl.alb, kind, ln1)?;

    let (ln2, raw2) = next_line(lines, cursor, kind, "int record 2")?;
    let s2 = parse_fortran_record(raw2, kind, ln2)?;
    check_no_extra_values(&s2, 4, kind, ln2)?;
    apply_i32(&s2, 0, &mut int_ctrl.istsbc, kind, ln2)?;
    apply_f64(&s2, 1, &mut int_ctrl.fctsbc, kind, ln2)?;
    apply_i32(&s2, 2, &mut int_ctrl.ibotbc, kind, ln2)?;
    apply_f64(&s2, 3, &mut int_ctrl.fcttbc, kind, ln2)?;

    let (ln3, raw3) = next_line(lines, cursor, kind, "int record 3")?;
    let s3 = parse_fortran_record(raw3, kind, ln3)?;
    check_no_extra_values(&s3, 8, kind, ln3)?;
    apply_i32(&s3, 0, &mut int_ctrl.mdintg, kind, ln3)?;
    apply_i32(&s3, 1, &mut int_ctrl.iriche, kind, ln3)?;
    apply_f64(&s3, 2, &mut int_ctrl.xfit, kind, ln3)?;
    apply_f64(&s3, 3, &mut int_ctrl.fcnorm, kind, ln3)?;
    apply_f64(&s3, 4, &mut int_ctrl.eps, kind, ln3)?;
    apply_f64(&s3, 5, &mut int_ctrl.epssol, kind, ln3)?;
    apply_i32(&s3, 6, &mut int_ctrl.itmax, kind, ln3)?;
    apply_f64(&s3, 7, &mut int_ctrl.dsigre, kind, ln3)?;

    let (ln4, raw4) = next_line(lines, cursor, kind, "int record 4")?;
    let s4 = parse_fortran_record(raw4, kind, ln4)?;
    check_no_extra_values(&s4, 6, kind, ln4)?;
    apply_f64(&s4, 0, &mut int_ctrl.fsig, kind, ln4)?;
    apply_f64(&s4, 1, &mut int_ctrl.dsigmx, kind, ln4)?;
    apply_i32(&s4, 2, &mut int_ctrl.irsevn, kind, ln4)?;
    apply_f64(&s4, 3, &mut int_ctrl.xmnevn, kind, ln4)?;
    apply_i32(&s4, 4, &mut int_ctrl.nftmax, kind, ln4)?;
    apply_i32(&s4, 5, &mut int_ctrl.itsord, kind, ln4)?;
    Ok(int_ctrl)
}

fn parse_out_section(
    lines: &[(usize, &str)],
    cursor: &mut usize,
    kind: &'static str,
) -> Result<AdiplsOutControls, AdiplsError> {
    let mut out = AdiplsOutControls::default();
    let (ln1, raw1) = next_line(lines, cursor, kind, "out record 1")?;
    let s1 = parse_fortran_record(raw1, kind, ln1)?;
    check_no_extra_values(&s1, 5, kind, ln1)?;
    apply_i32(&s1, 0, &mut out.istdpr, kind, ln1)?;
    apply_i32(&s1, 1, &mut out.nout, kind, ln1)?;
    apply_i32(&s1, 2, &mut out.nprcen, kind, ln1)?;
    apply_i32(&s1, 3, &mut out.irsord, kind, ln1)?;
    apply_i32(&s1, 4, &mut out.iekinr, kind, ln1)?;

    let (ln2, raw2) = next_line(lines, cursor, kind, "out record 2")?;
    let s2 = parse_fortran_record(raw2, kind, ln2)?;
    check_no_extra_values(&s2, 5, kind, ln2)?;
    apply_i32(&s2, 0, &mut out.iper, kind, ln2)?;
    apply_i32(&s2, 1, &mut out.ivarf, kind, ln2)?;
    apply_i32(&s2, 2, &mut out.kvarf, kind, ln2)?;
    apply_i32(&s2, 3, &mut out.npvarf, kind, ln2)?;
    apply_i32(&s2, 4, &mut out.nfmode, kind, ln2)?;

    let (ln3, raw3) = next_line(lines, cursor, kind, "out record 3")?;
    let s3 = parse_fortran_record(raw3, kind, ln3)?;
    check_no_extra_values(&s3, 5, kind, ln3)?;
    apply_i32(&s3, 0, &mut out.irotkr, kind, ln3)?;
    apply_i32(&s3, 1, &mut out.nprtkr, kind, ln3)?;
    apply_i32(&s3, 2, &mut out.igm1kr, kind, ln3)?;
    apply_i32(&s3, 3, &mut out.npgmkr, kind, ln3)?;
    apply_i32(&s3, 4, &mut out.ispcpr, kind, ln3)?;

    let (ln4, raw4) = next_line(lines, cursor, kind, "out record 4")?;
    let s4 = parse_fortran_record(raw4, kind, ln4)?;
    check_no_extra_values(&s4, 9, kind, ln4)?;
    apply_i32(&s4, 0, &mut out.icaswn, kind, ln4)?;
    apply_f64(&s4, 1, &mut out.sigwn1, kind, ln4)?;
    apply_f64(&s4, 2, &mut out.sigwn2, kind, ln4)?;
    apply_f64(&s4, 3, &mut out.frqwn1, kind, ln4)?;
    apply_f64(&s4, 4, &mut out.frqwn2, kind, ln4)?;
    apply_i32(&s4, 5, &mut out.iorwn1, kind, ln4)?;
    apply_i32(&s4, 6, &mut out.iorwn2, kind, ln4)?;
    apply_f64(&s4, 7, &mut out.frlwn1, kind, ln4)?;
    apply_f64(&s4, 8, &mut out.frlwn2, kind, ln4)?;
    Ok(out)
}

fn parse_dgn_section(
    lines: &[(usize, &str)],
    cursor: &mut usize,
    kind: &'static str,
) -> Result<AdiplsDgnControls, AdiplsError> {
    let mut dgn = AdiplsDgnControls::default();
    let (ln1, raw1) = next_line(lines, cursor, kind, "dgn record 1")?;
    let s1 = parse_fortran_record(raw1, kind, ln1)?;
    check_no_extra_values(&s1, 5, kind, ln1)?;
    apply_i32(&s1, 0, &mut dgn.itssol, kind, ln1)?;
    apply_i32(&s1, 1, &mut dgn.idgtss, kind, ln1)?;
    apply_i32(&s1, 2, &mut dgn.moddet, kind, ln1)?;
    apply_i32(&s1, 3, &mut dgn.iprdet, kind, ln1)?;
    apply_i32(&s1, 4, &mut dgn.npout, kind, ln1)?;

    let (ln2, raw2) = next_line(lines, cursor, kind, "dgn record 2")?;
    let s2 = parse_fortran_record(raw2, kind, ln2)?;
    check_no_extra_values(&s2, 4, kind, ln2)?;
    apply_i32(&s2, 0, &mut dgn.imstsl, kind, ln2)?;
    apply_i32(&s2, 1, &mut dgn.imissl, kind, ln2)?;
    apply_i32(&s2, 2, &mut dgn.imjssl, kind, ln2)?;
    apply_i32(&s2, 3, &mut dgn.idgnrk, kind, ln2)?;
    Ok(dgn)
}

/// Parse the contents of `redistrb.c.pruned.in` and apply `srdist` preset
/// rules (`srdist.c.d.f:217-348`).
///
/// # Errors
/// Returns [`AdiplsError::Parse`] when any of the 5 required records is missing
/// or contains invalid tokens, or when non-empty trailing data follows.
pub fn parse_redistrb_pruned_in(
    text: &str,
) -> Result<RedistrbInput, AdiplsError> {
    const KIND: &str = "redistrb.c.pruned.in";
    let lines: Vec<(usize, &str)> =
        text.lines().enumerate().map(|(i, s)| (i + 1, s)).collect();
    let mut cursor = 0_usize;
    let mut cfg = RedistrbInput::default();

    // Line 1: nn, icnmsh (`srdist.c.d.f:217`)
    let (ln1, raw1) =
        next_line(&lines, &mut cursor, KIND, "nn, icnmsh record")?;
    let s1 = parse_fortran_record(raw1, KIND, ln1)?;
    check_no_extra_values(&s1, 2, KIND, ln1)?;
    apply_i32(&s1, 0, &mut cfg.nn, KIND, ln1)?;
    apply_i32(&s1, 1, &mut cfg.icnmsh, KIND, ln1)?;
    if cfg.nn <= 0 {
        return Err(AdiplsError::Parse {
            kind: KIND,
            line: ln1,
            message: format!("redistrb nn must be positive, got {}", cfg.nn),
        });
    }

    // Line 2: icase, icvzbn, nsmth, xdisc, dlxdsc, dlgrmx, cacvzb (`srdist.c.d.f:220`)
    let (ln2, raw2) =
        next_line(&lines, &mut cursor, KIND, "icase..cacvzb record")?;
    let s2 = parse_fortran_record(raw2, KIND, ln2)?;
    check_no_extra_values(&s2, 7, KIND, ln2)?;
    apply_i32(&s2, 0, &mut cfg.icase, KIND, ln2)?;
    apply_i32(&s2, 1, &mut cfg.icvzbn, KIND, ln2)?;
    apply_i32(&s2, 2, &mut cfg.nsmth, KIND, ln2)?;
    apply_f64(&s2, 3, &mut cfg.xdisc, KIND, ln2)?;
    apply_f64(&s2, 4, &mut cfg.dlxdsc, KIND, ln2)?;
    apply_f64(&s2, 5, &mut cfg.dlgrmx, KIND, ln2)?;
    apply_f64(&s2, 6, &mut cfg.cacvzb, KIND, ln2)?;

    // Line 3: cg, cx, ca, cdgr, cddgr, cdg, alphsf, adda4, accrmn (`srdist.c.d.f:227`)
    let (ln3, raw3) =
        next_line(&lines, &mut cursor, KIND, "cg..accrmn record")?;
    let s3 = parse_fortran_record(raw3, KIND, ln3)?;
    check_no_extra_values(&s3, 9, KIND, ln3)?;
    apply_f64(&s3, 0, &mut cfg.cg, KIND, ln3)?;
    apply_f64(&s3, 1, &mut cfg.cx, KIND, ln3)?;
    apply_f64(&s3, 2, &mut cfg.ca, KIND, ln3)?;
    apply_f64(&s3, 3, &mut cfg.cdgr, KIND, ln3)?;
    apply_f64(&s3, 4, &mut cfg.cddgr, KIND, ln3)?;
    apply_f64(&s3, 5, &mut cfg.cdg, KIND, ln3)?;
    apply_f64(&s3, 6, &mut cfg.alphsf, KIND, ln3)?;
    apply_f64(&s3, 7, &mut cfg.adda4, KIND, ln3)?;
    apply_f64(&s3, 8, &mut cfg.accrmn, KIND, ln3)?;

    // Line 4: nout, cn, irsu, unew, iresa4 (`srdist.c.d.f:230`)
    let (ln4, raw4) =
        next_line(&lines, &mut cursor, KIND, "nout..iresa4 record")?;
    let s4 = parse_fortran_record(raw4, KIND, ln4)?;
    check_no_extra_values(&s4, 5, KIND, ln4)?;
    apply_i32(&s4, 0, &mut cfg.nout, KIND, ln4)?;
    apply_f64(&s4, 1, &mut cfg.cn, KIND, ln4)?;
    apply_i32(&s4, 2, &mut cfg.irsu, KIND, ln4)?;
    apply_f64(&s4, 3, &mut cfg.unew, KIND, ln4)?;
    apply_i32(&s4, 4, &mut cfg.iresa4, KIND, ln4)?;

    // Line 5: nmodel, kmodel, itsaml, ioldex, idsin, idsout (`srdist.c.d.f:233`)
    let (ln5, raw5) =
        next_line(&lines, &mut cursor, KIND, "nmodel..idsout record")?;
    let s5 = parse_fortran_record(raw5, KIND, ln5)?;
    check_no_extra_values(&s5, 6, KIND, ln5)?;
    apply_i32(&s5, 0, &mut cfg.nmodel, KIND, ln5)?;
    apply_i32(&s5, 1, &mut cfg.kmodel, KIND, ln5)?;
    apply_i32(&s5, 2, &mut cfg.itsaml, KIND, ln5)?;
    apply_i32(&s5, 3, &mut cfg.ioldex, KIND, ln5)?;
    apply_i32(&s5, 4, &mut cfg.idsin, KIND, ln5)?;
    apply_i32(&s5, 5, &mut cfg.idsout, KIND, ln5)?;

    for &(lineno, trailing) in &lines[cursor..] {
        if !trailing.trim().is_empty() {
            return Err(AdiplsError::Parse {
                kind: KIND,
                line: lineno,
                message: format!(
                    "unexpected trailing data in {KIND}: '{trailing}'"
                ),
            });
        }
    }

    // Apply `srdist` preset rules (`srdist.c.d.f:241-348`).
    apply_srdist_presets(&mut cfg, ln2)?;
    Ok(cfg)
}

fn apply_srdist_presets(
    cfg: &mut RedistrbInput,
    icase_line: usize,
) -> Result<(), AdiplsError> {
    if cfg.icase == 1 {
        cfg.cg = f64::from(0.001_f32);
        cfg.cx = f64::from(0.1_f32);
        cfg.ca = f64::from(0.01_f32);
        cfg.cn = 2.0_f64;
    } else if cfg.icase == 2 {
        cfg.cg = 1.0_f64;
        cfg.cx = 40.0_f64;
        cfg.ca = f64::from(0.01_f32);
        cfg.cn = 2.0_f64;
    } else if cfg.icase == 3 {
        cfg.cg = f64::from(0.05_f32);
        cfg.cx = 1.0_f64;
        cfg.ca = f64::from(0.01_f32);
        cfg.cn = 2.0_f64;
    } else if cfg.icase == 11 {
        cfg.cg = f64::from(0.001_f32);
        cfg.cx = f64::from(0.1_f32);
        cfg.ca = f64::from(0.003_f32);
        cfg.cn = 2.0_f64;
    } else if cfg.icase == 12 {
        cfg.cg = 4.0_f64;
        cfg.cx = 40.0_f64;
        cfg.ca = f64::from(0.01_f32);
        cfg.cn = 2.0_f64;
    } else if cfg.icase >= 20 {
        cfg.nsmth = 9;
        let scaled = if cfg.xdisc.abs() >= 1.0_f64 {
            cfg.xdisc
        } else {
            -(cfg.xdisc.abs() * f64::from(cfg.nn))
        };
        if !scaled.is_finite()
            || scaled < f64::from(i32::MIN)
            || scaled > f64::from(i32::MAX)
        {
            return Err(AdiplsError::Parse {
                kind: "redistrb.c.pruned.in",
                line: icase_line,
                message: format!("invalid xdisc value {}", cfg.xdisc),
            });
        }
        #[allow(clippy::cast_possible_truncation)]
        {
            cfg.ndisc = scaled as i32;
        }
        if cfg.icase < 100 || cfg.ndisc.abs() < 60 {
            cfg.ndisc = -60;
        }
        cfg.dlxdsc = f64::from(0.0002_f32);
        if cfg.icase == 21 {
            cfg.cg = f64::from(0.001_f32);
            cfg.cx = 1.0_f64;
            cfg.ca = f64::from(0.005_f32);
            cfg.cdg = f64::from(0.01_f32);
            cfg.cn = 2.0_f64;
        } else if cfg.icase == 22 {
            cfg.cg = 4.0_f64;
            cfg.cx = 40.0_f64;
            cfg.ca = f64::from(0.01_f32);
            cfg.cn = 2.0_f64;
        } else if cfg.icase == 23 {
            cfg.cg = f64::from(0.1_f32);
            cfg.cx = 1.0_f64;
            cfg.ca = f64::from(0.005_f32);
            cfg.cdg = f64::from(0.01_f32);
            cfg.cn = 2.0_f64;
        } else if cfg.icase == 24 {
            cfg.cg = f64::from(0.0001_f32);
            cfg.cx = 5.0_f64;
            cfg.ca = f64::from(0.005_f32);
            cfg.cdg = f64::from(0.01_f32);
            cfg.adda4 = 0.5_f64;
            cfg.cn = 2.0_f64;
        }
    }

    if cfg.nsmth == 0 {
        cfg.cacvzb = 0.0_f64;
    }
    if cfg.cg < 0.0_f64 {
        cfg.itstcg = 1;
        cfg.cg = cfg.cg.abs();
    } else {
        cfg.itstcg = 0;
    }
    cfg.ndisca = cfg.ndisc.abs();
    if cfg.ndisca > 0 && cfg.ndisca < 20 {
        cfg.ndisca = 20;
    }
    Ok(())
}

fn next_line<'a>(
    lines: &[(usize, &'a str)],
    cursor: &mut usize,
    kind: &'static str,
    expected: &str,
) -> Result<(usize, &'a str), AdiplsError> {
    if *cursor >= lines.len() {
        return Err(AdiplsError::Parse {
            kind,
            line: lines.last().map_or(0, |&(ln, _)| ln),
            message: format!("unexpected EOF waiting for {expected}"),
        });
    }
    let item = lines[*cursor];
    *cursor += 1;
    Ok(item)
}

/// Parse one Fortran list-directed input record into slots (`None` for comma
/// null items, `Some(val)` for tokens/strings). Supports repeat counts `k*val`
/// and `k*`.
fn parse_fortran_record(
    line: &str,
    kind: &'static str,
    lineno: usize,
) -> Result<Vec<Option<String>>, AdiplsError> {
    let bytes = line.as_bytes();
    let len = bytes.len();
    let mut i = skip_hws(bytes, 0);
    let mut slots = Vec::new();

    while i < len {
        if bytes[i] == b',' {
            slots.push(None);
            i = skip_hws(bytes, i + 1);
            continue;
        }

        if bytes[i] == b'\'' || bytes[i] == b'"' {
            let quote = bytes[i];
            i += 1;
            let mut val = String::new();
            let mut closed = false;
            while i < len {
                if bytes[i] == quote {
                    if i + 1 < len && bytes[i + 1] == quote {
                        val.push(char::from(quote));
                        i += 2;
                    } else {
                        i += 1;
                        closed = true;
                        break;
                    }
                } else {
                    val.push(char::from(bytes[i]));
                    i += 1;
                }
            }
            if !closed {
                return Err(AdiplsError::Parse {
                    kind,
                    line: lineno,
                    message: String::from(
                        "unterminated quoted string in Fortran record",
                    ),
                });
            }
            slots.push(Some(val));
        } else {
            let start = i;
            while i < len && !is_hws(bytes[i]) && bytes[i] != b',' {
                i += 1;
            }
            let raw = &line[start..i];
            if let Some((cnt_str, rest)) = raw.split_once('*') {
                let count = cnt_str.parse::<usize>().map_err(|_| {
                    AdiplsError::Parse {
                        kind,
                        line: lineno,
                        message: format!(
                            "invalid repeat factor '{cnt_str}' in '{raw}'"
                        ),
                    }
                })?;
                if count == 0 {
                    return Err(AdiplsError::Parse {
                        kind,
                        line: lineno,
                        message: format!("zero repeat factor in '{raw}'"),
                    });
                }
                let item = if rest.is_empty() {
                    None
                } else {
                    Some(rest.to_owned())
                };
                for _ in 0..count {
                    slots.push(item.clone());
                }
            } else {
                slots.push(Some(raw.to_owned()));
            }
        }

        i = skip_hws(bytes, i);
        if i < len && bytes[i] == b',' {
            i = skip_hws(bytes, i + 1);
        }
    }

    Ok(slots)
}

const fn is_hws(b: u8) -> bool {
    b == b' ' || b == b'\t'
}

fn skip_hws(bytes: &[u8], mut idx: usize) -> usize {
    while idx < bytes.len() && is_hws(bytes[idx]) {
        idx += 1;
    }
    idx
}

fn check_no_extra_values(
    slots: &[Option<String>],
    max_vars: usize,
    kind: &'static str,
    lineno: usize,
) -> Result<(), AdiplsError> {
    if let Some(extra) = slots.iter().skip(max_vars).flatten().next() {
        return Err(AdiplsError::Parse {
            kind,
            line: lineno,
            message: format!(
                "unexpected extra value '{extra}' past {max_vars} fields"
            ),
        });
    }
    Ok(())
}

fn apply_i32(
    slots: &[Option<String>],
    idx: usize,
    target: &mut i32,
    kind: &'static str,
    lineno: usize,
) -> Result<(), AdiplsError> {
    if let Some(Some(tok)) = slots.get(idx) {
        *target = parse_i32_token(tok, kind, lineno)?;
    }
    Ok(())
}

fn apply_f64(
    slots: &[Option<String>],
    idx: usize,
    target: &mut f64,
    kind: &'static str,
    lineno: usize,
) -> Result<(), AdiplsError> {
    if let Some(Some(tok)) = slots.get(idx) {
        *target = parse_f64_token(tok, kind, lineno)?;
    }
    Ok(())
}

fn parse_i32_token(
    tok: &str,
    kind: &'static str,
    lineno: usize,
) -> Result<i32, AdiplsError> {
    tok.trim().parse::<i32>().map_err(|_| AdiplsError::Parse {
        kind,
        line: lineno,
        message: format!("invalid integer token '{tok}'"),
    })
}

fn parse_f64_token(
    tok: &str,
    kind: &'static str,
    lineno: usize,
) -> Result<f64, AdiplsError> {
    let trimmed = tok.trim();
    let normalized = if trimmed.contains(['d', 'D']) {
        trimmed.replace(['d', 'D'], "e")
    } else {
        trimmed.to_owned()
    };
    let val = normalized.parse::<f64>().map_err(|_| AdiplsError::Parse {
        kind,
        line: lineno,
        message: format!("invalid float token '{tok}'"),
    })?;
    if !val.is_finite() {
        return Err(AdiplsError::Parse {
            kind,
            line: lineno,
            message: format!("non-finite float token '{tok}'"),
        });
    }
    Ok(val)
}
