# N1: numerics crates, Python/JAX layer, simplification/diagnosability

> **Reference note (`#141`):** Line numbers, commit diffs, and pre-rebaseline measurements in this historical audit register refer to MESA `main` at commit `0b2e6bb` (2026-11-04) and `FlexMESA` commit `91eda0b`. Active repository inputs, module goldens, `case_table.md`, and `defaults_diff_26_4_1.md` target **MESA release `26.4.1`** (tag `26.4.1`, commit `c8c2026`, Zenodo `19722306`).

Auditor scope: A (Rust numerics crates vs upstream), B (Python/JAX layer), C (structure, diagnosability, determinism).
FlexMESA root `the audited tree` (PR #138 head), upstream `the upstream checkout` @ `0b2e6bb`.
Nothing was compiled or run; all findings are from reading source.

## Scope A: numerics crates

### N1-1: `newuoa` and `bobyqa` are simplified stand-ins, not ports of Powell's algorithms
- Area: flexmesa-num
- Severity: unverified claim
- FlexMESA: `crates/flexmesa-num/src/newuoa.rs:1-2` ("Powell (2006) ... (`NEWUOA` in `num/private/mod_newuoa.f`)"), `newuoa.rs:134-137` (`fit_min_frobenius_quad(...)` re-solved from scratch every iteration), `newuoa.rs:283-297` (`kkt[...] = 0.5*dot*dot + if r == q { 1e-11 }`, `kkt[(m+j)*dim+(m+j)] = -1e-12`), `newuoa.rs:330-383` (`trsbox_unconstrained`, plain Steihaug CG), `newuoa.rs:413-428` (`geometry_step_unconstrained` = step of length rho toward the far point), `bobyqa.rs:8` (`use crate::newuoa::fit_min_frobenius_quad`)
- Upstream: `num/private/mod_newuoa.f` (1583 lines: NEWUOB, BIGDEN, BIGLAG, TRSAPP, UPDATE with the factored `ZMAT`/`BMAT` inverse-KKT update), `num/private/mod_bobyqa.f` (2243 lines: BOBYQB, ALTMOV, PRELIM, RESCUE, TRSBOX, UPDATE)
- Confidence: verified-by-reading-both-sides (structure and size; upstream not read line by line)
- What: The 428/422-line Rust files keep Powell's initial point layout and the rho-reduction schedule, but replace the whole model-maintenance machinery (BIGLAG/BIGDEN denominators, `UPDATE` of the inverse KKT factorisation, RESCUE, ALTMOV) with a dense regularised KKT refit (Tikhonov terms `1e-11`, `-1e-12`) every iteration, and a "geometry step" that just walks toward the farthest point. The interpolation-point replacement rule ("farthest from the new best point") is not Powell's sigma/denominator rule. Iterates, evaluation counts and final points will differ from MESA's for any non-trivial objective. Docstrings present both as the upstream algorithm. Consumers: `crates/flexmesa-astero/src/optimize.rs:7` (astero NEWUOA/BOBYQA drivers), so astero optimisation trajectories cannot match upstream.
- Proposed fix: rename to what they are (`dfo_trust_region_*`) and state "not a port" in docs, or port mod_newuoa.f/mod_bobyqa.f properly (L, ~3.5k lines of Fortran). Remove "parity" wording for astero optimisers until then.

### N1-2: `dop853` is `dopri5` with tolerances multiplied by 0.25
- Area: flexmesa-num
- Severity: unverified claim
- FlexMESA: `crates/flexmesa-num/src/rk.rs:336-361` (`dopri5(x0, x_end, y0, h0, hmax, (rtol * 0.25).max(1e-15), (atol * 0.25).max(1e-15), max_steps, fcn)`); `lib.rs:26-27` lists `dop853` among "Explicit Runge-Kutta integrators".
- Upstream: `num/private/mod_dop853.f` (836 lines, 12-stage 8(5,3) Dormand-Prince with its own tableau and `err`/`err2` combined estimator)
- Confidence: verified-by-reading-both-sides
- What: There is no 8th-order method. Anything that asks for `dop853` gets a 5th-order integration with slightly tightened tolerance, hence different step sequences, cost and error behaviour. Not used by the star crates today (grep finds only `dopri5` in `flexmesa-atm/src/t_tau_uniform.rs:21`, `t_tau_varying.rs:23`), so the impact is on the claim "ports ... `mod_dop853.f`" (`lib.rs:3-5`) and on Python bindings that expose it.
- Proposed fix: port the DOP853 tableau (M) or delete the function and the claim (S).

### N1-3: No implicit Radau/SDIRK/extrapolation integrators; `isolve` family absent
- Area: flexmesa-num
- Severity: unverified claim
- FlexMESA: `grep -rli "radau|seulex|sdirk|isolve" crates` returns nothing.
- Upstream: `num/public/num_isolve.dek`, `num/private/{decomr,decomc,slvrad,estrad,slvseu}.dek`, `num_radau5_debug_routine.dek`
- Confidence: verified-by-reading-flexmesa-only (absence), upstream file list read
- What: Upstream `num` ships the Hairer stiff family behind `isolve` (ros2..rodasp plus radau5/seulex-era support files). FlexMESA has the seven Rosenbrock variants only. Honest in `lib.rs` (it does not claim radau), but any "num module complete" statement in notes should be read as "Rosenbrock + explicit RK only".
- Proposed fix: none required; document as not ported (S).

### N1-4: Rosenbrock step-size controller differs from upstream (no Gustafsson predictive controller, different reject rule, no singular-matrix retry, finite-difference df/dt)
- Area: flexmesa-num
- Severity: wrong numerics (relative to "port" claim; algebraically a valid controller)
- FlexMESA: `crates/flexmesa-num/src/rosenbrock.rs:444-455` (`let fac = (0.9 * err.max(1e-10).powf(-1.0 / rc.order)).clamp(0.2, 3.0); ... h = posneg * (h.abs() * fac.min(0.5))` on reject), `rosenbrock.rs:352-361` (`dt_eps = 1e-8 * t.abs().max(1.0); f_dt = fcn(t + dt_eps, &y)` every step, even for autonomous systems), `rosenbrock.rs:363` (`factor_step(...)?` propagates LU failure)
- Upstream: `num/private/mod_rosenbrock.f:1051` (`fac=max(fac2,min(fac1,pow(err,eloi)/safe))`), `:1064` (`hnew=h/fac`), `:1075-1084` (`if (pred) ... facgus=(hacc/h)*pow(err**2/erracc,eloi)/safe; fac=max(fac,facgus)`; `pred=.true.` by default at `:378-379`), `:1126` (`if (reject) hnew=posneg*min(abs(hnew),abs(h))`), `:1139-1142` (`nsing=nsing+1; if (nsing >= 5) GOTO 176; h=h*0.5d0`), autonomous flag `autnms` and user `dfx` (`:692`)
- Confidence: verified-by-reading-both-sides
- What: The basic elementary controller (safety 0.9, ratio in [0.2, 3]) matches. Missing: (a) Gustafsson predictive controller, on by default upstream; (b) after a rejected step upstream forbids the next accepted step from growing (`min(|hnew|,|h|)`), FlexMESA instead caps the shrink factor at 0.5 on the rejected step itself; (c) on a singular stage matrix upstream halves h up to 5 times, FlexMESA returns `LinearSolveFailed`; (d) FlexMESA spends an extra RHS evaluation per step on a forward-difference df/dt with a fixed `1e-8` increment and does it even for autonomous problems, upstream uses the user-supplied `dfx` or skips it. Also `y_min`/`y_max` rejection and dense output are absent. Step sequences therefore differ from MESA's. Note also (N1-5) that the nuclear burner does not use this code at all.
- Proposed fix: port the `pred` branch and reject rule, add an `autonomous` flag (S/M).

### N1-5: Two independent Rosenbrock implementations (num vs net); the one in `flexmesa-num` is not used by any physics crate
- Area: flexmesa-num / flexmesa-net
- Severity: maintainability
- FlexMESA: `crates/flexmesa-net/src/burn_const_p.rs:15` (`use flexmesa_num::RosenbrockVariant;` only the enum), `burn_const_p.rs:370,391` (own stage loop `tab.ra[is][k].mul_add(ak[k][j], s)`), `crates/flexmesa-net/src/rosenbrock_tableau.rs:5`; vs `crates/flexmesa-num/src/rosenbrock.rs:279-458` and `rosenbrock_coeffs.rs`.
- Confidence: verified-by-reading-flexmesa-only
- What: `grep "use flexmesa_num"` across the workspace shows the only consumers of `flexmesa-num` integrators are `dopri5` in atm. The net burner carries its own tableau and stage loop (and uses `mul_add`, i.e. FMA contraction, which the num version does not, so the two would not even agree bitwise). Tests of `flexmesa-num::rosenbrock_*` therefore say nothing about the burner.
- Proposed fix: make the net burner call one generic Rosenbrock core (M, removes ~450 lines) or delete the unused one.

### N1-6: `flexmesa-math` ports the non-default `math_lib_intrinsic` file and physics crates mostly bypass it; bit-level parity with crlibm MESA is not attainable as built
- Area: flexmesa-math and all physics crates
- Severity: unverified claim
- FlexMESA: `crates/flexmesa-math/src/lib.rs:4` ("ports MESA's `math` module (`math/public/math_lib_intrinsic.f90` ...)"), `lib.rs` `pub fn pow(x, y) -> f64 { x.powf(y) }`, `pub fn powi(x, n) { x.powi(n) }`, `pub fn log1p(x) { (1.0 + x).ln() }`, `pub fn expm1(x) { x.exp() - 1.0 }`, `exp10` integer branch `10.0_f64.powi(ix.abs())`. Workspace call counts outside tests/bench/py (grep -F): `.powf(` 234, `.powi(` 375, `.exp()` 371, `.ln()` 171, `.log10()` 162, versus `flexmesa_math::` 62. `mul_add` (explicit FMA) appears 17 times, e.g. `crates/flexmesa-net/src/burn_const_p.rs:223-237,370,391`.
- Upstream: `math/public/math_lib_crmath.f90:219-254` (`pow_r_`: `if (x == 0._dp) pow_x = 0._dp` ... integer `y` with `ABS(iy) < 100` uses a repeated-multiply loop, else `pow_x = exp(log(x)*y)` with crlibm `exp`/`log`), `:163-189` (`exp10_`: integer `x` uses a multiply-by-10 loop, else `exp(x*ln10_m)`), `:192-216` (`pow_i_`: `x == 0 -> 0`, linear multiply loop); default build links crlibm (`USE_CRMATH`), `math_lib_intrinsic.f90:185,234` is the fallback (`LOG(1._dp + x)`, `EXP(x) - 1._dp`).
- Confidence: verified-by-reading-both-sides
- What: (1) Rust `f64::exp/ln/powf/log10/sin...` call the platform libm (glibc on Linux), which is not correctly rounded; MESA's default build uses crlibm. Results agree to roughly 1 ULP per call, not bitwise, and FlexMESA results become platform/libm dependent, the exact property crlibm exists to remove. (2) `powf(x, y)` is a different function from upstream `pow`: upstream evaluates integer exponents by repeated multiplication and everything else as `exp(log(x)*y)` (error of a few ULP amplified by `y`), and returns `0` for `x == 0` for every `y` (so `pow(0,0)=0`, `pow(0,-1)=0`), where `powf` gives `1` and `inf`. `powi` uses LLVM binary exponentiation, upstream a linear loop. (3) About 1300 direct std calls in physics crates never go through `flexmesa-math`, so the crate cannot be swapped for a correctly rounded implementation without touching every crate. (4) `mul_add` in the burner introduces FMA where gfortran-built MESA (no `-ffast-math`, x86-64 baseline) has none. Consequence: any "0 ULP"/"bit-for-bit" statement against Fortran MESA can only hold by coincidence on specific inputs; parity tests at `1e-10`..`1e-13` are consistent with this but trajectories over thousands of steps will decorrelate at roundoff level. `docs/design/scoping-and-architecture.md:172` describes the crlibm requirement and `notes/DECISIONS.md:14` mentions "strict IEEE-754 bit-for-bit parity (`crlibm` / disabled FMA)" as a goal; neither is implemented.
- Proposed fix: decide and record: either (a) depend on a correctly rounded libm (`core-math` crate) inside `flexmesa-math`, implement upstream `pow`/`powi`/`exp10` semantics there, and lint-ban direct `f64::{exp,ln,powf,...}` in physics crates (L, mechanical), or (b) drop bit-parity language everywhere and state "libm-level agreement" (S). Independently fix `pow(0, y)` semantics where upstream relies on it.

### N1-7: `log1p` and `expm1` are implemented naively (catastrophic cancellation for small |x|)
- Area: flexmesa-math, flexmesa-autodiff
- Severity: wrong numerics
- FlexMESA: `crates/flexmesa-math/src/lib.rs` `pub fn log1p(x: f64) -> f64 { (1.0 + x).ln() }`, `pub fn expm1(x: f64) -> f64 { x.exp() - 1.0 }` (doc comments at lib.rs:112,133 say "matching MESA's `log1p`/`expm1`"); consumed by `crates/flexmesa-autodiff/src/kernels.rs:13`, `kernels3.rs:11,65,116`.
- Upstream: `math/public/math_lib_crmath.f90:68,72` (crlibm `log1p`/`expm1`, correctly rounded) in the default build; the naive forms exist only in `math_lib_intrinsic.f90:185,234`.
- Confidence: verified-by-reading-both-sides
- What: For |x| < 1e-8 the Rust versions return 0 or lose most digits (relative error ~ 1e-16/|x|) where default MESA returns the correctly rounded value. Rust has `f64::ln_1p`/`exp_m1`; they are used only 5 times in the workspace. Low traffic today (dual-number kernels), but it is a silent accuracy trap for any caller that picks these for the reason they exist.
- Proposed fix: `x.ln_1p()` / `x.exp_m1()` (S, 2 lines + tests).

### N1-8: bcyclic and block-Thomas are algebraically faithful but not operation-order identical to upstream; singular pivots behave differently
- Area: flexmesa-mtx
- Severity: cosmetic (roundoff) / maintainability
- FlexMESA: `crates/flexmesa-mtx/src/dense.rs:56-80` (partial pivoting, `if max_val == 0.0 { return Err(SingularMatrix(k)) }`, `let mult = a[i*n+k] / diag;`), `bcyclic.rs:74-88` (odd row: factor D, solve for L, U, b), `bcyclic.rs:116-143` (`sub_block_mm(nvar, l_k, u_jm, d_k)` then `sub_block_mm(nvar, u_k, l_jp, d_k)`)
- Upstream: `mtx/public/mtx_solve_routines.inc:10-45` (`jp = j - 1 + maxloc(abs(a(j:lda,j)),dim=1)`; `da = one / a(j,j)`; `a(j+i,j) = da*a(j+i,j)`; zero pivot sets `info = j` and continues), `star/private/star_bcyclic.f90:277-298` (factor even 1-based blocks), `:311-357` (`lmat = -D^{-1} lmat`, `umat = -D^{-1} umat`), `:420-430` (`dmat = dmat + matmul(umat0, lnext) + matmul(lmat0,uprev)` via `my_gemm_plus_mm`), `:606-617` (default `my_getf2`; optional `use_DGESVX_in_bcyclic` with equilibration and rcond reporting)
- Confidence: verified-by-reading-both-sides
- What: Matches: same odd-even reduction (0-based odd rows eliminated each level), same unblocked partial-pivot LU with first-maximum tie-breaking, same default (no equilibration; upstream `use_DGESVX_in_bcyclic = .false.` at `star/defaults/controls.defaults:8885`). Differences: multiplier computed by division rather than multiplication by a reciprocal; Schur update accumulates the two products in a different order; no specialised n=4/n=5 kernels. These give last-bit differences only. Behavioural gaps: no DGESVX/equilibration path, no rcond reporting (`rcond_dense` exists in `dense.rs:271` but is not wired to the star solver), and `DecsolOption::BlockThomasRefine` (`lib.rs:83-85`) is a name with no implementation (`grep refine crates/flexmesa-mtx/src` hits only the enum). Parallel bcyclic is deterministic (each row's work is independent; no reductions), so thread count cannot change results.
- Proposed fix: remove the unimplemented `BlockThomasRefine` variant or implement it (S); optionally expose min-rcond per solve for diagnostics (S).

### N1-9: mtx hot-path overhead: O(n^3) redundant finiteness scans and per-call allocation of seven full block arrays
- Area: flexmesa-mtx (performance)
- Severity: maintainability
- FlexMESA: `crates/flexmesa-mtx/src/dense.rs:116` (`if is_bad_slice(lu) || is_bad_slice(rhs)` inside `dgetrs_trans`), `dense.rs:209-218` (`dgetrs_multi` copies each column into a temp and calls `dgetrs` per column), `dense.rs:50` (`is_bad_slice(a)` in `dgetrf`), `bcyclic.rs:197-204` (`l_work, d_work, u_work, b_work, l_tmp, u_tmp, b_tmp, ipiv_work` allocated per solve), `block_tridiag.rs:331-334`
- Confidence: verified-by-reading-flexmesa-only
- What: Every block solve `D^{-1} U` goes column by column through `dgetrs`, and each call rescans the whole `n*n` LU for NaN/Inf: `n` columns x `n^2` scan = an extra O(n^3) pass per block, twice per odd row (L and U), on top of the entry scan in `solve_block_tridiag_bcyclic`. For nvar ~ 30 (approx21 coupled) this is comparable to the factorisation itself. The solver also has no factor/solve split for bcyclic, so each Newton iteration reallocates ~5 x nz x nvar^2 doubles (nz=2000, nvar=30: ~72 MB of alloc+copy per iteration). The bench crate works around this with a private `fast_block_thomas_solve` (`crates/flexmesa-bench/src/tridiag_fast.rs:67`) instead of fixing the library (see C section).
- Proposed fix: validate once at the API boundary, add unchecked inner kernels, add a reusable `BcyclicWorkspace` (M).

### N1-10: Data pipeline: converters for the released binary bundles are not in the repository; tar bundles carry no per-file hashes
- Area: flexmesa data (`src/flexmesa/data`)
- Severity: unverified claim
- FlexMESA: `src/flexmesa/data/bundles.toml:36-71` (`eos_opal_scvh`, `eos_cms`, `eos_free_eos`, `mesa_test_inputs` all have `unpacked_sha256 = ""`), `src/flexmesa/data/fetch.py:197-201` (`is_bundle_ready`: stamp file + sentinel existence), `fetch.py:229-230` (`if ok and b.kind == "single_zst" and b.unpacked_sha256:` is the only content check after install), `bundles.toml:85` ("3,142 upstream MESA r24.08.1 golden test_output ... files"). `grep -rln "FMDTRW01|FMCMSR01"` finds only readers (`crates/flexmesa-eos/src/tables_4d/table_io.rs:21`, `cms/table.rs`); no writer for the `eos_4d_raw/*.raw` format exists in the tree. `src/flexmesa/data/converters.py` only covers `lodders03`, `isotopes` and the small `helm_table` safetensors.
- Upstream: n/a (data files are LFS stubs in the reference checkout)
- Confidence: verified-by-reading-flexmesa-only
- What: (1) The 116 `*.raw` EOS slices (2.2 GB) that all 4D EOS results rest on were produced by a tool that is not in the repository, so the mapping from upstream `*.data` to `FMDTRW01` (column order, any rounding, which upstream commit) cannot be re-derived or audited; only the archive hash is pinned. (2) After unpacking, `flexmesa data verify` checks tar bundles by stamp and two sentinel paths only; a truncated, edited or partially overwritten slice passes verification. (3) The bundle label says "r24.08.1" while the audited reference is commit `0b2e6bb` (brief); there is no commit hash or `MESA_VERSION` recorded in `bundles.toml` or inside the binary headers (the 84-byte `FMDTRW01` header holds only table `version`, X, Z and grid extents, `table_io.rs:137-158`).
- Proposed fix: commit the converter scripts and a per-file SHA-256 manifest inside each tar bundle; record the upstream commit hash in `bundles.toml` and in each binary header; make `verify` hash every file (M).

### N1-11: Runtime spline caches are written into the data directory, are unversioned and never verified, and take precedence over the verified raw data
- Area: flexmesa-eos, flexmesa-kap (data caches)
- Severity: wrong numerics (latent) / maintainability
- FlexMESA: `crates/flexmesa-eos/src/tables_4d/slice_cache.rs:47-59` (`let slice_bin = flex_dir.join("eos_4d_slices").join(format!("{stem}.bin")); if slice_bin.is_file() { if let Ok(tbl) = read_dt_slice_bin(&slice_bin) { return Ok(tbl); } } ... let _ = write_dt_slice_bin_atomic(&slice_bin, &tbl);`), `slice_cache.rs:17-21` (magic `FMDT4D01`/`FMCMS001` only), `crates/flexmesa-eos/src/helm/table_io.rs:76` (`let _ = write_helm_full_bin_atomic(&full_bin, &vertices);`), `crates/flexmesa-kap/src/loader_ascii.rs:79-92` (writes `kap_full/kap_tables_full.bin` and `aesopus_section.bin` when built from `MESA_DIR`). Observed on this machine: `the data cache/eos_4d_slices/` exists with `.bin` files created during a run, next to the fetched `eos_4d_raw/`.
- Confidence: verified-by-reading-flexmesa-only
- What: The first EOS call for each (Z, X) slice fits 16 bicubic splines and writes the fitted coefficients into `$FLEXMESA_DATA_DIR/eos_4d_slices/`. Later runs load the cache first and never look at the raw file again. The cache key is the file stem; there is no hash of the raw input, no code/format version beyond an 8-byte magic, and `flexmesa data verify` does not know the directory exists. Consequences: (a) a change to the spline fit, the column mapping (`table_io.rs:31-47`) or the NaN handling leaves old coefficients in use silently, so two checkouts sharing a data dir can give different physics and tests can pass or fail depending on cache history; (b) the cache can also be populated from an upstream Fortran cache (`slice_cache.rs:67-86`, `read_mesa_eosdt_cache_bin`) or ASCII when `MESA_DIR` is set, which is a different provenance from the pinned bundle, and afterwards is indistinguishable; (c) the "verified data directory" is mutated by ordinary runs, and concurrent first runs race (atomic rename makes this safe for content, not for provenance). The same pattern lets a run with `MESA_DIR` set overwrite/create `helm_table_full.bin` and `kap_tables_full.bin` in the verified location.
- Proposed fix: move derived caches to a separate `$XDG_CACHE_HOME/flexmesa/derived/<code-version>/` tree, key by SHA-256 of the raw file plus a builder version, and have `verify` report or purge them (M).

### N1-12: 4D EOS table build silently zeroes non-finite spline coefficients
- Area: flexmesa-eos tables_4d
- Severity: wrong numerics (latent)
- FlexMESA: `crates/flexmesa-eos/src/tables_4d/table_io.rs:98-101` (`dense[cell].coeffs[v * 4 + m] = if is_bad(val) { 0.0 } else { val };`)
- Upstream: not checked line by line (`eos/private/eosdt_load_tables.f90`); see Leads.
- Confidence: verified-by-reading-flexmesa-only
- What: If a raw table cell or the bicubic fit yields NaN/Inf, the coefficient becomes 0 and the table loads without any diagnostic. A zero for `lnPgas`/`lnE` function value corresponds to P = 1 dyn/cm^2 or E = 1 erg/g: physically absurd values delivered as valid EOS results. Because the result is then cached (N1-11) the corruption is permanent and invisible.
- Proposed fix: return an error naming table, variable and cell (S).

### N1-13: The opacity loader shells out to `python3` with an embedded script and stores AESOPUS opacities as float32
- Area: flexmesa-kap data builder
- Severity: wrong numerics (small, unquantified) / maintainability
- FlexMESA: `crates/flexmesa-kap/src/loader_ascii.rs:298-322` (embedded Python: `kap = np.asarray(g["kap"][:], dtype=np.float32)`), `:333-349` (`Command::new(bin).args(["-c", py_script, ...])` trying `$PYTHON`, `$VIRTUAL_ENV/bin/python`, `python3`), `crates/flexmesa-kap/src/loader.rs:180-200` (`raw_f32: Option<Vec<f32>>`, widened with `f64::from(v)` before the spline fit), `loader_parse.rs:294-296`.
- Upstream: `kap/private/kap_aesopus.f90:79` (`real(dp), allocatable :: table_data(:,:,:,:,:,:)`), `:257` (`call hi_ts% alloc_read_dset('kap', table_data)`): values are consumed as double precision.
- Confidence: verified-by-reading-both-sides (the dtype stored in the upstream HDF5 file could not be inspected; LFS stub)
- What: The AESOPUS log-opacity grids in the released `kap_tables_full.bin` are float32. Unless the upstream HDF5 dataset is itself float32, this is a lossy quantisation at ~6e-8 relative in log kappa (so ~1e-7 in kappa), below typical parity tolerances for tables but well above the `1e-10`/`1e-13` figures quoted for kap parity; all other kap sections are f64 (`loader_ascii.rs:112-165`). Separately, a library crate that executes whatever `$PYTHON` points to, at table-load time inside a physics call, is a surprising side effect for a `#![forbid(unsafe_code)]` numerical library (and needs `h5py` installed).
- Proposed fix: store f64 (costs ~45 MB) or prove the source is f32 and say so in the bundle description; move the HDF5 conversion into the offline converter (N1-10) (S/M).

### N1-14: Table loaders panic on missing or malformed data
- Area: flexmesa-kap
- Severity: maintainability
- FlexMESA: `crates/flexmesa-kap/src/loader_ascii.rs:98-104` (`panic!("FlexMESA opacity tables not found at {} ...")`), `loader_parse.rs:72` (`assert_eq!(&bytes[..8], b"FMKAP001", "invalid kap_tables.bin header")`), `loader.rs:202` (`.expect("bicubic spline fit failed on AESOPUS slice")`); contrast `crates/flexmesa-eos/src/helm/table_io.rs:81-86` which returns `Err`.
- Confidence: verified-by-reading-flexmesa-only
- What: A user who has not run `flexmesa data fetch`, or has a truncated file, gets a Rust panic from inside a PyO3 call (surfacing as `PanicException`, which is not an `Exception` subclass and bypasses normal `except Exception` handling) rather than a typed error. The cursor reads in `loader_parse.rs` also index without bounds checks, so a short file panics with an index error rather than a message.
- Proposed fix: return `KapError::MissingTable`/`ParseError` (S).

## Matches (Scope A so far)

- `flexmesa-const`: every constant in `crates/flexmesa-const/src/lib.rs:14-292` has the same literal or the same defining expression (including operation order, e.g. `BOLTZ_SIGMA` at lib.rs:151-152 vs `const_def.f90:86`; `QE` lib.rs:108 vs `:74`; `PC` lib.rs:164 vs `:91`; `MSUN = MU_SUN/STANDARD_CGRAV` lib.rs:230 vs `:124`; `LOGGSUN` literal lib.rs:254 vs `:128`) as `const/public/const_def.f90:44-144`; mixing-type codes match `:155-169`. verified-by-reading-both-sides.
- `safe_log`, `safe_log10` floor `1e-99` and bad-value return `-99` (`flexmesa-math/src/lib.rs`) match `math_lib_crmath.f90:127-160`.
- `exp10` non-integer branch `exp(x*ln10)` matches `math_lib_crmath.f90:185`; integer branch gives the same value for |x| <= 22 (exact powers), may differ in the last bit beyond that (powi squaring vs linear loop).
- dopri5 controller: `fac11 = err^0.17`, `fac = fac11/facold^0.04`, clamp to [0.1, 5] with safety 0.9, reject rule `h/min(5, fac11/0.9)`, `facold = max(err, 1e-4)`, no-growth after reject (`rk.rs:281-321`) match Hairer's DOPRI5 defaults as used in `num/private/mod_dopri5.f` (beta = 0.04). Initial step when `h0 == 0` is a fixed `1e-4` rather than `hinit` (minor).
- Rosenbrock elementary controller constants (safety 0.9, ratio bounds 0.2 and 3, RMS norm with `atol + rtol*max(|y|,|ynew|)`) match `mod_rosenbrock.f:413-431,1051`.
- Nelder-Mead simplex: reflection/expansion/contraction formulae, centroid weighting option and adaptive-random-search hook (`simplex.rs:260-314`) correspond to `num/private/mod_simplex.f90:134-139,288-321,338-364`; Park-Miller constants match (`simplex.rs:30-47`).
- Block-Thomas and bcyclic: algebra matches (N1-8). Pivot rule matches `my_getf2`.
- `resolve`/fetch: archive SHA-256 is verified before install (`fetch.py:276-281`), single-file bundles also verify the unpacked hash (`fetch.py:305-311`).


## Scope C (priority parts): determinism, hot-path cost, error design

### N1-15: Thread-local `NetWorkState`: stale-state leak? No for the paths in use; one latent hole in the reuse key
- Area: flexmesa-net / flexmesa-star (determinism)
- Severity: maintainability (latent wrong numerics)
- FlexMESA: `crates/flexmesa-net/src/derivs.rs:181-203` (`with_thread_local_work`; reuse test is `w.y.len() != g.num_isos || w.rate_raw.len() < g.num_reactions.max(120) || w.wk_lambda.len() != g.num_wk_reactions`), `derivs.rs:222-229` (`w.reset(g); dydt[..ni].fill(...)`), `crates/flexmesa-net/src/derivs_support.rs:141-152` (`reset` zeroes `eps_nuc_categories, eps_neu_total, d_eps_nuc_dy, d_dydt_dy, raw_rate, screened_rate, eps_nuc_rate, eps_neu_rate`), fields not reset: `y`, `rate_raw*`, `rate_screened*`, `wk_*`.
- Confidence: verified-by-reading-flexmesa-only
- What (answer to the brief's yes/no): **No leak on the general-net and approx21 paths as long as one thread only ever sees one network.** Evidence, field by field: `y[i]` is assigned for every isotope (`derivs.rs:49-52`); `rate_raw/rate_raw_dt/rate_raw_drho[i]` are assigned for every reaction with no skip branch (`initialize.rs:532-560`); `rate_screened*[i]` is assigned in all three branches of the per-reaction loop (`screen.rs:122-183`, final `else` copies raw); all seven `wk_*[k]` are assigned for every weak reaction (`derivs_support.rs:299-331`); the accumulators are zeroed by `reset`. For approx21 the 23 derived slots `m..m+22` (`approx21.rs:299-321`) are handled as follows: `m..m+8` are assigned in both branches of `set1/set2` (`approx21.rs:331-381`); `m+9..m+22` are never written in `rate_raw`, so they keep their constructor value 0, and `approx21_eval.rs:152-156` copies raw to screened for all `m..total_r` on every call, which re-zeroes whatever `approx21_special_reactions` wrote into `rate_screened[m+9..]` on the previous zone. `irpen/irnep/irn56ec/irco56ec` are assigned unconditionally (`approx21.rs:397-467`). With `just_dxdt` the Jacobian outputs are zero-filled by the caller (`micro_net.rs:213-217`). So results do not depend on which rayon worker handled the previous zone.
  The latent hole: the reuse key compares sizes, not network identity, and uses `<` for `rate_raw.len()`. If a thread first evaluates a general net with the same `num_isos` and `num_wk_reactions` but more than `m+9` reactions, then an approx21 net, `rate_raw[m+9..m+22]` holds leftover real rates from the first net and is copied into `rate_screened` as the starting value for the derived reactions. That needs two different nets of equal isotope count in one process (binary with different nets per star, net change during a run, or the Python API evaluating several nets), so it is not reachable in the single-net runs that were measured bit-identical, but nothing prevents it.
- Proposed fix: zero `rate_raw[m..total_r]` at the top of `eval_net_approx21_procs`, or key the thread-local on a network id/generation; add a test that interleaves two nets on one thread (S).

### N1-16: Other determinism checks (thread-local EOS cache, rayon, HashMap, time)
- Area: workspace
- Severity: cosmetic (mostly passes)
- FlexMESA: `crates/flexmesa-eos/src/tables_4d/table_set.rs:28-33` (`TL_FREE_CACHE: RefCell<VecDeque<(TlFreeKey, Arc<EosDtXzTable>)>>`); rayon call sites: `flexmesa-star/src/micro_net.rs:87-101`, `micro.rs:362,422`, `struct_burn_mix.rs:128`, `riemann.rs:262,306`, `flexmesa-mtx/src/bcyclic.rs:212-262`, plus table builders; `crates/flexmesa-net/src/initialize.rs:276-283` (`rxn_map.values().cloned().collect(); reactions.sort_by(|a, b| a.name.cmp(&b.name))`), `initialize.rs:269-270` (`chem_id.sort_unstable()`); `flexmesa-star/src/inlist_runner.rs:221,335,406,515` (`Instant` used only for `runtime_sec`).
- Confidence: verified-by-reading-flexmesa-only
- What: (a) The EOS thread-local holds `Arc`s to immutable tables keyed by (set id, ix, iz); it affects speed only. (b) Every rayon use I found is a per-index map into disjoint output slots (`par_iter_mut`/`par_chunks_mut`/indexed `collect`); `grep` for `.sum()/.reduce()/.fold()/min_by/max_by/find_any` following a parallel iterator finds none outside tests and bench, so there are no order-dependent floating-point reductions. (c) HashMap/HashSet contents that feed numerics are sorted before use (isotopes by chem id, reactions by name); inlist sections are looked up by key, not iterated. (d) Wall-clock is only reported. Two residual nondeterminisms, both diagnostic only: `try_for_each` over zones (`micro_net.rs:101`, `bcyclic.rs:221`) returns whichever failing zone a worker reports first, so the zone index in an error message can vary between runs; and on such an error other zones have already been overwritten (harmless only because callers restore from `*_old`). One caveat on (c): sorting reactions by name is deterministic but is not upstream's reaction order, which changes floating-point summation order relative to MESA (see Leads).
- Proposed fix: collect the minimum failing zone index for error messages (S).

### N1-17: `StarModel` is cloned whole every step that is allowed to remesh; the binary driver clones both stars two to three times per step and again on every implicit-RLOF trial
- Area: flexmesa-star, flexmesa-binary (hot path)
- Severity: maintainability (performance)
- FlexMESA: `crates/flexmesa-star/src/evolve_support.rs:315-323` (`let snap = model.clone(); let (ns, nm) = do_adjust_mesh(model, &mesh_cfg)?; if ns + nm > 0 { ... return Ok((ns, nm, Some(snap))); } Ok((ns, nm, None))`), `crates/flexmesa-binary/src/system_step.rs:74` (`let pre_step_snap = self.capture_snapshot();`), `:148` (`let mut trial_snap = self.capture_snapshot();` inside the outer retry loop), `:155,199,213` (`self.restore_snapshot(&trial_snap)?` per implicit try), `:215` (re-capture after donor switch), `crates/flexmesa-binary/src/system_rlof.rs:161-181` (`restore_stars`: `let d_ctrl = donor.controls.clone(); let d_job = donor.job.clone(); *donor = self.donor_state.clone(); donor.controls = d_ctrl; donor.job = d_job;`)
- Confidence: verified-by-reading-flexmesa-only
- What: Size of one `StarModel` (field count by script over the struct definitions): `StarModel` in `model.rs:32` has 154 fields of which 124 are `Vec`; its `HydroAuxState` (`hydro_state.rs`) adds 71 fields, 59 `Vec`; `StarControls` has 254 top-level fields (10 `Vec`, many `String`s and nested structs) and `StarJobControls` 86. So a clone is ~183 heap allocations plus the controls strings. Bytes: most vectors are `nz` long; about ten are `nz*species`; `d_dxdt_nuc_dx` is `nz*species^2` (`model.rs:527`). For nz = 2000, species = 21: ~183 x 16 kB = 2.9 MB, plus ~3.4 MB species-sized, plus 7 MB for the Jacobian, plus per-zone EOS derivative arrays: roughly 15-20 MB and ~200 mallocs per clone. Single star: one full clone per step whenever remeshing is permitted, and it is thrown away when no cell was split or merged (the common case). Binary with two stars: `pre_step_snap` (2 clones) + `trial_snap` (2 clones) per outer retry, then each implicit trial beyond the first restores by cloning the snapshot again (2 clones) after first cloning the live `controls` and `job` (which are then written back, i.e. controls are deliberately not rolled back; anything a trial changed in `controls` survives the "restore"). A 10-try implicit RLOF step is ~24 full model clones, ~0.4 GB of memcpy at the sizes above. Compared with a Newton iteration this is not dominant, but it is avoidable: the single-star driver already has a cheap `*_old` restore (`evolve.rs:76-113`) that the binary driver does not use.
- Proposed fix: (1) clone lazily in `remesh_star_before_step` (only the fields `smooth_post_remesh_hydro_and_rotation` reads, or clone after `do_adjust_mesh` reports a planned change); (2) give `StarModel` a `StepState` sub-struct holding only evolved state (xh, xa, dq, omega, j_rot, mdot, age, step_hist, RSP state) and snapshot that with `clone_from` into a reusable buffer; (3) restore with `clone_from` instead of `clone` so allocations are reused (M).

### N1-18: Full microphysics is recomputed several times per step on an unchanged state
- Area: flexmesa-star (hot path)
- Severity: maintainability (performance)
- FlexMESA: `crates/flexmesa-star/src/hydro_vars.rs:481-494` (`set_hydro_vars` = `set_basic_vars; do_eos; do_kap; do_neu; do_net(... jacobian ...)` over all zones), called at `evolve_support.rs:311` (before mesh adjustment), `evolve_support.rs:319` (after a mesh change), `adjust_mass.rs:381`, `evolve_support.rs:203` (after pre-physics hooks when premix/rotation ran), `struct_burn_mix.rs:513` (start of `do_struct_burn_mix`), `hydro_eqns.rs:572` (every residual evaluation, including each line-search trial), `struct_burn_mix.rs:547` (after convergence), `evolve.rs:114` (inside `restore_from_old` on every retry).
- Confidence: verified-by-reading-flexmesa-only
- What: On an ordinary step with no mesh change and no mass change, the state evaluated at `evolve_support.rs:311` is evaluated again at `struct_burn_mix.rs:513` and a third time by the first `eval_henyey_equations_into` (`hydro_eqns.rs:572`) in `solve_henyey_step` (`solver.rs:238`), each including the network with its dense Jacobian. With 3 to 5 Newton iterations per step that is roughly 40-60% extra microphysics. Line-search trials also pay for the full net Jacobian (`skip_net_jacobian` is only set when `!include_chem`) although only the residual is needed to accept or reject a trial.
- Proposed fix: add a `micro_valid_for` stamp (hash or generation counter bumped by anything that mutates `xh/xa/dq`) and skip `set_hydro_vars` when current; evaluate line-search trials with `skip_net_jacobian = true` and compute the Jacobian once for the accepted point (M).

### N1-19: Error types carry no location; the reason for a failed attempt is discarded by the retry loop
- Area: flexmesa-star, flexmesa-binary (diagnosability)
- Severity: maintainability
- FlexMESA: `crates/flexmesa-star/src/job_controls.rs:50-61` (`pub enum StarError { InvalidInput(String), IoError(String), MicrophysicsError(String), ConvergenceFailure(String) }`), `solver.rs:27-40` (`HenyeySolverReport { converged, iterations, residual_norm, max_residual, correction_norm, max_correction }`: no zone or variable index for the maxima), `solver.rs:363-365` (`"line search failed at Newton iteration {iter}"`), `micro.rs:237` (`format!("do_eos failed: {e}")`, no zone), `solver.rs:199-203` (`"block-tridiagonal linear solve failed: {e}"`; `MtxError::SingularMatrix(idx)` is a block row for bcyclic but a within-block pivot for Thomas, `block_tridiag.rs:250`), `evolve.rs:283-287` (`break other;` then `if let Ok(rep) = attempt { ... }` with no `else`: an `Err` from the attempt is dropped), `evolve.rs:323-329` (final error: `"star_evolve_step failed after {retries}/{max_retries} retries (dt=..., dt_floor=...)"`), `evolve.rs:310` (`let _ = restore_from_old(model);`).
- Upstream: for contrast, `star/private/star_solver.f90:140` (`max_corr_k, max_corr_j, max_resid_k, max_resid_j`), `:276` (`call sizequ(s, nvar, residual_norm, max_residual, max_resid_k, max_resid_j, ierr)`), `:516-517` (`s% retry_message = trim(message) // ' in solver'`).
- Confidence: verified-by-reading-both-sides (upstream by grep of the cited lines)
- What: When a step fails after N retries the caller learns only the count and the last dt. The per-attempt cause (EOS off-table at zone k, singular block, line-search failure, non-convergence with max residual in equation j at zone k, timestep-controller veto) is computed and then thrown away at `evolve.rs:283-287`. Non-convergence itself is not an error at all (`Ok(report)` with `converged: false`), so there is nothing to attach location to. Messages are free-form strings, so a test or driver cannot branch on "which equation" without parsing text. There is no per-step log: `grep` for `eprintln!/println!/log::/tracing::` in `flexmesa-star/src` finds ~20 ad hoc prints in a handful of files and no logging facade. `model.num_retries` is the only counter and it is carried in the model, so it is rolled back/overwritten by snapshot restores in the binary driver except where explicitly re-patched (`evolve.rs:335-337`).
- Proposed design (M/L):
  1. Structured error: `StarError { kind: ErrKind, ctx: ErrCtx, source: Option<Box<dyn Error>> }` with `ErrCtx { model_number, star_id, dt, retry, newton_iter: Option<usize>, zone: Option<usize>, var_or_eqn: Option<&'static str>, value: Option<f64> }`. Sub-crate errors (`EosError`, `KapError`, `NetError`, `MtxError`) stay typed and are attached as `source`, not formatted into strings. Zone-parallel loops return the minimum failing zone.
  2. Solver report: add `max_residual_at: (zone, eqn)`, `max_correction_at: (zone, var)`, `corr_coeff`, `n_line_search`, and a `StopReason` enum (`Converged | MaxIters | LineSearch | LinearSolve{row} | Micro{zone}`), returned for failed attempts too.
  3. Retry ledger that survives restores: a `StepTelemetry` value owned by the driver loop (not inside `StarModel`), holding counters by reason (`retries_by_reason[StopReason]`, `redo_mdot`, `remesh_splits/merges`, `eos_offtable`, `kap_clip`, `burn_substeps`, `newton_iters_total`, `micro_evals`) and the list of attempts for the current step `{dt, stop_reason, max_resid_at, max_corr_at}`. Snapshot/restore never touches it. Binary driver owns one per star plus one for the orbit (`implicit_tries`, `donor_switches`, `cut_dt`).
  4. Per-step structured log: one JSON line (or binary row) per accepted step and per rejected attempt, written by the driver behind a `StepSink` trait (`NullSink` default, `JsonlSink`, in-memory sink for tests). This replaces the scattered `eprintln!`.
  5. Checkpoints: see N1-20.

### N1-20: `.photo` restart does not serialise controls or job settings, is a hand-enumerated field list, and there is no on-disk checkpoint for a binary
- Area: flexmesa-star photo_io, flexmesa-binary
- Severity: unverified claim (re "restart parity") / maintainability
- FlexMESA: `crates/flexmesa-star/src/photo_io.rs:250-253` (`pub fn read_photo_bytes(..., controls: StarControls, job: StarJobControls)`), `:81-91` (only `net_name` and seven flags `v_flag, u_flag, rotation_flag, op_split_burn, rsp_flag, rsp2_flag, rti_flag` are written), `:319-330` (those are patched into the caller-supplied controls, then `StarModel::new(nz, chem_id, job, controls)`), `:100-170` (explicit lists of vectors written), `:504-523` (several `*_start` arrays restored only `if saved_lnd_start.len() == nz`, otherwise rebuilt from current state). Binary: `grep -rn "write_photo|read_photo|serde" crates/flexmesa-binary` returns nothing; the only binary snapshot is the in-memory `BinaryStepSnapshot` (`system_rlof.rs:62`).
- Confidence: verified-by-reading-flexmesa-only
- What: (1) A photo is not self-contained: restoring requires the caller to supply the same `StarControls`/`StarJobControls`. Controls that a run mutates in flight (relaxation stages, `tier*_extras` hooks changing controls, `op_split_burn` toggles, retry holds stored in controls) are not captured except the seven flags. A restart test that restores in the same process with the live controls object therefore proves less than it appears to: it shows the enumerated arrays round-trip, not that a fresh process given the inlist reproduces the run. (2) The format is a manual subset of the ~183 vectors; adding a state field to `StarModel` without adding it to both lists silently breaks restart with no compile-time or runtime signal (no field-count or schema hash in the header beyond `STAR_DEF_VERSION`). (3) A `BinarySystem` cannot be checkpointed to disk at all: orbital state, `change_factor`, donor index, `cumulative_j_ext`, point-mass state live only in memory, so a long binary run cannot be resumed and a failed binary step cannot be dumped for post-mortem.
- Proposed fix: derive serialisation (serde or a `#[derive(PhotoState)]` macro) for a dedicated `StepState` struct so the field list cannot drift; embed the effective controls (or their hash plus the diff from defaults) and a schema hash in the header; add `BinarySystem::write_checkpoint` = orbit block + photo per star; on step failure after max retries write `failed_step_<model>.photo` automatically (M).

### N1-21: Discarded `Result`s that hide physics failures
- Area: flexmesa-star, flexmesa-binary
- Severity: wrong numerics (latent) / maintainability
- FlexMESA (counts of `let _ =` outside tests: star 80, binary 12, eos 12, kap 4, net 2; many are file writes). Those that discard physics or input errors:
  - `crates/flexmesa-star/src/relax.rs:537` (`let _ = crate::evolve::star_evolve_step(model, false, ...)` in a 7-step relaxation ladder: a failed step is ignored and the ladder continues)
  - `evolve_support.rs:121` (`let _ = do_conv_premix(...)`), `evolve_support.rs:198` (`let _ = set_rotation_mixing_info(model);`)
  - `premix.rs:390-391` (`let _ = set_basic_vars(model); let _ = do_eos(model);` then energies from that EOS call are differenced into `eps_phase_sep`), `premix.rs:171`
  - `tier2_extras.rs:178` (`let _ = do_phase_separation(...)`), `tier2_extras.rs:526` and `tier1_extras.rs:472` (`let _ = do_element_diffusion(...)`), `tier1_extras.rs:441` (`let _ = do_conv_premix(model, 3, false);`), `tier1_extras.rs:491`
  - `mesh.rs:398` (`let _ = crate::riemann_ad::do_uface_and_pface(self);`), `hydro_state.rs:581` (`let _ = crate::mesh_remap::set_lnt_for_energy(...)`), `rsp2_remesh.rs:23`, `rsp2_eqns.rs:192`, `rsp2_ad.rs:507`, `rsp_grid.rs:362,472,519` (`let _ = model.normalize_dqs();`)
  - `evolve.rs:310` (`let _ = restore_from_old(model);` before returning a terminate error)
  - `inlist.rs:243` (`let _ = Self::load_file_recursive(case_dir, &astero_search, ...)`: a malformed `inlist_astero_search_controls` is silently ignored)
  - `crates/flexmesa-binary/src/system_step.rs:135,146,263,440`, `system.rs:447,484,506,583` (`let _ = self.refresh_tidal_deformation_hooks(false);`), `system_step.rs:214`, `system.rs:531` (`let _ = self.swap_donor_and_accretor();`)
- Confidence: verified-by-reading-flexmesa-only (call sites read; callee failure modes not all traced)
- What: Each of these lets a failed diffusion solve, phase separation, premix, EOS call, tidal-table lookup or donor swap leave the model in a partially updated state and carry on as if it had succeeded. In MESA the corresponding routines set `ierr` and the step is retried. Because the step then usually "converges", these never show up in pass/fail test results.
- Proposed fix: propagate with `?`; where failure is intentionally tolerated, record it in the telemetry ledger (N1-19) and restore the pre-call state (M).

### N1-22: Panics on user-reachable paths
- Area: workspace
- Severity: maintainability
- FlexMESA: counts outside tests: `.unwrap()` star 50, net 38; `.expect(` kap 16, star 9, binary 4, eos 3, net 3; `panic!` kap 1, eos 1. Examples: `crates/flexmesa-net/src/derivs.rs:50` (`nuclides.get_by_id(cid).unwrap()` per isotope per zone), `derivs.rs:153,239`; `derivs.rs:202` (`borrow.as_mut().unwrap()`); `crates/flexmesa-net/src/initialize.rs:569` (`cache_mutex.lock().unwrap()`: a panic in any thread while holding the lock poisons rate-table construction for the whole process); `crates/flexmesa-star/src/micro.rs:36-37` (`.expect("valid built-in basic.net")`); kap loader panics (N1-14). `debug_assert` is used only twice in `flexmesa-star/src`, so release-only loss of checks is not a pattern here.
- Confidence: verified-by-reading-flexmesa-only
- What: Most are "cannot happen with built-in data" unwraps, acceptable individually, but through PyO3 each becomes a `PanicException` and, with the mutex ones, a poisoned global. Slice indexing on user-controlled lengths (e.g. `aion: [f64; 32]` in `derivs.rs:237` indexed by `ni` for any net flagged approx21) is in the same class.
- Proposed fix: convert data-dependent unwraps to typed errors; use `parking_lot` or `lock().unwrap_or_else(PoisonError::into_inner)` for caches (S/M).

## Matches (Scope C)

- Rayon usage is reduction-free and writes disjoint slots (N1-16); parallel bcyclic is order-independent (N1-8).
- `NetWorkState` fields feeding results are fully overwritten per call on the paths in use (N1-15).
- Single-star retry path restores from `*_old` with `clone_from` (allocation reuse), `evolve.rs:84-100`.
- Table cache writes use temp file + atomic rename (`slice_cache.rs:195-228`, `helm/table_io.rs:149-181`).
- Workspace lints: `unsafe_code = "forbid"`, clippy `all = "deny"` (`Cargo.toml:44-50`).

## Scope B: Python/JAX layer (`src/flexmesa/**`, 49,953 lines) and `crates/flexmesa-py`

Per-module character (counts are `grep -c` of `_rust.` calls / `jnp.|jax.` references / `pure_callback|custom_jvp|custom_vjp|stop_gradient` over each package):

| package | lines | `_rust.` | jax refs | cb/custom | what the JAX side is |
| :-- | --: | --: | --: | --: | :-- |
| eos | 9714 | 37 | 2231 | 68 | independent JAX re-implementation of HELM/PC/Skye/blend formulae; table cell lookup goes to Rust through `pure_callback` |
| kap | 3078 | 10 | 718 | 10 | independent JAX kernels with the full option surface; tables loaded via Rust |
| net | 728 | 8 | 34 | 5 | thin wrapper: `pure_callback` to Rust `net_eval` + `custom_vjp` |
| turb | 2368 | 2 | 65 | 0 | mostly pure-Python scalar/dual-number code plus one `set_mlt_jax`; Rust only for the f64 fast path |
| star | 4649 | 84 | 847 | 3 | wrapper over `RustStarModel` + stand-alone JAX formula kernels; no JAX evolution |
| binary | 1227 | 15 | 250 | 2 | wrapper over `RustBinarySystem` + closed-form JAX kernels |
| astero | 1202 | 17 | 165 | 0 | wrapper over Rust + JAX scaling relations |
| mtx | 1768 | 6 | 97 | 6 | independent JAX PCR and Thomas; Rust FFI with IFT `custom_vjp` |
| num | 3614 | 0 | 44 | 4 | third, pure NumPy implementation of the solvers (including the simplified NEWUOA) |
| auto_diff | 1234 | 0 | 116 | 0 | pure-Python dual numbers |

### N1-23: There is no JAX stellar evolution step; "Rust vs JAX star parity < 1e-13" compares one residual formula written twice, fed with Rust-computed microphysics
- Area: `src/flexmesa/star`
- Severity: unverified claim
- FlexMESA: `src/flexmesa/star/jax_evolve.py:18-147` (the whole file is three kernels: `eval_mesh_gval_jax`, `eval_varcontrol_jax`, `eval_h211_timestep_jax`), `src/flexmesa/star/jax_kernels.py:347-377` (`eval_hydro_residuals_jax(dq, dm, dm_bar, m_grav, lnd, ln_t, ln_r, lum, lnd_start, ln_peos, ln_e, energy_start, gradt, eps_nuc, non_nuc_neu, eps_mdot, ln_peos_surf, ln_t_surf, dt, ...)`: pressure, energy, `gradt`, `eps_nuc`, neutrino losses and the surface boundary values are inputs), `:422` ("Evaluate the 4-equation hydrostatic Henyey residual array `(nz, 4)`"), `src/flexmesa/star/model.py:30-40` (`class StarModel(_arrays.StarModelArrays): def __init__(self, rust_model: _rust.RustStarModel)`), `model.py:391-422` (`eval_hydro_residuals_jax` passes `self.ln_peos`, `self.energy`, `self.gradt`, `self.eps_nuc`, ... read back from the Rust model), `tests/test_star_wave4_2.py:32-55` (`sys = model.eval_equations(include_chem=False); equ_jax = np.asarray(model.eval_hydro_residuals_jax()); np.testing.assert_allclose(sys.equ, equ_jax, rtol=1.0e-13, atol=1.0e-13)` on one 4 Msun ZAMS model).
- Confidence: verified-by-reading-flexmesa-only
- What: `StarModel.evolve_step/evolve/solve_henyey/step_struct_burn_mix` all delegate to Rust. The JAX side offers (a) a residual function for the 4 structure equations only (no composition equations, no velocity/TDC/RSP variables, no Newton driver, no coupling to EOS/kap/net: those arrive as arrays computed by Rust), (b) a linear solve, (c) assorted closed-form kernels (winds, rotation factors, HLLC flux, RSP2 terms). The parity test perturbs `lnd`/`ln_t`, lets Rust recompute all microphysics, then checks that the JAX algebra reproduces Rust's residual vector from Rust's own intermediate arrays. That is a self-consistency check of a transcription; it has no upstream content and cannot detect an error common to both (they were written from the same description). `jax.grad` of this function gives partials at fixed P, E, gradT, eps_nuc, not the Henyey Jacobian. CLAUDE.md's Tier B line listing "`star` microphysics/diffusion/rotation/riemann/rsp2" as dual Rust/JAX implementations should be read as "individual formula kernels exist in JAX".
- Proposed fix: describe the JAX star layer as "differentiable diagnostics kernels" in docs; label Rust-vs-JAX tests as consistency tests and keep them out of any "parity with MESA" tally (S).

### N1-24: `net` on the JAX side is a Rust callback with reverse-mode only and frozen composition moments
- Area: `src/flexmesa/net`
- Severity: unverified claim (differentiability scope)
- FlexMESA: `src/flexmesa/net/_jax.py:62-87` (`_rust.net_eval(str(net_name), ..., float(abar), float(zbar), float(z2bar), float(ye), float(eta), float(d_eta_dlnt), float(d_eta_dlnrho), None, float(weak_rate_factor), str(screening_mode), skip_jac)`), `:88-101` (`@jax.custom_vjp def _op(x_in, t_in, rho_in)` using `jax.pure_callback`)
- Confidence: verified-by-reading-flexmesa-only
- What: Gradients flow only with respect to `x`, `T`, `rho`, and only in reverse mode (`custom_vjp`; `jax.jvp`/`jacfwd` will fail). `abar, zbar, z2bar, ye, eta` are converted with `float(...)`, so they are constants of the differentiation even though they are functions of `x`; the `x`-gradient is therefore the partial at fixed screening/weak-rate composition moments (which matches MESA's own `d_dxdt_dx` convention, but is not what `jax.grad` of a "differentiable network" suggests). Being a host callback it cannot run on GPU or be `vmap`ped efficiently. This is a wrapper, not a JAX implementation; the docstring is accurate about using Rust derivatives.
- Proposed fix: document limits; add a `custom_jvp` if forward mode is advertised (S).

### N1-25: EOS/kap JAX kernels are genuine second implementations, but table access is a Rust host callback under `stop_gradient`
- Area: `src/flexmesa/eos`, `src/flexmesa/kap`
- Severity: maintainability / unverified claim (GPU)
- FlexMESA: `src/flexmesa/eos/helm.py:57-104` (`temp_sg = jax.lax.stop_gradient(...)`; unwraps `.primal` in a `while isinstance(temp_sg, jax.core.Tracer) and hasattr(temp_sg, "primal")` loop; falls back to `jax.pure_callback(_lookup_helm_cell_np, ..., vmap_method="expand_dims")`), `helm.py:41-43` (`_rust.eos_helm_cells_batch(lt_flat, d_flat)`), `src/flexmesa/eos/_helm_jax.py:28-60` (JAX evaluation of radiation + ions + electron-positron terms from the looked-up cell coefficients), `src/flexmesa/kap/core.py:139-171` (JAX branch calls `jax_kernels.kap_get_results_jax` with all options), `src/flexmesa/kap/tables_jax.py:162` (`_ = _rust.kap_get_fixed_metal_rust(0.02, 0.70, -6.0, 6.0)` to force the Rust loader).
- Confidence: verified-by-reading-flexmesa-only
- What: Matches the architecture claim better than star/net: the thermodynamic sums and the opacity interpolation/blending are written out in JAX, and differentiating inside a table cell is exact because the cell polynomial is evaluated in JAX (the stop-gradient on the cell index is correct). Caveats: (1) every JAX HELM evaluation calls back into Rust for the cell stencil, so it is CPU-bound and not GPU-portable, contrary to the "on-GPU `vmap`" motivation in CLAUDE.md Tier C/B; (2) the `.primal` unwrapping relies on JAX tracer internals; (3) both backends read the same Rust-loaded tables, so a table/loader error (N1-11..13) is invisible to cross-backend tests. The lead about a surviving no-data analytic HELM fallback was not confirmed: `grep -rni "smooth_builtin|fallback|no_data"` over `src/flexmesa/eos`, `crates/flexmesa-eos/src/helm` and `crates/flexmesa-py/src/eos*.rs` finds only the blend-region "fallback fraction" functions, and `load_or_build_full_helm_vertices` returns an error when no table is found (`crates/flexmesa-eos/src/helm/table_io.rs:81-86`).
- Proposed fix: state the CPU-callback limitation in docs; optionally ship the table as a device array for a pure-JAX path (M).

### N1-26: `backend="auto"` decides on a subset of the arguments
- Area: `src/flexmesa/_dispatch.py` and callers
- Severity: cosmetic
- FlexMESA: `src/flexmesa/_dispatch.py:28-44` (`is_jax_input(*values)` checks only top-level `isinstance(v, (jax.Array, jax.core.Tracer))`), `src/flexmesa/kap/core.py:139` (`resolved = _dispatch.resolve_backend(backend, z, x, zbase, log_rho, log_t)`; the other twelve numeric arguments, e.g. `xc, xn, xo, lnfree_e, eta, zbar`, are not inspected)
- Confidence: verified-by-reading-flexmesa-only
- What: If only an uninspected argument is traced (for example differentiating opacity with respect to `eta` or `xc` with float `log_t`), `auto` picks Rust and the call fails at `float(tracer)` with a JAX concretisation error. It fails loudly rather than returning a wrong value, so this is a usability defect, not a numerical one. JAX arrays nested in tuples/dataclasses (`turb.MltInputs`) are handled case by case by passing fields explicitly (`turb/mlt.py:85`).
- Proposed fix: pass all numeric arguments to `resolve_backend` (S).

### N1-27: Pure-Python duplicates of Tier A/C infrastructure contradict the architecture contract
- Area: `src/flexmesa/num`, `src/flexmesa/auto_diff`, `src/flexmesa/turb`
- Severity: maintainability
- FlexMESA: `src/flexmesa/num/*.py` (3,614 lines, zero `_rust.` calls; e.g. `num/newuoa.py:225-236` repeats the regularised KKT refit of N1-1 with `np.linalg.solve` and `except np.linalg.LinAlgError: return np.zeros(n), h_prev.copy()`), `src/flexmesa/auto_diff/*.py` (1,234 lines, zero `_rust.` calls), `src/flexmesa/turb/*.py` (2,368 lines, 2 `_rust.` calls; `turb/mlt.py:64-96` uses Rust only when inputs are plain floats). CLAUDE.md Tier A: "Never duplicate ... scalar Python dual-number operator-overloaded loops in pure Python".
- Confidence: verified-by-reading-flexmesa-only
- What: Several packages are a third implementation (Rust, JAX, and NumPy/pure Python) of the same algorithm. Each must be kept in step by hand, and Python-side tests of `num` exercise the Python copy, not the Rust code the star solver uses. The simplified NEWUOA exists twice.
- Proposed fix: bind `flexmesa.num`/`auto_diff` to `_rust` and delete the Python copies, or drop the contract sentence (M).

### Matches (Scope B)

- `_rust.pyi` vs PyO3 exports: script over `src/flexmesa/_rust.pyi` and `crates/flexmesa-py/src/*.rs` finds 328 `def` stubs and 328 `wrap_pyfunction!` registrations with identical name sets; the two classes map `RustStarModel`/`RustBinarySystem` to `PyStarModel`/`PyBinarySystem`. An argument-name/arity comparison over all 328 found no real mismatch (9 hits were artefacts of my parser on the argument named `m` and on default values). No drift detected.
- IFT reverse rule for the block-tridiagonal FFI solve is correct: `src/flexmesa/mtx/pcr.py:238-262` builds the transpose system (`l_t[k] = U[k-1]^T`, `u_t[k] = L[k+1]^T`), solves `A^T lam = xbar`, returns `Lbar_k = -lam_k x_{k-1}^T`, `Dbar_k = -lam_k x_k^T`, `Ubar_k = -lam_k x_{k+1}^T`, `bbar = lam`.
- `solve_block_tridiag_pcr` (`pcr.py:120-150`) is a real log-depth parallel cyclic reduction in JAX, differentiable by construction.
- No hard-coded answers found in the sampled packages (`eos/helm.py`, `kap/core.py`, `net/_jax.py`, `star/jax_kernels.py`, `mtx/pcr.py`); `except` clauses in `src/flexmesa` are few and specific (`grep` lists 12 outside tests).
- Missing data raises rather than silently degrading on the EOS path (`eos/_cache_io.py:88,141` `FileNotFoundError`; Rust helm loader returns `Err`). The kap path panics instead (N1-14).

## Scope A (remaining crates)

### Matches

- `flexmesa-interp1d` piecewise-monotonic: interior safe slope `(sign(s[i-1]) + sign(s[i])) * min(|s[i-1]|, |s[i]|, 0.5|p_i|)`, end-point parabola slopes and the `p*s <= 0 -> 0`, `|p| > 2|s| -> 2s` limiter (`crates/flexmesa-interp1d/src/pm.rs:96-109`) match `interp_1d/private/interp_1d_pm.f90:101-135`; the n = 2 case (`pm.rs:83-87`) matches `mk_pmlinear` (`interp_1d_pm.f90:352-358`). Upstream's separate `mk_pmquad` (n = 3) uses the same formulae (`:274-306`), so folding it into the general path is equivalent. verified-by-reading-both-sides.
- `flexmesa-interp1d` exposes the three monotonicity-preserving variants `interp_m3a/m3b/m3q` (`mp.rs:182,195,208`) corresponding to `average/super_bee/quartic` in `interp_1d/public/interp_1d_def.f90:30-32` (names only checked, bodies not compared).
- `flexmesa-interp2d` contains `bicub`, `bipm`, `akima`, `renka790` modules mirroring `interp_2d/private/{bicub,bipm,akima760,renka790}_db.f*` (presence only; not compared line by line).

### N1-28: Inlist handling silently ignores unknown controls and maps unknown option strings to a default
- Area: flexmesa-star inlist (touches flexmesa-schema use)
- Severity: wrong physics (silent misconfiguration)
- FlexMESA: `crates/flexmesa-star/src/inlist_populate.rs:389` (`_ => MltOption::Cox`), `:396` (`_ => AtmOption::TTau`), `:408` (`_ => AtmTTauRelation::Eddington`), `:416` (`_ => ScreeningMode::Extended`), `:422` (`_ => EnergyEqnOption::Dedt`); `inlist_populate.rs:40-58` (`f64_into/bool_into/str_into` copy a value only if the key is present; nothing enumerates keys that were present but not consumed); `grep -ni "unknown|unrecogni|unsupported|ignored" inlist.rs inlist_populate*.rs inlist_runner.rs star_job_init.rs` finds no diagnostic.
- Upstream: `star/defaults/controls.defaults` defines 1375 distinct control names (regex count); a Fortran namelist read fails on any name not declared. Of those 1375 names, 508 appear anywhere in FlexMESA's `controls*.rs`/`inlist_populate*.rs`/`job_controls.rs`/`inlist.rs` (upper bound on what can be honoured).
- Confidence: verified-by-reading-flexmesa-only for the code; the 508/1375 figure is a name-intersection script, inferred
- What: A test-suite inlist that sets one of the ~870 controls FlexMESA does not know (or misspells a known one) runs without any message, with that physics choice dropped. An option value FlexMESA has not implemented (e.g. an `atm_option` other than `table`/`simple_photosphere`, an unknown `MLT_option`, a different `energy_eqn_option`) is replaced by a default. For a project whose headline claim is "N/N upstream cases green", this is the mechanism by which a case can pass while not running the configuration its inlist specifies. A name-intersection also lists 36 control names that are stored but whose identifier appears nowhere else in `flexmesa-star/src` (e.g. `overshoot_f`, `overshoot_f0`, `predictive_*`, `gamma_lns_eps_grav_full_on/off`, `solver_max_tries_before_reject`, `use_other_kap`, `max_tries1`, `redo_limit`); some may be consumed under other names, so treat as leads.
- Proposed fix: after population, diff the set of keys present in each namelist section against the set consumed; error (or at least emit a per-case report) on unconsumed keys and on unrecognised option strings; use `flexmesa-schema`'s parsed `controls.defaults` as the authority for "known upstream control" (M).

## Scope C (remaining parts)

### N1-29: Duplicated code paths (structural)
- Area: workspace
- Severity: maintainability
- FlexMESA and rough size:
  - Three Rosenbrock/ODE stacks: `crates/flexmesa-num/src/rosenbrock.rs` + `rosenbrock_coeffs.rs` (759 lines), `crates/flexmesa-net/src/burn_const_p.rs` + `rosenbrock_tableau.rs` (own stage loop), `src/flexmesa/num/rosenbrock.py` (576 lines). See N1-5, N1-27.
  - Three block-Thomas implementations: `crates/flexmesa-mtx/src/block_tridiag.rs:231-310`, `crates/flexmesa-bench/src/tridiag_fast.rs:67-130` (copy with `unwrap()` and no input scans, used for the "zero-allocation" benchmark stage in `kernels_solver.rs:87-128`), `src/flexmesa/mtx/block_tridiag.py` (396 lines). The benchmark's fastest "Safe Rust" stage therefore times code the star solver does not call; `crates/flexmesa-bench/src/bicub_fast.rs:85` (`eval_bicub_prevalidated`) is the same pattern for bicubic evaluation.
  - `StarError` strings are rebuilt at ~180 call sites (`grep -o "StarError::[A-Za-z]*"` counts: IoError 56, ConvergenceFailure 48, MicrophysicsError 44, InvalidInput 29) with ad hoc wording; a context-carrying constructor would remove most of them.
  - Inlist population is split across `inlist_populate.rs` (456), `inlist_populate_limits.rs` (172), `inlist_populate_phase5b.rs` (263), and controls across `controls.rs` (548), `controls_defaults.rs` (272), `controls_mix_wind.rs` (461), `controls_timestep_stop.rs` (468): the split follows development phases ("phase5b") and the line budget, not a concept. One table-driven mapping generated from `controls.defaults` via `flexmesa-schema` would replace roughly 2,600 lines and give N1-28's unknown-key check for free.
  - Snapshot/restore exists three times with different field coverage: `restore_from_old` (`evolve.rs:76-113`, ~15 named arrays), `photo_io.rs` (hand-listed arrays), `BinaryStepSnapshot` (whole-struct clone). A single `StepState` (N1-17, N1-20) would serve all three and remove the risk that they disagree about what "the state" is.
- Confidence: verified-by-reading-flexmesa-only
- Proposed fix: as listed per item (M each).

### N1-30: File-size rule side effects and test-case logic in library crates
- Area: workspace layout
- Severity: maintainability
- FlexMESA: over all `crates/*/src/**/*.rs` (372 files): 1 file > 600 lines, 42 files between 560 and 600 lines, 97 between 401 and 559 (so 140 of 372 exceed the 400-line target and 42 sit within 40 lines of the hard ceiling). `rustfmt.toml:1` sets `max_width = 80`, yet `flexmesa-star/src` has 367 lines longer than 100 columns and `flexmesa-binary/src` 73 (e.g. `crates/flexmesa-binary/src/system_step.rs:84,150,157,189`, `crates/flexmesa-star/src/evolve.rs:252`), and there are 54 `#[rustfmt::skip]` attributes (e.g. `crates/flexmesa-net/src/derivs.rs:28,69,212`). Files split mid-concept: `evolve.rs`/`evolve_support.rs`, `solver.rs`/`solver_norms.rs`/`solver_support`-style helpers, `inlist_populate*.rs`, `controls*.rs`, `derivs.rs`/`derivs_support.rs`/`derivs_paired.rs`, `hydro_state.rs` holding both `HydroAuxState` and remesh smoothing. Test-case logic in the library: `crates/flexmesa-star/src/tier1_extras.rs` (509), `tier2_extras.rs` (562), `tier3_extras.rs` (575), `tier4_extras.rs` (524), `tier5_extras.rs` (452), `tier3_physics.rs`, plus `tests_fidelity*.rs` under `src/`.
- Confidence: verified-by-reading-flexmesa-only
- What: The 600-line ceiling is being met by packing: long unformatted lines, `rustfmt::skip`, and 20-argument functions on a few lines, which makes the densest numerical code (the binary step driver, the evolve loop, net derivs) the least readable and means `cargo fmt --check` cannot be a gate for those regions. Concept boundaries are drawn where the budget ran out. Structural recommendation (details belong to the test-suite auditor): move every `tier*_extras.rs` and per-case hook into a separate `flexmesa-testsuite` crate that depends on `flexmesa-star` through the same public hook trait a user would use (`run_star_extras` equivalent), so the library contains no case names and the hooks are visibly outside the physics.
- Proposed fix: replace the LOC rule with a per-function complexity/length lint and enforce `cargo fmt`; extract `flexmesa-testsuite` (M/L).

## Leads (not verified)

- Reaction ordering: `crates/flexmesa-net/src/initialize.rs:281-283` sorts reactions by name; upstream orders by reaction id from the net file/rates catalogue. Summation order in dydt and Jacobian rows would then differ from MESA (mitigated by `F64x2` accumulation). Not compared against `net/private/net_initialize.f90`.
- `crates/flexmesa-star/src/micro_net.rs:181-186` passes `1.0` for `weak_rate_factor` and `false, false, false` for `use_3a_fl87`, `use_suzuki`, `do_ecapture` unconditionally; whether the corresponding controls (`weak_rate_factor`, `net_suzuki`-style options) are meant to reach the net was not traced.
- `restore_from_old` (`crates/flexmesa-star/src/evolve.rs:76-113`) restores a named subset of arrays. State mutated during a failed attempt that is not in the list (e.g. TDC/`etrb`, `conv_vel`, `d_mix`, `aux.burn_*`, anything changed by `apply_step_pre_physics`) would carry into the retry unless recomputed by `set_hydro_vars`. Not traced field by field; overlaps the star-core auditor.
- `restore_stars` keeps the live `controls` across a snapshot restore (`crates/flexmesa-binary/src/system_rlof.rs:166-170`): if a trial step's hooks change controls, the retry does not start from the same configuration.
- NaN-to-zero in EOS table build (N1-12): upstream `eos/private/eosdt_load_tables.f90` behaviour on bad spline values not checked.
- AESOPUS float32 (N1-13): depends on the dtype inside upstream `AESOPUS_AGSS09.h5`, which is an LFS stub here.
- JAX PCR (`src/flexmesa/mtx/pcr.py:132-148`) inverts every diagonal block at every level with no inter-block pivoting; PCR is known to lose accuracy relative to Thomas/odd-even CR on poorly diagonally dominant Henyey matrices. No conditioning test seen.
- `exp10` in `crates/flexmesa-math/src/lib.rs` uses `powi` for integer arguments up to |x| = 308; beyond 10^22 this may differ in the last bit from upstream's repeated multiplication (both are inexact there).
- 36 controls that appear to be parsed but never read (list in N1-28) need a per-name check.
- `flexmesa-autodiff`, `flexmesa-utils`, `flexmesa-schema`, `flexmesa-data` crate internals (safetensors/sha256/json parsers) were only skimmed for API shape; not compared with upstream `auto_diff`/`utils`.

## Not covered

- Line-by-line comparison of Rosenbrock coefficient tables, `brent.rs`, `root.rs`, `simplex.rs` bodies, `interp_mp`, `bicub`/`akima`/`renka790`, banded and sparse `mtx` routines, eigen solvers.
- `flexmesa-autodiff` operator correctness versus upstream `auto_diff` generated types.
- Python packages not sampled in depth: `rates`, `neu`, `atm`, `chem`, `colors`, `ionization`, `interp_*`, `binary`/`astero` kernel bodies.
- Unused-control analysis beyond the name-intersection script; `#[allow]` audit beyond counts.
- Nothing was executed, so no numerical magnitudes (e.g. float32 AESOPUS error, libm vs crlibm drift) were measured.
