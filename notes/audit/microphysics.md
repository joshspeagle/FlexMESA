# F2: Microphysics fidelity audit (EOS, kap, net/rates, turb, atm, neu, chem, ionization, colors)

> **Reference note (`#141`):** Line numbers, commit diffs, and pre-rebaseline measurements in this historical audit register refer to MESA `main` at commit `0b2e6bb` (2026-11-04) and `FlexMESA` commit `91eda0b`. Active repository inputs, module goldens, `case_table.md`, and `defaults_diff_26_4_1.md` target **MESA release `26.4.1`** (tag `26.4.1`, commit `c8c2026`, Zenodo `19722306`).

Tree audited: `the audited tree` @ `91eda0b`. Upstream: `the upstream checkout` @ `0b2e6bb`.
All FlexMESA paths relative to the worktree root, upstream paths relative to the upstream root.
Method: reading both sides; no cargo/pytest was run.

## 0. Star-side wiring (`crates/flexmesa-star/src/micro.rs`, `inlist_populate.rs`)

### F2-1: Only 5 `&eos` and 5 `&kap` namelist controls are honoured; every other control is silently ignored
- Area: flexmesa-star (inlist -> EosConfig / KapConfig)
- Severity: wrong physics
- FlexMESA: `crates/flexmesa-star/src/inlist_populate.rs:259-292` (`eos.bool_into(&mut self.eos_config.use_free_eos, "use_freeeos"); map_bools!(eos, self.eos_config, [use_opal_scvh, use_pc, use_skye, use_cms])`; kap: `zbase`, `kap_file_prefix`, `kap_co_prefix`, `kap_lowt_prefix`, `use_type2_opacities`)
- Upstream: `eos/defaults/eos.defaults` (about 90 controls), `kap/defaults/kap.defaults`; read by `eos/private/eos_ctrls_io.f90`, `kap/private/kap_ctrls_io.f90`
- Confidence: verified-by-reading-both-sides
- What: The inlist reader maps only the five `use_*` EOS switches and five kap controls. All region boundaries (`logRho_min_for_all_Skye`, `Z_all_HELM`, `logT_all_OPAL`, FreeEOS cuts, `mass_fraction_limit_for_PC`, `Skye_min_gamma_for_solid`, ...), `include_radiation`, `include_elec_pos`, `coulomb_*_cut_HELM`, `eosDT_use_linear_interp_for_X`, `kap_blend_logT_*`, `cubic_interpolation_in_X/Z`, `include_electron_conduction`, `use_blouin_conductive_opacities`, `use_Zbase_for_Type1`, `kap_Type2_full_on/off_*`, `kap_lowT_Zbase` controls, `user_num_kap_*`, `use_other_*` are present in the config structs with upstream defaults but an inlist that sets them is accepted without any warning and evolved with defaults. Any upstream test_suite inlist that changes one of these runs with different physics from upstream with no diagnostic.
- Proposed fix: map every field that `EosConfig`/`KapConfig` already carries; emit a hard error (or at least a logged warning list) for any `&eos`/`&kap` key that is parsed but not consumed. Size M.

### F2-2: `do_eos` recomputes Prad as `crad*T^4/3` unconditionally and derives `gam` with its own formula
- Area: flexmesa-star `micro.rs::do_eos`
- Severity: wrong numerics (minor)
- FlexMESA: `crates/flexmesa-star/src/micro.rs:258-282` (`let prad = CRAD * t_k.powi(4) / 3.0; ... let a_e = (3.0/(FOUR_PI*AVO*rho_k*zbar.max(1e-10)/abar)).cbrt(); let gam = z53bar*QE*QE/(a_e*KERG*t_k)`)
- Upstream: `star/private/micro.f90` `store_eos_for_cell` (Prad from `s% Prad(k) = crad*T^4/3`, `s% gam(k) = s% z53bar(k)*qe*qe/(a_e*kerg*T)` in `set_micro_vars`/`eval_eos` path)
- Confidence: verified-by-reading-flexmesa-only (upstream formula recalled, line not re-read)
- What: Matches the upstream shape; listed so it is not mistaken for a divergence. The `zbar.max(1e-10)` floor is non-upstream but harmless.
- Proposed fix: none.

### F2-3: `do_neu` ignores all neutrino controls and hard-codes `log10_Tlim = 7.5`, `NeuFlags::default_stellar()`
- Area: flexmesa-star `micro.rs::do_neu`
- Severity: wrong numerics (control ignored)
- FlexMESA: `crates/flexmesa-star/src/micro.rs:351-380` (`let flags = NeuFlags::default_stellar(); ... neu_get(..., 7.5, flags)`)
- Upstream: `star/private/micro.f90` `do_neu_for_cell` (`flags` all true, `log10_Tlim = 7.5d0`, `s% use_other_neu`, `s% non_nuc_neu_factor`, `s% min_T_for_neu`... )
- Confidence: verified-by-reading-flexmesa-only (see neu section for the upstream comparison)
- What: `min_T_for_neu` and `non_nuc_neu_factor` are honoured; per-channel output (`d_nonnucneu_*`, `nonnucneu_plas/brem/phot/pair/reco`) is discarded (only `total` kept), so history/profile columns for neutrino channels cannot be upstream-faithful from this path.
- Proposed fix: store the 5 channels. Size S.

## 1. EOS (`crates/flexmesa-eos`)

### What subset is implemented
- Components present in Rust: HELM (`helm/`), OPAL/SCVH and FreeEOS 4D tables (`tables_4d/`), PC (`pc/`), Skye (`skye/`), CMS (`cms/`), ideal (`ideal.rs`), eosPT (`eos_pt.rs`), eosDT_get_Rho/T style inversions (`inverse.rs`), gamma-law eosDE (`gamma_de.rs`). `other_eos_component` / `other_eos_results` (level 0) are not ported.
- Cascade order CMS -> Skye -> PC -> FreeEOS -> OPAL/SCVH -> HELM -> ideal: `crates/flexmesa-eos/src/blend_dt.rs:300-430` vs `eos/private/eosdt_eval.f90:382-747`. Same order, same `alfa = 1` when a component is disabled.

### Matches (EOS)
- `combine_for_eosdt` control flow (pure-1st, skip-to-2nd on error when `okay_to_convert_ierr_to_skip`, pure-2nd, blend): `blend_dt.rs:200-240` vs `eosdt_eval.f90:1464-1582`. One nuance: upstream tests `alfa == 0d0`, FlexMESA `alfa <= 0.0`; equivalent because every alfa producer clamps to [0,1].
- `Do_Blend` quintic smoothing, `/ln10` conversion, blend-derivative terms, and "d_dxa ignores blend derivatives": `blend.rs:170-235` vs `eos/private/eosdt_support.f90:175-239`. Line-for-line.
- `Get_FreeEOS_alfa`: `tables_4d/blend_alfa.rs:118-200` vs `eosdt_eval.f90:808-973`. All 7 ramps, the cut-combination tree, and the `1 - blend` unpacking match. Defaults (`blend_alfa.rs:41-67`) match `eos/defaults/eos.defaults` (`logQ_max_FreeEOS_hi = 5.6` ... `logT_cut_FreeEOS_lo = 6.4`, `tiny_fuzz = 1d-6`).
- `get_opal_scvh_alfa_and_partials` (region determination + `set_alfa_and_partials` + Z factor): `blend_alfa.rs:212-400` vs `eosdt_eval.f90:1061-1460`. Every branch, constant (`logRho1_max = 3.71`, `logQ1 = 5.69`, `logQ2 = 5.68`, `logT5 = 4.899`, `logT6 = 4.890`, `logRho7/8`) and partial matches. Defaults match `eos.defaults` (`logT_all_OPAL = 7.5`, `logT_all_HELM = 7.6`, `Z_all_OPAL = 0.035`, `Z_all_HELM = 0.040`).
- `Get1_eosdt_Results` Z interpolation (linear in Z, `dZ = dcdZ*res`) and `get1_for_eosdt` mapping of `d_dX`/`0`/`d_dZ` onto species by charge 1 / 2 / other: `tables_4d/mod.rs:290-420` vs `eosdt_eval.f90:1584-1748`.
- OPAL/SCVH X grid `{0,.2,.4,.6,.8,(1.0 for Z=0)}` x Z `{0,.02,.04}` and FreeEOS 15-Z grid incl. `_CO` suffix for iz >= 13: `tables_4d/table_set.rs:43-120` vs `eos/private/eos_initialize.f90` / `eos.defaults` `suffix_for_FreeEOS_Z(13:15) = '_CO'`.
- HELM polygon blend geometry (`helm_blend_width = 0.1`, rectangle in (logRho, logT), winding-number containment, `eps = 1e-10` single-precision literal reproduced via `1.0e-10_f32 as f64`): `blend_dt.rs:140-170`, `blend.rs:18` vs `eosdt_eval.f90:749-805`, `eos/private/eos_blend.f90:136`.
- Skye/PC/OPAL/FreeEOS scalar defaults (`skye/mod.rs:57-70`, `pc/mod.rs:55-69`) match `eos.defaults`.
- HELM table is now required in full: `helm/table_io.rs:46-88` errors (and `HelmTable::builtin` panics, `helm/table.rs:283-287`) when neither `helm_table_full.bin` nor upstream `helm_table.dat` is present. See F2-4 for the history.

### F2-4: The "analytical HELM cell fallback" (d5a1172) and `helm_patches.bin` (47991c0, 36e8dc1, 18bc0b1) no longer exist at the audited head; while they existed they silently replaced the Timmes table with ideal-gas vertices
- Area: flexmesa-eos `helm/table.rs`, flexmesa-star `micro.rs`
- Severity: unverified claim (historical), now resolved in Rust
- FlexMESA: current `crates/flexmesa-eos/src/helm/table.rs:303-316` (`let (v00, v10, v01, v11) = (dense[row0], dense[row0 + 1], dense[row1], dense[row1 + 1]);` no fallback); `git show d5a1172 -- crates/flexmesa-eos/src/helm/table.rs` (`HelmVertex::analytic(d, t)`: `f0 = rt*(eta_id - 1)`, `dpdf: [rt, 0, R, 0]`, `ef: [eta_id, 1/d, -1.5/t, 0]`, `xf: [NA*d, NA, 0, 0]`, used "whenever a cell is not fully populated in the embedded patch table"); `git show 18bc0b1` (`HelmTable::smooth_builtin()` returned `new_empty_grid` = all-analytic table when the cache was absent, and `do_eos` force-set `use_free_eos = use_opal_scvh = false` when `EosDtTableSet::builtin().data_dir.is_none()`)
- Upstream: `eos/private/helm_alloc.f90` (reads the full table or fails)
- Confidence: verified-by-reading-flexmesa-only (git diffs + head)
- What: Between d5a1172 and bd33084 (#133) the crate shipped a sparse embedded subset of HELM vertices (`data/helm_patches.bin`, 4.8 to 8.3 MB) covering only the cells that golden tests touched. Any other cell was built from a closed-form non-degenerate Maxwell-Boltzmann electron free energy (no degeneracy, no relativity, no positrons, `fdtt = fddtt = 0`). The commit message claim "matching Timmes & Swesty (2000) helm_table.dat to 16 significant digits in the non-degenerate regime" is not something the code can deliver in degenerate cells, and nothing gated the fallback on regime: it was selected by "is this cell in the patch file", not by physics. With 18bc0b1 the star path used an entirely analytic HELM and disabled FreeEOS and OPAL/SCVH when data were absent, so a clean-CI stellar run was Skye + ideal-electron "HELM" + ideal, returning plausible numbers with no error. Results from that window (any CI-green claims made on runners without the data cache, including Wave 4 "stage parity without cache") do not test the upstream EOS. At the audited head all of this is removed: `data/helm_patches.bin` is gone, `HelmVertex::analytic`/`smooth_builtin` are gone (`grep -rn "helm_patches\|smooth_builtin" crates` is empty), and `micro.rs::do_eos` (`:226-236`) calls `HelmTable::builtin()` with no config mutation. With data present the patch file was never consulted on the dense path (`is_full_table` branch), so present-data results were unaffected.
- Proposed fix: none in Rust. Check the Python/JAX side (`src/flexmesa/eos/helm.py`, `_blend_dt_jax.py`), which 47991c0 routed through `_rust.eos_helm_cells_batch` "when helm_table.dat is not installed", for a surviving equivalent (not in my scope). Re-date any parity claim in notes that was produced in the d5a1172..bd33084 window. Size S.

### F2-5: Missing FreeEOS / OPAL-SCVH table files degrade silently to the next EOS level
- Area: flexmesa-eos `tables_4d`, `blend_dt.rs`
- Severity: wrong physics (when data absent)
- FlexMESA: `crates/flexmesa-eos/src/tables_4d/table_set.rs:171-185` (`data_dir: resolve_mesa_data_dir()` may be `None`), `:205-212` (`load_or_build_dt_slice(self.data_dir.as_deref(), kind, &stem)?`), `crates/flexmesa-eos/src/blend_dt.rs:214-219` (`Err(e) if cfg.okay_to_convert_ierr_to_skip => { let _ = e; eval_2nd() }`)
- Upstream: `eos/private/eosdt_load_tables.f90` (`load_single_eosDT_table_by_id` -> missing file is a fatal `mesa_error`), `eosdt_eval.f90:1505-1520`
- Confidence: verified-by-reading-both-sides for the Rust control flow; inferred that `load_or_build_dt_slice` returns `Err` rather than panicking for a missing file (I read the call site, not every branch of `slice_cache.rs`)
- What: Upstream treats a missing table file as fatal at load. In FlexMESA a table-load failure is an ordinary `EosError`, and with the default `okay_to_convert_ierr_to_skip = true` the cascade swallows it (`let _ = e`) and evaluates the next level (OPAL/SCVH, then HELM, then ideal). A solar-type envelope would then be computed with HELM (fully ionized, no molecular/partial-ionization physics) with no message. HELM itself is now mandatory (F2-4) but the 4D tables are not.
- Proposed fix: distinguish "table file missing / unreadable" (fatal, new `EosError::DataMissing`, never converted to skip) from "state outside component domain" (skip). Size S.

### F2-6: `helmeos2` clips (T, rho*Ye) to the table edge and returns a result; upstream returns `ierr`
- Area: flexmesa-eos `helm/eval.rs`
- Severity: wrong numerics (edge of domain)
- FlexMESA: `crates/flexmesa-eos/src/helm/eval.rs:80-104` (`if config.clip_to_table_boundaries { if log_temp <= HELM_LOGTLO { log_temp = HELM_LOGTLO; temp = templo; off_table = true; } ... din = denlo; den = din / ye; off_table = true; }`), default `clip_to_table_boundaries: true` (`:47`); the flag is discarded by the caller (`blend_dt.rs:258` `let (helm_res, _off_table) = helmeos2(...)`)
- Upstream: `eos/private/helm.f90:190-211` (`if (temp < ht% templo) then; ierr = 1; return; ... if (din > ht% denhi) then; ierr = 1; return`); `clip_to_table_boundaries` is a dummy argument that is never referenced in `helm.f90` or any `.dek`
- Confidence: verified-by-reading-both-sides
- What: At this upstream commit there is no clipping: outside [1e3, 1e13] K or rho*Ye outside [1e-12, 1e15] HELM fails and the cascade skips to ideal. FlexMESA instead moves the state onto the boundary, evaluates there, and then `do_convert_helm_results` is called with the original `rho`, `t` (`blend_dt.rs:266`), so lnPgas/lnE/lnS are those of a different thermodynamic state while the conversion uses the unclipped T and rho. The HELM polygon blend is in logRho, not log(rho*Ye), so for Ye < 1 there is a strip (logRho in about [-12, -11.7]) where the polygon says "HELM" but rho*Ye is below the table: upstream gives pure ideal there, FlexMESA gives clipped HELM. Rare in stellar interiors, reachable in extended atmospheres / low-density test cases.
- Proposed fix: default `clip_to_table_boundaries` to "error like upstream" (return `Err(1)`), or delete the clip branch. Size S.

### F2-7: HELM-edge blend partials are un-swapped relative to an upstream argument-order bug
- Area: flexmesa-eos `blend_dt.rs::get_helm_alfa`
- Severity: cosmetic (FlexMESA is the mathematically right one; it is a deviation from upstream numbers)
- FlexMESA: `crates/flexmesa-eos/src/blend_dt.rs:168` (`(blend.val, blend.d1[1], blend.d1[0])` consumed at `:306` as `(a, dt, dr)`)
- Upstream: `eos/private/eosdt_eval.f90:728` (`call get_HELM_alfa(rq, logRho, logT, alfa, d_alfa_dlogRho, d_alfa_dlogT, ierr)`) against the callee signature at `:749-750` (`rq, logRho, logT, alfa, d_alfa_dlogT, d_alfa_dlogRho, ierr`)
- Confidence: verified-by-reading-both-sides
- What: Upstream passes the two derivative actuals in the opposite order to the dummies, so inside the 0.1 dex HELM/ideal blend band upstream's `d_alfa_dlogT` holds d/dlogRho and vice versa. FlexMESA assigns them correctly. Consequence: `d_dlnd`/`d_dlnT` of blended results differ from upstream only inside that band (logT within 0.1 of 3 or 13, logRho within 0.1 of -12 or 15). No docs entry for this discrepancy was found in the code comments.
- Proposed fix: document in the discrepancy ledger. Size S.

### More EOS matches (read after the first block)
- `Get_Skye_alfa` 8-vertex polygon, `skye_blend_width = 0.1`, `max(7.3, log10(1e5 zbar^2))`, `max(2.3, log10(abar zbar^3))`, radiation offset: `skye/mod.rs:79-118` vs `eos/private/skye.f90:35-116`. Exact.
- Skye species selection above `mass_fraction_limit`, renormalisation, number fractions, ion offsets, `xnefer = compute_xne`, and "composition derivatives not provided" (`d_dxa = 0`): `skye/mod.rs:185-300` vs `skye.f90:236-398`.
- HELM `d_dxa` chain via `dabar_dx = abar*(A_i - abar)/A_i/sumx`, `dzbar_dx = abar*(Z_i - zbar)/A_i/sumx`: `blend_dt.rs:267-279` vs `eos/private/eos_helm_eval.f90:62-82`, `chem/public/chem_lib.f90:238-239`.
- HELM Coulomb section (Yakovlev & Shalybkov, `plasg >= 1` branch, all second derivatives I compared, `coulomb_temp_cut/den_cut` factor): `helm/coulomb.rs:26-200` vs `eos/private/helm_coulomb2.dek`. The port drops terms multiplied by `dxnidt`, `dxnidz` (identically zero for `xni = avo*den/abar`); algebraically identical.
- `helmeos2` failure when `pgas, egas or sgas < 1e-20`: `helm/eval.rs:140-142` vs `helm.f90:273-277`.
- `do_safe_get_Rho_T` residuals for `egas`, `log10 Ptot`, and generic `res(which_other)` including the upstream sign convention `dfdx = energy*dlnE_dlnRho*ln10 - derad_dlnRho`: `inverse.rs:33-70` vs `eosdt_eval.f90:2480-2532`. `newt_imax = 6`, `x > 50 -> error`, bad-Cv check also match. (FlexMESA renumbers `which_other = 0` (logPtot) to `-2` because indices are 0-based.)

### EOS: not examined line by line
PC (`pc/*.rs` vs `eos/private/pc_eos.f90`), Skye Coulomb fits (`skye/coulomb*.rs`), CMS, `eos_pt.rs`, `gamma_de.rs`, the HELM biquintic polynomials and electron-positron section, and the 4D table spline builder (`tables_4d/spline.rs`, `slice_cache.rs`) were not compared term by term. CMS defaults in `cms/config.rs:68-93` are concrete numbers (e.g. `logrho_min_for_any_cms: -8.0`) whereas `eos.defaults` sets every CMS limit to `-1`; harmless while `use_cms = false` but unverified against `eos_ctrls_io.f90`.

## 2. Opacity (`crates/flexmesa-kap`)

### What subset is implemented
- High-T Type 1: `gn93`, `gs98`, `a09`, `OP_gs98`, `OP_a09_nans_removed_by_hand` (`types.rs:23-52`). Not available: `oplib_*` (4 families present in upstream data), `gs98_aFe_*`.
- Type 2 (C/O enhanced): `gn93_co`, `gs98_co`, `a09_co` (`types.rs:57-103`). Not: `gs98_aFe_p4_co`.
- Low-T: `lowT_fa05_gs98`, `lowT_fa05_gn93`, `lowT_fa05_a09p`, `lowT_Freedman11`, `kapCN`, `AESOPUS` (`types.rs:108-140`). Not: `lowT_af94_gn93`, `lowT_fa05_ags04`, `lowT_fa05_mb22`, `lowT_fa05_aag21`, `lowT_fa05_gs98_aFe_*`.
- Compton (Poutanen 2017), electron conduction (Potekhin table + Blouin 2020 H/He correction), `kap_get_op_mono` family: not found in the crate (no OP monochromatic path).
- `use_other_*_opacity` hooks: not ported.

### Matches (kap)
- Compton blend in logT and logR with quintic smoothing, `blend = 1 - (1-bT)(1-bR)`, frac bookkeeping (`frac_Type2` not rescaled): `eval.rs:63-140` vs `kap/private/kap_eval.f90:105-222`.
- low-T/high-T blend, Townsend quintic, `d_alfa_dlnT*logKap*ln10` term (FlexMESA works in d/dlogT so the ln10 cancels), rho clip at `kap_min_logRho = -40`: `blend.rs:34-110` vs `kap_eval.f90:229-380`.
- Type 2 blend fractions (`frac_X`, `frac_dZ`), `dXC = max(0, XC - fC*Zbase)`, `min(Z, Zbase)` for Type 1 when `use_Zbase_for_Type1`, linear-in-kap combination: `blend.rs:125-215` vs `kap_eval.f90:477-697`.
- Fixed-metal X and Z interpolation (linear default, 4-point cubic when requested, end clamping): `fixed.rs:14-225` vs `kap/private/kap_eval_fixed.f90:34-630`.
- `combine_rad_with_conduction` harmonic sum and partials: `eval.rs:156-177` vs `kap_eval.f90:700-790`.
- Potekhin conduction table lookup, Z interpolation in log Z, `logkap = 3 logT - logRho - logK + log10(16 sigma/3)`: `conduction.rs:161-232` vs `kap/private/condint.f90` (`do_electron_conduction_potekhin`). Embedded `crates/flexmesa-kap/data/condtabl.data` is byte-identical to upstream `data/kap_data/condtabl.data` (`cmp`).
- Blouin et al. 2020 correction: all 14 constants, the H/He mixing in log Zbar, the rotated Gaussian, tanh switch: `conduction.rs:235-330` vs `condint.f90` (`do_electron_conduction_blouin`). Exact.
- Table storage as `i16` milli-dex (`loader_ascii.rs:149` `(v * 1000.0).round() as i16`) is lossless for the Type 1 / Type 2 / lowT ASCII tables, which carry 3 decimals (checked `data/kap_data/gs98_z0.02_x0.7.data`, `OP_gs98_...`). The `oplib_*` tables carry 4 decimals but are not loaded.
- Default table extents assumed by hard-coded constants agree with the default files: gs98/gn93/a09/OP_gs98 `logT_min = 3.75`, `logT_max = 8.70`, `logR_min = -8`; `lowT_fa05_gs98`/`a09p`/`Freedman11` `logT_max = 4.50` (file headers, line 3).

### F2-8: `KapConfig::default()` turns Type 2 opacities and `use_Zbase_for_Type1` OFF; upstream default is ON for both
- Area: flexmesa-kap `types.rs`, consumed by flexmesa-star
- Severity: wrong physics
- FlexMESA: `crates/flexmesa-kap/src/types.rs:209-210` (`use_zbase_for_type1: false, use_type2_opacities: false,`); star builds from this default (`crates/flexmesa-star/src/model.rs:395-398` `KapConfig { zbase: controls.zbase, ..KapConfig::default() }`) and only flips it when the inlist literally contains `use_Type2_opacities` (`inlist_populate.rs:288-291`); `use_Zbase_for_Type1` is never read from an inlist
- Upstream: `kap/defaults/kap.defaults` (`use_Zbase_for_Type1 = .true.`, `use_Type2_opacities = .true.`)
- Confidence: verified-by-reading-both-sides
- What: Upstream uses C/O-enhanced Type 2 tables wherever X < 1e-3 and Z - Zbase > 0.001 (He-burning cores, He/C/O shells, WD interiors, stripped stars), and evaluates Type 1 at min(Z, Zbase). FlexMESA by default evaluates fixed-metal Type 1 tables at the local Z (clamped to the largest tabulated Z, 0.1) everywhere. Every post-main-sequence model run from an inlist that relies on the upstream default gets different radiative opacity in exactly the regions where metals are being produced, and `frac_Type2` is 0 throughout. The same defaults are baked into the Python bindings (`crates/flexmesa-py/src/kap_py.rs:85`).
- Proposed fix: set both defaults to `true`, read `use_Zbase_for_Type1` from the inlist, and reproduce upstream's requirement that `Zbase` be supplied (upstream `kap_eval.f90:507-512` errors when `Zbase < 0` with Type 2 on). Size S, but expect golden values and test_suite baselines to move.

### F2-9: Reference C and O metal fractions for `gs98_co` and `a09_co` are wrong constants (upstream reads them from the table header)
- Area: flexmesa-kap `types.rs::KapCoOption::ref_zfracs`, used in `blend.rs`
- Severity: wrong numerics (genuine bug)
- FlexMESA: `crates/flexmesa-kap/src/types.rs:89-95` (`Self::Gs98Co => (0.165_659, 0.048_844, 0.453_345, 0.114_576), Self::A09Co => (0.176_246, 0.050_427, 0.428_279, 0.093_860)`), used at `blend.rs:157-159` (`let dxc = (xc - fc * zbase).max(0.0); let dxo = (xo - fo * zbase).max(0.0);`)
- Upstream: `kap/private/kap_eval.f90:536-542` (`fC = kap_co_z_tables(rq% kap_CO_option)% ar(1)% Zfrac_C ... dXC = max(0d0, xC - fC*Zbase)`); file headers `data/kap_data/gs98_co_z*_x*.data` line 3: `0.171835 0.050335 0.467356 0.104831`; `a09_co_*`: `0.176930 0.051828 0.428871 0.094039`; `gn93_co_*`: `0.173285 0.053152 0.482273 0.098668`
- Confidence: verified-by-reading-both-sides (constants compared against the actual upstream data headers for all 8 Z values of each family)
- What: Only the gn93 triple matches. For the default `gs98_co`, FlexMESA uses fC = 0.165659 and fO = 0.453345 instead of 0.171835 and 0.467356, so the excess carbon and oxygen `dXC`, `dXO` fed to the Type 2 interpolation are too large by 0.0062*Zbase and 0.0140*Zbase (1.2e-4 and 2.8e-4 at Zbase = 0.02). The loader already parses the header values into `KapCoZTable::zfrac_c/o` (`loader_parse.rs:139-142`) but `blend.rs` ignores them. Small shift in Type 2 opacity near the base composition; masked today because Type 2 is off by default (F2-8).
- Proposed fix: use `z_tables[0].zfrac_c/zfrac_o` from the loaded set; delete the constants. Size S.

### F2-10: Unknown `kap_file_prefix` / `kap_CO_prefix` / `kap_lowT_prefix` values are silently replaced by gs98 defaults
- Area: flexmesa-star `inlist_populate.rs`
- Severity: wrong physics
- FlexMESA: `crates/flexmesa-star/src/inlist_populate.rs:273-287` (`if let Some(opt) = flexmesa_kap::KapOption::from_prefix(s) { self.kap_config.kap_option = opt; }` with no `else`)
- Upstream: `kap/private/kap_ctrls_io.f90` / `kap_def.f90` option lookup, which errors on an unknown prefix
- Confidence: verified-by-reading-flexmesa-only (upstream error path not re-read)
- What: An inlist asking for `oplib_gs98`, `gs98_aFe_p4`, `lowT_fa05_mb22`, `lowT_af94_gn93`, etc. (all shipped in upstream `data/kap_data`) runs with `gs98` / `lowT_fa05_gs98` and no message.
- Proposed fix: return an error for an unsupported prefix. Size S.

### F2-11: Blend boundaries that upstream derives from the loaded tables are hard-coded
- Area: flexmesa-kap `blend.rs`, `types.rs`
- Severity: wrong numerics (non-default tables)
- FlexMESA: `crates/flexmesa-kap/src/blend.rs:40-46` (`let high_t_min = 3.75_f64; let low_t_max = match cfg.kap_low_t_option { KapLowTOption::KapCn => 4.05, _ => 4.50 };`), `types.rs:215-216` (`log_t_compton_blend_hi: 8.69, log_r_compton_blend_lo: -7.99`); no code updates the latter two from the tables (`grep` finds only `photo_io.rs` restoring them)
- Upstream: `kap/private/kap_eval.f90:267-298` (`logT_min = ...x_tables(1)% logT_min`, `lowT_logT_max = kap_lowT_z_tables(...)% ar(1)% x_tables(1)% logT_max`, AESOPUS `kA% max_logT`), `kap/private/load_kap.f90:65-69` (`logT_Compton_blend_hi = min(Type1 logT_max - 0.01, CO logT_max - 0.01)`, `logR_Compton_blend_lo = logR_min + 0.01`)
- Confidence: verified-by-reading-both-sides; table extents read from upstream data headers
- What: Correct for the default gs98 family. Wrong for (i) `OP_a09_nans_removed_by_hand`, whose header gives `logT_max = 8.000`: upstream starts the Compton blend at logT 7.49 and is pure Compton above 7.99, FlexMESA keeps `8.69` and then `fixed.rs:231-235` returns `OutOfBounds("logT > logT_max")` for 8.0 < logT < 8.69 (hard failure where upstream works); (ii) `lowT_fa05_gn93`, `logT_max = 4.100` (only matters if `kap_blend_logT_upper_bdy` is raised, which cannot be done from an inlist anyway, F2-1); (iii) AESOPUS, where upstream uses the file's `max_logT`.
- Proposed fix: compute the four numbers from the loaded tables at config build time. Size S.

### F2-12: Fixed-metal tables clip logT below `logT_min` and logR below `logR_min`; upstream returns `ierr`
- Area: flexmesa-kap `fixed.rs`
- Severity: wrong numerics (off-table)
- FlexMESA: `crates/flexmesa-kap/src/fixed.rs:236-248` (`let (mut log_t, clipped_log_t) = if log_t_in < xt.log_t_min { (xt.log_t_min, true) } ...; if log_r_in < xt.log_r_min { (xt.log_r_min, true) }`)
- Upstream: `kap/private/kap_eval_fixed.f90` `Get_Kap_for_logRho_logT` (`if (logT < x_tables(ix)% logT_min) then; ierr = -1; return` and `if (logR_in < x_tables(ix)% logR_min) then; ierr = -1; return`; only the high-R edge is clipped)
- Confidence: verified-by-reading-both-sides
- What: Below the low-T table floor (logT < 2.7 for fa05, 1.88 for Freedman) upstream fails the call; FlexMESA returns the edge opacity with zero T-derivative. With default blends the low-R case is shielded by the Compton blend, so the reachable case is cold outer layers.
- Proposed fix: return `KapError::OutOfBounds` as upstream. Size S.

### Kap: not examined line by line
`co.rs`/`co_find.rs` (Type 2 dXC/dXO interpolation), `low_t.rs` (kapCN, AESOPUS), `compton.rs`, `spline.rs` (bicubic not-a-knot build vs upstream `interp_mkbicub_db`), `loader_ascii.rs`.

## 3. Nuclear network and rates (`crates/flexmesa-net`, `crates/flexmesa-rates`, `flexmesa-star/src/micro_net.rs`)

### What subset is implemented
- Network definitions: a pre-parsed bundle `crates/flexmesa-net/data/nets.bin` (41 kB; "83-network bundle" per `micro.rs:44`) plus `from_name_or_dir` for a `.net` file on disk (`initialize.rs:245-262`). General-network RHS/Jacobian with REACLIB-kind pairing (`derivs.rs`, `derivs_paired.rs`), hardwired `approx21` / `approx21_plus_co56` (`approx21*.rs`), one-zone burners at constant T/rho, constant density and constant pressure (`burn*.rs`).
- Rates: `crates/flexmesa-rates/data/rates_tables.bin` (3.4 MB) holds 316 reaction records (307 predefined + r1..r9), 4450 REACLIB handles, 463 weaklib tables, Suzuki tables, and exactly 3 `rate_tables` entries (`r_al26-1_to_al26-2`, `r_al26-2_to_al26-1`, `r3` with 2 points). I parsed the bundle header with a small script; upstream `jina_reaclib_results_20171020_default` has 328,692 lines (roughly 80k rate sets), so the bundle is a subset (about 4.4k handles) chosen to cover the bundled nets.
- Screening: `no_screening`, `salpeter`, `extended`, `chugunov` (`flexmesa-rates/src/screen*.rs`, `flexmesa-net/src/screen.rs`).
- Not ported / not reachable from star: `other_net_get`, `rate_factors` / `special_rate_factor`, `weak_rate_factor`, `reaction_neuQs_factor`, `net_logTcut_*`, user `rate_tables_dir` (e.g. `r_c12_ag_o16_kunz.txt`, the `*_nacreII.dat` and `c12ag_deboer_*` tables shipped upstream), `nonlocal_NiCo_*`, `max_abs_eps_nuc`.

### Matches (net/rates)
- `eval_net` driver order (clip logT to `rattab_tlo`, molar abundances with `min(1, max(x, 0))/A`, weaklib rates, raw rates from tables, approx21 branch, screening, `get_derivs`, `Qconv` scaling, `dxdt = A*dydt`, `store_partials` `d_dxdt_dx = d_dydt_dy * A_i/A_j`, `d_eps_nuc_dx = Qconv*d_eps_nuc_dy/A`): `flexmesa-net/src/derivs.rs:28-180` vs `net/private/net_eval.f90:37-330`, `:636-655`, `:812-830`.
- Raw rate table interpolation: `nrattab = 10001` (2000 points/decade over logT 5.301..10.301), `ye`/`rho` exponent factors, cubic evaluation and derivative `(f2 + 2 dt (f3 + 1.5 dt f4)) * dtab/(T ln10)`, truncation at `max_safe_logT_for_rates` with zero dT: `flexmesa-net/src/initialize.rs:502-560`, `types.rs:17-38` vs `rates/private/rates_support.f90` `do_get_raw_rates`, `rates/public/rates_def.f90:295-304`.
- Chugunov (2007) screening: every branch (tn <= 0.1, 0.1..0.2 cosine blend, Gamma cap 590..600, h0 cap 300), all fit constants and all T/rho partials: `flexmesa-rates/src/screen_chugunov.rs:20-168` vs `rates/private/screen_chugunov.f90` `eval_screen_chugunov`. Exact (I checked each derivative expression).
- Fushiki & Lamb pycnonuclear 3-alpha constants: `flexmesa-rates/src/raw_rates.rs:161-217` (not compared to `pycno.f90` line by line; constants look like the published fit).
- `do1_net` zone gating: skip when `eps_nuc_factor == 0 && dxdt_nuc_factor == 0` or `abar > max_abar_for_burning`; clip to `max_logT_for_net` and zero the T partials; op-split skip when `T_start >= op_split_burn_min_T`; `d_epsnuc_dlnd = d_eps_dRho*rho`: `flexmesa-star/src/micro_net.rs:118-160`, `:200-240` vs `star/private/net.f90:150-200`, `:300-380`. Defaults `max_logT_for_net = 10.2`, `max_abar_for_burning = 199` match `star/defaults/controls.defaults:6691,6759`.

### F2-13: Nuclear burning is switched off below `min_logt_for_net` (default logT = 6.3), a control that does not exist upstream; tier "extras" raise it to 8.3 / 8.5
- Area: flexmesa-star `micro_net.rs`, `controls_defaults.rs`, `tier*_extras.rs`
- Severity: wrong physics
- FlexMESA: `crates/flexmesa-star/src/controls_defaults.rs:43` (`min_logt_for_net: 6.3,`); `micro_net.rs:126-141` (`|| log_t < min_logt { *eps_k = 0.0; ... dxdt_k.fill(0.0); ... return Ok(()) }`); `tier2_extras.rs:470` (`model.controls.min_logt_for_net = 8.3;`), `tier3_extras.rs:239`, `tier4_extras.rs:207`, `tier5_extras.rs:334` (`= 8.5`)
- Upstream: `star/private/net.f90:171-175` (the only early return is `eps_nuc_factor == 0 .and. dxdt_nuc_factor == 0` or `abar > max_abar_for_burning`); `grep -rn "min_logT_for_net"` over all upstream `.f90`, `.defaults`, `.inc` returns nothing
- Confidence: verified-by-reading-both-sides
- What: Upstream evaluates the network in every zone and lets the rates fall off. FlexMESA zeroes `eps_nuc`, `dxdt_nuc` and all partials for T < 2e6 K by default. That removes deuterium burning (T about 1e6 K) entirely and truncates Li/Be/B burning, so pre-main-sequence contraction, brown dwarfs and low-mass stars follow a different track with no warning; it also puts a discontinuity in eps_nuc at logT = 6.3. Worse, case-specific code raises the threshold to logT 8.3 or 8.5, which disables all hydrogen burning and most helium burning in those upstream test_suite cases (notes/DECISIONS.md:147,151 describe this as a device so the cases "converge"). A case that passes with burning suppressed below 2e8 K is not the upstream case.
- Proposed fix: delete the control (or default it to -inf and never set it from extras); re-baseline every case that touched it. Size S in code, L in consequences.

### F2-14: Default screening is `extended`; upstream default is `chugunov`; any unrecognised string also maps to `extended`
- Area: flexmesa-star controls
- Severity: wrong physics (default)
- FlexMESA: `crates/flexmesa-star/src/controls_defaults.rs:49` (`screening_mode: ScreeningMode::Extended,`); `inlist_populate.rs:411-418` (`"chugunov" => ScreeningMode::Chugunov, _ => ScreeningMode::Extended`)
- Upstream: `star/defaults/controls.defaults:6657` (`screening_mode = 'chugunov'`; comment: "MESA versions <=11435 used extended as the default value")
- Confidence: verified-by-reading-both-sides
- What: Every inlist that does not set `screening_mode` (the large majority) runs with the pre-2019 screening prescription instead of Chugunov. Screening factors differ at the percent level in the Sun and by large factors in dense C/O burning.
- Proposed fix: default to `Chugunov`; error on unknown strings. Size S.

### F2-15: `rate_factors`, `weak_rate_factor`, `reaction_neuQs_factor`, `max_abs_eps_nuc`, `fe56ec_fake_factor` are not wired from star
- Area: flexmesa-star `micro_net.rs`, `struct_burn_mix.rs`; flexmesa-net `initialize.rs`
- Severity: wrong physics (controls ignored; one wrong default)
- FlexMESA: `crates/flexmesa-star/src/micro_net.rs:55` (`let rate_factors = vec![1.0_f64; net_info.num_reactions];`), `:185` (literal `1.0` weak rate factor); `struct_burn_mix.rs:124,158,195` (same); `crates/flexmesa-net/src/initialize.rs:492-493` (`fe56ec_fake_factor: 1.0e-4, min_t_for_fe56ec_fake_factor: 3.0e9`) with no setter called from star (`grep fe56ec crates/flexmesa-star/src` is empty)
- Upstream: `star/private/net.f90:270-290` (passes `s% rate_factors, s% weak_rate_factor`, `reaction_neuQs`), `:311-316` (`max_abs_eps_nuc` clamp), `:747-748` (`call net_set_fe56ec_fake_factor(s% net_handle, s% fe56ec_fake_factor, s% min_T_for_fe56ec_fake_factor, ierr)`); `star/defaults/controls.defaults:8140` (`fe56ec_fake_factor = 1d-7`); `net/public/net_def.f90:406` (library default `1d-4`)
- Confidence: verified-by-reading-both-sides
- What: (i) `num_special_rate_factors` / `special_rate_factor` / `reaction_for_special_factor` inlist controls (used by several upstream test_suite cases, e.g. C12(a,g) variations) cannot have any effect. (ii) For `approx21` runs the fake Fe56 electron-capture rate uses the net-library default 1e-4, whereas a star run upstream overrides it with the star default 1e-7: a factor 1000 in the neutronisation rate at T >= 3e9 K in every approx21 pre-supernova model. (iii) Upstream rescales `eps_nuc_categories` by `eps_nuc/sum(categories)` (`net.f90:318-330`); FlexMESA stores the raw categories (`micro_net.rs:208-210`), so category columns need not sum to eps_nuc.
- Proposed fix: thread the controls through; set the fake factor from star controls; add the category renormalisation. Size M.

### F2-16: A REACLIB handle missing from the bundle evaluates to rate 0.0 instead of an error
- Area: flexmesa-rates `raw_rates.rs`
- Severity: wrong physics (silent, when a net needs a rate outside the bundle)
- FlexMESA: `crates/flexmesa-rates/src/raw_rates.rs:88-89` (`0 => eval_reaclib_with_cat(cat, &info.target_handle, tf.t9, false).map_or(0.0, |(f, _, _, _)| f),`), same pattern for op codes 29, 30, 31 (`:98-128`), rate tables (`:90` `map_or(0.0, ...)`), and `_ => 0.0` for unknown op codes (`:130`)
- Upstream: `rates/private/rates_initialize.f90` / `reaclib_eval.f90` (a reaction with no rate source is a fatal initialisation error)
- Confidence: verified-by-reading-flexmesa-only (upstream failure path not re-read)
- What: The rate-table builder (`flexmesa-net/src/initialize.rs:583-590`) calls `eval_reaction_raw` for each of 10001 temperatures; any lookup error becomes a zero rate, so the reaction is silently absent from the network. With only about 4.4k of roughly 80k REACLIB rate sets bundled this is reachable for custom or large nets; `synthesize_reaction_info` does error when the handle is absent (`reaction_synth.rs:168-170`), which protects the main path, but predefined records whose `target_handle` is not in the bundle, op code 30/31 component handles, and missing rate tables are not protected.
- Proposed fix: make `eval_reaction_raw` return `Result` and fail at net initialisation. Size S.

### Net/rates: not examined line by line
`derivs_support.rs`, `derivs_paired.rs` (2-to-1 / 2-to-2 pairing), all `approx21_*.rs` (about 3000 lines) against `net_approx21.f90`, `screen.rs` extended (Graboske/Alastuey) path and `screen_combo.rs`, `ratelib.rs` analytic rates and `tfactors.rs`, `weak.rs`/`ecapture.rs`, the three one-zone burners, and `parser.rs`. Rate-source selection: upstream's per-reaction preference between `ratelib` fits (NACRE/CF88) and REACLIB is encoded in the bundle's `op_code` (107 records op 0 = REACLIB, 109 op 29, 32 op 2 = weak, the rest analytic); I did not verify that the op-code assignment reproduces `rates/private/raw_rates.f90` `set_raw_rate`.

## 4. Turbulence / mixing-length (`crates/flexmesa-turb`, call site `flexmesa-star/src/hydro_vars.rs::eval_face_mlt`)

### What subset is implemented
- `MLT_option`: Cox, Henyey, ML1, ML2, Mihalas, and TDC (`mlt.rs:10-34`, `tdc*.rs`). `'none'` is emulated by setting `mixing_length_alpha = 0` (`inlist_populate.rs:385-388`).
- Semiconvection: `Langer_85` and `Langer_85 mixing; gradT = gradr` (`semiconvection.rs`).
- Thermohaline: `Kippenhahn`, `Traxler_Garaud_Stellmach_11`, `Brown_Garaud_Stellmach_13` (`thermohaline.rs`).
- Not ported at the star call site: `use_superad_reduction`, `max_conv_vel_div_csound`, `include_mlt_Pturb_in_thermodynamic_gradients`, `remove_small_D_limit`, `TDC_num_*_cells_forced_nonturbulent`, `other_mlt_results`, starspots, `gradr_factor(k)` from T-tau, velocity time centering of P and L in MLT, `Eq_div_w` for TDC with `TDC_alpha_M > 0`, `dominant_iso_for_thermohaline`.

### Matches (turb)
- MLT conductivities, `a0`, `A`, all four Henyey-family `ff1..ff4` sets, the Cardano root (`f`, `f0`, `f1`, `f2`, `Gamma`), `conv_vel = alpha*sqrt(Q P/(8 rho))*Gamma/A`, `D = conv_vel*Lambda/3`, `Zeta = exp(3 ln Gamma - ln Bcubed)`, `Zeta` cap at 1, `gradT = (1-Zeta) gradr + Zeta gradL`: `flexmesa-turb/src/mlt.rs:95-290` vs `turb/private/mlt.f90:73-180`.
- Semiconvection: `D`, `conv_vel = 3D/Lambda`, and every Langer 85 coefficient (`bc`, `LG`, `a0..a3`, `b1..b7`, `div`): `semiconvection.rs:54-140` vs `turb/private/semiconvection.f90`. (`a4+a5+a6` upstream equals `b1*a0`, which FlexMESA reuses.)
- Thermohaline: Kippenhahn `D = -3 K/(2 rho cp) * gradL_comp/max(1e-40, grada-gradr)`, Traxler eq. 24 (`101 sqrt(K_mu nu) exp(-3.6 r) (1-r)^1.1`), Brown `K_mu (Nu_mu - 1)` with the 200-try relaxation and fall-back to the analytic estimate, `get_diff_coeffs` (both Z > 2 and Z <= 2 branches including the extra `-ln 2`): `thermohaline.rs:84-240` vs `turb/private/thermohaline.f90`.
- Scale height with Eggleton alternative `sqrt(P/G)/rho` (upstream default `alt_scale_height_flag = .true.`): `hydro_vars.rs:171-177` vs `star/private/star_utils.f90:4014-4030`.
- Ledoux: `gradL = grada + gradL_composition_term` only when `use_Ledoux_criterion`; `mlt_vc = 0` unless convective: `hydro_vars.rs:186-192,403-407` vs `star/private/turb_support.f90` (`Get_results`, `do1_mlt_eval`).
- Defaults `mixing_length_alpha = 2`, `use_Ledoux_criterion = .false.`, `alpha_semiconvection = 0`, `semiconvection_option`, `thermohaline_coeff = 0`, `thermohaline_option = 'Kippenhahn'`, Henyey `nu = 8`, `y = 1/3`: `controls_defaults.rs:25-31`, `hydro_vars.rs:299-300` vs `star/defaults/controls.defaults:2028-2134,2319-2320`.

### F2-17: Default `MLT_option` is Cox; upstream default is TDC; unknown strings map to Cox
- Area: flexmesa-star controls
- Severity: wrong physics (default)
- FlexMESA: `crates/flexmesa-star/src/controls_defaults.rs:26` (`mlt_option: MltOption::Cox,`); `inlist_populate.rs:378-390` (`_ => MltOption::Cox`)
- Upstream: `star/defaults/controls.defaults:2203` (`MLT_option = 'TDC'`)
- Confidence: verified-by-reading-both-sides
- What: At this upstream commit an inlist that does not set `MLT_option` runs time-dependent convection. FlexMESA runs instantaneous Cox MLT. TDC relaxes to Cox MLT only for long timesteps; on short timesteps (late burning, pulsations, pre-MS relaxation) convective velocities and gradT differ. Even when TDC is selected, see F2-18.
- Proposed fix: default to `Tdc`; reject unknown option strings. Size S.

### F2-18: TDC call site: failures silently fall back to Cox MLT; TDC is skipped when `mlt_vc_old <= 0`; several inputs are not upstream's
- Area: flexmesa-star `hydro_vars.rs`
- Severity: wrong physics
- FlexMESA: `crates/flexmesa-star/src/hydro_vars.rs:225-229` (`let tdc_res = if mlt_opt == MltOption::Tdc && model.dt > 0.0 && k > 0 && vc_old > 0.0 {`), `:287` (`set_tdc(&tdc_params, &tdc_in).ok().map(|r| r.output)`), `:290-292` (`let out = if let Some(tdc_out) = tdc_res { tdc_out } else { ... set_mlt ... }`), `:244-262` (`tdc_alpha_d: 1.0, tdc_alpha_pt: 0.0, tdc_alpha_c: 1.0, tdc_alpha_s: 1.0, ... scale: lum[0].abs().max(lum[k].abs()).max(1.0e20), max_conv_vel: 1.0e99, ... y_face_guess: (gradr_f.val - gradl_f.val).max(0.0)`), `:283` (`eq_div_w: StarAdOrder1::ZERO`)
- Upstream: `star/private/turb_support.f90` `Get_results`: `using_TDC` depends on `MLT_option == 'TDC'`, `have_mlt_vc`, `k > 0`, `dt > 0`, `check_if_must_fall_back_to_MLT`; `scale = max(abs(s% L(k)), 1d-3*maxval(s% L(1:s% nz)))`; `Y_face_guess = s% Y_face(k)` only if `use_TDC_Y_face_seeded_newton`, else 0; `set_TDC(... s%TDC_alpha_D, s%TDC_alpha_R, s%TDC_alpha_Pt, ... s% TDC_alpha_C, s% TDC_alpha_S ...)`; `if (ierr /= 0) return`
- Confidence: verified-by-reading-both-sides
- What: (i) A TDC solver error is converted to `None` by `.ok()` and the face is then evaluated with Cox MLT; upstream propagates `ierr` so the step is retried. A TDC run can therefore be a face-by-face, iteration-by-iteration mixture of TDC and MLT without any record. (ii) TDC needs a positive start velocity in FlexMESA (`vc_old > 0.0`); upstream starts TDC from `conv_vel_start = 0` as well, which is how convection switches on in a time-dependent way. In FlexMESA a newly unstable face is given the instantaneous MLT velocity instead. (iii) The residual scale uses a 1e20 erg/s floor and the surface luminosity instead of `1e-3*max(L)`. (iv) `TDC_alpha_D`, `alpha_Pt`, `alpha_C`, `alpha_S`, `max_conv_vel` are literals, so inlist values are ignored (only `TDC_alpha_R` is read). (v) `dV` uses `lnd_old` interpolated to the face rather than upstream's `get_rho_start_face`.
- Proposed fix: propagate `set_tdc` errors; drop the `vc_old > 0` gate; use upstream's `scale` and guess rules; wire the alpha controls. Size M.

### F2-19: Surface face multiplies gradr by `1 + 2/(3 tau)` when tau < 2/3; no such factor exists upstream
- Area: flexmesa-star `hydro_vars.rs`
- Severity: wrong physics (outermost face)
- FlexMESA: `crates/flexmesa-star/src/hydro_vars.rs:182-184` (`if k == 0 && model.tau[0] < 2.0 / 3.0 && model.tau[0] > 0.0 { gradr_f = gradr_f * (1.0 + 2.0 / (3.0 * model.tau[0])); }`)
- Upstream: `star/private/turb_info.f90:132-141` (`gradr_ad = gradr_ad*gradr_factor` with `gradr_factor = s% gradr_factor(k)` or the rotation ratio); `star/private/hydro_vars.f90:597` (`s% gradr_factor(k) = 1.0_dp + s% gradr_factor(k)`, the T-tau `dq/dtau` correction, only with `use_T_tau_gradr_factor`); `grep` for `3d0*tau`, `3*tau` in `star/private` finds nothing
- Confidence: verified-by-reading-both-sides (absence established by grep, so slightly weaker than a positive match)
- What: With `tau_factor < 1` (surface at tau < 2/3) the radiative gradient at the outermost face is inflated by up to several times, which can make the surface face spuriously convective and changes gradT there. The upstream T-tau correction is `1 + dq/dtau`, which is exactly 1 for the Eddington relation, and is off by default.
- Proposed fix: remove; implement `use_T_tau_gradr_factor` properly if needed. Size S.

### F2-20: MLT small-`delta` series replaces upstream's Cardano root for `delta <= 1e-6`; early exits return errors instead of radiative defaults
- Area: flexmesa-turb `mlt.rs`
- Severity: wrong numerics (minor, intentional)
- FlexMESA: `crates/flexmesa-turb/src/mlt.rs:160-170` (`} else if delta.val() <= 1.0e-6 { let poly = one - delta + (2 - a0)*delta^2; let g = delta*poly; let z = a0*delta^2*poly^3;`), `:80-93` (input validation returning `TurbError::InvalidParameter`), `:233-235` (`is_bad(zeta)` returns `Err`)
- Upstream: `turb/private/mlt.f90` (always `f1 = pow(f1, one_third) ... Gamma = (four_13*f1 + f2 - 2d0)/(6d0*a0)`; `if (is_bad(Zeta%val)) return` with the preset radiative outputs)
- Confidence: verified-by-reading-both-sides
- What: The series is the correct expansion of the same cubic (I re-derived `Gamma = delta - delta^2 + (2 - a0) delta^3`), and is more accurate than Cardano where the latter cancels, but results are not bit-comparable to upstream for weakly superadiabatic faces and the switch at 1e-6 introduces an O(delta^4) step. Where upstream silently returns "no mixing" on a bad Zeta, FlexMESA raises an error that aborts `set_mlt_vars`.
- Proposed fix: document in the discrepancy ledger; mirror upstream's silent return for bad Zeta. Size S.

### F2-21: Thermohaline always uses He4 as the driving species and total hydrogen for XH1
- Area: flexmesa-star `hydro_vars.rs`
- Severity: wrong numerics (non-Kippenhahn options only)
- FlexMESA: `crates/flexmesa-star/src/hydro_vars.rs:336-341` (`xh1: model.xh_mass[k], ... species: DrivingSpecies::he4(),`)
- Upstream: `star/private/turb_support.f90` `do1_mlt_eval` (`iso = s% dominant_iso_for_thermohaline(k)`; `XH1 = s% xa(s% net_iso(ih1),k)`)
- Confidence: verified-by-reading-both-sides
- What: For `Traxler_Garaud_Stellmach_11` and `Brown_Garaud_Stellmach_13` the molecular diffusivity `K_mu` depends on the charge and mass of the species driving the mu inversion (C12 in accreting WDs, for example). FlexMESA fixes it to He4. The Kippenhahn default does not use it. Also the semiconvection/thermohaline block is gated on `k > 0 && use_ledoux_criterion` plus `(gradr - grada) > 1e-40`, where upstream gates on the sign of `gradL_composition_term` and `gradr > grada`; equivalent in effect because the composition term is zero without Ledoux.
- Proposed fix: port `dominant_iso_for_thermohaline`. Size S.

### Turb: not examined line by line
`tdc.rs`, `tdc_search.rs`, `tdc_support.rs` (about 1000 lines) vs `turb/private/tdc.f90`, `tdc_support.f90`; `check_for_redo_mlt`; `premix.rs`, `pre_ms.rs`, `rsp2_remesh.rs` call sites.

## 5. Atmosphere (`crates/flexmesa-atm`, call site `flexmesa-star/src/hydro_brunt_surf.rs::eval_surf_bc`)

### What subset is implemented
- Crate: `eval_t_tau_uniform`, `eval_t_tau_varying`, tables (`tau_100`, `tau_10`, `tau_1`, `tau_1m1`, `photosphere`, `WD_tau_25`, `DB_WD_tau_25`), `irradiated` (`flexmesa-atm/src/*.rs`).
- Reachable from star: `atm_option = 'T_tau'` with fixed opacity only, `'table'`, and a non-upstream `'simple_photosphere'`. `atm_T_tau_opacity = 'iterated'` / `'varying'`, `'irradiated_grey'`, `'fixed_Teff'`, `'fixed_Tsurf'`, `'fixed_Psurf'`, `'fixed_Psurf_and_Tsurf'` are not reachable from `eval_surf_bc` (the EOS/kap callbacks passed are stubs that return `Err`, `hydro_brunt_surf.rs:357-372`).

### Matches (atm)
- T(tau) relations: Eddington, solar Hopf (`Q1..Q5 = 1.0361, -0.3134, 2.44799995, -0.29589999, 30`), Krishna Swamy (`1.39, -0.815, 2.54, -0.025, 30`), Trampedach solar (`c0, c1, a, v, b, w`), their `dq/dtau`, and `tau_base` values (`2/3`, `0.4116433502`, `0.3121563`, `0.5147929058057147`): `flexmesa-atm/src/t_tau_relations.rs:9-125` vs `atm/private/atm_t_tau_relations.f90:53-59,134-138,170-174` and `eval_Trampedach_solar*`. Exact. (Upstream prints a warning for Trampedach beyond log10 tau >= 0.0741; FlexMESA is silent.)
- `'fixed'` opacity path calls the uniform T-tau evaluator with `max_iters = 0` and cell-1 opacity: `hydro_brunt_surf.rs:374-389` vs `star/private/atm_support.f90:281-290`.
- Defaults `atm_option = 'T_tau'`, `atm_T_tau_relation = 'Eddington'`: `controls_defaults.rs:36-38` vs `controls.defaults:4185,4280`.

### F2-22: `atm_option = 'table'` without an explicit `atm_table` uses an invented surface boundary condition instead of upstream's default `tau_100` tables
- Area: flexmesa-star `hydro_brunt_surf.rs`, `controls_defaults.rs`
- Severity: wrong physics
- FlexMESA: `crates/flexmesa-star/src/controls_defaults.rs:37` (`atm_table: String::new(),`); `hydro_brunt_surf.rs:216-217` (table branch requires `!model.controls.atm_table.is_empty()`), `:260-263` (`let legacy_table_proxy = model.controls.atm_option == AtmOption::Table;`), `:318-345` (`let p_start = model.peos[0] * (chi_rho*dln_d + chi_t*dln_t).clamp(-5.0, 5.0).exp(); ... let p_gas_surf = (p_start - dp_0 - 0.5 * p_rad).max(1.0); let kap_eff = (tau_surf * g_s / p_gas_surf).max(1.0e-10);`), `:395-405` (`lnt_ad.val = model.ln_t_start[0] + (teff / teff_s.max(1.0)).ln();`)
- Upstream: `star/defaults/controls.defaults:4385` (`atm_table = 'tau_100'`); `star/private/atm_support.f90:84-92` (`case ('table')` -> `get_table`)
- Confidence: verified-by-reading-both-sides
- What: When an inlist sets `atm_option = 'table'` and relies on the default table, FlexMESA does not read any atmosphere table. It manufactures an "effective opacity" from the start-of-step first-cell pressure so that the grey Eddington formula reproduces roughly the pressure the model already has, and sets the surface temperature to the start-of-step first-cell temperature scaled by Teff/Teff_start. The boundary condition therefore follows the interior instead of constraining it; the opacity partials are zeroed. This has no upstream counterpart.
- Proposed fix: default `atm_table` to `tau_100` and delete the proxy path. Size S.

### F2-23: Unknown `atm_option` / `atm_T_tau_relation` strings silently become T_tau / Eddington; `'simple_photosphere'` is implemented although upstream rejects it; `Pextra_factor` and `atm_T_tau_opacity` are ignored on the T_tau path
- Area: flexmesa-star `inlist_populate.rs`, `hydro_brunt_surf.rs`
- Severity: wrong physics (controls ignored)
- FlexMESA: `crates/flexmesa-star/src/inlist_populate.rs:392-410` (`_ => AtmOption::TTau`, `_ => AtmTTauRelation::Eddington`); `hydro_brunt_surf.rs:191-214` (simple photosphere with `1.6e-4 * pextra_factor` term); `:381` (literal `1.0` passed as `Pextra_factor` to `eval_t_tau_uniform`); `:386` (`0` max iterations always)
- Upstream: `star/private/atm_support.f90:72-105` (unknown option -> `get_legacy`, which prints guidance and fails; `:875-881` says `'simple_photosphere'` is deprecated), `:284` (`s% opacity(1), s% Pextra_factor`), `:293-330` (`'iterated'`, `'varying'`)
- Confidence: verified-by-reading-both-sides
- What: `irradiated_grey`, `fixed_Teff`, `fixed_Tsurf`, `fixed_Psurf*` inlists run with a grey Eddington atmosphere; `atm_T_tau_opacity = 'varying'` (used by several upstream test cases and required for accurate cool-star surfaces) runs as `'fixed'`; `Pextra_factor` set in an inlist (common for massive stars near the Eddington limit) has no effect unless the non-upstream simple photosphere is selected. Table evaluation failures also fall back to grey: `get_table_alfa_beta(...).unwrap_or((1.0, 0.0))` and `if let Ok(tbl_out) = eval_table(...)` (`hydro_brunt_surf.rs:231-236`) discard errors, which may or may not coincide with upstream's `atm_off_table_option = 'T_tau'` blend.
- Proposed fix: error on unsupported options; pass `pextra_factor`; wire iterated/varying. Size M.

### Atm: not examined line by line
`t_tau_uniform.rs`, `t_tau_varying.rs`, `table.rs` (table blend with `atm_off_table_option`), `irradiated.rs` against `atm/private/*.f90`.

## 6. Thermal neutrinos (`crates/flexmesa-neu`)

### Matches (neu)
- Driver: `Tmin_neu = 1e7`, `log10Tmin_neu = 7` early zero return, per-channel conversion to specific rate (`d_rho = d*deni - val*deni`), cosine cutoff below `log10_Tlim` with its T derivative, totals: `flexmesa-neu/src/lib.rs:215-320`, `inputs.rs:20-23` vs `neu/private/mod_neu.f90` `neutrinos`, `neu/public/neu_def.f90:24`.
- `log10_Tlim = 7.5`, `non_nuc_neu_factor <= 0` zeroing: `flexmesa-star/src/micro.rs:351-358,375` vs `star/private/neu.f90:60,67`.
- Not compared term by term: `pair_plas.rs`, `phot.rs`, `brem.rs`, `reco.rs` (Itoh et al. fits) against `mod_neu.f90`.

### F2-24: Star disables recombination neutrinos; upstream includes all five channels
- Area: flexmesa-star `micro.rs::do_neu`, flexmesa-neu `NeuFlags::default_stellar`
- Severity: wrong physics (small)
- FlexMESA: `crates/flexmesa-neu/src/lib.rs` `default_stellar()` (`pair: true, plas: true, phot: true, brem: true, reco: false`), used at `crates/flexmesa-star/src/micro.rs:360`
- Upstream: `star/private/neu.f90:72-73` (`flags = .true.` followed by the commented-out `!flags(reco_neu_type) = .false.`)
- Confidence: verified-by-reading-both-sides
- What: FlexMESA reproduces a line upstream deliberately commented out. Recombination neutrino losses matter at moderate density and T around 1e7 to 1e8 K for high-Z plasma (cooling white dwarfs, degenerate cores). `min_t_for_neu = 1e7` (`controls_defaults.rs:41`) is not an upstream control but duplicates the library's own `Tmin_neu`, so it is harmless at its default.
- Proposed fix: use `NeuFlags::all()`. Size S.

## 7. chem, ionization, colors

- chem: `get_composition_info` (xh, xhe, `xz = max(0, min(1, 1 - (xh + xhe)))`, abar, zbar, z2bar, z53bar from tabulated `Z53`, ye, mass_correction) matches: `crates/flexmesa-chem/src/composition.rs:167-228` vs `chem/public/chem_lib.f90:204-232`. Partials `dabar_dx`, `dzbar_dx` are re-derived in `flexmesa-eos/src/blend_dt.rs:270-272` with the upstream formula. `set_basic_vars` adds a non-upstream `clamp_unit_roundoff` on xh/xhe/z (`flexmesa-star/src/micro.rs:206-212`): values within 1e-3 outside [0, 1] are clamped, values further out pass through unclamped; upstream clamps only xz. Solar abundance scales, nuclide table contents, and `chem_isos` data were not checked against `chem/public/chem_def.f90` / `data/chem_data`.
- ionization: not audited beyond noting the crate's own description (`ion_table.rs:8`, "Locate_logQ, Locate_logT with +1e-4 index bias") and that commit 47991c0 touched `ion_table_load.rs` for the no-data case. Lead only.
- colors: `SedCube::from_builtin` (`crates/flexmesa-colors/src/sed_cube.rs:93-105`) tries an external MESA flux cube and on any failure (`if let Ok(cube)`) silently uses an embedded Kurucz subset. Not compared with `colors/private/*.f90`. Lead only.

## Leads (not verified)

- L1 (EOS): `HelmTable::eval_electron_positron_order3` used by Skye (`skye/mod.rs:262`) and `HelmTable::cell` (`helm/table.rs:296-300`) clamp `iat`/`jat` into range; upstream Skye's `compute_ideal_ele` path may error off-table. Combined with F2-6 this may let Skye return values outside the HELM table where upstream skips.
- L2 (EOS): CMS defaults in `cms/config.rs:68-93` versus `eos.defaults` (`-1` everywhere) and the hard-coded upstream values in `eos/private/eoscms_eval.f90`.
- L3 (EOS, Python side): the JAX path `src/flexmesa/eos/helm.py`, `_blend_dt_jax.py` was routed through `_rust.eos_helm_cells_batch` "when helm_table.dat is not installed" in 47991c0; check whether a no-data fallback survives there.
- L4 (kap): `KapXTable::spline_data` builds its own not-a-knot bicubic (`loader.rs:39-41`, `spline.rs`); upstream builds with `interp_mkbicub_db` and specific boundary conditions in `load_kap.f90`. Interior values agree at nodes by construction, between nodes only if the boundary conditions match.
- L5 (kap): `kapcn_get` substitutes `fc = fn = 1` when `zbase <= 0` (`blend.rs:232-236`), where upstream divides by `Zbase` (and errors for `Zbase < 0`).
- L6 (kap/star): `do_kap` passes `model.z_mass[k]`, `xh_mass[k]` after `clamp_unit_roundoff`; upstream `kap_support.f90` has additional controls (`use_simple_es_for_kap`, `opacity_max`, `opacity_min`, `use_other_kap`, `low_logT_op_mono_full_off` etc.) none of which appear in `micro.rs::do_kap`.
- L7 (net): the three bundled rate tables include `r3` with 2 temperature points; upstream `rate_list.txt` lists `r1..r9` and the two al26 isomer tables. Why only `r3` of the nine generic tables is present was not investigated.
- L8 (net): `struct_burn_mix.rs` op-split burner passes the same hard-coded `rate_factors`/`weak_rate_factor` and was not compared with `star/private/struct_burn_mix.f90` / `net_burn.f90` tolerances.
- L9 (net): notes/DECISIONS.md:147,151 state that `min_logt_for_net = 8.3 / 8.5` is set in tier extras so cases converge with "0 ULP .photo restart parity"; the affected case list should be cross-checked by whoever audits the test_suite claims (see F2-13).
- L10 (turb): `check_for_redo_mlt` uses `(dr / hp_f.val.max(1.0)).max(1.0e-6)` as alpha (`hydro_vars.rs:196-198`); compare with `star/private/turb_info.f90` `check_for_redo_MLT`.
- L11 (colors): embedded Kurucz subset fallback in `sed_cube.rs:93-105`.

## Not reached

PC EOS, Skye Coulomb fits, CMS, eosPT, 4D-table spline construction; kap Type 2 dXC/dXO interpolation (`co.rs`), kapCN, AESOPUS, Compton formula; all of approx21, the general-net pairing code, extended screening, analytic `ratelib` fits, weak/ecapture rates, one-zone burners; TDC kernel; atm tables/varying/irradiated; the four neutrino fits; ionization; colors; chem data tables.
