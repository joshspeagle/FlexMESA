// SPDX-License-Identifier: GPL-3.0-or-later
//! Zone-level equation of state (`do_eos`), thermal neutrino loss (`do_neu`),
//! Rosseland mean opacity and optical depth (`do_kap`, `get_tau`), and nuclear
//! reaction network (`do_net`) evaluation (`star/private/micro.f90` and
//! `star/private/net.f90`).

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use flexmesa_chem::{
    get_composition_info, get_composition_info_from_props, NuclideCatalog,
    SpeciesProps,
};
use flexmesa_const::{
    AVO, CRAD, FOUR_PI, FOUR_THIRDS_PI, KERG, LN10, ONE_THIRD, QE,
};
use flexmesa_eos::{
    eos_dt_get_from_weights_with_tables, eos_dt_get_rho,
    eos_dt_get_zones_with_logs, CmsTableSet, EosConfig, EosDtTableSet,
    EosEvalOutput, HelmTable, I_CHI_RHO, I_CHI_T, I_CP, I_CV, I_ETA,
    I_FRAC_IDEAL, I_FRAC_PC, I_FRAC_SKYE, I_GAMMA1, I_GAMMA3, I_GRAD_AD,
    I_LATENT_DDLNRHO, I_LATENT_DDLNT, I_LN_E, I_LN_FREE_E, I_LN_PGAS, I_LN_S,
    I_MU, NUM_EOS_BASIC_RESULTS,
};
use flexmesa_kap::get_kap_results;
use flexmesa_net::NetGeneralInfo;
use flexmesa_neu::{neu_get, NeuFlags, NeuResult};
use flexmesa_rates::RateTable1D;
use flexmesa_utils::is_bad;
use rayon::prelude::*;

use crate::controls::{StarError, VarLayout};
pub use crate::micro_net::{build_rate_factors, do_net};
use crate::model::StarModel;
use crate::star_ad::{StarAdExt, StarAdOrder1};

static BASIC_NET_INFO: OnceLock<NetGeneralInfo> = OnceLock::new();
type NetCacheKey = (
    String,
    Option<PathBuf>,
    u64,
    u64,
    u64,
    u64,
    u64,
    bool,
    bool,
    u64,
    Vec<(String, String)>,
    Option<(u64, u64)>,
);
static NET_CACHE: OnceLock<Mutex<HashMap<NetCacheKey, Arc<NetGeneralInfo>>>> =
    OnceLock::new();

#[inline]
fn resolve_local_file(dir: &Path, file_name: &str) -> Option<PathBuf> {
    let trimmed = file_name.trim();
    if trimmed.is_empty() {
        return None;
    }
    let direct = dir.join(trimmed);
    if direct.is_file() {
        return Some(direct);
    }
    let nested = dir.join("nets").join(trimmed);
    if nested.is_file() {
        return Some(nested);
    }
    None
}

#[inline]
fn hash_bytes_fnv1a(hash: &mut u64, bytes: &[u8]) {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    for &b in bytes {
        *hash ^= u64::from(b);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
    *hash ^= 0xff;
    *hash = hash.wrapping_mul(FNV_PRIME);
}

fn local_net_files_stamp(
    net_name: &str,
    canonical_dir: Option<&Path>,
    special_rate_files: &[(String, String)],
) -> Option<(u64, u64)> {
    let dir = canonical_dir?;
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut total_len = 0_u64;
    let mut found_local = false;
    for file_name in std::iter::once(net_name).chain(
        special_rate_files
            .iter()
            .map(|(_, file_name)| file_name.as_str()),
    ) {
        if let Some(bytes) = resolve_local_file(dir, file_name)
            .and_then(|path| fs::read(&path).ok())
        {
            found_local = true;
            total_len = total_len
                .saturating_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX));
            hash_bytes_fnv1a(&mut hash, &bytes);
        }
    }
    found_local.then_some((total_len, hash))
}

thread_local! {
    static ACTIVE_NET_SEARCH_DIR: RefCell<Vec<PathBuf>> =
        const { RefCell::new(Vec::new()) };
}

/// RAII guard pushing a canonicalized case directory onto the thread-local
/// `.net` search stack used by [`resolve_model_net`].
pub(crate) struct NetSearchDirGuard {
    _private: (),
}

impl NetSearchDirGuard {
    /// Enters a scoped network search directory for the current thread.
    pub(crate) fn new(dir: &Path) -> Self {
        let canonical = canonical_dir_path(dir);
        ACTIVE_NET_SEARCH_DIR.with(|stack| stack.borrow_mut().push(canonical));
        Self { _private: () }
    }
}

impl Drop for NetSearchDirGuard {
    fn drop(&mut self) {
        ACTIVE_NET_SEARCH_DIR.with(|stack| {
            let _ = stack.borrow_mut().pop();
        });
    }
}

fn active_net_search_dir() -> Option<PathBuf> {
    ACTIVE_NET_SEARCH_DIR.with(|stack| stack.borrow().last().cloned())
}

fn canonical_dir_path(dir: &Path) -> PathBuf {
    std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf())
}

/// Returns the cached [`NetGeneralInfo`] for `"basic.net"`.
#[must_use]
pub fn global_basic_net() -> &'static NetGeneralInfo {
    BASIC_NET_INFO.get_or_init(|| {
        NetGeneralInfo::from_name("basic.net")
            .expect("valid built-in basic.net")
    })
}

/// Returns a cached [`Arc<NetGeneralInfo>`] for `net_name`, searching
/// optional `search_dir` (`<dir>/<file>`, then `<dir>/nets/<file>`) before
/// falling back to the embedded network bundle
/// (`net/private/net_initialize.f90:286-302`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if the `.net` file cannot be parsed.
pub fn get_or_load_net(
    net_name: &str,
    search_dir: Option<&Path>,
) -> Result<Arc<NetGeneralInfo>, StarError> {
    get_or_load_net_with_fe56ec(net_name, search_dir, 1.0e-7, 3.0e9)
}

/// Returns a cached [`Arc<NetGeneralInfo>`] for `net_name` with explicit
/// `fe56ec_fake_factor` and `min_t_for_fe56ec_fake_factor` applied.
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if the `.net` file cannot be parsed.
pub fn get_or_load_net_with_fe56ec(
    net_name: &str,
    search_dir: Option<&Path>,
    fe56ec_fake_factor: f64,
    min_t_for_fe56ec_fake_factor: f64,
) -> Result<Arc<NetGeneralInfo>, StarError> {
    get_or_load_net_with_star_params(
        net_name,
        search_dir,
        fe56ec_fake_factor,
        min_t_for_fe56ec_fake_factor,
        -1.0,
        -1.0,
        1.0,
        false,
        false,
        flexmesa_net::MAX_SAFE_LOGT_FOR_RATES,
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
fn get_or_load_net_with_star_params(
    net_name: &str,
    search_dir: Option<&Path>,
    fe56ec_fake_factor: f64,
    min_t_for_fe56ec_fake_factor: f64,
    net_logtcut_lo: f64,
    net_logtcut_lim: f64,
    reaction_neuqs_factor: f64,
    use_3a_fl87: bool,
    use_suzuki_weak_rates: bool,
    max_safe_logt_for_rates: f64,
    special_rate_files: &[(String, String)],
) -> Result<Arc<NetGeneralInfo>, StarError> {
    let canonical_dir = search_dir.map(canonical_dir_path);
    get_or_load_net_with_canonical_dir(
        net_name,
        canonical_dir.as_deref(),
        fe56ec_fake_factor,
        min_t_for_fe56ec_fake_factor,
        net_logtcut_lo,
        net_logtcut_lim,
        reaction_neuqs_factor,
        use_3a_fl87,
        use_suzuki_weak_rates,
        max_safe_logt_for_rates,
        special_rate_files,
    )
}

#[allow(clippy::too_many_arguments)]
fn get_or_load_net_with_canonical_dir(
    net_name: &str,
    canonical_dir: Option<&Path>,
    fe56ec_fake_factor: f64,
    min_t_for_fe56ec_fake_factor: f64,
    net_logtcut_lo: f64,
    net_logtcut_lim: f64,
    reaction_neuqs_factor: f64,
    use_3a_fl87: bool,
    use_suzuki_weak_rates: bool,
    max_safe_logt_for_rates: f64,
    special_rate_files: &[(String, String)],
) -> Result<Arc<NetGeneralInfo>, StarError> {
    let key = net_name.trim();
    let local_stamp =
        local_net_files_stamp(key, canonical_dir, special_rate_files);
    let cache_key = (
        key.to_string(),
        canonical_dir.map(Path::to_path_buf),
        fe56ec_fake_factor.to_bits(),
        min_t_for_fe56ec_fake_factor.to_bits(),
        net_logtcut_lo.to_bits(),
        net_logtcut_lim.to_bits(),
        reaction_neuqs_factor.to_bits(),
        use_3a_fl87,
        use_suzuki_weak_rates,
        max_safe_logt_for_rates.to_bits(),
        special_rate_files.to_vec(),
        local_stamp,
    );
    let cache = NET_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(guard) = cache.lock() {
        if let Some(info) = guard.get(&cache_key) {
            return Ok(Arc::clone(info));
        }
    }
    let mut rate_overrides: Vec<(String, Arc<RateTable1D>)> =
        Vec::with_capacity(special_rate_files.len());
    for (rxn_name, file_name) in special_rate_files {
        let tbl = RateTable1D::resolve_and_load(file_name, canonical_dir)
            .map_err(|e| {
                StarError::MicrophysicsError(format!(
                    "failed to load filename_of_special_rate '{file_name}' for '{rxn_name}': {e}"
                ))
            })?;
        rate_overrides.push((rxn_name.clone(), Arc::new(tbl)));
    }
    let mut loaded = NetGeneralInfo::from_name_or_dir_with_rate_overrides(
        key,
        canonical_dir,
        &rate_overrides,
    )
    .map_err(|e| {
        StarError::MicrophysicsError(format!(
            "failed to load network '{key}': {e}"
        ))
    })?;
    // `star/private/net.f90:685,741-748` and `net/public/net_def.f90:588-606`:
    // unconditionally assign `s% net_logTcut_{lo,lim}` and `s% fe56ec_*`
    // (overriding standalone `net_def` defaults despite the stale `<= 0`
    // comment in `controls.defaults:6786`), plus `s% reaction_neuQs_factor`
    // (`star/private/net.f90:202-210`) and `use_3a_fl87` / `use_suzuki_weak_rates`.
    loaded.fe56ec_fake_factor = fe56ec_fake_factor;
    loaded.min_t_for_fe56ec_fake_factor = min_t_for_fe56ec_fake_factor;
    loaded.log_tcut_lo = net_logtcut_lo;
    loaded.log_tcut_lim = net_logtcut_lim;
    loaded.reaction_neuqs_factor = reaction_neuqs_factor;
    loaded.use_3a_fl87 = use_3a_fl87;
    loaded.use_suzuki_weak_rates = use_suzuki_weak_rates;
    loaded.max_safe_logt_for_rates = max_safe_logt_for_rates;
    let info = Arc::new(loaded);
    if let Ok(mut guard) = cache.lock() {
        guard.insert(cache_key, Arc::clone(&info));
    }
    Ok(info)
}

/// Returns a cached [`Arc<NetGeneralInfo>`] matching `model.job.net_name`,
/// `model.species`, and `model.controls` / `model.job` rate settings.
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if `model.job.net_name` cannot be
/// loaded or if `net_info.num_isos != model.species`.
pub fn resolve_model_net(
    model: &StarModel,
) -> Result<Arc<NetGeneralInfo>, StarError> {
    let c = &model.controls;
    let j = &model.job;
    // `star/private/net.f90:735` -> `rates/public/rates_lib.f90:135-150`:
    // every slot with non-empty `(reaction_for_special_factor(i),
    // filename_of_special_rate(i))` overrides that reaction's raw rate table.
    let special_rate_files: Vec<(String, String)> = j
        .reaction_for_special_factor
        .iter()
        .zip(&j.filename_of_special_rate)
        .filter_map(|(rxn, fname)| {
            let r = rxn.trim();
            let f = fname.trim();
            if r.is_empty() || f.is_empty() {
                None
            } else {
                Some((r.to_string(), f.to_string()))
            }
        })
        .collect();
    let active_dir = active_net_search_dir();
    let net_info = get_or_load_net_with_canonical_dir(
        &j.net_name,
        active_dir.as_deref(),
        c.fe56ec_fake_factor,
        c.min_t_for_fe56ec_fake_factor,
        c.net_logtcut_lo,
        c.net_logtcut_lim,
        c.reaction_neuqs_factor,
        j.use_3a_fl87,
        j.use_suzuki_weak_rates,
        c.output.max_safe_logt_for_rates,
        &special_rate_files,
    )?;
    if net_info.num_isos != model.species {
        return Err(StarError::MicrophysicsError(format!(
            "network '{}' has {} species, expected model.species={}",
            model.job.net_name, net_info.num_isos, model.species
        )));
    }
    Ok(net_info)
}

#[inline]
const fn net_composition_bucket(z: usize) -> usize {
    match z {
        0 => 0,   // neut
        1 => 1,   // h
        2 => 2,   // he
        6 => 3,   // c
        7 => 4,   // n
        8 => 5,   // o
        10 => 6,  // ne
        12 => 7,  // mg
        14 => 8,  // si
        16 => 9,  // s
        18 => 10, // ar
        20 => 11, // ca
        // `star/private/adjust_xyz.f90:342-343,431-432,533-536` accumulates
        // `Z = 28` (Ni) into `total_fe`, leaving the `ni` bucket (14) zero.
        26 | 28 => 12, // fe
        27 => 13,      // co
        _ => 15,       // other
    }
}

/// Evaluate the test $\log_{10} T$ used for light-element abundance injection
/// (`get_test_lgT` in `star/private/adjust_xyz.f90:617-627`).
#[inline]
fn get_test_log10_t(ln_t: &[f64], mlt_d: &[f64], nz: usize, loc: usize) -> f64 {
    let mut lg_t = ln_t.get(loc).copied().unwrap_or(0.0) / LN10;
    let kmax = nz.min(ln_t.len()).min(mlt_d.len());
    for (k, &d) in mlt_d.iter().enumerate().take(kmax).skip(loc + 1) {
        if d < 100.0 {
            break;
        }
        lg_t = ln_t[k] / LN10;
    }
    lg_t
}

/// Remap an abundance array from the previous network onto the new network
/// (`set_x_new` / `set_new_abundances` in `star/private/adjust_xyz.f90:203-629`
/// and `do_clean1` in `net/private/net_eval.f90:822-850`).
#[allow(clippy::too_many_arguments)]
fn remap_xa_for_network(
    nz: usize,
    old_species: usize,
    old_buckets: &[usize],
    old_net_iso: &[Option<usize>],
    old_xa: &[f64],
    new_species: usize,
    new_iso_names: &[String],
    new_z: &[usize],
    new_buckets: &[usize],
    new_chem_id: &[usize],
    ln_t: &[f64],
    mlt_d: &[f64],
    initial_z: f64,
    adjust_abundances_for_new_isos: bool,
) -> Result<Vec<f64>, StarError> {
    const LGT_LO: f64 = 5.2;
    const LGT_HI: f64 = 5.5;
    const MAX_SUM_ABS: f64 = 10.0;
    const XSUM_TOL: f64 = 1.0e-2;
    const TINY_X: f64 = 1.0e-99;

    let zsol = flexmesa_chem::ag89_zsol();
    let zfrac = initial_z / zsol;
    let mut new_xa = vec![0.0_f64; nz * new_species];
    let last = new_species - 1;

    for k in 0..nz {
        let old_row = &old_xa[k * old_species..(k + 1) * old_species];
        let new_row = &mut new_xa[k * new_species..(k + 1) * new_species];

        let mut old_totals = [0.0_f64; 16];
        for (j_old, &b) in old_buckets.iter().enumerate() {
            old_totals[b] += old_row[j_old];
        }

        let lg_t = get_test_log10_t(ln_t, mlt_d, nz, k);
        for (j, &cid) in new_chem_id.iter().enumerate() {
            if let Some(Some(i_old)) = old_net_iso.get(cid) {
                new_row[j] = old_row[*i_old];
            } else if !adjust_abundances_for_new_isos {
                new_row[j] = 0.0;
            } else if new_z[j] > 5 {
                let xsol = flexmesa_chem::chem_xsol(&new_iso_names[j]);
                new_row[j] = zfrac * xsol;
            } else if lg_t >= LGT_HI {
                new_row[j] = 0.0;
            } else {
                let xsol = flexmesa_chem::chem_xsol(&new_iso_names[j]);
                if lg_t > LGT_LO {
                    let phase = (lg_t - LGT_LO) / (LGT_HI - LGT_LO);
                    new_row[j] =
                        (xsol * 0.5) * (1.0 + flexmesa_math::cospi(phase));
                } else {
                    new_row[j] = xsol;
                }
            }
        }

        let mut totals = [0.0_f64; 16];
        for (j, &b) in new_buckets.iter().enumerate() {
            totals[b] += new_row[j];
        }

        let mut did = [false; 16];
        for (j, &b) in new_buckets.iter().enumerate() {
            if totals[b] > 0.0 {
                new_row[j] = (new_row[j] * old_totals[b]) / totals[b];
                did[b] = true;
            }
        }

        for (&old_total, &did_flag) in old_totals.iter().zip(&did) {
            if old_total > 0.0 && !did_flag {
                new_row[last] += old_total;
            }
        }

        let pre_clean_sum: f64 = new_row.iter().sum();
        if !pre_clean_sum.is_finite() || (pre_clean_sum - 1.0).abs() > 0.1 {
            return Err(StarError::InvalidInput(format!(
                "set_new_abundances sum out of range at zone {k}: {pre_clean_sum}"
            )));
        }

        // `net/private/net_eval.f90:822-850` (`do_clean1`)
        let abs_sum: f64 = new_row.iter().map(|v| v.abs()).sum();
        if is_bad(abs_sum) || abs_sum > MAX_SUM_ABS {
            return Err(StarError::InvalidInput(format!(
                "clean1 invalid abundance sum at zone {k}: {abs_sum}"
            )));
        }
        for val in &mut *new_row {
            *val = (*val).clamp(TINY_X, 1.0);
        }
        let xsum: f64 = new_row.iter().sum();
        if is_bad(xsum) || (xsum - 1.0).abs() > XSUM_TOL {
            return Err(StarError::InvalidInput(format!(
                "clean1 mass-fraction sum out of tolerance at zone {k}: {xsum}"
            )));
        }
        for val in &mut *new_row {
            *val /= xsum;
        }
    }

    Ok(new_xa)
}

/// Projects `model` from its current isotope set onto `new_net_name`
/// (`change_net` / `do_composition_fixup` in
/// `star/private/adjust_xyz.f90:51-629`), initializing missing isotopes via
/// `set_new_abundances`, conserving elemental totals, and finalizing via
/// `clean1`.
///
/// # Errors
/// Returns [`StarError`] if `new_net_name` cannot be loaded or mass fractions
/// cannot be normalized cleanly.
pub fn change_model_network(
    model: &mut StarModel,
    new_net_name: &str,
    search_dir: Option<&Path>,
) -> Result<(), StarError> {
    change_model_network_with_adjust(model, new_net_name, search_dir, true)
}

/// [`change_model_network`] with an explicit `adjust_abundances_for_new_isos`
/// (`change_net(id, adjust_abundances_for_new_isos, new_net_name, ierr)`,
/// `star/private/adjust_xyz.f90:51-59`): when `false`, isotopes absent from
/// the old net start at zero before `clean1` (`adjust_xyz.f90:357-358`).
///
/// # Errors
/// Returns [`StarError`] if `new_net_name` cannot be loaded or mass fractions
/// cannot be normalized cleanly.
pub fn change_model_network_with_adjust(
    model: &mut StarModel,
    new_net_name: &str,
    search_dir: Option<&Path>,
    adjust_abundances_for_new_isos: bool,
) -> Result<(), StarError> {
    let trimmed = new_net_name.trim();
    let net_info = get_or_load_net(trimmed, search_dir)?;
    let new_chem_id = net_info.chem_id.clone();
    let new_species = new_chem_id.len();
    if new_species == 0 {
        return Err(StarError::InvalidInput(format!(
            "empty species list in network '{new_net_name}'"
        )));
    }
    // `star/private/adjust_xyz.f90:93`: `if (new_net_name == s% net_name) return`
    if model.job.net_name.trim() == trimmed && new_chem_id == model.chem_id {
        return Ok(());
    }

    let old_species = model.species;
    let nz = model.nz;
    let mut new_net_iso = vec![None; 7857];
    let catalog = NuclideCatalog::builtin_ref();
    let mut new_iso_names = Vec::with_capacity(new_species);
    let mut new_z = Vec::with_capacity(new_species);
    let mut new_buckets = Vec::with_capacity(new_species);
    for (j_new, &cid) in new_chem_id.iter().enumerate() {
        let entry = catalog.get_by_id(cid);
        let name =
            entry.map_or_else(|| format!("iso_{cid}"), |n| n.name.clone());
        let z = entry.map_or(15, |n| n.z);
        new_iso_names.push(name);
        new_z.push(z);
        new_buckets.push(net_composition_bucket(z));
        if cid < new_net_iso.len() {
            new_net_iso[cid] = Some(j_new);
        }
    }

    let old_buckets: Vec<usize> = model
        .chem_id
        .iter()
        .map(|&cid| {
            let z = catalog.get_by_id(cid).map_or(15, |n| n.z);
            net_composition_bucket(z)
        })
        .collect();

    let new_xa = remap_xa_for_network(
        nz,
        old_species,
        &old_buckets,
        &model.net_iso,
        &model.xa,
        new_species,
        &new_iso_names,
        &new_z,
        &new_buckets,
        &new_chem_id,
        &model.ln_t,
        &model.mlt_d,
        model.controls.initial_z,
        adjust_abundances_for_new_isos,
    )?;

    // `star/private/adjust_xyz.f90:148`: `if (s% generations > 1) call do_xa(s% nz_old, s% xh_old, s% xa_old)`
    let has_valid_xa_old = isize::try_from(model.model_number)
        .is_ok_and(|m| m > model.init_model_number)
        && old_species > 0
        && model.xa_old.len() == nz * old_species
        && model
            .xa_old
            .chunks_exact(old_species)
            .all(|row| row.iter().any(|&v| v > 0.0));
    let new_xa_old = if has_valid_xa_old {
        remap_xa_for_network(
            nz,
            old_species,
            &old_buckets,
            &model.net_iso,
            &model.xa_old,
            new_species,
            &new_iso_names,
            &new_z,
            &new_buckets,
            &new_chem_id,
            &model.ln_t_old,
            &model.mlt_d,
            model.controls.initial_z,
            adjust_abundances_for_new_isos,
        )?
    } else {
        new_xa.clone()
    };

    model.job.net_name = trimmed.to_string();
    model.species = new_species;
    model.chem_id = new_chem_id;
    model.iso_names = new_iso_names;
    model.net_iso = new_net_iso;
    model.layout = VarLayout::with_flags(
        new_species,
        model.controls.v_flag,
        model.controls.u_flag,
        model.controls.rsp2_flag,
        model.controls.rti_flag,
    );
    model.xa.clone_from(&new_xa);
    model.xa_start = new_xa;
    model.xa_old = new_xa_old;
    // `star/private/adjust_xyz.f90:116-120`: reset reaction/composition
    // derivatives and invalidate cached microphysics before the next set_vars.
    model.d_epsnuc_dx.resize(nz * new_species, 0.0);
    model.d_epsnuc_dx.fill(0.0);
    model.dxdt_nuc.resize(nz * new_species, 0.0);
    model.dxdt_nuc.fill(0.0);
    model.d_dxdt_nuc_dlnd.resize(nz * new_species, 0.0);
    model.d_dxdt_nuc_dlnd.fill(0.0);
    model.d_dxdt_nuc_dlnt.resize(nz * new_species, 0.0);
    model.d_dxdt_nuc_dlnt.fill(0.0);
    model
        .d_dxdt_nuc_dx
        .resize(nz * new_species * new_species, 0.0);
    model.d_dxdt_nuc_dx.fill(0.0);
    model.aux.resize(nz, new_species);
    model.aux.d_eps_grav_dx.fill(0.0);
    model.aux.dln_e_dxa_for_partials.fill(0.0);
    model.aux.dln_peos_dxa_for_partials.fill(0.0);
    model.aux.micro_cache.clear();
    // `star/private/adjust_xyz.f90:159` & `star/private/net.f90:687`.
    model.aux.need_to_setvars = true;
    Ok(())
}

/// Evaluates `rho`, `t`, `r`, `rmid`, `grav`, and composition moments (`abar`,
/// `zbar`, `z2bar`, `z53bar`, `ye`, `xh_mass`, `yhe_mass`, `z_mass`) across all
/// `nz` zones (`set_basic_vars` in `star/private/hydro_vars.f90:696-760`).
///
/// # Errors
/// Returns [`StarError`] if any zone contains non-finite state or invalid
/// composition.
pub fn set_basic_vars(model: &mut StarModel) -> Result<(), StarError> {
    let nz = model.nz;
    let species = model.species;
    let catalog = NuclideCatalog::builtin_ref();

    model.set_m_and_dm();
    model.set_m_grav_and_grav();

    for k in 0..nz {
        let lnd_k = model.lnd[k];
        let lnt_k = model.ln_t[k];
        if !lnd_k.is_finite()
            || !lnt_k.is_finite()
            || !model.ln_r[k].is_finite()
        {
            return Err(StarError::MicrophysicsError(format!(
                "non-finite primary state at zone {k}: lnd={lnd_k}, lnT={lnt_k}"
            )));
        }
        model.rho[k] = flexmesa_math::exp(lnd_k);
        model.t[k] = flexmesa_math::exp(lnt_k);

        let xa_k = &model.xa[k * species..(k + 1) * species];
        let comp = get_composition_info(catalog, &model.chem_id, xa_k, true)
            .map_err(|e| {
                StarError::MicrophysicsError(format!(
                    "composition error at zone {k}: {e}"
                ))
            })?;
        model.abar[k] = comp.abar;
        model.zbar[k] = comp.zbar;
        model.z2bar[k] = comp.z2bar;
        model.z53bar[k] = comp.z53bar;
        model.ye[k] = comp.ye;
        // `star/private/star_utils.f90:2029-2043` (`get_XYZ`).
        let xh = comp.xh.clamp(0.0, 1.0);
        let xhe = comp.xhe.clamp(0.0, 1.0);
        model.xh_mass[k] = xh;
        model.yhe_mass[k] = xhe;
        model.z_mass[k] = (1.0 - (xh + xhe)).clamp(0.0, 1.0);
    }
    Ok(())
}

/// Returns the shared [`HelmTable`] or maps table I/O errors into [`StarError`].
#[inline]
pub(crate) fn builtin_helm_table() -> Result<&'static HelmTable, StarError> {
    HelmTable::builtin()
        .map_err(|e| StarError::MicrophysicsError(e.to_string()))
}

/// Ensures EOS-populated auxiliary buffers on `model.aux` match `nz` and
/// `nz * species`.
fn ensure_eos_aux_len(model: &mut StarModel) {
    let nz = model.nz;
    let n_sp = nz * model.species;
    if model.aux.eos_frac_pc.len() != nz {
        model.aux.eos_frac_pc.resize(nz, 0.0);
    }
    if model.aux.latent_ddlnt.len() != nz {
        model.aux.latent_ddlnt.resize(nz, 0.0);
    }
    if model.aux.latent_ddlnrho.len() != nz {
        model.aux.latent_ddlnrho.resize(nz, 0.0);
    }
    if model.aux.dln_e_dxa_for_partials.len() != n_sp {
        model.aux.dln_e_dxa_for_partials.resize(n_sp, 0.0);
    }
    if model.aux.dln_peos_dxa_for_partials.len() != n_sp {
        model.aux.dln_peos_dxa_for_partials.resize(n_sp, 0.0);
    }
}

/// Validates and stores one zone's EOS output (`store_eos_for_cell` in
/// `star/private/micro.f90:360-545`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if any thermodynamic output or
/// derivative is non-finite, out of physical range (`T, rho > 1e15`), or
/// non-positive (`cs^2 < 0`, `gamma1`, `grada`, `csound`, `chiT`, `chiRho`,
/// `Cv`, `Cp`).
pub(crate) fn store_eos_for_cell(
    model: &mut StarModel,
    k: usize,
    out: &EosEvalOutput,
) -> Result<(), StarError> {
    ensure_eos_aux_len(model);
    for i in 0..NUM_EOS_BASIC_RESULTS {
        if is_bad(out.res[i] + out.d_dlnd[i] + out.d_dln_t[i]) {
            return Err(StarError::MicrophysicsError(format!(
                "store_eos_for_cell: non-finite EOS result[{i}] at zone {k}"
            )));
        }
    }

    let t_k = model.t[k];
    let rho_k = model.rho[k];
    if t_k > 1.0e15 || rho_k > 1.0e15 {
        return Err(StarError::MicrophysicsError(format!(
            "store_eos_for_cell: T ({t_k}) or rho ({rho_k}) > 1e15 at zone {k}"
        )));
    }

    let ln_pgas = out.res[I_LN_PGAS];
    let pgas = flexmesa_math::exp(ln_pgas);
    // `star/private/micro.f90:464`: `s% Prad(k) = crad * T*T*T*T / 3d0`.
    let prad = ((((CRAD * t_k) * t_k) * t_k) * t_k) / 3.0;
    let peos = prad + pgas;
    let ln_peos = flexmesa_math::log(peos);

    let d_lnpgas_dlnd = out.d_dlnd[I_LN_PGAS];
    let d_lnpgas_dlnt = out.d_dln_t[I_LN_PGAS];
    let d_lnpeos_dlnd = pgas * d_lnpgas_dlnd / peos;
    let d_lnpeos_dlnt = (pgas * d_lnpgas_dlnt + 4.0 * prad) / peos;

    let ln_e = out.res[I_LN_E];
    let energy = flexmesa_math::exp(ln_e);
    let ln_s = out.res[I_LN_S];
    let entropy = flexmesa_math::exp(ln_s);
    let grada = out.res[I_GRAD_AD];
    let cv = out.res[I_CV];
    let cp = out.res[I_CP];
    let chi_rho = out.res[I_CHI_RHO];
    let chi_t = out.res[I_CHI_T];
    let gamma1 = out.res[I_GAMMA1];

    let cs2 = gamma1 * peos / rho_k;
    if cs2 < 0.0 {
        return Err(StarError::MicrophysicsError(format!(
            "eval_csound: negative cs^2 ({cs2}) at zone {k}"
        )));
    }
    let csound = cs2.sqrt();
    if gamma1 <= 0.0
        || grada <= 0.0
        || csound <= 0.0
        || is_bad(csound)
        || chi_t <= 0.0
        || chi_rho <= 0.0
        || cv <= 0.0
        || cp <= 0.0
    {
        return Err(StarError::MicrophysicsError(format!(
            concat!(
                "store_eos_for_cell: non-positive thermodynamic state at zone {} ",
                "(gamma1={}, grada={}, csound={}, chiT={}, chiRho={}, Cv={}, Cp={})"
            ),
            k, gamma1, grada, csound, chi_t, chi_rho, cv, cp
        )));
    }

    // `star/private/micro.f90:480-481`.
    let gam = model.z53bar[k]
        * QE
        * QE
        * flexmesa_math::pow(
            FOUR_THIRDS_PI * AVO * rho_k * model.zbar[k] / model.abar[k],
            ONE_THIRD,
        )
        / (KERG * t_k);

    model.pgas[k] = pgas;
    model.ln_pgas[k] = ln_pgas;
    model.prad[k] = prad;
    model.peos[k] = peos;
    model.ln_peos[k] = ln_peos;
    model.energy[k] = energy;
    model.ln_e[k] = ln_e;
    model.entropy[k] = entropy;
    model.ln_s[k] = ln_s;
    model.grada[k] = grada;
    model.cv[k] = cv;
    model.cp[k] = cp;
    model.chi_rho[k] = chi_rho;
    model.chi_t[k] = chi_t;
    model.gamma1[k] = gamma1;
    model.gamma3[k] = out.res[I_GAMMA3];
    model.eta[k] = out.res[I_ETA];
    model.gam[k] = gam;
    model.mu[k] = out.res[I_MU];
    model.lnfree_e[k] = out.res[I_LN_FREE_E];
    model.csound[k] = csound;

    model.aux.eos_frac_pc[k] = out.res[I_FRAC_PC];
    model.aux.latent_ddlnt[k] = out.res[I_LATENT_DDLNT];
    model.aux.latent_ddlnrho[k] = out.res[I_LATENT_DDLNRHO];
    let species = model.species;
    let sp_base = k * species;
    for j in 0..species {
        let [d_lnpgas_dxa, d_lne_dxa] =
            out.d_dxa.get(j).copied().unwrap_or([0.0, 0.0]);
        model.aux.dln_e_dxa_for_partials[sp_base + j] = d_lne_dxa;
        model.aux.dln_peos_dxa_for_partials[sp_base + j] =
            pgas * d_lnpgas_dxa / peos;
    }

    model.ln_peos_ad[k] =
        StarAdOrder1::from_dlnd_dlnt_00(ln_peos, d_lnpeos_dlnd, d_lnpeos_dlnt);
    model.ln_e_ad[k] = StarAdOrder1::from_dlnd_dlnt_00(
        ln_e,
        out.d_dlnd[I_LN_E],
        out.d_dln_t[I_LN_E],
    );
    model.ln_s_ad[k] = StarAdOrder1::from_dlnd_dlnt_00(
        ln_s,
        out.d_dlnd[I_LN_S],
        out.d_dln_t[I_LN_S],
    );

    model.eos_res[k] = out.res;
    model.eos_d_dlnd[k] = out.d_dlnd;
    model.eos_d_dlnt[k] = out.d_dln_t;
    Ok(())
}

/// Tolerances and iteration cap of the constant-`(Pgas, T)` density solve
/// (`LOGRHO_TOL`, `LOGPGAS_TOL` in `star/private/micro.f90:311`;
/// `MAX_ITER_FOR_SOLVE` in `star/private/eos_support.f90:38`).
const LOGRHO_TOL: f64 = 1.0e-8;
const LOGPGAS_TOL: f64 = 1.0e-8;
const MAX_ITER_FOR_SOLVE: usize = 100;

/// One cell of `do_eos_for_cell` with `fix_Pgas = .true.`
/// (`star/private/micro.f90:326-361`): solves for density at fixed
/// `(Pgas, T)` through `solve_eos_given_PgasT`
/// (`star/private/eos_support.f90:341-347`), sets `lnd = logRho*ln10` and
/// `rho = exp(lnd)`, then evaluates the EOS again at that state.
///
/// Returns `(lnd, rho, eos)`.
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if the density solve or the EOS
/// evaluation fails, or if `logRho < -25`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_fix_pgas_cell(
    ht: &HelmTable,
    chem_id: &[usize],
    xa_k: &[f64],
    k: usize,
    ln_t: f64,
    t: f64,
    ln_pgas: f64,
    lnd_guess: f64,
    cfg: &flexmesa_eos::EosConfig,
) -> Result<(f64, f64, EosEvalOutput), StarError> {
    let log_t = ln_t / LN10;
    let log_pgas = ln_pgas / LN10;
    let inv = eos_dt_get_rho(
        ht,
        chem_id,
        xa_k,
        log_t,
        I_LN_PGAS as isize,
        log_pgas * LN10,
        lnd_guess / LN10,
        LOGRHO_TOL,
        LOGPGAS_TOL * LN10,
        MAX_ITER_FOR_SOLVE,
        cfg,
    )
    .map_err(|e| {
        StarError::MicrophysicsError(format!(
            "solve_eos_given_PgasT failed at zone {k}: {e}"
        ))
    })?;
    if inv.log10_result < -25.0 {
        return Err(StarError::MicrophysicsError(format!(
            "do_eos_for_cell failed at zone {k}: logRho ({}) < -25",
            inv.log10_result
        )));
    }
    let lnd = inv.log10_result * LN10;
    let rho = flexmesa_math::exp(lnd);
    let log_rho = lnd / LN10;
    let mut out = eos_dt_get_zones_with_logs(
        ht,
        &[rho],
        &[log_rho],
        &[t],
        &[log_t],
        chem_id,
        xa_k,
        cfg,
    )
    .map_err(|e| {
        StarError::MicrophysicsError(format!(
            "do_eos_for_cell: get_eos failed at zone {k}: {e}"
        ))
    })?;
    let eos = out.pop().ok_or_else(|| {
        StarError::MicrophysicsError(format!(
            "do_eos_for_cell: get_eos returned no result at zone {k}"
        ))
    })?;
    Ok((lnd, rho, eos))
}

/// Evaluates constant-`(Pgas, T)` EOS (`fix_Pgas = .true.`) over zones
/// `k_lo..=k_hi`, updating composition moments, `lnd`/`rho`, and all
/// cell-centered thermodynamic fields through [`store_eos_for_cell`]
/// (`set_eos_with_mask` / `do_eos_for_cell` in `star/private/micro.f90:321-385`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if composition evaluation,
/// density inversion, `logRho >= -25`, or thermodynamic validation fails.
pub(crate) fn do_eos_fix_pgas_range(
    model: &mut StarModel,
    k_lo: usize,
    k_hi: usize,
) -> Result<(), StarError> {
    let nz = model.nz;
    if nz == 0 || k_lo >= nz || k_lo > k_hi {
        return Ok(());
    }
    let k_end = k_hi.min(nz - 1);
    let species = model.species;
    let ht = builtin_helm_table()?;
    let catalog = NuclideCatalog::builtin_ref();
    ensure_eos_aux_len(model);

    for k in k_lo..=k_end {
        let xa_k = &model.xa[k * species..(k + 1) * species];
        let comp = get_composition_info(catalog, &model.chem_id, xa_k, true)
            .map_err(|e| {
                StarError::MicrophysicsError(format!(
                    "do_eos_fix_pgas_range composition error at zone {k}: {e}"
                ))
            })?;
        model.abar[k] = comp.abar;
        model.zbar[k] = comp.zbar;
        model.z2bar[k] = comp.z2bar;
        model.z53bar[k] = comp.z53bar;
        model.ye[k] = comp.ye;
        let xh = comp.xh.clamp(0.0, 1.0);
        let xhe = comp.xhe.clamp(0.0, 1.0);
        model.xh_mass[k] = xh;
        model.yhe_mass[k] = xhe;
        model.z_mass[k] = (1.0 - (xh + xhe)).clamp(0.0, 1.0);

        let (lnd_new, rho_new, eos) = solve_fix_pgas_cell(
            ht,
            &model.chem_id,
            xa_k,
            k,
            model.ln_t[k],
            model.t[k],
            model.ln_pgas[k],
            model.lnd[k],
            &model.eos_config,
        )?;
        model.lnd[k] = lnd_new;
        model.rho[k] = rho_new;
        store_eos_for_cell(model, k, &eos)?;
    }
    Ok(())
}

/// Evaluates `eosDT_get` across all `nz` zones in parallel and populates
/// cell-centered thermodynamic quantities, composition partial derivatives
/// (`dln_e_dxa_for_partials`, `dln_peos_dxa_for_partials`), and `StarAdOrder1`
/// variables (`do_eos_for_cell` and `store_eos_for_cell` in
/// `star/private/micro.f90:292-545` and `star/private/eos_support.f90:44-101`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if EOS evaluation fails or yields
/// non-physical thermodynamic derivatives in any zone.
pub fn do_eos(model: &mut StarModel) -> Result<(), StarError> {
    let report = model.controls.output.report_ierr;
    let res = do_eos_inner(model);
    crate::controls_output::report_result_ierr(report, res)
}

fn do_eos_inner(model: &mut StarModel) -> Result<(), StarError> {
    let nz = model.nz;
    let mut log_rho_zones = Vec::with_capacity(nz);
    let mut log_t_zones = Vec::with_capacity(nz);
    for k in 0..nz {
        let log_rho = model.lnd[k] / LN10;
        let log_t = model.ln_t[k] / LN10;
        if log_rho < -25.0 {
            return Err(StarError::MicrophysicsError(format!(
                "do_eos failed at zone {k}: logRho ({log_rho}) < -25"
            )));
        }
        log_rho_zones.push(log_rho);
        log_t_zones.push(log_t);
    }

    let ht = builtin_helm_table()?;
    let eos_outputs = eos_dt_get_zones_with_logs(
        ht,
        &model.rho,
        &log_rho_zones,
        &model.t,
        &log_t_zones,
        &model.chem_id,
        &model.xa,
        &model.eos_config,
    )
    .map_err(|e| StarError::MicrophysicsError(format!("do_eos failed: {e}")))?;

    ensure_eos_aux_len(model);
    for (k, out) in eos_outputs.iter().enumerate() {
        store_eos_for_cell(model, k, out)?;
    }
    Ok(())
}

/// Pre-extracted nuclide weights and EOS table references for repeated
/// single-zone `get_eos` calls (`star/private/eos_support.f90:44-101`).
pub(crate) struct EosZoneContext {
    ht: &'static HelmTable,
    dt_tables: &'static EosDtTableSet,
    cms_tables: &'static CmsTableSet,
    props: SpeciesProps,
    z_int: Vec<usize>,
    azion: Vec<f64>,
    acmi: Vec<f64>,
    a_baryon: Vec<f64>,
}

impl EosZoneContext {
    /// Builds an [`EosZoneContext`] for `chem_id` using the built-in EOS
    /// tables and nuclide catalog.
    pub(crate) fn new(chem_id: &[usize]) -> Result<Self, StarError> {
        let ht = builtin_helm_table()?;
        let catalog = NuclideCatalog::builtin_ref();
        let props =
            SpeciesProps::from_catalog(catalog, chem_id).map_err(|e| {
                StarError::MicrophysicsError(format!(
                    "get_eos composition catalog error: {e}"
                ))
            })?;
        let n = chem_id.len();
        let mut z_int = Vec::with_capacity(n);
        let mut azion = Vec::with_capacity(n);
        let mut acmi = Vec::with_capacity(n);
        let mut a_baryon = Vec::with_capacity(n);
        for &cid in chem_id {
            let nuc = catalog.get_by_id(cid).ok_or_else(|| {
                StarError::MicrophysicsError(format!(
                    "get_eos unknown chem_id {cid}"
                ))
            })?;
            z_int.push(nuc.z as usize);
            azion.push(nuc.z as f64);
            acmi.push(nuc.w);
            a_baryon.push(nuc.z_plus_n as f64);
        }
        Ok(Self {
            ht,
            dt_tables: EosDtTableSet::builtin(),
            cms_tables: CmsTableSet::builtin(),
            props,
            z_int,
            azion,
            acmi,
            a_baryon,
        })
    }

    /// Evaluates single-zone `get_eos` at explicit `(xa, rho, log_rho, t, log_t)`
    /// (`get_eos` in `star/private/eos_support.f90:44-101`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn eval_cell(
        &self,
        k: usize,
        xa: &[f64],
        rho: f64,
        log_rho: f64,
        t: f64,
        log_t: f64,
        cfg: &EosConfig,
    ) -> Result<EosEvalOutput, StarError> {
        // `star/private/eos_support.f90:70-76`: hard lower logRho cutoff.
        if log_rho < -25.0 {
            return Err(StarError::MicrophysicsError(format!(
                "get_eos failed at zone {k}: logRho ({log_rho}) < -25"
            )));
        }
        let comp = get_composition_info_from_props(&self.props, xa, true);
        eos_dt_get_from_weights_with_tables(
            self.ht,
            self.dt_tables,
            self.cms_tables,
            comp.xh,
            comp.xz,
            rho,
            log_rho,
            t,
            log_t,
            comp.abar,
            comp.zbar,
            xa,
            &self.z_int,
            &self.azion,
            &self.acmi,
            &self.a_baryon,
            cfg,
        )
        .map_err(|e| {
            StarError::MicrophysicsError(format!(
                "get_eos failed at zone {k}: {e}"
            ))
        })
    }
}

/// Minimum composition change `|xa(j, k) - xa_start(j, k)|` required before
/// `fix_d_eos_dxa_partials` evaluates finite-difference EOS composition
/// partials (`dxa_threshold` in `star/private/hydro_eqns.f90:400`).
const FIX_DXA_THRESHOLD: f64 = 1.0e-4;

/// Revises `dln_e_dxa_for_partials` and `dln_peos_dxa_for_partials` via
/// finite differences for zones using EOS components that lack analytic
/// composition derivatives (`PC`, `Skye`, `ideal`)
/// (`fix_d_eos_dxa_partials` in `star/private/hydro_eqns.f90:131-137, 373-544`).
///
/// For each zone `k` with `frac_without_dxa = eos_frac_PC + eos_frac_Skye +
/// eos_frac_ideal > 0` and each species `j` with
/// `|xa(j, k) - xa_start(j, k)| >= 1e-4`, evaluates the EOS at `xa_start` and
/// at `xa_start` with species `j` shifted by `dxa = xa(j, k) - xa_start(j, k)`,
/// combining analytic `dres_dxa` with the `frac_without_dxa`-weighted finite
/// difference. On a perturbed-composition EOS failure or non-finite
/// `res(i_lnPgas)`, zeroes both partial arrays for cell `k` without error
/// (`star/private/hydro_eqns.f90:462-469`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if `get_eos` fails at `xa_start` or
/// if any revised composition partial is non-finite (`check_dxa` in
/// `star/private/hydro_eqns.f90:529-544`).
pub fn fix_d_eos_dxa_partials(model: &mut StarModel) -> Result<(), StarError> {
    let nz = model.nz;
    let species = model.species;
    if nz == 0
        || species == 0
        || model.xa_start.len() != nz * species
        || model.eos_res.len() != nz
    {
        return Ok(());
    }
    ensure_eos_aux_len(model);

    let mut active_zones: Vec<(usize, f64)> = Vec::new();
    for k in 0..nz {
        // `star/private/hydro_eqns.f90:410`.
        let frac_without_dxa = model.eos_res[k][I_FRAC_PC]
            + model.eos_res[k][I_FRAC_SKYE]
            + model.eos_res[k][I_FRAC_IDEAL];
        if frac_without_dxa <= 0.0 {
            continue;
        }
        let base = k * species;
        let xa_k = &model.xa[base..base + species];
        let xa_start_k = &model.xa_start[base..base + species];
        if xa_start_k.iter().all(|&x| x == 0.0) {
            continue;
        }
        let has_large_dxa = xa_k
            .iter()
            .zip(xa_start_k.iter())
            .any(|(&x, &xs)| (x - xs).abs() >= FIX_DXA_THRESHOLD);
        if has_large_dxa {
            active_zones.push((k, frac_without_dxa));
        }
    }
    if active_zones.is_empty() {
        return Ok(());
    }

    let ctx = EosZoneContext::new(&model.chem_id)?;
    let updates: Result<Vec<(usize, Vec<f64>, Vec<f64>)>, StarError> =
        active_zones
            .into_par_iter()
            .map(|(k, frac_without_dxa)| {
                let base = k * species;
                let xa_k = &model.xa[base..base + species];
                let xa_start_k = &model.xa_start[base..base + species];
                let rho = model.rho[k];
                let log_rho = model.lnd[k] / LN10;
                let t = model.t[k];
                let log_t = model.ln_t[k] / LN10;
                let pgas = model.pgas[k];
                let peos = model.peos[k];

                // `star/private/hydro_eqns.f90:441-451`: baseline EOS at
                // `xa_start(:, k)` is invariant across species `j`.
                let base_eos = ctx.eval_cell(
                    k,
                    xa_start_k,
                    rho,
                    log_rho,
                    t,
                    log_t,
                    &model.eos_config,
                )?;
                let lne_with_xa_start = base_eos.res[I_LN_E];
                let lnpgas_with_xa_start = base_eos.res[I_LN_PGAS];

                let mut dln_e_k =
                    model.aux.dln_e_dxa_for_partials[base..base + species]
                        .to_vec();
                let mut dln_p_k =
                    model.aux.dln_peos_dxa_for_partials[base..base + species]
                        .to_vec();
                let mut xa_start_1 = xa_start_k.to_vec();

                for j in 0..species {
                    let dxa = xa_k[j] - xa_start_k[j];
                    if dxa.abs() < FIX_DXA_THRESHOLD {
                        continue;
                    }
                    // `star/private/hydro_eqns.f90:455-473`.
                    xa_start_1.copy_from_slice(xa_start_k);
                    xa_start_1[j] = xa_start_k[j] + dxa;
                    let pert = ctx.eval_cell(
                        k,
                        &xa_start_1,
                        rho,
                        log_rho,
                        t,
                        log_t,
                        &model.eos_config,
                    );
                    let out = match pert {
                        Ok(out) if !is_bad(out.res[I_LN_PGAS]) => out,
                        _ => {
                            // `star/private/hydro_eqns.f90:462-469`: punt
                            // silently by zeroing cell partials.
                            dln_e_k.fill(0.0);
                            dln_p_k.fill(0.0);
                            return Ok((k, dln_e_k, dln_p_k));
                        }
                    };
                    let [d_lnpgas_dxa, d_lne_dxa] =
                        out.d_dxa.get(j).copied().unwrap_or([0.0, 0.0]);

                    // `star/private/hydro_eqns.f90:480-485`.
                    let dlne = d_lne_dxa
                        + frac_without_dxa
                            * (out.res[I_LN_E] - lne_with_xa_start)
                            / dxa;
                    if is_bad(dlne) {
                        return Err(StarError::MicrophysicsError(format!(
                            "fix_d_eos_dxa_partials: bad dlnE_dxa_for_partials \
                             ({dlne}) at zone {k}, species {j}"
                        )));
                    }
                    dln_e_k[j] = dlne;

                    let dlnp = pgas * d_lnpgas_dxa / peos
                        + frac_without_dxa
                            * (pgas / peos)
                            * (out.res[I_LN_PGAS] - lnpgas_with_xa_start)
                            / dxa;
                    if is_bad(dlnp) {
                        return Err(StarError::MicrophysicsError(format!(
                            "fix_d_eos_dxa_partials: bad dlnPeos_dxa_for_partials \
                             ({dlnp}) at zone {k}, species {j}"
                        )));
                    }
                    dln_p_k[j] = dlnp;
                }
                Ok((k, dln_e_k, dln_p_k))
            })
            .collect();

    for (k, dln_e_k, dln_p_k) in updates? {
        let base = k * species;
        model.aux.dln_e_dxa_for_partials[base..base + species]
            .copy_from_slice(&dln_e_k);
        model.aux.dln_peos_dxa_for_partials[base..base + species]
            .copy_from_slice(&dln_p_k);
    }
    Ok(())
}

/// Evaluates thermal neutrino loss rates `non_nuc_neu`, per-channel losses
/// (`nonnucneu_{plas,brem,phot,pair,reco}`), and partial derivatives across all
/// `nz` zones (`do_neu_for_cell` in `star/private/neu.f90:49-127`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if `use_other_neu` is active without a
/// hook, or [`StarError::MicrophysicsError`] if `neu_get` fails.
pub fn do_neu(model: &mut StarModel) -> Result<(), StarError> {
    let report = model.controls.output.report_ierr;
    let res = do_neu_inner(model);
    crate::controls_output::report_result_ierr(report, res)
}

fn do_neu_inner(model: &mut StarModel) -> Result<(), StarError> {
    if model.controls.use_other_neu {
        return Err(StarError::InvalidInput(String::from(
            "use_other_neu = .true. requires an other_neu hook",
        )));
    }
    let nz = model.nz;
    if model.aux.nonnucneu_channels.len() != nz {
        model.aux.nonnucneu_channels.resize(nz, [0.0; 5]);
    }
    let min_t = model.controls.min_t_for_neu;
    let factor = model.controls.non_nuc_neu_factor;
    if factor <= 0.0 {
        model.non_nuc_neu.fill(0.0);
        model.d_non_nuc_neu_dlnd.fill(0.0);
        model.d_non_nuc_neu_dlnt.fill(0.0);
        model.aux.nonnucneu_channels.fill([0.0; 5]);
        return Ok(());
    }

    let flags = NeuFlags::all();
    let results: Result<Vec<(f64, f64, f64, [f64; 5])>, StarError> = (0..nz)
        .into_par_iter()
        .map(|k| {
            let t_k = model.t[k];
            if t_k < min_t {
                return Ok((0.0, 0.0, 0.0, [0.0; 5]));
            }
            let rho_k = model.rho[k];
            let log_t = model.ln_t[k] / LN10;
            let log_rho = model.lnd[k] / LN10;
            let NeuResult {
                total,
                pair,
                plas,
                phot,
                brem,
                reco,
                ..
            } = neu_get(
                Some(t_k),
                Some(log_t),
                Some(rho_k),
                Some(log_rho),
                model.abar[k],
                model.zbar[k],
                7.5,
                flags,
            )
            .map_err(|e| {
                StarError::MicrophysicsError(format!(
                    "neu_get failed at zone {k}: {e}"
                ))
            })?;
            let val = total.val * factor;
            if is_bad(val) {
                return Err(StarError::MicrophysicsError(format!(
                    "do_neu_for_cell: non-finite non_nuc_neu ({val}) at zone {k}"
                )));
            }
            // `star/private/neu.f90:106-115`: `loss(:)` is scaled by
            // `non_nuc_neu_factor` while `sources(:, ineu)` channels remain unscaled.
            Ok((
                val,
                total.d_rho * rho_k * factor,
                total.d_t * t_k * factor,
                [plas.val, brem.val, phot.val, pair.val, reco.val],
            ))
        })
        .collect();

    for (k, (val, dlnd, dlnt, channels)) in results?.into_iter().enumerate() {
        model.non_nuc_neu[k] = val;
        model.d_non_nuc_neu_dlnd[k] = dlnd;
        model.d_non_nuc_neu_dlnt[k] = dlnt;
        model.aux.nonnucneu_channels[k] = channels;
    }
    Ok(())
}

/// Evaluates Rosseland mean opacity `opacity` (radiative + Compton + electron
/// conduction, or simple electron scattering) in parallel across all `nz` zones
/// and integrates optical depth `tau` inward from the surface (`do_kap_for_cell`
/// in `star/private/micro.f90:548-664` and `star/private/kap_support.f90:245-307`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if `use_other_kap` is active without a
/// hook, or [`StarError::MicrophysicsError`] if opacity evaluation fails.
pub fn do_kap(model: &mut StarModel) -> Result<(), StarError> {
    let report = model.controls.output.report_ierr;
    let res = (|| {
        if model.nz > 0 {
            eval_kap_for_range(model, 0, model.nz - 1)?;
        } else if model.controls.use_other_kap {
            return Err(StarError::InvalidInput(String::from(
                "use_other_kap = .true. requires an other_kap hook",
            )));
        }
        get_tau(model);
        check_tau_with_report_ierr(model)
    })();
    crate::controls_output::report_result_ierr(report, res)
}

/// Validates optical-depth increments against non-finite values, emitting
/// upstream cell diagnostics when `report_ierr` is active
/// (`get_dtau1` and `get_tau` in `star/private/star_utils.f90:846-902`).
///
/// # Errors
/// Returns [`StarError::MicrophysicsError`] if any zone produces a non-finite
/// `dtau` or `tau`.
pub fn check_tau_with_report_ierr(model: &StarModel) -> Result<(), StarError> {
    let nz = model.nz;
    let report = model.controls.output.report_ierr;
    for k in 0..nz {
        let dtau = model.dm[k] * model.opacity[k]
            / (FOUR_PI * model.rmid[k] * model.rmid[k]);
        if is_bad(dtau) || is_bad(model.tau[k]) {
            let zone1 = k + 1;
            if report {
                eprintln!(
                    "get_tau error: dtau({zone1}) = {dtau:.16e}, dm = {:.16e}, opacity = {:.16e}, rmid = {:.16e}",
                    model.dm[k], model.opacity[k], model.rmid[k]
                );
            }
            return Err(StarError::MicrophysicsError(format!(
                "get_tau failed at zone {k}: non-finite dtau ({dtau})"
            )));
        }
    }
    Ok(())
}

/// Evaluates `do_kap_for_cell` (`star/private/micro.f90:548-664` and
/// `star/private/kap_support.f90:245-307`) in parallel across cells
/// `k_lo..=k_hi`, updating `opacity`, `d_opacity_dlnd`, `d_opacity_dlnt`, and
/// `aux.kap_frac` in deterministic index order without touching `tau`.
///
/// Shared by [`do_kap`], `predictive_mix::eval_mixing_coeffs`,
/// `conv_premix::update_model`, and `phase_separation::update_model_range`.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if `use_other_kap` is active without a
/// hook, or [`StarError::MicrophysicsError`] if composition or opacity
/// evaluation fails.
pub(crate) fn eval_kap_for_range(
    model: &mut StarModel,
    k_lo: usize,
    k_hi: usize,
) -> Result<(), StarError> {
    if model.controls.use_other_kap {
        return Err(StarError::InvalidInput(String::from(
            "use_other_kap = .true. requires an other_kap hook",
        )));
    }
    let nz = model.nz;
    if nz == 0 || k_lo >= nz || k_lo > k_hi {
        return Ok(());
    }
    let k_end = k_hi.min(nz - 1);
    let species = model.species;
    if model.aux.kap_frac.len() != nz {
        model.aux.kap_frac.resize(nz, [0.0; 4]);
    }
    let catalog = NuclideCatalog::builtin_ref();
    let iso_z: Vec<usize> = model
        .chem_id
        .iter()
        .map(|&cid| catalog.get_by_id(cid).map_or(0, |n| n.z))
        .collect();
    let iso_h12: Vec<bool> = model
        .chem_id
        .iter()
        .map(|&cid| {
            catalog.get_by_id(cid).is_some_and(|n| {
                n.z == 1 && (n.z_plus_n == 1 || n.z_plus_n == 2)
            })
        })
        .collect();
    let iso_he34: Vec<bool> = model
        .chem_id
        .iter()
        .map(|&cid| {
            catalog.get_by_id(cid).is_some_and(|n| {
                n.z == 2 && (n.z_plus_n == 3 || n.z_plus_n == 4)
            })
        })
        .collect();
    let base_kap_factor = model.controls.opacity_factor;
    let use_simple_es = model.controls.use_simple_es_for_kap;
    // `micro.f90:589-595` reads `xa_start` and `zbar_start` whenever the control
    // is set, with no fallback to the current composition.
    let use_start_comp = model.controls.use_starting_composition_for_kap;
    if use_start_comp && model.xa_start.len() != nz * species {
        return Err(StarError::InvalidInput(format!(
            "use_starting_composition_for_kap = .true. but xa_start holds {} \
             values where nz * species = {}",
            model.xa_start.len(),
            nz * species
        )));
    }
    let opacity_max = model.controls.opacity_max;
    let opacity_min = model.controls.opacity_min;
    let min_off = model.controls.rsp.min_logt_for_opacity_factor_off;
    let min_on = model.controls.rsp.min_logt_for_opacity_factor_on;
    let max_on = model.controls.rsp.max_logt_for_opacity_factor_on;
    let max_off = model.controls.rsp.max_logt_for_opacity_factor_off;
    let kap_cfg = &model.kap_config;
    let chem_id = &model.chem_id;

    let kap_results: Result<Vec<(f64, f64, f64, [f64; 4])>, StarError> =
        (k_lo..=k_end)
            .into_par_iter()
            .map(|k| {
                let (xa_k, zbar_k, xh_k, xhe_k, z_k) = if use_start_comp {
                    let xs = &model.xa_start[k * species..(k + 1) * species];
                    let comp = get_composition_info(catalog, chem_id, xs, true)
                        .map_err(|e| {
                            StarError::MicrophysicsError(format!(
                                "get_kap starting composition error at zone {k}: {e}"
                            ))
                        })?;
                    let xh_s = comp.xh.clamp(0.0, 1.0);
                    let xhe_s = comp.xhe.clamp(0.0, 1.0);
                    let z_s = (1.0 - (xh_s + xhe_s)).clamp(0.0, 1.0);
                    (xs, comp.zbar, xh_s, xhe_s, z_s)
                } else {
                    // `star/private/kap_support.f90:250` + `star_utils.f90:2029-2043`
                    // (`get_XYZ` directly from `s% xa(:, k)` so partial EOS
                    // updates in `predictive_mix` see trial `xa_mx` without
                    // requiring `xh_mass`/`yhe_mass`/`z_mass` to be overwritten).
                    let xa_slice = &model.xa[k * species..(k + 1) * species];
                    let mut xh_s = 0.0;
                    let mut xhe_s = 0.0;
                    for (j, &xj) in xa_slice.iter().enumerate() {
                        if iso_h12[j] {
                            xh_s += xj;
                        } else if iso_he34[j] {
                            xhe_s += xj;
                        }
                    }
                    let xh_c = xh_s.clamp(0.0, 1.0);
                    let xhe_c = xhe_s.clamp(0.0, 1.0);
                    let z_c = (1.0 - (xh_c + xhe_c)).clamp(0.0, 1.0);
                    (xa_slice, model.zbar[k], xh_c, xhe_c, z_c)
                };

                let (mut xc, mut xn, mut xo, mut xne) = (0.0, 0.0, 0.0, 0.0);
                for (&z, &xj) in iso_z.iter().zip(xa_k.iter()) {
                    match z {
                        6 => xc += xj,
                        7 => xn += xj,
                        8 => xo += xj,
                        10 => xne += xj,
                        _ => {}
                    }
                }
                // `star/private/kap_support.f90:251,276-288`: `eps = 1d-6`.
                const EPS: f64 = 1.0e-6;
                if !(-EPS..=1.0 + EPS).contains(&xc)
                    || !(-EPS..=1.0 + EPS).contains(&xn)
                    || !(-EPS..=1.0 + EPS).contains(&xo)
                    || !(-EPS..=1.0 + EPS).contains(&xne)
                    || xh_k < 0.0
                    || xhe_k < 0.0
                    || z_k < 0.0
                {
                    return Err(StarError::MicrophysicsError(format!(
                        concat!(
                            "get_kap: invalid mass fractions at zone {} ",
                            "(xh={}, xhe={}, z={}, xc={}, xn={}, xo={}, xne={})"
                        ),
                        k, xh_k, xhe_k, z_k, xc, xn, xo, xne
                    )));
                }

                let log_rho = model.lnd[k] / LN10;
                let log_t = model.ln_t[k] / LN10;
                let (kap_raw, dlnkap_dlnd, dlnkap_dlnt, kap_fracs) =
                    if use_simple_es {
                        (0.2 * (1.0 + xh_k), 0.0, 0.0, [0.0; 4])
                    } else {
                        let out = get_kap_results(
                            kap_cfg,
                            zbar_k,
                            xh_k,
                            z_k,
                            xc,
                            xn,
                            xo,
                            xne,
                            log_rho,
                            log_t,
                            model.lnfree_e[k],
                            model.eos_d_dlnd[k][I_LN_FREE_E],
                            model.eos_d_dlnt[k][I_LN_FREE_E],
                            model.eta[k],
                            model.eos_d_dlnd[k][I_ETA],
                            model.eos_d_dlnt[k][I_ETA],
                        )
                        .map_err(|e| {
                            StarError::MicrophysicsError(format!(
                                "get_kap_results failed at zone {k}: {e}"
                            ))
                        })?;
                        (
                            out.kap,
                            out.dlnkap_dln_rho,
                            out.dlnkap_dln_t,
                            out.kap_fracs,
                        )
                    };

                if is_bad(kap_raw) {
                    return Err(StarError::MicrophysicsError(format!(
                        "do_kap_for_cell: non-finite opacity ({kap_raw}) at zone {k}"
                    )));
                }

                // `star/private/micro.f90:622-652`: piecewise-linear `log10_T` ramp
                // on `opacity_factor` (held constant in `d_opacity_dlnT`) followed
                // by `opacity_max` and `opacity_min` clamps.
                let mut opacity_factor = base_kap_factor;
                if min_off > 0.0 {
                    if log_t >= max_off || log_t <= min_off {
                        opacity_factor = 1.0;
                    } else if log_t > max_on {
                        opacity_factor = 1.0
                            + (opacity_factor - 1.0) * (log_t - max_off)
                                / (max_on - max_off);
                    } else if log_t < min_on {
                        opacity_factor = 1.0
                            + (opacity_factor - 1.0) * (log_t - min_off)
                                / (min_on - min_off);
                    }
                }

                let mut kap = kap_raw * opacity_factor;
                let mut dlnd = kap * dlnkap_dlnd;
                let mut dlnt = kap * dlnkap_dlnt;
                if opacity_max > 0.0 && kap > opacity_max {
                    kap = opacity_max;
                    dlnd = 0.0;
                    dlnt = 0.0;
                }
                if opacity_min > 0.0 && kap < opacity_min {
                    kap = opacity_min;
                    dlnd = 0.0;
                    dlnt = 0.0;
                }
                if is_bad(kap) {
                    return Err(StarError::MicrophysicsError(format!(
                        "do_kap_for_cell: non-finite scaled opacity ({kap}) at zone {k}"
                    )));
                }
                Ok((kap, dlnd, dlnt, kap_fracs))
            })
            .collect();

    for (offset, (kap, dlnd, dlnt, fracs)) in
        kap_results?.into_iter().enumerate()
    {
        let k = k_lo + offset;
        model.opacity[k] = kap;
        model.d_opacity_dlnd[k] = dlnd;
        model.d_opacity_dlnt[k] = dlnt;
        model.aux.kap_frac[k] = fracs;
    }
    Ok(())
}

/// Integrates optical depth $\tau_k$ inward from $\tau_0 = \tau_{\mathrm{factor}} \tau_{\mathrm{base}}$
/// (`get_tau` in `star/private/star_utils.f90:871-902`).
pub fn get_tau(model: &mut StarModel) {
    let nz = model.nz;
    if nz == 0 {
        return;
    }
    model.tau[0] = model.controls.tau_factor * model.tau_base;
    for k in 1..nz {
        let dtau = model.dm[k - 1] * model.opacity[k - 1]
            / (FOUR_PI * model.rmid[k - 1] * model.rmid[k - 1]);
        model.tau[k] = model.tau[k - 1] + dtau;
    }
}
