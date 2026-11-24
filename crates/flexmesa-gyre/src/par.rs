// SPDX-License-Identifier: GPL-3.0-only
//! GYRE parameter groups (`gyre/gyre/src/par/*_par_m.fypp`): the derived
//! types with their default initialisers, the `read_*_par` namelist
//! readers, and `select_par` (`common/util_m.fypp:163-213`).

use crate::namelist::{NmlError, NmlGroup, NmlValue};

const LEN_64: usize = 64;
const LEN_256: usize = 256;
const LEN_2048: usize = 2048;
/// ForUM `FILENAME_LEN` (`forum/src/lib/const_m.fypp:67`).
const FILENAME_LEN: usize = 256;

fn unknown(group: &str, obj: &str) -> NmlError {
    NmlError(format!("unknown object '{obj}' in namelist &{group}"))
}

/// `model_par_t` (`par/model_par_m.fypp:36-60`).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelPar {
    /// Component `theta_s`.
    pub theta_s: f64,
    /// Component `Gamma_1`.
    pub gamma_1: f64,
    /// Component `dx_snap`.
    pub dx_snap: f64,
    /// Component `x_i`.
    pub x_i: f64,
    /// Component `x_o`.
    pub x_o: f64,
    /// Component `s`.
    pub s: f64,
    /// Component `beta_m`.
    pub beta_m: f64,
    /// Component `beta_p`.
    pub beta_p: f64,
    /// Component `x_match`.
    pub x_match: f64,
    /// Component `model_type`.
    pub model_type: String,
    /// Component `grid_type`.
    pub grid_type: String,
    /// Component `file_format`.
    pub file_format: String,
    /// Component `data_format`.
    pub data_format: String,
    /// Component `interp_type`.
    pub interp_type: String,
    /// Component `deriv_type`.
    pub deriv_type: String,
    /// Component `file`.
    pub file: String,
    /// Component `n`.
    pub n: i32,
    /// Component `add_center`.
    pub add_center: bool,
    /// Component `repair_As`.
    pub repair_as: bool,
    /// Component `constrain_derivs`.
    pub constrain_derivs: bool,
    /// Component `force_linear`.
    pub force_linear: bool,
    /// Component `resolve_match`.
    pub resolve_match: bool,
    /// Component `use_nabla_rad`.
    pub use_nabla_rad: bool,
}

impl Default for ModelPar {
    fn default() -> Self {
        Self {
            theta_s: 0.0,
            gamma_1: 5.0 / 3.0,
            dx_snap: 0.0,
            x_i: 0.0,
            x_o: 1.0,
            s: 1.0,
            beta_m: 1.0,
            beta_p: 1.0,
            x_match: 0.5,
            model_type: "HOM".into(),
            grid_type: "UNI".into(),
            file_format: String::new(),
            data_format: String::new(),
            interp_type: "CUBIC".into(),
            deriv_type: "MONO".into(),
            file: String::new(),
            n: 10,
            add_center: true,
            repair_as: false,
            constrain_derivs: true,
            force_linear: false,
            resolve_match: true,
            use_nabla_rad: false,
        }
    }
}

impl ModelPar {
    fn set(&mut self, obj: &str, v: &NmlValue) -> Result<(), NmlError> {
        match obj {
            "theta_s" => self.theta_s = v.to_real(obj)?,
            "gamma_1" => self.gamma_1 = v.to_real(obj)?,
            "dx_snap" => self.dx_snap = v.to_real(obj)?,
            "x_i" => self.x_i = v.to_real(obj)?,
            "x_o" => self.x_o = v.to_real(obj)?,
            "s" => self.s = v.to_real(obj)?,
            "beta_m" => self.beta_m = v.to_real(obj)?,
            "beta_p" => self.beta_p = v.to_real(obj)?,
            "x_match" => self.x_match = v.to_real(obj)?,
            "model_type" => self.model_type = v.to_string_len(obj, LEN_256)?,
            "grid_type" => self.grid_type = v.to_string_len(obj, LEN_256)?,
            "file_format" => {
                self.file_format = v.to_string_len(obj, LEN_256)?
            }
            "data_format" => {
                self.data_format = v.to_string_len(obj, LEN_256)?
            }
            // `interp_type` is declared with LEN(ml_p%deriv_type) (= 256).
            "interp_type" => {
                self.interp_type = v.to_string_len(obj, LEN_256)?
            }
            "deriv_type" => self.deriv_type = v.to_string_len(obj, LEN_256)?,
            "file" => self.file = v.to_string_len(obj, FILENAME_LEN)?,
            "n" => self.n = v.to_int(obj)?,
            "add_center" => self.add_center = v.to_logical(obj)?,
            "repair_as" => self.repair_as = v.to_logical(obj)?,
            "constrain_derivs" => self.constrain_derivs = v.to_logical(obj)?,
            "force_linear" => self.force_linear = v.to_logical(obj)?,
            "resolve_match" => self.resolve_match = v.to_logical(obj)?,
            "use_nabla_rad" => self.use_nabla_rad = v.to_logical(obj)?,
            _ => return Err(unknown("model", obj)),
        }
        Ok(())
    }
}

/// `mode_par_t` (`par/mode_par_m.fypp:34-40`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModePar {
    /// Component `l`.
    pub l: i32,
    /// Component `m`.
    pub m: i32,
    /// Component `n_pg_min`.
    pub n_pg_min: i32,
    /// Component `n_pg_max`.
    pub n_pg_max: i32,
    /// Component `tag`.
    pub tag: String,
}

impl Default for ModePar {
    fn default() -> Self {
        Self {
            l: 0,
            m: 0,
            n_pg_min: -i32::MAX,
            n_pg_max: i32::MAX,
            tag: String::new(),
        }
    }
}

impl ModePar {
    fn set(&mut self, obj: &str, v: &NmlValue) -> Result<(), NmlError> {
        match obj {
            "l" => self.l = v.to_int(obj)?,
            "m" => self.m = v.to_int(obj)?,
            "n_pg_min" => self.n_pg_min = v.to_int(obj)?,
            "n_pg_max" => self.n_pg_max = v.to_int(obj)?,
            "tag" => self.tag = v.to_string_len(obj, LEN_64)?,
            _ => return Err(unknown("mode", obj)),
        }
        Ok(())
    }
}

/// `osc_par_t` (`par/osc_par_m.fypp:36-68`).
#[derive(Debug, Clone, PartialEq)]
pub struct OscPar {
    /// Component `x_ref`.
    pub x_ref: f64,
    /// Component `x_atm`.
    pub x_atm: f64,
    /// Component `alpha_grv`.
    pub alpha_grv: f64,
    /// Component `alpha_gbc`.
    pub alpha_gbc: f64,
    /// Component `alpha_thm`.
    pub alpha_thm: f64,
    /// Component `alpha_hfl`.
    pub alpha_hfl: f64,
    /// Component `alpha_gam`.
    pub alpha_gam: f64,
    /// Component `alpha_pi`.
    pub alpha_pi: f64,
    /// Component `alpha_kar`.
    pub alpha_kar: f64,
    /// Component `alpha_kat`.
    pub alpha_kat: f64,
    /// Component `alpha_egv`.
    pub alpha_egv: f64,
    /// Component `alpha_rht`.
    pub alpha_rht: f64,
    /// Component `alpha_trb`.
    pub alpha_trb: f64,
    /// Component `variables_set`.
    pub variables_set: String,
    /// Component `inner_bound`.
    pub inner_bound: String,
    /// Component `outer_bound`.
    pub outer_bound: String,
    /// Component `outer_bound_cutoff`.
    pub outer_bound_cutoff: String,
    /// Component `outer_bound_branch`.
    pub outer_bound_branch: String,
    /// Component `inertia_norm`.
    pub inertia_norm: String,
    /// Component `time_factor`.
    pub time_factor: String,
    /// Component `conv_scheme`.
    pub conv_scheme: String,
    /// Component `deps_source`.
    pub deps_source: String,
    /// Component `deps_file`.
    pub deps_file: String,
    /// Component `deps_file_format`.
    pub deps_file_format: String,
    /// Component `tag_list`.
    pub tag_list: String,
    /// Component `adiabatic`.
    pub adiabatic: bool,
    /// Component `nonadiabatic`.
    pub nonadiabatic: bool,
    /// Component `quasiad_eigfuncs`.
    pub quasiad_eigfuncs: bool,
    /// Component `reduce_order`.
    pub reduce_order: bool,
}

impl Default for OscPar {
    fn default() -> Self {
        Self {
            x_ref: 1.0,
            x_atm: -1.0,
            alpha_grv: 1.0,
            alpha_gbc: 1.0,
            alpha_thm: 1.0,
            alpha_hfl: 1.0,
            alpha_gam: 1.0,
            alpha_pi: 1.0,
            alpha_kar: 1.0,
            alpha_kat: 1.0,
            alpha_egv: 0.0,
            alpha_rht: 0.0,
            alpha_trb: 0.0,
            variables_set: "GYRE".into(),
            inner_bound: "REGULAR".into(),
            outer_bound: "VACUUM".into(),
            outer_bound_cutoff: String::new(),
            outer_bound_branch: "E_NEG".into(),
            inertia_norm: "BOTH".into(),
            time_factor: "OSC".into(),
            conv_scheme: "FROZEN_PESNELL_1".into(),
            deps_source: "MODEL".into(),
            deps_file: String::new(),
            deps_file_format: String::new(),
            tag_list: String::new(),
            adiabatic: true,
            nonadiabatic: false,
            quasiad_eigfuncs: false,
            reduce_order: true,
        }
    }
}

impl OscPar {
    fn set(&mut self, obj: &str, v: &NmlValue) -> Result<(), NmlError> {
        match obj {
            "x_ref" => self.x_ref = v.to_real(obj)?,
            "x_atm" => self.x_atm = v.to_real(obj)?,
            "alpha_grv" => self.alpha_grv = v.to_real(obj)?,
            "alpha_gbc" => self.alpha_gbc = v.to_real(obj)?,
            "alpha_thm" => self.alpha_thm = v.to_real(obj)?,
            "alpha_hfl" => self.alpha_hfl = v.to_real(obj)?,
            "alpha_gam" => self.alpha_gam = v.to_real(obj)?,
            "alpha_pi" => self.alpha_pi = v.to_real(obj)?,
            "alpha_kar" => self.alpha_kar = v.to_real(obj)?,
            "alpha_kat" => self.alpha_kat = v.to_real(obj)?,
            "alpha_rht" => self.alpha_rht = v.to_real(obj)?,
            "alpha_trb" => self.alpha_trb = v.to_real(obj)?,
            "alpha_egv" => self.alpha_egv = v.to_real(obj)?,
            "inner_bound" => self.inner_bound = v.to_string_len(obj, LEN_64)?,
            "outer_bound" => self.outer_bound = v.to_string_len(obj, LEN_64)?,
            "outer_bound_cutoff" => {
                self.outer_bound_cutoff = v.to_string_len(obj, LEN_64)?;
            }
            "outer_bound_branch" => {
                self.outer_bound_branch = v.to_string_len(obj, LEN_64)?;
            }
            "variables_set" => {
                self.variables_set = v.to_string_len(obj, LEN_64)?
            }
            "inertia_norm" => {
                self.inertia_norm = v.to_string_len(obj, LEN_64)?
            }
            "time_factor" => self.time_factor = v.to_string_len(obj, LEN_64)?,
            "conv_scheme" => self.conv_scheme = v.to_string_len(obj, LEN_64)?,
            "deps_source" => self.deps_source = v.to_string_len(obj, LEN_64)?,
            "deps_file" => {
                self.deps_file = v.to_string_len(obj, FILENAME_LEN)?
            }
            "deps_file_format" => {
                self.deps_file_format = v.to_string_len(obj, LEN_256)?;
            }
            "tag_list" => self.tag_list = v.to_string_len(obj, LEN_2048)?,
            "adiabatic" => self.adiabatic = v.to_logical(obj)?,
            "nonadiabatic" => self.nonadiabatic = v.to_logical(obj)?,
            "quasiad_eigfuncs" => self.quasiad_eigfuncs = v.to_logical(obj)?,
            "reduce_order" => self.reduce_order = v.to_logical(obj)?,
            _ => return Err(unknown("osc", obj)),
        }
        Ok(())
    }
}

/// `rot_par_t` (`par/rot_par_m.fypp:34-44`).
#[derive(Debug, Clone, PartialEq)]
pub struct RotPar {
    /// Component `Omega_rot`.
    pub omega_rot: f64,
    /// Component `coriolis_method`.
    pub coriolis_method: String,
    /// Component `Omega_rot_source`.
    pub omega_rot_source: String,
    /// Component `Omega_rot_units`.
    pub omega_rot_units: String,
    /// Component `tag_list`.
    pub tag_list: String,
    /// Component `complex_lambda`.
    pub complex_lambda: bool,
    /// Component `rossby`.
    pub rossby: bool,
}

impl Default for RotPar {
    fn default() -> Self {
        Self {
            omega_rot: 0.0,
            coriolis_method: "NULL".into(),
            omega_rot_source: "MODEL".into(),
            omega_rot_units: "NONE".into(),
            tag_list: String::new(),
            complex_lambda: false,
            rossby: false,
        }
    }
}

impl RotPar {
    fn set(&mut self, obj: &str, v: &NmlValue) -> Result<(), NmlError> {
        match obj {
            "omega_rot" => self.omega_rot = v.to_real(obj)?,
            "coriolis_method" => {
                self.coriolis_method = v.to_string_len(obj, LEN_64)?
            }
            "omega_rot_source" => {
                self.omega_rot_source = v.to_string_len(obj, LEN_64)?;
            }
            "omega_rot_units" => {
                self.omega_rot_units = v.to_string_len(obj, LEN_64)?
            }
            "tag_list" => self.tag_list = v.to_string_len(obj, LEN_2048)?,
            "complex_lambda" => self.complex_lambda = v.to_logical(obj)?,
            "rossby" => self.rossby = v.to_logical(obj)?,
            _ => return Err(unknown("rot", obj)),
        }
        Ok(())
    }
}

/// `num_par_t` (`par/num_par_m.fypp:34-48`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumPar {
    /// Component `n_iter_max`.
    pub n_iter_max: i32,
    /// Component `deflate_roots`.
    pub deflate_roots: bool,
    /// Component `restrict_roots`.
    pub restrict_roots: bool,
    /// Component `diff_scheme`.
    pub diff_scheme: String,
    /// Component `r_root_solver`.
    pub r_root_solver: String,
    /// Component `c_root_solver`.
    pub c_root_solver: String,
    /// Component `ad_search`.
    pub ad_search: String,
    /// Component `nad_search`.
    pub nad_search: String,
    /// Component `matrix_type`.
    pub matrix_type: String,
    /// Component `dump_contour_template`.
    pub dump_contour_template: String,
    /// Component `tag_list`.
    pub tag_list: String,
}

impl Default for NumPar {
    fn default() -> Self {
        Self {
            n_iter_max: 50,
            deflate_roots: true,
            restrict_roots: true,
            diff_scheme: "COLLOC_GL2".into(),
            r_root_solver: "BRENT".into(),
            c_root_solver: "RIDDERS".into(),
            ad_search: "BRACKET".into(),
            nad_search: "AD".into(),
            matrix_type: "BLOCK".into(),
            dump_contour_template: String::new(),
            tag_list: String::new(),
        }
    }
}

impl NumPar {
    fn set(&mut self, obj: &str, v: &NmlValue) -> Result<(), NmlError> {
        match obj {
            "n_iter_max" => self.n_iter_max = v.to_int(obj)?,
            "deflate_roots" => self.deflate_roots = v.to_logical(obj)?,
            "restrict_roots" => self.restrict_roots = v.to_logical(obj)?,
            "diff_scheme" => self.diff_scheme = v.to_string_len(obj, LEN_64)?,
            "r_root_solver" => {
                self.r_root_solver = v.to_string_len(obj, LEN_64)?
            }
            "c_root_solver" => {
                self.c_root_solver = v.to_string_len(obj, LEN_64)?
            }
            "ad_search" => self.ad_search = v.to_string_len(obj, LEN_64)?,
            "nad_search" => self.nad_search = v.to_string_len(obj, LEN_64)?,
            "dump_contour_template" => {
                self.dump_contour_template =
                    v.to_string_len(obj, FILENAME_LEN)?;
            }
            "matrix_type" => self.matrix_type = v.to_string_len(obj, LEN_64)?,
            "tag_list" => self.tag_list = v.to_string_len(obj, LEN_2048)?,
            _ => return Err(unknown("num", obj)),
        }
        Ok(())
    }
}

/// `grid_par_t` (`par/grid_par_m.fypp:34-52`).
#[derive(Debug, Clone, PartialEq)]
pub struct GridPar {
    /// Component `x_i`.
    pub x_i: f64,
    /// Component `x_o`.
    pub x_o: f64,
    /// Component `w_osc`.
    pub w_osc: f64,
    /// Component `w_exp`.
    pub w_exp: f64,
    /// Component `w_ctr`.
    pub w_ctr: f64,
    /// Component `w_thm`.
    pub w_thm: f64,
    /// Component `w_str`.
    pub w_str: f64,
    /// Component `dx_min`.
    pub dx_min: f64,
    /// Component `dx_max`.
    pub dx_max: f64,
    /// Component `n_iter_max`.
    pub n_iter_max: i32,
    /// Component `resolve_ctr`.
    pub resolve_ctr: bool,
    /// Component `scaffold_src`.
    pub scaffold_src: String,
    /// Component `file`.
    pub file: String,
    /// Component `file_format`.
    pub file_format: String,
    /// Component `tag_list`.
    pub tag_list: String,
}

impl Default for GridPar {
    fn default() -> Self {
        Self {
            x_i: -f64::MAX,
            x_o: f64::MAX,
            w_osc: 0.0,
            w_exp: 0.0,
            w_ctr: 0.0,
            w_thm: 0.0,
            w_str: 0.0,
            // sqrt(EPSILON(0._RD)), folded at compile time.
            dx_min: f64::EPSILON.sqrt(),
            dx_max: f64::MAX,
            n_iter_max: 32,
            resolve_ctr: true,
            scaffold_src: "MODEL".into(),
            file: String::new(),
            file_format: String::new(),
            tag_list: String::new(),
        }
    }
}

impl GridPar {
    fn set(&mut self, obj: &str, v: &NmlValue) -> Result<(), NmlError> {
        match obj {
            "x_i" => self.x_i = v.to_real(obj)?,
            "x_o" => self.x_o = v.to_real(obj)?,
            "w_osc" => self.w_osc = v.to_real(obj)?,
            "w_exp" => self.w_exp = v.to_real(obj)?,
            "w_ctr" => self.w_ctr = v.to_real(obj)?,
            "w_thm" => self.w_thm = v.to_real(obj)?,
            "w_str" => self.w_str = v.to_real(obj)?,
            "dx_min" => self.dx_min = v.to_real(obj)?,
            "dx_max" => self.dx_max = v.to_real(obj)?,
            "n_iter_max" => self.n_iter_max = v.to_int(obj)?,
            "resolve_ctr" => self.resolve_ctr = v.to_logical(obj)?,
            "scaffold_src" => {
                self.scaffold_src = v.to_string_len(obj, LEN_64)?
            }
            "file" => self.file = v.to_string_len(obj, FILENAME_LEN)?,
            "file_format" => self.file_format = v.to_string_len(obj, LEN_64)?,
            "tag_list" => self.tag_list = v.to_string_len(obj, LEN_2048)?,
            _ => return Err(unknown("grid", obj)),
        }
        Ok(())
    }
}

/// `scan_par_t` (`par/scan_par_m.fypp:34-48`).
#[derive(Debug, Clone, PartialEq)]
pub struct ScanPar {
    /// Component `freq_min`.
    pub freq_min: f64,
    /// Component `freq_max`.
    pub freq_max: f64,
    /// Component `n_freq`.
    pub n_freq: i32,
    /// Component `freq_units`.
    pub freq_units: String,
    /// Component `freq_min_units`.
    pub freq_min_units: String,
    /// Component `freq_max_units`.
    pub freq_max_units: String,
    /// Component `freq_frame`.
    pub freq_frame: String,
    /// Component `grid_type`.
    pub grid_type: String,
    /// Component `grid_frame`.
    pub grid_frame: String,
    /// Component `axis`.
    pub axis: String,
    /// Component `file`.
    pub file: String,
    /// Component `tag_list`.
    pub tag_list: String,
}

impl Default for ScanPar {
    fn default() -> Self {
        Self {
            freq_min: 1.0,
            freq_max: 10.0,
            n_freq: 10,
            freq_units: "NONE".into(),
            freq_min_units: String::new(),
            freq_max_units: String::new(),
            freq_frame: "INERTIAL".into(),
            grid_type: "LINEAR".into(),
            grid_frame: "INERTIAL".into(),
            axis: "REAL".into(),
            file: String::new(),
            tag_list: String::new(),
        }
    }
}

impl ScanPar {
    fn set(&mut self, obj: &str, v: &NmlValue) -> Result<(), NmlError> {
        match obj {
            "freq_min" => self.freq_min = v.to_real(obj)?,
            "freq_max" => self.freq_max = v.to_real(obj)?,
            "n_freq" => self.n_freq = v.to_int(obj)?,
            "freq_units" => self.freq_units = v.to_string_len(obj, LEN_64)?,
            "freq_min_units" => {
                self.freq_min_units = v.to_string_len(obj, LEN_64)?
            }
            "freq_max_units" => {
                self.freq_max_units = v.to_string_len(obj, LEN_64)?
            }
            "freq_frame" => self.freq_frame = v.to_string_len(obj, LEN_64)?,
            "grid_type" => self.grid_type = v.to_string_len(obj, LEN_64)?,
            "grid_frame" => self.grid_frame = v.to_string_len(obj, LEN_64)?,
            "axis" => self.axis = v.to_string_len(obj, LEN_64)?,
            "file" => self.file = v.to_string_len(obj, FILENAME_LEN)?,
            "tag_list" => self.tag_list = v.to_string_len(obj, LEN_2048)?,
            _ => return Err(unknown("scan", obj)),
        }
        Ok(())
    }
}

/// Every parameter group of a GYRE input file, as `gyre_mesa_m::init`
/// reads them (`lib/gyre_mesa_m.fypp:82-112`).
#[derive(Debug, Clone, PartialEq)]
pub struct GyrePars {
    /// Component `model`.
    pub model: ModelPar,
    /// Component `mode`.
    pub mode: Vec<ModePar>,
    /// Component `osc`.
    pub osc: Vec<OscPar>,
    /// Component `rot`.
    pub rot: Vec<RotPar>,
    /// Component `num`.
    pub num: Vec<NumPar>,
    /// Component `grid`.
    pub grid: Vec<GridPar>,
    /// Component `scan`.
    pub scan: Vec<ScanPar>,
}

fn read_all<T: Default>(
    groups: &[NmlGroup],
    name: &str,
    set: impl Fn(&mut T, &str, &NmlValue) -> Result<(), NmlError>,
) -> Result<Vec<T>, NmlError> {
    let mut out = Vec::new();
    for g in groups.iter().filter(|g| g.name == name) {
        let mut p = T::default();
        for (obj, v) in &g.items {
            set(&mut p, obj, v)?;
        }
        out.push(p);
    }
    Ok(out)
}

impl GyrePars {
    /// Parse a GYRE input file's text.
    ///
    /// # Errors
    ///
    /// [`NmlError`] for malformed input, unknown objects, or anything but
    /// exactly one `&model` group (`model_par_m.fypp:122`).
    pub fn parse(text: &str) -> Result<Self, NmlError> {
        let groups = crate::namelist::parse_groups(text)?;
        let model = read_all(&groups, "model", ModelPar::set)?;
        if model.len() != 1 {
            return Err(NmlError(
                "input file should contain exactly one &model namelist".into(),
            ));
        }
        let mut scan = read_all(&groups, "scan", ScanPar::set)?;
        // scan_par_m.fypp:125-126
        for s in &mut scan {
            if s.freq_min_units.is_empty() {
                s.freq_min_units.clone_from(&s.freq_units);
            }
            if s.freq_max_units.is_empty() {
                s.freq_max_units.clone_from(&s.freq_units);
            }
        }
        Ok(Self {
            model: model.into_iter().next().ok_or_else(|| {
                NmlError(
                    "input file should contain exactly one &model namelist"
                        .into(),
                )
            })?,
            mode: read_all(&groups, "mode", ModePar::set)?,
            osc: read_all(&groups, "osc", OscPar::set)?,
            rot: read_all(&groups, "rot", RotPar::set)?,
            num: read_all(&groups, "num", NumPar::set)?,
            grid: read_all(&groups, "grid", GridPar::set)?,
            scan,
        })
    }
}

/// Parameter types carrying a `tag_list`.
pub trait Tagged: Clone {
    /// The comma-separated tag list.
    fn tag_list(&self) -> &str;
    /// Group name for error messages.
    const NAME: &'static str;
}

macro_rules! tagged {
    ($t:ty, $n:literal) => {
        impl Tagged for $t {
            fn tag_list(&self) -> &str {
                &self.tag_list
            }
            const NAME: &'static str = $n;
        }
    };
}
tagged!(OscPar, "osc");
tagged!(RotPar, "rot");
tagged!(NumPar, "num");
tagged!(GridPar, "grid");
tagged!(ScanPar, "scan");

/// `split_list(list, ',')` for the single-element and multi-element
/// cases (`common/string_m.fypp:65-133`): elements left-adjusted, trailing
/// blanks ignored by the comparisons that use them.
fn split_list(list: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = list.to_string();
    loop {
        if rest.trim_end().is_empty() {
            break;
        }
        match rest.find(',') {
            None => {
                out.push(rest.trim_start().trim_end().to_string());
                break;
            }
            Some(j) => {
                out.push(rest[..j].trim_start().trim_end().to_string());
                rest = rest[j + 1..].to_string();
            }
        }
    }
    out
}

/// `select_par(par, tag, par_sel(:))` (`util_m.fypp:185-213`): the groups
/// with an empty `tag_list` or whose list contains `tag`.
///
/// # Errors
///
/// [`NmlError`] when no group matches (upstream `ASSERT`).
pub fn select_par_v<T: Tagged>(
    par: &[T],
    tag: &str,
) -> Result<Vec<T>, NmlError> {
    let sel: Vec<T> = par
        .iter()
        .filter(|p| {
            p.tag_list().is_empty()
                || (!tag.is_empty()
                    && split_list(p.tag_list()).iter().any(|e| e == tag))
        })
        .cloned()
        .collect();
    if sel.is_empty() {
        return Err(NmlError(format!("no matching &{} namelists", T::NAME)));
    }
    Ok(sel)
}

/// `select_par(par, tag, par_sel)` (`util_m.fypp:163-181`): the last
/// matching group.
///
/// # Errors
///
/// As [`select_par_v`].
pub fn select_par_1<T: Tagged>(par: &[T], tag: &str) -> Result<T, NmlError> {
    let mut v = select_par_v(par, tag)?;
    v.pop()
        .ok_or_else(|| NmlError(format!("no matching &{} namelists", T::NAME)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_par_takes_untagged_and_matching_groups_last_wins() {
        let text = "&model /\n&mode l=0 tag='l=0' /\n&osc /\n&osc tag_list='l=1,l=0' outer_bound='JCD' /\n&osc tag_list='l=2' /";
        let p = GyrePars::parse(text).expect("parse");
        let v = select_par_v(&p.osc, "l=0").expect("select");
        assert_eq!(v.len(), 2);
        let one = select_par_1(&p.osc, "l=0").expect("select");
        assert_eq!(one.outer_bound, "JCD");
        assert!(GyrePars::parse("&mode /").is_err());
        assert!(GyrePars::parse("&model bogus = 1 /").is_err());
    }
}
