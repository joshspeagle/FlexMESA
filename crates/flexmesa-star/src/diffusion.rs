// SPDX-License-Identifier: GPL-3.0-or-later
//! Coupled multi-species Burgers element diffusion solver ported from MESA
//! `26.4.1` `star/private/diffusion.f90`, including both the adaptive
//! backward-Euler block-tridiagonal solver (`BE_solve`) and the block-tridiagonal
//! Rosenbrock integrator (`solve_with_isolve`, matching
//! `num/private/mod_rosenbrock.f`).

use flexmesa_const::{AMU, BOLTZM, CLIGHT, HBAR, KERG, ME, PI, QE};
use flexmesa_mtx::solve_block_tridiag_bcyclic;

use crate::diffusion_procs::{
    fixup, get_limit_coeffs, set_new_xa, setup_struct_info,
};
use crate::diffusion_support::{
    get1_burgers_coeffs, get1_diffusion_velocities, get1_gradient_coeffs,
    get_dstar_ocp, get_matrix_coeffs, MatrixCoeffsInput, MatrixCoeffsOutput,
    SMALL_X, TINY_X,
};
use crate::job_controls::StarError;

pub use crate::element_diffusion::do_element_diffusion;

/// Minimum number of diffusion classes (`diffusion.f90:35`).
pub const DIFFUSION_MIN_NC: usize = 4;

/// Whether Jacobian assembly includes the `d(coeff)/dX_face` derivative term
/// (`diffusion.f90:36`).
const USE_DCOEFF_DX: bool = true;

/// Collision-integral formulation for standalone single-zone Burgers utilities
/// and legacy PyO3/Rust helper callers.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum DiffusionScheme {
    /// Iben & MacDonald (1985) pure screened Coulomb logarithm.
    IbenMacDonald,
    /// Paquette, Pelletier, Fontaine & Michaud (1986) screened Coulomb fits.
    Paquette,
    /// Stanton & Murillo (2016) Yukawa collision integrals (ions) + Paquette
    /// (electrons).
    StantonMurillo,
    /// Stanton & Murillo (2016) blended with Caplan, Bauer & Freeman (2022)
    /// strong-coupling OCP liquid-metal coefficients over $\Gamma \in [3, 10]$.
    CaplanOcp,
}

/// Output of the single-zone multi-species Burgers solver
/// ([`eval_burgers_velocities`]).
#[derive(Debug, Clone)]
pub struct BurgersZoneResult {
    /// Net drift velocity $v_{\mathrm{gt},i}$ (gravitational settling + thermal
    /// diffusion + radiative levitation) in $\mathrm{cm\,s^{-1}}$ (`len = nc`).
    pub v_gt: Vec<f64>,
    /// Self-diffusion coefficient $D_{ii}$ in $\mathrm{cm^2\,s^{-1}}$
    /// (`len = nc`).
    pub d_self: Vec<f64>,
    /// Ambipolar electric force $e E$ in $\mathrm{dyn}$ (`eE`).
    pub ee_force: f64,
    /// Typical ionic charges $Z_i$ used in the zone (`len = nc`).
    pub z_charge: Vec<f64>,
}

/// Rosenbrock ODE solver variants supported by `diffusion_isolve_solver`
/// (`num/public/num_solvers_options.dek:7-24`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolveSolver {
    /// 2-stage 2nd-order L-stable `ros2_solver`.
    Ros2,
    /// 3-stage 2nd-order L-stable `rose2_solver`.
    Rose2,
    /// 3-stage 3rd-order `ros3p_solver`.
    Ros3p,
    /// 4-stage 3rd-order L-stable `ros3pl_solver`.
    Ros3pl,
    /// 4-stage 3rd-order stiffly accurate `rodas3_solver`.
    Rodas3,
    /// 6-stage 4th-order L-stable `rodas4_solver`.
    Rodas4,
    /// 6-stage 4th-order parabolic `rodasp_solver`.
    Rodasp,
}

impl IsolveSolver {
    /// Parses a MESA `diffusion_isolve_solver` string.
    ///
    /// # Errors
    ///
    /// Returns [`StarError::InvalidInput`] for any unrecognized solver name.
    pub fn from_name(name: &str) -> Result<Self, StarError> {
        match name.trim() {
            "ros2_solver" => Ok(Self::Ros2),
            "rose2_solver" => Ok(Self::Rose2),
            "ros3p_solver" => Ok(Self::Ros3p),
            "ros3pl_solver" => Ok(Self::Ros3pl),
            "rodas3_solver" => Ok(Self::Rodas3),
            "rodas4_solver" => Ok(Self::Rodas4),
            "rodasp_solver" => Ok(Self::Rodasp),
            other => Err(StarError::InvalidInput(format!(
                "unknown solver name for diffusion_isolve_solver: '{other}'"
            ))),
        }
    }
}

struct RosTable {
    ns: usize,
    gamma: f64,
    elo: f64,
    ra: [[f64; 6]; 6],
    rc: [[f64; 6]; 6],
    alpha: [f64; 6],
    ros_m: [f64; 6],
    ros_e: [f64; 6],
    newf: [bool; 6],
    rd31: f64,
    rd32: f64,
}

#[allow(
    clippy::approx_constant,
    clippy::excessive_precision,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]
fn ros_table(solver: IsolveSolver) -> RosTable {
    let mut t = RosTable {
        ns: 2,
        gamma: 1.0,
        elo: 2.0,
        ra: [[0.0; 6]; 6],
        rc: [[0.0; 6]; 6],
        alpha: [0.0; 6],
        ros_m: [0.0; 6],
        ros_e: [0.0; 6],
        newf: [true; 6],
        rd31: 0.0,
        rd32: 0.0,
    };
    match solver {
        IsolveSolver::Ros2 => {
            let g = 1.0 + 1.0 / 2.0_f64.sqrt();
            t.ns = 2;
            t.gamma = g;
            t.elo = 2.0;
            t.ra[1][0] = 1.0 / g;
            t.rc[1][0] = -2.0 / g;
            t.alpha = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
            t.ros_m = [3.0 / (2.0 * g), 1.0 / (2.0 * g), 0.0, 0.0, 0.0, 0.0];
            t.ros_e = [1.0 / (2.0 * g), 1.0 / (2.0 * g), 0.0, 0.0, 0.0, 0.0];
        }
        IsolveSolver::Rose2 => {
            let sqrt2 = 1.414_213_562_373_1_f64;
            let g = 1.0 / (2.0 + sqrt2);
            let e32 = 6.0 + sqrt2;
            t.ns = 3;
            t.gamma = g;
            t.elo = 2.0;
            t.ra[1][0] = 0.5 / g;
            t.ra[2][0] = 1.0 / g;
            t.ra[2][1] = 1.0 / g;
            t.rc[1][0] = -1.0 / g;
            t.rc[2][0] = -(e32 + 2.0) / g;
            t.rc[2][1] = -e32 / g;
            t.rd31 = 2.0;
            t.rd32 = e32;
            t.alpha = [0.0, 0.5, 1.0, 0.0, 0.0, 0.0];
            t.ros_m = [1.0 / g, 1.0 / g, 0.0, 0.0, 0.0, 0.0];
            t.ros_e = [
                -1.0 / (6.0 * g),
                -1.0 / (3.0 * g),
                1.0 / (6.0 * g),
                0.0,
                0.0,
                0.0,
            ];
        }
        IsolveSolver::Ros3p => {
            t.ns = 3;
            t.gamma = 0.788_675_134_594_812_86;
            t.elo = 3.0;
            t.ra[1][0] = 1.267_949_192_431_122_73;
            t.ra[2][0] = 1.267_949_192_431_122_73;
            t.rc[1][0] = -1.607_695_154_586_736_30;
            t.rc[2][0] = -3.464_101_615_137_754_69;
            t.rc[2][1] = -1.732_050_807_568_877_34;
            t.newf = [true, true, false, true, true, true];
            t.alpha = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
            t.ros_m = [
                2.0,
                0.577_350_269_189_625_78,
                0.422_649_730_810_374_24,
                0.0,
                0.0,
                0.0,
            ];
            t.ros_e = [
                t.ros_m[0] - 2.113_248_654_051_871_24,
                t.ros_m[1] - 1.0,
                t.ros_m[2] - 0.422_649_730_810_374_24,
                0.0,
                0.0,
                0.0,
            ];
        }
        IsolveSolver::Ros3pl => {
            t.ns = 4;
            t.gamma = 0.435_866_521_508_459;
            t.elo = 3.0;
            t.ra[1][0] = 1.147_140_180_139_521;
            t.ra[2][0] = 2.463_070_773_030_053;
            t.ra[2][1] = 1.147_140_180_139_521;
            t.ra[3][0] = 2.463_070_773_030_053;
            t.ra[3][1] = 1.147_140_180_139_521;
            t.rc[1][0] = -2.631_861_185_781_065;
            t.rc[2][0] = -1.302_364_158_113_095;
            t.rc[2][1] = 2.769_432_022_251_304;
            t.rc[3][0] = -1.552_568_958_732_4;
            t.rc[3][1] = 2.587_743_501_215_153;
            t.rc[3][2] = -1.416_993_298_352_02;
            t.newf = [true, true, true, false, true, true];
            t.alpha = [0.0, 0.5, 1.0, 1.0, 0.0, 0.0];
            t.ros_m = [
                2.463_070_773_030_053,
                1.147_140_180_139_521,
                0.0,
                1.0,
                0.0,
                0.0,
            ];
            t.ros_e = [
                2.463_070_773_030_053 - 2.346_947_683_513_665,
                1.147_140_180_139_521 - 0.456_530_569_451_895,
                -0.056_949_243_945_495,
                1.0 - 0.738_684_936_166_224,
                0.0,
                0.0,
            ];
        }
        IsolveSolver::Rodas3 => {
            t.ns = 4;
            t.gamma = 0.5;
            t.elo = 3.0;
            t.ra[2][0] = 2.0;
            t.ra[3][0] = 2.0;
            t.ra[3][2] = 1.0;
            t.rc[1][0] = 4.0;
            t.rc[2][0] = 1.0;
            t.rc[2][1] = -1.0;
            t.rc[3][0] = 1.0;
            t.rc[3][1] = -1.0;
            t.rc[3][2] = -8.0 / 3.0;
            t.newf = [true, false, true, true, true, true];
            t.alpha = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
            t.ros_m = [2.0, 0.0, 1.0, 1.0, 0.0, 0.0];
            t.ros_e = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        }
        IsolveSolver::Rodas4 => {
            t.ns = 6;
            t.gamma = 0.25;
            t.elo = 4.0;
            t.alpha = [0.0, 0.386, 0.21, 0.63, 1.0, 1.0];
            t.ra[1][0] = 1.544;
            t.ra[2][0] = 0.946_678_528_081_582_6;
            t.ra[2][1] = 0.255_701_169_898_328_4;
            t.ra[3][0] = 3.314_825_187_068_521;
            t.ra[3][1] = 2.896_124_015_972_201;
            t.ra[3][2] = 0.998_641_913_997_781_7;
            t.ra[4][0] = 1.221_224_509_226_641;
            t.ra[4][1] = 6.019_134_481_288_629;
            t.ra[4][2] = 12.537_083_329_320_87;
            t.ra[4][3] = -0.687_886_036_105_895;
            t.ra[5] =
                [t.ra[4][0], t.ra[4][1], t.ra[4][2], t.ra[4][3], 1.0, 0.0];
            t.rc[1][0] = -5.668_8;
            t.rc[2][0] = -2.430_093_356_833_875;
            t.rc[2][1] = -0.206_359_915_709_191_5;
            t.rc[3][0] = -0.107_352_905_815_137_5;
            t.rc[3][1] = -9.594_562_251_023_355;
            t.rc[3][2] = -20.470_286_148_096_16;
            t.rc[4][0] = 7.496_443_313_967_647;
            t.rc[4][1] = -10.246_804_314_643_52;
            t.rc[4][2] = -33.999_903_528_199_05;
            t.rc[4][3] = 11.708_908_932_061_6;
            t.rc[5] = [
                8.083_246_795_921_522,
                -7.981_132_988_064_893,
                -31.521_594_328_743_71,
                16.319_305_431_231_36,
                -6.058_818_238_834_054,
                0.0,
            ];
            t.ros_m =
                [t.ra[4][0], t.ra[4][1], t.ra[4][2], t.ra[4][3], 1.0, 1.0];
            t.ros_e = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        }
        IsolveSolver::Rodasp => {
            t.ns = 6;
            t.gamma = 0.25;
            t.elo = 4.0;
            t.alpha = [0.0, 0.75, 0.21, 0.63, 1.0, 1.0];
            t.ra[1][0] = 3.0;
            t.ra[2][0] = 1.831_036_793_486_759;
            t.ra[2][1] = 0.495_518_396_743_379_5;
            t.ra[3][0] = 2.304_376_582_692_669;
            t.ra[3][1] = -0.052_492_752_457_430_01;
            t.ra[3][2] = -1.176_798_761_832_782;
            t.ra[4][0] = -7.170_454_962_423_024;
            t.ra[4][1] = -4.741_636_671_481_785;
            t.ra[4][2] = -16.310_026_313_309_71;
            t.ra[4][3] = -1.062_004_044_111_401;
            t.ra[5] =
                [t.ra[4][0], t.ra[4][1], t.ra[4][2], t.ra[4][3], 1.0, 0.0];
            t.rc[1][0] = -12.0;
            t.rc[2][0] = -8.791_795_173_947_035;
            t.rc[2][1] = -2.207_865_586_973_518;
            t.rc[3][0] = 10.817_930_568_571_53;
            t.rc[3][1] = 6.780_270_611_428_266;
            t.rc[3][2] = 19.534_859_446_424_1;
            t.rc[4][0] = 34.190_950_067_496_76;
            t.rc[4][1] = 15.496_711_537_259_63;
            t.rc[4][2] = 54.747_608_759_641_3;
            t.rc[4][3] = 14.160_053_921_485_34;
            t.rc[5] = [
                34.626_058_309_305_32,
                15.300_849_761_144_73,
                56.999_555_786_626_67,
                18.408_070_097_930_95,
                -5.714_285_714_285_717,
                0.0,
            ];
            t.ros_m =
                [t.ra[4][0], t.ra[4][1], t.ra[4][2], t.ra[4][3], 1.0, 1.0];
            t.ros_e = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        }
    }
    t
}

/// Inputs to [`do_solve_diffusion`], mirroring `diffusion.f90:40-97`.
pub struct SolveDiffusionInput<'a> {
    /// Total number of zones (`nz`).
    pub nz: usize,
    /// Number of network species (`species`).
    pub species: usize,
    /// Number of ion diffusion classes (`nc`).
    pub nc: usize,
    /// Class index (`0..nc`) for each network species, length `species`.
    pub class: &'a [usize],
    /// Representative atomic mass number `Z + N` for each class, length `nc`.
    pub class_a: &'a [f64],
    /// Representative atomic number `Z` for each class, length `nc`.
    pub class_z: &'a [usize],
    /// Whether `h1` is present in the network (checked per `diffusion.f90:206`).
    pub has_h1: bool,
    /// Whether `he4` is present in the network (checked per `diffusion.f90:211`).
    pub has_he4: bool,
    /// Mean atomic mass number per zone (`abar`), length `nz`.
    pub abar: &'a [f64],
    /// Free electrons per nucleon (`free_e`), length `nz`.
    pub free_e: &'a [f64],
    /// Face control-volume masses `dm_bar` (`g`), length `nz`.
    pub dm_bar_in: &'a [f64],
    /// Cell masses `dm` (`g`), length `nz`.
    pub dm_in: &'a [f64],
    /// Cell temperatures `T` (`K`), length `nz`.
    pub t: &'a [f64],
    /// Cell `ln(T)`, length `nz`.
    pub ln_t: &'a [f64],
    /// Cell densities `rho` (`g cm^-3`), length `nz`.
    pub rho: &'a [f64],
    /// Cell `ln(rho)`, length `nz`.
    pub lnd: &'a [f64],
    /// Cell mid-radii `rmid` (`cm`), length `nz`.
    pub r_mid: &'a [f64],
    /// Face radii `r` (`cm`), length `nz`.
    pub r_face_in: &'a [f64],
    /// Face `d ln P / dm`, length `nz`.
    pub dln_p_dm_face: &'a [f64],
    /// Face `d ln T / dm`, length `nz`.
    pub dln_t_dm_face: &'a [f64],
    /// Face `d ln rho / dm`, length `nz`.
    pub dln_rho_dm_face: &'a [f64],
    /// Face gravity `s% grav`, length `nz`.
    pub grav: &'a [f64],
    /// Electron degeneracy `s% eta`, length `nz`.
    pub eta: &'a [f64],
    /// Plasma coupling `s% gam`, length `nz`.
    pub gamma: &'a [f64],
    /// EOS crystallization phase `s% phase` in `[0, 1]`, length `nz`.
    pub phase: &'a [f64],
    /// Inner boundary radius `R_center` (`cm`).
    pub r_center: f64,
    /// Full stellar timestep `s% dt` (`s`).
    pub star_dt: f64,
    /// Total diffusion integration interval (`s`).
    pub total_time: f64,
    /// Whether to use pure Coulomb (Iben & MacDonald) coefficients.
    pub pure_coulomb: bool,
    /// Whether to use Paquette ion-ion coefficients.
    pub diffusion_use_paquette: bool,
    /// Whether to use Caplan, Bauer & Freeman strong-coupling blend.
    pub diffusion_use_caplan: bool,
    /// Whether to use the CGS Burgers solver.
    pub diffusion_use_cgs_solver: bool,
    /// Lower `eta` bound for CGS thermal diffusion.
    pub cgs_thermal_diffusion_eta_full_on: f64,
    /// Upper `eta` bound for CGS thermal diffusion.
    pub cgs_thermal_diffusion_eta_full_off: f64,
    /// Substep sizing factor (`diffusion_dt_div_timescale`).
    pub dt_div_timescale: f64,
    /// Minimum number of substeps (`diffusion_min_num_substeps`).
    pub min_num_substeps: usize,
    /// Maximum substeps (`diffusion_steps_hard_limit`).
    pub max_steps: usize,
    /// Maximum total BE iterations (`diffusion_iters_hard_limit`).
    pub max_iters_total: usize,
    /// Maximum BE iterations per substep (`diffusion_max_iters_per_substep`).
    pub max_iters_per_substep: usize,
    /// Maximum BE retries per substep (`diffusion_max_retries_per_substep`).
    pub max_retries_per_substep: usize,
    /// Whether to calculate ionization charges via `get_typical_charge`.
    pub calculate_ionization: bool,
    /// Number of charge smoothing passes (`diffusion_nsmooth_typical_charge`).
    pub nsmooth_typical_charge: usize,
    /// Minimum temperature for radiative acceleration (`K`).
    pub min_t_for_radaccel: f64,
    /// Maximum temperature for radiative acceleration (`K`).
    pub max_t_for_radaccel: f64,
    /// Maximum advective velocity (`diffusion_v_max`, cm/s).
    pub v_advection_max: f64,
    /// Velocity limiter parameter (`diffusion_Vlimit`).
    pub diffusion_vlimit: f64,
    /// Outer mass full-on threshold (`Msun`) for `Vlimit`.
    pub diffusion_vlimit_dm_full_on: f64,
    /// Outer mass full-off threshold (`Msun`) for `Vlimit`.
    pub diffusion_vlimit_dm_full_off: f64,
    /// Outer mass full-on threshold (`Msun`) for artificial diffusion.
    pub diffusion_ad_dm_full_on: f64,
    /// Outer mass full-off threshold (`Msun`) for artificial diffusion.
    pub diffusion_ad_dm_full_off: f64,
    /// Artificial diffusion boost factor (`diffusion_AD_boost_factor`).
    pub diffusion_ad_boost_factor: f64,
    /// Concentration diffusion multiplier (`diffusion_SIG_factor`).
    pub diffusion_sig_factor: f64,
    /// Advection multiplier (`diffusion_GT_factor`).
    pub diffusion_gt_factor: f64,
    /// Upwind switch velocity threshold (`diffusion_upwind_abs_v_limit`).
    pub upwind_limit: f64,
    /// Coupling threshold where diffusion is full on (`diffusion_gamma_full_on`).
    pub gamma_full_on: f64,
    /// Coupling threshold where diffusion is full off (`diffusion_gamma_full_off`).
    pub gamma_full_off: f64,
    /// Temperature threshold where diffusion is full on (`diffusion_T_full_on`).
    pub t_full_on: f64,
    /// Temperature threshold where diffusion is full off (`diffusion_T_full_off`).
    pub t_full_off: f64,
    /// Hard minimum mass fraction (`diffusion_min_X_hard_limit`).
    pub min_x_hard_limit: f64,
    /// Global conservation absolute tolerance (`diffusion_X_total_atol`).
    pub x_total_atol: f64,
    /// Global conservation relative tolerance (`diffusion_X_total_rtol`).
    pub x_total_rtol: f64,
    /// Max-norm BE convergence tolerance (`diffusion_tol_correction_max`).
    pub tol_correction_max: f64,
    /// Average-norm BE convergence tolerance (`diffusion_tol_correction_norm`).
    pub tol_correction_norm: f64,
    /// Per-class diffusion scaling factors, length `nc`.
    pub diffusion_factor: &'a [f64],
    /// Whether `use_other_diffusion_factor` is active.
    pub use_other_diffusion_factor: bool,
    /// Optional per-zone extra diffusion factor, flat `(nz, nc)`.
    pub extra_diffusion_factor: Option<&'a [f64]>,
    /// Whether to integrate with the Rosenbrock `isolve` solver instead of BE.
    pub use_isolve: bool,
    /// Relative tolerance for `isolve`.
    pub rtol_for_isolve: f64,
    /// Absolute tolerance for `isolve`.
    pub atol_for_isolve: f64,
    /// Maximum steps for `isolve`.
    pub maxsteps_for_isolve: usize,
    /// Rosenbrock solver name (`diffusion_isolve_solver`).
    pub isolve_solver_name: &'a str,
    /// Initial outermost active zone (`0`-based `nzlo`).
    pub nzlo_in: usize,
    /// Initial innermost active zone (`0`-based `nzhi`).
    pub nzhi_in: usize,
}

/// Output buffers updated by [`do_solve_diffusion`], mirroring
/// `diffusion.f90:88-96`.
pub struct SolveDiffusionOutput<'a> {
    /// Typical charge per class and zone, flat `(nz, nc)`.
    pub typical_charge: &'a mut [f64],
    /// Species mass fractions `xa`, flat `(nz, species)`, updated in-place.
    pub xa: &'a mut [f64],
    /// Initial class mass fractions, flat `(nz, nc)`.
    pub x_init: &'a mut [f64],
    /// Final class mass fractions, flat `(nz, nc)`.
    pub x_final: &'a mut [f64],
    /// Face self-diffusion coefficients, flat `(nz, nc)`.
    pub d_self_face: &'a mut [f64],
    /// Face advective velocities, flat `(nz, nc)`.
    pub v_advection_face: &'a mut [f64],
    /// Face total velocities, flat `(nz, nc)`.
    pub v_total_face: &'a mut [f64],
    /// Face pressure/gravity velocities, flat `(nz, nc)`.
    pub vln_p_face: &'a mut [f64],
    /// Face thermal velocities, flat `(nz, nc)`.
    pub vln_t_face: &'a mut [f64],
    /// Face radiative velocities, flat `(nz, nc)`.
    pub v_rad_face: &'a mut [f64],
    /// Face radiative accelerations, flat `(nz, nc)`.
    pub g_rad_face: &'a mut [f64],
    /// Face electric field, length `nz`.
    pub e_field_face: &'a mut [f64],
    /// Face gravitational field, length `nz`.
    pub g_field_face: &'a mut [f64],
}

/// Execution summary returned by [`do_solve_diffusion`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolveDiffusionStats {
    /// Accepted substeps (`steps_used`).
    pub steps_used: usize,
    /// Total backward-Euler iterations (`total_num_iters`; `0` for `isolve`).
    pub total_num_iters: usize,
    /// Total substep retries (`total_num_retries`).
    pub total_num_retries: usize,
    /// Effective 0-based outermost active zone (`nzlo`).
    pub nzlo: usize,
    /// Effective 0-based innermost active zone (`nzhi`).
    pub nzhi: usize,
}

/// Integrates the coupled multi-species element diffusion equations over
/// `inp.total_time` (`diffusion.f90:40-1203`).
///
/// # Errors
///
/// Returns [`StarError::InvalidInput`] on invalid dimensions, missing `h1`/`he4`,
/// unknown `diffusion_isolve_solver`, or unsupported radiative levitation
/// (`op_mono`), and [`StarError::ConvergenceFailure`] if the substep solver
/// cannot complete `total_time`.
#[allow(clippy::too_many_lines, clippy::similar_names)]
pub fn do_solve_diffusion(
    inp: &SolveDiffusionInput<'_>,
    out: &mut SolveDiffusionOutput<'_>,
) -> Result<SolveDiffusionStats, StarError> {
    let nz = inp.nz;
    let species = inp.species;
    let nc = inp.nc;
    let m = nc + 1;

    if nc == 0 || nz < 2 {
        return Err(StarError::InvalidInput(
            "diffusion requires nc >= 1 and nz >= 2".into(),
        ));
    }
    if !inp.has_h1 {
        return Err(StarError::InvalidInput(
            "isos must include h1 for diffusion".into(),
        ));
    }
    if !inp.has_he4 {
        return Err(StarError::InvalidInput(
            "isos must include he4 for diffusion".into(),
        ));
    }

    let mut nzlo = inp.nzlo_in;
    let mut nzhi = inp.nzhi_in.min(nz - 1);

    // Reset nzlo if outer zones are cooler than T_full_off (diffusion.f90:217-223).
    let mut nbound = nzlo;
    for k in nzlo..=nzhi {
        if inp.t[k] > inp.t_full_off {
            nbound = k;
            break;
        }
    }
    nzlo = nbound;

    // `diffusion.f90:241-248` calls `get_limit_coeffs` unconditionally.
    let mut limit_coeffs_face = vec![0.0_f64; nz];
    nzhi = get_limit_coeffs(
        nzlo,
        nzhi,
        inp.gamma_full_on,
        inp.gamma_full_off,
        inp.t_full_on,
        inp.t_full_off,
        inp.gamma,
        inp.t,
        inp.phase,
        &mut limit_coeffs_face,
    );

    if nzhi <= nzlo {
        return Ok(SolveDiffusionStats {
            steps_used: 0,
            total_num_iters: 0,
            total_num_retries: 0,
            nzlo,
            nzhi,
        });
    }
    let n = nzhi - nzlo + 1;
    let neqs = n * nc;

    let mut alfa_face = vec![0.0_f64; nz];
    alfa_face[0] = 1.0;
    for k in 1..nz {
        alfa_face[k] = inp.dm_in[k - 1] / (inp.dm_in[k] + inp.dm_in[k - 1]);
    }

    let mut r_face = vec![0.0_f64; nz + 1];
    r_face[(nzlo + 1)..=nzhi]
        .copy_from_slice(&inp.r_face_in[(nzlo + 1)..=nzhi]);
    r_face[nzlo] = inp.r_face_in[0];
    r_face[nzhi + 1] = inp.r_center;

    let mut a = vec![0.0_f64; m];
    a[..nc].copy_from_slice(&inp.class_a[..nc]);
    a[nc] = ME / AMU;

    let mut cell_dm = vec![0.0_f64; nz];
    let mut dm_bar = vec![0.0_f64; nz];
    let mut x = vec![0.0_f64; nz * m];

    for k in 0..nz {
        cell_dm[k] = inp.dm_in[k];
        dm_bar[k] = inp.dm_bar_in[k];
        for j in 0..species {
            let cls = inp.class[j];
            x[k * m + cls] += TINY_X.max(out.xa[k * species + j]);
        }
        for i in 0..nc {
            if x[k * m + i] <= 0.0 {
                x[k * m + i] = TINY_X;
            }
        }
        let mut tmp = 0.0;
        for j in 0..nc {
            tmp += x[k * m + j];
        }
        for j in 0..nc {
            x[k * m + j] /= tmp;
            out.x_init[k * nc + j] = x[k * m + j];
        }
        x[k * m + nc] = 0.0;
    }

    // Merge fully mixed surface region 0..=nzlo into cell nzlo (diffusion.f90:317-331).
    if nzlo > 0 {
        let mut mtotal_surf = 0.0;
        for &dm_k in &cell_dm[..=nzlo] {
            mtotal_surf += dm_k;
        }
        for j in 0..species {
            let mut acc = 0.0;
            for k in 0..=nzlo {
                acc += cell_dm[k] * out.xa[k * species + j];
            }
            out.xa[nzlo * species + j] = acc / mtotal_surf;
        }
        for j in 0..nc {
            let mut acc = 0.0;
            for k in 0..=nzlo {
                acc += cell_dm[k] * x[k * m + j];
            }
            x[nzlo * m + j] = acc / mtotal_surf;
        }
        cell_dm[nzlo] = mtotal_surf;
        let mut sumx = 0.0;
        for j in 0..nc {
            sumx += x[nzlo * m + j];
        }
        for j in 0..nc {
            x[nzlo * m + j] /= sumx;
            out.x_init[nzlo * nc + j] = x[nzlo * m + j];
        }
    }
    dm_bar[nzlo + 1] = 0.5 * (cell_dm[nzlo] + cell_dm[nzlo + 1]);

    let mut rho_face = vec![0.0_f64; nz];
    let mut t_face = vec![0.0_f64; nz];
    let mut four_pi_r2_rho_face = vec![0.0_f64; nz];
    let mut dln_p_dr_face = vec![0.0_f64; nz];
    let mut dln_t_dr_face = vec![0.0_f64; nz];
    let mut dln_rho_dr_face = vec![0.0_f64; nz];
    let mut z = vec![0.0_f64; nz * m];
    let mut xm_face = vec![0.0_f64; nz];
    let mut rad_accel_face = vec![0.0_f64; nz * m];

    let _kmax_rad_accel = setup_struct_info(
        nzlo,
        nzhi,
        nc,
        m,
        inp.dln_p_dm_face,
        inp.dln_t_dm_face,
        inp.dln_rho_dm_face,
        &cell_dm,
        inp.abar,
        inp.free_e,
        inp.t,
        inp.ln_t,
        inp.rho,
        inp.lnd,
        &r_face,
        &alfa_face,
        inp.class_z,
        inp.calculate_ionization,
        inp.nsmooth_typical_charge,
        inp.max_t_for_radaccel,
        &mut rho_face,
        &mut t_face,
        &mut four_pi_r2_rho_face,
        &mut dln_p_dr_face,
        &mut dln_t_dr_face,
        &mut dln_rho_dr_face,
        &mut z,
        out.typical_charge,
        &mut xm_face,
        &mut rad_accel_face,
        out.g_rad_face,
    )?;

    let mut mtotal = 0.0;
    for &dm_k in &cell_dm[nzlo..=nzhi] {
        mtotal += dm_k;
    }
    let mut xtotal_init = vec![0.0_f64; nc];
    for j in 0..nc {
        let mut acc = 0.0;
        for k in nzlo..=nzhi {
            acc += cell_dm[k] * x[k * m + j];
        }
        xtotal_init[j] = acc / mtotal;
    }

    let mut sum_ending_mass = vec![0.0_f64; nz];
    let mut ending_mass = vec![0.0_f64; nz * nc];
    let mut ending_dx_dm = vec![0.0_f64; nz * nc];

    fixup(
        nz,
        nc,
        m,
        nzlo,
        nzhi,
        inp.min_x_hard_limit,
        inp.x_total_atol,
        inp.x_total_rtol,
        &cell_dm,
        mtotal,
        &xtotal_init,
        &mut x,
        inp.ln_t,
        &mut sum_ending_mass,
        &mut ending_mass,
        &mut ending_dx_dm,
    )
    .map_err(|diag| {
        StarError::ConvergenceFailure(format!(
            "initial diffusion fixup failed: {}",
            diag.reason
        ))
    })?;

    // Working buffers shared between matrix evaluation and solvers.
    let mut x_face = vec![0.0_f64; nz * m];
    let mut c = vec![0.0_f64; nz * m];
    let mut c_div_x = vec![0.0_f64; nz * m];
    let mut c_div_x_face = vec![0.0_f64; nz * m];
    let mut e_ap = vec![0.0_f64; nz];
    let mut e_at = vec![0.0_f64; nz];
    let mut e_ar = vec![0.0_f64; nz];
    let mut e_ax = vec![0.0_f64; nz * m];
    let mut g_ap = vec![0.0_f64; nz];
    let mut g_at = vec![0.0_f64; nz];
    let mut g_ar = vec![0.0_f64; nz];
    let mut g_ax = vec![0.0_f64; nz * m];
    let mut gt_face = vec![0.0_f64; nz * nc];
    let mut ad_face = vec![0.0_f64; nz];
    let mut sig_face = vec![0.0_f64; nz * nc * nc];
    let mut sigma_lnc = vec![0.0_f64; nz * nc * nc];
    let mut dx = vec![0.0_f64; nz * nc];
    let mut dx_dt = vec![0.0_f64; nz * nc];
    let mut del = vec![0.0_f64; nz * nc];
    let mut em1 = vec![0.0_f64; nz * nc * nc];
    let mut e00 = vec![0.0_f64; nz * nc * nc];
    let mut ep1 = vec![0.0_f64; nz * nc * nc];

    let coeff_inp = MatrixCoeffsInput {
        nz,
        nc,
        m,
        nzlo,
        nzhi,
        pure_coulomb: inp.pure_coulomb,
        diffusion_use_paquette: inp.diffusion_use_paquette,
        diffusion_use_caplan: inp.diffusion_use_caplan,
        diffusion_use_cgs_solver: inp.diffusion_use_cgs_solver,
        cgs_thermal_diffusion_eta_full_on: inp
            .cgs_thermal_diffusion_eta_full_on,
        cgs_thermal_diffusion_eta_full_off: inp
            .cgs_thermal_diffusion_eta_full_off,
        use_other_diffusion_factor: inp.use_other_diffusion_factor,
        star_dt: inp.star_dt,
        v_advection_max: inp.v_advection_max,
        diffusion_vlimit: inp.diffusion_vlimit,
        diffusion_vlimit_dm_full_on: inp.diffusion_vlimit_dm_full_on,
        diffusion_vlimit_dm_full_off: inp.diffusion_vlimit_dm_full_off,
        diffusion_ad_dm_full_on: inp.diffusion_ad_dm_full_on,
        diffusion_ad_dm_full_off: inp.diffusion_ad_dm_full_off,
        diffusion_ad_boost_factor: inp.diffusion_ad_boost_factor,
        diffusion_sig_factor: inp.diffusion_sig_factor,
        diffusion_gt_factor: inp.diffusion_gt_factor,
        min_t_for_radaccel: inp.min_t_for_radaccel,
        max_t_for_radaccel: inp.max_t_for_radaccel,
        diffusion_factor: inp.diffusion_factor,
        extra_diffusion_factor: inp.extra_diffusion_factor,
        a: &a,
        z: &z,
        rho_face: &rho_face,
        t_face: &t_face,
        four_pi_r2_rho_face: &four_pi_r2_rho_face,
        xm_face: &xm_face,
        cell_dm: &cell_dm,
        dm_bar: &dm_bar,
        dln_p_dr_face: &dln_p_dr_face,
        dln_t_dr_face: &dln_t_dr_face,
        dln_rho_dr_face: &dln_rho_dr_face,
        r_mid: inp.r_mid,
        grav: inp.grav,
        eta: inp.eta,
        gam: inp.gamma,
        gamma_t_limit_coeffs_face: &limit_coeffs_face,
        alfa_face: &alfa_face,
        rad_accel_face: &rad_accel_face,
    };

    let mut steps_used = 0_usize;
    let mut total_num_iters = 0_usize;
    let mut total_num_retries = 0_usize;
    let mut time = 0.0_f64;

    if inp.use_isolve {
        let solver = IsolveSolver::from_name(inp.isolve_solver_name)?;
        let tbl = ros_table(solver);
        let nmax = if inp.maxsteps_for_isolve == 0 {
            100_000
        } else {
            inp.maxsteps_for_isolve
        };
        let uround = 1.0e-16_f64;
        if inp.atol_for_isolve <= 0.0 || inp.rtol_for_isolve <= 10.0 * uround {
            return Err(StarError::InvalidInput(
                "diffusion_atol_for_isolve or diffusion_rtol_for_isolve too small"
                    .into(),
            ));
        }

        let fac1 = 5.0_f64;
        let fac2 = 1.0_f64 / 3.0_f64;
        let safe = 0.9_f64;
        let eloi = 1.0 / tbl.elo;
        let y_min = inp.min_x_hard_limit;
        let y_max = 1.0 - inp.min_x_hard_limit;

        let mut y = vec![0.0_f64; neqs];
        for i in 0..n {
            let k = i + nzlo;
            for j in 0..nc {
                y[i * nc + j] = x[k * m + j];
            }
        }

        let mut x_t = 0.0_f64;
        let xend = inp.total_time;
        let hmaxn = xend.abs();
        let mut h = xend.abs();
        if h <= 10.0 * uround {
            h = 1.0e-6;
        }
        h = h.min(hmaxn);
        let mut hopt = h;
        let mut hacc = h;
        let mut erracc = 1.0e-2_f64;
        let mut reject = false;
        let mut last = false;
        let mut nsing = 0_usize;
        let mut nstep = 0_usize;
        let mut naccpt = 0_usize;

        let mut dy1 = vec![0.0_f64; neqs];
        let mut dy = vec![0.0_f64; neqs];
        let mut dy2 = vec![0.0_f64; neqs];
        let mut ynew = vec![0.0_f64; neqs];
        let mut cont = vec![0.0_f64; neqs];
        let mut ak = vec![vec![0.0_f64; neqs]; tbl.ns];
        let mut uf_lblk = vec![0.0_f64; n * nc * nc];
        let mut uf_dblk = vec![0.0_f64; n * nc * nc];
        let mut uf_ublk = vec![0.0_f64; n * nc * nc];
        let mut lblk = vec![0.0_f64; n * nc * nc];
        let mut dblk = vec![0.0_f64; n * nc * nc];
        let mut ublk = vec![0.0_f64; n * nc * nc];
        let mut rhs_stage = vec![0.0_f64; neqs];

        let idid: i32;

        'outer_step: loop {
            if nstep > nmax {
                idid = -2;
                break 'outer_step;
            }
            let hprev = h;
            if 0.1 * h.abs() <= x_t.abs() * uround {
                idid = -3;
                break 'outer_step;
            }
            if last {
                idid = 1;
                break 'outer_step;
            }
            hopt = h;
            if (x_t + h * 1.0001 - xend) >= 0.0 {
                h = xend - x_t;
                last = true;
            }

            // Analytical block-tridiagonal Jacobian + f(y) (jac_blk_dble).
            for i in 0..n {
                let k = i + nzlo;
                for j in 0..nc {
                    x[k * m + j] = y[i * nc + j];
                }
            }
            if fixup(
                nz,
                nc,
                m,
                nzlo,
                nzhi,
                inp.min_x_hard_limit,
                inp.x_total_atol,
                inp.x_total_rtol,
                &cell_dm,
                mtotal,
                &xtotal_init,
                &mut x,
                inp.ln_t,
                &mut sum_ending_mass,
                &mut ending_mass,
                &mut ending_dx_dm,
            )
            .is_err()
            {
                idid = -5;
                break 'outer_step;
            }
            for i in 0..n {
                let k = i + nzlo;
                for j in 0..nc {
                    y[i * nc + j] = x[k * m + j];
                }
            }
            {
                let mut coeff_out = MatrixCoeffsOutput {
                    x: &mut x,
                    x_face: &mut x_face,
                    c: &mut c,
                    c_div_x: &mut c_div_x,
                    c_div_x_face: &mut c_div_x_face,
                    e_ap: &mut e_ap,
                    e_at: &mut e_at,
                    e_ar: &mut e_ar,
                    e_ax: &mut e_ax,
                    e_field_face: out.e_field_face,
                    g_ap: &mut g_ap,
                    g_at: &mut g_at,
                    g_ar: &mut g_ar,
                    g_ax: &mut g_ax,
                    g_field_face: out.g_field_face,
                    v_advection_face: out.v_advection_face,
                    v_total_face: out.v_total_face,
                    vln_p_face: out.vln_p_face,
                    vln_t_face: out.vln_t_face,
                    v_rad_face: out.v_rad_face,
                    gt_face: &mut gt_face,
                    d_self_face: out.d_self_face,
                    ad_face: &mut ad_face,
                    sig_face: &mut sig_face,
                    sigma_lnc: &mut sigma_lnc,
                };
                if get_matrix_coeffs(&coeff_inp, &mut coeff_out).is_err() {
                    idid = -5;
                    break 'outer_step;
                }
            }

            get_eqn_matrix_entries(
                nzlo,
                nzhi,
                m,
                nc,
                true,
                &x,
                &x_face,
                &c_div_x,
                &gt_face,
                &ad_face,
                &sig_face,
                &alfa_face,
                &cell_dm,
                out.v_advection_face,
                inp.upwind_limit,
                0.0,
                &dx,
                &mut dx_dt,
                &mut del,
                &mut em1,
                &mut e00,
                &mut ep1,
            );

            for i in 0..n {
                let k = i + nzlo;
                for row in 0..nc {
                    dy1[i * nc + row] = dx_dt[k * nc + row];
                    for col in 0..nc {
                        let dst = (i * nc + row) * nc + col;
                        let src = (k * nc + row) * nc + col;
                        uf_lblk[dst] = em1[src];
                        uf_dblk[dst] = e00[src];
                        uf_ublk[dst] = ep1[src];
                    }
                }
            }

            'retry_same_jac: loop {
                if h <= hprev * 1.0e-30 {
                    idid = -8;
                    break 'outer_step;
                }
                if h <= 0.0 || !h.is_finite() {
                    idid = -3;
                    break 'outer_step;
                }
                let fac_mat = 1.0 / (h * tbl.gamma);
                for idx in 0..(n * nc * nc) {
                    lblk[idx] = -uf_lblk[idx];
                    dblk[idx] = -uf_dblk[idx];
                    ublk[idx] = -uf_ublk[idx];
                }
                for i in 0..n {
                    for j in 0..nc {
                        dblk[(i * nc + j) * nc + j] += fac_mat;
                    }
                }

                // Execute Rosenbrock stages 0..tbl.ns.
                let mut stage_cut_half = false;
                let mut stage_cut_tenth = false;
                let mut stage_singular = false;

                for is in 0..tbl.ns {
                    if is == 0 {
                        dy.copy_from_slice(&dy1);
                        rhs_stage.copy_from_slice(&dy);
                    } else {
                        if tbl.newf[is] {
                            for j in 0..neqs {
                                let mut acc = 0.0;
                                for l in 0..is {
                                    acc += tbl.ra[is][l] * ak[l][j];
                                }
                                let yj = y[j] + acc;
                                if yj < y_min || yj > y_max || !yj.is_finite() {
                                    stage_cut_tenth = true;
                                    break;
                                }
                                ynew[j] = yj;
                            }
                            if stage_cut_tenth {
                                break;
                            }

                            // fcn_blk_dble on ynew
                            for i in 0..n {
                                let k = i + nzlo;
                                for j in 0..nc {
                                    x[k * m + j] = ynew[i * nc + j];
                                }
                            }
                            if fixup(
                                nz,
                                nc,
                                m,
                                nzlo,
                                nzhi,
                                inp.min_x_hard_limit,
                                inp.x_total_atol,
                                inp.x_total_rtol,
                                &cell_dm,
                                mtotal,
                                &xtotal_init,
                                &mut x,
                                inp.ln_t,
                                &mut sum_ending_mass,
                                &mut ending_mass,
                                &mut ending_dx_dm,
                            )
                            .is_err()
                            {
                                stage_cut_half = true;
                                break;
                            }
                            for i in 0..n {
                                let k = i + nzlo;
                                for j in 0..nc {
                                    ynew[i * nc + j] = x[k * m + j];
                                }
                            }
                            {
                                let mut coeff_out = MatrixCoeffsOutput {
                                    x: &mut x,
                                    x_face: &mut x_face,
                                    c: &mut c,
                                    c_div_x: &mut c_div_x,
                                    c_div_x_face: &mut c_div_x_face,
                                    e_ap: &mut e_ap,
                                    e_at: &mut e_at,
                                    e_ar: &mut e_ar,
                                    e_ax: &mut e_ax,
                                    e_field_face: out.e_field_face,
                                    g_ap: &mut g_ap,
                                    g_at: &mut g_at,
                                    g_ar: &mut g_ar,
                                    g_ax: &mut g_ax,
                                    g_field_face: out.g_field_face,
                                    v_advection_face: out.v_advection_face,
                                    v_total_face: out.v_total_face,
                                    vln_p_face: out.vln_p_face,
                                    vln_t_face: out.vln_t_face,
                                    v_rad_face: out.v_rad_face,
                                    gt_face: &mut gt_face,
                                    d_self_face: out.d_self_face,
                                    ad_face: &mut ad_face,
                                    sig_face: &mut sig_face,
                                    sigma_lnc: &mut sigma_lnc,
                                };
                                if get_matrix_coeffs(&coeff_inp, &mut coeff_out)
                                    .is_err()
                                {
                                    stage_cut_half = true;
                                    break;
                                }
                            }
                            get_dx_dt(
                                nzlo,
                                nzhi,
                                m,
                                nc,
                                &x,
                                &x_face,
                                &c_div_x,
                                &gt_face,
                                &ad_face,
                                &sig_face,
                                &cell_dm,
                                out.v_advection_face,
                                inp.upwind_limit,
                                &mut dx_dt,
                            );
                            for i in 0..n {
                                let k = i + nzlo;
                                for j in 0..nc {
                                    dy[i * nc + j] = dx_dt[k * nc + j];
                                }
                            }
                            if tbl.rd32 != 0.0 {
                                dy2.copy_from_slice(&dy);
                            }
                        }

                        for j in 0..neqs {
                            let mut acc = 0.0;
                            for l in 0..is {
                                acc += (tbl.rc[is][l] / h) * ak[l][j];
                            }
                            cont[j] = acc;
                        }
                        if is == 2 && tbl.rd32 != 0.0 {
                            for j in 0..neqs {
                                cont[j] +=
                                    tbl.rd32 * dy2[j] + tbl.rd31 * dy1[j];
                            }
                        }
                        for j in 0..neqs {
                            rhs_stage[j] = dy[j] + cont[j];
                        }
                    }

                    if let Ok(sol) = solve_block_tridiag_bcyclic(
                        nc, n, &lblk, &dblk, &ublk, &rhs_stage, false,
                    ) {
                        ak[is].copy_from_slice(&sol);
                    } else {
                        if is == 0 {
                            stage_singular = true;
                        } else {
                            stage_cut_half = true;
                        }
                        break;
                    }
                }

                if stage_singular {
                    nsing += 1;
                    if nsing >= 5 {
                        idid = -4;
                        break 'outer_step;
                    }
                    h *= 0.5;
                    reject = true;
                    last = false;
                    continue 'retry_same_jac;
                }
                if stage_cut_tenth {
                    h *= 0.1;
                    reject = true;
                    last = false;
                    continue 'retry_same_jac;
                }
                if stage_cut_half {
                    h *= 0.5;
                    reject = true;
                    last = false;
                    continue 'retry_same_jac;
                }

                // Assemble trial solution ynew (mod_rosenbrock.f:997-1003).
                let mut out_of_range = false;
                for j in 0..neqs {
                    let mut acc = 0.0;
                    for l in 0..tbl.ns {
                        acc += tbl.ros_m[l] * ak[l][j];
                    }
                    let yj = y[j] + acc;
                    if yj < y_min || yj > y_max || !yj.is_finite() {
                        out_of_range = true;
                        break;
                    }
                    ynew[j] = yj;
                }
                if out_of_range {
                    h *= 0.1;
                    reject = true;
                    last = false;
                    continue 'retry_same_jac;
                }

                nstep += 1;
                let mut err_sq = 0.0;
                for i in 0..neqs {
                    let mut ei = 0.0;
                    for l in 0..tbl.ns {
                        ei += tbl.ros_e[l] * ak[l][i];
                    }
                    let sk = inp.atol_for_isolve
                        + inp.rtol_for_isolve * y[i].abs().max(ynew[i].abs());
                    let ratio = ei / sk;
                    err_sq += ratio * ratio;
                }
                let err = (err_sq / (neqs as f64)).sqrt();
                if !err.is_finite() {
                    h *= 0.5;
                    reject = true;
                    last = false;
                    continue 'retry_same_jac;
                }

                let mut fac =
                    fac2.max(fac1.min(flexmesa_math::pow(err, eloi) / safe));
                let mut hnew = h / fac;
                if !hnew.is_finite() {
                    h *= 0.5;
                    reject = true;
                    last = false;
                    continue 'retry_same_jac;
                }
                hopt = hnew;

                if err <= 1.0 {
                    naccpt += 1;
                    if naccpt > 1 {
                        let facgus_raw = (hacc / h)
                            * flexmesa_math::pow(err * err / erracc, eloi)
                            / safe;
                        let facgus = fac2.max(fac1.min(facgus_raw));
                        fac = fac.max(facgus);
                        hnew = h / fac;
                    }
                    hacc = h;
                    erracc = 1.0e-2_f64.max(err);
                    y.copy_from_slice(&ynew);
                    x_t += h;
                    if hnew.abs() > hmaxn {
                        hnew = hmaxn;
                    }
                    if reject {
                        hnew = hnew.abs().min(h.abs());
                    }
                    reject = false;
                    h = hnew;
                    continue 'outer_step;
                }

                reject = true;
                last = false;
                h = hnew;
            }
        }

        let _ = hopt;
        if idid != 1 {
            return Err(StarError::ConvergenceFailure(format!(
                "diffusion isolve ({}) failed with idid={idid} after {naccpt} accepted / {nstep} total steps",
                inp.isolve_solver_name
            )));
        }

        steps_used = naccpt;
        total_num_retries = nstep.saturating_sub(naccpt);
        time = xend;

        for i in 0..n {
            let k = i + nzlo;
            for j in 0..nc {
                x[k * m + j] = y[i * nc + j];
            }
        }
    } else {
        // Backward-Euler adaptive substep loop (diffusion.f90:850-1100).
        const MIN_LAMBDA: f64 = 1.0e-2;
        const DT_RETRY_FACTOR: f64 = 0.5;

        let min_dt = if inp.max_steps > 0 {
            inp.total_time / (1000.0 * (inp.max_steps as f64))
        } else {
            inp.total_time
        };
        let mut x_0 = vec![0.0_f64; nz * nc];
        let mut x_1 = vec![0.0_f64; nz * nc];
        let mut have_changed_matrix_coeffs = false;

        'step_loop: while inp.total_time - time > 1.0e-10 * inp.total_time
            && steps_used < inp.max_steps
        {
            steps_used += 1;

            {
                let mut coeff_out = MatrixCoeffsOutput {
                    x: &mut x,
                    x_face: &mut x_face,
                    c: &mut c,
                    c_div_x: &mut c_div_x,
                    c_div_x_face: &mut c_div_x_face,
                    e_ap: &mut e_ap,
                    e_at: &mut e_at,
                    e_ar: &mut e_ar,
                    e_ax: &mut e_ax,
                    e_field_face: out.e_field_face,
                    g_ap: &mut g_ap,
                    g_at: &mut g_at,
                    g_ar: &mut g_ar,
                    g_ax: &mut g_ax,
                    g_field_face: out.g_field_face,
                    v_advection_face: out.v_advection_face,
                    v_total_face: out.v_total_face,
                    vln_p_face: out.vln_p_face,
                    vln_t_face: out.vln_t_face,
                    v_rad_face: out.v_rad_face,
                    gt_face: &mut gt_face,
                    d_self_face: out.d_self_face,
                    ad_face: &mut ad_face,
                    sig_face: &mut sig_face,
                    sigma_lnc: &mut sigma_lnc,
                };
                get_matrix_coeffs(&coeff_inp, &mut coeff_out)?;
            }

            let remaining_time = inp.total_time - time;
            let timescale = get_timescale(
                nzlo,
                nzhi,
                m,
                nc,
                -inp.min_x_hard_limit * 0.5,
                out.v_advection_face,
                inp.upwind_limit,
                &x,
                &x_face,
                &c_div_x,
                &sig_face,
                &gt_face,
                &ad_face,
                &cell_dm,
            );

            let mut dt =
                (inp.dt_div_timescale * timescale).max(1.0e-4 * remaining_time);
            if inp.min_num_substeps > 0 {
                dt = dt.min(inp.total_time / (inp.min_num_substeps as f64));
            }
            if dt >= remaining_time {
                dt = remaining_time;
            } else if dt > 0.5 * remaining_time {
                dt = 0.5 * remaining_time;
            }

            let max_retries = inp.max_retries_per_substep;
            let mut substep_succeeded = false;

            for retry_count in 1..=max_retries {
                if dt < min_dt {
                    break 'step_loop;
                }
                if retry_count == 1 {
                    for k in nzlo..=nzhi {
                        for j in 0..nc {
                            x_0[k * nc + j] = x[k * m + j];
                            x_1[k * nc + j] = x[k * m + j];
                            dx[k * nc + j] = 0.0;
                        }
                    }
                } else {
                    dt *= DT_RETRY_FACTOR;
                    total_num_retries += 1;
                    for k in nzlo..=nzhi {
                        for j in 0..nc {
                            x[k * m + j] = x_0[k * nc + j];
                            x_1[k * nc + j] = x_0[k * nc + j];
                            dx[k * nc + j] = 0.0;
                        }
                    }
                    if have_changed_matrix_coeffs {
                        let mut coeff_out = MatrixCoeffsOutput {
                            x: &mut x,
                            x_face: &mut x_face,
                            c: &mut c,
                            c_div_x: &mut c_div_x,
                            c_div_x_face: &mut c_div_x_face,
                            e_ap: &mut e_ap,
                            e_at: &mut e_at,
                            e_ar: &mut e_ar,
                            e_ax: &mut e_ax,
                            e_field_face: out.e_field_face,
                            g_ap: &mut g_ap,
                            g_at: &mut g_at,
                            g_ar: &mut g_ar,
                            g_ax: &mut g_ax,
                            g_field_face: out.g_field_face,
                            v_advection_face: out.v_advection_face,
                            v_total_face: out.v_total_face,
                            vln_p_face: out.vln_p_face,
                            vln_t_face: out.vln_t_face,
                            v_rad_face: out.v_rad_face,
                            gt_face: &mut gt_face,
                            d_self_face: out.d_self_face,
                            ad_face: &mut ad_face,
                            sig_face: &mut sig_face,
                            sigma_lnc: &mut sigma_lnc,
                        };
                        get_matrix_coeffs(&coeff_inp, &mut coeff_out)?;
                    }
                }
                have_changed_matrix_coeffs = false;

                // BE_solve(theta = 1.0) (diffusion.f90:977-1099).
                let mut be_converged = false;
                for num_iters in 1..=inp.max_iters_per_substep {
                    if total_num_iters >= inp.max_iters_total {
                        return Err(StarError::ConvergenceFailure(format!(
                            "element diffusion exceeded diffusion_iters_hard_limit={}",
                            inp.max_iters_total
                        )));
                    }
                    total_num_iters += 1;

                    get_eqn_matrix_entries(
                        nzlo,
                        nzhi,
                        m,
                        nc,
                        false,
                        &x,
                        &x_face,
                        &c_div_x,
                        &gt_face,
                        &ad_face,
                        &sig_face,
                        &alfa_face,
                        &cell_dm,
                        out.v_advection_face,
                        inp.upwind_limit,
                        dt,
                        &dx,
                        &mut dx_dt,
                        &mut del,
                        &mut em1,
                        &mut e00,
                        &mut ep1,
                    );

                    let blk_lo = nzlo * nc * nc;
                    let blk_hi = (nzhi + 1) * nc * nc;
                    let vec_lo = nzlo * nc;
                    let vec_hi = (nzhi + 1) * nc;
                    let Ok(sol) = solve_block_tridiag_bcyclic(
                        nc,
                        n,
                        &em1[blk_lo..blk_hi],
                        &e00[blk_lo..blk_hi],
                        &ep1[blk_lo..blk_hi],
                        &del[vec_lo..vec_hi],
                        false,
                    ) else {
                        break; // retry with smaller dt
                    };
                    del[vec_lo..vec_hi].copy_from_slice(&sol);

                    let mut lambda = positivity(
                        nc,
                        nzlo,
                        nzhi,
                        &x_1,
                        -inp.min_x_hard_limit,
                        &del,
                    );
                    if lambda < MIN_LAMBDA {
                        lambda = MIN_LAMBDA;
                    }

                    for k in nzlo..=nzhi {
                        for j in 0..nc {
                            dx[k * nc + j] += lambda * del[k * nc + j];
                            x_1[k * nc + j] = x_0[k * nc + j] + dx[k * nc + j];
                            x[k * m + j] = x_1[k * nc + j];
                        }
                    }

                    if fixup(
                        nz,
                        nc,
                        m,
                        nzlo,
                        nzhi,
                        inp.min_x_hard_limit,
                        inp.x_total_atol,
                        inp.x_total_rtol,
                        &cell_dm,
                        mtotal,
                        &xtotal_init,
                        &mut x,
                        inp.ln_t,
                        &mut sum_ending_mass,
                        &mut ending_mass,
                        &mut ending_dx_dm,
                    )
                    .is_err()
                    {
                        break; // retry with smaller dt
                    }

                    if lambda == 1.0 {
                        let mut max_del = 0.0_f64;
                        let mut sum_abs_del = 0.0_f64;
                        for &d in &del[vec_lo..vec_hi] {
                            let ad = d.abs();
                            if ad > max_del {
                                max_del = ad;
                            }
                            sum_abs_del += ad;
                        }
                        let avg_del = sum_abs_del / (neqs as f64);
                        if max_del <= inp.tol_correction_max
                            && avg_del <= inp.tol_correction_norm
                        {
                            be_converged = true;
                            break;
                        }
                    }

                    if num_iters == inp.max_iters_per_substep {
                        break;
                    }

                    {
                        let mut coeff_out = MatrixCoeffsOutput {
                            x: &mut x,
                            x_face: &mut x_face,
                            c: &mut c,
                            c_div_x: &mut c_div_x,
                            c_div_x_face: &mut c_div_x_face,
                            e_ap: &mut e_ap,
                            e_at: &mut e_at,
                            e_ar: &mut e_ar,
                            e_ax: &mut e_ax,
                            e_field_face: out.e_field_face,
                            g_ap: &mut g_ap,
                            g_at: &mut g_at,
                            g_ar: &mut g_ar,
                            g_ax: &mut g_ax,
                            g_field_face: out.g_field_face,
                            v_advection_face: out.v_advection_face,
                            v_total_face: out.v_total_face,
                            vln_p_face: out.vln_p_face,
                            vln_t_face: out.vln_t_face,
                            v_rad_face: out.v_rad_face,
                            gt_face: &mut gt_face,
                            d_self_face: out.d_self_face,
                            ad_face: &mut ad_face,
                            sig_face: &mut sig_face,
                            sigma_lnc: &mut sigma_lnc,
                        };
                        get_matrix_coeffs(&coeff_inp, &mut coeff_out)?;
                    }
                    have_changed_matrix_coeffs = true;
                }

                if be_converged {
                    time += dt;
                    substep_succeeded = true;
                    break;
                }
            }

            if !substep_succeeded {
                break 'step_loop;
            }
        }
    }

    if inp.total_time - time > 1.0e-10 * inp.total_time || steps_used == 0 {
        return Err(StarError::ConvergenceFailure(format!(
            "do_solve_diffusion failed to complete total_time={:.6e} s \
             (reached {:.6e} s in {steps_used} steps, {total_num_iters} iters)",
            inp.total_time, time
        )));
    }

    set_new_xa(
        nzlo, nzhi, species, nc, m, inp.class, out.x_init, &x, out.xa,
    )?;

    for k in nzlo..=nzhi {
        for j in 0..nc {
            out.x_final[k * nc + j] = x[k * m + j];
        }
    }
    for k in 0..nzlo {
        for i in 0..species {
            out.xa[k * species + i] = out.xa[nzlo * species + i];
        }
        for j in 0..nc {
            out.x_final[k * nc + j] = x[nzlo * m + j];
        }
    }

    Ok(SolveDiffusionStats {
        steps_used,
        total_num_iters,
        total_num_retries,
        nzlo,
        nzhi,
    })
}

/// Estimates the explicit depletion timescale across interior zones
/// `(nzlo + 1)..nzhi` (`diffusion.f90:1206-1424`).
#[allow(clippy::too_many_arguments, clippy::similar_names)]
#[must_use]
pub fn get_timescale(
    nzlo: usize,
    nzhi: usize,
    m: usize,
    nc: usize,
    eps: f64,
    v_advection_face: &[f64],
    upwind_limit: f64,
    x: &[f64],
    x_face: &[f64],
    c_div_x: &[f64],
    sig_face: &[f64],
    gt_face: &[f64],
    ad_face: &[f64],
    cell_dm: &[f64],
) -> f64 {
    let mut dt = 1.0e99_f64;
    if nzhi < nzlo + 2 {
        return dt;
    }

    for k in (nzlo + 1)..nzhi {
        for i in 0..nc {
            if x[k * m + i] < eps {
                continue;
            }
            let gt_k = gt_face[k * nc + i];
            let mut flow_out =
                if v_advection_face[k * nc + i].abs() >= upwind_limit {
                    if gt_k > 0.0 {
                        gt_k * x[k * m + i]
                    } else {
                        gt_k * x[(k - 1) * m + i]
                    }
                } else {
                    gt_k * x_face[k * m + i]
                };

            let gt_kp1 = gt_face[(k + 1) * nc + i];
            let mut flow_in =
                if v_advection_face[(k + 1) * nc + i].abs() >= upwind_limit {
                    if gt_kp1 > 0.0 {
                        gt_kp1 * x[(k + 1) * m + i]
                    } else {
                        gt_kp1 * x[k * m + i]
                    }
                } else {
                    gt_kp1 * x_face[(k + 1) * m + i]
                };

            for j in 0..nc {
                let mut coeff = sig_face[((k + 1) * nc + i) * nc + j]
                    * x_face[(k + 1) * m + i]
                    / SMALL_X.max(x_face[(k + 1) * m + j]);
                if j == i {
                    coeff += ad_face[k + 1];
                }
                let f_in = coeff
                    * (c_div_x[(k + 1) * m + j] * x[(k + 1) * m + j]
                        - c_div_x[k * m + j] * x[k * m + j]);
                flow_in += f_in;

                let mut coeff_o = sig_face[(k * nc + i) * nc + j]
                    * x_face[k * m + i]
                    / SMALL_X.max(x_face[k * m + j]);
                if j == i {
                    coeff_o += ad_face[k];
                }
                let f_out = coeff_o
                    * (c_div_x[k * m + j] * x[k * m + j]
                        - c_div_x[(k - 1) * m + j] * x[(k - 1) * m + j]);
                flow_out += f_out;
            }

            let dxdt = (flow_in - flow_out) / cell_dm[k];
            if dxdt >= 0.0 {
                continue;
            }
            let dt1 = (x[k * m + i] + eps) / 1.0e-50_f64.max(-dxdt);
            if dt1 < dt {
                dt = dt1;
            }
        }
    }

    dt
}

/// Evaluates `dX/dt` across active zones `nzlo..=nzhi`
/// (`diffusion.f90:1427-1446`).
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn get_dx_dt(
    nzlo: usize,
    nzhi: usize,
    m: usize,
    nc: usize,
    x: &[f64],
    x_face: &[f64],
    c_div_x: &[f64],
    gt_face: &[f64],
    ad_face: &[f64],
    sig_face: &[f64],
    cell_dm: &[f64],
    v_advection_face: &[f64],
    upwind_limit: f64,
    dx_dt: &mut [f64],
) {
    for k in nzlo..=nzhi {
        get1_dx_dt(
            k,
            nzlo,
            nzhi,
            m,
            nc,
            x,
            x_face,
            c_div_x,
            gt_face,
            ad_face,
            sig_face,
            cell_dm,
            v_advection_face,
            upwind_limit,
            &mut dx_dt[k * nc..(k + 1) * nc],
        );
    }
}

/// Evaluates `dX/dt` in zone `k` (`diffusion.f90:1449-1534`).
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn get1_dx_dt(
    k: usize,
    nzlo: usize,
    nzhi: usize,
    m: usize,
    nc: usize,
    x: &[f64],
    x_face: &[f64],
    c_div_x: &[f64],
    gt_face: &[f64],
    ad_face: &[f64],
    sig_face: &[f64],
    cell_dm: &[f64],
    v_advection_face: &[f64],
    upwind_limit: f64,
    dx_dt_k: &mut [f64],
) {
    dx_dt_k[..nc].fill(0.0);

    if k > nzlo {
        for i in 0..nc {
            let c = gt_face[k * nc + i];
            if v_advection_face[k * nc + i].abs() >= upwind_limit {
                if c > 0.0 {
                    dx_dt_k[i] -= x[k * m + i] * c;
                } else {
                    dx_dt_k[i] -= x[(k - 1) * m + i] * c;
                }
            } else {
                dx_dt_k[i] -= x_face[k * m + i] * c;
            }
            for j in 0..nc {
                let c_sig = sig_face[(k * nc + i) * nc + j]
                    / SMALL_X.max(x_face[k * m + j]);
                let mut coeff = x_face[k * m + i] * c_sig;
                if j == i {
                    coeff += ad_face[k];
                }
                let dc = c_div_x[k * m + j] * x[k * m + j]
                    - c_div_x[(k - 1) * m + j] * x[(k - 1) * m + j];
                dx_dt_k[i] -= coeff * dc;
            }
        }
    }

    if k < nzhi {
        for i in 0..nc {
            let c = gt_face[(k + 1) * nc + i];
            if v_advection_face[(k + 1) * nc + i].abs() >= upwind_limit {
                if c > 0.0 {
                    dx_dt_k[i] += x[(k + 1) * m + i] * c;
                } else {
                    dx_dt_k[i] += x[k * m + i] * c;
                }
            } else {
                dx_dt_k[i] += x_face[(k + 1) * m + i] * c;
            }
            for j in 0..nc {
                let c_sig = sig_face[((k + 1) * nc + i) * nc + j]
                    / SMALL_X.max(x_face[(k + 1) * m + j]);
                let mut coeff = x_face[(k + 1) * m + i] * c_sig;
                if j == i {
                    coeff += ad_face[k + 1];
                }
                let dc = c_div_x[(k + 1) * m + j] * x[(k + 1) * m + j]
                    - c_div_x[k * m + j] * x[k * m + j];
                dx_dt_k[i] += coeff * dc;
            }
        }
    }

    let dm_k = cell_dm[k];
    for val in &mut dx_dt_k[..nc] {
        *val /= dm_k;
    }
}

/// Assembles `dX/dt`, RHS, and block-tridiagonal Jacobian/Newton blocks
/// `(em1, e00, ep1)` across `nzlo..=nzhi` (`diffusion.f90:1537-1567`).
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn get_eqn_matrix_entries(
    nzlo: usize,
    nzhi: usize,
    m: usize,
    nc: usize,
    jac_only: bool,
    x: &[f64],
    x_face: &[f64],
    c_div_x: &[f64],
    gt_face: &[f64],
    ad_face: &[f64],
    sig_face: &[f64],
    alfa_face: &[f64],
    cell_dm: &[f64],
    v_advection_face: &[f64],
    upwind_limit: f64,
    dt: f64,
    dx: &[f64],
    dx_dt: &mut [f64],
    rhs: &mut [f64],
    em1: &mut [f64],
    e00: &mut [f64],
    ep1: &mut [f64],
) {
    for k in nzlo..=nzhi {
        get1_eqn_matrix_entries(
            k,
            nzlo,
            nzhi,
            m,
            nc,
            jac_only,
            x,
            x_face,
            c_div_x,
            gt_face,
            ad_face,
            sig_face,
            alfa_face,
            cell_dm,
            v_advection_face,
            upwind_limit,
            dt,
            &dx[k * nc..(k + 1) * nc],
            &mut dx_dt[k * nc..(k + 1) * nc],
            &mut rhs[k * nc..(k + 1) * nc],
            &mut em1[k * nc * nc..(k + 1) * nc * nc],
            &mut e00[k * nc * nc..(k + 1) * nc * nc],
            &mut ep1[k * nc * nc..(k + 1) * nc * nc],
        );
    }
}

/// Assembles `dX/dt`, RHS, and row-major `(nc, nc)` blocks `(em1, e00, ep1)`
/// for zone `k` (`diffusion.f90:1570-1723`).
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn get1_eqn_matrix_entries(
    k: usize,
    nzlo: usize,
    nzhi: usize,
    m: usize,
    nc: usize,
    jac_only: bool,
    x: &[f64],
    x_face: &[f64],
    c_div_x: &[f64],
    gt_face: &[f64],
    ad_face: &[f64],
    sig_face: &[f64],
    alfa_face: &[f64],
    cell_dm: &[f64],
    v_advection_face: &[f64],
    upwind_limit: f64,
    dt: f64,
    dx_k: &[f64],
    dx_dt_k: &mut [f64],
    rhs_k: &mut [f64],
    em1_k: &mut [f64],
    e00_k: &mut [f64],
    ep1_k: &mut [f64],
) {
    dx_dt_k[..nc].fill(0.0);
    em1_k[..(nc * nc)].fill(0.0);
    e00_k[..(nc * nc)].fill(0.0);
    ep1_k[..(nc * nc)].fill(0.0);

    if k > nzlo {
        let alfa = alfa_face[k];
        let beta = 1.0 - alfa;
        for i in 0..nc {
            let c = gt_face[k * nc + i];
            if v_advection_face[k * nc + i].abs() >= upwind_limit {
                if c > 0.0 {
                    dx_dt_k[i] -= x[k * m + i] * c;
                    e00_k[i * nc + i] -= c;
                } else {
                    dx_dt_k[i] -= x[(k - 1) * m + i] * c;
                    em1_k[i * nc + i] -= c;
                }
            } else {
                dx_dt_k[i] -= x_face[k * m + i] * c;
                em1_k[i * nc + i] -= beta * c;
                e00_k[i * nc + i] -= alfa * c;
            }

            for j in 0..nc {
                let c_sig = sig_face[(k * nc + i) * nc + j]
                    / SMALL_X.max(x_face[k * m + j]);
                let mut coeff = x_face[k * m + i] * c_sig;
                if j == i {
                    coeff += ad_face[k];
                }
                let dc = c_div_x[k * m + j] * x[k * m + j]
                    - c_div_x[(k - 1) * m + j] * x[(k - 1) * m + j];
                dx_dt_k[i] -= coeff * dc;

                if j == i {
                    em1_k[i * nc + j] -= beta * c_sig * dc;
                    e00_k[i * nc + j] -= alfa * c_sig * dc;
                }

                let dc_dxjm1 = -c_div_x[(k - 1) * m + j];
                let dc_dxj00 = c_div_x[k * m + j];
                em1_k[i * nc + j] -= coeff * dc_dxjm1;
                e00_k[i * nc + j] -= coeff * dc_dxj00;

                if USE_DCOEFF_DX && x_face[k * m + j] > SMALL_X {
                    let dcoeff_dxjm1 = -beta * coeff / x_face[k * m + j];
                    let dcoeff_dxj00 = -alfa * coeff / x_face[k * m + j];
                    em1_k[i * nc + j] -= dcoeff_dxjm1 * dc;
                    e00_k[i * nc + j] -= dcoeff_dxj00 * dc;
                }
            }
        }
    }

    if k < nzhi {
        let alfa = alfa_face[k + 1];
        let beta = 1.0 - alfa;
        for i in 0..nc {
            let c = gt_face[(k + 1) * nc + i];
            if v_advection_face[(k + 1) * nc + i].abs() >= upwind_limit {
                if c > 0.0 {
                    dx_dt_k[i] += x[(k + 1) * m + i] * c;
                    ep1_k[i * nc + i] += c;
                } else {
                    dx_dt_k[i] += x[k * m + i] * c;
                    e00_k[i * nc + i] += c;
                }
            } else {
                dx_dt_k[i] += x_face[(k + 1) * m + i] * c;
                e00_k[i * nc + i] += beta * c;
                ep1_k[i * nc + i] += alfa * c;
            }

            for j in 0..nc {
                let c_sig = sig_face[((k + 1) * nc + i) * nc + j]
                    / SMALL_X.max(x_face[(k + 1) * m + j]);
                let mut coeff = x_face[(k + 1) * m + i] * c_sig;
                if j == i {
                    coeff += ad_face[k + 1];
                }
                let dc = c_div_x[(k + 1) * m + j] * x[(k + 1) * m + j]
                    - c_div_x[k * m + j] * x[k * m + j];
                dx_dt_k[i] += coeff * dc;

                if j == i {
                    e00_k[i * nc + j] += beta * c_sig * dc;
                    ep1_k[i * nc + j] += alfa * c_sig * dc;
                }

                let dc_dxj00 = -c_div_x[k * m + j];
                let dc_dxjp1 = c_div_x[(k + 1) * m + j];
                e00_k[i * nc + j] += coeff * dc_dxj00;
                ep1_k[i * nc + j] += coeff * dc_dxjp1;

                if USE_DCOEFF_DX && x_face[(k + 1) * m + j] > SMALL_X {
                    let dcoeff_dxj00 = -beta * coeff / x_face[(k + 1) * m + j];
                    let dcoeff_dxjp1 = -alfa * coeff / x_face[(k + 1) * m + j];
                    e00_k[i * nc + j] += dcoeff_dxj00 * dc;
                    ep1_k[i * nc + j] += dcoeff_dxjp1 * dc;
                }
            }
        }
    }

    let dm_k = cell_dm[k];
    if jac_only {
        for j in 0..nc {
            dx_dt_k[j] /= dm_k;
            for i in 0..nc {
                em1_k[i * nc + j] /= dm_k;
                e00_k[i * nc + j] /= dm_k;
                ep1_k[i * nc + j] /= dm_k;
            }
        }
        return;
    }

    let dt_div_dm = dt / dm_k;
    for j in 0..nc {
        dx_dt_k[j] /= dm_k;
        rhs_k[j] = dx_dt_k[j] * dt - dx_k[j];
        for i in 0..nc {
            em1_k[i * nc + j] = -em1_k[i * nc + j] * dt_div_dm;
            e00_k[i * nc + j] = -e00_k[i * nc + j] * dt_div_dm;
            ep1_k[i * nc + j] = -ep1_k[i * nc + j] * dt_div_dm;
        }
        e00_k[j * nc + j] += 1.0;
    }
}

/// Computes the damping factor `lambda` keeping `X + lambda * del >= -eps`
/// across active zones `nzlo..=nzhi` (`diffusion.f90:1785-1817`).
#[must_use]
pub fn positivity(
    nc: usize,
    nzlo: usize,
    nzhi: usize,
    x: &[f64],
    eps: f64,
    del: &[f64],
) -> f64 {
    let mut lambda = 1.0_f64;
    for k in nzlo..=nzhi {
        for j in 0..nc {
            let old_x = x[k * nc + j];
            if old_x < 1.0e-99 {
                continue;
            }
            let dx = del[k * nc + j];
            let new_x = old_x + dx;
            if new_x >= 0.0 {
                continue;
            }
            let alpha = -(old_x + eps) / dx;
            if alpha < lambda {
                lambda = alpha;
            }
        }
    }
    if lambda < 1.0 {
        lambda *= 0.8;
    }
    lambda
}

/// Maps network species onto 0-based diffusion classes `0..nc` using
/// `class_a_max` thresholds or 1-to-1 full-net mapping
/// (`diffusion.f90:1968-1999`).
#[must_use]
pub fn set_diffusion_classes(
    nc: usize,
    species_a: &[f64],
    class_a_max: &[f64],
    use_full_net: bool,
) -> Vec<usize> {
    let species = species_a.len();
    let mut class = vec![0_usize; species];
    if use_full_net {
        for (i, c) in class.iter_mut().enumerate() {
            *c = i;
        }
    } else {
        for (i, &a_val) in species_a.iter().enumerate() {
            let mut cls = nc - 1;
            for (j, &a_max) in class_a_max.iter().enumerate().take(nc - 1) {
                if a_val <= a_max {
                    cls = j;
                    break;
                }
            }
            class[i] = cls;
        }
    }
    class
}

/// H211b low-pass timestep controller ported from `diffusion.f90:2002-2048`.
#[must_use]
pub fn adjust_timestep(
    vc_target: f64,
    vc: f64,
    vc_old: f64,
    dt: f64,
    dt_old: f64,
    max_timestep_factor: f64,
    min_timestep_factor: f64,
) -> f64 {
    const ORDER: f64 = 1.0;
    const BETA1: f64 = 0.25 / ORDER;
    const BETA2: f64 = 0.25 / ORDER;
    const ALPHA2: f64 = 0.25;
    let limiter = |x: f64| 1.0 + 2.0 * flexmesa_math::atan((x - 1.0) / 2.0);

    let mut dt_next = 1.0e99_f64;
    if vc_old > 0.0 && dt_old > 0.0 {
        let vcratio = limiter(vc_target / vc);
        let vcratio_prev = limiter(vc_target / vc_old);
        let limtr = limiter(
            flexmesa_math::pow(vcratio, BETA1)
                * flexmesa_math::pow(vcratio_prev, BETA2)
                * flexmesa_math::pow(dt / dt_old, -ALPHA2),
        );
        dt_next = dt_next.min(dt * limtr);
    } else {
        dt_next = dt_next.min(dt * vc_target / vc);
    }

    if max_timestep_factor > 0.0 && dt_next > max_timestep_factor * dt {
        dt_next = max_timestep_factor * dt;
    }
    if min_timestep_factor > 0.0 && dt_next < min_timestep_factor * dt {
        dt_next = min_timestep_factor * dt;
    }
    dt_next
}

// ---------------------------------------------------------------------------
// Public Physics Helpers (backed by exact MESA 26.4.1 routines)
// ---------------------------------------------------------------------------

/// Evaluates the degenerate-corrected effective electron thermal energy
/// $k_{\mathrm{B}} T_{\mathrm{eff}} =
/// \sqrt{(k_{\mathrm{B}} T)^2 + (\tfrac{2}{3} E_{\mathrm{F}})^2}$ in erg.
///
/// Retained for [`crate::rotation`] thermal-diffusion screening.
#[must_use]
#[inline]
pub fn eval_effective_kt(n_e: f64, kt: f64) -> f64 {
    let kf = flexmesa_math::pow(3.0 * PI * PI * n_e.max(1.0e-10), 1.0 / 3.0);
    let ef = flexmesa_math::pow2(HBAR * kf) / (2.0 * ME);
    flexmesa_math::hypot(kt, (2.0 / 3.0) * ef)
}

/// Evaluates the Caplan, Bauer, & Freeman (2022) strong-coupling one-component
/// plasma (OCP) dimensionless self-diffusion coefficient $D^*(\Gamma, \kappa)$
/// via [`get_dstar_ocp`] (`diffusion_support.f90:1716-1743`).
#[must_use]
pub fn eval_caplan_ocp_dstar(gamma_i: f64, kappa: f64) -> f64 {
    get_dstar_ocp(gamma_i, kappa)
}

/// Evaluates collision-integral ratios `(ln_lam_or_k11, zdiff, zdiff1, zdiff2)`
/// for a pair of charges `(z_i, z_j)` using MESA `26.4.1` Paquette / Stanton &
/// Murillo / pure-Coulomb formulations (`paquette_coeffs.f90`,
/// `diffusion_support.f90`).
#[must_use]
pub fn eval_coulomb_collision_factors(
    scheme: DiffusionScheme,
    z_i: f64,
    z_j: f64,
    t: f64,
    n_e: f64,
    n_ions: f64,
    sum_n_z2: f64,
) -> (f64, f64, f64, f64) {
    let kt = KERG * t.max(100.0);
    let e2 = QE * QE;
    let z_prod = (z_i * z_j).abs().max(1.0e-6);

    let lam_i2 =
        flexmesa_math::pow(3.0 / (4.0 * PI * n_ions.max(1.0e-10)), 2.0 / 3.0);
    let lam_d2 = kt / (4.0 * PI * e2 * (n_e + sum_n_z2).max(1.0e-10));
    let lam = lam_d2.max(lam_i2).sqrt();

    match scheme {
        DiffusionScheme::IbenMacDonald => {
            let xij = 2.3939e3 * t * lam / z_prod;
            let cl = 0.81245
                * flexmesa_math::log1p(0.18769 * flexmesa_math::pow(xij, 1.2));
            (cl, 0.6, 1.3, 2.0)
        }
        DiffusionScheme::Paquette
        | DiffusionScheme::StantonMurillo
        | DiffusionScheme::CaplanOcp => {
            // Evaluate exact 2-ion + electron pair via `get1_burgers_coeffs`.
            let a = [1.0, 4.0, ME / AMU];
            let z = [z_i.abs().max(1.0), z_j.abs().max(1.0), -1.0];
            let x = [0.5, 0.5, 0.0];
            let rho = (n_ions * 2.5 * AMU).max(1.0e-12);
            let tmp = z[0] * x[0] / a[0] + z[1] * x[1] / a[1];
            let c = [x[0] / (a[0] * tmp), x[1] / (a[1] * tmp), 1.0];
            let mut kappa_st = [0.0_f64; 9];
            let mut zdiff = [0.0_f64; 9];
            let mut zdiff1 = [0.0_f64; 9];
            let mut zdiff2 = [0.0_f64; 9];
            get1_burgers_coeffs(
                2,
                3,
                &a,
                &z,
                &x,
                &c,
                rho,
                t.max(100.0),
                false,
                scheme == DiffusionScheme::Paquette,
                scheme == DiffusionScheme::CaplanOcp,
                1.0,
                &mut kappa_st,
                &mut zdiff,
                &mut zdiff1,
                &mut zdiff2,
            );
            (kappa_st[1], zdiff[1], zdiff1[1], zdiff2[1])
        }
    }
}

/// Solves the multi-species CGS Burgers equations in a single shell using MESA
/// `26.4.1` `get1_burgers_coeffs`, `get1_gradient_coeffs`, and
/// `get1_diffusion_velocities` (`diffusion_support.f90:365-747`).
///
/// # Errors
///
/// Returns [`StarError::ConvergenceFailure`] if the Burgers linear system is
/// singular or non-finite.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn eval_burgers_velocities(
    a: &[f64],
    z: &[f64],
    x_mass: &[f64],
    rho: f64,
    t: f64,
    dlnt_dr: f64,
    g_grav: f64,
    g_rad: Option<&[f64]>,
    scheme: DiffusionScheme,
) -> Result<BurgersZoneResult, StarError> {
    let nc = a.len();
    if nc == 0 || z.len() != nc || x_mass.len() != nc {
        return Err(StarError::InvalidInput(
            "a, z, and x_mass must have identical non-zero length".into(),
        ));
    }
    let m = nc + 1;
    let mut a_full = vec![0.0_f64; m];
    let mut z_full = vec![0.0_f64; m];
    let mut x_full = vec![0.0_f64; m];
    let mut c_full = vec![0.0_f64; m];
    let mut rad_full = vec![0.0_f64; m];

    let mut tmp = 0.0;
    for i in 0..nc {
        a_full[i] = a[i];
        z_full[i] = 1.0_f64.max(z[i]);
        x_full[i] = TINY_X.max(x_mass[i]);
        tmp += z_full[i] * x_full[i] / a_full[i];
        if let Some(gr) = g_rad {
            rad_full[i] = gr.get(i).copied().unwrap_or(0.0);
        }
    }
    a_full[nc] = ME / AMU;
    z_full[nc] = -1.0;
    let mut ac = 0.0;
    for i in 0..nc {
        c_full[i] = x_full[i] / (a_full[i] * tmp);
        ac += a_full[i] * c_full[i];
    }
    c_full[nc] = 1.0;
    x_full[nc] = a_full[nc] / ac;

    // Estimate plasma coupling Gamma for Caplan blending when requested.
    let mut n_ions = 0.0;
    let mut z_avg = 0.0;
    for i in 0..nc {
        let ni = rho * x_full[i] / (a_full[i] * AMU);
        n_ions += ni;
        z_avg += ni * flexmesa_math::pow(z_full[i], 5.0 / 3.0);
    }
    let a_e = flexmesa_math::pow(
        3.0 / (4.0 * PI * (rho * tmp / AMU).max(1.0e-30)),
        1.0 / 3.0,
    );
    let gamma_est =
        (z_avg / n_ions.max(1.0e-30)) * QE * QE / (a_e * BOLTZM * t.max(100.0));

    let mut kappa_st = vec![0.0_f64; m * m];
    let mut zdiff = vec![0.0_f64; m * m];
    let mut zdiff1 = vec![0.0_f64; m * m];
    let mut zdiff2 = vec![0.0_f64; m * m];

    get1_burgers_coeffs(
        nc,
        m,
        &a_full,
        &z_full,
        &x_full,
        &c_full,
        rho,
        t,
        scheme == DiffusionScheme::IbenMacDonald,
        scheme == DiffusionScheme::Paquette,
        scheme == DiffusionScheme::CaplanOcp,
        gamma_est,
        &mut kappa_st,
        &mut zdiff,
        &mut zdiff1,
        &mut zdiff2,
    );

    let mut ap = vec![0.0_f64; m];
    let mut at = vec![0.0_f64; m];
    let mut ar = vec![0.0_f64; m];
    let mut ax = vec![0.0_f64; m * m];
    let mut e_ap = 0.0;
    let mut e_at = 0.0;
    let mut e_ar = 0.0;
    let mut e_ax = vec![0.0_f64; m];
    let mut g_ap = 0.0;
    let mut g_at = 0.0;
    let mut g_ar = 0.0;
    let mut g_ax = vec![0.0_f64; m];

    get1_gradient_coeffs(
        0,
        m,
        &a_full,
        &z_full,
        &x_full,
        &c_full,
        rho,
        t,
        true,
        0.0,
        10.0,
        40.0,
        g_rad.is_some(),
        &rad_full,
        &kappa_st,
        &zdiff,
        &zdiff1,
        &zdiff2,
        &mut ap,
        &mut at,
        &mut ar,
        &mut ax,
        &mut e_ap,
        &mut e_at,
        &mut e_ar,
        &mut e_ax,
        &mut g_ap,
        &mut g_at,
        &mut g_ar,
        &mut g_ax,
    )?;

    let diff_fac = vec![1.0_f64; nc];
    let r_mid = [2.0_f64, 1.0_f64];
    let mut v_gt = vec![0.0_f64; nc];
    let mut vln_p = vec![0.0_f64; nc];
    let mut vln_t = vec![0.0_f64; nc];
    let mut vrad = vec![0.0_f64; nc];
    let mut sigma_lnc = vec![0.0_f64; nc * nc];

    get1_diffusion_velocities(
        1,
        nc,
        m,
        &ap,
        &at,
        &ar,
        &ax,
        rho,
        t,
        0.0,
        dlnt_dr,
        g_grav,
        0.0,
        &x_full,
        0.0,
        0.0,
        0.0,
        1.0,
        &r_mid,
        1.0,
        1.0,
        1.0e5,
        &diff_fac,
        true,
        &mut v_gt,
        &mut vln_p,
        &mut vln_t,
        &mut vrad,
        &mut sigma_lnc,
    );

    let mut d_self = vec![0.0_f64; nc];
    for i in 0..nc {
        d_self[i] = sigma_lnc[i * nc + i];
    }
    let ee_force = BOLTZM * t * dlnt_dr * e_at + AMU * g_grav * e_ap;

    Ok(BurgersZoneResult {
        v_gt,
        d_self,
        ee_force,
        z_charge: z.to_vec(),
    })
}

/// Computes the radiative levitation acceleration $g_{\mathrm{rad},i} =
/// \frac{\kappa_i L_r}{4\pi r^2 c}$ in $\mathrm{cm\,s^{-2}}$.
#[must_use]
pub fn eval_radiative_levitation_accel(
    kappa_species: f64,
    lum_face: f64,
    r_face: f64,
) -> f64 {
    let area = 4.0 * PI * flexmesa_math::pow2(r_face.max(1.0e5));
    (kappa_species.max(0.0) * lum_face.max(0.0)) / (area * CLIGHT)
}

/// Computes the white-dwarf gravitational sedimentation heating rate
/// $\epsilon_{\mathrm{WD,sed}}$ in $\mathrm{erg\,g^{-1}\,s^{-1}}$ for a single
/// zone following `element_diffusion.f90:391-395`.
#[must_use]
pub fn eval_wd_sedimentation_heating(
    a: &[f64],
    z: &[f64],
    x_mass: &[f64],
    v_gt: &[f64],
    abar: f64,
    zbar: f64,
    g_grav: f64,
) -> f64 {
    let n = a.len().min(z.len()).min(x_mass.len()).min(v_gt.len());
    let ee_grav = AMU * g_grav / (zbar.max(1.0) + 1.0);
    let mut eps_wd = 0.0;
    for i in 0..n {
        let ai = a[i].max(1.0);
        let zi = z[i].max(1.0);
        let f_e = (zi - ai * zbar / abar.max(1.0)) * ee_grav;
        eps_wd += (x_mass[i] / (ai * AMU)) * v_gt[i] * f_e;
    }
    eps_wd.max(0.0)
}
