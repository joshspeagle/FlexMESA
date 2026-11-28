// SPDX-License-Identifier: GPL-3.0-or-later
//! Thin wrappers over the `hydro_rotation` and `rotation_mix` ports: shell
//! `j_rot`-preserving distortion refresh, rotational mixing
//! augmentation of `d_mix`, and the post-Henyey `omega` transport sync.

use crate::controls::StarError;
use crate::model::StarModel;

/// Recomputes `(w_div_wc, i_rot, omega, fp_rot, ft_rot)` preserving shell
/// `j_rot` (`set_rotation_info` in `star/private/hydro_rotation.f90:537-586`).
pub fn conserve_shell_j_rot_and_distortion(model: &mut StarModel) {
    if !model.controls.rotation_flag || model.nz == 0 {
        return;
    }
    crate::hydro_rotation::set_rotation_info(model, false);
}

/// Augments `model.d_mix` with Heger et al. (2000) and Spruit (2002) diffusivities.
///
/// # Errors
/// Propagates `update_rotation_mixing_info` errors.
pub fn augment_d_mix_with_rotation(
    model: &mut StarModel,
) -> Result<(), StarError> {
    crate::rotation_mix::augment_d_mix_with_rotation(model)
}

/// Updates post-Henyey $\Omega_k = j_{\mathrm{rot},k} / i_{\mathrm{rot},k}$ and $\Omega$ transport.
///
/// # Errors
/// Propagates `update_rotation_mixing_info` and `solve_omega_mix` failures,
/// including a requested retry or termination.
pub fn sync_post_henyey_rotation(
    model: &mut StarModel,
    dt: f64,
) -> Result<(), StarError> {
    crate::rotation_mix::sync_post_henyey_rotation(model, dt)
}

#[cfg(test)]
mod tests {
    use super::conserve_shell_j_rot_and_distortion;
    use crate::controls::{StarControls, StarJobControls};
    use crate::hydro_rotation::use_xh_to_update_i_rot_and_j_rot;
    use crate::zams::load_zams_model;

    #[test]
    fn test_conserve_shell_j_rot_preserves_j_rot_and_rebuilds_omega() {
        let ctrl = StarControls {
            initial_mass: 1.0,
            rotation_flag: true,
            zbase: 0.02,
            ..StarControls::default()
        };
        let mut model = load_zams_model(ctrl, StarJobControls::default())
            .expect("1 Msun ZAMS");
        model.omega.fill(2.0e-4);
        use_xh_to_update_i_rot_and_j_rot(&mut model);
        let j_before = model.j_rot.clone();
        assert!(j_before.iter().all(|&j| j > 0.0));
        model.omega.fill(0.0);

        conserve_shell_j_rot_and_distortion(&mut model);
        assert!(model.nz > 0);
        for k in 0..model.nz {
            assert_eq!(
                model.j_rot[k].to_bits(),
                j_before[k].to_bits(),
                "j_rot[{k}] must be preserved"
            );
            assert_eq!(
                model.omega[k].to_bits(),
                (model.j_rot[k] / model.aux.i_rot[k]).to_bits(),
                "omega[{k}] must be j_rot / i_rot"
            );
        }
        assert!(model.aux.fp_rot[0] < 1.0 && model.aux.ft_rot[0] < 1.0);
    }

    #[test]
    fn test_conserve_shell_j_rot_is_noop_when_rotation_flag_false() {
        let ctrl = StarControls {
            initial_mass: 1.0,
            rotation_flag: false,
            zbase: 0.02,
            ..StarControls::default()
        };
        let mut model = load_zams_model(ctrl, StarJobControls::default())
            .expect("1 Msun ZAMS");
        model.omega.fill(5.0e-5);
        model.j_rot.fill(0.0);
        model.aux.fp_rot.fill(1.0);
        model.aux.ft_rot.fill(1.0);
        model.aux.i_rot.fill(0.0);
        let fp_before = model.aux.fp_rot.clone();
        let ft_before = model.aux.ft_rot.clone();
        let i_before = model.aux.i_rot.clone();
        let j_before = model.j_rot.clone();

        conserve_shell_j_rot_and_distortion(&mut model);
        assert_eq!(model.aux.fp_rot, fp_before);
        assert_eq!(model.aux.ft_rot, ft_before);
        assert_eq!(model.aux.i_rot, i_before);
        assert_eq!(model.j_rot, j_before);
    }
}
