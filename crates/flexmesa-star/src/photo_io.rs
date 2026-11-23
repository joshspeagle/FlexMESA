// SPDX-License-Identifier: LGPL-3.0-or-later
//! Binary `.photo` full-state checkpoint serializer, deserializer, and
//! 64-bit FNV-1a checksum (`photo_in.f90`, `photo_out.f90`, and `./ck`).

use flexmesa_utils::bin_codec;

use crate::controls::{StarControls, StarError, StarJobControls};
use crate::job_controls::VarLayout;
use crate::model::StarModel;

const PHOTO_MAGIC: &[u8; 8] = b"FMPHOTO2";

/// Binary `.photo` format version written by the generated `serde` encoder.
pub const PHOTO_FORMAT_VERSION: u32 = 30;

/// Computes a deterministic 64-bit FNV-1a checksum of `.photo` restart bytes
/// (`./ck` in upstream MESA `test_suite_helpers`).
#[must_use]
pub fn compute_photo_checksum(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3_u64);
    }
    hash
}

fn decode_photo_frame(
    bytes: &[u8],
) -> Result<(flexmesa_math::NumericsMode, &[u8]), StarError> {
    let (mode_tag, payload) = bin_codec::decode_framed_header(
        bytes,
        PHOTO_MAGIC,
        PHOTO_FORMAT_VERSION,
        "photo",
    )
    .map_err(StarError::IoError)?;
    let mode = flexmesa_math::NumericsMode::from_wire_tag(mode_tag)
        .ok_or_else(|| {
            StarError::IoError(format!(
                "invalid photo numerics_mode tag {mode_tag}"
            ))
        })?;
    Ok((mode, payload))
}

/// Serializes the full [`StarModel`] state (including `controls`, `job`, and
/// cumulative counters) into deterministic binary `.photo` checkpoint bytes.
///
/// # Errors
/// Returns [`StarError::IoError`] if `serde` encoding fails.
pub fn try_write_photo_bytes(model: &StarModel) -> Result<Vec<u8>, StarError> {
    let payload = bin_codec::to_bytes(model).map_err(|e| {
        StarError::IoError(format!("photo serialization failed: {e}"))
    })?;
    Ok(bin_codec::encode_framed_bytes(
        PHOTO_MAGIC,
        PHOTO_FORMAT_VERSION,
        flexmesa_math::numerics_mode().to_wire_tag(),
        &payload,
    ))
}

/// Serializes the full [`StarModel`] state into deterministic binary `.photo`
/// checkpoint bytes.
///
/// # Panics
/// Panics only if internal `serde` binary encoding reports a serialization error.
#[must_use]
pub fn write_photo_bytes(model: &StarModel) -> Vec<u8> {
    try_write_photo_bytes(model).expect("StarModel photo encoding succeeds")
}

/// Extracts the [`flexmesa_math::NumericsMode`] recorded in the header of
/// serialized `.photo` checkpoint `bytes`.
///
/// # Errors
/// Returns [`StarError::IoError`] if `bytes` has an invalid or truncated header.
pub fn photo_numerics_mode(
    bytes: &[u8],
) -> Result<flexmesa_math::NumericsMode, StarError> {
    decode_photo_frame(bytes).map(|(mode, _)| mode)
}

/// Restores a complete [`StarModel`] (including serialized `controls` and
/// `job`) from binary `.photo` checkpoint bytes without mutating stored `dq` bits.
///
/// # Errors
/// Returns [`StarError::IoError`] if header validation fails, `numerics_mode`
/// does not match the active backend mode, or binary decoding detects truncated
/// or trailing bytes.
pub fn load_photo_bytes(bytes: &[u8]) -> Result<StarModel, StarError> {
    let (saved_mode, payload) = decode_photo_frame(bytes)?;
    let active_mode = flexmesa_math::numerics_mode();
    if saved_mode != active_mode {
        return Err(StarError::IoError(format!(
            "photo numerics_mode mismatch: checkpoint saved in \"{saved_mode}\" \
             mode cannot be loaded while active mode is \"{active_mode}\""
        )));
    }
    let model: StarModel = bin_codec::from_bytes(payload)
        .map_err(|e| StarError::IoError(format!("photo decode failed: {e}")))?;
    Ok(model)
}

/// Restores a [`StarModel`] from binary `.photo` restart bytes and applies the
/// caller-supplied `controls` and `job` settings (updating [`VarLayout`] when
/// hydrodynamic variable flags change).
///
/// # Errors
/// Returns [`StarError::IoError`] if header validation, `numerics_mode`, or
/// binary decoding fails.
pub fn read_photo_bytes(
    bytes: &[u8],
    controls: StarControls,
    job: StarJobControls,
) -> Result<StarModel, StarError> {
    let mut model = load_photo_bytes(bytes)?;
    let layout_changed = model.controls.v_flag != controls.v_flag
        || model.controls.u_flag != controls.u_flag
        || model.controls.rsp2_flag != controls.rsp2_flag
        || model.controls.rti_flag != controls.rti_flag;
    model.controls = controls;
    model.job = job;
    if layout_changed {
        model.layout = VarLayout::with_flags(
            model.species,
            model.controls.v_flag,
            model.controls.u_flag,
            model.controls.rsp2_flag,
            model.controls.rti_flag,
        );
        model.resize_zones(model.nz);
    }
    Ok(model)
}
