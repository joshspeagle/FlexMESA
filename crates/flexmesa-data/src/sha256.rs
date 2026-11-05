// SPDX-License-Identifier: LGPL-3.0-or-later
//! Pure Safe Rust FIPS 180-4 `SHA-256` digest computation and verification.
//!
//! Provides streaming and in-memory `SHA-256` hashing with zero `unsafe` code
//! and zero external cryptographic dependencies, used to verify `.safetensors`
//! physics tables against `manifest.toml` digests before loading.

use std::fmt::Write as _;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::DataError;

/// FIPS 180-4 initial 256-bit state words (`H[0..8]`) derived from the
/// fractional parts of the square roots of the first 8 primes.
#[rustfmt::skip]
const INITIAL_STATE: [u32; 8] = [
    0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372, 0xa54f_f53a,
    0x510e_527f, 0x9b05_688c, 0x1f83_d9ab, 0x5be0_cd19,
];

/// FIPS 180-4 round constants (`K[0..64]`) derived from the fractional
/// parts of the cube roots of the first 64 primes.
#[rustfmt::skip]
const ROUND_CONSTANTS: [u32; 64] = [
    0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5,
    0x3956_c25b, 0x59f1_11f1, 0x923f_82a4, 0xab1c_5ed5,
    0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3,
    0x72be_5d74, 0x80de_b1fe, 0x9bdc_06a7, 0xc19b_f174,
    0xe49b_69c1, 0xefbe_4786, 0x0fc1_9dc6, 0x240c_a1cc,
    0x2de9_2c6f, 0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da,
    0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7,
    0xc6e0_0bf3, 0xd5a7_9147, 0x06ca_6351, 0x1429_2967,
    0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc, 0x5338_0d13,
    0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85,
    0xa2bf_e8a1, 0xa81a_664b, 0xc24b_8b70, 0xc76c_51a3,
    0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070,
    0x19a4_c116, 0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5,
    0x391c_0cb3, 0x4ed8_aa4a, 0x5b9c_ca4f, 0x682e_6ff3,
    0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208,
    0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7, 0xc671_78f2,
];

/// Streaming FIPS 180-4 `SHA-256` state machine over 512-bit (`64`-byte)
/// blocks.
///
/// Accumulates arbitrary byte slices in a 64-byte internal block buffer and
/// updates the 256-bit state array `H[0..8]` whenever a complete block fills.
struct Sha256Hasher {
    /// Current 256-bit intermediate hash state `H_0 .. H_7`.
    state: [u32; 8],
    /// Partial 512-bit message block buffer.
    buffer: [u8; 64],
    /// Number of valid bytes currently stored in `buffer` (`0..64`).
    buffer_len: usize,
    /// Total message bytes processed so far.
    total_bytes: u64,
}

impl Sha256Hasher {
    /// Construct a fresh `SHA-256` state machine initialized to FIPS 180-4
    /// `INITIAL_STATE`.
    fn new() -> Self {
        Self {
            state: INITIAL_STATE,
            buffer: [0u8; 64],
            buffer_len: 0,
            total_bytes: 0,
        }
    }

    /// Feed arbitrary input bytes into the streaming 512-bit block buffer.
    fn update(&mut self, mut data: &[u8]) {
        self.total_bytes = self.total_bytes.wrapping_add(data.len() as u64);
        if self.buffer_len > 0 {
            let take = (64 - self.buffer_len).min(data.len());
            self.buffer[self.buffer_len..self.buffer_len + take]
                .copy_from_slice(&data[..take]);
            self.buffer_len += take;
            data = &data[take..];
            if self.buffer_len == 64 {
                let block = self.buffer;
                compress_block(&mut self.state, &block);
                self.buffer_len = 0;
            }
        }
        let (blocks, remainder) = data.as_chunks::<64>();
        for block in blocks {
            compress_block(&mut self.state, block);
        }
        if !remainder.is_empty() {
            self.buffer[..remainder.len()].copy_from_slice(remainder);
            self.buffer_len = remainder.len();
        }
    }

    /// Apply FIPS 180-4 bit-length padding (`0x80 || 0x00* || bit_len_be64`)
    /// and format the final 256-bit state as a 64-character lowercase hex
    /// string.
    fn finalize_hex(mut self) -> String {
        let bit_len = self.total_bytes.wrapping_mul(8);
        self.buffer[self.buffer_len] = 0x80;
        self.buffer_len += 1;
        if self.buffer_len > 56 {
            self.buffer[self.buffer_len..64].fill(0);
            let block = self.buffer;
            compress_block(&mut self.state, &block);
            self.buffer_len = 0;
        }
        self.buffer[self.buffer_len..56].fill(0);
        self.buffer[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buffer;
        compress_block(&mut self.state, &block);

        let mut hex = String::with_capacity(64);
        for word in self.state {
            let _ = write!(hex, "{word:08x}");
        }
        hex
    }
}

/// Compress a single 512-bit (`64`-byte) block into `state`.
#[allow(clippy::many_single_char_names)]
fn compress_block(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for (i, chunk) in block.as_chunks::<4>().0.iter().enumerate() {
        w[i] = u32::from_be_bytes(*chunk);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7)
            ^ w[i - 15].rotate_right(18)
            ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17)
            ^ w[i - 2].rotate_right(19)
            ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(ROUND_CONSTANTS[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }

    for (st, val) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *st = st.wrapping_add(val);
    }
}

/// Return `true` if `digest` is a 64-character ASCII hexadecimal string.
///
/// # Examples
/// ```
/// use flexmesa_data::sha256::is_valid_sha256_hex;
///
/// assert!(is_valid_sha256_hex(&"a".repeat(64)));
/// assert!(!is_valid_sha256_hex("invalid"));
/// ```
#[must_use]
pub fn is_valid_sha256_hex(digest: &str) -> bool {
    let trimmed = digest.trim();
    trimmed.len() == 64 && trimmed.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Compute the lowercase 64-character `SHA-256` hex digest of `data`.
///
/// # Examples
/// ```
/// use flexmesa_data::sha256::compute_sha256_bytes;
///
/// assert_eq!(
///     compute_sha256_bytes(b""),
///     "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
/// );
/// ```
#[must_use]
pub fn compute_sha256_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256Hasher::new();
    hasher.update(data);
    hasher.finalize_hex()
}

/// Compute the lowercase 64-character `SHA-256` hex digest of a file on disk.
///
/// # Errors
/// Returns [`DataError::IoError`] if `path` cannot be opened or read.
pub fn compute_sha256_file(path: &Path) -> Result<String, DataError> {
    let mut file = File::open(path).map_err(|err| DataError::IoError {
        path: path.display().to_string(),
        message: err.to_string(),
    })?;
    let mut hasher = Sha256Hasher::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf).map_err(|err| DataError::IoError {
            path: path.display().to_string(),
            message: err.to_string(),
        })?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize_hex())
}

/// Verify that `path` matches `expected_sha256`.
///
/// # Errors
/// Returns [`DataError::ValidationError`] if `expected_sha256` is not a valid
/// 64-character hex string, [`DataError::IoError`] on file read failure, or
/// [`DataError::ChecksumMismatch`] if the computed digest differs.
pub fn verify_file_sha256(
    path: &Path,
    expected_sha256: &str,
) -> Result<String, DataError> {
    if !is_valid_sha256_hex(expected_sha256) {
        return Err(DataError::ValidationError(format!(
            "invalid expected SHA-256 hex digest: {expected_sha256:?}"
        )));
    }
    let expected_norm = expected_sha256.trim().to_ascii_lowercase();
    let actual = compute_sha256_file(path)?;
    if actual != expected_norm {
        return Err(DataError::ChecksumMismatch {
            path: path.display().to_string(),
            expected: expected_norm,
            actual,
        });
    }
    Ok(actual)
}
