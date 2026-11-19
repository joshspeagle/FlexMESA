// SPDX-License-Identifier: LGPL-3.0-or-later
//! Pure Safe Rust FIPS 180-4 `SHA-256` digest computation.
//!
//! Provides zero-dependency streaming and in-memory `SHA-256` hashing with zero
//! `unsafe` code, shared across `flexmesa-data`, `flexmesa-py`, and vendored
//! integrity checks.

use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

const INITIAL_STATE: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

const ROUND_CONSTANTS: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// Streaming FIPS 180-4 `SHA-256` state machine over 512-bit (`64`-byte)
/// blocks.
#[derive(Clone, Debug)]
pub struct Sha256Hasher {
    state: [u32; 8],
    buffer: [u8; 64],
    buffer_len: usize,
    total_bytes: u64,
}

impl Default for Sha256Hasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256Hasher {
    /// Construct a fresh `SHA-256` state machine initialized to FIPS 180-4
    /// `INITIAL_STATE`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: INITIAL_STATE,
            buffer: [0u8; 64],
            buffer_len: 0,
            total_bytes: 0,
        }
    }

    /// Feed arbitrary input bytes into the streaming 512-bit block buffer.
    pub fn update(&mut self, mut data: &[u8]) {
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

    /// Apply FIPS 180-4 bit-length padding and format the final 256-bit state
    /// as a 64-character lowercase hex string.
    #[must_use]
    pub fn finalize_hex(mut self) -> String {
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
#[must_use]
pub fn is_valid_sha256_hex(digest: &str) -> bool {
    let trimmed = digest.trim();
    trimmed.len() == 64 && trimmed.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Compute the lowercase 64-character `SHA-256` hex digest of `data`.
#[must_use]
pub fn compute_sha256_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256Hasher::new();
    hasher.update(data);
    hasher.finalize_hex()
}

/// Compute the lowercase 64-character `SHA-256` hex digest from a [`Read`]
/// stream in `64 KiB` chunks.
///
/// # Errors
/// Propagates any [`std::io::Error`] returned by `reader`.
pub fn compute_sha256_reader<R: Read>(
    reader: &mut R,
) -> std::io::Result<String> {
    let mut hasher = Sha256Hasher::new();
    let mut buf = vec![0u8; 64 * 1024].into_boxed_slice();
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize_hex())
}

/// Compute the raw 32-byte FIPS 180-4 `SHA-256` digest of `data`.
#[must_use]
pub fn compute_sha256_raw(data: &[u8]) -> [u8; 32] {
    let hex = compute_sha256_bytes(data);
    decode_sha256_hex(&hex).unwrap_or([0_u8; 32])
}

/// Decode a 64-character ASCII hex `SHA-256` digest into raw `[u8; 32]` bytes.
#[must_use]
pub fn decode_sha256_hex(hex: &str) -> Option<[u8; 32]> {
    let trimmed = hex.trim();
    if !is_valid_sha256_hex(trimmed) {
        return None;
    }
    let mut out = [0_u8; 32];
    for (slot, pair) in
        out.iter_mut().zip(trimmed.as_bytes().as_chunks::<2>().0)
    {
        let s = std::str::from_utf8(pair).ok()?;
        *slot = u8::from_str_radix(s, 16).ok()?;
    }
    Some(out)
}

/// Format a raw 32-byte `SHA-256` digest as a 64-character lowercase hex string.
#[must_use]
pub fn encode_sha256_hex(bytes: &[u8; 32]) -> String {
    let mut hex = String::with_capacity(64);
    for b in bytes {
        let _ = write!(hex, "{b:02x}");
    }
    hex
}

/// Compute the lowercase 64-character `SHA-256` hex digest of `path`.
///
/// # Errors
/// Propagates any [`std::io::Error`] encountered opening or reading `path`.
pub fn compute_sha256_file(path: &Path) -> std::io::Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(256 * 1024, file);
    compute_sha256_reader(&mut reader)
}
