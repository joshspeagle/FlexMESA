// SPDX-License-Identifier: GPL-3.0-or-later
//! Versioned, source-bound, SHA-256-verified runtime cache envelope (`FMCENV01`).
//!
//! Layout (`96`-byte little-endian header followed by `payload_len` bytes):
//! - `0..8`: `b"FMCENV01"`
//! - `8..12`: `schema_version` (`u32`, must equal [`RUNTIME_CACHE_SCHEMA_VERSION`])
//! - `12..16`: `builder_version` (`u32`)
//! - `16..24`: `payload_tag` (`[u8; 8]`, e.g. `b"FMDT4D01"`, `b"FMCMS001"`, `b"FMHELM01"`, `b"FMKAP001"`, `b"FMIONCA1"`)
//! - `24..56`: `source_sha256` (`[u8; 32]`, SHA-256 of the upstream input file)
//! - `56..88`: `payload_sha256` (`[u8; 32]`, SHA-256 of the payload bytes)
//! - `88..96`: `payload_len` (`u64`)
//! - `96..`: `payload`

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::sha256::{
    compute_sha256_raw, compute_sha256_reader, decode_sha256_hex,
    encode_sha256_hex,
};
use crate::RUNTIME_CACHE_SCHEMA_VERSION;

/// Magic header for versioned runtime derived caches (`FMCENV01`).
pub const CACHE_ENVELOPE_MAGIC: &[u8; 8] = b"FMCENV01";

/// Fixed byte size of the `FMCENV01` envelope header.
pub const CACHE_ENVELOPE_HEADER_BYTES: usize = 96;

static TMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Atomically write `payload` wrapped in a verified `FMCENV01` header to `path`.
///
/// # Errors
/// Returns an error string if `source_sha256_hex` is not a valid 64-char hex
/// digest or if creating, writing, flushing, or renaming the temp file fails.
pub fn write_verified_cache_atomic(
    path: &Path,
    builder_version: u32,
    payload_tag: &[u8; 8],
    source_sha256_hex: &str,
    payload: &[u8],
) -> Result<(), String> {
    let source_sha = decode_sha256_hex(source_sha256_hex).ok_or_else(|| {
        format!("invalid source SHA-256 '{source_sha256_hex}'")
    })?;
    let payload_sha = compute_sha256_raw(payload);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            format!("failed to create cache dir {}: {e}", parent.display())
        })?;
    }
    let seq = TMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let tmp_name = format!(
        "{}.{}.{}.tmp",
        path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("cache.bin"),
        std::process::id(),
        seq
    );
    let tmp_path = path.with_file_name(tmp_name);
    let write_res = (|| -> Result<(), String> {
        let file = File::create(&tmp_path).map_err(|e| {
            format!("failed to create temp cache {}: {e}", tmp_path.display())
        })?;
        let mut writer = BufWriter::with_capacity(512 * 1024, file);
        writer
            .write_all(CACHE_ENVELOPE_MAGIC)
            .map_err(|e| e.to_string())?;
        writer
            .write_all(&RUNTIME_CACHE_SCHEMA_VERSION.to_le_bytes())
            .map_err(|e| e.to_string())?;
        writer
            .write_all(&builder_version.to_le_bytes())
            .map_err(|e| e.to_string())?;
        writer.write_all(payload_tag).map_err(|e| e.to_string())?;
        writer.write_all(&source_sha).map_err(|e| e.to_string())?;
        writer.write_all(&payload_sha).map_err(|e| e.to_string())?;
        writer
            .write_all(&(payload.len() as u64).to_le_bytes())
            .map_err(|e| e.to_string())?;
        writer.write_all(payload).map_err(|e| e.to_string())?;
        writer.flush().map_err(|e| e.to_string())?;
        Ok(())
    })();
    if let Err(err) = write_res {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }
    fs::rename(&tmp_path, path).map_err(|e| {
        let _ = fs::remove_file(&tmp_path);
        format!("failed to move verified cache {}: {e}", path.display())
    })
}

fn open_checked_envelope_reader(
    path: &Path,
    expected_builder_version: u32,
    expected_payload_tag: [u8; 8],
    expected_source_sha: Option<&[u8; 32]>,
    expected_source_sha256_hex: Option<&str>,
) -> Result<(BufReader<File>, [u8; 32], u64), String> {
    let file = File::open(path).map_err(|e| {
        format!("failed to read runtime cache {}: {e}", path.display())
    })?;
    let actual_len = file
        .metadata()
        .map_err(|e| {
            format!("failed to read runtime cache {}: {e}", path.display())
        })?
        .len();
    let header_len = CACHE_ENVELOPE_HEADER_BYTES as u64;
    if actual_len < header_len {
        return Err(format!(
            "truncated runtime cache header in {} ({actual_len} < {CACHE_ENVELOPE_HEADER_BYTES} bytes)",
            path.display()
        ));
    }
    let mut reader = BufReader::with_capacity(256 * 1024, file);
    let mut header = [0_u8; CACHE_ENVELOPE_HEADER_BYTES];
    reader.read_exact(&mut header).map_err(|e| {
        format!(
            "failed to read runtime cache header in {}: {e}",
            path.display()
        )
    })?;
    if &header[0..8] != CACHE_ENVELOPE_MAGIC {
        return Err(format!(
            "invalid runtime cache magic in {} (expected FMCENV01)",
            path.display()
        ));
    }
    let schema_ver = u32::from_le_bytes(
        header[8..12]
            .try_into()
            .map_err(|_| "short schema version slice".to_string())?,
    );
    if schema_ver != RUNTIME_CACHE_SCHEMA_VERSION {
        return Err(format!(
            "runtime cache schema version mismatch in {}: expected {}, got {schema_ver}",
            path.display(),
            RUNTIME_CACHE_SCHEMA_VERSION
        ));
    }
    let builder_ver = u32::from_le_bytes(
        header[12..16]
            .try_into()
            .map_err(|_| "short builder version slice".to_string())?,
    );
    if builder_ver != expected_builder_version {
        return Err(format!(
            "runtime cache builder version mismatch in {}: expected {expected_builder_version}, got {builder_ver}",
            path.display()
        ));
    }
    let payload_tag = &header[16..24];
    if payload_tag != &expected_payload_tag[..] {
        return Err(format!(
            "runtime cache payload tag mismatch in {} (expected {:?}, got {:?})",
            path.display(),
            String::from_utf8_lossy(&expected_payload_tag),
            String::from_utf8_lossy(payload_tag)
        ));
    }
    if let (Some(expected_sha), Some(expected_hex)) =
        (expected_source_sha, expected_source_sha256_hex)
    {
        let mut stored_source = [0_u8; 32];
        stored_source.copy_from_slice(&header[24..56]);
        if stored_source != *expected_sha {
            return Err(format!(
                "runtime cache source SHA-256 mismatch in {}: expected {expected_hex}, got {}",
                path.display(),
                encode_sha256_hex(&stored_source)
            ));
        }
    }
    let mut stored_payload = [0_u8; 32];
    stored_payload.copy_from_slice(&header[56..88]);
    let payload_len = u64::from_le_bytes(
        header[88..96]
            .try_into()
            .map_err(|_| "short payload len slice".to_string())?,
    );
    let expected_total =
        header_len.checked_add(payload_len).ok_or_else(|| {
            format!("runtime cache size overflow in {}", path.display())
        })?;
    if actual_len != expected_total {
        return Err(format!(
            "runtime cache length mismatch in {}: expected {expected_total} bytes, got {actual_len}",
            path.display()
        ));
    }
    Ok((reader, stored_payload, payload_len))
}

/// Read `path`, verify its `FMCENV01` header, `source_sha256`, exact byte length,
/// and `payload_sha256`, and return the verified payload bytes.
///
/// # Errors
/// Returns a descriptive error string naming `path` on I/O failure, truncated
/// header/payload, trailing bytes, version/tag mismatch, source hash mismatch,
/// or payload SHA-256 corruption.
pub fn read_verified_cache(
    path: &Path,
    expected_builder_version: u32,
    expected_payload_tag: &[u8; 8],
    expected_source_sha256_hex: &str,
) -> Result<Vec<u8>, String> {
    let expected_source_sha = decode_sha256_hex(expected_source_sha256_hex)
        .ok_or_else(|| {
            format!(
                "invalid expected source SHA-256 '{expected_source_sha256_hex}'"
            )
        })?;
    let (mut reader, stored_payload, payload_len_u64) =
        open_checked_envelope_reader(
            path,
            expected_builder_version,
            *expected_payload_tag,
            Some(&expected_source_sha),
            Some(expected_source_sha256_hex),
        )?;
    let payload_len = usize::try_from(payload_len_u64).map_err(|_| {
        format!(
            "runtime cache payload length overflow in {}: {payload_len_u64}",
            path.display()
        )
    })?;
    let mut payload = vec![0_u8; payload_len];
    reader.read_exact(&mut payload).map_err(|e| {
        format!(
            "failed to read runtime cache payload in {}: {e}",
            path.display()
        )
    })?;
    let mut trailing = [0_u8; 1];
    if reader.read(&mut trailing).map_err(|e| {
        format!(
            "failed to verify runtime cache EOF in {}: {e}",
            path.display()
        )
    })? != 0
    {
        return Err(format!(
            "runtime cache length mismatch in {} (unexpected trailing bytes)",
            path.display()
        ));
    }
    let actual_payload_sha = compute_sha256_raw(&payload);
    if actual_payload_sha != stored_payload {
        return Err(format!(
            "corrupt runtime cache payload in {} (SHA-256 mismatch: expected {}, got {})",
            path.display(),
            encode_sha256_hex(&stored_payload),
            encode_sha256_hex(&actual_payload_sha)
        ));
    }
    Ok(payload)
}

/// Open `path`, verify its `FMCENV01` header, `source_sha256`, exact byte
/// length, and `payload_sha256` via a streaming `64 KiB` chunk pass, then
/// rewind to the start of the payload (`96`-byte offset) and return
/// `(reader, payload_len)`.
///
/// Unlike [`read_verified_cache`], this verifies multi-megabyte payloads
/// without buffering an extra full-payload copy in RAM.
///
/// # Errors
/// Returns a descriptive error string naming `path` on I/O failure, truncated
/// header/payload, trailing bytes, version/tag mismatch, source hash mismatch,
/// or payload SHA-256 corruption.
pub fn open_verified_cache_streaming(
    path: &Path,
    expected_builder_version: u32,
    expected_payload_tag: &[u8; 8],
    expected_source_sha256_hex: &str,
) -> Result<(BufReader<File>, u64), String> {
    let expected_source_sha = decode_sha256_hex(expected_source_sha256_hex)
        .ok_or_else(|| {
            format!(
                "invalid expected source SHA-256 '{expected_source_sha256_hex}'"
            )
        })?;
    let (mut reader, stored_payload, payload_len) =
        open_checked_envelope_reader(
            path,
            expected_builder_version,
            *expected_payload_tag,
            Some(&expected_source_sha),
            Some(expected_source_sha256_hex),
        )?;
    let actual_hex = compute_sha256_reader(
        &mut reader.by_ref().take(payload_len),
    )
    .map_err(|e| {
        format!(
            "failed to read runtime cache payload in {}: {e}",
            path.display()
        )
    })?;
    let expected_end = (CACHE_ENVELOPE_HEADER_BYTES as u64)
        .checked_add(payload_len)
        .ok_or_else(|| {
            format!("runtime cache size overflow in {}", path.display())
        })?;
    let pos = reader.stream_position().map_err(|e| {
        format!(
            "failed to verify runtime cache position in {}: {e}",
            path.display()
        )
    })?;
    let mut trailing = [0_u8; 1];
    let trailing_read = reader.read(&mut trailing).map_err(|e| {
        format!(
            "failed to verify runtime cache EOF in {}: {e}",
            path.display()
        )
    })?;
    if pos != expected_end || trailing_read != 0 {
        return Err(format!(
            "runtime cache length mismatch in {}: expected {expected_end} bytes, got {}",
            path.display(),
            pos.saturating_add(trailing_read as u64)
        ));
    }
    let actual_payload_sha = decode_sha256_hex(&actual_hex)
        .ok_or_else(|| "internal SHA-256 hex decode failure".to_string())?;
    if actual_payload_sha != stored_payload {
        return Err(format!(
            "corrupt runtime cache payload in {} (SHA-256 mismatch: expected {}, got {})",
            path.display(),
            encode_sha256_hex(&stored_payload),
            encode_sha256_hex(&actual_payload_sha)
        ));
    }
    reader
        .seek(SeekFrom::Start(CACHE_ENVELOPE_HEADER_BYTES as u64))
        .map_err(|e| {
            format!("failed to rewind verified cache {}: {e}", path.display())
        })?;
    Ok((reader, payload_len))
}

/// Open a previously verified `FMCENV01` cache file, validate its header
/// (`magic`, `schema_version`, `builder_version`, `payload_tag`) and exact
/// file byte length, and return `(reader, payload_len)` positioned at the
/// payload start without re-hashing the payload.
///
/// Callers must only use this on cache files whose `source_sha256` and
/// `payload_sha256` have already been verified in the current process.
///
/// # Errors
/// Returns a descriptive error string naming `path` on I/O failure, truncated
/// header, magic/version/tag mismatch, or total file-length mismatch.
pub fn open_preverified_cache_streaming(
    path: &Path,
    expected_builder_version: u32,
    expected_payload_tag: &[u8; 8],
) -> Result<(BufReader<File>, u64), String> {
    let (reader, _stored_payload, payload_len) = open_checked_envelope_reader(
        path,
        expected_builder_version,
        *expected_payload_tag,
        None,
        None,
    )?;
    Ok((reader, payload_len))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256::compute_sha256_bytes;
    use crate::{
        is_runtime_cache_disabled, resolve_flexmesa_cache_dir,
        resolve_flexmesa_data_dir,
    };
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_verified_cache_roundtrip_and_corruption_detection() {
        let dir = std::env::temp_dir()
            .join(format!("flexmesa_cenv_test_{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let cache_file = dir.join("sample.bin");
        let src_sha = compute_sha256_bytes(b"upstream-raw-slice-v1");
        let payload = b"FMDT4D01_PAYLOAD_DATA";

        write_verified_cache_atomic(
            &cache_file,
            1,
            b"FMDT4D01",
            &src_sha,
            payload,
        )
        .expect("write verified cache");
        let recovered =
            read_verified_cache(&cache_file, 1, b"FMDT4D01", &src_sha)
                .expect("read verified cache");
        assert_eq!(recovered, payload);

        // Wrong source hash rejected
        let other_src = compute_sha256_bytes(b"other");
        let err = read_verified_cache(&cache_file, 1, b"FMDT4D01", &other_src)
            .expect_err("source mismatch must fail");
        assert!(err.contains("source SHA-256 mismatch"));

        // Wrong builder version rejected
        let err = read_verified_cache(&cache_file, 2, b"FMDT4D01", &src_sha)
            .expect_err("builder version mismatch must fail");
        assert!(err.contains("builder version mismatch"));

        // Wrong payload tag rejected
        let err = read_verified_cache(&cache_file, 1, b"FMCMS001", &src_sha)
            .expect_err("tag mismatch must fail");
        assert!(err.contains("payload tag mismatch"));

        // Single-bit payload flip detected
        let mut flipped = fs::read(&cache_file).expect("read raw");
        let last = flipped.len() - 1;
        flipped[last] ^= 0x01;
        fs::write(&cache_file, &flipped).expect("write flipped");
        let err = read_verified_cache(&cache_file, 1, b"FMDT4D01", &src_sha)
            .expect_err("bit flip must fail");
        assert!(err.contains("SHA-256 mismatch"));

        // Trailing bytes rejected
        write_verified_cache_atomic(
            &cache_file,
            1,
            b"FMDT4D01",
            &src_sha,
            payload,
        )
        .expect("rewrite");
        let mut extended = fs::read(&cache_file).expect("read");
        extended.push(0xAA);
        fs::write(&cache_file, &extended).expect("write extended");
        let err = read_verified_cache(&cache_file, 1, b"FMDT4D01", &src_sha)
            .expect_err("trailing byte must fail");
        assert!(err.contains("length mismatch"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_resolve_cache_dir_separate_from_data_dir_and_disable_flag() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        let prev_data = std::env::var_os("FLEXMESA_DATA_DIR");
        let prev_cache = std::env::var_os("FLEXMESA_CACHE_DIR");
        let prev_dis = std::env::var_os("FLEXMESA_DISABLE_RUNTIME_CACHE");

        let fake_data = std::env::temp_dir().join("fm_data_only");
        let fake_cache = std::env::temp_dir().join("fm_cache_root");
        std::env::set_var("FLEXMESA_DATA_DIR", &fake_data);
        std::env::remove_var("FLEXMESA_CACHE_DIR");
        std::env::remove_var("FLEXMESA_DISABLE_RUNTIME_CACHE");

        assert_eq!(resolve_flexmesa_data_dir(), fake_data);
        let cache = resolve_flexmesa_cache_dir().expect("cache enabled");
        assert!(!cache.starts_with(&fake_data));
        assert!(cache.ends_with("v1"));
        assert!(!is_runtime_cache_disabled());

        std::env::set_var("FLEXMESA_CACHE_DIR", &fake_cache);
        assert_eq!(resolve_flexmesa_cache_dir(), Some(fake_cache.join("v1")));
        std::env::set_var("FLEXMESA_CACHE_DIR", fake_cache.join("v1"));
        assert_eq!(resolve_flexmesa_cache_dir(), Some(fake_cache.join("v1")));

        // Overlapping cache_dir and data_dir must never resolve inside data_dir
        std::env::set_var("FLEXMESA_CACHE_DIR", &fake_data);
        assert_eq!(resolve_flexmesa_cache_dir(), None);
        std::env::set_var("FLEXMESA_CACHE_DIR", fake_data.join("sub"));
        assert_eq!(resolve_flexmesa_cache_dir(), None);
        std::env::set_var("FLEXMESA_CACHE_DIR", &fake_cache);
        std::env::set_var(
            "FLEXMESA_DATA_DIR",
            fake_cache.join("v1").join("data"),
        );
        assert_eq!(resolve_flexmesa_cache_dir(), None);
        std::env::set_var("FLEXMESA_DATA_DIR", &fake_data);

        std::env::set_var("FLEXMESA_DISABLE_RUNTIME_CACHE", "1");
        assert!(is_runtime_cache_disabled());
        assert_eq!(resolve_flexmesa_cache_dir(), None);

        // Restore env
        match prev_data {
            Some(v) => std::env::set_var("FLEXMESA_DATA_DIR", v),
            None => std::env::remove_var("FLEXMESA_DATA_DIR"),
        }
        match prev_cache {
            Some(v) => std::env::set_var("FLEXMESA_CACHE_DIR", v),
            None => std::env::remove_var("FLEXMESA_CACHE_DIR"),
        }
        match prev_dis {
            Some(v) => std::env::set_var("FLEXMESA_DISABLE_RUNTIME_CACHE", v),
            None => std::env::remove_var("FLEXMESA_DISABLE_RUNTIME_CACHE"),
        }
    }

    #[test]
    fn test_streaming_cache_verification_roundtrip_and_corruption_detection() {
        let dir = std::env::temp_dir()
            .join(format!("flexmesa_cenv_stream_test_{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let cache_file = dir.join("stream_sample.bin");
        let src_sha = compute_sha256_bytes(b"upstream-raw-slice-v1");
        let payload = vec![0x5A_u8; 300 * 1024 + 17];

        write_verified_cache_atomic(
            &cache_file,
            1,
            b"FMDT4D01",
            &src_sha,
            &payload,
        )
        .expect("write verified cache");

        let (mut reader, payload_len) = open_verified_cache_streaming(
            &cache_file,
            1,
            b"FMDT4D01",
            &src_sha,
        )
        .expect("streaming open verified cache");
        assert_eq!(payload_len, payload.len() as u64);
        let mut recovered = Vec::new();
        reader.read_to_end(&mut recovered).expect("read payload");
        assert_eq!(recovered, payload);

        let (mut pre_reader, pre_len) =
            open_preverified_cache_streaming(&cache_file, 1, b"FMDT4D01")
                .expect("preverified open");
        assert_eq!(pre_len, payload.len() as u64);
        let mut pre_recovered = Vec::new();
        pre_reader
            .read_to_end(&mut pre_recovered)
            .expect("read preverified");
        assert_eq!(pre_recovered, payload);

        let other_src = compute_sha256_bytes(b"other");
        let err = open_verified_cache_streaming(
            &cache_file,
            1,
            b"FMDT4D01",
            &other_src,
        )
        .expect_err("source mismatch must fail");
        assert!(err.contains("source SHA-256 mismatch"));

        // Single-bit payload flip detected during streaming hash.
        let mut flipped = fs::read(&cache_file).expect("read raw");
        let mid = CACHE_ENVELOPE_HEADER_BYTES + 128 * 1024;
        flipped[mid] ^= 0x01;
        fs::write(&cache_file, &flipped).expect("write flipped");
        let err = open_verified_cache_streaming(
            &cache_file,
            1,
            b"FMDT4D01",
            &src_sha,
        )
        .expect_err("bit flip must fail");
        assert!(err.contains("SHA-256 mismatch"));

        // Truncated header rejected.
        fs::write(&cache_file, &flipped[..16]).expect("write short");
        let err = open_verified_cache_streaming(
            &cache_file,
            1,
            b"FMDT4D01",
            &src_sha,
        )
        .expect_err("truncated header must fail");
        assert!(err.contains("truncated runtime cache header"));

        let _ = fs::remove_dir_all(&dir);
    }
}
