// SPDX-License-Identifier: GPL-3.0-or-later
//! Pure Safe Rust RFC 1950 (`zlib`) and RFC 1951 (`DEFLATE`) decompressor.
//!
//! Used by [`crate::aesopus_h5`] to read gzip-compressed HDF5 dataset chunks
//! without external C libraries or `unsafe` code.

use crate::types::KapError;

const MAX_BITS: usize = 15;
const FAST_BITS: usize = 9;
const FAST_SIZE: usize = 1 << FAST_BITS;
const FAST_MASK: u32 = (FAST_SIZE as u32) - 1;

const CODE_LEN_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59,
    67, 83, 99, 115, 131, 163, 195, 227, 258,
];

const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5,
    5, 5, 5, 0,
];

const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513,
    769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10,
    11, 11, 12, 12, 13, 13,
];

fn err(msg: impl Into<String>) -> KapError {
    KapError::TableIo(msg.into())
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_buf: u64,
    bit_cnt: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit_buf: 0,
            bit_cnt: 0,
        }
    }

    #[inline]
    fn fill(&mut self, needed: u32) {
        while self.bit_cnt < needed && self.pos < self.data.len() {
            self.bit_buf |= u64::from(self.data[self.pos]) << self.bit_cnt;
            self.pos += 1;
            self.bit_cnt += 8;
        }
    }

    #[inline]
    fn read_bits(&mut self, n: u32) -> Result<u32, KapError> {
        if n == 0 {
            return Ok(0);
        }
        self.fill(n);
        if self.bit_cnt < n {
            return Err(err("unexpected EOF inside DEFLATE bitstream"));
        }
        let mask = (1_u64 << n) - 1;
        let val = (self.bit_buf & mask) as u32;
        self.bit_buf >>= n;
        self.bit_cnt -= n;
        Ok(val)
    }

    fn align_byte(&mut self) {
        let drop = self.bit_cnt & 7;
        self.bit_buf >>= drop;
        self.bit_cnt -= drop;
    }

    fn read_aligned_bytes(&mut self, n: usize) -> Result<Vec<u8>, KapError> {
        self.align_byte();
        let mut out = Vec::with_capacity(n);
        while out.len() < n && self.bit_cnt >= 8 {
            out.push((self.bit_buf & 0xff) as u8);
            self.bit_buf >>= 8;
            self.bit_cnt -= 8;
        }
        let rem = n - out.len();
        if rem > 0 {
            if self.pos + rem > self.data.len() {
                return Err(err("unexpected EOF in stored DEFLATE block"));
            }
            out.extend_from_slice(&self.data[self.pos..self.pos + rem]);
            self.pos += rem;
        }
        Ok(out)
    }

    fn remaining_aligned_bytes(&mut self) -> Result<&'a [u8], KapError> {
        self.align_byte();
        let buffered_bytes = (self.bit_cnt / 8) as usize;
        if buffered_bytes > self.pos {
            return Err(err("corrupt DEFLATE bit buffer offset"));
        }
        let start = self.pos - buffered_bytes;
        self.bit_buf = 0;
        self.bit_cnt = 0;
        self.pos = self.data.len();
        Ok(&self.data[start..])
    }
}

struct HuffmanTable {
    /// Direct lookup for codes up to `FAST_BITS`: `(symbol << 5) | bit_len`, or `0` if > `FAST_BITS` or invalid.
    fast: [u32; FAST_SIZE],
    /// Canonical first code for each bit length `1..=MAX_BITS`.
    first_code: [u32; MAX_BITS + 1],
    /// Offset into `symbols` for each bit length `1..=MAX_BITS`.
    first_sym: [usize; MAX_BITS + 1],
    /// Number of symbols at each bit length `1..=MAX_BITS`.
    counts: [u32; MAX_BITS + 1],
    /// Symbols sorted by `(bit_len, symbol)`.
    symbols: Vec<u16>,
}

fn reverse_bits(mut code: u32, len: u8) -> u32 {
    let mut rev = 0_u32;
    for _ in 0..len {
        rev = (rev << 1) | (code & 1);
        code >>= 1;
    }
    rev
}

impl HuffmanTable {
    fn from_lengths(lengths: &[u8]) -> Result<Self, KapError> {
        let mut counts = [0_u32; MAX_BITS + 1];
        for &len in lengths {
            let l = usize::from(len);
            if l > MAX_BITS {
                return Err(err("DEFLATE Huffman code length exceeds 15"));
            }
            if l > 0 {
                counts[l] += 1;
            }
        }

        let mut first_code = [0_u32; MAX_BITS + 1];
        let mut first_sym = [0_usize; MAX_BITS + 1];
        let mut code = 0_u32;
        let mut total_syms = 0_usize;
        for bits in 1..=MAX_BITS {
            code = (code + counts[bits - 1]) << 1;
            first_code[bits] = code;
            first_sym[bits] = total_syms;
            total_syms += counts[bits] as usize;
            if code + counts[bits] > (1_u32 << bits) {
                return Err(err("oversubscribed DEFLATE Huffman tree"));
            }
        }

        let mut next_offset = first_sym;
        let mut next_code = first_code;
        let mut symbols = vec![0_u16; total_syms];
        let mut fast = [0_u32; FAST_SIZE];

        for (sym, &len) in lengths.iter().enumerate() {
            let l = usize::from(len);
            if l == 0 {
                continue;
            }
            let c = next_code[l];
            next_code[l] += 1;
            let idx = next_offset[l];
            next_offset[l] += 1;
            symbols[idx] = sym as u16;

            if l <= FAST_BITS {
                let rev = reverse_bits(c, len) as usize;
                let step = 1_usize << l;
                let entry = ((sym as u32) << 5) | u32::from(len);
                let mut slot = rev;
                while slot < FAST_SIZE {
                    fast[slot] = entry;
                    slot += step;
                }
            }
        }

        Ok(Self {
            fast,
            first_code,
            first_sym,
            counts,
            symbols,
        })
    }

    #[inline]
    fn decode(&self, reader: &mut BitReader<'_>) -> Result<u16, KapError> {
        reader.fill(MAX_BITS as u32);
        if reader.bit_cnt > 0 {
            let entry = self.fast[(reader.bit_buf as u32 & FAST_MASK) as usize];
            if entry != 0 {
                let len = entry & 0x1f;
                if reader.bit_cnt >= len {
                    reader.bit_buf >>= len;
                    reader.bit_cnt -= len;
                    return Ok((entry >> 5) as u16);
                }
            }
        }
        let mut code = 0_u32;
        for bits in 1..=MAX_BITS {
            let bit = reader.read_bits(1)?;
            code = (code << 1) | bit;
            let cnt = self.counts[bits];
            if cnt > 0 {
                let fc = self.first_code[bits];
                if code >= fc && code - fc < cnt {
                    let sym_idx = self.first_sym[bits] + (code - fc) as usize;
                    return Ok(self.symbols[sym_idx]);
                }
            }
        }
        Err(err("invalid DEFLATE Huffman symbol"))
    }
}

fn fixed_tables() -> Result<(HuffmanTable, HuffmanTable), KapError> {
    let mut lit_lens = [0_u8; 288];
    for len in &mut lit_lens[0..=143] {
        *len = 8;
    }
    for len in &mut lit_lens[144..=255] {
        *len = 9;
    }
    for len in &mut lit_lens[256..=279] {
        *len = 7;
    }
    for len in &mut lit_lens[280..=287] {
        *len = 8;
    }
    let dist_lens = [5_u8; 32];
    Ok((
        HuffmanTable::from_lengths(&lit_lens)?,
        HuffmanTable::from_lengths(&dist_lens)?,
    ))
}

fn decode_dynamic_tables(
    reader: &mut BitReader<'_>,
) -> Result<(HuffmanTable, HuffmanTable), KapError> {
    let hlit = (reader.read_bits(5)? as usize) + 257;
    let hdist = (reader.read_bits(5)? as usize) + 1;
    let hclen = (reader.read_bits(4)? as usize) + 4;
    if hlit > 286 || hdist > 30 {
        return Err(err("invalid dynamic DEFLATE header counts"));
    }

    let mut cl_lens = [0_u8; 19];
    for &idx in CODE_LEN_ORDER.iter().take(hclen) {
        cl_lens[idx] = reader.read_bits(3)? as u8;
    }
    let cl_table = HuffmanTable::from_lengths(&cl_lens)?;

    let total = hlit + hdist;
    let mut all_lens = Vec::with_capacity(total);
    while all_lens.len() < total {
        let sym = cl_table.decode(reader)?;
        match sym {
            0..=15 => all_lens.push(sym as u8),
            16 => {
                let Some(&prev) = all_lens.last() else {
                    return Err(err(
                        "DEFLATE repeat code 16 with no preceding length",
                    ));
                };
                let rep = (reader.read_bits(2)? as usize) + 3;
                if all_lens.len() + rep > total {
                    return Err(err("DEFLATE code length overflow (sym 16)"));
                }
                all_lens.resize(all_lens.len() + rep, prev);
            }
            17 => {
                let rep = (reader.read_bits(3)? as usize) + 3;
                if all_lens.len() + rep > total {
                    return Err(err("DEFLATE code length overflow (sym 17)"));
                }
                all_lens.resize(all_lens.len() + rep, 0);
            }
            18 => {
                let rep = (reader.read_bits(7)? as usize) + 11;
                if all_lens.len() + rep > total {
                    return Err(err("DEFLATE code length overflow (sym 18)"));
                }
                all_lens.resize(all_lens.len() + rep, 0);
            }
            _ => return Err(err("invalid DEFLATE code-length symbol")),
        }
    }
    let lit_table = HuffmanTable::from_lengths(&all_lens[..hlit])?;
    let dist_table = HuffmanTable::from_lengths(&all_lens[hlit..])?;
    Ok((lit_table, dist_table))
}

fn decode_huffman_block(
    reader: &mut BitReader<'_>,
    lit_table: &HuffmanTable,
    dist_table: &HuffmanTable,
    out: &mut Vec<u8>,
) -> Result<(), KapError> {
    loop {
        let sym = lit_table.decode(reader)?;
        match sym.cmp(&256) {
            std::cmp::Ordering::Less => out.push(sym as u8),
            std::cmp::Ordering::Equal => return Ok(()),
            std::cmp::Ordering::Greater => {
                let len_idx = usize::from(sym) - 257;
                if len_idx >= LENGTH_BASE.len() {
                    return Err(err(format!(
                        "invalid DEFLATE length symbol {sym}"
                    )));
                }
                let length = usize::from(LENGTH_BASE[len_idx])
                    + (reader.read_bits(u32::from(LENGTH_EXTRA[len_idx]))?
                        as usize);
                let dist_sym = usize::from(dist_table.decode(reader)?);
                if dist_sym >= DIST_BASE.len() {
                    return Err(err(format!(
                        "invalid DEFLATE distance symbol {dist_sym}"
                    )));
                }
                let distance = usize::from(DIST_BASE[dist_sym])
                    + (reader.read_bits(u32::from(DIST_EXTRA[dist_sym]))?
                        as usize);
                if distance == 0 || distance > out.len() {
                    return Err(err(format!(
                        "DEFLATE back-reference distance {distance} exceeds output length {}",
                        out.len()
                    )));
                }
                out.reserve(length);
                for _ in 0..length {
                    let b = out[out.len() - distance];
                    out.push(b);
                }
            }
        }
    }
}

fn adler32(bytes: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65521;
    let mut s1 = 1_u32;
    let mut s2 = 0_u32;
    for chunk in bytes.chunks(5552) {
        for &b in chunk {
            s1 += u32::from(b);
            s2 += s1;
        }
        s1 %= MOD_ADLER;
        s2 %= MOD_ADLER;
    }
    (s2 << 16) | s1
}

/// Decompresses an RFC 1950 `zlib`-wrapped `DEFLATE` stream and verifies its
/// trailing Adler-32 checksum.
///
/// # Errors
/// Returns [`KapError::TableIo`] if the zlib header, DEFLATE stream, or
/// Adler-32 checksum is invalid.
pub(crate) fn zlib_decompress(
    input: &[u8],
    expected_capacity: usize,
) -> Result<Vec<u8>, KapError> {
    if input.len() < 6 {
        return Err(err("truncated zlib stream"));
    }
    let cmf = u16::from(input[0]);
    let flg = u16::from(input[1]);
    if (cmf & 0x0f) != 8 {
        return Err(err(format!(
            "unsupported zlib compression method {}",
            cmf & 0x0f
        )));
    }
    if ((cmf << 8) | flg) % 31 != 0 {
        return Err(err("invalid zlib header checksum"));
    }
    if (flg & 0x20) != 0 {
        return Err(err("preset zlib dictionary (FDICT) is not supported"));
    }

    let mut reader = BitReader::new(&input[2..]);
    let mut out = Vec::with_capacity(expected_capacity);

    loop {
        let bfinal = reader.read_bits(1)?;
        let btype = reader.read_bits(2)?;
        match btype {
            0 => {
                let hdr = reader.read_aligned_bytes(4)?;
                let len = u16::from_le_bytes([hdr[0], hdr[1]]);
                let nlen = u16::from_le_bytes([hdr[2], hdr[3]]);
                if len != !nlen {
                    return Err(err("corrupt stored DEFLATE block length"));
                }
                let raw = reader.read_aligned_bytes(usize::from(len))?;
                out.extend_from_slice(&raw);
            }
            1 => {
                let (lt, dt) = fixed_tables()?;
                decode_huffman_block(&mut reader, &lt, &dt, &mut out)?;
            }
            2 => {
                let (lt, dt) = decode_dynamic_tables(&mut reader)?;
                decode_huffman_block(&mut reader, &lt, &dt, &mut out)?;
            }
            _ => return Err(err("reserved DEFLATE block type 3")),
        }
        if bfinal != 0 {
            break;
        }
    }

    let trailer = reader.remaining_aligned_bytes()?;
    if trailer.len() < 4 {
        return Err(err("missing trailing zlib Adler-32 checksum"));
    }
    let expected_adler =
        u32::from_be_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]);
    let actual_adler = adler32(&out);
    if actual_adler != expected_adler {
        return Err(err(format!(
            "zlib Adler-32 mismatch: expected {expected_adler:#010x}, got {actual_adler:#010x}"
        )));
    }
    Ok(out)
}
