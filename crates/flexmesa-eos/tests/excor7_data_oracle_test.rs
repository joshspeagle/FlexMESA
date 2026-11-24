// SPDX-License-Identifier: GPL-3.0-or-later
//! Bit-level oracle verification of `Excor7Table` (`1,476,657` grid values)
//! against MESA `26.4.1`'s shipped `data/eosPC_support_data/EXCOR7.data`
//! (`#161`).

#![forbid(unsafe_code)]
#![allow(clippy::unreadable_literal)]

use std::fmt::Write as _;
use std::path::Path;

use flexmesa_eos::pc::excor_table::{
    Excor7Table, EXCOR7_N_LN_GAME, EXCOR7_N_LN_RS, NVALS_EXCOR7,
};

const VAR_NAMES: [&str; NVALS_EXCOR7] =
    ["FXC", "UXC", "PXC", "CVXC", "SXC", "PDTXC", "PDRXC"];

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

struct Sha256 {
    state: [u32; 8],
    bit_len: u64,
    buf: [u8; 64],
    buf_len: usize,
}

impl Sha256 {
    const fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f,
                0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
            ],
            bit_len: 0,
            buf: [0; 64],
            buf_len: 0,
        }
    }

    fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
        let mut sched = [0_u32; 64];
        for (idx, word) in sched.iter_mut().take(16).enumerate() {
            let off = idx * 4;
            *word = u32::from_be_bytes([
                block[off],
                block[off + 1],
                block[off + 2],
                block[off + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = sched[i - 15].rotate_right(7)
                ^ sched[i - 15].rotate_right(18)
                ^ (sched[i - 15] >> 3);
            let s1 = sched[i - 2].rotate_right(17)
                ^ sched[i - 2].rotate_right(19)
                ^ (sched[i - 2] >> 10);
            sched[i] = sched[i - 16]
                .wrapping_add(s0)
                .wrapping_add(sched[i - 7])
                .wrapping_add(s1);
        }

        let [mut va, mut vb, mut vc, mut vd, mut ve, mut vf, mut vg, mut vh] =
            *state;
        for (k_round, &w_i) in SHA256_K.iter().zip(&sched) {
            let s1 = ve.rotate_right(6) ^ ve.rotate_right(11) ^ ve.rotate_right(25);
            let ch = (ve & vf) ^ ((!ve) & vg);
            let temp1 = vh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(*k_round)
                .wrapping_add(w_i);
            let s0 = va.rotate_right(2) ^ va.rotate_right(13) ^ va.rotate_right(22);
            let maj = (va & vb) ^ (va & vc) ^ (vb & vc);
            let temp2 = s0.wrapping_add(maj);

            vh = vg;
            vg = vf;
            vf = ve;
            ve = vd.wrapping_add(temp1);
            vd = vc;
            vc = vb;
            vb = va;
            va = temp1.wrapping_add(temp2);
        }

        state[0] = state[0].wrapping_add(va);
        state[1] = state[1].wrapping_add(vb);
        state[2] = state[2].wrapping_add(vc);
        state[3] = state[3].wrapping_add(vd);
        state[4] = state[4].wrapping_add(ve);
        state[5] = state[5].wrapping_add(vf);
        state[6] = state[6].wrapping_add(vg);
        state[7] = state[7].wrapping_add(vh);
    }

    fn update(&mut self, data: &[u8]) {
        self.bit_len = self.bit_len.wrapping_add((data.len() as u64) * 8);
        let mut offset = 0;
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take]
                .copy_from_slice(&data[..take]);
            self.buf_len += take;
            offset += take;
            if self.buf_len == 64 {
                let block = self.buf;
                Self::compress(&mut self.state, &block);
                self.buf_len = 0;
            }
        }
        while offset + 64 <= data.len() {
            let mut block = [0_u8; 64];
            block.copy_from_slice(&data[offset..offset + 64]);
            Self::compress(&mut self.state, &block);
            offset += 64;
        }
        if offset < data.len() {
            let rem = data.len() - offset;
            self.buf[..rem].copy_from_slice(&data[offset..]);
            self.buf_len = rem;
        }
    }

    fn finalize_hex(mut self) -> String {
        let bit_len = self.bit_len;
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;
        if self.buf_len > 56 {
            self.buf[self.buf_len..].fill(0);
            let block = self.buf;
            Self::compress(&mut self.state, &block);
            self.buf_len = 0;
        }
        self.buf[self.buf_len..56].fill(0);
        self.buf[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buf;
        Self::compress(&mut self.state, &block);

        let mut out = String::with_capacity(64);
        for word in self.state {
            for byte in word.to_be_bytes() {
                let _ = write!(out, "{byte:02x}");
            }
        }
        out
    }
}

#[test]
fn test_excor7_table_matches_mesa_26_4_1_excor7_data_sha256_and_hex_nodes() {
    // Verify the internal SHA-256 implementation against NIST FIPS 180-4 vectors.
    let mut empty = Sha256::new();
    empty.update(b"");
    assert_eq!(
        empty.finalize_hex(),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    let mut abc = Sha256::new();
    abc.update(b"abc");
    assert_eq!(
        abc.finalize_hex(),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );

    let oracle_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/fortran_mesa/excor7_oracles/excor7_26_4_1_checksum.tsv");
    let text = std::fs::read_to_string(&oracle_path)
        .expect("read excor7_26_4_1_checksum.tsv");

    let mut expected_stream_sha: Option<String> = None;
    let mut expected_count: Option<usize> = None;
    let mut sample_rows: Vec<(usize, usize, usize, u64)> = Vec::new();
    let mut saw_header = false;

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("# ieee754_le_bytes_sha256:") {
            expected_stream_sha = Some(rest.trim().to_owned());
            continue;
        }
        if let Some(rest) = line.strip_prefix("# value_count:") {
            expected_count = Some(
                rest.trim()
                    .parse::<usize>()
                    .expect("parse value_count metadata"),
            );
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        if !saw_header {
            assert_eq!(
                line,
                "i_rs\tj_game\tlogRS\tlogGAME\tvar\tvalue\tvalue_hex"
            );
            saw_header = true;
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 7, "unexpected column count in TSV row: {line}");
        let i_rs = cols[0].parse::<usize>().expect("parse i_rs");
        let j_game = cols[1].parse::<usize>().expect("parse j_game");
        let var_idx = VAR_NAMES
            .iter()
            .position(|&name| name == cols[4])
            .expect("known EXCOR7 variable name");
        let expected_bits =
            u64::from_str_radix(cols[6], 16).expect("parse hex binary64");
        sample_rows.push((i_rs, j_game, var_idx, expected_bits));
    }

    let expected_sha =
        expected_stream_sha.expect("missing ieee754_le_bytes_sha256 metadata");
    assert_eq!(
        expected_count.expect("missing value_count metadata"),
        EXCOR7_N_LN_RS * EXCOR7_N_LN_GAME * NVALS_EXCOR7
    );
    assert_eq!(sample_rows.len(), 12 * NVALS_EXCOR7);

    let tbl = Excor7Table::global();
    for &(i_rs, j_game, var_idx, expected_bits) in &sample_rows {
        let actual_bits = tbl.node_value(i_rs, j_game, var_idx).to_bits();
        assert_eq!(
            actual_bits, expected_bits,
            "EXCOR7 node ({i_rs}, {j_game}) {} bit mismatch: actual={actual_bits:016X} expected={expected_bits:016X}",
            VAR_NAMES[var_idx]
        );
    }

    let mut hasher = Sha256::new();
    for i_rs in 0..EXCOR7_N_LN_RS {
        for j_game in 0..EXCOR7_N_LN_GAME {
            for var_idx in 0..NVALS_EXCOR7 {
                hasher.update(&tbl.node_value(i_rs, j_game, var_idx).to_le_bytes());
            }
        }
    }
    let actual_sha = hasher.finalize_hex();
    assert_eq!(
        actual_sha, expected_sha,
        "EXCOR7 1,476,657-value SHA-256 mismatch against MESA 26.4.1 EXCOR7.data"
    );
}
