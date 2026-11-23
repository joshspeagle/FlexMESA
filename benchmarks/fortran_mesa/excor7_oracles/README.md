# MESA 26.4.1 `EXCOR7.data` Checksum & Reference Oracle (`#161`)

Generates `excor7_26_4_1_checksum.tsv` directly from MESA `26.4.1`'s shipped
`data/eosPC_support_data/EXCOR7.data` table (`351 x 601 x 7 = 1,476,657`
double-precision values).

## Usage

```bash
python benchmarks/fortran_mesa/excor7_oracles/generate_excor7_checksum.py \
  --mesa-dir /path/to/mesa-26.04.1
```

## Schema (`excor7_26_4_1_checksum.tsv`)

- Metadata headers (`# key: value`) record `mesa_version`, `compiler`,
  `mesa_sdk_version`, `source`, `excor7_data_sha256`,
  `ieee754_le_bytes_sha256` (SHA-256 over the `1,476,657 * 8` little-endian
  `binary64` bytes in `(i_rs, j_game, var)` order), and `value_count`.
- Tabular rows emit `12` representative `(i_rs, j_game)` nodes (table corners
  plus interior grid points) with `logRS`, `logGAME`, `var`, decimal `value`,
  and `value_hex` (big-endian IEEE-754 `binary64` bit pattern).
