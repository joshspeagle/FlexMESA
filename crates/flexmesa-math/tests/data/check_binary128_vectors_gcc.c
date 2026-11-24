// SPDX-License-Identifier: GPL-3.0-or-later
// Replays binary128_vectors.tsv against GCC's `__float128`, whose arithmetic
// on x86_64 is libgcc's soft-fp (`__addtf3`, `__subtf3`, `__multf3`,
// `__divtf3`, `__extenddftf2`, `__trunctfdf2`, `__powitf2`), the same routines
// gfortran calls for `real(16)`.
//
// Build and run (no optimisation, so nothing is folded at compile time):
//   gcc -O0 -o check_binary128_vectors_gcc check_binary128_vectors_gcc.c
//   ./check_binary128_vectors_gcc binary128_vectors.tsv
// Exit status is 0 when every vector agrees.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned __int128 u128;
extern __float128 __powitf2(__float128, int);

static u128 parse128(const char *s) {
  u128 v = 0;
  for (; *s; s++) {
    int d = (*s <= '9') ? *s - '0' : *s - 'a' + 10;
    v = (v << 4) | (u128)d;
  }
  return v;
}
static __float128 q_of(u128 b) { __float128 q; memcpy(&q, &b, 16); return q; }
static u128 b_of(__float128 q) { u128 b; memcpy(&b, &q, 16); return b; }
static int is_nan128(u128 b) {
  return (b & ~((u128)1 << 127)) > ((u128)0x7fff << 112);
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  FILE *f = fopen(argv[1], "r");
  if (!f) return 2;
  char line[512], op[16], a[64], b[64], r[64];
  long n = 0, bad = 0;
  while (fgets(line, sizeof line, f)) {
    if (line[0] == '#') continue;
    if (sscanf(line, "%15s %63s %63s %63s", op, a, b, r) != 4) return 2;
    n++;
    int want_nan = strcmp(r, "nan") == 0;
    int ok;
    if (strcmp(op, "to_f64") == 0) {
      volatile __float128 x = q_of(parse128(a));
      volatile double d = (double)x;
      uint64_t got; memcpy(&got, (void *)&d, 8);
      ok = want_nan ? (d != d) : got == (uint64_t)parse128(r);
    } else {
      u128 got;
      if (strcmp(op, "from_f64") == 0) {
        uint64_t db = (uint64_t)parse128(a);
        volatile double d; memcpy((void *)&d, &db, 8);
        got = b_of((__float128)d);
      } else if (strcmp(op, "powi") == 0) {
        got = b_of(__powitf2(q_of(parse128(a)), atoi(b)));
      } else {
        volatile __float128 x = q_of(parse128(a)), y = q_of(parse128(b));
        volatile __float128 z;
        switch (op[0]) {
          case 'a': z = x + y; break;
          case 's': z = x - y; break;
          case 'm': z = x * y; break;
          default: z = x / y; break;
        }
        got = b_of(z);
      }
      ok = want_nan ? is_nan128(got) : got == parse128(r);
    }
    if (!ok) { bad++; if (bad <= 10) fprintf(stderr, "MISMATCH: %s", line); }
  }
  printf("%ld vectors, %ld mismatches\n", n, bad);
  return bad != 0;
}
