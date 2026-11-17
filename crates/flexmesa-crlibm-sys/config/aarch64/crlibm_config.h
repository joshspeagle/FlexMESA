/* Fixed committed crlibm configuration header for 64-bit aarch64 (little-endian, IEEE-754 FPU).
 * No autoconf runs at build time (#165).
 * Hardware x87 double-extended / FPU control-word mutation is intentionally
 * disabled (CRLIBM_HAS_FPU_CONTROL undefined) so all routines rely solely on
 * the default IEEE-754 round-to-nearest-even floating-point environment.
 */
#ifndef FLEXMESA_CRLIBM_CONFIG_H
#define FLEXMESA_CRLIBM_CONFIG_H 1

#define HAVE_FLOAT_H 1
#define HAVE_INTTYPES_H 1
#define HAVE_STDINT_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRING_H 1
#define PACKAGE "crlibm"
#define PACKAGE_NAME "crlibm"
#define PACKAGE_STRING "crlibm 1.0beta4"
#define PACKAGE_TARNAME "crlibm"
#define PACKAGE_VERSION "1.0beta4"
#define VERSION "1.0beta4"
#define SCS_NB_BITS 30
#define SCS_NB_WORDS 8
#define SIZEOF_INT 4
#define STDC_HEADERS 1

#endif /* FLEXMESA_CRLIBM_CONFIG_H */
