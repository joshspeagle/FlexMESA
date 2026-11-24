/* libm interposition for the ADIPLS libm measurement (#171, package A0,
 * DECISIONS #60).
 *
 * Linked into the oracle driver with -Wl,--wrap=pow,--wrap=exp,... so every
 * call that the ADIPLS objects make to these functions passes through here.
 * Each wrapper evaluates both the system libm (glibc, the function MESA's
 * ADIPLS calls) and the vendored crlibm 1.0beta4 round-to-nearest function
 * that FlexMESA's strict mode uses (flexmesa_math::intrinsic_*), counts the
 * arguments where the two differ, and returns one of them:
 *
 *   ADIPLS_LIBM_RETURN=glibc   (default) return the glibc value: outputs are
 *                              those of MESA's build, the counts measure the
 *                              mismatch on MESA's own arguments;
 *   ADIPLS_LIBM_RETURN=crlibm  return the crlibm value: outputs are those of
 *                              an ADIPLS that routes these calls through
 *                              crlibm, as the FlexMESA port will.
 *
 * At exit a summary is written to ADIPLS_LIBM_LOG (if set): one line per
 * function and calling routine (resolved with dladdr; link with -rdynamic)
 * with the call count, mismatch count and largest difference in ulps, plus
 * up to ADIPLS_LIBM_SAMPLES (default 64) mismatching argument vectors per
 * function as hex.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "crlibm.h"

enum { F_POW, F_EXP, F_LOG, F_LOG10, F_SIN, F_COS, F_ATAN2, NF };
static const char *fname[NF] = {"pow", "exp", "log", "log10", "sin", "cos", "atan2"};

#define MAXSITE 256
struct site {
    void *addr;
    int f;
    unsigned long long calls, mism;
    uint64_t maxulp;
};
static struct site sites[MAXSITE];
static int nsites = 0;
static int ret_crlibm = -1;
static int nsamples_max = 64;
static int nsamples[NF];
static char samples[NF][64][80];
static int registered = 0;

double __real_pow(double, double);
double __real_exp(double);
double __real_log(double);
double __real_log10(double);
double __real_sin(double);
double __real_cos(double);
double __real_atan2(double, double);

static uint64_t bits(double v) {
    uint64_t u;
    memcpy(&u, &v, sizeof u);
    return u;
}

static uint64_t ulpdiff(double a, double b) {
    if (isnan(a) && isnan(b)) return 0;
    int64_t ia = (int64_t)bits(a), ib = (int64_t)bits(b);
    if (ia < 0) ia = INT64_MIN - ia;
    if (ib < 0) ib = INT64_MIN - ib;
    return ia > ib ? (uint64_t)(ia - ib) : (uint64_t)(ib - ia);
}

static void dump(void) {
    const char *path = getenv("ADIPLS_LIBM_LOG");
    if (!path || !*path) return;
    FILE *fp = fopen(path, "w");
    if (!fp) return;
    fprintf(fp, "# return=%s\n", ret_crlibm ? "crlibm" : "glibc");
    fprintf(fp, "SITE\tfunction\troutine\toffset\tcalls\tmismatches\tmax_ulp\n");
    for (int i = 0; i < nsites; i++) {
        Dl_info info;
        const char *sym = "?";
        long off = 0;
        if (dladdr(sites[i].addr, &info) && info.dli_sname) {
            sym = info.dli_sname;
            off = (char *)sites[i].addr - (char *)info.dli_saddr;
        }
        fprintf(fp, "SITE\t%s\t%s\t%ld\t%llu\t%llu\t%llu\n", fname[sites[i].f], sym, off,
                sites[i].calls, sites[i].mism, (unsigned long long)sites[i].maxulp);
    }
    for (int f = 0; f < NF; f++)
        for (int k = 0; k < nsamples[f]; k++) fprintf(fp, "SAMPLE\t%s\t%s\n", fname[f], samples[f][k]);
    fclose(fp);
}

static void setup(void) {
    if (ret_crlibm >= 0) return;
    const char *r = getenv("ADIPLS_LIBM_RETURN");
    ret_crlibm = (r && strcmp(r, "crlibm") == 0) ? 1 : 0;
    const char *s = getenv("ADIPLS_LIBM_SAMPLES");
    if (s) nsamples_max = atoi(s) > 64 ? 64 : atoi(s);
    crlibm_init();
    if (!registered) {
        atexit(dump);
        registered = 1;
    }
}

/* Register the summary at start-up so that runs without any libm call (the
 * radial-only calls of astero_adipls) still write an (empty) summary. */
__attribute__((constructor)) static void setup_at_load(void) { setup(); }

static double record(int f, void *addr, double vg, double vc, double x, double y, int two) {
    int i;
    for (i = 0; i < nsites; i++)
        if (sites[i].addr == addr && sites[i].f == f) break;
    if (i == nsites && nsites < MAXSITE) {
        sites[i].addr = addr;
        sites[i].f = f;
        nsites++;
    }
    int differ = bits(vg) != bits(vc) && !(isnan(vg) && isnan(vc));
    if (i < MAXSITE) {
        sites[i].calls++;
        if (differ) {
            sites[i].mism++;
            uint64_t d = ulpdiff(vg, vc);
            if (d > sites[i].maxulp) sites[i].maxulp = d;
        }
    }
    if (differ && nsamples[f] < nsamples_max) {
        if (two)
            snprintf(samples[f][nsamples[f]], 80, "%016llX %016llX %016llX %016llX",
                     (unsigned long long)bits(x), (unsigned long long)bits(y),
                     (unsigned long long)bits(vg), (unsigned long long)bits(vc));
        else
            snprintf(samples[f][nsamples[f]], 80, "%016llX %016llX %016llX",
                     (unsigned long long)bits(x), (unsigned long long)bits(vg),
                     (unsigned long long)bits(vc));
        nsamples[f]++;
    }
    return ret_crlibm ? vc : vg;
}

#define WRAP1(NAME, F, CR)                                                     \
    double __wrap_##NAME(double x) {                                           \
        setup();                                                               \
        double vg = __real_##NAME(x);                                          \
        double vc = CR(x);                                                     \
        return record(F, __builtin_return_address(0), vg, vc, x, 0.0, 0);      \
    }

WRAP1(exp, F_EXP, exp_rn)
WRAP1(log, F_LOG, log_rn)
WRAP1(log10, F_LOG10, log10_rn)
WRAP1(sin, F_SIN, sin_rn)
WRAP1(cos, F_COS, cos_rn)

double __wrap_pow(double x, double y) {
    setup();
    double vg = __real_pow(x, y);
    double vc = pow_rn(x, y);
    return record(F_POW, __builtin_return_address(0), vg, vc, x, y, 1);
}

/* crlibm has no atan2. ADIPLS calls it only in eiginh (mdintg = 2), which no
 * MESA test case reaches; count the calls and return glibc's value. */
double __wrap_atan2(double y, double x) {
    setup();
    double vg = __real_atan2(y, x);
    return record(F_ATAN2, __builtin_return_address(0), vg, vg, y, x, 1);
}
