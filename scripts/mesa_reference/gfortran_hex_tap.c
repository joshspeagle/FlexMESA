#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <quadmath.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* GCC 8..15 libgfortran ABI (GFORTRAN_8) I/O structures */
#define IOPARM_DT_HAS_INTERNAL_UNIT (1 << 14)
#define GFC_BT_REAL    3
#define GFC_BT_COMPLEX 4

typedef struct {
    size_t elem_len;
    int version;
    signed char rank;
    signed char type;
    short attribute;
} gfc_dtype_t;

typedef struct {
    ptrdiff_t stride;
    ptrdiff_t lbound;
    ptrdiff_t ubound;
} gfc_dim_t;

typedef struct {
    void *base_addr;
    size_t offset;
    gfc_dtype_t dtype;
    ptrdiff_t span;
    gfc_dim_t dim[15];
} gfc_array_desc_t;

static pthread_mutex_t tap_lock = PTHREAD_MUTEX_INITIALIZER;
static FILE *tap_fp = NULL;
static int tap_initialized = 0;
static int target_unit = 6;
static long long global_record_idx = 0;

static __thread int tls_active = 0;
static __thread int tls_in_hook = 0;
static __thread long long tls_rec_idx = 0;
static __thread int tls_val_idx = 0;
static __thread const char *tls_src_file = "";
static __thread int tls_src_line = 0;
static __thread char tls_label[160];
static __thread size_t tls_label_len = 0;

static void init_tap_if_needed(void) {
    if (tap_initialized) return;
    tap_initialized = 1;
    const char *unit_env = getenv("HEX_TAP_UNIT");
    if (unit_env && *unit_env) {
        target_unit = atoi(unit_env);
    }
    const char *out_path = getenv("HEX_TAP_OUT");
    if (out_path && *out_path) {
        tap_fp = fopen(out_path, "a");
        if (tap_fp) {
            fseek(tap_fp, 0, SEEK_END);
            if (ftell(tap_fp) == 0) {
                fprintf(tap_fp,
                    "record_idx\tval_idx\tsrc_loc\tlabel\tkind\t"
                    "decimal\thex_float\traw_bits_hex\n");
                fflush(tap_fp);
            }
        }
    }
}

static const char *short_basename(const char *p) {
    if (!p) return "";
    const char *slash = strrchr(p, '/');
    return slash ? (slash + 1) : p;
}

static void append_label_bytes(const char *s, size_t n) {
    if (!s || n == 0 || tls_label_len >= sizeof(tls_label) - 2) return;
    /* Trim leading/trailing spaces of this token */
    size_t start = 0;
    while (start < n && (s[start] == ' ' || s[start] == '\t' || s[start] == '\r' || s[start] == '\n')) {
        start++;
    }
    size_t end = n;
    while (end > start && (s[end - 1] == ' ' || s[end - 1] == '\t' || s[end - 1] == '\r' || s[end - 1] == '\n')) {
        end--;
    }
    if (end <= start) return;
    if (tls_label_len > 0 && tls_label_len < sizeof(tls_label) - 2) {
        tls_label[tls_label_len++] = ' ';
    }
    for (size_t i = start; i < end && tls_label_len < sizeof(tls_label) - 1; i++) {
        char c = s[i];
        if (c == '\t' || c == '\n' || c == '\r' || (unsigned char)c < 32) c = ' ';
        tls_label[tls_label_len++] = c;
    }
    tls_label[tls_label_len] = '\0';
}

static void emit_real_scalar(const void *p, int kind, const char *tag_suffix) {
    if (!tap_fp || !p) return;
    char kind_col[32];
    char dec_col[80];
    char hex_col[96];
    char bits_col[64];

    if (kind == 8) {
        double v = 0.0;
        uint64_t b = 0;
        memcpy(&v, p, 8);
        memcpy(&b, p, 8);
        snprintf(kind_col, sizeof(kind_col), "f64%s", tag_suffix);
        snprintf(dec_col, sizeof(dec_col), "%.17e", v);
        snprintf(hex_col, sizeof(hex_col), "%a", v);
        snprintf(bits_col, sizeof(bits_col), "0x%016llx", (unsigned long long)b);
    } else if (kind == 4) {
        float v = 0.0f;
        uint32_t b = 0;
        memcpy(&v, p, 4);
        memcpy(&b, p, 4);
        snprintf(kind_col, sizeof(kind_col), "f32%s", tag_suffix);
        snprintf(dec_col, sizeof(dec_col), "%.9e", (double)v);
        snprintf(hex_col, sizeof(hex_col), "%a", (double)v);
        snprintf(bits_col, sizeof(bits_col), "0x%08x", b);
    } else if (kind == 16) {
        __float128 q = 0;
        uint64_t lo = 0, hi = 0;
        memcpy(&q, p, 16);
        memcpy(&lo, (const char *)p, 8);
        memcpy(&hi, (const char *)p + 8, 8);
        snprintf(kind_col, sizeof(kind_col), "f128%s", tag_suffix);
        quadmath_snprintf(dec_col, sizeof(dec_col), "%.36Qe", q);
        quadmath_snprintf(hex_col, sizeof(hex_col), "%Qa", q);
        snprintf(bits_col, sizeof(bits_col), "0x%016llx%016llx",
                 (unsigned long long)hi, (unsigned long long)lo);
    } else if (kind == 10) {
        long double ld = 0.0L;
        unsigned char raw[10] = {0};
        memcpy(&ld, p, sizeof(long double));
        memcpy(raw, p, 10);
        snprintf(kind_col, sizeof(kind_col), "f80%s", tag_suffix);
        snprintf(dec_col, sizeof(dec_col), "%.21Le", ld);
        snprintf(hex_col, sizeof(hex_col), "%La", ld);
        snprintf(bits_col, sizeof(bits_col),
                 "0x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x",
                 raw[9], raw[8], raw[7], raw[6], raw[5],
                 raw[4], raw[3], raw[2], raw[1], raw[0]);
    } else {
        return;
    }

    tls_val_idx++;
    pthread_mutex_lock(&tap_lock);
    fprintf(tap_fp, "%lld\t%d\t%s:%d\t%s\t%s\t%s\t%s\t%s\n",
            tls_rec_idx, tls_val_idx,
            short_basename(tls_src_file), tls_src_line,
            tls_label_len ? tls_label : "-",
            kind_col, dec_col, hex_col, bits_col);
    pthread_mutex_unlock(&tap_lock);
}

static void emit_array_desc(const gfc_array_desc_t *desc, int kind) {
    if (!tap_fp || !desc || !desc->base_addr) return;
    int rank = (int)desc->dtype.rank;
    int btype = (int)desc->dtype.type;
    if (rank < 1 || rank > 15) return;
    if (btype != GFC_BT_REAL && btype != GFC_BT_COMPLEX) return;

    ptrdiff_t extents[15];
    ptrdiff_t strides[15];
    ptrdiff_t idx[15];
    size_t total_elems = 1;
    for (int d = 0; d < rank; d++) {
        ptrdiff_t ext = desc->dim[d].ubound - desc->dim[d].lbound + 1;
        if (ext <= 0) return;
        extents[d] = ext;
        strides[d] = desc->dim[d].stride;
        idx[d] = 0;
        total_elems *= (size_t)ext;
    }
    ptrdiff_t elem_span = desc->span ? desc->span : (ptrdiff_t)desc->dtype.elem_len;
    const char *base = (const char *)desc->base_addr;

    for (size_t n = 0; n < total_elems; n++) {
        ptrdiff_t off_elems = 0;
        for (int d = 0; d < rank; d++) {
            off_elems += idx[d] * strides[d];
        }
        const char *elem_ptr = base + off_elems * elem_span;
        if (btype == GFC_BT_REAL) {
            emit_real_scalar(elem_ptr, kind, "_arr");
        } else if (btype == GFC_BT_COMPLEX) {
            emit_real_scalar(elem_ptr, kind, "_cmplx_re");
            emit_real_scalar(elem_ptr + kind, kind, "_cmplx_im");
        }
        for (int d = 0; d < rank; d++) {
            idx[d]++;
            if (idx[d] < extents[d]) break;
            idx[d] = 0;
        }
    }
}

void _gfortran_st_write(void *dtp) {
    static void (*real_fn)(void *) = NULL;
    if (!real_fn) real_fn = dlsym(RTLD_NEXT, "_gfortran_st_write");

    pthread_mutex_lock(&tap_lock);
    init_tap_if_needed();
    pthread_mutex_unlock(&tap_lock);

    int32_t flags = *(const int32_t *)((const char *)dtp + 0);
    int32_t unit  = *(const int32_t *)((const char *)dtp + 4);
    if (tap_fp && (flags & IOPARM_DT_HAS_INTERNAL_UNIT) == 0 && unit == target_unit) {
        pthread_mutex_lock(&tap_lock);
        global_record_idx++;
        tls_rec_idx = global_record_idx;
        pthread_mutex_unlock(&tap_lock);
        tls_active = 1;
        tls_val_idx = 0;
        tls_src_file = *(const char *const *)((const char *)dtp + 8);
        tls_src_line = *(const int32_t *)((const char *)dtp + 16);
        tls_label[0] = '\0';
        tls_label_len = 0;
    } else {
        tls_active = 0;
    }
    real_fn(dtp);
}

void _gfortran_st_write_done(void *dtp) {
    static void (*real_fn)(void *) = NULL;
    if (!real_fn) real_fn = dlsym(RTLD_NEXT, "_gfortran_st_write_done");
    real_fn(dtp);
    if (tls_active && tap_fp) {
        pthread_mutex_lock(&tap_lock);
        fflush(tap_fp);
        pthread_mutex_unlock(&tap_lock);
    }
    tls_active = 0;
}

void _gfortran_transfer_character_write(void *dtp, const char *p, size_t len) {
    static void (*real_fn)(void *, const char *, size_t) = NULL;
    if (!real_fn) real_fn = dlsym(RTLD_NEXT, "_gfortran_transfer_character_write");
    if (tls_active && !tls_in_hook) {
        tls_in_hook = 1;
        append_label_bytes(p, len);
        real_fn(dtp, p, len);
        tls_in_hook = 0;
        return;
    }
    real_fn(dtp, p, len);
}

void _gfortran_transfer_character(void *dtp, void *p, size_t len) {
    static void (*real_fn)(void *, void *, size_t) = NULL;
    if (!real_fn) real_fn = dlsym(RTLD_NEXT, "_gfortran_transfer_character");
    if (tls_active && !tls_in_hook) {
        tls_in_hook = 1;
        append_label_bytes((const char *)p, len);
        real_fn(dtp, p, len);
        tls_in_hook = 0;
        return;
    }
    real_fn(dtp, p, len);
}

void _gfortran_transfer_real_write(void *dtp, const void *p, int kind) {
    static void (*real_fn)(void *, const void *, int) = NULL;
    if (!real_fn) real_fn = dlsym(RTLD_NEXT, "_gfortran_transfer_real_write");
    if (tls_active && !tls_in_hook) {
        tls_in_hook = 1;
        emit_real_scalar(p, kind, "");
        real_fn(dtp, p, kind);
        tls_in_hook = 0;
        return;
    }
    real_fn(dtp, p, kind);
}

void _gfortran_transfer_real(void *dtp, void *p, int kind) {
    static void (*real_fn)(void *, void *, int) = NULL;
    if (!real_fn) real_fn = dlsym(RTLD_NEXT, "_gfortran_transfer_real");
    if (tls_active && !tls_in_hook) {
        tls_in_hook = 1;
        emit_real_scalar(p, kind, "");
        real_fn(dtp, p, kind);
        tls_in_hook = 0;
        return;
    }
    real_fn(dtp, p, kind);
}

void _gfortran_transfer_real128_write(void *dtp, const void *p, int kind) {
    static void (*real_fn)(void *, const void *, int) = NULL;
    if (!real_fn) real_fn = dlsym(RTLD_NEXT, "_gfortran_transfer_real128_write");
    if (tls_active && !tls_in_hook) {
        tls_in_hook = 1;
        emit_real_scalar(p, 16, "");
        real_fn(dtp, p, kind);
        tls_in_hook = 0;
        return;
    }
    real_fn(dtp, p, kind);
}

void _gfortran_transfer_complex_write(void *dtp, const void *p, int kind) {
    static void (*real_fn)(void *, const void *, int) = NULL;
    if (!real_fn) real_fn = dlsym(RTLD_NEXT, "_gfortran_transfer_complex_write");
    if (tls_active && p && !tls_in_hook) {
        tls_in_hook = 1;
        emit_real_scalar(p, kind, "_cmplx_re");
        emit_real_scalar((const char *)p + kind, kind, "_cmplx_im");
        real_fn(dtp, p, kind);
        tls_in_hook = 0;
        return;
    }
    real_fn(dtp, p, kind);
}

void _gfortran_transfer_complex(void *dtp, void *p, int kind) {
    static void (*real_fn)(void *, void *, int) = NULL;
    if (!real_fn) real_fn = dlsym(RTLD_NEXT, "_gfortran_transfer_complex");
    if (tls_active && p && !tls_in_hook) {
        tls_in_hook = 1;
        emit_real_scalar(p, kind, "_cmplx_re");
        emit_real_scalar((const char *)p + kind, kind, "_cmplx_im");
        real_fn(dtp, p, kind);
        tls_in_hook = 0;
        return;
    }
    real_fn(dtp, p, kind);
}

void _gfortran_transfer_array_write(void *dtp, void *desc, int kind, size_t charlen) {
    static void (*real_fn)(void *, void *, int, size_t) = NULL;
    if (!real_fn) real_fn = dlsym(RTLD_NEXT, "_gfortran_transfer_array_write");
    if (tls_active && !tls_in_hook) {
        tls_in_hook = 1;
        emit_array_desc((const gfc_array_desc_t *)desc, kind);
        real_fn(dtp, desc, kind, charlen);
        tls_in_hook = 0;
        return;
    }
    real_fn(dtp, desc, kind, charlen);
}

void _gfortran_transfer_array(void *dtp, void *desc, int kind, size_t charlen) {
    static void (*real_fn)(void *, void *, int, size_t) = NULL;
    if (!real_fn) real_fn = dlsym(RTLD_NEXT, "_gfortran_transfer_array");
    if (tls_active && !tls_in_hook) {
        tls_in_hook = 1;
        emit_array_desc((const gfc_array_desc_t *)desc, kind);
        real_fn(dtp, desc, kind, charlen);
        tls_in_hook = 0;
        return;
    }
    real_fn(dtp, desc, kind, charlen);
}
