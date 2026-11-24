/* Tap writer for the ADIPLS intermediate oracles (#171, package A0).
 *
 * a0trec(tag, n, v) is called from the taps inserted by apply_adipls_taps.py.
 * It appends one line "NAME<TAB>n<TAB>hex hex ..." to the file named by the
 * environment variable ADIPLS_TAP_FILE (nothing is written when it is unset).
 * Values are IEEE-754 binary64 bit patterns. The function reads its
 * arguments only.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE *tap_fp = NULL;
static int tap_init = 0;

static const char *tag_name(int tag) {
    static const char *names[] = {"ENTER", "DET", "MATCH", "ITC", "KINER", "VARFRQ", "GRAVPO"};
    if (tag >= 0 && tag < (int)(sizeof(names) / sizeof(names[0]))) return names[tag];
    return "UNKNOWN";
}

static void tap_close(void) {
    if (tap_fp) fclose(tap_fp);
    tap_fp = NULL;
}

void a0trec_(const int *tag, const int *n, const double *v) {
    if (!tap_init) {
        const char *path = getenv("ADIPLS_TAP_FILE");
        tap_init = 1;
        if (path && *path) {
            tap_fp = fopen(path, "w");
            if (tap_fp) atexit(tap_close);
        }
    }
    if (!tap_fp) return;
    fprintf(tap_fp, "%s\t%d\t", tag_name(*tag), *n);
    for (int i = 0; i < *n; i++) {
        uint64_t u;
        memcpy(&u, &v[i], sizeof u);
        fprintf(tap_fp, i ? " %016llX" : "%016llX", (unsigned long long)u);
    }
    fputc('\n', tap_fp);
}
