/*
 * Per-track history tap for MESA 26.4.1 astero search cases (#430).
 *
 * MESA's astero driver evolves one track per search sample by calling
 * run_star_support::run1_star from eval1 (astero_run_support.f90:224). Each
 * call reopens LOGS/history.data without "append" because
 * doing_first_model_of_run is set (star/private/history.f90:313-318), so only
 * the last track's history survives a run. This linker wrapper, linked with
 * -Wl,--wrap=__run_star_support_MOD_run1_star, calls the real run1_star and
 * then copies the history file to history_tap/track_NNNN.data, numbering
 * every track (accepted or not) from 1. It changes no MESA source and no
 * arithmetic: it only copies a file MESA has finished writing.
 *
 * Environment:
 *   HISTORY_TAP_SRC  history file to copy (default "LOGS/history.data").
 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

/* run1_star(do_alloc_star, do_free_star, okay_to_restart, id, restart,
 *           extras_controls, ierr, inlist_fname_arg): gfortran passes every
 * argument by reference and appends the hidden length of the optional
 * character argument (NULL and 0 when absent). */
extern void __real___run_star_support_MOD_run1_star(
    void *do_alloc_star, void *do_free_star, void *okay_to_restart, void *id,
    void *restart, void *extras_controls, int *ierr, char *inlist_fname_arg,
    size_t inlist_fname_len);

static int htap_track = 0;

static void htap_copy(const char *src, const char *dst) {
    FILE *in = fopen(src, "rb");
    if (!in) {
        fprintf(stderr, "history_tap: cannot open %s\n", src);
        return;
    }
    FILE *out = fopen(dst, "wb");
    if (!out) {
        fprintf(stderr, "history_tap: cannot open %s\n", dst);
        fclose(in);
        return;
    }
    char buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, in)) > 0) {
        fwrite(buf, 1, n, out);
    }
    fclose(out);
    fclose(in);
}

void __wrap___run_star_support_MOD_run1_star(
    void *do_alloc_star, void *do_free_star, void *okay_to_restart, void *id,
    void *restart, void *extras_controls, int *ierr, char *inlist_fname_arg,
    size_t inlist_fname_len) {
    __real___run_star_support_MOD_run1_star(
        do_alloc_star, do_free_star, okay_to_restart, id, restart,
        extras_controls, ierr, inlist_fname_arg, inlist_fname_len);
    htap_track += 1;
    const char *src = getenv("HISTORY_TAP_SRC");
    if (!src || !*src) {
        src = "LOGS/history.data";
    }
    mkdir("history_tap", 0755);
    char dst[64];
    snprintf(dst, sizeof dst, "history_tap/track_%04d.data", htap_track);
    htap_copy(src, dst);
    FILE *log = fopen("history_tap/tracks.tsv", "a");
    if (log) {
        fprintf(log, "%d\t%d\t%s\n", htap_track, ierr ? *ierr : -1, dst);
        fclose(log);
    }
}
