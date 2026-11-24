/*
 * Linker wrapper for pulse::export_pulse_data (#491): dumps the star state
 * through pulse_dump_tap.f90, calls the real routine, then copies the file
 * MESA wrote next to the dump. It also calls the real routine once more
 * with data_format 'OSC' on the same state (no test case writes OSC), so
 * every dump has MESA's OSC text as well. Link with
 * -Wl,--wrap=__pulse_MOD_export_pulse_data.
 *
 * export_pulse_data(id, data_format, filename, add_center_point,
 * keep_surface_point, add_atmosphere, ierr) with gfortran's hidden lengths
 * of the two character arguments appended.
 */
#include <stddef.h>
#include <stdio.h>
#include <string.h>

extern void __real___pulse_MOD_export_pulse_data(
    int *id, char *data_format, char *filename, int *add_center_point,
    int *keep_surface_point, int *add_atmosphere, int *ierr,
    size_t len_format, size_t len_filename);
extern void pulse_dump_tap_state(int *id, const char *data_format8,
                                 int *call_no);

static void copy_file(const char *src, const char *dst) {
    FILE *in = fopen(src, "rb");
    if (!in) return;
    FILE *out = fopen(dst, "wb");
    if (!out) { fclose(in); return; }
    char buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, in)) > 0) fwrite(buf, 1, n, out);
    fclose(out);
    fclose(in);
}

void __wrap___pulse_MOD_export_pulse_data(
    int *id, char *data_format, char *filename, int *add_center_point,
    int *keep_surface_point, int *add_atmosphere, int *ierr,
    size_t len_format, size_t len_filename) {
    char fmt8[8];
    memset(fmt8, ' ', sizeof fmt8);
    memcpy(fmt8, data_format, len_format < 8 ? len_format : 8);
    int call_no = 0;
    pulse_dump_tap_state(id, fmt8, &call_no);
    __real___pulse_MOD_export_pulse_data(
        id, data_format, filename, add_center_point, keep_surface_point,
        add_atmosphere, ierr, len_format, len_filename);
    if (call_no > 0) {
        char src[512], dst[64], meta[64];
        size_t n = len_filename < sizeof src - 1 ? len_filename : sizeof src - 1;
        memcpy(src, filename, n);
        while (n > 0 && src[n - 1] == ' ') n--;
        src[n] = '\0';
        snprintf(dst, sizeof dst, "pulse_dump/call_%04d.out", call_no);
        copy_file(src, dst);
        char osc_fmt[3] = {'O', 'S', 'C'};
        char osc_name[64];
        snprintf(osc_name, sizeof osc_name, "pulse_dump/call_%04d.osc", call_no);
        int osc_ierr = 0;
        __real___pulse_MOD_export_pulse_data(
            id, osc_fmt, osc_name, add_center_point, keep_surface_point,
            add_atmosphere, &osc_ierr, 3, strlen(osc_name));
        snprintf(meta, sizeof meta, "pulse_dump/call_%04d.flags", call_no);
        FILE *f = fopen(meta, "w");
        if (f) {
            fprintf(f, "add_center_point %d\nkeep_surface_point %d\n"
                       "add_atmosphere %d\nierr %d\n",
                    *add_center_point != 0, *keep_surface_point != 0,
                    *add_atmosphere != 0, *ierr);
            fclose(f);
        }
    }
}
