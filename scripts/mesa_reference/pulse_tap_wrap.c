#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

static uint64_t f64_bits(double v) {
    uint64_t u = 0;
    memcpy(&u, &v, sizeof(double));
    return u;
}

void ptap_ensure_dirs(void) {
    mkdir("pulse_tap", 0755);
    mkdir("pulse_tap/models", 0755);
}

void ptap_write_amdl(const char *path, int nn, int iaa_arg,
                     const double *data_arr, const double *x_arr,
                     const double *aa_arr) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return;
    int32_t hdr[2] = { (int32_t)nn, (int32_t)iaa_arg };
    fwrite(hdr, sizeof(int32_t), 2, fp);
    fwrite(data_arr, sizeof(double), 8, fp);
    fwrite(x_arr, sizeof(double), (size_t)nn, fp);
    fwrite(aa_arr, sizeof(double), (size_t)iaa_arg * (size_t)nn, fp);
    fclose(fp);
}

static int header_written = 0;

void ptap_write_mode(const char *code_str, int eval_id, int model_number,
                     double star_age, int l, int m, int n_pg, int n_p, int n_g,
                     double freq_uhz, double freq_hz, double inertia,
                     double growth_rad_s, double growth_hz,
                     const char *mod_file, const char *pulse_file) {
    ptap_ensure_dirs();
    FILE *fp = fopen("pulse_tap/modes.tsv", "a");
    if (!fp) return;
    if (!header_written) {
        fseek(fp, 0, SEEK_END);
        if (ftell(fp) == 0) {
            fprintf(fp,
                "code\teval_id\tmodel_number\tstar_age_yr\t"
                "l\tm\tn_pg\tn_p\tn_g\t"
                "freq_uHz\tfreq_uHz_hex\tfreq_uHz_bits\t"
                "freq_Hz\tfreq_Hz_hex\tfreq_Hz_bits\t"
                "inertia\tinertia_hex\tinertia_bits\t"
                "growth_rate_rad_s\tgrowth_rate_rad_s_hex\tgrowth_rate_rad_s_bits\t"
                "growth_rate_Hz\tgrowth_rate_Hz_hex\tgrowth_rate_Hz_bits\t"
                "saved_model_file\tsaved_pulse_file\n");
        }
        header_written = 1;
    }
    fprintf(fp,
        "%s\t%d\t%d\t%.17e\t"
        "%d\t%d\t%d\t%d\t%d\t"
        "%.17e\t%a\t0x%016llx\t"
        "%.17e\t%a\t0x%016llx\t"
        "%.17e\t%a\t0x%016llx\t"
        "%.17e\t%a\t0x%016llx\t"
        "%.17e\t%a\t0x%016llx\t"
        "%s\t%s\n",
        code_str, eval_id, model_number, star_age,
        l, m, n_pg, n_p, n_g,
        freq_uhz, freq_uhz, (unsigned long long)f64_bits(freq_uhz),
        freq_hz, freq_hz, (unsigned long long)f64_bits(freq_hz),
        inertia, inertia, (unsigned long long)f64_bits(inertia),
        growth_rad_s, growth_rad_s, (unsigned long long)f64_bits(growth_rad_s),
        growth_hz, growth_hz, (unsigned long long)f64_bits(growth_hz),
        mod_file, pulse_file);
    fclose(fp);
}

/* GYRE linker wrappers */
extern void __real___gyre_mesa_m_MOD_set_model(void *g, void *p, void *v);
extern void __real___gyre_mesa_m_MOD_get_modes(void *l, void *cb, void *ipar, void *rpar);
extern void __pulse_tap_gyre_mod_MOD_ptap_on_gyre_set_model(void *g, void *p, void *v);
extern void __pulse_tap_gyre_mod_MOD_ptap_gyre_get_modes(void *l, void *cb, void *ipar, void *rpar);

void c_call_real_gyre_get_modes_(void *l, void *cb, void *ipar, void *rpar) {
    __real___gyre_mesa_m_MOD_get_modes(l, cb, ipar, rpar);
}

void __wrap___gyre_mesa_m_MOD_set_model(void *g, void *p, void *v) {
    __pulse_tap_gyre_mod_MOD_ptap_on_gyre_set_model(g, p, v);
    __real___gyre_mesa_m_MOD_set_model(g, p, v);
}

void __wrap___gyre_mesa_m_MOD_get_modes(void *l, void *cb, void *ipar, void *rpar) {
    __pulse_tap_gyre_mod_MOD_ptap_gyre_get_modes(l, cb, ipar, rpar);
}

#ifdef PTAP_WITH_ADIPLS
extern void __real_adipls_(int *i_paramset, int *ierr_param, int *i_inout,
                           double *x, double *aa, double *data,
                           int *nn, int *ivarmd, int *iaa_arg);
extern void __real_spcout_adi_(double *x, double *y, double *aa, double *data,
                               int *nn, int *iy, int *iaa, int *ispcpr);
extern void __pulse_tap_adipls_mod_MOD_ptap_on_adipls_model(
    double *x, double *aa, double *data, int *nn, int *iaa_arg);
extern void __pulse_tap_adipls_mod_MOD_ptap_on_adipls_mode(void);

void __wrap_adipls_(int *i_paramset, int *ierr_param, int *i_inout,
                    double *x, double *aa, double *data,
                    int *nn, int *ivarmd, int *iaa_arg) {
    if (*nn > 0 && *i_paramset != 0) {
        __pulse_tap_adipls_mod_MOD_ptap_on_adipls_model(x, aa, data, nn, iaa_arg);
    }
    __real_adipls_(i_paramset, ierr_param, i_inout, x, aa, data, nn, ivarmd, iaa_arg);
}

void __wrap_spcout_adi_(double *x, double *y, double *aa, double *data,
                        int *nn, int *iy, int *iaa, int *ispcpr) {
    __real_spcout_adi_(x, y, aa, data, nn, iy, iaa, ispcpr);
    __pulse_tap_adipls_mod_MOD_ptap_on_adipls_mode();
}
#endif
