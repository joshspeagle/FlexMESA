/* Link-time wrapper: capture the star state before each GYRE set_model
 * call (#170, G3b). */
extern void __real___gyre_mesa_m_MOD_set_model(void *g, void *p, void *v);
extern void __pulse_state_tap_m_MOD_pstate_on_set_model(void *g, void *p, void *v);

void __wrap___gyre_mesa_m_MOD_set_model(void *g, void *p, void *v) {
    __pulse_state_tap_m_MOD_pstate_on_set_model(g, p, v);
    __real___gyre_mesa_m_MOD_set_model(g, p, v);
}
