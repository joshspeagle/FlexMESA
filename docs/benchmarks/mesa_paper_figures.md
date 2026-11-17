# MESA I-VI Comparison-Figure Roadmap (`#173`)

`FlexMESA` tracks future validation figures spanning the six MESA instrument papers under [Issue #173](https://github.com/joshspeagle/FlexMESA/issues/173) and [Issue #140](https://github.com/joshspeagle/FlexMESA/issues/140). Every figure added to this page must execute canonical Rust `"strict"` code on unmodified MESA `26.4.1` inlists and overplot real MESA `26.4.1` reference outputs from the same run (`DECISIONS #12`, `#17`).

## Removal of the Pre-Audit Synthetic Atlas (`#142`)

An earlier `18`-figure gallery and its synthetic numerical landmark table were removed in `#142`. Although several single-kernel panels evaluated `eos`, `kap`, `turb`, `auto_diff`, or ZAMS endpoints, the evolutionary tracks, giant-planet radius histories, `dedt` conservation residuals, and white-dwarf cooling delays plotted literal typed arrays, closed-form curves, or seeded random draws without running evolutionary steps or comparing against MESA outputs (`DECISIONS #12`). Retaining synthetic figures without `"Schematic"` labels or verified MESA references would misstate current engine capabilities.

---

## Planned MESA `26.4.1` Overplot Deliverables (`#173`)

Candidate MESA I-VI overplot comparisons await resolution of their prerequisite `#140` trajectory and bundled-code issues (`0 / 106` passing at baseline; see [Declared Known Gaps](known-gaps.md) and [Port Coverage Map](coverage-map.md)):

| Instrument Paper | Planned Real-Data Overplot Diagnostics | Blocking `#140` Prerequisites | Current Kernel Availability |
| :--- | :--- | :--- | :--- |
| **MESA I** ([Paxton et al. 2011](https://doi.org/10.1088/0067-0049/192/1/3)) | EOS/kap $(\log_{10}\rho, \log_{10} T)$ regime maps; $1\,M_\odot$ HR and $(\log_{10}\rho_c, \log_{10} T_c)$ tracks; Kippenhahn evolution | `#148`, `#150`-`#155`, `#161` | Pointwise `eos.get_pc_alfa`, `kap.kap_get_results`, and ZAMS catalog evaluate locally; full evolutionary comparisons blocked on solver/mesh parity |
| **MESA II** ([Paxton et al. 2013](https://doi.org/10.1088/0067-0049/208/1/4)) | Irradiated giant-planet contraction; solar/red-giant $\Delta P_1$ mixed modes; $15\,M_\odot$ Spruit-Tayler dynamo; CCSN infall profiles | `#152`, `#158`, `#162`, `#170`, `#171` | `astero` scaling/ratios and `star` wind/rotation formulas run pointwise; coupled evolution and GYRE/ADIPLS modes await parity |
| **MESA III** ([Paxton et al. 2015](https://doi.org/10.1088/0067-0049/220/1/15)) | Binary RLOF tracks (`double_bh`, `star_plus_point_mass`); weak Urca shells; HLLC shock and Duffell RTI mixing | `#152`, `#160`, `#161` | Pointwise RLOF rates, Kerr spin, and HLLC Riemann face fluxes run locally; coupled binary tracks remain declared gaps |
| **MESA IV** ([Paxton et al. 2018](https://doi.org/10.3847/1538-4365/aaa5a8)) | Convective premixing core boundaries; Burgers diffusion and radiative levitation; `RSP` Petersen period ratios and nonlinear light curves | `#148`, `#155`, `#156`, `#161` | Monochromatic `op_radacc` is absent (`1.4M_ms_op_mono`, `radiative_levitation`); `RSP` cases match within tol once `#148` extras move out |
| **MESA V** ([Paxton et al. 2019](https://doi.org/10.3847/1538-4365/ab2241)) | `RSP` work integrals; time-centered `dedt` energy conservation residuals vs. `eps_grav`; $0.6\,M_\odot$ WD crystallization and ${}^{22}\mathrm{Ne}$ cooling delays | `#148`, `#152`, `#156`, `#161` | `Skye`/`PC` EOS and `RSP` flux diagnostics run pointwise; multi-Gyr WD cooling tracks await full diffusion/phase-separation parity |
| **MESA VI** ([Jermyn et al. 2023](https://doi.org/10.3847/1538-4365/acae8d)) | Steady Cox MLT vs. `TDC` transient convergence; `DualOrder1` forward-AD accuracy vs. finite differences; `Skye` crystallization profiles | `#155`, `#157` | Local `turb.set_mlt`/`set_tdc`, `auto_diff.DualOrder1`, and `eos.eval_skye_eos_zones` evaluate locally; full coupled `TDC` hydro awaits `#157` |

---

## Inspecting Real MESA `26.4.1` Reference Histories for Overplots

Future validation builders under `#173` compare Rust `"strict"` logs directly against committed MESA `26.4.1` histories using `flexmesa.testing.history_compare` rather than hard-coded arrays:

```python
from flexmesa.testing import case_manifest, history_compare

hist_path = (
    case_manifest.repo_root()
    / "tests"
    / "fixtures"
    / "golden"
    / "star_test"
    / "history.data"
)
tbl = history_compare.load_history_file(hist_path)
assert tbl.mesa_version_number == "r26.4.1"
print(
    f"Loaded {hist_path.name} ({tbl.mesa_version_number}): "
    f"{tbl.num_rows} rows, final log_L={float(tbl.columns['log_L'][-1]):.6f}"
)
```
