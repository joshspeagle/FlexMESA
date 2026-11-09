# Bibliography & Primary Literature

`FlexMESA` translates the physical equations, tabulated microphysics, and numerical algorithms of **Modules for Experiments in Stellar Astrophysics (`MESA`, release `r24.08.1`)** into Safe Rust and Python/JAX. When publishing scientific results obtained with `FlexMESA`, please cite `FlexMESA` alongside the **MESA Instrument Papers** and the primary physics references for the active equations of state, opacities, nuclear reaction rates, and mixing prescriptions listed below.

---

## 1. MESA Instrument Papers (I through VI)

- **MESA I**: Paxton, B., Bildsten, L., Dotter, A., Herwig, F., Lesaffre, P., & Timmes, F. (2011), *Modules for Experiments in Stellar Astrophysics (MESA)*, ApJS, 192, 3. [doi:10.1088/0067-0049/192/1/3](https://doi.org/10.1088/0067-0049/192/1/3)
- **MESA II**: Paxton, B., Cantiello, M., Arras, P., et al. (2013), *Modules for Experiments in Stellar Astrophysics (MESA): Planets, Oscillations, Rotation, and Massive Stars*, ApJS, 208, 4. [doi:10.1088/0067-0049/208/1/4](https://doi.org/10.1088/0067-0049/208/1/4)
- **MESA III**: Paxton, B., Marchant, P., Schwab, J., et al. (2015), *Modules for Experiments in Stellar Astrophysics (MESA): Binaries, Pulsations, and Explosions*, ApJS, 220, 15. [doi:10.1088/0067-0049/220/1/15](https://doi.org/10.1088/0067-0049/220/1/15)
- **MESA IV**: Paxton, B., Schwab, J., Bauer, E. B., et al. (2018), *Modules for Experiments in Stellar Astrophysics (MESA): Convective Boundaries, Element Diffusion, and Massive Star Explosions*, ApJS, 234, 34. [doi:10.3847/1538-4365/aaa5a8](https://doi.org/10.3847/1538-4365/aaa5a8)
- **MESA V**: Paxton, B., Smolec, R., Schwab, J., et al. (2019), *Modules for Experiments in Stellar Astrophysics (MESA): Pulsating Variable Stars, Rotation, Convective Boundaries, and Energy Conservation*, ApJS, 243, 10. [doi:10.3847/1538-4365/ab2241](https://doi.org/10.3847/1538-4365/ab2241)
- **MESA VI**: Jermyn, A. S., Bauer, E. B., Schwab, J., et al. (2023), *Modules for Experiments in Stellar Astrophysics (MESA): Time-Dependent Convection, Energy Conservation, Automatic Differentiation, and Infrastructure*, ApJS, 265, 15. [doi:10.3847/1538-4365/acae8d](https://doi.org/10.3847/1538-4365/acae8d)

---

## 2. Equations of State (`flexmesa.eos`)

- **Aparicio Fermi-Dirac Integrals**: Aparicio, J. M. (1998), *A Simple and Accurate Approximation of the Fermi-Dirac Integrals*, ApJS, 117, 627. [doi:10.1086/313121](https://doi.org/10.1086/313121)
- **`HELM` EOS**: Timmes, F. X., & Swesty, F. D. (2000), *The Accuracy, Consistency, and Speed of an Electron-Positron Equation of State Based on Table Interpolation of the Helmholtz Free Energy*, ApJS, 126, 501. [doi:10.1086/313304](https://doi.org/10.1086/313304)
- **`PC` Degenerate & Coulomb EOS**: Potekhin, A. Y., & Chabrier, G. (2010), *Thermodynamic Functions of Dense Plasmas: Analytic Approximations for Astrophysical Applications*, Contributions to Plasma Physics, 50, 82. [doi:10.1002/ctpp.201010017](https://doi.org/10.1002/ctpp.201010017)
- **`Skye` Crystallizing EOS**: Jermyn, A. S., Schwab, J., Bauer, E., Timmes, F. X., & Potekhin, A. Y. (2021), *Skye: A Differentiable Equation of State*, ApJ, 913, 72. [doi:10.3847/1538-4357/abf48e](https://doi.org/10.3847/1538-4357/abf48e)
- **`OPAL` EOS**: Rogers, F. J., & Nayfonov, A. (2002), *Updated and Expanded OPAL Equation-of-State Tables: Implications for Helioseismology*, ApJ, 576, 1064. [doi:10.1086/341894](https://doi.org/10.1086/341894)
- **`SCVH` EOS**: Saumon, D., Chabrier, G., & van Horn, H. M. (1995), *An Equation of State for Low-Mass Stars and Giant Planets*, ApJS, 99, 713. [doi:10.1086/192204](https://doi.org/10.1086/192204)
- **`FreeEOS`**: Irwin, A. W. (2012), *FreeEOS: Equation of State Calculations for Stellar Interiors*, Astrophysics Source Code Library, `ascl:1211.002`.
- **`CMS` Hydrogen-Helium EOS**: Chabrier, G., Mazevet, S., & Soubiran, F. (2019), *A New Equation of State for Dense Hydrogen-Helium Mixtures*, ApJ, 872, 51. [doi:10.3847/1538-4357/aaf99f](https://doi.org/10.3847/1538-4357/aaf99f)

---

## 3. Opacities, Model Atmospheres & Synthetic Photometry (`kap`, `atm`, `colors`)

- **OPAL Radiative Opacities**: Iglesias, C. A., & Rogers, F. J. (1996), *Updated Opal Opacities*, ApJ, 464, 943. [doi:10.1086/177381](https://doi.org/10.1086/177381)
- **Opacity Project (`OP`)**: Badnell, N. R., Bautista, M. A., Butler, K., et al. (2005), *Updated Opacities from the Opacity Project*, MNRAS, 360, 458. [doi:10.1111/j.1365-2966.2005.08991.x](https://doi.org/10.1111/j.1365-2966.2005.08991.x)
- **Low-Temperature Molecular Opacities**: Ferguson, J. W., Alexander, D. R., Allard, F., et al. (2005), *Low-Temperature Opacities*, ApJ, 623, 585. [doi:10.1086/428642](https://doi.org/10.1086/428642)
- **Electron Conduction Opacities**: Cassisi, S., Potekhin, A. Y., Pietrinferni, A., Catelan, M., & Salaris, M. (2007), *Updated Electron-Conduction Opacities: The Impact on Low-Mass Stellar Models*, ApJ, 661, 1094. [doi:10.1086/516819](https://doi.org/10.1086/516819)
- **Relativistic Compton Scattering**: Poutanen, J. (2017), *Rosseland and Flux-Mean Opacities for Compton Scattering*, ApJ, 835, 119. [doi:10.3847/1538-4357/835/2/119](https://doi.org/10.3847/1538-4357/835/2/119)
- **Irradiated Planetary Atmospheres**: Guillot, T. (2010), *On the Radiative Equilibrium of Irradiated Planetary Atmospheres*, A&A, 520, A27. [doi:10.1051/0004-6361/200913396](https://doi.org/10.1051/0004-6361/200913396)
- **Kurucz/Castelli Model Atmospheres & SEDs**: Castelli, F., & Kurucz, R. L. (2003), *New Grids of ATLAS9 Model Atmospheres*, IAU Symp. 210, `arXiv:astro-ph/0405087`.

---

## 4. Nuclear Reaction Rates, Screening, Abundances & Neutrinos (`chem`, `neu`, `rates`, `net`)

- **Solar Abundances (`AGSS09`, `AAG21`, `MB22`)**: Asplund, M., Grevesse, N., Sauval, A. J., & Scott, P. (2009), ARA&A, 47, 481 ([doi:10.1146/annurev.astro.46.060407.145222](https://doi.org/10.1146/annurev.astro.46.060407.145222)); Asplund, M., Amarsi, A. M., & Grevesse, N. (2021), A&A, 653, A141 ([doi:10.1051/0004-6361/202140445](https://doi.org/10.1051/0004-6361/202140445)).
- **JINA REACLIB Database**: Cyburt, R. H., Amthor, A. M., Ferguson, R., et al. (2010), *The JINA REACLIB Database: Its Recent Updates and Impact on Type-I X-Ray Bursts*, ApJS, 189, 240. [doi:10.1088/0067-0049/189/1/240](https://doi.org/10.1088/0067-0049/189/1/240)
- **Plasma Screening (`Salpeter`, `DGC73`, `AJ78`, `Itoh79`, `Chugunov07`)**: Salpeter, E. E. (1954), Aust. J. Phys., 7, 373; DeWitt, H. E., Graboske, H. C., & Cooper, M. S. (1973), ApJ, 181, 439; Alastuey, A., & Jancovici, B. (1978), ApJ, 226, 1034; Itoh, N., Totsuji, H., Ichimaru, S., & DeWitt, H. E. (1979), ApJ, 234, 1079; Chugunov, A. I., DeWitt, H. E., & Yakovlev, D. G. (2007), Phys. Rev. D, 76, 025028.
- **Thermal Neutrino Losses**: Itoh, N., Hayashi, H., Nishikawa, A., & Kohyama, Y. (1996), *Neutrino Energy Loss in Stellar Interiors. VII. Pair, Photo-, Plasma, Bremsstrahlung, and Recombination Neutrino Processes*, ApJS, 102, 411. [doi:10.1086/192264](https://doi.org/10.1086/192264)

---

## 5. Convection, Diffusion, Phase Separation, Rotation & Winds (`turb`, `star`)

- **Mixing Length Theory (`MLT`)**: Cox, J. P., & Giuli, R. T. (1968), *Principles of Stellar Structure*; Henyey, L., Vardya, M. S., & Bodenheimer, P. (1965), ApJ, 142, 841.
- **Time-Dependent Convection (`TDC` & `RSP2`)**: Kuhfuss, R. (1986), *A Model for Time-Dependent Turbulent Convection*, A&A, 160, 116; Smolec, R., & Moskalik, P. (2008), Acta Astron., 58, 193.
- **Semiconvection & Thermohaline Mixing**: Langer, N., Sugimoto, D., & Fricke, K. J. (1983), A&A, 126, 207; Kippenhahn, R., Ruschenplatt, G., & Thomas, H.-C. (1980), A&A, 91, 175; Brown, J. M., Garaud, P., & Stellmach, S. (2013), ApJ, 768, 34.
- **Burgers Multi-Fluid & Strong-Coupling Atomic Diffusion**: Burgers, J. M. (1969), *Flow Equations for Composite Gases*; Thoul, A. A., Bahcall, J. N., & Loeb, A. (1994), ApJ, 421, 828; Caplan, M. E., Bauer, E. B., & Freeman, I. F. (2022), MNRAS, 513, L52 ([doi:10.1093/mnrasl/slac032](https://doi.org/10.1093/mnrasl/slac032)).
- **White Dwarf Crystallization & Phase Separation**: Bauer, E. B., Schwab, J., Bildsten, L., & Cheng, S. (2020), ApJ, 902, 93; Blouin, S., Daligault, J., & Saumon, D. (2021), ApJL, 911, L5.
- **Shellular Rotation & Spruit-Tayler Dynamo**: Endal, A. S., & Sofia, S. (1976), ApJ, 210, 184; Heger, A., Langer, N., & Woosley, S. E. (2000), ApJ, 528, 368; Spruit, H. C. (2002), A&A, 381, 923 ([doi:10.1051/0004-6361:20011465](https://doi.org/10.1051/0004-6361:20011465)).
- **Stellar Winds (`Reimers`, `Blocker`, `Vink`, `Dutch`)**: Reimers, D. (1975), Mem. Soc. R. Sci. Liege, 8, 369; Blöcker, T. (1995), A&A, 297, 727; Vink, J. S., de Koter, A., & Lamers, H. J. G. L. M. (2001), A&A, 369, 574.

---

## 6. Splines, Block-Tridiagonal Solvers & Timestep Control (`interp_1d`, `mtx`, `num`, `star`)

- **Monotonic 1D Splines**: Steffen, M. (1990), *A Simple Method for Monotonic Interpolation in One Dimension*, A&A, 239, 443; Huynh, H. T. (1993), SIAM J. Numer. Anal., 30, 57.
- **Odd-Even Parallel Block Cyclic Reduction (`bcyclic`)**: Hirshman, S. P., Perumalla, K. S., Lynch, V. E., & Sanchez, R. (2010), *BCYCLIC: A Parallel Block Tridiagonal Matrix Cyclic Solver*, J. Comput. Phys., 229, 6392. [doi:10.1016/j.jcp.2010.04.049](https://doi.org/10.1016/j.jcp.2010.04.049)
- **Digital Filter Timestep Control (`H211`)**: Söderlind, G., & Wang, L. (2006), *Adaptive Time-Stepping and Computational Stability*, J. Comput. Appl. Math., 185, 225. [doi:10.1016/j.cam.2005.03.008](https://doi.org/10.1016/j.cam.2005.03.008)

---

## 7. Interacting Binaries & Asteroseismology (`binary`, `astero`)

- **Roche-Lobe Geometry & Mass Transfer**: Eggleton, P. P. (1983), ApJ, 268, 368 ([doi:10.1086/160960](https://doi.org/10.1086/160960)); Ritter, H. (1988), A&A, 202, 93; Kolb, U., & Ritter, H. (1990), A&A, 236, 385.
- **Kerr Black-Hole Accretion, $\dot{J}$ & Tides**: Bardeen, J. M. (1970), Nature, 226, 64; Thorne, K. S. (1974), ApJ, 191, 507; Peters, P. C. (1964), Phys. Rev., 136, B1224; Rappaport, S., Verbunt, F., & Joss, P. C. (1983), ApJ, 275, 713; Hut, P. (1981), A&A, 99, 126.
- **Asteroseismic Scaling, Surface Corrections & Ratios**: Kjeldsen, H., & Bedding, T. R. (1995), A&A, 293, 87; Kjeldsen, H., Bedding, T. R., & Christensen-Dalsgaard, J. (2008), ApJL, 683, L175; Ball, W. H., & Gizon, L. (2014), A&A, 568, A123 ([doi:10.1051/0004-6361/201424325](https://doi.org/10.1051/0004-6361/201424325)); Sonoi, T., Samadi, R., Belkacem, K., et al. (2015), A&A, 583, A112; Roxburgh, I. W., & Vorontsov, S. V. (2003), A&A, 411, 215.
