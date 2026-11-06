//! Periodic table symbols, full element names, dominant isotopes,
//! partition function temperatures, and reaction category tables.
//!
//! Covers $Z \in [0, 112]$ (`neut` through `cn`) and the 24 nuclear
//! reaction categories defined in `chem_def.f90`.
//!
//! # Notes on Upstream MESA Conventions
//! - `EL_LONG_NAME[83]` preserves upstream MESA's literal spelling `"bisumth"`
//!   from `chem_def.f90`.
//! - `generate_nuclide_names` and `generate_long_nuclide_names` detect ground
//!   and isomeric states of ${}^{26}\mathrm{Al}$ (`"al-6"`, `"al*6"` and
//!   `"Aluminum-gs"`, `"Aluminum-ex"`) when multiple $Z = 13, A = 26$ entries
//!   appear in the same network list.

/// Maximum atomic number $Z$ in the nuclear database (`Cn`, $Z = 112$).
pub const MAX_EL_Z: usize = 112;

/// Number of chemical elements ($Z = 1..=112$).
pub const NUM_CHEM_ELEMENTS: usize = 112;

/// Number of nuclear reaction categories (`pp` through `other`).
pub const NUM_CATEGORIES: usize = 24;

/// Number of temperature grid points in the nuclear partition function table.
pub const NPART: usize = 24;

/// Partition function temperature grid $T_9 \in [0.1, 10.0]\,\mathrm{GK}$.
pub const T_PART: [f64; NPART] = [
    0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.50,
    2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00,
];

/// Short element symbols indexed by atomic number $Z \in [0, 112]$.
pub const EL_NAME: [&str; MAX_EL_Z + 1] = [
    "neut", "h", "he", "li", "be", "b", "c", "n", "o", "f", "ne", "na", "mg",
    "al", "si", "p", "s", "cl", "ar", "k", "ca", "sc", "ti", "v", "cr", "mn",
    "fe", "co", "ni", "cu", "zn", "ga", "ge", "as", "se", "br", "kr", "rb",
    "sr", "y", "zr", "nb", "mo", "tc", "ru", "rh", "pd", "ag", "cd", "in",
    "sn", "sb", "te", "i", "xe", "cs", "ba", "la", "ce", "pr", "nd", "pm",
    "sm", "eu", "gd", "tb", "dy", "ho", "er", "tm", "yb", "lu", "hf", "ta",
    "w", "re", "os", "ir", "pt", "au", "hg", "tl", "pb", "bi", "po", "at",
    "rn", "fr", "ra", "ac", "th", "pa", "u", "np", "pu", "am", "cm", "bk",
    "cf", "es", "fm", "md", "no", "lr", "rf", "db", "sg", "bh", "hs", "mt",
    "ds", "rg", "cn",
];

/// Full element names indexed by atomic number $Z \in [0, 112]$.
#[rustfmt::skip]
pub const EL_LONG_NAME: [&str; MAX_EL_Z + 1] = [
    "neutron", "hydrogen", "helium", "lithium", "beryllium", "boron",
    "carbon", "nitrogen", "oxygen", "fluorine", "neon", "sodium",
    "magnesium", "aluminum", "silicon", "phosphorus", "sulfur", "chlorine",
    "argon", "potassium", "calcium", "scandium", "titanium", "vanadium",
    "chromium", "manganese", "iron", "cobalt", "nickel", "copper", "zinc",
    "gallium", "germanium", "arsenic", "selenium", "bromine", "krypton",
    "rubidium", "strontium", "yttrium", "zirconium", "niobium", "molybdenum",
    "technetium", "ruthenium", "rhodium", "palladium", "silver", "cadmium",
    "indium", "tin", "antimony", "tellurium", "iodine", "xenon", "cesium",
    "barium", "lanthanum", "cerium", "praseodymium", "neodymium",
    "promethium", "samarium", "europium", "gadolinium", "terbium",
    "dysprosium", "holmium", "erbium", "thulium", "ytterbium", "lutetium",
    "hafnium", "tantalum", "tungsten", "rhenium", "osmium", "iridium",
    "platinum", "gold", "mercury", "thallium", "lead", "bisumth", "polonium",
    "astatine", "radon", "francium", "radium", "actinium", "thorium",
    "protactinium", "uranium", "neptunium", "plutonium", "americium",
    "curium", "berkelium", "californium", "einsteinium", "fermium",
    "mendelevium", "nobelium", "lawrencium", "rutherfordium", "dubnium",
    "seaborgium", "bohrium", "hassium", "meitnerium", "darmstadtium",
    "roentgenium", "copernicium",
];

/// Dominant isotope name for each element $Z \in [0, 112]$ (`Lodders 2003`).
pub const MAIN_ISO_NAME: [&str; MAX_EL_Z + 1] = [
    "neut", "h1", "he4", "li7", "be9", "b11", "c12", "n14", "o16", "f19",
    "ne20", "na23", "mg24", "al27", "si28", "p31", "s32", "cl35", "ar36",
    "k39", "ca40", "sc45", "ti48", "v51", "cr52", "mn55", "fe56", "co59",
    "ni58", "cu63", "zn64", "ga69", "ge74", "as75", "se80", "br79", "kr84",
    "rb85", "sr88", "y89", "zr90", "nb93", "mo98", "tc97", "ru102", "rh103",
    "pd106", "ag107", "cd114", "in115", "sn120", "sb121", "te130", "i127",
    "xe132", "cs133", "ba138", "la139", "ce140", "pr141", "nd142", "pm145",
    "sm152", "eu153", "gd158", "tb159", "dy164", "ho165", "er166", "tm169",
    "yb174", "lu175", "hf180", "ta181", "w184", "re187", "os192", "ir193",
    "pt195", "au197", "hg202", "tl205", "pb208", "bi209", "po210", "at210",
    "rn222", "fr223", "ra226", "ac227", "th232", "pa231", "u238", "np237",
    "pu244", "am243", "cm247", "bk247", "cf251", "es252", "fm257", "md258",
    "no259", "lr262", "rf261", "db268", "sg271", "bh274", "hs270", "mt278",
    "ds281", "rg281", "cn285",
];

/// Nuclear reaction category names (`1..=24` in 1-based Fortran indexing).
#[rustfmt::skip]
pub const CATEGORY_NAME: [&str; NUM_CATEGORIES] = [
    "pp", "cno", "tri_alpha", "c_alpha", "n_alpha", "o_alpha", "ne_alpha",
    "na_alpha", "mg_alpha", "si_alpha", "s_alpha", "ar_alpha", "ca_alpha",
    "ti_alpha", "cr_alpha", "fe_co_ni", "c12_c12", "c12_o16", "o16_o16",
    "photo", "pnhe4", "ni56_co56", "co56_fe56", "other",
];

/// Return the 1-based element ID ($Z \in [1, 112]$) for a short symbol.
///
/// Performs a case-insensitive trimmed lookup matching `chem_get_element_id`
/// in `chem_lib.f90`. Returns `None` if `name` is not found.
#[must_use]
pub fn chem_get_element_id(name: &str) -> Option<usize> {
    let target = name.trim();
    EL_NAME
        .iter()
        .enumerate()
        .skip(1)
        .find_map(|(z, &sym)| sym.eq_ignore_ascii_case(target).then_some(z))
}

/// Return the short element symbol for $Z \in [0, 112]$.
///
/// Index `0` returns `"neut"` and `1..=112` return `"h"` through `"cn"`.
#[must_use]
pub fn element_name(z: usize) -> Option<&'static str> {
    EL_NAME.get(z).copied()
}

/// Return the full element name for $Z \in [0, 112]$.
///
/// Index `0` returns `"neutron"` and `1..=112` return `"hydrogen"` through
/// `"copernicium"`.
#[must_use]
pub fn element_long_name(z: usize) -> Option<&'static str> {
    EL_LONG_NAME.get(z).copied()
}

/// Return the dominant isotope name for $Z \in [1, 112]$.
///
/// Matches `chem_element_main_iso_name(Z)` in `chem_def.f90`.
#[must_use]
pub fn element_main_iso_name(z: usize) -> Option<&'static str> {
    (1..=MAX_EL_Z).contains(&z).then(|| MAIN_ISO_NAME[z])
}

/// Return the 1-based reaction category ID ($1..=24$) for `cname`, or `0`.
///
/// Matches `category_id(cname)` in `chem_lib.f90`.
#[must_use]
pub fn category_id(cname: &str) -> usize {
    let target = cname.trim();
    CATEGORY_NAME
        .iter()
        .position(|&cat| cat == target)
        .map_or(0, |idx| idx + 1)
}

/// Return the reaction category name for 1-based `id` ($1..=24$).
///
/// Returns `None` when `id == 0` or `id > NUM_CATEGORIES`.
#[must_use]
pub fn category_name(id: usize) -> Option<&'static str> {
    (1..=NUM_CATEGORIES)
        .contains(&id)
        .then(|| CATEGORY_NAME[id - 1])
}

/// Return the 1-based partition function temperature index for $T_9$ (in GK).
///
/// Performs an 8-step binary search over [`T_PART`], matching
/// `get_partition_fcn_indx(T9)` in `chem_lib.f90`.
#[must_use]
pub fn get_partition_fcn_indx(t9: f64) -> usize {
    let mut low: usize = 1;
    let mut high: usize = NPART;
    if t9 < T_PART[0] {
        return 0;
    }
    for _ in 0..8 {
        if high - low <= 1 {
            return low;
        }
        let mid = usize::midpoint(high, low);
        if t9 < T_PART[mid - 1] {
            high = mid;
        } else {
            low = mid;
        }
    }
    low - 1
}

/// Shared formatter for [`generate_nuclide_names`] and
/// [`generate_long_nuclide_names`].
fn format_nuclide_names(
    z: &[usize],
    a: &[usize],
    table: &[&str; MAX_EL_Z + 1],
    sep: &str,
    al26_gs: &str,
    al26_ex: &str,
    out_of_range: &str,
) -> Vec<String> {
    let n = z.len().min(a.len());
    let use_al26_isomers =
        (0..n).filter(|&i| z[i] == 13 && a[i] == 26).count() > 1;
    let mut isomer_idx = 0usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        match z[i] {
            0 => out.push(table[0].to_string()),
            13 if a[i] == 26 && use_al26_isomers => {
                let label = if isomer_idx == 0 { al26_gs } else { al26_ex };
                isomer_idx += 1;
                out.push(label.to_string());
            }
            zi if zi <= MAX_EL_Z => {
                out.push(format!("{}{sep}{}", table[zi], a[i]));
            }
            _ => out.push(out_of_range.to_string()),
        }
    }
    out
}

/// Generate short nuclide names (such as `"fe56"` or `"al-6"`, `"al*6"`)
/// from parallel $(Z, A)$ arrays, matching `generate_nuclide_names`.
#[must_use]
pub fn generate_nuclide_names(z: &[usize], a: &[usize]) -> Vec<String> {
    format_nuclide_names(z, a, &EL_NAME, "", "al-6", "al*6", "*****")
}

/// Generate long nuclide names (such as `"iron-56"`) from parallel $(Z, A)$
/// arrays, matching `generate_long_nuclide_names`.
#[must_use]
pub fn generate_long_nuclide_names(z: &[usize], a: &[usize]) -> Vec<String> {
    format_nuclide_names(
        z,
        a,
        &EL_LONG_NAME,
        "-",
        "Aluminum-gs",
        "Aluminum-ex",
        "********",
    )
}
