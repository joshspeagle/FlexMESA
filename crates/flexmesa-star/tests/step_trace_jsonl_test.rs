// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration test verifying `FLEXMESA_STEP_TRACE` emits valid single-line
//! JSONL events into `FLEXMESA_STEP_TRACE_FILE` (`#150`).

use std::path::PathBuf;

use flexmesa_star::{
    load_zams_model, set_hydro_vars, star_evolve_step, StarControls,
    StarJobControls,
};

#[test]
fn test_flexmesa_step_trace_emits_valid_jsonl_records() {
    let trace_path: PathBuf = std::env::temp_dir().join(format!(
        "flexmesa_step_trace_jsonl_{}.jsonl",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&trace_path);
    std::env::set_var("FLEXMESA_STEP_TRACE", "1");
    std::env::set_var("FLEXMESA_STEP_TRACE_FILE", &trace_path);

    let mut model =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("load 1 Msun ZAMS");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    star_evolve_step(&mut model, false, false, true).expect("evolve step");

    let contents =
        std::fs::read_to_string(&trace_path).expect("read JSONL trace file");
    let _ = std::fs::remove_file(&trace_path);

    let lines: Vec<&str> =
        contents.lines().filter(|l| !l.trim().is_empty()).collect();
    assert!(
        lines.iter().any(|l| l.contains("\"event\":\"solver\"")),
        "missing solver JSONL record in {contents}"
    );
    assert!(
        lines.iter().any(|l| l.contains("\"event\":\"star_step\"")),
        "missing star_step JSONL record in {contents}"
    );
    for line in &lines {
        assert!(
            line.starts_with('{') && line.ends_with('}'),
            "not a single-line JSON object: {line}"
        );
        assert!(
            line.contains("\"event\":\"") && line.contains("\"model\":"),
            "missing required JSON keys: {line}"
        );
    }
}
