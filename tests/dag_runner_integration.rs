use std::process::Command;

fn cargo_bin() -> String {
    let output = Command::new("cargo")
        .args(["build", "--quiet"])
        .output()
        .expect("cargo build");
    assert!(output.status.success(), "cargo build failed");
    format!("{}/target/debug/forge", env!("CARGO_MANIFEST_DIR"))
}

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), name)
}

fn setup_registry(dir: &std::path::Path) {
    // Register identity kernel
    let output = Command::new(cargo_bin())
        .args(["verify", &fixture("identity_pass.json"), "--registry", &dir.to_string_lossy()])
        .output()
        .expect("verify identity");
    assert!(output.status.success(), "identity verify failed: {}", String::from_utf8_lossy(&output.stderr));

    // Register movement kernel
    let output = Command::new(cargo_bin())
        .args(["verify", &fixture("movement_pass.json"), "--registry", &dir.to_string_lossy()])
        .output()
        .expect("verify movement");
    assert!(output.status.success(), "movement verify failed: {}", String::from_utf8_lossy(&output.stderr));
}

#[test]
fn run_dag_identity_one_frame() {
    let tmp = tempfile::tempdir().unwrap();
    let registry = tmp.path().join("registry");
    setup_registry(&registry);

    let output = Command::new(cargo_bin())
        .args([
            "run-dag",
            &fixture("game_identity.json"),
            "--registry", &registry.to_string_lossy(),
            "--frames", "1",
        ])
        .output()
        .expect("run-dag identity");
    assert!(
        output.status.success(),
        "run-dag identity failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn run_dag_identity_with_output() {
    let tmp = tempfile::tempdir().unwrap();
    let registry = tmp.path().join("registry");
    setup_registry(&registry);

    let out_path = tmp.path().join("output.json");
    let output = Command::new(cargo_bin())
        .args([
            "run-dag",
            &fixture("game_identity.json"),
            "--registry", &registry.to_string_lossy(),
            "--frames", "1",
            "--output", &out_path.to_string_lossy(),
        ])
        .output()
        .expect("run-dag identity with output");
    assert!(
        output.status.success(),
        "run-dag identity with output failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let content = std::fs::read_to_string(&out_path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert_eq!(json["frame"], 1);
    assert!(json["buffers"]["EntityBuffer"].is_array());
}

#[test]
fn run_dag_identity_with_verify() {
    let tmp = tempfile::tempdir().unwrap();
    let registry = tmp.path().join("registry");
    setup_registry(&registry);

    let output = Command::new(cargo_bin())
        .args([
            "run-dag",
            &fixture("game_identity.json"),
            "--registry", &registry.to_string_lossy(),
            "--frames", "1",
            "--verify",
        ])
        .output()
        .expect("run-dag identity with verify");
    assert!(
        output.status.success(),
        "run-dag identity with verify failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn run_dag_movement_multi_frame() {
    let tmp = tempfile::tempdir().unwrap();
    let registry = tmp.path().join("registry");
    setup_registry(&registry);

    let out_path = tmp.path().join("output.json");
    let output = Command::new(cargo_bin())
        .args([
            "run-dag",
            &fixture("game_movement.json"),
            "--registry", &registry.to_string_lossy(),
            "--frames", "5",
            "--output", &out_path.to_string_lossy(),
        ])
        .output()
        .expect("run-dag movement");
    assert!(
        output.status.success(),
        "run-dag movement failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let content = std::fs::read_to_string(&out_path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert_eq!(json["frame"], 5);
}

#[test]
fn run_dag_missing_manifest() {
    let output = Command::new(cargo_bin())
        .args(["run-dag", "nonexistent.json"])
        .output()
        .expect("run-dag missing manifest");
    assert!(!output.status.success());
}
