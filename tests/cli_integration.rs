use std::path::PathBuf;
use std::process::Command;

fn forge_bin() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_BIN_EXE_forge"));
    // Fallback if the macro doesn't resolve
    if !path.exists() {
        path = PathBuf::from("target/debug/forge");
    }
    path
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn run_verify(fixture: &str) -> (bool, String) {
    let registry_dir = tempfile::tempdir().expect("tempdir");
    let output = Command::new(forge_bin())
        .arg("verify")
        .arg(fixtures_dir().join(fixture))
        .arg("--registry")
        .arg(registry_dir.path())
        .output()
        .expect("failed to execute forge");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    (output.status.success(), stdout)
}

fn run_validate_dag(fixture: &str, registry_dir: &std::path::Path) -> (bool, String) {
    let output = Command::new(forge_bin())
        .arg("validate-dag")
        .arg(fixtures_dir().join(fixture))
        .arg("--registry")
        .arg(registry_dir)
        .output()
        .expect("failed to execute forge");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    (output.status.success(), stdout)
}

fn parse_accepted(stdout: &str) -> Option<bool> {
    let v: serde_json::Value = serde_json::from_str(stdout).ok()?;
    v.get("accepted").and_then(|a| a.as_bool())
}

fn parse_valid(stdout: &str) -> Option<bool> {
    let v: serde_json::Value = serde_json::from_str(stdout).ok()?;
    v.get("valid").and_then(|a| a.as_bool())
}

// ===== Verify pass tests =====

#[test]
fn verify_identity_pass() {
    let (success, stdout) = run_verify("identity_pass.json");
    assert!(success, "forge verify should succeed.\nOutput: {}", stdout);
    assert_eq!(parse_accepted(&stdout), Some(true));
}

#[test]
fn verify_movement_pass() {
    let (success, stdout) = run_verify("movement_pass.json");
    assert!(success, "forge verify should succeed.\nOutput: {}", stdout);
    assert_eq!(parse_accepted(&stdout), Some(true));
}

// ===== Verify fail tests =====

#[test]
fn verify_break_invariant_fail() {
    let (success, stdout) = run_verify("break_invariant_fail.json");
    assert!(!success, "should fail");
    assert_eq!(parse_accepted(&stdout), Some(false));
}

#[test]
fn verify_dirty_input_fail() {
    let (success, stdout) = run_verify("dirty_input_fail.json");
    assert!(!success, "should fail");
    assert_eq!(parse_accepted(&stdout), Some(false));
}

#[test]
fn verify_movement_no_clamp_fail() {
    let (success, stdout) = run_verify("movement_no_clamp_fail.json");
    assert!(!success, "should fail");
    assert_eq!(parse_accepted(&stdout), Some(false));
}

#[test]
fn verify_modify_dead_fail() {
    let (success, stdout) = run_verify("modify_dead_fail.json");
    assert!(!success, "should fail");
    assert_eq!(parse_accepted(&stdout), Some(false));
}

#[test]
fn verify_bad_wgsl_fail() {
    let (success, stdout) = run_verify("bad_wgsl_fail.json");
    assert!(!success, "should fail");
    assert_eq!(parse_accepted(&stdout), Some(false));
}

#[test]
fn verify_wrong_bindings_fail() {
    let (success, stdout) = run_verify("wrong_bindings_fail.json");
    assert!(!success, "should fail");
    assert_eq!(parse_accepted(&stdout), Some(false));
}

#[test]
fn verify_struct_mismatch_fail() {
    let (success, stdout) = run_verify("struct_mismatch_fail.json");
    assert!(!success, "should fail");
    assert_eq!(parse_accepted(&stdout), Some(false));
}

// ===== DAG tests =====

#[test]
fn validate_dag_valid() {
    // First register the identity kernel
    let registry_dir = tempfile::tempdir().expect("tempdir");
    let output = Command::new(forge_bin())
        .arg("verify")
        .arg(fixtures_dir().join("identity_pass.json"))
        .arg("--registry")
        .arg(registry_dir.path())
        .output()
        .expect("failed to execute forge");
    assert!(output.status.success(), "identity should verify first");

    let (success, stdout) = run_validate_dag("valid_dag.json", registry_dir.path());
    assert!(success, "valid DAG should pass.\nOutput: {}", stdout);
    assert_eq!(parse_valid(&stdout), Some(true));
}

#[test]
fn validate_dag_cyclic_fail() {
    let registry_dir = tempfile::tempdir().expect("tempdir");
    // Register identity kernel so check 4 passes, but ordering check should fail
    let _ = Command::new(forge_bin())
        .arg("verify")
        .arg(fixtures_dir().join("identity_pass.json"))
        .arg("--registry")
        .arg(registry_dir.path())
        .output();

    let (success, stdout) = run_validate_dag("cyclic_dag_fail.json", registry_dir.path());
    assert!(!success, "cyclic DAG should fail");
    assert_eq!(parse_valid(&stdout), Some(false));
}

#[test]
fn validate_dag_missing_kernel_fail() {
    let registry_dir = tempfile::tempdir().expect("tempdir");

    let (success, stdout) = run_validate_dag("missing_kernel_fail.json", registry_dir.path());
    assert!(!success, "missing kernel DAG should fail");
    assert_eq!(parse_valid(&stdout), Some(false));
}
