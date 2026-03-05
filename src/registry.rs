//! Kernel registry — stores verified kernel contracts in .forge/registry/.

use crate::contract::KernelContractJSON;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryEntry {
    pub name: String,
    pub registry_id: String,
    pub contract: KernelContractJSON,
    pub bundle_hash: String,
    pub verified_at: String,
}

pub fn register_kernel(
    dir: &Path,
    name: &str,
    contract: &KernelContractJSON,
    bundle_hash: &str,
) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir)
        .map_err(|e| format!("Failed to create registry dir: {}", e))?;

    let registry_id = format!("{}_{}", name, bundle_hash);
    let entry = RegistryEntry {
        name: name.to_string(),
        registry_id,
        contract: contract.clone(),
        bundle_hash: bundle_hash.to_string(),
        verified_at: chrono_stub(),
    };

    let path = dir.join(format!("{}.json", name));
    let content = serde_json::to_string_pretty(&entry)
        .map_err(|e| format!("Failed to serialize registry entry: {}", e))?;
    std::fs::write(&path, content)
        .map_err(|e| format!("Failed to write registry entry: {}", e))?;

    Ok(path)
}

pub fn lookup_kernel(dir: &Path, name: &str) -> Result<Option<RegistryEntry>, String> {
    let path = dir.join(format!("{}.json", name));
    if !path.exists() {
        return Ok(None);
    }
    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read registry entry '{}': {}", name, e))?;
    let entry: RegistryEntry = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse registry entry '{}': {}", name, e))?;
    Ok(Some(entry))
}

pub fn list_kernels(dir: &Path) -> Result<Vec<RegistryEntry>, String> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut entries = Vec::new();
    let read_dir = std::fs::read_dir(dir)
        .map_err(|e| format!("Failed to read registry dir: {}", e))?;

    for entry in read_dir {
        let entry = entry.map_err(|e| format!("Dir entry error: {}", e))?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            let content = std::fs::read_to_string(&path)
                .map_err(|e| format!("Failed to read '{}': {}", path.display(), e))?;
            match serde_json::from_str::<RegistryEntry>(&content) {
                Ok(e) => entries.push(e),
                Err(err) => {
                    eprintln!("Warning: skipping invalid registry entry '{}': {}", path.display(), err);
                }
            }
        }
    }
    Ok(entries)
}

fn chrono_stub() -> String {
    // Simple timestamp without pulling in chrono crate
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{}", secs)
}
