//! DAG manifest parsing and validation.
//!
//! Validates that a pipeline DAG is well-formed:
//! 1. Ordering — every depends_on names an earlier node
//! 2. Buffer provenance — consumed buffers were produced by a prior node
//! 3. No write-write hazards — independent nodes can't both produce the same buffer
//! 4. Registry — every kernel is Forge-registered
//! 5. Contract match — consume/produce match kernel contracts

use crate::registry::{self, RegistryEntry};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DagNode {
    pub name: String,
    pub kernel: String,
    #[serde(default)]
    pub depends_on: Vec<String>,
    #[serde(default)]
    pub consumes: Vec<String>,
    #[serde(default)]
    pub produces: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DagManifest {
    pub name: String,
    pub design_params: String,
    pub pipeline: Vec<DagNode>,
}

#[derive(Debug, Serialize)]
pub struct DagValidationResult {
    pub valid: bool,
    pub errors: Vec<String>,
}

pub fn validate_dag(manifest: &DagManifest, registry_dir: &Path) -> DagValidationResult {
    let mut errors = Vec::new();

    // Index: node name -> position
    let mut name_to_idx: HashMap<&str, usize> = HashMap::new();
    for (i, node) in manifest.pipeline.iter().enumerate() {
        if name_to_idx.contains_key(node.name.as_str()) {
            errors.push(format!("Duplicate node name: '{}'", node.name));
        }
        name_to_idx.insert(&node.name, i);
    }

    // Check 1: Ordering — depends_on must reference earlier nodes
    for (i, node) in manifest.pipeline.iter().enumerate() {
        for dep in &node.depends_on {
            match name_to_idx.get(dep.as_str()) {
                None => {
                    errors.push(format!(
                        "Node '{}': depends_on '{}' not found in pipeline",
                        node.name, dep
                    ));
                }
                Some(&dep_idx) => {
                    if dep_idx >= i {
                        errors.push(format!(
                            "Node '{}': depends_on '{}' is not earlier in pipeline (index {} >= {})",
                            node.name, dep, dep_idx, i
                        ));
                    }
                }
            }
        }
    }

    // Check 2: Buffer provenance — consumed buffers must be produced by prior nodes
    // First node's consumed buffers are treated as DAG inputs
    let mut available_buffers: HashSet<&str> = HashSet::new();

    // Initial inputs: buffers consumed by first node (or any node with no dependencies)
    // are implicitly available as DAG inputs
    if let Some(first) = manifest.pipeline.first() {
        for buf in &first.consumes {
            available_buffers.insert(buf);
        }
    }

    for (i, node) in manifest.pipeline.iter().enumerate() {
        // Check consumed buffers are available
        if i > 0 {
            for buf in &node.consumes {
                if !available_buffers.contains(buf.as_str()) {
                    errors.push(format!(
                        "Node '{}': consumes buffer '{}' not produced by any prior node or DAG input",
                        node.name, buf
                    ));
                }
            }
        }
        // Add produced buffers
        for buf in &node.produces {
            available_buffers.insert(buf);
        }
    }

    // Check 3: Write-write hazards — independent nodes can't produce the same buffer
    // Two nodes are independent if neither depends (transitively) on the other
    let node_count = manifest.pipeline.len();
    // Build reachability: can_reach[i][j] = true if node i depends on node j (transitively)
    let mut can_reach = vec![vec![false; node_count]; node_count];
    for (i, node) in manifest.pipeline.iter().enumerate() {
        for dep in &node.depends_on {
            if let Some(&dep_idx) = name_to_idx.get(dep.as_str()) {
                can_reach[i][dep_idx] = true;
                // Inherit transitive deps
                for k in 0..node_count {
                    if can_reach[dep_idx][k] {
                        can_reach[i][k] = true;
                    }
                }
            }
        }
    }

    for i in 0..node_count {
        for j in (i + 1)..node_count {
            let dependent = can_reach[j][i] || can_reach[i][j];
            if !dependent {
                // Independent — check for shared produced buffers
                for buf_i in &manifest.pipeline[i].produces {
                    for buf_j in &manifest.pipeline[j].produces {
                        if buf_i == buf_j {
                            errors.push(format!(
                                "Write-write hazard: independent nodes '{}' and '{}' both produce '{}'",
                                manifest.pipeline[i].name, manifest.pipeline[j].name, buf_i
                            ));
                        }
                    }
                }
            }
        }
    }

    // Check 4: Registry — every kernel is registered
    let mut registry_entries: HashMap<String, RegistryEntry> = HashMap::new();
    for node in &manifest.pipeline {
        if !registry_entries.contains_key(&node.kernel) {
            match registry::lookup_kernel(registry_dir, &node.kernel) {
                Ok(Some(entry)) => {
                    registry_entries.insert(node.kernel.clone(), entry);
                }
                Ok(None) => {
                    errors.push(format!(
                        "Node '{}': kernel '{}' not found in registry",
                        node.name, node.kernel
                    ));
                }
                Err(e) => {
                    errors.push(format!(
                        "Node '{}': registry lookup error for '{}': {}",
                        node.name, node.kernel, e
                    ));
                }
            }
        }
    }

    // Check 5: Contract match — consume/produce must match kernel contract bindings
    for node in &manifest.pipeline {
        if let Some(entry) = registry_entries.get(&node.kernel) {
            let contract = &entry.contract;

            // Collect contract input schema names
            let contract_inputs: HashSet<&str> = contract.inputs.iter()
                .map(|b| b.schema.as_str())
                .collect();
            let contract_outputs: HashSet<&str> = contract.outputs.iter()
                .map(|b| b.schema.as_str())
                .collect();

            // Check consumes match contract inputs
            for buf in &node.consumes {
                if !contract_inputs.contains(buf.as_str()) {
                    errors.push(format!(
                        "Node '{}': consumes '{}' but kernel '{}' contract has no such input",
                        node.name, buf, node.kernel
                    ));
                }
            }

            // Check produces match contract outputs
            for buf in &node.produces {
                if !contract_outputs.contains(buf.as_str()) {
                    errors.push(format!(
                        "Node '{}': produces '{}' but kernel '{}' contract has no such output",
                        node.name, buf, node.kernel
                    ));
                }
            }
        }
    }

    DagValidationResult {
        valid: errors.is_empty(),
        errors,
    }
}

pub fn load_dag_manifest(path: &Path) -> Result<DagManifest, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read DAG manifest '{}': {}", path.display(), e))?;
    serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse DAG manifest: {}", e))
}
