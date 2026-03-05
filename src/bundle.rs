//! Bundle loading, WGSL reflection via naga, and full verification pipeline.

use crate::contract::{KernelContractJSON, VerificationResult, verify_contract};
use crate::schema::{
    BufferSchemaJSON, DesignParamsJSON, ResolvedSchema,
    check_design_invariants, resolve_params, resolve_schema,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ===== Bundle format =====

#[derive(Debug, Deserialize)]
pub struct VerifyBundle {
    pub design_params: DesignParamsJSON,
    pub schemas: Vec<BufferSchemaJSON>,
    pub contract: KernelContractJSON,
}

// ===== Verification output =====

#[derive(Debug, Serialize)]
pub struct StepResult {
    pub step: String,
    pub passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub test_sets: Option<usize>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct VerifyOutput {
    pub accepted: bool,
    pub kernel: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub registry_id: Option<String>,
    pub errors: Vec<String>,
    pub steps: Vec<StepResult>,
}

// ===== Bundle loading =====

pub fn load_bundle(path: &Path) -> Result<VerifyBundle, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read bundle '{}': {}", path.display(), e))?;
    let mut bundle: VerifyBundle = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse bundle JSON: {}", e))?;

    // Resolve wgsl_path relative to bundle directory
    if bundle.contract.wgsl_source.is_none() {
        if let Some(ref wgsl_path) = bundle.contract.wgsl_path {
            let bundle_dir = path.parent().unwrap_or(Path::new("."));
            let resolved = bundle_dir.join(wgsl_path);
            let source = std::fs::read_to_string(&resolved)
                .map_err(|e| format!("Failed to read WGSL '{}': {}", resolved.display(), e))?;
            bundle.contract.wgsl_source = Some(source);
        }
    }

    Ok(bundle)
}

// ===== Naga-based WGSL reflection =====

pub fn check_bindings(
    module: &naga::Module,
    contract: &KernelContractJSON,
) -> Vec<String> {
    let mut errors = Vec::new();

    // Collect declared bindings from contract
    let mut expected: HashMap<u32, (&str, bool)> = HashMap::new(); // binding -> (kind, needs_store)
    for b in &contract.inputs {
        let needs_store = b.access == crate::contract::BufferAccess::ReadWrite;
        expected.insert(b.binding, ("input", needs_store));
    }
    for b in &contract.outputs {
        expected.insert(b.binding, ("output", true));
    }

    // Walk shader global variables
    let mut found_bindings: HashMap<u32, bool> = HashMap::new();
    for (_, var) in module.global_variables.iter() {
        let binding = match &var.binding {
            Some(b) if b.group == 0 => b.binding,
            _ => continue,
        };

        let has_store = match var.space {
            naga::AddressSpace::Storage { access } => {
                access.contains(naga::StorageAccess::STORE)
            }
            _ => continue,
        };

        found_bindings.insert(binding, has_store);

        if !expected.contains_key(&binding) {
            errors.push(format!(
                "Shader declares undeclared binding @binding({})",
                binding
            ));
        }
    }

    // Check all contract bindings exist in shader
    for (binding, (kind, needs_store)) in &expected {
        match found_bindings.get(binding) {
            None => {
                errors.push(format!(
                    "Contract {} binding {} not found in shader",
                    kind, binding
                ));
            }
            Some(has_store) => {
                if *needs_store && !has_store {
                    errors.push(format!(
                        "Binding {} needs write access but shader declares read-only",
                        binding
                    ));
                }
            }
        }
    }

    errors
}

fn naga_scalar_matches_field_type(
    scalar: naga::Scalar,
    field_type: crate::schema::FieldType,
) -> bool {
    use crate::schema::FieldType;
    match field_type {
        FieldType::U32 => {
            scalar.kind == naga::ScalarKind::Uint && scalar.width == 4
        }
        FieldType::I32 | FieldType::Fix16 => {
            scalar.kind == naga::ScalarKind::Sint && scalar.width == 4
        }
        _ => false,
    }
}

pub fn check_struct_layouts(
    module: &naga::Module,
    contract: &KernelContractJSON,
    schemas: &HashMap<String, ResolvedSchema>,
) -> Vec<String> {
    let mut errors = Vec::new();

    // Build binding -> schema name map
    let mut binding_to_schema: HashMap<u32, &str> = HashMap::new();
    for b in &contract.inputs {
        binding_to_schema.insert(b.binding, &b.schema);
    }
    for b in &contract.outputs {
        binding_to_schema.insert(b.binding, &b.schema);
    }

    for (_, var) in module.global_variables.iter() {
        let binding_num = match &var.binding {
            Some(b) if b.group == 0 => b.binding,
            _ => continue,
        };

        let schema_name = match binding_to_schema.get(&binding_num) {
            Some(name) => *name,
            None => continue,
        };

        let schema = match schemas.get(schema_name) {
            Some(s) => s,
            None => continue, // Already caught by schema resolution
        };

        // Resolve the element type: var.ty -> Array { base } -> Struct
        let var_ty = &module.types[var.ty];
        let struct_type = match &var_ty.inner {
            naga::TypeInner::Array { base, .. } => {
                match &module.types[*base].inner {
                    naga::TypeInner::Struct { members, .. } => members,
                    _ => {
                        errors.push(format!(
                            "Binding {}: array element is not a struct",
                            binding_num
                        ));
                        continue;
                    }
                }
            }
            naga::TypeInner::Struct { members, .. } => members,
            _ => continue,
        };

        // Compare struct members against schema fields
        if struct_type.len() != schema.fields.len() {
            errors.push(format!(
                "Binding {} ({}): shader struct has {} fields, schema has {}",
                binding_num, schema_name, struct_type.len(), schema.fields.len()
            ));
            continue;
        }

        for (i, member) in struct_type.iter().enumerate() {
            let field = &schema.fields[i];

            // Check name
            if let Some(ref name) = member.name {
                if name != &field.name {
                    errors.push(format!(
                        "Binding {} field {}: name mismatch — shader '{}', schema '{}'",
                        binding_num, i, name, field.name
                    ));
                }
            }

            // Check offset
            if member.offset as usize != field.byte_offset {
                errors.push(format!(
                    "Binding {} field '{}': offset mismatch — shader {}, schema {}",
                    binding_num, field.name, member.offset, field.byte_offset
                ));
            }

            // Check scalar type (also accept atomic<T> matching T)
            let member_ty = &module.types[member.ty];
            match &member_ty.inner {
                naga::TypeInner::Scalar(scalar) => {
                    if !naga_scalar_matches_field_type(*scalar, field.type_) {
                        errors.push(format!(
                            "Binding {} field '{}': type mismatch — shader {:?}, schema {:?}",
                            binding_num, field.name, scalar.kind, field.type_
                        ));
                    }
                }
                naga::TypeInner::Atomic(scalar) => {
                    if !naga_scalar_matches_field_type(*scalar, field.type_) {
                        errors.push(format!(
                            "Binding {} field '{}': atomic type mismatch — shader {:?}, schema {:?}",
                            binding_num, field.name, scalar.kind, field.type_
                        ));
                    }
                }
                _ => {
                    errors.push(format!(
                        "Binding {} field '{}': expected scalar type in shader",
                        binding_num, field.name
                    ));
                }
            }
        }
    }

    errors
}

// ===== Full pipeline =====

pub fn run_full_pipeline(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bundle: &VerifyBundle,
) -> VerifyOutput {
    let kernel_name = bundle.contract.name.clone();
    let mut steps = Vec::new();
    let mut all_errors = Vec::new();

    // Step 1: Design params — resolve and check invariants
    let params = match resolve_params(&bundle.design_params) {
        Ok(p) => p,
        Err(e) => {
            steps.push(StepResult {
                step: "design_params".into(),
                passed: false,
                test_sets: None,
                errors: vec![e.clone()],
            });
            return VerifyOutput {
                accepted: false,
                kernel: kernel_name,
                registry_id: None,
                errors: vec![e],
                steps,
            };
        }
    };

    let inv_result = check_design_invariants(&params, &bundle.design_params.design_invariants);
    if !inv_result.valid {
        steps.push(StepResult {
            step: "design_params".into(),
            passed: false,
            test_sets: None,
            errors: inv_result.violations.clone(),
        });
        return VerifyOutput {
            accepted: false,
            kernel: kernel_name,
            registry_id: None,
            errors: inv_result.violations,
            steps,
        };
    }
    steps.push(StepResult {
        step: "design_params".into(),
        passed: true,
        test_sets: None,
        errors: vec![],
    });

    // Resolve schemas
    let mut schemas: HashMap<String, ResolvedSchema> = HashMap::new();
    for schema_json in &bundle.schemas {
        match resolve_schema(schema_json, &params) {
            Ok(s) => {
                schemas.insert(s.name.clone(), s);
            }
            Err(e) => {
                all_errors.push(e);
            }
        }
    }
    if !all_errors.is_empty() {
        steps.push(StepResult {
            step: "schema_resolution".into(),
            passed: false,
            test_sets: None,
            errors: all_errors.clone(),
        });
        return VerifyOutput {
            accepted: false,
            kernel: kernel_name,
            registry_id: None,
            errors: all_errors,
            steps,
        };
    }

    // Step 2: WGSL compilation via naga
    let wgsl_source = bundle.contract.wgsl_source.as_deref().unwrap_or("");
    let module = match naga::front::wgsl::parse_str(wgsl_source) {
        Ok(m) => m,
        Err(e) => {
            let err = format!("WGSL parse error: {}", e);
            steps.push(StepResult {
                step: "compilation".into(),
                passed: false,
                test_sets: None,
                errors: vec![err.clone()],
            });
            return VerifyOutput {
                accepted: false,
                kernel: kernel_name,
                registry_id: None,
                errors: vec![err],
                steps,
            };
        }
    };
    steps.push(StepResult {
        step: "compilation".into(),
        passed: true,
        test_sets: None,
        errors: vec![],
    });

    // Step 3: Binding check
    let binding_errors = check_bindings(&module, &bundle.contract);
    if !binding_errors.is_empty() {
        steps.push(StepResult {
            step: "binding_check".into(),
            passed: false,
            test_sets: None,
            errors: binding_errors.clone(),
        });
        return VerifyOutput {
            accepted: false,
            kernel: kernel_name,
            registry_id: None,
            errors: binding_errors,
            steps,
        };
    }
    steps.push(StepResult {
        step: "binding_check".into(),
        passed: true,
        test_sets: None,
        errors: vec![],
    });

    // Step 4: Structural schema check
    let layout_errors = check_struct_layouts(&module, &bundle.contract, &schemas);
    if !layout_errors.is_empty() {
        steps.push(StepResult {
            step: "struct_check".into(),
            passed: false,
            test_sets: None,
            errors: layout_errors.clone(),
        });
        return VerifyOutput {
            accepted: false,
            kernel: kernel_name,
            registry_id: None,
            errors: layout_errors,
            steps,
        };
    }
    steps.push(StepResult {
        step: "struct_check".into(),
        passed: true,
        test_sets: None,
        errors: vec![],
    });

    // Step 5: GPU execution + schema validation + postconditions (via verify_contract)
    let result: VerificationResult = verify_contract(
        device,
        queue,
        &bundle.contract,
        &schemas,
        &params,
    );

    // Determine test set count from contract input schemas
    let test_set_count = {
        let primary = bundle.contract.inputs.first()
            .and_then(|b| schemas.get(&b.schema));
        match primary {
            Some(_) => Some(6), // 4 adversarial + 1 single-active + 1 random
            None => None,
        }
    };

    if result.accepted {
        steps.push(StepResult {
            step: "execution".into(),
            passed: true,
            test_sets: test_set_count,
            errors: vec![],
        });
        steps.push(StepResult {
            step: "schema_validation".into(),
            passed: true,
            test_sets: None,
            errors: vec![],
        });
        steps.push(StepResult {
            step: "postconditions".into(),
            passed: true,
            test_sets: None,
            errors: vec![],
        });
        steps.push(StepResult {
            step: "clean_write".into(),
            passed: true,
            test_sets: None,
            errors: vec![],
        });
    } else {
        // Categorize errors
        let mut exec_errors = Vec::new();
        let mut schema_errors = Vec::new();
        let mut post_errors = Vec::new();
        let mut clean_errors = Vec::new();

        for err in &result.errors {
            if err.contains("clean write") || err.contains("modified by kernel") {
                clean_errors.push(err.clone());
            } else if err.contains("postcondition") || err.contains("dead entity") {
                post_errors.push(err.clone());
            } else if err.contains("validation") || err.contains("range") || err.contains("invariant") || err.contains("enum") {
                schema_errors.push(err.clone());
            } else {
                exec_errors.push(err.clone());
            }
        }

        steps.push(StepResult {
            step: "execution".into(),
            passed: exec_errors.is_empty(),
            test_sets: test_set_count,
            errors: exec_errors,
        });
        steps.push(StepResult {
            step: "schema_validation".into(),
            passed: schema_errors.is_empty(),
            test_sets: None,
            errors: schema_errors,
        });
        steps.push(StepResult {
            step: "postconditions".into(),
            passed: post_errors.is_empty(),
            test_sets: None,
            errors: post_errors,
        });
        steps.push(StepResult {
            step: "clean_write".into(),
            passed: clean_errors.is_empty(),
            test_sets: None,
            errors: clean_errors,
        });
    }

    VerifyOutput {
        accepted: result.accepted,
        kernel: kernel_name,
        registry_id: None,
        errors: result.errors,
        steps,
    }
}

/// Compute a simple hash for the bundle content (for registry IDs).
pub fn bundle_hash(content: &str) -> String {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    format!("{:08x}", hasher.finish() as u32)
}
