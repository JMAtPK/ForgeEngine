//! Contract checker for ForgeEngine
//!
//! Dispatches compute kernels on the GPU and verifies that outputs satisfy
//! schema constraints and postconditions.


use crate::schema::{
    BufferCategory, ResolvedParams, ResolvedSchema, read_element, validate_buffer,
    generate_random_buffer, generate_adversarial_buffers,
};
use rquickjs::{Context, Function, Object, Runtime};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ===== Contract Data Types =====

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BufferAccess {
    Read,
    Write,
    #[serde(rename = "readwrite")]
    ReadWrite,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BufferBinding {
    pub binding: u32,
    pub schema: String,
    pub access: BufferAccess,
    #[serde(default)]
    pub alias: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct KernelContractJSON {
    pub name: String,
    pub wgsl_path: Option<String>,
    #[serde(default)]
    pub wgsl_source: Option<String>,
    pub inputs: Vec<BufferBinding>,
    pub outputs: Vec<BufferBinding>,
    #[serde(default = "default_workgroup_size")]
    pub workgroup_size: u32,
    #[serde(default)]
    pub postconditions: Vec<String>,
}

fn default_workgroup_size() -> u32 {
    64
}

pub struct DispatchResult {
    pub outputs: Vec<Vec<u8>>,
    pub inputs_post: Vec<Vec<u8>>,
}

#[derive(Debug, Serialize)]
pub struct VerificationResult {
    pub accepted: bool,
    pub errors: Vec<String>,
}

// ===== GPU Dispatch =====

pub struct BufferSlot {
    pub binding: u32,
    pub data: Vec<u8>,
    pub read_only: bool,
    pub is_input: bool,
}

pub fn dispatch_kernel(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    buffers: &[BufferSlot],
    workgroup_size: u32,
) -> Result<DispatchResult, String> {
    // Compile shader
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("contract_shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });

    // Create GPU buffers
    let mut gpu_buffers = Vec::new();
    for slot in buffers {
        let size = slot.data.len() as u64;
        let usage = if slot.read_only {
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
        } else {
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
        };
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("binding_{}", slot.binding)),
            size,
            usage,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, &slot.data);
        gpu_buffers.push((slot.binding, buf, size, slot.read_only, slot.is_input));
    }

    // Build bind group layout entries
    let layout_entries: Vec<wgpu::BindGroupLayoutEntry> = gpu_buffers
        .iter()
        .map(|(binding, _, _, read_only, _)| wgpu::BindGroupLayoutEntry {
            binding: *binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: *read_only,
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("contract_bgl"),
        entries: &layout_entries,
    });

    let bind_group_entries: Vec<wgpu::BindGroupEntry> = gpu_buffers
        .iter()
        .map(|(binding, buf, _, _, _)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: buf.as_entire_binding(),
        })
        .collect();

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("contract_bg"),
        layout: &bind_group_layout,
        entries: &bind_group_entries,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("contract_pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("contract_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Determine dispatch count from first buffer element count
    let first_buf = buffers.first().ok_or("No buffers provided")?;
    // Assume first input buffer determines element count
    // struct_size will be divided out by caller — use raw byte count / 4 for u32 arrays
    // For entity buffers, the caller provides the right element count via buffer size
    let element_count = (first_buf.data.len() as u32) / 4; // conservative: at least this many u32s
    // Use the actual workgroup dispatch based on the contract's workgroup size
    // The kernel uses arrayLength internally, so we just need enough workgroups
    let workgroups = (element_count + workgroup_size - 1) / workgroup_size;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("contract_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("contract_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    // Create staging buffers and copy
    let mut staging_buffers = Vec::new();
    for (_, buf, size, _, _) in &gpu_buffers {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: *size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, *size);
        staging_buffers.push(staging);
    }

    queue.submit(Some(encoder.finish()));

    // Read back all buffers
    let mut readbacks = Vec::new();
    for staging in &staging_buffers {
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::PollType::Wait).expect("GPU poll failed");
        rx.recv().unwrap().map_err(|e| format!("Map failed: {:?}", e))?;

        let data = slice.get_mapped_range().to_vec();
        readbacks.push(data);
        staging.unmap();
    }

    // Split readbacks into inputs_post and outputs based on is_input flag
    let mut outputs = Vec::new();
    let mut inputs_post = Vec::new();
    for (i, (_, _, _, _, is_input)) in gpu_buffers.iter().enumerate() {
        if *is_input {
            inputs_post.push(readbacks[i].clone());
        } else {
            outputs.push(readbacks[i].clone());
        }
    }

    Ok(DispatchResult {
        outputs,
        inputs_post,
    })
}

// ===== Postcondition Sandbox =====

pub(crate) fn setup_postcondition_helpers(ctx: rquickjs::Ctx<'_>) -> rquickjs::Result<()> {
    // fix16_mul
    ctx.globals().set(
        "fix16_mul",
        Function::new(ctx.clone(), |a: f64, b: f64| -> f64 {
            let a = a as i64;
            let b = b as i64;
            ((a * b) >> 16) as f64
        })?,
    )?;

    // clamp
    ctx.globals().set(
        "clamp",
        Function::new(ctx.clone(), |value: f64, min: f64, max: f64| -> f64 {
            let v = value as i64;
            let mn = min as i64;
            let mx = max as i64;
            v.max(mn).min(mx) as f64
        })?,
    )?;

    // abs
    ctx.globals().set(
        "abs",
        Function::new(ctx.clone(), |value: f64| -> f64 {
            (value as i64).abs() as f64
        })?,
    )?;

    // min
    ctx.globals().set(
        "min",
        Function::new(ctx.clone(), |a: f64, b: f64| -> f64 {
            (a as i64).min(b as i64) as f64
        })?,
    )?;

    // max
    ctx.globals().set(
        "max",
        Function::new(ctx.clone(), |a: f64, b: f64| -> f64 {
            (a as i64).max(b as i64) as f64
        })?,
    )?;

    Ok(())
}

pub(crate) fn build_element_js_array<'js>(
    ctx: &rquickjs::Ctx<'js>,
    buf: &[u8],
    schema: &ResolvedSchema,
) -> rquickjs::Result<rquickjs::Value<'js>> {
    let arr = rquickjs::Array::new(ctx.clone())?;
    for i in 0..schema.capacity {
        let element = read_element(buf, schema, i);
        let obj = Object::new(ctx.clone())?;
        for (name, val) in &element {
            obj.set(name.as_str(), *val as f64)?;
        }
        arr.set(i, obj)?;
    }
    Ok(arr.into_value())
}

pub(crate) fn eval_postcondition<'js>(
    ctx: &rquickjs::Ctx<'js>,
    body: &str,
    input_arrays: &[(String, rquickjs::Value<'js>)],
    output_arrays: &[(String, rquickjs::Value<'js>)],
    params: &ResolvedParams,
) -> Result<(), String> {
    // Set up $ with params
    let dollar = Object::new(ctx.clone()).map_err(|e| format!("$ object: {}", e))?;
    for (k, v) in &params.raw {
        dollar
            .set(k.as_str(), *v as f64)
            .map_err(|e| format!("$ set: {}", e))?;
    }
    ctx.globals()
        .set("$", dollar)
        .map_err(|e| format!("$ global: {}", e))?;

    // Set all named arrays as globals (includes "input"/"output" and aliases)
    for (name, val) in input_arrays {
        ctx.globals()
            .set(name.as_str(), val.clone())
            .map_err(|e| format!("{} global: {}", name, e))?;
    }
    for (name, val) in output_arrays {
        ctx.globals()
            .set(name.as_str(), val.clone())
            .map_err(|e| format!("{} global: {}", name, e))?;
    }

    // Wrap postcondition body in a function and evaluate
    let script = format!(
        r#"(function() {{
            {}
        }})()"#,
        body
    );

    let result: rquickjs::Value = ctx
        .eval(script.clone())
        .map_err(|e| format!("Postcondition eval error: {}", e))?;

    // Check result: true = pass, false = fail, object with .failed = fail
    if let Some(obj) = result.as_object() {
        if let Ok(failed) = obj.get::<_, bool>("failed") {
            if failed {
                let msg: String = obj.get("message").unwrap_or_default();
                let idx: f64 = obj.get("index").unwrap_or(-1.0);
                return Err(format!(
                    "Postcondition failed at index {}: {}",
                    idx as i64, msg
                ));
            }
        }
    }

    if result.as_bool() == Some(false) {
        return Err(format!("Postcondition returned false: {}", body));
    }

    // Check for undefined/null — also a failure
    if result.is_undefined() || result.is_null() {
        return Err(format!("Postcondition returned undefined/null: {}", body));
    }

    Ok(())
}

// ===== Verification Pipeline =====

pub fn verify_contract(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    contract: &KernelContractJSON,
    schemas: &HashMap<String, ResolvedSchema>,
    params: &ResolvedParams,
) -> VerificationResult {
    let mut errors = Vec::new();

    // Get WGSL source
    let wgsl = match (&contract.wgsl_source, &contract.wgsl_path) {
        (Some(src), _) => src.clone(),
        (None, Some(path)) => match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                return VerificationResult {
                    accepted: false,
                    errors: vec![format!("Failed to read WGSL file '{}': {}", path, e)],
                };
            }
        },
        (None, None) => {
            return VerificationResult {
                accepted: false,
                errors: vec!["No WGSL source or path provided".into()],
            };
        }
    };

    // Resolve schemas for each binding
    let mut input_schemas: Vec<(u32, &ResolvedSchema)> = Vec::new();
    for b in &contract.inputs {
        match schemas.get(&b.schema) {
            Some(s) => input_schemas.push((b.binding, s)),
            None => {
                errors.push(format!("Unknown input schema: {}", b.schema));
            }
        }
    }
    let mut output_schemas: Vec<(u32, &ResolvedSchema)> = Vec::new();
    for b in &contract.outputs {
        match schemas.get(&b.schema) {
            Some(s) => output_schemas.push((b.binding, s)),
            None => {
                errors.push(format!("Unknown output schema: {}", b.schema));
            }
        }
    }
    if !errors.is_empty() {
        return VerificationResult {
            accepted: false,
            errors,
        };
    }

    // Generate test inputs: adversarial + random
    let primary_input_schema = input_schemas
        .iter()
        .find(|(_, s)| s.buffer_category == BufferCategory::State)
        .or(input_schemas.first());

    let test_input_sets: Vec<Vec<u8>> = if let Some((_, schema)) = primary_input_schema {
        let mut sets = match generate_adversarial_buffers(schema, params) {
            Ok(bufs) => bufs,
            Err(e) => {
                return VerificationResult {
                    accepted: false,
                    errors: vec![format!("Failed to generate adversarial buffers: {}", e)],
                };
            }
        };
        // Add a random buffer
        match generate_random_buffer(schema, params) {
            Ok(buf) => sets.push(buf),
            Err(e) => {
                errors.push(format!("Warning: random buffer generation failed: {}", e));
            }
        }
        sets
    } else {
        return VerificationResult {
            accepted: false,
            errors: vec!["No input schemas found".into()],
        };
    };

    // Run verification for each test input set
    for (set_idx, primary_input) in test_input_sets.iter().enumerate() {
        // Build buffer slots
        let mut buffer_slots = Vec::new();
        let mut input_snapshots: Vec<(u32, Vec<u8>)> = Vec::new();

        for (binding, schema) in &input_schemas {
            let is_primary = primary_input_schema
                .map(|(b, _)| *b == *binding)
                .unwrap_or(false);
            let data = if schema.buffer_category == BufferCategory::State || is_primary {
                primary_input.clone()
            } else {
                // For non-primary inputs (e.g., globals), generate a random buffer
                match generate_random_buffer(schema, params) {
                    Ok(buf) => buf,
                    Err(e) => {
                        errors.push(format!(
                            "Set {}: failed to generate buffer for binding {}: {}",
                            set_idx, binding, e
                        ));
                        continue;
                    }
                }
            };
            input_snapshots.push((*binding, data.clone()));
            let access = contract
                .inputs
                .iter()
                .find(|b| b.binding == *binding)
                .map(|b| b.access)
                .unwrap_or(BufferAccess::Read);
            buffer_slots.push(BufferSlot {
                binding: *binding,
                data,
                read_only: access == BufferAccess::Read,
                is_input: true,
            });
        }

        for (binding, schema) in &output_schemas {
            buffer_slots.push(BufferSlot {
                binding: *binding,
                data: vec![0u8; schema.total_size],
                read_only: false,
                is_input: false,
            });
        }

        // Sort by binding number for consistent layout
        buffer_slots.sort_by_key(|s| s.binding);

        // Dispatch
        let dispatch_result = match dispatch_kernel(
            device,
            queue,
            &wgsl,
            &buffer_slots,
            contract.workgroup_size,
        ) {
            Ok(r) => r,
            Err(e) => {
                errors.push(format!("Set {}: dispatch failed: {}", set_idx, e));
                continue;
            }
        };

        // Clean write check: compare input snapshots with post-dispatch input data
        for (i, (binding, snapshot)) in input_snapshots.iter().enumerate() {
            if i < dispatch_result.inputs_post.len() {
                let post = &dispatch_result.inputs_post[i];
                if snapshot != post {
                    errors.push(format!(
                        "Set {}: clean write check failed — input buffer (binding {}) was modified by kernel",
                        set_idx, binding
                    ));
                }
            }
        }

        // Schema validation on outputs
        for (i, (_, schema)) in output_schemas.iter().enumerate() {
            if i < dispatch_result.outputs.len() {
                let result = validate_buffer(&dispatch_result.outputs[i], schema, params);
                if !result.valid {
                    for err in &result.errors {
                        errors.push(format!("Set {}: output validation: {}", set_idx, err));
                    }
                }
            }
        }

        // Postcondition evaluation
        if !contract.postconditions.is_empty() && errors.is_empty() {
            let rt = Runtime::new().expect("QuickJS runtime");
            let ctx = Context::full(&rt).expect("QuickJS context");

            ctx.with(|ctx| {
                if let Err(e) = setup_postcondition_helpers(ctx.clone()) {
                    errors.push(format!("Set {}: postcondition setup failed: {}", set_idx, e));
                    return;
                }

                // Build JS arrays for inputs and outputs
                let mut input_js: Vec<(String, rquickjs::Value)> = Vec::new();
                let mut output_js: Vec<(String, rquickjs::Value)> = Vec::new();

                // Use input snapshots (pre-dispatch) for postcondition evaluation
                for (snap_idx, (_, snapshot)) in input_snapshots.iter().enumerate() {
                    let schema = input_schemas[snap_idx].1;
                    match build_element_js_array(&ctx, snapshot, schema) {
                        Ok(arr) => {
                            let default_name = if snap_idx == 0 { "input" } else { "input_extra" };
                            input_js.push((default_name.to_string(), arr.clone()));
                            if let Some(alias) = contract.inputs.get(snap_idx).and_then(|b| b.alias.as_ref()) {
                                if schema.capacity == 1 {
                                    // Capacity-1: alias is the singular unwrapped object
                                    if let Some(elem) = arr.as_array().and_then(|a| a.get::<rquickjs::Value>(0).ok()) {
                                        input_js.push((alias.clone(), elem));
                                    }
                                } else {
                                    input_js.push((alias.clone(), arr));
                                }
                            }
                        }
                        Err(e) => {
                            errors.push(format!(
                                "Set {}: failed to build input JS array: {}",
                                set_idx, e
                            ));
                        }
                    }
                }

                for (out_idx, (_, schema)) in output_schemas.iter().enumerate() {
                    if out_idx < dispatch_result.outputs.len() {
                        match build_element_js_array(&ctx, &dispatch_result.outputs[out_idx], schema)
                        {
                            Ok(arr) => {
                                let default_name = if out_idx == 0 { "output" } else { "output_extra" };
                                output_js.push((default_name.to_string(), arr.clone()));
                                if let Some(alias) = contract.outputs.get(out_idx).and_then(|b| b.alias.as_ref()) {
                                    if schema.capacity == 1 {
                                        if let Some(elem) = arr.as_array().and_then(|a| a.get::<rquickjs::Value>(0).ok()) {
                                            output_js.push((alias.clone(), elem));
                                        }
                                    } else {
                                        output_js.push((alias.clone(), arr));
                                    }
                                }
                            }
                            Err(e) => {
                                errors.push(format!(
                                    "Set {}: failed to build output JS array: {}",
                                    set_idx, e
                                ));
                            }
                        }
                    }
                }

                // Set up eq and fail helpers
                let _ = ctx.eval::<rquickjs::Value, _>(
                    r#"
                    function eq(a, b) {
                        if (a === b) return true;
                        if (typeof a !== 'object' || typeof b !== 'object') return a === b;
                        var keys = Object.keys(a);
                        for (var i = 0; i < keys.length; i++) {
                            if (a[keys[i]] !== b[keys[i]]) return false;
                        }
                        return true;
                    }
                    function fail(index, message) {
                        return { failed: true, index: index, message: message };
                    }
                    "#,
                );

                for (pc_idx, body) in contract.postconditions.iter().enumerate() {
                    match eval_postcondition(&ctx, body, &input_js, &output_js, params) {
                        Ok(()) => {}
                        Err(e) => {
                            errors.push(format!(
                                "Set {}: postcondition {}: {}",
                                set_idx, pc_idx, e
                            ));
                        }
                    }
                }
            });
        }

        // If we already found errors, we can stop early
        if !errors.is_empty() {
            break;
        }
    }

    VerificationResult {
        accepted: errors.is_empty(),
        errors,
    }
}

// ===== Tests =====

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu;
    use crate::schema::{
        BufferSchemaJSON, DesignParamsJSON, resolve_params, resolve_schema, to_fix16,
        write_element,
    };

    fn test_design_params() -> DesignParamsJSON {
        serde_json::from_str(
            r#"{
            "name": "test_design",
            "params": {
                "world_width": { "type": "fix16", "value": 1280 },
                "world_height": { "type": "fix16", "value": 720 },
                "max_speed": { "type": "fix16", "value": 500 },
                "player_health": { "type": "u32", "value": 100 },
                "enemy_health": { "type": "u32", "value": 30 },
                "boss_health": { "type": "u32", "value": 200 },
                "bullet_damage": { "type": "u32", "value": 10 },
                "bullet_health": { "type": "u32", "value": 1 },
                "max_entities": { "type": "u32", "value": 8 },
                "player_size": { "type": "fix16", "value": 16 },
                "max_delta_time": { "type": "fix16", "value": 0.05 }
            },
            "design_invariants": []
        }"#,
        )
        .unwrap()
    }

    fn entity_schema_json() -> BufferSchemaJSON {
        serde_json::from_str(
            r#"{
            "name": "EntityBuffer",
            "struct": {
                "position": {
                    "type": "fix16x2",
                    "range": [[0, "$world_width"], [0, "$world_height"]]
                },
                "velocity": {
                    "type": "fix16x2",
                    "range": [["$-max_speed", "$max_speed"], ["$-max_speed", "$max_speed"]]
                },
                "size": { "type": "i32", "range": [0, "$player_size"] },
                "health": { "type": "u32", "range": [0, "$boss_health"] },
                "max_health": { "type": "u32", "range": [0, "$boss_health"] },
                "entity_type": { "type": "u32", "enum": [0, 1, 2, 3] },
                "damage": { "type": "u32", "range": [0, "$bullet_damage"] },
                "alive": { "type": "u32", "enum": [0, 1] }
            },
            "capacity": "$max_entities",
            "buffer_category": "state",
            "invariants": ["health <= max_health"]
        }"#,
        )
        .unwrap()
    }

    fn globals_schema_json() -> BufferSchemaJSON {
        serde_json::from_str(
            r#"{
            "name": "GlobalsBuffer",
            "struct": {
                "delta_time": { "type": "i32", "range": [0, "$max_delta_time"] },
                "world_width": { "type": "i32" },
                "world_height": { "type": "i32" },
                "frame_count": { "type": "u32" }
            },
            "capacity": 1,
            "buffer_category": "input",
            "invariants": []
        }"#,
        )
        .unwrap()
    }

    fn build_test_context() -> (
        wgpu::Device,
        wgpu::Queue,
        ResolvedParams,
        HashMap<String, ResolvedSchema>,
    ) {
        let (device, queue) = gpu::init_device(wgpu::Backends::all());
        let design = test_design_params();
        let params = resolve_params(&design).unwrap();
        let entity_schema = resolve_schema(&entity_schema_json(), &params).unwrap();
        let globals_schema = resolve_schema(&globals_schema_json(), &params).unwrap();

        let mut schemas = HashMap::new();
        schemas.insert("EntityBuffer".into(), entity_schema);
        schemas.insert("GlobalsBuffer".into(), globals_schema);

        (device, queue, params, schemas)
    }

    fn load_shader(name: &str) -> String {
        let path = format!(
            "{}/src/shaders/{}.wgsl",
            env!("CARGO_MANIFEST_DIR"),
            name
        );
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("Failed to load {}: {}", path, e))
    }

    fn make_globals_buffer(params: &ResolvedParams, schema: &ResolvedSchema) -> Vec<u8> {
        let mut buf = vec![0u8; schema.total_size];
        let mut vals = HashMap::new();
        vals.insert("delta_time".into(), to_fix16(0.016));
        vals.insert("world_width".into(), params.raw["world_width"]);
        vals.insert("world_height".into(), params.raw["world_height"]);
        vals.insert("frame_count".into(), 1);
        write_element(&mut buf, schema, 0, &vals);
        buf
    }

    // Test 1: identity kernel passes
    #[test]
    fn identity_passes() {
        let (device, queue, params, schemas) = build_test_context();
        let wgsl = load_shader("test_identity");

        let contract = KernelContractJSON {
            name: "identity".into(),
            wgsl_source: Some(wgsl),
            wgsl_path: None,
            inputs: vec![BufferBinding {
                binding: 0,
                schema: "EntityBuffer".into(),
                access: BufferAccess::Read,
                alias: None,
            }],
            outputs: vec![BufferBinding {
                binding: 1,
                schema: "EntityBuffer".into(),
                access: BufferAccess::Write,
                alias: None,
            }],
            workgroup_size: 64,
            postconditions: vec![
                r#"
                for (var i = 0; i < input.length; i++) {
                    if (!eq(input[i], output[i])) {
                        return fail(i, "output differs from input");
                    }
                }
                return true;
                "#
                .into(),
            ],
        };

        let result = verify_contract(&device, &queue, &contract, &schemas, &params);
        assert!(
            result.accepted,
            "identity should pass: {:?}",
            result.errors
        );
    }

    // Test 2: movement kernel passes
    #[test]
    fn movement_passes() {
        let (device, queue, params, schemas) = build_test_context();
        let wgsl = load_shader("test_movement");
        let globals_schema = &schemas["GlobalsBuffer"];
        let _globals_buf = make_globals_buffer(&params, globals_schema);

        // For movement, we need a custom dispatch since it has 3 bindings
        // The verify_contract uses the primary input schema for generation
        // We need to handle globals specially

        // Build contract — the globals buffer is a second input
        let contract = KernelContractJSON {
            name: "movement".into(),
            wgsl_source: Some(wgsl),
            wgsl_path: None,
            inputs: vec![
                BufferBinding {
                    binding: 0,
                    schema: "EntityBuffer".into(),
                    access: BufferAccess::Read,
                    alias: None,
                },
                BufferBinding {
                    binding: 1,
                    schema: "GlobalsBuffer".into(),
                    access: BufferAccess::Read,
                    alias: None,
                },
            ],
            outputs: vec![BufferBinding {
                binding: 2,
                schema: "EntityBuffer".into(),
                access: BufferAccess::Write,
                alias: None,
            }],
            workgroup_size: 64,
            postconditions: vec![
                r#"
                for (var i = 0; i < input.length; i++) {
                    if (input[i].alive === 0) {
                        if (!eq(input[i], output[i])) {
                            return fail(i, "dead entity was modified");
                        }
                    }
                }
                return true;
                "#
                .into(),
            ],
        };

        let result = verify_contract(&device, &queue, &contract, &schemas, &params);
        assert!(
            result.accepted,
            "movement should pass: {:?}",
            result.errors
        );
    }

    // Test 3: break_invariant caught
    #[test]
    fn break_invariant_caught() {
        let (device, queue, params, schemas) = build_test_context();
        let wgsl = load_shader("test_break_invariant");

        let contract = KernelContractJSON {
            name: "break_invariant".into(),
            wgsl_source: Some(wgsl),
            wgsl_path: None,
            inputs: vec![BufferBinding {
                binding: 0,
                schema: "EntityBuffer".into(),
                access: BufferAccess::Read,
                alias: None,
            }],
            outputs: vec![BufferBinding {
                binding: 1,
                schema: "EntityBuffer".into(),
                access: BufferAccess::Write,
                alias: None,
            }],
            workgroup_size: 64,
            postconditions: vec![],
        };

        let result = verify_contract(&device, &queue, &contract, &schemas, &params);
        assert!(
            !result.accepted,
            "break_invariant should fail"
        );
        let has_invariant_error = result
            .errors
            .iter()
            .any(|e| e.contains("invariant") || e.contains("range"));
        assert!(
            has_invariant_error,
            "Should have invariant/range error: {:?}",
            result.errors
        );
    }

    // Test 4: dirty_input caught
    #[test]
    fn dirty_input_caught() {
        let (device, queue, params, schemas) = build_test_context();
        let wgsl = load_shader("test_dirty_input");

        // dirty_input.wgsl declares input as read_write in WGSL
        // The contract says it should be read-only, but GPU needs read_write to actually test
        // We declare it as ReadWrite in the contract to let the GPU write,
        // but the clean write check should still catch it
        let contract = KernelContractJSON {
            name: "dirty_input".into(),
            wgsl_source: Some(wgsl),
            wgsl_path: None,
            inputs: vec![BufferBinding {
                binding: 0,
                schema: "EntityBuffer".into(),
                access: BufferAccess::ReadWrite,
                alias: None,
            }],
            outputs: vec![BufferBinding {
                binding: 1,
                schema: "EntityBuffer".into(),
                access: BufferAccess::Write,
                alias: None,
            }],
            workgroup_size: 64,
            postconditions: vec![],
        };

        let result = verify_contract(&device, &queue, &contract, &schemas, &params);
        assert!(
            !result.accepted,
            "dirty_input should fail"
        );
        let has_clean_write_error = result
            .errors
            .iter()
            .any(|e| e.contains("clean write") || e.contains("modified"));
        assert!(
            has_clean_write_error,
            "Should have clean write error: {:?}",
            result.errors
        );
    }

    // Test 5: movement_no_clamp caught
    #[test]
    fn movement_no_clamp_caught() {
        let (device, queue, params, schemas) = build_test_context();
        let wgsl = load_shader("test_movement_no_clamp");

        let contract = KernelContractJSON {
            name: "movement_no_clamp".into(),
            wgsl_source: Some(wgsl),
            wgsl_path: None,
            inputs: vec![
                BufferBinding {
                    binding: 0,
                    schema: "EntityBuffer".into(),
                    access: BufferAccess::Read,
                    alias: None,
                },
                BufferBinding {
                    binding: 1,
                    schema: "GlobalsBuffer".into(),
                    access: BufferAccess::Read,
                    alias: None,
                },
            ],
            outputs: vec![BufferBinding {
                binding: 2,
                schema: "EntityBuffer".into(),
                access: BufferAccess::Write,
                alias: None,
            }],
            workgroup_size: 64,
            postconditions: vec![],
        };

        let result = verify_contract(&device, &queue, &contract, &schemas, &params);
        assert!(
            !result.accepted,
            "movement_no_clamp should fail"
        );
        let has_range_error = result
            .errors
            .iter()
            .any(|e| e.contains("range") || e.contains("validation"));
        assert!(
            has_range_error,
            "Should have range error: {:?}",
            result.errors
        );
    }

    // Test 6: modify_dead caught
    #[test]
    fn modify_dead_caught() {
        let (device, queue, params, schemas) = build_test_context();
        let wgsl = load_shader("test_modify_dead");

        let contract = KernelContractJSON {
            name: "modify_dead".into(),
            wgsl_source: Some(wgsl),
            wgsl_path: None,
            inputs: vec![
                BufferBinding {
                    binding: 0,
                    schema: "EntityBuffer".into(),
                    access: BufferAccess::Read,
                    alias: None,
                },
                BufferBinding {
                    binding: 1,
                    schema: "GlobalsBuffer".into(),
                    access: BufferAccess::Read,
                    alias: None,
                },
            ],
            outputs: vec![BufferBinding {
                binding: 2,
                schema: "EntityBuffer".into(),
                access: BufferAccess::Write,
                alias: None,
            }],
            workgroup_size: 64,
            postconditions: vec![
                r#"
                for (var i = 0; i < input.length; i++) {
                    if (input[i].alive === 0) {
                        if (!eq(input[i], output[i])) {
                            return fail(i, "dead entity was modified");
                        }
                    }
                }
                return true;
                "#
                .into(),
            ],
        };

        let result = verify_contract(&device, &queue, &contract, &schemas, &params);
        assert!(
            !result.accepted,
            "modify_dead should fail"
        );
        let has_postcondition_error = result
            .errors
            .iter()
            .any(|e| e.contains("postcondition") || e.contains("dead entity"));
        assert!(
            has_postcondition_error,
            "Should have postcondition error about dead entities: {:?}",
            result.errors
        );
    }
}
