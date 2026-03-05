//! DAG runner — loads a game manifest, allocates GPU buffers with ping-pong,
//! compiles kernels, and executes the simulation pipeline.

use crate::contract::{
    KernelContractJSON, build_element_js_array, eval_postcondition, setup_postcondition_helpers,
};
use crate::dag::DagNode;
use crate::registry;
use crate::schema::{
    BufferCategory, BufferSchemaJSON, DesignParamsJSON, ResolvedParams, ResolvedSchema,
    generate_random_buffer, read_element, resolve_params, resolve_schema, validate_buffer,
};
use rquickjs::{Context, Runtime};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

// ===== Game Manifest =====

#[derive(Debug, Deserialize)]
pub struct GameManifest {
    pub design_params: DesignParamsJSON,
    pub schemas: Vec<BufferSchemaJSON>,
    pub pipeline: Vec<DagNode>,
    #[serde(default)]
    pub render_kernel: Option<String>,
}

pub fn load_game_manifest(path: &Path) -> Result<GameManifest, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read game manifest '{}': {}", path.display(), e))?;
    serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse game manifest: {}", e))
}

// ===== Buffer Pool =====

struct BufferEntry {
    buffers: [wgpu::Buffer; 2],
    current: usize,
    size: u64,
    ping_pong: bool,
    pending_swap: bool,
}

pub struct BufferPool {
    entries: HashMap<String, BufferEntry>,
}

impl BufferPool {
    pub fn new(
        device: &wgpu::Device,
        schemas: &HashMap<String, ResolvedSchema>,
    ) -> Self {
        let mut entries = HashMap::new();
        for (name, schema) in schemas {
            let size = schema.total_size as u64;
            let usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC;
            let ping_pong = schema.buffer_category == BufferCategory::State
                || schema.buffer_category == BufferCategory::Transient;

            let buf_a = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{}_A", name)),
                size,
                usage,
                mapped_at_creation: false,
            });
            let buf_b = if ping_pong {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("{}_B", name)),
                    size,
                    usage,
                    mapped_at_creation: false,
                })
            } else {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("{}_B_unused", name)),
                    size,
                    usage,
                    mapped_at_creation: false,
                })
            };

            entries.insert(
                name.clone(),
                BufferEntry {
                    buffers: [buf_a, buf_b],
                    current: 0,
                    size,
                    ping_pong,
                    pending_swap: false,
                },
            );
        }
        BufferPool { entries }
    }

    pub fn upload(&self, queue: &wgpu::Queue, name: &str, data: &[u8]) {
        if let Some(entry) = self.entries.get(name) {
            queue.write_buffer(&entry.buffers[entry.current], 0, data);
        }
    }

    pub fn current_buffer(&self, name: &str) -> Result<&wgpu::Buffer, String> {
        let entry = self.entries.get(name)
            .ok_or_else(|| format!("Buffer '{}' not in pool", name))?;
        Ok(&entry.buffers[entry.current])
    }

    /// Mark a buffer for ping-pong swap and return the write target index.
    pub fn mark_write(&mut self, name: &str) -> Result<(), String> {
        let entry = self.entries.get_mut(name)
            .ok_or_else(|| format!("Buffer '{}' not in pool", name))?;
        if entry.ping_pong {
            entry.pending_swap = true;
        }
        Ok(())
    }

    /// Get the write target buffer (alternate if ping-pong pending, current otherwise).
    pub fn write_buffer(&self, name: &str) -> Result<&wgpu::Buffer, String> {
        let entry = self.entries.get(name)
            .ok_or_else(|| format!("Buffer '{}' not in pool", name))?;
        if entry.pending_swap {
            Ok(&entry.buffers[1 - entry.current])
        } else {
            Ok(&entry.buffers[entry.current])
        }
    }

    pub fn commit_swaps(&mut self) {
        for entry in self.entries.values_mut() {
            if entry.pending_swap {
                entry.current = 1 - entry.current;
                entry.pending_swap = false;
            }
        }
    }

    pub fn readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        name: &str,
    ) -> Result<Vec<u8>, String> {
        let entry = self.entries.get(name)
            .ok_or_else(|| format!("Buffer '{}' not in pool", name))?;
        let size = entry.size;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_encoder"),
        });
        encoder.copy_buffer_to_buffer(&entry.buffers[entry.current], 0, &staging, 0, size);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::PollType::Wait).expect("GPU poll failed");
        rx.recv()
            .unwrap()
            .map_err(|e| format!("Map failed: {:?}", e))?;

        let data = slice.get_mapped_range().to_vec();
        staging.unmap();
        Ok(data)
    }
}

// ===== Kernel Cache =====

struct CompiledKernel {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    contract: KernelContractJSON,
}

struct KernelCache {
    kernels: HashMap<String, CompiledKernel>,
}

impl KernelCache {
    fn new() -> Self {
        KernelCache {
            kernels: HashMap::new(),
        }
    }

    fn compile(
        &mut self,
        device: &wgpu::Device,
        name: &str,
        contract: &KernelContractJSON,
    ) -> Result<(), String> {
        if self.kernels.contains_key(name) {
            return Ok(());
        }

        let wgsl = contract
            .wgsl_source
            .as_deref()
            .ok_or_else(|| format!("Kernel '{}': no WGSL source", name))?;

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        // Build layout entries from contract bindings
        let mut layout_entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::new();
        for b in &contract.inputs {
            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: b.binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: b.access == crate::contract::BufferAccess::Read,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        for b in &contract.outputs {
            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: b.binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        layout_entries.sort_by_key(|e| e.binding);

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("{}_bgl", name)),
                entries: &layout_entries,
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{}_pl", name)),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{}_pipeline", name)),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        self.kernels.insert(
            name.to_string(),
            CompiledKernel {
                pipeline,
                bind_group_layout,
                contract: contract.clone(),
            },
        );
        Ok(())
    }

    fn get(&self, name: &str) -> Option<&CompiledKernel> {
        self.kernels.get(name)
    }
}

// ===== DAG Runner =====

pub struct FrameResult {
    pub frame: usize,
    pub errors: Vec<String>,
    pub passed: bool,
}

pub struct DagRunner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pool: BufferPool,
    cache: KernelCache,
    pipeline: Vec<DagNode>,
    schemas: HashMap<String, ResolvedSchema>,
    params: ResolvedParams,
    frame_count: usize,
}

impl DagRunner {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        manifest: &GameManifest,
        registry_dir: &Path,
    ) -> Result<Self, String> {
        let params = resolve_params(&manifest.design_params)?;

        let mut schemas = HashMap::new();
        for schema_json in &manifest.schemas {
            let resolved = resolve_schema(schema_json, &params)?;
            schemas.insert(resolved.name.clone(), resolved);
        }

        // Validate DAG structure
        let dag_manifest = crate::dag::DagManifest {
            name: manifest.design_params.name.clone(),
            design_params: manifest.design_params.name.clone(),
            pipeline: manifest.pipeline.clone(),
        };
        let dag_result = crate::dag::validate_dag(&dag_manifest, registry_dir);
        if !dag_result.valid {
            return Err(format!("DAG validation failed: {}", dag_result.errors.join("; ")));
        }

        let pool = BufferPool::new(&device, &schemas);

        // Initialize buffers with valid random data
        for (name, schema) in &schemas {
            let data = generate_random_buffer(schema, &params)
                .map_err(|e| format!("Failed to seed buffer '{}': {}", name, e))?;
            pool.upload(&queue, name, &data);
        }

        // Compile all kernels
        let mut cache = KernelCache::new();
        for node in &manifest.pipeline {
            let entry = registry::lookup_kernel(registry_dir, &node.kernel)?
                .ok_or_else(|| format!("Kernel '{}' not in registry", node.kernel))?;
            cache.compile(&device, &node.kernel, &entry.contract)?;
        }

        Ok(DagRunner {
            device,
            queue,
            pool,
            cache,
            pipeline: manifest.pipeline.clone(),
            schemas,
            params,
            frame_count: 0,
        })
    }

    pub fn run_frame(&mut self, verify: bool) -> Result<FrameResult, String> {
        let mut errors = Vec::new();
        self.frame_count += 1;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame_encoder"),
        });

        for node in &self.pipeline.clone() {
            let compiled = self.cache.get(&node.kernel)
                .ok_or_else(|| format!("Kernel '{}' not compiled", node.kernel))?;
            let contract = &compiled.contract;

            // Mark output buffers for swap (mutable phase)
            for b in &contract.outputs {
                self.pool.mark_write(&b.schema)?;
            }

            // Determine workgroup dispatch count from first input buffer
            let mut dispatch_elements = 1u32;
            if let Some(b) = contract.inputs.first() {
                if let Some(schema) = self.schemas.get(&b.schema) {
                    dispatch_elements = (schema.total_size as u32) / 4;
                }
            }

            // Collect all binding numbers and buffer names for bind group
            let mut bindings: Vec<(u32, String, bool)> = Vec::new(); // (binding, schema, is_output)
            for b in &contract.inputs {
                bindings.push((b.binding, b.schema.clone(), false));
            }
            for b in &contract.outputs {
                bindings.push((b.binding, b.schema.clone(), true));
            }
            bindings.sort_by_key(|(binding, _, _)| *binding);

            // Build bind group entries (all immutable borrows now)
            let bg_entries: Vec<wgpu::BindGroupEntry> = bindings
                .iter()
                .map(|(binding, schema, is_output)| {
                    let buf = if *is_output {
                        self.pool.write_buffer(schema).unwrap()
                    } else {
                        self.pool.current_buffer(schema).unwrap()
                    };
                    wgpu::BindGroupEntry {
                        binding: *binding,
                        resource: buf.as_entire_binding(),
                    }
                })
                .collect();

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{}_bg", node.name)),
                layout: &compiled.bind_group_layout,
                entries: &bg_entries,
            });

            let workgroups =
                (dispatch_elements + contract.workgroup_size - 1) / contract.workgroup_size;

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&node.name),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&compiled.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            self.pool.commit_swaps();
        }

        self.queue.submit(Some(encoder.finish()));

        // Postcondition verification
        if verify {
            for node in &self.pipeline.clone() {
                let contract = &self.cache.get(&node.kernel).unwrap().contract;

                // Readback output buffers and validate
                for b in &contract.outputs {
                    if let Some(schema) = self.schemas.get(&b.schema) {
                        let data = self.pool.readback(&self.device, &self.queue, &b.schema)?;
                        let vr = validate_buffer(&data, schema, &self.params);
                        for e in vr.errors {
                            errors.push(format!(
                                "Frame {}, node '{}', buffer '{}': {}",
                                self.frame_count, node.name, b.schema, e
                            ));
                        }

                        // Postcondition eval
                        if !contract.postconditions.is_empty() {
                            // Readback inputs for postcondition
                            let mut input_data: Vec<(&str, Vec<u8>)> = Vec::new();
                            for ib in &contract.inputs {
                                let idata =
                                    self.pool.readback(&self.device, &self.queue, &ib.schema)?;
                                input_data.push((&ib.schema, idata));
                            }

                            let rt = Runtime::new().expect("QuickJS runtime");
                            let ctx = Context::full(&rt).expect("QuickJS context");

                            ctx.with(|ctx| {
                                if let Err(e) = setup_postcondition_helpers(ctx.clone()) {
                                    errors.push(format!("Postcondition setup: {}", e));
                                    return;
                                }

                                // eq and fail helpers
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

                                let mut input_js: Vec<(&str, rquickjs::Value)> = Vec::new();
                                let mut output_js: Vec<(&str, rquickjs::Value)> = Vec::new();

                                for (i, (schema_name, idata)) in
                                    input_data.iter().enumerate()
                                {
                                    if let Some(s) = self.schemas.get(*schema_name) {
                                        match build_element_js_array(&ctx, idata, s) {
                                            Ok(arr) => {
                                                let name = if i == 0 {
                                                    "input"
                                                } else {
                                                    "input_extra"
                                                };
                                                input_js.push((name, arr));
                                            }
                                            Err(e) => {
                                                errors.push(format!(
                                                    "JS array build: {}",
                                                    e
                                                ));
                                            }
                                        }
                                    }
                                }

                                match build_element_js_array(&ctx, &data, schema) {
                                    Ok(arr) => output_js.push(("output", arr)),
                                    Err(e) => {
                                        errors.push(format!("JS array build: {}", e));
                                    }
                                }

                                for (pc_idx, body) in
                                    contract.postconditions.iter().enumerate()
                                {
                                    if let Err(e) = eval_postcondition(
                                        &ctx,
                                        body,
                                        &input_js,
                                        &output_js,
                                        &self.params,
                                    ) {
                                        errors.push(format!(
                                            "Frame {}, node '{}', postcondition {}: {}",
                                            self.frame_count, node.name, pc_idx, e
                                        ));
                                    }
                                }
                            });
                        }
                    }
                }
            }
        }

        Ok(FrameResult {
            frame: self.frame_count,
            errors: errors.clone(),
            passed: errors.is_empty(),
        })
    }

    pub fn dump_buffers(&self) -> Result<serde_json::Value, String> {
        let mut buffers = serde_json::Map::new();
        for (name, schema) in &self.schemas {
            let data = self.pool.readback(&self.device, &self.queue, name)?;
            let mut elements = Vec::new();
            for i in 0..schema.capacity {
                let element = read_element(&data, schema, i);
                let mut obj = serde_json::Map::new();
                for field in &schema.fields {
                    obj.insert(
                        field.name.clone(),
                        serde_json::Value::Number(
                            serde_json::Number::from(element[&field.name]),
                        ),
                    );
                }
                elements.push(serde_json::Value::Object(obj));
            }
            buffers.insert(name.clone(), serde_json::Value::Array(elements));
        }

        Ok(serde_json::json!({
            "frame": self.frame_count,
            "buffers": buffers,
        }))
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn upload_buffer(&self, name: &str, data: &[u8]) {
        self.pool.upload(&self.queue, name, data);
    }

    pub fn current_buffer(&self, name: &str) -> Result<&wgpu::Buffer, String> {
        self.pool.current_buffer(name)
    }

    pub fn params(&self) -> &ResolvedParams {
        &self.params
    }

    pub fn compile_kernel(&mut self, contract: &KernelContractJSON) -> Result<(), String> {
        self.cache.compile(&self.device, &contract.name, contract)
    }

    pub fn dispatch_kernel(&self, encoder: &mut wgpu::CommandEncoder, name: &str) -> Result<(), String> {
        let compiled = self.cache.get(name)
            .ok_or_else(|| format!("Kernel '{}' not compiled", name))?;

        let mut bindings: Vec<(u32, String, bool)> = Vec::new();
        for b in &compiled.contract.inputs {
            bindings.push((b.binding, b.schema.clone(), false));
        }
        for b in &compiled.contract.outputs {
            bindings.push((b.binding, b.schema.clone(), true));
        }
        bindings.sort_by_key(|(binding, _, _)| *binding);

        let bg_entries: Vec<wgpu::BindGroupEntry> = bindings
            .iter()
            .map(|(binding, schema, _is_output)| {
                let buf = self.pool.current_buffer(schema).unwrap();
                wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: buf.as_entire_binding(),
                }
            })
            .collect();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{}_render_bg", name)),
            layout: &compiled.bind_group_layout,
            entries: &bg_entries,
        });

        // Use the largest buffer for dispatch count (covers render kernels
        // where the output is much larger than the input)
        let mut dispatch_elements = 1u32;
        for b in compiled.contract.inputs.iter().chain(compiled.contract.outputs.iter()) {
            if let Some(schema) = self.schemas.get(&b.schema) {
                let elems = (schema.total_size as u32) / 4;
                dispatch_elements = dispatch_elements.max(elems);
            }
        }
        let workgroups = (dispatch_elements + compiled.contract.workgroup_size - 1) / compiled.contract.workgroup_size;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(name),
                timestamp_writes: None,
            });
            pass.set_pipeline(&compiled.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        Ok(())
    }
}
