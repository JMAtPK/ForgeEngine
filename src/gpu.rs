use wgpu::{self, Backends};

const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx < arrayLength(&output) {
        output[idx] = idx * 2u;
    }
}
"#;

const BUFFER_LEN: u64 = 256;
const BUFFER_SIZE: u64 = BUFFER_LEN * 4; // u32 = 4 bytes

pub fn parse_backend(name: &str) -> Backends {
    match name.to_lowercase().as_str() {
        "vulkan" => Backends::VULKAN,
        "metal" => Backends::METAL,
        "dx12" | "d3d12" => Backends::DX12,
        "gl" | "opengl" => Backends::GL,
        _ => {
            eprintln!("Unknown backend '{}', falling back to auto-detect", name);
            Backends::all()
        }
    }
}

pub fn init_device(backends: Backends) -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends,
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("No GPU adapter found. Ensure a compatible GPU and driver are available.");

    let info = adapter.get_info();
    eprintln!("GPU adapter: {} ({:?})", info.name, info.backend);

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("forge"),
        ..Default::default()
    }))
    .expect("Failed to create GPU device");

    (device, queue)
}

pub fn run_test_dispatch(device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u32> {
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test_shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("storage"),
        size: BUFFER_SIZE,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: BUFFER_SIZE,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("test_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((BUFFER_LEN as u32 + 63) / 64, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, BUFFER_SIZE);
    queue.submit(Some(encoder.finish()));

    let slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device.poll(wgpu::PollType::Wait).expect("GPU poll failed");
    rx.recv().unwrap().expect("Failed to map staging buffer");

    let data = slice.get_mapped_range();
    let result: Vec<u32> = bytemuck_cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    result
}

fn bytemuck_cast_slice(data: &[u8]) -> &[u32] {
    assert!(data.len() % 4 == 0);
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u32, data.len() / 4) }
}

pub fn verify_results(results: &[u32]) -> bool {
    if results.len() != BUFFER_LEN as usize {
        eprintln!("Expected {} values, got {}", BUFFER_LEN, results.len());
        return false;
    }
    for (i, &val) in results.iter().enumerate() {
        if val != (i as u32) * 2 {
            eprintln!("Mismatch at index {}: expected {}, got {}", i, i * 2, val);
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_round_trip() {
        let (device, queue) = init_device(Backends::all());
        let results = run_test_dispatch(&device, &queue);
        assert!(verify_results(&results), "GPU round-trip verification failed");
    }
}
