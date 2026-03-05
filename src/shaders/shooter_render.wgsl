struct Entity {
    position_x: i32, position_y: i32,
    velocity_x: i32, velocity_y: i32,
    size: i32,
    health: u32, max_health: u32,
    entity_type: u32, damage: u32, alive: u32,
    cooldown: u32, color: u32,
}

struct Globals {
    frame_number: u32,
    delta_time: i32,
    world_width: i32,
    world_height: i32,
    random_seed: u32,
    padding: u32,
    padding2: u32,
    padding3: u32,
}

struct Pixel {
    pixel: u32,
}

@group(0) @binding(0) var<storage, read> entities: array<Entity>;
@group(0) @binding(1) var<storage, read> globals: array<Globals>;
@group(0) @binding(2) var<storage, read_write> framebuffer: array<Pixel>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let pixel_idx = gid.x;
    let g = globals[0];
    let width = u32(g.world_width >> 16);
    let height = u32(g.world_height >> 16);
    let total = width * height;
    if (pixel_idx >= total) { return; }

    let px = pixel_idx % width;
    let py = pixel_idx / width;

    // Convert pixel coords to fix16
    let pixel_x = i32(px) << 16;
    let pixel_y = i32(py) << 16;

    // Dark background: 0x1A1A2E → RGBA8 0xFF2E1A1A (ABGR)
    var color = 0xFF2E1A1Au;

    let n = arrayLength(&entities);
    for (var i = 0u; i < n; i++) {
        let e = entities[i];
        if (e.alive == 0u) { continue; }

        let dx = pixel_x - e.position_x;
        let dy = pixel_y - e.position_y;

        // Check if pixel is within entity rect [pos-size, pos+size]
        if (dx >= -e.size && dx < e.size && dy >= -e.size && dy < e.size) {
            color = e.color;
            break;
        }
    }

    framebuffer[pixel_idx].pixel = color;
}
