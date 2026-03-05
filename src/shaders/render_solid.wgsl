// Render kernel: fills Framebuffer with a cycling color pattern.

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

@group(0) @binding(0) var<storage, read> globals: array<Globals>;
@group(0) @binding(1) var<storage, read_write> framebuffer: array<Pixel>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let g = globals[0];
    let width = u32(g.world_width >> 16);
    let height = u32(g.world_height >> 16);
    let total = width * height;
    if (idx >= total) { return; }

    let x = idx % width;
    let y = idx / width;

    let phase = g.frame_number / 2u;
    let r = (x + phase) % 256u;
    let g_ch = (y + phase) % 256u;
    let b = (x + y + phase) % 256u;

    // RGBA8: 0xAABBGGRR
    framebuffer[idx].pixel = r | (g_ch << 8u) | (b << 16u) | (255u << 24u);
}
