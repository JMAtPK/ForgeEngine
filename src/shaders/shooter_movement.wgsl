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

fn fix16_mul(a: i32, b: i32) -> i32 {
    let a_hi = a >> 16;
    let a_lo = u32(a) & 0xFFFFu;
    let b_hi = b >> 16;
    let b_lo = u32(b) & 0xFFFFu;
    return ((a_hi * b_hi) << 16)
         + a_hi * i32(b_lo)
         + i32(a_lo) * b_hi
         + i32((a_lo * b_lo) >> 16u);
}

@group(0) @binding(0) var<storage, read> entities_in: array<Entity>;
@group(0) @binding(1) var<storage, read> globals: array<Globals>;
@group(0) @binding(2) var<storage, read_write> entities_out: array<Entity>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    if (idx >= arrayLength(&entities_in)) { return; }

    var e = entities_in[idx];

    if (e.alive == 0u) {
        entities_out[idx] = e;
        return;
    }

    let g = globals[0];
    e.position_x = clamp(e.position_x + fix16_mul(e.velocity_x, g.delta_time), 0, g.world_width);
    e.position_y = clamp(e.position_y + fix16_mul(e.velocity_y, g.delta_time), 0, g.world_height);

    entities_out[idx] = e;
}
