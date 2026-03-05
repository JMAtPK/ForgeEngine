struct Entity {
    position_x: i32,
    position_y: i32,
    velocity_x: i32,
    velocity_y: i32,
    size: i32,
    health: u32,
    max_health: u32,
    entity_type: u32,
    damage: u32,
    alive: u32,
}

// BUG: input binding declared as read_write — kernel writes to its own input
@group(0) @binding(0) var<storage, read_write> entities_in: array<Entity>;
@group(0) @binding(1) var<storage, read_write> entities_out: array<Entity>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    if (idx >= arrayLength(&entities_in)) { return; }

    var e = entities_in[idx];
    entities_out[idx] = e;

    // BUG: writes back to input buffer
    var dirty = e;
    dirty.damage = 999u;
    entities_in[idx] = dirty;
}
