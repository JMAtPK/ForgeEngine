struct Entity {
    position_x: i32, position_y: i32,
    velocity_x: i32, velocity_y: i32,
    size: i32,
    health: u32, max_health: u32,
    entity_type: u32, damage: u32, alive: u32,
    cooldown: u32, color: u32,
}

struct Input {
    keys_down: u32,
    mouse_x: i32,
    mouse_y: i32,
    mouse_buttons: u32,
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
@group(0) @binding(1) var<storage, read> input_buf: array<Input>;
@group(0) @binding(2) var<storage, read_write> entities_out: array<Entity>;

const PLAYER_SPEED: i32 = 200 * 65536;  // fix16(200)
const FIX16_INV_SQRT2: i32 = 46341;     // fix16(0.7071)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    if (idx >= arrayLength(&entities_in)) { return; }

    var e = entities_in[idx];

    // Only modify the player entity (type 1)
    if (e.entity_type == 1u) {
        let keys = input_buf[0].keys_down;
        let dx = i32((keys >> 3u) & 1u) - i32((keys >> 1u) & 1u); // D - A
        let dy = i32((keys >> 2u) & 1u) - i32(keys & 1u);         // S - W

        let diagonal = (dx != 0 && dy != 0);

        if (diagonal) {
            e.velocity_x = fix16_mul(dx * PLAYER_SPEED, FIX16_INV_SQRT2);
            e.velocity_y = fix16_mul(dy * PLAYER_SPEED, FIX16_INV_SQRT2);
        } else {
            e.velocity_x = dx * PLAYER_SPEED;
            e.velocity_y = dy * PLAYER_SPEED;
        }
    }

    entities_out[idx] = e;
}
