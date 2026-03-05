struct Entity {
    position_x: i32, position_y: i32,
    velocity_x: i32, velocity_y: i32,
    size: i32,
    health: u32, max_health: u32,
    entity_type: u32, damage: u32, alive: u32,
    cooldown: u32, color: u32,
}

struct CollisionResult {
    entity_a: u32,
    entity_b: u32,
    collision_type: u32,
    padding: u32,
}

struct CollisionCount {
    count: atomic<u32>,
}

@group(0) @binding(0) var<storage, read> entities: array<Entity>;
@group(0) @binding(1) var<storage, read_write> collisions: array<CollisionResult>;
@group(0) @binding(2) var<storage, read_write> collision_count: array<CollisionCount>;

fn i32_abs(x: i32) -> i32 {
    if (x < 0) { return -x; }
    return x;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    let n = arrayLength(&entities);
    if (i >= n) { return; }

    let a = entities[i];
    if (a.alive == 0u) { return; }

    for (var j = i + 1u; j < n; j++) {
        let b = entities[j];
        if (b.alive == 0u) { continue; }

        // Determine collision type:
        // bullet(3)-enemy(2) → type 1, enemy(2)-player(1) → type 2
        var ctype = 0u;
        if ((a.entity_type == 3u && b.entity_type == 2u) ||
            (a.entity_type == 2u && b.entity_type == 3u)) {
            ctype = 1u;
        } else if ((a.entity_type == 2u && b.entity_type == 1u) ||
                   (a.entity_type == 1u && b.entity_type == 2u)) {
            ctype = 2u;
        }
        if (ctype == 0u) { continue; }

        // AABB overlap check using sizes
        let combined_size = a.size + b.size;
        let dx = i32_abs(a.position_x - b.position_x);
        let dy = i32_abs(a.position_y - b.position_y);

        if (dx < combined_size && dy < combined_size) {
            let slot = atomicAdd(&collision_count[0].count, 1u);
            if (slot < arrayLength(&collisions)) {
                collisions[slot].entity_a = i;
                collisions[slot].entity_b = j;
                collisions[slot].collision_type = ctype;
                collisions[slot].padding = 0u;
            }
        }
    }
}
