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
    count: u32,
}

@group(0) @binding(0) var<storage, read> entities_in: array<Entity>;
@group(0) @binding(1) var<storage, read> collisions: array<CollisionResult>;
@group(0) @binding(2) var<storage, read> collision_count: array<CollisionCount>;
@group(0) @binding(3) var<storage, read_write> entities_out: array<Entity>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    if (idx >= arrayLength(&entities_in)) { return; }

    var e = entities_in[idx];

    if (e.alive == 0u) {
        entities_out[idx] = e;
        return;
    }

    let num_collisions = min(collision_count[0].count, arrayLength(&collisions));

    // Accumulate damage from all collisions involving this entity
    var total_damage = 0u;
    for (var c = 0u; c < num_collisions; c++) {
        let col = collisions[c];
        if (col.entity_a == idx) {
            total_damage += entities_in[col.entity_b].damage;
        } else if (col.entity_b == idx) {
            total_damage += entities_in[col.entity_a].damage;
        }
    }

    if (total_damage > 0u) {
        if (total_damage >= e.health) {
            e.health = 0u;
            e.alive = 0u;
            e.entity_type = 0u;
        } else {
            e.health -= total_damage;
        }
    }

    entities_out[idx] = e;
}
