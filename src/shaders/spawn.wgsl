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
@group(0) @binding(1) var<storage, read> input_buf: array<Input>;
@group(0) @binding(2) var<storage, read> globals: array<Globals>;
@group(0) @binding(3) var<storage, read_write> entities_out: array<Entity>;

const PLAYER_HEALTH: u32 = 100u;
const PLAYER_SIZE: i32 = 16 * 65536;    // fix16(16)
const ENEMY_HEALTH: u32 = 30u;
const ENEMY_SIZE: i32 = 12 * 65536;     // fix16(12)
const ENEMY_SPEED: i32 = 80 * 65536;    // fix16(80)
const BULLET_DAMAGE: u32 = 10u;
const BULLET_HEALTH: u32 = 1u;
const BULLET_SIZE: i32 = 4 * 65536;     // fix16(4)
const BULLET_SPEED: i32 = 500 * 65536;  // fix16(500)
const SHOOT_COOLDOWN: u32 = 10u;
const ENEMY_SPAWN_INTERVAL: u32 = 60u;

var<workgroup> spawn_bullet: atomic<u32>;
var<workgroup> spawn_enemy: atomic<u32>;
var<workgroup> player_pos_x: i32;
var<workgroup> player_pos_y: i32;
var<workgroup> bullet_vx: i32;
var<workgroup> bullet_vy: i32;
var<workgroup> enemy_pos_x: i32;
var<workgroup> enemy_pos_y: i32;
var<workgroup> enemy_vx: i32;
var<workgroup> enemy_vy: i32;

fn isqrt(n: u32) -> u32 {
    if (n == 0u) { return 0u; }
    var x = n;
    var y = (x + 1u) / 2u;
    for (var i = 0u; i < 20u; i++) {
        if (y >= x) { break; }
        x = y;
        y = (x + n / x) / 2u;
    }
    return x;
}

fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wid: vec3u,
) {
    let idx = gid.x;
    let n = arrayLength(&entities_in);
    let g = globals[0];
    let inp = input_buf[0];

    // Phase 1: Thread 0 of workgroup 0 handles all spawn decisions
    if (wid.x == 0u && lid.x == 0u) {
        atomicStore(&spawn_bullet, 0u);
        atomicStore(&spawn_enemy, 0u);

        // Find player
        for (var i = 0u; i < n; i++) {
            if (entities_in[i].entity_type == 1u) {
                player_pos_x = entities_in[i].position_x;
                player_pos_y = entities_in[i].position_y;

                // Check bullet spawn: space pressed & cooldown == 0
                let space = (inp.keys_down >> 4u) & 1u;
                if (space == 1u && entities_in[i].cooldown == 0u) {
                    atomicStore(&spawn_bullet, 1u);

                    // Compute bullet direction toward mouse
                    let dx = inp.mouse_x - player_pos_x;
                    let dy = inp.mouse_y - player_pos_y;

                    // Normalize using f32 (TODO: replace with integer math)
                    let fdx = f32(dx) / 65536.0;
                    let fdy = f32(dy) / 65536.0;
                    let mag = sqrt(fdx * fdx + fdy * fdy);

                    if (mag > 0.001) {
                        let nx = fdx / mag;
                        let ny = fdy / mag;
                        bullet_vx = i32(nx * f32(BULLET_SPEED));
                        bullet_vy = i32(ny * f32(BULLET_SPEED));
                    } else {
                        // Default: fire right
                        bullet_vx = BULLET_SPEED;
                        bullet_vy = 0;
                    }
                }
                break;
            }
        }

        // Check enemy spawn
        if (g.frame_number % ENEMY_SPAWN_INTERVAL == 0u && g.frame_number > 0u) {
            atomicStore(&spawn_enemy, 1u);

            // Enemy position: random edge using PCG hash
            let h = pcg_hash(g.random_seed ^ g.frame_number);
            let side = h & 3u;  // 0=top, 1=bottom, 2=left, 3=right

            let ww = g.world_width;
            let wh = g.world_height;

            // Random position along that edge
            let h2 = pcg_hash(h);

            if (side == 0u) {
                // Top edge
                enemy_pos_x = i32(h2 % u32(ww >> 16)) << 16;
                enemy_pos_y = 0;
            } else if (side == 1u) {
                // Bottom edge
                enemy_pos_x = i32(h2 % u32(ww >> 16)) << 16;
                enemy_pos_y = wh;
            } else if (side == 2u) {
                // Left edge
                enemy_pos_x = 0;
                enemy_pos_y = i32(h2 % u32(wh >> 16)) << 16;
            } else {
                // Right edge
                enemy_pos_x = ww;
                enemy_pos_y = i32(h2 % u32(wh >> 16)) << 16;
            }

            // Velocity toward player
            let edx = player_pos_x - enemy_pos_x;
            let edy = player_pos_y - enemy_pos_y;

            let fedx = f32(edx) / 65536.0;
            let fedy = f32(edy) / 65536.0;
            let emag = sqrt(fedx * fedx + fedy * fedy);

            if (emag > 0.001) {
                let enx = fedx / emag;
                let eny = fedy / emag;
                enemy_vx = i32(enx * f32(ENEMY_SPEED));
                enemy_vy = i32(eny * f32(ENEMY_SPEED));
            } else {
                enemy_vx = 0;
                enemy_vy = 0;
            }
        }
    }

    workgroupBarrier();

    if (idx >= n) { return; }

    var e = entities_in[idx];

    // Player init: if slot 0 is dead and has no player, create one
    if (idx == 0u && e.alive == 0u) {
        e.position_x = g.world_width / 2;
        e.position_y = g.world_height / 2;
        e.velocity_x = 0;
        e.velocity_y = 0;
        e.size = PLAYER_SIZE;
        e.health = PLAYER_HEALTH;
        e.max_health = PLAYER_HEALTH;
        e.entity_type = 1u;
        e.damage = 0u;
        e.alive = 1u;
        e.cooldown = 0u;
        e.color = 0xFF00FF00u;  // Green ABGR
        entities_out[idx] = e;
        return;
    }

    // Player: update cooldown
    if (e.entity_type == 1u) {
        if (atomicLoad(&spawn_bullet) == 1u) {
            e.cooldown = SHOOT_COOLDOWN;
        } else if (e.cooldown > 0u) {
            e.cooldown -= 1u;
        }
        entities_out[idx] = e;
        return;
    }

    // Dead slot: try to claim for bullet or enemy spawn
    if (e.alive == 0u) {
        // Try bullet spawn first
        if (atomicLoad(&spawn_bullet) == 1u) {
            let claimed = atomicCompareExchangeWeak(&spawn_bullet, 1u, 0u);
            if (claimed.exchanged) {
                var bullet: Entity;
                bullet.position_x = player_pos_x;
                bullet.position_y = player_pos_y;
                bullet.velocity_x = bullet_vx;
                bullet.velocity_y = bullet_vy;
                bullet.size = BULLET_SIZE;
                bullet.health = BULLET_HEALTH;
                bullet.max_health = BULLET_HEALTH;
                bullet.entity_type = 3u;
                bullet.damage = BULLET_DAMAGE;
                bullet.alive = 1u;
                bullet.cooldown = 0u;
                bullet.color = 0xFF00FFFFu;  // Yellow ABGR
                entities_out[idx] = bullet;
                return;
            }
        }

        // Try enemy spawn
        if (atomicLoad(&spawn_enemy) == 1u) {
            let claimed = atomicCompareExchangeWeak(&spawn_enemy, 1u, 0u);
            if (claimed.exchanged) {
                var enemy: Entity;
                enemy.position_x = enemy_pos_x;
                enemy.position_y = enemy_pos_y;
                enemy.velocity_x = enemy_vx;
                enemy.velocity_y = enemy_vy;
                enemy.size = ENEMY_SIZE;
                enemy.health = ENEMY_HEALTH;
                enemy.max_health = ENEMY_HEALTH;
                enemy.entity_type = 2u;
                enemy.damage = 0u;
                enemy.alive = 1u;
                enemy.cooldown = 0u;
                enemy.color = 0xFF0000FFu;  // Red ABGR
                entities_out[idx] = enemy;
                return;
            }
        }
    }

    // Pass through unchanged
    entities_out[idx] = e;
}
