@group(0) @binding(0) var<storage, read_write> framebuffer: array<u32>;

struct Params {
    width: u32,
    height: u32,
    color: u32,
}
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i < params.width * params.height) {
        framebuffer[i] = params.color;
    }
}
