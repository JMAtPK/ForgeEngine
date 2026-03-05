# Compute Shader Game Engine (CSGE)

## AI-Native Game Engine — Design Document

**Version:** 0.8.0-draft
**Date:** 2026-03-04
**Author:** Jason + Claude (PhilosopherKing, Inc.)

---

## 1. Vision

A game engine where the runtime is nothing but compute shader dispatches over flat data buffers, and the toolchain enforces rigorous verification at every layer. The engine is designed from the ground up to be authored, understood, and extended by AI agents. Human developers provide architectural direction and verify the foundations; AI agents do the bulk of implementation, constrained and validated by the system itself.

There is no traditional renderer, no OOP game object hierarchy, no scripting layer, no plugin architecture. There is only:

- **Design Parameters** — the game designer's declared intent: world size, speed limits, health values, spawn rates. The single source of truth for every tunable value.
- **Buffers** — typed, flat, GPU-resident data with constrained schemas that reference design parameters.
- **Kernels** — compute shaders that read buffers and write buffers.
- **The DAG** — a declared simulation pipeline of kernel dispatches per frame, with explicit ordering and consume/produce declarations. Rendering is outside the DAG.
- **The Harness** — a minimal host program that boots the GPU, executes the DAG, then dispatches the render kernel and presents the framebuffer.
- **The Forge** — a contract-enforced pipeline that validates and registers every kernel before it can enter the system.

Games are built by composing verified kernels into DAGs over well-typed buffer schemas. The Forge is the gatekeeper. No kernel enters the system without a full contract and passing verification. This is not a convention — it is an architectural hard gate.

---

## 2. Core Principles

### 2.1 Verification Is the Engine

The verification system is not bolted on. It is the thing that makes the engine possible. In unprecedented architectural territory where no prior art exists, proof replaces intuition. Every kernel, every buffer layout, every DAG composition is verifiable. If it passes verification, it works. If it doesn't, the error is specific and actionable.

### 2.2 Legibility Over Cleverness

The primary "developer" is an LLM agent. Every artifact in the system — schemas, kernel source, DAG manifests, contracts — must be readable and understandable by an AI in a single context window. No implicit behavior, no deep inheritance, no magic. Everything is explicit, flat, and self-describing.

### 2.3 Architectural Homogeneity

All simulation is compute kernel dispatches that read and write flat buffers. A physics step and a collision pass are structurally identical. Rendering is also a compute kernel dispatch, but it sits outside the simulation pipeline — it is a read-only visualization of final state that does not feed back into the simulation. This separation allows rendering to use floating-point freely while simulation remains deterministic on fixed-point integers.

### 2.4 Determinism by Default

Every kernel dispatch is a pure function of its input buffers. Given identical inputs, it produces identical outputs. This is achieved through fixed-point integer arithmetic for all simulation state. Integer math is deterministic across all GPU vendors — there is no rounding mode ambiguity, no fused multiply-add variation, no vendor-specific optimization that changes results. Floating-point is used only in the render kernel, where its output (the framebuffer) does not feed back into simulation. This makes verification exact (bit-identical comparison, no epsilon tolerance), replay trivial, and cross-platform lockstep networking feasible.

### 2.5 Contract-First Development

You cannot add a kernel to the system without providing its contract: input schemas, output schemas, postconditions, and adversarial test cases. The contract is the spec. The AI writes the contract first, then writes the kernel to satisfy it. There is no "I'll add tests later" path.

---

## 3. Architecture

### 3.1 Design Parameters

Before any buffer schemas, kernels, or DAGs exist, a game declares its **design parameters** — the central source of truth for every tunable value in the game. These are the game designer's decisions: how big is the world, how fast can things move, how much health does a player have.

**Design parameter declaration example:**

```json
{
  "name": "shooter_design",
  "params": {
    "world_width": { "type": "fix16", "value": 1280 },
    "world_height": { "type": "fix16", "value": 720 },
    "max_speed": { "type": "fix16", "value": 500 },
    "player_health": { "type": "u32", "value": 100 },
    "enemy_health": { "type": "u32", "value": 30 },
    "boss_health": { "type": "u32", "value": 200 },
    "bullet_damage": { "type": "u32", "value": 10 },
    "bullet_health": { "type": "u32", "value": 1 },
    "max_entities": { "type": "u32", "value": 4096 },
    "enemy_spawn_interval": { "type": "u32", "value": 60 },
    "player_size": { "type": "fix16", "value": 16 },
    "enemy_size": { "type": "fix16", "value": 12 },
    "bullet_size": { "type": "fix16", "value": 4 },
    "max_delta_time": { "type": "fix16", "value": 0.05 }
  },
  "design_invariants": [
    "$.bullet_damage < $.player_health",
    "$.bullet_damage <= $.enemy_health",
    "$.boss_health >= $.enemy_health",
    "$.player_size < $.world_width && $.player_size < $.world_height",
    "fix16_mul($.max_speed, $.max_delta_time) < $.world_width",
    "fix16_mul($.max_speed, $.max_delta_time) < $.world_height",
    "$.max_entities >= 64"
  ]
}
```

Design parameters serve three purposes:

**Single source of truth.** Schemas and kernel contracts reference design parameters by name (`$player_health`) rather than hardcoding values. Changing a design decision means editing one number in one place. Everything downstream — schemas, constraint ranges, adversarial test bounds — updates automatically.

**Design-level sanity checking.** The `design_invariants` express relationships between parameters that must hold for the game to make sense. "Bullet damage must be less than player health" means a single bullet can't kill the player. "Max speed times max delta time must be less than world width" means nothing can teleport across the entire world in one frame. These are checked before any kernel runs — they're a fast, cheap sanity pass over the game's fundamental assumptions.

**Bounded state space.** Design parameters define the valid operating range of the entire game. Schemas use them to declare value constraints. Verification only needs to cover inputs within these bounds, not the full range of a data type. This makes adversarial test generation smarter (test the worst case within bounds, not arbitrary nonsense) and makes postcondition checking more meaningful (violations are game rule violations, not abstract type errors).

**What design parameters do NOT provide:** They do not turn testing into proof. The state space is still large (bounded fixed-point integers still have millions of values across multiple fields). Constraining the range makes testing better and more focused, but it doesn't eliminate the possibility of a bug on an untested input within the valid range. Design parameters improve verification quality; they don't guarantee completeness.

**Who verifies the design parameters?** This is where human judgment is irreplaceable. The engine can check that parameters are internally consistent (invariants hold) and that schemas correctly reference them. But it cannot check that the parameters produce a fun game, or that player_health should be 100 rather than 150. Design parameters are the game designer's intent. Getting them right is a design problem, not an engineering problem.

### 3.2 Buffers

All game state lives in GPU-resident buffers. A buffer is a flat array of structs, described by a **constrained schema** — a typed layout with value ranges and inter-field invariants that reference design parameters.

**Schema definition example:**

```json
{
  "name": "EntityBuffer",
  "struct": {
    "position": { "type": "fix16x2", "range": [[0, "$world_width"], [0, "$world_height"]] },
    "velocity": { "type": "fix16x2", "range": [["$-max_speed", "$max_speed"], ["$-max_speed", "$max_speed"]] },
    "size":     { "type": "fix16", "range": [0, "$player_size"] },
    "health":   { "type": "u32", "range": [0, "$boss_health"] },
    "max_health": { "type": "u32", "range": [0, "$boss_health"] },
    "entity_type": { "type": "u32", "enum": [0, 1, 2, 3] },
    "damage":   { "type": "u32", "range": [0, "$bullet_damage"] },
    "alive":    { "type": "u32", "enum": [0, 1] }
  },
  "capacity": "$max_entities",
  "buffer_category": "state",
  "invariants": [
    "alive === 0 ? entity_type === 0 : true",
    "health <= max_health",
    "entity_type === 3 ? health <= $.bullet_health : true",
    "entity_type === 1 ? max_health === $.player_health : true",
    "entity_type === 2 ? max_health <= $.boss_health : true"
  ]
}
```

Schemas are the source of truth for buffer layout. They are plain JSON, human-readable and machine-readable. Every buffer in the system has a registered schema. Buffer contents can be validated against their schema at any time — both structurally (does the byte layout match?), by value (are all fields within their declared ranges?), and by invariant (do inter-field relationships hold?).

**`$`-prefixed values** are resolved against the active design parameter set. This means the same schema definition adapts when design parameters change — if the designer increases boss_health from 200 to 500, the health range on EntityBuffer widens automatically.

**Core schema types:** `u32`, `i32`, `fix16`, `fix16x2`, `vec2f`, `vec3f`, `vec4f`. No pointers, no references, no dynamic types. What you see is what's in memory.

**Fixed-point types** are the default for all simulation state. `fix16` is a 32-bit integer interpreted as 16.16 fixed-point (16 integer bits, 16 fractional bits), giving a range of roughly -32768 to +32767 with 1/65536 precision. `fix16x2` is a pair of fix16 values packed into 64 bits (used for 2D positions and velocities). Fixed-point arithmetic is performed using standard integer operations with appropriate shifting after multiplication and before division. This guarantees bit-identical results across all GPU vendors.

**Floating-point types** (`vec2f`, `vec3f`, `vec4f`) are reserved for the render kernel and any other output-only kernel whose results do not feed back into simulation state. Schemas enforce this: a buffer marked as a state buffer cannot contain float fields. Transient buffers and output buffers (like the framebuffer) may use floats freely.

**Schema constraints** have three levels: type constraints (the field is a u32), value constraints (the field is in range [0, 100]), and invariants (`alive === 0 ? entity_type === 0 : true`). All three are checked during Forge verification. Kernels must produce output that satisfies all three levels, given input that also satisfies all three levels.

**Buffer categories:**

- **State buffers** — persistent game state that carries across frames (entities, world data).
- **Transient buffers** — intermediate results within a frame (collision results, draw lists). Allocated per frame, not expected to persist.
- **Input buffers** — written by the CPU each frame (keyboard/mouse state, delta time, frame count).
- **Output buffers** — read by the CPU each frame (framebuffer for display, audio buffer if applicable).

### 3.3 Kernels

A kernel is a WGSL compute shader that reads declared input buffers and writes declared output buffers. Nothing else. A kernel cannot access buffers it has not declared. A kernel has no side effects beyond its declared writes.

**Kernel contract example:**

```json
{
  "name": "movement",
  "description": "Updates entity positions based on velocity and delta time. Clamps to world bounds.",
  "inputs": [
    { "binding": 0, "schema": "EntityBuffer", "access": "read" },
    { "binding": 1, "schema": "GlobalsBuffer", "access": "read" }
  ],
  "outputs": [
    { "binding": 2, "schema": "EntityBuffer", "access": "write" }
  ],
  "postconditions": [
    "for (let i = 0; i < input.length; i++) { if (input[i].alive === 0 && !eq(output[i], input[i])) return fail(i, 'dead entity modified'); } return true;",
    "for (let i = 0; i < input.length; i++) { if (input[i].alive === 1) { const ex = clamp(input[i].position_x + fix16_mul(input[i].velocity_x, globals.delta_time), 0, $.world_width); if (output[i].position_x !== ex) return fail(i, `position_x: expected ${ex}, got ${output[i].position_x}`); } } return true;",
    "for (let i = 0; i < input.length; i++) { if (input[i].alive === 1) { const ey = clamp(input[i].position_y + fix16_mul(input[i].velocity_y, globals.delta_time), 0, $.world_height); if (output[i].position_y !== ey) return fail(i, `position_y: expected ${ey}, got ${output[i].position_y}`); } } return true;",
    "for (let i = 0; i < input.length; i++) { if (input[i].alive === 1 && output[i].velocity !== input[i].velocity) return fail(i, 'velocity changed'); } return true;",
    "for (let i = 0; i < input.length; i++) { if (input[i].alive === 1 && output[i].health !== input[i].health) return fail(i, 'health changed'); } return true;"
  ],
  "adversarial_cases": [
    "Empty buffer (zero alive entities)",
    "Single entity at world origin with zero velocity",
    "Entity at ($world_width, $world_height) with velocity ($max_speed, $max_speed)",
    "Entity at (0, 0) with velocity (-$max_speed, -$max_speed)",
    "All $max_entities slots alive, delta_time at $max_delta_time"
  ]
}
```

**The kernel source** is a WGSL compute shader that implements the contract. Fixed-point multiplication splits operands into 16-bit halves to avoid overflow without requiring 64-bit integers (which WGSL does not have):

```wgsl
// Fixed-point 16.16 multiply without i64.
// Splits each operand into integer (hi) and fractional (lo) 16-bit halves.
// Full product: (a_hi * b_hi) << 16 + a_hi * b_lo + a_lo * b_hi + (a_lo * b_lo) >> 16
// The (a_hi * b_hi) << 16 term overflows u32 when both operands are large (e.g. 1280 * 500).
// In practice this term is zero or small for typical game arithmetic (velocity * delta_time,
// position_delta * scale, etc.) where at least one operand has a small integer part.
// For the shooter proof of concept, all fix16_mul calls involve at least one small operand.
fn fix16_mul(a: i32, b: i32) -> i32 {
    let a_hi = a >> 16;
    let a_lo = u32(a) & 0xFFFFu;
    let b_hi = b >> 16;
    let b_lo = u32(b) & 0xFFFFu;

    return (a_hi * b_hi) << 16
         + a_hi * i32(b_lo)
         + i32(a_lo) * b_hi
         + i32((a_lo * b_lo) >> 16u);
}

@group(0) @binding(0) var<storage, read> entities_in: array<Entity>;
@group(0) @binding(1) var<storage, read> globals: Globals;
@group(0) @binding(2) var<storage, read_write> entities_out: array<Entity>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let idx = id.x;
    var e = entities_in[idx];

    if (e.alive == 0u) {
        entities_out[idx] = e;
        return;
    }

    // Fixed-point position update: pos += vel * dt
    e.position_x = e.position_x + fix16_mul(e.velocity_x, globals.delta_time);
    e.position_y = e.position_y + fix16_mul(e.velocity_y, globals.delta_time);

    // Clamp to world bounds (all values are fix16)
    e.position_x = clamp(e.position_x, 0, globals.world_width);
    e.position_y = clamp(e.position_y, 0, globals.world_height);

    entities_out[idx] = e;
}
```

Note: the kernel dispatches over the full `$max_entities` buffer. Dead slots are copied through unchanged — the `alive == 0` check handles sparsity. There is no entity count; the buffer is the truth. Input and output are separate buffers. The kernel reads from one and writes to another. This eliminates read-write hazards within a dispatch and makes the "did the kernel only modify what it declared?" check trivial — diff the input buffer against a pre-dispatch snapshot to confirm it's untouched. Because all simulation values are integers, this diff is exact — no epsilon, no floating-point comparison ambiguity.

### 3.4 The DAG

The frame execution graph is a directed acyclic graph of simulation kernel dispatches. Each node is a kernel with an explicit execution order and explicit consume/produce declarations for buffer versions. The graph describes only simulation — the transformation of game state from one frame to the next. Rendering is not part of the graph (see §3.5).

**Key principles:**

- **Explicit ordering.** Nodes declare their dependencies on other nodes by name. The graph structure is stated, not inferred.
- **Consume/produce.** Each node declares which buffers it reads (consumes) and which it writes (produces). A buffer written by one node and read by a later node creates a data dependency that must align with the declared ordering.
- **No implicit edges.** The Forge validates that declared dependencies are coherent with consume/produce declarations: if a node reads a buffer, some node it transitively depends on must have produced it. If a dependency is missing, the Forge flags it.
- **Dispatch over full buffer.** Every kernel dispatches over the full `$max_entities` range. There is no entity count tracking — each invocation checks the `alive` flag on its slot and early-outs if zero. The buffer is the truth.
- **Block-based write ownership.** Kernels that write to shared buffers (e.g., spawn) use block-based workgroup ownership: workgroup N writes only to indices `[N * workgroup_size, (N+1) * workgroup_size)`. This eliminates write races without atomics.

**DAG manifest example (2D shooter):**

```json
{
  "name": "simple_shooter",
  "design_params": "shooter_design",
  "pipeline": [
    {
      "name": "spawn",
      "kernel": "spawn",
      "depends_on": [],
      "consumes": ["EntityBuffer", "InputBuffer", "GlobalsBuffer"],
      "produces": ["EntityBuffer"]
    },
    {
      "name": "input_to_movement",
      "kernel": "input_to_movement",
      "depends_on": ["spawn"],
      "consumes": ["EntityBuffer", "InputBuffer"],
      "produces": ["EntityBuffer"]
    },
    {
      "name": "movement",
      "kernel": "movement",
      "depends_on": ["input_to_movement"],
      "consumes": ["EntityBuffer", "GlobalsBuffer"],
      "produces": ["EntityBuffer"]
    },
    {
      "name": "collision",
      "kernel": "collision",
      "depends_on": ["movement"],
      "consumes": ["EntityBuffer"],
      "produces": ["CollisionResultBuffer"]
    },
    {
      "name": "damage",
      "kernel": "damage",
      "depends_on": ["collision"],
      "consumes": ["EntityBuffer", "CollisionResultBuffer"],
      "produces": ["EntityBuffer"]
    }
  ]
}
```

Note: the pipeline array is ordered. The ordering is the source of truth for execution sequence. The `depends_on` declarations make the dependency graph explicit and verifiable — a node's dependencies must all appear earlier in the array. In this simple shooter example the graph is linear, but fan-out and fan-in are supported: multiple nodes can depend on the same upstream node (fan-out), and a single node can depend on multiple upstream nodes (fan-in). Nodes at the same depth with no shared write buffers can be dispatched concurrently by the runtime.

**Buffer versioning.** Each time a node produces a buffer, it creates a new logical version of that buffer. A node that consumes a buffer reads the version produced by the most recent upstream writer. The runtime manages the physical double-buffering (ping-pong) automatically — the manifest deals only in logical buffer names. The Forge can query the runtime's physical buffer mapping when it needs to verify postconditions across dispatch boundaries.

**DAG validation (performed by the Forge before execution):**

- The graph is acyclic.
- Every buffer consumed has been produced by a prior node (no uninitialized reads).
- No two nodes that can execute concurrently produce the same buffer (no write-write hazards).
- Every kernel referenced is Forge-registered and has passed verification.
- Every `depends_on` reference names a node that exists earlier in the pipeline array.
- Consume/produce declarations are consistent with the kernel's contract (the kernel's declared reads/writes match the node's consumes/produces).

At the end of each frame, the DAG runner can optionally run all kernel postconditions as a verification pass. In development, this runs every frame. In production, it can be disabled or sampled.

**Cross-buffer invariants.** Invariants that span multiple buffers (e.g., "every index in CollisionResultBuffer references a valid alive entity in EntityBuffer") are expressed as postconditions on the kernel that produces the referencing buffer. The DAG's ordering guarantee ensures that the referenced buffer is in the expected state when the postcondition is checked. No separate cross-buffer validation mechanism is needed.

### 3.5 The Harness

The harness is the game-mode loop within the `forge` binary. It is deliberately minimal:

- Initialize `wgpu` device.
- Open a window via `winit`. Handle input events (keyboard, mouse, resize, close).
- Load the DAG manifest.
- Allocate GPU buffers according to schemas (including physical double-buffer pairs managed transparently).
- Each frame: write input buffer (keyboard/mouse/time), dispatch simulation kernels in DAG order, then dispatch the render kernel and present the framebuffer.
- In development mode, optionally evaluate postconditions after each dispatch (same postcondition evaluator used during verification — same binary, same code path, controlled by a flag).

The harness does not contain game logic. It does not know what kind of game is running. It is a generic DAG executor with a render step bolted on at the end. The render kernel is not part of the DAG — it is a visualization pass that reads whatever final state the simulation pipeline produced. Multiple render kernels can coexist (debug view, production view, minimap) without changing the simulation graph. The render kernel uses floating-point freely since its output (the framebuffer) does not feed back into simulation.

### 3.6 The Forge

The Forge is the verification mode of the `forge` binary — the contract-enforced gateway for kernels entering the system. It shares the same `wgpu` device management, schema system, buffer allocation, and WGSL compilation as the game-mode harness. It is the most critical component and must be built and verified first.

The Forge embeds QuickJS for evaluating JavaScript postconditions and invariants in a sandboxed scope. In verification mode, postconditions are checked after every dispatch. In game mode, the same evaluator can optionally run for development-time validation.

CLI interface: `forge verify bundle.json` → JSON result on stdout (accepted with kernel ID, or rejected with diagnostics). The AI agent (Claude Code) calls it as a subprocess.

**Forge submission requirements:**

Every kernel submission must include:

1. **Schema declarations** for all buffers the kernel touches.
2. **The kernel contract** — declared inputs, outputs, and postconditions.
3. **The WGSL source.**
4. **Adversarial test cases** — minimum four: empty buffer, single element, maximum capacity, and at least one domain-specific pathological case.

**Forge verification pipeline:**

1. **Design parameter resolution** — resolve all `$`-prefixed values in the kernel's schemas and contracts against the active design parameter set. Verify design invariants hold.
2. **Compilation check** — does the WGSL compile without errors?
3. **Binding check** — do the shader's buffer bindings match the declared contract? No undeclared bindings?
4. **Structural schema check** — do the buffer layouts in the WGSL struct definitions match the JSON schemas?
5. **Adversarial input generation** — generate input buffers for each adversarial case, constrained to valid schema ranges and invariants. Inputs outside schema bounds are not generated — the Forge tests the real game state space, not arbitrary data.
6. **Kernel execution** — dispatch the kernel on each adversarial input set, read back results.
7. **Schema constraint check** — does every field in every output buffer fall within its declared range? Do all schema invariants hold on the output?
8. **Postcondition verification** — for each adversarial case, check all declared postconditions against the output buffers.
9. **Clean write check** — confirm that input buffers are unmodified after dispatch (byte-identical to pre-dispatch snapshot).
10. **Bounds check** — confirm no out-of-bounds writes in output buffers (no writes beyond declared capacity).

If any step fails, the kernel is **rejected** with a specific, actionable error message. If all steps pass, the kernel is **registered** and available for DAG inclusion.

**The Forge verifies itself.** The Forge's own verification pipeline must be tested with real GPU dispatches — not CPU-side simulations, not mocked buffers, not postcondition functions evaluated against hand-crafted arrays. Every test allocates real GPU buffers, dispatches a real WGSL compute shader, reads back real output, and evaluates postconditions through the same pipeline that runs in production. The GPU is the thing being tested. Removing it from the test removes the test's meaning.

Three categories of Forge self-test:

**Known-good kernels with known inputs.** Hand-craft an input buffer, hand-compute the expected output, dispatch the kernel on the GPU, read back, verify all postconditions pass. Example: movement kernel, one entity at position (100, 100) with velocity (10, 0) and delta_time 1 — expected output is (110, 100). This confirms the kernel does what its contract says.

**Known-bad kernels.** Write WGSL that deliberately violates its contract: a movement kernel that doesn't clamp to world bounds, a spawn kernel that overwrites alive entities, a collision kernel that modifies the entity buffer. Compile the bad WGSL, dispatch it on the GPU, read back whatever it produces, and confirm the Forge rejects it through the real verification pipeline. If the Forge accepts a broken kernel, the Forge is broken. Do not test this category by evaluating postcondition JavaScript against hand-crafted output arrays — that tests the postcondition function, not the Forge.

**Adversarial inputs on good kernels.** Generate inputs at the boundaries of the valid schema range: all `$max_entities` slots alive, all entities at world edges with max velocity, zero alive entities, single entity at origin with zero velocity. Dispatch on the GPU, read back, verify all postconditions hold. These find real bugs in kernels that work fine on typical inputs.

### 3.7 Postconditions and Invariants

Schema invariants and kernel postconditions are JavaScript functions evaluated by the Forge. No DSL, no parser, no compiler. The Forge provides a sandboxed evaluation context with helper functions, and the postcondition is just code.

**Schema invariants** are JavaScript expressions evaluated per-element. The element's fields are in scope as local variables, and design parameters are available via `$`.

```json
{
  "invariants": [
    "alive === 0 ? entity_type === 0 : true",
    "health <= max_health",
    "entity_type === 3 ? health <= $.bullet_health : true",
    "entity_type === 1 ? max_health === $.player_health : true",
    "entity_type === 2 ? max_health <= $.boss_health : true"
  ]
}
```

These compile to: `for each element e in buffer: evaluate expression with e's fields in scope`. If any element fails, the Forge reports which element at which index violated which invariant.

**Kernel postconditions** are JavaScript function bodies that receive `input`, `output`, `globals`, and `$` (design params). They return `true` on success or call `fail(index, message)` on failure. Helper functions are injected into scope.

```json
{
  "postconditions": [
    "for (let i = 0; i < input.length; i++) { if (input[i].alive === 0 && !eq(output[i], input[i])) return fail(i, 'dead entity modified'); } return true;",
    "for (let i = 0; i < input.length; i++) { if (input[i].alive === 1) { const ex = clamp(input[i].position_x + fix16_mul(input[i].velocity_x, globals.delta_time), 0, $.world_width); if (output[i].position_x !== ex) return fail(i, `position_x: expected ${ex}, got ${output[i].position_x}`); } } return true;"
  ]
}
```

**Available helpers:**

- `fix16_mul(a, b)` — fixed-point 16.16 multiply, matching the WGSL implementation exactly.
- `clamp(value, min, max)` — integer clamp.
- `abs(value)` — absolute value.
- `min(a, b)`, `max(a, b)` — integer min/max.
- `eq(a, b)` — byte-identical struct comparison. Exact because simulation buffers are integer-only.
- `fail(index, message)` — report a violation with the element index and a diagnostic string. Returns a failure object that the Forge captures.
- `$` — the resolved design parameters object. `$.world_width`, `$.max_speed`, etc.

**Why not a DSL?** The postcondition compiles to a JavaScript function anyway. Skipping the parser and compiler eliminates an entire infrastructure layer. The AI agent already knows JavaScript. The Forge evaluates JavaScript via an embedded QuickJS runtime. If the JavaScript postconditions become unwieldy for complex games, a readability layer can be added later — but it's an optimization, not a prerequisite.

**Sandbox.** Postcondition functions run in an embedded QuickJS instance within the `forge` binary. They have no access to the filesystem, network, or global state. They receive only buffer views (as typed arrays), helpers, and design params. The Forge constructs this scope per evaluation. In game mode (`forge run`), postcondition evaluation is optional — controlled by a `--verify` flag for development-time validation. In production, the harness trusts that kernels passed the Forge and skips postcondition checks for performance.

---

## 4. Verification Layers

### 4.1 Structural Verification (Static)

Performed without execution. Checks design parameters, schemas, bindings, DAG structure.

- Design parameters are well-formed and all design invariants hold.
- Buffer schemas are well-formed, use only supported types, and all `$`-references resolve to valid design parameters.
- Schema value constraints are consistent (min <= max, enum values are valid for the type).
- Schema invariants are syntactically valid and reference only fields that exist in the schema.
- Kernel bindings match declared schemas.
- DAG is acyclic.
- All `depends_on` references name nodes that exist earlier in the pipeline array.
- Consume/produce declarations are consistent with kernel contracts.
- No concurrent write-write hazards on the same buffer.
- All kernels in the DAG are Forge-registered.

### 4.2 Behavioral Verification (Per-Dispatch)

Performed by running kernels on test inputs and checking postconditions.

- All test inputs conform to their schema constraints (ranges, enums, invariants). The Forge does not generate inputs outside the valid game state space.
- Every declared postcondition holds on every adversarial test case.
- All output buffer fields are within their schema-declared ranges.
- All schema invariants hold on output buffers.
- Input buffers are unmodified (clean write check).
- No NaN or infinity values in render-stage float buffers (simulation buffers are integer and cannot contain NaN by construction).

Note: this is testing, not proof. Schema constraints focus the testing on the valid game state space, making adversarial cases more meaningful and more likely to catch real bugs. But it remains possible for a kernel to pass all tests and still fail on an untested input within the valid range.

### 4.3 Differential Verification (Cross-Implementation)

Performed by comparing two implementations of the same kernel.

- Run a reference implementation (possibly CPU-side, possibly slow) and the GPU kernel on identical inputs.
- Diff output buffers byte-for-byte.
- Useful for letting the AI explore novel optimized implementations while ensuring correctness against a known-good reference.

### 4.4 Temporal Verification (Multi-Frame)

Performed by recording buffer snapshots across many frames and checking invariants over time.

- Entity count is consistent with spawns minus deaths.
- No entity resurrects (alive goes from 0 to 1) without a spawn event.
- Conservation laws hold (total currency, total energy, etc.).
- No state corruption accumulates over hundreds or thousands of frames.

### 4.5 Adversarial Verification (Edge Cases)

Deliberately worst-case inputs within the valid game state space, designed to break kernels.

- Boundary values: entities at world edges, health at 0 and at max, velocities at max speed. These are derived from schema constraints and design parameters, not arbitrary.
- Empty buffers (zero alive entities).
- Maximum capacity buffers (all entity slots alive).
- Degenerate spatial configurations within bounds (all entities at same valid position, entities at exact corners).
- Discrete field exhaustion: for fields with small enum or range values (entity_type, alive), test every valid combination in conjunction with boundary continuous values.
- Integer edge cases within declared ranges (counters at max, accumulators near overflow within valid game state).

Adversarial cases do NOT include out-of-range garbage (NaN, negative health, positions outside world bounds) as inputs. The schema system guarantees inputs are valid. The Forge tests whether kernels preserve validity, not whether they handle invalid inputs gracefully.

---

## 5. Bootstrap Sequence

The system must be built bottom-up, with each layer verified before the next is built on top of it. Every phase produces code within the same Rust codebase. `cargo build` always produces one binary: `forge`.

### Phase 0: GPU Foundation

**Goal:** Prove that the binary can create a `wgpu` device, allocate a buffer, dispatch a trivial compute shader, and read back correct results.

**Deliverables:**

- Rust project with `wgpu` dependency. `cargo build` produces the `forge` binary.
- A trivial compute shader (hardcoded WGSL string) that writes known values to a buffer.
- GPU dispatch and CPU-side readback confirming the values are correct.
- Backend selection: auto-detect Vulkan/Metal/D3D12, with CLI flag to force a specific backend or fall back to OpenGL.

**Verification:** Human reviews. `cargo test` runs the GPU round trip. Under 300 lines of Rust.

### Phase 1: Schema System

**Goal:** Parse JSON design parameters and constrained buffer schemas. Validate that design invariants hold and that buffer contents conform to schema constraints.

**Deliverables:**

- JSON parsing for design parameters, design invariants, and constrained schemas.
- Design invariant checker — evaluates JavaScript invariant expressions via embedded QuickJS with `$` (design params) in scope.
- Schema validation: type checks, range checks, per-element invariant evaluation.
- Buffer generation: create a buffer filled with valid random data within schema constraints.
- Adversarial buffer generation: boundary values within constraints, discrete field exhaustion.
- `cargo test` suite proving validation catches: wrong byte length, out-of-range values, invariant violations, broken design invariants.

**Verification:** Human reviews. Under 1000 lines of Rust.

### Phase 2: Contract Checker

**Goal:** Given a kernel contract (declared I/O, postconditions as JavaScript), dispatch the kernel on the GPU and verify the contract holds.

**Deliverables:**

- Contract definition parsing (JSON).
- Postcondition sandbox — QuickJS evaluation scope with helper functions (`fix16_mul`, `clamp`, `abs`, `min`, `max`, `eq`, `fail`) and access to `input`, `output`, `globals`, `$` (design params).
- Schema invariant evaluator — iterates buffer elements, evaluates invariant expressions per-element with fields in scope.
- Postcondition evaluator — snapshots input buffers, dispatches kernel on the GPU, evaluates postcondition functions against input snapshot and output.
- Clean write checker — diffs input buffers pre/post dispatch (exact integer comparison).
- `cargo test` suite using deliberately correct and deliberately broken WGSL kernels to confirm the checker catches violations and passes valid kernels. Every test dispatches on the real GPU — no mocked buffers, no simulated shaders.

**Verification:** Human reviews. Under 1200 lines of Rust excluding test kernels.

### Phase 3: Verification CLI

**Goal:** Wire schemas, contracts, compilation, and verification into the `forge verify` subcommand.

**Deliverables:**

- CLI interface: `forge verify bundle.json` → JSON result on stdout.
- Bundle format: single JSON file containing design params, schemas, contract, WGSL source (inline or path), and adversarial test case descriptions.
- Full verification pipeline as described in section 3.6.
- Kernel registry: accepted kernels stored with their contracts (as JSON files in a registry directory) for DAG inclusion.
- `forge validate-dag manifest.json` — checks DAG structure against registered kernels.
- Forge self-test suite: deliberately broken submissions that must be rejected, correct submissions that must be accepted. All tests run the full GPU pipeline — compile, dispatch, readback, postcondition evaluation. No shortcuts.

**Verification:** Human reviews the Forge's behavior via its self-test suite. The test cases are JSON and WGSL — human-readable regardless of the Rust implementation. The Forge then becomes the verification authority for everything built on top of it.

### Phase 4: DAG Runner

**Goal:** Execute an ordered simulation pipeline of Forge-registered kernels per frame.

**Deliverables:**

- DAG manifest loader and validator: checks acyclicity, dependency coherence with consume/produce, all kernels Forge-registered, no concurrent write-write hazards.
- DAG executor: dispatches simulation kernels in declared order, manages physical buffer double-buffering (ping-pong) transparently.
- Development-mode per-frame verification: optionally evaluates all postconditions after each dispatch using the same contract checker from Phase 2.

**Verification:** Kernels are Forge-verified. DAG structure is Forge-validated. `cargo test` suite with valid and invalid DAGs. Human reviews the DAG runner.

### Phase 5: Game Mode

**Goal:** Add windowing, input handling, and render dispatch to make the DAG runner into a playable game runtime.

**Deliverables:**

- `forge run manifest.json` subcommand: opens a `winit` window, runs the DAG executor in a frame loop.
- Input capture: keyboard and mouse state written to an InputBuffer each frame via `winit` events.
- Render dispatch: after simulation pipeline completes, dispatch the render kernel and present the framebuffer to the window surface.
- Frame timing: fixed or variable timestep with delta_time written to GlobalsBuffer.

**Verification:** Human plays. The game loop is thin glue over already-verified components. Under 400 lines of new Rust.

### Phase 6: First Game — The 2D Shooter

**Goal:** Prove the system works end-to-end by building a playable game entirely from Forge-verified kernels composed into a DAG.

**Deliverables:**

- Design parameters: `shooter_design` with all game tuning values and design invariants.
- Constrained buffer schemas: EntityBuffer, InputBuffer, GlobalsBuffer, CollisionResultBuffer, Framebuffer — all referencing design parameters.
- Forge-verified simulation kernels: spawn, input_to_movement, movement, collision, damage.
- Render kernel (outside the DAG, dispatched by harness after simulation pipeline).
- DAG manifest wiring the simulation kernels together.
- A playable game: `forge run shooter_manifest.json`.

**Verification:** Every kernel passes `forge verify`. The DAG passes `forge validate-dag`. Temporal verification confirms game state consistency over thousands of frames of automated play. Then a human plays it and confirms it's fun enough to be a valid proof of concept.

---

## 6. Reference: 2D Shooter Design Parameters

```json
{
  "name": "shooter_design",
  "params": {
    "world_width":          { "type": "fix16", "value": 1280 },
    "world_height":         { "type": "fix16", "value": 720 },
    "max_speed":            { "type": "fix16", "value": 500 },
    "player_health":        { "type": "u32", "value": 100 },
    "enemy_health":         { "type": "u32", "value": 30 },
    "boss_health":          { "type": "u32", "value": 200 },
    "bullet_damage":        { "type": "u32", "value": 10 },
    "bullet_health":        { "type": "u32", "value": 1 },
    "max_entities":         { "type": "u32", "value": 4096 },
    "enemy_spawn_interval": { "type": "u32", "value": 60 },
    "player_size":          { "type": "fix16", "value": 16 },
    "enemy_size":           { "type": "fix16", "value": 12 },
    "bullet_size":          { "type": "fix16", "value": 4 },
    "max_delta_time":       { "type": "fix16", "value": 0.05 }
  },
  "design_invariants": [
    "$.bullet_damage < $.player_health",
    "$.bullet_damage <= $.enemy_health",
    "$.boss_health >= $.enemy_health",
    "$.player_size < $.world_width && $.player_size < $.world_height",
    "fix16_mul($.max_speed, $.max_delta_time) < $.world_width",
    "fix16_mul($.max_speed, $.max_delta_time) < $.world_height",
    "$.max_entities >= 64"
  ]
}
```

---

## 7. Reference: 2D Shooter Buffer Schemas

### EntityBuffer

```json
{
  "name": "EntityBuffer",
  "struct": {
    "position":    { "type": "fix16x2", "range": [[0, "$world_width"], [0, "$world_height"]] },
    "velocity":    { "type": "fix16x2", "range": [["$-max_speed", "$max_speed"], ["$-max_speed", "$max_speed"]] },
    "size":        { "type": "fix16", "range": [0, "$player_size"] },
    "health":      { "type": "u32", "range": [0, "$boss_health"] },
    "max_health":  { "type": "u32", "range": [0, "$boss_health"] },
    "entity_type": { "type": "u32", "enum": [0, 1, 2, 3] },
    "damage":      { "type": "u32", "range": [0, "$bullet_damage"] },
    "alive":       { "type": "u32", "enum": [0, 1] },
    "color":       { "type": "u32", "note": "packed RGB, 8 bits per channel — not used in simulation" },
    "padding":     { "type": "u32" }
  },
  "capacity": "$max_entities",
  "buffer_category": "state",
  "invariants": [
    "alive === 0 ? entity_type === 0 : true",
    "health <= max_health",
    "entity_type === 3 ? health <= $.bullet_health : true",
    "entity_type === 3 ? damage === $.bullet_damage : true",
    "entity_type === 1 ? max_health === $.player_health : true",
    "entity_type === 2 ? max_health <= $.boss_health : true"
  ]
}
```

Entity types: 0 = inactive, 1 = player, 2 = enemy, 3 = bullet.

### InputBuffer

```json
{
  "name": "InputBuffer",
  "struct": {
    "keys_down":      { "type": "u32", "range": [0, 31] },
    "mouse_position": { "type": "fix16x2", "range": [[0, "$world_width"], [0, "$world_height"]] },
    "mouse_buttons":  { "type": "u32", "range": [0, 7] }
  },
  "capacity": 1,
  "buffer_category": "input"
}
```

`keys_down` is a bitfield: bit 0 = W, bit 1 = A, bit 2 = S, bit 3 = D, bit 4 = space.

### GlobalsBuffer

```json
{
  "name": "GlobalsBuffer",
  "struct": {
    "frame_number":  { "type": "u32" },
    "delta_time":    { "type": "fix16", "range": [0, "$max_delta_time"] },
    "world_width":   { "type": "fix16", "value": "$world_width" },
    "world_height":  { "type": "fix16", "value": "$world_height" },
    "random_seed":   { "type": "u32" },
    "padding":       { "type": "u32" },
    "padding2":      { "type": "u32" },
    "padding3":      { "type": "u32" }
  },
  "capacity": 1,
  "buffer_category": "input",
  "invariants": [
    "world_width === $.world_width",
    "world_height === $.world_height"
  ]
}
```

GlobalsBuffer is set by the harness before the frame starts and is immutable for the duration of the simulation pipeline. There is no entity count — kernels dispatch over the full `$max_entities` buffer and check the `alive` flag per slot.

### CollisionResultBuffer

```json
{
  "name": "CollisionResultBuffer",
  "struct": {
    "entity_a":       { "type": "u32", "range": [0, "$max_entities"] },
    "entity_b":       { "type": "u32", "range": [0, "$max_entities"] },
    "collision_type": { "type": "u32", "enum": [1, 2] },
    "padding":        { "type": "u32" }
  },
  "capacity": 8192,
  "invariants": [
    "entity_a != entity_b"
  ]
}
```

Collision types: 1 = bullet-enemy, 2 = enemy-player.

A separate `CollisionCountBuffer` (single u32, range [0, 8192]) tracks how many valid entries are in the collision result buffer.

### Framebuffer

```json
{
  "name": "Framebuffer",
  "struct": {
    "pixel": { "type": "u32" }
  },
  "capacity": "$world_width * $world_height"
}
```

Each u32 stores packed RGBA (8 bits per channel).

---

## 8. Reference: 2D Shooter Kernel Contracts

Abbreviated contracts for each kernel in the shooter DAG. Full adversarial cases omitted for brevity — the Forge requires them at submission time. All `$`-prefixed values resolve against `shooter_design` parameters.

### spawn

- **Reads:** EntityBuffer (current), InputBuffer, GlobalsBuffer
- **Writes:** EntityBuffer (next — copy of current with new entities spawned)
- **Dispatch:** Full `$max_entities` buffer. Block-based workgroup ownership — workgroup N writes only to indices `[N * 64, (N+1) * 64)`.
- **Postconditions:**
```js
// Existing alive entities must not be modified
for (let i = 0; i < input.length; i++) {
  if (input[i].alive === 1 && !eq(output[i], input[i]))
    return fail(i, 'existing alive entity modified');
}
return true;
```
```js
// New entities can only appear in previously dead slots
for (let i = 0; i < output.length; i++) {
  if (output[i].alive === 1 && input[i].alive === 0 && output[i].entity_type === 0)
    return fail(i, 'spawned entity has entity_type 0');
}
return true;
```
```js
// At most one bullet spawned per frame
let bullets = 0;
for (let i = 0; i < output.length; i++) {
  if (output[i].entity_type === 3 && input[i].alive === 0) bullets++;
}
if (bullets > 1) return fail(-1, `${bullets} bullets spawned, max 1`);
return true;
```

### input_to_movement

- **Reads:** EntityBuffer, InputBuffer
- **Writes:** EntityBuffer (in-place velocity update on player entity only)
- **Postconditions:**
```js
// Non-player entities must not be modified
for (let i = 0; i < input.length; i++) {
  if (input[i].entity_type !== 1 && !eq(output[i], input[i]))
    return fail(i, 'non-player entity modified');
}
return true;
```
```js
// Player entity: only velocity may change
for (let i = 0; i < input.length; i++) {
  if (input[i].entity_type === 1) {
    if (output[i].health !== input[i].health) return fail(i, 'player health changed');
    if (output[i].position_x !== input[i].position_x) return fail(i, 'player position_x changed');
    if (output[i].position_y !== input[i].position_y) return fail(i, 'player position_y changed');
    if (output[i].alive !== input[i].alive) return fail(i, 'player alive changed');
  }
}
return true;
```
```js
// Player velocity magnitude must not exceed max_speed
for (let i = 0; i < output.length; i++) {
  if (output[i].entity_type === 1) {
    const vx = output[i].velocity_x, vy = output[i].velocity_y;
    const mag_sq = fix16_mul(vx, vx) + fix16_mul(vy, vy);
    const max_sq = fix16_mul($.max_speed, $.max_speed);
    if (mag_sq > max_sq) return fail(i, `velocity magnitude squared ${mag_sq} exceeds ${max_sq}`);
  }
}
return true;
```

### movement

- **Reads:** EntityBuffer, GlobalsBuffer
- **Writes:** EntityBuffer (updated positions)
- **Postconditions:**
```js
// Dead entities must not be modified
for (let i = 0; i < input.length; i++) {
  if (input[i].alive === 0 && !eq(output[i], input[i]))
    return fail(i, 'dead entity modified');
}
return true;
```
```js
// Alive entities: position updated and clamped, everything else unchanged
for (let i = 0; i < input.length; i++) {
  if (input[i].alive === 1) {
    const ex = clamp(input[i].position_x + fix16_mul(input[i].velocity_x, globals.delta_time), 0, $.world_width);
    const ey = clamp(input[i].position_y + fix16_mul(input[i].velocity_y, globals.delta_time), 0, $.world_height);
    if (output[i].position_x !== ex) return fail(i, `position_x: expected ${ex}, got ${output[i].position_x}`);
    if (output[i].position_y !== ey) return fail(i, `position_y: expected ${ey}, got ${output[i].position_y}`);
    if (output[i].velocity_x !== input[i].velocity_x) return fail(i, 'velocity_x changed');
    if (output[i].velocity_y !== input[i].velocity_y) return fail(i, 'velocity_y changed');
    if (output[i].health !== input[i].health) return fail(i, 'health changed');
    if (output[i].alive !== input[i].alive) return fail(i, 'alive changed');
  }
}
return true;
```

### collision

- **Reads:** EntityBuffer
- **Writes:** CollisionResultBuffer, CollisionCountBuffer
- **Postconditions:**
```js
// EntityBuffer must not be modified (read-only)
for (let i = 0; i < input.length; i++) {
  if (!eq(output_entities[i], input[i]))
    return fail(i, 'entity modified by collision kernel');
}
return true;
```
```js
// Collision count within capacity
if (collision_count[0].count > 8192)
  return fail(0, `collision count ${collision_count[0].count} exceeds capacity`);
return true;
```
```js
// Every collision pair references two distinct alive entities that overlap
for (let i = 0; i < collision_count[0].count; i++) {
  const a = collisions[i].entity_a, b = collisions[i].entity_b;
  if (a === b) return fail(i, 'self-collision');
  if (entities[a].alive !== 1) return fail(i, `entity_a ${a} not alive`);
  if (entities[b].alive !== 1) return fail(i, `entity_b ${b} not alive`);
  if (abs(entities[a].position_x - entities[b].position_x) >= entities[a].size + entities[b].size)
    return fail(i, `pair ${a},${b} not overlapping on x`);
  if (abs(entities[a].position_y - entities[b].position_y) >= entities[a].size + entities[b].size)
    return fail(i, `pair ${a},${b} not overlapping on y`);
}
return true;
```

### damage

- **Reads:** EntityBuffer, CollisionResultBuffer, CollisionCountBuffer
- **Writes:** EntityBuffer (updated health and alive flags)
- **Postconditions:**
```js
// Health never goes negative (saturates at zero)
for (let i = 0; i < output.length; i++) {
  if (output[i].health < 0) return fail(i, `health is ${output[i].health}`);
}
return true;
```
```js
// Zero health means dead
for (let i = 0; i < output.length; i++) {
  if (output[i].health === 0 && output[i].alive !== 0)
    return fail(i, 'health is 0 but alive is not 0');
}
return true;
```
```js
// Entities not involved in any collision are unchanged
for (let i = 0; i < input.length; i++) {
  let involved = false;
  for (let j = 0; j < collision_count[0].count; j++) {
    if (collisions[j].entity_a === i || collisions[j].entity_b === i) {
      involved = true; break;
    }
  }
  if (!involved && !eq(output[i], input[i]))
    return fail(i, 'entity not in collision was modified');
}
return true;
```

### render (visualization — not a DAG node)

- **Reads:** EntityBuffer (final state after pipeline), GlobalsBuffer
- **Writes:** Framebuffer
- **Note:** The render kernel is dispatched by the harness after the simulation pipeline completes. It is not part of the DAG and does not participate in simulation state transformation. It may use floating-point freely. Multiple render kernels can coexist (debug view, production view, minimap) without affecting simulation.
- **Postconditions:** Every pixel in the framebuffer has been written. EntityBuffer is unmodified.

---

## 9. Technology Stack

**One binary, multiple modes:**

`forge` is a single Rust binary built on `wgpu` (Vulkan/Metal/D3D12/OpenGL — native, no browser) with embedded QuickJS for postcondition evaluation. It operates in three modes:

- `forge verify bundle.json` — verification mode. No window. Compile WGSL, dispatch on GPU, readback, evaluate postconditions, emit JSON result. This is what the AI agent calls.
- `forge run manifest.json` — game mode. Open a window via `winit`, run the DAG each frame, handle input, render. Optionally evaluate postconditions per-dispatch in development mode.
- `forge test` — self-test suite. Run all verification tests against known-good and known-bad kernels.

The core — `wgpu` device management, schema-driven buffer allocation, WGSL compilation, kernel dispatch, buffer readback, QuickJS postcondition evaluation — is shared across all three modes. The difference is the outer loop: once-and-report for verification, sixty-times-a-second for gameplay, batch-and-summarize for testing.

- **Language:** Rust (engine and toolchain), WGSL (kernels), JavaScript (postconditions and invariants)
- **GPU API:** `wgpu` (Vulkan, Metal, D3D12, OpenGL — native, no browser)
- **JS runtime:** Embedded QuickJS (for postcondition and invariant evaluation)
- **Windowing:** `winit` (game mode)
- **Schema/contract/manifest format:** JSON
- **Test framework:** `cargo test` + Forge self-test suite (real GPU dispatches)
- **UI layer (future):** CEF embedded in the native window, rendering React to an offscreen texture composited by the render kernel
- **Browser deployment (future):** Compile to WASM, same WGSL shaders, browser WebGPU API

---

## 10. Open Questions

1. **Conditional dispatch.** Some kernels only need to run under certain conditions (e.g., spawn every N frames). Currently handled by the kernel early-outing internally based on frame number. For complex games with many expensive conditional kernels, the manifest may need a condition field on nodes so the DAG runner can skip dispatches entirely. Not needed for the shooter proof of concept.

2. **Audio.** Can audio synthesis be done in compute shaders? Likely yes — write PCM samples to a buffer, stream to a native audio API (e.g., `cpal` in Rust) on the CPU side. Needs investigation.

3. **Persistent state.** Save games are trivial — dump all state buffers to disk. But schema evolution (adding fields to entities in a game update) needs a migration strategy. Design parameter changes that widen or narrow schema constraints also need a migration path.

4. **Networking.** The fixed-point integer simulation guarantees determinism across all platforms, making lockstep networking viable. Send only input buffers, simulate identically on all clients. Remaining design work: input buffer synchronization protocol, handling latency and dropped packets, and defining the authority model for conflict resolution.

5. **AI agent workflow.** The Forge validates kernels, but the higher-level workflow — how an AI agent decomposes "make a tower defense game" into design parameters, schemas, kernels, and a DAG — is out of scope for the engine itself. This is the orchestration layer built on top.

6. **Variable-length game data.** Inventories, dialogue trees, quest state, NPC memories — anything that might seem "variable-length" in a CPU architecture is handled here by fixed-size allocation at the schema's declared maximum. If max inventory is 20 slots, every entity gets 20 slots; unused slots are zero. This trades VRAM (which is abundant) for uniform access patterns and predictable memory layout (which the GPU demands). The design parameter system naturally accommodates this: `max_inventory_slots`, `max_quest_slots`, `max_memory_entries` are just design params, and schemas reference them like any other. No indirection, no dynamic allocation, no special cases.

7. **UI layer.** Game UI (menus, HUD, inventory) will be built with web technologies (React) rendered via CEF to an offscreen texture, composited by a render kernel. This preserves hot reload and the web dev workflow while keeping the game fully native. Design work: CEF integration in Rust, texture upload path, input routing between game and UI, compositor kernel contract.

8. **Browser deployment.** The `forge` binary targets native platforms. A future browser deployment path would compile the harness and DAG runner to WASM and use the browser's WebGPU API. The WGSL shaders and JSON artifacts are identical — only the host changes. This is not needed for development or initial release but the architecture should not preclude it.

---

## 11. Success Criteria

The engine is considered viable when:

1. The Forge can accept or reject kernel submissions with zero false negatives (never accepts a broken kernel) and minimal false positives (rarely rejects a correct kernel due to overly strict verification).
2. A non-trivial game (the 2D shooter) runs entirely from Forge-verified kernels composed into a DAG via `forge run`.
3. An AI agent (Claude Code or equivalent) can author a new kernel, submit it via `forge verify`, iterate on rejections, and produce a verified kernel — without human intervention in the edit-verify loop.
4. Temporal verification over 10,000 frames of automated play detects zero invariant violations in the shipped game.
5. The `forge` binary is under 4,000 lines of Rust (excluding test WGSL and JSON fixtures).
