/// <reference types="vite/client" />
// ForgeEngine — Contract Checker Self-Test (browser)
import { initGPU } from "../harness";
import { resolveParams } from "../schema/params";
import { resolveSchema, writeElement } from "../schema/schema";
import { toFix16 } from "../schema/params";
import type { DesignParamsJSON, BufferSchemaJSON, ResolvedParams, ResolvedSchema } from "../schema/types";
import type { KernelContractJSON } from "./types";
import { resolveContract } from "./sandbox";
import { checkContractGPU } from "./checker";

import testIdentitySrc from "../shaders/test_identity.wgsl?raw";
import testMovementSrc from "../shaders/test_movement.wgsl?raw";
import testMovementNoClampSrc from "../shaders/test_movement_no_clamp.wgsl?raw";
import testModifyDeadSrc from "../shaders/test_modify_dead.wgsl?raw";
import testDirtyInputSrc from "../shaders/test_dirty_input.wgsl?raw";
import testBreakInvariantSrc from "../shaders/test_break_invariant.wgsl?raw";

// ── Design Parameters (small capacity for testing) ──

const DESIGN_PARAMS: DesignParamsJSON = {
  name: "test_design",
  params: {
    world_width: { type: "fix16", value: 1280 },
    world_height: { type: "fix16", value: 720 },
    max_speed: { type: "fix16", value: 500 },
    player_health: { type: "u32", value: 100 },
    enemy_health: { type: "u32", value: 30 },
    boss_health: { type: "u32", value: 200 },
    bullet_damage: { type: "u32", value: 10 },
    bullet_health: { type: "u32", value: 1 },
    max_entities: { type: "u32", value: 8 },
    player_size: { type: "fix16", value: 16 },
    enemy_size: { type: "fix16", value: 12 },
    bullet_size: { type: "fix16", value: 4 },
    max_delta_time: { type: "fix16", value: 0.05 },
  },
  design_invariants: [
    "$.bullet_damage < $.player_health",
    "$.bullet_damage <= $.enemy_health",
    "$.boss_health >= $.enemy_health",
    "$.max_entities >= 64 || $.max_entities >= 1",
  ],
};

const ENTITY_SCHEMA_JSON: BufferSchemaJSON = {
  name: "EntityBuffer",
  struct: {
    position: { type: "fix16x2", range: [[0, "$world_width"], [0, "$world_height"]] },
    velocity: { type: "fix16x2", range: [["$-max_speed", "$max_speed"], ["$-max_speed", "$max_speed"]] },
    size: { type: "fix16", range: [0, "$player_size"] },
    health: { type: "u32", range: [0, "$boss_health"] },
    max_health: { type: "u32", range: [0, "$boss_health"] },
    entity_type: { type: "u32", enum: [0, 1, 2, 3] },
    damage: { type: "u32", range: [0, "$bullet_damage"] },
    alive: { type: "u32", enum: [0, 1] },
  },
  capacity: "$max_entities",
  buffer_category: "state",
  invariants: [
    "alive === 0 ? entity_type === 0 : true",
    "health <= max_health",
  ],
};

const GLOBALS_SCHEMA_JSON: BufferSchemaJSON = {
  name: "GlobalsBuffer",
  struct: {
    delta_time: { type: "fix16", range: [0, "$max_delta_time"] },
    world_width: { type: "fix16" },
    world_height: { type: "fix16" },
    frame_count: { type: "u32" },
  },
  capacity: 1,
  buffer_category: "input",
};

// ── Movement kernel contract ──

const MOVEMENT_CONTRACT: KernelContractJSON = {
  name: "movement",
  description: "Updates entity positions based on velocity and delta time. Clamps to world bounds.",
  inputs: [
    { binding: 0, schema: "EntityBuffer", access: "read" },
    { binding: 1, schema: "GlobalsBuffer", access: "read" },
  ],
  outputs: [
    { binding: 2, schema: "EntityBuffer", access: "write" },
  ],
  postconditions: [
    "for (let i = 0; i < input.length; i++) { if (input[i].alive === 0 && !eq(output[i], input[i])) return fail(i, 'dead entity modified'); } return true;",
    "for (let i = 0; i < input.length; i++) { if (input[i].alive === 1) { const ex = clamp(input[i].position_x + fix16_mul(input[i].velocity_x, globals.delta_time), 0, $.world_width); if (output[i].position_x !== ex) return fail(i, `position_x: expected ${ex}, got ${output[i].position_x}`); } } return true;",
    "for (let i = 0; i < input.length; i++) { if (input[i].alive === 1) { const ey = clamp(input[i].position_y + fix16_mul(input[i].velocity_y, globals.delta_time), 0, $.world_height); if (output[i].position_y !== ey) return fail(i, `position_y: expected ${ey}, got ${output[i].position_y}`); } } return true;",
    "for (let i = 0; i < input.length; i++) { if (input[i].alive === 1 && (output[i].velocity_x !== input[i].velocity_x || output[i].velocity_y !== input[i].velocity_y)) return fail(i, 'velocity changed'); } return true;",
    "for (let i = 0; i < input.length; i++) { if (input[i].alive === 1 && output[i].health !== input[i].health) return fail(i, 'health changed'); } return true;",
  ],
};

// ── Identity kernel contract ──

const IDENTITY_CONTRACT: KernelContractJSON = {
  name: "identity",
  description: "Copies input to output unchanged.",
  inputs: [
    { binding: 0, schema: "EntityBuffer", access: "read" },
  ],
  outputs: [
    { binding: 1, schema: "EntityBuffer", access: "write" },
  ],
  postconditions: [
    "for (let i = 0; i < input.length; i++) { if (!eq(output[i], input[i])) return fail(i, 'output differs from input'); } return true;",
  ],
};

// ── Helpers ──

interface TestResult {
  name: string;
  pass: boolean;
  detail: string;
}

function makeEntityBuffer(
  schema: ResolvedSchema,
  entities: Record<string, number>[]
): ArrayBuffer {
  const buf = new ArrayBuffer(schema.totalSize);
  for (let i = 0; i < entities.length && i < schema.capacity; i++) {
    writeElement(buf, schema, i, entities[i]);
  }
  return buf;
}

function makeGlobalsBuffer(
  schema: ResolvedSchema,
  globals: Record<string, number>
): ArrayBuffer {
  const buf = new ArrayBuffer(schema.totalSize);
  writeElement(buf, schema, 0, globals);
  return buf;
}

function summarizeResult(
  result: Awaited<ReturnType<typeof checkContractGPU>>
): string {
  const parts: string[] = [];
  if (!result.cleanWrite.pass) {
    parts.push(`cleanWrite: ${result.cleanWrite.violations.join("; ")}`);
  }
  for (const inv of result.invariants) {
    if (!inv.pass) {
      parts.push(`invariant(${inv.schema}): ${inv.errors.slice(0, 3).join("; ")}`);
    }
  }
  for (let i = 0; i < result.postconditions.length; i++) {
    const pc = result.postconditions[i];
    if (!pc.pass) {
      parts.push(`postcondition[${i}]: ${pc.errors.slice(0, 3).join("; ")}`);
    }
  }
  return parts.length > 0 ? parts.join(" | ") : "all checks passed";
}

// ── Test Cases ──

async function test1_identity(
  device: GPUDevice,
  params: ResolvedParams,
  entitySchema: ResolvedSchema,
  globalsSchema: ResolvedSchema,
  schemas: Map<string, ResolvedSchema>
): Promise<TestResult> {
  const contract = resolveContract(IDENTITY_CONTRACT, schemas);

  const entityBuf = makeEntityBuffer(entitySchema, [
    {
      position_x: toFix16(100), position_y: toFix16(200),
      velocity_x: toFix16(10), velocity_y: toFix16(-5),
      size: toFix16(8), health: 50, max_health: 100,
      entity_type: 1, damage: 5, alive: 1,
    },
    {
      position_x: 0, position_y: 0,
      velocity_x: 0, velocity_y: 0,
      size: 0, health: 0, max_health: 0,
      entity_type: 0, damage: 0, alive: 0,
    },
  ]);

  const result = await checkContractGPU(
    device, contract, testIdentitySrc, [entityBuf], params
  );

  return {
    name: "Identity Kernel (known-good)",
    pass: result.pass,
    detail: summarizeResult(result),
  };
}

async function test2_movement(
  device: GPUDevice,
  params: ResolvedParams,
  entitySchema: ResolvedSchema,
  globalsSchema: ResolvedSchema,
  schemas: Map<string, ResolvedSchema>
): Promise<TestResult> {
  const contract = resolveContract(MOVEMENT_CONTRACT, schemas);

  // Entity at (100,100) velocity (10,0) dt=1 → expected (110,100)
  const entityBuf = makeEntityBuffer(entitySchema, [
    {
      position_x: toFix16(100), position_y: toFix16(100),
      velocity_x: toFix16(10), velocity_y: 0,
      size: toFix16(8), health: 50, max_health: 100,
      entity_type: 1, damage: 5, alive: 1,
    },
    // Dead entity — should be unchanged
    {
      position_x: toFix16(50), position_y: toFix16(50),
      velocity_x: toFix16(5), velocity_y: toFix16(5),
      size: 0, health: 0, max_health: 0,
      entity_type: 0, damage: 0, alive: 0,
    },
  ]);

  const globalsBuf = makeGlobalsBuffer(globalsSchema, {
    delta_time: toFix16(1),
    world_width: params.raw.world_width,
    world_height: params.raw.world_height,
    frame_count: 1,
  });

  const result = await checkContractGPU(
    device, contract, testMovementSrc, [entityBuf, globalsBuf], params
  );

  return {
    name: "Movement Kernel (known-good)",
    pass: result.pass,
    detail: summarizeResult(result),
  };
}

async function test3_movement_adversarial(
  device: GPUDevice,
  params: ResolvedParams,
  entitySchema: ResolvedSchema,
  globalsSchema: ResolvedSchema,
  schemas: Map<string, ResolvedSchema>
): Promise<TestResult> {
  const contract = resolveContract(MOVEMENT_CONTRACT, schemas);

  // Entity at world edge with max velocity — should clamp
  const entityBuf = makeEntityBuffer(entitySchema, [
    {
      position_x: params.raw.world_width,
      position_y: params.raw.world_height,
      velocity_x: params.raw.max_speed,
      velocity_y: params.raw.max_speed,
      size: toFix16(8), health: 50, max_health: 100,
      entity_type: 1, damage: 5, alive: 1,
    },
    // Entity at origin with negative max velocity — should clamp to 0
    {
      position_x: 0, position_y: 0,
      velocity_x: -params.raw.max_speed,
      velocity_y: -params.raw.max_speed,
      size: toFix16(4), health: 30, max_health: 30,
      entity_type: 2, damage: 0, alive: 1,
    },
  ]);

  const globalsBuf = makeGlobalsBuffer(globalsSchema, {
    delta_time: params.raw.max_delta_time,
    world_width: params.raw.world_width,
    world_height: params.raw.world_height,
    frame_count: 2,
  });

  const result = await checkContractGPU(
    device, contract, testMovementSrc, [entityBuf, globalsBuf], params
  );

  return {
    name: "Movement Adversarial (known-good, edge clamp)",
    pass: result.pass,
    detail: summarizeResult(result),
  };
}

async function test4_no_clamp(
  device: GPUDevice,
  params: ResolvedParams,
  entitySchema: ResolvedSchema,
  globalsSchema: ResolvedSchema,
  schemas: Map<string, ResolvedSchema>
): Promise<TestResult> {
  const contract = resolveContract(MOVEMENT_CONTRACT, schemas);

  // Entity near world edge — without clamping, position will exceed bounds
  const entityBuf = makeEntityBuffer(entitySchema, [
    {
      position_x: toFix16(1270), position_y: toFix16(710),
      velocity_x: toFix16(20), velocity_y: toFix16(20),
      size: toFix16(8), health: 50, max_health: 100,
      entity_type: 1, damage: 5, alive: 1,
    },
  ]);

  const globalsBuf = makeGlobalsBuffer(globalsSchema, {
    delta_time: toFix16(1),
    world_width: params.raw.world_width,
    world_height: params.raw.world_height,
    frame_count: 3,
  });

  const result = await checkContractGPU(
    device, contract, testMovementNoClampSrc, [entityBuf, globalsBuf], params
  );

  // Should FAIL — postcondition expects clamped values
  return {
    name: "No-Clamp Movement (known-bad)",
    pass: !result.pass,
    detail: result.pass
      ? "UNEXPECTED: checker accepted broken kernel"
      : `Correctly rejected: ${summarizeResult(result)}`,
  };
}

async function test5_modify_dead(
  device: GPUDevice,
  params: ResolvedParams,
  entitySchema: ResolvedSchema,
  globalsSchema: ResolvedSchema,
  schemas: Map<string, ResolvedSchema>
): Promise<TestResult> {
  const contract = resolveContract(MOVEMENT_CONTRACT, schemas);

  // Dead entity with nonzero velocity — kernel modifies it
  const entityBuf = makeEntityBuffer(entitySchema, [
    {
      position_x: toFix16(100), position_y: toFix16(100),
      velocity_x: toFix16(10), velocity_y: toFix16(10),
      size: 0, health: 0, max_health: 0,
      entity_type: 0, damage: 0, alive: 0,
    },
  ]);

  const globalsBuf = makeGlobalsBuffer(globalsSchema, {
    delta_time: toFix16(1),
    world_width: params.raw.world_width,
    world_height: params.raw.world_height,
    frame_count: 4,
  });

  const result = await checkContractGPU(
    device, contract, testModifyDeadSrc, [entityBuf, globalsBuf], params
  );

  return {
    name: "Modify Dead Entity (known-bad)",
    pass: !result.pass,
    detail: result.pass
      ? "UNEXPECTED: checker accepted kernel that modifies dead entities"
      : `Correctly rejected: ${summarizeResult(result)}`,
  };
}

async function test6_dirty_input(
  device: GPUDevice,
  params: ResolvedParams,
  entitySchema: ResolvedSchema,
  globalsSchema: ResolvedSchema,
  schemas: Map<string, ResolvedSchema>
): Promise<TestResult> {
  const contract = resolveContract(IDENTITY_CONTRACT, schemas);

  const entityBuf = makeEntityBuffer(entitySchema, [
    {
      position_x: toFix16(100), position_y: toFix16(200),
      velocity_x: 0, velocity_y: 0,
      size: toFix16(8), health: 50, max_health: 100,
      entity_type: 1, damage: 5, alive: 1,
    },
  ]);

  const result = await checkContractGPU(
    device, contract, testDirtyInputSrc, [entityBuf], params
  );

  // Should FAIL — clean write check catches input modification
  const cleanWriteFailed = !result.cleanWrite.pass;
  return {
    name: "Dirty Input (known-bad)",
    pass: cleanWriteFailed,
    detail: cleanWriteFailed
      ? `Correctly caught: ${result.cleanWrite.violations.join("; ")}`
      : "UNEXPECTED: checker did not catch input buffer modification",
  };
}

async function test7_break_invariant(
  device: GPUDevice,
  params: ResolvedParams,
  entitySchema: ResolvedSchema,
  globalsSchema: ResolvedSchema,
  schemas: Map<string, ResolvedSchema>
): Promise<TestResult> {
  const contract = resolveContract(IDENTITY_CONTRACT, schemas);

  const entityBuf = makeEntityBuffer(entitySchema, [
    {
      position_x: toFix16(100), position_y: toFix16(100),
      velocity_x: 0, velocity_y: 0,
      size: toFix16(8), health: 50, max_health: 100,
      entity_type: 1, damage: 5, alive: 1,
    },
  ]);

  const result = await checkContractGPU(
    device, contract, testBreakInvariantSrc, [entityBuf], params
  );

  // Should FAIL — schema invariant health <= max_health
  const invariantFailed = result.invariants.some((inv) => !inv.pass);
  return {
    name: "Break Invariant (known-bad)",
    pass: invariantFailed,
    detail: invariantFailed
      ? `Correctly caught: ${result.invariants.filter((i) => !i.pass).map((i) => i.errors.slice(0, 2).join("; ")).join(" | ")}`
      : "UNEXPECTED: checker did not catch invariant violation",
  };
}

// ── Main ──

async function main() {
  const results: TestResult[] = [];
  const log = (r: TestResult) => {
    results.push(r);
    console.log(`[${r.pass ? "PASS" : "FAIL"}] ${r.name}: ${r.detail}`);
  };

  // Init GPU
  let device: GPUDevice;
  try {
    const gpu = await initGPU();
    device = gpu.device;
    log({ name: "GPU Init", pass: true, detail: "device ready" });
  } catch (e: any) {
    log({ name: "GPU Init", pass: false, detail: e.message });
    console.error("Cannot continue without GPU");
    return;
  }

  // Resolve schemas
  const params = resolveParams(DESIGN_PARAMS);
  const entitySchema = resolveSchema(ENTITY_SCHEMA_JSON, params);
  const globalsSchema = resolveSchema(GLOBALS_SCHEMA_JSON, params);
  const schemas = new Map<string, ResolvedSchema>([
    ["EntityBuffer", entitySchema],
    ["GlobalsBuffer", globalsSchema],
  ]);

  log({ name: "Schema Resolution", pass: true, detail: `EntityBuffer: ${entitySchema.structSize}B x ${entitySchema.capacity}, GlobalsBuffer: ${globalsSchema.structSize}B x ${globalsSchema.capacity}` });

  // Run tests
  log(await test1_identity(device, params, entitySchema, globalsSchema, schemas));
  log(await test2_movement(device, params, entitySchema, globalsSchema, schemas));
  log(await test3_movement_adversarial(device, params, entitySchema, globalsSchema, schemas));
  log(await test4_no_clamp(device, params, entitySchema, globalsSchema, schemas));
  log(await test5_modify_dead(device, params, entitySchema, globalsSchema, schemas));
  log(await test6_dirty_input(device, params, entitySchema, globalsSchema, schemas));
  log(await test7_break_invariant(device, params, entitySchema, globalsSchema, schemas));

  const passed = results.filter((r) => r.pass).length;
  const total = results.length;
  console.log(`\n── Contract Checker: ${passed}/${total} passed ──`);
}

main();
