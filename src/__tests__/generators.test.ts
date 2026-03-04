import { describe, it, expect } from "vitest";
import {
  generateRandomElement,
  generateRandomBuffer,
  generateAdversarialBuffer,
  generateInvalidBuffer,
} from "../schema/generators";
import { resolveSchema, validateBuffer } from "../schema/schema";
import { resolveParams, toFix16 } from "../schema/params";
import type { DesignParamsJSON, BufferSchemaJSON } from "../schema/types";

const DESIGN: DesignParamsJSON = {
  name: "test_design",
  params: {
    world_width: { type: "fix16", value: 1280 },
    world_height: { type: "fix16", value: 720 },
    max_speed: { type: "fix16", value: 500 },
    player_health: { type: "u32", value: 100 },
    boss_health: { type: "u32", value: 200 },
    bullet_health: { type: "u32", value: 1 },
    bullet_damage: { type: "u32", value: 10 },
    max_entities: { type: "u32", value: 8 },
    player_size: { type: "fix16", value: 16 },
  },
  design_invariants: [],
};

const ENTITY_SCHEMA: BufferSchemaJSON = {
  name: "EntityBuffer",
  struct: {
    position: {
      type: "fix16x2",
      range: [
        [0, "$world_width"],
        [0, "$world_height"],
      ],
    },
    health: { type: "u32", range: [0, "$boss_health"] },
    max_health: { type: "u32", range: [0, "$boss_health"] },
    entity_type: { type: "u32", enum: [0, 1, 2, 3] },
    alive: { type: "u32", enum: [0, 1] },
  },
  capacity: "$max_entities",
  buffer_category: "state",
  invariants: ["health <= max_health"],
};

const params = resolveParams(DESIGN);
const schema = resolveSchema(ENTITY_SCHEMA, params);

describe("generateRandomBuffer", () => {
  it("produces buffer with correct byte length", () => {
    const buffer = generateRandomBuffer(schema, params);
    expect(buffer.byteLength).toBe(schema.totalSize);
  });

  it("produces buffer that passes validation", () => {
    const buffer = generateRandomBuffer(schema, params);
    const result = validateBuffer(buffer, schema, params);
    expect(result.valid).toBe(true);
  });
});

describe("generateRandomElement", () => {
  it("produces element that satisfies invariants", () => {
    for (let i = 0; i < 10; i++) {
      const el = generateRandomElement(schema, params);
      // health <= max_health
      expect(el.health).toBeLessThanOrEqual(el.max_health);
    }
  });
});

describe("generateAdversarialBuffer", () => {
  it("produces multiple buffers that all pass validation", () => {
    const buffers = generateAdversarialBuffer(schema, params);
    expect(buffers.length).toBeGreaterThanOrEqual(3);
    for (const buf of buffers) {
      expect(buf.byteLength).toBe(schema.totalSize);
      const result = validateBuffer(buf, schema, params);
      expect(result.valid).toBe(true);
    }
  });
});

describe("generateInvalidBuffer", () => {
  it("wrong_length fails validation with length error", () => {
    const buffer = generateInvalidBuffer(schema, params, "wrong_length");
    const result = validateBuffer(buffer, schema, params);
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("length"))).toBe(true);
  });

  it("out_of_range fails validation with range error", () => {
    const buffer = generateInvalidBuffer(schema, params, "out_of_range");
    const result = validateBuffer(buffer, schema, params);
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("range"))).toBe(true);
  });

  it("broken_invariant fails validation with invariant error", () => {
    const buffer = generateInvalidBuffer(schema, params, "broken_invariant");
    const result = validateBuffer(buffer, schema, params);
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("invariant"))).toBe(true);
  });

  it("bad_enum fails validation with enum error", () => {
    const buffer = generateInvalidBuffer(schema, params, "bad_enum");
    const result = validateBuffer(buffer, schema, params);
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("enum"))).toBe(true);
  });
});
