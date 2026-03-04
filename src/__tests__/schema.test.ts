import { describe, it, expect } from "vitest";
import {
  resolveRef,
  resolveSchema,
  validateBuffer,
  readElement,
  writeElement,
  bufferUsageForCategory,
} from "../schema/schema";
import { toFix16, resolveParams } from "../schema/params";
import type { DesignParamsJSON, BufferSchemaJSON, ResolvedParams } from "../schema/types";

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
    max_entities: { type: "u32", value: 64 },
    player_size: { type: "fix16", value: 16 },
  },
  design_invariants: [],
};

let params: ResolvedParams;

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

// Pre-resolve params
params = resolveParams(DESIGN);

describe("resolveRef", () => {
  it("passes through numbers", () => {
    expect(resolveRef(42, params)).toBe(42);
  });

  it("resolves $param", () => {
    expect(resolveRef("$world_width", params)).toBe(toFix16(1280));
  });

  it("resolves $-param (negation)", () => {
    expect(resolveRef("$-max_speed", params)).toBe(-toFix16(500));
  });

  it("throws on unknown $param", () => {
    expect(() => resolveRef("$nonexistent", params)).toThrow();
  });
});

describe("resolveSchema", () => {
  it("computes correct struct size with alignment", () => {
    const schema = resolveSchema(ENTITY_SCHEMA, params);
    // fix16x2=8, u32=4, u32=4, u32=4, u32=4 = 24 bytes
    expect(schema.structSize).toBe(24);
  });

  it("resolves capacity from param", () => {
    const schema = resolveSchema(ENTITY_SCHEMA, params);
    expect(schema.capacity).toBe(64);
  });

  it("computes total buffer size", () => {
    const schema = resolveSchema(ENTITY_SCHEMA, params);
    expect(schema.totalSize).toBe(schema.structSize * schema.capacity);
  });

  it("rejects float fields in state buffers", () => {
    const floatSchema: BufferSchemaJSON = {
      name: "BadState",
      struct: { pos: { type: "vec2f" } },
      capacity: 10,
      buffer_category: "state",
    };
    expect(() => resolveSchema(floatSchema, params)).toThrow(/float.*state/i);
  });

  it("allows float fields in output buffers", () => {
    const outputSchema: BufferSchemaJSON = {
      name: "RenderBuffer",
      struct: { color: { type: "vec4f" } },
      capacity: 10,
      buffer_category: "output",
    };
    expect(() => resolveSchema(outputSchema, params)).not.toThrow();
  });

  it("compiles invariant functions", () => {
    const schema = resolveSchema(ENTITY_SCHEMA, params);
    expect(schema.invariantFns).toHaveLength(1);
    expect(typeof schema.invariantFns[0]).toBe("function");
  });
});

describe("validateBuffer", () => {
  it("accepts a valid buffer", () => {
    const schema = resolveSchema(ENTITY_SCHEMA, params);
    const buffer = new ArrayBuffer(schema.totalSize);
    const view = new DataView(buffer);
    // Write valid element at index 0: position=(0,0), health=50, max_health=100, entity_type=1, alive=1
    view.setInt32(0, 0, true); // position_x
    view.setInt32(4, 0, true); // position_y
    view.setUint32(8, 50, true); // health
    view.setUint32(12, 100, true); // max_health
    view.setUint32(16, 1, true); // entity_type
    view.setUint32(20, 1, true); // alive
    const result = validateBuffer(buffer, schema, params);
    expect(result.valid).toBe(true);
  });

  it("rejects wrong byte length", () => {
    const schema = resolveSchema(ENTITY_SCHEMA, params);
    const buffer = new ArrayBuffer(10);
    const result = validateBuffer(buffer, schema, params);
    expect(result.valid).toBe(false);
    expect(result.errors[0]).toContain("length");
  });

  it("rejects out-of-range values", () => {
    const schema = resolveSchema(ENTITY_SCHEMA, params);
    const buffer = new ArrayBuffer(schema.totalSize);
    const view = new DataView(buffer);
    view.setUint32(8, 999, true); // health > boss_health(200)
    view.setUint32(12, 999, true); // max_health > boss_health(200)
    const result = validateBuffer(buffer, schema, params);
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("range"))).toBe(true);
  });

  it("rejects bad enum values", () => {
    const schema = resolveSchema(ENTITY_SCHEMA, params);
    const buffer = new ArrayBuffer(schema.totalSize);
    const view = new DataView(buffer);
    view.setUint32(16, 99, true); // entity_type not in [0,1,2,3]
    const result = validateBuffer(buffer, schema, params);
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("enum"))).toBe(true);
  });

  it("rejects broken invariant", () => {
    const schema = resolveSchema(ENTITY_SCHEMA, params);
    const buffer = new ArrayBuffer(schema.totalSize);
    const view = new DataView(buffer);
    view.setUint32(8, 100, true); // health = 100
    view.setUint32(12, 50, true); // max_health = 50 → health > max_health
    view.setUint32(16, 1, true); // entity_type = valid
    view.setUint32(20, 1, true); // alive = valid
    const result = validateBuffer(buffer, schema, params);
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("invariant"))).toBe(true);
  });
});

describe("readElement / writeElement", () => {
  it("roundtrips correctly", () => {
    const schema = resolveSchema(ENTITY_SCHEMA, params);
    const buffer = new ArrayBuffer(schema.totalSize);
    const values = {
      position_x: toFix16(100),
      position_y: toFix16(200),
      health: 50,
      max_health: 100,
      entity_type: 2,
      alive: 1,
    };
    writeElement(buffer, schema, 0, values);
    const read = readElement(buffer, schema, 0);
    expect(read).toEqual(values);
  });

  it("reads correct element by index", () => {
    const schema = resolveSchema(ENTITY_SCHEMA, params);
    const buffer = new ArrayBuffer(schema.totalSize);
    const v0 = { position_x: 0, position_y: 0, health: 10, max_health: 20, entity_type: 0, alive: 0 };
    const v1 = { position_x: toFix16(50), position_y: toFix16(60), health: 30, max_health: 40, entity_type: 1, alive: 1 };
    writeElement(buffer, schema, 0, v0);
    writeElement(buffer, schema, 1, v1);
    expect(readElement(buffer, schema, 1)).toEqual(v1);
  });
});

describe("bufferUsageForCategory", () => {
  it("returns correct flags for state", () => {
    const usage = bufferUsageForCategory("state");
    // STORAGE | COPY_SRC | COPY_DST
    expect(usage & 0x80).toBeTruthy(); // STORAGE
    expect(usage & 0x04).toBeTruthy(); // COPY_SRC
    expect(usage & 0x08).toBeTruthy(); // COPY_DST
  });

  it("returns correct flags for input", () => {
    const usage = bufferUsageForCategory("input");
    // STORAGE | COPY_DST
    expect(usage & 0x80).toBeTruthy(); // STORAGE
    expect(usage & 0x08).toBeTruthy(); // COPY_DST
  });

  it("returns correct flags for output", () => {
    const usage = bufferUsageForCategory("output");
    // STORAGE | COPY_SRC
    expect(usage & 0x80).toBeTruthy(); // STORAGE
    expect(usage & 0x04).toBeTruthy(); // COPY_SRC
  });
});
