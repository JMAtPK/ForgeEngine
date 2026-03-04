import { describe, it, expect } from "vitest";
import { toFix16, fromFix16, fix16Mul, resolveParams, checkDesignInvariants, loadDesignParams } from "../schema/params";
import type { DesignParamsJSON } from "../schema/types";

const VALID_PARAMS: DesignParamsJSON = {
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
    max_entities: { type: "u32", value: 64 },
    player_size: { type: "fix16", value: 16 },
    max_delta_time: { type: "fix16", value: 0.05 },
  },
  design_invariants: [
    "$.bullet_damage < $.player_health",
    "$.boss_health >= $.enemy_health",
    "$.max_entities >= 64",
    "fix16_mul($.max_speed, $.max_delta_time) < $.world_width",
  ],
};

const BROKEN_PARAMS: DesignParamsJSON = {
  name: "broken_design",
  params: {
    world_width: { type: "fix16", value: 1280 },
    player_health: { type: "u32", value: 5 },
    bullet_damage: { type: "u32", value: 10 },
  },
  design_invariants: [
    "$.bullet_damage < $.player_health", // 10 < 5 → false
  ],
};

describe("fix16 conversions", () => {
  it("toFix16 shifts correctly", () => {
    expect(toFix16(1)).toBe(65536);
    expect(toFix16(0)).toBe(0);
    expect(toFix16(-1)).toBe(-65536);
    expect(toFix16(0.5)).toBe(32768);
    expect(toFix16(1280)).toBe(1280 * 65536);
  });

  it("fromFix16 reverses toFix16", () => {
    expect(fromFix16(65536)).toBe(1);
    expect(fromFix16(0)).toBe(0);
    expect(fromFix16(toFix16(1280))).toBe(1280);
    expect(fromFix16(toFix16(0.5))).toBe(0.5);
  });

  it("roundtrips correctly", () => {
    for (const v of [0, 1, -1, 100, 0.5, 0.25, -32768, 32767]) {
      expect(fromFix16(toFix16(v))).toBeCloseTo(v, 4);
    }
  });
});

describe("fix16Mul", () => {
  it("multiplies 1 × 1 = 1", () => {
    expect(fix16Mul(toFix16(1), toFix16(1))).toBe(toFix16(1));
  });

  it("multiplies 2 × 3 = 6", () => {
    expect(fix16Mul(toFix16(2), toFix16(3))).toBe(toFix16(6));
  });

  it("multiplies 0.5 × 0.5 = 0.25", () => {
    expect(fix16Mul(toFix16(0.5), toFix16(0.5))).toBe(toFix16(0.25));
  });

  it("handles negatives", () => {
    expect(fix16Mul(toFix16(-2), toFix16(3))).toBe(toFix16(-6));
    expect(fix16Mul(toFix16(-2), toFix16(-3))).toBe(toFix16(6));
  });

  it("multiplies max_speed × max_delta_time correctly", () => {
    const result = fix16Mul(toFix16(500), toFix16(0.05));
    expect(fromFix16(result)).toBeCloseTo(25, 2);
  });
});

describe("resolveParams", () => {
  it("converts fix16 values to internal representation", () => {
    const resolved = resolveParams(VALID_PARAMS);
    expect(resolved.raw["world_width"]).toBe(toFix16(1280));
    expect(resolved.raw["max_delta_time"]).toBe(toFix16(0.05));
    expect(resolved.display["world_width"]).toBe(1280);
  });

  it("keeps u32 values as-is", () => {
    const resolved = resolveParams(VALID_PARAMS);
    expect(resolved.raw["player_health"]).toBe(100);
    expect(resolved.raw["max_entities"]).toBe(64);
  });

  it("rejects unknown types", () => {
    const bad: DesignParamsJSON = {
      name: "bad",
      params: { x: { type: "f32" as any, value: 1 } },
      design_invariants: [],
    };
    expect(() => resolveParams(bad)).toThrow();
  });
});

describe("checkDesignInvariants", () => {
  it("passes valid params", () => {
    const resolved = resolveParams(VALID_PARAMS);
    const result = checkDesignInvariants(resolved, VALID_PARAMS.design_invariants);
    expect(result.valid).toBe(true);
    expect(result.violations).toHaveLength(0);
  });

  it("catches violations", () => {
    const resolved = resolveParams(BROKEN_PARAMS);
    const result = checkDesignInvariants(resolved, BROKEN_PARAMS.design_invariants);
    expect(result.valid).toBe(false);
    expect(result.violations.length).toBeGreaterThan(0);
  });

  it("reports all violations not just first", () => {
    const params: DesignParamsJSON = {
      name: "multi_broken",
      params: {
        a: { type: "u32", value: 10 },
        b: { type: "u32", value: 5 },
      },
      design_invariants: ["$.a < $.b", "$.b > 100"],
    };
    const resolved = resolveParams(params);
    const result = checkDesignInvariants(resolved, params.design_invariants);
    expect(result.violations).toHaveLength(2);
  });

  it("handles malformed expressions", () => {
    const resolved = resolveParams(VALID_PARAMS);
    const result = checkDesignInvariants(resolved, ["this is not valid js %%"]);
    expect(result.valid).toBe(false);
    expect(result.violations[0]).toContain("error");
  });

  it("handles missing param references", () => {
    const resolved = resolveParams(VALID_PARAMS);
    const result = checkDesignInvariants(resolved, ["$.nonexistent > 0"]);
    expect(result.valid).toBe(false);
  });
});

describe("loadDesignParams", () => {
  it("succeeds for valid params", () => {
    const resolved = loadDesignParams(VALID_PARAMS);
    expect(resolved.name).toBe("test_design");
  });

  it("throws for broken invariants", () => {
    expect(() => loadDesignParams(BROKEN_PARAMS)).toThrow();
  });
});
