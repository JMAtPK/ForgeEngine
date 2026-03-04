import type { DesignParamsJSON, ResolvedParams } from "./types";

export function toFix16(value: number): number {
  return Math.round(value * 65536);
}

export function fromFix16(value: number): number {
  return value / 65536;
}

/**
 * Fixed-point 16.16 multiply matching the WGSL split-half approach.
 * Splits each operand into hi (integer) and lo (fractional) 16-bit halves.
 */
export function fix16Mul(a: number, b: number): number {
  const a_hi = a >> 16;
  const a_lo = (a >>> 0) & 0xffff;
  const b_hi = b >> 16;
  const b_lo = (b >>> 0) & 0xffff;

  const hi_hi = (a_hi * b_hi) << 16;
  const hi_lo = a_hi * b_lo;
  const lo_hi = a_lo * b_hi;
  const lo_lo = (a_lo * b_lo) >>> 16;

  return (hi_hi + hi_lo + lo_hi + lo_lo) | 0;
}

export function resolveParams(json: DesignParamsJSON): ResolvedParams {
  const raw: Record<string, number> = {};
  const display: Record<string, number> = {};
  const types: Record<string, "fix16" | "u32"> = {};

  for (const [name, def] of Object.entries(json.params)) {
    if (def.type === "fix16") {
      raw[name] = toFix16(def.value);
      display[name] = def.value;
      types[name] = "fix16";
    } else if (def.type === "u32") {
      raw[name] = def.value;
      display[name] = def.value;
      types[name] = "u32";
    } else {
      throw new Error(`Unknown param type: ${def.type} for param "${name}"`);
    }
  }

  return { name: json.name, raw, display, types };
}

export function checkDesignInvariants(
  params: ResolvedParams,
  invariants: string[]
): { valid: boolean; violations: string[] } {
  const violations: string[] = [];

  for (const expr of invariants) {
    try {
      const fn = new Function("$", "fix16_mul", `return (${expr});`);
      const result = fn(params.raw, fix16Mul);
      if (!result) {
        violations.push(`Invariant violated: ${expr}`);
      }
    } catch (e) {
      violations.push(`Invariant error: ${expr} — ${(e as Error).message}`);
    }
  }

  return { valid: violations.length === 0, violations };
}

export function loadDesignParams(json: DesignParamsJSON): ResolvedParams {
  const params = resolveParams(json);
  const result = checkDesignInvariants(params, json.design_invariants);
  if (!result.valid) {
    throw new Error(`Design invariant violations:\n${result.violations.join("\n")}`);
  }
  return params;
}
