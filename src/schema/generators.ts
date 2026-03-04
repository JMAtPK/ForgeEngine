import type { ResolvedSchema, ResolvedFieldDef, ResolvedParams } from "./types";
import { readElement, writeElement, validateBuffer } from "./schema";

function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function randomFieldValue(field: ResolvedFieldDef): number {
  if (field.enumValues) {
    return field.enumValues[randomInt(0, field.enumValues.length - 1)];
  }
  if (field.range && !Array.isArray(field.range[0])) {
    const [min, max] = field.range as [number, number];
    return randomInt(min, max);
  }
  return 0;
}

function minFieldValue(field: ResolvedFieldDef): number {
  if (field.enumValues) return field.enumValues[0];
  if (field.range && !Array.isArray(field.range[0])) return (field.range as [number, number])[0];
  return 0;
}

function maxFieldValue(field: ResolvedFieldDef): number {
  if (field.enumValues) return field.enumValues[field.enumValues.length - 1];
  if (field.range && !Array.isArray(field.range[0])) return (field.range as [number, number])[1];
  return 0;
}

function buildElement(
  schema: ResolvedSchema,
  valueFn: (field: ResolvedFieldDef) => number
): Record<string, number> {
  const result: Record<string, number> = {};
  for (const field of schema.fields) {
    result[field.name] = valueFn(field);
  }
  return result;
}

function checkInvariants(
  element: Record<string, number>,
  schema: ResolvedSchema,
  params: ResolvedParams
): boolean {
  for (const fn of schema.invariantFns) {
    try {
      if (!fn(element, params.raw)) return false;
    } catch {
      return false;
    }
  }
  return true;
}

export function generateRandomElement(
  schema: ResolvedSchema,
  params: ResolvedParams
): Record<string, number> {
  for (let attempt = 0; attempt < 100; attempt++) {
    const el = buildElement(schema, randomFieldValue);
    if (checkInvariants(el, schema, params)) return el;
  }
  throw new Error(`Failed to generate valid random element for "${schema.name}" after 100 attempts`);
}

export function generateRandomBuffer(
  schema: ResolvedSchema,
  params: ResolvedParams
): ArrayBuffer {
  const buffer = new ArrayBuffer(schema.totalSize);
  for (let i = 0; i < schema.capacity; i++) {
    const el = generateRandomElement(schema, params);
    writeElement(buffer, schema, i, el);
  }
  return buffer;
}

export type AdversarialStrategy = "min" | "max" | "zero" | "boundary";

export function generateAdversarialElement(
  schema: ResolvedSchema,
  params: ResolvedParams,
  strategy: AdversarialStrategy
): Record<string, number> {
  let el: Record<string, number>;

  switch (strategy) {
    case "zero":
      el = buildElement(schema, () => 0);
      break;
    case "min":
      el = buildElement(schema, minFieldValue);
      break;
    case "max":
      el = buildElement(schema, maxFieldValue);
      break;
    case "boundary":
      el = buildElement(schema, (field) => (Math.random() < 0.5 ? minFieldValue(field) : maxFieldValue(field)));
      break;
  }

  // If invariants fail, fall back to random valid element
  if (!checkInvariants(el, schema, params)) {
    return generateRandomElement(schema, params);
  }
  return el;
}

export function generateAdversarialBuffer(
  schema: ResolvedSchema,
  params: ResolvedParams
): ArrayBuffer[] {
  const strategies: AdversarialStrategy[] = ["zero", "min", "max", "boundary"];
  const buffers: ArrayBuffer[] = [];

  for (const strategy of strategies) {
    const buffer = new ArrayBuffer(schema.totalSize);
    for (let i = 0; i < schema.capacity; i++) {
      const el = generateAdversarialElement(schema, params, strategy);
      writeElement(buffer, schema, i, el);
    }
    buffers.push(buffer);
  }

  // Single-active: one random valid element, rest zero
  const singleActive = new ArrayBuffer(schema.totalSize);
  const activeEl = generateRandomElement(schema, params);
  writeElement(singleActive, schema, 0, activeEl);
  buffers.push(singleActive);

  return buffers;
}

export type InvalidViolation = "wrong_length" | "out_of_range" | "broken_invariant" | "bad_enum";

export function generateInvalidBuffer(
  schema: ResolvedSchema,
  params: ResolvedParams,
  violation: InvalidViolation
): ArrayBuffer {
  switch (violation) {
    case "wrong_length":
      return new ArrayBuffer(schema.totalSize + 7);

    case "out_of_range": {
      const buffer = new ArrayBuffer(schema.totalSize);
      // Write a valid element first, then corrupt a ranged field
      const el = generateRandomElement(schema, params);
      writeElement(buffer, schema, 0, el);
      const view = new DataView(buffer);
      const rangedField = schema.fields.find((f) => f.range && !Array.isArray(f.range[0]));
      if (rangedField) {
        const [, max] = rangedField.range as [number, number];
        view.setUint32(rangedField.byteOffset, max + 999, true);
      }
      return buffer;
    }

    case "broken_invariant": {
      const buffer = new ArrayBuffer(schema.totalSize);
      // For "health <= max_health", set health > max_health
      const el = generateRandomElement(schema, params);
      writeElement(buffer, schema, 0, el);
      const view = new DataView(buffer);
      // Find health and max_health fields and swap to break invariant
      const healthField = schema.fields.find((f) => f.name === "health");
      const maxHealthField = schema.fields.find((f) => f.name === "max_health");
      if (healthField && maxHealthField) {
        const maxRange = maxHealthField.range as [number, number] | undefined;
        const maxVal = maxRange ? maxRange[1] : 200;
        view.setUint32(healthField.byteOffset, maxVal, true);
        view.setUint32(maxHealthField.byteOffset, 1, true);
        // Ensure enum fields stay valid
        for (const f of schema.fields) {
          if (f.enumValues) {
            view.setUint32(f.byteOffset, f.enumValues[0], true);
          }
        }
      }
      return buffer;
    }

    case "bad_enum": {
      const buffer = new ArrayBuffer(schema.totalSize);
      const el = generateRandomElement(schema, params);
      writeElement(buffer, schema, 0, el);
      const view = new DataView(buffer);
      const enumField = schema.fields.find((f) => f.enumValues);
      if (enumField) {
        view.setUint32(enumField.byteOffset, 9999, true);
      }
      return buffer;
    }
  }
}
