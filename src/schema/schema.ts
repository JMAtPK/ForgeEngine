import type {
  FieldType,
  FieldDef,
  BufferSchemaJSON,
  BufferCategory,
  ResolvedFieldDef,
  ResolvedSchema,
  ResolvedParams,
} from "./types";
import { fix16Mul } from "./params";

const FIELD_SIZES: Record<FieldType, number> = {
  u32: 4,
  i32: 4,
  fix16: 4,
  fix16x2: 8,
  vec2f: 8,
  vec3f: 12,
  vec4f: 16,
};

const FIELD_ALIGNMENT: Record<FieldType, number> = {
  u32: 4,
  i32: 4,
  fix16: 4,
  fix16x2: 4, // two i32s, each 4-aligned
  vec2f: 8,
  vec3f: 16,
  vec4f: 16,
};

const FLOAT_TYPES: Set<FieldType> = new Set(["vec2f", "vec3f", "vec4f"]);

export function resolveRef(ref: number | string, params: ResolvedParams): number {
  if (typeof ref === "number") return ref;
  if (!ref.startsWith("$")) return Number(ref);

  const negated = ref.startsWith("$-");
  const paramName = negated ? ref.slice(2) : ref.slice(1);

  if (!(paramName in params.raw)) {
    throw new Error(`Unknown param reference: ${ref}`);
  }
  return negated ? -params.raw[paramName] : params.raw[paramName];
}

function resolveRange(
  range: FieldDef["range"],
  params: ResolvedParams
): [number, number] | [[number, number], [number, number]] | undefined {
  if (!range) return undefined;

  // fix16x2 range: [[minX, maxX], [minY, maxY]]
  if (Array.isArray(range[0]) && Array.isArray(range[0])) {
    const r = range as [[number | string, number | string], [number | string, number | string]];
    if (Array.isArray(r[0]) && Array.isArray(r[1])) {
      return [
        [resolveRef(r[0][0], params), resolveRef(r[0][1], params)],
        [resolveRef(r[1][0], params), resolveRef(r[1][1], params)],
      ];
    }
  }

  // Simple range: [min, max]
  const r = range as [number | string, number | string];
  return [resolveRef(r[0], params), resolveRef(r[1], params)];
}

function isNestedRange(r: [number, number] | [[number, number], [number, number]]): r is [[number, number], [number, number]] {
  return Array.isArray(r[0]);
}

function computeStructLayout(
  fields: { name: string; type: FieldType; range?: any; enumValues?: number[] }[]
): { resolvedFields: ResolvedFieldDef[]; structSize: number } {
  const resolvedFields: ResolvedFieldDef[] = [];
  let offset = 0;

  for (const field of fields) {
    const alignment = FIELD_ALIGNMENT[field.type];
    const size = FIELD_SIZES[field.type];

    // Align offset
    if (offset % alignment !== 0) {
      offset += alignment - (offset % alignment);
    }

    resolvedFields.push({
      name: field.name,
      type: field.type,
      byteOffset: offset,
      byteSize: size,
      range: field.range,
      enumValues: field.enumValues,
    });

    offset += size;
  }

  // Pad struct to 4-byte boundary
  if (offset % 4 !== 0) {
    offset += 4 - (offset % 4);
  }

  return { resolvedFields, structSize: offset };
}

function compileInvariant(
  source: string
): (element: Record<string, number>, $: Record<string, number>) => boolean {
  // Wrap field names: bare identifiers → element["name"]
  // The invariant uses bare names for element fields and $.param for params
  const fn = new Function(
    "element",
    "$",
    "fix16_mul",
    `with(element) { return (${source}); }`
  );
  return (element, $) => fn(element, $, fix16Mul);
}

export function resolveSchema(json: BufferSchemaJSON, params: ResolvedParams): ResolvedSchema {
  // Check float-in-state constraint
  if (json.buffer_category === "state") {
    for (const [name, field] of Object.entries(json.struct)) {
      if (FLOAT_TYPES.has(field.type)) {
        throw new Error(`Float type "${field.type}" not allowed in state buffer field "${name}"`);
      }
    }
  }

  // Build field list, expanding fix16x2
  const rawFields: { name: string; type: FieldType; range?: any; enumValues?: number[] }[] = [];

  for (const [name, field] of Object.entries(json.struct)) {
    const resolved = resolveRange(field.range, params);

    if (field.type === "fix16x2") {
      // Expand to two fix16 fields
      const nestedRange = resolved as [[number, number], [number, number]] | undefined;
      rawFields.push({
        name: `${name}_x`,
        type: "i32", // fix16 stored as i32
        range: nestedRange ? nestedRange[0] : undefined,
        enumValues: undefined,
      });
      rawFields.push({
        name: `${name}_y`,
        type: "i32",
        range: nestedRange ? nestedRange[1] : undefined,
        enumValues: undefined,
      });
    } else {
      rawFields.push({
        name,
        type: field.type,
        range: resolved as [number, number] | undefined,
        enumValues: field.enum,
      });
    }
  }

  const { resolvedFields, structSize } = computeStructLayout(rawFields);
  const capacity = typeof json.capacity === "string" ? resolveRef(json.capacity, params) : json.capacity;

  const invariantSources = json.invariants ?? [];
  const invariantFns = invariantSources.map(compileInvariant);

  return {
    name: json.name,
    fields: resolvedFields,
    structSize,
    capacity,
    bufferCategory: json.buffer_category,
    invariantSources,
    invariantFns,
    totalSize: structSize * capacity,
  };
}

export function readElement(
  buffer: ArrayBuffer,
  schema: ResolvedSchema,
  index: number
): Record<string, number> {
  const view = new DataView(buffer);
  const base = index * schema.structSize;
  const result: Record<string, number> = {};

  for (const field of schema.fields) {
    const offset = base + field.byteOffset;
    switch (field.type) {
      case "u32":
        result[field.name] = view.getUint32(offset, true);
        break;
      case "i32":
      case "fix16":
        result[field.name] = view.getInt32(offset, true);
        break;
      case "vec2f":
        result[field.name + "_x"] = view.getFloat32(offset, true);
        result[field.name + "_y"] = view.getFloat32(offset + 4, true);
        break;
      case "vec3f":
        result[field.name + "_x"] = view.getFloat32(offset, true);
        result[field.name + "_y"] = view.getFloat32(offset + 4, true);
        result[field.name + "_z"] = view.getFloat32(offset + 8, true);
        break;
      case "vec4f":
        result[field.name + "_x"] = view.getFloat32(offset, true);
        result[field.name + "_y"] = view.getFloat32(offset + 4, true);
        result[field.name + "_z"] = view.getFloat32(offset + 8, true);
        result[field.name + "_w"] = view.getFloat32(offset + 12, true);
        break;
    }
  }

  return result;
}

export function writeElement(
  buffer: ArrayBuffer,
  schema: ResolvedSchema,
  index: number,
  values: Record<string, number>
): void {
  const view = new DataView(buffer);
  const base = index * schema.structSize;

  for (const field of schema.fields) {
    const offset = base + field.byteOffset;
    switch (field.type) {
      case "u32":
        view.setUint32(offset, values[field.name] ?? 0, true);
        break;
      case "i32":
      case "fix16":
        view.setInt32(offset, values[field.name] ?? 0, true);
        break;
      case "vec2f":
        view.setFloat32(offset, values[field.name + "_x"] ?? 0, true);
        view.setFloat32(offset + 4, values[field.name + "_y"] ?? 0, true);
        break;
      case "vec3f":
        view.setFloat32(offset, values[field.name + "_x"] ?? 0, true);
        view.setFloat32(offset + 4, values[field.name + "_y"] ?? 0, true);
        view.setFloat32(offset + 8, values[field.name + "_z"] ?? 0, true);
        break;
      case "vec4f":
        view.setFloat32(offset, values[field.name + "_x"] ?? 0, true);
        view.setFloat32(offset + 4, values[field.name + "_y"] ?? 0, true);
        view.setFloat32(offset + 8, values[field.name + "_z"] ?? 0, true);
        view.setFloat32(offset + 12, values[field.name + "_w"] ?? 0, true);
        break;
    }
  }
}

export function validateBuffer(
  buffer: ArrayBuffer,
  schema: ResolvedSchema,
  params: ResolvedParams
): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  const MAX_ERRORS = 20;

  // Level 1: byte length
  if (buffer.byteLength !== schema.totalSize) {
    return {
      valid: false,
      errors: [`Buffer length mismatch: expected ${schema.totalSize}, got ${buffer.byteLength}`],
    };
  }

  for (let i = 0; i < schema.capacity && errors.length < MAX_ERRORS; i++) {
    const element = readElement(buffer, schema, i);

    // Level 2: per-field range and enum
    for (const field of schema.fields) {
      if (errors.length >= MAX_ERRORS) break;

      const value = element[field.name];
      if (value === undefined) continue;

      if (field.enumValues) {
        if (!field.enumValues.includes(value)) {
          errors.push(`Element ${i}, field "${field.name}": value ${value} not in enum [${field.enumValues}]`);
        }
      }

      if (field.range && !isNestedRange(field.range)) {
        const [min, max] = field.range;
        if (value < min || value > max) {
          errors.push(`Element ${i}, field "${field.name}": value ${value} out of range [${min}, ${max}]`);
        }
      }
    }

    // Level 3: per-element invariants
    if (errors.length < MAX_ERRORS) {
      for (let j = 0; j < schema.invariantFns.length; j++) {
        if (errors.length >= MAX_ERRORS) break;
        try {
          if (!schema.invariantFns[j](element, params.raw)) {
            errors.push(`Element ${i}: invariant violated: ${schema.invariantSources[j]}`);
          }
        } catch (e) {
          errors.push(`Element ${i}: invariant error: ${schema.invariantSources[j]} — ${(e as Error).message}`);
        }
      }
    }
  }

  return { valid: errors.length === 0, errors };
}

// GPUBufferUsage flag values (plain numbers, no GPU dependency)
const GPU_BUFFER_USAGE = {
  MAP_READ: 0x01,
  MAP_WRITE: 0x02,
  COPY_SRC: 0x04,
  COPY_DST: 0x08,
  INDEX: 0x10,
  VERTEX: 0x20,
  UNIFORM: 0x40,
  STORAGE: 0x80,
};

export function bufferUsageForCategory(category: BufferCategory): number {
  switch (category) {
    case "state":
      return GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_SRC | GPU_BUFFER_USAGE.COPY_DST;
    case "transient":
      return GPU_BUFFER_USAGE.STORAGE;
    case "input":
      return GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST;
    case "output":
      return GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_SRC;
  }
}

export function schemaToBufferDescriptor(schema: ResolvedSchema): { size: number; usage: number } {
  return {
    size: schema.totalSize,
    usage: bufferUsageForCategory(schema.bufferCategory),
  };
}
