// Design Parameters

export type ParamType = "fix16" | "u32";

export interface ParamDef {
  type: ParamType;
  value: number;
}

export interface DesignParamsJSON {
  name: string;
  params: Record<string, ParamDef>;
  design_invariants: string[];
}

export interface ResolvedParams {
  name: string;
  /** Raw internal values (fix16 shifted, u32 as-is) */
  raw: Record<string, number>;
  /** Human-readable values (as declared in JSON) */
  display: Record<string, number>;
  types: Record<string, ParamType>;
}

// Buffer Schemas

export type FieldType = "u32" | "i32" | "fix16" | "fix16x2" | "vec2f" | "vec3f" | "vec4f";

export type BufferCategory = "state" | "transient" | "input" | "output";

export interface FieldDef {
  type: FieldType;
  range?: [number | string, number | string] | [[number | string, number | string], [number | string, number | string]];
  enum?: number[];
}

export interface BufferSchemaJSON {
  name: string;
  struct: Record<string, FieldDef>;
  capacity: number | string;
  buffer_category: BufferCategory;
  invariants?: string[];
}

export interface ResolvedFieldDef {
  name: string;
  type: FieldType;
  byteOffset: number;
  byteSize: number;
  range?: [number, number] | [[number, number], [number, number]];
  enumValues?: number[];
}

export interface ResolvedSchema {
  name: string;
  fields: ResolvedFieldDef[];
  structSize: number;
  capacity: number;
  bufferCategory: BufferCategory;
  invariantSources: string[];
  invariantFns: ((element: Record<string, number>, $: Record<string, number>) => boolean)[];
  totalSize: number;
}
